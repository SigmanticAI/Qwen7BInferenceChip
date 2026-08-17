#!/usr/bin/env python3
# emit_cache.py — the per-layer TEMPLATE CACHE for MASK_B walk-program
# emission (token_loop's --emit-cache).
#
# ══ WHAT THIS IS ═══════════════════════════════════════════════════════════
# token_loop._emit_program builds each (step, layer)'s ~12.3k-op regops file
# from scratch every decode step (wfl.build_program + capture_ro +
# splice_fuel_arm + silence/census + cap counting) — measured ~1.2 s/token
# of pure EMISSION on the hw host (pooled 12-wide). But two programs for the
# SAME layer at two steps differ ONLY in step-varying payloads; the op
# STRUCTURE (op sequence, kinds, addresses, polls, caps) is invariant:
#
#   * the 64 descriptor words   — w @ WALK_DDATA (0x1064), contiguous after
#                                 the single WALK_DPTR=0 write; between two
#                                 steps of one layer only the re-point set
#                                 (wl.REPOINT_WORDS: 4 bases + RQ[H] +
#                                 JC_OPROJ) may move, and the bases do not
#   * the staged activations    — inject_jobs' encoding of the h8 (rows
#                                 0..13) and attn8 (rows 14..27) C-1 frames:
#                                 one squant composite per element
#                                 (pw @ QS 0x3020) and one K=2 weight beat
#                                 per element (w @ XW0 0x3040; XW1 is
#                                 always 0)
#   * the xrow residual preload — w @ LAYER_DATA (0x1078), one per f16 bit
#                                 pattern of the layer input row
#   * the name note             — the single _translate() note op carrying
#                                 the program name (the step tag)
#
# DETERMINISM, MEASURED NOT ASSUMED: wfl.build_program embeds no timestamps
# and draws no randomness (verified: byte-identical rebuilds), but the cache
# does NOT rely on that argument — the template is the RECORDED bytes of the
# first real emission for the layer (post capture_ro + splice_fuel_arm), and
# record_template() REFUSES unless (a) every line JSON-round-trips
# byte-identically, (b) the payload transform below reproduces the recorded
# payload ops bit-for-bit on the template's own subject, (c) every patch
# slot sits strictly BEFORE the WALK_GO write (outside the walk window),
# (d) the in-memory fence mirrors agree with the file-based originals
# (wfp.silence_predicate / wl.census) on the template.
#
# ══ THE PATCH FENCES (every patched program, before it is vouched) ═════════
#   * descriptor drift  — only wl.REPOINT_WORDS may move vs the template
#   * control-silence + census — re-run on the PATCHED op stream (mirrors
#     cross-checked against the originals at template time)
#   * cap-count         — the patched file must bake EXACTLY the template's
#     capture ledger
#   * (mode 'verify', enforced by the token_loop wrapper) the patched file
#     must be BYTE-IDENTICAL to a from-scratch full build of the same
#     subject — assert_byte_identical REFUSES otherwise
# Any mismatch anywhere REFUSES loudly (SystemExit); the cache never
# degrades to a silent rebuild once a template exists.
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif/top/l3"),
           str(REPO / "verif/seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import trace_to_regops as t2r                                  # noqa: E402
import gen_layer_ops as glo                                    # noqa: E402
import walk_fuel_layer as wfl                                  # noqa: E402
import walk_fuel_proj as wfp                                   # noqa: E402
import walk_layers_05b as wl                                   # noqa: E402
from apex_golden.fp import f64_to_f16_bits                     # noqa: E402

# ── the patch-slot addresses (single-sourced from the real modules) ─────────
QS_A = t2r.B_MB + t2r.QS            # 0x3020: squant composite push (pw)
XW0_A = t2r.B_MB + t2r.XW0          # 0x3040: K=2 weight beat, low word (w)
XW1_A = t2r.B_MB + t2r.XW1          # 0x3044: beat high word — always 0
LDATA_A = glo.LA["DATA"]            # 0x1078: LAYER_DATA (xrow preload)
WDPTR_A = 0x1000 | wfl.W_DPTR       # 0x1060: descriptor pointer
WDDATA_A = 0x1000 | wfl.W_DDATA     # 0x1064: descriptor data
WCTRL_A = 0x1000 | wfl.W_CTRL       # 0x105C: WALK_GO lives here (d=0x3)
WSTATUS_A = 0x1000 | wfl.W_STATUS   # 0x1068: the done poll

_SEP = (",", ":")                   # glo._emit's own JSON separators

# in-process template cache: (cache_dir, layer) -> {meta, lines, ops}.
# Pool workers each hold their own copy; the DISK template (recorded once
# per layer per run) is the shared source of truth.
_MEM: dict = {}


def _dline(o: dict) -> str:
    return json.dumps(o, separators=_SEP)


# ═══════════════ 1. the payload transform (inject_jobs' encoding) ══════════

def act_payloads(fams) -> tuple[list[int], list[int]]:
    """The staged-activation payload streams, in file order: one squant
    composite (QS push d) and one K=2 weight beat (XW0 write d) per staged
    element — EXACTLY g3.inject_jobs' encoding (decompose_f16 -> (v, comp);
    beat = (w0 & 0xFF) | ((w1 & 0xFF) << 8) with v = 127*w0 + w1).

    Vectorized, but carrying decompose_f16's OWN self-check wholesale: the
    one-RNE narrowing of v * 2^k must reproduce the staged f16 bits for
    every element, and the K=2 pair must be exactly representable — REFUSED
    otherwise, never rounded. Parity with the real builder emission is
    pinned by the token_loop selftest (inject_jobs through the LayerScript
    translator vs this function) and, end to end, by --emit-cache verify's
    byte-identity gate."""
    qs_all: list[int] = []
    xw_all: list[int] = []
    for fam in fams:
        arr = np.asarray(fam, dtype=np.float64).ravel()
        bits = np.asarray(f64_to_f16_bits(arr), dtype=np.uint16)
        s = (bits >> 15) & 1
        e = ((bits >> 10) & 0x1F).astype(np.int64)
        m = (bits & 0x3FF).astype(np.int64)
        if np.any(e == 0x1F):
            raise SystemExit("REFUSE: emit-cache payload — inf/nan f16 in "
                             "a staged activation family (not injectable)")
        mant = np.where(e == 0, m, 1024 + m)
        v = np.where(s == 1, -mant, mant)
        comp = (np.where(e == 0, -24, e - 25) + 127) << 23
        negz = (s == 1) & (e == 0) & (m == 0)     # -0.0: decompose's arm
        v = np.where(negz, -1, v)
        comp = np.where(negz, (-126 + 127) << 23, comp)
        # decompose_f16's self-check, wholesale: f16(v * 2^k) == bits
        k = (comp >> 23) - 127
        chk = np.asarray(f64_to_f16_bits(
            v.astype(np.float64) * np.power(2.0, k.astype(np.float64))),
            dtype=np.uint16)
        if not np.array_equal(chk, bits):
            bad = int(np.flatnonzero(chk != bits)[0])
            raise SystemExit(f"REFUSE: emit-cache payload self-check — "
                             f"element {bad} ({int(bits[bad]):#06x}) does "
                             f"not survive the one-RNE narrowing")
        w0 = np.rint(v / 127.0).astype(np.int64)
        w1 = v - 127 * w0
        if (np.any(w0 < -128) or np.any(w0 > 127)
                or np.any(w1 < -128) or np.any(w1 > 127)):
            raise SystemExit("REFUSE: emit-cache payload — a staged value "
                             "exceeds the K=2 loader-row bound")
        qs_all.extend(int(c) & 0xFFFFFFFF for c in comp)
        xw_all.extend(int((a & 0xFF) | ((b & 0xFF) << 8))
                      for a, b in zip(w0, w1))
    return qs_all, xw_all


# ═══════════════ 2. fence mirrors (cross-checked at template time) ═════════

def silence_ops(ops: list) -> dict:
    """wfp.silence_predicate on a parsed op list — same verdict dict, no
    file re-parse. record_template REFUSES unless this mirror agrees with
    the file-based original on the template, so mirror drift is loud."""
    go = [i for i, o in enumerate(ops)
          if o.get("op") == "w" and o.get("a") == WCTRL_A
          and o.get("d") == 0x3]
    if len(go) != 1:
        raise SystemExit(f"REFUSE: {len(go)} WALK_GO writes in the patched "
                         f"op stream")
    end = [i for i, o in enumerate(ops)
           if i > go[0] and o.get("op") == "poll"
           and o.get("a") == WSTATUS_A]
    if not end:
        raise SystemExit("REFUSE: no walk-done poll after WALK_GO in the "
                         "patched op stream")
    window = ops[go[0] + 1:end[0]]
    writes = [o for o in window if o.get("op") in ("w", "pw")]
    bad = [o for o in writes if o.get("a") != wfp.RO_ADV]
    return {"window_ops": len(window), "writes": len(writes),
            "ro_advances": len(writes) - len(bad),
            "control_writes": len(bad),
            "silent": not bad,
            "offenders": bad[:4]}


def census_ops(ops: list) -> dict:
    """wl.census on a parsed op list — same dict, cross-checked likewise."""
    from collections import Counter
    go = [i for i, o in enumerate(ops)
          if o.get("op") == "w" and o.get("a") == WCTRL_A
          and o.get("d") == 0x3]
    if len(go) != 1:
        raise SystemExit(f"REFUSE: {len(go)} WALK_GO writes in the patched "
                         f"op stream")
    done = [i for i, o in enumerate(ops)
            if i > go[0] and o.get("op") == "poll"
            and o.get("a") == WSTATUS_A]
    if not done:
        raise SystemExit("REFUSE: no walk-done poll in the patched op "
                         "stream")
    seg = {"pre_walk": ops[:go[0] + 1], "window": ops[go[0] + 1:done[0]],
           "post_walk": ops[done[0]:]}
    out = {k: dict(Counter(o["op"] for o in v)) for k, v in seg.items()}
    win_writes = [o for o in seg["window"] if o.get("op") in ("w", "pw")]
    out["window_ro_advances"] = sum(
        1 for o in win_writes if o.get("a") == wfp.RO_ADV)
    out["window_control_writes"] = sum(
        1 for o in win_writes if o.get("a") != wfp.RO_ADV)
    out["total_ops"] = len(ops)
    return out


# ═══════════════ 3. the per-patch fences ═══════════════════════════════════

def check_desc_drift(tpl_desc, desc, tag: str = "") -> list[int]:
    """The cross-step descriptor drift fence: vs the layer's template
    descriptor, only wl.REPOINT_WORDS may move (and for one layer the four
    bases never do — rq/comp are the words that actually change)."""
    tpl_desc, desc = list(tpl_desc), list(desc)
    if len(desc) != len(tpl_desc):
        raise SystemExit(f"REFUSE: {tag} emit-cache descriptor length "
                         f"{len(desc)} != template {len(tpl_desc)}")
    moved = [w for w in range(len(desc))
             if int(desc[w]) != int(tpl_desc[w])]
    if not set(moved) <= set(wl.REPOINT_WORDS):
        raise SystemExit(
            f"REFUSE: {tag} emit-cache descriptor moved words {moved} vs "
            f"the layer template — more than the re-point set "
            f"{sorted(wl.REPOINT_WORDS)} (wl.REPOINT_WORDS); the template "
            f"does not describe this walk and the patch is refused")
    return moved


def check_cap_fence(got: int, want: int, tag: str = "") -> None:
    """The capture-ledger fence: a patched program must bake EXACTLY the
    template's cap count — a moved ledger means the emission structure
    drifted and the cache refuses to vouch."""
    if int(got) != int(want):
        raise SystemExit(f"REFUSE: {tag} patched program bakes {got} caps, "
                         f"the layer template bakes {want} — capture "
                         f"ledger drift; the emit cache refuses")


def assert_byte_identical(p_patched, p_full, tag: str = "") -> int:
    """--emit-cache verify's teeth: the patched file must equal the full
    build byte for byte. Returns the byte count; REFUSES otherwise with
    the first divergent line named."""
    a = Path(p_patched).read_bytes()
    b = Path(p_full).read_bytes()
    if a == b:
        return len(a)
    la = a.decode(errors="replace").splitlines()
    lb = b.decode(errors="replace").splitlines()
    ln = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y),
              min(len(la), len(lb)))
    pa = la[ln][:120] if ln < len(la) else "<EOF>"
    pb = lb[ln][:120] if ln < len(lb) else "<EOF>"
    raise SystemExit(
        f"REFUSE: {tag} EMIT-CACHE VERIFY failed — patched {p_patched} is "
        f"NOT byte-identical to the full build {p_full} (first divergence "
        f"at line {ln + 1}: patched {pa!r} vs full {pb!r}); the "
        f"template/patch model does not reproduce the builders here and "
        f"the cache REFUSES")


# ═══════════════ 4. template record (from the REAL emission) ═══════════════

def _discover(ops: list, text: str, name: str, desc, sub) -> tuple[dict, str]:
    """Locate every patch slot STRUCTURALLY and verify the template's own
    payloads reproduce from the payload transform — the proof that patching
    a NEW subject through the same transform lands the builder's bytes."""
    def idxs(kind, addr):
        return [i for i, o in enumerate(ops)
                if o.get("op") == kind and o.get("a") == addr]

    qs_i = idxs("pw", QS_A)
    xw0_i = idxs("w", XW0_A)
    xw1_i = idxs("w", XW1_A)
    xr_i = idxs("w", LDATA_A)
    dp_i = idxs("w", WDPTR_A)
    dd_i = idxs("w", WDDATA_A)
    go_i = [i for i, o in enumerate(ops)
            if o.get("op") == "w" and o.get("a") == WCTRL_A
            and o.get("d") == 0x3]
    n_act = int(np.asarray(sub["h8"]).size + np.asarray(sub["attn8"]).size)
    n_xr = int(np.asarray(sub["xrow_bits"]).size)
    if len(dp_i) != 1 or ops[dp_i[0]].get("d") != 0:
        raise SystemExit(f"REFUSE: emit-cache template — {len(dp_i)} "
                         f"WALK_DPTR writes (want exactly one, d=0)")
    d0 = dp_i[0] + 1
    if dd_i != list(range(d0, d0 + len(desc))):
        raise SystemExit("REFUSE: emit-cache template — descriptor DDATA "
                         "writes are not one contiguous 64-word block")
    if len(go_i) != 1:
        raise SystemExit(f"REFUSE: emit-cache template — {len(go_i)} "
                         f"WALK_GO writes")
    if not (len(qs_i) == len(xw0_i) == len(xw1_i) == n_act):
        raise SystemExit(
            f"REFUSE: emit-cache template — staging stream counts "
            f"qs={len(qs_i)} xw0={len(xw0_i)} xw1={len(xw1_i)} != "
            f"{n_act} staged elements; not the cached MASK_B shape")
    if len(xr_i) != n_xr:
        raise SystemExit(f"REFUSE: emit-cache template — {len(xr_i)} "
                         f"LAYER_DATA writes != {n_xr} xrow elements")
    if any(ops[i].get("d") != 0 for i in xw1_i):
        raise SystemExit("REFUSE: emit-cache template — a nonzero XW1 beat "
                         "word (the K=2 encoding claim is false here)")
    last_slot = max(qs_i[-1], xw0_i[-1], xr_i[-1], d0 + len(desc) - 1)
    if last_slot >= go_i[0]:
        raise SystemExit("REFUSE: emit-cache template — a patch slot sits "
                         "at/after WALK_GO (inside the walk window)")
    # the template's own payloads must reproduce from the transform
    qs, xw0 = act_payloads([sub["h8"], sub["attn8"]])
    if [ops[i]["d"] for i in qs_i] != qs:
        raise SystemExit("REFUSE: emit-cache template — recorded QS stream "
                         "!= payload transform on the template subject")
    if [ops[i]["d"] for i in xw0_i] != xw0:
        raise SystemExit("REFUSE: emit-cache template — recorded XW0 "
                         "stream != payload transform on the template "
                         "subject")
    want_xr = [int(v) & 0xFFFF
               for v in np.asarray(sub["xrow_bits"]).ravel()]
    if [ops[i]["d"] for i in xr_i] != want_xr:
        raise SystemExit("REFUSE: emit-cache template — recorded xrow "
                         "preload != the subject's xrow bits")
    if [ops[i]["d"] for i in dd_i] != [int(v) & 0xFFFFFFFF for v in desc]:
        raise SystemExit("REFUSE: emit-cache template — recorded "
                         "descriptor words != the build descriptor")
    n_i = [i for i, o in enumerate(ops)
           if o.get("op") == "note" and name in str(o.get("s", ""))]
    if len(n_i) != 1 or text.count(name) != 1:
        raise SystemExit(
            f"REFUSE: emit-cache template — the program name {name!r} "
            f"appears {text.count(name)}x in the file / {len(n_i)}x in "
            f"note ops (want exactly 1 each; anything else would leave a "
            f"stale step tag in patched programs)")
    slots = {"qs": qs_i, "xw0": xw0_i, "xrow": xr_i, "desc0": d0,
             "note": n_i[0]}
    return slots, str(ops[n_i[0]]["s"])


def record_template(cfg: dict, li: int, path: Path, name: str, desc,
                    sub, n_caps: int) -> None:
    """Record layer li's template from the REAL emission at `path` (the
    post-capture_ro/splice_fuel_arm bytes token_loop just built and
    fence-checked) — never from a re-run. REFUSES unless every structural
    claim the patcher will rely on holds on these bytes."""
    d = Path(cfg["dir"])
    text = Path(path).read_text()
    lines = text.splitlines()
    ops = [json.loads(ln) for ln in lines if ln.strip()]
    if len(ops) != len(lines):
        raise SystemExit("REFUSE: emit-cache template — blank line in the "
                         "recorded emission")
    for i, (o, ln) in enumerate(zip(ops, lines)):
        if _dline(o) != ln:
            raise SystemExit(f"REFUSE: emit-cache template — op {i} does "
                             f"not JSON-round-trip byte-identically "
                             f"({ln[:80]!r}); patched files would drift")
    slots, note_s = _discover(ops, text, name, desc, sub)
    # fence-mirror cross-check on the template (drift in the mirrors is
    # caught HERE, once per layer, against the proven file-based originals)
    if silence_ops(ops) != wfp.silence_predicate(Path(path)):
        raise SystemExit("REFUSE: emit-cache — silence_ops mirror disagrees "
                         "with wfp.silence_predicate on the template")
    if census_ops(ops) != wl.census(Path(path)):
        raise SystemExit("REFUSE: emit-cache — census_ops mirror disagrees "
                         "with wl.census on the template")
    got_caps = sum(1 for o in ops if o.get("op") == "cap")
    if got_caps != int(n_caps):
        raise SystemExit(f"REFUSE: emit-cache template — {got_caps} caps "
                         f"in the file, builder reported {n_caps}")
    meta = {"run": cfg["run"], "li": int(li), "name": name,
            "note_s": note_s, "desc": [int(v) for v in desc],
            "n_caps": int(n_caps), "slots": slots,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
    d.mkdir(parents=True, exist_ok=True)
    (d / f"L{li:02d}.template.regops.jsonl").write_text(text)
    (d / f"L{li:02d}.meta.json").write_text(json.dumps(meta))
    _MEM[(cfg["dir"], int(li))] = {"meta": meta, "lines": lines, "ops": ops}


# ═══════════════ 5. the patcher ════════════════════════════════════════════

def _load(cfg: dict, li: int):
    key = (cfg["dir"], int(li))
    ent = _MEM.get(key)
    if ent is not None:
        return ent
    d = Path(cfg["dir"])
    meta_p = d / f"L{li:02d}.meta.json"
    tpl_p = d / f"L{li:02d}.template.regops.jsonl"
    if not (meta_p.is_file() and tpl_p.is_file()):
        return None
    meta = json.loads(meta_p.read_text())
    if meta.get("run") != cfg["run"]:
        raise SystemExit(
            f"REFUSE: emit-cache template for layer {li} carries run token "
            f"{meta.get('run')!r}, this run is {cfg['run']!r} — a stale "
            f"template must never patch a live run")
    raw = tpl_p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise SystemExit(f"REFUSE: emit-cache template bytes for layer "
                         f"{li} do not match their recorded sha — corrupt "
                         f"template")
    lines = raw.decode().splitlines()
    ops = [json.loads(ln) for ln in lines]
    ent = {"meta": meta, "lines": lines, "ops": ops}
    _MEM[key] = ent
    return ent


def patch_program(cfg: dict, li: int, sub: dict, desc, work: Path,
                  name: str):
    """Patch layer li's template with THIS step's payloads. Returns the
    _emit_program-shaped record {path, sil, n_caps} (regions are the
    caller's, unchanged), or None when no template exists yet (the caller
    then full-builds and records). Every fence failure REFUSES."""
    ent = _load(cfg, li)
    if ent is None:
        return None
    meta, lines, ops = ent["meta"], ent["lines"], ent["ops"]
    check_desc_drift(meta["desc"], desc, tag=name)
    qs, xw0 = act_payloads([sub["h8"], sub["attn8"]])
    xrow = [int(v) & 0xFFFF for v in np.asarray(sub["xrow_bits"]).ravel()]
    sl = meta["slots"]
    if (len(qs) != len(sl["qs"]) or len(xw0) != len(sl["xw0"])
            or len(xrow) != len(sl["xrow"]) or len(desc) != len(meta["desc"])):
        raise SystemExit(
            f"REFUSE: {name} emit-cache payload shape drift — "
            f"qs {len(qs)}/{len(sl['qs'])} xw0 {len(xw0)}/{len(sl['xw0'])} "
            f"xrow {len(xrow)}/{len(sl['xrow'])}; the subject no longer "
            f"matches the layer template")

    def put(idx_list, vals):
        for i, v in zip(idx_list, vals):
            o = ops[i]
            o["d"] = int(v) & 0xFFFFFFFF
            lines[i] = _dline(o)

    put(sl["qs"], qs)
    put(sl["xw0"], xw0)
    put(sl["xrow"], xrow)
    put(range(sl["desc0"], sl["desc0"] + len(desc)), desc)
    ni = sl["note"]
    ops[ni]["s"] = meta["note_s"].replace(meta["name"], name)
    lines[ni] = _dline(ops[ni])
    path = Path(work) / f"{name}.regops.jsonl"
    path.write_text("\n".join(lines) + "\n")
    # ── the cheap fences, re-run on the PATCHED output ─────────────────────
    sil = silence_ops(ops)
    cen = census_ops(ops)
    if cen["window_control_writes"] != 0 or not sil["silent"]:
        raise SystemExit(f"REFUSE: {name} patched walk window is not "
                         f"control-silent: {cen} / {sil}")
    n_caps = sum(1 for o in ops if o.get("op") == "cap")
    check_cap_fence(n_caps, meta["n_caps"], tag=name)
    return {"path": str(path), "sil": sil, "n_caps": n_caps}
