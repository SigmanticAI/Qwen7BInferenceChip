#!/usr/bin/env python3
# walk_layers_05b.py — THE OUTER LAYER LOOP: walk layer 0..N-1 of a REAL
# Qwen2.5-0.5B prompt decode step back to back, ONE fmt=1 descriptor per
# layer, against the 24-layer resident weight image.
#
#   python3 scripts/fpga/f2/walk_layers_05b.py selftest              # offline
#   python3 scripts/fpga/f2/walk_layers_05b.py run --n-layers 4      # sim
#   APEX_F2_HOST=... python3 ... run --executor hw \
#       --hw-ddr-attest <f2_ddr_load --full-verify result json>     # silicon
#
# ══ WHAT THIS IS (and the fence it lives above) ════════════════════════════
# E-6 (walk_fuel_layer.py, FLOWN 2026-08-06) proved ONE layer's walked chain:
# under one descriptor the sequencer fetches that layer's real weights from
# card DRAM, computes QKV + OPROJ with the on-tile requant epilogue, and
# drives r1 bit-exact. This driver is the LAYER LOOP above that proven chain:
# it re-runs the IDENTICAL walk for every layer of the model, with the host
# doing — per layer — only DATA MOVEMENT and CONTROL:
#
#   host, per layer l (all values PRE-COMPUTED before the loop; nothing is
#   computed between layers):
#     1. stage the two C-1 activation families into act bank 1
#        (rows 0..13: the layer's normed 'h' row for QKV; rows 14..27: the
#        layer's attention-output row for OPROJ — HEAD's fp_oproj_base)
#     2. preload the residual row RAM with the layer INPUT row X_l (f16)
#     3. write the 64-word fmt=1 descriptor — the ONLY things that change
#        layer to layer are the 4 tensor-table bases (re-pointed into the
#        resident image at layer l's regions) and the RQ[H]/JC_OPROJ
#        calibration slots (layer l's host-computed epilogue calibration,
#        the fmt=1 model's documented host-loaded fields)
#     4. arm fuel mode (walk_fuel_proj.splice_fuel_arm — the E-5/E-6 arm)
#     5. WALK_GO; drain the 144 raw QKV RO caps (data plane, E-5's
#        disclosed class — zero CONTROL writes in the window); ONE done-poll
#     6. read the 896 r1 caps back (produce mode — the walk's OUTPUT)
#   The descriptor-delta claim in 3 is not prose: selftest case 3 diffs the
#   64 words of consecutive layers' descriptors and REFUSES if anything but
#   W_TENS0+{WQ,WK,WV,WO}, W_RQ0+H and W_JC0+JC_OPROJ moved.
#
# ══ THE INTER-LAYER SEAM, STATED HONESTLY ══════════════════════════════════
# The walked step set is {QKV, OPROJ, RES1} — the deepest walked activation
# is r1. A layer's OUTPUT (the next layer's input X_{l+1}) is r2 = r1 +
# FFN(norm2(r1)), and NORM1/attention/NORM2/FFN/RES2 do NOT walk today
# (loud S2_CHECK refusals, measured in WALKED_EPILOGUE_E6.md). So the
# inter-layer carry is: the host stages golden's OWN continuation of the
# very r1 the tile just produced — X_{l+1} = golden r2_l, pre-computed once
# by the golden prefill before the loop, moved (not computed) between
# layers. The walked r1_l is graded BIT-EXACT against golden r1_l first,
# so the carried r2_l is the continuation OF A VERIFIED VALUE, but its
# NORM2/FFN/RES2 arithmetic is golden's, not the tile's. Said plainly:
# this drives and verifies 24 × {QKV, OPROJ, RES1} of a real decode step
# on the walked path; it does not make the unwalked five steps disappear.
#
# ══ GOLDEN IS THE ONLY ARBITER ═════════════════════════════════════════════
# Every expectation is computed from apex_golden public functions on the
# same operands, per layer, after the fact (walk_fuel_layer.golden_epilogue
# — reused, not reimplemented). Zero baked values. A layer whose C-1
# framing cannot be identity-staged (an amax != 127 frame) is REFUSED
# loudly and reported as refused — never silently skipped, never degraded.
#
# ══ EXECUTORS ══════════════════════════════════════════════════════════════
# sim (default): the DDR=1 E-6 twin, one f2sim process per layer (the
#   CYCMARK stamps are per-process, and f2_host_run's own per-file TILE_RST
#   gives hw the same per-layer reset semantics, so the loop's unit of
#   execution is honestly "one layer program" on both executors).
# hw: REFUSES loudly unless BOTH (a) remote_hw_exec.attach() installed the
#   shim ($APEX_F2_HOST set) and (b) --hw-ddr-attest points at an
#   f2_ddr_load.py --full-verify result whose image_sha256 matches THIS
#   image's manifest and whose full readback had 0 fails — a walk against a
#   card whose DDR content is unattested would grade nothing but hope.
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif/top/l3"),
           str(REPO / "verif/seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seq_walker_fmt as fmt                                   # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402
import fuel_proj_05b as fp                                     # noqa: E402
import walk_fuel_proj as wfp                                   # noqa: E402
import walk_fuel_layer as wfl                                  # noqa: E402
from apex_golden.fp import f64_to_f16_bits                     # noqa: E402

DEFAULT_IMAGE = REPO / "build/ddr_weights_05b_24L"
DEFAULT_WORK = REPO / "build/walk_layers_05b"
DEFAULT_OBJ = wfl.DEFAULT_OBJ                  # obj_e6r_b64_ddr1 (E-6 twin)
N_LAYERS_05B = 24

# ── clocks (labels, not knobs) ─────────────────────────────────────────────
# FLOWN: clkgen recipe A2 clk_extra_a1 = 15.625 MHz — the clock every flown
# APEX image verified at (remote_hw_exec.TILE_CLK_MHZ gates on it).
# ANALYZED: 62.5 MHz — the analyzed-not-flown faster point the E-6 doc
# quotes (WALKED_EPILOGUE_E6.md prediction section). Any number derived at
# it is a PREDICTION, never a measurement.
FLOWN_MHZ = 15.625
ANALYZED_MHZ = 62.5

# fmt=1 words allowed to differ between two layers' descriptors: the four
# tensor-table bases + the two per-layer epilogue calibration slots.
REPOINT_WORDS = tuple(sorted(
    [fmt.W_TENS0 + t for t in (fmt.TENS_WQ, fmt.TENS_WK, fmt.TENS_WV,
                               fmt.TENS_WO)]
    + [fmt.W_RQ0 + wfl.N_HEADS, fmt.W_JC0 + fmt.JC_OPROJ]))


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═══════════ 1. golden, ONCE, for every layer (the pre-loop pass) ══════════

def golden_all(idir: Path, n_layers: int) -> dict:
    """One golden prefill; per-layer subjects in walk_fuel_layer's own
    shape. Mirrors wfl.golden_subject but runs the (expensive) prefill
    ONCE for all layers instead of once per layer."""
    man = json.loads((idir / "ddr_image.json").read_text())
    have = set(man["layers"])
    want = set(range(n_layers))
    if not want <= have:
        raise SystemExit(
            f"REFUSE: image {idir} holds layers {sorted(have)} but the loop "
            f"needs {sorted(want)} — rebuild it (make_weight_image.py build "
            f"--layers {','.join(str(i) for i in range(n_layers))})")
    wdir = Path(man["weights_dir"])
    model = fp.rt.GoldenModel(wdir)
    fxs = fp.golden_prefill(model, fp.PROMPT_IDS, fp.rt.TIER_MAP["kvq8"], 128)
    return {"man": man, "wdir": wdir, "model": model, "fxs": fxs}


def subject_for(g: dict, layer: int) -> dict:
    """Layer `layer`'s walk subject — same field shape as
    wfl.golden_subject (checked against it, layer 0, in run()).
    c1_frames REFUSES (SystemExit) on a frame whose amax != 127; the
    caller turns that into a loud per-layer refusal record."""
    man, wdir, model, fxs = g["man"], g["wdir"], g["model"], g["fxs"]
    fx = fxs[layer]
    if layer == 0:
        xrow = np.asarray(model.embed_row(fp.PROMPT_IDS[-1]), dtype=np.float64)
    else:
        xrow = np.asarray(fxs[layer - 1].r2, dtype=np.float64)
    xrow_bits = np.asarray(f64_to_f16_bits(xrow), dtype=np.uint16)
    attn8, _ = fp.c1_frames(np.asarray(fx.attn, dtype=np.float64), wfl.D_TILE)
    h8, _ = fp.c1_frames(fp.activation_for(fx, "h"), wfl.D_TILE)
    Wo = np.load(wdir / f"L{layer:02d}_Wo.npy")
    s_wo = float(model.layers[layer].s_wo)
    bases = {t["tensor"]: t["base_64B"] for t in man["tensors"]
             if t["layer"] == layer}
    blocks = {n: np.asarray(np.load(wdir / f"L{layer:02d}_{n}.npy"),
                            dtype=np.int64) for n, _ in wfl.QKV}
    return {"man": man, "wdir": wdir, "bases": bases, "xrow_bits": xrow_bits,
            "attn8": np.asarray(attn8, dtype=np.int64),
            "h8": np.asarray(h8, dtype=np.int64),
            "Wo": np.asarray(Wo, dtype=np.int64), "s_wo": s_wo,
            "blocks": blocks}


# ═══════════ 2. per-layer host-op census (what the host DOES) ══════════════

def census(regops_path: Path) -> dict:
    """Count what the host does in one layer's program, by segment:
    pre-walk (staging + descriptor + arm + GO), the walk window (between
    GO and the done-poll), post-walk (drains/readback/final). Op counts
    come from the artefact itself, never from prose."""
    ops = [json.loads(l) for l in Path(regops_path).read_text().splitlines()
           if l.strip()]
    go = [i for i, o in enumerate(ops)
          if o.get("op") == "w" and o.get("a") == (0x1000 | wfl.W_CTRL)
          and o.get("d") == 0x3]
    if len(go) != 1:
        raise SystemExit(f"REFUSE: {len(go)} WALK_GO writes in {regops_path}")
    done = [i for i, o in enumerate(ops)
            if i > go[0] and o.get("op") == "poll"
            and o.get("a") == (0x1000 | wfl.W_STATUS)]
    if not done:
        raise SystemExit(f"REFUSE: no walk-done poll in {regops_path}")
    seg = {"pre_walk": ops[:go[0] + 1], "window": ops[go[0] + 1:done[0]],
           "post_walk": ops[done[0]:]}
    out = {k: dict(Counter(o["op"] for o in v)) for k, v in seg.items()}
    # wfp.silence_predicate's own classification: the RO advance (0x3228) is
    # the disclosed DATA plane (the QKV drain pop); anything else is control.
    win_writes = [o for o in seg["window"] if o.get("op") in ("w", "pw")]
    out["window_ro_advances"] = sum(
        1 for o in win_writes if o.get("a") == wfp.RO_ADV)
    out["window_control_writes"] = sum(
        1 for o in win_writes if o.get("a") != wfp.RO_ADV)
    out["total_ops"] = len(ops)
    return out


# ═══════════ 3. executor plumbing (sim default; hw gated LOUDLY) ═══════════

def require_hw_attached(args) -> None:
    """--executor hw: refuse unless the remote shim attached AND the card's
    DDR content is attested against THIS image. No silent fallbacks."""
    import remote_hw_exec
    if not remote_hw_exec.attach(bridge, args):
        raise SystemExit(
            "REFUSED --executor hw: remote_hw_exec.attach() installed no "
            "shim — $APEX_F2_HOST is not set. This driver never guesses an "
            "instance; export APEX_F2_HOST=ubuntu@<f2-ip> and "
            "APEX_F2_KEY=~/.ssh/apex-f2.pem (see remote_hw_exec.py header).")
    if not getattr(args, "hw_ddr_attest", None):
        raise SystemExit(
            "REFUSED --executor hw: no --hw-ddr-attest. The walk fetches "
            "weights from the CARD's DDR; without an f2_ddr_load.py "
            "--full-verify result for THIS image the run would grade an "
            "unattested memory. Load + verify first:\n"
            "  python3 scripts/fpga/f2/f2_ddr_load.py --image <image> "
            "--load --verify --full-verify --out ddr_attest.json")
    att = json.loads(Path(args.hw_ddr_attest).read_text())
    man = json.loads((Path(args.image) / "ddr_image.json").read_text())
    v = att.get("verify", {})
    if att.get("image_sha256") != man["image_sha256"]:
        raise SystemExit(
            f"REFUSED --executor hw: attestation sha "
            f"{att.get('image_sha256')} != this image's manifest sha "
            f"{man['image_sha256']} — the card holds a DIFFERENT image.")
    if v.get("mode") != "full" or v.get("fails", 1) != 0:
        raise SystemExit(
            f"REFUSED --executor hw: attestation is mode={v.get('mode')} "
            f"fails={v.get('fails')} — only a --full-verify readback with 0 "
            f"fails attests the resident bytes (sampled proves presence, "
            f"not content).")


def layer_regions(idir: Path, work: Path, li: int) -> Path:
    """SIM-HARNESS ECONOMY, disclosed: each per-layer f2sim process loads
    only layer li's 10 regions into the behavioral DDR instead of all 240.
    The bin is the FULL image and every base is the full-image address
    (L23's Wq still lands at 0x1476_4000), so the walker's fetches and the
    walk window are byte- and cycle-identical — only the per-process
    behavioral load+verify segment shrinks (~24x, measured: 13.99M ->
    ~0.6M shell cycles). A region-count mismatch REFUSES. Not a hardware
    knob: on the card the full image is resident once, attested."""
    src = idir / "ddr_image.regions.jsonl"
    dst = work / f"regions_L{li:02d}.jsonl"
    keep = [l for l in src.read_text().splitlines()
            if l.strip() and json.loads(l).get("layer") == li]
    if len(keep) != fmt.TENS_N:
        raise SystemExit(f"REFUSE: {len(keep)} regions for layer {li} in "
                         f"{src} (want {fmt.TENS_N}) — the image does not "
                         f"hold this layer")
    dst.write_text("\n".join(keep) + "\n")
    return dst


def run_one(files, args, binary, regions: Path):
    if args.executor == "sim":
        idir = Path(args.image)
        extra = (f"+ddr_image={idir}/ddr_image.bin",
                 f"+ddr_regions={regions}",
                 "+fuel_audit")
        return bridge.run_job(files, executor="sim", binary=str(binary),
                              extra_args=extra, timeout_s=args.timeout,
                              tile_div=wfl.TILE_DIV)
    # hw: the shim installed by require_hw_attached routes this; the DDR
    # plusargs are sim-only and the tile runs at the clock the shim's gate
    # verified, so neither is passed.
    return bridge.run_job(files, executor="hw", timeout_s=args.timeout)


# ═══════════ 4. the loop ═══════════════════════════════════════════════════

def run(args) -> int:
    if args.executor not in ("sim", "hw"):
        raise SystemExit(f"REFUSE: unknown executor {args.executor!r}")
    if args.executor == "hw":
        require_hw_attached(args)
    idir, work = Path(args.image), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    binary = REPO / "verif/f2sim" / args.obj / "f2sim"
    if args.executor == "sim" and not binary.is_file():
        raise SystemExit(f"REFUSE: sim binary missing: {binary} — build the "
                         f"E-6 twin first (walk_fuel_layer.py header recipe)")
    n = int(args.n_layers)

    print(f"[1] golden prefill (ONCE) + {n} per-layer subjects …")
    t0 = time.monotonic()
    g = golden_all(idir, n)
    t_prefill = time.monotonic() - t0
    print(f"    prefill {t_prefill:.1f}s (prompt {fp.PROMPT_IDS}, all "
          f"{N_LAYERS_05B} layers composed once, pre-loop)")

    # parity fence: my one-prefill subject builder against the flown
    # driver's own per-layer builder, on layer 0 — drift refuses here.
    subj0 = subject_for(g, 0)
    ref0 = wfl.golden_subject(idir, 0)
    for k in ("xrow_bits", "attn8", "h8", "Wo"):
        if not np.array_equal(subj0[k], ref0[k]):
            raise SystemExit(f"REFUSE: subject_for(0)[{k}] != "
                             f"wfl.golden_subject(0)[{k}] — the loop's "
                             f"subject builder drifted from the flown one")
    if subj0["bases"] != ref0["bases"] or subj0["s_wo"] != ref0["s_wo"]:
        raise SystemExit("REFUSE: layer-0 bases/s_wo drifted from "
                         "wfl.golden_subject")
    print("    layer-0 subject parity vs walk_fuel_layer.golden_subject: OK")

    jobs = wfl.expected_qkv_jobs()
    rows: list[dict] = []
    desc_prev = None
    for li in range(n):
        tag = f"L{li:02d}"
        print(f"[2.{li}] {tag}: subject -> descriptor -> program -> "
              f"{args.executor} …")
        try:
            sub = subj0 if li == 0 else subject_for(g, li)
        except SystemExit as e:
            print(f"    {tag} REFUSED (subject): {e}")
            rows.append({"layer": li, "refused": str(e)})
            continue
        ge = wfl.golden_epilogue(sub["attn8"], sub["Wo"], sub["s_wo"],
                                 sub["xrow_bits"])
        desc = wfl.walk_descriptor(sub["bases"], mask=wfl.MASK_B,
                                   rq=(ge["scale"], ge["shift"]),
                                   comp=ge["comp"])
        if desc_prev is not None:
            moved = [w for w in range(fmt.DESC_WORDS)
                     if desc[w] != desc_prev[w]]
            if not set(moved) <= set(REPOINT_WORDS):
                raise SystemExit(
                    f"REFUSE: {tag} descriptor moved words {moved} — more "
                    f"than the re-point set {list(REPOINT_WORDS)}; the "
                    f"'host only re-points and re-arms' claim would be "
                    f"false")
        desc_prev = desc
        man = wfl.build_program(sub, desc, work, f"walk_{tag}",
                                act8=sub["h8"], qkv_drain=True,
                                act8_oproj=sub["attn8"])
        path = Path(man["path"])
        wfp.capture_ro(path)
        wfp.splice_fuel_arm(path)
        sil = wfp.silence_predicate(path)
        cen = census(path)
        if cen["window_control_writes"] != 0 or not sil["silent"]:
            raise SystemExit(f"REFUSE: {tag} walk window is not "
                             f"control-silent: {cen} / {sil}")
        regions = (layer_regions(idir, work, li) if args.executor == "sim"
                   else idir / "ddr_image.regions.jsonl")
        t0 = time.monotonic()
        r = run_one([str(path)], args, binary, regions)
        wall = time.monotonic() - t0
        ro, lrd = wfl.split_caps(r["captures"])
        res_q = wfp.grade(ro, jobs, sub["h8"], sub["blocks"]) if r["ok"] \
            else {"error": f"rc={r['rc']}"}
        res_r1 = wfl.grade_r1(lrd, ge["r1_bits"]) if r["ok"] \
            else {"error": f"rc={r['rc']}"}
        cyc = wfl.parse_cycles(r["log"])
        aud = r["log"].count("-> ok")
        row = {"layer": li, "ok": bool(r["ok"]), "rc": r["rc"],
               "qkv": {k: v for k, v in res_q.items() if k != "per_job"},
               "r1": res_r1, "silence": sil, "census": cen,
               "cycles": cyc, "fuel_audit_ok": aud,
               "wall_s": round(wall, 2),
               "rq": [ge["scale"], ge["shift"]],
               "comp": f"{ge['comp']:#010x}"}
        rows.append(row)
        print(f"    {tag} rc={r['rc']} QKV bit-exact="
              f"{res_q.get('all_bit_exact')} ({res_q.get('jobs')} jobs) "
              f"r1 bit-exact={res_r1.get('bit_exact')} "
              f"({res_r1.get('n')} elems, bad={res_r1.get('bad')}) "
              f"tile_cyc={cyc.get('walk_tile_cycles')} wall={wall:.1f}s")

    walked = [r for r in rows if "refused" not in r]
    refused = [r for r in rows if "refused" in r]
    all_exact = walked and all(
        r["ok"] and r["qkv"].get("all_bit_exact") and r["r1"].get("bit_exact")
        for r in walked)
    pred = predict(walked, n)
    verdict = bool(all_exact and not refused and len(walked) == n)

    print("-" * 78)
    print(f"  layers walked        : {len(walked)}/{n}"
          + (f"  (REFUSED: {[r['layer'] for r in refused]})" if refused
             else ""))
    print(f"  every walked layer   : QKV 144/144 + r1 896/896 bit-exact -> "
          f"{'PASS' if all_exact else 'FAIL'}")
    print(f"  LAYER LOOP GATE      : {'PASS' if verdict else 'FAIL'}")
    print_prediction(pred)
    out = work / "walk_layers_result.json"
    out.write_text(json.dumps(
        {"verdict": verdict, "executor": args.executor,
         "image": str(idir), "n_layers": n,
         "prompt_ids": fp.PROMPT_IDS, "prefill_s": round(t_prefill, 1),
         "mask": "{FPROJ, QKV, OPROJ, RES1} (wfl.MASK_B)",
         "layers": rows, "prediction": pred}, indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if verdict else 1


# ═══════════ 5. measured aggregates + the LABELLED prediction ══════════════

def tile_ms(tile_cycles: float, mhz: float) -> float:
    return tile_cycles / (mhz * 1e6) * 1e3


def predict(walked: list[dict], n_asked: int) -> dict:
    """Measured sums + the labelled hw-wall prediction.

    MEASURED: per-layer walk-window tile cycles (CYCMARK shell counts /
    (2*TILE_DIV) — the corrected cl_apex.sv:82 conversion, sim only) and
    per-layer sim wall.
    PREDICTION (labelled): hw per-token wall = 24 x the mean measured
    walked window at each clock, plus the same for a HYPOTHETICAL fully
    walked layer scaled by MXE jobs (1808 vs the walked 256) at the
    measured cycles/job. Nothing here is a hardware measurement.
    """
    meas = [r for r in walked
            if r.get("cycles", {}).get("walk_tile_cycles")]
    out = {"measured": {}, "prediction": {}}
    if not meas:
        out["measured"]["note"] = ("no CYCMARK stamps (hw executor has no "
                                   "sim cycle counter) — walls only")
        out["measured"]["wall_s_per_layer"] = [r["wall_s"] for r in walked]
        return out
    tc = [r["cycles"]["walk_tile_cycles"] for r in meas]
    sc = [r["cycles"]["walk_shell_cycles"] for r in meas]
    walls = [r["wall_s"] for r in meas]
    jobs_walked = 256                     # 144 QKV + 112 OPROJ (chain B)
    jobs_full = full_layer_jobs()
    mean_tc = sum(tc) / len(tc)
    cyc_per_job = mean_tc / jobs_walked
    out["measured"] = {
        "n_layers_measured": len(meas),
        "tile_div": wfl.TILE_DIV,
        "walk_shell_cycles": sc, "walk_tile_cycles": tc,
        "tile_cycles_mean": round(mean_tc, 1),
        "tile_cycles_min_max": [min(tc), max(tc)],
        "tile_cycles_per_job_mean": round(cyc_per_job, 1),
        "sim_wall_s_per_layer": walls,
        "sim_wall_s_total": round(sum(walls), 1),
        "conversion": "tile = shell / (2*TILE_DIV)  [cl_apex.sv:82, the "
                      "corrected E-6 receipts rule]"}
    extrap = N_LAYERS_05B / len(meas) if len(meas) < N_LAYERS_05B else 1.0
    tok_tc = mean_tc * N_LAYERS_05B
    full_tc = cyc_per_job * jobs_full * N_LAYERS_05B
    out["prediction"] = {
        "LABEL": "PREDICTION — sim-derived, hardware-unvalidated. Walls "
                 "count the WALK WINDOW only: today's seam also spends "
                 "~12.3k host regops/layer on pre-walk staging + readback "
                 "(measured census in this record), which on hardware is "
                 "MMIO-bound and NOT included. They are NOT hardware "
                 "measurements.",
        "basis": {"measured_layers": len(meas),
                  "extrapolated_to_24_layers": extrap != 1.0,
                  "walked_jobs_per_layer": jobs_walked,
                  "full_layer_jobs": jobs_full,
                  "tile_cycles_per_job": round(cyc_per_job, 1)},
        "walked_steps_only_per_token": {
            "tile_cycles": round(tok_tc),
            "wall_ms_at_15p625MHz_flown": round(tile_ms(tok_tc, FLOWN_MHZ), 1),
            "wall_ms_at_62p5MHz_analyzed": round(
                tile_ms(tok_tc, ANALYZED_MHZ), 1),
            "covers": "24 x walked {QKV, OPROJ, RES1} ONLY — 3 of 13 step "
                      "families; NOT a full token"},
        "if_fully_walked_per_token": {
            "tile_cycles": round(full_tc),
            "wall_ms_at_15p625MHz_flown": round(tile_ms(full_tc, FLOWN_MHZ)),
            "wall_ms_at_62p5MHz_analyzed": round(
                tile_ms(full_tc, ANALYZED_MHZ)),
            "covers": "24 x ALL 1808 MXE jobs/layer at the measured "
                      "cycles/job — HYPOTHETICAL: NORM1/2, attention, "
                      "FFN/SWIGLU/DOWN, RES2 do not walk on today's RTL "
                      "(loud S2_CHECK refusals, WALKED_EPILOGUE_E6.md)"}}
    return out


def full_layer_jobs() -> int:
    """MXE jobs of a FULLY walked 0.5B layer, from the walker's own
    decomposition (fmt.jobs at d_tile=64) over the 7 weight tensors."""
    shapes = fmt.tensor_shapes(wfl.D_MODEL, 4864, wfl.N_KV_HEADS,
                               wfl.HEAD_DIM, t_max=fmt.T_MAX)
    return sum(len(fmt.jobs(*shapes[t], cfg_d=wfl.D_TILE))
               for t in (fmt.TENS_WQ, fmt.TENS_WK, fmt.TENS_WV, fmt.TENS_WO,
                         fmt.TENS_WG, fmt.TENS_WU, fmt.TENS_WD))


def print_prediction(pred: dict) -> None:
    m, p = pred.get("measured", {}), pred.get("prediction", {})
    if m.get("walk_tile_cycles"):
        print(f"  MEASURED (sim, tile_div={m['tile_div']}): "
              f"{m['tile_cycles_mean']:.0f} tile cyc/layer walked window "
              f"(min..max {m['tile_cycles_min_max']}), "
              f"{m['tile_cycles_per_job_mean']:.0f} cyc/job, "
              f"sim wall {m['sim_wall_s_total']}s total")
    if p:
        w = p["walked_steps_only_per_token"]
        f = p["if_fully_walked_per_token"]
        print(f"  {p['LABEL']}")
        print(f"    walked-steps-only/token: {w['tile_cycles']} tile cyc = "
              f"{w['wall_ms_at_15p625MHz_flown']} ms @ flown 15.625 MHz / "
              f"{w['wall_ms_at_62p5MHz_analyzed']} ms @ analyzed 62.5 MHz")
        print(f"    IF-fully-walked/token  : {f['tile_cycles']} tile cyc = "
              f"{f['wall_ms_at_15p625MHz_flown']} ms @ flown / "
              f"{f['wall_ms_at_62p5MHz_analyzed']} ms @ analyzed "
              f"(hypothetical — 5 step families do not walk today)")


# ═══════════ 6. selftest (offline: no sim, no card, no prefill) ════════════

def selftest() -> int:
    ok = []

    def chk(cond, msg):
        ok.append(bool(cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {len(ok)} {msg}")

    man_p = DEFAULT_IMAGE / "ddr_image.json"
    if not man_p.exists():
        print(f"SELFTEST REFUSED: no 24-layer image at {DEFAULT_IMAGE} — "
              f"build it first (make_weight_image.py build --layers 0..23)")
        return 1
    man = json.loads(man_p.read_text())

    # 1. the image really holds all 24 layers, every tensor, walker-addressable
    per_layer = Counter(t["layer"] for t in man["tensors"])
    chk(sorted(man["layers"]) == list(range(N_LAYERS_05B))
        and all(per_layer[li] == fmt.TENS_N for li in range(N_LAYERS_05B)),
        f"image holds {fmt.TENS_N} tensors x {N_LAYERS_05B} layers")
    chk(all(t["base_64B"] * 64 % 4096 == 0 for t in man["tensors"]),
        "every region 4 KiB aligned (sim loader + IB_FUEL rule)")
    chk(all(0 <= t["base_64B"] < (1 << fmt.FR_BASE_W)
            for t in man["tensors"]),
        f"every base fits the walker's {fmt.FR_BASE_W}-bit tensor-table "
        f"word")
    chk(int(man["geometry"]["K_JOB"]) == fmt.k_job(
        int(man["geometry"]["d_tile"])),
        "manifest K_JOB is the D-aware k_job (third-erratum current)")

    # 2. per-layer addressing is one uniform stride (report it)
    wq = {t["layer"]: t["base_64B"] * 64 for t in man["tensors"]
          if t["tensor"] == "Wq"}
    strides = {wq[l + 1] - wq[l] for l in range(N_LAYERS_05B - 1)}
    chk(len(strides) == 1,
        f"layer base stride is uniform: {strides} B "
        f"(base(L,T) = base(0,T) + L*stride)")

    # 3. THE RE-POINT CLAIM: consecutive layers' descriptors differ ONLY in
    # the 4 tensor bases + RQ[H] + JC_OPROJ (synthetic rq/comp — no prefill)
    bases = {li: {t["tensor"]: t["base_64B"] for t in man["tensors"]
                  if t["layer"] == li} for li in (0, 1)}
    d0 = wfl.walk_descriptor(bases[0], mask=wfl.MASK_B, rq=(1111, 5),
                             comp=0x3F000000)
    d1 = wfl.walk_descriptor(bases[1], mask=wfl.MASK_B, rq=(2222, 6),
                             comp=0x3E800000)
    moved = [w for w in range(fmt.DESC_WORDS) if d0[w] != d1[w]]
    chk(set(moved) == set(REPOINT_WORDS),
        f"L0->L1 descriptor delta is EXACTLY the re-point set "
        f"(words {sorted(moved)})")
    chk(all(fmt.check2(d[0], d[1], d[2], d[3], d[fmt.W_STEP],
                       cfg_d=wfl.HEAD_DIM) == fmt.ERR_NONE
            for d in (d0, d1)),
        "both layers' descriptors are fmt=1 LEGAL")

    # 4. hw refusal is LOUD: no $APEX_F2_HOST -> SystemExit before any run
    import os
    saved = os.environ.pop("APEX_F2_HOST", None)
    try:
        ns = argparse.Namespace(image=str(DEFAULT_IMAGE), hw_ddr_attest=None)
        try:
            require_hw_attached(ns)
            chk(False, "hw without $APEX_F2_HOST is REFUSED")
        except SystemExit as e:
            chk("APEX_F2_HOST" in str(e), "hw without $APEX_F2_HOST is "
                                          "REFUSED, loudly")
    finally:
        if saved is not None:
            os.environ["APEX_F2_HOST"] = saved

    # 5. a wrong-sha / sampled attestation is REFUSED (stub attach via env)
    import tempfile
    os.environ["APEX_F2_HOST"] = "ubuntu@203.0.113.1"      # TEST-NET-3
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump({"image_sha256": "not-the-sha",
                       "verify": {"mode": "full", "fails": 0}}, f)
        ns = argparse.Namespace(image=str(DEFAULT_IMAGE),
                                hw_ddr_attest=f.name)
        try:
            require_hw_attached(ns)
            chk(False, "wrong-sha DDR attestation is REFUSED")
        except SystemExit as e:
            chk("DIFFERENT image" in str(e),
                "wrong-sha DDR attestation is REFUSED")
        with open(f.name, "w") as fh:
            json.dump({"image_sha256": man["image_sha256"],
                       "verify": {"mode": "sampled", "fails": 0}}, fh)
        try:
            require_hw_attached(ns)
            chk(False, "sampled attestation is REFUSED")
        except SystemExit as e:
            chk("full" in str(e), "sampled (not full) attestation is REFUSED")
        os.unlink(f.name)
    finally:
        import remote_hw_exec
        remote_hw_exec.detach(bridge)
        os.environ.pop("APEX_F2_HOST", None)
        if saved is not None:
            os.environ["APEX_F2_HOST"] = saved

    # 6. the census on the FLOWN E-6 artefact (if present): GO found, zero
    # control writes inside the window
    e6 = REPO / "build/walk_fuel_layer/walk_qkv_oproj.regops.jsonl"
    if e6.exists():
        c = census(e6)
        chk(c["window_control_writes"] == 0,
            f"E-6 flown artefact: walk window has 0 control writes "
            f"(census {c['window']})")
    else:
        print("  SKIP  (no E-6 artefact for the census cross-check)")

    # 7. conversion + prediction arithmetic (the corrected rule)
    chk(abs(tile_ms(559_697, FLOWN_MHZ) - 35.82) < 0.1
        and abs(tile_ms(559_697, ANALYZED_MHZ) - 8.955) < 0.02,
        "tile-cycles -> wall arithmetic at both clocks (E-6 chain-B value)")
    fake = [{"cycles": {"walk_tile_cycles": 559_697,
                        "walk_shell_cycles": 559_697 * 2 * wfl.TILE_DIV},
             "wall_s": 1.0}]
    p = predict(fake, 1)
    chk(p["prediction"]["basis"]["full_layer_jobs"] == 1808
        and p["measured"]["tile_cycles_per_job_mean"] == round(559_697 / 256,
                                                              1),
        f"full-layer job count from fmt.jobs = 1808; cyc/job from the "
        f"measured window")
    chk("PREDICTION" in p["prediction"]["LABEL"],
        "the hw-wall numbers carry the PREDICTION label")

    print("=" * 62)
    print(f"WALK_LAYERS_05B SELFTEST: {'ALL PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(
        description="Walk layer 0..N-1 of a real 0.5B decode step back to "
                    "back — one fmt=1 descriptor per layer over the "
                    "24-layer resident weight image; host = data movement "
                    "+ control only between layers.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--image", default=str(DEFAULT_IMAGE))
    r.add_argument("--work", default=str(DEFAULT_WORK))
    r.add_argument("--obj", default=DEFAULT_OBJ)
    r.add_argument("--n-layers", type=int, default=N_LAYERS_05B)
    r.add_argument("--executor", default="sim", choices=("sim", "hw"))
    r.add_argument("--timeout", type=int, default=5400,
                   help="per-layer executor timeout (s)")
    r.add_argument("--hw-ddr-attest", default=None,
                   help="hw only: f2_ddr_load.py --full-verify result JSON "
                        "for THIS image (refused without it)")
    a = p.parse_args()
    return selftest() if a.cmd == "selftest" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
