#!/usr/bin/env python3
"""B1 stage-0: project an L3 case .txt into a canonical per-channel control trace.

The L3 choreography IS the walker's spec (docs/OPTIMIZATION.md:38). This tool
reads a generated `build/cases/<name>.txt` and emits the canonical
control-drive subsequence the walker must reproduce, split per channel and
annotated with the two things a naive line-diff loses:

  1. PHASE  — which choreography phase each op belongs to, so the buildable
     score+pv subset can be separated from the store_kv / q-inject injection
     scaffolding (which is testbench data-path setup, not decode control).

  2. PROVENANCE for every scale composite — where the fp16 scale that feeds a
     CS/QS word first becomes OBSERVABLE on the tile's taps, versus where the
     host emits the composite. This is the load-bearing measurement: the host
     precomputes scales from the golden and can emit composites early; an
     on-tile composite unit reads fs_*/ss_* and cannot. Every INVERTED or
     UNRESOLVED row here is a place where a correct walker MUST diverge from
     the host's emission order.

This is a *projection* of the spec stream. It never edits, imports, or
second-guesses the golden; gen_l3_vectors.py stays untouched.

Usage:
    ./extract_trace.py <build_dir>                    # all manifest cases
    ./extract_trace.py <build_dir> --case calib_d64_T128 --json out.json
    ./extract_trace.py <build_dir> --ordering         # ordering report only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── channel taxonomy ─────────────────────────────────────────────────────────
# What the walker must autonomously EMIT (B1_WALKER.md §2 drive-subset).
CONTROL = {"DESC", "ROUTE", "AJ", "WJ", "FJOB", "QJOB", "SJOB", "LJOB",
           "KVW", "CSRW", "IMP", "CS", "QS"}
# Host status round-trips. The walker replaces these with internal handshakes,
# but must preserve the ORDERING they enforce (change a route only when idle).
SYNC = {"KVP", "CSRP", "CSRR", "CSRN"}
# Tensor payload — explicitly NOT walked (B1_WALKER.md:42-46).
DATA = {"XR", "GR", "WB"}
# TB-side expectations (the checks; not driven).
EXPECT = {"EFS", "ESS", "ERO", "ETIP", "ETIPT", "ESTK"}
# Stateful tap headers: "<KIND> <count:hex>" then <count> BARE hex lines.
TAPS = {"TAPSC", "TAPPR", "TAPF16"}
MARKERS = {"ENDTAPS", "DONE"}

# ── phase mapping (every marker emitted by gen_l3_vectors.py) ────────────────
# Ordered longest-prefix-first; an unrecognised comment KEEPS the current phase
# (sub-comments like "// key group tokens .." must not reset it).
_PHASE_MARKS = [
    ("// L3 ", "header"),
    ("// phase A", "phase_a"),
    ("// inject K/V fp16 rows", "store_kv"),
    ("// block ", "store_kv"),
    ("// inject q row", "q_inject"),
    ("// q inject", "q_inject"),
    ("// phase B", "hidden"),
    ("// phase C", "store_kv"),
    ("// phase D", "q_inject"),
    ("// phase F", "score"),
    ("// phase G", "pv"),
    ("// final", "final"),
    ("// pass 1: CQ-8 store", "store_kv"),
    ("// pass 1 profiling", "profile"),
    ("// profile block", "profile"),
    ("// IMPORTANCE_BASE window", "auto_setup"),
    ("// AUTO MODE ON", "auto_setup"),
    ("// pass 2: TIP-DRIVEN", "store_kv"),
    ("// per-engine occupancy", "auto_setup"),
    ("// pass 2 attention", "score"),
]

# Phases whose QS words are the S-4 P-requant fold (walker-owned).
# Everywhere else QS is a decompose_f16 per-element injection composite.
_S4_PHASES = {"score", "profile"}
# Phases that are the buildable B1 target (see B1_WALKER.md §1 scope reality).
WALKER_TARGET_PHASES = {"score", "pv"}


# `loader_phase` is called both as a top-level phase AND nested inside
# tip_auto's pass-2 store (gen_l3_vectors.py:787). Its marker must therefore
# NOT clobber an enclosing phase, or ~9.4k pass-2 store ops get mislabelled.
_NESTABLE = {"// loader token": "loader"}

_MARKS_SEEN: set[str] = set()


def _phase_of(comment: str, current: str) -> str:
    for pfx, name in _NESTABLE.items():
        if comment.startswith(pfx):
            _MARKS_SEEN.add(pfx)
            # only a phase in its own right at top level; nested = transparent
            return name if current in ("header", "phase_a") else current
    for pfx, name in _PHASE_MARKS:
        if comment.startswith(pfx):
            _MARKS_SEEN.add(pfx)
            return name
    return current


# ── decoders ────────────────────────────────────────────────────────────────

def decode_desc(words: list[int]) -> dict:
    """4 hex words MSW-first -> mxe_desc_t fields.

    Packing per gen_l3_vectors.py:143-148, layout per rtl/apex_pkg.sv:38-52.
    """
    v = (words[0] << 96) | (words[1] << 64) | (words[2] << 32) | words[3]
    return {
        "opcode": v & 0xFF,
        "m": (v >> 8) & 0xFFF,
        "k": (v >> 20) & 0xFFF,
        "n": (v >> 32) & 0xFFF,
        "rq_scale": (v >> 44) & 0xFFFF,
        "rq_shift": (v >> 60) & 0x1F,
        "rq_en": (v >> 65) & 1,
        "acc": (v >> 66) & 1,
        "mode_os": (v >> 67) & 1,
    }


def decode_route(r: int) -> dict:
    """ROUTE level word -> rt_* fields (gen_l3_vectors.py:235-236)."""
    return {
        "fsrc": r & 1, "fdst": (r >> 1) & 1, "asrc": (r >> 2) & 1,
        "wsrc": (r >> 3) & 1, "rdst": (r >> 4) & 3, "qsrc": (r >> 6) & 1,
        "kvu": (r >> 7) & 1, "blk": (r >> 8) & 0x7F,
    }


def _parse_op(op: str, f: list[str]) -> dict:
    """Decode one op's arguments. Radix is MIXED: AJ/WJ op/bank/pat and QJOB
    mode are DECIMAL (python str()); everything else is bare lowercase hex.
    """
    if op == "DESC":
        return decode_desc([int(x, 16) for x in f])
    if op == "ROUTE":
        return {"r": int(f[0], 16), **decode_route(int(f[0], 16))}
    if op in ("AJ", "WJ"):
        return {"op": int(f[0]), "bank": int(f[1]), "pat": int(f[2]),
                "rows": int(f[3], 16), "nb": int(f[4], 16),
                "sel": int(f[5], 16)}
    if op == "QJOB":
        return {"mode": int(f[0]), "cols": int(f[1], 16)}
    if op == "SJOB":
        return {"cols": int(f[0], 16)}
    if op == "FJOB":
        return {"rows": int(f[0], 16)}
    if op == "LJOB":
        return {"beats": int(f[0], 16), "lanes": int(f[1], 16)}
    if op in ("KVW", "CSRW"):
        return {"addr": int(f[0], 16), "data": int(f[1], 16)}
    if op in ("KVP", "CSRP", "CSRR"):
        return {"addr": int(f[0], 16), "mask": int(f[1], 16),
                "exp": int(f[2], 16)}
    if op == "CSRN":
        return {"addr": int(f[0], 16), "mask": int(f[1], 16)}
    if op in ("CS", "QS"):
        return {"bits": int(f[0], 16)}
    if op in ("EFS", "ESS"):
        return {"bits": int(f[0], 16), "last": int(f[1])}
    if op == "IMP":
        return {"hi": int(f[0], 16), "lo": int(f[1], 16)}
    return {"raw": f}


# ── manifest ────────────────────────────────────────────────────────────────

def load_manifest(build: Path) -> list[dict]:
    p = Path(build) / "manifest.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())["cases"]


# ── the extractor ───────────────────────────────────────────────────────────

def extract_case(path: Path, params: dict | None = None) -> dict:
    """Project one case .txt to a canonical per-channel trace with provenance."""
    path = Path(path)
    channels: dict[str, list] = {k: [] for k in
                                 sorted(CONTROL | SYNC | DATA | EXPECT)}
    counts: dict[str, int] = {}
    phase_ops: dict[str, int] = {}
    tap_left = 0
    phase = "header"
    D = params.get("D") if params else None
    T = params.get("T") if params else None
    n_comment = n_blank = n_tap_payload = 0
    all_lines = path.read_text().splitlines()

    for ln, line in enumerate(all_lines, start=1):
        # stateful tap payload: <count> bare hex lines carrying no opcode
        if tap_left > 0:
            tap_left -= 1
            n_tap_payload += 1
            continue
        if not line.strip():
            n_blank += 1
            continue
        if line.startswith("//"):
            n_comment += 1
            phase = _phase_of(line, phase)
            continue

        f = line.split()
        op, args = f[0], f[1:]

        if op in TAPS:
            tap_left = int(args[0], 16)
            counts[op] = counts.get(op, 0) + 1
            continue
        if op in MARKERS:
            counts[op] = counts.get(op, 0) + 1
            continue

        counts[op] = counts.get(op, 0) + 1
        phase_ops[phase] = phase_ops.get(phase, 0) + 1

        rec = {"line": ln, "phase": phase, **_parse_op(op, args)}

        # D falls out of phase A's INFO_D readback if the manifest lacked it
        if op == "CSRR" and rec.get("addr") == 0x0C and D is None:
            D = rec["exp"]

        if op == "QS":
            rec["role"] = "s4_fold" if phase in _S4_PHASES else "inject"
        if op in channels:
            channels[op].append(rec)

    # ── provenance: where does each composite's input scale become observable?
    # s_q is published by the ESS that closes the q-inject phase (core_case:570,
    # full_case:652, tip_auto:737). tip_auto asserts r.s_q == r8.s_q (:722), so
    # the single q-inject ESS serves both of its passes.
    ess_q = [e for e in channels["ESS"] if e["phase"] == "q_inject"]
    s_q = ess_q[-1] if ess_q else None

    # s_k[t] / s_v[t] are computed FRESH by seam_feeder_quant on record read
    # (rtl/seam/seam_feeder_quant.sv:12,160) and surface on fs_* as EFS lines.
    efs_by_phase: dict[str, list] = {}
    for e in channels["EFS"]:
        efs_by_phase.setdefault(e["phase"], []).append(e)

    def _resolve(rec, kind):
        """Attach (input scale, producing line, ordering verdict).

        `rec["t"]` is the composite's index WITHIN ITS OWN PHASE — tip_auto
        emits two independent groups (a 4x8 profiling pass and the final
        T-wide attention), so a file-global index would cross-wire them.
        """
        p = {"s_q_bits": None, "s_k_bits": None, "s_v_bits": None,
             "input_line": None, "ordering": "UNRESOLVED"}
        if kind == "cs":
            # CS needs BOTH s_q and s_k[t]; the K-reads are in the SAME phase.
            src = efs_by_phase.get(rec["phase"], [])
            if s_q is not None:
                p["s_q_bits"] = s_q["bits"]
            if rec["t"] < len(src):
                p["s_k_bits"] = src[rec["t"]]["bits"]
                p["input_line"] = max(src[rec["t"]]["line"],
                                      s_q["line"] if s_q else 0)
        elif rec["phase"] == "score":
            # QS(S-4) needs s_v[t]; the V-reads live in the pv phase, which
            # runs a whole phase LATER. pv re-reads all T records BPR times,
            # so the first pass (index t) is the earliest observation.
            src = efs_by_phase.get("pv", [])
            if rec["t"] < len(src):
                p["s_v_bits"] = src[rec["t"]]["bits"]
                p["input_line"] = src[rec["t"]]["line"]
        else:
            # tip_auto's profiling QS folds s_v from the PASS-1 CQ-8 store.
            # Those records are re-read only in the pass-2 pv phase, at a
            # different tier and hence a different scale — the value is never
            # observable on any tap before it is needed. Genuinely unresolved,
            # and NOT to be cross-wired to the pass-2 pv EFS stream.
            p["note"] = "pass-1 profiling scale; never re-observed pre-use"
        if p["input_line"] is not None:
            p["ordering"] = "OK" if p["input_line"] < rec["line"] else "INVERTED"
        return p

    # index each composite within its own phase, then resolve
    for group, kind in ((channels["CS"], "cs"),
                        ([r for r in channels["QS"] if r["role"] == "s4_fold"],
                         "qs")):
        seen: dict[str, int] = {}
        for rec in group:
            rec["t"] = seen.get(rec["phase"], 0)
            seen[rec["phase"]] = rec["t"] + 1
            rec["provenance"] = _resolve(rec, kind)
    s4 = [r for r in channels["QS"] if r["role"] == "s4_fold"]

    # ── scope accounting ────────────────────────────────────────────────────
    def _n(names, pred=None):
        return sum(1 for k in names for e in channels[k]
                   if pred is None or pred(e))

    in_target = lambda e: e["phase"] in WALKER_TARGET_PHASES        # noqa: E731
    walker_owned = _n(CONTROL - {"QS"}, in_target) + \
        sum(1 for e in channels["QS"]
            if e["role"] == "s4_fold" and in_target(e))
    # Completeness: every line of the file is accounted for exactly once. If
    # this ever fails, the extractor is silently dropping or double-counting
    # ops and every number downstream is untrustworthy.
    n_op = sum(counts.values())
    assert n_op + n_comment + n_blank + n_tap_payload == len(all_lines), (
        f"{path.name}: line accounting does not balance — ops {n_op} + "
        f"comments {n_comment} + blanks {n_blank} + tap payload "
        f"{n_tap_payload} != {len(all_lines)} file lines")
    assert tap_left == 0, f"{path.name}: truncated tap payload ({tap_left} short)"

    scope = {
        "file_lines": len(all_lines),
        "op_lines": n_op,
        "comment_lines": n_comment,
        "tap_payload_lines": n_tap_payload,
        "control_all": _n(CONTROL),
        "sync_all": _n(SYNC),
        "data": _n(DATA),
        "qs_inject": sum(1 for e in channels["QS"] if e["role"] == "inject"),
        "qs_s4": len(s4),
        "walker_target_control": walker_owned,
        "walker_target_sync": _n(SYNC, in_target),
        "per_phase_ops": phase_ops,
    }

    ordering = {"OK": 0, "INVERTED": 0, "UNRESOLVED": 0}
    for rec in channels["CS"] + s4:
        ordering[rec["provenance"]["ordering"]] += 1

    return {
        "case": path.stem,
        "params": {"D": D, "T": T,
                   **{k: v for k, v in (params or {}).items()
                      if k in ("build", "kind", "checks", "expect")}},
        "channels": channels,
        "counts": counts,
        "scope": scope,
        "ordering": ordering,
    }


# ── reporting ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build", help="L3 build dir (contains manifest.json, cases/)")
    ap.add_argument("--case", help="single case name (default: all)")
    ap.add_argument("--json", help="write the extracted trace(s) here")
    ap.add_argument("--ordering", action="store_true",
                    help="composite ordering feasibility report only")
    a = ap.parse_args()

    b = Path(a.build)
    man = load_manifest(b)
    if not man:
        print(f"no manifest.json under {b} — run gen_l3_vectors.py first")
        return 2
    if a.case:
        man = [c for c in man if c["name"] == a.case] or [{"name": a.case}]

    out, tot = [], {"OK": 0, "INVERTED": 0, "UNRESOLVED": 0}
    print(f"{'case':<24} {'ctl':>7} {'sync':>6} {'data':>7} "
          f"{'walker':>7} {'CS':>5} {'QS4':>5}  ordering(OK/INV/UNRES)")
    print("-" * 96)
    for c in man:
        tr = extract_case(b / "cases" / f"{c['name']}.txt", params=c)
        out.append(tr)
        s, o = tr["scope"], tr["ordering"]
        for k in tot:
            tot[k] += o[k]
        print(f"{tr['case']:<24} {s['control_all']:>7} {s['sync_all']:>6} "
              f"{s['data']:>7} {s['walker_target_control']:>7} "
              f"{len(tr['channels']['CS']):>5} {s['qs_s4']:>5}  "
              f"{o['OK']:>5}/{o['INVERTED']:>5}/{o['UNRESOLVED']:>5}")

    print("-" * 96)
    # A phase marker that never fires is a silent misclassification waiting to
    # happen (it means gen_l3_vectors.py's comment text moved out from under us).
    unused = [p for p, _ in _PHASE_MARKS if p not in _MARKS_SEEN] + \
             [p for p in _NESTABLE if p not in _MARKS_SEEN]
    if unused and not a.case:
        print(f"WARNING: phase markers that never matched: {unused}")
        print("         (gen_l3_vectors.py comment text may have drifted)")
    n = sum(tot.values())
    print(f"composite words: {n}   emitted AFTER their input is observable: "
          f"{tot['OK']}   INVERTED: {tot['INVERTED']}   UNRESOLVED: "
          f"{tot['UNRESOLVED']}")
    if tot["INVERTED"] or tot["UNRESOLVED"]:
        print()
        print("READING — this measures TAP observability only (fs_*/ss_*):")
        print("  The host emits every composite before its input scale appears")
        print("  on a tap, because the host precomputes scales from the golden.")
        print("  A composite unit fed ONLY by the taps must therefore emit late,")
        print("  which would reorder the stream.")
        print()
        print("  BUT the taps are not the only on-tile source. The KVQ stores a")
        print("  per-record fp16 scale at store time (rtl/kvq/kvq_engine.sv:290,")
        print("  and scale_bank_store :431 for grouped keys). At CQ-8 that stored")
        print("  scale is BIT-IDENTICAL to the feeder's read-time s_k[t]/s_v[t]")
        print("  (verified over the suite tensors + 13k random tokens), so a")
        print("  store-time scale cache lets the walker emit in the host's EXACT")
        print("  order — no reorder allowance needed.")
        print()
        print("  Grouped tiers (CQ-4/CQ-4+) recompute a different read-time scale")
        print("  from the dequantized record; it is determined at store time but")
        print("  is NOT the stored per-record scale, so capturing it needs a")
        print("  feeder-equivalent commit-time amax pass. See B1_WALKER.md §4.")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
