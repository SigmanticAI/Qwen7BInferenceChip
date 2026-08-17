#!/usr/bin/env python3
"""mutate_compose.py — L4 SEGMENT-2 mutation gate: prove tb_l4_compose CATCHES
tile-level front-half integration bugs. Three mutants of rtl/top/apex_top.sv
(the RTL is NEVER touched — mutants are patched COPIES under build/), each at a
DISTINCT integration point of the xa->rmsnorm->feeder->astage front-half, each
caught ONLY on its specific FAIL signature (the l3/mutate.py discipline — an
"any-nonzero-exit counts" gate is a lying gate; the audit found that pattern):

  mC1 rms-data   : the RMSNorm output into the widen is HALVED (>>>1) — every
                   feeder-input magnitude halves, so the C-1 row scale s_h
                   halves. Caught by the EFS check (fs != r.s_h). [value]
  mC2 gamma-wire : the RMSNorm gamma stream is HALVED (>>>1) — a distinct wire
                   from mC1 (the gamma routing, not the norm-output routing), so
                   h and its C-1 scale s_h are wrong. Caught by EFS. [value]
  mC3 feeder-job : the feeder job_valid is gated off (& 1'b0 — the l3 M3 idiom
                   that keeps the source signal read, so no UNUSEDSIGNAL) — the
                   feeder never runs, the input path back-pressures and the
                   front-half hangs. Caught as an honest stall fatal. [hang]

These are SEGMENT-2 (front-half) mutants. The LAYER-path mutants named in
IB_LAYER.md §5 (m-L1 lr_rope_en / m-L2 residual-bank / m-L3 swiglu-index) are
SEGMENT-1's obligation (the o-proj->r1 back-half composition), not reachable by
the front-half-only seg-2 traffic.

Usage: mutate_compose.py <patch|check> <mC1|mC2|mC3> <outdir|runlog>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "rtl/top/apex_top.sv"

MUTS = {
    "mC1": {
        "old": "    .s_y     (rms_m_y),",
        "new": "    .s_y     (rms_m_y >>> 1),  // MUTANT mC1",
        "sig": ["[EFS]", "L4COMPOSE FAIL"],
        "kind": "value",
    },
    "mC2": {
        "old": "    .g_gamma          (xg_gamma),",
        "new": "    .g_gamma          (xg_gamma >>> 1),  // MUTANT mC2",
        "sig": ["[EFS]", "L4COMPOSE FAIL"],
        "kind": "value",
    },
    "mC3": {
        "old": "    .job_valid        (w_fj_valid),",
        "new": "    .job_valid        (w_fj_valid & 1'b0),  // MUTANT mC3",
        "sig": ["stall", "WATCHDOG", "EFS timeout"],
        "kind": "hang",
    },
}


def patch(m: str, outdir: str) -> int:
    mut = MUTS[m]
    src = SRC.read_text()
    n = src.count(mut["old"])
    assert n == 1, f"{m}: anchor found {n} times (expected exactly 1)"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "apex_top.sv").write_text(src.replace(mut["old"], mut["new"]))
    print(f"{m}: patched copy written to {out}/apex_top.sv")
    return 0


def check(m: str, runlog: str) -> int:
    mut = MUTS[m]
    txt = Path(runlog).read_text()
    if "L4COMPOSE PASS" in txt:
        print(f"{m}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    fatal = ("%Fatal" in txt) or ("L4COMPOSE FAIL" in txt) or ("%Error" in txt)
    if not fatal:
        print(f"{m}: run neither passed nor failed?! Gate FAIL.")
        return 1
    hit = [s for s in mut["sig"] if s in txt]
    if not hit:
        print(f"{m}: FAILED but without the expected signature {mut['sig']} "
              f"— inspect. Gate FAIL.")
        return 1
    print(f"{m}: CAUGHT ({mut['kind']}; signature {hit})")
    return 0


if __name__ == "__main__":
    mode, m = sys.argv[1], sys.argv[2]
    if mode == "patch":
        sys.exit(patch(m, sys.argv[3]))
    sys.exit(check(m, sys.argv[3]))
