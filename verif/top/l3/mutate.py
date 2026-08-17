#!/usr/bin/env python3
"""mutate.py — Layer-3 mutation gate: prove the L3 suite CATCHES tile-level
integration bugs. Three mutants of rtl/top/apex_top.sv (the RTL itself is
NEVER touched — mutants are patched COPIES under build/):

  M1 swap-stream : the act-stage load mux legs are swapped
                   (feeder codes <-> scale_quant codes) — a classic
                   integration miswire. Caught by value checks (TAPF16 /
                   EFS mismatches) on any full-chain case.
  M2 break-S4    : the ASU probability into the S-4 P-requant is
                   SIGN-EXTENDED instead of zero-extended — breaks the
                   D-021 S-4 fold exactly where p >= 0x8000 (p == 1.0).
                   Caught by adv_T1 (single-token row, p saturates to 1.0).
  M3 CSR-wire    : the tile error-pulse wire into the CSR is cut
                   (.desc_error_i tied 0) — STATUS[1] can never set.
                   Caught by the phase-A illegal-descriptor CSRP (honest
                   timeout fatal).

Usage: mutate.py <mode> <mutant> [args]
  patch <M1|M2|M3> <outdir>   write the patched apex_top.sv copy
  check <M1|M2|M3> <runlog>   assert the run FAILED with the expected
                              signature (exit 0 iff properly caught)

NOTE (B1 stage 4): M1/M2 anchor on the act-stage and squant source muxes.
Those now read the WALKER-MUXED nets (w_rt_act_src / w_rt_squant_src) because
apex_top routes every host control fanout through the walk_en mux. The mutants
are unchanged in what they break; only the anchor text moved.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "rtl/top/apex_top.sv"

MUTS = {
    "M1": {
        "old": "assign as_ld_beat  = w_rt_act_src ? sq_q_beat : fq_out_beat;",
        "new": "assign as_ld_beat  = w_rt_act_src ? fq_out_beat : sq_q_beat;  // MUTANT M1",
        "sig": ["tap f16", "EFS", "ERO"],   # any value-check failure
        "kind": "value",
    },
    "M2": {
        "old": "assign sq_v_data    = w_rt_squant_src ? $signed({16'b0, asu_m_prob})",
        "new": "assign sq_v_data    = w_rt_squant_src ? $signed({{16{asu_m_prob[15]}}, asu_m_prob})  // MUTANT M2",
        "sig": ["ESS", "ERO", "tap"],
        "kind": "value",
    },
    "M3": {
        "old": "    .desc_error_i      (any_err_pulse),",
        "new": "    .desc_error_i      (any_err_pulse & 1'b0),  // MUTANT M3",
        "sig": ["CSRP 04 timeout"],
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
    if "L3 PASS" in txt:
        print(f"{m}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    hit = [s for s in mut["sig"] if s in txt]
    fatal = ("%Fatal" in txt) or ("L3 FAIL" in txt) or ("%Error" in txt)
    if not fatal:
        print(f"{m}: run neither passed nor failed?! Gate FAIL.")
        return 1
    if not hit:
        print(f"{m}: FAILED but without the expected signature "
              f"{mut['sig']} — inspect. Gate FAIL.")
        return 1
    print(f"{m}: CAUGHT ({mut['kind']}; signature {hit})")
    return 0


if __name__ == "__main__":
    mode, m = sys.argv[1], sys.argv[2]
    if mode == "patch":
        sys.exit(patch(m, sys.argv[3]))
    sys.exit(check(m, sys.argv[3]))
