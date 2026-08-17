#!/usr/bin/env python3
"""mutate_wcomp.py — F6(i) bank-level mutation gate: prove tb_wcomp_bank
CATCHES routing bugs in apex_wcomp_bank.sv. The RTL is NEVER touched —
mutants are patched COPIES under build/ (the l3/mutate.py discipline: every
mutant must fail WITH ITS SPECIFIC SIGNATURE, never any-nonzero-exit).

  mW1 sel-zero    : the clamped index is forced to instance 0 for every
                    IN-RANGE select, i.e. every port routes to one cache —
                    the "is the select actually live?" mutant (written so
                    eng_sel is still READ, or -Wall would kill the mutant at
                    compile time instead of the TB killing it at run time).
                    Every group then answers with the last writer's scales;
                    caught on the phase-A composite mismatch.
  mW2 snoop-bcast : the store-time snoop fan-out ignores the select, so every
                    instance harvests every group's record. This is exactly
                    the F6 defect reproduced inside the bank (one shared
                    cache), and it reproduces its signature: the LAST group
                    written at each record survives, the earlier groups do
                    not. Caught on the phase-A composite mismatch.
  mW3 flush-route : snp_flush is routed by the select instead of broadcast,
                    so a partial record left in a DESELECTED instance
                    survives the D-020 abort and mis-frames the next record.
                    Caught on the phase-F err_frame check.
  mW4 err-noor    : err_stale is taken from instance 0 instead of OR-reduced,
                    so a miss raised by any other instance is LOST before the
                    WALK sticky can see it. Caught on the phase-S OR-reduce
                    check (the run still completes — a silent-loss mutant).

Usage: mutate_wcomp.py <mode> <mutant> [args]
  patch <mW1|mW2|mW3|mW4> <outdir>   write the patched apex_wcomp_bank.sv copy
  check <mW1|mW2|mW3|mW4> <runlog>   assert the run FAILED with the expected
                                     signature (exit 0 iff properly caught)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "rtl/top/glue/apex_wcomp_bank.sv"

MUTS = {
    "mW1": {
        "old": "  assign idx = (32'(eng_sel) >= N_ENG) ? '0 : eng_sel;",
        "new": "  assign idx = (32'(eng_sel) >= N_ENG) ? eng_sel : '0;  // MUTANT mW1",
        "sig": ["FAIL A cs g=1", "FAIL A cs g=2"],
        "kind": "routing",
    },
    "mW2": {
        "old": "      assign e_snp_valid[gs] = snp_valid && sel[gs];",
        "new": "      assign e_snp_valid[gs] = snp_valid;  // MUTANT mW2",
        "sig": ["FAIL A cs g=0", "FAIL A qs g=0"],
        "kind": "isolation",
    },
    "mW3": {
        "old": "        .snp_flush (snp_flush),          // broadcast: tile-wide abort",
        "new": "        .snp_flush (snp_flush && sel[gi]),  // MUTANT mW3",
        "sig": ["FAIL F: err_frame after broadcast flush"],
        "kind": "flush",
    },
    "mW4": {
        "old": "      err_stale |= e_err_stale[i];",
        "new": "      err_stale = e_err_stale[0];  // MUTANT mW4",
        "sig": ["FAIL S: no err_stale from instance"],
        "kind": "error-loss",
    },
}


def patch(m: str, outdir: str) -> int:
    mut = MUTS[m]
    src = SRC.read_text()
    n = src.count(mut["old"])
    assert n == 1, f"{m}: anchor found {n} times (expected exactly 1)"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "apex_wcomp_bank.sv").write_text(src.replace(mut["old"], mut["new"]))
    print(f"{m}: patched copy written to {out}/apex_wcomp_bank.sv")
    return 0


def check(m: str, runlog: str) -> int:
    mut = MUTS[m]
    txt = Path(runlog).read_text()
    if "WCOMP BANK: ALL PASS" in txt:
        print(f"{m}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    hit = [s for s in mut["sig"] if s in txt]
    failed = ("WCOMP BANK: FAIL" in txt) or ("%Error" in txt) or ("%Fatal" in txt)
    if not failed:
        print(f"{m}: run neither passed nor failed?! Gate FAIL.")
        return 1
    if not hit:
        print(f"{m}: FAILED but without the expected signature "
              f"{mut['sig']} — inspect. Gate FAIL.")
        return 1
    print(f"{m}: CAUGHT ({mut['kind']}; signature {hit})")
    return 0


if __name__ == "__main__":
    mode, mut_name = sys.argv[1], sys.argv[2]
    if mode == "patch":
        sys.exit(patch(mut_name, sys.argv[3]))
    sys.exit(check(mut_name, sys.argv[3]))
