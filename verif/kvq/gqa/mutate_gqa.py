#!/usr/bin/env python3
"""mutate_gqa.py — S4b bank-level mutation gate: prove tb_kvq_gqa CATCHES
engine-select routing bugs in apex_kvq_gqa_bank.sv. The RTL is NEVER touched
— mutants are patched COPIES under build/ (the l3/mutate.py discipline:
each mutant must fail WITH ITS SPECIFIC SIGNATURE, never any-nonzero-exit).

  mS1 sel-stuck   : the per-engine select compare is stuck at engine 0
                    (idx==0 broadcast / idx!=0 nobody) — the classic
                    fan-out miswire. Engine 1's very first AXI read can
                    never answer -> hang; caught by the phase-tagged
                    watchdog at phase=info e1.
  mS2 rb-mux-cross: the m_axis DATA mux indexes the NEIGHBOR engine
                    (idx^1) while valid/last still track idx — readback
                    returns the wrong engine's (idle, reset-zero) data.
                    Caught by "hat e*" value mismatches (run completes,
                    final $fatal FAILURES).
  mS3 store-mirror: the s_axis valid fan-out selects the MIRRORED engine
                    ((N_ENG-1)-gs) — the stream routes to an engine whose
                    ready is NOT the one the bank exports, so §5 stream
                    legality breaks at the mis-routed engine. Caught by
                    the per-engine bound SVA pack (kvq_axis_sva
                    ap_s_stable, "[SVA §5] s_axis" at a dut.g_eng[...]
                    scope) — measured 2026-07-26; the B1c "OCC stored"
                    per-engine occupancy checks are the value-class
                    backstop had the stream happened to stay legal.

Usage: mutate_gqa.py <mode> <mutant> [args]
  patch <mS1|mS2|mS3> <outdir>   write the patched apex_kvq_gqa_bank.sv copy
  check <mS1|mS2|mS3> <runlog>   assert the run FAILED with the expected
                                 signature (exit 0 iff properly caught)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "rtl/top/glue/apex_kvq_gqa_bank.sv"

MUTS = {
    "mS1": {
        "old": "      assign sel[gs]        = (idx == ENG_W'(gs));",
        "new": "      assign sel[gs]        = (idx == ENG_W'(0));  // MUTANT mS1",
        "sig": ["phase=info e1"],
        "kind": "hang",
    },
    "mS2": {
        "old": "  assign m_axis_kv_tdata  = e_m_tdata [idx];",
        "new": "  assign m_axis_kv_tdata  = e_m_tdata [idx ^ ENG_W'(1)];  // MUTANT mS2",
        "sig": ["hat e"],
        "kind": "value",
    },
    "mS3": {
        "old": "      assign e_s_tvalid[gs] = s_axis_kv_tvalid && sel[gs];",
        "new": "      assign e_s_tvalid[gs] = s_axis_kv_tvalid && sel[(N_ENG-1)-gs];  // MUTANT mS3",
        "sig": ["[SVA §5] s_axis"],
        "kind": "sva",
    },
}


def patch(m: str, outdir: str) -> int:
    mut = MUTS[m]
    src = SRC.read_text()
    n = src.count(mut["old"])
    assert n == 1, f"{m}: anchor found {n} times (expected exactly 1)"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "apex_kvq_gqa_bank.sv").write_text(src.replace(mut["old"], mut["new"]))
    print(f"{m}: patched copy written to {out}/apex_kvq_gqa_bank.sv")
    return 0


def check(m: str, runlog: str) -> int:
    mut = MUTS[m]
    txt = Path(runlog).read_text()
    if "ALL PASS" in txt:
        print(f"{m}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    hit = [s for s in mut["sig"] if s in txt]
    fatal = ("%Fatal" in txt) or ("FAILURES" in txt) or ("%Error" in txt)
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
