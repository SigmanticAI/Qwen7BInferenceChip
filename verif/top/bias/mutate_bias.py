#!/usr/bin/env python3
"""mutate_bias.py — IC-BIAS (gap B) mutation gate: prove tb_bias_tile CATCHES
projection-bias bugs. Four mutants at FOUR DISTINCT integration points, each
caught ONLY on its own FAIL signature (the l3/mutate.py discipline — an
"any-nonzero-exit counts" gate is a lying gate; the audit found that pattern).
The repo RTL is NEVER edited: mutants are patched COPIES under build/.

  mB1 bias-addend  rtl/top/glue/apex_proj_bias.sv — the bias term is dropped
                   from the exact sum, i.e. the block degenerates to plain
                   S-2 narrowing. Caught in arm A1-K on the FIRST biased K
                   beat. [value]  signature: "[tap f16][A1-K]"
  mB2 window-guard rtl/top/glue/apex_proj_bias.sv — the W1 alignment guard is
                   forced true, so an out-of-window element is silently
                   ROUNDED instead of refused. Caught in arm A3: beats leave
                   a job that must emit none, and LAYER_STATUS never reports
                   BIAS_WINDOW. [refusal-bypass]
                   signature: "[tap f16][A3]" / "[CSRR 80]"
  mB3 route-level  rtl/top/apex_top.sv — the per-job selection ignores
                   l_bias_en (stuck selected), so a bias-DISABLED job is
                   still biased. Caught in arm A2, which pins the unbiased
                   narrowing. [route]  signature: "[tap f16][A2]"
  mB4 sideband-fork rtl/top/apex_top.sv — the composite fork feeds the WRONG
                   block, so the bias unit never leaves its sideband load.
                   Caught as an honest stall. [hang]
                   signature: "stall" / "WATCHDOG" / "CSRP"

Usage: mutate_bias.py patch <mB*> <outdir>
       mutate_bias.py check <mB*> <runlog>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PB = REPO / "rtl/top/glue/apex_proj_bias.sv"
TOP = REPO / "rtl/top/apex_top.sv"

MUTS = {
    "mB1": {
        "file": PB,
        "old": "    bfix  = okw ? (57'(bmagv) << sb) : 57'd0;",
        # `& 57'd0` (not a bare 0) so bmagv stays READ — the l3 M3 idiom that
        # keeps the mutant build -Wall clean instead of dying on UNUSEDSIGNAL
        "new": "    bfix  = okw ? ((57'(bmagv) << sb) & 57'd0) : 57'd0;  // MUTANT mB1",
        "sig": ["[tap f16][A1-K]"],
        "kind": "value",
    },
    "mB2": {
        "file": PB,
        "old": "    okw   = (sx <= PB_SX_MAX) && (sb <= PB_SB_MAX);            // W1",
        "new": "    okw   = (sx <= PB_SX_MAX) || (sb <= PB_SB_MAX);  // MUTANT mB2",
        "sig": ["[tap f16][A3]", "[CSRR 80]"],
        "kind": "refusal-bypass",
    },
    "mB3": {
        "file": TOP,
        "old": "  wire pb_take = l_bias_en_q && !w_qj_mode;   // MODE_F16 jobs only",
        "new": "  wire pb_take = !w_qj_mode;  // MUTANT mB3: l_bias_en ignored",
        "sig": ["[tap f16][A2]"],
        "kind": "route",
    },
    "mB4": {
        "file": TOP,
        "old": "  assign pb_cs_v    = w_qs_valid &&  pb_sel_q;",
        "new": "  assign pb_cs_v    = w_qs_valid && !pb_sel_q;  // MUTANT mB4",
        "sig": ["stall", "WATCHDOG", "CSRP"],
        "kind": "hang",
    },
}


def patch(m: str, outdir: str) -> int:
    mut = MUTS[m]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for src in (PB, TOP):
        txt = src.read_text()
        if src == mut["file"]:
            n = txt.count(mut["old"])
            assert n == 1, f"{m}: anchor found {n} times in {src.name} (want 1)"
            txt = txt.replace(mut["old"], mut["new"])
            print(f"{m}: patched copy of {src.name} written to {out}")
        else:
            shutil.copyfile(src, out / src.name)
            continue
        (out / src.name).write_text(txt)
    return 0


def check(m: str, runlog: str) -> int:
    mut = MUTS[m]
    txt = Path(runlog).read_text()
    if "BIASTILE PASS" in txt:
        print(f"{m}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    fatal = ("%Fatal" in txt) or ("BIASTILE FAIL" in txt) or ("%Error" in txt)
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
    mode, name = sys.argv[1], sys.argv[2]
    if mode == "patch":
        raise SystemExit(patch(name, sys.argv[3]))
    raise SystemExit(check(name, sys.argv[3]))
