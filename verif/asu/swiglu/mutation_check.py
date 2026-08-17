#!/usr/bin/env python3
"""mutation_check.py — apex_layer_deq + asu_swiglu mutation gate (IB-LAYER
S3; house pattern verif/mxe/w4). RTL never modified; one exact-string
mutation per scratch copy; detection REQUIRED. TIMEOUT/WATCHDOG = KILL
(the w4 precedent): a bypassed refusal that accepts an undriven job hangs
the TB, and the watchdog $fatal / subprocess timeout counts as detection.

Mutants (IB_LAYER.md §4 S3 row, as landed):
  D1 deq grade-check bypass        ungraded composite accepted -> hang/kill
  D2 deq exact-or-refused bypass   inexact products emitted -> unexpected
                                   outputs + error-accounting mismatch
  D3 deq exponent rebias off-by-one    every output x2
  S1 swiglu Q5.10 RNE -> floor     killed by the constructed ties (+ mass)
  S2 swiglu clamp value 8192->8191 killed by the exact-clamp battery
  S3 swiglu up-composite reads comp_g  killed wherever cg != cu
  S4 swiglu product exponent off-by-one  every product x2
  S5 swiglu grade-check bypass     illegal-composite job accepted -> hang
  S6 pkg normal-path RNE tie->away (shared narrower, killed HERE by the
     constructed product ties — independent of the rope-suite kill)
  (The overflow top-check mutant is an EQUIVALENT MUTANT at every existing
   call site — see the pkg comment; the OVFL vectors gate saturation
   BEHAVIOR in the main runs instead.)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUILD = HERE / "build"
# Resolved via PATH (override: VERILATOR=/path/to/verilator). The old
# hardcoded /opt/homebrew path made this script macOS-only.
VERILATOR = os.environ.get("VERILATOR", "verilator")

RTL_FILES = ["misc/f16_arith_pkg.sv", "xbr/stream_skid.sv",
             "top/glue/apex_layer_deq.sv", "asu/asu_silu.sv",
             "asu/asu_swiglu.sv"]

MUTANTS = [
    ("D1_grade_bypass", "deq", "top/glue/apex_layer_deq.sv",
     "&& (c_m[12:0] == 13'h0);          // fp16-grade (C2)",
     "&& 1'b1;          // fp16-grade (C2) BYPASSED"),
    ("D2_exact_bypass", "deq", "top/glue/apex_layer_deq.sv",
     "exact_ok = (prod == '0)",
     "exact_ok = 1'b1 || (prod == '0)"),
    ("D3_rebias_off_by_one", "deq", "top/glue/apex_layer_deq.sv",
     "biased = 8'((p + int'({24'b0, e8_q})) - 10);",
     "biased = 8'((p + int'({24'b0, e8_q})) - 9);"),
    ("S1_q510_rne_floor", "swg", "asu/asu_swiglu.sv",
     "g_shifted = g_shifted + {56'b0, g_round};",
     "g_shifted = g_shifted + 57'd0;"),
    ("S2_clamp_value", "swg", "asu/asu_swiglu.sv",
     "? 14'd8192\n                                     : g_q14_of(g_shifted);  // clamp, rne path",
     "? 14'd8191\n                                     : g_q14_of(g_shifted);  // clamp, rne path"),
    ("S3_wrong_composite", "swg", "asu/asu_swiglu.sv",
     "ku_q   <= 11'(({1'b1, jb_comp_u[22:0]} >> 13));",
     "ku_q   <= 11'(({1'b1, jb_comp_g[22:0]} >> 13));"),
    ("S4_pexp_off_by_one", "swg", "asu/asu_swiglu.sv",
     "p_exp  = (int'({24'b0, eu_q}) - 137) + f16_exp(s_f16[14:10]);",
     "p_exp  = (int'({24'b0, eu_q}) - 136) + f16_exp(s_f16[14:10]);"),
    ("S5_grade_bypass", "swg", "asu/asu_swiglu.sv",
     "return (c[31] == 1'b0) && (c[30:23] != 8'h00) && (c[30:23] != 8'hFF)\n          && (c[12:0] == 13'h0);",
     "return 1'b1;"),
    ("S6_tie_away_normal", "swg", "misc/f16_arith_pkg.sv",
     "roundup = (rem > halfb) || ((rem == halfb) && keep[0]);  // RNE, normal path",
     "roundup = (rem >= halfb);  // RNE, normal path"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    ok = 0
    for mid, which, mfile, old, new in MUTANTS:
        mdir = BUILD / f"mut_{mid}"
        if mdir.exists():
            shutil.rmtree(mdir)
        rtl = mdir / "rtl"
        for sub in ("misc", "xbr", "top/glue", "asu"):
            (rtl / sub).mkdir(parents=True, exist_ok=True)
        for f in RTL_FILES:
            shutil.copy(REPO / "rtl" / f, rtl / f)
        shutil.copy(REPO / "rtl/asu/silu_lut_tables.svh",
                    rtl / "asu/silu_lut_tables.svh")
        src = (rtl / mfile).read_text()
        assert src.count(old) == 1, f"{mid}: mutation site not unique/found"
        (rtl / mfile).write_text(src.replace(old, new))

        if which == "deq":
            top, tb, vec = "tb_layer_deq", "tb_layer_deq.sv", "vectors_deq.txt"
            files = ["misc/f16_arith_pkg.sv", "xbr/stream_skid.sv",
                     "top/glue/apex_layer_deq.sv"]
        else:
            top, tb, vec = "tb_swiglu", "tb_swiglu.sv", "vectors_swiglu.txt"
            files = ["misc/f16_arith_pkg.sv", "xbr/stream_skid.sv",
                     "asu/asu_silu.sv", "asu/asu_swiglu.sv"]
        objdir = mdir / "obj"
        b = run([VERILATOR, "--binary", "-Wall", "-Wno-UNUSEDPARAM",
                 "-Wno-UNUSEDSIGNAL", "--timing", "--assert",
                 "--timescale", "1ns/1ps", f"-I{rtl}/asu",
                 "--top-module", top, "-Mdir", str(objdir)]
                + [str(rtl / f) for f in files] + [str(HERE / tb)])
        if b.returncode != 0:
            print(f"MUTANT {mid}: BUILD FAILED (counts as undetected)")
            print(b.stderr[-500:])
            continue
        try:
            r = run([str(objdir / f"V{top}"), f"+vectors={BUILD}/{vec}",
                     "+bp_mode=1", "+stall_mode=0", "+seed=606"], timeout=300)
            rc, out = r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            rc, out = -9, "(subprocess timeout)"
        if rc != 0:
            first = next((l for l in out.splitlines()
                          if "MISMATCH" in l or "FAIL" in l), out[:60])
            print(f"MUTANT {mid}: DETECTED (rc={rc}) — {first[:90]}")
            ok += 1
        else:
            print(f"MUTANT {mid}: **SURVIVED** — checker hole")
    print(f"SWIGLU/DEQ MUTATION GATE: {ok}/{len(MUTANTS)} mutants detected"
          + (" — checkers proven live" if ok == len(MUTANTS) else " — FAIL"))
    sys.exit(0 if ok == len(MUTANTS) else 1)


if __name__ == "__main__":
    main()
