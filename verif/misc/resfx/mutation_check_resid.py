#!/usr/bin/env python3
"""mutation_check_resid.py — apex_residual datapath mutation gate (S5).

House pattern (see mutation_check.py in this dir). Mutants:
  RM1 window shift off-by-one   (e8-112 -> e8-111: every b doubles)
  RM2 inf-shortcut threshold    (>=144 -> >=143: the e8=143 finite-32.0
                                 distinguisher vector must catch it)
  RM3 in-place writeback index off-by-one
  RM4 zero-sign rule broken     (&& -> ||: the +-0 battery catches it)
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
             "top/glue/apex_residual.sv"]

MUTANTS = [
    ("RM1_window_shift", "top/glue/apex_residual.sv",
     "sh_b    = int'({24'b0, b_e8}) - 112;",
     "sh_b    = int'({24'b0, b_e8}) - 111;"),
    ("RM2_inf_threshold", "top/glue/apex_residual.sv",
     "b_inf_sc = (b_e8 >= 8'd144);",
     "b_inf_sc = (b_e8 >= 8'd143);"),
    # RM3 site updated 2026-07-31: R3 renamed the writeback index cnt ->
    # eff_idx (window base) and this pattern went stale — the gate had been
    # failing "site not found" since then (found by the E-1 relint/rerun).
    ("RM3_writeback_index", "top/glue/apex_residual.sv",
     "row[eff_idx[AW-1:0]] <= y_bits;",
     "row[eff_idx[AW-1:0] + AW'(1)] <= y_bits;"),
    ("RM4_zero_sign", "top/glue/apex_residual.sv",
     "y_sign = a_sign && b_sign;",
     "y_sign = a_sign || b_sign;"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    ok = 0
    for mid, mfile, old, new in MUTANTS:
        mdir = BUILD / f"mut_{mid}"
        if mdir.exists():
            shutil.rmtree(mdir)
        rtl = mdir / "rtl"
        for sub in ("misc", "xbr", "top/glue"):
            (rtl / sub).mkdir(parents=True, exist_ok=True)
        for f in RTL_FILES:
            shutil.copy(REPO / "rtl" / f, rtl / f)
        src = (rtl / mfile).read_text()
        assert src.count(old) == 1, f"{mid}: mutation site not unique/found"
        (rtl / mfile).write_text(src.replace(old, new))
        objdir = mdir / "obj"
        b = run([VERILATOR, "--binary", "-Wall", "-Wno-UNUSEDPARAM",
                 "-Wno-UNUSEDSIGNAL", "--timing", "--assert",
                 "--timescale", "1ns/1ps", "--top-module", "tb_resid_unit",
                 "-Mdir", str(objdir)]
                + [str(rtl / f) for f in RTL_FILES]
                + [str(HERE / "tb_resid_unit.sv")])
        if b.returncode != 0:
            print(f"MUTANT {mid}: BUILD FAILED (counts as undetected)")
            print(b.stderr[-400:])
            continue
        try:
            r = run([str(objdir / "Vtb_resid_unit"),
                     f"+vectors={BUILD}/vectors_resid.txt",
                     "+stall_mode=0", "+seed=99"], timeout=300)
            rc, outp = r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            rc, outp = -9, "(timeout)"
        if rc != 0:
            first = next((l for l in outp.splitlines()
                          if "MISMATCH" in l or "FAIL" in l), outp[:60])
            print(f"MUTANT {mid}: DETECTED (rc={rc}) — {first[:90]}")
            ok += 1
        else:
            print(f"MUTANT {mid}: **SURVIVED** — checker hole")
    print(f"RESID MUTATION GATE: {ok}/{len(MUTANTS)} mutants detected"
          + (" — checkers proven live" if ok == len(MUTANTS) else " — FAIL"))
    sys.exit(0 if ok == len(MUTANTS) else 1)


if __name__ == "__main__":
    main()
