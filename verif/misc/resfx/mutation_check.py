#!/usr/bin/env python3
"""mutation_check.py — prove the resfx checkers catch injected bugs
(house pattern: verif/mxe/w4/mutation_check.py).

Each mutant: copy the RTL into build/mut_<id>/ (rtl/ never modified),
apply ONE exact-string mutation (asserted to hit exactly once), rebuild
the TB against the mutated copy, run with a reduced in-sim count, REQUIRE
a detection (non-zero exit). Builds run strictly sequentially (one
Verilator at a time — the §9.1 machine rule).

Mutants (IB_LAYER.md §4 S2 row, RELOCATED per measurement — the B3 M5
precedent "the mutant list was corrected, not waived"):
  E1 tie-break RNE->away, NORMAL path      (rem >  half -> rem >= half)
  E2 v24 align off-by-one                  (<< (e-1)    -> << e)
  E3 residual zero-sign rule broken        (&&          -> ||)
  (E4 overflow RELOCATED after BOTH sites survived here, each measured
     equivalent AT THIS CALL SITE: the top E>15 check cannot fire
     distinguishably because residual sums cap at E=16 which the
     post-carry path encodes as inf anyway (keep==1024 -> {s,31,0}), and
     the post-carry check is GLOBALLY equivalent by that same encoding
     identity (documented-defensive in the pkg now). The top check IS
     later measured/derived equivalent at EVERY call site — see the pkg
     comment and verif/rope/row/mutation_check.py; both overflow checks
     are documented-defensive, gated on BEHAVIOR (inf saturation vectors)
     rather than by a mutant kill.)
  (E5 subnormal tie->away RELOCATED to verif/rope/row: the subnormal RNE
     path is UNREACHABLE at FRAC=24 — the residual grid is already 2^-24,
     sh == 0, exact. rope's FRAC=38 reaches it; R6 kills it there.)
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

RTL_FILES = ["misc/f16_arith_pkg.sv", "misc/residual_add_fx.sv",
             "misc/residual_add.sv"]

MUTANTS = [
    ("E1_tie_away_normal", "misc/f16_arith_pkg.sv",
     "roundup = (rem > halfb) || ((rem == halfb) && keep[0]);  // RNE, normal path",
     "roundup = (rem >= halfb);  // RNE, normal path"),
    ("E2_align_off_by_one", "misc/f16_arith_pkg.sv",
     "mag = {32'b0, 1'b1, m} << (e - 5'd1);",
     "mag = {32'b0, 1'b1, m} << e;"),
    ("E3_zero_sign", "misc/residual_add_fx.sv",
     "sgn = a[15] && b[15];",
     "sgn = a[15] || b[15];"),
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
        (rtl / "misc").mkdir(parents=True)
        for f in RTL_FILES:
            shutil.copy(REPO / "rtl" / f, rtl / f)
        src = (rtl / mfile).read_text()
        assert src.count(old) == 1, f"{mid}: mutation site not unique/found"
        (rtl / mfile).write_text(src.replace(old, new))

        objdir = mdir / "obj"
        b = run([VERILATOR, "--binary", "-Wall", "--timing", "--assert",
                 "--timescale", "1ns/1ps", "--top-module", "tb_resfx",
                 "-Mdir", str(objdir)]
                + [str(rtl / f) for f in RTL_FILES]
                + [str(HERE / "tb_resfx.sv")])
        if b.returncode != 0:
            print(f"MUTANT {mid}: BUILD FAILED (counts as undetected) rc={b.returncode}")
            print(b.stderr[-800:])
            continue
        r = run([str(objdir / "Vtb_resfx"),
                 f"+vectors={BUILD}/vectors_resfx.txt",
                 "+insim=200000", "+seed=911"], timeout=600)
        if r.returncode != 0:
            first = next((l for l in r.stdout.splitlines()
                          if "MISMATCH" in l or "FAIL" in l), "")
            print(f"MUTANT {mid}: DETECTED (rc={r.returncode}) — {first}")
            ok += 1
        else:
            print(f"MUTANT {mid}: **SURVIVED** — checker hole")
    print(f"RESFX MUTATION GATE: {ok}/{len(MUTANTS)} mutants detected"
          + (" — checkers proven live" if ok == len(MUTANTS) else " — FAIL"))
    sys.exit(0 if ok == len(MUTANTS) else 1)


if __name__ == "__main__":
    main()
