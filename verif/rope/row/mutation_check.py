#!/usr/bin/env python3
"""mutation_check.py — rope_row / rope_pair_fx / f16_arith_pkg mutation gate
(house pattern: verif/mxe/w4/mutation_check.py; RTL never modified, one
exact-string mutation per scratch copy, detection REQUIRED).

Mutants (IB_LAYER.md §4 S2 row + the two relocations from verif/misc/resfx,
documented there):
  R1 pair-order swap        rope_row emits y_i in the hi pass and vice versa
  R2 cos/sin table swap     rope_pair_fx cos interp reads the SIN base table
  R3 phase index off-by-one ph_addr = pair_j + 1
  R4 RNE->truncate          pkg normal-path roundup forced 0 (kill needs the
                            engineered tie_n rows; random rows also hit)
  R5 x_buf hi-read off-by-one   pair operand b reads x[j+half-1]
  R6 subnormal tie->away    pkg subnormal-path RNE (RELOCATED from resfx —
                            unreachable at FRAC=24; rope's FRAC=38 reaches
                            it; killed by the engineered tie_s rows)
  (R7 overflow top-check: EQUIVALENT MUTANT at every existing call site
     — measured here and at resfx, derived for swiglu (its products
     multiply two >=2^10 significands, so the sh<=0 overflow flow is
     unreachable everywhere; the post-carry check + carry-encoding
     identity cover every sh>0 overflow). Both overflow checks stay in
     the pkg as documented-defensive (B3-M5 class, "no port-level TB can
     kill it and none should be expected to"); rope's sat rows and the
     swiglu ovfl vectors gate saturation BEHAVIOR instead.)
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
             "rope/rope_pair_fx.sv", "rope/rope_row.sv", "rope/rope.sv"]

MUTANTS = [
    ("R1_pair_swap", "rope/rope_row.sv",
     "o_vec   = {(k == CNT_W'(D - 1)), (sel_hi ? y_ih : y_i)};",
     "o_vec   = {(k == CNT_W'(D - 1)), (sel_hi ? y_i : y_ih)};"),
    ("R2_table_swap", "rope/rope_pair_fx.sv",
     "c_val = $signed({ROPE_COS_BASE[idx][15], ROPE_COS_BASE[idx]})",
     "c_val = $signed({ROPE_SIN_BASE[idx][15], ROPE_SIN_BASE[idx]})"),
    ("R3_phase_off_by_one", "rope/rope_row.sv",
     "assign ph_addr = pair_j;",
     "assign ph_addr = pair_j + PAIR_W'(1);"),
    ("R4_rne_truncate_normal", "misc/f16_arith_pkg.sv",
     "roundup = (rem > halfb) || ((rem == halfb) && keep[0]);  // RNE, normal path",
     "roundup = 1'b0;  // RNE, normal path"),
    ("R5_xbuf_off_by_one", "rope/rope_row.sv",
     "xb_q <= x_buf[{1'b0, pair_j} + $clog2(D)'(HALF)];",
     "xb_q <= x_buf[{1'b0, pair_j} + $clog2(D)'(HALF - 1)];"),
    ("R6_tie_away_subnormal", "misc/f16_arith_pkg.sv",
     "roundup = (rem > halfb) || ((rem == halfb) && keep[0]);  // RNE, subnormal path",
     "roundup = (rem >= halfb);  // RNE, subnormal path"),
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
        for sub in ("misc", "xbr", "rope"):
            (rtl / sub).mkdir(parents=True, exist_ok=True)
        for f in RTL_FILES:
            shutil.copy(REPO / "rtl" / f, rtl / f)
        shutil.copy(REPO / "rtl" / "rope" / "rope_lut_tables.svh",
                    rtl / "rope" / "rope_lut_tables.svh")
        src = (rtl / mfile).read_text()
        assert src.count(old) == 1, f"{mid}: mutation site not unique/found"
        (rtl / mfile).write_text(src.replace(old, new))

        objdir = mdir / "obj"
        # mutants may orphan signals/params (e.g. the table swap) — the
        # gate proves DETECTION, not mutant lint hygiene
        b = run([VERILATOR, "--binary", "-Wall", "-Wno-UNUSEDPARAM",
                 "-Wno-UNUSEDSIGNAL", "--timing", "--assert",
                 "--timescale", "1ns/1ps", f"-I{rtl}/rope",
                 "--top-module", "tb_rope_row", "-Mdir", str(objdir)]
                + [str(rtl / f) for f in RTL_FILES]
                + [str(HERE / "tb_rope_row.sv")])
        if b.returncode != 0:
            print(f"MUTANT {mid}: BUILD FAILED (counts as undetected)")
            print(b.stderr[-500:])
            continue
        r = run([str(objdir / "Vtb_rope_row"),
                 f"+vectors={BUILD}/vectors_rope_d64.txt",
                 "+bp_mode=1", "+stall_mode=0", "+seed=707"], timeout=900)
        if r.returncode != 0:
            first = next((l for l in r.stdout.splitlines()
                          if "MISMATCH" in l or "FAIL" in l), "")
            print(f"MUTANT {mid}: DETECTED (rc={r.returncode}) — {first[:90]}")
            ok += 1
        else:
            print(f"MUTANT {mid}: **SURVIVED** — checker hole")
    print(f"ROPEROW MUTATION GATE: {ok}/{len(MUTANTS)} mutants detected"
          + (" — checkers proven live" if ok == len(MUTANTS) else " — FAIL"))
    sys.exit(0 if ok == len(MUTANTS) else 1)


if __name__ == "__main__":
    main()
