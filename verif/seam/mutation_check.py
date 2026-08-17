#!/usr/bin/env python3
"""mutation_check.py — prove the D-021 seam checkers CATCH injected RTL bugs
(house pattern: verif/asu/sb/mutation_check.py, a repeatable build gate).

Each mutant: copy the RTL into a scratch dir under build/ (rtl/ is NEVER
modified), apply one exact-string mutation (asserted to hit exactly once),
rebuild the scoreboard TB against the mutated copy, run it, and REQUIRE a
detection (non-zero exit / FAIL / assertion). An undetected mutant fails
this gate — 'make all' therefore proves checker credibility, not just PASS.

Mutants:
  M1 C-1 rounding        : feeder code divide RNE -> round-half-up —
                           detectable ONLY by the engineered s*(k+0.5)
                           code-tie rows (even-quotient ties)
  M2 C-1 EPS floor       : feeder EPS-floor threshold off by one exponent —
                           detectable only by the EPS-floor directed rows
  M3 S-3 rounding (C-2)  : score dequant final shift RNE -> truncation —
                           broad numeric corruption, scoreboard catches
  M4 D-006 protocol      : score dequant done fires without waiting for
                           post-skid drain -> seam_job_sva / TB monitors
                           under backpressure storm
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent.parent
BUILD = HERE / "build"
# Resolved via PATH (override: VERILATOR=/path/to/verilator). The old
# hardcoded /opt/homebrew path made this script macOS-only.
VERILATOR = os.environ.get("VERILATOR", "verilator")

RTL_FQ = ["rtl/apex_pkg.sv", "rtl/xbr/stream_skid.sv",
          "rtl/seam/seam_feeder_quant.sv"]
RTL_SD = ["rtl/apex_pkg.sv", "rtl/xbr/stream_skid.sv",
          "rtl/seam/seam_score_dequant.sv"]

MUTANTS = [
    dict(
        name="M1_fq_code_tie_half_up",
        rtl=RTL_FQ, tb="tb_seam_feeder_sb.sv", top="tb_seam_feeder_sb",
        gparams=["-GD_CFG=64"],
        mut_file="rtl/seam/seam_feeder_quant.sv",
        old="    inc = (r2 > d2) || ((r2 == d2) && qd[0]);    "
            "// round half to even (C-2)",
        new="    inc = (r2 >= d2);    // MUTANT: round half up",
        plusargs=["+vectors=build/vectors_fq_directed_d64.txt",
                  "+bp_mode=1", "+stall_mode=1", "+seed=31"],
    ),
    dict(
        name="M2_fq_eps_floor_off_by_one",
        rtl=RTL_FQ, tb="tb_seam_feeder_sb.sv", top="tb_seam_feeder_sb",
        gparams=["-GD_CFG=64"],
        mut_file="rtl/seam/seam_feeder_quant.sv",
        old="    if (es < -14) return 16'h0400;         "
            "// exact quotient < 2^-14 -> EPS (P2)",
        new="    if (es < -15) return 16'h0400;         // MUTANT: floor off",
        plusargs=["+vectors=build/vectors_fq_directed_d64.txt",
                  "+bp_mode=1", "+stall_mode=1", "+seed=32"],
    ),
    dict(
        name="M3_sd_rne_to_truncation",
        rtl=RTL_SD, tb="tb_seam_score_sb.sv", top="tb_seam_score_sb",
        gparams=[],
        mut_file="rtl/seam/seam_score_dequant.sv",
        old="        inc  = (rem > half) || ((rem == half) && fl[0]);   "
            "// half to even",
        # keep rem/half/fl referenced so the mutant still builds under -Wall
        # (a bare `inc = 1'b0` dies at lint, not at the checkers)
        new="        inc  = ((rem > half) || ((rem == half) && fl[0])) && 1'b0;"
            "   // MUTANT: truncation",
        plusargs=["+vectors=build/vectors_sd_directed.txt",
                  "+bp_mode=1", "+stall_mode=1", "+seed=33"],
    ),
    dict(
        name="M4_sd_done_pre_drain",
        rtl=RTL_SD, tb="tb_seam_score_sb.sv", top="tb_seam_score_sb",
        gparams=[],
        mut_file="rtl/seam/seam_score_dequant.sv",
        old="        ST_WAIT: begin                       "
            "// D-006: post-skid acceptance\n"
            "          if (score_acc == CNT_W'(cols_q)) begin",
        new="        ST_WAIT: begin                       // MUTANT: no wait\n"
            "          if (1'b1) begin",
        plusargs=["+vectors=build/vectors_sd_storm.txt",
                  "+bp_mode=2", "+stall_mode=0", "+seed=34"],
    ),
]


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=2400)
    return p.returncode, p.stdout


def main() -> int:
    detected = 0
    for m in MUTANTS:
        name = m["name"]
        scratch = BUILD / f"mut_{name}"
        if scratch.exists():
            shutil.rmtree(scratch)

        # scratch-copy the RTL (preserving relative paths so the per-file
        # lint waivers still match), apply the mutation to the target file
        srcs = []
        for rel in m["rtl"]:
            dst = scratch / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = (REPO / rel).read_text()
            if rel == m["mut_file"]:
                assert text.count(m["old"]) == 1, \
                    f"{name}: mutation site not found exactly once in {rel}"
                text = text.replace(m["old"], m["new"])
            dst.write_text(text)
            srcs.append(str(dst))
        # sanity: mutated copy differs from the original
        assert (scratch / m["mut_file"]).read_text() \
            != (REPO / m["mut_file"]).read_text(), f"{name}: no-op mutation"

        obj = scratch / "obj"
        rc, out = run([VERILATOR, "--binary", "-Wall", "--timing", "--assert",
                       "--timescale", "1ns/1ps", "lint_waivers.vlt",
                       "-I" + str(REPO / "verif/common"), "-I" + str(HERE)]
                      + m["gparams"]
                      + ["--top-module", m["top"], "-Mdir", str(obj)]
                      + srcs + [m["tb"]], HERE)
        if rc != 0:
            print(f"MUTANT {name}: BUILD FAILED (rc={rc}) — cannot evaluate")
            print(out[-2000:])
            return 1

        rc, out = run([str(obj / ("V" + m["top"]))] + m["plusargs"], HERE)
        (scratch / "run.log").write_text(out)
        clean_pass = (rc == 0) and ("TB PASS" in out) \
            and ("FAIL" not in out) and ("Assertion failed" not in out)
        if clean_pass:
            print(f"MUTANT {name}: *** NOT DETECTED *** (checkers are blind)")
            return 1
        first_fail = next((ln for ln in out.splitlines()
                           if "FAIL" in ln or "Assertion failed" in ln
                           or "%Error" in ln), "(nonzero exit)")
        print(f"MUTANT {name}: DETECTED (rc={rc}) — {first_fail.strip()}")
        detected += 1

    print(f"MUTATION GATE: {detected}/{len(MUTANTS)} mutants detected — "
          "checkers proven live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
