#!/usr/bin/env python3
"""mutation_check.py — prove the ASU checkers CATCH injected RTL bugs
(house pattern: verif/mxe RESULT.md §6, made a repeatable build gate).

Each mutant: copy the RTL into a scratch dir under build/ (rtl/ is NEVER
modified), apply one exact-string mutation (asserted to hit exactly once),
rebuild the scoreboard TB against the mutated copy, run it, and REQUIRE a
detection (non-zero exit / FAIL / assertion). An undetected mutant fails
this gate — 'make all' therefore proves checker credibility, not just PASS.

Mutants:
  M1 data corruption      : asu_softmax output bit-4 XOR at the skid input
                            -> scoreboard bit-exact compare must fail
  M2 ASU contract (D-014) : exp LUT ENTRY OFF-BY-ONE (idx+1) — the
                            block-specific contract violation
  M3 C-2 rounding         : rmsnorm RNE -> round-half-up — detectable ONLY
                            by the engineered even-parity tie lanes
  M4 D-006 protocol       : softmax done fires without post-skid drain ->
                            TB D-006 monitor / SVA under backpressure storm
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent.parent.parent
BUILD = HERE / "build"
# Resolved via PATH (override: VERILATOR=/path/to/verilator). The old
# hardcoded /opt/homebrew path made this script macOS-only.
VERILATOR = os.environ.get("VERILATOR", "verilator")

RTL_SM = ["rtl/apex_pkg.sv", "rtl/xbr/stream_skid.sv",
          "rtl/asu/asu_exp_lut.sv", "rtl/asu/asu_softmax.sv"]
RTL_RMS = ["rtl/apex_pkg.sv", "rtl/xbr/stream_skid.sv",
           "rtl/asu/rsqrt.sv", "rtl/asu/asu_rmsnorm.sv"]

MUTANTS = [
    dict(
        name="M1_sm_data_corrupt",
        rtl=RTL_SM, tb="tb_asu_softmax_sb.sv", top="tb_asu_softmax_sb",
        mut_file="rtl/asu/asu_softmax.sv",
        old=".s_data  ({quot, o_last}),",
        new=".s_data  ({quot ^ 16'h0010, o_last}),",
        plusargs=["+vectors=build/vectors_sm_directed.txt",
                  "+bp_mode=1", "+stall_mode=1", "+seed=11"],
    ),
    dict(
        name="M2_lut_entry_off_by_one",
        rtl=RTL_SM, tb="tb_asu_softmax_sb.sv", top="tb_asu_softmax_sb",
        mut_file="rtl/asu/asu_exp_lut.sv",
        old="idx  = off[14] ? 8'hFF   : off[13:6];",
        new="idx  = off[14] ? 8'hFF   : (off[13:6] + 8'd1);",
        plusargs=["+vectors=build/vectors_sm_directed.txt",
                  "+bp_mode=1", "+stall_mode=1", "+seed=11"],
    ),
    dict(
        name="M3_rms_rne_to_half_up",
        rtl=RTL_RMS, tb="tb_asu_rmsnorm_sb.sv", top="tb_asu_rmsnorm_sb",
        mut_file="rtl/asu/asu_rmsnorm.sv",
        old="assign rnd_up = (rem_v > 18'h20000) || "
            "((rem_v == 18'h20000) && y_tr[0]);",
        new="assign rnd_up = (rem_v >= 18'h20000);",
        plusargs=["+vectors=build/vectors_rms_directed.txt",
                  "+bp_mode=1", "+stall_mode=1", "+g_mode=0", "+seed=21"],
    ),
    dict(
        name="M4_sm_done_pre_drain",
        rtl=RTL_SM, tb="tb_asu_softmax_sb.sv", top="tb_asu_softmax_sb",
        mut_file="rtl/asu/asu_softmax.sv",
        old="        ST_DRAIN: begin                   // D-006: wait for "
            "post-skid accepts\n          if (out_cnt == n_len) begin",
        new="        ST_DRAIN: begin                   // MUTANT: no wait\n"
            "          if (1'b1) begin",
        plusargs=["+vectors=build/vectors_sm_directed.txt",
                  "+bp_mode=2", "+stall_mode=0", "+seed=14"],
    ),
]


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=1200)
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
                       "-I" + str(REPO / "verif/common"),
                       "+incdir+" + str(REPO / "rtl/asu"),
                       "--top-module", m["top"], "-Mdir", str(obj)]
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
