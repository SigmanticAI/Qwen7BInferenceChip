#!/usr/bin/env python3
"""mutation_check_wide.py — prove the WIDE-D RMSNorm checkers catch injected
RTL bugs (house pattern: verif/asu/sb/mutation_check.py; contract
docs/design/WIDE_RMSNORM.md §4).

Each mutant: copy the RTL into a scratch dir under build/ (rtl/ is NEVER
modified), apply one exact-string mutation (asserted to hit exactly once),
rebuild the wide TB against the mutated copy, run it, and REQUIRE detection.

Mutants:
  MW1 C-2 rounding    : rmsnorm RNE -> round-half-up on the SHARED emission
                        line (M3 reuse) — caught only by the engineered
                        even/odd tie lanes in the wide directed vectors
  MW2 μ off-by-one    : MU[k]+1 — caught by the engineered μ-divisor-boundary
                        D=3584 row (den flips m−1 -> m => dbg_norm mismatch)
  MW3 μ shift         : >>(S+16) -> >>(S+15) — mean2 doubles, any wide row
  MW4 SUM_W truncation: SUM_W formula -> 27 — caught ONLY by the all-(−128)
                        D=8192 corner (sum2 == 2^27 wraps to 0), RMS_D_MAX=
                        8192 elaboration
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

RTL_RMS = ["rtl/apex_pkg.sv", "rtl/xbr/stream_skid.sv",
           "rtl/asu/rsqrt.sv", "rtl/asu/asu_rmsnorm.sv"]

MUTANTS = [
    dict(
        name="MW1_rms_rne_to_half_up",
        mut_file="rtl/asu/asu_rmsnorm.sv",
        old="assign rnd_up = (rem_v > 18'h20000) || "
            "((rem_v == 18'h20000) && y_tr[0]);",
        new="assign rnd_up = (rem_v >= 18'h20000);",
        dmax=3584,
        plusargs=["+vectors=build/vectors_wide_directed.txt",
                  "+bp_mode=1", "+stall_mode=1", "+g_mode=0", "+seed=41"],
    ),
    dict(
        name="MW2_mu_off_by_one",
        mut_file="rtl/asu/asu_rmsnorm.sv",
        old="assign mu_v     = WIDE_RMS_MU[k_idx];",
        new="assign mu_v     = 17'(WIDE_RMS_MU[k_idx] + 17'd1);",
        dmax=3584,
        plusargs=["+vectors=build/vectors_wide_directed.txt",
                  "+bp_mode=1", "+stall_mode=0", "+g_mode=0", "+seed=42"],
    ),
    dict(
        name="MW3_mu_shift_off_by_one",
        mut_file="rtl/asu/asu_rmsnorm.sv",
        old="assign sh_v     = 5'(s_v) + 5'd16;",
        new="assign sh_v     = 5'(s_v) + 5'd15;",
        dmax=3584,
        plusargs=["+vectors=build/vectors_wide_directed.txt",
                  "+bp_mode=0", "+stall_mode=0", "+g_mode=0", "+seed=43"],
    ),
    dict(
        name="MW4_sumw_trunc_27",
        mut_file="rtl/asu/asu_rmsnorm.sv",
        old="localparam int unsigned SUM_W  = "
            "$clog2(32'(RMS_D_MAX) * 32'd16384 + 32'd1);",
        new="localparam int unsigned SUM_W  = 27;",
        dmax=8192,
        plusargs=["+vectors=build/vectors_wide8k.txt",
                  "+bp_mode=1", "+stall_mode=0", "+g_mode=0", "+seed=44"],
    ),
]


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=1800)
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
        for rel in RTL_RMS:
            dst = scratch / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = (REPO / rel).read_text()
            if rel == m["mut_file"]:
                assert text.count(m["old"]) == 1, \
                    f"{name}: mutation site not found exactly once in {rel}"
                text = text.replace(m["old"], m["new"])
            dst.write_text(text)
            srcs.append(str(dst))
        assert (scratch / m["mut_file"]).read_text() \
            != (REPO / m["mut_file"]).read_text(), f"{name}: no-op mutation"

        obj = scratch / "obj"
        rc, out = run([VERILATOR, "--binary", "-Wall", "--timing", "--assert",
                       "--timescale", "1ns/1ps", "lint_waivers.vlt",
                       "-I" + str(REPO / "verif/common"),
                       "+incdir+" + str(REPO / "rtl/asu"),
                       "--top-module", "tb_asu_rmsnorm_wide",
                       f"-GD_MAX={m['dmax']}", "-Mdir", str(obj)]
                      + srcs + ["tb_asu_rmsnorm_wide.sv"], HERE)
        if rc != 0:
            print(f"MUTANT {name}: BUILD FAILED (rc={rc}) — cannot evaluate")
            print(out[-2000:])
            return 1

        rc, out = run([str(obj / "Vtb_asu_rmsnorm_wide")] + m["plusargs"],
                      HERE)
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

    print(f"WIDE MUTATION GATE: {detected}/{len(MUTANTS)} mutants detected — "
          "checkers proven live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
