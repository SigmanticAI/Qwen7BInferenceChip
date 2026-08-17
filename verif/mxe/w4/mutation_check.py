#!/usr/bin/env python3
"""mutation_check.py — prove the B3 native-W4 weight-feeder checkers CATCH
injected RTL bugs (house pattern: verif/seam/mutation_check.py, a repeatable
build gate).

Each mutant: copy the RTL into a scratch dir under build/ (rtl/ is NEVER
modified, and build/obj_w4 — the golden build — is never touched), apply one
exact-string mutation (asserted to hit exactly once), rebuild the scoreboard
TB against the mutated copy, run the DIRECTED vector set, and REQUIRE a
detection (non-zero exit / FAIL / assertion / watchdog). An undetected mutant
fails this gate — 'make all' therefore proves checker credibility, not just
PASS.

Builds run STRICTLY SEQUENTIALLY: Verilator is the heavy step and this suite
shares one machine with the other lanes.

TIMEOUT = KILL. A mutant that wedges the feeder (M5 is the candidate: the
odd tail's padding half never retires) shows up as a HANG, not a mismatch.
The TB's own JOB_TIMEOUT/$fatal watchdog normally converts that into a
non-zero exit; the subprocess timeout below is the backstop, and a run that
hits it is counted as KILLED and labelled "timeout" in the table.

Mutants (docs/design/B3_WEIGHT_PATH.md §4, "Mutation-gate expectations"):
  M1 nibble order       : unpack_half takes the OTHER half's nibbles
                          (low<->high swap) — killed by any beat compare
  M2 sign extension     : {{4{nib[3]}}, nib} -> {4'b0, nib} (zero-extend) —
                          killed only by the negative-code vectors
  M3 stuck w4_en        : mode_q forced to 1 at job accept — bypass of mode,
                          killed only by the passthrough (w4_en=0) jobs
  M4 lane off-by-one    : unpack_half lane index 8*half+r -> +1 (mod 16) —
                          the lane-doubling mutant realization (A) must kill
  M5 odd tail           : need_new's ceil -> floor, so an odd job asks for
                          one packed beat too few and can never complete —
                          killed via the TB's JOB_TIMEOUT hang path

WHY M5 TARGETS `need_new` AND NOT `hold_retire` (measured 2026-07-20):
the contract (§4) originally named "hold_retire drops `|| last_beat`" as the
odd-tail mutant. That was BUILT AND RUN, and it SURVIVES every regime
(directed, random, storm, reset x bp_mode 0/1/2 — 300 legal jobs, 61 of them
odd-tail). It is an EQUIVALENT MUTANT, not a checker hole: on the cycle the
odd tail's final beat is emitted, ST_RUN takes `if (last_beat) state <=
ST_WAIT`, and `o_valid = (state == ST_RUN) && hold_valid`, so the stale half
can never be emitted from ST_WAIT; the next job accept re-clears
hold_valid/half anyway. The term only clears two flops one cycle earlier —
no port-level TB can kill it, and none should be expected to. The term is
KEPT in the RTL as documented-defensive (the seam_feeder_quant P5
precedent: never weaken a check, document the reachability), and the gate is
re-pointed here at `need_new`'s ceil, which IS load-bearing for odd tails.
That relocation is recorded in docs/design/B3_WEIGHT_PATH.md §4 — the mutant
list was corrected against measurement, not waived.
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

# per-mutant wall-clock caps (seconds): the golden directed run is ~0.01 s,
# so the run cap is pure hang insurance.
BUILD_TIMEOUT = 1800
RUN_TIMEOUT = 300

RTL = ["rtl/apex_pkg.sv", "rtl/xbr/stream_skid.sv", "rtl/mxe/mxe_wfeed_w4.sv"]
MUT_FILE = "rtl/mxe/mxe_wfeed_w4.sv"
TB = "tb_wfeed_w4_sb.sv"
TOP = "tb_wfeed_w4_sb"
VECTORS = "build/vectors_w4_directed.txt"

MUTANTS = [
    dict(
        name="M1_nibble_order_swap",
        rtl=RTL, tb=TB, top=TOP, gparams=[], mut_file=MUT_FILE,
        old="      nib = packed_in[4*(8*int'(half) + r) +: 4];",
        new="      nib = packed_in[4*(8*int'(!half) + r) +: 4];"
            "   // MUTANT: halves swapped",
        plusargs=[f"+vectors={VECTORS}", "+bp_mode=1", "+stall_mode=1",
                  "+seed=301"],
    ),
    dict(
        name="M2_sign_ext_dropped",
        rtl=RTL, tb=TB, top=TOP, gparams=[], mut_file=MUT_FILE,
        old="      unpack_half[8*r +: 8] = {{4{nib[3]}}, nib};"
            "      // sign-extend",
        new="      unpack_half[8*r +: 8] = {4'b0, nib};"
            "      // MUTANT: zero-extend",
        plusargs=[f"+vectors={VECTORS}", "+bp_mode=1", "+stall_mode=1",
                  "+seed=302"],
    ),
    dict(
        name="M3_stuck_w4_en",
        rtl=RTL, tb=TB, top=TOP, gparams=[], mut_file=MUT_FILE,
        old="              mode_q     <= w4_en;"
            "         // held for the job (no mid-job tear)",
        new="              mode_q     <= 1'b1;"
            "          // MUTANT: mode stuck W4 (bypass of mode)",
        plusargs=[f"+vectors={VECTORS}", "+bp_mode=0", "+stall_mode=0",
                  "+seed=303"],
    ),
    dict(
        name="M4_lane_off_by_one",
        rtl=RTL, tb=TB, top=TOP, gparams=[], mut_file=MUT_FILE,
        # guarded (mod 16) so the part-select stays inside the 64-bit beat:
        # an out-of-range index would die at lint, not at the checkers.
        old="      nib = packed_in[4*(8*int'(half) + r) +: 4];",
        new="      nib = packed_in[4*((8*int'(half) + r + 1) % 16) +: 4];"
            "   // MUTANT: lane off-by-one",
        plusargs=[f"+vectors={VECTORS}", "+bp_mode=1", "+stall_mode=1",
                  "+seed=304"],
    ),
    dict(
        name="M5_odd_tail_floor_not_ceil",
        rtl=RTL, tb=TB, top=TOP, gparams=[], mut_file=MUT_FILE,
        old="  assign need_new = w4_en ? ((CNT_W'(job_beats) + CNT_W'(1)) >> 1)",
        new="  assign need_new = w4_en ? ((CNT_W'(job_beats)) >> 1)"
            "   // MUTANT: floor, not ceil",
        # The LOAD-BEARING odd-tail site: floor instead of ceil under-counts
        # the packed beats an odd job needs, so the feeder stops accepting
        # input one beat early and the job can never complete. That is a HANG,
        # not a mismatch — the TB's JOB_TIMEOUT/$fatal fires first and
        # RUN_TIMEOUT is the backstop; both count as KILLED.
        plusargs=[f"+vectors={VECTORS}", "+bp_mode=2", "+stall_mode=2",
                  "+seed=305", "+watchdog=2000000"],
    ),
]


def run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, bool]:
    """Return (rc, output, timed_out). A timeout is a real result here, not
    an error: for a wedging mutant it IS the detection."""
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return -1, out, True


def main() -> int:
    results = []          # (name, verdict, rc, evidence)
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

        # own -Mdir per mutant: builds never collide, and build/obj_w4 (the
        # golden build) is never written to
        obj = scratch / "obj"
        rc, out, _to = run([VERILATOR, "--binary", "-Wall", "--timing",
                            "--assert", "--timescale", "1ns/1ps",
                            "lint_waivers.vlt",
                            "-I" + str(REPO / "verif/common"), "-I" + str(HERE)]
                           + m["gparams"]
                           + ["--top-module", m["top"], "-Mdir", str(obj)]
                           + srcs + [m["tb"]], HERE, BUILD_TIMEOUT)
        if rc != 0:
            print(f"MUTANT {name}: BUILD FAILED (rc={rc}) — cannot evaluate")
            print(out[-2000:])
            results.append((name, "BUILD_FAIL", rc, "build did not complete"))
            continue

        rc, out, timed_out = run([str(obj / ("V" + m["top"]))] + m["plusargs"],
                                 HERE, RUN_TIMEOUT)
        (scratch / "run.log").write_text(out)
        clean_pass = (not timed_out) and (rc == 0) and ("TB PASS" in out) \
            and ("FAIL" not in out) and ("Assertion failed" not in out)
        if clean_pass:
            print(f"MUTANT {name}: *** SURVIVED *** (checkers are blind)")
            results.append((name, "SURVIVED", rc,
                            "TB PASS with the mutation in place"))
            continue
        if timed_out:
            evidence = f"timeout after {RUN_TIMEOUT}s (hang == kill)"
        else:
            evidence = next((ln for ln in out.splitlines()
                             if "FAIL" in ln or "Assertion failed" in ln
                             or "%Error" in ln), "(nonzero exit)").strip()
        print(f"MUTANT {name}: KILLED (rc={rc}) — {evidence}")
        results.append((name, "KILLED", rc, evidence))

    killed = sum(1 for r in results if r[1] == "KILLED")
    survived = [r[0] for r in results if r[1] == "SURVIVED"]
    broken = [r[0] for r in results if r[1] == "BUILD_FAIL"]

    print()
    print(f"{'mutant':<26}{'verdict':<11}{'rc':>4}  evidence")
    print("-" * 100)
    for (name, verdict, rc, evidence) in results:
        print(f"{name:<26}{verdict:<11}{rc:>4}  {evidence[:56]}")
    print("-" * 100)
    print("NOTE: a run that hits the subprocess timeout is counted as KILLED "
          "— a wedged feeder is a detection.")

    if broken:
        print(f"\nMUTATION FAIL: {len(broken)} mutant(s) did not build: "
              f"{broken}")
        return 1
    if survived:
        print(f"\nMUTATION FAIL: {len(survived)} mutant(s) SURVIVED: "
              f"{survived}")
        return 1
    print(f"\nMUTATION GATE: {killed}/{len(MUTANTS)} mutants killed — "
          "checkers proven live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
