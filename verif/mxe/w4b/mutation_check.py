#!/usr/bin/env python3
"""mutation_check.py — W4B feeder checker-credibility gate (D-031; house
pattern verif/asu/sb + verif/mxe/w4). Scratch copies only; rtl/ never
modified. A mutant that hangs the TB (watchdog/timeout $fatal) counts as
CAUGHT — nonzero exit is detection, per the (A)-suite precedent.

Mutants (each targets logic THIS lane added; the divider internals are
fparith-certified upstream and mutation-gated there):
  MW1 shortcut removal : drop the sh>=23 saturation branch — the 41-bit
        width proof's teeth: sh>=28 now overflows (crafted big-sg/tiny-s8
        jobs in the directed vectors hit sh=29)
  MW2 gid cadence      : p >> S3 -> p >> (S3+1) — wrong group boundary,
        any multi-group job with distinct scales dies
  MW3 sideband desync  : advance sb_rp on f_vld instead of pv_out[0] —
        neg/zero flags misalign with divider results
  MW4 odd-tail hang    : pw_consume drops the last_emit arm — odd jobs
        never consume their final packed beat (caught as timeout)
  MW5 legality removal : the beats==KB*N cross-check deleted — the I
        records' reject checks fail
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

RTL = ["rtl/apex_pkg.sv", "rtl/xbr/stream_skid.sv",
       "rtl/kvq/cores/cq_fp_pkg.sv", "rtl/kvq/cores/cq_rne_div_pipe.sv",
       "rtl/mxe/w4b_fp_pkg.sv", "rtl/mxe/mxe_wfeed_w4b.sv"]

MUTANTS = [
    dict(name="MW1_shortcut_removed", mut_file="rtl/mxe/w4b_fp_pkg.sv",
         old="""      if (sh >= 23) begin
        // saturation shortcut (width proof above): quotient >= 2^12 always
        f.n = {41{1'b1}};
        f.d = 41'(m8);
      end else if (sh >= 0) begin""",
         new="""      if (sh >= 0) begin""",
         vec="vectors_w4b_directed.txt"),
    dict(name="MW2_gid_cadence", mut_file="rtl/mxe/mxe_wfeed_w4b.sv",
         old="assign gid_now = GW'(32'(col_q) * 32'(ngk_q) + 32'(prow_q >> S3));",
         new="assign gid_now = GW'(32'(col_q) * 32'(ngk_q) + 32'(prow_q >> (S3 + 1)));",
         vec="vectors_w4b_directed.txt"),
    dict(name="MW3_sideband_desync", mut_file="rtl/mxe/mxe_wfeed_w4b.sv",
         old="""        if (pv_out[0])
          sb_rp <= ($clog2(SB_D))'((32'(sb_rp) + 1) % SB_D);""",
         new="""        if (f_vld)
          sb_rp <= ($clog2(SB_D))'((32'(sb_rp) + 1) % SB_D);""",
         vec="vectors_w4b_directed.txt"),
    dict(name="MW4_odd_tail_hang", mut_file="rtl/mxe/mxe_wfeed_w4b.sv",
         old="wire pw_consume = issue && (half_q || last_emit);",
         new="wire pw_consume = issue && half_q;",
         vec="vectors_w4b_directed.txt"),
    dict(name="MW5_legality_removed", mut_file="rtl/mxe/mxe_wfeed_w4b.sv",
         old="                   && (32'(job_beats) == 32'(kb_calc) * 32'(job_n));",
         new="                   ;",
         vec="vectors_w4b_directed.txt"),
]


def run(cmd, cwd, timeout=900):
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or "") + "\n[HARNESS TIMEOUT]"


def main() -> int:
    detected = 0
    for m in MUTANTS:
        name = m["name"]
        scratch = BUILD / f"mut_{name}"
        if scratch.exists():
            shutil.rmtree(scratch)
        srcs = []
        for rel in RTL:
            dst = scratch / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = (REPO / rel).read_text()
            if rel == m["mut_file"]:
                assert text.count(m["old"]) == 1, \
                    f"{name}: site not found exactly once"
                text = text.replace(m["old"], m["new"])
            dst.write_text(text)
            srcs.append(str(dst))
        obj = scratch / "obj"
        rc, out = run([VERILATOR, "--binary", "-Wno-fatal", "--timing",
                       "--timescale", "1ns/1ps", "-CFLAGS", "-std=gnu++17",
                       "--top-module", "tb_w4b_sb", "-Mdir", str(obj)]
                      + srcs + [str(HERE / "tb_w4b_sb.sv")], HERE)
        if rc != 0:
            print(f"MUTANT {name}: BUILD FAILED — cannot evaluate")
            print(out[-1500:])
            return 1
        rc, out = run([str(obj / "Vtb_w4b_sb"),
                       f"+vectors=build/{m['vec']}",
                       "+bp_mode=1", "+stall_mode=1", "+gs_mode=0",
                       "+seed=71"], HERE, timeout=600)
        (scratch / "run.log").write_text(out)
        clean = (rc == 0) and ("TB PASS" in out) and ("FAIL" not in out)
        if clean:
            print(f"MUTANT {name}: *** NOT DETECTED ***")
            return 1
        first = next((ln for ln in out.splitlines()
                      if "FAIL" in ln or "Fatal" in ln or "TIMEOUT" in ln),
                     "(nonzero exit)")
        print(f"MUTANT {name}: DETECTED (rc={rc}) — {first.strip()[:110]}")
        detected += 1
    print(f"W4B MUTATION GATE: {detected}/{len(MUTANTS)} mutants detected — "
          "checkers proven live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
