#!/usr/bin/env python
"""coverage_report.py — aggregate manual coverage buckets (verif/kvq/sb).

Reads:
  * "COV <name> <n>"     lines from the run logs (TB-counted, run-proven)
  * "COVGEN <name> <n>"  lines from build/vec_*/cov_gen.txt and
                         build/cov_proofs.txt (generator-counted numeric
                         conditions + reachability proofs)

Gates the build: every REQUIRED bucket must meet its minimum or exit 1
(required-holes fail the build, ARCHITECTURE.md §8 / run2 pattern).

Reachability notes (why some tempting buckets are NOT required):
  * quant saturation clamp (sat_pos) and qmin (-8/-128): UNREACHABLE through
    kvq_engine — the scale is computed from the same data's amax, so
    |x|/s <= qmax*(1+2^-11) < qmax+0.5 and rne never exceeds qmax. Proven by
    the generator's exhaustive sweep over all finite fp16 amax magnitudes
    (COVGEN satpos_unreachable_proof / qmin_unreachable_proof are REQUIRED
    instead — the proof must have run).
  * rst_soft_p3/p4/p5/p8 (STORE/RLOAD/RWAIT/KFEED parked): single-cycle
    states cannot be *parked* for an AXI-latency soft reset; the two
    deterministic collision tests (rst_collide_kfeed / rst_collide_rload)
    cover the reachable soft-reset overlap with those states instead.
  * rst_soft_p10 is covered by rst_kemit_probe (kind-4 X + Y follow-up).
"""
import re
import sys
from collections import Counter
from pathlib import Path

REQUIRED = {
    # scoreboard volume
    "v_tok": 200, "k_tok": 150, "reads": 250, "peeks": 200,
    "rand_v": 150, "rand_k": 80,
    # numeric conditions (generator-proven present in the stimulus)
    "tie_hits": 8, "eps_clamp": 2, "zero_tok": 2, "subnormal_tok": 1,
    "maxnorm_tok": 2, "neg_amax_winner": 3, "negzero_outlier": 1,
    # structural key-path shapes
    "k_group_full": 3, "k_flush_g1": 1, "k_flush_gm1": 2, "k_flush_mid": 1,
    "flush_midtoken": 1, "flush_on_full": 1, "flush_noop_idle": 1,
    "flush_during_emit": 1, "flush_pulse": 5,
    # storage behaviors
    "overwrite": 5, "occ_check": 3, "sram_full_seen": 1,
    # A2/D-026 persistent scale bank: row compares (TB-counted), sticky
    # SB_OVWR observed + W1C'd (TB), reads that hit an OVERWRITTEN set
    # (generator-proven: expected hat computed from the reused row), and the
    # k=5 outlier-lane geometry generated (k5 config)
    "bank_check": 20, "sb_ovwr_check": 4, "sb_ovwr_w1c": 1,
    "stale_key_read": 3, "outlier_lanes5": 1,
    # error path
    "rderr": 3, "w1c": 3, "irq_masked": 1,
    # protocol adversaries
    "bp_mode_0": 1, "bp_mode_1": 1, "bp_mode_2": 1,
    "stall_mode_0": 1, "stall_mode_1": 1, "stall_mode_2": 1,
    "framing_early": 1, "framing_notlast": 1, "framing_late": 1,
    # mid-operation reset: hard in EVERY phase, soft where parkable,
    # deterministic collisions for the single-cycle states
    **{f"rst_hard_p{i}": 1 for i in range(1, 12)},
    **{f"rst_soft_p{i}": 1 for i in (1, 2, 6, 7, 9, 11)},
    "rst_soft_p10": 1,
    "rst_collide_kfeed": 1, "rst_collide_rload": 1, "rst_kemit_probe": 1,
    # reachability proofs must have executed
    "satpos_unreachable_proof": 1, "qmin_unreachable_proof": 1,
}

UNREACHABLE_NOTES = [
    "sat_pos / qmin(-8): unreachable through the top (exhaustive fp16 sweep)",
    "rst_soft_p3/p4/p5/p8: single-cycle states — covered by the deterministic",
    "  collision tests (rst_collide_rload / rst_collide_kfeed) instead",
]


def main(paths):
    got = Counter()
    n_logs = 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        n_logs += 1
        for line in p.read_text(errors="replace").splitlines():
            m = re.match(r"^\s*(COV|COVGEN)\s+(\S+)\s+(-?\d+)\s*$", line)
            if m:
                got[m.group(2)] += int(m.group(3))
    holes = []
    print(f"==== KVQ SB coverage ({n_logs} inputs) ====")
    for name in sorted(set(REQUIRED) | set(got)):
        need = REQUIRED.get(name, 0)
        have = got.get(name, 0)
        mark = "OK " if have >= need else "HOLE"
        if need == 0:
            mark = "----"
        print(f"  [{mark}] {name:26s} {have:8d} (min {need})")
        if have < need:
            holes.append(name)
    print("---- reachability notes ----")
    for n in UNREACHABLE_NOTES:
        print(f"  {n}")
    if holes:
        print(f"COVERAGE GATE: FAIL — {len(holes)} required hole(s): {holes}")
        sys.exit(1)
    print("COVERAGE GATE: PASS — all required buckets closed")


if __name__ == "__main__":
    main(sys.argv[1:])
