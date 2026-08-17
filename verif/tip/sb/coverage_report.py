#!/usr/bin/env python3
"""coverage_report.py — aggregate the manual TIP coverage buckets (run2
pattern) printed by tb_tip_sb.sv ("COV <name> <count>") across all run logs
and print a hit/hole table with reachability notes.

Exit status: 1 if any REQUIRED bucket has zero aggregate hits, else 0 —
'make all' therefore gates on coverage closure, not just test pass.
"""
import re
import sys
from collections import defaultdict

BUCKETS = {}
# threshold sweep: all 32 register values, incl. 0 (hardware clamp to 1)
for t in range(32):
    note = "THRESHOLD_REG=0 -> thr_eff clamp to 1 (D-017)" if t == 0 else \
        f"THRESHOLD_REG={t} tiles golden-checked"
    BUCKETS[f"thr_t{t}"] = (True, note)

BUCKETS.update({
    "dec_fp16":        (True, "FP16 verdicts delivered + checked"),
    "dec_int8":        (True, "INT8 verdicts delivered + checked"),
    "tie_exact":       (True, "exact LHS==RHS tie (strict > : must be INT8)"),
    "tie_off1":        (True, "one ULP either side of the tie"),
    "mag_zero_tile":   (True, "all-zero tile (0 > T*0 false -> INT8, inc 0)"),
    "mag_int32min":    (True, "INT32_MIN present (|.| = 2^31 exact)"),
    "mag_int32max":    (True, "INT32_MAX present"),
    "len_1":           (True, "1-score tile (decision uses the full N shift)"),
    "len_short":       (True, "1 < len < N tiles"),
    "len_full":        (True, "len == N tiles"),
    "imp_inc0":        (True, "bucket-0 update (max_mag==0): acc must hold"),
    "imp_sat":         (True, "accumulator saturates to 0xFFFF (no wrap)"),
    "imp_eq_lo":       (True, "acc == imp_lo exactly (>= boundary: CQ4P)"),
    "imp_eq_hi":       (True, "acc == imp_hi exactly (>= boundary: CQ8)"),
    "tier_cq4":        (True, "CQ4 tier observed on the decision stream"),
    "tier_cq4p":       (True, "CQ4P tier observed on the decision stream"),
    "tier_cq8":        (True, "CQ8 tier observed on the decision stream"),
    "frame_onlast":    (True, "overrun exactly ON the s_last beat (N+1)"),
    "frame_abort":     (True, "overrun mid-tile -> abort + drain (>N+1)"),
    "sticky_clear":    (True, "frame_err_sticky cleared via frame_err_clear"),
    "imp_clear_midrun": (True, "CSR imp_clear wipes accumulators mid-run"),
    "rst_p1":          (True, "mid-op reset: beat pending in the input skid"),
    "rst_p2":          (True, "mid-op reset: engine mid-tile (cnt >= 2)"),
    "rst_p3":          (True, "mid-op reset: during F-2 abort drain"),
    "rst_p4":          (True, "mid-op reset: dec_pend (decision in flight)"),
    "rst_p5":          (True, "mid-op reset: decision hold register occupied"),
    "rst_p6":          (True, "mid-op reset: output skid full (2 beats)"),
    "bp_full":         (True, "run with d_ready held high"),
    "bp_random":       (True, "run with ~75% duty backpressure"),
    "bp_storm":        (True, "run with ~6% duty backpressure storm"),
    "gap_none":        (True, "run with full-rate score feed"),
    "gap_random":      (True, "run with short random feed gaps"),
    "gap_storm":       (True, "run with bursty stalled feeds (up to 24 cyc)"),
    "ntile_64":        (True, "BLOCK 8x8 build (N=64)"),
    "ntile_4096":      (True, "BLOCK 64x64 build (N=4096, frozen-143 subset)"),
})


def main():
    counts = defaultdict(int)
    pat = re.compile(r"^COV\s+(\S+)\s+(\d+)\s*$")
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                mm = pat.match(line)
                if mm:
                    counts[mm.group(1)] += int(mm.group(2))

    print(f"{'bucket':<18} {'hits':>7}  {'status':<8} note")
    print("-" * 78)
    holes = 0
    for name, (required, note) in BUCKETS.items():
        hits = counts.get(name, 0)
        if hits > 0:
            status = "HIT"
        elif required:
            status = "HOLE"
            holes += 1
        else:
            status = "hole(ok)"
        print(f"{name:<18} {hits:>7}  {status:<8} {note}")
    unknown = set(counts) - set(BUCKETS)
    for name in sorted(unknown):
        print(f"{name:<18} {counts[name]:>7}  {'?':<8} (bucket not in plan)")
    print("-" * 78)
    if holes:
        print(f"COVERAGE: {holes} REQUIRED hole(s)")
        return 1
    print("COVERAGE: all required buckets hit "
          f"({sum(1 for b, (r, _) in BUCKETS.items() if r)} required, "
          f"{len(BUCKETS)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
