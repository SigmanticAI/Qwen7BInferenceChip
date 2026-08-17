#!/usr/bin/env python3
"""coverage_report.py — aggregate the manual CSR coverage buckets (run2
pattern) printed by tb_csr_sb.sv ("COV <name> <count>") across all run logs
and print a hit/hole table with reachability notes.

Exit status: 1 if any REQUIRED bucket has zero aggregate hits, else 0 —
'make all' therefore gates on coverage closure, not just test pass.
"""
import re
import sys
from collections import defaultdict

REG = {
    0x00: "CTRL", 0x04: "STATUS", 0x08: "INFO_N", 0x0C: "INFO_D",
    0x10: "INFO_G", 0x14: "INFO_TIER", 0x18: "INFO_VERSION",
    0x1C: "reserved-in-map", 0x20: "TIER_CTRL", 0x24: "THRESHOLD_REG",
    0x28: "FLUSH", 0x2C: "IMPORTANCE_BASE", 0x30: "PERF_CTRL",
    0x34: "PERF_CYCLES",
}
for i in range(8):
    REG[0x38 + 4 * i] = f"PERF_BUSY_{i}"

BUCKETS = {}
for a, name in REG.items():
    BUCKETS[f"rd_{a:02x}"] = (True, f"read of {name}")
    # RO counters are not in the random write pool (writes are no-ops anyway)
    wr_req = a not in (0x34, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54)
    BUCKETS[f"wr_{a:02x}"] = (wr_req, f"write to {name}"
                              + ("" if wr_req else " (RO, optional)"))

BUCKETS.update({
    "rsvd_rd":        (True, "aligned reserved/unmapped read -> 0xDEADBEEF"),
    "unaligned_rd":   (True, "unaligned read -> 0xDEADBEEF"),
    "rsvd_wr":        (True, "reserved/unaligned write: no side effects"),
    "w1c_write":      (True, "STATUS bit1 W1C write issued"),
    "sticky_set":     (True, "desc_error pulse -> sticky observed"),
    "sticky_race":    (True, "set and W1C in the SAME cycle (set must win)"),
    "tier_clamp":     (True, "TIER_CTRL code 2'b11 clamps to CQ4"),
    "thresh_mask":    (True, "THRESHOLD write with garbage above [4:0]"),
    "imp_mask":       (True, "IMPORTANCE_BASE write with garbage above field"),
    "busy_all":       (True, "block_busy_i all-ones (idle=0, busy=0xFF)"),
    "busy_some":      (True, "partial busy patterns"),
    "idle_seen":      (True, "block_busy_i zero (idle=1)"),
    "perf_en":        (True, "PERF_CTRL enable set"),
    "perf_clear":     (True, "PERF_CTRL clear (without enable)"),
    "perf_clr_en":    (True, "PERF_CTRL clear+enable in one write"),
    "perf_dis_hold":  (True, "counter nonzero and HELD while disabled"),
    "perf_sat":       (True, "counter pegged at all-ones (PERF_W=8 build)"),
    "perf_nonzero":   (True, "counter read back nonzero vs mirror"),
    "rw_simul":       (True, "simultaneous read+write (read-before-write)"),
    "b2b_rd":         (True, "back-to-back pipelined reads"),
    "hard_reset":     (True, "mid-run hard reset -> defaults re-checked"),
    "out_enable_chk": (True, "enable output port cross-checked vs CTRL"),
    "out_thresh_chk": (True, "threshold output port cross-checked"),
    "out_tier_chk":   (True, "tier_sel/tip_override ports cross-checked"),
    "perfw_32":       (True, "PERF_W=32 build ran"),
    "perfw_8":        (True, "PERF_W=8 build ran (saturation reachable)"),
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
