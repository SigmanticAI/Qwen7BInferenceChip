#!/usr/bin/env python3
"""coverage_report.py — aggregate the manual SEQ coverage buckets (run2
pattern) printed by tb_seq_sb.sv ("COV <name> <count>") across all run logs
and print a hit/hole table with reachability notes.

Exit status: 1 if any REQUIRED bucket has zero aggregate hits, else 0 —
'make all' therefore gates on coverage closure, not just test pass.
"""
import re
import sys
from collections import defaultdict

BUCKETS = {
    # descriptor mix (golden-annotated categories)
    "cat_legal_plain": (True, "plain legal descriptors walked"),
    "cat_legal_k1":    (True, "legal K=1 boundary"),
    "cat_legal_k2048": (True, "legal K=2048 boundary (D-003 cap)"),
    "cat_legal_m1":    (True, "legal M=1 (single-beat job)"),
    "cat_bad_opcode":  (True, "illegal opcode -> reject"),
    "cat_k_zero":      (True, "K=0 -> reject"),
    "cat_k_big":       (True, "K>2048 -> reject (D-003)"),
    "cat_m_zero":      (True, "M=0 -> reject"),
    "cat_n_zero":      (True, "N=0 -> reject"),
    "cat_m_big":       (True, "M>64 -> reject (§3 implemented range)"),
    "cat_n_big":       (True, "N>8 -> reject (§3 implemented range)"),
    "cat_buf_bad":     (True, "nonzero buffer id -> reject"),
    # walk / queue behaviour
    "drains":          (True, "full-drain checkpoints (N records)"),
    "q_full_seen":     (True, "FIFO at QDEPTH observed"),
    "skid_bp":         (True, "input skid backpressured (ds_ready low)"),
    "b2b_dispatch":    (True, "dispatch within 3 cycles of previous done"),
    "beats_last":      (True, "result-beat `last` framing checked"),
    "jobs_done":       (True, "legal jobs completed (done pulses)"),
    "jobs_rejected":   (True, "illegal descriptors rejected"),
    # abort / gating
    "abort_inflight":  (True, "soft abort with a job IN FLIGHT (must finish)"),
    "abort_queued":    (True, "soft abort with queued-only work (all dropped)"),
    "abort_empty":     (True, "soft abort while empty (no-op resync)"),
    "abort_feed":      (True, "ds handshake completed IN the abort window "
                              "(incl. its final cycle -> skid-resident beat)"),
    "gate_off":        (True, "CTRL.enable=0 holds dispatch"),
    "gate_on":         (True, "CTRL.enable=1 resumes"),
    "qcheck":          (True, "settled queue-state compare (Q records)"),
    # mid-op reset phases
    "rst_p1":          (True, "hard reset: queue filling, head stalled"),
    "rst_p2":          (True, "hard reset: descriptor presented, unaccepted"),
    "rst_p3":          (True, "hard reset: job in flight"),
    "rst_p4":          (True, "hard reset: mid-abort drain"),
    # run configurations
    "bp_m0":           (True, "result backpressure: none"),
    "bp_m1":           (True, "result backpressure: ~75% duty"),
    "bp_m2":           (True, "result backpressure: storm"),
    "gap_m0":          (True, "descriptor feed: full rate"),
    "gap_m1":          (True, "descriptor feed: short gaps"),
    "gap_m2":          (True, "descriptor feed: bursty stalls"),
    "lat_m0":          (True, "stub: zero-latency accept/beats"),
    "lat_m1":          (True, "stub: random latencies"),
    "lat_m2":          (True, "stub: storm latencies"),
}


def main():
    counts = defaultdict(int)
    pat = re.compile(r"^COV\s+(\S+)\s+(\d+)\s*$")
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                mm = pat.match(line)
                if mm:
                    counts[mm.group(1)] += int(mm.group(2))

    print(f"{'bucket':<18} {'hits':>9}  {'status':<8} note")
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
        print(f"{name:<18} {hits:>9}  {status:<8} {note}")
    unknown = set(counts) - set(BUCKETS)
    for name in sorted(unknown):
        print(f"{name:<18} {counts[name]:>9}  {'?':<8} (bucket not in plan)")
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
