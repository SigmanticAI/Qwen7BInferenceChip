#!/usr/bin/env python3
"""coverage_report.py — aggregate the manual coverage buckets (run2 pattern)
printed by tb_mxe_sb.sv ("COV <name> <count>") across all run logs and print
a hit/hole table with reachability notes.

Exit status: 1 if any REQUIRED bucket has zero aggregate hits, else 0 —
'make all' therefore gates on coverage closure, not just test pass.
"""
import re
import sys
from collections import defaultdict

# bucket -> (required, reachability note)
BUCKETS = {
    "mode_ws":       (True,  "OP_GEMM_WS jobs"),
    "mode_os":       (True,  "OP_GEMM_OS jobs"),
    "rq_on":         (True,  "requant epilogue exercised (C-2)"),
    "rq_off":        (True,  "raw INT32 drain exercised (D-003)"),
    "os_acc0":       (True,  "OS overwriting job (clr on pass 0)"),
    "os_acc1":       (True,  "OS accumulate chain link (resident retention)"),
    "m_eq_1":        (True,  "M=1 single-beat drain (last on beat 0)"),
    "m_eq_64":       (True,  "M=M_TILE_MAX (abuf/acc row capacity)"),
    "m_mid":         (True,  "1 < M < 64"),
    "k_eq_1":        (True,  "K=1 (klast=1, single chunk)"),
    "k_mult8":       (True,  "K multiple of 8 (klast=8, no tail mask)"),
    "k_tail":        (True,  "K % 8 != 0 (tail weight-row masking)"),
    "k_eq_2048":     (True,  "K=K_MAX, 256 passes (C-3/D-003 bound)"),
    "n_eq_1":        (True,  "N=1 (7 zero-pad columns)"),
    "n_eq_8":        (True,  "N=MXE_N full width"),
    "n_mid":         (True,  "1 < N < 8"),
    "sat_pos":       (True,  "requant clipped at +127"),
    "sat_neg":       (True,  "requant clipped at -128"),
    "rne_tie":       (True,  "exact half-to-even tie in requant (C-2)"),
    "ill_opcode":    (True,  "reject: opcode not in {WS,OS}"),
    "ill_k0":        (True,  "reject: K=0"),
    "ill_kbig":      (True,  "reject: K>K_MAX"),
    "ill_m0":        (True,  "reject: M=0"),
    "ill_mbig":      (True,  "reject: M>M_TILE_MAX (buffer overrun class)"),
    "ill_n0":        (True,  "reject: N=0"),
    "ill_nbig":      (True,  "reject: N>MXE_N"),
    "ill_buf":       (True,  "reject: unimplemented buffer id"),
    "rst_ingest":    (True,  "mid-op reset in S_INGEST"),
    "rst_wload":     (True,  "mid-op reset in S_WLOAD"),
    "rst_compute":   (True,  "mid-op reset in S_COMPUTE"),
    "rst_flush":     (True,  "mid-op reset in S_FLUSH"),
    "rst_drain":     (True,  "mid-op reset in S_DRAIN"),
    "rst_wait_done": (False, "1-2 cycle window post-drain; recovery path "
                             "identical to S_DRAIN reset (best effort)"),
    "rst_idle":      (False, "abort landed after job completion — reset from "
                             "idle, not a mid-op case (informational)"),
    "bp_full":       (True,  "run with res_ready held high"),
    "bp_random":     (True,  "run with ~75% duty backpressure"),
    "bp_storm":      (True,  "run with ~6% duty backpressure storm (V0.2)"),
    "stall_none":    (True,  "run with full-rate act/wgt feeds"),
    "stall_random":  (True,  "run with short random feed gaps"),
    "stall_storm":   (True,  "run with bursty stalled feeds (up to 32 cyc)"),
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

    print(f"{'bucket':<15} {'hits':>7}  {'status':<8} note")
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
        print(f"{name:<15} {hits:>7}  {status:<8} {note}")
    unknown = set(counts) - set(BUCKETS)
    for name in sorted(unknown):
        print(f"{name:<15} {counts[name]:>7}  {'?':<8} (bucket not in plan)")
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
