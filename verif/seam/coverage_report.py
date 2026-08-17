#!/usr/bin/env python3
"""coverage_report.py — aggregate the manual D-021 seam coverage buckets
(house pattern, verif/mxe/sb lineage) printed by tb_seam_feeder_sb.sv and
tb_seam_score_sb.sv ("COV <name> <count>") across all run logs and print a
hit/hole table with reachability notes.

Exit status: 1 if any REQUIRED bucket has zero aggregate hits, else 0 —
'make all' therefore gates on coverage closure, not just test pass.
"""
import re
import sys
from collections import defaultdict

# bucket -> (required, reachability note)
BUCKETS = {
    # ── feeder job shapes ────────────────────────────────────────────────────
    "fq_rows_1":        (True,  "single-row job"),
    "fq_rows_max":      (True,  "rows = ROWS_MAX = 64 (TB config)"),
    "fq_rows_mid":      (True,  "1 < rows < 64"),
    # ── feeder numerics (C-1 / D-021) ────────────────────────────────────────
    "fq_zero_row":      (True,  "all-zero row: amax=0 -> s=EPS, codes 0"),
    "fq_eps_row":       (True,  "amax>0 with amax/127 < 2^-14 -> EPS floor"),
    "fq_scale_tie":     (True,  "amax/127 exactly on an fp16 RNE tie point"),
    "fq_scale_bnd":     (True,  "amax/127 at f16 max / tie-to-inf boundary"),
    "fq_inf_scale":     (True,  "amax/127 overflows fp16 -> s=+inf, codes 0"),
    "fq_code_tie":      (True,  "element exactly at s*(k+0.5): RNE tie, both "
                                "parities"),
    "fq_subnormal":     (True,  "fp32 subnormal elements (code 0 by proof)"),
    "fq_neg_zero":      (True,  "-0.0 elements (magnitude compare, code 0)"),
    # NOTE: code saturation (+/-127 clamp binding, |x|/s >= 127.5) is
    # UNREACHABLE BY CONSTRUCTION when s derives from the row's own amax:
    # |x|/s <= 127/(1 - 2^-12) < 127.04 (module header proof P5). The clamp
    # is defensive RTL; no bucket is claimable without violating C-1.
    # ── feeder protocol / job semantics ─────────────────────────────────────
    "fq_ill_rows0":     (True,  "rows = 0 reject (§3)"),
    "fq_ill_rows_big":  (True,  "rows > ROWS_MAX reject (§3)"),
    "fq_rst_ingest":    (True,  "mid-op reset in ST_INGEST"),
    "fq_rst_scale":     (True,  "mid-op reset in ST_SCALE (1-cycle, targeted)"),
    "fq_rst_spush":     (True,  "mid-op reset in ST_SPUSH"),
    "fq_rst_drain":     (True,  "mid-op reset in ST_DRAIN"),
    "fq_rst_wait":      (True,  "mid-op reset in ST_WAIT (post-skid drain)"),
    "fq_rst_random":    (True,  "cycle-count (untargeted) aborts"),
    "fq_rst_idle":      (False, "abort landed after the job completed — not "
                                "a mid-op case (informational)"),
    "fq_bp_full":       (True,  "run with out/scl ready held high"),
    "fq_bp_random":     (True,  "run with ~75% duty backpressure"),
    "fq_bp_storm":      (True,  "run with ~6% duty backpressure storm"),
    "fq_stall_none":    (True,  "run with full-rate fp32 feed"),
    "fq_stall_random":  (True,  "run with short random feed gaps"),
    "fq_stall_storm":   (True,  "run with bursty stalled feed (<=32 cyc)"),
    "fq_cfg_d64":       (True,  "D=64 build ran (D-021 config)"),
    "fq_cfg_d128":      (True,  "D=128 build ran (D-021 config)"),
    # ── score-dequant job shapes ─────────────────────────────────────────────
    "sd_cols_1":        (True,  "single-column job"),
    "sd_cols_max":      (True,  "cols = N_MAX = 1024"),
    "sd_cols_mid":      (True,  "1 < cols < 1024"),
    "sd_cols_tail":     (True,  "cols % 8 != 0 (partial final acc beat)"),
    # ── score-dequant numerics (S-3) ─────────────────────────────────────────
    "sd_acc_zero":      (True,  "acc = 0 lanes"),
    "sd_acc_extreme":   (True,  "|acc| >= 2^30 (INT32 extremes)"),
    "sd_tie":           (True,  "acc*comp exactly on an integer RNE tie"),
    "sd_p53":           (True,  "|acc|*mant(comp) >= 2^53: the f64 "
                                "significand pre-round path (incl. the "
                                "constructed double-round discriminators)"),
    "sd_underflow":     (True,  "nonzero product rounds to score 0"),
    "sd_range_err":     (True,  "|score| >= 2^31: golden-assert mirror fires "
                                "(pulse count + sticky checked)"),
    "sd_comp_subnormal": (True, "subnormal composite scale (hidden bit 0)"),
    "sd_comp_one":      (True,  "comp = 1.0 identity (INT32 passthrough)"),
    # ── score-dequant protocol / job semantics ──────────────────────────────
    "sd_ill_cols0":     (True,  "cols = 0 reject (§3)"),
    "sd_ill_cols_big":  (True,  "cols > N_MAX reject (§3)"),
    "sd_rst_load":      (True,  "mid-op reset in ST_LOAD"),
    "sd_rst_proc":      (True,  "mid-op reset in ST_PROC"),
    "sd_rst_wait":      (True,  "mid-op reset in ST_WAIT"),
    "sd_rst_idle":      (False, "abort landed after the job completed "
                                "(informational)"),
    "sd_rst_random":    (True,  "cycle-count (untargeted) aborts"),
    "sd_bp_full":       (True,  "run with score_ready held high"),
    "sd_bp_random":     (True,  "run with ~75% duty backpressure"),
    "sd_bp_storm":      (True,  "run with ~6% duty backpressure storm"),
    "sd_stall_none":    (True,  "run with full-rate cmp/acc feeds"),
    "sd_stall_random":  (True,  "run with short random feed gaps"),
    "sd_stall_storm":   (True,  "run with bursty stalled feeds (<=32 cyc)"),
}


def main(argv) -> int:
    counts = defaultdict(int)
    pat = re.compile(r"^COV\s+(\S+)\s+(\d+)\s*$")
    for path in argv:
        with open(path) as f:
            for line in f:
                m = pat.match(line)
                if m:
                    counts[m.group(1)] += int(m.group(2))

    unknown = sorted(set(counts) - set(BUCKETS))
    holes = []
    print(f"{'bucket':<22}{'hits':>10}  note")
    print("-" * 78)
    for name, (required, note) in BUCKETS.items():
        hits = counts.get(name, 0)
        flag = ""
        if hits == 0:
            flag = " <-- HOLE (required)" if required else " (optional, unhit)"
            if required:
                holes.append(name)
        print(f"{name:<22}{hits:>10}  {note}{flag}")
    for name in unknown:
        print(f"{name:<22}{counts[name]:>10}  ?? bucket not in the plan")

    if unknown:
        print(f"\nCOVERAGE FAIL: {len(unknown)} unknown bucket(s): {unknown}")
        return 1
    if holes:
        print(f"\nCOVERAGE FAIL: {len(holes)} required hole(s): {holes}")
        return 1
    print("\nCOVERAGE PASS: all required buckets hit "
          f"({len(BUCKETS)} buckets, {len(argv)} logs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
