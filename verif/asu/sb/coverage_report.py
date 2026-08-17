#!/usr/bin/env python3
"""coverage_report.py — aggregate the manual ASU coverage buckets (house
pattern, verif/mxe/sb lineage) printed by tb_asu_softmax_sb.sv and
tb_asu_rmsnorm_sb.sv ("COV <name> <count>") across all run logs and print a
hit/hole table with reachability notes.

Exit status: 1 if any REQUIRED bucket has zero aggregate hits, else 0 —
'make all' therefore gates on coverage closure, not just test pass.
"""
import re
import sys
from collections import defaultdict

# bucket -> (required, reachability note)
BUCKETS = {
    # ── softmax rows ─────────────────────────────────────────────────────────
    "sm_n_eq_1":         (True,  "single-element row (p = 0x8000 exactly 1.0)"),
    "sm_n_eq_2":         (True,  "two-element row"),
    "sm_n_mid":          (True,  "2 < n < SM_ROW_MAX"),
    "sm_n_eq_max":       (True,  "n = SM_ROW_MAX = 1024 (row_buf capacity)"),
    "sm_p_zero":         (True,  "probability quantizes to 0"),
    "sm_p_one":          (True,  "probability = 0x8000 (UQ1.15 top)"),
    "sm_p_mid":          (True,  "0 < p < 1.0"),
    "sm_dup_max":        (True,  "duplicated maximum (equal-to-max path)"),
    "sm_resc_ge2":       (True,  ">= 2 l-rescales (ascending maxima)"),
    "sm_all_equal":      (True,  "all-equal row (n rescale-free adds)"),
    "sm_clamp":          (True,  "exp argument below -16.0 (clamped)"),
    "sm_clamp_boundary": (True,  "exp argument exactly -16384 (floor entry)"),
    "sm_i32_extreme":    (True,  "|score| >= 2^30 (33-bit diff arithmetic)"),
    "sm_neg_scores":     (True,  "negative scores present"),
    "sm_lut_frac":       (True,  "LUT interpolation frac != 0 inside softmax "
                                 "(needs the SCORE_FRAC=10 build; frac 0 "
                                 "makes every exp arg a multiple of 2^10)"),
    "sm_cfg_frac0":      (True,  "SCORE_FRAC=0 build ran (C-5 product config)"),
    "sm_cfg_frac10":     (True,  "SCORE_FRAC=10 build ran (interp-rich cfg)"),
    "sm_rej_direct":     (True,  "oversize reject, last ON overflow beat"),
    "sm_rej_flush":      (True,  "oversize reject drained via ST_FLUSH"),
    # ── softmax mid-op reset phases (run2 hole; hierarchical targeting) ─────
    "sm_rst_collect":    (True,  "reset in ST_COLLECT"),
    "sm_rst_flush":      (True,  "reset in ST_FLUSH (oversize row mid-drain)"),
    "sm_rst_load":       (True,  "reset in ST_LOAD (1-cycle state, targeted)"),
    "sm_rst_div":        (True,  "reset in ST_DIV (mid-division)"),
    "sm_rst_push":       (True,  "reset in ST_PUSH (beat offered to skid)"),
    "sm_rst_drain":      (True,  "reset in ST_DRAIN (post-skid wait)"),
    "sm_rst_random":     (True,  "cycle-count (untargeted) aborts"),
    "sm_rst_idle":       (False, "abort landed after row completed — not a "
                                 "mid-op case (informational)"),
    # ── softmax protocol regimes ─────────────────────────────────────────────
    "sm_bp_full":        (True,  "run with m_ready held high"),
    "sm_bp_random":      (True,  "run with ~75% duty backpressure"),
    "sm_bp_storm":       (True,  "run with ~6% duty backpressure storm"),
    "sm_stall_none":     (True,  "run with full-rate score feed"),
    "sm_stall_random":   (True,  "run with short random feed gaps"),
    "sm_stall_storm":    (True,  "run with bursty stalled feed (<=32 cyc)"),
    # ── rmsnorm rows ─────────────────────────────────────────────────────────
    "rms_d_eq_1":        (True,  "D=1 row (last on first beat)"),
    "rms_d_eq_128":      (True,  "D=RMS_D_MAX=128 (x_buf capacity)"),
    "rms_d_mid":         (True,  "D in {2..64} powers of two"),
    "rms_x_extreme":     (True,  "x contains -128 / +127"),
    "rms_all_zero_x":    (True,  "all-zero row (n2=1, r=8192 sat-free path)"),
    "rms_g_zero_lane":   (True,  "gamma=0 lane with x!=0 (y=0)"),
    "rms_g_extreme":     (True,  "gamma = -32768 / +32767"),
    "rms_tie_odd":       (True,  "RNE tie, odd floor quotient (rounds up)"),
    "rms_tie_even":      (True,  "RNE tie, even floor quotient (holds) — "
                                 "the discriminator vs round-half-up"),
    "rms_y_sat_pos":     (False, "UNREACHABLE at D<=128: max |pre-sat t| is "
                                 "~11.9k < 32767 (measured bound printed by "
                                 "the generator); y_sat clamp is defensive"),
    "rms_y_sat_neg":     (False, "UNREACHABLE — same structural bound"),
    "rms_rej_nonpow2":   (True,  "reject: D not a power of two"),
    "rms_rej_over_direct": (True, "reject: beat 129 carries last"),
    "rms_rej_over_flush":  (True, "reject: >129 beats, drained via ST_FLUSH"),
    # ── rmsnorm mid-op reset phases ──────────────────────────────────────────
    "rms_rst_collect":   (True,  "reset in ST_COLLECT"),
    "rms_rst_flush":     (True,  "reset in ST_FLUSH"),
    "rms_rst_issue":     (True,  "reset in ST_ISSUE (1-cycle rsqrt issue)"),
    "rms_rst_wait":      (True,  "reset in ST_WAIT (rsqrt mid-flight, D-018)"),
    "rms_rst_emit":      (True,  "reset in ST_EMIT (gamma consumption)"),
    "rms_rst_drain":     (True,  "reset in ST_DRAIN (post-skid wait)"),
    "rms_rst_random":    (True,  "cycle-count (untargeted) aborts"),
    "rms_rst_idle":      (False, "abort landed after row completed — not a "
                                 "mid-op case (informational)"),
    # ── rmsnorm protocol regimes ─────────────────────────────────────────────
    "rms_bp_full":       (True,  "run with m_ready held high"),
    "rms_bp_random":     (True,  "run with ~75% duty backpressure"),
    "rms_bp_storm":      (True,  "run with ~6% duty backpressure storm"),
    "rms_stall_none":    (True,  "run with full-rate x feed"),
    "rms_stall_random":  (True,  "run with short random x-feed gaps"),
    "rms_stall_storm":   (True,  "run with bursty stalled x feed"),
    "rms_g_lockstep":    (True,  "gamma delivered lockstep"),
    "rms_g_late":        (True,  "gamma delayed 0..20 cycles per beat "
                                 "(EMIT stalls on the gamma stream)"),
    "rms_g_eager":       (True,  "gamma pumped from row start (parks in the "
                                 "g-skid through COLLECT/ISSUE/WAIT)"),
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

    print(f"{'bucket':<21} {'hits':>7}  {'status':<8} note")
    print("-" * 96)
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
        print(f"{name:<21} {hits:>7}  {status:<8} {note}")
    unknown = set(counts) - set(BUCKETS)
    for name in sorted(unknown):
        print(f"{name:<21} {counts[name]:>7}  {'?':<8} (bucket not in plan)")
    print("-" * 96)
    if holes:
        print(f"COVERAGE: {holes} REQUIRED hole(s)")
        return 1
    print("COVERAGE: all required buckets hit "
          f"({sum(1 for b, (r, _) in BUCKETS.items() if r)} required, "
          f"{len(BUCKETS)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
