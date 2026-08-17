#!/usr/bin/env python
"""audit_coverage.py — coverage gate for the KVQ D-020 re-audit
(verif/kvq/audit). Aggregates "COV <name> <n>" lines from the audit run logs
(mutant-run logs excluded) and "COVGEN <name> <n>" from the generator, and
FAILS the build on any required hole.

Required-buckets rationale:
  * >=200 total soft-reset injections, with the LANDING FSM state binned
    hierarchically (dut.state sampled in the ctrl_reset cycle) — every state
    p1..p11 must be hit, the 1-cycle states (STORE/RLOAD/RWAIT/KFEED) via the
    deterministic collision tactics.
  * pending-beat (reset lands in OUTPUT/OFLUSH with a beat held under
    backpressure), mid-token, mid-group and beat-coincident conditions.
  * every reset followed by a bit-exact clean job (clean_v_jobs == resets is
    checked by the TB flow; the gate requires the volume).
  * -0.0 directed content generated at BOTH D=16 and D=64: outlier lanes
    (B-4), non-outlier INT4 path, all-zero channel column (EPS scale), value
    path, +0.0 / negative-subnormal outlier neighbors.
  * A2/D-026: bank-survival reset attacks (records + persistent scale-bank
    row bit-exact ACROSS a soft reset — the guard against wiring
    scale_bank_store to dp_clear) and the SB_OVWR sticky/W1C probe, once per
    KEYG mode-0 run.
"""
import re
import sys
from collections import Counter
from pathlib import Path

REQUIRED = {
    # reset-attack volume + landing-state closure
    "rst_total": 200,
    "rst_land_p1": 3, "rst_land_p2": 3, "rst_land_p3": 2, "rst_land_p4": 2,
    "rst_land_p5": 2, "rst_land_p6": 3, "rst_land_p7": 3, "rst_land_p8": 2,
    "rst_land_p9": 3, "rst_land_p10": 3, "rst_land_p11": 3,
    # conditions
    "rst_pending_beat": 15, "rst_mid_token": 25, "rst_mid_group": 15,
    "rst_beat_collide": 3,
    "rst_collide_store": 2, "rst_collide_rload": 2, "rst_collide_rwait": 2,
    "rst_collide_kfeed": 2,
    # §5-under-reset: bursts crossed by a reset drained EXACTLY (D beats,
    # bit-exact, tlast on D-1)
    "rst_drain_exact": 15,
    # every reset followed by a full bit-exact clean job
    "clean_v_jobs": 200, "clean_k_jobs": 50,
    "audit_reads": 300, "audit_peeks": 300,
    # D-026: 4 KEYG mode-0 runs x N_BANK(6) bank-survival events = 24
    "rst_bank_survive": 24,
    # D-026: bank rows scored per group = clean_k_jobs floor (50) + 2 per
    # bank-survival event (2 x 24 = 48) => 98
    "audit_bank_peeks": 98,
    # D-026: SB_OVWR sticky-set + W1C proven once per KEYG mode-0 run (4 runs)
    "sb_ovwr_sticky": 4, "sb_ovwr_w1c": 4,
    "idle_lie_mon_armed": 1,
    # -0.0 directed content (generator-proven present), D=16 AND D=64
    "negz_value_d16": 1, "negz_value_all_d16": 1, "negz_value_mix_d16": 1,
    "negz_value_d64": 1, "negz_value_all_d64": 1, "negz_value_mix_d64": 1,
    "negz_key_nonoutlier_d16": 1, "negz_col_all_zero_d16": 1,
    "negz_key_nonoutlier_d64": 1, "negz_col_all_zero_d64": 1,
    "negz_outlier_d16": 1, "posz_outlier_d16": 1, "negsub_outlier_d16": 1,
    "negz_outlier_d64": 1, "posz_outlier_d64": 1, "negsub_outlier_d64": 1,
}


def main(paths):
    got = Counter()
    n_logs = 0
    for p in paths:
        p = Path(p)
        if not p.exists() or p.name.startswith("run_mut_"):
            continue
        n_logs += 1
        for line in p.read_text(errors="replace").splitlines():
            m = re.match(r"^\s*(COV|COVGEN)\s+(\S+)\s+(-?\d+)\s*$", line)
            if m:
                got[m.group(2)] += int(m.group(3))
    holes = []
    print(f"==== KVQ AUDIT coverage ({n_logs} inputs) ====")
    for name in sorted(set(REQUIRED) | set(got)):
        need = REQUIRED.get(name, 0)
        have = got.get(name, 0)
        mark = "OK " if have >= need else "HOLE"
        if need == 0:
            mark = "----"
        print(f"  [{mark}] {name:26s} {have:8d} (min {need})")
        if have < need:
            holes.append(name)
    if holes:
        print(f"COVERAGE GATE: FAIL — {len(holes)} required hole(s): {holes}")
        sys.exit(1)
    print("COVERAGE GATE: PASS — all required buckets closed")


if __name__ == "__main__":
    main(sys.argv[1:])
