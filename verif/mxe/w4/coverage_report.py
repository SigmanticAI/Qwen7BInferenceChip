#!/usr/bin/env python3
"""coverage_report.py — aggregate the manual B3 native-W4 weight-feeder
coverage buckets (house pattern, verif/seam lineage) printed by
tb_wfeed_w4_sb.sv ("COV <name> <count>") across all run logs and print a
hit/hole table with reachability notes.

It additionally reports, from the SAME parsed logs and never from hand-typed
constants (ARCHITECTURE.md:203-205):
  - the job census, from the TB's "TB PASS:" lines (legal / illegal / reset);
  - the B3 headline beat ratio, from the TB's "PERF [...]: consumed=<c>
    emitted=<e>" lines, plus an independent re-check of the weight_codec S6
    identity  emitted == 2*consumed - (emitted & 1)  on every measured job.
    The unqualified "emitted == 2*consumed" is FALSE for legal odd KB*N, so
    the ratio is reported as a RANGE, never as a single "2x" claim.

Exit status: 1 if any REQUIRED bucket has zero aggregate hits, if an unknown
bucket appears, if a log did not reach "TB PASS:", or if a parsed PERF line
violates S6 — else 0. 'make all' therefore gates on coverage closure and on
the headline number's own consistency, not just on test pass.
"""
import re
import sys
from collections import defaultdict

# bucket -> (required, reachability note)
BUCKETS = {
    # ── mode (w4_en latched at job accept, mxe_wfeed_w4.sv:228) ──────────────
    "w4_mode_w4":       (True,  "w4_en = 1: unpack 1 packed beat -> 2 INT8"),
    "w4_mode_pass":     (True,  "w4_en = 0: transparent 1:1 passthrough"),
    # ── S6 tail geometry ────────────────────────────────────────────────────
    "w4_tail_odd":      (True,  "job_beats odd: final packed beat's upper "
                                "half is padding, dropped never emitted"),
    "w4_tail_even":     (True,  "job_beats even: exact 2x expansion"),
    # ── job shapes ──────────────────────────────────────────────────────────
    "w4_beats_1":       (True,  "job_beats = 1 (single emitted beat)"),
    "w4_beats_max":     (True,  "job_beats = BEATS_MAX_TB = 256"),
    "w4_beats_mid":     (True,  "1 < job_beats < 256"),
    # ── unpack numerics (S4: nibble -> sign-extended INT8) ───────────────────
    "w4_code_min":      (True,  "code -8 present (INT4 min, D-001)"),
    "w4_code_max":      (True,  "code +7 present (INT4 max, D-001)"),
    "w4_code_zero":     (True,  "an all-zero emitted beat (amax=0 EPS tile) — "
                                "the case where a dropped sign-extension is "
                                "INVISIBLE, so it must coexist with min/max"),
    "w4_code_neg":      (True,  "a negative code: sign extension is live"),
    # NOTE: there is no saturation/clamp bucket and no fp16 bucket by
    # CONSTRUCTION, not by omission. Realization (A) (module header,
    # docs/design/B3_WEIGHT_PATH.md §2) makes the feeder a PURE UNPACK: the
    # nibble domain is the whole operand domain (16 codes, all covered by
    # min/max/zero/neg above) and the group scale never enters this datapath,
    # so no rounding, no clamp and no scale sideband exist here to cover.
    # ── job legality (§3-style conservative reject) ──────────────────────────
    "w4_ill_beats0":    (True,  "job_beats = 0 reject (§3)"),
    "w4_ill_beats_big": (True,  "job_beats > BEATS_MAX reject (§3)"),
    # ── mid-operation reset (ARCHITECTURE.md:189-190, REQUIRED per block) ────
    "w4_rst_run":       (True,  "mid-op reset in ST_RUN (targeted)"),
    "w4_rst_wait":      (True,  "mid-op reset in ST_WAIT (post-skid drain)"),
    "w4_rst_random":    (True,  "cycle-count (untargeted) aborts"),
    # REACHABILITY: the reset vectors steer the FSM to ST_RUN (tphase=1) and
    # ST_WAIT (tphase=2); the untargeted aborts fire after a fixed cycle count
    # while a job is in flight. Landing in ST_IDLE therefore means the abort
    # arrived after the job had already retired — INCIDENTAL, not a mid-op
    # case, and not something the vector set can be made to guarantee without
    # asserting on cycle-accurate arrival. It is recorded, not required; the
    # mid-op requirement is discharged by rst_run / rst_wait / rst_random.
    "w4_rst_idle":      (False, "abort landed in ST_IDLE — the job had "
                                "already completed, so not a mid-op case "
                                "(informational)"),
    # ── protocol adversaries (§5 stream contract) ───────────────────────────
    "w4_bp_full":       (True,  "run with w8_ready held high"),
    "w4_bp_random":     (True,  "run with ~75% duty backpressure"),
    "w4_bp_storm":      (True,  "run with ~6% duty backpressure storm"),
    "w4_stall_none":    (True,  "run with full-rate packed-W4 feed"),
    "w4_stall_random":  (True,  "run with short random feed gaps"),
    "w4_stall_storm":   (True,  "run with bursty stalled feed (<=32 cyc)"),
    # ── PERF (weight_codec S6, docs/OPTIMIZATION.md:67) ─────────────────────
    "w4_perf_halved":   (True,  "w4 job with EVEN beats: emitted = 2*consumed"),
    "w4_perf_oddtail":  (True,  "w4 job with ODD beats: emitted = "
                                "2*consumed - 1 (the corrected S6 form)"),
    "w4_perf_pass":     (True,  "passthrough job: emitted = consumed"),
}

COV_RE = re.compile(r"^COV\s+(\S+)\s+(\d+)\s*$")
PASS_RE = re.compile(
    r"^TB PASS:\s+(\d+) legal jobs,\s+(\d+) illegal jobs,\s+"
    r"(\d+) mid-op resets,\s+(\d+) errors")
PERF_RE = re.compile(
    r"^PERF\s+\[(.+?)\]:\s+consumed=(\d+)\s+emitted=(\d+)")


def main(argv) -> int:
    counts = defaultdict(int)
    jobs = dict(legal=0, illegal=0, resets=0, errors=0)
    runs = 0
    no_pass = []
    perf = []            # (name, consumed, emitted)
    for path in argv:
        seen_pass = False
        with open(path) as f:
            for line in f:
                m = COV_RE.match(line)
                if m:
                    counts[m.group(1)] += int(m.group(2))
                    continue
                m = PASS_RE.match(line)
                if m:
                    seen_pass = True
                    runs += 1
                    jobs["legal"] += int(m.group(1))
                    jobs["illegal"] += int(m.group(2))
                    jobs["resets"] += int(m.group(3))
                    jobs["errors"] += int(m.group(4))
                    continue
                m = PERF_RE.match(line)
                if m:
                    perf.append((m.group(1), int(m.group(2)), int(m.group(3))))
        if not seen_pass:
            no_pass.append(path)

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

    # ── job census (parsed from the TB's own PASS lines) ─────────────────────
    print()
    print("JOBS (parsed from 'TB PASS:' lines — never hand-written)")
    print("-" * 78)
    print(f"  runs completed      : {runs} of {len(argv)} log(s)")
    print(f"  legal jobs          : {jobs['legal']}")
    print(f"  illegal jobs        : {jobs['illegal']}")
    print(f"  mid-op reset jobs   : {jobs['resets']}")
    print(f"  TB-reported errors  : {jobs['errors']}")
    for path in no_pass:
        print(f"  INCOMPLETE          : {path} never reached 'TB PASS:'")

    # ── PERF: the B3 headline number, measured (weight_codec S6) ─────────────
    print()
    print("PERF (parsed from 'PERF [...]: consumed/emitted' — "
          "ARCHITECTURE.md:203-205)")
    print("-" * 78)
    s6_bad = []
    if not perf:
        print("  no PERF lines parsed (no w4_en=1 job in these logs)")
    else:
        ratios = [(e / c, n, c, e) for (n, c, e) in perf]
        lo = min(ratios)
        hi = max(ratios)
        tot_c = sum(c for (_n, c, _e) in perf)
        tot_e = sum(e for (_n, _c, e) in perf)
        odd = sum(1 for (_n, _c, e) in perf if e & 1)
        for (n, c, e) in perf:
            if e != 2 * c - (e & 1):
                s6_bad.append((n, c, e))
        print(f"  w4 jobs measured    : {len(perf)} "
              f"({len(perf) - odd} even, {odd} odd-tail)")
        print(f"  emitted/consumed min: {lo[0]:.4f}x  "
              f"[{lo[1]}: consumed={lo[2]} emitted={lo[3]}]")
        print(f"  emitted/consumed max: {hi[0]:.4f}x  "
              f"[{hi[1]}: consumed={hi[2]} emitted={hi[3]}]")
        print(f"  aggregate           : {tot_e} emitted / {tot_c} consumed "
              f"= {tot_e / tot_c:.4f}x")
        print(f"  S6 identity emitted == 2*consumed - (emitted & 1): "
              f"{len(perf) - len(s6_bad)}/{len(perf)} jobs hold")
        for (n, c, e) in s6_bad:
            print(f"  S6 VIOLATION        : [{n}] consumed={c} emitted={e}")

    # ── verdict / exit-code discipline ──────────────────────────────────────
    if unknown:
        print(f"\nCOVERAGE FAIL: {len(unknown)} unknown bucket(s): {unknown}")
        return 1
    if no_pass:
        print(f"\nCOVERAGE FAIL: {len(no_pass)} log(s) without 'TB PASS:': "
              f"{no_pass}")
        return 1
    if s6_bad:
        print(f"\nCOVERAGE FAIL: {len(s6_bad)} PERF line(s) violate the "
              "weight_codec S6 beat identity")
        return 1
    if holes:
        print(f"\nCOVERAGE FAIL: {len(holes)} required hole(s): {holes}")
        return 1
    print("\nCOVERAGE PASS: all required buckets hit "
          f"({len(BUCKETS)} buckets, {len(argv)} logs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
