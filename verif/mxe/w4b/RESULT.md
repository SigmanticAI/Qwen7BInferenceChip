# W4-B feeder verification (D-031) — RESULT

**Date:** 2026-07-24 (evidence log re-captured complete 2026-07-26) · **DUT:** `rtl/mxe/mxe_wfeed_w4b.sv` +
`rtl/mxe/w4b_fp_pkg.sv` (G=32 ship, G=16 buildable) · **Branch:**
`comp/w4b-feeder` · **Contract:** docs/design/W4B_FEEDER.md
**Golden arbiter:** `weight_codec.wfeed_w4b_to_i8` (the given-scale (B)
chain; reduction-identity-gated to the landed `wfeed_w4_to_i8("B")`).
**Tools:** Verilator 5.044 (pinned local), Python 3 + numpy.

## Verdict: PASS — bit-exact vs the golden at every gate

Full log (complete, untruncated — `make -C verif/mxe/w4b logged` from clean,
exit 0): [`logs/run_all_full.log`](logs/run_all_full.log). The gate banners,
verbatim from that log:

```
W4B PKG GATE: 31024 vectors, ALL PASS
TB PASS: 18 jobs, 3 illegal, 0 resets, 0 errors (bp=1 stall=1 gs=0 seed=51)
TB PASS: 18 jobs, 3 illegal, 0 resets, 0 errors (bp=0 stall=0 gs=0 seed=52)
TB PASS: 120 jobs, 7 illegal, 0 resets, 0 errors (bp=1 stall=1 gs=1 seed=53)
TB PASS: 120 jobs, 7 illegal, 0 resets, 0 errors (bp=2 stall=2 gs=2 seed=54)
TB PASS: 120 jobs, 7 illegal, 0 resets, 0 errors (bp=1 stall=0 gs=1 seed=55)
TB PASS: 7 jobs, 0 illegal, 7 resets, 0 errors (bp=1 stall=1 gs=0 seed=56)
W4B EXHAUSTIVE SWEEP: 4063104 operand points, ALL PASS
W4B MUTATION GATE: 5/5 mutants detected — checkers proven live
W4B SUITE: all gates green (pkg, 6 regimes, exhaustive sweep, mutants)
```

## What was proven

- **403 job runs / 0 errors** across 6 adversary regimes (backpressure
  none/75%/storm × packed-feed stalls × gs upfront/late/interleaved),
  incl. tile/stripe/tiny s8 modes, odd tails, passthrough, all-zero, and
  a hand-built extreme job (forced code −8 + max-normal group scales vs
  the min-subnormal s8 — operands unreachable from amax-derived
  compression, exactly the region where MW1's overflow lives).
- **The REQUIRED §8.4 exhaustive sweep:** every positive finite fp16
  group scale (31,743, subnormals included) × all 16 codes × an 8-value
  s8 panel = 4,063,104 operand points, bit-exact vs the golden
  per-element rule via the comb twin (the pipeline registers the SAME
  functions — cq_quant_pipe precedent).
- **Mid-op resets:** RUN-phase, DRAIN-phase and cycle-abort (7 total),
  clean recovery + clean jobs after; divider pipes flushed per contract.
- **Legality:** beats==ceil(K/8)·N cross-check, N/K ranges, zero-state
  rejects (pulse + sticky) — 17 illegal-job executions.
- **Checker credibility 5/5:** shortcut-removal (killed by the sh=39
  extreme job — the 41-bit width proof has teeth), gid cadence, sideband
  desync, odd-tail hang (caught as timeout), legality removal.
- **Meta-lesson banked:** the first `make all` printed "all green" while
  MW1 was NOT detected — `tee` masked the mutation gate's exit (missing
  pipefail). Fixed; the gate can no longer lie. Same class as S12's
  BSD-sed silent no-op.
- **Evidence-log fix (audit batch #2, 2026-07-26):** the original
  `logs/run_all_2026-07-24.log` was `| tail`-stripped — only 2 of the 10
  banners above (mutation gate + suite) were actually in it, so the
  "Verbatim" claim was false for the other 8 (pkg, 6× TB PASS, sweep). Fixed
  with the `logged` target, which assembles the complete per-gate output;
  the re-run reproduced every count bit-identically (deterministic).

## Scope

- TB builds with `-Wno-fatal` (width-cast style warnings in TB tasks —
  the DUT itself is `-Wall` clean, apex_pkg waiver class only). Polish
  chip-filed.
- Integration wiring (route/CSR, apex_top, walker descriptor fields,
  host `prepare --w4-direct`) = contract stage-4 notes, executed by the
  combine session. W4B remains CSR-disabled until the full matrix reruns
  with it instantiated.
