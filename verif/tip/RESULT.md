# TIP independent verification — RESULT

**Date:** 2026-07-07 · **DUT:** `rtl/tip/` @ working tree (tip_top + tip_decide + tip_importance, xbr/stream_skid)
**Verifier:** independent pass (does NOT trust the implementer's smoke claims; their smoke report moved to `verif/tip/smoke/RESULT.md`)
**Tools:** Verilator 5.044 `--binary --timing --assert`, `-Wall` · Python 3.11.5 + NumPy
**Golden:** independent model in `verif/tip/sb/gen_tip_sb_vectors.py` (dual-oracle: unbounded integer math vs the RTL's fixed widths), arbitrated against the V0.3 frozen upstream vectors (D-013); `golden/apex_golden/` has no TIP-rule mirror — the block-specific rule was re-derived from ARCHITECTURE.md §1/C-5/D-011/D-017 and the RTL module-header contracts, then cross-checked against the frozen 143

## Verdict: PASS — 0 DUT bugs found

4,247 golden-checked tiles in the independent suite (2,787 mine across 7 run
configurations + the implementer's 13,290-tile smoke reproduced from clean),
36 malformed/overrun frames, 24 phase-targeted mid-operation resets (all 6
phases — the Layer-1 hole the smoke left open), coverage gate 67/67 required
buckets, checker credibility proven by 4/4 detected mutations.

## 1. Deliverables (`make -C verif/tip all` runs everything, exit 0)

| File | What |
|---|---|
| `verif/tip/sb/gen_tip_sb_vectors.py` | Independent golden model + vector generator; re-proves frozen-143 upstream equivalence independently at generation time |
| `verif/tip/sb/tb_tip_sb.sv` | Vector-driven scoreboard TB (verif/mxe/sb pattern): T/F/C/R/E records, per-tile programmable threshold, LFSR backpressure + feed-gap adversaries, phase-targeted mid-op reset, full 128-block CSR readout checks, $fatal watchdog |
| `verif/tip/sb/tip_sb_sva.svh` | TIP conservation/busy/F-2 checker (16 assertions incl. a produced-vs-delivered decision transaction model), bound white-box into `tip_top` |
| house `verif/common/apex_stream1_sva.svh` | REUSED §5 stability core, bound onto BOTH TIP streams (the full `apex_stream_sva.svh` pack is descriptor/job-shaped and cannot bind to a descriptor-less block — checked, same conclusion as the implementer) |
| `verif/tip/sb/coverage_report.py` | Manual-bucket aggregation with reachability notes; **gates `make all`** (67 required buckets) |
| `verif/tip/sb/mutate.py` + Makefile `mutants` | 4 scratch-copy RTL mutations; `make all` **fails unless every mutant is caught** |
| `verif/tip/Makefile`, `verif/tip/sb/Makefile` | `all = smoke + sb` (lint, vectors, 2 builds, 7 runs, coverage gate, mutation gate) |

## 2. Reproduction of the implementer's claims (verified, not trusted)

`make -C verif/tip/smoke clean && make smoke`: **exit 0**, verbatim:

```
LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)
  provenance scores.hex: sha256 191ecb48e6456b8d0bbafcf5f9ca3927a3700f0d67320d09796a8c165e48a294
  provenance expected.hex: sha256 af620b05a1644b020dd190b7a9ba55106053f5083a9decda8dbfad1c97a395ad
143-TILE SELF-CHECK PASSED: APEX golden (W=32, programmable T=10) == our golden expected for all 143 tiles
TOTAL tiles: 13290
 ALL TESTS PASSED   (x7 runs: replay_t10, sweep_t1/t5/t31, clamp_t0, imp, frame)
```

All three headline claims hold: (1) lint clean, zero waivers on new RTL;
(2) 13,290 tiles, dual-oracle; (3) frozen-143 equivalence + N=4096 RTL replay.
Additionally re-proved (2)/(3) **independently**: my own golden model asserts
the same verdict as the frozen upstream `expected.hex` for all 143 tiles
(same sha256s as above printed in `sb/build/vectors.log`), and a 24-tile
frozen subset is round-tripped through the RTL in my own N=4096 build.

## 3. Independent suite — verbatim results (`make -C verif/tip/sb all`, exit 0)

```
LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)
INDEPENDENT frozen-143 re-proof: my golden @ T=10 == upstream expected for all 143 tiles
TOTAL tiles=2787 frames=36 resets=24
TB PASS: 1200 tiles, 3 overrun frames, 0 mid-op resets, 1 clears, 1200 decisions checked (+0 sacrificial discarded), 0 errors (N=64 bp=1 gap=1 seed=101)
TB PASS: 1200 tiles, 3 overrun frames, 0 mid-op resets, 1 clears, 1200 decisions checked (+0 sacrificial discarded), 0 errors (N=64 bp=0 gap=0 seed=102)
TB PASS: 1216 tiles, 24 overrun frames, 0 mid-op resets, 2 clears, 1216 decisions checked (+0 sacrificial discarded), 0 errors (N=64 bp=1 gap=1 seed=103)
TB PASS: 260 tiles, 7 overrun frames, 0 mid-op resets, 0 clears, 260 decisions checked (+0 sacrificial discarded), 0 errors (N=64 bp=2 gap=2 seed=104)
TB PASS: 260 tiles, 7 overrun frames, 0 mid-op resets, 0 clears, 260 decisions checked (+0 sacrificial discarded), 0 errors (N=64 bp=0 gap=2 seed=105)
TB PASS: 72 tiles, 0 overrun frames, 24 mid-op resets, 0 clears, 72 decisions checked (+0 sacrificial discarded), 0 errors (N=64 bp=1 gap=0 seed=106)
TB PASS: 39 tiles, 2 overrun frames, 0 mid-op resets, 0 clears, 39 decisions checked (+0 sacrificial discarded), 0 errors (N=4096 bp=1 gap=1 seed=107)
COVERAGE: all required buckets hit (67 required, 67 total)
MUTATION GATE: 4/4 mutants caught
```

Run configurations (`bp` = decision backpressure none/75%/storm-6%; `gap` =
score-feed gaps none/short/bursty-24): directed, directed-full-rate, random,
**backpressure storm**, **stalled feeds**, **mid-op reset**, **N=4096**.

Directed content (all vs the independent golden, every decision beat checked
in order for `d_fp16` + `d_tier` + `d_blk`): exact LHS==RHS ties and ±1
straddles for **every** THRESHOLD_REG value 0..31 (0 = the D-017 clamp);
|INT32_MIN| = 2^31 exact ties at T∈{1,2,4,8,16}; INT32_MAX floor straddles;
all-zero / all-MIN / all-MAX / alternating tiles; spikes on zero background;
len∈{1, short, N} incl. all-equal short-tile ties; importance: exact
acc==imp_lo and acc==imp_hi boundary hits (>= semantics), saturation to
0xFFFF + hold (no wrap), inc==0 (bucket-0) holds, INT8-weighted (×1)
updates; CSR `imp_clear` mid-run with full pre-clear 128-block compare;
final full 128-block `rd_data`+`rd_tier` readout in every run.

Random content: 1,216 seeded tiles (rng 20260707), thresholds grouped over
all 32 values, lengths 1/2..63/64, five magnitude kinds (full-range int32,
small, log-spread, spike-contaminated, tie-targeted ±2), hot-block
concentration so all three KVQ tiers are crossed, 24 interleaved overrun
frames, 2 mid-run clears.

Malformed framing (V0.3 F-2), 36 overrun tiles total: overrun exactly ON the
s_last beat (N+1) and mid-tile aborts (up to N+33 and 4096+9): each produced
exactly ONE `frame_err` pulse, sticky set, NO decision beat, NO importance
update; sticky cleared via `frame_err_clear` and re-checked after every
frame; next tiles decide correctly (clean-state proof).

Mid-op reset (first for this block — the smoke explicitly owed it): 24
phase-TARGETED resets, 4 rounds × 6 phases, hierarchically observed and
asserted **while the phase is active**: (1) beat in input skid, (2) engine
mid-tile, (3) during F-2 abort drain, (4) `dec_pend` decision in flight,
(5) decision hold register occupied, (6) output skid full (2 beats, forced
`d_ready=0`). After every reset: busy=0, d_valid=0, s_ready restored, sticky
cleared, all 128 accumulators zero, and the immediately following tie/spike/
random tiles decide bit-exact. Sample log:

```
RESET cyc=789:  mid-op reset in phase 3 (in_skid=1 cnt=64 abort=1 dec_pend=0 hold=0 oskid=0)
RESET cyc=1152: mid-op reset in phase 4 (in_skid=1 cnt=0 abort=0 dec_pend=1 hold=0 oskid=0)
RESET cyc=1515: mid-op reset in phase 5 (in_skid=1 cnt=0 abort=0 dec_pend=0 hold=1 oskid=1)
```

## 4. SVA (bound, compiled in every build)

- REUSE: `apex_stream1_sva` (house §5 stability core) on **both** streams —
  no valid retraction, payload stable while `valid && !ready`.
- `tip_sb_sva.svh` (16 assertions, white-box bind into `tip_top`): decision
  conservation model (produced vs post-skid delivered: capacity ≤3, no
  orphan/dup, pending⇒busy); hold register never overwritten (the feed-gate
  guarantee, D-006 no-drop); feed gated after every s_last consume; busy
  composition (input skid / engine / dec_pend / hold / post-skid beat each
  ⇒ busy; !busy ⇒ nothing in flight); `d_tier`≠3 encoding legality; F-2
  pulse-1-cycle / sticky set / hold / clear semantics; no `frame_err`
  without a consumed beat.

## 5. Coverage (aggregated over all 7 runs — `sb/build/coverage.txt`)

67/67 required buckets HIT, zero waived holes: thr_t0..thr_t31 (all 32
threshold values through the shift-add tree), decision both verdicts, exact
tie + off-by-one, INT32_MIN/MAX/zero tiles, len 1/short/full, importance
inc0/saturation/eq-lo/eq-hi, all three tiers observed on the decision
stream, frame on-last + abort classes, sticky-clear, imp_clear, all 6 reset
phases, all 3 backpressure × all 3 gap regimes, both N builds (64, 4096).
Reachability notes live in `coverage_report.py`.

## 6. Checker mutation evidence (scratch copies under `sb/build/mut_*`; `rtl/` untouched)

| # | Mutation (kind) | Caught by | Verbatim |
|---|---|---|---|
| 1 | `tie_ge`: ratio test `>` → `>=` (TIP threshold/tie off-by-one — the block-specific contract violation) | scoreboard, first tie tile | `FAIL cyc=117 tile#0: got fp16=1 ... exp fp16=0` |
| 2 | `skid_xor`: stream_skid direct path XORs payload bit 0 (data corruption in the stream fabric) | scoreboard | `FAIL cyc=230 tile#1: got fp16=0 ... exp fp16=1` |
| 3 | `sat_wrap`: importance saturation clamp removed → 16-bit wrap (data corruption) | tier compare on the d beat + CSR spot check | `FAIL cyc=21531 [spot]: importance[77] rd_data=0 exp=65535` |
| 4 | `busy_hole`: `d_valid` dropped from busy (§5 busy contract) | **bound SVA** under storm | `%Error: tip_sb_sva.svh:102: ... ap_busy_dvalid: [SVA §5] output beat pending (post-skid) but busy low` |

`make mutants` (part of `make all`) fails the build unless all 4 are caught.

## 7. Observations / challenges (not bugs; recorded for integration)

1. **Short tiles use N, not the actual length** (`LHS = max << log2(N)`
   regardless of len): a len-L<N tile is FP16 whenever `max·N > T·sum`, which
   biases partial tiles toward FP16 (a len-1 nonzero tile is FP16 at every
   legal T ≤ 31 since 64 > T). ARCHITECTURE.md's `max·N > THRESHOLD·sum`
   doesn't pin N's meaning for partial tiles; the RTL matches the upstream
   fixed-N semantics and documents it. Verified as documented — but sequence
   tails at KV-length edges will skew FP16; SEQ/KVQ integration should know.
2. **The importance update rule is implementer-defined** (log2-magnitude
   bucket, ×2 FP16 weight, two >= thresholds): ARCHITECTURE.md §1 only says
   "saturating per-block importance accumulators driving tier select". No
   conflict — but the rule deserves a decision-register entry (D-020) since
   KVQ tier behavior now depends on it.
3. **threshold is sampled combinationally on the s_last-consume cycle**: a
   CSR write racing that exact cycle decides the tile with the new value.
   Documented quasi-static assumption (D-011); CSR should write between
   tiles, worth an integration assert.
4. **s_blk is sampled only on the s_last beat** (documented); mid-tile blk
   variation is silently ignored. My TB drives it constant per tile, per
   convention.
5. **kvq_tier_e occupies 2 bits with 3 legal values**; `d_tier==3` is
   unreachable from `tier_of` (SVA-checked here) but downstream decoders
   should still treat 3 defensively.

## 8. Honest boundaries

- Simulation-only (Verilator); no synthesis/timing, no Icarus cross-check.
- All 32 threshold values exhaustive at N=64; at N=4096 only T∈{1,10,31}
  (+ frozen T=10). Overflow-freedom at N=4096 rests on the static bound
  proof + the F-2 guard, both verified in behavior but not swept per-T.
- `imp_clear` concurrent with an in-flight update (clear-wins arbitration in
  `tip_importance`) is not explicitly raced — my clears run on a drained
  pipe (the CSR programming model per §7).
- My golden implements the same documented rule as the implementer's (both
  dual-oracled against unbounded math); true independence anchors are the
  frozen upstream vectors (independent re-proof passed) and ARCHITECTURE.md.
