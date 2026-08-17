# TIP — Token Importance & Precision (D-011 / D-017 rewrite) — smoke suite

**Verdict: PASS** — 13,290 golden-checked tiles across 7 run configurations,
0 mismatches; frozen-143 upstream equivalence proven mechanically; framing
guard (V0.3 F-2) regression passes; 3/3 mutation checks caught by the TB.

Implements ARCHITECTURE.md §1 TIP row, D-011 (SCORE_WIDTH=32, per-layer
THRESHOLD_REG), D-017 (THRESHOLD datapath rewrite: programmable shift-add,
CMP_W = SUM_W + clog2(T_MAX+1), tile-length framing guard), §5/D-006 streams
via `stream_skid` (D-019 vendored), D-012 (Verilator-first, SVA compiled).

- RTL: `rtl/tip/tip_decide.sv` (ratio test, T ∈ 1..31 via 5-tap conditional
  shift-add tree, 0 clamped to 1; F-2 guard: >N scores/tile → `frame_err`
  pulse + sticky, tile aborted, clean after), `rtl/tip/tip_importance.sv`
  (128 × 16-bit saturating accumulators; increment = decision-weighted
  log-magnitude bucket, defined in the module header; 2-threshold
  `kvq_tier_e` suggestion; combinational CSR read port),
  `rtl/tip/tip_top.sv` (skid on score-in and decision-out; lossless feed
  gating so no decision beat can ever be dropped under backpressure; §5 busy).
- Golden model: `verif/tip/smoke/gen_tip_vectors.py` (bit-exact mirror,
  dual-oracle checked vs the unbounded math rule `max·N > T·sum` per tile).
- Tools: Verilator 5.044 (`--binary --timing --assert -Wall`, waivers scoped
  to frozen `apex_pkg.sv` only), Python 3.11.5 + numpy.
- Reproduce: `make smoke` in `verif/tip/` (~40 s from clean).

## Evidence (fresh full run, 2026-07-07)

| Run | Config | Result |
|---|---|---|
| lint | `tip_top` -Wall, zero waivers on new RTL | LINT CLEAN |
| vectors | 13,290 tiles, golden==math dual oracle | all cross-checks pass |
| run_replay | frozen V0.3 143 tiles, N=4096, T=10, gaps+bp | 143/143, fails=0 |
| run_sweep_t1 | 3,024 tiles, N=64, T=1, gaps+bp | fails=0 |
| run_sweep_t5 | 3,024 tiles, N=64, T=5, gaps+bp | fails=0 |
| run_sweep_t31 | 3,024 tiles, N=64, T=31, gaps+bp | fails=0 |
| run_clamp_t0 | same tile set, `+threshold=0` vs T=1 golden | fails=0 (clamp proven) |
| run_imp | 1,051 directed tiles: tier == lo/hi exact, saturation to 0xFFFF + hold, inc=0, INT8 weight | fails=0, full 128-block readout checked |
| run_frame | N+1-score tile (overrun on s_last) and N+5-score tile (abort+drain): 2 err pulses, 0 decision beats, sticky set/clear, imp NOT updated by erroring tiles, clean decisions after | fails=0 |

Every vector run checks, per decision beat and in order: `d_fp16`, `d_tier`
(post-update importance suggestion) and `d_blk`; at end-of-run: decision
count == tiles, zero frame errors, `busy` deasserted, and the FULL 128-block
importance state + `rd_tier` via the CSR read port. The §5 stream stability
and F-2 sticky semantics are checked every cycle by the bound
`verif/tip/smoke/tip_sva.svh` (the house `apex_stream_sva.svh` pack is
descriptor/job-shaped and cannot bind onto a descriptor-less stream block;
its §5 core properties were carried over — see the checker header).

### Upstream equivalence at T=10 (frozen 143)

`gen_tip_vectors.py` loads the V0.3 frozen vectors
(self-generated golden vectors)
into `build/vectors.log`), sign-extends the int8 scores to W=32, and ASSERTS
the APEX golden decision (programmable datapath, T=10) equals the upstream
frozen expected bit for all 143 tiles — then the RTL replays them under
Verilator with random gaps + decision backpressure: 143/143.

### Mutation checks (TB kill confirmation, house pattern)

| Mutation | Expected catch | Result |
|---|---|---|
| `lhs > rhs` → `lhs >= rhs` (tie flip) | sweep directed tie tiles | CAUGHT (first at tile 1) |
| saturation → wrap-to-zero on overflow | imp_directed saturation tiles | CAUGHT (at the saturating tile) |
| framing guard never fires | frame regression | CAUGHT (10 failures, incl. the silent-wrap decision corruption the guard exists to prevent) |

All mutations reverted; final run is from pristine RTL (`make smoke` clean).

## Notes / boundaries

- Overflow-freedom of the CMP_W datapath is a static bound (see
  `tip_decide.sv` header) now HARDWARE-enforced by the F-2 guard (≤ N
  scores/tile), not an integration promise as in V0.3.
- Threshold values are covered at T ∈ {0(clamp),1,4,5,10,31}; the shift-add
  tree is exercised on all 5 tap positions (1=b00001, 5=b00101, 10=b01010,
  31=b11111). Not exhaustive over all 31 values.
- Simulation-only: nothing about synthesis/timing.
- Mid-operation reset test (Layer-1 requirement) is NOT in this smoke suite —
  owed to the independent TIP verification suite alongside constrained-random
  with per-seed reproduction (this is the implementer's smoke, mirroring
  `verif/mxe/smoke` scope).
