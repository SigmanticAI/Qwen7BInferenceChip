# MXE independent verification — RESULT

**Date:** 2026-07-07 · **Addendum §8 (D-004/D-005 closure): 2026-07-09** ·
**DUT:** `rtl/mxe/` @ working tree (mxe_top + ctrl/array/pe/requant/buf/cfg_pkg, xbr/stream_skid)
**Verifier:** independent pass (does NOT trust the implementer's smoke claims)
**Tools:** Verilator 5.044 `--binary --timing --assert`, `-Wall` · Python 3.11.5 + NumPy
**Golden arbiter:** `golden/apex_golden/compute.py` (`gemm_i8`, `requant_i32_to_i8`) per C-2/C-3

## Verdict: PASS — 0 DUT bugs found

836 legal jobs (directed + ≥520 seeded random), 65 illegal descriptors, 35
mid-operation resets, across 6 protocol configurations, all bit-exact against
the golden model with the §5/D-006 SVA pack bound and firing on every cycle.
Checker credibility proven by 3 detected mutations (§6).

## 1. Deliverables

| File | What |
|---|---|
| `verif/common/apex_stream_sva.svh` | Reusable §5/D-006 SVA pack (17 concurrent assertions + transaction model), bound into `mxe_top`, compiled under `--assert` in every build |
| `verif/mxe/sb/gen_mxe_vectors.py` | Vector generator; mirrors the RTL's resident-accumulator semantics (64×8 INT32 file, clr rules, lane masking, INT32 wraparound on chains); golden model is the arbiter |
| `verif/mxe/sb/tb_mxe_sb.sv` | Vector-driven scoreboard TB (V0 pattern): J/I/X records, configurable backpressure + feed-stall adversaries, phase-targeted mid-op reset, per-cycle FAIL traces, $fatal watchdog |
| `verif/mxe/sb/coverage_report.py` | Manual-bucket aggregation (run2 pattern) with reachability notes; **gates `make all`** on required-bucket closure |
| `verif/mxe/sb/Makefile`, `verif/mxe/Makefile` | `make -C verif/mxe all` = implementer smoke + full independent suite |

## 2. Reproduction of the implementer's claims (verified, not trusted)

- `make -C verif/mxe/smoke smoke` from clean: **exit 0**, verbatim:
  ```
  LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)
  MXE SMOKE: ALL TESTS PASSED
    T1 WS 4x8x8 requant bit-exact (RNE tie + sat8 exercised)
    T2 2 illegal descriptors rejected, zero side effects
    T3 OS accumulate chain bit-exact after each descriptor
    T4 WS 4x13x5 multi-pass bit-exact (tail+column masking)
  ```
- Vector provenance: re-ran their `gen_vectors.py` against the golden model and
  diffed every literal against the TB constants: `VECTORS MATCH GOLDEN REGEN`.

## 3. Independent suite — verbatim results (`make -C verif/mxe all`, exit 0)

```
LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)
TB PASS: 48 legal jobs, 13 illegal descriptors, 0 mid-op resets, 0 errors (bp=1 stall=1 seed=1)
TB PASS: 48 legal jobs, 13 illegal descriptors, 0 mid-op resets, 0 errors (bp=0 stall=0 seed=2)
TB PASS: 520 legal jobs, 32 illegal descriptors, 0 mid-op resets, 0 errors (bp=1 stall=1 seed=3)
TB PASS: 90 legal jobs, 10 illegal descriptors, 0 mid-op resets, 0 errors (bp=2 stall=2 seed=4)
TB PASS: 90 legal jobs, 10 illegal descriptors, 0 mid-op resets, 0 errors (bp=0 stall=2 seed=5)
TB PASS: 40 legal jobs, 0 illegal descriptors, 35 mid-op resets, 0 errors (bp=1 stall=0 seed=6)
COVERAGE: all required buckets hit (38 required, 40 total)
```

Run configurations (`bp` = result backpressure none/75%/storm-6%; `stall` =
feed gaps none/short/bursty-32): directed, directed-full-rate, random,
**backpressure storm** (the V0.2 lesson), **stalled feeds**, **mid-op reset**.

Directed content: identity (B=I); all −128 at K=K_MAX (acc=+2²⁵) and the
−128×127 negative twin; saturation both signs through the epilogue; exact RNE
half-to-even ties (both parities, both signs, shift- and scale-driven);
M/K/N edges incl. 1, M=64, K=2048 (256 passes), 64×2048×8 full-capacity tile,
K tails 7/9/13/15/17/23/33/1023, N=1..8 sweep; OS chains ×4, tail chains,
requant-on-final-link, OS acc=1 directly after WS (residency retention);
13 illegal descriptors interleaved between legal jobs (D-019 regression —
every following job checked bit-exact). Padding lanes carry generator
garbage (not zeros) so DUT masking is what is proven.

Random content: 520 seeded jobs (`seed=0xA9EC2026`), dims weighted across
1..64 × 1..~300 (plus K=2048 every 97th) × 1..8, WS/OS 65/35, acc chains onto
arbitrary prior resident state (incl. shape-mismatched chains — modeled with
INT32 wraparound), scale 0..65535, shift 0..31, 32 interleaved illegals.

Mid-op reset (the run2 hole): 30 phase-TARGETED resets (5 shapes × all 6 FSM
phases S_INGEST..S_WAIT_DONE, TB observes `u_ctrl.state` and resets in-phase)
plus 5 random-cycle aborts of OS-accumulate jobs. After every reset: busy=0,
sticky cleared, desc_ready restored, zero stray beats/dones over a 10-cycle
window, and the immediately following job (incl. a fresh OS chain) bit-exact.
All 35 logged, e.g.:
```
RESET [X5 16x16x8 abort=0 phase=5] cyc=1224: mid-op reset in ctrl phase 5
RESET [X6 16x16x8 abort=0 phase=6] cyc=1608: mid-op reset in ctrl phase 6
TB PASS: 40 legal jobs, 0 illegal descriptors, 35 mid-op resets, 0 errors (bp=1 stall=0 seed=6)
```

## 4. SVA pack (bound, always compiled)

17 assertions in `apex_stream_sva.svh` — §5 data-stability on all four
streams; D-006 done⇒all-M-beats-accepted-post-skid (transaction-model count);
no-dup/overrun; `last` exactly on the final beat; no result beat without a
legal job in flight (the D-019 stale-beat check); one done per job; done is a
1-cycle pulse; desc_ready gated while in flight and in the done cycle; busy
high throughout a job and low at done; desc_error only on a real handshake,
with no busy/done side effects; sticky sets and holds until reset. Verilator
subset only (simple `|->`/`|=>`, `$stable`, `$past`).

## 5. Coverage (aggregated over all 6 runs — `sb/build/coverage.txt`)

All 38 required buckets HIT; the 2 non-required are documented:
`rst_wait_done` (5 hits — targeted; 1–2-cycle window) and `rst_idle`
(0 — a reset from idle is not a mid-op case; informational only).
Highlights: mode_ws 538 / mode_os 298 / os_acc1 143 / rq_on 446 /
m_eq_64 28 / k_eq_2048 20 / k_tail 548 / n_eq_1 94 / sat_pos 393 /
sat_neg 395 / rne_tie 133 / all 8 illegal classes / all 6 reset phases /
all 3 backpressure + all 3 stall regimes.

## 6. Checker mutation evidence (then reverted; `git status rtl/` clean of edits)

1. **Scoreboard**: corrupted 1 nibble of one expected beat in the directed
   vectors → `FAIL [J1 8x8x8 op=1 acc=0 rq=0] cyc=90: beat 0` →
   `%Fatal ... TB FAIL: 1 error(s) across 48 jobs`.
2. **SVA / D-006**: scratch-copy RTL mutation `accn_cnt==m_cnt` →
   `sent_cnt==m_cnt` (done on skid-entry, pre-acceptance) → under storm
   backpressure: `%Error: apex_stream_sva.svh:169: Assertion failed ...
   ap_done_means_idle: [SVA §5] busy still high at done`.
3. **Numerics / C-2**: scratch-copy `mxe_requant.sv` mutation
   `rem > half || (rem==half && q[0])` → `rem >= half` (round-half-up, the
   run2 rounding) → the RNE tie jobs fail exactly:
   `FAIL [J6 1x1x8 ...] / [J7] / [J8] ... TB FAIL: 3 error(s)`.

## 7. Observations (not bugs; recorded for integration)

- `desc` interface has no skid (documented "permitted reg-stage" deviation
  from §5's every-boundary-skid rule); fields are registered on accept and
  desc_ready gating makes it race-free — SVA-verified.
- `busy` covers OUTPUT-pending beats only (per §5's letter). Beats a producer
  pushes into the act/wgt input skids while the DUT is not ingesting are
  retained and would be consumed by the NEXT job — SEQ must not overfeed.
- `desc.mode_os` is ignored; the opcode is authoritative (documented in
  mxe_ctrl). A SEQ that sets them inconsistently silently gets opcode
  semantics — worth an SEQ-level assert or an apex_pkg comment in v0.2.
- OS `accumulate=1` retains resident state from ANY prior job incl. WS
  (documented, verified bit-exact incl. shape-mismatched chains); chain
  hygiene is software/SEQ responsibility.

## 8. Addendum 2026-07-09 — D-004 / D-005 PARTIAL closure

Two new suites join `make -C verif/mxe all` (now: smoke + sb + struct + perf);
the whole target re-ran green from clean with the changed controller
(exit 0; sb re-run: same 836 legal jobs / 65 illegal / 35 resets, 0 errors,
all 38 required buckets incl. `rst_wload 5 HIT`).

### 8.1 D-004 — structural discriminator (`verif/mxe/struct/`)

What was missing: the 836-job suite proves C = A·B bit-exact, but a broadcast
MAC wall computes the same function — the *systolic structure* claim had no
discriminating test. The discriminating property is the documented timing law
of `mxe_array.sv` (P1..P5 in `tb_mxe_struct.sv`): (r+c)-dependent activation
arrival at each PE west input, per-hop registered psum staggering
(prefix value at T+r+c+1), column-staggered accumulator writes (T+MXE_N+c),
1-beat/cycle sustained wavefronts, and single-cycle glitch hop locality
(a deposit on PE(2,3).act_out reaches column 4+k exactly k cycles later and
is never visible at farther columns in the injection cycle).

- Real RTL: `STRUCT PASS: 20800 whitebox timing checks` (`build/run_real.log`).
- Mutation check (`build/mutation.txt`): a *realistic* broadcast-wall mutant
  (`mutant/mxe_array.sv` — broadcast + combinational column adder trees +
  same-cycle accumulator writes, functionally equivalent at the ports)
  PASSES `tb_mxe_smoke` (`MXE SMOKE: ALL TESTS PASSED` — proof the functional
  suite cannot discriminate) and is KILLED by the structural TB
  (`STRUCT FAIL: 2677/13120 checks failed`). Caveat: the mutant is the
  realistic run1/run2-style wall; an adversarial wall with per-column delay
  lines mimicking internal timing would be caught only by the P5 glitch test
  (which requires a physical act path between adjacent columns).

### 8.2 D-005 — load-under-compute (`rtl/mxe/mxe_ctrl.sv` + `verif/mxe/perf/`)

Controller change (D-006-contract-internal; desc/act/wgt/res port protocol,
beat counts/order, done/busy, legality and reset semantics all UNCHANGED):
the weight loader is decoupled from the compute FSM — chunk k+1's weights
stream into the shadow bank while chunk k computes from the live bank (and
chunk 0 loads under INGEST); the bank swap moved from "last WLOAD strobe" to
COMPUTE entry (wavefront provably clear: post-FLUSH or pre-first-compute);
S_WLOAD survives as a wait-for-shadow-bank state, entered only when the
weight stream lags (same enum encoding — sb TB phase targeting unchanged,
except phase-2 targeting now withholds the weight feed to force the wait).

Regression-first evidence: `tb_mxe_perf +require_overlap=1` against the
sequential controller FAILED with `overlap=0` on all 4 multi-chunk jobs
(kept permanently as `make run_base_reqov` against the pinned baseline copy
`perf/baseline/mxe_ctrl.sv` — `[CONFIRMED-GAP] ... PERF OVERLAP MISSING`).

Perf gate (`build/perf_gate.txt`, bit-exact results asserted in both builds,
§5/D-006 SVA pack bound):

```
job                 base cyc   new cyc   saved  speedup  overlap
K2048_M64              56901     53573    3328   1.062x     2048
K2048_M64_wgap3        60741     53573    7168   1.134x     2048
K64_M16                  645       541     104   1.192x       64
K64_M16_wgap3            765       579     186   1.321x       53
```

overlap > 0 asserted per job; every weight-load strobe of every chunk hides
under INGEST/COMPUTE/FLUSH at full feed rate (overlap 2048 = 256 chunks x 8
strobes for K=2048). The bandwidth-limited rows are the run2-pathology
regime: at 1 weight beat / 4 cycles the overlapped controller runs K=2048 at
the SAME total as full rate (53573 — load fully hidden), where the
sequential baseline pays +7168 cycles.
