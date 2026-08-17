# ASU independent verification — RESULT

**Date:** 2026-07-07 · **DUT:** `rtl/asu/` @ working tree (`asu_exp_lut` +
generated tables, `asu_softmax`, `asu_rmsnorm` + vendored `rsqrt_unit`,
`xbr/stream_skid`)
**Verifier:** independent pass (does NOT trust the implementer's smoke claims)
**Tools:** Verilator 5.044 `--binary --timing --assert`, `-Wall` · Python
3.11.5 + NumPy
**Golden arbiter:** `golden/apex_golden/compute.py` (`exp_fx`,
`online_softmax_fx`, `softmax_ref`) per §6/C-2/C-5/D-014. RMSNorm has NO
fixed-point twin in the arbiter yet ("fixed-point twin lands with the RTL"),
so its oracle is the verifier's OWN transcription of the implementer's
documented pseudocode (`rtl/asu/asu_rmsnorm.sv` header), cross-checked at
generation time against (a) the implementer's mirror — 200 random rows
identical — and (b) the float64 arbiter `rmsnorm_ref` (quality, reported §7).

## Verdict: PASS — 0 DUT bugs found (2 architecture-level findings, §7)

2,369 legal rows (directed + 3×520 seeded random sets) across two blocks and
13 protocol configurations, all bit-exact against the golden oracles; 87
malformed-framing rejects; 28 mid-operation resets covering EVERY reachable
FSM phase of both blocks; §5 SVA bound and firing on every cycle. Checker
credibility proven by 4/4 detected mutations (§6). The implementer's smoke
claims fully reproduced from a clean tree (§2).

## 1. Deliverables (all under `verif/asu/`, beside the implementer's `smoke/`)

| File | What |
|---|---|
| `Makefile` | `make -C verif/asu all` = implementer smoke + full independent suite |
| `sb/gen_asu_sb_vectors.py` | Independent vector generator + numeric audits (D-014 re-measurement, ARCH §6 ULP acceptance audit, rmsnorm saturation-reachability proof, oracle cross-check) |
| `sb/tb_asu_softmax_sb.sv` | Vector-driven scoreboard TB (house pattern, `verif/mxe/sb` lineage): R/O/X records, backpressure + feed-stall adversaries, hierarchical phase-targeted mid-op reset (`dut.st`), per-cycle FAIL traces, $fatal watchdog; builds in 2 configs (SCORE_FRAC=0 product, SCORE_FRAC=10 interpolation-rich) |
| `sb/tb_asu_rmsnorm_sb.sv` | Same pattern for RMSNorm: N/B/O/Z records, 3-stream SVA, gamma-delivery adversaries (lockstep / late / eager), dbg_norm check |
| `sb/coverage_report.py` | Manual-bucket aggregation with reachability notes; **gates `make all`** (required holes fail the build) |
| `sb/mutation_check.py` | Repeatable mutation gate (scratch copies under `build/`, `rtl/` never modified); **gates `make all`** |
| `sb/lint_waivers.vlt` | Waivers scoped to frozen `apex_pkg.sv` + vendored `rsqrt_unit.sv` only; zero waivers on new RTL/SVA/TBs |

SVA: the MXE pack `verif/common/apex_stream_sva.svh` is `mxe_desc_t`-typed and
cannot bind to descriptor-less ASU; the factored single-stream pack
`verif/common/apex_stream1_sva.svh` (same §5 property) is bound on **all five
stream boundaries** (softmax s/m, rmsnorm x/g/y) in every build, plus
job-level D-006/§3/D-019 concurrent assertions in each TB
(`done` 1-cycle pulse, `done ⇒ !busy`, output-only-while-busy [stale-beat],
`row/len_error ⇒ !done`, sticky sets & holds, `m_prob ≤ 0x8000` range).

## 2. Reproduction of the implementer's claims (verified, not trusted)

`make -C verif/asu/smoke clean && make smoke`: **exit 0**, verbatim key lines:

```
TABLES CHECK: rtl/asu/asu_exp_lut_tables.svh matches the golden arbiter
LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg.sv + vendored rsqrt_unit.sv only)
ASU EXP LUT SMOKE: ALL TESTS PASSED (t=65538000)
  65536/65536 input patterns bit-exact vs apex_golden exp_fx
  (full Q6.10 domain + both clamp directions)
ASU SOFTMAX SMOKE: ALL TESTS PASSED
ASU RMSNORM SMOKE: ALL TESTS PASSED
ASU SMOKE: all three TBs passed (see logs above)
```

All three headline claims (tables provenance, lint-clean scope, exhaustive
exp domain) reproduce. Waiver-scope claim verified by reading
`smoke/lint_waivers.vlt`: only `apex_pkg.sv` (UNUSEDPARAM) and
`vendor/rsqrt_unit.sv` (4 per-line width rules + SYNCASYNCNET, which is
file-scoped rather than per-line — cosmetic deviation from the "per-line"
claim, same style as verif/mxe).

## 3. Independent suite — verbatim results (`make -C verif/asu all`, exit 0)

```
AUDIT exp LUT: max |exp_fx - exp| = 4.628203e-04 (budget 2^-10 = 9.765625e-04) -> OK
AUDIT rmsnorm oracle: 200 random rows identical to the implementer's documented mirror
TB PASS: 16 rows, 2 rejects, 0 mid-op resets, 0 errors (frac=0 bp=1 stall=1 seed=11)
TB PASS: 16 rows, 2 rejects, 0 mid-op resets, 0 errors (frac=0 bp=0 stall=0 seed=12)
TB PASS: 520 rows, 20 rejects, 0 mid-op resets, 0 errors (frac=0 bp=1 stall=1 seed=13)
TB PASS: 16 rows, 2 rejects, 0 mid-op resets, 0 errors (frac=0 bp=2 stall=2 seed=14)
TB PASS: 520 rows, 20 rejects, 0 mid-op resets, 0 errors (frac=0 bp=0 stall=2 seed=15)
TB PASS: 153 rows, 0 rejects, 0 mid-op resets, 0 errors (frac=10 bp=1 stall=1 seed=17)
TB PASS: 14 rows, 0 rejects, 14 mid-op resets, 0 errors (frac=0 bp=1 stall=0 seed=16)
TB PASS: 20 rows, 5 rejects, 0 mid-op resets, 0 errors (bp=1 stall=1 g=0 seed=21)
TB PASS: 20 rows, 5 rejects, 0 mid-op resets, 0 errors (bp=0 stall=0 g=0 seed=22)
TB PASS: 520 rows, 13 rejects, 0 mid-op resets, 0 errors (bp=1 stall=1 g=1 seed=23)
TB PASS: 20 rows, 5 rejects, 0 mid-op resets, 0 errors (bp=2 stall=2 g=2 seed=24)
TB PASS: 520 rows, 13 rejects, 0 mid-op resets, 0 errors (bp=1 stall=0 g=2 seed=25)
TB PASS: 14 rows, 0 rejects, 14 mid-op resets, 0 errors (bp=1 stall=0 g=0 seed=26)
COVERAGE: all required buckets hit (60 required, 64 total)
MUTATION GATE: 4/4 mutants detected — checkers proven live
ASU SB: full independent suite passed (scoreboards, SVA, resets, coverage gate, mutation gate)
```

Run regimes: `bp` = output backpressure none / ~75% / storm ~6% duty;
`stall` = input feed gaps none / short / bursty ≤32 cyc; `g` (rmsnorm) =
gamma lockstep / late 0..20 cyc/beat / eager (parked in the g-skid through
COLLECT→ISSUE→WAIT).

**Softmax directed content:** n=1 (p=0x8000), n=2 equal (0x4000 each), INT32
extremes in both orders incl. {−2³¹,−2³¹,2³¹−1,2³¹−1} (33-bit diff +
rescale-through-clamp), all-equal ×100, strictly ascending ×64 (63 l-rescales),
strictly descending, exact −16.0 clamp boundary (z=−16384 floor entry) next to
−17 clamped, triplicate max, deep-tail p=0 rows, full n=1024 capacity row;
oversize rejects both arms (1025 = last-on-overflow-beat, 1064 = FLUSH drain)
with clean rows replayed after. Random: 520 rows (row lengths 1..1024,
clustered/spread/wild-INT32 mixtures) + 20 interleaved oversize rejects.
**SCORE_FRAC=10 config** (153 rows): with the product's SCORE_FRAC=0, every
exp argument is a multiple of 2¹⁰ so the LUT interpolator is frac=0 in-datapath
— the f10 build drives nonzero interpolation fractions through the actual
softmax pipeline (the implementer's smoke never does).

**RMSNorm directed content:** every legal D ∈ {1,2,4,8,16,32,64,128}, all-zero
rows (n2=1, r=8192), x=±128/127 with g=±32768/32767 (|p|=2²² corner), zero-gamma
lanes, max-ratio single-hot rows (saturation probe, §7.3), **engineered RNE
ties of both parities** (independent search; the even tie is the discriminator
against round-half-up), dbg_norm (floor-sqrt) checked on every row; rejects:
non-pow-2 D ∈ {3,12,127}, oversize {129 direct, 160 FLUSH}. Random: 520 rows
(x full-range / tiny-norm / extreme-heavy mixtures, g full int16) + 13
interleaved rejects. `r` (inv_norm) is not a DUT output; it is verified
indirectly (every nonzero y lane is a bit-exact function of r) plus dbg_norm
directly.

**Mid-op reset (the run2 hole):** 22 phase-TARGETED resets — softmax
COLLECT/FLUSH/LOAD/DIV/PUSH/DRAIN ×2 shapes, rmsnorm
COLLECT/FLUSH/ISSUE/WAIT/EMIT/DRAIN ×2 shapes (TB observes hierarchical
`dut.st` at negedge, catches even the 1-cycle ISSUE/LOAD states) — plus 6
random-cycle aborts. After every reset: busy=0, sticky cleared, zero stray
beats/dones over a 10-cycle window, and the immediately following row
bit-exact. Log evidence:

```
RESET [X11 n=1084 phase=2 abort=0] cyc=5789: mid-op reset in FSM phase 2
RESET [Z6 d=64 phase=4 abort=0] cyc=759: mid-op reset in FSM phase 4   (rsqrt mid-flight)
TB PASS: 14 rows, 0 rejects, 14 mid-op resets, 0 errors (bp=1 stall=0 seed=16)
```

## 4. Coverage (aggregated over all 13 runs — `sb/build/coverage.txt`)

All 60 required buckets HIT. The 4 non-required holes are documented:
`sm_rst_idle` / `rms_rst_idle` (reset-from-idle is not a mid-op case) and
`rms_y_sat_pos` / `rms_y_sat_neg` — **structurally unreachable**: at D ≤ 128
the pre-saturation |t| is bounded by ≈ g·8192·√D / 2¹⁸ ≤ ~11.9k < 32767
(measured max 11904 by the generator's max-ratio probes), so `y_sat`'s clamp
is defensive dead logic. Highlights: n=1024 capacity ×3, INT32-extreme rows
×491, clamp boundary ×295, both reject arms of both blocks, all 6+6 reset
phases, all 3 bp × 3 stall × 3 gamma regimes, LUT-frac≠0 ×147 (f10 build).

## 5. SVA (bound, always compiled)

`apex_stream1_sva` (§5 no-retraction + data-stability) on all 5 boundaries ×
every run; 7 job-level concurrent assertions per TB (D-006 pulse/idle,
D-019 stale-beat, §3 error/sticky, §6 range). The D-006
"done ⇒ all beats accepted post-skid" is additionally checked by a
transaction-count monitor at every done pulse.

## 6. Checker mutation evidence (`make mutants`, a repeatable gate; scratch copies only — `rtl/` untouched)

```
MUTANT M1_sm_data_corrupt: DETECTED (rc=1) — %Error ... ap_prob_range: [SVA §6] probability above UQ1.15 1.0 (32784)
MUTANT M2_lut_entry_off_by_one: DETECTED (rc=1) — FAIL [R8 n=64] cyc=6394: p[53] got 0x0001 exp 0x0000
MUTANT M3_rms_rne_to_half_up: DETECTED (rc=1) — FAIL [N18 d=64] cyc=4258: y[9] got 0x0323 exp 0x0322
MUTANT M4_sm_done_pre_drain: DETECTED (rc=1) — %Error ... ap_out_only_in_job: [SVA D-019] output beat with no job in flight (stale beat)
MUTATION GATE: 4/4 mutants detected — checkers proven live
```

1. **Data corruption** (softmax output bit-4 XOR) → SVA range + scoreboard.
2. **ASU-specific contract: LUT ENTRY OFF-BY-ONE** (`idx+1`) → scoreboard,
   even at SCORE_FRAC=0.
3. **C-2 RNE → round-half-up** (rmsnorm) → caught exactly at lane 9, the
   engineered EVEN-parity tie — proving the tie vectors discriminate.
4. **D-006 done-before-drain** (softmax skips ST_DRAIN wait) → bound SVA
   under backpressure storm.

## 7. Findings & design notes (challenges; none is an RTL-vs-oracle bug)

1. **ARCH §6 softmax ULP acceptance is violated at SCORE_FRAC=10 (golden-model
   level, not RTL).** §6 acceptance: "softmax output vs float64 NumPy within
   ≤ 2 ULP of Q1.15 per element". Independently measured across all generated
   rows: max 2 ULP at the shipping SCORE_FRAC=0 (PASSES, at the budget edge),
   but **up to 6 ULP** on SCORE_FRAC=10 rows (e.g. generator rows 3/11/15 of
   `vectors_sm_f10.txt`, ULP 4–6). The RTL is bit-exact to
   `online_softmax_fx`, so this is a property of the arbiter's fixed-point
   scheme (LUT truncation compounding through the l-rescale + floor division),
   chargeable to the golden model / §6 budget, not to this RTL. If MXE↔ASU
   integration ever feeds fractional scores (SCORE_FRAC > 0), the §6
   acceptance needs re-derivation or the budget relaxed. Evidence:
   `sb/build/gen.log` "AUDIT softmax quality".
2. **RMSNorm epsilon diverges from the arbiter's float reference.**
   ARCH §6 / `compute.py:rmsnorm_ref` use ε = 2⁻¹⁴ in *value* units; the RTL
   contract pins EPS_INT = 1.0 in raw-integer x² units — about 2¹⁴× larger
   relative to the same domain. Deliberate (guarantees norm2 ≥ 1 so the
   divider never saturates) and documented in the module header, but it is a
   *numerics contract decision* living only in a module header. Measured
   effect: ≤ ~3% relative deviation vs float on full-scale rows, up to 87% on
   tiny-norm rows (|x| ≤ 4, where integer isqrt granularity dominates
   anyway). Should be ratified as a D-number in ARCHITECTURE.md before KVQ/MXE
   integration relies on it. (apex_pkg is frozen; no change needed there —
   the constant lives in the RMSNorm module.)
3. **`y_sat` clamp is unreachable** at D ≤ 128 / Q2.13 gamma / UQ2.13 r
   (bound §4). Harmless-defensive, but the module header's "saturated to
   [-32768, 32767]" implies a reachable behavior that cannot be exercised —
   worth a comment so nobody chases the sat path in coverage.
4. **Softmax buffers the whole row (two-pass), diverging from §6's FA-3
   "applied on the fly / no materialized score tensor" narrative.** The golden
   `online_softmax_fx` does the same, and its docstring even says the second
   pass belongs in the P·V̂ consumer. Consequences: 4 KB row buffer
   (SM_ROW_MAX×32b), row length capped at SM_ROW_MAX=1024 < K_MAX=2048 —
   a decode step with more than 1024 keys per tile row cannot pass through
   this ASU as-parameterized — and emission throughput of ~33 cycles/element
   (bit-serial restoring divider), which will dominate the MXE↔ASU pipe.
   All documented in the header (LATENCY note), none contract-violating, but
   the SM_ROW_MAX-vs-K_MAX mismatch needs an explicit integration decision.
5. **Gamma stream has no framing — misalignment leaks across rows.** Gamma
   for a REJECTED row (or excess beats) parks in the g-skid and WILL be
   consumed by the next legal row's emission, silently corrupting it. The
   documented contract ("exactly D beats per row, consumed during emission")
   makes alignment the environment's job — same class as MXE's "SEQ must not
   overfeed" note. Verified safe under lockstep/late/eager delivery for legal
   rows; the reject-leak hazard is by-construction and needs a SEQ/XBR rule
   at integration.
6. Minor: `smoke/lint_waivers.vlt`'s SYNCASYNCNET waiver is file-scoped, not
   per-line (claim said per-line); `compute.py:_lut_tables` docstring still
   says "64 (base, slope) pairs" (it's 256, comment rot); `exp_fx`'s
   `frac_rem` comment says "= 8" (it's 6). None affect behavior.

## 8. What was NOT covered (honest scope)

- `asu_softmax` SM_ROW_MAX values other than 1024 and SCORE_FRAC other than
  {0, 10} (elaboration-legality `$error` generates were not exercised).
- rsqrt_unit internals beyond its wrapper contract (V0.4 owns that; dbg_norm
  + bit-exact y re-verify it end-to-end here, incl. reset mid-WAIT).
- Mid-op reset while `rst_n` glitches shorter than 3 cycles; multi-clock or
  async-reset scenarios (tile is single-clock sync per §5).
- Icarus cross-simulation (Verilator-only, per D-012 primary).
