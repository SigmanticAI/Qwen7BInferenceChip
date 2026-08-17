# B3 — Native W4 Weight Path — Golden-First Design Contract & Take-Over Brief

**Block:** B3 native W4 weight path · **Branch:** `comp/b3-weight-path`
(`docs/design/LEVEL_C_PARALLEL.md:21`) · **Contract frame:** `docs/OPTIMIZATION.md:40,67`
(Tier B / Top-5 rank 5) · **Numerics anchor:** `ARCHITECTURE.md` C-4 (lines 67–69),
D-021 (lines 232–260) · **apex_pkg stays FROZEN** (APEX_VERSION `0x0001_0000`,
`rtl/apex_pkg.sv:14`).

This brief is self-contained: a fresh session can build + unit-verify B3 in a
worktree without the other two lanes existing. Golden is the arbiter — never
edit golden to match RTL (`docs/design/LEVEL_C_PARALLEL.md:55`).

---

## 1. What the block does & where it sits

Today the MXE array is fed INT8 weights only, streamed from the host over the
`xw` lane8 port (`rtl/top/apex_top.sv:171–174`, "v0.1 has no on-tile weight
memory"), muxed into `mxe_top.wgt` at `rtl/top/apex_top.sv:775–778`. The weight
loader inside `mxe_ctrl` consumes those beats column-major — beat `(p,c)` lane
`r` = `B[8p+r][c]` — one INT8 weight per lane (`rtl/mxe/mxe_ctrl.sv:46–59,
177–213`; `w_ready` at `:190`, lane assembly at `:206–213`).

B3 inserts a **weight feeder** on the `xw`→MXE path that consumes **packed
4-bit** weights and emits the INT8 beats the array already understands. Per
**C-4** (`ARCHITECTURE.md:67–69`): *"the MXE array sees exactly one dtype
(INT8). W4 weights, **when adopted (v1.1)**, unpack+dequant at the weight
feeder, never inside PEs."* (The v1.1 qualifier is part of the clause — B3 IS
that adoption step; earlier drafts of this brief dropped it.) The array,
`mxe_ctrl`, `mxe_requant`, and the descriptor are untouched; the feeder is a
new module upstream of `mxe_top.wgt`.

Leverage (`docs/OPTIMIZATION.md:3,16,67`): weight bytes dominate decode traffic
at D=64 (≈16 KB/layer INT8 projections vs 4–8 KB KV). **PERF target: `xw`
beats/job halved** — one packed-W4 lane8 beat (64 b = 16 nibbles) expands to two
INT8 weight beats. Two honest qualifiers, both verified in stage 0:
- Beats/job = `ceil(K/8)·N` (`mxe_ctrl.sv:46–59`, already coded as `nw = kb*n`
  at `verif/mxe/perf/tb_mxe_perf.sv:216`). Exactly halved **only when `KB·N` is
  even**; N is legal 1..8, so odd `KB·N` is legal and gives
  `emitted = 2·consumed − 1`. It is even at the shipped D=64/N=8 config.
- The perf model books this feature at **4.125 b/w = a 48.4% cut**, not 50%
  (`perf/apex_perf_model.py:101–109`), because it counts the fp16 group scales
  in the byte total. "Halved" is the beat count; 48.4% is the byte count.
- There is **no `xw` beat counter in the repo today** (the only `xw_cnt` is an
  F2 mailbox FIFO free-space register, `apex_f2_mailbox.sv:222`). The PERF
  number must be *built*, not read.

Mode selection is **route-level / CSR, NOT a descriptor flag** — this is the
whole reason apex_pkg can stay frozen (`docs/OPTIMIZATION.md:40`). Follow the
existing route-signal pattern (`rt_wgt_src` at `rtl/top/apex_top.sv:231,776`).
For unit verification the mode is a plain module input driven by the TB; the CSR
bit is deferred to integration (see §7 — `csr_regs.sv` is B1's, do not touch it).

---

## 2. The golden reference (arbiter)

No W4 weight path exists in golden today: `LayerWeights` stores INT8 weights +
per-tensor float dequant scales (`golden/apex_golden/transformer.py:278–303`),
consumed by `gemm_i8` (`golden/apex_golden/compute.py:22–36`).

> **CORRECTION (stage 0, 2026-07-20).** The v0 text said the per-tensor `s_w`
> is "folded into the C-2 epilogue by `_proj_epilogue` via
> `attention.calib_requant`". That is **false**, and it mislocates the whole
> stage-0 decision. `calib_requant`'s only input is the integer accumulator
> amax (`attention.py:207–219`, `ratio = target/float(amax_acc)`) — it is
> scale-agnostic and never sees `s_w`. `s_w` rides a **separate float64
> factor applied after** the requant: `transformer.py:369`,
> `s_out = s_a_val * float(s_w) * float(1 << shift) / float(scale)`. There is
> no float-scale → `(rq_scale, rq_shift)` mapping anywhere in golden.
>
> Worse, the C-2 epilogue **is not running on the weight path at all**. Every
> projection job is issued as `desc_words(OP_GEMM_WS, 1, D, 8)` with `rq_en`
> defaulting to 0 (`verif/top/l3/gen_l3_vectors.py:143,629–632,648–652`); the
> single `rq_en=1` in the whole L3 script is the P·V̂ epilogue at `:418`. The
> weight scale is applied **downstream by `apex_scale_quant`**, one fp32
> composite per output element. So `mxe_requant`'s one-scale-per-job is NOT
> the binding constraint on B3; `apex_scale_quant`'s scale-grade contract is
> (see the scale-grade note below).

So B3 **adds a new golden ref**, built only from already-trusted primitives:

**New golden module `golden/apex_golden/weight_codec.py`** (own file — no edit to
`cq_codec.py`, `compute.py`, or `attention.py`), exporting:

- `compress_weights_w4(W8, group="tile", pow2_scale=False) -> WeightBlob`
  — per **group** (`"tile"` | `"col"` | int G = G K-rows × 1 column) run the C-1
  INT4 machinery already in `cq_codec.py`: `scale_from_amax`
  (`cq_codec.py:45–51`), `quant_codes(bits=4)` (`:53–57`), `pack_int4`
  (`:68–73`). This is the *same* [-8,7] / EPS=2⁻¹⁴ / RNE contract as KV keys.
  Input is the **INT8 weight matrix** as `LayerWeights` already stores it — the
  v0 signature's `W_i8_or_real` ambiguity is resolved to INT8 because the
  cq_codec primitives take fp16 *bit patterns*, and INT8 codes narrow to fp16
  **losslessly** (every integer ≤ 2048 is fp16-exact — asserted in
  `weights_to_f16`, gated in test section B). So there is **no extra
  quantization hop**, and the per-tensor `s_w` is never narrowed: it stays a
  float64 factor of the composite exactly as today.
- `wfeed_w4_to_i8(blob, realization) -> (i8_stream, scale_f16)`
  — the **exact bytes the RTL feeder must emit**: `unpack_int4`
  (`cq_codec.py:76–82`, element 2i→low nibble, 2i+1→high, two's-complement)
  producing the sign-extended INT8 operand stream, plus the single fp16 scale
  that rides downstream in place of `s_w`. Bit-exact against the existing
  `cq_codec.dequant_f32` (`:60–63`) / `cq_fp_pkg::dequant_one`
  (`rtl/kvq/cores/cq_fp_pkg.sv:368+`, wrapped by `cq_dequant_unit.sv`) — the W4
  unpack is the *same primitive family* as the KV INT4 dequant.

**The byte-level stream contract is normative in `weight_codec.py` (S1–S6),**
because nothing in the repo pinned it before and RTL, vectors and the mutation
gate must all agree: beat order chunk-major/column-minor, flat element
`e = 8·(p·N + c) + r`, packed beat *j* carrying flat elements 16j..16j+15 with
`pack_int4`'s nibble rule, `ceil(KB·N/2)` packed beats, and the odd-tail rule.

### The stage-0 realization decision — RESOLVED 2026-07-20 (measured)

The v0 framing was: "two contract-legal realizations, gated on the D-023
≤5%-of-value-scale budget; pick (B) only if (A) misses budget." Three of those
premises did not survive contact with the code and the measurement. What
follows is the resolution and the numbers that produced it. Repro:
`make -C golden weightcodec`; full table in `docs/results/b3_w4_golden/`.

**The two realizations, restated correctly.** A group scale leaves the
contraction `acc[m][c] = Σ_k A[m][k]·B[k][c]` as one downstream factor only if
it is constant over **both k and c**:
- **(A) scale-in-epilogue** — feeder output = sign-extend INT4 code (−8..7) to
  INT8; the group's fp16 scale rides downstream as a factor of the composite,
  where `s_w` rides today. **Requires `group == "tile"`.** The v0 text's
  recommended "per output column, along K" is *structurally impossible*: one
  job's weight tile is `W[:, 8j:8j+8]` — **eight** output columns
  (`gen_l3_vectors.py:151–161,629–632`), so per-column needs N=1 tiling and an
  ~8× job-count cost. Per-K-chunk grouping cannot work at all: K-chunks
  accumulate *inside* one job (`mxe_ctrl.sv:334–363`), so a scale varying along
  k never leaves the sum.
- **(B) dequant-then-requant** — feeder dequants `q4·s_g`→real (exact,
  `cq_codec.dequant_f32`) then requants the tile to one INT8 scale: the D-021
  `seam_feeder_quant` chain (`rtl/seam/seam_feeder_quant.sv:1–63,161–237`)
  reused for weights. Any group geometry is legal.

**Finding 1 — the choice is not an accuracy choice.** Measured projection error
`max|a·W_rec − a·W_real| / max|a·W_real|` (activations identical both sides),
K=64 N=8, seed 90210:

| weights | INT8 (shipped) | A tile | B col | B G=32 | B G=16 |
|---|---|---|---|---|---|
| gaussian | 0.00492 | 0.11542 | 0.09987 | 0.07758 | 0.07262 |
| uniform  | 0.00398 | 0.06541 | 0.07817 | 0.07817 | 0.06092 |
| outlier (19× col range) | 0.00833 | 0.13109 | 0.13171 | 0.12554 | 0.09259 |

(A) and (B) land within 25% of each other on every distribution — (A) is
*better* on uniform and outlier. So "pick (B) only if (A) misses budget" has no
budget gap to arbitrate, and its ordering assumption is wrong. **Decision: (A),
on engineering grounds** — a pure unpack is trivially bit-exact, needs no fp16
arithmetic in the feeder and no scale sideband, and gives the cleanest mutation
gate. (B) stays the documented fallback if a future geometry needs it.

**Finding 2 — the accuracy is intrinsic to the 4-bit code width, not to
anything the feeder chooses.** (A) tracks INT4-straight-onto-real-weights
within 20% on all three distributions, and swapping (B)'s per-tile INT8 output
scale for a per-column one (which the hardware cannot do) moves gaussian only
0.09987 → 0.09127. The per-tile INT8 funnel is *not* the bottleneck. The only
material lever is **finer K-grouping**: G=16 beats tile-wide by 37% / 7% / 29%.

**Finding 3 — the binding hardware constraint is `apex_scale_quant`, not
`mxe_requant`.** Its C2 contract requires the fp32 sideband scale to carry ≤11
significant bits (`frac[12:0] == 0`), else `scale_error` pulse+sticky
(`rtl/top/glue/apex_scale_quant.sv:29–31,227`). Today's flows satisfy that only
because `s_w` is a power of two (`gen_l3_vectors.py:589`, `S_W = 2**-6`). An
fp16 W4 group scale makes the composite ~22 significant bits and trips it.
**Mitigation: round the COMPOSITE to fp16 grade** (`weight_codec.f16_grade`) —
measured cost ≤ 5e-5 absolute, i.e. free. The obvious-looking alternative,
snapping the group scale up to a power of two, costs **1.7–2.1×** and is
rejected on that number.

**Finding 4 — the headline B3 must be judged on: native W4 costs 11–24× the
INT8 baseline's projection error** (0.061–0.132 vs 0.004–0.008), on every
distribution and every realization. Whether that is acceptable is a *product*
decision about end-to-end token quality, not a feeder decision, and it is
**not** settled by this stage. See §5 stage 6.

**On the D-023 gate.** Gating "exactly like D-021/D-023" as v0 specified is
**vacuous for W4**: every existing float64 reference is built from the same
dequantized weights (`transformer.py:519–521`, `attention.py:538–541`), so a W4
error measured against them cancels identically and the gate reads the INT8
plumbing baseline. D-023's ≤5%-of-*value*-scale is also an attention/V-path
metric, not a weight metric. Stage 0 therefore measures against the **original
real weights** and pins the result as a regression register (the
`test_effective_bits.py` discipline), rather than asserting a threshold it
cannot justify.

---

## 3. Interface contract

**apex_pkg (`rtl/apex_pkg.sv`) stays byte-identical.** No new descriptor field
(D-021 seam blocks added none either); no APEX_VERSION bump. Reuse frozen types:
`lane8_beat_t` (`:57–60`) for both the packed-W4 input and the INT8 output.

New module (own file, add-only), suggested `rtl/mxe/mxe_wfeed_w4.sv` (or
`rtl/seam/seam_weight_w4.sv`):
```
input  logic        clk, rst_n;
input  logic        w4_en;              // route/CSR select (TB-driven at unit level)
input  logic        job_valid; output logic job_ready;   // §5/D-006 job frame
input  logic [DIM_W-1:0] job_beats;     // packed-W4 beats this job (legality-checked)
input  logic        pw_valid; output logic pw_ready; input lane8_beat_t pw_beat;  // packed W4 in
output logic        w8_valid; input  logic w8_ready; output lane8_beat_t w8_beat; // INT8 out (→ mxe.wgt)
// realization (B) only: fp16 scale sideband, mirror seam_feeder_quant scl_* (:92–96)
output logic        busy, done, job_error, job_error_sticky;
```
Stream/job discipline is the frozen one: `valid`/`ready`, 2-deep `stream_skid`
on every boundary, `done` ⇒ post-skid acceptance, `desc_ready` for N+1 gated on
N's `done` (`ARCHITECTURE.md` §5 / D-006; `apex_pkg.sv:67–72`; template
`seam_feeder_quant.sv:57–62,111–158`). Legality reject = pulse + sticky, zero
state change (`§3`; `mxe_ctrl.sv:299–303`, `seam_feeder_quant.sv:264,318–321`).

**Route/CSR selection:** a new route bit (e.g. `rt_wgt_w4`) selects the feeder
vs INT8 passthrough at `apex_top.sv:775–778`. That top wiring + the CSR bit are
**integration-time** work — B3 unit-verifies with `w4_en` as a module input and
does **not** edit `rtl/csr/csr_regs.sv` (B1's file, §7).

---

## 4. Verification target

**Acceptance = bit-exact vs the new golden ref, and the golden ref bit-exact vs
existing trusted golden.** Two layers:

1. **Feeder equivalence (the core gate):** drive packed-W4 + `job_beats`; the
   emitted INT8 stream must equal `weight_codec.wfeed_w4_to_i8(...)` byte-for-byte,
   under backpressure/stall/mid-op-reset regimes. TB pattern is **copied from
   `verif/seam/`**: `tb_seam_feeder_sb.sv` (vector-driven scoreboard, TB only
   drives/collects/compares; header `:1–24`), `gen_seam_vectors.py` (stimulus +
   expected from the golden arbiter), `seam_job_sva.svh` + `apex_stream1_sva`
   bound every cycle, `+bp_mode/+stall_mode/+seed` adversaries, `$fatal`
   watchdog, `coverage_report.py` gate, `mutation_check.py` gate.

2. **Unpack/dequant exhaustive sweep (the C-2 equivalence pattern):**
   `ARCHITECTURE.md:56–58` requires requant equivalence "by exhaustive sweep over
   the operand domain." The W4 primitive's operand domain is **tiny and fully
   enumerable**: all 16 nibble codes × all finite fp16 group scales. Mirror
   `verif/kvq/fparith/prove.py` (`dequant_one` over codes × every distinct scale,
   `:87–100`; `scale_from_amax` exhaustive, `:63–73`) and its checksum
   vector-gen `gen_fparith_vectors.py`. This *proves* the feeder's unpack+dequant
   equals the KV-family primitive with zero gaps.

3. **Optional end-to-end sanity:** feeder→`mxe_top`→`mxe_requant` on a real
   projection tile equals `gemm_i8` + `requant_i32_to_i8`
   (`compute.py:22–36,76–96`) with the W4-composite epilogue — reuse the
   `verif/mxe/sb/` GEMM+requant scoreboard (`gen_mxe_vectors.py:40,131`).

**Mutation-gate expectations** (`verif/seam/mutation_check.py` style) —
**MEASURED 2026-07-20, 5/5 killed**, `verif/mxe/w4/mutation_check.py`:

| # | mutant | verdict |
|---|---|---|
| M1 | nibble-order swap (low↔high) | KILLED |
| M2 | dropped sign-extension (`{{4{nib[3]}},nib}` → `{4'b0,nib}`) | KILLED |
| M3 | stuck `w4_en` / bypass-of-mode | KILLED (only by the passthrough jobs) |
| M4 | off-by-one lane doubling | KILLED |
| M5 | odd-tail: `need_new`'s ceil → floor | KILLED (via the TB's JOB_TIMEOUT hang path) |

**M5 was RELOCATED against measurement.** The v0 text named "hold_retire drops
`|| last_beat`" as the odd-tail mutant. That mutant was built and run, and it
**survives every regime** (300 legal jobs, 61 odd-tail). It is an *equivalent
mutant*, not a checker hole: on the cycle the odd tail's final beat is
emitted, `ST_RUN` takes `if (last_beat) state <= ST_WAIT`, and `o_valid` is
gated on `ST_RUN`, so the stale padding half cannot be emitted with or without
the term; the next job accept re-clears the flops anyway. No port-level TB can
kill it and none should be expected to. The term is **kept** in the RTL as
documented-defensive (the `seam_feeder_quant` P5 precedent — never weaken a
check, document the reachability), and the gate now targets `need_new`'s ceil,
which IS load-bearing for odd tails. The mutant list was corrected, not waived.

Mid-operation reset test is REQUIRED per block (`ARCHITECTURE.md:189–190`;
`+vectors=*_reset_*`): **15 mid-op resets, 0 errors**, hitting `ST_RUN` (13)
and `ST_WAIT` (2). `ST_IDLE` is a documented reachability-noted bucket — an
abort that lands there means the job had already completed, so it is not a
mid-op case and is not required.

**PERF check:** count packed-W4 beats vs emitted INT8 beats in the TB and assert
`emitted == 2 × consumed − (KB·N & 1)` — the corrected form (`weight_codec` S6;
the unqualified `2 × consumed` is false for legal odd `KB·N`, and the TB must
cover an odd config so the bug cannot hide). Reported from parsed TB output,
never hand-written (`ARCHITECTURE.md:203–205`). Note that "xw beats/job halved"
(`docs/OPTIMIZATION.md:67`) and the perf model's 4.125 b/w = 48.4% byte cut
(`perf/apex_perf_model.py:101–109`) are two different quantities; quote both or
neither. No `xw` beat counter exists yet — building it is part of this lane.

---

## 5. Staged landing plan (golden-first, machine-aware)

Serialize only the Verilator stages on the one box; run each on a cheap AWS
`c6a.xlarge` if the queue is hot (`docs/design/LEVEL_C_PARALLEL.md:30–35`).

| stage | what | machine need |
|---|---|---|
| 0 | ✅ **DONE 2026-07-20.** `golden/apex_golden/weight_codec.py` + `golden/tests/test_weight_codec.py`: W4 pack/unpack/dequant on `cq_codec` primitives, bit-exact vs `cq_codec.dequant_f32`, S1–S6 stream contract pinned, realization resolved to **(A) tile** (see §2), wired into `make -C golden test` (banner byte-identical for `gen_status.py:273`) | edit only (Python) |
| 1 | new feeder RTL (`mxe_wfeed_w4.sv`): unpack + sign-extend + §5/D-006 frame + legality faults; lint clean `-Wall` (waivers scoped to frozen `apex_pkg.sv` only, `verif/seam/Makefile` rule). Realization (A) ⇒ **no fp16 arithmetic and no scale sideband in the feeder** | edit only |
| 2 | ✅ **DONE 2026-07-20.** `verif/mxe/w4/` (own TB dir/vectors/SVA): `gen_w4_vectors.py`, `tb_wfeed_w4_sb.sv`, bound `w4_job_sva` + `apex_stream1_sva`, adversarial + reset regimes, coverage gate. 6 runs, 334 legal + 4 illegal jobs + 15 mid-op resets, **0 errors, all bit-exact vs golden** | Verilator (serialize) |
| 3 | ✅ **DONE 2026-07-20.** Mutation gate **5/5 killed** (M5 relocated, see §4) + PERF beats assertion in-TB on every job: **255 W4 jobs, S6 identity `emitted == 2*consumed - (beats&1)` holds 255/255**, aggregate 1.9960×. Coverage: all 26 required buckets closed. NOTE: the operand-domain sweep is *structurally unnecessary for realization (A)* — the feeder is a pure unpack, so its operand domain is the 16 nibble codes, exhausted by the vectors; there is no fp16 scale arithmetic in the datapath to sweep. The `verif/kvq/fparith` sweep pattern applies only if (B) is ever adopted | Verilator (serialize) |
| 4 | optional feeder→`mxe_top` e2e tile vs `gemm_i8`+`requant_i32_to_i8` | Verilator (serialize) |
| 5 | integration handoff notes: route bit + CSR wiring + `apex_top` mux — LEFT for the combine step (coordinate `csr_regs.sv` with B1). Also: the host must narrow the composite to fp16 grade (§2 finding 3), and the feeder's placement relative to the `rt_wgt_src` mux at `apex_top.sv:775–779` must be pinned — v0 asserted both "on the `xw` path" and "at the mux", which are different designs | none |
| 6 | **ADOPTION GATE — ✅ CLOSED 2026-07-22 (all full sets landed). FINAL RECIPE: W4 G=32/(B) with DIRECT-from-source host prep — −1.0 pt vs shipped weights (z=−4.73), BETTER than the current per-tensor INT8 feed (+0.0077, z=+2.75); the INT8-hop prep is DEPRECATED for product use (costs −1.7 pts, z=+5.85). Group size MEASURED on the adopted chain: direct@G16 = −0.63 pt vs shipped (z=−3.46), +0.41 over direct@G32 (z=+2.05, borderline) at +11% traffic ⇒ FREEZE AT G=32; G=16 = validated fallback. Matrix complete, every cell at n=10,042. 07-21 definitive decomposition: total cost of the adopted G=32/(B) config = −2.7 pts vs the shipped mlx-4bit weights (paired z=−10.2) = −1.8 per-tensor-INT8 hop (z=−7.1) + −0.9 W4-G32 increment (z=−4.0); every component definitively real; the n=1000 'W4 increment indistinguishable' reading is SUPERSEDED (under-powered). The INT8 hop, not 4-bit width, dominates → open follow-ups in RESULTS.md §rec 5: W4-direct-from-source host prep (skips the hop, feeder contract unchanged) + optional G=16 full set before the group size freezes at integration. Initial verdict (stands): realization (A) tile-scale is REJECTED as the product weight config — it collapses Qwen2.5-7B (HellaSwag acc_norm 0.351 vs 0.673 base, paired z=−17.2, definitively real; greedy tokens garbage from step 0). Finer K-grouping rescues completely and saturates at G=32: W4 G=32 via a (B)-style chain costs −0.003±0.008 (z=−0.38) vs the INT8 golden feed — indistinguishable at n=1000 — and its token stream stays coherent. Adoption path = G=32 (B)-style feeder (or host pre-pass) with a host-computed stripe-global sideband scale (design delta from D-021 at K>2048, disclosed; per-job scales would be same-or-better). Perf flag: G=32 stores 4.5 b/w vs the modeled 4.125 — refresh the S7 model before quoting W4-dependent numbers. Full evidence + gates (54M golden stripe checks/0 fails per tier, S5-harness preflight-licensed pairing): `docs/results/b3_w4_adoption/RESULTS.md`. REMAINING: n=10,042 for w-int8 + w4-g32-b before any public wording; the stage-1–3 (A) feeder RTL stays verified/inert (unpack datapath reusable for (B)).** Original spec: §2 finding 4 measures W4 at 11–24× the INT8 projection error; no existing gate can say whether that is acceptable — e2e token-quality on the real model vs the *original* mlx-4bit weights (the honest baseline), finer K-grouping (G=16/32) as the lever. (Note: the run_tinynpu.py golden-pipeline form is stage-5 work — `_proj_epilogue` carries one scalar s_w per tensor (frozen C-2), so per-stripe scales cannot thread through without editing golden; the gate ran on the MLX path with the golden-chain-certified installer, disclosed in RESULTS.md.) | edit only (Python) |

---

## 6. Files to read / touch & repro

**Read (do not modify):** `ARCHITECTURE.md` (C-4 67–69, D-021 232–260, §2/§3/§5),
`docs/OPTIMIZATION.md:40,67`, `rtl/mxe/mxe_ctrl.sv` (weight loader 46–59,177–213),
`rtl/mxe/mxe_buf.sv`, `rtl/mxe/mxe_top.sv` (wgt path 79–89, requant 184–198),
`rtl/mxe/mxe_requant.sv` (C-2), `rtl/seam/seam_feeder_quant.sv` (D-021 template),
`rtl/kvq/cores/cq_dequant_unit.sv` + `cq_fp_pkg.sv:368+`,
`golden/apex_golden/cq_codec.py` (primitives), `golden/apex_golden/compute.py`,
`golden/apex_golden/transformer.py:278–303`, `verif/kvq/fparith/prove.py`,
`verif/seam/*` (TB skeleton), `rtl/top/apex_top.sv:171–174,775–778`.

**Create (add-only, this lane's own files):** `golden/apex_golden/weight_codec.py`,
`golden/tests/test_weight_codec.py`, `rtl/mxe/mxe_wfeed_w4.sv`, `verif/mxe/w4/`
(Makefile, `gen_w4_vectors.py`, `tb_wfeed_w4_sb.sv`, SVA include, coverage +
mutation scripts).

**Repro commands** (in the worktree):
```
git worktree add ../apex-weightpath comp/b3-weight-path   # LEVEL_C_PARALLEL.md:41
make -C golden test                    # golden gate incl. new test_weight_codec
make -C verif/mxe/w4 all               # lint, vectors, build, run, coverage, mutate
make -C verif/kvq/fparith all          # confirm the reused dequant primitive still green
```

---

## 7. Independence notes (collision avoidance)

Per `docs/design/LEVEL_C_PARALLEL.md:18–24`, the three lanes touch **disjoint**
RTL. B3's declared surface is `rtl/mxe/*` and `rtl/seam/seam_feeder_quant.sv`.

- **`rtl/apex_pkg.sv` — FROZEN, shared by all.** B3 must not touch it (route/CSR
  select, not a descriptor field — that is the entire contract discipline,
  `docs/OPTIMIZATION.md:40,87`). If you ever feel you need a new descriptor field,
  stop: the design is wrong.
- **`rtl/csr/csr_regs.sv` and `rtl/seq/seq_walker.sv` are B1-walker's**
  (`LEVEL_C_PARALLEL.md:20`). Do **not** add the W4 CSR bit here — expose `w4_en`
  as a module input, verify standalone, and defer CSR/route wiring to integration.
- **`rtl/asu/asu_rmsnorm.sv` is the wide-RMSNorm lane's** (`:22`). No overlap.
- **Prefer a NEW module over editing `mxe_top.sv`/`mxe_ctrl.sv`.** The feeder as a
  stream pre-processor on `xw` keeps `mxe_top` byte-identical; if a `wgt` mux edit
  is unavoidable, keep it a single add-only guarded assignment and note it for the
  combine step. Do **not** edit `golden/apex_golden/cq_codec.py` — import its
  primitives from the new `weight_codec.py`.
- **Own TB dir** (`verif/mxe/w4/`), own vectors, own SVA include — no shared verif
  state with B1/wide-RMSNorm. Commit atomically with explicit paths on
  `comp/b3-weight-path` (`LEVEL_C_PARALLEL.md:59`).

---

## 8. Stage-6 fallout — integration design notes for the adopted G=32/(B) path

*(Added 2026-07-21 after the adoption gate,
`docs/results/b3_w4_adoption/RESULTS.md`. Doc only — no RTL in this lane;
this section is the §5-row-5 handoff material for the combine step.)*

The gate rejected realization (A) tile-scale (quality-fatal on the real 7B)
and measured **G=32 K-grouping through a (B)-style chain** as the adoption
config. What that means for integration:

1. **The feeder becomes (B)-shaped**: packed-W4 in → unpack/sign-extend
   (the landed `mxe_wfeed_w4.sv` stages are reusable as the front half) →
   dequant `code4 × s_g` (G=32 fp16 group scales, `cq_dequant_unit`
   family) → **one per-job INT8 requant against a scale supplied by the
   host as a sideband INPUT** — the `seam_feeder_quant` chain reused for
   weights, with one key delta from D-021: the requant scale is
   **host-computed and consumed**, not feeder-derived. Reason: at 7B
   geometry every tensor is a multi-job K-split chain (K=3584/18944 >
   K_MAX 2048) whose raw-INT32 partials share ONE epilogue factor, so all
   segments of a stripe must requant against the same scale — only the
   host sees the whole stripe. This is exactly the form the gate measured
   (disclosed in RESULTS.md); a per-job-derived-scale variant would need
   host dequant-per-segment composition instead (finer scales,
   same-or-better accuracy, different host contract) — decide at
   integration with the walker's descriptor scheme in view.
2. **Group-scale storage/traffic**: fp16 scale per 32 weights ⇒ 4.5 b/w
   (PERF_MODEL.md §3b½ prices it: 2k decode 13.0→11.9 tok/s, 32k floor
   clears at 10.6, ≥10 tok/s flat region ~48k ctx). The scale stream is a
   new sideband lane on the weight path — sizing/framing TBD at
   integration (candidates: interleaved with packed beats vs a separate
   `scl_*`-style channel per `seam_feeder_quant.sv:92–96`).
3. **Scale-grade rule carries over** (§2 finding 3): the composite the
   epilogue receives must stay fp16-GRADE for `apex_scale_quant`
   (`f16_grade` on the composite — landed in `weight_codec.py`; measured
   free). The host computes `s8_stripe × s_w` and grades it.
4. **Verification is already golden-anchored**: `wfeed_w4_to_i8(...,
   realization="B")` in the landed `weight_codec.py` is the arbiter for
   the full (B) chain (the stage-6 eval ran 54M golden stripe checks
   against it on real model bytes). The §4 operand-domain exhaustive sweep
   (fparith pattern) becomes REQUIRED for (B) — the feeder now contains
   fp16 dequant arithmetic (16 codes × all finite fp16 scales, fully
   enumerable). TB extends `verif/mxe/w4/` with scale-sideband vectors +
   the same adversarial/reset/mutation regimes.
5. **Host prep is part of the adopted contract (07-22 finals)**: W4
   blobs MUST be quantized DIRECT from the source weights — never via the
   per-tensor INT8 intermediate (deprecated: −1.7 pts, z=+5.85; the
   direct recipe beats even the current INT8 feed, z=+2.75). The feeder
   is agnostic (same codes+scales format); this binds the host tooling
   (`run_tinynpu.py prepare` grows a W4-direct mode at integration).
   Group size FROZEN AT G=32 per the completed 07-22 matrix (direct@G16 buys +0.41 pt, z=+2.05, for +11% traffic — validated fallback only).
6. **The (A) config stays in the tree but is quality-quarantined**: do not
   route product jobs through tile-scale mode; keep the landed stages 1–3
   as verified IP (pure unpack) and the TB green. Any future use of (A)
   requires its own quality gate.
