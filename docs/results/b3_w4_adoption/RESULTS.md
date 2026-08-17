# B3 stage-6 ADOPTION GATE — native-W4 weight-path token quality (Qwen2.5-7B)

**Date:** 2026-07-21 · **Branch:** `comp/b3-weight-path` (RTL stages 0–3 landed
at 4104897, inert) · **Machine:** the 18 GB M-series box, EDA-quiet-gated
**FINAL verdict (all full-set runs landed 2026-07-21/22): realization (A)
tile-scale COLLAPSES the model (z = −17.2 — resolved at n=1000, rejected).
The ADOPTED RECIPE is W4 G=32 through the (B)-style chain with
DIRECT-FROM-SOURCE host prep: −0.0104 ± 0.0022 vs the model's shipped
mlx-4bit weights (z = −4.73) — a −1.0-pt cost, smaller than KVQ4's
accepted −1.17, and BETTER than the current per-tensor INT8 golden feed
(+0.0077, z = +2.75) at ~half its weight traffic. The INT8-hop prep
variant is DEPRECATED for product use: routing W4 through the per-tensor
INT8 intermediate costs an extra −1.7 pts (z = +5.85 for removing it).
Group size measured on the ADOPTED chain (07-22 pm run): direct-G16 =
−0.63 pt vs shipped weights (z = −3.46), +0.41 over direct-G32
(z = +2.05, borderline) at +11% weight traffic — RECOMMENDATION: freeze
the feeder at G=32; G=16 stays a validated quality option. The matrix is
COMPLETE: every cell measured at n=10,042, nothing extrapolated.**

## Setup

Harness: `eval/kv_eval/run_hellaswag_w4.py` — byte-for-byte the S5 scoring
semantics (`docs/results/s5_eval7b/`: one full-sequence forward per
(context, ending), fresh cache, batch 1, mlx-lm whole-string tokenization),
with the KV cache an **identity hook** in every tier so deltas isolate the
WEIGHT path (the mirror image of S5's design). mlx 0.32.0 / mlx_lm 0.31.3 /
lm_eval 0.4.12 (version-equality vs the committed S5 base record asserted at
preflight). Weight tiers are installed by `eval/kv_eval/w4_weights_mlx.py`,
which constructs the model's own quantized representation (q, scales, biases)
**directly from the golden chain** — no re-quantization hop; the only
narrowing is the f64→fp16 RNE cast of the composite scale (measured
≤4.9e-4 relative on every tensor, i.e. fp16 unit roundoff; the same 11-bit
grade the hardware `f16_grade` applies).

**Gates (all counts recorded in each JSON, verbatim lines in
`matrix_run.log`):**

- S5 KV twin gate re-run per tier: `TWIN BIT-EXACT GATE: checks=2254080 fails=0`
- Weight twin gate (vectorized chain vs golden `weight_codec` on synthetic
  tiles incl. all-zero/amax corners): `checks=583760 fails=0`
- MLX packed-format proof vs `mx.quantize`/`mx.dequantize`: `checks=8192 fails=0`
- Install gate per W4 tier: 197 tensors, 586 sampled stripes (2 random + the
  amax stripe per tensor) re-derived through the actual golden
  `compress_weights_w4`/`wfeed_w4_to_i8`: **54,002,836 (tile-A) /
  54,002,250 (G-tiers) checks, 0 fails**; dequant probe 0 ulp; 0 subnormal
  scales.
- Harness equivalence: n=100 preflight reproduced the committed S5 base
  per-doc outcomes exactly (`PREFLIGHT HARNESS-EQUIVALENCE: docs=100
  mismatches=0 PASS`) — this licenses pairing tier runs against the
  committed S5 base samples.

## Tiers

| tier | chain |
|---|---|
| base | stock mlx-4bit g64 checkpoint — **the model's real shipped weights, THE baseline** (committed S5 base samples reused; preflight-licensed) |
| w-int8 | the S8 golden feed: mlx-4bit dequant → per-tensor symmetric INT8, bytes read from the committed `build/s8_weights` cache (unfolded s_w) |
| w4-tile-a | **B3 as landed**: INT8 → `weight_codec` realization (A), ONE fp16 scale per job tile = 8 output columns × full contraction K (required by (A): K-split partials sum raw INT32 before the single epilogue factor) |
| w4-g32-b | the lever: INT8 → G=32 K-groups → realization-(B) dequant + one INT8 requant per 8-col × full-K stripe |
| w4-g16-b | same, G=16 |

⚠️ **(B)-tier realization disclosure:** at this model's K (3584/18944 >
K_MAX 2048) a D-021 feeder-**derived** per-job scale cannot ride the single
K-split epilogue factor, so the G-tiers model a **host-computed stripe-global
requant scale consumed as a sideband** — a design delta from D-021; no (B)
RTL exists (stages 1–3 built realization (A) only). This is the conservative
reading for the lever: a per-job-scale design would use finer scales,
same-or-better accuracy. Scope: the 7 projections × 28 layers + lm_head
(197 tensors — matching both the S8 feed and the perf model's W4 byte
accounting); embedding untouched (lookup, not a streamed GEMM).

## DEFINITIVE results — full validation set (n=10,042, run 2026-07-21)

Same harness, same gates (install gate re-run per tier: 54,002,250 golden
stripe checks / 0 fails for w4-g32-b; preflight re-passed 100/100 at the
10k launch), full 10,042-doc HellaSwag validation split, paired per-doc vs
the committed S5 base samples:

| tier | acc_norm | paired Δ vs base ± SE | z | agree / right-only / wrong-only |
|---|---|---|---|---|
| base (stock mlx-4bit) | 0.7814 | — | — | — |
| w-int8 | 0.7634 | **−0.0180 ± 0.0026** | **−7.06** | 9,381 / 240 / 421 |
| w4-g32-b (via INT8) | 0.7543 | **−0.0271 ± 0.0027** | **−10.17** | 9,320 / 225 / 497 |
| w4-g16-b (via INT8, run 07-22) | 0.7628 | **−0.0186 ± 0.0025** | **−7.37** | 9,395 / 230 / 417 |
| **w4-direct-g32-b (run 07-22)** | **0.7711** | **−0.0104 ± 0.0022** | **−4.73** | 9,558 / 190 / 294 |
| w4-direct-g16-b (run 07-22 pm) | 0.7751 | **−0.0063 ± 0.0018** | **−3.46** | 9,711 / 134 / 197 |

Tier-vs-tier contrasts (all n=10,042 paired):

| contrast | paired Δ ± SE | z | reading |
|---|---|---|---|
| w4-g32-b vs w-int8 | −0.0091 ± 0.0023 | −4.01 | W4 increment inside the hop chain: real |
| w4-direct-g32-b vs w4-g32-b | **+0.0167 ± 0.0029** | **+5.85** | removing the INT8 hop from W4 prep: +1.7 pts |
| w4-direct-g32-b vs w-int8 | **+0.0077 ± 0.0028** | **+2.75** | **W4-direct BEATS the current INT8 feed** |
| w4-g16-b vs w4-g32-b | +0.0085 ± 0.0021 | +4.09 | group size G=32→16: +0.85 pts (via-INT8 chain) |
| w4-direct-g32-b vs w4-g16-b | +0.0083 ± 0.0027 | +3.01 | direct@G32 still beats via-INT8@G16 |
| w4-direct-g16-b vs w4-direct-g32-b | +0.0041 ± 0.0020 | +2.05 | group-size knob on the ADOPTED chain: +0.41 pt, borderline |
| w4-direct-g16-b vs w-int8 | +0.0118 ± 0.0026 | +4.49 | direct@G16 beats the current feed by 1.2 pts |

**Group-size call (completes the matrix):** direct-G16 lands at −0.63 pt vs
the shipped weights — the best weight-path quality measured — but its gain
over direct-G32 is +0.41 pt at z = +2.05 (nominal p ≈ 0.04, the weakest
resolved effect in this table) for +11% weight traffic (5.0 vs 4.5 b/w,
≈ −4–5% decode against the ≥10 tok/s floors). **Recommendation: freeze the
feeder at G=32 (−1.0 pt, 4.5 b/w); G=16 is a validated quality option if a
future configuration has bandwidth headroom.** Owner may overrule at
integration; both cells are now measured, nothing is extrapolated.

**Definitive reading:**
- Every component is now resolved: the per-tensor INT8 hop costs a real
  −1.8 pts; W4-G32 adds a real −0.9 pts on top; total −2.7 pts vs the
  shipped weights. For scale: KVQ4's definitive full-set cost is −1.17 pts
  and it ships (with KVQ4+ as the roadmap answer); KVQ8 is −0.05 (noise).
- The n=1000 phase's z = −0.38 for the W4 increment was under-powered, as
  was the "G=16 adds nothing over G=32" call (made at a resolution that
  could not see 0.3-pt effects) — a G=16 full set is an open option below.
- **The dominant cost is the INT8 intermediate, not 4-bit width — MEASURED
  and confirmed (07-22 runs).** The hardware never requires W4 codes
  derived via the per-tensor INT8 hop (host-prep lineage only). The
  `w4-direct-g32-b` tier — the same G=32/(B) chain sourced straight from
  the mlx-4bit-dequant weights, feeder contract unchanged, gated by
  108,270,351 golden checks / 0 fails with the reference built from the
  cq_codec C-1 primitives — reclaims +1.7 pts (z = +5.85) and lands at
  −1.0 vs the shipped weights, BETTER than the current INT8 feed
  (z = +2.75). Finer-granularity 4-bit beats coarser 8-bit on this model.
- **Group size is a real secondary lever** (+0.85 pts for G=16 over G=32
  inside the via-INT8 chain, z = +4.09, at 5.0 vs 4.5 b/w scale traffic).
  The one unmeasured cell is **direct prep at G=16** — if the group-size
  gain transfers, it could approach parity with the shipped weights; a
  ~3 h optional run before integration freezes the feeder group size.
- w4-tile-a needed no full set (z = −17.2 at n=1000); its rejection
  stands. Absolutes: the full split scores higher than the first-1000
  slice for every tier (base 0.781 vs 0.673) — the slice is not
  representative; deltas remain the comparable quantity.

## Phase-1 results — HellaSwag n=1000, paired per-doc vs committed S5 base
### (superseded on the small deltas by the DEFINITIVE section above)

| tier | acc_norm | paired Δ vs base ± SE | paired z | agree / right-only / wrong-only |
|---|---|---|---|---|
| base | 0.6730 | — | — | — |
| w-int8 | 0.6580 | −0.0150 ± 0.0090 | −1.67 | 919 / 33 / 48 |
| **w4-tile-a** | **0.3510** | **−0.3220 ± 0.0187** | **−17.20** | 546 / 66 / 388 |
| w4-g32-b | 0.6550 | −0.0180 ± 0.0093 | −1.94 | 914 / 34 / 52 |
| w4-g16-b | 0.6570 | −0.0160 ± 0.0087 | −1.84 | 924 / 30 / 46 |

Tier-vs-tier contrasts (paired on the same 1000 docs):

| contrast | paired Δ ± SE | z | agree / r / w |
|---|---|---|---|
| w4-g32-b vs w-int8 (incremental W4 cost over the INT8 feed) | −0.0030 ± 0.0079 | −0.38 | 937 / 30 / 33 |
| w4-g16-b vs w-int8 | −0.0010 ± 0.0073 | −0.14 | 947 / 26 / 27 |
| w4-g16-b vs w4-g32-b | +0.0020 ± 0.0074 | +0.27 | 946 / 28 / 26 |

**Reading (wording per S5 rules — no "within noise" at n=1000):**

- **w4-tile-a is definitively broken**: −32.2 pts, z = −17.2 (388 docs flip
  wrong vs 66 right). No n=10,042 run is needed to resolve this verdict. Not
  a harness artifact: w-int8 ran the *identical* install plumbing and scored
  0.658, and the installed W4 values are golden-gated (54M checks/0 fails on
  the real bytes). Stage-0's §F saw tile-A at only ~1.4–1.8× the G=16
  projection error — the token-level collapse shows per-projection RMS error
  does not predict end-to-end quality (which is precisely why this gate
  exists; §F could not and did not claim otherwise).
- **The finer-grouping lever works and saturates at G=32**: w4-g32-b's
  incremental cost over the current INT8 golden feed is −0.003 ± 0.008
  (z = −0.38); its cost vs the stock 4-bit checkpoint is −0.018 ± 0.009
  (z = −1.94), the same class as the INT8 feed's own −0.015 ± 0.009
  (z = −1.67). Neither of the ~1.5-pt-class deltas is resolved at n=1000 —
  the full set decides them. G=16 vs G=32: +0.002 (z = +0.27), nothing.
- **New finding about the CURRENT feed:** the per-tensor INT8 double-quant
  (w-int8) itself shows a −1.5-pt-class delta vs the stock checkpoint —
  unresolved at n=1000 but consistent in sign across all three fine tiers.
  S8's token probes could not see this (greedy tokens matched for 150
  steps). If the 10k run resolves it as real, the honest S8/eval disclosure
  set should note it.

## Token-level sanity (greedy, S8 Moon prompt, 150 tokens, MLX path)

Artifacts: `token_sanity_base_w4-tile-a_w-int8_n150.json`,
`token_sanity_base_w4-g32-b_n150.json` (full texts + token ids + gates).

- **base:** coherent Moon facts throughout.
- **w4-tile-a:** garbage from step 0 — `1訪れ Parenthood status*
  0<|endoftext|>*…` — human-visible corroboration of the benchmark collapse.
- **w-int8:** first divergence at step 9; stays coherent (drifts into a
  repetition loop late, as greedy decoding does).
- **w4-g32-b:** first divergence at step 17; coherent and factually correct
  continuation (Armstrong 1969 / Cernan 1972), one repeated wrong figure
  ("1.27%") of the same character as base's own greedy artifacts.

⚠️ **Scope:** these streams run the MLX forward path, NOT the golden
fixed-point pipeline. Golden's `_proj_epilogue` carries ONE scalar `s_w` per
tensor (frozen C-2 contract), so realization-(A)'s per-stripe scales cannot
thread through `run_tinynpu.py` without editing golden — the golden-pipeline
W4 token stream is **stage-5 integration work** (where the per-job scale
plumbing lands). What these streams show is the token-level effect of the W4
weight *values*, installed by the same certified installer as the benchmark.

## Ship recommendation (owner decision pending the full set)

1. **Do NOT adopt realization (A) tile-scale as the product weight config.**
   The stage-1–3 feeder RTL remains verified and inert — its unpack datapath
   is reusable — but the (A) scale geometry (one scale per K×8 job tile,
   which at 7B geometry means one scale per 3584–18944×8 tile) is
   quality-fatal on the real model.
2. **The adoption path is G=32 K-grouping through a (B)-style chain**
   (dequant + requant at the feeder, or an equivalent host-side pre-pass),
   with the scale a host-computed stripe-global sideband (see disclosure —
   a per-job-scale design would be same-or-better). This means new feeder
   work at integration: the D-021 `seam_feeder_quant` chain reused for
   weights, consuming a sideband scale rather than deriving it.
3. **Perf-model flag:** the S7 model prices W4 at 4.125 b/w = 4 + 16/128
   (G=128-amortized scales). G=32 stores 4 + 16/32 = **4.5 b/w (+9.1% weight
   bytes)**; decode is weight-BW-bound, so the modeled 12–14 tok/s shifts
   down ~8% (still above the ≥10 floor). Refresh `perf/apex_perf_model.py`
   with a G-parameterized W4 mode before quoting any W4-dependent number.
4. ✅ **DONE 2026-07-21 evening:** the n=10,042 runs of `w-int8` and
   `w4-g32-b` landed — see the DEFINITIVE section. Public wording quotes
   only from there: total −2.7 pts vs shipped weights (−1.8 INT8 hop
   + −0.9 W4-G32), all components definitively real, paired z-scores.
5. ✅ **Follow-ups MEASURED 2026-07-22 (both at n=10,042; DEFINITIVE
   section):** (a) W4-direct-from-source prep reclaims +1.7 pts and beats
   the current INT8 feed — **direct prep is now the REQUIRED host recipe
   for the adopted config; the INT8-hop prep is deprecated for product
   use** (it remains the eval's attribution tool). (b) G=16 recovers
   +0.85 pts over G=32 within the via-INT8 chain (z = +4.09).
6. ✅ **direct@G16 MEASURED 2026-07-22 pm (n=10,042):** −0.63 pt vs
   shipped weights (z = −3.46), +0.41 over direct-G32 (z = +2.05,
   borderline) at 5.0 vs 4.5 b/w. **Freeze at G=32** (recommendation —
   the marginal gain does not justify +11% traffic against the ≥10 tok/s
   floors); G=16 is a validated fallback if BW headroom appears. A G=16
   PERF_MODEL row is still required before quoting any G=16 perf number.

## Disclosures

- All S5 disclosures carry over: weights are 4-bit MLX (the baseline IS the
  shipped 4-bit checkpoint — this gate measures the *additional* cost of the
  B3 chain, which is double-quantized mlx4→INT8→W4); absolutes are not
  comparable to S4/HFLM numbers; the first-1000 doc slice is not
  representative of the full split (base 0.673 here vs 0.781 full-set) —
  deltas, not absolutes, are the comparable quantity.
- All tiers including base share the same fp16 dequant grid (the model's own
  quantized-weight format), so tier deltas are format-symmetric; MLX kernel
  internals may hold more precision than fp16 — measured 0 ulp off the
  expected grid at every probe.
- The (B) tiers model a host-computed stripe-global requant scale (design
  delta from D-021, conservative for the lever) — see Tiers section.
- Token streams are MLX-path, not golden-pipeline (see Token-level sanity).
- The committed S8 weight cache is the INT8 source of truth
  (`build/s8_weights/Qwen2.5-7B-4bit`, regenerable via
  `python run_tinynpu.py prepare`); its per-tensor scales are consumed
  unfolded (gamma folds are golden-pipeline-internal and pow2-exact).

## Reproduce

```
source ~/.venvs/apex-eval/bin/activate
# S8 weight cache required at build/s8_weights/Qwen2.5-7B-4bit
mkdir -p docs/results/b3_w4_adoption && \
( nohup caffeinate -ims bash eval/kv_eval/run_w4_matrix.sh \
    > docs/results/b3_w4_adoption/nohup.out 2>&1 & )   # quiet-gated
python eval/kv_eval/w4_token_sanity.py --tokens 150 --tiers base,w4-g32-b
python eval/kv_eval/w4_paired_stats.py \
    --base docs/results/s5_eval7b/Qwen2.5-7B-4bit_base_n10042.samples.json \
    docs/results/b3_w4_adoption/*_n1000.samples.json
```
