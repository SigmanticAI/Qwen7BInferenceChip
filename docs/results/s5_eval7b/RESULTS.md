# S5 — EVAL-7B: Qwen2.5-7B HellaSwag through the verified KVQ codec

**Date:** 2026-07-14
**Setup:** Qwen2.5-7B (base, `mlx-community/Qwen2.5-7B-4bit` — **4-bit MLX affine
weights, group_size 64; NOT fp16**, see disclosures), MLX 0.32.0 / mlx-lm 0.31.3,
lm-eval-harness 0.4.12, HellaSwag validation (`"task": "hellaswag"` recorded in
each result JSON), `--limit 1000`, batch_size=1. stderr in the JSONs is
lm-eval's analytic binomial √(p(1−p)/(n−1)) ≈ ±0.015 (recorded as
`stderr_method`); the sharper PAIRED statistics below come from the committed
per-document outcome files (`*.samples.json`).
**Model shape:** 28 layers, 28 Q heads / 4 KV heads (GQA — the hook injects
per-KV-head tensors, so no GQA support is needed in the codec), hidden 3584,
**head_dim = 128** (the D=128 path).
**Codec path:** every layer's post-RoPE K/V routed through the same vectorized
twin of `golden/apex_golden/cq_codec.py` used by S4. The bit-exact gate re-ran
before each run and its counts line is preserved in every JSON's `twin_gate`
field: `TWIN BIT-EXACT GATE: checks=2254080 fails=0` (case grid — D∈{64,128},
T∈{1,7,64,128,130,257}, 4 data kinds, all tiers — enumerated in
`eval/kv_eval/test_twin_bitexact.py`, the gate script). Injection point is
mlx-lm's `KVCache.update_and_fetch` (keys stored post-RoPE, as in the HF
path); storage stays raw fp16 and the codec is applied to the full buffer at
read (cache-at-rest, G=128 key groups aligned to position 0). Baseline runs an
identity hook on the identical code path; the pre-flight on the 7B itself
verified identity-hooked logits are **bitwise-equal** to no-cache logits
(`MLX IDENTITY CHECK: PASS` in `matrix_run.log`), so deltas isolate the codec.

## DEFINITIVE results — full validation set (n=10,042, run 2026-07-17/18)

Same harness, same gate-enforced twin codec (`TWIN BIT-EXACT GATE:
checks=2254080 fails=0` re-run per tier), full 10,042-document HellaSwag
validation split; paired per-document stats from the committed
`*_n10042.samples.json` files:

| tier | acc_norm | paired Δ vs base ± SE | paired z | agree / right-only / wrong-only |
|---|---|---|---|---|
| base (4-bit weights, identity KV) | 0.7814 | — | — | — |
| KVQ8 | 0.7809 | **−0.0005 ± 0.0009** | **−0.57** | 9,965 / 36 / 41 |
| KVQ4 | 0.7698 | **−0.0117 ± 0.0020** | **−5.87** | 9,645 / 140 / 257 |

**Definitive reading:**
- **KVQ8: no detectable effect, now at full-set power** (z = −0.57; 9,965 of
  10,042 documents score identically). The n=1000 run's +0.004 was noise in
  the impossible direction; at 10k the estimate sits at −0.0005, within
  ±0.0009.
- **KVQ4: a small, definitively real degradation of ~1.2 points** (z = −5.9,
  p ≈ 4×10⁻⁹) — *smaller* than the n=1000 estimate of −0.017. Still the
  reason KVQ4+ exists; the D=128 loadable mask remains scheduled work.
- Absolute scores are higher than the n=1000 subset's (0.781 vs 0.673 base):
  the first-1000 slice is not representative of the full split. Deltas, not
  absolutes, remain the comparable quantity (weights are still 4-bit MLX —
  all original disclosures below apply).

## Initial run (acc_norm, n=1000) — SUPERSEDED by the full set above

| tier | acc_norm | Δ vs base (unpaired) | paired Δ ± SE | paired z |
|---|---|---|---|---|
| base (4-bit weights, identity KV) | 0.6730 | — | — | — |
| KVQ8 | 0.6770 | +0.004 | +0.0040 ± 0.0035 | +1.15 |
| KVQ4 | 0.6560 | −0.017 | −0.0170 ± 0.0079 | −2.15 |

(acc, unnormalized: 0.5120 / 0.5140 / 0.5030. Paired stats from the
per-document `*.samples.json`; discordant docs — KVQ8: 8 right-only vs 4
wrong-only, 988 agree; KVQ4: 23 vs 40, 937 agree.)

## Pilot honest reading (n=1000) — SUPERSEDED; use the DEFINITIVE section's wording

- **KVQ8 shows no detectable effect on the 7B model** (paired z = +1.15, and
  the sign is the impossible direction — quantization cannot add information,
  so the +0.004 is noise). 988/1000 documents score identically to baseline.
- **KVQ4 shows a small degradation that the paired test resolves as likely
  real**: −0.017 ± 0.008 paired (z = −2.15, nominal p ≈ 0.03; 40 vs 23
  discordant documents). Do NOT call KVQ4 "within noise" at n=1000 — the
  single-run ±0.015 stderr hides what the paired design sees. A −1.7-point
  HellaSwag cost from 4-bit keys+values is consistent with this method
  family's own register: D-022 documents KVQ4-alone as the quality-fragile
  tier — which is exactly why KVQ4+ (outlier lanes) exists. KVQ4+ at
  head_dim=128 is deferred until S12 (maskless D=128 RTL build), disclosed
  below.
- To our knowledge this is the **first public 7B-model accuracy measurement in
  this ChannelQuant-family method line** (prior figures in this line cover
  0.5B/1.5B models; the eval repository behind them is not currently public).
  Wording per the claim ladder: "demonstrated on a real 7B model" — never
  "runs Qwen-7B" / "7B chip" unqualified.
- The full 10,042-document run this section called for has now been executed
  (2026-07-17/18) — its results are the DEFINITIVE section at the top of
  this file, which supersedes the estimates here
  (`LIMIT=10042 caffeinate -ims bash eval/kv_eval/run_matrix_7b.sh`,
  ~3 h/tier measured).

## Reproduce

```
source ~/.venvs/apex-eval/bin/activate
caffeinate -ims bash eval/kv_eval/run_matrix_7b.sh   # waits for EDA-quiet, skips completed runs
```

Determinism: the matrix was run twice end-to-end on this machine (first run
archived in `run1_no_samples/`, before per-sample logging existed); the re-run
reproduced **identical** accuracies for all three tiers.

## Disclosures

- **Weights are 4-bit** (MLX affine, group_size 64), not fp16: 7B fp16
  (≈15 GB at 2 bytes/param — estimate) does not fit this 18 GB machine. The
  baseline tier runs the SAME 4-bit weights with an identity KV hook on the
  identical code path, so tier deltas isolate the KV codec; but absolute
  scores carry the weight-quantization penalty and are NOT an fp16-weights
  measurement. An fp16-weight replication needs a GPU rental (optional S5
  follow-up).
- **KVQ4+ is deferred at this head_dim**: the D=128 RTL build ships maskless
  until S12, so only KVQ8/KVQ4 are reported. Given the paired KVQ4 result
  above, the 7B 4-bit-KV quality story runs through KVQ4+ — S12 matters.
- **Absolutes are not comparable to S4's numbers**
  ([../s4_head2head/RESULTS.md](../s4_head2head/RESULTS.md)): different eval
  adapter (mlx-lm whole-string tokenization — context+ending encoded
  together, split at the context token count — vs HFLM's separate-encode
  convention) on top of the weight-precision difference. Deltas, not
  absolutes, are the comparable quantity. For scale: S4's worst tier delta on
  fp16 weights was also −0.017 (Qwen2-1.5B KVQ8, unpaired).
- The scorer was cross-validated against stock `mlx_lm.evaluate` at n=50:
  identical acc/acc_norm (0.42/0.54) — both artifacts committed under
  `crossval_n50/`.
- fp32 dequant is cast back to fp16 (model dtype) — one extra RNE our
  hardware's fp32 read bus does not have; conservative against us.
- Full-sequence scoring quantizes each G=128-token key group at once
  (cache-at-rest semantics), one fresh-cache forward per (context, ending) —
  deliberately NOT mlx-lm's shared-prefix scoring path, which would quantize
  the same position under two different partial-group scales across forwards.
- Runs were serialized against the concurrent Verilator session via a quiet
  gate (10 consecutive EDA-quiet minutes before model load); the gate log is
  committed alongside this file (`matrix_run.log`, both runs).
