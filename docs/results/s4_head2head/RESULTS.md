# S4 — Head-to-head KV-compression accuracy: HellaSwag through the verified codec

**Date:** 2026-07-13
**Setup:** Qwen2-0.5B + Qwen2-1.5B (FP16 weights, MPS), lm-eval-harness 0.4.12, HellaSwag
validation, `--limit 1000`, `batch_size=1`, `bootstrap_iters=1000` — the same models,
benchmark, and sample count used by published ChannelQuant-family KV-cache engines.
**Codec path:** every layer's post-RoPE K/V routed through a **vectorized twin of
`golden/apex_golden/cq_codec.py`, certified bit-exact against the golden arbiter
(2,254,080 fp32-bit checks, 0 fails)** — the gate re-runs before every eval
(`eval/kv_eval/test_twin_bitexact.py`); the golden codec is itself what the RTL is
verified bit-exact against. Baseline runs through an identity hook on the identical
code path (identity-hooked logits == no-cache logits exactly, verified), so deltas
isolate the codec alone. Raw per-run JSON (versions, seeds, method) sits beside this file.

## DEFINITIVE results — full validation set (n=10,042, run 2026-07-18)

Same harness and gate-certified twin codec, full HellaSwag validation split
(per-run analytic stderr ≈ ±0.005; delta stderr ≈ ±0.007, unpaired — this
harness does not emit per-document samples):

| model | FP16 base | KVQ8 | KVQ4 | KVQ4+ |
|---|---|---|---|---|
| Qwen2-0.5B | 0.4905 | 0.4817 (−0.0089) | 0.4720 (−0.0185) | 0.4774 (−0.0131) |
| Qwen2-1.5B | 0.6544 | 0.6400 (−0.0143) | 0.6461 (−0.0083) | 0.6462 (−0.0082) |

**Definitive reading (replaces the n=1000 wording):**
- At full-set power the codec's costs resolve as **small but partly real**:
  deltas span −0.008 to −0.019 (1.2σ–2.6σ unpaired). The n=1000 claim
  "statistically indistinguishable from FP16" does **not** survive at
  n=10,042 and is retired; the honest phrase is *near-lossless: worst-case
  ≈1.9 points, typical ≈1 point, measured and reproducible from a clone*.
- **Tier-vs-tier orderings remain noise** even at 10k: on 1.5B, KVQ8
  measures below both 4-bit tiers — physically implausible (KVQ8 is strictly
  finer) — so differences ≤~1 point between tiers still must not be read as
  rankings. The paired 7B measurement (`../s5_eval7b/`) is the sharper
  instrument: KVQ8 −0.0005, KVQ4 −0.0117.
- Published ChannelQuant-family deltas on this setup (−0.004..−0.016) sit in
  the same band as ours — parity, with the same caveat that sub-point
  differences are noise. Never "beats".

## Initial run (acc_norm, n=1000, stderr ≈ ±0.016) — SUPERSEDED by the full set above

| model | FP16 base | KVQ8 | KVQ4 | KVQ4+ |
|---|---|---|---|---|
| Qwen2-0.5B | 0.4880 | 0.4830 (−0.005) | **0.4860 (−0.002)** | 0.4770 (−0.011) |
| Qwen2-1.5B | 0.5910 | 0.5740 (−0.017) | **0.5810 (−0.010)** | 0.5740 (−0.017) |

Published ChannelQuant-family deltas on the same setup, for reference: 0.5B −0.009
(4-bit) / −0.004 (4-bit + outlier lanes); 1.5B −0.016 / −0.008.

## Pilot honest reading (n=1000) — SUPERSEDED; use the DEFINITIVE section's wording

- **Every tier is statistically indistinguishable from FP16 at n=1000** (all |delta| ≤
  0.017 ≈ 1σ). That is the claim: *near-lossless, measured, reproducible from a clone*.
- **Tier-vs-tier orderings at n=1000 are noise.** Our KVQ8 measures below KVQ4 on 1.5B
  and KVQ4+ below KVQ4 on both models — physically implausible (KVQ8 is strictly finer
  quantization; KVQ4+'s outlier lanes are exact-identity on the two largest-amax channels
  and leave all other channels untouched, so its per-tensor fidelity is ≥ KVQ4 by
  construction). Implausible orderings at this n mean the differences between tiers (and
  between our deltas and published ones) are below the noise floor. Do not read any tier
  as "beating" another here; all are ~0 ± 0.016.
- For deltas with teeth (±0.005), run the full 10,042-doc validation set (~overnight on
  this machine) before any public use.

## Reproduce

```
source ~/.venvs/apex-eval/bin/activate
bash eval/kv_eval/run_matrix.sh          # skips completed runs
```

## Disclosures

- Absolute baselines differ from some published numbers (ours 0.4880/0.5910 vs
  0.4260/0.5210 reported elsewhere): different eval stacks (standard lm-eval-harness vs
  other harnesses — likely prompt-format/slice differences). Deltas, not absolutes, are
  the comparable quantity, and everything here re-runs from a git clone.
- fp32 dequant is cast back to fp16 for attention (model dtype) — one extra RNE our
  hardware's fp32 read bus does not have; conservative against us.
- KVQ4+ outlier channels: static top-2 per (layer, head) by |K| amax over 32 HellaSwag
  TRAIN contexts (no test leakage); per-run JSON records the indices.
- Full-sequence scoring quantizes each G=128-token key group at once (cache-at-rest
  semantics); the RTL additionally supports partial-group flush (D-008).
- The D=128 b128 RTL build is maskless today, so KVQ4+ at head_dim=128 (1.5B)
  is golden-side for now — the loadable outlier mask is scheduled work (its
  contract is already pinned golden-first in `golden/tests/test_mask_semantics.py`).
