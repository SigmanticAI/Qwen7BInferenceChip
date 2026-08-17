# B3 stage 0 — native-W4 weight path: golden codec + realization decision

**Date:** 2026-07-20 · **Branch:** `comp/b3-weight-path` · **Base commit:** e578872
**Command:** `make -C golden weightcodec` (also runs inside `make -C golden test`)
**Result:** exit 0 · **187 checks PASS, 0 FAIL** · full verbatim output in
[`weightcodec.log`](weightcodec.log)

Contract: [`docs/design/B3_WEIGHT_PATH.md`](../../design/B3_WEIGHT_PATH.md) §2,
amended in the same commit to match what was measured here.

## What landed

`golden/apex_golden/weight_codec.py` — the arbiter for the W4 weight feeder,
built only on already-trusted primitives (`cq_codec.scale_from_amax` /
`quant_codes(bits=4)` / `pack_int4` / `unpack_int4` / `dequant_f32`, and for
realization (B) the D-021 `attention.quant_rows_i8` rule). It also pins, for
the first time, the **byte-level stream contract S1–S6** that the RTL feeder,
the vector generator and the mutation gate must all agree on — beat order,
flat element index, nibble↔lane mapping, packed-beat count, and the odd-tail
rule. Nothing in the repo defined those before.

## The stage-0 decision: realization (A), tile-wide groups

Measured projection error `max|a·W_rec − a·W_real| / max|a·W_real|`
(activations identical on both sides, so this is the weight path alone),
K=64 N=8 M=32, seed 90210 — verbatim from the run:

```
  gaussian  (column dynamic range 1.5x):
    INT8 baseline  rel_err = 0.00492   (shipped path)
    A tile         rel_err = 0.11542   ( 23.5x INT8)
    A tile  raw    rel_err = 0.11537   ( 23.4x INT8)
    A tile  pow2   rel_err = 0.19695   ( 40.0x INT8)
    B col          rel_err = 0.09987   ( 20.3x INT8)
    B G=32         rel_err = 0.07758   ( 15.8x INT8)
    B G=16         rel_err = 0.07262   ( 14.8x INT8)

  uniform  (column dynamic range 1.0x):
    INT8 baseline  rel_err = 0.00398   (shipped path)
    A tile         rel_err = 0.06541   ( 16.4x INT8)
    A tile  raw    rel_err = 0.06541   ( 16.4x INT8)
    A tile  pow2   rel_err = 0.13297   ( 33.4x INT8)
    B col          rel_err = 0.07817   ( 19.6x INT8)
    B G=32         rel_err = 0.07817   ( 19.6x INT8)
    B G=16         rel_err = 0.06092   ( 15.3x INT8)

  outlier  (column dynamic range 19.1x):
    INT8 baseline  rel_err = 0.00833   (shipped path)
    A tile         rel_err = 0.13109   ( 15.7x INT8)
    A tile  raw    rel_err = 0.13108   ( 15.7x INT8)
    A tile  pow2   rel_err = 0.27894   ( 33.5x INT8)
    B col          rel_err = 0.13171   ( 15.8x INT8)
    B G=32         rel_err = 0.12554   ( 15.1x INT8)
    B G=16         rel_err = 0.09259   ( 11.1x INT8)
```

**1. The (A)-vs-(B) choice is not an accuracy choice.** They land within 25% of
each other on every distribution, and (A) is *better* on uniform and outlier.
The contract's decision rule — "pick (B) only if (A) misses budget" — had no
budget gap to arbitrate and its ordering assumption was wrong. (A) is chosen on
engineering grounds: a pure unpack needs no fp16 arithmetic in the feeder and
no scale sideband, and gives the cleanest mutation gate.

**2. The loss is intrinsic to the 4-bit code width.** (A) tracks
INT4-straight-onto-real-weights within 20% everywhere. Giving realization (B)
a per-column INT8 output scale (which the hardware cannot do) moves gaussian
only 0.09987 → 0.09127 — so the per-tile INT8 funnel is **not** the bottleneck.
The one material lever is finer K-grouping: G=16 beats tile-wide by 37% / 7% /
29%.

**3. The binding hardware constraint is `apex_scale_quant`, not
`mxe_requant`.** Its C2 contract needs the fp32 sideband scale to carry ≤11
significant bits (`frac[12:0] == 0`, `apex_scale_quant.sv:29-31,227`); today's
flows satisfy it only because `s_w` is a power of two
(`gen_l3_vectors.py:589`). An fp16 W4 group scale makes the composite ~22 bits
and trips `scale_error`. Rounding the **composite** to fp16 grade costs
≤ 5e-5 absolute (`A tile` vs `A tile raw` above — free). Snapping the **group
scale** to a power of two instead costs **1.71× / 2.03× / 2.13×** and is
rejected on that measurement.

## The number B3's viability actually turns on

**Native W4 costs 11–24× the INT8 baseline's projection error** — 0.061–0.132
vs 0.004–0.008 — on every distribution and every realization tested. No
realization choice available to the feeder changes that materially.

This is **not** an accept/reject verdict. No existing gate can produce one:
D-023's ≤5%-of-value-scale is an attention/V-path metric, and every float64
reference in the repo is built from the same dequantized weights
(`transformer.py:519-521`, `attention.py:538-541`), so a W4 error measured
against them cancels identically and reads the INT8 plumbing baseline instead.
Deciding adoption needs an e2e token-quality run on the real model — added to
the contract as **stage 6**, and it blocks *product adoption*, not the RTL.

## Discipline notes

- The 21 measured values are **pinned** in `test_weight_codec.py:PINNED`
  (tolerance 5e-4) under the `test_effective_bits.py` conscious-update rule.
  They are a regression tripwire, not a quality claim.
- `golden/Makefile:12`'s banner is left **byte-identical** — `gen_status.py:273`
  matches it as a literal substring, so appending to it would silently flip the
  whole STATUS.md golden gate to FAIL. The new gate reports on its own line.
  Verified: the substring is still present and `run_golden()` still reads PASS.
- Full suite re-run green after the change: `make -C golden test` exit 0, all
  seven pre-existing suite banners unchanged.

## Repro

```
git worktree add ../apex-weightpath comp/b3-weight-path
make -C golden weightcodec     # this result, exit 0
make -C golden test            # full golden gate incl. the new target
```
