# APEX-7B spec — floor-traceability table (S13)

> Maps **every floor-bearing line of [`APEX7B_SPEC.md`](APEX7B_SPEC.md)** to the
> anchor in the machine-generated
> [`docs/results/perf_model/PERF_MODEL.md`](../results/perf_model/PERF_MODEL.md)
> (or the CI gate) that it quotes. Rule of the spec, restated: **the spec may only
> quote; it may never originate a performance number.** If `PERF_MODEL.md` is
> regenerated and any verbatim figure below changes, the spec and this table must
> be re-edited — the generated file always wins.
>
> All figures below are **[PROJECTED]** unless tagged PINNED (CI-asserted
> constant), MEASURED (cited artifact), or DERIVED (inline arithmetic on
> PINNED/PROJECTED inputs). PROPOSED lines (design choices with no number) have no
> anchor and are listed in §5 of this table.

## 0. Verification state backing this table

`python3 perf/apex_perf_model.py --check`, re-run 2026-07-14 before this package
was written (verbatim output):

```
CALIBRATION VALIDATION (model must reproduce the measured anchors)
  PASS  ASU divider cycles @T=128 (~540k, OPTIMIZATION.md): model 5.407e+05 vs anchor 5.4e+05
  PASS  divider/MXE cycle ratio (~8x, OPTIMIZATION.md): model 8.318 vs anchor 8.3
  PASS  MXE util 30.6% <-> 3.3x-off-ideal (OPTIMIZATION.md): model 3.268 vs anchor 3.3
  PASS  weight-port roofline @9.04 MHz (tok/s): model 0.01023 vs anchor 0.01
  PASS  weight-port roofline @100 MHz (tok/s): model 0.1131 vs anchor 0.113
  PASS  weight-port roofline @400 MHz (tok/s): model 0.4526 vs anchor 0.452
  PASS  Qwen2.5-7B weight MACs/token (~7.07G): model 7.07e+09 vs anchor 7.07e+09
  PASS  KV values/token (2*4*128*28 = 28,672): model 2.867e+04 vs anchor 2.867e+04
  PASS  KVQ4 D=128 stored ratio (3.5068x, effbits gate): model 3.507 vs anchor 3.507
  -> ALL ANCHORS REPRODUCED
```

Storage-geometry constants are additionally pinned by
[`golden/tests/test_effective_bits.py`](../../golden/tests/test_effective_bits.py)
(the "effbits gate", runs in `make -C golden test`; last verified PASS 2026-07-14,
"ALL PINNED ACCOUNTINGS MATCH" — per the S13 session that authored the spec text).

Anchor links below: [§1](../results/perf_model/PERF_MODEL.md#1-measured-calibration-anchors-the-only-measurements-here) ·
[§2](../results/perf_model/PERF_MODEL.md#2-assumptions-everything-that-is-not-measured) ·
[§3a](../results/perf_model/PERF_MODEL.md#3a-toks-vs-memory-system-and-weight-precision-ctx--2k) ·
[§3b](../results/perf_model/PERF_MODEL.md#3b-toks-vs-context--the-kvq-flat-region-claim) ·
[§3c](../results/perf_model/PERF_MODEL.md#3c-host-sequencing-why-the-walker-b1-is-mandatory) ·
[§4](../results/perf_model/PERF_MODEL.md#4-projected-prefill--ttft) ·
[§5](../results/perf_model/PERF_MODEL.md#5-projected-energy-per-token-vs-context-kvq-onoff) ·
[§6](../results/perf_model/PERF_MODEL.md#6-comparator-table--published-single-stream-7b8b-q4-decode) ·
[§7](../results/perf_model/PERF_MODEL.md#7-projected-multi-tile-scaling) ·
[§8](../results/perf_model/PERF_MODEL.md#8-spec-floors-check-s13-decode-10-toks-to-32k-ctx-ttft1k-5-s-5-w)

## 1. The floors and the sizing decisions (spec §2–§3)

| spec line | figure quoted | anchor | verbatim source figure | tag |
|---|---|---|---|---|
| §2 floors-check table (all four config rows) | 11.4 tok/s / 3.4 s / 2.9 W PASS; 5.5 s FAIL; 11.0 s FAIL; 5.7 tok/s FAIL | §8 table | identical rows, quoted verbatim | PROJECTED |
| §2/§3 clock bound "≥0.65 GHz; 0.5 GHz misses TTFT at 65% util" | 5.5 s @0.5 GHz/65%; 4.0 s @90% | §8 closing note + §4 row `64×64 / 0.5 GHz` | "misses the 5 s TTFT floor at 65% utilization but passes at 90% (4.0 s)"; "robust spec point is **64×64 @ ≥0.65 GHz with LPDDR5X ×64**" | PROJECTED |
| §2/§3 array choice "32×32 fails TTFT at any plausible clock" | 11.0 s @32×32/1.0 GHz (9.2 s @1.2 GHz) | §4 grid + §8 row 3 | "32×32 @ 1 GHz … 11.0 s TTFT at target utilization" | PROJECTED |
| §3 memory row "LPDDR5X ×64, 68.3 GB/s peak, 70% eff (65–75% swept)" | 68.3 GB/s; 0.65/0.7/0.75 | §2 assumption + §3a table row | "peak GB/s = MT/s × bus bytes; sustained efficiency swept (0.65, 0.7, 0.75) (default 70%)" | assumption (PROJECTED inputs) |
| §3 weights row "native W4, 4.125 b/w incl. scales" | 4.125 b/w | §2 assumption (B3) | "B3 native-W4 weight path (4.125 b/w incl. scales)" | assumption, floor-critical |
| §3 sequencing row "B1 walker ≤3 MMIO/layer-step" | ≤3 MMIO/layer-step | §2 assumption + §3c | "walker mode 3 MMIO/layer-step" | assumption, floor-critical |
| §3 headline "decode 12–14 tok/s short-context" | 12–14 | §6 APEX-7B comparator row | "12–14 · ~2.1–3.6 W · ~0.16–0.28 · PROJECTED — no hardware" | PROJECTED |
| §3 headline "11.4 @32k · TTFT(1k) 3.4 s · ~0.16–0.28 J/token at ~2.1–3.6 W" | 11.4; 3.4 s; 0.16–0.28; 2.1–3.6 W | §8 PASS row; §4 row `64×64 / 0.8 GHz`; §5 headline | "~0.16–0.28 J/token at ~2.1–3.6 W (die+DRAM; excludes host SoC)" | PROJECTED |
| §3 sizing logic "prefill sizes the array; decode sizes the memory" | — | §3 intro + §4 intro | "Decode is memory-bound…"; "Prefill is compute-bound…" | model structure |

## 2. Floor-critical dependencies (spec §4)

| spec line | figure quoted | anchor | verbatim source figure | tag |
|---|---|---|---|---|
| B1 walker: "486,016 tile jobs/token … 0.27 tok/s regardless of memory" | 486,016; 5 MMIO/job; 1.5 µs; 3.6 s/token; 0.27 tok/s | §3c | "486,016 tile jobs per decode token … 3.6 s/token of pure transaction time → 0.27 tok/s" | PROJECTED (MMIO count MEASURED, §1) |
| B1 walker: "with it 126 µs/token (~0.2%) → 13.0 tok/s" | 126 µs; 13.0 | §3c | "host overhead 126 µs/token (~0.2% of the token) → 13.0 tok/s" | PROJECTED |
| W4: "INT8 weights on ×64: 6.7 tok/s @2k — below floor" | 6.7 vs 13.0 | §3a table, `LPDDR5X x64` row | "INT8 weights + ×64 projects to 6.7 tok/s — also below floor. W4 (B3) is load-bearing" | PROJECTED |
| ×64: "single ×32 (34.1 GB/s): 6.5 tok/s @2k, 5.7 @32k — fails F1" | 6.5; 5.7; 34.1 GB/s | §3a row `LPDDR5X x32` + §8 row 4 | "A single ×32 LPDDR5X channel projects to 6.5 tok/s — below the 10 tok/s floor" | PROJECTED |
| A1+B4: "at measured 30.6% util, TTFT(1k) = 7.3 s — fails F2; 65% target: 3.4 s" | 30.6%; 7.3 s; 3.4 s | §4 row `64×64 / 0.8 GHz` + §9 limitation 3 | "At today's *measured* 30.6% utilization the same array would need 7.3 s" | 30.6% MEASURED (§1); TTFT PROJECTED |

## 3. Workload contract, KV geometry, memory budgets (spec §5–§6)

| spec line | figure quoted | anchor | verbatim source figure | tag |
|---|---|---|---|---|
| §5 workload: "7.07 G weight-MACs (233.0 M × 28 + 545 M); attention ≈6.58 G @32k" | 7.07 G; 233.0 M; 545 M; 6.58 G | §2 workload bullet (asserted by `--check`) | "Derived weight MACs/token: **7.07 G** … Attention adds … ≈ 6.58 G at 32k ctx" | DERIVED (model-internal, `--check`-asserted) |
| §5 workload: "28,672 KV values/token = 56 KiB FP16, 16.0 KiB stored" | 28,672; 56 KiB; 16.0 KiB | §2 (asserted by `--check`) | "28,672 values/token (GQA), i.e. 56 KiB/token at FP16, 16.0 KiB/token as KVQ4 stored records" | DERIVED/PINNED |
| §6.1 record table (all six tier rows: 320/576/384/640/576/1088 b rows; 3.16×/3.51×/2.64×/3.16×/1.78×/1.88×) | all | effbits gate (not the perf model) | `test_effective_bits.py` pinned accountings; `--check` cross-asserts 3.507× | **PINNED** |
| §6.1 "method ceiling 3.66–3.88×; >4× impossible" | 3.66–3.88 | effbits gate; 3.88 also §1 anchor row | "KVQ4 D=128 codec ceiling · 4.125 b/v = 3.88×" | PINNED |
| §6.2 KVQ SRAM: 9.0 + 2.0 + 32.0 = 43.0 KiB/engine, ×4 = 172 KiB | 43 KiB; 172 KiB | **no model anchor** — arithmetic inline in spec §6.2 on PINNED D-026 geometry (576-b row, G=128, SETS=8, DEPTH=128); memory list per `docs/results/s10_fmax/BASELINE.md` | — | DERIVED |
| §6.2 rest-of-tile SRAM (≈2.2 MiB total) | 1.75 MiB slab etc. | **no anchor — PROPOSED sizes**, sizing rules inline; only the "activations stay on-die" assumption is the model's (§2/§9) | "Activation traffic between layers is assumed on-die" (§9) | PROPOSED |
| §6.3 DRAM budget: W4 ~3.65 GB, embedding ~0.28 GB, KV 512 MiB/1.0 GiB/2.0 GiB @32k/64k/128k, ≥8 GB | all | **no model anchor** — arithmetic inline on §2 constants (7.07 G weights × 4.125 b/w; 16.0 KiB/token) | — | DERIVED |
| §6.3 crossover "KV read crosses weight traffic at ~62k uncompressed / ~218k compressed" | 62k; 218k | §3b closing note | "KV-read traffic crosses weight traffic at **~62k ctx uncompressed** vs **~218k compressed (stored)**" | PROJECTED |

## 4. Projected performance sections (spec §7)

| spec line | figure quoted | anchor | verbatim source figure | tag |
|---|---|---|---|---|
| §7.1 decode-vs-context row (13.1 / 13.0 / 12.6 / 12.2 / 11.4 / 10.1 / 8.2) | all | §3b table (KVQ4-as-stored column at ctx 0/2k/8k/16k/32k/65,536/131,072) | identical cells | PROJECTED |
| §7.1 FP16-KV row (13.1 / 12.7 / 11.6 / 10.4 / 8.6 / 6.4 / 4.3) | all | §3b table, `KV FP16` column | identical cells | PROJECTED |
| §7.1 flat region "~19k FP16 → ~67k KVQ4 stored (~74k ceiling; ~38k INT8 KV)" | 19k; 67k; 74k; 38k | §3b flat-region table | "KVQ stretches the ≥10 tok/s region from ~19k to ~67k context as stored today (~74k at the codec ceiling)" | PROJECTED |
| §7.1 "10 tok/s at 128k is NOT true (it is 8.2)" | 8.2 vs 4.3 | §3b last row + closing | "At 128k ctx: 8.2 vs 4.3 tok/s — KVQ keeps long-context decode ~2× faster" | PROJECTED |
| §7.2 TTFT table (7.3/3.4/2.5 s @0.8 GHz; 11.7/5.5/4.0 s @0.5 GHz) | all | §4 grid rows `64×64` | identical cells | PROJECTED |
| §7.3 energy table (0.16/0.22/0.28 short; 0.18/0.25/0.32 @32k; 0.26/0.36/0.45 @128k; FP16 0.22/0.33/0.67; 2.8–2.9 W) | all | §5 table rows ctx 0–2,048 / 32,768 / 131,072 | identical cells | PROJECTED |
| §7.3 "DRAM 60–75% of power; 4–8 pJ/bit moves J/token ±35%" | 60–75%; ±35% | §2 energy bullet + §5 headline | "DRAM energy dominates (60–75% of total)"; "The DRAM pJ/bit assumption alone moves this ±35%" | assumption sensitivity |
| §7.3 "5–10× less energy vs desktop GPUs, stock, single-stream; GPUs ~8–14× faster; power-capped/TRT stack compresses to ~2–6×" | 5–10×; 8–14×; 2–6× | §6 notes under comparator table | "Desktop discrete GPUs are ~8–14× *faster* … A power-capped 4090 or a TensorRT-LLM/speculative stack compresses the gap to ~2–6×" | PROJECTED vs cited measurements |
| §7.3 "edge parts already at 0.3–0.5 J/token" | 0.30; 0.45–0.5 | §6 rows M4 Max / Hailo-10H | "Apple M4 Max … ~0.30"; "Hailo-10H … ~0.45–0.5" | third-party cited |
| §7.4 multi-tile "TTFT 3.45 s → 0.61 s at 8 tiles; 1,675 prefill tok/s BW ceiling; decode flat 13.0" | all | §7 table | identical cells | PROJECTED |

## 5. Lines with deliberately NO perf-model anchor

| spec line | why no anchor | guard |
|---|---|---|
| §3 array/clock/memory/W4/walker as *design choices* | [PROPOSED] — the model sizes them (anchors above) but nothing is built | spec §5 lineage table marks every unbuilt block |
| §6.2 rest-of-tile SRAM sizes | [PROPOSED] — never quote externally as modeled | flagged in spec §6.2 |
| §8 APEX-Ambient, entire section | **EXPLORATORY** — `apex_perf_model.py` has no LPDDR4X/eviction rows; nothing quotable exists | spec §8 banner: "none of these numbers may leave this document"; promotion needs a perf-model extension + A3 gate |

## 6. Re-verification procedure

```
python3 perf/apex_perf_model.py --check   # must print "ALL ANCHORS REPRODUCED"
make -C golden effbits                    # must print "ALL PINNED ACCOUNTINGS MATCH"
python3 perf/apex_perf_model.py           # regenerate PERF_MODEL.md, then re-diff
                                          # every verbatim figure in this table
```

If any verbatim figure in this table stops matching the regenerated
`PERF_MODEL.md`, the spec is stale: fix `APEX7B_SPEC.md` and this table to match
the generated file (never the reverse), per the project ground rule "golden/generated
artifacts are the arbiter."

*Hand-written (S13). Last cross-checked against `docs/results/perf_model/PERF_MODEL.md` as generated by commit b9f7540's model on 2026-07-14, `--check` PASS pasted above.*
