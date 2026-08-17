# S8 — GOLDEN-7B-TOKEN artifact run: 150-token chunk-crossing generation + RTL replay

**Date:** 2026-07-16 · **Commit at run:** 79e8b2b · **Tier:** KVQ8 (`CQ-8`), G=128
**Command:** `caffeinate -ims bash scripts/s8_artifact_run.sh` (quiet-gated: 10
consecutive EDA-quiet minutes before model load; full gate history in
[`artifact_run.log`](artifact_run.log))

## What ran

`run_tinynpu.py --prompt` streamed a greedy **Qwen2.5-7B** generation through
the golden fixed-point pipeline (28 layers/step, M=1 host-sequenced, KV
through the verified codec): prompt "Here are five interesting facts about
the Moon:\n1." (11 tokens) + **150 generated tokens** → session length
**T = 161, crossing the T=128 chunk boundary** (C-CHUNK per-chunk tile jobs
past step 128). Output is coherent English ("The moon is the fifth largest
moon in the solar system.\n2. …" — full text and token ids in
[`artifact_trace/run.json`](artifact_trace/run.json), `complete: true`).

## Evidence (verbatim from the run log)

1. **Trace self-verification (golden replay):** 18 sampled hardware-shaped
   attention jobs re-loaded and re-run from their `.npz` records —
   `TRACE VERIFY: 18/18 jobs replay bit-exact`. The set includes the two
   chunk-crossing heads, each traced as one job **per chunk**
   (`job_s140_L14_h05_c0` T=128 + `_c1` T=13; `job_s151_L04_h07_c0` T=128 +
   `_c1` T=24) — T>128 decode is evidenced end-to-end, not just gated in the
   golden suite.
2. **RTL replay of the real-model rows:** the trace's fp16 K/V rows replayed
   through the actual KVQ cores RTL (`cq_value_path`/`cq_key_path`,
   Verilator 5.044, `verif/kvq/replay/`):
   - vector gen self-check: `SELF-CHECK OK: every stored blob field matches
     a fresh golden compression of the raw rows` (NV=2,940 rows, D=128,
     CQ-8) — [`gen_artifact.log`](../../../verif/kvq/replay/logs/gen_artifact.log)
   - simulation: `CONFIG artifact: checks=382200 fails=0` ·
     `CORES RESULT [artifact]: PASS` ·
     `S8 RTL REPLAY [artifact]: real-model KVQ rows bit-exact in RTL`
     — [`run_artifact.log`](../../../verif/kvq/replay/logs/run_artifact.log)

**Chain closed:** real 7B model → golden fixed-point pipeline (arbiter) →
traced hardware-shaped jobs → bit-exact golden replay (18/18) → real-model
KV rows bit-exact through the RTL datapath (382,200 checks / 0 fails).

## Float64-yardstick probes (worst-element per-layer relative error)

`--ref-check 3 --ref-every 50` compares each layer's fixed-point output
against a float64 reference at probe steps. Worst single element per layer
(median across the 28 layers in parentheses):

| step | worst layer | worst-element rel | median layer rel |
|---|---|---|---|
| 0 | L26 | 1.28 | 0.006 |
| 1 | L27 | 0.42 | 0.061 |
| 2 | L27 | 0.27 | 0.058 |
| 50 | L03 | 0.64 | 0.035 |
| 100 | L03 | 0.39 | 0.043 |
| 150 | L03 | 0.52 | 0.067 |

Consistent with the phase-1 finding (committed in `smoke_trace/run.json`):
worst-element outliers concentrate in layers {0–3, 26–27}, where real-weight
activation outliers meet per-token INT8 feeders; medians stay at the few-%
level and greedy token selection is unaffected across all 150 steps (the
generation matches step-0-onward greedy continuation; determinism selftest
in `run_tinynpu.py`). These are worst-ELEMENT numbers, not norms — the D-023
e2e budget applies to the attention-core jobs, which replay bit-exact.

## Scope (what this does and does not show)

- Tier is KVQ8: the RTL replay exercises the 8-bit record path on real-model
  rows (NG=0 — grouped-key CQ-4/4+ paths carry their own coverage in
  `verif/kvq/cores` and `verif/kvq/sb`, synthetic vectors).
- The golden pipeline is the arbiter, not the RTL, for full-layer execution;
  RTL coverage here is the KVQ datapath + the traced attention jobs' golden
  replay. "Demonstrated on a real 7B model" per the approved scope sentence —
  never "runs Qwen-7B" as a chip claim.

## Reproduce

```sh
caffeinate -ims bash scripts/s8_artifact_run.sh          # generation + trace verify
make -C verif/kvq/replay gen NAME=artifact TRACE=$PWD/docs/results/s8_7b_token/artifact_trace
make -C verif/kvq/replay run NAME=artifact               # QUIET MACHINE ONLY
```
