# TRANSPORT COLLAPSE — 44 EXECUTOR INVOCATIONS PER LAYER DOWN TO 5, SAME JOB SET

**Date:** 2026-08-04 · **Branch:** `comp/prompt-b-c` · **Executor:** `sim`
(b64_05b twin, `verif/f2sim/obj_b64_05b/f2sim`) · **No hardware was touched by
the work in §1-§4** (the same-day card session that settled §4's prediction is
recorded in §4b, log `p05b_hw_check2.log` / `p05b_fat_hw2.log` /
`p05b_collapse_hw.log` committed beside this file). · **Drivers:** `scripts/fpga/f2/layer_offload.py`,
`layer05b.py`, `prompt05b.py` · **Logs:** `build/p05b_collapse2.log`,
`build/p05b_nocollapse2.log`, `build/c2_collapse.log` ·
**Records:** `build/p05b_collapse2/prompt05b_result.json`,
`build/p05b_nocollapse2/prompt05b_result.json`,
`docs/results/prompt_on_chip/collapse_hw_prediction.json`.

`build/` is gitignored, so every run below is committed verbatim beside this
file: `collapse_sim.log` / `collapse_result.json`, `nocollapse_sim.log` /
`nocollapse_result.json`, `recon_sim.log` (six families) and `c2_sim.log` /
`c2_result.json` (the 7B regression); the prediction transcript is
`docs/results/prompt_on_chip/collapse_prediction.log`.

## 1. The measured bottleneck this attacks

The fat/burst projection transport got a 0.5B layer-step from 2,160 tile
programs to 25 and the hw per-layer wall from 578 s to a measured 193.7 s. The
fitted transport model
(`docs/results/prompt_on_chip/fat_hw_prediction.json`) says what is left:

```
wall ≈ 4.8 s × invocations + 45.8 ms × programs + 22.1 µs × PEEKS
```

At 37 invocations per layer that is **177.6 s of the 184.9 s prediction — 96%
of the residual is per-INVOCATION setup** (ssh, remote python start, SDK
attach), and **30 of the 37 invocations were single-program calls**: 14 RoPE
rows, 14 attention heads and 2 residual windows, each entered on its own
because `layer_offload`'s wrappers need each value before golden emits the
next operand.

## 2. Why they could not simply be gathered — and what was done instead

golden calls RoPE and attention inside ONE per-head loop and consumes each
value before the next operand exists (`transformer.py:535-553`). At the first
head there is nothing to gather.

So the layer is **REPLAYED** instead. `layer_offload.Runner` gained a capture
pool; `Runner.acquire()` is `run()` with a memory and answers exactly one
question — *is this program's capture set already in hand?* On a miss it
QUEUES the program and says so, and the op wrapper leaves that value on the
host **for that pass**. `LayerOffloader._layer` replays the layer until a pass
serves every program it asks for; each replay costs one `flush()`, and a flush
is **one executor invocation per `--max-batch-mb` of regops**.

Measured convergence for the four exact-egress families: **3 passes, 2 flushes,
5 invocations per layer** (the 64 MiB size bound splits the pass-1 flush of
214 MB of projections into 4). Pass 1 discovers 54 programs — the golden
fallback trajectory IS the tile trajectory for the bit-exact families, so
those programs are already final. Pass 2 discovers 26 more: the 14 attention
programs, whose q row is now the tile's own C-1 reconstruction and therefore
honestly a different operand; the 11 fat-vs-thin cross-check references, which
pass 1 never reached because `_gemm` returns before the cross-check while its
own programs are still queued; and the `r2` residual window, whose input row
does not exist until `r1` is served. Pass 3 is served entirely from the pool
and enters no executor at all.

The cost of the replay is visible in the same numbers: **80 programs executed
per layer against 66 for the per-op transport** — the 14 pass-1 attention
programs are superseded and thrown away. On the fitted hw model that waste is
0.6 s/layer of per-program time, against at least 154 s/layer of invocation
setup removed (32 fewer entries than the model's 37-invocation baseline, 39
fewer than this run's measured 44). The prediction below is charged for the
waste, not credited with the smaller final program set.

**What a pooled capture asserts.** The pool is keyed by
`layer_offload.program_key` = the program's NAME and its exact BYTES. Reusing
an entry claims only that an executor ran a byte-identical program, in this
session, on this image, under the per-file `TILE_RST` discipline both
executors apply before every file (`f2_host_run.py:146`; `sim_main.cpp`
reset-per-file). Every decode, every grade, every substitution and every
consumption check is re-run from those captures on the pass that is kept.
A pass that could not serve everything is **DISCARDED**, not reported as
partial coverage.

## 3. The A/B — same job set, same code, one flag apart

```
python3 scripts/fpga/f2/prompt05b.py --ids 785 6722 315 9625 374 \
    --layers 0-1 --executor sim --cross-check --poison 0.5 \
    --work-dir build/p05b_collapse2            #  (--no-collapse for arm B)
```

| | `--no-collapse` | collapse (default) |
| --- | ---: | ---: |
| **executor invocations / layer** | **44** | **5** |
| executor invocations, whole run (2 layers + control + poison arms) | 162 | 13 |
| replay passes / layer | 1 | 3 |
| programs executed / layer | 66 | 80 |
| distinct programs, whole run | 171 | 199 |
| per-layer wall (sim) | 112.6 / 112.9 s | 115.4 / 115.3 s |
| executor time, whole run (sim) | 111.7 s | 110.6 s |
| whole-process wall (sim) | 256.9 s | 261.0 s |
| substitution / consumption checks | 82/82 PASS | 82/82 PASS |
| fat-vs-thin cross-checks | 14/14 EQUAL | 14/14 EQUAL |
| geometry audit (INFO_D, per program) | PASS, 0 unaudited | PASS, 0 unaudited |
| projections served | 27,392/27,392 bit-exact | 27,392/27,392 bit-exact |
| TOKEN == PURE HOST | PASS | PASS |
| poison discriminator | `[12095]` → `[3881]`, max&#124;dlogit&#124; 8.543 | identical |

The two result JSONs were diffed field by field: `tokens_on`, `tokens_host`,
`tokens_bus_on`, `logit_geometry`, `flop_share`, `op_families_served`,
`per_layer_ops` and the whole `poison` block are **identical**. The only
differing ledger entries are wall-clock seconds and the `emission`
provenance tag (`emitted` vs `operand memo (not re-emitted)`) — i.e. the
collapse changes when the executor is entered and nothing else.

### The six-family mode converges too

```
python3 scripts/fpga/f2/prompt05b.py --ids 785 6722 315 9625 374 --layers 0 \
    --proj-cols 8 --include-reconstructed --executor sim --poison 0.5 \
    --work-dir build/p05b_recon_collapse
```

6/6 families, **4 passes / 3 invocations per layer** (the extra pass is the
lossy C-1 re-entry of RMSNorm-2 and SwiGLU walking one more link of the
layer's dependency chain), 35/35 checks, geometry audit PASS with 0 unaudited,
poison still flips the token (`[7407]` → `[279]`, max|dlogit| 15.05).
`build/p05b_recon_collapse.log`.

**The sim wall does not improve, and that is the honest result.** In
simulation an "invocation" is a local process start of a few milliseconds, so
there is no 4.8 s constant to remove; what the collapse costs in sim is
visible instead — 14 superseded attention programs per layer and two extra
host-side replays of the cheap emitters, together **+2.8 s per layer (+2.5%)**.
The saving this change makes is entirely in the term only a card pays.

## 4. PREDICTED hw per-layer wall — A PREDICTION, NOT A MEASUREMENT

Evaluated with `scripts/fpga/f2/fatproof.py --predict` on the same fitted
2-parameter model and the same fit points as the committed prediction
(`build/collapse_prediction.log`,
`docs/results/prompt_on_chip/collapse_hw_prediction.json`):

| shape | invocations / layer | PREDICTED s / layer |
| --- | ---: | ---: |
| thin (pre-fat, the 2026-08-03 flown shape) | 43 | 470 (measured wall 578) |
| fat + burst (today's flown shape) | 37 | 185 (**measured 193.7**) |
| **fat + burst + collapse (this change)** | **5** | **33** |

```
  the PREDICTED 'new' wall, by term:
      invocation setup                              24.0 s  (73%)
      per-program (TILE_RST + parse + marker)        3.7 s  (11%)
      BAR0 peeks                                     5.1 s  (16%)
  C_inv sensitivity: 4.8 s -> 32.8 s | 2.5 -> 21.3 | 1.5 -> 16.3 | 1.0 -> 13.8
```

Read it with these limits, which are the same ones the committed prediction
carries: two parameters fitted to two silicon points; the model predicts the
EXECUTOR term and the flown shape's wall ran 8.8 s/layer above its own
prediction (host-side emit + grade, which this change slightly *increases*);
and at 5 invocations the setup term is only 73% of the predicted wall, so that
unmodelled residual weighs relatively more than it did at 37. A defensible
statement of the expectation is therefore **~33 s of modelled transport plus
the same ~9-12 s of host-side residual — call it ~40-45 s/layer**, against
193.7 s measured today. Only a card session settles it.

### 4b. MEASURED OUTCOME (card session, 2026-08-04 evening) — the prediction
### was WRONG in magnitude, right in direction

The card session ran the same night (`p05b_collapse_hw.log`, committed
beside this file; image `agfi-0ecab46b8a8376b21`, executor hw, collapse ON):

```
per-layer wall : min 131.8 s  median 133.4 s  max 133.4 s   (MEASURED, 2 layers)
invocations    : 5 per layer (10 total for 138 programs)  — exactly as predicted
checks         : 68/68 substitution/consumption PASS; 27,392/27,392 projections
                 bit-exact; geometry audit PASS 0 unaudited
token          : OFFLOAD ON = PURE HOST = ' Paris' (270 s vs 4 s host)
```

**Reconciliation, stated plainly:** the invocation count collapsed 44 → 5 as
predicted, and the wall improved 193.7 → 133.4 s/layer (1.45×) — but the §4
prediction said ~40-45 s/layer, and the measurement is **~3× above it**. The
unmodelled residual dominates at this invocation count: measured executor
time was ~72 s/layer (vs the model's 33 s of transport) with 437.5 MB of
regops uploaded per run, plus emit/audit/grade. The fitted 2-point model is
hereby retired for extrapolation below ~10 invocations/layer; the 133 s/layer
figure is the measured rung the MASTER_TABLE ladder quotes.

## 5. Every honesty gate, and what happened to it

| gate | status |
| --- | --- |
| per-program geometry audit (`INFO_D == 64`) | **UNCHANGED in force, moved to `Runner.prepare`** — called once per DISTINCT program *before* it is keyed, queued, executed or served from the pool. The run reports `distinct programs audited`, `distinct programs executed` and `executed WITHOUT an audit` (must be 0), and `_execute` REFUSES a path `prepare` never saw. |
| disclosed INFO_TIER retarget (exactly one site per program) | unchanged, same hook, count disclosed per run (`444` audit passes over `199` distinct programs in the A/B — re-emitted files are re-audited and re-retargeted, never skipped) |
| zero baked expectations / produce-mode | unchanged (emitters untouched) |
| substitution + consumption checks | unchanged and re-run from the pooled captures on the kept pass — 82/82 |
| grading seams (`grade_codes_scales`, `grade_resid`, `grade_compute_job`, `decode_multiblock`) | unchanged and re-run every pass |
| fat-vs-thin cross-check | unchanged; the thin reference now rides the same flush instead of costing its own invocation — 14/14 EQUAL |
| token identity A/B (3 runs) | unchanged — PASS |
| poison discriminator | unchanged — still flips the token |
| refuse-loudly on missing captures / no hw attach | unchanged (`_execute` raises on any non-`ok` result; the unattached-hw refusal is covered by the selftest) |
| **new** — a program set that never stabilises | `REFUSE` after `--max-passes` (default 12) rather than looping or reporting a partial layer |
| **new** — pool identity | a pooled capture set is reused only for a program whose name AND bytes match; selftested by rewriting a program and proving the stale captures are not reused |

### The one gate that was restructured

`geometry audit == programs run` used to be an equality between the number of
`audit_geometry` calls and `runner.n_jobs`. Under the collapse a program is
audited once and read back many times, and a *re-emitted* program is audited
again — so that equality no longer means what it said. It was replaced by two
statements that are strictly stronger:

1. `_execute` **refuses** any path `prepare` never saw (so an unaudited
   program cannot reach an executor at all — structural, not statistical), and
2. the report prints `unaudited_executed`, recomputed from the audited-key set
   against the executed-key set, and **FAILS the run if it is not 0**.

Both are exercised by `layer_offload.py --selftest`.

### The one place emission is memoised instead of re-verified

Re-emitting the full-width projections on every replay would cost ~28 s of
staging and ~214 MB of regops *per pass per layer* (measured), so
`prompt05b.MultiOffloader05B._gemm` memoises that emission on an
`operand_key` — a digest over the exact operand bytes, their shapes and
dtypes, and every transport/geometry knob `_plan_programs` reads. A memo hit
skips **emission only**: the captures it returns are decoded, spliced and
**re-graded against golden's own accumulators for that pass's operands**, so a
stale or mis-keyed entry surfaces as `bit-exact=False` and fails the run.
`--no-collapse` never uses the memo.

## 6. Regression — the 7B C2 path did not move

```
python3 scripts/fpga/f2/layer_offload.py --prompt "The capital of France is" \
    --layer 0 --offload-step 4 --executor sim --poison 0.5 \
    --work-dir build/c2_collapse
```

All six op types of layer 0 of a real Qwen2.5-7B decode, D=128 twin,
`--proj-cols 8` (`c2_sim.log`, `c2_result.json`):

```
  OP TYPES SERVED BY THE TILE: 6/6 (proj, rope, attn, resid, norm, swiglu)
  token OFFLOAD ON  : ids=[12095] text=' Paris'
  token PURE HOST   : ids=[12095] text=' Paris'
  token HOST+BUS_ON : ids=[12095] text=' Paris'  -> same as pure host
  discriminator     : tile values x0.5 -> ids=[279], max|dlogit|=12.97
  TOKEN IDENTITY: PASS      MILESTONE C2: PASS      checks 63/63
```

Same token, same discriminator id and the same `max|dlogit| 12.97` as the
committed C2 record (`docs/results/prompt_on_chip/C2_PROMPT_ALL_OPS_RESULT.md`).
Transport: **4 passes / 7 invocations for the layer** (14 for the whole
3-run + discriminator session) against **67 per layer** for the per-op shape
at this scope — 7 projection calls, 28 RoPE rows, 28 attention heads, 2
residual windows, 1 RMSNorm-2 and 1 SwiGLU.

One property of the size bound is visible here and is worth stating: the 7B
SwiGLU stage is a **single 77.6 MB program**, larger than `--max-batch-mb`, so
it becomes its own batch. A program bigger than the bound is never split — the
bound caps how many programs are gathered, never what one program contains.
