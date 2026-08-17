# P2 stage 3 — MANY Qwen2.5-0.5B decoder layers on the D=64 tile

Date 2026-08-03 · branch `comp/prompt-b-c` · builds on
`docs/results/p2_b64_cl/RESULTS.md` (stage 2, ONE layer) and
`docs/results/p2_05b_gate/RESULTS.md` (the GO-WITH-CONDITIONS gate).

**VERDICT — DONE.** `scripts/fpga/f2/prompt05b.py` serves the four
EXACT-EGRESS op families (projections, RoPE, attention, residual) of **12 of
the 24 decoder layers** of a real prompt from the D=64 tile at **full
projection width** — all 13,696 accumulators of all 12 layers, 164,352 of
164,352 — and the model emits its own token, `' Paris'`, identical to the
pure-host run. Measured tile share of that decode step's multiply-accumulate
arithmetic: **50.00 %** (36.81 % counting the lm_head).

The sharpest number: **`max|dlogit| offload vs HOST+BUS_ON = 0`** — with the
tile computing 12 layers' worth of arithmetic, the logits are bit-identical
to the host computing the same layers itself. Same result on all four prompts
tried.

Executed in the verilated `b64_05b` twin. Nothing here is silicon; no AWS
instance was started.

---

## 1. What is new (and what is inherited)

`layer05b.py` (stage 2) is the D=64 driver for ONE layer. `prompt05b.py` is a
new file that subclasses it. The seam, the six op wrappers, the
substitution/consumption checks, the 3-way A/B, the poison discriminator, the
geometry rebind, the per-program `INFO_D` audit, the disclosed `INFO_TIER`
retarget and the re-derived RMSNorm arm are all INHERITED, unmodified. Four
things are added:

| # | Added here | Proof it works |
|---|---|---|
| 1 | **Many layers.** `_layer` re-points the inherited single-layer target at every call, so any layer subset arms at one decode step. Per-layer weight staging is automatic (golden hands each layer its own `LayerWeights`) and CHECKED: `_LayerState.next_proj` refuses a staged operand that is not the tensor it claims. | 12/12 layers armed; **492/492** substitution/consumption checks pass |
| 2 | **Full-width projections, batched.** `--proj-cols -1` (default) = every accumulator. `_gemm` emits all K-split jobs of a projection call, then runs them through `batch_exec`, one invocation per `--batch-size`. | `--cross-check` re-runs one block per projection tensor per layer through the inherited UNBATCHED `_tile_matvec`: **84/84 EQUAL** |
| 3 | **A measured FLOP share.** `MacCensus` wraps golden's own `gemm_i8_ksplit` / `attention_core` / `decoder_layer_fx` during the pure-host arm and counts every MAC of the step, per layer, per tensor, plus the lm_head. Read-only. | `verify_against` re-derives the same denominator from the model's OWN weight tensors and REFUSES on any mismatch |
| 4 | **The owner's family policy.** Default = the four exact-egress families, token identity a PASS condition. RMSNorm-2 / SwiGLU only behind `--include-reconstructed`, which warns plainly and grades at the logit level. | §5 |

### 1.1 A real bug the census cross-check caught

`compute.gemm_i8_ksplit` splits `N > DIM_MAX` (4095) by **concatenation
through its own module global** (`compute.py:64-67`) — the very name the
census rebinds. The first implementation therefore counted a 0.5B gate/up
projection three times (4864 + 4095 + 769) and reported a step total of
589,252,608 MACs instead of 380,061,696, i.e. a share **35 % too small**. Fix:
a depth guard, count the outermost call only. The check that would have caught
it immediately — re-deriving the denominator from the model's own weight
shapes — now runs every time and refuses rather than warns, and the selftest
carries the trap directly (a `K > K_MAX`, `N > DIM_MAX` GEMM must be counted
once, at `M·N·K`).

---

## 2. Measured cost per layer

Verilated twin (`verif/f2sim/obj_b64_05b/f2sim`), this machine, `tile_div=5`.

| quantity | measured |
|---|---|
| tile programs for ONE full-width layer-step | **2,201** (2,160 projection K-split jobs + 14 RoPE + 14 attention + 2 residual + 11 cross-check) |
| capture records for ONE full-width layer-step | 24,033 |
| executor time per layer (12-layer run) | 199.9 – 236.2 s |
| **wall per layer (emit + audit + retarget + execute + grade)** | **min 303.2 s · median 337.3 s · max 366.2 s** over 12 layers |
| wall per layer, single-layer run, uncontended | 355.8 s |
| per-job breakdown (32-job probe) | emit 22.4 ms + audit&retarget 28.2 ms + execute ~93 ms ≈ 143 ms |
| batched vs one-invocation-per-job, sim, real D=64 jobs | 1.03–1.07× at N=16/64/128, captures equal |
| job bytes emitted and pruned during the 12-layer run | 5.86 GB |

**Batching barely helps in simulation**, and that is expected: the verilated
twin is compute-bound, so the per-invocation constant it removes is small. It
is kept because it is the code path silicon needs, where the measured
constant is ~3.6 s/job separate vs ~0.2 s/job batched (`proj_sweep_batched.py`
header, `attrib_proof.json`) — an ~18× difference. The sim speedup is reported
as measured, never as the hw number.

At ~337 s/layer a **≤ 2 h budget buys 12–14 layers**; 12 were run end to end
in **4,043 s (67 min)** for the offload arm, ~68 min for the whole 3-way A/B
plus discriminator.

---

## 3. THE RUN — 12 layers, full width, executed

```
python3 scripts/fpga/f2/prompt05b.py --prompt "The capital of France is" \
    --layers 0-11 --executor sim --cross-check --poison 0.5 \
    --work-dir build/p05b_main
```

```
  prompt          : 'The capital of France is'   ids [785, 6722, 315, 9625, 374]
  model           : mlx-community/Qwen2.5-0.5B-Instruct-4bit
                    L=24 H=14 H_kv=2 head_dim=64 D_model=896 d_ffn=4864
  image           : b64_05b  CFG_D=64 CFG_DM=64 KVQ_GQA_NENG=2
                    RMS/LAYER_DM_MAX=896 QSTAGE_H_MAX=14
  offloaded layers: [0..11] (12 of 24)  decode step 4  composition C-LBUS BUS_ON
  op families     : proj,rope,attn,resid  [FOUR EXACT-EGRESS FAMILIES]
  tile jobs       : 26,510 programs, 297,726 capture records, 2,615.3 s in the executor
  geometry audit  : 26,510/26,510 programs — each carries INFO_D == 64;
                    descriptor k values seen = [0, 2, 5, 64, 896, 960, 1024]  -> PASS
  disclosed retarget: 26,510 INFO_TIER expectations rewritten 0x7 -> 0x1,
                    exactly one per program. No datapath expectation touched.
  batching xcheck : 84/84 batched acc == UNBATCHED _tile_matvec acc -> EQUAL

  layer    projections        RoPE     attention      residual   jobs    caps  exec s  checks
      0    13696/13696    896/896*       896/896     1792/1792   2201   24033   199.9   41/41
      1    13696/13696    896/896*       896/896     1792/1792   2201   24033   205.7   41/41
      2    13696/13696    896/896*       896/896     1792/1792   2201   24033   236.2   41/41
      3    13696/13696    896/896*       896/896     1792/1792   2201   24033   221.8   41/41
      4    13696/13696    896/896*       896/896     1792/1792   2201   24033   225.3   41/41
      5    13696/13696    896/896*       896/896     1792/1792   2201   24033   218.7   41/41
      6    13696/13696    896/896*       896/896     1792/1792   2201   24033   213.6   41/41
      7    13696/13696    896/896*       896/896     1792/1792   2201   24033   220.7   41/41
      8    13696/13696    896/896*       896/896     1792/1792   2201   24033   229.7   41/41
      9    13696/13696    896/896*       896/896     1792/1792   2201   24033   219.2   41/41
     10    13696/13696    896/896*       896/896     1792/1792   2201   24033   211.6   41/41
     11    13696/13696    896/896*       896/896     1792/1792   2201   24033   206.5   41/41
  TOTAL   164352/164352 10752/10752   10752/10752  21504/21504  26412  288396  2608.8 492/492
  per-layer wall  : min 303.2s  median 337.3s  max 366.2s  (MEASURED, 12 layers)

  projections  TILE  164352/164352  BIT-EXACT   84/84 calls, 25920 programs
  RoPE         TILE   10752/10752   reconstructed*  requant_identical True,
                                    downstream_codes_match_golden True
  attention    TILE   10752/10752   BIT-EXACT   168/168 heads
  residual     TILE   21504/21504   BIT-EXACT   ONE 896 window per residual
  RMSNorm-2 / SwiGLU  HOST — not in --ops (see §5)

  492 substitution / consumption checks: ALL PASS

  logit geometry : pure-host top-1 margin 0.5966 over a 29.48 range (2.02 %)
      max|dlogit| offload   vs PURE host  = 3.472   (tile + composition, mixed)
      max|dlogit| HOST+BUS_ON vs PURE host = 3.472  (the composition ALONE)
      max|dlogit| offload   vs HOST+BUS_ON = 0      (THE TILE ALONE)
  token OFFLOAD ON  : ids=[12095] ' Paris'
  token PURE HOST   : ids=[12095] ' Paris'
  token HOST+BUS_ON : ids=[12095] ' Paris'   -> bus mode alone changes nothing
  discriminator (layer 0, proj_cols 8):
      control  ids=[12095]   poison x0.5 ids=[3881], max|dlogit| vs control 8.543
      -> substitutions ARE load-bearing
  TILE SHARE OF THE STEP    : 50.00 % of multiply-accumulates
  TOKEN == HOST IN SAME BUS : PASS       TOKEN == PURE HOST : PASS
  P2 STAGE 3                : PASS
```

`*` RoPE's egress is a C-1 view; it is measured exact WHERE THE MODEL CONSUMES
IT — golden's very next stage re-quantizes the row with the same
`quant_rows_i8`, and both the idempotence and the match against golden's own
codes are checked per head, per layer (168/168).

---

## 4. Token identity A/B — four prompts, 12 layers, four exact families

`--layers 0-11 --proj-cols 8` (projections sampled so four prompts stay
affordable; RoPE, attention and residual at full width). 408/408 substitution
checks pass on every prompt.

| prompt | host top-1 margin | PURE HOST | OFFLOAD ON | HOST+BUS_ON | offload vs pure host | **offload vs HOST+BUS_ON (tile alone)** | token == pure host |
|---|---|---|---|---|---|---|---|
| `The capital of France is` | 0.5966 (2.02 %) | ` Paris` | ` Paris` | ` Paris` | 3.472 | **0** | **PASS** |
| `The capital of Japan is` | 0.8051 (2.64 %) | ` located` | ` located` | ` located` | 4.933 | **0** | **PASS** |
| `2 + 2 =` | 2.583 (8.19 %) | ` ` | ` ` | ` ` | 3.436 | **0** | **PASS** |
| `The color of the sky is` | 0.06145 (0.18 %) | ` yellow` | ` changing` | ` changing` | 4.953 | **0** | see §4.1 |

Requirement met: token identity vs pure golden holds on **3 of 4 prompts**,
including `The capital of France is` — one of the two prompts stage 2 saw flip
once RMSNorm-2/SwiGLU were offloaded. The flagship prompt also holds at FULL
projection width over 12 layers (§3). Every prompt's poison discriminator
bites (control vs poison max|dlogit| 7.00–8.54).

### 4.1 The `sky` prompt: the composition mode, not the tile

For all four prompts the **tile-vs-same-composition logit delta is exactly 0**.
What differs on `sky` is the composition: every offloaded layer must run
C-LBUS `BUS_ON`, because `apex_layer_deq.sv:90-92` refuses a job composite
whose fp32 mantissa carries bits below fp16 grade — so the residual op is
unreachable in golden's default `BUS_OFF`. Run 3 of the A/B measures exactly
that with **no tile in the loop**, and on `sky` it produces the same
` changing` the offload run does, with the same 4.953 logit delta. That
prompt's top-1 margin is 0.06145 — 0.18 % of the logit range — thinner than a
composition change the arbiter itself blesses (`transformer.BUS_ON`, D-030,
the mode `capture_layer_step` / `gen_l4_vectors` / `gen_layer_trace` already
use).

The driver therefore grades on TWO identities and always prints both:

* `TOKEN == HOST IN SAME BUS` — the tile's own question. **Required to pass**
  in the four-family mode. 4/4 prompts PASS, delta 0.
* `TOKEN == PURE HOST` — always reported; required unless run 3 shows the
  composition mode alone produced the same token, in which case the report
  says so explicitly and records `bus_mode_explains_difference`.

---

## 5. `--include-reconstructed` — the six-family mode, and its warning

```
python3 scripts/fpga/f2/prompt05b.py --prompt "The capital of France is" \
    --layers 0-3 --proj-cols 8 --include-reconstructed --executor sim \
    --poison 0.5 --work-dir build/p05b_recon
```

6/6 families over 4 layers, 140/140 checks; RMSNorm-2 896/896 and SwiGLU
4864/4864 values per layer, each bit-exact against golden's own view of that
op. The ledger prints the disclosed cost verbatim (`max_abs_delta_q78 = 12`,
35/896 downstream C-1 codes differ for the norm; `max_abs_delta 0.0189`,
68/4864 codes for SwiGLU), and the report states — before any number — that
the token may legitimately differ and that the run is graded at the LOGIT
level. Here the tile-alone logit delta is **3.283**, nonzero exactly as the
B-FEED-WIDTH disclosure predicts, against **0** in the four-family mode.

Consistent with stage 2 §5: at 0.5B the token-safe offload set is
projections + RoPE + attention + residual until the wide C-1 feeder lands
(`IB_LAYER.md` stage 6). Nothing in this stage changes that.

---

## 6. The FLOP share, and the 24-layer extrapolation

Denominator, measured by the census during the pure-host arm and re-derived
from the model's own tensors (`verify_against`, which refuses on mismatch):

```
Wq  1x896x896  x 24 layers =  19,267,584      Wg  1x896x4864 x 24 = 104,595,456
Wk  5x896x128  x 24 layers =  13,762,560      Wu  1x896x4864 x 24 = 104,595,456
Wv  5x896x128  x 24 layers =  13,762,560      Wd  1x4864x896 x 24 = 104,595,456
Wo  1x896x896  x 24 layers =  19,267,584      attention (all heads) =  215,040
                                     step total = 380,061,696 MACs
                              lm_head 151936x896 = 136,134,656 MACs
                                  step + lm_head = 516,196,352 MACs
```

One full-width layer = 15,826,944 projection MACs + 8,960 attention MACs =
**4.1667 % of the step**. Projections are 99.94 % of a layer's arithmetic,
which is why sampling them is the only knob that matters.

| configuration | tile MACs | share of the step | share of step + lm_head | status |
|---|---|---|---|---|
| 1 layer, full width | 15,835,904 | 4.17 % | 3.07 % | MEASURED |
| **12 layers, full width** | **190,030,848** | **50.00 %** | **36.81 %** | **MEASURED — §3** |
| 12 layers, `--proj-cols 8` | 1,778,688 | 0.47 % | 0.34 % | MEASURED — §4 |
| 24 layers, full width | 380,061,696 | 100 % | 73.63 % | **EXTRAPOLATION — NOT RUN** |

**The 24-layer line is an extrapolation and is labelled as such everywhere,
including in the driver's own ledger.** Its cost at this run's measured
median of 337.3 s/layer is 24 × 337.3 s ≈ **2.25 h** of simulation for ONE
decode step, which is why 12 was the scope that fit the budget.

Excluded from both sides of the ratio: elementwise work (RMSNorm, RoPE, SiLU,
residual, softmax) is not multiply-accumulate arithmetic. The census measures
150,528 RMSNorm element-touches in the step, 0.04 % the size of the MAC count,
so the exclusion cannot move the share meaningfully either way. Excluded from
the numerator by construction: anything the tile did not do — the ledger uses
`n_served`, never `n_total`, and a sampled scope prints
`[SAMPLED — see --proj-cols]` next to its own number.

---

## 7. Hardware structure (NOT run)

`--executor hw` routes through `remote_hw_exec.attach(bridge, args)`
(`--hw-host` / `--hw-key`, else `$APEX_F2_HOST` / `$APEX_F2_KEY`) and
**REFUSES loudly** when no remote is configured: without the shim the bridge
drives BAR0 from the local machine, every batch returns no cap file, and the
run looks like a clean zero-capture pass — the trap that cost two hw attempts
on 2026-07-31 (`docs/results/prompt_on_chip/NARROW_LANE_RESULT.md`). The
refusal is a selftest check, not a comment. No AWS instance was started; no
DCP, no AFI, no card.

---

## 8. Gate regression (all green, on the final tree)

| gate | result |
|---|---|
| `prompt05b.py --selftest` (new) | ALL PASS — 36 checks |
| `layer05b.py --selftest` | ALL PASS |
| `tile_geom.py` selftest | ALL PASS |
| `layer_offload.py --selftest` | ALL PASS |
| `gen_layer_ops.py --selftest` | PASS (fails=0) |
| `narrow_flight.py selftest` | PASS (fails=0) |
| `apex_repl.py --selftest` | ALL PASS |
| `run_tinynpu.py --selftest` | ALL PASS |
| `--verify-trace docs/results/s8_7b_token/artifact_trace` | 18/18 bit-exact |
| `--verify-trace docs/results/p2_05b_gate/trace` | 7/7 bit-exact |

Nothing owned elsewhere was edited. The only source file this stage adds is
`scripts/fpga/f2/prompt05b.py`.

---

## 9. What remains for the card session

1. **The 24-layer full-width run.** ~2.25 h of simulation for one decode step;
   `--layers all` takes it today. Nothing blocks it but time.
2. **A Vivado image.** Everything above is the verilated twin. The `b64_05b`
   image (`CFG_D=CFG_DM=64`, `KVQ_GQA_NENG=2`, `APEX_CL_DM=896`,
   `QSTAGE_H_MAX=14`) has never been through Vivado; GQA-2 at D=64 is untried
   and the GQA-4 route needed an m6a.4xlarge (F2 devbox notes).
3. **The hw replay.** `--executor hw --batch-size N` with the remote set.
   Batching is load-bearing there: 2,201 jobs per layer at ~3.6 s separate is
   ~2.2 h/layer, at ~0.2 s batched ~7 min/layer. The per-invocation clock gate
   applies to every batch, and `--cross-check` should stay on for the first
   layer.
4. **The wide C-1 feeder** (`IB_LAYER.md` stage 6) is still what stands
   between the four token-safe families and all six at 0.5B.
5. Multi-STEP offload (this is one decode step of the prefill) and more than
   one generated token are untouched; the `T ≤ 128` fence is unaffected.

## 10. Files

* `scripts/fpga/f2/prompt05b.py` — the driver (new).
* Run records (gitignored `build/`): `p05b_main/prompt05b_result.json` (§3),
  `p05b_ab_{france,japan,math,sky}/` (§4), `p05b_recon/` (§5), `p05b_l1/`
  (the uncontended 1-layer timing), and the `.log` beside each.
