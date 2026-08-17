# P2 stage 2 — the b64 CL config (ITEM A) and the 0.5B D=64 driver (ITEM B)

Date 2026-08-03 · branch `comp/prompt-b-c` · builds on the GO-WITH-CONDITIONS
gate in `docs/results/p2_05b_gate/RESULTS.md` (conditions §5.1 and §5.2).

**VERDICT.** ITEM A is **DONE**: the CL takes the two missing knobs, the
D=128 default build is proven **byte-identical**, and the intended 0.5B image
builds and replays the gate's real-0.5B attention jobs 0-fail. ITEM B is
**DONE WITH ONE MEASURED LIMIT**: all six op families of one real 0.5B
decoder layer are served by the D=64 tile and every one is **bit-exact
against golden's own view of that op**, but end-to-end token identity holds
for **2 of the 4 prompts tried** — the two C-1-view families (RMSNorm-2,
SwiGLU) re-enter the host through a 64-wide feeder (B-FEED-WIDTH) and on
0.5B that disclosed cost is comparable to the model's top-1 margin. The
four EXACT-egress families are token-identical on the flagship prompt.
Numbers and the consequence for "most of the model" are in §5.

---

## 1. ITEM A — the b64 CL config

### 1.1 What was actually missing

The gate listed `APEX_CL_GQA` and `APEX_CL_DM` as already-plumbed
(`cl_apex.sv:541-548`) and named two knobs that were not:

| apex_top parameter | before | now | default |
|---|---|---|---|
| `CFG_DM` (GAP D model-wide family, `apex_top.sv:112-124`) | never passed → `CFG_DM = CFG_D` | `+define+APEX_CL_DMODEL` | `` `APEX_CL_D `` — i.e. `CFG_DM = CFG_D`, apex_top's own default |
| `QSTAGE_H_MAX` (per-head q staging, `apex_top.sv:138-148`) | never passed → 1 | `+define+APEX_CL_QSTAGE` | 1 |

Both defaults reproduce the shipped elaboration exactly, which is what makes
the byte-identity proof in §1.3 possible. Three elaboration guards were added
next to them (`cl_apex.sv`, the `g_chk_dm` idiom): `QSTAGE ∈ [1, min(STAGE_R_MAX,
seq_walker_pkg::WALK2_H_MAX)]`, `DMODEL ∈ {64,128}`
(`seam_feeder_quant.sv:100-102`), `DMODEL >= APEX_CL_D` (GAP D).

### 1.2 A gate-integrity fix the knobs exposed

The guards fired but **did not fail the build**. `verif/f2sim/Makefile`
passes `-Wno-fatal` (needed for the aws-fpga kit's own WIDTH warnings), which
demotes every elaboration-time `$error` to a warning, and the recipe's
`| tail -3` then hid the text. An illegal `+define` produced a green build
and a silently mis-elaborated twin — including for apex_top's own
pre-existing `g_chk_dm` / `asu_rmsnorm` guards. Fixed by adding
`-Werror-USERERROR` to the build line, which promotes `$error` back to a hard
error and leaves the kit's warnings alone.

Measured, after the fix (`rc`, and whether a binary was produced):

```
NEG1  +define+APEX_CL_QSTAGE=99   (> STAGE_R_MAX=31)     rc=2  binary=ABSENT
NEG2  +define+APEX_CL_DMODEL=256  (feeder-illegal)       rc=2  binary=ABSENT
NEG3  D=128 +define+APEX_CL_DMODEL=64  (CFG_DM < CFG_D)  rc=2  binary=ABSENT
```

Before the fix all three produced `rc=0` and a working `f2sim`.

### 1.3 The D=128 default build is BYTE-IDENTICAL

Verilator's generated C++ *is* the elaborated design. Concatenated content
hash of all 15 generated `.cpp`/`.h` files, `make build D=128 DDR=0`:

```
pre-edit  (pristine cl_apex.sv, pristine Makefile)   2158138cfd25134c5b98083688779996572082983fd2d82ea1b815ad2da05c48
post-edit (both knobs + guards)                      2158138cfd25134c5b98083688779996572082983fd2d82ea1b815ad2da05c48
post-edit (+ the -Werror-USERERROR Makefile change)  2158138cfd25134c5b98083688779996572082983fd2d82ea1b815ad2da05c48
```

Reproduce:

```
cd verif/f2sim && make build D=128 DDR=0 OBJ=obj_x
(cd obj_x && find . -type f \( -name '*.cpp' -o -name '*.h' \) | sort | xargs cat | shasum -a 256)
```

(The `.d` dependency files and the `.a` archive differ between two `OBJ=`
directories because they embed that directory's path; no generated source
does.)

### 1.4 The intended 0.5B image

```
cd verif/f2sim && make build D=64 DDR=0 OBJ=obj_b64_05b \
  VFLAGS_EXTRA="+define+APEX_CL_DM=896 +define+APEX_CL_GQA=2 \
                +define+APEX_CL_QSTAGE=14 +define+APEX_CL_DMODEL=64"
```

→ `CFG_D=64, CFG_DM=64, KVQ_GQA_NENG=2, RMS_D_MAX=LAYER_DM_MAX=896,
QSTAGE_H_MAX=14, DDR_PRESENT=0`.

`CFG_DM` is deliberately left at `CFG_D=64`. A split `CFG_D=64/CFG_DM=128`
build passes `g_chk_dm` but breaks the routes that push **per-head** rows
through the same feeder — the RoPE q_sink egress (`apex_top.sv:1186`) and the
KVQ-readback requant route — which `apex_top.sv:1858-1864` records as "NOT
drivable at CFG_DM != CFG_D". So the 0.5B model row is framed **14 × 64**,
not 7 × 128; see §3.2 for what that costs the R4 norm arm.

### 1.5 One disclosed retarget on this image

`INFO_TIER` (BAR0 `0x1014`) is `{(KVQ_OUTLIER_K>0)&&mask_valid,
(KVQ_GQA_NENG==1), 1'b1}` — `apex_top.sv:2540-2543`. With `KVQ_GQA_NENG=2`
the build is CQ-8-only and truthfully reads **0x1**, where the plain b64 CL
reads 0x3 (the p2 gate's retarget) and the L3 d64 *reference* choreography
bakes 0x7 (`gen_l3_vectors.py:349`). Running the gate's 7 jobs unretargeted
on this image fails **exactly those 7 identity reads and nothing else** —
the checks demonstrably bite. `tile_geom.retarget_info_tier` rewrites that
one expectation per program and **refuses** if a program does not carry
exactly one.

### 1.6 ITEM A regression

| gate | result |
|---|---|
| `make capgate` (D=128 twin) | `CAPGATE: PASS (job=job_s019_L19_h03 caps=505 values_matched=505/505 tile_div=5 executor=sim:obj_d128_ddr0)` |
| `make run D=128 DDR=0` (canonical 18-job S8 7B set) | `F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS` |
| `make -C verif/seq_walker all` | `==== B1 WALKER SUITE: ALL TARGETS COMPLETE ====` rc=0 (incl. mutation gates 1-4) |
| gate's 0.5B 7 jobs on the **plain** b64 twin | `F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS` |
| gate's 0.5B 7 jobs on the **new 0.5B image** (retargeted 0x7→0x1) | `F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS` |
| `gen_layer_ops.py --selftest` | `GENLAYEROPS SELFTEST: PASS (fails=0)` |
| `narrow_flight.py selftest` | `NARROW_FLIGHT SELFTEST: PASS (fails=0)` |
| `layer_offload.py --selftest` | `LAYER_OFFLOAD SELFTEST: ALL PASS` |

---

## 2. ITEM B — the 0.5B host driver

Two new files; nothing owned was edited.

* `scripts/fpga/f2/tile_geom.py` — run the frozen emitters at a geometry
  other than 128 (`at_d`), prove it reached the wire (`audit_geometry`), the
  disclosed `retarget_info_tier`, and the ONE re-derivation (§3.2).
* `scripts/fpga/f2/layer05b.py` — the driver. It **subclasses**
  `layer_offload.LayerOffloader` / `.Runner`: the seam, the six op wrappers,
  the substitution/consumption checks, the 3-way A/B and the poison
  discriminator are inherited verbatim, because they are geometry-neutral
  once `layer_offload.D_TILE` is reframed.

Selftests: `tile_geom.py` → `TILE_GEOM SELFTEST: ALL PASS` (17 checks);
`layer05b.py --selftest` → `LAYER05B SELFTEST: ALL PASS` (18 checks, on a
tile-legal tiny model carrying the 0.5B shape invariants: head_dim 64, GQA,
≥2 μ-chunks, 64-column SwiGLU chunks).

---

## 3. What did NOT generalize from D=128

### 3.1 Geometry — rebound, then verified

`at_d(d)` rebinds `gen_layer_ops` / `gemm_job` module geometry
(`D_TILE`, `BPR`, `ROWS_PER_DESC`, `INJ_K128_LANES/MAX`) — the same
"rebind a module global, never edit the arbiter" technique `layer_offload`
uses on golden — and restores on every exit path. It is not trusted:
`audit_geometry` reads each emitted program's own phase-A `INFO_D`
expectation, which every emitter bakes as `Script(D_TILE).D`, and refuses
unless it is exactly `d`. **87/87 programs of the flagship run audited
INFO_D == 64.**

### 3.2 The R4 norm arm's `k` — the one real re-derivation

`gen_layer_ops.build_norm2_chunked` uses `k = dm // D_TILE` for **two
different quantities**, which coincide only at 128:

* the **streamed** chunk count = `dm / CFG_DM` = **14** (the C-1 feeder frame)
* the **μ-table index** `ext_k` = `dm / 128` = **7** — frozen at 128 by
  `asu_wide_rms_params.svh` (`WIDE_RMS_CHUNK = 128`), with
  `asu_rmsnorm.sv:300-302` elaboration-erroring if a regenerated table ever
  changes it. The arm computes `mean2 = (ext_sum2·MU[ext_k]) >> (S[ext_k]+16)`.

Arming 14 would normalize a 896-wide row by 1792. `tile_geom.build_norm2_chunked_d`
streams 14 × 64 and arms `ext_k = 7`. **Red/green, both on the 0.5B twin,
same operands, same golden row:**

```
re-derived (ext_k=7) : 896/896 codes + 14/14 row scales BIT-EXACT vs golden;
                       14 chunk-sum exports exact, total exact
frozen     (k=14)    : bit-exact = False, 68 of 896 codes differ,
                       row scales NOT equal
```

The all-(−128) row sits **exactly** on the arm's `k·2^21` bound at k=7
(896·2^14 = 7·2^21 = 14,680,064), so the emission-side mirror of that refusal
is tight, not loose.

### 3.3 Everything else generalized

Residual 896 ≤ `RESID_WIN` 1024 → **ONE** window job per residual (7B needed
4); SwiGLU 4864 = 76 × 64 → exactly one chunk per C-1 row; projections
K=896 staged on 64-wide rows (15 rows, `K_total`=960 ≤ `K_MAX`); attention
`head_dim=64` needs no change at all.

---

## 4. The flagship run (executed, verbatim)

```
python3 scripts/fpga/f2/layer05b.py --prompt "The capital of France is" \
    --layer 0 --executor sim --poison 0.5 --norm-k-red \
    --work-dir build/layer05b_full
```

```
  prompt          : 'The capital of France is'   ids [785, 6722, 315, 9625, 374]
  model           : mlx-community/Qwen2.5-0.5B-Instruct-4bit
                    L=24 H=14 H_kv=2 head_dim=64 D_model=896 d_ffn=4864
  image           : b64_05b  CFG_D=64 CFG_DM=64 KVQ_GQA_NENG=2
                    RMS/LAYER_DM_MAX=896 QSTAGE_H_MAX=14
  offloaded layer : 0  step 4  composition C-LBUS BUS_ON
  tile jobs       : 87 programs, 23390 capture records, 12.8s in the executor
  geometry audit  : 87/87 programs — each carries INFO_D == 64;
                    descriptor k values seen = [0, 2, 5, 64, 896, 960, 1024]  -> PASS
  disclosed retarget: 87 INFO_TIER expectations rewritten 0x7 -> 0x1,
                    exactly one per program. No datapath expectation touched.

  op family                  who         served/total     exact?   tile vs golden
  projections q/k/v/o/g/u/d  TILE (sim)  56/13696         BIT-EXACT      bit-exact
  RoPE (decode-token q)      TILE (sim)  896/896          reconstructed* bit-exact
  attention (score + PV)     TILE (sim)  896/896          BIT-EXACT      bit-exact
  residual (r1, r2)          TILE (sim)  1792/1792        BIT-EXACT      bit-exact
  RMSNorm-2                  TILE (sim)  896/896          reconstructed  bit-exact
  SwiGLU                     TILE (sim)  4864/4864        reconstructed  bit-exact

    RoPE    max_abs_delta 0.1401 | requant_identical True |
            downstream_codes_match_golden True   (exact WHERE CONSUMED)
    resid   ONE 896 window, 1/1 slice pass, reassembled == golden
    norm2   14 x 64 frames | max_abs_delta_q78 12 | downstream C-1 codes
            differing 35/896 | row scale equal
    swiglu  76 x 64 frames | max_abs_delta 0.01886 | downstream C-1 codes
            differing 68/4864 | row scale equal

  38 substitution / consumption checks: ALL PASS
  RED ARM (frozen emitter's k=14): bit-exact False, 68/896 codes differ
           -> the re-derivation is LOAD-BEARING (PROVEN)

  logit geometry    : pure-host top-1 margin 0.5966 over a 29.48 logit range
                      (2.02%); the tile-served layer moved logits by
                      max|dlogit| = 4.661
  token OFFLOAD ON  : ids=[7407] text=' located'
  token PURE HOST   : ids=[12095] text=' Paris'
  token HOST+BUS_ON : ids=[12095] text=' Paris'  -> bus mode alone changes nothing
  discriminator     : tile values x0.5 -> ids=[279], max|dlogit|=15.05
                      (substitutions ARE load-bearing)
  OP FAMILIES SERVED BY THE TILE: 6/6
  TOKEN IDENTITY: FAIL
```

### 4.1 The four EXACT-egress families ARE token-identical

Same prompt, same image, `--ops proj,rope,attn,resid`:

```
  tile jobs       : 82 programs, 9186 capture records, 5.2s in the executor
  token OFFLOAD ON  : ids=[12095] text=' Paris'
  token PURE HOST   : ids=[12095] text=' Paris'
  discriminator     : tile values x0.5 -> ids=[3881], max|dlogit|=8.543
  OP FAMILIES SERVED BY THE TILE: 4/6      TOKEN IDENTITY: PASS
```

---

## 5. The measured limit, and what it means for "most of the model"

Every family's tile output is bit-exact against golden's own view of that op,
so no tile value is wrong. What moves the token is the **C-1 re-entry** of
RMSNorm-2 and SwiGLU: golden composes a whole-row C-1 (ONE amax over 896),
the tile can only hand back independently-scaled 64-wide frames — the
`B-FEED-WIDTH` blocker (`gen_layer_ops.BLOCKERS`;
`seam_feeder_quant.sv:67/100`, `apex_top.sv:1866`).

**All four prompts tried, none omitted** (6/6 families each):

| prompt | host top-1 margin | % of logit range | max\|dlogit\| | token identity |
|---|---|---|---|---|
| `The capital of France is` | 0.5966 | 2.02% | 4.661 | **FAIL** (' Paris' → ' located') |
| `The capital of Japan is` | 0.8051 | 2.64% | 5.52 | **PASS** (' located') |
| `2 + 2 =` | 2.583 | 8.19% | 3.376 | **PASS** (' ') |
| `The color of the sky is` | 0.0615 | 0.18% | 4.465 | **FAIL** (' yellow' → ' changing') |

Isolation on the flagship prompt: RMSNorm-2 alone flips it, and SwiGLU alone
flips it; the four exact-egress families together do not.

Consequences to carry forward:

1. **7B ≠ 0.5B here.** The same two substitutions did not flip 7B's token
   (`C2_PROMPT_ALL_OPS_RESULT.md`). 0.5B's top-1 margins are frequently
   ≤ 3% of the logit range, which is inside the disclosed C-1 reconstruction
   cost. This is a property of the model, not a new defect — but it means
   **a multi-layer 0.5B offload that includes norm/SwiGLU cannot be expected
   to preserve tokens**, and any claim that says otherwise must be measured
   per prompt, not inherited from the 7B result.
2. The fix is RTL, not host: the wide C-1 feeder (`IB_LAYER.md` stage 6,
   named at `apex_top.sv:20-22`). Until it lands, the token-safe offload set
   at 0.5B is **projections + RoPE + attention + residual**.

---

## 6. What remains before "most of the model"

1. **Multi-layer driver.** This is ONE layer at ONE decode step. The driver
   already parameterizes the target `(layer, step)`, but a many-layer run
   needs (a) per-layer weight staging the wrappers currently take from the
   in-memory `LayerWeights`, and (b) a decision on §5: either fence the
   offload to the four token-safe families, or accept measured token drift
   and grade on logits instead of tokens.
2. **Full-width projections.** 56 of 13,696 accumulators were tile-served
   here (`--proj-cols 8`). The 7 projection calls of this layer at this step
   (K/V carry T+1 = 5 rows, the rest 1) are 13,696 accumulators = **1,712**
   8-column MXE jobs. At this run's measured projection rate — 11 programs in
   1.52 s = **0.138 s/job** — full-width projections for ONE layer at ONE
   step are **~240 s**. That is cheap; the cost that actually scales is
   layers × steps, and a 24-layer prefill+decode of this prompt would be
   ~24 × (240 s + the ~11 s the other five families took) ≈ **1.7 h per
   decode step** in simulation. Sim is the bottleneck, not the tile.
3. **The image build.** Everything above is the **verilated twin**. No DCP,
   no AFI, no card. The b64 0.5B image has never been through Vivado; the
   GQA-4 route needed an m6a.4xlarge (see the F2 devbox notes) and GQA-2 at
   D=64 is untried. Nothing here claims silicon.
4. **A card session** to replay the same regops with `--executor hw`. Not
   run: no AWS instance was started for this work.

## 7. Files

* `scripts/fpga/f2/cl_apex/design/cl_apex.sv` — the two knobs + three guards.
* `verif/f2sim/Makefile` — `-Werror-USERERROR`.
* `scripts/fpga/f2/tile_geom.py`, `scripts/fpga/f2/layer05b.py` — ITEM B.
* Run records (gitignored `build/`): `layer05b_full/layer05b_result.json`,
  `layer05b_exact/`, `layer05b_p{1,2,3}/`, and the `.log` beside each.
