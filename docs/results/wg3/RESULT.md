# W-G3 — partial-layer walk + fuel-fed projection: DECLARATION

Branch `comp/ib-wg3x`, worktree `../apex-wg3x`, **merged with
`comp/level-c-integration` @ `bafa8f3` (combine-11: the D-029/F1 constant
split) before the gate run** — every number below is from the merged tree.
Local pinned Verilator 5.044. No AWS, no hardware: everything here is
simulation or pure Python, and none of it is a hardware claim.

**Status**

| half | state |
|---|---|
| WALKED — unit level (walker RTL + golden trace arbiter) | **GREEN.** The FULL real-7B layer step: 32,172 emissions, 0 errors, 16,532 consumer-legal descriptors. The FENCED subset: 16,139 / 0. |
| WALKED — tile level, SINGLE-HEAD (§4.4) | **GREEN.** Real Qwen2.5-7B head 3 at 7B per-head geometry, walked through the real WALK CSR window: host 5,541 / 0, walked 5,544 / 0. |
| WALKED — tile level, MULTI-HEAD (H=28, H_kv=4) | **BLOCKED by F6(ii)** — the single-arm `hd_ready` sticky. F6(i) (the shared composite cache) was localised here by A/B (§4.3) and is **FIXED** in combine-13 (`apex_wcomp_bank`). |
| K-ROPE — GQA-4, host mode (§4.5) | **GREEN.** 80 real K rows across all 4 GQA engines, 10,262 / 0, plus an F5 discrimination arm that goes RED as predicted. |
| FUEL — tile leg, `xw` → MXE (§5.3) | **GREEN.** Real Qwen `Wq` on the external `xw` port, exact INT32 result. |
| FUEL — **DDR leg** (§5.4) | **GREEN.** Weights arrive loader → `sh_ddr` → `apex_fuel_reader` → afifo → `xw` → tile: 32 / 0, `FUELAUDIT err=00000000`, with a perturbed-DDR arm RED as required. |

**Scope label for any quotation of this work:** *single-head partial-layer walk
at real 7B per-head geometry + GQA-4 K-rope store coverage + a fuel-fed
projection GEMM.* The multi-head walk is NOT claimed, and nothing here is a
hardware claim.

**What "fuel-fed" means here, precisely.** One projection GEMM
(`m=1, k=128, n=8`, `OP_GEMM_WS`) over real Qwen2.5-7B `Wq` weights, whose
weight beats were written into behavioral DDR over PCIS and then read back out
by `apex_fuel_reader` into the tile's `xw` port — with every mailbox weight
push REMOVED, so there is no other source they could have come from. It covers
**128 of the 3584** input channels of that projection: one legal descriptor's
worth. It is NOT a whole projection, NOT a walked projection (F1's fence), and
NOT silicon.

---

## 0. Scope statement — read this before quoting any number

The deliverable was specified as **"partial-layer walk + fuel-fed
projection"**, NOT a walked full layer. That framing still holds, and this
document narrows it further with measurements.

**What is NOT walked, and why:**

- **QKV / o-proj / FFN projections** — originally `F1` (D-029 erratum), now
  **FIXED** in combine-11 and measured green here (§3). They remain outside
  the TILE-level half for a different reason: their activation path. Gap **D**
  (`CFG_D` overloaded across per-head and D_model-wide units) sizes the
  activation stage buffer's row at `CFG_D`, so a `k = 2048` projection
  descriptor has no activation source at this build point. There is no
  accumulate chain to work around it either — `accumulate` is `OP_GEMM_OS`-only
  while the fuel path is WS (§5.2) — so the fuel half runs ONE legal
  descriptor, `k = 128`, covering 128 of the projection's 3584 input channels.
- **RES1 / RES2 / NORM2** — they lose their producer once o-proj is fenced out
  of the tile-level half.
- **q on-tile** — gap **A** (no narrow-AND-rope sink for `q`) and gap **C**
  (Q7 does not narrow); the q row is host-staged, as it is in every L3 case.
- **projection bias** — gap **B** (no bias adder exists in any RTL).
- **the norm gammas from DDR** — `F4` (the fuel reader's only sink is `xw`;
  gammas enter through the separate `xg` port), which is why the fuel exercise
  is a separate host-driven half and not a convenience choice.
- **the fence itself is UNPOLICED** — `F3`: `walk_desc2_check` has no
  dependency clauses, so choosing a coherent step subset and staging the
  fenced-out steps' outputs is HOST responsibility. Every enabled step's
  inputs are staged and golden-gated in the artifacts used here.
- **NEW — the multi-head, multi-KV-group attention walk itself**: `F6`, §4.
  Found and localised by A/B here. **(i)** the shared composite cache is FIXED
  upstream (combine-13, `apex_wcomp_bank`); **(ii)** the single-arm `hd_ready`
  sticky is STILL OPEN and alone still blocks a multi-head walk. The
  tile-level walked half is therefore SINGLE-HEAD (§4.4), per the lead's
  ruling.

**Provenance.** One committed decode step of one real Qwen2.5-7B layer:
**run `docs/results/s8_7b_token/artifact_trace`, step 19, layer 19, T=20,
q_pos=19, tier CQ-8, G=128**, model `mlx-community/Qwen2.5-7B-4bit`
(D=3584, H=28, H_kv=4, head_dim=128, d_ffn=18944, theta=1e6). The re-derived
per-head fields are asserted BIT-EXACT against the committed record
`job_s019_L19_h03.npz` on 22 arrays — a replay off by one token or one layer
cannot pass.

---

## 1. Stage A — the traced layer step (provenance gate)

```
$ python3 verif/walkgold/trace_layer_golden.py --step 19 --layer 19
[trace] mlx-community/Qwen2.5-7B-4bit  D=3584 H=28 H_kv=4 hd=128 d_ffn=18944 layers=28 theta=1000000.0
[trace] target step=19 layer=19 -> T=20, q_pos=19; replaying 20 steps x layers 0..19
[trace]   step   0/19  T=1  (18s)
...
[trace]   step  19/19  T=20  (429s)
[xcheck] job_s019_L19_h03.npz: head 3 bit-exact on 22 fields
[trace] wrote .../build/walkgold/layer_s019_L19.npz (1.1 MB) + .json
WG3 LAYER TRACE: PASS (step 19 layer 19, T=20 q_pos=19 CQ-8; provenance 1 committed head record(s), 22 arrays bit-exact; 28 heads, 4 kv groups, 7 weight tensors sha-pinned)
```
rc=0.

## 2. Stage B — the golden-gated replay artifacts (9 self-checks)

```
$ python3 verif/walkgold/gen_walk_golden.py
[wg3] swizzle selftest: 8 column groups match the scalar gen_l3_vectors reference (and un-swizzle round-trips)
[wg3] bundle layer_s019_L19.npz: step 19 layer 19 T=20 CQ-8 (mlx-community/Qwen2.5-7B-4bit)
[wg3] (a) descriptor + emissions: 64 DW words, 32172 expected emissions, 16532 MXE DESCs (all legal: n<=8, m<=64, k<=2048)
[wg3] (b) DDR image: 3641824 words (233.1 MB), 10 tensor regions, 16000 weight jobs, round-trip verified on 7/7 weight tensors
           Wq  base_64B=0         beats_64B=200704    lane_beats=1605632
           Wk  base_64B=200704    beats_64B=28672     lane_beats=229376
           Wv  base_64B=229376    beats_64B=28672     lane_beats=229376
           Wo  base_64B=258048    beats_64B=200704    lane_beats=1605632
           Wg  base_64B=458752    beats_64B=1060864   lane_beats=8486912
           Wu  base_64B=1519616   beats_64B=1060864   lane_beats=8486912
           Wd  base_64B=2580480   beats_64B=1060864   lane_beats=8486912
           g1  base_64B=3641344   beats_64B=112       lane_beats=896
           g2  base_64B=3641456   beats_64B=112       lane_beats=896
        phase  base_64B=3641568   beats_64B=256       lane_beats=2048
[wg3] (d) FENCED WALK (mask 0x03d = NORM1|ROPE|STOREKV|SCORE|PV): 16139 emissions (16128 attention), 532 DESCs all tile-legal; K-rope: 80 rows staged PRE-RoPE over 4 KV groups, arbiter = golden rope_fx on the f16 S-2 bus value == BusMode(rope_in_f16=True) (D-030)
[wg3] (c) per-step MMIO: 39 ops (H+11 at H=28), expectations: 28 heads, 4 KVQ groups, r1/r2 checkpoints on the fp16 grid
WG3 GOLDEN COMPILE: PASS (S1 descriptor legal; S2 epilogue replicas + RQ table golden-exact; S3 JC canonical-graded; S4 image round-trip 7/7; S5 decomposition tiles exactly; S6 divisibility held on 10 regions; S7 16532 descriptors tile-legal; S8 fuel_req widths ok; S9 MMIO == H+11)
```
rc=0. Self-check counts: **S1** 1 descriptor · **S2** 2 epilogue replicas +
28 RQ slots · **S3** 4 JC composites · **S4** 7/7 tensors round-tripped ·
**S5** 7 tilings · **S6** 10 regions · **S7** 16,532 descriptors ·
**S8** 10 fuel_req records · **S9** 39 == H+11 MMIO.

## 3. Stage C — the executor cases (self-checks C1–C5)

```
$ make -C verif/top/wg3 case
[wg3c] (C-1) fenced_h28.sub.ops: 64 DW words, 16139 emissions (532 DESCs), 160 staged records over 4 KV groups, 28 heads
[wg3c] (C-2) fuel_case: ONE Wq job block k=2048 n=8 = 16384 B = 256 DDR words (sha fe773ebe6495d87e), acc range [-6331, 2949]
WG3 CASE RENDER: PASS (C1 emissions byte-identical to stage B; C2 160 records from golden K_f16/V_f16; C3 fuel block byte-identical to the full image and un-swizzles to Wq[0:2048, 0:8]; C4 reference recomputed independently, INT32-safe; C5 full-k composition reproduces golden q_real[0:8] bit-exactly (q_real[0]=-50.571805))
```
rc=0.

**C5 is the fuel half's golden tie**: the INT32 accumulator of the golden `h8`
decode row against the on-disk `Wq`, composed with the golden's own `s_h`,
`s_wq` and `bq`, reproduces the traced `q_real[0:8]` of the real layer with
delta exactly 0.0. The fuel result check is therefore anchored to a real
Qwen2.5-7B projection value, not to an arithmetic self-consistency.

### F1 — measured before AND after the fix

**Before** (this branch's base, `0c99a32`): the FULL `EN_ALL` stream compiled
at the implemented tile's legal width diverged at the first projection
descriptor —

```
FAIL [layer_s019_L19 D=128 T=20 H=28 HKV=4 DM=3584 DF=18944] check 16210: expected 'AJ 1 1 0 1 10 0' got 'DESC 02 1 200 e00 1 1 fa90 18 1'
WALKER2 RESULT: cases=1 checks=16210 errors=16208 kvw=9528 heads=28
WALKER2 FAIL
```
`n = 0xe00 = 3584` against the implemented `MXE_N = 8` — F1 reproduced on a
real traced layer.

**After** the combine-11 constant split (`WALK2_N_MXE=8` for the MXE n-split,
`WALK2_N_JOB=4095` kept for `LAYER_JOB` cols), on the merged tree — the same
vectors, unchanged:

```
$ ./build/obj128w2/Vtb_walker2_sb +vectors=build/layer/layer_s019_L19.sub.ops +lat_mode=1 +bp_mode=1 +seed=503
WALKER2 RESULT: cases=1 checks=32172 errors=0 kvw=9528 heads=28 mxelegal=16532
WALKER2 PASS
```
rc=0. **The FULL real-Qwen2.5-7B decoder-layer step — NORM1 | QKV | ROPE |
STOREKV | SCORE | PV | OPROJ | RES1 | NORM2 | FFN | RES2 — now walks bit-exact
and in-order at unit level: 32,172 emissions, 0 errors, and all 16,532 MXE
descriptors asserted legal against the CONSUMER's imported limits
(`mxelegal=16532`, the newly closed blind spot).** F1 is fixed; nothing in
this lane touched it.

**Same caveat as §4.1 applies to this number**: it is the `verif/seq_walker`
harness — the real walker RTL and the real composite RTL, with the golden
layer trace as arbiter and every DESC now asserted against the consumer's
imported limits, but `N_ENG` MODELLED per-KV-head caches rather than the
tile's. It is a UNIT-level result and may not be quoted as a tile-level gate.

**Byte-identity of my artifacts across the fix.** `gen_walk_golden.py` already
defaulted `--n-job` to `MXE_N = 8` and passes it explicitly, so the merge does
not move any artifact:

```
IDENTICAL  layer_s019_L19.ops
IDENTICAL  layer_s019_L19.sub.ops
IDENTICAL  layer_fenced_walk.ops
IDENTICAL  ddr_layer_image.bin
IDENTICAL  ddr_layer_image.json
IDENTICAL  layer_step.mmio.jsonl
IDENTICAL  layer_expect.json
```
rc=0, and stage B's nine self-checks pass with identical counts.

## 4. The WALKED half — single-head tile-level GREEN; multi-head blocked by F6

### 4.1 The fenced real-7B walk (unit level) — GREEN

(The FULL `EN_ALL` stream is also green post-F1-fix — §3. The fenced stream
below is the subset the TILE-level half targets, and it is the vector the A/B
in §4.3 uses.)

The fenced stream (mask `0x03D` = NORM1|ROPE|STOREKV|SCORE|PV) at real 7B
geometry — H=28, H_kv=4, head_dim=128, T=20, GQA-4 engine banking — against
the golden layer trace:

```
$ make -C verif/top/wg3 fenced
WG3-TOPO[control-Neng-caches] RESULT: cases=1 checks=16139 errors=0 kvw=9528 heads=28
WG3-TOPO PASS
WG3 FENCED: PASS
```
rc=0. **16,139 emissions bit-exact and in-order; 0 errors; 28 head interlocks;
9,528 KVQ writes; the per-head engine select checked against the golden GQA
mapping `h // (H/H_kv)` at every head interlock and every KVQ read-address
write.** All 532 MXE descriptors in this stream are tile-legal by S7.

**What this is and is not.** This is the `verif/seq_walker` harness topology:
the real `seq_layer_walker2` RTL, the real `seq_walker_comp` RTL, and the
golden layer trace as the arbiter — but `N_ENG` modelled per-KV-head composite
caches rather than the tile's. It is NOT the real `apex_top`. Per the F1
lesson (a walker-only scoreboard can be a self-consistent oracle), this result
may not be quoted as a tile-level gate.

### 4.1b K-rope artifacts — the golden side (the RTL run is §4.5)

`build/walkgold/stage_K_g{0..3}_f16.npy` (K staged PRE-RoPE) and
`expect_Krope_g{0..3}_f16.npy` (the expectation) are generated and checked:
**80 rows over 4 KV groups**, of which 76 rotate and 4 are the `t = 0`
identity (19 of 20 rows differ from the staged value in each group).

Per **F5** — now a contract-level rule in §9.1 — the expectation is pinned to
the **D-030 arbiter**: golden `rope_fx` applied to the f16 S-2 BUS value,
verified equal to `BusMode(rope_in_f16=True)`. It is **not** pinned to the
legacy `K_f16`/`K_rope` trace field, which would fail a correct tile in 76 of
80 rows because `K_real` is not on the fp16 grid.

These artifacts are consumed by the RTL run in **§4.5**, which is green,
including a must-fail arm pinned to the legacy field.

### 4.2 F6 — the capability gap this lane found (i) FIXED, (ii) OPEN

**Status after combine-13:** F6(i) — the single shared composite cache — is
**FIXED** upstream by `apex_wcomp_bank` (per-KV-head caches behind the one
engine select, byte-identical at default, `mW2` reproducing this signature).
F6(ii) — the single-arm `hd_ready` sticky — is **still open**, and it alone
still blocks a multi-HEAD walk. The record below is what this lane measured
and escalated; it is kept because it is the evidence the fix was made against.

`apex_top` cannot complete a multi-head, multi-KV-group fmt=1 attention walk.
Two structural facts, both at source:

- **(a) one shared composite cache.** `rtl/top/apex_top.sv:583` instantiates a
  SINGLE `seq_walker_comp u_wcomp` — no generate loop. Its `sc_mem` is indexed
  by RECORD ADDRESS only (`seq_walker_comp.sv:126`, 256 entries), and the
  address it caches is the raw KVQ `WRITE_ADDR` snoop
  (`apex_top.sv:574-581`). At `kv_map=1` every KV group writes records
  `0..2T-1`, so the 4 groups **collide**: only the last-staged group's `s_k`
  and `s_v` survive. The verified scoreboard models `N_ENG` caches precisely
  because the walker needs them (`tb_walker2_sb.sv:146-166`).
- **(b) a single-arm head interlock.** `hd_ready` is the sticky
  `hd_sq_seen_q` (`apex_top.sv:624-629`), armed by any accepted `ss` tap beat
  and consumed by the head handshake. While `walk_en_q` is set,
  `apex_top.sv:511-546` forces **every** host job port's ready to 0
  (`fj/qj/dj/aj/wj/qs/cs/ds`) and parks the whole KVQ AXI window
  (`apex_top.sv:552-567`), so no later head's q row can be staged mid-walk.
  The RTL says so itself at `apex_top.sv:618-621`: *"only single-head fmt=1
  images can complete an attention walk (n_heads=1 — the walkfmt2 case)"*.

This is the RTL-side counterpart of the §9.1 W-G2 disclosed caveat
("hd_ready's final arbitration … single-head arming until the q-projection
path"). **The composite-cache instance count is new — it is not in the ledger,
and it blocks the walk independently of the interlock.**

### 4.3 The A/B — the topology is the blocker, not the walker and not the vectors

*(Since combine-13, arm B no longer models the current tile — it is retained
as the REGRESSION WITNESS for the fixed F6(i) defect, and its banner now reads
`pre-F6i-single-cache`. A green arm A does NOT mean a multi-head walk works:
F6(ii) still blocks it.)*

Same walker RTL, same vectors, same scoreboard, same seed; the ONLY delta is
the tile topology (`verif/top/wg3/tb_wg3_topo_ab.sv`, a copy of the verified
scoreboard with one `+topo` plusarg).

```
$ make -C verif/top/wg3 topo
TOPO A/B: PASS (control green, apex_top-as-built RED as expected)
WG3-TOPO[control-Neng-caches] RESULT: cases=1 checks=16139 errors=0   kvw=9528 heads=28
WG3-TOPO[apex_top-as-built]   RESULT: cases=1 checks=16139 errors=980 kvw=9528 heads=28
```
rc=0 (a discrimination gate: the control arm must be green and the
apex_top arm must be RED — an arm that stopped failing would mean the
experiment had lost its power).

**The failure fingerprint is exactly the diagnosis.** All 980 errors are
composite words, and nothing else — the walk's STRUCTURE is unaffected
(`kvw=9528` and `heads=28` identical in both arms):

| word | wrong | total | why |
|---|---|---|---|
| `CS` | **560** | 560 | needs `s_q` (stale from head 1 on) AND `s_k[t]` (last-staged group only) — every one is wrong |
| `QS` | **420** | 560 | needs only `s_v[t]`; the 140 belonging to the last-staged group survive |

Attributing every failing check back to its head confirms the mechanism rather
than assuming it — the surviving `QS` words are **exactly** the last-staged KV
group, and nothing else:

```
QS-wrong heads  : [0 .. 20]
QS-correct heads: [21, 22, 23, 24, 25, 26, 27]  -> kv groups [3]
CS-wrong heads  : 28 of 28
```
`h // (H/H_kv) = h // 7`, so heads 21–27 are KV group 3 — the last group
staged, hence the only one whose `s_v` entries are still resident in the one
shared cache when the walk runs. `CS` additionally needs `s_q`, which the
single latch cannot refresh per head, so no head escapes.

First divergence, at head 0's very first composite:
```
FAIL [fenced_h28 D=128 T=20 H=28 HKV=4 DM=3584 mask=03d] check 14: expected 'CS 414a0ff4' got 'CS 40ec634e'
```

### 4.4 TILE-LEVEL walked half, SINGLE-HEAD — GREEN (the lead's option 3)

Integration-lead ruling, 2026-07-26: F6 accepted, re-scope the tile-level half
to `n_heads = 1` at real 7B PER-HEAD geometry. Built and green.

**The case.** `wg3_s019_L19_h03_hd128_T20` — the REAL traced Qwen2.5-7B
**head 3** of step 19 / layer 19 (the head carrying the committed S8 record),
`head_dim = 128`, `T = 20`, CQ-8 — at the I-B CL build point
`CFG_D=128 KVQ_DEPTH=256 KVQ_GQA_NENG=4 RMS_D_MAX=3584 LAYER_DM_MAX=3584`.

It is built by a NEW SIBLING (`verif/top/wg3/gen_wg3_tile_case.py`) that
IMPORTS `gen_l3_vectors` and calls its own `core_case` builder — the verified
host choreography — with the traced head's arrays. `core_case` takes `q`,
`K16`, `V16` as INPUTS and injects them through squant, which is what makes
this possible at all: the real head's K/V **cannot be produced on-tile here**
— gap **B** (no projection-bias adder; Qwen2.5 k/v projections carry biases)
and gap **D** (`CFG_D` sizes the activation stage row, so the real `k = 3584`
projection has no activation path). Injection stages the traced values
bit-exactly and sidesteps both.

```
$ make -C verif/top/wg3 tilehost
L3 RESULT: cycles=146100 checks=5541 errors=0
L3 PASS
WG3 TILE HOST: PASS (real 7B head, host-sequenced)

$ make -C verif/top/wg3 tilewalk
[49284]  WALKFMT2: fmt=1 image loaded (deep words 21/22), walking single-engine attention subset (T=20)
[145720] WALKFMT2: fmt=1 WALKED clean (idle, no error, FMT_SUP=0011)
[145722] WALKER: score+pv complete
L3 RESULT: cycles=145740 checks=5544 errors=0
L3 PASS
WG3 TILE WALK: PASS (real 7B head, SINGLE-HEAD walked via fmt=1)
```
both rc=0. The fmt=1 image is loaded through the **real WALK CSR window**
(including the deep SRAM words 21/22), the walker drives score+pv, and
**walker mode loses no checks** vs host mode (5,544 >= 5,541) — the B1 §B-1
rule. Generator self-checks P1–P6: P1 re-asserts the committed-record
provenance on **21 fields** at the tile-case boundary; P2 the staged q/K/V are
the traced fields themselves; P3/P4 fmt=1 legality + per-head v1 equivalence;
P5 675 descriptors tile-legal (+1 deliberate `k=0` reject probe, excluded BY
VALUE, not by position); P6 below.

**P6 — a lying expectation caught and corrected.** `gen_l3_vectors.phase_a`
scripts `INFO_TIER = 0x3` at D=128, which is the TIER-BANK build's truth. At
`KVQ_GQA_NENG=4` the GQA bank REPLACES the tier bank and the tile is
CQ-8-only, so `apex_top.sv:1786-1789` reports `0x1` — *"a GQA build
(KVQ_GQA_NENG>1) is CQ-8-only — INFO_TIER never lies (D-027)"*. The generator
rewrites that one line to `0x1` by VALUE with an exact-count assert. Scripting
`0x3` here would have been the lying gate.

### 4.5 K-ROPE coverage, GQA-4, host mode — GREEN, with an F5 discrimination arm

This needs no walk, so **F6 does not block it** — it is the GQA-4 hardware
exercise available today. 80 real Qwen K rows (10,240 fp16 elements) staged
PRE-RoPE through the real S-2 path — `apex_scale_quant` MODE_F16 →
`rope_row` → the per-KV-head `apex_kvq_gqa_bank` engine — across **all four**
engines, selected in host mode through `LAYER_CTRL[17:15]`
(`gqa_eng_sel = walk_en_q ? wk_kv_eng_sel : l_kv_map_q`, `apex_top.sv:1236`).
The `dbg_f16` tap carries the POST-RoPE beat
(`kv_s_tdata = l_rope_en_q ? rr_m_data : sqf_data`, `apex_top.sv:935-937`).

```
$ make -C verif/top/wg3 krope
L3 RESULT: cycles=118077 checks=10262 errors=0
L3 PASS
WG3 KROPE: PASS (80 real K rows, 4 GQA engines, D-030 arbiter)
F5 DISCRIMINATION: PASS (legacy-pinned arm RED as predicted)
[55885000] %Error: tb_apex_l3.sv:391: Assertion failed in tb_apex_l3: [tap f16] idx 128: got bc5c exp bc5d
```
rc=0.

**F5, now MEASURED on the real tile rather than asserted from the ledger.**
The gate's arbiter is golden `rope_fx` on the f16 S-2 bus value (== D-030
`BusMode(rope_in_f16=True)`). A second arm pins the identical case to the
LEGACY `K_f16` trace field and **must fail** — and does, at the first rotated
row (`idx 128` = row 1, element 0) by exactly **1 fp16 ULP**. Across the case,
**76 of 80 rows** differ between the two arbiters — precisely the figure §9.1
F5 predicts. This is what the segment-1 O3′-1 K-path gate needed.

### 4.6 Two NEW findings, both of the F3 "unpoliced" class

Found while building §4.5. Nothing refuses either mistake, and both produce
**plausible fp16 values, never an X** — so both are silent-wrong-answer traps.

- **(a) `CSR STATUS.idle` does not cover `rope_row`.** The §3b LAYER level
  registers (`l_rope_pos`, `l_kv_map`) may only change while the streams they
  steer are quiescent, but `rr_busy` is not a CSR STATUS lane at all — it
  lives in **`LAYER_STATUS[3]`** alongside `res/swg/ldq`
  (`apex_top.sv:917-919`). Quiescing on CSR STATUS alone let the next row's
  `LAYER_CTRL` write land mid-row: **pairs 0–2 of row 0 took row 0's phase and
  pair 3 took row 1's.** Correct quiesce = CSR `STATUS.idle` **and** KVQ
  `STATUS.idle` **and** `LAYER_STATUS[3:0] == 0`.
- **(b) `phase_a`'s D-020 soft reset only reaches engine 0.** It writes the
  KVQ window, which routes to `engine[l_kv_map]` — 0 at that point. In a GQA
  build engines 1..N-1 are never initialised and **wedge the `qj` port on
  their first record**. Each engine needs its own reset.

## 5. The FUEL half — GREEN end to end (tile leg + DDR leg), geometry CORRECTED

Per **F1** no WALKED projection may be attempted; this half is host-driven at
the legal `n = 8`.

### 5.1 The compact DDR case image

The **compact** DDR case image — ONE real `Wq` weight block lifted
byte-for-byte out of the 233 MB stage-B image, with its regions manifest and
the golden INT32 reference (`build/walkgold/fuel_case/`). The full image is
deliberately NOT used: 233 MB through the PCIS model is not a sane load.

```
[wg3c] (C-2) fuel_case: ONE Wq job block k=128 n=8 = 1024 B = 16 DDR words (sha ef59dad01481b7ff), acc range [-2734, 3193]
```
Self-checks C3–C5 hold at this geometry: the block is byte-identical to the
corresponding slice of the full image, un-swizzles back to `Wq[0:128, 0:8]`,
re-swizzles identically, and the **C5 golden tie** still binds — the full-k
composition of the same operand pair reproduces the traced `q_real[0:8]`
bit-exactly (`q_real[0] = -50.571805`).

### 5.2 CORRECTION — my own de-risking note was WRONG

The previous revision of this document proposed consuming the full `k = 2048`
decomposition job (256 DDR words) as **16 accumulating `(m=1, k=128, n=8)`
descriptors**, since gap D bounds a WS activation row at `CFG_D`. **The RTL
refutes that**, and I am recording it rather than forcing it:

```
rtl/mxe/mxe_ctrl.sv:286
    clear_job <= !((desc.opcode == OP_GEMM_OS) && desc.accumulate);
rtl/apex_pkg.sv:44
    logic accumulate;   // [66]  OS: retain accumulators
```

`accumulate` is honoured for **`OP_GEMM_OS` only** — a WS job ALWAYS clears its
accumulators. And the fuel line feeds the **external `xw` weight port**, which
is the WS projection path (`rt_wgt_src=0`); an OS job takes its B operand from
the weight stage buffer instead. So on the fuel-fed path there is **no
accumulate chain available at all**, and `k` stays bounded by the activation
row.

**Corrected drivable geometry:** `m <= 64`, `k <= CFG_D = 128`, `n = 8` —
**1024 B = 16 DDR words**, not 256. This costs nothing in image layout: the
swizzle is `p`-major over `p = k/8`, so `W[0:128, 0:8]` is exactly the first
1024 bytes of the `k=2048` job's block — a strict prefix. Only how much of the
region one legal descriptor consumes changes.

The consequence for the claim is real and should be stated when the fuel half
lands: a single fuel-fed WS descriptor covers **128 of the 3584** input
channels of the q projection. Covering the whole projection needs either
host-side summation of 28 such results, or the gap-D fix (a D_model-wide
activation path), or an OS-mode fuel path — none of which is in this lane.

### 5.3 The TILE leg — GREEN

The projection GEMM itself is built and green, with the weight bytes arriving
on the **external `xw` port** — the exact port the IB-FUEL reader feeds:

```
$ make -C verif/top/wg3 fuel
[wg3f] case wg3_fuel_wq_k2944_hd128: REAL Wq[2944:3072, 0:8] of step 19 layer 19, m=1 k=128 n=8 OP_GEMM_WS accumulate=False
[wg3f] 128 lane8 beats on the external xw port = 1024 B = 16 DDR words (sha 34845401d5e2ac5e)
[wg3f] 20 scripted checks; expected INT32 acc = [-127, -983, -1931, -703, -2000, -3068, 468, 732]
WG3 FUEL CASE: PASS (F1 slice holds the global amax; F2 re-quantization is the identity; F3 beats byte-identical to the DDR image block; F4 descriptor tile-legal, no accumulate; F5g reference recomputed independently, INT32-safe)
L3 RESULT: cycles=2155 checks=24 errors=0
L3 PASS
WG3 FUEL (tile leg): PASS (real Qwen Wq via the external xw port)
```
rc=0. Real Qwen2.5-7B `Wq[2944:3072, 0:8]`, the golden `h8` decode-row slice
as activation, and the exact INT32 accumulators checked on the result bus.

**F3 is the join.** The beats this case pushes are asserted BYTE-IDENTICAL to
the compact DDR case image's block, so the tile leg and the DDR leg meet on a
proven identity rather than an assumption.

**The activation-staging obstacle is solved.** `inject_jobs` MODE_QUANT
RE-quantizes what it is handed (`q8 = round(v·127/amax_local)`), so an
arbitrary 128-slice of `h8` does NOT stage bit-exactly. Fix: choose the
128-aligned slice that CONTAINS the row's global amax (here `k0 = 2944`), so
`amax_local == 127` and the re-quantization is the identity — asserted (F1/F2),
not assumed.

### 5.4 The DDR leg — GREEN

The same projection GEMM with the weight beats sourced from DDR
(loader → `sh_ddr` → `apex_fuel_reader` → afifo → `xw` → tile) instead of the
host mailbox. **All 128 of the projection GEMM's WB triples are removed**, so
nothing pushes weights over the mailbox at all.

```
$ make -C verif/top/wg3 fuelddr
# arm 1 — host baseline (mailbox weights, full cl_apex executor)
[.../wg3_fuel_wq_k2944_hd128.regops.jsonl] 1323 ops, 30 checks, 0 fails (cyc 143209)
F2SIM RESULT: files=1 checks=30 fails=0 -> PASS

# arm 2 — FUEL: weights FROM DDR
DDRLOAD [wg3_fuel_wq_k2944_hd128] base=0x0 words=16 tag=0 ok
DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS
[.../wg3_fuel_wq_k2944_hd128.fuel.regops.jsonl] 949 ops, 32 checks, 0 fails (cyc 111049)
FUELAUDIT [.../wg3_fuel_wq_k2944_hd128.fuel.regops.jsonl] err=00000000 stat=00000009 -> ok
F2SIM RESULT: files=1 checks=32 fails=0 -> PASS

# arm 3 — DISCRIMINATION: one INT8 lane perturbed in DDR. MUST FAIL.
DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS
FAIL L921: [3204] got ffffff88 want ffffff81 (mask ffffffff)
F2SIM RESULT: files=1 checks=32 fails=1 -> FAIL

WG3 FUEL (DDR leg): PASS (weights from DDR, golden-gated; perturbed-DDR arm RED as required)
```
rc=0.

**The result is the same reference the tile leg used** —
`[-127, -983, -1931, -703, -2000, -3068, 468, 732]`, recomputed independently
by `einsum` and range-checked for INT32 (F5g), never re-derived from the tile.

**Provenance is asserted positively, not inferred from absence.** The fuel arm
polls `FUEL_STAT.ddr_ready`, waits for the record to retire
(`FUEL_STAT[2:1] == 0`), reads back `FUEL_STAT[15:8] == tag`, and checks
`FUEL_ERR[3:0] == 0` (no AXI error, no bad descriptor, no rejected mode
switch, no mailbox push while `src=fuel`).

**Arm 3 is the load-bearing evidence.** The perturbed image's manifest digest
is updated to match, so it passes the loader's own readback + SHA check
(`rb_fails=0 sha_fails=0`) and can only be caught **by the GEMM result** —
lane 0 comes out `-120` instead of `-127`, exactly the delta from a +1 on the
first weight byte. The gate requires arm 3 to fail *at the result* and errors
out if it fails at load instead, because that would be weaker evidence.

**Two things the case had to get right** (both recorded in the generator):

1. **`inject_jobs` also uses WB triples** — it stages the ACTIVATION with K=2
   WS GEMMs, 128 more triples. Only the projection GEMM's 128 may be
   stripped, split at its own note marker. And fuel mode must be armed *only*
   there: a mailbox push while `src=fuel` is REJECTED and sets
   `FUEL_ERR.host_push` (IB_FUEL §5 bit 3), so arming it earlier would break
   the activation staging.
2. **ONE case, TWO build points.** The wg3 tile binary is the I-B CL build
   point (`KVQ_GQA_NENG=4` → `INFO_TIER = 0x1`), while `cl_apex` instantiates
   `apex_top` with `KVQ_GQA_NENG` at its DEFAULT of 1
   (`cl_apex.sv:579-590` → the tier bank → `INFO_TIER = 0x3`). Each variant
   asserts its OWN build's truth; asserting the other would be a lying gate.

**The named obstacle is solved as proposed.** `trace_to_regops.Xlate` is a
general op-script → regops translator, so it is IMPORTED and driven directly
rather than made to walk a trace directory.

## 6. Build point

`CFG_D=128 KVQ_DEPTH=256 KVQ_GQA_NENG=4 RMS_D_MAX=3584 LAYER_DM_MAX=3584`
elaborates `-Wall` clean (waivers scoped to frozen/vendored files only). The
recipe is `make -C verif/top/wg3 lint`; it scrapes the tile file list from the
L3 suite's `RTL_CORE` rather than copying it, so the two cannot drift, and
hard-fails if the scrape comes back empty rather than "linting nothing":

```
$ make -C verif/top/wg3 lint
- Verilator: Built from 8.819 MB sources in 54 modules, into 3.039 MB in 10 C++ files needing 0.000 MB
- Verilator: Walltime 0.465 s (elab=0.049, cvt=0.318, bld=0.000); cpu 0.447 s on 1 threads; alloced 56.172 MB
WG3 BUILD POINT LINT CLEAN (CFG_D=128 KVQ_DEPTH=256 KVQ_GQA_NENG=4 RMS_D_MAX=3584 LAYER_DM_MAX=3584)
```
rc=0.

## 7. Pre-existing suites — re-run on the merged tree

All eleven gates below were re-run **after rebasing onto the integration tip**
(`e88a77c` — combine-13, which landed F6(i) `apex_wcomp_bank`), serially,
`set -o pipefail`, one Verilator build at a time per the machine rule.
**Every rc is 0.**

| gate | rc |
|---|---|
| `make -C verif/top/l3 all` | 0 |
| `make -C verif/top/l3 run_walker` (inside `all`) | 0 |
| `make -C verif/top/l3 run_walkfmt` | 0 |
| `make -C verif/top/l3 run_walkfmt2` | 0 |
| `make -C verif/seq_walker all` | 0 |
| `make -C verif/top/l4 compose levels mutate` | 0 |
| `make -C verif/kvq/gqa all` | 0 |
| `make -C verif/kvq/mask all` | 0 |
| `make -C verif/f2sim clean build run mutants` | 0 |
| `make -C golden test` | 0 |
| `make -C verif/top/wg3 all` (this lane) | 0 |

This lane is purely ADDITIVE — `verif/walkgold/gen_wg3_case.py`,
`verif/top/wg3/`, `docs/results/wg3/`. No existing RTL, generator, TB or doc
was edited, so nothing above could move; the numbers are unchanged.

Two notes so the logs are not misread:

- `verif/seq_walker`'s log contains `FAIL C2 row …` lines and
  `verif/f2sim`'s contains `F2SIM RESULT: files=1 … -> FAIL`. Both are
  **mutation gates catching their mutants** (`MUTATION GATE 3: 2/2 …`,
  `MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS`). They are the
  gates working, not regressions.
- `verif/f2sim` needs three gitignored inputs. Regenerated in the documented
  order before the run: `trace_to_regops.py` (18 jobs) →
  `make_ddr_image.py` (+ `--check`: `DDRIMG CHECK: jobs=18 fails=0 -> PASS`)
  → `trace_to_fuel.py` (18 jobs). The executor's counted checks come out at
  the contracted **27,996**, and the mutants gate exercises the FUEL path
  end-to-end (`FUELAUDIT … err=00000000 stat=00000009 -> ok`) — so the DDR →
  reader → afifo → `xw` → tile line is live in this tree, just not yet
  carrying a W-G3 projection case (§5).

Verbatim gate banners:

```
$ make -C verif/top/l3 all
LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg + vendored files only)
B1 COMPOSITE GATE: PASS
STAGEBUF PATD RESULT: cycles=505 checks=184 errors=0
STAGEBUF PATD PASS
WALKER-MODE L3: all cases passed
MUTATION GATE: PASS (3/3 tile-level mutants caught)
RC=0

$ make -C verif/top/l3 run_walkfmt
[30784] WALKFMT: FMT_SUP=0011, deep DPTR ok, fmt=1 REFUSED (DESC), W1C ok
L3 PASS
RC=0

$ make -C verif/top/l3 run_walkfmt2
[30776] WALKFMT2: fmt=1 image loaded (deep words 21/22), walking single-engine attention subset (T=8)
[41563] WALKFMT2: fmt=1 WALKED clean (idle, no error, FMT_SUP=0011)
L3 PASS
RC=0

$ make -C verif/seq_walker all
MUTANT m12_waddr CAUGHT:
MUTANT m13_nmxe: seq/seq_walker_pkg.sv patched
MUTANT m13_nmxe CAUGHT by the required signature:
SPEC MUTANT m13_nmxe_spec CAUGHT by the required signature:
CHECKEQ RESULT: c2=6013 c1=6007 errors=0
CHECKEQ PASS
MUTANT m4_fmtskip: seq/seq_walker_pkg.sv patched
CHK MUTANT m4_fmtskip CAUGHT:
MUTANT m9_resvskip: seq/seq_walker_pkg.sv patched
CHK MUTANT m9_resvskip CAUGHT:
RC=0

$ make -C verif/top/l4 compose levels mutate
L4COMPOSE PASS (segment 2 RMSNorm-1 front-half; h8 deferred to segment 1, IB_LAYER sec 3c)
L4LEVELS RESULT: checks=27 errors=0 -> PASS
L4LEVELS PASS
mC1: CAUGHT (value; signature ['[EFS]'])
mC2: CAUGHT (value; signature ['[EFS]'])
L4 SEG2 MUTATION GATE: PASS (3/3 front-half mutants caught)
RC=0

$ make -C verif/kvq/gqa all
GQA VECTORS: 22 records over 4 engines (+1 rewrite row), golden = cq_codec compress/decompress_values (CQ-8), aliasing self-checks PASS
GQA BANK gqa4: ALL PASS
GQA SUITE: ALL PASS (gqa4 1859 checks, 3/3 signature mutants caught)
RC=0

$ make -C verif/kvq/mask all
MASK GATE mask_rom: ALL PASS
MASK GATE mask_csr: ALL PASS
MASK SUITE: ALL PASS (rom 800 + csr 1887 checks, 2 mutants caught)
RC=0

$ make -C verif/f2sim clean build run mutants
GATE[control]: exit=0 want=PASS got=PASS  OK
F2SIM RESULT: files=1 checks=507 fails=469 -> FAIL
GATE[M1]: exit=1 want=FAIL got=FAIL  OK
F2SIM RESULT: files=1 checks=507 fails=128 -> FAIL
GATE[M2]: exit=1 want=FAIL got=FAIL  OK
F2SIM RESULT: files=1 checks=9 fails=1 -> FAIL
GATE[M3]: exit=1 want=FAIL got=FAIL  OK
MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS
RC=0

$ make -C golden test
W4B FEEDER ORACLE (D-031): 7 checks, ALL PASS
GOLDEN SUITE: contract + compute + attention + transformer + plumbing7b + effbits + masksem ALL PASS
GOLDEN SUITE (B3): + weightcodec ALL PASS
GOLDEN SUITE (IB-LAYER): + layerbus ALL PASS
GOLDEN SUITE (W4B): + w4bfeeder ALL PASS
RC=0

$ make -C verif/top/wg3 all   # this lane
WG3 TILE HOST: PASS (real 7B head, host-sequenced)
L3 RESULT: cycles=145740 checks=5544 errors=0
WG3 TILE WALK: PASS (real 7B head, SINGLE-HEAD walked via fmt=1)
WG3 KROPE CASE: PASS (R1 staged rows are the stage-B artifacts; R2 t=0 identity + 19/20 rows rotate per group; R3 phase codes fit the 14-bit LAYER word; R4 all 10240 elements injectable)
L3 RESULT: cycles=118077 checks=10262 errors=0
WG3 KROPE: PASS (80 real K rows, 4 GQA engines, D-030 arbiter)
WG3 KROPE CASE: PASS (R1 staged rows are the stage-B artifacts; R2 t=0 identity + 19/20 rows rotate per group; R3 phase codes fit the 14-bit LAYER word; R4 all 10240 elements injectable)
F5 DISCRIMINATION: PASS (legacy-pinned arm RED as predicted)
WG3 FUEL CASE: PASS (F1 slice holds the global amax; F2 re-quantization is the identity; F3 beats byte-identical to the DDR image block; F4 descriptor tile-legal, no accumulate; F5g reference recomputed independently, INT32-safe)
L3 RESULT: cycles=2155 checks=24 errors=0
WG3 FUEL (tile leg): PASS (real Qwen Wq via the external xw port)
F2SIM RESULT: files=1 checks=30 fails=0 -> PASS
DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS
FUELAUDIT [build/fuel/wg3_fuel_wq_k2944_hd128.fuel.regops.jsonl] err=00000000 stat=00000009 -> ok
F2SIM RESULT: files=1 checks=32 fails=0 -> PASS
WG3 FUEL (DDR leg): PASS (weights from DDR, golden-gated; perturbed-DDR arm RED as required)
DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS
FUELAUDIT [build/fuel/wg3_fuel_wq_k2944_hd128.fuel.regops.jsonl] err=00000000 stat=00000009 -> ok
F2SIM RESULT: files=1 checks=32 fails=0 -> PASS
DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS
FUELAUDIT [build/fuel/wg3_fuel_wq_k2944_hd128.fuel.regops.jsonl] err=00000000 stat=00000009 -> ok
F2SIM RESULT: files=1 checks=32 fails=1 -> FAIL
RC=0
```

## 8. Declaration — status, provenance, and what is NOT claimed

### 8.1 The F6 ruling — EXECUTED

Integration lead, 2026-07-26: F6 accepted and recorded in
`LEVEL_C_INTEGRATION` §9.1. **Ruling = option 3**: re-scope the tile-level
walked half to `n_heads = 1` at real 7B per-head geometry, plus the host-mode
GQA-4 K-rope coverage. Option 1 (per-KV-head composite caches) is being done
in PARALLEL by a separate lane — **this lane did not touch `apex_top`**.
Option 2 stays deferred, and the interim patch-CSR variant was REFUSED
precisely because it would move the disclosed ~39 MMIO/step figure.

Both halves of the ruling are **GREEN**: §4.4 (single-head walked, real 7B
head, through the real WALK window) and §4.5 (GQA-4 K-rope with an F5
discrimination arm). The final task — the **fuel DDR leg** — is also green
(§5.4). The multi-head walk remains blocked by **F6(ii)** and is not claimed
anywhere in this document.

**Re-gate done after combine-13.** F6(i) landed upstream as
`apex_wcomp_bank`, so `make -C verif/top/wg3 topo`'s arm B stopped being a
model of the tile. It has been **relabelled** rather than deleted: its banner
now reads `pre-F6i-single-cache` and it is retained as the REGRESSION WITNESS
for the fixed defect, still reproducing the exact 980-error signature the fix
had to eliminate. The gate still requires arm A green and arm B RED. **A green
arm A does not imply a working multi-head walk** — F6(ii)'s single-arm
`hd_ready` sticky is independent and still open.

### 8.2 Scoreboard

| item | state |
|---|---|
| tile-level walked half, single-head | **DONE** — §4.4, green |
| GQA-4 K-rope coverage | **DONE** — §4.5, green, with a must-fail F5 arm |
| fuel half — TILE leg (`xw` → MXE → golden) | **DONE** — §5.3, green, byte-identical to the DDR image block |
| fuel half — DDR leg (loader → DDR → reader → afifo → `xw`) | **DONE** — §5.4, green, with a perturbed-DDR discrimination arm |
| multi-head walked half | F6(i) FIXED upstream (combine-13); still blocked by **F6(ii)** `hd_ready`; not this lane |
| F1 | fixed in combine-11, verified green here |
| F6 | accepted and ruled on; the tile fix is another lane's |
| F5 | promoted from ledger claim to a MEASURED tile result (§4.5) |
| new: LAYER-level quiesce + per-engine reset (§4.6) | recorded; both are host-responsibility traps of the F3 class, unpoliced by any check |

### 8.3 The provenance chain

Every number in this document traces back through an unbroken chain of
bit-exact links; no step re-derives a golden value:

1. **Committed run** — `docs/results/s8_7b_token/artifact_trace/run.json`:
   the S8 token sequence (prompt + generated ids) of a real Qwen2.5-7B decode.
2. **Stage A** replays that exact sequence through the S8 `GoldenModel` /
   `decoder_layer_fx` and captures **step 19, layer 19** (T=20, q_pos=19,
   CQ-8, G=128). Its **provenance gate**: the re-derived per-head fields are
   asserted BIT-EXACT against the committed record `job_s019_L19_h03.npz` on
   **22 arrays**. A replay off by one token or one layer cannot pass. All
   seven weight tensors are sha256-pinned in the bundle metadata.
3. **Stage B** compiles that bundle into the replay artifacts through the
   EXISTING IB-WALK compiler (one wire-format source), with 9 self-checks
   (S1–S9) including a 7/7 DDR-image round-trip.
4. **Stage C** re-renders those artifacts into executor form, copying every
   emission VERBATIM (C1) and taking every staged value straight from the
   bundle (C2). C5 ties the fuel operands back to the traced `q_real[0:8]`
   bit-exactly.
5. **The tile case** re-asserts the committed-record provenance a second time,
   at the tile-case boundary, on **21 fields** (P1) — so the case the RTL runs
   is pinned to the same committed head, not merely to stage A.
6. **The fuel block** is lifted BYTE-FOR-BYTE out of the stage-B image (C3),
   its beats are asserted byte-identical to what the tile case pushes (F3),
   and the DDR image that f2sim loads is that same byte string, verified by
   readback + SHA-256 at load (`rb_fails=0 sha_fails=0`).

Independent-arbiter note: the walked and K-rope gates are checked against
`golden/apex_golden`, and the fuel gate against an `einsum` recomputation of
the same INT8 operands — never against the tile's own output.

### 8.4 Honest summary of what W-G3 may be quoted as
> A single-head partial-layer walk at real Qwen2.5-7B per-head geometry
> (head_dim 128, T=20, CQ-8, one committed traced head), driven through the
> real WALK CSR window on the I-B CL build point, plus host-mode GQA-4 K-rope
> store coverage across all four engines — both bit-exact against golden.

> …plus a host-driven projection GEMM at the legal `n = 8` over real
> Qwen2.5-7B `Wq` weights **fed from DDR** through
> loader → `sh_ddr` → `apex_fuel_reader` → afifo → `xw` → tile, bit-exact
> against an independently recomputed INT32 reference, with a perturbed-DDR
> arm proving the result depends on the DDR contents.

**Not claimable:** a walked FULL layer; a walked MULTI-head step (F6(ii)); a
walked PROJECTION (F1's fence, and gaps A/B/D); a WHOLE projection from fuel
(one descriptor covers 128 of 3584 input channels); anything on SILICON — every
number here is Verilator simulation against a behavioral DDR model, and
`sh_ddr_beh.sv` is this project's own fabrication of an encrypted controller,
never a hardware claim.
