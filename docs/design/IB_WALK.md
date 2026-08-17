# IB-WALK — full-layer walker descriptor extension (lane contract, stage 0)

**Lane:** `comp/ib-walk` off `comp/level-c-integration` @ 335dea0
(`docs/design/LEVEL_C_INTEGRATION.md` §9, IB-WALK row).
**Status:** stages 1-5 LANDED 2026-07-22 (stage 5 at unit level, unparked
on IB-LAYER's frozen §3b) — fmt=1 wire format + v1.1 hardening (s1);
attention/projection/fetch subset walk (s2); SV-vs-mirror check equivalence
over 12,020 rows (s3); `apex_top` fmt=1 window with both L3 modes
byte-identical (s4, zero new CSR addresses); **the FULL §3b decoder-layer
step set walking bit-exact — 6,858 checks at 7B geometry, every step, 28
heads, 4 engines (s5)**. 17 mutants killed across three gates + 3 tile
mutants. FUEL interface CONFIRMED (§2.6). Per-step figure **H+11 = 39**
(RATIFIED — §9.1 amended with the full H+5→H+6→H+11 history). **Stage-6
doc half DONE: the D-029 ARCHITECTURE entry is landed** with the four
approved kill citations. **LANE PARKED pending the COMBINE**, which owns:
walker2 instance + FMT_SUP flip against LAYER's glue + FUEL's reader; the
W-G2 tile gate vs LAYER's L4 (their S5 is running — D-030 owner-signed);
and the W-G3 flagship replay. All three §9.1 reconcile flags RESOLVED
2026-07-23 (§4 stage-5 row): (a) CONFORMED to the canonical D-030 grade
(the fp16-trip replaced, suite re-green), (b) L4 arbitration accepted with
the three extended choreography rules recorded as combine expectations
(incl. the named one-pc consumer-first swap), (c) resident phase table
won — zero steady-state phase loads. **Reconciled to `LEVEL_C_INTEGRATION.md` §9.1 (2026-07-22): where
this doc conflicted with §9.1, §9.1 won — R1 fuel_req widths folded in
(§2.6), D-029 confirmed (R2), R3 provisional kv-mapping recorded (§7 Q2),
plus IB-FUEL's s1 fuel_req bit-layout freeze.** Machine rule (§9.1): at most
ONE Verilator build at a time across lanes; `pgrep -f verilator` first.
**Parent contract:** `docs/design/B1_WALKER.md` (D-028, landed 2026-07-21 —
score+pv @ CQ-8, 24/27 L3 cases walked, 3/27 refused, host mode
byte-identical). B1 was *designed* so that walking the full decoder layer is
a **descriptor extension, not a rewrite** (`B1_WALKER.md` §1 scope note); this
lane cashes that design in.
**Discipline unchanged:** golden is the arbiter, one-directional; no PASS
without pasted output; `rtl/apex_pkg.sv` FROZEN (`APEX_VERSION 0x0001_0000`)
— every new type goes in `rtl/seq/seq_walker_pkg.sv`; `csr_regs.sv`
unmodified; shared files (`scripts/gen_status.py`, `docs/design/LEVEL_C_*`,
`docs/OPTIMIZATION.md`, `verif/top/l3/gen_l3_vectors.py`, `tb_apex_l3.sv`
non-additive paths) are not edited by this lane.

---

## 1. Scope — what "walk the full layer" means

The target op stream is one decode step of the full decoder layer,
`golden/apex_golden/transformer.py` (module docstring graph, `:14-28`):

```
RMSNorm-1 → Q/K/V WS projections (+Qwen bias) → RoPE(q @ m, k_new @ m)
→ store K̂/V into KVQ → per-head attention (score+pv, ×H, GQA H_kv groups)
→ o-proj (+C-2 epilogue) → residual-1 → RMSNorm-2
→ gate/up WS projections → SiLU(gate)·up → down proj (+epilogue) → residual-2
```

at the Qwen2.5-7B geometry (`golden/tests/test_7b_plumbing.py:31-33`):
**D_model=3584, H=28, H_kv=4, head_dim=128, d_ffn=18944, GQA 7:1, QKV
biases, CQ-8**. K-split (K_MAX=2048, `apex_pkg.sv:18`) and N-split
(DIM_W=12 ⇒ n ≤ 4095) per C-KSPLIT (`golden/apex_golden/compute.py`,
`gemm_i8_ksplit` — "host-side, or on-tile via the frozen descriptor's
`accumulate` flag (apex_pkg bit [66])"; K_TOTAL_MAX = 2¹⁷−1 covers 18944).

**Division of labor (who builds what):**

- **IB-WALK (this lane)** owns *sequencing*: the v2 descriptor format, the
  walker FSM extension that emits the full-layer control stream (descriptors,
  routes, job commands, KVQ addressing, composites, DDR fetch *requests*),
  and its verification harness.
- **IB-LAYER** owns the *datapath*: RoPE/SiLU/residual/o-proj path/wide
  RMSNorm integrated into `apex_top`, their route levels + job ports, the
  KVQ store path from the projection outputs, and per-KV-head record
  addressing. The walker's new step templates drive job ports **IB-LAYER
  defines**; until those exist, everything above attention is verified
  against stubs (§4).
- **IB-FUEL** owns the DDR *reader*: `sh_ddr` DDR_PRESENT=1, PCIS bulk load,
  burst reader + async FIFO into the `xw` lane8 stream (`apex_top.sv:173-175`;
  the mux `apex_top.sv:1057-1058` already selects external `xw` when
  `rt_wgt_src=0`). **The DDR image IS the wire format** (host pre-swizzles;
  `LEVEL_C_INTEGRATION.md` §5). IB-WALK **specifies** the walker→reader
  request record and the job-decomposition rule that fixes the image order
  (§2.6 — joint-normative with IB-FUEL), and implements against their
  interface later.

**Carried fences (unchanged from D-028):** CQ-8 only — any other tier is
REFUSED (`WALK_ERR_TIER`), never walked relaxed; T ≤ WALK_T_MAX=128 (the
C-CHUNK T>128 host merge is out of scope); one layer per walk kick — the
host loops the 28 layers; the PV requant pair is host-LOADED (causally
circular on-tile, `B1_WALKER.md` §2 correction 1) — and at layer scope that
limit *grows*, see §2.5, which is named, not hidden.

---

## 2. Descriptor format delta vs D-028

### 2.1 Versioning — how a CL/tile knows which format it speaks

D-028's `walk_desc_unpack` deliberately declared reserved bits
(`seq_walker_pkg.sv:112-115`: `w_geom[31:24]`, `w_geom[19:18]`,
`w_rq[31:21]`, `w_mask[31:2]`) "so the wire format can grow without a
layout change". It also does **not** check them (`walk_desc_check`,
`:132-148`) — so a v2 descriptor loaded into a landed D-028 tile would be
**silently mis-walked as v1**. The version mechanism therefore has two
directions plus one hardening:

1. **Tile→host discovery: `WALK_STATUS[15:12] = FMT_SUP`**, a
   one-bit-per-format support mask. The landed glue reads 0 in those bits
   (`apex_top.sv:426-429` builds the read as `{20'b0, err_code, sticky,
   4'b0, phase, busy}`), so the host decodes `FMT_SUP==0` as "fmt-0 only"
   — backward-compatible discovery with **zero** change to shipped tiles.
   v2 glue reads `4'b0011` (fmt 0 and 1).
2. **Host→tile declaration: `GEOM0[31:28] = fmt`.** `0` = D-028 v1
   (3-word descriptor, score+pv). `1` = the layer format below. Every v1
   descriptor ever emitted has 0 there (`gen_walker_desc.py` writes the
   three words with those bits clear).
3. **Hardening (stage 1, v1.1):** extend `walk_desc_check` to refuse
   `GEOM0[31:28] != 0` with `WALK_ERR_DESC` — additive refusal through the
   existing §A-1 machinery, affects no legal v1 descriptor; closes the
   silent-mis-walk hazard for tiles built between now and v2. Gate: all
   existing `verif/seq_walker` vectors byte-identical + one new refusal
   vector that must fail without the check.
4. **Host rule:** never load fmt=N unless `FMT_SUP[N]` reads 1.

The WALK CSR window keeps its landed addresses **0x5C–0x6C unchanged**
(`WALK_CTRL/DPTR/DDATA/STATUS/RQ`; B3's CSR range stays disjoint per
`B1_WALKER.md` §7; 0x70+ is IB-LAYER's per the stage-4 fence). fmt=1 only
*deepens* the descriptor SRAM behind `WALK_DPTR/WALK_DDATA` (3 → 64 words;
`walk_dptr_q` widens 2 → 6 bits in glue); the per-step patch rides the same
DPTR/DDATA path and `WALK_RQ` keeps its v1 meaning (§2.5).

### 2.2 fmt=1 word map (64×32b descriptor SRAM)

Bit layouts are normative; "resv" bits load as written but MUST be 0
(v2 `walk_desc_check` refuses nonzero resv — no silent growth).

| words | region | layout |
|---|---|---|
| W0 | `GEOM0` | `[31:28]` fmt=1 · `[27:24]` resv · `[23:20]` outlier_k · `[19:18]` resv · `[17:16]` tier (`kvq_tier_e`; != CQ8 ⇒ refused) · `[15:8]` **resv in fmt=1** (t_rows moved to W21, single source) · `[7:0]` head_dim (must equal build CFG_D ∈ {64,128}) |
| W1 | `MODEL0` | `[31:16]` d_ffn (18944 fits) · `[15:0]` d_model (3584) |
| W2 | `MODEL1` | `[31:24]` resv · `[23:16]` H_kv (4) · `[15:8]` H (28) · `[7:2]` resv · `[1:0]` kv_map — per-KV-head record mapping mode (encoding pinned by IB-LAYER, §7 Q2; `2'b00` = flat `base_g = g·2T`) |
| W3 | `MASK` | step-enable bits, `[11:0]`: en_norm1, en_qkv, en_rope, en_storekv, en_score, en_pv, en_oproj, en_res1, en_norm2, en_ffn, en_res2, resv — partial-layer bring-up cases are first-class (staged verification depends on them); all-zero ⇒ `WALK_ERR_DESC` |
| W4–W10 | `WSCALE[7]` | per-tensor weight scales s_wq/k/v/o/g/u/d as fp32 bits — **provisional**: carried only if IB-LAYER's dequant datapath consumes them from the descriptor rather than its own CSRs (§7 Q1); resv-0 until then |
| W11–W20 | `TENS[10]` | DDR base table, one word per tensor, order: Wq, Wk, Wv, Wo, Wg, Wu, Wd, gamma1, gamma2, rope_phase — `[31:30]` resv · `[29:0]` base in **64-byte units** (covers the 64 GB DIMM; every tensor image 64B-aligned by the pre-swizzler). Per-job offsets/lengths are DERIVED (§2.6), not stored |
| W21 | `STEP` | `[31:16]` resv · `[15:8]` t_rows (1..128) · `[7:0]` pos_m — RoPE position of the decode token (self-inclusive q_pos = T−1 per S8 composition; ≤127 in the T≤128 envelope). **Per-step word** |
| W22–W53 | `RQ[32]` | requant table, one word per entry, same packing as D-028's RQ word: `[31:21]` resv · `[20:16]` rq_shift · `[15:0]` rq_scale. Slot assignment: `[0..H-1]` per-head PV epilogue pairs, `[H]` o-proj epilogue, `[H+1]` down-proj epilogue (H ≤ 30 envelope). **Per-step words** |
| W54–W57 | `JC[4]` | stage-5 (IB_LAYER.md §3b JOBC): per-step fp16-GRADE positive-normal f32 composites, one word each — `[0]` deq comp_a after o-proj, `[1]` deq comp_a after down, `[2]` swiglu comp_a (gate), `[3]` swiglu comp_b (up). Host-loaded like the RQ pairs (same causal/precision logic). **Grade = the D-030 CANONICAL narrowing (combine-agenda (a) CLOSED 2026-07-23):** f64 chain → fp32 IEEE-RNE → significand-RNE to 11 bits AT RETAINED fp32 EXPONENT — NOT an fp16 round-trip (this lane's first implementation was exactly that trip and DIVERGED in the sub-2^-14 zone its own tile-scale composites occupy; fixed, conformance example pasted in the stage-5 row). All four slots are members of §3b's definitive graded set (`_proj_epilogue` s_out Wo/Wd; FFN s_h2·s_w{g,u}). **Per-step words** |
| W58–W63 | resv | zero |

`walk_desc_check` v2 (all before any state change, per §3 style): fmt known;
tier == CQ8; head_dim == CFG_D; 1 ≤ t_rows ≤ 128; pos_m < 128;
H·head_dim == d_model; H % H_kv == 0; H ≤ 30; mask != 0; resv bits 0;
en_qkv without en_storekv (and similar dependency violations) ⇒
`WALK_ERR_DESC`.

### 2.3 New step types (walk phases)

`walk_phase_e` grows (WALK_STATUS `[7:1]` has 7 bits allocated, 3 used —
room without a CSR change). Each step type is an FSM template in
`seq_layer_walker`; **score and pv are the landed D-028 engines reused
verbatim, per head** — that is the "extension, not rewrite" claim made
concrete:

| step | template | drives | geometry loops (all pure in the descriptor scalars) |
|---|---|---|---|
| `WPH_NORM1/NORM2` | wide-RMSNorm job | IB-LAYER norm job port + gamma stream fetch (TENS gamma1/2) | d_model = 28·128 chunks (C-RMSW) |
| `WPH_QKV` | WS projection ×3 | route (wgt_src=0 ⇒ external `xw`), MXE DESC per (k-split, n-split), fetch requests to IB-FUEL | Wq: 2 k-splits × 1 n-split; Wk/Wv: 2 × 1 (N=512); accumulate=(k0>0) per C-KSPLIT |
| `WPH_ROPE` | RoPE job on q (per head) and k_new | IB-LAYER rope job port + phase-row fetch (TENS rope_phase + pos_m·row_stride) | H query heads + H_kv new-K rows; V not rotated |
| `WPH_STOREKV` | KVQ store of the new token's K̂/V rows | KVQ AXI-Lite WADDR + commit sequencing (write index = t_rows−1 per group; the D-028 store-snoop scale cache captures the scales as a side effect) | H_kv groups |
| `WPH_SCORE/WPH_PV` | **D-028 engines, unchanged** | as landed (`seq_layer_walker.sv`) | outer head loop h = 0..H−1; KV group g = h·H_kv/H (GQA — pure geometry); per-group record base per W2.kv_map; composite cache indexed per group; per-head RQ slot h |
| `WPH_OPROJ` | WS projection | as WPH_QKV over Wo (2 k-splits), epilogue rq from RQ[H] | |
| `WPH_RES1/RES2` | residual-add job | IB-LAYER residual job port | d_model beats |
| `WPH_FFN` | gate/up WS proj → SiLU·up → down proj | IB-LAYER silu/mult job port + MXE DESCs + fetches | gate/up: 2 k-splits × 5 n-splits each; down: 10 k-splits × 1 n-split; down epilogue rq from RQ[H+1] |

Ordering constraints inherit `B1_STAGE1_NOTES.md` §4 verbatim where the
templates reuse score/pv (the EMIT→DESC→EMIT pv order, route-change-only-
when-idle, the `fq_out_ready` footgun). New templates get their own
ordering table at stage 1, derived the same way (independent re-derivation
against the golden op stream, then diffed).

### 2.4 What is walked vs still not walked

D-028's "data is NOT walked" line splits in v2: **weight data movement is
still not performed by the walker, but its *initiation* is** — the walker
issues (base, beats) fetch requests; the reader moves bytes. Activations
never leave the tile (residual/RMSNorm chain on-tile — IB-LAYER datapath).
The L3 `store_kv`/`q-inject` *injection scaffolding* (87.5% of the stream,
`B1_WALKER.md` §8 row 3) remains TB-only: in a real layer those tensors
come from the projection GEMMs, which v2 *does* walk.

### 2.5 Per-step cost — the honest restated metric

D-028's "3 MMIO/step" was for one attention head with ONE loaded rq pair.
A full layer has **H+2 = 30** causally-circular epilogue pairs per step
(28 per-head PV + o-proj + down-proj — each is `calib_requant(amax|acc|)`
of the very GEMM it configures, `transformer.py` `_proj_epilogue` +
`attention_core`), plus the two step scalars (t_rows, pos_m). None are
computable on-tile in one pass; all are host-loaded, same honest encoding
as D-028.

**Patch path (stage-4 fence ruling): the fmt=1 per-step patch rides the
EXISTING `WALK_DPTR`/`WALK_DDATA` auto-inc window — no dedicated register,
no new address.** `WALK_RQ` (0x6C) keeps its D-028 v1 fast-path meaning
unchanged (write word 1). A steady-state fmt=1 step costs
**1 (DPTR→W21) + 1 (STEP word) + (H+2) (RQ words) + 1 (DPTR→W54) + 4 (JC
words) + 1 (GO) + 1 (STATUS poll) = H+11 = 39 MMIO/layer/step at H=28**
(DERIVED from the register interface, not measured; same caveat class as
D-028's honesty note). *Figure history:* §9.1 R2 ratified "~33 (H+5)"
(stage-0 WALK_RQ-auto-inc variant); the stage-4 fence ruling made it H+6 =
34 (DPTR/DDATA path, ratified); **stage 5 adds the 4-slot JC table (§3b
JOBC composites are per-step calibration exactly like the RQ pairs) + its
DPTR re-point: H+11 = 39 — flagged to the lead for the §9.1 figure
amendment.** Quote **H+11** until ruled.
Host-mode full-layer control at 7B geometry is O(10⁵) ops/layer/step
(per-head drive+poll alone is 4,328 at D=128/T=100 —
`B1_STAGE1_NOTES.md` §6 table — ×28 heads ≈ 1.2·10⁵ before projections),
so the walker still removes ~3.5 orders of magnitude. **The ≤3 figure is
NOT claimed for I-B.**

**Named consequence for the flagship — RATIFIED (§9.1 R2, 2026-07-22):**
host-loaded rq at layer scale means the host runs the golden model
alongside decode (exactly the S8 `run_tinynpu.py` mode) — the I-B claim is
"golden-driven replay on silicon", disclosed at ~33 MMIO/layer/step (H+5,
derived), not autonomous generation. On-tile two-pass amax +
`calib_requant` stays OUT of I-B — a named future exit
(`B1_WALKER.md` §2 correction 1 named it; still not smuggled in here).

### 2.6 Walker→reader request record + job-decomposition rule (joint-normative with IB-FUEL)

New walker port (stage-2 RTL), valid/ready stream into IB-FUEL's reader,
FIFO ≥ 2 deep. **Record FROZEN per §9.1 R1 + IB-FUEL's s1 addendum
(comp/ib-fuel bbe5064/d62b230)** — the stage-0 20-bit beats proposal was
rejected (it cannot address the 67.9 MB down-proj tensor ≈ 1.06M beats):

```
fuel_req[63:0] = { tag[63:56],        // 8 b; value = TENS index ([3:0] used)
                   beats_64B[55:30],  // 26 b, 64-BYTE WORD units
                   base_64B[29:0] }   // 30 b, 64-BYTE WORD units
```

One record per weight-consuming step; a step consuming several tensors
(QKV) issues one per tensor, in template order — **CONFIRMED by IB-FUEL
(their s2, relayed 2026-07-22): tag-as-TENS-index accepted (the reader
carries the tag OPAQUE and echoes it in `FUEL_STAT.last_tag` for audit);
per-TENSOR records are the N-records-in-order form already legal in their
§2.4; they added a 2-deep ingress skid honoring this lane's FIFO ≥ 2
reservation. No layout change — the record below is final.** FUEL-verified
induced invariant (their `make_ddr_image.py` hard-errors on violation):
every committed job's lane-beat count divisible by 8, every row-granular
tensor row a 64 B multiple — holds for every fmt=1 shape (weight-block k is
always a 64-multiple; gamma/phase rows are 64 B multiples at both
head_dims; the mirror's selftest asserts both). SV packed-struct
declaration order is {tag, beats64, base64} so each field lands at the
frozen bit positions (`seq_walker_pkg.sv` `walk2_freq_t`).

**Decomposition rule** (this fixes the DDR image order — the image IS the
wire format, so IB-FUEL's pre-swizzler must mirror it): for each projection
tensor `W[K_total, N_total]`, jobs iterate **n-split-major, k-split-minor**;
job (n0, k0) is `{m, k=min(K_MAX, K−k0·K_MAX), n=min(N_MXE, N−n0·N_MXE),
accumulate=(k0>0)}` and its pre-swizzled weight block appears in the image
in exactly that job order, so `base_j = tensor_base + Σ(prior job beats)` —
sequential within a tensor, zero reformatting in the reader. The bounds and
the per-job beat formula are constants defined ONCE in `seq_walker_pkg` v2
with a Python mirror consumed by both the vector generator and IB-FUEL's
image builder (single source, no drift).

**The two bounds, and why they are not one** (D-029 ERRATUM, I-B — the
landed stage-5 walker used a single constant for both, and that overload was
the defect):

| bound | value | what it limits | why THAT number |
|---|---|---|---|
| `WALK2_N_MXE` | `MXE_N` = **8** | an **MXE job's** `n_dim` | the IMPLEMENTED tile capacity. `mxe_cfg_pkg.sv`: "N ≤ MXE_N (8) — one array width per descriptor, no N tiling v0.1"; `mxe_ctrl.sv`'s `legal` term `(desc.n_dim != '0) && (desc.n_dim <= DIM_W'(MXE_N))` raises `desc_error` and does not accept the job |
| `WALK2_M_MXE` | `M_TILE_MAX` = **64** | an **MXE job's** `m_dim` | same class. The K/V projections emit `m = t_rows`, and `WALK_T_MAX` (128) is twice the implemented M, so `t_rows > 64` with QKV enabled is FENCED at `S2_CHECK` (§A-1: refuse, never degrade) |
| `WALK2_K_JOB` | `K_MAX` = **2048** | an **MXE job's** `k_dim` | the frozen contract limit; the tile reaches it by multi-pass accumulation |
| `WALK2_N_JOB` | `(1<<DIM_W)−1` = **4095** | a **LAYER_JOB's** `cols` FIELD | the field-width bound (IB_LAYER §3b, 0x7C `[11:0]`) — what the wire can CARRY. ~~Unchanged by the erratum~~ — see the row below: it is NOT by itself a legal `cols` |
| LAYER-unit `COLS_MAX` | swiglu **64**, deq **4095**, resid `LAYER_DM_MAX` | a **LAYER_JOB's** `cols` AT THE UNIT | **SECOND ERRATUM (2026-07-30 audit)** — the units stream columns, but each is instantiated with its own width and `apex_top` hands it a SLICE of the 12-bit field: `apex_top.sv:1234` `asu_swiglu #(.COLS_MAX(64))` fed by `apex_top.sv:1237` `.jb_cols(7'(lj_cols_q))`. Chunking `d_ffn`=18944 at 4095 gives `[4095,4095,4095,4095,2564]`: the 4095s arrive as 127 and trip `job_error` (loud), but the tail **2564 arrives as 4 — inside the unit's own legality rule, so it is ACCEPTED and computes 4 of 2564 columns with no error**. Legal `cols` = min(field, unit `COLS_MAX`); `seq_walker_fmt.lu_chunks(cols, unit)` and its `assert_layer_job_legal` mirror carry the split, gated by `mutants5` + `swiglu_cols_check.py`. **RTL still open**: `walk2_lu_chunks` / `seq_layer_walker2.sv:468` chunk at the field for every unit |

The defect: `N_JOB` (the field-width number) sized the MXE n-split, so every
walked 7B projection descriptor — QKV, o-proj and FFN alike, `n_total` =
3584 / 4608 / 18944 → `n_cur` = 4095 — was refused by the tile. Correcting
it does not change the n-major/k-minor ORDER, so IB-FUEL's image layout rule
is unaffected; it changes only the chunk WIDTH, and therefore the job count:
**7B projection descriptors per layer step 38 → 16,000** (Wq/Wo 896 each,
Wk/Wv 128 each, Wg/Wu 4,736 each, Wd 4,480). A correct-but-narrow
decomposition is the I-B position: the ~39-MMIO/step figure is host-side and
untouched, and a descriptor the tile refuses has no throughput at all.

**Why nothing caught it, and what now does.** `verif/seq_walker`'s
scoreboard is a SELF-CONSISTENT oracle — expectations are generated from the
same decomposition constants the walker walks by — so a walker and a spec
that are uniformly wrong agree perfectly and the suite reports PASS. The
W-G2 tile case only walked attention descriptors, which are legal. Closed on
both sides, each reading the CONSUMER's constants and never the walker's:

- **RTL side:** `tb_walker_sb` / `tb_walker2_sb` replay every accepted
  `mxe_desc_t` through `mxe_ctrl`'s own M/K/N rule, with `M_TILE_MAX`
  imported from `mxe_cfg_pkg` and `MXE_N`/`K_MAX` from `apex_pkg`. Gated by
  `m13_nmxe`, a SIGNATURE-REQUIRED mutant (reverts the constant; the run
  must fail *and* print `MXE-ILLEGAL DESC`, since a bare scoreboard
  mismatch is no evidence that the independent check works).
- **Spec side:** `gen_layer_trace.py` refuses to emit a stream containing a
  descriptor the tile would reject. Gated by `make mutants4`, which moves
  the bound in the MIRROR ONLY — reproducing the both-sides-wrong shape the
  scoreboard is structurally blind to — and requires a hard error.
The RoPE phase table is deliberately a *tensor* (TENS[9]): the golden
quantizes phases ONCE from float64 (`transformer.py` C-ROPE — an on-tile
phase generator cannot reproduce `frac(m·θ/2π)` in f64, so bit-exactness
REQUIRES a precomputed table); per step the walker fetches one row at
`base + pos_m · row_stride`.

---

## 3. What this buys / what it does not

Buys: one GO walks an entire decoder layer; per-step host traffic drops to
H+5 writes; the attention inner loop is the already-verified D-028 engine;
every new step type is enable-maskable for staged bring-up.
Does not buy: tier widening (still CQ-8), T>128, multi-layer autonomy,
requant autonomy (§2.5), or any datapath — a v2 descriptor on a tile
without IB-LAYER's blocks refuses (`WALK_ERR_DESC` on en-bits whose job
ports are not built, elaboration-gated).

---

## 4. Staged plan and gates

Machine discipline per `LEVEL_C_INTEGRATION.md` §6 (Verilator on the Mac or
c6a verify box, serialize behind the shared queue; stages 0-1 edit-only).

| stage | what | gate (no PASS without pasted output) |
|---|---|---|
| **0 ✅ (this doc)** | contract + descriptor spec; sanity of the landed baseline | pasted in §8: L3 vectors 27 cases / 135,133 scripted checks; walker descriptors 24 walkable + 3 refusal; composite replica PASS 1,664 words |
| **1 ✅ DONE 2026-07-22** | `seq_walker_pkg` fmt=1 declarations (fmt ids/FMT_SUP, `walk_desc2_t`+unpack/check, word map, `walk_step_e`, decomposition constants, frozen `walk2_freq_t`) + Python mirror `verif/seq_walker/seq_walker_fmt.py` + the v1.1 fmt-refusal hardening (§2.1.3) + `verif/seq_walker/gen_layer_trace.py` — the fmt=1 op-stream spec, golden-gated field-by-field against `decoder_layer_fx` (`gen_l3_vectors.py` untouched). **Evidence (all local, pasted in the stage-1 report):** mirror selftest PASS (2,000 round-trips, 12 check negatives, fuel_req layout pinned); generator PASS (3 cases incl. qwen7b_T8 — epilogue replicas bit-exact, store-scale identity on every head, emitters == B1 §6 formulas); suite: LINT CLEAN, 5,940 emissions preserved (166+2,963+166+2,811 + 2 refusals incl. fmt err_code=2), comp 30,720×2+64×2 zero errors, **mutants 5/5** (new m4_fmtskip caught mis-walking `ROUTE 00ef/SJOB 8`); `make -C golden test` ALL PASS. Deferred to stage 2 BY NAME: dynamic SV-vs-mirror equivalence of `walk_desc2_unpack/check` (the SV functions are lint-elaborated but have no RTL consumer yet — the stage-2 TB gates them against the mirror). | generator self-test PASS ✅; pack/unpack round-trip ✅ (mirror; SV side stage-2); existing `verif/seq_walker` suite counts identical ✅ (refuse set extended 1→2 by design); new fmt-refusal vector kills ✅ (m4) |
| **2 ✅ DONE 2026-07-22 (attention/projection/fetch SUBSET — the §9.1 greenlight scope)** | `rtl/seq/seq_layer_walker2.sv`: fmt=1 step sequencer that INSTANTIATES the unchanged v1 engine per head (per-head v1 descriptor synthesized from fmt=1 state — "extension, not rewrite" made literal), + WS-proj templates (n-major/k-minor, accumulate chains, rq on last k-split), fetch requests at the frozen fuel_req layout, division-free golden GQA mapping (running-remainder `g=floor(h·nk/nh)`), and a new `hd_valid/ready/head` per-head interlock (the s_q staging hand-off; final tie-off with Q1). PENDING-Q1 en-bits + `n_kv_heads>N_ENG` are REFUSED (stage-2 fence, §A-1 pattern). TB `tb_walker2_sb.sv`: N_ENG=4 per-engine comp caches routed by `kv_eng_sel` (sel CHECKED per head + per KVQ write), fetch stub, 9-field DESC capture (accumulate checked). **Evidence:** run2 l64/l128/l7b = 290/461/**6,821** checks, 0 errors (7B: 28 heads, 4 engines, 3,808 KVW); 2 directed stage-2 refusals err_code=2; **mutants2 4/4 killed** (m5_gqamap — head 7 on engine 0 at the non-pow-2 group boundary; m6_accskip — accumulate dropped on k-split ≥1, 7B-only-visible; m7_rqslot; m8_fetchbase; stage-0's planned "fmt-check bypass" landed early as stage-1's m4); v1 runs byte-count-identical through the SAME build (5,940); lint 3 tops clean; golden ALL PASS. **Residuals, named:** PENDING-Q1 steps stay parked for IB-LAYER's port table; the mirror's 12 directed check negatives through the SV `walk_desc2_check` (beyond the refusal paths now exercised) ride with stage 4's glue TB. | walker-vs-layer-trace **bit-exact and in-order** incl. 7B ✅; **≥4 new mutants killed** ✅; all v1 vectors byte-identical through the SAME build ✅ |
| **3 ✅ DONE 2026-07-22** | The composite/scale-plumbing scope was ABSORBED into stage 2's structure and is evidenced there: per-head s_q via the `hd_*` interlock, per-KV-group caches as N_ENG real `seq_walker_comp` instances (routing checked per head + per KVQ write; 7B composites bit-exact at head_dim=128 through real tensors), RQ-table routing mutant-gated (m7), comp sweeps unchanged and m2 still killed in the same `all`. Stage 3's own deliverable — greenlit early by the lead — is the **SV-vs-mirror descriptor-check equivalence harness** (`gen_check_vectors.py` + `tb_check_sb.sv`): the shared directed corpora (`seq_walker_fmt.py` now also mirrors the v1 check incl. the v1.1 fmt clause) + biased-random and uniform tuples, mirror verdict attached per row, replayed through the REAL pkg functions. **Evidence:** `CHECKEQ RESULT: c2=6013 c1=6007 errors=0`; generator coverage gate (every code {NONE,TIER,DESC} produced on both checks: 585/608/4,820 and 1,022/963/4,022); **mutation gate 3: 2/2** (m4_fmtskip caught on C1 fmt rows, new m9_resvskip — resv-blind check2 — caught on the geom-resv rows). This CLOSES the stage-1/2 residual "mirror negatives through the SV check". | comp counts unchanged + m2 killed ✅ (stage-2 `all`); SV-vs-mirror equivalence dynamic over 12,020 rows ✅; harness mutant-gated 2/2 ✅ |
| **4 ✅ DONE 2026-07-22 (pulled forward by the lead, file-fenced)** | `apex_top` glue delta, region (a) ONLY, zero new CSR addresses (0x70+ stays IB-LAYER's): descriptor SRAM 3→64 words + 3-word `walk_desc_v1` alias feeding the UNCHANGED v1 engine, DPTR 2→6 bits (wrap mod 64; the old drop-at-3 guard removed — a state no D-028 flow reached), `WALK_STATUS[15:12] = FMT_SUP` publishing `WALK_FMT_SUP_V1` (v1-only: walker2 is NOT instantiated here — that flips with the IB-LAYER/IB-FUEL integration, together with the mask), fmt=1 patch via DPTR/DDATA per the fence ruling (`WALK_RQ` v1 semantics untouched). New-mode coverage = the `+walkfmt` probe on its own target (`run_walkfmt`, additive `+walker`-pattern TB branch): FMT_SUP read through the REAL CSR bus, deep-DPTR park-write-wrap proven by the refusal landing at the wrapped pointer, tile-level fmt=1 refusal (WALK_ERR_DESC, no walk), W1C, then the case's normal walk bit-exact (1,158 = 1,155 + 3 probe checks). **Evidence:** pristine-tree baseline captured FIRST, then post-edit `diff` — **host-mode `run_all.txt` and walker-mode `run_walker_all.txt` BYTE-IDENTICAL** (27/27 pass; 24/27+3/27 split; re-diffed against the final binary); L3 LINT CLEAN (full apex_top, -Wall, warnings FATAL — no `-Wno-fatal` anywhere in this lane's flows, per the IB-FUEL s3 lesson; propagation-not-readback argued per check in the probe comment); tile mutation gate 3/3. Also found+fixed by the baseline capture: the stage-1 pkg had broken the L3 -Wall build (UNUSEDPARAM on fmt=1 contract declarations) — scoped package waiver per B1's own stated rationale, separate fix commit. | host-mode L3 byte-identical ✅ (diff-proven vs pristine baseline); walker-mode counts unchanged ✅ (diff-proven); new-mode coverage ✅ (`run_walkfmt`); scripted total 135,133 ✅ (vectors log; 135,241 simulated stays the pinned figure, not re-measured) |
| **5 ✅ UNIT-LEVEL DONE 2026-07-22 (unparked on IB-LAYER's frozen §3b; W-G2 tile gate = combine)** | The §3b table transcribed into the walker: `seq_layer_walker2` rebuilt as a micro-step program sequencer (29-pc §2.3 template; disabled steps skipped, stage-2 PENDING fence retired) with the FULL step set — NORM1/2 via the existing `lj_*` port; ROPE = LCTL level-arm (`{rope_en, bank=0, pos=pos_m}` — the phase-K table is per-config RESIDENT per §3b memories, so NO per-step row fetch); STOREKV = per-engine WRITE_ADDR (0x28) sequencing, idle-polled, K@T−1/V@2T−1; deq/swiglu/residual via the §3b drive nets (LCTL levels incl. `ser_dst`/`fsrc_ext=2` for the swiglu-p→down path, JOBC alternating composites REWRITTEN per chunk per the index-reset rule, LAYER_JOB pushes with 12-bit col chunking at 7B d_ffn). JC values ride descriptor slots W54–W57 (golden-gated: o-proj/down `s_out` from the replica, gate/up from the s_h2 replica; grade = fp16-RNE). **Registered-accept everywhere** (§7 Q1 note): the TB's LAYER-unit stub models the combinational drop-at-accepting-edge ready. Both executor lessons recorded for the TILE executor (1-cycle window reads; NEGEDGE CSR stimulus — the unit TB's drive tasks already run on negedge). **Evidence:** run2 l64/l128/**l7b = 309/480/6,858 checks, 0 errors** (7B: every §3b step + 28 heads + 4 engines + 3,816 KVQ writes incl. 8 store WADDR); refuse2 2/2; **mutation gate 2 now 7/7** (new: m10_jcorder — comp_b re-emitting gate, killable because the recipe differentiates s_wu; m11_ropepos — LCTL pos+1; m12_waddr — store writes RADDR, caught as KVW-vs-KVWA); v1 5,940 + comp + CHECKEQ + golden all green in the same `all`. **Combine-owned remainder (NOT this lane):** walker2 `apex_top` instance + FMT_SUP→0b0011 flip (together, as contracted) against LAYER's landed glue + FUEL's physical reader; the W-G2 tile gate vs their L4 harness. **Reconcile flags — ALL THREE RESOLVED (2026-07-23):** **(a) CLOSED — this lane CONFORMS to the D-030 canonical grade** (IB_LAYER §3b @ d4f9563; the fp16-trip implementation was replaced; conformance evidence: `grade(1.811241e-6) = 0x35f32000 = 1.8114224076e-6 @ e8=107` matching their published value exactly, the fp16 trip of that same graded value returning `1.7881393433e-6` ≠ graded, sub-2^-24 `4.87e-8 → 4.869e-8` canonical vs `5.96e-8` trip; idempotence + `&0x1FFF==0` + `0<e8<255` over the emitted set + 2,000 randoms in the mirror selftest; vectors regenerated, full suite + 7/7 mutants re-green). **(b) RESOLVED — L4 arbitrates on the merged tree; LAYER's canonical choreography CONFIRMS this lane's arm-before-stream and JC-before-push and EXTENDS with three combine expectations binding on the walker's tile integration:** (i) consumer-before-producer push in chained flows — the residual JOB pushes BEFORE the deq JOBC+JOB (this lane's unit order pushes deq first: a one-pc swap at the combine, named here); (ii) at most ONE pending push with a STATUS poll before the next (glue raises JOB err on push-while-pending); (iii) 1-cycle readback sampling on the LAYER window (late sample reads 0xDEADBEEF). **(c) RESOLVED in this lane's favor — phase table RESIDENT, zero steady-state phase loads; LAYER's stage-7 inventory line struck.** | walker-vs-layer-trace bit-exact incl. every §3b step at 7B ✅; ≥2 new mutants ✅ (3); v1/comp/CHECKEQ byte-identical ✅; W-G2 tile gate → combine |
| **6 — DOC HALF ✅ DONE 2026-07-22 (W-G3 replay = COMBINE-owned, per the lead's ruling: do not attempt in-lane)** | **D-029 ARCHITECTURE entry LANDED** (appended after D-028's row, the B1 stage-6 adjacent-line protocol) with all four approved kill-signature citations (m4_fmtskip's silent mis-walk, m5_gqamap's non-pow-2 boundary miss, m10_jcorder's gate·gate swap, m12_waddr's read-launching store). STATUS/OPTIMIZATION rows stay paste-ready-at-combine (the IB-LAYER stage-7 pattern; `gen_status.py` and `OPTIMIZATION.md` are shared files this lane does not edit). **W-G3 (`layer_walk_golden` end-to-end replay: traced real Qwen2.5-7B layer step, pre-swizzled DDR image, loader→DDR→reader→walker→tile) waits for the merged tree + FUEL's physical reader**; its definition stands below unchanged. | D-029 entry committed ✅; W-G3: `layer_walk_golden` replayed with 0 fails, log under `docs/results/` — at combine |

| **ERRATUM (I-B) ✅ DONE 2026-07-26 — D-029 F-1: the MXE n-split was sized by the descriptor FIELD** | `WALK2_N_JOB` was ONE constant serving two consumers with different limits, so every walked 7B projection descriptor (`n_cur` = 4095) was `desc_error`-refused by `mxe_ctrl`; QKV, o-proj and FFN alike. Split into `WALK2_N_MXE` (= `MXE_N` = 8, MXE job n) and `WALK2_N_JOB` (= 4095, LAYER_JOB cols — unchanged), plus `WALK2_M_MXE` (= `M_TILE_MAX` = 64) fencing the second instance of the same class: the K/V projections emit `m = t_rows` and `WALK_T_MAX` is 128, so `t_rows > 64` with QKV enabled is now REFUSED at `S2_CHECK` rather than walked into a desc_error. `seq_walker_pkg` imports `M_TILE_MAX` from `mxe_cfg_pkg` (not a restated copy). Mirror `seq_walker_fmt.py` carries the same split. **The blind spot mattered more than the constant** and is closed on both sides (§2.6): the TBs replay every accepted descriptor through `mxe_ctrl`'s own rule reading `mxe_cfg_pkg`/`apex_pkg` symbols, and the generator refuses to bless an illegal stream. **Evidence:** suite green with the new counts — l64/l128/l7b = **430/639/22,820** checks (was 309/480/6,858), `mxelegal` = 146/184/16,476 descriptors legality-checked, 3 refusals (new: t_rows=65); mutation gate 2 **8/8** with `m13_nmxe` SIGNATURE-REQUIRED, new gate 4 **1/1** (spec-side); v1 5,940 + comp + CHECKEQ unchanged; L3 `run_all.txt` and `run_walker_all.txt` **byte-identical** vs a pristine-baseline worktree, walkfmt/walkfmt2 identical but for Verilator's wall-clock footer; l4 levels/compose identical; golden ALL PASS. | 7B projection descriptors ACCEPTABLE by `mxe_ctrl` ✅; emission counts change as predicted (38 → 16,000 proj DESCs/step) and are explained ✅; independent legality check mutant-gated on BOTH sides ✅; every untouched path byte-identical ✅ |

**The layer-walk golden replay gate, defined exactly (three rungs):**

- **W-G1 (unit, this lane, stage 2):** the walker's emitted control stream
  is bit-exact and in-order vs the canonical trace extracted from the
  stage-1 layer generator — which is itself golden-gated: every tensor,
  scale, rq pair and tap in its case files is asserted equal to the
  corresponding `decoder_layer_fx` field at generation time.
- **W-G2 (tile, stage 5, joint):** those same case files replayed through
  the IB-LAYER-integrated `apex_top` in BOTH modes; walker mode bit-exact
  with unchanged check counts (the L3 §B-1 rule lifted to the layer).
- **W-G3 (flagship, stage 6):** `layer_walk_golden` = one committed decode
  step of one real Qwen2.5-7B layer traced from `run_tinynpu.py` (extend
  the S8 tracer to dump a full-layer record: X row, all seven weight
  tensors + gammas + phase rows, per-tensor scales, the H+2 rq pairs, and
  the golden stage taps incl. r2), pre-swizzled into a DDR image, replayed
  loader→DDR→reader→walker→tile in the extended f2sim — weights genuinely
  from (modeled, then real) DDR. This is the artifact the §9 gate names;
  its silicon run inherits the I-A bring-up truths verbatim
  (`AWS_CLK_GEN` naming, clkgen recipe reload after AFI load, TILE_RST
  parity — `docs/results/f2_stage2_hw/RESULT.md`).

---

## 5. B1b and "prefetch overlap" — explicit disposition

**Naming drift — flagged at stage 0, ACKNOWLEDGED in §9.1 R2 (the lead's
own note) with this lane's both-readings-deferred handling ratified:** the
§9 lane row says "fold B1b prefetch-overlap in or defer it". `B1_WALKER.md`
§5 and `docs/OPTIMIZATION.md:63` define **B1b = the commit-time
feeder-equivalent amax pass that lets grouped tiers (CQ-4/CQ-4+) walk in
strict order** — nothing to do with prefetch. No other definition of a B1b
prefetch exists in the tree (grep over docs/rtl/verif: the only other
`prefetch*` hit is the unrelated PCI "prefetchable BAR" note,
`LEVEL_C_INTEGRATION.md:144`).

- **B1b proper (grouped-tier amax): DEFERRED.** The CQ-8-only fence and its
  refusal gate carry into fmt=1 unchanged. Rationale: the S8/7B artifact
  path — the only real-model evidence chain — is CQ-8; tier widening is
  orthogonal to layer-walking and pairs with the S12/D-027 KVQ-4 track.
- **Prefetch overlap (fetch job j+1 during compute of job j): DEFERRED as
  behavior, absorbed as interface.** The request record + ≥2-deep FIFO
  (§2.6) make issue-ahead possible with a later FSM-only change — no
  descriptor-format change will be needed. Not exploited in v2 because
  (a) at demo clocks the fuel line is capacity-bound, not bandwidth-bound
  (`LEVEL_C_INTEGRATION.md` §5: tile consumes ≤8 B/cyc ≈ 125 MB/s, ~150×
  DDR4 headroom), so fetch time per job sits 1–2 orders below compute
  time and overlap is not load-bearing for any I-B claim; (b) no perf
  reordering while the bit-exact gates are being established.

---

## 6. NOT in scope (owned elsewhere or explicitly future)

- **`apex_top` datapath blocks** — RoPE/SiLU/residual/o-proj path, wide
  RMSNorm elaboration (RMS_D_MAX→3584), their route levels/CSR map, KVQ
  store path from projections, per-KV-head banking → **IB-LAYER**.
- **DDR reader and everything behind it** — `sh_ddr` DDR_PRESENT=1, XCI,
  PCIS/BAR4 decode, burst reader, async FIFO, `sh_ddr.stub.sv` sparse
  model, host loader/pre-swizzler → **IB-FUEL**. This contract SPECIFIES
  the request record + image-order rule (§2.6); implementation lands
  against IB-FUEL's reader interface later.
- **On-tile `calib_requant` (two-pass amax)** — the requant-autonomy exit
  (§2.5); its own future contract.
- **Grouped tiers (B1b), KVQ-4 quality** — S12/D-027 track.
- **T > 128 (C-CHUNK merge) and multi-layer/model-level autonomy** — host
  loops layers and merges chunks, as in the S8 pipeline.
- **`apex_pkg.sv`** — frozen; `mxe_desc_t` untouched (v2 emits the same
  128-bit descriptors); no version bump.
- **Shared files** — `gen_l3_vectors.py`, `tb_apex_l3.sv` (beyond the
  established additive `+walker` pattern), `scripts/gen_status.py`,
  `LEVEL_C_*.md`, `OPTIMIZATION.md` (stage-6 rows excepted, per the
  adjacent-line-merge protocol).

---

## 7. Dependencies and open questions for the integration lead

Consumed from **IB-LAYER** (their contract should pin; walker ROM cannot be
written until then):

1. **Q1 — job-port shapes + route levels** for norm/rope/silu/residual/
   store-path blocks, published as a table IB-WALK transcribes verbatim
   (like `B1_STAGE1_NOTES.md` §1 did for score/pv). Also: are the
   per-tensor weight scales descriptor-fed (W4–W10) or IB-LAYER-CSR-fed?
   And the k-split realization: OS-mode projections with `accumulate=1`
   (C-KSPLIT's stated on-tile path) or a WS-mode accumulate extension?
   **STILL OPEN after §9.1 — IB-LAYER publishes with their S4 (greenlit;
   lead forwards on landing). Stage 1 was sequenced to need none of it:
   the generator emits typed `PENDING-Q1` markers for those steps
   (geometry final, port args pending) and encodes projection DESCs as
   OS+accumulate with the provisional-encoding note in its header;
   regenerate on their ruling.**

   **Stage-5 design note — REGISTERED-ACCEPT against LAYER job ports
   (lesson relayed by the lead from IB-LAYER S2/S3, 2026-07-22):** their
   new job-port units expose COMBINATIONAL job-ready that DROPS at the
   accepting edge — post-edge polling deadlocks. Rule for the stage-5
   walker templates and the `hd_*`/job interlocks against their units:
   present-and-hold valid, and capture `ready` in the SAME `always_ff`
   edge that transitions (the D-028/walker2 FSM idiom — `if (xx_ready)
   state <= …` while valid is held — already does this); NEVER split
   present/accept across states or re-check ready in a later state; where
   a boundary can't follow the idiom, front their port with a
   `stream_skid` (skid-fronted streams don't have the issue). Audit every
   NEW stage-5 wait state against this rule before the first build.
2. **Q2 — per-KV-head KVQ record addressing — §9.1 R3 PROVISIONAL, stage-1
   confirmation done with ONE escalation.** R3: per-KV-head mapping onto
   D-024-style banked engines, flat offset within engine, DEPTH=256 = 2·128
   fits T≤128 per engine. Stage-1 findings (2026-07-22): (a) per-engine
   sizing HOLDS — 2T ≤ 256 with zero headroom at T=128 (generator asserts
   it per case; DEPTH=256 is the verified L3/f2 build config, `-GDEPTH=256`
   / `CL_KVQ_DEPTH=256` — the `apex_top` RTL default is 128); (b)
   **ESCALATED: R3's formula `h >> log2(H/H_kv)` assumes a pow-2 group and
   is inapplicable at the 7B geometry (28/4 = 7)** — this spec uses the
   golden mapping `h // (H//H_kv)` (`transformer.py` GQA slicing), same
   intent (engine = KV-head index), correct at every geometry; (c) engine
   COUNT: today's `apex_kvq_bank` has THREE per-TIER engines (D-024,
   `apex_top.sv:42`) — per-KV-head banking needs H_kv CQ-8-tier instances
   (4 at 7B), an IB-LAYER build-out; flat-in-one-engine does NOT fit
   (4·2T = 1024 > 256 at T=128). `kv_map=2'b01` encodes the R3 mode
   provisionally; final encoding stays IB-LAYER's.
3. **Q3 — CLOSED (§9.1 R2):** framing ratified as "golden-driven replay on
   silicon" (§2.5); two-pass amax stays out of I-B.

Coordinated with **IB-FUEL — FROZEN and CONFIRMED (§9.1 R1 + their s1
addendum + their s2 confirmation of the per-TENSOR/tag reading, §2.6):**
the 64-bit `fuel_req` layout, 64 B-word units, 8-bit opaque tag with
`FUEL_STAT.last_tag` audit echo, 2-deep ingress skid on their side; their
pre-swizzler and this lane's mirror share the decomposition rule — their
`make_ddr_image.py` enforces the divisibility invariant. Coordinated at
**integration — CLOSED:** D-029 confirmed (D-030 = IB-LAYER's bus-
composition mode); the §9 B1b label acknowledged (§5). Still open at
integration: W-G2 harness ownership split (IB-LAYER builds the layer
harness, IB-WALK the walker-mode path in it).

---

## 8. Stage-0 baseline — re-verified this session (pasted)

Fresh worktree `comp/ib-walk` @ 335dea0, commands from `B1_WALKER.md` §6
(the ~2 s no-Verilator set; the full `verif/seq_walker` suite builds
Verilator binaries and was NOT run — nothing beyond these three is claimed):

```
$ make -C verif/top/l3 vectors        # 1.3 s
L3 cases: 27 total (23 b64, 4 b128), 135133 scripted checks

$ python3 verif/top/l3/extract_trace.py verif/top/l3/build   # tail
  ... a store-time scale cache lets the walker emit in the host's EXACT
  order — no reorder allowance needed.

$ python3 verif/top/l3/gen_walker_desc.py verif/top/l3/build
walker descriptors: 27 cases (24 walkable CQ-8, 3 refusal)

$ python3 verif/top/l3/walker_composite_golden.py verif/top/l3/build  # 0.7 s
COMPOSITE REPLICA: PASS (1664 composite words bit-exact vs the L3 op
stream; 32 structurally tap-unresolvable folds excluded)
```

The simulated-check figure 135,241 and the walker-mode 24/27+3/27 L3 run
are the landed D-028 evidence (`docs/OPTIMIZATION.md:63`, ARCHITECTURE
D-028 row) — quoted, not re-measured here (`l3_rerun.log` is stale; do not
cite it).
