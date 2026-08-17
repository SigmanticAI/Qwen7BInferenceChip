# IB-LAYER — Full Decoder-Layer Ops into apex_top (RoPE / SwiGLU / o-proj / residual + wide RMSNorm)

**Status:** stage 0 (this contract) — **stage ledger in §4 is the live
status: S1-S4/S4b LANDED, S5 partially landed, S6 (wide geometry) NOT RUN.**
**[Reconcile note 2026-08-05: this lane's original premise — that the
full-row ops need the wide image — was dissolved on 2026-07-31/08-01 by the
N-lane (elementwise-sliced residual + R4 chunked RMSNorm-2, all 6 op types
on the FPGA with no wide image; `docs/results/prompt_on_chip/
SIX_OF_SIX_RESULT.md`). The wide feeder / single-pass full-row elaborations
(stage 6) remain real and remain gated on aws-fpga#799; nothing in this
contract's stage-6 scope has been built or claimed.]**
**Branch:** `comp/ib-layer` off
`comp/level-c-integration` @ 335dea0. **Contract frame:**
`docs/design/LEVEL_C_INTEGRATION.md` §9 (IB-LAYER lane row) — tranche I-B.
**Golden arbiter:** `golden/apex_golden/transformer.py` (`decoder_layer_fx`
and the C-ROPE/C-SILU/C-6 contracts in its module docstring) — golden is THE
arbiter, one-directional; never edit an existing golden function to match RTL.
**Contract discipline:** `rtl/apex_pkg.sv` stays FROZEN (`APEX_VERSION
0x0001_0000`, `apex_pkg.sv:14`) — route-level + CSR additions only, the
D-024/B3 pattern (`docs/design/B3_WEIGHT_PATH.md` §1/§3 "route/CSR, NOT a
descriptor flag") as executed by B1 (glue CSR window, `apex_top.sv:367-431`).
`rtl/csr/csr_regs.sv` is instantiated UNMODIFIED; new addresses use its
reserved fall-through (`csr_regs.sv:178` reads `0xDEAD_BEEF`, write no-op,
ready acked).
**Anti-fabrication:** every claim below marked *verified* names the command
that produced it. Stage 0 ran greps/reads only — **no simulation, synthesis,
eval, or hardware run happened in this stage**, and none is claimed.

---

## 0.1 Stage-0 reconcile (LEVEL_C §9.1, 2026-07-22) — rulings binding on this lane

Where this doc conflicts with `LEVEL_C_INTEGRATION.md` §9.1, **§9.1 wins**.
The rulings that touch IB-LAYER:

- **R4 (was §8 Q1) — RESOLVED:** the additive golden bus-composition mode is
  design-approved as **ARCHITECTURE decision D-030** (flags default-OFF,
  banner byte-identical). **OWNER SIGN-OFF is the S1 gate** and is PENDING —
  the mode is landed and measured (`docs/results/ib_layer_bus/RESULTS.md`),
  but **no RTL stage (S2+) may gate against `BUS_ON` until the sign-off is
  recorded here.** **Sign-off record: OWNER SIGNED 2026-07-23 — recorded in
  `LEVEL_C_INTEGRATION.md` §9.1 (integration branch). D-030/C-LBUS is the
  L4 arbiter; S5 gates against `BUS_ON`. The fence is LIFTED.**
- **R5 (was §8 Q2) — RESOLVED:** glue-CSR-window control CONFIRMED, zero new
  `apex_top` ports. Mux point = the B1 stage-5 idiom: ONE glue-level
  host/walker mode mux; the IB-WALK walker ROM later drives the SAME glue
  registers the host writes (single ingress point, L3-pattern tap mirror).
- **R6 (was §8 Q3) — RESOLVED, probe EXECUTED:** the wide
  `seam_feeder_quant` yosys probe was pulled forward and is DONE — results
  in §1b below. Chunked-amax feeder stays speced as the fallback (§8 Q3)
  but the area evidence says it is not needed.
- **Cross-lane (R1/R2):** IB-WALK's fmt=1 layer descriptor is **D-029** and
  carries one `{base_64B[29:0], beats_64B[25:0], tag}` fuel record per
  weight-consuming phase. **IB-WALK is blocked on this lane's job-port/route
  table for its ROM** — publishing that table in this doc as soon as the S4
  glue shape firms is a named stage-4 deliverable (§4), not a courtesy.
- **Machine rule (all lanes, stage 1+):** at most ONE Verilator build at a
  time across the three lanes — `pgrep -f verilator` before launching, wait
  if busy; Mac model evals outrank EDA (§6 of the hub contract).
- **KVQ per-KV-head banking (R3 amended, IB-WALK stage-1 finding — NEW S4
  scope):** GQA at H_kv=4/T=128 cannot fit the flat storage (4·2T = 1024 >
  256); the CL/apex_top glue needs **H_kv CQ-8 engine instances** with the
  golden mapping `h // (H/H_kv)` — and H/H_kv = 7 is NOT a power of two, so
  no shift tricks anywhere. Pricing: KVQ store area multiplies ×H_kv in the
  wide build (the §1b feeder probe did NOT cover this side). **Assessment
  (no escalation):** the I-B target is the VU47P Small-Shell CL — 4 CQ-8
  engines at DEPTH=256 are small against that part (the entire S10a ECP5
  ship tile incl. 3 engines is ~19.4k LUT4); a KVQ-engine synth probe joins
  the §4 stage-6 probe list so the ×H_kv figure lands MEASURED, not argued.
- **UNDRIVEN-output lesson (IB-FUEL s3 cross-lane):** CSR readback of an
  internal register is NOT evidence the level left the module — FUEL
  shipped undriven output ports behind green BAR0 readback (Verilator's
  UNDRIVEN warning swallowed by -Wno-fatal). Binding on stage 4: lint the
  glue with warnings FATAL (this lane's suites already run plain `-Wall`
  with no -Wno-fatal anywhere), and the L4 TB gains a directed
  LEVEL-PROPAGATION probe per exported route level (drive via CSR, observe
  the level at its CONSUMER, not at readback).

- **Ranking ruling (2026-07-24, dependency-graph): S5 (b)-(d) FIRST, S4b
  immediately after.** tb_l4_compose is W-G2's arbiter (combine flip being
  built at N_ENG=1 on `comp/ib-combine`; W-G2 runs the moment S5 lands);
  S4b gates only W-G3's 28-head geometry (on S4b landing, the flip bumps
  N_ENG 1→4 and W-G3 runs). S5 slice `6585c7d` is already merged into the
  integration tree; (b)-(d) build on this tip.
- **S4b plan — APPROVED VERBATIM (2026-07-24):** sibling
  `apex_kvq_gqa_bank` copying the D-024 `apex_kvq_bank` pattern (the tier
  bank stays untouched/verified); **N_ENG=4 CQ-8 `kvq_engine` instances at
  DEPTH=256 behind `kv_eng_sel[1:0]`** — walker-side `kv_eng_sel` in walk
  mode, host-side via the reserved `LAYER_CTRL[17:15]` `l_kv_map` carve;
  the `h // (H/H_kv)` mapping stays SEQUENCER-side (walker2), never in the
  bank; parameter-gated to today's build (byte-identity, the S4
  discipline); **×H_kv synth probe included** (engine cost measured, §1b
  flow) + bank-level TB (kvq suite patterns) + byte-identity gates.
- **S4b LANDED (2026-07-26) — this branch, staged commits (bank / glue /
  suite / probe).** `rtl/top/glue/apex_kvq_gqa_bank.sv` (D-024 sibling;
  eng_sel quasi-static routing contract; the mapping stays sequencer-side
  by construction — the bank consumes a ready engine index). apex_top gains
  **`KVQ_GQA_NENG`** (default 1 = today's build BYTE-IDENTICAL): the KVQ
  subsystem position is now ONE generate — `==1` elaborates the untouched
  tier-bank instantiation verbatim, `>1` elaborates the GQA bank with
  engine select = walk mode walker2 `kv_eng_sel` / host mode the §3b
  `l_kv_map` carve (single ingress per R5), and INFO_TIER drops the
  CQ-4/CQ-4+ bits (D-027 never-lies; OFF folds to the pre-S4b constant).
  **Gates:** bank suite `verif/kvq/gqa` — 1,859 checks fails=0 (pin derived
  then confirmed), 3/3 signature-required mutants, per-engine §5 SVA bound;
  apex_top -Wall lint CLEAN in OFF (l3+smoke configs) AND ON
  (`KVQ_GQA_NENG=4`/`KVQ_DEPTH=256`); ×H_kv probe EXECUTED (table in
  `verif/kvq/gqa/RESULT.md`): US+ 31→124 RAMB18E2, 1→4 DSP48E2,
  10,211→41,147 LUT2-6, 9,150→36,600 FF — engine cost ×4 with ~0.3k-LUT
  select fabric and zero extra BRAM/DSP; ECP5 30→120 DP16KD (yosys mapping
  evidence only, timing build-report-only); byte-identity with the
  parameter OFF vs stashed pre-edit baselines — golden banners, l3 host AND
  walker AND walkfmt, l4 levels+compose, kvq mask suite (diff evidence in
  the S4b commits). **Integration flip (the N_ENG 1→4 bump, walker2
  instantiation side — combine-owned):** (1) set `KVQ_GQA_NENG=4` at the
  build point (with `KVQ_DEPTH=256`, the R3 sizing); (2) `u_walk`:
  `.N_ENG(1)` → `.N_ENG(KVQ_GQA_NENG)`; (3) widen the `wk_kv_eng_sel`
  declaration `logic [0:0]` →
  `logic [(KVQ_GQA_NENG > 1 ? $clog2(KVQ_GQA_NENG) : 1)-1:0]` (defaults
  byte-identical; the S4b mux width-casts `wk_kv_eng_sel`, so either flip
  order elaborates). Host-mode contract: write `LAYER_CTRL[17:15]` = KV-head
  index while the KVQ subsystem is quiescent; in GQA builds the tier CSR
  machinery is inert and INFO_TIER reads 3'b001.

**S1 status (2026-07-22): LANDED, gate evidence pasted in
`docs/results/ib_layer_bus/RESULTS.md` §3** — `make -C golden test` rc=0,
main banner byte-identical, `layerbus` gate green (OFF bit-identity, 2,008
exact-Fraction realizability checks, deltas pinned at rtol 1e-6). The L4
generator skeleton runs (6 cases, 41,802 checkpoint words, determinism
self-check; op-stream emission is an S4 stub by design). S1 closes when the
owner sign-off lands in the R4 record above.

---

## 1. What the lane does & where it sits

Today `apex_top` realizes the **attention decode step** (host- or
walker-sequenced; D-028). The rest of the decoder layer exists as:

- **standalone, composition-verified blocks** — `rtl/rope/rope.sv`,
  `rtl/asu/asu_silu.sv`, `rtl/misc/residual_add.sv`; PASS on real
  `decoder_layer_fx` stage tensors in `verif/layer/RESULT.md` (Verilator AND
  Icarus), but **TB-side composition only** — none is instantiated in
  `apex_top` (B1_WALKER.md §1 "Scope reality" note);
- **golden/host-side steps** — o-proj and FFN GEMM prep, the C-1 quant
  boundaries between sublayers, and both RMSNorm sites at hidden D
  (`apex_top`'s `u_rms` is elaborated `RMS_D_MAX=128`, `apex_top.sv:634`).

IB-LAYER integrates RoPE, SwiGLU/SiLU, o-proj prep, and the two residual adds
into `apex_top` behind route-level + CSR selection, and elaborates the wide
RMSNorm (`RMS_D_MAX→3584`), so a full decoder-layer decode step can execute
tile-side under **host sequencing**. Walking that step autonomously is
IB-WALK's lane; feeding weights from DDR is IB-FUEL's.

### Stage-0 reality checks (all file-level; command per claim)

1. **`rope.sv` and `residual_add.sv` datapaths are SV `real` (float64) —
   simulation-exact, NOT synthesizable.** Verified: `grep -n "real cr\|real
   ar" rtl/rope/rope.sv rtl/misc/residual_add.sv` → `rope.sv:193` (`real cr,
   sr, xr_i, xr_ih, lo, hi`), `residual_add.sv:136` (`real ar, br, sr`).
   `asu_silu.sv` is integer-only (its single `real` hit is a comment, `:50`).
   The house already crossed this bridge once: `rtl/kvq/cores/cq_fp_pkg.sv:5`
   — "This file replaces the earlier `real` (float64)" — and
   `rtl/seq/seq_walker_comp.sv` proves fp16→f32 arithmetic in pure integer
   RTL with significand-space proofs (its P1–P6). **Consequence: synthesizable
   integer re-implementations of the RoPE rotation MAC and the fp16 residual
   add are IN this lane's scope (stage 2)** — otherwise the I-B CL build
   cannot elaborate the layer.
2. **The CL source list contains none of the three blocks.** Verified: `grep
   -n "rope\|silu\|residual" scripts/fpga/f2/cl_apex/design/apex_sources.f`
   → only `kvq/cores/residual_buffer.sv` (unrelated). Adding the new files to
   `apex_sources.f` is stage-4 work.
3. **CSR window 0x70+ is free.** `csr_regs` decodes ≤ `0x54`, `0x58` is
   ERR_STICKY glue (`apex_top.sv:1160`), `0x5C/0x60/0x64/0x68/0x6C` are the
   WALK window (`rtl/seq/seq_walker_pkg.sv:45-54`). Verified no other decode:
   `grep -rn "8'h7\|8'h8\|8'h9" rtl/top/apex_top.sv rtl/seq/seq_walker_pkg.sv
   rtl/csr/csr_regs.sv` → no address hits.
4. **Whole-row C-1 quant at hidden D needs a wide feeder, not just wide
   RMSNorm.** `seam_feeder_quant` is a two-pass row-buffering quantizer
   (pass 1 buffer row + amax, pass 2 quantize; `seam_feeder_quant.sv:11-13,
   242-245`) with row length `D` an elaboration parameter (`:67`).
   `quant_rows_i8`'s per-row amax at D_model=3584 therefore requires a
   D=3584 feeder elaboration (row buffer ≈ 3584×32 b ≈ 115 kbit) — a synth
   probe obligation this lane owns (stage 6). The LEVEL_C §9 row prices only
   the RMSNorm elaboration; this is additional and was not previously priced.
   **→ PROBED 2026-07-22 under ruling R6 — the wide feeder is cheap; see
   §1b. Measured delta vs the D=128 CL build point (US+ abc9): +6 RAMB18E2,
   +1 DSP48E2, ~+0.1k LUT; the row buffer infers block RAM as-is.**
5. **The fp16-bus composition question is pre-existing and documented, not
   new.** `verif/layer/gen_layer_vectors.py:21-27` narrows RoPE inputs to
   fp16 and checks the residual primitive `f16(f16(a)+f16(b))` — explicitly
   NOT `decoder_layer_fx.r1`, which adds pre-fp16 float64 reals (≤1-ULP
   double-rounding class delta, documented there). §2b makes this a named
   contract decision instead of a per-TB footnote.
6. **Wide-RMSNorm cost is measured, timing is not.** `verif/asu/wide/
   RESULT.md` synth probe: +1 RAMB36E2, +5 DSP48E2, ~+1.2k LUT on US+ at
   `RMS_D_MAX=3584`; **the 45-bit μ-multiply path timing must be READ from
   the I-B build report — never assumed** (RESULT.md scope note; LEVEL_C §3).

### 1b. R6 probe — wide `seam_feeder_quant` synth (EXECUTED 2026-07-22)

House sv2v→yosys flow (yosys 0.66, the wide-rms probe's toolchain), wrapper
`feeder_wide_synth` = `seam_feeder_quant #(.D(…), .ROWS_MAX(16))` + apex_pkg
+ `stream_skid`, three elaborations through identical commands
(`sv2v … > flat.v` then `yosys -p "read_verilog flat.v; hierarchy -top
feeder_wide_synth; synth_xilinx -family xcup -abc9; stat"`; ECP5 via
`synth_ecp5 -abc9`). Verbatim mapping lines:

```
mapping memory ...seam_feeder_quant.rbuf via $__XILINX_BLOCKRAM_SDP_   (D=3584, xcup)
mapping memory ...seam_feeder_quant.rbuf via $__XILINX_BLOCKRAM_SDP_   (D=128,  xcup)
mapping memory ...seam_feeder_quant.rbuf via $__XILINX_LUTRAM_SDP_     (D=64,   xcup)
mapping memory ...seam_feeder_quant.rbuf via $__PDPW16KD_              (D=3584 and D=64, ecp5)
```

| US+ (`synth_xilinx -family xcup -abc9`), CHECK 0 problems each | D=3584 | D=128 (CL build point) | D=64 |
|---|---|---|---|
| RAMB18E2 | **7** | 1 | 0 (5 RAM64M8 LUTRAM) |
| DSP48E2 | **1** | 0 | 0 |
| LUT2-6 total | 1,417 | 1,310 | 1,259 |
| FDRE / CARRY4 | 502 / 338 | 478 / 335 | 449 / 331 |

| ECP5 (`synth_ecp5 -abc9`) | D=3584 | D=64 |
|---|---|---|
| DP16KD / MULT18X18D | **7 / 1** | 1 / 0 |
| LUT4 / TRELLIS_FF | 1,532 / 500 | 1,243 / 474 |

**Verdict:** the wide feeder elaboration is CHEAP — **+6 RAMB18E2,
+1 DSP48E2, ~+0.1k LUT vs the D=128 CL build point** (+7/+1/+0.16k vs
D=64); the comb-read `rbuf` infers block RAM as-is on both targets, exactly
the wide-rms `x_buf` outcome. The chunked-amax fallback (§8 Q3) stays
speced but is NOT needed on area evidence. Scope caveats, mirroring the
wide-rms probe: yosys mapping evidence only — **no P&R, no timing** (timing
comes only from the I-B Vivado build report), and the post-synth netlist
was not re-simulated. A no-abc9 `synth_xilinx` control run shows the same
memory mapping and delta shape (+7 RAMB18E2, +1 DSP, +0.9k LUT of a ~13k
un-abc9 total) — flow named because absolute LUT counts differ wildly with
abc9 on this block.

---

## 2. Golden composition points (the arbiter map)

Every new `apex_top` path must be bit-exact against these functions, in this
order (all cites `golden/apex_golden/transformer.py` unless noted):

| # | tile path | golden function(s) | ordering / constraints |
|---|---|---|---|
| L1 | RMSNorm-1, wide | `attention.rmsnorm_fx_wide` (`attention.py:124`; μ table `_wide_mu` `:109`) over `quant_rows_i8(X)` rows with `gamma1`; then C-1 re-quant `h8, s_h = quant_rows_i8(h/256)` (`:405-413`) | tile realization: `asu_rmsnorm #(RMS_D_MAX=3584)` (params provenance vs `_wide_mu`, the `verif/asu/wide` params-check gate) → `apex_q78_to_fp32` (exact `/256`) → `seam_feeder_quant` (C-1 amax row quant, fp16 scale on `fs_*`). D=3584=28·128 is a legal wide length. Wide feeder needed for the 3584 row (reality check 4). |
| L2 | Q/K/V projections (+ Qwen bias) | `gemm_i8_ksplit` (`compute.py:41-73`; K=3584 legal, ≤ `K_TOTAL_MAX` `compute.py:17`), dequant by `s_h[t]·s_w*`, bias `bq/bk/bv` added in REAL units BEFORE the single fp16 narrowing (`:417-428`, `LayerWeights` `:289-303`) | K/V write port = S-2 fp16 bus (existing scale_quant→KVQ path). GQA: `H_kv` groups, query head h reads KV head `h // (H/H_kv)` — host choreography, no datapath change (S6). Bias-add-before-narrowing is a composition point the S-2 path must honor when biases are non-None (§2b-3). |
| L3 | RoPE | `rope_fx` (`:197-215`) = HF half-split pairs `(i, i+half)`; phases `rope_phase_q` (`:185-194`), LUT `cos_fx/sin_fx` (`:152-166`); tables byte-mirrored via `golden/gen_rope_lut_tables.py` | K rows rotated at m=t (per cached position), **q rotated at m=q_pos** — S8 self-inclusive decode: q_pos=t with the decode row duplicated (`:383-390`); **V is NOT rotated**; RoPE sits AFTER bias add, BEFORE KV store. **`rope_theta_base` = 1e6 for Qwen2/2.5** (`:169-182` docstring, field default 10000 at `:303` is the frozen test anchor only). The RTL is base-agnostic — it consumes precomputed `u_q` codes — so the base lives in host tooling + the L4 vector generator, and the L4 suite MUST carry a base=1e6 case so Qwen parity is tested, not assumed. **Phase source is a loaded per-(m,i) table computed host-side in float64 and quantized ONCE (C-ROPE). On-tile incremental phase accumulation is BANNED — it is not bit-exact vs the golden's per-m float64 rounding.** |
| L4 | attention core | unchanged — the L3-proven step (`attention.py` via the existing taps) | no edits; per-head jobs at head_dim ≤ CFG_D (Qwen 7B head_dim=128 fits CFG_D=128). |
| L5 | o-proj | `_proj_epilogue(attn, Wo, s_wo)` (`:352-370`): C-1 `quant_rows_i8` of the attn concat → `gemm_i8_ksplit` → C-2 `requant_i32_to_i8` with **host-loaded `(rq_scale, rq_shift)`** → `s_out` fold | `calib_requant`'s input is the amax of the very GEMM the descriptor configures — the same causal circularity D-028 resolved for PV: the rq pair is a HOST-supplied per-step input, carried in the WS descriptor's existing `rq_*` fields (no new CSR needed). The no-clip assert (`:365`) becomes a checked TB expectation, not RTL. attn arrives tile-side as PV `o8` codes; dequant to the feeder is exact in f32 when the composite is fp16-grade (§2b-1). |
| L6 | residual-1 / residual-2 | C-6 (`:79-91`): `r1 = f16(X[T] + attn_proj)` (`:464`), `r2 = f16(r1 + ffn_out)` (`:484`) — ONE RNE per sum | RTL primitive = `residual_add` semantics (fp16 operands in, single RNE out); the incoming residual row is fp16-grid in steady state (it is the previous layer's `r2`). §2b-2 governs the sublayer-operand grid. |
| L7 | RMSNorm-2 | `quant_rows_i8(r1)` → `rmsnorm_fx_wide(·, gamma2)` → C-1 `h2_8` (`:466-473`) | v1 composition: r1 is read back and C-1-quantized by the HOST (a host job, v0.1-boundary style), re-entering via `xa_*/xg_*` — bit-exact because `quant_rows_i8` is deterministic on the read-back fp16 row. On-tile loopback is an IB-WALK-era optimization, not this lane's obligation. |
| L8 | SwiGLU FFN | gate/up = `gemm_i8_ksplit` + dequant (`:476-479`); `silu_apply` (`:259-267`) = **RNE to Q5.10** → `silu_fx` LUT (`:248-256`) → `/2^12` → f16; `p = f16(silu_apply(gate) · up)` (`:480`); down-proj = `_proj_epilogue(p, Wd, s_wd)` (`:481`) | `asu_silu.sv` is the verified LUT core (tables gated vs `_silu_lut_tables`). The Q5.10 input quantization and the product's single f16 narrowing are new RTL (stage 3). d_ffn=18944 splits by N (`n_chunk ≤ DIM_MAX`) and K (down-proj K=18944 ksplit) — job choreography, no datapath change. |

### 2b. The fp16-grade / fp16-bus composition addendum (stage-1 decision, owner-gated)

Four places in `decoder_layer_fx` operate on float64 operands that no
bus-realizable datapath can reproduce bit-exactly as written, because the
per-tensor weight scales `s_w*` are arbitrary float64 and the sideband
contract (`apex_scale_quant` C2: composite must be fp16-grade, ≤11
significant bits, else `scale_error` — B3_WEIGHT_PATH.md §2 finding 3) plus
the fp16 buses narrow them:

1. **o-proj / FFN activation prep**: `quant_rows_i8` consumes `attn` /
   `swiglu` built from `o8·s_out` with full-float64 `s_out`. With an
   fp16-grade f32 composite, `o8 × composite` is EXACT in f32 (≤19
   significand bits) — the tile path is exact *given* the narrowed composite;
   the golden must consume the same narrowed composite for bit-identity.
   Precedent: B3 finding 3 measured fp16-grade composite rounding at ≤5e-5
   absolute — "free".
2. **Residual sublayer operand and SwiGLU `up` operand**: golden adds/
   multiplies unnarrowed reals (`attn_proj`, `up_real`); the bus carries
   fp16-grid or exact-f32 values.
3. **RoPE input grid**: `decoder_layer_fx` rotates unnarrowed `q_real` /
   `K_real`; the hardware rotates the S-2 fp16 bus values
   (`gen_layer_vectors.py:21-27` already composed it this way, documented).
4. **Projection-bias add point** interacts with (3): bias is added before the
   single narrowing — the tile's S-2 path must reproduce that ordering when
   biases are enabled.

**Resolution path (stage 1):** land an ADDITIVE golden composition mode —
new function/flags (e.g. `decoder_layer_fx(..., bus_grade=...)` defaulting
OFF, or a sibling `decoder_layer_fx_bus`) — with a three-part gate:
(a) flags OFF is bit-identical to today's `decoder_layer_fx` (regression);
(b) flags ON stays within the derived budget vs `decoder_layer_ref`, and the
per-point deltas are pinned as regression registers (the
`test_effective_bits.py` discipline);
(c) flags ON becomes the L4 arbiter.
This is a numeric-contract addition, so it takes an **owner sign-off** (the
§A-1 pattern) before stage 5 can gate against it. No existing golden function
is edited.

> **S1 LANDED (2026-07-22, D-030 per ruling R4; owner sign-off PENDING —
> §0.1).** Implemented as `BusMode(x_f16, rope_in_f16, scale_f16grade)` on
> `decoder_layer_fx` (default `None` == bit-identical legacy; gated).
> Points (1) and (2) above **dissolved into `scale_f16grade` plus a tested
> exactness lemma** — with graded composites, every operand feeding a
> single-RNE narrowing (residual sums, `silu·up`, `o8·composite`) is EXACT
> in float64, so no operand-narrowing flag exists and the RTL can be
> bit-exact with one RNE per contract point (2,008 exact-Fraction checks,
> `test_layer_bus.py` §C). Point (3) is `rope_in_f16`, point (4) is its
> ordering. Measured deltas + the rope-absorption finding:
> `docs/results/ib_layer_bus/RESULTS.md` (pinned at rtol 1e-6).

---

## 3. CSR / route map proposal

**Principles.** `apex_pkg.sv` frozen; `csr_regs.sv` untouched; **zero new
`apex_top` ports proposed** — all new control rides a glue CSR window
(ERR_STICKY/WALK pattern, `apex_top.sv:367-431` incl. the `csr_rdata`
override at `:1198-1200`), which avoids PINMISSING churn in every TB/CL that
instantiates `apex_top` and matches how the F2 host already drives the tile
(CSR regops). Open question Q2 asks integration to confirm this over new
`rt_*` input pins. Route bits are quasi-static levels — change only while
the touched streams are quiescent (the existing `rt_*` discipline).

**New addresses (all glue-owned; csr_regs reserved fall-through). Nothing
existing moves:** ≤`0x54` csr_regs, `0x58` ERR_STICKY, `0x5C-0x6C` WALK.

> **SUPERSEDED (2026-07-22): the table below is the stage-0 SKETCH.** The
> FROZEN S4 register map — different LAYER_CTRL bit layout (l_ser_dst /
> l_fsrc_ext consumer selects, l_rope_pos, the l_kv_map carve), the JOBC
> alternating composite file, and the memory shapes — is **§3b**, which is
> what the RTL implements and what IB-WALK's ROM consumes. Kept for the
> design-derivation record only.

| Addr | Name | Dir | Meaning |
|---|---|---|---|
| `0x70` | `LAYER_CTRL` | RW | route/mode levels: `[0]` lr_rope_en (RoPE stage live on the S-2 fp16 write bus; 0 ⇒ today's path byte-identical), `[1]` lr_rope_bank (phase bank for the rows now streaming: 0=K bank per-t, 1=q bank at q_pos), `[2]` lr_swiglu_en (MXE-res→SwiGLU route), `[3]` lr_fsrc_ext (feeder-source extension: `{lr_fsrc_ext, rt_feeder_src}` = 00 widen / 01 KVQ / 10 layer-deq / 11 SwiGLU p), `[4]` lr_resid_arm (residual unit consumes the next deq'd sublayer row), `[5]` lr_resid_bank (0: r1 = row+deq, update row RAM in place; 1: r2 = row+deq), `[31:6]` reserved-0 |
| `0x74` | `LAYER_PTR` | RW | auto-inc LOAD pointer; `[29:28]` bank (0 phase-K RAM, 1 phase-q RAM, 2 residual row RAM), `[15:0]` entry index |
| `0x78` | `LAYER_DATA` | RW | data write at PTR (auto-inc): phase banks pack 2×14-bit `u_q`/word; residual bank packs 2×fp16/word |
| `0x7C` | `LAYER_JOB` | RW | job kick for the framed units (D-006-style; legality-checked, reject = pulse+sticky, no state change): `[11:0]` cols (DIM_W grade), `[13:12]` unit (0 layer-deq, 1 SwiGLU, 2 residual) |
| `0x80` | `LAYER_STATUS` | RO/W1C | `[0]` deq_busy, `[1]` swiglu_busy, `[2]` resid_busy, `[3]` job_ready, `[8]` layer_err sticky (W1C; same-cycle set WINS), `[11:9]` layer_err_code (0 none, 1 JOB illegal, 2 GRADE composite not fp16-grade, 3 FRAME, 4 ABORT) |
| `0x84` | `LAYER_RPTR` | RW | auto-inc READ pointer into the residual/result row RAM (r1/r2 read-back for L7 and the layer output) |
| `0x88` | `LAYER_RDATA` | RO | read at RPTR (auto-inc): 2×fp16/word |
| `0x8C` | `LAYER_JOBC` | RW | f32 composite for the next layer-deq job (fp16-grade ENFORCED — violation raises layer_err GRADE, job refused) |

### 3b. FROZEN S4 job-port/route table — PUBLISHED 2026-07-22 for IB-WALK (D-029 ROM)

**Status: FROZEN for the walker ROM.** Changes after this point follow the
B1 §A-1 rule: update the table, the glue, and the walker ROM together, never
silently. Merge shape honored: the LAYER glue is a NEW parallel region after
the B1 WALK region; the B1 region (a) hunks (`:367-457`, `:557-568` at the
WALK fmt=1 delta) are not reflowed.

| Addr | Reg | Bits (RESET all-0) |
|---|---|---|
| `0x70` | `LAYER_CTRL` RW | `[0]` **l_rope_en** — rope_row live on the S-2 write bus (0 = byte-identical legacy path); `[1]` **l_rope_bank** — phase source 0 = K-table @ l_rope_pos, 1 = q bank; `[3:2]` **l_ser_dst** — serializer-output consumer: 0 legacy (squant path per rt_squant_src), 1 layer-deq, 2 swiglu (its GATE/UP phase self-selects), 3 reserved-0; `[5:4]` **l_fsrc_ext** — feeder-source override: 0 legacy (rt_feeder_src mux), 1 layer-deq fp32, 2 swiglu-p (f16→f32 exact widen), 3 reserved-0; `[6]` **l_resid_arm** — deq-output consumer: 0 feeder path, 1 apex_residual; `[14:8]` **l_rope_pos[6:0]** — K-phase table position; `[17:15]` **l_kv_map[2:0]** — S4b LIVE: host-mode engine select for the per-KV-head GQA bank (latched/read-back only in `KVQ_GQA_NENG>1` builds — writes ignored and reads 0 otherwise, byte-identical to the S4 reserve; walk mode bypasses it, walker2 `kv_eng_sel` drives the select directly); `[31:18]` reserved-0 |
| `0x74` | `LAYER_PTR` RW | `[29:28]` bank: 0 phase-K table, 1 phase-q bank, 2 residual row RAM; `[15:0]` linear index (phase-K: `pos·(CFG_D/2)+pair`). Auto-inc on each LAYER_DATA write |
| `0x78` | `LAYER_DATA` RW | write-at-PTR: phase banks `[13:0]` one u_q code/write; residual bank `[15:0]` one fp16/write |
| `0x7C` | `LAYER_JOB` RW | `[11:0]` cols, `[13:12]` unit: 0 layer-deq, 1 swiglu, 2 residual. Write = job push (legality-checked at the unit; reject = err+sticky, NO state change) |
| `0x80` | `LAYER_STATUS` RO/W1C | `[0]` deq_busy `[1]` swiglu_busy `[2]` resid_busy `[3]` rope_busy; `[8]` layer_err sticky (W1C; same-cycle set WINS); `[12:9]` layer_err_code: 1 JOB, 2 GRADE, 3 FRAME, 4 EXACT, 5 RESID_WINDOW |
| `0x84` | `LAYER_RPTR` RW | `[15:0]` residual-row read index; auto-inc on each LAYER_RDATA read |
| `0x88` | `LAYER_RDATA` RO | `[15:0]` fp16 at RPTR (r1 after a residual job; r2 == the layer output after the second) |
| `0x8C` | `LAYER_JOBC` RW | composite file, alternating: 1st write latches comp_a, 2nd comp_b; index resets on JOB push. deq consumes comp_a; swiglu a=gate, b=up. All composites fp16-GRADE positive-normal f32 (C2) or the job refuses (GRADE) |

**Internal drive nets (the walker's future mux point — ONE ingress per R5):**
level registers `l_rope_en_q, l_rope_bank_q, l_rope_pos_q[6:0], l_ser_dst_q[1:0],
l_fsrc_ext_q[1:0], l_resid_arm_q` + job pulse/payload `lj_push, lj_unit[1:0],
lj_cols[11:0], ljc_a[31:0], ljc_b[31:0]` + load ports `lp_bank/lp_idx/ld_we/ld_data`.
The walker drives these SAME registers' inputs behind a walk-mode mux
(B1 stage-5 idiom); host CSR writes are the S4 path.

**Consumer map (level → the mux it feeds):** l_rope_en → the S-2 seam mux
(`apex_scale_quant.f_*` → rope_row → `kv_s_t*`; off = direct, byte-identical);
l_ser_dst → `ls_out_*` fanout (squant-v / deq-in / swiglu-in);
l_fsrc_ext → the feeder-source 4-way (widen / KVQ / deq-fp32 / swiglu-p-widened);
l_resid_arm → deq-output fanout (feeder / residual);
l_rope_bank + l_rope_pos → the phase-RAM read mux feeding `rope_row.ph_*`.

**Memories (host-loaded via PTR/DATA; walker-loadable via the same ports):**
phase-K `[T_ROW_MAX][CFG_D/2]`×14 b (+ q bank `[CFG_D/2]`) — the C-ROPE
per-(m,i) table, float64-folded host-side (theta base 1e6 for Qwen2/2.5);
residual row `[LAYER_DM_MAX]`×16 b (param default 128; wide build 3584),
updated in place by each residual job.

**D-030 CANONICAL grade narrowing (JOBC composite values — binding for
WALK's replicas, logged to the combine agenda):** a composite is
`weight_codec.f16_grade(comp_f64)` where `comp_f64` is the float64 product
of that composite's exact factor chain, graded ONCE at the final composite
(never per-factor). Bit-level: (1) IEEE-RNE round `comp_f64` to fp32;
(2) the result must be a POSITIVE NORMAL fp32 or the consuming unit refuses
(GRADE); (3) RNE-round the fp32 significand to 11 significant bits — clear
`frac[12:0]`, and round up (`+0x2000`, carry into the exponent is correct)
iff `frac[12:0] > 0x1000` or (`== 0x1000` and bit 13 set — tie-to-even).
**This RETAINS the fp32 EXPONENT range and is NOT an fp16 round-trip:** an
fp16 cast clamps to [2^-24, 65504] and DENORMALIZES below 2^-14, silently
destroying significand bits once the graded value needs more precision
than the fp16 subnormal grid holds (`s_h·s_w` chains land there).
**Verified distinguishing example (2026-07-23):** comp ≈ 1.811241e-6 →
`f16_grade` = 1.8114224076e-6 (11 significand bits at e8=107); the fp16
round-trip of that SAME graded value returns 1.7881393433e-6 ≠ graded —
and below 2^-24 the fp16 trip is off by orders of magnitude
(4.87e-8 → 5.96e-8). Conformance self-check for any producer:
`f16_grade(x) == x` (idempotence), `(bits & 0x1FFF) == 0`, `0 < e8 < 255`.
The graded set (S1/§2b): Q/K/V dequant comps `s_h[t]·s_w{q,k,v}`; per-head
attn output `s_out`; `_proj_epilogue` `s_out` (Wo, Wd); FFN `s_h2·s_w{g,u}`.

**CANONICAL LAYER choreography (the L4 harness order — the arbiter on the
merged tree; WALK's derived rules are CONFIRMED and extended):**
1. `LAYER_CTRL` levels set while the touched paths are quiescent —
   **including `l_resid_arm` BEFORE the producing deq job is pushed**
   ("arm-before-stream": confirmed).
2. **Consumer-before-producer:** in a chained flow (deq → residual) the
   CONSUMER's JOB is pushed first (residual JOB, then deq JOBC+JOB, then
   the serializer job, then beats). A consumer without a pending job
   presents ready=0 and stalls the chain — safe, but a deadlock until the
   push; hence the rule.
3. `LAYER_JOBC` writes (comp_a [, comp_b]) come IMMEDIATELY before their
   `LAYER_JOB` push ("JC-before-push": confirmed — the push resets the
   JOBC index, so interleaving another pair between JC and push is
   illegal). At most ONE pending push; poll STATUS before the next (the
   glue raises JOB err code 1 on push-while-pending).
4. Stream the beats through the selected routes; poll `LAYER_STATUS`
   (busy bits fall; err sticky checked, W1C).
5. Readback: `LAYER_RPTR` then `LAYER_RDATA` reads — **1-CYCLE sample
   discipline** (the WALK read pattern; a late sample reads csr_regs'
   reserved `0xDEADBEEF` — measured, §4 S4 row).

**apex_residual numeric contract (measured before freezing):** sublayer
operand arrives as EXACT fp32 from layer-deq; the add runs on a 2^-38-grid
57-bit window, one RNE (C-6). Out-of-window operands REFUSE loudly
(RESID_WINDOW; a value ≥ 2^17 short-circuits to ±inf — always correct there).
**Measured over all 1,536 BUS_ON L4 sublayer operands: worst window margin
+19 bits, 0 would-refuse, 0 inf-shortcuts** (`verif/top/l4` measurement,
2026-07-22) — the window is not near any real operand.

`0x90+` is explicitly left free; first claimant at integration should be the
B3 W4-B route/CSR select. New-block error pulses land in `LAYER_STATUS[8]`,
**never** `err_sticky[15]` — that bit is reserved-0 and pinned by every L3
case's `ESTK` expectation (the B1 rule, `apex_top.sv:415-422`). Footgun
carried forward from B1 §8 row 11: the KVQ engine AXI-Lite window reuses the
same numerals on a physically separate bus — no conflict; keep documenting it.

**New RTL (own files, add-only):**

- `rtl/misc/f16_arith_pkg.sv` — synthesizable fp16 primitives (decode,
  exact-fixed-point align/add, RNE narrower) in integer RTL; proof style =
  `seq_walker_comp` P1-P6 / `cq_fp_pkg` lineage.
- `rtl/rope/rope_row.sv` — row framer for the S-2 bus: half-row pair buffer
  (D/2 × 16 b), phase-RAM read, integer rotation MAC (replaces `rope.sv`'s
  `real` rotation for synthesis; `rope.sv` stays as the behavioral
  cross-check in TBs). Sits between `apex_scale_quant`'s `f_*` output and
  the KVQ `s_axis` (`apex_top.sv:823-826` / `:735-742`), UPSTREAM of the
  `kv_s_t*` nets — so the B1 store-snoop (`seq_walker_comp`, `:519-523`) and
  the `dbg_f16_*` tap observe ROTATED K̂ beats exactly as the feeder will
  read them back; rope-off is byte-identical by mux.
- `rtl/misc/residual_add_fx.sv` + `rtl/top/glue/apex_residual.sv` — integer
  fp16 residual add; row RAM (LAYER_D_MAX × 16 b, parameter default 128),
  in-place r1 update, r1/r2 read-back.
- `rtl/top/glue/apex_layer_deq.sv` — int32 stream × fp16-grade f32 composite
  → exact fp32 (mirrors `seam_score_dequant` numerics), feeding the feeder
  source mux and the residual/SwiGLU units.
- `rtl/asu/asu_swiglu.sv` — Q5.10 RNE quantizer → `asu_silu` core → hold-N
  register file → gate·up product → single f16 narrowing.

**Elaboration parameters added to `apex_top` (defaults preserve today's
build byte-identically):** `RMS_D_MAX` (default 128; I-B CL sets 3584),
`LAYER_D_MAX` (residual/phase RAM depth, default 128), plus the stage-6 wide
feeder elaboration knob. Parameter additions with unchanged defaults are
additive (existing instantiations elaborate identically).

**No norm unit (IB-WALK cross-reference).** The table above has NO norm
`LAYER_JOB` unit and NO `l_*` norm level — RMSNorm-1/2 are HOST-STREAMED
(`xa`/`xg` → `u_rms`; L1/L7). The walker-mode norm-step completion is the
existing **`dn_rms`** done event, not a published port. Ratified by the S5
segment-2 harness (which drives rmsnorm exactly this way); full ruling +
the `hd_ready`/`ss_* s_q` q-staging arbitration in §3c.

### 3c. L4 host-composition harness spec (the generator/choreography contract)

**Status: S5 (b) in progress.** This is the spec the S5-row "generator unstub"
cite points at (the frozen §3b table above is the port/choreography canon;
this is how `verif/top/l4/gen_l4_vectors.py` + `tb_l4_compose.sv` consume it).
Harness discipline = the L3 pattern (`verif/top/l3/`): the generator emits a
text **op stream** (CSRW/CSRR/CSRP/ROUTE/XR/GR/WB/FJOB/QJOB/SJOB/LJOB/AJ/WJ/
QS/CS/EFS/ESS/ERO + the LAYER-window CSR ops) the TB interprets; the arbiter
is `decoder_layer_fx(bus=BUS_ON)` (D-030); every checkpoint is bit-exact.

**Segment 1 — o-proj → r1 (Option 1, owner-confirmed 2026-07-26).** Arbiter
tie = `RDATA == r.r1`. The o-proj epilogue internals `decoder_layer_fx`
DISCARDS (`transformer.py:563` keeps only `attn_proj`) are **RE-DERIVED via
PUBLIC golden functions** — `quant_rows_i8` / `gemm_i8_ksplit` /
`calib_requant` / `requant_i32_to_i8` / `f16_grade` on `r.attn` + `w.Wo`, an
exact inline replica of `_proj_epilogue(..., grade=True)`, **ZERO golden
edits, no new sign-off surface**. This yields `a8` (C-1 activation),
`(scale,shift)` (the MXE requant descriptor fields), `o8` (the requantized
serializer codes) and the fp16-graded `s_out` (the layer-deq JOBC composite);
residual row = `f16(X[T])`. **MANDATORY generator self-check (landed,
`gen_l4_vectors.py::seg1_selfcheck`): before any RTL runs, assert the
re-derived chain reproduces `r.attn_proj` AND `r.r1` bit-exactly — a failure
is the generator's, never golden. PROVEN 6/6 cases 2026-07-26.** The tile tail
routes `rt_res_dst=1` (MXE res → serializer) → `l_ser_dst=1` (ls_out →
layer-deq, JOBC comp_a = graded `s_out`) → `l_resid_arm=1` (deq → residual,
row pre-loaded via LAYER_PTR bank-2 + LAYER_DATA) → LAYER_RPTR/RDATA readback
(1-cycle sample). MXE descriptor: `OP_GEMM_WS` **with `rq_en=1`, `rq_scale`,
`rq_shift`** (WS+requant honored — `mxe_ctrl.sv:287`; the L3 projections are
WS-no-requant, the PV epilogue is OS-with-requant — this composes both).

> **DRIVE-PATH FINDING (2026-07-26, escalated).** The o-proj GEMM cannot be
> driven **standalone** to hit `r.r1`: its activation `attn` is the attention
> output, and the tile has **no host port that injects arbitrary act codes**.
> The act-stage loads only from the feeder (`fq_out_beat`) or scale_quant
> (`sq_q_beat`) (`apex_top.sv:1337-1339`); the feeder's only host-reachable
> input is the RMSNorm path (`xa→u_rms→u_widen`) or the deq/swiglu overrides
> (`:1056-1073`) — none carries `attn`. So a bit-exact-`r.r1` segment-1 must
> source `attn` from the **attention back-half** (PV `o8` → layer-deq
> `l_fsrc_ext=1` → feeder → o-proj GEMM), i.e. it composes the full L3
> attention drive + a first deq pass THEN the o-proj + second deq/residual
> pass. The single-GEMM shape in the §4/S5 sketch is not reachable in
> isolation.
>
> **OWNER RULING (2026-07-26): segment 2 first (below, LANDED), then segment 1
> as the full back-half composition (Option 2) — the NAMED NEXT SLICE.**
> Segment-1 drive plan (arbiter `RDATA == r.r1`, re-derived internals proven
> 6/6 by `seg1_selfcheck`): (a) run the L3 attention drive (the `core_case`
> idiom) to produce the per-head PV `o8`; (b) route PV `o8` → layer-deq
> (`l_ser_dst=1`) with the per-head graded composite, `l_fsrc_ext=1` so the
> deq output feeds the FEEDER → C-1 `a8` (= `quant_rows_i8(attn)`); (c) load
> `Wo` host weights + descriptor `OP_GEMM_WS,1,D,8, rq_en=1, rq_scale, rq_shift`
> (the re-derived pair) → serializer `o8`; (d) re-route `l_ser_dst=1`,
> `l_resid_arm=1`, JOBC comp_a = graded `s_out`, residual row = `f16(X[T])`
> → deq → residual → LAYER_RPTR/RDATA == `r.r1`. Two deq passes, level re-route
> between them; this is where the `hd_ready`/`ss_* s_q` q-staging arbitration
> is settled.

#### 3c-1. SEGMENT-1 EXECUTION FINDINGS (2026-07-26, IB-LAYER segment-1 session) — step (a) is BLOCKED; owner ruling requested

Work order: execute the Option-2 drive plan above. Before writing harness code
the session mapped every wire the plan names, on the merged tip (`e219c6a`).
Two of the plan's premises are **CONFIRMED**; two are **BLOCKERS** that make
"`RDATA == r.r1` on all six cases" unreachable as written. All four are
file-and-line measurements, not arguments. **No harness code was written
against a blocked plan** and no gate is claimed; pre-edit baselines were
captured first (17/17 targets rc=0, §3c-1 "baseline" below).

**CONFIRMED premise 1 — WS+requant is honored (plan step (c) is sound).**
`rtl/mxe/mxe_ctrl.sv:287` latches `requant_en_q <= desc.requant_en` in `S_IDLE`
on descriptor accept, with `rq_scale_q`/`rq_shift_q`, **unconditionally on
opcode** — the only opcode-dependent field is `clear_job` (the OS-accumulate
case). So `OP_GEMM_WS` + `rq_en=1` + the re-derived `(rq_scale, rq_shift)`
requantizes exactly as the plan assumes. The §3c citation is correct.

**CONFIRMED premise 2 — multi-head `attn` composes into ONE feeder row.**
`decoder_layer_fx` builds `r.attn` as H per-head slices, each dequantized with
its OWN graded `s_out` (`transformer.py:558`), while `apex_layer_deq` carries
exactly one composite per job (`ljc_a_q`, `apex_top.sv:960`) — so an H-head row
needs H deq jobs. That is safe: **the feeder frames by ELEMENT COUNT, not by an
incoming `last`** (`apex_top.sv:1343`, and the deq→feeder mux at `:1326` passes
`valid`/`data` only — no `last`). H consecutive `head_dim`-column deq jobs
therefore concatenate into a single `D_model` feeder row, and the C-1 row scale
`s_a` is taken over the whole concatenated row, exactly as `quant_rows_i8(attn)`
does. Plan step (b) works — it just needs H jobs, not one; §3c's singular "the
per-head graded composite" should read "one deq job per head, comp_a =
`f16_grade(head.s_out)`".

**BLOCKER A (hard, kills step (a) for ALL SIX cases) — a RoPE-rotated `q` row
has no bit-exact path to the act stage.** The act stage loads from exactly two
sources: `as_ld_valid = w_rt_act_src ? sq_q_valid : fq_out_valid`
(`apex_top.sv:1604-1610`) — the feeder, or `apex_scale_quant`'s Q7 beat. The
feeder's four sources are widen(`u_rms`) / KVQ-read / layer-deq / swiglu-p
(`:1323-1338`). **`rope_row`'s output goes to ONE place: `kv_s_t*`** (`:935-937`),
whose only consumers are the two KVQ banks (`:1196`, `:1265`), the B1 store
snoop (`:587`) and the `dbg_f16_*` tap (`:1799`). There is no route from
`rope_row` to the act stage, and none back into `apex_scale_quant` — whose two
modes are mutually exclusive (`fo_valid = ... && !mode_q`, `apex_scale_quant.sv:441`):
MODE_F16 emits `f_*` (→rope→KVQ) and MODE_QUANT emits `q_*`/`s_*` (the q8 codes
+ `s_q`), never both. So the Q7 path that produces `q8`/`s_q` **bypasses rope
entirely**, and the only rope→act route is store-into-KVQ-and-read-back, which
is CQ-8 lossy and therefore not bit-exact.
**This bites every case: RoPE is non-identity on `q` in all six**
(measured `max|q_rope − q_real|`): `l4_h2_hd64` 1.966, `l4_h1_hd128` 2.739,
`l4_gqa_h2kv1` 2.532, `l4_qwen_theta` 1.679, `l4_bias` 2.232, `l4_selfinc` 1.939.
Note this is a HALF-BUILT feature, not an oversight in the harness: §3b
publishes `l_rope_bank=1` ("q bank at q_pos") and the `ph_q[L_HALF]` phase RAM
is implemented and host-loadable (`apex_top.sv:751`, `:859`) — **the level and
the memory exist; the datapath SINK for the rotated q does not.** L3 never
exposed this because L3's `attention_core` takes an already-rotated `q` as a
given input; L3 has no rope in the tile path at all.

**BLOCKER B (scoped to one case) — `l4_bias` has no tile bias path.**
`decoder_layer_fx` adds `bq`/`bk`/`bv` in REAL units to `q_real`/`K_real`/`V_real`
BEFORE the single fp16 narrowing (`transformer.py:509-514`). A repo-wide search
(`grep -rniE 'bias' rtl/ --include='*.sv'`) finds only floating-point
exponent-bias arithmetic — **no projection-bias adder anywhere in the RTL**;
`rope_row.sv:4` even documents the assumption ("q row, bias already folded
upstream"). The tile's projection chain is int8 GEMM → serializer → scale_quant
composite multiply; there is no point at which a per-channel fp16 bias can be
added. Folding the bias into the INT8 weights or into an extra constant
activation row both leave the int8 grid and are not bit-exact. **`l4_bias`
cannot have its q/K/V produced by the tile at any q_pos**, so "all six cases"
is unreachable regardless of Blocker A.

**Scale note (not a blocker, but the plan understates it).** Step (a) "run the
L3 attention drive" is not a small adapter: it is phases A–G of
`gen_l3_vectors.py` (CSR init, K/V projections + KVQ AXI stores, q projection,
score/softmax/P-requant, PV+epilogue) plus every port `tb_l4_compose` currently
ties off (`kv_*` AXI-Lite, `ds_*`, `xw_*`, `qj_*`, `dj_*`, `lj_*`, `qs_*`, `cs_*`,
`wj_*`) and their expectation ops (ESS/ERO/ETIP/ESTK/CSRR/CSRN/taps) — i.e. most
of `tb_apex_l3.sv` (1,267 lines) rebuilt against the L4 record, and re-emitted
per head from `LayerFx.heads[]` with GQA record addressing and K-rope staging.
The per-head `AttnCore` records do carry the same fields L3's generator consumes,
so the port is mechanical — but it is a multi-stage slice, not a tail extension.

**FINDING C (scoping correction) — a HOST-mode segment-1 harness cannot, by
construction, settle `hd_ready` on its own.** §3c says segment 1 is where the
`hd_ready`/`ss_* s_q` arbitration lands. Half of that is right and half is not.
The sticky latch has two sides (`apex_top.sv:624-629`): it ARMS on
`ss_valid && ss_ready && ss_last` and is CONSUMED by `wk_hd_valid && hd_sq_seen_q`.
Its only consumer is `u_walk`'s `hd_ready` port (`:665`), and walker2 asserts
`hd_valid` in exactly one state — `assign hd_valid = (state == S2_HLOAD)`
(`seq_layer_walker2.sv:511`) — reachable only after `S2_IDLE: if (walk_en &&
walk_go ...)` (`:587`). **In host mode `wk_hd_valid` is never asserted, so the
latch arms on every `ss_last` and never consumes.** A host-mode L4 harness can
therefore establish the ARM side — that the `ss_*` `s_q` event is well-formed
and that exactly one `ss_last` occurs per q-projection per head, which is what
the arming term keys off — but it cannot exercise the consume-on-accepted-head
term at all. **Definitive arbitration of the sticky form needs a WALKER-mode
MULTI-HEAD case** (a fmt=1 image with `n_heads > 1`), which is IB-WALK's D-029
surface, not this lane's host harness. Recommend §3c be amended to split the
rider: segment 1 (via O3 below) ratifies the arm event; the multi-head consume
semantics — the open question the W-G2 note actually flags ("Multi-head arming
(a FRESH snoop per head, mid-walk)") — move to a walker-mode gate.
**Status this session: `hd_ready` remains PROVISIONALLY ratified, unchanged.
Neither confirmed nor superseded — the drive that would have settled the arm
side is blocked by Blocker A.**

**OPTIONS (owner ruling requested).**

- **O1 — add the q-rope sink in RTL (the real fix; unblocks all rope cases).**
  Extend the LAYER glue with a mux that routes `rope_row`'s output, when
  `l_rope_bank=1`, into the Q7 input (or as a third act-stage source), behind a
  new default-0 `LAYER_CTRL` level so every existing build stays byte-identical.
  Completes the half-built q-bank feature §3b already publishes. Cost: new RTL +
  its own unit gate, then the step-(a) port. Highest value, largest delta,
  and it is squarely IB-LAYER scope.
- **O2 — re-scope segment 1 to `q_pos=0` cases (no RTL change; MEASURED to
  work).** At m=0 the RoPE rotation is the identity, so `q_rope == q_real`
  exactly and the Q7 path is already golden-exact — **verified this session:
  H2/hd64/T8 and H1/hd128/T16 at `q_pos=0` both give `q_rope == q_real` True,
  while K stays genuinely rotated (7/8 and 15/16 rows)**, so `rope_row` on the
  S-2 bus is still exercised for K. Adding `q_pos=0` cases to a SEG1-only case
  list (leaving `CASES` untouched keeps segment 2's 67 EFS checks
  byte-identical) makes the full (a)→(d) chain drivable on today's RTL, closes
  `RDATA == r.r1` on those cases, closes the h8 rider, and settles `hd_ready`.
  The six shipped cases then follow once O1 lands.
- **O3 — split the riders off the full composition.** Land phase D alone
  (`h8` → q-projection GEMM → serializer → scale_quant Q7 → `ss_*`), gated on
  `ESS s_q == r.s_q`. On a `q_pos=0` case this is golden-exact and it (i)
  closes the h8 rider DIRECTLY and earlier than the o8 indirection §3c relies on
  (a wrong `h8` moves the q amax, so `s_q` mismatches), and (ii) drives the real
  `ss_*` `s_q` event that `hd_ready` snoops, which is the only thing the
  `hd_ready` arbitration actually needs. Much smaller than (a)→(d).
- **O4 — synthetic o8 producer for the back-half only.** Deliver the per-head
  `o8` into the serializer via a host-loaded "delivery GEMM" instead of the
  attention path, then run (b)→(d). This WOULD give `RDATA == r.r1` bit-exact
  and would genuinely gate the LAYER back-half (deq composite route, feeder C-1,
  WS+rq o-proj, second deq, residual, RDATA). **It does NOT close the h8 rider**
  (nothing chains `h8` to `o8`) and does not touch `ss_*`, so it cannot arbitrate
  `hd_ready`. Acceptable only if labelled as back-half-only coverage.

**RECOMMENDATION: O3 then O2, with O1 scheduled.** O3 is the smallest step that
closes two of the three riders (h8 directly, `hd_ready` definitively) and needs
no new RTL; O2 then completes `RDATA == r.r1` end-to-end on rope-free-q cases
with the same machinery; O1 is the only path to the six shipped cases and should
be planned as its own staged delta with a unit gate. **`l4_bias` stays out of
every tile-composition gate until a bias path exists** — see the RULING in
§3c-2, which keeps it NAMED rather than struck.

**Baseline (pre-edit, merged tip `e219c6a`, captured before any edit; machine
rule honored — every build gated on `until ! pgrep -f '[v]erilator_bin'`):**
17/17 targets rc=0 — golden `test`; l3 `lint`/`vectors`/`composite`/`build`/
`unit`/`run` (28/28 cases) /`coverage`/`walkdesc`/`run_walker` (24 walked +
4 refused) /`run_walkfmt`/`run_walkfmt2`/`mutate` (3/3); `verif/kvq/gqa all`;
l4 `levels`/`compose` (checks=67 errors=0) /`mutate` (3/3). This doc change is
text-only — no generator, TB, RTL or Makefile was touched, so those results
stand unchanged by construction.

#### 3c-2. RULING on §3c-1 + GAP C (the NP-r narrowing), 2026-07-26

**RULING (integration lead, accepted; no golden change, no spend).**
1. **O3 first, then O2.** O3 = phase D alone (`h8`→q-projection→Q7), O2 =
   re-scope segment 1's case set to `q_pos=0` AND bias-free sub-paths. The
   re-scope is stated in the TB banner and here; segment 2's `CASES` list is
   left untouched so its 67 EFS checks stay byte-identical.
2. **Blockers A and B are NAMED TILE CAPABILITY GAPS, not struck cases.**
   (A) q-staging act-stage sink (= O1) and (B) projection-bias path are **I-B
   RTL work items owned by a follow-on lane**, not by this harness. `l4_bias`
   is excluded from tile-composition gates *by named gap B*, and stays in the
   case set and in every golden-side gate.
3. **`hd_ready` scoping correction ACCEPTED as written (§3c-1 Finding C).**
   The latch's consume side is walker-mode-only, so **final arbitration
   TRANSFERS to a walker-mode multi-head fmt=1 image (D-029 scope,
   W-G3-adjacent)** — not to this lane's host harness. This lane may ratify the
   ARM event only.
4. Mutants + byte-identity ride the re-scoped harness as before.

**GAP C (measured this session while executing the O3 ruling) — the tile's Q7
path omits the NP-r fp16 narrowing, so `ESS s_q == r.s_q` is NOT achievable
bit-exactly, even at `q_pos=0`.** Under `BUS_ON` (D-030, owner-signed) the
arbiter narrows the projection output to fp16 **before** rotation and
quantization — `if bus.rope_in_f16: r.q_real = _f16(r.q_real)`
(`transformer.py:515`/`:521`), the documented "projection → S-2 dequant/narrow →
rope_row → KVQ store" order. The tile's `apex_scale_quant` MODE_QUANT does
**not** narrow: its own header states "**C-1 per-row INT8 quant of exact
products** … `q8, s_q = quant_rows_i8(q_real[None,:])`, `q_real = acc_q * c`"
(`apex_scale_quant.sv:13-15`), and the two modes are mutually exclusive
(`:441`) — MODE_F16 narrows but emits `f_*` (→rope→KVQ), MODE_QUANT emits
`q_*`/`s_*` without narrowing.
**Measured over 70 `q_pos=0` head-rows (14 seeds × 3 geometries, bias-free):**
f16-narrowed `s_q` == golden `r.s_q` **70/70 (100%)**; exact-product `s_q` ==
golden `r.s_q` **49/70 (70.0%)** — **21/70 differ by ±1 ULP** (examples
tile/golden: 8405/8406, 8740/8739, 8618/8617). This is a 30% miss rate, not a
rare corner. L3 never saw it because L3's arbiter is `attention.py`, which has
no NP-r narrowing; the delta is specific to the `decoder_layer_fx` BUS_ON
arbiter this lane gates against.
**Consequence — Blocker A is BROADER than §3c-1 stated:** the tile's intended q
staging is *S-2 narrow → rope → C-1 quant*, and MODE_QUANT is an L3-era
shortcut that skips **both** the narrowing and the rotation. **O1 must add a
narrow-AND-rope sink, not a rope sink alone.** `q_pos=0` removes the rotation
delta; it does not remove the narrowing delta.

**REVISED PROPOSAL O3′ (supersedes O3 — same intent, but golden-exact where O3
cannot be; owner ruling requested).** Split the two things O3 was trying to do:
- **O3′-1 — close the h8 rider GOLDEN-EXACTLY via the K/V projection path
  (no re-scope needed, no gap).** The K path already has *both* missing pieces:
  MODE_F16 performs exactly golden's one-RNE narrowing
  (`K_f16 = f64_to_f16_bits(K_real)`, header `:9`) and `rope_row` then
  applies the rotation on the S-2 seam → `kv_s_t*`. So the tile's rotated K̂
  beats equal golden's `r.K_rope` bit-exactly. Gate = the `dbg_f16_*` tap (the
  L3 `TAPF16` idiom) vs `f64_to_f16_bits(r.K_rope)` / `V_f16`. Because
  `K = h8[:T] @ Wk · comp`, **any h8 error breaks every K beat — the h8 rider
  closes directly and bit-exactly against golden**, on 5 of the 6 SHIPPED cases
  (all but `l4_bias`, named gap B), at any `q_pos`. It also gives `rope_row`
  its first in-tile composition coverage.
- **O3′-2 — ratify the `hd_ready` ARM EVENT by shape, not by value.** Run the
  Q7 q-projection job and check the arm event `ss_valid && ss_ready && ss_last`
  occurs exactly once per Q7 job with the expected cadence — which is all the
  arming term keys off. **Do NOT gate on the `s_q` value** (Gap C); record the
  measured delta instead. Per ruling item 3 the consume side is out of scope
  here anyway.
**Recommendation: adopt O3′-1 + O3′-2, then O2.** O3′-1 is strictly stronger
than O3 for the h8 rider (golden-exact vs a re-derived reference with a
disclosed 30% ULP delta) and needs no case re-scope. Gating `ESS s_q` against a
re-derived tile-semantics reference was considered and **rejected as a lying
gate risk**: it would read as "golden-exact" while silently differing from the
signed arbiter in 30% of rows.
**Status: O3′ APPROVED (integration lead, 2026-07-26), both parts. See §3c-3
for the geometry constraint found while implementing it.** Byte-identity unaffected (doc-only commits).

#### 3c-3. GAP D — `CFG_D` is overloaded, so O3′-1 covers 1 of 6 shipped cases (not 5)

**Premise re-verified first (O3′-1 is sound):** re-deriving the tile's K path
with PUBLIC golden fns — `gemm_i8_ksplit(r.h8[:T], Wk)` → `_grade_rows(s_h·s_wk)`
→ `_f16()` (MODE_F16's one RNE) → `rope_fx` per KV head — reproduces
`f64_to_f16_bits(r.K_rope)` **bit-exactly on all six cases**, and the V path
reproduces `r.V_real` likewise. The gate is real: `K = h8[:T] @ Wk · comp`, so
any h8 error breaks every K beat.

**GAP D (measured while wiring the drive).** `CFG_D` is a SINGLE build parameter
feeding two families of units with incompatible width meanings:
- **per-head width** — `rope_row #(.D(CFG_D))` (`apex_top.sv:941`; the module
  header is explicit: "*D is the per-head row length (head_dim build point:
  64/128) — RoPE is per-head and never sees hidden-D rows*"), the KVQ banks
  (`:1170`, `:1241`), and the phase RAM `L_HALF = CFG_D/2` (`:731`).
- **D_model width** — `seam_feeder_quant #(.D(CFG_D))` (`:1298`, the C-1 row for
  h8) and both `apex_stage_buf #(.D(CFG_D))` act/weight stages (`:1557`, `:1580`,
  which set the projection GEMM's contraction row).

A single build therefore requires **head_dim == D_model, i.e. H == 1**. Measured
over the shipped case set: only **`l4_h1_hd128`** (H=1, head_dim=128=D_model)
satisfies it. The other five are H=2/head_dim=64/D_model=128 — at `CFG_D=128`
`rope_row` would frame a 128-beat row as ONE head (pairing i with i+64) where
golden pairs within each 64-wide head; at `CFG_D=64` the h8 row and the K=128
projection contraction no longer fit one act row.

| case | H | head_dim | D_model | head_dim==D_model | bias | drivable @CFG_D=128 |
|---|---|---|---|---|---|---|
| l4_h2_hd64 | 2 | 64 | 128 | False | — | **False** |
| **l4_h1_hd128** | 1 | 128 | 128 | True | — | **True** |
| l4_gqa_h2kv1 | 2 | 64 | 128 | False | — | **False** |
| l4_qwen_theta | 2 | 64 | 128 | False | — | **False** |
| l4_bias | 2 | 64 | 128 | False | yes (gap B) | **False** |
| l4_selfinc | 2 | 64 | 128 | False | — | **False** |

**This is not a defect — it is the stage-6 envelope showing up early.** The
architecture already separates these widths for the real geometry (7B:
D_model=3584, head_dim=128): §4 stage 6 carries `RMS_D_MAX=3584`, the **wide
`seam_feeder_quant` elaboration**, and the **C-KSPLIT decision** precisely
because a `K = D_model` contraction cannot ride one `CFG_D = head_dim` act row.
Multi-head L4 cases need the same machinery (`CFG_D=64` + a 2-way K-split of the
128-wide contraction), so **the H>1 cases are gated behind the stage-6 C-KSPLIT
realization**, not behind new capability.

**Consequence for O3′-1: scope it to `l4_h1_hd128` at `CFG_D=128`.** That still
closes the **h8 rider golden-exactly on a real shipped case** (h8 → K projection
→ MODE_F16 narrow → `rope_row` → `dbg_f16_*` tap == `r.K_rope`) and gives
`rope_row` its **first in-tile composition coverage** — both firsts for this
lane. The remaining five cases join when C-KSPLIT lands (stage 6), and are
tracked there rather than as a new gap. **Recommendation: proceed on
`l4_h1_hd128` and state the 1-of-6 scope in the TB banner and the S5 row; do not
re-scope the case set.**

**Segment 2 — front-half `xa→rmsnorm→feeder→astage` — LANDED GREEN
(2026-07-26).** `verif/top/l4/tb_l4_compose.sv` + `compose_seg2.ops` drive all
six L4 cases' RMSNorm-1 front-half through the REAL apex_top (CFG_D=128, the
L3 phase-B loader idiom: XR/GR → RMSNorm-1 → feeder C-1) and check the feeder
scale bus `fs == r.s_h` per row (66 EFS checkpoints, `make -C verif/top/l4
compose` → `L4COMPOSE RESULT: checks=67 errors=0 → PASS`). Generator
`seg2_selfcheck` re-derives `x8 = quant_rows_i8(f16(X))` and asserts the
front-half reproduces `r.h` AND `r.s_h` bit-exactly before any RTL (public
fns, zero golden edits). **h8 bit-exactness is a DISCLOSED CARVE-OUT deferred
to segment 1** (the o-proj GEMM consumes the h8 codes vs the re-derived o8, so
any h8 error breaks the o8/r1 match — a real indirect check). Mutation gate
(`make -C verif/top/l4 mutate`, `mutate_compose.py`, l3 signature-required
discipline — never any-nonzero-exit): 3 front-half mutants at distinct
integration wires — mC1 (RMSNorm→widen data halved), mC2 (gamma wire halved),
mC3 (feeder job gated off, hang) — each caught only on its `[EFS]`/stall
signature; the patched-copy build never edits the RTL.

**Walker-mode step-completion arbitration (this harness IS the arbiter —
cross-lane ruling from the combine flip, `comp/ib-combine`, recorded here as
one truth).** IB-WALK §2.3 assumed a "norm job port" that this lane never
publishes. **RATIFIED: there is NO norm unit** — RMSNorm-1/2 are HOST-STREAMED
(§3b memories; L1/L7): the host streams `xa`/`xg` into `u_rms` (no
`LAYER_JOB` unit, no `l_*` level), exactly as segment 2 drives it. The
walker-mode NORM step therefore consumes the existing **`dn_rms` done event**
(observed while the norm row is presented) as its completion — ratifying the
flip's provisional tie; no new port supersedes it. **`hd_ready` (per-head
q-staging): PROVISIONALLY ratified** as the `ss_*` `s_q` snoop event the flip
used (q-staging is the projection→scale_quant path, not touched by the
front-half). Its DEFINITIVE arbitration lands with segment 1, which exercises
the q-projection→`ss_*` `s_q` path and will confirm or supersede here.

### 3d. IC-BIAS — the projection-bias path (I-C gap B), 2026-07-26

**Status: LANDED on `comp/ic-bias`.** This subsection is an ADDITIVE amendment
to the frozen §3b table (B1 §A-1 rule: table + glue + walker ROM move
together). It adds ONE level bit, ONE `LAYER_PTR` bank encoding and TWO error
codes; **no existing §3b field, address, bit position or code changes**, and
the walker ROM is unaffected because the walker drives no new port (see the
walk-mode fence below). Bit allocation was chosen to leave `LAYER_CTRL[7]` —
the natural next bit, and the one the IC-QPATH q-sink level is expected to
take — free.

**Why a bias path is needed (the gap, restated at source).** Qwen2/2.5 puts
biases on q/k/v only. `decoder_layer_fx` adds them in REAL units to the
DEQUANTIZED projection output and BEFORE the single fp16 bus narrowing —
`transformer.py:506-508` (`acc·comp`), `:509-514` (`+ bq/bk/bv`), `:521-522`
(`_f16` on q and K under `rope_in_f16`), `:541` (the same single narrowing on
V at the attention-core input). So the only bit-exact insertion point is
INSIDE the exact product, between `acc × composite` and the one RNE. Adding
the bias to an already-narrowed `f16(acc·c)` is a double rounding; folding it
into the INT8 weights or an extra activation row leaves the int8 grid. Both
were rejected in §3c-1 and remain rejected.

**WHERE it enters — an additive glue block, not an edit to a verified one.**
The exact 36-bit product lives inside `apex_scale_quant`, which is a VERIFIED
block carrying a published bit-exactness proof (P1–P5) and a large suite.
Rather than open that proof, `rtl/top/glue/apex_proj_bias.sv` is a **SIBLING on
the same S-2 seam**: same job / sideband / value contract, the same exact
25×11 product, the fp16 bias summed in on the common exact grid, and ONE RNE
through the already-verified `f16_arith_pkg::f16_pack_real` (the S2 primitive
swept by the resfx/rope suites). The two are selected per job by a route level
and their `f_*` outputs are muxed onto `sqf_*` upstream of the rope mux, so
`rope_row`, the KVQ store, the B1 store-snoop and the `dbg_f16_*` tap all see
exactly one producer. **The duplication is bounded and gated:** with a `+0`
bias vector `apex_proj_bias` is bit-identical to `apex_scale_quant` MODE_F16
on every legal element *and* on both contract-violation behaviours (C1 clamp,
C2 bad-composite ⇒ `16'h0000`), and the unit suite co-simulates the two blocks
and requires that identity beat-for-beat.

**Numerics (the exact-or-refused contract).** With `v` the INT32 accumulator
(C1: `|v| ≤ 2^24`), `c` the fp16-grade positive-normal fp32 composite (C2) and
`b` the fp16 bias element:

    x = v·c = ±P·2^ex,  P = |v|·m11 < 2^35,  ex = ec − 137      (exact)
    b       = ±B·2^-24, B < 2^40                                 (exact)
    g   = min(ex, −24);  ACC = ±(P << (ex−g)) ± (B << (−24−g))   (exact)
    y   = f16_pack_real(sign(ACC), |ACC|, g)                     (ONE RNE)

Two refusals, both LOUD (`window_error` pulse+sticky, job aborts, remaining
beats drained — the `apex_layer_deq` / `apex_residual` idiom):
* **W1 alignment window** — `ex ∈ [−40, −3]`, i.e. composite in `[2^-30, 2^7]`,
  which keeps both shifted operands under 2^56 so their sum fits the 57-bit
  `f16_pack_real` significand with no bit lost.
* **W2 float64 realizability** — golden performs the bias add in float64, so a
  sum needing more than 53 significand bits would be ROUNDED there and a
  single-RNE datapath could then differ. The span of the exact sum is measured
  (msb−lsb, the `apex_layer_deq` idiom) and a span > 53 bits REFUSES. Inside
  the window golden's add is provably exact, so "one RNE here" equals "golden's
  f64 add then `f64_to_f16_bits`" bit-for-bit — and the generator ASSERTS that
  exactness with `Fraction` arithmetic for every emitted vector rather than
  assuming it.
**Measured margin over every element driven in the tile gate (l4_bias q/K/V,
2,176 elements): 13 binades below the low edge and 23 above the high edge —
refusal is not reachable in contract**, the same style of measurement §3b
records for `apex_residual` (+19 bits).
Zero sign: `x` is never −0 (`v` is an integer, `c` positive normal), so
golden's "−0 only from (−0)+(−0)" can never fire and an exactly-zero sum is +0.

**Control surface (delta to the §3b table only).**

| field | delta |
|---|---|
| `LAYER_CTRL[18]` **l_bias_en** | RW, RESET 0. 1 ⇒ a MODE_F16 (S-2) job pushed on the scale_quant job port runs on `apex_proj_bias` with the staged bias vector. MODE_QUANT (Q7 / S-4 P-requant) ALWAYS runs on `apex_scale_quant`, so the attention path is untouched while a biased layer is in flight. Reads back at `LAYER_CTRL[18]`; reserved-0 in a `PROJ_BIAS_EN=0` build. `[7]` and `[19]`+ stay free. |
| `LAYER_PTR[29:28] = 3` | the projection-bias RAM (banks 0/1/2 unchanged). `LAYER_DATA[15:0]` = ONE fp16 bias per write, auto-increment — the identical load-port shape the phase-K / phase-q / residual banks use. Depth `LAYER_DM_MAX`. |
| `LAYER_STATUS[4]` **bias_busy** | RO; was part of the `[7:4]` reserved field. Reserved-0 in a `PROJ_BIAS_EN=0` build. |
| `LAYER_STATUS[12:9]` codes **6 / 7** | 6 = `BIAS_WINDOW` (W1/W2 exact-or-refused), 7 = `BIAS_CONTRACT` (C1 range / C2 grade monitors). Codes 3/4/5 untouched; the bias block's own `job_error` joins code **2**, matching the existing convention for the other LAYER units (`apex_layer_deq` / `asu_swiglu` job rejects already report 2, not 1 — code 1 stays the glue's push-while-pending / illegal-unit event). New-block errors land in `LAYER_STATUS[8]` and **never** `err_sticky[15]` (the B1 rule) — the tile gate checks `ERR_STICKY` stays 0 across every refusal arm. |

**Staging protocol.** The bias RAM is read by the element index WITHIN the
job, so the host stages the SLICE for the row being projected at index 0. All
`T` token rows of one KV head share one bias slice, so a layer stages `bk`,
`bv` and `bq` once each (128 `LAYER_DATA` writes per tensor at the tile
geometry). Choreography: poll `LAYER_STATUS[4] == 0`, write `LAYER_PTR`
`{bank=3, idx=0}`, stream the fp16 words, then set `LAYER_CTRL[18]` and push
the S-2 job — the §3b "levels while quiescent" rule, unchanged. A per-tensor
BASE OFFSET (so several slices can be resident) is the obvious future
extension and is deliberately NOT built here.

**Format.** fp16, one word per output channel. This is not a choice — it is
the golden contract: `transformer.py:348` states the biases are "in REAL units
on the fp16 grid", and `run_tinynpu.py:216-219` casts the real Qwen2.5
checkpoint bias to `float16` before golden ever sees it. The generator asserts
`f16_bits_to_f64(bits) == w.bq` before emitting anything.

**Elaboration gate.** `apex_top` parameter `PROJ_BIAS_EN` (default `1'b0`).
At 0 the whole region is one `generate` else-branch of straight-through
assigns: no bias RAM, no adder, no route mux, `LAYER_CTRL[18]` /
`LAYER_STATUS[4]` reserved-0 and the four `pb_*` error wires tied 0 — every
existing build is byte-identical (proved by re-running the full gate set, both
lints, and both existing lint build points).

**Walk-mode fence (explicit — this lane does NOT build walker support).**
There is no walker port for `l_bias_en` and none for the bias RAM. In walk
mode the level HOLDS its host-written value (same shape as the GQA
`l_kv_map` register), and the RAM is loaded through `LAYER_PTR`/`LAYER_DATA`
exactly like the phase-K/phase-q/residual banks — which are themselves
host-loaded today. So a walked biased projection is reachable the moment
D-029 gains a bias level in its LVL step and the memories gain their walker
load path; that is IB-WALK's surface, not this lane's, and nothing here
pre-empts it.

**Gate (this lane's suite, `verif/top/bias/`).** `unit` — 13 jobs / 2,037
checks / 0 errors over `apex_proj_bias` co-simulated with the real
`apex_scale_quant` (operating regime, `+0`-bias equivalence, directed corners
and constructed RNE ties, window edges with 6 alignment + 1 span-53 refusals,
a C2-monitor arm). `tile` — the `l4_bias` case's q/K/V projections through a
real `apex_top` at `CFG_D=128`: 2,325 checks / 0 errors, of which 2,176 are
biased S-2 beats compared to `f64_to_f16_bits(r.K_real / r.V_real / r.q_real)`
under BUS_ON, 128 are the bias-disabled discrimination arm (the tile
reproduces the UNBIASED narrowing exactly, and it differs from the biased
arbiter in **128/128** elements), plus the refusal/legality arms. `mutate` —
4/4 signature-required mutants at four distinct integration points.

**Scope note vs §3c-3.** O3′-1 was scoped to 1 of 6 cases by GAP D because
`rope_row` frames a row as one head. This gate runs with `l_rope_en = 0` and
the bias+narrow seam is PER ELEMENT, so GAP D does not bind it: all of
`l4_bias`'s q/K/V rows (`kv_dim = D_model = 128 = CFG_D`) are drivable today.
**What is still NOT closed for q**: gaps A and C (no narrow-AND-rope sink to
the act stage) are IC-QPATH's; this lane proves the biased, narrowed q ROW is
produced bit-exactly at the S-2 seam, not that a rotated, quantized q reaches
the attention core.

---

## 4. Staged landing plan (golden-first, machine-aware)

| stage | what | machine need | gate (no PASS without pasted output) |
|---|---|---|---|
| **0 ✅ (this doc)** | Lane contract; reality checks §1 (file-level, commands named); CSR/route proposal §3; composition map §2 | none | n/a — no runs claimed |
| **1 🟡 LANDED, sign-off pending** | §2b golden bus-composition addendum (additive; flags default OFF) + its tests; owner sign-off recorded in this doc; `verif/top/l4/gen_l4_vectors.py` skeleton emitting the full-layer host op stream from `decoder_layer_fx(+bus)` intermediates (L3 generator pattern — new sibling file, no golden edit). **Done 2026-07-22:** C-LBUS/D-030 mode + `layerbus` gate + pinned-delta doc (`docs/results/ib_layer_bus/RESULTS.md`) + checkpoint-emitting skeleton (op stream = S4 stub). **Gate closes only on the §0.1 R4 owner sign-off.** | edit only (Python) | `make -C golden test` — banner byte-identical for `gen_status.py`, new tests PASS; flags-OFF bit-identity test PASS — **rc=0 evidence pasted in RESULTS.md §3** |
| **2 ✅ DONE 2026-07-22** | `f16_arith_pkg` + `rope_pair_fx`/`rope_row` + `residual_add_fx` RTL + unit TBs. **Evidence:** resfx — 238,091 golden-fp vectors + 2,000,000 in-sim pairs co-simulated vs behavioral `residual_add`, 0 mismatches, 3/3 mutants; rope/row — all 2^14 phase codes swept, 646-row D=64 + 368-row D=128 sets (gen self-checked vs `rope_fx` incl. theta=1e6 tensors), every beat vs golden file AND a shadow behavioral `rope.sv`, 5 regimes + 37 mid-op resets + frame paths, 0 mismatches, 6/6 mutants; lint `-Wall` clean, no waivers. **Mutant relocations measured, not waived** (overflow checks call-site-equivalent — pkg documents the carry-encoding identity; subnormal tie moved to rope where FRAC=38 reaches it). **RE-SCOPE (recorded): `apex_residual` (row-RAM wrapper) moved to S4** — its contract IS the LAYER window's (LAYER_PTR/RDATA banks, in-place r1 update); building it before the window shape froze would duplicate S4; the S2 numeric risk it carries (`residual_add_fx`) is fully retired above. | Verilator (serialize) | unit suites green — pasted; ALL mutants killed; lint clean |
| **3 ✅ DONE 2026-07-22** | `apex_layer_deq` (int32 × graded-f32 → EXACT fp32, C2 legality, per-element **exact-or-refused**) + `asu_swiglu` (clamp(RNE(acc·comp·2^10)) → verified `asu_silu` → one f16 RNE; exact ≤54-bit product → one f16 RNE — the D-030 lemma in hardware) + `verif/asu/swiglu/`. **Evidence:** 7 regimes green — deq 26 jobs/4,138 beats (exhaustive 256-code × 16-composite o8 sweep, over-range/grade/frame refusals, resets), swiglu 23 jobs/289 products vs golden `silu_apply`·up (constructed Q5.10 + product RNE ties in BOTH keep parities, exact-clamp battery incl. the 8192/idx-255 invariant, inf saturation, zeros, resets); mutation gate **9/9** (refusal bypasses detected via watchdog kill — timeout=KILL, w4 precedent). S1's first-cut survival was measured (mass gates saturated in the clamp; ties lacked an odd keep) and fixed by construction, then killed. | Verilator (serialize) | unit suites green — pasted; 9/9 mutants; lint clean |
| **4 ✅ GLUE LANDED 2026-07-22 (datapath proof = S5)** | `apex_top` glue — fenced additive regions only: LAYER CSR window (0x70-0x8C), route extensions (§3), block instantiations (**now incl. `apex_residual` — the S2 re-scope — and the R3-amended per-KV-head CQ-8 engine banking, mapping `h // (H/H_kv)`, H/H_kv=7 non-pow2**), `RMS_D_MAX`/`LAYER_D_MAX` parameters (defaults 128); `apex_sources.f` additions. **Plus (ruling R5/R1 cross-lane deliverable): publish the frozen job-port/route table in this doc the moment the glue shape firms — IB-WALK's fmt=1 (D-029) walker ROM is BLOCKED on it.** Gate addition (FUEL lesson, §0.1): warnings-fatal lint on all glue + a directed level-propagation probe per exported route level. **EVIDENCE (2026-07-22):** frozen §3b map implemented as a NEW parallel region (B1 WALK region untouched; merge-shape directive honored); `-Wall`-fatal lint CLEAN in both tile configs; propagation probe `verif/top/l4/tb_l4_levels` **27/27 checks** (every level at its consumer, memories round-tripped, GRADE refusal + W1C through the window); byte-identity vs stashed-pre-edit baselines — smoke (3 configs), l2, **l3 host AND walker (27 cases, split 24/27+3 unchanged)**, seq_walker: ALL RESULT lines BYTE-IDENTICAL; unit suites re-gated green on the final pkg. **Two probe-earned lessons recorded:** LAYER reads are 1-CYCLE (the WALK pattern — the F2 executor must sample in the strobe+1 window or it reads 0xDEADBEEF), and CSR TB stimulus must be negedge-driven (same-slot blocking drives race the DUT sample). **Deferred out of this delta, explicitly:** apex_residual DATApath bit-exactness (S5 L4 runs; its add core is the S2-gated residual_add_fx; S4 gates wiring + window structure) and the R3 per-KV-head engine banking (S4b before combine, with its probe) — **S4b LANDED 2026-07-26, record + flip instructions in §0.1; suite/probe evidence `verif/kvq/gqa/RESULT.md`**. | Verilator (serialize) | **compat proof, S10/B1-stage-4 style:** `verif/top/smoke` + `verif/top/l2` + `verif/top/l3` in HOST mode AND walker mode byte-identical on cycles/checks/errors across all cases; `verif/seq_walker` unchanged; every count identical to pre-stage-4 |
| **5 🟡 IN PROGRESS** | `verif/top/l4/` — host-mode full-layer composition suite through the REAL `apex_top` (details §5); tile-level mutation adds. **LANDED 2026-07-24: the apex_residual DATAPATH closure (the S4 deferral) — `verif/misc/resfx` resid suite: all six BUS_ON L4 cases' r1 AND r2 chained in-place BIT-EXACT vs the signed D-030 arbiter (1,551 elements), ±0/window/inf-boundary/frame/geometry corners, 3 regimes incl. clean-restart reset, 4/4 mutants (RM2 killed by the engineered e8=143 finite-32.0 shortcut distinguisher). LANDED 2026-07-26: §3b/§3c L4-harness spec authored; `gen_l4_vectors.py` segment-1 o-proj re-derivation + MANDATORY pre-RTL self-check (re-derived chain reproduces r.attn_proj AND r.r1 bit-exactly, 6/6 cases, zero golden edits) + per-case `.seg1` references. REMAINING (see §3c): tb_l4_compose tail drive — segment-1 o-proj needs the attention back-half for its `attn` input (drive-path finding, §3c, owner review); segment-2 front-half is the tractable first slice; 3 tile mutants, coverage, byte-identity re-check.** | Verilator (serialize) | all L4 cases "L4 PASS", per-case check counts printed; bit-exact vs the §2 arbiter at every checkpoint; L4 mutants 3/3; coverage gate; existing suites still byte-identical |
| **6** | Wide geometry: `RMS_D_MAX=3584` L4W build (both norm sites on 3584 rows), wide `seam_feeder_quant` elaboration (whole-row C-1), C-KSPLIT job realization decision (host-summed partials vs OS `accumulate` — `compute.py:44-48` blesses both as identical integer math); **synth probes** (house sv2v→yosys flow, the `verif/asu/wide` RESULT.md pattern) for every wide elaboration. **The feeder probe is already DONE (§1b, pulled forward under R6: +6 RAMB18E2 +1 DSP48E2 ~+0.1k LUT vs D=128, BRAM inferred as-is)** — remaining here: the L4W sim build, the KSPLIT decision, and probes for any OTHER wide elaboration stage 4 introduces | Verilator + yosys (serialize) | L4W cases bit-exact; probe cell tables pasted; **timing claims only ever READ from the I-B Vivado build report (integration-owned) — this lane publishes no timing number** |
| **7** | Handoff: IB-WALK phase/descriptor input inventory — **RECONCILED 2026-07-23 against the frozen §3b memories (combine-agenda item): "per-step phase rows" is STRUCK.** The phase-K table is RESIDENT per configuration (`[T_ROW_MAX][CFG_D/2]`, loaded once; the q row at position p IS table row p, so steady-state decode loads NO phases). Per-STEP inputs are exactly: 3× rq pairs (PV / o-proj / down-proj), the per-job graded JOBC composites, and the `LAYER_CTRL.l_rope_pos` update; the residual row is loaded at layer-0 entry only (r2 persists in the row RAM as the next layer's X). Per-CONFIG: phase tables, CTRL levels. Plus paste-ready `gen_status.py` suite row + STATUS/OPTIMIZATION snippets FOR INTEGRATION TO APPLY; ARCHITECTURE decision-entry proposal (next free D-number at landing time) | none | doc-complete; integration acknowledges receipt of the two shared-file snippets |

Stages 2-6 serialize behind the shared Verilator queue (check
`ps aux | grep -cE "[v]erilator|[o]bj_dir/V"`); the c6a verify box
(`scripts/aws/apex_verify_box.sh`, LEVEL_C §6) is the overflow valve — spend
per line needs owner approval. Stages 0/1/7 are edit-only.

---

## 5. Verification target (the L4 suite)

**Shape:** copy the L3 harness discipline (`verif/top/l3/` — generator emits
op streams + expectations from the golden arbiter; TB drives/collects/
compares; `run_cases.py`-style mechanical pass criterion; coverage +
mutation gates). New dir `verif/top/l4/` — own files, zero edits to the L3
harness.

**Cases (tile scale first — the CAMPAIGN §3-§4 "tile scale" qualifier
applies to every claim):**

1. `layer_h2_hd64` and `layer_h1_hd128` — the two `verif/layer` anchor
   configs (H=2/hd=64/d_ffn=256/T=8; H=1/hd=128/d_ffn=344/T=16), now through
   the real tile.
2. `layer_gqa` — H_kv < H (S6 grouping choreography).
3. `layer_qwen_theta` — **rope_theta_base=1e6** (the Qwen2/2.5 config value;
   parity tested, not assumed).
4. `layer_selfinc` — S8 self-inclusive q_pos composition (duplicated decode
   row).
5. `layer_bias` — Qwen bq/bk/bv non-None (bias-before-narrowing ordering).
6. L4W (stage 6): both norm sites at D=3584 through the wide elaboration;
   full-layer-at-3584 only if sim cost permits — measure, decide, record.

**Bit-exact checkpoints per case** (vs `decoder_layer_fx` with the §2b bus
mode ON, via existing taps + the LAYER read-back window): `h` rows; `h8/s_h`;
rotated q̂/K̂ and unrotated V̂ fp16 beats (`dbg_f16_*`); score/PV streams
(existing L3-grade checks); o-proj `o8`; `r1` row; `h2`; gate/up accs;
SwiGLU `p` products; down-proj `o8`; `r2` row (the layer output).

**Existing-suite byte-identity (the non-negotiable):** every pre-existing
suite's counts byte-identical after every RTL-touching stage — `verif/top/
smoke`, `l2`, `l3` (host AND walker mode, unchanged check counts — the D-028
acceptance is not allowed to degrade), `seq_walker`, `asu/*`, `mxe/*`,
`kvq/*`, `seam`, `rope/smoke`, `asu/silu`, `residual`, `layer`, golden.
`verif/f2sim` runs at combine (integration's G-I4-class gate); this lane
runs it opportunistically if the queue allows.

**Mutation obligations:** every new unit TB ≥2 killed mutants (named in §4);
tile-level L4 adds ≥3: (m-L1) `lr_rope_en` decode stuck (unrotated K̂ must
break the `dbg_f16` checks), (m-L2) residual bank select inverted (r2
written from stale row), (m-L3) SwiGLU hold-file index off-by-one. The
existing L3 3/3 tile gate must stay green with the layer path compiled in.

---

## 6. NOT in scope (explicit)

- **Walker/descriptor work** — extending D-028 to walk the layer, new WALK
  phases, descriptor format deltas, MMIO/step accounting: **IB-WALK**. This
  lane keeps walker-mode L3 byte-identical and delivers the stage-7 input
  inventory; it does not touch `seq_layer_walker.sv`, `seq_walker_comp.sv`,
  or `seq_walker_pkg.sv`.
- **DDR / DMA / BAR4 / `sh_ddr` / `cl_ddr4` / f2sim DDR modelling**: **IB-FUEL**.
- **Hardware/AWS work**: no Vivado builds, no AFI, no F2 runs, no devbox;
  timing numbers are read from integration's build reports only. Stage-6
  yosys probes are local mapping evidence, not P&R/timing.
- **Model evals / token-quality claims**: none. Composition claims are
  bit-exact-vs-golden only; "whole-7B-on-FPGA" ladder wording is
  integration-owned and stays banned until evidence (LEVEL_C §9).
- **Frozen/foreign files**: `rtl/apex_pkg.sv` (frozen), `rtl/csr/
  csr_regs.sv`, `rtl/kvq/*` (B2 lane), `rtl/mxe/*` datapath, existing golden
  functions (additive-only per §2b), `scripts/gen_status.py`,
  `docs/design/LEVEL_C_*.md`, `docs/OPTIMIZATION.md`, `docs/STATUS.md`
  (integration owns; snippets handed over at stage 7).
- **B3/W4**: no W4 enablement, no weight-path edits; the fp16-grade
  composite constraint is consumed as a boundary condition, not reopened.
- **KVQ-tier scope**: L4 v1 composes at CQ-8 (the S8/B1 verified tier);
  grouped-tier layer composition inherits the B1b/S12 tracks.

---

## 7. Files to read / touch & repro

**Read (do not modify):** `golden/apex_golden/{transformer,attention,compute,
fp}.py` · `verif/layer/*` (composition anchors) · `rtl/rope/rope.sv`,
`rtl/asu/asu_silu.sv`, `rtl/misc/residual_add.sv` (behavioral references) ·
`rtl/top/apex_top.sv` (WALK window `:367-431`, ERR_STICKY `:1160-1200`,
seam muxes `:781-784, 1047-1082`) · `rtl/seam/seam_feeder_quant.sv` ·
`rtl/top/glue/apex_scale_quant.sv` (C2 scale grade) ·
`rtl/seq/seq_walker_comp.sv` (fp16-in-integer proof pattern) ·
`docs/design/{B1_WALKER,B3_WEIGHT_PATH,WIDE_RMSNORM,LEVEL_C_INTEGRATION}.md`
· `verif/asu/wide/RESULT.md` · `verif/top/l3/*` (harness pattern).

**Create (add-only, this lane's own paths):** `rtl/misc/f16_arith_pkg.sv`,
`rtl/rope/rope_row.sv`, `rtl/misc/residual_add_fx.sv`,
`rtl/top/glue/{apex_residual,apex_layer_deq}.sv`, `rtl/asu/asu_swiglu.sv`,
`verif/rope/row/`, `verif/misc/resfx/`, `verif/asu/swiglu/`,
`verif/top/l4/`, golden addendum + tests per §2b. **Touch (fenced):**
`rtl/top/apex_top.sv` (stage-4 glue regions only),
`scripts/fpga/f2/cl_apex/design/apex_sources.f` (file-list additions).

**Repro (from the worktree root):**
```
make -C golden test                      # arbiter gate — must stay byte-identical
make -C verif/layer                      # standalone composition anchors still green
make -C verif/top/l3                     # host-mode L3 — byte-identity reference
make -C verif/top/l3 run_walker          # walker-mode L3 — unchanged counts
# stage 5+:
make -C verif/top/l4 all                 # to be added by this lane
```

---

## 8. Open design questions — ALL THREE RULED (LEVEL_C §9.1, see §0.1)

1. ~~**The §2b bus-composition addendum needs an owner-gated numeric-contract
   decision.**~~ **RULED (R4):** design-approved as **D-030**, B3-finding-3
   pinned-delta measurement is the evidence form; **owner sign-off is the S1
   gate and is PENDING** (§0.1 holds the record slot). S1 implementation
   landed 2026-07-22; two of the four points dissolved into a tested
   exactness lemma (§2b S1 note).
2. ~~**Control ingress: glue CSR window vs new `rt_*`/job input pins.**~~
   **RULED (R5):** glue CSR window CONFIRMED, zero new `apex_top` ports;
   single glue-level mux point, B1 stage-5 idiom — the IB-WALK walker later
   drives the SAME glue registers (one ingress point).
3. ~~**The wide feeder is the unpriced elaboration.**~~ **RULED (R6) and
   EXECUTED:** probe pulled forward, results §1b — +6 RAMB18E2 +1 DSP48E2
   ~+0.1k LUT vs the D=128 CL build point, row buffer infers BRAM as-is.
   Chunked-amax fallback (two-pass over 28×128-beat chunks, exact same amax
   by max-associativity) **stays speced as the named fallback** but is not
   needed on area evidence. Timing remains build-report-only.

**Standing next question for the lead (new, from S1):** the rope-absorption
finding (`d(rope)=0` e2e with 33/254 q̂/K̂ bus words differing, RESULTS.md
§2.2) means e2e checks alone would never catch a broken `rope_row` at these
operating points — confirm the L4 acceptance keeps the q̂/K̂ **bus
checkpoints** (dbg_f16-level) as first-class gates, per §5.
