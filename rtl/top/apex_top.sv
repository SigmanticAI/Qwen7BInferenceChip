// apex_top.sv — the APEX attention tile, v0.1: every verified block wired
// into the ARCHITECTURE.md §0 dataflow.
//
//   x8 -> [ASU rmsnorm] -> [widen] -> [feeder quant] -> [MXE WS proj]
//      -> [scale_quant F16 (S-2 write port)] -> [KVQ compress]
//   KVQ readback -> [feeder quant] -> [MXE OS Q·K̂ᵀ] -> [score dequant]
//      -> {TIP tap, ASU softmax} -> [scale_quant QUANT (S-4 P-requant)]
//      -> [MXE OS P·V̂ + C-2 epilogue] -> o8 out
//
// Verified blocks are instantiated UNMODIFIED: csr_regs, seq_walker, mxe_top,
// kvq_engine (D-015..D-020 hardened), asu_softmax (f10 config), asu_rmsnorm,
// tip_top, seam_feeder_quant, seam_score_dequant, stream_skid. New glue lives
// in rtl/top/glue/ with per-file golden contracts (apex_q78_to_fp32,
// apex_scale_quant, apex_lane32_ser, apex_stage_buf, apex_score_fork).
//
// ══ v0.1 SCOPE BOUNDARY — host-sequenced tile (READ THIS FIRST) ═══════════
// The tile AUTOMATES: all §5 stream transport between blocks (every seam
// skid-buffered), all numerics (bit-exact against golden/apex_golden/
// attention.py under the documented contracts), MXE job serialization via
// SEQ (D-006), CSR status/PERF aggregation, and the KVQ store/readback path.
// The HOST DRIVES, phase by phase (this is the v0.1 boundary; a hardware
// layer-walker that issues these itself is v0.2):
//   1. MXE descriptors (via the SEQ ds_* stream) and projection WEIGHTS via
//      xw_* — v0.1 has no on-tile weight memory.
//   2. Glue job commands (fj/qj/dj/lj/aj/wj_*) and the rt_* route levels
//      (levels; change only while the touched streams are quiescent).
//   3. The seam SCALE COMPOSITES: per D-021/S-3 the single f32 narrowing of
//      each composite happens in "the integration glue / SEQ" — in v0.1 the
//      host IS that glue: it reads the fp16 scale taps (fs_*/ss_*) and feeds
//      back f32 composites on qs_*/cs_*. Tensor DATA never leaves the tile
//      between stages; only scale METADATA crosses the host loop.
//   4. KVQ control through the tier bank's AXI-Lite window (kv_*): enable,
//      WRITE_ADDR/READ_ADDR sequencing, and D-020 soft reset — PER ENGINE
//      (the window routes to engine[live tier]; the host enables/resets
//      each tier it uses). CSR CTRL.soft_reset drives ONLY the SEQ abort
//      contract (§7 names exactly SEQ + KVQ as consumers; each KVQ
//      engine's D-020 reset is its own CTRL register, host-issued).
//   5. Epilogue calibration: descriptor rq_scale/rq_shift are host-supplied
//      (golden calib_requant is deterministic; calibration is a host job).
//
// ── KVQ TIER (F-2/D-022 CLOSED, D-024) ─────────────────────────────────────
// The KVQ subsystem is apex_kvq_bank: THREE verified kvq_engine instances
// (CQ-8 / CQ-4 / CQ-4+) behind a live tier mux. The live tier is:
//     live_tier = TIER_CTRL.tip_override ? auto_tier[rt_tip_blk]   (AUTO)
//                                        : TIER_CTRL.tier_sel      (HOST)
// where auto_tier is a 128-entry (one per TIP block) register file, reset
// to CQ-4 (the §4 default) and written by every ACCEPTED TIP decision beat
// (td_* handshake: auto_tier[td_blk] <= td_tier) — TIP DRIVES the tier
// (D-022 actuation). GRANULARITY (honest): the tier select is quasi-static
// per KVQ transaction run — per token for value stores, per key GROUP for
// grouped-key stores, per record for reads; switch only while the bank is
// quiescent. In AUTO mode the host addresses a block's storage by setting
// rt_tip_blk (per-block granularity). kvq_engine's own TIER stays a
// synthesis parameter — runtime control is by BANKING verified engines.
//   * Keys at CQ-4/CQ-4+ take the grouped KEY path: rt_kv_user=0 routes
//     s_axis tuser=0; program the engine WRITE_ADDR to the group base,
//     stream G tokens back-to-back, CSR FLUSH (D-008) closes a partial
//     group. Values (and CQ-8 keys) use rt_kv_user=1 (value path).
//   * CQ-4+ outlier lane: KVQ_OUTLIER_K/KVQ_MASK_FILE tile parameters feed
//     engine 2. With KVQ_OUTLIER_K=0 engine 2 degenerates to CQ-4 and the
//     CSR INFO_TIER bitmap truthfully drops the CQ4P bit: INFO_TIER =
//     {OUTLIER_K>0, 1, 1} — the build's TRUTH, never a hardwired 0x7.
//     The per-engine truth remains readable via kv_* INFO_TIER (0x0C).
//
// ── CSR wiring (§7) ────────────────────────────────────────────────────────
//   CTRL.enable        -> SEQ enable (gates new MXE dispatches)
//   CTRL.soft_reset    -> SEQ abort_req (D-020-lineage abort; see boundary)
//   STATUS.idle        =  ~|block_busy (see lanes below)
//   STATUS bit1 sticky <-  OR of EVERY error pulse in the tile (desc_error,
//                          job/row/range/scale/frame errors) — the per-source
//                          stickies are individually visible on err_sticky.
//   TIER_CTRL          -> live KVQ tier select + tip_override (D-024 block)
//   THRESHOLD_REG      -> TIP threshold (D-011/D-017)
//   FLUSH              -> KVQ flush_req (D-008; routed to engine[live tier])
//   IMPORTANCE_BASE    <-> TIP imp_rd_* window
//   ERR_STICKY (0x58)  -- TILE-window extension, implemented in integration
//                         glue below (csr_regs acks 0x58 as reserved; this
//                         file owns the data — F-3 closure). READ: the
//                         16-bit per-source sticky bundle (same bits as the
//                         err_sticky pins). WRITE: W1C per bit; a same-cycle
//                         error pulse WINS over the clear (never lose an
//                         error). Bit 14 W1C additionally pulses TIP's
//                         frame_err_clear so the block-internal sticky
//                         clears coherently. The per-source stickies are
//                         LATCHED HERE from the blocks' error pulses (the
//                         blocks' internal rst_n-only stickies are no
//                         longer exported — they have no clear pins).
//   PERF busy lanes    :  0 SEQ · 1 MXE · 2 seam/glue (feeder|scale_quant|
//                         serializer|stage bufs|fork) · 3 score-dequant ·
//                         4 ASU softmax · 5 RMSNorm · 6 TIP · 7 KVQ read-
//                         side (m_axis beat pending). KNOWN v0.1 GAP: the
//                         KVQ lane covers only the read/output side — the
//                         engine exports no busy pin and is not modified
//                         here (verified as-is); its input-side busy is
//                         host-visible via its own STATUS.idle (D-020:
//                         "can never lie"). Poll BOTH before trusting
//                         whole-tile idleness.
//
// §5: every inter-block stream crosses at least one V0.5-verified
// stream_skid — the verified blocks own skids on all their ports, and every
// new glue block does too; the route muxes below sit BETWEEN two skidded
// endpoints and add no unregistered path across a block boundary.

module apex_top
  import apex_pkg::*;
  import seq_walker_pkg::*;
  import f16_arith_pkg::*;   // IB-LAYER S4 (f16_to_f32_bits widen)
#(
  parameter int unsigned CFG_D         = 64,   // PER-HEAD row length (KVQ
                                               // record / rope / walker
                                               // attention head_dim)
  // I-C GAP D (IB_LAYER.md §3c-3): CFG_D was ONE parameter feeding TWO
  // incompatible width families — the PER-HEAD family (rope_row, the KVQ
  // banks, the phase RAM) and the D_MODEL-WIDE family (the feeder C-1
  // quantization row, the act/weight stage GEMM-contraction rows), forcing
  // head_dim == D_model in any single build. CFG_DM is the MODEL-WIDE
  // family's row length. Default CFG_D keeps every pre-split build
  // byte-identical (the families coincide when head_dim == D_model); a
  // SPLIT build (CFG_DM > CFG_D, e.g. 64/128) is the shape the five H=2
  // L4 cases and the 7B geometry (128/3584) need. Constraints: CFG_DM >=
  // CFG_D and lane8-framed (elaboration guard g_chk_dm below); the
  // feeder's own D legality (64/128, D-021) bounds CFG_DM until the
  // stage-6 wide-feeder elaboration lands (IB_LAYER.md §4 stage 6).
  parameter int unsigned CFG_DM        = CFG_D,
  parameter int unsigned KVQ_G         = 128,  // INFO_G / KVQ key group
  parameter int unsigned KVQ_DEPTH     = 128,  // KVQ SRAM records (per engine)
  // CQ-4+ outlier lane (engine 2 of the tier bank, D-024): top-k fp16 key
  // channels + the static calibrated mask ROM. OUTLIER_K=0 (no mask) makes
  // engine 2 degenerate to CQ-4 and clears INFO_TIER bit 2 (build truth).
  parameter int unsigned KVQ_OUTLIER_K = 0,
  parameter              KVQ_MASK_FILE = "",
  parameter int unsigned TIP_BLOCK_M   = 1,    // TIP tile geometry (M*N pow2)
  parameter int unsigned TIP_BLOCK_N   = 8,
  parameter int unsigned SM_ROW_MAX    = 1024, // ASU softmax row bound (verified cfg)
  parameter int unsigned FEED_ROWS_MAX = 16,   // feeder rows/job bound
  parameter int unsigned STAGE_R_MAX   = 16,   // stage buffer rows/bank
  parameter int unsigned SEQ_QDEPTH    = 8,
  // ── I-C IC-QPATH, F6 part (ii): per-head q staging depth ────────────────
  // How many heads' q rows a walk may consume. 1 (DEFAULT) = the pre-I-C
  // tile EXACTLY: one s_q holder, one act-stage q row, the single
  // `hd_sq_seen_q` sticky — every existing build elaborates unchanged, and
  // the whole per-head block below constant-folds away. > 1 elaborates the
  // per-head s_q file whose entry `hd_head` is replayed into the composite
  // cache at each head interlock, which is what lets a MULTI-HEAD fmt=1
  // attention walk complete (LEVEL_C_INTEGRATION.md §9.1 F6(ii)).
  // Must be <= WALK2_H_MAX (30, the fmt=1 RQ table cap) and <= STAGE_R_MAX
  // (the q rows live in act-stage bank 1, one row per head).
  parameter int unsigned QSTAGE_H_MAX  = 1,
  // T_ROW_MAX — the tile's per-job attention-row ENVELOPE (F-1 CLOSED
  // 2026-07-09): one score-dequant job == one last-framed ASU softmax row,
  // so T <= T_ROW_MAX per job. Sizes seam_score_dequant's per-job composite
  // buffer AND apex_scale_quant's per-job element buffers (the S-4 P-requant
  // job is also T columns). Must be <= SM_ROW_MAX (ASU row bound). 128 is
  // the verified v0.2 envelope (full-length calib d64_T128 / adv outlier1000
  // / d128_T100 replays in verif/top/l3).
  parameter int unsigned T_ROW_MAX     = 128,
  // IB-LAYER (S4): wide-RMSNorm elaboration point and the residual/phase
  // memory depth. Defaults preserve every existing build byte-identically;
  // the I-B CL sets RMS_D_MAX=3584 (and LAYER_DM_MAX=3584) per IB_LAYER.md.
  parameter int unsigned RMS_D_MAX     = 128,
  parameter int unsigned LAYER_DM_MAX  = 128,
  // IB-LAYER (S4b): per-KV-head CQ-8 GQA banking (LEVEL_C §9.1 R3-AMENDED;
  // approved plan IB_LAYER.md §0.1). 1 (default) = today's build
  // byte-identically — the D-024 tier bank is the KVQ subsystem and no GQA
  // hardware elaborates. >1 (the I-B CL sets 4 = H_kv, with KVQ_DEPTH=256)
  // elaborates apex_kvq_gqa_bank in its place: per-KV-head CQ-8 engines
  // behind the walker/l_kv_map engine select; the tile is CQ-8-only in that
  // build and INFO_TIER reads the truth (see the 0x14 override below).
  parameter int unsigned KVQ_GQA_NENG  = 1,
  // IC-BIAS (gap B): the projection-bias S-2 seam. 0 (default) = today's
  // build byte-identically — no bias RAM, no adder, no route mux elaborates
  // and LAYER_CTRL[18]/LAYER_STATUS[4] stay reserved-0. 1 elaborates
  // apex_proj_bias as a SIBLING of apex_scale_quant on the f_* seam, selected
  // per S-2 job by LAYER_CTRL[18] l_bias_en (IB_LAYER.md §3d).
  parameter bit          PROJ_BIAS_EN  = 1'b0,
  // W4 INGEST LANE (D-031 combine; docs/design/W4_DATAPATH.md): the adopted
  // G=32/(B) 4-bit weight path on the external xw pipe. 0 (default) = today's
  // build byte-identically — no glue, no divider pipes, no CSR window
  // elaborates and 0x9C-0xAC stay csr_regs-reserved (read 0xDEADBEEF: the
  // absent-feature probe). 1 elaborates apex_w4_ingest (which owns the
  // verified mxe_wfeed_w4b) behind the quasi-static W4_CTRL.lane_en route
  // bit on the xw -> MXE weight leg; lane_en=0 keeps the datapath muxes in
  // their exact legacy forms, so the host path stays bit-identical at
  // runtime too. W4_G is the scale-group geometry (D-031 ship freeze: 32;
  // 16 stays buildable per the feeder's own contract).
  parameter bit          W4_LANE       = 1'b0,
  parameter int unsigned W4_G          = 32,
  // F-5a: stage-buffer beats-per-row field width, DERIVED from the row
  // length so the value BPR itself is expressible (16 beats needs 5 bits;
  // the old fixed [3:0] ports truncated nb=16 to 0 and wedged every D=128
  // job). GAP D: stage rows are MODEL rows (the GEMM contraction), so this
  // width follows CFG_DM. WK_NB_W is the WALKER's aj/wj nb width — the
  // walker frames PER-HEAD attention rows and derives its own field from
  // its CFG_D (seq_layer_walker2 header localparam); the host/walker mode
  // muxes zero-extend WK_NB_W -> STAGE_NB_W (equal in a non-split build).
  localparam int unsigned STAGE_NB_W   = $clog2(CFG_DM / 8) + 1,
  localparam int unsigned WK_NB_W      = $clog2(CFG_D / 8) + 1
)(
  input  logic         clk,
  input  logic         rst_n,              // synchronous, active-low

  // ── CSR simple bus (csr_regs contract) ───────────────────────────────────
  input  logic [7:0]   csr_addr,
  input  logic [31:0]  csr_wdata,
  input  logic         csr_write,
  input  logic         csr_read,
  output logic [31:0]  csr_rdata,
  output logic         csr_ready,

  // ── KVQ AXI-Lite window (v0.1 host-driven; see boundary note 4) ──────────
  input  logic [7:0]   kv_awaddr,
  input  logic         kv_awvalid,
  output logic         kv_awready,
  input  logic [31:0]  kv_wdata,
  input  logic         kv_wvalid,
  output logic         kv_wready,
  output logic [1:0]   kv_bresp,
  output logic         kv_bvalid,
  input  logic         kv_bready,
  input  logic [7:0]   kv_araddr,
  input  logic         kv_arvalid,
  output logic         kv_arready,
  output logic [31:0]  kv_rdata,
  output logic [1:0]   kv_rresp,
  output logic         kv_rvalid,
  input  logic         kv_rready,
  output logic         kv_irq,
  output logic         kv_evict_needed,
  output logic [$clog2(KVQ_DEPTH)-1:0] kv_evict_addr,

  // ── SEQ descriptor stream in (host; D-006-proven path to MXE) ────────────
  input  logic         ds_valid,
  output logic         ds_ready,
  input  mxe_desc_t    ds_desc,

  // ── external weight stream in (host; projection weights, v0.1) ───────────
  input  logic         xw_valid,
  output logic         xw_ready,
  input  lane8_beat_t  xw_beat,

  // ── x8 activation stream in (tile data input; INT8 + last) ───────────────
  input  logic         xa_valid,
  output logic         xa_ready,
  input  logic signed [7:0] xa_x,
  input  logic         xa_last,

  // ── gamma stream in (Q2.13) ───────────────────────────────────────────────
  input  logic         xg_valid,
  output logic         xg_ready,
  input  logic signed [15:0] xg_gamma,

  // ── seam sidebands in (host-computed composites; boundary note 3) ────────
  input  logic         qs_valid,           // scale_quant sideband
  output logic         qs_ready,
  input  logic [31:0]  qs_data,
  input  logic         cs_valid,           // score-dequant composites
  output logic         cs_ready,
  input  logic [31:0]  cs_data,

  // ── glue job command ports (host-sequenced, D-006 handshakes) ────────────
  input  logic             fj_valid,       // feeder
  output logic             fj_ready,
  input  logic [DIM_W-1:0] fj_rows,
  input  logic             qj_valid,       // scale_quant
  output logic             qj_ready,
  input  logic             qj_mode,
  input  logic [DIM_W-1:0] qj_cols,
  input  logic             dj_valid,       // score dequant
  output logic             dj_ready,
  input  logic [DIM_W-1:0] dj_cols,
  input  logic             lj_valid,       // lane32 serializer
  output logic             lj_ready,
  input  logic [7:0]       lj_beats,
  input  logic [3:0]       lj_lanes,
  input  logic             aj_valid,       // act stage buffer
  output logic             aj_ready,
  input  logic             aj_op,
  input  logic             aj_bank,
  input  logic [1:0]       aj_pat,
  input  logic [4:0]       aj_rows,
  input  logic [STAGE_NB_W-1:0] aj_nb,     // F-5a: derived width, holds BPR
  input  logic [4:0]       aj_sel,
  input  logic             wj_valid,       // wgt stage buffer
  output logic             wj_ready,
  input  logic             wj_op,
  input  logic             wj_bank,
  input  logic [1:0]       wj_pat,
  input  logic [4:0]       wj_rows,
  input  logic [STAGE_NB_W-1:0] wj_nb,     // F-5a: derived width, holds BPR
  input  logic [4:0]       wj_sel,

  // ── route levels (host; change only while the touched paths are idle) ────
  input  logic         rt_feeder_src,      // 0=rmsnorm widen, 1=KVQ read bus
  input  logic         rt_feeder_dst,      // 0=act stage load, 1=wgt stage load
  input  logic         rt_act_src,         // act load: 0=feeder, 1=scale_quant
  input  logic         rt_wgt_src,         // MXE wgt: 0=external, 1=wgt stage
  input  logic [1:0]   rt_res_dst,         // MXE res: 0=out, 1=ser, 2=score-dq
  input  logic         rt_squant_src,      // squant v: 0=ser, 1=ASU probs
  input  logic         rt_kv_user,         // KVQ s_axis tuser: 0=key, 1=value
  input  logic [6:0]   rt_tip_blk,         // TIP s_blk + AUTO-mode tier index
  input  logic [15:0]  rt_imp_hi,
  input  logic [15:0]  rt_imp_lo,
  input  logic         rt_imp_clear,

  // ── scale taps out (host reads scales to build composites) ───────────────
  output logic         fs_valid,           // feeder per-row fp16 scales
  input  logic         fs_ready,
  output logic [15:0]  fs_data,
  output logic         fs_last,
  output logic         ss_valid,           // scale_quant row scale
  input  logic         ss_ready,
  output logic [15:0]  ss_data,
  output logic         ss_last,

  // ── TIP decision out ──────────────────────────────────────────────────────
  output logic         td_valid,
  input  logic         td_ready,
  output logic         td_fp16,
  output kvq_tier_e    td_tier,
  output logic [6:0]   td_blk,

  // ── result stream out (MXE res when routed here; o8/raw beats) ───────────
  output logic         ro_valid,
  input  logic         ro_ready,
  output lane32_beat_t ro_beat,

  // ── walker fetch requests out (combine W-G3 flip: walker2's wf_* to
  //    IB-FUEL's reader; frozen 64-bit fuel_req layout {tag[63:56],
  //    beats_64B[55:30], base_64B[29:0]} per §9.1 R1 / IB_FUEL.md §2.4.
  //    NEW ports by combine-owned surgery — the zero-new-ports fence (R5)
  //    was IB-LAYER's stage-4 rule for ITS glue, not the combine's; every
  //    instantiation site is updated in the same commit.) ──────────────────
  output logic         wf_valid,
  input  logic         wf_ready,
  output logic [63:0]  wf_req,

  // ── done pulses (host sequencing aids) ────────────────────────────────────
  output logic         dn_mxe,
  output logic         dn_feeder,
  output logic         dn_squant,
  output logic         dn_scored,
  output logic         dn_ser,
  output logic         dn_astage,
  output logic         dn_wstage,
  output logic         dn_rms,
  output logic         dn_asu,

  // ── sticky error bundle (bit map in the header of this section below) ────
  output logic [15:0]  err_sticky,

  // ── passive debug taps (accepted-beat mirrors; no flow control) ──────────
  output logic         dbg_f16_v,          // scale_quant F16 -> KVQ s_axis
  output logic [15:0]  dbg_f16_data,
  output logic         dbg_f16_last,
  output logic         dbg_sc_v,           // score-dequant -> fork
  output logic [31:0]  dbg_sc_data,
  output logic         dbg_sc_last,
  output logic         dbg_pr_v,           // ASU probabilities -> P-requant
  output logic [15:0]  dbg_pr_data,
  output logic         dbg_pr_last
);


  // ═══════════════════════════════════════════════════════════════════════════
  // B1 LAYER-WALKER GLUE — docs/design/B1_WALKER.md §3/§4/§7. TWO fenced,
  // additive regions, both confined to this file:
  //   (a) the WALK CSR window at 0x5C-0x6C, copying the ERR_STICKY pattern
  //       below (csr_regs acks the address as reserved; this file owns the
  //       data and overrides the read);
  //   (b) the walk_en mode mux: with walk_en=1 the walker owns the control
  //       fanouts and the external host-driven ds_*/job/route/seam/kv_* inputs
  //       are held off. walk_en=0 is the verified host path, bit-identical.
  //
  // §B-1's fs_*/ss_* accepted-beat MIRROR proved UNNECESSARY once §A-2 chose
  // the store-snoop scale source: the composite unit never CONSUMES fs_*/ss_*
  // as a stream, it only observes ss_* passively (no ready), so the TB remains
  // the tap sink and every EFS/ESS check survives walker mode unchanged. The
  // mirror was budgeted against a tap-FED composite unit; that design is not
  // the one built. No new ports, no third region.
  // ═══════════════════════════════════════════════════════════════════════════

  // effective (muxed) control fanouts — every consumer in this file reads these
  logic                  w_rt_feeder_src, w_rt_feeder_dst, w_rt_act_src;
  logic                  w_rt_wgt_src, w_rt_squant_src, w_rt_kv_user;
  logic [1:0]            w_rt_res_dst;
  logic                  w_fj_valid, w_fj_ready;
  logic [DIM_W-1:0]      w_fj_rows;
  logic                  w_qj_valid, w_qj_ready, w_qj_mode;
  logic [DIM_W-1:0]      w_qj_cols;
  logic                  w_dj_valid, w_dj_ready;
  logic [DIM_W-1:0]      w_dj_cols;
  logic                  w_aj_valid, w_aj_ready, w_aj_op, w_aj_bank;
  logic [1:0]            w_aj_pat;
  logic [4:0]            w_aj_rows, w_aj_sel;
  logic [STAGE_NB_W-1:0] w_aj_nb;
  logic                  w_wj_valid, w_wj_ready, w_wj_op, w_wj_bank;
  logic [1:0]            w_wj_pat;
  logic [4:0]            w_wj_rows, w_wj_sel;
  logic [STAGE_NB_W-1:0] w_wj_nb;
  logic                  w_qs_valid, w_qs_ready, w_cs_valid, w_cs_ready;
  logic [31:0]           w_qs_data, w_cs_data;
  logic                  w_ds_valid, w_ds_ready;
  mxe_desc_t             w_ds_desc;
  logic [7:0]            w_kv_awaddr, w_kv_araddr;
  logic                  w_kv_awvalid, w_kv_wvalid, w_kv_arvalid;
  logic                  w_kv_bready, w_kv_rready;
  logic [31:0]           w_kv_wdata;
  logic                  bk_awready, bk_wready, bk_bvalid, bk_arready, bk_rvalid;
  logic [31:0]           bk_rdata;
  logic [1:0]            bk_bresp, bk_rresp;

  // walker outputs
  logic                  wk_busy, wk_err;
  walk_phase_e           wk_phase;
  walk_err_e             wk_err_code;
  // walker2 superset (combine W-G3 flip): status/step, fetch record, the
  // per-head interlock, the §3b LAYER drive nets, and the S4b engine select
  walk_step_e            wk2_step;
  walk2_freq_t           wk_wf_req;
  logic                  wk_hd_valid;
  logic [7:0]            wk_hd_head;
  logic                  wk_lw_rope_en, wk_lw_rope_bank, wk_lw_resid_arm;
  logic [6:0]            wk_lw_rope_pos;
  logic [1:0]            wk_lw_ser_dst;
  // E-3 (E2E_TOY_LANE.md §4, 2026-08-01): 3 BITS. E-1 put l_fsrc_ext[2] at
  // LAYER_CTRL[7], but this walker-side net stayed 2 bits and the walk-mode
  // mux below ZERO-EXTENDED it — so a walked program could not name feeder
  // source code 4 and the in-tile norm feed was host-CSR-only.
  logic [2:0]            wk_lw_fsrc_ext;
  // E-4b (E2E_TOY_LANE.md §4, 2026-08-04): the l_nsrc walker level + its
  // OWNERSHIP level. l_nsrc (LAYER_CTRL[19]) was the one E-lane level with
  // NO walker port — HOST-held across a walk (the l_bias_en idiom), which
  // is exactly why walk_e4 needed two kicks. The l_nsrc_q register below
  // tracks wk_lw_nsrc ONLY while wk_lw_nsrc_own is high (a walk whose
  // descriptor names the previously-reserved mask bit W2_EN_NSRC); every
  // legacy walked image keeps the HOLD semantics byte-identically.
  logic                  wk_lw_nsrc, wk_lw_nsrc_own;
  // E-7: the walker's GAMMA WINDOW level (lw_gsrc) — high only inside an
  // FGAM walk's G1/G2-fetch-to-norm-done window. The registered copy
  // gam_win_q below steers xw -> apex_gam_unpack -> the norm's g port;
  // every legacy walk (bit 15 = 0) leaves it 0, so the xw -> MXE and
  // xg -> norm paths are byte-identical.
  logic                  wk_lw_gsrc;
  // E-3: LAYER_STATUS[5] fed BACK to the walker as the NFEED step's
  // completion observable. Declaration hoisted here (the E-1/E-2 region
  // below assigns it) because u_walk is instantiated above that region —
  // same hoist the fq_busy declaration already uses.
  logic                  nf_busy;
  logic                  wk_lu_valid, wk_lu_ready;
  logic [1:0]            wk_lu_unit;
  logic [11:0]           wk_lu_cols;
  // E-3b: the walker's SERIALIZER job channel (QSTAGE production framing).
  // Muxed onto u_ser's job port below; the HOST lj_* port is held off
  // during a walk (no walked program to date pushed it mid-walk — the
  // walked drains use reads only, and every emitter's serializer pushes
  // are pre-GO or post-walk).
  logic                  wk_sj_v;
  logic [7:0]            wk_sj_beats;
  logic [3:0]            wk_sj_lanes;
  logic                  wk_jc_valid, wk_jc_ready;
  logic [31:0]           wk_jc_data;
  logic                  wk_lj_valid;
  logic [DIM_W-1:0]      wk_lj_cols;
  // kv_eng_sel is [0:0] at N_ENG=1 (a single-engine build has one KV group,
  // so the select is a constant and no consumer exists — the net is named,
  // carried and sunk). At KVQ_GQA_NENG>1 it is the walk-mode half of the
  // per-KV-head engine select (§9.1 R3 as amended): ONE width definition
  // for the walker port, the S4b KVQ bank and the F6 composite bank.
  localparam int unsigned GQA_ENG_W = (KVQ_GQA_NENG > 1)
                                    ? $clog2(KVQ_GQA_NENG) : 1;
  logic [GQA_ENG_W-1:0]  wk_kv_eng_sel;
  logic                  wk_ds_valid;
  mxe_desc_t             wk_ds_desc;
  logic                  wk_rt_fsrc, wk_rt_fdst, wk_rt_asrc, wk_rt_wsrc,
                         wk_rt_qsrc, wk_rt_kvu;
  logic [1:0]            wk_rt_rdst;
  logic                  wk_fj_valid, wk_qj_valid, wk_qj_mode, wk_dj_valid;
  logic [DIM_W-1:0]      wk_fj_rows, wk_qj_cols, wk_dj_cols;
  logic                  wk_aj_valid, wk_aj_op, wk_aj_bank;
  logic [1:0]            wk_aj_pat;
  logic [4:0]            wk_aj_rows, wk_aj_sel;
  logic [WK_NB_W-1:0]    wk_aj_nb;    // GAP D: walker frames per-head rows
  logic                  wk_wj_valid, wk_wj_op, wk_wj_bank;
  logic [1:0]            wk_wj_pat;
  logic [4:0]            wk_wj_rows, wk_wj_sel;
  logic [WK_NB_W-1:0]    wk_wj_nb;    // GAP D: walker frames per-head rows
  logic                  wk_qs_valid, wk_cs_valid;
  logic [31:0]           wk_qs_data, wk_cs_data;
  logic                  wk_aw_v, wk_w_v, wk_ar_v, wk_b_r, wk_r_r;
  logic [7:0]            wk_aw_a, wk_ar_a;
  logic [31:0]           wk_w_d;
  logic                  wc_req_v, wc_req_r, wc_req_qs, wc_res_v, wc_res_r;
  logic [WALK_SC_AW-1:0] wc_req_idx;
  logic [31:0]           wc_res_data;
  logic                  wc_err_frame, wc_err_stale;
  // DBG-v5 attribution wires (driven by whichever composite gen-branch is
  // elaborated; read by the error-latch always_ff above the branches)
  logic [KVQ_GQA_NENG-1:0] wc_dbg_stale_eng;
  logic                  wc_dbg_sc, wc_dbg_sq;

  // ── (a) WALK CSR window (0x5C-0x6C) ───────────────────────────────────────
  // csr_regs decodes only to 0x54 and ERR_STICKY owns 0x58, so this range
  // falls through to its reserved path (read 0xDEAD_BEEF, write no-op, ready
  // still acked) — exactly the seam the ERR_STICKY window uses.
  //
  // IB-WALK stage 4 (D-029, IB_WALK.md §2.1/§4 — SAME addresses, no new CSR):
  // the descriptor SRAM grows 3 -> 64 words and DPTR 2 -> 6 bits so a host
  // can load an fmt=1 image through the EXISTING DPTR/DDATA path (the fmt=1
  // per-step patch is DPTR=W2_STEP + DDATA writes — no dedicated register),
  // and WALK_STATUS[15:12] publishes FMT_SUP.
  //
  // COMBINE W-G3 FLIP (§9.1 combine agenda; IB_WALK.md §4 stage 5/6): u_walk
  // is seq_layer_walker2 and FMT_SUP = WALK_FMT_SUP_LAYER (0b0011 — fmt 0
  // AND fmt 1 accepted), flipped TOGETHER as contracted. The instance reads
  // the FULL 64-word SRAM; a loaded fmt=0 image rides walker2's transparent
  // v1 forward to its wrapped, unchanged D-028 engine (bit-identical v1
  // semantics — the old 3-word walk_desc_v1 alias is gone because the
  // walker itself owns the word map now), fmt=1 images run the full-layer
  // ROM, and fmt>1 still refuses (WALK_ERR_DESC). WALK_RQ keeps its D-028
  // fast-path meaning (word 1).
  logic        walk_en_q, walk_go_q;
  logic        walk_err_sticky;
  logic        walk_dbg_frame_q, walk_dbg_stale_q;   // WALK_DBG (0x98)
  logic [9:0]  walk_dbg_ctx_q;                       // first-error ctx (v5)
  logic [9:0]  walk_dbg_acc_q;                       // last ACCEPTED request
  logic [1:0]  walk_dbg_seng_q;                      // which engine staled
  logic [1:0]  walk_dbg_term_q;                      // {sq_miss, sc_miss}
  logic [7:0]  walk_dbg_snp_q;                       // DBG-v2 snooped-row count
  logic [5:0]  walk_dbg_ca_q;                        // DBG-v3 last-commit addr
  logic        walk_dbg_we_q;                        // DBG-v4 commit-eng OR
  walk_err_e   walk_err_code_q;
  logic [WALK2_DPTR_W-1:0] walk_dptr_q;
  logic [31:0] walk_dsram [WALK2_DESC_WORDS];
  logic        walk_rd_q;
  logic [31:0] walk_rdata_q;
  logic        wctrl_wr, wdptr_wr, wddata_wr, wstat_wr, wrq_wr;

  assign wctrl_wr  = csr_write && (csr_addr == WALK_CTRL_ADDR);
  assign wdptr_wr  = csr_write && (csr_addr == WALK_DPTR_ADDR);
  assign wddata_wr = csr_write && (csr_addr == WALK_DDATA_ADDR);
  assign wstat_wr  = csr_write && (csr_addr == WALK_STATUS_ADDR);
  assign wrq_wr    = csr_write && (csr_addr == WALK_RQ_ADDR);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      walk_en_q       <= 1'b0;
      walk_go_q       <= 1'b0;
      walk_dptr_q     <= '0;
      walk_err_sticky <= 1'b0;
      walk_dbg_frame_q <= 1'b0;
      walk_dbg_stale_q <= 1'b0;
      walk_dbg_ctx_q  <= '0;
      walk_dbg_acc_q  <= '0;
      walk_dbg_seng_q <= '0;
      walk_dbg_term_q <= '0;
      walk_dbg_snp_q  <= '0;
      walk_dbg_ca_q   <= '0;
      walk_dbg_we_q   <= 1'b0;
      walk_err_code_q <= WALK_ERR_NONE;
      walk_rd_q       <= 1'b0;
      walk_rdata_q    <= '0;
      for (int i = 0; i < WALK2_DESC_WORDS; i++) walk_dsram[i] <= '0;
    end else begin
      walk_go_q <= 1'b0;                       // self-clearing kick
      if (wctrl_wr) begin
        walk_en_q <= csr_wdata[0];
        walk_go_q <= csr_wdata[1];
      end
      if (wdptr_wr) walk_dptr_q <= csr_wdata[WALK2_DPTR_W-1:0];
      // per-step fast path: rewrite ONLY the requant word, no pointer walk
      if (wrq_wr) walk_dsram[WALK_DW_RQ] <= csr_wdata;
      if (wddata_wr) begin
        // every 6-bit pointer value addresses a real word now; the window
        // wraps mod 64 (stage 4: the old 3-word guard dropped writes at
        // ptr 3, a state no D-028 flow ever reached — DPTR is always set
        // before a burst)
        walk_dsram[walk_dptr_q] <= csr_wdata;
        walk_dptr_q <= walk_dptr_q + WALK2_DPTR_W'(1);
      end
      // W1C on the sticky; a same-cycle error WINS (never lose an error)
      if (wstat_wr && csr_wdata[8]) walk_err_sticky <= 1'b0;
      if (wk_err) begin
        walk_err_sticky <= 1'b1;
        walk_err_code_q <= wk_err_code;
      end else if (wc_err_frame || wc_err_stale) begin
        // Composite-unit faults (record framing / scale-cache miss) land in
        // the WALKER's sticky, NOT err_sticky[15]: that bit is reserved-0 and
        // is pinned by every L3 case's ESTK expectation, so routing a new
        // error there would break all 27 cases.
        walk_err_sticky <= 1'b1;
        walk_err_code_q <= WALK_ERR_SEQ;
      end
      // WALK_DBG (0x98): which composite fault fired — same W1C, same
      // never-lose-an-error ordering as the sticky above. Latches even
      // when wk_err wins the code mux: the CSR answers "did the composite
      // complain at all", independent of code arbitration.
      if (wstat_wr && csr_wdata[8]) begin
        walk_dbg_frame_q <= 1'b0;
        walk_dbg_stale_q <= 1'b0;
        walk_dbg_ctx_q   <= '0;
        walk_dbg_snp_q   <= '0;
        walk_dbg_ca_q    <= '0;
        walk_dbg_we_q    <= 1'b0;
      end
      // DBG-v3 (2026-08-07, the commit-address question): at each record
      // tlast (a commit attempt — framing already proven clean by [0]=0 on
      // the flights), latch WHERE the snoop said it went. The 2026-08-07
      // seam flight proved writes happen and the read misses; this says
      // whether the WRITES went to the wrong slots or the READ lies.
      if (kv_s_tvalid && kv_s_tready && kv_s_tlast) begin
        walk_dbg_ca_q <= walk_dbg_ca_q | 6'(snoop_addr_q);
        // DBG-v4: WHICH ENGINE took the commit — the one attribution the
        // 2026-08-07 verdict flight could not see (a store into engine 1
        // with a walk reading engine 0 explains every observation while
        // staying self-consistent for host jobs)
        walk_dbg_we_q <= walk_dbg_we_q | 1'(kv_eng_sel);
      end
      // DBG-v5 (2026-08-08, THE CAPTURE-SKEW FIX): the v2 ctx latched
      // req_idx/qs/eng at the ERROR-PULSE cycle — one cycle AFTER the
      // acceptance the unit actually checked, so those fields were never
      // trustworthy (the fix-image flight exposed it: the word was
      // bit-identical across three netlists). Latch the request identity
      // at the ACCEPTANCE handshake instead; the first error freezes it.
      if (wc_req_v && wc_req_r)
        walk_dbg_acc_q <= {8'(wc_req_idx), wc_req_qs, 1'(kv_eng_sel)};
      if ((wc_err_frame || wc_err_stale)
          && !(walk_dbg_frame_q || walk_dbg_stale_q)) begin
        walk_dbg_ctx_q  <= walk_dbg_acc_q;
        walk_dbg_seng_q <= 2'(wc_dbg_stale_eng);
        walk_dbg_term_q <= {wc_dbg_sq, wc_dbg_sc};
      end
      // DBG-v2 write-side liveness: completed KV rows snooped into the
      // composite cache since the last W1C (saturating). The read side
      // (ctx above) says what the failing request asked for; THIS says
      // whether the staging's snoop stream reached the cache at all.
      if (kv_s_tvalid && kv_s_tready && kv_s_tlast
          && walk_dbg_snp_q != 8'hFF)
        walk_dbg_snp_q <= walk_dbg_snp_q + 8'd1;
      if (wc_err_frame) walk_dbg_frame_q <= 1'b1;
      if (wc_err_stale) walk_dbg_stale_q <= 1'b1;
      // (the error-cycle ctx capture was retired by the v5 acceptance
      // latch above — its one-cycle skew is the documented reason.)
      // read pipeline mirror of csr_regs (1-cycle; read-before-write)
      walk_rd_q    <= csr_read && ((csr_addr == WALK_CTRL_ADDR)
                                || (csr_addr == WALK_STATUS_ADDR)
                                || (csr_addr == WALK_DBG_ADDR));
      // WALK_STATUS[15:12] = FMT_SUP (D-029 discovery): the W-G3 flip
      // publishes WALK_FMT_SUP_LAYER (0b0011) — fmt 0 and fmt 1 both walk
      // on this tile; the host must not load fmt=N unless bit N reads 1
      walk_rdata_q <= (csr_addr == WALK_CTRL_ADDR)
                    ? {30'b0, walk_go_q, walk_en_q}
                    : (csr_addr == WALK_DBG_ADDR)
                    // 32-bit DBG-v5 word (MSB..LSB):
                    // [31:24] snp count   [23:16] accepted req_idx
                    // [15:10] commit-addr OR  [9:8] {sq_miss,sc_miss}
                    // [7:6] stale engine  [5] commit-eng OR  [4] live map
                    // [3:2] {qs,eng} of the accepted req  [1] stale [0] frame
                    ? {walk_dbg_snp_q, walk_dbg_ctx_q[9:2],
                       walk_dbg_ca_q,
                       walk_dbg_term_q,
                       walk_dbg_seng_q,
                       walk_dbg_we_q, 1'(l_kv_map_r[0]),
                       walk_dbg_ctx_q[1:0],
                       walk_dbg_stale_q, walk_dbg_frame_q}
                    : {16'b0, WALK_FMT_SUP_LAYER,
                       walk_err_code_q, walk_err_sticky,
                       4'b0, wk_phase, wk_busy};
    end
  end

  // ── (b) walk_en mode mux ──────────────────────────────────────────────────
  assign w_rt_feeder_src = walk_en_q ? wk_rt_fsrc : rt_feeder_src;
  assign w_rt_feeder_dst = walk_en_q ? wk_rt_fdst : rt_feeder_dst;
  assign w_rt_act_src    = walk_en_q ? wk_rt_asrc : rt_act_src;
  assign w_rt_wgt_src    = walk_en_q ? wk_rt_wsrc : rt_wgt_src;
  assign w_rt_res_dst    = walk_en_q ? wk_rt_rdst : rt_res_dst;
  assign w_rt_squant_src = walk_en_q ? wk_rt_qsrc : rt_squant_src;
  assign w_rt_kv_user    = walk_en_q ? wk_rt_kvu  : rt_kv_user;

  assign w_ds_valid = walk_en_q ? wk_ds_valid : ds_valid;
  assign w_ds_desc  = walk_en_q ? wk_ds_desc  : ds_desc;
  assign ds_ready   = walk_en_q ? 1'b0 : w_ds_ready;

  assign w_fj_valid = walk_en_q ? wk_fj_valid : fj_valid;
  assign w_fj_rows  = walk_en_q ? wk_fj_rows  : fj_rows;
  assign fj_ready   = walk_en_q ? 1'b0 : w_fj_ready;
  assign w_qj_valid = walk_en_q ? wk_qj_valid : qj_valid;
  assign w_qj_mode  = walk_en_q ? wk_qj_mode  : qj_mode;
  assign w_qj_cols  = walk_en_q ? wk_qj_cols  : qj_cols;
  assign qj_ready   = walk_en_q ? 1'b0 : w_qj_ready;
  assign w_dj_valid = walk_en_q ? wk_dj_valid : dj_valid;
  assign w_dj_cols  = walk_en_q ? wk_dj_cols  : dj_cols;
  assign dj_ready   = walk_en_q ? 1'b0 : w_dj_ready;

  assign w_aj_valid = walk_en_q ? wk_aj_valid : aj_valid;
  assign w_aj_op    = walk_en_q ? wk_aj_op    : aj_op;
  assign w_aj_bank  = walk_en_q ? wk_aj_bank  : aj_bank;
  assign w_aj_pat   = walk_en_q ? wk_aj_pat   : aj_pat;
  assign w_aj_rows  = walk_en_q ? wk_aj_rows  : aj_rows;
  assign w_aj_nb    = walk_en_q ? STAGE_NB_W'(wk_aj_nb) : aj_nb;
  assign w_aj_sel   = walk_en_q ? wk_aj_sel   : aj_sel;
  assign aj_ready   = walk_en_q ? 1'b0 : w_aj_ready;
  assign w_wj_valid = walk_en_q ? wk_wj_valid : wj_valid;
  assign w_wj_op    = walk_en_q ? wk_wj_op    : wj_op;
  assign w_wj_bank  = walk_en_q ? wk_wj_bank  : wj_bank;
  assign w_wj_pat   = walk_en_q ? wk_wj_pat   : wj_pat;
  assign w_wj_rows  = walk_en_q ? wk_wj_rows  : wj_rows;
  assign w_wj_nb    = walk_en_q ? STAGE_NB_W'(wk_wj_nb) : wj_nb;
  assign w_wj_sel   = walk_en_q ? wk_wj_sel   : wj_sel;
  assign wj_ready   = walk_en_q ? 1'b0 : w_wj_ready;

  assign w_qs_valid = walk_en_q ? wk_qs_valid : qs_valid;
  assign w_qs_data  = walk_en_q ? wk_qs_data  : qs_data;
  assign qs_ready   = walk_en_q ? 1'b0 : w_qs_ready;
  assign w_cs_valid = walk_en_q ? wk_cs_valid : cs_valid;
  assign w_cs_data  = walk_en_q ? wk_cs_data  : cs_data;
  assign cs_ready   = walk_en_q ? 1'b0 : w_cs_ready;

  // E-3b: the lane32 SERIALIZER job port joins the (b) mode mux — the
  // walker's QSTAGE step frames its per-head q production stream with it.
  // Host mode (walk_en_q=0) is the untouched path, byte-identical.
  logic       w_slj_valid, w_slj_ready;
  logic [7:0] w_slj_beats;
  logic [3:0] w_slj_lanes;
  assign w_slj_valid = walk_en_q ? wk_sj_v     : lj_valid;
  assign w_slj_beats = walk_en_q ? wk_sj_beats : lj_beats;
  assign w_slj_lanes = walk_en_q ? wk_sj_lanes : lj_lanes;
  assign lj_ready    = walk_en_q ? 1'b0 : w_slj_ready;

  // KVQ AXI-Lite: the walker masters the bank during a walk; the external
  // window is parked (the host contract already forbids touching the bank
  // mid-walk, and the temporal split is clean - host does store/inject,
  // WALK_GO covers score+pv, host resumes for the final phase).
  assign w_kv_awaddr  = walk_en_q ? wk_aw_a : kv_awaddr;
  assign w_kv_awvalid = walk_en_q ? wk_aw_v : kv_awvalid;
  assign w_kv_wdata   = walk_en_q ? wk_w_d  : kv_wdata;
  assign w_kv_wvalid  = walk_en_q ? wk_w_v  : kv_wvalid;
  assign w_kv_araddr  = walk_en_q ? wk_ar_a : kv_araddr;
  assign w_kv_arvalid = walk_en_q ? wk_ar_v : kv_arvalid;
  assign w_kv_bready  = walk_en_q ? wk_b_r  : kv_bready;
  assign w_kv_rready  = walk_en_q ? wk_r_r  : kv_rready;
  assign kv_awready = walk_en_q ? 1'b0 : bk_awready;
  assign kv_wready  = walk_en_q ? 1'b0 : bk_wready;
  assign kv_bvalid  = walk_en_q ? 1'b0 : bk_bvalid;
  assign kv_arready = walk_en_q ? 1'b0 : bk_arready;
  assign kv_rvalid  = walk_en_q ? 1'b0 : bk_rvalid;
  assign kv_rdata   = bk_rdata;
  assign kv_bresp   = bk_bresp;
  assign kv_rresp   = bk_rresp;

  // ── store observer: which record does each snooped fp16 frame belong to? ──
  // The host programs WRITE_ADDR (0x28) on kv_* before each CQ-8 record
  // (store_kv_phase:544,547), so the glue tracks it passively. This runs
  // whether or not walk_en is set - the scale cache must already be populated
  // when WALK_GO lands.
  localparam logic [7:0] KVQ_WADDR_A = 8'h28;
  logic [WALK_SC_AW-1:0] snoop_addr_q;
  always_ff @(posedge clk) begin
    if (!rst_n) snoop_addr_q <= '0;
    else if (w_kv_awvalid && bk_awready && w_kv_wvalid && bk_wready
             && (w_kv_awaddr == KVQ_WADDR_A))
      snoop_addr_q <= w_kv_wdata[WALK_SC_AW-1:0];
  end

  // ── per-KV-head engine select — THE SINGLE INGRESS (R5) ──────────────────
  // One register, one mode mux, two consumers: the S4b KVQ bank (which
  // engine stores the record) and the F6 composite bank below (which cache
  // harvests that record's scale). They MUST see the same value in the same
  // cycle — a split ingress would let a record land in engine g while its
  // scale lands in cache g'. Walk mode = walker2's kv_eng_sel (g_idx during
  // head walks, sk_g during STOREKV — both quiescent-switched by the
  // walker's poll-idle invariants); host mode = the §3b LAYER_CTRL[17:15]
  // l_kv_map level register (quasi-static: change only while the KVQ
  // subsystem is idle, the rt_* rule). l_kv_map_r is the CSR read view,
  // constant 0 when no GQA build elaborates so the S4 LAYER_CTRL readback
  // stays byte-identical.
  logic [2:0]           l_kv_map_r;
  logic [GQA_ENG_W-1:0] kv_eng_sel;

  // The composite cache's s_q tap. Two producers, never simultaneous: the
  // legacy S-2 `ss_*` snoop (host-mode staging, and the pre-I-C single-head
  // walk), and the F6(ii) per-head REPLAY below, which fires only while the
  // walker holds a head presented and the tile is otherwise idle. At
  // QSTAGE_H_MAX==1 `qh_sq_push` is a constant 0 and both nets reduce to the
  // pre-I-C expressions — byte-identical.
  logic        wc_sq_valid;
  logic [15:0] wc_sq_data;
  // 2026-08-10 KEEP-CHAIN (the kill-shot flight's verdict): with sc_val
  // flops FORCED to exist (dont_touch), the walked stale persists and the
  // v5 word is bit-identical — the snoop WRITE-ENABLE chain into the bank
  // is what synthesis const-folds, even though the IDENTICAL conjunction
  // sampled by the walk_dbg_snp counter is alive on silicon (snp=2 every
  // flight). Name and KEEP the conjunction so no segment can fold.
  (* keep = "true", dont_touch = "true" *)
  wire wc_snp_valid_k = kv_s_tvalid && kv_s_tready;
  // 2026-08-10 FLOPPED SNOOP BUNDLE (sledgehammer flight + forensics 6-9,
  // SLEDGEHAMMER_FLIGHT.md): the keep on the wire above is NOT ENOUGH — the
  // routed netlist feeds the bank's snp_valid from a REPLICA
  // (`wc_snp_valid_k_inferred_i_1`, LUT5) re-derived inside g_eng[1] with
  // the kvq bank's e_s_tready[idx] mux FOLDED to engine 1 — no idx term, no
  // eng0 ready — while the kept net (and the walk_dbg_snp counter it feeds)
  // keeps the live mux. One RTL expression, two cones, one wrong: commits
  // never reach the cache on silicon, sim can't see it. A REGISTER is the
  // fix a keep can't be: replication copies a flop exactly (same D net) and
  // can never re-derive its function. The whole bundle shifts one cycle
  // together, so pairwise alignment into the comp unit is unchanged; commits
  // land a cycle later and are consumed hundreds of cycles later. eng_sel
  // stays live: it moves only between heads/jobs, never adjacent to an
  // active snoop beat (sim battery re-proves bit-exactness regardless).
  (* dont_touch = "true" *)
  logic                  wc_snp_valid_q;
  logic [15:0]           wc_snp_data_q;
  logic                  wc_snp_last_q;
  logic [WALK_SC_AW-1:0] wc_snp_addr_q;
  // 2026-08-12 SECOND STAGE (A0 ladder, CLOCK_LADDER.md §9): at 62.5 MHz
  // the tile-level q-stage -> bank sc_mem BRAM write is THE critical path
  // (top 12 paths, WNS −8.68 — the dont_touch fence rightly forbids
  // retiming into the bank, so the route pays the full physical distance).
  // A second registered stage halves the path; the bundle shifts TWO
  // cycles together now — alignment unchanged, commits consumed hundreds
  // of cycles later. Also pure margin at A2.
  (* dont_touch = "true" *)
  logic                  wc_snp_valid_q2;
  logic [15:0]           wc_snp_data_q2;
  logic                  wc_snp_last_q2;
  logic [WALK_SC_AW-1:0] wc_snp_addr_q2;
  always_ff @(posedge clk) begin
    if (!rst_n) wc_snp_valid_q <= 1'b0;
    else        wc_snp_valid_q <= wc_snp_valid_k;
    wc_snp_data_q <= kv_s_tdata;
    wc_snp_last_q <= kv_s_tlast;
    wc_snp_addr_q <= snoop_addr_q;
    if (!rst_n) wc_snp_valid_q2 <= 1'b0;
    else        wc_snp_valid_q2 <= wc_snp_valid_q;
    wc_snp_data_q2 <= wc_snp_data_q;
    wc_snp_last_q2 <= wc_snp_last_q;
    wc_snp_addr_q2 <= wc_snp_addr_q;
  end
  assign wc_sq_valid = (ss_valid && ss_ready) || qh_sq_push;
  assign wc_sq_data  = qh_sq_push ? qh_sq_data : ss_data;

  // ── composite scale cache(s) ─────────────────────────────────────────────
  // KVQ_GQA_NENG==1 (default): today's SINGLE seq_walker_comp instance,
  // verbatim — one KV group, one record space, no select. >1: the F6(i)
  // per-KV-head bank (rtl/top/glue/apex_wcomp_bank.sv — N_ENG of the SAME
  // verified unit behind the select above), because a single cache indexed
  // by record address only has all H_kv groups colliding in [0,T)/[T,2T)
  // (LEVEL_C_INTEGRATION.md §9.1 F6, A/B-proven: 980/16,139 composite words
  // wrong, the survivors being exactly the last-staged group). Parameter-
  // gated exactly like the S4b KVQ region below: the default build is
  // untouched.
  generate if (KVQ_GQA_NENG == 1) begin : g_wcomp_one

  assign l_kv_map_r = 3'b0;      // no GQA bank: [17:15] stay reserved-0
  assign kv_eng_sel = '0;        // one cache/one engine: constant select

  assign wc_dbg_stale_eng[0] = wc_err_stale;
  seq_walker_comp #(.CFG_D(CFG_D)) u_wcomp (
    .dbg_stale_sc (wc_dbg_sc),
    .dbg_stale_sq (wc_dbg_sq),
    .clk        (clk),
    .rst_n      (rst_n),
    // the accepted fp16 beats going into KVQ (the nets behind dbg_f16_*)
    .snp_valid  (wc_snp_valid_q2),
    .snp_data   (wc_snp_data_q2),
    .snp_last   (wc_snp_last_q2),
    .snp_addr   (wc_snp_addr_q2),
    .snp_flush  (csr_soft_reset),
    // s_q, observed PASSIVELY on the ss_* tap (no ready taken - the TB stays
    // the sink, so ESS checks are unaffected)
    .sq_valid   (wc_sq_valid),
    .sq_data    (wc_sq_data),
    .req_valid  (wc_req_v), .req_ready (wc_req_r), .req_is_qs (wc_req_qs),
    .req_idx    (wc_req_idx),
    .res_valid  (wc_res_v), .res_ready (wc_res_r), .res_data (wc_res_data),
    .err_frame  (wc_err_frame), .err_stale (wc_err_stale)
  );

  // the select has no consumer in a single-engine build (the D-024 tier
  // bank has no eng_sel port) — sunk so the surface stays -Wall clean
  logic unused_engsel_ok;
  assign unused_engsel_ok = &{1'b0, kv_eng_sel};

  end else begin : g_wcomp_gqa

  // 2026-08-12 L_KV_MAP SLEDGEHAMMER (E7_LIVE_T_DEFECT.md; the D-033 /
  // SLEDGEHAMMER_FLIGHT.md pattern, third instance): the FIRST program to
  // host-stage KV into engine 1 — the e7 token-loop ATT kick — REFUSED on
  // silicon with every head's o8 under-accumulated while the identical
  // bytes stayed sim-green: the signature of the HOST branch of this
  // engine select never reaching the kvq bank's stream gating on the
  // routed netlist. l_kv_map_q + the select mux live at TILE level,
  // OUTSIDE the dont_touch-fenced bank instances, and fan out to THREE
  // consumer cones (kvq bank AXIL window, kvq bank s_axis stream gating,
  // wcomp bank snoop routing) — exactly the shape synthesis replicated
  // and const-folded PER CONE in D-033 ("one RTL expression, two cones,
  // one wrong"). Supporting forensics from the failing flight: the fs
  // scale ladders MATCHED while the o8 records under-accumulated — the
  // scale (wcomp) cone routed right while the record (stream) cone routed
  // wrong — only a per-cone fold explains both at once. A keep on the
  // wire is NOT ENOUGH (the D-033 lesson, re-measured on the kill-shot
  // flight): replication re-derives a WIRE's function freely but can only
  // COPY a flop (same D net). So the sledgehammer, both halves:
  //   (1) dont_touch the host map register itself, and
  //   (2) REGISTER the merged select ONCE and fan the one flop out to
  //       every consumer — no cone is ever re-derived from the mux terms.
  // The one-cycle select lag is safe by the bank's own routing contract
  // (apex_kvq_gqa_bank.sv header): eng_sel is QUASI-STATIC — host mode
  // moves it only between quiescent store runs (the rt_* level rule),
  // walk mode only between idle-polled store/read runs (walker2's
  // poll-idle-before-WRITE_ADDR / poll-before-READ_ADDR invariants) —
  // never adjacent to an active beat, so the whole select shifts one
  // cycle exactly as the flopped wc_snp_* bundle above shifts two: the
  // alignment argument is the same one, already carried. (The only
  // observable: the FIRST STATUS poll after a select move can still read
  // the PREVIOUS engine — an engine the choreography just proved idle —
  // one poll early; every subsequent handshake sees the settled select.)
  (* dont_touch = "true" *)
  logic [2:0] l_kv_map_q;
  always_ff @(posedge clk) begin
    if (!rst_n)                     l_kv_map_q <= '0;
    else if (!walk_en_q && lw_ctrl) l_kv_map_q <= csr_wdata[17:15];
  end
  assign l_kv_map_r = l_kv_map_q;
  (* dont_touch = "true" *)
  logic [GQA_ENG_W-1:0] kv_eng_sel_q;
  always_ff @(posedge clk) begin
    if (!rst_n) kv_eng_sel_q <= '0;
    else        kv_eng_sel_q <= walk_en_q ? GQA_ENG_W'(wk_kv_eng_sel)
                                          : GQA_ENG_W'(l_kv_map_q);
  end
  assign kv_eng_sel = kv_eng_sel_q;

  // 2026-08-08 THE SLEDGEHAMMER (netlist forensics, TERM_READ_VERDICT.md
  // follow-up): the routed checkpoint shows BOTH composite units swept to
  // ~10 cells each — no FSM, no sc_val, no sc_mem, no beat counter —
  // synthesis proved the snoop-commit chain dead and constant-folded the
  // cache to "always stale". dont_touch on the INSTANCE forbids any
  // optimization into or across the bank: every port, every register,
  // every RAM survives verbatim. Area is trivial (2x ~300 cells).
  (* dont_touch = "true", keep_hierarchy = "yes" *)
  apex_wcomp_bank #(.N_ENG(KVQ_GQA_NENG), .CFG_D(CFG_D)) u_wcomp (
    .dbg_stale_eng (wc_dbg_stale_eng),
    .dbg_stale_sc  (wc_dbg_sc),
    .dbg_stale_sq  (wc_dbg_sq),
    .clk        (clk),
    .rst_n      (rst_n),
    .eng_sel    (kv_eng_sel),
    // same taps as the single-instance build; the bank routes them to the
    // instance the KVQ bank is storing into this cycle
    .snp_valid  (wc_snp_valid_q2),
    .snp_data   (wc_snp_data_q2),
    .snp_last   (wc_snp_last_q2),
    .snp_addr   (wc_snp_addr_q2),
    .snp_flush  (csr_soft_reset),
    .sq_valid   (wc_sq_valid),
    .sq_data    (wc_sq_data),
    .req_valid  (wc_req_v), .req_ready (wc_req_r), .req_is_qs (wc_req_qs),
    .req_idx    (wc_req_idx),
    .res_valid  (wc_res_v), .res_ready (wc_res_r), .res_data (wc_res_data),
    .err_frame  (wc_err_frame), .err_stale (wc_err_stale)
  );

  end endgenerate

  // ── hd_* per-head interlock tie (combine W-G3 flip; W-G2 refinement) ─────
  // walker2's header contract: hd_ready == "this head's q row is staged and
  // its s_q snooped". PROVISIONALLY RATIFIED (IB_LAYER.md §3c, 2026-07-26)
  // as exactly this ss_* s_q snoop event; segment 1 owns the DEFINITIVE
  // arbitration. The event is u_wcomp's sq point (ss_valid && ss_ready —
  // the S-2 tap), with ss_last marking a COMPLETED row's scale.
  //
  // W-G2 refinement (first walkable fmt=1 tile case): the flag is a STICKY
  // "s_q snooped, unconsumed" latch — armed by the snoop event WHENEVER it
  // occurs, consumed by the accepted head handshake. The flip's first cut
  // armed only while hd_valid was presented, which is UNSATISFIABLE on
  // today's tile: the (b) mode mux holds off every host job port during a
  // walk (temporal split), so no ss beat can ever occur while a head is
  // presented — the q row is staged and snooped BEFORE WALK_GO (the D-028
  // choreography), and the sticky form is what "staged and snooped" means
  // for that pre-staged head. It also mirrors u_wcomp's own state, which
  // holds exactly one s_q at a time. Multi-head arming (a FRESH snoop per
  // head, mid-walk) needs the IB-LAYER S5 segment-1 q-projection path and
  // arbitrates there; until then only single-head fmt=1 images can
  // complete an attention walk (n_heads=1 — the walkfmt2 case).
  // THIS IS F6 PART (ii) (§9.1), AND IT IS STILL OPEN: the composite bank
  // above closes part (i) — per-KV-head caches, so a multi-GROUP walk no
  // longer collides — but this single sticky is why a multi-HEAD fmt=1 walk
  // still cannot complete on this tile. The two are independent causes and
  // part (i) is necessary, not sufficient.
  // Registered-accept: the walker holds hd_valid and captures this
  // registered level at its transition edge.
  logic hd_sq_seen_q;
  always_ff @(posedge clk) begin
    if (!rst_n)                               hd_sq_seen_q <= 1'b0;
    else if (ss_valid && ss_ready && ss_last) hd_sq_seen_q <= 1'b1;
    else if (wk_hd_valid && hd_sq_seen_q)     hd_sq_seen_q <= 1'b0;
  end

  // ── F6 PART (ii): per-head q staging (I-C IC-QPATH) ──────────────────────
  // The sticky above is a SINGLE holder, and each seq_walker_comp holds
  // exactly one s_q at a time (seq_walker_comp.sv:128 `s_q_q`), so on a
  // multi-head walk every head after the first composes its CS words from
  // whatever s_q was staged last. The walk cannot fix that by re-staging:
  // the (b) mode mux holds every HOST job port's ready low while walk_en_q
  // is set, so no later head's q row can be pushed mid-walk.
  //
  // The fix keeps BOTH of those facts and adds nothing per step: the H q
  // rows are staged BEFORE WALK_GO — through the gap-A sink, the same
  // injection scaffolding the single-head case already uses (IB_WALK.md §2.4
  // names it TB-side) — and the glue REMEMBERS each row's scale in a small
  // file. At each head interlock the file's entry `hd_head` is replayed into
  // the composite cache the head's KV group selects, so head h composes from
  // head h's s_q. The walker's own kv_eng_sel is already g(h) at S2_HLOAD
  // (S2_STEP arms g_idx with h_idx; S2_HWAIT advances both together), so the
  // replay lands in the right instance by construction — the same single
  // ingress the record store uses (R5).
  //
  // NO NEW REGISTER, NO NEW ADDRESS, NO EXTRA PER-STEP MMIO: the file's
  // write pointer is armed by the RISING EDGE OF THE q SINK ITSELF
  // (l_fsrc_ext -> 3), which the host already writes to stage q at all, and
  // the entries are captured from the feeder's own scale bus. The disclosed
  // ~39 MMIO/layer/step (H+11) figure is therefore unmoved: staging H rows
  // instead of 1 costs data-plane beats, not WALK-window register writes.
  localparam int unsigned QS_AW = (QSTAGE_H_MAX > 1) ? $clog2(QSTAGE_H_MAX)
                                                     : 1;
  logic                  qh_sq_push;      // replay strobe into the cache
  logic [15:0]           qh_sq_data;
  logic                  hd_ready_int;    // what the walker actually sees

  generate if (QSTAGE_H_MAX == 1) begin : g_qstage_one

  // Default build: the pre-I-C tile, verbatim.
  assign qh_sq_push  = 1'b0;
  assign qh_sq_data  = '0;
  assign hd_ready_int = hd_sq_seen_q;

  end else begin : g_qstage_multi

  logic [15:0]              sq_mem [QSTAGE_H_MAX];
  logic [QSTAGE_H_MAX-1:0]  sq_val;
  logic [QS_AW-1:0]         sq_wr_q;
  logic                     q_sink_q1, qh_pushed_q;

  // the staged head this interlock wants; out-of-range heads never arm
  // (a descriptor asking for more heads than the build stages is refused by
  // failing to become ready — never by silently reusing entry 0)
  wire in_range = (32'({24'b0, wk_hd_head}) < QSTAGE_H_MAX);
  wire [QS_AW-1:0] rd_idx = QS_AW'(wk_hd_head);
  wire have_sq = in_range && sq_val[rd_idx];

  // 2026-08-13 THE REPLAY SETTLE (fix-flight-1 verdict; capture forensics +
  // sim repro): the paragraph above says the replay "lands in the right
  // instance by construction" because kv_eng_sel is already g(h) at
  // S2_HLOAD. That was written for the COMBINATIONAL select — fa603d3's
  // D-033 sledgehammer registered it (kv_eng_sel_q above), so on the FIRST
  // presented cycle of every head interlock the bank still routes by
  // g(h-1): the s_q beat is fire-and-forget through the wcomp bank's
  // combinational sel gating, so head h's s_q lands SILENTLY in the
  // previous head's cache at every KV-group boundary (and, across walks,
  // head 0's in the LAST group's cache — g_idx resets at the K_ATTN
  // dispatch edge). The starved instance then refuses that head's first
  // CS request as s_q-stale — refusals present no response — and the walk
  // wedges in W_S_CSRES with the host spinning on its drain poll.
  // Sim-measured on the 0.5B ATT kick (H=14, GQA 2): heads 0..6 drained,
  // wedge exactly at head 7's first s_k fs beat (fs_840), the g 0->1
  // boundary. One flop is the fix (the walker-side S2_SKS argument, same
  // date): hd_pres_q defers the push to the SECOND presented cycle, by
  // which the registered select has settled to g(h). The walker holds
  // hd_valid until hd_ready (= registered qh_pushed_q), so the head
  // interlock simply stretches one cycle — once per head, per walk.
  logic hd_pres_q;
  assign qh_sq_push = wk_hd_valid && hd_pres_q && !qh_pushed_q && have_sq;
  assign qh_sq_data = sq_mem[rd_idx];
  assign hd_ready_int = qh_pushed_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sq_val      <= '0;
      sq_wr_q     <= '0;
      q_sink_q1   <= 1'b0;
      qh_pushed_q <= 1'b0;
      hd_pres_q   <= 1'b0;
    end else begin
      q_sink_q1 <= q_sink;
      // the replay-settle presentation flop (comment above): set on the
      // first presented cycle, cleared with the presentation — the same
      // clear term qh_pushed_q rides
      if (csr_soft_reset || !wk_hd_valid) hd_pres_q <= 1'b0;
      else                                hd_pres_q <= 1'b1;
      // ARM: a fresh q-staging sequence starts when the sink is armed.
      if (csr_soft_reset) begin
        sq_val  <= '0;
        sq_wr_q <= '0;
      end else if (q_sink && !q_sink_q1) begin
        sq_val  <= '0;
        sq_wr_q <= '0;
      end else if (q_sink && fs_valid && fs_ready
                   && (32'({{(32-QS_AW){1'b0}}, sq_wr_q}) < QSTAGE_H_MAX)) begin
        // CAPTURE: the feeder's row scale IS this head's s_q (gap A puts the
        // q row through the feeder's C-1 quant, so its scale bus carries
        // exactly golden's `quant_rows_i8(q_rope)` scale).
        sq_mem[sq_wr_q] <= fs_data;
        sq_val[sq_wr_q] <= 1'b1;
        sq_wr_q         <= sq_wr_q + QS_AW'(1);
      end
      // REPLAY: one push per presented head, cleared between heads.
      if (csr_soft_reset || !wk_hd_valid) qh_pushed_q <= 1'b0;
      else if (qh_sq_push)                qh_pushed_q <= 1'b1;
    end
  end

  end endgenerate

  // ── walker norm-job (lj_*) tie — COMBINE FINDING, reported ───────────────
  // IB_WALK.md §2.3 names an "IB-LAYER norm job port", but the FROZEN §3b
  // table landed NO norm unit and no lj_* norm-job mux exists in this file:
  // IB_LAYER.md L1/L7 keep RMSNorm host-streamed (xa_*/xg_* with host-loop
  // C-1 re-entry). Until IB-LAYER S5 lands a real norm job port, the
  // walker's norm push completes on the REAL norm event: ready arms on
  // dn_rms (the row's rmsnorm done) observed while the job is presented —
  // the W-G2 host-assisted choreography streams the row, the walker
  // sequences on its completion. Same registered-level shape as hd_* above;
  // wk_lj_cols is carried for the future port (sunk below).
  logic lj_done_seen_q;
  always_ff @(posedge clk) begin
    if (!rst_n || !wk_lj_valid) lj_done_seen_q <= 1'b0;
    else if (dn_rms)            lj_done_seen_q <= 1'b1;
  end

  assign wf_req = wk_wf_req;   // packed walk2_freq_t -> frozen 64-bit record

  // COMBINE W-G3 flip (§9.1 agenda; IB_WALK.md §4 stage-5 "combine-owned
  // remainder"): u_walk is now the fmt=1 sequencer seq_layer_walker2, which
  // WRAPS the unchanged v1 engine — a loaded fmt=0 image is forwarded to it
  // transparently inside walker2 (same-cycle pass-through), so every D-028
  // v1 walk is bit-identical through this instance; fmt=1 images run the
  // full-layer ROM. N_ENG=1 until IB-LAYER S4b lands the per-KV-head CQ-8
  // engine banks: walker2's build-envelope fence refuses n_kv_heads > N_ENG,
  // keeping 7B-geometry descriptors safely refused while single-engine
  // fmt=1 shapes walk. The instance reads the FULL 64-word SRAM.
  // E-3: FEED_DM/FEED_ROWS hand the walker the C-1 feeder's ACTUAL
  // instantiated shape (u_feeder below is #(.D(CFG_DM), .ROWS_MAX
  // (FEED_ROWS_MAX))), so the NFEED step's envelope fence measures the
  // consumer instead of assuming it.
  // E-5: STAGE_ROWS hands the walker the ACT STAGE BUFFER's actual rows-
  // per-bank bound (u_astage below is apex_stage_buf #(.D(CFG_DM),
  // .R_MAX(STAGE_R_MAX))), so the fuel-fed projection's act-row family
  // fence measures the consumer instead of assuming it — the same rule
  // FEED_DM/FEED_ROWS follow above.
  seq_layer_walker2 #(.CFG_D(CFG_D), .N_ENG(KVQ_GQA_NENG),
                      .Q_ROWS(QSTAGE_H_MAX),
                      .FEED_DM(CFG_DM), .FEED_ROWS(FEED_ROWS_MAX),
                      .STAGE_ROWS(STAGE_R_MAX)) u_walk (
    .clk (clk), .rst_n (rst_n),
    .walk_en (walk_en_q), .walk_go (walk_go_q), .abort_req (csr_soft_reset),
    .desc2_word (walk_dsram), .tile_idle (~|blk_busy),
    .walk2_busy (wk_busy), .walk2_step (wk2_step), .attn_phase (wk_phase),
    .walk2_err (wk_err), .walk2_err_code (wk_err_code),
    .wf_valid (wf_valid), .wf_ready (wf_ready), .wf_req (wk_wf_req),
    .hd_valid (wk_hd_valid), .hd_ready (hd_ready_int), .hd_head (wk_hd_head),
    .lw_rope_en (wk_lw_rope_en), .lw_rope_bank (wk_lw_rope_bank),
    .lw_rope_pos (wk_lw_rope_pos), .lw_ser_dst (wk_lw_ser_dst),
    .lw_fsrc_ext (wk_lw_fsrc_ext), .lw_resid_arm (wk_lw_resid_arm),
    .lw_nsrc (wk_lw_nsrc), .lw_nsrc_own (wk_lw_nsrc_own),
    .lw_gsrc (wk_lw_gsrc),
    .lu_valid (wk_lu_valid), .lu_ready (wk_lu_ready), .lu_unit (wk_lu_unit),
    .lu_cols (wk_lu_cols),
    .jc_valid (wk_jc_valid), .jc_ready (wk_jc_ready), .jc_data (wk_jc_data),
    .lj_valid (wk_lj_valid), .lj_ready (lj_done_seen_q), .lj_cols (wk_lj_cols),
    .sj_valid (wk_sj_v), .sj_ready (w_slj_ready), .sj_beats (wk_sj_beats),
    .sj_lanes (wk_sj_lanes),
    .nf_busy (nf_busy),
    .kv_eng_sel (wk_kv_eng_sel),
    .ds_valid (wk_ds_valid), .ds_ready (w_ds_ready), .ds_desc (wk_ds_desc),
    .rt_feeder_src (wk_rt_fsrc), .rt_feeder_dst (wk_rt_fdst),
    .rt_act_src (wk_rt_asrc), .rt_wgt_src (wk_rt_wsrc),
    .rt_res_dst (wk_rt_rdst), .rt_squant_src (wk_rt_qsrc),
    .rt_kv_user (wk_rt_kvu),
    .fj_valid (wk_fj_valid), .fj_ready (w_fj_ready), .fj_rows (wk_fj_rows),
    .qj_valid (wk_qj_valid), .qj_ready (w_qj_ready), .qj_mode (wk_qj_mode),
    .qj_cols (wk_qj_cols),
    .dj_valid (wk_dj_valid), .dj_ready (w_dj_ready), .dj_cols (wk_dj_cols),
    .aj_valid (wk_aj_valid), .aj_ready (w_aj_ready), .aj_op (wk_aj_op),
    .aj_bank (wk_aj_bank), .aj_pat (wk_aj_pat), .aj_rows (wk_aj_rows),
    .aj_nb (wk_aj_nb), .aj_sel (wk_aj_sel),
    .wj_valid (wk_wj_valid), .wj_ready (w_wj_ready), .wj_op (wk_wj_op),
    .wj_bank (wk_wj_bank), .wj_pat (wk_wj_pat), .wj_rows (wk_wj_rows),
    .wj_nb (wk_wj_nb), .wj_sel (wk_wj_sel),
    .qs_valid (wk_qs_valid), .qs_ready (w_qs_ready), .qs_data (wk_qs_data),
    .cs_valid (wk_cs_valid), .cs_ready (w_cs_ready), .cs_data (wk_cs_data),
    .comp_req_valid (wc_req_v), .comp_req_ready (wc_req_r),
    .comp_req_is_qs (wc_req_qs), .comp_req_idx (wc_req_idx),
    .comp_res_valid (wc_res_v), .comp_res_ready (wc_res_r),
    .comp_res_data (wc_res_data),
    .kvm_awvalid (wk_aw_v), .kvm_awready (bk_awready), .kvm_awaddr (wk_aw_a),
    .kvm_wvalid (wk_w_v), .kvm_wready (bk_wready), .kvm_wdata (wk_w_d),
    .kvm_bvalid (bk_bvalid), .kvm_bready (wk_b_r),
    .kvm_arvalid (wk_ar_v), .kvm_arready (bk_arready), .kvm_araddr (wk_ar_a),
    .kvm_rvalid (bk_rvalid), .kvm_rready (wk_r_r), .kvm_rdata (bk_rdata)
  );

  // walker2 spares, consumed for -Wall cleanliness: walk2_step stays
  // INTERNAL (no §3b/IB_WALK clause extends the WALK_STATUS read view —
  // the landed [3:1] walk_phase_e view via attn_phase is the reconciled
  // stage-4 shape, seq_walker_pkg walk_step_e note); wk_hd_head awaits the
  // per-head staging path; wk_kv_eng_sel drives the KVQ + composite banks in
  // a GQA build and has no consumer at KVQ_GQA_NENG==1 (one KV group), so it
  // stays in this sink for the default build; wk_lj_cols awaits the
  // IB-LAYER S5 norm job port (ties above).
  logic unused_walk2_ok;
  assign unused_walk2_ok = &{1'b0, wk2_step, wk_hd_head, wk_kv_eng_sel,
                             wk_lj_cols};

  // ═══════════════════════════════════════════════════════════════════════════
  // IB-LAYER GLUE (S4) — docs/design/IB_LAYER.md §3/§3b. NEW PARALLEL REGION:
  // the LAYER CSR window 0x70-0x8C + the decoder-layer op datapath. It does
  // NOT touch the B1 WALK region above (merge-shape directive — the WALK
  // fmt=1 delta owns :367-457/:557-568); zero new module ports (R5). The
  // walker later drives the SAME l_*/lj_* registers behind its mode mux —
  // one ingress point. New-block errors land in LAYER_STATUS[8], NEVER
  // err_sticky[15] (reserved-0, ESTK-pinned — the B1 rule).
  // ═══════════════════════════════════════════════════════════════════════════

  localparam logic [7:0] LAYER_CTRL_ADDR   = 8'h70;
  localparam logic [7:0] LAYER_PTR_ADDR    = 8'h74;
  localparam logic [7:0] LAYER_DATA_ADDR   = 8'h78;
  localparam logic [7:0] LAYER_JOB_ADDR    = 8'h7C;
  localparam logic [7:0] LAYER_STATUS_ADDR = 8'h80;
  localparam logic [7:0] LAYER_RPTR_ADDR   = 8'h84;
  localparam logic [7:0] LAYER_RDATA_ADDR  = 8'h88;
  localparam logic [7:0] LAYER_JOBC_ADDR   = 8'h8C;

  localparam int unsigned L_HALF   = CFG_D / 2;
  localparam int unsigned L_PAIRW  = $clog2(L_HALF);
  localparam int unsigned L_PHKN   = T_ROW_MAX * L_HALF;
  localparam int unsigned L_DMAW   = $clog2(LAYER_DM_MAX);

  // level registers (the frozen §3b table; E-1/E-2 widen l_fsrc_ext to 3
  // bits — [2] lives at LAYER_CTRL[7], reserved-0 until 2026-07-31, so every
  // legacy level word still encodes exactly the code it always did)
  logic        l_rope_en_q, l_rope_bank_q, l_resid_arm_q;
  logic [1:0]  l_ser_dst_q;
  logic [2:0]  l_fsrc_ext_q;
  logic [6:0]  l_rope_pos_q;
  logic [29:0] l_ptr_q;                     // [29:28] bank, [15:0] index
  logic [15:0] l_rptr_q;
  logic [31:0] ljc_a_q, ljc_b_q;
  logic        ljc_sel_q;
  logic        layer_rd_q;
  logic [31:0] layer_rdata_q;
  logic        l_err_stk_q;
  logic [3:0]  l_err_code_q;

  // memories
  logic [13:0] ph_k [L_PHKN];
  logic [13:0] ph_q [L_HALF];

  // unit nets
  logic        sqf_valid, sqf_ready, sqf_last;
  logic [15:0] sqf_data;
  // GAP A (I-C IC-QPATH): the S-2 seam, named so its DESTINATION can be
  // steered. q_sink selects the feeder instead of the KVQ write port.
  logic        q_sink;
  logic        seam_valid, seam_ready, seam_last;
  logic [15:0] seam_data;
  logic        rr_s_valid, rr_s_ready, rr_m_valid, rr_m_ready, rr_m_last;
  logic [15:0] rr_m_data;
  logic [L_PAIRW-1:0] rr_ph_addr;
  logic [13:0] rr_ph_data;
  logic        rr_busy, rr_ferr, rr_ferr_stk;
  logic        ldq_jb_v_q, swg_jb_v_q, res_jb_v_q;
  logic [11:0] lj_cols_q;
  logic [1:0]  lj_base_q;
  logic        ldq_jb_r, swg_jb_r, res_jb_r;
  logic        ldq_iv, ldq_ir_port, ldq_ov, ldq_or, ldq_olast;
  logic [31:0] ldq_odata;
  logic        ldq_busy, ldq_done, ldq_jerr, ldq_jerr_stk;
  logic        ldq_xerr, ldq_xerr_stk, ldq_ferr, ldq_ferr_stk;
  logic        swg_gv, swg_gr, swg_uv, swg_ur, swg_pv, swg_pr, swg_plast;
  logic [15:0] swg_pdata;
  logic        swg_busy, swg_done, swg_jerr, swg_jerr_stk;
  logic        swg_ferr, swg_ferr_stk;
  logic        res_iv, res_ir, res_busy, res_done;
  logic        res_ferr, res_ferr_stk, res_werr, res_werr_stk;
  logic        res_lw_en;
  logic [15:0] res_rd_data;

  // ═════════════════════════════════════════════════════════════════════════
  // E-1 / E-2 (E2E_TOY_LANE.md §4, 2026-07-31) — the residual -> norm seam
  // CLOSED INSIDE THE TILE. Before this the layer chain crossed the host
  // twice (layer entry X -> NORM1, mid-layer r1 -> NORM2): the residual row
  // RAM's only reader was the CSR path (rd_data -> layer_rdata_q) and
  // asu_rmsnorm's x input was the TOP-LEVEL xa_* port. Three additive
  // pieces, every one defaulted OFF and byte-identical when off:
  //   E-1  apex_residual grows an EGRESS job port (ej_*, R3 window-base
  //        addressing) + fp16 stream (ev/er/edata) — the row's internal
  //        exit. Feeder input source code 4 (l_fsrc_ext[2] at
  //        LAYER_CTRL[7]) widens those beats f16->f32 into the C-1 feeder,
  //        exactly the q-sink idiom (code 3).
  //   E-2a l_nsrc (LAYER_CTRL[19]): the feeder's INT8 codes, unpacked
  //        lane8 -> serial by apex_lane8_unpack, drive the norm's x port
  //        instead of xa_* — golden's own layer-entry order
  //        `rmsnorm_fx(quant_rows_i8(row))` (transformer.py:485/:568; the
  //        row scale is DISCARDED there — RMSNorm is scale-invariant — so
  //        the fs tap simply carries it out for grading).
  //   E-2b LAYER_JOB unit 3 = the NORM/EGRESS job: cols ([11:0], must be a
  //        nonzero multiple of the feeder row CFG_DM) + window base
  //        ([15:14]) pushed to ej_*. A push with the feeder input mux NOT
  //        on code 4 would stream into a dead mux — REFUSED at push,
  //        LAYER_STATUS sticky + NEW err_code 9 (NFEED_ROUTE), never a
  //        silent wedge. Geometry (base*1024+cols > DM_MAX, cols==0) is
  //        refused IN-UNIT with the R3 frame_error class (code 3).
  //        Completion observables: LAYER_STATUS[5] = nf_busy (the whole
  //        feed path: job pending | egress | feeder-when-code-4 | unpack),
  //        then dn_rms / rms busy for the norm itself — the walker's
  //        existing lj_* tie (dn_rms) works unchanged, and its LU channel
  //        can issue unit 3 through the same single ingress.
  //   E-3  (2026-08-01) the WALKER can drive all of this. Its lw_fsrc_ext is
  //        3 bits and lands whole (the mux below no longer zero-extends), and
  //        seq_layer_walker2's PC_NFEED step arms code 4, pushes the C-1
  //        feeder job, pushes the unit-3 job through the SAME LU ingress the
  //        host CSR path writes, and waits on nf_busy — so the sequencer,
  //        not the host, drives the residual -> norm chain.
  // Walk mode: l_nsrc HOLDS its host-written value (the l_bias_en idiom)
  // for every legacy image. E-4b (2026-08-04): a walk whose descriptor
  // names the previously-reserved mask bit W2_EN_NSRC makes the WALKER the
  // level's owner — see the l_nsrc_q register below — which is what fused
  // walk_e4's two kicks (QSTAGE needs codes->act, NFEED needs codes->norm;
  // no single host-held value serves both) into one.
  // ═════════════════════════════════════════════════════════════════════════
  logic        nrm_jb_v_q;                // E-2b held-valid egress push
  logic        res_ej_r;                  // residual egress job accept
  logic        res_ev, res_er, res_elast, res_ebusy;
  logic [15:0] res_edata;
  logic        l_nsrc_q;                  // E-2a norm-input-source level
  logic        nup_s_ready, nup_m_valid, nup_m_ready, nup_m_last, nup_busy;
  logic signed [7:0] nup_m_x;
  logic        fq_busy;                   // (decl hoisted from the feeder
                                          //  region for nf_busy below)
  // the LAYER_STATUS[5] poll term: every stage of the feed path in flight.
  // E-3: ALSO fed back to the walker (`nf_busy` is declared with the other
  // wk_* nets above, since u_walk is instantiated before this region) — the
  // walked NFEED step waits on exactly the term the host polls, not a
  // second, parallel notion of "done".
  assign nf_busy = nrm_jb_v_q | res_ebusy | nup_busy
                 | ((l_fsrc_ext_q == 3'd4) && fq_busy);

  // ═════════════════════════════════════════════════════════════════════════
  // R4 (C-RMSW chunk composition, 2026-07-31) — the chunked-norm CSR pair.
  // A NARROW build (RMS_D_MAX=128, the only image aws-fpga#799 lets us fly)
  // computes the FULL wide RMSNorm row as golden rmsnorm_fx_wide's own
  // composition: pass 1 exports each 128-chunk's sum2 through RMS_SUM2's
  // READ view; the host accumulates (pure INT32 adds — golden step 2); pass
  // 2 stages the total back through RMS_SUM2's WRITE view and arms RMS_EXT,
  // and the per-element datapath runs UNCHANGED per chunk with the one
  // broadcast r (μ + rsqrt on the tile, asu_rmsnorm R4 header).
  //   0x90 RMS_SUM2  W: stage ext_sum2[27:0] ([31:28] MUST be 0 — refused,
  //                     never truncated). R: {4'b0, last captured chunk sum}.
  //   0x94 RMS_EXT   W: [0] sum_en, [1] ext_en, [8:2] k = D_wide/128,
  //                     [9] capture-count clear.
  //                  R: {8'b0, scnt[7:0], 1'b0, k[6:0], 6'b0, ext_en, sum_en}.
  // REFUSALS (LOUD, R3's idiom: LAYER_STATUS[8] sticky + err_code 8, state
  // unchanged — the write NEVER partially lands): any write while the norm
  // unit is busy (the levels are quasi-static by construction); sum_en and
  // ext_en both set; ext arm with k outside [2, 64] (the generated μ-table
  // envelope, asu_wide_rms_params.svh); ext arm or sum2 stage violating
  // sum2 <= k·2^21 (the all-(−128) reachable max — pins mean2 <= 2^14, the
  // rsqrt range proof); a nonzero [31:28] on the sum2 stage.
  // Walk mode never touches these addresses, so walked jobs are unaffected;
  // with both levels 0 every replayed program is byte-identical.
  // ═════════════════════════════════════════════════════════════════════════
  localparam logic [7:0] RMS_SUM2_ADDR = 8'h90;
  localparam logic [7:0] RMS_EXT_ADDR  = 8'h94;

  logic        rms_busy;                 // (decl hoisted from the RMS block)
  logic        rms_xsum_q, rms_xext_q;   // R4 mode levels
  logic [6:0]  rms_xk_q;                 // D_wide / 128
  logic [27:0] rms_xs2_q;                // staged external sum2
  logic [7:0]  rms_scnt_q;               // pass-1 capture count
  logic [27:0] rms_s2cap_q;              // last captured chunk sum
  logic        rms_erd_q;                // 1-cycle read pipeline (ERR/WALK
  logic [31:0] rms_erdata_q;             //  pattern)
  logic        rms_s2_push;              // from u_rms (R4 exports)
  logic [27:0] rms_s2_val;

  // ── CSR window decode (falls through csr_regs' reserved path) ────────────
  logic lw_ctrl, lw_ptr, lw_data, lw_job, lw_stat, lw_rptr, lw_jobc;
  assign lw_ctrl = csr_write && (csr_addr == LAYER_CTRL_ADDR);
  assign lw_ptr  = csr_write && (csr_addr == LAYER_PTR_ADDR);
  assign lw_data = csr_write && (csr_addr == LAYER_DATA_ADDR);
  assign lw_job  = csr_write && (csr_addr == LAYER_JOB_ADDR);
  assign lw_stat = csr_write && (csr_addr == LAYER_STATUS_ADDR);
  assign lw_rptr = csr_write && (csr_addr == LAYER_RPTR_ADDR);
  assign lw_jobc = csr_write && (csr_addr == LAYER_JOBC_ADDR);
  wire   lr_rdata = csr_read && (csr_addr == LAYER_RDATA_ADDR);

  // R4 decode + refusal terms (header above). The bound compare uses the
  // value being STAGED (sum2 write) or the k being ARMED (ext write) against
  // the OTHER side's registered state, so the pair is legal in either write
  // order and can never be armed inconsistent.
  wire lw_rs2  = csr_write && (csr_addr == RMS_SUM2_ADDR);
  wire lw_rext = csr_write && (csr_addr == RMS_EXT_ADDR);
  wire        rx_sum = csr_wdata[0];
  wire        rx_ext = csr_wdata[1];
  wire [6:0]  rx_k   = csr_wdata[8:2];
  wire rms_r4_busy_err  = (lw_rs2 || lw_rext) && rms_busy;
  wire rms_r4_hi_err    = lw_rs2 && !rms_busy && (csr_wdata[31:28] != 4'h0);
  wire rms_r4_s2bnd_err = lw_rs2 && !rms_busy && rms_xext_q
                        && ({4'b0, csr_wdata[27:0]} > (32'(rms_xk_q) << 21));
  wire rms_r4_modes_err = lw_rext && !rms_busy && rx_sum && rx_ext;
  wire rms_r4_k_err     = lw_rext && !rms_busy && rx_ext
                        && ((rx_k < 7'd2) || (rx_k > 7'd64));
  wire rms_r4_kbnd_err  = lw_rext && !rms_busy && rx_ext && !rms_r4_k_err
                        && ({4'b0, rms_xs2_q} > (32'(rx_k) << 21));
  wire rms_r4_refuse    = rms_r4_busy_err | rms_r4_hi_err | rms_r4_s2bnd_err
                        | rms_r4_modes_err | rms_r4_k_err | rms_r4_kbnd_err;

  // IC-BIAS taps: pb_* are tied 0 in a PROJ_BIAS_EN=0 build (the g_pbias_off
  // branch), so this expression and the code chain below are unchanged there.
  logic pb_werr, pb_cerr, pb_jerr, pb_ferr, pb_busy_r, l_bias_en_r;

  wire l_any_err = ldq_jerr | swg_jerr | ldq_xerr | res_werr
                 | ldq_ferr | swg_ferr | res_ferr | rr_ferr
                 | pb_werr | pb_cerr | pb_jerr | pb_ferr;

  // ── walker ingress onto the §3b window (combine W-G3 flip; R5 single
  // ingress): in walk mode the walker2 nets drive the SAME l_*_q registers,
  // JOB push, and JOBC file the host CSR path writes — the B1 stage-5
  // host/walker mode-mux idiom (walk_en_q selects, exactly like the (b)
  // fanout muxes above). Host mode (walk_en_q=0) is the untouched S4 path,
  // byte-identical. The walker JOB channel is valid/ready with ready = "no
  // unit-job pending", so choreography rule 3's at-most-ONE-pending-push is
  // enforced by backpressure and the push-while-pending JOB error stays a
  // host-path-only event; the JOBC file always accepts (two registers).
  wire l_units_idle = !(ldq_jb_v_q || swg_jb_v_q || res_jb_v_q
                        || nrm_jb_v_q);
  assign wk_lu_ready = walk_en_q && l_units_idle;
  assign wk_jc_ready = walk_en_q;
  wire        l_push      = walk_en_q ? (wk_lu_valid && wk_lu_ready) : lw_job;
  wire [1:0]  l_push_unit = walk_en_q ? wk_lu_unit : csr_wdata[13:12];
  wire [11:0] l_push_cols = walk_en_q ? wk_lu_cols : csr_wdata[11:0];
  // R3: LAYER_JOB[15:14] = the RESIDUAL unit's window base (stride 1024,
  // apex_residual jb_base; other units ignore it). The walker LU channel
  // carries no base — walked residual jobs stay window-0 by construction.
  wire [1:0]  l_push_base = walk_en_q ? 2'd0 : csr_wdata[15:14];
  wire        l_jc_wr     = walk_en_q ? wk_jc_valid : lw_jobc;
  wire [31:0] l_jc_data   = walk_en_q ? wk_jc_data  : csr_wdata;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      l_rope_en_q  <= 1'b0;
      l_rope_bank_q <= 1'b0;
      l_resid_arm_q <= 1'b0;
      l_ser_dst_q  <= '0;
      l_fsrc_ext_q <= '0;
      l_rope_pos_q <= '0;
      l_ptr_q      <= '0;
      l_rptr_q     <= '0;
      ljc_a_q      <= '0;
      ljc_b_q      <= '0;
      ljc_sel_q    <= 1'b0;
      lj_cols_q    <= '0;
      lj_base_q    <= '0;
      ldq_jb_v_q   <= 1'b0;
      swg_jb_v_q   <= 1'b0;
      res_jb_v_q   <= 1'b0;
      nrm_jb_v_q   <= 1'b0;
      layer_rd_q   <= 1'b0;
      layer_rdata_q <= '0;
      l_err_stk_q  <= 1'b0;
      l_err_code_q <= '0;
      rr_ph_data   <= '0;
    end else begin
      // levels (quasi-static: change only while the touched paths are idle).
      // Walk mode: the registers TRACK walker2's registered level nets (the
      // ROM's LVL steps re-arm every field absolutely; the 1-cycle copy lag
      // is settled long before any stream flows). Host mode: the S4 CSR
      // write path, byte-identical.
      if (walk_en_q) begin
        l_rope_en_q   <= wk_lw_rope_en;
        l_rope_bank_q <= wk_lw_rope_bank;
        l_ser_dst_q   <= wk_lw_ser_dst;
        // E-3 (2026-08-01): the walker level net is 3 BITS and lands WHOLE.
        // This line used to read `{1'b0, wk_lw_fsrc_ext}` — a deliberate
        // zero-extend of a 2-bit net, and the exact reason a WALKED program
        // could not arm the E-1 residual egress (code 4). Widened, so the
        // walker's NFEED step can name it; codes 0-3 are unchanged, and a
        // legacy walked program (whose LVL steps only ever write 0 or 2)
        // still drives bit [2] low every cycle it drives this register.
        l_fsrc_ext_q  <= wk_lw_fsrc_ext;
        l_resid_arm_q <= wk_lw_resid_arm;
        l_rope_pos_q  <= wk_lw_rope_pos;
      end else if (lw_ctrl) begin
        l_rope_en_q   <= csr_wdata[0];
        l_rope_bank_q <= csr_wdata[1];
        l_ser_dst_q   <= csr_wdata[3:2];
        // E-1: [7] is l_fsrc_ext[2] (reserved-0 pre-E1: every legacy level
        // word writes 0 there, so codes 0-3 encode exactly as before)
        l_fsrc_ext_q  <= {csr_wdata[7], csr_wdata[5:4]};
        l_resid_arm_q <= csr_wdata[6];
        l_rope_pos_q  <= csr_wdata[14:8];
        // csr_wdata[17:15] = l_kv_map — latched by the S4b GQA region's
        // parameter-gated register (reserved-0/ignored at KVQ_GQA_NENG==1)
      end
      // load pointer + memories (auto-inc on DATA)
      if (lw_ptr) l_ptr_q <= {csr_wdata[29:28], 12'b0, csr_wdata[15:0]};
      if (lw_data) begin
        unique case (l_ptr_q[29:28])
          2'd0: ph_k[l_ptr_q[$clog2(L_PHKN)-1:0]] <= csr_wdata[13:0];
          2'd1: ph_q[l_ptr_q[L_PAIRW-1:0]]        <= csr_wdata[13:0];
          default: ;                        // bank 2 handled by res_lw_en
        endcase
        l_ptr_q[15:0] <= l_ptr_q[15:0] + 16'd1;
      end
      // JOBC alternating composite file; index resets on JOB push (source =
      // the walker ingress mux above; host mode is the untouched path)
      if (l_jc_wr) begin
        if (!ljc_sel_q) ljc_a_q <= l_jc_data;
        else            ljc_b_q <= l_jc_data;
        ljc_sel_q <= ~ljc_sel_q;
      end
      // JOB push -> per-unit held valid (cleared on unit accept)
      if (l_push) begin
        ljc_sel_q <= 1'b0;
        if (!l_units_idle) begin
          l_err_stk_q  <= 1'b1;            // push while pending: JOB error
          l_err_code_q <= 4'd1;            // (host path only — walker ready
        end else begin                     //  is gated on l_units_idle)
          lj_cols_q <= l_push_cols;
          lj_base_q <= l_push_base;
          unique case (l_push_unit)
            2'd0: ldq_jb_v_q <= 1'b1;
            // R1: asu_swiglu's jb_cols port is $clog2(COLS_MAX+1) = 7 of
            // these 12 bits, so cols beyond the unit's capacity must be
            // REFUSED HERE — the unit's own job_error class, code 2 —
            // never truncated: 12'd2564 arrived as 7'd4 and quietly
            // computed 4 of the 2564 columns asked (no such gate is needed
            // for deq, whose COLS_MAX is the field maximum, or residual,
            // which takes all 12 bits and refuses base*1024 + cols > DM_MAX
            // in-unit — R3 window base at LAYER_JOB[15:14]).
            2'd1: begin
              if (l_push_cols > 12'(WALK2_N_LU_SWIGLU)) begin
                l_err_stk_q  <= 1'b1;
                l_err_code_q <= 4'd2;
              end else begin
                swg_jb_v_q <= 1'b1;
              end
            end
            2'd2: res_jb_v_q <= 1'b1;
            // E-2b: unit 3 = the NORM/EGRESS job — stream cols fp16 beats
            // of the resident row (R3 window base) into the C-1 feeder.
            // ROUTE-ARM refusal (R1's push-gate idiom): with the feeder
            // input mux not on code 4 the stream would drive a dead mux
            // and wedge silently; with cols not framing whole CFG_DM rows
            // the feeder would starve mid-row. Both REFUSED HERE, LOUDLY —
            // err_code 9 (NFEED_ROUTE). Geometry (base/cols vs DM_MAX) is
            // the unit's own accept-time frame_error refusal (code 3).
            default: begin
              if ((l_push_unit == 2'd3)
                  && (l_fsrc_ext_q == 3'd4)
                  && (l_push_cols != '0)
                  && ((l_push_cols % 12'(CFG_DM)) == '0)) begin
                nrm_jb_v_q <= 1'b1;
              end else begin
                l_err_stk_q  <= 1'b1;
                l_err_code_q <= (l_push_unit == 2'd3) ? 4'd9 : 4'd1;
              end
            end
          endcase
        end
      end
      if (ldq_jb_v_q && ldq_jb_r) ldq_jb_v_q <= 1'b0;
      if (swg_jb_v_q && swg_jb_r) swg_jb_v_q <= 1'b0;
      if (res_jb_v_q && res_jb_r) res_jb_v_q <= 1'b0;
      if (nrm_jb_v_q && res_ej_r) nrm_jb_v_q <= 1'b0;
      // read pointer (auto-inc on RDATA read)
      if (lw_rptr) l_rptr_q <= csr_wdata[15:0];
      else if (lr_rdata) l_rptr_q <= l_rptr_q + 16'd1;
      // error sticky/code: W1C on [8]; a same-cycle set WINS
      if (lw_stat && csr_wdata[8]) l_err_stk_q <= 1'b0;
      // (code 8 = NORM_EXT is the R4 refusal below; code 9 = NFEED_ROUTE
      //  is the E-2b push-gate refusal above. The residual EGRESS geometry
      //  refusal arrives here as res_ferr -> the FRAME class, exactly like
      //  the R3 add-job geometry refusal.)
      if (l_any_err) begin
        l_err_stk_q <= 1'b1;
        l_err_code_q <= (ldq_jerr | swg_jerr | pb_jerr)   ? 4'd2   // GRADE/JOB
                      : (ldq_xerr)                       ? 4'd4   // EXACT
                      : (res_werr)                       ? 4'd5   // RESID_WINDOW
                      : (pb_werr)                        ? 4'd6   // BIAS_WINDOW
                      : (pb_cerr)                        ? 4'd7   // BIAS_CONTRACT
                                                         : 4'd3;  // FRAME
      end
      // R4: a refused RMS_SUM2/RMS_EXT write (header above). Placed after
      // the unit-error mux so a same-cycle collision (a CSR write racing a
      // unit pulse — not reachable from any emitted program) reports the
      // refusal's own code, never a blend.
      if (rms_r4_refuse) begin
        l_err_stk_q  <= 1'b1;
        l_err_code_q <= 4'd8;                            // NORM_EXT
      end
      // read pipeline (1-cycle, read-before-write — the WALK/ERR pattern)
      layer_rd_q <= csr_read && ((csr_addr == LAYER_CTRL_ADDR)
                              || (csr_addr == LAYER_PTR_ADDR)
                              || (csr_addr == LAYER_STATUS_ADDR)
                              || (csr_addr == LAYER_RPTR_ADDR)
                              || (csr_addr == LAYER_RDATA_ADDR));
      unique case (csr_addr)
        // E-1/E-2 read view: [7] = l_fsrc_ext[2], [19] = l_nsrc — both
        // reserved-0 pre-E1, so every legacy program reads byte-identically
        LAYER_CTRL_ADDR:  layer_rdata_q <= {12'b0, l_nsrc_q,
                                            l_bias_en_r, l_kv_map_r,
                                            l_rope_pos_q,
                                            l_fsrc_ext_q[2], l_resid_arm_q,
                                            l_fsrc_ext_q[1:0], l_ser_dst_q,
                                            l_rope_bank_q, l_rope_en_q};
        LAYER_PTR_ADDR:   layer_rdata_q <= {2'b0, l_ptr_q};
        // [5] = nf_busy (E-2b): the whole residual->feeder->norm feed path
        // in flight — the host/walker poll term; 0 in every legacy program
        LAYER_STATUS_ADDR: layer_rdata_q <= {19'b0, l_err_code_q,
                                             l_err_stk_q, 2'b0, nf_busy,
                                             pb_busy_r, rr_busy,
                                             res_busy, swg_busy, ldq_busy};
        LAYER_RPTR_ADDR:  layer_rdata_q <= {16'b0, l_rptr_q};
        default:          layer_rdata_q <= {16'b0, res_rd_data};  // RDATA
      endcase
      // phase RAM registered read for rope_row
      rr_ph_data <= l_rope_bank_q
                  ? ph_q[rr_ph_addr]
                  : ph_k[$clog2(L_PHKN)'({6'b0, l_rope_pos_q} * 13'(L_HALF))
                         + $clog2(L_PHKN)'(rr_ph_addr)];
    end
  end

  assign res_lw_en = lw_data && (l_ptr_q[29:28] == 2'd2);

  // ── R4 register file (header above; refusal terms set the LAYER sticky in
  // the block above — an accepted write lands WHOLE, a refused one not at
  // all). The capture side latches u_rms's s2 export and counts it; the
  // count is the pass-1 completion observable (no done pulse in sum mode).
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rms_xsum_q   <= 1'b0;
      rms_xext_q   <= 1'b0;
      rms_xk_q     <= '0;
      rms_xs2_q    <= '0;
      rms_scnt_q   <= '0;
      rms_s2cap_q  <= '0;
      rms_erd_q    <= 1'b0;
      rms_erdata_q <= '0;
    end else begin
      if (lw_rs2 && !rms_r4_refuse) rms_xs2_q <= csr_wdata[27:0];
      if (lw_rext && !rms_r4_refuse) begin
        rms_xsum_q <= rx_sum;
        rms_xext_q <= rx_ext;
        if (rx_ext) rms_xk_q <= rx_k;
        if (csr_wdata[9]) begin
          rms_scnt_q  <= '0;
          rms_s2cap_q <= '0;
        end
      end
      if (rms_s2_push) begin
        rms_s2cap_q <= rms_s2_val;
        rms_scnt_q  <= rms_scnt_q + 8'd1;
      end
      // read pipeline (1-cycle, read-before-write — the WALK/ERR pattern)
      rms_erd_q    <= csr_read && ((csr_addr == RMS_SUM2_ADDR)
                                || (csr_addr == RMS_EXT_ADDR));
      rms_erdata_q <= (csr_addr == RMS_SUM2_ADDR)
                    ? {4'b0, rms_s2cap_q}
                    : {8'b0, rms_scnt_q, 1'b0, rms_xk_q,
                       6'b0, rms_xext_q, rms_xsum_q};
    end
  end

  // ── E-2a l_nsrc level (LAYER_CTRL[19]) — host CSR path, PLUS the E-4b
  // walker-owned window. Host mode: the S4 write, byte-identical. Walk
  // mode: the level HOLDS its host-written value (the l_bias_en idiom)
  // UNLESS the walker has claimed ownership — wk_lw_nsrc_own is high, from
  // the descriptor-check pass to walk end, only for a walk whose mask names
  // the previously-reserved W2_EN_NSRC bit — in which case the register
  // TRACKS wk_lw_nsrc exactly like the (b)-mux levels above. That opt-in is
  // what lets ONE kick both stage q (codes -> act stage, nsrc 0) and run
  // NFEED (codes -> norm, nsrc 1): E-4's measured two-kick finding,
  // retired. Every legacy image (bit 13 = 0) never raises the own level,
  // so its walks are byte-identical to the landed hold behavior.
  // 0 (reset) = the norm's x input is the top-level xa_* port and the
  // feeder codes go to the stage buffers — the pre-E2 tile byte-identically.
  always_ff @(posedge clk) begin
    if (!rst_n)                           l_nsrc_q <= 1'b0;
    else if (walk_en_q && wk_lw_nsrc_own) l_nsrc_q <= wk_lw_nsrc;
    else if (!walk_en_q && lw_ctrl)       l_nsrc_q <= csr_wdata[19];
  end

  // ── the S-2 seam rope mux (l_rope_en=0: pure passthrough) ────────────────
  // GAP A (LEVEL_C_INTEGRATION.md §9.1, IB_LAYER.md §3c-1 blocker A + §3c-2
  // gap C). The seam's PRODUCER is untouched; only its DESTINATION is now
  // steerable, by `l_fsrc_ext`'s RESERVED code 3 (`q_sink`). At q_sink=0
  // every net below reduces to the S4 expression verbatim — byte-identical.
  //
  // At q_sink=1 the same beats go to the FEEDER instead of the KVQ write
  // port, which makes the tile's q staging
  //
  //     narrow   apex_scale_quant MODE_F16   — golden's ONE RNE narrowing
  //     rotate   rope_row (q phase bank)     — the C-ROPE half-split
  //     quant    seam_feeder_quant           — C-1, == quant_rows_i8
  //     stage    act stage
  //
  // i.e. EXACTLY the arbiter's order: `_f16(q_real)` (transformer.py:521,
  // BUS_ON/D-030 NP-r) -> `rope_fx` (:538) -> `quant_rows_i8` (Q7,
  // attention.py:381). MODE_QUANT's Q7 shortcut skipped BOTH steps: it
  // quantizes the EXACT product with no narrowing (its own header, :13-15)
  // and never sees rope_row, and the two modes are mutually exclusive
  // (apex_scale_quant.sv:441) so it could not do otherwise.
  //
  // kv_s_tvalid is forced low here: a q row is NOT a KV record, so it must
  // never reach the KVQ write port, the store-time scale snoop, or a
  // record's amax reduce (u_wcomp/apex_wcomp_bank snoop
  // kv_s_tvalid && kv_s_tready and would commit sc_mem[snoop_addr_q]).
  assign q_sink      = (l_fsrc_ext_q == 3'd3);
  assign rr_s_valid  = l_rope_en_q && sqf_valid;
  assign seam_valid  = l_rope_en_q ? rr_m_valid : sqf_valid;
  assign seam_data   = l_rope_en_q ? rr_m_data  : sqf_data;
  assign seam_last   = l_rope_en_q ? rr_m_last  : sqf_last;
  assign seam_ready  = q_sink ? fq_in_ready : kv_s_tready;
  assign kv_s_tvalid = seam_valid && !q_sink;
  assign kv_s_tdata  = seam_data;
  assign kv_s_tlast  = seam_last;
  assign sqf_ready   = l_rope_en_q ? rr_s_ready : seam_ready;
  assign rr_m_ready  = l_rope_en_q && seam_ready;

  rope_row #(.D(CFG_D)) u_rope_row (
    .clk (clk), .rst_n (rst_n),
    .s_valid (rr_s_valid), .s_ready (rr_s_ready),
    .s_data (sqf_data), .s_last (sqf_last),
    .m_valid (rr_m_valid), .m_ready (rr_m_ready),
    .m_data (rr_m_data), .m_last (rr_m_last),
    .ph_addr (rr_ph_addr), .ph_data (rr_ph_data),
    .busy (rr_busy), .frame_error (rr_ferr),
    .frame_error_sticky (rr_ferr_stk)
  );

  // ── layer-deq (serializer -> exact fp32) + its output fanout ─────────────
  assign ldq_iv = ls_out_valid && (l_ser_dst_q == 2'd1);
  assign ldq_or = l_resid_arm_q ? res_ir
                : (l_fsrc_ext_q == 3'd1) ? fq_in_ready : 1'b0;

  apex_layer_deq #(.COLS_MAX(4095)) u_ldeq (
    .clk (clk), .rst_n (rst_n),
    .jb_valid (ldq_jb_v_q), .jb_ready (ldq_jb_r),
    .jb_cols (lj_cols_q), .jb_comp (ljc_a_q),
    .iv (ldq_iv), .ir (ldq_ir_port), .idata (ls_out_data),
    .ilast (ls_out_last),
    .ov (ldq_ov), .orr (ldq_or), .odata (ldq_odata), .olast (ldq_olast),
    .busy (ldq_busy), .done (ldq_done),
    .job_error (ldq_jerr), .job_error_sticky (ldq_jerr_stk),
    .exact_error (ldq_xerr), .exact_error_sticky (ldq_xerr_stk),
    .frame_error (ldq_ferr), .frame_error_sticky (ldq_ferr_stk)
  );

  // ── swiglu (serializer gate/up phases -> fp16 products) ──────────────────
  // B-SWG-PHASE fix (R2). One asu_swiglu job = a GATE frame THEN an UP frame
  // on two SEPARATE skid-buffered ports (asu_swiglu.sv:106-282), but each
  // skid's s_ready is `~skid_valid` (stream_skid.sv:113) — registered and
  // INDEPENDENT of the unit's phase. Driving both valids from the same
  // serializer beat therefore latched that beat into BOTH skids whenever
  // both had room: the up skid swallowed the first 2 gate beats (and vice
  // versa) and the up frame's `last` landed 2 beats early — LAYER_STATUS
  // FRAME (err_code 3), job aborted, products lost. Every serializer beat
  // must go to EXACTLY the active phase's skid.
  //
  // The steering phase is tracked HERE, at the skid INPUT boundary, by
  // counting accepted `last` beats (frame boundary == phase boundary,
  // asu_swiglu.sv:246/:263) — NOT from the unit's FSM state: the FSM runs up
  // to 2 skid beats BEHIND this boundary, so st can still be ST_GATE while
  // the first up beats arrive. Reset state is the gate phase (a job's first
  // frame). NOTE on aborts: a malformed GATE frame (frame_error) leaves this
  // tracker one phase ahead of the re-idled unit; the FRAME sticky is
  // latched and recovery is a hard reset — exactly the unit's own contract
  // for its silu_buf/sticky state (asu_swiglu.sv:23-24).
  logic swg_up_q;
  wire  swg_sv = ls_out_valid && (l_ser_dst_q == 2'd2);
  wire  swg_sr = swg_up_q ? swg_ur : swg_gr;
  assign swg_gv = swg_sv && !swg_up_q;
  assign swg_uv = swg_sv &&  swg_up_q;
  always_ff @(posedge clk) begin
    if (!rst_n)                              swg_up_q <= 1'b0;
    else if (swg_sv && swg_sr && ls_out_last) swg_up_q <= ~swg_up_q;
  end
  assign swg_pr = (l_fsrc_ext_q == 3'd2) && fq_in_ready;

  // COLS_MAX = seq_walker_pkg::WALK2_N_LU_SWIGLU (single source: the
  // walker2 chunk bound, the R1 push-refusal gate above and this unit's
  // capacity cannot drift apart). The jb_cols narrowing is safe ONLY
  // because the push gate refuses cols > COLS_MAX before a job is issued.
  asu_swiglu #(.COLS_MAX(WALK2_N_LU_SWIGLU)) u_swg (
    .clk (clk), .rst_n (rst_n),
    .jb_valid (swg_jb_v_q), .jb_ready (swg_jb_r),
    .jb_cols (($clog2(WALK2_N_LU_SWIGLU + 1))'(lj_cols_q)),
    .jb_comp_g (ljc_a_q), .jb_comp_u (ljc_b_q),
    .gv (swg_gv), .gr (swg_gr), .gdata (ls_out_data), .glast (ls_out_last),
    .uv (swg_uv), .ur (swg_ur), .udata (ls_out_data), .ulast (ls_out_last),
    .pv (swg_pv), .pr (swg_pr), .pdata (swg_pdata), .plast (swg_plast),
    .busy (swg_busy), .done (swg_done),
    .job_error (swg_jerr), .job_error_sticky (swg_jerr_stk),
    .frame_error (swg_ferr), .frame_error_sticky (swg_ferr_stk)
  );

  // ── residual unit (deq stream consumer when armed) ───────────────────────
  assign res_iv = ldq_ov && l_resid_arm_q;

  apex_residual #(.DM_MAX(LAYER_DM_MAX)) u_resid (
    .clk (clk), .rst_n (rst_n),
    .lw_en (res_lw_en), .lw_addr (l_ptr_q[L_DMAW-1:0]),
    .lw_data (csr_wdata[15:0]),
    .rd_addr (l_rptr_q[L_DMAW-1:0]), .rd_data (res_rd_data),
    .jb_valid (res_jb_v_q), .jb_ready (res_jb_r), .jb_cols (lj_cols_q),
    .jb_base (lj_base_q),
    .iv (res_iv), .ir (res_ir), .idata (ldq_odata), .ilast (ldq_olast),
    // E-1 egress: the unit-3 job (cols/base share the push registers — one
    // pending LAYER job at a time, l_units_idle) streams the resident row
    .ej_valid (nrm_jb_v_q), .ej_ready (res_ej_r), .ej_cols (lj_cols_q),
    .ej_base (lj_base_q),
    .ev (res_ev), .er (res_er), .edata (res_edata), .elast (res_elast),
    .ebusy (res_ebusy),
    .busy (res_busy), .done (res_done),
    .frame_error (res_ferr), .frame_error_sticky (res_ferr_stk),
    .window_error (res_werr), .window_error_sticky (res_werr_stk)
  );

  // observability spares / documented-unused (the units' own stickies are
  // aggregated into LAYER_STATUS; done pulses are host-poll fodder)
  logic unused_layer_ok;
  assign unused_layer_ok = &{1'b0, ldq_done, swg_done, res_done, swg_plast,
                             rr_ferr_stk, ldq_jerr_stk, ldq_xerr_stk,
                             ldq_ferr_stk, swg_jerr_stk, swg_ferr_stk,
                             res_ferr_stk, res_werr_stk, l_ptr_q[27:16],
                             // E-1: the feeder frames by element count; a
                             // rows>1 misuse is refused by the norm's own
                             // length legality (unpack header)
                             res_elast};

  // ── SEQ <-> MXE (verified D-006 path) ─────────────────────────────────────
  logic      md_valid, md_ready;
  mxe_desc_t md_desc;
  logic      mxe_desc_error, mxe_desc_error_sticky, mxe_busy;
  logic      seq_desc_error, seq_q_empty, seq_q_full, seq_busy, seq_aborting;
  logic [$clog2(SEQ_QDEPTH+1)-1:0] seq_q_count;
  logic      csr_enable, csr_soft_reset;

  seq_walker #(.QDEPTH(SEQ_QDEPTH)) u_seq (
    .clk            (clk),
    .rst_n          (rst_n),
    .enable         (csr_enable),
    .abort_req      (csr_soft_reset),
    .ds_valid       (w_ds_valid),
    .ds_ready       (w_ds_ready),
    .ds_desc        (w_ds_desc),
    .md_valid       (md_valid),
    .md_ready       (md_ready),
    .md_desc        (md_desc),
    .mxe_done       (dn_mxe),
    .mxe_desc_error (mxe_desc_error),
    .desc_error     (seq_desc_error),
    .q_empty        (seq_q_empty),
    .q_full         (seq_q_full),
    .q_count        (seq_q_count),
    .busy           (seq_busy),
    .aborting       (seq_aborting)
  );

  // ── MXE ────────────────────────────────────────────────────────────────────
  logic         mxe_act_valid, mxe_act_ready;
  lane8_beat_t  mxe_act_beat;
  logic         mxe_wgt_valid, mxe_wgt_ready;
  lane8_beat_t  mxe_wgt_beat;
  logic         mxe_res_valid, mxe_res_ready;
  lane32_beat_t mxe_res_beat;

  mxe_top u_mxe (
    .clk               (clk),
    .rst_n             (rst_n),
    .desc_valid        (md_valid),
    .desc_ready        (md_ready),
    .desc              (md_desc),
    .desc_error        (mxe_desc_error),
    .desc_error_sticky (mxe_desc_error_sticky),
    .busy              (mxe_busy),
    .done              (dn_mxe),
    .act_valid         (mxe_act_valid),
    .act_ready         (mxe_act_ready),
    .act_beat          (mxe_act_beat),
    .wgt_valid         (mxe_wgt_valid),
    .wgt_ready         (mxe_wgt_ready),
    .wgt_beat          (mxe_wgt_beat),
    .res_valid         (mxe_res_valid),
    .res_ready         (mxe_res_ready),
    .res_beat          (mxe_res_beat)
  );

  // ── ASU RMSNorm (tile data entry) ─────────────────────────────────────────
  logic               rms_m_valid, rms_m_ready;
  logic signed [15:0] rms_m_y;
  logic               rms_m_last;
  logic               rms_len_error, rms_len_error_sticky;
  logic [13:0]        rms_dbg_norm;   // (rms_busy decl hoisted to the R4
                                      //  CSR-pair region above)

  // E-2a: the norm x-input mux. l_nsrc=0 reduces every net to the xa_* port
  // verbatim (byte-identical); l_nsrc=1 feeds the tile's OWN C-1 codes
  // (feeder -> apex_lane8_unpack below) — golden's layer-entry order
  // `rmsnorm_fx(quant_rows_i8(row))`, with the row never leaving the tile.
  logic              nrm_s_valid, nrm_s_ready, nrm_s_last;
  logic signed [7:0] nrm_s_x;
  assign nrm_s_valid = l_nsrc_q ? nup_m_valid : xa_valid;
  assign nrm_s_x     = l_nsrc_q ? nup_m_x     : xa_x;
  assign nrm_s_last  = l_nsrc_q ? nup_m_last  : xa_last;
  assign xa_ready    = !l_nsrc_q && nrm_s_ready;
  assign nup_m_ready =  l_nsrc_q && nrm_s_ready;

  // ── E-7: the norm GAMMA-input mux — the xw -> xg route that never
  // existed (the measured NORM walk fence). gam_win_q is the registered
  // copy of the walker's lw_gsrc level (the fsrc_ext/l_nsrc idiom; the
  // settle margin is the whole fuel-record round trip, cycles vs the one
  // register). gam_win_q=0 reduces every net to the xg_* port verbatim —
  // byte-identical for every legacy walk and all of host mode — and no
  // legacy image can raise it (W2_MASK[15] was reserved-refused).
  logic               gam_win_q;
  logic               gu_s_valid, gu_s_ready, gu_m_valid, gu_m_ready;
  logic signed [15:0] gu_m_gamma;
  logic               gu_busy;
  logic               rms_g_valid, rms_g_ready;
  logic signed [15:0] rms_g_gamma;
  always_ff @(posedge clk) begin
    if (!rst_n) gam_win_q <= 1'b0;
    else        gam_win_q <= walk_en_q && wk_lw_gsrc;
  end

  apex_gam_unpack u_gam_unpack (
    .clk (clk), .rst_n (rst_n),
    .win (gam_win_q),
    .s_valid (gu_s_valid), .s_ready (gu_s_ready), .s_data (xw_beat.data),
    .m_valid (gu_m_valid), .m_ready (gu_m_ready), .m_gamma (gu_m_gamma),
    .busy (gu_busy)
  );

  assign rms_g_valid = gam_win_q ? gu_m_valid : xg_valid;
  assign rms_g_gamma = gam_win_q ? gu_m_gamma : xg_gamma;
  assign xg_ready    = !gam_win_q && rms_g_ready;
  assign gu_m_ready  =  gam_win_q && rms_g_ready;

  asu_rmsnorm #(.RMS_D_MAX(RMS_D_MAX)) u_rms (   // IB-LAYER S4: parameterized
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (nrm_s_valid),
    .s_ready          (nrm_s_ready),
    .s_x              (nrm_s_x),
    .s_last           (nrm_s_last),
    .g_valid          (rms_g_valid),
    .g_ready          (rms_g_ready),
    .g_gamma          (rms_g_gamma),
    .m_valid          (rms_m_valid),
    .m_ready          (rms_m_ready),
    .m_y              (rms_m_y),
    .m_last           (rms_m_last),
    .busy             (rms_busy),
    .done             (dn_rms),
    .len_error        (rms_len_error),
    .len_error_sticky (rms_len_error_sticky),
    .dbg_norm         (rms_dbg_norm),
    // R4 chunk composition (the CSR pair 0x90/0x94 above)
    .ext_sum_en       (rms_xsum_q),
    .ext_r_en         (rms_xext_q),
    .ext_sum2         (rms_xs2_q),
    .ext_k            (rms_xk_q),
    .s2_push          (rms_s2_push),
    .s2_val           (rms_s2_val)
  );

  // ── exact Q7.8 -> fp32 widen (glue; S-1 prep) ─────────────────────────────
  logic        wid_m_valid, wid_m_ready;
  logic [31:0] wid_m_data;

  apex_q78_to_fp32 u_widen (
    .s_valid (rms_m_valid),
    .s_ready (rms_m_ready),
    .s_y     (rms_m_y),
    .s_last  (rms_m_last),
    .m_valid (wid_m_valid),
    .m_ready (wid_m_ready),
    .m_data  (wid_m_data)
  );

  // ── KVQ tier bank (D-024: 3 verified engines behind the live tier mux) ───
  logic        kv_s_tvalid, kv_s_tready, kv_s_tlast;
  logic [15:0] kv_s_tdata;
  logic [31:0] kv_m_tdata;
  logic        kv_m_tvalid, kv_m_tready, kv_m_tlast;
  logic        kv_m_pending;
  logic        kv_mask_valid_cq4p;    // D-027 (S12): e2 live-mask validity
  logic        csr_flush;

  // AUTO-mode per-block tier map: written by every ACCEPTED TIP decision
  // beat (D-022 actuation — TIP DRIVES the tier); reset to CQ-4 (§4
  // default). Indexed by rt_tip_blk when TIER_CTRL.tip_override is set.
  kvq_tier_e   auto_tier [128];
  kvq_tier_e   live_tier;
  logic        csr_tip_override;
  kvq_tier_e   csr_tier_sel;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int unsigned i = 0; i < 128; i++) auto_tier[i] <= KVQ_CQ4;
    end else if (td_valid && td_ready) begin
      auto_tier[td_blk] <= td_tier;
    end
  end

  assign live_tier = csr_tip_override ? auto_tier[rt_tip_blk] : csr_tier_sel;

  // D-026 integration: the v0.1 host flow stores a whole T <= T_ROW_MAX job's
  // key groups BEFORE reading any record back, so every group's scale row
  // must still be live at read time — the persistent bank needs
  // SETS >= ceil(T_ROW_MAX / KVQ_G), rounded up to a power of two (the
  // engine's commit-time allocator wraps by SSET_W bits), floored at the
  // engine's D-026 default of 4. Undersizing is NOT silent data corruption
  // at the engine level (SB_OVWR fires), but the tile contract is bit-exact
  // K-hat readback across the whole job, so size it here.
  localparam int unsigned KVQ_GROUPS_MAX = (T_ROW_MAX + KVQ_G - 1) / KVQ_G;
  localparam int unsigned KVQ_SETS       = (KVQ_GROUPS_MAX <= 4)
                                         ? 4
                                         : (32'd1 << $clog2(KVQ_GROUPS_MAX));

  // ═══════════════════════════════════════════════════════════════════════════
  // IB-LAYER S4b — per-KV-head CQ-8 GQA banking (LEVEL_C §9.1 R3-AMENDED;
  // approved plan IB_LAYER.md §0.1). NEW fenced region: the ONE elaboration
  // choice for the KVQ subsystem position. KVQ_GQA_NENG==1 (default)
  // elaborates the untouched D-024 tier-bank instantiation byte-identically
  // (its text below is verbatim pre-S4b). KVQ_GQA_NENG>1 elaborates the
  // apex_kvq_gqa_bank sibling instead: N_ENG per-KV-head CQ-8 engines
  // behind the engine select `kv_eng_sel` — the SINGLE ingress declared with
  // the composite bank above (F6: the two banks must switch together), whose
  // host-mode half is the §3b LAYER_CTRL[17:15] l_kv_map carve and whose
  // walk-mode half is walker2's kv_eng_sel. The golden h // (H/H_kv) mapping
  // lives in the SEQUENCER, never here; l_kv_map_r is the CSR read view.
  // ═══════════════════════════════════════════════════════════════════════════

  generate if (KVQ_GQA_NENG == 1) begin : g_kvq_tier

  apex_kvq_bank #(
    .CFG_D     (CFG_D),
    .KVQ_G     (KVQ_G),
    .KVQ_DEPTH (KVQ_DEPTH),
    .KVQ_SETS  (KVQ_SETS),
    .OUTLIER_K (KVQ_OUTLIER_K),
    .MASK_FILE (KVQ_MASK_FILE)
  ) u_kvq (
    .clk              (clk),
    .rst_n            (rst_n),
    .tier_sel         (live_tier),
    .axil_awaddr      (w_kv_awaddr),
    .axil_awvalid     (w_kv_awvalid),
    .axil_awready     (bk_awready),
    .axil_wdata       (w_kv_wdata),
    .axil_wvalid      (w_kv_wvalid),
    .axil_wready      (bk_wready),
    .axil_bresp       (bk_bresp),
    .axil_bvalid      (bk_bvalid),
    .axil_bready      (w_kv_bready),
    .axil_araddr      (w_kv_araddr),
    .axil_arvalid     (w_kv_arvalid),
    .axil_arready     (bk_arready),
    .axil_rdata       (bk_rdata),
    .axil_rresp       (bk_rresp),
    .axil_rvalid      (bk_rvalid),
    .axil_rready      (w_kv_rready),
    .s_axis_kv_tdata  (kv_s_tdata),
    .s_axis_kv_tvalid (kv_s_tvalid),
    .s_axis_kv_tready (kv_s_tready),
    .s_axis_kv_tlast  (kv_s_tlast),
    .s_axis_kv_tuser  (w_rt_kv_user),     // F-2: 0=key (grouped), 1=value
    .m_axis_kv_tdata  (kv_m_tdata),
    .m_axis_kv_tvalid (kv_m_tvalid),
    .m_axis_kv_tready (kv_m_tready),
    .m_axis_kv_tlast  (kv_m_tlast),
    .flush_req        (csr_flush),      // D-008: CSR FLUSH -> engine[tier]
    .irq              (kv_irq),
    .evict_needed     (kv_evict_needed),
    .evict_addr       (kv_evict_addr),
    .m_pending        (kv_m_pending),
    .mask_valid_cq4p  (kv_mask_valid_cq4p)  // D-027: INFO_TIER bit-2 term
  );

  end else begin : g_kvq_gqa

  // ── S4b ON build: per-KV-head CQ-8 banking replaces the tier bank ─────────
  // The engine select is NOT recomputed here: it is `kv_eng_sel`, the single
  // R5 ingress declared with the composite bank (walk mode = walker2's
  // kv_eng_sel — g_idx during head walks, sk_g during STOREKV, both
  // quiescent-switched by the walker's poll-idle invariants; host mode = the
  // l_kv_map level register, quasi-static per the rt_* rule). One net drives
  // both banks so a record and its harvested scale can never diverge (F6).
  // Tier machinery (live_tier / TIP auto map) has no consumer in this
  // CQ-8-only build: consumed by the sink so the surface stays -Wall clean;
  // INFO_TIER reads the CQ-8-only truth via the 0x14 override.
  apex_kvq_gqa_bank #(
    .N_ENG     (KVQ_GQA_NENG),
    .CFG_D     (CFG_D),
    .KVQ_G     (KVQ_G),
    .KVQ_DEPTH (KVQ_DEPTH),
    .KVQ_SETS  (KVQ_SETS)
  ) u_kvq (
    .clk              (clk),
    .rst_n            (rst_n),
    .eng_sel          (kv_eng_sel),
    .axil_awaddr      (w_kv_awaddr),
    .axil_awvalid     (w_kv_awvalid),
    .axil_awready     (bk_awready),
    .axil_wdata       (w_kv_wdata),
    .axil_wvalid      (w_kv_wvalid),
    .axil_wready      (bk_wready),
    .axil_bresp       (bk_bresp),
    .axil_bvalid      (bk_bvalid),
    .axil_bready      (w_kv_bready),
    .axil_araddr      (w_kv_araddr),
    .axil_arvalid     (w_kv_arvalid),
    .axil_arready     (bk_arready),
    .axil_rdata       (bk_rdata),
    .axil_rresp       (bk_rresp),
    .axil_rvalid      (bk_rvalid),
    .axil_rready      (w_kv_rready),
    .s_axis_kv_tdata  (kv_s_tdata),
    .s_axis_kv_tvalid (kv_s_tvalid),
    .s_axis_kv_tready (kv_s_tready),
    .s_axis_kv_tlast  (kv_s_tlast),
    .s_axis_kv_tuser  (w_rt_kv_user),   // CQ-8 K and V ride the value path
    .m_axis_kv_tdata  (kv_m_tdata),
    .m_axis_kv_tvalid (kv_m_tvalid),
    .m_axis_kv_tready (kv_m_tready),
    .m_axis_kv_tlast  (kv_m_tlast),
    .flush_req        (csr_flush),      // D-008 -> engine[eng_sel] (no-op
    .irq              (kv_irq),         //  on an idle CQ-8 engine)
    .evict_needed     (kv_evict_needed),
    .evict_addr       (kv_evict_addr),
    .m_pending        (kv_m_pending)
  );

  // no CQ-4+ lane in this build — the INFO_TIER bit-2 term is 0 by truth
  assign kv_mask_valid_cq4p = 1'b0;
  // KVQ_MASK_FILE has no consumer in a CQ-8-only build; reference it so the
  // parameter surface stays identical across configs (-Wall fatal, no waiver)
  localparam bit GQA_NO_MASK = (KVQ_MASK_FILE == "");
  logic unused_gqa_ok;
  assign unused_gqa_ok = &{1'b0, live_tier, GQA_NO_MASK};

  end endgenerate

  // ── D-021 feeder (fp32 -> INT8 codes + fp16 scales) ───────────────────────
  logic        fq_in_valid, fq_in_ready;
  logic [31:0] fq_in_data;
  logic        fq_out_valid, fq_out_ready;
  lane8_beat_t fq_out_beat;
  logic        fq_job_error, fq_job_error_sticky;  // (fq_busy decl hoisted
                                                   //  to the E-1/E-2 region)

  // GAP D guard (elaboration-time): the model-wide family must be at least
  // as wide as the per-head family and stay lane8-framed.
  if (CFG_DM < CFG_D || (CFG_DM % 8) != 0) begin : g_chk_dm
    $error("apex_top: CFG_DM must be >= CFG_D and a multiple of 8 (GAP D)");
  end

  // GAP D: the feeder is the MODEL-WIDE family — its C-1 rows are the h/h2
  // hidden rows (quant_rows_i8 over D_model, transformer.py:491/:572), so
  // it elaborates at CFG_DM. NOTE the KVQ-readback requant route
  // (rt_feeder_src=1) frames PER-HEAD rows through this same feeder; in a
  // split build that route needs per-job framing (the stage-6 wide-feeder
  // item) and is NOT drivable at CFG_DM != CFG_D — measured and recorded
  // in the GAP-D lane report.
  seam_feeder_quant #(.D(CFG_DM), .ROWS_MAX(FEED_ROWS_MAX)) u_feeder (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (w_fj_valid),
    .job_ready        (w_fj_ready),
    .job_rows         (w_fj_rows),
    .job_error        (fq_job_error),
    .job_error_sticky (fq_job_error_sticky),
    .busy             (fq_busy),
    .done             (dn_feeder),
    .in_valid         (fq_in_valid),
    .in_ready         (fq_in_ready),
    .in_data          (fq_in_data),
    .out_valid        (fq_out_valid),
    .out_ready        (fq_out_ready),
    .out_beat         (fq_out_beat),
    .scl_valid        (fs_valid),
    .scl_ready        (fs_ready),
    .scl_data         (fs_data),
    .scl_last         (fs_last)
  );

  // feeder input mux (route: widen h-path vs KVQ read bus), extended by the
  // IB-LAYER l_fsrc_ext override (0 = legacy, byte-identical; 1 = layer-deq
  // exact fp32; 2 = swiglu-p widened f16->f32; 3 = reserved)
  always_comb begin
    unique case (l_fsrc_ext_q)
      3'd1: begin
        fq_in_valid = ldq_ov && !l_resid_arm_q;
        fq_in_data  = ldq_odata;
      end
      3'd2: begin
        fq_in_valid = swg_pv;
        fq_in_data  = f16_to_f32_bits(swg_pdata);
      end
      3'd3: begin
        // GAP A: the q sink. The rotated fp16 seam beat, widened EXACTLY to
        // fp32 (every fp16 — normal, subnormal, signed zero — is exact in
        // fp32, f16_arith_pkg:168-193), so the feeder's C-1 row quant is
        // golden's `quant_rows_i8` applied to the fp16-grid rotated q row.
        fq_in_valid = seam_valid;
        fq_in_data  = f16_to_f32_bits(seam_data);
      end
      3'd4: begin
        // E-1: the residual row's INTERNAL egress — same exact-widen
        // argument as the q sink above, so the feeder's C-1 is golden's
        // `quant_rows_i8` over the RESIDENT fp16 row (X at layer entry, r1
        // mid-layer). This is the seam the layer chain used to cross the
        // host twice; the row now reaches the norm without leaving the tile.
        fq_in_valid = res_ev;
        fq_in_data  = f16_to_f32_bits(res_edata);
      end
      default: begin
        fq_in_valid = w_rt_feeder_src ? kv_m_tvalid : wid_m_valid;
        fq_in_data  = w_rt_feeder_src ? kv_m_tdata  : wid_m_data;
      end
    endcase
  end
  assign wid_m_ready = (l_fsrc_ext_q == 3'd0) && !w_rt_feeder_src && fq_in_ready;
  assign kv_m_tready = (l_fsrc_ext_q == 3'd0) && w_rt_feeder_src  && fq_in_ready;
  assign res_er      = (l_fsrc_ext_q == 3'd4) && fq_in_ready;

  logic unused_kv_m_tlast;
  assign unused_kv_m_tlast = kv_m_tlast;   // feeder frames by element count

  // ── scale_quant seam block (S-2 / Q7 / S-4) ───────────────────────────────
  logic               sq_v_valid, sq_v_ready, sq_v_last;
  logic signed [31:0] sq_v_data;
  logic               sq_q_valid, sq_q_ready;
  lane8_beat_t        sq_q_beat;
  logic sq_job_error, sq_job_error_sticky;
  logic sq_range_error, sq_range_error_sticky;
  logic sq_scale_error, sq_scale_error_sticky;
  logic sq_frame_error, sq_frame_error_sticky;
  logic sq_busy;

  // per-job element buffers must hold BOTH a D-column S-2/Q7 job and a
  // T-column S-4 P-requant job (F-1 envelope: T <= T_ROW_MAX)
  localparam int unsigned SQ_COLS_MAX = (CFG_D > T_ROW_MAX) ? CFG_D
                                                            : T_ROW_MAX;

  // IC-BIAS: the job / sideband / value / f_* / done endpoints of this
  // instance are nets the IC-BIAS region below drives. In a PROJ_BIAS_EN=0
  // build that region is a set of straight-through assigns (g_pbias_off) and
  // this instance sees exactly the S4 nets it saw before.
  logic sq_job_v, sq_job_r, sq_cs_v, sq_cs_r, sq_v_v, sq_v_r, sq_done;
  logic sqf_sq_valid, sqf_sq_ready, sqf_sq_last;
  logic [15:0] sqf_sq_data;

  apex_scale_quant #(.D(SQ_COLS_MAX)) u_squant (
    .clk                (clk),
    .rst_n              (rst_n),
    .job_valid          (sq_job_v),
    .job_ready          (sq_job_r),
    .job_mode           (w_qj_mode),
    .job_cols           (w_qj_cols),
    .job_error          (sq_job_error),
    .job_error_sticky   (sq_job_error_sticky),
    .busy               (sq_busy),
    .done               (sq_done),
    .cs_valid           (sq_cs_v),
    .cs_ready           (sq_cs_r),
    .cs_data            (w_qs_data),
    .v_valid            (sq_v_v),
    .v_ready            (sq_v_r),
    .v_data             (sq_v_data),
    .v_last             (sq_v_last),
    .f_valid            (sqf_sq_valid), // IB-LAYER S4: S-2 seam via the
    .f_ready            (sqf_sq_ready), // rope mux (l_rope_en=0 is a pure
    .f_data             (sqf_sq_data),  // passthrough, byte-identical)
    .f_last             (sqf_sq_last),
    .q_valid            (sq_q_valid),
    .q_ready            (sq_q_ready),
    .q_beat             (sq_q_beat),
    .s_valid            (ss_valid),
    .s_ready            (ss_ready),
    .s_data             (ss_data),
    .s_last             (ss_last),
    .range_error        (sq_range_error),
    .range_error_sticky (sq_range_error_sticky),
    .scale_error        (sq_scale_error),
    .scale_error_sticky (sq_scale_error_sticky),
    .frame_error        (sq_frame_error),
    .frame_error_sticky (sq_frame_error_sticky)
  );

  // ── lane32 serializer (MXE res -> serial INT32) ───────────────────────────
  logic               ls_in_valid, ls_in_ready;
  logic               ls_out_valid, ls_out_ready, ls_out_last;
  logic signed [31:0] ls_out_data;
  logic               ls_job_error, ls_job_error_sticky, ls_busy;

  apex_lane32_ser #(.BEATS_MAX(255)) u_ser (
    .clk              (clk),
    .rst_n            (rst_n),
    // E-3b: host/walker muxed job port (w_slj_* in the (b) mode mux region)
    .job_valid        (w_slj_valid),
    .job_ready        (w_slj_ready),
    .job_beats        (w_slj_beats),
    .job_lanes        (w_slj_lanes),
    .job_error        (ls_job_error),
    .job_error_sticky (ls_job_error_sticky),
    .busy             (ls_busy),
    .done             (dn_ser),
    .in_valid         (ls_in_valid),
    .in_ready         (ls_in_ready),
    .in_beat          (mxe_res_beat),
    .out_valid        (ls_out_valid),
    .out_ready        (ls_out_ready),
    .out_data         (ls_out_data),
    .out_last         (ls_out_last)
  );

  // ── score dequant + fork + ASU softmax + TIP ──────────────────────────────
  logic               sd_acc_valid, sd_acc_ready;
  logic               sd_sc_valid, sd_sc_ready, sd_sc_last;
  logic signed [31:0] sd_sc_data;
  logic sd_job_error, sd_job_error_sticky;
  logic sd_range_error, sd_range_error_sticky;
  logic sd_frame_error, sd_frame_error_sticky;
  logic sd_busy;

  seam_score_dequant #(.N_MAX(T_ROW_MAX)) u_scored (   // F-1: T envelope
    .clk                (clk),
    .rst_n              (rst_n),
    .job_valid          (w_dj_valid),
    .job_ready          (w_dj_ready),
    .job_cols           (w_dj_cols),
    .job_error          (sd_job_error),
    .job_error_sticky   (sd_job_error_sticky),
    .busy               (sd_busy),
    .done               (dn_scored),
    .cmp_valid          (w_cs_valid),
    .cmp_ready          (w_cs_ready),
    .cmp_data           (w_cs_data),
    .acc_valid          (sd_acc_valid),
    .acc_ready          (sd_acc_ready),
    .acc_beat           (mxe_res_beat),
    .score_valid        (sd_sc_valid),
    .score_ready        (sd_sc_ready),
    .score_data         (sd_sc_data),
    .score_last         (sd_sc_last),
    .range_error        (sd_range_error),
    .range_error_sticky (sd_range_error_sticky),
    .frame_error        (sd_frame_error),
    .frame_error_sticky (sd_frame_error_sticky)
  );

  logic               fk_a_valid, fk_a_ready, fk_a_last;
  logic signed [31:0] fk_a_data;
  logic               fk_b_valid, fk_b_ready, fk_b_last;
  logic signed [31:0] fk_b_data;
  logic               fk_busy;

  apex_score_fork u_fork (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (sd_sc_valid),
    .s_ready (sd_sc_ready),
    .s_data  (sd_sc_data),
    .s_last  (sd_sc_last),
    .a_valid (fk_a_valid),
    .a_ready (fk_a_ready),
    .a_data  (fk_a_data),
    .a_last  (fk_a_last),
    .b_valid (fk_b_valid),
    .b_ready (fk_b_ready),
    .b_data  (fk_b_data),
    .b_last  (fk_b_last),
    .busy    (fk_busy)
  );

  logic        asu_m_valid, asu_m_ready;
  logic [15:0] asu_m_prob;
  logic        asu_m_last;
  logic        asu_busy, asu_row_error, asu_row_error_sticky;

  asu_softmax #(
    .SM_ROW_MAX (SM_ROW_MAX),
    .SCORE_FRAC (ASU_IN_FRAC)            // S-5: SCORE_FRAC = 10, SH = 0
  ) u_asu (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (fk_a_valid),
    .s_ready          (fk_a_ready),
    .s_score          (fk_a_data),
    .s_last           (fk_a_last),
    .m_valid          (asu_m_valid),
    .m_ready          (asu_m_ready),
    .m_prob           (asu_m_prob),
    .m_last           (asu_m_last),
    .busy             (asu_busy),
    .done             (dn_asu),
    .row_error        (asu_row_error),
    .row_error_sticky (asu_row_error_sticky)
  );

  logic [4:0]  csr_threshold;
  logic [6:0]  csr_imp_rd_addr;
  logic [15:0] tip_imp_rd_data;
  kvq_tier_e   tip_imp_rd_tier;
  logic        tip_frame_err, tip_frame_err_sticky, tip_busy;
  logic        tip_frame_err_clear;   // F-3: pulsed by ERR_STICKY[14] W1C

  tip_top #(
    .SCORE_WIDTH (32),
    .BLOCK_M     (TIP_BLOCK_M),
    .BLOCK_N     (TIP_BLOCK_N),
    .BLOCKS      (128),
    .ACC_W       (16),
    .T_MAX       (31)
  ) u_tip (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (fk_b_valid),
    .s_ready          (fk_b_ready),
    .s_data           (fk_b_data),
    .s_last           (fk_b_last),
    .s_blk            (rt_tip_blk),
    .d_valid          (td_valid),
    .d_ready          (td_ready),
    .d_fp16           (td_fp16),
    .d_tier           (td_tier),
    .d_blk            (td_blk),
    .threshold        (csr_threshold),
    .imp_thresh_hi    (rt_imp_hi),
    .imp_thresh_lo    (rt_imp_lo),
    .imp_clear        (rt_imp_clear),
    .imp_rd_addr      (csr_imp_rd_addr),
    .imp_rd_data      (tip_imp_rd_data),
    .imp_rd_tier      (tip_imp_rd_tier),
    .frame_err        (tip_frame_err),
    .frame_err_sticky (tip_frame_err_sticky),
    .frame_err_clear  (tip_frame_err_clear), // F-3: ERR_STICKY[14] W1C
    .busy             (tip_busy)
  );

  // ── stage buffers (act replay / wgt transposition, glue) ─────────────────
  logic        as_ld_valid, as_ld_ready;
  lane8_beat_t as_ld_beat;
  logic        as_job_error, as_job_error_sticky, as_busy;
  logic        ws_ld_valid, ws_ld_ready;
  logic        ws_em_valid, ws_em_ready;
  lane8_beat_t ws_em_beat;
  logic        ws_job_error, ws_job_error_sticky, ws_busy;

  // GAP D: the stage buffers set the GEMM contraction row — MODEL-WIDE
  // family, elaborated at CFG_DM. Per-head attention rows still stage here
  // with job_nb = CFG_D/8 < BPR (nb is per-job; D is only the row bound).
  apex_stage_buf #(.D(CFG_DM), .R_MAX(STAGE_R_MAX)) u_astage (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (w_aj_valid),
    .job_ready        (w_aj_ready),
    .job_op           (w_aj_op),
    .job_bank         (w_aj_bank),
    .job_pat          (w_aj_pat),
    .job_rows         (w_aj_rows),
    .job_nb           (w_aj_nb),
    .job_sel          (w_aj_sel),
    .job_error        (as_job_error),
    .job_error_sticky (as_job_error_sticky),
    .busy             (as_busy),
    .done             (dn_astage),
    .ld_valid         (as_ld_valid),
    .ld_ready         (as_ld_ready),
    .ld_beat          (as_ld_beat),
    .em_valid         (mxe_act_valid),
    .em_ready         (mxe_act_ready),
    .em_beat          (mxe_act_beat)
  );

  apex_stage_buf #(.D(CFG_DM), .R_MAX(STAGE_R_MAX)) u_wstage (   // GAP D
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (w_wj_valid),
    .job_ready        (w_wj_ready),
    .job_op           (w_wj_op),
    .job_bank         (w_wj_bank),
    .job_pat          (w_wj_pat),
    .job_rows         (w_wj_rows),
    .job_nb           (w_wj_nb),
    .job_sel          (w_wj_sel),
    .job_error        (ws_job_error),
    .job_error_sticky (ws_job_error_sticky),
    .busy             (ws_busy),
    .done             (dn_wstage),
    .ld_valid         (ws_ld_valid),
    .ld_ready         (ws_ld_ready),
    .ld_beat          (fq_out_beat),
    .em_valid         (ws_em_valid),
    .em_ready         (ws_em_ready),
    .em_beat          (ws_em_beat)
  );

  // ── E-2a: the norm-feed unpacker — feeder lane8 codes -> serial INT8
  // into the norm's x mux (u_rms above). Elaborates always; with l_nsrc=0
  // its input valid is forced low and every path below reduces to the
  // pre-E2 expression verbatim — byte-identical.
  apex_lane8_unpack u_nfeed (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (fq_out_valid && l_nsrc_q),
    .s_ready (nup_s_ready),
    .s_beat  (fq_out_beat),
    .m_valid (nup_m_valid),
    .m_ready (nup_m_ready),
    .m_x     (nup_m_x),
    .m_last  (nup_m_last),
    .busy    (nup_busy)
  );

  // feeder-codes demux + act-stage load-source mux (E-2a: l_nsrc steers the
  // codes to the norm feed instead of either stage buffer)
  assign as_ld_valid = w_rt_act_src ? sq_q_valid
                                  : (fq_out_valid && !w_rt_feeder_dst
                                     && !l_nsrc_q);
  assign as_ld_beat  = w_rt_act_src ? sq_q_beat : fq_out_beat;
  assign ws_ld_valid = fq_out_valid && w_rt_feeder_dst && !l_nsrc_q;
  assign fq_out_ready = l_nsrc_q ? nup_s_ready
                      : w_rt_feeder_dst ? ws_ld_ready
                                      : (w_rt_act_src ? 1'b0 : as_ld_ready);
  assign sq_q_ready   = w_rt_act_src ? as_ld_ready : 1'b0;

  // ═══════════════════════════════════════════════════════════════════════
  // W4 INGEST LANE (D-031 combine; docs/design/W4_DATAPATH.md). CSR window
  // 0x9C-0xAC on the ERR_STICKY/WALK seam idiom: csr_regs acks the range as
  // reserved, this file owns the data and overrides the read. Elaboration-
  // gated on W4_LANE; the OFF branch ties every net so the weight mux below
  // constant-folds to its exact legacy form.
  //   W4_CTRL (0x9C)  [0] lane_en (quasi-static route level — change only
  //                       while W4_STAT.busy=0, the rt_* rule)
  //                   [1] mode_pass (1 = INT8 passthrough framing debug)
  //                   [8] GO (W1 pulse: kick the job in W4_JOB)
  //                   [9] W1C err_sticky  [10] W1C done_seen
  //   W4_JOB  (0xA0)  {s8_f16[31:16], k[15:4], n[3:0]} — beats/ngtot derive
  //   W4_STAT (0xA4)  RO {present=1[31], phase[7:6], done_seen[2],
  //                       err_sticky[1], busy[0]}
  //   W4_CNTI (0xA8)  RO {gs_beats[31:16], pw_beats[15:0]}   (cumulative)
  //   W4_CNTO (0xAC)  RO int8 beats emitted toward the MXE   (cumulative)
  // ═══════════════════════════════════════════════════════════════════════
  localparam logic [7:0] W4_CTRL_ADDR = 8'h9C;
  localparam logic [7:0] W4_JOB_ADDR  = 8'hA0;
  localparam logic [7:0] W4_STAT_ADDR = 8'hA4;
  localparam logic [7:0] W4_CNTI_ADDR = 8'hA8;
  localparam logic [7:0] W4_CNTO_ADDR = 8'hAC;

  // parameter legality in EVERY build (also the W4_G use when !W4_LANE):
  // the geometry must match the feeder's own contract before it elaborates
  generate if (!(W4_G == 16 || W4_G == 32)) begin : g_chk_w4g
    $error("apex_top: W4_G must be 16 or 32 (D-031 freeze: ship 32)");
  end endgenerate

  logic        w4_lane_act;                 // route level; constant 0 if !W4_LANE
  logic        w4i_in_valid, w4i_in_ready;
  logic        w4o_valid, w4o_ready;
  lane8_beat_t w4o_beat;
  logic        w4_rd_q;
  logic [31:0] w4_rdata_q;

  generate if (W4_LANE) begin : g_w4

  logic        w4_lane_en_q, w4_pass_q, w4_go_q, w4_err_clr_q;
  logic [11:0] w4_k_q;
  logic [3:0]  w4_n_q;
  logic [15:0] w4_s8_q;
  logic        w4_busy, w4_done, w4_err, w4_err_sticky, w4_done_seen_q;
  logic [1:0]  w4_phase;
  logic [15:0] w4_cnt_pw, w4_cnt_gs;
  logic [31:0] w4_cnt_out;

  wire w4ctrl_wr = csr_write && (csr_addr == W4_CTRL_ADDR);
  wire w4job_wr  = csr_write && (csr_addr == W4_JOB_ADDR);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w4_lane_en_q   <= 1'b0;
      w4_pass_q      <= 1'b0;
      w4_go_q        <= 1'b0;
      w4_err_clr_q   <= 1'b0;
      w4_k_q         <= '0;
      w4_n_q         <= '0;
      w4_s8_q        <= '0;
      w4_done_seen_q <= 1'b0;
      w4_rd_q        <= 1'b0;
      w4_rdata_q     <= '0;
    end else begin
      w4_go_q      <= 1'b0;                 // self-clearing kick
      w4_err_clr_q <= 1'b0;
      if (w4ctrl_wr) begin
        w4_lane_en_q <= csr_wdata[0];
        w4_pass_q    <= csr_wdata[1];
        w4_go_q      <= csr_wdata[8];
        w4_err_clr_q <= csr_wdata[9];
        if (csr_wdata[10]) w4_done_seen_q <= 1'b0;
      end
      if (w4job_wr) begin
        w4_s8_q <= csr_wdata[31:16];
        w4_k_q  <= csr_wdata[15:4];
        w4_n_q  <= csr_wdata[3:0];
      end
      // a done in the same cycle as the W1C WINS (never lose a completion)
      if (w4_done) w4_done_seen_q <= 1'b1;
      // read pipeline mirror of csr_regs (1-cycle; read-before-write)
      w4_rd_q    <= csr_read && ((csr_addr == W4_CTRL_ADDR)
                              || (csr_addr == W4_JOB_ADDR)
                              || (csr_addr == W4_STAT_ADDR)
                              || (csr_addr == W4_CNTI_ADDR)
                              || (csr_addr == W4_CNTO_ADDR));
      w4_rdata_q <= (csr_addr == W4_CTRL_ADDR)
                  ? {30'b0, w4_pass_q, w4_lane_en_q}
                  : (csr_addr == W4_JOB_ADDR)
                  ? {w4_s8_q, w4_k_q, w4_n_q}
                  : (csr_addr == W4_STAT_ADDR)
                  ? {1'b1, 23'b0, w4_phase, 3'b0,
                     w4_done_seen_q, w4_err_sticky, w4_busy}
                  : (csr_addr == W4_CNTI_ADDR)
                  ? {w4_cnt_gs, w4_cnt_pw}
                  : w4_cnt_out;
    end
  end

  apex_w4_ingest #(
    .G         (W4_G),
    .BEATS_MAX (2048),
    .K_MAX_P   (K_MAX)
  ) u_w4i (
    .clk        (clk),
    .rst_n      (rst_n),
    .mode_pass  (w4_pass_q),
    .go         (w4_go_q),
    .job_k      (w4_k_q),
    .job_n      (w4_n_q),
    .job_s8     (w4_s8_q),
    .in_valid   (w4i_in_valid),
    .in_ready   (w4i_in_ready),
    .in_beat    (xw_beat),
    .out_valid  (w4o_valid),
    .out_ready  (w4o_ready),
    .out_beat   (w4o_beat),
    .busy       (w4_busy),
    .done       (w4_done),
    .err        (w4_err),
    .err_sticky (w4_err_sticky),
    .err_clr    (w4_err_clr_q),
    .phase_r    (w4_phase),
    .cnt_pw     (w4_cnt_pw),
    .cnt_gs     (w4_cnt_gs),
    .cnt_out    (w4_cnt_out)
  );

  assign w4_lane_act = w4_lane_en_q;

  // the err PULSE is glue-internal detail; the sticky is the CSR surface
  logic unused_w4_ok;
  assign unused_w4_ok = &{1'b0, w4_err};

  end else begin : g_w4_off

  assign w4_lane_act  = 1'b0;
  assign w4o_valid    = 1'b0;
  assign w4o_beat     = '0;
  assign w4i_in_ready = 1'b0;
  assign w4_rd_q      = 1'b0;
  assign w4_rdata_q   = '0;

  logic unused_w4off_ok;
  assign unused_w4off_ok = &{1'b0, w4i_in_valid, w4o_ready};

  end endgenerate

  // MXE weight-source mux (external host weights vs staged on-tile weights)
  // E-7: the gamma window (gam_win_q) intercepts the xw stream WHOLE — a
  // fetched G1/G2 tensor's beats go to the gamma unpacker (-> the norm's g
  // port) and never present to the MXE. Exclusive by the window contract:
  // no MXE traffic exists inside a gamma window (the walker's S2_NORM wait
  // sits between the gamma fetch and any weight fetch), and gam_win_q is 0
  // on every legacy walk, reducing all four nets to their exact old forms.
  // W4: the lane level splices apex_w4_ingest into the xw leg the same way
  // — at w4_lane_act=0 (or a !W4_LANE build, where it is constant 0) every
  // net below reduces to its exact legacy form.
  assign mxe_wgt_valid = w_rt_wgt_src ? ws_em_valid
                       : w4_lane_act  ? w4o_valid
                                      : (xw_valid && !gam_win_q);
  assign mxe_wgt_beat  = w_rt_wgt_src ? ws_em_beat
                       : w4_lane_act  ? w4o_beat
                                      : xw_beat;
  assign xw_ready      = gam_win_q    ? gu_s_ready
                       : w4_lane_act  ? w4i_in_ready
                                      : (!w_rt_wgt_src && mxe_wgt_ready);
  assign ws_em_ready   = w_rt_wgt_src  && mxe_wgt_ready;
  assign gu_s_valid    = gam_win_q && xw_valid;
  assign w4i_in_valid  = w4_lane_act && xw_valid && !gam_win_q;
  assign w4o_ready     = w4_lane_act && !w_rt_wgt_src && mxe_wgt_ready;

  // MXE result demux (out / serializer / score dequant)
  assign ro_valid     = mxe_res_valid && (w_rt_res_dst == 2'd0);
  assign ro_beat      = mxe_res_beat;
  assign ls_in_valid  = mxe_res_valid && (w_rt_res_dst == 2'd1);
  assign sd_acc_valid = mxe_res_valid && (w_rt_res_dst == 2'd2);
  always_comb begin
    unique case (w_rt_res_dst)
      2'd0:    mxe_res_ready = ro_ready;
      2'd1:    mxe_res_ready = ls_in_ready;
      2'd2:    mxe_res_ready = sd_acc_ready;
      default: mxe_res_ready = 1'b0;
    endcase
  end

  // scale_quant value-source mux (serializer vs ASU probabilities, S-4);
  // the serializer OUTPUT consumer is IB-LAYER-selectable (l_ser_dst:
  // 0 = this legacy path, 1 = layer-deq, 2 = swiglu — steered to the ACTIVE
  // phase's skid by swg_up_q; the old `swg_gr | swg_ur` OR-of-readies was
  // the B-SWG-PHASE defect, see the R2 note at the swiglu instantiation)
  assign sq_v_valid   = w_rt_squant_src ? asu_m_valid
                                        : (ls_out_valid && (l_ser_dst_q == 2'd0));
  assign sq_v_data    = w_rt_squant_src ? $signed({16'b0, asu_m_prob})
                                      : ls_out_data;
  assign sq_v_last    = w_rt_squant_src ? asu_m_last : ls_out_last;
  always_comb begin
    unique case (l_ser_dst_q)
      2'd1:    ls_out_ready = ldq_ir_port;
      2'd2:    ls_out_ready = swg_sr;    // ACTIVE phase's ready only (R2)
      default: ls_out_ready = !w_rt_squant_src && sq_v_ready;
    endcase
  end
  assign asu_m_ready  = w_rt_squant_src  && sq_v_ready;

  // ═══════════════════════════════════════════════════════════════════════════
  // IC-BIAS REGION (I-C gap B) — docs/design/IB_LAYER.md §3d. NEW PARALLEL
  // REGION, add-only: it touches neither the B1 WALK region nor the IB-LAYER
  // S4 glue datapath, and its ONLY taps into the LAYER region are the four
  // pb_* error wires, pb_busy_r and l_bias_en_r (all tied 0 below when
  // PROJ_BIAS_EN=0, so every existing build is byte-identical).
  //
  // WHAT IT DOES. Qwen2/2.5 q/k/v projections carry biases; the golden
  // arbiter adds them in REAL units to acc*composite and BEFORE the single
  // fp16 bus narrowing (transformer.py:509-514 then :521-522 / :541). That
  // insertion point lives INSIDE apex_scale_quant's exact product, which is a
  // VERIFIED block with a published proof (P1-P5) — so instead of editing it,
  // apex_proj_bias is a SIBLING on the same seam: same job/sideband/value
  // contract, same exact 25x11 product, bias summed in on the common exact
  // grid, ONE RNE through the verified f16_arith_pkg::f16_pack_real. With a
  // +0 bias vector it is bit-identical to apex_scale_quant MODE_F16 (the unit
  // suite co-simulates the two and requires that identity).
  //
  // SELECTION. Per S-2 job: a MODE_F16 job pushed while LAYER_CTRL[18]
  // l_bias_en = 1 runs on apex_proj_bias; MODE_QUANT (Q7 / S-4 P-requant)
  // always runs on apex_scale_quant, so the attention path is untouched while
  // a biased layer is in flight. The selection is LATCHED at job accept and
  // steers the sideband/value streams for that job's duration (levels are
  // quasi-static, the rt_* rule, but the mode arrives on the job port only).
  //
  // BIAS VECTOR. LAYER_PTR bank 3 + LAYER_DATA[15:0], one fp16 per write,
  // auto-increment — the same load port the phase-K/phase-q/residual banks
  // use, so the walker reaches it exactly the way it will reach those. There
  // is NO walker level port for l_bias_en (D-029 surface, not this lane's):
  // in walk mode the level HOLDS its host-written value.
  // ═══════════════════════════════════════════════════════════════════════════

  generate if (!PROJ_BIAS_EN) begin : g_pbias_off

  // no bias hardware: every seam net is the S4 straight-through connection
  assign sq_job_v     = w_qj_valid;
  assign w_qj_ready   = sq_job_r;
  assign sq_cs_v      = w_qs_valid;
  assign w_qs_ready   = sq_cs_r;
  assign sq_v_v       = sq_v_valid;
  assign sq_v_ready   = sq_v_r;
  assign sqf_valid    = sqf_sq_valid;
  assign sqf_data     = sqf_sq_data;
  assign sqf_last     = sqf_sq_last;
  assign sqf_sq_ready = sqf_ready;
  assign dn_squant    = sq_done;
  assign pb_werr      = 1'b0;
  assign pb_cerr      = 1'b0;
  assign pb_jerr      = 1'b0;
  assign pb_ferr      = 1'b0;
  assign pb_busy_r    = 1'b0;
  assign l_bias_en_r  = 1'b0;   // LAYER_CTRL[18] stays reserved-0

  end else begin : g_pbias

  // LAYER_CTRL[18] l_bias_en — host CSR path only (the §3b level shape);
  // walk mode ignores CSR writes and the level holds, exactly like the GQA
  // l_kv_map register above.
  logic l_bias_en_q;
  always_ff @(posedge clk) begin
    if (!rst_n)                     l_bias_en_q <= 1'b0;
    else if (!walk_en_q && lw_ctrl) l_bias_en_q <= csr_wdata[18];
  end
  assign l_bias_en_r = l_bias_en_q;

  wire pb_take = l_bias_en_q && !w_qj_mode;   // MODE_F16 jobs only

  logic pb_sel_q;                             // latched for the job's streams
  always_ff @(posedge clk) begin
    if (!rst_n)                        pb_sel_q <= 1'b0;
    else if (w_qj_valid && w_qj_ready) pb_sel_q <= pb_take;
  end

  logic pb_job_v, pb_job_r, pb_cs_v, pb_cs_r, pb_v_v, pb_v_r, pb_done;
  logic pb_f_valid, pb_f_ready, pb_f_last, pb_busy;
  logic [15:0] pb_f_data;
  logic pb_jerr_i, pb_jstk, pb_rerr, pb_rstk, pb_serr, pb_sstk;
  logic pb_ferr_i, pb_fstk, pb_werr_i, pb_wstk;

  // job fork (combinational on the job's own mode bit)
  assign pb_job_v   = w_qj_valid &&  pb_take;
  assign sq_job_v   = w_qj_valid && !pb_take;
  assign w_qj_ready = pb_take ? pb_job_r : sq_job_r;

  // sideband + value forks (steered by the LATCHED per-job selection)
  assign pb_cs_v    = w_qs_valid &&  pb_sel_q;
  assign sq_cs_v    = w_qs_valid && !pb_sel_q;
  assign w_qs_ready = pb_sel_q ? pb_cs_r : sq_cs_r;
  assign pb_v_v     = sq_v_valid &&  pb_sel_q;
  assign sq_v_v     = sq_v_valid && !pb_sel_q;
  assign sq_v_ready = pb_sel_q ? pb_v_r : sq_v_r;

  // f_* seam mux — the rope mux / KVQ store / dbg_f16 tap see ONE producer
  assign sqf_valid    = pb_sel_q ? pb_f_valid : sqf_sq_valid;
  assign sqf_data     = pb_sel_q ? pb_f_data  : sqf_sq_data;
  assign sqf_last     = pb_sel_q ? pb_f_last  : sqf_sq_last;
  assign pb_f_ready   =  pb_sel_q && sqf_ready;
  assign sqf_sq_ready = !pb_sel_q && sqf_ready;

  assign dn_squant   = sq_done | pb_done;     // one S-2 "job done" event
  assign pb_busy_r   = pb_busy;
  assign pb_jerr     = pb_jerr_i;
  assign pb_ferr     = pb_ferr_i;
  assign pb_werr     = pb_werr_i;
  assign pb_cerr     = pb_rerr | pb_serr;     // C1/C2 contract monitors

  wire pb_lw_en = lw_data && (l_ptr_q[29:28] == 2'd3);

  apex_proj_bias #(.D(SQ_COLS_MAX), .BN_MAX(LAYER_DM_MAX)) u_pbias (
    .clk (clk), .rst_n (rst_n),
    .job_valid (pb_job_v), .job_ready (pb_job_r), .job_cols (w_qj_cols),
    .job_error (pb_jerr_i), .job_error_sticky (pb_jstk),
    .busy (pb_busy), .done (pb_done),
    .lw_en (pb_lw_en), .lw_addr (l_ptr_q[L_DMAW-1:0]),
    .lw_data (csr_wdata[15:0]),
    .cs_valid (pb_cs_v), .cs_ready (pb_cs_r), .cs_data (w_qs_data),
    .v_valid (pb_v_v), .v_ready (pb_v_r), .v_data (sq_v_data),
    .v_last (sq_v_last),
    .f_valid (pb_f_valid), .f_ready (pb_f_ready), .f_data (pb_f_data),
    .f_last (pb_f_last),
    .range_error (pb_rerr), .range_error_sticky (pb_rstk),
    .scale_error (pb_serr), .scale_error_sticky (pb_sstk),
    .frame_error (pb_ferr_i), .frame_error_sticky (pb_fstk),
    .window_error (pb_werr_i), .window_error_sticky (pb_wstk)
  );

  // the block's own stickies are aggregated into LAYER_STATUS[8] by the
  // pulses above; the per-monitor stickies stay for waveform debug
  logic unused_pbias_ok;
  assign unused_pbias_ok = &{1'b0, pb_jstk, pb_rstk, pb_sstk, pb_fstk,
                             pb_wstk};

  end endgenerate

  // ── CSR ───────────────────────────────────────────────────────────────────
  logic [7:0] blk_busy;
  logic       any_err_pulse;
  logic       csr_desc_error_sticky;

  assign blk_busy[0] = seq_busy;
  assign blk_busy[1] = mxe_busy;
  // E-7: gu_busy (buffered gamma elements) joins the stream-glue term so
  // tile_idle covers the gamma window's in-flight tail — 0 outside a
  // window, so every legacy busy word is identical.
  assign blk_busy[2] = gu_busy
                     || fq_busy || sq_busy || ls_busy || as_busy || ws_busy
                     || fk_busy || pb_busy_r    // pb_busy_r == 0 when off
                     || res_ebusy || nup_busy;  // E-1/E-2 feed path (0 when
                                                // the feature is unused)
  assign blk_busy[3] = sd_busy;
  assign blk_busy[4] = asu_busy;
  assign blk_busy[5] = rms_busy;
  assign blk_busy[6] = tip_busy;
  assign blk_busy[7] = kv_m_pending;     // read-side only (header gap note;
                                         // any engine of the bank)

  assign any_err_pulse = seq_desc_error
                       | fq_job_error
                       | sq_job_error | sq_range_error | sq_scale_error
                       | sq_frame_error
                       | sd_job_error | sd_range_error | sd_frame_error
                       | asu_row_error
                       | rms_len_error
                       | ls_job_error
                       | as_job_error | ws_job_error
                       | tip_frame_err;

  logic [31:0] csr_rdata_int;    // csr_regs read data pre-ERR_STICKY override

  csr_regs #(
    .N_BLOCKS (8),
    .CFG_D    (CFG_D),
    .CFG_G    (KVQ_G),
    .IMP_AW   (7),
    .PERF_W   (32),
    // INFO_TIER TRUTH (F-2 closure): all three engines elaborate, but CQ-4+
    // is only genuinely available when a real outlier mask is configured.
    .TIERS    ({KVQ_OUTLIER_K > 0, 2'b11})
  ) u_csr (
    .clk               (clk),
    .rst_n             (rst_n),
    .addr              (csr_addr),
    .wdata             (csr_wdata),
    .write             (csr_write),
    .read              (csr_read),
    .rdata             (csr_rdata_int),
    .ready             (csr_ready),
    .enable            (csr_enable),
    .soft_reset        (csr_soft_reset),
    .block_busy_i      (blk_busy),
    .desc_error_i      (any_err_pulse),
    .desc_error_sticky (csr_desc_error_sticky),
    .tier_sel          (csr_tier_sel),
    .tip_override      (csr_tip_override),
    .threshold         (csr_threshold),
    .flush             (csr_flush),
    .imp_rd_addr       (csr_imp_rd_addr),
    .imp_rd_data_i     (tip_imp_rd_data),
    .imp_rd_tier_i     (tip_imp_rd_tier)
  );

  // ── sticky error bundle + ERR_STICKY window (F-3 CLOSED) ──────────────────
  //   [0] MXE desc_error  [1] feeder job   [2] squant job  [3] squant range
  //   [4] squant scale    [5] squant frame [6] scored job  [7] scored range
  //   [8] scored frame    [9] ASU row      [10] RMS len    [11] ser job
  //   [12] act-stage job  [13] wgt-stage job  [14] TIP frame  [15] reserved 0
  //
  // The bundle is LATCHED HERE from the blocks' 1-cycle error pulses and is
  // W1C-clearable through the tile CSR window at 0x58 (ERR_STICKY — glue-
  // owned; csr_regs treats 0x58 as reserved: write acked/no side effects,
  // read overridden below). Same-cycle set WINS over W1C. Bit 14 also ORs
  // TIP's own frame_err_sticky and its W1C pulses tip frame_err_clear, so
  // the one block-internal sticky WITH a clear pin stays coherent; every
  // other block-internal sticky is rst_n-only by construction (no clear
  // pins) and is intentionally no longer exported — the pulses carry the
  // same information into this clearable bank.
  localparam logic [7:0] ERR_STICKY_ADDR = 8'h58;

  logic [15:0] stk_q, stk_set;
  logic        err_stk_wr, err_stk_rd_q;
  logic [15:0] err_stk_rdata_q;

  assign err_stk_wr = csr_write && (csr_addr == ERR_STICKY_ADDR);
  assign tip_frame_err_clear = err_stk_wr && csr_wdata[14];

  assign stk_set = {1'b0,
                    tip_frame_err,
                    ws_job_error, as_job_error,
                    ls_job_error,
                    rms_len_error,
                    asu_row_error,
                    sd_frame_error, sd_range_error,
                    sd_job_error,
                    sq_frame_error, sq_scale_error,
                    sq_range_error, sq_job_error,
                    fq_job_error,
                    mxe_desc_error};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      stk_q           <= '0;
      err_stk_rd_q    <= 1'b0;
      err_stk_rdata_q <= '0;
    end else begin
      // W1C, set wins (never lose an error)
      stk_q <= (stk_q & ~(err_stk_wr ? csr_wdata[15:0] : 16'h0)) | stk_set;
      // read pipeline mirror of csr_regs (1-cycle response; a same-cycle
      // read+write returns the PRE-write value — read-before-write)
      err_stk_rd_q    <= csr_read && (csr_addr == ERR_STICKY_ADDR);
      err_stk_rdata_q <= err_sticky;
    end
  end

  assign err_sticky = stk_q | {1'b0, tip_frame_err_sticky, 14'b0};

  // ── D-027 (S12 stage 4): LIVE INFO_TIER — fenced, additive glue ───────────
  // csr_regs' TIERS parameter stays the STRUCTURAL build truth; bit 2
  // ("CQ-4+ available") additionally requires engine 2's live mask to be
  // valid: (KVQ_OUTLIER_K > 0) && mask_valid (D-027 §5, "INFO_TIER never
  // lies"). Read override in the ERR_STICKY/WALK pattern (1-cycle read
  // pipeline mirror of csr_regs). For MASK_FILE and OUTLIER_K=0 builds the
  // value equals the TIERS parameter in every reachable state — existing
  // configs read byte-identically; only the maskless OUTLIER_K>0 build
  // (b128 + CSR mask) ever sees bit 2 change at a commit.
  localparam logic [7:0] INFO_TIER_ADDR = 8'h14;
  logic        tier_rd_q;
  logic [31:0] tier_rdata_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      tier_rd_q    <= 1'b0;
      tier_rdata_q <= '0;
    end else begin
      tier_rd_q    <= csr_read && (csr_addr == INFO_TIER_ADDR);
      // S4b: a GQA build (KVQ_GQA_NENG>1) is CQ-8-only — INFO_TIER never
      // lies (D-027): the CQ-4 and CQ-4+ bits drop. OFF builds fold to the
      // pre-S4b constant exactly (bit1 == 1'b1, bit2 term unchanged).
      tier_rdata_q <= {29'b0,
                       (KVQ_OUTLIER_K > 0) && kv_mask_valid_cq4p,
                       (KVQ_GQA_NENG == 1), 1'b1};
    end
  end

  assign csr_rdata  = err_stk_rd_q ? {16'h0, err_stk_rdata_q}
                    : walk_rd_q    ? walk_rdata_q
                    : layer_rd_q   ? layer_rdata_q
                    : tier_rd_q    ? tier_rdata_q
                    : rms_erd_q    ? rms_erdata_q          // R4 (0x90/0x94)
                    : w4_rd_q      ? w4_rdata_q            // W4 (0x9C-0xAC)
                                   : csr_rdata_int;

  // ── passive debug taps (accepted beats only) ──────────────────────────────
  // GAP A: the f16 tap follows the SEAM, not the KVQ port, so it still sees
  // the post-RoPE beat when the seam is routed to the q sink. At q_sink=0
  // seam_{valid,ready,data,last} ARE kv_s_t{valid,ready,data,last} by
  // construction above — byte-identical; at q_sink=1 it carries the rotated
  // q row, which is what the golden-gated q-path check reads (the L3
  // TAPF16 idiom, same as the W-G3 K-rope gate).
  assign dbg_f16_v    = seam_valid && seam_ready;
  assign dbg_f16_data = seam_data;
  assign dbg_f16_last = seam_last;
  assign dbg_sc_v     = sd_sc_valid && sd_sc_ready;
  assign dbg_sc_data  = unsigned'(sd_sc_data);
  assign dbg_sc_last  = sd_sc_last;
  assign dbg_pr_v     = asu_m_valid && asu_m_ready;
  assign dbg_pr_data  = asu_m_prob;
  assign dbg_pr_last  = asu_m_last;

  // ── intentionally unconsumed (documented boundary items) ─────────────────
  // csr_desc_error_sticky: host reads it via STATUS bit 1 (CSR-internal).
  // seq queue status / rms_dbg_norm: observability spares.
  // *_error_sticky (block-internal): superseded by the clearable tile bank
  //   above (F-3) — latched here from the same pulses; TIP's is still used.
  logic unused_ok;
  assign unused_ok = &{1'b0,
                       csr_desc_error_sticky, seq_q_empty, seq_q_full,
                       seq_q_count, seq_aborting, rms_dbg_norm,
                       mxe_desc_error_sticky, fq_job_error_sticky,
                       sq_job_error_sticky, sq_range_error_sticky,
                       sq_scale_error_sticky, sq_frame_error_sticky,
                       sd_job_error_sticky, sd_range_error_sticky,
                       sd_frame_error_sticky, asu_row_error_sticky,
                       rms_len_error_sticky, ls_job_error_sticky,
                       as_job_error_sticky, ws_job_error_sticky};

endmodule
