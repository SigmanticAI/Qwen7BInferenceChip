// seq_layer_walker2.sv — IB-WALK stage 5: fmt=1 FULL-LAYER step sequencer.
//
// Implements: docs/design/IB_WALK.md §2.2-§2.3 (fmt=1 descriptor + step
//             templates, D-029), §2.6 (fetch requests at IB-FUEL's frozen
//             fuel_req layout + the n-major/k-minor job decomposition with
//             accumulate=(k0>0), C-KSPLIT's on-tile realization),
//             IB_LAYER.md §3b FROZEN S4 table (LAYER drive nets: levels,
//             LAYER_JOB pushes, JOBC alternating composites, norm lj_*,
//             KVQ WRITE_ADDR store sequencing), LEVEL_C_INTEGRATION.md §9.1
//             R3 as amended acbd7df (engine = KV-head index by the GOLDEN
//             mapping h // (H/H_kv)), §A-1 refusal discipline, D-006 via
//             the wrapped engine.
// Spec:       verif/seq_walker/gen_layer_trace.py (.sub.ops) — golden-gated
//             against decoder_layer_fx; the walker must reproduce its
//             emission stream bit-exact and in-order.
//
// ── WHY A SIBLING THAT INSTANTIATES seq_layer_walker, NOT AN EXTENSION ──────
// The per-head attention walk IS a D-028 v1 walk. This module wraps the
// UNCHANGED, verified v1 engine: per head it synthesizes the 3-word v1
// descriptor {D=CFG_D, T, tier=CQ8, RQ[h], mask={pv,score}} from fmt=1
// state and pulses the engine's go — so the score/pv ROM, its D-006/§5
// behavior, and its KVQ poll-before-RADDR invariant are inherited
// bit-identically ("a descriptor extension, not a rewrite", B1_WALKER.md §1).
//
// ── STEP SEQUENCER (stage 5) ────────────────────────────────────────────────
// A micro-step program counter walks the §2.3 template order; each pc is
// decoded combinationally to {kind, en-bit, args} and a disabled pc is
// skipped in one cycle. Unit-job pcs follow the §3b JOBC protocol: the
// composite file is (re)written IMMEDIATELY before every LAYER_JOB push
// (the JOBC index resets on push), and jobs whose cols exceed the CONSUMING
// UNIT's capacity split into <=walk2_lu_bound(unit) chunks (second erratum:
// swiglu chunks at WALK2_N_LU_SWIGLU=64, deq/resid at the field maximum),
// composites rewritten per chunk.
// CONSUMER-BEFORE-PRODUCER (combine, IB_LAYER.md §3b choreography rule 2 /
// IB_WALK.md §4 stage-5 flag (b)(i), ratified §9.1): in the chained
// deq->residual flows the RESIDUAL job is pushed BEFORE the deq JOBC+JOB —
// PC_URES1 precedes PC_UOPRJ and PC_URES2 precedes PC_UDOWN (the named
// one-pc swap; a consumer without a pending job presents ready=0 and would
// stall the chain).
// ROPE is a LEVEL arm — the phase-K table is per-config RESIDENT in the
// LAYER RAM (l_rope_bank=0 reads row l_rope_pos) — no per-step row fetch.
//
// ── E-3: THE NORM-FEED STEP (E2E_TOY_LANE.md §4, 2026-08-01) ────────────────
// E-1/E-2 closed the residual->norm seam INSIDE the tile, but only the HOST
// CSR path could arm it: this walker's `lw_fsrc_ext` was 2 BITS, so a walked
// program physically could not name feeder input source code 4 (apex_top
// zero-extended it), and no step pushed the LAYER_JOB unit-3 NORM/EGRESS
// job. Consequence: the sequencer had never driven that datapath. Closed by
//   (i)  widening lw_fsrc_ext to 3 bits end to end (pkg walk2_lctl -> this
//        module -> apex_top's walk-mode mux, which now copies all 3), and
//   (ii) PC_NFEED — one new pc, gated by the previously-reserved mask bit
//        W2_EN_NFEED, that performs the host demonstrator's exact verbs in
//        the same order (scripts/fpga/f2/elane_norm_feed.py stage [3]):
//          S2_NFL  arm l_fsrc_ext = 4 (ONLY that field — every other level
//                  HOLDS, so the step composes with the surrounding LVL
//                  steps of a full-layer walk and re-arms nothing)
//          S2_NFF  push the C-1 feeder job: d_model/FEED_DM rows
//          S2_NFU  push LAYER_JOB unit 3, cols = d_model, window base 0
//                  (the base the walk-mode mux pins for every LAYER job)
//          S2_NFW  hold until the WHOLE feed path is quiet — nf_busy, the
//                  same LAYER_STATUS[5] term the host polls
// The level is armed a full state BEFORE the push because apex_top's
// walk-mode level copy is registered: l_fsrc_ext_q sees code 4 one cycle
// after lw_fsrc_ext does, and S2_NFF absorbs that lag, so the unit-3 push in
// S2_NFU can never race apex_top's route-arm gate (err_code 9). The step is
// placed at the r1 -> NORM2 seam, immediately before the gamma-2 fetch:
// feed the row, then gamma, then PC_NORM2 waits on the norm's own done.
//
// The tile-geometry premises (d_model frames whole feeder rows, the row
// count fits the feeder's ROWS_MAX, cols fits the LAYER_JOB field) are
// FENCED at S2_CHECK, not assumed: a violation would be REFUSED downstream
// (apex_top code 9 / the feeder's job_error) AFTER the walker had committed
// to waiting on nf_busy — i.e. a silent wedge. §A-1: refuse, never degrade.
//
// ── E-4b: WALKER-DRIVABLE l_nsrc (E2E_TOY_LANE.md §4, 2026-08-04) ───────────
// E-4's measured structural finding (STEP_MATRIX.md): l_nsrc (LAYER_CTRL
// [19], the E-2a norm-input-source route) was HOST-held across a walk (the
// l_bias_en idiom), and it steers the ONE feeder output for the whole walk —
// QSTAGE needs codes -> act stage (nsrc=0), NFEED needs codes -> norm
// (nsrc=1) — so walk_e4's full chain {QSTAGE, SCORE, PV, NFEED} needed TWO
// kicks with a host CSR write between them. Closed the E-3a/E-3b way:
//   (i)  the previously-reserved mask bit W2_EN_NSRC (W2_MASK[13]) makes the
//        walker the level's OWNER for that walk. lw_nsrc carries the value;
//        lw_nsrc_own (registered, high from the S2_CHECK pass to walk end)
//        gates apex_top's walk-mode copy, so every legacy image (bit 13 = 0)
//        keeps the host-held HOLD semantics byte-identically, and a NEW
//        image on OLD RTL is refused loudly (the old resv clause).
//   (ii) the arms ride the EXISTING states: S2_NFL raises nsrc WITH the
//        code-4 flip (one settle state before the pushes, same registered-
//        copy argument as fsrc_ext); S2_QSL and every S2_LVL case re-arm it
//        0 absolutely (writes of the reset value — byte-identical traces
//        for every walk without the bit). The PC order already serves the
//        one-kick chain (PC_QSTAGE < PC_ATTN < PC_NFEED); l_nsrc was the
//        only blocker.
//   (iii) S2_CHECK fences the cross-bit rules (§A-1: refuse, never
//        degrade): NSRC without NFEED owns the mux for nothing — refused;
//        QSTAGE+NFEED without NSRC is the measured wedge shape (no single
//        host-held value serves both steps) — refused, never walked into a
//        stuck nf_busy wait.
//
// ── E-6: THE WALKED EPILOGUE (2026-08-05) ───────────────────────────────────
// E-5 proved a walked, fuel-fed projection but FENCED OUT the o8 requant
// epilogue (pc_hasrq — OPROJ/DOWN): its walks egress RAW INT32 on the RO
// lanes and the mask had to be exactly {FPROJ, QKV}. E-6 un-fences the
// epilogue for OPROJ: an {FPROJ, OPROJ, RES1, ...} walk runs the SAME
// S2_PAE/S2_PJOB/S2_PAD machinery with three deltas, all gated on the
// dispatched pc's own pc_hasrq (registered as fp_rq_q), so every E-5 QKV
// walk and every legacy walk is byte-identical:
//   (i)   S2_PSJ — the walker pushes the serializer job that frames the
//         projection's o8 result stream (n_splits beats x 8 lanes = the
//         deq job's cols, out_last on the final element) BEFORE the first
//         act EMIT. The consumer jobs were already pushed in template
//         order (PC_URES1's residual add, PC_UOPRJ's JOBC+deq), so the
//         tile's whole o8 -> deq -> residual chain is armed before any
//         result beat exists — consumer-before-producer, one level deeper.
//   (ii)  RT_FPRQ (8'h94) — RT_FPROJ with rdst 0 -> 1: the requantised o8
//         posts to the serializer instead of the RO lanes, the measured
//         host-mode o8 leg's own destination (gen_layer_ops.py:795).
//   (iii) the OPROJ act family emits from rows fp_oproj_base.. so it
//         coexists with the staged-q rows / QKV family (aj_sel note).
// The descriptor's requant pair rides RQ[H] exactly as the pj_desc always
// encoded (rq_word); nothing about the epilogue encoding changed — what
// changed is that it now EXECUTES and its product lands in the residual
// row (r1 = f16(X + o8*comp)) INSIDE the tile, where PC_NFEED can feed it
// onward. DOWN's epilogue is the same pc class but stays refused with the
// FFN mask (fp_mask_ok note: template order starves swiglu; at D=64 its
// k = d_ffn also exceeds the 31-row stage bank — the frozen-spec K_JOB
// erratum tracked in the W1 lane). Completion: S2_PJW's tile_idle wait
// covers the serializer (ls_busy in blk_busy[2]); the deq/residual jobs
// drain in the grid domain and the residual's OWN ej exclusivity
// (apex_residual.sv:137 ej_ready waits out a streaming add) orders any
// following NFEED read-after-update; the host's post-walk readback polls
// LAYER_STATUS[LDQ|RES] — the host-mode idiom.
//
// ── E-7: THE FETCHED GAMMA (2026-08-05, the convergence session) ────────────
// The last NORM fence falls. E-6 kept NORM1/NORM2 refused under FPROJ with a
// measured reason: the gamma fetch would ISSUE (fuel armed) and the bytes
// would dead-end in the fuel FIFO — no xw -> xg route existed. E-7 builds
// the route the house way: mask bit W2_EN_FGAM (W2_MASK[15], reserved-0
// before) arms a walker-owned GAMMA WINDOW level (lw_gsrc), high from the
// G1/G2 fetch dispatch to S2_GDW's tile_idle. apex_top's glue steers the xw
// stream into a 4-per-beat int16 unpacker (apex_gam_unpack — the
// make_weight_image gamma_payload layout) feeding asu_rmsnorm's g port; the
// fetch delivers EXACTLY d_model gammas and the norm consumes EXACTLY
// d_model per row (asu_rmsnorm.sv:104-106 "consumed DURING EMISSION"), so
// when dn_rms fires the fetched stream is drained by construction.
// A fetched-gamma norm EMITS mid-walk (gamma is what releases emission),
// so the walker also arms the norm's OUTPUT drain itself — S2_GDL/GDA/GDF
// perform the host drain choreography (elane_walk_norm stage [8]: levels,
// act-LOAD job at the fp_top row window, C-1 feeder job; route RT_YDRAIN)
// as walker verbs, else dn_rms never fires (done requires every y beat
// ACCEPTED post-skid) and S2_NORM would wedge. S2_GDW then holds the route
// until tile_idle (the S2_PJW tail argument) and closes the window. The
// next weight fetch cannot race the window: its record is not ISSUED until
// S2_GDW has passed. Fences (S2_CHECK, §A-1): FGAM requires FPROJ (a gamma
// fetch without fuel is the S2_FETCH park) and a named NORM step (a window
// with no consumer); ONE norm per FGAM walk (both y-drains would land at
// the same fp_top window — a read-back ambiguity); the drain envelope
// (NF_ROWS_EXACT, n_heads <= FEED_ROWS, fp_top + n_heads <= STAGE_ROWS);
// NORMx under FPROJ without FGAM keeps the E-6 poisoning refusal verbatim.
//
// ── E-7b: THE SEPARATED DOWN (same session) ─────────────────────────────────
// DOWN's pcs were enabled by W2_EN_FFN alone; W2_EN_DOWN (W2_MASK[16])
// enables exactly {PC_LVLD, PC_UDOWN, PC_WDF, PC_WDJ} without the FFN
// steps, paired with RES2 (the OPROJ<->RES1 rule). The DOWN epilogue is
// the pc_hasrq class E-6 proved on OPROJ — RQ[H+1], JC_DOWN, S2_PSJ frame
// of d_model elements, RT_FPRQ — at its own act-family window ABOVE the
// OPROJ rows (fp_base). Its k = d_ffn must fit ONE k-split
// (walk2_k_job(FEED_DM)) and the whole staged footprint one bank: at the
// 0.5B geometry d_ffn = 4864 needs 76 stage rows against the 31-row bank
// (apex_stage_buf.sv:103-104 R_MAX cap), and a per-k-chunk re-staged
// family has NO in-tile replay source (S2_PAE emits a resident family;
// nothing re-fills it mid-walk) — so 0.5B DOWN is REFUSED at S2_CHECK
// with that reason, and the epilogue class is proven at geometries that
// fit. The act family is the swiglu-product codes, HOST-staged — the
// disclosed E-6 o8->act seam class; the in-tile producer is the FFN
// interleave follow-on (fence 2), not faked here.
//
// ── REGISTERED-ACCEPT (IB_WALK.md §7 Q1 stage-5 note, binding) ──────────────
// The §3b units expose combinational job-ready that DROPS at the accepting
// edge. Every wait state here presents-and-holds valid and captures ready
// at the transitioning always_ff edge (`if (xx_ready) ...` with valid held
// combinationally from `state`); nothing re-checks a ready after its edge.
//
// ── PROJECTION ENCODING NOTE (Q1-provisional) ───────────────────────────────
// GEMM jobs are emitted OP_GEMM_OS + mode_os=1 + accumulate=(k0>0) — the
// C-KSPLIT on-tile realization available under the FROZEN mxe_desc_t. The
// epilogue requant pair rides the LAST k-split of each n-split (RQ[H] for
// o-proj, RQ[H+1] for down).
//
// ── D-029 ERRATUM (I-B): n IS BOUNDED BY THE ARRAY, NOT THE FIELD ───────────
// The n-split uses WALK2_N_MXE (= MXE_N = 8, the implemented per-descriptor
// N limit that mxe_ctrl's `legal` term enforces), NOT the 12-bit field width.
// The landed stage-5 code used one overloaded constant for BOTH this and the
// LAYER_JOB cols chunking, so every 7B projection descriptor asked the tile
// for n=4095 and was refused with desc_error. The cols chunking below is
// per-unit too (SECOND ERRATUM — the old claim here that WALK2_N_JOB "is
// genuinely a field-width bound" was wrong for swiglu, whose port is 7 of
// the 12 bits: see seq_walker_pkg's erratum note). Consequence
// of the fix: many more, much narrower projection jobs (7B: 38 -> 16,000
// descriptors per layer step). Correct and slow beats fast and refused; the
// n-major/k-minor ORDER — and therefore IB-FUEL's pre-swizzled image order —
// is unchanged, only the chunk width is.
//
// ── fmt=0 TRANSPARENT FORWARD (combine W-G3 flip; FMT_SUP = 0b0011) ─────────
// With walker2 as the tile's ONE walker instance (the §9.1 combine-agenda
// instance+mask flip), a loaded fmt=0 (D-028 v1) image must keep walking
// bit-identically. The wrapped v1 engine IS that walker, so the forward is
// pure combinational pass-through: when the image's fmt nibble reads v1 and
// the fmt=1 sequencer is idle, u_head sees walk_en/walk_go/words 0-2 exactly
// as the direct D-028 instance did (same cycle, same values — L3 walker-mode
// byte-identity is gated on it) and busy/err/err_code mirror back the same
// way. The fmt=1 sequencer ignores the go (S2_IDLE gate); fmt=1 semantics
// are unchanged, and fmt>1 images still land in walk_desc2_check's DESC
// refusal.
`ifndef SEQ_LAYER_WALKER2_SV
`define SEQ_LAYER_WALKER2_SV

module seq_layer_walker2
  import apex_pkg::*;
  import seq_walker_pkg::*;
#(
  parameter int unsigned CFG_D = 64,
  // I-C IC-QPATH, F6(ii): how many heads' q rows the tile has pre-staged in
  // act-stage bank 1. 1 (DEFAULT) reproduces the pre-I-C behaviour exactly —
  // the wrapped engine always re-emits row 0 — so every existing build is
  // bit-identical. > 1 selects row h for head h, which is what makes a
  // MULTI-HEAD fmt=1 attention walk consume the RIGHT q per head. Mirrors
  // apex_top's QSTAGE_H_MAX; the tile passes its own value down.
  parameter int unsigned Q_ROWS = 1,
  parameter int unsigned N_ENG = 4,   // physical CQ-8 engines (per-KV-head,
                                      // §9.1 R3; build-time like CFG_D)
  // E-3: the C-1 feeder's row width and per-job row bound, as INSTANTIATED
  // in this build (apex_top passes CFG_DM / FEED_ROWS_MAX). They are the
  // NFEED step's only tile-shape inputs, and they are kept as parameters —
  // not re-derived from CFG_D — precisely so the fence below measures the
  // consumer rather than the walker's own assumption. Defaults reproduce a
  // pre-E-3 instantiation (CFG_DM == CFG_D is every buildable image today;
  // E2E_TOY_LANE.md §2 names the wide-feeder split as unbuilt work).
  parameter int unsigned FEED_DM   = CFG_D,
  parameter int unsigned FEED_ROWS = 16,
  // E-5: the ACT STAGE BUFFER's rows-per-bank bound, as INSTANTIATED
  // (apex_top passes STAGE_R_MAX; u_astage is apex_stage_buf #(.D(CFG_DM),
  // .R_MAX(STAGE_R_MAX))). Same rule as FEED_DM/FEED_ROWS above: the
  // fuel-fed projection's act-row family must FIT the consumer, so the
  // walker is told the consumer's real bound instead of assuming one. The
  // default reproduces apex_top's own STAGE_R_MAX default.
  parameter int unsigned STAGE_ROWS = 16,
  localparam int unsigned STAGE_NB_W = $clog2(CFG_D / 8) + 1,
  localparam int unsigned ENG_W      = (N_ENG > 1) ? $clog2(N_ENG) : 1
) (
  input  logic                  clk,
  input  logic                  rst_n,        // synchronous, active low

  // ── control (WALK CSR window glue; fmt=1 SRAM is 64 words) ───────────────
  input  logic                  walk_en,
  input  logic                  walk_go,      // 1-cycle kick
  input  logic                  abort_req,    // CTRL.soft_reset (D-020)
  input  logic [31:0]           desc2_word [WALK2_DESC_WORDS],
  input  logic                  tile_idle,

  // ── status ───────────────────────────────────────────────────────────────
  output logic                  walk2_busy,
  output walk_step_e            walk2_step,
  output walk_phase_e           attn_phase,   // wrapped engine's phase
  output logic                  walk2_err,    // 1-cycle pulse
  output walk_err_e             walk2_err_code,

  // ── fetch requests to IB-FUEL's reader (frozen fuel_req layout) ──────────
  output logic                  wf_valid,
  input  logic                  wf_ready,
  output walk2_freq_t           wf_req,

  // ── per-head datapath interlock ──────────────────────────────────────────
  // Presented before each head's attention walk; the tile's glue ties
  // hd_ready to "this head's q row is staged and its s_q snooped"
  // (registered-accept: valid held, ready captured at the transition edge).
  output logic                  hd_valid,
  input  logic                  hd_ready,
  output logic [7:0]            hd_head,

  // ── LAYER drive nets (IB_LAYER.md §3b, single ingress per R5) ────────────
  // Levels are registered copies of the §3b l_*_q registers' walker-side
  // inputs; the glue muxes them against the host CSR path. Job/composite
  // channels are valid/ready with registered-accept on this side.
  output logic                  lw_rope_en,
  output logic                  lw_rope_bank,
  output logic [6:0]            lw_rope_pos,
  output logic [1:0]            lw_ser_dst,
  // E-3: 3 bits. [2] is LAYER_CTRL[7] (E-1's l_fsrc_ext[2]); code 4 = the
  // residual row's internal egress. Was 2 bits, which is exactly why a
  // walked program could not arm the in-tile norm feed.
  output logic [2:0]            lw_fsrc_ext,
  output logic                  lw_resid_arm,
  // E-4b: the l_nsrc level (LAYER_CTRL[19]) + its OWNERSHIP level (header
  // note). lw_nsrc_own is high from the descriptor-check pass to walk end
  // of a walk whose mask names W2_EN_NSRC; apex_top copies lw_nsrc into
  // l_nsrc_q only while it is high, so an unowned walk — every legacy
  // image — leaves the host-armed level untouched (the l_bias_en HOLD).
  output logic                  lw_nsrc,
  output logic                  lw_nsrc_own,
  // E-7: the GAMMA WINDOW level — high from an FGAM walk's G1/G2 fetch
  // dispatch to its S2_NORM exit. apex_top steers the xw stream into the
  // gamma unpacker (-> asu_rmsnorm g port) while it is high. Never raised
  // by a walk whose mask lacks W2_EN_FGAM, so every legacy image leaves
  // the xw -> MXE and xg -> norm paths byte-identical.
  output logic                  lw_gsrc,
  output logic                  lu_valid,     // LAYER_JOB push (0x7C image)
  input  logic                  lu_ready,
  output logic [1:0]            lu_unit,
  output logic [11:0]           lu_cols,
  output logic                  jc_valid,     // LAYER_JOBC write (alternating)
  input  logic                  jc_ready,
  output logic [31:0]           jc_data,
  output logic                  lj_valid,     // EXISTING norm-job port (lj_*)
  input  logic                  lj_ready,
  output logic [DIM_W-1:0]      lj_cols,
  // E-3b: SERIALIZER job push (apex_lane32_ser's job port — apex_top muxes
  // this onto u_ser and holds the HOST lj_* serializer-job port off during
  // a walk; no walked program to date pushed that port mid-walk). The
  // QSTAGE step frames the per-head q production stream with it.
  output logic                  sj_valid,
  input  logic                  sj_ready,
  output logic [7:0]            sj_beats,
  output logic [3:0]            sj_lanes,
  // E-3: LAYER_STATUS[5] (apex_top's nf_busy) — the whole residual-egress ->
  // feeder -> unpack -> norm feed path in flight. The NFEED step's ONLY
  // completion observable, and the SAME term the host demonstrator polls.
  input  logic                  nf_busy,

  // ── KVQ engine select (per-KV-head banks, §9.1 R3 amended) ───────────────
  output logic [ENG_W-1:0]      kv_eng_sel,

  // ── pass-through control fanouts (identical to seq_layer_walker) ─────────
  output logic                  ds_valid,
  input  logic                  ds_ready,
  output mxe_desc_t             ds_desc,
  output logic                  rt_feeder_src,
  output logic                  rt_feeder_dst,
  output logic                  rt_act_src,
  output logic                  rt_wgt_src,
  output logic [1:0]            rt_res_dst,
  output logic                  rt_squant_src,
  output logic                  rt_kv_user,
  output logic                  fj_valid,
  input  logic                  fj_ready,
  output logic [DIM_W-1:0]      fj_rows,
  output logic                  qj_valid,
  input  logic                  qj_ready,
  output logic                  qj_mode,
  output logic [DIM_W-1:0]      qj_cols,
  output logic                  dj_valid,
  input  logic                  dj_ready,
  output logic [DIM_W-1:0]      dj_cols,
  output logic                  aj_valid,
  input  logic                  aj_ready,
  output logic                  aj_op,
  output logic                  aj_bank,
  output logic [1:0]            aj_pat,
  output logic [4:0]            aj_rows,
  output logic [STAGE_NB_W-1:0] aj_nb,
  output logic [4:0]            aj_sel,
  output logic                  wj_valid,
  input  logic                  wj_ready,
  output logic                  wj_op,
  output logic                  wj_bank,
  output logic [1:0]            wj_pat,
  output logic [4:0]            wj_rows,
  output logic [STAGE_NB_W-1:0] wj_nb,
  output logic [4:0]            wj_sel,
  output logic                  qs_valid,
  input  logic                  qs_ready,
  output logic [31:0]           qs_data,
  output logic                  cs_valid,
  input  logic                  cs_ready,
  output logic [31:0]           cs_data,
  output logic                  comp_req_valid,
  input  logic                  comp_req_ready,
  output logic                  comp_req_is_qs,
  output logic [WALK_SC_AW-1:0] comp_req_idx,
  input  logic                  comp_res_valid,
  output logic                  comp_res_ready,
  input  logic [31:0]           comp_res_data,
  output logic                  kvm_awvalid,
  input  logic                  kvm_awready,
  output logic [7:0]            kvm_awaddr,
  output logic                  kvm_wvalid,
  input  logic                  kvm_wready,
  output logic [31:0]           kvm_wdata,
  input  logic                  kvm_bvalid,
  output logic                  kvm_bready,
  output logic                  kvm_arvalid,
  input  logic                  kvm_arready,
  output logic [7:0]            kvm_araddr,
  input  logic                  kvm_rvalid,
  output logic                  kvm_rready,
  input  logic [31:0]           kvm_rdata
);

  if (!(CFG_D == 64 || CFG_D == 128)) begin : g_chk_d
    $error("seq_layer_walker2: CFG_D must be 64 or 128 (D-021)");
  end
  if (N_ENG < 1 || N_ENG > 16) begin : g_chk_eng
    $error("seq_layer_walker2: N_ENG must be 1..16");
  end
  // E-3: the NFEED step divides d_model by FEED_DM and bounds the quotient
  // by FEED_ROWS; both must be sane at elaboration or the runtime fence
  // below is meaningless.
  if (FEED_DM == 0 || FEED_ROWS == 0) begin : g_chk_feed
    $error("seq_layer_walker2: FEED_DM and FEED_ROWS must be nonzero");
  end

  localparam logic [7:0] KV_WADDR = 8'h28;   // kvq_engine REG_WRITE_ADDR
  localparam logic [7:0] KV_STAT  = 8'h04;

  // ── the wrapped, UNCHANGED v1 engine ─────────────────────────────────────
  logic        h_en, h_go;
  logic [31:0] h_desc [WALK_DESC_WORDS];
  logic        h_busy, h_err;
  walk_phase_e h_phase;
  walk_err_e   h_err_code;
  logic        h_ds_valid;
  logic        h_ds_ready;
  mxe_desc_t   h_ds_desc;
  // u_head's KVQ master (muxed with the STOREKV master onto kvm_*)
  logic        h_aw_v, h_w_v, h_ar_v, h_b_r, h_r_r;
  logic [7:0]  h_aw_a, h_ar_a;
  logic [31:0] h_w_d;
  logic        in_store;
  // u_head's feeder-job port (E-3: muxed with the NFEED step's push)
  logic             h_fj_valid, h_fj_ready;
  logic [DIM_W-1:0] h_fj_rows;
  // E-3b: u_head's squant-job / squant-sideband / act-stage-job / route
  // nets, internalized so the QSTAGE step can own those channels in its
  // states (the h_fj mux idiom) and the fmt=1 route override can hand a
  // DETERMINISTIC word back (see rt_f1_q below).
  logic             h_qj_valid, h_qj_ready, h_qj_mode;
  logic [DIM_W-1:0] h_qj_cols;
  logic             h_aj_valid, h_aj_ready, h_aj_op, h_aj_bank;
  logic [1:0]       h_aj_pat;
  logic [4:0]       h_aj_rows, h_aj_sel;
  logic [STAGE_NB_W-1:0] h_aj_nb;
  logic             h_qs_valid, h_qs_ready;
  logic [31:0]      h_qs_data;
  logic             h_rt_fsrc, h_rt_fdst, h_rt_asrc, h_rt_wsrc, h_rt_qsrc,
                    h_rt_kvu;
  logic [1:0]       h_rt_rdst;

  // fmt=0 transparent forward (header note): combinational pass-through of
  // en/go and the 3-word view whenever the loaded image is v1 and the fmt=1
  // sequencer is idle. state is declared below; Verilator resolves the
  // forward reference inside the generate-free module scope.
  logic        v1_fwd;
  logic [31:0] h_desc_eff [WALK_DESC_WORDS];
  always_comb begin
    h_desc_eff = h_desc;
    if (v1_fwd) begin
      h_desc_eff[WALK_DW_GEOM] = desc2_word[WALK_DW_GEOM];
      h_desc_eff[WALK_DW_RQ]   = desc2_word[WALK_DW_RQ];
      h_desc_eff[WALK_DW_MASK] = desc2_word[WALK_DW_MASK];
    end
  end

  seq_layer_walker #(.CFG_D(CFG_D)) u_head (
    .clk(clk), .rst_n(rst_n),
    .walk_en(v1_fwd ? walk_en : h_en), .walk_go(v1_fwd ? walk_go : h_go),
    .abort_req(abort_req),
    .desc_word(h_desc_eff), .tile_idle(tile_idle),
    .q_row_sel(h_q_row),
    .walk_busy(h_busy), .walk_phase(h_phase),
    .walk_err(h_err), .walk_err_code(h_err_code),
    .ds_valid(h_ds_valid), .ds_ready(h_ds_ready), .ds_desc(h_ds_desc),
    // E-3b: routes go through the fmt=1 override mux below (byte-identical
    // for every walk that never enters a QSTAGE state — the tracked copy
    // equals u_head's value at every mux boundary).
    .rt_feeder_src(h_rt_fsrc), .rt_feeder_dst(h_rt_fdst),
    .rt_act_src(h_rt_asrc), .rt_wgt_src(h_rt_wsrc),
    .rt_res_dst(h_rt_rdst), .rt_squant_src(h_rt_qsrc),
    .rt_kv_user(h_rt_kvu),
    // E-3: the feeder job channel is now SHARED — the wrapped v1 engine
    // drives it during attention, the NFEED step drives it for the C-1 row
    // framing. Muxed below exactly like ds_* (they never overlap: a head
    // walk and a step-sequencer state are mutually exclusive by the FSM).
    // E-3b adds the QSTAGE step as a third, equally-exclusive owner of
    // fj/qj/qs/aj.
    .fj_valid(h_fj_valid), .fj_ready(h_fj_ready), .fj_rows(h_fj_rows),
    .qj_valid(h_qj_valid), .qj_ready(h_qj_ready), .qj_mode(h_qj_mode),
    .qj_cols(h_qj_cols),
    .dj_valid(dj_valid), .dj_ready(dj_ready), .dj_cols(dj_cols),
    .aj_valid(h_aj_valid), .aj_ready(h_aj_ready), .aj_op(h_aj_op),
    .aj_bank(h_aj_bank), .aj_pat(h_aj_pat), .aj_rows(h_aj_rows),
    .aj_nb(h_aj_nb), .aj_sel(h_aj_sel),
    .wj_valid(wj_valid), .wj_ready(wj_ready), .wj_op(wj_op),
    .wj_bank(wj_bank), .wj_pat(wj_pat), .wj_rows(wj_rows), .wj_nb(wj_nb),
    .wj_sel(wj_sel),
    .qs_valid(h_qs_valid), .qs_ready(h_qs_ready), .qs_data(h_qs_data),
    .cs_valid(cs_valid), .cs_ready(cs_ready), .cs_data(cs_data),
    .comp_req_valid(comp_req_valid), .comp_req_ready(comp_req_ready),
    .comp_req_is_qs(comp_req_is_qs), .comp_req_idx(comp_req_idx),
    .comp_res_valid(comp_res_valid), .comp_res_ready(comp_res_ready),
    .comp_res_data(comp_res_data),
    .kvm_awvalid(h_aw_v), .kvm_awready(kvm_awready && !in_store),
    .kvm_awaddr(h_aw_a),
    .kvm_wvalid(h_w_v), .kvm_wready(kvm_wready && !in_store),
    .kvm_wdata(h_w_d),
    .kvm_bvalid(kvm_bvalid && !in_store), .kvm_bready(h_b_r),
    .kvm_arvalid(h_ar_v), .kvm_arready(kvm_arready && !in_store),
    .kvm_araddr(h_ar_a),
    .kvm_rvalid(kvm_rvalid && !in_store), .kvm_rready(h_r_r),
    .kvm_rdata(kvm_rdata)
  );

  assign attn_phase = h_phase;

  // ── walk state ───────────────────────────────────────────────────────────
  // E-5 widened this 5 -> 6 bits: the 32 five-bit encodings were ALL taken
  // (S2_QSW = 31), and the fuel-fed projection needs two more states. The
  // enum is module-local — `state` is never a port, never packed into the
  // debug word, and walk2_step maps it by name (the mapping below is
  // unchanged) — so the widening is invisible outside this file.
  typedef enum logic [5:0] {
    S2_IDLE  = 6'd0,  S2_CHECK = 6'd1,  S2_STEP  = 6'd2,
    S2_FETCH = 6'd3,  S2_PJOB  = 6'd4,  S2_NORM  = 6'd5,
    S2_LVL   = 6'd6,  S2_JCA   = 6'd7,  S2_JCB   = 6'd8,  S2_LU = 6'd9,
    S2_SKPAR = 6'd10, S2_SKPR  = 6'd11, S2_SKWR  = 6'd12, S2_SKB = 6'd13,
    S2_HLOAD = 6'd14, S2_HGO   = 6'd15, S2_HWAIT = 6'd16,
    S2_DONE  = 6'd17, S2_ERR   = 6'd18,
    // E-3 NFEED sub-sequence: level arm, feeder job, unit-3 job, feed wait
    S2_NFL   = 6'd19, S2_NFF   = 6'd20, S2_NFU   = 6'd21, S2_NFW = 6'd22,
    // E-3b QSTAGE sub-sequence (per head): level arm, feeder job, MODE_F16
    // squant job, composite sideband xD, serializer job, then per beat the
    // k2-injection {act EMIT, WS descriptor}, the act-bank-1 LOAD at row h,
    // and a whole-path drain wait. The exit edge restores fsrc_ext (the
    // touch-one-field S2_NFL idiom, in reverse).
    S2_QSL   = 6'd23, S2_QSF   = 6'd24, S2_QSQ   = 6'd25, S2_QSC = 6'd26,
    S2_QSS   = 6'd27, S2_QSE   = 6'd28, S2_QSD   = 6'd29, S2_QSA = 6'd30,
    S2_QSW   = 6'd31,
    // E-5 FUEL-FED PROJECTION (W2_MASK[14]): the act-family EMIT that feeds
    // the MXE's activation port for a walked projection job (S2_PAE), the
    // per-job n-cursor advance (S2_PAD), and the post-projection drain wait
    // that holds the RT_FPROJ route until the last result beat has left the
    // tile (S2_PJW). All three are entered ONLY on an EN_FPROJ walk, so a
    // legacy projection walk is byte-identical: K_PJOB still dispatches
    // straight to S2_PJOB and S2_PJOB still returns straight to S2_STEP.
    S2_PAE   = 6'd32, S2_PJW   = 6'd33, S2_PAD = 6'd34,
    // E-6 THE WALKED EPILOGUE (2026-08-05): before an epilogue-bearing
    // (pc_hasrq) fuel-fed projection's first act EMIT, the walker pushes
    // the SERIALIZER job that frames the o8 result stream for the LAYER
    // deq unit — the host-mode o8->deq->residual leg's own framing verb
    // (gen_layer_ops.py inject_frame's ljob), issued by the SEQUENCER.
    // Only reachable on an EN_FPROJ walk of an OPROJ-class step, so every
    // legacy and every E-5 QKV walk is byte-identical.
    S2_PSJ   = 6'd35,
    // E-6 pre-epilogue DRAIN FENCE — MEASURED on the one-kick chain: the
    // wrapped engine's PV walk retires with its last o8 beats still in the
    // result pipeline, and the epilogue window's rdst 0 -> 1 flip is only
    // ~10 cycles behind it (LVL/JCA/LU/FETCH are all 1-cycle handshakes),
    // so the o8 TAIL got misrouted into the serializer and pre-consumed
    // the epilogue's frame. S2_PDW holds the window entry (presenting NO
    // valid — D-020-clean) until tile_idle, whose mxe/serializer terms
    // guarantee every attention result beat has LEFT the demux; the
    // RT_FPRQ flip rides fp_hold from S2_PSJ onward, after the fence. The
    // QKV -> OPROJ hand-off never had the hazard (S2_PJW's own tile_idle
    // sits between the windows), which is why CLAIM B was green without
    // this state.
    S2_PDW   = 6'd36,
    // E-7 THE FETCHED-GAMMA NORM (2026-08-05): a NORM step whose gamma is
    // fuel-fetched EMITS during the walk (gamma is consumed at emission),
    // so its output row must have an armed consumer or dn_rms never fires
    // — the S2_NORM wait would be a WEDGE. The walker therefore performs
    // the host drain choreography (elane_walk_norm stage [8], verbatim
    // verbs) itself before the wait:
    //   S2_GDL  arm the drain levels {fsrc_ext=0, nsrc=0 (owned walks)} +
    //           the RT_YDRAIN route override (the host's own measured
    //           route(rdst=0, asrc=0) word) — one settle state, the S2_NFL
    //           registered-copy argument
    //   S2_GDA  push the act-stage LOAD job: n_heads rows at the y-base
    //           row window (fp_top — above every staged family)
    //   S2_GDF  push the C-1 feeder job (n_heads rows): norm y -> widen ->
    //           feeder codes -> act bank 1, the whole row in-tile
    //   (then S2_NORM waits dn_rms as ever)
    //   S2_GDW  hold the route until tile_idle — dn_rms means every y beat
    //           was ACCEPTED post-skid, not that the widen/feeder/act tail
    //           has landed; reverting mid-flight would misroute the tail
    //           (the S2_PJW lesson). The gamma window closes HERE.
    S2_GDL   = 6'd37, S2_GDA = 6'd38, S2_GDF = 6'd39, S2_GDW = 6'd40,
    // FFN INTERLEAVE (fence-2/3, 2026-08-08): the PER-CHUNK swiglu-product
    // stage — the y-drain choreography aimed at the DOWN act family. After
    // each chunk's up jobs drain (S2_PJW), the walker arms the feeder to
    // the SWIGLU PRODUCT source (fsrc_ext=SWGP; the route word is
    // RT_YDRAIN — feeder->act is asrc=0/fdst=0 both here and there),
    // pushes the act LOAD at row fp_dwn_base + ffn_chunk_q (S2_CSA), the
    // C-1 feeder job (S2_CSF), waits the whole path quiet (S2_CSW),
    // restores fsrc, and loops PC_USWI or falls through to DOWN — whose
    // act family the loop has now staged chunk by chunk (fp_r_down =
    // d_ffn>>FP_SH rows: the row math anticipated exactly this).
    S2_CSL   = 6'd41, S2_CSA = 6'd42, S2_CSF = 6'd43, S2_CSW = 6'd44,
    // 2026-08-13 THE STOREKV SELECT SETTLE (fix-flight-1 verdict; capture
    // forensics + sim repro): fa603d3's D-033 sledgehammer REGISTERED the
    // tile-level KVQ engine select (apex_top kv_eng_sel_q) — one cycle
    // late by design, argued safe by the bank's quasi-static routing
    // contract. The walked-STOREKV step is the ONE choreography that
    // breaks the argument: S2_SKB advanced sk_g 0->1 and re-entered
    // S2_SKPAR asserting kvm_arvalid THE VERY NEXT CYCLE — inside the
    // stale-select window. The GQA bank routes combinationally off the
    // registered select and the engine's arready is combinational
    // (!rvalid || rready), so engine 0 ACCEPTED engine 1's STATUS read;
    // a cycle later the select settled and engine 0's registered rvalid
    // had no route home — the walker spun in S2_SKPR forever, and on
    // silicon every ATT program (24/24 layers) aborted at its FIRST
    // drain poll. The fix is one dead state: EVERY entry into S2_SKPAR
    // passes through S2_SKS, which holds in_store (kv_eng_sel = sk_g
    // routes; NO valid presented — D-020-clean, the S2_PDW idiom) for
    // exactly one cycle, so the registered select has settled before
    // kvm_arvalid can rise. Uniform on all four entries (dispatch,
    // poll-retry, K->V record, engine advance): one invariant, no
    // per-path analysis to maintain. Walked STOREKV runs once per
    // layer — the added cycles are noise.
    S2_SKS   = 6'd45
  } s2_e;
  s2_e state;

  // E-3b: the QSTAGE override window (route mux below) and the per-head
  // production counters. qs_cnt counts the CFG_D repeated composite words;
  // grp_cnt the CFG_D/8 injection beats.
  localparam int unsigned QS_BPR = CFG_D / 8;
  logic [7:0] qs_cnt;
  logic [4:0] grp_cnt;
  wire qs_hold = (state inside {S2_QSL, S2_QSF, S2_QSQ, S2_QSC, S2_QSS,
                                S2_QSE, S2_QSD, S2_QSA, S2_QSW});

  // ── E-5 fuel-fed projection: shape, window and the act-row cursor ────────
  // The MXE's activation port is fed by the act stage buffer's EMIT
  // (apex_top.sv:2215). PAT_ROW emits ONE stored row as nb lane8 beats, so
  // a K-wide contraction is FP_ROWS = K/FEED_DM consecutive PAT_ROW emits
  // into the same descriptor's ingest — the "32 stage rows" arithmetic
  // make_weight_image.py already prints for a k-chunk. The row family is
  // HOST-LOADED before walk_en (the descriptor's documented host-loaded
  // fields, exactly like the RQ/QC calibration slots and the X row), and
  // it is REPLAYED per n-job: one projection = n_splits x FP_ROWS emits.
  localparam int unsigned FP_BPR = CFG_D / 8;   // beats per act row
  localparam int unsigned FP_SH  = $clog2(CFG_D);
  // The act stage buffer is elaborated at CFG_DM (== FEED_DM here) while
  // this walker's aj_nb field is sized by CFG_D (STAGE_NB_W above). They
  // agree in every buildable image (the GAP-D note at FEED_DM), and an
  // image where they do not is REFUSED rather than fed a truncated nb.
  localparam bit FP_SHAPE_OK = (FEED_DM == CFG_D);
  // ── E-6: which masks may carry EN_FPROJ (fence (2) at S2_CHECK) ──────────
  // E-5 pinned the mask to EXACTLY {FPROJ, QKV}. E-6 un-fences the o8
  // epilogue, so legality is a predicate now — each clause is a MEASURED
  // structural fact, not a preference:
  //   * NORM1/NORM2/FFN excluded: their K_FETCH pcs fetch tensors the fuel
  //     stream cannot deliver to a consumer. The gammas dead-end (no
  //     xw -> xg route exists — STEP_MATRIX.md p_norm2 class), and with
  //     fuel ARMED the fetch would ISSUE and park kilobytes of gamma bytes
  //     in the fuel FIFO ahead of the next step's weight beats — silent
  //     stream poisoning, the worst failure shape there is. FFN is doubly
  //     out: the template runs ALL gate jobs then ALL up jobs (PC_WGJ <
  //     PC_WUJ) while asu_swiglu alternates gate/up phases per 64-col
  //     serializer frame (asu_swiglu.sv ST_GATE/ST_UP; apex_top.sv:1548
  //     phase steer) — a walked gate stream would be eaten as up data and
  //     the chunked USWI jobs starve. DOWN is mask-inseparable from FFN
  //     (PC_WDF/WDJ gate on W2_EN_FFN), so the walked-DOWN epilogue rides
  //     the same refusal until the FFN template interleaves; its epilogue
  //     CLASS (pc_hasrq) is the one proven here on OPROJ.
  //   * OPROJ <-> RES1 paired: the epilogue's deq output has exactly one
  //     armed consumer, the residual add job (apex_top.sv:1576 res_iv =
  //     ldq_ov && l_resid_arm_q). OPROJ without RES1 wedges the deq stream
  //     on a consumer that never arrives; RES1 without OPROJ is the
  //     measured data-starvation class (STEP_MATRIX.md p_res1).
  //   * QSTAGE is refused under FPROJ OUTRIGHT: the FPROJ fence requires
  //     fuel armed at S2_CHECK (wf_ready), and with fuel_src=1 the CL's xw
  //     mux (cl_apex.sv:1074 `fuel_src ? fuel_xw : mailbox`) disconnects
  //     the MAILBOX stream QSTAGE's k2 injection descriptors consume —
  //     S2_QSD would starve forever on weight beats the host can no longer
  //     deliver. A wedge, so it is refused here (§A-1), never walked into.
  //     SCORE/PV compose fine (their q rows are host-PRE-staged, the l3
  //     walker-mode idiom — staging happens before fuel mode is armed).
  //   * QKV excludes SCORE/PV: the QKV act family occupies act-bank-1 rows
  //     0..fp_rows-1 — the very rows a pre-staged attention walk's
  //     per-head q re-emit reads (q_row_sel).
  // E-7/E-7b widened this predicate again (E-6 widened E-5's exact
  // {FPROJ, QKV}). Each clause is a MEASURED structural fact, re-verified
  // by execution 2026-08-05 before this change:
  //   * FFN stays excluded (fence 2, OPEN): the template runs ALL gate
  //     jobs then ALL up jobs (PC_WGJ < PC_WUJ) while asu_swiglu
  //     alternates gate/up phases per 64-col serializer frame
  //     (asu_swiglu.sv:106 ST_GATE/ST_UP; apex_top.sv:1547-1555 swg_up_q
  //     flips per accepted `last`) — a walked gate stream would be eaten
  //     as up data and the chunked USWI jobs deadlock at the second push
  //     (one pending LAYER job; its frames cannot arrive until pcs the
  //     walker has not reached). Closing this is a template-order
  //     interleave (fetch/job/frame per 64 cols) + a re-laid image, not a
  //     route fix — left fenced, named follow-on.
  //   * NORM1/NORM2 are legal IFF FGAM (E-7): the gamma fetch now has a
  //     consumer — the lw_gsrc window steers xw into the gamma unpacker.
  //     Without FGAM the E-6 poisoning refusal holds verbatim.
  //   * QSTAGE stays refused OUTRIGHT (fence 4, STRUCTURAL): its k2
  //     injection weights are DATA-dependent (gen_l3_vectors.py:495-505
  //     w0/w1 = decompose of the very f16 values being staged, produced
  //     MID-walk) and ride the mailbox xw stream that fuel_src=1
  //     disconnects (cl_apex.sv:1074) — they cannot be pre-baked into a
  //     DDR image, and no second xw ingress exists. Real datapath work.
  //   * OPROJ <-> RES1 paired; DOWN <-> RES2 paired (E-7b) — either
  //     alone is the measured starvation class (STEP_MATRIX.md p_res1).
  //   * QKV+attention now COMPOSE (fence 5 CLOSED): the collision was the
  //     QKV act family emitting from rows 0..fp_rows-1 — the very rows a
  //     pre-staged attention walk's per-head q re-emit reads (q_row_sel).
  //     fp_base below stacks every family in template order instead
  //     (q rows, then QKV, then OPROJ, then DOWN), so legality is now
  //     just the fp_top capacity bound. Masks legal BEFORE this change
  //     keep their exact bases (each new term is 0 there) — byte-identical.
  wire fp_gam       = d2.en_mask[W2_EN_FGAM];
  wire fp_dwn       = d2.en_mask[W2_EN_DOWN];
  wire fp_norms     = d2.en_mask[W2_EN_NORM1] || d2.en_mask[W2_EN_NORM2];
  // FFN interleave (2026-08-08): EN_FFN left this list — the chunk loop
  // (PC_USWI..PC_WUJ re-entrant, per-chunk fuel records, S2_PSJ framing)
  // IS the sequencing the fence demanded. QSTAGE stays refused (one xw
  // ingress), as do gamma-less norms.
  wire fp_bad_steps = d2.en_mask[W2_EN_QSTAGE]
                    || (fp_norms && !fp_gam)
                    // chunk width == act row width only at CFG_D=64; a
                    // D=128 chunked stage would fill half-rows — refuse
                    || (d2.en_mask[W2_EN_FFN] && (CFG_D != 64));
  wire fp_attn      = d2.en_mask[W2_EN_SCORE] || d2.en_mask[W2_EN_PV];
  // (FFN is refused under FPROJ, so DOWN <-> RES2 pairing needs no FFN
  //  carve-out here — the legacy FFN+RES2 shape cannot reach this clause.)
  wire fp_mask_ok   = (d2.en_mask[W2_EN_QKV] || d2.en_mask[W2_EN_OPROJ]
                       // FFN interleave: a chunked gate/up walk is a legal
                       // fuel family of its own now (2026-08-08)
                       || d2.en_mask[W2_EN_FFN]
                       || fp_dwn || (fp_gam && fp_norms))
                    && !fp_bad_steps
                    && (d2.en_mask[W2_EN_OPROJ] == d2.en_mask[W2_EN_RES1])
                    && (fp_dwn == d2.en_mask[W2_EN_RES2])
                    && (!fp_gam || fp_norms);
  logic [4:0] fp_row;                            // act-row cursor
  wire        fproj_en = d2.en_mask[W2_EN_FPROJ];
  // the CURRENT projection pc's family row count — pc_ktot is d_model for
  // QKV/OPROJ and d_ffn for DOWN, so one expression serves all three
  // (pc is stable across the whole S2_PAE/PJOB/PAD loop).
  wire [15:0] fp_krows = pc_ktot >> FP_SH;
  wire        fp_last_row = (int'({27'b0, fp_row}) == int'({16'b0, fp_krows}) - 1);
  // E-7: per-family row counts (mask + geometry only — deterministic, so
  // the host stages each family at the same rows) and the template-order
  // stack. The rows below a family are owned by, in order: the staged-q
  // rows (attention), the QKV act family, the OPROJ (o8) family, the DOWN
  // (swiglu-product) family.
  wire [15:0] fp_r_attn  = fp_attn ? 16'({8'b0, d2.n_heads}) : 16'b0;
  wire [15:0] fp_r_qkv   = d2.en_mask[W2_EN_QKV]
                         ? (16'(d2.d_model) >> FP_SH) : 16'b0;
  wire [15:0] fp_r_oproj = d2.en_mask[W2_EN_OPROJ]
                         ? (16'(d2.d_model) >> FP_SH) : 16'b0;
  // FFN interleave (2026-08-08): the FFN's OWN x-row family — the input
  // the gate/up jobs emit — gets a stack slot between OPROJ and the
  // product rows, closing the NCHUNK>=2 collision (the products used to
  // land on the x row itself in a minimal mask).
  wire [15:0] fp_r_ffnx  = d2.en_mask[W2_EN_FFN]
                         ? (16'(d2.d_model) >> FP_SH) : 16'b0;
  // FFN interleave tail support (2026-08-08): the LAST chunk may be
  // narrower than 64 (synthetic stress shapes; every real-model d_ffn is
  // 64-aligned). The swiglu accepts <=64 cols per phase by contract.
  wire [15:0] ffn_rem = d2.d_ffn - 16'(32'(ffn_chunk_q) << 6);
  wire [15:0] ffn_cw  = (ffn_rem > 16'd64) ? 16'd64 : ffn_rem;
  wire [15:0] fp_r_down  = fp_dwn ? (16'(d2.d_ffn) >> FP_SH) : 16'b0;
  // the CURRENT pc's family base: QKV pcs sit below PC_WOF, OPROJ's below
  // PC_WDF — strict template order, so two comparators suffice.
  wire [15:0] fp_base = fp_r_attn
                      + ((pc >= PC_WOF)  ? fp_r_qkv   : 16'b0)
                      // OPROJ's rows join the base from the FFN segment on
                      // (PC_LVLG) — for pcs >= PC_WDF this is the same sum
                      // as the old (pc >= PC_WDF) term, so DOWN-only and
                      // legacy walks are byte-identical
                      + ((pc >= PC_LVLG) ? fp_r_oproj : 16'b0)
                      + ((pc >= PC_WDF)  ? fp_r_ffnx  : 16'b0);
  // the whole staged footprint the S2_CHECK fence bounds against STAGE_ROWS
  wire [15:0] fp_top = fp_r_attn + fp_r_qkv + fp_r_oproj + fp_r_ffnx
                     + fp_r_down;
  wire        fp_hold  = fproj_en && (state inside {S2_FETCH, S2_PSJ, S2_PAE,
                                                    S2_PJOB, S2_PAD, S2_PJW});
  // E-6: the epilogue window flag — registered at the K_FETCH/K_PJOB
  // dispatch from the pc's own pc_hasrq, it selects RT_FPRQ over RT_FPROJ
  // for the whole held window. Registered, not combinational, because pc
  // has ALREADY advanced in S2_PJW and a combinational pc_hasrq would
  // revert rdst to the RO lanes while the o8 tail is still in flight.
  logic       fp_rq_q;
  // E-7: the GAMMA WINDOW flag — registered at a G1/G2 K_FETCH dispatch of
  // an FGAM walk, cleared at the S2_NORM exit (dn_rms has fired, so the
  // norm has consumed EXACTLY the fetch's d_model gammas — the stream is
  // drained by construction) and on every walk exit. The next weight
  // fetch's record cannot ISSUE before the clear, so gamma beats and
  // weight beats can never interleave inside one window.
  logic       fp_gam_q;
  assign lw_gsrc = fp_gam_q;

  // the loaded image's format nibble selects the walker (header note): v1
  // images ride u_head untouched, fmt=1 runs the sequencer below
  assign v1_fwd = (desc2_word[W2_GEOM0][31:28] == WALK_FMT_V1)
                && (state == S2_IDLE);

  // ── the micro-step program (§2.3 template order, §3b concrete) ───────────
  localparam logic [5:0] PC_G1F  = 6'd0,  PC_NORM1 = 6'd1,
                         PC_WQF  = 6'd2,  PC_WQJ   = 6'd3,
                         PC_WKF  = 6'd4,  PC_WKJ   = 6'd5,
                         PC_WVF  = 6'd6,  PC_WVJ   = 6'd7,
                         PC_ROPE = 6'd8,  PC_STORE = 6'd9,
                         // E-3b: the walker-era q staging (E2E_TOY_LANE.md
                         // §4 E-3b). Placed after STOREKV, before the
                         // attention heads that consume the staged rows —
                         // the host's own pre-WALK_GO staging position.
                         // pc values below it shift by one; they are
                         // module-internal (the trace spec orders
                         // EMISSIONS, never pc encodings).
                         PC_QSTAGE = 6'd10,
                         PC_ATTN = 6'd11,
                         PC_LVLO = 6'd12, PC_URES1 = 6'd13, PC_UOPRJ = 6'd14,
                         PC_WOF  = 6'd15, PC_WOJ   = 6'd16,
                         // E-3: the r1 -> NORM2 seam, fed IN-TILE. Placed
                         // before the gamma-2 fetch so the row reaches the
                         // norm first and gamma releases its emission —
                         // the host demonstrator's order, verbatim.
                         PC_NFEED = 6'd17,
                         PC_G2F  = 6'd18, PC_NORM2 = 6'd19,
                         PC_LVLG = 6'd20, PC_USWI  = 6'd21,
                         PC_WGF  = 6'd22, PC_WGJ   = 6'd23,
                         PC_WUF  = 6'd24, PC_WUJ   = 6'd25,
                         PC_LVLD = 6'd26, PC_URES2 = 6'd27, PC_UDOWN = 6'd28,
                         PC_WDF  = 6'd29, PC_WDJ   = 6'd30,
                         PC_END  = 6'd31;

  typedef enum logic [3:0] {
    K_FETCH = 4'd0, K_PJOB = 4'd1, K_NORM = 4'd2, K_LVL = 4'd3,
    K_UJOB  = 4'd4, K_STORE = 4'd5, K_ATTN = 4'd6, K_END = 4'd7,
    K_NFEED = 4'd8, K_QSTAGE = 4'd9
  } kind_e;

  // registered error pulse/code of the fmt=1 sequencer (the module outputs
  // are combinational merges with the v1-forward mirror above)
  logic                walk2_err_q;
  walk_err_e           walk2_err_code_q;

  walk_desc2_t         d2;
  logic [5:0]          pc;
  logic [7:0]          h_idx;
  logic [ENG_W-1:0]    g_idx;
  logic [15:0]         gnum;
  logic [15:0]         k_rem, n_rem;
  logic [15:0]         ffn_chunk_q;   // FFN interleave: current 64-col chunk
  logic                ffn_win_q;     // inside the FFN chunk window
  logic                pj_first;
  logic [15:0]         lu_rem;      // unit-job chunk remainder
  logic [ENG_W-1:0]    sk_g;        // store-phase engine
  logic                sk_rec;      // 0 = K record, 1 = V record
  logic                abort_l;

  logic [15:0] kv_dim;
  assign kv_dim = 16'(d2.n_kv_heads) * 16'({8'b0, d2.head_dim});

  // per-pc decode
  kind_e       pc_kind;
  logic        pc_en;
  logic [3:0]  pc_tens;
  logic [15:0] pc_ktot, pc_ntot;
  logic        pc_mrows, pc_hasrq;
  logic [1:0]  pc_lvl;              // 0 rope, 1 o, 2 g, 3 d
  logic [1:0]  pc_unit;
  logic [1:0]  pc_ncomp;            // composites per chunk (0/1/2)
  logic [1:0]  pc_jca, pc_jcb;      // JC slot indices
  logic [15:0] pc_ucols;            // unit-job total cols

  always_comb begin
    pc_kind  = K_END;
    pc_en    = 1'b1;
    pc_tens  = 4'd0;
    pc_ktot  = d2.d_model;
    pc_ntot  = d2.d_model;
    pc_mrows = 1'b0;
    pc_hasrq = 1'b0;
    pc_lvl   = 2'd0;
    pc_unit  = WALK2_LU_DEQ;
    pc_ncomp = 2'd0;
    pc_jca   = 2'(WALK2_JC_OPROJ);
    pc_jcb   = 2'(WALK2_JC_UP);
    pc_ucols = d2.d_model;
    unique case (pc)
      PC_G1F:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_G1);
                      pc_en = d2.en_mask[W2_EN_NORM1]; end
      PC_NORM1: begin pc_kind = K_NORM; pc_en = d2.en_mask[W2_EN_NORM1]; end
      PC_WQF:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_WQ);
                      pc_en = d2.en_mask[W2_EN_QKV]; end
      PC_WQJ:   begin pc_kind = K_PJOB; pc_en = d2.en_mask[W2_EN_QKV]; end
      PC_WKF:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_WK);
                      pc_ntot = kv_dim; pc_en = d2.en_mask[W2_EN_QKV]; end
      PC_WKJ:   begin pc_kind = K_PJOB; pc_ntot = kv_dim; pc_mrows = 1'b1;
                      pc_en = d2.en_mask[W2_EN_QKV]; end
      PC_WVF:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_WV);
                      pc_ntot = kv_dim; pc_en = d2.en_mask[W2_EN_QKV]; end
      PC_WVJ:   begin pc_kind = K_PJOB; pc_ntot = kv_dim; pc_mrows = 1'b1;
                      pc_en = d2.en_mask[W2_EN_QKV]; end
      PC_ROPE:  begin pc_kind = K_LVL; pc_lvl = 2'd0;
                      pc_en = d2.en_mask[W2_EN_ROPE]; end
      PC_STORE: begin pc_kind = K_STORE;
                      pc_en = d2.en_mask[W2_EN_STOREKV]; end
      PC_QSTAGE: begin pc_kind = K_QSTAGE;
                       pc_en = d2.en_mask[W2_EN_QSTAGE]; end
      PC_ATTN:  begin pc_kind = K_ATTN;
                      pc_en = d2.en_mask[W2_EN_SCORE]
                              || d2.en_mask[W2_EN_PV]; end
      PC_LVLO:  begin pc_kind = K_LVL; pc_lvl = 2'd1;
                      pc_en = d2.en_mask[W2_EN_OPROJ]; end
      PC_UOPRJ: begin pc_kind = K_UJOB; pc_unit = WALK2_LU_DEQ;
                      pc_ncomp = 2'd1; pc_jca = 2'(WALK2_JC_OPROJ);
                      pc_en = d2.en_mask[W2_EN_OPROJ]; end
      PC_URES1: begin pc_kind = K_UJOB; pc_unit = WALK2_LU_RESID;
                      pc_en = d2.en_mask[W2_EN_RES1]; end
      PC_WOF:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_WO);
                      pc_en = d2.en_mask[W2_EN_OPROJ]; end
      PC_WOJ:   begin pc_kind = K_PJOB; pc_hasrq = 1'b1;
                      pc_en = d2.en_mask[W2_EN_OPROJ]; end
      PC_NFEED: begin pc_kind = K_NFEED;
                      pc_en = d2.en_mask[W2_EN_NFEED]; end
      PC_G2F:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_G2);
                      pc_en = d2.en_mask[W2_EN_NORM2]; end
      PC_NORM2: begin pc_kind = K_NORM; pc_en = d2.en_mask[W2_EN_NORM2]; end
      PC_LVLG:  begin pc_kind = K_LVL; pc_lvl = 2'd2;
                      pc_en = d2.en_mask[W2_EN_FFN]; end
      // FFN INTERLEAVE (fence-2 closure): the segment runs PER 64-COL
      // CHUNK — one swiglu job, one gate frame+jobs, one up frame+jobs —
      // and the S2_PAD/S2_PJOB completion loops PC_WUJ back to PC_USWI
      // ffn_nchunk times. A 64-col slice of the EXISTING Wg/Wu tensors is
      // contiguous in the weight-stationary image (job blocks are laid
      // column-major), so the per-chunk fuel record is pure address
      // arithmetic — NO image re-lay (FFN_INTERLEAVE.md deliverable B
      // dissolved, 2026-08-08).
      PC_USWI:  begin pc_kind = K_UJOB; pc_unit = WALK2_LU_SWIGLU;
                      pc_ncomp = 2'd2; pc_jca = 2'(WALK2_JC_GATE);
                      pc_jcb = 2'(WALK2_JC_UP);
                      pc_ucols = ffn_cw;
                      pc_en = d2.en_mask[W2_EN_FFN]; end
      PC_WGF:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_WG);
                      pc_ntot = ffn_cw;
                      pc_en = d2.en_mask[W2_EN_FFN]; end
      PC_WGJ:   begin pc_kind = K_PJOB; pc_ntot = ffn_cw;
                      pc_en = d2.en_mask[W2_EN_FFN]; end
      PC_WUF:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_WU);
                      pc_ntot = ffn_cw;
                      pc_en = d2.en_mask[W2_EN_FFN]; end
      PC_WUJ:   begin pc_kind = K_PJOB; pc_ntot = ffn_cw;
                      pc_en = d2.en_mask[W2_EN_FFN]; end
      // E-7b gave DOWN its own bit; the FFN interleave (2026-08-08)
      // DECOUPLES the legacy EN_FFN||EN_DOWN enable: no EN_FFN walk ever
      // ran under the coupled template (fence 2 refused them all), so the
      // coupling had zero working users — and an FFN-only walk must not
      // drag un-parameterized DOWN jobs in. The composed layer enables
      // BOTH bits explicitly.
      PC_LVLD:  begin pc_kind = K_LVL; pc_lvl = 2'd3;
                      pc_en = d2.en_mask[W2_EN_DOWN]; end
      PC_UDOWN: begin pc_kind = K_UJOB; pc_unit = WALK2_LU_DEQ;
                      pc_ncomp = 2'd1; pc_jca = 2'(WALK2_JC_DOWN);
                      pc_en = d2.en_mask[W2_EN_DOWN]; end
      PC_URES2: begin pc_kind = K_UJOB; pc_unit = WALK2_LU_RESID;
                      pc_en = d2.en_mask[W2_EN_RES2]; end
      PC_WDF:   begin pc_kind = K_FETCH; pc_tens = 4'(WALK2_TENS_WD);
                      pc_ktot = d2.d_ffn;
                      pc_en = d2.en_mask[W2_EN_DOWN]; end
      PC_WDJ:   begin pc_kind = K_PJOB; pc_ktot = d2.d_ffn; pc_hasrq = 1'b1;
                      pc_en = d2.en_mask[W2_EN_DOWN]; end
      default:  pc_kind = K_END;
    endcase
  end

  // requant slot for the epilogue-bearing projections: RQ[H] (o-proj at
  // pc<PC_G2F) or RQ[H+1] (down)
  logic [31:0] rq_word;
  assign rq_word = desc2_word[W2_RQ0 + 32'({24'b0, d2.n_heads})
                              + ((pc == PC_WDJ) ? 32'd1 : 32'd0)];

  // current MXE job shape from the remainder counters. n is bounded by
  // WALK2_N_MXE (the array width the tile actually implements), NOT by the
  // 12-bit descriptor field — see the D-029 erratum note in seq_walker_pkg.
  // THIRD ERRATUM (W1, 2026-08-05): k is bounded by walk2_k_job(FEED_DM) —
  // the act stage buffer (elaborated at CFG_DM == FEED_DM, apex_top:2203)
  // holds at most 31 rows per bank, so K_MAX itself is only a legal chunk
  // at a 128-wide row; at 64 the bound is 1984 (pkg note at walk2_k_job).
  // A D=128 walk is byte-identical: walk2_k_job(128) == WALK2_K_JOB.
  localparam int unsigned K_CHUNK = walk2_k_job(FEED_DM);
  logic [15:0] k_cur, n_cur;
  logic        last_k, last_n;
  assign k_cur  = (k_rem > 16'(K_CHUNK)) ? 16'(K_CHUNK) : k_rem;
  assign n_cur  = (n_rem > 16'(WALK2_N_MXE)) ? 16'(WALK2_N_MXE) : n_rem;
  assign last_k = (k_rem <= 16'(K_CHUNK));
  assign last_n = (n_rem <= 16'(WALK2_N_MXE));

  // unit-job chunk (JOBC rewritten per chunk; §3b index-reset rule). This is
  // the LAYER_JOB cols path. SECOND ERRATUM (2026-07-30, R1): the chunk
  // bound is PER UNIT — the CONSUMING unit's capacity (walk2_lu_bound),
  // never the 12-bit field: apex_top hands asu_swiglu only 7 of these 12
  // bits, so a field-bound chunk arrived truncated (2564 -> 4, silently
  // accepted). See the erratum note in seq_walker_pkg.
  logic [15:0] lu_bound, lu_cur;
  logic        lu_last;
  assign lu_bound = 16'(walk2_lu_bound(pc_unit));
  assign lu_cur   = (lu_rem > lu_bound) ? lu_bound : lu_rem;
  assign lu_last  = (lu_rem <= lu_bound);

  // ── E-3 NFEED shape, DIVISION-FREE ───────────────────────────────────────
  // The step needs "how many C-1 feeder rows is d_model" and "how many
  // columns of the residual row to egress". walk_desc2_check already pins
  // d_model == n_heads * head_dim and head_dim == CFG_D, so WHEN the build's
  // feeder row equals the per-head row (FEED_DM == CFG_D — every image this
  // repo can build or fly, E2E_TOY_LANE.md §2) the row count is EXACTLY
  // n_heads and no divider is needed. A build where they differ is the
  // unbuilt wide-feeder split: the step is REFUSED there (fence at S2_CHECK)
  // rather than fed a guessed row count.
  localparam bit NF_ROWS_EXACT = (FEED_DM == CFG_D);
  logic [DIM_W-1:0] nf_rows;
  assign nf_rows = DIM_W'({4'b0, d2.n_heads});

  // ── fetch request (whole tensor; gammas are 2*d_model bytes) ─────────────
  logic [31:0] fr_bytes;
  assign fr_bytes = (pc_tens == 4'(WALK2_TENS_G1)
                     || pc_tens == 4'(WALK2_TENS_G2))
                  ? {15'b0, d2.d_model, 1'b0}
                  : 32'(pc_ktot) * 32'(pc_ntot);
  assign wf_valid       = (state == S2_FETCH);
  assign wf_req.tag     = {4'b0, pc_tens};
  assign wf_req.beats64 = WALK2_FR_BEATS_W'(fr_bytes >> 6);
  // FFN interleave: chunk c of Wg/Wu starts c * (d_model*64/64) = c *
  // d_model 64B-beats into the tensor (contiguous column-major job blocks).
  // The product cannot exceed FR_BASE width in any legal geometry (chunk
  // count <= d_ffn/64 <= 1024, d_model <= 2^16 -> < 2^26 beats).
  wire [WALK2_FR_BASE_W-1:0] ffn_chunk_off =
      (pc == PC_WGF || pc == PC_WUF)
    ? WALK2_FR_BASE_W'(32'(ffn_chunk_q) * 32'(d2.d_model))
    : '0;
  assign wf_req.base64  = desc2_word[W2_TENS0 + {28'b0, pc_tens}]
                          [WALK2_FR_BASE_W-1:0]
                        + ffn_chunk_off;

  // ── projection descriptor (Q1-provisional encoding, header note) ─────────
  logic        pj_valid;
  mxe_desc_t   pj_desc;
  assign pj_valid = (state == S2_PJOB);
  always_comb begin
    pj_desc            = '0;
    // E-5 — THE PROVISIONAL OPCODE WAS WRONG FOR A RESIDENT IMAGE, and the
    // walked projection had never executed, so nothing caught it. The DDR
    // weight image's byte order is WEIGHT-STATIONARY: make_weight_image.py
    // lays each job block out as byte (p*8+c)*8+r == W[8p+r][c] and
    // cross-checks that against the frozen gen_l3_vectors.wgt_beats_ws, and
    // the host-mode proof that grades bit-exact pushes OP_GEMM_WS
    // (gemm_job.py:367 desc_words(OP_GEMM_WS, 1, 896, 8), mode_os=0). A
    // walked OP_GEMM_OS descriptor would read those same bytes in the
    // output-stationary order and grade RED for a reason that has nothing
    // to do with the fuel path. FPROJ therefore emits the order the image
    // is IN; the legacy (never-executed) OS encoding is left untouched for
    // non-FPROJ walks rather than silently redefined.
    pj_desc.opcode     = fproj_en ? OP_GEMM_WS : OP_GEMM_OS;
    pj_desc.mode_os    = fproj_en ? 1'b0 : 1'b1;
    pj_desc.m_dim      = pc_mrows ? DIM_W'({4'b0, d2.t_rows}) : DIM_W'(1);
    pj_desc.k_dim      = DIM_W'(k_cur);
    pj_desc.n_dim      = DIM_W'(n_cur);
    pj_desc.accumulate = ~pj_first;
    if (pc_hasrq && last_k) begin
      pj_desc.requant_en = 1'b1;
      pj_desc.rq_scale   = rq_word[15:0];
      pj_desc.rq_shift   = rq_word[20:16];
    end
  end

  // E-3b: the QSTAGE production descriptor — the TB-side k2-injection GEMM
  // shape (OP_GEMM_WS, m=1, k=2, n=8) whose weight beats arrive on the xw
  // stream (rt_wgt_src=0 under the QSTAGE route) and whose act beat is the
  // loader row's first beat (the aj EMIT in S2_QSE).
  logic      qd_valid;
  mxe_desc_t qd_desc;
  assign qd_valid = (state == S2_QSD);
  always_comb begin
    qd_desc        = '0;
    qd_desc.opcode = OP_GEMM_WS;
    qd_desc.m_dim  = DIM_W'(1);
    qd_desc.k_dim  = DIM_W'(2);
    qd_desc.n_dim  = DIM_W'(MXE_N);
  end

  // ── ds mux: projections, QSTAGE production and the wrapped engine never
  //    overlap (one FSM state at a time) ────────────────────────────────────
  assign ds_valid   = (pj_valid || qd_valid) ? 1'b1 : h_ds_valid;
  assign ds_desc    = pj_valid ? pj_desc : qd_valid ? qd_desc : h_ds_desc;
  assign h_ds_ready = (pj_valid || qd_valid) ? 1'b0 : ds_ready;

  // ── LAYER channels ───────────────────────────────────────────────────────
  assign lj_valid = (state == S2_NORM);
  assign lj_cols  = DIM_W'(d2.d_model);
  assign jc_valid = (state == S2_JCA) || (state == S2_JCB);
  assign jc_data  = desc2_word[W2_JC0 + ((state == S2_JCB)
                                         ? {30'b0, pc_jcb}
                                         : {30'b0, pc_jca})];
  // E-3: the unit-3 push shares this single LAYER_JOB ingress (R5) — same
  // channel, same registered-accept, a different unit id and an UNCHUNKED
  // cols (see the walk2_lu_bound note in seq_walker_pkg: an egress names a
  // window, so splitting it would re-read window 0 per chunk).
  assign lu_valid = (state == S2_LU) || (state == S2_NFU);
  assign lu_unit  = (state == S2_NFU) ? WALK2_LU_NORM : pc_unit;
  assign lu_cols  = (state == S2_NFU) ? d2.d_model[11:0] : lu_cur[11:0];

  // E-3: feeder-job mux (the ds_* idiom above). The wrapped v1 engine owns
  // this channel during attention; the NFEED step owns it in S2_NFF and
  // the E-3b QSTAGE step in S2_QSF. All owners are mutually exclusive by
  // construction — a head walk lives in S2_HLOAD/HGO/HWAIT, and the
  // sequencer is in exactly one state.
  // E-7 adds the y-drain feeder job (S2_GDF) as a fourth exclusive owner:
  // n_heads rows, exactly the NFEED framing (the norm row IS d_model wide).
  wire fj_own = (state == S2_NFF) || (state == S2_QSF) || (state == S2_GDF)
             || (state == S2_CSF);
  assign fj_valid   = fj_own ? 1'b1 : h_fj_valid;
  assign fj_rows    = (state == S2_NFF || state == S2_GDF) ? nf_rows
                    : (state == S2_QSF || state == S2_CSF) ? DIM_W'(1)
                    : h_fj_rows;
  assign h_fj_ready = fj_own ? 1'b0 : fj_ready;

  // ── E-3b QSTAGE channel muxes (same ownership idiom) ─────────────────────
  // squant job: ONE MODE_F16 job of the head row (mode 0, cols = CFG_D) —
  // golden's `_f16(q_real)` narrowing point, exactly the host staging's.
  wire qj_own = (state == S2_QSQ);
  assign qj_valid   = qj_own ? 1'b1 : h_qj_valid;
  assign qj_mode    = qj_own ? 1'b0 : h_qj_mode;
  assign qj_cols    = qj_own ? DIM_W'(CFG_D) : h_qj_cols;
  assign h_qj_ready = qj_own ? 1'b0 : qj_ready;
  // squant sideband: the ONE per-step S-2 composite (W2_QC), repeated
  // CFG_D times (the per-tensor weight-scale model — pkg note at W2_QC).
  wire qsb_own = (state == S2_QSC);
  assign qs_valid   = qsb_own ? 1'b1 : h_qs_valid;
  assign qs_data    = qsb_own ? desc2_word[W2_QC] : h_qs_data;
  assign h_qs_ready = qsb_own ? 1'b0 : qs_ready;
  // serializer job: frames the head's production stream (CFG_D/8 lane32
  // beats x 8 lanes) — the walker-era half of E-3b's new surface.
  // E-6 joins as a second owner (S2_PSJ): ONE job framing the epilogue
  // projection's whole o8 result stream — n_splits beats (one m=1/n=8 job
  // = one lane32 result beat) x 8 lanes = pc_ntot serial elements, exactly
  // the deq job's cols, with out_last on the final element (the frame
  // apex_layer_deq's ilast contract requires). The host-mode o8 leg frames
  // with the same verb (gen_layer_ops.py inject_frame `ljob(beats, 8)`).
  // The beats field is 8 bits; the S2_CHECK fence bounds n_splits <= 255.
  assign sj_valid = (state == S2_QSS) || (state == S2_PSJ);
  assign sj_beats = (state == S2_PSJ) ? 8'((pc_ntot + 16'd7) >> 3)
                                      : 8'(QS_BPR);
  assign sj_lanes = 4'd8;
  // act-stage job: the k2-injection loader-row EMIT (bank 0, one beat) per
  // production group, then the staged row's LOAD into act bank 1 ROW h —
  // the row the attention walk's per-head re-emit reads (q_row_sel).
  // E-5 joins this mux as a third owner (S2_PAE). Every QSTAGE-side value
  // below is unchanged when fp_own is low, so an E-3b walk is byte-
  // identical; S2_PAE is unreachable without W2_MASK[14].
  wire fp_own = (state == S2_PAE);
  // E-7: the y-drain act LOAD (S2_GDA) — n_heads rows into act bank 1 at
  // the fp_top window (above every staged family), the host drain's own
  // aj verb with a row count and base.
  wire gd_own = (state == S2_GDA);
  wire cs_own = (state == S2_CSA);
  // the DOWN act family's base row — the fp_base sum a PC_WDF-range pc
  // would see, computed here because the stage happens while pc is still
  // in the FFN range (bounded by the fp_top <= STAGE_ROWS S2_CHECK fence).
  // AFTER the FFN x family: the products must never overwrite the x row.
  wire [15:0] fp_dwn_base = fp_r_attn + fp_r_qkv + fp_r_oproj + fp_r_ffnx;
  wire aj_own = (state == S2_QSE) || (state == S2_QSA) || fp_own || gd_own
             || cs_own;
  assign aj_valid = aj_own ? 1'b1 : h_aj_valid;
  assign aj_op    = aj_own ? (fp_own ? 1'b1 : (state == S2_QSE)) : h_aj_op;
  // (cs_own falls into the LOAD arm: aj_op 0 like S2_GDA/S2_QSA)
  // bank 1 — the act bank the host-mode proof stages its C-1 activation
  // family into and emits from (gemm_job.py:346-351 `aj(0,1,0,1,8,r)` LOAD
  // / :366-369 `aj(1,1,0,1,8,r)` EMIT). Bank 0 is the loader row's.
  assign aj_bank  = aj_own ? ((fp_own || gd_own || cs_own) ? 1'b1
                                                            : (state == S2_QSA))
                           : h_aj_bank;
  assign aj_pat   = aj_own ? 2'd0 : h_aj_pat;          // PAT_ROW for both
  assign aj_rows  = aj_own ? (gd_own ? 5'(d2.n_heads) : 5'd1) : h_aj_rows;
  // (cs_own: 1 row — the else arm above)
  assign aj_nb    = aj_own ? ((fp_own || gd_own || cs_own)
                                ? STAGE_NB_W'(FP_BPR)
                              : (state == S2_QSE) ? STAGE_NB_W'(1)
                                                  : STAGE_NB_W'(QS_BPR))
                           : h_aj_nb;
  // E-6/E-7: every fuel-fed family emits from its OWN row window —
  // fp_base + fp_row, the template-order stack (staged-q rows, QKV, OPROJ,
  // DOWN; bounded <= STAGE_ROWS at S2_CHECK, so the 5-bit cast is exact).
  // pc is stable across the whole S2_PAE/PJOB/PAD loop, so fp_base is too.
  // Masks legal before E-7 see their exact old bases (fp_base reduces to 0
  // for QKV-without-attention and to E-6's fp_oproj_base for OPROJ).
  assign aj_sel   = aj_own ? (fp_own ? 5'(fp_base + {11'b0, fp_row})
                              : gd_own ? 5'(fp_top)
                              : cs_own ? 5'(fp_dwn_base + ffn_chunk_q)
                              : (state == S2_QSA) ? 5'(h_idx) : 5'd0)
                           : h_aj_sel;
  assign h_aj_ready = aj_own ? 1'b0 : aj_ready;

  // ── E-3b fmt=1 route override ────────────────────────────────────────────
  // {kvu, qsrc, rdst[1:0], wsrc, asrc, fdst, fsrc}. RT_QSTAGE is the
  // production route: MXE result -> serializer (rdst=1), serializer ->
  // squant (qsrc=0), feeder codes -> ACT stage (fdst=0, asrc=0), MXE
  // weights external (wsrc=0 — the xw injection stream). RT_SCORE is
  // u_head's own score constant, loaded at the QSTAGE exit so the
  // hand-back value is DETERMINISTIC (u_head's parked value could be its
  // pv word from an earlier walk). rt_f1_q otherwise TRACKS u_head, so a
  // walk that never enters a QSTAGE state sees u_head's routes verbatim
  // at every mux boundary (values equal whenever the select flips).
  localparam logic [7:0] RT_QSTAGE = 8'h91;
  localparam logic [7:0] RT_SCORE  = 8'hEF;
  // ── E-5 fmt=1 route override: THE FUEL-FED PROJECTION ────────────────────
  // Same {kvu, qsrc, rdst[1:0], wsrc, asrc, fdst, fsrc} encoding. RT_FPROJ
  // is RT_SCORE with EXACTLY TWO bits moved, and both moves are the
  // measured blockers named in STEP_MATRIX.md:
  //   wsrc 1 -> 0 : the MXE takes its weight beats from the EXTERNAL xw
  //                 stream (apex_top.sv:2273-2276). In this build that port
  //                 is driven by the fuel mux (cl_apex.sv:1074, fuel_src ?
  //                 fuel_xw : mailbox), so wsrc=0 IS the DDR path. This is
  //                 the pin STEP_MATRIX called "rt_wgt_src pinned 1 in walk
  //                 mode -> the external xw path is dead".
  //   rdst 2 -> 0 : the INT32 accumulators egress on `ro` (apex_top.sv:2281)
  //                 — the cap lanes the host-mode fuel proof grades. rdst=2
  //                 would post them to the score dequant, which is the
  //                 attention route, not a projection's.
  // The remaining bits are NOT invented here: RT_FPROJ reproduces, bit for
  // bit, the route word the PROVEN host-mode fuel projection writes to
  // mailbox ROUTE0 (0x3140) — gemm_job.py:355 `route(rdst=0, wsrc=0,
  // asrc=1)` = 0x0084, the arm docs/results/ib_fuel_05b grades bit-exact
  // against golden. fsrc/fdst/qsrc are 0 there and 0 here; kvu is 1 there
  // (gen_l3_vectors.py:227's historical hardwire) and 1 here. Copying the
  // measured-good word rather than deriving a new one is deliberate: this
  // step's whole claim is that the WALKER can drive the datapath the host
  // already drives correctly, so the route must be the SAME route.
  localparam logic [7:0] RT_FPROJ  = 8'h84;
  // ── E-6 route override: THE WALKED EPILOGUE ──────────────────────────────
  // RT_FPRQ is RT_FPROJ with EXACTLY ONE field moved:
  //   rdst 0 -> 1 : the requantised o8 results post to the lane32
  //                 SERIALIZER (apex_top.sv:2287 ls_in_valid) instead of
  //                 the RO cap lanes — the front of the tile's own
  //                 o8 -> apex_layer_deq -> apex_residual chain. That rdst
  //                 is not invented: it is the measured host-mode o8 leg's
  //                 own destination (gen_layer_ops.py:795 `route(rdst=1)`
  //                 = 0x0090 for the deq/residual inject, and :800
  //                 `route(rdst=1, asrc=1)` = 0x0094 — this word — where
  //                 the act port also emits from the stage buffer, exactly
  //                 the E-5 activation-family arrangement kept here).
  // Selected over RT_FPROJ by fp_rq_q — the registered pc_hasrq of the
  // dispatched projection — for the whole held window including S2_PJW, so
  // a raw (QKV-class) projection still egresses on RO byte-identically and
  // the o8 tail can never be misrouted mid-drain.
  localparam logic [7:0] RT_FPRQ   = 8'h94;
  // ── E-7 route override: THE WALKED NORM-OUTPUT DRAIN ─────────────────────
  // RT_YDRAIN is the HOST's own measured drain route — elane_walk_norm
  // stage [3]/[8] arms `route(rdst=0, asrc=0)` = 0x0080 (kvu=1, all path
  // selects 0) so the norm's y -> widen -> C-1 feeder codes land in the
  // ACT stage buffer (as_ld takes the feeder output when asrc=0/fdst=0,
  // apex_top.sv as_ld_valid). Copying the measured-good word rather than
  // deriving one is the RT_FPROJ rule: the walker must drive the SAME
  // route the host already drives correctly.
  localparam logic [7:0] RT_YDRAIN = 8'h80;
  // FFN interleave window word: rdst=1 (MXE results -> serializer -> the
  // swiglu), wsrc=0 (fuel weights), asrc=0/fdst=0 (feeder -> act LOAD for
  // the product row), kvu=1 — RT_FPRQ with asrc moved to the feeder side.
  localparam logic [7:0] RT_FFN    = 8'h90;
  logic [7:0] rt_f1_q;
  wire gd_hold = fp_gam_q && (state inside {S2_GDL, S2_GDA, S2_GDF,
                                            S2_NORM, S2_GDW});
  // (the CS states ride the ffn_win_q RT_FFN hold below — no separate
  //  cs_hold: the window word already carries the feeder->act routing)
  wire head_direct = (state inside {S2_HLOAD, S2_HGO, S2_HWAIT}) || v1_fwd;
  wire qs_last_head = (int'({24'b0, h_idx}) == int'({24'b0, d2.n_heads}) - 1);
  always_ff @(posedge clk) begin
    if (!rst_n)                            rt_f1_q <= RT_SCORE;
    else if (state == S2_QSL && tile_idle) rt_f1_q <= RT_QSTAGE;
    else if (state == S2_QSW && tile_idle
             && qs_last_head)              rt_f1_q <= RT_SCORE;
    // E-5: hold RT_FPROJ across the WHOLE fuel-fed projection window —
    // fetch, act emits, descriptor pushes AND the post-job drain. Holding
    // it through S2_PJW is load-bearing: rdst selects the MXE result demux
    // COMBINATIONALLY, so reverting to RT_SCORE while accumulators are
    // still in flight would post the tail of the projection to the score
    // dequant instead of the cap lanes. S2_PJW waits for tile_idle, so the
    // revert happens only once the tile has nothing left to egress.
    // The one-cycle register lag is harmless here (unlike the QSTAGE arm,
    // which is why that one gates on tile_idle): the earliest consumer of
    // wsrc is a DDR weight beat, which cannot arrive until the fuel record
    // has crossed to the shell and back.
    else if (fproj_en && ffn_win_q)        rt_f1_q <= RT_FFN;
    else if (fp_hold)                      rt_f1_q <= fp_rq_q ? RT_FPRQ
                                                             : RT_FPROJ;
    // E-7: the y-drain window holds the host's own drain route from the
    // S2_GDL arm until S2_GDW's tile_idle — the S2_PJW argument: dn_rms
    // only means the y beats were ACCEPTED, not that the widen/feeder/act
    // tail has landed, so a mid-flight revert would misroute the tail.
    else if (gd_hold)                      rt_f1_q <= RT_YDRAIN;
    else if (!qs_hold)                     rt_f1_q <= {h_rt_kvu, h_rt_qsrc,
                                                       h_rt_rdst, h_rt_wsrc,
                                                       h_rt_asrc, h_rt_fdst,
                                                       h_rt_fsrc};
  end
  assign rt_feeder_src = head_direct ? h_rt_fsrc : rt_f1_q[0];
  assign rt_feeder_dst = head_direct ? h_rt_fdst : rt_f1_q[1];
  assign rt_act_src    = head_direct ? h_rt_asrc : rt_f1_q[2];
  assign rt_wgt_src    = head_direct ? h_rt_wsrc : rt_f1_q[3];
  assign rt_res_dst    = head_direct ? h_rt_rdst : rt_f1_q[5:4];
  assign rt_squant_src = head_direct ? h_rt_qsrc : rt_f1_q[6];
  assign rt_kv_user    = head_direct ? h_rt_kvu  : rt_f1_q[7];

  // ── STOREKV WADDR master (poll idle, then WRITE_ADDR; §3b/L3 store) ──────
  assign in_store    = (state == S2_SKPAR) || (state == S2_SKPR)
                    || (state == S2_SKWR) || (state == S2_SKB)
                    || (state == S2_SKS);
  assign kvm_awvalid = in_store ? (state == S2_SKWR) : h_aw_v;
  assign kvm_wvalid  = in_store ? (state == S2_SKWR) : h_w_v;
  assign kvm_awaddr  = in_store ? KV_WADDR : h_aw_a;
  // record indices: K at T-1, V at 2T-1 (the L3 store-address map)
  assign kvm_wdata   = in_store
                     ? {23'b0, (sk_rec ? {d2.t_rows, 1'b0} - 9'd1
                                       : {1'b0, d2.t_rows} - 9'd1)}
                     : h_w_d;
  assign kvm_arvalid = in_store ? (state == S2_SKPAR) : h_ar_v;
  assign kvm_araddr  = in_store ? KV_STAT : h_ar_a;
  assign kvm_bready  = in_store ? 1'b1 : h_b_r;
  assign kvm_rready  = in_store ? 1'b1 : h_r_r;

  assign kv_eng_sel = in_store ? sk_g : g_idx;
  // F6(ii): the act-stage row holding this head's staged q8. A v1-forwarded
  // walk never leaves h_idx at anything but 0, and Q_ROWS==1 pins it to 0
  // regardless, so both legacy paths are bit-identical.
  logic [4:0] h_q_row;
  assign h_q_row = (Q_ROWS > 1) ? 5'(h_idx) : 5'd0;

  assign hd_valid   = (state == S2_HLOAD);
  assign hd_head    = h_idx;
  // v1-forward: a forwarded walk lives entirely in u_head while state stays
  // S2_IDLE, so busy mirrors h_busy; the OR is exact in fmt=1 mode too
  // (head walks only happen inside non-IDLE states)
  assign walk2_busy = h_busy || (state != S2_IDLE);
  // v1-forward error mirror: same-cycle pass of u_head's pulse+code (the
  // direct D-028 instance's timing); the fmt=1 sequencer's own registered
  // pulse path is untouched (S2_HWAIT re-raises head errors one cycle
  // later, exactly as before the flip)
  assign walk2_err      = walk2_err_q || (v1_fwd && h_err);
  assign walk2_err_code = (v1_fwd && h_err) ? h_err_code : walk2_err_code_q;
  assign walk2_step = (state == S2_IDLE || state == S2_CHECK) ? WS2_IDLE
                    : (state == S2_DONE)                      ? WS2_DONE
                    : (state == S2_ERR)                       ? WS2_ERR
                    : (state == S2_HLOAD || state == S2_HGO
                       || state == S2_HWAIT)
                        ? ((h_phase == WPH_PV) ? WS2_PV : WS2_SCORE)
                    : in_store                                ? WS2_STOREKV
                    : (pc <= PC_NORM1)                        ? WS2_NORM1
                    // E-3b: q staging IS the q half of the QKV step
                    : (pc == PC_QSTAGE)                       ? WS2_QKV
                    : (pc <= PC_WVJ)                          ? WS2_QKV
                    : (pc == PC_ROPE)                         ? WS2_ROPE
                    : (pc == PC_URES1)                        ? WS2_RES1
                    : (pc <= PC_WOJ)                          ? WS2_OPROJ
                    // E-3: PC_NFEED lands in this bucket on purpose — the
                    // in-tile row feed IS the NORM2 step's front half
                    : (pc <= PC_NORM2)                        ? WS2_NORM2
                    : (pc == PC_USWI)                         ? WS2_SILU
                    : (pc <= PC_WUJ)                          ? WS2_FFN_GU
                    : (pc == PC_URES2)                        ? WS2_RES2
                                                              : WS2_FFN_DN;

  // ── the walk ─────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state          <= S2_IDLE;
      d2             <= '0;
      pc             <= '0;
      h_idx          <= '0;
      g_idx          <= '0;
      gnum           <= '0;
      k_rem          <= '0;
      n_rem          <= '0;
      ffn_chunk_q    <= '0;
      ffn_win_q      <= 1'b0;
      pj_first       <= 1'b1;
      lu_rem         <= '0;
      sk_g           <= '0;
      sk_rec         <= 1'b0;
      qs_cnt         <= '0;
      grp_cnt        <= '0;
      fp_row         <= '0;
      fp_rq_q        <= 1'b0;
      fp_gam_q       <= 1'b0;
      abort_l        <= 1'b0;
      h_en           <= 1'b0;
      h_go           <= 1'b0;
      for (int i = 0; i < WALK_DESC_WORDS; i++) h_desc[i] <= '0;
      lw_rope_en     <= 1'b0;
      lw_rope_bank   <= 1'b0;
      lw_rope_pos    <= '0;
      lw_ser_dst     <= '0;
      lw_fsrc_ext    <= '0;
      lw_resid_arm   <= 1'b0;
      lw_nsrc        <= 1'b0;
      lw_nsrc_own    <= 1'b0;
      walk2_err_q    <= 1'b0;
      walk2_err_code_q <= WALK_ERR_NONE;
    end else begin
      walk2_err_q <= 1'b0;                       // 1-cycle pulse
      h_go      <= 1'b0;
      if (abort_req) abort_l <= 1'b1;

      // D-020: abort between handshakes, never by retracting a presented
      // valid (§5). During a head walk the wrapped engine handles the abort
      // itself and reports WALK_ERR_ABORT, forwarded below.
      if (abort_l && (state inside {S2_CHECK, S2_STEP, S2_DONE})) begin
        state          <= S2_IDLE;
        abort_l        <= 1'b0;
        h_en           <= 1'b0;
        lw_nsrc_own    <= 1'b0;              // E-4b: ownership ends with
        fp_gam_q       <= 1'b0;              // E-7: so does the gamma window
        walk2_err_q    <= 1'b1;              // the walk, like every exit
        walk2_err_code_q <= WALK_ERR_ABORT;
      end else begin
        unique case (state)
          // fmt=0 images belong to u_head via the v1 forward (header note):
          // the fmt=1 sequencer must NOT react to their go, or its check
          // would latch a spurious DESC sticky over a clean v1 walk
          S2_IDLE: if (walk_en && walk_go && !abort_l
                       && (desc2_word[W2_GEOM0][31:28] != WALK_FMT_V1)) begin
            d2    <= walk_desc2_unpack(desc2_word[W2_GEOM0],
                                       desc2_word[W2_MODEL0],
                                       desc2_word[W2_MODEL1],
                                       desc2_word[W2_MASK],
                                       desc2_word[W2_STEP]);
            state <= S2_CHECK;
          end

          S2_CHECK: begin
            automatic walk_err_e e;
            e = walk_desc2_check(desc2_word[W2_GEOM0], desc2_word[W2_MODEL0],
                                 desc2_word[W2_MODEL1], desc2_word[W2_MASK],
                                 desc2_word[W2_STEP], CFG_D);
            // build-envelope fences (§A-1 pattern: REFUSE, never degrade).
            // The t_rows fence is the D-029 erratum's second clause: the K/V
            // projection jobs carry m = t_rows, and mxe_ctrl rejects m_dim >
            // M_TILE_MAX — WALK_T_MAX (128) is twice that, so a descriptor
            // with t_rows > WALK2_M_MXE and QKV enabled would emit an
            // MXE-ILLEGAL job. Refuse it here instead of walking into a
            // desc_error the walker cannot see. (Attention-only walks are
            // unaffected: those descriptors are m=1.)
            if (e == WALK_ERR_NONE
                && (((d2.kv_map == 2'b01)
                     && (int'({24'b0, d2.n_kv_heads}) > int'(N_ENG)))
                    || (d2.en_mask[W2_EN_FFN] && d2.d_ffn == 16'b0)
                    || (d2.en_mask[W2_EN_QKV]
                        && (int'({24'b0, d2.t_rows}) > int'(WALK2_M_MXE)))
                    // E-3 NFEED envelope (header note). Every clause guards
                    // a downstream refusal the walker could not SEE: the
                    // unit-3 push is gated on cols framing whole feeder rows
                    // (apex_top err_code 9) and the feeder job on
                    // 1 <= rows <= ROWS_MAX (its own job_error). Either
                    // refusal lands AFTER the push handshake the walker
                    // already took, leaving S2_NFW waiting on an nf_busy
                    // that will never rise — a silent wedge. Refuse the
                    // descriptor instead, before any state changes.
                    || (d2.en_mask[W2_EN_NFEED]
                        && (!NF_ROWS_EXACT
                            || (int'({24'b0, d2.n_heads}) > int'(FEED_ROWS))
                            || (int'({16'b0, d2.d_model})
                                > int'(WALK2_N_JOB))))
                    // E-3b QSTAGE envelope (§A-1: refuse, never degrade):
                    // QKV+QSTAGE would produce q twice; a head count past
                    // the build's staging depth (Q_ROWS = the tile's
                    // QSTAGE_H_MAX) would stage rows the attention walk
                    // cannot select; and a split-feeder build cannot frame
                    // the head row as ONE C-1 row (the NFEED fence's
                    // argument, per-head family).
                    || (d2.en_mask[W2_EN_QSTAGE]
                        && (d2.en_mask[W2_EN_QKV]
                            || (int'({24'b0, d2.n_heads}) > int'(Q_ROWS))
                            || !NF_ROWS_EXACT))
                    // E-4b cross-bit fences (header note; §A-1: refuse,
                    // never degrade). NSRC without NFEED would own the norm
                    // x-mux for a walk with nothing to feed it — the level
                    // pinned 0 over a host arm, a silent semantic change.
                    // QSTAGE+NFEED without NSRC is the MEASURED wedge: no
                    // single host-held l_nsrc serves codes->act (staging)
                    // and codes->norm (feed) in one kick — the staged row
                    // detours and S2_NFW waits on an nf_busy that never
                    // falls. Refuse both before any state changes.
                    || (d2.en_mask[W2_EN_NSRC]
                        && !d2.en_mask[W2_EN_NFEED])
                    || (d2.en_mask[W2_EN_QSTAGE] && d2.en_mask[W2_EN_NFEED]
                        && !d2.en_mask[W2_EN_NSRC])
                    // ── E-7 cross-bit fences (§A-1: refuse, never degrade)
                    // FGAM without FPROJ: the gamma fetch would present
                    // wf_valid with fuel unarmed — the S2_FETCH park wedge
                    // D-020 forbids escaping. FGAM without a NORM step:
                    // a gamma window with no consumer (checked again in
                    // fp_mask_ok for clause-order stability).
                    || (d2.en_mask[W2_EN_FGAM]
                        && (!d2.en_mask[W2_EN_FPROJ]
                            || !(d2.en_mask[W2_EN_NORM1]
                                 || d2.en_mask[W2_EN_NORM2])))
                    // ── E-7b cross-bit fence, amended by the FFN
                    // interleave (2026-08-08): the DOWN pcs are EN_DOWN-only
                    // now, so DOWN+FFN is no longer a double enable — it is
                    // THE COMPOSED LAYER (the chunk loop stages the DOWN act
                    // family; DOWN consumes it). The kept arm refuses the
                    // DOWN-ONLY shape without fuel (the E-7b fetch-park
                    // belt); the composed shape follows the same rule every
                    // fetch-bearing walk always had — fuel armed on the real
                    // tile, wf_ready stubbed at unit level.
                    || (d2.en_mask[W2_EN_DOWN]
                        && !d2.en_mask[W2_EN_FPROJ]
                        && !d2.en_mask[W2_EN_FFN])
                    // ── E-5 FPROJ envelope (§A-1: refuse, never degrade) ──
                    // (1) THE FUEL FENCE, and the reason it lives HERE.
                    //     S2_FETCH presents wf_valid and waits; D-020 forbids
                    //     aborting by retracting a presented valid, which is
                    //     exactly why the abort set above excludes S2_FETCH
                    //     and why STEP_MATRIX measured that park as a WEDGE
                    //     with no escape. So the escape is a PRE-ENTRY
                    //     REFUSAL: wf_ready is a LEVEL (apex_fuel_ctl.sv:195
                    //     wk_req_ready = src_q && !drain_q && !go && skid
                    //     room), so with the host quiet and nothing in
                    //     flight it reads "fuel is armed" one cycle before
                    //     the walk can reach a fetch. A walk that would park
                    //     is refused loudly instead of hanging the tile.
                    //     HONEST LIMIT, FENCED NOT FIXED: this proves fuel
                    //     was armed AT CHECK. Fuel mode dropping MID-walk
                    //     still wedges S2_FETCH — that needs either an abort
                    //     path that does not retract a valid, or a timeout,
                    //     and neither is in this step.
                    // (2) the FPROJ mask predicate (E-6 widened it from
                    //     E-5's exact {FPROJ, QKV}): a weight-consuming
                    //     step must be named (QKV or OPROJ); NORM1/NORM2/
                    //     FFN/RES2 are refused (gamma fetches would poison
                    //     the armed fuel stream — no xw consumer; the FFN
                    //     template order starves asu_swiglu's per-frame
                    //     gate/up alternation, which also keeps DOWN out);
                    //     OPROJ and RES1 come paired (the epilogue's deq
                    //     output has exactly one armed consumer, the
                    //     residual add — either alone is the measured
                    //     starvation class); QKV excludes the attention
                    //     steps (act-bank row collision with the staged-q
                    //     rows). See the fp_mask_ok block.
                    // (3) the act families must tile the contraction
                    //     EXACTLY into whole stage rows and fit ONE bank
                    //     TOGETHER (fp_top stacks the OPROJ family above
                    //     the q rows / the QKV family), and the stage
                    //     buffer's row width must equal the one this
                    //     walker's aj_nb field is sized by (the GAP-D note
                    //     at FEED_DM).
                    // (4) ONE k-split only — see the S2_PAD note.
                    // (5) E-6: an epilogue projection's o8 frame must fit
                    //     the serializer job's 8-bit beats field.
                    // NOT fenced here (the walker cannot know it): the
                    // residual RAM's DM_MAX vs d_model — the same consumer-
                    // internal refusal exposure the NFEED fence documents;
                    // every legal 0.5B/toy image is inside it.
                    || (d2.en_mask[W2_EN_FPROJ]
                        && (!wf_ready
                            || !fp_mask_ok
                            || !FP_SHAPE_OK
                            || (d2.d_model == 16'b0)
                            || ((d2.d_model & 16'(CFG_D - 1)) != 16'b0)
                            || (int'({16'b0, fp_top}) > int'(STAGE_ROWS))
                            // THIRD-ERRATUM reconciliation: the one-k-split
                            // bound is K_CHUNK (walk2_k_job(FEED_DM), 1984
                            // at D=64), not the raw MXE ceiling — a d_model
                            // in (K_CHUNK, K_MAX] would k-chunk in S2_PJOB
                            // while S2_PAE carries no act-row base cursor.
                            || (int'({16'b0, d2.d_model})
                                > int'(K_CHUNK))
                            // E-7b: DOWN's contraction is d_ffn — the same
                            // one-k-split rule at ITS k. At the 0.5B
                            // geometry (d_ffn = 4864 > K_CHUNK = 1984 @
                            // D=64) this clause is what refuses the walked
                            // DOWN, and the reason is MEASURED, not
                            // stylistic: 4864/64 = 76 stage rows cannot be
                            // resident in the 31-row bank
                            // (apex_stage_buf.sv:103-104), and a
                            // per-k-chunk re-staged family has no in-tile
                            // replay source (S2_PAE emits a HOST-staged
                            // resident family; nothing re-fills act bank 1
                            // mid-walk). Real datapath work — fenced.
                            || (fp_dwn
                                && ((d2.d_ffn == 16'b0)
                                    || ((d2.d_ffn & 16'(CFG_D - 1))
                                        != 16'b0)
                                    || (int'({16'b0, d2.d_ffn})
                                        > int'(K_CHUNK))))
                            // E-7b: the epilogue o8 frame bound covers the
                            // DOWN epilogue too (its frame is d_model
                            // elements, exactly OPROJ's).
                            || ((d2.en_mask[W2_EN_OPROJ] || fp_dwn)
                                && (int'({16'b0, d2.d_model}) > 2040))
                            // E-7: the fetched-gamma NORM's y-drain
                            // envelope — the drain frames d_model as
                            // n_heads C-1 rows (the NFEED argument:
                            // division-free only when FEED_DM == CFG_D),
                            // the feeder job takes at most FEED_ROWS rows,
                            // and the drained codes land at the fp_top
                            // window, which must fit the stage bank ABOVE
                            // every staged family. Two NORMs in one FGAM
                            // walk would land both drains at the SAME
                            // window — the second silently overwrites the
                            // first, a read-back ambiguity: refused.
                            || (fp_gam
                                && (!NF_ROWS_EXACT
                                    || (int'({24'b0, d2.n_heads})
                                        > int'(FEED_ROWS))
                                    || (int'({16'b0, fp_top})
                                        + int'({24'b0, d2.n_heads})
                                        > int'(STAGE_ROWS))
                                    || (d2.en_mask[W2_EN_NORM1]
                                        && d2.en_mask[W2_EN_NORM2])))))))
              e = WALK_ERR_DESC;
            if (e != WALK_ERR_NONE) begin
              state          <= S2_ERR;
              walk2_err_q    <= 1'b1;
              walk2_err_code_q <= e;
            end else begin
              pc          <= '0;
              ffn_chunk_q <= '0;           // FFN interleave: fresh loop
              ffn_win_q   <= 1'b0;
              // E-4b: claim (or decline) l_nsrc ownership for this walk,
              // and start the owned level from a deterministic 0 — the
              // codes->act default every pre-NFEED step wants. Ownership
              // NEVER engages on a refused walk (the refusal-leaves-state-
              // unchanged rule), which is why this lives on the pass arm.
              lw_nsrc_own <= d2.en_mask[W2_EN_NSRC];
              lw_nsrc     <= 1'b0;
              state <= S2_STEP;
            end
          end

          // ── dispatch: one cycle per skipped (disabled) pc ───────────────
          S2_STEP: begin
            if (pc >= PC_END) state <= S2_DONE;
            else if (!pc_en)  pc <= pc + 6'd1;
            else begin
              unique case (pc_kind)
                // E-6: fp_rq_q tracks the dispatched pc's own hasrq at both
                // FPROJ-window entries (every K_FETCH pc carries hasrq=0,
                // so this is a clear before the fetch hold).
                // E-7: a G1/G2 fetch of an FGAM walk opens the gamma
                // window (a weight fetch closes it — belt over the
                // S2_NORM-exit clear, and a legacy walk writes 0).
                K_FETCH: begin
                  fp_rq_q  <= pc_hasrq;
                  fp_gam_q <= fp_gam
                              && ((pc_tens == 4'(WALK2_TENS_G1))
                                  || (pc_tens == 4'(WALK2_TENS_G2)));
                  state    <= S2_FETCH;
                end
                K_PJOB: begin
                  k_rem <= pc_ktot; n_rem <= pc_ntot; pj_first <= 1'b1;
                  // E-5: a fuel-fed projection stages its activation family
                  // before the descriptor — the PROVEN E-3b order (S2_QSE's
                  // act EMIT precedes S2_QSD's descriptor). Legacy walks
                  // still go straight to S2_PJOB.
                  // E-6: an EPILOGUE projection (pc_hasrq) first frames its
                  // o8 stream for the serializer (S2_PSJ) — the consumer
                  // jobs (resid, deq) were already pushed by PC_URES1 /
                  // PC_UOPRJ in template order, so after this push the
                  // whole ser -> deq -> residual chain is armed before any
                  // result beat exists.
                  fp_row  <= '0;
                  fp_rq_q <= pc_hasrq;
                  // hasrq goes through the S2_PDW drain fence first (the
                  // attention-tail hazard note at the state's declaration).
                  // FFN interleave: gate/up jobs frame their raw-INT32
                  // stream for the serializer first (S2_PSJ pushes
                  // (pc_ntot+7)>>3 = 8 beats) — the swiglu consumes one
                  // such frame per phase; no rq drain is involved.
                  state   <= fproj_en
                           ? (pc_hasrq ? S2_PDW
                              : ((pc == PC_WGJ || pc == PC_WUJ) ? S2_PSJ
                                                                : S2_PAE))
                           : S2_PJOB;
                end
                // E-7: a fetched-gamma NORM arms its own output drain
                // first (declaration note) — the levels/route settle
                // state, then the act LOAD and feeder-job pushes.
                K_NORM:  state <= fp_gam_q ? S2_GDL : S2_NORM;
                K_LVL:   state <= S2_LVL;
                K_UJOB: begin
                  lu_rem <= pc_ucols;
                  // FFN interleave: the chunk window opens at the swiglu
                  // push and holds RT_FPRQ (rdst=1: MXE results -> the
                  // serializer frames feeding the swiglu) until the CS
                  // stage closes it — the swg holds BETWEEN phases by
                  // design, so no tile_idle fence fits inside the window.
                  if (pc == PC_USWI) ffn_win_q <= 1'b1;
                  state  <= (pc_ncomp != 2'd0) ? S2_JCA : S2_LU;
                end
                K_NFEED: state <= S2_NFL;
                K_QSTAGE: begin
                  h_idx <= '0;
                  state <= S2_QSL;
                end
                K_STORE: begin
                  sk_g   <= '0;
                  sk_rec <= 1'b0;
                  state  <= S2_SKS;
                end
                K_ATTN: begin
                  h_idx <= '0; g_idx <= '0; gnum <= '0;
                  state <= S2_HLOAD;
                end
                default: state <= S2_DONE;
              endcase
            end
          end

          S2_FETCH: if (wf_ready) begin
            pc    <= pc + 6'd1;
            state <= S2_STEP;
          end

          // E-6: the pre-epilogue drain fence (declaration note) — no
          // valid presented, so there is nothing D-020 could forbid.
          S2_PDW: if (tile_idle) state <= S2_PSJ;

          // E-6: the epilogue projection's serializer frame — registered-
          // accept like every job push here (valid held from `state`, ready
          // captured at the edge). Pushed before the first act EMIT.
          S2_PSJ: if (sj_ready) state <= S2_PAE;

          // ── E-5: the act-row family for ONE projection job ──────────────
          // FP_ROWS PAT_ROW emits from act bank 1, rows 0..FP_ROWS-1, each
          // FP_BPR lane8 beats — together the K = d_model contraction row
          // the descriptor ingests. Re-run for every n-job (the stage
          // buffer's documented replay use).
          //
          // THE ORDER IS NOT FREE, and it is not ours: the host-mode proof
          // pushes EMIT(row 0) -> DESCRIPTOR -> EMIT(rows 1..N-1)
          // (gemm_job.py:366-369, rationale at :360-365). Emitting the whole
          // family BEFORE the descriptor deadlocks — apex_stage_buf gates
          // job_ready until the previous EMIT's beats have been ACCEPTED
          // downstream (D-006, apex_stage_buf.sv header), and the MXE act
          // port accepts nothing until a descriptor is active. So the first
          // emit fills the skid and S2_PAE waits on an aj_ready that cannot
          // come. This split reproduces the proven order exactly.
          S2_PAE: if (aj_ready) begin
            if (fp_row == 5'd0) begin
              fp_row <= 5'd1;
              state  <= S2_PJOB;            // descriptor after the first row
            end else if (fp_last_row) begin
              state  <= S2_PAD;             // family complete -> job done
            end else begin
              fp_row <= fp_row + 5'd1;
            end
          end

          S2_PJOB: if (ds_ready) begin
            if (fproj_en) begin
              // FP_ROWS == 1 means row 0 WAS the whole family.
              state <= (int'({16'b0, fp_krows}) == 1) ? S2_PAD : S2_PAE;
            end else if (!last_k) begin
              k_rem    <= k_rem - 16'(K_CHUNK);
              pj_first <= 1'b0;
            end else if (!last_n) begin
              n_rem    <= n_rem - 16'(WALK2_N_MXE);
              k_rem    <= pc_ktot;
              pj_first <= 1'b1;
            end else begin
              pj_first <= 1'b1;
              if (pc == PC_WUJ
                  && ((32'(ffn_chunk_q) + 32'd1) << 6)
                     < 32'(d2.d_ffn)) begin
                ffn_chunk_q <= ffn_chunk_q + 16'd1;
                pc          <= PC_USWI;
              end else begin
                pc <= pc + 6'd1;
              end
              state    <= S2_STEP;
            end
          end

          // E-5: one fuel-fed job retired — advance the n cursor. The k
          // cursor never moves: S2_CHECK refuses d_model > K_CHUNK (the
          // third-erratum walk2_k_job(FEED_DM) bound), so an FPROJ
          // projection is always ONE k-split and `accumulate` stays 0 (a
          // k-chunked FPROJ would need an act-row BASE cursor + a
          // re-staged family per chunk, which this step does not carry —
          // refuse, never degrade; the walked DOWN re-staging is the named
          // follow-on).
          S2_PAD: begin
            if (!last_n) begin
              n_rem    <= n_rem - 16'(WALK2_N_MXE);
              k_rem    <= pc_ktot;
              pj_first <= 1'b1;
              fp_row   <= '0;
              state    <= S2_PAE;        // replay the family for job n+1
            end else begin
              pj_first <= 1'b1;
              // FFN interleave: NO tile_idle fence inside the chunk window
              // — the swiglu is busy BETWEEN phases by design. The gate
              // half falls straight through to the up fetch; the up half
              // goes straight to the product stage (whose arm is what
              // makes the tile drainable); everything else keeps the
              // proven S2_PJW drain.
              if (pc == PC_WGJ) begin
                // arm the product consumer BEFORE the up phase: the swg
                // emits products DURING up consumption, and the feeder's
                // whole-row rbuf absorbs them without the act LOAD (the
                // B-STAGE-PORT one-job-port rule: EMITs own the port
                // during the up jobs; the LOAD comes after).
                pc    <= pc + 6'd1;
                state <= S2_CSL;
              end else if (pc == PC_WUJ) begin
                state <= S2_CSA;
              end else begin
                pc    <= pc + 6'd1;
                // hold the RT_FPROJ route until the tile has drained (see
                // the rt_f1_q note) instead of reverting mid-egress
                state <= S2_PJW;
              end
            end
          end

          // E-5: post-projection drain. pc has already advanced, so this
          // state only holds the route; the next step dispatches normally.
          S2_PJW: if (tile_idle) state <= S2_STEP;

          // ── E-7: the y-drain arm states (fetched-gamma NORMs only —
          //    unreachable when fp_gam_q is low, so every legacy NORM
          //    walk dispatches straight to S2_NORM byte-identically) ────
          // S2_GDL touches the drain levels the host drain touches:
          // fsrc_ext back to the legacy widen source (the y path), and an
          // OWNED l_nsrc re-armed 0 (codes -> act stage, not the norm) —
          // a write of the reset value on unowned walks' trace (none:
          // unowned FGAM walks simply skip the write). One settle state,
          // the S2_NFL registered-copy argument; RT_YDRAIN rides gd_hold
          // from this state's edge.
          S2_GDL: begin
            lw_fsrc_ext <= WALK2_FSRC_LEGACY;
            if (d2.en_mask[W2_EN_NSRC]) lw_nsrc <= 1'b0;
            state       <= S2_GDA;
          end
          S2_GDA: if (aj_ready) state <= S2_GDF;
          S2_GDF: if (fj_ready) state <= S2_NORM;

          S2_NORM: if (lj_ready) begin
            // E-7: a fetched-gamma NORM holds the drain route through
            // S2_GDW until the widen/feeder/act tail has landed; a legacy
            // NORM advances exactly as before.
            if (fp_gam_q) state <= S2_GDW;
            else begin
              pc    <= pc + 6'd1;
              state <= S2_STEP;
            end
          end

          // E-7: the y-drain tail wait — tile_idle covers the feeder, the
          // stage buffers AND the gamma unpacker (gu_busy in blk_busy[2]),
          // so the window closes with nothing in flight anywhere.
          S2_GDW: if (tile_idle) begin
            fp_gam_q <= 1'b0;
            pc       <= pc + 6'd1;
            state    <= S2_STEP;
          end

          // ── FFN interleave: the per-chunk product stage ────────────────
          // CSL/CSF run BETWEEN the gate and up phases (arm the consumer);
          // CSA/CSW after the up jobs (LOAD the completed product row).
          S2_CSL: begin
            lw_fsrc_ext <= WALK2_FSRC_SWGP;    // feeder eats the product
            state       <= S2_CSF;
          end
          S2_CSF: if (fj_ready) state <= S2_STEP;   // -> PC_WUF dispatch
          S2_CSA: if (aj_ready) state <= S2_CSW;
          S2_CSW: if (tile_idle) begin
            lw_fsrc_ext <= WALK2_FSRC_LEGACY;  // touch-one-field restore
            ffn_win_q   <= 1'b0;               // the chunk window closes
            if (((32'(ffn_chunk_q) + 32'd1) << 6) < 32'(d2.d_ffn)) begin
              ffn_chunk_q <= ffn_chunk_q + 16'd1;
              pc          <= PC_USWI;          // next chunk (tail-aware)
            end else begin
              pc <= pc + 6'd1;                 // all chunks staged -> DOWN
            end
            state <= S2_STEP;
          end

          // ── E-3 NFEED: arm the route, frame the row, push the egress,
          //    then hold until the WHOLE feed path is quiet ───────────────
          // S2_NFL touches ONE level field. Every other l_* holds, so this
          // step composes with the surrounding LVL steps instead of
          // re-arming them (PC_LVLG re-arms fsrc_ext absolutely right after,
          // which is what returns the mux to legacy).
          // E-4b: an OWNED walk raises l_nsrc on the same edge — the norm
          // x-mux flips WITH the code-4 route, one settle state before the
          // pushes (apex_top's registered walk-mode copy lags one cycle;
          // S2_NFF absorbs it, exactly the fsrc_ext argument above). An
          // unowned walk writes nothing: the host's pre-GO arm HOLDS, the
          // landed E-3 semantics byte-for-byte.
          S2_NFL: begin
            lw_fsrc_ext <= WALK2_FSRC_RESID;
            if (d2.en_mask[W2_EN_NSRC]) lw_nsrc <= 1'b1;
            state       <= S2_NFF;
          end
          // The feeder job also gives apex_top's registered walk-mode level
          // copy its settle cycle: l_fsrc_ext_q reaches code 4 during THIS
          // state, so the unit-3 push in S2_NFU can never race the route-arm
          // gate (which would refuse with err_code 9 and wedge S2_NFW).
          S2_NFF: if (fj_ready) state <= S2_NFU;
          S2_NFU: if (lu_ready) state <= S2_NFW;
          // nf_busy is high from the accepted feeder job onward (fq_busy is
          // one of its terms while code 4 is armed) and the unit-3 push has
          // been accepted by now, so this samples a genuinely busy path and
          // falls only when the last element has reached the norm — the
          // host demonstrator's LAYER_STATUS[5] poll, in the walker.
          S2_NFW: if (!nf_busy) begin
            pc    <= pc + 6'd1;
            state <= S2_STEP;
          end

          // ── E-3b QSTAGE: per head, the walker performs the HOST's whole
          //    q-staging choreography (build_rope_stage + inject_jobs order,
          //    verbatim): LVL {rope-q + q_sink} once, then per head FJOB,
          //    MODE_F16 QJOB, the composite sideband xCFG_D, the serializer
          //    job, CFG_D/8 x {loader-row act EMIT, k2 WS descriptor}, the
          //    act-bank-1 LOAD at row h, and a whole-path drain wait. The
          //    level flip waits on tile_idle (rule (d)); so does the exit.
          S2_QSL: if (tile_idle) begin
            lw_rope_en   <= 1'b1;
            lw_rope_bank <= 1'b1;             // q phase table
            // (cast form differs from S2_LVL's on purpose: mutate.py's
            //  m11_ropepos anchors the S2_LVL line by FIRST occurrence)
            lw_rope_pos  <= 7'(d2.pos_m);
            lw_ser_dst   <= 2'd0;             // serializer -> squant legacy
            lw_resid_arm <= 1'b0;
            lw_fsrc_ext  <= 3'd3;             // the gap-A q sink
            // E-4b: absolute re-arm of the owned level to its staging value
            // (codes -> act). A write of the reset value: unowned walks and
            // legacy traces are byte-identical.
            lw_nsrc      <= 1'b0;
            state        <= S2_QSF;
          end
          S2_QSF: if (fj_ready) begin
            qs_cnt <= '0;
            state  <= S2_QSQ;
          end
          S2_QSQ: if (qj_ready) state <= S2_QSC;
          S2_QSC: if (qs_ready) begin
            if (int'({24'b0, qs_cnt}) == int'(CFG_D) - 1) begin
              grp_cnt <= '0;
              state   <= S2_QSS;
            end else qs_cnt <= qs_cnt + 8'd1;
          end
          S2_QSS: if (sj_ready) state <= S2_QSE;
          S2_QSE: if (aj_ready) state <= S2_QSD;
          S2_QSD: if (ds_ready) begin
            if (int'({27'b0, grp_cnt}) == int'(QS_BPR) - 1) state <= S2_QSA;
            else begin
              grp_cnt <= grp_cnt + 5'd1;
              state   <= S2_QSE;
            end
          end
          S2_QSA: if (aj_ready) state <= S2_QSW;
          S2_QSW: if (tile_idle) begin
            if (qs_last_head) begin
              // restore: touch ONE field (the S2_NFL idiom, in reverse) —
              // the seam returns to the KV write port before any attention
              // or store traffic can flow. rt_f1_q loads RT_SCORE on this
              // same edge (the override block above).
              lw_fsrc_ext <= 3'd0;
              pc          <= pc + 6'd1;
              state       <= S2_STEP;
            end else begin
              h_idx <= h_idx + 8'd1;
              state <= S2_QSF;
            end
          end

          // level settle: registers written, one cycle, no handshake
          S2_LVL: begin
            lw_rope_en   <= 1'b1;              // armed from ROPE onward
            lw_rope_bank <= 1'b0;              // K-table (resident) source
            lw_rope_pos  <= d2.pos_m[6:0];
            // E-4b: every LVL step re-arms the owned l_nsrc 0 ABSOLUTELY
            // (the "re-arms each field absolutely" doctrine) — in a full
            // walk PC_LVLG is what returns the norm x-mux to xa after the
            // NFEED arm, exactly as it returns fsrc_ext to legacy. A write
            // of the reset value: byte-identical when unowned.
            lw_nsrc      <= 1'b0;
            unique case (pc_lvl)
              // E-3: the source codes are named constants now that the
              // field is 3 bits wide; the VALUES are byte-identical to the
              // 2-bit literals these lines used to carry.
              2'd0: begin
                lw_ser_dst <= 2'd0; lw_fsrc_ext <= WALK2_FSRC_LEGACY;
                lw_resid_arm <= 1'b0;
              end
              2'd1: begin
                lw_ser_dst <= 2'd1; lw_fsrc_ext <= WALK2_FSRC_LEGACY;
                lw_resid_arm <= 1'b1;
              end
              2'd2: begin
                lw_ser_dst <= 2'd2; lw_fsrc_ext <= WALK2_FSRC_LEGACY;
                lw_resid_arm <= 1'b1;
              end
              default: begin
                lw_ser_dst <= 2'd1; lw_fsrc_ext <= WALK2_FSRC_SWGP;
                lw_resid_arm <= 1'b1;
              end
            endcase
            pc    <= pc + 6'd1;
            state <= S2_STEP;
          end

          // JOBC then push, composites REWRITTEN per chunk (§3b reset rule)
          S2_JCA: if (jc_ready) state <= (pc_ncomp == 2'd2) ? S2_JCB : S2_LU;
          S2_JCB: if (jc_ready) state <= S2_LU;
          S2_LU:  if (lu_ready) begin
            if (!lu_last) begin
              lu_rem <= lu_rem - lu_bound;
              state  <= (pc_ncomp != 2'd0) ? S2_JCA : S2_LU;
            end else begin
              pc    <= pc + 6'd1;
              state <= S2_STEP;
            end
          end

          // ── STOREKV: per engine, K then V WRITE_ADDR (idle-polled) ──────
          // one dead cycle so the registered tile-level engine select has
          // settled before the poll's AR can fire (the S2_SKS enum note)
          S2_SKS:   state <= S2_SKPAR;
          S2_SKPAR: if (kvm_arready) state <= S2_SKPR;
          S2_SKPR:  if (kvm_rvalid) begin
            if (kvm_rdata[1]) begin
              state          <= S2_ERR;
              walk2_err_q    <= 1'b1;
              walk2_err_code_q <= WALK_ERR_SEQ;
            end else if (kvm_rdata[0]) state <= S2_SKWR;
            else                       state <= S2_SKS;
          end
          S2_SKWR: if (kvm_awready && kvm_wready) state <= S2_SKB;
          S2_SKB:  if (kvm_bvalid) begin
            if (!sk_rec) begin
              sk_rec <= 1'b1;
              state  <= S2_SKS;
            end else if (int'({{(32-ENG_W){1'b0}}, sk_g})
                         < int'({24'b0, d2.n_kv_heads}) - 1) begin
              sk_g   <= sk_g + ENG_W'(1);
              sk_rec <= 1'b0;
              state  <= S2_SKS;
            end else begin
              pc    <= pc + 6'd1;
              state <= S2_STEP;
            end
          end

          // ── per-head v1 walk (engine descriptor synthesized from fmt=1) ─
          S2_HLOAD: if (hd_ready) begin
            h_desc[WALK_DW_GEOM] <= {8'b0, 4'b0, 2'b00, 2'b00,
                                     d2.t_rows, 8'(CFG_D)};
            h_desc[WALK_DW_RQ]   <= desc2_word[W2_RQ0 + {24'b0, h_idx}];
            h_desc[WALK_DW_MASK] <= {30'b0, d2.en_mask[W2_EN_PV],
                                     d2.en_mask[W2_EN_SCORE]};
            h_en  <= 1'b1;
            state <= S2_HGO;
          end
          S2_HGO: begin
            h_go  <= 1'b1;
            state <= S2_HWAIT;
          end
          S2_HWAIT: begin
            if (h_err) begin
              state          <= S2_ERR;
              h_en           <= 1'b0;
              walk2_err_q    <= 1'b1;
              walk2_err_code_q <= h_err_code;
            end else if (!h_busy && !h_go) begin
              if (int'({24'b0, h_idx}) == int'({24'b0, d2.n_heads}) - 1) begin
                h_en  <= 1'b0;
                pc    <= pc + 6'd1;
                state <= S2_STEP;
              end else begin
                // GOLDEN GQA mapping, division-free: g(h) = floor(h*nk/nh)
                // via the running remainder gnum (§9.1 R3 as amended)
                h_idx <= h_idx + 8'd1;
                if (gnum + 16'({8'b0, d2.n_kv_heads})
                    >= 16'({8'b0, d2.n_heads})) begin
                  gnum  <= gnum + 16'({8'b0, d2.n_kv_heads})
                                - 16'({8'b0, d2.n_heads});
                  g_idx <= g_idx + ENG_W'(1);
                end else begin
                  gnum <= gnum + 16'({8'b0, d2.n_kv_heads});
                end
                state <= S2_HLOAD;
              end
            end
          end

          // levels are NOT auto-reset at walk end: they are LEVELS (the §3b
          // l_*_q registers), and every walk re-arms each field absolutely
          // in its LVL steps — a tail-state reset here would surface as a
          // spurious level change with no host-side counterpart.
          // E-4b: OWNERSHIP does end here — lw_nsrc_own is a walk-scoped
          // claim on apex_top's mux, not a level. l_nsrc_q then HOLDS the
          // walker's last-armed value (the levels-hold doctrine) until the
          // host, back in host mode, rewrites it.
          S2_DONE: begin
            state <= S2_IDLE; lw_nsrc_own <= 1'b0; fp_gam_q <= 1'b0;
          end
          S2_ERR:  begin
            state <= S2_IDLE; lw_nsrc_own <= 1'b0; fp_gam_q <= 1'b0;
          end
          default: state <= S2_IDLE;
        endcase
      end
    end
  end

`ifdef APEX_E6_DBG
  // DEBUG-ONLY (never in a shipped twin): fmt=1 state-transition trace
  s2_e dbg_q;
  logic dbg_hb_q;
  int unsigned dbg_cyc;   // tile-clock cycles since reset (trace correlation)
  always_ff @(posedge clk) begin
    dbg_q <= state;
    dbg_hb_q <= h_busy;
    dbg_cyc <= dbg_cyc + 1;
    if (state != dbg_q)
      $display("[W2DBG] st %0d->%0d pc=%0d krem=%0d nrem=%0d fprow=%0d tcyc=%0d",
               dbg_q, state, pc, k_rem, n_rem, fp_row, dbg_cyc);
    if (h_go)
      $display("[W2DBG] HGO desc0=%08x rq=%08x mask=%08x",
               h_desc[WALK_DW_GEOM], h_desc[WALK_DW_RQ],
               h_desc[WALK_DW_MASK]);
    if (h_busy != dbg_hb_q)
      $display("[W2DBG] h_busy=%b phase=%0d err=%b code=%0d",
               h_busy, h_phase, h_err, h_err_code);
    if (ds_valid && ds_ready)
      $display("[W2DBG] DS op=%0d m=%0d k=%0d n=%0d rq=%b",
               ds_desc.opcode, ds_desc.m_dim, ds_desc.k_dim,
               ds_desc.n_dim, ds_desc.requant_en);
  end
`endif

  // deliberately-unread d2 fields: legality is checked on the RAW words
  // (walk_desc2_check); outlier_k is informational
  logic unused_ok;
  assign unused_ok = &{1'b0, d2.pos_m[7], d2.outlier_k, d2.tier, d2.fmt,
                       d2.head_dim, rq_word[31:21], k_cur[15:12],
                       n_cur[15:12], lu_cur[15:12], kvm_rdata[31:2]};

endmodule

`endif // SEQ_LAYER_WALKER2_SV
