// seq_layer_walker.sv — B1: autonomous layer-walker, score+pv @ CQ-8.
//
// Implements: docs/design/B1_WALKER.md §1 (emit-mode walker), §3 (layer
//             descriptor + WALK CSR window), §4 Acceptance A (bit-exact and
//             strictly in-order vs the extracted trace), §A-1 (CQ-8-only
//             refusal gate), §A-2 (scale source), §5 stream semantics,
//             D-006 (SEQ serializes jobs), D-020 (soft-reset abort/drain).
// Spec:       verif/top/l3/gen_l3_vectors.py score_phase(:360-401) and
//             pv_phase(:404-431). The op ROM is transcribed from
//             docs/design/B1_STAGE1_NOTES.md §1 (every argument as a formula
//             in (D, T, BPR, ci, t, j); DESC encodings verified against all
//             24 CQ-8 cases).
//
// ── WHY A SIBLING MODULE, NOT AN EXTENSION OF seq_walker.sv ─────────────────
// This block drives ds_* exactly as the host does today, so rtl/seq/
// seq_walker.sv stays BIT-IDENTICAL: its verified D-006 FSM keeps enforcing
// "descriptor N+1 is never presented before job N's done" for the walker too,
// and host-mode regression risk is zero by construction. The walker never
// bypasses SEQ.
//
// ── WHAT IT DOES NOT WALK ───────────────────────────────────────────────────
// phase_a / loader / store_kv / q-inject / final stay host-driven. Those are
// dominated by decompose_f16 injection composites — testbench data-path
// scaffolding (87.5% of the L3 stream), not decode control; in a real decode
// K/V arrive from the projection GEMMs. The walker's target is score+pv, which
// IS the per-token attention control stream. See B1_WALKER.md §2.
//
// ── ORDERING CONSTRAINTS THAT ARE DEADLOCKS, NOT STYLE (notes §4) ───────────
// (a) pv's c8 emission is EMIT -> DESC -> EMIT, never EMIT -> EMIT -> DESC.
//     apex_stage_buf.sv:216 gates job_ready on the previous job draining, and
//     mxe_ctrl.sv:189 only accepts activation beats in S_INGEST (entered on
//     descriptor acceptance). So AJ,AJ,DESC wedges: job 1 cannot drain because
//     MXE is idle, so job 2 is never accepted, so the DESC that would unblock
//     it is never emitted. Only exercised when split (nbT > BPR, i.e. T > D) —
//     3 of the 24 CQ-8 cases. A naive walker passes 21/24 and looks green.
// (b) Score is WJ-EMIT -> AJ-EMIT -> DESC (different stage_buf instances, so
//     neither blocks the other). Do NOT normalise (a) and (b) into one
//     template — that breaks pv.
// (c) FJOB must follow its WJ-LOAD and precede the reads.
// (d) Change a route level only when the touched paths are idle (the phase-F
//     wedge fix of 2026-07-09, gen_l3_vectors.py:224-226).
// (e) Walker v1 emits exactly TWO route levels across both phases; the
//     poll-free mid-phase re-routes are blkmap-only and blkmap cases are
//     grouped-tier, hence refused.
// (f) fq_out_ready footgun (apex_top.sv:771-772): rt_feeder_dst=0 with
//     rt_act_src=1 hard-stalls the feeder with NO error. Both ROM routes set
//     fdst=1; a route word that drops bit 1 while bit 2 is set silently wedges.
`ifndef SEQ_LAYER_WALKER_SV
`define SEQ_LAYER_WALKER_SV

module seq_layer_walker
  import apex_pkg::*;
  import seq_walker_pkg::*;
#(
  parameter int unsigned CFG_D = 64,
  // F-5a: derived width, never a magic number — must match apex_top.sv:132
  // exactly, or the stage-buffer nb field silently truncates.
  localparam int unsigned STAGE_NB_W = $clog2(CFG_D / 8) + 1
) (
  input  logic                  clk,
  input  logic                  rst_n,        // synchronous, active low

  // ── control (WALK CSR window, decoded in apex_top glue) ──────────────────
  input  logic                  walk_en,      // WALK_CTRL[0]
  input  logic                  walk_go,      // WALK_CTRL[1], 1-cycle kick
  input  logic                  abort_req,    // CTRL.soft_reset (D-020)
  input  logic [31:0]           desc_word [WALK_DESC_WORDS],  // descriptor SRAM
                                              // (the walker owns the word map,
                                              // so the glue stays layout-blind)
  input  logic                  tile_idle,    // ~|block_busy (STATUS.idle)

  // ── I-C IC-QPATH, F6(ii): which act-stage row holds THIS head's q8 ───────
  // The score phase re-emits the staged q8 row (S11, W_S_AJEM). On a
  // multi-head walk each head has its OWN q row, pre-staged into act bank 1
  // rows 0..H-1 through the gap-A sink, so the emission must select row h.
  // fmt=0 / v1-forwarded walks tie this to 0 and are bit-identical.
  // DEFAULTED so the port is purely ADDITIVE: every existing instantiation
  // (verif/seq_walker's TBs among them) may leave it unconnected and keeps
  // the pre-I-C behaviour — re-emit act row 0 — with no edit and no waiver.
  input  logic [4:0]            q_row_sel = 5'd0,

  // ── status (WALK_STATUS) ─────────────────────────────────────────────────
  output logic                  walk_busy,
  output walk_phase_e           walk_phase,
  output logic                  walk_err,     // 1-cycle pulse; glue makes it sticky
  output walk_err_e             walk_err_code,

  // ── SEQ descriptor stream out (drives seq_walker's ds_* in walker mode) ──
  output logic                  ds_valid,
  input  logic                  ds_ready,
  output mxe_desc_t             ds_desc,

  // ── route levels ─────────────────────────────────────────────────────────
  output logic                  rt_feeder_src,
  output logic                  rt_feeder_dst,
  output logic                  rt_act_src,
  output logic                  rt_wgt_src,
  output logic [1:0]            rt_res_dst,
  output logic                  rt_squant_src,
  output logic                  rt_kv_user,

  // ── glue job commands ────────────────────────────────────────────────────
  output logic                  fj_valid,     // feeder
  input  logic                  fj_ready,
  output logic [DIM_W-1:0]      fj_rows,
  output logic                  qj_valid,     // scale_quant
  input  logic                  qj_ready,
  output logic                  qj_mode,
  output logic [DIM_W-1:0]      qj_cols,
  output logic                  dj_valid,     // score dequant
  input  logic                  dj_ready,
  output logic [DIM_W-1:0]      dj_cols,
  output logic                  aj_valid,     // act stage buffer
  input  logic                  aj_ready,
  output logic                  aj_op,
  output logic                  aj_bank,
  output logic [1:0]            aj_pat,
  output logic [4:0]            aj_rows,
  output logic [STAGE_NB_W-1:0] aj_nb,
  output logic [4:0]            aj_sel,
  output logic                  wj_valid,     // wgt stage buffer
  input  logic                  wj_ready,
  output logic                  wj_op,
  output logic                  wj_bank,
  output logic [1:0]            wj_pat,
  output logic [4:0]            wj_rows,
  output logic [STAGE_NB_W-1:0] wj_nb,
  output logic [4:0]            wj_sel,

  // ── seam composites (sourced from seq_walker_comp) ───────────────────────
  output logic                  qs_valid,
  input  logic                  qs_ready,
  output logic [31:0]           qs_data,
  output logic                  cs_valid,
  input  logic                  cs_ready,
  output logic [31:0]           cs_data,

  // ── composite-unit request port ──────────────────────────────────────────
  output logic                  comp_req_valid,
  input  logic                  comp_req_ready,
  output logic                  comp_req_is_qs,
  output logic [WALK_SC_AW-1:0] comp_req_idx,
  input  logic                  comp_res_valid,
  output logic                  comp_res_ready,
  input  logic [31:0]           comp_res_data,

  // ── KVQ AXI-Lite master (muxed onto the bank in walker mode) ─────────────
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

  localparam int unsigned BPR       = CFG_D / 8;
  localparam int unsigned NCH_MAX   = (WALK_T_MAX + 7) / 8;
  localparam int unsigned CH_W      = $clog2(NCH_MAX + 1);
  localparam int unsigned BJ_W      = $clog2(WALK_D_MAX / 8 + 1);
  localparam logic [7:0]  KV_STATUS = 8'h04;
  localparam logic [7:0]  KV_RADDR  = 8'h2C;
  localparam logic [1:0]  PAT_ROW   = 2'd0;
  localparam logic [1:0]  PAT_T     = 2'd1;
  localparam logic [1:0]  PAT_D     = 2'd2;

  if (!(CFG_D == 64 || CFG_D == 128)) begin : g_chk_d
    $error("seq_layer_walker: CFG_D must be 64 or 128 (D-021)");
  end

  // ── walk state ───────────────────────────────────────────────────────────
  typedef enum logic [4:0] {
    W_IDLE     = 5'd0,  W_CHECK    = 5'd1,
    // score
    W_S_FENCE  = 5'd2,  W_S_SJOB   = 5'd3,
    W_S_CSREQ  = 5'd4,  W_S_CSRES  = 5'd5,  W_S_CSEMIT = 5'd6,
    W_S_QJOB   = 5'd7,
    W_S_QSREQ  = 5'd8,  W_S_QSRES  = 5'd9,  W_S_QSEMIT = 5'd10,
    W_S_WJLD   = 5'd11, W_S_FJOB   = 5'd12, W_S_KV     = 5'd13,
    W_S_WJEM   = 5'd14, W_S_AJEM   = 5'd15, W_S_DESC   = 5'd16,
    W_S_C8A    = 5'd17, W_S_C8B    = 5'd18,
    // pv
    W_P_FENCE  = 5'd19, W_P_AJHD   = 5'd20, W_P_DESC   = 5'd21,
    W_P_AJTL   = 5'd22, W_P_WJLD   = 5'd23, W_P_FJOB   = 5'd24,
    W_P_KV     = 5'd25, W_P_WJEM   = 5'd26,
    W_DONE     = 5'd27, W_ERR      = 5'd28, W_P_RT     = 5'd29,
    W_DRAIN    = 5'd30
  } wstate_e;
  wstate_e state;

  walk_desc_t          desc_q;
  logic [WALK_T_W-1:0] t_rows;                 // T
  logic [CH_W-1:0]     nch;                    // ceil(T/8)
  logic [WALK_T_W-1:0] nbt;                    // ceil(T/8) as a beat count
  logic                split;                  // nbT > BPR
  logic [CH_W-1:0]     ci;                     // chunk index
  logic [WALK_T_W-1:0] tk;                     // token index (absolute)
  logic [BJ_W-1:0]     bj;                     // pv output block
  logic [WALK_T_W-1:0] cw;                     // composite word index
  logic [4:0]          nc;                     // tokens in the current chunk
  logic [WALK_T_W-1:0] c0;                     // first token of the chunk
  logic [31:0]         comp_q;                 // latched composite word
  logic                abort_l;                // latched soft reset

  // geometry of the current chunk: nc = min(8, T - 8*ci)
  logic [WALK_T_W-1:0] rem;
  assign rem = t_rows - c0;
  assign nc  = (rem > WALK_T_W'(8)) ? 5'd8 : rem[4:0];

  // ── KVQ AXI-Lite master sub-FSM ──────────────────────────────────────────
  // Invariant (notes §4): never issue an RADDR write unless a STATUS read has
  // returned bit0==1 — the engine silently DROPS read_req outside ST_IDLE, so
  // this is correctness, not throughput hygiene.
  typedef enum logic [2:0] {
    K_IDLE = 3'd0, K_PAR = 3'd1, K_PR = 3'd2,
    K_WR   = 3'd3, K_B   = 3'd4, K_DONE = 3'd5
  } kstate_e;
  kstate_e                 kst;
  logic                    kv_start;
  logic [WALK_SC_AW-1:0]   kv_addr;
  logic                    kv_done;
  logic                    kv_rderr;

  assign kvm_bready  = 1'b1;                   // held, never retracted (AXI)
  assign kvm_rready  = 1'b1;
  assign kvm_arvalid = (kst == K_PAR);
  assign kvm_araddr  = KV_STATUS;
  assign kvm_awvalid = (kst == K_WR);
  assign kvm_wvalid  = (kst == K_WR);
  assign kvm_awaddr  = KV_RADDR;
  assign kvm_wdata   = {{(32-WALK_SC_AW){1'b0}}, kv_addr};
  assign kv_done     = (kst == K_DONE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      kst      <= K_IDLE;
      kv_rderr <= 1'b0;
    end else begin
      kv_rderr <= 1'b0;
      unique case (kst)
        K_IDLE: if (kv_start) kst <= K_PAR;
        K_PAR:  if (kvm_arready) kst <= K_PR;
        K_PR:   if (kvm_rvalid) begin
                  if (kvm_rdata[1]) begin      // RD_ERR: a template address
                    kv_rderr <= 1'b1;          // cannot legally be unwritten
                    kst      <= K_IDLE;
                  end else if (kvm_rdata[0]) kst <= K_WR;
                  else                        kst <= K_PAR;
                end
        K_WR:   if (kvm_awready && kvm_wready) kst <= K_B;
        K_B:    if (kvm_bvalid) kst <= K_DONE;
        K_DONE: kst <= K_IDLE;
        default: kst <= K_IDLE;
      endcase
    end
  end

  // ── route levels (two constants; notes §4e) ──────────────────────────────
  // score 0x00EF: fsrc=1 fdst=1 asrc=1 wsrc=1 rdst=2 qsrc=1 kvu=1
  // pv    0x00CF: as score but rdst=0 (MXE result -> ro out)
  // rt_* are LEVELS, not handshakes: the pv level must be REGISTERED, not
  // decoded from the state. Decoding it made the level snap back to the score
  // value as the FSM passed through W_DONE, emitting a spurious third ROUTE
  // (caught by the stage-2 scoreboard on its first run). It is SET when the
  // pv fence clears and CLEARED at walk start — but the clear, being a route
  // level change, is subject to constraint (d) like any other.
  //
  // ── I-C IC-MHWALK: the walk-start clear IS a (d) event (F6(ii) residue) ───
  // It was cleared unconditionally in W_CHECK, one cycle after the kick. The
  // SINGLE-head flow could never expose that: nothing follows a walk, the
  // tile is always idle at the kick, and the level sat at pv while the MXE
  // drained the final PV block (levels are deliberately NOT reset at walk
  // end). A MULTI-HEAD fmt=1 walk puts a walk START immediately after a walk
  // END — walker2's S2_HWAIT advances on `!h_busy`, which is the WALKER's
  // done, not the datapath's — so W_CHECK flipped rt_res_dst 0 -> 2 while the
  // previous head's LAST PV result beat was still in the MXE, and apex_top's
  // result demux (apex_top.sv:1856-1859) is combinational in that level.
  // Measured on the 4-head case (nh4_g4_hd128_T20): RES_DST 0->2 at cyc
  // 312936 with tile_idle=0, and at 312959 that head's 16th PV beat
  // (lane 0 = 0x18) was accepted by the SCORE DEQUANT. Head h therefore lost
  // its last ERO beat and head h+1's score stream was prepended with 8 int32
  // words, shifting every one of its T score beats by 8 (the tile's recovered
  // acc[8..19] == golden acc_s[0..11], exactly).
  //
  // The fix keeps the W_CHECK clear for the idle case — so every pre-existing
  // walk is cycle-for-cycle and op-for-op what it was — and detours through
  // W_DRAIN when the datapath is still busy. W_DRAIN is WPH_IDLE, so the
  // level change still lands on the boundary INTO the score phase and an
  // in-phase route observer sees the same two ROUTE levels per head as
  // before (a third would appear if the fence were inside the score phase —
  // measured: +27 on the 28-head W-G3 topo scoreboard).
  logic pv_route;
  logic in_pv;
  assign in_pv         = pv_route;
  assign rt_feeder_src = 1'b1;
  assign rt_feeder_dst = 1'b1;                 // (f): never drop with asrc=1
  assign rt_act_src    = 1'b1;
  assign rt_wgt_src    = 1'b1;
  assign rt_squant_src = 1'b1;
  assign rt_kv_user    = 1'b1;
  assign rt_res_dst    = in_pv ? 2'd0 : 2'd2;

  // ── job/emit outputs ─────────────────────────────────────────────────────
  assign dj_valid = (state == W_S_SJOB);
  assign dj_cols  = {{(DIM_W-WALK_T_W){1'b0}}, t_rows};

  assign qj_valid = (state == W_S_QJOB);
  assign qj_mode  = 1'b1;                      // S-4 P-requant fold
  assign qj_cols  = {{(DIM_W-WALK_T_W){1'b0}}, t_rows};

  assign fj_valid = (state == W_S_FJOB) || (state == W_P_FJOB);
  assign fj_rows  = {{(DIM_W-5){1'b0}}, nc};

  assign cs_valid = (state == W_S_CSEMIT);
  assign cs_data  = comp_q;
  assign qs_valid = (state == W_S_QSEMIT);
  assign qs_data  = comp_q;

  assign comp_req_valid = (state == W_S_CSREQ) || (state == W_S_QSREQ);
  assign comp_req_is_qs = (state == W_S_QSREQ);
  // cs[t] reads the K record scale at index t; qs[t] the V record at T+t.
  assign comp_req_idx   = (state == W_S_QSREQ)
                        ? WALK_SC_AW'({1'b0, t_rows} + {1'b0, cw})
                        : WALK_SC_AW'({1'b0, cw});
  assign comp_res_ready = (state == W_S_CSRES) || (state == W_S_QSRES);

  // act-stage jobs: score emits the q8 row per chunk and the c8 LOAD pair at
  // the end; pv emits the c8 row (head + optional tail) per output block.
  always_comb begin
    aj_valid = 1'b0;
    aj_op    = 1'b0;
    aj_bank  = 1'b0;
    aj_pat   = PAT_ROW;
    aj_rows  = 5'd1;
    aj_nb    = STAGE_NB_W'(BPR);
    aj_sel   = 5'd0;
    unique case (state)
      W_S_AJEM: begin                          // S11: re-emit q8 from bank1
        aj_valid = 1'b1; aj_op = 1'b1; aj_bank = 1'b1;
        aj_sel   = q_row_sel;                  // F6(ii): this head's q row
      end
      W_S_C8A: begin                           // S15: LOAD c8 head -> bank1
        aj_valid = 1'b1; aj_op = 1'b0; aj_bank = 1'b1;
        aj_nb    = split ? STAGE_NB_W'(BPR) : STAGE_NB_W'(nbt);
      end
      W_S_C8B: begin                           // S16: LOAD c8 tail -> bank0
        aj_valid = 1'b1; aj_op = 1'b0; aj_bank = 1'b0;
        aj_nb    = STAGE_NB_W'(nbt - WALK_T_W'(BPR));
      end
      W_P_AJHD: begin                          // P2: EMIT c8 head from bank1
        aj_valid = 1'b1; aj_op = 1'b1; aj_bank = 1'b1;
        aj_nb    = split ? STAGE_NB_W'(BPR) : STAGE_NB_W'(nbt);
      end
      W_P_AJTL: begin                          // P4: EMIT c8 tail from bank0
        aj_valid = 1'b1; aj_op = 1'b1; aj_bank = 1'b0;
        aj_nb    = STAGE_NB_W'(nbt - WALK_T_W'(BPR));
      end
      default: ;
    endcase
  end

  // wgt-stage jobs: LOAD the k8/v8 chunk, then EMIT it (PAT_T in score,
  // PAT_D block j in pv — nb=1 and sel=j there, NOT BPR/0).
  always_comb begin
    wj_valid = 1'b0;
    wj_op    = 1'b0;
    wj_bank  = 1'b0;
    wj_pat   = PAT_ROW;
    wj_rows  = nc;
    wj_nb    = STAGE_NB_W'(BPR);
    wj_sel   = 5'd0;
    unique case (state)
      W_S_WJLD, W_P_WJLD: begin
        wj_valid = 1'b1; wj_op = 1'b0; wj_pat = PAT_ROW;
      end
      W_S_WJEM: begin
        wj_valid = 1'b1; wj_op = 1'b1; wj_pat = PAT_T;
      end
      W_P_WJEM: begin
        wj_valid = 1'b1; wj_op = 1'b1; wj_pat = PAT_D;
        wj_nb    = STAGE_NB_W'(1);
        wj_sel   = 5'(bj);
      end
      default: ;
    endcase
  end

  // ── descriptors ──────────────────────────────────────────────────────────
  // score: OP_GEMM_OS, m=1, k=D, n=nc, mode_os=1
  // pv:    OP_GEMM_OS, m=1, k=T, n=8,  mode_os=1, rq_en=1, rq_* from the
  //        LOADED descriptor (causally circular on-tile, B1_WALKER.md §2).
  assign ds_valid = (state == W_S_DESC) || (state == W_P_DESC);
  always_comb begin
    ds_desc            = '0;
    ds_desc.opcode     = OP_GEMM_OS;
    ds_desc.m_dim      = DIM_W'(1);
    ds_desc.mode_os    = 1'b1;
    if (state == W_P_DESC) begin
      ds_desc.k_dim      = {{(DIM_W-WALK_T_W){1'b0}}, t_rows};
      ds_desc.n_dim      = DIM_W'(MXE_N);
      ds_desc.requant_en = 1'b1;
      ds_desc.rq_scale   = desc_q.rq_scale;
      ds_desc.rq_shift   = desc_q.rq_shift;
    end else begin
      ds_desc.k_dim      = DIM_W'(CFG_D);
      ds_desc.n_dim      = {{(DIM_W-5){1'b0}}, nc};
    end
  end

  assign walk_busy  = (state != W_IDLE);
  // No phase is active during the legality check or after a refusal — saying
  // "SCORE" there would let a refused walk look like a started one in
  // WALK_STATUS, which is exactly the ambiguity §A-1 forbids.
  // W_DRAIN joins the W_CHECK class: busy, but no phase has begun — saying
  // SCORE there would both lie about the phase and (because the pv route
  // level is still applied while the previous head's beat drains) make an
  // in-phase observer record a spurious third ROUTE per head.
  assign walk_phase = ((state == W_IDLE) || (state == W_CHECK)
                       || (state == W_DRAIN) || (state == W_ERR))
                                                     ? WPH_IDLE
                    : (state == W_DONE)              ? WPH_DONE
                    : (((state >= W_P_FENCE) && (state <= W_P_WJEM))
                       || (state == W_P_RT))         ? WPH_PV
                                                     : WPH_SCORE;

  // last token of the current chunk / last chunk / last block
  logic last_tk, last_ch, last_bj, last_cw;
  assign last_tk = (tk == (c0 + {{(WALK_T_W-5){1'b0}}, nc} - WALK_T_W'(1)));
  assign last_ch = (ci == (nch - CH_W'(1)));
  assign last_bj = (bj == BJ_W'(BPR - 1));
  assign last_cw = (cw == (t_rows - WALK_T_W'(1)));

  // ── the walk ─────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state         <= W_IDLE;
      desc_q        <= '0;
      t_rows        <= '0;
      nch           <= '0;
      nbt           <= '0;
      split         <= 1'b0;
      ci            <= '0;
      tk            <= '0;
      bj            <= '0;
      cw            <= '0;
      c0            <= '0;
      comp_q        <= '0;
      pv_route      <= 1'b0;
      abort_l       <= 1'b0;
      kv_start      <= 1'b0;
      kv_addr       <= '0;
      walk_err      <= 1'b0;
      walk_err_code <= WALK_ERR_NONE;
    end else begin
      walk_err <= 1'b0;                        // 1-cycle pulse
      kv_start <= 1'b0;
      if (abort_req) abort_l <= 1'b1;

      // D-020: a soft reset aborts the walk. The in-flight MXE job is NOT
      // killed — seq_walker drains it (never violate §5) — the walker simply
      // stops emitting and reports. The AXI master is left to finish its
      // current transaction (K_IDLE is a transaction boundary).
      if (abort_l && (state != W_IDLE) && (kst == K_IDLE)) begin
        state         <= W_IDLE;
        abort_l       <= 1'b0;
        walk_err      <= 1'b1;
        walk_err_code <= WALK_ERR_ABORT;
      end else if (kv_rderr && (state != W_IDLE)) begin
        state         <= W_ERR;
        walk_err      <= 1'b1;
        walk_err_code <= WALK_ERR_SEQ;
      end else begin
        unique case (state)
          // ── kick + legality (§3: checked BEFORE any state change) ────────
          W_IDLE: if (walk_en && walk_go && !abort_l) begin
            desc_q <= walk_desc_unpack(desc_word[WALK_DW_GEOM],
                                       desc_word[WALK_DW_RQ],
                                       desc_word[WALK_DW_MASK]);
            state  <= W_CHECK;
          end

          W_CHECK: begin
            automatic walk_desc_t d = desc_q;
            automatic walk_err_e  e = walk_desc_check(d, CFG_D);
            if (e != WALK_ERR_NONE) begin
              // §A-1: an unsupported tier is REFUSED, never walked with a
              // relaxed equivalence. Host mode is retained.
              state         <= W_ERR;
              walk_err      <= 1'b1;
              walk_err_code <= e;
            end else begin
              t_rows <= d.t_rows;
              nbt    <= (d.t_rows + WALK_T_W'(7)) >> 3;
              nch    <= CH_W'((d.t_rows + WALK_T_W'(7)) >> 3);
              split  <= ((d.t_rows + WALK_T_W'(7)) >> 3) > WALK_T_W'(BPR);
              ci     <= '0; tk <= '0; bj <= '0; cw <= '0; c0 <= '0;
              // (d): clearing pv_route IS a route-level change, so it is
              // legal here ONLY while the touched paths are idle. When they
              // are — every single-head walk, every host-kicked walk, every
              // unit-TB walk — this is the pre-I-C behaviour, same cycle,
              // same value, same next state. When they are NOT (a multi-head
              // head boundary: walker2 kicks head h+1 one cycle after head
              // h's walker finished, while the MXE still holds head h's last
              // PV beat) the walk detours through W_DRAIN below.
              if (tile_idle) begin
                pv_route <= 1'b0;
                state    <= d.en_score ? W_S_FENCE : W_P_FENCE;
              end else begin
                state    <= W_DRAIN;
              end
            end
          end

          // ── SCORE ───────────────────────────────────────────────────────
          // (d) change the route level only while the touched paths are idle
          // The drain fence: reached ONLY when the previous head's datapath
          // is still busy at the kick. WPH_IDLE (no phase is active — the
          // W_CHECK class), so the route level change still lands on the
          // boundary INTO the score phase, exactly where it does when the
          // tile was already idle.
          W_DRAIN: if (tile_idle) begin
            pv_route <= 1'b0;
            state    <= desc_q.en_score ? W_S_FENCE : W_P_FENCE;
          end

          W_S_FENCE: if (tile_idle) state <= W_S_SJOB;
          W_S_SJOB:  if (dj_ready)  state <= W_S_CSREQ;

          W_S_CSREQ: if (comp_req_ready) state <= W_S_CSRES;
          W_S_CSRES: if (comp_res_valid) begin
            comp_q <= comp_res_data;
            state  <= W_S_CSEMIT;
          end
          W_S_CSEMIT: if (cs_ready) begin
            if (last_cw) begin cw <= '0; state <= W_S_QJOB; end
            else         begin cw <= cw + WALK_T_W'(1); state <= W_S_CSREQ; end
          end

          W_S_QJOB: if (qj_ready) state <= W_S_QSREQ;
          W_S_QSREQ: if (comp_req_ready) state <= W_S_QSRES;
          W_S_QSRES: if (comp_res_valid) begin
            comp_q <= comp_res_data;
            state  <= W_S_QSEMIT;
          end
          W_S_QSEMIT: if (qs_ready) begin
            if (last_cw) begin cw <= '0; state <= W_S_WJLD; end
            else         begin cw <= cw + WALK_T_W'(1); state <= W_S_QSREQ; end
          end

          // per chunk: WJ-LOAD, FJOB, {poll+RADDR}*nc, WJ-EMIT, AJ-EMIT, DESC
          W_S_WJLD: if (wj_ready) state <= W_S_FJOB;
          W_S_FJOB: if (fj_ready) begin                 // (c) FJOB after LOAD
            tk       <= c0;
            kv_addr  <= WALK_SC_AW'({1'b0, c0});
            kv_start <= 1'b1;
            state    <= W_S_KV;
          end
          W_S_KV: if (kv_done) begin
            if (last_tk) state <= W_S_WJEM;
            else begin
              tk       <= tk + WALK_T_W'(1);
              kv_addr  <= WALK_SC_AW'({1'b0, tk} + 9'd1);
              kv_start <= 1'b1;
            end
          end
          W_S_WJEM: if (wj_ready) state <= W_S_AJEM;    // (b) weights first
          W_S_AJEM: if (aj_ready) state <= W_S_DESC;
          W_S_DESC: if (ds_ready) begin
            if (last_ch) begin
              ci <= '0; c0 <= '0;
              state <= W_S_C8A;
            end else begin
              ci    <= ci + CH_W'(1);
              c0    <= c0 + WALK_T_W'(8);
              state <= W_S_WJLD;
            end
          end

          W_S_C8A: if (aj_ready) state <= split ? W_S_C8B : W_P_FENCE;
          W_S_C8B: if (aj_ready) state <= W_P_FENCE;

          // ── PV ──────────────────────────────────────────────────────────
          W_P_RT: state <= W_P_AJHD;

          W_P_FENCE: if (tile_idle) begin
            if (desc_q.en_pv) begin
              bj <= '0; ci <= '0; c0 <= '0;
              pv_route <= 1'b1;
              // one settling cycle: the route LEVEL must be established
              // before the first job of the phase is presented, exactly as
              // the host's ROUTE precedes its first AJ. Without this the
              // level change and the AJ handshake land on the same edge.
              state <= W_P_RT;
            end else state <= W_DONE;
          end
          // (a) EMIT -> DESC -> EMIT. Reordering this deadlocks when split.
          W_P_AJHD: if (aj_ready) state <= W_P_DESC;
          W_P_DESC: if (ds_ready) state <= split ? W_P_AJTL : W_P_WJLD;
          W_P_AJTL: if (aj_ready) state <= W_P_WJLD;

          W_P_WJLD: if (wj_ready) state <= W_P_FJOB;
          W_P_FJOB: if (fj_ready) begin
            tk       <= c0;
            kv_addr  <= WALK_SC_AW'({1'b0, t_rows} + {1'b0, c0});
            kv_start <= 1'b1;
            state    <= W_P_KV;
          end
          W_P_KV: if (kv_done) begin
            if (last_tk) state <= W_P_WJEM;
            else begin
              tk       <= tk + WALK_T_W'(1);
              kv_addr  <= WALK_SC_AW'({1'b0, t_rows} + {1'b0, tk} + 9'd1);
              kv_start <= 1'b1;
            end
          end
          W_P_WJEM: if (wj_ready) begin
            if (!last_ch) begin
              ci    <= ci + CH_W'(1);
              c0    <= c0 + WALK_T_W'(8);
              state <= W_P_WJLD;
            end else if (!last_bj) begin
              ci    <= '0; c0 <= '0;
              bj    <= bj + BJ_W'(1);
              state <= W_P_AJHD;
            end else begin
              state <= W_DONE;
            end
          end

          W_DONE: state <= W_IDLE;
          W_ERR:  state <= W_IDLE;
          default: state <= W_IDLE;
        endcase
      end
    end
  end

  // Only STATUS[0] (idle) and STATUS[1] (RD_ERR) are contractually defined
  // for the walker's poll (kvq_engine.sv:950-954); the rest is not ours to read.
  logic unused_ok;
  assign unused_ok = &{1'b0, kvm_rdata[31:2]};

endmodule

`endif // SEQ_LAYER_WALKER_SV
