// tb_wg3_topo_ab.sv — W-G3 A/B: does the TILE's composite/interlock TOPOLOGY
//                     (not the walker, not the vectors) block a multi-head,
//                     multi-KV-group fmt=1 attention walk?
//
// A COPY of verif/seq_walker/tb_walker2_sb.sv (never edited in place — the
// B1 §7 sibling rule), refreshed from that file AFTER the D-029/F1 constant
// split (combine-11), with ONE added plusarg, `+topo`:
//
//   +topo=0  CONTROL — the verified harness topology: N_ENG per-KV-head
//            seq_walker_comp caches, and the presented head's OWN s_q pushed
//            at each hd_* interlock. This is what verif/seq_walker gates.
//
//   +topo=1  the PRE-F6(i) tile topology. NOTE (updated after combine-13):
//            apex_top at KVQ_GQA_NENG>1 now instantiates apex_wcomp_bank —
//            N_ENG per-KV-head caches — so this arm NO LONGER MODELS THE
//            CURRENT TILE. It is retained as the REGRESSION WITNESS for the
//            F6(i) defect: it reproduces the exact 980-error signature the
//            single-cache topology produced, and it is what the fix had to
//            eliminate. F6(ii) (the single-arm hd_ready sticky) still blocks
//            multi-HEAD walks, so a green arm A does NOT mean a multi-head
//            walk works. Modelled from apex_top as it stood at 0c99a32:
//            (a) ONE seq_walker_comp instance (apex_top.sv:583 `u_wcomp` —
//                a single instance, no generate loop), shared by every
//                engine and indexed by RECORD ADDRESS only, so the 4 KV
//                groups collide in [0,T)/[T,2T);
//            (b) the s_q latch loads ONCE, pre-walk, with head 0's scale —
//                apex_top.sv:624-629 `hd_sq_seen_q` is a single sticky armed
//                by an ss-tap beat, and apex_top.sv:511-546 forces every host
//                job port's ready to 0 while `walk_en_q` is set, so no
//                further head's q row can be staged mid-walk.
//
// Everything else — the walker, the vectors, the scoreboard, the stubs, and
// the consumer-limit DESC asserts — is byte-identical between the two arms.
// The DELTA is the tile topology alone.
//
// Gate W-G1 at unit level (IB_WALK.md §4): the fmt=1 walker, given only the
// 64-word descriptor image, emits a stream BIT-EXACT and IN-ORDER against
// the layer-trace spec (.sub.ops from gen_layer_trace.py — golden-gated
// against decoder_layer_fx at generation time). The reference is never
// re-derived here.
//
// Structure inherits tb_walker_sb (the house pattern) with three additions:
//   * N_ENG seq_walker_comp instances model the per-KV-head engine caches
//     (§9.1 R3 as amended); comp/kvm traffic is routed by the DUT's
//     kv_eng_sel, and the sel value is CHECKED against the spec's per-head
//     engine table at every head interlock and every KVQ write.
//   * a fetch-reader stub (randomized backpressure) captures fuel_req
//     records; tag/base/beats are scoreboarded like any other emission.
//   * the hd_valid/hd_ready head interlock is the per-head s_q injection
//     hook: on each presented head the TB pushes that head's s_q into the
//     head's GROUP cache — the stand-in for the IB-LAYER q-path (Q1).
//
// DESC capture is 9-field (accumulate included): the k-split accumulate
// chain is exactly what m6 must be able to break.
//
// ── D-029 ERRATUM (I-B): THE INDEPENDENT LEGALITY CHECK ─────────────────────
// The scoreboard above is a SELF-CONSISTENT oracle: expectations come from
// the same decomposition constants the DUT walks by, so a walker and a spec
// that are uniformly wrong agree perfectly and the suite reports PASS. That
// is not a hypothetical — it is how the n-split-by-field-width defect (every
// 7B projection descriptor asking for n=4095 against an 8-wide array) passed
// W-G1. The fix is a second, INDEPENDENT predicate: every accepted mxe_desc
// is replayed through the CONSUMER's acceptance rule, with M_TILE_MAX
// imported from mxe_cfg_pkg and MXE_N/K_MAX from apex_pkg — the same symbols
// mxe_ctrl.sv's `legal` term reads, never restated here. A decomposition
// change cannot move this check, which is exactly what makes it worth having.
`timescale 1ns/1ps

module tb_wg3_topo_ab;

  import apex_pkg::*;
  import mxe_cfg_pkg::M_TILE_MAX;
  import seq_walker_pkg::*;

  localparam int unsigned CFG_D      = `CFG_D_TB;
  localparam int unsigned N_ENG      = 4;
  localparam int unsigned ENG_W      = $clog2(N_ENG);
  localparam int unsigned STAGE_NB_W = $clog2(CFG_D / 8) + 1;

  logic clk, rst_n;
  initial begin clk = 1'b0; rst_n = 1'b0; end
  always #5 clk = ~clk;

  string vec_file;
  int    lat_mode, bp_mode, seed;
  int    topo;          // 0 = verified harness, 1 = apex_top as built
  int    n_checks, n_errors, n_cases;

  // ── DUT wiring ───────────────────────────────────────────────────────────
  logic        walk_en, walk_go, abort_req, tile_idle;
  logic [31:0] desc2_word [WALK2_DESC_WORDS];
  logic        walk2_busy, walk2_err;
  walk_step_e  walk2_step;
  walk_phase_e attn_phase;
  walk_err_e   walk2_err_code;

  logic        wf_valid, wf_ready;
  walk2_freq_t wf_req;
  logic        hd_valid, hd_ready;
  logic [7:0]  hd_head;
  logic [ENG_W-1:0] kv_eng_sel;
  logic        lw_rope_en, lw_rope_bank, lw_resid_arm;
  logic [6:0]  lw_rope_pos;
  logic [1:0]  lw_ser_dst;
  logic [2:0]  lw_fsrc_ext;           // E-3: 3 bits ([2] = LAYER_CTRL[7])
  logic        lw_nsrc, lw_nsrc_own;  // E-4b: l_nsrc level + ownership (no
                                      //  NSRC case walks in this TB)
  logic        sj_valid;              // E-3b serializer job (sunk: no
  logic [7:0]  sj_beats;              //  QSTAGE case walks in this TB)
  logic [3:0]  sj_lanes;
  logic        lu_valid, lu_ready, jc_valid, jc_ready, ljn_valid, ljn_ready;
  logic [1:0]  lu_unit;
  logic [11:0] lu_cols;
  logic [31:0] jc_data;
  logic [DIM_W-1:0] ljn_cols;

  logic        ds_valid, ds_ready;
  mxe_desc_t   ds_desc;
  logic        rt_fsrc, rt_fdst, rt_asrc, rt_wsrc, rt_qsrc, rt_kvu;
  logic [1:0]  rt_rdst;
  logic        fj_valid, fj_ready, qj_valid, qj_ready, qj_mode,
               dj_valid, dj_ready;
  logic [DIM_W-1:0] fj_rows, qj_cols, dj_cols;
  logic        aj_valid, aj_ready, aj_op, aj_bank, wj_valid, wj_ready,
               wj_op, wj_bank;
  logic [1:0]  aj_pat, wj_pat;
  logic [4:0]  aj_rows, aj_sel, wj_rows, wj_sel;
  logic [STAGE_NB_W-1:0] aj_nb, wj_nb;
  logic        qs_valid, qs_ready, cs_valid, cs_ready;
  logic [31:0] qs_data, cs_data;

  logic                  cq_req_v, cq_req_r, cq_req_is_qs, cq_res_v, cq_res_r;
  logic [WALK_SC_AW-1:0] cq_req_idx;
  logic [31:0]           cq_res_data;

  logic        kaw_v, kaw_r, kw_v, kw_r, kb_v, kb_r, kar_v, kar_r, kr_v, kr_r;
  logic [7:0]  kaw_a, kar_a;
  logic [31:0] kw_d, kr_d;

  seq_layer_walker2 #(.CFG_D(CFG_D), .N_ENG(N_ENG)) u_walk2 (
    .clk(clk), .rst_n(rst_n),
    .walk_en(walk_en), .walk_go(walk_go), .abort_req(abort_req),
    .desc2_word(desc2_word), .tile_idle(tile_idle),
    .walk2_busy(walk2_busy), .walk2_step(walk2_step),
    .attn_phase(attn_phase),
    .walk2_err(walk2_err), .walk2_err_code(walk2_err_code),
    .wf_valid(wf_valid), .wf_ready(wf_ready), .wf_req(wf_req),
    .hd_valid(hd_valid), .hd_ready(hd_ready), .hd_head(hd_head),
    .lw_rope_en(lw_rope_en), .lw_rope_bank(lw_rope_bank),
    .lw_rope_pos(lw_rope_pos), .lw_ser_dst(lw_ser_dst),
    .lw_fsrc_ext(lw_fsrc_ext), .lw_resid_arm(lw_resid_arm),
    .lw_nsrc(lw_nsrc), .lw_nsrc_own(lw_nsrc_own),
    .lu_valid(lu_valid), .lu_ready(lu_ready), .lu_unit(lu_unit),
    .lu_cols(lu_cols),
    .jc_valid(jc_valid), .jc_ready(jc_ready), .jc_data(jc_data),
    .lj_valid(ljn_valid), .lj_ready(ljn_ready), .lj_cols(ljn_cols),
    .sj_valid(sj_valid), .sj_ready(1'b1), .sj_beats(sj_beats),
    .sj_lanes(sj_lanes),
    .nf_busy(1'b0),
    .kv_eng_sel(kv_eng_sel),
    .ds_valid(ds_valid), .ds_ready(ds_ready), .ds_desc(ds_desc),
    .rt_feeder_src(rt_fsrc), .rt_feeder_dst(rt_fdst), .rt_act_src(rt_asrc),
    .rt_wgt_src(rt_wsrc), .rt_res_dst(rt_rdst), .rt_squant_src(rt_qsrc),
    .rt_kv_user(rt_kvu),
    .fj_valid(fj_valid), .fj_ready(fj_ready), .fj_rows(fj_rows),
    .qj_valid(qj_valid), .qj_ready(qj_ready), .qj_mode(qj_mode),
    .qj_cols(qj_cols),
    .dj_valid(dj_valid), .dj_ready(dj_ready), .dj_cols(dj_cols),
    .aj_valid(aj_valid), .aj_ready(aj_ready), .aj_op(aj_op),
    .aj_bank(aj_bank), .aj_pat(aj_pat), .aj_rows(aj_rows), .aj_nb(aj_nb),
    .aj_sel(aj_sel),
    .wj_valid(wj_valid), .wj_ready(wj_ready), .wj_op(wj_op),
    .wj_bank(wj_bank), .wj_pat(wj_pat), .wj_rows(wj_rows), .wj_nb(wj_nb),
    .wj_sel(wj_sel),
    .qs_valid(qs_valid), .qs_ready(qs_ready), .qs_data(qs_data),
    .cs_valid(cs_valid), .cs_ready(cs_ready), .cs_data(cs_data),
    .comp_req_valid(cq_req_v), .comp_req_ready(cq_req_r),
    .comp_req_is_qs(cq_req_is_qs), .comp_req_idx(cq_req_idx),
    .comp_res_valid(cq_res_v), .comp_res_ready(cq_res_r),
    .comp_res_data(cq_res_data),
    .kvm_awvalid(kaw_v), .kvm_awready(kaw_r), .kvm_awaddr(kaw_a),
    .kvm_wvalid(kw_v), .kvm_wready(kw_r), .kvm_wdata(kw_d),
    .kvm_bvalid(kb_v), .kvm_bready(kb_r),
    .kvm_arvalid(kar_v), .kvm_arready(kar_r), .kvm_araddr(kar_a),
    .kvm_rvalid(kr_v), .kvm_rready(kr_r), .kvm_rdata(kr_d)
  );

  // ── N_ENG per-group scale caches (the per-KV-head engine model) ──────────
  logic        snp_valid [N_ENG];
  logic        snp_last  [N_ENG];
  logic        snp_flush [N_ENG];
  logic        sqv       [N_ENG];
  logic [15:0] snp_data  [N_ENG];
  logic [15:0] sqd       [N_ENG];
  logic [WALK_SC_AW-1:0] snp_addr [N_ENG];
  logic        eng_req_r  [N_ENG];
  logic        eng_res_v  [N_ENG];
  logic [31:0] eng_res_d  [N_ENG];
  logic        eng_err_f  [N_ENG];
  logic        eng_err_s  [N_ENG];

  // topo=1 collapses the per-KV-head caches onto instance 0 — apex_top's
  // single `u_wcomp`, shared by every engine and addressed by record index.
  wire [ENG_W-1:0] comp_sel = (topo != 0) ? ENG_W'(0) : kv_eng_sel;

  for (genvar gi = 0; gi < int'(N_ENG); gi++) begin : g_comp
    logic req_v_i, res_r_i;
    assign req_v_i = cq_req_v && (int'(comp_sel) == gi);
    assign res_r_i = cq_res_r && (int'(comp_sel) == gi);
    seq_walker_comp #(.CFG_D(CFG_D)) u_comp (
      .clk(clk), .rst_n(rst_n),
      .snp_valid(snp_valid[gi]), .snp_data(snp_data[gi]),
      .snp_last(snp_last[gi]), .snp_addr(snp_addr[gi]),
      .snp_flush(snp_flush[gi]),
      .sq_valid(sqv[gi]), .sq_data(sqd[gi]),
      .req_valid(req_v_i), .req_ready(eng_req_r[gi]),
      .req_is_qs(cq_req_is_qs), .req_idx(cq_req_idx),
      .res_valid(eng_res_v[gi]), .res_ready(res_r_i),
      .res_data(eng_res_d[gi]),
      .err_frame(eng_err_f[gi]), .err_stale(eng_err_s[gi])
    );
  end

  assign cq_req_r    = eng_req_r[comp_sel];
  assign cq_res_v    = eng_res_v[comp_sel];
  assign cq_res_data = eng_res_d[comp_sel];

  // ── consumer stubs: randomized accept latency (v1 pattern + wf/hd) ───────
  int unsigned lfsr;
  function automatic int unsigned rnd();
    lfsr = (lfsr >> 1) ^ (32'hEDB8_8320 & {32{lfsr[0]}});
    rnd  = lfsr;
  endfunction

  logic [7:0] stall_ds, stall_aj, stall_wj, stall_fj, stall_qj, stall_dj,
              stall_cs, stall_qs, stall_wf, stall_jc, stall_ljn;

  function automatic logic [7:0] new_stall();
    case (bp_mode)
      0: new_stall = 8'd0;
      1: new_stall = 8'(rnd() % 3);
      default: new_stall = 8'(rnd() % 9);
    endcase
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      stall_ds <= '0; stall_aj <= '0; stall_wj <= '0; stall_fj <= '0;
      stall_qj <= '0; stall_dj <= '0; stall_cs <= '0; stall_qs <= '0;
      stall_wf <= '0;
    end else begin
      stall_ds <= (ds_valid && ds_ready) ? new_stall()
                : (stall_ds != 0 ? stall_ds - 8'd1 : 8'd0);
      stall_aj <= (aj_valid && aj_ready) ? new_stall()
                : (stall_aj != 0 ? stall_aj - 8'd1 : 8'd0);
      stall_wj <= (wj_valid && wj_ready) ? new_stall()
                : (stall_wj != 0 ? stall_wj - 8'd1 : 8'd0);
      stall_fj <= (fj_valid && fj_ready) ? new_stall()
                : (stall_fj != 0 ? stall_fj - 8'd1 : 8'd0);
      stall_qj <= (qj_valid && qj_ready) ? new_stall()
                : (stall_qj != 0 ? stall_qj - 8'd1 : 8'd0);
      stall_dj <= (dj_valid && dj_ready) ? new_stall()
                : (stall_dj != 0 ? stall_dj - 8'd1 : 8'd0);
      stall_cs <= (cs_valid && cs_ready) ? new_stall()
                : (stall_cs != 0 ? stall_cs - 8'd1 : 8'd0);
      stall_qs <= (qs_valid && qs_ready) ? new_stall()
                : (stall_qs != 0 ? stall_qs - 8'd1 : 8'd0);
      stall_wf <= (wf_valid && wf_ready) ? new_stall()
                : (stall_wf != 0 ? stall_wf - 8'd1 : 8'd0);
      stall_jc <= (jc_valid && jc_ready) ? new_stall()
                : (stall_jc != 0 ? stall_jc - 8'd1 : 8'd0);
      stall_ljn <= (ljn_valid && ljn_ready) ? new_stall()
                : (stall_ljn != 0 ? stall_ljn - 8'd1 : 8'd0);
    end
  end

  assign ds_ready = (stall_ds == 0);
  assign aj_ready = (stall_aj == 0);
  assign wj_ready = (stall_wj == 0);
  assign fj_ready = (stall_fj == 0);
  assign qj_ready = (stall_qj == 0);
  assign dj_ready = (stall_dj == 0);
  assign cs_ready = (stall_cs == 0);
  assign qs_ready = (stall_qs == 0);
  assign wf_ready = (stall_wf == 0);
  assign jc_ready = (stall_jc == 0);
  assign ljn_ready = (stall_ljn == 0);

  // LAYER unit stub — models the §3b failure mode the executor lesson warns
  // about: job-ready is COMBINATIONAL through the busy counter and DROPS at
  // the accepting edge (busy loads on accept). A walker that re-checked
  // ready after the edge instead of capturing it at the edge would deadlock
  // here; registered-accept (the DUT's idiom) is what this stub verifies.
  int unsigned lu_busy_cnt;
  assign lu_ready = (lu_busy_cnt == 0);
  always_ff @(posedge clk) begin
    if (!rst_n) lu_busy_cnt <= 0;
    else if (lu_valid && lu_ready)
      lu_busy_cnt <= 1 + (rnd() % ((lat_mode == 0) ? 1 : 12));
    else if (lu_busy_cnt != 0) lu_busy_cnt <= lu_busy_cnt - 1;
  end

  // ── KVQ AXI-Lite slave stub (v1 semantics: poll-before-RADDR enforced) ───
  int unsigned kv_busy_cnt;
  logic        kv_saw_idle_poll;
  int          kv_writes;

  assign kaw_r = (kaw_v && kw_v);
  assign kw_r  = (kaw_v && kw_v);
  assign kar_r = 1'b1;

  always begin
    @(posedge clk);
    if (!rst_n) begin
      kv_busy_cnt      <= 0;
      kb_v             <= 1'b0;
      kr_v             <= 1'b0;
      kr_d             <= '0;
      kv_saw_idle_poll <= 1'b0;
    end else begin
      if (kv_busy_cnt != 0) kv_busy_cnt <= kv_busy_cnt - 1;
      kr_v <= kar_v && kar_r;
      if (kar_v && kar_r) begin
        kr_d <= {30'b0, 1'b0, (kv_busy_cnt == 0)};
        if (kv_busy_cnt == 0) kv_saw_idle_poll <= 1'b1;
      end
      kb_v <= kaw_v && kaw_r;
      if (kaw_v && kaw_r) begin
        kv_writes <= kv_writes + 1;
        if (!kv_saw_idle_poll) begin
          $display("FAIL: READ_ADDR write with no preceding idle poll");
          n_errors++;
        end
        kv_saw_idle_poll <= 1'b0;
        case (lat_mode)
          0: kv_busy_cnt <= 0;
          1: kv_busy_cnt <= (rnd() % 6);
          default: kv_busy_cnt <= 4 + (rnd() % 20);
        endcase
      end
    end
  end

  // ── expected-trace queue + emission capture ──────────────────────────────
  string exp_q[$];
  string cur_case;

  // ── D-029 erratum: the tile's own acceptance rule, replayed ──────────────
  // A TRANSCRIPTION of mxe_ctrl.sv's `legal` M/K/N terms, reading the very
  // symbols it reads (MXE_N/K_MAX from apex_pkg, M_TILE_MAX from
  // mxe_cfg_pkg). Independent of seq_walker_pkg's decomposition bounds BY
  // CONSTRUCTION: nothing here mentions WALK2_*, so no change to the walker's
  // chunking can move it. Counted separately from n_checks so the scoreboard
  // emission totals stay directly comparable across this change.
  int n_mxechk;

  task automatic mxe_desc_legal(input logic [DIM_W-1:0] m,
                                input logic [DIM_W-1:0] k,
                                input logic [DIM_W-1:0] n);
    begin
      n_mxechk++;
      if ((n == '0) || (n > DIM_W'(MXE_N))
          || (m == '0) || (m > DIM_W'(M_TILE_MAX))
          || (k == '0) || (k > DIM_W'(K_MAX))) begin
        // ONE string literal: {"a","b"} is a BIT concatenation in SV, not a
        // string join, and $display would print it as a number
        $display("FAIL [%s]: MXE-ILLEGAL DESC m=%0d k=%0d n=%0d — mxe_ctrl raises desc_error (legal: 1<=m<=%0d, 1<=k<=%0d, 1<=n<=%0d)",
                 cur_case, m, k, n, M_TILE_MAX, K_MAX, MXE_N);
        n_errors++;
      end
    end
  endtask

  task automatic chk(input string got);
    string want;
    begin
      n_checks++;
      if (exp_q.size() == 0) begin
        $display("FAIL [%s]: emission past end of reference: '%s'", cur_case, got);
        n_errors++;
      end else begin
        want = exp_q.pop_front();
        if (want != got) begin
          $display("FAIL [%s] check %0d: expected '%s' got '%s'", cur_case, n_checks, want, got);
          n_errors++;
        end
      end
    end
  endtask

  // route level: capture on change while the WRAPPED engine runs a phase
  logic [15:0] rt_word, rt_prev;
  logic        rt_seen;
  assign rt_word = {8'd0, rt_kvu, rt_qsrc, rt_rdst, rt_wsrc, rt_asrc,
                    rt_fdst, rt_fsrc};

  always begin
    @(posedge clk);
    if (!rst_n) begin
      rt_prev <= '0; rt_seen <= 1'b0;
    end else if (walk2_busy && (attn_phase == WPH_SCORE
                                || attn_phase == WPH_PV)) begin
      if (!rt_seen || (rt_word != rt_prev)) begin
        chk($sformatf("ROUTE %04x", rt_word));
        rt_prev <= rt_word;
        rt_seen <= 1'b1;
      end
    end else begin
      rt_seen <= 1'b0;
    end
  end

  always begin
    @(posedge clk);
    if (rst_n) begin
      if (dj_valid && dj_ready) chk($sformatf("SJOB %0x", dj_cols));
      if (qj_valid && qj_ready) chk($sformatf("QJOB %0d %0x", qj_mode, qj_cols));
      if (fj_valid && fj_ready) chk($sformatf("FJOB %0x", fj_rows));
      if (cs_valid && cs_ready) chk($sformatf("CS %08x", cs_data));
      if (qs_valid && qs_ready) chk($sformatf("QS %08x", qs_data));
      if (aj_valid && aj_ready)
        chk($sformatf("AJ %0d %0d %0d %0x %0x %0x", aj_op, aj_bank, aj_pat,
                      aj_rows, aj_nb, aj_sel));
      if (wj_valid && wj_ready)
        chk($sformatf("WJ %0d %0d %0d %0x %0x %0x", wj_op, wj_bank, wj_pat,
                      wj_rows, wj_nb, wj_sel));
      // KVQ writes split by address: RADDR reads (attention) vs WRITE_ADDR
      // (store phase, engine id folded into the expectation)
      if (kaw_v && kaw_r && kaw_a == 8'h2C) chk($sformatf("KVW %0x", kw_d));
      if (kaw_v && kaw_r && kaw_a == 8'h28)
        chk($sformatf("KVWA %0x %0x", kv_eng_sel, kw_d));
      // 9-field DESC: accumulate is part of the checked contract (m6 target)
      if (ds_valid && ds_ready) begin
        chk($sformatf("DESC %02x %0x %0x %0x %0d %0d %0x %0x %0d",
                      ds_desc.opcode, ds_desc.m_dim, ds_desc.k_dim,
                      ds_desc.n_dim, ds_desc.accumulate, ds_desc.requant_en,
                      ds_desc.rq_scale, ds_desc.rq_shift, ds_desc.mode_os));
        mxe_desc_legal(ds_desc.m_dim, ds_desc.k_dim,   // D-029 erratum
                       ds_desc.n_dim);                // header note
      end
      if (wf_valid && wf_ready)
        chk($sformatf("FETCH %0x %0x %0x", wf_req.tag, wf_req.base64,
                      wf_req.beats64));
      // §3b LAYER channels
      if (ljn_valid && ljn_ready) chk($sformatf("LJOB %0x", ljn_cols));
      if (lu_valid && lu_ready)
        chk($sformatf("LUJOB %0d %0x", lu_unit, lu_cols));
      if (jc_valid && jc_ready) chk($sformatf("LJC %08x", jc_data));
    end
  end

  // LAYER level word: capture on change while walking (ROUTE/LCTL pattern).
  // E-4b: [19] carries lw_nsrc (the pkg walk2_lctl layout; hand-rolled here
  // as before, kept in step with the package bit map).
  logic [WALK2_LCTL_W-1:0] lctl_word, lctl_prev;
  logic                    lctl_seen;
  assign lctl_word = {lw_nsrc, 4'b0,
                      lw_rope_pos, lw_fsrc_ext[2], lw_resid_arm, lw_fsrc_ext[1:0],
                      lw_ser_dst, lw_rope_bank, lw_rope_en};

  always begin
    @(posedge clk);
    if (!rst_n) begin
      lctl_prev <= '0; lctl_seen <= 1'b0;
    end else if (walk2_busy) begin
      if (!lctl_seen) begin
        lctl_prev <= lctl_word;                // reset image, not an emission
        lctl_seen <= 1'b1;
      end else if (lctl_word != lctl_prev) begin
        chk($sformatf("LCTL %04x", lctl_word));
        lctl_prev <= lctl_word;
      end
    end else begin
      lctl_seen <= 1'b0;
    end
  end

  // engine-select checks: attention RADDR writes must ride the current
  // head's engine (cur_g, latched at the hd_* interlock). Store-phase
  // WRITE_ADDR writes carry their engine IN the scoreboard expectation
  // ("KVWA <g> <rec>"), so they are checked there, not here.
  logic [ENG_W-1:0] cur_g;
  always begin
    @(posedge clk);
    if (rst_n && kaw_v && kaw_r && kaw_a == 8'h2C && kv_eng_sel != cur_g)
    begin
      $display("FAIL [%s]: KVQ read-addr on engine %0d, expected %0d",
               cur_case, kv_eng_sel, cur_g);
      n_errors++;
    end
  end

  // composite-unit error pulses are never expected in a good run
  always begin
    @(posedge clk);
    for (int gi = 0; gi < int'(N_ENG); gi++) begin
      if (rst_n && eng_err_f[gi]) begin
        $display("FAIL [%s]: comp[%0d] err_frame", cur_case, gi);
        n_errors++;
      end
      if (rst_n && eng_err_s[gi]) begin
        $display("FAIL [%s]: comp[%0d] err_stale", cur_case, gi);
        n_errors++;
      end
    end
  end

  // AXI-Lite contract checks (v1)
  always begin
    @(posedge clk);
    if (rst_n) begin
      if (kar_v && kar_a != 8'h04) begin
        $display("FAIL: AXIL read of 0x%02h", kar_a);
        n_errors++;
      end
      if (kaw_v && kaw_a != 8'h2C && kaw_a != 8'h28) begin
        // RADDR (attention reads) and WRITE_ADDR (store phase) only
        $display("FAIL: AXIL write to 0x%02h", kaw_a);
        n_errors++;
      end
      if (!kb_r || !kr_r) begin
        $display("FAIL: walker retracted bready/rready");
        n_errors++;
      end
    end
  end

  logic unused_tb;
  assign unused_tb = &{1'b0, ds_desc[127:80], ds_desc[79:68], walk2_step,
                       hd_head, sj_valid, sj_beats, sj_lanes, lw_nsrc_own};

  // ── per-head interlock: push s_q into the head's group cache, check g ────
  logic [15:0]      hsq  [0:31];
  logic [ENG_W-1:0] hgrp [0:31];
  int               n_heads_seen;

  task automatic push_sq(input int g, input logic [15:0] v);
    begin
      if (g >= 0 && g < int'(N_ENG)) begin
        @(negedge clk);
        sqv[g] = 1'b1; sqd[g] = v;
        @(negedge clk);
        sqv[g] = 1'b0;
      end
    end
  endtask

  always begin
    @(posedge clk);
    if (rst_n && hd_valid && !hd_ready) begin
      // spec's engine table is the arbiter for the mapping (R3 amended)
      if (kv_eng_sel != hgrp[hd_head[4:0]]) begin
        $display("FAIL [%s]: head %0d on engine %0d, expected %0d",
                 cur_case, hd_head, kv_eng_sel, hgrp[hd_head[4:0]]);
        n_errors++;
      end
      cur_g = kv_eng_sel;
      // topo=1: the tile's s_q latch was loaded ONCE before WALK_GO with the
      // pre-staged head's scale; no host job port is reachable mid-walk, so
      // no later head can refresh it. Model that by pushing nothing here.
      if (topo == 0)
        push_sq(int'(hgrp[hd_head[4:0]]), hsq[hd_head[4:0]]);
      n_heads_seen++;
      @(negedge clk);
      hd_ready = 1'b1;
      @(negedge clk);
      hd_ready = 1'b0;
    end
  end

  // ── vector replay ────────────────────────────────────────────────────────
  int    fd;
  string line;
  int    expect_refuse;
  int    expect_code;
  int    cur_grp;

  task automatic reset_dut();
    begin
      rst_n = 1'b0; walk_en = 1'b0; walk_go = 1'b0; abort_req = 1'b0;
      tile_idle = 1'b1; hd_ready = 1'b0; cur_g = '0; cur_grp = 0;
      for (int g = 0; g < int'(N_ENG); g++) begin
        snp_valid[g] = 1'b0; snp_last[g] = 1'b0; snp_flush[g] = 1'b0;
        snp_data[g] = '0; snp_addr[g] = '0; sqv[g] = 1'b0; sqd[g] = '0;
      end
      for (int i = 0; i < WALK2_DESC_WORDS; i++) desc2_word[i] = '0;
      for (int i = 0; i < 32; i++) begin hsq[i] = '0; hgrp[i] = '0; end
      repeat (4) @(posedge clk);
      rst_n = 1'b1;
      @(posedge clk);
    end
  endtask

  task automatic push_record(input int g, input logic [WALK_SC_AW-1:0] addr,
                             input int n, input logic [15:0] vals []);
    begin
      if (g >= 0 && g < int'(N_ENG)) begin
        for (int i = 0; i < n; i++) begin
          @(negedge clk);
          snp_valid[g] = 1'b1;
          snp_data[g]  = vals[i];
          snp_addr[g]  = addr;
          snp_last[g]  = (i == n - 1);
        end
        @(negedge clk);
        snp_valid[g] = 1'b0;
        snp_last[g]  = 1'b0;
      end
    end
  endtask

  task automatic run_walk();
    int guard;
    begin
      // topo=1: the ONE pre-walk s_q staging event (head 0's), into the ONE
      // cache — apex_top's `hd_sq_seen_q` arm + `u_wcomp.s_q_q` load.
      if (topo != 0) push_sq(0, hsq[0]);
      @(negedge clk); walk_en = 1'b1; walk_go = 1'b1;
      @(negedge clk); walk_go = 1'b0;
      guard = 0;
      while (walk2_busy && guard < 40_000_000) begin
        @(posedge clk); guard++;
      end
      if (guard >= 40_000_000) begin
        $display("FAIL [%s]: walk did not complete (watchdog)", cur_case);
        n_errors++;
      end
      repeat (4) @(posedge clk);
      @(negedge clk); walk_en = 1'b0;
    end
  endtask

  logic [15:0] rec_vals [];
  logic        saw_err;
  walk_err_e   saw_code;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_err <= 1'b0; saw_code <= WALK_ERR_NONE;
    end else if (walk2_err) begin
      saw_err <= 1'b1; saw_code <= walk2_err_code;
    end
  end

  initial begin
    if (!$value$plusargs("vectors=%s", vec_file)) begin
      $display("FAIL: +vectors=<file> required"); $finish;
    end
    if (!$value$plusargs("lat_mode=%d", lat_mode)) lat_mode = 1;
    if (!$value$plusargs("bp_mode=%d",  bp_mode))  bp_mode  = 1;
    if (!$value$plusargs("seed=%d",     seed))     seed     = 1;
    if (!$value$plusargs("topo=%d",     topo))     topo     = 0;
    lfsr = 32'(seed) | 32'h1;
    n_checks = 0; n_errors = 0; n_cases = 0; n_heads_seen = 0; n_mxechk = 0;
    kv_busy_cnt = 0; kv_saw_idle_poll = 1'b0; kv_writes = 0;

    rec_vals = new [CFG_D];
    fd = $fopen(vec_file, "r");
    if (fd == 0) begin
      $display("FAIL: cannot open %s", vec_file); $finish;
    end

    reset_dut();
    expect_refuse = 0;
    expect_code = 0;
    cur_case = "?";

    while ($fgets(line, fd) != 0) begin
      if (line.len() < 2) continue;
      if (line.substr(0, 0) == "#") continue;
      if (line.substr(0, 3) == "CASE") begin
        cur_case = line.substr(5, line.len() - 2);
      end else if (line.substr(0, 1) == "DW") begin
        int idx; logic [31:0] w;
        void'($sscanf(line, "DW %d %h", idx, w));
        if (idx >= 0 && idx < int'(WALK2_DESC_WORDS)) desc2_word[idx] = w;
      end else if (line.substr(0, 4) == "GROUP") begin
        void'($sscanf(line, "GROUP %d", cur_grp));
      end else if (line.substr(0, 2) == "REC") begin
        int p; logic [WALK_SC_AW-1:0] a; logic [15:0] v;
        void'($sscanf(line, "REC %h", a));
        p = 9;
        for (int i = 0; i < CFG_D; i++) begin
          void'($sscanf(line.substr(p, p + 3), "%h", v));
          rec_vals[i] = v;
          p += 5;
        end
        push_record((topo != 0) ? 0 : cur_grp, a, CFG_D, rec_vals);
      end else if (line.substr(0, 5) == "HEADSQ") begin
        int h; logic [15:0] s;
        void'($sscanf(line, "HEADSQ %h %h", h, s));
        if (h >= 0 && h < 32) hsq[h] = s;
      end else if (line.substr(0, 3) == "HEAD") begin
        int h, g;
        void'($sscanf(line, "HEAD %h %h", h, g));
        if (h >= 0 && h < 32 && g >= 0 && g < int'(N_ENG))
          hgrp[h] = ENG_W'(g);
      end else if (line.substr(0, 12) == "EXPECT_REFUSE") begin
        void'($sscanf(line, "EXPECT_REFUSE %d", expect_code));
        expect_refuse = 1;
      end else if (line.substr(0, 1) == "E ") begin
        exp_q.push_back(line.substr(2, line.len() - 2));
      end else if (line.substr(0, 2) == "END") begin
        n_cases++;
        run_walk();
        if (expect_refuse != 0) begin
          n_checks++;
          if (!saw_err || int'(saw_code) != expect_code) begin
            $display("FAIL: refusal expected code=%0d, saw err=%0d code=%0d",
                     expect_code, saw_err, saw_code);
            n_errors++;
          end else begin
            $display("  refusal OK: err_code=%0d, no emissions", saw_code);
          end
        end else if (exp_q.size() != 0) begin
          $display("FAIL [%s]: %0d reference emissions never produced (first: '%s')",
                   cur_case, exp_q.size(), exp_q[0]);
          n_errors++;
        end
        exp_q.delete();
        expect_refuse = 0;
        expect_code = 0;
        reset_dut();
      end
    end
    $fclose(fd);

    $display("WG3-TOPO[%0s] RESULT: cases=%0d checks=%0d errors=%0d kvw=%0d heads=%0d",
             (topo == 0) ? "control-Neng-caches" : "pre-F6i-single-cache",
             n_cases, n_checks, n_errors, kv_writes, n_heads_seen);
    if (n_errors == 0 && n_checks > 0) $display("WG3-TOPO PASS");
    else                               $display("WG3-TOPO FAIL");
    $finish;
  end

endmodule
