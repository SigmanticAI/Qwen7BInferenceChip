// tb_walker_sb.sv — B1 stage-2 scoreboard TB for seq_layer_walker +
//                   seq_walker_comp (house verif/seq pattern).
//
// Checks Acceptance A (B1_WALKER.md §4) at unit level: the walker, given only
// a compact layer descriptor, emits a control stream whose ordered score+pv
// drive subsequence is BIT-EXACT and IN-ORDER against the subsequence
// extracted from the L3 case file — the spec (OPTIMIZATION.md:38).
//
// The reference is produced by gen_walker_vectors.py via the stage-0
// extractor; this TB never re-derives it. The composite words are NOT
// preloaded: the K/V fp16 records are replayed into the composite unit's
// SNOOP port exactly as the store phase would, so cs/qs are genuinely
// computed on-tile (§A-2) and a broken scale cache fails here.
//
// Stubs model the D-006 contract with randomized latency/backpressure: every
// job consumer accepts on its own schedule, and the KVQ AXI-Lite slave holds
// STATUS.idle low for a random burst after each READ_ADDR write — so a walker
// that skips the poll (the engine silently DROPS read_req outside ST_IDLE)
// desynchronises and is caught.
`timescale 1ns/1ps

module tb_walker_sb;

  import apex_pkg::*;
  import mxe_cfg_pkg::M_TILE_MAX;
  import seq_walker_pkg::*;

  localparam int unsigned CFG_D      = `CFG_D_TB;
  localparam int unsigned STAGE_NB_W = $clog2(CFG_D / 8) + 1;

  logic clk, rst_n;
  initial begin clk = 1'b0; rst_n = 1'b0; end
  always #5 clk = ~clk;

  // ── plusargs ─────────────────────────────────────────────────────────────
  string vec_file;
  int    lat_mode, bp_mode, seed;
  int    n_checks, n_errors, n_cases;

  // ── DUT wiring ───────────────────────────────────────────────────────────
  logic        walk_en, walk_go, abort_req, tile_idle;
  logic [31:0] desc_word [WALK_DESC_WORDS];
  logic        walk_busy, walk_err;
  walk_phase_e walk_phase;
  walk_err_e   walk_err_code;

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

  logic        snp_valid, snp_last, snp_flush, sq_valid;
  logic [15:0] snp_data, sq_data;
  logic [WALK_SC_AW-1:0] snp_addr;
  logic        comp_err_frame, comp_err_stale;

  logic        kaw_v, kaw_r, kw_v, kw_r, kb_v, kb_r, kar_v, kar_r, kr_v, kr_r;
  logic [7:0]  kaw_a, kar_a;
  logic [31:0] kw_d, kr_d;

  seq_layer_walker #(.CFG_D(CFG_D)) u_walk (
    .clk(clk), .rst_n(rst_n),
    .walk_en(walk_en), .walk_go(walk_go), .abort_req(abort_req),
    .desc_word(desc_word), .tile_idle(tile_idle),
    .walk_busy(walk_busy), .walk_phase(walk_phase),
    .walk_err(walk_err), .walk_err_code(walk_err_code),
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

  seq_walker_comp #(.CFG_D(CFG_D)) u_comp (
    .clk(clk), .rst_n(rst_n),
    .snp_valid(snp_valid), .snp_data(snp_data), .snp_last(snp_last),
    .snp_addr(snp_addr), .snp_flush(snp_flush),
    .sq_valid(sq_valid), .sq_data(sq_data),
    .req_valid(cq_req_v), .req_ready(cq_req_r), .req_is_qs(cq_req_is_qs),
    .req_idx(cq_req_idx),
    .res_valid(cq_res_v), .res_ready(cq_res_r), .res_data(cq_res_data),
    .err_frame(comp_err_frame), .err_stale(comp_err_stale),
    .dbg_stale_sc(u_dbg_sc), .dbg_stale_sq(u_dbg_sq)
  );

  // ── consumer stubs: randomized accept latency (D-006 contract shape) ─────
  int unsigned lfsr;
  function automatic int unsigned rnd();
    lfsr = (lfsr >> 1) ^ (32'hEDB8_8320 & {32{lfsr[0]}});
    rnd  = lfsr;
  endfunction

  // each channel gets its own independent stall stream
  logic [7:0] stall_ds, stall_aj, stall_wj, stall_fj, stall_qj, stall_dj,
              stall_cs, stall_qs;

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

  // ── KVQ AXI-Lite slave stub ──────────────────────────────────────────────
  // Models the engine: after a READ_ADDR write it goes BUSY for a random
  // burst, and STATUS.idle reads 0 until it drains. A walker that writes
  // READ_ADDR without a preceding idle poll is therefore driving a request
  // the real engine would silently DROP — flagged here as an error.
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
    end else begin  // kv_writes accumulates across cases (per-case reset would hide it)
      if (kv_busy_cnt != 0) kv_busy_cnt <= kv_busy_cnt - 1;

      // read channel: STATUS
      kr_v <= kar_v && kar_r;
      if (kar_v && kar_r) begin
        kr_d <= {30'b0, 1'b0, (kv_busy_cnt == 0)};
        if (kv_busy_cnt == 0) kv_saw_idle_poll <= 1'b1;
      end

      // write channel: READ_ADDR
      kb_v <= kaw_v && kaw_r;
      if (kaw_v && kaw_r) begin
        kv_writes <= kv_writes + 1;
        if (!kv_saw_idle_poll) begin
          $display("FAIL: READ_ADDR write with no preceding idle poll (the engine would DROP this read_req)");
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

  // ── D-029 erratum (I-B): the tile's own acceptance rule, replayed ────────
  // Same predicate as tb_walker2_sb's, for the same reason: the scoreboard
  // below is a self-consistent oracle and cannot see a descriptor that is
  // uniformly wrong on both sides. A TRANSCRIPTION of mxe_ctrl.sv's `legal`
  // M/K/N terms reading the SAME symbols (MXE_N/K_MAX from apex_pkg,
  // M_TILE_MAX from mxe_cfg_pkg), never a local restatement. The v1 engine's
  // descriptors are m=1, k<=T_MAX/CFG_D, n<=MXE_N and so pass today — the
  // check is here so a future shaping change cannot quietly stop passing.
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

  // route level: capture on change (rt_* are LEVELS, not handshakes)
  logic [15:0] rt_word, rt_prev;
  logic        rt_seen;
  assign rt_word = {8'd0, rt_kvu, rt_qsrc, rt_rdst, rt_wsrc, rt_asrc,
                    rt_fdst, rt_fsrc};

  always begin
    @(posedge clk);
    if (!rst_n) begin
      rt_prev <= '0; rt_seen <= 1'b0;
    end else if (walk_busy && (walk_phase == WPH_SCORE
                               || walk_phase == WPH_PV)) begin
      // a route LEVEL counts as an emitted ROUTE op only once a phase is
      // actually running — a refused walk establishes no phase and no route
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
      if (kaw_v && kaw_r) chk($sformatf("KVW %0x", kw_d));
      if (ds_valid && ds_ready) begin
        chk($sformatf("DESC %02x %0x %0x %0x %0d %0x %0x %0d",
                      ds_desc.opcode, ds_desc.m_dim, ds_desc.k_dim,
                      ds_desc.n_dim, ds_desc.requant_en, ds_desc.rq_scale,
                      ds_desc.rq_shift, ds_desc.mode_os));
        mxe_desc_legal(ds_desc.m_dim, ds_desc.k_dim,   // D-029 erratum
                       ds_desc.n_dim);                // header note
      end
    end
  end

  // composite-unit error pulses are never expected in a good run
  always begin
    @(posedge clk);
    if (rst_n && comp_err_frame) begin
      $display("FAIL [%s]: composite err_frame (record framing)", cur_case);
      n_errors++;
    end
    if (rst_n && comp_err_stale) begin
      $display("FAIL [%s]: composite err_stale (scale cache miss)", cur_case);
      n_errors++;
    end
  end

  // AXI-Lite contract checks (these signals exist to be CHECKED, not ignored):
  // the walker may only touch STATUS 0x04 and READ_ADDR 0x2C (notes §4 v1
  // scope), and it must never retract bready/rready.
  always begin
    @(posedge clk);
    if (rst_n) begin
      if (kar_v && kar_a != 8'h04) begin
        $display("FAIL: AXIL read of 0x%02h; walker may only read STATUS 0x04",
                 kar_a);
        n_errors++;
      end
      if (kaw_v && kaw_a != 8'h2C) begin
        $display("FAIL: AXIL write to 0x%02h; walker may only write READ_ADDR 0x2C",
                 kaw_a);
        n_errors++;
      end
      if (!kb_r || !kr_r) begin
        $display("FAIL: walker retracted bready/rready (AXI violation)");
        n_errors++;
      end
    end
  end

  // deliberately-unobserved DUT outputs (checked elsewhere or informational)
  logic unused_tb;
  assign unused_tb = &{1'b0, ds_desc[127:68], ds_desc[66]};

  // ── vector replay ────────────────────────────────────────────────────────
  int    fd;
  string line;
  int    expect_refuse;
  int    expect_code;

  task automatic reset_dut();
    begin
      rst_n = 1'b0; walk_en = 1'b0; walk_go = 1'b0; abort_req = 1'b0;
      tile_idle = 1'b1; snp_valid = 1'b0; snp_flush = 1'b0; sq_valid = 1'b0;
      snp_data = '0; snp_addr = '0; snp_last = 1'b0; sq_data = '0;
      for (int i = 0; i < WALK_DESC_WORDS; i++) desc_word[i] = '0;
      repeat (4) @(posedge clk);
      rst_n = 1'b1;
      @(posedge clk);
    end
  endtask

  task automatic push_record(input logic [WALK_SC_AW-1:0] addr, input int n,
                             input logic [15:0] vals []);
    begin
      for (int i = 0; i < n; i++) begin
        @(negedge clk);
        snp_valid = 1'b1;
        snp_data  = vals[i];
        snp_addr  = addr;
        snp_last  = (i == n - 1);
      end
      @(negedge clk);
      snp_valid = 1'b0;
      snp_last  = 1'b0;
    end
  endtask

  task automatic run_walk();
    int guard;
    begin
      @(negedge clk); walk_en = 1'b1; walk_go = 1'b1;
      @(negedge clk); walk_go = 1'b0;
      guard = 0;
      while (walk_busy && guard < 20_000_000) begin
        @(posedge clk); guard++;
      end
      if (guard >= 20_000_000) begin
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
    end else if (walk_err) begin
      saw_err <= 1'b1; saw_code <= walk_err_code;
    end
  end

  initial begin
    if (!$value$plusargs("vectors=%s", vec_file)) begin
      $display("FAIL: +vectors=<file> required"); $finish;
    end
    if (!$value$plusargs("lat_mode=%d", lat_mode)) lat_mode = 1;
    if (!$value$plusargs("bp_mode=%d",  bp_mode))  bp_mode  = 1;
    if (!$value$plusargs("seed=%d",     seed))     seed     = 1;
    lfsr = 32'(seed) | 32'h1;
    n_checks = 0; n_errors = 0; n_cases = 0; n_mxechk = 0;
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
      if (line.substr(0, 0) == "#") begin
        cur_case = line.substr(2, line.len() - 2);
        continue;
      end
      if (line.substr(0, 8) == "DESC_GEOM") begin
        void'($sscanf(line, "DESC_GEOM %h", desc_word[WALK_DW_GEOM]));
      end else if (line.substr(0, 6) == "DESC_RQ") begin
        void'($sscanf(line, "DESC_RQ %h", desc_word[WALK_DW_RQ]));
      end else if (line.substr(0, 8) == "DESC_MASK") begin
        void'($sscanf(line, "DESC_MASK %h", desc_word[WALK_DW_MASK]));
      end else if (line.substr(0, 1) == "SQ") begin
        void'($sscanf(line, "SQ %h", sq_data));
        @(negedge clk); sq_valid = 1'b1; @(negedge clk); sq_valid = 1'b0;
      end else if (line.substr(0, 2) == "REC") begin
        int p; logic [WALK_SC_AW-1:0] a; logic [15:0] v;
        void'($sscanf(line, "REC %h", a));
        p = 9;                                 // "REC xxxx "
        for (int i = 0; i < CFG_D; i++) begin
          void'($sscanf(line.substr(p, p + 3), "%h", v));
          rec_vals[i] = v;
          p += 5;
        end
        push_record(a, CFG_D, rec_vals);
      end else if (line.substr(0, 12) == "EXPECT_REFUSE") begin
        // the argument is the EXPECTED walk_err_e code (1 = WALK_ERR_TIER for
        // the §A-1 grouped-tier case, 2 = WALK_ERR_DESC for the D-029 fmt
        // hardening) — a refusal must fail for the RIGHT reason, or a
        // differently-broken walk could masquerade as the tested refusal
        void'($sscanf(line, "EXPECT_REFUSE %d", expect_code));
        expect_refuse = 1;
      end else if (line.substr(0, 1) == "E ") begin
        exp_q.push_back(line.substr(2, line.len() - 2));
      end else if (line.substr(0, 2) == "END") begin
        n_cases++;
        run_walk();
        if (expect_refuse != 0) begin
          // A refusal must be REFUSED with the expected code and emit
          // nothing (§A-1 for grouped tiers; IB_WALK.md §2.1 for unknown
          // formats) — a generic or different error would let a
          // differently-broken walk masquerade as the tested refusal.
          n_checks++;
          if (!saw_err || int'(saw_code) != expect_code) begin
            $display("FAIL: refusal expected code=%0d, saw err=%0d code=%0d", expect_code, saw_err, saw_code);
            n_errors++;
          end else if (saw_code == WALK_ERR_TIER) begin
            $display("  refusal OK: WALK_ERR_TIER, no emissions");
          end else begin
            $display("  refusal OK: err_code=%0d, no emissions", saw_code);
          end
        end else if (exp_q.size() != 0) begin
          $display("FAIL [%s]: %0d reference emissions never produced (first: '%s')", cur_case, exp_q.size(), exp_q[0]);
          n_errors++;
        end
        exp_q.delete();
        expect_refuse = 0;
        expect_code = 0;
        reset_dut();
      end
    end
    $fclose(fd);

    $display("WALKER RESULT: cases=%0d checks=%0d errors=%0d kvw=%0d mxelegal=%0d",
             n_cases, n_checks, n_errors, kv_writes, n_mxechk);
    if (n_errors == 0 && n_checks > 0) $display("WALKER PASS");
    else                               $display("WALKER FAIL");
    $finish;
  end

  logic u_dbg_sc, u_dbg_sq;
  wire  _unused_dbg = &{1'b0, u_dbg_sc, u_dbg_sq};
endmodule
