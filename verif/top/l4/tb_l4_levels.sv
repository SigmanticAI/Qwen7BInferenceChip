// tb_l4_levels.sv — S4 LEVEL-PROPAGATION probe (the IB-FUEL UNDRIVEN
// lesson, IB_LAYER.md §0.1): CSR readback of a register is NOT evidence the
// level left the module. Every exported LAYER level is driven via the CSR
// window and observed AT ITS CONSUMER — the actual mux outputs / instance
// port nets, via hierarchical references (house precedent: the wide-rms TB
// steers on dut.st) — plus negative controls proving the legacy paths
// reconnect byte-identically when the levels drop.
//
// This is a WIRING probe, not a data-path suite (S5's L4 runs carry the
// full-traffic proof; the unit suites carry the block numerics). It exists
// to make an undriven/unconnected glue net IMPOSSIBLE to ship behind a
// green readback.
//
// Exit: "L4LEVELS RESULT: checks=<n> errors=0 -> PASS" + "L4LEVELS PASS".

module tb_l4_levels;

  logic clk;
  logic rst_n;
  initial begin clk = 0; rst_n = 0; end
  always #5 clk = ~clk;

  // ── DUT: apex_top, tile defaults (CFG_D=64) ──────────────────────────────
  logic [7:0]  csr_addr;
  logic [31:0] csr_wdata, csr_rdata;
  logic        csr_write, csr_read, csr_ready;

  // all non-CSR inputs tied off / outputs open (idle-tile probe)
  apex_top u_dut (
    .clk (clk), .rst_n (rst_n),
    .csr_addr (csr_addr), .csr_wdata (csr_wdata), .csr_write (csr_write),
    .csr_read (csr_read), .csr_rdata (csr_rdata), .csr_ready (csr_ready),
    .kv_awaddr ('0), .kv_awvalid (1'b0), .kv_awready (),
    .kv_wdata ('0), .kv_wvalid (1'b0), .kv_wready (),
    .kv_bresp (), .kv_bvalid (), .kv_bready (1'b0),
    .kv_araddr ('0), .kv_arvalid (1'b0), .kv_arready (),
    .kv_rdata (), .kv_rresp (), .kv_rvalid (), .kv_rready (1'b0),
    .kv_irq (), .kv_evict_needed (), .kv_evict_addr (),
    .ds_valid (1'b0), .ds_ready (), .ds_desc ('0),
    .xw_valid (1'b0), .xw_ready (), .xw_beat ('0),
    .xa_valid (1'b0), .xa_ready (), .xa_x ('0), .xa_last (1'b0),
    .xg_valid (1'b0), .xg_ready (), .xg_gamma ('0),
    .qs_valid (1'b0), .qs_ready (), .qs_data ('0),
    .cs_valid (1'b0), .cs_ready (), .cs_data ('0),
    .fj_valid (1'b0), .fj_ready (), .fj_rows ('0),
    .qj_valid (1'b0), .qj_ready (), .qj_mode (1'b0), .qj_cols ('0),
    .dj_valid (1'b0), .dj_ready (), .dj_cols ('0),
    .lj_valid (1'b0), .lj_ready (), .lj_beats ('0), .lj_lanes ('0),
    .aj_valid (1'b0), .aj_ready (), .aj_op (1'b0), .aj_bank (1'b0),
    .aj_pat ('0), .aj_rows ('0), .aj_nb ('0), .aj_sel ('0),
    .wj_valid (1'b0), .wj_ready (), .wj_op (1'b0), .wj_bank (1'b0),
    .wj_pat ('0), .wj_rows ('0), .wj_nb ('0), .wj_sel ('0),
    .rt_feeder_src (1'b0), .rt_feeder_dst (1'b0), .rt_act_src (1'b0),
    .rt_wgt_src (1'b0), .rt_res_dst ('0), .rt_squant_src (1'b0),
    .rt_kv_user (1'b0), .rt_tip_blk ('0),
    .rt_imp_hi ('0), .rt_imp_lo ('0), .rt_imp_clear (1'b0),
    .fs_valid (), .fs_ready (1'b1), .fs_data (), .fs_last (),
    .ss_valid (), .ss_ready (1'b1), .ss_data (), .ss_last (),
    .td_valid (), .td_ready (1'b1), .td_fp16 (), .td_tier (), .td_blk (),
    .ro_valid (), .ro_ready (1'b1), .ro_beat (),
    .wf_valid (), .wf_ready (1'b1), .wf_req (),   // W-G3 flip: no fmt=1 walk
    .dn_mxe (), .dn_feeder (), .dn_squant (), .dn_scored (), .dn_ser (),
    .dn_astage (), .dn_wstage (), .dn_rms (), .dn_asu (),
    .err_sticky (),
    .dbg_f16_v (), .dbg_f16_data (), .dbg_f16_last (),
    .dbg_sc_v (), .dbg_sc_data (), .dbg_sc_last (),
    .dbg_pr_v (), .dbg_pr_data (), .dbg_pr_last ()
  );

  int n_chk, n_err;
  logic unused_tb_ok;
  assign unused_tb_ok = &{1'b0, csr_ready, rd[31:16]};

  // NEGEDGE-driven CSR tasks: driving csr_* with blocking assigns in the
  // posedge time slot races the DUT's sample (the first cut of this probe
  // did that and chased a phantom bug — same-slot process ordering let the
  // DUT see the strobe an edge "early"). Mid-cycle drives make the strobe
  // span exactly one posedge, race-free.
  task automatic csrw(input logic [7:0] a, input logic [31:0] d);
    begin
      @(negedge clk);
      csr_addr  = a;
      csr_wdata = d;
      csr_write = 1'b1;
      @(negedge clk);
      csr_write = 1'b0;
      @(posedge clk);
    end
  endtask

  // the WALK/ERR read pattern is 1-CYCLE: rdata is valid THE CYCLE AFTER
  // the read strobe and only that cycle — sample inside it (a late sample
  // lands on csr_regs' reserved 0xDEADBEEF; this probe initially did
  // exactly that, which is worth remembering: the F2 host executor must
  // capture LAYER reads with the same 1-cycle discipline as WALK reads)
  task automatic csrr(input logic [7:0] a, output logic [31:0] d);
    begin
      @(negedge clk);                 // set up mid-cycle (race-free)
      csr_addr = a;
      csr_read = 1'b1;
      @(negedge clk);                 // strobe posedge has passed; rdata
      csr_read = 1'b0;                // window is THIS cycle (WALK pattern)
      #1;
      d = csr_rdata;
      @(posedge clk);
    end
  endtask

  task automatic chk(input logic cond, input string what);
    begin
      n_chk++;
      if (!cond) begin
        n_err++;
        $display("L4LEVELS ERR: %s", what);
      end
    end
  endtask

  logic [31:0] rd;
  initial begin
    csr_addr = '0; csr_wdata = '0; csr_write = 0; csr_read = 0;
    n_chk = 0; n_err = 0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (4) @(posedge clk);

    // ── negative control: all levels 0 == legacy wiring ───────────────────
    chk(u_dut.kv_s_tvalid === u_dut.sqf_valid, "legacy S-2 passthrough valid");
    chk(u_dut.sqf_ready === u_dut.kv_s_tready, "legacy S-2 passthrough ready");
    chk(u_dut.wid_m_ready === u_dut.fq_in_ready, "legacy feeder src (widen)");

    // ── l_rope_en -> the S-2 seam mux (consumer side both directions) ─────
    csrw(8'h70, 32'h0000_0001);
    @(posedge clk);
    chk(u_dut.kv_s_tvalid === u_dut.u_rope_row.m_valid, "rope-en: kv_s from rope m_valid");
    chk(u_dut.sqf_ready === u_dut.u_rope_row.s_ready, "rope-en: squant sees rope s_ready");
    chk(u_dut.u_rope_row.s_valid === u_dut.sqf_valid, "rope-en: rope fed by squant");
    csrw(8'h70, 32'h0);
    @(posedge clk);
    chk(u_dut.kv_s_tvalid === u_dut.sqf_valid, "rope-off: passthrough restored");

    // ── l_ser_dst -> serializer-output consumer fanout ────────────────────
    csrw(8'h70, 32'h0000_0004);              // dst=1: layer-deq
    @(posedge clk);
    chk(u_dut.ls_out_ready === u_dut.u_ldeq.ir, "ser_dst=1: ls_out_ready is deq.ir");
    chk(u_dut.u_ldeq.iv === u_dut.ls_out_valid, "ser_dst=1: deq.iv from ls_out");
    csrw(8'h70, 32'h0000_0008);              // dst=2: swiglu
    @(posedge clk);
    chk(u_dut.ls_out_ready === (u_dut.u_swg.gr | u_dut.u_swg.ur),
        "ser_dst=2: ls_out_ready is swiglu gr|ur");
    chk(u_dut.u_swg.gv === u_dut.ls_out_valid, "ser_dst=2: swiglu.gv from ls_out");
    csrw(8'h70, 32'h0);

    // ── l_fsrc_ext -> feeder-source override ──────────────────────────────
    csrw(8'h70, 32'h0000_0010);              // fsrc=1: deq fp32
    @(posedge clk);
    chk(u_dut.fq_in_valid === u_dut.u_ldeq.ov, "fsrc=1: feeder fed by deq.ov");
    chk(u_dut.wid_m_ready === 1'b0, "fsrc=1: widen path disconnected");
    csrw(8'h70, 32'h0000_0020);              // fsrc=2: swiglu-p widened
    @(posedge clk);
    chk(u_dut.fq_in_valid === u_dut.u_swg.pv, "fsrc=2: feeder fed by swiglu.pv");
    csrw(8'h70, 32'h0);
    @(posedge clk);
    chk(u_dut.wid_m_ready === u_dut.fq_in_ready, "fsrc=0: widen path restored");

    // ── l_resid_arm -> deq-output consumer fanout ─────────────────────────
    csrw(8'h70, 32'h0000_0040);              // arm: deq -> residual
    @(posedge clk);
    chk(u_dut.u_resid.iv === u_dut.u_ldeq.ov, "resid-arm: residual fed by deq");
    chk(u_dut.ldq_or === u_dut.u_resid.ir, "resid-arm: deq sees residual ready");
    csrw(8'h70, 32'h0);

    // ── phase RAM: PTR/DATA write -> the rope consumer read register ──────
    csrw(8'h74, 32'h1000_0000);              // bank=1 (q), idx 0
    csrw(8'h78, 32'h0000_1ABC);
    csrw(8'h70, 32'h0000_0002);              // bank sel q (rope idle: addr 0)
    repeat (2) @(posedge clk);
    chk(u_dut.rr_ph_data === 14'h1ABC, "phase-q write reaches rope ph_data");
    csrw(8'h74, 32'h0000_0000);              // bank=0 (K), pos 0 pair 0
    csrw(8'h78, 32'h0000_2DEF);
    csrw(8'h70, 32'h0);                      // bank sel K, pos 0
    repeat (2) @(posedge clk);
    chk(u_dut.rr_ph_data === 14'h2DEF, "phase-K write reaches rope ph_data");

    // ── residual row RAM: DATA write -> RDATA readback (+RPTR auto-inc) ───
    csrw(8'h74, 32'h2000_0005);              // bank=2 (residual), idx 5
    csrw(8'h78, 32'h0000_ABCD);
    csrw(8'h84, 32'h0000_0005);
    csrr(8'h88, rd);
    chk(rd[15:0] === 16'hABCD, "residual row write -> RDATA");
    csrr(8'h88, rd);                         // auto-inc read side effect
    chk(u_dut.l_rptr_q === 16'd7, "RPTR auto-inc on RDATA reads");

    // ── LAYER_JOB -> unit job ports; GRADE refusal -> STATUS code + W1C ───
    csrw(8'h8C, 32'h3DCC_CCCD);              // comp_a: ungraded 0.1f
    csrw(8'h7C, 32'h0000_0001);              // unit=0 deq, cols=1
    repeat (3) @(posedge clk);
    csrr(8'h80, rd);
    chk(rd[8] === 1'b1, "GRADE refusal sets layer_err sticky");
    chk(rd[12:9] === 4'd2, "err code == GRADE");
    csrw(8'h80, 32'h0000_0100);              // W1C
    repeat (2) @(posedge clk);
    csrr(8'h80, rd);
    chk(rd[8] === 1'b0, "sticky W1C clears");
    // legal residual job push: jb_valid visible at the unit port + busy
    csrw(8'h7C, 32'h0000_2004);              // unit=2 residual, cols=4
    // the held jb_valid clears on the SAME edge the unit accepts, so the
    // durable acceptance evidence is the unit's own state, not the level
    chk(u_dut.u_resid.jb_cols === 12'd4, "JOB cols reach resid port");
    @(posedge clk);
    #1;                               // post-NBA: the unit's accept has landed
    chk(u_dut.u_resid.busy === 1'b1, "JOB push accepted at resid (busy)");
    repeat (2) @(posedge clk);
    csrr(8'h80, rd);
    chk(rd[2] === 1'b1, "residual busy visible in STATUS");
    // residual geometry reject: cols=0 push while... (fresh unit: cols=0)
    // (busy unit ignores jb; this probe stops at the accepted-job evidence)

    if (n_err == 0) begin
      $display("L4LEVELS RESULT: checks=%0d errors=0 -> PASS", n_chk);
      $display("L4LEVELS PASS");
      $finish;
    end else begin
      $fatal(1, "L4LEVELS FAIL: %0d/%0d checks failed", n_err, n_chk);
    end
  end

  initial begin
    #10_000_000;
    $fatal(1, "L4LEVELS FAIL: watchdog");
  end

endmodule
