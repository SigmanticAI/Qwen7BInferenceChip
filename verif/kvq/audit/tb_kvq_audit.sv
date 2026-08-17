// tb_kvq_audit.sv — adversarial re-audit TB for the D-020-fixed kvq_engine
// (verif/kvq/audit). Attacks the B-1..B-4 fixes specifically:
//
//  * +mode=0 (reset attack): N_RST randomized CTRL.soft_reset injections with
//    hierarchical landing-state binning (dut.state sampled in the ctrl_reset
//    cycle), across parked / mid-token / mid-group / pending-beat /
//    beat-coincident conditions plus deterministic 1-cycle-state collisions
//    (ST_STORE via cqv_out_valid, ST_RLOAD/ST_RWAIT via back-to-back AXI
//    writes, ST_KFEED via final-beat+CTRL coincidence). EVERY reset is
//    followed IMMEDIATELY (no quiesce) by a full clean value job checked
//    bit-exact vs the golden pool (record peek + fp32 readback); key-ish
//    events and every 3rd event additionally run a full clean KEY-GROUP job.
//    D-020 contract checks after every reset:
//      - landing outside ST_OUTPUT/ST_OFLUSH: FSM in ST_IDLE within 6 cycles,
//        ZERO m_axis beats leaked after the reset (read victims: 0 beats if
//        the read had not produced its first beat, else the full D);
//      - landing in ST_OUTPUT/ST_OFLUSH: the burst COMPLETES — exactly D
//        beats, tlast on beat D-1, every beat bit-exact, STATUS.idle=0 while
//        parked, FSM never leaves the output phases early (no retraction —
//        the §5 SVA pack and the procedural stability monitor stay armed).
//      - continuous whitebox IDLE-LIE monitor: the STATUS.idle expression
//        (idle && !cqv_busy && !kp_busy) may never be true while the FSM is
//        outside ST_IDLE or an m_axis beat is live.
//  * +mode=1 (directed): runs every pool job once — carries the -0.0
//    outlier/non-outlier directed content (B-4).
//  * A2/D-026 additions (mode=0, KEYG only, after the attack loop):
//      - bank-survival bucket: commit a key group, verify records + the
//        persistent scale-bank row (dut.g_key.u_bank_store.mem), pulse
//        ctrl_reset, then the SAME records must still read back bit-exact
//        fp32 AND the bank row must be untouched — the dedicated guard
//        against wiring scale_bank_store to dp_clear.
//      - SB_OVWR bucket: commit >SCALE_SETS groups (1-token flushed
//        partials), assert IRQ_STATUS[1] goes sticky, drives irq under
//        IRQ_MASK[1], W1C-clears, and re-arms on the next overwrite.
//    Key records are scored D-026 style: exp_rec holds the golden image with
//    ssid0=0; do_kpeek checks tag bit0==1 and tag[7:1]==the LIVE ssid
//    (dut.alloc_ptr sampled before the commit), then compares the record
//    with tag[7:1] masked. Bank rows are scored against the golden
//    pack_key_records rows (exp_bank.f16.hex, outlier lanes pinned 0).
//
// Vector pool + expectations: gen_audit_vectors.py (golden/apex_golden is the
// arbiter, D-001/D-013/D-026). Pool layout convention is computed, not
// scripted — see the generator header. TB discipline: negedge stimulus,
// posedge observation, $fatal watchdog, cycle-stamped FAILs (V0/sb lineage).

`timescale 1ns/1ps

module tb_kvq_audit;

  // ── configuration (verilator -G) ───────────────────────────────────────────
  parameter int D        = 16;
  parameter int TIER     = 1;
  parameter int G        = 4;
  parameter int K_OUT    = 0;
  parameter int DEPTH    = 64;
  parameter int NV       = 24;    // value jobs in the pool
  parameter int NK       = 8;     // key-group jobs in the pool
  parameter     MASKFILE = ".";
  parameter     CFGNAME  = "cfg";

  // ── record-layout mirror (D-026/§4) ─────────────────────────────────────────
  // Mirrors golden cq_codec.key_rec_raw_bits / val_rec_raw_bits / sram_row_bits
  // (and kvq_engine KEY_REC_RAW/VAL_REC_RAW/REC_RAW) EXACTLY — the generator
  // imports the golden functions; this SV copy exists only to size exp_rec.
  localparam int BPV     = (TIER == 0) ? 8 : 4;
  localparam int KEYG    = (TIER != 0) ? 1 : 0;
  localparam int VAL_RAW = 8 + 16 + D * BPV;
  localparam int KEY_RAW = 8 + K_OUT * 16 + D * 4;  // D-026: tag + k lanes + D×4
  localparam int REC_RAW = (KEYG == 1 && KEY_RAW > VAL_RAW) ? KEY_RAW : VAL_RAW;
  localparam int SRAMW   = 64 * ((REC_RAW + 63) / 64);
  localparam int AW      = $clog2(DEPTH);
  localparam int SSETS   = 4;   // kvq_engine SCALE_SETS default (not overridden)

  // pool index convention (generator header)
  localparam int RV_TOK   = NV + NK * G;      // read-victim job (tok == ridx)
  localparam int VT_BASE  = NV + NK * G + 1;  // victim tokens (G of them)

  // address map (disjoint: clean jobs never touch victim addresses)
  localparam int KB0      = 0;                // clean key-group base A
  localparam int KB1      = G;                // clean key-group base B
  localparam int VICT     = 2 * G;            // victim ops base
  localparam int VCLEAN_N = 6;
  localparam int VCLEAN   = DEPTH - 1 - VCLEAN_N;  // clean value addrs
  localparam int RV_ADDR  = DEPTH - 1;        // pre-written read-victim record

  // FSM state codes (kvq_engine.sv, whitebox targeting)
  localparam [3:0] ST_IDLE = 4'd0,  ST_COLLECT = 4'd1, ST_COMPRESS = 4'd2,
                   ST_STORE = 4'd3, ST_RLOAD = 4'd4,   ST_RWAIT = 4'd5,
                   ST_OUTPUT = 4'd6, ST_KCOLLECT = 4'd7, ST_KFEED = 4'd8,
                   ST_KACCEPT = 4'd9, ST_KEMIT = 4'd10, ST_OFLUSH = 4'd11;

  localparam [7:0] REG_CTRL = 8'h00, REG_STATUS = 8'h04,
                   REG_WA = 8'h28, REG_RA = 8'h2C,
                   REG_IRQ_MASK = 8'h34, REG_IRQ_STATUS = 8'h38;

  // ── clock / cycles ──────────────────────────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  // ── DUT ─────────────────────────────────────────────────────────────────────
  logic [7:0]  awaddr;
  logic        awvalid;
  wire         awready;
  logic [31:0] wdata;
  logic        wvalid;
  wire         wready;
  wire  [1:0]  bresp;
  wire         bvalid;
  logic        bready;
  logic [7:0]  araddr;
  logic        arvalid;
  wire         arready;
  wire  [31:0] rdata;
  wire  [1:0]  rresp;
  wire         rvalid;
  logic        rready;

  logic [15:0] s_tdata;
  logic        s_tvalid;
  wire         s_tready;
  logic        s_tlast;
  logic        s_tuser;
  wire  [31:0] m_tdata;
  wire         m_tvalid;
  logic        m_tready;
  wire         m_tlast;
  logic        flush_r;
  wire         irq_w;
  wire         evict_needed;
  wire [AW-1:0] evict_addr;

  wire dut_mask_valid;  // D-027 stage-4 port (sunk; the mask suite checks it)
  wire _tb_unused_ok = &{1'b0, dut_mask_valid, awready, wready, bresp, bvalid, arready, rresp,
                         rvalid, irq_w, evict_needed, evict_addr,
                         ST_COMPRESS[0]};

  kvq_engine #(
    .VECTOR_DIM(D), .TIER(TIER), .KEY_GROUP(G), .OUTLIER_K(K_OUT),
    .SCALE_WIDTH(16), .SRAM_DEPTH(DEPTH), .COORD_WIDTH(16), .OUT_WIDTH(32),
    .MASK_FILE(MASKFILE)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .axil_awaddr(awaddr), .axil_awvalid(awvalid), .axil_awready(awready),
    .axil_wdata(wdata), .axil_wvalid(wvalid), .axil_wready(wready),
    .axil_bresp(bresp), .axil_bvalid(bvalid), .axil_bready(bready),
    .axil_araddr(araddr), .axil_arvalid(arvalid), .axil_arready(arready),
    .axil_rdata(rdata), .axil_rresp(rresp), .axil_rvalid(rvalid),
    .axil_rready(rready),
    .s_axis_kv_tdata(s_tdata), .s_axis_kv_tvalid(s_tvalid),
    .s_axis_kv_tready(s_tready), .s_axis_kv_tlast(s_tlast),
    .s_axis_kv_tuser(s_tuser),
    .m_axis_kv_tdata(m_tdata), .m_axis_kv_tvalid(m_tvalid),
    .m_axis_kv_tready(m_tready), .m_axis_kv_tlast(m_tlast),
    .flush_req(flush_r), .irq(irq_w),
    .evict_needed(evict_needed), .evict_addr(evict_addr),
    .mask_valid(dut_mask_valid)
  );

  // ── plusargs ────────────────────────────────────────────────────────────────
  int unsigned mode         = 0;   // 0 = reset attack, 1 = directed pool run
  int unsigned n_rst        = 40;
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned seed         = 32'hA0D1_2026;
  int unsigned watchdog_cyc = 12_000_000;
  initial begin
    void'($value$plusargs("mode=%d", mode));
    void'($value$plusargs("n_rst=%d", n_rst));
    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
  end

  // ── watchdog (MANDATORY) ────────────────────────────────────────────────────
  initial begin
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: %s did not finish within %0d cycles", CFGNAME,
           watchdog_cyc);
  end

  initial begin
    if ($test$plusargs("dump")) begin
      $dumpfile("dump.fst");
      $dumpvars(0, tb_kvq_audit);
    end
  end

  // ── stimulus / expectation pool ─────────────────────────────────────────────
  logic [15:0]      stim     [0:16383];
  logic [SRAMW-1:0] exp_rec  [0:255];
  logic [31:0]      exp_hat  [0:16383];
  logic [D*16-1:0]  exp_bank [0:63];    // D-026: golden bank row per key job

  // ── D-026 scale-bank mirror ────────────────────────────────────────────────
  // dut.g_key only elaborates when KEYG; a flat combinational mirror keeps the
  // hierarchical reference out of TIER-0 builds (tasks may reference bank_flat
  // unconditionally).
  wire [SSETS*D*16-1:0] bank_flat;
  generate
    if (KEYG == 1) begin : g_bank_mirror
      for (genvar s = 0; s < SSETS; s++) begin : g_set
        assign bank_flat[s*D*16 +: D*16] = dut.g_key.u_bank_store.mem[s];
      end
    end else begin : g_bank_zero
      assign bank_flat = '0;
    end
  endgenerate

  // ── m_axis collector (posedge — exactly what the DUT commits) ──────────────
  logic [31:0] rq_data [$];
  bit          rq_last [$];
  int          burst_acc;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      burst_acc <= 0;
    end else if (m_tvalid && m_tready) begin
      rq_data.push_back(m_tdata);
      rq_last.push_back(m_tlast);
      burst_acc <= m_tlast ? 0 : burst_acc + 1;
    end
  end

  // ── m_axis backpressure driver + park control ──────────────────────────────
  logic        blk_arm;      // park the burst once burst_acc >= blk_thresh
  int          blk_thresh;
  logic [31:0] lfsr;
  always @(negedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lfsr     <= seed;
      m_tready <= 1'b1;
    end else begin
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      if (blk_arm && burst_acc >= blk_thresh) begin
        m_tready <= 1'b0;
      end else begin
        unique case (bp_mode)
          0:       m_tready <= 1'b1;
          1:       m_tready <= |lfsr[1:0];
          default: m_tready <= (lfsr[3:0] == 4'h0);
        endcase
      end
    end
  end

  // ── m_axis stability monitor (procedural mirror of the SVA rule) ───────────
  int          viol_stable;
  logic        pend;
  logic [31:0] pend_d;
  logic        pend_l;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pend   <= 1'b0;
      pend_d <= '0;
      pend_l <= 1'b0;
    end else begin
      if (pend && (!m_tvalid || (m_tdata !== pend_d) || (m_tlast !== pend_l)))
        begin
          viol_stable <= viol_stable + 1;
          $display("STABILITY-VIOL cyc=%0d: m_axis retracted/changed under backpressure (valid=%b data=%08h/%08h last=%b/%b)",
                   cyc, m_tvalid, m_tdata, pend_d, m_tlast, pend_l);
        end
      pend   <= (m_tvalid && !m_tready);
      pend_d <= m_tdata;
      pend_l <= m_tlast;
    end
  end

  // ── continuous IDLE-LIE monitor (whitebox, B-2 regression guard) ───────────
  // STATUS.idle = idle && !cqv_busy && !kp_busy (kvq_engine.sv). If that
  // expression is ever true while the FSM is outside ST_IDLE or an m_axis
  // beat is still live, a host polling STATUS would be lied to.
  int viol_idle;
  always @(negedge clk) begin
    if (dut.idle && !dut.cqv_busy && !dut.kp_busy
        && (dut.state != ST_IDLE || m_tvalid)) begin
      viol_idle <= viol_idle + 1;
      $display("IDLE-LIE cyc=%0d: STATUS-visible idle while FSM state=%0d m_tvalid=%b",
               cyc, dut.state, m_tvalid);
    end
  end

  // ── landing-state monitor: bins the FSM state in every ctrl_reset cycle ────
  int       cov_rst_total;
  int       cov_land [0:11];
  int       cov_pend_beat, cov_mid_tok, cov_mid_grp, cov_beat_collide;
  int       cov_col_store, cov_col_rload, cov_col_rwait, cov_col_kfeed;
  bit       land_seen;
  logic [3:0] land_state;
  always @(negedge clk) begin
    if (dut.ctrl_reset) begin
      land_state <= dut.state;
      land_seen  <= 1'b1;
      cov_rst_total <= cov_rst_total + 1;
      cov_land[dut.state] <= cov_land[dut.state] + 1;
      if ((dut.state == ST_OUTPUT || dut.state == ST_OFLUSH)
          && m_tvalid && !m_tready) cov_pend_beat <= cov_pend_beat + 1;
      if (dut.state == ST_COLLECT || dut.state == ST_KCOLLECT)
        cov_mid_tok <= cov_mid_tok + 1;
      if (dut.state == ST_KFEED || dut.state == ST_KACCEPT
          || dut.state == ST_KEMIT
          || (dut.state == ST_KCOLLECT && dut.grp_tok_cnt != '0))
        cov_mid_grp <= cov_mid_grp + 1;
      if (s_tvalid && s_tready) cov_beat_collide <= cov_beat_collide + 1;
      if (dut.state == ST_STORE) cov_col_store <= cov_col_store + 1;
      if (dut.state == ST_RLOAD) cov_col_rload <= cov_col_rload + 1;
      if (dut.state == ST_RWAIT) cov_col_rwait <= cov_col_rwait + 1;
      if (dut.state == ST_KFEED) cov_col_kfeed <= cov_col_kfeed + 1;
    end
  end

  // ── TB RNG / bookkeeping ────────────────────────────────────────────────────
  int unsigned rng_state;
  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction

  int checks;
  int fails;
  int printed;
  localparam int MAXPRINT = 40;

  task automatic note_fail(input string msg);
    fails++;
    if (printed < MAXPRINT) begin
      printed++;
      $display("FAIL cyc=%0d: %s", cyc, msg);
    end else if (printed == MAXPRINT) begin
      printed++;
      $display("(further FAIL prints suppressed)");
    end
  endtask

  int cov_drain_exact, cov_clean_v, cov_clean_k, cov_miss;
  int cov_reads, cov_peeks;
  int cov_bank_peeks, cov_bank_rst;        // D-026 buckets
  int cov_sb_sticky, cov_sb_w1c;
  int vclean_ctr, kclean_ctr;

  // ── AXI-Lite tasks (negedge-driven, sb lineage) ─────────────────────────────
  task automatic awrite(input [7:0] a, input [31:0] dv);
    @(negedge clk);
    awaddr = a; wdata = dv; awvalid = 1; wvalid = 1;
    @(negedge clk);
    awvalid = 0; wvalid = 0;
  endtask

  task automatic aread(input [7:0] a, output [31:0] dv);
    @(negedge clk);
    araddr = a; arvalid = 1; rready = 1;
    @(negedge clk);
    arvalid = 0;
    @(negedge clk);
    dv = rdata; rready = 0;
  endtask

  task automatic wait_idle(input int max_polls, input string who);
    int p;
    logic [31:0] st;
    p = 0; st = 0;
    while (((st & 32'h1) == 32'h0) && p < max_polls) begin
      aread(REG_STATUS, st);
      p++;
    end
    checks++;
    if ((st & 32'h1) == 32'h0)
      note_fail($sformatf("%s: STATUS.idle never set after %0d polls", who,
                          max_polls));
  endtask

  // ── stream drivers ──────────────────────────────────────────────────────────
  function automatic int unsigned gap_len();
    unique case (stall_mode)
      0:       gap_len = 0;
      1:       gap_len = ((rnd() & 32'd1) != 0) ? (rnd() % 32'd4) : 0;
      default: gap_len = ((rnd() % 32'd8) == 0) ? (rnd() % 32'd24)
                                                : (rnd() % 32'd3);
    endcase
  endfunction

  task automatic stream_tok(input int tok, input bit isval);
    int d_, g_, gp;
    d_ = 0; g_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      s_tdata  = stim[tok * D + d_];
      s_tvalid = 1;
      s_tuser  = isval;
      s_tlast  = (d_ == D - 1);
      if (s_tready) begin
        d_ = d_ + 1;
        if (stall_mode != 0 && d_ < D) begin
          gp = int'(gap_len());
          if (gp > 0) begin
            @(negedge clk);
            s_tvalid = 0;
            repeat (gp) @(negedge clk);
          end
        end
      end
      g_++;
      if (g_ > 200000) begin
        note_fail($sformatf("stream_tok tok=%0d: beat %0d never accepted", tok, d_));
        break;
      end
    end
    @(negedge clk);
    s_tvalid = 0; s_tlast = 0;
  endtask

  task automatic stream_partial(input int tok, input bit isval, input int nbeats);
    int d_, g_;
    d_ = 0; g_ = 0;
    while (d_ < nbeats && g_ < 200000) begin
      @(negedge clk);
      s_tdata = stim[tok * D + d_]; s_tvalid = 1; s_tuser = isval; s_tlast = 0;
      if (s_tready) d_ = d_ + 1;
      g_++;
    end
    @(negedge clk);
    s_tvalid = 0;
  endtask

  task automatic wait_tready(input string who);
    int g_;
    g_ = 0;
    while (!s_tready && g_ < 500000) begin
      @(negedge clk);
      g_++;
    end
    checks++;
    if (!s_tready) note_fail({who, ": s_axis_kv_tready never re-asserted"});
  endtask

  task automatic pulse_flush();
    @(negedge clk);
    flush_r = 1;
    @(negedge clk);
    flush_r = 0;
  endtask

  // ── golden-pool checkers ────────────────────────────────────────────────────
  task automatic do_read(input int addr, input int hidx, input string who);
    int g_;
    awrite(REG_RA, addr);
    g_ = 0;
    while (rq_data.size() < D && g_ < 400000) begin
      @(negedge clk);
      g_++;
    end
    checks++;
    if (rq_data.size() != D) begin
      note_fail($sformatf("%s R addr=%0d: got %0d beats, expected %0d", who,
                          addr, rq_data.size(), D));
    end else begin
      for (int i = 0; i < D; i++) begin
        checks++;
        if (rq_data[i] !== exp_hat[hidx * D + i])
          note_fail($sformatf("%s R addr=%0d beat=%0d: got %08h exp %08h (hidx=%0d)",
                              who, addr, i, rq_data[i], exp_hat[hidx * D + i],
                              hidx));
        checks++;
        if (rq_last[i] !== bit'(i == D - 1))
          note_fail($sformatf("%s R addr=%0d beat=%0d: tlast=%b wrong", who,
                              addr, i, rq_last[i]));
      end
    end
    repeat (6) @(negedge clk);
    checks++;
    if (rq_data.size() > D)
      note_fail($sformatf("%s R addr=%0d: %0d EXTRA beats after tlast", who,
                          addr, rq_data.size() - D));
    rq_data.delete();
    rq_last.delete();
    cov_reads++;
  endtask

  task automatic do_peek(input int addr, input int ridx, input string who);
    logic [SRAMW-1:0] rec;
    rec = dut.u_sram.mem[addr % DEPTH];
    checks++;
    if (rec !== exp_rec[ridx])
      note_fail($sformatf("%s P addr=%0d ridx=%0d: record mismatch\n  got %0h\n  exp %0h",
                          who, addr, ridx, rec, exp_rec[ridx]));
    cov_peeks++;
  endtask

  // D-026 key-record peek: tag bit0 must be 1 and tag[7:1] must equal the
  // LIVE ssid (commit-time allocator) — never match the whole byte 8'h01; the
  // golden image (exp_rec, packed with ssid0=0) is compared with tag[7:1]
  // masked (that masked image pins the k fp16 lanes, all D code nibbles incl.
  // the outlier sentinels 4'd1, and the pad).
  task automatic do_kpeek(input int addr, input int ridx, input int ssid,
                          input string who);
    logic [SRAMW-1:0] rec, body;
    rec = dut.u_sram.mem[addr % DEPTH];
    checks++;
    if (rec[0] !== 1'b1 || rec[7:1] !== 7'(ssid))
      note_fail($sformatf("%s KP addr=%0d ridx=%0d: tag=%02h exp {ssid=%0d,1'b1}",
                          who, addr, ridx, rec[7:0], ssid));
    body = rec;
    body[7:1] = '0;
    checks++;
    if (body !== exp_rec[ridx])
      note_fail($sformatf("%s KP addr=%0d ridx=%0d: record body mismatch\n  got %0h\n  exp %0h",
                          who, addr, ridx, body, exp_rec[ridx]));
    cov_peeks++;
  endtask

  // D-026 bank-row peek: the persistent scale set the group committed into
  // (dut.g_key.u_bank_store.mem via bank_flat) vs the golden pack_key_records
  // row — nk keep-channel group scales + k outlier lanes pinned 16'h0000.
  task automatic do_bankpeek(input int set_i, input int kidx, input string who);
    logic [D*16-1:0] row, expr;
    row  = bank_flat[set_i*D*16 +: D*16];
    expr = exp_bank[kidx];
    for (int c = 0; c < D; c++) begin
      checks++;
      if (row[c*16 +: 16] !== expr[c*16 +: 16])
        note_fail($sformatf("%s B set=%0d kjob=%0d ch=%0d: scale %04h exp %04h",
                            who, set_i, kidx, c, row[c*16 +: 16],
                            expr[c*16 +: 16]));
    end
    cov_bank_peeks++;
  endtask

  // clean value job i (pool convention: tok == ridx == hidx == i)
  task automatic clean_vjob(input int i, input string who);
    int addr;
    addr = VCLEAN + (vclean_ctr % VCLEAN_N);
    vclean_ctr++;
    awrite(REG_WA, addr);
    stream_tok(i, 1'b1);
    wait_idle(60000, {who, " vjob"});
    do_peek(addr, i, who);
    do_read(addr, i, who);
    cov_clean_v++;
  endtask

  // clean key-group job k (pool: tokens NV+k*G.., ridx NV+k*G+j). D-026: the
  // group's ssid is the commit-time allocator value — sample dut.alloc_ptr
  // while idle, BEFORE streaming (this job performs the only commit between
  // sample and check).
  task automatic clean_kjob(input int k, input string who);
    int base, ssid;
    base = (kclean_ctr % 2 == 0) ? KB0 : KB1;
    kclean_ctr++;
    ssid = int'(dut.alloc_ptr);
    awrite(REG_WA, base);
    for (int j = 0; j < G; j++) stream_tok(NV + k * G + j, 1'b0);
    wait_idle(600000, {who, " kjob"});
    for (int j = 0; j < G; j++) begin
      do_kpeek(base + j, NV + k * G + j, ssid, who);
      do_read(base + j, NV + k * G + j, who);
    end
    do_bankpeek(ssid, k, who);
    cov_clean_k++;
  endtask

  // ── D-026 bank-survival reset attack ────────────────────────────────────────
  // Commit + fully verify a key group, pulse ctrl_reset (soft reset), then the
  // SAME records must still read back bit-exact fp32 and the bank row must be
  // untouched. This is THE dedicated guard against wiring scale_bank_store to
  // dp_clear: the record codes survive in the SRAM either way, but a cleared
  // bank row would dequant every keep channel with scale 0.
  task automatic bank_reset_attack(input int k, input string who);
    int base, ssid;
    base = (kclean_ctr % 2 == 0) ? KB0 : KB1;
    kclean_ctr++;
    ssid = int'(dut.alloc_ptr);
    awrite(REG_WA, base);
    for (int j = 0; j < G; j++) stream_tok(NV + k * G + j, 1'b0);
    wait_idle(600000, {who, " bank-commit"});
    for (int j = 0; j < G; j++) begin
      do_kpeek(base + j, NV + k * G + j, ssid, who);
      do_read(base + j, NV + k * G + j, who);
    end
    do_bankpeek(ssid, k, who);
    // the attack: soft reset (lands ST_IDLE; counted by the landing monitor)
    awrite(REG_CTRL, 32'h3);
    repeat (2) @(negedge clk);
    wait_idle(100, {who, " post-rst"});
    do_bankpeek(ssid, k, {who, " post-rst"});
    for (int j = 0; j < G; j++) begin
      do_kpeek(base + j, NV + k * G + j, ssid, {who, " post-rst"});
      do_read(base + j, NV + k * G + j, {who, " post-rst"});
    end
    cov_bank_rst++;
  endtask

  // ── D-026 SB_OVWR check (IRQ_STATUS[1] sticky + W1C + irq line) ─────────────
  // Commits 1-token flushed partial groups into the victim address range; 5
  // commits guarantee at least one lands on a still-live set (SCALE_SETS=4)
  // regardless of how many groups committed before, raising SB_OVWR.
  task automatic sb_ovwr_commit(input int i, input string who);
    awrite(REG_WA, VICT);
    stream_tok(VT_BASE + (i % G), 1'b0);
    pulse_flush();
    wait_idle(600000, {who, " ovwr-commit"});
  endtask

  task automatic sb_ovwr_check(input string who);
    logic [31:0] r32;
    // W1C any SB_OVWR raised by earlier group commits; verify it cleared
    awrite(REG_IRQ_STATUS, 32'h2);
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== 32'h0)
      note_fail({who, ": IRQ_STATUS[1] not W1C-clearable before the probe"});
    for (int i = 0; i < 5; i++) sb_ovwr_commit(i, who);
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== 32'h2)
      note_fail({who, ": SB_OVWR not raised after 5 commits (>SCALE_SETS live sets)"});
    else cov_sb_sticky++;
    // sticky must drive irq once unmasked
    awrite(REG_IRQ_MASK, 32'h2);
    @(negedge clk);
    checks++;
    if (irq_w !== 1'b1)
      note_fail({who, ": irq low with IRQ_MASK[1]=1 and SB_OVWR set"});
    // W1C drops the sticky bit AND the irq line
    awrite(REG_IRQ_STATUS, 32'h2);
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== 32'h0)
      note_fail({who, ": SB_OVWR did not W1C-clear"});
    else cov_sb_w1c++;
    checks++;
    if (irq_w !== 1'b0)
      note_fail({who, ": irq still high after SB_OVWR W1C"});
    // re-arms sticky on the next overwrite (all sets are live by now)
    sb_ovwr_commit(5, who);
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== 32'h2)
      note_fail({who, ": SB_OVWR did not re-arm on the next live-set commit"});
    awrite(REG_IRQ_STATUS, 32'h2);
    awrite(REG_IRQ_MASK, 32'h0);
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== 32'h0)
      note_fail({who, ": SB_OVWR not clear at probe exit"});
  endtask

  // ── hard-reset rescue (only on FAIL paths) ─────────────────────────────────
  task automatic clear_tb_drives();
    s_tvalid = 0; s_tlast = 0; s_tuser = 0; flush_r = 0;
    awvalid = 0; wvalid = 0; arvalid = 0; rready = 0;
    blk_arm = 0; blk_thresh = 0;
  endtask

  task automatic do_hard_reset(input string who);
    logic [31:0] r32;
    rst_n = 0;
    clear_tb_drives();
    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (2) @(negedge clk);
    rq_data.delete();
    rq_last.delete();
    awrite(REG_CTRL, 32'h2);
    aread(REG_STATUS, r32);
    checks++;
    if ((r32 & 32'h1) !== 32'h1) note_fail({who, ": not idle after hard reset"});
    // the read-victim record was lost with the SRAM: restore it
    awrite(REG_WA, RV_ADDR);
    stream_tok(RV_TOK, 1'b1);
    wait_idle(60000, {who, " RV restore"});
  endtask

  // ── deterministic 1-cycle-state collisions ─────────────────────────────────
  // ST_STORE: the CTRL write is launched at the negedge of the LAST compress
  // cycle (state==COMPRESS && cqv_out_valid), so ctrl_reset is high exactly in
  // the single ST_STORE cycle.
  task automatic store_collide();
    bit hit;
    int t;
    t = 0;
    while (!(dut.state == ST_COMPRESS && dut.cqv_out_valid) && t < 200000) begin
      @(negedge clk);
      t++;
      if (dut.state == ST_IDLE && t > 10) break;   // op finished — drift
    end
    awaddr = REG_CTRL; wdata = 32'h3; awvalid = 1; wvalid = 1;
    @(negedge clk);
    awvalid = 0; wvalid = 0;
    // this negedge is inside the STORE/ctrl_reset collision cycle
    hit = (dut.state === ST_STORE) && dut.ctrl_reset;
    if (!hit)
      $display("NOTE cyc=%0d: STORE collision missed (state=%0d) — timing drift",
               cyc, dut.state);
    @(negedge clk);
  endtask

  // ST_RLOAD / ST_RWAIT: READ_ADDR write then CTRL write, back-to-back for
  // RLOAD, one bubble later for RWAIT.
  task automatic rload_collide(input int addr, input int rwait);
    bit hit;
    @(negedge clk);
    awaddr = REG_RA; wdata = 32'(addr); awvalid = 1; wvalid = 1;
    @(negedge clk);
    if (rwait != 0) begin
      awvalid = 0; wvalid = 0;
      @(negedge clk);
      awaddr = REG_CTRL; wdata = 32'h3; awvalid = 1; wvalid = 1;
    end else begin
      awaddr = REG_CTRL; wdata = 32'h3;   // valids stay high: back-to-back
    end
    @(negedge clk);
    awvalid = 0; wvalid = 0;
    // this negedge is inside the collision cycle
    hit = dut.ctrl_reset && ((dut.state === ST_RLOAD) || (dut.state === ST_RWAIT));
    if (!hit)
      $display("NOTE cyc=%0d: RLOAD/RWAIT collision missed (state=%0d) — drift",
               cyc, dut.state);
    @(negedge clk);
  endtask

  // ST_KFEED: final key beat and the CTRL write accepted at the SAME posedge.
  task automatic kfeed_collide(input int tok);
    int d_;
    d_ = 0;
    while (d_ < D - 1) begin
      @(negedge clk);
      s_tdata = stim[tok * D + d_]; s_tvalid = 1; s_tuser = 0; s_tlast = 0;
      if (s_tready) d_ = d_ + 1;
    end
    @(negedge clk);
    s_tdata = stim[tok * D + D - 1]; s_tvalid = 1; s_tlast = 1;
    awaddr = REG_CTRL; wdata = 32'h3; awvalid = 1; wvalid = 1;
    @(negedge clk);
    s_tvalid = 0; s_tlast = 0; awvalid = 0; wvalid = 0;
    if (!(dut.ctrl_reset && dut.state === ST_KFEED))
      $display("NOTE cyc=%0d: KFEED collision missed (state=%0d) — drift",
               cyc, dut.state);
    @(negedge clk);
  endtask

  // beat-coincident reset: CTRL write timed so ctrl_reset is high in a cycle
  // where an s_axis beat is being ACCEPTED (D-020: that beat is discarded).
  task automatic beat_collide(input int tok, input bit isval);
    int k, d_;
    k  = 1 + int'(rnd() % (D - 2));   // reset coincides with beat k's accept
    d_ = 0;
    while (d_ < k) begin
      @(negedge clk);
      s_tdata = stim[tok * D + d_]; s_tvalid = 1; s_tuser = isval; s_tlast = 0;
      if (s_tready) d_ = d_ + 1;
    end
    // beat k-1 accepted at the last posedge. Launch CTRL now: it is accepted
    // together with beat k? No — one cycle staggered: CTRL accepted at the
    // next posedge, ctrl_reset high the cycle AFTER, where beat k+1-ish is
    // still streaming with tvalid high — the accept coincides with the reset.
    @(negedge clk);
    s_tdata = stim[tok * D + d_]; s_tvalid = 1;
    awaddr  = REG_CTRL; wdata = 32'h3; awvalid = 1; wvalid = 1;
    @(negedge clk);
    awvalid = 0; wvalid = 0;
    s_tdata = stim[tok * D + d_ + 1];   // beat presented DURING the reset cycle
    @(negedge clk);
    s_tvalid = 0; s_tlast = 0;
  endtask

  // ── post-reset D-020 contract checks ────────────────────────────────────────
  // is_read: victim was a read of the RV record (hidx RV_TOK)
  task automatic post_reset(input string who, input bit is_read);
    logic [31:0] r32;
    int t;
    // settle: the landing monitor commits via NBA at the negedge of the
    // ctrl_reset cycle; sample it strictly after
    repeat (2) @(negedge clk);
    if (!land_seen) begin
      cov_miss++;
      $display("NOTE [%s] cyc=%0d: ctrl_reset never observed — AXI drift", who, cyc);
      blk_arm = 0;
      wait_idle(60000, {who, " miss-recovery"});
      rq_data.delete();
      rq_last.delete();
      return;
    end
    if (land_state == ST_OUTPUT || land_state == ST_OFLUSH) begin
      // burst must complete; while parked, STATUS.idle must read 0 and the
      // FSM must not leave the output phases (that would retract the beat)
      if (blk_arm) begin
        checks++;
        if (!(dut.state == ST_OUTPUT || dut.state == ST_OFLUSH))
          note_fail($sformatf("%s: burst abandoned — FSM state=%0d left OUTPUT/OFLUSH with beat pending",
                              who, dut.state));
        aread(REG_STATUS, r32);
        checks++;
        if ((r32 & 32'h1) === 32'h1)
          note_fail({who, ": STATUS.idle=1 while the reset-crossed burst is draining"});
      end
      blk_arm = 0;
      t = 0;
      while (dut.state !== ST_IDLE && t < 200000) begin
        @(negedge clk);
        t++;
      end
      checks++;
      if (dut.state !== ST_IDLE) begin
        note_fail({who, ": engine never returned to IDLE after reset-crossed burst"});
        do_hard_reset(who);
        return;
      end
      checks++;
      if (rq_data.size() != D) begin
        note_fail($sformatf("%s: reset-crossed burst did not complete exactly — %0d beats (exp %0d)",
                            who, rq_data.size(), D));
      end else begin
        for (int i = 0; i < D; i++) begin
          checks++;
          if (rq_data[i] !== exp_hat[RV_TOK * D + i])
            note_fail($sformatf("%s: drained beat %0d corrupt: got %08h exp %08h",
                                who, i, rq_data[i], exp_hat[RV_TOK * D + i]));
          checks++;
          if (rq_last[i] !== bit'(i == D - 1))
            note_fail($sformatf("%s: drained beat %0d tlast=%b (framing lost)",
                                who, i, rq_last[i]));
        end
        cov_drain_exact++;
      end
    end else begin
      // reset must be IMMEDIATE: ST_IDLE within 6 cycles of the reset cycle
      t = 0;
      while (dut.state !== ST_IDLE && t < 6) begin
        @(negedge clk);
        t++;
      end
      checks++;
      if (dut.state !== ST_IDLE) begin
        note_fail($sformatf("%s: soft reset NOT honored — FSM state=%0d %0d cycles after ctrl_reset (landed in %0d)",
                            who, dut.state, t, land_state));
        aread(REG_STATUS, r32);
        checks++;
        if ((r32 & 32'h1) === 32'h1)
          note_fail($sformatf("%s: STATUS.idle=1 while FSM state=%0d — idle flag LIES",
                              who, dut.state));
        do_hard_reset(who);
        return;
      end
      repeat (4) @(negedge clk);   // grace for in-flight collector commits
      if (is_read) begin
        // a read at/before RLOAD/RWAIT: dropped with ZERO beats; a read that
        // completed before the reset: the full D bit-exact beats. Nothing else.
        checks++;
        if (rq_data.size() == D) begin
          for (int i = 0; i < D; i++) begin
            checks++;
            if (rq_data[i] !== exp_hat[RV_TOK * D + i])
              note_fail($sformatf("%s: pre-reset-completed burst beat %0d corrupt: got %08h exp %08h",
                                  who, i, rq_data[i], exp_hat[RV_TOK * D + i]));
          end
        end else if (rq_data.size() != 0) begin
          note_fail($sformatf("%s: read victim produced %0d beats after reset (exp 0 or %0d)",
                              who, rq_data.size(), D));
        end
      end else begin
        checks++;
        if (rq_data.size() != 0)
          note_fail($sformatf("%s: %0d m_axis beat(s) LEAKED after soft reset",
                              who, rq_data.size()));
      end
    end
    rq_data.delete();
    rq_last.delete();
    // D-020: the abort is SAME-CYCLE — the datapath cores must already be
    // quiesced (whitebox; catches a tied-off dp_clear that the STATUS
    // busy-AND alone would mask behind a slow poll) ...
    checks++;
    if (dut.cqv_busy !== 1'b0 || dut.kp_busy !== 1'b0)
      note_fail($sformatf("%s: datapath still walking after soft reset (cqv_busy=%b kp_busy=%b) — D-020 same-cycle abort violated",
                          who, dut.cqv_busy, dut.kp_busy));
    // ... and STATUS.idle must therefore read 1 IMMEDIATELY (demand, not a
    // poll — a poll would hide a datapath that quiesces late)
    aread(REG_STATUS, r32);
    checks++;
    if ((r32 & 32'h1) !== 32'h1)
      note_fail($sformatf("%s: STATUS.idle=0 immediately after the reset settled (aborted work still in flight?)",
                          who));
    repeat (2) @(negedge clk);
  endtask

  // ── the attack event ────────────────────────────────────────────────────────
  localparam int NT = (KEYG == 1) ? 15 : 10;
  localparam int N_BANK = 6;   // D-026 bank-survival events per mode-0 KEYG run

  task automatic attack_event(input int e);
    int tact, vt, j, nb, dly;
    string who;
    vt   = VT_BASE + int'(rnd() % G);          // victim token
    tact = (e < 2 * NT) ? (e % NT) : int'(rnd() % NT);
    who  = $sformatf("E%0d t%0d", e, tact);
    land_seen = 1'b0;
    rq_data.delete();
    rq_last.delete();
    case (tact)
      0: begin // reset while idle (sanity, lands p0)
        awrite(REG_CTRL, 32'h3);
      end
      1: begin // mid VALUE token, parked
        nb = 1 + int'(rnd() % (D - 1));
        awrite(REG_WA, VICT);
        stream_partial(vt, 1'b1, nb);
        awrite(REG_CTRL, 32'h3);
      end
      2: begin // during COMPRESS (long divider walk), random delay
        awrite(REG_WA, VICT);
        stream_tok(vt, 1'b1);
        dly = int'(rnd() % (D * 14));
        repeat (dly) @(negedge clk);
        awrite(REG_CTRL, 32'h3);
      end
      3: begin // ST_STORE collision
        awrite(REG_WA, VICT);
        stream_tok(vt, 1'b1);
        store_collide();
      end
      4: begin // ST_RLOAD collision (read of the pre-written RV record)
        rload_collide(RV_ADDR, 0);
      end
      5: begin // ST_RWAIT collision
        rload_collide(RV_ADDR, 1);
      end
      6: begin // OUTPUT parked with pending beat after k accepted beats
        blk_thresh = int'(rnd() % (D - 1));
        blk_arm    = 1;
        awrite(REG_RA, RV_ADDR);
        j = 0;
        while (!((dut.state == ST_OUTPUT || dut.state == ST_OFLUSH)
                 && m_tvalid && !m_tready) && j < 200000) begin
          @(negedge clk);
          j++;
        end
        awrite(REG_CTRL, 32'h3);
      end
      7: begin // OFLUSH parked (final tlast beat pending)
        blk_thresh = D - 1;
        blk_arm    = 1;
        awrite(REG_RA, RV_ADDR);
        j = 0;
        while (!(dut.state == ST_OFLUSH && m_tvalid && !m_tready)
               && j < 200000) begin
          @(negedge clk);
          j++;
        end
        awrite(REG_CTRL, 32'h3);
      end
      8: begin // read flowing free, random-delay reset (land varies)
        awrite(REG_RA, RV_ADDR);
        dly = int'(rnd() % (2 * D + 8));
        repeat (dly) @(negedge clk);
        awrite(REG_CTRL, 32'h3);
      end
      9: begin // beat-coincident reset during VALUE COLLECT
        awrite(REG_WA, VICT);
        beat_collide(vt, 1'b1);
      end
      10: begin // mid KEY token (j prior full tokens -> also mid-group)
        j  = int'(rnd() % (G - 1));
        nb = 1 + int'(rnd() % (D - 1));
        awrite(REG_WA, VICT);
        for (int i = 0; i < j; i++) begin
          stream_tok(VT_BASE + i, 1'b0);
          wait_tready(who);
        end
        stream_partial(vt, 1'b0, nb);
        awrite(REG_CTRL, 32'h3);
      end
      11: begin // parked in KACCEPT with an open group
        j = 1 + int'(rnd() % (G - 1));
        awrite(REG_WA, VICT);
        for (int i = 0; i < j; i++) begin
          stream_tok(VT_BASE + i, 1'b0);
          wait_tready(who);
        end
        awrite(REG_CTRL, 32'h3);
      end
      12: begin // ST_KFEED collision after j prior tokens (j=G-1 hits the
                // group-completing KFEED)
        j = int'(rnd() % G);
        awrite(REG_WA, VICT);
        for (int i = 0; i < j; i++) begin
          stream_tok(VT_BASE + i, 1'b0);
          wait_tready(who);
        end
        kfeed_collide(vt);
      end
      13: begin // during KEMIT (full group or flushed partial), random delay
        awrite(REG_WA, VICT);
        if ((rnd() & 1) != 0) begin
          for (int i = 0; i < G; i++) begin
            stream_tok(VT_BASE + i, 1'b0);
            if (i < G - 1) wait_tready(who);
          end
        end else begin
          j = 1 + int'(rnd() % (G - 1));
          for (int i = 0; i < j; i++) begin
            stream_tok(VT_BASE + i, 1'b0);
            wait_tready(who);
          end
          pulse_flush();
        end
        j = 0;
        while (dut.state != ST_KEMIT && j < 200000) begin
          @(negedge clk);
          j++;
        end
        dly = int'(rnd() % (D * 14));
        repeat (dly) @(negedge clk);
        awrite(REG_CTRL, 32'h3);
      end
      14: begin // beat-coincident reset during KEY COLLECT
        awrite(REG_WA, VICT);
        beat_collide(vt, 1'b0);
      end
      default: ;
    endcase
    post_reset(who, tact inside {4, 5, 6, 7, 8});
    // ── clean jobs, IMMEDIATELY (no quiesce — attacks B-2 head-on) ───────────
    clean_vjob(e % NV, who);
    if (KEYG == 1 && (tact >= 10 || e % 3 == 0))
      clean_kjob(e % NK, who);
  endtask

  // ── coverage report ─────────────────────────────────────────────────────────
  function automatic void print_coverage();
    $display("COV rst_total %0d", cov_rst_total);
    for (int i = 0; i <= 11; i++) $display("COV rst_land_p%0d %0d", i, cov_land[i]);
    $display("COV rst_pending_beat %0d", cov_pend_beat);
    $display("COV rst_mid_token %0d", cov_mid_tok);
    $display("COV rst_mid_group %0d", cov_mid_grp);
    $display("COV rst_beat_collide %0d", cov_beat_collide);
    $display("COV rst_collide_store %0d", cov_col_store);
    $display("COV rst_collide_rload %0d", cov_col_rload);
    $display("COV rst_collide_rwait %0d", cov_col_rwait);
    $display("COV rst_collide_kfeed %0d", cov_col_kfeed);
    $display("COV rst_drain_exact %0d", cov_drain_exact);
    $display("COV rst_miss %0d", cov_miss);
    $display("COV clean_v_jobs %0d", cov_clean_v);
    $display("COV clean_k_jobs %0d", cov_clean_k);
    $display("COV audit_reads %0d", cov_reads);
    $display("COV audit_peeks %0d", cov_peeks);
    $display("COV audit_bank_peeks %0d", cov_bank_peeks);
    $display("COV rst_bank_survive %0d", cov_bank_rst);
    $display("COV sb_ovwr_sticky %0d", cov_sb_sticky);
    $display("COV sb_ovwr_w1c %0d", cov_sb_w1c);
    $display("COV idle_lie_mon_armed 1");
  endfunction

  // ── main ────────────────────────────────────────────────────────────────────
  initial begin : main
    string vecdir;

    rst_n = 0;
    checks = 0; fails = 0; printed = 0;
    viol_stable = 0; viol_idle = 0;
    vclean_ctr = 0; kclean_ctr = 0;
    clear_tb_drives();
    bready = 1;
    m_tready = 1;
    s_tdata = '0; araddr = '0; awaddr = '0; wdata = '0;

    if (!$value$plusargs("vecdir=%s", vecdir))
      $fatal(1, "missing +vecdir=<dir>");
    rng_state = seed ^ 32'h5EED_A0D1;

    $display("==== tb_kvq_audit cfg=%s D=%0d TIER=%0d G=%0d K_OUT=%0d DEPTH=%0d NV=%0d NK=%0d mode=%0d n_rst=%0d bp=%0d stall=%0d seed=%0d",
             CFGNAME, D, TIER, G, K_OUT, DEPTH, NV, NK, mode, n_rst, bp_mode,
             stall_mode, seed);

    $readmemh({vecdir, "/stim.f16.hex"}, stim);
    $readmemh({vecdir, "/exp_rec.hex"}, exp_rec);
    $readmemh({vecdir, "/exp_hat.f32.hex"}, exp_hat);
    if (NK > 0) $readmemh({vecdir, "/exp_bank.f16.hex"}, exp_bank);

    repeat (8) @(negedge clk);
    rst_n = 1;
    repeat (4) @(negedge clk);
    awrite(REG_CTRL, 32'h2);   // enable

    // prologue: pre-write + verify the read-victim record
    awrite(REG_WA, RV_ADDR);
    stream_tok(RV_TOK, 1'b1);
    wait_idle(60000, "prologue RV");
    do_peek(RV_ADDR, RV_TOK, "prologue");
    do_read(RV_ADDR, RV_TOK, "prologue");

    if (mode == 1) begin
      // directed pool run (-0.0 outlier / non-outlier content)
      for (int i = 0; i < NV; i++) clean_vjob(i, "directed");
      for (int k = 0; k < NK; k++) clean_kjob(k, "directed");
    end else begin
      for (int e = 0; e < int'(n_rst); e++) attack_event(e);
      // D-026: bank-survival reset attacks + SB_OVWR probe (KEYG only; these
      // run AFTER the attack loop so they perturb no rnd() draw — the attack
      // trajectory and its clean_k counts stay identical to the baseline)
      if (KEYG == 1) begin
        for (int b = 0; b < N_BANK; b++)
          bank_reset_attack(b % NK, $sformatf("BANK%0d", b));
        sb_ovwr_check("SBOVWR");
      end
      // the pre-written RV record must have survived every soft reset
      do_peek(RV_ADDR, RV_TOK, "epilogue");
      do_read(RV_ADDR, RV_TOK, "epilogue");
    end

    repeat (5) @(negedge clk);
    checks++;
    if (viol_stable != 0)
      note_fail($sformatf("m_axis stability violations: %0d", viol_stable));
    checks++;
    if (viol_idle != 0)
      note_fail($sformatf("STATUS.idle lie(s) observed: %0d", viol_idle));

    print_coverage();
    $display("CONFIG %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
    if (fails == 0) begin
      $display("TB PASS [%s]: resets=%0d clean_v=%0d clean_k=%0d reads=%0d peeks=%0d bank_peeks=%0d bank_rst=%0d",
               CFGNAME, cov_rst_total, cov_clean_v, cov_clean_k, cov_reads,
               cov_peeks, cov_bank_peeks, cov_bank_rst);
      $finish;
    end else begin
      $display("TB FAIL [%s]: %0d fail(s) across %0d checks", CFGNAME, fails,
               checks);
      $fatal(1, "TB FAIL");
    end
  end

endmodule
