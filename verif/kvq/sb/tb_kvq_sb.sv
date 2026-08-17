// tb_kvq_sb.sv — independent KVQ vector-driven scoreboard TB (verif/kvq/sb),
// house pattern of verif/mxe/sb: stimulus + bit-exact expectations come from
// gen_kvq_sb_vectors.py (golden arbiter golden/apex_golden, D-001/D-013); the
// TB only drives, collects and compares. Verilator --binary --timing --assert.
//
// Script commands (see the generator header for full semantics):
//   A V K F G R P E O S X Y M B Q
//
// A2/D-026 KV-REC-DEDUP: key records carry [tag {ssid[6:0],1'b1}][OUTLIER_K
// fp16 lanes][D x int4][pad64]; the per-group scales live in the persistent
// scale bank (dut.g_key.u_bank_store.mem, SCALE_SETS=4 rows of D x 16b).
// P scores the FULL golden-emitted record image (tag bit0 + ssid[7:1] are
// part of the exact bits — never matched as a whole 8'h01 byte); B scores a
// full bank row; Q scores the sticky SB_OVWR flag (IRQ_STATUS[1]). A KEY
// record read takes ONE extra ST_RWAIT cycle (bank priming) — all wait
// loops here are bounded polls, so no cycle-count assumption breaks.
//
// Protocol adversaries (plusargs):
//   +bp_mode=0|1|2     m_axis backpressure: none / ~75% duty / storm (~6%)
//   +stall_mode=0|1|2  s_axis feed gaps: none / short random / bursty storm
//   +seed=<n>
//
// The §5 stream contract is additionally checked every cycle by the bound
// kvq_sb_sva pack (kvq_sb_sva.sv) — violations print "SVA-VIOL" (Makefile
// grep-gate) so expected-failure runs can still complete and report.
//
// Mid-operation reset (the run2 hole, ARCHITECTURE.md §8 Layer-1 REQUIREMENT):
// X commands target every reachable FSM phase by hierarchical state watch
// (dut.state), hard (rst_n) and soft (CTRL.soft_reset), including two
// deterministic single-cycle collision tests (ST_KFEED / ST_RLOAD).
//
// TB discipline (V0/mxe lineage): stimulus and handshake decisions at NEGEDGE,
// commits observed at posedge; $fatal watchdog bounds the run; every FAIL
// message carries the cycle number.

`timescale 1ns/1ps

module tb_kvq_sb;

  // ── configuration (verilator -G) ───────────────────────────────────────────
  parameter int D        = 16;
  parameter int TIER     = 1;
  parameter int G        = 4;
  parameter int K_OUT    = 0;
  parameter int DEPTH    = 64;
  parameter     MASKFILE = ".";
  parameter     CFGNAME  = "cfg";

  // ── record-layout mirror of kvq_engine.sv (§4 + A2/D-026) ──────────────────
  // Mirrors golden cq_codec.sram_row_bits: KEY record dropped the D x fp16
  // field block for OUTLIER_K fp16 lanes (KEY_RAW = 8 + 16k + 4D); row width
  // = pad64(max(key, val)) on keyed engines — at D=64 k<=2 VAL_RAW dominates.
  localparam int BPV     = (TIER == 0) ? 8 : 4;
  localparam int KEYG    = (TIER != 0) ? 1 : 0;
  localparam int VAL_RAW = 8 + 16 + D * BPV;
  localparam int KEY_RAW = 8 + K_OUT * 16 + D * 4;
  localparam int REC_RAW = (KEYG == 1 && KEY_RAW > VAL_RAW) ? KEY_RAW : VAL_RAW;
  localparam int SRAMW   = 64 * ((REC_RAW + 63) / 64);
  localparam int AW      = $clog2(DEPTH);
  localparam int SSETS   = 4;   // kvq_engine SCALE_SETS default (D-026)

  // FSM state codes (kvq_engine.sv, whitebox phase targeting)
  localparam [3:0] ST_IDLE = 4'd0,  ST_COLLECT = 4'd1, ST_COMPRESS = 4'd2,
                   ST_STORE = 4'd3, ST_RLOAD = 4'd4,   ST_RWAIT = 4'd5,
                   ST_OUTPUT = 4'd6, ST_KCOLLECT = 4'd7, ST_KFEED = 4'd8,
                   ST_KACCEPT = 4'd9, ST_KEMIT = 4'd10, ST_OFLUSH = 4'd11;

  localparam [7:0] REG_CTRL = 8'h00, REG_STATUS = 8'h04, REG_OCC = 8'h24,
                   REG_WA = 8'h28, REG_RA = 8'h2C, REG_IRQ_MASK = 8'h34,
                   REG_IRQ_STATUS = 8'h38;

  // ── clock / cycle counter ───────────────────────────────────────────────────
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
                         rvalid, evict_needed, evict_addr,
                         ST_COMPRESS[0], ST_STORE[0], ST_OUTPUT[0], ST_OFLUSH[0]};

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
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned seed         = 32'hACE1_2026;
  int unsigned watchdog_cyc = 8_000_000;
  initial begin
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
      $dumpvars(0, tb_kvq_sb);
    end
  end

  // ── stimulus / expectation storage ──────────────────────────────────────────
  logic [15:0]      stim     [0:16383];
  logic [SRAMW-1:0] exp_rec  [0:1023];
  logic [31:0]      exp_hat  [0:16383];
  logic [D*16-1:0]  exp_bank [0:255];   // golden persistent-bank rows (D-026)

  // D-026 persistent scale bank whitebox: the u_bank_store instance only
  // elaborates inside kvq_engine's `g_key` generate scope (keyed tiers), so
  // the hierarchical peek must live in a matching TB generate — a TIER-0
  // build has no such scope and scores no B commands.
  wire [D*16-1:0] bank_mirror [0:SSETS-1];
  generate
    if (KEYG == 1) begin : g_bankmir
      for (genvar bs = 0; bs < SSETS; bs++) begin : g_row
        assign bank_mirror[bs] = dut.g_key.u_bank_store.mem[bs];
      end
    end else begin : g_bankmir_z
      for (genvar bs = 0; bs < SSETS; bs++) begin : g_row
        assign bank_mirror[bs] = '0;
      end
    end
  endgenerate

  // ── result collector (posedge — exactly what the DUT commits) ──────────────
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

  // ── m_axis backpressure driver (bp_mode) + reset-test overrides ────────────
  logic        block_mready;
  logic        oflush_arm;
  logic [31:0] lfsr;
  always @(negedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lfsr     <= seed;
      m_tready <= 1'b1;
    end else begin
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      if (block_mready || (oflush_arm && burst_acc >= D - 1)) begin
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
      pend <= 1'b0;
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

  // ── TB-side randomization (feed gaps) ───────────────────────────────────────
  int unsigned rng_state = 32'hC0FFEE01;
  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction

  function automatic int unsigned gap_len();
    unique case (stall_mode)
      0:       gap_len = 0;
      1:       gap_len = ((rnd() & 32'd1) != 0) ? (rnd() % 32'd4) : 0;
      default: gap_len = ((rnd() % 32'd8) == 0) ? (rnd() % 32'd24)
                                                : (rnd() % 32'd3);
    endcase
  endfunction

  // ── bookkeeping ─────────────────────────────────────────────────────────────
  int checks;
  int fails;
  int printed;
  localparam int MAXPRINT = 25;

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

  // ── coverage buckets (manual, run2/mxe pattern) ─────────────────────────────
  int cov_v_tok, cov_k_tok, cov_reads, cov_peeks, cov_rderr, cov_w1c;
  int cov_irq_masked, cov_occ_check, cov_sram_full_seen, cov_flush_pulse;
  int cov_bank_check, cov_sbovwr_check, cov_sbovwr_w1c;
  int cov_bp [0:2];
  int cov_stall [0:2];
  int cov_rst_hard [1:11];
  int cov_rst_soft [1:11];
  int cov_rst_collide_kfeed, cov_rst_collide_rload, cov_rst_kemit_probe;
  int cov_frame_early, cov_frame_notlast, cov_frame_late;

  function automatic void print_coverage();
    $display("COV v_tok %0d", cov_v_tok);
    $display("COV k_tok %0d", cov_k_tok);
    $display("COV reads %0d", cov_reads);
    $display("COV peeks %0d", cov_peeks);
    $display("COV rderr %0d", cov_rderr);
    $display("COV w1c %0d", cov_w1c);
    $display("COV irq_masked %0d", cov_irq_masked);
    $display("COV occ_check %0d", cov_occ_check);
    $display("COV sram_full_seen %0d", cov_sram_full_seen);
    $display("COV flush_pulse %0d", cov_flush_pulse);
    $display("COV bank_check %0d", cov_bank_check);
    $display("COV sb_ovwr_check %0d", cov_sbovwr_check);
    $display("COV sb_ovwr_w1c %0d", cov_sbovwr_w1c);
    for (int i = 0; i < 3; i++) $display("COV bp_mode_%0d %0d", i, cov_bp[i]);
    for (int i = 0; i < 3; i++) $display("COV stall_mode_%0d %0d", i, cov_stall[i]);
    for (int i = 1; i <= 11; i++) $display("COV rst_hard_p%0d %0d", i, cov_rst_hard[i]);
    for (int i = 1; i <= 11; i++) $display("COV rst_soft_p%0d %0d", i, cov_rst_soft[i]);
    $display("COV rst_collide_kfeed %0d", cov_rst_collide_kfeed);
    $display("COV rst_collide_rload %0d", cov_rst_collide_rload);
    $display("COV rst_kemit_probe %0d", cov_rst_kemit_probe);
    $display("COV framing_early %0d", cov_frame_early);
    $display("COV framing_notlast %0d", cov_frame_notlast);
    $display("COV framing_late %0d", cov_frame_late);
  endfunction

  // ── AXI-Lite tasks (V0/smoke lineage, negedge-driven) ───────────────────────
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

  task automatic wait_state(input [3:0] st, input int max_cyc,
                            input string who, output bit ok);
    int t;
    t = 0;
    while (dut.state != st && t < max_cyc) begin
      @(negedge clk);
      t++;
    end
    ok = (dut.state == st);
    checks++;
    if (!ok)
      note_fail($sformatf("%s: FSM never reached phase %0d (state=%0d after %0d cyc)",
                          who, st, dut.state, max_cyc));
  endtask

  // ── stream drivers ──────────────────────────────────────────────────────────
  int flush_arm;   // 0 none; 1/3 = pulse flush_req mid-token on next stream

  // mal: 0 normal; 1 early tlast @beat D/2; 2 no tlast; 3 late tlast (extra beat)
  task automatic stream_tok(input int tok, input bit isval, input int mal);
    int d_, g_, gp, nbeats;
    bit pulsed;
    nbeats = (mal == 1) ? (D / 2 + 1) : D;
    d_ = 0; g_ = 0; pulsed = 0;
    while (d_ < nbeats) begin
      @(negedge clk);
      s_tdata  = stim[tok * D + d_];
      s_tvalid = 1;
      s_tuser  = isval;
      s_tlast  = (mal == 1) ? (d_ == nbeats - 1)
               : (mal == 2 || mal == 3) ? 1'b0
               : (d_ == D - 1);
      if (flush_arm != 0 && d_ == D / 2 && !pulsed) begin
        flush_r = 1;
        pulsed  = 1;
        cov_flush_pulse++;
      end else begin
        flush_r = 0;
      end
      if (s_tready) begin
        d_ = d_ + 1;
        // optional inter-beat feed gap with valid LOW. The beat commits at
        // the posedge right after this negedge, so deasserting at the NEXT
        // negedge is a legal (post-accept) valid drop — never a retraction.
        if (mal == 0 && stall_mode != 0 && d_ < nbeats) begin
          gp = int'(gap_len());
          if (gp > 0) begin
            @(negedge clk);
            s_tvalid = 0;
            flush_r  = 0;
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
    s_tvalid = 0; s_tlast = 0; flush_r = 0;
    if (mal == 3) begin   // one extra beat carrying the (late) tlast
      g_ = 0;
      @(negedge clk);
      s_tdata = stim[tok * D]; s_tvalid = 1; s_tuser = isval; s_tlast = 1;
      while (!s_tready && g_ < 200000) begin
        @(negedge clk);
        g_++;
      end
      @(negedge clk);
      s_tvalid = 0; s_tlast = 0;
    end
    flush_arm = 0;
  endtask

  // partial token: nbeats then park (mid-token stall / reset targeting)
  task automatic stream_partial(input int tok, input bit isval, input int nbeats);
    int d_;
    d_ = 0;
    while (d_ < nbeats) begin
      @(negedge clk);
      s_tdata = stim[tok * D + d_]; s_tvalid = 1; s_tuser = isval; s_tlast = 0;
      if (s_tready) d_ = d_ + 1;
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
    cov_flush_pulse++;
    @(negedge clk);
    flush_r = 0;
  endtask

  // ── checkers ────────────────────────────────────────────────────────────────
  task automatic do_read(input int addr, input int hidx);
    int g_;
    awrite(REG_RA, addr);
    g_ = 0;
    while (rq_data.size() < D && g_ < 400000) begin
      @(negedge clk);
      g_++;
    end
    checks++;
    if (rq_data.size() != D) begin
      note_fail($sformatf("R addr=%0d: got %0d beats, expected %0d", addr,
                          rq_data.size(), D));
    end else begin
      for (int i = 0; i < D; i++) begin
        checks++;
        if (rq_data[i] !== exp_hat[hidx * D + i])
          note_fail($sformatf("R addr=%0d beat=%0d: got %08h exp %08h (hidx=%0d)",
                              addr, i, rq_data[i], exp_hat[hidx * D + i], hidx));
        checks++;
        if (rq_last[i] !== bit'(i == D - 1))
          note_fail($sformatf("R addr=%0d beat=%0d: tlast=%b wrong", addr, i,
                              rq_last[i]));
      end
    end
    repeat (6) @(negedge clk);
    checks++;
    if (rq_data.size() > D)
      note_fail($sformatf("R addr=%0d: %0d EXTRA beats after tlast", addr,
                          rq_data.size() - D));
    rq_data.delete();
    rq_last.delete();
    cov_reads++;
  endtask

  task automatic do_peek(input int addr, input int ridx);
    logic [SRAMW-1:0] rec;
    rec = dut.u_sram.mem[addr % DEPTH];
    checks++;
    if (rec !== exp_rec[ridx])
      note_fail($sformatf("P addr=%0d ridx=%0d: record mismatch\n  got %0h\n  exp %0h",
                          addr, ridx, rec, exp_rec[ridx]));
    cov_peeks++;
  endtask

  // D-026: full-row compare of a persistent scale-bank set (tag ssid target)
  task automatic do_bank(input int set_i, input int bidx);
    logic [D*16-1:0] row;
    row = bank_mirror[set_i % SSETS];
    checks++;
    if (row !== exp_bank[bidx])
      note_fail($sformatf("B set=%0d bidx=%0d: bank row mismatch\n  got %0h\n  exp %0h",
                          set_i, bidx, row, exp_bank[bidx]));
    cov_bank_check++;
  endtask

  // D-026: SB_OVWR (IRQ_STATUS[1], sticky, W1C) — live-set reuse flag
  task automatic do_sbovwr(input int expbit, input int w1c);
    logic [31:0] r32;
    if (w1c != 0) begin
      awrite(REG_IRQ_STATUS, 32'h2);   // W1C bit 1 only
      repeat (2) @(negedge clk);
      cov_sbovwr_w1c++;
    end
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== ((expbit != 0) ? 32'h2 : 32'h0))
      note_fail($sformatf("Q: IRQ_STATUS[1] SB_OVWR=%0d expected %0d (w1c=%0d)",
                          (r32 & 32'h2) != 0, expbit, w1c));
    cov_sbovwr_check++;
  endtask

  task automatic do_err(input int addr, input int mask);
    logic [31:0] r32;
    awrite(REG_IRQ_MASK, 32'(mask));
    awrite(REG_RA, addr);
    repeat (30) @(negedge clk);        // RD_WAIT_MAX=8 + margin
    checks++;
    if (rq_data.size() != 0)
      note_fail($sformatf("E addr=%0d: unwritten read produced %0d beats", addr,
                          rq_data.size()));
    aread(REG_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== 32'h2) note_fail("E: STATUS[1] RD_ERR mirror not set");
    checks++;
    if ((r32 & 32'h1) !== 32'h1) note_fail("E: engine not idle after dropped read");
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h1) !== 32'h1) note_fail("E: IRQ_STATUS[0] RD_ERR not sticky");
    checks++;
    if (irq_w !== ((mask != 0) ? 1'b1 : 1'b0))
      note_fail($sformatf("E: irq pin=%b, expected %b (mask=%0d)", irq_w,
                          (mask != 0), mask));
    if (mask == 0) cov_irq_masked++;
    awrite(REG_IRQ_STATUS, 32'h1);     // W1C
    repeat (2) @(negedge clk);
    aread(REG_IRQ_STATUS, r32);
    checks++;
    if ((r32 & 32'h1) !== 32'h0) note_fail("E: W1C did not clear RD_ERR");
    aread(REG_STATUS, r32);
    checks++;
    if ((r32 & 32'h2) !== 32'h0) note_fail("E: STATUS[1] still set after W1C");
    checks++;
    if (irq_w !== 1'b0) note_fail("E: irq pin still high after W1C");
    cov_rderr++;
    cov_w1c++;
  endtask

  task automatic do_occ(input int n);
    logic [31:0] r32;
    aread(REG_OCC, r32);
    checks++;
    if (r32 !== 32'(n))
      note_fail($sformatf("O: occupancy=%0d expected %0d", r32, n));
    cov_occ_check++;
  endtask

  task automatic do_sfull(input int b);
    logic [31:0] r32;
    aread(REG_STATUS, r32);
    checks++;
    if ((r32 & 32'h8) !== ((b != 0) ? 32'h8 : 32'h0))
      note_fail($sformatf("S: STATUS.sram_full=%0d expected %0d",
                          (r32 & 32'h8) != 0, b != 0));
    cov_sram_full_seen++;
  endtask

  // ── reset machinery ─────────────────────────────────────────────────────────
  task automatic clear_tb_drives();
    s_tvalid = 0; s_tlast = 0; s_tuser = 0; flush_r = 0; flush_arm = 0;
    awvalid = 0; wvalid = 0; arvalid = 0; rready = 0;
    block_mready = 0; oflush_arm = 0;
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
    awrite(REG_CTRL, 32'h2);           // re-enable
    aread(REG_STATUS, r32);
    checks++;
    if ((r32 & 32'h1) !== 32'h1) note_fail({who, ": not idle after hard reset"});
    aread(REG_OCC, r32);
    checks++;
    if (r32 !== 32'h0)
      note_fail($sformatf("%s: occupancy=%0d after hard reset (exp 0)", who, r32));
  endtask

  // launch an operation and park/observe the FSM in the requested phase.
  // Uses vtok as the pre-written record for read-side phases.
  task automatic launch_phase(input int phase, input int tok, input int vtok,
                              input int addr, output bit ok);
    ok = 1'b1;
    case (phase)
      1: begin
        awrite(REG_WA, addr);
        stream_partial(tok, 1'b1, 3);
        wait_state(ST_COLLECT, 100, "launch p1", ok);
      end
      2, 3: begin
        awrite(REG_WA, addr);
        stream_tok(tok, 1'b1, 0);
        wait_state(4'(phase), D * 40 + 2000, $sformatf("launch p%0d", phase), ok);
      end
      4, 5: begin
        awrite(REG_WA, addr);
        stream_tok(vtok, 1'b1, 0);
        wait_idle(3000, "launch prereq");
        @(negedge clk);
        awaddr = REG_RA; wdata = 32'(addr); awvalid = 1; wvalid = 1;
        @(negedge clk);
        awvalid = 0; wvalid = 0;
        wait_state(4'(phase), 20, $sformatf("launch p%0d", phase), ok);
      end
      6, 11: begin
        awrite(REG_WA, addr);
        stream_tok(vtok, 1'b1, 0);
        wait_idle(3000, "launch prereq");
        if (phase == 6) block_mready = 1;
        else            oflush_arm   = 1;
        awrite(REG_RA, addr);
        wait_state(4'(phase), D * 8 + 200, $sformatf("launch p%0d", phase), ok);
      end
      7: begin
        awrite(REG_WA, addr);
        stream_partial(tok, 1'b0, 3);
        wait_state(ST_KCOLLECT, 100, "launch p7", ok);
      end
      8: begin
        awrite(REG_WA, addr);
        stream_tok(tok, 1'b0, 0);
        wait_state(ST_KFEED, 10, "launch p8", ok);
      end
      9: begin
        awrite(REG_WA, addr);
        stream_tok(tok, 1'b0, 0);
        wait_state(ST_KACCEPT, 30, "launch p9", ok);
      end
      10: begin
        awrite(REG_WA, addr);
        for (int i = 0; i < G; i++) stream_tok(tok + i, 1'b0, 0);
        wait_state(ST_KEMIT, 100, "launch p10", ok);
      end
      default: begin
        note_fail($sformatf("launch_phase: bad phase %0d", phase));
        ok = 1'b0;
      end
    endcase
  endtask

  task automatic roundtrip(input int vtok, input int addr, input int ridx,
                           input int hidx, input string who);
    awrite(REG_WA, addr);
    stream_tok(vtok, 1'b1, 0);
    wait_idle(3000, who);
    do_peek(addr, ridx);
    do_read(addr, hidx);
    cov_v_tok++;
  endtask

  // soft reset + whitebox contract checks (see RESULT.md findings).
  //
  // D-020 semantics (kvq_engine.sv header) split the expectation in two:
  //  * outburst==0 (no read burst in flight): the reset is IMMEDIATE — FSM in
  //    ST_IDLE within 6 cycles, ZERO m_axis beats delivered after the reset.
  //  * outburst==1 (parked in ST_OUTPUT/ST_OFLUSH with a beat pending): §5 has
  //    no reset carve-out, so the burst must COMPLETE — the FSM must STAY in
  //    the output phases (leaving early would retract the pending beat),
  //    STATUS.idle must read 0 while draining, and after backpressure release
  //    the burst must finish EXACTLY: D beats total, tlast on beat D-1, every
  //    beat bit-exact vs exp_hat[hidx]. No retraction ever (the stability
  //    monitor and the bound SVA stay armed throughout).
  task automatic soft_reset_check(input string who, input bit outburst,
                                  input int hidx);
    logic [31:0] r32;
    int q0;
    q0 = rq_data.size();               // beats accepted BEFORE the reset
    awrite(REG_CTRL, 32'h3);           // soft_reset | enable
    repeat (6) @(negedge clk);
    if (!outburst) begin
      checks++;
      if (dut.state !== ST_IDLE) begin
        note_fail($sformatf("%s: soft reset NOT honored — FSM state=%0d 6 cycles after CTRL.soft_reset",
                            who, dut.state));
        aread(REG_STATUS, r32);
        checks++;
        if ((r32 & 32'h1) === 32'h1)
          note_fail($sformatf("%s: STATUS.idle=1 while FSM state=%0d — idle flag LIES after soft reset",
                              who, dut.state));
      end
    end else begin
      // D-020: the in-flight burst must still be held (no retraction) ...
      checks++;
      if (!(dut.state == ST_OUTPUT || dut.state == ST_OFLUSH))
        note_fail($sformatf("%s: burst abandoned — FSM state=%0d left OUTPUT/OFLUSH with beat(s) pending (retraction)",
                            who, dut.state));
      // ... and STATUS.idle must NOT lie while the burst is draining
      aread(REG_STATUS, r32);
      checks++;
      if ((r32 & 32'h1) === 32'h1)
        note_fail({who, ": STATUS.idle=1 while the reset-crossed burst is still draining"});
    end
    // release any parking and let the engine self-recover (bounded)
    block_mready = 0;
    oflush_arm   = 0;
    begin
      int t;
      t = 0;
      while (dut.state !== ST_IDLE && t < 100000) begin
        @(negedge clk);
        t++;
      end
      checks++;
      if (dut.state !== ST_IDLE) begin
        note_fail({who, ": engine NEVER returned to IDLE after soft reset — hard-reset rescue"});
        do_hard_reset(who);
      end
    end
    if (outburst) begin
      // D-020: the burst must have completed EXACTLY — D beats, tlast on the
      // last, every beat bit-exact (data registers must survive the reset)
      checks++;
      if (rq_data.size() != D) begin
        note_fail($sformatf("%s: reset-crossed burst did not complete cleanly — %0d beats total (exp %0d)",
                            who, rq_data.size(), D));
      end else begin
        for (int i = 0; i < D; i++) begin
          checks++;
          if (rq_data[i] !== exp_hat[hidx * D + i])
            note_fail($sformatf("%s: drained beat %0d corrupt: got %08h exp %08h",
                                who, i, rq_data[i], exp_hat[hidx * D + i]));
          checks++;
          if (rq_last[i] !== bit'(i == D - 1))
            note_fail($sformatf("%s: drained beat %0d tlast=%b wrong (framing lost)",
                                who, i, rq_last[i]));
        end
      end
    end else if (rq_data.size() > q0) begin
      // no burst in flight: any beat after the reset is a leak
      checks++;
      note_fail($sformatf("%s: %0d m_axis beat(s) delivered AFTER soft reset",
                          who, rq_data.size() - q0));
    end
    rq_data.delete();
    rq_last.delete();
    repeat (4) @(negedge clk);
  endtask

  // deterministic ST_KFEED collision: final key beat + CTRL.soft_reset accepted
  // at the SAME posedge, so ctrl_reset is high exactly in the KFEED cycle.
  task automatic kfeed_collide(input int tok, input int addr, output bit hit);
    int d_;
    awrite(REG_WA, addr);
    d_ = 0;
    while (d_ < D - 1) begin
      @(negedge clk);
      s_tdata = stim[tok * D + d_]; s_tvalid = 1; s_tuser = 0; s_tlast = 0;
      if (s_tready) d_ = d_ + 1;
    end
    @(negedge clk);                     // final beat + CTRL write together
    s_tdata = stim[tok * D + D - 1]; s_tvalid = 1; s_tlast = 1;
    awaddr = REG_CTRL; wdata = 32'h3; awvalid = 1; wvalid = 1;
    @(negedge clk);
    s_tvalid = 0; s_tlast = 0; awvalid = 0; wvalid = 0;
    // this negedge is inside the KFEED/ctrl_reset collision cycle
    hit = (dut.state === ST_KFEED);
    if (!hit)
      $display("NOTE cyc=%0d: KFEED collision missed (state=%0d) — timing drift",
               cyc, dut.state);
  endtask

  // deterministic ST_RLOAD collision: READ_ADDR write then CTRL write on
  // back-to-back cycles, so ctrl_reset lands exactly in the RLOAD cycle.
  // The hit is sampled IN the collision cycle itself (state==RLOAD while
  // ctrl_reset is high) — sampling one cycle later only "hit" on the buggy
  // RTL because the lost reset left the FSM in RLOAD/RWAIT; a correct DUT is
  // already back in IDLE by then.
  task automatic rload_collide(input int addr, output bit hit);
    @(negedge clk);
    awaddr = REG_RA; wdata = 32'(addr); awvalid = 1; wvalid = 1;
    @(negedge clk);
    awaddr = REG_CTRL; wdata = 32'h3;   // valids stay high: back-to-back accept
    @(negedge clk);
    awvalid = 0; wvalid = 0;
    // this negedge is inside the RLOAD/ctrl_reset collision cycle
    hit = (dut.state === ST_RLOAD) || (dut.state === ST_RWAIT);
    if (!hit)
      $display("NOTE cyc=%0d: RLOAD collision missed (state=%0d) — timing drift",
               cyc, dut.state);
    @(negedge clk);
  endtask

  // ── X command: mid-operation reset test ─────────────────────────────────────
  task automatic do_x(input int kind, input int phase, input int tok,
                      input int vtok, input int addr, input int ridx,
                      input int hidx);
    bit ok, hit;
    int f0;
    string who;
    who = $sformatf("X k=%0d p=%0d", kind, phase);
    wait_idle(3000, {who, " prologue"});
    repeat (10) @(negedge clk);
    case (kind)
      0: begin // hard rst_n in the targeted phase
        launch_phase(phase, tok, vtok, addr, ok);
        $display("RESET [%s] cyc=%0d: hard reset in FSM phase %0d (target %0d)",
                 who, cyc, dut.state, phase);
        if (ok) cov_rst_hard[phase]++;
        do_hard_reset(who);
        roundtrip(vtok, addr, ridx, hidx, {who, " recovery"});
      end
      1, 4: begin // soft reset, parked in the targeted phase
        launch_phase(phase, tok, vtok, addr, ok);
        $display("RESET [%s] cyc=%0d: soft reset in FSM phase %0d (target %0d)",
                 who, cyc, dut.state, phase);
        if (ok) cov_rst_soft[phase]++;
        // phases 6/11 park an output burst: D-020 says it must complete
        soft_reset_check(who, phase == 6 || phase == 11, hidx);
        if (kind == 4) begin
          cov_rst_kemit_probe++;
          return;                        // Y probe follows immediately
        end
        f0 = fails;
        roundtrip(vtok, addr, ridx, hidx, {who, " immediate reuse"});
        if (fails != f0) begin
          $display("NOTE [%s] cyc=%0d: IMMEDIATE reuse after soft reset failed (datapath not reset); retrying after quiesce",
                   who, cyc);
          repeat (D * 40 + 2000) @(negedge clk);
          roundtrip(vtok, addr, ridx, hidx, {who, " post-quiesce reuse"});
        end
      end
      2: begin // soft reset colliding with single-cycle ST_KFEED
        kfeed_collide(tok, addr, hit);
        if (hit) cov_rst_collide_kfeed++;
        soft_reset_check_collide(who, hit);
        roundtrip(vtok, addr, ridx, hidx, {who, " recovery"});
      end
      3: begin // soft reset colliding with single-cycle ST_RLOAD
        awrite(REG_WA, addr);
        stream_tok(vtok, 1'b1, 0);
        wait_idle(3000, {who, " prereq"});
        rload_collide(addr, hit);
        if (hit) cov_rst_collide_rload++;
        soft_reset_check_collide(who, hit);
        roundtrip(vtok, addr, ridx, hidx, {who, " recovery"});
      end
      default: note_fail($sformatf("do_x: bad kind %0d", kind));
    endcase
  endtask

  // post-collision checks: the FSM must honor the reset (return to IDLE at
  // once); a second soft reset is issued as rescue if it did not.
  task automatic soft_reset_check_collide(input string who, input bit hit);
    logic [31:0] r32;
    int t, q0;
    q0 = rq_data.size();               // beats accepted BEFORE the collision
    repeat (5) @(negedge clk);
    if (hit) begin
      checks++;
      if (dut.state !== ST_IDLE) begin
        note_fail($sformatf("%s: soft reset LOST in collision — FSM state=%0d 5 cycles later (case-override defeats ctrl_reset)",
                            who, dut.state));
        aread(REG_STATUS, r32);
        checks++;
        if ((r32 & 32'h1) === 32'h1)
          note_fail($sformatf("%s: STATUS.idle=1 while FSM state=%0d after lost soft reset",
                              who, dut.state));
      end
    end
    // rescue: wait for self-recovery, then a second soft reset from a safe state
    t = 0;
    while (dut.state !== ST_IDLE && dut.state !== ST_KACCEPT && t < 100000) begin
      @(negedge clk);
      t++;
    end
    awrite(REG_CTRL, 32'h3);
    repeat (6) @(negedge clk);
    checks++;
    if (dut.state !== ST_IDLE) begin
      note_fail({who, ": rescue soft reset also failed — hard-reset rescue"});
      do_hard_reset(who);
    end
    if (rq_data.size() > q0) begin
      checks++;
      note_fail($sformatf("%s: %0d m_axis beat(s) delivered after colliding soft reset",
                          who, rq_data.size() - q0));
    end
    rq_data.delete();
    rq_last.delete();
  endtask

  // ── Y command: immediate key group after soft-reset-mid-KEMIT ──────────────
  task automatic do_y(input int tokbase, input int addr, input int r0,
                      input int h0);
    int p;
    logic [31:0] st;
    awrite(REG_WA, addr);
    for (int i = 0; i < G; i++) begin
      stream_tok(tokbase + i, 1'b0, 0);
      cov_k_tok++;
    end
    p = 0; st = 0;
    while (((st & 32'h1) == 32'h0) && p < 60000) begin
      aread(REG_STATUS, st);
      p++;
    end
    checks++;
    if ((st & 32'h1) == 32'h0) begin
      note_fail("Y: engine WEDGED processing a key group started right after soft reset (datapath never quiesced, no host-visible busy)");
      do_hard_reset("Y rescue");
      return;
    end
    for (int i = 0; i < G; i++) begin
      do_peek(addr + i, r0 + i);
      do_read(addr + i, h0 + i);
    end
  endtask

  // ── M command: malformed framing ────────────────────────────────────────────
  task automatic do_m(input int tok, input int mode, input int addr,
                      input int ridx, input int hidx);
    logic [31:0] r32;
    string who;
    who = $sformatf("M mode=%0d", mode);
    awrite(REG_WA, addr);
    case (mode)
      1: begin // early tlast: engine compresses a short/garbage token — must
         // not hang, must self-return to IDLE, next op must be clean
        stream_tok(tok, 1'b1, 1);
        wait_idle(3000, {who, " self-completion"});
        roundtrip(tok, addr, ridx, hidx, {who, " post-recovery"});
        cov_frame_early++;
      end
      2: begin // missing tlast: beat count alone must complete the token,
         // record must be BIT-EXACT (tlast is informational, ISA §3)
        stream_tok(tok, 1'b1, 2);
        wait_idle(3000, {who, " completion"});
        do_peek(addr, ridx);
        do_read(addr, hidx);
        cov_frame_notlast++;
      end
      3: begin // late tlast: D good beats then a stray tlast beat — the stray
         // starts a phantom token (framing desync); engine must be parked,
         // recoverable by soft reset, then fully usable
        stream_tok(tok, 1'b1, 3);
        repeat (D * 40 + 2000) @(negedge clk);  // let the real token store
        aread(REG_STATUS, r32);
        checks++;
        if ((r32 & 32'h1) === 32'h1)
          $display("NOTE [%s] cyc=%0d: engine idle after stray-tlast beat (phantom token completed?)",
                   who, cyc);
        else
          $display("OBSERVED [%s] cyc=%0d: framing desync — engine parked mid-phantom-token (STATUS.idle=0), as expected for malformed input",
                   who, cyc);
        awrite(REG_CTRL, 32'h3);                // soft reset recovers
        repeat (6) @(negedge clk);
        checks++;
        if (dut.state !== ST_IDLE)
          note_fail({who, ": soft reset did not recover framing desync"});
        roundtrip(tok, addr, ridx, hidx, {who, " post-recovery"});
        cov_frame_late++;
      end
      default: note_fail($sformatf("do_m: bad mode %0d", mode));
    endcase
  endtask

  // ── script reader ───────────────────────────────────────────────────────────
  int    fd;
  string cur_line;

  function automatic bit next_line();
    while ($fgets(cur_line, fd) > 0) begin
      if (cur_line.len() > 1 && cur_line.getc(0) != "#") return 1'b1;
    end
    return 1'b0;
  endfunction

  // ── main sequence ───────────────────────────────────────────────────────────
  initial begin : main
    string vecdir, script_path;
    byte   tag0;
    int    a0, a1, a2, a3, a4, a5, a6, cnt;
    string tg;

    rst_n = 0;
    checks = 0; fails = 0; printed = 0;
    viol_stable = 0;
    clear_tb_drives();
    bready = 1;
    m_tready = 1;
    s_tdata = '0; araddr = '0; awaddr = '0; wdata = '0;

    if (!$value$plusargs("vecdir=%s", vecdir))
      $fatal(1, "missing +vecdir=<dir>");
    if (!$value$plusargs("script=%s", script_path))
      $fatal(1, "missing +script=<file>");
    rng_state = seed ^ 32'hDEAD_4EA1;
    cov_bp[bp_mode > 2 ? 2 : bp_mode]++;
    cov_stall[stall_mode > 2 ? 2 : stall_mode]++;

    $display("==== tb_kvq_sb cfg=%s D=%0d TIER=%0d G=%0d K_OUT=%0d DEPTH=%0d bp=%0d stall=%0d seed=%0d",
             CFGNAME, D, TIER, G, K_OUT, DEPTH, bp_mode, stall_mode, seed);
    $display("     vecdir=%s script=%s", vecdir, script_path);

    $readmemh({vecdir, "/stim.f16.hex"}, stim);
    $readmemh({vecdir, "/exp_rec.hex"}, exp_rec);
    $readmemh({vecdir, "/exp_hat.f32.hex"}, exp_hat);
    if (KEYG == 1)   // bank rows exist only for keyed configs
      $readmemh({vecdir, "/exp_scalebank.hex"}, exp_bank);

    fd = $fopen(script_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", script_path);

    repeat (8) @(negedge clk);
    rst_n = 1;
    repeat (4) @(negedge clk);
    awrite(REG_CTRL, 32'h2);   // enable

    while (next_line()) begin
      tag0 = cur_line.getc(0);
      tg   = "";
      unique case (tag0)
        "A": begin
          cnt = $sscanf(cur_line, "%s %d", tg, a0);
          if (cnt != 2) $fatal(1, "bad A: %s", cur_line);
          awrite(REG_WA, 32'(a0));
        end
        "V": begin
          cnt = $sscanf(cur_line, "%s %d", tg, a0);
          if (cnt != 2) $fatal(1, "bad V: %s", cur_line);
          stream_tok(a0, 1'b1, 0);
          wait_idle(3000, "V");
          cov_v_tok++;
        end
        "K": begin
          cnt = $sscanf(cur_line, "%s %d %d", tg, a0, a1);
          if (cnt != 3) $fatal(1, "bad K: %s", cur_line);
          stream_tok(a0, 1'b0, 0);
          if (a1 == 1)      wait_idle(400000, "K group emit");
          else if (a1 == 0) wait_tready("K");
          cov_k_tok++;
        end
        "F": begin
          cnt = $sscanf(cur_line, "%s %d", tg, a0);
          if (cnt != 2) $fatal(1, "bad F: %s", cur_line);
          if (a0 == 0) begin
            pulse_flush();
            wait_idle(400000, "F tail flush");
          end else if (a0 == 2) begin
            pulse_flush();
          end else begin
            flush_arm = a0;   // 1 or 3: mid-token pulse on the next K
          end
        end
        "G": wait_idle(400000, "G");
        "R": begin
          cnt = $sscanf(cur_line, "%s %d %d", tg, a0, a1);
          if (cnt != 3) $fatal(1, "bad R: %s", cur_line);
          do_read(a0, a1);
        end
        "P": begin
          cnt = $sscanf(cur_line, "%s %d %d", tg, a0, a1);
          if (cnt != 3) $fatal(1, "bad P: %s", cur_line);
          do_peek(a0, a1);
        end
        "B": begin
          cnt = $sscanf(cur_line, "%s %d %d", tg, a0, a1);
          if (cnt != 3) $fatal(1, "bad B: %s", cur_line);
          do_bank(a0, a1);
        end
        "Q": begin
          cnt = $sscanf(cur_line, "%s %d %d", tg, a0, a1);
          if (cnt != 3) $fatal(1, "bad Q: %s", cur_line);
          do_sbovwr(a0, a1);
        end
        "E": begin
          cnt = $sscanf(cur_line, "%s %d %d", tg, a0, a1);
          if (cnt != 3) $fatal(1, "bad E: %s", cur_line);
          do_err(a0, a1);
        end
        "O": begin
          cnt = $sscanf(cur_line, "%s %d", tg, a0);
          if (cnt != 2) $fatal(1, "bad O: %s", cur_line);
          do_occ(a0);
        end
        "S": begin
          cnt = $sscanf(cur_line, "%s %d", tg, a0);
          if (cnt != 2) $fatal(1, "bad S: %s", cur_line);
          do_sfull(a0);
        end
        "X": begin
          cnt = $sscanf(cur_line, "%s %d %d %d %d %d %d %d", tg, a0, a1, a2,
                        a3, a4, a5, a6);
          if (cnt != 8) $fatal(1, "bad X: %s", cur_line);
          do_x(a0, a1, a2, a3, a4, a5, a6);
        end
        "Y": begin
          cnt = $sscanf(cur_line, "%s %d %d %d %d", tg, a0, a1, a2, a3);
          if (cnt != 5) $fatal(1, "bad Y: %s", cur_line);
          do_y(a0, a1, a2, a3);
        end
        "M": begin
          cnt = $sscanf(cur_line, "%s %d %d %d %d %d", tg, a0, a1, a2, a3, a4);
          if (cnt != 6) $fatal(1, "bad M: %s", cur_line);
          do_m(a0, a1, a2, a3, a4);
        end
        default: $fatal(1, "unexpected record: %s", cur_line);
      endcase
    end
    $fclose(fd);
    if (tg == "?") $display("%s", tg);   // consume the sscanf tag (lint)

    repeat (5) @(negedge clk);
    checks++;
    if (viol_stable != 0)
      note_fail($sformatf("m_axis stability violations: %0d", viol_stable));

    print_coverage();
    $display("CONFIG %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
    if (fails == 0) begin
      $display("TB PASS [%s]: v=%0d k=%0d reads=%0d peeks=%0d (bp=%0d stall=%0d seed=%0d)",
               CFGNAME, cov_v_tok, cov_k_tok, cov_reads, cov_peeks, bp_mode,
               stall_mode, seed);
      $finish;
    end else begin
      $display("TB FAIL [%s]: %0d fail(s) across %0d checks", CFGNAME, fails,
               checks);
      $fatal(1, "TB FAIL");
    end
  end

endmodule
