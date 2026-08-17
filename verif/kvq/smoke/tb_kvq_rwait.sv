// tb_kvq_rwait.sv — D-016(a) regression: read of an UNWRITTEN SRAM address.
//
// The unfixed-baseline behavior (proven here with -DBASELINE -GEXPECT_BUG=1):
// ST_RWAIT waits for sram_rd_valid, which never asserts for an address whose
// valid bit is clear -> the FSM hangs FOREVER (STATUS.idle stays 0, all
// subsequent reads silently dropped, only a soft reset recovers).
//
// kvq_engine required behavior (EXPECT_BUG=0):
//   * bounded wait: engine returns to idle within a few cycles;
//   * the read is dropped CLEANLY — zero m_axis beats;
//   * error response: IRQ_STATUS[0] (RD_ERR) set + STATUS[1] mirror + irq pin
//     (IRQ_MASK[0] armed); W1C on REG_IRQ_STATUS clears all three;
//   * the engine is immediately usable: a following valid read is bit-exact.
//
// Golden data: frozen d64_T128_G64__CQ4 value vectors (VECDIR).

`timescale 1ns/1ps

module tb_kvq_rwait;

  parameter int D          = 64;
  parameter int TIER       = 1;
  parameter int G          = 64;
  parameter int DEPTH      = 64;
  parameter int EXPECT_BUG = 0;
  parameter     MAX_CYCLES = 2_000_000;
  parameter     VECDIR     = ".";
  parameter     CFGNAME    = "rwait";

  // ---------------- clock / reset -------------------------------------------
  reg clk; initial begin clk = 1'b0; forever #5 clk = ~clk; end
  reg rst_n;

  // ---------------- DUT interface -------------------------------------------
  reg  [7:0]  awaddr;  reg awvalid;  wire awready;
  reg  [31:0] wdata;   reg wvalid;   wire wready;
  wire [1:0]  bresp;   wire bvalid;  reg bready;
  reg  [7:0]  araddr;  reg arvalid;  wire arready;
  wire [31:0] rdata;   wire [1:0] rresp; wire rvalid; reg rready;

  reg  [15:0] s_tdata; reg s_tvalid; wire s_tready; reg s_tlast, s_tuser;
  wire [31:0] m_tdata; wire m_tvalid; reg m_tready; wire m_tlast;
  wire        evict_needed;
  wire [$clog2(DEPTH)-1:0] evict_addr;
  wire        irq_w;
  wire dut_mask_valid;  // D-027 stage-4 port (sunk; the mask suite checks it)
  wire _tb_unused_ok = &{1'b0, dut_mask_valid, awready, wready, bresp, bvalid, arready,
                         rresp, rvalid, evict_needed, evict_addr, m_tlast};

`ifdef BASELINE
  // V0-patched engine (D-015 baseline): no flush_req / irq ports
  assign irq_w = 1'b0;
  kv_cache_engine #(
    .VECTOR_DIM(D), .TIER(TIER), .KEY_GROUP(G), .OUTLIER_K(0),
    .SCALE_WIDTH(16), .SRAM_DEPTH(DEPTH), .COORD_WIDTH(16), .OUT_WIDTH(32)
  ) dut (
`else
  kvq_engine #(
    .VECTOR_DIM(D), .TIER(TIER), .KEY_GROUP(G), .OUTLIER_K(0),
    .SCALE_WIDTH(16), .SRAM_DEPTH(DEPTH), .COORD_WIDTH(16), .OUT_WIDTH(32)
  ) dut (
    .flush_req(1'b0), .irq(irq_w),
`endif
    .clk(clk), .rst_n(rst_n),
    .axil_awaddr(awaddr), .axil_awvalid(awvalid), .axil_awready(awready),
    .axil_wdata(wdata), .axil_wvalid(wvalid), .axil_wready(wready),
    .axil_bresp(bresp), .axil_bvalid(bvalid), .axil_bready(bready),
    .axil_araddr(araddr), .axil_arvalid(arvalid), .axil_arready(arready),
    .axil_rdata(rdata), .axil_rresp(rresp), .axil_rvalid(rvalid), .axil_rready(rready),
    .s_axis_kv_tdata(s_tdata), .s_axis_kv_tvalid(s_tvalid), .s_axis_kv_tready(s_tready),
    .s_axis_kv_tlast(s_tlast), .s_axis_kv_tuser(s_tuser),
    .m_axis_kv_tdata(m_tdata), .m_axis_kv_tvalid(m_tvalid), .m_axis_kv_tready(m_tready),
    .m_axis_kv_tlast(m_tlast),
    .evict_needed(evict_needed), .evict_addr(evict_addr),
    .mask_valid(dut_mask_valid)
  );

  // ---------------- watchdog (MANDATORY) -------------------------------------
  initial begin
    #(MAX_CYCLES * 10);   // 10 ns clock period
    $fatal(1, "WATCHDOG: %s exceeded %0d cycles — DUT or TB hung", CFGNAME, MAX_CYCLES);
  end

  // ---------------- output-beat monitor ---------------------------------------
  int m_accepts;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)                       m_accepts <= 0;
    else if (m_tvalid && m_tready)    m_accepts <= m_accepts + 1;
  end

  // ---------------- golden vectors --------------------------------------------
  logic [15:0] in_v [0:16383];
  logic [31:0] vhat [0:16383];

  // ---------------- bookkeeping ------------------------------------------------
  int checks = 0, fails = 0;
  task automatic note_fail(input string msg);
    fails++;
    $display("  FAIL: %s", msg);
  endtask

  // ---------------- AXI-Lite tasks ---------------------------------------------
  localparam [7:0] REG_CTRL = 8'h00, REG_STATUS = 8'h04, REG_RA = 8'h2C,
                   REG_WA = 8'h28, REG_IRQ_MASK = 8'h34, REG_IRQ_STATUS = 8'h38;

  task automatic awrite(input [7:0] a, input [31:0] dv);
    @(negedge clk); awaddr = a; wdata = dv; awvalid = 1; wvalid = 1;
    @(negedge clk); awvalid = 0; wvalid = 0;
  endtask

  task automatic aread(input [7:0] a, output [31:0] dv);
    @(negedge clk); araddr = a; arvalid = 1; rready = 1;
    @(negedge clk); arvalid = 0;
    @(negedge clk); dv = rdata; rready = 0;
  endtask

  task automatic wait_idle(input int max_polls, output bit got_idle);
    int p; logic [31:0] st;
    p = 0; st = 0;
    while (((st & 32'h1) == 32'h0) && p < max_polls) begin aread(REG_STATUS, st); p++; end
    got_idle = ((st & 32'h1) == 32'h1);
  endtask

  task automatic stream_token(input int base);
    int d_;
    d_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      s_tdata  = in_v[base + d_];
      s_tvalid = 1; s_tuser = 1; s_tlast = (d_ == D - 1);
      if (s_tready) d_ = d_ + 1;
    end
    @(negedge clk); s_tvalid = 0; s_tlast = 0;
  endtask

  task automatic rd_check(input int addr, input int tok);
    int d_, g_;
    awrite(REG_RA, addr);
    d_ = 0; g_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      if (m_tvalid) begin
        checks++;
        if (m_tdata !== vhat[tok*D + d_])
          note_fail($sformatf("rd a=%0d d=%0d: got %08h exp %08h", addr, d_, m_tdata, vhat[tok*D + d_]));
        d_ = d_ + 1;
      end
      g_++;
      if (g_ > 200000) begin note_fail($sformatf("rd a=%0d: burst hang", addr)); return; end
    end
  endtask

  // ---------------- flow ---------------------------------------------------------
  logic [31:0] rd32;
  bit ok;
  int beats_before;

  initial begin
    rst_n = 0;
    awaddr = 0; awvalid = 0; wdata = 0; wvalid = 0; bready = 1;
    araddr = 0; arvalid = 0; rready = 0;
    s_tdata = 0; s_tvalid = 0; s_tlast = 0; s_tuser = 0;
    m_tready = 1;

    $display("==== tb_kvq_rwait config=%s expect_bug=%0d (D-016a: unwritten-address read)",
             CFGNAME, EXPECT_BUG);
    $readmemh({VECDIR, "/input_v.f16.hex"},        in_v);
    $readmemh({VECDIR, "/expected_v_hat.f32.hex"}, vhat);

    repeat (8) @(posedge clk);
    rst_n = 1;
    repeat (4) @(posedge clk);
    awrite(REG_CTRL, 32'h2);        // enable
    awrite(REG_IRQ_MASK, 32'h1);    // arm RD_ERR -> irq pin (kvq only; noop on baseline)

    // one valid token at addr 0 (sanity target for the recovery read)
    awrite(REG_WA, 0);
    stream_token(0);
    wait_idle(1000, ok);
    checks++; if (!ok) note_fail("engine not idle after value token");
    rd_check(0, 0);
    $display("[R] sanity round-trip at addr 0 OK (checks=%0d fails=%0d)", checks, fails);

    // ---- THE REGRESSION: read unwritten address 5 --------------------------
    repeat (4) @(posedge clk);      // let the sanity burst's last accept settle
    beats_before = m_accepts;
    awrite(REG_RA, 5);

    if (EXPECT_BUG == 1) begin
      // baseline: engine must wedge in ST_RWAIT forever
      repeat (2000) @(posedge clk);
      aread(REG_STATUS, rd32);
      if ((rd32 & 32'h1) !== 32'h0)
        $fatal(1, "tb_kvq_rwait [%s]: bug NOT reproduced — engine went idle after unwritten read", CFGNAME);
      if (m_accepts != beats_before)
        $fatal(1, "tb_kvq_rwait [%s]: bug NOT reproduced — beats emitted for unwritten address", CFGNAME);
      $display("[B] confirmed: STATUS.idle=0 2000 cycles after the unwritten-address read (ST_RWAIT hang)");
      // follow-up reads are silently swallowed while hung
      awrite(REG_RA, 0);
      repeat (2000) @(posedge clk);
      if (m_accepts != beats_before)
        $fatal(1, "tb_kvq_rwait [%s]: unexpected beats while hung", CFGNAME);
      $display("[B] confirmed: follow-up read of a VALID address dropped while hung (0 beats)");
      awrite(REG_CTRL, 32'h3);      // soft reset recovers
      repeat (4) @(posedge clk);
      aread(REG_STATUS, rd32);
      if ((rd32 & 32'h1) !== 32'h1)
        $fatal(1, "tb_kvq_rwait [%s]: soft reset did not recover the baseline", CFGNAME);
      $display("D-016a RESULT [%s]: BUG REPRODUCED — unwritten-address read hangs ST_RWAIT until soft reset", CFGNAME);
      $finish;
    end else begin
      // kvq_engine: bounded wait + clean drop + error response
      wait_idle(100, ok);
      checks++; if (!ok) note_fail("engine did not return to idle after unwritten read (bounded wait broken)");
      checks++; if (m_accepts != beats_before)
        note_fail($sformatf("unwritten read emitted %0d beats (must be 0)", m_accepts - beats_before));
      aread(REG_STATUS, rd32);
      checks++; if ((rd32 & 32'h2) !== 32'h2) note_fail("STATUS[1] (RD_ERR mirror) not set");
      aread(REG_IRQ_STATUS, rd32);
      checks++; if ((rd32 & 32'h1) !== 32'h1) note_fail("IRQ_STATUS[0] (RD_ERR) not set");
      checks++; if (irq_w !== 1'b1)   note_fail("irq pin not asserted with IRQ_MASK[0]=1");
      // W1C clears the sticky error
      awrite(REG_IRQ_STATUS, 32'h1);
      repeat (2) @(posedge clk);
      aread(REG_IRQ_STATUS, rd32);
      checks++; if ((rd32 & 32'h1) !== 32'h0) note_fail("IRQ_STATUS[0] not W1C-cleared");
      aread(REG_STATUS, rd32);
      checks++; if ((rd32 & 32'h2) !== 32'h0) note_fail("STATUS[1] not cleared after W1C");
      checks++; if (irq_w !== 1'b0)   note_fail("irq pin still asserted after W1C");
      // engine immediately usable — no soft reset needed
      rd_check(0, 0);
      $display("============================================================");
      $display("CONFIG %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
      if (fails == 0) begin
        $display("D-016a RESULT [%s]: PASS — bounded RWAIT, clean drop, RD_ERR irq + W1C, engine usable", CFGNAME);
        $finish;
      end else $fatal(1, "D-016a RESULT [%s]: FAIL", CFGNAME);
    end
  end

endmodule
