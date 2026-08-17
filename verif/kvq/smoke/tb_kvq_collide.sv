// tb_kvq_collide.sv — D-016(b) regression: ST_IDLE read_req vs accepted s_axis
// beat collision.
//
// The TB times an AXI-Lite READ_ADDR write so its registered read_req pulse
// lands in EXACTLY the cycle where the first beat of the next value token is
// accepted (s_tvalid && s_tready handshake in ST_IDLE).
//
// Baseline behavior (-DBASELINE -GEXPECT_BUG=1, V0-patched top): read_req wins
// and the ACCEPTED beat is silently discarded; worse, tready stays high during
// the read walk, so the token's remaining beats are also handshaken and lost.
// Signature: all 64 beats accepted, the read burst plays DURING the stream,
// engine ends idle, OCCUPANCY never moves — the token evaporated.
//
// kvq_engine required behavior (EXPECT_BUG=0): the accepted beat wins; the
// token is collected/compressed/stored bit-exact; the read is DEFERRED (not
// dropped) and its burst appears only after the token completes, bit-exact.
//
// Golden data: frozen d64_T128_G64__CQ4 value vectors (VECDIR).

`timescale 1ns/1ps

module tb_kvq_collide;

  parameter int D          = 64;
  parameter int TIER       = 1;
  parameter int G          = 64;
  parameter int DEPTH      = 64;
  parameter int EXPECT_BUG = 0;
  parameter     MAX_CYCLES = 2_000_000;
  parameter     VECDIR     = ".";
  parameter     CFGNAME    = "collide";

  // §4/D-026 record layout mirror (kvq peeks only). SRAMW must follow the
  // engine's unified-row rule (mirrors cq_codec.sram_row_bits): keyed engines
  // pad64(max(KEY_REC_RAW = 8 + 16*K_OUT + 4D, VAL_REC_RAW = 8 + 16 + 4D)).
  // Here K_OUT=0, CQ-4: max(264, 280) -> 320 bits — the VALUE record
  // dominates the D-026-shrunken key record (it no longer carries D fields).
  localparam int VAL_SCALE_LO = 8;
  localparam int VAL_PAY_LO   = 24;
  localparam int VPB          = D / 2;   // CQ-4 payload bytes/token
  localparam int KEY_REC_RAW  = 8 + 4*D;        // D-026 key record (K_OUT=0)
  localparam int VAL_REC_RAW  = 8 + 16 + 4*D;   // CQ-4 value record
  localparam int REC_RAW      = (KEY_REC_RAW > VAL_REC_RAW) ? KEY_REC_RAW : VAL_REC_RAW;
  localparam int SRAMW        = 64 * ((REC_RAW + 63) / 64);

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
                         rresp, rvalid, evict_needed, evict_addr, m_accepts, m_tlast};

`ifdef BASELINE
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

  // ---------------- stream monitors -------------------------------------------
  int s_accepts;                   // s_axis handshakes (producer-side truth)
  int m_accepts;                   // m_axis beats accepted
  bit streaming = 0;               // TB is mid-token
  int m_during_stream;             // read-burst beats that played DURING the stream
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_accepts <= 0; m_accepts <= 0; m_during_stream <= 0;
    end else begin
      if (s_tvalid && s_tready) s_accepts <= s_accepts + 1;
      if (m_tvalid && m_tready) begin
        m_accepts <= m_accepts + 1;
        if (streaming) m_during_stream <= m_during_stream + 1;
      end
    end
  end

  // ---------------- golden vectors --------------------------------------------
  logic [15:0] in_v   [0:16383];
  logic [31:0] vhat   [0:16383];
  logic [15:0] vscale [0:511];
  logic [7:0]  vpay   [0:16383];

  // ---------------- bookkeeping ------------------------------------------------
  int checks = 0, fails = 0;
  task automatic note_fail(input string msg);
    fails++;
    $display("  FAIL: %s", msg);
  endtask

  // ---------------- AXI-Lite tasks ---------------------------------------------
  localparam [7:0] REG_CTRL = 8'h00, REG_STATUS = 8'h04, REG_OCC = 8'h24,
                   REG_WA = 8'h28, REG_RA = 8'h2C;

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

  // collect one full read burst on m_axis and compare vs vhat[tok]
  task automatic collect_burst(input int tok, input string what);
    int d_, g_;
    d_ = 0; g_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      if (m_tvalid) begin
        checks++;
        if (m_tdata !== vhat[tok*D + d_])
          note_fail($sformatf("%s d=%0d: got %08h exp %08h", what, d_, m_tdata, vhat[tok*D + d_]));
        d_ = d_ + 1;
      end
      g_++;
      if (g_ > 200000) begin note_fail($sformatf("%s: burst never arrived/hung", what)); return; end
    end
  endtask

  // ---------------- flow ---------------------------------------------------------
  logic [31:0] rd32, occ;
  bit ok;
  int d_;

  initial begin
    rst_n = 0;
    awaddr = 0; awvalid = 0; wdata = 0; wvalid = 0; bready = 1;
    araddr = 0; arvalid = 0; rready = 0;
    s_tdata = 0; s_tvalid = 0; s_tlast = 0; s_tuser = 0;
    m_tready = 1;

    $display("==== tb_kvq_collide config=%s expect_bug=%0d (D-016b: read_req vs accepted beat)",
             CFGNAME, EXPECT_BUG);
    $readmemh({VECDIR, "/input_v.f16.hex"},        in_v);
    $readmemh({VECDIR, "/expected_v_hat.f32.hex"}, vhat);
    $readmemh({VECDIR, "/val_scales.f16.hex"},     vscale);
    $readmemh({VECDIR, "/val_payload.u8.hex"},     vpay);

    repeat (8) @(posedge clk);
    rst_n = 1;
    repeat (4) @(posedge clk);
    awrite(REG_CTRL, 32'h2);   // enable

    // token 0 -> addr 0: a valid target for the colliding read
    awrite(REG_WA, 0);
    stream_token(0);
    wait_idle(1000, ok);
    checks++; if (!ok) note_fail("engine not idle after token 0");
    $display("[C] token 0 stored at addr 0; arming the collision");

    // ---- THE COLLISION -------------------------------------------------------
    // AXI-Lite write cycle N -> registered read_req pulses in cycle N+1; the
    // first beat of token 1 goes valid in exactly that cycle (tready is high
    // in ST_IDLE), so the beat handshake and read_req coincide.
    awrite(REG_WA, 1);
    streaming = 1;
    @(negedge clk); awaddr = REG_RA; wdata = 0; awvalid = 1; wvalid = 1;
    @(negedge clk); awvalid = 0; wvalid = 0;
      s_tdata = in_v[D + 0]; s_tvalid = 1; s_tuser = 1; s_tlast = 0;  // beat 0 NOW
    d_ = 0;
    while (d_ < D) begin
      if (s_tready) d_ = d_ + 1;
      @(negedge clk);
      if (d_ < D) begin
        s_tdata = in_v[D + d_]; s_tlast = (d_ == D - 1);
      end else begin
        s_tvalid = 0; s_tlast = 0;
      end
    end
    streaming = 0;
    checks++;
    if (s_accepts != 2*D)
      note_fail($sformatf("producer saw %0d handshakes, expected %0d — collision framing off", s_accepts, 2*D));

    if (EXPECT_BUG == 1) begin
      // baseline signature: burst played DURING the stream, token evaporated
      repeat (2000) @(posedge clk);
      aread(REG_OCC, occ);
      aread(REG_STATUS, rd32);
      if (fails != 0)
        $fatal(1, "tb_kvq_collide [%s]: harness framing errors", CFGNAME);
      if (m_during_stream == 0)
        $fatal(1, "tb_kvq_collide [%s]: bug NOT reproduced — no read burst during the stream", CFGNAME);
      if (occ !== 32'd1)
        $fatal(1, "tb_kvq_collide [%s]: bug NOT reproduced — occupancy=%0d (token was stored?)", CFGNAME, occ);
      if ((rd32 & 32'h1) !== 32'h1)
        $fatal(1, "tb_kvq_collide [%s]: engine not idle after the lost token", CFGNAME);
      $display("[B] confirmed: read burst played DURING the stream (%0d beats), all %0d write beats", m_during_stream, s_accepts);
      $display("[B] confirmed: handshaken but engine OCCUPANCY still 1 and idle — the accepted token is LOST");
      $display("D-016b RESULT [%s]: BUG REPRODUCED — colliding read_req discards the accepted s_axis beat(s)", CFGNAME);
      $finish;
    end else begin
      // kvq: the beat wins, the read is deferred — burst must appear only now
      checks++;
      if (m_during_stream != 0)
        note_fail($sformatf("%0d read beats played DURING the stream (read must be deferred)", m_during_stream));
      collect_burst(0, "deferred read of addr 0");
      wait_idle(1000, ok);
      checks++; if (!ok) note_fail("engine not idle after deferred read");
      aread(REG_OCC, occ);
      checks++; if (occ !== 32'd2)
        note_fail($sformatf("occupancy=%0d, expected 2 (token 1 must be stored)", occ));
      // token 1's record is bit-exact (§4/D-026 layout peek)
      begin
        logic [SRAMW-1:0] rec;   // full §4/D-026 unified SRAM row
        int b_;
        rec = dut.u_sram.mem[1] ;
        for (b_ = 0; b_ < VPB; b_++) begin
          checks++;
          if (rec[VAL_PAY_LO + b_*8 +: 8] !== vpay[VPB + b_])
            note_fail($sformatf("tok1 rec paybyte %0d: got %02h exp %02h", b_, rec[VAL_PAY_LO + b_*8 +: 8], vpay[VPB + b_]));
        end
        checks++;
        if (rec[VAL_SCALE_LO +: 16] !== vscale[1])
          note_fail($sformatf("tok1 rec scale: got %04h exp %04h", rec[VAL_SCALE_LO +: 16], vscale[1]));
      end
      // and reads back bit-exact
      awrite(REG_RA, 1);
      collect_burst(1, "readback of addr 1");
      $display("============================================================");
      $display("CONFIG %s: checks=%0d fails=%0d (irq=%b)", CFGNAME, checks, fails, irq_w);
      if (fails == 0) begin
        $display("D-016b RESULT [%s]: PASS — accepted beat wins, token stored bit-exact, read deferred not dropped", CFGNAME);
        $finish;
      end else $fatal(1, "D-016b RESULT [%s]: FAIL", CFGNAME);
    end
  end

endmodule
