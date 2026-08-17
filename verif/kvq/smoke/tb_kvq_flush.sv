// tb_kvq_flush.sv — D-008 regression: partial-group FLUSH.
//
// Streams T=70 key tokens at G=64 (one full group + a g=6 partial tail),
// pulses flush_req, then reads back ALL 70 tokens and checks the §4/D-026
// SRAM key records ([tag {ssid,1'b1}][D x int4 codes][pad], K_OUT=0 here),
// the persistent scale-bank rows (dut.g_key.u_bank_store — NOTE: each flushed
// partial group consumes one ssid, so the full group commits set 0 and the
// flushed tail set 1) and the fp32 readback bit-exact against the PYTHON
// golden (apex_golden.cq_codec pack_key_records via gen_kvq_vectors.py —
// which self-checks bit-identical against the frozen d64_T70_G64__CQ4 set,
// including the record/bank images, before this runs).
//
// Baseline behavior (-DBASELINE -GEXPECT_BUG=1, V0-patched top): no flush path
// exists — after the 6-token tail the engine wedges in ST_KACCEPT; the flush
// pulse (unconnected there) changes nothing; only a soft reset (destroying the
// tail) recovers. This reproduces the V0 partial-group demo as a REGRESSION.
//
// kvq_engine (EXPECT_BUG=0): flush_req quantizes the partial group through the
// existing datapath (per-channel scales over g=6 tokens, D-001 semantics),
// stores records at grp_base+idx, unwedges to idle, and a flush with no open
// group is a no-op.

`timescale 1ns/1ps

module tb_kvq_flush;

  parameter int D          = 64;
  parameter int TIER       = 1;
  parameter int G          = 64;
  parameter int T          = 70;
  parameter int DEPTH      = 128;
  parameter int EXPECT_BUG = 0;
  parameter     MAX_CYCLES = 20_000_000;
  parameter     VECDIR     = ".";   // frozen inputs (input_k.f16.hex)
  parameter     KEYDIR     = ".";   // python-golden expected (gen_kvq_vectors.py)
  parameter     CFGNAME    = "flush";

  localparam int TAIL = T - G;      // 6
  // §4/D-026 record layout mirror (K_OUT=0): key record [tag {ssid[6:0],1'b1}]
  // [D x int4 codes][pad->64b]. SRAMW mirrors cq_codec.sram_row_bits: keyed
  // engines unify the row to pad64(max(KEY_REC_RAW, VAL_REC_RAW)) — here the
  // CQ-4 VALUE record (8+16+4D = 280 raw) dominates the shrunken key record.
  localparam int KEY_CODES_LO = 8;
  localparam int KEY_REC_RAW  = 8 + D*4;
  localparam int VAL_REC_RAW  = 8 + 16 + D*4;
  localparam int REC_RAW      = (KEY_REC_RAW > VAL_REC_RAW) ? KEY_REC_RAW : VAL_REC_RAW;
  localparam int SRAMW        = 64 * ((REC_RAW + 63) / 64);
  localparam int ROWB         = SRAMW / 8;   // golden record-image bytes/row
  localparam int SCALE_SETS   = 4;           // engine D-026 default (unoverridden)

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
                         rresp, rvalid, evict_needed, evict_addr};
  reg         flush_r;              // driven by the TB; unconnected on baseline

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
    .flush_req(flush_r), .irq(irq_w),
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

  // ---------------- golden vectors --------------------------------------------
  logic [15:0] in_k   [0:16383];    // frozen inputs
  logic [31:0] khat   [0:16383];    // python-golden expected fp32
  logic [7:0]  krec   [0:4095];     // python-golden D-026 record image (T x ROWB)
  logic [15:0] bankg  [0:255];      // python-golden scale-bank rows (2 groups x D)

  // ---------------- bookkeeping ------------------------------------------------
  int checks = 0, fails = 0, printed = 0;
  localparam int MAXPRINT = 12;
  task automatic note_fail(input string msg);
    fails++;
    if (printed < MAXPRINT) begin printed++; $display("  FAIL: %s", msg); end
    else if (printed == MAXPRINT) begin printed++; $display("  (further FAIL prints suppressed)"); end
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

  task automatic stream_key_token(input int base);
    int d_;
    d_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      s_tdata  = in_k[base + d_];
      s_tvalid = 1; s_tuser = 0; s_tlast = (d_ == D - 1);
      if (s_tready) d_ = d_ + 1;
    end
    @(negedge clk); s_tvalid = 0; s_tlast = 0;
  endtask

  // §4/D-026 key record check vs the python-golden record image (emitted by
  // gen_kvq_vectors.py via cq_codec.pack_key_records — the packer is the
  // oracle, nothing re-derived here). No outliers in this config (K_OUT=0:
  // no lane fields). seq = per-engine group commit sequence: the full group
  // commits first (ssid 0); the FLUSHED partial group consumes the next ssid
  // (1). tok = global token index (golden image row), addr = SRAM address.
  logic [SRAMW-1:0] rec;
  task automatic check_keyrec(input int addr, input int seq, input int tok);
    int c_, b_;
    logic [SRAMW-1:0] gold;
    logic [6:0] ssid_e;
    rec = dut.u_sram.mem[addr % DEPTH];
    gold = '0;
    for (b_ = 0; b_ < ROWB; b_++) gold[b_*8 +: 8] = krec[tok*ROWB + b_];
    ssid_e = 7'(seq % SCALE_SETS);
    if (gold[7:0] !== {ssid_e, 1'b1})
      $fatal(1, "golden key_records row %0d tag %02h != {ssid=%0d,1'b1} — generated vectors/TB out of sync", tok, gold[7:0], ssid_e);
    checks++;
    if (rec[0] !== 1'b1)
      note_fail($sformatf("K rec a=%0d: tag bit0 %b != 1 (not a key record)", addr, rec[0]));
    checks++;
    if (rec[7:1] !== ssid_e)
      note_fail($sformatf("K rec a=%0d: ssid %0d exp %0d (commit seq %0d)", addr, rec[7:1], ssid_e, seq));
    for (c_ = 0; c_ < D; c_++) begin
      checks++;
      if (rec[KEY_CODES_LO + c_*4 +: 4] !== gold[KEY_CODES_LO + c_*4 +: 4])
        note_fail($sformatf("K rec a=%0d ch%0d: code %0h exp %0h", addr, c_,
                  rec[KEY_CODES_LO + c_*4 +: 4], gold[KEY_CODES_LO + c_*4 +: 4]));
    end
  endtask

  // D-026 bank peek: after commit `seq`, persistent scale set (seq %
  // SCALE_SETS) must equal the python-golden scale_bank.f16.hex row seq
  // (all D channels are keep channels here — no zero-forced outlier lanes).
  task automatic check_bank(input int seq);
    int c_, set_;
    set_ = seq % SCALE_SETS;
    for (c_ = 0; c_ < D; c_++) begin
      checks++;
      if (dut.g_key.u_bank_store.mem[set_][c_*16 +: 16] !== bankg[seq*D + c_])
        note_fail($sformatf("bank set%0d ch%0d: %04h exp %04h", set_, c_,
                  dut.g_key.u_bank_store.mem[set_][c_*16 +: 16], bankg[seq*D + c_]));
    end
  endtask

  task automatic check_hat(input int addr, input int tok);
    int d_, g_;
    awrite(REG_RA, addr);
    d_ = 0; g_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      if (m_tvalid) begin
        checks++;
        if (m_tlast !== ((d_ == D-1) ? 1'b1 : 1'b0))
          note_fail($sformatf("rd a=%0d beat=%0d: tlast=%b wrong", addr, d_, m_tlast));
        checks++;
        if (m_tdata !== khat[tok*D + d_])
          note_fail($sformatf("k_hat a=%0d tok=%0d d=%0d: got %08h exp %08h",
                    addr, tok, d_, m_tdata, khat[tok*D + d_]));
        d_ = d_ + 1;
      end
      g_++;
      if (g_ > 200000) begin note_fail($sformatf("rd a=%0d: burst hang", addr)); return; end
    end
  endtask

  // ---------------- flow ---------------------------------------------------------
  logic [31:0] rd32, occ_b, occ_a;
  bit ok;
  int t;

  initial begin
    rst_n = 0;
    awaddr = 0; awvalid = 0; wdata = 0; wvalid = 0; bready = 1;
    araddr = 0; arvalid = 0; rready = 0;
    s_tdata = 0; s_tvalid = 0; s_tlast = 0; s_tuser = 0;
    m_tready = 1; flush_r = 0;

    $display("==== tb_kvq_flush config=%s expect_bug=%0d (D-008: partial-group flush, T=%0d G=%0d tail g=%0d)",
             CFGNAME, EXPECT_BUG, T, G, TAIL);
    $readmemh({VECDIR, "/input_k.f16.hex"},        in_k);
    $readmemh({KEYDIR, "/expected_k_hat.f32.hex"}, khat);
    $readmemh({KEYDIR, "/key_records.u8.hex"},     krec);
    $readmemh({KEYDIR, "/scale_bank.f16.hex"},     bankg);

    repeat (8) @(posedge clk);
    rst_n = 1;
    repeat (4) @(posedge clk);
    awrite(REG_CTRL, 32'h2);   // enable

    // group 0: 64 key tokens at base 0 (natural full-group flush)
    $display("[F] streaming full key group 0 (G=%0d) at base 0", G);
    awrite(REG_WA, 0);
    for (t = 0; t < G; t++) stream_key_token(t*D);
    wait_idle(400000, ok);
    checks++; if (!ok) note_fail("engine not idle after full group 0");

    // partial tail: 6 key tokens at base 64 — engine parks in ST_KACCEPT
    $display("[F] streaming partial tail (g=%0d) at base %0d", TAIL, G);
    awrite(REG_WA, G);
    for (t = 0; t < TAIL; t++) stream_key_token((G + t)*D);
    repeat (200) @(posedge clk);
    aread(REG_STATUS, rd32);
    checks++;
    if ((rd32 & 32'h1) !== 32'h0) note_fail("engine idle with an open partial group (tail auto-flushed?!)");
    else $display("[F] confirmed: engine parked waiting for %0d more tokens (STATUS.idle=0)", G - TAIL);
    aread(REG_OCC, occ_b);

    // ---- THE FLUSH -----------------------------------------------------------
    $display("[F] pulsing flush_req");
    @(negedge clk); flush_r = 1;
    @(negedge clk); flush_r = 0;

    if (EXPECT_BUG == 1) begin
      // baseline: no flush path — the pulse (unconnected) must change nothing
      repeat (20000) @(posedge clk);
      aread(REG_STATUS, rd32);
      if ((rd32 & 32'h1) !== 32'h0)
        $fatal(1, "tb_kvq_flush [%s]: wedge NOT reproduced — engine went idle (a flush path exists?)", CFGNAME);
      aread(REG_OCC, occ_a);
      if (occ_a !== occ_b)
        $fatal(1, "tb_kvq_flush [%s]: wedge NOT reproduced — occupancy moved %0d->%0d", CFGNAME, occ_b, occ_a);
      $display("[B] confirmed: engine still wedged in ST_KACCEPT 20000 cycles after the flush pulse");
      $display("[B] confirmed: OCCUPANCY unchanged (%0d) — the %0d tail tokens remain stranded", occ_a, TAIL);
      awrite(REG_CTRL, 32'h3);   // soft reset destroys the tail but recovers
      repeat (4) @(posedge clk);
      aread(REG_STATUS, rd32);
      if ((rd32 & 32'h1) !== 32'h1)
        $fatal(1, "tb_kvq_flush [%s]: soft reset did not recover the baseline", CFGNAME);
      $display("D-008 RESULT [%s]: WEDGE REPRODUCED — no partial-group flush path; soft reset loses the tail", CFGNAME);
      $finish;
    end else begin
      // kvq: flush quantizes the tail through the datapath and unwedges
      wait_idle(400000, ok);
      checks++; if (!ok) note_fail("engine did not return to idle after flush_req (still wedged)");
      aread(REG_OCC, occ_a);
      checks++;
      if (occ_a !== 32'(T)) note_fail($sformatf("occupancy=%0d, expected %0d (tail records not stored)", occ_a, T));
      else $display("[F] flush emitted the tail: OCCUPANCY %0d -> %0d", occ_b, occ_a);

      $display("[F] checking %0d unified D-026 key records (full group + flushed tail) vs python golden", T);
      for (t = 0; t < G; t++)    check_keyrec(t, 0, t);
      for (t = 0; t < TAIL; t++) check_keyrec(G + t, 1, G + t);
      $display("[F] checking persistent scale-bank rows (commit seq 0 -> set 0; the flushed partial group consumed ssid 1)");
      check_bank(0);
      check_bank(1);

      $display("[F] reading back ALL %0d tokens (fp32) vs python golden", T);
      for (t = 0; t < T; t++) check_hat(t, t);

      // a flush with no open group is a no-op
      aread(REG_OCC, occ_b);
      @(negedge clk); flush_r = 1;
      @(negedge clk); flush_r = 0;
      repeat (100) @(posedge clk);
      aread(REG_STATUS, rd32);
      checks++; if ((rd32 & 32'h1) !== 32'h1) note_fail("no-op flush disturbed the idle engine");
      aread(REG_OCC, occ_a);
      checks++; if (occ_a !== occ_b) note_fail("no-op flush changed occupancy");

      $display("============================================================");
      $display("CONFIG %s: checks=%0d fails=%0d (irq=%b)", CFGNAME, checks, fails, irq_w);
      if (fails == 0) begin
        $display("D-008 RESULT [%s]: PASS — partial group (g=%0d) flushed, all %0d tokens bit-exact vs python golden", CFGNAME, TAIL, T);
        $finish;
      end else $fatal(1, "D-008 RESULT [%s]: FAIL", CFGNAME);
    end
  end

endmodule
