// tb_kvq_mask.sv — S12/D-027 loadable-outlier-mask suite (verif/kvq/mask).
//
// Two builds of this one TB prove the D-027 contract on kvq_engine
// (golden/tests/test_mask_semantics.py is the executable contract; every
// record/bank/fp32 expectation here comes from gen_mask_vectors.py, which
// imports the golden packers — nothing re-derived):
//
//   rom build (IS_CSR=0, MASKFILE=mask_m1.u8.hex):
//     R1  reset truth: MASK_CTRL = {owned:0, valid:1} (ROM valid out of hard
//         reset, D-027 §5.1), MASK0-3 read 0 (readback is the STAGED value,
//         not the live mask — §1)
//     R2  run1 stores (full group + flushed partial) + full fp32 readback,
//         bit-exact vs golden under the ROM mask
//
//   csr build (IS_CSR=1, no ROM — the b128 ship shape at D=64):
//     C1  reset truth: MASK_CTRL = {owned:0, valid:0}, MASK0-3 = 0
//     C2  commit legality (§3.2): popcount K-1 / K+1 / 0 commits are
//         REJECTED — sticky MASK_ERR (IRQ_STATUS[2], W1C), owned stays 0
//     C3  staged writes + legal commit at empty: staged readback, beyond-D
//         words RAZ/WI (write MASK2=FFFF_FFFF, reads 0, legality unaffected
//         — §5.7), commit -> no faults, {owned:1, valid:1}
//     C3b open-group swap flag at occupancy==0 (§5.5): one key token fed
//         (ST_KACCEPT, no records), re-commit -> MASK_SWAP via the
//         key-open term alone; W1C; soft reset discards the group and the
//         mask PERSISTS (§4)
//     C4  run1 — IDENTICAL vectors to the rom build: records, bank rows and
//         readback byte-identical to R2's by both matching the same golden
//     C5  swap while occupied (§3.3): stage+commit M2 -> commit EFFECTIVE +
//         sticky MASK_SWAP (occupancy>0); W1C
//     C6  run2 groupA under M2: ssid sequence CONTINUES (golden §B — the
//         expected tags carry ssid 2)
//     C7  D-020 soft reset mid-run2 -> mask/owned/valid PERSIST, then run2
//         groupB: records still bit-exact with ssid 3 (bank+allocator+mask
//         all survived — the expectations only hold if §4 holds)
//     C8  staged-vs-live split: re-stage M1 (no commit) -> MASK0 reads M1
//         while the live mask is still M2 (proven by C6/C7's records)
//     C9  fault-hygiene close: IRQ_STATUS reads 0 (no stray RD_ERR/SB_OVWR
//         — the geometry never wraps the allocator)
//
// Whitebox surfaces (house pattern): dut.u_sram.mem record peeks (smoke),
// the D-026 bank mirror over dut.g_key.u_bank_store.mem (audit), and the §5
// SVA pack ../sb/kvq_sb_sva.sv riding along (grep-gated "SVA-VIOL").
`default_nettype none

module tb_kvq_mask;

  // ── config (‑G overridden per build) ────────────────────────────────────────
  parameter int unsigned D        = 64;
  parameter int unsigned TIER     = 2;
  parameter int unsigned G        = 8;
  parameter int unsigned K_OUT    = 2;
  parameter int unsigned DEPTH    = 64;
  parameter int unsigned SSETS    = 4;
  parameter bit          IS_CSR   = 1;
  parameter              MASKFILE = "";
  parameter              CFGNAME  = "mask_csr";

  localparam int unsigned T1 = 12;          // run1 tokens (8 full + 4 partial)
  localparam int unsigned T2 = 16;          // run2 tokens (8 + 8)
  localparam int unsigned NTOK = T1 + T2;

  // engine row width (mirror of kvq_engine's derivation, TIER!=0)
  localparam int unsigned VAL_REC_RAW = 8 + 16 + D * 4;
  localparam int unsigned KEY_REC_RAW = 8 + K_OUT * 16 + D * 4;
  localparam int unsigned REC_RAW =
      (KEY_REC_RAW > VAL_REC_RAW) ? KEY_REC_RAW : VAL_REC_RAW;
  localparam int unsigned SRAMW = 64 * ((REC_RAW + 63) / 64);
  localparam int unsigned AW = $clog2(DEPTH);

  // ── engine CSR map (D-027 window included) ──────────────────────────────────
  localparam [7:0] A_CTRL      = 8'h00;
  localparam [7:0] A_STATUS    = 8'h04;
  localparam [7:0] A_WRITE_ADDR= 8'h28;
  localparam [7:0] A_READ_ADDR = 8'h2C;
  localparam [7:0] A_IRQ_STATUS= 8'h38;
  localparam [7:0] A_MASK0     = 8'h50;
  localparam [7:0] A_MASK1     = 8'h54;
  localparam [7:0] A_MASK2     = 8'h58;
  localparam [7:0] A_MASK3     = 8'h5C;
  localparam [7:0] A_MASK_CTRL = 8'h60;

  // M1 = {5,50}: word0 bit5, word1 bit18 · M2 = {5,60}: word0 bit5, word1 bit28
  localparam [31:0] M1_W0 = 32'h0000_0020, M1_W1 = 32'h0004_0000;
  localparam [31:0] M2_W0 = 32'h0000_0020, M2_W1 = 32'h1000_0000;

  // ── clock / reset ───────────────────────────────────────────────────────────
  logic clk = 1'b0;
  logic rst_n;
  int   cyc;
  initial forever #5 clk = ~clk;
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
  wire         dut_mask_valid;   // D-027 stage-4 port — the tile INFO_TIER term

  kvq_engine #(
    .VECTOR_DIM(D), .TIER(TIER), .KEY_GROUP(G), .OUTLIER_K(K_OUT),
    .SCALE_SETS(SSETS), .SCALE_WIDTH(16), .SRAM_DEPTH(DEPTH),
    .COORD_WIDTH(16), .OUT_WIDTH(32), .MASK_FILE(MASKFILE)
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

  // §5 stream SVA pack (grep-gated: prints "SVA-VIOL" on violation)
  kvq_sb_sva #(.D(D), .DW(16), .OW(32)) u_sva (
    .clk(clk), .rst_n(rst_n),
    .s_valid(s_tvalid), .s_ready(s_tready), .s_data(s_tdata),
    .s_last(s_tlast), .s_user(s_tuser),
    .m_valid(m_tvalid), .m_ready(m_tready), .m_data(m_tdata),
    .m_last(m_tlast),
    .fsm_state(dut.state)
  );

  wire _tb_unused_ok = &{1'b0, bresp, rresp, irq_w, evict_needed, evict_addr};

  // ── D-026 bank mirror (audit pattern; TIER==2 always elaborates g_key) ──────
  wire [SSETS*D*16-1:0] bank_flat;
  generate
    if (TIER != 0) begin : g_bank_mirror
      for (genvar s = 0; s < SSETS; s++) begin : g_set
        assign bank_flat[s*D*16 +: D*16] = dut.g_key.u_bank_store.mem[s];
      end
    end else begin : g_bank_zero
      assign bank_flat = '0;
    end
  endgenerate

  // ── vectors ─────────────────────────────────────────────────────────────────
  logic [15:0]      stim     [0:NTOK*D-1];
  logic [SRAMW-1:0] exp_rec  [0:NTOK-1];
  logic [D*16-1:0]  exp_bank [0:3];
  logic [31:0]      exp_hat  [0:NTOK*D-1];
  string vecdir;
  initial begin
    if (!$value$plusargs("vecdir=%s", vecdir)) vecdir = "build/vec";
    $readmemh({vecdir, "/stim.f16.hex"},     stim);
    $readmemh({vecdir, "/exp_rec.hex"},      exp_rec);
    $readmemh({vecdir, "/exp_bank.f16.hex"}, exp_bank);
    $readmemh({vecdir, "/exp_hat.f32.hex"},  exp_hat);
  end

  // ── watchdog (MANDATORY) ────────────────────────────────────────────────────
  int unsigned watchdog_cyc = 2_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: %s did not finish within %0d cycles", CFGNAME,
           watchdog_cyc);
  end

  // ── bookkeeping ─────────────────────────────────────────────────────────────
  int checks, fails, printed;
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

  task automatic check32(input string what, input [31:0] got, input [31:0] exp);
    checks++;
    if (got !== exp)
      note_fail($sformatf("%s: got %08h exp %08h", what, got, exp));
  endtask

  // ── CSR tasks (single-flight AXI-Lite; bready/rready pinned 1) ──────────────
  task automatic csr_wr(input [7:0] a, input [31:0] d);
    begin
      @(negedge clk);
      awaddr = a; wdata = d; awvalid = 1'b1; wvalid = 1'b1;
      @(posedge clk);
      while (!(awready && wready)) @(posedge clk);
      @(negedge clk);
      awvalid = 1'b0; wvalid = 1'b0;
      while (!bvalid) @(posedge clk);
      @(negedge clk);
    end
  endtask

  task automatic csr_rd(input [7:0] a, output [31:0] d);
    begin
      @(negedge clk);
      araddr = a; arvalid = 1'b1;
      @(posedge clk);
      while (!arready) @(posedge clk);
      @(negedge clk);
      arvalid = 1'b0;
      while (!rvalid) @(posedge clk);
      d = rdata;
      @(negedge clk);
    end
  endtask

  task automatic csr_check(input string what, input [7:0] a, input [31:0] exp);
    logic [31:0] v;
    begin
      csr_rd(a, v);
      check32(what, v, exp);
    end
  endtask

  // D-027 stage 4: the exported mask_valid port (what the tile INFO_TIER
  // bit-2 term consumes) must agree with the CSR-visible truth
  task automatic check_mv(input string what, input logic exp);
    checks++;
    if (dut_mask_valid !== exp)
      note_fail($sformatf("%s: mask_valid port %b exp %b", what,
                          dut_mask_valid, exp));
  endtask

  task automatic wait_idle();
    logic [31:0] st;
    // full-word mask compare (audit csr_poll shape) — consumes all 32 bits,
    // -Wall/UNUSEDSIGNAL clean
    do csr_rd(A_STATUS, st); while ((st & 32'h1) == 32'h0);
  endtask

  // ── stream tasks (negedge-drive; the negedge tready sample is exactly what
  //    the next posedge consumes — single-accept safe) ─────────────────────────
  task automatic send_tok(input int unsigned t);
    begin
      for (int unsigned b = 0; b < D; b++) begin
        @(negedge clk);
        s_tdata  = stim[t*D + b];
        s_tvalid = 1'b1;
        s_tlast  = (b == D - 1);
        s_tuser  = 1'b0;              // key stream
        while (s_tready !== 1'b1) @(negedge clk);
      end
      @(negedge clk);
      s_tvalid = 1'b0; s_tlast = 1'b0;
    end
  endtask

  // store ntok tokens at base (addr == token index by the address plan),
  // closing a partial group with a D-008 flush pulse
  task automatic store_group(input int unsigned base, input int unsigned tok0,
                             input int unsigned ntok, input bit do_flush);
    begin
      csr_wr(A_WRITE_ADDR, 32'(base));
      for (int unsigned t = 0; t < ntok; t++) send_tok(tok0 + t);
      if (do_flush) begin
        @(negedge clk); flush_r = 1'b1;
        @(negedge clk); flush_r = 1'b0;
      end
      wait_idle();
    end
  endtask

  // fp32 readback of one record vs golden (D data beats + 1 tlast check).
  // ONE negedge advance per beat: with m_tready pinned 1 the engine emits a
  // beat per cycle, and each loop iteration's entry @(negedge) lands on the
  // next beat exactly (a second wait here made the first suite run sample
  // every OTHER beat — got[k]==golden[2k], the classic double-advance).
  task automatic read_tok(input int unsigned addr, input int unsigned t);
    begin
      csr_wr(A_READ_ADDR, 32'(addr));
      for (int unsigned b = 0; b < D; b++) begin
        @(negedge clk);
        while (m_tvalid !== 1'b1) @(negedge clk);
        check32($sformatf("hat t%0d ch%0d", t, b), m_tdata, exp_hat[t*D + b]);
        if (b == D - 1) begin
          checks++;
          if (m_tlast !== 1'b1)
            note_fail($sformatf("hat t%0d: no tlast on final beat", t));
        end
      end
      wait_idle();
    end
  endtask

  // record + bank checks (whitebox peeks, full-row equality)
  task automatic check_rec(input int unsigned addr, input int unsigned t);
    checks++;
    if (dut.u_sram.mem[addr] !== exp_rec[t])
      note_fail($sformatf("rec t%0d @%0d: got %h exp %h", t, addr,
                          dut.u_sram.mem[addr], exp_rec[t]));
  endtask

  task automatic check_bank(input int unsigned set_i);
    checks++;
    if (bank_flat[set_i*D*16 +: D*16] !== exp_bank[set_i])
      note_fail($sformatf("bank set %0d mismatch", set_i));
  endtask

  // run1/run2 phase runners (shared between builds)
  task automatic do_run1(input string tag);
    begin
      store_group(0, 0, 8, 1'b0);            // group0 full  -> ssid 0
      store_group(8, 8, 4, 1'b1);            // group1 partial(4), flush -> ssid 1
      for (int unsigned t = 0; t < T1; t++) check_rec(t, t);
      check_bank(0);
      check_bank(1);
      for (int unsigned t = 0; t < T1; t++) read_tok(t, t);
      $display("[%s] run1 stored+verified (%0d records, 2 banks)", tag, T1);
    end
  endtask

  // ── the two flows ───────────────────────────────────────────────────────────
  task automatic flow_rom();
    begin
      // R1: build-default truth out of hard reset
      csr_check("R1 MASK_CTRL (rom: owned=0 valid=1)", A_MASK_CTRL, 32'h1);
      check_mv("R1 mask_valid port (ROM valid)", 1'b1);
      csr_check("R1 MASK0 staged==0", A_MASK0, '0);
      csr_check("R1 MASK1 staged==0", A_MASK1, '0);
      csr_check("R1 MASK2 staged==0", A_MASK2, '0);
      csr_check("R1 MASK3 staged==0", A_MASK3, '0);
      // R2: run1 under the ROM mask
      do_run1("rom");
    end
  endtask

  task automatic flow_csr();
    begin
      // C1: maskless build truth out of hard reset
      csr_check("C1 MASK_CTRL (csr: owned=0 valid=0)", A_MASK_CTRL, '0);
      check_mv("C1 mask_valid port (maskless invalid)", 1'b0);
      csr_check("C1 MASK0==0", A_MASK0, '0);
      csr_check("C1 MASK1==0", A_MASK1, '0);
      csr_check("C1 MASK2==0", A_MASK2, '0);
      csr_check("C1 MASK3==0", A_MASK3, '0);

      // C2: illegal commits (popcount K-1 / K+1 / 0) are rejected + flagged
      csr_wr(A_MASK0, M1_W0);                          // popcount 1
      csr_wr(A_MASK_CTRL, 32'h1);
      csr_check("C2a MASK_ERR sticky", A_IRQ_STATUS, 32'h4);
      csr_check("C2a owned still 0", A_MASK_CTRL, '0);
      csr_wr(A_IRQ_STATUS, 32'h4);                     // W1C
      csr_check("C2a W1C cleared", A_IRQ_STATUS, '0);
      csr_wr(A_MASK1, M1_W1 | 32'h1);                  // popcount 3
      csr_wr(A_MASK_CTRL, 32'h1);
      csr_check("C2b MASK_ERR sticky", A_IRQ_STATUS, 32'h4);
      csr_check("C2b owned still 0", A_MASK_CTRL, '0);
      csr_wr(A_IRQ_STATUS, 32'h4);
      csr_check("C2b W1C cleared", A_IRQ_STATUS, '0);
      csr_wr(A_MASK0, '0);                             // popcount 0
      csr_wr(A_MASK1, '0);
      csr_wr(A_MASK_CTRL, 32'h1);
      csr_check("C2c MASK_ERR sticky", A_IRQ_STATUS, 32'h4);
      csr_check("C2c owned still 0", A_MASK_CTRL, '0);
      csr_wr(A_IRQ_STATUS, 32'h4);
      csr_check("C2c W1C cleared", A_IRQ_STATUS, '0);

      // C3: stage M1, beyond-D RAZ/WI, legal commit at empty
      csr_wr(A_MASK0, M1_W0);
      csr_wr(A_MASK1, M1_W1);
      csr_wr(A_MASK2, 32'hFFFF_FFFF);                  // beyond D=64: WI
      csr_check("C3 MASK0 staged", A_MASK0, M1_W0);
      csr_check("C3 MASK1 staged", A_MASK1, M1_W1);
      csr_check("C3 MASK2 RAZ", A_MASK2, '0);
      csr_check("C3 MASK3 RAZ", A_MASK3, '0);
      csr_wr(A_MASK_CTRL, 32'h1);
      csr_check("C3 no faults", A_IRQ_STATUS, '0);
      csr_check("C3 owned+valid", A_MASK_CTRL, 32'h3);
      check_mv("C3 mask_valid port after commit", 1'b1);

      // C3b: open-group swap flag with occupancy==0 (the key-open term)
      csr_wr(A_WRITE_ADDR, 32'd40);                    // scratch base
      send_tok(0);                                     // one token -> ST_KACCEPT
      csr_wr(A_MASK_CTRL, 32'h1);                      // re-commit M1
      csr_check("C3b MASK_SWAP via key-open", A_IRQ_STATUS, 32'h8);
      csr_wr(A_IRQ_STATUS, 32'h8);
      csr_check("C3b W1C cleared", A_IRQ_STATUS, '0);
      csr_wr(A_CTRL, 32'h3);                           // soft reset (keep enable)
      wait_idle();
      csr_check("C3b mask persists soft reset", A_MASK_CTRL, 32'h3);
      check_mv("C3b mask_valid port persists", 1'b1);

      // C4: run1 — byte-identical to the rom build by construction
      do_run1("csr");

      // C5: swap while occupied — commit effective + MASK_SWAP sticky
      csr_wr(A_MASK1, M2_W1);                          // {5,60}
      csr_wr(A_MASK_CTRL, 32'h1);
      csr_check("C5 MASK_SWAP sticky", A_IRQ_STATUS, 32'h8);
      csr_check("C5 owned+valid (commit effective)", A_MASK_CTRL, 32'h3);
      csr_wr(A_IRQ_STATUS, 32'h8);
      csr_check("C5 W1C cleared", A_IRQ_STATUS, '0);

      // C6: run2 groupA under M2 — ssid continues (exp tags carry ssid 2)
      store_group(12, T1, 8, 1'b0);
      for (int unsigned t = T1; t < T1 + 8; t++) check_rec(t, t);
      check_bank(2);
      for (int unsigned t = T1; t < T1 + 8; t++) read_tok(t, t);

      // C7: soft reset mid-run2 — mask + bank + allocator persist
      csr_wr(A_CTRL, 32'h3);
      wait_idle();
      csr_check("C7 owned+valid persist", A_MASK_CTRL, 32'h3);
      csr_check("C7 MASK0 staged persists", A_MASK0, M2_W0);
      csr_check("C7 MASK1 staged persists", A_MASK1, M2_W1);
      store_group(20, T1 + 8, 8, 1'b0);
      for (int unsigned t = T1 + 8; t < NTOK; t++) check_rec(t, t);
      check_bank(3);
      for (int unsigned t = T1 + 8; t < NTOK; t++) read_tok(t, t);

      // C8: staged readback is the STAGE, not the live mask
      csr_wr(A_MASK1, M1_W1);                          // re-stage M1 (no commit)
      csr_check("C8 MASK1 staged=M1 (live=M2)", A_MASK1, M1_W1);
      csr_check("C8 owned+valid unchanged", A_MASK_CTRL, 32'h3);

      // C9: no stray faults anywhere in the flow
      csr_check("C9 IRQ_STATUS clean", A_IRQ_STATUS, '0);
    end
  endtask

  // ── main ────────────────────────────────────────────────────────────────────
  initial begin
    awaddr = '0; awvalid = 0; wdata = '0; wvalid = 0; bready = 1'b1;
    araddr = '0; arvalid = 0; rready = 1'b1;
    s_tdata = '0; s_tvalid = 0; s_tlast = 0; s_tuser = 0;
    m_tready = 1'b1; flush_r = 0;
    checks = 0; fails = 0; printed = 0;
    rst_n = 1'b0;
    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    csr_wr(A_CTRL, 32'h2);                             // enable
    if (IS_CSR) flow_csr();
    else        flow_rom();

    $display("MASK GATE %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
    if (fails != 0) $fatal(1, "MASK GATE %s: %0d FAILURES", CFGNAME, fails);
    $display("MASK GATE %s: ALL PASS", CFGNAME);
    $finish;
  end

endmodule
`default_nettype wire
