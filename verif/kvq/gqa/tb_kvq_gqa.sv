// tb_kvq_gqa.sv — IB-LAYER S4b bank-level TB for apex_kvq_gqa_bank:
// N_ENG=4 per-KV-head CQ-8 engines behind the live eng_sel mux.
//
// Golden arbiter: build/vec/* from gen_gqa_vectors.py (cq_codec CQ-8
// compress/decompress_values — the D-001 primitives, never re-derived).
// The engines are the verified kvq_engine; what THIS TB proves is the
// BANK's routing contract (apex_kvq_gqa_bank.sv header):
//
//   B1  per-engine identity: INFO regs / CTRL / OCCUPANCY read from the
//       SELECTED engine (per-engine NREC differ, so occupancy distinguishes
//       engines structurally);
//   B2  store/readback isolation: distinct-by-construction records stored
//       per engine at OVERLAPPING addresses read back bit-exact per engine
//       (a select miswire can never alias — generator self-check);
//   B3  same-address rewrite on ONE engine leaves the other engines'
//       records at that address untouched;
//   B4  D-016a error isolation: an unwritten-address read latches RD_ERR
//       on the selected engine ONLY; irq is the OR across engines (visible
//       while another engine is selected); W1C recovers;
//   B5  m_pending mirrors any pending m_axis beat under backpressure
//       (§5 no-retract), engine-independent of the select;
//   B6  D-008 flush on an idle CQ-8 engine is the documented no-op.
//
// Stimulus discipline: negedge-driven (the kvq suite pattern — the negedge
// tready/handshake sample is exactly what the next posedge consumes);
// eng_sel changes ONLY while the bank is quiescent (selected engine idle
// polled, no m_axis beat pending) — the routing contract.
//
// The §5 stream SVA pack (verif/kvq/smoke/kvq_axis_sva.sv) is compiled into
// this build: its `bind kvq_engine` rides into ALL N_ENG engine instances,
// so per-engine stream legality is asserted on the gated streams for free.
// The run gate greps the log for "[SVA" (any assertion print fails the run).
//
// WATCHDOG SIGNATURE: the watchdog $fatal prints the live `phase` string —
// the mutation gate (mutate_gqa.py) keys hang-class mutants on it, so keep
// phase updates honest.

`default_nettype none

module tb_kvq_gqa;

  // geometry (Makefile -G; keep in lockstep with gen_gqa_vectors.py)
  parameter int unsigned D       = 64;
  parameter int unsigned N_ENG   = 4;
  parameter int unsigned DEPTH   = 256;
  parameter int unsigned G       = 16;
  parameter int unsigned SSETS   = 4;
  parameter CFGNAME = "gqa4";

  localparam int unsigned ENG_W = (N_ENG > 1) ? $clog2(N_ENG) : 1;
  localparam int unsigned AW    = $clog2(DEPTH);

  // per-engine record plan (generator lockstep; NREC distinct per engine)
  localparam int unsigned NREC [4] = '{4, 5, 6, 7};
  localparam int unsigned OFF  [4] = '{0, 4, 9, 15};
  localparam int unsigned TOTREC   = 22;
  localparam int unsigned REW_E    = 2;   // rewrite target (e2, rec 1)
  localparam int unsigned REW_R    = 1;

  if (N_ENG != 4) begin : g_chk
    $error("tb_kvq_gqa: the record plan is pinned to N_ENG=4");
  end

  // ── clock / DUT nets ────────────────────────────────────────────────────────
  logic clk = 1'b0;
  initial forever #5 clk = ~clk;
  int unsigned cyc;
  always @(posedge clk) cyc <= cyc + 1;

  logic rst_n;
  logic [ENG_W-1:0] eng_sel;
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
  wire         m_pending;

  apex_kvq_gqa_bank #(
    .N_ENG(N_ENG), .CFG_D(D), .KVQ_G(G), .KVQ_DEPTH(DEPTH), .KVQ_SETS(SSETS)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .eng_sel(eng_sel),
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
    .m_pending(m_pending)
  );

  wire _tb_unused_ok = &{1'b0, bresp, rresp, evict_needed, evict_addr};

  // engine CSR map (kvq_engine REG_*)
  localparam logic [7:0] A_CTRL       = 8'h00;
  localparam logic [7:0] A_STATUS     = 8'h04;
  localparam logic [7:0] A_INFO_DIM   = 8'h08;
  localparam logic [7:0] A_INFO_TIER  = 8'h0C;
  localparam logic [7:0] A_INFO_DEPTH = 8'h14;
  localparam logic [7:0] A_OCCUPANCY  = 8'h24;
  localparam logic [7:0] A_WRITE_ADDR = 8'h28;
  localparam logic [7:0] A_READ_ADDR  = 8'h2C;
  localparam logic [7:0] A_IRQ_MASK   = 8'h34;
  localparam logic [7:0] A_IRQ_STATUS = 8'h38;

  // ── vectors ─────────────────────────────────────────────────────────────────
  logic [15:0] stim  [0:TOTREC*D-1];
  logic [31:0] exp_h [0:TOTREC*D-1];
  logic [15:0] stim2 [0:D-1];
  logic [31:0] exp2h [0:D-1];
  string vecdir;
  initial begin
    if (!$value$plusargs("vecdir=%s", vecdir)) vecdir = "build/vec";
    $readmemh({vecdir, "/stim.f16.hex"},     stim);
    $readmemh({vecdir, "/exp_hat.f32.hex"},  exp_h);
    $readmemh({vecdir, "/stim2.f16.hex"},    stim2);
    $readmemh({vecdir, "/exp2_hat.f32.hex"}, exp2h);
  end

  // ── watchdog (MANDATORY; prints the live phase — the hang signature) ───────
  string phase = "init";
  int unsigned watchdog_cyc = 2_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: %s hung in phase=%s after %0d cycles", CFGNAME,
           phase, watchdog_cyc);
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

  task automatic check_bit(input string what, input logic got, input logic exp);
    checks++;
    if (got !== exp)
      note_fail($sformatf("%s: got %b exp %b", what, got, exp));
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

  task automatic wait_idle();
    logic [31:0] st;
    do csr_rd(A_STATUS, st); while ((st & 32'h1) == 32'h0);
  endtask

  // engine select — ONLY while quiescent (callers guarantee wait_idle first)
  task automatic sel_eng(input int unsigned e);
    begin
      if (e >= N_ENG) $fatal(1, "sel_eng: engine %0d out of range", e);
      @(negedge clk);
      eng_sel = ENG_W'(e);
      @(negedge clk);
    end
  endtask

  // ── stream tasks (negedge-drive; kvq suite pattern) ─────────────────────────
  // one CQ-8 value-record store: WRITE_ADDR, then D fp16 beats (tuser=1)
  task automatic store_rec(input int unsigned addr, input int unsigned base,
                           input bit alt);
    begin
      csr_wr(A_WRITE_ADDR, 32'(addr));
      for (int unsigned b = 0; b < D; b++) begin
        @(negedge clk);
        s_tdata  = alt ? stim2[b] : stim[base + b];
        s_tvalid = 1'b1;
        s_tlast  = (b == D - 1);
        s_tuser  = 1'b1;              // value path (CQ-8 K and V, L3 flow)
        while (s_tready !== 1'b1) @(negedge clk);
      end
      @(negedge clk);
      s_tvalid = 1'b0; s_tlast = 1'b0;
      wait_idle();
    end
  endtask

  // fp32 readback of one record vs golden (D data beats + 1 tlast check);
  // ONE negedge advance per beat (the mask-TB double-advance lesson)
  task automatic read_rec(input string tag, input int unsigned addr,
                          input int unsigned base, input bit alt);
    begin
      csr_wr(A_READ_ADDR, 32'(addr));
      for (int unsigned b = 0; b < D; b++) begin
        @(negedge clk);
        while (m_tvalid !== 1'b1) @(negedge clk);
        check32($sformatf("hat %s ch%0d", tag, b), m_tdata,
                alt ? exp2h[b] : exp_h[base + b]);
        if (b == D - 1) begin
          checks++;
          if (m_tlast !== 1'b1)
            note_fail($sformatf("hat %s: no tlast on final beat", tag));
        end
      end
      wait_idle();
    end
  endtask

  // ── main ────────────────────────────────────────────────────────────────────
  int unsigned rd_order [4] = '{2, 0, 3, 1};   // select-churn read order
  logic [31:0] v32;
  initial begin
    awaddr = '0; awvalid = 0; wdata = '0; wvalid = 0; bready = 1'b1;
    araddr = '0; arvalid = 0; rready = 1'b1;
    s_tdata = '0; s_tvalid = 0; s_tlast = 0; s_tuser = 0;
    m_tready = 1'b1; flush_r = 0; eng_sel = '0;
    checks = 0; fails = 0; printed = 0;
    rst_n = 1'b0;
    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    // B1a — per-engine identity: INFO regs from the SELECTED engine
    for (int unsigned e = 0; e < N_ENG; e++) begin
      phase = $sformatf("info e%0d", e);
      sel_eng(e);
      csr_check($sformatf("e%0d INFO_DIM", e),   A_INFO_DIM,   32'(D));
      csr_check($sformatf("e%0d INFO_TIER", e),  A_INFO_TIER,  32'd0);
      csr_check($sformatf("e%0d INFO_DEPTH", e), A_INFO_DEPTH, 32'(DEPTH));
      csr_check($sformatf("e%0d OCC reset", e),  A_OCCUPANCY,  32'd0);
    end

    // B1b — enable each engine, readback CTRL
    for (int unsigned e = 0; e < N_ENG; e++) begin
      phase = $sformatf("enable e%0d", e);
      sel_eng(e);
      csr_wr(A_CTRL, 32'h2);
      csr_check($sformatf("e%0d CTRL", e), A_CTRL, 32'h2);
    end

    // B2a — stores: NREC[e] records per engine at addresses 0..NREC[e]-1
    for (int unsigned e = 0; e < N_ENG; e++) begin
      sel_eng(e);
      for (int unsigned r = 0; r < NREC[e]; r++) begin
        phase = $sformatf("store e%0d r%0d", e, r);
        store_rec(r, (OFF[e] + r) * D, 1'b0);
      end
    end

    // B1c — occupancy distinguishes engines (distinct NREC per engine)
    for (int unsigned e = 0; e < N_ENG; e++) begin
      phase = $sformatf("occ e%0d", e);
      sel_eng(e);
      csr_check($sformatf("e%0d OCC stored", e), A_OCCUPANCY, 32'(NREC[e]));
    end

    // B2b — full readback, engine order churned (2,0,3,1)
    for (int unsigned i = 0; i < N_ENG; i++) begin
      automatic int unsigned e = rd_order[i];
      sel_eng(e);
      for (int unsigned r = 0; r < NREC[e]; r++) begin
        phase = $sformatf("rb e%0d r%0d", e, r);
        read_rec($sformatf("e%0d r%0d", e, r), r, (OFF[e] + r) * D, 1'b0);
      end
    end

    // B3 — same-address rewrite on e2 only; neighbors at that address hold
    phase = "rewrite e2 r1";
    sel_eng(REW_E);
    store_rec(REW_R, 0, 1'b1);
    read_rec("e2 r1 REWRITE", REW_R, 0, 1'b1);
    csr_check("e2 OCC rewrite-stable", A_OCCUPANCY, 32'(NREC[REW_E]));
    for (int unsigned e = 0; e < N_ENG; e++) begin
      if (e == REW_E) continue;
      phase = $sformatf("rb-hold e%0d r%0d", e, REW_R);
      sel_eng(e);
      read_rec($sformatf("e%0d r%0d HOLD", e, REW_R), REW_R,
               (OFF[e] + REW_R) * D, 1'b0);
    end

    // B4 — D-016a RD_ERR isolation + irq OR across engines
    phase = "rderr arm e1";
    check_bit("irq clean pre-error", irq_w, 1'b0);
    sel_eng(1);
    csr_wr(A_IRQ_MASK, 32'h1);                 // unmask RD_ERR on e1
    csr_wr(A_READ_ADDR, 32'(DEPTH - 1));       // never-written address
    phase = "rderr poll e1";
    do csr_rd(A_IRQ_STATUS, v32); while ((v32 & 32'h1) == 32'h0);
    check32("e1 RD_ERR sticky", v32 & 32'h1, 32'h1);
    check_bit("e1 zero-beat drop", m_tvalid, 1'b0);
    csr_rd(A_STATUS, v32);
    check32("e1 STATUS[1] mirror", (v32 >> 1) & 32'h1, 32'h1);
    wait_idle();
    for (int unsigned e = 0; e < N_ENG; e++) begin
      if (e == 1) continue;
      phase = $sformatf("rderr clean e%0d", e);
      sel_eng(e);
      csr_check($sformatf("e%0d IRQ clean", e), A_IRQ_STATUS, 32'h0);
    end
    phase = "rderr irq-or e0";
    sel_eng(0);
    check_bit("irq ORs from unselected e1", irq_w, 1'b1);
    phase = "rderr w1c e1";
    sel_eng(1);
    csr_wr(A_IRQ_STATUS, 32'h1);               // W1C
    csr_check("e1 IRQ cleared", A_IRQ_STATUS, 32'h0);
    check_bit("irq falls after W1C", irq_w, 1'b0);
    csr_wr(A_IRQ_MASK, 32'h0);
    phase = "rderr recover e1";
    read_rec("e1 r0 RECOVER", 0, OFF[1] * D, 1'b0);

    // B5 — m_pending under backpressure (§5 no-retract), on e3
    phase = "mpend e3";
    sel_eng(3);
    csr_wr(A_READ_ADDR, 32'd0);
    @(negedge clk);
    while (m_tvalid !== 1'b1) @(negedge clk);
    m_tready = 1'b0;
    repeat (3) @(negedge clk);
    check_bit("m_pending under stall", m_pending, 1'b1);
    check_bit("m_tvalid held (no retract)", m_tvalid, 1'b1);
    m_tready = 1'b1;
    for (int unsigned b = 0; b < D; b++) begin
      if (b != 0) begin
        @(negedge clk);
        while (m_tvalid !== 1'b1) @(negedge clk);
      end
      check32($sformatf("hat e3 r0 stall ch%0d", b), m_tdata,
              exp_h[(OFF[3] + 0) * D + b]);
      if (b == D - 1) begin
        checks++;
        if (m_tlast !== 1'b1) note_fail("e3 r0 stall: no tlast");
      end
    end
    wait_idle();

    // B6 — D-008 flush on an idle CQ-8 engine: documented no-op
    phase = "flush e0";
    sel_eng(0);
    @(negedge clk); flush_r = 1'b1;
    @(negedge clk); flush_r = 1'b0;
    repeat (4) @(negedge clk);
    csr_check("e0 IRQ after flush", A_IRQ_STATUS, 32'h0);
    csr_check("e0 OCC after flush", A_OCCUPANCY, 32'(NREC[0]));

    phase = "done";
    $display("GQA BANK %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
    if (fails != 0) $fatal(1, "GQA BANK %s: %0d FAILURES", CFGNAME, fails);
    $display("GQA BANK %s: ALL PASS", CFGNAME);
    $finish;
  end

endmodule
`default_nettype wire
