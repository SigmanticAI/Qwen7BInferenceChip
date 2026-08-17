// tb_kvq_top.sv — APEX KVQ parity + stall testbench: the V0 harness
// (verif/v0/kve/tb/tb_kve_top.sv) adapted to rtl/kvq/kvq_engine.sv, proving the
// KVQ rebuild causes NO functional regression (same flows, same golden vectors,
// IDENTICAL check counts as the V0 runs recorded in verif/v0/kve/RESULT.md).
//
// Differences vs the V0 TB — ONLY what the KVQ wrapper changes:
//   * DUT is kvq_engine (flush_req tied 0 here — the parity flows must behave
//     exactly like the V0-patched engine, including the PARTIAL_DEMO wedge;
//     the D-008 flush feature has its own regression, tb_kvq_flush.sv).
//   * SRAM record peeks use the APEX §4/D-026 (A2 KV-REC-DEDUP) layout:
//       key   [tag={ssid[6:0],1'b1}:8b][lanes K_OUTxfp16][codes Dx4b][pad->64b]
//       value [tag=8'h00:8b][scale 16b][payload DxBPV][pad->64b]   (unchanged)
//     Group scales live in the persistent scale bank (dut.g_key.u_bank_store,
//     one DxDW row per committed group, outlier lanes forced 16'h0000). Key
//     records and bank rows are compared against the COMMITTED golden images
//     (key_records.u8.hex / scale_bank.f16.hex, emitted by
//     golden/apex_golden/cq_codec.py pack_key_records — the single source of
//     truth; this TB re-derives no packing).
//   * The fp32-readback and value-record check STRUCTURE is untouched (those
//     subtotals still match V0 exactly); the key-record subtotal moved with
//     the layout (1+2D -> 2+K_OUT+D per record, + D bank-row checks per
//     committed group). The per-config totals are DERIVED (arithmetic in the
//     Makefile next to the CHK_* pins).
//
// Flows (unchanged):
//   STALL_MODE==0 : golden parity (+ optional PARTIAL_DEMO of the g<G wedge,
//                   which still exists when no flush is requested)
//   STALL_MODE==1 : stalling-consumer read bursts (D-007 re-proof on kvq_engine)
//
// Every run has a cycle watchdog that $fatal's.

`timescale 1ns/1ps

module tb_kvq_top;

  // ---------------- configuration (overridden with verilator -G...) ---------
  parameter int     D            = 64;   // VECTOR_DIM
  parameter int     T            = 128;  // tokens in the golden vector set
  parameter int     TIER         = 1;    // 0 CQ-8, 1 CQ-4, 2 CQ-4+
  parameter int     G            = 64;   // DUT KEY_GROUP == golden key grouping
  parameter int     K_OUT        = 0;    // OUTLIER_K
  parameter int     DEPTH        = 256;  // SRAM_DEPTH
  parameter int     KEY_NGRP     = 2;    // FULL key groups reachable via the top
  parameter int     PARTIAL_DEMO = 0;    // demo the partial-group (g<G) wedge
  parameter int     STALL_MODE   = 0;    // V0.2 flow
  parameter int     STALL_NV     = 8;    // value tokens used in the stall flow
  parameter int     STALL_ROUNDS = 3;    // re-read rounds w/ different patterns
  parameter int     EXPECT_BUG   = 0;    // 1 = unpatched upstream (bug expected)
  parameter         MAX_CYCLES   = 20_000_000;
  // untyped string parameters (portable across Verilator -G and iverilog #(...))
  parameter         VECDIR       = ".";  // golden dir: inputs + value golden
  parameter         KEYDIR       = ".";  // key golden dir (differs for G=128 gen)
  parameter         MASKFILE     = ".";  // outlier mask hex (TIER==2)
  parameter         CFGNAME      = "cfg";

  // ---------------- DUT record-layout mirror (kvq_engine.sv, §4/D-026) ------
  // KEY record (A2/D-026): [tag {ssid[6:0],1'b1}][K_OUT x fp16 outlier lanes]
  // [D x int4 codes][pad->64b]; group scales moved to the persistent scale
  // bank (dut.g_key.u_bank_store, one D x 16b row per committed group).
  // SRAMW mirrors cq_codec.sram_row_bits: keyed engines unify the row to
  // pad64(max(KEY_REC_RAW, VAL_REC_RAW)) — at D=64 K_OUT<=2 the VALUE record
  // (280 raw) dominates the shrunken key record.
  localparam int VAL_BPV        = (TIER == 0) ? 8 : 4;
  localparam int PAY_BITS       = D * VAL_BPV;
  localparam int KEY_GROUPED    = (TIER != 0) ? 1 : 0;
  localparam int KEY_CODES_BITS = D * 4;
  localparam int KEY_LANE_BITS  = K_OUT * 16;        // D-026 outlier fp16 lanes
  localparam int VAL_REC_RAW    = 8 + 16 + PAY_BITS;
  localparam int KEY_REC_RAW    = 8 + KEY_LANE_BITS + KEY_CODES_BITS;
  localparam int REC_RAW        = (KEY_GROUPED == 1 && KEY_REC_RAW > VAL_REC_RAW)
                                ? KEY_REC_RAW : VAL_REC_RAW;
  localparam int SRAMW          = 64 * ((REC_RAW + 63) / 64);   // pad to 64b (§4)
  localparam int ROWB           = SRAMW / 8;         // golden record-image bytes/row
  localparam int VAL_SCALE_LO   = 8;                 // value: fp16 scale
  localparam int VAL_PAY_LO     = 24;                // value: packed payload
  localparam int KEY_LANE_LO    = 8;                 // key: outlier fp16 lanes
  localparam int KEY_CODES_LO   = 8 + KEY_LANE_BITS; // key: per-channel codes
  localparam int SCALE_SETS     = 4;                 // engine D-026 default (unoverridden)
  localparam int VPB            = (TIER == 0) ? D : D / 2;      // value payload bytes/token
  localparam int KEY_BASE       = T;                 // keys stored after values

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

  wire irq_w;
  // AXI-Lite response/handshake outputs and evict pins are intentionally not
  // exercised by the parity flows; consume them for -Wall cleanliness.
  wire dut_mask_valid;  // D-027 stage-4 port (sunk; the mask suite checks it)
  wire _tb_unused_ok = &{1'b0, dut_mask_valid, awready, wready, bresp, bvalid, arready,
                         rresp, rvalid, evict_needed, evict_addr};
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
    .axil_rdata(rdata), .axil_rresp(rresp), .axil_rvalid(rvalid), .axil_rready(rready),
    .s_axis_kv_tdata(s_tdata), .s_axis_kv_tvalid(s_tvalid), .s_axis_kv_tready(s_tready),
    .s_axis_kv_tlast(s_tlast), .s_axis_kv_tuser(s_tuser),
    .m_axis_kv_tdata(m_tdata), .m_axis_kv_tvalid(m_tvalid), .m_axis_kv_tready(m_tready),
    .m_axis_kv_tlast(m_tlast),
    .flush_req(1'b0),               // parity: behave exactly like the V0 engine
    .irq(irq_w),
    .evict_needed(evict_needed), .evict_addr(evict_addr),
    .mask_valid(dut_mask_valid)
  );

  // ---------------- D-026 persistent scale-bank mirror -----------------------
  // Clocked sample of dut.g_key.u_bank_store.mem so check_bank can peek the
  // committed rows without a hierarchical ref into a generate scope that
  // TIER-0 (value-only) builds do not elaborate. Sampling lag is irrelevant:
  // rows are only checked after wait_idle, long after the commit.
  logic [D*16-1:0] tb_bank_row [0:SCALE_SETS-1];
  generate
    if (TIER != 0) begin : g_bank_mirror
      always @(posedge clk) begin
        for (int s_ = 0; s_ < SCALE_SETS; s_++)
          tb_bank_row[s_] <= dut.g_key.u_bank_store.mem[s_];
      end
    end else begin : g_bank_tie
      initial for (int s_ = 0; s_ < SCALE_SETS; s_++) tb_bank_row[s_] = '0;
    end
  endgenerate

  // ---------------- watchdog (MANDATORY) -------------------------------------
  initial begin
    #(MAX_CYCLES * 10);   // 10 ns clock period
    $fatal(1, "WATCHDOG: %s exceeded %0d cycles — DUT or TB hung", CFGNAME, MAX_CYCLES);
  end

  // ---------------- golden vectors ------------------------------------------
  localparam int MAXE = 16384;
  logic [15:0] in_k   [0:MAXE-1];
  logic [15:0] in_v   [0:MAXE-1];
  logic [31:0] khat   [0:MAXE-1];
  logic [31:0] vhat   [0:MAXE-1];
  logic [15:0] vscale [0:511];
  logic [7:0]  vpay   [0:MAXE-1];
  logic [15:0] kscale [0:511];      // CQ-8 per-token key scales (check_valrec)
  logic [7:0]  kpay   [0:MAXE-1];   // CQ-8 per-token key payload (check_valrec)
  logic [7:0]  krec   [0:MAXE-1];   // D-026 golden key-record image (T x ROWB bytes)
  logic [15:0] bankg  [0:1023];     // D-026 golden scale-bank rows (ngroups x D)

  // ---------------- AXI-Stream output protocol monitor -----------------------
  // Counts violations of "tdata/tlast stable & tvalid held while tvalid&&!tready"
  // (AXI4-Stream §2.2.1 / ARCHITECTURE.md §5 handshake rule).
  int   viol_stable;
  int   viol_sva    = 0;
  logic pend;
  logic [31:0] pend_d;
  logic pend_l;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pend <= 1'b0; pend_d <= '0; pend_l <= 1'b0; viol_stable <= 0;
    end else begin
      if (pend) begin
        if (!m_tvalid || (m_tdata !== pend_d) || (m_tlast !== pend_l))
          viol_stable <= viol_stable + 1;
      end
      pend   <= (m_tvalid && !m_tready);
      pend_d <= m_tdata;
      pend_l <= m_tlast;
    end
  end

`ifdef USE_SVA
  // Same rule as SVA (compiled in every Verilator build; ARCHITECTURE.md D-012).
  // Action block counts instead of aborting so the EXPECT_BUG run can complete
  // and report the full failure signature.
  logic sva_en;      // async-reset gate: rst_n itself is never edge-sampled
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) sva_en <= 1'b0;
    else        sva_en <= 1'b1;
  end
  property p_hold;
    @(posedge clk)
      (sva_en && m_tvalid && !m_tready) |=> (m_tvalid && $stable(m_tdata) && $stable(m_tlast));
  endproperty
  ap_hold: assert property (p_hold) else viol_sva++;
`endif

  // ---------------- randomized m_axis backpressure (V0.2) --------------------
  parameter int STALL_SEED = 32'h5EED_C0DE;
  logic [31:0] lfsr;
  int          stretch;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_tready <= 1'b1; lfsr <= STALL_SEED; stretch <= 0;
    end else if (STALL_MODE != 0) begin
      if (stretch == 0) begin
        lfsr     <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        m_tready <= lfsr[0];
        stretch  <= 1 + {29'd0, lfsr[3:1]};   // hold ready/!ready 1..8 cycles
      end else begin
        stretch <= stretch - 1;
      end
    end else begin
      m_tready <= 1'b1;
    end
  end

  // ---------------- bookkeeping ----------------------------------------------
  int checks = 0;
  int fails  = 0;
  int printed = 0;
  localparam int MAXPRINT = 12;

  task automatic note_fail(input string msg);
    fails++;
    if (printed < MAXPRINT) begin printed++; $display("  FAIL: %s", msg); end
    else if (printed == MAXPRINT) begin printed++; $display("  (further FAIL prints suppressed)"); end
  endtask

  // ---------------- AXI-Lite tasks (upstream tb_top_stream style) ------------
  localparam [7:0] REG_CTRL = 8'h00, REG_STATUS = 8'h04, REG_OCC = 8'h24,
                   REG_WA   = 8'h28, REG_RA     = 8'h2C;

  task automatic awrite(input [7:0] a, input [31:0] dv);
    @(negedge clk); awaddr = a; wdata = dv; awvalid = 1; wvalid = 1;
    @(negedge clk); awvalid = 0; wvalid = 0;
  endtask

  task automatic aread(input [7:0] a, output [31:0] dv);
    @(negedge clk); araddr = a; arvalid = 1; rready = 1;
    @(negedge clk); arvalid = 0;
    @(negedge clk); dv = rdata; rready = 0;
  endtask

  task automatic wait_idle(input int max_polls);
    int p; logic [31:0] st;
    p = 0; st = 0;
    while (((st & 32'h1) == 32'h0) && p < max_polls) begin aread(REG_STATUS, st); p++; end
    checks++;
    if ((st & 32'h1) == 32'h0) note_fail($sformatf("wait_idle: STATUS.idle never set after %0d polls", max_polls));
  endtask

  // ---------------- stream one token (D fp16 beats, ISA §3 framing) ----------
  task automatic stream_token(input bit isval, input int base, input bit wait_tail);
    int d_, g_;
    d_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      s_tdata  = isval ? in_v[base + d_] : in_k[base + d_];
      s_tvalid = 1; s_tuser = isval; s_tlast = (d_ == D - 1);
      if (s_tready) d_ = d_ + 1;
    end
    @(negedge clk); s_tvalid = 0; s_tlast = 0;
    if (wait_tail) begin
      g_ = 0;
      while (!s_tready && g_ < 500000) begin @(negedge clk); g_++; end
      checks++;
      if (!s_tready) note_fail("stream_token: s_axis_kv_tready never re-asserted");
    end
  endtask

  // ---------------- clean read burst (parity mode, tready held high) ---------
  logic [31:0] outb [0:255];
  task automatic rd_burst_clean(input int addr);
    int d_, g_;
    awrite(REG_RA, addr);
    d_ = 0; g_ = 0;
    while (d_ < D) begin
      @(negedge clk);
      if (m_tvalid) begin
        outb[d_] = m_tdata;
        checks++;
        if (m_tlast !== ((d_ == D-1) ? 1'b1 : 1'b0))
          note_fail($sformatf("rd addr=%0d beat=%0d: tlast=%b wrong", addr, d_, m_tlast));
        d_ = d_ + 1;
      end
      g_++;
      if (g_ > 200000) begin note_fail($sformatf("rd addr=%0d: burst hang", addr)); return; end
    end
    @(negedge clk);
    checks++;
    if (m_tvalid) note_fail($sformatf("rd addr=%0d: extra beats after tlast", addr));
  endtask

  // ---------------- SRAM record peek helpers ---------------------------------
  logic [SRAMW-1:0] rec;
  task automatic peek(input int addr);
    rec = dut.u_sram.mem[addr % DEPTH];
  endtask

  // value-format record (values all tiers; CQ-8 keys). golden arrays selected by iskey.
  task automatic check_valrec(input int addr, input int tok, input bit iskey);
    logic [7:0] gb; int b_;
    peek(addr);
    for (b_ = 0; b_ < VPB; b_++) begin
      gb = iskey ? kpay[tok*VPB + b_] : vpay[tok*VPB + b_];
      checks++;
      if (rec[VAL_PAY_LO + b_*8 +: 8] !== gb)
        note_fail($sformatf("%s rec a=%0d tok=%0d paybyte %0d: got %02h exp %02h",
                  iskey?"K8":"V", addr, tok, b_, rec[VAL_PAY_LO + b_*8 +: 8], gb));
    end
    checks++;
    if (rec[VAL_SCALE_LO +: 16] !== (iskey ? kscale[tok] : vscale[tok]))
      note_fail($sformatf("%s rec a=%0d tok=%0d scale: got %04h exp %04h",
                iskey?"K8":"V", addr, tok, rec[VAL_SCALE_LO +: 16], iskey?kscale[tok]:vscale[tok]));
    if (KEY_GROUPED == 1) begin
      checks++;
      if (rec[7:0] !== 8'h00) note_fail($sformatf("V rec a=%0d: tag byte != 00", addr));
    end
  endtask

  // unified per-channel key record (CQ-4 / CQ-4+), §4/D-026 layout, compared
  // against the COMMITTED golden record image (key_records.u8.hex, packed by
  // cq_codec.pack_key_records — nothing re-derived here): tag bit0 (key) +
  // tag[7:1] vs the TRACKED per-engine commit-sequence ssid + K_OUT outlier
  // lanes + D code nibbles (keep -> quant code, outlier -> sentinel 4'd1).
  // seq = group commit sequence since hard reset (groups commit in stream
  // order here, so seq == gi); tok = global token index (golden image row).
  task automatic check_keyrec(input int addr, input int seq, input int tok);
    int c_, j_, b_;
    logic [SRAMW-1:0] gold;
    logic [6:0] ssid_e;
    peek(addr);
    gold = '0;
    for (b_ = 0; b_ < ROWB; b_++) gold[b_*8 +: 8] = krec[tok*ROWB + b_];
    ssid_e = 7'(seq % SCALE_SETS);
    if (gold[7:0] !== {ssid_e, 1'b1})
      $fatal(1, "golden key_records row %0d tag %02h != {ssid=%0d,1'b1} — committed vectors/TB out of sync", tok, gold[7:0], ssid_e);
    checks++;
    if (rec[0] !== 1'b1)
      note_fail($sformatf("K rec a=%0d: tag bit0 %b != 1 (not a key record)", addr, rec[0]));
    checks++;
    if (rec[7:1] !== ssid_e)
      note_fail($sformatf("K rec a=%0d: ssid %0d exp %0d (commit seq %0d)", addr, rec[7:1], ssid_e, seq));
    for (j_ = 0; j_ < K_OUT; j_++) begin
      checks++;
      if (rec[KEY_LANE_LO + j_*16 +: 16] !== gold[KEY_LANE_LO + j_*16 +: 16])
        note_fail($sformatf("K rec a=%0d lane%0d: %04h exp %04h", addr, j_,
                  rec[KEY_LANE_LO + j_*16 +: 16], gold[KEY_LANE_LO + j_*16 +: 16]));
    end
    for (c_ = 0; c_ < D; c_++) begin
      checks++;
      if (rec[KEY_CODES_LO + c_*4 +: 4] !== gold[KEY_CODES_LO + c_*4 +: 4])
        note_fail($sformatf("K rec a=%0d ch%0d: code %0h exp %0h", addr, c_,
                  rec[KEY_CODES_LO + c_*4 +: 4], gold[KEY_CODES_LO + c_*4 +: 4]));
    end
  endtask

  // D-026 bank peek: after the group with commit sequence `seq` commits, the
  // persistent scale set (seq % SCALE_SETS) must equal the golden
  // scale_bank.f16.hex row seq — keep channels hold that group's fp16
  // per-channel scales, outlier lanes are forced 16'h0000 by the writer.
  task automatic check_bank(input int seq);
    int c_, set_;
    set_ = seq % SCALE_SETS;
    for (c_ = 0; c_ < D; c_++) begin
      checks++;
      if (tb_bank_row[set_][c_*16 +: 16] !== bankg[seq*D + c_])
        note_fail($sformatf("bank set%0d ch%0d: %04h exp %04h", set_, c_,
                  tb_bank_row[set_][c_*16 +: 16], bankg[seq*D + c_]));
    end
  endtask

  // ---------------- fp32 readback compare -------------------------------------
  task automatic check_hat(input int addr, input int tok, input bit iskey);
    int d_;
    logic [31:0] e;
    rd_burst_clean(addr);
    for (d_ = 0; d_ < D; d_++) begin
      e = iskey ? khat[tok*D + d_] : vhat[tok*D + d_];
      checks++;
      if (outb[d_] !== e)
        note_fail($sformatf("%s_hat a=%0d tok=%0d d=%0d: got %08h exp %08h",
                  iskey?"K":"V", addr, tok, d_, outb[d_], e));
    end
  endtask

  // ---------------- V0.2 backpressured read burst -----------------------------
  int st_bursts = 0, st_acc = 0, st_exp = 0, st_drop = 0, st_mism = 0, st_nolast = 0;
  task automatic rd_burst_stall(input int addr, input int tok, input bit iskey);
    int g_, acc; bit gotlast; logic [31:0] e;
    awrite(REG_RA, addr);
    g_ = 0;
    @(negedge clk);
    while (!m_tvalid && g_ < 200000) begin @(negedge clk); g_++; end
    if (!m_tvalid) begin fails++; $display("  FAIL: stall burst a=%0d never started", addr); return; end
    acc = 0; gotlast = 0;
    while (m_tvalid) begin
      if (m_tready) begin
        if (acc < D) begin
          e = iskey ? khat[tok*D + acc] : vhat[tok*D + acc];
          if (m_tdata !== e) st_mism++;
        end
        if (m_tlast) gotlast = 1;
        acc++;
      end
      @(negedge clk);
      g_++;
      if (g_ > 400000) begin fails++; $display("  FAIL: stall burst a=%0d hang", addr); break; end
    end
    st_bursts++;
    st_exp  += D;
    st_acc  += acc;
    if (acc < D) st_drop += (D - acc);
    if (!gotlast) st_nolast++;
    if (st_bursts <= 3)
      $display("  burst[%0d] addr=%0d: beats_accepted=%0d/%0d tlast_seen=%0d", st_bursts-1, addr, acc, D, gotlast);
  endtask

  // ---------------- flows ------------------------------------------------------
  logic [31:0] rd32, occ_b, occ_a;
  int t, gi, r, tail, npulse, w;

  initial begin
    rst_n = 0;
    awaddr = 0; awvalid = 0; wdata = 0; wvalid = 0; bready = 1;
    araddr = 0; arvalid = 0; rready = 1;
    s_tdata = 0; s_tvalid = 0; s_tlast = 0; s_tuser = 0;

    $display("==== tb_kvq_top config=%s D=%0d T=%0d TIER=%0d G=%0d K=%0d NGRP=%0d partial=%0d stall=%0d expect_bug=%0d",
             CFGNAME, D, T, TIER, G, K_OUT, KEY_NGRP, PARTIAL_DEMO, STALL_MODE, EXPECT_BUG);
    $display("     VECDIR=%s", VECDIR);
    $display("     KEYDIR=%s", KEYDIR);

    // golden vectors ($readmemh; hex images are the frozen reference output)
    $readmemh({VECDIR, "/input_k.f16.hex"},        in_k);
    $readmemh({VECDIR, "/input_v.f16.hex"},        in_v);
    $readmemh({VECDIR, "/expected_v_hat.f32.hex"}, vhat);
    $readmemh({VECDIR, "/val_scales.f16.hex"},     vscale);
    $readmemh({VECDIR, "/val_payload.u8.hex"},     vpay);
    $readmemh({KEYDIR, "/expected_k_hat.f32.hex"}, khat);
    $readmemh({KEYDIR, "/key_scales.f16.hex"},     kscale);
    $readmemh({KEYDIR, "/key_payload.u8.hex"},     kpay);
    if (TIER != 0) begin
      // D-026 golden images: packed key records + persistent scale-bank rows
      $readmemh({KEYDIR, "/key_records.u8.hex"}, krec);
      $readmemh({KEYDIR, "/scale_bank.f16.hex"}, bankg);
    end

    repeat (8) @(posedge clk);
    rst_n = 1;
    repeat (4) @(posedge clk);
    awrite(REG_CTRL, 32'h2);   // enable

    if (STALL_MODE == 0) begin
      // ================= V0.1 PARITY FLOW =================================
      // --- values: per-token compress (ISA §3: WRITE_ADDR then D beats) ---
      $display("[V] streaming %0d value tokens", T);
      for (t = 0; t < T; t++) begin
        awrite(REG_WA, t);
        stream_token(1'b1, t*D, 1'b1);
      end
      wait_idle(1000);
      $display("[V] compress side-effects: checking %0d SRAM value records", T);
      for (t = 0; t < T; t++) check_valrec(t, t, 1'b0);
      $display("[V] decompress: reading back %0d tokens (fp32)", T);
      for (t = 0; t < T; t++) check_hat(t, t, 1'b0);

      // --- keys ------------------------------------------------------------
      if (TIER == 0) begin
        $display("[K] CQ-8 per-token keys: streaming %0d key tokens", T);
        for (t = 0; t < T; t++) begin
          awrite(REG_WA, KEY_BASE + t);
          stream_token(1'b0, t*D, 1'b1);
        end
        wait_idle(1000);
        $display("[K] compress side-effects: checking %0d SRAM key records", T);
        for (t = 0; t < T; t++) check_valrec(KEY_BASE + t, t, 1'b1);
        $display("[K] decompress: reading back %0d key tokens (fp32)", T);
        for (t = 0; t < T; t++) check_hat(KEY_BASE + t, t, 1'b1);
      end else begin
        for (gi = 0; gi < KEY_NGRP; gi++) begin
          $display("[K] streaming key group %0d (G=%0d tokens) at base %0d", gi, G, KEY_BASE + gi*G);
          awrite(REG_WA, KEY_BASE + gi*G);   // group base (ISA §3)
          for (t = 0; t < G; t++) stream_token(1'b0, (gi*G + t)*D, 1'b0);
          wait_idle(400000);                 // group flush: serialized D-ch quant walk
          $display("[K] group %0d flushed (commit seq %0d -> ssid %0d); checking %0d D-026 key records + bank row + readback",
                   gi, gi, gi % SCALE_SETS, G);
          for (t = 0; t < G; t++) check_keyrec(KEY_BASE + gi*G + t, gi, gi*G + t);
          check_bank(gi);
          for (t = 0; t < G; t++) check_hat(KEY_BASE + gi*G + t, gi*G + t, 1'b1);
        end
      end

      // --- partial-group limitation demo (upstream: flush is a follow-up) ---
      if (PARTIAL_DEMO == 1) begin
        tail = T - KEY_NGRP*G;
        $display("[P] PARTIAL-GROUP DEMO: streaming %0d-token tail (g<G=%0d) — no flush path in the top", tail, G);
        aread(REG_OCC, occ_b);
        awrite(REG_WA, KEY_BASE + KEY_NGRP*G);
        for (t = 0; t < tail; t++) stream_token(1'b0, (KEY_NGRP*G + t)*D, 1'b0);
        repeat (20000) @(posedge clk);
        aread(REG_STATUS, rd32);
        checks++;
        if ((rd32 & 32'h1) !== 32'h0) note_fail("partial demo: engine went idle (a flush path exists?!)");
        else $display("[P] confirmed: STATUS.idle=0 — engine wedged in ST_KACCEPT waiting for %0d more tokens", G - tail);
        aread(REG_OCC, occ_a);
        checks++;
        if (occ_a !== occ_b) note_fail($sformatf("partial demo: occupancy moved %0d->%0d (tail records emitted?!)", occ_b, occ_a));
        else $display("[P] confirmed: OCCUPANCY unchanged (%0d) — the %0d tail tokens are stranded in the residual buffer", occ_a, tail);
        // read requests are silently dropped while wedged (read_req only honored in ST_IDLE)
        awrite(REG_RA, 0);
        npulse = 0;
        for (w = 0; w < 5000; w++) begin @(negedge clk); if (m_tvalid) npulse++; end
        checks++;
        if (npulse != 0) note_fail("partial demo: a read burst started while wedged");
        else $display("[P] confirmed: READ_ADDR write while wedged is silently dropped (0 output beats in 5000 cycles)");
        // recovery: soft reset (CTRL.soft_reset|enable), then prove a value round-trip
        awrite(REG_CTRL, 32'h3);
        repeat (4) @(posedge clk);
        aread(REG_STATUS, rd32);
        checks++;
        if ((rd32 & 32'h1) !== 32'h1) note_fail("partial demo: soft reset did not return engine to idle");
        else $display("[P] soft reset recovers the engine (STATUS.idle=1); tail data is LOST");
        awrite(REG_WA, 0);
        stream_token(1'b1, 0, 1'b1);
        wait_idle(1000);
        check_hat(0, 0, 1'b0);
        $display("[P] post-recovery value round-trip OK — limitation demonstrated, engine usable again");
      end

      // --- protocol monitor must be clean in parity mode --------------------
      checks++;
      if (viol_stable != 0 || viol_sva != 0)
        note_fail($sformatf("AXI-Stream stability violations in parity mode: proc=%0d sva=%0d", viol_stable, viol_sva));

      $display("============================================================");
      $display("CONFIG %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
      if (fails == 0) begin
        $display("KVQ PARITY [%s]: PASS (bit-exact records + fp32 readback, irq=%b)", CFGNAME, irq_w);
        $finish;
      end else begin
        $display("KVQ PARITY [%s]: FAIL", CFGNAME);
        $fatal(1, "parity mismatches");
      end

    end else begin
      // ================= V0.2 STALLING-CONSUMER FLOW ======================
      // write STALL_NV value tokens + (grouped tiers) one key group, then read
      // everything back STALL_ROUNDS times under random multi-cycle tready stalls.
      $display("[S] writing %0d value tokens", STALL_NV);
      for (t = 0; t < STALL_NV; t++) begin
        awrite(REG_WA, t);
        stream_token(1'b1, t*D, 1'b1);
      end
      if (TIER != 0) begin
        $display("[S] writing one key group (G=%0d) at base %0d", G, STALL_NV);
        awrite(REG_WA, STALL_NV);
        for (t = 0; t < G; t++) stream_token(1'b0, t*D, 1'b0);
      end
      wait_idle(400000);

      $display("[S] read bursts under random tready stalls (1..8-cycle stretches, ~50%% duty)");
      for (r = 0; r < STALL_ROUNDS; r++) begin
        for (t = 0; t < STALL_NV; t++) rd_burst_stall(t, t, 1'b0);
        if (TIER != 0) for (t = 0; t < G; t++) rd_burst_stall(STALL_NV + t, t, 1'b1);
      end

      $display("============================================================");
      $display("STALL: bursts=%0d beats_expected=%0d beats_accepted=%0d beats_dropped=%0d",
               st_bursts, st_exp, st_acc, st_drop);
      $display("STALL: data_mismatches_on_accepted=%0d bursts_missing_tlast=%0d", st_mism, st_nolast);
      $display("STALL: stability_violations proc=%0d sva=%0d", viol_stable, viol_sva);
      if (EXPECT_BUG == 1) begin
        if (st_drop > 0 || st_mism > 0 || viol_stable > 0 || st_nolast > 0) begin
          $display("KVQ STALL [%s]: TREADY BUG REPRODUCED — ST_OUTPUT ignores m_axis_kv_tready", CFGNAME);
          $display("  signature: DUT streams %0d beats in %0d consecutive cycles regardless of tready;", D, D);
          $display("  beats coinciding with tready=0 are overwritten (dropped), the accepted");
          $display("  subsequence is channel-misaligned (data corruption), and tlast is lost");
          $display("  whenever tready=0 during the final beat's only cycle on the bus.");
          if (fails == 0) $finish;
          else $fatal(1, "stall harness errors");
        end else begin
          $fatal(1, "KVQ STALL [%s]: bug NOT reproduced — investigate", CFGNAME);
        end
      end else begin
        checks++;
        if (st_drop != 0 || st_mism != 0 || viol_stable != 0 || viol_sva != 0 || st_nolast != 0 || st_acc != st_exp)
          note_fail("patched RTL still drops/corrupts under backpressure");
        if (fails == 0) begin
          $display("KVQ STALL [%s]: PASS — all beats delivered bit-exact under backpressure (irq=%b)", CFGNAME, irq_w);
          $finish;
        end else begin
          $fatal(1, "KVQ STALL [%s]: FAIL on kvq_engine", CFGNAME);
        end
      end
    end
  end

endmodule
