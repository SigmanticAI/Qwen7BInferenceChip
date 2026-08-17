// tb_csr_attack.sv — INDEPENDENT adversarial audit TB for rtl/csr/csr_regs.sv.
// Audit collateral (verif/audit_csr): binds the deliverable's csr_sb_sva
// unchanged; everything else is an independent re-implementation of the
// csr_regs.sv header contract (golden read mirror + register model), NOT of
// the RTL. Attacks beyond the shipped suite:
//
//   CA-1  FULL address sweep: every byte address 0x00..0xFF read AND written
//         (all four alignments), checked against the golden mirror
//         (reserved/unaligned -> 0xDEAD_BEEF, writes side-effect-free).
//   CA-2  W1C / sticky corners: set-vs-W1C same-cycle race, W1C with
//         wdata[1]=0, W1C on an UNALIGNED STATUS address (must NOT clear),
//         W1C bundled with garbage in other bits.
//   CA-3  soft_reset write must NOT disturb CSR contents (sticky, tier,
//         threshold, imp_addr, perf) — only the pulse + the CTRL bits.
//   CA-4  PERF accuracy against a cycle-counting monitor while every busy
//         lane toggles with randomized burst lengths EVERY cycle (their
//         suite holds busy constant between B records), including clears,
//         clear+enable same write, reads mid-count, and PERF_W=8 saturation.
//   CA-5  fully-pipelined bus storm: back-to-back read/write/read+write
//         every cycle at random addresses with random derr pulses — every
//         read checked against the mirror; write->read of the same register
//         on consecutive cycles must see the new value; read+write same
//         cycle must see the PRE-write value.
//
// TB discipline: negedge stimulus; checker samples DUT registers at the
// NEXT posedge (pre-NBA values) against the model; $fatal watchdog.

`include "csr_sb_sva.svh"

module tb_csr_attack #(
    parameter int unsigned PERF_W = 32
);

  import apex_pkg::*;

  localparam int unsigned N_BLOCKS = 8;
  localparam int unsigned IMP_AW   = 7;

  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    repeat (10_000_000) @(posedge clk);
    $fatal(1, "WATCHDOG: simulation did not finish (cyc=%0d)", cyc);
  end

  // ── DUT ────────────────────────────────────────────────────────────────────
  logic [7:0]          bus_addr;
  logic [31:0]         bus_wdata;
  logic                bus_wr, bus_rd;
  logic [31:0]         rdata;
  logic                ready;
  logic                enable, soft_reset;
  logic [N_BLOCKS-1:0] busy_lvl;
  logic                derr;
  logic                desc_error_sticky;
  kvq_tier_e           tier_sel;
  logic                tip_override;
  logic [4:0]          threshold;
  logic                flush;
  logic [IMP_AW-1:0]   imp_rd_addr;
  logic [15:0]         imp_rd_data_i;
  logic [1:0]          imp_rd_tier_i;

  csr_regs #(
    .N_BLOCKS (N_BLOCKS),
    .IMP_AW   (IMP_AW),
    .PERF_W   (PERF_W)
  ) dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .addr              (bus_addr),
    .wdata             (bus_wdata),
    .write             (bus_wr),
    .read              (bus_rd),
    .rdata             (rdata),
    .ready             (ready),
    .enable            (enable),
    .soft_reset        (soft_reset),
    .block_busy_i      (busy_lvl),
    .desc_error_i      (derr),
    .desc_error_sticky (desc_error_sticky),
    .tier_sel          (tier_sel),
    .tip_override      (tip_override),
    .threshold         (threshold),
    .flush             (flush),
    .imp_rd_addr       (imp_rd_addr),
    .imp_rd_data_i     (imp_rd_data_i),
    .imp_rd_tier_i     (imp_rd_tier_i)
  );

  // fake TIP window (same convention as the shipped TB)
  function automatic logic [15:0] imp_data(input logic [IMP_AW-1:0] a);
    return 16'((32'(a) * 32'h9E37) ^ 32'h1234);
  endfunction
  assign imp_rd_data_i = imp_data(imp_rd_addr);
  assign imp_rd_tier_i = 2'(32'(imp_rd_addr) % 3);

  // the deliverable's own CSR SVA, reused and armed
  bind csr_regs csr_sb_sva u_sva (
    .clk               (clk),
    .rst_n             (rst_n),
    .read              (read),
    .write             (write),
    .ready             (ready),
    .desc_error_i      (desc_error_i),
    .desc_error_sticky (desc_error_sticky),
    .w1c_fire          (w1c_fire),
    .soft_reset        (soft_reset),
    .flush             (flush),
    .wr_srst           (wr_srst),
    .wr_flush          (wr_flush)
  );

  // ── golden model (independent, from the csr_regs.sv header ONLY) ──────────
  localparam logic [PERF_W-1:0] PMAX = {PERF_W{1'b1}};

  logic              m_enable, m_sticky, m_tip_ov, m_perf_en;
  logic [1:0]        m_tier;
  logic [4:0]        m_thresh;
  logic [IMP_AW-1:0] m_imp;
  logic [PERF_W-1:0] m_cyc;
  logic [PERF_W-1:0] m_cnt [N_BLOCKS];

  function automatic void model_reset();
    m_enable = 1'b0;
    m_sticky = 1'b0;
    m_tier   = 2'(KVQ_CQ4);
    m_tip_ov = 1'b0;
    m_thresh = 5'd1;
    m_imp    = '0;
    m_perf_en = 1'b0;
    m_cyc    = '0;
    for (int i = 0; i < int'(N_BLOCKS); i++) m_cnt[i] = '0;
  endfunction

  function automatic logic [31:0] golden_rmux(input logic [7:0] a);
    logic [5:0] w;
    if (a[1:0] != 2'b00) return 32'hDEAD_BEEF;
    w = a[7:2];
    case (w)
      6'h00: return {30'b0, 1'b0, m_enable};
      6'h01: return {16'b0, 8'(busy_lvl), 6'b0, m_sticky, ~|busy_lvl};
      6'h02: return 32'(MXE_N);
      6'h03: return 32'd64;                       // CFG_D
      6'h04: return 32'd128;                      // CFG_G
      6'h05: return 32'h0000_0007;
      6'h06: return APEX_VERSION;
      6'h08: return {23'b0, m_tip_ov, 6'b0, m_tier};
      6'h09: return {27'b0, m_thresh};
      6'h0A: return 32'h0;
      6'h0B: return {14'b0, 2'(32'(m_imp) % 3), imp_data(m_imp)};
      6'h0C: return {30'b0, 1'b0, m_perf_en};
      6'h0D: return 32'(m_cyc);
      default: begin
        for (int i = 0; i < int'(N_BLOCKS); i++)
          if (w == 6'h0E + 6'(i)) return 32'(m_cnt[i]);
        return 32'hDEAD_BEEF;
      end
    endcase
  endfunction

  // ── response checker + model stepper (single posedge process) ─────────────
  int errors, checked;
  logic        req_q, rd_q;
  logic [31:0] exp_q;
  logic [7:0]  addr_q;
  bit          model_on;

  always @(posedge clk) begin
    if (rst_n && model_on) begin
      logic wr_al;
      logic [5:0] w;

      // 1) previous request's response (rdata/ready registered last edge,
      //    NBA: sampled values here are still the pre-this-edge ones)
      if (req_q) begin
        if (ready !== 1'b1) begin
          errors++;
          $display("FAIL cyc=%0d: no ready for request @%02x", cyc, addr_q);
        end
        if (rd_q && rdata !== exp_q) begin
          errors++;
          $display("FAIL cyc=%0d: read @%02x got %08x exp %08x",
                   cyc, addr_q, rdata, exp_q);
        end
        if (!rd_q && rdata !== 32'h0) begin
          errors++;
          $display("FAIL cyc=%0d: write-only rdata not 0 @%02x", cyc, addr_q);
        end
        checked++;
      end else if (ready !== 1'b0) begin
        errors++;
        $display("FAIL cyc=%0d: spurious ready", cyc);
      end

      // 1b) output ports (values registered at the PREVIOUS edge) vs model
      if (enable !== m_enable || desc_error_sticky !== m_sticky
          || 2'(tier_sel) !== m_tier || tip_override !== m_tip_ov
          || threshold !== m_thresh || imp_rd_addr !== m_imp) begin
        errors++;
        $display("FAIL cyc=%0d: output ports diverge from model (en %0b/%0b stk %0b/%0b tier %0d/%0d ov %0b/%0b th %0d/%0d imp %0d/%0d)",
                 cyc, enable, m_enable, desc_error_sticky, m_sticky,
                 2'(tier_sel), m_tier, tip_override, m_tip_ov,
                 threshold, m_thresh, imp_rd_addr, m_imp);
      end

      // 2) expected read data for THIS cycle's request (pre-write state)
      req_q  <= (bus_rd || bus_wr);
      rd_q   <= bus_rd;
      addr_q <= bus_addr;
      exp_q  <= bus_rd ? golden_rmux(bus_addr) : 32'h0;

      // 3) PERF counting with the OLD perf_en; clear wins
      wr_al = bus_wr && (bus_addr[1:0] == 2'b00);
      w     = bus_addr[7:2];
      if (wr_al && w == 6'h0C && bus_wdata[1]) begin
        m_cyc = '0;
        for (int i = 0; i < int'(N_BLOCKS); i++) m_cnt[i] = '0;
      end else if (m_perf_en) begin
        if (m_cyc != PMAX) m_cyc = m_cyc + 1'b1;
        for (int i = 0; i < int'(N_BLOCKS); i++)
          if (busy_lvl[i] && m_cnt[i] != PMAX) m_cnt[i] = m_cnt[i] + 1'b1;
      end

      // 4) sticky: set wins over same-cycle W1C
      if (derr)                                     m_sticky = 1'b1;
      else if (wr_al && w == 6'h01 && bus_wdata[1]) m_sticky = 1'b0;

      // 5) register writes
      if (wr_al) begin
        case (w)
          6'h00: m_enable = bus_wdata[0];
          6'h08: begin
            m_tier   = (bus_wdata[1:0] == 2'b11) ? 2'(KVQ_CQ4) : bus_wdata[1:0];
            m_tip_ov = bus_wdata[8];
          end
          6'h09: m_thresh  = bus_wdata[4:0];
          6'h0B: m_imp     = bus_wdata[IMP_AW-1:0];
          6'h0C: m_perf_en = bus_wdata[0];
          default: ;
        endcase
      end
    end else begin
      req_q  <= 1'b0;
      rd_q   <= 1'b0;
      exp_q  <= '0;
      addr_q <= '0;
    end
  end

  // ── busy-lane burst driver (per-lane randomized burst lengths) ────────────
  int unsigned rng_state;
  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction

  bit busy_random;
  int burst_cnt [N_BLOCKS];
  always @(negedge clk) begin
    if (busy_random) begin
      for (int i = 0; i < int'(N_BLOCKS); i++) begin
        if (burst_cnt[i] <= 0) begin
          busy_lvl[i] <= ~busy_lvl[i];
          burst_cnt[i] <= 1 + int'(rnd() % ((i % 3 == 0) ? 64 : 7));
        end else begin
          burst_cnt[i] <= burst_cnt[i] - 1;
        end
      end
    end
  end

  // ── bus driver primitives (negedge) ────────────────────────────────────────
  task automatic op(input bit r, input bit w, input logic [7:0] a,
                    input logic [31:0] d, input bit de = 1'b0);
    @(negedge clk);
    bus_rd    = r;
    bus_wr    = w;
    bus_addr  = a;
    bus_wdata = d;
    derr      = de;
  endtask

  task automatic idle(input int n = 1);
    @(negedge clk);
    bus_rd  = 1'b0;
    bus_wr  = 1'b0;
    derr    = 1'b0;
    repeat (n - 1) @(negedge clk);
  endtask

  // ── main ───────────────────────────────────────────────────────────────────
  initial begin : main
    int unsigned seed = 32'hC5AA_7ACC;
    logic [7:0] a;

    rst_n     = 1'b0;
    bus_rd    = 1'b0;
    bus_wr    = 1'b0;
    bus_addr  = '0;
    bus_wdata = '0;
    busy_lvl  = '0;
    derr      = 1'b0;
    errors    = 0;
    checked   = 0;
    model_on  = 1'b0;
    busy_random = 1'b0;
    for (int i = 0; i < int'(N_BLOCKS); i++) burst_cnt[i] = 0;

    void'($value$plusargs("seed=%d", seed));
    rng_state = seed;
    $display("[TB] csr attack: PERF_W=%0d seed=%0d", PERF_W, seed);

    model_reset();
    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    @(negedge clk);
    model_on = 1'b1;

    // ── CA-1: full byte-address sweep, reads then writes ────────────────────
    $display("[CA-1] full 0x00..0xFF read sweep (all alignments)");
    for (int i = 0; i < 256; i++) op(1, 0, 8'(i), '0);
    idle(2);

    $display("[CA-1] full 0x00..0xFF write sweep (random data) + re-read all");
    for (int i = 0; i < 256; i++) op(0, 1, 8'(i), rnd());
    idle(2);
    for (int i = 0; i < 256; i++) op(1, 0, 8'(i), '0);
    idle(2);

    // ── CA-2: W1C / sticky corners ──────────────────────────────────────────
    $display("[CA-2] W1C corners");
    op(0, 0, 8'h00, '0, 1'b1);            // derr pulse -> sticky set
    idle(2);
    op(1, 0, 8'h04, '0);                  // read STATUS: sticky=1 expected
    op(0, 1, 8'h04, 32'hFFFF_FFFD);       // write with bit1=0: must NOT clear
    op(1, 0, 8'h04, '0);
    op(0, 1, 8'h05, 32'h0000_0002);       // UNALIGNED W1C: must NOT clear
    op(1, 0, 8'h04, '0);
    op(0, 1, 8'h04, 32'hFFFF_FFFF, 1'b1); // W1C racing a same-cycle SET: set wins
    op(1, 0, 8'h04, '0);
    op(0, 1, 8'h04, 32'h0000_0002);       // uncontested W1C: clears
    op(1, 0, 8'h04, '0);
    // read+write same cycle: read must return PRE-write (sticky set again)
    op(0, 0, 8'h00, '0, 1'b1);
    idle(2);
    op(1, 1, 8'h04, 32'h0000_0002);       // simultaneous: pre-write shows sticky=1
    op(1, 0, 8'h04, '0);                  // now cleared
    idle(2);

    // ── CA-3: soft_reset must not disturb CSR contents ──────────────────────
    $display("[CA-3] soft_reset non-destructiveness");
    op(0, 1, 8'h20, 32'h0000_0102);       // tier=CQ4P, override=1
    op(0, 1, 8'h24, 32'h0000_001F);       // threshold=31
    op(0, 1, 8'h2C, 32'h0000_007B);       // imp addr
    op(0, 0, 8'h00, '0, 1'b1);            // sticky set
    idle(2);
    op(0, 1, 8'h00, 32'h0000_0003);       // enable=1 + soft_reset pulse
    idle(2);
    op(1, 0, 8'h20, '0);                  // all preserved (model agrees)
    op(1, 0, 8'h24, '0);
    op(1, 0, 8'h2C, '0);
    op(1, 0, 8'h04, '0);
    op(1, 0, 8'h00, '0);
    idle(2);

    // ── CA-4: PERF accuracy under per-cycle randomized busy bursts ──────────
    $display("[CA-4] PERF vs cycle-counting monitor (randomized busy bursts)");
    busy_random = 1'b1;
    op(0, 1, 8'h30, 32'h0000_0003);       // clear + enable same write
    idle(1);
    for (int k = 0; k < 60; k++) begin
      idle(1 + int'(rnd() % 50));
      op(1, 0, 8'h34, '0);                // PERF_CYCLES mid-count
      for (int i = 0; i < int'(N_BLOCKS); i++)
        op(1, 0, 8'(8'h38 + 8'(4 * i)), '0);   // all PERF_BUSY_i b2b pipelined
      if ((rnd() % 5) == 0) op(0, 1, 8'h30, {30'b0, 1'b1, rnd() % 2 == 0});
      if ((rnd() % 7) == 0) op(0, 1, 8'h30, 32'h0);   // disable
      if ((rnd() % 7) == 0) op(0, 1, 8'h30, 32'h1);   // re-enable
    end
    idle(2);

    // ── CA-5: fully-pipelined random storm ──────────────────────────────────
    $display("[CA-5] pipelined random storm (b2b, derr, random addr/data)");
    for (int k = 0; k < 30000; k++) begin
      int unsigned r;
      r = rnd();
      a = 8'(r);
      if ((r >> 8 & 3) != 0) a = {2'b0, 4'(r >> 10), 2'b00};  // bias to mapped
      case (r >> 29)
        0, 1, 2: op(1, 0, a, '0, (r >> 27 & 1) == 1);
        3, 4:    op(0, 1, a, rnd(), (r >> 27 & 1) == 1);
        5:       op(1, 1, a, rnd(), (r >> 27 & 1) == 1);
        6:       begin op(0, 1, a, rnd()); op(1, 0, a, '0); end // write->read
        default: idle(1 + int'(r % 3));
      endcase
    end
    idle(4);
    busy_random = 1'b0;

    // ── summary ─────────────────────────────────────────────────────────────
    if (errors == 0) begin
      $display("TB PASS: csr attack suite clean — %0d checked responses (PERF_W=%0d seed=%0d)",
               checked, PERF_W, seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s), %0d checked (PERF_W=%0d)",
             errors, checked, PERF_W);
    end
  end

endmodule
