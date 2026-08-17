// sh_ddr_beh.sv — IB-FUEL s1: behavioral sparse DDR model, sim-only.
//
// Drop-in replacement for the kit's EMPTY sh_ddr.stub.sv in the Verilator
// f2sim build (the synthesis sh_ddr.sv is encrypted; the stub is the open
// port surface). Port list and parameter defaults are copied VERBATIM from
// tools/aws-fpga hdk/common/shell_stable/design/sh_ddr/sh_ddr.stub.sv
// (v2.3.3). The kit checkout is NOT edited — verif/f2sim/Makefile points
// SH_DDR here instead (docs/design/IB_FUEL.md §1 s1).
//
//   DDR_PRESENT == 0 : INERT — every output driven '0, exactly the 2-state
//                      value Verilator gives the stub's undriven outputs,
//                      so today's cl_apex (unused_ddr_template.inc ties
//                      DDR_PRESENT=0) replays byte-identically. Gate s1(b).
//   DDR_PRESENT != 0 : sparse memory behind a behavioral AXI4 slave.
//
// Behavioral contract (DDR_PRESENT=1) — deliberately STRICTER than real
// hardware; violations are $fatal, because every master here is our own
// RTL/BFM and a violation is a bug, not an environment:
//   * INCR bursts only (AxBURST==2'b01), AxSIZE==6 (64 B), no burst may
//     cross a 4 KiB boundary, AxLEN any (<=255).
//   * ONE outstanding write burst and ONE outstanding read burst
//     (readiness backpressure — legal AXI; the real controller pipelines
//     more, ours being stricter only keeps our masters conservative).
//   * W beats are accepted only after their AW (wready waits on aw_busy).
//   * Unwritten bytes read back a deterministic address-salted POISON, so
//     a read-before-write is loudly wrong instead of silently zero:
//         lane k (64-bit) of 64 B word w:  {8'hD0, w[47:0], 8'(k)}
//     Partial-strobe writes read-modify-write against that same poison
//     (matches "garbage in unwritten bytes", but deterministically).
//   * Sparse storage: 64-bit values keyed by {word_index[57:0], lane[2:0]}
//     — plain 64-bit assoc keys/values for maximum Verilator portability.
//
// Runtime knobs (plusargs, all optional):
//   +ddr_lat=N    AR-accept -> first R beat latency, cycles (default 25)
//   +ddr_stall=S  S!=0 seeds a 16-bit LFSR that randomly deasserts
//                 awready/arready/bvalid and gaps rvalid (default 0 = off;
//                 the CDC/backpressure shake analog of +tile_div)
//   +ddr_rdy=N    cycles from reset release to sh_cl_ddr_is_ready
//                 (default 100)
//
// The stats interface acks unconditionally (rdata 0) — the shell-side
// training traffic it carries on hardware has no sim observer here.

`timescale 1ns/1ps

module sh_ddr
  #(
    parameter DDR_PRESENT = 1,

    // NOTE TO CL DEVELOPERS:
    // The below two parameters should not be changed.
    parameter DDR_IO = 1,

    // AXI4 geometry — kit defaults, do not change.
    parameter DATA_WIDTH   = 512,
    parameter ADDR_WIDTH   = 64,
    parameter ID_WIDTH     = 16,
    parameter LEN_WIDTH    = 8,
    parameter AXUSER_WIDTH = 1
    )
   (
    input                             clk,
    input                             rst_n,

    input                             stat_clk,
    input                             stat_rst_n,

    // DDR4 x72 RDIMM physical pins — unmodeled, left undriven (as the stub)
    input logic                       CLK_DIMM_DP,
    input logic                       CLK_DIMM_DN,
    output logic                      M_ACT_N,
    output logic [17:0]               M_MA,
    output logic [1:0]                M_BA,
    output logic [1:0]                M_BG,
    output logic [1:0]                M_CKE,
    output logic [1:0]                M_ODT,
    output logic [1:0]                M_CS_N,
    output logic [1:0]                M_CLK_DN,
    output logic [1:0]                M_CLK_DP,
    output logic                      M_PAR,
    inout        [63:0]               M_DQ,
    inout        [7:0]                M_ECC,
    inout        [17:0]               M_DQS_DP,
    inout        [17:0]               M_DQS_DN,
    output logic                      cl_RST_DIMM_N,

    // AXI-4 from CL
    input logic [ID_WIDTH-1:0]        cl_sh_ddr_axi_awid,
    input logic [ADDR_WIDTH-1:0]      cl_sh_ddr_axi_awaddr,
    input logic [LEN_WIDTH-1:0]       cl_sh_ddr_axi_awlen,
    input logic [2:0]                 cl_sh_ddr_axi_awsize,
    input logic                       cl_sh_ddr_axi_awvalid,
    input logic [1:0]                 cl_sh_ddr_axi_awburst,
    input logic                       cl_sh_ddr_axi_awuser,
    output logic                      cl_sh_ddr_axi_awready,

    input logic [DATA_WIDTH-1:0]      cl_sh_ddr_axi_wdata,
    input logic [(DATA_WIDTH>>3)-1:0] cl_sh_ddr_axi_wstrb,
    input logic                       cl_sh_ddr_axi_wlast,
    input logic                       cl_sh_ddr_axi_wvalid,
    output logic                      cl_sh_ddr_axi_wready,

    output logic [ID_WIDTH-1:0]       cl_sh_ddr_axi_bid,
    output logic [1:0]                cl_sh_ddr_axi_bresp,
    output logic                      cl_sh_ddr_axi_bvalid,
    input logic                       cl_sh_ddr_axi_bready,

    input logic [ID_WIDTH-1:0]        cl_sh_ddr_axi_arid,
    input logic [ADDR_WIDTH-1:0]      cl_sh_ddr_axi_araddr,
    input logic [LEN_WIDTH-1:0]       cl_sh_ddr_axi_arlen,
    input logic [2:0]                 cl_sh_ddr_axi_arsize,
    input logic                       cl_sh_ddr_axi_arvalid,
    input logic [1:0]                 cl_sh_ddr_axi_arburst,
    input logic                       cl_sh_ddr_axi_aruser,
    output logic                      cl_sh_ddr_axi_arready,

    output logic [ID_WIDTH-1:0]       cl_sh_ddr_axi_rid,
    output logic [DATA_WIDTH-1:0]     cl_sh_ddr_axi_rdata,
    output logic [1:0]                cl_sh_ddr_axi_rresp,
    output logic                      cl_sh_ddr_axi_rlast,
    output logic                      cl_sh_ddr_axi_rvalid,
    input logic                       cl_sh_ddr_axi_rready,

    // CFG stats interface
    input logic [7:0]                 sh_ddr_stat_bus_addr,
    input logic [31:0]                sh_ddr_stat_bus_wdata,
    input logic                       sh_ddr_stat_bus_wr,
    input logic                       sh_ddr_stat_bus_rd,
    output logic                      sh_ddr_stat_bus_ack,
    output logic [31:0]               sh_ddr_stat_bus_rdata,

    output logic [7:0]                ddr_sh_stat_int,
    output logic                      sh_cl_ddr_is_ready
   );

  // physical pins: unmodeled in sim, constant-inactive like an idle PHY
  assign M_ACT_N       = 1'b0;
  assign M_MA          = '0;
  assign M_BA          = '0;
  assign M_BG          = '0;
  assign M_CKE         = '0;
  assign M_ODT         = '0;
  assign M_CS_N        = '0;
  assign M_CLK_DN      = '0;
  assign M_CLK_DP      = '0;
  assign M_PAR         = 1'b0;
  assign cl_RST_DIMM_N = 1'b0;

  generate
  if (DDR_PRESENT == 0) begin : g_inert
    // ── stub parity: all outputs 0 (Verilator's undriven-output value) ──────
    assign cl_sh_ddr_axi_awready = 1'b0;
    assign cl_sh_ddr_axi_wready  = 1'b0;
    assign cl_sh_ddr_axi_bid     = '0;
    assign cl_sh_ddr_axi_bresp   = '0;
    assign cl_sh_ddr_axi_bvalid  = 1'b0;
    assign cl_sh_ddr_axi_arready = 1'b0;
    assign cl_sh_ddr_axi_rid     = '0;
    assign cl_sh_ddr_axi_rdata   = '0;
    assign cl_sh_ddr_axi_rresp   = '0;
    assign cl_sh_ddr_axi_rlast   = 1'b0;
    assign cl_sh_ddr_axi_rvalid  = 1'b0;
    assign sh_ddr_stat_bus_ack   = 1'b0;
    assign sh_ddr_stat_bus_rdata = '0;
    assign ddr_sh_stat_int       = '0;
    assign sh_cl_ddr_is_ready    = 1'b0;
  end else begin : g_beh
    // ── sparse storage: 64-bit lanes keyed {word[57:0], lane[2:0]} ──────────
    logic [63:0] mem64 [bit [60:0]];

    function automatic logic [63:0] poison_lane(input bit [57:0] w,
                                                input bit [2:0]  k);
      poison_lane = {8'hD0, w[47:0], 5'b0, k};
    endfunction

    // runtime knobs
    int unsigned cfg_lat, cfg_stall, cfg_rdy, cfg_pipe;
    initial begin
      cfg_lat   = 25;
      cfg_stall = 0;
      cfg_rdy   = 100;
      // +ddr_pipe=2 accepts ONE read AR ahead of the streaming burst (its
      // latency counts down while queued — overlapped activation). The
      // DEFAULT stays 1: today's strict single-outstanding readiness, so
      // every existing gate is knob-for-knob identical. The real
      // controller pipelines more than either setting (IB_FUEL.md §6
      // honest-model disclosure — no sim number is a hardware claim).
      cfg_pipe  = 1;
      void'($value$plusargs("ddr_lat=%d",   cfg_lat));
      void'($value$plusargs("ddr_stall=%d", cfg_stall));
      void'($value$plusargs("ddr_rdy=%d",   cfg_rdy));
      void'($value$plusargs("ddr_pipe=%d",  cfg_pipe));
      if (cfg_lat < 1) cfg_lat = 1;
      if (cfg_pipe < 1) cfg_pipe = 1;
      if (cfg_pipe > 2) cfg_pipe = 2;
    end

    // 16-bit Galois LFSR for ready/valid shake (cfg_stall==0 -> disabled)
    logic [15:0] lfsr_q;
    wire         stall_en = (cfg_stall != 0);
    wire ok_aw = !stall_en | lfsr_q[1];
    wire ok_ar = !stall_en | lfsr_q[6];
    wire ok_b  = !stall_en | lfsr_q[9];
    wire ok_r  = !stall_en | lfsr_q[13];

    // is_ready countdown
    int unsigned rdy_cnt;

    // ── write side: one outstanding burst ───────────────────────────────────
    logic                  aw_busy;
    logic [ID_WIDTH-1:0]   aw_id_q;
    logic [ADDR_WIDTH-1:0] w_addr_q;
    logic [LEN_WIDTH-1:0]  aw_len_q;
    logic [LEN_WIDTH-1:0]  w_beat_q;
    logic                  b_pend;

    assign cl_sh_ddr_axi_awready = !aw_busy && !b_pend && ok_aw;
    assign cl_sh_ddr_axi_wready  = aw_busy;
    assign cl_sh_ddr_axi_bvalid  = b_pend && ok_b;
    assign cl_sh_ddr_axi_bid     = aw_id_q;
    assign cl_sh_ddr_axi_bresp   = 2'b00;

    // ── read side: one streaming burst + (cfg_pipe==2) one pended AR ────────
    logic                  r_busy;
    logic [ID_WIDTH-1:0]   ar_id_q;
    logic [ADDR_WIDTH-1:0] r_addr_q;
    logic [LEN_WIDTH-1:0]  ar_len_q;
    logic [LEN_WIDTH-1:0]  r_beat_q;
    int unsigned           lat_q;
    logic                  p_vld;
    logic [ID_WIDTH-1:0]   p_id;
    logic [ADDR_WIDTH-1:0] p_addr;
    logic [LEN_WIDTH-1:0]  p_len;
    int unsigned           p_lat;

    logic [DATA_WIDTH-1:0] r_word;
    always_comb begin
      for (int k = 0; k < 8; k++) begin
        bit [60:0] key;
        key = {r_addr_q[63:6], 3'(k)};
        r_word[64*k +: 64] = mem64.exists(key) ? mem64[key]
                             : poison_lane(r_addr_q[63:6], 3'(k));
      end
    end

    assign cl_sh_ddr_axi_arready = (!r_busy || ((cfg_pipe > 1) && !p_vld))
                                   && ok_ar;
    assign cl_sh_ddr_axi_rvalid  = r_busy && (lat_q == 0) && ok_r;
    assign cl_sh_ddr_axi_rid     = ar_id_q;
    assign cl_sh_ddr_axi_rdata   = r_word;
    assign cl_sh_ddr_axi_rresp   = 2'b00;
    assign cl_sh_ddr_axi_rlast   = r_busy && (r_beat_q == ar_len_q);

    // burst-legality check, shared by AW and AR
    function automatic void chk_burst(input string ch,
                                      input logic [ADDR_WIDTH-1:0] a,
                                      input logic [LEN_WIDTH-1:0]  len,
                                      input logic [2:0]            size,
                                      input logic [1:0]            burst);
      if (size != 3'd6)
        $fatal(1, "sh_ddr_beh: %s size %0d != 6 (64B only)", ch, size);
      if (burst != 2'b01)
        $fatal(1, "sh_ddr_beh: %s burst %0d != INCR", ch, burst);
      if (a[5:0] != 6'b0)
        $fatal(1, "sh_ddr_beh: %s addr %h not 64B-aligned", ch, a);
      if ((32'(a[11:0]) + 64 * (32'(len) + 1)) > 32'd4096)
        $fatal(1, "sh_ddr_beh: %s burst @%h len %0d crosses 4KiB", ch, a, len);
    endfunction

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        aw_busy  <= 1'b0;
        b_pend   <= 1'b0;
        r_busy   <= 1'b0;
        aw_id_q  <= '0;
        w_addr_q <= '0;
        aw_len_q <= '0;
        w_beat_q <= '0;
        ar_id_q  <= '0;
        r_addr_q <= '0;
        ar_len_q <= '0;
        r_beat_q <= '0;
        lat_q    <= 0;
        p_vld    <= 1'b0;
        p_id     <= '0;
        p_addr   <= '0;
        p_len    <= '0;
        p_lat    <= 0;
        lfsr_q   <= 16'h0001;
        rdy_cnt  <= 0;
      end else begin
        // LFSR (re-seed once from plusarg; free-runs after)
        lfsr_q <= {lfsr_q[14:0], 1'b0} ^ (lfsr_q[15] ? 16'h002D : 16'h0);
        if (rdy_cnt == 0 && stall_en) lfsr_q <= 16'(cfg_stall) | 16'h0001;

        if (rdy_cnt <= cfg_rdy) rdy_cnt <= rdy_cnt + 1;

        // AW accept
        if (cl_sh_ddr_axi_awvalid && cl_sh_ddr_axi_awready) begin
          chk_burst("AW", cl_sh_ddr_axi_awaddr, cl_sh_ddr_axi_awlen,
                    cl_sh_ddr_axi_awsize, cl_sh_ddr_axi_awburst);
          aw_busy  <= 1'b1;
          aw_id_q  <= cl_sh_ddr_axi_awid;
          w_addr_q <= cl_sh_ddr_axi_awaddr;
          aw_len_q <= cl_sh_ddr_axi_awlen;
          w_beat_q <= '0;
        end

        // W beat commit (RMW against poison for partial strobes)
        if (cl_sh_ddr_axi_wvalid && cl_sh_ddr_axi_wready) begin
          for (int k = 0; k < 8; k++) begin
            bit [60:0]  key;
            logic [63:0] lane;
            key = {w_addr_q[63:6], 3'(k)};
            if (|cl_sh_ddr_axi_wstrb[8*k +: 8]) begin
              lane = mem64.exists(key) ? mem64[key]
                     : poison_lane(w_addr_q[63:6], 3'(k));
              for (int b = 0; b < 8; b++)
                if (cl_sh_ddr_axi_wstrb[8*k + b])
                  lane[8*b +: 8] = cl_sh_ddr_axi_wdata[64*k + 8*b +: 8];
              mem64[key] = lane;
            end
          end
          if (cl_sh_ddr_axi_wlast) begin
            if (w_beat_q != aw_len_q)
              $fatal(1, "sh_ddr_beh: wlast at beat %0d, awlen %0d",
                     w_beat_q, aw_len_q);
            aw_busy <= 1'b0;
            b_pend  <= 1'b1;
          end else begin
            if (w_beat_q == aw_len_q)
              $fatal(1, "sh_ddr_beh: missing wlast at beat %0d", w_beat_q);
            w_addr_q <= w_addr_q + 64;
            w_beat_q <= w_beat_q + 1'b1;
          end
        end

        // B retire
        if (cl_sh_ddr_axi_bvalid && cl_sh_ddr_axi_bready)
          b_pend <= 1'b0;

        // AR accept — into the streaming slot, or (cfg_pipe==2, busy) the
        // ONE pend slot; arready above blocks a second pend
        if (cl_sh_ddr_axi_arvalid && cl_sh_ddr_axi_arready) begin
          chk_burst("AR", cl_sh_ddr_axi_araddr, cl_sh_ddr_axi_arlen,
                    cl_sh_ddr_axi_arsize, cl_sh_ddr_axi_arburst);
          if (!r_busy) begin
            r_busy   <= 1'b1;
            ar_id_q  <= cl_sh_ddr_axi_arid;
            r_addr_q <= cl_sh_ddr_axi_araddr;
            ar_len_q <= cl_sh_ddr_axi_arlen;
            r_beat_q <= '0;
            lat_q    <= cfg_lat;
          end else begin
            p_vld  <= 1'b1;
            p_id   <= cl_sh_ddr_axi_arid;
            p_addr <= cl_sh_ddr_axi_araddr;
            p_len  <= cl_sh_ddr_axi_arlen;
            p_lat  <= cfg_lat;
          end
        end

        if (r_busy && lat_q != 0) lat_q <= lat_q - 1;
        if (p_vld && p_lat != 0) p_lat <= p_lat - 1;   // overlapped activate

        // R beat retire; on rlast promote the pended burst (in-order, its
        // remaining latency already partially or fully elapsed)
        if (cl_sh_ddr_axi_rvalid && cl_sh_ddr_axi_rready) begin
          if (r_beat_q == ar_len_q) begin
            if (p_vld) begin
              ar_id_q  <= p_id;
              r_addr_q <= p_addr;
              ar_len_q <= p_len;
              r_beat_q <= '0;
              lat_q    <= p_lat;
              p_vld    <= 1'b0;
            end else begin
              r_busy <= 1'b0;
            end
          end else begin
            r_addr_q <= r_addr_q + 64;
            r_beat_q <= r_beat_q + 1'b1;
          end
        end
      end
    end

    // stats: unconditional ack one cycle after the pulse, no data
    logic stat_ack_q;
    always_ff @(posedge stat_clk) begin
      if (!stat_rst_n) stat_ack_q <= 1'b0;
      else stat_ack_q <= sh_ddr_stat_bus_wr | sh_ddr_stat_bus_rd;
    end
    assign sh_ddr_stat_bus_ack   = stat_ack_q;
    assign sh_ddr_stat_bus_rdata = '0;
    assign ddr_sh_stat_int       = '0;
    assign sh_cl_ddr_is_ready    = (rdy_cnt > cfg_rdy);
  end
  endgenerate

  // unused-input tieoff audit (keeps lint quiet without masking real nets)
  wire unused_ok = &{1'b0, CLK_DIMM_DP, CLK_DIMM_DN,
                     cl_sh_ddr_axi_awuser, cl_sh_ddr_axi_aruser,
                     sh_ddr_stat_bus_addr, sh_ddr_stat_bus_wdata};

endmodule
