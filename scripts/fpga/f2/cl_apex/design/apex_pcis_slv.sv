// apex_pcis_slv.sv — IB-FUEL s2: PCIS -> DDR address decode (shell domain).
//
// Sits between the shell's DMA_PCIS 512-bit AXI4 slave port (AppPF BAR4
// maps onto it, AWS_Shell_Interface_Specification.md:128,216) and the CL's
// sh_ddr AXI4 port. DDR occupies PCIS base 0x0, 64 GiB — the
// cl_dram_hbm_dma precedent (cl_dma_pcis_slv.sv:255) minus its SmartConnect
// crossbar: one target, so the "decode" is a range check.
//
//   in-range  (addr[63:36]==0): forwarded 1:1 to sh_ddr (IDs pass through
//                               and return on sh_ddr's own reflection).
//   out-of-range: never touches sh_ddr — writes are sunk to wlast and
//                 answered DECERR; reads stream arlen+1 DECERR beats with
//                 a recognizable poison payload. The PF never wedges on a
//                 stray BAR4 access (same never-wedge philosophy as the
//                 cl_apex OCL guard and apex_ocl_cdc's poison).
//
// ONE outstanding write burst and ONE outstanding read burst (write and
// read channels independent, per AXI). The load-once fuel line needs no
// pipelining — even ~100 MB/s loads a 7 GiB image in ~70 s once per boot
// (LEVEL_C_INTEGRATION.md §5). Acceptance is gated on ddr_is_ready, which
// hardware-truthfully encodes "no DDR traffic before calibration"
// (IB_FUEL.md §1 s5) — and doubles as the whole-slave tie-off when the CL
// is built without DDR (is_ready stays 0, every output idles at the old
// unused_dma_pcis_template.inc values).
//
// AR-channel ownership note (§9.1 R8): in s3 the reader's AR mux inserts
// between this module's m_ar*/m_r* and sh_ddr (LOAD=here / RUN=reader,
// CSR-switched while idle). s2 wires them straight through.

`timescale 1ns/1ps

module apex_pcis_slv (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         ddr_is_ready,
  // IB-FUEL s3 (R8 RUN mode): while the burst reader owns the DDR AR
  // channel, PCIS READS are answered DECERR here instead of stalling
  // against a mux that will never grant — BAR4 readback during RUN cannot
  // wedge the PF. Writes are unaffected (AW/W/B are always PCIS-owned).
  input  logic         rd_block,

  // ── PCIS slave face (shell DMA_PCIS port set, cl_ports.vh:174-219) ───────
  input  logic [15:0]  s_awid,
  input  logic [63:0]  s_awaddr,
  input  logic [7:0]   s_awlen,
  input  logic [2:0]   s_awsize,
  input  logic [1:0]   s_awburst,
  input  logic         s_awvalid,
  output logic         s_awready,
  input  logic [511:0] s_wdata,
  input  logic [63:0]  s_wstrb,
  input  logic         s_wlast,
  input  logic         s_wvalid,
  output logic         s_wready,
  output logic [15:0]  s_bid,
  output logic [1:0]   s_bresp,
  output logic         s_bvalid,
  input  logic         s_bready,
  input  logic [15:0]  s_arid,
  input  logic [63:0]  s_araddr,
  input  logic [7:0]   s_arlen,
  input  logic [2:0]   s_arsize,
  input  logic [1:0]   s_arburst,
  input  logic         s_arvalid,
  output logic         s_arready,
  output logic [15:0]  s_rid,
  output logic [511:0] s_rdata,
  output logic [1:0]   s_rresp,
  output logic         s_rlast,
  output logic         s_rvalid,
  input  logic         s_rready,

  // ── DDR master face (sh_ddr cl_sh_ddr_axi_* port set) ────────────────────
  output logic [15:0]  m_awid,
  output logic [63:0]  m_awaddr,
  output logic [7:0]   m_awlen,
  output logic [2:0]   m_awsize,
  output logic         m_awvalid,
  output logic [1:0]   m_awburst,
  input  logic         m_awready,
  output logic [511:0] m_wdata,
  output logic [63:0]  m_wstrb,
  output logic         m_wlast,
  output logic         m_wvalid,
  input  logic         m_wready,
  input  logic [15:0]  m_bid,
  input  logic [1:0]   m_bresp,
  input  logic         m_bvalid,
  output logic         m_bready,
  output logic [15:0]  m_arid,
  output logic [63:0]  m_araddr,
  output logic [7:0]   m_arlen,
  output logic [2:0]   m_arsize,
  output logic         m_arvalid,
  output logic [1:0]   m_arburst,
  input  logic         m_arready,
  input  logic [15:0]  m_rid,
  input  logic [511:0] m_rdata,
  input  logic [1:0]   m_rresp,
  input  logic         m_rlast,
  input  logic         m_rvalid,
  output logic         m_rready
);

  localparam logic [1:0]   RESP_DECERR = 2'b11;
  localparam logic [511:0] RD_POISON   = {16{32'hDEC0_0DD0}};

  // DDR at PCIS 0x0, 64 GiB: in-range iff addr[63:36] == 0
  function automatic logic in_range(input logic [63:0] a);
    in_range = (a[63:36] == 28'b0);
  endfunction

  // ── write channel FSM ─────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    W_IDLE, W_FWD_AW, W_FWD_DATA, W_FWD_B, W_SINK_DATA, W_ERR_B
  } w_st_e;
  w_st_e       w_st;
  logic [15:0] aw_id_q;
  logic [63:0] aw_addr_q;
  logic [7:0]  aw_len_q;
  logic [2:0]  aw_size_q;
  logic [1:0]  aw_burst_q;

  assign s_awready = (w_st == W_IDLE) && ddr_is_ready;

  assign m_awid    = aw_id_q;
  assign m_awaddr  = aw_addr_q;
  assign m_awlen   = aw_len_q;
  assign m_awsize  = aw_size_q;
  assign m_awburst = aw_burst_q;
  assign m_awvalid = (w_st == W_FWD_AW);

  // W pass-through while forwarding; sink accepts unconditionally
  assign m_wdata  = s_wdata;
  assign m_wstrb  = s_wstrb;
  assign m_wlast  = s_wlast;
  assign m_wvalid = (w_st == W_FWD_DATA) && s_wvalid;
  assign s_wready = (w_st == W_FWD_DATA) ? m_wready
                  : (w_st == W_SINK_DATA);

  // idle values are the old unused_dma_pcis_template.inc constants (all 0)
  // so a DDR-absent build is literally the old tie-off, not merely
  // valid-qualified-equivalent
  assign s_bid    = (w_st == W_FWD_B) ? m_bid
                  : (w_st == W_ERR_B) ? aw_id_q : '0;
  assign s_bresp  = (w_st == W_FWD_B) ? m_bresp
                  : (w_st == W_ERR_B) ? RESP_DECERR : 2'b00;
  assign s_bvalid = (w_st == W_FWD_B) ? m_bvalid : (w_st == W_ERR_B);
  assign m_bready = (w_st == W_FWD_B) && s_bready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w_st       <= W_IDLE;
      aw_id_q    <= '0;
      aw_addr_q  <= '0;
      aw_len_q   <= '0;
      aw_size_q  <= '0;
      aw_burst_q <= '0;
    end else begin
      unique case (w_st)
        W_IDLE: if (s_awvalid && s_awready) begin
          aw_id_q    <= s_awid;
          aw_addr_q  <= s_awaddr;
          aw_len_q   <= s_awlen;
          aw_size_q  <= s_awsize;
          aw_burst_q <= s_awburst;
          w_st       <= in_range(s_awaddr) ? W_FWD_AW : W_SINK_DATA;
        end
        W_FWD_AW:   if (m_awready) w_st <= W_FWD_DATA;
        W_FWD_DATA: if (s_wvalid && s_wready && s_wlast) w_st <= W_FWD_B;
        W_FWD_B:    if (m_bvalid && s_bready) w_st <= W_IDLE;
        W_SINK_DATA: if (s_wvalid && s_wlast) w_st <= W_ERR_B;
        W_ERR_B:    if (s_bready) w_st <= W_IDLE;
        default:    w_st <= W_IDLE;
      endcase
    end
  end

  // ── read channel FSM ──────────────────────────────────────────────────────
  typedef enum logic [1:0] { R_IDLE, R_FWD_AR, R_FWD_DATA, R_ERR_DATA } r_st_e;
  r_st_e       r_st;
  logic [15:0] ar_id_q;
  logic [63:0] ar_addr_q;
  logic [7:0]  ar_len_q;
  logic [2:0]  ar_size_q;
  logic [1:0]  ar_burst_q;
  logic [7:0]  r_beat_q;

  assign s_arready = (r_st == R_IDLE) && ddr_is_ready;

  assign m_arid    = ar_id_q;
  assign m_araddr  = ar_addr_q;
  assign m_arlen   = ar_len_q;
  assign m_arsize  = ar_size_q;
  assign m_arburst = ar_burst_q;
  assign m_arvalid = (r_st == R_FWD_AR);

  assign s_rid    = (r_st == R_FWD_DATA) ? m_rid
                  : (r_st == R_ERR_DATA) ? ar_id_q : '0;
  assign s_rdata  = (r_st == R_FWD_DATA) ? m_rdata
                  : (r_st == R_ERR_DATA) ? RD_POISON : '0;
  assign s_rresp  = (r_st == R_FWD_DATA) ? m_rresp
                  : (r_st == R_ERR_DATA) ? RESP_DECERR : 2'b00;
  assign s_rlast  = (r_st == R_FWD_DATA) ? m_rlast
                  : (r_st == R_ERR_DATA) && (r_beat_q == ar_len_q);
  assign s_rvalid = (r_st == R_FWD_DATA) ? m_rvalid : (r_st == R_ERR_DATA);
  assign m_rready = (r_st == R_FWD_DATA) && s_rready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      r_st       <= R_IDLE;
      ar_id_q    <= '0;
      ar_addr_q  <= '0;
      ar_len_q   <= '0;
      ar_size_q  <= '0;
      ar_burst_q <= '0;
      r_beat_q   <= '0;
    end else begin
      unique case (r_st)
        R_IDLE: if (s_arvalid && s_arready) begin
          ar_id_q    <= s_arid;
          ar_addr_q  <= s_araddr;
          ar_len_q   <= s_arlen;
          ar_size_q  <= s_arsize;
          ar_burst_q <= s_arburst;
          r_beat_q   <= '0;
          r_st       <= (in_range(s_araddr) && !rd_block) ? R_FWD_AR
                                                         : R_ERR_DATA;
        end
        R_FWD_AR:   if (m_arready) r_st <= R_FWD_DATA;
        R_FWD_DATA: if (m_rvalid && s_rready && m_rlast) r_st <= R_IDLE;
        R_ERR_DATA: if (s_rready) begin
          if (r_beat_q == ar_len_q) r_st <= R_IDLE;
          else r_beat_q <= r_beat_q + 1'b1;
        end
        default:    r_st <= R_IDLE;
      endcase
    end
  end

endmodule
