// cl_apex.sv — APEX attention tile behind the AWS F2 Small Shell.
// S15 stage 1: REAL CL top (supersedes the cl_apex_wrapper.sv skeleton),
// authored against the pinned kit checkout (aws-fpga branch f2, v2.3.3,
// hdk/common/shell_stable/design/interfaces/cl_ports.vh) and lint-checked
// locally with Verilator. ⏸ NOT yet built with Vivado / never on hardware —
// first DCP build validates it (BRINGUP.md step 4).
//
// Integration map (BRINGUP.md §2.2) — OCL AXI-Lite (AppPF BAR0), window
// decode on addr[15:12]:
//   0x0___  bridge regs (this file):
//     0x0000 ID       RO  32'h4139_4558 ("A9EX")
//     0x0004 VERSION  RO  32'h0000_0002 (stage-1 CL)
//     0x0008 SCRATCH  RW
//     0x000C TILE_RST RW  bit0: hold tile in reset (level, default 0)
//     0x0010 ERRSTK   RO  tile err_sticky[15:0] live mirror
//     0x0014 DONESTK  RW  sticky dn_* pulse accumulator {asu,rms,wstage,
//                         astage,ser,scored,squant,feeder,mxe}; W1C any bit
//     0x0018 KVSIDE   RO  {evict_addr, evict_needed, kv_irq} live
//   0x1___  tile csr_* simple bus  (word addr[7:0]; pulse protocol per
//           rtl/csr/csr_regs.sv header: 1-cycle strobe, ready pulses +1cyc)
//   0x2___  KVQ engine kv_* AXI-Lite (addr[7:0]) — forwarded 1:1
// Everything else: smoke-idle — all tile streams tied off, all job pulses
// deasserted, route levels 0. Stage 2 adds the mailbox job driver; stage 3
// adds PCIS/SDE bulk feed (Small Shell has no built-in DMA).
//
// DECISION-F2-1 (b64 first light): tile ran directly on clk_main_a0 (fixed
// 250 MHz), no CDC — and the D=128 stage-2 build then FAILED timing
// (-34.766 ns, ~26 MHz u_squant path; docs/results/f2_stage2_hw/RESULT.md).
//
// DECISION-LC-1 (Level-C integration, 2026-07-21): the CL is now TWO-CLOCK.
// The shell OCL AXI-Lite stays on clk_main_a0; apex_ocl_cdc (4-phase
// toggle req/ack bridge, single-outstanding — matching this FSM) crosses
// into clk_tile, and the ENTIRE original CL body (this FSM + mailbox +
// tile) runs on clk_tile unchanged. clk_tile source:
//   * simulation (+define+APEX_SIM_TILE_DIV): an in-RTL divider from
//     clk_main_a0, ratio runtime-set via +tile_div=N (freq = 250/(2N) MHz;
//     N=5 -> 25 MHz). Edge-aligned is fine for protocol proof; true async
//     margin is the build's constraint job, not the sim's.
//   * synthesis: the CL-internal clock-generator IP output (AWS_CLK_GEN
//     pattern per the kit's multi-clock example) — see the `else branch
//     and cl_apex_cdc.xdc notes; frequency chosen at build per the
//     ~26 MHz u_squant bound (or higher if u_squant is later pipelined).

`timescale 1ns/1ps

module cl_apex
  import apex_pkg::*;
(
  `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "unused_flr_template.inc"
// NOTE: unused_ddr_template.inc and unused_dma_pcis_template.inc are gone
// as of IB-FUEL s2 — sh_ddr is instantiated for real (DDR_PRESENT build-
// selectable via APEX_CL_DDR) behind the PCIS->DDR decode at the end of
// this file. At APEX_CL_DDR=0 every replaced tie-off value is reproduced
// exactly (see the fuel section).
// NOTE: unused_cl_sda_template.inc deliberately NOT included — SDA is
// tied off in the sim branch below and feeds AWS_CLK_GEN's control AXIL
// in the synth branch (cl_mem_perf pattern; needed for MMCM_LOCK_REG).
`include "unused_apppf_irq_template.inc"
`include "unused_pcim_template.inc"

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;

  // ── reset pipeline (kit idiom: register rst_main_n twice). Named
  // rst_main_n_sync because the kit's unused_ddr_template.inc references
  // that signal — declaring + driving it here binds the template to a real
  // reset instead of an implicit undriven net (the kit example relies on
  // implicit creation; we don't).
  logic rstn_q, rst_main_n_sync;
  always_ff @(posedge clk_main_a0) begin
    rstn_q          <= rst_main_n;
    rst_main_n_sync <= rstn_q;
  end

  // ── DECISION-LC-1: tile clock + tile-domain reset ─────────────────────────
  logic clk_tile;
`ifdef APEX_SIM_TILE_DIV
  // simulation: divide clk_main_a0 by 2*tile_div (runtime +tile_div=N).
  // Free-running from time 0 (initial), independent of reset — the tile
  // reset synchronizer below needs live clk_tile edges to release.
  int unsigned tile_div;
  logic [7:0]  divcnt;
  initial begin
    tile_div = 5;                      // default 25 MHz
    void'($value$plusargs("tile_div=%d", tile_div));
    if (tile_div < 1) tile_div = 1;
    clk_tile = 1'b0;
    divcnt   = 8'd0;
  end
  always @(posedge clk_main_a0) begin
    if (divcnt >= 8'(tile_div - 1)) begin
      divcnt   <= 8'd0;
      clk_tile <= ~clk_tile;
    end else begin
      divcnt <= divcnt + 8'd1;
    end
  end
  // sim: SDA unused — tie off exactly as unused_cl_sda_template.inc does
  assign cl_sda_awready = 1'b0;
  assign cl_sda_wready  = 1'b0;
  assign cl_sda_bvalid  = 1'b0;
  assign cl_sda_bresp   = 2'b0;
  assign cl_sda_arready = 1'b0;
  assign cl_sda_rvalid  = 1'b0;
  assign cl_sda_rdata   = 32'b0;
  assign cl_sda_rresp   = 2'b0;
`else
  // synthesis (DECISION-LC-1 RESOLVED 2026-07-21, kit facts verified):
  // clk_tile = AWS_CLK_GEN's o_clk_extra_a1 under RECIPE A2 = 15.625 MHz
  // (64 ns period vs the 38.99 ns u_squant bound — 60% margin, no tile
  // pipelining). The dynamic-clkgen route is DEAD: fpga_clkgen min is
  // 87 MHz (sdk fpga_libs FREQ table). Build requirements:
  //   * aws_build_dcp_from_cl.py --aws_clk_gen --clock_recipe_a A2
  //     (--aws_clk_gen is a HARD GATE: script aborts on recipe w/o it)
  //   * instantiate hdk/common/lib/aws_clk_gen.sv here per the kit's
  //     cl_mem_perf pattern: o_clk_extra_a1 -> clk_tile, groups B/C/HBM
  //     disabled, s_axil_ctrl_* wired to the shell SDA AXIL (needed for
  //     MMCM_LOCK_REG / SYS_RST_REG);
  //   * cl_apex_cdc.xdc: ASYNC_REG on apex_ocl_cdc *_meta/*_sync pairs +
  //     set_max_delay -datapath_only on addr_q/wdat_q/wstrb_q/is_wr_q and
  //     resp_rdata/resp_code (kit tcl auto-creates the recipe clocks);
  //   * HOST BRING-UP ORDER (AWS_CLK_GEN_spec.md:165): extra clocks are
  //     HELD STATIC LOW until MMCM lock — poll MMCM_LOCK_REG and release
  //     SYS_RST BEFORE any BAR0 traffic, else apex_ocl_cdc's dead-clock
  //     guard poisons (by design) and the CL answers SLVERR until reset.
  //
  // Sources: aws_clk_gen.sv + deps are read by the kit's synth_cl_header.tcl
  // into EVERY CL build (no filelist change). Control AXIL = the shell SDA
  // bus, whole (we use SDA for nothing else); cl_mem_perf splits it with an
  // sda_xbar — we don't need the xbar.
  aws_clk_gen #(
    .CLK_GRP_A_EN (1),
    .CLK_GRP_B_EN (0),
    .CLK_GRP_C_EN (0),
    .CLK_HBM_EN   (0)
  ) AWS_CLK_GEN (   // NAME IS LOAD-BEARING: aws_clock_properties.tcl
                    // hardcodes WRAPPER/CL/AWS_CLK_GEN/... to apply the
                    // recipe onto the MMCM — any other name = recipe
                    // silently NOT applied (built at XCI-default 125 MHz)

    .i_clk_main_a0       (clk_main_a0),
    .i_rst_main_n        (rst_main_n_sync),
    .i_clk_hbm_ref       (clk_hbm_ref),
    .s_axil_ctrl_awaddr  (sda_cl_awaddr),
    .s_axil_ctrl_awvalid (sda_cl_awvalid),
    .s_axil_ctrl_awready (cl_sda_awready),
    .s_axil_ctrl_wdata   (sda_cl_wdata),
    .s_axil_ctrl_wstrb   (sda_cl_wstrb),
    .s_axil_ctrl_wvalid  (sda_cl_wvalid),
    .s_axil_ctrl_wready  (cl_sda_wready),
    .s_axil_ctrl_bresp   (cl_sda_bresp),
    .s_axil_ctrl_bvalid  (cl_sda_bvalid),
    .s_axil_ctrl_bready  (sda_cl_bready),
    .s_axil_ctrl_araddr  (sda_cl_araddr),
    .s_axil_ctrl_arvalid (sda_cl_arvalid),
    .s_axil_ctrl_arready (cl_sda_arready),
    .s_axil_ctrl_rdata   (cl_sda_rdata),
    .s_axil_ctrl_rresp   (cl_sda_rresp),
    .s_axil_ctrl_rvalid  (cl_sda_rvalid),
    .s_axil_ctrl_rready  (sda_cl_rready),
    .o_clk_hbm_ref       (),
    .o_clk_main_a0       (),
    .o_clk_extra_a1      (clk_tile),
    .o_clk_extra_a2      (),
    .o_clk_extra_a3      (),
    .o_clk_extra_b0      (),
    .o_clk_extra_b1      (),
    .o_clk_extra_c0      (),
    .o_clk_extra_c1      (),
    .o_clk_hbm_axi       (),
    .o_cl_rst_hbm_axi_n  (),
    .o_cl_rst_hbm_ref_n  (),
    .o_cl_rst_c1_n       (),
    .o_cl_rst_c0_n       (),
    .o_cl_rst_b1_n       (),
    .o_cl_rst_b0_n       (),
    .o_cl_rst_a3_n       (),
    .o_cl_rst_a2_n       (),
    .o_cl_rst_a1_n       (),
    .o_cl_rst_main_n     ()
  );
`endif

  logic t_rstn_q, rst_tile_n;
  always_ff @(posedge clk_tile) begin
    t_rstn_q   <= rst_main_n;
    rst_tile_n <= t_rstn_q;
  end

  // ── OCL CDC bridge: shell 250 MHz ⇄ tile clock ────────────────────────────
  // b_* = the bridge's tile-side AXI-Lite, consumed by the ORIGINAL FSM
  // below exactly as it consumed the shell's ocl_* signals.
  logic        b_awvalid, b_awready, b_wvalid, b_wready;
  logic [31:0] b_awaddr, b_wdata;
  logic [3:0]  b_wstrb;
  logic        b_bvalid, b_bready;
  logic [1:0]  b_bresp;
  logic        b_arvalid, b_arready, b_rvalid, b_rready;
  logic [31:0] b_araddr, b_rdata;
  logic [1:0]  b_rresp;

  apex_ocl_cdc u_ocl_cdc (
    .clk_shell   (clk_main_a0),
    .rst_shell_n (rst_main_n_sync),
    .s_awvalid   (ocl_cl_awvalid),
    .s_awaddr    (ocl_cl_awaddr),
    .s_awready   (cl_ocl_awready),
    .s_wvalid    (ocl_cl_wvalid),
    .s_wdata     (ocl_cl_wdata),
    .s_wstrb     (ocl_cl_wstrb),
    .s_wready    (cl_ocl_wready),
    .s_bvalid    (cl_ocl_bvalid),
    .s_bresp     (cl_ocl_bresp),
    .s_bready    (ocl_cl_bready),
    .s_arvalid   (ocl_cl_arvalid),
    .s_araddr    (ocl_cl_araddr),
    .s_arready   (cl_ocl_arready),
    .s_rvalid    (cl_ocl_rvalid),
    .s_rdata     (cl_ocl_rdata),
    .s_rresp     (cl_ocl_rresp),
    .s_rready    (ocl_cl_rready),
    .clk_tile    (clk_tile),
    .rst_tile_n  (rst_tile_n),
    .t_awvalid   (b_awvalid),
    .t_awaddr    (b_awaddr),
    .t_awready   (b_awready),
    .t_wvalid    (b_wvalid),
    .t_wdata     (b_wdata),
    .t_wstrb     (b_wstrb),
    .t_wready    (b_wready),
    .t_bvalid    (b_bvalid),
    .t_bresp     (b_bresp),
    .t_bready    (b_bready),
    .t_arvalid   (b_arvalid),
    .t_araddr    (b_araddr),
    .t_arready   (b_arready),
    .t_rvalid    (b_rvalid),
    .t_rdata     (b_rdata),
    .t_rresp     (b_rresp),
    .t_rready    (b_rready)
  );

  // byte strobes were never consumed by this CL (full-word writes only)
  wire unused_wstrb = &{1'b0, b_wstrb};

  // ── bridge registers ──────────────────────────────────────────────────────
  localparam [31:0] APEX_CL_ID  = 32'h4139_4558;   // "A9EX"
  localparam [31:0] APEX_CL_VER = 32'h0000_0002;   // stage-1 CL

  // Tile config MUST track the verified L3 reference build (tb_apex_l3.sv:
  // DEPTH=256, FEED_ROWS_MAX=31, STAGE_R_MAX=31) — the trace-replay regops
  // are generated by that choreography. DEPTH=256 holds 2*T records at the
  // T_ROW_MAX=128 envelope (gen_l3_vectors F-4); at 128 every job with T>64
  // stalls polling KVQ OCC.
  localparam int unsigned CL_KVQ_DEPTH = 256;
  localparam int unsigned CL_ROWS_MAX  = 31;

  logic [31:0] scratch_q;
  logic        tile_rst_q;
  logic [8:0]  done_stk_q;                          // sticky dn_* bits

  // ── tile-facing wires ─────────────────────────────────────────────────────
  logic [7:0]  t_csr_addr;
  logic [31:0] t_csr_wdata;
  logic        t_csr_write, t_csr_read;
  logic [31:0] t_csr_rdata;
  logic        t_csr_ready;

  // W-G3 flip: tile walker fetch-request channel (tile domain), plumbed to
  // apex_fuel_ctl's walker producer (wk_req_*, IB_FUEL.md §2.4)
  logic        t_wf_valid, t_wf_ready;
  logic [63:0] t_wf_req;

  logic [7:0]  t_kv_awaddr, t_kv_araddr;
  logic        t_kv_awvalid, t_kv_awready;
  logic [31:0] t_kv_wdata;
  logic        t_kv_wvalid, t_kv_wready;
  logic [1:0]  t_kv_bresp;
  logic        t_kv_bvalid, t_kv_bready;
  logic        t_kv_arvalid, t_kv_arready;
  logic [31:0] t_kv_rdata;
  logic [1:0]  t_kv_rresp;
  logic        t_kv_rvalid, t_kv_rready;
  logic        t_kv_irq, t_kv_evict_needed;
  logic [$clog2(CL_KVQ_DEPTH)-1:0] t_kv_evict_addr;

  logic [15:0] t_err_sticky;
  logic        t_dn_mxe, t_dn_feeder, t_dn_squant, t_dn_scored, t_dn_ser,
               t_dn_astage, t_dn_wstage, t_dn_rms, t_dn_asu;
  wire  [8:0]  dn_bus = {t_dn_asu, t_dn_rms, t_dn_wstage, t_dn_astage,
                         t_dn_ser, t_dn_scored, t_dn_squant, t_dn_feeder,
                         t_dn_mxe};

  // ── stage-2 mailbox (window 0x3___): streams / jobs / routes / captures ──
  logic [31:0] mb_rdata;
  logic        mb_xa_v, mb_xa_r, mb_xa_last;
  logic signed [7:0]  mb_xa_x;
  logic        mb_xg_v, mb_xg_r;
  logic signed [15:0] mb_xg_g;
  logic        mb_qs_v, mb_qs_r, mb_cs_v, mb_cs_r;
  logic [31:0] mb_qs_d, mb_cs_d;
  logic        mb_xw_v, mb_xw_r, mb_ds_v, mb_ds_r;
  lane8_beat_t mb_xw_b;
  mxe_desc_t   mb_ds_d;
  // IB-FUEL s3: muxed xw into the tile + the fuel-side leg (fuel section)
  logic        tile_xw_v, tile_xw_r;
  lane8_beat_t tile_xw_b;
  logic        fuel_xw_valid, fuel_xw_ready;
  logic [63:0] fuel_xw_data;
  logic        mb_fj_v, mb_fj_r, mb_qj_v, mb_qj_r, mb_dj_v, mb_dj_r;
  logic        mb_lj_v, mb_lj_r, mb_aj_v, mb_aj_r, mb_wj_v, mb_wj_r;
  logic [11:0] mb_fj_rows, mb_qj_cols, mb_dj_cols;
  logic        mb_qj_mode;
  logic [7:0]  mb_lj_beats;
  logic [3:0]  mb_lj_lanes;
  logic        mb_aj_op, mb_aj_bank, mb_wj_op, mb_wj_bank;
  logic [1:0]  mb_aj_pat, mb_wj_pat;
  logic [4:0]  mb_aj_rows, mb_aj_sel, mb_wj_rows, mb_wj_sel;
  logic [4:0]  mb_aj_nb, mb_wj_nb;               // sized for max D=128 (NBW=5); D=64 uses [3:0]
  logic        mb_rt_fsrc, mb_rt_fdst, mb_rt_asrc, mb_rt_wsrc, mb_rt_ssrc,
               mb_rt_kvu, mb_rt_impclr;
  logic [1:0]  mb_rt_rdst;
  logic [6:0]  mb_rt_tblk;
  logic [15:0] mb_rt_ihi, mb_rt_ilo;
  logic        mb_ro_v, mb_ro_r, mb_fs_v, mb_fs_r, mb_ss_v, mb_ss_r,
               mb_td_v, mb_td_r;
  lane32_beat_t mb_ro_b;
  logic [15:0] mb_fs_d, mb_ss_d;
  logic        mb_fs_l, mb_ss_l, mb_td_fp16;
  kvq_tier_e   mb_td_tier;
  logic [6:0]  mb_td_blk;

  // ── OCL AXI-Lite slave: ONE outstanding transaction, writes then reads ───
  typedef enum logic [3:0] {
    ST_IDLE, ST_WR_DISPATCH, ST_WR_CSR_WAIT, ST_WR_KV_DATA, ST_WR_KV_B,
    ST_WR_RESP, ST_RD_DISPATCH, ST_RD_CSR_WAIT, ST_RD_KV_AR, ST_RD_KV_R,
    ST_RD_RESP
  } st_e;
  st_e         st_q;
  logic [15:0] a_q;                                  // latched addr[15:0]
  logic [31:0] wdat_q, rdat_q;
  logic [1:0]  resp_q;
  logic        kv_aw_done_q, kv_w_done_q;
  logic [6:0]  guard_q;                              // hang guard (SLVERR)

  wire [3:0] win = a_q[15:12];                       // 0=bridge 1=csr 2=kvq

  // tile-domain slave face (fed by the CDC bridge; protocol unchanged)
  assign b_awready = (st_q == ST_IDLE) && b_awvalid && b_wvalid;
  assign b_wready  = b_awready;
  assign b_bresp   = resp_q;
  assign b_bvalid  = (st_q == ST_WR_RESP);
  assign b_arready = (st_q == ST_IDLE) &&
                     !(b_awvalid && b_wvalid) && b_arvalid;
  assign b_rdata   = rdat_q;
  assign b_rresp   = resp_q;
  assign b_rvalid  = (st_q == ST_RD_RESP);

  // tile csr strobes: exactly one cycle each (state-decoded)
  assign t_csr_addr  = a_q[7:0];
  assign t_csr_wdata = wdat_q;
  assign t_csr_write = (st_q == ST_WR_DISPATCH) && (win == 4'h1);
  assign t_csr_read  = (st_q == ST_RD_DISPATCH) && (win == 4'h1);

  // mailbox strobes (window 0x3): single-cycle, rdata combinational
  wire mb_write = (st_q == ST_WR_DISPATCH) && (win == 4'h3);
  wire mb_read  = (st_q == ST_RD_DISPATCH) && (win == 4'h3);

  // ── IB-FUEL s3: FUEL CSR faces (window 0x0 regs 0x20-0x30, tile domain;
  //    IB_FUEL.md §5) — state lives in apex_fuel_ctl, decoded here ─────────
  logic [31:0] fuel_ctrl_rd, fuel_base_rd, fuel_beats_rd, fuel_stat_rd,
               fuel_err_rd;
  logic        fuel_src, fuel_arown;
  wire fuel_ctrl_wr  = (st_q == ST_WR_DISPATCH) && (win == 4'h0)
                       && (a_q[7:0] == 8'h20);
  wire fuel_base_wr  = (st_q == ST_WR_DISPATCH) && (win == 4'h0)
                       && (a_q[7:0] == 8'h24);
  wire fuel_beats_wr = (st_q == ST_WR_DISPATCH) && (win == 4'h0)
                       && (a_q[7:0] == 8'h28);
  wire fuel_err_wr   = (st_q == ST_WR_DISPATCH) && (win == 4'h0)
                       && (a_q[7:0] == 8'h30);
  // audit: a mailbox XW_PUSH while the fuel source is selected is a host
  // bug — sticky FUEL_ERR.host_push (the beat still lands in the mailbox
  // FIFO, which is stalled in fuel mode, never corrupting the wire stream)
  wire fuel_hostpush = mb_write && (a_q[11:0] == 12'h048) && fuel_src;

  // kvq AXI-Lite forwarding
  assign t_kv_awaddr  = a_q[7:0];
  assign t_kv_awvalid = (st_q == ST_WR_KV_DATA) && !kv_aw_done_q;
  assign t_kv_wdata   = wdat_q;
  assign t_kv_wvalid  = (st_q == ST_WR_KV_DATA) && !kv_w_done_q;
  assign t_kv_bready  = (st_q == ST_WR_KV_B);
  assign t_kv_araddr  = a_q[7:0];
  assign t_kv_arvalid = (st_q == ST_RD_KV_AR);
  assign t_kv_rready  = (st_q == ST_RD_KV_R);

  function automatic logic [31:0] bridge_rd(input logic [7:0] a);
    case (a)
      8'h00:   bridge_rd = APEX_CL_ID;
      8'h04:   bridge_rd = APEX_CL_VER;
      8'h08:   bridge_rd = scratch_q;
      8'h0C:   bridge_rd = {31'b0, tile_rst_q};
      8'h10:   bridge_rd = {16'b0, t_err_sticky};
      8'h14:   bridge_rd = {23'b0, done_stk_q};
      // pad derived from the evict-address width so it tracks CL_KVQ_DEPTH
      8'h18:   bridge_rd = {{(30 - $clog2(CL_KVQ_DEPTH)){1'b0}},
                            t_kv_evict_addr, t_kv_evict_needed, t_kv_irq};
      // IB-FUEL s3 window (IB_FUEL.md §5)
      8'h20:   bridge_rd = fuel_ctrl_rd;
      8'h24:   bridge_rd = fuel_base_rd;
      8'h28:   bridge_rd = fuel_beats_rd;
      8'h2C:   bridge_rd = fuel_stat_rd;
      8'h30:   bridge_rd = fuel_err_rd;
      default: bridge_rd = {24'hDEAD00, a};
    endcase
  endfunction

  always_ff @(posedge clk_tile) begin
    if (!rst_tile_n) begin
      st_q         <= ST_IDLE;
      scratch_q    <= 32'h0;
      tile_rst_q   <= 1'b0;
      done_stk_q   <= 9'h0;
      kv_aw_done_q <= 1'b0;
      kv_w_done_q  <= 1'b0;
      guard_q      <= 7'h0;
      a_q          <= 16'h0;
      wdat_q       <= 32'h0;
      rdat_q       <= 32'h0;
      resp_q       <= 2'b00;
    end else begin
      done_stk_q <= done_stk_q | dn_bus;             // sticky accumulate

      unique case (st_q)
        ST_IDLE: begin
          guard_q <= 7'h0;
          resp_q  <= 2'b00;
          if (b_awvalid && b_wvalid) begin           // write wins arbitration
            a_q          <= b_awaddr[15:0];
            wdat_q       <= b_wdata;
            kv_aw_done_q <= 1'b0;
            kv_w_done_q  <= 1'b0;
            st_q         <= ST_WR_DISPATCH;
          end else if (b_arvalid) begin
            a_q  <= b_araddr[15:0];
            st_q <= ST_RD_DISPATCH;
          end
        end

        // ── writes ──────────────────────────────────────────────────────────
        ST_WR_DISPATCH: begin
          unique case (win)
            4'h0: begin                              // bridge regs
              case (a_q[7:0])
                8'h08: scratch_q  <= wdat_q;
                8'h0C: tile_rst_q <= wdat_q[0];
                8'h14: done_stk_q <= (done_stk_q & ~wdat_q[8:0]) | dn_bus;
                default: ;                           // ignored, OKAY
              endcase
              st_q <= ST_WR_RESP;
            end
            4'h1: st_q <= ST_WR_CSR_WAIT;            // strobe fired this cycle
            4'h2: st_q <= ST_WR_KV_DATA;
            4'h3: st_q <= ST_WR_RESP;                // mailbox: 1-cycle write
            default: begin resp_q <= 2'b10; st_q <= ST_WR_RESP; end // SLVERR
          endcase
        end
        ST_WR_CSR_WAIT: begin
          guard_q <= guard_q + 7'h1;
          if (t_csr_ready) st_q <= ST_WR_RESP;
          else if (&guard_q) begin resp_q <= 2'b10; st_q <= ST_WR_RESP; end
        end
        ST_WR_KV_DATA: begin
          guard_q <= guard_q + 7'h1;
          if (t_kv_awvalid && t_kv_awready) kv_aw_done_q <= 1'b1;
          if (t_kv_wvalid  && t_kv_wready)  kv_w_done_q  <= 1'b1;
          if ((kv_aw_done_q || (t_kv_awvalid && t_kv_awready)) &&
              (kv_w_done_q  || (t_kv_wvalid  && t_kv_wready)))
            st_q <= ST_WR_KV_B;
          else if (&guard_q) begin resp_q <= 2'b10; st_q <= ST_WR_RESP; end
        end
        ST_WR_KV_B: begin
          guard_q <= guard_q + 7'h1;
          if (t_kv_bvalid) begin resp_q <= t_kv_bresp; st_q <= ST_WR_RESP; end
          else if (&guard_q) begin resp_q <= 2'b10; st_q <= ST_WR_RESP; end
        end
        ST_WR_RESP: if (b_bready) st_q <= ST_IDLE;

        // ── reads ───────────────────────────────────────────────────────────
        ST_RD_DISPATCH: begin
          unique case (win)
            4'h0: begin rdat_q <= bridge_rd(a_q[7:0]); st_q <= ST_RD_RESP; end
            4'h1: st_q <= ST_RD_CSR_WAIT;             // strobe fired this cycle
            4'h2: st_q <= ST_RD_KV_AR;
            4'h3: begin rdat_q <= mb_rdata; st_q <= ST_RD_RESP; end // mailbox
            default: begin
              rdat_q <= 32'hDEAD_FFFF; resp_q <= 2'b10; st_q <= ST_RD_RESP;
            end
          endcase
        end
        ST_RD_CSR_WAIT: begin
          guard_q <= guard_q + 7'h1;
          if (t_csr_ready) begin rdat_q <= t_csr_rdata; st_q <= ST_RD_RESP; end
          else if (&guard_q) begin resp_q <= 2'b10; st_q <= ST_RD_RESP; end
        end
        ST_RD_KV_AR: begin
          guard_q <= guard_q + 7'h1;
          if (t_kv_arready) st_q <= ST_RD_KV_R;
          else if (&guard_q) begin resp_q <= 2'b10; st_q <= ST_RD_RESP; end
        end
        ST_RD_KV_R: begin
          guard_q <= guard_q + 7'h1;
          if (t_kv_rvalid) begin
            rdat_q <= t_kv_rdata; resp_q <= t_kv_rresp; st_q <= ST_RD_RESP;
          end else if (&guard_q) begin resp_q <= 2'b10; st_q <= ST_RD_RESP; end
        end
        ST_RD_RESP: if (b_rready) st_q <= ST_IDLE;

        default: st_q <= ST_IDLE;
      endcase
    end
  end

  // build-selectable head dim: +define+APEX_CL_D=128 for the 7B-trace
  // replay build (b128); default 64 = the smoke/first-light config (b64)
`ifndef APEX_CL_D
  `define APEX_CL_D 64
`endif
  localparam int unsigned CL_D   = `APEX_CL_D;

  // I-C build knobs (same +define pattern as APEX_CL_D). Defaults reproduce
  // today's shipped CL EXACTLY, so an unflagged build is byte-identical:
  //   +define+APEX_CL_GQA=4   -> per-KV-head KVQ + composite banks (S4b/F6i)
  //   +define+APEX_CL_DM=3584 -> wide RMSNorm + LAYER model width
  // NOTE: these were NOT plumbed before, so any earlier "GQA-4/wide-D CL"
  // build would silently have been the default GQA-1/narrow config.
`ifndef APEX_CL_GQA
  `define APEX_CL_GQA 1
`endif
`ifndef APEX_CL_DM
  `define APEX_CL_DM 128
`endif
  // ── P2 b64 knobs (Qwen2.5-0.5B: hidden 896, H=14/H_kv=2, head_dim 64) ────
  // Two apex_top parameters the CL never plumbed, both defaulted to the
  // value apex_top itself defaults to, so an UNFLAGGED build elaborates
  // byte-identically (proven: the D=128 DDR=0 verilated sources hash the
  // same before and after this block landed — docs/results/p2_b64_cl).
  //
  //   +define+APEX_CL_DMODEL=<n>  -> CFG_DM, the MODEL-WIDE row family
  //       (apex_top.sv:112-124 GAP D). DEFAULT `APEX_CL_D` reproduces
  //       apex_top's own `CFG_DM = CFG_D`. Legal values are bounded by the
  //       C-1 feeder, which apex_top elaborates at CFG_DM
  //       (apex_top.sv:1866 seam_feeder_quant #(.D(CFG_DM))) and which
  //       refuses anything but 64/128 (seam_feeder_quant.sv:100-102).
  //       THE 0.5B IMAGE LEAVES THIS AT CFG_D=64 ON PURPOSE: a SPLIT build
  //       (CFG_D=64 / CFG_DM=128) is legal to g_chk_dm but breaks the
  //       per-head routes that frame PER-HEAD rows through the same feeder
  //       — the RoPE q_sink egress (apex_top.sv:1186) and the KVQ-readback
  //       requant route — which is the GAP-D note at apex_top.sv:1858-1864.
  //       At CFG_DM=CFG_D=64 a 896-wide model row is 14 x 64 feeder frames.
  //
  //   +define+APEX_CL_QSTAGE=<h>  -> QSTAGE_H_MAX, how many heads' q rows a
  //       fmt=1 walk may consume (apex_top.sv:138-148). DEFAULT 1 = the
  //       pre-I-C tile verbatim (the whole per-head s_q file constant-folds
  //       away). Must be <= WALK2_H_MAX (30) and <= STAGE_R_MAX (the q rows
  //       live in act-stage bank 1, one row per head); the 0.5B image sets
  //       14 = H. Guarded at elaboration below — a knob that silently
  //       clipped would be worse than no knob.
`ifndef APEX_CL_DMODEL
  `define APEX_CL_DMODEL `APEX_CL_D
`endif
`ifndef APEX_CL_QSTAGE
  `define APEX_CL_QSTAGE 1
`endif
  // ── W4-DEBUT OVERLAY (scratch tree only, NOT in the committed repo) ──────
  //   +define+APEX_CL_W4=1 -> apex_top W4_LANE=1 (the D-031 W4 ingest lane,
  //   rtl/top/glue/apex_w4_ingest.sv; CSR window 0x9C-0xAC). Default 0 keeps
  //   the CL byte-identical to the committed cl_apex.sv. The committed CL
  //   never plumbed W4_LANE, so a twin built WITHOUT this define elaborates
  //   the lane ABSENT (0x9C-0xAC read 0xDEADBEEF) — exactly what the debut
  //   program's phase-A presence probe distinguishes on silicon.
`ifndef APEX_CL_W4
  `define APEX_CL_W4 0
`endif
  localparam bit          CL_W4     = 1'(`APEX_CL_W4);
  localparam int unsigned CL_GQA    = `APEX_CL_GQA;
  localparam int unsigned CL_DM     = `APEX_CL_DM;
  localparam int unsigned CL_DMODEL = `APEX_CL_DMODEL;
  localparam int unsigned CL_QSTAGE = `APEX_CL_QSTAGE;
  localparam int unsigned CL_NBW = $clog2(CL_D / 8) + 1;

  // Elaboration guards for the two NEW knobs (the apex_top-side rules the
  // tile itself cannot restate for a CL that never passed the parameter).
  // WALK2_H_MAX is seq_walker_pkg's fmt=1 RQ-table cap; CL_ROWS_MAX is what
  // this CL passes as STAGE_R_MAX.
  if (CL_QSTAGE < 1 || CL_QSTAGE > CL_ROWS_MAX
      || CL_QSTAGE > seq_walker_pkg::WALK2_H_MAX) begin : g_chk_qstage
    $error("cl_apex: APEX_CL_QSTAGE must be in [1, min(STAGE_R_MAX, WALK2_H_MAX)]");
  end
  if (CL_DMODEL != 64 && CL_DMODEL != 128) begin : g_chk_dmodel
    $error("cl_apex: APEX_CL_DMODEL must be 64 or 128 (seam_feeder_quant.sv:100-102)");
  end
  if (CL_DMODEL < CL_D) begin : g_chk_dmodel_ge
    $error("cl_apex: APEX_CL_DMODEL must be >= APEX_CL_D (apex_top GAP D g_chk_dm)");
  end

  apex_f2_mailbox #(.NBW(CL_NBW)) u_mb (
    .clk (clk_tile), .rst_n (rst_tile_n & ~tile_rst_q),
    .mb_addr (a_q[11:0]), .mb_wdata (wdat_q),
    .mb_write (mb_write), .mb_read (mb_read), .mb_rdata (mb_rdata),
    .xa_valid (mb_xa_v), .xa_ready (mb_xa_r), .xa_x (mb_xa_x),
    .xa_last (mb_xa_last),
    .xg_valid (mb_xg_v), .xg_ready (mb_xg_r), .xg_gamma (mb_xg_g),
    .qs_valid (mb_qs_v), .qs_ready (mb_qs_r), .qs_data (mb_qs_d),
    .cs_valid (mb_cs_v), .cs_ready (mb_cs_r), .cs_data (mb_cs_d),
    .xw_valid (mb_xw_v), .xw_ready (mb_xw_r), .xw_beat (mb_xw_b),
    .ds_valid (mb_ds_v), .ds_ready (mb_ds_r), .ds_desc (mb_ds_d),
    .fj_valid (mb_fj_v), .fj_ready (mb_fj_r), .fj_rows (mb_fj_rows),
    .qj_valid (mb_qj_v), .qj_ready (mb_qj_r), .qj_mode (mb_qj_mode),
    .qj_cols (mb_qj_cols),
    .dj_valid (mb_dj_v), .dj_ready (mb_dj_r), .dj_cols (mb_dj_cols),
    .lj_valid (mb_lj_v), .lj_ready (mb_lj_r), .lj_beats (mb_lj_beats),
    .lj_lanes (mb_lj_lanes),
    .aj_valid (mb_aj_v), .aj_ready (mb_aj_r), .aj_op (mb_aj_op),
    .aj_bank (mb_aj_bank), .aj_pat (mb_aj_pat), .aj_rows (mb_aj_rows),
    .aj_nb (mb_aj_nb), .aj_sel (mb_aj_sel),
    .wj_valid (mb_wj_v), .wj_ready (mb_wj_r), .wj_op (mb_wj_op),
    .wj_bank (mb_wj_bank), .wj_pat (mb_wj_pat), .wj_rows (mb_wj_rows),
    .wj_nb (mb_wj_nb), .wj_sel (mb_wj_sel),
    .rt_feeder_src (mb_rt_fsrc), .rt_feeder_dst (mb_rt_fdst),
    .rt_act_src (mb_rt_asrc), .rt_wgt_src (mb_rt_wsrc),
    .rt_res_dst (mb_rt_rdst), .rt_squant_src (mb_rt_ssrc),
    .rt_kv_user (mb_rt_kvu), .rt_tip_blk (mb_rt_tblk),
    .rt_imp_hi (mb_rt_ihi), .rt_imp_lo (mb_rt_ilo),
    .rt_imp_clear (mb_rt_impclr),
    .ro_valid (mb_ro_v), .ro_ready (mb_ro_r), .ro_beat (mb_ro_b),
    .fs_valid (mb_fs_v), .fs_ready (mb_fs_r), .fs_data (mb_fs_d),
    .fs_last (mb_fs_l),
    .ss_valid (mb_ss_v), .ss_ready (mb_ss_r), .ss_data (mb_ss_d),
    .ss_last (mb_ss_l),
    .td_valid (mb_td_v), .td_ready (mb_td_r), .td_fp16 (mb_td_fp16),
    .td_tier (mb_td_tier), .td_blk (mb_td_blk)
  );

  // ── APEX tile: ship-parameter b64 config, streams host-driven via the
  //    stage-2 mailbox (smoke traffic = none until the host pushes) ────────
  // (b64 CQ-4+ variant: KVQ_OUTLIER_K=2 + KVQ_MASK_FILE — add the hex to the
  //  synth sources when building that config; S12's loadable mask retires
  //  the ROM dependency later.)
  apex_top #(
    .CFG_D         (CL_D),
    .CFG_DM        (CL_DMODEL),   // GAP D model-wide family (default = CFG_D)
    .QSTAGE_H_MAX  (CL_QSTAGE),   // per-head q staging depth (default 1)
    .KVQ_G         (16),   // L3 reference build (-GG_CFG=16)
                           // the trace-replay regops are generated by the
                           // L3 choreography, so the CL must build to its
                           // key-group. (S5 cache-at-rest G=128 is the codec
                           // convention, distinct from the RTL key-group.)
    .KVQ_DEPTH     (CL_KVQ_DEPTH),
    .KVQ_OUTLIER_K (0),
    .KVQ_MASK_FILE (""),
    .FEED_ROWS_MAX (CL_ROWS_MAX),
    .STAGE_R_MAX   (CL_ROWS_MAX),
    .RMS_D_MAX     (CL_DM),
    .LAYER_DM_MAX  (CL_DM),
    .KVQ_GQA_NENG  (CL_GQA),
    .W4_LANE       (CL_W4)      // W4-DEBUT OVERLAY: default 0 == committed CL
  ) u_tile (
    .clk             (clk_tile),                     // DECISION-LC-1
    .rst_n           (rst_tile_n & ~tile_rst_q),
    // tile CSR (pulse protocol)
    .csr_addr        (t_csr_addr),
    .csr_wdata       (t_csr_wdata),
    .csr_write       (t_csr_write),
    .csr_read        (t_csr_read),
    .csr_rdata       (t_csr_rdata),
    .csr_ready       (t_csr_ready),
    // KVQ AXI-Lite
    .kv_awaddr       (t_kv_awaddr),
    .kv_awvalid      (t_kv_awvalid),
    .kv_awready      (t_kv_awready),
    .kv_wdata        (t_kv_wdata),
    .kv_wvalid       (t_kv_wvalid),
    .kv_wready       (t_kv_wready),
    .kv_bresp        (t_kv_bresp),
    .kv_bvalid       (t_kv_bvalid),
    .kv_bready       (t_kv_bready),
    .kv_araddr       (t_kv_araddr),
    .kv_arvalid      (t_kv_arvalid),
    .kv_arready      (t_kv_arready),
    .kv_rdata        (t_kv_rdata),
    .kv_rresp        (t_kv_rresp),
    .kv_rvalid       (t_kv_rvalid),
    .kv_rready       (t_kv_rready),
    .kv_irq          (t_kv_irq),
    .kv_evict_needed (t_kv_evict_needed),
    .kv_evict_addr   (t_kv_evict_addr),
    // streams + jobs: host-driven via the stage-2 mailbox.
    // xw is the ONE muxed stream (IB-FUEL s3): host mailbox path (reset
    // default, byte-identical) vs the DDR fuel path — see the fuel section.
    .ds_valid        (mb_ds_v), .ds_ready (mb_ds_r), .ds_desc (mb_ds_d),
    .xw_valid        (tile_xw_v), .xw_ready (tile_xw_r), .xw_beat (tile_xw_b),
    .xa_valid        (mb_xa_v), .xa_ready (mb_xa_r), .xa_x (mb_xa_x),
    .xa_last         (mb_xa_last),
    .xg_valid        (mb_xg_v), .xg_ready (mb_xg_r), .xg_gamma (mb_xg_g),
    .qs_valid        (mb_qs_v), .qs_ready (mb_qs_r), .qs_data (mb_qs_d),
    .cs_valid        (mb_cs_v), .cs_ready (mb_cs_r), .cs_data (mb_cs_d),
    .fj_valid        (mb_fj_v), .fj_ready (mb_fj_r), .fj_rows (mb_fj_rows),
    .qj_valid        (mb_qj_v), .qj_ready (mb_qj_r), .qj_mode (mb_qj_mode),
    .qj_cols         (mb_qj_cols),
    .dj_valid        (mb_dj_v), .dj_ready (mb_dj_r), .dj_cols (mb_dj_cols),
    .lj_valid        (mb_lj_v), .lj_ready (mb_lj_r), .lj_beats (mb_lj_beats),
    .lj_lanes        (mb_lj_lanes),
    .aj_valid        (mb_aj_v), .aj_ready (mb_aj_r), .aj_op (mb_aj_op),
    .aj_bank         (mb_aj_bank), .aj_pat (mb_aj_pat), .aj_rows (mb_aj_rows),
    .aj_nb           (mb_aj_nb), .aj_sel (mb_aj_sel),
    .wj_valid        (mb_wj_v), .wj_ready (mb_wj_r), .wj_op (mb_wj_op),
    .wj_bank         (mb_wj_bank), .wj_pat (mb_wj_pat), .wj_rows (mb_wj_rows),
    .wj_nb           (mb_wj_nb), .wj_sel (mb_wj_sel),
    // route levels: mailbox registers (reset = all default paths)
    .rt_feeder_src   (mb_rt_fsrc), .rt_feeder_dst (mb_rt_fdst),
    .rt_act_src      (mb_rt_asrc), .rt_wgt_src (mb_rt_wsrc),
    .rt_res_dst      (mb_rt_rdst), .rt_squant_src (mb_rt_ssrc),
    .rt_kv_user      (mb_rt_kvu), .rt_tip_blk (mb_rt_tblk),
    .rt_imp_hi       (mb_rt_ihi), .rt_imp_lo (mb_rt_ilo),
    .rt_imp_clear    (mb_rt_impclr),
    // output taps: captured into mailbox FIFOs
    .fs_valid        (mb_fs_v), .fs_ready (mb_fs_r), .fs_data (mb_fs_d),
    .fs_last         (mb_fs_l),
    .ss_valid        (mb_ss_v), .ss_ready (mb_ss_r), .ss_data (mb_ss_d),
    .ss_last         (mb_ss_l),
    .td_valid        (mb_td_v), .td_ready (mb_td_r), .td_fp16 (mb_td_fp16),
    .td_tier         (mb_td_tier), .td_blk (mb_td_blk),
    .ro_valid        (mb_ro_v), .ro_ready (mb_ro_r), .ro_beat (mb_ro_b),
    // W-G3 flip: walker fetch-request channel (tile domain) -> the fuel
    // controller's walker producer (u_fuel_ctl wk_req_*, IB_FUEL.md §2.4).
    // Same clk_tile domain — no CDC; the 4-phase to the shell-side reader
    // stays inside apex_fuel_ctl exactly as for host-GO records.
    .wf_valid        (t_wf_valid), .wf_ready (t_wf_ready), .wf_req (t_wf_req),
    // done pulses + sticky errors → bridge registers
    .dn_mxe          (t_dn_mxe), .dn_feeder (t_dn_feeder),
    .dn_squant       (t_dn_squant), .dn_scored (t_dn_scored),
    .dn_ser          (t_dn_ser), .dn_astage (t_dn_astage),
    .dn_wstage       (t_dn_wstage), .dn_rms (t_dn_rms), .dn_asu (t_dn_asu),
    .err_sticky      (t_err_sticky),
    // debug taps: unobserved in smoke
    .dbg_f16_v (), .dbg_f16_data (), .dbg_f16_last (),
    .dbg_sc_v  (), .dbg_sc_data  (), .dbg_sc_last  (),
    .dbg_pr_v  (), .dbg_pr_data  (), .dbg_pr_last  ()
  );

  // ── IB-FUEL s2: DDR weight store + PCIS decode (ALL shell domain) ─────────
  // docs/design/IB_FUEL.md §1 s2. DDR at PCIS base 0x0, 64 GiB
  // (cl_dram_hbm_dma decode precedent); sh_ddr on clk_main_a0 exactly as in
  // the kit example. Build knob: +define+APEX_CL_DDR=1 -> DDR_PRESENT=1
  // (the f2sim config; the s5 DCP additionally needs cl_ddr4.xci +
  // `define USE_64GB_DDR_DIMM per Supported_DDR_Modes.md). Default 0 keeps
  // today's DCP config: sh_ddr inert exactly as the old
  // unused_ddr_template.inc instantiated it, PCIS answered exactly as the
  // old unused_dma_pcis_template.inc tied it (apex_pcis_slv idles all-zero
  // while ddr_is_ready=0, which is identical), and the stats bus is
  // ack-high per the kit rule for absent DDR (cl_ports.vh:145-150).
`ifndef APEX_CL_DDR
  `define APEX_CL_DDR 0
`endif
  localparam int unsigned CL_DDR_PRESENT = `APEX_CL_DDR;

  // sh_ddr AXI face (single master today: the PCIS decode; the s3 burst
  // reader inserts its AR/R mux here per IB_FUEL.md §5 R8)
  logic [15:0]  ddr_awid;
  logic [63:0]  ddr_awaddr;
  logic [7:0]   ddr_awlen;
  logic [2:0]   ddr_awsize;
  logic [1:0]   ddr_awburst;
  logic         ddr_awvalid, ddr_awready;
  logic [511:0] ddr_wdata;
  logic [63:0]  ddr_wstrb;
  logic         ddr_wlast, ddr_wvalid, ddr_wready;
  logic [15:0]  ddr_bid;
  logic [1:0]   ddr_bresp;
  logic         ddr_bvalid, ddr_bready;
  logic [15:0]  ddr_arid;
  logic [63:0]  ddr_araddr;
  logic [7:0]   ddr_arlen;
  logic [2:0]   ddr_arsize;
  logic [1:0]   ddr_arburst;
  logic         ddr_arvalid, ddr_arready;
  logic [15:0]  ddr_rid;
  logic [511:0] ddr_rdata;
  logic [1:0]   ddr_rresp;
  logic         ddr_rlast, ddr_rvalid, ddr_rready;

  logic         ddr_stat_ack_w;
  logic [31:0]  ddr_stat_rdata_w;
  logic [7:0]   ddr_stat_int_w;
  logic         ddr_is_ready_w;

  // IB-FUEL s3 legs (declared ahead of first use in u_pcis below):
  // PCIS-side AR/R, reader-side AR/R, afifo faces, ar_owner shell sync
  logic [15:0]  pcis_arid;
  logic [63:0]  pcis_araddr;
  logic [7:0]   pcis_arlen;
  logic [2:0]   pcis_arsize;
  logic [1:0]   pcis_arburst;
  logic         pcis_arvalid, pcis_arready, pcis_rvalid, pcis_rready;
  logic [15:0]  rdr_arid;
  logic [63:0]  rdr_araddr;
  logic [7:0]   rdr_arlen;
  logic [2:0]   rdr_arsize;
  logic [1:0]   rdr_arburst;
  logic         rdr_arvalid, rdr_arready, rdr_rvalid, rdr_rready;
  logic         fifo_wvalid, fifo_wready, fifo_rvalid, fifo_rready;
  logic [63:0]  fifo_wdata, fifo_rdata;
  // 2026-08-11: ASYNC_REG+DONT_TOUCH in SOURCE, not just the xdc. The
  // 2026_08_10-214113 build (one unrelated apex_top edit) shifted synth
  // enough that these two equivalent flops were sequentially merged — the
  // *arown_*_reg* cells vanished and the CDC gate (correctly) aborted the
  // build. Post-synth xdc name-matching cannot protect cells synthesis has
  // already renamed; the attribute pins them in every run.
  (* ASYNC_REG = "TRUE", DONT_TOUCH = "TRUE" *)
  logic         arown_meta, arown_sync;

  sh_ddr #(
    .DDR_PRESENT (CL_DDR_PRESENT)
  ) SH_DDR (
    .clk                   (clk_main_a0),
    .rst_n                 (rst_main_n_sync),
    .stat_clk              (clk_main_a0),
    .stat_rst_n            (rst_main_n_sync),
    .CLK_DIMM_DP           (CLK_DIMM_DP),
    .CLK_DIMM_DN           (CLK_DIMM_DN),
    .M_ACT_N               (M_ACT_N),
    .M_MA                  (M_MA),
    .M_BA                  (M_BA),
    .M_BG                  (M_BG),
    .M_CKE                 (M_CKE),
    .M_ODT                 (M_ODT),
    .M_CS_N                (M_CS_N),
    .M_CLK_DN              (M_CLK_DN),
    .M_CLK_DP              (M_CLK_DP),
    .M_PAR                 (M_PAR),
    .M_DQ                  (M_DQ),
    .M_ECC                 (M_ECC),
    .M_DQS_DP              (M_DQS_DP),
    .M_DQS_DN              (M_DQS_DN),
    .cl_RST_DIMM_N         (RST_DIMM_N),
    .cl_sh_ddr_axi_awid    (ddr_awid),
    .cl_sh_ddr_axi_awaddr  (ddr_awaddr),
    .cl_sh_ddr_axi_awlen   (ddr_awlen),
    .cl_sh_ddr_axi_awsize  (ddr_awsize),
    .cl_sh_ddr_axi_awvalid (ddr_awvalid),
    .cl_sh_ddr_axi_awburst (ddr_awburst),
    .cl_sh_ddr_axi_awuser  (1'b0),
    .cl_sh_ddr_axi_awready (ddr_awready),
    .cl_sh_ddr_axi_wdata   (ddr_wdata),
    .cl_sh_ddr_axi_wstrb   (ddr_wstrb),
    .cl_sh_ddr_axi_wlast   (ddr_wlast),
    .cl_sh_ddr_axi_wvalid  (ddr_wvalid),
    .cl_sh_ddr_axi_wready  (ddr_wready),
    .cl_sh_ddr_axi_bid     (ddr_bid),
    .cl_sh_ddr_axi_bresp   (ddr_bresp),
    .cl_sh_ddr_axi_bvalid  (ddr_bvalid),
    .cl_sh_ddr_axi_bready  (ddr_bready),
    .cl_sh_ddr_axi_arid    (ddr_arid),
    .cl_sh_ddr_axi_araddr  (ddr_araddr),
    .cl_sh_ddr_axi_arlen   (ddr_arlen),
    .cl_sh_ddr_axi_arsize  (ddr_arsize),
    .cl_sh_ddr_axi_arvalid (ddr_arvalid),
    .cl_sh_ddr_axi_arburst (ddr_arburst),
    .cl_sh_ddr_axi_aruser  (1'b0),
    .cl_sh_ddr_axi_arready (ddr_arready),
    .cl_sh_ddr_axi_rid     (ddr_rid),
    .cl_sh_ddr_axi_rdata   (ddr_rdata),
    .cl_sh_ddr_axi_rresp   (ddr_rresp),
    .cl_sh_ddr_axi_rlast   (ddr_rlast),
    .cl_sh_ddr_axi_rvalid  (ddr_rvalid),
    .cl_sh_ddr_axi_rready  (ddr_rready),
    .sh_ddr_stat_bus_addr  (sh_cl_ddr_stat_addr),
    .sh_ddr_stat_bus_wdata (sh_cl_ddr_stat_wdata),
    .sh_ddr_stat_bus_wr    (sh_cl_ddr_stat_wr),
    .sh_ddr_stat_bus_rd    (sh_cl_ddr_stat_rd),
    .sh_ddr_stat_bus_ack   (ddr_stat_ack_w),
    .sh_ddr_stat_bus_rdata (ddr_stat_rdata_w),
    .ddr_sh_stat_int       (ddr_stat_int_w),
    .sh_cl_ddr_is_ready    (ddr_is_ready_w)
  );

  // stats toward the shell: real when DDR present; kit-mandated ack-high
  // tie when absent (constant select — synthesizes to one branch)
  assign cl_sh_ddr_stat_ack   = (CL_DDR_PRESENT != 0) ? ddr_stat_ack_w   : 1'b1;
  assign cl_sh_ddr_stat_rdata = (CL_DDR_PRESENT != 0) ? ddr_stat_rdata_w : '0;
  assign cl_sh_ddr_stat_int   = (CL_DDR_PRESENT != 0) ? ddr_stat_int_w   : '0;

  apex_pcis_slv u_pcis (
    .clk          (clk_main_a0),
    .rst_n        (rst_main_n_sync),
    .ddr_is_ready ((CL_DDR_PRESENT != 0) ? ddr_is_ready_w : 1'b0),
    .rd_block     (arown_sync),
    .s_awid       (sh_cl_dma_pcis_awid),
    .s_awaddr     (sh_cl_dma_pcis_awaddr),
    .s_awlen      (sh_cl_dma_pcis_awlen),
    .s_awsize     (sh_cl_dma_pcis_awsize),
    .s_awburst    (sh_cl_dma_pcis_awburst),
    .s_awvalid    (sh_cl_dma_pcis_awvalid),
    .s_awready    (cl_sh_dma_pcis_awready),
    .s_wdata      (sh_cl_dma_pcis_wdata),
    .s_wstrb      (sh_cl_dma_pcis_wstrb),
    .s_wlast      (sh_cl_dma_pcis_wlast),
    .s_wvalid     (sh_cl_dma_pcis_wvalid),
    .s_wready     (cl_sh_dma_pcis_wready),
    .s_bid        (cl_sh_dma_pcis_bid),
    .s_bresp      (cl_sh_dma_pcis_bresp),
    .s_bvalid     (cl_sh_dma_pcis_bvalid),
    .s_bready     (sh_cl_dma_pcis_bready),
    .s_arid       (sh_cl_dma_pcis_arid),
    .s_araddr     (sh_cl_dma_pcis_araddr),
    .s_arlen      (sh_cl_dma_pcis_arlen),
    .s_arsize     (sh_cl_dma_pcis_arsize),
    .s_arburst    (sh_cl_dma_pcis_arburst),
    .s_arvalid    (sh_cl_dma_pcis_arvalid),
    .s_arready    (cl_sh_dma_pcis_arready),
    .s_rid        (cl_sh_dma_pcis_rid),
    .s_rdata      (cl_sh_dma_pcis_rdata),
    .s_rresp      (cl_sh_dma_pcis_rresp),
    .s_rlast      (cl_sh_dma_pcis_rlast),
    .s_rvalid     (cl_sh_dma_pcis_rvalid),
    .s_rready     (sh_cl_dma_pcis_rready),
    .m_awid       (ddr_awid),
    .m_awaddr     (ddr_awaddr),
    .m_awlen      (ddr_awlen),
    .m_awsize     (ddr_awsize),
    .m_awvalid    (ddr_awvalid),
    .m_awburst    (ddr_awburst),
    .m_awready    (ddr_awready),
    .m_wdata      (ddr_wdata),
    .m_wstrb     (ddr_wstrb),
    .m_wlast      (ddr_wlast),
    .m_wvalid     (ddr_wvalid),
    .m_wready     (ddr_wready),
    .m_bid        (ddr_bid),
    .m_bresp      (ddr_bresp),
    .m_bvalid     (ddr_bvalid),
    .m_bready     (ddr_bready),
    .m_arid       (pcis_arid),
    .m_araddr     (pcis_araddr),
    .m_arlen      (pcis_arlen),
    .m_arsize     (pcis_arsize),
    .m_arvalid    (pcis_arvalid),
    .m_arburst    (pcis_arburst),
    .m_arready    (pcis_arready),
    .m_rid        (ddr_rid),
    .m_rdata      (ddr_rdata),
    .m_rresp      (ddr_rresp),
    .m_rlast      (ddr_rlast),
    .m_rvalid     (pcis_rvalid),
    .m_rready     (pcis_rready)
  );

  // remaining PCIS-adjacent outputs: wr/rd_full were the old template's
  // ties; ruser was left undriven there (implicit 0) and is driven 0 here
  assign cl_sh_dma_pcis_ruser = '0;
  assign cl_sh_dma_wr_full    = 1'b0;
  assign cl_sh_dma_rd_full    = 1'b0;

  // ── IB-FUEL s3: burst reader + async FIFO + fuel_req + xw mux ─────────────
  // (docs/design/IB_FUEL.md §4/§5). Reader + AR mux live in the SHELL
  // domain; the controller (CSRs, skid, 4-phase tile half) lives in the
  // tile domain. Exactly two crossings: the apex_afifo data FIFO and the
  // apex_ocl_cdc-idiom 4-phase; plus 2FF'd single-bit status levels.

  // AR-channel ownership (§9.1 R8, static LOAD/RUN mux): ar_owner is a
  // tile-domain quasi-static level (apex_fuel_ctl rejects changes while
  // busy), 2FF-synced here. LOAD=0 -> PCIS owns sh_ddr AR/R (readback);
  // RUN=1 -> reader owns; PCIS reads answer DECERR via rd_block.
  always_ff @(posedge clk_main_a0) begin
    if (!rst_main_n_sync) begin
      arown_meta <= 1'b0;
      arown_sync <= 1'b0;
    end else begin
      arown_meta <= fuel_arown;
      arown_sync <= arown_meta;
    end
  end

  assign ddr_arid     = arown_sync ? rdr_arid    : pcis_arid;
  assign ddr_araddr   = arown_sync ? rdr_araddr  : pcis_araddr;
  assign ddr_arlen    = arown_sync ? rdr_arlen   : pcis_arlen;
  assign ddr_arsize   = arown_sync ? rdr_arsize  : pcis_arsize;
  assign ddr_arburst  = arown_sync ? rdr_arburst : pcis_arburst;
  assign ddr_arvalid  = arown_sync ? rdr_arvalid : pcis_arvalid;
  assign pcis_arready = !arown_sync && ddr_arready;
  assign rdr_arready  =  arown_sync && ddr_arready;
  assign pcis_rvalid  = !arown_sync && ddr_rvalid;
  assign rdr_rvalid   =  arown_sync && ddr_rvalid;
  assign ddr_rready   = arown_sync ? rdr_rready : pcis_rready;

  // 4-phase + payload between ctl (tile) and reader (shell)
  logic        fuel_req_tgl, fuel_abort_tgl, fuel_ack_tgl;
  logic [29:0] fuel_fr_base;
  logic [25:0] fuel_fr_beats;
  logic        fuel_rdr_busy, fuel_axi_err;

  apex_fuel_reader u_fuel_rdr (
    .clk          (clk_main_a0),
    .rst_n        (rst_main_n_sync),
    .req_tgl_in   (fuel_req_tgl),
    .abort_tgl_in (fuel_abort_tgl),
    .fr_base_64B  (fuel_fr_base),
    .fr_beats_64B (fuel_fr_beats),
    .ack_tgl_out  (fuel_ack_tgl),
    .rdr_busy     (fuel_rdr_busy),
    .axi_err      (fuel_axi_err),
    .m_arid       (rdr_arid),
    .m_araddr     (rdr_araddr),
    .m_arlen      (rdr_arlen),
    .m_arsize     (rdr_arsize),
    .m_arburst    (rdr_arburst),
    .m_arvalid    (rdr_arvalid),
    .m_arready    (rdr_arready),
    .m_rid        (ddr_rid),
    .m_rdata      (ddr_rdata),
    .m_rresp      (ddr_rresp),
    .m_rlast      (ddr_rlast),
    .m_rvalid     (rdr_rvalid),
    .m_rready     (rdr_rready),
    .fw_valid     (fifo_wvalid),
    .fw_data      (fifo_wdata),
    .fw_ready     (fifo_wready)
  );

  // the ONE data crossing: shell -> tile. Reset rule (apex_afifo header):
  // BOTH sides on their domain's GLOBAL reset — never TILE_RST.
  apex_afifo #(.W(64), .LOG2D(9)) u_fuel_fifo (
    .wclk   (clk_main_a0),
    .wrst_n (rst_main_n_sync),
    .wvalid (fifo_wvalid),
    .wready (fifo_wready),
    .wdata  (fifo_wdata),
    .rclk   (clk_tile),
    .rrst_n (rst_tile_n),
    .rvalid (fifo_rvalid),
    .rready (fifo_rready),
    .rdata  (fifo_rdata)
  );

  apex_fuel_ctl u_fuel_ctl (
    .clk           (clk_tile),
    .rst_glb_n     (rst_tile_n),
    .rst_csr_n     (rst_tile_n & ~tile_rst_q),
    .ctrl_wr       (fuel_ctrl_wr),
    .base_wr       (fuel_base_wr),
    .beats_wr      (fuel_beats_wr),
    .err_wr        (fuel_err_wr),
    .wdata         (wdat_q),
    .ctrl_rd       (fuel_ctrl_rd),
    .base_rd       (fuel_base_rd),
    .beats_rd      (fuel_beats_rd),
    .stat_rd       (fuel_stat_rd),
    .err_rd        (fuel_err_rd),
    .host_push_err (fuel_hostpush),
    // walker producer (combine W-G3): the tile walker's fetch records feed
    // the same 2-deep skid as host-CSR GO (IB_FUEL.md §2.4)
    .wk_req_valid  (t_wf_valid),
    .wk_req_ready  (t_wf_ready),
    .wk_req        (t_wf_req),
    .fuel_src      (fuel_src),
    .ar_owner      (fuel_arown),
    .req_tgl_out   (fuel_req_tgl),
    .abort_tgl_out (fuel_abort_tgl),
    .fr_base_64B   (fuel_fr_base),
    .fr_beats_64B  (fuel_fr_beats),
    .ack_tgl_in    (fuel_ack_tgl),
    .rdr_busy_in   (fuel_rdr_busy),
    .ddr_ready_in  ((CL_DDR_PRESENT != 0) ? ddr_is_ready_w : 1'b0),
    .axi_err_in    (fuel_axi_err),
    .f_rvalid      (fifo_rvalid),
    .f_rdata       (fifo_rdata),
    .f_rready      (fifo_rready),
    .fuel_xw_valid (fuel_xw_valid),
    .fuel_xw_data  (fuel_xw_data),
    .fuel_xw_ready (fuel_xw_ready)
  );

  // xw source mux (tile domain): reset default = host mailbox path,
  // byte-identical to the pre-fuel CL
  lane8_beat_t fuel_xw_beat;
  assign fuel_xw_beat.data = fuel_xw_data;
  assign fuel_xw_beat.last = 1'b0;               // WB parity (§2.2)
  assign tile_xw_v     = fuel_src ? fuel_xw_valid : mb_xw_v;
  assign tile_xw_b     = fuel_src ? fuel_xw_beat  : mb_xw_b;
  assign mb_xw_r       = !fuel_src && tile_xw_r;
  assign fuel_xw_ready = fuel_src && tile_xw_r;

endmodule
