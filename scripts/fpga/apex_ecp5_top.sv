// =============================================================================
// scripts/fpga/apex_ecp5_top.sv — THIN FPGA pin wrapper for ECP5 P&R
//
// WHY: apex_top's raw port list is ~1000 bits (lane32_beat_t result stream
// alone is 257b) — far beyond the ~200 usable I/O of LFE5U-85F CABGA381.
// nextpnr cannot place it bare. This wrapper reduces the tile to 5 pins so
// the CORE logic is fully placed-and-routed and a real core Fmax is reported.
//
// WHAT IT DOES (test-harness idiom, core UNCHANGED):
//   * clk, rst_n, si (serial stimulus in), se (shift enable), so (serial out)
//   * ALL apex_top inputs are driven from taps of one long input shift
//     register (IN_BITS flops) fed by si — every core input remains a live,
//     unconstrainable-to-constant signal, so yosys cannot const-fold the core.
//   * ALL apex_top outputs are XOR-compressed through a 2-stage pipelined
//     reduction tree into so — every core output stays observable, so nothing
//     is swept; the 2-stage pipeline keeps the wrapper off the critical path.
//   * apex_top is instantiated with DEFAULT parameters, port-for-port,
//     zero modifications. (* keep *) guards the instance.
//
// This is a P&R feasibility/timing harness, NOT a functional board interface.
// =============================================================================
`default_nettype none

module apex_ecp5_top
  import apex_pkg::*;
(
  input  wire clk,
  input  wire rst_n,
  input  wire si,      // serial stimulus in
  input  wire se,      // shift enable
  output logic so      // XOR-compressed observation out
);

  // ── core parameter mirrors (defaults of apex_top) ─────────────────────────
  localparam int unsigned CFG_D      = 64;
  localparam int unsigned KVQ_DEPTH  = 128;
  localparam int unsigned STAGE_NB_W = $clog2(CFG_D/8) + 1;      // = 4
  localparam int unsigned EV_W       = $clog2(KVQ_DEPTH);        // = 7

  // ── input-side widths (must sum to IN_BITS; kept explicit for audit) ─────
  localparam int unsigned W_CSR  = 8+32+1+1;                     // 42
  localparam int unsigned W_KVI  = 8+1+32+1+1+8+1+1;             // 53
  localparam int unsigned W_DS   = 1 + $bits(mxe_desc_t);        // 129
  localparam int unsigned W_XW   = 1 + $bits(lane8_beat_t);      // 66
  localparam int unsigned W_XA   = 1+8+1;                        // 10
  localparam int unsigned W_XG   = 1+16;                         // 17
  localparam int unsigned W_QS   = 1+32;                         // 33
  localparam int unsigned W_CS   = 1+32;                         // 33
  localparam int unsigned W_FJ   = 1+DIM_W;                      // 13
  localparam int unsigned W_QJ   = 1+1+DIM_W;                    // 14
  localparam int unsigned W_DJ   = 1+DIM_W;                      // 13
  localparam int unsigned W_LJ   = 1+8+4;                        // 13
  localparam int unsigned W_AJ   = 1+1+1+2+5+STAGE_NB_W+5;       // 19
  localparam int unsigned W_WJ   = W_AJ;                         // 19
  localparam int unsigned W_RT   = 1+1+1+1+2+1+1+7+16+16+1;      // 48
  localparam int unsigned W_RDY  = 5;                // fs/ss/td/ro/wf ready
  localparam int unsigned IN_BITS = W_CSR+W_KVI+W_DS+W_XW+W_XA+W_XG+W_QS+W_CS
                                  + W_FJ+W_QJ+W_DJ+W_LJ+W_AJ+W_WJ+W_RT+W_RDY;

  // ── input shift register: si -> every core input ──────────────────────────
  logic [IN_BITS-1:0] sr_in;
  always_ff @(posedge clk) begin
    if (!rst_n)      sr_in <= '0;
    else if (se)     sr_in <= {sr_in[IN_BITS-2:0], si};
  end

  // ── core input signals, mapped as one big concatenation of SR taps ───────
  logic [7:0]   csr_addr;   logic [31:0] csr_wdata;  logic csr_write, csr_read;
  logic [7:0]   kv_awaddr;  logic kv_awvalid;        logic [31:0] kv_wdata;
  logic         kv_wvalid, kv_bready;                logic [7:0]  kv_araddr;
  logic         kv_arvalid, kv_rready;
  logic         ds_valid;   mxe_desc_t   ds_desc;
  logic         xw_valid;   lane8_beat_t xw_beat;
  logic         xa_valid;   logic signed [7:0]  xa_x;     logic xa_last;
  logic         xg_valid;   logic signed [15:0] xg_gamma;
  logic         qs_valid;   logic [31:0] qs_data;
  logic         cs_valid;   logic [31:0] cs_data;
  logic         fj_valid;   logic [DIM_W-1:0] fj_rows;
  logic         qj_valid, qj_mode;  logic [DIM_W-1:0] qj_cols;
  logic         dj_valid;   logic [DIM_W-1:0] dj_cols;
  logic         lj_valid;   logic [7:0] lj_beats;  logic [3:0] lj_lanes;
  logic         aj_valid, aj_op, aj_bank;  logic [1:0] aj_pat;
  logic [4:0]   aj_rows;    logic [STAGE_NB_W-1:0] aj_nb;  logic [4:0] aj_sel;
  logic         wj_valid, wj_op, wj_bank;  logic [1:0] wj_pat;
  logic [4:0]   wj_rows;    logic [STAGE_NB_W-1:0] wj_nb;  logic [4:0] wj_sel;
  logic         rt_feeder_src, rt_feeder_dst, rt_act_src, rt_wgt_src;
  logic [1:0]   rt_res_dst;  logic rt_squant_src, rt_kv_user;
  logic [6:0]   rt_tip_blk;  logic [15:0] rt_imp_hi, rt_imp_lo;
  logic         rt_imp_clear;
  logic         fs_ready, ss_ready, td_ready, ro_ready, wf_ready;

  assign {csr_addr, csr_wdata, csr_write, csr_read,
          kv_awaddr, kv_awvalid, kv_wdata, kv_wvalid, kv_bready,
          kv_araddr, kv_arvalid, kv_rready,
          ds_valid, ds_desc,
          xw_valid, xw_beat,
          xa_valid, xa_x, xa_last,
          xg_valid, xg_gamma,
          qs_valid, qs_data,
          cs_valid, cs_data,
          fj_valid, fj_rows,
          qj_valid, qj_mode, qj_cols,
          dj_valid, dj_cols,
          lj_valid, lj_beats, lj_lanes,
          aj_valid, aj_op, aj_bank, aj_pat, aj_rows, aj_nb, aj_sel,
          wj_valid, wj_op, wj_bank, wj_pat, wj_rows, wj_nb, wj_sel,
          rt_feeder_src, rt_feeder_dst, rt_act_src, rt_wgt_src, rt_res_dst,
          rt_squant_src, rt_kv_user, rt_tip_blk, rt_imp_hi, rt_imp_lo,
          rt_imp_clear,
          fs_ready, ss_ready, td_ready, ro_ready, wf_ready} = sr_in;

  // ── core output signals ───────────────────────────────────────────────────
  logic [31:0]  csr_rdata;  logic csr_ready;
  logic         kv_awready, kv_wready;  logic [1:0] kv_bresp;  logic kv_bvalid;
  logic         kv_arready; logic [31:0] kv_rdata;  logic [1:0] kv_rresp;
  logic         kv_rvalid, kv_irq, kv_evict_needed;
  logic [EV_W-1:0] kv_evict_addr;
  logic         ds_ready, xw_ready, xa_ready, xg_ready, qs_ready, cs_ready;
  logic         fj_ready, qj_ready, dj_ready, lj_ready, aj_ready, wj_ready;
  logic         fs_valid, fs_last;  logic [15:0] fs_data;
  logic         ss_valid, ss_last;  logic [15:0] ss_data;
  logic         td_valid, td_fp16;  kvq_tier_e td_tier;  logic [6:0] td_blk;
  logic         ro_valid;  lane32_beat_t ro_beat;
  logic         wf_valid;  logic [63:0] wf_req;    // W-G3 walker fetch out
  logic         dn_mxe, dn_feeder, dn_squant, dn_scored, dn_ser;
  logic         dn_astage, dn_wstage, dn_rms, dn_asu;
  logic [15:0]  err_sticky;
  logic         dbg_f16_v, dbg_f16_last;  logic [15:0] dbg_f16_data;
  logic         dbg_sc_v,  dbg_sc_last;   logic [31:0] dbg_sc_data;
  logic         dbg_pr_v,  dbg_pr_last;   logic [15:0] dbg_pr_data;

  // ── the tile, UNCHANGED, default parameters ───────────────────────────────
  (* keep *)
  apex_top u_core (
    .clk, .rst_n,
    .csr_addr, .csr_wdata, .csr_write, .csr_read, .csr_rdata, .csr_ready,
    .kv_awaddr, .kv_awvalid, .kv_awready, .kv_wdata, .kv_wvalid, .kv_wready,
    .kv_bresp, .kv_bvalid, .kv_bready, .kv_araddr, .kv_arvalid, .kv_arready,
    .kv_rdata, .kv_rresp, .kv_rvalid, .kv_rready,
    .kv_irq, .kv_evict_needed, .kv_evict_addr,
    .ds_valid, .ds_ready, .ds_desc,
    .xw_valid, .xw_ready, .xw_beat,
    .xa_valid, .xa_ready, .xa_x, .xa_last,
    .xg_valid, .xg_ready, .xg_gamma,
    .qs_valid, .qs_ready, .qs_data,
    .cs_valid, .cs_ready, .cs_data,
    .fj_valid, .fj_ready, .fj_rows,
    .qj_valid, .qj_ready, .qj_mode, .qj_cols,
    .dj_valid, .dj_ready, .dj_cols,
    .lj_valid, .lj_ready, .lj_beats, .lj_lanes,
    .aj_valid, .aj_ready, .aj_op, .aj_bank, .aj_pat, .aj_rows, .aj_nb, .aj_sel,
    .wj_valid, .wj_ready, .wj_op, .wj_bank, .wj_pat, .wj_rows, .wj_nb, .wj_sel,
    .rt_feeder_src, .rt_feeder_dst, .rt_act_src, .rt_wgt_src, .rt_res_dst,
    .rt_squant_src, .rt_kv_user, .rt_tip_blk, .rt_imp_hi, .rt_imp_lo,
    .rt_imp_clear,
    .fs_valid, .fs_ready, .fs_data, .fs_last,
    .ss_valid, .ss_ready, .ss_data, .ss_last,
    .td_valid, .td_ready, .td_fp16, .td_tier, .td_blk,
    .ro_valid, .ro_ready, .ro_beat,
    .wf_valid, .wf_ready, .wf_req,
    .dn_mxe, .dn_feeder, .dn_squant, .dn_scored, .dn_ser,
    .dn_astage, .dn_wstage, .dn_rms, .dn_asu,
    .err_sticky,
    .dbg_f16_v, .dbg_f16_data, .dbg_f16_last,
    .dbg_sc_v,  .dbg_sc_data,  .dbg_sc_last,
    .dbg_pr_v,  .dbg_pr_data,  .dbg_pr_last
  );

  // ── output compression: 2-stage pipelined XOR reduction -> so ────────────
  // Stage 1: register 64b-chunk XORs; stage 2: XOR the chunk flops into so.
  // Keeps the observation tree at ~6 LUT levels per stage — off the core's
  // critical path — while making EVERY core output bit observable.
  localparam int unsigned OUT_BITS =
      32+1                            // csr
    + 1+1+2+1+1+32+2+1+1+1+EV_W       // kv window + irq/evict
    + 6 + 6                           // stream/job readies
    + 1+16+1 + 1+16+1                 // fs, ss scale taps
    + 1+1+2+7                         // td
    + 1+$bits(lane32_beat_t)          // ro
    + 1+64                            // wf (W-G3 walker fetch record)
    + 9 + 16                          // done pulses + err_sticky
    + 1+16+1 + 1+32+1 + 1+16+1;       // dbg taps

  logic [OUT_BITS-1:0] outs;
  assign outs = {csr_rdata, csr_ready,
                 kv_awready, kv_wready, kv_bresp, kv_bvalid, kv_arready,
                 kv_rdata, kv_rresp, kv_rvalid, kv_irq, kv_evict_needed,
                 kv_evict_addr,
                 ds_ready, xw_ready, xa_ready, xg_ready, qs_ready, cs_ready,
                 fj_ready, qj_ready, dj_ready, lj_ready, aj_ready, wj_ready,
                 fs_valid, fs_data, fs_last,
                 ss_valid, ss_data, ss_last,
                 td_valid, td_fp16, td_tier, td_blk,
                 ro_valid, ro_beat,
                 wf_valid, wf_req,
                 dn_mxe, dn_feeder, dn_squant, dn_scored, dn_ser,
                 dn_astage, dn_wstage, dn_rms, dn_asu,
                 err_sticky,
                 dbg_f16_v, dbg_f16_data, dbg_f16_last,
                 dbg_sc_v,  dbg_sc_data,  dbg_sc_last,
                 dbg_pr_v,  dbg_pr_data,  dbg_pr_last};

  localparam int unsigned NCHUNK = (OUT_BITS + 63) / 64;
  logic [NCHUNK-1:0] chunk_x;
  always_ff @(posedge clk) begin
    for (int unsigned c = 0; c < NCHUNK; c++) begin
      logic x;
      x = 1'b0;
      for (int unsigned b = 0; b < 64; b++)
        if (c*64 + b < OUT_BITS) x = x ^ outs[c*64 + b];
      chunk_x[c] <= x;
    end
    so <= ^chunk_x;
  end

endmodule

`default_nettype wire
