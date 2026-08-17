// mxe_top.sv — MXE (Matrix eXecution Engine) top level: ctrl + systolic
// array + activation buffer + requant epilogue + boundary skids.
//
// Implements: ARCHITECTURE.md §3 (MXE), §5 / D-006 (every external stream
//             crosses a V0.5-verified stream_skid; done/busy per contract),
//             D-004/D-005 (array), C-2 (requant epilogue with raw-INT32
//             bypass), D-003 (INT32 lanes out).
//
// External streams (all valid/ready, payloads from apex_pkg):
//   desc  : mxe_desc_t in — simple direct interface (fields are registered
//           by mxe_ctrl on accept, the permitted "reg-stage" for descriptors).
//   act   : lane8_beat_t in  — A tile, row-major/chunk-minor (see mxe_ctrl).
//   wgt   : lane8_beat_t in  — B columns, chunk-major (see mxe_ctrl).
//   res   : lane32_beat_t out — M beats per job, beat m lane c =
//             requant_en ? sign-extended INT8 y[m][c]  (C-2 epilogue)
//                        : raw INT32 accumulator C[m][c];
//           lanes c ≥ N are zero; `last` set on the final beat of the job.
//
// done pulses only after the final result beat is accepted DOWNSTREAM of the
// output skid (res_valid && res_ready observed here), and desc_ready for the
// next job reasserts strictly after done — D-006, closing run1's D-019 wart.

module mxe_top
  import apex_pkg::*;
  import mxe_cfg_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,             // synchronous, active-low

  // descriptor
  input  logic         desc_valid,
  output logic         desc_ready,
  input  mxe_desc_t    desc,
  output logic         desc_error,
  output logic         desc_error_sticky,
  output logic         busy,
  output logic         done,

  // activation stream in
  input  logic         act_valid,
  output logic         act_ready,
  input  lane8_beat_t  act_beat,

  // weight stream in
  input  logic         wgt_valid,
  output logic         wgt_ready,
  input  lane8_beat_t  wgt_beat,

  // result stream out
  output logic         res_valid,
  input  logic         res_ready,
  output lane32_beat_t res_beat
);

  localparam int unsigned L8W  = $bits(lane8_beat_t);
  localparam int unsigned L32W = $bits(lane32_beat_t);

  // ── input skids (D-006 / §5: skid on every external stream) ──────────────
  logic        act_i_valid, act_i_ready;
  logic [L8W-1:0] act_i_data;
  lane8_beat_t act_i_beat;

  stream_skid #(.WIDTH(L8W)) u_skid_act (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (act_valid),
    .s_ready (act_ready),
    .s_data  (L8W'(act_beat)),
    .m_valid (act_i_valid),
    .m_ready (act_i_ready),
    .m_data  (act_i_data)
  );
  assign act_i_beat = lane8_beat_t'(act_i_data);

  logic        wgt_i_valid, wgt_i_ready;
  logic [L8W-1:0] wgt_i_data;
  lane8_beat_t wgt_i_beat;

  stream_skid #(.WIDTH(L8W)) u_skid_wgt (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (wgt_valid),
    .s_ready (wgt_ready),
    .s_data  (L8W'(wgt_beat)),
    .m_valid (wgt_i_valid),
    .m_ready (wgt_i_ready),
    .m_data  (wgt_i_data)
  );
  assign wgt_i_beat = lane8_beat_t'(wgt_i_data);

  // ── activation buffer ─────────────────────────────────────────────────────
  logic               abuf_we;
  logic [ABUF_AW-1:0] abuf_waddr, abuf_raddr;
  logic [8*MXE_N-1:0] abuf_wdata, abuf_rdata;

  mxe_buf #(.WIDTH(8*MXE_N), .DEPTH(ABUF_DEPTH)) u_abuf (
    .clk   (clk),
    .we    (abuf_we),
    .waddr (abuf_waddr),
    .wdata (abuf_wdata),
    .raddr (abuf_raddr),
    .rdata (abuf_rdata)
  );

  // ── systolic array ────────────────────────────────────────────────────────
  logic               arr_en, arr_beat_valid, arr_beat_clear;
  logic [8*MXE_N-1:0] arr_act_lanes, arr_wload_lanes;
  logic [ROW_W-1:0]   arr_beat_row, arr_rd_row;
  logic               arr_live_sel, arr_wload_en;
  acc_t               arr_rd_lanes [MXE_N];

  mxe_array u_array (
    .clk         (clk),
    .rst_n       (rst_n),
    .en          (arr_en),
    .act_lanes   (arr_act_lanes),
    .beat_valid  (arr_beat_valid),
    .beat_clear  (arr_beat_clear),
    .beat_row    (arr_beat_row),
    .live_sel    (arr_live_sel),
    .wload_en    (arr_wload_en),
    .wload_lanes (arr_wload_lanes),
    .rd_row      (arr_rd_row),
    .rd_lanes    (arr_rd_lanes)
  );

  // ── controller ────────────────────────────────────────────────────────────
  logic        r_valid, r_ready, r_last, res_accepted;
  logic        requant_en_q;
  logic [15:0] rq_scale_q;
  logic [4:0]  rq_shift_q;
  logic [3:0]  n_active;

  mxe_ctrl u_ctrl (
    .clk               (clk),
    .rst_n             (rst_n),
    .desc_valid        (desc_valid),
    .desc_ready        (desc_ready),
    .desc              (desc),
    .desc_error        (desc_error),
    .desc_error_sticky (desc_error_sticky),
    .busy              (busy),
    .done              (done),
    .w_valid           (wgt_i_valid),
    .w_ready           (wgt_i_ready),
    .w_beat            (wgt_i_beat),
    .a_valid           (act_i_valid),
    .a_ready           (act_i_ready),
    .a_beat            (act_i_beat),
    .abuf_we           (abuf_we),
    .abuf_waddr        (abuf_waddr),
    .abuf_wdata        (abuf_wdata),
    .abuf_raddr        (abuf_raddr),
    .abuf_rdata        (abuf_rdata),
    .arr_en            (arr_en),
    .arr_act_lanes     (arr_act_lanes),
    .arr_beat_valid    (arr_beat_valid),
    .arr_beat_clear    (arr_beat_clear),
    .arr_beat_row      (arr_beat_row),
    .arr_live_sel      (arr_live_sel),
    .arr_wload_en      (arr_wload_en),
    .arr_wload_lanes   (arr_wload_lanes),
    .arr_rd_row        (arr_rd_row),
    .r_valid           (r_valid),
    .r_ready           (r_ready),
    .r_last            (r_last),
    .res_accepted      (res_accepted),
    .requant_en_q      (requant_en_q),
    .rq_scale_q        (rq_scale_q),
    .rq_shift_q        (rq_shift_q),
    .n_active          (n_active)
  );

  // ── drain data path: lane mask → requant/bypass → output skid ────────────
  acc_t  raw_lanes [MXE_N];
  out8_t rq_lanes  [MXE_N];

  always_comb begin
    for (int c = 0; c < int'(MXE_N); c++)
      raw_lanes[c] = (c < int'({28'd0, n_active})) ? arr_rd_lanes[c]
                                                   : acc_t'(0);
  end

  mxe_requant u_requant (
    .in_lanes  (raw_lanes),
    .rq_scale  (rq_scale_q),
    .rq_shift  (rq_shift_q),
    .out_lanes (rq_lanes)
  );

  lane32_beat_t r_beat;
  always_comb begin
    r_beat.last = r_last;
    for (int c = 0; c < int'(MXE_N); c++)
      r_beat.data[32*c +: 32] = requant_en_q
        ? {{24{rq_lanes[c][7]}}, rq_lanes[c]}    // sign-extended INT8
        : raw_lanes[c];
  end

  // ── output skid + post-skid acceptance tap (D-006) ────────────────────────
  logic [L32W-1:0] res_o_data;

  stream_skid #(.WIDTH(L32W)) u_skid_res (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (r_valid),
    .s_ready (r_ready),
    .s_data  (L32W'(r_beat)),
    .m_valid (res_valid),
    .m_ready (res_ready),
    .m_data  (res_o_data)
  );
  assign res_beat     = lane32_beat_t'(res_o_data);
  assign res_accepted = res_valid && res_ready;

endmodule
