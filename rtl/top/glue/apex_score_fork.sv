// apex_score_fork.sv — v0.1 glue: lossless 2-way fork of the score stream.
//
// Implements the "TIP tap" of the ARCHITECTURE.md §0 dataflow: the
// seam_score_dequant output feeds BOTH the ASU online softmax and the TIP
// importance/precision block. Every accepted input beat is delivered to BOTH
// outputs exactly once (join-style: a new beat is accepted only when both
// copies of the previous one have been handed off), so neither consumer can
// ever miss or double-see a score. Pure routing — NO numerics.
//
// §5 discipline: the input goes through a stream_skid (V0.5/D-019); each
// output is a held register (valid asserted until ready — never retracted,
// data stable by construction). Both downstream consumers (asu_softmax,
// tip_top) own input skids, so every boundary crossing is skid-buffered.
// busy = any beat resident anywhere in the fork.

module apex_score_fork (
  input  logic               clk,
  input  logic               rst_n,       // synchronous, active low

  // score input (seam_score_dequant output shape)
  input  logic               s_valid,
  output logic               s_ready,
  input  logic signed [31:0] s_data,
  input  logic               s_last,

  // copy A (ASU softmax s_score/s_last)
  output logic               a_valid,
  input  logic               a_ready,
  output logic signed [31:0] a_data,
  output logic               a_last,

  // copy B (TIP s_data/s_last)
  output logic               b_valid,
  input  logic               b_ready,
  output logic signed [31:0] b_data,
  output logic               b_last,

  output logic               busy
);

  // ── input skid (§5) ───────────────────────────────────────────────────────
  logic        i_valid, i_ready;
  logic [32:0] i_vec;

  stream_skid #(.WIDTH(33)) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (s_valid),
    .s_ready (s_ready),
    .s_data  ({s_data, s_last}),
    .m_valid (i_valid),
    .m_ready (i_ready),
    .m_data  (i_vec)
  );

  // ── hold register + per-branch pending flags ──────────────────────────────
  logic        pend_a, pend_b;
  logic [32:0] hold;

  assign i_ready = !pend_a && !pend_b;

  assign a_valid = pend_a;
  assign b_valid = pend_b;
  assign a_data  = $signed(hold[32:1]);
  assign a_last  = hold[0];
  assign b_data  = $signed(hold[32:1]);
  assign b_last  = hold[0];

  assign busy = pend_a || pend_b || i_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      pend_a <= 1'b0;
      pend_b <= 1'b0;
      hold   <= '0;
    end else begin
      if (i_valid && i_ready) begin
        hold   <= i_vec;
        pend_a <= 1'b1;
        pend_b <= 1'b1;
      end else begin
        if (pend_a && a_ready) pend_a <= 1'b0;
        if (pend_b && b_ready) pend_b <= 1'b0;
      end
    end
  end

endmodule
