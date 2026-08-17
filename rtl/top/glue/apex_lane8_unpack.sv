// apex_lane8_unpack.sv — E-2a glue (E2E_TOY_LANE.md §4): one lane8 beat ->
// MXE_N serial INT8 element beats, in lane order.
//
// WHY IT EXISTS. asu_rmsnorm's x input is a serial INT8 stream framed by
// `last` — before E-2 that stream was a TOP-LEVEL PORT of apex_top
// (xa_valid/xa_x), structurally host-fed. The tile's own C-1 quantizer
// (seam_feeder_quant) produces the codes the norm must consume — golden's
// layer entry is literally `x8, _ = quant_rows_i8(X); rmsnorm_fx(x8, g)`
// (transformer.py:485/:568, the scale is DISCARDED: RMSNorm is
// scale-invariant) — but it emits them packed 8/beat (lane8_beat_t). This
// block unpacks that beat stream so the feeder can drive the norm's x port
// through apex_top's l_nsrc input mux.
//
// ORDER + FRAMING (from the producer, seam_feeder_quant ST_DRAIN): element
// j of each 8-group is packed at data[8*j +: 8], and `last` marks the FINAL
// beat of the JOB (total_beats = rows*D/8). So `last` passes through on the
// final element of a `last` beat, which for a rows=1 job frames exactly one
// C-1 row — the shape asu_rmsnorm expects. A rows>1 job frames rows*D
// elements as ONE stream row, which the norm's own length legality then
// REFUSES loudly (len_error) — no silent wrong answer is reachable through
// this block; it adds no error modes of its own (pure width conversion,
// framing passed through).
//
// §5: both ports skid-buffered (every new glue block owns skids on all its
// ports). busy covers the input skid, the held beat, and the output skid —
// it is the "codes still in flight to the norm" term of apex_top's
// LAYER_STATUS norm-feed busy bit (the poll the host choreography uses to
// know the whole row has reached the norm).

module apex_lane8_unpack
  import apex_pkg::*;
(
  input  logic              clk,
  input  logic              rst_n,           // synchronous, active low

  // lane8 beat stream in (from the C-1 feeder's out port)
  input  logic              s_valid,
  output logic              s_ready,
  input  lane8_beat_t       s_beat,

  // serial INT8 element stream out (to the norm's x port mux)
  output logic              m_valid,
  input  logic              m_ready,
  output logic signed [7:0] m_x,
  output logic              m_last,

  output logic              busy
);

  // ── input skid (§5) ───────────────────────────────────────────────────────
  logic                            i_valid, i_ready;
  logic [$bits(lane8_beat_t)-1:0]  i_vec;
  lane8_beat_t                     i_beat;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (s_valid),
    .s_ready (s_ready),
    .s_data  ({s_beat.data, s_beat.last}),
    .m_valid (i_valid),
    .m_ready (i_ready),
    .m_data  (i_vec)
  );
  assign i_beat = lane8_beat_t'(i_vec);

  // ── serializer: 8 element beats per lane8 beat, lane order ────────────────
  logic [2:0] idx;
  logic       o_valid, o_ready, o_last;
  logic [7:0] o_x;

  assign o_valid = i_valid;
  assign o_x     = i_beat.data[8 * idx +: 8];
  assign o_last  = i_beat.last && (idx == 3'd7);
  assign i_ready = o_valid && o_ready && (idx == 3'd7);

  always_ff @(posedge clk) begin
    if (!rst_n)                   idx <= '0;
    else if (o_valid && o_ready)  idx <= idx + 3'd1;   // wraps 7 -> 0
  end

  // ── output skid (§5) ──────────────────────────────────────────────────────
  logic [8:0] m_bus;

  stream_skid #(.WIDTH(9)) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  ({o_x, o_last}),
    .m_valid (m_valid),
    .m_ready (m_ready),
    .m_data  (m_bus)
  );
  assign m_x    = signed'(m_bus[8:1]);
  assign m_last = m_bus[0];

  assign busy = i_valid || m_valid || (idx != '0);

endmodule
