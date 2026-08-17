// apex_lane32_ser.sv — v0.1 glue: MXE lane32 result beats -> serial INT32.
//
// Purpose (ARCHITECTURE.md §9b / D-021 integration): the MXE result stream
// carries 8 packed INT32 lanes per beat (apex_pkg lane32_beat_t); the
// scale/quant seam consumers (apex_scale_quant: golden stages Q4/S-2, Q7,
// Q11/S-4 of golden/apex_golden/attention.py) consume ONE element per beat.
// This block unpacks a job of `job_beats` beats x `job_lanes` low lanes into
// job_beats*job_lanes serial elements, in beat-major lane-minor order —
// exactly np.ndarray row-major order of the MXE accumulator tile, i.e. the
// element order of golden `acc` arrays. Pure routing: NO numerics.
//
// Framing: out_last marks the final element of THIS job. The input beat
// `last` flags are NOT interpreted (a serializer job may deliberately span
// several M=1 MXE jobs, each of which carries its own last=1 — the
// host-sequenced v0.1 flow does exactly that); element count is the frame.
//
// Job legality (§3 style): 1 <= beats, 1 <= lanes <= MXE_N; violation =>
// job_error pulse (one cycle after the handshake) + sticky, NO state change.
// D-006: done <=> every element ACCEPTED downstream post-skid; job_ready
// gated until strictly after done. busy = (FSM != IDLE) || output pending.
// Both ports skid-buffered (§5, D-019).

module apex_lane32_ser
  import apex_pkg::*;
#(
  parameter int unsigned BEATS_MAX = 255
)(
  input  logic               clk,
  input  logic               rst_n,           // synchronous, active low

  // job interface (D-006)
  input  logic               job_valid,
  output logic               job_ready,
  input  logic [7:0]         job_beats,
  input  logic [3:0]         job_lanes,
  output logic               job_error,
  output logic               job_error_sticky,
  output logic               busy,
  output logic               done,

  // lane32 beat input (MXE res stream shape)
  input  logic               in_valid,
  output logic               in_ready,
  input  lane32_beat_t       in_beat,

  // serial INT32 element output
  output logic               out_valid,
  input  logic               out_ready,
  output logic signed [31:0] out_data,
  output logic               out_last
);

  if (BEATS_MAX < 1 || BEATS_MAX > 255) begin : g_chk_beats
    $error("apex_lane32_ser: BEATS_MAX must be in [1, 255]");
  end

  localparam int unsigned CNT_W = 12;   // elements: <= 255*8 = 2040

  // ── input skid (§5) ───────────────────────────────────────────────────────
  logic                            i_valid, i_ready;
  logic [$bits(lane32_beat_t)-1:0] i_vec;
  lane32_beat_t                    i_beat;

  stream_skid #(.WIDTH($bits(lane32_beat_t))) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (in_valid),
    .s_ready (in_ready),
    .s_data  ({in_beat.data, in_beat.last}),
    .m_valid (i_valid),
    .m_ready (i_ready),
    .m_data  (i_vec)
  );
  assign i_beat = lane32_beat_t'(i_vec);

  // ── output skid ───────────────────────────────────────────────────────────
  logic        o_valid, o_ready;
  logic [32:0] o_data, out_vec;

  stream_skid #(.WIDTH(33)) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  (o_data),
    .m_valid (out_valid),
    .m_ready (out_ready),
    .m_data  (out_vec)
  );
  assign out_data = $signed(out_vec[32:1]);
  assign out_last = out_vec[0];

  // ── FSM ───────────────────────────────────────────────────────────────────
  typedef enum logic [1:0] { ST_IDLE = 2'd0, ST_PROC = 2'd1, ST_WAIT = 2'd2 } state_e;
  state_e           state;
  logic [7:0]       beats_q, beat_no;
  logic [3:0]       lanes_q, lane_no;
  logic             have;
  logic [255:0]     lanes_hold;
  logic [CNT_W-1:0] total_q, out_acc;

  logic legal, legal_beats_hi;
  if (BEATS_MAX < 255) begin : g_beats_bound
    assign legal_beats_hi = (job_beats <= 8'(BEATS_MAX));
  end else begin : g_beats_full        // full 8-bit range: bound is vacuous
    assign legal_beats_hi = 1'b1;
  end
  assign legal = (job_beats != 8'd0) && legal_beats_hi
              && (job_lanes != 4'd0) && (job_lanes <= 4'(MXE_N));

  assign i_ready = (state == ST_PROC) && !have;
  assign o_valid = (state == ST_PROC) && have;
  assign o_data  = {lanes_hold[32*lane_no[2:0] +: 32],
                    (beat_no == beats_q - 8'd1) && (lane_no == lanes_q - 4'd1)};

  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || out_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state            <= ST_IDLE;
      done             <= 1'b0;
      job_error        <= 1'b0;
      job_error_sticky <= 1'b0;
      beats_q          <= '0;
      lanes_q          <= '0;
      beat_no          <= '0;
      lane_no          <= '0;
      have             <= 1'b0;
      lanes_hold       <= '0;
      total_q          <= '0;
      out_acc          <= '0;
    end else begin
      done      <= 1'b0;
      job_error <= 1'b0;
      if (out_valid && out_ready) out_acc <= out_acc + CNT_W'(1);

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state   <= ST_PROC;
              beats_q <= job_beats;
              lanes_q <= job_lanes;
              total_q <= CNT_W'(job_beats) * CNT_W'(job_lanes);
              beat_no <= '0;
              lane_no <= '0;
              have    <= 1'b0;
              out_acc <= '0;
            end else begin                 // §3: pulse + sticky, no effects
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
            end
          end
        end

        ST_PROC: begin
          if (!have) begin
            if (i_valid && i_ready) begin
              lanes_hold <= i_beat.data;
              have       <= 1'b1;
            end
          end else if (o_ready) begin      // element accepted into out skid
            if (lane_no == lanes_q - 4'd1) begin
              have    <= 1'b0;
              lane_no <= '0;
              if (beat_no == beats_q - 8'd1) state <= ST_WAIT;
              else                           beat_no <= beat_no + 8'd1;
            end else begin
              lane_no <= lane_no + 4'd1;
            end
          end
        end

        ST_WAIT: begin                     // D-006: post-skid acceptance
          if (out_acc == total_q) begin
            done  <= 1'b1;
            state <= ST_IDLE;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

  // input beat `last` flags deliberately not interpreted (see header)
  logic unused_ok;
  assign unused_ok = i_beat.last;

endmodule
