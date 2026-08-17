// mxe_wfeed_w4.sv — B3 native-W4 weight feeder (C-4 dequant-on-feeder).
//
// Implements: ARCHITECTURE.md C-4 ("the MXE array sees exactly one dtype
//             (INT8). W4 weights, when adopted (v1.1), unpack+dequant at the
//             weight feeder, never inside PEs"), §5 stream contract,
//             D-006 job semantics, §3-style conservative legality reject.
// Golden arbiter: golden/apex_golden/weight_codec.py — wfeed_w4_to_i8 with
//             realization "A", and the S1-S6 stream contract stated there.
//             This module is bit-exact against it (TB: verif/mxe/w4).
//
// REALIZATION (A), resolved by measurement in B3 stage 0 (see
// docs/design/B3_WEIGHT_PATH.md §2 and docs/results/b3_w4_golden/RESULT.md):
// the feeder is a PURE UNPACK. Output = the INT4 code sign-extended to INT8;
// the group's fp16 scale rides downstream as a factor of the composite,
// exactly where the per-tensor s_w rides today. Consequences, all load-bearing
// for this file being as small as it is:
//   - NO fp16 arithmetic here, and NO scale sideband. The scale never enters
//     the datapath; the host folds it into the composite (which it must round
//     to fp16 grade — apex_scale_quant.sv:29-31,227 C2 — see the contract).
//   - The group MUST be tile-wide (one scale per job weight tile). A scale
//     leaves the contraction acc[m][c] = SUM_k A[m][k]*B[k][c] as a single
//     downstream factor only if it is constant over BOTH k and c.
//   - Realization (B) (dequant->requant, the seam_feeder_quant chain) is the
//     documented fallback for a future finer geometry. It was measured within
//     25% of (A) on every distribution, so it buys no accuracy here.
//
// ── Stream contract (weight_codec.py S1-S6, normative) ─────────────────────
//   The MXE loader consumes KB*N INT8 beats per job, KB = ceil(K/8), beat
//   n = p*N + c, lane r = B[8p+r][c] (mxe_ctrl.sv:46-59, :206-213). Flat
//   element index e = 8n + r. A packed-W4 beat carries flat elements
//   16j..16j+15 as cq_codec.pack_int4 nibbles: element e at bits [4e +: 4],
//   two's complement. So the two INT8 beats a packed beat expands to are
//       half 0: lane r = pw.data[4*r      +: 4]   (bytes 0..3)
//       half 1: lane r = pw.data[4*r + 32 +: 4]   (bytes 4..7)
//   each sign-extended to 8 bits.
//
// ── job_beats counts EMITTED INT8 beats (= KB*N) ───────────────────────────
// The design contract left the unit ambiguous. It is pinned here to the
// OUTPUT count because that is the quantity the downstream loader is framed
// on, and because it makes the odd tail representable: the feeder consumes
// ceil(job_beats/2) packed beats, and when job_beats is ODD the final packed
// beat's upper half is padding that is dropped, never emitted. Hence
//     emitted == 2*consumed - (job_beats & 1)
// and the unqualified "emitted == 2*consumed" is FALSE for legal odd job_beats
// (N is legal 1..8, so KB*N odd is legal). In passthrough mode
// (w4_en = 0) consumed == emitted == job_beats.
//
// ── Mode ───────────────────────────────────────────────────────────────────
// w4_en is a route/CSR-level select, NOT a descriptor field — that is what
// keeps rtl/apex_pkg.sv frozen (APEX_VERSION 0x0001_0000). It is sampled at
// job accept and held for the job's duration, so a mid-job route change
// cannot tear a stream. w4_en = 0 makes this module a transparent 1:1
// passthrough, so it can sit unconditionally on the weight path and the top
// level needs only the route bit, not a datapath mux.
//
// Job legality (§3-style): 1 <= job_beats <= BEATS_MAX, else job_error pulse
// (one cycle) + sticky, ZERO state change, no busy/done, stream untouched.
//
// D-006: done <=> every emitted beat of the job has been ACCEPTED downstream
// (post-skid). job_ready is gated until strictly after done.
// busy = (FSM != IDLE) || beats pending in the output skid.

module mxe_wfeed_w4
  import apex_pkg::*;
#(
  // KB <= 256 (K <= 2048, D-003) x N <= 8 (MXE_N) = 2048 beats per job.
  parameter int unsigned BEATS_MAX = 2048
)(
  input  logic              clk,
  input  logic              rst_n,           // synchronous, active low

  // mode select (route/CSR level; TB-driven at unit level)
  input  logic              w4_en,

  // job interface (D-006)
  input  logic              job_valid,
  output logic              job_ready,
  input  logic [DIM_W-1:0]  job_beats,       // EMITTED INT8 beats (see above)
  output logic              job_error,       // 1-cycle pulse on illegal job
  output logic              job_error_sticky,// cleared only by reset
  output logic              busy,
  output logic              done,            // 1-cycle pulse, D-006

  // packed-W4 stream in (or INT8 passthrough when w4_en = 0)
  input  logic              pw_valid,
  output logic              pw_ready,
  input  lane8_beat_t       pw_beat,

  // INT8 weight stream out (-> mxe_top.wgt)
  output logic              w8_valid,
  input  logic              w8_ready,
  output lane8_beat_t       w8_beat
);

  // ── parameter legality (elaboration-time) ─────────────────────────────────
  if (BEATS_MAX == 0 || BEATS_MAX > 4095) begin : g_chk_beats
    $error("mxe_wfeed_w4: BEATS_MAX must be in [1, 4095] (DIM_W field)");
  end

  localparam int unsigned CNT_W = DIM_W + 1;   // beat counters, 0..BEATS_MAX

  // ── input skid (§5: every boundary crossing) ──────────────────────────────
  logic                           i_valid, i_ready;
  logic [$bits(lane8_beat_t)-1:0] i_vec;
  lane8_beat_t                    i_beat;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (pw_valid),
    .s_ready (pw_ready),
    .s_data  ({pw_beat.data, pw_beat.last}),
    .m_valid (i_valid),
    .m_ready (i_ready),
    .m_data  (i_vec)
  );
  assign i_beat = lane8_beat_t'(i_vec);

  // ── output skid ───────────────────────────────────────────────────────────
  logic                           o_valid, o_ready;
  lane8_beat_t                    o_beat;
  logic [$bits(lane8_beat_t)-1:0] o_vec;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  ({o_beat.data, o_beat.last}),
    .m_valid (w8_valid),
    .m_ready (w8_ready),
    .m_data  (o_vec)
  );
  assign w8_beat = lane8_beat_t'(o_vec);

  // ── the W4 unpack: nibble -> sign-extended INT8 lane (S4) ────────────────
  // half h, lane r  <-  flat element e = 8h + r  <-  bits [4e +: 4].
  // Two's-complement sign extension is the same rule as cq_codec.unpack_int4
  // (nib >= 8 ? nib - 16 : nib) and the same primitive family as the KV INT4
  // dequant (cq_fp_pkg::dequant_one via cq_dequant_unit).
  function automatic logic [8*MXE_N-1:0] unpack_half(
      input logic [8*MXE_N-1:0] packed_in, input logic half);
    logic [3:0] nib;
    for (int r = 0; r < int'(MXE_N); r++) begin
      nib = packed_in[4*(8*int'(half) + r) +: 4];
      unpack_half[8*r +: 8] = {{4{nib[3]}}, nib};      // sign-extend
    end
  endfunction

  // ── FSM / datapath state ─────────────────────────────────────────────────
  typedef enum logic [1:0] {
    ST_IDLE = 2'd0,
    ST_RUN  = 2'd1,   // consume packed beats, emit INT8 beats
    ST_WAIT = 2'd2    // D-006: wait for post-skid acceptance
  } state_e;

  state_e             state;
  logic               mode_q;        // w4_en latched at job accept
  logic [CNT_W-1:0]   total_q;       // emitted INT8 beats this job
  logic [CNT_W-1:0]   need_q;        // packed beats this job consumes
  logic [CNT_W-1:0]   sent_cnt;      // beats pushed into the output skid
  logic [CNT_W-1:0]   recv_cnt;      // packed beats accepted from the skid
  logic [CNT_W-1:0]   out_acc;       // beats accepted POST-skid (D-006)
  logic [8*MXE_N-1:0] hold;          // captured packed beat
  logic               hold_valid;
  logic               half;          // which half of `hold` is next out

  logic legal;
  assign legal = (job_beats != '0) && (job_beats <= DIM_W'(BEATS_MAX));

  // packed beats needed = ceil(total/2) unpacking, total passing through.
  // This is the S6 relation emitted == 2*consumed - (total & 1), expressed
  // on the consuming side so the odd tail costs no extra input beat.
  logic [CNT_W-1:0] need_new;
  assign need_new = w4_en ? ((CNT_W'(job_beats) + CNT_W'(1)) >> 1)
                          : CNT_W'(job_beats);

  // last emitted beat of the job also drops an odd tail's padding half
  logic last_beat;
  assign last_beat = (sent_cnt == total_q - CNT_W'(1));

  // the held beat is fully consumed this cycle: passthrough, or the high
  // half went out, or it is the job's final beat (odd tail -> padding half
  // is dropped, never emitted)
  //
  // REACHABILITY (mutation gate, seam_feeder_quant P5 precedent): the
  // `|| last_beat` disjunct is DEFENSIVE and port-unobservable. It is the
  // only term covering (mode_q && !half && last_beat) — the odd tail's final
  // beat — but on that same cycle the emit block takes `if (last_beat)
  // state <= ST_WAIT`, so `half` never advances, and o_valid is gated on
  // ST_RUN; the stale padding half therefore cannot be emitted with or
  // without this term, and the next job accept re-clears hold_valid/half
  // unconditionally. Dropping it only delays clearing two flops by a cycle.
  // Verified: a mutant with the term removed passes all four regimes (300
  // legal jobs, 61 of them odd-tail). It is kept because it states the
  // intent, and it is NOT claimed as a killable mutant — the mutation gate
  // targets `need_new`'s ceil instead, which IS load-bearing for odd tails.
  logic hold_retire;
  assign hold_retire = o_valid && o_ready && (!mode_q || half || last_beat);

  // handshakes. Accepting while retiring keeps the loader fed back-to-back:
  // without it capture and emit would be mutually exclusive and the feeder
  // would deliver at most 2 beats per 3 cycles, below the 1 beat/cycle the
  // MXE weight loader can take.
  assign i_ready   = (state == ST_RUN) && (recv_cnt < need_q)
                     && (!hold_valid || hold_retire);
  assign o_valid   = (state == ST_RUN) && hold_valid;
  assign o_beat.data = mode_q ? unpack_half(hold, half) : hold;
  assign o_beat.last = last_beat;

  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || w8_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state            <= ST_IDLE;
      done             <= 1'b0;
      job_error        <= 1'b0;
      job_error_sticky <= 1'b0;
      mode_q           <= 1'b0;
      total_q          <= '0;
      need_q           <= '0;
      sent_cnt         <= '0;
      recv_cnt         <= '0;
      out_acc          <= '0;
      hold             <= '0;
      hold_valid       <= 1'b0;
      half             <= 1'b0;
    end else begin
      done      <= 1'b0;
      job_error <= 1'b0;

      // post-skid acceptance counter (any state; cleared on job accept)
      if (w8_valid && w8_ready) out_acc <= out_acc + CNT_W'(1);

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state      <= ST_RUN;
              mode_q     <= w4_en;         // held for the job (no mid-job tear)
              total_q    <= CNT_W'(job_beats);
              need_q     <= need_new;
              sent_cnt   <= '0;
              recv_cnt   <= '0;
              out_acc    <= '0;
              hold_valid <= 1'b0;
              half       <= 1'b0;
            end else begin                 // §3: pulse + sticky, no effects
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
            end
          end
        end

        ST_RUN: begin
          // emit first; the capture below deliberately overrides hold_valid
          // and half when both fire in the same cycle (retire + refill).
          if (o_valid && o_ready) begin
            sent_cnt <= sent_cnt + CNT_W'(1);
            if (last_beat)            state <= ST_WAIT;
            else if (mode_q && !half) half  <= 1'b1;   // high half next
          end

          if (i_valid && i_ready) begin
            hold       <= i_beat.data;
            hold_valid <= 1'b1;
            recv_cnt   <= recv_cnt + CNT_W'(1);
            half       <= 1'b0;
          end else if (hold_retire) begin
            hold_valid <= 1'b0;
            half       <= 1'b0;
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

  // informational input field, consumed for lint
  logic unused_fields;
  assign unused_fields = &{1'b0, i_beat.last};

endmodule
