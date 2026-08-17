// asu_softmax.sv — ASU streaming ONLINE softmax over one row of INT32 scores.
//
// Implements ARCHITECTURE.md §6 (online softmax, FA-3 running-max + rescaled
// running sum), C-5 (consumes INT32 scores directly), D-014 (exp via
// asu_exp_lut), §5/D-006 stream & job semantics. Bit-exact mirror of the
// golden arbiter apex_golden.compute.online_softmax_fx(scores, SCORE_FRAC),
// including _clamp_z, the running-max rescale of l, and the final
// (e << 15) // l floor division.
//
// Dataflow (two-pass emission, exactly the golden model):
//   PASS 1 (streaming, 1 beat/cycle): for each accepted score x_i
//     first beat : m = x_i;                l = exp_fx(0)          = 32768
//     x_i >  m   : l = ((l * exp(clamp((m - x_i) << SH))) >> 15) + 32768;
//                  m = x_i                        (rescale old sum, add new max)
//     x_i <= m   : l = l + exp(clamp((x_i - m) << SH))
//     with SH = ASU_IN_FRAC - SCORE_FRAC and clamp = _clamp_z (saturate the
//     Q6.10 exp argument at -16384; it is never positive by construction).
//     The raw score is also written to a row buffer (up to SM_ROW_MAX).
//   PASS 2 (emission): for i = 0..n-1:
//     p_i = (exp(clamp((x_i - m) << SH)) << 15) / l    (floor division,
//     restoring sequential divider — see LATENCY). l >= 32768 always (the
//     max element contributes exp(0) = 32768 and rescale never shrinks past
//     the subsequent add), so the golden model's l > 0 guard is structurally
//     satisfied and the quotient fits 16 bits: num < 2^31, l >= 2^15.
//
// Output: UNSIGNED Q1.15 probabilities, range 0..32768; 32768 (0x8000)
// encodes exactly 1.0 (reachable, e.g. single-element rows). `last` marks
// the row's final probability.
//
// Streams (§5): valid/ready, data stable while valid && !ready. BOTH ends go
// through rtl/xbr/stream_skid.sv (V0.5-verified, D-019).
//
// Job semantics (D-006): a job = one `last`-framed input row.
//   busy = (FSM != IDLE). The FSM stays in ST_DRAIN until every output beat
//          of the row has been accepted POST-SKID, so busy covers beats
//          pending anywhere in the block, skids included.
//   done = 1-cycle pulse, asserted only after all n beats are accepted
//          post-skid (and after busy has fallen: done cycle has busy = 0).
//
// Legality (mirrors §3 reject discipline): a row longer than SM_ROW_MAX is
// rejected — row_error 1-cycle pulse + row_error_sticky (clear on reset
// only), the offending row is consumed through its `last` beat with NO state
// change, NO output beats, and NO done pulse.
//
// LATENCY / THROUGHPUT (documented per deliverable spec):
//   Pass 1 accepts 1 score/cycle (post-skid). Emission costs per element:
//   1 (LOAD: buffer read + exp LUT) + 31 (DIV: restoring divider, one cycle
//   per numerator bit, NUM_W = 16 + ASU_OUT_FRAC = 31) + >=1 (PUSH into the
//   output skid) ~= 33 cycles/element unstalled; done follows the final
//   post-skid acceptance by 2 cycles.

module asu_softmax
  import apex_pkg::*;
#(
  parameter int unsigned SM_ROW_MAX = 1024,  // legality bound on row length
  parameter int unsigned SCORE_FRAC = 0      // frac bits of the score stream
                                             // (C-5: raw INT32 scores -> 0)
) (
  input  logic               clk,
  input  logic               rst_n,      // synchronous, active low

  // score input stream: single INT32 lane + last (lane32-style, 1 lane)
  input  logic               s_valid,
  output logic               s_ready,
  input  logic signed [31:0] s_score,
  input  logic               s_last,

  // probability output stream: UQ1.15 + last
  output logic               m_valid,
  input  logic               m_ready,
  output logic        [15:0] m_prob,
  output logic               m_last,

  // job / status (D-006, §3)
  output logic               busy,
  output logic               done,
  output logic               row_error,
  output logic               row_error_sticky
);

  // elaboration-time legality of the configuration
  generate
    if (SCORE_FRAC > ASU_IN_FRAC) begin : g_cfg_err_frac
      $error("asu_softmax: SCORE_FRAC > ASU_IN_FRAC unsupported");
    end
    if (SM_ROW_MAX < 1) begin : g_cfg_err_row
      $error("asu_softmax: SM_ROW_MAX must be >= 1");
    end
  endgenerate

  localparam int unsigned SH    = ASU_IN_FRAC - SCORE_FRAC;       // left shift
  localparam int unsigned CNT_W = $clog2(SM_ROW_MAX + 1);         // 0..SM_ROW_MAX
  // l bound proof: each beat adds <= 2^15 and rescale never grows l
  // ((l*e)>>15 <= l since e <= 2^15), so l <= SM_ROW_MAX * 2^15.
  localparam int unsigned L_W   = ASU_OUT_FRAC + 1 + $clog2(SM_ROW_MAX); // 26
  localparam int unsigned NUM_W = 16 + ASU_OUT_FRAC;              // 31
  localparam int unsigned IT_W  = $clog2(NUM_W);                  // 5
  localparam int unsigned IDX_W = (SM_ROW_MAX > 1) ? $clog2(SM_ROW_MAX) : 1;
  localparam logic [L_W-1:0] L_ONE = L_W'(1 << ASU_OUT_FRAC);     // exp_fx(0)

  // ── input skid (§5: every boundary crossing) ─────────────────────────────
  logic               i_valid, i_ready;
  logic        [32:0] i_bus;
  logic signed [31:0] i_score;
  logic               i_last;

  stream_skid #(.WIDTH(33)) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (s_valid),
    .s_ready (s_ready),
    .s_data  ({s_score, s_last}),
    .m_valid (i_valid),
    .m_ready (i_ready),
    .m_data  (i_bus)
  );
  assign i_score = signed'(i_bus[32:1]);
  assign i_last  = i_bus[0];

  // ── state ─────────────────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    ST_IDLE, ST_COLLECT, ST_FLUSH, ST_LOAD, ST_DIV, ST_PUSH, ST_DRAIN
  } st_e;
  st_e st;

  logic [CNT_W-1:0]   cnt;        // beats accepted this row
  logic [CNT_W-1:0]   n_len;      // final row length
  logic [CNT_W-1:0]   emit_idx;   // element being emitted
  logic [CNT_W-1:0]   out_cnt;    // POST-SKID accepted output beats
  logic signed [31:0] mmax;       // running max m
  logic [L_W-1:0]     lsum;       // running rescaled sum l (frac ASU_OUT_FRAC)
  logic signed [31:0] row_buf [SM_ROW_MAX];

  // ── exp argument: d = _clamp_z(diff << SH), one shared LUT ───────────────
  // COLLECT, new max : diff = m − x_i   (< 0)   — rescale argument
  // COLLECT, else    : diff = x_i − m   (<= 0)  — accumulate argument
  // LOAD (emission)  : diff = buf[i] − m (<= 0) — emission argument
  logic               is_newmax;
  logic signed [31:0] xi_c;
  logic signed [32:0] diff33;
  logic signed [42:0] dsh;
  logic signed [15:0] z_q;
  logic        [15:0] e_lut;

  assign is_newmax = (i_score > mmax);
  assign xi_c      = (st == ST_COLLECT) ? i_score
                                        : row_buf[emit_idx[IDX_W-1:0]];
  assign diff33    = (st == ST_COLLECT && is_newmax)
                   ? (33'(mmax) - 33'(i_score))
                   : (33'(xi_c) - 33'(mmax));
  assign dsh       = 43'(diff33) <<< SH;
  assign z_q       = (dsh < -43'sd16384) ? -16'sd16384 : 16'(dsh);

  asu_exp_lut u_exp (
    .z_q   (z_q),
    .exp_q (e_lut)
  );

  // ── l update (pass 1) ─────────────────────────────────────────────────────
  logic [L_W+15:0] resc_full;             // 26u x 16u = 42 bits, exact
  logic [L_W-1:0]  l_resc;
  logic [L_W-1:0]  l_next;

  assign resc_full = 42'(lsum) * 42'(e_lut);
  assign l_resc    = L_W'(resc_full >> ASU_OUT_FRAC);   // <= lsum, fits L_W

  always_comb begin
    if (st == ST_IDLE)   l_next = L_ONE;                // first beat: exp(0)
    else if (is_newmax)  l_next = l_resc + L_ONE;       // rescale + new max
    else                 l_next = lsum + L_W'(e_lut);   // accumulate
  end

  // ── restoring divider: q = (e << 15) / l, floor, 31 cycles ───────────────
  // rem_r needs only L_W bits: the restoring invariant rem < l holds before
  // every shift, and after a shift-in either rem_shift < l (no take) or
  // rem_shift - l < l (take), so the stored remainder always fits L_W bits.
  logic [NUM_W-1:0] num;
  logic [L_W-1:0]   rem_r;
  logic [15:0]      quot;
  logic [IT_W-1:0]  iter;

  logic [L_W:0]   rem_shift;
  logic           div_take;
  logic [L_W-1:0] rem_next;

  assign rem_shift = {rem_r, num[NUM_W-1]};
  assign div_take  = (rem_shift >= (L_W+1)'(lsum));
  assign rem_next  = div_take ? L_W'(rem_shift - (L_W+1)'(lsum))
                              : L_W'(rem_shift);
  // quotient fits 16 bits: num < 2^31 and l >= 2^15 imply q < 2^16, so the
  // bits shifted out of quot's MSB during the first 15 iterations are 0.

  // ── output skid ───────────────────────────────────────────────────────────
  logic o_valid, o_ready, o_last;
  logic in_xfer, m_xfer;

  assign o_last  = (emit_idx == n_len - CNT_W'(1));
  assign o_valid = (st == ST_PUSH);
  assign i_ready = (st == ST_IDLE) || (st == ST_COLLECT) || (st == ST_FLUSH);
  assign in_xfer = i_valid && i_ready;
  assign m_xfer  = m_valid && m_ready;

  stream_skid #(.WIDTH(17)) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  ({quot, o_last}),
    .m_valid (m_valid),
    .m_ready (m_ready),
    .m_data  ({m_prob, m_last})
  );

  assign busy = (st != ST_IDLE);

  // ── FSM + datapath registers ──────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st               <= ST_IDLE;
      cnt              <= '0;
      n_len            <= '0;
      emit_idx         <= '0;
      out_cnt          <= '0;
      mmax             <= '0;
      lsum             <= '0;
      num              <= '0;
      rem_r            <= '0;
      quot             <= '0;
      iter             <= '0;
      done             <= 1'b0;
      row_error        <= 1'b0;
      row_error_sticky <= 1'b0;
    end else begin
      done      <= 1'b0;
      row_error <= 1'b0;
      if (m_xfer) out_cnt <= out_cnt + CNT_W'(1);

      case (st)
        ST_IDLE: begin
          if (in_xfer) begin              // first beat of a new row
            row_buf[0] <= i_score;
            mmax       <= i_score;
            lsum       <= l_next;         // = L_ONE
            cnt        <= CNT_W'(1);
            out_cnt    <= '0;
            if (i_last) begin
              n_len    <= CNT_W'(1);
              emit_idx <= '0;
              st       <= ST_LOAD;
            end else begin
              st <= ST_COLLECT;
            end
          end
        end

        ST_COLLECT: begin
          if (in_xfer) begin
            if (cnt == CNT_W'(SM_ROW_MAX)) begin
              // beat SM_ROW_MAX+1: row too long — legality reject (§3 style)
              row_error        <= 1'b1;
              row_error_sticky <= 1'b1;
              st               <= i_last ? ST_IDLE : ST_FLUSH;
            end else begin
              row_buf[cnt[IDX_W-1:0]] <= i_score;
              if (is_newmax) mmax <= i_score;
              lsum <= l_next;
              cnt  <= cnt + CNT_W'(1);
              if (i_last) begin
                n_len    <= cnt + CNT_W'(1);
                emit_idx <= '0;
                st       <= ST_LOAD;
              end
            end
          end
        end

        ST_FLUSH: begin                   // consume rejected row, no effects
          if (in_xfer && i_last) st <= ST_IDLE;
        end

        ST_LOAD: begin                    // e = exp for element emit_idx
          num   <= {e_lut, 15'b0};        // numerator e << ASU_OUT_FRAC
          rem_r <= '0;
          quot  <= '0;
          iter  <= '0;
          st    <= ST_DIV;
        end

        ST_DIV: begin                     // one quotient bit per cycle
          rem_r <= rem_next;
          quot  <= {quot[14:0], div_take};
          num   <= num << 1;
          if (iter == IT_W'(NUM_W - 1)) st <= ST_PUSH;
          else                          iter <= iter + IT_W'(1);
        end

        ST_PUSH: begin                    // offer p_i to the output skid
          if (o_ready) begin
            if (o_last) begin
              st <= ST_DRAIN;
            end else begin
              emit_idx <= emit_idx + CNT_W'(1);
              st       <= ST_LOAD;
            end
          end
        end

        ST_DRAIN: begin                   // D-006: wait for post-skid accepts
          if (out_cnt == n_len) begin
            done <= 1'b1;
            st   <= ST_IDLE;
          end
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule
