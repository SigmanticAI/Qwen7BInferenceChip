// apex_layer_deq.sv — exact INT32 x fp16-grade-fp32 dequantizer for the
// IB-LAYER o-proj/attn activation-prep path (IB_LAYER.md §3): int32 result
// codes (PV o8 via the serializer, or any epilogue stream) times ONE
// fp16-graded fp32 composite per job, emitting the EXACT fp32 values the
// C-1 feeder (seam_feeder_quant) consumes.
//
// CONTRACT — "exact or refused", the C2/scale_error discipline extended:
//   * job composite must be a POSITIVE NORMAL fp32 with fp16-grade
//     significand (frac[12:0] == 0 — apex_scale_quant C2, the same
//     weight_codec.f16_grade form D-030 pins). Violation = job_error
//     pulse + sticky, job consumed, NO state change (§3 legality shape).
//   * every product acc*comp must be EXACTLY representable in fp32:
//     significand span <= 24 bits and biased exponent in [1, 254].
//     A violating element raises exact_error (pulse + sticky) and ABORTS
//     the job: remaining inputs are consumed and dropped, nothing more is
//     emitted. Silent rounding here would silently break the bit-exact
//     composition chain — refusal is the honest behavior. (o8-grade
//     streams are exact by construction: 8-bit x 11-bit <= 19-bit span;
//     the D-030 lemma. Wide accs are refused, not rounded.)
//   * frame: `ilast` must arrive exactly on beat cols-1. Early last drops
//     the job (frame_error, nothing further emitted); missing last raises
//     frame_error and resyncs to the next last. Same shape as rope_row.
//
// Numerics (all integer, one fp32 encode, no rounding anywhere):
//   comp = k * 2^(e8-137), k = (2^23 + m23) >> 13 in [1024, 2047]  (graded)
//   |acc| * k = prod (<= 43 bits);  value = prod * 2^(e8-137)
//   fp32: biased = msb(prod) + e8 - 10; mant = prod aligned to 24 bits
//   (exact because span <= 24); sign = acc < 0; acc == 0 -> +0.
//
// Timing: 1 element/cycle steady state; skids on both stream boundaries.

module apex_layer_deq #(
  parameter int unsigned COLS_MAX = 4095,
  localparam int unsigned CW = $clog2(COLS_MAX + 1)
)(
  input  logic          clk,
  input  logic          rst_n,

  // job port (D-006 style: ready gated on IDLE, reject = pulse + sticky)
  input  logic          jb_valid,
  output logic          jb_ready,
  input  logic [CW-1:0] jb_cols,
  input  logic [31:0]   jb_comp,

  // int32 element stream in
  input  logic          iv,
  output logic          ir,
  input  logic signed [31:0] idata,
  input  logic          ilast,

  // exact fp32 element stream out
  output logic          ov,
  input  logic          orr,
  output logic [31:0]   odata,
  output logic          olast,

  output logic          busy,
  output logic          done,
  output logic          job_error,          // composite illegal (pulse)
  output logic          job_error_sticky,
  output logic          exact_error,        // product not fp32-exact (pulse)
  output logic          exact_error_sticky,
  output logic          frame_error,
  output logic          frame_error_sticky
);

  logic unused_ok;

  // ── skids ─────────────────────────────────────────────────────────────────
  logic        s_iv, s_ir, s_ilast;
  logic [31:0] s_idata;
  stream_skid #(.WIDTH(33)) u_in_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (iv), .s_ready (ir), .s_data ({ilast, idata}),
    .m_valid (s_iv), .m_ready (s_ir), .m_data ({s_ilast, s_idata})
  );
  logic        s_ov, s_or;
  logic [32:0] s_ovec, m_ovec;
  stream_skid #(.WIDTH(33)) u_out_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (s_ov), .s_ready (s_or), .s_data (s_ovec),
    .m_valid (ov), .m_ready (orr), .m_data (m_ovec)
  );
  assign odata = m_ovec[31:0];
  assign olast = m_ovec[32];

  // ── composite legality + decode ───────────────────────────────────────────
  wire [7:0]  c_e8 = jb_comp[30:23];
  wire [22:0] c_m  = jb_comp[22:0];
  wire        comp_legal = (jb_comp[31] == 1'b0)          // positive
                        && (c_e8 != 8'h00) && (c_e8 != 8'hFF)  // normal
                        && (c_m[12:0] == 13'h0);          // fp16-grade (C2)

  typedef enum logic [1:0] {ST_IDLE, ST_RUN, ST_DRAIN} st_e;
  st_e            st;
  logic [CW-1:0]  cols_q, cnt;
  logic [10:0]    k_q;                       // graded significand [1024,2047]
  logic [7:0]     e8_q;
  logic           err_jb, err_ex, err_fr;
  logic           stk_jb, stk_ex, stk_fr;

  // ── exact product -> fp32 encode (combinational on the accepted beat) ────
  logic        [31:0] mag;
  logic        [42:0] prod;
  int                 p, l;
  logic        [7:0]  biased;
  logic        [23:0] m24;
  logic               p_sign, exact_ok;
  logic        [31:0] enc;

  always_comb begin
    p_sign = s_idata[31];
    mag    = p_sign ? (~s_idata + 32'd1) : s_idata;
    prod   = mag * {32'b0, k_q};
    p      = 0;
    l      = 42;
    for (int i = 0; i < 43; i++) begin
      if (prod[i]) p = i;
      if (prod[42 - i]) l = 42 - i;
    end
    // biased exponent = p + e8 - 10  (value = prod * 2^(e8-137))
    exact_ok = (prod == '0)
             || (((p - l) < 24)
                 && ((p + int'({24'b0, e8_q})) > 10)          // biased >= 1
                 && ((p + int'({24'b0, e8_q})) < 265));       // biased <= 254
    biased = 8'((p + int'({24'b0, e8_q})) - 10);
    // align the (span<=24) significand to 24 bits — exact by construction
    if (p >= 23) m24 = 24'(prod >> (p - 23));
    else         m24 = 24'(prod << (23 - p));
    enc = (prod == '0) ? 32'h0
        : {p_sign, biased, m24[22:0]};
  end
  assign unused_ok = &{1'b0, m24[23]};

  wire frame_ok = (s_ilast == (cnt == cols_q - CW'(1)));
  assign jb_ready = (st == ST_IDLE);
  assign s_ir     = (st == ST_RUN) ? s_or            // 1-in-1-out coupling
                  : (st == ST_DRAIN);
  assign s_ov     = (st == ST_RUN) && s_iv && exact_ok && frame_ok;
  assign s_ovec   = {(cnt == cols_q - CW'(1)), enc};
  assign busy     = (st != ST_IDLE);
  assign job_error          = err_jb;
  assign exact_error        = err_ex;
  assign frame_error        = err_fr;
  assign job_error_sticky   = stk_jb;
  assign exact_error_sticky = stk_ex;
  assign frame_error_sticky = stk_fr;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st     <= ST_IDLE;
      cols_q <= '0;
      cnt    <= '0;
      k_q    <= '0;
      e8_q   <= '0;
      err_jb <= 1'b0; err_ex <= 1'b0; err_fr <= 1'b0;
      stk_jb <= 1'b0; stk_ex <= 1'b0; stk_fr <= 1'b0;
      done   <= 1'b0;
    end else begin
      err_jb <= 1'b0; err_ex <= 1'b0; err_fr <= 1'b0;
      done   <= 1'b0;
      unique case (st)
        ST_IDLE: begin
          if (jb_valid) begin
            if (!comp_legal || (jb_cols == '0)) begin
              err_jb <= 1'b1;                 // reject: consumed, no state
              stk_jb <= 1'b1;
            end else begin
              cols_q <= jb_cols;
              cnt    <= '0;
              k_q    <= 11'(({1'b1, c_m} >> 13));
              e8_q   <= c_e8;
              st     <= ST_RUN;
            end
          end
        end
        ST_RUN: begin
          if (s_iv && s_ir) begin
            if (s_ilast != (cnt == cols_q - CW'(1))) begin
              err_fr <= 1'b1;
              stk_fr <= 1'b1;
              if (s_ilast) begin              // early last: job dropped
                st   <= ST_IDLE;
                done <= 1'b1;
              end else begin                  // missing last: resync
                st <= ST_DRAIN;
              end
            end else if (!exact_ok) begin
              err_ex <= 1'b1;                 // exact-or-refused: abort job
              stk_ex <= 1'b1;
              if (s_ilast) begin
                st   <= ST_IDLE;
                done <= 1'b1;
              end else begin
                st <= ST_DRAIN;
              end
            end else if (s_ilast) begin
              st   <= ST_IDLE;
              done <= 1'b1;
            end else begin
              cnt <= cnt + CW'(1);
            end
          end
        end
        ST_DRAIN: begin                       // consume to last, emit nothing
          if (s_iv && s_ir && s_ilast) begin
            st   <= ST_IDLE;
            done <= 1'b1;
          end
        end
        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule
