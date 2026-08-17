// asu_swiglu.sv — the SwiGLU seam unit (IB-LAYER S3):
//   p[i] = f16( silu_apply(gate[i]) * up[i] )        (transformer.py:480)
// consuming the RAW int32 GEMM accumulators plus one fp16-graded fp32
// composite per operand tensor (gate: s_h2*s_wg, up: s_h2*s_wu — the
// D-030/C-LBUS NP-s grades), so every internal value is EXACT integer
// arithmetic and each contract narrowing is a single RNE:
//
//   gate element:  xq = clamp(RNE(acc_g * comp_g * 2^10), +-8192)  (Q5.10,
//                  golden silu_apply's rne + silu_fx's clip, same order)
//                  y  = asu_silu(xq)                (verified LUT core)
//                  silu_f16 = f16_pack_real(sign(y), |y|, -12)  (one RNE —
//                  golden silu_apply's _f16)
//   up element:    up = acc_u * k_u * 2^(e_u-137)   (exact, <= 43-bit sig)
//   product:       p  = f16_pack_real(s_silu^s_up, sig_silu*sig_up,
//                                     e_silu+e_up)  (one RNE — golden's
//                  _f16(silu*up); sig product <= 54 bits, exact — the
//                  D-030 realizability lemma in hardware)
//
// Sequencing: one job = {cols, comp_g, comp_u}; GATE phase consumes cols
// int32 beats and fills the silu buffer; UP phase consumes cols int32
// beats and emits cols fp16 product beats. Composite legality at job
// accept (positive normal + fp16-grade, C2) — violation = job_error
// pulse + sticky, job consumed, no state change. Frame violations (last
// off cols-1) = frame_error + job abort (drop / resync), rope_row shape.
//
// OVERFLOW: a product with E > 15 saturates to +-inf through
// f16_pack_real (golden: numpy f16 cast of a huge float64 -> inf). The
// suite's ovfl vectors gate this behavior explicitly.

module asu_swiglu
  import f16_arith_pkg::*;
#(
  parameter int unsigned COLS_MAX = 64,
  localparam int unsigned CW = $clog2(COLS_MAX + 1)
)(
  input  logic          clk,
  input  logic          rst_n,

  input  logic          jb_valid,
  output logic          jb_ready,
  input  logic [CW-1:0] jb_cols,
  input  logic [31:0]   jb_comp_g,
  input  logic [31:0]   jb_comp_u,

  // gate int32 accumulator stream
  input  logic          gv,
  output logic          gr,
  input  logic signed [31:0] gdata,
  input  logic          glast,

  // up int32 accumulator stream
  input  logic          uv,
  output logic          ur,
  input  logic signed [31:0] udata,
  input  logic          ulast,

  // fp16 product stream out
  output logic          pv,
  input  logic          pr,
  output logic [15:0]   pdata,
  output logic          plast,

  output logic          busy,
  output logic          done,
  output logic          job_error,
  output logic          job_error_sticky,
  output logic          frame_error,
  output logic          frame_error_sticky
);

  // ── skids ─────────────────────────────────────────────────────────────────
  logic        g_v, g_r, g_last;
  logic [31:0] g_d;
  stream_skid #(.WIDTH(33)) u_g_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (gv), .s_ready (gr), .s_data ({glast, gdata}),
    .m_valid (g_v), .m_ready (g_r), .m_data ({g_last, g_d})
  );
  logic        u_v, u_r, u_last;
  logic [31:0] u_d;
  stream_skid #(.WIDTH(33)) u_u_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (uv), .s_ready (ur), .s_data ({ulast, udata}),
    .m_valid (u_v), .m_ready (u_r), .m_data ({u_last, u_d})
  );
  logic        o_v, o_r;
  logic [16:0] o_vec, p_vec;
  stream_skid #(.WIDTH(17)) u_p_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (o_v), .s_ready (o_r), .s_data (o_vec),
    .m_valid (pv), .m_ready (pr), .m_data (p_vec)
  );
  assign pdata = p_vec[15:0];
  assign plast = p_vec[16];

  // ── composite legality ────────────────────────────────────────────────────
  function automatic logic comp_legal(input logic [31:0] c);
    logic unused_hi;
    begin
      unused_hi = &{1'b0, c[22:13]};   // graded bits carried, not checked
      return (c[31] == 1'b0) && (c[30:23] != 8'h00) && (c[30:23] != 8'hFF)
          && (c[12:0] == 13'h0);
    end
  endfunction

  typedef enum logic [1:0] {ST_IDLE, ST_GATE, ST_UP, ST_DRAIN} st_e;
  st_e            st;
  logic           drain_gate_q;              // which stream DRAIN consumes
  logic [CW-1:0]  cols_q, cnt;
  logic [10:0]    kg_q, ku_q;
  logic [7:0]     eg_q, eu_q;
  logic [15:0]    silu_buf [COLS_MAX];
  logic           err_jb, err_fr, stk_jb, stk_fr;

  // ── gate element math (combinational on the accepted gate beat) ──────────
  logic        [31:0] g_mag;
  logic        [42:0] g_prod;
  logic        [56:0] g_shifted;
  logic        [13:0] g_q14;                 // |xq| clamped to 8192 fits 14b
  logic               g_neg, g_round;
  logic signed [15:0] xq;
  int                 g_sh;

  always_comb begin
    g_neg  = g_d[31];
    g_mag  = g_neg ? (~g_d + 32'd1) : g_d;
    g_prod = g_mag * {32'b0, kg_q};
    // value*2^10 = g_prod * 2^(eg_q - 127)
    g_sh   = 127 - int'({24'b0, eg_q});
    g_q14  = 14'd0;
    g_round = 1'b0;
    g_shifted = '0;
    if (g_prod == '0) begin
      g_q14 = 14'd0;
    end else if (g_sh <= 0) begin
      if ((-g_sh) >= 14) g_q14 = 14'd8192;                  // >> clamp
      else begin
        g_shifted = {14'b0, g_prod} << (-g_sh);
        g_q14 = (g_shifted > 57'd8192) ? 14'd8192
                                       : g_q14_of(g_shifted);  // clamp, shift path
      end
    end else if (g_sh > 43) begin
      g_q14 = 14'd0;                                        // rounds to 0
    end else begin
      // RNE shift by g_sh, then clamp (golden: clip AFTER rne)
      g_shifted = {14'b0, g_prod} >> g_sh;
      g_round = (({14'b0, g_prod} & ((57'd1 << g_sh) - 57'd1))
                   > (57'd1 << (g_sh - 1)))
             || ((({14'b0, g_prod} & ((57'd1 << g_sh) - 57'd1))
                   == (57'd1 << (g_sh - 1))) && g_shifted[0]);
      g_shifted = g_shifted + {56'b0, g_round};
      g_q14 = (g_shifted > 57'd8192) ? 14'd8192
                                     : g_q14_of(g_shifted);  // clamp, rne path
    end
    xq = g_neg ? -$signed({2'b0, g_q14}) : $signed({2'b0, g_q14});
  end

  function automatic logic [13:0] g_q14_of(input logic [56:0] v);
    logic unused_hi;
    begin
      unused_hi = &{1'b0, v[56:14]};   // guarded by the 8192 clamp compare
      return v[13:0];
    end
  endfunction

  // silu LUT core (verified, combinational) + the single f16 narrowing
  logic signed [15:0] silu_y;
  asu_silu u_silu (.x_q (xq), .y_q (silu_y));
  wire        silu_neg = silu_y < 0;
  wire [15:0] silu_mag = silu_neg ? 16'(-silu_y) : 16'(silu_y);
  wire [15:0] silu_f16 = f16_pack_real(silu_neg,
                                       {41'b0, silu_mag}, -12);

  // ── up/product math (combinational on the accepted up beat) ──────────────
  logic        [31:0] u_mag;
  logic        [42:0] u_prod;
  logic        [15:0] s_f16;
  logic        [56:0] pp;
  logic               p_sign;
  int                 p_exp;
  logic        [15:0] p_bits;

  always_comb begin
    u_mag  = u_d[31] ? (~u_d + 32'd1) : u_d;
    u_prod = u_mag * {32'b0, ku_q};
    s_f16  = silu_buf[cnt[$clog2(COLS_MAX)-1:0]];
    pp     = {14'b0, u_prod} * {46'b0, f16_sig(s_f16[14:0])};
    p_sign = s_f16[15] ^ u_d[31];
    p_exp  = (int'({24'b0, eu_q}) - 137) + f16_exp(s_f16[14:10]);
    p_bits = f16_pack_real(p_sign, pp, p_exp);
  end

  wire g_frame_ok = (g_last == (cnt == cols_q - CW'(1)));
  wire u_frame_ok = (u_last == (cnt == cols_q - CW'(1)));

  assign jb_ready = (st == ST_IDLE);
  assign g_r = (st == ST_GATE) || ((st == ST_DRAIN) && drain_gate_q);
  assign u_r = (st == ST_UP) ? o_r : ((st == ST_DRAIN) && !drain_gate_q);
  assign o_v = (st == ST_UP) && u_v && u_frame_ok;
  assign o_vec = {(cnt == cols_q - CW'(1)), p_bits};
  assign busy = (st != ST_IDLE);
  assign job_error          = err_jb;
  assign frame_error        = err_fr;
  assign job_error_sticky   = stk_jb;
  assign frame_error_sticky = stk_fr;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st     <= ST_IDLE;
      cols_q <= '0; cnt <= '0;
      kg_q   <= '0; ku_q <= '0; eg_q <= '0; eu_q <= '0;
      err_jb <= 1'b0; err_fr <= 1'b0; stk_jb <= 1'b0; stk_fr <= 1'b0;
      done   <= 1'b0;
      drain_gate_q <= 1'b0;
    end else begin
      err_jb <= 1'b0; err_fr <= 1'b0; done <= 1'b0;
      unique case (st)
        ST_IDLE: begin
          if (jb_valid) begin
            if (!comp_legal(jb_comp_g) || !comp_legal(jb_comp_u)
                || (jb_cols == '0)
                || (jb_cols > CW'(COLS_MAX))) begin
              err_jb <= 1'b1;
              stk_jb <= 1'b1;
            end else begin
              cols_q <= jb_cols;
              cnt    <= '0;
              kg_q   <= 11'(({1'b1, jb_comp_g[22:0]} >> 13));
              eg_q   <= jb_comp_g[30:23];
              ku_q   <= 11'(({1'b1, jb_comp_u[22:0]} >> 13));
              eu_q   <= jb_comp_u[30:23];
              st     <= ST_GATE;
            end
          end
        end
        ST_GATE: begin
          if (g_v && g_r) begin
            if (!g_frame_ok) begin
              err_fr <= 1'b1;
              stk_fr <= 1'b1;
              cnt    <= '0;
              if (g_last) begin st <= ST_IDLE; done <= 1'b1; end
              else begin drain_gate_q <= 1'b1; st <= ST_DRAIN; end
            end else begin
              silu_buf[cnt[$clog2(COLS_MAX)-1:0]] <= silu_f16;
              if (g_last) begin
                cnt <= '0;
                st  <= ST_UP;
              end else begin
                cnt <= cnt + CW'(1);
              end
            end
          end
        end
        ST_UP: begin
          if (u_v && u_r) begin
            if (!u_frame_ok) begin
              err_fr <= 1'b1;
              stk_fr <= 1'b1;
              cnt    <= '0;
              if (u_last) begin st <= ST_IDLE; done <= 1'b1; end
              else begin drain_gate_q <= 1'b0; st <= ST_DRAIN; end
            end else if (u_last) begin
              cnt  <= '0;
              st   <= ST_IDLE;
              done <= 1'b1;
            end else begin
              cnt <= cnt + CW'(1);
            end
          end
        end
        ST_DRAIN: begin
          if (drain_gate_q ? (g_v && g_r && g_last)
                           : (u_v && u_r && u_last)) begin
            st   <= ST_IDLE;
            done <= 1'b1;
          end
        end
        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule
