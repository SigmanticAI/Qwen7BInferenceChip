// apex_proj_bias.sv — the PROJECTION-BIAS S-2 seam (I-C gap B).
//
// WHY THIS BLOCK EXISTS. Qwen2/2.5 puts biases on the q/k/v projections
// (o/gate/up/down have none). The golden arbiter adds them in REAL units to
// the DEQUANTIZED projection output and BEFORE the single fp16 bus narrowing
// — golden/apex_golden/transformer.py:506-522:
//
//     r.q_real = acc_q * comp_q ; r.K_real = acc_k * comp_k ; ...   (:506-508)
//     r.q_real += bq ; r.K_real += bk ; r.V_real += bv              (:509-514)
//     if bus.rope_in_f16: r.q_real = _f16(r.q_real)                 (:521-522)
//                         r.K_real = _f16(r.K_real)
//     (V: the same ONE narrowing sits at the attention-core input,
//      V_f16 = f64_to_f16_bits(r.V_real), :541 — V is not narrowed twice)
//
// so the ONLY bit-exact insertion point is INSIDE the exact product, between
// "acc x composite" and the single RNE to fp16. Adding the bias to an already
// narrowed f16(acc*c) is a DOUBLE rounding and is not bit-exact; folding it
// into the INT8 weights or an extra activation row leaves the int8 grid.
//
// This block therefore reproduces apex_scale_quant's MODE_F16 element
// datapath (the same exact 25x11 product) with the bias summed into it on a
// common exact grid, narrowed ONCE through the verified
// f16_arith_pkg::f16_pack_real. It is a SIBLING of apex_scale_quant on the
// same seam, selected by a route level — apex_scale_quant is a VERIFIED block
// with a published bit-exactness proof (P1-P5) and is not edited.
//
// EQUIVALENCE (the reason this is not a risky duplicate): with a bias vector
// of +0 this block is bit-identical to apex_scale_quant MODE_F16 on every
// legal element, including both contract-violation behaviours (C1 clamp, C2
// bad-composite -> 16'h0000). The unit suite co-simulates the two blocks and
// requires that identity beat-for-beat.
//
// ── NUMERIC MODEL ─────────────────────────────────────────────────────────
//   v : INT32 accumulator, |v| <= 2^24            (C1, else range_error)
//   c : fp32 sideband, POSITIVE NORMAL, fp16-GRADE frac[12:0]==0
//                                                 (C2, else scale_error)
//   b : fp16 bias element (the golden contract: "biases in REAL units on the
//       fp16 grid", transformer.py:348; run_tinynpu.py:216 casts the real
//       Qwen checkpoint bias to float16 before it ever reaches golden)
//
//   x = v*c   = +-P * 2^ex,  P = |v| * m11 < 2^35,  ex = ec - 137   (exact)
//   b         = +-B * 2^-24, B < 2^40                               (exact)
//   g   = min(ex, -24)                       — the common exact grid
//   ACC = +-(P << (ex-g)) +- (B << (-24-g))  — exact integer sum on 2^g
//   y   = f16_pack_real(sign(ACC), |ACC|, g) — the ONE RNE
//
// ── EXACT-OR-REFUSED (window_error; the apex_residual / apex_layer_deq rule)
//   W1 alignment window: (ex-g) <= PB_SX_MAX and (-24-g) <= PB_SB_MAX keeps
//      both shifted operands < 2^56, so their sum fits the 57-bit
//      f16_pack_real significand port with no bit lost. In ex terms the
//      window is ex in [-40, -3], i.e. composite c in [2^-30, 2^7] — every
//      s_h*s_w composite the tile produces sits deep inside it (measured
//      margins are in the suite log; the L4/7B composites land near 2^-20).
//   W2 float64 realizability: golden performs the bias add in float64, so a
//      sum needing MORE than 53 significand bits would be ROUNDED there and
//      this block's single-RNE result could then differ. The span of the
//      exact sum is measured (msb-lsb, the apex_layer_deq idiom) and a span
//      > 53 bits REFUSES. Refusing is the honest behaviour: inside the
//      window golden's add is provably exact, so "one RNE here" == "golden's
//      f64 add then f64_to_f16_bits" bit-for-bit.
//   A refused element raises window_error (pulse + sticky) and ABORTS the
//   job: remaining inputs are consumed and dropped, nothing more is emitted.
//
// ── ZERO SIGN ─────────────────────────────────────────────────────────────
//   x is never -0 (v is an integer and c is positive normal, so v==0 gives
//   +0), hence golden's RNE "-0 only from (-0)+(-0)" can never fire and an
//   exactly-zero sum is +0. Documented, not assumed.
//
// ── BIAS VECTOR ───────────────────────────────────────────────────────────
//   [BN_MAX] x 16 b, host-staged through the LAYER window (LAYER_PTR bank 3
//   + LAYER_DATA, the same auto-increment load port the phase/residual banks
//   use, so the walker reaches it exactly the way it will reach those).
//   Read address = the element index WITHIN the job: the host stages the
//   slice for the row being projected. All T token rows of one KV head share
//   one bias slice, so a layer stages bk / bv / bq once each.
//
// ── JOB / STREAM SEMANTICS ────────────────────────────────────────────────
//   Identical to apex_scale_quant MODE_F16 (D-006, §3 reject discipline):
//   job = {cols}, 1 <= cols <= min(D, BN_MAX) else job_error pulse+sticky
//   with NO state change; load `cols` sideband composites FIRST, then consume
//   `cols` v elements; v `last` must mark element cols-1 (frame_error
//   otherwise); done <=> every output beat ACCEPTED post-skid.

module apex_proj_bias
  import apex_pkg::*;
  import f16_arith_pkg::*;
#(
  parameter int unsigned D      = 128,   // max elements (columns) per job
  parameter int unsigned BN_MAX = 128,   // staged bias-vector depth
  localparam int unsigned BAW   = $clog2(BN_MAX)
)(
  input  logic               clk,
  input  logic               rst_n,        // synchronous, active low

  // job interface (D-006)
  input  logic               job_valid,
  output logic               job_ready,
  input  logic [DIM_W-1:0]   job_cols,
  output logic               job_error,
  output logic               job_error_sticky,
  output logic               busy,
  output logic               done,

  // bias-vector load port (LAYER_PTR bank 3 / LAYER_DATA)
  input  logic               lw_en,
  input  logic [BAW-1:0]     lw_addr,
  input  logic [15:0]        lw_data,

  // fp32 composite sideband in (one per element, loaded first)
  input  logic               cs_valid,
  output logic               cs_ready,
  input  logic [31:0]        cs_data,

  // INT32 value stream in
  input  logic               v_valid,
  output logic               v_ready,
  input  logic signed [31:0] v_data,
  input  logic               v_last,

  // fp16 element stream out (the S-2 seam shape: last on element cols-1)
  output logic               f_valid,
  input  logic               f_ready,
  output logic [15:0]        f_data,
  output logic               f_last,

  // contract monitors (golden-domain guards; pulse + sticky)
  output logic               range_error,   // C1: |v| > 2^24
  output logic               range_error_sticky,
  output logic               scale_error,   // C2: sideband not fp16-grade normal
  output logic               scale_error_sticky,
  output logic               frame_error,   // v `last` mis-framed
  output logic               frame_error_sticky,
  output logic               window_error,  // W1/W2: exact-or-refused
  output logic               window_error_sticky
);

  localparam int unsigned CW      = 12;                // element counters
  localparam logic [31:0] MAG_MAX = 32'h0100_0000;     // 2^24 (C1)

  // W1 shift bounds: P < 2^35 and |B| < 2^40 must stay < 2^56 after their
  // alignment shift so the SUM fits the 57-bit f16_pack_real significand.
  localparam int PB_SX_MAX   = 21;
  localparam int PB_SB_MAX   = 16;
  localparam int PB_SPAN_MAX = 53;                     // W2: float64 significand
  localparam int unsigned PB_ACC_W = 58;               // signed, 57 magnitude bits

  // ── input skids (§5) ──────────────────────────────────────────────────────
  logic        c_valid, c_ready;
  logic [31:0] c_bus;

  stream_skid #(.WIDTH(32)) u_cs_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (cs_valid), .s_ready (cs_ready), .s_data (cs_data),
    .m_valid (c_valid), .m_ready (c_ready), .m_data (c_bus)
  );

  logic        i_valid, i_ready;
  logic [32:0] i_bus;
  logic signed [31:0] i_v;
  logic               i_last;

  stream_skid #(.WIDTH(33)) u_v_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (v_valid), .s_ready (v_ready), .s_data ({v_data, v_last}),
    .m_valid (i_valid), .m_ready (i_ready), .m_data (i_bus)
  );
  assign i_v    = $signed(i_bus[32:1]);
  assign i_last = i_bus[0];

  // ── output skid ───────────────────────────────────────────────────────────
  logic        fo_valid, fo_ready;
  logic [16:0] fo_bus, f_vec;

  stream_skid #(.WIDTH(17)) u_f_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (fo_valid), .s_ready (fo_ready), .s_data (fo_bus),
    .m_valid (f_valid), .m_ready (f_ready), .m_data (f_vec)
  );
  assign f_data = f_vec[16:1];
  assign f_last = f_vec[0];

  // ── sideband decode (C2; the apex_scale_quant encoding, bit for bit) ─────
  logic        cd_bad, cd_grade_viol;
  logic [7:0]  cd_ec;
  logic [10:0] cd_m11;
  assign cd_ec         = c_bus[30:23];
  assign cd_m11        = {1'b1, c_bus[22:13]};
  assign cd_bad        = c_bus[31] || (cd_ec == 8'h00) || (cd_ec == 8'hFF);
  assign cd_grade_viol = cd_bad || (c_bus[12:0] != 13'd0);

  // ── the biased element: {ok, fp16} = { W1&W2, f16(v*c + b) } ─────────────
  function automatic logic [16:0] pb_elem(input logic        sgn,
                                          input logic [24:0] mag,
                                          input logic [10:0] m11,
                                          input logic [7:0]  ec,
                                          input logic        bad,
                                          input logic [15:0] b16);
    logic [34:0]                 p35;
    int                          ex, g, sx, sb, hi, lo;
    logic signed [F16_V24_W-1:0] bv24;
    logic [F16_V24_W-1:0]        bmagv;
    logic [56:0]                 xfix, bfix, amag;
    logic signed [PB_ACC_W-1:0]  accv;
    logic                        okw;
    if (bad) return {1'b1, 16'h0000};      // C2: apex_scale_quant parity
    p35 = 35'(mag) * 35'(m11);             // exact (proof P1)
    ex  = int'({24'b0, ec}) - 137;
    if (ex >= -24) begin g = -24; sx = ex + 24; sb = 0;        end
    else           begin g = ex;  sx = 0;       sb = -24 - ex; end
    okw   = (sx <= PB_SX_MAX) && (sb <= PB_SB_MAX);            // W1
    bv24  = f16_to_v24(b16);
    bmagv = b16[15] ? F16_V24_W'(-bv24) : F16_V24_W'(bv24);
    xfix  = okw ? (57'(p35)   << sx) : 57'd0;
    bfix  = okw ? (57'(bmagv) << sb) : 57'd0;
    accv  = (sgn     ? -$signed({1'b0, xfix}) : $signed({1'b0, xfix}))
          + (b16[15] ? -$signed({1'b0, bfix}) : $signed({1'b0, bfix}));
    amag  = (accv < 0) ? 57'(-accv) : 57'(accv);
    hi = 0; lo = 56;                       // W2 span (apex_layer_deq idiom)
    for (int i = 0; i < 57; i++) begin
      if (amag[i])      hi = i;
      if (amag[56 - i]) lo = 56 - i;
    end
    if (!okw || ((amag != '0) && ((hi - lo) >= PB_SPAN_MAX)))
      return {1'b0, 16'h0000};             // REFUSE, exact-or-refused
    // exact-zero sum is +0: x is never -0 (v integer, c positive normal)
    return {1'b1, f16_pack_real(accv < 0, amag, g)};
  endfunction

  // ── FSM / state (apex_scale_quant MODE_F16 shape) ────────────────────────
  typedef enum logic [2:0] {
    ST_IDLE   = 3'd0,
    ST_LOAD   = 3'd1,   // load per-element sideband composites
    ST_INGEST = 3'd2,   // consume v elements: convert + emit
    ST_DRAIN  = 3'd3,   // refused job: consume to last, emit nothing
    ST_WAIT   = 3'd4    // D-006: wait for post-skid acceptance
  } state_e;

  state_e        state;
  logic [CW-1:0] cols_q, load_cnt, idx, f_acc, n_emit;
  logic          have;
  logic [25:0]   vhold;      // {sign, mag25}
  logic [19:0]   chold;      // {bad, ec, m11}
  logic [15:0]   bhold;
  logic [15:0]   bias [BN_MAX];
  logic [19:0]   cmem [D];

  logic legal;
  assign legal = (job_cols != '0) && ({20'b0, job_cols} <= 32'(D))
              && ({20'b0, job_cols} <= 32'(BN_MAX));

  // ingest decode (C1)
  logic        in_sgn;
  logic [31:0] in_mag32;
  logic        in_rng;
  logic [24:0] in_mag;
  assign in_sgn   = i_v[31];
  assign in_mag32 = in_sgn ? (~$unsigned(i_v) + 32'd1) : $unsigned(i_v);
  assign in_rng   = (in_mag32 > MAG_MAX);
  assign in_mag   = in_rng ? 25'(MAG_MAX) : in_mag32[24:0];

  logic [16:0] el_out;
  assign el_out = pb_elem(vhold[25], vhold[24:0], chold[10:0], chold[18:11],
                          chold[19], bhold);

  assign c_ready  = (state == ST_LOAD);
  assign i_ready  = (state == ST_DRAIN) || ((state == ST_INGEST) && !have);
  assign fo_valid = (state == ST_INGEST) && have && el_out[16];
  assign fo_bus   = {el_out[15:0], (idx == cols_q - CW'(1))};

  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || f_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state               <= ST_IDLE;
      done                <= 1'b0;
      job_error           <= 1'b0;
      job_error_sticky    <= 1'b0;
      range_error         <= 1'b0;
      range_error_sticky  <= 1'b0;
      scale_error         <= 1'b0;
      scale_error_sticky  <= 1'b0;
      frame_error         <= 1'b0;
      frame_error_sticky  <= 1'b0;
      window_error        <= 1'b0;
      window_error_sticky <= 1'b0;
      cols_q              <= '0;
      load_cnt            <= '0;
      idx                 <= '0;
      f_acc               <= '0;
      n_emit              <= '0;
      have                <= 1'b0;
      vhold               <= '0;
      chold               <= '0;
      bhold               <= '0;
    end else begin
      done         <= 1'b0;
      job_error    <= 1'b0;
      range_error  <= 1'b0;
      scale_error  <= 1'b0;
      frame_error  <= 1'b0;
      window_error <= 1'b0;
      if (lw_en) bias[lw_addr] <= lw_data;
      if (f_valid && f_ready)   f_acc  <= f_acc + CW'(1);
      if (fo_valid && fo_ready) n_emit <= n_emit + CW'(1);

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state    <= ST_LOAD;
              cols_q   <= CW'(job_cols);
              load_cnt <= '0;
              idx      <= '0;
              f_acc    <= '0;
              n_emit   <= '0;
              have     <= 1'b0;
            end else begin                 // §3: pulse + sticky, no effects
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
            end
          end
        end

        ST_LOAD: begin
          if (c_valid && c_ready) begin
            cmem[load_cnt[$clog2(D)-1:0]] <= {cd_bad, cd_ec, cd_m11};
            if (cd_grade_viol) begin       // C2 guard
              scale_error        <= 1'b1;
              scale_error_sticky <= 1'b1;
            end
            if (load_cnt == cols_q - CW'(1)) begin
              state    <= ST_INGEST;
              load_cnt <= '0;
            end else begin
              load_cnt <= load_cnt + CW'(1);
            end
          end
        end

        ST_INGEST: begin
          if (i_valid && i_ready) begin
            if (in_rng) begin              // C1 guard
              range_error        <= 1'b1;
              range_error_sticky <= 1'b1;
            end
            if (i_last != (idx == cols_q - CW'(1))) begin
              frame_error        <= 1'b1;
              frame_error_sticky <= 1'b1;
            end
            vhold <= {in_sgn, in_mag};
            chold <= cmem[idx[$clog2(D)-1:0]];
            bhold <= bias[idx[BAW-1:0]];
            have  <= 1'b1;
          end else if (have && !el_out[16]) begin
            window_error        <= 1'b1;   // W1/W2: refuse + abort the job
            window_error_sticky <= 1'b1;
            have                <= 1'b0;
            if (idx == cols_q - CW'(1)) state <= ST_WAIT;
            else                        state <= ST_DRAIN;
          end else if (have && fo_ready) begin
            have <= 1'b0;                  // element accepted into the skid
            if (idx == cols_q - CW'(1)) state <= ST_WAIT;
            else                         idx  <= idx + CW'(1);
          end
        end

        ST_DRAIN: begin                    // consume to last, emit nothing
          if (i_valid && i_ready && i_last) state <= ST_WAIT;
        end

        ST_WAIT: begin                     // D-006: post-skid acceptance
          if (f_acc == n_emit) begin       // holds for refused (short) jobs too
            done  <= 1'b1;
            state <= ST_IDLE;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
