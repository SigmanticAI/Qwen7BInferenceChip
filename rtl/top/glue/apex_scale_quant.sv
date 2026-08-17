// apex_scale_quant.sv — v0.1 glue: the "acc × scale" numeric seam micro-block.
//
// ONE block, TWO modes, serving three golden stages of
// golden/apex_golden/attention.py (the NORMATIVE arbiter, D-021/§9b):
//
//   MODE_F16 (job_mode=0) — S-2 KV-cache WRITE-PORT conversion (stage Q4):
//       K_f16[t][j] = f16_RNE( acc[t][j] * (s_h[t] * s_wk) )
//     golden lines: K_real = acc_k * (s_h_val[:T,None]*s_wk);
//                   K_f16 = f64_to_f16_bits(K_real)          # ONE RNE narrowing
//     Per element: x = v * c computed EXACTLY, then a single RNE to fp16
//     (normals, subnormals, signed zero, overflow->inf — full IEEE domain).
//
//   MODE_QUANT (job_mode=1) — C-1 per-row INT8 quant of exact products:
//     * stage Q7 (Q feeder from raw accumulators):
//         q8, s_q = quant_rows_i8(q_real[None,:]),  q_real = acc_q * c
//     * stage Q11 / seam S-4 (P-requant fold; the resolution D-021 pins):
//         c[t] = p[t] * s_v[t]  (fold s_v into the probability BEFORE the
//         requant — contraction-axis scales cannot live in the epilogue),
//         c8, s_c = quant_rows_i8(c[None,:])
//       here v[t] = p[t] (UQ1.15 zero-extended) and the sideband constant is
//       f32(s_v[t] * 2^-15) — exact, so x = v*c IS golden's c[t].
//     Row algorithm (quant_rows_i8, C-1): amax = max|x_i| over the EXACT
//     products; s = f16_RNE(max(amax/127, EPS)), EPS = 2^-14; codes =
//     clamp(RNE(x_i / s), -128, 127); scale emitted first, then codes packed
//     LSB-first into lane8 beats (MXE act convention, as seam_feeder_quant).
//
// ── INPUT CONTRACT (the bit-exactness precondition; violations FLAGGED) ────
//   C1. |v| <= 2^24            (else range_error pulse+sticky; v clamped)
//   C2. sideband scales c are POSITIVE NORMAL fp32 with an fp16-GRADE
//       mantissa: frac[12:0] == 0, i.e. <= 11 significant bits
//                              (else scale_error pulse+sticky; see below)
//   The v0.1 tile flows satisfy both by construction: |acc| <= K*127*127
//   (K<=64 in the smoke chain, and C1 is checked, not assumed) and every
//   composite is fp16-value * power-of-two (s_h*s_w with s_w a power of two;
//   s_v * 2^-15), which is exact in fp32 with an 11-bit mantissa.
//
// ── BIT-EXACTNESS PROOF vs the float64 golden (under C1+C2) ────────────────
//   P1 (products are exact in BOTH domains): the exact product x = v*c has a
//     significand of <= 25+11 = 36 bits < 53, so golden's f64 multiply is
//     EXACT; this block computes the same value as an integer pair
//     (P = |v|*m11 <= 2^35, exponent from c) — identical exact rationals.
//   P2 (F16 mode): golden then applies ONE RNE narrowing to fp16; this block
//     RNE-rounds the same exact value to fp16 directly. Identical.
//   P3 (scale): golden computes RNE_f64(amax/127) then RNE_f16 (a double
//     rounding). Exact ties amax/127 = (2m+1)*2^k are f64-exact and round
//     identically. A non-tie exact quotient is >= 2^(E-36)/127 > 2^(E-43)
//     away from any fp16 tie point (the difference amax*2^s - 127*tie is a
//     nonzero integer at the 36-bit granularity), while the f64 rounding
//     error is < 2^(E-52): the intermediate rounding can never cross a tie.
//     Hence the exact-rational RNE_f16(amax/127) computed here (14 guard
//     bits + sticky, mirroring seam_feeder_quant proof P1) equals golden.
//     EPS floor mirrors feeder proof P2 (es < -14 => EPS, exact monotone).
//   P4 (codes): golden computes rint(RNE_f64(x/s)). Exact half-integer ties
//     are f64-exact (num*2 = odd*den with values <= 2^8) and round-half-even
//     identically both ways. A non-tie exact ratio is >= 1/(2*den) >= 2^-38
//     from any half-integer (den = m_s * 2^shift <= 2^37), while the f64
//     quotient error is <= 2^-45 (ratio < 2^8): the double rounding equals
//     the exact-rational RNE computed here (integer division + remainder).
//   P5 (clamps): with s derived from the row's own amax, |x|/s < 127.5, so
//     the +/-127/-128 clamps and the w >= 18 window are DEFENSIVE
//     (unreachable in contract). Kept — never weaken checks.
//   Outside C1/C2 the hardware remains deterministic (documented clamp /
//   truncation) but bit-equality with the f64 golden is NOT claimed — that
//   is exactly what the sticky error bits report.
//
// ── JOB / STREAM SEMANTICS (§5, D-006, §3 reject discipline) ───────────────
//   job = {mode, cols}: 1 <= cols <= D, else job_error pulse+sticky, NO
//   state change. Flow: load `cols` sideband scales FIRST (so the host can
//   pre-load them before upstream data exists — deadlock-free sequencing),
//   then consume `cols` v elements. v `last` must mark exactly element
//   cols-1 (frame_error pulse+sticky otherwise; processing continues).
//   done <=> every output beat of the job ACCEPTED downstream post-skid;
//   job_ready gated until strictly after done; busy = (FSM != IDLE) ||
//   beats pending in any output skid. ALL five streams are skid-buffered.

module apex_scale_quant
  import apex_pkg::*;
#(
  parameter int unsigned D = 64            // max elements (columns) per job
)(
  input  logic               clk,
  input  logic               rst_n,        // synchronous, active low

  // job interface (D-006)
  input  logic               job_valid,
  output logic               job_ready,
  input  logic               job_mode,     // 0 = F16 (S-2), 1 = QUANT (Q7/S-4)
  input  logic [DIM_W-1:0]   job_cols,
  output logic               job_error,
  output logic               job_error_sticky,
  output logic               busy,
  output logic               done,

  // fp32 scale sideband in (one constant per element, loaded first)
  input  logic               cs_valid,
  output logic               cs_ready,
  input  logic [31:0]        cs_data,

  // INT32 value stream in
  input  logic               v_valid,
  output logic               v_ready,
  input  logic signed [31:0] v_data,
  input  logic               v_last,

  // MODE_F16 output: fp16 elements (KVQ s_axis shape; last on element cols-1)
  output logic               f_valid,
  input  logic               f_ready,
  output logic [15:0]        f_data,
  output logic               f_last,

  // MODE_QUANT outputs: packed INT8 code beats + fp16 row scale
  output logic               q_valid,
  input  logic               q_ready,
  output lane8_beat_t        q_beat,
  output logic               s_valid,
  input  logic               s_ready,
  output logic [15:0]        s_data,
  output logic               s_last,       // always 1 (one row per job)

  // contract monitors (golden-domain guards; pulse + sticky)
  output logic               range_error,  // C1: |v| > 2^24
  output logic               range_error_sticky,
  output logic               scale_error,  // C2: sideband not fp16-grade normal
  output logic               scale_error_sticky,
  output logic               frame_error,  // v `last` mis-framed
  output logic               frame_error_sticky
);

  if (D != 64 && D != 128) begin : g_chk_d
    $error("apex_scale_quant: D must be 64 or 128");
  end

  localparam int unsigned CW    = 8;              // element counters (<=128)
  localparam int unsigned NBW   = 5;              // beat counters (<=16)
  localparam logic [31:0] MAG_MAX = 32'h0100_0000; // 2^24 (C1)

  // ── input skids (§5) ──────────────────────────────────────────────────────
  logic        c_valid, c_ready;
  logic [31:0] c_bus;

  stream_skid #(.WIDTH(32)) u_cs_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (cs_valid),
    .s_ready (cs_ready),
    .s_data  (cs_data),
    .m_valid (c_valid),
    .m_ready (c_ready),
    .m_data  (c_bus)
  );

  logic        i_valid, i_ready;
  logic [32:0] i_bus;
  logic signed [31:0] i_v;
  logic               i_last;

  stream_skid #(.WIDTH(33)) u_v_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (v_valid),
    .s_ready (v_ready),
    .s_data  ({v_data, v_last}),
    .m_valid (i_valid),
    .m_ready (i_ready),
    .m_data  (i_bus)
  );
  assign i_v    = $signed(i_bus[32:1]);
  assign i_last = i_bus[0];

  // ── output skids ──────────────────────────────────────────────────────────
  logic        fo_valid, fo_ready;
  logic [16:0] fo_bus, f_vec;

  stream_skid #(.WIDTH(17)) u_f_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (fo_valid),
    .s_ready (fo_ready),
    .s_data  (fo_bus),
    .m_valid (f_valid),
    .m_ready (f_ready),
    .m_data  (f_vec)
  );
  assign f_data = f_vec[16:1];
  assign f_last = f_vec[0];

  logic                           qo_valid, qo_ready;
  lane8_beat_t                    qo_beat;
  logic [$bits(lane8_beat_t)-1:0] q_vec;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_q_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (qo_valid),
    .s_ready (qo_ready),
    .s_data  ({qo_beat.data, qo_beat.last}),
    .m_valid (q_valid),
    .m_ready (q_ready),
    .m_data  (q_vec)
  );
  assign q_beat = lane8_beat_t'(q_vec);

  logic        so_valid, so_ready;
  logic [16:0] s_vec;

  stream_skid #(.WIDTH(17)) u_s_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (so_valid),
    .s_ready (so_ready),
    .s_data  ({sbits, 1'b1}),
    .m_valid (s_valid),
    .m_ready (s_ready),
    .m_data  (s_vec)
  );
  assign s_data = s_vec[16:1];
  assign s_last = s_vec[0];

  // ── sideband decode (C2) ──────────────────────────────────────────────────
  // stored entry: {bad, ec[7:0], m11[10:0]} — bad => arithmetic zero result.
  logic        cd_bad, cd_grade_viol;
  logic [7:0]  cd_ec;
  logic [10:0] cd_m11;
  assign cd_ec         = c_bus[30:23];
  assign cd_m11        = {1'b1, c_bus[22:13]};
  assign cd_bad        = c_bus[31] || (cd_ec == 8'h00) || (cd_ec == 8'hFF);
  assign cd_grade_viol = cd_bad || (c_bus[12:0] != 13'd0);

  // ── helpers ───────────────────────────────────────────────────────────────
  function automatic logic [5:0] msb35(input logic [34:0] p);
    msb35 = 6'd0;
    for (int i = 0; i < 35; i++) begin
      if (p[i]) msb35 = 6'(i);
    end
  endfunction

  // exact product magnitude: P = |v| * m11  (<= 2^24 * 2047 < 2^35, proof P1)
  function automatic logic [34:0] prod35(input logic [24:0] mag,
                                         input logic [10:0] m11);
    return 35'(mag) * 35'(m11);
  endfunction

  // ── MODE_F16 element: fp16_RNE(sign * P * 2^(ec-137)) — proof P2 ─────────
  function automatic logic [15:0] f16_of(input logic        sgn,
                                         input logic [24:0] mag,
                                         input logic [10:0] m11,
                                         input logic [7:0]  ec,
                                         input logic        bad);
    logic [34:0] p;
    logic [5:0]  m, sh;
    int          e16, shr;
    logic [10:0] mant11;
    logic        rb, stk, inc, unused_hid;
    logic [11:0] m12, fl12, r11;
    if (bad)        return 16'h0000;
    if (mag == '0)  return {sgn, 15'b0};
    p   = prod35(mag, m11);
    m   = msb35(p);
    e16 = int'({26'b0, m}) + int'({24'b0, ec}) - 122;
    if (e16 >= 31) return {sgn, 5'h1F, 10'b0};            // overflow -> inf
    if (e16 >= 1) begin                                   // normal
      sh         = m - 6'd10;
      mant11     = 11'(p >> sh);
      rb         = (sh != 6'd0) && p[sh - 6'd1];
      stk        = (sh > 6'd1) && ((p & ((35'd1 << (sh - 6'd1)) - 35'd1)) != '0);
      inc        = rb && (stk || mant11[0]);
      m12        = {1'b0, mant11} + {11'b0, inc};
      unused_hid = m12[10];                               // hidden bit
      if (m12[11]) begin                                  // 2.0 -> 1.0 * 2^+1
        e16 = e16 + 1;
        if (e16 >= 31) return {sgn, 5'h1F, 10'b0};
        return {sgn, 5'(e16), 10'b0};
      end
      return {sgn, 5'(e16), m12[9:0]};
    end
    // subnormal / underflow: R = RNE(P * 2^(ec-113)) in 2^-24 units, <= 1024
    shr = 113 - int'({24'b0, ec});                        // >= m-9 >= 1
    if (shr >= 36) return {sgn, 15'b0};
    fl12 = 12'(p >> shr);                                 // fl <= 1023 here
    rb   = p[shr - 1];
    stk  = (shr > 1) && ((p & ((35'd1 << (shr - 1)) - 35'd1)) != '0);
    inc  = rb && (stk || fl12[0]);
    r11  = fl12 + {11'b0, inc};                           // <= 1024
    return {sgn, 15'(r11)};
  endfunction

  // ── C-1 scale from the exact row amax — proof P3 ──────────────────────────
  // amax = a_pn * 2^(a_exp - 172), a_pn left-normalized (bit 35 set).
  function automatic logic [15:0] scale_calc(input logic [8:0]  a_exp,
                                             input logic [35:0] a_pn);
    logic [49:0] num;
    logic [43:0] q;
    logic [6:0]  r;
    logic [10:0] mant11;
    logic        rb, stk, inc, unused_hid;
    logic [11:0] m12;
    int          es;
    if (a_pn == '0) return 16'h0400;                      // amax 0 -> EPS
    num = {a_pn, 14'b0};                                  // 14 guard bits
    q   = 44'(num / 50'd127);
    r   = 7'(num % 50'd127);
    if (q[43]) begin
      mant11 = q[43:33];
      rb     = q[32];
      stk    = (q[31:0] != '0) || (r != '0);
      es     = int'({23'b0, a_exp}) - 143;
    end else begin
      mant11 = q[42:32];
      rb     = q[31];
      stk    = (q[30:0] != '0) || (r != '0);
      es     = int'({23'b0, a_exp}) - 144;
    end
    if (es < -14) return 16'h0400;                        // EPS floor (P3)
    inc        = rb && (stk || mant11[0]);
    m12        = {1'b0, mant11} + {11'b0, inc};
    unused_hid = m12[10];                                 // hidden bit
    if (m12[11]) begin                                    // mantissa carry
      m12 = 12'h400;
      es  = es + 1;
    end
    if (es > 15) return 16'h7C00;                         // fp16 overflow
    return {1'b0, 5'(es + 15), m12[9:0]};
  endfunction

  // ── C-1 code: clamp(RNE(x / s)) — proofs P4/P5 ────────────────────────────
  // 2026-08-07 (the A0 62.5 MHz timing hunt): the one-cycle form —
  // vmem/cmem read -> 25x11 product -> 42/37 combinational DIVIDE -> round
  // -> pack — measured 39.2 ns / 249 logic levels / 193 CARRY8 at post-route
  // (the top-10 violated paths of the first A0 build, all into pack_reg).
  // Split at the num/den boundary into a 3-stage drain pipeline (read ->
  // code_pre -> code_div). The composition code_div(code_pre(x)) is the OLD
  // code_of expression-for-expression; only a register sits between them.
  // The divide is an explicit 9-iteration restoring divider: the quotient
  // provably fits 9 bits (w <= 17 => qd <= 255 pre-round), which the
  // generic `/` operator cannot know — it built the full 42-stage array.
  function automatic logic [81:0] code_pre(input logic        sgn,
                                           input logic [24:0] mag,
                                           input logic [10:0] m11,
                                           input logic [7:0]  ec,
                                           input logic        bad,
                                           input logic [15:0] s16,
                                           input logic        pad);
    logic [34:0] p;
    logic [5:0]  m;
    logic [10:0] m_s;
    logic [4:0]  es5;
    int          del, w;
    logic [41:0] num;
    logic [36:0] den;
    logic        zero, sat;
    m_s  = {1'b1, s16[9:0]};
    es5  = s16[14:10];                                    // >= 1 (s >= EPS)
    p    = prod35(mag, m11);
    m    = msb35(p);
    del  = int'({24'b0, ec}) - int'({27'b0, es5}) - 112;
    w    = int'({26'b0, m}) + del;
    zero = pad || bad || (mag == '0) || (s16 == 16'h7C00) // inf scale (P5/P6)
           || (w <= 8);                                   // |x/s| < 1/2
    sat  = !zero && (w >= 18);                            // |x/s| > 128 (P5)
    if (zero || sat) begin
      num = '0;                                           // keep stage C benign
      den = 37'd1;
    end else if (del >= 0) begin
      num = 42'(p) << del;                                // < 2^(w+1) <= 2^18
      den = 37'(m_s);
    end else begin
      num = 42'(p);
      den = 37'(m_s) << (-del);                           // -del <= m-9 <= 25
    end
    return {num, den, sgn, zero, sat};
  endfunction

  function automatic logic signed [7:0] code_div(input logic [41:0] num,
                                                 input logic [36:0] den,
                                                 input logic        sgn,
                                                 input logic        zero,
                                                 input logic        sat);
    logic [45:0] r;                                       // running remainder
    logic [45:0] dsh;
    logic [8:0]  qd;
    logic [36:0] rd;
    logic        inc;
    logic [9:0]  mag9;
    if (zero) return 8'sd0;
    if (sat)  return sgn ? 8'sh80 : 8'sd127;
    r = 46'(num);
    for (int i = 8; i >= 0; i--) begin                    // restoring divide,
      dsh = 46'(den) << i;                                // 9 quotient bits
      if (r >= dsh) begin
        r     = r - dsh;
        qd[i] = 1'b1;
      end else qd[i] = 1'b0;
    end
    rd  = r[36:0];                                        // num % den
    inc = ({rd, 1'b0} > {1'b0, den}) ||
          (({rd, 1'b0} == {1'b0, den}) && qd[0]);         // half to even
    mag9 = {1'b0, qd} + {9'b0, inc};                      // <= 256
    if (mag9 >= 10'd128) return sgn ? 8'sh80 : 8'sd127;   // clamp (P5)
    return sgn ? -$signed({1'b0, mag9[6:0]}) : $signed({1'b0, mag9[6:0]});
  endfunction

  // ── FSM / state ───────────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    ST_IDLE   = 3'd0,
    ST_LOAD   = 3'd1,   // load per-element sideband scales
    ST_INGEST = 3'd2,   // consume v elements (F16: convert+emit; QUANT: store)
    ST_SCALE  = 3'd3,   // QUANT: compute the C-1 fp16 row scale
    ST_SPUSH  = 3'd4,   // QUANT: offer the scale to its skid
    ST_DRAIN  = 3'd5,   // QUANT: quantize + pack + push beats
    ST_WAIT   = 3'd6    // D-006: wait for post-skid acceptance
  } state_e;

  state_e           state;
  logic             mode_q;
  logic [CW-1:0]    cols_q, load_cnt, idx, f_acc, out_slot;
  logic [NBW-1:0]   nbeats, beat_idx, q_acc;
  logic             s_acc;
  logic             have;
  logic [15:0]      sbits;
  logic [63:0]      pack;
  logic [3:0]       pack_cnt;
  // drain pipeline (2026-08-07 A0 timing): A=read, B=code_pre, C=code_div
  logic [25:0]      p1_v;
  logic [19:0]      p1_c;
  logic             p1_pad, p1_val;
  logic [41:0]      p2_num;
  logic [36:0]      p2_den;
  logic             p2_sgn, p2_zero, p2_sat, p2_val;
  logic [8:0]       a_exp;
  logic [35:0]      a_pn;
  // element holds: {sign, mag25} and {bad, ec, m11}
  logic [25:0]      vmem [D];
  logic [19:0]      cmem [D];
  logic [25:0]      vhold;
  logic [19:0]      chold;

  logic legal;
  assign legal = (job_cols != '0) && (job_cols <= DIM_W'(D));

  // ingest decode
  logic        in_sgn;
  logic [31:0] in_mag32;
  logic        in_rng;
  logic [24:0] in_mag;
  assign in_sgn   = i_v[31];
  assign in_mag32 = in_sgn ? (~$unsigned(i_v) + 32'd1) : $unsigned(i_v);
  assign in_rng   = (in_mag32 > MAG_MAX);                 // C1 violation
  assign in_mag   = in_rng ? 25'(MAG_MAX) : in_mag32[24:0];

  // amax candidate (QUANT ingest)
  logic [19:0] in_c;
  logic        in_zero;
  logic [34:0] in_p;
  logic [5:0]  in_m;
  logic [8:0]  cand_exp;
  logic [35:0] cand_pn;
  logic        cand_better;
  assign in_c        = cmem[idx[$clog2(D)-1:0]];
  assign in_zero     = in_c[19] || (in_mag == '0);        // bad c or v == 0
  assign in_p        = prod35(in_mag, in_c[10:0]);
  assign in_m        = msb35(in_p);
  assign cand_exp    = {3'b0, in_m} + {1'b0, in_c[18:11]};
  assign cand_pn     = 36'(in_p) << (6'd35 - in_m);
  assign cand_better = (cand_exp > a_exp) ||
                       ((cand_exp == a_exp) && (cand_pn > a_pn));

  // drain read (stage A of the pipeline; one element per cycle sustained)
  logic [25:0] dr_v;
  logic [19:0] dr_c;
  assign dr_v = vmem[out_slot[$clog2(D)-1:0]];
  assign dr_c = cmem[out_slot[$clog2(D)-1:0]];

  // handshakes
  assign c_ready = (state == ST_LOAD);
  assign i_ready = (state == ST_INGEST) && (mode_q ? 1'b1 : !have);

  assign fo_valid = (state == ST_INGEST) && !mode_q && have;
  assign fo_bus   = {f16_of(vhold[25], vhold[24:0], chold[10:0],
                            chold[18:11], chold[19]),
                     (idx == cols_q - CW'(1))};

  assign so_valid     = (state == ST_SPUSH);
  assign qo_valid     = (state == ST_DRAIN) && (pack_cnt == 4'd8);
  assign qo_beat.data = pack;
  assign qo_beat.last = (beat_idx == nbeats - NBW'(1));

  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || f_valid || q_valid || s_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state              <= ST_IDLE;
      done               <= 1'b0;
      job_error          <= 1'b0;
      job_error_sticky   <= 1'b0;
      range_error        <= 1'b0;
      range_error_sticky <= 1'b0;
      scale_error        <= 1'b0;
      scale_error_sticky <= 1'b0;
      frame_error        <= 1'b0;
      frame_error_sticky <= 1'b0;
      mode_q             <= 1'b0;
      cols_q             <= '0;
      load_cnt           <= '0;
      idx                <= '0;
      f_acc              <= '0;
      out_slot           <= '0;
      nbeats             <= '0;
      beat_idx           <= '0;
      q_acc              <= '0;
      s_acc              <= 1'b0;
      have               <= 1'b0;
      sbits              <= '0;
      pack               <= '0;
      pack_cnt           <= '0;
      a_exp              <= '0;
      a_pn               <= '0;
      vhold              <= '0;
      chold              <= '0;
      p1_v               <= '0;
      p1_c               <= '0;
      p1_pad             <= 1'b0;
      p1_val             <= 1'b0;
      p2_num             <= '0;
      p2_den             <= '0;
      p2_sgn             <= 1'b0;
      p2_zero            <= 1'b0;
      p2_sat             <= 1'b0;
      p2_val             <= 1'b0;
    end else begin
      done        <= 1'b0;
      job_error   <= 1'b0;
      range_error <= 1'b0;
      scale_error <= 1'b0;
      frame_error <= 1'b0;
      if (f_valid && f_ready) f_acc <= f_acc + CW'(1);
      if (q_valid && q_ready) q_acc <= q_acc + NBW'(1);
      if (s_valid && s_ready) s_acc <= 1'b1;

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state    <= ST_LOAD;
              mode_q   <= job_mode;
              cols_q   <= CW'(job_cols);
              nbeats   <= NBW'((32'(job_cols) + 32'd7) >> 3);
              load_cnt <= '0;
              idx      <= '0;
              f_acc    <= '0;
              q_acc    <= '0;
              s_acc    <= 1'b0;
              have     <= 1'b0;
              a_exp    <= '0;
              a_pn     <= '0;
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
            if (mode_q) begin              // QUANT: store + track exact amax
              vmem[idx[$clog2(D)-1:0]] <= {in_sgn, in_mag};
              if (!in_zero && cand_better) begin
                a_exp <= cand_exp;
                a_pn  <= cand_pn;
              end
              if (idx == cols_q - CW'(1)) begin
                state <= ST_SCALE;
              end else begin
                idx <= idx + CW'(1);
              end
            end else begin                 // F16: hold element for emission
              vhold <= {in_sgn, in_mag};
              chold <= in_c;
              have  <= 1'b1;
            end
          end else if (!mode_q && have && fo_ready) begin
            have <= 1'b0;                  // element accepted into f16 skid
            if (idx == cols_q - CW'(1)) begin
              state <= ST_WAIT;
            end else begin
              idx <= idx + CW'(1);
            end
          end
        end

        ST_SCALE: begin
          sbits <= scale_calc(a_exp, a_pn);
          state <= ST_SPUSH;
        end

        ST_SPUSH: begin
          if (so_ready) begin              // scale accepted into its skid
            state    <= ST_DRAIN;
            out_slot <= '0;
            beat_idx <= '0;
            pack_cnt <= '0;
            p1_val   <= 1'b0;              // pipeline starts empty (2 prime
            p2_val   <= 1'b0;              // cycles before the first pack)
          end
        end

        ST_DRAIN: begin
          // 3-stage pipe: A reads, B pre-computes num/den, C divides+packs.
          // The ONLY stall is a full pack awaiting its skid; the stages HOLD
          // through it (their contents are the next pack's lead elements),
          // so sustained throughput stays one element per cycle.
          if (pack_cnt == 4'd8) begin
            if (qo_ready) begin            // beat accepted into codes skid
              pack_cnt <= '0;
              if (beat_idx == nbeats - NBW'(1)) begin
                state  <= ST_WAIT;
                p1_val <= 1'b0;            // discard the tail prefetch
                p2_val <= 1'b0;
              end else begin
                beat_idx <= beat_idx + NBW'(1);
              end
            end
          end else begin
            // stage C: pack the element staged two cycles ago
            if (p2_val) begin
              pack[8*pack_cnt[2:0] +: 8]
                <= code_div(p2_num, p2_den, p2_sgn, p2_zero, p2_sat);
              pack_cnt <= pack_cnt + 4'd1;
            end
            // stage B: everything up to the divide, registered
            {p2_num, p2_den, p2_sgn, p2_zero, p2_sat}
              <= code_pre(p1_v[25], p1_v[24:0], p1_c[10:0], p1_c[18:11],
                          p1_c[19], sbits, p1_pad);
            p2_val <= p1_val;
            // stage A: the RAM read (tail-beat padding decided at read time)
            p1_v   <= dr_v;
            p1_c   <= dr_c;
            p1_pad <= !(out_slot < cols_q);
            p1_val <= 1'b1;
            out_slot <= out_slot + CW'(1);
          end
        end

        ST_WAIT: begin                     // D-006: post-skid acceptance
          if (mode_q ? ((q_acc == nbeats) && s_acc)
                     : (f_acc == cols_q)) begin
            done  <= 1'b1;
            state <= ST_IDLE;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
