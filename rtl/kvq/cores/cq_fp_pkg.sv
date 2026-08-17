// cq_fp_pkg.sv — clean-room fp16/fp32 numeric primitives for the KVQ
// per-channel INT4 KV datapath (APEX ARCHITECTURE.md §2 C-1/C-2, D-001/D-010).
//
// SYNTHESIZABLE, pure-INTEGER, bit-exact twin of golden/apex_golden/
// {fp.py,cq_codec.py}. This file replaces the earlier `real` (float64)
// behavioral implementation: every fp16 operand is decoded to an exact
// integer significand/exponent pair, all arithmetic is exact integer
// shift/subtract, and every narrowing performs round-half-to-even (RNE)
// from an EXACT integer remainder — so there is no float intermediate
// and no double-rounding to reason about. The op-for-op Python mirror is
// verif/kvq/fparith/int_model.py, proven EXHAUSTIVELY against the golden
// arbiter by verif/kvq/fparith/prove.py; the SV below is re-proven in HW by
// verif/kvq/fparith/tb_fparith.sv (exhaustive sweeps + seeded-error checks).
//
// Decomposition of an fp16 bit pattern b = {s, e[4:0], m[9:0]}:
//   |x| = M * 2^E   with M in [0, 2047]:
//     e == 0      : M = m,        E = -24    (zero / subnormal)
//     1 <= e <= 30: M = 1024 + m, E = e - 25 (normal, implicit 1)
//     e == 31     : M = 2047,     E = 5      (inf/nan: OUT OF CONTRACT — the
//                   codec never produces them; defensively treated as 65504)
//
// F1.5 AREA REWORK (the seam the original header documented): the general
// 64/64 `$div`+`$mul` RNE divide (rne_div_u) is REPLACED by rne_div_bounded,
// a 12-step restoring subtract-compare array with NO divide and NO multiply
// cells. The bound is exact where the callers need exactness:
//   * scale_from_amax's quotient is the fp16 significand candidate, provably
//     in [1024, 2048] — well under the 2^12 bound, always the exact path;
//   * quant_one's quotient is clamped to [qmin, qmax] ⊆ [-128, 127]
//     afterwards, so any true quotient >= 2^12 saturates to 2^12 and clamps
//     to the same code the exact value would (proven exhaustively by
//     verif/kvq/fparith against the golden model).
// The exact remainder falls out of the restoring walk, so the single RNE
// tie decision (mutation target: mut_tie) is unchanged in meaning.
// All functions remain combinational (single-cycle); the per-channel
// SHARING of these units is done by the cq_value_path / cq_key_path FSMs,
// which now walk one channel per cycle through ONE scale and ONE quant unit
// instead of instantiating D of each (see those files).
//
// S10 PIPELINE REFACTOR: rne_div_bounded / scale_from_amax / quant_one are
// each recomposed from front/step/back stage functions defined here, so the
// registered pipelines (cq_rne_div_pipe / cq_quant_pipe / cq_scale_pipe) and
// these combinational twins are compositions of the SAME functions, differing
// only in register placement — the fparith exhaustive proof of the twins is
// the bit-exactness certificate for the pipes.
//
// This is a genuinely independent implementation: the decomposition (integer
// significand/exponent decode + a single exact-remainder RNE per narrowing)
// is my own; only the numeric RESULTS are contracted to match the golden
// vectors.

`ifndef CQ_FP_PKG_SV
`define CQ_FP_PKG_SV

package cq_fp_pkg;

  // EPS = 2^-14 (contract §1, FINAL) == the smallest normal fp16, 0x0400.
  localparam logic [15:0] EPS_F16 = 16'h0400;

  // symmetric signed quant limits for `bits` (C-1): qmax = 2^(b-1)-1.
  function automatic longint qmax_of(input int bits);
    qmax_of = (longint'(1) << (bits - 1)) - 1;
  endfunction
  function automatic longint qmin_of(input int bits);
    qmin_of = -(longint'(1) << (bits - 1));
  endfunction

  // fp16 magnitude bits [14:0] -> integer significand M in [0,2047]
  // (|x| = M * 2^E).
  function automatic logic [10:0] f16_sig(input logic [14:0] b);
    logic [4:0] e;
    begin
      e = b[14:10];
      if (e == 5'd0)       f16_sig = {1'b0, b[9:0]};   // zero / subnormal
      else if (e == 5'd31) f16_sig = 11'd2047;         // inf/nan (defensive)
      else                 f16_sig = {1'b1, b[9:0]};   // normal
    end
  endfunction

  // fp16 exponent field -> exponent E of the significand's LSB (|x| = M*2^E).
  function automatic int f16_exp(input logic [4:0] e);
    begin
      if (e == 5'd0)       f16_exp = -24;
      else if (e == 5'd31) f16_exp = 5;                // inf/nan (defensive)
      else                 f16_exp = int'({27'd0, e}) - 25;
    end
  endfunction

  // round-half-to-even of the EXACT nonnegative rational n/d, BOUNDED:
  // exact for floor(n/d) < 2^12, else saturates to 2^12 (callers only need
  // exactness below their clamp/normalization bound — see the header).
  // Implementation is a 12-step restoring compare-subtract array: NO $div,
  // NO $mul; the exact remainder survives the walk, so the tie test
  // (r2 == dd) is exact — no float anywhere, hence no double rounding.
  // THE tie line (in div_back) is the single RNE decision point shared by
  // scale_from_amax and quant_one (mutation target: mut_tie).
  // d == 0 (out of contract: the codec scale is always >= EPS) hits the
  // saturation branch deterministically.
  //
  // S10 PIPELINE SPLIT: the walk is decomposed into per-stage functions —
  // div_front (saturation pre-compare + rem/q init), div_step (ONE restoring
  // step at a constant bit index i) and div_back (the single RNE decision) —
  // so the SAME integer steps can be composed two ways: combinationally
  // (rne_div_bounded below, byte-identical results, re-certified exhaustively
  // by verif/kvq/fparith) and as a registered pipeline (cq_rne_div_pipe.sv,
  // one div_step per stage). NO pipe-private arithmetic exists: every stage
  // body IS one of these functions, so the exhaustive proof of the comb twin
  // is the proof of the pipe.
  typedef struct packed {
    logic [53:0] rem;   // running remainder
    logic [53:0] dd;    // widened divisor (stage shifts are wired constants)
    logic [12:0] q;     // quotient bits assembled MSB-first
    logic        sat;   // quotient >= 2^12: bypass the walk, emit 4096
  } div_state_t;

  // stage 0: saturation pre-compare + remainder/quotient init
  function automatic div_state_t div_front(input logic [40:0] n,
                                           input logic [40:0] d);
    div_state_t s;
    begin
      s.dd  = {13'd0, d};
      s.sat = ({13'd0, n} >= (s.dd << 12));
      s.rem = {13'd0, n};
      s.q   = 13'd0;
      div_front = s;
    end
  endfunction

  // stages 1..12: one restoring compare-subtract step at constant index i
  // (i = 11 down to 0); a saturated walk passes through untouched.
  function automatic div_state_t div_step(input div_state_t s, input int i);
    div_state_t t;
    begin
      t = s;
      if (!s.sat && (s.rem >= (s.dd << i))) begin
        t.rem  = s.rem - (s.dd << i);
        t.q[i] = 1'b1;
      end
      div_step = t;
    end
  endfunction

  // stage 13: the RNE tie decision on the exact remainder (mut_tie target)
  function automatic logic [12:0] div_back(input div_state_t s);
    logic [53:0] rem, dd, r2;
    logic [12:0] q;
    begin
      if (s.sat) begin
        div_back = 13'd4096;                // saturated: quotient >= 2^12
      end else begin
        rem = s.rem;
        dd  = s.dd;
        q   = s.q;
        r2 = rem << 1;                      // exact remainder, rem < d
        if (r2 > dd)                 q = q + 13'd1;
        else if ((r2 == dd) && q[0]) q = q + 13'd1;   // RNE tie -> nearest even
        div_back = q;
      end
    end
  endfunction

  // combinational recomposition: front + 12x step + back — the bit-exact
  // twin of cq_rne_div_pipe (identical functions, registers removed).
  function automatic logic [12:0] rne_div_bounded(input logic [40:0] n,
                                                  input logic [40:0] d);
    div_state_t s;
    int i;
    begin
      s = div_front(n, d);
      for (i = 11; i >= 0; i = i - 1) s = div_step(s, i);
      rne_div_bounded = div_back(s);
    end
  endfunction

  // index of the most-significant set bit (input != 0). Fixed-bound loop —
  // synthesizes to a priority encoder.
  function automatic int msb_idx(input logic [63:0] m);
    int i;
    begin
      msb_idx = 0;
      for (i = 1; i < 64; i = i + 1)
        if (m[i]) msb_idx = i;
    end
  endfunction

  // ── codec primitives (mirror cq_codec.py §1) ────────────────────────────

  // scale = max(amax/qmax, EPS) -> fp16 bits, RNE (scale_from_amax /
  // cq_scale_unit). amax is a sign-cleared fp16 magnitude (amax reduce);
  // a set sign bit is honored defensively the way the golden float chain
  // would: a negative amax gives a negative quotient < EPS, so s = EPS.
  // Exact integer scheme: the quotient binade
  // j = floor(log2(Ma/qmax)) comes from an MSB compare (j is msb(Ma)-msb(q)
  // or one less), then the 11-bit significand of the result is
  // RNE(Ma * 2^(10-j) / qmax) with the exact remainder deciding ties.
  // Dividend Ma<<(10-j) <= 2047<<17 < 2^28 (j >= -7 for qmax >= 7); divisor
  // is the small constant qmax (7 or 127); the quotient is in [1024, 2048],
  // far below rne_div_bounded's 2^12 exactness bound.
  // S10 PIPELINE SPLIT (same scheme as rne_div_bounded): scale_front carries
  // the f16 decode + binade search + shifted dividend and the EPS/zero
  // short-circuit as a bypass flag; scale_back does the m==2048 renormalize +
  // fp16 pack + EPS mux. scale_from_amax below recomposes them around the
  // shared divider — the comb twin of cq_scale_pipe.sv.
  typedef struct packed {
    logic               eps;   // amax <= 0 or e <= -15: result is EPS_F16
    logic signed [31:0] e;     // quotient binade (int, carried to the back)
    logic [40:0]        n;     // shifted dividend Ma << sh
    logic [40:0]        d;     // divisor qmax (possibly shifted, defensive)
  } scale_front_t;

  function automatic scale_front_t scale_front(input logic [15:0] amax_f16,
                                               input int bits);
    scale_front_t f;
    logic [10:0] ma;
    int          ea, k, l, j0, j, e, sh;
    logic [40:0] q;
    begin
      ma = f16_sig(amax_f16[14:0]);
      ea = f16_exp(amax_f16[14:10]);
      q  = 41'(qmax_of(bits));
      f.eps = 1'b0;
      f.e   = 32'sd0;
      f.n   = 41'd0;
      f.d   = q;
      if ((ma == 11'd0) || amax_f16[15]) begin
        f.eps = 1'b1;                            // amax <= 0 -> s = EPS
      end else begin
        // binade j: q*2^j <= Ma < q*2^(j+1); j is j0=msb(Ma)-msb(q) or j0-1
        k  = msb_idx(64'(ma));
        l  = msb_idx(64'(q));
        j0 = k - l;
        if (j0 >= 0) j = (41'(ma)          >= (q << j0)) ? j0 : j0 - 1;
        else         j = ((41'(ma) << (-j0)) >= q      ) ? j0 : j0 - 1;
        e = j + ea;                              // binade of a = amax/qmax
        // EPS floor: a < 2^-14 <=> e <= -15 -> s = EPS exactly (golden takes
        // max() BEFORE the fp16 narrowing, so no rounding happens here).
        // (mutation target: mut_eps)
        if (e <= -15) begin
          f.eps = 1'b1;
        end else begin
          // normal fp16: m = RNE(Ma * 2^(10-j) / qmax) in [1024, 2048]
          sh  = 10 - j;                          // >= 0 for qmax >= 4 (j <= 8)
          f.e = e;
          if (sh >= 0) begin f.n = 41'(ma) << sh; f.d = q;          end
          else         begin f.n = 41'(ma);       f.d = q << (-sh); end   // defensive
        end
      end
      scale_front = f;
    end
  endfunction

  function automatic logic [15:0] scale_back(input logic [12:0] m_in,
                                             input logic signed [31:0] e_in,
                                             input logic eps);
    logic [12:0] m;
    int          e;
    begin
      if (eps) begin
        scale_back = EPS_F16;
      end else begin
        m = m_in;
        e = int'(e_in);
        if (m == 13'd2048) begin e = e + 1; m = 13'd1024; end
        // e in [-14, 14] here (max amax 65504 / qmax 7 -> e 13, +1 carry):
        // always a normal fp16, never overflows to inf.
        scale_back = {1'b0, 5'(e + 15), 10'(m - 13'd1024)};
      end
    end
  endfunction

  function automatic logic [15:0] scale_from_amax(input logic [15:0] amax_f16,
                                                  input int bits);
    scale_front_t f;
    logic [12:0]  m;
    begin
      f = scale_front(amax_f16, bits);
      m = rne_div_bounded(f.n, f.d);
      scale_from_amax = scale_back(m, f.e, f.eps);
    end
  endfunction

  // q = clamp(RNE(x/s), qmin, qmax) (quant_codes / cq_quant_unit). Signed
  // code. Exact fp16/fp16 divide: x/s = (Mx/Ms) * 2^(Ex-Es); the shift
  // Ex-Es is in [-29, 29], so the dividend is at most 2047<<29 < 2^40 (or
  // the divisor at most 2047<<29 < 2^40) and the exact remainder drives the
  // RNE tie (golden: apex_golden.cq_codec.quant_codes). The bounded divide
  // saturates any quotient >= 2^12 — the clamp below maps it to the same
  // code the exact quotient would (|clamp bound| <= 128 < 2^12).
  // S10 PIPELINE SPLIT (same scheme): quant_front carries the sign, the f16
  // decode, the Ex-Es shift pick (the 41-bit barrel) and the mx==0 flag as a
  // bypass bit; quant_back does the negate + clamp. quant_one below is the
  // comb twin of cq_quant_pipe.sv.
  typedef struct packed {
    logic        neg;    // quotient sign = sign(x) XOR sign(s)
    logic        zero;   // mx == 0: +-0 -> code 0 (bypass)
    logic [40:0] n;      // dividend (Mx, shifted when Ex-Es >= 0)
    logic [40:0] d;      // divisor  (Ms, shifted when Ex-Es <  0)
  } quant_front_t;

  function automatic quant_front_t quant_front(input logic [15:0] x_f16,
                                               input logic [15:0] s_f16);
    quant_front_t f;
    logic [10:0] mx, ms;
    int          ex, es, sh;
    begin
      // quotient sign = sign(x) XOR sign(s); the contract scale is always
      // positive (>= EPS) but a signed s is honored the way the golden
      // float divide would. RNE on the magnitude == RNE on the signed value
      // (tie-to-even is symmetric about zero).
      f.neg  = x_f16[15] ^ s_f16[15];
      mx  = f16_sig(x_f16[14:0]);
      ex  = f16_exp(x_f16[14:10]);
      ms  = f16_sig(s_f16[14:0]);
      es  = f16_exp(s_f16[14:10]);
      f.zero = (mx == 11'd0);
      sh = ex - es;                              // in [-29, 29]
      if (sh >= 0) begin f.n = 41'(mx) << sh; f.d = 41'(ms);          end
      else         begin f.n = 41'(mx);       f.d = 41'(ms) << (-sh); end
      quant_front = f;
    end
  endfunction

  function automatic longint quant_back(input logic [12:0] mag,
                                        input logic neg,
                                        input logic zero,
                                        input int bits);
    longint q, lo, hi;
    begin
      if (zero) begin
        quant_back = 0;                          // +-0 -> code 0
      end else begin
        q  = neg ? -longint'(mag) : longint'(mag);
        lo = qmin_of(bits);
        hi = qmax_of(bits);
        if (q < lo) q = lo;
        if (q > hi) q = hi;
        quant_back = q;
      end
    end
  endfunction

  function automatic longint quant_one(input logic [15:0] x_f16,
                                       input logic [15:0] s_f16,
                                       input int bits);
    quant_front_t f;
    logic [12:0]  mag;
    begin
      f   = quant_front(x_f16, s_f16);
      mag = rne_div_bounded(f.n, f.d);
      quant_one = quant_back(mag, f.neg, f.zero, bits);
    end
  endfunction

  // x_hat = code * s -> fp32 bits (dequant_f32 / cq_dequant_unit, D-010 bus).
  // EXACT — no rounding at all: |code| <= 256 (9-bit signed port) and
  // Ms <= 2047 (11 bits), so the product has at most 20 significant bits and
  // the fp32 24-bit significand holds it exactly (proof of no double
  // rounding: the mantissa left-shift below never discards a bit). The
  // exponent es + k + 127 is in [103, 151] (es >= -24, k <= 19), so the
  // result is always a NORMAL fp32 — no subnormal or overflow case exists.
  // Signed zero: product 0 keeps sign = code_sign XOR s_sign (matches the
  // float64 0.0*-s / -0.0*s behavior of the golden). The product is a
  // width-exact 9x11 -> 20-bit multiply (F1.5: was a 64x64 $mul).
  //
  // The CQ-4+ outlier identity lane IS this function with code == +1
  // (kvq_engine stores raw fp16 with code +1): dequant_one(+1, raw) is the
  // exact f16 -> f32 widen, purely combinational (item proven exhaustively
  // by verif/kvq/fparith).
  function automatic logic [31:0] dequant_one(input logic signed [8:0] code,
                                              input logic [15:0] s_f16);
    logic        sign;
    logic [10:0] ms;
    logic [8:0]  ca;
    int          es, k;
    logic [19:0] p;
    begin
      ms = f16_sig(s_f16[14:0]);
      es = f16_exp(s_f16[14:10]);
      ca = code[8] ? 9'(-code) : 9'(code);        // |code| in [0, 256]
      p  = 20'(ca) * 20'(ms);                     // exact, < 2^20
      sign = code[8] ^ s_f16[15];
      if (p == 20'd0) begin
        dequant_one = {sign, 31'd0};              // signed zero
      end else begin
        k = msb_idx(64'(p));                      // 0..19
        // implicit 1 lands at bit 23; the 23' truncation keeps exactly the
        // fraction bits [22:0] and discards nothing else (see header proof).
        dequant_one = {sign, 8'(es + k + 127), 23'(43'(p) << (23 - k))};
      end
    end
  endfunction

endpackage

`endif
