// w4b_fp_pkg.sv — W4-B feeder arithmetic front (D-031).
//
// The W4-B chain is q8 = clamp(RNE((code4 · s_g) / s8), -128, 127) with
// code4 a 4-bit two's-complement weight code, s_g the fp16 GROUP scale and
// s8 the fp16 host-computed stripe requant scale (docs/design/W4B_FEEDER.md;
// golden: weight_codec.wfeed_w4b_to_i8).
//
// NO NEW ARITHMETIC PRIMITIVES: this package only builds the 41-bit
// dividend/divisor pair and defers to the fparith-certified cq_fp_pkg
// machinery (rne_div_bounded / cq_rne_div_pipe + quant_back). It is the
// quant_front of cq_fp_pkg §S10 with exactly one change: the dividend
// significand is |code|·Mg (4b × 11b <= 16376, 14 bits) instead of an fp16's
// 11-bit Mx — the exact product of the dequant (code·s_g is exact, see
// cq_fp_pkg dequant_one's proof; we never materialize the fp32, we divide
// the exact significands directly, so there is no double rounding anywhere).
//
// WIDTH PROOF (the one delta from quant_front): sh = Eg - E8 in [-29, 29].
//   sh >= 23: cm·2^sh >= 1·2^23 > 2047·2^12 >= M8·2^12 for every nonzero cm,
//             so the quotient ALWAYS saturates the bounded divider (mag 4096
//             -> quant_back clamps to the rail). We force n = all-ones,
//             which satisfies n >= d<<12 for every legal d (d <= 2047 <<
//             nothing here) — bit-identical result, no 14+29 = 43-bit
//             overflow ever constructed.
//   0 <= sh <= 22: n = cm << sh <= (2^14-1)·2^22 < 2^36 — fits 41 bits.
//   sh < 0:        d = M8 << (-sh) <= 2047·2^29 < 2^40 — fits (unchanged).
// Zero: cm == 0 (code 0 or subnormal-zero s_g) -> code 0, matching the
// golden rne(0/s) = 0. d == 0 (s8 = ±0) is out of contract (host scale is
// >= EPS) and hits the divider's deterministic saturation branch, same as
// the cq contract.
//
// The comb twin w4b_quant_one is the bit-exact composition used by the TB
// and the exhaustive sweep; the pipeline (mxe_wfeed_w4b.sv) composes the
// SAME functions with cq_rne_div_pipe stages in between — the fparith
// argument carries over exactly as it did for cq_quant_pipe.

`default_nettype none

package w4b_fp_pkg;

  // dividend/divisor build — returns cq_fp_pkg's own quant_front_t so the
  // downstream (divider + quant_back) is shared verbatim
  function automatic cq_fp_pkg::quant_front_t w4b_quant_front(
      input logic signed [3:0] code4,
      input logic [15:0]       sg_f16,
      input logic [15:0]       s8_f16);
    cq_fp_pkg::quant_front_t f;
    logic [10:0] mg, m8;
    logic [3:0]  cmag;
    logic [14:0] cm;                 // |code| * Mg <= 8*2047 = 16376
    int          eg, e8, sh;
    begin
      // |code| in 4 unsigned bits: two's-complement negate wraps -(-8) to
      // 4'b1000 = 8, exactly the magnitude we need — no special case
      cmag = code4[3] ? (4'd0 - $unsigned(code4)) : $unsigned(code4);
      mg = cq_fp_pkg::f16_sig(sg_f16[14:0]);
      eg = cq_fp_pkg::f16_exp(sg_f16[14:10]);
      m8 = cq_fp_pkg::f16_sig(s8_f16[14:0]);
      e8 = cq_fp_pkg::f16_exp(s8_f16[14:10]);

      cm     = 15'(cmag) * 15'(mg);
      f.neg  = code4[3] ^ sg_f16[15] ^ s8_f16[15];
      f.zero = (cm == 15'd0);
      sh     = eg - e8;                          // in [-29, 29]
      if (sh >= 23) begin
        // saturation shortcut (width proof above): quotient >= 2^12 always
        f.n = {41{1'b1}};
        f.d = 41'(m8);
      end else if (sh >= 0) begin
        f.n = 41'(cm) << sh;                     // <= 2^36, fits
        f.d = 41'(m8);
      end else begin
        f.n = 41'(cm);
        f.d = 41'(m8) << (-sh);                  // <= 2^40, fits (unchanged)
      end
      w4b_quant_front = f;
    end
  endfunction

  // comb twin of the full W4-B lane: front -> certified bounded divide ->
  // certified clamp/negate. Bit-exact vs golden wfeed_w4b_to_i8 per element.
  function automatic logic signed [8:0] w4b_quant_one(
      input logic signed [3:0] code4,
      input logic [15:0]       sg_f16,
      input logic [15:0]       s8_f16);
    cq_fp_pkg::quant_front_t f;
    logic [12:0] mag;
    begin
      f   = w4b_quant_front(code4, sg_f16, s8_f16);
      mag = cq_fp_pkg::rne_div_bounded(f.n, f.d);
      w4b_quant_one = 9'(cq_fp_pkg::quant_back(mag, f.neg, f.zero, 8));
    end
  endfunction

endpackage

`default_nettype wire
