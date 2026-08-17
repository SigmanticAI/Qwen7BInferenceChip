// residual_add_fx.sv — SYNTHESIZABLE integer form of the C-6 residual
// primitive:  y = f16(a + b), ONE round-half-to-even narrowing.
//
// Replaces rtl/misc/residual_add.sv's SV-`real` datapath (simulation-only,
// IB_LAYER.md §1 reality check 1) for the apex_top integration. The
// behavioral block REMAINS the cross-check reference: verif/misc/resfx
// co-simulates both and requires bit-identical fp16 outputs over the full
// finite domain sweeps, plus golden-fp expected vectors
// (f64_to_f16_bits(f16_bits_to_f64(a) + f16_bits_to_f64(b))).
//
// Numerics: both finite fp16 operands are exact integers on the 2^-24 grid
// (f16_arith_pkg v24 form). Their sum is exact in 44 bits; f16_pack_real
// performs the single RNE (exp = -24). No other rounding exists.
//
// IEEE zero-sign rule (matches the behavioral double add, RNE mode):
//   nonzero result       : sign of the exact sum
//   exact-zero result    : -0 iff BOTH addends are -0 (else +0; includes
//                          the nonzero-cancellation case a = -b -> +0)
// Overflow (|sum| > fp16 max) saturates to +-inf via pack's E>15 path —
// the same fp16-native saturation as the behavioral block.
//
// Domain: finite fp16 (the C-6 bus never carries inf/NaN; behavioral
// header documents the same precondition).
//
// Timing: purely combinational, mirroring residual_add.sv; the integrating
// datapath registers around it (ARCHITECTURE.md §0 discipline).

module residual_add_fx
  import f16_arith_pkg::*;
(
  input  logic [15:0] a,     // fp16 bits, incoming residual element
  input  logic [15:0] b,     // fp16 bits, sublayer output element
  output logic [15:0] y      // fp16 bits, f16(a + b), single RNE
);

  logic signed [F16_V24_W:0]   sum;      // 44-bit signed: exact a+b on 2^-24
  logic        [F16_SIG_W-1:0] mag;
  logic                        sgn;

  always_comb begin
    sum = $signed({f16_to_v24(a)[F16_V24_W-1], f16_to_v24(a)})
        + $signed({f16_to_v24(b)[F16_V24_W-1], f16_to_v24(b)});
    if (sum == '0)
      sgn = a[15] && b[15];              // -0 only from (-0) + (-0)
    else
      sgn = sum < 0;
    mag = (sum < 0) ? F16_SIG_W'(-sum) : F16_SIG_W'(sum);
    y   = f16_pack_real(sgn, mag, -24);
  end

endmodule
