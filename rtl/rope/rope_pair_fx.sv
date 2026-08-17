// rope_pair_fx.sv — SYNTHESIZABLE integer form of the C-ROPE channel-pair
// rotation:  y_i  = f16( x_i*cos(phi)  - x_ih*sin(phi) )
//            y_ih = f16( x_ih*cos(phi) + x_i*sin(phi)  )     (ONE RNE each)
//
// Replaces rtl/rope/rope.sv's SV-`real` rotation datapath (simulation-only,
// IB_LAYER.md §1 reality check 1). The cos/sin LUT + interpolation stage is
// byte-copied from rope.sv (same generated rope_lut_tables.svh — the golden
// provenance gate covers both consumers); only the rotation multiply moves
// from float64 to exact integer arithmetic:
//
//   x (fp16)          = v24 * 2^-24                (f16_arith_pkg, exact)
//   cos/sin (Q2.14)   = c_val * 2^-14              (signed 17-bit, LUT+interp)
//   product           = (v24 * c_val) * 2^-38      (exact, <= 56 bits)
//   sum/difference    : exact 57-bit signed
//   narrowing         : f16_pack_real(sign, |P|, -38) — the single RNE
//
// IEEE zero-sign rules (match the behavioral double math bit-for-bit; c or
// s can be EXACTLY zero at table nodes, and x can be +-0):
//   product sign: sx ^ (coef < 0)   (coef == 0 -> +0 coefficient, sign sx)
//   lo = p1 - p2 == 0 exactly : -0 iff p1 is -0 AND p2 is +0
//   hi = p3 + p4 == 0 exactly : -0 iff BOTH addend products are -0
//   (nonzero cancellation -> +0 in both cases; the formulas below reduce to
//   that because equal nonzero products carry equal signs)
//
// Domain: finite fp16 x (the S-2 bus); u_q in [0, 2^14) — periodic, no clamp
// (rope.sv header). Timing: purely combinational, like rope.sv; rope_row
// registers around it.

module rope_pair_fx
  import f16_arith_pkg::*;
(
  input  logic [13:0] u_q,    // Q0.14 turn-fraction phase
  input  logic [15:0] x_i,    // fp16 bits, x[i]
  input  logic [15:0] x_ih,   // fp16 bits, x[i+half]
  output logic [15:0] y_i,
  output logic [15:0] y_ih
);

  // ── C-ROPE constants + the generated golden tables (same file rope.sv
  //    includes; tables-check gates byte-identity vs the golden arbiter) ────
  localparam int unsigned CS_FRAC_REM = 6;

`include "rope_lut_tables.svh"

  // ── cos/sin LUT interpolation — byte-mirror of rope.sv:184-197 ───────────
  logic        [ 7:0] idx;
  logic        [ 5:0] frac6;
  logic signed [17:0] cprod, sprod;
  logic signed [16:0] c_val, s_val;

  // ── exact integer rotation ───────────────────────────────────────────────
  logic signed [F16_V24_W-1:0] vi, vih;
  logic signed [60:0]          p1, p2, p3, p4, lo, hi;
  logic                        p1s, p2s, p3s, p4s;      // product signs
  logic                        lo_s, hi_s;
  logic        [F16_SIG_W-1:0] lo_mag, hi_mag;

  always_comb begin
    idx   = u_q[13:6];
    frac6 = u_q[5:0];

    cprod = ROPE_COS_SLOPE[idx] * $signed({1'b0, frac6});    // 11s*7s -> 18s
    sprod = ROPE_SIN_SLOPE[idx] * $signed({1'b0, frac6});
    c_val = $signed({ROPE_COS_BASE[idx][15], ROPE_COS_BASE[idx]})
              + 17'(cprod >>> CS_FRAC_REM);
    s_val = $signed({ROPE_SIN_BASE[idx][15], ROPE_SIN_BASE[idx]})
              + 17'(sprod >>> CS_FRAC_REM);

    vi  = f16_to_v24(x_i);
    vih = f16_to_v24(x_ih);

    p1 = 61'(vi)  * 61'(c_val);          // x_i  * cos, grid 2^-38, exact
    p2 = 61'(vih) * 61'(s_val);          // x_ih * sin
    p3 = 61'(vih) * 61'(c_val);          // x_ih * cos
    p4 = 61'(vi)  * 61'(s_val);          // x_i  * sin
    lo = p1 - p2;
    hi = p3 + p4;

    p1s = x_i[15]  ^ (c_val < 0);
    p2s = x_ih[15] ^ (s_val < 0);
    p3s = x_ih[15] ^ (c_val < 0);
    p4s = x_i[15]  ^ (s_val < 0);

    // exact-zero signs (see header); nonzero results take the sum's sign
    lo_s = (lo == 0) ? (p1s && !p2s) : (lo < 0);
    hi_s = (hi == 0) ? (p3s &&  p4s) : (hi < 0);

    lo_mag = (lo < 0) ? F16_SIG_W'(-lo) : F16_SIG_W'(lo);
    hi_mag = (hi < 0) ? F16_SIG_W'(-hi) : F16_SIG_W'(hi);

    y_i  = f16_pack_real(lo_s, lo_mag, -38);
    y_ih = f16_pack_real(hi_s, hi_mag, -38);
  end

endmodule
