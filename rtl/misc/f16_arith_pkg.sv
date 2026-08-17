// f16_arith_pkg.sv — synthesizable integer fp16 arithmetic primitives for the
// IB-LAYER decoder-layer blocks (rope_row / residual_add_fx / asu_swiglu).
//
// WHY THIS PACKAGE EXISTS: rtl/rope/rope.sv:193 and rtl/misc/residual_add.sv:136
// carry SV `real` (float64) datapaths — bit-exact in simulation, NOT
// synthesizable (IB_LAYER.md §1 reality check 1). This package is the
// integer replacement, following the house precedent rtl/kvq/cores/
// cq_fp_pkg.sv:5 ("replaces the earlier `real` (float64)") and the
// seq_walker_comp P1-P6 proof discipline.
//
// NUMERIC MODEL. Every finite fp16 value is an integer multiple of 2^-24
// (the subnormal LSB). The primitives carry values as
//     (sign, sig, exp)  with  value = (-1)^sign * sig * 2^exp
// in EXACT integer form — wide enough that no intermediate ever rounds —
// and narrow ONCE with round-half-to-even in f16_pack_real. That mirrors
// the golden contracts exactly: C-6 (residual f16(a+b), one RNE), C-ROPE
// (two exact products + one RNE), C-SILU/SwiGLU product (exact product +
// one RNE) — golden/apex_golden/transformer.py, and the D-030 C-LBUS
// realizability lemma (test_layer_bus.py §C: all these operands are exact
// in <= 53 significand bits, so single-RNE integer datapaths can be
// bit-exact).
//
// DOMAIN: finite fp16 only (+-0, subnormals, normals). inf/NaN are not
// produced by any upstream bus (S-2 write port, C-2 epilogues; the same
// precondition the behavioral blocks document as "decoded defensively,
// unused") and are EXCLUDED from the contract of f16_to_v24/f16_sig_exp.
// Consumers reject or never receive them; the verification suites drive
// the full finite domain including both signed zeros and all subnormals.
//
// OVERFLOW: f16_pack_real saturates E > 15 to +-inf — the fp16-native
// "saturation" the behavioral real_to_f16 implements (E > 15 -> inf, incl.
// the round-up-to-inf path). Bit-identical by the resfx/rope suites.
`ifndef F16_ARITH_PKG_SV
`define F16_ARITH_PKG_SV

package f16_arith_pkg;

  // ── widths ────────────────────────────────────────────────────────────────
  // v24 form: value * 2^24 as a signed integer. |max fp16| = 65504, so
  // |v24| <= 65504 * 2^24 < 2^40 — 41 magnitude bits; carry one spare.
  localparam int unsigned F16_V24_W = 43;   // signed value*2^24
  localparam int unsigned F16_SIG_W = 57;   // f16_pack_real significand port

  // ── unpack: fp16 bits -> (sig, exp) with value = sig * 2^exp ─────────────
  // normal   (e in [1,30]) : sig = 1024 + m, exp = e - 25
  // subnormal/zero (e = 0) : sig = m,        exp = -24
  // sign is bits[15], returned separately by callers that need it.
  // (magnitude bits [14:0] only — the sign is the caller's, keeping every
  // function -Wall clean about unused bits)
  function automatic logic [10:0] f16_sig(input logic [14:0] bm);
    logic [4:0] e;
    logic [9:0] m;
    begin
      e = bm[14:10];
      m = bm[9:0];
      f16_sig = (e == 5'd0) ? {1'b0, m} : {1'b1, m};
    end
  endfunction

  function automatic int f16_exp(input logic [4:0] e);
    // pass bits[14:10]; subnormal/zero (e=0) sits on the 2^-24 grid
    f16_exp = (e == 5'd0) ? -24 : (int'({27'b0, e}) - 25);
  endfunction

  // ── fp16 -> signed value*2^24 (exact; finite domain) ─────────────────────
  function automatic logic signed [F16_V24_W-1:0] f16_to_v24(
      input logic [15:0] b);
    logic [4:0]                     e;
    logic [9:0]                     m;
    logic [F16_V24_W-1:0]           mag;
    begin
      e = b[14:10];
      m = b[9:0];
      if (e == 5'd0)
        mag = {33'b0, m};                          // m * 2^-24 * 2^24 = m
      else
        // (1024+m) * 2^(e-25) * 2^24 = (1024+m) << (e-1)
        mag = {32'b0, 1'b1, m} << (e - 5'd1);
      f16_to_v24 = b[15] ? -$signed(mag) : $signed(mag);
    end
  endfunction

  // ── the ONE narrowing: (sign, sig, exp) -> fp16 bits, RNE ────────────────
  // value = sig * 2^exp. sig == 0 returns the CALLER-provided sign with a
  // zero magnitude — IEEE zero-sign rules live at the call site (they
  // depend on the operation: add vs sub vs mul), exactly as in the
  // behavioral double math.
  function automatic logic [15:0] f16_pack_real(
      input logic                 sgn,
      input logic [F16_SIG_W-1:0] sig,
      input int                   exp);
    int                   p, E, sh;
    logic [F16_SIG_W:0]   keep, rem, halfb, mask;
    logic                 roundup;
    logic [15:0]          out;
    begin
      out = {sgn, 15'd0};
      if (sig != '0) begin
        p = 0;
        for (int i = 0; i < F16_SIG_W; i++) if (sig[i]) p = i;
        E = p + exp;
        if (E > 15) begin
          out = {sgn, 5'h1F, 10'd0};                       // overflow -> inf
        end else if (E >= -14) begin
          sh = p - 10;
          if (sh <= 0) begin
            // exact: normalize left, no bits dropped
            keep = {1'b0, sig} << (-sh);
            out  = {sgn, 5'(E + 15), keep[9:0]};
          end else begin
            mask    = ({{F16_SIG_W{1'b0}}, 1'b1} << sh) - 1'b1;
            rem     = {1'b0, sig} & mask;
            halfb   = {{F16_SIG_W{1'b0}}, 1'b1} << (sh - 1);
            keep    = {1'b0, sig} >> sh;                   // in [1024, 2047]
            roundup = (rem > halfb) || ((rem == halfb) && keep[0]);  // RNE, normal path
            keep    = keep + {{F16_SIG_W{1'b0}}, roundup};
            if (keep[11]) begin                            // 2048 -> carry
              keep = 'd1024;
              E    = E + 1;
            end
            // DOCUMENTED-DEFENSIVE (measured across the resfx + rope
            // mutation passes; derived for swiglu): this check is
            // GLOBALLY equivalent — it only ever sees E=16 via the carry
            // path, where keep==1024 makes the else-encoding
            // {sgn, 5'(31), 10'd0} the inf pattern anyway. It also
            // catches pre-round E=16/17 in every sh>0 flow, which makes
            // the TOP-of-function check equivalent at EVERY existing call
            // site too: sh<=0 requires sig < 2^11, and all current
            // producers have sig >= 2^10 * 2^10 when overflow is possible
            // (swiglu products multiply two >=2^10 significands; FRAC
            // sites overflow only with p >= 40). BOTH checks are kept per
            // the never-weaken rule; their mutants are equivalent mutants
            // (B3-M5 class) and no port-level TB is expected to kill
            // them. A FUTURE call site feeding sig < 2^11 with dynamic
            // positive exp MUST add that kill to its suite.
            if (E > 15) out = {sgn, 5'h1F, 10'd0};         // rounded to inf
            else        out = {sgn, 5'(E + 15), keep[9:0]};
          end
        end else begin
          // subnormal target: m = RNE(sig * 2^(exp+24)), grid 2^-24
          sh = -(exp + 24);
          if (sh <= 0) begin
            keep = {1'b0, sig} << (-sh);                   // exact, < 1024
            out  = {sgn, keep[14:0]};
          end else if (sh > F16_SIG_W) begin
            out = {sgn, 15'd0};                            // rounds to zero
          end else begin
            mask    = ({{F16_SIG_W{1'b0}}, 1'b1} << sh) - 1'b1;
            rem     = {1'b0, sig} & mask;
            halfb   = {{F16_SIG_W{1'b0}}, 1'b1} << (sh - 1);
            keep    = {1'b0, sig} >> sh;
            roundup = (rem > halfb) || ((rem == halfb) && keep[0]);  // RNE, subnormal path
            keep    = keep + {{F16_SIG_W{1'b0}}, roundup};
            // keep may reach 1024 = smallest normal: {e=1, m=0} — the
            // [14:0] encoding is exactly that (behavioral keep[14:0] trick)
            out = {sgn, keep[14:0]};
          end
        end
      end
      f16_pack_real = out;
    end
  endfunction

  // ── fp16 -> fp32 bits, EXACT (every finite fp16 is fp32-representable).
  // Normal: e32 = e+112, m23 = m<<13. Subnormal: normalize (msb p):
  // e32 = 103+p, m23 = low bits shifted to 23. Signed zero preserved.
  // inf/NaN map to e32=255 (defensive; no bus produces them).
  function automatic logic [31:0] f16_to_f32_bits(input logic [15:0] b);
    logic [4:0]  e;
    logic [9:0]  m;
    int          p;
    logic [22:0] m23;
    logic [31:0] out;
    begin
      e = b[14:10];
      m = b[9:0];
      if (e == 5'd0) begin
        if (m == 10'd0) begin
          out = {b[15], 31'd0};
        end else begin
          p = 0;
          for (int i = 0; i < 10; i++) if (m[i]) p = i;
          m23 = 23'({22'b0, m} << (23 - p)); // hidden bit shifts out of [22:0]
          out = {b[15], 8'(103 + p), m23};
        end
      end else if (e == 5'd31) begin
        out = {b[15], 8'hFF, (m == 10'd0) ? 23'd0 : {1'b1, 22'd0}};
      end else begin
        out = {b[15], 8'({3'b0, e} + 8'd112), {m, 13'b0}};
      end
      f16_to_f32_bits = out;
    end
  endfunction

endpackage

`endif // F16_ARITH_PKG_SV
