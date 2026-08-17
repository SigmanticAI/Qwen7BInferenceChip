// residual_add.sv — element-wise RESIDUAL add on the fp16 bus (contract C-6).
//
// The APEX decoder-layer residual stream (golden/apex_golden/transformer.py,
// contract C-6) is carried on the fp16 bus with a SINGLE round-half-to-even
// narrowing per sublayer output:  r = f16(a + b).  This block computes exactly
// that for one element:
//
//   in : a  (fp16 bits)   incoming residual element  (X[T] / r1 on the fp16 bus)
//        b  (fp16 bits)   sublayer output element     (attn_proj / ffn_out,
//                                                       fp16-narrowed)
//   out: y = f16( f64(a) + f64(b) )   (fp16 bits, ONE RNE)
//
// NOTE ON "INT add + saturation": the session brief suggested an INT saturating
// adder.  The golden residual stream is NOT integer — decoder_layer_fx carries
// it as fp16 (`_f16(X[T] + attn_proj)`, `_f16(r1 + ffn_out)`), so a bit-exact
// residual for THIS layer must be an fp16 add with a single RNE narrowing (the
// same discipline as the S-2 KV-write bus and rtl/rope/rope.sv's output).  An
// INT8 saturating add would not be bit-exact against the golden and is not what
// the layer's contract needs; this block therefore implements the fp16 residual
// primitive.  Because both operands are fp16-grid (<=11 significand bits), their
// exact sum is representable in IEEE double with NO intermediate rounding, so
// carrying the add in SV `real` is bit-identical to the golden's numpy float64
// add regardless of simulator; the ONLY rounding is the explicit round-half-to-
// even double->half in real_to_f16 (byte-identical to rtl/rope/rope.sv).
//
// Overflow (a+b beyond fp16 max) saturates to +/-inf via the RNE narrower's
// E>15 path — the fp16-native form of "saturation"; in-contract residual
// operands never reach it (documented, defensive).
//
// Timing: purely COMBINATIONAL (0-cycle), mirroring rope / asu_silu; the
// integrating datapath registers around it (ARCHITECTURE.md §0).

module residual_add (
  input  logic [15:0] a,     // fp16 bits, incoming residual element
  input  logic [15:0] b,     // fp16 bits, sublayer output element
  output logic [15:0] y      // fp16 bits, f16(a + b)  (single RNE)
);

  // ── pow2(p): exact real 2^p (p small; exact in IEEE double) ────────────────
  function automatic real pow2(input int p);
    real r;
    int  i;
    begin
      r = 1.0;
      if (p >= 0) for (i = 0; i < p;  i++) r = r * 2.0;
      else        for (i = 0; i < -p; i++) r = r / 2.0;
      return r;
    end
  endfunction

  // ── f16_to_real: fp16 bit pattern -> exact float64 value (mirror rope.sv) ──
  function automatic real f16_to_real(input logic [15:0] bts);
    logic       sgn;
    logic [4:0] e;
    logic [9:0] m;
    real        mag;
    begin
      sgn = bts[15];
      e   = bts[14:10];
      m   = bts[9:0];
      if (e == 5'd0)
        mag = $itor({22'b0, m}) * pow2(-24);                 // subnormal / zero
      else if (e == 5'd31)
        mag = pow2(1024);                                    // inf/NaN (unused)
      else
        mag = $itor({21'b0, (11'd1024 + {1'b0, m})}) * pow2(int'({27'b0, e}) - 25);
      return sgn ? -mag : mag;
    end
  endfunction

  // ── real_to_f16: exact float64 -> fp16 bits, ROUND-HALF-TO-EVEN ────────────
  // The ONE narrowing of the C-6 residual contract (byte-identical to rope.sv).
  function automatic logic [15:0] real_to_f16(input real v);
    logic [63:0]        db;
    logic               ds;
    logic [10:0]        de;
    logic [51:0]        dm;
    logic signed [12:0] E;
    logic [52:0]        sig;
    logic [52:0]        keep;
    logic [63:0]        rem, halfbit, sig64, mask;
    logic               roundup;
    int                 sh;
    logic [4:0]         eh;
    logic [15:0]        out;
    begin
      db = $realtobits(v);
      ds = db[63];
      de = db[62:52];
      dm = db[51:0];
      if (de == 11'd0) begin
        out = {ds, 15'd0};                             // 0 / double-subnormal
      end else if (de == 11'h7FF) begin
        out = {ds, 5'h1F, (dm == 0) ? 10'd0 : 10'h200};  // inf / qNaN (unused)
      end else begin
        E   = $signed({2'b0, de}) - 13'sd1023;
        sig = {1'b1, dm};
        if (E > 13'sd15) begin
          out = {ds, 5'h1F, 10'd0};                    // overflow -> inf (sat)
        end else if (E >= -13'sd14) begin
          keep    = {42'b0, sig[52:42]};               // 11 bits in [1024,2047]
          rem     = {22'b0, sig[41:0]};
          halfbit = 64'd1 << 41;
          roundup = (rem > halfbit) || (rem == halfbit && keep[0]);
          keep    = keep + (roundup ? 53'd1 : 53'd0);
          if (keep[11]) begin                          // 2048 -> carry
            keep = 53'd1024;
            E    = E + 13'sd1;
          end
          if (E > 13'sd15) begin
            out = {ds, 5'h1F, 10'd0};                  // rounded up to inf (sat)
          end else begin
            eh  = 5'(int'(E) + 15);                    // [1, 30]
            out = {ds, eh, keep[9:0]};
          end
        end else begin
          sh = 28 - int'(E);                           // >= 43
          if (sh >= 64) begin
            out = {ds, 15'd0};
          end else begin
            sig64   = {11'b0, sig};
            mask    = (64'd1 << sh) - 64'd1;
            keep    = 53'(sig64 >> sh);
            rem     = sig64 & mask;
            halfbit = 64'd1 << (sh - 1);
            roundup = (rem > halfbit) || (rem == halfbit && keep[0]);
            keep    = keep + (roundup ? 53'd1 : 53'd0);
            out     = {ds, keep[14:0]};
          end
        end
      end
      return out;
    end
  endfunction

  real ar, br, sr;

  always_comb begin
    ar = f16_to_real(a);
    br = f16_to_real(b);
    sr = ar + br;                 // exact in float64 (both operands fp16-grid)
    y  = real_to_f16(sr);         // single RNE (C-6 residual bus)
  end

endmodule
