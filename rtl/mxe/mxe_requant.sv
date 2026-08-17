// mxe_requant.sv — C-2 requant epilogue, INT32 → INT8 per lane:
//
//   y = sat8( rne( (acc * rq_scale) >> rq_shift ) ),  RNE = round-half-to-even
//
// Implements: C-2 / D-002. Bit-exact mirror of the golden arbiter
// apex_golden.compute.requant_i32_to_i8:
//   p    = acc * scale            — exact in signed 48 bits (|acc| < 2^31,
//                                   scale < 2^16 ⇒ |p| < 2^47)
//   shift==0 → q = p              — no rounding
//   else: q   = p >>> shift       — floor (arithmetic shift)
//         rem = p & ((1<<shift)-1)— non-negative remainder
//         q  += (rem > half) || (rem == half && q[0])   — half-to-even
//   y    = clip(q, -128, 127)
//
// Purely combinational; MXE_N lanes. Inputs must be stable while the drain
// beat is valid (guaranteed: accumulators are frozen during DRAIN and the
// descriptor's scale/shift are registered for the job's duration).

module mxe_requant
  import apex_pkg::*;
(
  input  acc_t        in_lanes [MXE_N],
  input  logic [15:0] rq_scale,          // unsigned magnitude (contract)
  input  logic [4:0]  rq_shift,
  output out8_t       out_lanes [MXE_N]
);

  function automatic out8_t rq_one(acc_t a, logic [15:0] s, logic [4:0] sh);
    logic signed [47:0] p, q;
    logic [47:0]        rem, half, mask;
    logic               round_up;
    // exact product: signed 32 x unsigned 16 (zero-extended into signed 17)
    p = 48'(a) * 48'($signed({1'b0, s}));
    if (sh == 5'd0) begin
      q = p;
    end else begin
      half     = 48'd1 << (sh - 5'd1);
      mask     = (48'd1 << sh) - 48'd1;
      q        = p >>> sh;                         // floor
      rem      = $unsigned(p) & mask;              // non-negative remainder
      round_up = (rem > half) || ((rem == half) && q[0]);
      q        = q + 48'(round_up);
    end
    if (q > 48'sd127)        rq_one = out8_t'(8'sd127);
    else if (q < -48'sd128)  rq_one = out8_t'(-8'sd128);
    else                     rq_one = out8_t'(q[7:0]);
  endfunction

  always_comb begin
    for (int c = 0; c < int'(MXE_N); c++)
      out_lanes[c] = rq_one(in_lanes[c], rq_scale, rq_shift);
  end

endmodule
