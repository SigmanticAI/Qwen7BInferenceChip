// apex_q78_to_fp32.sv — v0.1 glue: EXACT Q7.8 -> fp32 widen beat adapter.
//
// Implements the S-1 input preparation of the golden chain
// (golden/apex_golden/attention.py, stage Q2): the RMSNorm output h (signed
// 16-bit Q7.8, value h/2^8) is per-token INT8-quantized by the D-021 feeder
// machinery, whose input bus is fp32 (D-010 shape). Golden arbiter line:
//     r.h8, r.s_h = quant_rows_i8(r.h.astype(np.float64) / 256.0)   # Q2/S-1
// This adapter produces fp32 bit patterns whose values are EXACTLY h/256:
// |h| <= 2^15 needs <= 16 significand bits < fp32's 24, and /256 is a pure
// exponent shift — the conversion is exact for the entire input domain, so
// the downstream seam_feeder_quant (proven bit-exact vs quant_rows_i8 on the
// fp32 grid) sees exactly the golden model's operands. NO rounding anywhere.
//
// Protocol: pure combinational pass-through of a §5 valid/ready stream
// (valid/data functions of the input only, ready wired straight back), so
// §5 stability is inherited from the producer; the producer (asu_rmsnorm)
// and consumer (seam_feeder_quant) both own boundary skids (D-019).
// The x-stream `last` is consumed here (the feeder frames by count).

module apex_q78_to_fp32 (
  // Q7.8 stream in (asu_rmsnorm m_y/m_last shape)
  input  logic               s_valid,
  output logic               s_ready,
  input  logic signed [15:0] s_y,
  input  logic               s_last,     // informational; framing is by count

  // fp32 element stream out (seam_feeder_quant in_data shape)
  output logic               m_valid,
  input  logic               m_ready,
  output logic        [31:0] m_data
);

  function automatic logic [31:0] q78_to_f32(input logic signed [15:0] y);
    logic        sgn;
    logic [16:0] mag;              // |y| <= 32768 = 2^15
    logic [4:0]  msb;
    logic [23:0] sh24;             // mag << (23 - msb): < 2^24 exactly
    logic        unused_hid;
    if (y == 16'sd0) return 32'h0000_0000;
    sgn = y[15];
    mag = sgn ? (~{1'b0, y} + 17'd1) : {1'b0, y};
    msb = 5'd0;
    for (int i = 0; i < 16; i++) begin
      if (mag[i]) msb = 5'(i);
    end
    // value = mag * 2^-8 = 1.frac * 2^(msb-8); biased exp = msb - 8 + 127
    sh24       = 24'(mag) << (5'd23 - msb);
    unused_hid = sh24[23];         // hidden bit (implied downstream)
    return {sgn, 8'(msb) + 8'd119, sh24[22:0]};
  endfunction

  assign m_valid = s_valid;
  assign s_ready = m_ready;
  assign m_data  = q78_to_f32(s_y);

  logic unused_ok;
  assign unused_ok = s_last;       // framing is by element count downstream

endmodule
