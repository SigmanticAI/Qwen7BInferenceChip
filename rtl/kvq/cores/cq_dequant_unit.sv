// cq_dequant_unit.sv — x_hat = code * s -> fp32 (clean-room).
// Thin combinational wrapper over cq_fp_pkg::dequant_one (cq_codec.py
// dequant_f32 / §1; fp32 read bus, D-010).
`default_nettype none
module cq_dequant_unit (
    input  wire signed [8:0] code,
    input  wire [15:0]       s,       // fp16 scale
    output wire [31:0]       hat      // fp32 bits
);
    assign hat = cq_fp_pkg::dequant_one(code, s);
endmodule
`default_nettype wire
