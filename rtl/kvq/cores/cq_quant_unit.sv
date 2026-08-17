// cq_quant_unit.sv — q = clamp(rne(x/s), qmin, qmax) (clean-room).
// Thin combinational wrapper over cq_fp_pkg::quant_one (cq_codec.py
// quant_codes / §1) — pure integer, synthesizable. The round-half-to-even
// tie decision lives in cq_fp_pkg::rne_div_u (mutation target: mut_tie,
// verif/kvq/fparith). Output is a signed code.
`default_nettype none
module cq_quant_unit (
    input  wire [15:0] x,        // fp16 element
    input  wire [15:0] s,        // fp16 scale
    input  wire [3:0]  bits,
    output wire [8:0]  code      // signed [-128,127]
);
    assign code = 9'(cq_fp_pkg::quant_one(x, s, int'(bits)));
endmodule
`default_nettype wire
