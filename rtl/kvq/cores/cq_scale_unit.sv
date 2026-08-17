// cq_scale_unit.sv — s = max(amax/qmax, EPS) -> fp16 (clean-room).
// Thin combinational wrapper over cq_fp_pkg::scale_from_amax (cq_codec.py
// scale_from_amax / §1). qmax is derived from `bits` (4 -> INT4, 8 -> INT8).
`default_nettype none
module cq_scale_unit (
    input  wire [15:0] amax,     // sign-cleared fp16 magnitude
    input  wire [3:0]  bits,     // 4 (INT4) or 8 (INT8)
    output wire [15:0] scale     // fp16 scale bits
);
    assign scale = cq_fp_pkg::scale_from_amax(amax, int'(bits));
endmodule
`default_nettype wire
