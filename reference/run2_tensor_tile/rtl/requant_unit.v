// requant_unit.v
// Combinational single-lane requantization + saturation for INT8 GEMM output.
//   q = clip_int8( round_to_nearest( acc * scale ) >>> shift )
//
// Requirement traceability:
//   REQ-GEMM-005 : requantization (accumulator * unsigned scale, arithmetic
//                  right shift with round-to-nearest)
//   REQ-GEMM-006 : clip result to signed INT8 range
//   ERR-002      : saturation flag asserted whenever clipping occurs
//
// Purely combinational: no clock, no reset.

module requant_unit #(
    parameter integer ACC_W   = 32,
    parameter integer SCALE_W = 16,
    parameter integer SHIFT_W = 5,
    parameter integer DW_OUT  = 8
) (
    input  signed [ACC_W-1:0]   acc_in,
    input  [SCALE_W-1:0]        scale,   // unsigned multiplier
    input  [SHIFT_W-1:0]        shift,   // arithmetic right-shift amount
    output signed [DW_OUT-1:0]  q_out,   // clipped INT8 result
    output                      sat      // 1 if clipping/saturation occurred
);

    // ---------------------------------------------------------------
    // Width planning (elaboration-time constants):
    //   scale is unsigned; widen by 1 bit with a leading zero so it can
    //   be treated as a non-negative signed operand.
    //   prod width = ACC_W + (SCALE_W+1)  (exact, no truncation for signed*signed)
    //   WIDE_W adds margin for the rounding-constant addition and to
    //   comfortably hold the post-shift value prior to clipping.
    // ---------------------------------------------------------------
    localparam integer EXT_SCALE_W = SCALE_W + 1;
    localparam integer PROD_W      = ACC_W + EXT_SCALE_W;
    localparam integer WIDE_W      = PROD_W + 2;

    // Step 0: widen scale to a non-negative signed value.
    wire signed [EXT_SCALE_W-1:0] scale_ext = {1'b0, scale};

    // Step 1: prod = acc_in * scale  (signed x signed, exact width)
    wire signed [PROD_W-1:0] prod = acc_in * scale_ext;

    // Sign-extend product into the wide working width.
    wire signed [WIDE_W-1:0] prod_wide = {{(WIDE_W-PROD_W){prod[PROD_W-1]}}, prod};

    // Step 2: round-to-nearest before the shift.
    reg signed [WIDE_W-1:0] round_const;
    reg signed [WIDE_W-1:0] rounded;
    reg signed [WIDE_W-1:0] shifted;

    always @* begin
        if (shift == {SHIFT_W{1'b0}}) begin
            round_const = {WIDE_W{1'b0}};
        end else begin
            round_const = ({{(WIDE_W-1){1'b0}}, 1'b1} <<< (shift - 1'b1));
        end
        rounded = prod_wide + round_const;

        // Step 3: arithmetic right shift (signed).
        shifted = rounded >>> shift;
    end

    // Step 4: clip to signed [-(2^(DW_OUT-1)), 2^(DW_OUT-1)-1] range
    // (DW_OUT=8 gives the required INT8 range [-128,127]).
    localparam signed [WIDE_W-1:0] MAXV = ({{(WIDE_W-1){1'b0}}, 1'b1} <<< (DW_OUT-1)) - 1'b1;
    localparam signed [WIDE_W-1:0] MINV = -(({{(WIDE_W-1){1'b0}}, 1'b1} <<< (DW_OUT-1)));

    reg signed [DW_OUT-1:0] q_out_r;
    reg                     sat_r;

    always @* begin
        if (shifted > MAXV) begin
            q_out_r = MAXV[DW_OUT-1:0];
            sat_r   = 1'b1;
        end else if (shifted < MINV) begin
            q_out_r = MINV[DW_OUT-1:0];
            sat_r   = 1'b1;
        end else begin
            q_out_r = shifted[DW_OUT-1:0];
            sat_r   = 1'b0;
        end
    end

    assign q_out = q_out_r;
    assign sat   = sat_r;

endmodule
