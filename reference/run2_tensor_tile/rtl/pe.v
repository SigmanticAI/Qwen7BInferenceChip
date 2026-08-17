// pe.v
// Single processing element (PE) for INT8 output-stationary GEMM tile.
// Accumulates signed 8x8 -> 16b product into a wide signed accumulator.
//
// Requirement traceability:
//   REQ-GEMM-003 : signed 8x8 multiply, sign-extended to ACC_W before add
//   REQ-GEMM-004 : accumulation over successive beats, wraps on overflow
//   ERR-001      : overflow wraps naturally at ACC_W bits (two's complement)
//   REQ-STOR-003 : accumulator storage lives inside pe (this module)
//
// Reset: synchronous, active-low (rst_n).

module pe #(
    parameter integer ACC_W = 32
) (
    input                      clk,
    input                      rst_n,      // synchronous active-low reset
    input  signed [7:0]        a_in,       // activation element (row)
    input  signed [7:0]        w_in,       // weight element (column, current k row)
    input                      acc_en,     // accumulate product this cycle
    input                      acc_clr,    // synchronous clear (priority over acc_en)
    output signed [ACC_W-1:0]  acc_out
);

    reg signed [ACC_W-1:0] acc;

    // signed 8x8 -> 16-bit signed product
    wire signed [15:0] product;
    assign product = a_in * w_in;

    // sign-extend product to ACC_W before accumulation
    wire signed [ACC_W-1:0] product_ext;
    assign product_ext = {{(ACC_W-16){product[15]}}, product};

    always @(posedge clk) begin
        if (!rst_n) begin
            acc <= {ACC_W{1'b0}};
        end else if (acc_clr) begin
            acc <= {ACC_W{1'b0}};
        end else if (acc_en) begin
            acc <= acc + product_ext; // wraps naturally at ACC_W bits (ERR-001)
        end
    end

    assign acc_out = acc;

endmodule
