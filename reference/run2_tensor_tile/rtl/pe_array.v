// pe_array.v
// NxN array of PEs realizing output-stationary GEMM accumulation:
//   acc[i][j] += a_vec[i] * w_row[j]   each enabled beat, for K=N beats.
//
// Requirement traceability:
//   REQ-GEMM-001 : NxN systolic/output-stationary PE array structure
//   REQ-GEMM-002 : per-cycle broadcast of activation row / weight row to array
//   REQ-GEMM-003/004 : realized inside each pe instance (see pe.v)
//   REQ-STOR-003 : accumulator storage lives inside pe (instantiated here)
//
// Reset: synchronous, active-low (rst_n), passed through to all pe instances.

module pe_array #(
    parameter integer N     = 8,
    parameter integer ACC_W = 32
) (
    input                          clk,
    input                          rst_n,
    input  [N*8-1:0]               a_vec,     // element i at [i*8 +: 8], signed INT8
    input  [N*8-1:0]               w_row,     // element j at [j*8 +: 8], signed INT8
    input                          acc_en,
    input                          acc_clr,
    output [N*N*ACC_W-1:0]         acc_flat   // acc[i][j] at [(i*N+j)*ACC_W +: ACC_W]
);

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : ROW
            for (j = 0; j < N; j = j + 1) begin : COL

                wire signed [ACC_W-1:0] acc_out_ij;

                pe #(
                    .ACC_W(ACC_W)
                ) u_pe (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .a_in    (a_vec[i*8 +: 8]),
                    .w_in    (w_row[j*8 +: 8]),
                    .acc_en  (acc_en),
                    .acc_clr (acc_clr),
                    .acc_out (acc_out_ij)
                );

                assign acc_flat[(i*N+j)*ACC_W +: ACC_W] = acc_out_ij;

            end
        end
    endgenerate

endmodule
