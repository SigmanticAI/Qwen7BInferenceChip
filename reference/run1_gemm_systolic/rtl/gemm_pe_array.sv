// =============================================================================
// File        : gemm_pe_array.sv
// Module      : gemm_pe_array
// Purpose     : Output-stationary NxN broadcast MAC array for INT8 GEMM.
//               PE[i][j] holds a signed ACC_W accumulator for output element
//               C[i][j] = sum_k A[i][k]*B[k][j]. On each MAC step (mac_en=1)
//               the array consumes one column-slice of A (a_vec) and one
//               row-slice of B (w_vec), broadcasting a_vec[i] across row i
//               and w_vec[j] across column j, and accumulates
//               PE[i][j].acc += signed(a_vec[i]) * signed(w_vec[j]).
//
// Parameters  :
//   N        - array dimension (rows = cols = N), default 8
//   DATA_W   - INT8 operand width, default 8
//   ACC_W    - INT32 accumulator width, default 32
//   PW       - packed vector width = N*DATA_W (local)
//   ROWSEL_W - row-select width = $clog2(N) (local)
//
// Requirements implemented:
//   REQ-GEMM-004 - signed MAC per PE, product sign-extended into ACC_W
//   REQ-GEMM-005 - accumulate across K-tiles: array only clears on clear_acc
//                  or reset; it never self-clears between MAC steps, so the
//                  FSM controls tile sequencing/accumulation externally.
//   REQ-GEMM-013 - all arithmetic is signed two's-complement
//   REQ-STORE-003- accumulator storage cleared on reset and on clear_acc
//
// Notes:
//   - clear_acc has priority over mac_en when both are asserted in the same
//     cycle (checked in that order below).
//   - acc_row is a purely combinational read of the row selected by row_sel;
//     there is no read latency.
//   - Reset is synchronous, active-low: on rst_n==0 (at posedge clk) all
//     accumulators are cleared, same as clear_acc.
//   - No latches: all outputs are either registered (acc[][]) or driven by
//     unconditional continuous assignment (acc_row).
// =============================================================================

module gemm_pe_array #(
    parameter int N        = 8,
    parameter int DATA_W   = 8,
    parameter int ACC_W    = 32,
    parameter int PW       = N*DATA_W,
    parameter int ROWSEL_W = $clog2(N)
) (
    input  logic                  clk,
    input  logic                  rst_n,          // synchronous active-low
    input  logic                  clear_acc,      // priority clear, all PEs
    input  logic                  mac_en,         // perform one MAC step
    input  logic [PW-1:0]         a_vec,          // N signed INT8 activations
    input  logic [PW-1:0]         w_vec,          // N signed INT8 weights
    input  logic [ROWSEL_W-1:0]   row_sel,        // selects accumulator row
    output logic [N*ACC_W-1:0]    acc_row         // combinational row read
);

    // -------------------------------------------------------------------
    // Unpack input vectors into per-lane signed INT8 operands.
    // Element i occupies bits [i*DATA_W +: DATA_W], element 0 = LSBs.
    // -------------------------------------------------------------------
    logic signed [DATA_W-1:0] a_elem [0:N-1];
    logic signed [DATA_W-1:0] w_elem [0:N-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi++) begin : g_unpack_a
            assign a_elem[gi] = a_vec[gi*DATA_W +: DATA_W];
        end
        for (gj = 0; gj < N; gj++) begin : g_unpack_w
            assign w_elem[gj] = w_vec[gj*DATA_W +: DATA_W];
        end
    endgenerate

    // -------------------------------------------------------------------
    // NxN accumulator storage, one signed ACC_W register per PE.
    // -------------------------------------------------------------------
    logic signed [ACC_W-1:0] acc [0:N-1][0:N-1];

    generate
        for (gi = 0; gi < N; gi++) begin : g_row
            for (gj = 0; gj < N; gj++) begin : g_col
                // Product width sized for DATA_W x DATA_W signed multiply,
                // then sign-extended into ACC_W before accumulate.
                logic signed [2*DATA_W-1:0]  prod;
                logic signed [ACC_W-1:0]     prod_ext;

                assign prod     = a_elem[gi] * w_elem[gj];
                assign prod_ext = {{(ACC_W-2*DATA_W){prod[2*DATA_W-1]}}, prod};

                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        acc[gi][gj] <= '0;
                    end else if (clear_acc) begin
                        // clear_acc has priority over mac_en
                        acc[gi][gj] <= '0;
                    end else if (mac_en) begin
                        acc[gi][gj] <= acc[gi][gj] + prod_ext;
                    end
                    // else: hold value (accumulate across K-tiles per
                    // REQ-GEMM-005; no self-clear between MAC steps)
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------
    // Combinational row read: pack accumulator row row_sel into acc_row.
    // -------------------------------------------------------------------
    generate
        for (gj = 0; gj < N; gj++) begin : g_read
            assign acc_row[gj*ACC_W +: ACC_W] = acc[row_sel][gj];
        end
    endgenerate

endmodule
