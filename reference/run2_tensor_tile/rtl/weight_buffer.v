// weight_buffer.v
// REQ-STOR-001 : NxN INT8 weight storage (register-file / SRAM model)
// REQ-GEMM-007 : row-wise combinational read of weights for PE array MAC
// REQ-GEMM-016 : reset behavior for this module -- SRAM contents are NOT
//                force-cleared on reset (no pointers/flags live in this
//                module; the "weights loaded" valid flag is owned and
//                cleared by the FSM agent, not here).
//
// Address mapping: linear write address addr = k*N + j (row-major),
// row k, column j.  Read port returns the full row k = rd_row packed as
// N INT8 lanes: rd_row_data[j*8 +: 8] = mem[rd_row*N + j].
//
// Address widths are derived with $clog2 and clamped to a minimum of 1
// bit so N == 1 (a degenerate 1x1 "tile") still elaborates legally.

module weight_buffer #(
    parameter integer N  = 8,
    // AW = width of the linear write address (covers N*N entries)
    localparam integer AW = (N*N > 1) ? $clog2(N*N) : 1,
    // RW = width of the row index (covers N rows)
    localparam integer RW = (N   > 1) ? $clog2(N)   : 1
) (
    input                      clk,
    input                      rst_n,       // synchronous active-low (unused: no cleared state here)
    input                      wr_en,       // write one INT8 weight this cycle
    input      [AW-1:0]        wr_addr,     // linear write address, row-major: addr = k*N + j
    input      [7:0]           wr_data,     // one INT8 weight
    input      [RW-1:0]        rd_row,      // read row index k (0..N-1) during compute
    output     [N*8-1:0]       rd_row_data  // N weights of row rd_row: element j at bits [j*8 +: 8]
);

    // REQ-STOR-001: NxN INT8 storage array, N*N entries (64 entries for N=8).
    reg [7:0] mem [0:(N*N)-1];

    // REQ-GEMM-007 / REQ-STOR-001: synchronous write, one INT8 lane per cycle.
    // Reset is synchronous active-low but this SRAM-like array is
    // intentionally NOT cleared on reset per the spec (contents persist
    // until explicitly overwritten by the loader); rst_n is sampled only
    // to keep the write port well-defined and free of inferred latches.
    always @(posedge clk) begin
        if (!rst_n) begin
            // No storage clear here by design (REQ-GEMM-016: no
            // pointers/flags live in this module). Write port simply
            // holds off during reset.
        end else if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // REQ-GEMM-007: combinational row read so the PE array can consume
    // rd_row_data the same cycle rd_row is presented.
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : ROW_READ
            assign rd_row_data[j*8 +: 8] = mem[rd_row*N + j];
        end
    endgenerate

endmodule
