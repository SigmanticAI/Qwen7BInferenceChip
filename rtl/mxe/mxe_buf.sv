// mxe_buf.sv — simple synchronous SRAM model for MXE tile buffers.
//
// Implements the ARCHITECTURE.md sizing-guidance buffer: 1 write port,
// 1 read port, 1-cycle read latency, no reset (SRAM semantics — the
// controller never reads a location it has not written for the current job).
// Write-during-read to the same address returns OLD data (read samples the
// array before the write commits), which the MXE controller never relies on.

module mxe_buf #(
  parameter int unsigned WIDTH = 64,
  parameter int unsigned DEPTH = 16384,
  localparam int unsigned AW   = $clog2(DEPTH)
) (
  input  logic             clk,

  input  logic             we,
  input  logic [AW-1:0]    waddr,
  input  logic [WIDTH-1:0] wdata,

  input  logic [AW-1:0]    raddr,
  output logic [WIDTH-1:0] rdata     // valid 1 cycle after raddr
);

  logic [WIDTH-1:0] mem [DEPTH];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end

endmodule
