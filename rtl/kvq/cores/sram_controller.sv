// sram_controller.sv — behavioral record store for the KVQ engine (clean-room).
//
// One §4 byte-aligned record per row. Semantics required by kvq_engine.sv
// (see its header, D-016a): a read is ACKNOWLEDGED (rd_valid) only for an
// address that has actually been written; a read of an unwritten address never
// acks (the engine's bounded ST_RWAIT times out and reports RD_ERR). A valid
// read answers in exactly one cycle. `occupancy` counts DISTINCT written
// addresses (overwriting an address does not double-count); `full` asserts when
// every address holds a record. Writes always succeed (overwrite in place).
`default_nettype none
module sram_controller #(
    parameter int unsigned SRAM_DEPTH = 2,
    parameter int unsigned DATA_WIDTH = 64,
    parameter int unsigned ADDR_WIDTH = 1
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    wr_en,
    input  wire [ADDR_WIDTH-1:0]   wr_addr,
    input  wire [DATA_WIDTH-1:0]   wr_data,
    input  wire                    rd_en,
    input  wire [ADDR_WIDTH-1:0]   rd_addr,
    output reg  [DATA_WIDTH-1:0]   rd_data,
    output reg                     rd_valid,
    output wire [ADDR_WIDTH:0]     occupancy,
    output wire                    full
);
    reg [DATA_WIDTH-1:0] mem   [0:SRAM_DEPTH-1];
    reg                  valid [0:SRAM_DEPTH-1];
    reg [ADDR_WIDTH:0]   occ;

    integer i;
    // S10a/BRAM: the DATA array's ports live in clk-only processes (yosys
    // frontend mem2reg's arrays touched from async-reset processes — the
    // ship-config FF blowup). Contents are valid-gated so the array itself
    // never needs clearing; D-020 preserves it across soft reset anyway.
    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    // registered 1-cycle read; enable gives the HOLD-through-burst contract
    // (rd_data stable from ack through the whole ST_OUTPUT walk). No reset:
    // pure datapath, same pattern as residual_buffer.rd_r — rd_data is only
    // ever consumed under rd_valid, which IS reset (below).
    always @(posedge clk)
        if (rd_en && valid[rd_addr]) rd_data <= mem[rd_addr];

    // control: ack-only-written bitmap, occupancy, rd_valid — ordinary FFs,
    // async reset as before, cycle-aligned with rd_data on the same posedge
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            occ      <= '0;
            rd_valid <= 1'b0;
            for (i = 0; i < int'(SRAM_DEPTH); i = i + 1) valid[i] <= 1'b0;
        end else begin
            if (wr_en && !valid[wr_addr]) begin
                valid[wr_addr] <= 1'b1;
                occ            <= occ + {{ADDR_WIDTH{1'b0}}, 1'b1};
            end
            rd_valid <= rd_en && valid[rd_addr];
        end
    end

    assign occupancy = occ;
    assign full      = (occ == (ADDR_WIDTH+1)'(SRAM_DEPTH));
endmodule
`default_nettype wire
