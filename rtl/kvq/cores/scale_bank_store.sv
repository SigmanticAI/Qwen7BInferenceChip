// scale_bank_store.sv — persistent per-group scale rows (A2 / D-026).
//
// One row per COMMITTED key group: D fp16 per-channel scales, outlier lanes
// forced 16'h0000 by the writer (deterministic image, pinned by the golden
// packer cq_codec.pack_key_records). This is the persistent cousin of the
// transient scale_bank inside cq_key_path: that one holds the group being
// compressed; this one holds the scales of every group whose records live in
// the record SRAM.
//
// Deliberately NO clear port (do not wire dp_clear here — D-020 preserves
// stored records across a soft reset, so their group scales must survive
// too; only hard rst_n clears). Whole-row write at group commit (mutually
// exclusive with reads by FSM construction: ST_KEMIT vs ST_RWAIT), registered
// whole-row read, one-cycle latency, row stable until the next rd_en.
`default_nettype none
module scale_bank_store #(
    parameter int unsigned SETS = 4,    // SCALE_SETS (D-026 default)
    parameter int unsigned D    = 16,
    parameter int unsigned SW   = 16
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    wr_en,
    input  wire [$clog2(SETS)-1:0] wr_set,
    input  wire [D*SW-1:0]         wr_row,
    input  wire                    rd_en,
    input  wire [$clog2(SETS)-1:0] rd_set,
    output reg  [D*SW-1:0]         rd_row
);
    reg [D*SW-1:0] mem [0:SETS-1];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < int'(SETS); i = i + 1) mem[i] <= '0;
            rd_row <= '0;
        end else begin
            if (wr_en) mem[wr_set] <= wr_row;
            if (rd_en) rd_row <= mem[rd_set];
        end
    end
endmodule
`default_nettype wire
