// scale_bank.sv — per-channel fp16 scale storage (clean-room).
//
// Holds one fp16 scale per channel for the grouped-key path: a group's
// per-channel scales are computed once, loaded ONE CHANNEL PER CYCLE by the
// serialized scale walk (F1.5 — the load port was previously a D*SW-wide
// parallel bus fed by D parallel cq_scale_unit instances), then read back
// (whole-bus) for every token emitted in that group. `clear` (soft reset,
// D-020 B-2) wipes the bank so an aborted group leaves no stale scale behind.
`default_nettype none
module scale_bank #(
    parameter int unsigned D  = 16,
    parameter int unsigned SW = 16
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire                  wr_en,      // write one channel's scale
    input  wire [$clog2(D)-1:0]  wr_idx,
    input  wire [SW-1:0]         wr_data,
    output wire [D*SW-1:0]       bus         // registered per-channel scales
);
    reg [D*SW-1:0] mem;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)      mem <= '0;
        else if (clear)  mem <= '0;
        else if (wr_en)  mem[wr_idx*SW +: SW] <= wr_data;
    end
    assign bus = mem;
endmodule
`default_nettype wire
