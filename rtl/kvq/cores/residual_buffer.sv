// residual_buffer.sv — holds up to G in-flight key tokens (fp16) for the
// grouped-key path (clean-room). D-016(c): the fill counter is $clog2(G+1)
// bits but the memory index is $clog2(G) bits, so an unguarded write of the
// (G+1)-th token would truncate cnt==G onto slot 0 and clobber the group's
// first token. This implementation DROPS any write past G and saturates the
// fill count at G (regression: verif/kvq/smoke/tb_residual_alias.sv).
//
// `clear` marks the first token of a new group: it writes slot 0 and resets
// the fill to 1 in the same cycle (a group always starts with a valid token).
//
// F1.5 AREA REWORK: the read port is now REGISTERED (one-cycle latency,
// rd_vec = mem[rd_idx sampled at the previous posedge]). The original
// asynchronous read made the G x (DIM*DW)-bit store unmappable to block RAM
// (at the ship config D=64/G=128 it synthesized to 131,072 discrete FFs plus
// a 128:1 x 1024b mux — per key engine). With the sync read the store infers
// as a true $mem/BRAM. cq_key_path absorbs the latency: the amax sweep runs
// a 2-cycle address/consume pattern and the emit walk primes the row once
// per token (the row register then stays stable for the whole channel walk).
`default_nettype none
module residual_buffer #(
    parameter int unsigned DIM = 8,
    parameter int unsigned DW  = 16,
    parameter int unsigned G   = 64
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    wr_valid,
    input  wire [DIM*DW-1:0]       wr_vec,
    input  wire                    clear,     // first token of a new group
    output wire [$clog2(G+1)-1:0]  fill,
    input  wire [$clog2(G)-1:0]    rd_idx,
    output wire [DIM*DW-1:0]       rd_vec     // registered: mem[rd_idx] @ prev edge
);
    localparam int unsigned CNT_W = $clog2(G+1);
    localparam int unsigned IDX_W = $clog2(G);

    reg [DIM*DW-1:0] mem [0:G-1];
    reg [CNT_W-1:0]  cnt;
    reg [DIM*DW-1:0] rd_r;

    // S10a/BRAM: the WRITE port must live in a clk-only process — yosys's
    // frontend mem2reg's any array touched from an async-reset process, even
    // though mem itself was never reset (the 131k-FF ship-config blowup).
    // Write condition/address are IDENTICAL to the old fused process:
    // a write happens iff wr_valid && (clear || cnt < G); addr 0 on clear.
    wire            mem_we   = wr_valid && (clear || (cnt < CNT_W'(G)));
    wire [IDX_W-1:0] mem_wa  = clear ? '0 : cnt[IDX_W-1:0];
    always @(posedge clk)
        if (mem_we) mem[mem_wa] <= wr_vec;

    // fill counter keeps its async reset in its own process (ordinary FFs)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= '0;
        end else if (wr_valid) begin
            if (clear)                    cnt <= CNT_W'(1);
            else if (cnt < CNT_W'(G))     cnt <= cnt + CNT_W'(1);
            // cnt == G with clear==0: overflow write DROPPED, cnt saturates.
        end
    end

    // registered read port (no reset: pure datapath, keeps BRAM inference)
    always @(posedge clk) rd_r <= mem[rd_idx];

    assign fill   = cnt;
    assign rd_vec = rd_r;
endmodule
`default_nettype wire
