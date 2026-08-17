// act_fifo.v
// REQ-STOR-002 : synchronous FIFO storage for activation stream
// REQ-GEMM-008 : activation FIFO feeding PE array (FWFT read)
// REQ-GEMM-015 : full/empty backpressure (no data loss / duplication)
// REQ-GEMM-016 : reset behavior -- read/write pointers and count clear to 0
//                (FIFO comes up empty) on synchronous active-low reset.
// flush         : synchronous robustness clear (helps ERR-005) -- when
//                 asserted (and not in reset) clears wr_ptr/rd_ptr/count to
//                 0, i.e. empties the FIFO exactly like reset, so the
//                 top-level FSM can drop residual activation beats between
//                 tiles. Reset has priority over flush.
//
// First-word-fall-through: rd_data always combinationally reflects the
// current head entry whenever the FIFO is not empty, so a consumer that
// asserts rd_en while !empty can consume rd_data that very same cycle.
// A pop only advances the read pointer / decrements count on the clock
// edge when (rd_en && !empty). Pushes are ignored when full, pops are
// ignored when empty -- no data loss or duplication (REQ-GEMM-015).

module act_fifo #(
    parameter integer W     = 65,   // data width (top uses {a_in_last, a_in_data[63:0]} = 65 bits)
    parameter integer DEPTH = 16,
    localparam integer PW   = (DEPTH > 1) ? $clog2(DEPTH)     : 1, // pointer width
    localparam integer CW   = (DEPTH > 1) ? $clog2(DEPTH+1)   : 1  // count width (0..DEPTH inclusive)
) (
    input                  clk,
    input                  rst_n,      // synchronous active-low
    input                  flush,      // synchronous: clear wr_ptr/rd_ptr/count to 0 (empty). reset has priority.
    input                  wr_en,      // push when wr_en && !full
    input      [W-1:0]     wr_data,
    input                  rd_en,      // pop when rd_en && !empty
    output     [W-1:0]     rd_data,    // FWFT: head word visible whenever !empty (same-cycle as rd_en)
    output                 full,
    output                 empty
);

    // REQ-STOR-002: circular buffer storage, DEPTH entries wide W bits.
    reg [W-1:0]  mem [0:DEPTH-1];

    // REQ-GEMM-016: pointers/count cleared on reset.
    reg [PW-1:0] wr_ptr;
    reg [PW-1:0] rd_ptr;
    reg [CW-1:0] count;

    wire do_push = wr_en && !full;
    wire do_pop  = rd_en && !empty;

    assign full  = (count == DEPTH[CW-1:0]);
    assign empty = (count == {CW{1'b0}});

    // FWFT combinational read of the current head entry.
    assign rd_data = mem[rd_ptr];

    // REQ-GEMM-016: synchronous active-low reset clears pointers/count.
    // flush (robustness, ERR-005): same clear, synchronous, but only when
    // not in reset -- reset takes priority over flush.
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= {PW{1'b0}};
            rd_ptr <= {PW{1'b0}};
            count  <= {CW{1'b0}};
        end else if (flush) begin
            wr_ptr <= {PW{1'b0}};
            rd_ptr <= {PW{1'b0}};
            count  <= {CW{1'b0}};
        end else begin
            // REQ-STOR-002 / REQ-GEMM-008: push side
            if (do_push) begin
                mem[wr_ptr] <= wr_data;
                if (wr_ptr == DEPTH[PW-1:0]-1'b1)
                    wr_ptr <= {PW{1'b0}};
                else
                    wr_ptr <= wr_ptr + 1'b1;
            end

            // REQ-GEMM-008 / REQ-GEMM-015: pop side (FWFT: rd_ptr advances,
            // rd_data for the *next* cycle reflects the new head).
            if (do_pop) begin
                if (rd_ptr == DEPTH[PW-1:0]-1'b1)
                    rd_ptr <= {PW{1'b0}};
                else
                    rd_ptr <= rd_ptr + 1'b1;
            end

            // REQ-GEMM-015: count-based full/empty tracking, simultaneous
            // push+pop leaves count unchanged; push-only increments;
            // pop-only decrements.
            case ({do_push, do_pop})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count; // 2'b00 hold, 2'b11 net-zero
            endcase
        end
    end

endmodule
