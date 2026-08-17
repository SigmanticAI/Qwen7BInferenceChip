// =============================================================================
// ctrl_fsm.v
//
// Control FSM for INT8 output-stationary GEMM tile (fixed K = N).
// Phases: IDLE -> LOAD_WEIGHTS -> COMPUTE -> DRAIN -> IDLE
//
// Requirement traceability (owned by FSM/Control agent):
//   REQ-GEMM-009  : LOAD_WEIGHTS must fully complete (N*N beats) before COMPUTE
//   REQ-GEMM-010  : legal phase sequencing IDLE->LOAD->COMPUTE->DRAIN->IDLE
//   REQ-GEMM-011  : busy / done status signals
//   REQ-GEMM-012  : accept weight beats only while in LOAD_WEIGHTS
//   REQ-GEMM-013  : accept activation/output beats only in COMPUTE/DRAIN resp.
//   REQ-STATE-001 : explicit state encoding, illegal state -> IDLE
//   REQ-STATE-002 : wbuf_loaded flag, valid after full load, cleared on new load
//   REQ-GEMM-016  : synchronous active-low reset defined for all state
//   REQ-GEMM-017  : fresh-load-required -- wbuf_loaded cleared when new load begins
//   ERR-003       : start cannot lead to COMPUTE without a completed load; the
//                    FSM structurally enforces this by always routing
//                    IDLE -> LOAD_WEIGHTS -> COMPUTE and only asserting
//                    wbuf_loaded / entering COMPUTE after N*N beats are written.
//                    No separate error output exists in the fixed port list, so
//                    this requirement is satisfied by construction, not by an
//                    explicit error flag.
//   ERR-005       : a_fifo_flush asserted throughout IDLE and LOAD_WEIGHTS so
//                    the activation FIFO is guaranteed empty before COMPUTE
//                    begins, clearing any residual over-supplied beats left
//                    over from a prior tile (robustness fix; combinational,
//                    does not alter valid/ready handshake semantics).
//
// Notes / assumptions:
//   - start is assumed to be a single-cycle pulse (per architecture contract);
//     this module does not filter multi-cycle asserted start, it simply leaves
//     LOAD_WEIGHTS/COMPUTE/DRAIN via their own completion conditions, and start
//     is only sampled while in IDLE.
//   - cfg_mode is reserved/ignored per architecture contract and intentionally
//     has no port here.
//   - w_load_last is an observed input only; per the architecture contract the
//     beat COUNT (not w_load_last) governs LOAD_WEIGHTS completion.
// =============================================================================

module ctrl_fsm #(
    parameter integer N = 8
) (
    input                                                clk,
    input                                                rst_n,

    input                                                start,          // single-cycle pulse: IDLE -> LOAD_WEIGHTS

    // Weight load handshake (this module drives ready + write strobe/addr)
    input                                                w_load_valid,
    output                                               w_load_ready,   // = (state==LOAD_WEIGHTS) && (load_cnt < N*N)
    input                                                w_load_last,    // observed, not required to gate (count governs)
    output                                               wbuf_wr_en,     // = w_load_valid && w_load_ready
    output [((N*N>1) ? $clog2(N*N) : 1)-1:0]             wbuf_wr_addr,   // = load_cnt
    output                                               wbuf_loaded,    // REQ-STATE-002 weight-loaded valid flag

    // Compute / activation FIFO side (FIFO is FWFT; this module pops it)
    input                                                a_fifo_empty,
    output                                               a_fifo_rd_en,   // pop = (state==COMPUTE) && !a_fifo_empty
    output                                               acc_en,         // = a_fifo_rd_en
    output                                               acc_clr,        // = (state==IDLE || state==LOAD_WEIGHTS)
    output [((N>1) ? $clog2(N) : 1)-1:0]                 wbuf_rd_row,    // current k (0..N-1) during COMPUTE
    output                                               compute_active, // = (state==COMPUTE)
    output                                               load_active,    // = (state==LOAD_WEIGHTS)
    output                                               a_fifo_flush,   // = (state==IDLE)||(state==LOAD_WEIGHTS): keep act_fifo empty until COMPUTE

    // Drain / result output handshake
    output                                               drain_active,   // = (state==DRAIN)
    output [((N>1) ? $clog2(N) : 1)-1:0]                 drain_row,      // current output row i (0..N-1)
    input                                                y_out_ready,
    output                                               y_out_valid,    // = (state==DRAIN)
    output                                               y_out_last,     // = (state==DRAIN) && (drain_row == N-1)

    // Status
    output                                               busy,           // = (state != IDLE)
    output                                               done            // single-cycle pulse at end of drain
);

    // -------------------------------------------------------------------
    // Local width parameters (mirrors of the port-width expressions above)
    // -------------------------------------------------------------------
    localparam integer ADDR_W = (N*N > 1) ? $clog2(N*N)   : 1;
    localparam integer ROW_W  = (N   > 1) ? $clog2(N)     : 1;
    localparam integer CNT_W  = $clog2(N*N + 1);            // must represent 0..N*N inclusive

    // Width-matched constants to avoid implicit 32-bit comparisons.
    // Values are always representable in the target width by construction
    // (CNT_W = clog2(N*N+1), ROW_W = clog2(N)), so truncation is safe.
    /* verilator lint_off WIDTHTRUNC */
    localparam [CNT_W-1:0] NN_C   = N*N;      // total weight beats
    localparam [CNT_W-1:0] NNM1_C = N*N - 1;  // last beat index
    localparam [ROW_W-1:0] NM1_C  = N - 1;    // last row/column index
    /* verilator lint_on WIDTHTRUNC */

    // -------------------------------------------------------------------
    // State encoding (REQ-STATE-001)
    // -------------------------------------------------------------------
    localparam [1:0] ST_IDLE         = 2'd0;
    localparam [1:0] ST_LOAD_WEIGHTS = 2'd1;
    localparam [1:0] ST_COMPUTE      = 2'd2;
    localparam [1:0] ST_DRAIN        = 2'd3;

    reg [1:0]         state;
    reg [CNT_W-1:0]   load_cnt;
    reg [ROW_W-1:0]   k;
    reg [ROW_W-1:0]   drain_row_r;
    reg               wbuf_loaded_r;

    // -------------------------------------------------------------------
    // Combinational control outputs
    // -------------------------------------------------------------------
    wire load_full = (load_cnt == NN_C);

    assign load_active    = (state == ST_LOAD_WEIGHTS);
    assign compute_active = (state == ST_COMPUTE);
    assign drain_active   = (state == ST_DRAIN);
    assign busy           = (state != ST_IDLE);

    assign w_load_ready = load_active && !load_full;                 // REQ-GEMM-012
    assign wbuf_wr_en   = w_load_valid && w_load_ready;
    assign wbuf_wr_addr = load_cnt[ADDR_W-1:0];
    assign wbuf_loaded  = wbuf_loaded_r;                              // REQ-STATE-002

    assign acc_clr      = (state == ST_IDLE) || (state == ST_LOAD_WEIGHTS);
    assign a_fifo_flush = (state == ST_IDLE) || (state == ST_LOAD_WEIGHTS); // ERR-005: keep act_fifo empty until COMPUTE
    assign a_fifo_rd_en = compute_active && !a_fifo_empty;            // REQ-GEMM-013
    assign acc_en       = a_fifo_rd_en;
    assign wbuf_rd_row  = k;

    assign drain_row   = drain_row_r;
    assign y_out_valid = drain_active;                                // REQ-GEMM-013
    assign y_out_last  = drain_active && (drain_row_r == NM1_C);
    assign done         = y_out_valid && y_out_ready && y_out_last;   // REQ-GEMM-011 single-cycle pulse

    // -------------------------------------------------------------------
    // Sequential state / counters (synchronous active-low reset)
    // REQ-GEMM-016 / REQ-GEMM-017
    // -------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            load_cnt      <= {CNT_W{1'b0}};
            k             <= {ROW_W{1'b0}};
            drain_row_r   <= {ROW_W{1'b0}};
            wbuf_loaded_r <= 1'b0;
        end else begin
            case (state)

                ST_IDLE: begin
                    if (start) begin
                        state         <= ST_LOAD_WEIGHTS;
                        load_cnt      <= {CNT_W{1'b0}};
                        k             <= {ROW_W{1'b0}};
                        wbuf_loaded_r <= 1'b0;   // REQ-GEMM-017: fresh load required, clear stale flag
                    end
                end

                ST_LOAD_WEIGHTS: begin
                    if (wbuf_wr_en) begin
                        if (load_cnt == NNM1_C) begin
                            load_cnt      <= load_cnt + 1'b1;
                            wbuf_loaded_r <= 1'b1;      // REQ-STATE-002
                            k             <= {ROW_W{1'b0}};
                            state         <= ST_COMPUTE; // REQ-GEMM-009: only after full N*N beats
                        end else begin
                            load_cnt <= load_cnt + 1'b1;
                        end
                    end
                end

                ST_COMPUTE: begin
                    if (a_fifo_rd_en) begin
                        if (k == NM1_C) begin
                            k           <= {ROW_W{1'b0}};
                            drain_row_r <= {ROW_W{1'b0}};
                            state       <= ST_DRAIN;
                        end else begin
                            k <= k + 1'b1;
                        end
                    end
                end

                ST_DRAIN: begin
                    if (y_out_valid && y_out_ready) begin
                        if (drain_row_r == NM1_C) begin
                            drain_row_r <= {ROW_W{1'b0}};
                            state       <= ST_IDLE;
                        end else begin
                            drain_row_r <= drain_row_r + 1'b1;
                        end
                    end
                end

                default: begin
                    // REQ-STATE-001: illegal/unreachable encoding recovers to IDLE
                    state <= ST_IDLE;
                end

            endcase
        end
    end

    // w_load_last is intentionally unused for completion gating: LOAD_WEIGHTS
    // completion is governed by the beat count (load_cnt), not by w_load_last,
    // per architecture contract. Tied off into an unused net to keep the
    // exact required port list while avoiding an "unused input" lint flag.
    /* verilator lint_off UNUSEDSIGNAL */
    wire w_load_last_unused = w_load_last;
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
