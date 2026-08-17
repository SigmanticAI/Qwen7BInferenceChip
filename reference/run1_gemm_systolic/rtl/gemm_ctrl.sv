// =============================================================================
// File        : gemm_ctrl.sv
// Module      : gemm_ctrl
// Purpose     : Control FSM + descriptor decode for the INT8 systolic GEMM
//               accelerator (gemm_systolic_accel). Owns: descriptor handshake,
//               weight/activation load sequencing, PE-array MAC step sequencing
//               (address generation + enable timing honoring 1-cycle buffer
//               read latency), requant row sequencing, and result writeback
//               handshake (valid/ready, y_last, done pulse). Drives buffer
//               address/enable ports; does NOT touch data paths (a_vec/w_vec/
//               wdata/acc_row wiring is done by the top-level integrator).
//
// Ownership   : FSM and Control RTL Agent. No datapath computation is
//               performed or altered in this file.
//
// -----------------------------------------------------------------------------
// DESCRIPTOR BIT MAP (UQ-002 resolution, implemented EXACTLY as specified):
//   desc_data[7:0]    = tiles_m
//   desc_data[15:8]   = tiles_k
//   desc_data[23:16]  = tiles_n
//   desc_data[29:24]  = req_shift            (SHIFT_W=6 bits used)
//   desc_data[31:30]  = reserved, must be 0, ignored on decode
//   desc_data[63:32]  = req_scale            (signed 32-bit)
//   desc_data[95:64]  = flags                (reserved, ignored, default 0)
//   desc_data[127:96] = reserved             (ignored)
//
// -----------------------------------------------------------------------------
// UQ-003 RESOLUTION (verbatim, illegal descriptor handling, ERR-002),
//   EXTENDED with a buffer-capacity conservative-reject guard:
//   In DECODE: if ANY of tiles_m, tiles_k, tiles_n == 0 OR > its respective
//   MAX_TILES_* (tiles_m vs MAX_TILES_M, tiles_k vs MAX_TILES_K, tiles_n vs
//   MAX_TILES_N) -> the job is ILLEGAL. The job is ALSO ILLEGAL if the word
//   count it would require exceeds the configured buffer depth:
//   (tiles_k*tiles_n*N) > WBUF_DEPTH  OR  (tiles_m*tiles_k*N) > ABUF_DEPTH.
//   This second condition is a conservative reject added to prevent a
//   tile-count-legal descriptor from silently wrapping wbuf_waddr/
//   abuf_waddr and corrupting buffer contents when its word requirement
//   exceeds the buffer depth (e.g. defaults N=8, MAX_TILES_*=8,
//   WBUF_DEPTH=ABUF_DEPTH=64 admit tile-count-legal descriptors needing up
//   to 512 words). On ILLEGAL (either condition): the job is discarded and
//   the FSM returns directly to IDLE. No weight-load, activation-load, MAC,
//   requant, or writeback activity is emitted for this job (no wbuf/abuf
//   writes, no PE activity, no y_valid beats, no done pulse). err_overflow is
//   NOT set for an illegal descriptor (illegal-descriptor is a distinct,
//   non-sticky condition from arithmetic saturation overflow; no dedicated
//   illegal-descriptor status port exists in the contracted port list, so this
//   condition is silently discarded exactly as specified -- it is not conflated
//   with ERR-001 saturation overflow).
//
// UQ-004 RESOLUTION (verbatim, truncated load handling, ERR-003):
//   Weight load (LOAD_WEIGHT) and activation load (STREAM_ACT_MAC act-load
//   sub-phase) both terminate on whichever comes first: (a) the expected word
//   count for the descriptor (tiles_k*tiles_n*N for weights, tiles_m*tiles_k*N
//   for activations) is reached, or (b) w_last / a_last is asserted on an
//   accepted beat. The number of words ACTUALLY loaded is latched as
//   eff_wwords / eff_awords respectively. Any MAC step whose computed wbuf/abuf
//   word index is >= eff_wwords / eff_awords (i.e. beyond what was actually
//   loaded because the load was truncated early by w_last/a_last) is SKIPPED:
//   no buffer read, no pe_mac_en pulse is issued for that step, which is
//   numerically equivalent to zero-padding the missing operand data. This
//   never blocks forward progress -- the K-accumulation loop still completes
//   all tiles_k*N nominal steps, silently contributing zero for truncated
//   positions.
//
// CHANNEL WIRING NOTE: the top-level integrator wires the descriptor
//   channel STRAIGHT THROUGH to this module with NO skid buffer, so the
//   IDLE-gated desc_ready produced here IS the true external desc_ready
//   seen by the requester (REQ-GEMM-001/REQ-PROTO-004) -- there is no
//   upstream elastic stage masking this module's back-pressure timing. The
//   weight and activation load channels, by contrast, ARE registered by
//   separate skid buffers at the top level before reaching w_valid/a_valid
//   here. This module itself implements plain single-cycle valid/ready
//   acceptance on all three channels (ready combinationally asserted while
//   able to accept, data sampled the cycle valid&&ready both hold) and
//   performs no additional skid/elastic buffering of its own; the
//   distinction above concerns only what sits upstream of this module's
//   ports at the top level. (Note: ASM-006 is the RESULT-PACKING
//   assumption -- row-major y_data packing per REQ-GEMM-009 -- and is
//   unrelated to channel de-skidding; it does not apply to this note.)
//
// -----------------------------------------------------------------------------
// FSM (REQ-STATE-001): IDLE, DECODE, LOAD_WEIGHT, STREAM_ACT_MAC, REQUANT,
//   WRITEBACK. No other top-level states are used (per RTLPlan constraint).
//   Phase order (REQ-GEMM-011): IDLE -> DECODE -> LOAD_WEIGHT ->
//   STREAM_ACT_MAC -> REQUANT -> WRITEBACK, looping WRITEBACK back into
//   STREAM_ACT_MAC (activation reload skipped, weights reload skipped) for
//   each remaining output tile (m_idx,n_idx), returning to IDLE once all
//   tiles_m*tiles_n output tiles have been produced (REQ-STATE-002).
//
//   Sub-phases (registers only, not separate top-level FSM states):
//     STREAM_ACT_MAC: sa_phase in {SA_ACT_LOAD, SA_CLEAR, SA_READ, SA_STEP}
//       - SA_ACT_LOAD entered only the first time a job reaches
//         STREAM_ACT_MAC (act_loaded==0); loads the full activation tensor.
//       - SA_CLEAR/SA_READ/SA_STEP implement the per-output-tile K-tile x N
//         rank-1 accumulation sequence with 1-cycle buffer read latency
//         (address+re asserted in SA_READ, pe_mac_en asserted one cycle
//         later in SA_STEP once the buffer word is valid).
//     REQUANT: rq_phase in {RQ_ISSUE, RQ_WAIT} implements the 1-cycle
//       requant pipeline latency (rq_in_valid pulsed in RQ_ISSUE, result
//       sampled in RQ_WAIT once rq_out_valid arrives), one row i=0..N-1 at
//       a time, feeding WRITEBACK.
//
// Reset (REQ-GEMM-010 / ERR-004): synchronous, active-low. state->IDLE, all
//   counters/pointers/flags cleared, y_valid/w_ready/a_ready/busy/done
//   deasserted, err_overflow cleared.
//
// Requirements owned by this file: REQ-STATE-001, REQ-STATE-002,
//   REQ-GEMM-001, REQ-GEMM-002, REQ-GEMM-003, REQ-GEMM-005, REQ-GEMM-009,
//   REQ-GEMM-010, REQ-GEMM-011, REQ-GEMM-012, REQ-PROTO-002, REQ-PROTO-004,
//   REQ-TIME-001 (control-owned registered outputs), REQ-TIME-003,
//   ERR-002 (UQ-003), ERR-003 (UQ-004).
// =============================================================================

/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
module gemm_ctrl #(
    parameter int N            = 8,
    parameter int DATA_W       = 8,
    parameter int ACC_W        = 32,  // kept for port-list parity with datapath modules; not consumed directly by ctrl (acc_row wiring is top-level's job)
    parameter int SCALE_W      = 32,
    parameter int SHIFT_W      = 6,
    parameter int WBUF_DEPTH   = 64,
    parameter int ABUF_DEPTH   = 64,
    parameter int MAX_TILES_M  = 8,
    parameter int MAX_TILES_K  = 8,
    parameter int MAX_TILES_N  = 8,
    parameter int DESC_W       = 128,
    localparam int PW          = N*DATA_W,
    localparam int WAW         = $clog2(WBUF_DEPTH),
    localparam int AAW         = $clog2(ABUF_DEPTH),
    localparam int ROWSEL_W    = $clog2(N)
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // descriptor channel
    input  logic                  desc_valid,
    output logic                  desc_ready,
    input  logic [DESC_W-1:0]     desc_data,

    // weight load channel
    input  logic                  w_valid,
    output logic                  w_ready,
    input  logic                  w_last,

    // activation load channel
    input  logic                  a_valid,
    output logic                  a_ready,
    input  logic                  a_last,

    // result channel
    output logic                  y_valid,
    input  logic                  y_ready,
    output logic [PW-1:0]         y_data,
    output logic                  y_last,

    // status
    output logic                  busy,
    output logic                  done,
    output logic                  err_overflow,

    // weight buffer control
    output logic                  wbuf_we,
    output logic [WAW-1:0]        wbuf_waddr,
    output logic                  wbuf_re,
    output logic [WAW-1:0]        wbuf_raddr,

    // activation buffer control
    output logic                  abuf_we,
    output logic [AAW-1:0]        abuf_waddr,
    output logic                  abuf_re,
    output logic [AAW-1:0]        abuf_raddr,

    // PE array control
    output logic                  pe_clear_acc,
    output logic                  pe_mac_en,
    output logic [ROWSEL_W-1:0]   pe_row_sel,

    // requant control
    output logic                  rq_in_valid,
    output logic [SCALE_W-1:0]    rq_req_scale,
    output logic [SHIFT_W-1:0]    rq_req_shift,
    input  logic                  rq_out_valid,
    input  logic [PW-1:0]         rq_y_vec,
    input  logic                  rq_ovf
);

    // -------------------------------------------------------------------
    // Top-level FSM state (REQ-STATE-001) : exactly these 6 states.
    // -------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,
        DECODE,
        LOAD_WEIGHT,
        STREAM_ACT_MAC,
        REQUANT,
        WRITEBACK
    } state_e;

    state_e state;

    // STREAM_ACT_MAC internal sub-phase (register, not a top-level state).
    typedef enum logic [1:0] {
        SA_ACT_LOAD,
        SA_CLEAR,
        SA_READ,
        SA_STEP
    } sa_phase_e;

    sa_phase_e sa_phase;

    // REQUANT internal sub-phase (register, not a top-level state).
    typedef enum logic {
        RQ_ISSUE,
        RQ_WAIT
    } rq_phase_e;

    rq_phase_e rq_phase;

    // -------------------------------------------------------------------
    // Latched descriptor fields.
    // -------------------------------------------------------------------
    logic [7:0]           tiles_m_r, tiles_n_r, tiles_k_r;
    logic [SHIFT_W-1:0]   req_shift_r;
    logic [SCALE_W-1:0]   req_scale_r;

    // -------------------------------------------------------------------
    // Load counters (generic 16-bit: max possible word count is
    // MAX_TILES_K*MAX_TILES_N*N or MAX_TILES_M*MAX_TILES_K*N, which for the
    // mandated defaults (8x8x8=512) fits comfortably; ports themselves are
    // truncated to the mandated WAW/AAW widths on assignment).
    // -------------------------------------------------------------------
    logic [15:0] wr_cnt, a_cnt;
    logic [15:0] expected_wwords, expected_awords;
    logic [15:0] eff_wwords, eff_awords;
    // NOTE: activation-load-once semantics are implemented directly via the
    // explicit sa_phase transition taken at each entry point into
    // STREAM_ACT_MAC (SA_ACT_LOAD only from DECODE, SA_CLEAR when looping
    // back from WRITEBACK for the next output tile) rather than via a
    // separate "act_loaded" status register, since the entering state
    // already statically determines which case applies.

    // Output-tile loop indices and K-accumulation indices.
    logic [7:0]        m_idx, n_idx, k_tile;
    logic [ROWSEL_W-1:0] k_cnt;
    logic [ROWSEL_W-1:0] row_i;

    logic [15:0] widx, aidx;
    logic        widx_valid, aidx_valid;

    logic is_last_tile;

    logic err_overflow_r;

    // Registered writeback outputs.
    logic [PW-1:0] y_data_r;
    logic          y_valid_r;
    logic          y_last_r;

    // -------------------------------------------------------------------
    // Combinational helpers.
    // -------------------------------------------------------------------
    // All operands widened to 16 bits before multiply/add so the maximum
    // representable product (MAX_TILES*MAX_TILES*N, well under 65536 for the
    // mandated defaults and any reasonably sized MAX_TILES_*/N) never
    // truncates; explicit widths avoid implicit WIDTH warnings.
    assign expected_wwords = 16'(tiles_k_r) * 16'(tiles_n_r) * 16'(N);
    assign expected_awords = 16'(tiles_m_r) * 16'(tiles_k_r) * 16'(N);

    assign widx = (16'(n_idx) * 16'(tiles_k_r) + 16'(k_tile)) * 16'(N) + 16'(k_cnt);
    assign aidx = (16'(m_idx) * 16'(tiles_k_r) + 16'(k_tile)) * 16'(N) + 16'(k_cnt);
    assign widx_valid = (widx < eff_wwords);
    assign aidx_valid = (aidx < eff_awords);

    assign is_last_tile = (m_idx == (tiles_m_r - 8'd1)) && (n_idx == (tiles_n_r - 8'd1));

    assign busy         = (state != IDLE);
    assign err_overflow = err_overflow_r;
    assign y_valid       = y_valid_r;
    assign y_data        = y_data_r;
    assign y_last        = y_last_r;
    assign done          = y_valid_r && y_ready && y_last_r;

    // -------------------------------------------------------------------
    // Descriptor legality check (UQ-003, extended by buffer-capacity
    // guard), combinational on latched fields.
    //
    // Buffer-capacity guard: expected_wwords/expected_awords (computed above
    // as 16-bit widened products, wide enough not to overflow for any
    // reasonably sized MAX_TILES_*/N) are compared against WBUF_DEPTH/
    // ABUF_DEPTH. A descriptor that is legal by tile-count alone can still
    // demand more words than the buffers hold (e.g. defaults N=8,
    // MAX_TILES_*=8, WBUF_DEPTH=ABUF_DEPTH=64 permit a tile-count-legal
    // descriptor needing up to 512 words), which would otherwise silently
    // wrap wbuf_waddr/abuf_waddr (WAW/AAW-bit address ports) and corrupt
    // buffer contents with no error indication. Rejecting here guarantees
    // every ACCEPTED descriptor's word requirement fits within the
    // configured buffer depths, so wbuf_waddr/abuf_waddr never wrap.
    // -------------------------------------------------------------------
    logic illegal_desc;
    assign illegal_desc = (tiles_m_r == 8'd0) || (tiles_m_r > MAX_TILES_M[7:0]) ||
                           (tiles_k_r == 8'd0) || (tiles_k_r > MAX_TILES_K[7:0]) ||
                           (tiles_n_r == 8'd0) || (tiles_n_r > MAX_TILES_N[7:0]) ||
                           (expected_wwords > 16'(WBUF_DEPTH)) ||
                           (expected_awords > 16'(ABUF_DEPTH));

    // -------------------------------------------------------------------
    // Combinational handshake / control-strobe outputs (default-0, no
    // latches: every branch of every case is covered by the initial
    // defaults below).
    // -------------------------------------------------------------------
    always_comb begin
        // Defaults (no latches).
        desc_ready   = 1'b0;
        w_ready      = 1'b0;
        a_ready      = 1'b0;

        wbuf_we      = 1'b0;
        wbuf_waddr   = '0;
        wbuf_re      = 1'b0;
        wbuf_raddr   = '0;

        abuf_we      = 1'b0;
        abuf_waddr   = '0;
        abuf_re      = 1'b0;
        abuf_raddr   = '0;

        pe_clear_acc = 1'b0;
        pe_mac_en    = 1'b0;
        pe_row_sel   = '0;

        rq_in_valid   = 1'b0;
        rq_req_scale  = req_scale_r;
        rq_req_shift  = req_shift_r;

        unique case (state)
            IDLE: begin
                desc_ready = 1'b1; // desc_ready asserted only here (REQ-GEMM-001/PROTO-004)
            end

            DECODE: begin
                // single-cycle validate; no outputs asserted
            end

            LOAD_WEIGHT: begin
                w_ready = 1'b1;
                if (w_valid && w_ready) begin
                    wbuf_we    = 1'b1;
                    wbuf_waddr = wr_cnt[WAW-1:0];
                end
            end

            STREAM_ACT_MAC: begin
                unique case (sa_phase)
                    SA_ACT_LOAD: begin
                        a_ready = 1'b1;
                        if (a_valid && a_ready) begin
                            abuf_we    = 1'b1;
                            abuf_waddr = a_cnt[AAW-1:0];
                        end
                    end
                    SA_CLEAR: begin
                        pe_clear_acc = 1'b1;
                    end
                    SA_READ: begin
                        if (widx_valid) begin
                            wbuf_re    = 1'b1;
                            wbuf_raddr = widx[WAW-1:0];
                        end
                        if (aidx_valid) begin
                            abuf_re    = 1'b1;
                            abuf_raddr = aidx[AAW-1:0];
                        end
                    end
                    SA_STEP: begin
                        if (widx_valid && aidx_valid) begin
                            pe_mac_en = 1'b1;
                        end
                    end
                    default: ; // unreachable
                endcase
            end

            REQUANT: begin
                pe_row_sel = row_i;
                if (rq_phase == RQ_ISSUE) begin
                    rq_in_valid = 1'b1;
                end
            end

            WRITEBACK: begin
                pe_row_sel = row_i;
                // y_valid/y_data/y_last are registered (see sequential block).
            end

            default: ; // unreachable
        endcase
    end

    // -------------------------------------------------------------------
    // Sequential state, counters, and registered writeback/status outputs.
    // -------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= IDLE;
            sa_phase     <= SA_ACT_LOAD;
            rq_phase     <= RQ_ISSUE;

            tiles_m_r    <= '0;
            tiles_n_r    <= '0;
            tiles_k_r    <= '0;
            req_shift_r  <= '0;
            req_scale_r  <= '0;

            wr_cnt       <= '0;
            a_cnt        <= '0;
            eff_wwords   <= '0;
            eff_awords   <= '0;

            m_idx        <= '0;
            n_idx        <= '0;
            k_tile       <= '0;
            k_cnt        <= '0;
            row_i        <= '0;

            err_overflow_r <= 1'b0;

            y_valid_r    <= 1'b0;
            y_data_r     <= '0;
            y_last_r     <= 1'b0;

        end else begin
            unique case (state)

                // -------------------------------------------------------
                IDLE: begin
                    if (desc_valid && desc_ready) begin
                        tiles_m_r      <= desc_data[7:0];
                        tiles_k_r      <= desc_data[15:8];
                        tiles_n_r      <= desc_data[23:16];
                        req_shift_r    <= desc_data[29:24];
                        req_scale_r    <= desc_data[63:32];
                        err_overflow_r <= 1'b0; // REQ-GEMM-012 clear on new job
                        state          <= DECODE;
                    end
                end

                // -------------------------------------------------------
                DECODE: begin
                    if (illegal_desc) begin
                        // UQ-003: silently discard, no side effects, back to IDLE.
                        state <= IDLE;
                    end else begin
                        wr_cnt     <= '0;
                        eff_wwords <= '0;
                        a_cnt      <= '0;
                        eff_awords <= '0;
                        m_idx      <= '0;
                        n_idx      <= '0;
                        k_tile     <= '0;
                        k_cnt      <= '0;
                        row_i      <= '0;
                        sa_phase   <= SA_ACT_LOAD;
                        state      <= LOAD_WEIGHT;
                    end
                end

                // -------------------------------------------------------
                LOAD_WEIGHT: begin
                    if (w_valid && w_ready) begin
                        logic [15:0] new_wr_cnt;
                        new_wr_cnt = wr_cnt + 16'd1;
                        if ((new_wr_cnt >= expected_wwords) || w_last) begin
                            eff_wwords <= new_wr_cnt;
                            wr_cnt     <= '0;
                            sa_phase   <= SA_ACT_LOAD;
                            state      <= STREAM_ACT_MAC;
                        end else begin
                            wr_cnt <= new_wr_cnt;
                        end
                    end
                end

                // -------------------------------------------------------
                STREAM_ACT_MAC: begin
                    unique case (sa_phase)

                        SA_ACT_LOAD: begin
                            if (a_valid && a_ready) begin
                                logic [15:0] new_a_cnt;
                                new_a_cnt = a_cnt + 16'd1;
                                if ((new_a_cnt >= expected_awords) || a_last) begin
                                    eff_awords <= new_a_cnt;
                                    a_cnt      <= '0;
                                    sa_phase   <= SA_CLEAR;
                                end else begin
                                    a_cnt <= new_a_cnt;
                                end
                            end
                        end

                        SA_CLEAR: begin
                            k_tile   <= '0;
                            k_cnt    <= '0;
                            sa_phase <= SA_READ;
                        end

                        SA_READ: begin
                            sa_phase <= SA_STEP;
                        end

                        SA_STEP: begin
                            // advance k, then k_tile; when K-accumulation for
                            // this output tile completes, move to REQUANT.
                            if (k_cnt == ROWSEL_W'(N-1)) begin
                                k_cnt <= '0;
                                if (k_tile == (tiles_k_r - 8'd1)) begin
                                    // K-accumulation for this tile complete.
                                    k_tile   <= '0;
                                    row_i    <= '0;
                                    rq_phase <= RQ_ISSUE;
                                    state    <= REQUANT;
                                end else begin
                                    k_tile   <= k_tile + 8'd1;
                                    sa_phase <= SA_READ;
                                end
                            end else begin
                                k_cnt    <= k_cnt + 1'b1;
                                sa_phase <= SA_READ;
                            end
                        end

                        default: ; // unreachable
                    endcase
                end

                // -------------------------------------------------------
                REQUANT: begin
                    unique case (rq_phase)
                        RQ_ISSUE: begin
                            rq_phase <= RQ_WAIT;
                        end
                        RQ_WAIT: begin
                            if (rq_out_valid) begin
                                y_data_r       <= rq_y_vec;
                                y_last_r       <= (row_i == ROWSEL_W'(N-1)) && is_last_tile;
                                y_valid_r      <= 1'b1;
                                if (rq_ovf) begin
                                    err_overflow_r <= 1'b1; // sticky (ERR-001/REQ-GEMM-012)
                                end
                                state <= WRITEBACK;
                            end
                        end
                        default: ; // unreachable
                    endcase
                end

                // -------------------------------------------------------
                WRITEBACK: begin
                    if (y_valid_r && y_ready) begin
                        y_valid_r <= 1'b0;
                        if (row_i == ROWSEL_W'(N-1)) begin
                            // tile's N rows all written
                            if (is_last_tile) begin
                                // job complete
                                row_i      <= '0;
                                m_idx      <= '0;
                                n_idx      <= '0;
                                state      <= IDLE;
                            end else begin
                                if (n_idx == (tiles_n_r - 8'd1)) begin
                                    n_idx <= '0;
                                    m_idx <= m_idx + 8'd1;
                                end else begin
                                    n_idx <= n_idx + 8'd1;
                                end
                                row_i    <= '0;
                                sa_phase <= SA_CLEAR; // weights+acts already loaded
                                state    <= STREAM_ACT_MAC;
                            end
                        end else begin
                            row_i    <= row_i + 1'b1;
                            rq_phase <= RQ_ISSUE;
                            state    <= REQUANT;
                        end
                    end
                end

                default: state <= IDLE; // unreachable
            endcase
        end
    end

endmodule
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on UNUSEDPARAM */
