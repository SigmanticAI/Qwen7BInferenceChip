// cq_key_path.sv — per-channel grouped INT4 key quant with an FP16 outlier lane
// (clean-room). Mirrors cq_codec.py compress_keys / decompress_keys for one
// group: buffer up to G tokens, take a per-CHANNEL amax over the g buffered
// tokens (partial final group uses g, not G — §3.1), turn each into an fp16
// group scale, quantize every token's keep channels to INT4, and emit one
// record per token. Outlier channels bypass quant (kvq stores their raw fp16
// with code +1; identity replay, D-010/B-4).
//
// F1.5 AREA REWORK: the original elaboration instantiated D cq_scale_unit and
// D cq_quant_unit lanes (2*D combinational dividers per instance — with the
// value path lanes, the #1 F1 area item). This version keeps ONE of each and
// walks channels one per cycle; the residual buffer's read port is registered
// (BRAM-inferable, see residual_buffer.sv) and the scale bank is written one
// channel per cycle.
//
// S10 TIMING REWORK: the shared scale/quant units are the PIPELINED
// compositions of the same cq_fp_pkg functions (cq_scale_pipe / cq_quant_pipe,
// II=1, in-order, latency LS/LQ = 16 at 1 step/stage); each walk splits into
// an ISSUE side (one channel per cycle, tag = chan) and a RETIRE side:
//   * S_SCALE retires straight into the scale bank (wr_idx = tag; the outlier
//     1.0 park moves to the retire side, same values); the FSM advances to
//     S_EMIT only on the LAST retire, so all D bank writes structurally
//     precede the first tok_valid (D-026).
//   * S_EMIT retires recompute the compacted keep slot (kk_wb) in strict
//     issue order; tok_valid pulses ONE cycle after the CH_LAST retire, so
//     tok_codes/emit_vec/scales_bus are all stable during the pulse.
// Numeric results are bit-identical (same functions, same operands); only
// the cycle counts change:
//   * amax sweep: 2 cycles per buffered token (address/consume);
//   * scale phase: D + LS cycles (issue walk + pipe drain) — was D;
//   * emit: D + LQ + 2 cycles per token (prime + issue walk + drain +
//     tok_valid) — was D+2.
// Group close -> last record is now ~2g + (D + LS) + g*(D + LQ + 2) cycles;
// every consumer (kvq_engine ST_KEMIT, tb_cores) is handshake-driven on
// tok_valid/group_valid, and collecting the group costs g*D input beats, so
// the steady-state rate stays input-bound within ~2x.
//
// Handshake expected by kvq_engine.sv: tokens arrive as `in_valid` pulses with
// group_start / group_last flags; a partial group is closed by a `flush` pulse
// (D-008). After a group closes the core streams `tok_valid` beats (one per
// token, carrying tok_idx / tok_codes / scales_bus / emit_vec) and raises
// `group_valid` on the last, then returns to idle. `clear` (D-020 B-2) aborts
// all in-flight group state AND flushes both pipes (every stage valid dies in
// one clock — no stale retire can write the bank or a token slot later); the
// pipes are also defensively flushed on S_AMAX entry, though a live pipe can
// never overlap S_COLL by construction. busy covers the drain states (S_SCALE/
// S_EMIT end on retires, never on issues). The {dec_codes,dec_scales,dec_idx}
// ->dec_hat port is a combinational fp32 read of one channel.
`default_nettype none
module cq_key_path #(
    parameter int unsigned D  = 16,
    parameter int unsigned DW = 16,
    parameter int unsigned G  = 4
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clear,
    input  wire [D-1:0]         outlier_mask,
    input  wire                 in_valid,
    input  wire [D*DW-1:0]      in_vec,
    input  wire                 group_start,
    input  wire                 group_last,
    input  wire                 flush,
    output wire                 busy,
    output wire                 group_valid,
    output wire [D*DW-1:0]      scales_bus,      // per-channel fp16 field (keep)
    output wire [$clog2(G+1)-1:0] g_out,
    output wire                 tok_valid,
    output wire [$clog2(G)-1:0] tok_idx,
    output wire [(D/2)*8-1:0]   tok_pay,         // unused by kvq
    output wire [D*8-1:0]       tok_codes,       // compacted keep codes (nibble)
    output wire [D*DW-1:0]      emit_vec,        // emitting token raw fp16
    input  wire [D*8-1:0]       dec_codes,
    input  wire [D*DW-1:0]      dec_scales,
    input  wire [$clog2(D)-1:0] dec_idx,
    output wire [31:0]          dec_hat
);
    localparam int unsigned CNT_W = $clog2(G+1);
    localparam int unsigned IDX_W = $clog2(G);
    localparam int unsigned CH_W  = $clog2(D);
    localparam [CH_W-1:0]   CH_LAST = CH_W'(D - 1);
    localparam [15:0]       ONE_F16 = 16'h3C00;   // 1.0, guards outlier div-by-0

    localparam [1:0] S_COLL = 2'd0, S_AMAX = 2'd1, S_SCALE = 2'd2, S_EMIT = 2'd3;
    reg [1:0]        state;
    reg [CNT_W-1:0]  tcnt;      // tokens fed in the current group
    reg [CNT_W-1:0]  g_reg;     // frozen group size
    reg [IDX_W-1:0]  ptr;       // token pointer (amax sweep / emit)
    reg [CH_W-1:0]   chan;      // channel pointer (scale / emit issue walks)
    reg              ph;        // 0 = address/prime cycle, 1 = consume/walk
    reg              iss_done;  // current walk: all D channels issued
    reg [CH_W-1:0]   kk_wb;     // compacted keep-code slot (RETIRE side)
    reg [D*DW-1:0]   amax_acc;  // per-channel running amax over the group
    reg [D*8-1:0]    tok_codes_r;
    reg              tv_r;      // 1-cycle tok_valid pulse

    // ── residual buffer (the in-flight token store; registered read) ────────
    wire               rb_wr_valid = (state == S_COLL) && in_valid;
    wire [D*DW-1:0]    rb_rd_vec;   // = mem[ptr] one cycle after ptr settles
    wire [CNT_W-1:0]   rb_fill;     // g is tracked locally; expose but do not use
    residual_buffer #(.DIM(D), .DW(DW), .G(G)) u_rbuf (
        .clk(clk), .rst_n(rst_n),
        .wr_valid(rb_wr_valid), .wr_vec(in_vec), .clear(group_start),
        .fill(rb_fill), .rd_idx(ptr), .rd_vec(rb_rd_vec)
    );
    localparam [DW-1:0] MAG_MASK = {1'b0, {(DW-1){1'b1}}};   // 0x7FFF

    // pipes flushed on clear (abort) and, defensively, on S_AMAX entry —
    // a live pipe never overlaps S_COLL by construction (S10)
    wire amax_entry = (state == S_COLL)
                    && ((in_valid && group_last)
                        || (!in_valid && flush && (tcnt != 0)));
    wire pipe_flush = clear || amax_entry;

    // ── ONE scale pipe, issued over channels in S_SCALE, retiring into the
    //    scale bank (tag-indexed; outlier park on the retire side) ───────────
    wire            sp_ov;
    wire [15:0]     sp_scale;
    wire [CH_W-1:0] sp_tag;
    cq_scale_pipe #(.TAG_W(CH_W)) u_scale (
        .clk(clk), .rst_n(rst_n), .flush(pipe_flush),
        .bits (4'd4),
        .in_valid((state == S_SCALE) && !iss_done),
        .amax (amax_acc[chan*DW +: DW]),
        .tag_in(chan),
        .out_valid(sp_ov), .scale(sp_scale), .tag_out(sp_tag)
    );

    wire [D*DW-1:0] bank_bus;
    scale_bank #(.D(D), .SW(DW)) u_bank (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .wr_en  ((state == S_SCALE) && sp_ov),
        .wr_idx (sp_tag),
        // outlier channels are not quantized; park a benign 1.0 in the bank
        .wr_data(outlier_mask[sp_tag] ? ONE_F16 : sp_scale),
        .bus    (bank_bus)
    );

    // ── ONE quant pipe, issued over channels in S_EMIT (ph 1) ───────────────
    wire            qp_ov;
    wire [8:0]      qp_code;
    wire [CH_W-1:0] qp_tag;
    cq_quant_pipe #(.TAG_W(CH_W)) u_quant (
        .clk(clk), .rst_n(rst_n), .flush(pipe_flush),
        .bits(4'd4),
        .in_valid((state == S_EMIT) && ph && !iss_done),
        .x   (rb_rd_vec[chan*DW +: DW]),
        .s   (bank_bus[chan*DW +: DW]),
        .tag_in(chan),
        .out_valid(qp_ov), .code(qp_code), .tag_out(qp_tag)
    );

    // per-channel keep field, combinational (bank is stable through the emit)
    genvar c;
    generate
        for (c = 0; c < int'(D); c = c + 1) begin : g_sbus
            assign scales_bus[c*DW +: DW] =
                outlier_mask[c] ? {DW{1'b0}} : bank_bus[c*DW +: DW];
        end
    endgenerate

    // ── control FSM ─────────────────────────────────────────────────────────
    integer jc;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_COLL; tcnt <= '0; g_reg <= '0; ptr <= '0; chan <= '0;
            ph <= 1'b0; iss_done <= 1'b0; kk_wb <= '0; amax_acc <= '0;
            tok_codes_r <= '0; tv_r <= 1'b0;
        end else if (clear) begin
            state <= S_COLL; tcnt <= '0; ptr <= '0; chan <= '0;
            ph <= 1'b0; iss_done <= 1'b0; tv_r <= 1'b0;   // abort in-flight
        end else begin
            case (state)
                S_COLL: begin
                    if (in_valid) begin
                        if (group_last) begin
                            g_reg    <= group_start ? CNT_W'(1) : (tcnt + CNT_W'(1));
                            tcnt     <= group_start ? CNT_W'(1) : (tcnt + CNT_W'(1));
                            ptr      <= '0;
                            ph       <= 1'b0;
                            amax_acc <= '0;
                            state    <= S_AMAX;
                        end else begin
                            tcnt <= group_start ? CNT_W'(1) : (tcnt + CNT_W'(1));
                        end
                    end else if (flush && (tcnt != 0)) begin
                        g_reg    <= tcnt;
                        ptr      <= '0;
                        ph       <= 1'b0;
                        amax_acc <= '0;
                        state    <= S_AMAX;
                    end
                end
                // amax sweep: 2 cycles/token — ph 0 lets the registered read
                // settle on token `ptr`; ph 1 folds it into the accumulators.
                S_AMAX: begin
                    if (!ph) begin
                        ph <= 1'b1;
                    end else begin
                        for (jc = 0; jc < int'(D); jc = jc + 1) begin
                            if (!outlier_mask[jc]
                                && ((rb_rd_vec[jc*DW +: DW] & MAG_MASK)
                                    > amax_acc[jc*DW +: DW]))
                                amax_acc[jc*DW +: DW] <= rb_rd_vec[jc*DW +: DW] & MAG_MASK;
                        end
                        ph <= 1'b0;
                        if (ptr == IDX_W'(g_reg - 1)) begin
                            ptr      <= '0;
                            chan     <= '0;
                            iss_done <= 1'b0;
                            state    <= S_SCALE;
                        end else begin
                            ptr <= ptr + IDX_W'(1);
                        end
                    end
                end
                // scale walk: issue one channel per cycle; retires land in the
                // scale bank below. Advance only on the LAST retire (D-026:
                // every bank write precedes the first tok_valid).
                S_SCALE: begin
                    if (!iss_done) begin
                        if (chan == CH_LAST) iss_done <= 1'b1;
                        else                 chan     <= chan + CH_W'(1);
                    end
                    if (sp_ov && (sp_tag == CH_LAST)) begin
                        chan     <= '0;
                        ptr      <= '0;
                        ph       <= 1'b0;
                        iss_done <= 1'b0;
                        state    <= S_EMIT;
                    end
                end
                // emit: per token — prime the row read (ph 0), issue one
                // channel per cycle, retire the compacted keep codes below,
                // then pulse tok_valid for one cycle with everything stable.
                S_EMIT: begin
                    if (!ph) begin
                        ph       <= 1'b1;          // rb_rd_vec <- mem[ptr]
                        chan     <= '0;
                        kk_wb    <= '0;
                        iss_done <= 1'b0;
                    end else if (tv_r) begin
                        ph <= 1'b0;                // pulse done; next token / idle
                        if (ptr == IDX_W'(g_reg - 1)) begin
                            state <= S_COLL;
                            tcnt  <= '0;
                        end else begin
                            ptr <= ptr + IDX_W'(1);
                        end
                    end else if (!iss_done) begin
                        if (chan == CH_LAST) iss_done <= 1'b1;
                        else                 chan     <= chan + CH_W'(1);
                    end
                end
                default: state <= S_COLL;
            endcase
            // retire side (S_EMIT): strict in-order retires recompute the
            // compacted slot; tv_r fires the cycle AFTER the CH_LAST retire
            // (codes registered -> stable during the pulse) and stays a
            // STRICT 1-cycle pulse.
            tv_r <= qp_ov && (qp_tag == CH_LAST);
            if (qp_ov && !outlier_mask[qp_tag]) begin
                // sign-extend the INT4 code at the compacted slot
                tok_codes_r[kk_wb*8 +: 8] <= {{4{qp_code[3]}}, qp_code[3:0]};
                kk_wb <= kk_wb + CH_W'(1);
            end
        end
    end

    // ── outputs ─────────────────────────────────────────────────────────────
    assign busy        = (state != S_COLL) || (tcnt != 0);
    assign tok_valid   = tv_r;
    assign group_valid = tv_r && (ptr == IDX_W'(g_reg - 1));
    assign tok_idx     = ptr;
    assign tok_codes   = tok_codes_r;
    assign emit_vec    = rb_rd_vec;
    assign tok_pay     = '0;
    assign g_out       = g_reg;

    // ── combinational dequant readback (one channel per beat, D-010) ────────
    wire signed [8:0] dsel = {dec_codes[dec_idx*8 + 7], dec_codes[dec_idx*8 +: 8]};
    cq_dequant_unit u_dec (
        .code(dsel),
        .s   (dec_scales[dec_idx*DW +: DW]),
        .hat (dec_hat)
    );

    // residual_buffer.fill is intentionally unused (group size tracked
    // locally); qp_code[8:4] are the sign-extension bits of the 9-bit code
    // port — the INT4 lane consumes [3:0] only
    wire _unused_kp_ok = &{1'b0, rb_fill, qp_code[8:4]};
endmodule
`default_nettype wire
