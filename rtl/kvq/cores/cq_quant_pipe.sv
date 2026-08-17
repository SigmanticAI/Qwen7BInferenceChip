// cq_quant_pipe.sv — pipelined q = clamp(rne(x/s), qmin, qmax) (S10,
// clean-room). The registered composition of cq_fp_pkg::quant_front (input
// stage: sign / f16 decode / the 41-bit Ex-Es barrel shift / mx==0 bypass
// flag), cq_rne_div_pipe (the shared restoring divider) and
// cq_fp_pkg::quant_back (output stage: negate + clamp). Identical functions
// to the combinational twin cq_quant_unit / cq_fp_pkg::quant_one — bit-exact
// by construction, certified by verif/kvq/fparith on the twin.
//
// II = 1, strictly in-order; tag_in is echoed on tag_out with the retiring
// code (the callers index their writeback by it). `bits` must be STATIC for
// an instance's lifetime (both callers hardwire or strap it per-config), so
// qmin/qmax constant-fold exactly as in the comb unit. `flush` kills all
// in-flight work in one clock (see cq_rne_div_pipe).
//
// LATENCY = 1 (front) + 14 (divider) + 1 (back) = 16 at 1 step/stage.
`default_nettype none
module cq_quant_pipe #(
    parameter int unsigned TAG_W           = 8,
    parameter int unsigned STEPS_PER_STAGE = 1
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             flush,
    input  wire [3:0]       bits,      // 4 (INT4) or 8 (INT8), static
    input  wire             in_valid,
    input  wire [15:0]      x,         // fp16 element
    input  wire [15:0]      s,         // fp16 scale
    input  wire [TAG_W-1:0] tag_in,
    output wire             out_valid,
    output wire [8:0]       code,      // signed [-128,127]
    output wire [TAG_W-1:0] tag_out
);
    // ── input stage: register quant_front (barrel shift isolated here) ──────
    reg                     f_vld;
    cq_fp_pkg::quant_front_t f_r;
    reg [TAG_W-1:0]         f_tag;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            f_vld <= 1'b0; f_r <= '0; f_tag <= '0;
        end else begin
            f_r   <= cq_fp_pkg::quant_front(x, s);
            f_tag <= tag_in;
            f_vld <= flush ? 1'b0 : in_valid;
        end
    end

    // ── shared divider pipe; neg/zero bypass bits ride the tag ──────────────
    wire               d_ov;
    wire [12:0]        d_q;
    wire [TAG_W+1:0]   d_tag;
    cq_rne_div_pipe #(
        .NW(41), .TAG_W(TAG_W + 2), .STEPS_PER_STAGE(STEPS_PER_STAGE)
    ) u_div (
        .clk(clk), .rst_n(rst_n), .flush(flush),
        .in_valid(f_vld), .n(f_r.n), .d(f_r.d),
        .tag_in({f_tag, f_r.neg, f_r.zero}),
        .out_valid(d_ov), .q(d_q), .tag_out(d_tag)
    );

    // ── output stage: register quant_back (negate + clamp) ──────────────────
    reg             ov_r;
    reg [8:0]       code_r;
    reg [TAG_W-1:0] tago_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ov_r <= 1'b0; code_r <= '0; tago_r <= '0;
        end else begin
            code_r <= 9'(cq_fp_pkg::quant_back(d_q, d_tag[1], d_tag[0],
                                               int'(bits)));
            tago_r <= d_tag[2 +: TAG_W];
            ov_r   <= flush ? 1'b0 : d_ov;
        end
    end

    assign out_valid = ov_r;
    assign code      = code_r;
    assign tag_out   = tago_r;
endmodule
`default_nettype wire
