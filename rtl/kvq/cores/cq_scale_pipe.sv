// cq_scale_pipe.sv — pipelined s = max(amax/qmax, EPS) -> fp16 (S10,
// clean-room). The registered composition of cq_fp_pkg::scale_front (input
// stage: f16 decode / msb binade search / shifted dividend, with the
// EPS / amax<=0 short-circuit carried as a bypass flag), cq_rne_div_pipe
// (the shared restoring divider) and cq_fp_pkg::scale_back (output stage:
// m==2048 renormalize + fp16 pack + EPS mux). Identical functions to the
// combinational twin cq_scale_unit / cq_fp_pkg::scale_from_amax — bit-exact
// by construction, certified by verif/kvq/fparith on the twin.
//
// II = 1, strictly in-order; tag_in echoed on tag_out with the retiring
// scale. `bits` must be STATIC (both callers hardwire or strap it), so qmax
// constant-folds into the divider exactly as in the comb unit. `flush` kills
// all in-flight work in one clock (see cq_rne_div_pipe). The binade e and
// the eps flag ride the divider's tag alongside the caller tag.
//
// LATENCY = 1 (front) + 14 (divider) + 1 (back) = 16 at 1 step/stage.
`default_nettype none
module cq_scale_pipe #(
    parameter int unsigned TAG_W           = 8,
    parameter int unsigned STEPS_PER_STAGE = 1
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             flush,
    input  wire [3:0]       bits,      // 4 (INT4) or 8 (INT8), static
    input  wire             in_valid,
    input  wire [15:0]      amax,      // sign-cleared fp16 magnitude
    input  wire [TAG_W-1:0] tag_in,
    output wire             out_valid,
    output wire [15:0]      scale,     // fp16 scale bits
    output wire [TAG_W-1:0] tag_out
);
    // ── input stage: register scale_front (binade search isolated here) ─────
    reg                     f_vld;
    cq_fp_pkg::scale_front_t f_r;
    reg [TAG_W-1:0]         f_tag;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            f_vld <= 1'b0; f_r <= '0; f_tag <= '0;
        end else begin
            f_r   <= cq_fp_pkg::scale_front(amax, int'(bits));
            f_tag <= tag_in;
            f_vld <= flush ? 1'b0 : in_valid;
        end
    end

    // ── shared divider pipe; e/eps bypass ride the tag ──────────────────────
    wire                d_ov;
    wire [12:0]         d_q;
    wire [TAG_W+32:0]   d_tag;   // {tag, e[31:0], eps}
    cq_rne_div_pipe #(
        .NW(41), .TAG_W(TAG_W + 33), .STEPS_PER_STAGE(STEPS_PER_STAGE)
    ) u_div (
        .clk(clk), .rst_n(rst_n), .flush(flush),
        .in_valid(f_vld), .n(f_r.n), .d(f_r.d),
        .tag_in({f_tag, f_r.e, f_r.eps}),
        .out_valid(d_ov), .q(d_q), .tag_out(d_tag)
    );

    // ── output stage: register scale_back (renormalize + pack + EPS mux) ────
    reg              ov_r;
    reg [15:0]       scale_r;
    reg [TAG_W-1:0]  tago_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ov_r <= 1'b0; scale_r <= '0; tago_r <= '0;
        end else begin
            scale_r <= cq_fp_pkg::scale_back(d_q, signed'(d_tag[1 +: 32]),
                                             d_tag[0]);
            tago_r  <= d_tag[33 +: TAG_W];
            ov_r    <= flush ? 1'b0 : d_ov;
        end
    end

    assign out_valid = ov_r;
    assign scale     = scale_r;
    assign tag_out   = tago_r;
endmodule
`default_nettype wire
