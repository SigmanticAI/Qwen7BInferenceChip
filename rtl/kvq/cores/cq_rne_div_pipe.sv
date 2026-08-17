// cq_rne_div_pipe.sv — fully pipelined restoring RNE divider (S10, clean-room).
//
// The REGISTERED composition of cq_fp_pkg's div_front / div_step / div_back:
// stage 0 registers div_front(n, d); the step stages each register
// STEPS_PER_STAGE applications of div_step (bit indices 11 down to 0); the
// final stage registers div_back. Every stage body IS one of those package
// functions — no pipe-private arithmetic — so the exhaustive fparith proof of
// the combinational twin (cq_fp_pkg::rne_div_bounded, the same functions with
// the registers removed) certifies this pipe bit-exact (S10 / D-010 numerics).
//
// Handshake: II = 1 (a new n/d may be presented every cycle with in_valid);
// out_valid pulses LATENCY cycles later with q and the caller's tag echoed.
// The pipe never stalls and never reorders — retires are strictly in issue
// order, which the callers' tag-indexed writeback relies on.
//
// `flush` (soft-reset abort / walk restart, D-020 B-2) clears every stage
// valid in ONE clock — including the intake presented that same cycle — so an
// aborted walk can never retire a stale result later. Data registers are
// don't-care while their valid is low.
//
// LATENCY = 1 (front) + 12/STEPS_PER_STAGE (steps) + 1 (back) = 14 at the
// default 1 step/stage.
`default_nettype none
module cq_rne_div_pipe #(
    parameter int unsigned NW              = 41,  // operand width (pkg contract)
    parameter int unsigned TAG_W           = 8,
    parameter int unsigned STEPS_PER_STAGE = 1    // must divide 12
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             flush,      // kill ALL in-flight work this clock
    input  wire             in_valid,
    input  wire [NW-1:0]    n,
    input  wire [NW-1:0]    d,
    input  wire [TAG_W-1:0] tag_in,
    output wire             out_valid,
    output wire [12:0]      q,
    output wire [TAG_W-1:0] tag_out
);
    localparam int unsigned NSTEP = 12 / STEPS_PER_STAGE;  // step stages

`ifndef SYNTHESIS
    initial begin
        if (NW != 41)
            $fatal(1, "cq_rne_div_pipe: NW (%0d) != 41 (cq_fp_pkg contract)", NW);
        if (NSTEP * STEPS_PER_STAGE != 12)
            $fatal(1, "cq_rne_div_pipe: STEPS_PER_STAGE (%0d) must divide 12",
                   STEPS_PER_STAGE);
    end
`endif

    // STEPS_PER_STAGE div_steps starting at bit index `hi` (comb helper —
    // strictly a composition of cq_fp_pkg::div_step, nothing private)
    function automatic cq_fp_pkg::div_state_t step_n(
        input cq_fp_pkg::div_state_t s_in, input int hi);
        cq_fp_pkg::div_state_t t;
        int j;
        begin
            t = s_in;
            for (j = 0; j < int'(STEPS_PER_STAGE); j = j + 1)
                t = cq_fp_pkg::div_step(t, hi - j);
            step_n = t;
        end
    endfunction

    // carried state after the front stage and after each step stage
    cq_fp_pkg::div_state_t st_r  [0:NSTEP];
    reg [TAG_W-1:0]        tag_r [0:NSTEP];
    reg [NSTEP:0]          vld_r;
    reg [12:0]             q_r;
    reg [TAG_W-1:0]        tago_r;
    reg                    ov_r;

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_r <= '0;
            ov_r  <= 1'b0;
            for (k = 0; k <= int'(NSTEP); k = k + 1) begin
                st_r[k]  <= '0;
                tag_r[k] <= '0;
            end
            q_r    <= '0;
            tago_r <= '0;
        end else begin
            // datapath advances unconditionally (don't-care when invalid)
            st_r[0]  <= cq_fp_pkg::div_front(n, d);
            tag_r[0] <= tag_in;
            for (k = 1; k <= int'(NSTEP); k = k + 1) begin
                st_r[k]  <= step_n(st_r[k-1],
                                   11 - (k-1)*int'(STEPS_PER_STAGE));
                tag_r[k] <= tag_r[k-1];
            end
            q_r    <= cq_fp_pkg::div_back(st_r[NSTEP]);
            tago_r <= tag_r[NSTEP];
            // valid spine: flush (and nothing else) clears every stage in one
            // clock, including the same-cycle intake (D-020 B-2 abort)
            if (flush) begin
                vld_r <= '0;
                ov_r  <= 1'b0;
            end else begin
                vld_r <= {vld_r[NSTEP-1:0], in_valid};
                ov_r  <= vld_r[NSTEP];
            end
        end
    end

    assign out_valid = ov_r;
    assign q         = q_r;
    assign tag_out   = tago_r;
endmodule
`default_nettype wire
