// cq_value_path.sv — per-token value compress + one-channel dequant readback
// (clean-room). Mirrors cq_codec.py compress_values / decompress_values for
// a single token: one per-token scale over all D dims (§2), then INT4/INT8
// quant and byte-packing (§5). No cross-token buffering (values stream).
//
// F1.5 AREA REWORK: the original elaboration instantiated D parallel
// cq_quant_unit lanes (D combinational dividers per instance — the #1 F1
// area item). This version walks the token ONE CHANNEL PER CYCLE through a
// single amax comparator, a single scale unit and a single quant unit.
//
// S10 TIMING REWORK: the shared scale/quant units are now the PIPELINED
// compositions of the same cq_fp_pkg functions (cq_scale_pipe /
// cq_quant_pipe, II=1, in-order, latency LS/LQ = 16 at 1 step/stage), and
// the walk is split into an ISSUE side (one channel per cycle, tag = chan)
// and a RETIRE side (tag-indexed writeback on the pipes' out_valid). The
// amax walk gains one operand register (mag_r; the 64:1 element mux no
// longer feeds the compare-and-update cone directly). Numeric results are
// bit-identical (same functions, same operands, same order-preserving
// reduction); only the latency changes:
//   in_valid -> out_valid is now (D+1) (amax walk) + LS (scale pipe)
//   + D (quant issue walk) + LQ (quant drain) ~= 2D+33 cycles (was ~2D+1).
// Both callers (kvq_engine ST_COMPRESS and verif/kvq/cores/tb_cores) are
// handshake-driven on out_valid, and the AXIS input needs >= D beats/token,
// so steady-state throughput stays within ~2x of the input-bound rate.
//
// Timing contract expected by kvq_engine.sv: `in_valid` presents an assembled
// token; `busy` covers the compute; `out_valid` pulses when {out_scale,out_pay}
// are valid and then HELD (kvq samples them in the following ST_STORE cycle).
// HOLD REQUIREMENT (both callers comply): `in_vec` must stay stable from
// the in_valid pulse until out_valid — the walk reads it live instead of
// burning a D*DW-bit copy register (kvq_engine's tok_vec cannot change while
// tready is low in ST_COMPRESS; tb_cores holds vtok through the wait).
// `clear` (D-020 B-2 soft reset) aborts in flight so no out_valid ever fires
// for the aborted token: clear (or an in_valid restart) FLUSHES both pipes —
// every stage valid dies in one clock, so no stale retire can fire later.
// `busy` covers the drain states: st returns to V_IDLE only on the LAST
// RETIRE, never at last issue. The dequant port {dec_codes,dec_scale,dec_idx}
// ->dec_hat is a purely combinational fp32 read of one channel (D-010 bus).
`default_nettype none
module cq_value_path #(
    parameter int unsigned D  = 16,
    parameter int unsigned DW = 16
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clear,
    input  wire [3:0]           bits,       // 4 or 8
    input  wire                 in_valid,
    input  wire [D*DW-1:0]      in_vec,
    output wire                 busy,
    output wire                 out_valid,
    output wire [15:0]          out_scale,
    output wire [D*8-1:0]       out_codes,  // per-elem signed code (8b), unused by kvq
    output wire [D*8-1:0]       out_pay,    // §5 packed byte stream (low bits used)
    input  wire [D*8-1:0]       dec_codes,
    input  wire [15:0]          dec_scale,
    input  wire [$clog2(D)-1:0] dec_idx,
    output wire [31:0]          dec_hat
);
    localparam int unsigned IDXW = $clog2(D);
    localparam [IDXW-1:0]   CH_LAST  = IDXW'(D - 1);
    localparam [DW-1:0]     MAG_MASK = {1'b0, {(DW-1){1'b1}}};   // 0x7FFF

    // ── serialized compress walk: issue one channel per cycle, retire by tag ─
    localparam [2:0] V_IDLE   = 3'd0,   // wait for in_valid
                     V_AMAX   = 3'd1,   // amax walk (mag_r operand register)
                     V_AFIN   = 3'd2,   // lagged final compare -> issue scale
                     V_SCALE  = 3'd3,   // scale pipe in flight
                     V_QUANT  = 3'd4,   // quant issue walk (retires overlap)
                     V_QDRAIN = 3'd5;   // last issue done; awaiting last retire
    reg [2:0]      st;
    reg [IDXW-1:0] chan;
    reg [DW-1:0]   amax_r;      // running per-token amax (sign-cleared)
    reg [DW-1:0]   mag_r;       // S10: registered |lane| (chan leads by one)
    reg            mag_v;       // mag_r holds a live lane
    reg            sc_iss;      // 1-cycle scale-pipe issue pulse
    reg            ov_r;
    reg [15:0]     scale_r;
    reg [D*8-1:0]  pay_r, codes_r;

    // live channel select (in_vec is held by contract — see header)
    wire [DW-1:0] elem = in_vec[chan*DW +: DW];
    wire [DW-1:0] mag  = elem & MAG_MASK;

    // ── the shared arithmetic pipes (ONE of each per instance, S10) ──────────
    // clear aborts mid-divide (D-020 B-2: no late out_valid, no phantom
    // ST_STORE); an in_valid restart kills stale in-flight retires from an
    // aborted walk.
    wire pipe_flush = clear || in_valid;

    wire        sp_ov;
    wire [15:0] sp_scale;
    wire        sp_tag_nc;
    cq_scale_pipe #(.TAG_W(1)) u_scale (
        .clk(clk), .rst_n(rst_n), .flush(pipe_flush),
        .bits(bits), .in_valid(sc_iss), .amax(amax_r), .tag_in(1'b0),
        .out_valid(sp_ov), .scale(sp_scale), .tag_out(sp_tag_nc)
    );

    wire            qp_ov;
    wire [8:0]      qp_code;
    wire [IDXW-1:0] qp_tag;
    cq_quant_pipe #(.TAG_W(IDXW)) u_quant (
        .clk(clk), .rst_n(rst_n), .flush(pipe_flush),
        .bits(bits), .in_valid(st == V_QUANT), .x(elem), .s(scale_r),
        .tag_in(chan),
        .out_valid(qp_ov), .code(qp_code), .tag_out(qp_tag)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st      <= V_IDLE; chan <= '0; amax_r <= '0; ov_r <= 1'b0;
            mag_r   <= '0;     mag_v <= 1'b0; sc_iss <= 1'b0;
            scale_r <= '0;     pay_r <= '0; codes_r <= '0;
        end else begin
            ov_r   <= 1'b0;
            sc_iss <= 1'b0;
            if (clear) begin
                st    <= V_IDLE;                // abort in-flight (D-020 B-2)
                mag_v <= 1'b0;
            end else if (in_valid) begin
                st     <= V_AMAX;               // (re)start on a new token
                chan   <= '0;
                amax_r <= '0;
                mag_v  <= 1'b0;
            end else begin
                case (st)
                    // amax = unsigned max of |lane|; the compare runs one
                    // cycle behind the element mux through mag_r (S10)
                    V_AMAX: begin
                        mag_r <= mag;
                        mag_v <= 1'b1;
                        if (mag_v && (mag_r > amax_r)) amax_r <= mag_r;
                        if (chan == CH_LAST) begin
                            chan <= '0;
                            st   <= V_AFIN;
                        end else begin
                            chan <= chan + IDXW'(1);
                        end
                    end
                    V_AFIN: begin               // lagged compare of last lane
                        if (mag_r > amax_r) amax_r <= mag_r;
                        mag_v  <= 1'b0;
                        sc_iss <= 1'b1;         // amax_r final next cycle
                        st     <= V_SCALE;
                    end
                    V_SCALE: begin              // wait the scale-pipe retire
                        if (sp_ov) begin
                            scale_r <= sp_scale;
                            pay_r   <= '0;
                            codes_r <= '0;
                            chan    <= '0;
                            st      <= V_QUANT;
                        end
                    end
                    V_QUANT: begin              // issue one channel per cycle
                        if (chan == CH_LAST) begin
                            chan <= '0;
                            st   <= V_QDRAIN;
                        end else begin
                            chan <= chan + IDXW'(1);
                        end
                    end
                    default: ;                  // V_IDLE / V_QDRAIN: retires below
                endcase
                // retire side: tag-indexed quant writeback (§5 pack); the
                // token completes on the LAST RETIRE, one drain after issue
                if (qp_ov) begin
                    codes_r[qp_tag*8 +: 8] <= qp_code[7:0];
                    if (bits == 4'd8) pay_r[qp_tag*8 +: 8] <= qp_code[7:0];
                    else              pay_r[qp_tag*4 +: 4] <= qp_code[3:0];
                    if (qp_tag == CH_LAST) begin
                        ov_r <= 1'b1;
                        st   <= V_IDLE;
                    end
                end
            end
        end
    end

    assign busy      = (st != V_IDLE);
    assign out_valid = ov_r;
    assign out_scale = scale_r;
    assign out_pay   = pay_r;
    assign out_codes = codes_r;

    // ── combinational dequant readback (one channel per beat, D-010) ─────────
    wire signed [8:0] dsel = {dec_codes[dec_idx*8 + 7], dec_codes[dec_idx*8 +: 8]};
    cq_dequant_unit u_dec (.code(dsel), .s(dec_scale), .hat(dec_hat));

    // qp_code[8] is the (redundant here) sign-extension bit of the 9-bit code
    // port; the byte echo/pack consume [7:0] only. The scale pipe's 1-bit tag
    // is structurally unused (one scale in flight per token).
    wire _unused_vp_ok = &{1'b0, qp_code[8], sp_tag_nc};
endmodule
`default_nettype wire
