// =============================================================================
// File        : gemm_requant.sv
// Module      : gemm_requant
// Purpose     : Per-row requantization of N signed INT32 accumulators down
//               to N saturated signed INT8 results, using a signed scale
//               multiply and arithmetic right shift, with saturation and
//               sticky-overflow reporting.
//
// Parameters  :
//   N        - number of lanes (accumulators per row), default 8
//   DATA_W   - output INT8 width, default 8
//   ACC_W    - input INT32 accumulator width, default 32
//   SCALE_W  - signed scale multiplier width, default 32
//   SHIFT_W  - shift amount width, default 6
//   PW       - packed output vector width = N*DATA_W (local)
//
// Requirements implemented:
//   REQ-GEMM-006 - requantize: prod = signed(acc) * signed(req_scale);
//                  shifted = prod >>> req_shift
//   REQ-GEMM-007 - saturate shifted result to signed INT8 [-128, 127]
//   REQ-GEMM-013 - all arithmetic is signed two's-complement
//   REQ-TIME-001 (partial) - requant output registered exactly one cycle
//                  after in_valid (out_valid follows in_valid by 1 clk)
//   ERR-001      - ovf = OR of per-lane saturation flags this beat, feeds
//                  a sticky err_overflow register upstream (owned by FSM/
//                  status logic, not this module)
//
// Rounding policy:
//   The right shift (>>> ) on the signed product is an ARITHMETIC shift
//   which TRUNCATES TOWARD -INFINITY (i.e. floor division by 2**shift),
//   not round-to-nearest and not truncate-toward-zero. This is the
//   documented rounding policy for this module: TRUNCATE (floor), per
//   REQ-GEMM-006.
//
// Notes:
//   - Internal product width is ACC_W+SCALE_W bits, sufficient to hold the
//     full signed product of an ACC_W x SCALE_W multiply with no overflow
//     prior to the shift.
//   - Pipeline: exactly one register stage. in_valid at cycle T results in
//     out_valid=1 with valid y_vec/ovf at cycle T+1.
//   - On rst_n==0 (synchronous, active-low): out_valid=0, ovf=0. y_vec is
//     also cleared for determinism (not required to hold data reset).
//   - No latches: all combinational lane logic is used only to feed the
//     single registered pipeline stage.
// =============================================================================

module gemm_requant #(
    parameter int N        = 8,
    parameter int DATA_W   = 8,
    parameter int ACC_W    = 32,
    parameter int SCALE_W  = 32,
    parameter int SHIFT_W  = 6,
    parameter int PW       = N*DATA_W
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,       // one row of N accs presented
    input  logic [N*ACC_W-1:0]  acc_in,         // N signed INT32, elem j @ j*ACC_W
    input  logic [SCALE_W-1:0]  req_scale,      // signed INT32 multiplier
    input  logic [SHIFT_W-1:0]  req_shift,      // arithmetic right shift amt
    output logic                out_valid,      // registered, 1 cyc after in_valid
    output logic [PW-1:0]       y_vec,          // N saturated signed INT8
    output logic                ovf             // OR of per-lane saturation
);

    localparam int PROD_W = ACC_W + SCALE_W;

    logic signed [ACC_W-1:0]   acc_elem   [0:N-1];
    logic signed [SCALE_W-1:0] scale_s;
    logic signed [PROD_W-1:0]  prod       [0:N-1];
    logic signed [PROD_W-1:0]  shifted    [0:N-1];
    logic signed [DATA_W-1:0]  sat_elem   [0:N-1];
    logic                      lane_ovf   [0:N-1];

    assign scale_s = req_scale;

    genvar gj;
    generate
        for (gj = 0; gj < N; gj++) begin : g_lane

            // Unpack accumulator lane
            assign acc_elem[gj] = acc_in[gj*ACC_W +: ACC_W];

            // Signed multiply, sized to avoid overflow before shift
            assign prod[gj] = acc_elem[gj] * scale_s;

            // Arithmetic right shift: truncates toward -infinity (floor).
            // Documented rounding policy: TRUNCATE (no rounding applied).
            assign shifted[gj] = prod[gj] >>> req_shift;

            // Saturate to signed INT8 [-128, 127]
            localparam signed [PROD_W-1:0] INT8_MAX = 127;
            localparam signed [PROD_W-1:0] INT8_MIN = -128;

            always_comb begin
                if (shifted[gj] > INT8_MAX) begin
                    sat_elem[gj] = INT8_MAX[DATA_W-1:0];
                    lane_ovf[gj] = 1'b1;
                end else if (shifted[gj] < INT8_MIN) begin
                    sat_elem[gj] = INT8_MIN[DATA_W-1:0];
                    lane_ovf[gj] = 1'b1;
                end else begin
                    sat_elem[gj] = shifted[gj][DATA_W-1:0];
                    lane_ovf[gj] = 1'b0;
                end
            end

        end
    endgenerate

    // Combine per-lane saturation flags into a single overflow bit
    logic ovf_comb;
    always_comb begin
        ovf_comb = 1'b0;
        for (int k = 0; k < N; k++) begin
            ovf_comb |= lane_ovf[k];
        end
    end

    // Pack saturated lanes into combinational y_vec candidate
    logic [PW-1:0] y_vec_comb;
    generate
        for (gj = 0; gj < N; gj++) begin : g_pack
            assign y_vec_comb[gj*DATA_W +: DATA_W] = sat_elem[gj];
        end
    endgenerate

    // Single-stage output register: out_valid follows in_valid by 1 cycle
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            ovf       <= 1'b0;
            y_vec     <= '0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                y_vec <= y_vec_comb;
                ovf   <= ovf_comb;
            end else begin
                ovf <= 1'b0;
            end
        end
    end

endmodule
