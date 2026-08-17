// =============================================================================
// File        : gemm_abuf.sv
// Module      : gemm_abuf
// Purpose     : Activation buffer for INT8 GEMM systolic accelerator
//               (gemm_systolic_accel). Behavioral single-clock synchronous
//               SRAM / register-file model storing DEPTH packed words of
//               N INT8 activation elements each.
// Requirement : REQ-STORE-002 (activation buffer, ABUF_DEPTH x PW storage array)
// -----------------------------------------------------------------------------
// Parameters:
//   N          - number of INT8 elements packed per word (default 8)
//   DATA_W     - bit width of a single INT8 element (default 8)
//   DEPTH      - number of storage words (default ABUF_DEPTH = 64)
//   PW         - local packed word width = N*DATA_W
//   AW         - local address width = $clog2(DEPTH)
//
// Read/Write contract:
//   - Single write port, single read port, single clock (clk).
//   - Write: synchronous, unconditional on we==1 at posedge clk.
//       we=1  -> mem[waddr] <= wdata  (write completes at this clk edge)
//   - Read: synchronous, REGISTERED, ONE CYCLE LATENCY.
//       re=1 at cycle T (with raddr valid at cycle T)
//         -> rdata updates to mem[raddr] value as of cycle T, visible at cycle T+1.
//       If re=0, rdata holds its previous value (no read update that cycle).
//   - Memory array (mem) is NOT reset. Contents are undefined until written.
//     No reset logic is applied to mem, per spec (behavioral SRAM model).
//   - Simultaneous read/write to same address: this model returns the OLD
//     data (standard read-old-data synchronous RAM behavior for separate
//     read/write ports as coded here).
// -----------------------------------------------------------------------------
// Ownership   : Register/Storage RTL Agent (buffers only). No FSM, PE array,
//               requant, or top-level logic included in this file.
// =============================================================================

module gemm_abuf #(
    parameter int N          = 8,
    parameter int DATA_W     = 8,
    parameter int DEPTH      = 64,           // ABUF_DEPTH default
    localparam int PW        = N*DATA_W,
    localparam int AW        = $clog2(DEPTH)
) (
    input  logic            clk,
    input  logic            we,               // write enable
    input  logic [AW-1:0]   waddr,
    input  logic [PW-1:0]   wdata,            // N packed INT8 activations
    input  logic            re,               // read enable
    input  logic [AW-1:0]   raddr,
    output logic [PW-1:0]   rdata             // registered, 1-cycle read latency
);

    // Storage array: NOT reset (undefined until written), per REQ-STORE-002.
    logic [PW-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we) begin
            mem[waddr] <= wdata;
        end
        if (re) begin
            rdata <= mem[raddr];
        end
    end

endmodule
