// apex_stream1_sva.svh — REUSABLE single-stream SVA checker for the
// ARCHITECTURE.md §5 valid/ready contract, derived from apex_stream_sva.svh
// (the MXE job-level pack). The full pack's ports are mxe_desc_t-typed and
// its transaction model is descriptor-driven, so it cannot bind to
// descriptor-less blocks (ASU); this file factors out the per-stream
// properties so every stream in every block gets the same checks.
//
// Written for the Verilator-5.x SVA subset: simple concurrent assertions
// with |=> / $stable only, compiled under --assert as a build gate (D-012).
//
// Checked (§5): while valid && !ready, valid must hold (no retraction) and
// the payload must be stable.
//
// Usage (bind into the block or TB where the stream lives):
//   bind asu_softmax apex_stream1_sva #(.WIDTH(33), .NAME("sm.s")) u_sva_s
//     (.clk(clk), .rst_n(rst_n), .valid(s_valid), .ready(s_ready),
//      .data({s_score, s_last}));

`ifndef APEX_STREAM1_SVA_SVH
`define APEX_STREAM1_SVA_SVH

module apex_stream1_sva #(
  parameter int    WIDTH = 8,
  parameter string NAME  = "stream"
) (
  input logic             clk,
  input logic             rst_n,
  input logic             valid,
  input logic             ready,
  input logic [WIDTH-1:0] data
);

  ap_stable: assert property (@(posedge clk) disable iff (!rst_n)
    (valid && !ready) |=> (valid && $stable(data)))
    else $error("[SVA §5] %s: valid retracted or data unstable under backpressure",
                NAME);

endmodule

`endif // APEX_STREAM1_SVA_SVH
