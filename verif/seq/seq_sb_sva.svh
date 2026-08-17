// seq_sb_sva.svh — SEQ (seq_walker) contract checker, bound INTO seq_walker
// by tb_seq_sb.sv. Compiled under --assert in every build (D-012).
//
// Written for the Verilator 5.x SVA subset: simple concurrent assertions
// with |-> / |=> / $past / $rose only.
//
// Checked contracts:
//   - D-006 job serialization: after a descriptor is accepted on md_*,
//     md_valid must stay LOW until that job's done or desc_error — the
//     next descriptor is never presented while one is in flight;
//   - md_valid only rises when dispatch was enabled;
//   - §5 busy honesty: a presented descriptor, an in-flight job, a
//     non-empty queue, or a draining abort all imply busy;
//   - queue bookkeeping sanity: q_count <= QDEPTH, q_empty/q_full mirror
//     q_count and are mutually exclusive;
//   - abort drains: while `aborting`, the queue is (and stays) empty.
//
// (Payload stability on the ds/md streams is checked by the reusable house
// pack apex_stream1_sva, bound separately by the TB.)

`ifndef SEQ_SB_SVA_SVH
`define SEQ_SB_SVA_SVH

module seq_sb_sva #(
  parameter int unsigned QDEPTH = 8
) (
  input logic clk,
  input logic rst_n,
  input logic enable,
  input logic md_valid,
  input logic md_ready,
  input logic mxe_done,
  input logic mxe_desc_error,
  input logic busy,
  input logic aborting,
  input logic q_empty,
  input logic q_full,
  input logic [$clog2(QDEPTH+1)-1:0] q_count
);

  // job-in-flight model (accept .. done/desc_error, exclusive)
  logic in_flight;
  always_ff @(posedge clk) begin
    if (!rst_n)                          in_flight <= 1'b0;
    else if (md_valid && md_ready)       in_flight <= 1'b1;
    else if (mxe_done || mxe_desc_error) in_flight <= 1'b0;
  end

  ap_d006_serial: assert property (@(posedge clk) disable iff (!rst_n)
    in_flight |-> !md_valid)
    else $error("[SVA seq] D-006 violated: descriptor presented while a job is in flight");

  ap_dispatch_enabled: assert property (@(posedge clk) disable iff (!rst_n)
    $rose(md_valid) |-> $past(enable))
    else $error("[SVA seq] dispatch while CTRL.enable was clear");

  ap_busy_md: assert property (@(posedge clk) disable iff (!rst_n)
    md_valid |-> busy)
    else $error("[SVA seq] busy hole: descriptor presented but not busy");

  ap_busy_inflight: assert property (@(posedge clk) disable iff (!rst_n)
    in_flight |-> busy)
    else $error("[SVA seq] busy hole: job in flight but not busy");

  ap_busy_queue: assert property (@(posedge clk) disable iff (!rst_n)
    (q_count != '0) |-> busy)
    else $error("[SVA seq] busy hole: queue non-empty but not busy");

  ap_busy_abort: assert property (@(posedge clk) disable iff (!rst_n)
    aborting |-> busy)
    else $error("[SVA seq] busy hole: abort draining but not busy");

  ap_qcount_range: assert property (@(posedge clk) disable iff (!rst_n)
    32'(q_count) <= QDEPTH)
    else $error("[SVA seq] q_count exceeds QDEPTH");

  ap_empty_mirror: assert property (@(posedge clk) disable iff (!rst_n)
    q_empty == (q_count == '0))
    else $error("[SVA seq] q_empty does not mirror q_count");

  ap_full_mirror: assert property (@(posedge clk) disable iff (!rst_n)
    q_full == (32'(q_count) == QDEPTH))
    else $error("[SVA seq] q_full does not mirror q_count");

  ap_abort_drained: assert property (@(posedge clk) disable iff (!rst_n)
    aborting |-> q_empty)
    else $error("[SVA seq] queued descriptors survived an abort");

endmodule

`endif // SEQ_SB_SVA_SVH
