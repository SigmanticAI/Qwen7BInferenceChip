// =============================================================================
// File   : gemm_bfm.svh
// -----------------------------------------------------------------------------
// PURPOSE
//   Reusable, plain-SystemVerilog (NO UVM, NO clocking-block/modport) BFM
//   tasks for the gemm_systolic_accel DUT. This file is `include`-d directly
//   INSIDE the gemm_tb_top module body, so every task below references the
//   DUT-facing signals declared in gemm_tb_top by their EXACT port names
//   (clk, rst_n, desc_valid, desc_ready, desc_data, w_valid, w_ready, w_data,
//   w_last, a_valid, a_ready, a_data, a_last, y_valid, y_ready, y_data,
//   y_last, busy, done, err_overflow) plus the TB bookkeeping ints
//   (error_count, check_count) and the N localparam (sized from the TB_N
//   compile-time define), all of which MUST be in scope at the `include
//   site.
//
//   This file contains STRUCTURAL BFM PLUMBING ONLY:
//     - drive/sample valid/ready handshakes exactly per REQ-PROTO-002/003
//     - optional randomized driver-side / receiver-side stall injection
//     - NO expected-value computation, NO scoreboard compare, NO covergroups
//   Other specialists (stimulus_sequence, scoreboard_checker, coverage_sva)
//   call these tasks; they do not need to touch DUT signals directly.
//
// TASK CONTRACT (signatures other specialists depend on -- do not change)
//   task apply_reset(int cycles);
//     Drives rst_n=0 for `cycles` clock edges (default caller should pass
//     >=2), deasserts all driven-side valids/inputs to idle/benign values
//     during reset, then releases rst_n=1 synchronized to posedge clk.
//     Returns after rst_n has been high for one full posedge (DUT sees a
//     clean synchronous reset per REQ-GEMM-010).
//
//   task send_descriptor(input logic [127:0] desc);
//     Drives desc_valid=1 / desc_data=desc and HOLDS both stable
//     (REQ-PROTO-002) until desc_ready&&desc_valid observed on a posedge,
//     then deasserts desc_valid on the following edge.
//
//   task send_weight_beat(input logic [N*8-1:0] data, input logic last,
//                          input int max_stall);
//     Drives w_valid=1/w_data=data/w_last=last and HOLDS them stable
//     (no valid-drop before handshake, REQ-PROTO-002) while w_ready is low.
//     Before asserting valid, optionally waits 0..max_stall idle cycles
//     (driver-side stall / gap between beats -- NOT a violation of
//     REQ-PROTO-002 because valid is not yet asserted during that gap).
//     Returns the cycle after the handshake completes, with w_valid low.
//
//   task send_act_beat(input logic [N*8-1:0] data, input logic last,
//                       input int max_stall);
//     Same shape/contract as send_weight_beat for the activation channel.
//
//   task recv_result_beat(output logic [N*8-1:0] data, output logic last,
//                          input int max_ystall);
//     Consumer side: optionally deasserts y_ready for 0..max_ystall cycles
//     (legal backpressure per REQ-PROTO-003 -- ready may move independently
//     of valid), then asserts y_ready=1 and waits for y_valid, samples
//     y_data/y_last into the output args on the accepting edge, then
//     deasserts y_ready.
//
//   function [127:0] make_descriptor(...)  -- see gemm_tb_pkg::make_descriptor.
//     (Re-exported here as a thin wrapper for convenience; identical
//     semantics, no duplicated field logic.)
//
//   task check(input bit cond, input string msg);
//     Bookkeeping only: increments check_count always, increments
//     error_count and prints "CHECK FAIL: <msg>" when cond==0.
// =============================================================================

// ---------------------------------------------------------------------------
// apply_reset : synchronous active-low reset sequencing (REQ-GEMM-010)
// ---------------------------------------------------------------------------
task automatic apply_reset(int cycles);
  begin
    rst_n      = 1'b0;
    desc_valid = 1'b0;
    desc_data  = '0;
    w_valid    = 1'b0;
    w_data     = '0;
    w_last     = 1'b0;
    a_valid    = 1'b0;
    a_data     = '0;
    a_last     = 1'b0;
    y_ready    = 1'b0;
    repeat (cycles) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
  end
endtask

// ---------------------------------------------------------------------------
// send_descriptor : descriptor channel, no skid, direct handshake with ctrl
// ---------------------------------------------------------------------------
// NOTE ON SAMPLING ORDER (root-caused during hang triage): gemm_ctrl's
// desc_ready is a purely combinational function of its registered FSM
// `state` (desc_ready=1 only while state==IDLE), wired straight through
// with NO skid buffer (by design, see gemm_ctrl.sv/gemm_systolic_accel.sv
// header notes). This means desc_ready DROPS on the very same clock edge
// that accepts the descriptor (state IDLE->DECODE), unlike w_ready/a_ready/
// y_valid which are registered through a stream_skid stage and therefore
// stay observably stable across that edge. A "wait-for-edge-THEN-check"
// idiom (`do @(posedge clk); while(!(valid&&ready))`) races against that
// same-edge combinational drop: whichever simulator scheduling order
// resolves first, the task can end up perpetually observing the
// already-dropped (post-transition) value of desc_ready and never detect
// the acceptance edge -- an infinite wait. The race-free fix is to
// check-BEFORE-the-edge instead: desc_ready is stable (settled from the
// previous edge) at any point before the next posedge, so sampling it
// there and only then advancing across the accepting edge is safe. (This
// is not needed for send_weight_beat/send_act_beat/recv_result_beat,
// whose ready/valid signals are registered by stream_skid, so no such
// same-edge combinational-drop race exists there.)
task automatic send_descriptor(input logic [127:0] desc);
  begin
    desc_data  = desc;
    desc_valid = 1'b1;
    while (!desc_ready) @(posedge clk);
    @(posedge clk); // the edge on which desc_valid&&desc_ready is accepted
    desc_valid = 1'b0;
  end
endtask

// ---------------------------------------------------------------------------
// send_weight_beat : weight streaming channel (REQ-PROTO-002/003)
// ---------------------------------------------------------------------------
task automatic send_weight_beat(input logic [N*8-1:0] data,
                                 input logic               last,
                                 input int                 max_stall);
  int stall_n;
  begin
    if (max_stall > 0) begin
      stall_n = $urandom_range(max_stall, 0);
      repeat (stall_n) @(posedge clk);
    end
    w_data  = data;
    w_last  = last;
    w_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!(w_valid && w_ready));
    w_valid = 1'b0;
  end
endtask

// ---------------------------------------------------------------------------
// send_act_beat : activation streaming channel (same contract as weight)
// ---------------------------------------------------------------------------
task automatic send_act_beat(input logic [N*8-1:0] data,
                              input logic               last,
                              input int                 max_stall);
  int stall_n;
  begin
    if (max_stall > 0) begin
      stall_n = $urandom_range(max_stall, 0);
      repeat (stall_n) @(posedge clk);
    end
    a_data  = data;
    a_last  = last;
    a_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!(a_valid && a_ready));
    a_valid = 1'b0;
  end
endtask

// ---------------------------------------------------------------------------
// recv_result_beat : result streaming channel, consumer-side backpressure
// ---------------------------------------------------------------------------
task automatic recv_result_beat(output logic [N*8-1:0] data,
                                 output logic               last,
                                 input  int                  max_ystall);
  int stall_n;
  begin
    y_ready = 1'b0;
    if (max_ystall > 0) begin
      stall_n = $urandom_range(max_ystall, 0);
      repeat (stall_n) @(posedge clk);
    end
    y_ready = 1'b1;
    do begin
      @(posedge clk);
    end while (!(y_valid && y_ready));
    data = y_data;
    last = y_last;
    y_ready = 1'b0;
  end
endtask

// ---------------------------------------------------------------------------
// make_descriptor : thin wrapper over gemm_tb_pkg::make_descriptor
// ---------------------------------------------------------------------------
function automatic logic [127:0] make_descriptor(
    input logic [7:0]  tiles_m,
    input logic [7:0]  tiles_k,
    input logic [7:0]  tiles_n,
    input logic [5:0]  req_shift,
    input logic [31:0] req_scale
);
  return gemm_tb_pkg::make_descriptor(tiles_m, tiles_k, tiles_n, req_shift, req_scale);
endfunction

// ---------------------------------------------------------------------------
// check : generic pass/fail bookkeeping (no scoreboard math lives here --
// downstream scoreboard_checker/stimulus_sequence call this with their own
// computed conditions).
// ---------------------------------------------------------------------------
task automatic check(input bit cond, input string msg);
  begin
    check_count = check_count + 1;
    if (!cond) begin
      error_count = error_count + 1;
      $display("CHECK FAIL: %s", msg);
    end
  end
endtask
