// =============================================================================
// File   : gemm_tb_top.sv
// Module : gemm_tb_top
// -----------------------------------------------------------------------------
// PURPOSE
//   STRUCTURAL, non-UVM testbench harness (shell) for gemm_systolic_accel.
//   Built for Verilator 5.044, plain 2-state clock-toggling harness (no
//   `--timing`, no clocking blocks/modports -- Verilator --binary compatible).
//
//   Owns ONLY:
//     - DUT instantiation (real port names, unmodified)
//     - clock generation + synchronous active-low reset sequencing
//     - reusable BFM tasks (see tb/gemm_bfm.svh, `include-d below) that other
//       specialists (stimulus_sequence, scoreboard_checker, coverage_sva)
//       call into
//     - global error_count/check_count bookkeeping + check() task
//     - a simulation timeout watchdog
//     - a trivial reset-smoke self-test proving the harness elaborates/runs
//       standalone right now
//
//   Does NOT own (left for other specialist agents):
//     - transaction stimulus / directed sequences beyond the reset smoke
//     - scoreboard / reference-model compare logic
//     - covergroups / SVA (bind hooks are provided, no assertions authored
//       here)
//     - any UVM class/env/test/run_test()
//
// SIZING
//   TB_N (default 8) controls BOTH the DUT's N parameter and all BFM vector
//   widths, so "make N=16" (see sim/Makefile) produces a consistent N=16
//   build without editing this file. Override via a +define+TB_N=16 passed
//   on the command line.
// =============================================================================

`ifndef TB_N
`define TB_N 8
`endif

// NOTE: gemm_tb_pkg.sv is compiled as its own top-level source file (see
// sim/Makefile file list) -- packages cannot be `include-d from inside a
// module body. This module merely imports it below.
module gemm_tb_top;

  import gemm_tb_pkg::*;

  // ---------------------------------------------------------------------
  // Sizing
  // ---------------------------------------------------------------------
  localparam int N       = `TB_N;
  localparam int DATA_W  = 8;
  localparam int ACC_W   = 32;
  localparam int SCALE_W = 32;
  localparam int SHIFT_W = 6;
  localparam int WBUF_DEPTH = 64;
  localparam int ABUF_DEPTH = 64;
  localparam int DESC_W  = 128;
  localparam int PW      = N*DATA_W;

  // Simulation timeout watchdog (in clock cycles, applied below).
  localparam int TIMEOUT_CYCLES = 5000; // DEBUG-TEMP: lowered for fast iteration, MUST restore to 200000

  // ---------------------------------------------------------------------
  // DUT-facing signals (exact port names of gemm_systolic_accel)
  // ---------------------------------------------------------------------
  logic                  clk;
  logic                  rst_n;

  logic                  desc_valid;
  logic                  desc_ready;
  logic [DESC_W-1:0]     desc_data;

  logic                  w_valid;
  logic                  w_ready;
  logic [PW-1:0]         w_data;
  logic                  w_last;

  logic                  a_valid;
  logic                  a_ready;
  logic [PW-1:0]         a_data;
  logic                  a_last;

  logic                  y_valid;
  logic                  y_ready;
  logic [PW-1:0]         y_data;
  logic                  y_last;

  logic                  busy;
  logic                  done;
  logic                  err_overflow;

  // ---------------------------------------------------------------------
  // TB bookkeeping (downstream specialists increment/read these via the
  // check() task; no scoreboard "expected value" math lives here).
  // ---------------------------------------------------------------------
  int unsigned error_count = 0;
  int unsigned check_count = 0;

  // ---------------------------------------------------------------------
  // DUT instantiation -- real port names, unmodified.
  // ---------------------------------------------------------------------
  gemm_systolic_accel #(
      .N           (N),
      .DATA_W      (DATA_W),
      .ACC_W       (ACC_W),
      .SCALE_W     (SCALE_W),
      .SHIFT_W     (SHIFT_W),
      .WBUF_DEPTH  (WBUF_DEPTH),
      .ABUF_DEPTH  (ABUF_DEPTH),
      .DESC_W      (DESC_W)
  ) dut (
      .clk          (clk),
      .rst_n        (rst_n),

      .desc_valid   (desc_valid),
      .desc_ready   (desc_ready),
      .desc_data    (desc_data),

      .w_valid      (w_valid),
      .w_ready      (w_ready),
      .w_data       (w_data),
      .w_last       (w_last),

      .a_valid      (a_valid),
      .a_ready      (a_ready),
      .a_data       (a_data),
      .a_last       (a_last),

      .y_valid      (y_valid),
      .y_ready      (y_ready),
      .y_data       (y_data),
      .y_last       (y_last),

      .busy         (busy),
      .done         (done),
      .err_overflow (err_overflow)
  );

  // ---------------------------------------------------------------------
  // Clock generation: plain 2-state toggling, 10-unit period, no --timing
  // needed. Starts at 0.
  // ---------------------------------------------------------------------
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // ---------------------------------------------------------------------
  // Simulation timeout watchdog (cycle-based, not fixed #time, so it
  // scales with the clock generator above regardless of timescale).
  // ---------------------------------------------------------------------
  int unsigned cycle_count;
  initial cycle_count = 0;
  always @(posedge clk) begin
    cycle_count <= cycle_count + 1;
    if (cycle_count >= TIMEOUT_CYCLES) begin
      $display("TB_RESULT: FAIL (TIMEOUT after %0d cycles)", TIMEOUT_CYCLES);
      $fatal(1, "simulation timeout");
    end
  end

  // ---------------------------------------------------------------------
  // Reusable BFM tasks (apply_reset, send_descriptor, send_weight_beat,
  // send_act_beat, recv_result_beat, make_descriptor, check). `include-d
  // here (rather than a separate module) so tasks share this module's
  // DUT-facing signals by name, per Verilator plain-SV harness convention.
  // ---------------------------------------------------------------------
  `include "gemm_bfm.svh"

  // ---------------------------------------------------------------------
  // Directed + constrained-random test scenarios (stimulus_sequence
  // specialist deliverable). `include-d here for the same reason as
  // gemm_bfm.svh: tasks need this module's DUT-facing signals in scope.
  // Defines run_all_tests(), called from the initial block below.
  // ---------------------------------------------------------------------
  `include "gemm_tests.svh"

  // ---------------------------------------------------------------------
  // Minimal reset smoke test: proves the harness elaborates + runs
  // standalone right now. Applies reset, checks the DUT's reset-state
  // outputs per REQ-GEMM-010 (busy/done/y_valid deasserted after reset),
  // then prints the pass/fail summary and exits.
  //
  // Downstream stimulus_sequence / scoreboard_checker agents are expected
  // to replace or extend this initial block with real traffic; it is
  // intentionally trivial.
  // ---------------------------------------------------------------------
  initial begin
    $display("TB: gemm_tb_top starting, N=%0d, DATA_W=%0d", N, DATA_W);

    apply_reset(4);

    check(busy         == 1'b0, "busy should be 0 immediately after reset");
    check(y_valid       == 1'b0, "y_valid should be 0 immediately after reset");
    check(done           == 1'b0, "done should be 0 immediately after reset");
    check(err_overflow  == 1'b0, "err_overflow should be 0 immediately after reset");
    check(desc_ready     == 1'b1, "desc_ready should be 1 (IDLE) after reset");

    // Give a few idle cycles so any registered outputs settle before summary.
    repeat (4) @(posedge clk);

    // -------------------------------------------------------------------
    // Directed + constrained-random scenario suite (stimulus_sequence
    // specialist deliverable, tb/gemm_tests.svh). Accumulates into the
    // same error_count/check_count bookkeeping as the reset smoke test
    // above; per-test PASS/FAIL lines are printed by run_all_tests()
    // itself.
    // -------------------------------------------------------------------
    run_all_tests();

    if (error_count == 0) begin
      $display("TB_RESULT: PASS  (checks=%0d errors=%0d)", check_count, error_count);
      $finish;
    end else begin
      $display("TB_RESULT: FAIL  (checks=%0d errors=%0d)", check_count, error_count);
      $fatal(1, "reset smoke test failed");
    end
  end

endmodule : gemm_tb_top
