// =============================================================================
// tb_tensor_core_tile.sv
// STATIC STRUCTURAL TESTBENCH HARNESS for tensor_core_tile (INT8 systolic GEMM
// tile). Owner: TB Harness and Interface Agent.
// Target simulator: Verilator version 5.044. Build/elaborate this harness
// with the following command line (see accompanying report for the actual
// passing transcript):
//   --binary --timing -Wno-fatal -Irtl -Itb tb/tb_tensor_core_tile.sv
//   rtl/tensor_core_tile.v rtl/ctrl_fsm.v rtl/weight_buffer.v
//   rtl/act_fifo.v rtl/pe_array.v rtl/pe.v rtl/requant_unit.v
//   --top-module tb_tensor_core_tile
// NOT UVM. No run_test(). No UVM classes/packages. No scoreboard compare
// logic and no stimulus/sequence logic live in THIS file -- see the
// `include contract below.
//
// =============================== SHARED CONTRACT ============================
// This file declares ALL DUT-facing signals as plain module-level logic and
// wires them directly to the DUT instance `dut`. Downstream agents
// (stimulus, scoreboard, SVA/bind) drive/observe these SAME signals -- there
// are no SystemVerilog interfaces, no clocking blocks, no virtual interface
// handles in this contract. Plain @(posedge clk) sequential tasks are the
// only synchronization primitive used anywhere (chosen for Verilator 5.044
// --timing compatibility; some tools do not support clocking blocks used
// through a modport, so this harness avoids that construct entirely).
//
// SIGNAL CONTRACT (declared in this module, names/widths copied verbatim
// from the DUT port list -- DO NOT RENAME, DO NOT CHANGE WIDTH):
//   logic                clk, rst_n;
//   logic                start;
//   logic [1:0]          cfg_mode;
//   logic                busy, done, overflow;
//   logic                w_load_valid, w_load_ready, w_load_last;
//   logic [7:0]          w_load_data;
//   logic                a_in_valid, a_in_ready, a_in_last;
//   logic [N*8-1:0]      a_in_data;
//   logic                y_out_valid, y_out_ready, y_out_last;
//   logic [N*DW_OUT-1:0] y_out_data;
//   logic [SCALE_W-1:0]  req_scale;
//   logic [SHIFT_W-1:0]  req_shift;
// Localparams N=8, ACC_W=32, DW_OUT=8, SCALE_W=16, SHIFT_W=5, AFIFO_DEPTH=16
// are declared once in this file and are visible (via `include textual
// substitution) to tct_scoreboard.svh and tct_stimulus.svh -- do not
// redeclare them there.
//
// RESET / CLOCK CONTRACT (owned + implemented HERE only -- do not
// re-implement clock generation or the top-level reset pulse elsewhere):
//   clk:  initial clk = 1'b0; always #5 clk = ~clk;   (10-time-unit period)
//   rst_n: SYNCHRONOUS active-low, sampled on posedge clk (per DUT contract).
//   task automatic tb_apply_reset(int unsigned cycles = 4);
//       Drives rst_n=0 for `cycles` posedge clk edges, drives every DUT
//       input to a benign idle value (0) so nothing is dangling/X at the
//       first sample, then deasserts rst_n and waits one more posedge
//       before returning. Called EXACTLY ONCE from the main initial block,
//       before the stimulus entry task runs. Stimulus MAY call it again
//       mid-test if a directed reset scenario is needed -- it is safe/
//       idempotent and is the ONLY reset-sequencing code in the whole TB.
//
// TASK API PROTOTYPES -- bodies live in ./tb/tct_stimulus.svh, owned by the
// Stimulus agent. This harness only `includes that file and calls the
// single top-level entry point from its main initial block:
//   task automatic stim_run_all();
//       Top-level stimulus entry point (REQUIRED name/signature). Runs all
//       directed + constrained-random sequences, drives the shared DUT
//       signals listed above directly (no wrapper handles/interfaces),
//       calls the scoreboard hooks (sb_* below) for every checked event,
//       and thereby updates pass_count/fail_count (declared in the
//       scoreboard include). Must RETURN (not $finish/$fatal) when done so
//       this harness can print the final summary and exit itself.
//   Suggested (not enforced by this harness) lower-level driver tasks the
//   stimulus agent may define for its own internal use, operating on the
//   shared signals above via @(posedge clk):
//       task automatic drv_pulse_start();
//       task automatic drv_load_weight_beat(input logic [7:0] data, input logic last);
//       task automatic drv_send_activation_beat(input logic [N*8-1:0] data, input logic last);
//       task automatic drv_wait_output_beat(output logic [N*DW_OUT-1:0] data, output logic last);
//       task automatic drv_set_requant(input logic [SCALE_W-1:0] scale, input logic [SHIFT_W-1:0] shift);
//
// SCOREBOARD HOOKS -- bodies live in ./tb/tct_scoreboard.svh, owned by the
// Scoreboard agent. This harness only `includes that file:
//   integer pass_count;   // SINGLE POINT OF TRUTH -- declared ONLY in
//   integer fail_count;   // tct_scoreboard.svh. Do NOT redeclare in this
//                         // file or in tct_stimulus.svh.
//   Golden/reference-model storage (weight shadow memory, activation queue
//   shadow, accumulator shadow, requant golden function, etc.) is owned and
//   declared entirely inside tct_scoreboard.svh -- this harness makes no
//   assumption about its internal shape.
//   Suggested (not enforced by this harness) checker entry points the
//   scoreboard agent may define, to be called by stimulus as beats occur:
//       task automatic sb_on_reset();
//       task automatic sb_on_weight_beat(input logic [7:0] data, input logic last);
//       task automatic sb_on_activation_beat(input logic [N*8-1:0] data, input logic last);
//       task automatic sb_on_output_beat(input logic [N*DW_OUT-1:0] data, input logic last,
//                                        input logic [SCALE_W-1:0] scale, input logic [SHIFT_W-1:0] shift);
//
// INCLUDE ORDER (fixed -- do not reorder): tct_scoreboard.svh is `included
// BEFORE tct_stimulus.svh (stimulus tasks may reference scoreboard hooks/
// vars; the scoreboard must never reference stimulus tasks/vars).
//
// TIMEOUT WATCHDOG: after TB_TIMEOUT_CYCLES posedges of clk without the
// main initial block having finished, this harness prints "TIMEOUT" and
// exits with a nonzero status via $fatal. This is the ONLY self-driven
// pass/fail-adjacent logic in this file; it does not inspect DUT behavior.
//
// WAVEFORM DUMP: guarded behind `+define+TCT_DUMP` at compile time; never
// left on unconditionally (VCDs are throwaway run artifacts).
//
// SVA BIND HOOK: `include "tct_sva_bind.svh" is wired below, guarded behind
// `ifdef TCT_SVA_BIND (compile with +define+TCT_SVA_BIND once the SVA agent
// delivers that file). It is included after the `dut` instance so bind
// targets resolve against it. This harness does not itself declare any
// assertions.
//
// DEPENDENCY NOTE: ./tb/tct_scoreboard.svh and ./tb/tct_stimulus.svh are
// currently PLACEHOLDER STUBS written by this agent SOLELY so this harness
// can be proven to elaborate/run standalone (see SELF-VERIFY evidence in
// the accompanying report). They contain no reference model and no
// stimulus and MUST be replaced by the Scoreboard and Stimulus agents
// respectively; the prototypes/behavior documented above are the contract
// they must honor.
// =============================================================================

`timescale 1ns/1ps

module tb_tensor_core_tile;

  // ---------------------------------------------------------------------
  // Parameters mirroring DUT defaults (must match rtl/tensor_core_tile.v)
  // ---------------------------------------------------------------------
  localparam int N           = 8;
  localparam int ACC_W       = 32;
  localparam int DW_OUT      = 8;
  localparam int SCALE_W     = 16;
  localparam int SHIFT_W     = 5;
  localparam int AFIFO_DEPTH = 16;

  // ---------------------------------------------------------------------
  // Shared DUT-facing signals -- see header "SIGNAL CONTRACT". These are
  // the ONLY handles stimulus/scoreboard/SVA agents should drive/sample.
  // ---------------------------------------------------------------------
  logic                    clk;
  logic                    rst_n;

  logic                    start;
  logic [1:0]              cfg_mode;
  logic                    busy;
  logic                    done;
  logic                    overflow;

  logic                    w_load_valid;
  logic                    w_load_ready;
  logic [7:0]              w_load_data;
  logic                    w_load_last;

  logic                    a_in_valid;
  logic                    a_in_ready;
  logic [N*8-1:0]          a_in_data;
  logic                    a_in_last;

  logic                    y_out_valid;
  logic                    y_out_ready;
  logic [N*DW_OUT-1:0]     y_out_data;
  logic                    y_out_last;

  logic [SCALE_W-1:0]      req_scale;
  logic [SHIFT_W-1:0]      req_shift;

  // ---------------------------------------------------------------------
  // Clock generation -- 10-time-unit period, owned exclusively here.
  // ---------------------------------------------------------------------
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // ---------------------------------------------------------------------
  // DUT instantiation -- exact port names/widths per rtl/tensor_core_tile.v
  // ---------------------------------------------------------------------
  tensor_core_tile #(
      .N           (N),
      .ACC_W       (ACC_W),
      .DW_OUT      (DW_OUT),
      .SCALE_W     (SCALE_W),
      .SHIFT_W     (SHIFT_W),
      .AFIFO_DEPTH (AFIFO_DEPTH)
  ) dut (
      .clk          (clk),
      .rst_n        (rst_n),

      .start        (start),
      .cfg_mode     (cfg_mode),
      .busy         (busy),
      .done         (done),
      .overflow     (overflow),

      .w_load_valid (w_load_valid),
      .w_load_ready (w_load_ready),
      .w_load_data  (w_load_data),
      .w_load_last  (w_load_last),

      .a_in_valid   (a_in_valid),
      .a_in_ready   (a_in_ready),
      .a_in_data    (a_in_data),
      .a_in_last    (a_in_last),

      .y_out_valid  (y_out_valid),
      .y_out_ready  (y_out_ready),
      .y_out_data   (y_out_data),
      .y_out_last   (y_out_last),

      .req_scale    (req_scale),
      .req_shift    (req_shift)
  );

  // ---------------------------------------------------------------------
  // Reset sequencing task -- owned by this harness (see header). This is
  // the ONLY place reset is sequenced; benign idle values (0) are driven
  // onto every DUT input so nothing is dangling/X before stimulus starts.
  // ---------------------------------------------------------------------
  task automatic tb_apply_reset(input int unsigned cycles = 4);
    int unsigned i;
    begin
      rst_n        = 1'b0;
      start        = 1'b0;
      cfg_mode     = 2'b00;
      w_load_valid = 1'b0;
      w_load_data  = 8'h00;
      w_load_last  = 1'b0;
      a_in_valid   = 1'b0;
      a_in_data    = '0;
      a_in_last    = 1'b0;
      y_out_ready  = 1'b0;
      req_scale    = '0;
      req_shift    = '0;
      for (i = 0; i < cycles; i = i + 1) begin
        @(posedge clk);
      end
      rst_n = 1'b1;
      @(posedge clk);
    end
  endtask

  // ---------------------------------------------------------------------
  // Timeout watchdog -- structural safety net only, does not inspect DUT
  // functional behavior.
  // ---------------------------------------------------------------------
  localparam int unsigned TB_TIMEOUT_CYCLES = 200000;
  bit tb_done_flag = 1'b0;

  initial begin
    int unsigned n;
    n = 0;
    while (!tb_done_flag && n < TB_TIMEOUT_CYCLES) begin
      @(posedge clk);
      n = n + 1;
    end
    if (!tb_done_flag) begin
      $display("TIMEOUT");
      $display("TB SUMMARY: PASS=%0d FAIL=%0d", pass_count, fail_count);
      $display("TEST FAILED");
      $fatal(1, "tb_tensor_core_tile: global timeout after %0d cycles", TB_TIMEOUT_CYCLES);
    end
  end

  // ---------------------------------------------------------------------
  // Optional waveform dump -- throwaway artifact, opt-in only via
  // +define+TCT_DUMP. Never left on unconditionally.
  // ---------------------------------------------------------------------
`ifdef TCT_DUMP
  initial begin
    $dumpfile("tb_tensor_core_tile.vcd");
    $dumpvars(0, tb_tensor_core_tile);
  end
`endif

  // ---------------------------------------------------------------------
  // Scoreboard include: owns pass_count/fail_count + golden model.
  // See header "SCOREBOARD HOOKS". Must precede the stimulus include.
  // PLACEHOLDER STUB currently in tree -- see DEPENDENCY NOTE in header.
  // ---------------------------------------------------------------------
  `include "tct_scoreboard.svh"

  // ---------------------------------------------------------------------
  // Stimulus include: owns stim_run_all() + driver tasks.
  // See header "TASK API PROTOTYPES".
  // PLACEHOLDER STUB currently in tree -- see DEPENDENCY NOTE in header.
  // ---------------------------------------------------------------------
  `include "tct_stimulus.svh"

  // ---------------------------------------------------------------------
  // Optional SVA bind hook -- populated once the SVA agent delivers
  // tct_sva_bind.svh. Compile with +define+TCT_SVA_BIND to enable.
  // ---------------------------------------------------------------------
`ifdef TCT_SVA_BIND
  `include "tct_sva_bind.svh"
`endif

  // ---------------------------------------------------------------------
  // Optional manual functional-coverage bind hook -- populated by the
  // Coverage/SVA agent (coverage/tct_coverage.svh). Verilator has no SV
  // covergroup support, so this file implements coverage as plain
  // always-block bucket counters + a cov_report()/cov_holes() API.
  // Compile with +define+TCT_COVERAGE (and -Icoverage) to enable.
  // ---------------------------------------------------------------------
`ifdef TCT_COVERAGE
  `include "tct_coverage.svh"
`endif

  // ---------------------------------------------------------------------
  // Main sequencing block: reset -> stimulus -> summary -> exit.
  // This is the only place run-control/exit decisions are made.
  // ---------------------------------------------------------------------
  initial begin
    tb_apply_reset();
    stim_run_all();
    tb_done_flag = 1'b1;
`ifdef TCT_SVA_BIND
    sva_report();
`endif
`ifdef TCT_COVERAGE
    cov_report();
`endif
    $display("TB SUMMARY: PASS=%0d FAIL=%0d", pass_count, fail_count);
    if (fail_count == 0 && pass_count > 0) begin
      $display("TEST PASSED");
      $finish;
    end else begin
      $display("TEST FAILED");
      $fatal(1, "tb_tensor_core_tile: PASS=%0d FAIL=%0d", pass_count, fail_count);
    end
  end

endmodule
