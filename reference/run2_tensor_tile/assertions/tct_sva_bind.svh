// =============================================================================
// tct_sva_bind.svh
// Concurrent SVA properties for tensor_core_tile (INT8 GEMM tile).
// Owner: Coverage and SVA Agent.
//
// TARGET: Verilator 5.044. Verilator supports a SUBSET of SVA: simple
// clocked 'assert property' / 'cover property' with 'disable iff' and
// '|->' / '|=>' implication ARE supported. Concurrent assertions are only
// actually checked at runtime when the simulation is invoked with
// The tool's '--assert' flag (without it, 'assert property' statements
// are parsed/elaborated but not evaluated). This file does NOT use
// covergroups/coverpoints/bins (Verilator does not support SV functional
// covergroups) -- functional coverage is implemented separately in
// ./coverage/tct_coverage.svh via plain always-blocks + manual counters.
//
// INCLUDE / COMPILE CONTRACT:
//   This file is designed to be 'included INSIDE the tb_tensor_core_tile
//   module body, AFTER the 'dut' instance is elaborated (the harness's
//   existing 'ifdef TCT_SVA_BIND hook already does this -- see
//   tb/tb_tensor_core_tile.sv, guarded by '+define+TCT_SVA_BIND').
//   It relies on these EXACT identifiers already being in scope at the
//   'include point (all declared earlier in tb_tensor_core_tile.sv, do
//   NOT redeclare them here):
//     clk, rst_n, start, busy, done, overflow,
//     w_load_valid, w_load_ready,
//     a_in_valid, a_in_ready,
//     y_out_valid, y_out_ready, y_out_last, y_out_data
//   and the DUT instance handle 'dut' (tensor_core_tile), reached via
//   hierarchical references:
//     dut.compute_active                  (top-level wire, tensor_core_tile.v)
//     dut.u_ctrl_fsm.state                (ctrl_fsm internal state reg, 2'd0..2'd3)
//     dut.u_ctrl_fsm.acc_clr              (ctrl_fsm output port)
//     dut.u_ctrl_fsm.acc_en               (ctrl_fsm output port)
//     dut.u_ctrl_fsm.load_full            (ctrl_fsm internal wire)
//   These are plain compile-time SystemVerilog hierarchical references
//   (not VPI), which Verilator resolves at elaboration without needing
//   '--public' annotations on the RTL.
//
// REQUIRED VERILATOR COMPILE FLAGS (in addition to the harness's existing
// command line) for this file to be found and for assertions to actually
// be evaluated at runtime:
//     -Iassertions           <-- so 'include "tct_sva_bind.svh" resolves
//     +define+TCT_SVA_BIND   <-- enables the harness's include hook
//     --assert                <-- REQUIRED for Verilator to evaluate
//                                  'assert property' / 'cover property'
//                                  at runtime (without it they are parsed
//                                  but not checked -- see SELF-VERIFY notes
//                                  in the accompanying report).
//
// PASS EVIDENCE CONVENTION: every 'assert property' below, on failure,
// prints a $error via Verilator's native assertion-failure reporting
// (or the explicit else-$error clause) and increments Verilator's
// internal assertion-failure count (visible as nonzero exit status /
// "%Error" lines in simulation stdout). There is no separate SVA
// pass-counter in this file (Verilator does not expose a per-assertion
// pass tally through this construct) -- "0 assertion failures printed
// during a simulation run that reached TB SUMMARY" is the pass evidence
// for this file's properties. See sva_report() below.
//
// STATE ENCODING (from rtl/ctrl_fsm.v, REQ-STATE-001):
//   ST_IDLE=2'd0, ST_LOAD_WEIGHTS=2'd1, ST_COMPUTE=2'd2, ST_DRAIN=2'd3
// =============================================================================

// -----------------------------------------------------------------------
// A_WREADY_PHASE -- REQ-GEMM-012
// Weight-load ready may only be asserted while the FSM is in LOAD_WEIGHTS.
// -----------------------------------------------------------------------
property p_wready_phase;
  @(posedge clk) disable iff (!rst_n)
    w_load_ready |-> (dut.u_ctrl_fsm.state == 2'd1);
endproperty
A_WREADY_PHASE: assert property (p_wready_phase)
  else $error("A_WREADY_PHASE (REQ-GEMM-012): w_load_ready asserted outside LOAD_WEIGHTS, state=%0d", dut.u_ctrl_fsm.state);

// -----------------------------------------------------------------------
// A_AREADY_PHASE -- REQ-GEMM-013
// Activation-in ready may only be asserted during COMPUTE.
// -----------------------------------------------------------------------
property p_aready_phase;
  @(posedge clk) disable iff (!rst_n)
    a_in_ready |-> (dut.u_ctrl_fsm.state == 2'd2);
endproperty
A_AREADY_PHASE: assert property (p_aready_phase)
  else $error("A_AREADY_PHASE (REQ-GEMM-013): a_in_ready asserted outside COMPUTE, state=%0d", dut.u_ctrl_fsm.state);

// -----------------------------------------------------------------------
// A_YVALID_PHASE -- REQ-GEMM-014
// Result-output valid may only be asserted during DRAIN.
// -----------------------------------------------------------------------
property p_yvalid_phase;
  @(posedge clk) disable iff (!rst_n)
    y_out_valid |-> (dut.u_ctrl_fsm.state == 2'd3);
endproperty
A_YVALID_PHASE: assert property (p_yvalid_phase)
  else $error("A_YVALID_PHASE (REQ-GEMM-014): y_out_valid asserted outside DRAIN, state=%0d", dut.u_ctrl_fsm.state);

// -----------------------------------------------------------------------
// A_ACC_MUTEX -- spec pinned decision (accumulator clear/enable mutual
// exclusion; ctrl_fsm.v: acc_clr = IDLE||LOAD_WEIGHTS, acc_en = a_fifo_rd_en
// which only fires in COMPUTE -- these must never overlap on the same edge).
// -----------------------------------------------------------------------
property p_acc_mutex;
  @(posedge clk) disable iff (!rst_n)
    !(dut.u_ctrl_fsm.acc_clr && dut.u_ctrl_fsm.acc_en);
endproperty
A_ACC_MUTEX: assert property (p_acc_mutex)
  else $error("A_ACC_MUTEX: acc_clr and acc_en both asserted simultaneously");

// -----------------------------------------------------------------------
// A_YDATA_STABLE -- REQ-GEMM-014 / REQ-PROT-001
// Under backpressure (valid high, ready low), valid must stay high and
// data must not change on the following cycle (valid-hold rule).
// -----------------------------------------------------------------------
property p_ydata_stable;
  @(posedge clk) disable iff (!rst_n)
    ($past(dut.u_ctrl_fsm.drain_active) && dut.u_ctrl_fsm.drain_active &&
     $stable(dut.u_ctrl_fsm.drain_row))
      |-> (y_out_valid && $stable(y_out_data));
endproperty
A_YDATA_STABLE: assert property (p_ydata_stable)
  else $error("A_YDATA_STABLE (REQ-GEMM-014/REQ-PROT-001): y_out_valid dropped or y_out_data changed under backpressure");

// -----------------------------------------------------------------------
// A_YLAST_ONLY_DRAIN -- REQ-PROT-003
// y_out_last must never be asserted without y_out_valid.
// -----------------------------------------------------------------------
property p_ylast_only_drain;
  @(posedge clk) disable iff (!rst_n)
    y_out_last |-> y_out_valid;
endproperty
A_YLAST_ONLY_DRAIN: assert property (p_ylast_only_drain)
  else $error("A_YLAST_ONLY_DRAIN (REQ-PROT-003): y_out_last asserted without y_out_valid");

// -----------------------------------------------------------------------
// A_DONE_PULSE -- REQ-GEMM-011
// done may only pulse on a cycle where the final drain beat is accepted.
// -----------------------------------------------------------------------
property p_done_pulse;
  @(posedge clk) disable iff (!rst_n)
    done |-> (y_out_valid && y_out_ready && y_out_last);
endproperty
A_DONE_PULSE: assert property (p_done_pulse)
  else $error("A_DONE_PULSE (REQ-GEMM-011): done asserted without accepted final drain beat");

// -----------------------------------------------------------------------
// A_BUSY_IDLE -- REQ-GEMM-010
// busy must exactly track (state != IDLE).
// -----------------------------------------------------------------------
property p_busy_idle;
  @(posedge clk) disable iff (!rst_n)
    busy == (dut.u_ctrl_fsm.state != 2'd0);
endproperty
A_BUSY_IDLE: assert property (p_busy_idle)
  else $error("A_BUSY_IDLE (REQ-GEMM-010): busy=%0b does not match (state!=IDLE), state=%0d", busy, dut.u_ctrl_fsm.state);

// -----------------------------------------------------------------------
// A_START_ONLY_IDLE -- REQ-GEMM-009 / ERR-003
// The FSM may only leave IDLE on a cycle where start was asserted; if the
// FSM is in IDLE and start is low, it must still be in IDLE next cycle.
// -----------------------------------------------------------------------
property p_start_only_idle;
  @(posedge clk) disable iff (!rst_n)
    (dut.u_ctrl_fsm.state == 2'd0 && !start) |=> (dut.u_ctrl_fsm.state == 2'd0 || start);
endproperty
A_START_ONLY_IDLE: assert property (p_start_only_idle)
  else $error("A_START_ONLY_IDLE (REQ-GEMM-009/ERR-003): FSM left IDLE without start authorizing the transition");

// -----------------------------------------------------------------------
// A_WREADY_NOT_WHEN_FULL -- REQ-GEMM-012
// Once the weight buffer has accepted N*N=64 beats (load_full), the
// tile must stop asserting w_load_ready.
// -----------------------------------------------------------------------
property p_wready_not_full;
  @(posedge clk) disable iff (!rst_n)
    dut.u_ctrl_fsm.load_full |-> !w_load_ready;
endproperty
A_WREADY_NOT_WHEN_FULL: assert property (p_wready_not_full)
  else $error("A_WREADY_NOT_WHEN_FULL (REQ-GEMM-012): w_load_ready asserted while load_full");

// -----------------------------------------------------------------------
// A_NO_AREADY_OUT_OF_COMPUTE -- REQ-GEMM-013 / RESOLVED-008
// a_in_ready must be low whenever compute is not active.
// -----------------------------------------------------------------------
property p_no_aready_out_of_compute;
  @(posedge clk) disable iff (!rst_n)
    !dut.compute_active |-> !a_in_ready;
endproperty
A_NO_AREADY_OUT_OF_COMPUTE: assert property (p_no_aready_out_of_compute)
  else $error("A_NO_AREADY_OUT_OF_COMPUTE (REQ-GEMM-013/RESOLVED-008): a_in_ready asserted while !compute_active");

// =============================================================================
// COVER PROPERTIES (existence coverage for key protocol events; these
// complement, but do not replace, the manual functional coverage buckets
// in coverage/tct_coverage.svh).
// =============================================================================

// C_WLOAD_FULL -- weight load completes (64th beat accepted, wbuf_loaded
// flag raised). REQ-GEMM-012 / REQ-STATE-002.
property p_cov_wload_full;
  @(posedge clk) disable iff (!rst_n)
    dut.u_ctrl_fsm.wbuf_loaded;
endproperty
C_WLOAD_FULL: cover property (p_cov_wload_full);

// C_DRAIN_LAST -- a final drain beat (y_out_last) is actually accepted.
// REQ-GEMM-011 / REQ-PROT-003.
property p_cov_drain_last_accepted;
  @(posedge clk) disable iff (!rst_n)
    (y_out_valid && y_out_ready && y_out_last);
endproperty
C_DRAIN_LAST: cover property (p_cov_drain_last_accepted);

// C_BACKPRESSURE -- a backpressure event on the result-output stream is
// observed (y_out_valid high while y_out_ready low). REQ-GEMM-015.
property p_cov_backpressure;
  @(posedge clk) disable iff (!rst_n)
    (y_out_valid && !y_out_ready);
endproperty
C_BACKPRESSURE: cover property (p_cov_backpressure);

// =============================================================================
// sva_report() -- documentation-only summary task. Verilator does not
// expose a per-assertion pass tally for 'assert property' through this
// mechanism; every FAILURE self-reports via $error above (and, when run
// with --assert, contributes to Verilator's nonzero exit status). Call
// this once at end-of-test alongside the scoreboard summary so the
// "0 assertion failures" pass criterion is stated explicitly in the log.
// =============================================================================
task automatic sva_report();
  begin
    $display("---- SVA REPORT (tct_sva_bind.svh) ----");
    $display("11 assert property checks + 3 cover property checks active.");
    $display("Pass criterion: ZERO '%%Error' assertion-failure lines above this point in the log,");
    $display("AND simulation was compiled/run with Verilator --assert (see file header).");
  end
endtask
