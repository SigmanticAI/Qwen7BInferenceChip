// =============================================================================
// File   : tb/gemm_sva_bind.sv
// Module : gemm_sva_checker (+ bind into gemm_systolic_accel)
// -----------------------------------------------------------------------------
// PURPOSE
//   Separate, non-invasive protocol-checker + functional-coverage module for
//   gemm_systolic_accel. Attached purely via `bind` -- rtl/*.sv is NOT edited.
//
//   Contains:
//     - concurrent SVA (assert property) for protocol/timing rules
//     - `cover property` functional coverage points (see COVERAGE NOTE below)
//
// COVERAGE NOTE (tool-support reality for this flow)
//   The Verilator 5.044 open-source simulator used for this flow does NOT
//   support SystemVerilog `covergroup` / `coverpoint` / `cross` constructs
//   (those require a full commercial simulator such as VCS/Questa/Xcelium).
//   Attempting to compile covergroups under this tool fails elaboration.
//   Its own coverage model is line/toggle coverage (via --coverage) plus
//   support for `cover property` statements inside concurrent assertion
//   syntax, which IS supported and DOES get counted in its coverage
//   database when run with --coverage (or simply compiled with --assert,
//   which still elaborates cover property as an executable check). This
//   file therefore implements every functional coverage goal as a
//   `cover property` statement, NOT a covergroup. No covergroup/coverpoint/
//   cross constructs appear anywhere in this file.
//
// CLOCK / RESET (exact signals used, taken from gemm_systolic_accel ports)
//   clk   : posedge-sampled clock, port name "clk"
//   rst_n : synchronous active-low reset, port name "rst_n"
//   All properties below are `@(posedge clk) disable iff (!rst_n)`.
//
// BIND TARGET
//   bind gemm_systolic_accel gemm_sva_checker u_sva (.*);
//   -- attaches to every instance of gemm_systolic_accel (e.g. the `dut`
//   instance in tb/gemm_tb_top.sv) with no RTL source edits.
//
// REQUIREMENT TRACEABILITY (see per-assertion / per-cover comments below)
//   REQ-PROTO-002, REQ-PROTO-004, REQ-GEMM-001, REQ-GEMM-008, REQ-GEMM-009,
//   REQ-GEMM-010, REQ-GEMM-012, REQ-TIME-003, ASM-006
// =============================================================================

module gemm_sva_checker #(
    parameter int N      = 8,
    parameter int DATA_W = 8,
    parameter int DESC_W = 128,
    localparam int PW    = N*DATA_W
) (
    input logic              clk,
    input logic              rst_n,

    input logic               desc_valid,
    input logic               desc_ready,
    input logic [DESC_W-1:0]  desc_data,

    input logic               w_valid,
    input logic               w_ready,
    input logic [PW-1:0]      w_data,
    input logic               w_last,

    input logic               a_valid,
    input logic               a_ready,
    input logic [PW-1:0]      a_data,
    input logic               a_last,

    input logic               y_valid,
    input logic               y_ready,
    input logic [PW-1:0]      y_data,
    input logic               y_last,

    input logic               busy,
    input logic               done,
    input logic               err_overflow
);

  // ===========================================================================
  // SECTION 1 : CONCURRENT ASSERTIONS (SVA)
  // ===========================================================================

  // ---------------------------------------------------------------------
  // REQ-PROTO-002 / REQ-GEMM-009 -- stability-until-accept on all 4 channels.
  // If valid && !ready this cycle, next cycle valid must still be high and
  // the payload must be stable, until the transfer actually completes.
  // ---------------------------------------------------------------------

  // desc channel
  a_desc_stable: assert property (
    @(posedge clk) disable iff (!rst_n)
    (desc_valid && !desc_ready) |=> (desc_valid && $stable(desc_data))
  ) else $error("REQ-PROTO-002/REQ-GEMM-009: desc_valid/desc_data not held stable across stall");

  // weight channel
  a_w_stable: assert property (
    @(posedge clk) disable iff (!rst_n)
    (w_valid && !w_ready) |=> (w_valid && $stable(w_data) && $stable(w_last))
  ) else $error("REQ-PROTO-002/REQ-GEMM-009: w_valid/w_data/w_last not held stable across stall");

  // activation channel
  a_a_stable: assert property (
    @(posedge clk) disable iff (!rst_n)
    (a_valid && !a_ready) |=> (a_valid && $stable(a_data) && $stable(a_last))
  ) else $error("REQ-PROTO-002/REQ-GEMM-009: a_valid/a_data/a_last not held stable across stall");

  // result (output) channel
  a_y_stable: assert property (
    @(posedge clk) disable iff (!rst_n)
    (y_valid && !y_ready) |=> (y_valid && $stable(y_data) && $stable(y_last))
  ) else $error("REQ-PROTO-002/REQ-GEMM-009: y_valid/y_data/y_last not held stable across stall");

  // ---------------------------------------------------------------------
  // REQ-PROTO-004 / REQ-GEMM-001 -- desc_ready must never be high while busy;
  // a descriptor may only be accepted while the core is idle.
  // ---------------------------------------------------------------------
  a_desc_ready_not_busy: assert property (
    @(posedge clk) disable iff (!rst_n)
    busy |-> !desc_ready
  ) else $error("REQ-PROTO-004/REQ-GEMM-001: desc_ready high while busy");

  a_desc_accept_only_idle: assert property (
    @(posedge clk) disable iff (!rst_n)
    (desc_valid && desc_ready) |-> !busy
  ) else $error("REQ-GEMM-001: descriptor accepted while busy (not idle)");

  // ---------------------------------------------------------------------
  // REQ-GEMM-012 / REQ-TIME-003 -- done is asserted iff it coincides with the
  // final-beat external handshake (y_valid && y_ready && y_last).
  // ---------------------------------------------------------------------
  a_done_implies_final_beat: assert property (
    @(posedge clk) disable iff (!rst_n)
    done |-> (y_valid && y_ready && y_last)
  ) else $error("REQ-GEMM-012/REQ-TIME-003: done asserted without a final-beat y handshake");

  // ---------------------------------------------------------------------
  // REQ-GEMM-012 -- err_overflow is sticky within a job: once set, it stays
  // set until a new descriptor is accepted (or reset, handled by disable iff).
  // ---------------------------------------------------------------------
  a_err_overflow_sticky: assert property (
    @(posedge clk) disable iff (!rst_n)
    (err_overflow && !(desc_valid && desc_ready)) |=> err_overflow
  ) else $error("REQ-GEMM-012: err_overflow dropped without a new descriptor accept");

  // ---------------------------------------------------------------------
  // busy semantics cross-checks.
  //   - done can only fire the cycle after busy was high (result phase of a
  //     real job, not a spurious pulse while idle).
  //   - y_valid may only be presented while the core is busy (or finishing
  //     the job on the same cycle that done pulses, which is itself gated
  //     by busy on the prior cycle via a_done_implies_final_beat above).
  // ---------------------------------------------------------------------
  a_done_after_busy: assert property (
    @(posedge clk) disable iff (!rst_n)
    done |-> $past(busy)
  ) else $error("REQ-GEMM-012: done pulsed without the core having been busy the prior cycle");

  a_yvalid_implies_busy: assert property (
    @(posedge clk) disable iff (!rst_n)
    y_valid |-> busy
  ) else $error("REQ-GEMM-012: y_valid asserted while core not busy");

  // ---------------------------------------------------------------------
  // REQ-GEMM-010 -- reset behavior: the cycle after rst_n is sampled low,
  // y_valid/busy/done/err_overflow must all read 0.
  // Implemented as an immediate assertion inside an always block (not gated
  // by disable iff, since it specifically checks the post-reset cycle).
  // ---------------------------------------------------------------------
  always @(posedge clk) begin
    if ($past(!rst_n)) begin
      a_reset_clears_status: assert (!y_valid && !busy && !done && !err_overflow)
        else $error("REQ-GEMM-010: y_valid/busy/done/err_overflow not deasserted the cycle after reset");
    end
  end

  // ---------------------------------------------------------------------
  // REQ-GEMM-008 / ASM-006 -- y_last must never assert without y_valid.
  // ---------------------------------------------------------------------
  a_ylast_implies_yvalid: assert property (
    @(posedge clk) disable iff (!rst_n)
    y_last |-> y_valid
  ) else $error("REQ-GEMM-008/ASM-006: y_last asserted without y_valid");

  // ---------------------------------------------------------------------
  // Optional -- no y_valid while in reset (rst_n low).
  // ---------------------------------------------------------------------
  a_no_yvalid_in_reset: assert property (
    @(posedge clk) (!rst_n) |-> !y_valid
  ) else $error("REQ-GEMM-010: y_valid asserted while rst_n low");

  // ===========================================================================
  // SECTION 2 : FUNCTIONAL COVERAGE (cover property -- Verilator-supported
  // subset; NOT covergroup/coverpoint, see COVERAGE NOTE above).
  // ===========================================================================

  // Completed job observed (final beat accepted -> done pulse), REQ-GEMM-012.
  cp_job_complete: cover property (
    @(posedge clk) disable iff (!rst_n)
    y_valid && y_ready && y_last
  );

  // Backpressure actually exercised on each of the 4 channels, REQ-GEMM-009.
  cp_desc_backpressure: cover property (
    @(posedge clk) disable iff (!rst_n)
    desc_valid && !desc_ready
  );

  cp_w_backpressure: cover property (
    @(posedge clk) disable iff (!rst_n)
    w_valid && !w_ready
  );

  cp_a_backpressure: cover property (
    @(posedge clk) disable iff (!rst_n)
    a_valid && !a_ready
  );

  cp_y_backpressure: cover property (
    @(posedge clk) disable iff (!rst_n)
    y_valid && !y_ready
  );

  // err_overflow observed set at least once, REQ-GEMM-012/ERR-001.
  cp_err_overflow_seen: cover property (
    @(posedge clk) disable iff (!rst_n)
    err_overflow
  );

  // Descriptor accepted, REQ-GEMM-001.
  cp_desc_accept: cover property (
    @(posedge clk) disable iff (!rst_n)
    desc_valid && desc_ready
  );

  // Multi-beat result observed: y_last handshake preceded by >=2 prior
  // y-beat handshakes within the same job window (tracked via a small
  // counter reset on job completion), REQ-GEMM-008.
  int unsigned y_beat_cnt;
  always @(posedge clk) begin
    if (!rst_n) begin
      y_beat_cnt <= '0;
    end else if (y_valid && y_ready) begin
      if (y_last) y_beat_cnt <= '0;
      else        y_beat_cnt <= y_beat_cnt + 1'b1;
    end
  end

  cp_multi_beat_result: cover property (
    @(posedge clk) disable iff (!rst_n)
    (y_valid && y_ready && y_last && (y_beat_cnt >= 2))
  );

endmodule : gemm_sva_checker

// =============================================================================
// bind: attach gemm_sva_checker to every instance of gemm_systolic_accel.
// Port-name binding (.*) relies on identical port names between the checker
// and the DUT for the signals both declare; N/DATA_W/DESC_W parameters are
// re-derived below from the bound instance's own parameters so the checker
// tracks whatever N the DUT was elaborated with (e.g. N=8 default or N=16).
// =============================================================================
bind gemm_systolic_accel gemm_sva_checker #(
    .N      (N),
    .DATA_W (DATA_W),
    .DESC_W (DESC_W)
) u_sva (.*);
