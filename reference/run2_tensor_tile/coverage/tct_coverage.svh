// =============================================================================
// tct_coverage.svh
// MANUAL functional coverage for tensor_core_tile (INT8 GEMM tile).
// Owner: Coverage and SVA Agent.
//
// IMPORTANT -- WHY NOT SV COVERGROUPS:
//   The Verilator 5.044 tool does NOT support SystemVerilog functional coverage
//   constructs ('covergroup' / 'coverpoint' / 'bins' / 'cross'). This file
//   therefore implements functional coverage MANUALLY: plain
//   'always @(posedge clk)' blocks set one-hot "hit" flag bits / increment
//   counters when an event of interest is observed, and a final report
//   task prints which buckets were hit (and how many times) plus a
//   holes-count function for pass/fail gating by the calling testbench.
//   These are NOT SV covergroups; they are hand-rolled bucket counters
//   that are 100% synthesizable-subset SystemVerilog, fully supported by
//   the tool's '--binary --timing' flow.
//
// INCLUDE / COMPILE CONTRACT:
//   'included INSIDE the tb_tensor_core_tile module body, AFTER the 'dut'
//   instance is elaborated. Relies on these identifiers already being in
//   scope (declared in tb_tensor_core_tile.sv, do NOT redeclare here):
//     clk, rst_n, start, busy, done, overflow,
//     a_in_valid, a_in_ready, y_out_valid, y_out_ready
//   and hierarchical references into the DUT:
//     dut.u_ctrl_fsm.state          (2'd0 IDLE, 2'd1 LOAD_WEIGHTS,
//                                     2'd2 COMPUTE, 2'd3 DRAIN)
//     dut.u_act_fifo.full
//     dut.u_act_fifo.empty
//     dut.u_act_fifo.flush
//     dut.compute_active
//     dut.REQUANT_LANE[j].u_requant_unit.sat   (per-lane saturation, j=0..N-1)
//     dut.REQUANT_LANE[j].u_requant_unit.q_out (clipped INT8 lane output)
//   req_shift is sampled directly off the shared 'req_shift' TB signal.
//   N (systolic array dimension) is expected to already be declared as a
//   localparam int N in the including module (tb_tensor_core_tile.sv
//   declares 'localparam int N = 8;' before this include point).
//
// REQUIRED VERILATOR COMPILE FLAGS:
//     -Icoverage             <-- so 'include "tct_coverage.svh" resolves
//   (No special Verilator coverage flags are required since this is plain
//   procedural code, not a 'covergroup'. Verilator's own line/toggle
//   coverage, if desired, is an orthogonal '--coverage' flag and is not
//   used/needed by this file.)
//
// CALL CONTRACT: the including testbench should call
//   cov_report();          at end-of-test (prints bucket hit table)
//   $unsigned'(cov_holes()) to get an un-hit bucket count for gating.
// =============================================================================

// -----------------------------------------------------------------------
// Bucket group 1: FSM state visited (4 buckets) -- REQ-GEMM-009/010,
// REQ-STATE-001.
// -----------------------------------------------------------------------
bit cov_hit_fsm_idle;
bit cov_hit_fsm_load;
bit cov_hit_fsm_compute;
bit cov_hit_fsm_drain;
int unsigned cov_cnt_fsm_idle;
int unsigned cov_cnt_fsm_load;
int unsigned cov_cnt_fsm_compute;
int unsigned cov_cnt_fsm_drain;

always @(posedge clk) begin
  if (rst_n) begin
    case (dut.u_ctrl_fsm.state)
      2'd0: begin cov_hit_fsm_idle    <= 1'b1; cov_cnt_fsm_idle    <= cov_cnt_fsm_idle    + 1; end
      2'd1: begin cov_hit_fsm_load    <= 1'b1; cov_cnt_fsm_load    <= cov_cnt_fsm_load    + 1; end
      2'd2: begin cov_hit_fsm_compute <= 1'b1; cov_cnt_fsm_compute <= cov_cnt_fsm_compute + 1; end
      2'd3: begin cov_hit_fsm_drain   <= 1'b1; cov_cnt_fsm_drain   <= cov_cnt_fsm_drain   + 1; end
      default: ;
    endcase
  end
end

// -----------------------------------------------------------------------
// Bucket group 2: Requant saturation (3 buckets) -- REQ-GEMM-006, ERR-002.
// Observed across all N requant lanes during an accepted/valid drain beat.
// -----------------------------------------------------------------------
bit cov_hit_sat_pos127;
bit cov_hit_sat_neg128;
bit cov_hit_sat_none;
int unsigned cov_cnt_sat_pos127;
int unsigned cov_cnt_sat_neg128;
int unsigned cov_cnt_sat_none;

genvar cov_j;
generate
  for (cov_j = 0; cov_j < N; cov_j = cov_j + 1) begin : COV_REQUANT_LANE
    always @(posedge clk) begin
      if (rst_n && dut.u_ctrl_fsm.drain_active && y_out_valid) begin
        if (dut.REQUANT_LANE[cov_j].u_requant_unit.sat) begin
          if (dut.REQUANT_LANE[cov_j].u_requant_unit.q_out == 8'sd127) begin
            cov_hit_sat_pos127 <= 1'b1;
            cov_cnt_sat_pos127 <= cov_cnt_sat_pos127 + 1;
          end else if (dut.REQUANT_LANE[cov_j].u_requant_unit.q_out == -8'sd128) begin
            cov_hit_sat_neg128 <= 1'b1;
            cov_cnt_sat_neg128 <= cov_cnt_sat_neg128 + 1;
          end
        end else begin
          cov_hit_sat_none <= 1'b1;
          cov_cnt_sat_none <= cov_cnt_sat_none + 1;
        end
      end
    end
  end
endgenerate

// -----------------------------------------------------------------------
// Bucket group 3: Rounding shift exercised (2 buckets) -- REQ-GEMM-005.
// Sampled off the shared req_shift config input during an active drain
// beat (quasi-static per ASSUMPTION-003 / REQ-TIME-003).
// -----------------------------------------------------------------------
bit cov_hit_shift_zero;
bit cov_hit_shift_nonzero;
int unsigned cov_cnt_shift_zero;
int unsigned cov_cnt_shift_nonzero;

always @(posedge clk) begin
  if (rst_n && dut.u_ctrl_fsm.drain_active && y_out_valid) begin
    if (req_shift == '0) begin
      cov_hit_shift_zero    <= 1'b1;
      cov_cnt_shift_zero    <= cov_cnt_shift_zero + 1;
    end else begin
      cov_hit_shift_nonzero <= 1'b1;
      cov_cnt_shift_nonzero <= cov_cnt_shift_nonzero + 1;
    end
  end
end

// -----------------------------------------------------------------------
// Bucket group 4: Activation FIFO events (3 buckets) -- REQ-GEMM-008/015,
// REQ-STOR-002, NOTE-FIFO-FLUSH/ERR-005.
// -----------------------------------------------------------------------
bit cov_hit_afifo_full;
bit cov_hit_afifo_empty_in_compute;
bit cov_hit_afifo_flush;
int unsigned cov_cnt_afifo_full;
int unsigned cov_cnt_afifo_empty_in_compute;
int unsigned cov_cnt_afifo_flush;

always @(posedge clk) begin
  if (rst_n) begin
    if (dut.u_act_fifo.full) begin
      cov_hit_afifo_full <= 1'b1;
      cov_cnt_afifo_full <= cov_cnt_afifo_full + 1;
    end
    if (dut.compute_active && dut.u_act_fifo.empty) begin
      cov_hit_afifo_empty_in_compute <= 1'b1;
      cov_cnt_afifo_empty_in_compute <= cov_cnt_afifo_empty_in_compute + 1;
    end
    if (dut.u_act_fifo.flush) begin
      cov_hit_afifo_flush <= 1'b1;
      cov_cnt_afifo_flush <= cov_cnt_afifo_flush + 1;
    end
  end
end

// -----------------------------------------------------------------------
// Bucket group 5: Backpressure (2 buckets) -- REQ-GEMM-015, REQ-PROT-002.
// -----------------------------------------------------------------------
bit cov_hit_bp_yout;
bit cov_hit_bp_ain;
int unsigned cov_cnt_bp_yout;
int unsigned cov_cnt_bp_ain;

always @(posedge clk) begin
  if (rst_n) begin
    if (y_out_valid && !y_out_ready) begin
      cov_hit_bp_yout <= 1'b1;
      cov_cnt_bp_yout <= cov_cnt_bp_yout + 1;
    end
    if (a_in_valid && !a_in_ready) begin
      cov_hit_bp_ain <= 1'b1;
      cov_cnt_bp_ain <= cov_cnt_bp_ain + 1;
    end
  end
end

// -----------------------------------------------------------------------
// Bucket group 6: Overflow sticky (2 buckets) -- RESOLVED-007, ERR-001/002.
// -----------------------------------------------------------------------
bit cov_hit_ovf_set;
bit cov_hit_ovf_cleared_by_start;
int unsigned cov_cnt_ovf_set;
int unsigned cov_cnt_ovf_cleared_by_start;

bit cov_ovf_prev;
always @(posedge clk) begin
  if (!rst_n) begin
    cov_ovf_prev <= 1'b0;
  end else begin
    if (!cov_ovf_prev && overflow) begin
      cov_hit_ovf_set <= 1'b1;
      cov_cnt_ovf_set <= cov_cnt_ovf_set + 1;
    end
    if (cov_ovf_prev && start && !busy) begin
      cov_hit_ovf_cleared_by_start <= 1'b1;
      cov_cnt_ovf_cleared_by_start <= cov_cnt_ovf_cleared_by_start + 1;
    end
    cov_ovf_prev <= overflow;
  end
end

// -----------------------------------------------------------------------
// Bucket group 7: done pulse observed (1 bucket) -- REQ-GEMM-011.
// -----------------------------------------------------------------------
bit cov_hit_done_pulse;
int unsigned cov_cnt_done_pulse;

always @(posedge clk) begin
  if (rst_n && done) begin
    cov_hit_done_pulse <= 1'b1;
    cov_cnt_done_pulse <= cov_cnt_done_pulse + 1;
  end
end

// =============================================================================
// cov_report() -- prints every bucket's hit/miss status and count.
// =============================================================================
task automatic cov_report();
  begin
    $display("==== MANUAL FUNCTIONAL COVERAGE REPORT (tct_coverage.svh) ====");
    $display("(Verilator has no SV covergroup support; these are hand-rolled bucket counters)");
    $display("-- FSM state visited (4 buckets, REQ-GEMM-009/010/REQ-STATE-001) --");
    $display("  IDLE           hit=%0d count=%0d", cov_hit_fsm_idle,    cov_cnt_fsm_idle);
    $display("  LOAD_WEIGHTS   hit=%0d count=%0d", cov_hit_fsm_load,    cov_cnt_fsm_load);
    $display("  COMPUTE        hit=%0d count=%0d", cov_hit_fsm_compute, cov_cnt_fsm_compute);
    $display("  DRAIN          hit=%0d count=%0d", cov_hit_fsm_drain,   cov_cnt_fsm_drain);
    $display("-- Requant saturation (3 buckets, REQ-GEMM-006/ERR-002) --");
    $display("  SAT_POS127     hit=%0d count=%0d", cov_hit_sat_pos127, cov_cnt_sat_pos127);
    $display("  SAT_NEG128     hit=%0d count=%0d", cov_hit_sat_neg128, cov_cnt_sat_neg128);
    $display("  NO_CLIP        hit=%0d count=%0d", cov_hit_sat_none,   cov_cnt_sat_none);
    $display("-- Rounding shift exercised (2 buckets, REQ-GEMM-005) --");
    $display("  SHIFT_ZERO     hit=%0d count=%0d", cov_hit_shift_zero,    cov_cnt_shift_zero);
    $display("  SHIFT_NONZERO  hit=%0d count=%0d", cov_hit_shift_nonzero, cov_cnt_shift_nonzero);
    $display("-- Activation FIFO events (3 buckets, REQ-GEMM-008/015, NOTE-FIFO-FLUSH) --");
    $display("  A_FIFO_FULL          hit=%0d count=%0d", cov_hit_afifo_full,             cov_cnt_afifo_full);
    $display("  A_FIFO_EMPTY_COMPUTE hit=%0d count=%0d", cov_hit_afifo_empty_in_compute, cov_cnt_afifo_empty_in_compute);
    $display("  A_FIFO_FLUSH         hit=%0d count=%0d", cov_hit_afifo_flush,            cov_cnt_afifo_flush);
    $display("-- Backpressure (2 buckets, REQ-GEMM-015) --");
    $display("  BP_YOUT        hit=%0d count=%0d", cov_hit_bp_yout, cov_cnt_bp_yout);
    $display("  BP_AIN         hit=%0d count=%0d", cov_hit_bp_ain,  cov_cnt_bp_ain);
    $display("-- Overflow sticky (2 buckets, RESOLVED-007) --");
    $display("  OVF_SET             hit=%0d count=%0d", cov_hit_ovf_set,              cov_cnt_ovf_set);
    $display("  OVF_CLEARED_BYSTART hit=%0d count=%0d", cov_hit_ovf_cleared_by_start, cov_cnt_ovf_cleared_by_start);
    $display("-- done pulse (1 bucket, REQ-GEMM-011) --");
    $display("  DONE_PULSE     hit=%0d count=%0d", cov_hit_done_pulse, cov_cnt_done_pulse);
    $display("TOTAL BUCKETS=17  HOLES=%0d", cov_holes());
  end
endtask

// =============================================================================
// cov_holes() -- returns number of un-hit buckets (17 total).
// =============================================================================
function automatic int cov_holes();
  int h;
  begin
    h = 0;
    if (!cov_hit_fsm_idle)              h++;
    if (!cov_hit_fsm_load)              h++;
    if (!cov_hit_fsm_compute)           h++;
    if (!cov_hit_fsm_drain)             h++;
    if (!cov_hit_sat_pos127)            h++;
    if (!cov_hit_sat_neg128)            h++;
    if (!cov_hit_sat_none)              h++;
    if (!cov_hit_shift_zero)            h++;
    if (!cov_hit_shift_nonzero)         h++;
    if (!cov_hit_afifo_full)            h++;
    if (!cov_hit_afifo_empty_in_compute) h++;
    if (!cov_hit_afifo_flush)           h++;
    if (!cov_hit_bp_yout)               h++;
    if (!cov_hit_bp_ain)                h++;
    if (!cov_hit_ovf_set)               h++;
    if (!cov_hit_ovf_cleared_by_start)  h++;
    if (!cov_hit_done_pulse)            h++;
    cov_holes = h;
  end
endfunction
