// =============================================================================
// tct_stimulus.svh
// REAL STIMULUS -- owned by the Stimulus/Sequence agent. NON-UVM. `included
// into tb_tensor_core_tile module scope, AFTER tct_scoreboard.svh. Drives the
// shared DUT-facing signals declared in tb_tensor_core_tile.sv directly via
// @(posedge clk) sequential tasks (no interfaces/clocking blocks), and calls
// the scoreboard hooks (sb_* -- see tct_scoreboard.svh) so pass_count/
// fail_count become a real, meaningful result.
//
// Legal mechanism only: this file constructs transaction payloads on the
// shared module-scope signals and lets the existing DUT translate them; it
// never forces or releases or backdoors any signal, never writes any
// hierarchical path into the interface or design instances, and never
// redeclares or edits any environment component.
//
// Scenario -> requirement / coverage traceability (see report for full
// bundle): each run_op() call below is annotated with the requirement IDs
// and coverage buckets it targets.
//
// HONESTY NOTE (ERR-001): a genuine 32-bit accumulator wrap is UNREACHABLE
// with N=8,K=8 INT8 operands (max |acc| = 8*128*127 = 130048 << 2^31). No
// scenario below attempts to fabricate a wrap; the golden model's wrap
// arithmetic is exercised as a structural no-op at these magnitudes, which
// is the honest, reachable behavior. See LIMITATIONS in the delivered
// report for this and other stimulus-lane scope notes.
// =============================================================================

// ---------------------------------------------------------------------------
// Low-level driver tasks (suggested names from the harness header).
// ---------------------------------------------------------------------------

// drv_pulse_start: wait for the DUT to be idle (busy==0), then present a
// single-cycle start pulse on the posedge that samples it (start is only
// sampled while state==IDLE per rtl/ctrl_fsm.v).
task automatic drv_pulse_start();
  begin
    while (busy !== 1'b0) @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;
  end
endtask

// drv_set_requant: latch req_scale/req_shift onto the shared signals. No
// handshake exists for this config -- it is held stable for the whole op.
task automatic drv_set_requant(input logic [SCALE_W-1:0] scale, input logic [SHIFT_W-1:0] shift);
  begin
    req_scale = scale;
    req_shift = shift;
  end
endtask

// drv_load_weight_beat: present one weight beat and consume its accepting
// edge once w_load_ready is observed high (robust handshake pattern -- ready
// is combinational and holds high for the whole LOAD_WEIGHTS phase per the
// DUT timing contract, so the wait below normally falls straight through).
task automatic drv_load_weight_beat(input logic [7:0] data, input logic last);
  int unsigned guard;
  begin
    w_load_data  = data;
    w_load_last  = last;
    w_load_valid = 1'b1;
    guard = 0;
    while (w_load_ready !== 1'b1 && guard < 1000) begin
      @(posedge clk);
      guard = guard + 1;
    end
    @(posedge clk); // accepting edge
  end
endtask

// drv_load_weights: drive all 64 weight beats (linear addr = k*8+j,
// row-major, matching sb_load_weights()/weight_buffer.v). If do_stalls is
// set, insert an idle (valid=0) gap mid-stream to exercise the DUT's
// tolerance of a stalled loader without data loss.
task automatic drv_load_weights(input byte signed w[64], input bit do_stalls);
  int idx;
  begin
    for (idx = 0; idx < 64; idx = idx + 1) begin
      if (do_stalls && (idx == 17 || idx == 40)) begin
        w_load_valid = 1'b0;
        @(posedge clk);
      end
      drv_load_weight_beat(w[idx], (idx == 63));
    end
    w_load_valid = 1'b0;
  end
endtask

// drv_send_activation_beat: present one packed activation beat (word[i*8+:8]
// = A[k][i]) and consume its accepting edge once a_in_ready is observed
// high.
task automatic drv_send_activation_beat(input logic [N*8-1:0] data, input logic last);
  int unsigned guard;
  begin
    a_in_data  = data;
    a_in_last  = last;
    a_in_valid = 1'b1;
    guard = 0;
    while (a_in_ready !== 1'b1 && guard < 1000) begin
      @(posedge clk);
      guard = guard + 1;
    end
    @(posedge clk); // accepting edge
  end
endtask

// drv_send_activations: drive all 8 activation beats. If do_stalls is set,
// insert a multi-cycle gap (valid=0) mid-stream to exercise the DUT's
// tolerance of a stalled activation source without data loss.
task automatic drv_send_activations(input byte signed a[8][8], input bit do_stalls);
  int k, i;
  logic [N*8-1:0] word;
  begin
    for (k = 0; k < 8; k = k + 1) begin
      if (do_stalls && (k == 3)) begin
        a_in_valid = 1'b0;
        @(posedge clk);
        @(posedge clk);
      end
      word = '0;
      for (i = 0; i < 8; i = i + 1) begin
        word[i*8 +: 8] = a[k][i];
      end
      drv_send_activation_beat(word, (k == 7));
    end
    a_in_valid = 1'b0;
  end
endtask

// drv_wait_output_beat: sample one drained output beat WHILE y_out_valid is
// high and BEFORE the accepting posedge (per DUT timing contract), then
// return the sampled payload to the caller. Does not itself check anything.
task automatic drv_wait_output_beat(output logic [N*DW_OUT-1:0] data, output logic last);
  int unsigned guard;
  begin
    guard = 0;
    while (y_out_valid !== 1'b1 && guard < 1000) begin
      @(posedge clk);
      guard = guard + 1;
    end
    data = y_out_data;
    last = y_out_last;
  end
endtask

// drv_drain: accept all 8 output beats, calling sb_check_out_beat() per
// beat (row_idx = drain sequence position, matching drain_row) and
// sb_check_overflow_final() once after the final accepting edge. If
// do_backpressure is set, deassert y_out_ready for a couple of cycles
// mid-drain to exercise backpressure tolerance (y_out_data is combinational
// off drain_row, which only advances on an accepted beat, so it holds
// stable across the stall by construction).
task automatic drv_drain(input bit do_backpressure);
  int row;
  logic [N*DW_OUT-1:0] data;
  logic last;
  begin
    for (row = 0; row < 8; row = row + 1) begin
      if (do_backpressure && (row == 2)) begin
        y_out_ready = 1'b0;
        @(posedge clk);
        @(posedge clk);
      end
      y_out_ready = 1'b1;
      drv_wait_output_beat(data, last);
      sb_check_out_beat(row, data, last, overflow);
      @(posedge clk); // accepting edge
    end
    sb_check_overflow_final(overflow);
    y_out_ready = 1'b0;
  end
endtask

// ---------------------------------------------------------------------------
// run_op: full single-op sequence -- golden model setup (exact call order
// required by the scoreboard contract), then drive load/compute/drain on
// the DUT. Waits for the DUT to return to IDLE before starting (single
// weight buffer -- REQ-STATE-002/REQ-GEMM-017 require a fresh load per op).
// ---------------------------------------------------------------------------
task automatic run_op(
    input byte signed          w[64],
    input byte signed          a[8][8],
    input logic [SCALE_W-1:0]  scale,
    input logic [SHIFT_W-1:0]  shift,
    input bit                  stall_w,
    input bit                  stall_a,
    input bit                  backpressure_y
);
  begin
    while (busy !== 1'b0) @(posedge clk);

    // Golden model setup -- REQUIRED order: op_start -> load_weights ->
    // set_requant -> set_activations (sb_set_activations recomputes the
    // golden acc+requant+expected-sticky from the currently loaded
    // weights/requant config).
    sb_op_start();
    sb_load_weights(w);
    sb_set_requant(scale, shift);
    sb_set_activations(a);

    // Drive the DUT for the same op.
    drv_set_requant(scale, shift);
    drv_pulse_start();
    drv_load_weights(w, stall_w);
    drv_send_activations(a, stall_a);
    drv_drain(backpressure_y);
  end
endtask

// ---------------------------------------------------------------------------
// Small helpers for building deterministic directed matrices and for the
// constrained-random scenario.
// ---------------------------------------------------------------------------
function automatic byte signed rand_sbyte();
  bit [7:0] v;
  begin
    v = 8'($urandom_range(0, 255));
    rand_sbyte = v; // reinterpret bit pattern as signed INT8
  end
endfunction

// ---------------------------------------------------------------------------
// stim_run_all: top-level entry point (REQUIRED name/signature). Runs all
// directed + constrained-random scenarios below and returns (never
// $finish/$fatal). Each scenario is commented with the requirement IDs /
// coverage buckets it targets.
// ---------------------------------------------------------------------------
task automatic stim_run_all();
  byte signed w1a[64];
  byte signed a1a[8][8];
  byte signed w1b[64];
  byte signed a1b[8][8];
  byte signed w2[64];
  byte signed a2[8][8];
  byte signed w3[64];
  byte signed a3[8][8];
  byte signed wr[64];
  byte signed ar[8][8];
  int k, j, i, op;
  logic [SCALE_W-1:0] rscale;
  logic [SHIFT_W-1:0] rshift;
  bit rsw, rsa, rbp;
  begin

    // -----------------------------------------------------------------
    // Coverage-only probe: BP_AIN. Bucket BP_AIN (coverage/tct_coverage.svh)
    // fires when (a_in_valid && !a_in_ready) is observed. Per
    // rtl/tensor_core_tile.v, a_in_ready = compute_active && !a_fifo_full,
    // so a_in_ready is forced low whenever the FSM is outside COMPUTE. The
    // DUT is guaranteed idle/IDLE here (tb_apply_reset() just returned,
    // busy==0, no start pulsed yet), so asserting a_in_valid now cannot be
    // accepted -- a_in_ready stays 0, hitting BP_AIN -- and the activation
    // FIFO is continuously flushed while not in COMPUTE (NOTE-FIFO-FLUSH),
    // so no beat is captured and nothing is corrupted for the real
    // scenarios that follow. This drives no accepted beat and adds no
    // scoreboard checks. Targets: BP_AIN, REQ-GEMM-015, REQ-PROT-002.
    // -----------------------------------------------------------------
    while (busy !== 1'b0) @(posedge clk);
    a_in_data  = '0;
    a_in_last  = 1'b0;
    a_in_valid = 1'b1;
    repeat (3) @(posedge clk);
    a_in_valid = 1'b0;
    a_in_data  = '0;
    a_in_last  = 1'b0;

    // -----------------------------------------------------------------
    // Scenario 1a: directed simple, identity-like weights (scaled by 3)
    // + a known small signed activation matrix, exact-passthrough requant
    // (scale=256,shift=8 => shifted = acc exactly, no rounding error since
    // 256 is an exact power-of-two multiple of the shift). No saturation.
    // Targets: REQ-GEMM-001..004 (bit-exact MAC), REQ-GEMM-005/006
    // (requant passthrough), REQ-PROT-003 (last marker), SHIFT_NONZERO.
    // -----------------------------------------------------------------
    for (k = 0; k < 8; k = k + 1) begin
      for (j = 0; j < 8; j = j + 1) begin
        w1a[k*8+j] = (k == j) ? 8'sd3 : 8'sd0;
      end
    end
    for (k = 0; k < 8; k = k + 1) begin
      for (i = 0; i < 8; i = i + 1) begin
        a1a[k][i] = byte'(i - k); // range -7..7
      end
    end
    run_op(w1a, a1a, 16'h0100, 5'd8, 1'b0, 1'b0, 1'b0);

    // -----------------------------------------------------------------
    // Scenario 1b: directed simple, general small-value weight + activation
    // matrix (non-identity), same exact-passthrough requant, no saturation.
    // Targets: REQ-GEMM-001..004, REQ-GEMM-014 (per-beat row compare).
    // -----------------------------------------------------------------
    for (k = 0; k < 8; k = k + 1) begin
      for (j = 0; j < 8; j = j + 1) begin
        w1b[k*8+j] = byte'(((k*3+j) % 5) - 2); // range -2..2
      end
    end
    for (k = 0; k < 8; k = k + 1) begin
      for (i = 0; i < 8; i = i + 1) begin
        a1b[k][i] = byte'(((k + 2*i) % 5) - 2); // range -2..2
      end
    end
    run_op(w1b, a1b, 16'h0100, 5'd8, 1'b0, 1'b0, 1'b0);

    // -----------------------------------------------------------------
    // Scenario 2: directed signed/negative mix, shift==0 exact passthrough
    // (scale=1,shift=0 => shifted=acc exactly). Weights and activations
    // both span negative and positive INT8 values.
    // Targets: REQ-GEMM-001..004 signed MAC, SHIFT_ZERO coverage bucket.
    // -----------------------------------------------------------------
    for (k = 0; k < 8; k = k + 1) begin
      for (j = 0; j < 8; j = j + 1) begin
        w2[k*8+j] = byte'(((k+j) % 7) - 3); // range -3..3
      end
    end
    for (k = 0; k < 8; k = k + 1) begin
      for (i = 0; i < 8; i = i + 1) begin
        a2[k][i] = byte'(((k*3+i) % 7) - 3); // range -3..3
      end
    end
    run_op(w2, a2, 16'h0001, 5'd0, 1'b0, 1'b0, 1'b0);

    // -----------------------------------------------------------------
    // Scenario 3: requant SATURATION (ERR-002). Constant activation=5 on
    // every beat/lane, weight column j-4 replicated on every row k, so
    // acc[i][j] = 40*(j-4) is IDENTICAL for every output row i and spans
    // both large negative (j=0 -> -160) and large positive (j=7 -> 120)
    // magnitudes. With scale=0xFFFF, shift=1 both extremes clip hard:
    // j=0 clips to -128 (SAT_NEG128), j=7 clips to +127 (SAT_POS127), on
    // every one of the 8 drained rows. Verifies sticky overflow==1 via
    // sb_check_overflow_final and per-lane clipped values via
    // sb_check_out_beat. Targets: ERR-002, REQ-GEMM-006, SAT_POS127,
    // SAT_NEG128, SHIFT_NONZERO (shift=1).
    // -----------------------------------------------------------------
    for (k = 0; k < 8; k = k + 1) begin
      for (j = 0; j < 8; j = j + 1) begin
        w3[k*8+j] = byte'(j - 4); // range -4..3, identical across k
      end
    end
    for (k = 0; k < 8; k = k + 1) begin
      for (i = 0; i < 8; i = i + 1) begin
        a3[k][i] = 8'sd5;
      end
    end
    run_op(w3, a3, 16'hFFFF, 5'd1, 1'b0, 1'b0, 1'b0);

    // -----------------------------------------------------------------
    // Scenario 3b: OVF_CLEARED_BYSTART. Immediately follow the saturating
    // op above with a fresh non-saturating op (reuse the 1a config). The
    // DUT clears overflow_r on the start&&!busy edge (rtl/tensor_core_tile.v)
    // and the golden model clears sb_sticky_exp in sb_op_start() at the top
    // of run_op(), so sb_check_overflow_final() at the end of THIS op must
    // observe overflow==0, proving the sticky flag was cleared by the new
    // start rather than latched from the prior op.
    // Targets: U-007, ERR-002 (clear-on-start), OVF_CLEARED_BYSTART bucket.
    // -----------------------------------------------------------------
    run_op(w1a, a1a, 16'h0100, 5'd8, 1'b0, 1'b0, 1'b0);

    // -----------------------------------------------------------------
    // Scenario 5: backpressure on y_out_ready during DRAIN (deassert for
    // 2 cycles mid-drain in drv_drain when row==2). Output data is
    // combinational off drain_row, which only advances on an accepted
    // beat, so it holds stable across the stall; the same
    // sb_check_out_beat() per-beat compare after the ready pulse proves
    // no corruption occurred. Targets: REQ-PROT (ready backpressure
    // tolerance), REQ-GEMM-014.
    // -----------------------------------------------------------------
    run_op(w1b, a1b, 16'h0100, 5'd8, 1'b0, 1'b0, 1'b1);

    // -----------------------------------------------------------------
    // Scenario 6: stalls -- gaps in w_load_valid mid-load and a_in_valid
    // mid-activation-stream (reuses the signed/negative matrix from
    // scenario 2). Verifies no data loss: final MAC still bit-exact.
    // Targets: REQ-GEMM-012/013 (accept-only-while-valid-and-ready),
    // REQ-GEMM-009 (load must fully complete despite stalls).
    // -----------------------------------------------------------------
    run_op(w2, a2, 16'h0001, 5'd0, 1'b1, 1'b1, 1'b0);

    // -----------------------------------------------------------------
    // Directed reset-recovery: apply a mid-suite reset (idempotent, DUT is
    // idle here since run_op always waits for busy==0 and the drain loop
    // above fully returns to IDLE), then confirm normal operation resumes
    // with a fresh op. Targets: REQ-GEMM-016 (sync active-low reset),
    // general reset-recovery robustness.
    // -----------------------------------------------------------------
    tb_apply_reset();
    run_op(w1a, a1a, 16'h0100, 5'd8, 1'b0, 1'b0, 1'b0);

    // -----------------------------------------------------------------
    // Scenario 8: constrained-random -- randomized signed weights/
    // activations/scale/shift, randomized stall/backpressure knobs, golden
    // computed per-op by the scoreboard. 12 ops, well under the global
    // 200000-cycle watchdog (each op is <= ~90 cycles).
    // Targets: broad coverage sweep across SHIFT_ZERO/SHIFT_NONZERO,
    // SAT_POS127/SAT_NEG128 (whichever randomly occur), general signed MAC.
    // -----------------------------------------------------------------
    for (op = 0; op < 12; op = op + 1) begin
      for (k = 0; k < 64; k = k + 1) begin
        wr[k] = rand_sbyte();
      end
      for (k = 0; k < 8; k = k + 1) begin
        for (i = 0; i < 8; i = i + 1) begin
          ar[k][i] = rand_sbyte();
        end
      end
      rscale = SCALE_W'($urandom_range(1, 65535));
      rshift = SHIFT_W'($urandom_range(0, 31));
      rsw    = 1'($urandom_range(0, 1));
      rsa    = 1'($urandom_range(0, 1));
      rbp    = 1'($urandom_range(0, 1));
      run_op(wr, ar, rscale, rshift, rsw, rsa, rbp);
    end

  end
endtask
