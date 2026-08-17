// =============================================================================
// File   : gemm_tests.svh
// -----------------------------------------------------------------------------
// PURPOSE
//   Directed + constrained-random test scenarios for gemm_systolic_accel.
//   `include`-d INSIDE gemm_tb_top's module body (same convention as
//   gemm_bfm.svh), so every task below shares gemm_tb_top's DUT-facing
//   signals (clk, rst_n, desc_valid/desc_data/desc_ready, w_*, a_*, y_*,
//   busy, done, err_overflow), the N localparam, and the BFM tasks
//   (apply_reset, send_descriptor, send_weight_beat, send_act_beat,
//   recv_result_beat, make_descriptor, check) plus the bookkeeping ints
//   (error_count, check_count).
//
//   STIMULUS ONLY: every DUT-facing signal is driven exclusively through the
//   existing BFM tasks (send_descriptor/send_weight_beat/send_act_beat/
//   recv_result_beat/apply_reset) or through the same plain module-scope
//   signals those BFM tasks themselves drive (desc_valid/desc_data, used
//   directly ONLY to model a second host descriptor request arriving while
//   the core is busy -- there is no other way to express "attempt a
//   descriptor while busy" through the provided task API, and this uses the
//   exact same signals/mechanism the provided BFM already uses, not a
//   backdoor). No force/release, no vif./tb_top./dut. hierarchical writes,
//   no assign statements, no scoreboard math (that lives in gemm_ref_pkg,
//   called here only through its published compare functions).
//
// REQUIREMENT COVERAGE (see run_all_tests() below for the full list mapped
//   to each test task):
//   REQ-GEMM-001/004/005/006/007/008/009/010/012/013, REQ-PROTO-002/003/004,
//   ERR-001/002/003/004, UQ-003/UQ-004.
// =============================================================================

import gemm_ref_pkg::*;

// ---------------------------------------------------------------------------
// Global test bookkeeping (monitor-only reads of DUT status outputs; never
// drives them).
// ---------------------------------------------------------------------------
localparam int unsigned RANDOM_SEED = 32'hFEED_1234;

int unsigned done_pulse_count;

// gemm_tb_top declares error_count as `int unsigned`, but the ref_pkg
// compare functions take their running total by `ref int` (signed). The
// tool requires an exact type match on ref arguments, so this local signed
// shadow is used as the pass-through argument and folded back into the
// real (unsigned) error_count immediately after each call via the sync
// helpers below. No semantics change: same monotonically increasing error
// tally, just bridged across the signed/unsigned ref argument boundary.
int ref_err_shadow;

function automatic void sync_shadow_from_errcount();
  begin
    ref_err_shadow = int'(error_count);
  end
endfunction

function automatic void sync_errcount_from_shadow();
  begin
    error_count = error_count + (ref_err_shadow - int'(error_count));
  end
endfunction

// Wrapper tasks bridging the signed `ref int error_count` argument required
// by gemm_ref_pkg's compare functions to this module's `int unsigned
// error_count` bookkeeping variable (shared with gemm_bfm.svh's check()
// task). Pure plumbing -- no comparison semantics added or altered here.
task automatic wrap_ref_compare(input gemm_ref_pkg::ref_beat_t expected_q[$],
                                 input gemm_ref_pkg::ref_beat_t actual_q[$],
                                 input int n, input string tag,
                                 output bit pass);
  begin
    sync_shadow_from_errcount();
    pass = gemm_ref_pkg::ref_compare(expected_q, actual_q, n, tag, ref_err_shadow);
    sync_errcount_from_shadow();
  end
endtask

task automatic wrap_ref_compare_overflow(input bit expected_ovf, input bit actual_ovf,
                                          input string tag, output bit pass);
  begin
    sync_shadow_from_errcount();
    pass = gemm_ref_pkg::ref_compare_overflow(expected_ovf, actual_ovf, tag, ref_err_shadow);
    sync_errcount_from_shadow();
  end
endtask

task automatic wrap_ref_compare_produces_output(input bit expected_po, input bit actual_produced,
                                                 input string tag, output bit pass);
  begin
    sync_shadow_from_errcount();
    pass = gemm_ref_pkg::ref_compare_produces_output(expected_po, actual_produced, tag, ref_err_shadow);
    sync_errcount_from_shadow();
  end
endtask

// Background monitor: counts `done` pulses. Reads the DUT status output
// only -- never drives it. Started once via `fork monitor_done_bg();
// join_none;` at the top of run_all_tests().
task automatic monitor_done_bg();
  begin
    done_pulse_count = 0;
    forever begin
      @(posedge clk);
      if (done) done_pulse_count = done_pulse_count + 1;
    end
  end
endtask

// ---------------------------------------------------------------------------
// Beat-building helpers (pure data construction, no DUT signal access).
// ---------------------------------------------------------------------------
function automatic gemm_ref_pkg::ref_beat_t beat_from_lanes(input byte lanes[N], input bit last);
  gemm_ref_pkg::ref_beat_t b;
  int i;
  begin
    b.data = '0;
    for (i = 0; i < N; i++) b.data[i*8 +: 8] = lanes[i];
    b.last = last;
    return b;
  end
endfunction

function automatic gemm_ref_pkg::ref_beat_t beat_const(input byte v, input bit last);
  byte lanes[N];
  int i;
  begin
    for (i = 0; i < N; i++) lanes[i] = v;
    return beat_from_lanes(lanes, last);
  end
endfunction

function automatic gemm_ref_pkg::ref_beat_t rnd_beat(input bit last);
  byte lanes[N];
  int i;
  begin
    for (i = 0; i < N; i++) lanes[i] = byte'($urandom_range(0, 255));
    return beat_from_lanes(lanes, last);
  end
endfunction

task automatic build_random_queue(output gemm_ref_pkg::ref_beat_t q[$], input int count);
  int i;
  begin
    q.delete();
    for (i = 0; i < count; i++) q.push_back(rnd_beat(i == count-1));
  end
endtask

task automatic build_uniform_queue(output gemm_ref_pkg::ref_beat_t q[$], input int count, input byte v);
  int i;
  begin
    q.delete();
    for (i = 0; i < count; i++) q.push_back(beat_const(v, i == count-1));
  end
endtask

// One-hot activation / constant-row-weight job builder used by the
// requant_boundary test: with act_beats[k].lane[i] = (i==k) and
// weight_beats[k].lane[j] = targets[k] for all j, acc[i][j] == targets[i]
// exactly (single surviving K term per output row), letting each output row
// hit a precisely chosen accumulator value.
task automatic build_onehot_job(input byte targets[N],
                                 output gemm_ref_pkg::ref_beat_t wq[$],
                                 output gemm_ref_pkg::ref_beat_t aq[$]);
  byte lanes[N];
  int i, k;
  begin
    wq.delete();
    aq.delete();
    for (k = 0; k < N; k++) begin
      wq.push_back(beat_const(targets[k], k == N-1));
      for (i = 0; i < N; i++) lanes[i] = (i == k) ? byte'(1) : byte'(0);
      aq.push_back(beat_from_lanes(lanes, k == N-1));
    end
  end
endtask

// ---------------------------------------------------------------------------
// Streaming helpers -- thin wrappers over the provided BFM tasks.
// ---------------------------------------------------------------------------
task automatic drive_weight_queue(input gemm_ref_pkg::ref_beat_t q[$], input int max_stall);
  begin
    foreach (q[i]) send_weight_beat(q[i].data[N*8-1:0], q[i].last, max_stall);
  end
endtask

task automatic drive_act_queue(input gemm_ref_pkg::ref_beat_t q[$], input int max_stall);
  begin
    foreach (q[i]) send_act_beat(q[i].data[N*8-1:0], q[i].last, max_stall);
  end
endtask

task automatic capture_result_queue(output gemm_ref_pkg::ref_beat_t q[$], input int max_ystall);
  logic [N*8-1:0] d;
  logic            lst;
  gemm_ref_pkg::ref_beat_t b;
  int guard;
  begin
    q.delete();
    guard = 0;
    forever begin
      recv_result_beat(d, lst, max_ystall);
      b.data      = '0;
      b.data[N*8-1:0] = d;
      b.last      = lst;
      q.push_back(b);
      guard++;
      if (lst) break;
      if (guard > 100000) begin
        check(1'b0, "capture_result_queue: safety guard exceeded waiting for y_last (possible DUT hang)");
        break;
      end
    end
  end
endtask

// Drives weight+act and drains results concurrently (independent channels;
// matches real host behavior -- the host need not know internal FSM phase
// boundaries, and the skid buffers on w_in/a_in/result_out decouple timing).
task automatic stream_and_capture(
    input  gemm_ref_pkg::ref_beat_t wq[$],
    input  gemm_ref_pkg::ref_beat_t aq[$],
    input  int max_wstall, input int max_astall, input int max_ystall,
    output gemm_ref_pkg::ref_beat_t actual_q[$]
  );
  begin
    fork
      drive_weight_queue(wq, max_wstall);
      drive_act_queue(aq, max_astall);
      capture_result_queue(actual_q, max_ystall);
    join
  end
endtask

// Full legal/illegal-agnostic job runner: sends the descriptor, streams the
// given (possibly truncated) weight/act queues, captures the result stream,
// computes the golden expectation via gemm_ref_pkg, and bit-exact compares.
task automatic run_and_check_job(
    input  int    tiles_m, input int tiles_k, input int tiles_n,
    input  int    req_shift, input int req_scale,
    input  gemm_ref_pkg::ref_beat_t wq[$],
    input  gemm_ref_pkg::ref_beat_t aq[$],
    input  int    max_wstall, input int max_astall, input int max_ystall,
    input  string tag,
    output bit    pass
  );
  logic [127:0] desc;
  gemm_ref_pkg::ref_desc_t rdesc;
  gemm_ref_pkg::ref_beat_t expected_q[$], actual_q[$];
  bit produces_output, expected_ovf, cmp_ok, ovf_ok;
  int eff_w, eff_a;
  begin
    desc = make_descriptor(8'(tiles_m), 8'(tiles_k), 8'(tiles_n), 6'(req_shift), 32'(req_scale));

    rdesc.tiles_m   = tiles_m;
    rdesc.tiles_k   = tiles_k;
    rdesc.tiles_n   = tiles_n;
    rdesc.req_shift = req_shift;
    rdesc.req_scale = req_scale;

    send_descriptor(desc);
    stream_and_capture(wq, aq, max_wstall, max_astall, max_ystall, actual_q);

    eff_w = gemm_ref_pkg::ref_eff_words(wq, gemm_ref_pkg::ref_expected_wwords(tiles_k, tiles_n, N));
    eff_a = gemm_ref_pkg::ref_eff_words(aq, gemm_ref_pkg::ref_expected_awords(tiles_m, tiles_k, N));

    gemm_ref_pkg::ref_compute_expected(rdesc, wq, aq, eff_w, eff_a,
                                        expected_q, produces_output, expected_ovf, N);

    wrap_ref_compare(expected_q, actual_q, N, tag, cmp_ok);
    wrap_ref_compare_overflow(expected_ovf, err_overflow, tag, ovf_ok);
    check_count = check_count + 2;
    pass = cmp_ok && ovf_ok;
  end
endtask

// ---------------------------------------------------------------------------
// TEST 1 : reset_midjob (REQ-GEMM-010, ERR-004)
// ---------------------------------------------------------------------------
task automatic test_reset_midjob();
  logic [127:0] desc;
  gemm_ref_pkg::ref_beat_t wq[$], aq[$];
  gemm_ref_pkg::ref_beat_t b;
  bit pass;
  int i;
  begin
    desc = make_descriptor(8'd1, 8'd1, 8'd1, 6'd0, 32'sd1);
    send_descriptor(desc);
    check(busy == 1'b1, "reset_midjob: busy should assert after descriptor accept");

    // Stream a partial weight burst (3 of 8 required beats), never last.
    for (i = 0; i < 3; i++) begin
      b = rnd_beat(1'b0);
      send_weight_beat(b.data[N*8-1:0], 1'b0, 0);
    end

    // Reset mid-job (ERR-004 / REQ-GEMM-010).
    apply_reset(4);

    check(busy         == 1'b0, "reset_midjob: busy must clear after mid-job reset");
    check(y_valid       == 1'b0, "reset_midjob: y_valid must clear (no spurious beats) after mid-job reset");
    check(done           == 1'b0, "reset_midjob: done must clear after mid-job reset");
    check(err_overflow  == 1'b0, "reset_midjob: err_overflow must clear after mid-job reset");
    check(desc_ready     == 1'b1, "reset_midjob: desc_ready must reassert (IDLE) after mid-job reset");

    // Recovery: a clean legal job must complete correctly after the abandoned job.
    build_random_queue(wq, 8);
    build_random_queue(aq, 8);
    run_and_check_job(1, 1, 1, 0, 1, wq, aq, 0, 0, 0, "reset_midjob_recovery", pass);
    check(pass, "reset_midjob: recovery job after reset must match reference model");
  end
endtask

// ---------------------------------------------------------------------------
// TEST 2 : desc_decode_and_busy (REQ-GEMM-001, REQ-PROTO-004)
// ---------------------------------------------------------------------------
task automatic test_desc_decode_and_busy();
  logic [127:0] d1, d2;
  gemm_ref_pkg::ref_beat_t wq1[$], aq1[$], actual1[$], expected1[$];
  gemm_ref_pkg::ref_beat_t wq2[$], aq2[$], actual2[$], expected2[$];
  gemm_ref_pkg::ref_desc_t rd;
  bit produces_output, exp_ovf;
  int guard;
  begin
    d1 = make_descriptor(8'd1, 8'd1, 8'd1, 6'd0, 32'sd1);
    d2 = make_descriptor(8'd1, 8'd1, 8'd1, 6'd0, 32'sd1);

    send_descriptor(d1);
    check(busy       == 1'b1, "desc_decode_and_busy: busy must assert after job1 accept");
    check(desc_ready == 1'b0, "desc_decode_and_busy: desc_ready must deassert while busy");

    // Attempt a 2nd descriptor while busy: drive desc_valid/desc_data
    // directly (same signals the BFM's own send_descriptor task drives) and
    // hold them stable across job1's remaining lifetime, exactly as
    // REQ-PROTO-002 requires once a producer asserts valid with payload.
    // The SVA bind's a_desc_ready_not_busy / a_desc_accept_only_idle
    // properties independently police that no acceptance ever happens
    // while busy is high.
    desc_valid = 1'b1;
    desc_data  = d2;

    build_random_queue(wq1, 8);
    build_random_queue(aq1, 8);
    stream_and_capture(wq1, aq1, 0, 0, 0, actual1);

    // job1 has now completed (busy back to 0). desc_valid/d2 has been held
    // stable throughout; it must be accepted promptly now that the core is
    // idle.
    guard = 0;
    while (!(desc_valid && desc_ready) && (guard < 20)) begin
      @(posedge clk);
      guard++;
    end
    check(desc_valid && desc_ready,
          "desc_decode_and_busy: 2nd descriptor must be accepted once core returns to IDLE");
    @(posedge clk); // consume the accepting edge, same as send_descriptor's own contract
    desc_valid = 1'b0;

    // Drain job2 to leave the DUT idle/clean for later tests, and bonus
    // bit-exact-check both jobs against the reference model.
    build_random_queue(wq2, 8);
    build_random_queue(aq2, 8);
    stream_and_capture(wq2, aq2, 0, 0, 0, actual2);

    rd.tiles_m = 1; rd.tiles_k = 1; rd.tiles_n = 1; rd.req_shift = 0; rd.req_scale = 1;

    gemm_ref_pkg::ref_compute_expected(rd, wq1, aq1,
        gemm_ref_pkg::ref_eff_words(wq1, 8), gemm_ref_pkg::ref_eff_words(aq1, 8),
        expected1, produces_output, exp_ovf, N);
    begin
      bit dd_pass1, dd_pass2;
      wrap_ref_compare(expected1, actual1, N, "desc_decode_job1", dd_pass1);
      check_count = check_count + 1;

      gemm_ref_pkg::ref_compute_expected(rd, wq2, aq2,
          gemm_ref_pkg::ref_eff_words(wq2, 8), gemm_ref_pkg::ref_eff_words(aq2, 8),
          expected2, produces_output, exp_ovf, N);
      wrap_ref_compare(expected2, actual2, N, "desc_decode_job2", dd_pass2);
      check_count = check_count + 1;
    end
  end
endtask

// ---------------------------------------------------------------------------
// TEST 3 : illegal_desc_reject (ERR-002 / UQ-003)
// ---------------------------------------------------------------------------
task automatic test_illegal_case(input logic [7:0] tm, input logic [7:0] tk, input logic [7:0] tn,
                                  input string tag);
  logic [127:0] desc;
  gemm_ref_pkg::ref_desc_t rd;
  bit saw_yvalid;
  int unsigned dpc0;
  int i;
  begin
    rd.tiles_m = int'(tm); rd.tiles_k = int'(tk); rd.tiles_n = int'(tn);
    rd.req_shift = 0; rd.req_scale = 1;
    check(gemm_ref_pkg::ref_is_illegal(rd, N),
          $sformatf("%s: reference model must independently confirm this descriptor is illegal", tag));

    desc = make_descriptor(tm, tk, tn, 6'd0, 32'sd1);
    dpc0 = done_pulse_count;
    saw_yvalid = 1'b0;

    // DEBUG-TEMP: timestamp before sampling window.
    $display("DBG-ILLEGAL[%s] t=%0t PRE-SEND state=%s illegal_desc=%0b busy=%0b y_valid=%0b done_pulse_count=%0d",
              tag, $time, dut.u_ctrl.state.name(), dut.u_ctrl.illegal_desc, busy, y_valid, done_pulse_count);

    // desc_ready is unconditionally high in IDLE regardless of legality;
    // legality is only evaluated in DECODE, so the handshake itself always
    // completes here -- what matters is what happens (or doesn't) after.
    send_descriptor(desc);

    // DEBUG-TEMP: timestamp right after send_descriptor returns (handshake done).
    $display("DBG-ILLEGAL[%s] t=%0t POST-SEND state=%s illegal_desc=%0b tiles_m_r=%0d tiles_k_r=%0d tiles_n_r=%0d exp_w=%0d exp_a=%0d busy=%0b y_valid=%0b",
              tag, $time, dut.u_ctrl.state.name(), dut.u_ctrl.illegal_desc,
              dut.u_ctrl.tiles_m_r, dut.u_ctrl.tiles_k_r, dut.u_ctrl.tiles_n_r,
              dut.u_ctrl.expected_wwords, dut.u_ctrl.expected_awords, busy, y_valid);

    for (i = 0; i < 10; i++) begin
      @(posedge clk);
      // DEBUG-TEMP: cycle-accurate FSM/status monitor over the sampling window.
      $display("DBG-ILLEGAL[%s] t=%0t i=%0d state=%s illegal_desc=%0b tiles_m_r=%0d tiles_k_r=%0d tiles_n_r=%0d busy=%0b y_valid=%0b y_ready=%0b done=%0b done_pulse_count=%0d",
                tag, $time, i, dut.u_ctrl.state.name(), dut.u_ctrl.illegal_desc,
                dut.u_ctrl.tiles_m_r, dut.u_ctrl.tiles_k_r, dut.u_ctrl.tiles_n_r,
                busy, y_valid, y_ready, done, done_pulse_count);
      if (y_valid) saw_yvalid = 1'b1;
    end

    check(busy         == 1'b0, $sformatf("%s: core must return to IDLE after illegal descriptor", tag));
    check(desc_ready    == 1'b1, $sformatf("%s: desc_ready must reassert after illegal descriptor discarded", tag));
    check(!saw_yvalid,           $sformatf("%s: no y_valid beats must be produced for an illegal descriptor", tag));
    check(done_pulse_count == dpc0, $sformatf("%s: done must never pulse for an illegal descriptor", tag));
    check(err_overflow == 1'b0, $sformatf("%s: err_overflow must stay 0 for an illegal (non-numeric) reject", tag));

    begin
      bit po_pass;
      wrap_ref_compare_produces_output(1'b0, saw_yvalid, tag, po_pass);
      check_count = check_count + 1;
    end
  end
endtask

task automatic test_illegal_desc_reject();
  begin
    test_illegal_case(8'd0, 8'd1, 8'd1, "illegal_tiles_m_zero");
    test_illegal_case(8'd1, 8'd9, 8'd1, "illegal_tiles_k_over_max");
    test_illegal_case(8'd8, 8'd8, 8'd8, "illegal_capacity_overflow_512words");
  end
endtask

// ---------------------------------------------------------------------------
// TEST 4 : single_tile_mac (REQ-GEMM-004/013, REQ-GEMM-008)
// ---------------------------------------------------------------------------
task automatic test_single_tile_mac();
  gemm_ref_pkg::ref_beat_t wq[$], aq[$];
  bit pass;
  begin
    build_random_queue(wq, N); // expected_wwords = tiles_k*tiles_n*N = 1*1*N
    build_random_queue(aq, N); // expected_awords = tiles_m*tiles_k*N = 1*1*N
    run_and_check_job(1, 1, 1, 0, 1, wq, aq, 0, 0, 0, "single_tile_mac", pass);
    check(pass, "single_tile_mac: 1x1x1 tile output must match reference model bit-exactly, y_last on final beat");
  end
endtask

// ---------------------------------------------------------------------------
// TEST 5 : multi_tile (REQ-GEMM-005)
// ---------------------------------------------------------------------------
task automatic test_multi_tile();
  gemm_ref_pkg::ref_beat_t wq[$], aq[$];
  bit pass;
  begin
    build_random_queue(wq, 2*2*N); // tiles_k*tiles_n*N = 2*2*N = 32 (<=64)
    build_random_queue(aq, 2*2*N); // tiles_m*tiles_k*N = 2*2*N = 32 (<=64)
    run_and_check_job(2, 2, 2, 3, 1, wq, aq, 0, 0, 0, "multi_tile", pass);
    check(pass, "multi_tile: 2x2x2 K-accumulation + M/N tiling + output ordering must match reference model");
  end
endtask

// ---------------------------------------------------------------------------
// TEST 6 : requant_saturation (REQ-GEMM-006/007, REQ-GEMM-012, ERR-001)
// ---------------------------------------------------------------------------
task automatic test_requant_saturation();
  gemm_ref_pkg::ref_beat_t wq[$], aq[$];
  gemm_ref_pkg::ref_beat_t wq2[$], aq2[$], actual2[$], expected2[$];
  gemm_ref_pkg::ref_desc_t rd2;
  bit produces_output, exp_ovf2;
  logic [127:0] desc2;
  bit pass;
  begin
    // Force every lane of every row to saturate: 8 K-steps of 127*127 each
    // accumulate to acc=8*16129=129032, far beyond INT8 range at scale=1/shift=0.
    build_uniform_queue(wq, N, byte'(127));
    build_uniform_queue(aq, N, byte'(127));
    run_and_check_job(1, 1, 1, 0, 1, wq, aq, 0, 0, 0, "requant_saturation_force", pass);
    check(pass, "requant_saturation: forced-saturation job must match reference model (clamp + sticky ovf)");
    check(err_overflow == 1'b1, "requant_saturation: err_overflow must be sticky-set after a saturating job");

    // A NEW legal descriptor must clear the sticky flag on acceptance (REQ-GEMM-012).
    desc2 = make_descriptor(8'd1, 8'd1, 8'd1, 6'd0, 32'sd1);
    send_descriptor(desc2);
    check(err_overflow == 1'b0, "requant_saturation: err_overflow must clear on next descriptor accept");

    build_random_queue(wq2, N);
    build_random_queue(aq2, N);
    stream_and_capture(wq2, aq2, 0, 0, 0, actual2);

    rd2.tiles_m = 1; rd2.tiles_k = 1; rd2.tiles_n = 1; rd2.req_shift = 0; rd2.req_scale = 1;
    gemm_ref_pkg::ref_compute_expected(rd2, wq2, aq2, N, N, expected2, produces_output, exp_ovf2, N);
    begin
      bit rec_pass;
      wrap_ref_compare(expected2, actual2, N, "requant_saturation_recovery", rec_pass);
      check_count = check_count + 1;
    end
  end
endtask

// ---------------------------------------------------------------------------
// TEST 7 : requant_boundary (REQ-GEMM-006/007)
// ---------------------------------------------------------------------------
task automatic test_requant_boundary();
  gemm_ref_pkg::ref_beat_t wqA[$], aqA[$];
  gemm_ref_pkg::ref_beat_t wqB[$], aqB[$];
  byte targetsA[N], targetsB[N];
  bit passA, passB;
  int i;
  begin
    // Job A: req_scale=1, req_shift=0 -- exact +127 / -128 (and adjacent
    // values), no saturation expected anywhere.
    for (i = 0; i < N; i++) targetsA[i] = byte'(0);
    targetsA[0] = byte'(127);
    targetsA[1] = byte'(-128);
    if (N > 2) targetsA[2] = byte'(126);
    if (N > 3) targetsA[3] = byte'(-127);
    build_onehot_job(targetsA, wqA, aqA);
    run_and_check_job(1, 1, 1, 0, 1, wqA, aqA, 0, 0, 0, "requant_boundary_exact", passA);
    check(passA, "requant_boundary: exact +127/-128 rows must pass through with NO saturation");
    check(err_overflow == 1'b0, "requant_boundary: exact-boundary job must not set err_overflow");

    // Job B: req_scale=2, req_shift=0 -- row0 target=64 -> acc=64 ->
    // shifted=128, exactly one past +127 -> must saturate to 127, ovf=1.
    for (i = 0; i < N; i++) targetsB[i] = byte'(0);
    targetsB[0] = byte'(64);
    build_onehot_job(targetsB, wqB, aqB);
    run_and_check_job(1, 1, 1, 0, 2, wqB, aqB, 0, 0, 0, "requant_boundary_just_over", passB);
    check(passB, "requant_boundary: just-over-boundary value must saturate to exactly 127");
    check(err_overflow == 1'b1, "requant_boundary: just-over-boundary job must set err_overflow");
  end
endtask

// ---------------------------------------------------------------------------
// TEST 8 : premature_last_truncation (ERR-003 / UQ-004)
// ---------------------------------------------------------------------------
task automatic test_premature_last_truncation();
  gemm_ref_pkg::ref_beat_t wq_trunc[$], aq_full[$];
  gemm_ref_pkg::ref_beat_t wq_full2[$], aq_trunc2[$];
  bit passA, passB;
  int unsigned dpc0;
  begin
    // (a) weight channel truncated early: only 5 of the 8 expected beats,
    // w_last asserted on beat #5 (index 4). Activation channel full (8).
    build_random_queue(wq_trunc, 5);
    build_random_queue(aq_full, N);
    dpc0 = done_pulse_count;
    run_and_check_job(1, 1, 1, 0, 1, wq_trunc, aq_full, 0, 0, 0, "premature_w_last", passA);
    check(passA, "premature_last_truncation: weight-channel truncate-and-proceed must match reference model");
    check(done_pulse_count == dpc0 + 1, "premature_last_truncation: done must still pulse exactly once (weight trunc)");

    // (b) activation channel truncated early: only 6 of the 8 expected
    // beats. Weight channel full (8).
    build_random_queue(wq_full2, N);
    build_random_queue(aq_trunc2, 6);
    dpc0 = done_pulse_count;
    run_and_check_job(1, 1, 1, 0, 1, wq_full2, aq_trunc2, 0, 0, 0, "premature_a_last", passB);
    check(passB, "premature_last_truncation: activation-channel truncate-and-proceed must match reference model");
    check(done_pulse_count == dpc0 + 1, "premature_last_truncation: done must still pulse exactly once (act trunc)");
  end
endtask

// ---------------------------------------------------------------------------
// TEST 9 : backpressure_stress (REQ-GEMM-009, REQ-PROTO-002/003)
// ---------------------------------------------------------------------------
task automatic test_backpressure_stress();
  gemm_ref_pkg::ref_beat_t wq[$], aq[$];
  bit pass;
  begin
    build_random_queue(wq, 2*2*N);
    build_random_queue(aq, 2*2*N);
    // Random 0..3 cycle stalls on w/a producer sides and 0..3 cycle
    // backpressure on the y consumer side. The SVA bind's stability
    // assertions (a_w_stable/a_a_stable/a_y_stable) independently police
    // REQ-PROTO-002 across every stall observed in this run.
    run_and_check_job(2, 2, 2, 2, 1, wq, aq, 3, 3, 3, "backpressure_stress", pass);
    check(pass, "backpressure_stress: randomized stalls on all 4 channels must still be bit-exact");
  end
endtask

// ---------------------------------------------------------------------------
// TEST 10 : constrained_random (REQ-GEMM-004/005/006)
// ---------------------------------------------------------------------------
task automatic test_constrained_random();
  localparam int NUM_ITERS = 20;
  int iter, tm, tk, tn, shift_v, scale_v, tries;
  gemm_ref_pkg::ref_beat_t wq[$], aq[$];
  bit pass;
  int cr_pass, cr_fail;
  begin
    cr_pass = 0;
    cr_fail = 0;
    for (iter = 0; iter < NUM_ITERS; iter++) begin
      tries = 0;
      do begin
        tm = $urandom_range(1, 8);
        tk = $urandom_range(1, 8);
        tn = $urandom_range(1, 8);
        tries++;
      end while (((tk*tn*N > 64) || (tm*tk*N > 64)) && (tries < 100));
      if ((tk*tn*N > 64) || (tm*tk*N > 64)) begin
        tm = 1; tk = 1; tn = 1; // capacity-safe fallback (should not trigger in practice)
      end
      shift_v = $urandom_range(0, 4);
      scale_v = $urandom_range(0, 8) - 4; // -4..+4

      build_random_queue(wq, tk*tn*N);
      build_random_queue(aq, tm*tk*N);

      run_and_check_job(tm, tk, tn, shift_v, scale_v, wq, aq,
                         $urandom_range(0, 2), $urandom_range(0, 2), $urandom_range(0, 2),
                         $sformatf("constrained_random_iter%0d_m%0dk%0dn%0d", iter, tm, tk, tn),
                         pass);
      if (pass) cr_pass++;
      else      cr_fail++;
    end
    $display("CONSTRAINED_RANDOM SUMMARY: iters=%0d pass=%0d fail=%0d", NUM_ITERS, cr_pass, cr_fail);
    check(cr_fail == 0, "constrained_random: all randomized legal jobs must match reference model");
  end
endtask

// ---------------------------------------------------------------------------
// Per-test pass/fail reporting.
// ---------------------------------------------------------------------------
task automatic report_test(input string name, input string reqs, input int unsigned err_before);
  begin
    if (error_count == err_before)
      $display("TEST %s [%s] : PASS", name, reqs);
    else
      $display("TEST %s [%s] : FAIL (new_errors=%0d)", name, reqs, error_count - err_before);
  end
endtask

// ---------------------------------------------------------------------------
// run_all_tests() : single entry point called from gemm_tb_top's initial
// block (after the existing reset-smoke checks).
// ---------------------------------------------------------------------------
task automatic run_all_tests();
  int unsigned err0;
  begin
    fork
      monitor_done_bg();
    join_none

    void'($urandom(RANDOM_SEED));
    $display("TB RANDOM SEED = 0x%08h (%0d)", RANDOM_SEED, RANDOM_SEED);

    err0 = error_count; $display("DBG: starting reset_midjob"); test_reset_midjob();              report_test("reset_midjob", "REQ-GEMM-010,ERR-004", err0);
    err0 = error_count; $display("DBG: starting desc_decode_and_busy"); test_desc_decode_and_busy();       report_test("desc_decode_and_busy", "REQ-GEMM-001,REQ-PROTO-004", err0);
    err0 = error_count; $display("DBG: starting illegal_desc_reject"); test_illegal_desc_reject();        report_test("illegal_desc_reject", "ERR-002,UQ-003", err0);
    err0 = error_count; $display("DBG: starting single_tile_mac"); test_single_tile_mac();             report_test("single_tile_mac", "REQ-GEMM-004,REQ-GEMM-013,REQ-GEMM-008", err0);
    err0 = error_count; $display("DBG: starting multi_tile"); test_multi_tile();                  report_test("multi_tile", "REQ-GEMM-005", err0);
    err0 = error_count; $display("DBG: starting requant_saturation"); test_requant_saturation();          report_test("requant_saturation", "REQ-GEMM-006,REQ-GEMM-007,REQ-GEMM-012,ERR-001", err0);
    err0 = error_count; $display("DBG: starting requant_boundary"); test_requant_boundary();            report_test("requant_boundary", "REQ-GEMM-006,REQ-GEMM-007", err0);
    err0 = error_count; $display("DBG: starting premature_last_truncation"); test_premature_last_truncation();   report_test("premature_last_truncation", "ERR-003,UQ-004", err0);
    err0 = error_count; $display("DBG: starting backpressure_stress"); test_backpressure_stress();         report_test("backpressure_stress", "REQ-GEMM-009,REQ-PROTO-002,REQ-PROTO-003", err0);
    err0 = error_count; $display("DBG: starting constrained_random"); test_constrained_random();          report_test("constrained_random", "REQ-GEMM-004,REQ-GEMM-005,REQ-GEMM-006", err0);
  end
endtask
