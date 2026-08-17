// tb_mxe_perf.sv — D-005 load-under-compute PERF evidence TB.
//
// D-005's claim has two halves: (a) double-buffered weight banks + wide load
// port EXIST (functionally proven by the sb suite), and (b) they are USED to
// hide weight loading under compute — the run2 "7x load-bound" fix. Half (b)
// is what this TB measures and asserts:
//
//   - drives multi-chunk WS jobs (K=64/M=16 and K=2048/M=64 per the D-005
//     closure task, plus weight-bandwidth-limited variants with a stream
//     gap) at full result rate, checks every result beat bit-exact;
//   - measures per-job total cycles (descriptor accept -> done);
//   - measures overlap_cycles: weight-load column strobes issued while the
//     ctrl FSM is in S_INGEST/S_COMPUTE/S_FLUSH (i.e. NOT in the dedicated
//     wait-for-weights state) — the direct witness of load-under-compute;
//   - with +require_overlap=1, FAILS any multi-chunk job whose
//     overlap_cycles == 0. Run against the pinned sequential controller
//     (baseline/mxe_ctrl_seq.sv) this is a PERMANENT kept-failing
//     demonstration that the sequential scheduler lacks the property.
//
// The perf GATE itself (new total cycles < sequential-baseline total cycles,
// speedups quoted) is perf_compare.py over this TB's PERF lines from the two
// builds (obj_new = rtl/mxe/mxe_ctrl.sv, obj_base = baseline copy).
//
// The §5/D-006 apex_stream_sva pack stays bound in both builds — the perf
// claim is only meaningful if the job/stream contract still holds.

`include "apex_stream_sva.svh"

module tb_mxe_perf;

  import apex_pkg::*;

  localparam int MAX_ACT     = 16384;   // M_TILE_MAX * KB_MAX
  localparam int MAX_WGT     = 2048;    // KB_MAX * MXE_N
  localparam int MAX_M       = 64;
  localparam int JOB_TIMEOUT = 300000;

  // ── clock / cycle counter ──────────────────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    repeat (2_000_000) @(posedge clk);
    $fatal(1, "WATCHDOG: perf TB did not finish");
  end

  // ── DUT ────────────────────────────────────────────────────────────────────
  logic         desc_valid, desc_ready;
  mxe_desc_t    desc;
  logic         desc_error, desc_error_sticky, busy, done;
  logic         act_valid, act_ready;
  lane8_beat_t  act_beat;
  logic         wgt_valid, wgt_ready;
  lane8_beat_t  wgt_beat;
  logic         res_valid, res_ready;
  lane32_beat_t res_beat;

  mxe_top dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .desc_valid        (desc_valid),
    .desc_ready        (desc_ready),
    .desc              (desc),
    .desc_error        (desc_error),
    .desc_error_sticky (desc_error_sticky),
    .busy              (busy),
    .done              (done),
    .act_valid         (act_valid),
    .act_ready         (act_ready),
    .act_beat          (act_beat),
    .wgt_valid         (wgt_valid),
    .wgt_ready         (wgt_ready),
    .wgt_beat          (wgt_beat),
    .res_valid         (res_valid),
    .res_ready         (res_ready),
    .res_beat          (res_beat)
  );

  bind mxe_top apex_stream_sva u_apex_sva (
    .clk               (clk),
    .rst_n             (rst_n),
    .desc_valid        (desc_valid),
    .desc_ready        (desc_ready),
    .desc              (desc),
    .desc_error        (desc_error),
    .desc_error_sticky (desc_error_sticky),
    .busy              (busy),
    .done              (done),
    .act_valid         (act_valid),
    .act_ready         (act_ready),
    .act_beat          (act_beat),
    .wgt_valid         (wgt_valid),
    .wgt_ready         (wgt_ready),
    .wgt_beat          (wgt_beat),
    .res_valid         (res_valid),
    .res_ready         (res_ready),
    .res_beat          (res_beat)
  );

  // error/status outputs observed by the bound SVA pack, not by this TB
  logic unused_status;
  assign unused_status = &{1'b0, desc_error, desc_error_sticky, busy};

  // ── plusargs ───────────────────────────────────────────────────────────────
  int unsigned require_overlap = 0;
  initial begin
    void'($value$plusargs("require_overlap=%d", require_overlap));
  end

  // ── monitors ───────────────────────────────────────────────────────────────
  int done_count;
  int unsigned t_accept;
  always @(posedge clk) begin
    if (!rst_n) begin
      done_count <= 0;
    end else begin
      if (desc_valid && desc_ready) t_accept <= cyc;
      if (done)                     done_count <= done_count + 1;
    end
  end

  // D-005 overlap witness: weight-load strobes issued while the ctrl is in
  // S_INGEST(1) / S_COMPUTE(3) / S_FLUSH(4) — i.e. anywhere but the
  // dedicated weight-wait state S_WLOAD(2). The sequential controller only
  // ever strobes in S_WLOAD, so it measures exactly 0 here.
  int overlap_cnt;
  always @(posedge clk) begin
    if (!rst_n) begin
      overlap_cnt <= 0;
    end else if (dut.u_ctrl.arr_wload_en
                 && (int'(dut.u_ctrl.state) inside {1, 3, 4})) begin
      overlap_cnt <= overlap_cnt + 1;
    end
  end

  // full-rate result consumer
  logic [255:0] got_q [$];
  bit           got_last_q [$];
  always @(negedge clk) res_ready <= rst_n;
  always @(posedge clk) begin
    if (rst_n && res_valid && res_ready) begin
      got_q.push_back(res_beat.data);
      got_last_q.push_back(res_beat.last);
    end
  end

  // ── stimulus / expected storage ───────────────────────────────────────────
  logic [63:0]  act_mem [MAX_ACT];
  logic [63:0]  wgt_mem [MAX_WGT];
  logic [255:0] exp_mem [MAX_M];

  int errors;

  // ── drivers (negedge discipline, sb lineage) ──────────────────────────────
  task automatic send_desc(input mxe_desc_t d);
    @(negedge clk);
    desc       = d;
    desc_valid = 1'b1;
    while (!desc_ready) @(negedge clk);
    @(negedge clk);
    desc_valid = 1'b0;
  endtask

  task automatic send_act(input logic [63:0] w, input bit last);
    @(negedge clk);
    act_beat.data = w;
    act_beat.last = last;
    act_valid     = 1'b1;
    while (!act_ready) @(negedge clk);
    @(negedge clk);
    act_valid = 1'b0;
  endtask

  task automatic send_wgt(input logic [63:0] w, input bit last);
    @(negedge clk);
    wgt_beat.data = w;
    wgt_beat.last = last;
    wgt_valid     = 1'b1;
    while (!wgt_ready) @(negedge clk);
    @(negedge clk);
    wgt_valid = 1'b0;
  endtask

  task automatic wait_done_rel(input int d0, input string name);
    int t;
    t = 0;
    while (done_count == d0) begin
      @(negedge clk);
      t++;
      if (t > JOB_TIMEOUT)
        $fatal(1, "PERF SUITE FAIL [%s]: job timeout (cyc=%0d)", name, cyc);
    end
  endtask

  // ── deterministic job content + golden-in-TB expected (raw INT32 drain) ──
  function automatic int a_val(input int mm, kk);
    return ((mm * 31 + kk * 7) % 255) - 127;    // [-127, 127]
  endfunction
  function automatic int b_val(input int kk, c);
    return ((kk * 13 + c * 5) % 255) - 127;     // [-127, 127]
  endfunction

  task automatic run_job(input string name, input int m, k, n, wgap);
    int kb, na, nw, d0, ov0;
    int unsigned t_done;
    int acc;
    int idx;
    kb = (k + 7) / 8;
    na = m * kb;
    nw = kb * n;

    // pack activation beats (row-major, chunk-minor; lanes >= k zeroed)
    idx = 0;
    for (int mm = 0; mm < m; mm++)
      for (int p = 0; p < kb; p++) begin
        logic [63:0] w;
        w = '0;
        for (int j = 0; j < 8; j++)
          if (8 * p + j < k) w[8*j +: 8] = 8'(a_val(mm, 8 * p + j));
        act_mem[idx] = w;
        idx++;
      end
    // pack weight beats (chunk-major, ascending column; rows >= k zeroed)
    idx = 0;
    for (int p = 0; p < kb; p++)
      for (int c = 0; c < n; c++) begin
        logic [63:0] w;
        w = '0;
        for (int r = 0; r < 8; r++)
          if (8 * p + r < k) w[8*r +: 8] = 8'(b_val(8 * p + r, c));
        wgt_mem[idx] = w;
        idx++;
      end
    // expected raw INT32 beats (C-3: exact in int32 for K <= 2048)
    for (int mm = 0; mm < m; mm++) begin
      exp_mem[mm] = '0;
      for (int c = 0; c < n; c++) begin
        acc = 0;
        for (int kk = 0; kk < k; kk++) acc += a_val(mm, kk) * b_val(kk, c);
        exp_mem[mm][32*c +: 32] = acc;
      end
    end

    d0  = done_count;
    ov0 = overlap_cnt;
    begin
      mxe_desc_t d;
      d            = '0;
      d.opcode     = 8'(OP_GEMM_WS);
      d.m_dim      = 12'(m);
      d.k_dim      = 12'(k);
      d.n_dim      = 12'(n);
      send_desc(d);
    end
    fork
      for (int i = 0; i < na; i++) send_act(act_mem[i], i == na - 1);
      for (int i = 0; i < nw; i++) begin
        repeat (wgap) @(negedge clk);
        send_wgt(wgt_mem[i], i == nw - 1);
      end
    join
    wait_done_rel(d0, name);
    t_done = cyc;

    // bit-exact result check (perf numbers mean nothing on wrong data)
    if (got_q.size() != m) begin
      errors++;
      $display("PERF CHECK FAIL [%s] cyc=%0d: %0d result beats (exp %0d)",
               name, cyc, got_q.size(), m);
    end else begin
      for (int i = 0; i < m; i++) begin
        if (got_q[i] !== exp_mem[i]) begin
          errors++;
          $display("PERF CHECK FAIL [%s] beat %0d\n  got %064x\n  exp %064x",
                   name, i, got_q[i], exp_mem[i]);
        end
        if (got_last_q[i] !== bit'(i == m - 1)) begin
          errors++;
          $display("PERF CHECK FAIL [%s] beat %0d last=%0b", name, i,
                   got_last_q[i]);
        end
      end
    end
    got_q.delete();
    got_last_q.delete();

    $display("PERF job=%s m=%0d k=%0d n=%0d wgap=%0d cycles=%0d overlap=%0d",
             name, m, k, n, wgap, t_done - t_accept, overlap_cnt - ov0);

    if (require_overlap != 0 && kb > 1 && (overlap_cnt - ov0) == 0) begin
      errors++;
      $display("PERF OVERLAP MISSING [%s]: multi-chunk WS job ran with ZERO load-under-compute overlap (D-005 half (b) absent)",
               name);
    end
  endtask

  // ── main ───────────────────────────────────────────────────────────────────
  initial begin : main
    rst_n      = 1'b0;
    errors     = 0;
    desc_valid = 1'b0;
    desc       = '0;
    act_valid  = 1'b0;
    act_beat   = '0;
    wgt_valid  = 1'b0;
    wgt_beat   = '0;

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (3) @(negedge clk);

    // the two D-005 closure jobs (full-rate weight stream) ...
    run_job("K64_M16",         16,   64, 8, 0);
    run_job("K2048_M64",       64, 2048, 8, 0);
    // ... and the run2-pathology regime: weight-bandwidth-limited feeds
    // (1 beat / 4 cycles) where hiding the load is the whole point
    run_job("K64_M16_wgap3",   16,   64, 8, 3);
    run_job("K2048_M64_wgap3", 64, 2048, 8, 3);

    repeat (5) @(negedge clk);
    if (errors == 0) begin
      $display("PERF TB PASS: 4 jobs bit-exact%s",
               require_overlap != 0 ? ", overlap>0 on every multi-chunk job (D-005)" : "");
      $finish;
    end else begin
      $fatal(1, "PERF SUITE FAIL: %0d error(s)", errors);
    end
  end

endmodule
