// tb_asu_rmsnorm_wide.sv — vector-driven scoreboard TB for the WIDE-D
// asu_rmsnorm elaboration (C-RMSW, docs/design/WIDE_RMSNORM.md §4).
// Structure follows verif/asu/sb/tb_asu_rmsnorm_sb.sv (the house adversarial
// template); expected y/r/nrm come from gen_wide_rms_vectors.py whose oracle
// is the GOLDEN arbiter rmsnorm_fx_wide (golden/apex_golden/attention.py) —
// bit-exact per output beat, framing exact, dbg_norm == nrm.
//
// D_MAX parameterizes the DUT elaboration: 3584 (default, the 7B hidden
// build) or 8192 (-GD_MAX=8192, the sum2 == 2^27 corner build).
//
// Vector records (same grammar as verif/asu/sb):
//   N <d> <flags>          legal row — d x-beats + d gammas, compare d
//                          Q7.8 outputs bit-exact + dbg_norm + framing.
//   B <d>                  illegal-length reject (d <= D_MAX; rejected at
//                          `last`: not pow-2 <= 128, not a multiple of 128)
//   O <d>                  oversize reject (d > D_MAX); d == D_MAX+1 hits
//                          the last-on-overflow-beat arm, larger d FLUSH
//   Z <d> <phase> <abort>  mid-op reset (1=COLLECT 2=FLUSH 3=ISSUE 4=WAIT
//                          5=EMIT 6=DRAIN via hierarchical dut.st targeting;
//                          phase 0 = raw cycle count)
//
// Adversaries: +bp_mode output backpressure none/75%/storm; +stall_mode
// x-feed gaps; +g_mode gamma delivery 0=lockstep 1=late 2=eager. All modes
// must be bit-identical (only timing differs). §5 SVA bound on all three
// boundaries; D-006/§3 asserted here; watchdog.

`include "apex_stream1_sva.svh"

module tb_asu_rmsnorm_wide #(
  parameter int unsigned D_MAX = 3584
);

  import apex_pkg::*;

  localparam int unsigned FEED_MAX    = D_MAX + 72;
  localparam int unsigned ROW_TIMEOUT = 400000;

  // ── clock / cycle counter ──────────────────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  // ── DUT ────────────────────────────────────────────────────────────────────
  logic               s_valid, s_ready;
  logic signed [7:0]  s_x;
  logic               s_last;
  logic               g_valid, g_ready;
  logic signed [15:0] g_gamma;
  logic               m_valid, m_ready;
  logic signed [15:0] m_y;
  logic               m_last;
  logic               busy, done, len_error, len_error_sticky;
  logic        [13:0] dbg_norm;

  asu_rmsnorm #(
    .RMS_D_MAX (D_MAX)
  ) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (s_valid),
    .s_ready          (s_ready),
    .s_x              (s_x),
    .s_last           (s_last),
    .g_valid          (g_valid),
    .g_ready          (g_ready),
    .g_gamma          (g_gamma),
    .m_valid          (m_valid),
    .m_ready          (m_ready),
    .m_y              (m_y),
    .m_last           (m_last),
    .busy             (busy),
    .done             (done),
    .len_error        (len_error),
    .len_error_sticky (len_error_sticky),
    .dbg_norm         (dbg_norm),
    // R4 chunk composition: OFF (both levels 0) — this TB exercises
    // the frozen contract, which R4 leaves bit-identical when off
    .ext_sum_en       (1'b0),
    .ext_r_en         (1'b0),
    .ext_sum2         (28'd0),
    .ext_k            (7'd0),
    .s2_push          (s2_push),
    .s2_val           (s2_val)
  );

  // R4 OFF-mode guard: with both ext levels tied 0 the sum-capture export
  // must never fire — the frozen contract this TB grades is byte-identical
  logic        s2_push;
  logic [27:0] s2_val;
  always @(posedge clk) begin
    if (rst_n && (s2_push !== 1'b0 || s2_val !== 28'd0)) begin
      $display("FAIL: R4 export fired with ext levels off");
      $fatal(1);
    end
  end

  // ── §5 stream SVA on all three boundaries ─────────────────────────────────
  bind asu_rmsnorm apex_stream1_sva #(.WIDTH(9), .NAME("rmsw.x")) u_sva_x (
    .clk   (clk),
    .rst_n (rst_n),
    .valid (s_valid),
    .ready (s_ready),
    .data  ({s_x, s_last})
  );
  bind asu_rmsnorm apex_stream1_sva #(.WIDTH(16), .NAME("rmsw.g")) u_sva_g (
    .clk   (clk),
    .rst_n (rst_n),
    .valid (g_valid),
    .ready (g_ready),
    .data  (g_gamma)
  );
  bind asu_rmsnorm apex_stream1_sva #(.WIDTH(17), .NAME("rmsw.y")) u_sva_y (
    .clk   (clk),
    .rst_n (rst_n),
    .valid (m_valid),
    .ready (m_ready),
    .data  ({m_y, m_last})
  );

  // ── job-level D-006 / §3 assertions ───────────────────────────────────────
  ap_done_pulse_1cyc: assert property (@(posedge clk) disable iff (!rst_n)
    done |=> !done)
    else $error("[SVA D-006] done held more than one cycle");

  ap_done_means_idle: assert property (@(posedge clk) disable iff (!rst_n)
    done |-> !busy)
    else $error("[SVA §5] busy still high at done");

  ap_out_only_in_job: assert property (@(posedge clk) disable iff (!rst_n)
    m_valid |-> busy)
    else $error("[SVA D-019] output beat with no job in flight");

  ap_err_no_done: assert property (@(posedge clk) disable iff (!rst_n)
    len_error |-> !done)
    else $error("[SVA §3] done pulsed for a rejected row");

  ap_sticky_sets: assert property (@(posedge clk) disable iff (!rst_n)
    len_error |=> len_error_sticky)
    else $error("[SVA §3] sticky not set after len_error pulse");

  ap_sticky_holds: assert property (@(posedge clk) disable iff (!rst_n)
    len_error_sticky |=> len_error_sticky)
    else $error("[SVA §3] len_error_sticky cleared without reset");

  // ── plusarg config ─────────────────────────────────────────────────────────
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned g_mode       = 0;
  int unsigned seed         = 32'h35_8402;
  int unsigned watchdog_cyc = 12_000_000;
  initial begin
    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
    void'($value$plusargs("g_mode=%d", g_mode));
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
  end

  // ── watchdog / optional trace ──────────────────────────────────────────────
  initial begin
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: simulation did not finish within %0d cycles",
           watchdog_cyc);
  end

  initial begin
    if ($test$plusargs("dump")) begin
      $dumpfile("dump_rms_wide.fst");
      $dumpvars(0, tb_asu_rmsnorm_wide);
    end
  end

  // ── monitors ───────────────────────────────────────────────────────────────
  int done_count;
  int err_pulses;
  always @(posedge clk) begin
    if (!rst_n) begin
      done_count <= 0;
      err_pulses <= 0;
    end else begin
      if (done)      done_count <= done_count + 1;
      if (len_error) err_pulses <= err_pulses + 1;
    end
  end

  logic [15:0] got_q [$];
  bit          got_last_q [$];
  logic [31:0] lfsr;
  always @(negedge clk) begin
    if (!rst_n) begin
      lfsr    <= seed | 32'h1;
      m_ready <= 1'b0;
    end else begin
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      unique case (bp_mode)
        0:       m_ready <= 1'b1;
        1:       m_ready <= |lfsr[1:0];
        default: m_ready <= (lfsr[3:0] == 4'h0);
      endcase
    end
  end
  always @(posedge clk) begin
    if (rst_n && m_valid && m_ready) begin
      got_q.push_back(16'(m_y));
      got_last_q.push_back(m_last);
    end
  end

  int exp_beats_at_done;
  int d006_fails;
  always @(posedge clk) begin
    if (!rst_n) begin
      d006_fails <= 0;
    end else if (done && (got_q.size() != exp_beats_at_done)) begin
      d006_fails <= d006_fails + 1;
      $display("FAIL [D-006] cyc=%0d: done with %0d/%0d beats accepted post-skid",
               cyc, got_q.size(), exp_beats_at_done);
    end
  end

  // ── TB-side randomization ──────────────────────────────────────────────────
  int unsigned rng_state = 32'hC0FF_3584;
  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction

  task automatic stall_gap();
    int unsigned g;
    unique case (stall_mode)
      0:       g = 0;
      1:       g = ((rnd() & 32'd1) != 0) ? (rnd() % 32'd4) : 0;
      default: g = ((rnd() % 32'd8) == 0) ? (rnd() % 32'd32) : (rnd() % 32'd3);
    endcase
    repeat (g) @(negedge clk);
  endtask

  // ── stimulus memories ──────────────────────────────────────────────────────
  logic [7:0]  x_mem   [FEED_MAX];
  logic [15:0] g_mem   [D_MAX];
  logic [15:0] exp_mem [D_MAX];
  // r (inv_norm) is NOT observable on the DUT interface; it is verified
  // indirectly (every nonzero y lane is a bit-exact function of r) and the
  // floor-sqrt directly via dbg_norm. The Q line's r field is provenance.
  logic [13:0] exp_nrm;

  // ── coverage buckets ───────────────────────────────────────────────────────
  int cov_d_eq_1, cov_d_narrow, cov_d_wide_pow2, cov_d_wide_np2, cov_d_eq_max;
  int cov_x_extreme, cov_all_zero_x, cov_g_zero_lane, cov_g_extreme;
  int cov_tie_odd, cov_tie_even, cov_mu_boundary;
  int cov_rej_bad, cov_rej_over_direct, cov_rej_over_flush;
  int cov_rst_collect, cov_rst_flush, cov_rst_issue, cov_rst_wait;
  int cov_rst_emit, cov_rst_drain, cov_rst_random, cov_rst_idle;

  function automatic void cov_row(input int d, flags);
    if (d == 1)                    cov_d_eq_1++;
    else if (d <= 128)             cov_d_narrow++;
    else if (d == int'(D_MAX))     cov_d_eq_max++;
    else if ((d & (d - 1)) == 0)   cov_d_wide_pow2++;
    else                           cov_d_wide_np2++;
    if ((flags & 1)  != 0)  cov_tie_odd++;
    if ((flags & 2)  != 0)  cov_tie_even++;
    if ((flags & 4)  != 0)  cov_x_extreme++;
    if ((flags & 8)  != 0)  cov_all_zero_x++;
    if ((flags & 16) != 0)  cov_g_zero_lane++;
    if ((flags & 32) != 0)  cov_g_extreme++;
    if ((flags & 64) != 0)  cov_mu_boundary++;
  endfunction

  function automatic void cov_reset_phase(input int st, input bit targeted);
    if (!targeted) begin
      cov_rst_random++;
      return;
    end
    unique case (st)
      1:       cov_rst_collect++;
      2:       cov_rst_flush++;
      3:       cov_rst_issue++;
      4:       cov_rst_wait++;
      5:       cov_rst_emit++;
      6:       cov_rst_drain++;
      default: cov_rst_idle++;
    endcase
  endfunction

  function automatic void print_coverage();
    $display("COV rmsw_d_eq_1 %0d",        cov_d_eq_1);
    $display("COV rmsw_d_narrow %0d",      cov_d_narrow);
    $display("COV rmsw_d_wide_pow2 %0d",   cov_d_wide_pow2);
    $display("COV rmsw_d_wide_np2 %0d",    cov_d_wide_np2);
    $display("COV rmsw_d_eq_max %0d",      cov_d_eq_max);
    $display("COV rmsw_x_extreme %0d",     cov_x_extreme);
    $display("COV rmsw_all_zero_x %0d",    cov_all_zero_x);
    $display("COV rmsw_g_zero_lane %0d",   cov_g_zero_lane);
    $display("COV rmsw_g_extreme %0d",     cov_g_extreme);
    $display("COV rmsw_tie_odd %0d",       cov_tie_odd);
    $display("COV rmsw_tie_even %0d",      cov_tie_even);
    $display("COV rmsw_mu_boundary %0d",   cov_mu_boundary);
    $display("COV rmsw_rej_bad %0d",       cov_rej_bad);
    $display("COV rmsw_rej_over_direct %0d", cov_rej_over_direct);
    $display("COV rmsw_rej_over_flush %0d",  cov_rej_over_flush);
    $display("COV rmsw_rst_collect %0d",   cov_rst_collect);
    $display("COV rmsw_rst_flush %0d",     cov_rst_flush);
    $display("COV rmsw_rst_issue %0d",     cov_rst_issue);
    $display("COV rmsw_rst_wait %0d",      cov_rst_wait);
    $display("COV rmsw_rst_emit %0d",      cov_rst_emit);
    $display("COV rmsw_rst_drain %0d",     cov_rst_drain);
    $display("COV rmsw_rst_random %0d",    cov_rst_random);
    $display("COV rmsw_rst_idle %0d",      cov_rst_idle);
  endfunction

  int errors;
  bit exp_sticky;

  // ── drivers ────────────────────────────────────────────────────────────────
  task automatic send_x(input logic [7:0] w, input bit last);
    @(negedge clk);
    s_x     = signed'(w);
    s_last  = last;
    s_valid = 1'b1;
    while (!s_ready) @(negedge clk);
    @(negedge clk);
    s_valid = 1'b0;
  endtask

  task automatic send_gamma(input logic [15:0] w);
    @(negedge clk);
    g_gamma = signed'(w);
    g_valid = 1'b1;
    while (!g_ready) @(negedge clk);
    @(negedge clk);
    g_valid = 1'b0;
  endtask

  task automatic feed_x(input int d);
    for (int i = 0; i < d; i++) begin
      stall_gap();
      send_x(x_mem[i], i == d - 1);
    end
  endtask

  task automatic feed_gamma(input int d);
    for (int i = 0; i < d; i++) begin
      unique case (g_mode)
        0:       ;                                       // lockstep: no gap
        1:       repeat (rnd() % 32'd21) @(negedge clk); // late
        default: ;                                       // eager: pump ASAP
      endcase
      send_gamma(g_mem[i]);
    end
  endtask

  task automatic wait_done_rel(input int d0, input string name);
    int t;
    t = 0;
    while (done_count == d0) begin
      @(negedge clk);
      t++;
      if (t > int'(ROW_TIMEOUT))
        $fatal(1, "FAIL [%s]: row timeout after %0d cycles (cyc=%0d)",
               name, t, cyc);
    end
  endtask

  // ── vector-file reader ─────────────────────────────────────────────────────
  int    fd;
  string cur_line;
  int    n_rows, n_rejects, n_resets;

  function automatic bit next_line();
    while ($fgets(cur_line, fd) > 0) begin
      if (cur_line.len() > 1 && cur_line.getc(0) != "#") return 1'b1;
    end
    return 1'b0;
  endfunction

  task automatic read_payload(input int d);
    string tg;
    logic [31:0] w;
    logic [31:0] r_i, nrm_i;
    logic unused_hi;
    int cnt;
    for (int i = 0; i < d; i++) begin
      if (!next_line()) $fatal(1, "EOF inside D data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "D") $fatal(1, "bad D line: %s", cur_line);
      x_mem[i] = w[7:0];
    end
    for (int i = 0; i < d; i++) begin
      if (!next_line()) $fatal(1, "EOF inside G data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "G") $fatal(1, "bad G line: %s", cur_line);
      g_mem[i] = w[15:0];
    end
    for (int i = 0; i < d; i++) begin
      if (!next_line()) $fatal(1, "EOF inside Y data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "Y") $fatal(1, "bad Y line: %s", cur_line);
      exp_mem[i] = w[15:0];
    end
    if (!next_line()) $fatal(1, "EOF before Q line");
    cnt = $sscanf(cur_line, "%s %d %d", tg, r_i, nrm_i);
    if (cnt != 3 || tg != "Q") $fatal(1, "bad Q line: %s", cur_line);
    exp_nrm   = nrm_i[13:0];
    unused_hi = &{1'b0, w[31:16], r_i, nrm_i[31:14]};
  endtask

  // ── record handlers ────────────────────────────────────────────────────────
  task automatic do_row(input string hdr);
    int d, flags, d0, cnt;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d %d", unused_tag, d, flags);
    if (cnt != 3) $fatal(1, "bad N line: %s", hdr);
    read_payload(d);
    cov_row(d, flags);
    n_rows++;
    name = $sformatf("N%0d d=%0d", n_rows, d);

    d0 = done_count;
    exp_beats_at_done = d;
    fork
      feed_x(d);
      feed_gamma(d);
    join
    wait_done_rel(d0, name);
    if (done_count != d0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done_count=%0d (exp %0d)",
               name, cyc, done_count, d0 + 1);
    end
    if (got_q.size() != d) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: expected %0d beats, got %0d",
               name, cyc, d, got_q.size());
    end else begin
      for (int i = 0; i < d; i++) begin
        if (got_q[i] !== exp_mem[i]) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: y[%0d] got 0x%04x exp 0x%04x",
                   name, cyc, i, got_q[i], exp_mem[i]);
        end
        if (got_last_q[i] !== bit'(i == d - 1)) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d last=%0b (exp %0b)",
                   name, cyc, i, got_last_q[i], i == d - 1);
        end
      end
    end
    if (dbg_norm !== exp_nrm) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: dbg_norm=%0d exp %0d (rsqrt floor-sqrt)",
               name, cyc, dbg_norm, exp_nrm);
    end
    got_q.delete();
    got_last_q.delete();
    repeat (2) @(negedge clk);
    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy stuck after done", name, cyc);
    end
    if (len_error_sticky !== exp_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky=%0b expected %0b",
               name, cyc, len_error_sticky, exp_sticky);
    end
  endtask

  task automatic do_reject(input string hdr, input bit oversize);
    int d, e0, d0, cnt;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d", unused_tag, d);
    if (cnt != 2) $fatal(1, "bad B/O line: %s", hdr);
    n_rejects++;
    name = $sformatf("%s%0d d=%0d", oversize ? "O" : "B", n_rejects, d);
    if (!oversize)                  cov_rej_bad++;
    else if (d == int'(D_MAX) + 1)  cov_rej_over_direct++;
    else                            cov_rej_over_flush++;

    e0 = err_pulses;
    d0 = done_count;
    exp_beats_at_done = 0;
    for (int i = 0; i < d; i++) begin
      stall_gap();
      send_x(8'(i * 7 + 3), i == d - 1);
    end
    exp_sticky = 1'b1;
    repeat (5) @(negedge clk);
    if (err_pulses != e0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: err_pulses=%0d (exp %0d)",
               name, cyc, err_pulses, e0 + 1);
    end
    if (!len_error_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky bit not set", name, cyc);
    end
    if (done_count != d0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done pulsed for rejected row", name, cyc);
    end
    if (got_q.size() != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: %0d output beat(s) from rejected row",
               name, cyc, got_q.size());
    end
    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy stuck after reject", name, cyc);
    end
  endtask

  task automatic do_reset_job(input string hdr);
    int d, tphase, abortcyc, cnt, phase, t;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d %d %d", unused_tag, d, tphase, abortcyc);
    if (cnt != 4) $fatal(1, "bad Z line: %s", hdr);
    n_resets++;
    name = $sformatf("Z%0d d=%0d phase=%0d abort=%0d",
                     n_resets, d, tphase, abortcyc);

    fork
      begin
        fork
          begin
            for (int i = 0; i < d; i++) begin
              stall_gap();
              send_x(8'(i * 5 + 1), i == d - 1);
            end
          end
          begin
            for (int i = 0; i < d && i < int'(D_MAX); i++)
              send_gamma(16'(i * 257 + 5));
          end
        join
        forever @(negedge clk);   // hold branch open until abort fires
      end
      begin
        if (tphase != 0) begin
          t = 0;
          while (int'(dut.st) != tphase) begin
            @(negedge clk);
            t++;
            if (t > int'(ROW_TIMEOUT))
              $fatal(1, "FAIL [%s]: FSM never reached phase %0d (cyc=%0d)",
                     name, tphase, cyc);
          end
        end else begin
          repeat (abortcyc) @(negedge clk);
        end
      end
    join_any
    disable fork;

    phase = int'(dut.st);
    cov_reset_phase(phase, tphase != 0);
    $display("RESET [%s] cyc=%0d: mid-op reset in FSM phase %0d",
             name, cyc, phase);

    rst_n   = 1'b0;
    s_valid = 1'b0;
    s_last  = 1'b0;
    g_valid = 1'b0;
    repeat (3) @(negedge clk);
    rst_n      = 1'b1;
    exp_sticky = 1'b0;
    got_q.delete();
    got_last_q.delete();
    repeat (3) @(negedge clk);

    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy after mid-op reset", name, cyc);
    end
    if (len_error_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky not cleared by reset", name, cyc);
    end
    repeat (10) @(negedge clk);
    if (got_q.size() != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: %0d stray beat(s) after mid-op reset",
               name, cyc, got_q.size());
    end
    if (done_count != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: stray done after mid-op reset", name, cyc);
    end
  endtask

  // ── main sequence ──────────────────────────────────────────────────────────
  initial begin : main
    string vec_path;
    byte   tag0;

    rst_n      = 1'b0;
    errors     = 0;
    exp_sticky = 1'b0;
    s_valid    = 1'b0;
    s_x        = '0;
    s_last     = 1'b0;
    g_valid    = 1'b0;
    g_gamma    = '0;
    n_rows     = 0;
    n_rejects  = 0;
    n_resets   = 0;
    exp_beats_at_done = 0;

    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'hDEAD_3584;

    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);
    if (busy) begin
      errors++;
      $display("FAIL [reset] cyc=%0d: busy asserted out of reset", cyc);
    end
    if (len_error_sticky) begin
      errors++;
      $display("FAIL [reset] cyc=%0d: sticky error out of reset", cyc);
    end

    while (next_line()) begin
      tag0 = cur_line.getc(0);
      unique case (tag0)
        "N":     do_row(cur_line);
        "B":     do_reject(cur_line, 1'b0);
        "O":     do_reject(cur_line, 1'b1);
        "Z":     do_reset_job(cur_line);
        default: $fatal(1, "unexpected record: %s", cur_line);
      endcase
    end
    $fclose(fd);

    repeat (5) @(negedge clk);
    errors += d006_fails;
    print_coverage();
    if (errors == 0) begin
      $display("TB PASS: %0d rows, %0d rejects, %0d mid-op resets, 0 errors (D_MAX=%0d bp=%0d stall=%0d g=%0d seed=%0d)",
               n_rows, n_rejects, n_resets, D_MAX, bp_mode, stall_mode,
               g_mode, seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s) across %0d rows", errors, n_rows);
    end
  end

endmodule
