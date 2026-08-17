// tb_asu_softmax_sb.sv — INDEPENDENT vector-driven scoreboard TB for
// asu_softmax (house pattern: verif/mxe/sb). Stimulus + bit-exact expected
// probabilities come from gen_asu_sb_vectors.py (golden arbiter
// apex_golden.compute.online_softmax_fx); the TB only drives, collects,
// compares. Verilator 5.x --binary --timing --assert.
//
// Vector records:
//   R <n> <nresc> <flags>  legal row — drive n scores (last on beat n-1),
//                          wait done, compare n probabilities bit-exact +
//                          last framing + busy/done/sticky discipline.
//   O <n>                  oversize row (n > SM_ROW_MAX) — exactly one
//                          row_error pulse + sticky, ZERO output beats, NO
//                          done, busy falls; n == SM_ROW_MAX+1 hits the
//                          last-on-overflow-beat path, larger n the FLUSH arm.
//   X <n> <phase> <abort>  mid-op reset: phase!=0 resets the moment dut.st
//                          equals the target (1=COLLECT 2=FLUSH 3=LOAD 4=DIV
//                          5=PUSH 6=DRAIN — hierarchical targeting, house
//                          style), phase==0 resets after <abort> cycles.
//                          Clean-recovery checks + the following R row proves
//                          state integrity.
//
// Adversaries (plusargs): +bp_mode=0|1|2 output backpressure none/75%/storm;
// +stall_mode=0|1|2 input feed gaps none/short/bursty; +seed=<n>.
// §5 stream stability: apex_stream1_sva bound on both boundaries. Job-level
// D-006/§3 contract asserted here. $fatal watchdog bounds the run.
//
// TB_SCORE_FRAC (default 0 = C-5 raw INT32) is overridden to 10 in the f10
// run so LUT interpolation is exercised INSIDE the softmax datapath (with
// SCORE_FRAC=0 every exp argument is a multiple of 2^10 and frac is 0).

`include "apex_stream1_sva.svh"

module tb_asu_softmax_sb #(
  parameter int unsigned TB_SCORE_FRAC = 0
);

  import apex_pkg::*;

  localparam int unsigned ROW_MAX  = 1024;
  localparam int unsigned FEED_MAX = 1200;      // O/X records exceed ROW_MAX
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
  logic signed [31:0] s_score;
  logic               s_last;
  logic               m_valid, m_ready;
  logic        [15:0] m_prob;
  logic               m_last;
  logic               busy, done, row_error, row_error_sticky;

  asu_softmax #(
    .SM_ROW_MAX (ROW_MAX),
    .SCORE_FRAC (TB_SCORE_FRAC)
  ) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (s_valid),
    .s_ready          (s_ready),
    .s_score          (s_score),
    .s_last           (s_last),
    .m_valid          (m_valid),
    .m_ready          (m_ready),
    .m_prob           (m_prob),
    .m_last           (m_last),
    .busy             (busy),
    .done             (done),
    .row_error        (row_error),
    .row_error_sticky (row_error_sticky)
  );

  // ── §5 stream SVA (reused pack) on both DUT boundaries ────────────────────
  bind asu_softmax apex_stream1_sva #(.WIDTH(33), .NAME("sm.s")) u_sva_s (
    .clk   (clk),
    .rst_n (rst_n),
    .valid (s_valid),
    .ready (s_ready),
    .data  ({s_score, s_last})
  );
  bind asu_softmax apex_stream1_sva #(.WIDTH(17), .NAME("sm.m")) u_sva_m (
    .clk   (clk),
    .rst_n (rst_n),
    .valid (m_valid),
    .ready (m_ready),
    .data  ({m_prob, m_last})
  );

  // ── job-level D-006 / §3 / range assertions ───────────────────────────────
  ap_done_pulse_1cyc: assert property (@(posedge clk) disable iff (!rst_n)
    done |=> !done)
    else $error("[SVA D-006] done held more than one cycle");

  ap_done_means_idle: assert property (@(posedge clk) disable iff (!rst_n)
    done |-> !busy)
    else $error("[SVA §5] busy still high at done (beats pending after done?)");

  ap_out_only_in_job: assert property (@(posedge clk) disable iff (!rst_n)
    m_valid |-> busy)
    else $error("[SVA D-019] output beat with no job in flight (stale beat)");

  ap_err_no_done: assert property (@(posedge clk) disable iff (!rst_n)
    row_error |-> !done)
    else $error("[SVA §3] done pulsed for a rejected row");

  ap_sticky_sets: assert property (@(posedge clk) disable iff (!rst_n)
    row_error |=> row_error_sticky)
    else $error("[SVA §3] sticky not set after row_error pulse");

  ap_sticky_holds: assert property (@(posedge clk) disable iff (!rst_n)
    row_error_sticky |=> row_error_sticky)
    else $error("[SVA §3] row_error_sticky cleared without reset");

  ap_prob_range: assert property (@(posedge clk) disable iff (!rst_n)
    m_valid |-> (m_prob <= 16'd32768))
    else $error("[SVA §6] probability above UQ1.15 1.0 (%0d)", m_prob);

  // ── plusarg config ─────────────────────────────────────────────────────────
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned seed         = 32'h5B_2026;
  int unsigned watchdog_cyc = 8_000_000;
  initial begin
    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
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
      $dumpfile("dump_sm.fst");
      $dumpvars(0, tb_asu_softmax_sb);
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
      if (row_error) err_pulses <= err_pulses + 1;
    end
  end

  // output consumer with configurable backpressure
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
        0:       m_ready <= 1'b1;                 // no backpressure
        1:       m_ready <= |lfsr[1:0];           // ~75% duty
        default: m_ready <= (lfsr[3:0] == 4'h0);  // storm: ~6% duty
      endcase
    end
  end
  always @(posedge clk) begin
    if (rst_n && m_valid && m_ready) begin
      got_q.push_back(m_prob);
      got_last_q.push_back(m_last);
    end
  end

  // D-006: at the done pulse every beat must ALREADY be accepted post-skid
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

  // ── TB-side randomization (feed gaps) ──────────────────────────────────────
  int unsigned rng_state = 32'hC0FFEE01;
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

  // ── stimulus memories (declared before the coverage fns that read them) ──
  logic [31:0] in_mem  [FEED_MAX];
  logic [15:0] exp_mem [ROW_MAX];

  // ── coverage buckets (manual, house pattern) ──────────────────────────────
  int cov_n_eq_1, cov_n_eq_2, cov_n_mid, cov_n_eq_max;
  int cov_p_zero, cov_p_one, cov_p_mid;
  int cov_dup_max, cov_resc_ge2, cov_all_equal;
  int cov_clamp, cov_clamp_boundary, cov_i32_extreme, cov_neg_scores;
  int cov_lut_frac, cov_cfg_frac0, cov_cfg_frac10;
  int cov_rej_direct, cov_rej_flush;
  int cov_rst_collect, cov_rst_flush, cov_rst_load, cov_rst_div;
  int cov_rst_push, cov_rst_drain, cov_rst_random, cov_rst_idle;
  int cov_bp_full, cov_bp_random, cov_bp_storm;
  int cov_stall_none, cov_stall_random, cov_stall_storm;

  function automatic void cov_row(input int n, nresc, flags);
    if (n == 1)             cov_n_eq_1++;
    else if (n == 2)        cov_n_eq_2++;
    else if (n == int'(ROW_MAX)) cov_n_eq_max++;
    else                    cov_n_mid++;
    if ((flags & 1)  != 0)  cov_dup_max++;
    if ((flags & 2)  != 0)  cov_clamp++;
    if ((flags & 4)  != 0)  cov_all_equal++;
    if ((flags & 8)  != 0)  cov_clamp_boundary++;
    if ((flags & 16) != 0)  cov_i32_extreme++;
    if ((flags & 32) != 0)  cov_neg_scores++;
    if ((flags & 64) != 0)  cov_lut_frac++;
    if (nresc >= 2)         cov_resc_ge2++;
  endfunction

  function automatic void cov_probs(input int n);
    for (int i = 0; i < n; i++) begin
      if (exp_mem[i] == 16'd0)          cov_p_zero++;
      else if (exp_mem[i] == 16'h8000)  cov_p_one++;
      else                              cov_p_mid++;
    end
  endfunction

  function automatic void cov_reset_phase(input int st, input bit targeted);
    if (!targeted) begin
      cov_rst_random++;
      return;
    end
    unique case (st)
      1:       cov_rst_collect++;
      2:       cov_rst_flush++;
      3:       cov_rst_load++;
      4:       cov_rst_div++;
      5:       cov_rst_push++;
      6:       cov_rst_drain++;
      default: cov_rst_idle++;
    endcase
  endfunction

  function automatic void print_coverage();
    string pfx;
    pfx = "sm";   // one namespace; cfg_frac0/cfg_frac10 buckets split configs
    $display("COV %s_n_eq_1 %0d",         pfx, cov_n_eq_1);
    $display("COV %s_n_eq_2 %0d",         pfx, cov_n_eq_2);
    $display("COV %s_n_mid %0d",          pfx, cov_n_mid);
    $display("COV %s_n_eq_max %0d",       pfx, cov_n_eq_max);
    $display("COV %s_p_zero %0d",         pfx, cov_p_zero);
    $display("COV %s_p_one %0d",          pfx, cov_p_one);
    $display("COV %s_p_mid %0d",          pfx, cov_p_mid);
    $display("COV %s_dup_max %0d",        pfx, cov_dup_max);
    $display("COV %s_resc_ge2 %0d",       pfx, cov_resc_ge2);
    $display("COV %s_all_equal %0d",      pfx, cov_all_equal);
    $display("COV %s_clamp %0d",          pfx, cov_clamp);
    $display("COV %s_clamp_boundary %0d", pfx, cov_clamp_boundary);
    $display("COV %s_i32_extreme %0d",    pfx, cov_i32_extreme);
    $display("COV %s_neg_scores %0d",     pfx, cov_neg_scores);
    $display("COV %s_lut_frac %0d",       pfx, cov_lut_frac);
    $display("COV %s_cfg_frac0 %0d",      pfx, cov_cfg_frac0);
    $display("COV %s_cfg_frac10 %0d",     pfx, cov_cfg_frac10);
    $display("COV %s_rej_direct %0d",     pfx, cov_rej_direct);
    $display("COV %s_rej_flush %0d",      pfx, cov_rej_flush);
    $display("COV %s_rst_collect %0d",    pfx, cov_rst_collect);
    $display("COV %s_rst_flush %0d",      pfx, cov_rst_flush);
    $display("COV %s_rst_load %0d",       pfx, cov_rst_load);
    $display("COV %s_rst_div %0d",        pfx, cov_rst_div);
    $display("COV %s_rst_push %0d",       pfx, cov_rst_push);
    $display("COV %s_rst_drain %0d",      pfx, cov_rst_drain);
    $display("COV %s_rst_random %0d",     pfx, cov_rst_random);
    $display("COV %s_rst_idle %0d",       pfx, cov_rst_idle);
    $display("COV %s_bp_full %0d",        pfx, cov_bp_full);
    $display("COV %s_bp_random %0d",      pfx, cov_bp_random);
    $display("COV %s_bp_storm %0d",       pfx, cov_bp_storm);
    $display("COV %s_stall_none %0d",     pfx, cov_stall_none);
    $display("COV %s_stall_random %0d",   pfx, cov_stall_random);
    $display("COV %s_stall_storm %0d",    pfx, cov_stall_storm);
  endfunction

  int errors;
  bit exp_sticky;   // TB model of row_error_sticky (cleared only by reset)

  task automatic send_score(input logic [31:0] w, input bit last);
    @(negedge clk);
    s_score = signed'(w);
    s_last  = last;
    s_valid = 1'b1;
    while (!s_ready) @(negedge clk);
    @(negedge clk);                       // cross the committing posedge
    s_valid = 1'b0;
  endtask

  task automatic feed_row(input int n);
    for (int i = 0; i < n; i++) begin
      stall_gap();
      send_score(in_mem[i], i == n - 1);
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

  task automatic read_payload(input int n);
    string tg;
    logic [31:0] w;
    int cnt;
    for (int i = 0; i < n; i++) begin
      if (!next_line()) $fatal(1, "EOF inside S data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "S") $fatal(1, "bad S line: %s", cur_line);
      in_mem[i] = w;
    end
    for (int i = 0; i < n; i++) begin
      if (!next_line()) $fatal(1, "EOF inside P data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "P") $fatal(1, "bad P line: %s", cur_line);
      exp_mem[i] = w[15:0];
    end
  endtask

  // ── record handlers ────────────────────────────────────────────────────────
  task automatic do_row(input string hdr);
    int n, nresc, flags, d0, cnt;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d %d %d", unused_tag, n, nresc, flags);
    if (cnt != 4) $fatal(1, "bad R line: %s", hdr);
    read_payload(n);
    cov_row(n, nresc, flags);
    cov_probs(n);
    n_rows++;
    name = $sformatf("R%0d n=%0d", n_rows, n);

    d0 = done_count;
    exp_beats_at_done = n;
    feed_row(n);
    wait_done_rel(d0, name);
    if (done_count != d0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done_count=%0d (exp %0d)",
               name, cyc, done_count, d0 + 1);
    end
    if (got_q.size() != n) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: expected %0d beats, got %0d",
               name, cyc, n, got_q.size());
    end else begin
      for (int i = 0; i < n; i++) begin
        if (got_q[i] !== exp_mem[i]) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: p[%0d] got 0x%04x exp 0x%04x",
                   name, cyc, i, got_q[i], exp_mem[i]);
        end
        if (got_last_q[i] !== bit'(i == n - 1)) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d last=%0b (exp %0b)",
                   name, cyc, i, got_last_q[i], i == n - 1);
        end
      end
    end
    got_q.delete();
    got_last_q.delete();
    repeat (2) @(negedge clk);
    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy stuck after done", name, cyc);
    end
    if (row_error_sticky !== exp_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky=%0b expected %0b",
               name, cyc, row_error_sticky, exp_sticky);
    end
  endtask

  task automatic do_reject(input string hdr);
    int n, e0, d0, cnt;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d", unused_tag, n);
    if (cnt != 2) $fatal(1, "bad O line: %s", hdr);
    n_rejects++;
    name = $sformatf("O%0d n=%0d", n_rejects, n);
    if (n == int'(ROW_MAX) + 1) cov_rej_direct++;
    else                        cov_rej_flush++;

    e0 = err_pulses;
    d0 = done_count;
    exp_beats_at_done = 0;
    for (int i = 0; i < n; i++) begin
      stall_gap();
      send_score(32'(i * 3 + 7), i == n - 1);
    end
    exp_sticky = 1'b1;
    repeat (5) @(negedge clk);
    if (err_pulses != e0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: err_pulses=%0d (exp %0d)",
               name, cyc, err_pulses, e0 + 1);
    end
    if (!row_error_sticky) begin
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
    int n, tphase, abortcyc, cnt, phase, t;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d %d %d", unused_tag, n, tphase, abortcyc);
    if (cnt != 4) $fatal(1, "bad X line: %s", hdr);
    n_resets++;
    name = $sformatf("X%0d n=%0d phase=%0d abort=%0d",
                     n_resets, n, tphase, abortcyc);

    fork
      begin
        for (int i = 0; i < n; i++) begin
          stall_gap();
          send_score(32'(i * 5 + 11), i == n - 1);
        end
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
    if (row_error_sticky) begin
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
    s_score    = '0;
    s_last     = 1'b0;
    n_rows     = 0;
    n_rejects  = 0;
    n_resets   = 0;
    exp_beats_at_done = 0;

    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'hDEAD_4EA1;
    if (TB_SCORE_FRAC == 0) cov_cfg_frac0++;
    else                    cov_cfg_frac10++;
    unique case (bp_mode)
      0: cov_bp_full++; 1: cov_bp_random++; default: cov_bp_storm++;
    endcase
    unique case (stall_mode)
      0: cov_stall_none++; 1: cov_stall_random++; default: cov_stall_storm++;
    endcase

    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);
    if (busy) begin
      errors++;
      $display("FAIL [reset] cyc=%0d: busy asserted out of reset", cyc);
    end
    if (row_error_sticky) begin
      errors++;
      $display("FAIL [reset] cyc=%0d: sticky error out of reset", cyc);
    end

    while (next_line()) begin
      tag0 = cur_line.getc(0);
      unique case (tag0)
        "R":     do_row(cur_line);
        "O":     do_reject(cur_line);
        "X":     do_reset_job(cur_line);
        default: $fatal(1, "unexpected record: %s", cur_line);
      endcase
    end
    $fclose(fd);

    repeat (5) @(negedge clk);
    errors += d006_fails;
    print_coverage();
    if (errors == 0) begin
      $display("TB PASS: %0d rows, %0d rejects, %0d mid-op resets, 0 errors (frac=%0d bp=%0d stall=%0d seed=%0d)",
               n_rows, n_rejects, n_resets, TB_SCORE_FRAC, bp_mode,
               stall_mode, seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s) across %0d rows", errors, n_rows);
    end
  end

endmodule
