// tb_wfeed_w4_sb.sv — mxe_wfeed_w4 vector-driven scoreboard TB (the proven
// V0/seam/mxe pattern): stimulus + bit-exact expected results come from
// gen_w4_vectors.py (golden arbiter apex_golden.weight_codec,
// wfeed_w4_to_i8 realization "A"); the TB only drives, collects and
// compares. Simulator flags: --binary --timing --assert.
//
// Vector records (see gen_w4_vectors.py header): F legal / I illegal /
// R mid-op reset. Protocol adversaries (plusargs):
//   +bp_mode=0|1|2     w8 output backpressure: none / ~75% duty / storm
//   +stall_mode=0|1|2  packed-W4 input feed gaps: none / short / bursty
//   +seed=<n>          LFSR seed for the consumer + stalls
//
// The §5/D-006 contract is additionally checked every cycle by the bound
// w4_job_sva (job/count/last framing on the INT8 output) and
// apex_stream1_sva (data stability on all three streams).
//
// PERF (docs/OPTIMIZATION.md:67, weight_codec S6): the TB counts packed
// beats CONSUMED and INT8 beats EMITTED per job and asserts
//     emitted == 2*consumed - (beats & 1)        [w4_en = 1]
//     emitted == consumed                        [w4_en = 0, passthrough]
// The odd-tail term is load-bearing — the unqualified "emitted == 2*consumed"
// is false for legal odd ceil(K/8)*N, and the directed set covers it.
//
// TB discipline: stimulus and handshake decisions at NEGEDGE, commits
// observed at posedge; $fatal watchdog bounds the run.

`include "apex_stream1_sva.svh"
`include "w4_job_sva.svh"

module tb_wfeed_w4_sb;

  import apex_pkg::*;

  localparam int unsigned BEATS_MAX_TB = 256;    // matches gen_w4_vectors.py
  localparam int unsigned JOB_TIMEOUT  = 400000;

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
  logic             w4_en;
  logic             job_valid, job_ready;
  logic [DIM_W-1:0] job_beats;
  logic             job_error, job_error_sticky, busy, done;
  logic             pw_valid, pw_ready;
  lane8_beat_t      pw_beat;
  logic             w8_valid, w8_ready;
  lane8_beat_t      w8_beat;

  mxe_wfeed_w4 #(.BEATS_MAX(BEATS_MAX_TB)) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .w4_en            (w4_en),
    .job_valid        (job_valid),
    .job_ready        (job_ready),
    .job_beats        (job_beats),
    .job_error        (job_error),
    .job_error_sticky (job_error_sticky),
    .busy             (busy),
    .done             (done),
    .pw_valid         (pw_valid),
    .pw_ready         (pw_ready),
    .pw_beat          (pw_beat),
    .w8_valid         (w8_valid),
    .w8_ready         (w8_ready),
    .w8_beat          (w8_beat)
  );

  // ── bound SVA: D-006 job pack on the INT8 output stream ───────────────────
  bind mxe_wfeed_w4 w4_job_sva #(
    .CNT_W(13), .CHECK_JOB(1'b1), .NAME("wf.w8")
  ) u_sva_job (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (job_valid),
    .job_ready        (job_ready),
    .job_error        (job_error),
    .job_error_sticky (job_error_sticky),
    .busy             (busy),
    .done             (done),
    .exp_beats        (13'(job_beats)),
    .out_valid        (w8_valid),
    .out_ready        (w8_ready),
    .out_last         (w8_beat.last)
  );

  // §5 data stability on every stream
  bind mxe_wfeed_w4 apex_stream1_sva #(.WIDTH(65), .NAME("wf.pw"))
    u_sva_pw  (.clk(clk), .rst_n(rst_n), .valid(pw_valid), .ready(pw_ready),
               .data({pw_beat.data, pw_beat.last}));
  bind mxe_wfeed_w4 apex_stream1_sva #(.WIDTH(65), .NAME("wf.w8"))
    u_sva_w8  (.clk(clk), .rst_n(rst_n), .valid(w8_valid), .ready(w8_ready),
               .data({w8_beat.data, w8_beat.last}));
  bind mxe_wfeed_w4 apex_stream1_sva #(.WIDTH(12), .NAME("wf.job"))
    u_sva_job_s (.clk(clk), .rst_n(rst_n), .valid(job_valid),
                 .ready(job_ready), .data(job_beats));

  // mode is sampled at accept and must not change a job in flight (module
  // header): the TB never moves w4_en mid-job, and this proves it.
  ap_mode_stable: assert property (@(posedge clk) disable iff (!rst_n)
    busy |-> $stable(w4_en))
    else $error("[SVA wf] w4_en moved while a job was in flight");

  // ── plusarg config ─────────────────────────────────────────────────────────
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned seed         = 32'h04FE_2026;
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
      $dumpfile("dump.fst");
      $dumpvars(0, tb_wfeed_w4_sb);
    end
  end

  // ── monitors (posedge — exactly what the DUT commits) ─────────────────────
  int done_count;
  int err_pulses;
  int pw_taken;                       // packed beats CONSUMED (PERF)
  always @(posedge clk) begin
    if (!rst_n) begin
      done_count <= 0;
      err_pulses <= 0;
    end else begin
      if (done)      done_count <= done_count + 1;
      if (job_error) err_pulses <= err_pulses + 1;
    end
  end
  always @(posedge clk) begin
    if (rst_n && pw_valid && pw_ready) pw_taken <= pw_taken + 1;
  end

  // consumer with configurable backpressure
  logic [63:0] got_beat_q [$];
  bit          got_blast_q [$];
  logic [31:0] lfsr_o;
  always @(negedge clk) begin
    if (!rst_n) begin
      lfsr_o   <= seed;
      w8_ready <= 1'b0;
    end else begin
      lfsr_o <= {lfsr_o[30:0], lfsr_o[31] ^ lfsr_o[21] ^ lfsr_o[1] ^ lfsr_o[0]};
      unique case (bp_mode)
        0:       w8_ready <= 1'b1;
        1:       w8_ready <= |lfsr_o[1:0];
        default: w8_ready <= (lfsr_o[3:0] == 4'h0);
      endcase
    end
  end
  always @(posedge clk) begin
    if (rst_n && w8_valid && w8_ready) begin
      got_beat_q.push_back(w8_beat.data);
      got_blast_q.push_back(w8_beat.last);
    end
  end

  // ── TB-side randomization (input stall gaps) ───────────────────────────────
  int unsigned rng_state = 32'hC0FFEE05;
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

  // ── stimulus / expected storage ────────────────────────────────────────────
  logic [63:0] pw_mem   [BEATS_MAX_TB];
  logic [63:0] beat_exp [BEATS_MAX_TB];

  // ── coverage buckets (manual, run2 pattern) ────────────────────────────────
  int cov_mode_w4, cov_mode_pass;
  int cov_tail_odd, cov_tail_even;
  int cov_beats_1, cov_beats_max, cov_beats_mid;
  int cov_code_min, cov_code_max, cov_code_zero, cov_code_neg;
  int cov_ill_beats0, cov_ill_beats_big;
  int cov_rst_idle, cov_rst_run, cov_rst_wait, cov_rst_random;
  int cov_bp_full, cov_bp_random, cov_bp_storm;
  int cov_stall_none, cov_stall_random, cov_stall_storm;
  int cov_perf_halved, cov_perf_oddtail, cov_perf_pass;

  function automatic void cov_job(input int beats, mode, odd,
                                  cmin, cmax, czero, cneg);
    if (mode != 0) cov_mode_w4++;   else cov_mode_pass++;
    if (odd  != 0) cov_tail_odd++;  else cov_tail_even++;
    if (beats == 1)                      cov_beats_1++;
    else if (beats == int'(BEATS_MAX_TB)) cov_beats_max++;
    else                                 cov_beats_mid++;
    if (cmin  != 0) cov_code_min++;
    if (cmax  != 0) cov_code_max++;
    if (czero != 0) cov_code_zero++;
    if (cneg  != 0) cov_code_neg++;
  endfunction

  function automatic void print_coverage();
    $display("COV w4_mode_w4 %0d",      cov_mode_w4);
    $display("COV w4_mode_pass %0d",    cov_mode_pass);
    $display("COV w4_tail_odd %0d",     cov_tail_odd);
    $display("COV w4_tail_even %0d",    cov_tail_even);
    $display("COV w4_beats_1 %0d",      cov_beats_1);
    $display("COV w4_beats_max %0d",    cov_beats_max);
    $display("COV w4_beats_mid %0d",    cov_beats_mid);
    $display("COV w4_code_min %0d",     cov_code_min);
    $display("COV w4_code_max %0d",     cov_code_max);
    $display("COV w4_code_zero %0d",    cov_code_zero);
    $display("COV w4_code_neg %0d",     cov_code_neg);
    $display("COV w4_ill_beats0 %0d",   cov_ill_beats0);
    $display("COV w4_ill_beats_big %0d", cov_ill_beats_big);
    $display("COV w4_rst_idle %0d",     cov_rst_idle);
    $display("COV w4_rst_run %0d",      cov_rst_run);
    $display("COV w4_rst_wait %0d",     cov_rst_wait);
    $display("COV w4_rst_random %0d",   cov_rst_random);
    $display("COV w4_bp_full %0d",      cov_bp_full);
    $display("COV w4_bp_random %0d",    cov_bp_random);
    $display("COV w4_bp_storm %0d",     cov_bp_storm);
    $display("COV w4_stall_none %0d",   cov_stall_none);
    $display("COV w4_stall_random %0d", cov_stall_random);
    $display("COV w4_stall_storm %0d",  cov_stall_storm);
    $display("COV w4_perf_halved %0d",  cov_perf_halved);
    $display("COV w4_perf_oddtail %0d", cov_perf_oddtail);
    $display("COV w4_perf_pass %0d",    cov_perf_pass);
  endfunction

  // ── low-level drivers (negedge-driven) ─────────────────────────────────────
  int errors;
  bit exp_sticky;

  task automatic send_job(input logic [DIM_W-1:0] beats, input bit mode);
    @(negedge clk);
    w4_en     = mode;
    job_beats = beats;
    job_valid = 1'b1;
    while (!job_ready) @(negedge clk);
    @(negedge clk);
    job_valid = 1'b0;
  endtask

  task automatic feed_packed(input int num);
    for (int i = 0; i < num; i++) begin
      stall_gap();
      @(negedge clk);
      pw_beat.data = pw_mem[i];
      pw_beat.last = (i == num - 1);
      pw_valid     = 1'b1;
      while (!pw_ready) @(negedge clk);
      @(negedge clk);
      pw_valid = 1'b0;
    end
  endtask

  task automatic wait_done_rel(input int d0, input string name);
    int t;
    t = 0;
    while (done_count == d0) begin
      @(negedge clk);
      t++;
      if (t > int'(JOB_TIMEOUT))
        $fatal(1, "FAIL [%s]: job timeout after %0d cycles (cyc=%0d)",
               name, t, cyc);
    end
  endtask

  // ── vector-file reader ─────────────────────────────────────────────────────
  int    fd;
  string cur_line;

  function automatic bit next_line();
    while ($fgets(cur_line, fd) > 0) begin
      if (cur_line.len() > 1 && cur_line.getc(0) != "#") return 1'b1;
    end
    return 1'b0;
  endfunction

  task automatic read_p_lines(input int n);
    string tg;
    logic [63:0] w;
    int cnt;
    for (int i = 0; i < n; i++) begin
      if (!next_line()) $fatal(1, "EOF inside P data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "P") $fatal(1, "bad P line: %s", cur_line);
      pw_mem[i] = w;
    end
  endtask

  // ── job records ────────────────────────────────────────────────────────────
  int n_jobs, n_illegal, n_resets;

  task automatic check_results(input string name, input int beats,
                               input int mode, input int consumed);
    int exp_consumed;
    if (got_beat_q.size() != beats) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: expected %0d beats, got %0d",
               name, cyc, beats, got_beat_q.size());
    end else begin
      for (int i = 0; i < beats; i++) begin
        if (got_beat_q[i] !== beat_exp[i]) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d\n  got %016x\n  exp %016x",
                   name, cyc, i, got_beat_q[i], beat_exp[i]);
        end
        if (got_blast_q[i] !== bit'(i == beats - 1)) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d last=%0b (exp %0b)",
                   name, cyc, i, got_blast_q[i], i == beats - 1);
        end
      end
    end

    // PERF: the beats-halved relation, measured not assumed (S6)
    exp_consumed = (mode != 0) ? ((beats + 1) / 2) : beats;
    if (consumed != exp_consumed) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: PERF consumed=%0d expected %0d",
               name, cyc, consumed, exp_consumed);
    end else if (mode != 0) begin
      if (beats != 2 * consumed - (beats & 1)) begin
        errors++;
        $display("FAIL [%s] cyc=%0d: PERF emitted=%0d != 2*%0d-%0d",
                 name, cyc, beats, consumed, beats & 1);
      end
      $display("PERF [%s]: consumed=%0d emitted=%0d ratio=%0d/%0d",
               name, consumed, beats, beats, consumed);
      if ((beats & 1) != 0) cov_perf_oddtail++; else cov_perf_halved++;
    end else begin
      cov_perf_pass++;
    end

    got_beat_q.delete();
    got_blast_q.delete();
  endtask

  task automatic post_job_checks(input string name);
    repeat (2) @(negedge clk);
    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy stuck after done", name, cyc);
    end
    if (!job_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: job_ready not restored after done",
               name, cyc);
    end
    if (job_error_sticky !== exp_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky=%0b expected %0b",
               name, cyc, job_error_sticky, exp_sticky);
    end
  endtask

  task automatic do_legal(input string hdr);
    int beats, mode, odd, cmin, cmax, czero, cneg;
    int d0, cnt, consumed, p0;
    string unused_tag, name, tg;
    logic [63:0] w64;
    cnt = $sscanf(hdr, "%s %d %d %d %d %d %d %d", unused_tag,
                  beats, mode, odd, cmin, cmax, czero, cneg);
    if (cnt != 8) $fatal(1, "bad F line: %s", hdr);
    consumed = (mode != 0) ? ((beats + 1) / 2) : beats;
    read_p_lines(consumed);
    for (int i = 0; i < beats; i++) begin
      if (!next_line()) $fatal(1, "EOF inside E data");
      cnt = $sscanf(cur_line, "%s %h", tg, w64);
      if (cnt != 2 || tg != "E") $fatal(1, "bad E line: %s", cur_line);
      beat_exp[i] = w64;
    end
    cov_job(beats, mode, odd, cmin, cmax, czero, cneg);
    n_jobs++;
    name = $sformatf("F%0d beats=%0d w4=%0d", n_jobs, beats, mode);

    d0 = done_count;
    p0 = pw_taken;
    send_job(DIM_W'(beats), bit'(mode));
    @(negedge clk);
    if (job_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: job_ready high while job in flight",
               name, cyc);
    end
    if (!busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy low right after job accept", name, cyc);
    end
    feed_packed(consumed);
    wait_done_rel(d0, name);
    if (done_count != d0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done_count=%0d (exp %0d)",
               name, cyc, done_count, d0 + 1);
    end
    check_results(name, beats, mode, pw_taken - p0);
    post_job_checks(name);
  endtask

  task automatic do_illegal(input string hdr);
    int beats, e0, d0, cnt;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d", unused_tag, beats);
    if (cnt != 2) $fatal(1, "bad I line: %s", hdr);
    if (beats == 0) cov_ill_beats0++; else cov_ill_beats_big++;
    n_illegal++;
    name = $sformatf("I%0d beats=%0d", n_illegal, beats);

    e0 = err_pulses;
    d0 = done_count;
    send_job(DIM_W'(beats), 1'b1);
    repeat (3) @(negedge clk);
    exp_sticky = 1'b1;
    if (err_pulses != e0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: err_pulses=%0d (exp %0d)",
               name, cyc, err_pulses, e0 + 1);
    end
    if (!job_error_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky bit not set", name, cyc);
    end
    if (done_count != d0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done pulsed for illegal job", name, cyc);
    end
    if (got_beat_q.size() != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: output from an illegal job", name, cyc);
    end
    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy after illegal job", name, cyc);
    end
    if (!job_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: job_ready lost after reject (D-019 gate)",
               name, cyc);
    end
  endtask

  task automatic do_reset_job(input string hdr);
    int beats, mode, abortcyc, tphase, cnt, phase, t, consumed;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d %d %d %d", unused_tag, beats, mode,
                  abortcyc, tphase);
    if (cnt != 5) $fatal(1, "bad R line: %s", hdr);
    consumed = (mode != 0) ? ((beats + 1) / 2) : beats;
    read_p_lines(consumed);
    n_resets++;
    name = $sformatf("R%0d beats=%0d abort=%0d phase=%0d",
                     n_resets, beats, abortcyc, tphase);

    send_job(DIM_W'(beats), bit'(mode));
    fork
      begin
        feed_packed(consumed);
        forever @(negedge clk);
      end
      begin
        if (tphase != 0) begin
          t = 0;
          while (int'(dut.state) != tphase) begin
            @(negedge clk);
            t++;
            if (t > int'(JOB_TIMEOUT))
              $fatal(1, "FAIL [%s]: FSM never reached phase %0d (cyc=%0d)",
                     name, tphase, cyc);
          end
        end else begin
          repeat (abortcyc) @(negedge clk);
        end
      end
    join_any
    disable fork;

    phase = int'(dut.state);
    unique case (phase)
      1:       cov_rst_run++;
      2:       cov_rst_wait++;
      default: cov_rst_idle++;
    endcase
    if (tphase == 0) cov_rst_random++;
    $display("RESET [%s] cyc=%0d: mid-op reset in FSM phase %0d",
             name, cyc, phase);

    rst_n     = 1'b0;
    pw_valid  = 1'b0;
    job_valid = 1'b0;
    repeat (3) @(negedge clk);
    rst_n      = 1'b1;
    exp_sticky = 1'b0;
    got_beat_q.delete();
    got_blast_q.delete();
    repeat (3) @(negedge clk);

    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy after mid-op reset", name, cyc);
    end
    if (job_error_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky not cleared by reset", name, cyc);
    end
    if (!job_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: job_ready not restored after reset",
               name, cyc);
    end
    repeat (10) @(negedge clk);
    if (got_beat_q.size() != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: stray output after mid-op reset",
               name, cyc);
    end
  endtask

  // ── main sequence ──────────────────────────────────────────────────────────
  initial begin : main
    string vec_path;
    byte   tag0;

    rst_n        = 1'b0;
    errors       = 0;
    exp_sticky   = 1'b0;
    w4_en        = 1'b0;
    job_valid    = 1'b0;
    job_beats    = '0;
    pw_valid     = 1'b0;
    pw_beat.data = '0;
    pw_beat.last = 1'b0;
    pw_taken     = 0;
    n_jobs       = 0;
    n_illegal    = 0;
    n_resets     = 0;

    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'hFEED_4EA1;
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
    if (job_error_sticky) begin
      errors++;
      $display("FAIL [reset] cyc=%0d: sticky error out of reset", cyc);
    end

    while (next_line()) begin
      tag0 = cur_line.getc(0);
      unique case (tag0)
        "F":     do_legal(cur_line);
        "I":     do_illegal(cur_line);
        "R":     do_reset_job(cur_line);
        default: $fatal(1, "unexpected record: %s", cur_line);
      endcase
    end
    $fclose(fd);

    repeat (5) @(negedge clk);
    print_coverage();
    if (errors == 0) begin
      $display("TB PASS: %0d legal jobs, %0d illegal jobs, %0d mid-op resets, 0 errors (bp=%0d stall=%0d seed=%0d)",
               n_jobs, n_illegal, n_resets, bp_mode, stall_mode, seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s) across %0d jobs", errors, n_jobs);
    end
  end

endmodule
