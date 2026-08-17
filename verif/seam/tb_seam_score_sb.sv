// tb_seam_score_sb.sv — seam_score_dequant vector-driven scoreboard TB
// (V0/mxe/asu pattern): stimulus + bit-exact expected scores come from
// gen_seam_vectors.py (golden arbiter apex_golden.attention
// score_dequant_fx / its comp-bits mirror); the TB only drives, collects
// and compares. Runs under Verilator (--binary --timing --assert).
//
// This TB carries the D-021 S-3 EQUIVALENCE PROOF: the random vector set
// replays >= 1M (acc, comp) operand pairs from the true S-3 distribution
// plus the directed edge/discriminator cases, all compared bit-exact.
//
// Vector records (see gen_seam_vectors.py header): J legal / I illegal /
// R mid-op reset. The J header carries the expected range-error element
// count (golden `assert |score| < 2^31` mirror): the TB requires exactly
// that many range_error pulses and the matching sticky value.
//
// Protocol adversaries (plusargs):
//   +bp_mode=0|1|2     score-stream backpressure: none / ~75% duty / storm
//   +stall_mode=0|1|2  cmp+acc feed gaps: none / short random / bursty
//   +seed=<n>
//
// §5/D-006 checked every cycle by the bound seam_job_sva + apex_stream1_sva.
// The comp-finite input contract is asserted here (ap_cmp_finite).

`include "apex_stream1_sva.svh"
`include "seam_job_sva.svh"

module tb_seam_score_sb;

  import apex_pkg::*;

  localparam int unsigned N_MAX_TB    = 1024;
  localparam int unsigned MAX_BEATS   = N_MAX_TB / 8;
  localparam int unsigned JOB_TIMEOUT = 400000;

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
  logic               job_valid, job_ready;
  logic [DIM_W-1:0]   job_cols;
  logic               job_error, job_error_sticky, busy, done;
  logic               cmp_valid, cmp_ready;
  logic [31:0]        cmp_data;
  logic               acc_valid, acc_ready;
  lane32_beat_t       acc_beat;
  logic               score_valid, score_ready;
  logic signed [31:0] score_data;
  logic               score_last;
  logic               range_error, range_error_sticky;
  logic               frame_error, frame_error_sticky;

  seam_score_dequant #(.N_MAX(N_MAX_TB)) dut (
    .clk                (clk),
    .rst_n              (rst_n),
    .job_valid          (job_valid),
    .job_ready          (job_ready),
    .job_cols           (job_cols),
    .job_error          (job_error),
    .job_error_sticky   (job_error_sticky),
    .busy               (busy),
    .done               (done),
    .cmp_valid          (cmp_valid),
    .cmp_ready          (cmp_ready),
    .cmp_data           (cmp_data),
    .acc_valid          (acc_valid),
    .acc_ready          (acc_ready),
    .acc_beat           (acc_beat),
    .score_valid        (score_valid),
    .score_ready        (score_ready),
    .score_data         (score_data),
    .score_last         (score_last),
    .range_error        (range_error),
    .range_error_sticky (range_error_sticky),
    .frame_error        (frame_error),
    .frame_error_sticky (frame_error_sticky)
  );

  // ── bound SVA: D-006 job pack + §5 stability on every stream ───────────────
  bind seam_score_dequant seam_job_sva #(
    .CNT_W(13), .CHECK_JOB(1'b1), .NAME("sd.score")
  ) u_sva_job (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (job_valid),
    .job_ready        (job_ready),
    .job_error        (job_error),
    .job_error_sticky (job_error_sticky),
    .busy             (busy),
    .done             (done),
    .exp_beats        (13'(job_cols)),
    .out_valid        (score_valid),
    .out_ready        (score_ready),
    .out_last         (score_last)
  );

  bind seam_score_dequant apex_stream1_sva #(.WIDTH(32), .NAME("sd.cmp"))
    u_sva_cmp (.clk(clk), .rst_n(rst_n), .valid(cmp_valid), .ready(cmp_ready),
               .data(cmp_data));
  bind seam_score_dequant apex_stream1_sva #(.WIDTH(257), .NAME("sd.acc"))
    u_sva_acc (.clk(clk), .rst_n(rst_n), .valid(acc_valid), .ready(acc_ready),
               .data({acc_beat.data, acc_beat.last}));
  bind seam_score_dequant apex_stream1_sva #(.WIDTH(33), .NAME("sd.score"))
    u_sva_sco (.clk(clk), .rst_n(rst_n), .valid(score_valid),
               .ready(score_ready), .data({score_data, score_last}));
  bind seam_score_dequant apex_stream1_sva #(.WIDTH(12), .NAME("sd.job"))
    u_sva_job_s (.clk(clk), .rst_n(rst_n), .valid(job_valid),
                 .ready(job_ready), .data(job_cols));

  // input contract (module header): composite scales are finite fp32
  ap_cmp_finite: assert property (@(posedge clk) disable iff (!rst_n)
    cmp_valid |-> (cmp_data[30:23] != 8'hFF))
    else $error("[SVA sd] non-finite composite scale offered");

  // ── plusarg config ─────────────────────────────────────────────────────────
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned seed         = 32'h5C0E_2026;
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
      $dumpvars(0, tb_seam_score_sb);
    end
  end

  // ── monitors ───────────────────────────────────────────────────────────────
  int done_count, err_pulses, rerr_pulses;
  always @(posedge clk) begin
    if (!rst_n) begin
      done_count  <= 0;
      err_pulses  <= 0;
      rerr_pulses <= 0;
    end else begin
      if (done)        done_count  <= done_count + 1;
      if (job_error)   err_pulses  <= err_pulses + 1;
      if (range_error) rerr_pulses <= rerr_pulses + 1;
    end
  end

  // frame_error must NEVER fire (TB frames the acc stream correctly)
  always @(posedge clk) begin
    if (rst_n && frame_error)
      $fatal(1, "FAIL cyc=%0d: frame_error fired on a correctly framed feed",
             cyc);
  end

  // score consumer with configurable backpressure
  logic [31:0] got_q [$];
  bit          got_last_q [$];
  logic [31:0] lfsr;
  always @(negedge clk) begin
    if (!rst_n) begin
      lfsr        <= seed;
      score_ready <= 1'b0;
    end else begin
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      unique case (bp_mode)
        0:       score_ready <= 1'b1;
        1:       score_ready <= |lfsr[1:0];
        default: score_ready <= (lfsr[3:0] == 4'h0);
      endcase
    end
  end
  always @(posedge clk) begin
    if (rst_n && score_valid && score_ready) begin
      got_q.push_back($unsigned(score_data));
      got_last_q.push_back(score_last);
    end
  end

  // ── TB-side randomization (feed gaps) ──────────────────────────────────────
  int unsigned rng_state = 32'hC0FFEE07;
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
  logic [31:0]  cmp_mem  [N_MAX_TB];
  logic [255:0] acc_mem  [MAX_BEATS];
  logic [31:0]  exp_mem  [N_MAX_TB];

  // ── coverage buckets ───────────────────────────────────────────────────────
  int cov_cols_1, cov_cols_max, cov_cols_mid, cov_cols_tail;
  int cov_acc_zero, cov_acc_extreme, cov_tie, cov_p53, cov_underflow;
  int cov_range_err, cov_comp_subnormal, cov_comp_one;
  int cov_ill_cols0, cov_ill_cols_big;
  int cov_rst_load, cov_rst_proc, cov_rst_wait, cov_rst_idle, cov_rst_random;
  int cov_bp_full, cov_bp_random, cov_bp_storm;
  int cov_stall_none, cov_stall_random, cov_stall_storm;

  function automatic void cov_job(input int cols, rerr, fz, fext, ftie, fp53,
                                  fund, fsub, fone);
    if (cols == 1)                     cov_cols_1++;
    else if (cols == int'(N_MAX_TB))   cov_cols_max++;
    else                               cov_cols_mid++;
    if (cols % 8 != 0) cov_cols_tail++;
    if (rerr != 0) cov_range_err++;
    if (fz   != 0) cov_acc_zero++;
    if (fext != 0) cov_acc_extreme++;
    if (ftie != 0) cov_tie++;
    if (fp53 != 0) cov_p53++;
    if (fund != 0) cov_underflow++;
    if (fsub != 0) cov_comp_subnormal++;
    if (fone != 0) cov_comp_one++;
  endfunction

  function automatic void print_coverage();
    $display("COV sd_cols_1 %0d",         cov_cols_1);
    $display("COV sd_cols_max %0d",       cov_cols_max);
    $display("COV sd_cols_mid %0d",       cov_cols_mid);
    $display("COV sd_cols_tail %0d",      cov_cols_tail);
    $display("COV sd_acc_zero %0d",       cov_acc_zero);
    $display("COV sd_acc_extreme %0d",    cov_acc_extreme);
    $display("COV sd_tie %0d",            cov_tie);
    $display("COV sd_p53 %0d",            cov_p53);
    $display("COV sd_underflow %0d",      cov_underflow);
    $display("COV sd_range_err %0d",      cov_range_err);
    $display("COV sd_comp_subnormal %0d", cov_comp_subnormal);
    $display("COV sd_comp_one %0d",       cov_comp_one);
    $display("COV sd_ill_cols0 %0d",      cov_ill_cols0);
    $display("COV sd_ill_cols_big %0d",   cov_ill_cols_big);
    $display("COV sd_rst_load %0d",       cov_rst_load);
    $display("COV sd_rst_proc %0d",       cov_rst_proc);
    $display("COV sd_rst_wait %0d",       cov_rst_wait);
    $display("COV sd_rst_idle %0d",       cov_rst_idle);
    $display("COV sd_rst_random %0d",     cov_rst_random);
    $display("COV sd_bp_full %0d",        cov_bp_full);
    $display("COV sd_bp_random %0d",      cov_bp_random);
    $display("COV sd_bp_storm %0d",       cov_bp_storm);
    $display("COV sd_stall_none %0d",     cov_stall_none);
    $display("COV sd_stall_random %0d",   cov_stall_random);
    $display("COV sd_stall_storm %0d",    cov_stall_storm);
  endfunction

  // ── low-level drivers ──────────────────────────────────────────────────────
  int errors;
  bit exp_sticky;

  task automatic send_job(input logic [DIM_W-1:0] cols);
    @(negedge clk);
    job_cols  = cols;
    job_valid = 1'b1;
    while (!job_ready) @(negedge clk);
    @(negedge clk);
    job_valid = 1'b0;
  endtask

  task automatic feed_cmp(input int num);
    for (int i = 0; i < num; i++) begin
      stall_gap();
      @(negedge clk);
      cmp_data  = cmp_mem[i];
      cmp_valid = 1'b1;
      while (!cmp_ready) @(negedge clk);
      @(negedge clk);
      cmp_valid = 1'b0;
    end
  endtask

  task automatic feed_acc(input int nb);
    for (int i = 0; i < nb; i++) begin
      stall_gap();
      @(negedge clk);
      acc_beat.data = acc_mem[i];
      acc_beat.last = (i == nb - 1);
      acc_valid     = 1'b1;
      while (!acc_ready) @(negedge clk);
      @(negedge clk);
      acc_valid = 1'b0;
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

  task automatic read_ca_lines(input int cols);
    string tg;
    logic [31:0] w;
    logic [31:0] lane [8];
    int cnt, nb;
    for (int i = 0; i < cols; i++) begin
      if (!next_line()) $fatal(1, "EOF inside C data (col %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "C") $fatal(1, "bad C line: %s", cur_line);
      cmp_mem[i] = w;
    end
    nb = (cols + 7) / 8;
    for (int i = 0; i < nb; i++) begin
      if (!next_line()) $fatal(1, "EOF inside A data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h %h %h %h %h %h %h %h", tg,
                    lane[0], lane[1], lane[2], lane[3],
                    lane[4], lane[5], lane[6], lane[7]);
      if (cnt != 9 || tg != "A") $fatal(1, "bad A line: %s", cur_line);
      for (int c = 0; c < 8; c++) acc_mem[i][32*c +: 32] = lane[c];
    end
  endtask

  // ── job records ────────────────────────────────────────────────────────────
  int n_jobs, n_illegal, n_resets;

  task automatic check_results(input string name, input int cols);
    if (got_q.size() != cols) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: expected %0d scores, got %0d",
               name, cyc, cols, got_q.size());
    end else begin
      for (int i = 0; i < cols; i++) begin
        if (got_q[i] !== exp_mem[i]) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: score %0d got %08x exp %08x (acc=%08x comp=%08x)",
                   name, cyc, i, got_q[i], exp_mem[i],
                   acc_mem[i/8][32*(i%8) +: 32], cmp_mem[i]);
        end
        if (got_last_q[i] !== bit'(i == cols - 1)) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: score %0d last=%0b (exp %0b)",
                   name, cyc, i, got_last_q[i], i == cols - 1);
        end
      end
    end
    got_q.delete();
    got_last_q.delete();
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
    int cols, rerr, fz, fext, ftie, fp53, fund, fsub, fone;
    int d0, r0, cnt;
    string unused_tag, name, tg;
    logic [31:0] w;
    bit exp_rsticky0;
    cnt = $sscanf(hdr, "%s %d %d %d %d %d %d %d %d %d", unused_tag,
                  cols, rerr, fz, fext, ftie, fp53, fund, fsub, fone);
    if (cnt != 10) $fatal(1, "bad J line: %s", hdr);
    read_ca_lines(cols);
    for (int i = 0; i < cols; i++) begin
      if (!next_line()) $fatal(1, "EOF inside E data");
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "E") $fatal(1, "bad E line: %s", cur_line);
      exp_mem[i] = w;
    end
    cov_job(cols, rerr, fz, fext, ftie, fp53, fund, fsub, fone);
    n_jobs++;
    name = $sformatf("J%0d cols=%0d rerr=%0d", n_jobs, cols, rerr);

    d0 = done_count;
    r0 = rerr_pulses;
    exp_rsticky0 = range_error_sticky;
    send_job(DIM_W'(cols));
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
    fork
      feed_cmp(cols);
      feed_acc((cols + 7) / 8);
    join
    wait_done_rel(d0, name);
    if (done_count != d0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done_count=%0d (exp %0d)",
               name, cyc, done_count, d0 + 1);
    end
    check_results(name, cols);
    // golden range-assert mirror: exactly `rerr` pulses, sticky iff any ever
    if (rerr_pulses != r0 + rerr) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: range_error pulses=%0d (exp %0d)",
               name, cyc, rerr_pulses - r0, rerr);
    end
    if (range_error_sticky !== (exp_rsticky0 || (rerr != 0))) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: range_error_sticky=%0b (exp %0b)",
               name, cyc, range_error_sticky, exp_rsticky0 || (rerr != 0));
    end
    post_job_checks(name);
  endtask

  task automatic do_illegal(input string hdr);
    int cols, e0, d0, cnt;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d", unused_tag, cols);
    if (cnt != 2) $fatal(1, "bad I line: %s", hdr);
    if (cols == 0) cov_ill_cols0++; else cov_ill_cols_big++;
    n_illegal++;
    name = $sformatf("I%0d cols=%0d", n_illegal, cols);

    e0 = err_pulses;
    d0 = done_count;
    send_job(DIM_W'(cols));
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
    if (got_q.size() != 0) begin
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
    int cols, abortcyc, tphase, cnt, phase, t;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d %d %d", unused_tag, cols, abortcyc, tphase);
    if (cnt != 4) $fatal(1, "bad R line: %s", hdr);
    read_ca_lines(cols);
    n_resets++;
    name = $sformatf("R%0d cols=%0d abort=%0d phase=%0d",
                     n_resets, cols, abortcyc, tphase);

    send_job(DIM_W'(cols));
    fork
      begin
        fork
          feed_cmp(cols);
          feed_acc((cols + 7) / 8);
        join
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
      1:       cov_rst_load++;
      2:       cov_rst_proc++;
      3:       cov_rst_wait++;
      default: cov_rst_idle++;
    endcase
    if (tphase == 0) cov_rst_random++;
    $display("RESET [%s] cyc=%0d: mid-op reset in FSM phase %0d",
             name, cyc, phase);

    rst_n     = 1'b0;
    cmp_valid = 1'b0;
    acc_valid = 1'b0;
    job_valid = 1'b0;
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
    if (job_error_sticky || range_error_sticky || frame_error_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky not cleared by reset", name, cyc);
    end
    if (!job_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: job_ready not restored after reset",
               name, cyc);
    end
    repeat (10) @(negedge clk);
    if (got_q.size() != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: stray output after mid-op reset", name, cyc);
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
    job_valid  = 1'b0;
    job_cols   = '0;
    cmp_valid  = 1'b0;
    cmp_data   = '0;
    acc_valid  = 1'b0;
    acc_beat   = '0;
    n_jobs     = 0;
    n_illegal  = 0;
    n_resets   = 0;

    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'h5C0E_4EA1;
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
    if (job_error_sticky || range_error_sticky || frame_error_sticky) begin
      errors++;
      $display("FAIL [reset] cyc=%0d: sticky error out of reset", cyc);
    end

    while (next_line()) begin
      tag0 = cur_line.getc(0);
      unique case (tag0)
        "J":     do_legal(cur_line);
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
