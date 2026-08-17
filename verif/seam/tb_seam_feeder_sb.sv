// tb_seam_feeder_sb.sv — seam_feeder_quant vector-driven scoreboard TB
// (the proven V0/mxe/asu pattern): stimulus + bit-exact expected results
// come from gen_seam_vectors.py (golden arbiter apex_golden.attention
// quant_rows_i8); the TB only drives, collects and compares. Simulator
// flags: --binary --timing --assert.
//
// Vector records (see gen_seam_vectors.py header): F legal / I illegal /
// R mid-op reset. Protocol adversaries (plusargs):
//   +bp_mode=0|1|2     out+scl backpressure: none / ~75% duty / storm
//   +stall_mode=0|1|2  fp32 input feed gaps: none / short random / bursty
//   +seed=<n>          LFSR seed for consumers + stalls
//
// The §5/D-006 contract is additionally checked every cycle by the bound
// seam_job_sva (job/count/last framing on BOTH output streams) and
// apex_stream1_sva (data stability on all four streams). The fp32-finite
// input contract is asserted here (ap_in_finite).
//
// Build configs: -GD_CFG=64 (default) and -GD_CFG=128 (D-021 both row
// lengths). TB discipline: stimulus and handshake decisions at NEGEDGE,
// commits observed at posedge; $fatal watchdog bounds the run.

`include "apex_stream1_sva.svh"
`include "seam_job_sva.svh"

module tb_seam_feeder_sb;

  import apex_pkg::*;

  parameter int unsigned D_CFG = 64;

  localparam int unsigned ROWS_MAX_TB = 64;
  localparam int unsigned BPR         = D_CFG / 8;
  localparam int unsigned MAX_ELEMS   = ROWS_MAX_TB * 128;
  localparam int unsigned MAX_BEATS   = ROWS_MAX_TB * 16;
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
  logic              job_valid, job_ready;
  logic [DIM_W-1:0]  job_rows;
  logic              job_error, job_error_sticky, busy, done;
  logic              in_valid, in_ready;
  logic [31:0]       in_data;
  logic              out_valid, out_ready;
  lane8_beat_t       out_beat;
  logic              scl_valid, scl_ready;
  logic [15:0]       scl_data;
  logic              scl_last;

  seam_feeder_quant #(.D(D_CFG), .ROWS_MAX(ROWS_MAX_TB)) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (job_valid),
    .job_ready        (job_ready),
    .job_rows         (job_rows),
    .job_error        (job_error),
    .job_error_sticky (job_error_sticky),
    .busy             (busy),
    .done             (done),
    .in_valid         (in_valid),
    .in_ready         (in_ready),
    .in_data          (in_data),
    .out_valid        (out_valid),
    .out_ready        (out_ready),
    .out_beat         (out_beat),
    .scl_valid        (scl_valid),
    .scl_ready        (scl_ready),
    .scl_data         (scl_data),
    .scl_last         (scl_last)
  );

  // ── bound SVA: D-006 job pack on BOTH framed output streams ───────────────
  bind seam_feeder_quant seam_job_sva #(
    .CNT_W(17), .CHECK_JOB(1'b1), .NAME("fq.out")
  ) u_sva_job_out (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (job_valid),
    .job_ready        (job_ready),
    .job_error        (job_error),
    .job_error_sticky (job_error_sticky),
    .busy             (busy),
    .done             (done),
    .exp_beats        (17'(job_rows) * 17'(D / 8)),
    .out_valid        (out_valid),
    .out_ready        (out_ready),
    .out_last         (out_beat.last)
  );

  bind seam_feeder_quant seam_job_sva #(
    .CNT_W(17), .CHECK_JOB(1'b0), .NAME("fq.scl")
  ) u_sva_job_scl (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (job_valid),
    .job_ready        (job_ready),
    .job_error        (job_error),
    .job_error_sticky (job_error_sticky),
    .busy             (busy),
    .done             (done),
    .exp_beats        (17'(job_rows)),
    .out_valid        (scl_valid),
    .out_ready        (scl_ready),
    .out_last         (scl_last)
  );

  // §5 data stability on every stream
  bind seam_feeder_quant apex_stream1_sva #(.WIDTH(32), .NAME("fq.in"))
    u_sva_in  (.clk(clk), .rst_n(rst_n), .valid(in_valid), .ready(in_ready),
               .data(in_data));
  bind seam_feeder_quant apex_stream1_sva #(.WIDTH(65), .NAME("fq.out"))
    u_sva_out (.clk(clk), .rst_n(rst_n), .valid(out_valid), .ready(out_ready),
               .data({out_beat.data, out_beat.last}));
  bind seam_feeder_quant apex_stream1_sva #(.WIDTH(17), .NAME("fq.scl"))
    u_sva_scl (.clk(clk), .rst_n(rst_n), .valid(scl_valid), .ready(scl_ready),
               .data({scl_data, scl_last}));
  bind seam_feeder_quant apex_stream1_sva #(.WIDTH(12), .NAME("fq.job"))
    u_sva_job (.clk(clk), .rst_n(rst_n), .valid(job_valid), .ready(job_ready),
               .data(job_rows));

  // input contract (module header): fp32 elements are finite
  ap_in_finite: assert property (@(posedge clk) disable iff (!rst_n)
    in_valid |-> (in_data[30:23] != 8'hFF))
    else $error("[SVA fq] non-finite fp32 offered to the feeder");

  // ── plusarg config ─────────────────────────────────────────────────────────
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned seed         = 32'hFEED_2026;
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
      $dumpvars(0, tb_seam_feeder_sb);
    end
  end

  // ── monitors (posedge — exactly what the DUT commits) ─────────────────────
  int done_count;
  int err_pulses;
  always @(posedge clk) begin
    if (!rst_n) begin
      done_count <= 0;
      err_pulses <= 0;
    end else begin
      if (done)      done_count <= done_count + 1;
      if (job_error) err_pulses <= err_pulses + 1;
    end
  end

  // consumers with independent configurable backpressure
  logic [63:0] got_beat_q [$];
  bit          got_blast_q [$];
  logic [15:0] got_scl_q [$];
  bit          got_slast_q [$];
  logic [31:0] lfsr_o, lfsr_s;
  always @(negedge clk) begin
    if (!rst_n) begin
      lfsr_o    <= seed;
      lfsr_s    <= seed ^ 32'h5A5A_00FF;
      out_ready <= 1'b0;
      scl_ready <= 1'b0;
    end else begin
      lfsr_o <= {lfsr_o[30:0], lfsr_o[31] ^ lfsr_o[21] ^ lfsr_o[1] ^ lfsr_o[0]};
      lfsr_s <= {lfsr_s[30:0], lfsr_s[31] ^ lfsr_s[21] ^ lfsr_s[1] ^ lfsr_s[0]};
      unique case (bp_mode)
        0: begin out_ready <= 1'b1;           scl_ready <= 1'b1;            end
        1: begin out_ready <= |lfsr_o[1:0];   scl_ready <= |lfsr_s[1:0];    end
        default: begin
           out_ready <= (lfsr_o[3:0] == 4'h0);
           scl_ready <= (lfsr_s[3:0] == 4'h0);
        end
      endcase
    end
  end
  always @(posedge clk) begin
    if (rst_n && out_valid && out_ready) begin
      got_beat_q.push_back(out_beat.data);
      got_blast_q.push_back(out_beat.last);
    end
    if (rst_n && scl_valid && scl_ready) begin
      got_scl_q.push_back(scl_data);
      got_slast_q.push_back(scl_last);
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
  logic [31:0] in_mem   [MAX_ELEMS];
  logic [15:0] scl_exp  [ROWS_MAX_TB];
  logic [63:0] beat_exp [MAX_BEATS];

  // ── coverage buckets (manual, run2 pattern) ────────────────────────────────
  int cov_rows_1, cov_rows_max, cov_rows_mid;
  int cov_zero_row, cov_eps_row, cov_scale_tie, cov_scale_bnd, cov_inf_scale;
  int cov_code_tie, cov_subnormal, cov_neg_zero;
  int cov_ill_rows0, cov_ill_rows_big;
  int cov_rst_ingest, cov_rst_scale, cov_rst_spush, cov_rst_drain;
  int cov_rst_wait, cov_rst_idle, cov_rst_random;
  int cov_bp_full, cov_bp_random, cov_bp_storm;
  int cov_stall_none, cov_stall_random, cov_stall_storm;
  int cov_cfg_d64, cov_cfg_d128;

  function automatic void cov_job(input int rows, fz, feps, stie, sbnd, sinf,
                                  ctie, fsub, fnegz);
    if (rows == 1)                    cov_rows_1++;
    else if (rows == int'(ROWS_MAX_TB)) cov_rows_max++;
    else                              cov_rows_mid++;
    if (fz    != 0) cov_zero_row++;
    if (feps  != 0) cov_eps_row++;
    if (stie  != 0) cov_scale_tie++;
    if (sbnd  != 0) cov_scale_bnd++;
    if (sinf  != 0) cov_inf_scale++;
    if (ctie  != 0) cov_code_tie++;
    if (fsub  != 0) cov_subnormal++;
    if (fnegz != 0) cov_neg_zero++;
  endfunction

  function automatic void print_coverage();
    $display("COV fq_rows_1 %0d",       cov_rows_1);
    $display("COV fq_rows_max %0d",     cov_rows_max);
    $display("COV fq_rows_mid %0d",     cov_rows_mid);
    $display("COV fq_zero_row %0d",     cov_zero_row);
    $display("COV fq_eps_row %0d",      cov_eps_row);
    $display("COV fq_scale_tie %0d",    cov_scale_tie);
    $display("COV fq_scale_bnd %0d",    cov_scale_bnd);
    $display("COV fq_inf_scale %0d",    cov_inf_scale);
    $display("COV fq_code_tie %0d",     cov_code_tie);
    $display("COV fq_subnormal %0d",    cov_subnormal);
    $display("COV fq_neg_zero %0d",     cov_neg_zero);
    $display("COV fq_ill_rows0 %0d",    cov_ill_rows0);
    $display("COV fq_ill_rows_big %0d", cov_ill_rows_big);
    $display("COV fq_rst_ingest %0d",   cov_rst_ingest);
    $display("COV fq_rst_scale %0d",    cov_rst_scale);
    $display("COV fq_rst_spush %0d",    cov_rst_spush);
    $display("COV fq_rst_drain %0d",    cov_rst_drain);
    $display("COV fq_rst_wait %0d",     cov_rst_wait);
    $display("COV fq_rst_idle %0d",     cov_rst_idle);
    $display("COV fq_rst_random %0d",   cov_rst_random);
    $display("COV fq_bp_full %0d",      cov_bp_full);
    $display("COV fq_bp_random %0d",    cov_bp_random);
    $display("COV fq_bp_storm %0d",     cov_bp_storm);
    $display("COV fq_stall_none %0d",   cov_stall_none);
    $display("COV fq_stall_random %0d", cov_stall_random);
    $display("COV fq_stall_storm %0d",  cov_stall_storm);
    $display("COV fq_cfg_d64 %0d",      cov_cfg_d64);
    $display("COV fq_cfg_d128 %0d",     cov_cfg_d128);
  endfunction

  // ── low-level drivers (negedge-driven) ─────────────────────────────────────
  int errors;
  bit exp_sticky;

  task automatic send_job(input logic [DIM_W-1:0] rows);
    @(negedge clk);
    job_rows  = rows;
    job_valid = 1'b1;
    while (!job_ready) @(negedge clk);
    @(negedge clk);
    job_valid = 1'b0;
  endtask

  task automatic feed_elems(input int num);
    for (int i = 0; i < num; i++) begin
      stall_gap();
      @(negedge clk);
      in_data  = in_mem[i];
      in_valid = 1'b1;
      while (!in_ready) @(negedge clk);
      @(negedge clk);
      in_valid = 1'b0;
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

  task automatic read_x_lines(input int n);
    string tg;
    logic [31:0] w;
    int cnt;
    for (int i = 0; i < n; i++) begin
      if (!next_line()) $fatal(1, "EOF inside X data (elem %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "X") $fatal(1, "bad X line: %s", cur_line);
      in_mem[i] = w;
    end
  endtask

  // ── job records ────────────────────────────────────────────────────────────
  int n_jobs, n_illegal, n_resets;

  task automatic check_results(input string name, input int rows);
    int nb;
    nb = rows * int'(BPR);
    if (got_scl_q.size() != rows) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: expected %0d scales, got %0d",
               name, cyc, rows, got_scl_q.size());
    end else begin
      for (int r = 0; r < rows; r++) begin
        if (got_scl_q[r] !== scl_exp[r]) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: scale %0d got %04x exp %04x",
                   name, cyc, r, got_scl_q[r], scl_exp[r]);
        end
        if (got_slast_q[r] !== bit'(r == rows - 1)) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: scale %0d last=%0b (exp %0b)",
                   name, cyc, r, got_slast_q[r], r == rows - 1);
        end
      end
    end
    if (got_beat_q.size() != nb) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: expected %0d beats, got %0d",
               name, cyc, nb, got_beat_q.size());
    end else begin
      for (int i = 0; i < nb; i++) begin
        if (got_beat_q[i] !== beat_exp[i]) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d\n  got %016x\n  exp %016x",
                   name, cyc, i, got_beat_q[i], beat_exp[i]);
        end
        if (got_blast_q[i] !== bit'(i == nb - 1)) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d last=%0b (exp %0b)",
                   name, cyc, i, got_blast_q[i], i == nb - 1);
        end
      end
    end
    got_beat_q.delete();
    got_blast_q.delete();
    got_scl_q.delete();
    got_slast_q.delete();
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
    int rows, fz, feps, stie, sbnd, sinf, ctie, fsub, fnegz;
    int d0, cnt;
    string unused_tag, name;
    logic [15:0] w16;
    logic [63:0] w64;
    string tg;
    cnt = $sscanf(hdr, "%s %d %d %d %d %d %d %d %d %d", unused_tag,
                  rows, fz, feps, stie, sbnd, sinf, ctie, fsub, fnegz);
    if (cnt != 10) $fatal(1, "bad F line: %s", hdr);
    read_x_lines(rows * int'(D_CFG));
    for (int r = 0; r < rows; r++) begin
      if (!next_line()) $fatal(1, "EOF inside S data");
      cnt = $sscanf(cur_line, "%s %h", tg, w16);
      if (cnt != 2 || tg != "S") $fatal(1, "bad S line: %s", cur_line);
      scl_exp[r] = w16;
    end
    for (int i = 0; i < rows * int'(BPR); i++) begin
      if (!next_line()) $fatal(1, "EOF inside E data");
      cnt = $sscanf(cur_line, "%s %h", tg, w64);
      if (cnt != 2 || tg != "E") $fatal(1, "bad E line: %s", cur_line);
      beat_exp[i] = w64;
    end
    cov_job(rows, fz, feps, stie, sbnd, sinf, ctie, fsub, fnegz);
    n_jobs++;
    name = $sformatf("F%0d rows=%0d", n_jobs, rows);

    d0 = done_count;
    send_job(DIM_W'(rows));
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
    feed_elems(rows * int'(D_CFG));
    wait_done_rel(d0, name);
    if (done_count != d0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done_count=%0d (exp %0d)",
               name, cyc, done_count, d0 + 1);
    end
    check_results(name, rows);
    post_job_checks(name);
  endtask

  task automatic do_illegal(input string hdr);
    int rows, e0, d0, cnt;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d", unused_tag, rows);
    if (cnt != 2) $fatal(1, "bad I line: %s", hdr);
    if (rows == 0) cov_ill_rows0++; else cov_ill_rows_big++;
    n_illegal++;
    name = $sformatf("I%0d rows=%0d", n_illegal, rows);

    e0 = err_pulses;
    d0 = done_count;
    send_job(DIM_W'(rows));
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
    if (got_beat_q.size() != 0 || got_scl_q.size() != 0) begin
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
    int rows, abortcyc, tphase, cnt, phase, t;
    string unused_tag, name;
    cnt = $sscanf(hdr, "%s %d %d %d", unused_tag, rows, abortcyc, tphase);
    if (cnt != 4) $fatal(1, "bad R line: %s", hdr);
    read_x_lines(rows * int'(D_CFG));
    n_resets++;
    name = $sformatf("R%0d rows=%0d abort=%0d phase=%0d",
                     n_resets, rows, abortcyc, tphase);

    send_job(DIM_W'(rows));
    fork
      begin
        feed_elems(rows * int'(D_CFG));
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
      1:       cov_rst_ingest++;
      2:       cov_rst_scale++;
      3:       cov_rst_spush++;
      4:       cov_rst_drain++;
      5:       cov_rst_wait++;
      default: cov_rst_idle++;
    endcase
    if (tphase == 0) cov_rst_random++;
    $display("RESET [%s] cyc=%0d: mid-op reset in FSM phase %0d",
             name, cyc, phase);

    rst_n     = 1'b0;
    in_valid  = 1'b0;
    job_valid = 1'b0;
    repeat (3) @(negedge clk);
    rst_n      = 1'b1;
    exp_sticky = 1'b0;
    got_beat_q.delete();
    got_blast_q.delete();
    got_scl_q.delete();
    got_slast_q.delete();
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
    if (got_beat_q.size() != 0 || got_scl_q.size() != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: stray output after mid-op reset",
               name, cyc);
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
    job_rows   = '0;
    in_valid   = 1'b0;
    in_data    = '0;
    n_jobs     = 0;
    n_illegal  = 0;
    n_resets   = 0;

    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'hFEED_4EA1;
    unique case (bp_mode)
      0: cov_bp_full++; 1: cov_bp_random++; default: cov_bp_storm++;
    endcase
    unique case (stall_mode)
      0: cov_stall_none++; 1: cov_stall_random++; default: cov_stall_storm++;
    endcase
    if (D_CFG == 64) cov_cfg_d64++; else cov_cfg_d128++;

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
      $display("TB PASS: %0d legal jobs, %0d illegal jobs, %0d mid-op resets, 0 errors (D=%0d bp=%0d stall=%0d seed=%0d)",
               n_jobs, n_illegal, n_resets, D_CFG, bp_mode, stall_mode, seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s) across %0d jobs", errors, n_jobs);
    end
  end

endmodule
