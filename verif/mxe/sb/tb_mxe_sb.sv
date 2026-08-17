// tb_mxe_sb.sv — MXE vector-driven scoreboard testbench (the proven V0
// pattern): stimulus + bit-exact expected results come from gen_mxe_vectors.py
// (golden arbiter apex_golden.compute), the TB only drives, collects and
// compares. Verilator --binary --timing --assert.
//
// Vector file records (see gen_mxe_vectors.py header for the format):
//   J  legal job  — drive descriptor + data, wait done, compare M result
//                   beats bit-exact, check busy/done/desc_ready/sticky.
//   I  illegal descriptor — expect exactly one desc_error pulse + sticky,
//                   ZERO side effects (no busy, no done, no beats) — §3 and
//                   the run1/D-019 regression (next legal job must be clean).
//   X  mid-operation reset — drive the job, assert rst_n mid-flight at the
//                   given cycle, verify clean recovery (the run2 hole).
//
// Protocol adversaries (plusargs):
//   +bp_mode=0|1|2     result-stream backpressure: none / ~75% duty / storm
//   +stall_mode=0|1|2  act+wgt feed gaps: none / short random / bursty storm
//   +seed=<n>          LFSR seed for the consumer + stall randomization
//
// The §5/D-006 contract is additionally checked every cycle by the bound
// apex_stream_sva property pack (verif/common/apex_stream_sva.svh).
// All FAIL messages carry the cycle number (cyc) for waveform correlation;
// build with `make waves` + run with +dump for an FST trace.
//
// TB discipline (V0/smoke lineage): stimulus and handshake decisions at
// NEGEDGE, commits observed at posedge; $fatal watchdog bounds the run.

`include "apex_stream_sva.svh"

module tb_mxe_sb;

  import apex_pkg::*;

  localparam int unsigned MAX_ACT     = 16384;  // M_TILE_MAX * KB_MAX
  localparam int unsigned MAX_WGT     = 2048;   // KB_MAX * MXE_N
  localparam int unsigned MAX_M       = 64;
  localparam int unsigned JOB_TIMEOUT = 300000;

  // ── clock / cycle counter ──────────────────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;   // 2-state, starts at 0
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

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

  // §5/D-006 contract pack, bound into the DUT (compiled under --assert)
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

  // ── plusarg config ─────────────────────────────────────────────────────────
  int unsigned bp_mode      = 1;
  int unsigned stall_mode   = 1;
  int unsigned seed         = 32'hACE1_2026;
  int unsigned watchdog_cyc = 8_000_000;
  initial begin
    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
  end

  // ── watchdog / optional trace ──────────────────────────────────────────────
  initial begin
    @(posedge clk);          // let plusargs initial run first
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: simulation did not finish within %0d cycles",
           watchdog_cyc);
  end

  initial begin
    if ($test$plusargs("dump")) begin
      $dumpfile("dump.fst");
      $dumpvars(0, tb_mxe_sb);
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
      if (done)       done_count <= done_count + 1;
      if (desc_error) err_pulses <= err_pulses + 1;
    end
  end

  // result consumer with configurable backpressure
  logic [255:0] got_q [$];
  bit           got_last_q [$];
  logic [31:0]  lfsr;
  always @(negedge clk) begin
    if (!rst_n) begin
      lfsr      <= seed;
      res_ready <= 1'b0;
    end else begin
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      unique case (bp_mode)
        0:       res_ready <= 1'b1;                 // no backpressure
        1:       res_ready <= |lfsr[1:0];           // ~75% duty
        default: res_ready <= (lfsr[3:0] == 4'h0);  // storm: ~6% duty
      endcase
    end
  end
  always @(posedge clk) begin
    if (rst_n && res_valid && res_ready) begin
      got_q.push_back(res_beat.data);
      got_last_q.push_back(res_beat.last);
    end
  end

  // ── TB-side randomization (stall gaps) ─────────────────────────────────────
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

  // ── stimulus storage (per job, loaded from the vector file) ───────────────
  logic [63:0]  act_mem [MAX_ACT];
  logic [63:0]  wgt_mem [MAX_WGT];
  logic [255:0] exp_mem [MAX_M];

  // ── coverage buckets (manual, run2 pattern) ───────────────────────────────
  int cov_mode_ws, cov_mode_os, cov_rq_on, cov_rq_off;
  int cov_os_acc0, cov_os_acc1;
  int cov_m_eq_1, cov_m_eq_64, cov_m_mid;
  int cov_k_eq_1, cov_k_mult8, cov_k_tail, cov_k_eq_2048;
  int cov_n_eq_1, cov_n_eq_8, cov_n_mid;
  int cov_sat_pos, cov_sat_neg, cov_rne_tie;
  int cov_ill_opcode, cov_ill_k0, cov_ill_kbig, cov_ill_m0, cov_ill_mbig;
  int cov_ill_n0, cov_ill_nbig, cov_ill_buf;
  int cov_rst_ingest, cov_rst_wload, cov_rst_compute, cov_rst_flush;
  int cov_rst_drain, cov_rst_wait_done, cov_rst_idle;
  int cov_bp_full, cov_bp_random, cov_bp_storm;
  int cov_stall_none, cov_stall_random, cov_stall_storm;

  function automatic void cov_job(input int op, m, k, n, acc, rq,
                                  input int ftie, fsatp, fsatn);
    if (op == int'(OP_GEMM_WS)) cov_mode_ws++;
    if (op == int'(OP_GEMM_OS)) begin
      cov_mode_os++;
      if (acc != 0) cov_os_acc1++; else cov_os_acc0++;
    end
    if (rq != 0) cov_rq_on++; else cov_rq_off++;
    if (m == 1) cov_m_eq_1++; else if (m == 64) cov_m_eq_64++; else cov_m_mid++;
    if (k == 1) cov_k_eq_1++;
    if (k == 2048) cov_k_eq_2048++;
    if (k % 8 == 0) cov_k_mult8++; else cov_k_tail++;
    if (n == 1) cov_n_eq_1++; else if (n == 8) cov_n_eq_8++; else cov_n_mid++;
    if (ftie  != 0) cov_rne_tie++;
    if (fsatp != 0) cov_sat_pos++;
    if (fsatn != 0) cov_sat_neg++;
  endfunction

  function automatic void cov_illegal(input int op, m, k, n, ba, bb, bd);
    if (op != int'(OP_GEMM_WS) && op != int'(OP_GEMM_OS)) cov_ill_opcode++;
    if (k == 0)    cov_ill_k0++;
    if (k > 2048)  cov_ill_kbig++;
    if (m == 0)    cov_ill_m0++;
    if (m > 64)    cov_ill_mbig++;
    if (n == 0)    cov_ill_n0++;
    if (n > 8)     cov_ill_nbig++;
    if (ba != 0 || bb != 0 || bd != 0) cov_ill_buf++;
  endfunction

  function automatic void cov_reset_phase(input int st);
    unique case (st)
      1:       cov_rst_ingest++;
      2:       cov_rst_wload++;
      3:       cov_rst_compute++;
      4:       cov_rst_flush++;
      5:       cov_rst_drain++;
      6:       cov_rst_wait_done++;
      default: cov_rst_idle++;
    endcase
  endfunction

  function automatic void print_coverage();
    $display("COV mode_ws %0d",       cov_mode_ws);
    $display("COV mode_os %0d",       cov_mode_os);
    $display("COV rq_on %0d",         cov_rq_on);
    $display("COV rq_off %0d",        cov_rq_off);
    $display("COV os_acc0 %0d",       cov_os_acc0);
    $display("COV os_acc1 %0d",       cov_os_acc1);
    $display("COV m_eq_1 %0d",        cov_m_eq_1);
    $display("COV m_eq_64 %0d",       cov_m_eq_64);
    $display("COV m_mid %0d",         cov_m_mid);
    $display("COV k_eq_1 %0d",        cov_k_eq_1);
    $display("COV k_mult8 %0d",       cov_k_mult8);
    $display("COV k_tail %0d",        cov_k_tail);
    $display("COV k_eq_2048 %0d",     cov_k_eq_2048);
    $display("COV n_eq_1 %0d",        cov_n_eq_1);
    $display("COV n_eq_8 %0d",        cov_n_eq_8);
    $display("COV n_mid %0d",         cov_n_mid);
    $display("COV sat_pos %0d",       cov_sat_pos);
    $display("COV sat_neg %0d",       cov_sat_neg);
    $display("COV rne_tie %0d",       cov_rne_tie);
    $display("COV ill_opcode %0d",    cov_ill_opcode);
    $display("COV ill_k0 %0d",        cov_ill_k0);
    $display("COV ill_kbig %0d",      cov_ill_kbig);
    $display("COV ill_m0 %0d",        cov_ill_m0);
    $display("COV ill_mbig %0d",      cov_ill_mbig);
    $display("COV ill_n0 %0d",        cov_ill_n0);
    $display("COV ill_nbig %0d",      cov_ill_nbig);
    $display("COV ill_buf %0d",       cov_ill_buf);
    $display("COV rst_ingest %0d",    cov_rst_ingest);
    $display("COV rst_wload %0d",     cov_rst_wload);
    $display("COV rst_compute %0d",   cov_rst_compute);
    $display("COV rst_flush %0d",     cov_rst_flush);
    $display("COV rst_drain %0d",     cov_rst_drain);
    $display("COV rst_wait_done %0d", cov_rst_wait_done);
    $display("COV rst_idle %0d",      cov_rst_idle);
    $display("COV bp_full %0d",       cov_bp_full);
    $display("COV bp_random %0d",     cov_bp_random);
    $display("COV bp_storm %0d",      cov_bp_storm);
    $display("COV stall_none %0d",    cov_stall_none);
    $display("COV stall_random %0d",  cov_stall_random);
    $display("COV stall_storm %0d",   cov_stall_storm);
  endfunction

  // ── low-level drivers (negedge-driven, V0/smoke lineage) ──────────────────
  int errors;
  bit exp_sticky;   // TB's model of desc_error_sticky (cleared only by reset)

  function automatic mxe_desc_t mk_desc(input int op, m, k, n, acc, rq,
                                        input int scale, shft, ba, bb, bd);
    mxe_desc_t d;
    // header parse fields are int-typed for $sscanf %d; only the
    // descriptor-width low bits are meaningful (consume the rest for lint)
    logic unused_hdr_hi;
    unused_hdr_hi = &{1'b0, m[31:12], k[31:12], n[31:12], scale[31:16],
                      shft[31:5], ba[31:4], bb[31:4], bd[31:4]};
    d            = '0;
    d.opcode     = 8'(op);
    d.m_dim      = 12'(m);
    d.k_dim      = 12'(k);
    d.n_dim      = 12'(n);
    d.mode_os    = (op == int'(OP_GEMM_OS));
    d.accumulate = (acc != 0);
    d.requant_en = (rq != 0);
    d.rq_scale   = 16'(scale);
    d.rq_shift   = 5'(shft);
    d.src_a_buf  = 4'(ba);
    d.src_b_buf  = 4'(bb);
    d.dst_buf    = 4'(bd);
    return d;
  endfunction

  task automatic send_desc(input mxe_desc_t d);
    @(negedge clk);
    desc       = d;
    desc_valid = 1'b1;
    while (!desc_ready) @(negedge clk);
    @(negedge clk);                       // cross the committing posedge
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

  task automatic feed_act(input int num);
    for (int i = 0; i < num; i++) begin
      stall_gap();
      send_act(act_mem[i], i == num - 1);
    end
  endtask

  task automatic feed_wgt(input int num);
    for (int i = 0; i < num; i++) begin
      stall_gap();
      send_wgt(wgt_mem[i], i == num - 1);
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

  task automatic read_data_lines(input int na, input int nw);
    string tg;
    logic [63:0] w;
    int cnt;
    for (int i = 0; i < na; i++) begin
      if (!next_line()) $fatal(1, "EOF inside A data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "A")
        $fatal(1, "bad A line (beat %0d): %s", i, cur_line);
      act_mem[i] = w;
    end
    for (int i = 0; i < nw; i++) begin
      if (!next_line()) $fatal(1, "EOF inside W data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, w);
      if (cnt != 2 || tg != "W")
        $fatal(1, "bad W line (beat %0d): %s", i, cur_line);
      wgt_mem[i] = w;
    end
  endtask

  task automatic read_exp_lines(input int m);
    string tg;
    logic [31:0] lane [8];
    int cnt;
    for (int i = 0; i < m; i++) begin
      if (!next_line()) $fatal(1, "EOF inside E data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h %h %h %h %h %h %h %h", tg,
                    lane[0], lane[1], lane[2], lane[3],
                    lane[4], lane[5], lane[6], lane[7]);
      if (cnt != 9 || tg != "E")
        $fatal(1, "bad E line (beat %0d): %s", i, cur_line);
      for (int c = 0; c < 8; c++) exp_mem[i][32*c +: 32] = lane[c];
    end
  endtask

  // ── job records ────────────────────────────────────────────────────────────
  int n_jobs, n_illegal, n_resets;

  task automatic check_results(input string name, input int m);
    if (got_q.size() != m) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: expected %0d result beats, got %0d",
               name, cyc, m, got_q.size());
    end else begin
      for (int i = 0; i < m; i++) begin
        if (got_q[i] !== exp_mem[i]) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d\n  got %064x\n  exp %064x",
                   name, cyc, i, got_q[i], exp_mem[i]);
        end
        if (got_last_q[i] !== bit'(i == m - 1)) begin
          errors++;
          $display("FAIL [%s] cyc=%0d: beat %0d last=%0b (exp %0b)",
                   name, cyc, i, got_last_q[i], i == m - 1);
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
    if (!desc_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: desc_ready not restored after done",
               name, cyc);
    end
    if (desc_error_sticky !== exp_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky=%0b expected %0b",
               name, cyc, desc_error_sticky, exp_sticky);
    end
  endtask

  task automatic do_legal(input string hdr);
    int op, m, k, n, acc, rq, scale, shft, ba, bb, bd, ftie, fsp, fsn;
    int kb, na, nw, d0, cnt;
    string unused_tag, name;   // tag already dispatched by caller
    cnt = $sscanf(hdr, "%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d", unused_tag,
                  op, m, k, n, acc, rq, scale, shft, ba, bb, bd,
                  ftie, fsp, fsn);
    if (cnt != 15) $fatal(1, "bad J line: %s", hdr);
    kb = (k + 7) / 8;
    na = m * kb;
    nw = kb * n;
    read_data_lines(na, nw);
    read_exp_lines(m);
    cov_job(op, m, k, n, acc, rq, ftie, fsp, fsn);
    n_jobs++;
    name = $sformatf("J%0d %0dx%0dx%0d op=%0d acc=%0d rq=%0d",
                     n_jobs, m, k, n, op, acc, rq);

    d0 = done_count;
    send_desc(mk_desc(op, m, k, n, acc, rq, scale, shft, ba, bb, bd));
    @(negedge clk);
    if (desc_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: desc_ready high while job in flight",
               name, cyc);
    end
    if (!busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy low right after descriptor accept",
               name, cyc);
    end
    fork
      feed_act(na);
      feed_wgt(nw);
    join
    wait_done_rel(d0, name);
    if (done_count != d0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done_count=%0d (exp %0d)",
               name, cyc, done_count, d0 + 1);
    end
    check_results(name, m);
    post_job_checks(name);
  endtask

  task automatic do_illegal(input string hdr);
    int op, m, k, n, acc, rq, scale, shft, ba, bb, bd;
    int e0, d0, cnt;
    string unused_tag, name;   // tag already dispatched by caller
    cnt = $sscanf(hdr, "%s %d %d %d %d %d %d %d %d %d %d %d", unused_tag,
                  op, m, k, n, acc, rq, scale, shft, ba, bb, bd);
    if (cnt != 12) $fatal(1, "bad I line: %s", hdr);
    cov_illegal(op, m, k, n, ba, bb, bd);
    n_illegal++;
    name = $sformatf("I%0d op=%0d %0dx%0dx%0d buf=%0d%0d%0d",
                     n_illegal, op, m, k, n, ba, bb, bd);

    e0 = err_pulses;
    d0 = done_count;
    send_desc(mk_desc(op, m, k, n, acc, rq, scale, shft, ba, bb, bd));
    repeat (3) @(negedge clk);
    exp_sticky = 1'b1;
    if (err_pulses != e0 + 1) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: err_pulses=%0d (exp %0d)",
               name, cyc, err_pulses, e0 + 1);
    end
    if (!desc_error_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky bit not set", name, cyc);
    end
    if (done_count != d0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: done pulsed for illegal descriptor",
               name, cyc);
    end
    if (got_q.size() != 0) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: %0d result beat(s) from illegal descriptor",
               name, cyc, got_q.size());
    end
    if (busy) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: busy after illegal descriptor", name, cyc);
    end
    if (!desc_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: desc_ready lost after reject (D-019 gate)",
               name, cyc);
    end
  endtask

  task automatic do_reset_job(input string hdr);
    int op, m, k, n, acc, rq, scale, shft, ba, bb, bd, abortcyc, tphase;
    int kb, na, nw, cnt, phase, t;
    string unused_tag, name;   // tag already dispatched by caller
    cnt = $sscanf(hdr, "%s %d %d %d %d %d %d %d %d %d %d %d %d %d", unused_tag,
                  op, m, k, n, acc, rq, scale, shft, ba, bb, bd,
                  abortcyc, tphase);
    if (cnt != 14) $fatal(1, "bad X line: %s", hdr);
    kb = (k + 7) / 8;
    na = m * kb;
    nw = kb * n;
    read_data_lines(na, nw);
    n_resets++;
    name = $sformatf("X%0d %0dx%0dx%0d abort=%0d phase=%0d",
                     n_resets, m, k, n, abortcyc, tphase);

    send_desc(mk_desc(op, m, k, n, acc, rq, scale, shft, ba, bb, bd));
    fork
      begin
        fork
          feed_act(na);
          begin
            // D-005 load-under-compute: the weight loader consumes beats
            // under S_INGEST, so S_WLOAD (now a wait-for-shadow-bank state)
            // only occurs when the weight stream LAGS. For a phase-2 target,
            // withhold the weight feed until the ctrl is observed waiting
            // there — the reset then fires in a real S_WLOAD cycle.
            if (tphase == 2)
              while (int'(dut.u_ctrl.state) != 2) @(negedge clk);
            feed_wgt(nw);
          end
        join
        forever @(negedge clk);   // hold the branch open until abort fires
      end
      begin
        if (tphase != 0) begin
          // exact phase targeting: reset the moment the ctrl FSM is observed
          // in the requested state (1=S_INGEST .. 6=S_WAIT_DONE)
          t = 0;
          while (int'(dut.u_ctrl.state) != tphase) begin
            @(negedge clk);
            t++;
            if (t > int'(JOB_TIMEOUT))
              $fatal(1, "FAIL [%s]: ctrl never reached phase %0d (cyc=%0d)",
                     name, tphase, cyc);
          end
        end else begin
          repeat (abortcyc) @(negedge clk);
        end
      end
    join_any
    disable fork;

    phase = int'(dut.u_ctrl.state);       // record which phase we hit
    cov_reset_phase(phase);
    $display("RESET [%s] cyc=%0d: mid-op reset in ctrl phase %0d",
             name, cyc, phase);

    // mid-operation reset (synchronous, active-low)
    rst_n      = 1'b0;
    act_valid  = 1'b0;
    wgt_valid  = 1'b0;
    desc_valid = 1'b0;
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
    if (desc_error_sticky) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: sticky not cleared by reset", name, cyc);
    end
    if (!desc_ready) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: desc_ready not restored after reset",
               name, cyc);
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
    desc_valid = 1'b0;
    desc       = '0;
    act_valid  = 1'b0;
    act_beat   = '0;
    wgt_valid  = 1'b0;
    wgt_beat   = '0;
    n_jobs     = 0;
    n_illegal  = 0;
    n_resets   = 0;

    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'hDEAD_4EA1;
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
    if (desc_error_sticky) begin
      errors++;
      $display("FAIL [reset] cyc=%0d: sticky error out of reset", cyc);
    end

    while (next_line()) begin
      tag0 = cur_line.getc(0);
      unique case (tag0)
        "J":     do_legal(cur_line);
        "I":     do_illegal(cur_line);
        "X":     do_reset_job(cur_line);
        default: $fatal(1, "unexpected record: %s", cur_line);
      endcase
    end
    $fclose(fd);

    repeat (5) @(negedge clk);
    print_coverage();
    if (errors == 0) begin
      $display("TB PASS: %0d legal jobs, %0d illegal descriptors, %0d mid-op resets, 0 errors (bp=%0d stall=%0d seed=%0d)",
               n_jobs, n_illegal, n_resets, bp_mode, stall_mode, seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s) across %0d jobs", errors, n_jobs);
    end
  end

endmodule
