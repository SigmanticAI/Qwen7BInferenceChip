// tb_w4b_sb.sv — scoreboard TB for mxe_wfeed_w4b (D-031).
// Expected bytes from gen_w4b_vectors.py = golden wfeed_w4b_to_i8 only.
// Records: F (job w/ gs sideband + packed in + expected out), I (illegal),
// R (mid-op reset). Adversaries: +bp_mode out backpressure 0/1/2,
// +stall_mode packed-feed gaps, +gs_mode 0=upfront 1=late-trickle
// 2=interleaved. All modes must be bit-identical (timing-only).
`timescale 1ns/1ps

module tb_w4b_sb;

  import apex_pkg::*;

  localparam int unsigned BEATS_MAX_TB = 2048;
  localparam int unsigned NG_MAX_TB    = 512;
  localparam int unsigned ROW_TIMEOUT  = 400000;

  logic clk, rst_n;
  int unsigned cyc;
  initial begin clk = 0; forever #5 clk = ~clk; end
  always @(posedge clk) cyc <= cyc + 1;

  // ── DUT ──────────────────────────────────────────────────────────────────
  logic        w4b_en;
  logic        job_valid, job_ready, job_error, job_error_sticky;
  logic [15:0] job_beats, job_s8;
  logic [11:0] job_k;
  logic [3:0]  job_n;
  logic        pw_valid, pw_ready;
  lane8_beat_t pw_beat;
  logic        gs_valid, gs_ready;
  logic [15:0] gs_data;
  logic        xw_valid, xw_ready;
  lane8_beat_t xw_beat;
  logic        busy, done;

  mxe_wfeed_w4b #(.G(32)) dut (.*);

  // ── plusargs / watchdog ──────────────────────────────────────────────────
  int unsigned bp_mode = 1, stall_mode = 1, gs_mode = 0, seed = 32'hD031;
  initial begin
    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
    void'($value$plusargs("gs_mode=%d", gs_mode));
    void'($value$plusargs("seed=%d", seed));
  end
  initial begin
    repeat (30_000_000) @(posedge clk);
    $fatal(1, "WATCHDOG");
  end

  // ── monitors ─────────────────────────────────────────────────────────────
  int done_count, err_pulses, errors;
  logic xw_ready_block;
  always @(posedge clk) begin
    if (!rst_n) begin done_count <= 0; err_pulses <= 0; end
    else begin
      if (done)      done_count <= done_count + 1;
      if (job_error) err_pulses <= err_pulses + 1;
    end
  end

  logic [63:0] got_q [$];
  logic [31:0] lfsr;
  always @(negedge clk) begin
    if (!rst_n) begin lfsr <= seed | 1; xw_ready <= 0; end
    else begin
      lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
      if (xw_ready_block) xw_ready <= 1'b0;
      else unique case (bp_mode)
        0: xw_ready <= 1'b1;
        1: xw_ready <= |lfsr[1:0];
        default: xw_ready <= (lfsr[3:0] == 0);
      endcase
    end
  end
  always @(posedge clk)
    if (rst_n && xw_valid && xw_ready) got_q.push_back(xw_beat.data);

  // D-006: done only after all beats accepted post-skid
  int exp_beats_at_done, d006_fails;
  always @(posedge clk) begin
    if (!rst_n) d006_fails <= 0;
    else if (done && (got_q.size() != exp_beats_at_done)) begin
      d006_fails <= d006_fails + 1;
      $display("FAIL [D-006] cyc=%0d: done with %0d/%0d beats",
               cyc, got_q.size(), exp_beats_at_done);
    end
  end

  // ── rng / gaps ───────────────────────────────────────────────────────────
  int unsigned rng_state = 32'hC0DE_D031;
  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction
  task automatic gap(input int unsigned mode);
    int unsigned g;
    unique case (mode)
      0: g = 0;
      1: g = ((rnd() & 1) != 0) ? (rnd() % 4) : 0;
      default: g = ((rnd() % 8) == 0) ? (rnd() % 24) : (rnd() % 3);
    endcase
    repeat (g) @(negedge clk);
  endtask

  // ── stimulus memories ────────────────────────────────────────────────────
  logic [63:0] p_mem [BEATS_MAX_TB];
  logic [63:0] e_mem [BEATS_MAX_TB];
  logic [15:0] s_mem [NG_MAX_TB];
  int n_pk, n_ex, n_sc;

  // coverage
  int cov_odd, cov_cmin, cov_cmax, cov_czero, cov_cneg, cov_satr,
      cov_pass, cov_illegal, cov_rst_run, cov_rst_drain, cov_rst_cyc,
      cov_gs_late_stall;

  // ── drivers ──────────────────────────────────────────────────────────────
  task automatic send_pw(input logic [63:0] w);
    @(negedge clk);
    pw_beat       = '0;
    pw_beat.data  = w;
    pw_valid      = 1;
    while (!pw_ready) @(negedge clk);
    @(negedge clk);
    pw_valid = 0;
  endtask
  task automatic send_gs(input logic [15:0] s);
    @(negedge clk);
    gs_data  = s;
    gs_valid = 1;
    while (!gs_ready) @(negedge clk);
    @(negedge clk);
    gs_valid = 0;
  endtask

  task automatic feed_pw(input int n);
    for (int i = 0; i < n; i++) begin
      gap(stall_mode);
      send_pw(p_mem[i]);
    end
  endtask
  task automatic feed_gs(input int n);
    for (int i = 0; i < n; i++) begin
      unique case (gs_mode)
        0: ;                                    // upfront: pump ASAP
        1: repeat (rnd() % 40) @(negedge clk);  // late trickle (stalls DUT)
        default: repeat (rnd() % 6) @(negedge clk);
      endcase
      send_gs(s_mem[i]);
    end
  endtask

  task automatic pulse_job(input logic en, input int beats, input int K,
                           input int N, input logic [15:0] s8);
    @(negedge clk);
    w4b_en    = en;
    job_beats = 16'(beats);
    job_k     = 12'(K);
    job_n     = 4'(N);
    job_s8    = s8;
    job_valid = 1;
    while (!job_ready) @(negedge clk);
    @(negedge clk);
    job_valid = 0;
  endtask

  // ── vector reader ────────────────────────────────────────────────────────
  int fd, n_jobs, n_illegal, n_resets;
  string line;
  function automatic bit next_line();
    while ($fgets(line, fd) > 0)
      if (line.len() > 1 && line.getc(0) != "#") return 1;
    return 0;
  endfunction

  task automatic read_sp(input int nsc, input int npk, input int nex);
    string tg; logic [63:0] w; int c;
    for (int i = 0; i < nsc; i++) begin
      void'(next_line()); c = $sscanf(line, "%s %h", tg, w);
      if (c != 2 || tg != "S") $fatal(1, "bad S: %s", line);
      s_mem[i] = w[15:0];
    end
    for (int i = 0; i < npk; i++) begin
      void'(next_line()); c = $sscanf(line, "%s %h", tg, w);
      if (c != 2 || tg != "P") $fatal(1, "bad P: %s", line);
      p_mem[i] = w;
    end
    for (int i = 0; i < nex; i++) begin
      void'(next_line()); c = $sscanf(line, "%s %h", tg, w);
      if (c != 2 || tg != "E") $fatal(1, "bad E: %s", line);
      e_mem[i] = w;
    end
    n_sc = nsc; n_pk = npk; n_ex = nex;
  endtask

  task automatic do_job(input string hdr);
    int beats, en, K, N, odd, cmin, cmax, czero, cneg, satr, c, d0;
    logic [31:0] s8w;
    string tg, name;
    c = $sscanf(hdr, "%s %d %d %d %d %h %d %d %d %d %d %d",
                tg, beats, en, K, N, s8w, odd, cmin, cmax, czero, cneg, satr);
    if (c != 12) $fatal(1, "bad F: %s", hdr);
    read_sp(en ? nsc_of(K, N) : 0,
            en ? (beats + 1) / 2 : beats, beats);
    n_jobs++;
    name = $sformatf("F%0d K=%0d N=%0d en=%0d", n_jobs, K, N, en);
    cov_odd += odd; cov_cmin += cmin; cov_cmax += cmax;
    cov_czero += czero; cov_cneg += cneg; cov_satr += satr;
    if (!en) cov_pass++;

    d0 = done_count;
    exp_beats_at_done = beats;
    pulse_job(en[0], beats, K, N, s8w[15:0]);
    fork
      feed_pw(n_pk);
      if (en) feed_gs(n_sc);
    join
    begin
      int t = 0;
      while (done_count == d0) begin
        @(negedge clk); t++;
        if (t > ROW_TIMEOUT) $fatal(1, "FAIL [%s]: timeout", name);
      end
    end
    if (got_q.size() != beats) begin
      errors++;
      $display("FAIL [%s] cyc=%0d: %0d beats, exp %0d",
               name, cyc, got_q.size(), beats);
    end else begin
      for (int i = 0; i < beats; i++)
        if (got_q[i] !== e_mem[i]) begin
          errors++;
          if (errors < 20)
            $display("FAIL [%s] beat %0d: got %016x exp %016x",
                     name, i, got_q[i], e_mem[i]);
        end
    end
    got_q.delete();
    repeat (2) @(negedge clk);
    if (busy) begin errors++; $display("FAIL [%s]: busy stuck", name); end
  endtask

  function automatic int nsc_of(input int K, input int N);
    return N * ((K + 31) / 32);          // G=32
  endfunction

  task automatic do_illegal(input string hdr);
    int beats, K, N, c, e0;
    string tg;
    c = $sscanf(hdr, "%s %d %d %d", tg, beats, K, N);
    if (c != 4) $fatal(1, "bad I: %s", hdr);
    n_illegal++; cov_illegal++;
    e0 = err_pulses;
    pulse_job(1'b1, beats, K, N, 16'h3C00);
    repeat (5) @(negedge clk);
    if (err_pulses != e0 + 1) begin
      errors++; $display("FAIL [I beats=%0d]: no reject pulse", beats);
    end
    if (busy) begin errors++; $display("FAIL [I]: state changed"); end
  endtask

  task automatic do_reset(input string hdr);
    int beats, en, K, N, abortc, tphase, c, fed;
    logic [31:0] s8w;
    string tg;
    c = $sscanf(hdr, "%s %d %d %d %d %h %d %d",
                tg, beats, en, K, N, s8w, abortc, tphase);
    if (c != 8) $fatal(1, "bad R: %s", hdr);
    read_sp(en ? nsc_of(K, N) : 0, en ? (beats + 1) / 2 : beats, 0);
    n_resets++;
    pulse_job(en[0], beats, K, N, s8w[15:0]);
    fork
      begin
        fork
          feed_pw(n_pk);
          if (en) feed_gs(n_sc);
        join
        forever @(negedge clk);
      end
      begin
        if (tphase == 2) begin
          // DRAIN is transient but persists while the FIFO backlog drains
          // (>= backlog cycles under bp); do NOT block the sink — a blocked
          // sink exhausts issue credit and the FSM can never REACH drain
          wait (dut.st == 2);
        end else if (tphase == 1) begin
          wait (dut.st == 1);
          repeat (rnd() % 30) @(negedge clk);
        end else begin
          repeat (abortc) @(negedge clk);
        end
      end
    join_any
    disable fork;
    if (tphase == 1) cov_rst_run++;
    else if (tphase == 2) cov_rst_drain++;
    else cov_rst_cyc++;

    rst_n = 0; job_valid = 0; pw_valid = 0; gs_valid = 0;
    xw_ready_block = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    got_q.delete();
    repeat (6) @(negedge clk);
    if (busy) begin errors++; $display("FAIL [Z]: busy after reset"); end
    if (got_q.size() != 0) begin
      errors++; $display("FAIL [Z]: stray beats after reset");
    end
  endtask

  // ── main ─────────────────────────────────────────────────────────────────
  initial begin : main
    string vec, tag;
    rst_n = 0; errors = 0; xw_ready_block = 0;
    job_valid = 0; pw_valid = 0; gs_valid = 0;
    w4b_en = 0; job_beats = 0; job_k = 0; job_n = 0; job_s8 = 0;
    pw_beat = '0; gs_data = 0;
    exp_beats_at_done = 0;
    if (!$value$plusargs("vectors=%s", vec)) $fatal(1, "no +vectors");
    rng_state = seed ^ 32'hD031_5EED;
    fd = $fopen(vec, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec);
    repeat (5) @(negedge clk);
    rst_n = 1;
    repeat (2) @(negedge clk);

    while (next_line()) begin
      tag = line.substr(0, 0);
      unique case (tag)
        "F": do_job(line);
        "I": do_illegal(line);
        "R": do_reset(line);
        default: $fatal(1, "bad record: %s", line);
      endcase
    end
    $fclose(fd);
    repeat (5) @(negedge clk);
    errors += d006_fails;
    $display("COV w4b_odd %0d cmin %0d cmax %0d czero %0d cneg %0d satr %0d pass %0d illegal %0d rst_run %0d rst_drain %0d rst_cyc %0d",
             cov_odd, cov_cmin, cov_cmax, cov_czero, cov_cneg, cov_satr,
             cov_pass, cov_illegal, cov_rst_run, cov_rst_drain, cov_rst_cyc);
    if (errors == 0) begin
      $display("TB PASS: %0d jobs, %0d illegal, %0d resets, 0 errors (bp=%0d stall=%0d gs=%0d seed=%0d)",
               n_jobs, n_illegal, n_resets, bp_mode, stall_mode, gs_mode, seed);
      $finish;
    end else $fatal(1, "TB FAIL: %0d error(s)", errors);
  end

endmodule
