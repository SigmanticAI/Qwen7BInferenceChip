// tb_pbias_unit.sv — apex_proj_bias UNIT suite (IC-BIAS, gap B).
//
// Drives apex_proj_bias and the REAL apex_scale_quant (MODE_F16) with the
// SAME job / composite / value stream, from golden-generated vectors
// (gen_bias_vectors.py, arbiter = the float64 element model verified exact
// with Fractions before emission). Three requirements per job:
//
//   1. every emitted fp16 beat == the golden expectation, bit for bit;
//   2. exact-or-refused: a job containing an out-of-window element emits
//      exactly the beats BEFORE it, raises window_error and completes;
//   3. on a bias == +0 job the two blocks' fp16 outputs are IDENTICAL beat
//      for beat — the equivalence that makes a sibling block safe next to a
//      verified one.
//
// Vector file via +vectors= (default build/pbias_unit.vec).
// Exit: "PBIAS RESULT: jobs=<n> checks=<n> errors=0 -> PASS" + "PBIAS PASS".

module tb_pbias_unit;

  parameter int unsigned D = 128;

  import apex_pkg::*;

  localparam int unsigned WAIT_LIMIT = 200000;

  logic clk, rst_n;
  int unsigned cyc;
  initial begin clk = 1'b0; forever #5 clk = ~clk; end
  always @(posedge clk) cyc <= cyc + 1;

  int unsigned watchdog_cyc = 4_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: pbias unit did not finish within %0d cycles",
           watchdog_cyc);
  end

  // ── shared stimulus nets ──────────────────────────────────────────────────
  logic             job_valid, job_mode;
  logic [DIM_W-1:0] job_cols;
  logic             cs_valid;
  logic [31:0]      cs_data;
  logic             v_valid, v_last;
  logic signed [31:0] v_data;

  logic        lw_en;
  logic [$clog2(D)-1:0] lw_addr;
  logic [15:0] lw_data;

  // ── DUT: apex_proj_bias ───────────────────────────────────────────────────
  logic pb_job_ready, pb_job_err, pb_job_stk, pb_busy, pb_done;
  logic pb_cs_ready, pb_v_ready;
  logic pb_f_valid, pb_f_last;
  logic [15:0] pb_f_data;
  logic pb_rerr, pb_rstk, pb_serr, pb_sstk, pb_ferr, pb_fstk, pb_werr, pb_wstk;

  apex_proj_bias #(.D(D), .BN_MAX(D)) u_pb (
    .clk (clk), .rst_n (rst_n),
    .job_valid (job_valid), .job_ready (pb_job_ready), .job_cols (job_cols),
    .job_error (pb_job_err), .job_error_sticky (pb_job_stk),
    .busy (pb_busy), .done (pb_done),
    .lw_en (lw_en), .lw_addr (lw_addr), .lw_data (lw_data),
    .cs_valid (cs_valid), .cs_ready (pb_cs_ready), .cs_data (cs_data),
    .v_valid (v_valid), .v_ready (pb_v_ready), .v_data (v_data),
    .v_last (v_last),
    .f_valid (pb_f_valid), .f_ready (1'b1), .f_data (pb_f_data),
    .f_last (pb_f_last),
    .range_error (pb_rerr), .range_error_sticky (pb_rstk),
    .scale_error (pb_serr), .scale_error_sticky (pb_sstk),
    .frame_error (pb_ferr), .frame_error_sticky (pb_fstk),
    .window_error (pb_werr), .window_error_sticky (pb_wstk)
  );

  // ── reference sibling: the VERIFIED apex_scale_quant in MODE_F16 ─────────
  logic sq_job_ready, sq_job_err, sq_job_stk, sq_busy, sq_done;
  logic sq_cs_ready, sq_v_ready;
  logic sq_f_valid, sq_f_last;
  logic [15:0] sq_f_data;
  logic sq_q_valid, sq_s_valid, sq_s_last;
  lane8_beat_t sq_q_beat;
  logic [15:0] sq_s_data;
  logic sq_rerr, sq_rstk, sq_serr, sq_sstk, sq_ferr, sq_fstk;

  apex_scale_quant #(.D(D)) u_sq (
    .clk (clk), .rst_n (rst_n),
    .job_valid (job_valid), .job_ready (sq_job_ready), .job_mode (job_mode),
    .job_cols (job_cols),
    .job_error (sq_job_err), .job_error_sticky (sq_job_stk),
    .busy (sq_busy), .done (sq_done),
    .cs_valid (cs_valid), .cs_ready (sq_cs_ready), .cs_data (cs_data),
    .v_valid (v_valid), .v_ready (sq_v_ready), .v_data (v_data),
    .v_last (v_last),
    .f_valid (sq_f_valid), .f_ready (1'b1), .f_data (sq_f_data),
    .f_last (sq_f_last),
    .q_valid (sq_q_valid), .q_ready (1'b1), .q_beat (sq_q_beat),
    .s_valid (sq_s_valid), .s_ready (1'b1), .s_data (sq_s_data),
    .s_last (sq_s_last),
    .range_error (sq_rerr), .range_error_sticky (sq_rstk),
    .scale_error (sq_serr), .scale_error_sticky (sq_sstk),
    .frame_error (sq_ferr), .frame_error_sticky (sq_fstk)
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, pb_job_stk, pb_rstk, pb_sstk, pb_fstk, pb_busy,
                       pb_rerr, pb_ferr, pb_job_err, pb_wstk,
                       sq_f_last,
                       sq_job_ready, sq_job_err, sq_job_stk, sq_busy, sq_done,
                       sq_cs_ready, sq_rerr, sq_rstk, sq_sstk,
                       sq_ferr, sq_fstk, sq_q_valid, sq_q_beat.data,
                       sq_q_beat.last, sq_s_valid, sq_s_data, sq_s_last};

  // ── output collectors ─────────────────────────────────────────────────────
  logic [16:0] pb_q [$];
  logic [15:0] sq_qq [$];
  int n_checks, n_errors, n_jobs;
  always @(posedge clk) if (rst_n) begin
    if (pb_f_valid) pb_q.push_back({pb_f_data, pb_f_last});
    if (sq_f_valid) sq_qq.push_back(sq_f_data);
  end

  // contract-monitor observation (pulses, latched per job by the driver)
  logic werr_seen, serr_seen, sq_serr_seen;
  always @(posedge clk) if (rst_n) begin
    if (pb_werr) werr_seen    <= 1'b1;
    if (pb_serr) serr_seen    <= 1'b1;
    if (sq_serr) sq_serr_seen <= 1'b1;
  end

  task automatic chk(input logic cond, input string what);
    n_checks++;
    if (!cond) begin n_errors++; $error("[chk] %s @%0d", what, cyc); end
  endtask

  // ── drivers (negedge discipline, §3b rule) ────────────────────────────────
  task automatic push_job(input logic [DIM_W-1:0] cols);
    int wd = 0;
    @(negedge clk);
    job_cols = cols; job_mode = 1'b0; job_valid = 1'b1;
    while (!(pb_job_ready && sq_job_ready)) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "push_job stall @%0d", cyc);
    end
    @(negedge clk); job_valid = 1'b0;
  endtask

  task automatic push_cs(input logic [31:0] c);
    int wd = 0;
    @(negedge clk);
    cs_data = c; cs_valid = 1'b1;
    while (!(pb_cs_ready && sq_cs_ready)) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "push_cs stall @%0d", cyc);
    end
    @(negedge clk); cs_valid = 1'b0;
  endtask

  task automatic push_v(input logic signed [31:0] v, input logic lst);
    int wd = 0;
    @(negedge clk);
    v_data = v; v_last = lst; v_valid = 1'b1;
    while (!(pb_v_ready && sq_v_ready)) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "push_v stall @%0d", cyc);
    end
    @(negedge clk); v_valid = 1'b0;
  endtask

  task automatic bias_load(input int n, input logic [15:0] b []);
    for (int i = 0; i < n; i++) begin
      @(negedge clk);
      lw_en = 1'b1; lw_addr = ($clog2(D))'(i); lw_data = b[i];
      @(negedge clk);
      lw_en = 1'b0;
    end
  endtask

  // ── interpreter ───────────────────────────────────────────────────────────
  initial begin : interp
    string vec_path, cmd;
    /* verilator lint_off UNUSEDSIGNAL */
    string cmt;                       // comment sink for $fgets
    /* verilator lint_on UNUSEDSIGNAL */
    int fd, i, cols, zero_bias, exp_serr, n_ok, wd;
    bit done_seen;
    logic [31:0] tv, tc;
    logic [15:0] tb_, ty;
    logic [31:0] tok;
    logic [31:0] vv [D];
    logic [31:0] cc [D];
    logic [15:0] bb [D];
    logic [15:0] yy [D];
    int          ook [D];
    logic [16:0] got;
    logic        unused_got;

    rst_n = 1'b0;
    job_valid = 1'b0; job_mode = 1'b0; job_cols = '0;
    cs_valid = 1'b0; cs_data = '0;
    v_valid = 1'b0; v_data = '0; v_last = 1'b0;
    lw_en = 1'b0; lw_addr = '0; lw_data = '0;
    n_checks = 0; n_errors = 0; n_jobs = 0;
    werr_seen = 1'b0; serr_seen = 1'b0; sq_serr_seen = 1'b0;

    vec_path = "build/pbias_unit.vec";
    void'($value$plusargs("vectors=%s", vec_path));
    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (3) @(negedge clk);

    while ($fscanf(fd, "%s", cmd) == 1) begin
      if (cmd.len() >= 2 && cmd.substr(0, 1) == "//") begin
        void'($fgets(cmt, fd));
        continue;
      end
      case (cmd)
        "JOB": begin
          void'($fscanf(fd, "%h %d %d", tv, zero_bias, exp_serr));
          cols = int'(tv);
          for (i = 0; i < cols; i++) begin
            void'($fscanf(fd, "%s", cmd));
            if (cmd != "E") $fatal(1, "expected E, got '%s'", cmd);
            void'($fscanf(fd, "%h %h %h %h %d", tv, tc, tb_, ty, tok));
            vv[i] = tv; cc[i] = tc; bb[i] = tb_;
            yy[i] = ty; ook[i] = int'(tok);
          end
          n_ok = cols;
          for (i = 0; i < cols; i++)
            if ((ook[i] == 0) && (n_ok == cols)) n_ok = i;

          // stage the bias vector, then run the job on BOTH blocks
          werr_seen = 1'b0; serr_seen = 1'b0; sq_serr_seen = 1'b0;
          pb_q.delete(); sq_qq.delete();
          bias_load(cols, bb);
          push_job(DIM_W'(cols));
          for (i = 0; i < cols; i++) push_cs(cc[i]);
          for (i = 0; i < cols; i++)
            push_v($signed(vv[i]), i == cols - 1);
          wd = 0;
          while (!pb_done) begin
            @(posedge clk); wd++;
            if (wd > WAIT_LIMIT) $fatal(1, "job %0d never completed @%0d",
                                        n_jobs, cyc);
          end
          repeat (4) @(posedge clk);

          // (1)+(2) exact-or-refused against the golden expectations
          chk(pb_q.size() == n_ok,
              $sformatf("job %0d: emitted %0d beats, expected %0d",
                        n_jobs, pb_q.size(), n_ok));
          for (i = 0; i < n_ok && i < pb_q.size(); i++) begin
            got = pb_q[i];
            unused_got = got[0];
            n_checks++;
            if (got[16:1] !== yy[i]) begin
              n_errors++;
              $error("[pb] job %0d el %0d: got %04x exp %04x (v=%08x c=%08x b=%04x)",
                     n_jobs, i, got[16:1], yy[i], vv[i], cc[i], bb[i]);
            end
          end
          chk(werr_seen == (n_ok != cols),
              $sformatf("job %0d: window_error %0b, refusals expected %0b",
                        n_jobs, werr_seen, (n_ok != cols)));
          chk(serr_seen == (exp_serr != 0),
              $sformatf("job %0d: scale_error %0b, expected %0b",
                        n_jobs, serr_seen, (exp_serr != 0)));
          chk(serr_seen == sq_serr_seen,
              $sformatf("job %0d: C2 monitor differs from apex_scale_quant",
                        n_jobs));
          if (n_ok == cols && pb_q.size() == cols)
            chk(pb_q[cols - 1][0] === 1'b1,
                $sformatf("job %0d: f_last on the final element", n_jobs));

          // (3) bias == +0 equivalence against the verified sibling
          if (zero_bias != 0) begin
            chk(sq_qq.size() == cols,
                $sformatf("job %0d: scale_quant emitted %0d/%0d",
                          n_jobs, sq_qq.size(), cols));
            for (i = 0; i < cols && i < pb_q.size() && i < sq_qq.size(); i++)
            begin
              n_checks++;
              if (pb_q[i][16:1] !== sq_qq[i]) begin
                n_errors++;
                $error("[eq] job %0d el %0d: pbias %04x != scale_quant %04x",
                       n_jobs, i, pb_q[i][16:1], sq_qq[i]);
              end
            end
          end
          n_jobs++;
        end
        "END": begin
          $display("PBIAS RESULT: cycles=%0d jobs=%0d checks=%0d errors=%0d",
                   cyc, n_jobs, n_checks, n_errors);
          if (n_errors != 0) $fatal(1, "PBIAS FAIL: %0d errors", n_errors);
          $display("%s%s", "PBIAS PASS (golden element model + ",
                   "exact-or-refused + bias=+0 equivalence vs apex_scale_quant)");
          done_seen = 1'b1;
          $finish;
        end
        default: $fatal(1, "unknown command '%s'", cmd);
      endcase
      if (done_seen) break;
    end
    if (!done_seen) $fatal(1, "vector file ended without END");
  end

endmodule
