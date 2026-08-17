// tb_layer_deq.sv — apex_layer_deq scoreboard TB (IB-LAYER S3).
// Vector-driven: JOB records with modes OK / REFUSE_JOB (illegal composite,
// job_error, nothing driven) / REFUSE_AT_i (exact-or-refused: i exact
// prefix outputs then exact_error + drain) / FRAMEE_0 / FRAMEM_e.
// Regimes: +bp_mode, +stall_mode, +seed, +resets=N (abandon every Nth OK
// job mid-stream; error accounting exact only when +resets=0).
// Exit banner: "DEQ RESULT: ... mismatches=0 -> PASS" + "DEQ PASS".

module tb_layer_deq;

  localparam int unsigned CW = $clog2(4096);
  localparam int MAXJ = 64, MAXE = 300;

  logic clk;
  logic rst_n;
  initial begin clk = 0; rst_n = 0; end
  always #5 clk = ~clk;

  logic          jb_valid, jb_ready;
  logic [CW-1:0] jb_cols;
  logic [31:0]   jb_comp;
  logic          iv, ir, ilast;
  logic signed [31:0] idata;
  logic          ov, orr, olast;
  logic [31:0]   odata;
  logic busy, done, job_error, job_error_sticky;
  logic exact_error, exact_error_sticky, frame_error, frame_error_sticky;

  apex_layer_deq u_dut (
    .clk (clk), .rst_n (rst_n),
    .jb_valid (jb_valid), .jb_ready (jb_ready),
    .jb_cols (jb_cols), .jb_comp (jb_comp),
    .iv (iv), .ir (ir), .idata (idata), .ilast (ilast),
    .ov (ov), .orr (orr), .odata (odata), .olast (olast),
    .busy (busy), .done (done),
    .job_error (job_error), .job_error_sticky (job_error_sticky),
    .exact_error (exact_error), .exact_error_sticky (exact_error_sticky),
    .frame_error (frame_error), .frame_error_sticky (frame_error_sticky)
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, busy, job_error_sticky, exact_error_sticky,
                       frame_error_sticky};

  // job table: kind 0=OK 1=REFUSE_JOB 2=REFUSE_AT 3=FRAMEE 4=FRAMEM
  int          n_jobs;
  int          jkind [MAXJ], jarg [MAXJ], jcols [MAXJ];
  int          jnin  [MAXJ], jnexp [MAXJ];
  logic [31:0] jcomp [MAXJ];
  logic [31:0] jin   [MAXJ][MAXE];
  logic [31:0] jexp  [MAXJ][MAXE];

  int bp_mode, stall_mode, resets_every;
  logic [31:0] rng;
  int n_beats, n_mism, n_jerr, n_xerr, n_ferr, n_resets;
  int exp_jerr, exp_xerr, exp_ferr;
  int cur_job, chk_i;
  logic done_seen;
  // registered job-accept capture: jb_ready is combinational (st==IDLE) and
  // drops AT the accepting edge, so a post-edge poll of jb_ready deadlocks;
  // the NBA capture holds the pre-edge handshake truth for the driver.
  logic jb_acc;
  always @(posedge clk) jb_acc <= jb_valid && jb_ready && rst_n;

  function automatic logic [31:0] xorshift(input logic [31:0] x);
    logic [31:0] r;
    begin
      r = x ^ (x << 13); r = r ^ (r >> 17); r = r ^ (r << 5); return r;
    end
  endfunction

  task automatic load_vectors(input string fn);
    int fh, cols, v, i;
    string line, mode;
    logic [31:0] comp;
    begin
      fh = $fopen(fn, "r");
      if (fh == 0) $fatal(1, "DEQ FAIL: cannot open %s", fn);
      n_jobs = 0;
      while ($fgets(line, fh) != 0) begin
        if ($sscanf(line, "JOB %d %h %s", cols, comp, mode) == 3) begin
          jcols[n_jobs] = cols;
          jcomp[n_jobs] = comp;
          jarg[n_jobs]  = 0;
          if (mode == "OK") begin
            jkind[n_jobs] = 0; jnin[n_jobs] = cols; jnexp[n_jobs] = cols;
          end else if (mode == "REFUSE_JOB") begin
            jkind[n_jobs] = 1; jnin[n_jobs] = 0; jnexp[n_jobs] = 0;
            exp_jerr++;
          end else if ($sscanf(mode, "REFUSE_AT_%d", v) == 1) begin
            jkind[n_jobs] = 2; jarg[n_jobs] = v;
            jnin[n_jobs] = cols; jnexp[n_jobs] = v;
            exp_xerr++;
          end else if ($sscanf(mode, "FRAMEE_%d", v) == 1) begin
            jkind[n_jobs] = 3; jarg[n_jobs] = v;
            jnin[n_jobs] = v + 1; jnexp[n_jobs] = 0;
            exp_ferr++;
          end else if ($sscanf(mode, "FRAMEM_%d", v) == 1) begin
            jkind[n_jobs] = 4; jarg[n_jobs] = v;
            jnin[n_jobs] = cols + v; jnexp[n_jobs] = 0;
            exp_ferr++;
          end else $fatal(1, "DEQ FAIL: bad mode %s", mode);
          for (i = 0; i < jnin[n_jobs]; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            jin[n_jobs][i] = 32'(v);
          end
          for (i = 0; i < jnexp[n_jobs]; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            jexp[n_jobs][i] = 32'(v);
          end
          n_jobs++;
          if (n_jobs >= MAXJ) $fatal(1, "DEQ FAIL: MAXJ");
        end
      end
      $fclose(fh);
      if (n_jobs < 10) $fatal(1, "DEQ FAIL: too few jobs");
    end
  endtask

  // backpressure
  always @(posedge clk) begin
    rng <= xorshift(rng);
    case (bp_mode)
      0: orr <= 1'b1;
      1: orr <= (rng[1:0] != 2'b00);
      default: orr <= (rng[3:0] == 4'h1) || (rng[7:4] == 4'h2);
    endcase
  end

  // monitor (NBA-only; deltas combinational)
  wire accept  = ov && orr && rst_n;
  wire data_mm = accept && ((odata !== jexp[cur_job][chk_i])
                         || (olast !== (chk_i == jcols[cur_job] - 1)));
  wire over_mm = accept && (chk_i >= jnexp[cur_job]);
  always @(posedge clk) begin
    if (data_mm && !over_mm)
      $display("MISMATCH job=%0d i=%0d got=%h exp=%h last=%b",
               cur_job, chk_i, odata, jexp[cur_job][chk_i], olast);
    if (over_mm)
      $display("MISMATCH job=%0d: unexpected output %0d (mode kind %0d)",
               cur_job, chk_i, jkind[cur_job]);
    n_mism <= n_mism + ((data_mm && !over_mm) ? 1 : 0) + (over_mm ? 1 : 0);
    if (accept) begin
      n_beats <= n_beats + 1;
      chk_i   <= chk_i + 1;
    end
    if (job_error && rst_n)   n_jerr <= n_jerr + 1;
    if (exact_error && rst_n) n_xerr <= n_xerr + 1;
    if (frame_error && rst_n) n_ferr <= n_ferr + 1;
    if (n_mism > 20) $fatal(1, "DEQ FAIL: too many mismatches");
    if (done && rst_n) done_seen <= 1'b1;
  end

  task automatic drive_beat(input logic [31:0] d, input logic l);
    begin
      while (stall_mode != 0 && (rng[11:9] == 3'b101)) @(posedge clk);
      idata = d;
      ilast = l;
      iv    = 1'b1;
      do @(posedge clk); while (!ir);
      iv    = 1'b0;
      ilast = 1'b0;
    end
  endtask

  task automatic push_job();   // uses the driver's current job index j
    begin
      jb_cols  = CW'(jcols[j]);
      jb_comp  = jcomp[j];
      jb_valid = 1'b1;
      do @(posedge clk); while (!jb_acc);
      jb_valid = 1'b0;
    end
  endtask

  int j;
  initial begin
    string vfile;
    int seed;
    if (!$value$plusargs("vectors=%s", vfile)) vfile = "build/vectors_deq.txt";
    if (!$value$plusargs("bp_mode=%d", bp_mode)) bp_mode = 1;
    if (!$value$plusargs("stall_mode=%d", stall_mode)) stall_mode = 1;
    if (!$value$plusargs("seed=%d", seed)) seed = 32'hDE0;
    if (!$value$plusargs("resets=%d", resets_every)) resets_every = 0;
    rng = 32'(seed);
    exp_jerr = 0; exp_xerr = 0; exp_ferr = 0;
    load_vectors(vfile);
    done_seen = 1'b0;
    n_beats = 0; n_mism = 0; n_jerr = 0; n_xerr = 0; n_ferr = 0; n_resets = 0;
    jb_valid = 0; iv = 0; ilast = 0; idata = '0; jb_cols = '0; jb_comp = '0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    for (j = 0; j < n_jobs; j++) begin
      cur_job   = j;
      chk_i     = 0;
      done_seen = 1'b0;
      if (resets_every != 0 && jkind[j] == 0
          && (j % resets_every) == (resets_every - 1)) begin
        push_job();
        for (int i = 0; i < (jnin[j] > 2 ? 2 : 1); i++)
          drive_beat(jin[j][i], 1'b0);
        rst_n = 1'b0;
        iv = 0; jb_valid = 0;
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        n_resets = n_resets + 1;
        repeat (2) @(posedge clk);
        continue;
      end
      push_job();
      if (jkind[j] == 1) begin
        repeat (4) @(posedge clk);            // job_error pulse window
      end else begin
        for (int i = 0; i < jnin[j]; i++) begin
          logic l;
          l = (jkind[j] == 3) ? (i == jarg[j]) : (i == jnin[j] - 1);
          drive_beat(jin[j][i], l);
        end
        // wait for completion (done) then let outputs drain
        wait (done_seen);
        @(posedge clk);
        wait (chk_i >= jnexp[j]);
        repeat (4) @(posedge clk);
      end
    end
    repeat (10) @(posedge clk);

    if (resets_every == 0 &&
        (n_jerr != exp_jerr || n_xerr != exp_xerr || n_ferr != exp_ferr)) begin
      $display("DEQ FAIL: errs jerr=%0d/%0d xerr=%0d/%0d ferr=%0d/%0d",
               n_jerr, exp_jerr, n_xerr, exp_xerr, n_ferr, exp_ferr);
      $fatal(1, "DEQ FAIL: error accounting");
    end
    if (n_mism == 0) begin
      $display("DEQ RESULT: jobs=%0d beats=%0d jerr=%0d xerr=%0d ferr=%0d resets=%0d mismatches=0 -> PASS",
               n_jobs, n_beats, n_jerr, n_xerr, n_ferr, n_resets);
      $display("DEQ PASS");
      $finish;
    end else begin
      $fatal(1, "DEQ FAIL: %0d mismatches", n_mism);
    end
  end

  initial begin
    #100_000_000;
    $fatal(1, "DEQ FAIL: watchdog");
  end

endmodule
