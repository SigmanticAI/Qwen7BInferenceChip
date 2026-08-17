// tb_swiglu.sv — asu_swiglu scoreboard TB (IB-LAYER S3).
// Arbiter: golden silu_apply(gate)*up composition (gen_swiglu_vectors.py).
// Modes: OK/tieN/tieS/OVFL (cols products expected, incl. the engineered
// Q5.10 + product RNE ties and the inf-saturation battery), REFUSE_JOB
// (illegal composite), FRAMEG_0 (gate early-last), FRAMEU_e (up missing
// last -> resync). Regimes: +bp_mode/+stall_mode/+seed/+resets=N.
// Exit banner: "SWIGLU RESULT: ... mismatches=0 -> PASS" + "SWIGLU PASS".

module tb_swiglu;

  localparam int unsigned COLS_MAX = 64;
  localparam int unsigned CW = $clog2(COLS_MAX + 1);
  localparam int MAXJ = 64, MAXE = 80;

  logic clk;
  logic rst_n;
  initial begin clk = 0; rst_n = 0; end
  always #5 clk = ~clk;

  logic          jb_valid, jb_ready;
  logic [CW-1:0] jb_cols;
  logic [31:0]   jb_comp_g, jb_comp_u;
  logic gv, gr, glast, uv, ur, ulast;
  logic signed [31:0] gdata, udata;
  logic pv, pr, plast;
  logic [15:0] pdata;
  logic busy, done, job_error, job_error_sticky;
  logic frame_error, frame_error_sticky;

  asu_swiglu #(.COLS_MAX(COLS_MAX)) u_dut (
    .clk (clk), .rst_n (rst_n),
    .jb_valid (jb_valid), .jb_ready (jb_ready), .jb_cols (jb_cols),
    .jb_comp_g (jb_comp_g), .jb_comp_u (jb_comp_u),
    .gv (gv), .gr (gr), .gdata (gdata), .glast (glast),
    .uv (uv), .ur (ur), .udata (udata), .ulast (ulast),
    .pv (pv), .pr (pr), .pdata (pdata), .plast (plast),
    .busy (busy), .done (done),
    .job_error (job_error), .job_error_sticky (job_error_sticky),
    .frame_error (frame_error), .frame_error_sticky (frame_error_sticky)
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, busy, job_error_sticky, frame_error_sticky};

  // job table: kind 0=normal(OK/tieN/tieS/OVFL) 1=REFUSE_JOB 2=FRAMEG 3=FRAMEU
  int          n_jobs;
  int          jkind [MAXJ], jcols [MAXJ], jng [MAXJ], jnu [MAXJ], jnexp [MAXJ];
  logic [31:0] jcg [MAXJ], jcu [MAXJ];
  logic [31:0] jg [MAXJ][MAXE], ju [MAXJ][MAXE];
  logic [15:0] jexp [MAXJ][MAXE];

  int bp_mode, stall_mode, resets_every;
  logic [31:0] rng;
  int n_beats, n_mism, n_jerr, n_ferr, n_resets, exp_jerr, exp_ferr;
  int cur_job, chk_i;
  logic done_seen;
  // registered job-accept capture (see tb_layer_deq.sv note): jb_ready is
  // combinational and drops at the accepting edge.
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
    logic [31:0] cg, cu;
    begin
      fh = $fopen(fn, "r");
      if (fh == 0) $fatal(1, "SWIGLU FAIL: cannot open %s", fn);
      n_jobs = 0;
      while ($fgets(line, fh) != 0) begin
        if ($sscanf(line, "JOB %d %h %h %s", cols, cg, cu, mode) == 4) begin
          jcols[n_jobs] = cols; jcg[n_jobs] = cg; jcu[n_jobs] = cu;
          if (mode == "REFUSE_JOB") begin
            jkind[n_jobs] = 1; jng[n_jobs] = 0; jnu[n_jobs] = 0;
            jnexp[n_jobs] = 0; exp_jerr++;
          end else if ($sscanf(mode, "FRAMEG_%d", v) == 1) begin
            jkind[n_jobs] = 2; jng[n_jobs] = v + 1; jnu[n_jobs] = 0;
            jnexp[n_jobs] = 0; exp_ferr++;
          end else if ($sscanf(mode, "FRAMEU_%d", v) == 1) begin
            jkind[n_jobs] = 3; jng[n_jobs] = cols; jnu[n_jobs] = cols + v;
            jnexp[n_jobs] = 0; exp_ferr++;
          end else begin
            jkind[n_jobs] = 0; jng[n_jobs] = cols; jnu[n_jobs] = cols;
            jnexp[n_jobs] = cols;
          end
          for (i = 0; i < jng[n_jobs]; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            jg[n_jobs][i] = 32'(v);
          end
          for (i = 0; i < jnu[n_jobs]; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            ju[n_jobs][i] = 32'(v);
          end
          for (i = 0; i < jnexp[n_jobs]; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            jexp[n_jobs][i] = 16'(v);
          end
          n_jobs++;
          if (n_jobs >= MAXJ) $fatal(1, "SWIGLU FAIL: MAXJ");
        end
      end
      $fclose(fh);
      if (n_jobs < 10) $fatal(1, "SWIGLU FAIL: too few jobs");
    end
  endtask

  always @(posedge clk) begin
    rng <= xorshift(rng);
    case (bp_mode)
      0: pr <= 1'b1;
      1: pr <= (rng[1:0] != 2'b00);
      default: pr <= (rng[3:0] == 4'h1) || (rng[7:4] == 4'h2);
    endcase
  end

  wire accept  = pv && pr && rst_n;
  wire over_mm = accept && (chk_i >= jnexp[cur_job]);
  wire data_mm = accept && !over_mm
               && ((pdata !== jexp[cur_job][chk_i])
                || (plast !== (chk_i == jcols[cur_job] - 1)));
  always @(posedge clk) begin
    if (data_mm)
      $display("MISMATCH job=%0d i=%0d got=%h exp=%h last=%b",
               cur_job, chk_i, pdata, jexp[cur_job][chk_i], plast);
    if (over_mm)
      $display("MISMATCH job=%0d: unexpected product %0d", cur_job, chk_i);
    n_mism <= n_mism + (data_mm ? 1 : 0) + (over_mm ? 1 : 0);
    if (accept) begin
      n_beats <= n_beats + 1;
      chk_i   <= chk_i + 1;
    end
    if (job_error && rst_n)   n_jerr <= n_jerr + 1;
    if (frame_error && rst_n) n_ferr <= n_ferr + 1;
    if (n_mism > 20) $fatal(1, "SWIGLU FAIL: too many mismatches");
    if (done && rst_n) done_seen <= 1'b1;
  end

  task automatic drive_g(input logic [31:0] d, input logic l);
    begin
      while (stall_mode != 0 && (rng[11:9] == 3'b101)) @(posedge clk);
      gdata = d; glast = l; gv = 1'b1;
      do @(posedge clk); while (!gr);
      gv = 1'b0; glast = 1'b0;
    end
  endtask

  task automatic drive_u(input logic [31:0] d, input logic l);
    begin
      while (stall_mode != 0 && (rng[14:12] == 3'b011)) @(posedge clk);
      udata = d; ulast = l; uv = 1'b1;
      do @(posedge clk); while (!ur);
      uv = 1'b0; ulast = 1'b0;
    end
  endtask

  int j;
  initial begin
    string vfile;
    int seed;
    if (!$value$plusargs("vectors=%s", vfile))
      vfile = "build/vectors_swiglu.txt";
    if (!$value$plusargs("bp_mode=%d", bp_mode)) bp_mode = 1;
    if (!$value$plusargs("stall_mode=%d", stall_mode)) stall_mode = 1;
    if (!$value$plusargs("seed=%d", seed)) seed = 32'h5B16;
    if (!$value$plusargs("resets=%d", resets_every)) resets_every = 0;
    rng = 32'(seed);
    exp_jerr = 0; exp_ferr = 0;
    load_vectors(vfile);
    done_seen = 1'b0;
    n_beats = 0; n_mism = 0; n_jerr = 0; n_ferr = 0; n_resets = 0;
    jb_valid = 0; gv = 0; glast = 0; gdata = '0;
    uv = 0; ulast = 0; udata = '0; jb_cols = '0; jb_comp_g = '0;
    jb_comp_u = '0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    for (j = 0; j < n_jobs; j++) begin
      cur_job   = j;
      chk_i     = 0;
      done_seen = 1'b0;
      jb_cols   = CW'(jcols[j]);
      jb_comp_g = jcg[j];
      jb_comp_u = jcu[j];
      if (resets_every != 0 && jkind[j] == 0
          && (j % resets_every) == (resets_every - 1)) begin
        jb_valid = 1'b1;
        do @(posedge clk); while (!jb_acc);
        jb_valid = 1'b0;
        drive_g(jg[j][0], 1'b0);
        rst_n = 1'b0; gv = 0; uv = 0;
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        n_resets = n_resets + 1;
        repeat (2) @(posedge clk);
        continue;
      end
      jb_valid = 1'b1;
      do @(posedge clk); while (!jb_acc);
      jb_valid = 1'b0;
      if (jkind[j] == 1) begin
        repeat (4) @(posedge clk);
      end else begin
        for (int i = 0; i < jng[j]; i++)
          drive_g(jg[j][i],
                  (jkind[j] == 2) ? (i == jng[j] - 1)      // early last @0
                                  : (i == jng[j] - 1));
        // FRAMEG: early last is at index jng-1 == arg position by
        // construction (arg=0 -> 1 beat with last)
        if (jkind[j] != 2) begin
          for (int i = 0; i < jnu[j]; i++)
            drive_u(ju[j][i],
                    (jkind[j] == 3) ? (i == jnu[j] - 1)    // resync last
                                    : (i == jnu[j] - 1));
        end
        wait (done_seen);
        @(posedge clk);
        wait (chk_i >= jnexp[j]);
        repeat (4) @(posedge clk);
      end
    end
    repeat (10) @(posedge clk);

    if (resets_every == 0 && (n_jerr != exp_jerr || n_ferr != exp_ferr)) begin
      $display("SWIGLU FAIL: errs jerr=%0d/%0d ferr=%0d/%0d",
               n_jerr, exp_jerr, n_ferr, exp_ferr);
      $fatal(1, "SWIGLU FAIL: error accounting");
    end
    if (n_mism == 0) begin
      $display("SWIGLU RESULT: jobs=%0d beats=%0d jerr=%0d ferr=%0d resets=%0d mismatches=0 -> PASS",
               n_jobs, n_beats, n_jerr, n_ferr, n_resets);
      $display("SWIGLU PASS");
      $finish;
    end else begin
      $fatal(1, "SWIGLU FAIL: %0d mismatches", n_mism);
    end
  end

  initial begin
    #100_000_000;
    $fatal(1, "SWIGLU FAIL: watchdog");
  end

endmodule
