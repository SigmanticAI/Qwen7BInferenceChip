// tb_resid_unit.sv — apex_residual DATAPATH closure (IB-LAYER S5; closes
// the S4 deferral). Vector-driven: the unit consumes the REAL BUS_ON
// operand pairs (fp16 X row loaded via lw_*, EXACT-fp32 attn_proj/ffn_out
// streamed) and every row element is read back and compared against the
// D-030 arbiter's r1 then r2 (chained in place) — plus the +-0 battery,
// window-refusal, the e8=144 inf-shortcut boundary (with the e8=143
// finite-32.0 distinguisher), frame and geometry rejects.
// Regimes: +stall_mode input gaps, +resets=1 (abandon the first case's
// job-1 mid-stream, then re-drive the whole case — clean-restart proof).
// Exit: "RESID RESULT: ... mismatches=0 -> PASS" + "RESID PASS".

module tb_resid_unit;

  localparam int unsigned DM_MAX = 128;
  localparam int unsigned AW = $clog2(DM_MAX);
  localparam int MAXC = 12, MAXJ = 6, MAXE = 160;

  logic clk;
  logic rst_n;
  initial begin clk = 0; rst_n = 0; end
  always #5 clk = ~clk;

  logic          lw_en;
  logic [AW-1:0] lw_addr, rd_addr;
  logic [15:0]   lw_data, rd_data;
  logic          jb_valid, jb_ready;
  logic [11:0]   jb_cols;
  logic          iv, ir, ilast;
  logic [31:0]   idata;
  logic busy, done, frame_error, frame_error_sticky;
  logic window_error, window_error_sticky;

  // E-1 egress ports: held OFF in this TB (the add-datapath suite predates
  // the egress; the egress is exercised at apex_top level by the E-lane
  // programs). Tied exactly like an idle tile: no job, consumer ready.
  logic        ej_ready, e_v, e_last, e_busy;
  logic [15:0] e_data;

  apex_residual #(.DM_MAX(DM_MAX)) u_dut (
    .clk (clk), .rst_n (rst_n),
    .lw_en (lw_en), .lw_addr (lw_addr), .lw_data (lw_data),
    .rd_addr (rd_addr), .rd_data (rd_data),
    .jb_valid (jb_valid), .jb_ready (jb_ready), .jb_cols (jb_cols),
    .jb_base (2'd0),
    .iv (iv), .ir (ir), .idata (idata), .ilast (ilast),
    .ej_valid (1'b0), .ej_ready (ej_ready), .ej_cols (12'd0),
    .ej_base (2'd0),
    .ev (e_v), .er (1'b1), .edata (e_data), .elast (e_last),
    .ebusy (e_busy),
    .busy (busy), .done (done),
    .frame_error (frame_error), .frame_error_sticky (frame_error_sticky),
    .window_error (window_error), .window_error_sticky (window_error_sticky)
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, busy, jb_ready, frame_error_sticky,
                       window_error_sticky,
                       ej_ready, e_v, e_data, e_last, e_busy};
  // int loop indices / parse locals always leave high bits unused
  logic unused_tb;
  int   unused_vci_probe;
  always_comb unused_tb = &{1'b0, jarg[0][0][31:0],
                            unused_vci_probe[31:0]};

  // registered handshake captures (the S3/S4 lesson: combinational
  // jb_ready drops at the accepting edge; done is a pulse)
  logic jb_acc, done_seen;
  always @(posedge clk) begin
    jb_acc <= jb_valid && jb_ready && rst_n;
    if (done && rst_n) done_seen <= 1'b1;
  end

  // vectors: cases -> {row, jobs}; job kinds 0 OK, 1 WREFUSE(arg),
  // 2 FRAMEE, 3 GEO_ZERO, 4 GEO_BIG
  int          n_cases;
  string       cname   [MAXC];
  int          ccols   [MAXC], cnjobs [MAXC];
  logic [15:0] crow    [MAXC][MAXE];
  int          jkind   [MAXC][MAXJ], jarg [MAXC][MAXJ];
  int          jnin    [MAXC][MAXJ], jnexp [MAXC][MAXJ];
  logic [31:0] jin     [MAXC][MAXJ][MAXE];
  logic [15:0] jexp    [MAXC][MAXJ][MAXE];

  int stall_mode, resets_mode;
  logic [31:0] rng;
  int n_jobs, n_elems, n_mism, n_ferr, n_werr, exp_ferr, exp_werr, n_resets;

  function automatic logic [31:0] xorshift(input logic [31:0] x);
    logic [31:0] r;
    begin
      r = x ^ (x << 13); r = r ^ (r >> 17); r = r ^ (r << 5); return r;
    end
  endfunction

  int cur_ji;

  task automatic load_vectors(input string fn);
    int fh, v, i, vci, ji;
    // vci indexes arrays sized MAXC (4 bits used); park the high bits

    string line, tok;
    begin
      fh = $fopen(fn, "r");
      if (fh == 0) $fatal(1, "RESID FAIL: cannot open %s", fn);
      n_cases = 0;
      ci = -1; ji = 0;
      while ($fgets(line, fh) != 0) begin
        if ($sscanf(line, "CASE %s %d", tok, v) == 2) begin
          vci = n_cases;
          cname[vci] = tok;
          ccols[vci] = v;
          ji = 0;
        end else if (line.substr(0, 2) == "ROW") begin
          for (i = 0; i < ccols[vci]; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            crow[vci][i] = 16'(v);
          end
        end else if ($sscanf(line, "JOB %s", tok) == 1) begin
          jarg[vci][ji] = 0;
          if (tok == "OK") begin
            jkind[vci][ji] = 0; jnin[vci][ji] = ccols[vci];
            jnexp[vci][ji] = ccols[vci];
          end else if ($sscanf(tok, "WREFUSE_%d", v) == 1) begin
            jkind[vci][ji] = 1; jarg[vci][ji] = v;
            jnin[vci][ji] = ccols[vci]; jnexp[vci][ji] = v;
            exp_werr++;
          end else if (tok == "FRAMEE") begin
            jkind[vci][ji] = 2; jnin[vci][ji] = 1; jnexp[vci][ji] = 0;
            exp_ferr++;
          end else if (tok == "GEO_ZERO") begin
            jkind[vci][ji] = 3; jnin[vci][ji] = 0; jnexp[vci][ji] = 0;
            exp_ferr++;
          end else if (tok == "GEO_BIG") begin
            jkind[vci][ji] = 4; jnin[vci][ji] = 0; jnexp[vci][ji] = 0;
            exp_ferr++;
          end else $fatal(1, "RESID FAIL: bad mode %s", tok);
          for (i = 0; i < jnin[vci][ji]; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            jin[vci][ji][i] = 32'(v);
          end
          if (jnexp[vci][ji] > 0) begin
            void'($fgets(line, fh));            // "EXP"
            for (i = 0; i < jnexp[vci][ji]; i++) begin
              void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
              jexp[vci][ji][i] = 16'(v);
            end
          end
          ji++;
        end else if (line.substr(0, 6) == "ENDCASE") begin
          cnjobs[vci] = ji;
          n_cases++;
          if (n_cases >= MAXC) $fatal(1, "RESID FAIL: MAXC");
        end
      end
      $fclose(fh);
      unused_vci_probe = vci;
      if (n_cases < 5) $fatal(1, "RESID FAIL: too few cases");
    end
  endtask

  always @(posedge clk) begin
    rng <= xorshift(rng);
    if (frame_error && rst_n)  n_ferr <= n_ferr + 1;
    if (window_error && rst_n) n_werr <= n_werr + 1;
  end

  task automatic load_row();          // uses driver-scope ci
    begin
      for (int i = 0; i < ccols[ci]; i++) begin
        @(negedge clk);
        lw_en   = 1'b1;
        lw_addr = AW'(i);
        lw_data = crow[ci][i];
      end
      @(negedge clk);
      lw_en = 1'b0;
      @(posedge clk);
    end
  endtask

  task automatic push_job(input logic [11:0] pcols);
    begin
      @(negedge clk);
      jb_cols  = pcols;
      jb_valid = 1'b1;
      do @(posedge clk); while (!jb_acc);
      jb_valid = 1'b0;
    end
  endtask

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

  task automatic readback_check(input int n);  // driver-scope ci/cur_ji
    begin
      for (int i = 0; i < n; i++) begin
        rd_addr = AW'(i);
        #1;
        if (rd_data !== jexp[ci][cur_ji][i]) begin
          n_mism++;
          $display("MISMATCH %s job%0d row[%0d] got=%h exp=%h",
                   cname[ci], cur_ji, i, rd_data, jexp[ci][cur_ji][i]);
        end
        n_elems++;
      end
      if (n_mism > 20) $fatal(1, "RESID FAIL: too many mismatches");
    end
  endtask

  task automatic run_job();           // driver-scope ci/cur_ji
    begin
      done_seen = 1'b0;
      if (jkind[ci][cur_ji] == 3) begin
        push_job(12'd0);
        repeat (3) @(posedge clk);
      end else if (jkind[ci][cur_ji] == 4) begin
        push_job(12'(DM_MAX + 1));
        repeat (3) @(posedge clk);
      end else begin
        push_job(12'(ccols[ci]));
        for (int i = 0; i < jnin[ci][cur_ji]; i++)
          drive_beat(jin[ci][cur_ji][i],
                     (jkind[ci][cur_ji] == 2) ? 1'b1            // early last
                                              : (i == jnin[ci][cur_ji] - 1));
        wait (done_seen);
        @(posedge clk);
        readback_check(jnexp[ci][cur_ji]);
      end
      n_jobs++;
    end
  endtask

  int ci;
  initial begin
    string vfile;
    int seed;
    if (!$value$plusargs("vectors=%s", vfile))
      vfile = "build/vectors_resid.txt";
    if (!$value$plusargs("stall_mode=%d", stall_mode)) stall_mode = 1;
    if (!$value$plusargs("seed=%d", seed)) seed = 32'h51D;
    if (!$value$plusargs("resets=%d", resets_mode)) resets_mode = 0;
    rng = 32'(seed);
    exp_ferr = 0; exp_werr = 0;
    load_vectors(vfile);
    n_jobs = 0; n_elems = 0; n_mism = 0; n_ferr = 0; n_werr = 0; n_resets = 0;
    lw_en = 0; lw_addr = '0; lw_data = '0; rd_addr = '0;
    jb_valid = 0; jb_cols = '0; iv = 0; ilast = 0; idata = '0;
    done_seen = 0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    for (ci = 0; ci < n_cases; ci++) begin
      load_row();
      if (resets_mode != 0 && ci == 0) begin
        // abandon job 1 mid-stream, then re-drive the WHOLE case clean
        push_job(12'(ccols[ci]));
        for (int i = 0; i < ccols[ci] / 2; i++)
          drive_beat(jin[ci][0][i], 1'b0);
        @(negedge clk);
        rst_n = 1'b0;
        iv = 0; jb_valid = 0;
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        n_resets = n_resets + 1;
        repeat (2) @(posedge clk);
        load_row();                         // row partially updated: reload
      end
      for (cur_ji = 0; cur_ji < cnjobs[ci]; cur_ji++)
        run_job();
    end
    repeat (10) @(posedge clk);

    if (n_ferr != exp_ferr || n_werr != exp_werr) begin
      $display("RESID FAIL: errs ferr=%0d/%0d werr=%0d/%0d",
               n_ferr, exp_ferr, n_werr, exp_werr);
      $fatal(1, "RESID FAIL: error accounting");
    end
    if (n_mism == 0) begin
      $display("RESID RESULT: cases=%0d jobs=%0d elems=%0d ferr=%0d werr=%0d resets=%0d mismatches=0 -> PASS",
               n_cases, n_jobs, n_elems, n_ferr, n_werr, n_resets);
      $display("RESID PASS");
      $finish;
    end else begin
      $fatal(1, "RESID FAIL: %0d mismatches", n_mism);
    end
  end

  initial begin
    #100_000_000;
    $fatal(1, "RESID FAIL: watchdog");
  end

endmodule
