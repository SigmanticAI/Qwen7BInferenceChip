// tb_rope_row.sv — rope_row scoreboard TB (IB-LAYER S2).
//
// Vector-driven: build/vectors_rope_d<D>.txt rows (golden-fp expected, gen
// self-checked against rope_fx on the tensor rows). Every accepted output
// beat is compared against BOTH the file expectation and a SHADOW instance
// of the behavioral rtl/rope/rope.sv driven with the same (u_q, x_i, x_ih)
// operands — the integer DUT is pinned to golden AND behavioral at once.
//
// Regimes: +bp_mode (0 free / 1 random / 2 storm output backpressure),
// +stall_mode (input gaps), +seed, +resets=N (pulse rst_n mid-row every
// N-th row: current row abandoned, next row must run clean — clean-row
// recovery; frame-error ACCOUNTING is exact only when +resets=0, noted).
// Bad frames: BADE (early last) and BADM (missing last -> RESYNC) rows
// must raise exactly one frame_error pulse each and emit NOTHING.
//
// Exit: "ROPEROW RESULT: rows=<n> beats=<m> errs=<k> mismatches=0 -> PASS"
// + "ROPEROW PASS", else $fatal.

module tb_rope_row #(
  parameter int unsigned D = 64
);
  localparam int unsigned HALF   = D / 2;
  localparam int unsigned PAIR_W = $clog2(HALF);
  localparam int unsigned MAXR   = 1024;

  logic clk;
  logic rst_n;
  initial begin
    clk   = 1'b0;
    rst_n = 1'b0;
  end
  always #5 clk = ~clk;

  // DUT
  logic              s_valid, s_ready, s_last;
  logic [15:0]       s_data;
  logic              m_valid, m_ready, m_last;
  logic [15:0]       m_data;
  logic [PAIR_W-1:0] ph_addr;
  logic [13:0]       ph_data;
  logic              busy, frame_error, frame_error_sticky;

  rope_row #(.D(D)) u_dut (
    .clk (clk), .rst_n (rst_n),
    .s_valid (s_valid), .s_ready (s_ready), .s_data (s_data), .s_last (s_last),
    .m_valid (m_valid), .m_ready (m_ready), .m_data (m_data), .m_last (m_last),
    .ph_addr (ph_addr), .ph_data (ph_data),
    .busy (busy), .frame_error (frame_error),
    .frame_error_sticky (frame_error_sticky)
  );

  // shadow behavioral reference (real-datapath rope.sv), registered operands
  logic [13:0] sh_u;
  logic [15:0] sh_xa, sh_xb, she_q;
  logic        shv_q, shhi_q;
  logic [15:0] yb_i, yb_ih;
  rope u_shadow (.u_q (sh_u), .x_i (sh_xa), .x_ih (sh_xb),
                 .y_i (yb_i), .y_ih (yb_ih));

  // ── vectors ───────────────────────────────────────────────────────────────
  // kind: 0 good, 1 BADE(arg=last pos), 2 BADM(arg=extra beats)
  int          n_rows;
  int          kind   [MAXR];
  int          arg    [MAXR];
  logic [13:0] phs    [MAXR][HALF];
  logic [15:0] din    [MAXR][2 * D];   // bad rows may carry up to D+arg beats
  int          din_n  [MAXR];
  logic [15:0] dexp   [MAXR][D];

  int bp_mode, stall_mode, resets_every;
  logic [31:0] rng;
  int n_beats, n_mism, n_errs, n_exp_errs, n_resets;
  int emit_row;                        // row currently being emitted/checked
  int chk_k;                           // next expected output index
  event row_done_ev;

  function automatic logic [31:0] xorshift(input logic [31:0] x);
    logic [31:0] r;
    begin
      r = x ^ (x << 13); r = r ^ (r >> 17); r = r ^ (r << 5); return r;
    end
  endfunction

  logic unused_ok;
  assign unused_ok = &{1'b0, busy};

  task automatic load_vectors(input string fn);
    int fh, v, i;
    string line, k;
    begin
      fh = $fopen(fn, "r");
      if (fh == 0) $fatal(1, "ROPEROW FAIL: cannot open %s", fn);
      n_rows = 0;
      while ($fgets(line, fh) != 0) begin
        if ($sscanf(line, "ROW %s", k) == 1) begin
          void'(k.len());
          kind[n_rows] = 0; arg[n_rows] = 0;
          for (i = 0; i < HALF; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            phs[n_rows][i] = 14'(v);
          end
          for (i = 0; i < D; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            din[n_rows][i] = 16'(v);
          end
          din_n[n_rows] = D;
          for (i = 0; i < D; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            dexp[n_rows][i] = 16'(v);
          end
          n_rows++;
        end else if ($sscanf(line, "BADE %d", v) == 1) begin
          kind[n_rows] = 1; arg[n_rows] = v; din_n[n_rows] = v + 1;
          for (i = 0; i < v + 1; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            din[n_rows][i] = 16'(v);
          end
          n_rows++;
        end else if ($sscanf(line, "BADM %d", v) == 1) begin
          kind[n_rows] = 2; arg[n_rows] = 5; din_n[n_rows] = D + 5;
          for (i = 0; i < D + 5; i++) begin
            void'($fgets(line, fh)); void'($sscanf(line, "%h", v));
            din[n_rows][i] = 16'(v);
          end
          n_rows++;
        end
        if (n_rows >= MAXR) $fatal(1, "ROPEROW FAIL: MAXR overflow");
      end
      $fclose(fh);
      if (n_rows < 100) $fatal(1, "ROPEROW FAIL: too few rows (%0d)", n_rows);
    end
  endtask

  // phase RAM model: 1-cycle registered read serving the row under emission
  always @(posedge clk) begin
    if (emit_row < n_rows) ph_data <= phs[emit_row][ph_addr];
    else                   ph_data <= '0;
  end

  // output backpressure
  always @(posedge clk) begin
    rng <= xorshift(rng);
    case (bp_mode)
      0: m_ready <= 1'b1;
      1: m_ready <= (rng[1:0] != 2'b00);
      default: m_ready <= (rng[3:0] == 4'h1) || (rng[7:4] == 4'h2);
    endcase
  end

  // monitor + shadow compare (NBA-only clocked process; deltas computed
  // combinationally so multiple hits in one cycle are never lost)
  wire accept = m_valid && m_ready && rst_n;
  int mon_pair;
  always_comb mon_pair = (chk_k < int'(HALF)) ? chk_k : (chk_k - int'(HALF));
  wire cur_good  = (kind[emit_row] == 0);
  wire data_mm   = accept && cur_good && (m_data !== dexp[emit_row][chk_k]);
  wire last_mm   = accept && cur_good && (m_last !== (chk_k == int'(D) - 1));
  wire spur_mm   = accept && !cur_good;
  wire shadow_mm = shv_q && ((shhi_q ? yb_ih : yb_i) !== she_q);
  wire [2:0] mism_add = {2'b0, data_mm} + {2'b0, last_mm}
                      + {2'b0, spur_mm} + {2'b0, shadow_mm};

  always @(posedge clk) begin
    if (shadow_mm)
      $display("SHADOW MISMATCH u=%h xa=%h xb=%h beh=%h exp=%h",
               sh_u, sh_xa, sh_xb, (shhi_q ? yb_ih : yb_i), she_q);
    if (data_mm)
      $display("MISMATCH row=%0d k=%0d got=%h exp=%h",
               emit_row, chk_k, m_data, dexp[emit_row][chk_k]);
    if (last_mm) $display("MISMATCH last row=%0d k=%0d", emit_row, chk_k);
    if (spur_mm) $display("MISMATCH: output during bad row %0d", emit_row);
    n_mism <= n_mism + int'(mism_add);
    shv_q  <= 1'b0;
    if (accept && cur_good) begin
      sh_u   <= phs[emit_row][mon_pair];
      sh_xa  <= din[emit_row][mon_pair];
      sh_xb  <= din[emit_row][mon_pair + int'(HALF)];
      shhi_q <= (chk_k >= int'(HALF));
      she_q  <= dexp[emit_row][chk_k];
      shv_q  <= 1'b1;
    end
    if (accept) begin
      n_beats <= n_beats + 1;
      if (chk_k == int'(D) - 1) begin
        chk_k <= 0;
        -> row_done_ev;
      end else begin
        chk_k <= chk_k + 1;
      end
    end
    if (frame_error && rst_n) n_errs <= n_errs + 1;
    if (n_mism > 20) $fatal(1, "ROPEROW FAIL: too many mismatches");
  end

  // ── driver ────────────────────────────────────────────────────────────────
  task automatic drive_beat(input logic [15:0] d, input logic l);
    begin
      // input stalls
      while (stall_mode != 0 && (rng[11:9] == 3'b101)) @(posedge clk);
      s_data  = d;
      s_last  = l;
      s_valid = 1'b1;
      do @(posedge clk); while (!s_ready);
      s_valid = 1'b0;
      s_last  = 1'b0;
    end
  endtask

  task automatic pulse_reset();
    begin
      rst_n   = 1'b0;
      s_valid = 1'b0;
      s_last  = 1'b0;
      repeat (2) @(posedge clk);
      rst_n = 1'b1;
      @(posedge clk);
      n_resets = n_resets + 1;
    end
  endtask

  int r;
  initial begin
    string vfile;
    int seed;
    if (!$value$plusargs("vectors=%s", vfile))
      vfile = $sformatf("build/vectors_rope_d%0d.txt", D);
    if (!$value$plusargs("bp_mode=%d", bp_mode)) bp_mode = 1;
    if (!$value$plusargs("stall_mode=%d", stall_mode)) stall_mode = 1;
    if (!$value$plusargs("seed=%d", seed)) seed = 32'h505E;
    if (!$value$plusargs("resets=%d", resets_every)) resets_every = 0;
    rng = 32'(seed);
    load_vectors(vfile);

    n_beats = 0; n_mism = 0; n_errs = 0; n_exp_errs = 0; n_resets = 0;
    emit_row = 0; chk_k = 0;
    s_valid = 0; s_last = 0; s_data = '0; shv_q = 0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    for (r = 0; r < n_rows; r++) begin
      emit_row = r;
      chk_k    = 0;
      if (resets_every != 0 && (r % resets_every) == (resets_every - 1)
          && kind[r] == 0) begin
        // drive a PARTIAL row, reset mid-flight, abandon it (clean-row
        // recovery is proven by the NEXT rows checking clean)
        for (int i = 0; i < D / 2 + 3; i++)
          drive_beat(din[r][i], 1'b0);
        pulse_reset();
        continue;
      end
      for (int i = 0; i < din_n[r]; i++)
        drive_beat(din[r][i],
                   (kind[r] == 1) ? (i == arg[r])          // early last
                                  : (i == din_n[r] - 1));  // normal/resync
      if (kind[r] == 0) begin
        @(row_done_ev);                     // all D beats checked
        @(posedge clk);
      end else begin
        n_exp_errs++;
        repeat (6) @(posedge clk);          // give the err pulse time
      end
    end
    repeat (20) @(posedge clk);

    if (resets_every == 0 && n_errs != n_exp_errs) begin
      $display("ROPEROW FAIL: frame_error pulses %0d != expected %0d",
               n_errs, n_exp_errs);
      $fatal(1, "ROPEROW FAIL");
    end
    if ((n_exp_errs > 0) && resets_every == 0 && !frame_error_sticky)
      $fatal(1, "ROPEROW FAIL: sticky not set after bad frames");
    if (n_mism == 0) begin
      $display("ROPEROW RESULT: rows=%0d beats=%0d errs=%0d resets=%0d mismatches=0 -> PASS",
               n_rows, n_beats, n_errs, n_resets);
      $display("ROPEROW PASS");
      $finish;
    end else begin
      $fatal(1, "ROPEROW FAIL: %0d mismatches", n_mism);
    end
  end

  initial begin
    #200_000_000;
    $fatal(1, "ROPEROW FAIL: watchdog");
  end

endmodule
