// tb_mxe_struct.sv — D-004 STRUCTURAL discriminator for the MXE systolic
// array (whitebox, mxe_array standalone).
//
// WHAT THIS PROVES (and why the functional suite cannot):
//   The 836-job scoreboard suite proves the FUNCTIONAL contract (C = A·B,
//   bit-exact) — but a broadcast MAC wall (run1/run2 lineage, critic 1.15)
//   computes the same function. D-004's claim is STRUCTURAL: a true systolic
//   array with PE-to-PE pipelining, 1 cycle/hop. The discriminating property
//   is the documented timing law of mxe_array.sv:
//
//     P1  activation arrival is (r+c)-dependent: a beat presented at cycle T
//         appears on the west input of PE(r,c) — act_h[r][c] — at EXACTLY
//         cycle T+r+c, and at no other cycle (west-edge skew registers +
//         one registered act hop per column).
//     P2  partials hop north→south one registered stage per row: PE(r,c)'s
//         south output — psum_v[r+1][c] — carries the beat's row-0..r prefix
//         sum at EXACTLY cycle T+r+c+1.
//     P3  column accumulators are written column-staggered: acc[c][m] of a
//         beat injected at T changes at EXACTLY cycle T+MXE_N+c (visible
//         T+MXE_N+c+1) — the tag pipe's MXE_N+c delay.
//     P4  the pipelining SUSTAINS at 1 beat/cycle (8 back-to-back beats form
//         8 non-interfering wavefronts, each obeying P1/P2/P3).
//     P5  (GLITCH_TEST builds only) a single-cycle glitch forced onto ONE
//         PE-to-PE boundary register (PE(2,3).act_out) reaches column c
//         ≥ 4 only after c-4 further hops (one column per cycle), and is
//         NEVER visible in the same cycle at any farther column.
//
//   A broadcast MAC wall broadcasts the activation to ALL columns in the
//   same cycle, reduces each column combinationally, and writes all column
//   accumulators simultaneously — it satisfies the functional contract but
//   violates P1–P4 at almost every (r,c) probe. The mutation check
//   (verif/mxe/struct/mutant/mxe_array_bcast.sv, a functionally-equivalent
//   broadcast wall that PASSES tb_mxe_smoke) must FAIL this TB — that is
//   the discrimination proof. Note the mutant is a *realistic* wall
//   (run1/run2 style); a wall that added per-column delay lines purely to
//   mimic systolic-internal timing would still be caught by P5, which
//   requires a physical act path between adjacent columns.
//
// TB discipline: stimulus at negedge; the recorder samples at posedge and
// therefore captures the DURING-CYCLE value of every probe (pre-NBA reads);
// $fatal watchdog bounds the run. Verilator --binary --timing --assert.

module tb_mxe_struct;

  import apex_pkg::*;
  import mxe_cfg_pkg::*;

  localparam int N    = int'(MXE_N);   // 8
  localparam int WREC = 40;            // recording window (cycles)

  // ── clock / cycle counter ──────────────────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    repeat (200000) @(posedge clk);
    $fatal(1, "WATCHDOG: struct TB did not finish");
  end

  // ── DUT: the array alone (whitebox) ────────────────────────────────────────
  logic               en;
  logic [8*MXE_N-1:0] act_lanes;
  logic               beat_valid, beat_clear;
  logic [ROW_W-1:0]   beat_row;
  logic               live_sel, wload_en;
  logic [8*MXE_N-1:0] wload_lanes;
  logic [ROW_W-1:0]   rd_row;
  acc_t               rd_lanes [MXE_N];

  mxe_array u_arr (
    .clk         (clk),
    .rst_n       (rst_n),
    .en          (en),
    .act_lanes   (act_lanes),
    .beat_valid  (beat_valid),
    .beat_clear  (beat_clear),
    .beat_row    (beat_row),
    .live_sel    (live_sel),
    .wload_en    (wload_en),
    .wload_lanes (wload_lanes),
    .rd_row      (rd_row),
    .rd_lanes    (rd_lanes)
  );

  // rd port exercised only as a liveness sanity at the end
  logic unused_rd;
  always_comb begin
    unused_rd = 1'b0;
    for (int c = 0; c < N; c++) unused_rd = unused_rd ^ (^rd_lanes[c]);
  end

  // ── whitebox probe recorder ────────────────────────────────────────────────
  // rec_*[i] = the DURING-CYCLE value at offset i from arming:
  //   rec_act[i][r][c]  = act_h[r][c]    (west input of PE(r,c))
  //   rec_psum[i][r][c] = psum_v[r+1][c] (south output of PE(r,c))
  //   rec_acc[i][c][m]  = acc[c][m]      (column accumulators, rows m < 8)
  int rec_act  [WREC][8][8];
  int rec_psum [WREC][8][8];
  int rec_acc  [WREC][8][8];
  bit rec_on;
  int rec_idx;

  always @(posedge clk) begin
    if (rec_on && rec_idx < WREC) begin
      for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) begin
          rec_act[rec_idx][r][c]  <= int'(u_arr.act_h[r][c]);
          rec_psum[rec_idx][r][c] <= int'(u_arr.psum_v[r+1][c]);
          rec_acc[rec_idx][c][r]  <= int'(u_arr.acc[c][r]);
        end
      rec_idx <= rec_idx + 1;
    end
  end

  // consume mxe_cfg_pkg params this whitebox TB does not otherwise need,
  // as a real consistency assertion: the ctrl's flush window must cover the
  // array's full transit (vertical MXE_N + horizontal skew MXE_N-1)
  initial begin
    if (int'(FLUSH_CYC) < 2 * N - 1 || int'(PASS_W) < 1 || int'(ACNT_W) < 1)
      $fatal(1, "mxe_cfg_pkg sizing inconsistent with the array transit");
  end

  // ── error bookkeeping ──────────────────────────────────────────────────────
  int errors;
  int checks;
  localparam int MAX_PRINT = 25;

  function automatic void expect_int(input string what, input int i, r, c,
                                     input int got, input int exp);
    checks++;
    if (got != exp) begin
      errors++;
      if (errors <= MAX_PRINT)
        $display("STRUCT FAIL [%s] off=%0d r=%0d c=%0d: got %0d exp %0d",
                 what, i, r, c, got, exp);
      if (errors == MAX_PRINT + 1)
        $display("STRUCT FAIL: ... further mismatches suppressed ...");
    end
  endfunction

  // ── low-level drivers ──────────────────────────────────────────────────────
  task automatic do_reset();
    @(negedge clk);
    rst_n       = 1'b0;
    en          = 1'b0;
    act_lanes   = '0;
    beat_valid  = 1'b0;
    beat_clear  = 1'b0;
    beat_row    = '0;
    wload_en    = 1'b0;
    wload_lanes = '0;
    rd_row      = '0;
    rec_on      = 1'b0;
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);
    en = 1'b1;
    repeat (4) @(negedge clk);
  endtask

  // load all-ones weights into the shadow bank (MXE_N column strobes,
  // east-edge entry, west shift — the mxe_ctrl protocol), then make it live
  task automatic load_ones_and_swap();
    for (int s = 0; s < N; s++) begin
      @(negedge clk);
      wload_en    = 1'b1;
      wload_lanes = {N{8'h01}};
    end
    @(negedge clk);
    wload_en    = 1'b0;
    wload_lanes = '0;
    live_sel    = ~live_sel;
    repeat (2) @(negedge clk);
  endtask

  task automatic run_out_window();
    while (rec_idx < WREC) @(negedge clk);
    @(negedge clk);
    rec_on = 1'b0;
  endtask

  // ── Test A: single beat — P1/P2/P3 exact arrival offsets ──────────────────
  task automatic test_single_beat();
    int V [8];
    int P [8];
    int S;
    for (int r = 0; r < 8; r++) V[r] = r + 1;
    P[0] = V[0];
    for (int r = 1; r < 8; r++) P[r] = P[r-1] + V[r];
    S = P[7];   // full column dot product (weights all 1) = 36

    @(negedge clk);
    rec_idx = 0;
    rec_on  = 1'b1;
    for (int r = 0; r < 8; r++) act_lanes[8*r +: 8] = 8'(V[r]);
    beat_valid = 1'b1;
    beat_clear = 1'b1;
    beat_row   = '0;
    @(negedge clk);
    act_lanes  = '0;
    beat_valid = 1'b0;
    beat_clear = 1'b0;
    run_out_window();

    for (int i = 0; i < WREC; i++)
      for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) begin
          // P1: act_h[r][c] carries lane r EXACTLY at offset r+c
          expect_int("A/P1 act_h", i, r, c, rec_act[i][r][c],
                     (i == r + c) ? V[r] : 0);
          // P2: psum_v[r+1][c] carries prefix(r) EXACTLY at offset r+c+1
          expect_int("A/P2 psum_v", i, r, c, rec_psum[i][r][c],
                     (i == r + c + 1) ? P[r] : 0);
        end
    // P3: acc[c][0] written at END of offset MXE_N+c (visible MXE_N+c+1)
    for (int i = 0; i < WREC; i++)
      for (int c = 0; c < 8; c++)
        expect_int("A/P3 acc", i, 0, c, rec_acc[i][c][0],
                   (i >= N + c + 1) ? S : 0);
  endtask

  // ── Test B: 8 back-to-back beats — P4 sustained wavefronts ────────────────
  task automatic test_pipelined_beats();
    int b;
    @(negedge clk);
    rec_idx = 0;
    rec_on  = 1'b1;
    for (int m = 0; m < 8; m++) begin
      // beat m: all lanes = m+1, row m, clr (WS pass-0 semantics)
      for (int r = 0; r < 8; r++) act_lanes[8*r +: 8] = 8'(m + 1);
      beat_valid = 1'b1;
      beat_clear = 1'b1;
      beat_row   = ROW_W'(m);
      @(negedge clk);
    end
    act_lanes  = '0;
    beat_valid = 1'b0;
    beat_clear = 1'b0;
    beat_row   = '0;
    run_out_window();

    for (int i = 0; i < WREC; i++)
      for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) begin
          // P1/P4: act_h[r][c] at offset i carries beat b = i-r-c
          b = i - r - c;
          expect_int("B/P4 act_h", i, r, c, rec_act[i][r][c],
                     (b >= 0 && b < 8) ? (b + 1) : 0);
          // P2/P4: psum_v[r+1][c] at offset i carries beat b = i-1-r-c,
          // value prefix(r) of that beat = (r+1)*(b+1) (weights all 1)
          b = i - 1 - r - c;
          expect_int("B/P4 psum_v", i, r, c, rec_psum[i][r][c],
                     (b >= 0 && b < 8) ? ((r + 1) * (b + 1)) : 0);
        end
    // P3/P4: acc[c][m] (beat m injected at offset m) changes at m+MXE_N+c,
    // visible from offset m+MXE_N+c+1; value = 8*(m+1)
    for (int i = 0; i < WREC; i++)
      for (int c = 0; c < 8; c++)
        for (int m = 0; m < 8; m++)
          expect_int("B/P3 acc", i, m, c, rec_acc[i][c][m],
                     (i >= m + N + c + 1) ? (8 * (m + 1)) : 0);
  endtask

`ifdef GLITCH_TEST
  // ── Test C: single-cycle glitch at ONE PE boundary — P5 hop locality ──────
  // Force PE(GR,GC).act_out (the registered west→east boundary into column
  // GC+1) to GV for EXACTLY one cycle. In a true systolic array the glitch
  // walks east one column per cycle and ripples south one row per cycle; in
  // a broadcast wall there is no such boundary and any equivalent injection
  // is visible at all columns in the same cycle.
  localparam int GR = 2;
  localparam int GC = 3;
  localparam int GV = 77;

  task automatic test_glitch();
    // quiescence: en running, all-zero activations, weights all 1
    repeat (10) @(negedge clk);

    @(negedge clk);
    rec_idx = 0;
    rec_on  = 1'b1;
    // one-cycle glitch: procedural deposit onto the boundary register; the
    // PE's own always_ff (en held high, act_in zero) overwrites it at the
    // next posedge, so the corruption lives exactly one cycle.
    u_arr.g_row[GR].g_col[GC].u_pe.act_out = 8'sd77;
    run_out_window();

    for (int i = 0; i < WREC; i++)
      for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) begin
          // P5 act path: only row GR, only columns > GC, one hop per cycle:
          // act_h[GR][c] == GV exactly at offset c-(GC+1) for c >= GC+1
          expect_int("C/P5 act_h", i, r, c, rec_act[i][r][c],
                     (r == GR && c >= GC + 1 && i == c - (GC + 1)) ? GV : 0);
          // P5 psum ripple: PE(GR,c) adds GV*1 at offset c-(GC+1); the
          // partial then hops south one row per cycle:
          // psum_v[r+1][c] == GV at offset (c-(GC+1)) + 1 + (r-GR), r >= GR
          expect_int("C/P5 psum_v", i, r, c, rec_psum[i][r][c],
                     (r >= GR && c >= GC + 1
                      && i == (c - (GC + 1)) + 1 + (r - GR)) ? GV : 0);
          // no tags in flight: accumulators must not move
          expect_int("C/P5 acc", i, r, c, rec_acc[i][c][r], 0);
        end
  endtask
`endif

  // ── main ───────────────────────────────────────────────────────────────────
  initial begin : main
    errors   = 0;
    checks   = 0;
    rec_on   = 1'b0;
    rec_idx  = 0;
    live_sel = 1'b0;
    do_reset();

    load_ones_and_swap();
    test_single_beat();

    do_reset();
    load_ones_and_swap();
    test_pipelined_beats();

`ifdef GLITCH_TEST
    do_reset();
    load_ones_and_swap();
    test_glitch();
`endif

    if (errors == 0) begin
      $display("STRUCT PASS: %0d whitebox timing checks — (r+c)-offset act arrival, per-hop psum staggering, column-staggered acc writes%s (D-004 systolic discriminator)",
               checks,
`ifdef GLITCH_TEST
               ", single-cycle glitch hop locality"
`else
               ""
`endif
               );
      $finish;
    end else begin
      $fatal(1, "STRUCT FAIL: %0d/%0d checks failed — DUT does not exhibit systolic PE-to-PE pipelining (D-004)",
             errors, checks);
    end
  end

endmodule
