// tb_stagebuf_patd.sv — F-5 directed unit regression for apex_stage_buf at
// D=128 (kept-passing; found 2026-07-09, fixed same day).
//
//   F-5a: a LOAD / PAT_ROW EMIT job with nb = BPR = 16 must be LEGAL — the
//         pre-fix 4-bit job_nb truncated 16 -> 0 (illegal job, silent wedge).
//   F-5b: PAT_D emits of column blocks sel = 8..15 must read the ACTUAL
//         block — the pre-fix 4'(sel_q[2:0]) slice aliased blocks 8..15 onto
//         0..7. This test loads a pattern whose byte value depends on the
//         BEAT INDEX (delta over 8 beats is 13*8 = 104 != 0 mod 256), reads
//         back every column block 0..15, and fails byte-exact on any alias.
//
// Plus: PAT_D zero-fill of lanes >= rows, PAT_ROW full-row (nb=16) replay,
// and the sel >= BPR legality reject (pulse + sticky, no state change).
// Watchdogged; self-checking; -Wall clean.

module tb_stagebuf_patd;

  import apex_pkg::*;

  localparam int unsigned D     = 128;
  localparam int unsigned R_MAX = 16;
  localparam int unsigned BPR   = D / 8;            // 16 beats per row
  localparam int unsigned NB_W  = $clog2(BPR) + 1;  // 5 (F-5a derived width)
  localparam int unsigned ROWS  = 8;                // loaded rows (= MXE_N)

  // ── clock / reset / watchdog ───────────────────────────────────────────────
  logic clk, rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    repeat (2_000_000) @(posedge clk);
    $fatal(1, "WATCHDOG: stagebuf PAT_D unit test did not finish");
  end

  // ── DUT ────────────────────────────────────────────────────────────────────
  logic            job_valid, job_ready, job_op, job_bank;
  logic [1:0]      job_pat;
  logic [4:0]      job_rows, job_sel;
  logic [NB_W-1:0] job_nb;
  logic            job_error, job_error_sticky, busy, done;

  logic        ld_valid, ld_ready;
  lane8_beat_t ld_beat;
  logic        em_valid, em_ready;
  lane8_beat_t em_beat;

  apex_stage_buf #(.D(D), .R_MAX(R_MAX)) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (job_valid),
    .job_ready        (job_ready),
    .job_op           (job_op),
    .job_bank         (job_bank),
    .job_pat          (job_pat),
    .job_rows         (job_rows),
    .job_nb           (job_nb),
    .job_sel          (job_sel),
    .job_error        (job_error),
    .job_error_sticky (job_error_sticky),
    .busy             (busy),
    .done             (done),
    .ld_valid         (ld_valid),
    .ld_ready         (ld_ready),
    .ld_beat          (ld_beat),
    .em_valid         (em_valid),
    .em_ready         (em_ready),
    .em_beat          (em_beat)
  );

  logic unused_ok;
  assign unused_ok = busy;

  // ── scoreboard ─────────────────────────────────────────────────────────────
  int n_checks, n_errors;
  logic [64:0] em_q [$];               // {data, last}

  assign em_ready = 1'b1;
  always @(posedge clk) begin
    if (rst_n && em_valid) em_q.push_back({em_beat.data, em_beat.last});
  end

  // pattern: byte value depends on (row, BEAT, lane-byte); the beat term has
  // delta 13 per beat, and 13*8 = 104 != 0 mod 256, so block sel vs sel-8
  // aliasing ALWAYS changes the data (the F-5b discriminator)
  function automatic logic [7:0] pat_byte(input int unsigned r,
                                          input int unsigned b,
                                          input int unsigned c);
    return 8'((r * 32 + b * 13 + c * 67) % 256);
  endfunction

  function automatic logic [63:0] pat_word(input int unsigned r,
                                           input int unsigned b);
    logic [63:0] w;
    w = '0;
    for (int unsigned c = 0; c < 8; c++) w[8*c +: 8] = pat_byte(r, b, c);
    return w;
  endfunction

  // ── tasks ──────────────────────────────────────────────────────────────────
  task automatic drive_job(input logic op, input logic bank,
                           input logic [1:0] pat, input logic [4:0] rows,
                           input logic [NB_W-1:0] nb, input logic [4:0] sel,
                           input logic exp_error);
    int wd = 0;
    @(negedge clk);
    job_op    = op;
    job_bank  = bank;
    job_pat   = pat;
    job_rows  = rows;
    job_nb    = nb;
    job_sel   = sel;
    job_valid = 1'b1;
    while (!job_ready) begin
      @(negedge clk);
      wd++;
      if (wd > 100000) $fatal(1, "job stall @%0d", cyc);
    end
    @(negedge clk);
    job_valid = 1'b0;
    // job_error pulses at the posedge that accepted the job — visible here
    n_checks++;
    if (job_error !== exp_error) begin
      n_errors++;
      $error("[job] error pulse got %0b exp %0b @%0d", job_error, exp_error,
             cyc);
    end
  endtask

  task automatic wait_done;
    int wd = 0;
    while (!done) begin
      @(negedge clk);
      wd++;
      if (wd > 100000) $fatal(1, "done stall @%0d", cyc);
    end
  endtask

  task automatic drive_beat(input logic [63:0] w, input logic lst);
    int wd = 0;
    @(negedge clk);
    ld_beat.data = w;
    ld_beat.last = lst;
    ld_valid     = 1'b1;
    while (!ld_ready) begin
      @(negedge clk);
      wd++;
      if (wd > 100000) $fatal(1, "load stall @%0d", cyc);
    end
    @(negedge clk);
    ld_valid = 1'b0;
  endtask

  task automatic expect_beat(input logic [63:0] w, input logic lst,
                             input string what);
    int wd = 0;
    logic [64:0] got;
    while (em_q.size() == 0) begin
      @(posedge clk);
      wd++;
      if (wd > 100000) $fatal(1, "%s: emit timeout @%0d", what, cyc);
    end
    got = em_q.pop_front();
    n_checks++;
    if (got !== {w, lst}) begin
      n_errors++;
      $error("[%s] got %016x/%0b exp %016x/%0b", what, got[64:1], got[0],
             w, lst);
    end
  endtask

  // ── test ───────────────────────────────────────────────────────────────────
  initial begin
    logic [63:0] exp_w;
    rst_n = 1'b0;
    job_valid = 1'b0; job_op = 1'b0; job_bank = 1'b0; job_pat = '0;
    job_rows = '0; job_nb = '0; job_sel = '0;
    ld_valid = 1'b0; ld_beat = '0;
    n_checks = 0; n_errors = 0;
    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (3) @(negedge clk);

    // F-5a: LOAD 8 rows x nb=16 (= BPR, expressible only with the derived
    // NB_W field) into bank 0
    $display("[%0d] LOAD bank0: %0d rows x nb=%0d (F-5a)", cyc, ROWS, BPR);
    drive_job(1'b0, 1'b0, 2'd0, 5'(ROWS), NB_W'(BPR), 5'd0, 1'b0);
    for (int unsigned r = 0; r < ROWS; r++)
      for (int unsigned b = 0; b < BPR; b++)
        drive_beat(pat_word(r, b), (r == ROWS - 1) && (b == BPR - 1));
    wait_done();

    // F-5b: PAT_D readback of EVERY column block, 8..15 included — beat c
    // lane r must be stored row r, beat sel, byte c
    for (int unsigned sel = 0; sel < BPR; sel++) begin
      $display("[%0d] EMIT PAT_D sel=%0d (F-5b)", cyc, sel);
      drive_job(1'b1, 1'b0, 2'd2, 5'(ROWS), NB_W'(1), 5'(sel), 1'b0);
      for (int unsigned c = 0; c < 8; c++) begin
        exp_w = '0;
        for (int unsigned r = 0; r < ROWS; r++)
          exp_w[8*r +: 8] = pat_byte(r, sel, c);
        expect_beat(exp_w, c == 7, $sformatf("PAT_D sel=%0d beat=%0d", sel, c));
      end
      wait_done();
    end

    // PAT_D zero-fill: rows=3 < MXE_N — lanes 3..7 must be zero
    $display("[%0d] EMIT PAT_D sel=%0d rows=3 (zero-fill)", cyc, BPR - 1);
    drive_job(1'b1, 1'b0, 2'd2, 5'd3, NB_W'(1), 5'(BPR - 1), 1'b0);
    for (int unsigned c = 0; c < 8; c++) begin
      exp_w = '0;
      for (int unsigned r = 0; r < 3; r++)
        exp_w[8*r +: 8] = pat_byte(r, BPR - 1, c);
      expect_beat(exp_w, c == 7, $sformatf("PAT_D zfill beat=%0d", c));
    end
    wait_done();

    // F-5a on the emit side: PAT_ROW full-row replay, nb = BPR = 16
    $display("[%0d] EMIT PAT_ROW row 3, nb=%0d (F-5a)", cyc, BPR);
    drive_job(1'b1, 1'b0, 2'd0, 5'd1, NB_W'(BPR), 5'd3, 1'b0);
    for (int unsigned b = 0; b < BPR; b++)
      expect_beat(pat_word(3, b), b == BPR - 1,
                  $sformatf("PAT_ROW beat=%0d", b));
    wait_done();

    // legality reject: PAT_D sel = BPR (out of range) — pulse + sticky,
    // no beats, and the block accepts a legal job right after
    $display("[%0d] EMIT PAT_D sel=%0d (illegal, expect reject)", cyc, BPR);
    drive_job(1'b1, 1'b0, 2'd2, 5'(ROWS), NB_W'(1), 5'(BPR), 1'b1);
    n_checks++;
    if (job_error_sticky !== 1'b1) begin
      n_errors++;
      $error("[reject] sticky not set");
    end
    n_checks++;
    if (em_q.size() != 0) begin
      n_errors++;
      $error("[reject] emitted %0d beats after illegal job", em_q.size());
    end
    drive_job(1'b1, 1'b0, 2'd2, 5'(ROWS), NB_W'(1), 5'(BPR - 1), 1'b0);
    for (int unsigned c = 0; c < 8; c++) begin
      exp_w = '0;
      for (int unsigned r = 0; r < ROWS; r++)
        exp_w[8*r +: 8] = pat_byte(r, BPR - 1, c);
      expect_beat(exp_w, c == 7, $sformatf("post-reject beat=%0d", c));
    end
    wait_done();

    n_checks++;
    if (em_q.size() != 0) begin
      n_errors++;
      $error("[final] %0d leftover emit beats", em_q.size());
    end
    $display("STAGEBUF PATD RESULT: cycles=%0d checks=%0d errors=%0d",
             cyc, n_checks, n_errors);
    if (n_errors != 0) $fatal(1, "STAGEBUF PATD FAIL: %0d errors", n_errors);
    $display("STAGEBUF PATD PASS");
    $finish;
  end

endmodule
