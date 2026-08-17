// tb_tip_smoke.sv — TIP smoke testbench (Verilator --binary --timing --assert).
//
// Implements the ARCHITECTURE.md Layer-1 discipline for TIP (D-011/D-012/
// D-017): vector-driven scoreboard TB — all expected decisions, tiers and
// importance state come from the Python golden model in gen_tip_vectors.py
// (dual-oracle checked against the unbounded math rule at generation time);
// the TB only drives, collects and compares. The §5 stream contract and the
// V0.3 F-2 framing-guard sticky semantics are additionally checked every
// cycle by the bound tip_sva checker.
//
// Modes (plusargs):
//   vector mode (default): +scores/+meta/+impfinal/+tiles/+threshold/
//     +imp_hi/+imp_lo (the gen script writes these into <run>.args),
//     +gaps=1 random s_valid idle cycles, +bp=1 random d_ready backpressure.
//     Checks per decision beat: fp16 + tier + blk in tile order; at the end:
//     decision count, zero frame errors, busy deassertion, and the FULL
//     128-block importance state + rd_tier via the CSR read port.
//   +frame_test=1: directed V0.3 F-2 framing-guard regression — an (N+1)-score
//     tile (overrun ON the s_last beat) and an (N+5)-score tile (overrun ->
//     abort -> drain) each produce exactly one frame_err pulse and NO decision
//     beat; state is clean after (following tiles decide correctly, erroring
//     tiles leave NO importance update); sticky set/clear via frame_err_clear.
//
// TB discipline (V0.3/mxe lineage): all stimulus changes at NEGEDGE, checks
// sample 1 ns after the edge (race-free under Verilator --timing, V0.3 F-3);
// $fatal watchdog bounds every run.

`include "tip_sva.svh"

module tb_tip_smoke #(
    parameter int unsigned BLOCK_M    = 8,
    parameter int unsigned BLOCK_N    = 8,
    parameter int unsigned MAX_SCORES = 262144,
    parameter int unsigned MAX_TILES  = 8192
);

  import apex_pkg::*;

  localparam int unsigned SCORE_WIDTH = 32;   // D-011
  localparam int unsigned BLOCKS      = 128;
  localparam int unsigned ACC_W       = 16;
  localparam int unsigned N_TILE      = BLOCK_M * BLOCK_N;

  // ── clock / reset ──────────────────────────────────────────────────────────
  logic clk;
  logic rst_n;
  initial begin
    clk = 1'b0;
    forever #2.5 clk = ~clk;
  end

  // ── DUT ────────────────────────────────────────────────────────────────────
  logic                          s_valid, s_ready, s_last;
  logic signed [SCORE_WIDTH-1:0] s_data;
  logic [6:0]                    s_blk;
  logic                          d_valid, d_ready, d_fp16;
  kvq_tier_e                     d_tier;
  logic [6:0]                    d_blk;
  logic [4:0]                    threshold;
  logic [ACC_W-1:0]              imp_thresh_hi, imp_thresh_lo;
  logic                          imp_clear;
  logic [6:0]                    imp_rd_addr;
  logic [ACC_W-1:0]              imp_rd_data;
  kvq_tier_e                     imp_rd_tier;
  logic                          frame_err, frame_err_sticky, frame_err_clear;
  logic                          busy;

  tip_top #(
    .SCORE_WIDTH (SCORE_WIDTH),
    .BLOCK_M     (BLOCK_M),
    .BLOCK_N     (BLOCK_N),
    .BLOCKS      (BLOCKS),
    .ACC_W       (ACC_W)
  ) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (s_valid),
    .s_ready          (s_ready),
    .s_data           (s_data),
    .s_last           (s_last),
    .s_blk            (s_blk),
    .d_valid          (d_valid),
    .d_ready          (d_ready),
    .d_fp16           (d_fp16),
    .d_tier           (d_tier),
    .d_blk            (d_blk),
    .threshold        (threshold),
    .imp_thresh_hi    (imp_thresh_hi),
    .imp_thresh_lo    (imp_thresh_lo),
    .imp_clear        (imp_clear),
    .imp_rd_addr      (imp_rd_addr),
    .imp_rd_data      (imp_rd_data),
    .imp_rd_tier      (imp_rd_tier),
    .frame_err        (frame_err),
    .frame_err_sticky (frame_err_sticky),
    .frame_err_clear  (frame_err_clear),
    .busy             (busy)
  );

  // §5 + F-2 contract checker, bound into the DUT (compiled under --assert)
  bind tip_top tip_sva #(.SCORE_WIDTH(32), .BLOCKS(128)) u_tip_sva (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (s_valid),
    .s_ready          (s_ready),
    .s_data           (s_data),
    .s_last           (s_last),
    .s_blk            (s_blk),
    .d_valid          (d_valid),
    .d_ready          (d_ready),
    .d_fp16           (d_fp16),
    .d_tier           (d_tier),
    .d_blk            (d_blk),
    .frame_err        (frame_err),
    .frame_err_sticky (frame_err_sticky),
    .frame_err_clear  (frame_err_clear),
    .busy             (busy)
  );

  // ── plusargs ───────────────────────────────────────────────────────────────
  string  scores_f, meta_f, impfinal_f;
  int unsigned num_tiles, thr_arg, hi_arg, lo_arg;
  int unsigned gaps, bp, frame_test, seed;

  // ── storage / bookkeeping ──────────────────────────────────────────────────
  logic [31:0] scores_mem [0:MAX_SCORES-1];
  logic [31:0] meta_mem   [0:MAX_TILES-1];
  logic [15:0] impf_mem   [0:BLOCKS-1];

  int unsigned dec_count;
  int unsigned err_count;
  int unsigned fails;
  logic [9:0]  frame_dec_q [$];        // frame mode: {blk[6:0], tier[1:0], fp16}

  // meta unpack helpers (callers pass the exact field slice)
  function automatic int unsigned meta_len (input logic [12:0] f);
    return int'(f);
  endfunction

  // TB mirror of the tier rule (for the CSR read-port rd_tier check)
  function automatic logic [1:0] tier_of(input logic [15:0] a);
    if      (a >= imp_thresh_hi[15:0]) tier_of = 2'(KVQ_CQ8);
    else if (a >= imp_thresh_lo[15:0]) tier_of = 2'(KVQ_CQ4P);
    else                               tier_of = 2'(KVQ_CQ4);
  endfunction

  // ── watchdog ───────────────────────────────────────────────────────────────
  longint unsigned cycle_cnt;
  longint unsigned budget;
  initial begin
    cycle_cnt = 0;
    budget    = 64'd1_000_000_000;     // tightened after plusargs parse
  end
  always @(posedge clk) begin
    cycle_cnt <= cycle_cnt + 64'd1;
    if (cycle_cnt > budget)
      $fatal(1, "WATCHDOG: exceeded %0d cycles (tiles=%0d N=%0d) — hang",
             budget, num_tiles, N_TILE);
  end

  // ── frame_err pulse counter (post-NBA sample, V0.3 style) ─────────────────
  always begin
    @(posedge clk);
    #1;
    if (rst_n && frame_err) err_count = err_count + 1;
  end

  // ── decision consumer: random backpressure + in-order scoreboard ─────────
  logic [9:0] cur_meta;   // {blk[6:0], tier[1:0], fp16} of the expected tile
  always begin
    @(negedge clk);
    if (!rst_n) begin
      d_ready = 1'b0;
    end else begin
      d_ready = (bp != 0) ? ($urandom_range(0, 3) != 0) : 1'b1;
      #1;
      if (d_valid && d_ready) begin
        // both stable until the accepting posedge (d_valid post-skid
        // registered, d_ready only changes at negedge) — safe capture-ahead
        if (frame_test != 0) begin
          frame_dec_q.push_back({d_blk, 2'(d_tier), d_fp16});
        end else begin
          if (dec_count >= num_tiles) begin
            $display("[FAIL] extra decision beat #%0d (expected %0d total)",
                     dec_count + 1, num_tiles);
            fails = fails + 1;
          end else begin
            cur_meta = meta_mem[dec_count][9:0];
            if (d_fp16 !== cur_meta[0] ||
                2'(d_tier) !== cur_meta[2:1] ||
                d_blk  !== cur_meta[9:3]) begin
              $display("[FAIL] tile %0d: got fp16=%0b tier=%0d blk=%0d, exp fp16=%0b tier=%0d blk=%0d",
                       dec_count, d_fp16, 2'(d_tier), d_blk,
                       cur_meta[0], cur_meta[2:1],
                       cur_meta[9:3]);
              fails = fails + 1;
            end
          end
        end
        dec_count = dec_count + 1;
      end
    end
  end

  // ── score-stream driver primitive ──────────────────────────────────────────
  // Drives at NEGEDGE, holds the beat stable until s_ready is seen high (the
  // beat is then accepted at the following posedge) — §5 compliant.
  task automatic send_beat(input logic [31:0] data, input logic last,
                           input logic [6:0] blk);
    @(negedge clk);
    if (gaps != 0) begin
      while ($urandom_range(0, 3) == 0) begin
        s_valid = 1'b0;
        s_last  = 1'b0;
        @(negedge clk);
      end
    end
    s_valid = 1'b1;
    s_data  = signed'(data);
    s_last  = last;
    s_blk   = blk;
    #1;
    while (!s_ready) begin
      @(negedge clk);
      #1;
    end
  endtask

  task automatic idle_input();
    @(negedge clk);
    s_valid = 1'b0;
    s_last  = 1'b0;
  endtask

  // send a whole tile from a queue of values
  task automatic send_tile(input logic [31:0] vals [$], input logic [6:0] blk);
    for (int i = 0; i < vals.size(); i++)
      send_beat(vals[i], i == vals.size() - 1, blk);
    idle_input();
  endtask

  // wait for the decision count to reach n (watchdog bounds the wait)
  task automatic wait_decisions(input int unsigned n);
    while (dec_count < n) @(negedge clk);
    repeat (10) @(negedge clk);        // drain skids / late stragglers
  endtask

  // read + check one importance accumulator via the CSR read port
  task automatic check_imp(input logic [6:0] addr, input logic [15:0] exp_acc);
    @(negedge clk);
    imp_rd_addr = addr;
    #1;
    if (imp_rd_data !== exp_acc) begin
      $display("[FAIL] importance[%0d]: rd_data=%0d exp=%0d",
               addr, imp_rd_data, exp_acc);
      fails = fails + 1;
    end
    if (2'(imp_rd_tier) !== tier_of(exp_acc)) begin
      $display("[FAIL] importance[%0d]: rd_tier=%0d exp=%0d (acc=%0d)",
               addr, 2'(imp_rd_tier), tier_of(exp_acc), exp_acc);
      fails = fails + 1;
    end
  endtask

  // ── frame-mode helpers ─────────────────────────────────────────────────────
  task automatic frame_expect_dec(input string what, input logic exp_fp16,
                                  input logic [6:0] exp_blk);
    logic [9:0] got;
    if (frame_dec_q.size() == 0) begin
      $display("[FAIL] frame: missing decision beat for %s", what);
      fails = fails + 1;
      return;
    end
    got = frame_dec_q.pop_front();
    // all frame-mode tiles stay far below imp_lo=32768 -> tier is always CQ4
    if (got[0] !== exp_fp16 || got[9:3] !== exp_blk ||
        got[2:1] !== 2'(KVQ_CQ4)) begin
      $display("[FAIL] frame %s: got fp16=%0b tier=%0d blk=%0d, exp fp16=%0b tier=%0d blk=%0d",
               what, got[0], got[2:1], got[9:3],
               exp_fp16, 2'(KVQ_CQ4), exp_blk);
      fails = fails + 1;
    end
  endtask

  task automatic frame_check(input string what, input bit cond);
    if (!cond) begin
      $display("[FAIL] frame: %s", what);
      fails = fails + 1;
    end
  endtask

  // ── main ───────────────────────────────────────────────────────────────────
  int unsigned total_scores;
  int unsigned ptr;
  logic [31:0] tile_q [$];

  initial begin
    // defaults / init
    rst_n           = 1'b0;
    s_valid         = 1'b0;
    s_data          = '0;
    s_last          = 1'b0;
    s_blk           = '0;
    d_ready         = 1'b0;
    threshold       = 5'd10;
    imp_thresh_hi   = 16'hFFFF;
    imp_thresh_lo   = 16'h8000;
    imp_clear       = 1'b0;
    imp_rd_addr     = '0;
    frame_err_clear = 1'b0;
    dec_count       = 0;
    err_count       = 0;
    fails           = 0;
    num_tiles       = 0;
    thr_arg         = 10;
    hi_arg          = 32'h0000_FFFF;
    lo_arg          = 32'h0000_8000;
    gaps            = 0;
    bp              = 0;
    frame_test      = 0;
    seed            = 1;

    void'($value$plusargs("gaps=%d", gaps));
    void'($value$plusargs("bp=%d", bp));
    void'($value$plusargs("frame_test=%d", frame_test));
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("threshold=%d", thr_arg));
    void'($value$plusargs("imp_hi=%d", hi_arg));
    void'($value$plusargs("imp_lo=%d", lo_arg));
    void'($urandom(seed));   // seed this process's RNG (deterministic runs)

    threshold     = 5'(thr_arg);
    imp_thresh_hi = 16'(hi_arg);
    imp_thresh_lo = 16'(lo_arg);

    if (frame_test != 0) begin
      budget = 64'd200_000 + longint'(N_TILE) * 64'd200;
      run_frame_test();
    end else begin
      if (!$value$plusargs("scores=%s", scores_f))
        $fatal(1, "missing +scores=<file>");
      if (!$value$plusargs("meta=%s", meta_f))
        $fatal(1, "missing +meta=<file>");
      if (!$value$plusargs("impfinal=%s", impfinal_f))
        $fatal(1, "missing +impfinal=<file>");
      if (!$value$plusargs("tiles=%d", num_tiles))
        $fatal(1, "missing +tiles=<n>");
      if (num_tiles > MAX_TILES)
        $fatal(1, "tiles=%0d exceeds TB storage", num_tiles);
      $readmemh(scores_f, scores_mem);
      $readmemh(meta_f, meta_mem);
      $readmemh(impfinal_f, impf_mem);

      total_scores = 0;
      for (int t = 0; t < int'(num_tiles); t++) begin
        if (meta_len(meta_mem[t][22:10]) == 0 || meta_len(meta_mem[t][22:10]) > N_TILE)
          $fatal(1, "meta tile %0d: len=%0d out of [1,%0d]",
                 t, meta_len(meta_mem[t][22:10]), N_TILE);
        total_scores += meta_len(meta_mem[t][22:10]);
      end
      if (total_scores > MAX_SCORES)
        $fatal(1, "total scores %0d exceeds TB storage", total_scores);
      budget = longint'(total_scores) * (gaps != 0 ? 64'd8 : 64'd4)
             + longint'(num_tiles) * 64'd64 + 64'd10_000;
      run_vector_mode();
    end
  end

  task automatic run_vector_mode();
    $display("[TB] vector mode: N=%0d tiles=%0d scores=%0d T=%0d hi=%0d lo=%0d gaps=%0d bp=%0d budget=%0d",
             N_TILE, num_tiles, total_scores, thr_arg, hi_arg, lo_arg,
             gaps, bp, budget);
    repeat (4) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    ptr = 0;
    for (int t = 0; t < int'(num_tiles); t++) begin
      int unsigned len;
      len = meta_len(meta_mem[t][22:10]);
      for (int unsigned i = 0; i < len; i++)
        send_beat(scores_mem[ptr + i], i == len - 1, meta_mem[t][9:3]);
      ptr += len;
      if ((t + 1) % 1000 == 0)
        $display("[TB] ... %0d/%0d tiles driven, decisions=%0d fails=%0d",
                 t + 1, num_tiles, dec_count, fails);
    end
    idle_input();
    wait_decisions(num_tiles);

    if (dec_count !== num_tiles) begin
      $display("[FAIL] decision count %0d != tiles %0d", dec_count, num_tiles);
      fails = fails + 1;
    end
    if (err_count != 0 || frame_err_sticky) begin
      $display("[FAIL] unexpected frame errors: pulses=%0d sticky=%0b",
               err_count, frame_err_sticky);
      fails = fails + 1;
    end
    if (busy) begin
      $display("[FAIL] busy still high after drain");
      fails = fails + 1;
    end
    // full importance state + rd_tier via the CSR read port
    for (int b = 0; b < int'(BLOCKS); b++)
      check_imp(7'(b), impf_mem[b]);

    report();
  endtask

  // directed V0.3 F-2 framing-guard regression
  task automatic run_frame_test();
    $display("[TB] frame_test: N=%0d T=%0d budget=%0d", N_TILE, thr_arg, budget);
    repeat (4) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    // A: clean all-ones tile -> INT8 (1*N < T*N for T=10), imp[3] += 1
    tile_q.delete();
    repeat (N_TILE) tile_q.push_back(32'd1);
    send_tile(tile_q, 7'd3);
    wait_decisions(1);
    frame_expect_dec("tile A (clean int8)", 1'b0, 7'd3);
    frame_check("no error after clean tile", err_count == 0 && !frame_err_sticky);

    // B: N+1 scores, overrun ON the s_last beat -> 1 pulse, NO decision
    tile_q.delete();
    repeat (N_TILE + 1) tile_q.push_back(32'd1);
    send_tile(tile_q, 7'd3);
    repeat (20) @(negedge clk);
    frame_check("tile B: exactly one frame_err pulse", err_count == 1);
    frame_check("tile B: sticky set", frame_err_sticky === 1'b1);
    frame_check("tile B: no decision beat", frame_dec_q.size() == 0);

    // C: clean spike tile -> FP16 (state clean after the error), imp[3] += 64
    tile_q.delete();
    tile_q.push_back(32'h8000_0000);                 // INT32_MIN, |.| = 2^31
    repeat (N_TILE - 1) tile_q.push_back(32'd0);
    send_tile(tile_q, 7'd3);
    wait_decisions(2);
    frame_expect_dec("tile C (clean fp16 after error)", 1'b1, 7'd3);
    frame_check("sticky still set (no clear yet)", frame_err_sticky === 1'b1);

    // clear the sticky bit via CSR pulse
    @(negedge clk);
    frame_err_clear = 1'b1;
    @(negedge clk);
    frame_err_clear = 1'b0;
    repeat (2) @(negedge clk);
    frame_check("sticky cleared by frame_err_clear", frame_err_sticky === 1'b0);

    // D: N+5 scores (overrun -> abort -> drain to s_last) -> 1 pulse, no beat
    tile_q.delete();
    repeat (N_TILE + 5) tile_q.push_back(32'd1);
    send_tile(tile_q, 7'd3);
    repeat (20) @(negedge clk);
    frame_check("tile D: second frame_err pulse", err_count == 2);
    frame_check("tile D: sticky re-set", frame_err_sticky === 1'b1);
    frame_check("tile D: no decision beat", frame_dec_q.size() == 0);

    // E: clean all-ones tile again -> INT8, imp[3] += 1
    tile_q.delete();
    repeat (N_TILE) tile_q.push_back(32'd1);
    send_tile(tile_q, 7'd3);
    wait_decisions(3);
    frame_expect_dec("tile E (clean int8 after abort)", 1'b0, 7'd3);

    frame_check("exactly 3 decision beats total", dec_count == 3);
    frame_check("busy deasserted at end", busy === 1'b0);
    // erroring tiles B and D must NOT have updated importance:
    // imp[3] = 1 (A) + 64 (C) + 1 (E) = 66
    check_imp(7'd3, 16'd66);

    report();
  endtask

  task automatic report();
    $display("================================================");
    $display(" tiles=%0d decisions=%0d frame_err_pulses=%0d fails=%0d",
             num_tiles, dec_count, err_count, fails);
    if (fails == 0) begin
      $display(" ALL TESTS PASSED");
      $display("================================================");
      $finish;
    end else begin
      $display(" *** FAILURES: %0d ***", fails);
      $display("================================================");
      $fatal(1, "TB FAILED");
    end
  endtask

endmodule
