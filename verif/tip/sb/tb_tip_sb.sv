// tb_tip_sb.sv — TIP independent vector-driven scoreboard TB (house pattern,
// verif/mxe/sb lineage). Verilator --binary --timing --assert.
//
// All expected decisions / tiers / importance state come from the independent
// golden model in gen_tip_sb_vectors.py (dual-oracle: unbounded math vs the
// RTL's fixed widths, checked at generation time). The TB streams the vector
// file, drives the §5 score stream with configurable feed gaps, consumes the
// decision stream with configurable backpressure, and compares every beat in
// order. Record types (see the generator header): T tile / F overrun tile /
// C drain+CSR-compare+imp_clear / R phase-targeted mid-op reset / E end.
//
// The §5 stability core is bound from the house pack
// (verif/common/apex_stream1_sva.svh) onto BOTH streams; the TIP-shaped
// conservation/busy/framing contract is bound from tip_sb_sva.svh. Both are
// compiled under --assert in every build (D-012).
//
// Plusargs: +vectors=<file> +imp_hi= +imp_lo= +bp_mode=0|1|2 (none/75%/storm)
//           +gap_mode=0|1|2 (none/short/bursty) +seed=<n>
//
// TB discipline (V0/mxe lineage): stimulus at NEGEDGE, checks sampled 1 ns
// after the edge or at POSEDGE commits; $fatal watchdog bounds the run.

`include "apex_stream1_sva.svh"
`include "tip_sb_sva.svh"

module tb_tip_sb #(
    parameter int unsigned BLOCK_M = 8,
    parameter int unsigned BLOCK_N = 8
);

  import apex_pkg::*;

  localparam int unsigned SCORE_WIDTH = 32;
  localparam int unsigned BLOCKS      = 128;
  localparam int unsigned ACC_W       = 16;
  localparam int unsigned N_TILE      = BLOCK_M * BLOCK_N;
  localparam int unsigned MAXBUF      = N_TILE + 64;
  localparam int unsigned WAIT_LIMIT  = 2_000_000;   // per blocking wait

  // ── clock / cycle counter / watchdog ──────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    repeat (60_000_000) @(posedge clk);
    $fatal(1, "WATCHDOG: simulation did not finish (cyc=%0d)", cyc);
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

  // ── house §5 stability pack on BOTH streams (reused, not rewritten) ───────
  bind tip_top apex_stream1_sva #(.WIDTH(40), .NAME("tip.score"))
    u_sva_s (.clk(clk), .rst_n(rst_n), .valid(s_valid), .ready(s_ready),
             .data({s_blk, s_last, unsigned'(s_data)}));

  bind tip_top apex_stream1_sva #(.WIDTH(10), .NAME("tip.decision"))
    u_sva_d (.clk(clk), .rst_n(rst_n), .valid(d_valid), .ready(d_ready),
             .data({d_blk, 2'(d_tier), d_fp16}));

  // ── TIP conservation / busy / F-2 contract checker (white-box bind) ───────
  bind tip_top tip_sb_sva u_sva_tip (
    .clk              (clk),
    .rst_n            (rst_n),
    .d_valid          (d_valid),
    .d_ready          (d_ready),
    .d_tier           (d_tier),
    .frame_err        (frame_err),
    .frame_err_sticky (frame_err_sticky),
    .frame_err_clear  (frame_err_clear),
    .busy             (busy),
    .eng_d_valid      (eng_d_valid),
    .eng_fire         (eng_fire),
    .in_last          (in_last),
    .in_m_valid       (in_m_valid),
    .eng_tile_active  (eng_tile_active),
    .hold_valid       (hold_valid),
    .dec_pend         (dec_pend)
  );

  // ── plusargs ───────────────────────────────────────────────────────────────
  int unsigned bp_mode  = 1;
  int unsigned gap_mode = 1;
  int unsigned seed     = 32'h71D_2026;
  int unsigned hi_arg   = 1000;
  int unsigned lo_arg   = 100;
  string       vec_path;

  // ── bookkeeping ────────────────────────────────────────────────────────────
  int errors;
  int tiles_done, frames_done, resets_done, clears_done;
  int dec_checked, discarded;
  int err_seen, exp_errs;
  bit discard;         // sacrificial window around mid-op resets
  bit force_d0;        // hold d_ready low (out-skid fill for reset phases 5/6)
  logic [9:0] exp_q [$];        // {blk[6:0], tier[1:0], fp16} in tile order

  // ── coverage buckets (manual, run2 pattern; "COV <name> <n>") ─────────────
  int cov_thr [0:31];
  int cov_dec_fp16, cov_dec_int8, cov_tie_exact, cov_tie_off1;
  int cov_mag_zero, cov_mag_min, cov_mag_max;
  int cov_len_1, cov_len_short, cov_len_full;
  int cov_imp_inc0, cov_imp_sat, cov_imp_eq_lo, cov_imp_eq_hi;
  int cov_tier_cq4, cov_tier_cq4p, cov_tier_cq8;
  int cov_frame_onlast, cov_frame_abort, cov_sticky_clear, cov_imp_clear;
  int cov_rst [1:6];
  int cov_bp_full, cov_bp_random, cov_bp_storm;
  int cov_gap_none, cov_gap_random, cov_gap_storm;
  int cov_n64, cov_n4096;

  function automatic void print_coverage();
    for (int t = 0; t < 32; t++) $display("COV thr_t%0d %0d", t, cov_thr[t]);
    $display("COV dec_fp16 %0d",     cov_dec_fp16);
    $display("COV dec_int8 %0d",     cov_dec_int8);
    $display("COV tie_exact %0d",    cov_tie_exact);
    $display("COV tie_off1 %0d",     cov_tie_off1);
    $display("COV mag_zero_tile %0d", cov_mag_zero);
    $display("COV mag_int32min %0d", cov_mag_min);
    $display("COV mag_int32max %0d", cov_mag_max);
    $display("COV len_1 %0d",        cov_len_1);
    $display("COV len_short %0d",    cov_len_short);
    $display("COV len_full %0d",     cov_len_full);
    $display("COV imp_inc0 %0d",     cov_imp_inc0);
    $display("COV imp_sat %0d",      cov_imp_sat);
    $display("COV imp_eq_lo %0d",    cov_imp_eq_lo);
    $display("COV imp_eq_hi %0d",    cov_imp_eq_hi);
    $display("COV tier_cq4 %0d",     cov_tier_cq4);
    $display("COV tier_cq4p %0d",    cov_tier_cq4p);
    $display("COV tier_cq8 %0d",     cov_tier_cq8);
    $display("COV frame_onlast %0d", cov_frame_onlast);
    $display("COV frame_abort %0d",  cov_frame_abort);
    $display("COV sticky_clear %0d", cov_sticky_clear);
    $display("COV imp_clear_midrun %0d", cov_imp_clear);
    for (int p = 1; p <= 6; p++) $display("COV rst_p%0d %0d", p, cov_rst[p]);
    $display("COV bp_full %0d",      cov_bp_full);
    $display("COV bp_random %0d",    cov_bp_random);
    $display("COV bp_storm %0d",     cov_bp_storm);
    $display("COV gap_none %0d",     cov_gap_none);
    $display("COV gap_random %0d",   cov_gap_random);
    $display("COV gap_storm %0d",    cov_gap_storm);
    $display("COV ntile_64 %0d",     cov_n64);
    $display("COV ntile_4096 %0d",   cov_n4096);
  endfunction

  // ── frame_err pulse monitor (posedge = what the DUT commits) ──────────────
  always begin
    @(posedge clk);
    if (rst_n && frame_err) err_seen = err_seen + 1;
  end

  // ── decision consumer: LFSR backpressure + in-order scoreboard ────────────
  logic [31:0] lfsr;
  always @(negedge clk) begin
    if (!rst_n) begin
      lfsr    <= seed;
      d_ready <= 1'b0;
    end else begin
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      if (force_d0) d_ready <= 1'b0;
      else begin
        unique case (bp_mode)
          0:       d_ready <= 1'b1;
          1:       d_ready <= |lfsr[1:0];            // ~75% duty
          default: d_ready <= (lfsr[3:0] == 4'h0);   // storm ~6%
        endcase
      end
    end
  end

  logic [9:0] exp_beat;
  always begin
    @(posedge clk);
    if (rst_n && d_valid && d_ready) begin
      if (discard) begin
        discarded = discarded + 1;
      end else if (exp_q.size() == 0) begin
        errors = errors + 1;
        $display("FAIL cyc=%0d: ORPHAN decision beat fp16=%0b tier=%0d blk=%0d",
                 cyc, d_fp16, 2'(d_tier), d_blk);
      end else begin
        exp_beat = exp_q.pop_front();
        if ({d_blk, 2'(d_tier), d_fp16} !== exp_beat) begin
          errors = errors + 1;
          $display("FAIL cyc=%0d tile#%0d: got fp16=%0b tier=%0d blk=%0d, exp fp16=%0b tier=%0d blk=%0d",
                   cyc, dec_checked, d_fp16, 2'(d_tier), d_blk,
                   exp_beat[0], exp_beat[2:1], exp_beat[9:3]);
        end
        unique case (exp_beat[2:1])
          2'd0: cov_tier_cq8++;
          2'd1: cov_tier_cq4++;
          2'd2: cov_tier_cq4p++;
          default: ;
        endcase
        dec_checked = dec_checked + 1;
      end
    end
  end

  // ── score-stream driver primitives (negedge discipline) ───────────────────
  int unsigned rng_state;
  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction

  task automatic feed_gap();
    int unsigned g;
    unique case (gap_mode)
      0:       g = 0;
      1:       g = ((rnd() & 32'd3) == 0) ? (1 + rnd() % 32'd3) : 0;
      default: g = ((rnd() % 32'd8) == 0) ? (rnd() % 32'd24) : (rnd() % 32'd2);
    endcase
    if (g != 0) begin
      s_valid = 1'b0;
      s_last  = 1'b0;
      repeat (g) @(negedge clk);
    end
  endtask

  task automatic send_beat(input logic [31:0] data, input logic last,
                           input logic [6:0] blk);
    @(negedge clk);
    feed_gap();
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

  // ── waits / drains (all watchdog-bounded) ─────────────────────────────────
  task automatic wait_cond_fatal(input string what);
    // (helper body inlined at call sites; kept for message uniformity)
    $fatal(1, "TIMEOUT cyc=%0d: %s", cyc, what);
  endtask

  task automatic drain_pipe();
    int unsigned t;
    t = 0;
    idle_input();
    while (exp_q.size() != 0) begin
      @(negedge clk);
      t++;
      if (t > WAIT_LIMIT) wait_cond_fatal("drain: expected decisions pending");
    end
    t = 0;
    #1;
    while (busy) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) wait_cond_fatal("drain: busy never deasserted");
    end
    repeat (2) @(negedge clk);
  endtask

  // ── CSR importance readout checks ──────────────────────────────────────────
  logic [15:0] kmem [0:BLOCKS-1];

  function automatic logic [1:0] exp_tier_of(input logic [15:0] a);
    if      (a >= imp_thresh_hi) exp_tier_of = 2'(KVQ_CQ8);
    else if (a >= imp_thresh_lo) exp_tier_of = 2'(KVQ_CQ4P);
    else                         exp_tier_of = 2'(KVQ_CQ4);
  endfunction

  task automatic check_imp_one(input logic [6:0] addr,
                               input logic [15:0] exp_acc,
                               input string what);
    @(negedge clk);
    imp_rd_addr = addr;
    #1;
    if (imp_rd_data !== exp_acc) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d [%s]: importance[%0d] rd_data=%0d exp=%0d",
               cyc, what, addr, imp_rd_data, exp_acc);
    end
    if (2'(imp_rd_tier) !== exp_tier_of(exp_acc)) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d [%s]: importance[%0d] rd_tier=%0d exp=%0d (acc=%0d)",
               cyc, what, addr, 2'(imp_rd_tier), exp_tier_of(exp_acc), exp_acc);
    end
  endtask

  task automatic check_imp_all(input string what);
    for (int b = 0; b < int'(BLOCKS); b++)
      check_imp_one(7'(b), kmem[b], what);
  endtask

  task automatic check_imp_zero(input string what);
    for (int b = 0; b < int'(BLOCKS); b++)
      check_imp_one(7'(b), 16'd0, what);
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

  logic [31:0] tbuf [0:MAXBUF-1];

  task automatic read_scores(input int unsigned n);
    string tg;
    logic [31:0] v;
    int cnt;
    if (n > MAXBUF) $fatal(1, "tile len %0d exceeds TB buffer", n);
    for (int unsigned i = 0; i < n; i++) begin
      if (!next_line()) $fatal(1, "EOF inside S data (beat %0d)", i);
      cnt = $sscanf(cur_line, "%s %h", tg, v);
      if (cnt != 2 || tg != "S") $fatal(1, "bad S line: %s", cur_line);
      tbuf[i] = v;
    end
  endtask

  task automatic read_kmem();
    string tg;
    logic [15:0] v;
    int cnt;
    for (int b = 0; b < int'(BLOCKS); b++) begin
      if (!next_line()) $fatal(1, "EOF inside K block (line %0d)", b);
      cnt = $sscanf(cur_line, "%s %h", tg, v);
      if (cnt != 2 || tg != "K") $fatal(1, "bad K line: %s", cur_line);
      kmem[b] = v;
    end
  endtask

  // ── record handlers ────────────────────────────────────────────────────────
  int cur_thr;   // last threshold driven; change requires a drained pipe

  task automatic set_threshold(input int thr);
    if (thr != cur_thr) begin
      // the input skid may still hold the previous tile's beats; threshold is
      // sampled by the ENGINE on the s_last beat, so drain before changing
      drain_pipe();
      @(negedge clk);
      threshold = 5'(thr);
      cur_thr   = thr;
    end
  endtask

  task automatic do_tile(input string hdr);
    int thr, blk, len, fp16, tr;
    int unsigned acc, flags;
    int cnt;
    string tg;
    logic unused_hi;
    cnt = $sscanf(hdr, "%s %d %d %d %d %d %h %h",
                  tg, thr, blk, len, fp16, tr, acc, flags);
    if (cnt != 8 || tg != "T") $fatal(1, "bad T line: %s", hdr);
    unused_hi = &{1'b0, blk[31:7], fp16[31:1], tr[31:2], acc[31:16]};
    read_scores(len);
    set_threshold(thr);
    exp_q.push_back({7'(blk), 2'(tr), 1'(fp16)});
    for (int i = 0; i < len; i++)
      send_beat(tbuf[i], i == len - 1, 7'(blk));
    tiles_done++;

    // coverage from record fields
    cov_thr[thr & 31]++;
    if (fp16 != 0) cov_dec_fp16++; else cov_dec_int8++;
    if (len == 1) cov_len_1++;
    else if (len == int'(N_TILE)) cov_len_full++;
    else cov_len_short++;
    if ((flags & 'h001) != 0) cov_tie_exact++;
    if ((flags & 'h002) != 0) cov_tie_off1++;
    if ((flags & 'h004) != 0) cov_mag_zero++;
    if ((flags & 'h008) != 0) cov_mag_min++;
    if ((flags & 'h010) != 0) cov_mag_max++;
    if ((flags & 'h020) != 0) cov_imp_inc0++;
    if ((flags & 'h040) != 0) cov_imp_sat++;
    if ((flags & 'h080) != 0) cov_imp_eq_lo++;
    if ((flags & 'h100) != 0) cov_imp_eq_hi++;
    // exact-boundary / saturation tiles get an immediate CSR spot check
    if ((flags & 'h1C0) != 0) begin
      drain_pipe();
      check_imp_one(7'(blk), 16'(acc), "spot");
    end
  endtask

  task automatic do_frame(input string hdr);
    int thr, blk, len;
    int cnt, e0;
    int unsigned t;
    string tg;
    logic unused_hi;
    cnt = $sscanf(hdr, "%s %d %d %d", tg, thr, blk, len);
    if (cnt != 4 || tg != "F") $fatal(1, "bad F line: %s", hdr);
    unused_hi = &{1'b0, blk[31:7]};
    if (len <= int'(N_TILE)) $fatal(1, "F record len %0d not an overrun", len);
    read_scores(len);
    set_threshold(thr);
    e0 = err_seen;
    for (int i = 0; i < len; i++)
      send_beat(tbuf[i], i == len - 1, 7'(blk));
    idle_input();
    t = 0;
    while (err_seen == e0) begin
      @(negedge clk);
      t++;
      if (t > WAIT_LIMIT) wait_cond_fatal("frame_err pulse never seen");
    end
    exp_errs++;
    frames_done++;
    // exactly one pulse per overrun tile; drain the tile fully first
    drain_pipe();
    if (err_seen != e0 + 1) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: overrun tile produced %0d pulses (exp 1)",
               cyc, err_seen - e0);
    end
    #1;
    if (frame_err_sticky !== 1'b1) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: sticky not set after overrun tile", cyc);
    end
    @(negedge clk);
    frame_err_clear = 1'b1;
    @(negedge clk);
    frame_err_clear = 1'b0;
    repeat (2) @(negedge clk);
    #1;
    if (frame_err_sticky !== 1'b0) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: sticky not cleared by frame_err_clear", cyc);
    end
    cov_sticky_clear++;
    if (len == int'(N_TILE) + 1) cov_frame_onlast++; else cov_frame_abort++;
  endtask

  task automatic do_clear();
    read_kmem();
    drain_pipe();
    check_imp_all("pre-clear");
    @(negedge clk);
    imp_clear = 1'b1;
    @(negedge clk);
    imp_clear = 1'b0;
    repeat (2) @(negedge clk);
    check_imp_zero("post-clear");
    clears_done++;
    cov_imp_clear++;
  endtask

  // ── mid-operation reset (phase-targeted, hierarchical observation) ────────
  function automatic bit phase_cond(input int phase);
    unique case (phase)
      1: return dut.u_in_skid.m_valid;                       // beat in in-skid
      2: return dut.u_decide.cnt >= 2;                       // engine mid-tile
      3: return dut.u_decide.abort_tile;                     // abort drain
      4: return dut.dec_pend;                                // decision pending
      5: return dut.hold_valid && dut.u_out_skid.skid_valid; // hold occupied
      6: return dut.u_out_skid.skid_valid;                   // out-skid full
      default: return 1'b0;
    endcase
  endfunction

  task automatic sac_driver(input int phase);
    // sacrificial stimulus sized to reach the requested phase
    forever begin
      unique case (phase)
        3: begin
          // overrun tile with NO last: forces overrun -> abort_tile
          for (int i = 0; i < int'(N_TILE) + 8; i++)
            send_beat(32'(i + 1), 1'b0, 7'd0);
        end
        5, 6: send_beat(32'd1, 1'b1, 7'd0);   // 1-beat tiles fill decisions
        default: begin
          for (int i = 0; i < 8; i++)
            send_beat(32'(i + 1), i == 7, 7'd0);
        end
      endcase
    end
  endtask

  task automatic do_reset(input string hdr);
    int phase, cnt;
    int unsigned t;
    string tg;
    cnt = $sscanf(hdr, "%s %d", tg, phase);
    if (cnt != 2 || tg != "R" || phase < 1 || phase > 6)
      $fatal(1, "bad R line: %s", hdr);

    drain_pipe();
    discard = 1'b1;
    if (phase >= 5) force_d0 = 1'b1;

    fork
      sac_driver(phase);
      begin
        t = 0;
        @(negedge clk);
        #1;
        while (!phase_cond(phase)) begin
          @(negedge clk);
          #1;
          t++;
          if (t > WAIT_LIMIT)
            $fatal(1, "TIMEOUT cyc=%0d: reset phase %0d never reached",
                   cyc, phase);
        end
      end
    join_any
    disable fork;

    // phase condition is TRUE in the current cycle: assert reset before the
    // next posedge so the reset lands while the phase is active
    $display("RESET cyc=%0d: mid-op reset in phase %0d (in_skid=%0b cnt=%0d abort=%0b dec_pend=%0b hold=%0b oskid=%0b)",
             cyc, phase, dut.u_in_skid.m_valid, dut.u_decide.cnt,
             dut.u_decide.abort_tile, dut.dec_pend, dut.hold_valid,
             dut.u_out_skid.skid_valid);
    rst_n   = 1'b0;
    s_valid = 1'b0;
    s_last  = 1'b0;
    repeat (3) @(negedge clk);
    err_seen = 0;                 // phase-3 induction legitimately pulsed
    exp_errs = 0;
    rst_n    = 1'b1;
    force_d0 = 1'b0;
    repeat (3) @(negedge clk);
    discard  = 1'b0;

    // clean-state checks (mxe X-record discipline)
    #1;
    if (busy !== 1'b0) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: busy stuck after mid-op reset (phase %0d)",
               cyc, phase);
    end
    if (d_valid !== 1'b0) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: stray decision beat after mid-op reset", cyc);
    end
    if (s_ready !== 1'b1) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: s_ready not restored after mid-op reset", cyc);
    end
    if (frame_err_sticky !== 1'b0) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: sticky not cleared by reset", cyc);
    end
    check_imp_zero("post-reset");
    resets_done++;
    cov_rst[phase]++;
  endtask

  // ── end-of-run ─────────────────────────────────────────────────────────────
  task automatic do_end();
    read_kmem();
    drain_pipe();
    #1;
    if (busy !== 1'b0) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: busy high after final drain", cyc);
    end
    if (err_seen != exp_errs) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: frame_err pulses %0d != expected %0d",
               cyc, err_seen, exp_errs);
    end
    if (frame_err_sticky !== 1'b0) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: sticky set at end of run", cyc);
    end
    if (dec_checked != tiles_done) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: %0d decisions checked != %0d tiles driven",
               cyc, dec_checked, tiles_done);
    end
    check_imp_all("final");
    print_coverage();
    if (errors == 0) begin
      $display("TB PASS: %0d tiles, %0d overrun frames, %0d mid-op resets, %0d clears, %0d decisions checked (+%0d sacrificial discarded), 0 errors (N=%0d bp=%0d gap=%0d seed=%0d)",
               tiles_done, frames_done, resets_done, clears_done,
               dec_checked, discarded, N_TILE, bp_mode, gap_mode, seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s) across %0d tiles (N=%0d bp=%0d gap=%0d seed=%0d)",
             errors, tiles_done, N_TILE, bp_mode, gap_mode, seed);
    end
  endtask

  // ── main ───────────────────────────────────────────────────────────────────
  initial begin : main
    byte tag0;

    rst_n           = 1'b0;
    s_valid         = 1'b0;
    s_data          = '0;
    s_last          = 1'b0;
    s_blk           = '0;
    threshold       = 5'd1;
    imp_clear       = 1'b0;
    imp_rd_addr     = '0;
    frame_err_clear = 1'b0;
    discard         = 1'b0;
    force_d0        = 1'b0;
    errors          = 0;
    tiles_done      = 0;
    frames_done     = 0;
    resets_done     = 0;
    clears_done     = 0;
    dec_checked     = 0;
    discarded       = 0;
    err_seen        = 0;
    exp_errs        = 0;
    cur_thr         = -1;

    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("gap_mode=%d", gap_mode));
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("imp_hi=%d", hi_arg));
    void'($value$plusargs("imp_lo=%d", lo_arg));
    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state     = seed ^ 32'h71AD_BEEF;
    imp_thresh_hi = 16'(hi_arg);
    imp_thresh_lo = 16'(lo_arg);

    unique case (bp_mode)
      0: cov_bp_full++; 1: cov_bp_random++; default: cov_bp_storm++;
    endcase
    unique case (gap_mode)
      0: cov_gap_none++; 1: cov_gap_random++; default: cov_gap_storm++;
    endcase
    if (N_TILE == 64) cov_n64++;
    if (N_TILE == 4096) cov_n4096++;

    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);

    $display("[TB] N=%0d vectors=%s hi=%0d lo=%0d bp=%0d gap=%0d seed=%0d",
             N_TILE, vec_path, hi_arg, lo_arg, bp_mode, gap_mode, seed);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);
    #1;
    if (busy !== 1'b0 || frame_err_sticky !== 1'b0) begin
      errors = errors + 1;
      $display("FAIL cyc=%0d: busy/sticky asserted out of reset", cyc);
    end

    begin : records
      bit ended;
      ended = 1'b0;
      while (!ended && next_line()) begin
        tag0 = cur_line.getc(0);
        unique case (tag0)
          "T": do_tile(cur_line);
          "F": do_frame(cur_line);
          "C": do_clear();
          "R": do_reset(cur_line);
          "E": begin
            do_end();          // $finish (PASS) or $fatal (FAIL) scheduled
            ended = 1'b1;
          end
          default: $fatal(1, "unexpected record: %s", cur_line);
        endcase
      end
      if (!ended) $fatal(1, "vector file ended without an E record");
    end
  end

endmodule
