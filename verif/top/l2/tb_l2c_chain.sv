// tb_l2c_chain.sv — Layer-2 chain (c): the TIP tap alongside the ASU —
// REAL seam_score_dequant -> apex_score_fork -> {asu_softmax, tip_top},
// wired and parameterized exactly as rtl/top/apex_top.sv (TIP BLOCK_M=1,
// BLOCK_N=8, BLOCKS=128, ACC_W=16, T_MAX=31; ASU SCORE_FRAC=10).
//
// What this chain proves:
//   * the fork is lossless under INDEPENDENT backpressure storms on the two
//     consumers: the ASU probabilities AND the TIP decisions both stay
//     bit-exact vs golden while td_ready/pr_ready fight each other;
//   * TIP decisions (fp16 flag + post-update tier suggestion), importance
//     accumulator updates and the IMPORTANCE readout window vs the golden
//     mirror (tip_decide_golden / imp_update_golden / tier_golden);
//   * the tile-length frame guard (tile of 9 > BLOCK_M*BLOCK_N=8): TIP
//     aborts the tile, emits NO decision, raises the sticky — while the ASU
//     row of the same forked stream stays bit-exact (the documented L3
//     T>8 reality of the committed apex_top);
//   * mid-operation reset + full clean re-run.
//
// Command grammar:
//   DJ cols / CMP w32       dequant job + composite sideband
//   ACB l0..l7 last         one raw acc beat (8x32b lanes) into the dequant
//   BLK b                   set the s_blk level (rt_tip_blk role; only while
//                           quiescent, like the ROUTE contract)
//   THR t / IHI h / ILO l   TIP threshold / importance thresholds (levels)
//   ETIP blk tier fp16      expected TIP decision beat (full compare)
//   IMPRD addr data tier    importance window read check
//   EPR n <n {w16,last}>    expected ASU probabilities
//   STK mask exp            {tip_frame, asu_row, sd_frame, sd_range, sd_job}
//   IDLE / RST / ENDTAB / DONE     (+bp_mode/+stall_mode/+seed as l2a)

`include "apex_stream1_sva.svh"

module tb_l2c_chain;

  import apex_pkg::*;

  localparam int unsigned WAIT_LIMIT = 200000;

  logic clk, rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  int unsigned watchdog_cyc = 6_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: l2c did not finish within %0d cycles", watchdog_cyc);
  end

  initial begin
    if ($test$plusargs("dump")) begin
      $dumpfile("dump.fst");
      $dumpvars(0, tb_l2c_chain);
    end
  end

  int unsigned prng = 32'hC0FFEE03;
  function automatic int unsigned rnd();
    prng = prng ^ (prng << 13);
    prng = prng ^ (prng >> 17);
    prng = prng ^ (prng << 5);
    return prng;
  endfunction

  int unsigned bp_mode = 0;
  int unsigned stall_mode = 0;

  // ── dequant -> fork -> {ASU, TIP} ────────────────────────────────────────
  logic        dj_valid, dj_ready;
  logic [DIM_W-1:0] dj_cols;
  logic sd_job_error, sd_job_error_sticky, sd_busy, sd_done;
  logic        cmp_valid, cmp_ready;
  logic [31:0] cmp_data;
  logic        acc_valid, acc_ready;
  lane32_beat_t acc_beat;
  logic        sc_valid, sc_ready, sc_last;
  logic signed [31:0] sc_data;
  logic sd_range_error, sd_range_error_sticky;
  logic sd_frame_error, sd_frame_error_sticky;

  seam_score_dequant #(.N_MAX(64)) u_sd (
    .clk                (clk),
    .rst_n              (rst_n),
    .job_valid          (dj_valid),
    .job_ready          (dj_ready),
    .job_cols           (dj_cols),
    .job_error          (sd_job_error),
    .job_error_sticky   (sd_job_error_sticky),
    .busy               (sd_busy),
    .done               (sd_done),
    .cmp_valid          (cmp_valid),
    .cmp_ready          (cmp_ready),
    .cmp_data           (cmp_data),
    .acc_valid          (acc_valid),
    .acc_ready          (acc_ready),
    .acc_beat           (acc_beat),
    .score_valid        (sc_valid),
    .score_ready        (sc_ready),
    .score_data         (sc_data),
    .score_last         (sc_last),
    .range_error        (sd_range_error),
    .range_error_sticky (sd_range_error_sticky),
    .frame_error        (sd_frame_error),
    .frame_error_sticky (sd_frame_error_sticky)
  );

  logic fk_a_valid, fk_a_ready, fk_a_last;
  logic signed [31:0] fk_a_data;
  logic fk_b_valid, fk_b_ready, fk_b_last;
  logic signed [31:0] fk_b_data;
  logic fk_busy;

  apex_score_fork u_fork (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (sc_valid),
    .s_ready (sc_ready),
    .s_data  (sc_data),
    .s_last  (sc_last),
    .a_valid (fk_a_valid),
    .a_ready (fk_a_ready),
    .a_data  (fk_a_data),
    .a_last  (fk_a_last),
    .b_valid (fk_b_valid),
    .b_ready (fk_b_ready),
    .b_data  (fk_b_data),
    .b_last  (fk_b_last),
    .busy    (fk_busy)
  );

  logic        pr_valid, pr_ready, pr_last;
  logic [15:0] pr_prob;
  logic        asu_busy, asu_done, asu_row_error, asu_row_error_sticky;

  asu_softmax #(
    .SM_ROW_MAX (1024),
    .SCORE_FRAC (ASU_IN_FRAC)
  ) u_asu (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (fk_a_valid),
    .s_ready          (fk_a_ready),
    .s_score          (fk_a_data),
    .s_last           (fk_a_last),
    .m_valid          (pr_valid),
    .m_ready          (pr_ready),
    .m_prob           (pr_prob),
    .m_last           (pr_last),
    .busy             (asu_busy),
    .done             (asu_done),
    .row_error        (asu_row_error),
    .row_error_sticky (asu_row_error_sticky)
  );

  logic [6:0]  blk_r;
  logic        td_valid, td_ready, td_fp16;
  kvq_tier_e   td_tier;
  logic [6:0]  td_blk;
  logic [4:0]  thr_r;
  logic [15:0] ihi_r, ilo_r;
  logic [6:0]  imp_addr;
  logic [15:0] imp_data;
  kvq_tier_e   imp_tier;
  logic        tip_frame_err, tip_frame_err_sticky, tip_busy;

  tip_top #(
    .SCORE_WIDTH (32),
    .BLOCK_M     (1),
    .BLOCK_N     (8),
    .BLOCKS      (128),
    .ACC_W       (16),
    .T_MAX       (31)
  ) u_tip (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (fk_b_valid),
    .s_ready          (fk_b_ready),
    .s_data           (fk_b_data),
    .s_last           (fk_b_last),
    .s_blk            (blk_r),
    .d_valid          (td_valid),
    .d_ready          (td_ready),
    .d_fp16           (td_fp16),
    .d_tier           (td_tier),
    .d_blk            (td_blk),
    .threshold        (thr_r),
    .imp_thresh_hi    (ihi_r),
    .imp_thresh_lo    (ilo_r),
    .imp_clear        (1'b0),
    .imp_rd_addr      (imp_addr),
    .imp_rd_data      (imp_data),
    .imp_rd_tier      (imp_tier),
    .frame_err        (tip_frame_err),
    .frame_err_sticky (tip_frame_err_sticky),
    .frame_err_clear  (1'b0),
    .busy             (tip_busy)
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, sd_done, asu_done, sd_job_error, sd_range_error,
                       sd_frame_error, asu_row_error, tip_frame_err};

  // ── SVA (§5 on every new-glue stream) ────────────────────────────────────
  bind seam_score_dequant apex_stream1_sva #(.WIDTH(33), .NAME("sd.score"))
    u_sva_sc (.clk(clk), .rst_n(rst_n), .valid(score_valid),
              .ready(score_ready), .data({unsigned'(score_data), score_last}));
  bind apex_score_fork apex_stream1_sva #(.WIDTH(33), .NAME("fork.a"))
    u_sva_a (.clk(clk), .rst_n(rst_n), .valid(a_valid), .ready(a_ready),
             .data({unsigned'(a_data), a_last}));
  bind apex_score_fork apex_stream1_sva #(.WIDTH(33), .NAME("fork.b"))
    u_sva_b (.clk(clk), .rst_n(rst_n), .valid(b_valid), .ready(b_ready),
             .data({unsigned'(b_data), b_last}));
  bind asu_softmax apex_stream1_sva #(.WIDTH(17), .NAME("asu.m"))
    u_sva_pr (.clk(clk), .rst_n(rst_n), .valid(m_valid), .ready(m_ready),
              .data({m_prob, m_last}));
  bind tip_top apex_stream1_sva #(.WIDTH(10), .NAME("tip.d"))
    u_sva_td (.clk(clk), .rst_n(rst_n), .valid(d_valid), .ready(d_ready),
              .data({d_blk, 2'(d_tier), d_fp16}));

  // ── independent backpressure on BOTH consumers ────────────────────────────
  int unsigned bp_hold, bp_hold2;
  always @(posedge clk) begin
    if (!rst_n) begin
      pr_ready <= 1'b1;
      td_ready <= 1'b1;
      bp_hold  <= 0;
      bp_hold2 <= 0;
    end else begin
      unique case (bp_mode)
        0: begin
          pr_ready <= 1'b1;
          td_ready <= 1'b1;
        end
        1: begin
          pr_ready <= rnd()[0];
          td_ready <= rnd()[1];
        end
        default: begin
          if (bp_hold != 0) begin
            bp_hold  <= bp_hold - 1;
            pr_ready <= 1'b0;
          end else begin
            pr_ready <= 1'b1;
            if (rnd() % 12 == 0) bp_hold <= 20 + (rnd() % 180);
          end
          if (bp_hold2 != 0) begin
            bp_hold2 <= bp_hold2 - 1;
            td_ready <= 1'b0;
          end else begin
            td_ready <= 1'b1;
            if (rnd() % 10 == 0) bp_hold2 <= 20 + (rnd() % 150);
          end
        end
      endcase
    end
  end

  task automatic stall_gap();
    int unsigned g;
    unique case (stall_mode)
      0: g = 0;
      1: g = rnd() % 4;
      default: g = (rnd() % 10 == 0) ? (20 + rnd() % 120) : (rnd() % 3);
    endcase
    repeat (g) @(negedge clk);
  endtask

  // ── scoreboard ─────────────────────────────────────────────────────────────
  int n_checks, n_errors;
  int tap_checks, tap_errors;

  logic [16:0] epr_exp [4096];
  int epr_n, epr_i;
  logic [9:0] tip_q [$];

  always @(posedge clk) begin
    if (rst_n && pr_valid && pr_ready) begin
      if (epr_i >= epr_n) begin
        tap_errors <= tap_errors + 1;
        $error("[epr] unexpected prob %04x @%0d", pr_prob, cyc);
      end else begin
        tap_checks <= tap_checks + 1;
        if ({pr_prob, pr_last} !== epr_exp[epr_i]) begin
          tap_errors <= tap_errors + 1;
          $error("[epr] idx %0d: got %04x/%0b exp %04x/%0b", epr_i,
                 pr_prob, pr_last, epr_exp[epr_i][16:1], epr_exp[epr_i][0]);
        end
      end
      epr_i <= epr_i + 1;
    end
    if (rst_n && td_valid && td_ready)
      tip_q.push_back({td_blk, 2'(td_tier), td_fp16});
  end

  // ── drivers ────────────────────────────────────────────────────────────────
  task automatic drive_dj(input logic [DIM_W-1:0] cols);
    int wd = 0;
    @(negedge clk);
    dj_cols  = cols;
    dj_valid = 1'b1;
    while (!dj_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_dj stall @%0d", cyc);
    end
    @(negedge clk);
    dj_valid = 1'b0;
  endtask

  task automatic drive_cmp(input logic [31:0] w);
    int wd = 0;
    stall_gap();
    @(negedge clk);
    cmp_data  = w;
    cmp_valid = 1'b1;
    while (!cmp_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_cmp stall @%0d", cyc);
    end
    @(negedge clk);
    cmp_valid = 1'b0;
  endtask

  task automatic drive_acb(input logic [255:0] lanes, input logic lst);
    int wd = 0;
    stall_gap();
    @(negedge clk);
    acc_beat.data = lanes;
    acc_beat.last = lst;
    acc_valid = 1'b1;
    while (!acc_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_acb stall @%0d", cyc);
    end
    @(negedge clk);
    acc_valid = 1'b0;
  endtask

  task automatic expect_tip(input logic [6:0] blk, input logic [1:0] tier,
                            input logic fp16);
    int wd = 0;
    logic [9:0] got;
    while (tip_q.size() == 0) begin
      @(posedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "ETIP timeout @%0d", cyc);
    end
    got = tip_q.pop_front();
    n_checks++;
    if (got !== {blk, tier, fp16}) begin
      n_errors++;
      $error("[ETIP] got blk=%0d tier=%0d fp16=%0b exp blk=%0d tier=%0d fp16=%0b",
             got[9:3], got[2:1], got[0], blk, tier, fp16);
    end
  endtask

  task automatic wait_idle();
    int wd = 0;
    @(posedge clk);
    while (sd_busy || asu_busy || tip_busy || fk_busy || sc_valid
           || pr_valid || td_valid) begin
      @(posedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "IDLE timeout @%0d", cyc);
    end
  endtask

  function automatic void chk32(input string what, input logic [31:0] got,
                                input logic [31:0] exp);
    n_checks++;
    if (got !== exp) begin
      n_errors++;
      $error("[%s] got %08x exp %08x", what, got, exp);
    end
  endfunction

  // ── interpreter ────────────────────────────────────────────────────────────
  initial begin : interp
    string vec_path, cmd, rest;
    bit done_seen;
    int fd, i;
    logic [31:0] m, e, d;
    logic [255:0] lanes;
    logic [4:0]  stk;

    done_seen = 1'b0;
    rst_n = 1'b0;
    dj_valid = 1'b0; dj_cols = '0;
    cmp_valid = 1'b0; cmp_data = '0;
    acc_valid = 1'b0; acc_beat = '{default: '0};
    blk_r = '0; thr_r = 5'd1; ihi_r = 16'hFFFF; ilo_r = '0;
    imp_addr = '0;
    n_checks = 0; n_errors = 0;
    epr_n = 0; epr_i = 0;

    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
    void'($value$plusargs("seed=%d", prng));
    if (prng == 0) prng = 32'hC0FFEE03;

    vec_path = "build/vectors_l2c_directed.txt";
    void'($value$plusargs("vectors=%s", vec_path));
    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (3) @(negedge clk);

    while ($fscanf(fd, "%s", cmd) == 1) begin
      if (cmd.len() >= 2 && cmd.substr(0, 1) == "//") begin
        void'($fgets(rest, fd));
        $display("[%0d] %s %s", cyc, cmd, rest);
        continue;
      end
      case (cmd)
        "DJ": begin
          void'($fscanf(fd, "%h", d));
          drive_dj(d[DIM_W-1:0]);
        end
        "CMP": begin
          void'($fscanf(fd, "%h", d));
          drive_cmp(d);
        end
        "ACB": begin
          for (i = 0; i < 8; i++) begin
            void'($fscanf(fd, "%h", d));
            lanes[32*i +: 32] = d;
          end
          void'($fscanf(fd, "%h", d));
          drive_acb(lanes, d[0]);
        end
        "BLK": begin
          void'($fscanf(fd, "%h", d));
          @(negedge clk);
          blk_r = d[6:0];
          @(negedge clk);
        end
        "THR": begin
          void'($fscanf(fd, "%h", d));
          @(negedge clk);
          thr_r = d[4:0];
          @(negedge clk);
        end
        "IHI": begin
          void'($fscanf(fd, "%h", d));
          @(negedge clk);
          ihi_r = d[15:0];
          @(negedge clk);
        end
        "ILO": begin
          void'($fscanf(fd, "%h", d));
          @(negedge clk);
          ilo_r = d[15:0];
          @(negedge clk);
        end
        "ETIP": begin
          void'($fscanf(fd, "%h %h %h", d, m, e));
          expect_tip(d[6:0], m[1:0], e[0]);
        end
        "IMPRD": begin
          void'($fscanf(fd, "%h %h %h", d, m, e));
          @(negedge clk);
          imp_addr = d[6:0];
          @(negedge clk);
          chk32("IMPRD data", 32'(imp_data), m);
          chk32("IMPRD tier", 32'(2'(imp_tier)), e);
        end
        "EPR": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h %h", m, e));
            epr_exp[epr_n + i] = {m[15:0], e[0]};
          end
          epr_n += int'(d);
        end
        "STK": begin
          void'($fscanf(fd, "%h %h", m, e));
          stk = {tip_frame_err_sticky, asu_row_error_sticky,
                 sd_frame_error_sticky, sd_range_error_sticky,
                 sd_job_error_sticky};
          chk32("STK", 32'(stk) & m, e);
        end
        "IDLE": wait_idle();
        "RST": begin
          @(negedge clk);
          dj_valid = 1'b0; cmp_valid = 1'b0; acc_valid = 1'b0;
          rst_n = 1'b0;
          repeat (4) @(negedge clk);
          rst_n = 1'b1;
          repeat (3) @(negedge clk);
          chk32("RST epr drain", 32'(epr_i), 32'(epr_n));
          chk32("RST tip drain", 32'(tip_q.size()), 0);
          tip_q.delete();
          $display("[%0d] mid-operation reset applied", cyc);
        end
        "ENDTAB": begin
          chk32("ENDTAB epr", 32'(epr_i), 32'(epr_n));
          chk32("ENDTAB tip", 32'(tip_q.size()), 0);
        end
        "DONE": begin
          wait_idle();
          n_checks += tap_checks;
          n_errors += tap_errors;
          $display("L2C RESULT: cycles=%0d checks=%0d errors=%0d",
                   cyc, n_checks, n_errors);
          if (n_errors != 0) $fatal(1, "L2C FAIL: %0d errors", n_errors);
          $display("L2C PASS");
          done_seen = 1'b1;
          $finish;
        end
        default: $fatal(1, "unknown command '%s'", cmd);
      endcase
      if (done_seen) break;
    end
    if (!done_seen) $fatal(1, "vector file ended without DONE");
  end

endmodule
