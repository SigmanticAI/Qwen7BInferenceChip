// tb_l2a_chain.sv — Layer-2 chain (a): REAL mxe_top -> seam_score_dequant ->
// asu_softmax, wired exactly as rtl/top/apex_top.sv wires them (MXE res beats
// straight into the dequant acc port, dequant score stream straight into the
// ASU), driven by a golden-generated command script (gen_l2a_vectors.py, the
// arbiter being golden/apex_golden: gemm_i8 -> score-dequant RNE mirror ->
// online_softmax_fx).
//
// What this chain proves (integration, not re-proving the L1 blocks):
//   * MXE res framing (last on the job's final drain beat) against the
//     dequant input framing contract — both the clean single-job case AND
//     the L3 v0.1 reality of a T>8 dequant job spanning several M=1 MXE
//     jobs (each mid-stream last=1 raises the DOCUMENTED frame_error sticky
//     while the numerics stay bit-exact — checked both ways here);
//   * INT32 accumulator lanes -> serial scores -> Q1.15 probabilities are
//     bit-exact vs the golden chain end to end under backpressure storms
//     and input stall storms;
//   * mid-operation reset: rst_n dropped with a job in flight in every
//     block, then the full expected set replayed clean;
//   * §5/D-006 SVA compiled into the build (apex_stream_sva on the MXE,
//     apex_stream1_sva on the dequant score and ASU prob streams).
//
// Command grammar (one whitespace-separated command per action):
//   DESC w3 w2 w1 w0     descriptor -> mxe desc port
//   AB w64 l             one activation beat (data, last)
//   WB w64               one weight beat
//   DJ cols              score-dequant job
//   CMP w32              one composite word into the dequant sideband
//   ESC n <n w32>        append n expected scores ({data} in accept order)
//   EPR n <n w16>        append n expected probabilities
//   IDLE                 wait for the whole chain to drain (busy==0)
//   RST                  mid-operation reset: pulse rst_n, flush scoreboards
//   STK mask exp         check {asu_row, sd_frame, sd_range, sd_job, mxe_desc}
//   ENDTAB               assert every table entry was consumed
//   DONE                 final bookkeeping + verdict
//
// Watchdogged; honest fail on any hang. Backpressure/stall regimes are
// plusargs so the same vector file runs under several protocol climates:
//   +bp_mode=0/1/2   ASU prob-stream ready: always / ~50% random / storm
//   +stall_mode=0/1/2 driver gaps: none / small random / burst storms
//   +seed=N          TB-local PRNG seed (storms only — data is scripted)

`include "apex_stream_sva.svh"
`include "apex_stream1_sva.svh"

module tb_l2a_chain;

  import apex_pkg::*;

  localparam int unsigned WAIT_LIMIT = 200000;

  // ── clock / reset / watchdog ──────────────────────────────────────────────
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
    $fatal(1, "WATCHDOG: l2a did not finish within %0d cycles", watchdog_cyc);
  end

  initial begin
    if ($test$plusargs("dump")) begin
      $dumpfile("dump.fst");
      $dumpvars(0, tb_l2a_chain);
    end
  end

  // ── TB PRNG (xorshift32; storms only, never data) ─────────────────────────
  int unsigned prng = 32'hC0FFEE01;
  function automatic int unsigned rnd();
    prng = prng ^ (prng << 13);
    prng = prng ^ (prng >> 17);
    prng = prng ^ (prng << 5);
    return prng;
  endfunction

  int unsigned bp_mode = 0;
  int unsigned stall_mode = 0;

  // ── DUT chain: mxe_top -> seam_score_dequant -> asu_softmax ──────────────
  logic         desc_valid, desc_ready;
  logic [127:0] desc_r;
  logic         mxe_desc_error, mxe_desc_error_sticky, mxe_busy, mxe_done;
  logic         act_valid, act_ready;
  logic [63:0]  act_data;
  logic         act_last;
  logic         wgt_valid, wgt_ready;
  logic [63:0]  wgt_data;
  logic         res_valid, res_ready;
  lane32_beat_t res_beat;

  mxe_top u_mxe (
    .clk               (clk),
    .rst_n             (rst_n),
    .desc_valid        (desc_valid),
    .desc_ready        (desc_ready),
    .desc              (mxe_desc_t'(desc_r)),
    .desc_error        (mxe_desc_error),
    .desc_error_sticky (mxe_desc_error_sticky),
    .busy              (mxe_busy),
    .done              (mxe_done),
    .act_valid         (act_valid),
    .act_ready         (act_ready),
    .act_beat          (lane8_beat_t'({act_data, act_last})),
    .wgt_valid         (wgt_valid),
    .wgt_ready         (wgt_ready),
    .wgt_beat          (lane8_beat_t'({wgt_data, 1'b0})),
    .res_valid         (res_valid),
    .res_ready         (res_ready),
    .res_beat          (res_beat)
  );

  logic        dj_valid, dj_ready;
  logic [DIM_W-1:0] dj_cols;
  logic        sd_job_error, sd_job_error_sticky, sd_busy, sd_done;
  logic        cmp_valid, cmp_ready;
  logic [31:0] cmp_data;
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
    .acc_valid          (res_valid),
    .acc_ready          (res_ready),
    .acc_beat           (res_beat),
    .score_valid        (sc_valid),
    .score_ready        (sc_ready),
    .score_data         (sc_data),
    .score_last         (sc_last),
    .range_error        (sd_range_error),
    .range_error_sticky (sd_range_error_sticky),
    .frame_error        (sd_frame_error),
    .frame_error_sticky (sd_frame_error_sticky)
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
    .s_valid          (sc_valid),
    .s_ready          (sc_ready),
    .s_score          (sc_data),
    .s_last           (sc_last),
    .m_valid          (pr_valid),
    .m_ready          (pr_ready),
    .m_prob           (pr_prob),
    .m_last           (pr_last),
    .busy             (asu_busy),
    .done             (asu_done),
    .row_error        (asu_row_error),
    .row_error_sticky (asu_row_error_sticky)
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, mxe_done, sd_done, asu_done, mxe_desc_error,
                       sd_job_error, sd_range_error, sd_frame_error,
                       asu_row_error};

  // ── SVA (§5/D-006, compiled — D-012 gate) ─────────────────────────────────
  bind mxe_top apex_stream_sva u_sva_mxe (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(desc_valid), .desc_ready(desc_ready), .desc(desc),
    .desc_error(desc_error), .desc_error_sticky(desc_error_sticky),
    .busy(busy), .done(done),
    .act_valid(act_valid), .act_ready(act_ready), .act_beat(act_beat),
    .wgt_valid(wgt_valid), .wgt_ready(wgt_ready), .wgt_beat(wgt_beat),
    .res_valid(res_valid), .res_ready(res_ready), .res_beat(res_beat));

  bind seam_score_dequant apex_stream1_sva #(.WIDTH(33), .NAME("sd.score"))
    u_sva_sc (.clk(clk), .rst_n(rst_n), .valid(score_valid),
              .ready(score_ready), .data({unsigned'(score_data), score_last}));
  bind asu_softmax apex_stream1_sva #(.WIDTH(17), .NAME("asu.m"))
    u_sva_pr (.clk(clk), .rst_n(rst_n), .valid(m_valid), .ready(m_ready),
              .data({m_prob, m_last}));

  // ── backpressure generator on the ASU output ─────────────────────────────
  int unsigned bp_hold;
  always @(posedge clk) begin
    if (!rst_n) begin
      pr_ready <= 1'b1;
      bp_hold  <= 0;
    end else begin
      unique case (bp_mode)
        0: pr_ready <= 1'b1;
        1: pr_ready <= rnd()[0];
        default: begin                       // storm: long dead windows
          if (bp_hold != 0) begin
            bp_hold  <= bp_hold - 1;
            pr_ready <= 1'b0;
          end else begin
            pr_ready <= 1'b1;
            if (rnd() % 12 == 0) bp_hold <= 20 + (rnd() % 180);
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

  // ── scoreboard: score tap + prob collector vs expectation tables ─────────
  int n_checks, n_errors;
  int tap_checks, tap_errors;

  logic [31:0] esc_exp [4096];
  logic [16:0] epr_exp [4096];
  int esc_n, esc_i, epr_n, epr_i;

  always @(posedge clk) begin
    if (rst_n && sc_valid && sc_ready) begin
      if (esc_i >= esc_n) begin
        tap_errors <= tap_errors + 1;
        $error("[esc] unexpected score %08x @%0d", sc_data, cyc);
      end else begin
        tap_checks <= tap_checks + 1;
        if (unsigned'(sc_data) !== esc_exp[esc_i]) begin
          tap_errors <= tap_errors + 1;
          $error("[esc] idx %0d: got %08x exp %08x", esc_i,
                 unsigned'(sc_data), esc_exp[esc_i]);
        end
      end
      esc_i <= esc_i + 1;
    end
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
  end

  // ── drivers (negedge discipline, watchdogged waits) ───────────────────────
  task automatic drive_desc(input logic [127:0] d);
    int wd = 0;
    @(negedge clk);
    desc_r     = d;
    desc_valid = 1'b1;
    while (!desc_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_desc stall @%0d", cyc);
    end
    @(negedge clk);
    desc_valid = 1'b0;
  endtask

  task automatic drive_ab(input logic [63:0] w, input logic lst);
    int wd = 0;
    stall_gap();
    @(negedge clk);
    act_data  = w;
    act_last  = lst;
    act_valid = 1'b1;
    while (!act_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_ab stall @%0d", cyc);
    end
    @(negedge clk);
    act_valid = 1'b0;
  endtask

  task automatic drive_wb(input logic [63:0] w);
    int wd = 0;
    stall_gap();
    @(negedge clk);
    wgt_data  = w;
    wgt_valid = 1'b1;
    while (!wgt_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_wb stall @%0d", cyc);
    end
    @(negedge clk);
    wgt_valid = 1'b0;
  endtask

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

  task automatic wait_idle();
    int wd = 0;
    @(posedge clk);
    while (mxe_busy || sd_busy || asu_busy || sc_valid || pr_valid) begin
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
    logic [63:0] w64;
    logic [31:0] dw [4];
    logic [4:0]  stk;

    done_seen = 1'b0;
    rst_n = 1'b0;
    desc_valid = 1'b0; desc_r = '0;
    act_valid = 1'b0; act_data = '0; act_last = 1'b0;
    wgt_valid = 1'b0; wgt_data = '0;
    dj_valid = 1'b0; dj_cols = '0;
    cmp_valid = 1'b0; cmp_data = '0;
    n_checks = 0; n_errors = 0;
    esc_n = 0; esc_i = 0; epr_n = 0; epr_i = 0;

    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
    void'($value$plusargs("seed=%d", prng));
    if (prng == 0) prng = 32'hC0FFEE01;

    vec_path = "build/vectors_l2a_directed.txt";
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
        "DESC": begin
          void'($fscanf(fd, "%h %h %h %h", dw[3], dw[2], dw[1], dw[0]));
          drive_desc({dw[3], dw[2], dw[1], dw[0]});
        end
        "AB": begin
          void'($fscanf(fd, "%h %h", w64, d));
          drive_ab(w64, d[0]);
        end
        "WB": begin
          void'($fscanf(fd, "%h", w64));
          drive_wb(w64);
        end
        "DJ": begin
          void'($fscanf(fd, "%h", d));
          drive_dj(d[DIM_W-1:0]);
        end
        "CMP": begin
          void'($fscanf(fd, "%h", d));
          drive_cmp(d);
        end
        "ESC": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h", m));
            esc_exp[esc_n + i] = m;
          end
          esc_n += int'(d);
        end
        "EPR": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h %h", m, e));
            epr_exp[epr_n + i] = {m[15:0], e[0]};
          end
          epr_n += int'(d);
        end
        "IDLE": wait_idle();
        "RST": begin
          // mid-operation reset: deassert everything, drop rst_n, flush
          // the scoreboard state (pre-reset partial work is sacrificial
          // by construction — the generator guarantees no checked output
          // was in flight)
          @(negedge clk);
          desc_valid = 1'b0; act_valid = 1'b0; wgt_valid = 1'b0;
          dj_valid = 1'b0; cmp_valid = 1'b0;
          rst_n = 1'b0;
          repeat (4) @(negedge clk);
          rst_n = 1'b1;
          repeat (3) @(negedge clk);
          chk32("RST tap drain sc", 32'(esc_i), 32'(esc_n));
          chk32("RST tap drain pr", 32'(epr_i), 32'(epr_n));
          $display("[%0d] mid-operation reset applied", cyc);
        end
        "STK": begin
          void'($fscanf(fd, "%h %h", m, e));
          stk = {asu_row_error_sticky, sd_frame_error_sticky,
                 sd_range_error_sticky, sd_job_error_sticky,
                 mxe_desc_error_sticky};
          chk32("STK", 32'(stk) & m, e);
        end
        "ENDTAB": begin
          chk32("ENDTAB esc", 32'(esc_i), 32'(esc_n));
          chk32("ENDTAB epr", 32'(epr_i), 32'(epr_n));
        end
        "DONE": begin
          wait_idle();
          n_checks += tap_checks;
          n_errors += tap_errors;
          $display("L2A RESULT: cycles=%0d checks=%0d errors=%0d",
                   cyc, n_checks, n_errors);
          if (n_errors != 0) $fatal(1, "L2A FAIL: %0d errors", n_errors);
          $display("L2A PASS");
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
