// tb_seq_attack.sv — INDEPENDENT adversarial audit TB for rtl/seq/seq_walker.sv.
// Audit collateral (verif/audit_seq): reuses the deliverable's mxe_stub and
// SVA packs unchanged; adds attack scenarios their suite does not cover:
//
//   AT-1  soft abort (CTRL.soft_reset) fired in EVERY walker state:
//         IDLE-empty, IDLE-queued (enable=0), S_REQ presented-and-stalled
//         (force_stall), S_WAIT mid-beats (long job), and during a draining
//         abort (double abort).
//   AT-2  abort-window discard contract: EVERY descriptor whose ds_* handshake
//         completes while (abort_req || aborting) must NEVER be dispatched
//         (seq_walker.sv header: "accepted and DISCARDED"), and every
//         descriptor queued-but-not-presented at abort time must be discarded.
//         Checked with a serial-number watermark in the rq_scale field
//         (legality ignores it).
//   AT-3  descriptor flood: continuous ds_valid feeder + randomized abort
//         storm + random stub latency — D-006 / §5 SVAs bound and armed.
//   AT-4  queue capacity race: with dispatch gated, exactly QDEPTH+2 beats
//         (FIFO + 2-deep skid) are accepted before ds_ready stalls; q_full
//         and q_count checked; then in-order drain of all of them.
//   AT-5  busy honesty for a beat held ONLY in the input skid (count==0,
//         FSM==IDLE): busy must already be high (§5: "output beats pending
//         anywhere ... skids included" — mirrored for the input side by the
//         header's busy definition).
//
// Any FAIL prints a precise cycle trace. TB discipline: negedge stimulus,
// #1-after-posedge sampling, $fatal watchdog (house rules).

`include "apex_stream1_sva.svh"
`include "seq_sb_sva.svh"

module tb_seq_attack;

  import apex_pkg::*;

  localparam int unsigned QDEPTH     = 8;
  localparam int unsigned DW         = 128;
  localparam int unsigned WAIT_LIMIT = 200_000;

  // ── clock / watchdog ───────────────────────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    repeat (20_000_000) @(posedge clk);
    $fatal(1, "WATCHDOG: simulation did not finish (cyc=%0d)", cyc);
  end

  // ── plusargs ───────────────────────────────────────────────────────────────
  int unsigned seed     = 32'hA0D1_75E9;
  int unsigned lat_mode = 1;
  int unsigned n_aborts = 400;

  // ── DUT + stub ─────────────────────────────────────────────────────────────
  logic          enable, abort_req;
  logic          ds_valid, ds_ready;
  logic [DW-1:0] ds_bits;
  mxe_desc_t     ds_desc;
  logic          md_valid, md_ready;
  mxe_desc_t     md_desc;
  logic          stub_done, stub_derr, stub_busy_w;
  logic          seq_derr, q_empty, q_full, busy, aborting;
  logic [3:0]    q_count;

  assign ds_desc = mxe_desc_t'(ds_bits);

  seq_walker #(.QDEPTH(QDEPTH)) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .enable         (enable),
    .abort_req      (abort_req),
    .ds_valid       (ds_valid),
    .ds_ready       (ds_ready),
    .ds_desc        (ds_desc),
    .md_valid       (md_valid),
    .md_ready       (md_ready),
    .md_desc        (md_desc),
    .mxe_done       (stub_done),
    .mxe_desc_error (stub_derr),
    .desc_error     (seq_derr),
    .q_empty        (q_empty),
    .q_full         (q_full),
    .q_count        (q_count),
    .busy           (busy),
    .aborting       (aborting)
  );

  logic         force_stall;
  logic         res_valid, res_ready;
  lane32_beat_t res_beat;
  logic         acc_pulse, acc_legal;

  mxe_stub u_stub (
    .clk         (clk),
    .rst_n       (rst_n),
    .lat_mode    (2'(lat_mode)),
    .seed        (seed),
    .force_stall (force_stall),
    .desc_valid  (md_valid),
    .desc_ready  (md_ready),
    .desc        (md_desc),
    .desc_error  (stub_derr),
    .busy        (stub_busy_w),
    .done        (stub_done),
    .res_valid   (res_valid),
    .res_ready   (res_ready),
    .res_beat    (res_beat),
    .acc_pulse   (acc_pulse),
    .acc_legal   (acc_legal)
  );

  // free-running result consumer with mild backpressure
  logic [31:0] blfsr;
  always @(negedge clk) begin
    if (!rst_n) begin
      blfsr     <= (seed ^ 32'h0B50_1EAF) | 32'h1;
      res_ready <= 1'b0;
    end else begin
      blfsr <= {blfsr[30:0], blfsr[31] ^ blfsr[21] ^ blfsr[1] ^ blfsr[0]};
      res_ready <= |blfsr[1:0];
    end
  end

  // ── the deliverable's own SVA packs, reused and armed ─────────────────────
  bind seq_walker apex_stream1_sva #(.WIDTH(128), .NAME("aud.ds"))
    u_sva_ds (.clk(clk), .rst_n(rst_n), .valid(ds_valid), .ready(ds_ready),
              .data(128'(ds_desc)));

  bind seq_walker apex_stream1_sva #(.WIDTH(128), .NAME("aud.md"))
    u_sva_md (.clk(clk), .rst_n(rst_n), .valid(md_valid), .ready(md_ready),
              .data(128'(md_desc)));

  bind seq_walker seq_sb_sva #(.QDEPTH(QDEPTH)) u_sva_seq (
    .clk            (clk),
    .rst_n          (rst_n),
    .enable         (enable),
    .md_valid       (md_valid),
    .md_ready       (md_ready),
    .mxe_done       (mxe_done),
    .mxe_desc_error (mxe_desc_error),
    .busy           (busy),
    .aborting       (aborting),
    .q_empty        (q_empty),
    .q_full         (q_full),
    .q_count        (q_count)
  );

  // ── serial-watermark scoreboard (AT-2 leak check + order check) ───────────
  int errors;
  int exp_q [$];                    // serials expected to dispatch, in order
  bit must_discard [int];           // serial -> accepted-in-abort-window
  int accept_cyc  [int];            // serial -> ds-handshake cycle
  int dispatched, done_cnt, leak_cnt;
  logic abort_req_q;

  function automatic logic [DW-1:0] mk_desc(input int serial, input int m);
    mxe_desc_t d;
    d = '0;
    d.opcode   = 8'(OP_GEMM_WS);
    d.m_dim    = 12'(m);
    d.k_dim    = 12'd16;
    d.n_dim    = 12'd4;
    d.rq_scale = 16'(serial);       // watermark (legality ignores this field)
    return DW'(d);
  endfunction

  always @(posedge clk) begin
    if (rst_n) begin
      // 1) dispatch (stub accept) — check against scoreboard
      if (acc_pulse) begin
        int s;
        s = int'(md_desc.rq_scale);
        dispatched++;
        if (must_discard.exists(s)) begin
          errors++;
          leak_cnt++;
          $display("FAIL cyc=%0d: ABORT-WINDOW LEAK — serial %0d accepted on ds at cyc=%0d while abort_req/aborting=1 was DISPATCHED to MXE",
                   cyc, s, accept_cyc[s]);
        end else if (exp_q.size() == 0) begin
          errors++;
          $display("FAIL cyc=%0d: ORPHAN dispatch serial=%0d (discarded-at-abort descriptor executed)",
                   cyc, s);
        end else if (exp_q[0] != s) begin
          errors++;
          $display("FAIL cyc=%0d: ORDER dispatch serial=%0d expected %0d",
                   cyc, s, exp_q[0]);
          void'(exp_q.pop_front());
        end else begin
          void'(exp_q.pop_front());
        end
      end
      if (stub_done) done_cnt++;

      // 2) abort marking: queued-but-not-presented serials must be discarded
      if (abort_req && !abort_req_q) begin
        int keep;
        keep = (md_valid && !acc_pulse) ? 1 : 0;   // presented one may finish
        while (exp_q.size() > keep) begin
          int s;
          s = exp_q.pop_back();
          must_discard[s] = 1'b1;
          if (!accept_cyc.exists(s)) accept_cyc[s] = -1;
        end
      end

      // 3) ds handshake bookkeeping
      if (ds_valid && ds_ready) begin
        int s;
        s = int'(ds_desc.rq_scale);
        accept_cyc[s] = int'(cyc);
        if (abort_req || aborting) must_discard[s] = 1'b1;   // contract: discard
        else                       exp_q.push_back(s);
      end
    end
    abort_req_q <= abort_req && rst_n;
  end

  // ── feeder ─────────────────────────────────────────────────────────────────
  int serial;
  bit feed_on;
  int unsigned rng_state;

  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction

  task automatic send_one(input int m);
    int unsigned t;
    @(negedge clk);
    serial++;
    ds_bits  = mk_desc(serial, m);
    ds_valid = 1'b1;
    #1;
    t = 0;
    while (!ds_ready) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: ds_ready never rose", cyc);
    end
    @(negedge clk);
    ds_valid = 1'b0;
  endtask

  // continuous-feed process (AT-3): max-rate descriptor flood while enabled
  initial begin : feeder
    wait (rst_n === 1'b1);
    forever begin
      @(negedge clk);
      if (feed_on) begin
        int unsigned t;
        serial++;
        ds_bits  = mk_desc(serial, 1 + int'(rnd() % 8));
        ds_valid = 1'b1;
        #1;
        t = 0;
        // complete the handshake even if feed_on drops (§5: no retraction)
        while (!ds_ready) begin
          @(negedge clk);
          #1;
          t++;
          if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: feeder stalled", cyc);
        end
        @(negedge clk);
        ds_valid = 1'b0;
      end
      // when idle the feeder does NOT touch ds_valid (send_one owns it)
    end
  end

  // ── helpers ────────────────────────────────────────────────────────────────
  task automatic pulse_abort();
    @(negedge clk);
    abort_req = 1'b1;
    @(negedge clk);
    abort_req = 1'b0;
  endtask

  task automatic wait_abort_drained(input string what);
    int unsigned t;
    t = 0;
    #1;
    while (aborting) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: %s abort never drained", cyc, what);
    end
  endtask

  task automatic drain_all(input string what);
    int unsigned t;
    t = 0;
    #1;
    while (busy || stub_busy_w || aborting || exp_q.size() != 0) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT)
        $fatal(1, "TIMEOUT cyc=%0d: %s drain (busy=%0b stub=%0b ab=%0b exp=%0d)",
               cyc, what, busy, stub_busy_w, aborting, exp_q.size());
    end
    repeat (2) @(negedge clk);
    #1;
    if (q_count !== 4'd0 || q_empty !== 1'b1 || md_valid !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: not quiescent after drain", cyc, what);
    end
  endtask

  // ── main ───────────────────────────────────────────────────────────────────
  initial begin : main
    int unsigned t;
    int done_before;

    rst_n       = 1'b0;
    enable      = 1'b1;
    abort_req   = 1'b0;
    ds_valid    = 1'b0;
    ds_bits     = '0;
    force_stall = 1'b0;
    feed_on     = 1'b0;
    errors      = 0;
    serial      = 0;

    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("lat_mode=%d", lat_mode));
    void'($value$plusargs("n_aborts=%d", n_aborts));
    rng_state = seed ^ 32'hA77A_C4EE;
    $display("[TB] seq attack: lat=%0d seed=%0d n_aborts=%0d", lat_mode, seed, n_aborts);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    // ── AT-5: busy honesty with the beat only in the input skid ─────────────
    $display("[AT-5] skid-held beat busy honesty");
    enable = 1'b0;
    send_one(1);
    #1;
    for (int i = 0; i < 3; i++) begin
      if (busy !== 1'b1) begin
        errors++;
        $display("FAIL cyc=%0d [AT-5]: accepted descriptor pending (skid/FIFO) but busy=0 (qc=%0d)",
                 cyc, q_count);
      end
      @(negedge clk);
      #1;
    end
    enable = 1'b1;
    drain_all("AT-5");

    // ── AT-4: capacity + full/empty race ────────────────────────────────────
    $display("[AT-4] queue capacity: FIFO(%0d) + skid(2)", QDEPTH);
    enable = 1'b0;
    for (int i = 0; i < int'(QDEPTH) + 2; i++) send_one(2);
    repeat (4) @(negedge clk);
    #1;
    if (q_full !== 1'b1 || 32'(q_count) !== QDEPTH) begin
      errors++;
      $display("FAIL cyc=%0d [AT-4]: after %0d beats q_full=%0b q_count=%0d",
               cyc, QDEPTH + 2, q_full, q_count);
    end
    if (ds_ready !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d [AT-4]: FIFO+skid full but ds_ready still high", cyc);
    end
    // 11th beat must stall until dispatch frees space
    fork
      send_one(2);
      begin
        repeat (6) @(negedge clk);
        #1;
        if (32'(q_count) !== QDEPTH) begin
          errors++;
          $display("FAIL cyc=%0d [AT-4]: q_count moved while enable=0", cyc);
        end
        enable = 1'b1;             // release: all 11 must dispatch in order
      end
    join
    drain_all("AT-4");

    // ── AT-1 directed aborts in every state ─────────────────────────────────
    $display("[AT-1] abort in IDLE-empty");
    pulse_abort();
    wait_abort_drained("idle-empty");
    drain_all("AT-1a");

    $display("[AT-1] abort with queued descriptors (enable=0)");
    enable = 1'b0;
    for (int i = 0; i < 4; i++) send_one(3);
    repeat (3) @(negedge clk);
    pulse_abort();
    wait_abort_drained("idle-queued");
    enable = 1'b1;
    repeat (60) @(negedge clk);      // leaked serials would dispatch here
    drain_all("AT-1b");

    $display("[AT-1] abort in S_REQ (presented + stalled)");
    force_stall = 1'b1;
    enable      = 1'b1;
    for (int i = 0; i < 3; i++) send_one(4);
    t = 0;
    #1;
    while (!md_valid) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT: md_valid never rose");
    end
    done_before = done_cnt;
    pulse_abort();
    repeat (12) @(negedge clk);
    #1;
    if (aborting !== 1'b1 || md_valid !== 1'b1) begin
      errors++;
      $display("FAIL cyc=%0d [AT-1c]: presented descriptor retracted or abort ended early (ab=%0b mv=%0b)",
               cyc, aborting, md_valid);
    end
    force_stall = 1'b0;              // presented job must now run and complete
    t = 0;
    #1;
    while (done_cnt == done_before) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT: presented job never completed after abort");
    end
    wait_abort_drained("s_req");
    drain_all("AT-1c");

    $display("[AT-1] abort in S_WAIT (mid-beats) + double abort");
    send_one(48);                    // long job
    t = 0;
    #1;
    while (!stub_busy_w) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT: job never started");
    end
    done_before = done_cnt;
    pulse_abort();
    repeat (3) @(negedge clk);
    pulse_abort();                   // double abort while draining
    t = 0;
    #1;
    while (done_cnt == done_before) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT: in-flight job never completed under abort");
    end
    wait_abort_drained("s_wait");
    drain_all("AT-1d");

    // ── AT-2/AT-3: flood + abort storm ──────────────────────────────────────
    $display("[AT-2/3] descriptor flood + %0d random-phase aborts", n_aborts);
    feed_on = 1'b1;
    for (int a = 0; a < n_aborts; a++) begin
      repeat (2 + (rnd() % 40)) @(negedge clk);
      pulse_abort();
      if ((rnd() % 4) == 0) begin    // sometimes abort again immediately
        @(negedge clk);
        pulse_abort();
      end
      wait_abort_drained("storm");
    end
    feed_on = 1'b0;
    @(negedge clk);
    drain_all("storm");

    // ── summary ─────────────────────────────────────────────────────────────
    $display("AUDIT SUMMARY: %0d dispatched, %0d done, %0d leaks, %0d errors",
             dispatched, done_cnt, leak_cnt, errors);
    if (errors == 0) begin
      $display("TB PASS: seq attack suite clean (seed=%0d)", seed);
      $finish;
    end else begin
      $fatal(1, "TB FAIL: %0d error(s) — see traces above (seed=%0d)", errors, seed);
    end
  end

endmodule
