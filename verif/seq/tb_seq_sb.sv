// tb_seq_sb.sv — SEQ (seq_walker) vector-driven scoreboard TB (house
// pattern, verif/mxe/sb lineage). Verilator --binary --timing --assert.
//
// MXE is a verified black box (verif/mxe 9/9): it is stubbed here with
// mxe_stub.sv, a behavioral model of its D-006/§3/§5 CONTRACT (job accept
// -> busy -> M result beats -> done strictly after the last beat is
// accepted; illegal descriptor -> desc_error, no done, no beats) with
// randomized accept latencies, compute latencies, inter-beat gaps and
// result-side backpressure.
//
// Scoreboard: descriptors come from the independent golden queue model in
// gen_seq_vectors.py. Every descriptor accepted by the stub is compared
// IN ORDER and bit-exact against the pushed sequence (D-006 walk order);
// the generator's legality verdict is cross-checked against the stub's own
// (dual-oracle); done/err counts, forwarded desc_error pulses, queue
// status, abort-and-drain semantics, enable gating, and phase-targeted
// mid-operation hard resets are all checked at drain points.
//
// SVA: house §5 stability pack (verif/common/apex_stream1_sva.svh) bound on
// ALL THREE streams (ds in, md out, stub result), plus the SEQ-shaped
// contract (seq_sb_sva.svh: D-006 serialization, busy honesty, queue
// bookkeeping, abort drain). Compiled under --assert in every build (D-012).
//
// Plusargs: +vectors=<file> +seed=<n> +lat_mode=0|1|2 (stub fast/random/
//           storm) +bp_mode=0|1|2 (result backpressure) +gap_mode=0|1|2
//           (descriptor feed gaps)
// Records: see gen_seq_vectors.py header (D/N/A/G/Q/R/E).
//
// TB discipline (V0/mxe lineage): stimulus at NEGEDGE, checks sampled 1 ns
// after the edge; $fatal watchdog bounds the run.

`include "apex_stream1_sva.svh"
`include "seq_sb_sva.svh"

module tb_seq_sb;

  import apex_pkg::*;

  localparam int unsigned QDEPTH     = 8;
  localparam int unsigned DW         = 128;
  localparam int unsigned WAIT_LIMIT = 2_000_000;

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

  // ── plusargs ───────────────────────────────────────────────────────────────
  int unsigned seed     = 32'h5E9_0001;
  int unsigned lat_mode = 1;
  int unsigned bp_mode  = 1;
  int unsigned gap_mode = 1;
  string       vec_path;

  // ── DUT ────────────────────────────────────────────────────────────────────
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

  // ── MXE contract stub ──────────────────────────────────────────────────────
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

  // ── house §5 stability pack on ALL streams (reused, not rewritten) ────────
  bind seq_walker apex_stream1_sva #(.WIDTH(128), .NAME("seq.ds"))
    u_sva_ds (.clk(clk), .rst_n(rst_n), .valid(ds_valid), .ready(ds_ready),
              .data(128'(ds_desc)));

  bind seq_walker apex_stream1_sva #(.WIDTH(128), .NAME("seq.md"))
    u_sva_md (.clk(clk), .rst_n(rst_n), .valid(md_valid), .ready(md_ready),
              .data(128'(md_desc)));

  bind mxe_stub apex_stream1_sva #(.WIDTH(257), .NAME("mxe_stub.res"))
    u_sva_res (.clk(clk), .rst_n(rst_n), .valid(res_valid), .ready(res_ready),
               .data(257'(res_beat)));

  // ── SEQ contract checker (D-006 / busy honesty / queue bookkeeping) ───────
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

  // ── result-stream consumer (LFSR backpressure) ────────────────────────────
  logic [31:0] blfsr;
  always @(negedge clk) begin
    if (!rst_n) begin
      blfsr     <= (seed ^ 32'h0B50_1EAF) | 32'h1;
      res_ready <= 1'b0;
    end else begin
      blfsr <= {blfsr[30:0], blfsr[31] ^ blfsr[21] ^ blfsr[1] ^ blfsr[0]};
      unique case (bp_mode)
        0:       res_ready <= 1'b1;
        1:       res_ready <= |blfsr[1:0];           // ~75%
        default: res_ready <= (blfsr[3:0] == 4'h0);  // storm ~6%
      endcase
    end
  end

  // ── bookkeeping ────────────────────────────────────────────────────────────
  int errors;
  logic [DW-1:0] exp_d [$];
  bit            exp_l [$];
  int dispatched, done_cnt, err_cnt, fwd_err, discarded;
  bit discard;
  bit in_flight_tb, cur_legal;
  int cur_m, beat_cnt;
  int unsigned last_done_cyc;
  logic md_valid_q;

  // ── coverage buckets (manual, run2 pattern; "COV <name> <n>") ─────────────
  int cov_cat [0:11];
  int cov_drains, cov_qfull, cov_skid_bp, cov_b2b, cov_beats_last;
  int cov_abort_inflight, cov_abort_queued, cov_abort_empty, cov_abort_feed;
  int cov_gate_off, cov_gate_on, cov_qcheck;
  int cov_rst [1:4];
  int cov_bp [0:2];
  int cov_gap [0:2];
  int cov_lat [0:2];
  int done_cnt_total, err_cnt_total;   // never resynced (coverage only)

  localparam string CATN [0:11] = '{
    "legal_plain", "legal_k1", "legal_k2048", "legal_m1", "bad_opcode",
    "k_zero", "k_big", "m_zero", "n_zero", "m_big", "n_big", "buf_bad"};

  function automatic void print_coverage();
    for (int c = 0; c <= 11; c++)
      $display("COV cat_%s %0d", CATN[c], cov_cat[c]);
    $display("COV drains %0d",         cov_drains);
    $display("COV q_full_seen %0d",    cov_qfull);
    $display("COV skid_bp %0d",        cov_skid_bp);
    $display("COV b2b_dispatch %0d",   cov_b2b);
    $display("COV beats_last %0d",     cov_beats_last);
    $display("COV jobs_done %0d",      done_cnt_total);
    $display("COV jobs_rejected %0d",  err_cnt_total);
    $display("COV abort_inflight %0d", cov_abort_inflight);
    $display("COV abort_queued %0d",   cov_abort_queued);
    $display("COV abort_empty %0d",    cov_abort_empty);
    $display("COV abort_feed %0d",     cov_abort_feed);
    $display("COV gate_off %0d",       cov_gate_off);
    $display("COV gate_on %0d",        cov_gate_on);
    $display("COV qcheck %0d",         cov_qcheck);
    for (int p = 1; p <= 4; p++) $display("COV rst_p%0d %0d", p, cov_rst[p]);
    for (int m = 0; m <= 2; m++) begin
      $display("COV bp_m%0d %0d",  m, cov_bp[m]);
      $display("COV gap_m%0d %0d", m, cov_gap[m]);
      $display("COV lat_m%0d %0d", m, cov_lat[m]);
    end
  endfunction

  // ── monitors (single posedge process; TB bookkeeping is blocking) ─────────
  logic [DW-1:0] mon_exp_d;
  bit            mon_exp_l;
  // (tip lineage: procedural always + @(posedge) so TB bookkeeping can use
  //  blocking assignments without BLKSEQ noise)
  always begin
    @(posedge clk);
    if (rst_n) begin
      // descriptor accepted by the stub (the D-006 walk order)
      if (acc_pulse) begin
        if (exp_d.size() == 0) begin
          if (!discard) begin
            errors++;
            $display("FAIL cyc=%0d: ORPHAN dispatch desc=%032x", cyc,
                     128'(md_desc));
          end
        end else begin
          mon_exp_d = exp_d.pop_front();
          mon_exp_l = exp_l.pop_front();
          if (!discard) begin
            if (128'(md_desc) !== mon_exp_d) begin
              errors++;
              $display("FAIL cyc=%0d: dispatch #%0d got %032x exp %032x",
                       cyc, dispatched, 128'(md_desc), mon_exp_d);
            end
            if (mon_exp_l !== acc_legal) begin
              errors++;
              $display("FAIL cyc=%0d: LEGALITY oracle mismatch (golden %0b, stub %0b) desc=%032x",
                       cyc, mon_exp_l, acc_legal, 128'(md_desc));
            end
            dispatched++;
          end else begin
            discarded++;
          end
        end
        cur_legal    = acc_legal;
        cur_m        = int'(md_desc.m_dim);
        in_flight_tb = 1'b1;
        beat_cnt     = 0;
      end

      // job end: done (legal) / desc_error (rejected)
      if (stub_done) begin
        if (!discard) begin
          done_cnt++;
          done_cnt_total++;
          if (!cur_legal) begin
            errors++;
            $display("FAIL cyc=%0d: done for an ILLEGAL descriptor", cyc);
          end
          if (beat_cnt != cur_m) begin
            errors++;
            $display("FAIL cyc=%0d: done after %0d beats, expected M=%0d",
                     cyc, beat_cnt, cur_m);
          end
        end
        in_flight_tb  = 1'b0;
        beat_cnt      = 0;
        last_done_cyc = cyc;
      end
      if (stub_derr) begin
        if (!discard) begin
          err_cnt++;
          err_cnt_total++;
          if (cur_legal) begin
            errors++;
            $display("FAIL cyc=%0d: LEGAL descriptor rejected", cyc);
          end
          if (beat_cnt != 0) begin
            errors++;
            $display("FAIL cyc=%0d: rejected descriptor produced beats", cyc);
          end
        end
        in_flight_tb = 1'b0;
      end
      if (seq_derr && !discard) fwd_err++;

      // result beats: exactly M per legal job, last on the final one
      if (res_valid && res_ready) begin
        beat_cnt++;
        if (!in_flight_tb && !discard) begin
          errors++;
          $display("FAIL cyc=%0d: result beat with no job in flight", cyc);
        end
        if (res_beat.last) begin
          if (!discard) begin
            cov_beats_last++;
            if (beat_cnt != cur_m) begin
              errors++;
              $display("FAIL cyc=%0d: last on beat %0d, expected M=%0d",
                       cyc, beat_cnt, cur_m);
            end
          end
        end
      end

      // coverage observers
      if (q_full) cov_qfull++;
      if (ds_valid && !ds_ready) cov_skid_bp++;
      if (md_valid && !md_valid_q && (cyc - last_done_cyc) <= 3) cov_b2b++;
    end
    md_valid_q = md_valid && rst_n;
  end

  // beat payloads are the stub's own encoding; SEQ's contract is count +
  // last-flag (checked above) — the payload itself is not SEQ-visible
  logic unused_res;
  assign unused_res = &{1'b0, res_beat.data};

  // ── descriptor feed (negedge discipline) ───────────────────────────────────
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
      ds_valid = 1'b0;
      repeat (g) @(negedge clk);
    end
  endtask

  task automatic send_raw(input logic [DW-1:0] d);
    int unsigned t;
    @(negedge clk);
    feed_gap();
    ds_valid = 1'b1;
    ds_bits  = d;
    #1;
    t = 0;
    while (!ds_ready) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: ds_ready never rose", cyc);
    end
  endtask

  task automatic idle_input();
    @(negedge clk);
    ds_valid = 1'b0;
  endtask

  // ── drains / checks (all watchdog-bounded) ─────────────────────────────────
  task automatic drain_pipe();
    int unsigned t;
    idle_input();
    t = 0;
    #1;
    while (exp_d.size() != 0 || busy || stub_busy_w || in_flight_tb) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: drain never completed (exp=%0d busy=%0b stub=%0b)",
                                 cyc, exp_d.size(), busy, stub_busy_w);
    end
    repeat (2) @(negedge clk);
  endtask

  task automatic check_counts(input int e_done, input int e_err,
                              input string what);
    #1;
    if (done_cnt != e_done) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: done_cnt %0d exp %0d", cyc, what,
               done_cnt, e_done);
    end
    if (err_cnt != e_err) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: err_cnt %0d exp %0d", cyc, what,
               err_cnt, e_err);
    end
    if (fwd_err != err_cnt) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: forwarded desc_error pulses %0d != %0d",
               cyc, what, fwd_err, err_cnt);
    end
    if (dispatched != done_cnt + err_cnt) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: dispatched %0d != done+err %0d", cyc,
               what, dispatched, done_cnt + err_cnt);
    end
    if (q_empty !== 1'b1 || q_count !== 4'd0 || md_valid !== 1'b0
        || aborting !== 1'b0 || busy !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: not quiescent after drain (qe=%0b qc=%0d mv=%0b ab=%0b busy=%0b)",
               cyc, what, q_empty, q_count, md_valid, aborting, busy);
    end
  endtask

  // ── abort-window feed (auditor blind-spot closure, verif/audit_seq) ───────
  // The original do_abort idled the ds stream before pulsing abort, so the
  // suite never exercised a descriptor whose handshake completes DURING the
  // abort window — in particular one completing in the window's FINAL cycle,
  // which is still inside the input skid when `aborting` falls (the leak the
  // audit exposed). do_abort now keeps the ds stream running at full rate
  // through the window with serial-watermarked legal descriptors (rq_scale;
  // legality ignores it): every in-window accept must be discarded by the
  // DUT (a leak dispatches a serial that is not in — or out of order with —
  // the golden queue and fails the bit-exact compare / quiescence checks);
  // post-window accepts are pushed to the golden queue, proving the fix
  // does not over-discard and preserves order across the boundary.

  int abort_serial;

  function automatic logic [DW-1:0] af_desc();
    mxe_desc_t d;
    d = '0;
    d.opcode   = 8'(OP_GEMM_WS);
    d.m_dim    = 12'd2;
    d.k_dim    = 12'd16;
    d.n_dim    = 12'd4;
    abort_serial++;
    d.rq_scale = 16'(abort_serial);      // watermark; legality ignores it
    return DW'(d);
  endfunction

  // pre: ds_valid==1 with a watermarked beat, set at the abort-pulse negedge.
  // Feeds back-to-back until 3 beats have been accepted AFTER the window.
  // Window state is sampled 1 ns after the negedge: abort_req is negedge-
  // driven and `aborting` is a register, so the sampled value is exactly the
  // state at the accepting posedge.
  task automatic abort_feed_through();
    int unsigned t;
    int post;
    logic [DW-1:0] d;
    post = 0;
    d = ds_bits;
    forever begin
      #1;
      t = 0;
      while (!ds_ready) begin
        @(negedge clk);
        #1;
        t++;
        if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: abort-feed stalled", cyc);
      end
      if (abort_req || aborting) begin   // in-window: DUT must discard it
        discarded++;
        cov_abort_feed++;
      end else begin                     // post-window: must survive, in order
        exp_d.push_back(d);
        exp_l.push_back(1'b1);
        post++;
      end
      @(negedge clk);
      if (post >= 3) break;
      d       = af_desc();
      ds_bits = d;
    end
    ds_valid = 1'b0;
  endtask

  task automatic do_abort(input int kind);
    int unsigned t;
    int keep;
    bit acc1;
    idle_input();
    if (kind == 0) begin
      t = 0;
      #1;
      while (!in_flight_tb
             && !(exp_d.size() == 0 && !busy && !stub_busy_w)) begin
        @(negedge clk);
        #1;
        t++;
        if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: abort-0 wait", cyc);
      end
      if (in_flight_tb) cov_abort_inflight++;
      else cov_abort_empty++;
    end else if (kind == 1) begin
      cov_abort_queued++;
    end else begin
      cov_abort_empty++;
    end

    // pulse the abort with the ds stream RUNNING (first fed beat lands in
    // the abort_req cycle itself)
    @(negedge clk);
    abort_req = 1'b1;
    ds_valid  = 1'b1;
    ds_bits   = af_desc();

    // golden queue: everything not yet presented dies at the pulse; a
    // presented descriptor (md_valid) completes (§5: never retracted) and
    // stays at the queue front so post-window fed beats line up behind it
    keep = md_valid ? 1 : 0;
    while (exp_d.size() > keep) begin
      void'(exp_d.pop_back());
      void'(exp_l.pop_back());
      discarded++;
    end

    if (enable) begin
      fork
        begin
          @(negedge clk);
          abort_req = 1'b0;
        end
      join_none
      abort_feed_through();
    end else begin
      // dispatch gated off: post-window beats would sit in the queue and
      // desync the golden counts, so feed exactly the 2-cycle window —
      // the second accept IS the final-window-cycle (skid-resident) case
      #1;
      acc1 = ds_ready;                   // completes at the abort_req posedge
      if (acc1) begin
        discarded++;
        cov_abort_feed++;
      end
      @(negedge clk);
      abort_req = 1'b0;
      if (acc1) ds_bits = af_desc();     // second beat: FINAL window cycle
      #1;
      if (!ds_ready) begin
        errors++;
        $display("FAIL cyc=%0d: ds_ready low in the final abort cycle (skid not draining)", cyc);
      end else if (!(abort_req || aborting)) begin
        errors++;
        $display("FAIL cyc=%0d: abort window closed before its final cycle (enable=0 feed)", cyc);
      end else begin
        discarded++;
        cov_abort_feed++;
      end
      @(negedge clk);
      ds_valid = 1'b0;
    end

    t = 0;
    #1;
    while (aborting || busy || stub_busy_w || in_flight_tb) begin
      @(negedge clk);
      #1;
      t++;
      if (t > WAIT_LIMIT) $fatal(1, "TIMEOUT cyc=%0d: abort never drained", cyc);
    end

    // everything not dispatched was discarded; resync (golden does the same)
    discarded += exp_d.size();
    exp_d.delete();
    exp_l.delete();
    done_cnt   = 0;
    err_cnt    = 0;
    fwd_err    = 0;
    dispatched = 0;

    repeat (2) @(negedge clk);
    #1;
    if (q_empty !== 1'b1 || q_count !== 4'd0 || md_valid !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d: queue not drained after abort (qc=%0d)",
               cyc, q_count);
    end
  endtask

  // ── mid-operation reset (phase-targeted) ───────────────────────────────────
  function automatic bit phase_cond(input int phase);
    unique case (phase)
      1: return 32'(q_count) >= 2;              // queue filling, head stalled
      2: return md_valid && !md_ready;          // descriptor presented, stalled
      3: return in_flight_tb && !md_valid;      // job in flight
      4: return aborting;                       // abort draining
      default: return 1'b0;
    endcase
  endfunction

  function automatic logic [DW-1:0] sac_desc(input int m);
    mxe_desc_t d;
    d = '0;
    d.opcode = 8'(OP_GEMM_WS);
    d.m_dim  = 12'(m % 4096);
    d.k_dim  = 12'd16;
    d.n_dim  = 12'd4;
    return DW'(d);
  endfunction

  task automatic sac_driver(input int phase);
    if (phase == 4) begin
      repeat (4) send_raw(sac_desc(48));
      @(negedge clk);
      abort_req = 1'b1;
      @(negedge clk);
      abort_req = 1'b0;
      forever @(negedge clk);
    end else begin
      forever send_raw(sac_desc(phase == 3 ? 48 : 4));
    end
  endtask

  task automatic do_reset(input int phase);
    int unsigned t;
    drain_pipe();
    discard = 1'b1;
    if (phase <= 2) force_stall = 1'b1;

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

    $display("RESET cyc=%0d: mid-op reset in phase %0d (qc=%0d mv=%0b inflight=%0b aborting=%0b)",
             cyc, phase, q_count, md_valid, in_flight_tb, aborting);
    rst_n     = 1'b0;
    ds_valid  = 1'b0;
    abort_req = 1'b0;
    repeat (3) @(negedge clk);
    rst_n       = 1'b1;
    force_stall = 1'b0;
    repeat (3) @(negedge clk);

    exp_d.delete();
    exp_l.delete();
    done_cnt     = 0;
    err_cnt      = 0;
    fwd_err      = 0;
    dispatched   = 0;
    in_flight_tb = 1'b0;
    beat_cnt     = 0;
    discard      = 1'b0;

    #1;
    if (busy !== 1'b0 || q_empty !== 1'b1 || q_count !== 4'd0
        || md_valid !== 1'b0 || aborting !== 1'b0 || ds_ready !== 1'b1
        || stub_busy_w !== 1'b0 || res_valid !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d: dirty state after mid-op reset (phase %0d)",
               cyc, phase);
    end
    cov_rst[phase]++;
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

  // ── main ───────────────────────────────────────────────────────────────────
  initial begin : main
    string tag;
    logic [DW-1:0] d;
    int lg, cat, n1, n2, n3, cnt;
    bit ended;

    rst_n       = 1'b0;
    enable      = 1'b1;
    abort_req   = 1'b0;
    ds_valid    = 1'b0;
    ds_bits     = '0;
    force_stall = 1'b0;
    discard     = 1'b0;
    errors      = 0;
    abort_serial = 0;

    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("lat_mode=%d", lat_mode));
    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("gap_mode=%d", gap_mode));
    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'h5E9D_BEEF;
    cov_bp[bp_mode]++;
    cov_gap[gap_mode]++;
    cov_lat[lat_mode]++;

    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);
    $display("[TB] vectors=%s lat=%0d bp=%0d gap=%0d seed=%0d",
             vec_path, lat_mode, bp_mode, gap_mode, seed);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);
    #1;
    if (busy !== 1'b0 || q_empty !== 1'b1 || ds_ready !== 1'b1) begin
      errors++;
      $display("FAIL cyc=%0d: dirty state out of reset", cyc);
    end

    ended = 1'b0;
    while (!ended && next_line()) begin
      case (cur_line.getc(0))
        "D": begin
          cnt = $sscanf(cur_line, "%s %h %d %d", tag, d, lg, cat);
          if (cnt != 4 || tag != "D") $fatal(1, "bad D line: %s", cur_line);
          exp_d.push_back(d);
          exp_l.push_back(lg != 0);
          if (cat >= 0 && cat <= 11) cov_cat[cat]++;
          send_raw(d);
        end
        "N": begin
          cnt = $sscanf(cur_line, "%s %d %d", tag, n1, n2);
          if (cnt != 3) $fatal(1, "bad N line: %s", cur_line);
          drain_pipe();
          check_counts(n1, n2, "drain");
          cov_drains++;
        end
        "A": begin
          cnt = $sscanf(cur_line, "%s %d", tag, n1);
          if (cnt != 2) $fatal(1, "bad A line: %s", cur_line);
          do_abort(n1);
        end
        "G": begin
          cnt = $sscanf(cur_line, "%s %d", tag, n1);
          if (cnt != 2) $fatal(1, "bad G line: %s", cur_line);
          idle_input();      // drop ds_valid FIRST: exactly one accept per send
          enable = 1'(n1);
          if (n1 == 0) cov_gate_off++;
          else cov_gate_on++;
        end
        "Q": begin
          cnt = $sscanf(cur_line, "%s %d %d %d", tag, n1, n2, n3);
          if (cnt != 4) $fatal(1, "bad Q line: %s", cur_line);
          idle_input();
          repeat (12) @(negedge clk);
          #1;
          if (q_full !== 1'(n1) || 32'(q_count) !== n2
              || exp_d.size() != n3) begin
            errors++;
            $display("FAIL cyc=%0d [Q]: full=%0b/%0d count=%0d/%0d undisp=%0d/%0d",
                     cyc, q_full, n1, q_count, n2, exp_d.size(), n3);
          end
          cov_qcheck++;
        end
        "R": begin
          cnt = $sscanf(cur_line, "%s %d", tag, n1);
          if (cnt != 2 || n1 < 1 || n1 > 4) $fatal(1, "bad R line: %s", cur_line);
          do_reset(n1);
        end
        "E": begin
          ended = 1'b1;
          drain_pipe();
          #1;
          if (res_valid !== 1'b0 || stub_busy_w !== 1'b0) begin
            errors++;
            $display("FAIL cyc=%0d: stub not idle at end of run", cyc);
          end
          print_coverage();
          if (errors == 0) begin
            $display("TB PASS: %0d dispatched (%0d done, %0d rejected, %0d discarded by abort/reset), %0d last-beats checked, 0 errors (lat=%0d bp=%0d gap=%0d seed=%0d)",
                     dispatched, done_cnt, err_cnt, discarded,
                     cov_beats_last, lat_mode, bp_mode, gap_mode, seed);
            $finish;
          end else begin
            $fatal(1, "TB FAIL: %0d error(s) (lat=%0d bp=%0d gap=%0d seed=%0d)",
                   errors, lat_mode, bp_mode, gap_mode, seed);
          end
        end
        default: $fatal(1, "unexpected record: %s", cur_line);
      endcase
    end
    if (!ended) $fatal(1, "vector file ended without an E record");
  end

endmodule
