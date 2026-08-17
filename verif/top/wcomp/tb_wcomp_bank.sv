// tb_wcomp_bank.sv — F6(i) gate: apex_wcomp_bank keeps H_kv INDEPENDENT
// composite caches and routes every port by the engine select.
//
// What this TB is for (and what it is NOT for): the composite ARITHMETIC of
// seq_walker_comp is proven exhaustively by verif/seq_walker's comp sweeps
// (30,720 vectors per geometry, mutant-gated). This suite proves the
// property that unit cannot have: per-KV-group ISOLATION and ROUTING —
// LEVEL_C_INTEGRATION.md §9.1 F6 part (i), where ONE cache indexed by record
// address only had all four KV groups colliding in [0,T)/[T,2T) and 980 of
// 16,139 checks came back wrong, the survivors being exactly the LAST-staged
// group. The vectors' expected words still come from the one arbiter
// (verif/top/l3/walker_composite_golden.py) via gen_wcomp_vectors.py, and
// every scale enters through the REAL snoop path — nothing is back-doored.
//
// The four phases map 1:1 onto the routing contract in the bank's header:
//   A  isolation grid  — all groups store at the SAME record indices FIRST,
//                        then every group is asked; the emission order makes
//                        the highest group the last writer, so a collapsed
//                        bank reproduces the exact F6 signature (only the
//                        last-staged group survives).
//   B  re-select       — the grid is re-asked in reverse group order: moving
//                        the select must not disturb a cache's state.
//   S  stale isolation — a record written ONLY in group 0 must MISS in group
//                        1 (err_stale, no response), and the err pulse must
//                        reach the bank output from a non-zero instance
//                        (the OR-reduce), and group 0 must still answer.
//   F  flush broadcast — a partial record left in group 2 must be dropped by
//                        snp_flush pulsed while ANOTHER group is selected
//                        (D-020 is tile-wide); a routed flush leaves the
//                        stale beat count and the next record frames wrong.
`timescale 1ns/1ps

`include "walker_comp_sva.svh"

module tb_wcomp_bank;

  import apex_pkg::*;
  import seq_walker_pkg::*;

  localparam int unsigned CFG_D = `CFG_D_TB;
  localparam int unsigned N_ENG = `N_ENG_TB;
  localparam int unsigned ENG_W = (N_ENG > 1) ? $clog2(N_ENG) : 1;

  logic clk, rst_n;
  initial begin clk = 1'b0; rst_n = 1'b0; end
  always #5 clk = ~clk;

  string vec_file;
  int    n_checks, n_errors;
  string phase;

  logic [ENG_W-1:0]      eng_sel;
  logic                  snp_valid, snp_last, snp_flush, sq_valid;
  logic [15:0]           snp_data, sq_data;
  logic [WALK_SC_AW-1:0] snp_addr;
  logic                  req_valid, req_ready, req_is_qs;
  logic [WALK_SC_AW-1:0] req_idx;
  logic                  res_valid, res_ready;
  logic [31:0]           res_data;
  logic                  err_frame, err_stale;
  // DBG-v5 (2026-08-08) observability outputs — passive, not checked here
  // (this suite gates ROUTING/ISOLATION; the dbg attribution words are
  // read on silicon). Hooked so the -Wall PINMISSING gate stays honest.
  logic [N_ENG-1:0]      dbg_stale_eng;
  logic                  dbg_stale_sc, dbg_stale_sq;

  apex_wcomp_bank #(.N_ENG(N_ENG), .CFG_D(CFG_D)) dut (
    .clk(clk), .rst_n(rst_n),
    .eng_sel(eng_sel),
    .snp_valid(snp_valid), .snp_data(snp_data), .snp_last(snp_last),
    .snp_addr(snp_addr), .snp_flush(snp_flush),
    .sq_valid(sq_valid), .sq_data(sq_data),
    .req_valid(req_valid), .req_ready(req_ready), .req_is_qs(req_is_qs),
    .req_idx(req_idx),
    .res_valid(res_valid), .res_ready(res_ready), .res_data(res_data),
    .err_frame(err_frame), .err_stale(err_stale),
    .dbg_stale_eng(dbg_stale_eng),
    .dbg_stale_sc(dbg_stale_sc), .dbg_stale_sq(dbg_stale_sq)
  );

  logic unused_dbg_ok;
  assign unused_dbg_ok = &{1'b0, dbg_stale_eng, dbg_stale_sc, dbg_stale_sq};

  // the P1 precondition rides into EVERY instance of the bank (D-012)
  bind seq_walker_comp walker_comp_sva u_sva (
    .clk(clk), .rst_n(rst_n),
    .in_mul(state == ST_MUL), .sk_q(sk_q), .s_q_q(s_q_q)
  );

  // sticky observers for the pulse outputs. Cleared by REQUEST from the
  // stimulus (clr_*), so every sticky keeps exactly one procedural driver.
  logic saw_frame, saw_stale, clr_frame, clr_stale;
  always @(posedge clk) begin
    if (!rst_n || clr_frame) saw_frame <= 1'b0;
    else if (err_frame)      saw_frame <= 1'b1;
    if (!rst_n || clr_stale) saw_stale <= 1'b0;
    else if (err_stale)      saw_stale <= 1'b1;
  end

  // watchdog: a routing mutant can deadlock instead of mismatching
  int unsigned wd_cycles, wd_limit;
  always @(posedge clk) begin
    if (!rst_n) wd_cycles <= 0;
    else begin
      wd_cycles <= wd_cycles + 1;
      if (wd_cycles > wd_limit) begin
        $display("FAIL: WATCHDOG in phase %s after %0d cycles",
                 phase, wd_cycles);
        $display("WCOMP BANK d%0d: checks=%0d fails=%0d",
                 CFG_D, n_checks, n_errors + 1);
        $display("WCOMP BANK: FAIL");
        $finish;
      end
    end
  end

  task automatic chk_eq(input string what, input logic [31:0] exp,
                        input logic [31:0] got);
    begin
      n_checks++;
      if (exp !== got) begin
        $display("FAIL %s: expected %08h got %08h", what, exp, got);
        n_errors++;
      end
    end
  endtask

  task automatic chk_true(input string what, input logic cond);
    begin
      n_checks++;
      if (cond !== 1'b1) begin
        $display("FAIL %s", what);
        n_errors++;
      end
    end
  endtask

  // ── stimulus primitives (negedge drive, the house pattern) ───────────────
  task automatic sel_group(input logic [ENG_W-1:0] g);
    begin
      @(negedge clk);
      eng_sel = g;
      @(negedge clk);
    end
  endtask

  task automatic push_sq(input logic [15:0] v);
    begin
      @(negedge clk); sq_valid = 1'b1; sq_data = v;
      @(negedge clk); sq_valid = 1'b0;
    end
  endtask

  // a record of `n` identical elements at `addr`; last only when n == CFG_D
  task automatic push_record(input logic [WALK_SC_AW-1:0] addr,
                             input logic [15:0] v, input int n);
    begin
      for (int i = 0; i < n; i++) begin
        @(negedge clk);
        snp_valid = 1'b1; snp_data = v; snp_addr = addr;
        snp_last  = (i == int'(CFG_D) - 1);
      end
      @(negedge clk);
      snp_valid = 1'b0; snp_last = 1'b0;
    end
  endtask

  task automatic pulse_flush();
    begin
      @(negedge clk); snp_flush = 1'b1;
      @(negedge clk); snp_flush = 1'b0;
    end
  endtask

  // let a registered error pulse propagate into its sticky before it is
  // read — and, since the 2026-08-14 A0 write-cone pipeline, let an
  // in-flight commit drain: bank ingress (+1) + the unit's W1/W2/W3 write
  // pipe (+3) put sc_mem/sc_val 4 edges behind the last snoop beat, so 8
  // covers the pipe with margin (the tile's own closest commit->request
  // distance is ≥ ~70 cycles; see seq_walker_comp.sv's pipeline comment)
  task automatic settle();
    begin
      repeat (8) @(posedge clk);
      @(negedge clk);
    end
  endtask

  task automatic clear_err();
    begin
      @(negedge clk); clr_frame = 1'b1; clr_stale = 1'b1;
      @(negedge clk); clr_frame = 1'b0; clr_stale = 1'b0;
      @(negedge clk);
    end
  endtask

  // request one composite; got_v=0 means the bank refused (no response)
  task automatic ask(input logic is_qs, input logic [WALK_SC_AW-1:0] idx,
                     output logic [31:0] got, output logic got_v);
    int guard;
    begin
      @(negedge clk);
      req_valid = 1'b1; req_is_qs = is_qs; req_idx = idx;
      guard = 0;
      while (!(req_valid && req_ready) && guard < 1000) begin
        @(negedge clk); guard++;
      end
      @(negedge clk);
      req_valid = 1'b0;
      res_ready = 1'b1;
      guard = 0;
      while (!res_valid && guard < 200) begin @(posedge clk); guard++; end
      got_v = res_valid;
      got   = res_data;
      @(negedge clk);
      res_ready = 1'b0;
    end
  endtask

  // ── vector store ─────────────────────────────────────────────────────────
  localparam int unsigned MAXA = 256;
  int          a_g   [MAXA];
  int          a_rec [MAXA];
  logic [15:0] a_sq  [MAXA];
  logic [15:0] a_el  [MAXA];
  logic [31:0] a_cs  [MAXA];
  logic [31:0] a_qs  [MAXA];
  int          n_a;

  int          s_gw, s_gp, s_rec;
  logic [15:0] s_sq, s_el;
  logic [31:0] s_cs, s_qs;
  int          have_s;

  int          f_g, f_rec;
  logic [15:0] f_sq, f_big, f_small;
  logic [31:0] f_cs, f_qs;
  int          have_f;

  int    fd, cfgd_seen, neng_seen;
  string line;
  logic [31:0] got;
  logic        got_v;
  int          t_a, t_b;                 // $sscanf scratch (no array refs)
  logic [15:0] t_c, t_d;
  logic [31:0] t_e, t_f;

  initial begin
    if (!$value$plusargs("vectors=%s", vec_file)) begin
      $display("FAIL: +vectors=<file> required"); $finish;
    end
    if (!$value$plusargs("watchdog=%d", wd_limit)) wd_limit = 2_000_000;

    n_checks = 0; n_errors = 0; n_a = 0; have_s = 0; have_f = 0;
    cfgd_seen = 0; neng_seen = 0; phase = "init";
    eng_sel = '0; snp_valid = 1'b0; snp_last = 1'b0; snp_flush = 1'b0;
    sq_valid = 1'b0; snp_data = '0; snp_addr = '0; sq_data = '0;
    req_valid = 1'b0; req_is_qs = 1'b0; req_idx = '0; res_ready = 1'b0;
    clr_frame = 1'b0; clr_stale = 1'b0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    fd = $fopen(vec_file, "r");
    if (fd == 0) begin
      $display("FAIL: cannot open %s", vec_file); $finish;
    end
    while ($fgets(line, fd) != 0) begin
      if (line.len() < 2) continue;
      if (line.substr(0, 0) == "#") continue;
      if (line.substr(0, 3) == "CFGD") begin
        void'($sscanf(line, "CFGD %d", cfgd_seen));
        continue;
      end
      if (line.substr(0, 3) == "NENG") begin
        void'($sscanf(line, "NENG %d", neng_seen));
        continue;
      end
      if (line.substr(0, 0) == "A") begin
        void'($sscanf(line, "A %d %d %h %h %h %h", t_a, t_b, t_c, t_d,
                      t_e, t_f));
        a_g[n_a] = t_a; a_rec[n_a] = t_b; a_sq[n_a] = t_c;
        a_el[n_a] = t_d; a_cs[n_a] = t_e; a_qs[n_a] = t_f;
        n_a++;
        continue;
      end
      if (line.substr(0, 0) == "S") begin
        void'($sscanf(line, "S %d %d %d %h %h %h %h", s_gw, s_gp, s_rec,
                      s_sq, s_el, s_cs, s_qs));
        have_s = 1;
        continue;
      end
      if (line.substr(0, 0) == "F") begin
        void'($sscanf(line, "F %d %d %h %h %h %h %h", f_g, f_rec, f_sq,
                      f_big, f_small, f_cs, f_qs));
        have_f = 1;
        continue;
      end
    end
    $fclose(fd);

    if (cfgd_seen != int'(CFG_D) || neng_seen != int'(N_ENG)) begin
      $display("FAIL: vectors are D=%0d/N_ENG=%0d, build is D=%0d/N_ENG=%0d",
               cfgd_seen, neng_seen, CFG_D, N_ENG);
      $display("WCOMP BANK: FAIL");
      $finish;
    end
    if (n_a == 0 || have_s == 0 || have_f == 0) begin
      $display("FAIL: vector file is missing a phase (A=%0d S=%0d F=%0d)",
               n_a, have_s, have_f);
      $display("WCOMP BANK: FAIL");
      $finish;
    end

    // ── phase A part 1: EVERY group stores before ANY group is asked ───────
    phase = "A-store";
    for (int i = 0; i < n_a; i++) begin
      sel_group(ENG_W'(a_g[i]));
      push_sq(a_sq[i]);
      push_record(WALK_SC_AW'(a_rec[i]), a_el[i], int'(CFG_D));
    end
    settle();
    chk_true("A-store: unexpected err_frame", !saw_frame);
    chk_true("A-store: unexpected err_stale", !saw_stale);

    // ── phase A part 2: each group must still hold its OWN scales ──────────
    phase = "A-check";
    for (int i = 0; i < n_a; i++) begin
      sel_group(ENG_W'(a_g[i]));
      ask(1'b0, WALK_SC_AW'(a_rec[i]), got, got_v);
      chk_true($sformatf("A cs g=%0d rec=%0d: no response", a_g[i], a_rec[i]),
               got_v);
      chk_eq($sformatf("A cs g=%0d rec=%0d", a_g[i], a_rec[i]), a_cs[i], got);
      ask(1'b1, WALK_SC_AW'(a_rec[i]), got, got_v);
      chk_true($sformatf("A qs g=%0d rec=%0d: no response", a_g[i], a_rec[i]),
               got_v);
      chk_eq($sformatf("A qs g=%0d rec=%0d", a_g[i], a_rec[i]), a_qs[i], got);
    end

    // ── phase B: re-ask in REVERSE order (select churn must not disturb) ───
    phase = "B-reselect";
    for (int i = n_a - 1; i >= 0; i--) begin
      sel_group(ENG_W'(a_g[i]));
      ask(1'b0, WALK_SC_AW'(a_rec[i]), got, got_v);
      chk_true($sformatf("B cs g=%0d rec=%0d: no response", a_g[i], a_rec[i]),
               got_v);
      chk_eq($sformatf("B cs g=%0d rec=%0d", a_g[i], a_rec[i]), a_cs[i], got);
    end

    // ── phase S: stale isolation + the error OR-reduce ─────────────────────
    phase = "S-stale";
    sel_group(ENG_W'(s_gw));
    push_sq(s_sq);
    push_record(WALK_SC_AW'(s_rec), s_el, int'(CFG_D));
    clear_err();
    sel_group(ENG_W'(s_gp));
    ask(1'b0, WALK_SC_AW'(s_rec), got, got_v);
    chk_true($sformatf("S: group %0d answered a record only group %0d wrote",
                       s_gp, s_gw), !got_v);
    chk_true($sformatf("S: no err_stale from instance %0d (OR-reduce)", s_gp),
             saw_stale);
    sel_group(ENG_W'(s_gw));
    ask(1'b0, WALK_SC_AW'(s_rec), got, got_v);
    chk_true("S: writer group lost its record", got_v);
    chk_eq($sformatf("S cs g=%0d rec=%0d", s_gw, s_rec), s_cs, got);
    ask(1'b1, WALK_SC_AW'(s_rec), got, got_v);
    chk_eq($sformatf("S qs g=%0d rec=%0d", s_gw, s_rec), s_qs, got);

    // ── phase F: snp_flush is BROADCAST, not routed ────────────────────────
    phase = "F-flush";
    sel_group(ENG_W'(f_g));
    push_sq(f_sq);
    push_record(WALK_SC_AW'(f_rec), f_big, int'(CFG_D) / 2);   // partial
    sel_group(ENG_W'(0));                                      // move away
    pulse_flush();                                             // tile-wide
    sel_group(ENG_W'(f_g));
    clear_err();
    push_record(WALK_SC_AW'(f_rec), f_small, int'(CFG_D));     // full record
    settle();
    chk_true("F: err_frame after broadcast flush (partial record survived)",
             !saw_frame);
    ask(1'b0, WALK_SC_AW'(f_rec), got, got_v);
    chk_true("F: no response after broadcast flush", got_v);
    chk_eq($sformatf("F cs g=%0d rec=%0d", f_g, f_rec), f_cs, got);
    ask(1'b1, WALK_SC_AW'(f_rec), got, got_v);
    chk_eq($sformatf("F qs g=%0d rec=%0d", f_g, f_rec), f_qs, got);

    phase = "done";
    $display("WCOMP BANK d%0d: checks=%0d fails=%0d", CFG_D, n_checks,
             n_errors);
    if (n_errors == 0) $display("WCOMP BANK: ALL PASS");
    else               $display("WCOMP BANK: FAIL");
    $finish;
  end

endmodule
