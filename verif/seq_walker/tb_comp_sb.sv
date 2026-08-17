// tb_comp_sb.sv — B1 stage-3 scoreboard TB for seq_walker_comp.
//
// Stage 2 proved the walker emits the right SEQUENCE. This proves the
// composite ARITHMETIC (B1_WALKER.md §5 row 3) against the stage-0 oracle
// verif/top/l3/walker_composite_golden.py, which was itself validated
// bit-exactly against the live L3 op stream.
//
// Every scale enters through the REAL path: a record of D identical fp16
// elements is pushed into the snoop port, and the unit derives the cached
// scale with cq_fp_pkg::scale_from_amax exactly as the KVQ engine does.
// Nothing is back-doored into the cache, so this simultaneously exercises the
// snoop, the amax reduce, the scale function and the composite datapath.
//
// The bound walker_comp_sva additionally holds the P1 precondition (every
// consumed scale is positive-normal) that the software oracle CANNOT check.
`timescale 1ns/1ps

`include "walker_comp_sva.svh"

module tb_comp_sb;

  import apex_pkg::*;
  import seq_walker_pkg::*;

  localparam int unsigned CFG_D = `CFG_D_TB;

  logic clk, rst_n;
  initial begin clk = 1'b0; rst_n = 1'b0; end
  always #5 clk = ~clk;

  string vec_file;
  int    n_cs, n_qs, n_err, n_vec;

  logic                  snp_valid, snp_last, snp_flush, sq_valid;
  logic [15:0]           snp_data, sq_data;
  logic [WALK_SC_AW-1:0] snp_addr;
  logic                  req_valid, req_ready, req_is_qs;
  logic [WALK_SC_AW-1:0] req_idx;
  logic                  res_valid, res_ready;
  logic [31:0]           res_data;
  logic                  err_frame, err_stale;

  seq_walker_comp #(.CFG_D(CFG_D)) dut (
    .clk(clk), .rst_n(rst_n),
    .snp_valid(snp_valid), .snp_data(snp_data), .snp_last(snp_last),
    .snp_addr(snp_addr), .snp_flush(snp_flush),
    .sq_valid(sq_valid), .sq_data(sq_data),
    .req_valid(req_valid), .req_ready(req_ready), .req_is_qs(req_is_qs),
    .req_idx(req_idx),
    .res_valid(res_valid), .res_ready(res_ready), .res_data(res_data),
    .err_frame(err_frame), .err_stale(err_stale),
    .dbg_stale_sc(u_dbg_sc), .dbg_stale_sq(u_dbg_sq)
  );

  bind seq_walker_comp walker_comp_sva u_sva (
    .clk(clk), .rst_n(rst_n),
    .in_mul(state == ST_MUL), .sk_q(sk_q), .s_q_q(s_q_q)
  );

  always begin
    @(posedge clk);
    if (rst_n && err_frame) begin
      $display("FAIL: unexpected err_frame"); n_err++;
    end
    if (rst_n && err_stale) begin
      $display("FAIL: unexpected err_stale (cache miss)"); n_err++;
    end
  end

  // push a record of D identical elements at `addr` — amax is that element
  task automatic push_uniform(input logic [WALK_SC_AW-1:0] addr,
                              input logic [15:0] v);
    begin
      for (int i = 0; i < int'(CFG_D); i++) begin
        @(negedge clk);
        snp_valid = 1'b1; snp_data = v; snp_addr = addr;
        snp_last  = (i == int'(CFG_D) - 1);
      end
      @(negedge clk);
      snp_valid = 1'b0; snp_last = 1'b0;
    end
  endtask

  task automatic ask(input logic is_qs, input logic [WALK_SC_AW-1:0] idx,
                     output logic [31:0] got);
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
      while (!res_valid && guard < 1000) begin @(posedge clk); guard++; end
      got = res_data;
      @(negedge clk);
      res_ready = 1'b0;
    end
  endtask

  int          fd, cfgd_seen;
  string       line;
  logic [15:0] v_sq, v_el;
  logic [31:0] v_cs, v_qs, got;
  logic [WALK_SC_AW-1:0] idx;

  initial begin
    if (!$value$plusargs("vectors=%s", vec_file)) begin
      $display("FAIL: +vectors=<file> required"); $finish;
    end
    n_cs = 0; n_qs = 0; n_err = 0; n_vec = 0; cfgd_seen = 0; idx = '0;
    snp_valid = 1'b0; snp_last = 1'b0; snp_flush = 1'b0; sq_valid = 1'b0;
    req_valid = 1'b0; res_ready = 1'b0; snp_data = '0; snp_addr = '0;
    sq_data = '0; req_is_qs = 1'b0; req_idx = '0;

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
        if (cfgd_seen != int'(CFG_D)) begin
          $display("FAIL: vector file is for D=%0d, build is D=%0d",
                   cfgd_seen, CFG_D);
          n_err++;
          break;
        end
        continue;
      end
      if (line.substr(0, 1) != "V ") continue;

      void'($sscanf(line, "V %h %h %h %h", v_sq, v_el, v_cs, v_qs));
      n_vec++;

      // s_q, then the record whose scale becomes s_k / s_v
      @(negedge clk); sq_valid = 1'b1; sq_data = v_sq;
      @(negedge clk); sq_valid = 1'b0;
      push_uniform(idx, v_el);
      // 2026-08-14 A0 write-cone pipeline (seq_walker_comp W1/W2/W3): a
      // commit lands 3 edges after the last snoop beat, so give the pipe
      // that long (+margin) before asking. This models the tile, where the
      // closest legal commit->request distance is ≥ ~70 cycles (the
      // STOREKV idle-poll / QSTAGE / walk-kick chain — see the unit's
      // pipeline comment); back-to-back was never a tile-reachable pacing.
      repeat (6) @(posedge clk);

      ask(1'b0, idx, got);                       // cs (S-3)
      n_cs++;
      if (got != v_cs) begin
        $display("FAIL cs: s_q=%04h elem=%04h expected %08h got %08h",
                 v_sq, v_el, v_cs, got);
        n_err++;
      end

      ask(1'b1, idx, got);                       // qs (S-4)
      n_qs++;
      if (got != v_qs) begin
        $display("FAIL qs: elem=%04h expected %08h got %08h",
                 v_el, v_qs, got);
        n_err++;
      end

      if (n_err > 20) begin
        $display("FAIL: too many mismatches, stopping early");
        break;
      end
      idx = idx + WALK_SC_AW'(1);                // rotate the cache entry
    end
    $fclose(fd);

    $display("COMP RESULT: D=%0d vectors=%0d cs=%0d qs=%0d errors=%0d",
             CFG_D, n_vec, n_cs, n_qs, n_err);
    if (n_err == 0 && n_vec > 0) $display("COMP PASS");
    else                         $display("COMP FAIL");
    $finish;
  end

  logic u_dbg_sc, u_dbg_sq;
  wire  _unused_dbg = &{1'b0, u_dbg_sc, u_dbg_sq};
endmodule
