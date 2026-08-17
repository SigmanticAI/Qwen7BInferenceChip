// tb_residual_alias.sv — D-016(c) regression: residual_buffer cnt==G index alias.
//
// UNIT-LEVEL (the alias is not reachable through the top: both the upstream and
// the KVQ FSMs cap a group at G tokens — grp_tok_cnt wraps and flush only ever
// emits g<=G — so the top-level regression for this fix is impossible by
// construction; this TB drives the buffer directly, per the V0.1 finding
// "benign as driven — latent hazard", verif/v0/kve/RESULT.md §6).
//
// The hazard: cnt is $clog2(G+1) bits but mem[] needs $clog2(G) index bits.
// Upstream writes mem[cnt] unguarded: on the (G+1)-th write of a group,
// cnt==G truncates to index 0 and OVERWRITES the group's first token, and
// `fill` reads G+1. The vendored copy (rtl/kvq/vendor/residual_buffer.sv)
// drops the overflowing write and saturates cnt at G.
//
// Build with the upstream file + -GEXPECT_BUG=1: the alias must REPRODUCE.
// Build with the vendored file + -GEXPECT_BUG=0: mem[0] intact, fill==G.

`timescale 1ns/1ps

module tb_residual_alias;

  parameter int DIM        = 8;
  parameter int DW         = 16;
  parameter int G          = 64;
  parameter int EXPECT_BUG = 0;
  parameter     MAX_CYCLES = 100_000;
  parameter     CFGNAME    = "resid_alias";

  reg clk; initial begin clk = 1'b0; forever #5 clk = ~clk; end
  reg rst_n;

  reg                     wr_valid;
  reg  [DIM*DW-1:0]       wr_vec;
  reg                     clear;
  wire [$clog2(G+1)-1:0]  fill;
  reg  [$clog2(G)-1:0]    rd_idx;
  wire [DIM*DW-1:0]       rd_vec;

  residual_buffer #(.DIM(DIM), .DW(DW), .G(G)) dut (
    .clk(clk), .rst_n(rst_n),
    .wr_valid(wr_valid), .wr_vec(wr_vec), .clear(clear), .fill(fill),
    .rd_idx(rd_idx), .rd_vec(rd_vec)
  );

  // ---------------- watchdog (MANDATORY) -------------------------------------
  initial begin
    #(MAX_CYCLES * 10);   // 10 ns clock period
    $fatal(1, "WATCHDOG: %s exceeded %0d cycles — DUT or TB hung", CFGNAME, MAX_CYCLES);
  end

  // distinct, index-derived token pattern
  function automatic [DIM*DW-1:0] tokvec(input int i);
    int e;
    logic [DIM*DW-1:0] v;
    begin
      v = '0;
      for (e = 0; e < DIM; e++) v[e*DW +: DW] = 16'(32'h0000_1000 + i*DIM + e);
      tokvec = v;
    end
  endfunction

  int checks = 0, fails = 0;
  task automatic note_fail(input string msg);
    fails++;
    $display("  FAIL: %s", msg);
  endtask

  int i;
  logic [DIM*DW-1:0] mem0;

  initial begin
    rst_n = 0; wr_valid = 0; wr_vec = '0; clear = 0; rd_idx = '0;
    $display("==== tb_residual_alias G=%0d expect_bug=%0d (D-016c: cnt==G write alias)", G, EXPECT_BUG);
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // token 0 with clear (group start), then tokens 1..G-1 -> buffer full
    @(negedge clk); wr_valid = 1; clear = 1; wr_vec = tokvec(0);
    @(negedge clk); clear = 0;
    for (i = 1; i < G; i++) begin
      wr_vec = tokvec(i);
      @(negedge clk);
    end
    wr_valid = 0;
    @(negedge clk);
    checks++;
    if (fill !== ($clog2(G+1))'(G))
      note_fail($sformatf("fill=%0d after exactly G writes, expected %0d", fill, G));

    // THE HAZARD: a (G+1)-th write with cnt==G
    @(negedge clk); wr_valid = 1; wr_vec = tokvec(G);
    @(negedge clk); wr_valid = 0;
    @(negedge clk);

    rd_idx = '0;
    @(negedge clk);     // registered read port (F1.5): one edge to settle
    mem0 = rd_vec;      // read of slot 0

    if (EXPECT_BUG == 1) begin
      if (mem0 === tokvec(G) && fill === ($clog2(G+1))'(G + 1)) begin
        $display("[B] confirmed: overflow write ALIASED mem[0] (slot 0 now holds token %0d) and fill=%0d", G, fill);
        $display("D-016c RESULT [%s]: BUG REPRODUCED — cnt==G write truncates the index onto mem[0]", CFGNAME);
        $finish;
      end else begin
        $fatal(1, "tb_residual_alias [%s]: bug NOT reproduced (mem0=%h fill=%0d) — investigate", CFGNAME, mem0, fill);
      end
    end else begin
      checks++;
      if (mem0 !== tokvec(0))
        note_fail($sformatf("mem[0] clobbered by the overflow write: %h != %h", mem0, tokvec(0)));
      checks++;
      if (fill !== ($clog2(G+1))'(G))
        note_fail($sformatf("fill=%0d after overflow write, expected saturation at %0d", fill, G));
      rd_idx = ($clog2(G))'(G - 1);
      @(negedge clk);   // registered read port (F1.5)
      checks++;
      if (rd_vec !== tokvec(G - 1))
        note_fail($sformatf("mem[G-1] wrong after overflow: %h != %h", rd_vec, tokvec(G - 1)));
      $display("============================================================");
      $display("CONFIG %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
      if (fails == 0) begin
        $display("D-016c RESULT [%s]: PASS — overflow write dropped, mem[0..G-1] and fill intact", CFGNAME);
        $finish;
      end else $fatal(1, "D-016c RESULT [%s]: FAIL", CFGNAME);
    end
  end

endmodule
