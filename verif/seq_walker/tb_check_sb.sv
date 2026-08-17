// tb_check_sb.sv — IB-WALK stage 3: descriptor-check equivalence replay.
//
// Replays gen_check_vectors.py rows through the REAL package functions
// (seq_walker_pkg::walk_desc_check / walk_desc2_check) and compares each
// verdict against the Python mirror's, attached per row. This makes the
// "single source in two places" rule of IB_WALK.md §2.2 a dynamic gate:
// any clause-order, field-extraction, or width divergence between the SV
// and seq_walker_fmt.py fails here — including on the deep clauses
// (geometry products, divisibility, kv_map, resv) that the walk-level TBs
// only reach on their refusal paths.
//
// Pure function evaluation: no DUT state, no clock. Mutation gate 3 runs
// the same replay against pkg mutants (m4_fmtskip, m9_resvskip) — the
// harness must fail on both or it is decorative.
`timescale 1ns/1ps

module tb_check_sb;

  import apex_pkg::*;
  import seq_walker_pkg::*;

  // house TB clock (also keeps the build on the same Verilator timing flow
  // as the rest of the suite); one row is evaluated per edge
  logic clk;
  initial clk = 1'b0;
  always #5 clk = ~clk;

  int    fd, n2, n1, n_errors;
  string vec_file, line;

  initial begin
    if (!$value$plusargs("vectors=%s", vec_file)) begin
      $display("FAIL: +vectors=<file> required"); $finish;
    end
    n2 = 0; n1 = 0; n_errors = 0;
    @(posedge clk);

    fd = $fopen(vec_file, "r");
    if (fd == 0) begin
      $display("FAIL: cannot open %s", vec_file); $finish;
    end

    while ($fgets(line, fd) != 0) begin
      if (line.len() < 2 || line.substr(0, 0) == "#") continue;
      @(posedge clk);
      if (line.substr(0, 1) == "C2") begin
        logic [31:0] w0, w1, w2, w3, w4;
        int unsigned cfgd;
        int exp;
        walk_err_e   got;
        void'($sscanf(line, "C2 %h %h %h %h %h %d %d",
                      w0, w1, w2, w3, w4, cfgd, exp));
        got = walk_desc2_check(w0, w1, w2, w3, w4, cfgd);
        n2++;
        if (int'(got) != exp) begin
          $display("FAIL C2 row %0d: SV=%0d mirror=%0d  words=%08x %08x %08x %08x %08x cfg=%0d",
                   n2, got, exp, w0, w1, w2, w3, w4, cfgd);
          n_errors++;
        end
      end else if (line.substr(0, 1) == "C1") begin
        logic [31:0] g, r, m;
        int unsigned cfgd;
        int exp;
        walk_err_e   got;
        void'($sscanf(line, "C1 %h %h %h %d %d", g, r, m, cfgd, exp));
        got = walk_desc_check(walk_desc_unpack(g, r, m), cfgd);
        n1++;
        if (int'(got) != exp) begin
          $display("FAIL C1 row %0d: SV=%0d mirror=%0d  words=%08x %08x %08x cfg=%0d",
                   n1, got, exp, g, r, m, cfgd);
          n_errors++;
        end
      end
    end
    $fclose(fd);

    $display("CHECKEQ RESULT: c2=%0d c1=%0d errors=%0d", n2, n1, n_errors);
    if (n_errors == 0 && (n2 + n1) > 0) $display("CHECKEQ PASS");
    else                                $display("CHECKEQ FAIL");
    $finish;
  end

endmodule
