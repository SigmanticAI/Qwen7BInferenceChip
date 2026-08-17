// tb_w4b_pkg.sv — package-level gate: w4b_quant_one vs golden expectations
`timescale 1ns/1ps
module tb_w4b_pkg;
  int fd, n, errors, total;
  logic [3:0] c; logic [15:0] sg, s8; logic [8:0] exp9;
  logic signed [8:0] got;
  string line;
  initial begin
    fd = $fopen("build/pkg_vectors.txt", "r");
    if (fd == 0) $fatal(1, "no vectors");
    errors = 0; total = 0;
    while ($fgets(line, fd) > 0) begin
      n = $sscanf(line, "%h %h %h %h", c, sg, s8, exp9);
      if (n != 4) continue;
      got = w4b_fp_pkg::w4b_quant_one(signed'(c), sg, s8);
      if (got !== signed'(exp9)) begin
        errors++;
        if (errors <= 5)
          $display("FAIL c=%0d sg=%04x s8=%04x got=%0d exp=%0d",
                   signed'(c), sg, s8, got, signed'(exp9));
      end
      total++;
    end
    if (errors == 0) $display("W4B PKG GATE: %0d vectors, ALL PASS", total);
    else $fatal(1, "W4B PKG GATE: %0d/%0d FAIL", errors, total);
    $finish;
  end
endmodule
