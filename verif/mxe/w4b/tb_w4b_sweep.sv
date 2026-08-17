// tb_w4b_sweep.sv — exhaustive operand sweep of the W4B lane arithmetic
// (comb twin w4b_quant_one — the SAME functions the pipeline registers;
// see w4b_fp_pkg.sv header for why the proof carries). Order must match
// gen_sweep.py exactly: for s8 in panel, for code in -8..7, for sg in
// 0x0001..0x7BFF.
`timescale 1ns/1ps
module tb_w4b_sweep;
  byte exp_mem [8 * 16 * 31743];
  logic [15:0] s8_panel [8];
  int fd, n, errors, total;
  logic signed [8:0] got;
  string meta;
  initial begin
    fd = $fopen("build/sweep_expect.bin", "rb");
    if (fd == 0) $fatal(1, "no sweep_expect.bin — run gen_sweep.py");
    n = $fread(exp_mem, fd);
    $fclose(fd);
    if (n != 8 * 16 * 31743) $fatal(1, "short read %0d", n);
    fd = $fopen("build/sweep_meta.txt", "r");
    void'($fgets(meta, fd));
    void'($sscanf(meta, "%h %h %h %h %h %h %h %h",
                  s8_panel[0], s8_panel[1], s8_panel[2], s8_panel[3],
                  s8_panel[4], s8_panel[5], s8_panel[6], s8_panel[7]));
    $fclose(fd);
    errors = 0; total = 0;
    for (int s = 0; s < 8; s++)
      for (int c = -8; c < 8; c++)
        for (int g = 1; g < 'h7C00; g++) begin
          got = w4b_fp_pkg::w4b_quant_one(4'(c), 16'(g), s8_panel[s]);
          if (got !== 9'(signed'(exp_mem[total]))) begin
            errors++;
            if (errors <= 5)
              $display("FAIL s8=%04x c=%0d sg=%04x got=%0d exp=%0d",
                       s8_panel[s], c, g, got, exp_mem[total]);
          end
          total++;
        end
    if (errors == 0)
      $display("W4B EXHAUSTIVE SWEEP: %0d operand points, ALL PASS", total);
    else $fatal(1, "W4B SWEEP: %0d/%0d FAIL", errors, total);
    $finish;
  end
endmodule
