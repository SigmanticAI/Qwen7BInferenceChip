// tb_residual_add.sv — bit-exact check of residual_add (the C-6 fp16 residual
// primitive y = f16(a+b)) vs the golden numpy fp16 add (gen_residual_vectors.py).
// Sims: Verilator (--binary --timing --assert) PRIMARY; Icarus SECONDARY.
//
// The DUT is combinational; each vector is applied and compared === with no
// tolerance. A time-based $fatal watchdog bounds the run. Two spot invariants
// (a+0 == a for a representative value; +inf saturation) make the log self-
// explaining beyond the swept oracle.

module tb_residual_add;

  logic [15:0] a, b, y;

  residual_add dut (.a(a), .b(b), .y(y));

  `include "res_nvec.svh"   // localparam int RES_NVEC

  logic [15:0] v_a   [0:RES_NVEC-1];
  logic [15:0] v_b   [0:RES_NVEC-1];
  logic [15:0] v_exp [0:RES_NVEC-1];

  int errors;

  initial begin
    #20_000_000;
    $fatal(1, "WATCHDOG: residual_add sweep did not finish in time");
  end

  initial begin
    errors = 0;
    $readmemh("build/res_a.hex",   v_a);
    $readmemh("build/res_b.hex",   v_b);
    $readmemh("build/res_exp.hex", v_exp);

    for (int k = 0; k < RES_NVEC; k++) begin
      a = v_a[k];
      b = v_b[k];
      #1;
      if (y !== v_exp[k]) begin
        errors++;
        if (errors <= 20)
          $display("FAIL k=%0d a=%04x b=%04x : got %04x exp %04x",
                   k, a, b, y, v_exp[k]);
      end
    end

    // ── spot invariants ──────────────────────────────────────────────────────
    a = 16'h4200; b = 16'h0000; #1;                 // 3.0 + 0 == 3.0
    if (y !== 16'h4200) begin
      errors++; $display("FAIL: a+0 != a (got %04x)", y);
    end
    a = 16'h7BFF; b = 16'h7BFF; #1;                 // max + max -> +inf
    if (y !== 16'h7C00) begin
      errors++; $display("FAIL: overflow not +inf (got %04x)", y);
    end

    if (errors == 0) begin
      $display("RESIDUAL_ADD SMOKE: ALL TESTS PASSED (t=%0t)", $time);
      $display("  %0d/%0d fp16 residual-add vectors bit-exact vs numpy fp16",
               RES_NVEC, RES_NVEC);
    end else begin
      $display("RESIDUAL_ADD SMOKE: %0d FAILURES", errors);
      $fatal(1, "residual_add mismatch");
    end
    $finish;
  end

endmodule
