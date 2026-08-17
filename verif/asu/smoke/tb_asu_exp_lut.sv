// tb_asu_exp_lut.sv — EXHAUSTIVE-domain check of asu_exp_lut vs the golden
// arbiter apex_golden.compute.exp_fx (Verilator --binary --timing --assert).
//
// The oracle build/exp_expected.hex (from gen_asu_vectors.py) holds
// exp_fx(clamp(z)) for ALL 65536 16-bit input patterns, so this covers:
//   * the full contract domain [-16384, 0] bit-exactly (D-014 obligation
//     "verified exhaustively over the input domain vs NumPy"),
//   * the below-domain clamp (z < -16384 -> floor entry), and
//   * the above-domain clamp (z > 0 -> 32768) —
// every pattern a 16-bit input port can carry. Bit-exact (===), no tolerance.
//
// The DUT is combinational; a time-based $fatal watchdog bounds the run.

module tb_asu_exp_lut;

  logic signed [15:0] z_q;
  logic        [15:0] exp_q;

  asu_exp_lut dut (
    .z_q   (z_q),
    .exp_q (exp_q)
  );

  logic [15:0] expected [0:65535];
  int errors;

  // ── watchdog (combinational TB: time-based) ───────────────────────────────
  initial begin
    #5_000_000; // 5 ms; the sweep needs ~65.6 us
    $fatal(1, "WATCHDOG: exp LUT sweep did not finish in time");
  end

  // ── exhaustive sweep ──────────────────────────────────────────────────────
  initial begin
    errors = 0;
    $readmemh("build/exp_expected.hex", expected);

    for (int i = 0; i < 65536; i++) begin
      z_q = 16'(i);
      #1;
      if (exp_q !== expected[i]) begin
        errors++;
        if (errors <= 20) begin
          $display("FAIL: z_q=%0d (0x%04x) got %0d exp %0d",
                   $signed(z_q), 16'(i), exp_q, expected[i]);
        end
      end
    end

    // spot-assert the documented boundary semantics (redundant with the
    // sweep, but makes the log self-explaining)
    z_q = 16'sd0;      #1;
    if (exp_q !== 16'd32768) begin
      errors++; $display("FAIL: exp(0) != 32768 (got %0d)", exp_q);
    end
    z_q = -16'sd16384; #1;
    if (exp_q !== expected[16'(-16'sd16384)]) begin
      errors++; $display("FAIL: floor-entry mismatch");
    end

    if (errors == 0) begin
      $display("ASU EXP LUT SMOKE: ALL TESTS PASSED (t=%0t)", $time);
      $display("  65536/65536 input patterns bit-exact vs apex_golden exp_fx");
      $display("  (full Q6.10 domain + both clamp directions)");
      $finish;
    end else begin
      $fatal(1, "ASU EXP LUT SMOKE: %0d MISMATCH(ES)", errors);
    end
  end

endmodule
