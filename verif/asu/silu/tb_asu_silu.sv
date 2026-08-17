// tb_asu_silu.sv — EXHAUSTIVE-domain check of asu_silu vs the golden arbiter
// apex_golden.transformer.silu_fx (Verilator --binary --timing --assert;
// Icarus secondary).
//
// The oracle build/silu_expected.hex (from gen_silu_vectors.py) holds
// silu_fx(clamp(x)) for ALL 65536 16-bit input patterns, so this covers:
//   * the full contract domain [-8192, +8192] bit-exactly (C-SILU),
//   * the below-domain clamp (x < -8192 -> floor entry), and
//   * the above-domain clamp (x > +8192 -> idx=255/frac=64) —
// every pattern a signed 16-bit input port can carry. Bit-exact (===), no
// tolerance. The DUT is combinational; a time-based $fatal watchdog bounds it.

module tb_asu_silu;

  logic signed [15:0] x_q;
  logic signed [15:0] y_q;

  asu_silu dut (
    .x_q (x_q),
    .y_q (y_q)
  );

  logic [15:0] expected [0:65535];   // signed Q4.12, two's complement pattern
  int errors;

  // ── watchdog (combinational TB: time-based) ───────────────────────────────
  initial begin
    #5_000_000; // 5 ms; the sweep needs ~65.6 us
    $fatal(1, "WATCHDOG: silu LUT sweep did not finish in time");
  end

  // ── exhaustive sweep ──────────────────────────────────────────────────────
  initial begin
    errors = 0;
    $readmemh("build/silu_expected.hex", expected);

    for (int i = 0; i < 65536; i++) begin
      x_q = 16'(i);
      #1;
      if (y_q !== $signed(expected[i])) begin
        errors++;
        if (errors <= 20) begin
          $display("FAIL: x_q=%0d (0x%04x) got %0d exp %0d",
                   $signed(x_q), 16'(i), y_q, $signed(expected[i]));
        end
      end
    end

    // spot-assert the documented boundary semantics (redundant with the
    // sweep, but makes the log self-explaining)
    x_q = 16'sd0;       #1;
    if (y_q !== 16'sd0) begin
      errors++; $display("FAIL: silu(0) != 0 (got %0d)", y_q);
    end
    x_q = 16'sd8192;    #1;   // upper clamp -> BASE[255]+SLOPE[255] = 32757
    if (y_q !== 16'sd32757) begin
      errors++; $display("FAIL: silu(8192) != 32757 (got %0d)", y_q);
    end
    x_q = -16'sd8192;   #1;   // lower clamp -> floor entry
    // address = unsigned bit pattern of -8192 = 0xE000 (57344). Written as the
    // literal so the index is unsigned in BOTH Verilator and Icarus.
    if (y_q !== $signed(expected[16'hE000])) begin
      errors++; $display("FAIL: floor-entry mismatch");
    end
    x_q = 16'sh7FFF;    #1;   // far above domain -> same as +8192 clamp
    if (y_q !== 16'sd32757) begin
      errors++; $display("FAIL: above-domain clamp != 32757 (got %0d)", y_q);
    end
    x_q = 16'sh8000;    #1;   // far below domain -> same as -8192 clamp
    if (y_q !== $signed(expected[16'hE000])) begin  // 0xE000 = -8192 pattern
      errors++; $display("FAIL: below-domain clamp mismatch");
    end

    if (errors == 0) begin
      $display("ASU SILU SMOKE: ALL TESTS PASSED (t=%0t)", $time);
      $display("  65536/65536 input patterns bit-exact vs apex_golden silu_fx");
      $display("  (full Q5.10 domain + both clamp directions, signed Q4.12)");
      $finish;
    end else begin
      $fatal(1, "ASU SILU SMOKE: %0d MISMATCH(ES)", errors);
    end
  end

endmodule
