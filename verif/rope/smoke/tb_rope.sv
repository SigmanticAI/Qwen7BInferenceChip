// tb_rope.sv — bit-exact check of rope vs the golden arbiter
// apex_golden.transformer.rope_fx (Verilator --binary --timing --assert
// PRIMARY; Icarus SECONDARY).
//
// The oracle (from gen_rope_vectors.py) holds, for every HF half-split channel
// pair (i, i+half) swept over head_dim in {64,128} and position m in 0..127:
//   u_q    Q0.14 phase code    x_i / x_ih  fp16 inputs
//   exp_i / exp_ih             fp16 rope_fx outputs
// The DUT rotates each pair and must match exp_i/exp_ih EXACTLY (===), no
// tolerance. The DUT is combinational; a time-based $fatal watchdog bounds it.
//
// Two spot invariants beyond the sweep make the log self-explaining:
//   * u_q = 0 (phi = 0)  -> identity: y_i == x_i, y_ih == x_ih
//     (cos_fx(0)=16384=1.0, sin_fx(0)=0, and f16(x*1 - 0) == x on the grid).
//   * u_q = 2^12 (quarter turn) -> y_i == f16(-x_ih), y_ih == f16(x_i)
//     (cos_fx(1/4)=0, sin_fx(1/4)=16384=1.0).

module tb_rope;

  logic [13:0] u_q;
  logic [15:0] x_i, x_ih, y_i, y_ih;

  rope dut (
    .u_q  (u_q),
    .x_i  (x_i),
    .x_ih (x_ih),
    .y_i  (y_i),
    .y_ih (y_ih)
  );

  `include "rope_nvec.svh"   // localparam int ROPE_NVEC

  logic [13:0] v_u_q  [0:ROPE_NVEC-1];
  logic [15:0] v_x_i  [0:ROPE_NVEC-1];
  logic [15:0] v_x_ih [0:ROPE_NVEC-1];
  logic [15:0] v_e_i  [0:ROPE_NVEC-1];
  logic [15:0] v_e_ih [0:ROPE_NVEC-1];

  int errors;

  // ── watchdog (combinational TB: time-based) ────────────────────────────────
  initial begin
    #20_000_000; // 20 ms; the sweep needs ~ROPE_NVEC us
    $fatal(1, "WATCHDOG: rope sweep did not finish in time");
  end

  // ── swept check ────────────────────────────────────────────────────────────
  initial begin
    errors = 0;
    $readmemh("build/rope_u_q.hex",    v_u_q);
    $readmemh("build/rope_x_i.hex",    v_x_i);
    $readmemh("build/rope_x_ih.hex",   v_x_ih);
    $readmemh("build/rope_exp_i.hex",  v_e_i);
    $readmemh("build/rope_exp_ih.hex", v_e_ih);

    for (int k = 0; k < ROPE_NVEC; k++) begin
      u_q  = v_u_q[k];
      x_i  = v_x_i[k];
      x_ih = v_x_ih[k];
      #1;
      if (y_i !== v_e_i[k] || y_ih !== v_e_ih[k]) begin
        errors++;
        if (errors <= 20) begin
          $display("FAIL k=%0d u_q=%0d x_i=%04x x_ih=%04x : got (%04x,%04x) exp (%04x,%04x)",
                   k, u_q, x_i, x_ih, y_i, y_ih, v_e_i[k], v_e_ih[k]);
        end
      end
    end

    // ── spot invariants ──────────────────────────────────────────────────────
    // phi = 0 -> identity (use a representative fp16 pair)
    u_q = 14'd0; x_i = 16'h4200; x_ih = 16'hC500; #1;   // 3.0, -5.0
    if (y_i !== 16'h4200 || y_ih !== 16'hC500) begin
      errors++; $display("FAIL: u_q=0 not identity: (%04x,%04x)", y_i, y_ih);
    end
    // phi = quarter turn (u_q = 2^12) -> y_i = f16(-x_ih), y_ih = f16(x_i)
    u_q = 14'h1000; x_i = 16'h4200; x_ih = 16'h4500; #1; // 3.0, 5.0
    if (y_i !== 16'hC500 || y_ih !== 16'h4200) begin
      errors++; $display("FAIL: u_q=1/4turn wrong: (%04x,%04x)", y_i, y_ih);
    end

    if (errors == 0) begin
      $display("ROPE SMOKE: ALL TESTS PASSED (t=%0t)", $time);
      $display("  %0d/%0d channel-pair vectors bit-exact vs apex_golden rope_fx",
               ROPE_NVEC, ROPE_NVEC);
      $display("  (head_dim in {64,128}, position m in 0..127, HF half-split)");
      $finish;
    end else begin
      $fatal(1, "ROPE SMOKE: %0d MISMATCH(ES)", errors);
    end
  end

endmodule
