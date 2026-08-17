// tb_resfx.sv — residual_add_fx vs (a) golden-fp expected vectors and
// (b) the behavioral rtl/misc/residual_add.sv, co-simulated on EVERY vector.
//
// Part 1 (file): build/vectors_resfx.txt lines "a b exp" (hex fp16). The
//   integer DUT must match exp AND the behavioral block bit-for-bit.
// Part 2 (in-sim): +insim=<N> random finite pairs (xorshift32, seeded by
//   +seed) with corner-density forcing (subnormal / max-exponent / zero
//   classes), integer DUT vs behavioral bit-for-bit. This is the mass
//   cross-check; the file part carries the golden anchors and the
//   engineered classes.
//
// Domain: finite fp16 only (inf/NaN never driven — out of the C-6 bus
// contract, and the behavioral block's double math is documented
// "defensive, unused" there).
//
// Exit: "RESFX RESULT: file=<n> insim=<m> mismatches=0 -> PASS" + rc 0,
// else $fatal (nonzero) — the Makefile gates on both.

module tb_resfx;

  logic [15:0] a, b;
  logic [15:0] y_fx, y_beh;

  residual_add_fx u_fx  (.a (a), .b (b), .y (y_fx));
  residual_add    u_beh (.a (a), .b (b), .y (y_beh));

  integer      n_file, n_insim, n_mism;
  integer      fh, code;
  logic [15:0] va, vb, ve;
  logic [31:0] rng;
  string       vfile;
  integer      insim_n;

  function automatic logic [31:0] xorshift(input logic [31:0] x);
    logic [31:0] r;
    begin
      r = x ^ (x << 13);
      r = r ^ (r >> 17);
      r = r ^ (r << 5);
      return r;
    end
  endfunction

  // finite-forcing: collapse e=31 draws into legal space
  function automatic logic [15:0] mk_finite(input logic [15:0] raw);
    logic [15:0] v;
    begin
      v = raw;
      if (v[14:10] == 5'h1F) v[14:10] = {1'b0, v[13:10]};
      return v;
    end
  endfunction

  task automatic check_one(input logic [15:0] xa, xb,
                           input logic       has_exp,
                           input logic [15:0] xe);
    begin
      a = xa;
      b = xb;
      #1;
      if (y_fx !== y_beh) begin
        n_mism++;
        $display("MISMATCH beh a=%04x b=%04x fx=%04x beh=%04x",
                 xa, xb, y_fx, y_beh);
      end
      if (has_exp && (y_fx !== xe)) begin
        n_mism++;
        $display("MISMATCH exp a=%04x b=%04x fx=%04x exp=%04x",
                 xa, xb, y_fx, xe);
      end
      if (n_mism > 20) $fatal(1, "RESFX FAIL: too many mismatches");
    end
  endtask

  initial begin
    n_file  = 0;
    n_insim = 0;
    n_mism  = 0;
    if (!$value$plusargs("vectors=%s", vfile)) vfile = "build/vectors_resfx.txt";
    if (!$value$plusargs("insim=%d", insim_n)) insim_n = 2_000_000;
    if (!$value$plusargs("seed=%d", rng)) rng = 32'hC61D_A7E5;

    // ── part 1: golden-fp expected file ────────────────────────────────────
    fh = $fopen(vfile, "r");
    if (fh == 0) $fatal(1, "RESFX FAIL: cannot open %s", vfile);
    while (!$feof(fh)) begin
      code = $fscanf(fh, "%h %h %h\n", va, vb, ve);
      if (code == 3) begin
        check_one(va, vb, 1'b1, ve);
        n_file++;
      end
    end
    $fclose(fh);
    if (n_file < 100_000)
      $fatal(1, "RESFX FAIL: suspiciously few file vectors (%0d)", n_file);

    // ── part 2: in-sim mass sweep vs the behavioral reference ──────────────
    for (int i = 0; i < insim_n; i++) begin
      logic [15:0] xa, xb;
      rng = xorshift(rng);
      xa  = mk_finite(rng[15:0]);
      rng = xorshift(rng);
      xb  = mk_finite(rng[15:0]);
      // corner densification
      if ((i & 15) == 0)  xa = {xa[15], 5'd0, xa[9:0]};        // subnormal
      if ((i & 15) == 1)  xb = {xb[15], 5'd0, xb[9:0]};
      if ((i & 31) == 2)  xa = {xa[15], 5'd30, xa[9:0]};       // near-max
      if ((i & 31) == 3)  xb = {xb[15], 5'd30, xb[9:0]};
      if ((i & 63) == 4)  xa = {xa[15], 15'd0};                // signed zero
      if ((i & 63) == 5)  xb = {xb[15], 15'd0};
      if ((i & 63) == 6)  xb = {~xa[15], xa[14:0]};            // exact cancel
      check_one(xa, xb, 1'b0, 16'h0);
      n_insim++;
    end

    if (n_mism == 0) begin
      $display("RESFX RESULT: file=%0d insim=%0d mismatches=0 -> PASS",
               n_file, n_insim);
      $display("RESFX PASS");
      $finish;
    end else begin
      $fatal(1, "RESFX FAIL: %0d mismatches", n_mism);
    end
  end

  initial begin
    #400_000_000;   // watchdog
    $fatal(1, "RESFX FAIL: watchdog");
  end

endmodule
