// =============================================================================
// File   : tb/gemm_ref_selftest.sv
// Module : gemm_ref_selftest
// -----------------------------------------------------------------------------
// PURPOSE
//   Standalone Verilator --binary top proving tb/gemm_ref_pkg.sv COMPILES and
//   that its golden model produces the HAND-COMPUTED result for a small,
//   fully worked 1x1x1-tile, N=8 case. No DUT is instantiated (this proves
//   the package compiles/runs in isolation, per the pure-package
//   requirement).
//
// HAND-COMPUTED CASE (worked below and in the $display/assert output):
//   N=8, tiles_m=1, tiles_k=1, tiles_n=1, req_scale=2, req_shift=1.
//
//   Activation tensor: one-hot per k -- act_beats[k].lane[i] = 1 if i==k
//     else 0  (i.e. an 8x8 identity matrix A[i][k]=delta(i,k)).
//   Weight tensor: weight_beats[k].lane[j] = (k+1) for every lane j
//     (row k of weight matrix is constant (k+1) across all columns).
//
//   Since aidx=widx=k_cnt for a single 1x1x1 tile (m_idx=n_idx=k_tile=0):
//     acc[i][j] = sum_{k=0}^{7} A[i][k] * W[k][j]
//               = W[i][j]              (only k=i contributes, A[i][i]=1)
//               = i+1                   (constant across j, by construction)
//
//   Requant per row i, all 8 lanes equal:
//     prod    = (i+1) * req_scale = (i+1)*2 = 2*(i+1)
//     shifted = prod >>> 1 = (2*(i+1)) >>> 1 = (i+1)     (exact, no rounding
//               needed since prod is always even)
//     saturate: i+1 ranges 1..8, well inside [-128,127] -> no saturation.
//   Expected result: row i, beat.data = all 8 lanes == (i+1), for i=0..7.
//   Expected err_overflow = 0 (no saturation). Expected produces_output=1
//   (tiles_m=tiles_k=tiles_n=1 is legal, well under MAX_TILES_*=8 and
//   expected_wwords=expected_awords=8 <= WBUF_DEPTH=ABUF_DEPTH=64).
//
//   Concretely checked by hand below: row i=3 -> acc=4, prod=8, shifted=4
//   (=8>>>1), all 8 lanes must read 0x04. row i=7 -> acc=8, prod=16,
//   shifted=8, all lanes 0x08.
//
// SECOND HAND CHECK (arithmetic-shift FLOOR semantics, REQ-GEMM-006):
//   Independent of the tile above -- proves >>> truncates toward -infinity,
//   not toward zero: -3 >>> 1 must equal -2 (floor(-1.5) = -2), NOT -1
//   (which would be truncate-toward-zero). Checked directly on a longint.
// =============================================================================

`timescale 1ns/1ps

module gemm_ref_selftest;

  import gemm_ref_pkg::*;

  localparam int N = 8;

  ref_desc_t desc;
  ref_beat_t weight_beats[$];
  ref_beat_t act_beats[$];
  ref_beat_t expected_beats[$];
  bit        produces_output;
  bit        expected_ovf;
  int        eff_w, eff_a;
  int        errors;

  byte lanes[REF_MAX_N];

  int k, i, j;
  longint signed floor_check;

  initial begin
    errors = 0;

    // -------------------------------------------------------------------
    // Build the hand-worked 1x1x1 case described above.
    // -------------------------------------------------------------------
    desc.tiles_m   = 1;
    desc.tiles_k   = 1;
    desc.tiles_n   = 1;
    desc.req_scale = 2;
    desc.req_shift = 1;

    for (k = 0; k < N; k++) begin
      // weight word k: every lane j = (k+1)
      for (j = 0; j < N; j++) lanes[j] = byte'(k+1);
      begin
        ref_beat_t wb;
        wb.data = ref_pack_lanes(lanes, N);
        wb.last = (k == N-1);
        weight_beats.push_back(wb);
      end

      // activation word k: one-hot, lane i = (i==k) ? 1 : 0
      for (i = 0; i < N; i++) lanes[i] = (i == k) ? byte'(1) : byte'(0);
      begin
        ref_beat_t ab;
        ab.data = ref_pack_lanes(lanes, N);
        ab.last = (k == N-1);
        act_beats.push_back(ab);
      end
    end

    eff_w = ref_eff_words(weight_beats, ref_expected_wwords(desc.tiles_k, desc.tiles_n, N));
    eff_a = ref_eff_words(act_beats,    ref_expected_awords(desc.tiles_m, desc.tiles_k, N));

    $display("SELFTEST: expected_wwords=%0d expected_awords=%0d eff_w=%0d eff_a=%0d",
              ref_expected_wwords(desc.tiles_k, desc.tiles_n, N),
              ref_expected_awords(desc.tiles_m, desc.tiles_k, N), eff_w, eff_a);

    if (eff_w != 8 || eff_a != 8) begin
      $display("SELFTEST FAIL: eff word clamp wrong (expected 8/8, got %0d/%0d)", eff_w, eff_a);
      errors++;
    end

    ref_compute_expected(desc, weight_beats, act_beats, eff_w, eff_a,
                          expected_beats, produces_output, expected_ovf, N);

    $display("SELFTEST: produces_output=%0b expected_ovf=%0b beats=%0d",
              produces_output, expected_ovf, expected_beats.size());

    if (produces_output !== 1'b1) begin
      $display("SELFTEST FAIL: expected produces_output=1 for legal 1x1x1 descriptor");
      errors++;
    end
    if (expected_ovf !== 1'b0) begin
      $display("SELFTEST FAIL: expected err_overflow=0 (no saturation in this case)");
      errors++;
    end
    if (expected_beats.size() != N) begin
      $display("SELFTEST FAIL: expected %0d result beats (one per row), got %0d", N, expected_beats.size());
      errors++;
    end

    // ---------------------------------------------------------------
    // Hand-checked value #1: row i=3 -> every lane must equal 4 (0x04).
    //   acc[3][j] = 4 ; prod = 4*2 = 8 ; shifted = 8>>>1 = 4.
    // ---------------------------------------------------------------
    for (j = 0; j < N; j++) begin
      byte got;
      got = ref_lane(expected_beats[3], j);
      $display("SELFTEST: row=3 lane=%0d expected=4 got=%0d", j, got);
      if (got !== byte'(4)) begin
        $display("SELFTEST FAIL: row 3 lane %0d expected 4 got %0d", j, got);
        errors++;
      end
    end
    if (expected_beats[3].last !== 1'b0) begin
      $display("SELFTEST FAIL: row 3 should not carry last (only row 7 of the single/final tile does)");
      errors++;
    end

    // ---------------------------------------------------------------
    // Hand-checked value #2: row i=7 (last row of the only/last tile) ->
    //   every lane must equal 8 (0x08), and .last must be asserted
    //   (ASM-006/UQ-008: final row of final tile).
    //   acc[7][j] = 8 ; prod = 8*2 = 16 ; shifted = 16>>>1 = 8.
    // ---------------------------------------------------------------
    for (j = 0; j < N; j++) begin
      byte got;
      got = ref_lane(expected_beats[7], j);
      $display("SELFTEST: row=7 lane=%0d expected=8 got=%0d", j, got);
      if (got !== byte'(8)) begin
        $display("SELFTEST FAIL: row 7 lane %0d expected 8 got %0d", j, got);
        errors++;
      end
    end
    if (expected_beats[7].last !== 1'b1) begin
      $display("SELFTEST FAIL: row 7 (final row of final tile) must have last=1");
      errors++;
    end

    // ---------------------------------------------------------------
    // Hand check #3: arithmetic-shift FLOOR semantics (independent of the
    // tile math above). -3 >>> 1 must be -2 (floor(-1.5)), never -1.
    // ---------------------------------------------------------------
    floor_check = longint'(-3) >>> 1;
    $display("SELFTEST: floor-shift check: -3 >>> 1 = %0d (expect -2)", floor_check);
    if (floor_check != -2) begin
      $display("SELFTEST FAIL: arithmetic shift floor semantics wrong: got %0d expected -2", floor_check);
      errors++;
    end

    // ---------------------------------------------------------------
    // Hand check #4: illegal-descriptor predicate (ERR-002/UQ-003).
    // tiles_k=8,tiles_n=8,N=8 -> expected_wwords = 8*8*8=512 > WBUF_DEPTH(64)
    // -> illegal, must yield produces_output=0, no beats, ovf=0.
    // ---------------------------------------------------------------
    begin
      ref_desc_t bad_desc;
      ref_beat_t empty_q[$];
      ref_beat_t bad_expected[$];
      bit        bad_po;
      bit        bad_ovf;
      bad_desc.tiles_m   = 8;
      bad_desc.tiles_k   = 8;
      bad_desc.tiles_n   = 8;
      bad_desc.req_scale = 1;
      bad_desc.req_shift = 0;
      ref_compute_expected(bad_desc, empty_q, empty_q, 0, 0,
                            bad_expected, bad_po, bad_ovf, N);
      $display("SELFTEST: illegal-descriptor case: produces_output=%0b beats=%0d ovf=%0b",
                bad_po, bad_expected.size(), bad_ovf);
      if (bad_po !== 1'b0 || bad_expected.size() != 0 || bad_ovf !== 1'b0) begin
        $display("SELFTEST FAIL: illegal descriptor must yield produces_output=0, 0 beats, ovf=0");
        errors++;
      end
    end

    // ---------------------------------------------------------------
    // Hand check #5: N=16 genericity smoke check. ref_compute_expected /
    // ref_eff_words / ref_is_illegal all take N as a runtime argument
    // (not a compile-time package parameter), so the SAME compiled code
    // must work for N=16. tiles_m=k=n=1, weight word: lane j=(j+1),
    // activation word: lane i=1 for all i (all-ones row). Then
    // acc[i][j] = sum_k A[i][k]*W[k][j] ; with tiles_k=1 only k=0 exists,
    // A[i][0]=1 for all i, W[0][j]=(j+1) -> acc[i][j]=(j+1) for every row i.
    // req_scale=1,req_shift=0 -> shifted=acc directly (no scale/shift),
    // so row 0, lane 15 must equal 16 (0x10).
    // ---------------------------------------------------------------
    begin
      localparam int N16 = 16;
      ref_desc_t   d16;
      ref_beat_t   w16[$], a16[$], exp16[$];
      bit          po16, ovf16;
      byte         lanes16[REF_MAX_N];
      int          ew16, ea16;
      byte         got16;

      d16.tiles_m = 1; d16.tiles_k = 1; d16.tiles_n = 1;
      d16.req_scale = 1; d16.req_shift = 0;

      for (j = 0; j < N16; j++) lanes16[j] = byte'(j+1);
      begin
        ref_beat_t wb16;
        wb16.data = ref_pack_lanes(lanes16, N16);
        wb16.last = 1'b1;
        w16.push_back(wb16);
      end
      for (i = 0; i < N16; i++) lanes16[i] = byte'(1);
      begin
        ref_beat_t ab16;
        ab16.data = ref_pack_lanes(lanes16, N16);
        ab16.last = 1'b1;
        a16.push_back(ab16);
      end

      ew16 = ref_eff_words(w16, ref_expected_wwords(d16.tiles_k, d16.tiles_n, N16));
      ea16 = ref_eff_words(a16, ref_expected_awords(d16.tiles_m, d16.tiles_k, N16));
      ref_compute_expected(d16, w16, a16, ew16, ea16, exp16, po16, ovf16, N16);

      got16 = ref_lane(exp16[0], 15);
      $display("SELFTEST: N=16 case: produces_output=%0b beats=%0d row0 lane15 expected=16 got=%0d",
                po16, exp16.size(), got16);
      if (po16 !== 1'b1 || exp16.size() != N16 || got16 !== byte'(16)) begin
        $display("SELFTEST FAIL: N=16 genericity case mismatch");
        errors++;
      end
    end

    if (errors == 0) begin
      $display("SELFTEST_RESULT: PASS (all hand-checked values matched)");
      $finish;
    end else begin
      $display("SELFTEST_RESULT: FAIL (%0d mismatches)", errors);
      $fatal(1, "gemm_ref_pkg selftest failed");
    end
  end

endmodule : gemm_ref_selftest
