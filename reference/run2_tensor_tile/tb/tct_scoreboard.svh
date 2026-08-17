// =============================================================================
// tct_scoreboard.svh
// SELF-CHECKING SCOREBOARD + BIT-EXACT GOLDEN REFERENCE MODEL for
// tensor_core_tile (INT8 output-stationary systolic GEMM tile).
// Owner: Scoreboard/Checker Agent. NON-UVM. `included into
// tb_tensor_core_tile module scope, BEFORE tct_stimulus.svh.
//
// This file is the SINGLE POINT OF TRUTH for:
//   - pass_count / fail_count (declared exactly once, here)
//   - the golden/reference model (shadow weight memory, shadow activation
//     matrix, shadow 32-bit wrapping accumulators, golden requant+clip)
//   - all expected-vs-actual compare logic and mismatch reporting, tied to
//     requirement IDs.
//
// Requirement traceability for the compares implemented in this file:
//   REQ-GEMM-001/002/003/004 : NxN output-stationary MAC, signed 8x8
//                               multiply, sign-extended add, 32-bit wrap
//                               accumulation -- modeled bit-exact below.
//   ERR-001                  : accumulator overflow wraps at ACC_W=32 bits,
//                               two's-complement -- modeled by truncating
//                               the full-precision sum to the low 32 bits.
//   REQ-GEMM-005/006         : requantization (signed acc * unsigned scale,
//                               round-to-nearest, arithmetic right shift)
//                               and clip to signed INT8 -- modeled bit-exact
//                               in sb_requant() below, matching
//                               rtl/requant_unit.v algebraically.
//   ERR-002                  : per-lane saturation -> sticky overflow.
//   REQ-PROT-003             : y_out_last exactly on the final drain row
//                               (row_idx == N-1).
//   REQ-GEMM-014             : per-beat compare of the full drained output
//                               row against the golden row.
//
// Do NOT weaken these checks to make a test pass. Do NOT invent expected
// behavior beyond what is specified in spec.json / the RTL contract that
// spec.json resolves (RESOLVED-003, RESOLVED-006, RESOLVED-007,
// RESOLVED-010, RESOLVED-011).
// =============================================================================

// ---------------------------------------------------------------------------
// Scoreboard state -- SINGLE POINT OF TRUTH for pass/fail counters.
// Do NOT redeclare pass_count/fail_count anywhere else (tb harness and
// stimulus both reference these by name).
// ---------------------------------------------------------------------------
integer pass_count;
integer fail_count;

// ---------------------------------------------------------------------------
// Golden/reference-model storage.
//   sb_w[0:63]      : shadow weight buffer, row-major linear addr = k*N+j
//                      (matches rtl/weight_buffer.v write-address mapping).
//   sb_a[k][i]      : shadow activation matrix, beat k, element i (matches
//                      pe_array.v a_vec[i] broadcast on beat k).
//   sb_acc[i][j]    : golden accumulator (signed 32-bit, post-wrap) for
//                      output row i, column j -- mirrors pe.v's acc after
//                      all N=8 MAC beats (REQ-GEMM-001..004, ERR-001).
//   sb_scale/shift  : latched requant config (REQ-GEMM-005/006).
//   sb_y_exp[i]     : golden packed output row word (N*DW_OUT = 64 bits),
//                      lane j at bits [j*8 +: 8], matching y_out_data.
//   sb_sticky_exp   : expected sticky overflow accumulated since the last
//                      sb_op_start() (mirrors overflow_r in
//                      rtl/tensor_core_tile.v, cleared on start&&!busy).
// ---------------------------------------------------------------------------
byte signed          sb_w   [0:63];
byte signed          sb_a   [0:7][0:7];
logic signed [31:0]  sb_acc [0:7][0:7];
logic        [15:0]  sb_scale;
logic        [4:0]   sb_shift;
logic        [63:0]  sb_y_exp [0:7];
bit                  sb_sticky_exp;

initial begin
  integer sb_init_i, sb_init_j;
  pass_count    = 0;
  fail_count    = 0;
  sb_scale      = '0;
  sb_shift      = '0;
  sb_sticky_exp = 1'b0;
  for (sb_init_i = 0; sb_init_i < 64; sb_init_i = sb_init_i + 1) begin
    sb_w[sb_init_i] = 8'sd0;
  end
  for (sb_init_i = 0; sb_init_i < 8; sb_init_i = sb_init_i + 1) begin
    sb_y_exp[sb_init_i] = 64'd0;
    for (sb_init_j = 0; sb_init_j < 8; sb_init_j = sb_init_j + 1) begin
      sb_a[sb_init_i][sb_init_j]   = 8'sd0;
      sb_acc[sb_init_i][sb_init_j] = 32'sd0;
    end
  end
end

// ---------------------------------------------------------------------------
// sb_requant : bit-exact golden requantize + clip for one 32-bit signed
// accumulator lane (function, purely combinational -- no timing control),
// matching rtl/requant_unit.v exactly:
//   prod    = signed64(acc32) * unsigned(scale)     (signed*non-negative)
//   round   = (shift>0) ? (1 <<< (shift-1)) : 0
//   shifted = (prod + round) >>> shift              (arithmetic, signed)
//   clip to [-128,127], sat=1 iff clipping occurred
// A 64-bit (longint) signed accumulator is wide enough: |acc32| <= 2^31,
// scale <= 2^16-1, so |prod| < 2^47, well inside signed 64-bit range, and
// the round constant (<= 2^30) cannot push it out of range either.
// Requirement traceability: REQ-GEMM-005, REQ-GEMM-006, ERR-002.
// ---------------------------------------------------------------------------
function automatic void sb_requant(
    input  logic signed [31:0] acc32,
    input  logic        [15:0] scale,
    input  logic        [4:0]  shift,
    output byte  signed        q,
    output bit                 sat
);
  longint signed prod;
  longint signed round_c;
  longint signed shifted;
  begin
    prod = (longint'(acc32)) * (longint'(scale));

    if (shift == 5'd0) begin
      round_c = 64'sd0;
    end else begin
      round_c = (64'sd1 <<< (shift - 5'd1));
    end

    shifted = (prod + round_c) >>> shift;

    if (shifted > 64'sd127) begin
      q   = 8'sd127;
      sat = 1'b1;
    end else if (shifted < -64'sd128) begin
      q   = -8'sd128;
      sat = 1'b1;
    end else begin
      q   = shifted[7:0];
      sat = 1'b0;
    end
  end
endfunction

// ---------------------------------------------------------------------------
// sb_load_weights : shadow-load the weight buffer, row-major linear address
// (addr = k*N + j), mirroring rtl/weight_buffer.v write-address mapping
// driven by ctrl_fsm's load_cnt (RESOLVED-003).
// ---------------------------------------------------------------------------
function automatic void sb_load_weights(input byte signed w[64]);
  integer idx;
  begin
    for (idx = 0; idx < 64; idx = idx + 1) begin
      sb_w[idx] = w[idx];
    end
  end
endfunction

// ---------------------------------------------------------------------------
// sb_set_requant : latch scale (unsigned 16b) / shift (0..31) used by the
// golden requant model for subsequent sb_set_activations() calls.
// ---------------------------------------------------------------------------
function automatic void sb_set_requant(input logic [15:0] scale, input logic [4:0] shift);
  begin
    sb_scale = scale;
    sb_shift = shift;
  end
endfunction

// ---------------------------------------------------------------------------
// sb_op_start : clear expected sticky-overflow accumulation for a new op.
// Mirrors rtl/tensor_core_tile.v: overflow_r cleared on (start && !busy).
// Call this exactly when the stimulus issues a start pulse that the DUT
// will actually accept as a fresh op (idle-qualified), matching Risk-2.
// ---------------------------------------------------------------------------
function automatic void sb_op_start();
  begin
    sb_sticky_exp = 1'b0;
  end
endfunction

// ---------------------------------------------------------------------------
// sb_set_activations : shadow-load the N=8 activation beats (a[k][i], beat
// k, element i), then compute the full golden MAC + requant model:
//   acc[i][j] = wrap32( sum_{k=0..7} A[k][i] * W[k*8+j] )   (signed 8x8,
//               accumulated at full precision then truncated to the low
//               32 bits to reproduce the DUT's 32-bit wrapping
//               accumulator bit-exactly -- REQ-GEMM-001..004, ERR-001)
//   y_exp[i][j] = clip(round((acc[i][j]*scale + round_const) >>> shift))
//               (REQ-GEMM-005/006, ERR-002)
// sb_sticky_exp accumulates (OR's) any lane saturation across the whole
// drained result of this op, matching the DUT's sticky overflow_r which is
// set whenever any lane saturates during any drained beat since the last
// clearing start (REQ-GEMM-005/006, ERR-002, U-007).
// ---------------------------------------------------------------------------
function automatic void sb_set_activations(input byte signed a[8][8]);
  integer i, j, k;
  longint signed acc_temp;
  logic   signed [31:0] acc32;
  byte    signed        qlane;
  bit                   satlane;
  logic   [63:0]        row_word;
  begin
    // shadow-copy activation beats: a[k][i] -> sb_a[k][i]
    for (k = 0; k < 8; k = k + 1) begin
      for (i = 0; i < 8; i = i + 1) begin
        sb_a[k][i] = a[k][i];
      end
    end

    // golden MAC: full-precision accumulate, then truncate to 32 bits
    // (two's-complement wrap) -- equivalent bit-exactly to the DUT's
    // incremental mod-2^32 wrapping accumulator since modular addition
    // is associative/commutative under truncation.
    for (i = 0; i < 8; i = i + 1) begin
      for (j = 0; j < 8; j = j + 1) begin
        acc_temp = 64'sd0;
        for (k = 0; k < 8; k = k + 1) begin
          acc_temp = acc_temp + (longint'(sb_a[k][i]) * longint'(sb_w[k*8+j]));
        end
        acc32          = acc_temp[31:0];
        sb_acc[i][j]   = acc32;
      end
    end

    // golden requant + clip, per row, packing into the expected row word
    for (i = 0; i < 8; i = i + 1) begin
      row_word = 64'd0;
      for (j = 0; j < 8; j = j + 1) begin
        sb_requant(sb_acc[i][j], sb_scale, sb_shift, qlane, satlane);
        row_word[j*8 +: 8] = qlane;
        if (satlane) begin
          sb_sticky_exp = 1'b1;
        end
      end
      sb_y_exp[i] = row_word;
    end
  end
endfunction

// ---------------------------------------------------------------------------
// sb_expected_row : accessor for directed tests wanting the raw golden
// packed output row word without going through the beat checker.
// ---------------------------------------------------------------------------
function automatic logic [63:0] sb_expected_row(input int row_idx);
  begin
    sb_expected_row = sb_y_exp[row_idx];
  end
endfunction

// ---------------------------------------------------------------------------
// sb_check_out_beat : bit-exact compare of one drained output beat against
// the golden model. Compares all 8 INT8 lanes independently (element-wise,
// signed decimal + hex on mismatch) and checks y_out_last against the
// expected drain-row-index contract (last exactly on row_idx==N-1).
// One pass_count/fail_count increment per beat call (a beat either fully
// matches or it doesn't -- lane-level detail is printed but not double
// counted).
//
// Requirement IDs implicated on mismatch:
//   REQ-GEMM-002 : bit-exact MAC-derived output value (lane data mismatch)
//   REQ-GEMM-005 : requant scale/shift arithmetic (lane data mismatch)
//   REQ-GEMM-006 : clip-to-INT8 range (lane data mismatch at saturation)
//   REQ-PROT-003 : y_out_last marker placement
// ---------------------------------------------------------------------------
task automatic sb_check_out_beat(
    input int          row_idx,
    input logic [63:0] dut_y,
    input logic        dut_last,
    input logic        dut_overflow
);
  integer     lane;
  bit         mismatch;
  byte signed exp_lane;
  byte signed act_lane;
  bit         exp_last;
  begin
    mismatch = 1'b0;

    for (lane = 0; lane < 8; lane = lane + 1) begin
      exp_lane = sb_y_exp[row_idx][lane*8 +: 8];
      act_lane = dut_y[lane*8 +: 8];
      if (exp_lane !== act_lane) begin
        mismatch = 1'b1;
        $display("MISMATCH [REQ-GEMM-002/REQ-GEMM-005/REQ-GEMM-006] row=%0d lane=%0d expected=%0d (0x%02h) actual=%0d (0x%02h)",
                  row_idx, lane, exp_lane, exp_lane, act_lane, act_lane);
      end
    end

    exp_last = (row_idx == 7);
    if (dut_last !== exp_last) begin
      mismatch = 1'b1;
      $display("MISMATCH [REQ-PROT-003] row=%0d y_out_last expected=%0b actual=%0b",
                row_idx, exp_last, dut_last);
    end

    if (mismatch) begin
      fail_count = fail_count + 1;
    end else begin
      pass_count = pass_count + 1;
    end

    // dut_overflow is observed here only informationally (per-beat sticky
    // read-back); the authoritative overflow compare happens once per op
    // in sb_check_overflow_final() below -- no separate pass/fail emitted
    // here for it to avoid double-counting the same sticky bit N times.
  end
endtask

// ---------------------------------------------------------------------------
// sb_check_overflow_final : after an op completes (all N rows drained),
// compare the DUT's sticky overflow output against the golden sticky
// overflow accumulated in sb_set_activations() since the last sb_op_start().
// Requirement ID implicated on mismatch: ERR-002 (saturation flag).
// ---------------------------------------------------------------------------
function automatic void sb_check_overflow_final(input logic dut_overflow);
  begin
    if (dut_overflow !== sb_sticky_exp) begin
      fail_count = fail_count + 1;
      $display("MISMATCH [ERR-002] sticky overflow expected=%0b actual=%0b",
                sb_sticky_exp, dut_overflow);
    end else begin
      pass_count = pass_count + 1;
    end
  end
endfunction
