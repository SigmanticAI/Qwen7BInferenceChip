// =============================================================================
// File    : tb/gemm_ref_pkg.sv
// Package : gemm_ref_pkg
// -----------------------------------------------------------------------------
// PURPOSE
//   Bit-exact, PURE (no DUT hierarchical references) SystemVerilog reference
//   model + scoreboard-compare library for gemm_systolic_accel. Compiles as a
//   standalone top-level source (Verilator 5.044, non-UVM). Imported by the
//   testbench / stimulus_sequence specialist, which calls:
//       ref_compute_expected(desc, weight_beats, act_beats, eff_w, eff_a, N)
//   to obtain the golden beat queue for one job, and
//       ref_compare(expected_q, actual_q, N, tag, error_count)
//   to compare it against beats captured from the DUT's monitor.
//
//   This file contains ONLY functions/typedefs. It never references any DUT
//   signal, module, or hierarchy. It does not drive stimulus and does not
//   assemble an env/test.
//
// MATH MODELED (verbatim from rtl/gemm_ctrl.sv, rtl/gemm_pe_array.sv,
//   rtl/gemm_requant.sv, rtl/gemm_systolic_accel.sv -- RTL NOT modified):
//
//   STORAGE/INDEXING (REQ-GEMM-002/003, REQ-STORE-001/002, ERR-003/UQ-004):
//     - weight beats stored linearly wbuf[0],wbuf[1],... ; eff_wwords = beats
//       actually accepted (clamped at w_last or expected_wwords reached).
//     - activation beats stored linearly abuf[0],abuf[1],... ; eff_awords
//       likewise for a_last / expected_awords.
//     - expected_wwords = tiles_k*tiles_n*N ; expected_awords = tiles_m*tiles_k*N.
//     - widx = (n_idx*tiles_k + k_tile)*N + k_cnt
//       aidx = (m_idx*tiles_k + k_tile)*N + k_cnt
//       iterate k_tile=0..tiles_k-1 (outer), k_cnt=0..N-1 (inner).
//     - A MAC step contributes only when widx<eff_wwords AND aidx<eff_awords;
//       otherwise SKIPPED (contributes zero) -- ERR-003/UQ-004.
//
//   MAC (REQ-GEMM-004/005/013, REQ-STORE-003):
//     - acc[i][j] (signed 32-bit, modeled with SV `int`, which naturally
//       wraps mod 2^32 on overflow exactly like the RTL's plain signed
//       32-bit accumulator register) cleared to 0 at the start of each
//       output tile.
//     - For each valid MAC step: acc[i][j] += signed(a_beat.lane[i]) *
//       signed(w_beat.lane[j]) for all i,j (output-stationary broadcast).
//
//   REQUANT (REQ-GEMM-006/007/013, ERR-001):
//     - prod (>=64-bit signed) = signed(acc[i][j]) * signed(req_scale)
//     - shifted = prod >>> req_shift  (arithmetic shift = floor toward -inf)
//     - saturate to signed INT8 [-128,127]; ovf set per lane on saturation.
//     - y beat for row i packs lane j at byte j (col0 = LSByte).
//
//   OUTPUT ORDERING (ASM-006/UQ-008, REQ-GEMM-008/011/012):
//     - tiles emitted m_idx=0..tiles_m-1 (outer), n_idx=0..tiles_n-1 (inner).
//     - N beats per tile, one per row i=0..N-1.
//     - last asserted on row N-1 of the final tile (m=tiles_m-1,n=tiles_n-1).
//     - err_overflow sticky across the whole job: set if ANY lane of ANY beat
//       saturates; illegal descriptors never set it (ERR-002/UQ-003).
//
//   ILLEGAL DESCRIPTOR PREDICATE (ERR-002/UQ-003), EXACT:
//     illegal = tiles_m==0 || tiles_m>MAX_TILES_M ||
//               tiles_k==0 || tiles_k>MAX_TILES_K ||
//               tiles_n==0 || tiles_n>MAX_TILES_N ||
//               (tiles_k*tiles_n*N) > WBUF_DEPTH ||
//               (tiles_m*tiles_k*N) > ABUF_DEPTH
//     illegal => produces_output=0, no beats, done never pulsed,
//                err_overflow stays 0.
//
// REQUIREMENT TRACEABILITY (embedded in $display diagnostics emitted by
//   ref_compare / ref_compare_overflow / ref_compare_produces_output so every
//   mismatch is tagged with the requirement id(s) it violates):
//     REQ-GEMM-001, REQ-GEMM-002, REQ-GEMM-003, REQ-GEMM-004, REQ-GEMM-005,
//     REQ-GEMM-006, REQ-GEMM-007, REQ-GEMM-008, REQ-GEMM-011, REQ-GEMM-012,
//     REQ-GEMM-013, ASM-006, UQ-008, ERR-001, ERR-002, ERR-003, UQ-003, UQ-004.
//
// ORDERING / LATENCY NOTE FOR CALLERS
//   This package models CONTENT and ORDER of the expected beat stream (queue
//   order == required emission order per ASM-006/UQ-008). It does not model
//   wall-clock latency (REQ-TIME-001/002/003 are handshake/timing properties
//   that must be checked by the caller against the monitor's cycle-accurate
//   capture, e.g. verifying done is a single-cycle pulse coincident with the
//   y_last beat's accepted handshake, and that no beat is compared out of
//   the order it was actually accepted). ref_compare performs a strict,
//   INDEX-ALIGNED comparison (expected_q[i] vs actual_q[i]) -- callers MUST
//   feed actual_q in the exact order beats were accepted on y_valid/y_ready,
//   never reordered/sorted, so any DUT reordering bug is caught, not masked.
// =============================================================================

package gemm_ref_pkg;

  // ---------------------------------------------------------------------
  // Sizing (max supported lane count; N is a runtime function argument so
  // both N=8 and N=16 use the SAME compiled code).
  // ---------------------------------------------------------------------
  localparam int REF_MAX_N       = 16;
  localparam int REF_MAX_LANES_W = REF_MAX_N * 8;   // 128 bits

  // Defaults mirroring rtl/gemm_systolic_accel.sv parameter defaults.
  localparam int REF_DEFAULT_N           = 8;
  localparam int REF_DEFAULT_MAX_TILES_M = 8;
  localparam int REF_DEFAULT_MAX_TILES_K = 8;
  localparam int REF_DEFAULT_MAX_TILES_N = 8;
  localparam int REF_DEFAULT_WBUF_DEPTH  = 64;
  localparam int REF_DEFAULT_ABUF_DEPTH  = 64;

  // ---------------------------------------------------------------------
  // Job descriptor (decoded fields only; bit-packing into desc_data is a
  // separate concern owned by gemm_tb_pkg::make_descriptor -- this package
  // never touches desc_data bit layout, only decoded integer fields).
  // ---------------------------------------------------------------------
  typedef struct {
    int tiles_m;
    int tiles_k;
    int tiles_n;
    int req_shift;   // unsigned, SHIFT_W bits wide in RTL (0..63 for SHIFT_W=6)
    int req_scale;   // signed, SCALE_W bits wide in RTL (32-bit signed)
  } ref_desc_t;

  // ---------------------------------------------------------------------
  // Generic N-wide streaming beat (weight / activation / result all share
  // this shape). data holds up to REF_MAX_N lanes of 8 bits; lane i lives
  // at bits [i*8 +: 8] (element 0 = LSByte), matching w_data/a_data/y_data
  // packing in the RTL exactly. Only the low N*8 bits are meaningful for a
  // given N; callers must be consistent about N across a single job.
  // ---------------------------------------------------------------------
  typedef struct {
    logic [REF_MAX_LANES_W-1:0] data;
    bit                         last;
  } ref_beat_t;

  // =====================================================================
  // Lane pack/unpack helpers
  // =====================================================================
  function automatic byte signed ref_lane(input ref_beat_t beat, input int lane);
    return byte'(beat.data[lane*8 +: 8]);
  endfunction

  function automatic logic [REF_MAX_LANES_W-1:0] ref_pack_lanes(input byte lanes[], input int n);
    logic [REF_MAX_LANES_W-1:0] d;
    d = '0;
    for (int i = 0; i < n; i++) d[i*8 +: 8] = lanes[i];
    return d;
  endfunction

  // =====================================================================
  // Expected word-count helpers (REQ-GEMM-002/003)
  // =====================================================================
  function automatic int ref_expected_wwords(input int tiles_k, input int tiles_n, input int n);
    return tiles_k * tiles_n * n;
  endfunction

  function automatic int ref_expected_awords(input int tiles_m, input int tiles_k, input int n);
    return tiles_m * tiles_k * n;
  endfunction

  // =====================================================================
  // ILLEGAL descriptor predicate -- EXACT (ERR-002/UQ-003)
  // =====================================================================
  function automatic bit ref_is_illegal(
      input ref_desc_t desc,
      input int n            = REF_DEFAULT_N,
      input int max_tiles_m  = REF_DEFAULT_MAX_TILES_M,
      input int max_tiles_k  = REF_DEFAULT_MAX_TILES_K,
      input int max_tiles_n  = REF_DEFAULT_MAX_TILES_N,
      input int wbuf_depth   = REF_DEFAULT_WBUF_DEPTH,
      input int abuf_depth   = REF_DEFAULT_ABUF_DEPTH
  );
    int ew, ea;
    ew = ref_expected_wwords(desc.tiles_k, desc.tiles_n, n);
    ea = ref_expected_awords(desc.tiles_m, desc.tiles_k, n);
    return (desc.tiles_m == 0) || (desc.tiles_m > max_tiles_m) ||
           (desc.tiles_k == 0) || (desc.tiles_k > max_tiles_k) ||
           (desc.tiles_n == 0) || (desc.tiles_n > max_tiles_n) ||
           (ew > wbuf_depth) || (ea > abuf_depth);
  endfunction

  // =====================================================================
  // Effective-word clamp (ERR-003/UQ-004): replays the same beat-by-beat
  // "reached expected count OR last asserted" clamp gemm_ctrl.sv performs,
  // given the ordered queue of beats actually sent (each carrying its own
  // .last flag) and the expected (nominal) word count for the descriptor.
  // =====================================================================
  function automatic int ref_eff_words(input ref_beat_t beats[$], input int expected_words);
    int cnt;
    cnt = 0;
    foreach (beats[i]) begin
      cnt++;
      if ((cnt >= expected_words) || beats[i].last) return cnt;
    end
    return cnt; // fewer beats sent than expected and no last seen
  endfunction

  // =====================================================================
  // ref_compute_expected : THE golden model.
  //
  //   Given a decoded descriptor, the ordered queues of weight/activation
  //   beats actually sent, and the effective (clamped) word counts
  //   (eff_wwords/eff_awords -- typically produced by ref_eff_words()),
  //   produces the full ordered expected output-beat queue plus the
  //   expected sticky err_overflow and produces_output flag.
  //
  //   Illegal descriptor => expected_beats empty, produces_output=0,
  //   expected_err_overflow=0 (ERR-002/UQ-003; err_overflow is reserved
  //   for numeric saturation, never set for a rejected descriptor).
  // =====================================================================
  function automatic void ref_compute_expected(
      input  ref_desc_t   desc,
      input  ref_beat_t   weight_beats[$],
      input  ref_beat_t   act_beats[$],
      input  int          eff_wwords,
      input  int          eff_awords,
      output ref_beat_t   expected_beats[$],
      output bit          produces_output,
      output bit          expected_err_overflow,
      input  int          n            = REF_DEFAULT_N,
      input  int          max_tiles_m  = REF_DEFAULT_MAX_TILES_M,
      input  int          max_tiles_k  = REF_DEFAULT_MAX_TILES_K,
      input  int          max_tiles_n  = REF_DEFAULT_MAX_TILES_N,
      input  int          wbuf_depth   = REF_DEFAULT_WBUF_DEPTH,
      input  int          abuf_depth   = REF_DEFAULT_ABUF_DEPTH
  );
    int acc[REF_MAX_N][REF_MAX_N];
    int m_idx, n_idx, k_tile, k_cnt, i, j;
    int widx, aidx;
    longint signed prod64, shifted64;
    int sat;
    bit lane_ovf, beat_ovf;
    ref_beat_t obeat;
    byte lanes_out[REF_MAX_N];

    expected_beats.delete();
    produces_output       = 1'b0;
    expected_err_overflow = 1'b0;

    if (ref_is_illegal(desc, n, max_tiles_m, max_tiles_k, max_tiles_n, wbuf_depth, abuf_depth)) begin
      return; // ERR-002/UQ-003: illegal -> no output, no ovf, no done.
    end

    produces_output = 1'b1;

    for (m_idx = 0; m_idx < desc.tiles_m; m_idx++) begin
      for (n_idx = 0; n_idx < desc.tiles_n; n_idx++) begin

        // REQ-STORE-003: clear accumulators at the start of each output tile.
        for (i = 0; i < n; i++)
          for (j = 0; j < n; j++)
            acc[i][j] = 0;

        // REQ-GEMM-004/005: K-tile accumulation, k_tile outer / k_cnt inner.
        for (k_tile = 0; k_tile < desc.tiles_k; k_tile++) begin
          for (k_cnt = 0; k_cnt < n; k_cnt++) begin
            widx = (n_idx * desc.tiles_k + k_tile) * n + k_cnt;
            aidx = (m_idx * desc.tiles_k + k_tile) * n + k_cnt;

            // ERR-003/UQ-004: skip (contribute zero) unless both indices
            // fall within the actually-loaded (effective) buffer range.
            if ((widx < eff_wwords) && (aidx < eff_awords) &&
                (widx < weight_beats.size()) && (aidx < act_beats.size())) begin
              for (i = 0; i < n; i++) begin
                for (j = 0; j < n; j++) begin
                  // signed(a)*signed(w), accumulated into a 32-bit-wrapping
                  // `int` exactly like the RTL's plain signed 32-bit adder.
                  acc[i][j] = acc[i][j] +
                              (int'(ref_lane(act_beats[aidx], i)) *
                               int'(ref_lane(weight_beats[widx], j)));
                end
              end
            end
          end
        end

        // REQ-GEMM-006/007/013: requantize + emit one beat per row i.
        for (i = 0; i < n; i++) begin
          beat_ovf = 1'b0;
          for (j = 0; j < n; j++) begin
            prod64    = longint'(acc[i][j]) * longint'(desc.req_scale);
            shifted64 = prod64 >>> desc.req_shift; // arithmetic shift = floor(-inf)
            if (shifted64 > 127) begin
              sat = 127; lane_ovf = 1'b1;
            end else if (shifted64 < -128) begin
              sat = -128; lane_ovf = 1'b1;
            end else begin
              sat = int'(shifted64); lane_ovf = 1'b0;
            end
            lanes_out[j] = byte'(sat);
            if (lane_ovf) beat_ovf = 1'b1;
          end

          obeat.data = ref_pack_lanes(lanes_out, n);
          // ASM-006/UQ-008: last on final row of final (m,n) tile.
          obeat.last = (i == n - 1) && (m_idx == desc.tiles_m - 1) && (n_idx == desc.tiles_n - 1);
          expected_beats.push_back(obeat);

          // ERR-001/REQ-GEMM-012: sticky across the whole job.
          if (beat_ovf) expected_err_overflow = 1'b1;
        end
      end
    end
  endfunction

  // =====================================================================
  // ref_compare : strict, index-aligned expected-vs-actual beat compare.
  //   Never weakens on mismatch (no tolerance/fuzz -- INT8 GEMM output is
  //   bit-exact). Prints one diagnostic line per mismatching beat/lane,
  //   tagged with the requirement id(s) it validates, and increments
  //   error_count (caller-owned running total). Returns 1 (pass) only if
  //   beat counts match AND every beat's data+last matches exactly.
  // =====================================================================
  function automatic bit ref_compare(
      input ref_beat_t expected_q[$],
      input ref_beat_t actual_q[$],
      input int        n,
      input string     tag,
      ref  int         error_count
  );
    int ne, na, common, i, j;
    bit pass;
    logic [REF_MAX_LANES_W-1:0] mask, ed, ad;
    pass = 1'b1;
    ne = expected_q.size();
    na = actual_q.size();
    mask = ({REF_MAX_LANES_W{1'b1}} << (n*8)) ^ {REF_MAX_LANES_W{1'b1}}; // low n*8 bits set, n runtime

    if (ne != na) begin
      $display("SCOREBOARD MISMATCH [REQ-GEMM-008][ASM-006][UQ-008] %s: beat COUNT mismatch expected=%0d actual=%0d",
                tag, ne, na);
      error_count++;
      pass = 1'b0;
    end

    common = (ne < na) ? ne : na;
    for (i = 0; i < common; i++) begin
      ed = expected_q[i].data & mask;
      ad = actual_q[i].data & mask;
      if (ed !== ad) begin
        $display("SCOREBOARD MISMATCH [REQ-GEMM-006][REQ-GEMM-007][REQ-GEMM-013] %s: beat=%0d expected=0x%0h actual=0x%0h",
                  tag, i, ed, ad);
        for (j = 0; j < n; j++) begin
          if (ref_lane(expected_q[i], j) !== ref_lane(actual_q[i], j)) begin
            $display("    [REQ-GEMM-004][REQ-GEMM-006] lane=%0d expected=%0d(0x%0h) actual=%0d(0x%0h)",
                      j,
                      ref_lane(expected_q[i], j), ref_lane(expected_q[i], j) & 8'hFF,
                      ref_lane(actual_q[i], j),   ref_lane(actual_q[i], j) & 8'hFF);
          end
        end
        error_count++;
        pass = 1'b0;
      end
      if (expected_q[i].last !== actual_q[i].last) begin
        $display("SCOREBOARD MISMATCH [ASM-006][UQ-008][REQ-GEMM-008] %s: beat=%0d last-flag expected=%0b actual=%0b",
                  tag, i, expected_q[i].last, actual_q[i].last);
        error_count++;
        pass = 1'b0;
      end
    end
    return pass;
  endfunction

  // =====================================================================
  // ref_compare_overflow : sticky err_overflow compare (ERR-001/REQ-GEMM-012)
  // =====================================================================
  function automatic bit ref_compare_overflow(
      input bit    expected_ovf,
      input bit    actual_ovf,
      input string tag,
      ref  int     error_count
  );
    if (expected_ovf !== actual_ovf) begin
      $display("SCOREBOARD MISMATCH [REQ-GEMM-012][ERR-001] %s: err_overflow expected=%0b actual=%0b",
                tag, expected_ovf, actual_ovf);
      error_count++;
      return 1'b0;
    end
    return 1'b1;
  endfunction

  // =====================================================================
  // ref_compare_produces_output : illegal-descriptor discard check
  //   (ERR-002/UQ-003/REQ-GEMM-001). actual_produced should be computed by
  //   the caller as "did the DUT emit >=1 result beat / pulse done for this
  //   descriptor" (1) vs none (0).
  // =====================================================================
  function automatic bit ref_compare_produces_output(
      input bit    expected_po,
      input bit    actual_produced,
      input string tag,
      ref  int     error_count
  );
    if (expected_po !== actual_produced) begin
      $display("SCOREBOARD MISMATCH [ERR-002][UQ-003][REQ-GEMM-001] %s: produces_output expected=%0b actual=%0b",
                tag, expected_po, actual_produced);
      error_count++;
      return 1'b0;
    end
    return 1'b1;
  endfunction

endpackage : gemm_ref_pkg
