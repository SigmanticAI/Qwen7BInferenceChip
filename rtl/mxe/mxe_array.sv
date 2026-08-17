// mxe_array.sv — MXE_N x MXE_N true-systolic INT8 array with west-edge
// activation skew, a bank-B weight-load column shift chain, and south-edge
// INT32 accumulators (register file, M_TILE_MAX rows per column).
//
// Implements: D-004 (true systolic, 1 cycle/hop), D-005 (double-buffered
//             weights, wide column load), C-3 (INT32 accumulation),
//             ARCHITECTURE.md §3 (WS + OS dataflow modes — see OS SEMANTICS).
//
// ── Geometry / orientation ─────────────────────────────────────────────────
//   PE(r,c): r = 0..MXE_N-1 rows (K direction, north→south psum flow),
//            c = 0..MXE_N-1 columns (N direction, west→east act flow).
//   Stationary weight of PE(r,c) for K-chunk p is  W[r][c] = B[MXE_N*p + r][c].
//
// ── Weight-load column shift chain (D-005) ─────────────────────────────────
//   One `wload_en` strobe shifts one COLUMN of MXE_N INT8 values into the
//   NON-live bank: entry at the EAST edge (wload_lanes, lane r = row r),
//   shifting WEST each strobe. Exactly MXE_N strobes load a full chunk; the
//   i-th entered column comes to rest in array column i (entry i undergoes
//   MXE_N-1-i subsequent west shifts: column MXE_N-1 → column i). The
//   controller therefore always issues MXE_N strobes per pass: N descriptor
//   columns (in ascending c order) followed by MXE_N-N zero columns, and
//   zero-masks lanes r >= (K - MXE_N*p) on the final K-chunk.
//
// ── Compute timing (derivation the tag pipeline relies on) ─────────────────
//   A beat presented at cycle T (act_lanes, lane k = A[m][MXE_N*p + k])
//   passes the west-edge skew (row i delayed i cycles):
//     PE(r,c) sees the beat's activation on its west input at cycle T+r+c;
//     its psum register updates at end of T+r+c, so PE(r,c).psum_out shows
//     the row-r partial at cycle T+r+c+1. Bottom of column c (PE(MXE_N-1,c))
//     is therefore valid at cycle  T + MXE_N + c.
//   The tag {valid, clr, row} injected with the beat is delayed MXE_N+c
//   cycles to strobe column c's accumulator write. All pipes share `en`, so
//   alignment is preserved if the controller ever stalls the array.
//
// ── South accumulators / OS SEMANTICS (OP_GEMM_OS, precise definition) ─────
//   acc[c][m] is an INT32 register file (MXE_N columns x M_TILE_MAX rows).
//   On a valid tag for column c: acc[c][m] <= (clr ? 0 : acc[c][m]) + bottom.
//   The controller computes C = A[MxK]·B[KxN] as ceil(K/MXE_N) passes; the
//   per-beat `clr` flag is 1 only on pass 0 of a job that OVERWRITES.
//   * OP_GEMM_WS:  pass-0 beats have clr=1 → acc[c][m] = C[m][c] afterwards.
//   * OP_GEMM_OS:  the results are RESIDENT in these accumulators:
//       - accumulate=0: pass-0 clr=1, job computes acc[m][c]  = C[m][c].
//       - accumulate=1: pass-0 clr=0, job computes acc[m][c] += C[m][c],
//         retaining prior contents — this is the cross-descriptor K-dim
//         tiling primitive (split K > K_MAX or streamed Q·K̂ᵀ / P·V̂ chunks
//         into several OS descriptors, accumulate=1 on all but the first).
//     Draining (rd_row read-out) is NON-destructive: accumulator contents
//     persist until the next overwriting job touches the same rows. Software
//     must keep M/N consistent across an accumulate chain (as on real HW,
//     mismatched chained tiles are numerically meaningless but not illegal).
//   Rows m >= M and columns c >= N of a job are never strobed (tags carry
//   only real rows; unused columns hold zero weights so add zero anyway).

module mxe_array
  import apex_pkg::*;
  import mxe_cfg_pkg::*;
(
  input  logic clk,
  input  logic rst_n,            // synchronous, active-low

  // compute pipeline advance (shared by skew regs, PEs, tag pipe, acc write)
  input  logic en,

  // activation beat injection (pre-skew, lane k = data[8k +: 8])
  input  logic [8*MXE_N-1:0] act_lanes,
  input  logic               beat_valid,
  input  logic               beat_clear,   // clr flag captured with this beat
  input  logic [ROW_W-1:0]   beat_row,     // m index of this beat

  // stationary-weight double-buffer control (D-005)
  input  logic               live_sel,     // computing bank
  input  logic               wload_en,     // column shift strobe (bank ~live)
  input  logic [8*MXE_N-1:0] wload_lanes,  // east-edge column, lane r = row r

  // accumulator drain read (combinational; stable while en==0)
  input  logic [ROW_W-1:0]   rd_row,
  output acc_t               rd_lanes [MXE_N]
);

  // ── west-edge activation skew: row i delayed i cycles ────────────────────
  act_t act_west [MXE_N];

  for (genvar r = 0; r < int'(MXE_N); r++) begin : g_skew
    if (r == 0) begin : g_row0
      assign act_west[0] = act_t'(act_lanes[7:0]);
    end else begin : g_rown
      act_t pipe [r];
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          for (int j = 0; j < r; j++) pipe[j] <= '0;
        end else if (en) begin
          pipe[0] <= act_t'(act_lanes[8*r +: 8]);
          for (int j = 1; j < r; j++) pipe[j] <= pipe[j-1];
        end
      end
      assign act_west[r] = pipe[r-1];
    end
  end

  // ── PE grid ───────────────────────────────────────────────────────────────
  act_t act_h  [MXE_N][MXE_N+1];   // act_h[r][c]  = west input of PE(r,c)
  acc_t psum_v [MXE_N+1][MXE_N];   // psum_v[r][c] = north input of PE(r,c)
  act_t wchain [MXE_N][MXE_N+1];   // wchain[r][c+1] = load input of PE(r,c)

  for (genvar c = 0; c < int'(MXE_N); c++) begin : g_ptop
    assign psum_v[0][c] = '0;      // north edge: zero partials
  end

  for (genvar r = 0; r < int'(MXE_N); r++) begin : g_row
    assign act_h[r][0]        = act_west[r];
    assign wchain[r][MXE_N]   = act_t'(wload_lanes[8*r +: 8]);  // east edge in
    for (genvar c = 0; c < int'(MXE_N); c++) begin : g_col
      mxe_pe u_pe (
        .clk       (clk),
        .rst_n     (rst_n),
        .en        (en),
        .act_in    (act_h[r][c]),
        .act_out   (act_h[r][c+1]),
        .psum_in   (psum_v[r][c]),
        .psum_out  (psum_v[r+1][c]),
        .live_sel  (live_sel),
        .wload_en  (wload_en),
        .wload_in  (wchain[r][c+1]),
        .wload_out (wchain[r][c])
      );
    end
  end

  // east-edge act outputs and west-edge chain outputs terminate here
  logic unused_edges;
  always_comb begin
    unused_edges = 1'b0;
    for (int r = 0; r < int'(MXE_N); r++)
      unused_edges = unused_edges ^ (^act_h[r][MXE_N]) ^ (^wchain[r][0]);
  end

  // ── tag pipeline: delay MXE_N+c strobes column c's accumulator ───────────
  typedef struct packed {
    logic             valid;
    logic             clr;
    logic [ROW_W-1:0] row;
  } tag_t;

  // tag_pipe[j] holds the tag injected j+1 cycles ago (delays 1..TAGP_DEPTH);
  // column c uses delay MXE_N+c  →  tap index MXE_N+c-1  (c = 0..MXE_N-1).
  tag_t tag_pipe [TAGP_DEPTH];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int j = 0; j < int'(TAGP_DEPTH); j++) tag_pipe[j] <= '0;
    end else if (en) begin
      tag_pipe[0] <= '{valid: beat_valid, clr: beat_clear, row: beat_row};
      for (int j = 1; j < int'(TAGP_DEPTH); j++) tag_pipe[j] <= tag_pipe[j-1];
    end
  end

  // ── south-edge INT32 accumulators (register file, per-column write) ──────
  acc_t acc [MXE_N][M_TILE_MAX];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int c = 0; c < int'(MXE_N); c++)
        for (int m = 0; m < int'(M_TILE_MAX); m++)
          acc[c][m] <= '0;
    end else if (en) begin
      for (int c = 0; c < int'(MXE_N); c++) begin
        if (tag_pipe[int'(MXE_N) + c - 1].valid) begin
          acc[c][tag_pipe[int'(MXE_N)+c-1].row] <=
            (tag_pipe[int'(MXE_N)+c-1].clr
               ? acc_t'(0)
               : acc[c][tag_pipe[int'(MXE_N)+c-1].row])
            + psum_v[MXE_N][c];
        end
      end
    end
  end

  for (genvar c = 0; c < int'(MXE_N); c++) begin : g_rd
    assign rd_lanes[c] = acc[c][rd_row];
  end

endmodule
