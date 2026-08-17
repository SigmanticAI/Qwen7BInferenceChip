// mxe_cfg_pkg.sv — MXE implementation-local sizing (NOT part of the frozen
// apex_pkg contract; changing these does not change the tile's external
// numerics/stream contract, only the implemented tile capacity).
//
// Implements ARCHITECTURE.md sizing guidance: M/K/N tile handling is kept
// within on-chip buffers defined here. K is supported up to the full
// contract K_MAX = 2048 via multi-pass accumulation (ceil(K/MXE_N) passes of
// MXE_N-deep systolic chunks). The implemented per-descriptor tile limits
// beyond the frozen legality rules are:
//   M ≤ M_TILE_MAX (64)  — bounded by the activation buffer / accumulator RAM
//   N ≤ MXE_N      (8)   — one array width per descriptor (no N tiling v0.1)
// Descriptors exceeding these are legality-rejected as "buffer overrun"
// (ARCHITECTURE.md §3), with desc_error semantics identical to K > K_MAX.
`ifndef MXE_CFG_PKG_SV
`define MXE_CFG_PKG_SV

package mxe_cfg_pkg;

  import apex_pkg::*;

  // max rows (M) of a single descriptor's A/C tile
  localparam int unsigned M_TILE_MAX = 64;
  // max number of K-chunks (passes): full contract K_MAX supported
  localparam int unsigned KB_MAX     = K_MAX / MXE_N;              // 256

  localparam int unsigned ROW_W      = $clog2(M_TILE_MAX);         // 6
  localparam int unsigned PASS_W     = $clog2(KB_MAX) + 1;         // 9

  // activation buffer: one 8-lane INT8 beat per (row m, K-chunk kb),
  // addressed m*KB + kb (row-major beats, chunk-minor)
  localparam int unsigned ABUF_DEPTH = M_TILE_MAX * KB_MAX;        // 16384
  localparam int unsigned ABUF_AW    = $clog2(ABUF_DEPTH);         // 14
  localparam int unsigned ACNT_W     = ABUF_AW + 1;                // beat counters

  // array tag pipeline: vertical transit (MXE_N) + horizontal column skew
  // (MXE_N-1) + 1 margin — see mxe_array.sv timing derivation
  localparam int unsigned TAGP_DEPTH = 2*MXE_N - 1;                // 15
  localparam int unsigned FLUSH_CYC  = 2*MXE_N;                    // 16

endpackage

`endif // MXE_CFG_PKG_SV
