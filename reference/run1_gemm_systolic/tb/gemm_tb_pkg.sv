// =============================================================================
// File   : gemm_tb_pkg.sv
// Package: gemm_tb_pkg
// -----------------------------------------------------------------------------
// PURPOSE
//   Shared, tool-agnostic (plain SystemVerilog, no UVM) package for all
//   testbench specialists (stimulus_sequence, scoreboard_checker,
//   coverage_sva, ...) that plug into the gemm_tb_top structural harness.
//
//   This package owns ONLY:
//     - compile-time sizing localparams (mirrors of DUT parameter defaults)
//     - the descriptor field layout struct + make_descriptor() packer
//     - a generic streaming-beat transaction struct (used identically for
//       weight / activation / result beats -- same {data,last} shape)
//     - a tiny checked-transaction struct for scoreboard use (expected vs
//       actual), provided as a TYPE ONLY -- no comparison logic lives here.
//
//   No stimulus, no scoreboard math, no covergroups, no assertions are
//   defined in this file (owned by other specialist agents). This file is
//   pure data-shape + encode helper, safe for every specialist to import.
//
// USAGE
//   import gemm_tb_pkg::*;
//   gemm_tb_pkg::desc_fields_t f;
//   logic [127:0] d = gemm_tb_pkg::make_descriptor(tiles_m, tiles_k, tiles_n,
//                                                   req_shift, req_scale);
// =============================================================================
`ifndef GEMM_TB_PKG_SV
`define GEMM_TB_PKG_SV

package gemm_tb_pkg;

  // ---------------------------------------------------------------------
  // Sizing (mirrors DUT defaults; downstream code may still instantiate
  // the DUT with different parameters -- these are just TB-side defaults
  // used for vector sizing in the BFM/harness).
  // ---------------------------------------------------------------------
  localparam int TB_DATA_W  = 8;
  localparam int TB_ACC_W   = 32;
  localparam int TB_SCALE_W = 32;
  localparam int TB_SHIFT_W = 6;
  localparam int TB_DESC_W  = 128;

  // ---------------------------------------------------------------------
  // Descriptor field layout (RESOLVED, per spec):
  //   [7:0]    tiles_m
  //   [15:8]   tiles_k
  //   [23:16]  tiles_n
  //   [29:24]  req_shift
  //   [31:30]  reserved (=0)
  //   [63:32]  req_scale (signed INT32)
  //   [95:64]  flags (=0)
  //   [127:96] reserved (=0)
  // ---------------------------------------------------------------------
  typedef struct packed {
    logic [31:0] reserved_hi;   // [127:96]
    logic [31:0] flags;         // [95:64]  (=0)
    logic [31:0] req_scale;     // [63:32]  signed INT32
    logic [1:0]  reserved_lo;   // [31:30]
    logic [5:0]  req_shift;     // [29:24]
    logic [7:0]  tiles_n;       // [23:16]
    logic [7:0]  tiles_k;       // [15:8]
    logic [7:0]  tiles_m;       // [7:0]
  } desc_fields_t;

  // Packer: build a 128b descriptor word from individual fields.
  function automatic logic [127:0] make_descriptor(
      input logic [7:0]  tiles_m,
      input logic [7:0]  tiles_k,
      input logic [7:0]  tiles_n,
      input logic [5:0]  req_shift,
      input logic [31:0] req_scale
  );
    desc_fields_t f;
    f.tiles_m      = tiles_m;
    f.tiles_k      = tiles_k;
    f.tiles_n      = tiles_n;
    f.req_shift    = req_shift;
    f.reserved_lo  = '0;
    f.req_scale    = req_scale;
    f.flags        = '0;
    f.reserved_hi  = '0;
    return f;
  endfunction

  // ---------------------------------------------------------------------
  // Generic streaming-beat transaction. Same shape used for weight,
  // activation, and result beats (all are {data[N*DATA_W-1:0], last}).
  // Width is left as a plain logic vector at MAX supported width (128b,
  // covers N up to 16 @ DATA_W=8); callers slice down to N*DATA_W-1:0.
  // ---------------------------------------------------------------------
  typedef struct {
    logic [127:0] data;
    logic         last;
    int           lane_width;   // informational: N*DATA_W actually used
  } beat_txn_t;

  // ---------------------------------------------------------------------
  // Scoreboard-facing expected/actual pair. TYPE ONLY -- no compare logic.
  // The scoreboard_checker specialist owns comparison semantics; this is
  // just a shared shape so it doesn't need to redefine it.
  // ---------------------------------------------------------------------
  typedef struct {
    beat_txn_t expected;
    beat_txn_t actual;
    bit        checked;
  } result_check_t;

endpackage : gemm_tb_pkg

`endif // GEMM_TB_PKG_SV
