// apex_pkg.sv — APEX NPU single source of truth for parameters, types, and
// interface contracts. Mirrored by golden/apex_golden (Python) and every TB.
//
// Implements ARCHITECTURE.md: C-1..C-5 numerics, §3 MXE descriptor, §5 stream
// semantics (D-006), D-003 (INT32 accumulators, K ≤ 2048), D-014 (exp LUT).
// Any change here is a contract change: bump APEX_VERSION and update the
// golden mirrors in the same commit.
`ifndef APEX_PKG_SV
`define APEX_PKG_SV

package apex_pkg;

  // ── identity ──────────────────────────────────────────────────────────────
  localparam int unsigned APEX_VERSION = 32'h0001_0000;   // v0.1

  // ── array / tile geometry ─────────────────────────────────────────────────
  localparam int unsigned MXE_N   = 8;      // systolic array is MXE_N x MXE_N
  localparam int unsigned K_MAX   = 2048;   // C-3: 21+11 = exactly INT32
  localparam int unsigned DIM_W   = 12;     // M/K/N descriptor fields

  // ── datatypes (C-1..C-5) ──────────────────────────────────────────────────
  typedef logic signed [7:0]  act_t;        // INT8 activations (and weights v0.1)
  typedef logic signed [31:0] acc_t;        // C-3 accumulator
  typedef logic signed [7:0]  out8_t;       // requant epilogue output

  // ── MXE opcodes / dataflow modes ─────────────────────────────────────────
  typedef enum logic [7:0] {
    OP_NOP     = 8'h00,
    OP_GEMM_WS = 8'h01,   // weight-stationary: projections / FFN
    OP_GEMM_OS = 8'h02    // output-stationary: Q·K̂ᵀ, P·V̂
  } mxe_op_e;

  // ── MXE descriptor — 128-bit, run1 lineage, layout FROZEN ────────────────
  // Legality (checked before ANY state change; violation → desc_error pulse +
  // sticky CSR bit, descriptor consumed, NO side effects — §3):
  //   opcode ∈ {OP_GEMM_WS, OP_GEMM_OS}; 1 ≤ K ≤ K_MAX; M,N ≥ 1;
  //   shift ≤ 31 (field-width enforced); buffer ids within implemented range.
  typedef struct packed {           // [127:0], MSB first
    logic [47:0]  reserved;         // [127:80]
    logic [3:0]   dst_buf;          // [79:76]
    logic [3:0]   src_b_buf;        // [75:72]
    logic [3:0]   src_a_buf;        // [71:68]
    logic         mode_os;          // [67]  0=WS, 1=OS
    logic         accumulate;       // [66]  OS: retain accumulators
    logic         requant_en;       // [65]  epilogue INT32→INT8 (C-2)
    logic [4:0]   rq_shift;         // [64:60]
    logic [15:0]  rq_scale;         // [59:44] unsigned magnitude
    logic [DIM_W-1:0] n_dim;        // [43:32]
    logic [DIM_W-1:0] k_dim;        // [31:20]
    logic [DIM_W-1:0] m_dim;        // [19:8]
    logic [7:0]   opcode;           // [7:0]
  } mxe_desc_t;

  // ── stream payloads ───────────────────────────────────────────────────────
  // All inter-block streams use valid/ready with 2-deep skids (stream_skid,
  // V0.5-verified) at every boundary. Data MUST be stable while valid&&!ready.
  typedef struct packed {
    logic [8*MXE_N-1:0] data;       // MXE_N packed INT8 lanes
    logic               last;       // informational tile-end marker
  } lane8_beat_t;

  typedef struct packed {
    logic [32*MXE_N-1:0] data;      // MXE_N packed INT32 lanes
    logic                last;
  } lane32_beat_t;

  // ── job semantics (D-006, SVA-enforced in apex_stream_sva.svh) ───────────
  //   done  ⇒ every result beat of the job has been ACCEPTED downstream
  //           (post-skid). One `done` pulse per accepted legal descriptor.
  //   busy  = (FSM ≠ IDLE) || (result beats pending anywhere, skids incl.)
  //   desc_ready for job N+1 MUST NOT assert before job N's done.  (SEQ + MXE
  //   both enforce; this closes run1's root-caused wart, D-019.)

  // ── requant epilogue contract (C-2) ──────────────────────────────────────
  //   y = sat8( rne( (acc * rq_scale) >> rq_shift ) ), RNE = half-to-even.
  //   Golden mirror: apex_golden.compute.requant_i32_to_i8. Product held in
  //   48-bit signed (32+16); rounding computed in exact integer arithmetic.

  // ── ASU fixed-point formats (§6, D-014) ───────────────────────────────────
  localparam int unsigned ASU_LUT_BITS  = 8;    // 256-entry exp LUT
  localparam int unsigned ASU_IN_FRAC   = 10;   // exp input Q6.10 in [-16,0]
  localparam int unsigned ASU_OUT_FRAC  = 15;   // probabilities Q1.15
  localparam int unsigned RSQRT_LATENCY = 31;   // D-018: measured, not the doc's 30

  // ── KVQ tiers (INFO_TIER register encoding) ─────────────────────
  typedef enum logic [1:0] {
    KVQ_CQ8  = 2'd0,
    KVQ_CQ4  = 2'd1,
    KVQ_CQ4P = 2'd2
  } kvq_tier_e;

endpackage

`endif // APEX_PKG_SV
