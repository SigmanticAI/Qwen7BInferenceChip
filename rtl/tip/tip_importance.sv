// tip_importance.sv — per-block saturating token-importance accumulators
// (TIP-lite, the the reference design concept) driving a per-block KVQ tier SUGGESTION.
//
// Implements: ARCHITECTURE.md §1 TIP row ("saturating per-block importance
// accumulators"), D-011 (drives per-block KVQ tier select). Golden mirror:
// verif/tip/smoke/gen_tip_vectors.py (imp_update_golden / tier_golden).
//
// UPDATE RULE (defined precisely — the "decision-weighted magnitude bucket"):
//   One update per TIP decision (upd_valid pulse from tip_decide via tip_top),
//   addressed by the tile's block id upd_blk:
//
//     bucket(max_mag) = 0                      if max_mag == 0
//                       floor(log2(max_mag))+1 otherwise
//                     = index of the highest set bit, 1-based
//                     ∈ [0, MAG_W]  (MAG_W=32 ⇒ bucket ≤ 32, 6 bits)
//
//     increment = bucket * 2   if upd_fp16 (FP16/important decision)
//                 bucket * 1   otherwise  (INT8 decision)
//               ∈ [0, 64]      (7 bits)
//
//     acc[blk] ← min(acc[blk] + increment, 2^ACC_W - 1)     (saturating)
//
//   Rationale: log-magnitude buckets make the accumulator scale-free across
//   layers (raw INT32 score magnitudes span 31 octaves); the x2 decision
//   weight biases blocks whose tiles keep tripping the FP16 ratio test —
//   exactly the blocks KVQ should keep at higher precision.
//
// TIER SUGGESTION (2 threshold registers, >= compares):
//     acc >= thresh_hi              → KVQ_CQ8   (most important, full 8-bit)
//     thresh_lo <= acc < thresh_hi  → KVQ_CQ4P  (CQ-4+)
//     acc <  thresh_lo              → KVQ_CQ4   (default tier)
//   upd_tier is the suggestion for the POST-update value of upd_blk, valid
//   combinationally in the upd_valid cycle (tip_top pairs it with the
//   decision beat). rd_tier is the suggestion for the STORED (already
//   committed) value at rd_addr.
//
// CSR read port: combinational (a 128x16 register-file mux) — rd_addr in,
// rd_data/rd_tier out, no handshake (IMPORTANCE_BASE window, §7).
// clear_all: synchronous whole-array clear (per-layer/CSR reuse).

module tip_importance
  import apex_pkg::*;
#(
  parameter int unsigned BLOCKS = 128,
  parameter int unsigned ACC_W  = 16,
  parameter int unsigned MAG_W  = 32          // = SCORE_WIDTH of tip_decide
) (
  input  logic                       clk,
  input  logic                       rst_n,   // synchronous, active-low

  // update port — one pulse per TIP decision
  input  logic                       upd_valid,
  input  logic [$clog2(BLOCKS)-1:0]  upd_blk,
  input  logic                       upd_fp16,
  input  logic [MAG_W-1:0]           upd_max_mag,
  output kvq_tier_e                  upd_tier,   // post-update suggestion (comb)

  // tier thresholds + clear (CSR)
  input  logic [ACC_W-1:0]           thresh_hi,
  input  logic [ACC_W-1:0]           thresh_lo,
  input  logic                       clear_all,

  // CSR read port (combinational)
  input  logic [$clog2(BLOCKS)-1:0]  rd_addr,
  output logic [ACC_W-1:0]           rd_data,
  output kvq_tier_e                  rd_tier
);

  localparam int unsigned BKT_W = $clog2(MAG_W + 1);   // 6 for MAG_W=32
  localparam int unsigned INC_W = BKT_W + 1;           // bucket*2 fits

  logic [ACC_W-1:0] acc [BLOCKS];

  // ── bucket = 1-based index of the highest set bit of max_mag ─────────────
  function automatic logic [BKT_W-1:0] mag_bucket(input logic [MAG_W-1:0] m);
    mag_bucket = '0;
    for (int unsigned i = 0; i < MAG_W; i++)
      if (m[i]) mag_bucket = BKT_W'(i + 1);
  endfunction

  // ── tier rule (shared by upd_tier and rd_tier) ────────────────────────────
  function automatic kvq_tier_e tier_of(input logic [ACC_W-1:0] a,
                                        input logic [ACC_W-1:0] hi,
                                        input logic [ACC_W-1:0] lo);
    if      (a >= hi) tier_of = KVQ_CQ8;
    else if (a >= lo) tier_of = KVQ_CQ4P;
    else              tier_of = KVQ_CQ4;
  endfunction

  // ── saturating update ─────────────────────────────────────────────────────
  logic [INC_W-1:0]  inc;
  logic [ACC_W:0]    acc_sum;                  // 1 carry bit
  logic [ACC_W-1:0]  acc_post;

  assign inc      = upd_fp16 ? {mag_bucket(upd_max_mag), 1'b0}
                             : {1'b0, mag_bucket(upd_max_mag)};
  assign acc_sum  = {1'b0, acc[upd_blk]} + {{(ACC_W-INC_W+1){1'b0}}, inc};
  assign acc_post = acc_sum[ACC_W] ? {ACC_W{1'b1}} : acc_sum[ACC_W-1:0];

  assign upd_tier = tier_of(acc_post, thresh_hi, thresh_lo);

  always_ff @(posedge clk) begin
    if (!rst_n || clear_all) begin
      for (int unsigned i = 0; i < BLOCKS; i++) acc[i] <= '0;
    end else if (upd_valid) begin
      acc[upd_blk] <= acc_post;
    end
  end

  // ── CSR read port (stored/committed values) ───────────────────────────────
  assign rd_data = acc[rd_addr];
  assign rd_tier = tier_of(rd_data, thresh_hi, thresh_lo);

endmodule
