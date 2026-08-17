// tip_top.sv — TIP (Token Importance & Precision) block top: tip_decide
// (D-011/D-017 programmable ratio test + V0.3 F-2 framing guard) +
// tip_importance (TIP-lite per-block accumulators + KVQ tier suggestion),
// wired to §5/D-006 streams with a stream_skid (V0.5-verified) on the score
// input and on the decision output.
//
// Implements: ARCHITECTURE.md §1 TIP row, §5 stream discipline, D-006, D-011,
// D-017. Golden mirror: verif/tip/smoke/gen_tip_vectors.py.
//
// STREAMS (single valid/ready discipline, data stable while valid && !ready):
//   score in : s_valid/s_ready, payload {s_blk, s_last, s_data} through a
//              2-deep stream_skid. One INT32 raw pre-softmax score per beat
//              (C-5); s_last marks the tile end; s_blk is the tile's KV block
//              id — SAMPLED ON THE s_last BEAT (earlier beats' s_blk is
//              don't-care, by convention constant across the tile).
//   decision out : d_valid/d_ready, payload {d_blk, d_tier, d_fp16} through a
//              2-deep stream_skid. Exactly one beat per accepted
//              (non-frame-errored) tile, in tile order. d_tier is the
//              POST-update tip_importance suggestion for that tile's block.
//
// D-006 semantics for a streaming (descriptor-less) block:
//   - every decision beat is delivered through the output skid and is only
//     "done" when ACCEPTED downstream; nothing is dropped under backpressure:
//     the feed into tip_decide is GATED (see below) so a new decision is
//     never produced while the previous one has not been handed to the
//     output skid — backpressure propagates losslessly to s_ready.
//   - busy = beat pending in the input skid, OR tip_decide mid-tile, OR a
//     decision in flight/held, OR a decision beat pending in the output skid.
//
// Feed gating (why no decision can ever be lost): tip_decide emits d_valid
// exactly 1 cycle after consuming an s_last beat. tip_top consumes beats from
// the input skid only while BOTH (a) the 1-entry decision hold register is
// empty and (b) no s_last beat was consumed in the previous cycle (dec_pend).
// Hence at most one decision is in flight and it always lands in an empty
// hold register; the hold register drains into the output skid whenever the
// skid has room. Throughput cost: 2 idle feed cycles per tile plus any
// downstream stall — irrelevant against the N-beat tile length.
//
// Frame errors (V0.3 F-2): frame_err / frame_err_sticky / frame_err_clear
// are forwarded from tip_decide (CSR STATUS material, §7). An erroring tile
// produces NO decision beat and leaves the engine clean after its s_last.
//
// CSR-facing (quasi-static per D-011, §7): threshold (THRESHOLD_REG, 0 is
// clamped to 1 in tip_decide), imp_thresh_hi/lo (TIER_CTRL), imp_clear,
// imp_rd_addr → imp_rd_data/imp_rd_tier (IMPORTANCE_BASE window).

module tip_top
  import apex_pkg::*;
#(
  parameter int unsigned SCORE_WIDTH = 32,   // D-011
  parameter int unsigned BLOCK_M     = 64,
  parameter int unsigned BLOCK_N     = 64,
  parameter int unsigned BLOCKS      = 128,
  parameter int unsigned ACC_W       = 16,
  parameter int unsigned T_MAX       = 31
) (
  input  logic                          clk,
  input  logic                          rst_n,      // synchronous, active-low

  // score stream in (§5, via stream_skid)
  input  logic                          s_valid,
  output logic                          s_ready,
  input  logic signed [SCORE_WIDTH-1:0] s_data,
  input  logic                          s_last,
  input  logic [$clog2(BLOCKS)-1:0]     s_blk,

  // decision + tier stream out (§5, via stream_skid)
  output logic                          d_valid,
  input  logic                          d_ready,
  output logic                          d_fp16,
  output kvq_tier_e                     d_tier,
  output logic [$clog2(BLOCKS)-1:0]     d_blk,

  // CSR-facing controls (§7)
  input  logic [$clog2(T_MAX+1)-1:0]    threshold,       // THRESHOLD_REG
  input  logic [ACC_W-1:0]              imp_thresh_hi,   // TIER_CTRL
  input  logic [ACC_W-1:0]              imp_thresh_lo,
  input  logic                          imp_clear,
  input  logic [$clog2(BLOCKS)-1:0]     imp_rd_addr,     // IMPORTANCE_BASE
  output logic [ACC_W-1:0]              imp_rd_data,
  output kvq_tier_e                     imp_rd_tier,

  // framing guard (V0.3 F-2; STATUS sticky, §7)
  output logic                          frame_err,
  output logic                          frame_err_sticky,
  input  logic                          frame_err_clear,

  // §5 busy semantics (beats pending anywhere in the block, skids included)
  output logic                          busy
);

  localparam int unsigned BLK_W = $clog2(BLOCKS);
  localparam int unsigned IN_W  = BLK_W + 1 + SCORE_WIDTH;  // {blk, last, data}
  localparam int unsigned OUT_W = BLK_W + 2 + 1;            // {blk, tier, fp16}

  // ── input skid (XBR discipline: skid on every external stream) ───────────
  logic             in_m_valid, in_m_ready;
  logic [IN_W-1:0]  in_m_data;

  stream_skid #(.WIDTH(IN_W)) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (s_valid),
    .s_ready (s_ready),
    // no unsigned'() cast: concat operands are self-determined unsigned
    // already, and sv2v would re-emit $unsigned() inside the port expression,
    // which trips a yosys 0.66 frontend assert (genrtlil.cc:2214). Same bits.
    .s_data  ({s_blk, s_last, s_data}),
    .m_valid (in_m_valid),
    .m_ready (in_m_ready),
    .m_data  (in_m_data)
  );

  logic [BLK_W-1:0]       in_blk;
  logic                   in_last;
  // signed: matches tip_decide.s_data, so no signed'() cast is needed at the
  // port (sv2v turns that cast into $signed() in a port connection, which
  // asserts in yosys 0.66). The lvalue-concat unpack below is bitwise —
  // identical bits either way.
  logic signed [SCORE_WIDTH-1:0] in_score;
  assign {in_blk, in_last, in_score} = in_m_data;

  // ── feed gate: at most one decision in flight, hold reg always free ──────
  logic dec_pend;     // an s_last beat was consumed last cycle
  logic hold_valid;   // decision hold register occupied
  logic feed_gate;
  logic eng_fire;
  assign feed_gate  = !hold_valid && !dec_pend;
  assign in_m_ready = feed_gate;
  assign eng_fire   = in_m_valid && feed_gate;

  // ── decision engine (D-011/D-017 rewrite) ─────────────────────────────────
  logic                   eng_d_valid, eng_d_fp16;
  logic [SCORE_WIDTH-1:0] eng_d_max;
  logic                   eng_tile_active;

  tip_decide #(
    .SCORE_WIDTH (SCORE_WIDTH),
    .BLOCK_M     (BLOCK_M),
    .BLOCK_N     (BLOCK_N),
    .T_MAX       (T_MAX)
  ) u_decide (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_valid          (eng_fire),
    .s_data           (in_score),
    .s_last           (in_last),
    .threshold        (threshold),
    .d_valid          (eng_d_valid),
    .d_fp16           (eng_d_fp16),
    .d_max_mag        (eng_d_max),
    .frame_err        (frame_err),
    .frame_err_sticky (frame_err_sticky),
    .frame_err_clear  (frame_err_clear),
    .tile_active      (eng_tile_active)
  );

  // ── importance accumulators + tier suggestion (TIP-lite) ─────────────────
  logic [BLK_W-1:0] blk_q;   // block id sampled on the consumed s_last beat
  kvq_tier_e        upd_tier;

  tip_importance #(
    .BLOCKS (BLOCKS),
    .ACC_W  (ACC_W),
    .MAG_W  (SCORE_WIDTH)
  ) u_importance (
    .clk         (clk),
    .rst_n       (rst_n),
    .upd_valid   (eng_d_valid),
    .upd_blk     (blk_q),
    .upd_fp16    (eng_d_fp16),
    .upd_max_mag (eng_d_max),
    .upd_tier    (upd_tier),
    .thresh_hi   (imp_thresh_hi),
    .thresh_lo   (imp_thresh_lo),
    .clear_all   (imp_clear),
    .rd_addr     (imp_rd_addr),
    .rd_data     (imp_rd_data),
    .rd_tier     (imp_rd_tier)
  );

  // ── decision hold register → output skid ─────────────────────────────────
  logic             out_s_ready;
  logic [OUT_W-1:0] hold_beat;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      dec_pend   <= 1'b0;
      blk_q      <= '0;
      hold_valid <= 1'b0;
      hold_beat  <= '0;
    end else begin
      dec_pend <= eng_fire && in_last;
      if (eng_fire && in_last) blk_q <= in_blk;
      if (hold_valid && out_s_ready) hold_valid <= 1'b0;
      if (eng_d_valid) begin
        // hold is empty by construction (feed gate) — never overwrites
        hold_valid <= 1'b1;
        hold_beat  <= {blk_q, 2'(upd_tier), eng_d_fp16};
      end
    end
  end

  logic [OUT_W-1:0] out_m_data;

  stream_skid #(.WIDTH(OUT_W)) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (hold_valid),
    .s_ready (out_s_ready),
    .s_data  (hold_beat),
    .m_valid (d_valid),
    .m_ready (d_ready),
    .m_data  (out_m_data)
  );

  logic [1:0] out_tier_bits;
  assign {d_blk, out_tier_bits, d_fp16} = out_m_data;
  assign d_tier = kvq_tier_e'(out_tier_bits);

  // ── §5 busy: any beat pending anywhere in the block, skids included ──────
  // (skid-slot occupancy implies m_valid in stream_skid, so m_valid covers
  //  both stages of each skid)
  assign busy = in_m_valid || eng_tile_active || dec_pend || hold_valid
             || d_valid;

endmodule
