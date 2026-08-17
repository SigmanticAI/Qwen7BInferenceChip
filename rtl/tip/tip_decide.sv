// tip_decide.sv — TIP importance ratio test at SCORE_WIDTH=32 with a
// PROGRAMMABLE threshold (D-011/D-017): a hardware peakedness detector on raw
// pre-softmax scores. CMP_W = SUM_W + clog2(T_MAX+1) so any threshold
// 1..T_MAX is exact (no hardcoded multiplier).
//
// Implements: ARCHITECTURE.md §1 TIP row, C-5, D-011, D-017 (incl. the V0.3
// F-2 framing guard). Golden mirror: verif/tip/smoke/gen_tip_vectors.py
// (tip_decide_golden — bit-exact, same masked widths).
//
// Decision rule (on raw pre-softmax scores, |·| = two's-complement magnitude;
// the abs of INT_MIN is the exact unsigned magnitude 2^(W-1), exact unsigned magnitude):
//
//   FP16  ⇔  max(|s|) * N  >  T * sum(|s|)          (strict >; tie → INT8)
//
//   LHS = max << LOG2_N                              (free shift, N = 2^LOG2_N)
//   RHS = T * sum, T = threshold_reg ∈ 1..T_MAX=31   (PROGRAMMABLE shift-add
//         tree: RHS = Σ_{i=0..4} (T[i] ? sum<<i : 0) — five conditional
//         partial products, i.e. the exact generalization of upstream's
//         hardwired (sum<<3)+(sum<<1); chosen over a DSP-style sum*T multiply
//         so the datapath stays multiplier-free like the rest of the tile and
//         the no-overflow bound below is inspectable per partial product)
//   CMP_W = SUM_W + $clog2(T_MAX+1) = SUM_W + 5      (D-017, replaces the +4)
//
// threshold_reg contract: legal values 1..T_MAX. The all-zeros value is
// CLAMPED to 1 in hardware (thr_eff) so a never-programmed CSR cannot yield
// the degenerate RHS=0 (everything nonzero → FP16); golden mirrors the clamp.
// threshold is sampled combinationally on the s_last beat (a per-layer
// quasi-static CSR per D-011; mid-tile writes only matter on that beat).
//
// No-overflow bound (W=SCORE_WIDTH, L=LOG2_N, guard enforces ≤ N scores/tile):
//   abs ≤ 2^(W-1)  ⇒  sum ≤ N·2^(W-1) = 2^(SUM_W-1)   (no sum wrap, 1 bit spare)
//   LHS ≤ 2^(SUM_W-1) < 2^CMP_W
//   RHS ≤ 31·2^(SUM_W-1) < 2^(SUM_W+4) = 2^(CMP_W-1)  (each partial product
//         sum<<i ≤ 2^(SUM_W-1+i) ≤ 2^(CMP_W-2): no partial or total wrap)
//
// V0.3 F-2 framing guard: upstream silently WRAPS sum_acc when more than N
// scores arrive between s_last pulses (proven in tb_overflow_demo). Here a
// tile-length counter hard-errors instead: on the (N+1)-th score of a tile,
//   - frame_err pulses for exactly 1 cycle, frame_err_sticky sets (CSR-style
//     sticky, cleared only by rst_n or frame_err_clear),
//   - the tile is ABORTED: no decision is emitted for it, remaining scores
//     up to and including its s_last are discarded, accumulators and the
//     counter reset on that s_last — the next tile starts clean.
//
// Interface (upstream-compatible core): feed one score per cycle with
// s_valid; s_last marks the final score of a tile; d_valid pulses exactly one
// cycle after an accepted (non-aborted) s_last beat, with d_fp16 and
// d_max_mag (the tile's max magnitude — feeds tip_importance's bucket).
// No ready on this engine interface (1 score/cycle by construction); stream
// backpressure/skids are tip_top's job (§5/D-006).

module tip_decide #(
  parameter int unsigned SCORE_WIDTH = 32,   // D-011 operating point
  parameter int unsigned BLOCK_M     = 64,
  parameter int unsigned BLOCK_N     = 64,
  parameter int unsigned T_MAX       = 31    // threshold_reg range 1..T_MAX
) (
  input  logic                            clk,
  input  logic                            rst_n,     // synchronous, active-low

  // score stream in (single lane, C-5: raw INT32 pre-softmax scores)
  input  logic                            s_valid,
  input  logic signed [SCORE_WIDTH-1:0]   s_data,
  input  logic                            s_last,

  // per-layer programmable threshold (D-011); 0 is clamped to 1
  input  logic [$clog2(T_MAX+1)-1:0]      threshold,

  // decision out: 1-cycle pulse, one per accepted (non-aborted) tile
  output logic                            d_valid,
  output logic                            d_fp16,    // 1 = FP16, 0 = INT8
  output logic [SCORE_WIDTH-1:0]          d_max_mag, // tile max magnitude

  // V0.3 F-2 framing guard
  output logic                            frame_err,        // 1-cycle pulse
  output logic                            frame_err_sticky,
  input  logic                            frame_err_clear,

  // busy component for tip_top: mid-tile (scores seen, no s_last yet) or abort
  output logic                            tile_active
);

  localparam int unsigned N_TILE = BLOCK_M * BLOCK_N;
  localparam int unsigned LOG2_N = $clog2(N_TILE);
  localparam int unsigned SUM_W  = SCORE_WIDTH + LOG2_N;
  localparam int unsigned T_W    = $clog2(T_MAX + 1);
  localparam int unsigned CMP_W  = SUM_W + T_W;             // D-017
  localparam int unsigned CNT_W  = LOG2_N + 1;              // counts 0..N_TILE

  localparam logic [CNT_W-1:0] CNT_MAX = CNT_W'(N_TILE);

  initial begin
    if (N_TILE != (32'd1 << LOG2_N))
      $fatal(1, "tip_decide: N_TILE=%0d must be a power of two (free-shift LHS)",
             N_TILE);
    if (T_MAX != 31)
      $fatal(1, "tip_decide: T_MAX=%0d unsupported (D-011/D-017 fix T_MAX=31)",
             T_MAX);
  end

  // ── |s| : two's-complement magnitude (abs(INT_MIN) = 2^(W-1), exact) ──────
  logic [SCORE_WIDTH-1:0] u_data, abs_score;
  assign u_data    = unsigned'(s_data);
  assign abs_score = s_data[SCORE_WIDTH-1] ? ((~u_data) + SCORE_WIDTH'(1))
                                           : u_data;

  // ── accumulators + tile-length counter ────────────────────────────────────
  logic [SCORE_WIDTH-1:0] max_acc;
  logic [SUM_W-1:0]       sum_acc;
  logic [CNT_W-1:0]       cnt;
  logic                   abort_tile;

  // combinational next-state includes the current score so the decision is
  // final on the s_last cycle (upstream semantics, V0.3-verified)
  logic [SCORE_WIDTH-1:0] max_next;
  logic [SUM_W-1:0]       sum_next;
  assign max_next = (abs_score > max_acc) ? abs_score : max_acc;
  assign sum_next = sum_acc + {{LOG2_N{1'b0}}, abs_score};

  // ── threshold clamp + programmable shift-add tree (D-017) ─────────────────
  logic [T_W-1:0]   thr_eff;
  assign thr_eff = (threshold == '0) ? T_W'(1) : threshold;

  logic [CMP_W-1:0] lhs, rhs, sum_ext;
  assign lhs     = {{T_W{1'b0}}, max_next, {LOG2_N{1'b0}}};   // max << LOG2_N
  assign sum_ext = {{T_W{1'b0}}, sum_next};

  always_comb begin
    rhs = '0;
    for (int unsigned i = 0; i < T_W; i++)
      if (thr_eff[i]) rhs = rhs + (sum_ext << i);             // T * sum
  end

  // ── framing guard verdicts for the current beat ───────────────────────────
  logic overrun;   // this s_valid beat is the (N_TILE+1)-th score of the tile
  assign overrun = s_valid && !abort_tile && (cnt == CNT_MAX);

  // ── state ─────────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      max_acc          <= '0;
      sum_acc          <= '0;
      cnt              <= '0;
      abort_tile       <= 1'b0;
      d_valid          <= 1'b0;
      d_fp16           <= 1'b0;
      d_max_mag        <= '0;
      frame_err        <= 1'b0;
      frame_err_sticky <= 1'b0;
    end else begin
      d_valid   <= 1'b0;                          // 1-cycle pulses
      frame_err <= 1'b0;
      if (frame_err_clear) frame_err_sticky <= 1'b0;

      if (s_valid) begin
        if (abort_tile) begin
          // discard scores of the erroring tile; recover cleanly on its last
          if (s_last) begin
            abort_tile <= 1'b0;
            max_acc    <= '0;
            sum_acc    <= '0;
            cnt        <= '0;
          end
        end else if (overrun) begin
          // V0.3 F-2: hard error instead of silent sum_acc wrap
          frame_err        <= 1'b1;
          frame_err_sticky <= 1'b1;
          if (s_last) begin                       // erroring tile ends here
            max_acc <= '0;
            sum_acc <= '0;
            cnt     <= '0;
          end else begin
            abort_tile <= 1'b1;
          end
        end else begin
          max_acc <= max_next;
          sum_acc <= sum_next;
          cnt     <= cnt + CNT_W'(1);
          if (s_last) begin
            d_fp16    <= (lhs > rhs);
            d_max_mag <= max_next;
            d_valid   <= 1'b1;
            // reset for the next tile (last-write wins, upstream semantics)
            max_acc <= '0;
            sum_acc <= '0;
            cnt     <= '0;
          end
        end
      end
    end
  end

  assign tile_active = (cnt != '0) || abort_tile;

endmodule
