// tip_sb_sva.svh — INDEPENDENT TIP contract checker (verification suite).
//
// Scope: §5/D-006 semantics specialized to TIP's descriptor-less streaming
// shape, plus a decision conservation model (no drop / no dup / no orphan)
// and the V0.3 F-2 framing-guard CSR semantics. The per-stream §5 stability
// core is NOT duplicated here — it is reused from the house pack via
// verif/common/apex_stream1_sva.svh, bound onto both TIP streams by the TB.
//
// White-box: bound into tip_top, so ports connect to tip_top-scope signals
// (eng_d_valid, eng_fire, in_last, in_m_valid, eng_tile_active, hold_valid,
// dec_pend) in addition to the block's external ports. The conservation
// model counts decisions PRODUCED by the engine (eng_d_valid pulse) against
// decisions ACCEPTED downstream post-skid (d_valid && d_ready): a dropped
// beat leaves pending stuck > 0 (caught by ap_pending_busy + the TB's drain
// checks); a duplicated/orphan beat trips ap_d_needs_pending immediately.
//
// Written for the Verilator-5.x SVA subset only: simple concurrent
// assertions, |->, |=>, $stable. Compiled under --assert (D-012 gate).

`ifndef TIP_SB_SVA_SVH
`define TIP_SB_SVA_SVH

module tip_sb_sva
  import apex_pkg::*;
(
  input logic       clk,
  input logic       rst_n,

  // decision stream (post-skid view)
  input logic       d_valid,
  input logic       d_ready,
  input kvq_tier_e  d_tier,

  // framing guard CSR
  input logic       frame_err,
  input logic       frame_err_sticky,
  input logic       frame_err_clear,

  // §5 busy
  input logic       busy,

  // white-box taps (tip_top internals)
  input logic       eng_d_valid,     // decision produced by tip_decide
  input logic       eng_fire,        // beat consumed from the input skid
  input logic       in_last,         // ... and it was a tile-end beat
  input logic       in_m_valid,      // input skid occupied
  input logic       eng_tile_active, // tip_decide mid-tile or aborting
  input logic       hold_valid,      // decision hold register occupied
  input logic       dec_pend         // s_last consumed last cycle
);

  // ── conservation model: produced vs delivered decisions ───────────────────
  // capacity: 1 in hold + 2 in the output skid = 3 in flight max
  logic [2:0] pending;
  logic       d_xfer;
  assign d_xfer = d_valid && d_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      pending <= '0;
    end else begin
      pending <= pending + {2'b0, eng_d_valid} - {2'b0, d_xfer};
    end
  end

  ap_pending_capacity: assert property (@(posedge clk) disable iff (!rst_n)
    pending <= 3'd3)
    else $error("[SVA TIP] >3 decisions in flight (pending=%0d) — dup?", pending);

  ap_d_needs_pending: assert property (@(posedge clk) disable iff (!rst_n)
    d_xfer |-> (pending != '0))
    else $error("[SVA TIP] decision beat delivered with none in flight (orphan/dup)");

  ap_pending_busy: assert property (@(posedge clk) disable iff (!rst_n)
    (pending != '0) |-> busy)
    else $error("[SVA TIP] %0d decision(s) in flight but busy low", pending);

  // ── feed-gate contract: hold register can never be overwritten ────────────
  ap_hold_never_overwritten: assert property (@(posedge clk) disable iff (!rst_n)
    eng_d_valid |-> !hold_valid)
    else $error("[SVA TIP] decision produced while hold register occupied (lost beat)");

  ap_gate_after_last: assert property (@(posedge clk) disable iff (!rst_n)
    (eng_fire && in_last) |=> !eng_fire)
    else $error("[SVA TIP] feed not gated in the cycle after an s_last consume");

  // ── §5 busy composition: every occupied resource implies busy ─────────────
  ap_busy_in_skid:  assert property (@(posedge clk) disable iff (!rst_n)
    in_m_valid |-> busy)
    else $error("[SVA §5] input-skid beat pending but busy low");

  ap_busy_tile:     assert property (@(posedge clk) disable iff (!rst_n)
    eng_tile_active |-> busy)
    else $error("[SVA §5] engine mid-tile but busy low");

  ap_busy_hold:     assert property (@(posedge clk) disable iff (!rst_n)
    (hold_valid || dec_pend) |-> busy)
    else $error("[SVA §5] decision held/pending but busy low");

  ap_busy_dvalid:   assert property (@(posedge clk) disable iff (!rst_n)
    d_valid |-> busy)
    else $error("[SVA §5] output beat pending (post-skid) but busy low");

  ap_idle_is_idle:  assert property (@(posedge clk) disable iff (!rst_n)
    !busy |-> (!d_valid && pending == '0 && !in_m_valid))
    else $error("[SVA §5] busy low with work in flight");

  // ── decision payload legality ──────────────────────────────────────────────
  ap_tier_legal: assert property (@(posedge clk) disable iff (!rst_n)
    d_valid |-> (2'(d_tier) != 2'b11))
    else $error("[SVA TIP] d_tier=3 is not a kvq_tier_e value");

  // ── V0.3 F-2 framing guard: pulse + sticky CSR semantics ──────────────────
  ap_err_pulse: assert property (@(posedge clk) disable iff (!rst_n)
    frame_err |=> !frame_err)
    else $error("[SVA F-2] frame_err held more than one cycle");

  ap_err_sets_sticky: assert property (@(posedge clk) disable iff (!rst_n)
    frame_err |=> frame_err_sticky)
    else $error("[SVA F-2] sticky not set after frame_err");

  ap_sticky_holds: assert property (@(posedge clk) disable iff (!rst_n)
    (frame_err_sticky && !frame_err_clear) |=> frame_err_sticky)
    else $error("[SVA F-2] sticky dropped without frame_err_clear");

  ap_sticky_clears: assert property (@(posedge clk) disable iff (!rst_n)
    (frame_err_clear && !frame_err) |=> (!frame_err_sticky || frame_err))
    else $error("[SVA F-2] frame_err_clear did not clear sticky");

  // ── no spurious errors: frame_err only right after a consumed beat ────────
  ap_err_needs_beat: assert property (@(posedge clk) disable iff (!rst_n)
    frame_err |-> $past(eng_fire))
    else $error("[SVA F-2] frame_err with no consumed score beat");

endmodule

`endif // TIP_SB_SVA_SVH
