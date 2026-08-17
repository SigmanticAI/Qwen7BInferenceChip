// tip_sva.svh — TIP §5-stream + framing-guard SVA checker, bound into
// tip_top (D-012: assertions compiled in every build, --assert gate).
//
// NOTE on reuse: the house pack verif/common/apex_stream_sva.svh is
// descriptor/job-shaped (desc/done/busy + act/wgt/res streams of the MXE
// contract) and cannot bind onto TIP's descriptor-less streaming interface.
// This checker carries over its §5 core — valid/ready with no retraction and
// data stable while valid && !ready — onto TIP's two streams, plus the
// D-017/V0.3-F-2 framing-guard sticky semantics. Verilator SVA subset only
// (simple concurrent assertions, |->, |=>, $stable).

`ifndef TIP_SVA_SVH
`define TIP_SVA_SVH

module tip_sva
  import apex_pkg::*;
#(
  parameter int unsigned SCORE_WIDTH = 32,
  parameter int unsigned BLOCKS      = 128
) (
  input logic                          clk,
  input logic                          rst_n,

  // score stream in
  input logic                          s_valid,
  input logic                          s_ready,
  input logic signed [SCORE_WIDTH-1:0] s_data,
  input logic                          s_last,
  input logic [$clog2(BLOCKS)-1:0]     s_blk,

  // decision stream out
  input logic                          d_valid,
  input logic                          d_ready,
  input logic                          d_fp16,
  input kvq_tier_e                     d_tier,
  input logic [$clog2(BLOCKS)-1:0]     d_blk,

  // framing guard
  input logic                          frame_err,
  input logic                          frame_err_sticky,
  input logic                          frame_err_clear,

  input logic                          busy
);

  // ── §5: score stream — no retraction, data stable under backpressure ─────
  // (checks the producer side of the contract; the TB driver must honor it)
  ap_s_stable: assert property (@(posedge clk) disable iff (!rst_n)
    (s_valid && !s_ready) |=>
      (s_valid && $stable(s_data) && $stable(s_last) && $stable(s_blk)))
    else $error("[SVA §5] score stream: valid retracted or data unstable while stalled");

  // ── §5: decision stream — no retraction, data stable under backpressure ──
  ap_d_stable: assert property (@(posedge clk) disable iff (!rst_n)
    (d_valid && !d_ready) |=>
      (d_valid && $stable(d_fp16) && $stable(d_tier) && $stable(d_blk)))
    else $error("[SVA §5] decision stream: valid retracted or data unstable while stalled");

  // ── D-006 flavor: a pending decision beat implies busy ────────────────────
  ap_dvalid_busy: assert property (@(posedge clk) disable iff (!rst_n)
    d_valid |-> busy)
    else $error("[SVA §5] decision beat pending (post-skid) but busy low");

  // ── V0.3 F-2 framing guard: pulse + sticky CSR semantics ─────────────────
  ap_err_pulse_1cyc: assert property (@(posedge clk) disable iff (!rst_n)
    frame_err |=> !frame_err)
    else $error("[SVA F-2] frame_err held more than one cycle");

  ap_err_sets_sticky: assert property (@(posedge clk) disable iff (!rst_n)
    frame_err |=> frame_err_sticky)
    else $error("[SVA F-2] frame_err_sticky not set after error pulse");

  ap_sticky_holds: assert property (@(posedge clk) disable iff (!rst_n)
    (frame_err_sticky && !frame_err_clear) |=> frame_err_sticky)
    else $error("[SVA F-2] frame_err_sticky cleared without frame_err_clear");

  // (a clear coinciding with a NEW overrun loses to the error — set wins;
  //  that new error is visible as next cycle's frame_err pulse)
  ap_sticky_clears: assert property (@(posedge clk) disable iff (!rst_n)
    (frame_err_clear && !frame_err) |=> (!frame_err_sticky || frame_err))
    else $error("[SVA F-2] frame_err_clear did not clear the sticky bit");

endmodule

`endif // TIP_SVA_SVH
