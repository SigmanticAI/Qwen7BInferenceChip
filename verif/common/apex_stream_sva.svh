// apex_stream_sva.svh — REUSABLE SVA property pack for the ARCHITECTURE.md §5
// stream/job contract and D-006 done/busy semantics.
//
// Written for the Verilator 5.x SVA subset: simple concurrent assertions with
// |-> / |=> / $stable / $past only (no multi-clock sequences, no first_match,
// no strong operators). Compiled under --assert as a build gate (§5, D-012).
//
// Usage: `include this file, then bind into the block under test, e.g.:
//
//   bind mxe_top apex_stream_sva u_apex_sva (
//     .clk(clk), .rst_n(rst_n),
//     .desc_valid(desc_valid), .desc_ready(desc_ready), .desc(desc),
//     .desc_error(desc_error), .desc_error_sticky(desc_error_sticky),
//     .busy(busy), .done(done),
//     .act_valid(act_valid), .act_ready(act_ready), .act_beat(act_beat),
//     .wgt_valid(wgt_valid), .wgt_ready(wgt_ready), .wgt_beat(wgt_beat),
//     .res_valid(res_valid), .res_ready(res_ready), .res_beat(res_beat));
//
// Contract checked (IDs referenced in each assertion label):
//   §5    valid/ready with data stable while valid && !ready (all 4 streams)
//   D-006 done ⇒ every result beat of the job ACCEPTED post-skid; one done
//         pulse per accepted legal descriptor; desc_ready for job N+1 gated
//         until strictly after job N's done (closes run1's D-019 wart)
//   §5    busy = (FSM ≠ IDLE) || result beats pending anywhere (skids incl.)
//   §3    illegal descriptor ⇒ desc_error pulse + sticky, ZERO side effects
//
// No-drop/no-dup transaction model: a small job tracker counts POST-SKID
// accepted result beats against the accepted descriptor's m_dim. Data
// equality is the scoreboard TB's job; beat COUNT/order framing is proven
// here (drop ⇒ done fires with count < M or never; dup ⇒ overrun assertion).

`ifndef APEX_STREAM_SVA_SVH
`define APEX_STREAM_SVA_SVH

module apex_stream_sva
  import apex_pkg::*;
(
  input logic         clk,
  input logic         rst_n,

  // descriptor / job interface
  input logic         desc_valid,
  input logic         desc_ready,
  input mxe_desc_t    desc,
  input logic         desc_error,
  input logic         desc_error_sticky,
  input logic         busy,
  input logic         done,

  // input streams
  input logic         act_valid,
  input logic         act_ready,
  input lane8_beat_t  act_beat,
  input logic         wgt_valid,
  input logic         wgt_ready,
  input lane8_beat_t  wgt_beat,

  // result stream (post output skid)
  input logic         res_valid,
  input logic         res_ready,
  input lane32_beat_t res_beat
);

  // ── transaction model ──────────────────────────────────────────────────────
  // accept_pend: a descriptor was accepted last cycle; its legality verdict
  //   (desc_error, registered) is visible THIS cycle.
  // inflight:    a legal job is executing (accept+1 .. its done pulse cycle).
  // exp_m:       m_dim captured at accept — the job's result-beat count.
  // beat_cnt:    POST-SKID accepted result beats of the current job.
  logic                accept_pend;
  logic                inflight;
  logic [DIM_W-1:0]    exp_m;
  logic [DIM_W:0]      beat_cnt;

  logic desc_accept;
  logic res_xfer;
  assign desc_accept = desc_valid && desc_ready;
  assign res_xfer    = res_valid && res_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      accept_pend <= 1'b0;
      inflight    <= 1'b0;
      exp_m       <= '0;
      beat_cnt    <= '0;
    end else begin
      if (accept_pend) begin
        accept_pend <= 1'b0;
        if (!desc_error) inflight <= 1'b1;   // verdict: legal → job in flight
      end
      if (done) inflight <= 1'b0;
      if (res_xfer) beat_cnt <= beat_cnt + {{DIM_W{1'b0}}, 1'b1};
      if (desc_accept) begin                 // (cannot coincide with done:
        accept_pend <= 1'b1;                 //  desc_ready is gated by done)
        exp_m       <= desc.m_dim;
        beat_cnt    <= '0;
      end
    end
  end

  // ── §5: data stability while valid && !ready, no valid retraction ────────
  ap_res_stable: assert property (@(posedge clk) disable iff (!rst_n)
    (res_valid && !res_ready) |=> (res_valid && $stable(res_beat)))
    else $error("[SVA §5] res stream: valid retracted or data unstable under backpressure");

  ap_act_stable: assert property (@(posedge clk) disable iff (!rst_n)
    (act_valid && !act_ready) |=> (act_valid && $stable(act_beat)))
    else $error("[SVA §5] act stream: valid retracted or data unstable while stalled");

  ap_wgt_stable: assert property (@(posedge clk) disable iff (!rst_n)
    (wgt_valid && !wgt_ready) |=> (wgt_valid && $stable(wgt_beat)))
    else $error("[SVA §5] wgt stream: valid retracted or data unstable while stalled");

  ap_desc_stable: assert property (@(posedge clk) disable iff (!rst_n)
    (desc_valid && !desc_ready) |=> (desc_valid && $stable(desc)))
    else $error("[SVA §5] desc: valid retracted or fields unstable while stalled");

  // ── D-006: done ⇒ ALL result beats accepted post-skid (no drop) ───────────
  ap_done_all_beats_accepted: assert property (@(posedge clk) disable iff (!rst_n)
    done |-> (beat_cnt == {1'b0, exp_m}))
    else $error("[SVA D-006] done with %0d/%0d beats accepted post-skid",
                beat_cnt, exp_m);

  // ── no-dup / no-overrun: never more than M result beats per job ──────────
  ap_no_beat_overrun: assert property (@(posedge clk) disable iff (!rst_n)
    (res_xfer && inflight) |-> (beat_cnt < {1'b0, exp_m}))
    else $error("[SVA D-006] result beat overrun: beat %0d of a %0d-beat job",
                beat_cnt, exp_m);

  // `last` exactly on the final beat of the job (frame integrity)
  ap_last_iff_final_beat: assert property (@(posedge clk) disable iff (!rst_n)
    (res_xfer && inflight) |->
      (res_beat.last == (beat_cnt == ({1'b0, exp_m} - {{DIM_W{1'b0}}, 1'b1}))))
    else $error("[SVA §5] res last=%0b on beat %0d of a %0d-beat job",
                res_beat.last, beat_cnt, exp_m);

  // no stale/orphan result beats: res_valid only while a legal job is live.
  // (Strict by design: a stale beat left from a previous/rejected job — the
  //  run1/D-019 failure mode — trips this immediately.)
  ap_res_only_in_job: assert property (@(posedge clk) disable iff (!rst_n)
    res_valid |-> inflight)
    else $error("[SVA D-019] result beat with no legal job in flight (stale beat)");

  // ── D-006: one done per job, done is a 1-cycle pulse ──────────────────────
  ap_done_needs_job: assert property (@(posedge clk) disable iff (!rst_n)
    done |-> inflight)
    else $error("[SVA D-006] done pulse with no job in flight (duplicate done)");

  ap_done_pulse_1cyc: assert property (@(posedge clk) disable iff (!rst_n)
    done |=> !done)
    else $error("[SVA D-006] done held more than one cycle");

  // ── D-006/D-019: desc_ready for job N+1 gated until strictly after done ──
  ap_desc_ready_gated: assert property (@(posedge clk) disable iff (!rst_n)
    (inflight || (accept_pend && !desc_error)) |-> !desc_ready)
    else $error("[SVA D-006] desc_ready asserted while a job is in flight");

  ap_done_excludes_ready: assert property (@(posedge clk) disable iff (!rst_n)
    done |-> !desc_ready)
    else $error("[SVA D-006] desc_ready asserted in the done cycle (not strictly after)");

  // ── §5 busy semantics ─────────────────────────────────────────────────────
  ap_busy_while_inflight: assert property (@(posedge clk) disable iff (!rst_n)
    (inflight && !done) |-> busy)
    else $error("[SVA §5] busy low while a job is in flight");

  ap_done_means_idle: assert property (@(posedge clk) disable iff (!rst_n)
    done |-> !busy)
    else $error("[SVA §5] busy still high at done (beats pending after done?)");

  // ── §3 illegal-descriptor semantics: pulse + sticky, zero side effects ───
  ap_err_needs_handshake: assert property (@(posedge clk) disable iff (!rst_n)
    desc_error |-> $past(desc_valid && desc_ready))
    else $error("[SVA §3] desc_error pulse without a descriptor handshake");

  ap_err_no_busy: assert property (@(posedge clk) disable iff (!rst_n)
    desc_error |-> (!busy && !inflight))
    else $error("[SVA §3] illegal descriptor caused busy/job activity");

  ap_err_no_done: assert property (@(posedge clk) disable iff (!rst_n)
    desc_error |-> !done)
    else $error("[SVA §3] done pulsed for an illegal descriptor");

  ap_sticky_sets: assert property (@(posedge clk) disable iff (!rst_n)
    desc_error |=> desc_error_sticky)
    else $error("[SVA §3] desc_error_sticky not set after error pulse");

  ap_sticky_holds: assert property (@(posedge clk) disable iff (!rst_n)
    desc_error_sticky |=> desc_error_sticky)
    else $error("[SVA §3] desc_error_sticky cleared without reset");

endmodule

`endif // APEX_STREAM_SVA_SVH
