// seam_job_sva.svh — REUSABLE job-level SVA pack for the D-021 seam blocks
// (descriptor-less variant of verif/common/apex_stream_sva.svh: the MXE pack
// is mxe_desc_t-typed, so it cannot bind to seam_feeder_quant /
// seam_score_dequant, whose "descriptor" is a single count field).
//
// Written for the Verilator 5.x SVA subset (|->, |=>, $stable, $past only),
// compiled under --assert as a build gate (D-012).
//
// Transaction model (apex_stream_sva lineage): a job tracker captures the
// expected POST-SKID output-beat count at the job handshake and counts
// accepted beats. Data equality is the scoreboard TB's job; beat COUNT and
// `last` framing are proven here (drop => done with count < exp or never;
// dup => overrun assertion).
//
// Contract checked (§5 / D-006 / §3):
//   done  => every output beat of the job ACCEPTED post-skid (count == exp)
//   never more than exp beats per job; last exactly on the final beat
//   no output beat outside a legal job (stale-beat / D-019 regression)
//   done needs a job, is a 1-cycle pulse, and excludes job_ready
//   job_ready gated while a job is in flight (D-006)
//   busy while in flight; !busy at done
//   illegal job: error pulse follows the handshake, sticky sets and holds,
//   zero side effects (no busy, no done)
//
// A block with TWO framed output streams (seam_feeder_quant: beats + scales)
// binds two instances — CHECK_JOB=1 on the primary (job/busy/done/error
// properties enabled) and CHECK_JOB=0 on the secondary (count/framing only).

`ifndef SEAM_JOB_SVA_SVH
`define SEAM_JOB_SVA_SVH

module seam_job_sva #(
  parameter int unsigned CNT_W     = 17,
  parameter bit          CHECK_JOB = 1'b1,
  parameter string       NAME      = "seam"
) (
  input logic             clk,
  input logic             rst_n,

  // job interface
  input logic             job_valid,
  input logic             job_ready,
  input logic             job_error,
  input logic             job_error_sticky,
  input logic             busy,
  input logic             done,
  input logic [CNT_W-1:0] exp_beats,   // expected post-skid beats, sampled at accept

  // one post-skid output stream
  input logic             out_valid,
  input logic             out_ready,
  input logic             out_last
);

  // ── transaction model (apex_stream_sva lineage) ───────────────────────────
  logic             accept_pend;   // handshake last cycle; verdict this cycle
  logic             inflight;
  logic [CNT_W-1:0] exp_q;
  logic [CNT_W-1:0] cnt;

  logic job_accept, out_xfer;
  assign job_accept = job_valid && job_ready;
  assign out_xfer   = out_valid && out_ready;

  // consume the job-semantics ports even when CHECK_JOB=0 (secondary-stream
  // instances) so every instance stays -Wall clean without waivers
  logic unused_job_ports;
  assign unused_job_ports = &{1'b0, job_error_sticky, busy};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      accept_pend <= 1'b0;
      inflight    <= 1'b0;
      exp_q       <= '0;
      cnt         <= '0;
    end else begin
      if (accept_pend) begin
        accept_pend <= 1'b0;
        if (!job_error) inflight <= 1'b1;
      end
      if (done) inflight <= 1'b0;
      if (out_xfer) cnt <= cnt + CNT_W'(1);
      if (job_accept) begin
        accept_pend <= 1'b1;
        exp_q       <= exp_beats;
        cnt         <= '0;
      end
    end
  end

  // ── stream/job framing (checked on every instance) ────────────────────────
  ap_done_all_beats_accepted: assert property (@(posedge clk) disable iff (!rst_n)
    done |-> (cnt == exp_q))
    else $error("[SVA D-006] %s: done with %0d/%0d beats accepted post-skid",
                NAME, cnt, exp_q);

  ap_no_beat_overrun: assert property (@(posedge clk) disable iff (!rst_n)
    (out_xfer && inflight) |-> (cnt < exp_q))
    else $error("[SVA D-006] %s: beat overrun (beat %0d of a %0d-beat job)",
                NAME, cnt, exp_q);

  ap_last_iff_final_beat: assert property (@(posedge clk) disable iff (!rst_n)
    (out_xfer && inflight) |-> (out_last == (cnt == exp_q - CNT_W'(1))))
    else $error("[SVA §5] %s: last=%0b on beat %0d of a %0d-beat job",
                NAME, out_last, cnt, exp_q);

  ap_out_only_in_job: assert property (@(posedge clk) disable iff (!rst_n)
    out_valid |-> inflight)
    else $error("[SVA D-019] %s: output beat with no legal job in flight", NAME);

  // ── job-level semantics (primary instance only) ───────────────────────────
  if (CHECK_JOB) begin : g_job

    ap_done_needs_job: assert property (@(posedge clk) disable iff (!rst_n)
      done |-> inflight)
      else $error("[SVA D-006] %s: done pulse with no job in flight", NAME);

    ap_done_pulse_1cyc: assert property (@(posedge clk) disable iff (!rst_n)
      done |=> !done)
      else $error("[SVA D-006] %s: done held more than one cycle", NAME);

    ap_ready_gated: assert property (@(posedge clk) disable iff (!rst_n)
      (inflight || (accept_pend && !job_error)) |-> !job_ready)
      else $error("[SVA D-006] %s: job_ready asserted while a job in flight", NAME);

    ap_done_excludes_ready: assert property (@(posedge clk) disable iff (!rst_n)
      done |-> !job_ready)
      else $error("[SVA D-006] %s: job_ready asserted in the done cycle", NAME);

    ap_busy_while_inflight: assert property (@(posedge clk) disable iff (!rst_n)
      (inflight && !done) |-> busy)
      else $error("[SVA §5] %s: busy low while a job is in flight", NAME);

    ap_done_means_idle: assert property (@(posedge clk) disable iff (!rst_n)
      done |-> !busy)
      else $error("[SVA §5] %s: busy still high at done", NAME);

    ap_err_needs_handshake: assert property (@(posedge clk) disable iff (!rst_n)
      job_error |-> $past(job_valid && job_ready))
      else $error("[SVA §3] %s: job_error pulse without a job handshake", NAME);

    ap_err_no_busy: assert property (@(posedge clk) disable iff (!rst_n)
      job_error |-> (!busy && !inflight))
      else $error("[SVA §3] %s: illegal job caused busy/job activity", NAME);

    ap_err_no_done: assert property (@(posedge clk) disable iff (!rst_n)
      job_error |-> !done)
      else $error("[SVA §3] %s: done pulsed for an illegal job", NAME);

    ap_sticky_sets: assert property (@(posedge clk) disable iff (!rst_n)
      job_error |=> job_error_sticky)
      else $error("[SVA §3] %s: sticky not set after error pulse", NAME);

    ap_sticky_holds: assert property (@(posedge clk) disable iff (!rst_n)
      job_error_sticky |=> job_error_sticky)
      else $error("[SVA §3] %s: sticky cleared without reset", NAME);

  end : g_job

endmodule

`endif // SEAM_JOB_SVA_SVH
