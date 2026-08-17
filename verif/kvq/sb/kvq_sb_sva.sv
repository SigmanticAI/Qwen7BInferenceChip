// kvq_sb_sva.sv — ARCHITECTURE.md §5 stream-contract SVA for kvq_engine,
// bound into every verif/kvq/sb build (D-012: SVA compiled as a build gate).
//
// The house pack verif/common/apex_stream_sva.svh is descriptor/job-shaped
// (mxe_desc_t, done/busy, desc_error) — none of which exists on KVQ's
// AXI-Stream + AXI-Lite surface; binding it tied-off would be vacuous
// (dishonest coverage, same call the smoke suite made). This pack lifts the
// same §5 stream rules in the same Verilator-subset style (|->, |=>, $stable)
// and adds two KVQ-shaped rules:
//   * burst framing: tlast exactly on beat D-1 of every read burst
//   * output beats only while the engine FSM is in ST_OUTPUT/ST_OFLUSH
//     (bound to the internal state — catches stale/orphan beats, e.g. beats
//     leaking out after a soft reset)
//
// Violations $display "SVA-VIOL" and count, instead of $error/abort, so
// expected-failure runs (mutations, soft-reset probes) complete and report;
// the Makefile grep-gates "SVA-VIOL" out of every functional run.

`default_nettype none

module kvq_sb_sva #(
    parameter int D  = 16,
    parameter int DW = 16,
    parameter int OW = 32
) (
    input wire          clk,
    input wire          rst_n,

    // s_axis (fp16 token write stream — DUT is the consumer)
    input wire          s_valid,
    input wire          s_ready,
    input wire [DW-1:0] s_data,
    input wire          s_last,
    input wire          s_user,

    // m_axis (fp32 decompressed read stream — DUT is the producer)
    input wire          m_valid,
    input wire          m_ready,
    input wire [OW-1:0] m_data,
    input wire          m_last,

    // engine FSM state (bind connects the internal reg)
    input wire [3:0]    fsm_state
);

  int viol_m_stable;
  int viol_s_stable;
  int viol_last_pos;
  int viol_state;

  // async-reset enable flop: rst_n is never sampled at the clock edge
  // (SYNCASYNCNET-safe, same idiom as the smoke pack); rises one cycle after
  // reset release.
  logic sva_en;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) sva_en <= 1'b0;
    else        sva_en <= 1'b1;
  end

  // ── §5: data stability + no valid retraction while stalled ────────────────
  ap_m_stable: assert property (@(posedge clk)
    (sva_en && m_valid && !m_ready) |=>
      (m_valid && $stable(m_data) && $stable(m_last)))
    else begin
      viol_m_stable++;
      $display("SVA-VIOL [ap_m_stable] m_axis valid retracted or data/last unstable under backpressure");
    end

  ap_s_stable: assert property (@(posedge clk)
    (sva_en && s_valid && !s_ready) |=>
      (s_valid && $stable(s_data) && $stable(s_last) && $stable(s_user)))
    else begin
      viol_s_stable++;
      $display("SVA-VIOL [ap_s_stable] s_axis valid retracted or data/last/user unstable while stalled (t=%0t v=%b r=%b d=%h l=%b u=%b)",
               $time, s_valid, s_ready, s_data, s_last, s_user);
    end

  // ── burst framing: every read burst is exactly D beats, tlast on the last ─
  logic [31:0] mcnt;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)                  mcnt <= '0;
    else if (m_valid && m_ready) mcnt <= m_last ? '0 : mcnt + 32'd1;
  end

  ap_m_last_pos: assert property (@(posedge clk)
    (sva_en && m_valid && m_ready) |-> (m_last == (mcnt == 32'(D - 1))))
    else begin
      viol_last_pos++;
      $display("SVA-VIOL [ap_m_last_pos] tlast=%0b on beat %0d of a %0d-beat burst",
               m_last, mcnt, D);
    end

  // ── output beats only from the output phases (stale/orphan-beat catcher) ──
  ap_m_only_output_phase: assert property (@(posedge clk)
    (sva_en && m_valid) |-> (fsm_state == 4'd6 || fsm_state == 4'd11))
    else begin
      viol_state++;
      $display("SVA-VIOL [ap_m_only_output_phase] m_valid while FSM state=%0d (not OUTPUT/OFLUSH)",
               fsm_state);
    end

endmodule

// ride along in every kvq_engine instantiation of builds compiling this file
bind kvq_engine kvq_sb_sva #(
    .D(VECTOR_DIM), .DW(COORD_WIDTH), .OW(OUT_WIDTH)
) u_sb_sva (
    .clk(clk), .rst_n(rst_n),
    .s_valid(s_axis_kv_tvalid), .s_ready(s_axis_kv_tready),
    .s_data(s_axis_kv_tdata), .s_last(s_axis_kv_tlast),
    .s_user(s_axis_kv_tuser),
    .m_valid(m_axis_kv_tvalid), .m_ready(m_axis_kv_tready),
    .m_data(m_axis_kv_tdata), .m_last(m_axis_kv_tlast),
    .fsm_state(state)
);

`default_nettype wire
