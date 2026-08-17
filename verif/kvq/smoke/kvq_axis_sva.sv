// kvq_axis_sva.sv — ARCHITECTURE.md §5 stream-contract SVA for kvq_engine,
// bound into every KVQ smoke build (D-012: SVA compiled as a build gate).
//
// WHY NOT verif/common/apex_stream_sva.svh DIRECTLY: the house pack is shaped
// around the MXE descriptor/job contract (mxe_desc_t, done/busy, desc_error) —
// none of which exists on the KVQ engine's AXI-Stream + AXI-Lite surface.
// Binding it with tied-off descriptor signals would make every job property
// vacuous and trip ap_res_only_in_job on legitimate read bursts: dishonest
// coverage. This checker lifts the pack's §5 stream properties (same rules,
// same labels, same Verilator-subset style: |->, |=>, $stable only) and adds
// the KVQ burst-framing rule (tlast exactly on beat D-1 of every read burst).
//
// Bound below into kvq_engine, so it rides along in ALL kvq builds that
// compile this file — parity, stall, and the D-016/D-008 regressions.

`default_nettype none

module kvq_axis_sva #(
    parameter int D  = 64,
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
    input wire          m_last
);

  // ── §5: data stability + no valid retraction while stalled ────────────────
  // Assertions are gated by sva_en, a locally derived async-reset flop, rather
  // than sampling rst_n at the clock edge (`disable iff` or an antecedent read
  // would count as a SYNC use of the vendored ASYNC reset net — SYNCASYNCNET).
  // sva_en rises one cycle after reset release; rst_n never re-asserts mid-run
  // in these TBs, so the checked window is identical in practice.
  logic sva_en;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) sva_en <= 1'b0;
    else        sva_en <= 1'b1;
  end

  ap_m_stable: assert property (@(posedge clk)
    (sva_en && m_valid && !m_ready) |=> (m_valid && $stable(m_data) && $stable(m_last)))
    else $error("[SVA §5] m_axis: valid retracted or data/last unstable under backpressure");

  ap_s_stable: assert property (@(posedge clk)
    (sva_en && s_valid && !s_ready) |=> (s_valid && $stable(s_data) && $stable(s_last) && $stable(s_user)))
    else $error("[SVA §5] s_axis: valid retracted or data/last/user unstable while stalled");

  // ── read-burst framing: every burst is exactly D beats, tlast on the last ─
  logic [31:0] mcnt;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)                    mcnt <= '0;
    else if (m_valid && m_ready)   mcnt <= m_last ? '0 : mcnt + 32'd1;
  end

  ap_m_last_position: assert property (@(posedge clk)
    (sva_en && m_valid && m_ready) |-> (m_last == (mcnt == 32'(D - 1))))
    else $error("[SVA KVQ] m_axis tlast=%0b on beat %0d of a %0d-beat burst",
                m_last, mcnt, D);

endmodule

// ride along in every kvq_engine instantiation of the builds compiling this file
bind kvq_engine kvq_axis_sva #(
    .D(VECTOR_DIM), .DW(COORD_WIDTH), .OW(OUT_WIDTH)
) u_kvq_axis_sva (
    .clk(clk), .rst_n(rst_n),
    .s_valid(s_axis_kv_tvalid), .s_ready(s_axis_kv_tready),
    .s_data(s_axis_kv_tdata), .s_last(s_axis_kv_tlast), .s_user(s_axis_kv_tuser),
    .m_valid(m_axis_kv_tvalid), .m_ready(m_axis_kv_tready),
    .m_data(m_axis_kv_tdata), .m_last(m_axis_kv_tlast)
);

`default_nettype wire
