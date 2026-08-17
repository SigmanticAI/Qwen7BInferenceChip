// =============================================================================
// File        : stream_skid.sv
// Module      : stream_skid
// -----------------------------------------------------------------------------
// PURPOSE
//   Generic, reusable 2-register "skid buffer" for AXI-Stream-like valid/ready
//   channels. Provides fully registered valid/data on both the upstream
//   (slave/subordinate, s_*) and downstream (master, m_*) interfaces so that:
//     - s_ready is a pure register output (never combinationally derived from
//       s_valid or m_valid), avoiding any valid->ready->valid combinational
//       loop / deadlock risk.
//     - Once asserted, m_valid and m_data are held stable, unchanged, until
//       the cycle in which m_ready is observed high (a transfer occurs).
//     - Full throughput (one accepted transfer per clock, zero bubbles) is
//       achieved whenever the downstream consumer holds m_ready high.
//     - Backpressure from downstream (m_ready low) is absorbed for one
//       additional in-flight beat via the internal skid register, then
//       correctly propagated upstream via s_ready deasserting.
//
//   Intended for use at channel boundaries of the INT8 GEMM systolic
//   accelerator core (gemm_systolic_accel), e.g. on the result_out channel,
//   and optionally on input streaming channels, at integration time. This
//   module implements NO FSM, datapath, or storage/buffer semantics of the
//   accelerator itself -- it is a protocol-only building block.
//
// PARAMETERS
//   WIDTH : int, default 8 -- bit width of the data payload (s_data/m_data).
//           Sized by the integrator to match the channel being registered
//           (e.g. WIDTH=8 for INT8 lanes, or wider for packed/accumulated
//           result words).
//
// PORTS
//   clk      : input  -- system clock, posedge-triggered, unchanged/passed
//                         through, not gated or divided by this module.
//   rst_n    : input  -- synchronous active-low reset. While rst_n==0, all
//                         internal state (m_valid, skid occupancy) is
//                         cleared on the next posedge clk; s_ready returns
//                         to the empty-buffer (asserted) state.
//   s_valid  : input  -- upstream producer asserts data validity.
//   s_ready  : output -- registered; this module can accept s_data. Not a
//                         function of s_valid (REQ-PROTO-003).
//   s_data   : input  -- upstream payload, sampled when s_valid&&s_ready.
//   m_valid  : output -- registered; asserted data is valid and stable.
//   m_ready  : input  -- downstream consumer can accept m_data this cycle.
//   m_data   : output -- registered downstream payload, WIDTH bits.
//
// REQUIREMENT TRACEABILITY (owned by this module)
//   REQ-GEMM-009  : Backpressure/handshake compliance at channel boundary.
//   REQ-PROTO-002 : Asserted valid + payload held stable/unchanged until the
//                    receiving side asserts ready (no data/valid retraction).
//   REQ-PROTO-003 : ready is not combinationally dependent on valid (no
//                    ready<->valid combinational loop / deadlock hazard).
//   REQ-TIME-001/002 (partial): registered, single-cycle-granularity
//                    handshake timing; deterministic 1-cycle pipeline delay.
//
// THROUGHPUT / LATENCY CHARACTERISTICS (for integration)
//   - Latency    : 1 clock cycle from an accepted upstream beat
//                   (s_valid&&s_ready at cycle N) to that beat becoming
//                   visible on m_data/m_valid (cycle N+1), when the skid
//                   buffer is empty.
//   - Throughput : Sustained 1 accepted transfer per clock cycle (no
//                   bubbles) as long as m_ready is held high by the
//                   consumer. This is a non-blocking (fully pipelined)
//                   skid register, not a store-and-forward FIFO.
//   - Backpressure capacity: up to 2 beats may be in flight inside this
//                   module (1 in the primary/output register, 1 in the
//                   internal skid register) before s_ready deasserts.
//   - Reset      : m_valid deasserts synchronously; s_ready reasserts
//                   (buffer empty) on the same edge that clears state.
//
// IMPLEMENTATION NOTES
//   Standard "double-buffered" skid design:
//     - Primary register (m_data/m_valid) drives the downstream output and
//       is only updated when it is safe to do so: either the consumer is
//       accepting the current beat (m_ready&&m_valid) or the primary
//       register is currently empty (!m_valid). This is exactly the
//       condition required to guarantee REQ-PROTO-002 stability.
//     - Secondary/skid register (skid_data/skid_valid) absorbs one
//       additional upstream beat when the primary register is stalled
//       (m_valid&&!m_ready) and the producer still has data to offer.
//     - s_ready = ~skid_valid : purely registered, depends only on the
//       skid occupancy flag, never on s_valid or m_valid combinationally.
//   No latches; all state is in always_ff blocks with complete
//   if/else coverage (no inferred latches).
// =============================================================================

module stream_skid #(
    parameter int WIDTH = 8
) (
    input  logic             clk,
    input  logic             rst_n,        // synchronous active-low

    // upstream (producer -> skid)
    input  logic             s_valid,
    output logic             s_ready,
    input  logic [WIDTH-1:0] s_data,

    // downstream (skid -> consumer)
    output logic             m_valid,
    input  logic             m_ready,
    output logic [WIDTH-1:0] m_data
);

    // Internal skid (backup) register occupancy/data.
    logic             skid_valid;
    logic [WIDTH-1:0] skid_data;

    // s_ready is a pure register output: asserted whenever the skid
    // (backup) register is empty, i.e. this module has at least one free
    // slot to accept a new upstream beat. Never a function of s_valid or
    // m_valid -> satisfies REQ-PROTO-003 and avoids any combinational
    // ready/valid loop.
    assign s_ready = ~skid_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_valid    <= 1'b0;
            skid_valid <= 1'b0;
            m_data     <= '0;
            skid_data  <= '0;
        end else begin
            if (m_ready || !m_valid) begin
                // Primary/output register may be updated this cycle:
                // either the current beat is being consumed (m_ready), or
                // the output register is currently empty.
                if (skid_valid) begin
                    // Drain the skid register into the primary output.
                    m_data     <= skid_data;
                    m_valid    <= 1'b1;
                    skid_valid <= 1'b0;
                end else if (s_valid && s_ready) begin
                    // No skid backlog: accept directly from upstream.
                    m_data  <= s_data;
                    m_valid <= 1'b1;
                end else begin
                    // Nothing to offer: output goes/stays empty.
                    m_valid <= 1'b0;
                end
            end else begin
                // m_valid==1 && m_ready==0: consumer not ready. Primary
                // output register MUST hold m_data/m_valid unchanged here
                // (REQ-PROTO-002). If the producer still has data and the
                // skid register is free, absorb one extra beat into skid.
                if (s_valid && s_ready) begin
                    skid_data  <= s_data;
                    skid_valid <= 1'b1;
                end
            end
        end
    end

endmodule
