// seq_walker.sv — SEQ (sequencer-lite): descriptor-queue walker FSM.
//
// Implements: ARCHITECTURE.md §1 SEQ row ("fetch descriptor -> dispatch
//             block ops -> barrier -> next"; not a RISC), §5 stream/job
//             semantics, D-006 (SEQ serializes jobs: descriptor N+1 is
//             NEVER presented before job N's done), §7 CTRL.soft_reset
//             abort semantics (drain the queue, never violate §5 on the
//             in-flight job — wait for its done/desc_error).
//
// Structure:
//   ds_* (desc stream in, §5 valid/ready) -> stream_skid (V0.5-verified,
//   D-019) -> 8-deep descriptor FIFO -> walker FSM -> md_* (MXE descriptor
//   interface out) + mxe_done/mxe_desc_error job status back.
//
// Walker FSM (one job at a time — the v0.1 "barrier" IS job serialization):
//   S_IDLE : if enable && queue non-empty && not aborting: pop the head
//            into a hold register and go present it.
//   S_REQ  : md_valid high, payload from the hold register (stable, never
//            retracted — §5). On md_ready accept -> S_WAIT.
//   S_WAIT : job in flight. Wait mxe_done (legal job, post-skid-acceptance
//            per D-006) OR mxe_desc_error (rejected descriptor: consumed,
//            no state change, NO done — §3). Then -> S_IDLE.
//   md_valid can therefore never assert while a job is in flight: D-006
//   holds by construction (and is checked by verif/seq SVA + the TB stub).
//
// Soft abort (CSR CTRL.soft_reset -> abort_req, a 1-cycle pulse; latched
// here because the abort can outlast the pulse):
//   - the FIFO is cleared IMMEDIATELY (queued, not-yet-dispatched
//     descriptors are discarded — abort-and-discard, D-020 lineage);
//   - descriptor beats arriving on ds_* while aborting are accepted and
//     DISCARDED (the §5 handshake is never violated; nothing stalls).
//     This holds for EVERY acceptance timing: the input skid has one cycle
//     of latency, so a beat whose ds_* handshake completes in the FINAL
//     cycle of the abort window is still inside the skid when the abort
//     deasserts. Such beats are tracked (skid_drop counter) and discarded
//     deterministically as they emerge — they are never pushed into the
//     FIFO and never dispatched (abort-window-leak fix, audit 2026-07-08);
//   - a descriptor already PRESENTED on md_* (S_REQ) is never retracted:
//     it completes the handshake and becomes the in-flight job;
//   - the in-flight job is never aborted: SEQ waits for its done /
//     desc_error, then `aborting` deasserts. busy falls only when the
//     block is truly empty.
//
// enable (CSR CTRL.enable): gates NEW dispatches only. enable=0 holds the
// queue (beats still accepted until full) and lets the in-flight job finish.
//
// desc_error out: mxe_desc_error forwarded (gated to S_WAIT), for the CSR
// STATUS sticky bit. Queue status (q_empty/q_full/q_count) is FIFO-proper;
// the input skid can hold up to 2 further beats (total capacity QDEPTH+2)
// — skid occupancy is folded into busy, not q_count.
//
// busy (§5): FSM != IDLE, or any descriptor pending anywhere (FIFO, input
// skid), or an abort still draining.

module seq_walker
  import apex_pkg::*;
#(
  parameter int unsigned QDEPTH = 8            // descriptor FIFO depth (power of 2)
) (
  input  logic                       clk,
  input  logic                       rst_n,    // synchronous, active-low

  // CSR controls
  input  logic                       enable,     // CTRL.enable (level)
  input  logic                       abort_req,  // CTRL.soft_reset (pulse)

  // descriptor stream in (§5, via stream_skid)
  input  logic                       ds_valid,
  output logic                       ds_ready,
  input  mxe_desc_t                  ds_desc,

  // MXE descriptor interface out (mxe_top desc port group)
  output logic                       md_valid,
  input  logic                       md_ready,
  output mxe_desc_t                  md_desc,

  // MXE job status back (mxe_top)
  input  logic                       mxe_done,       // pulse: legal job complete (D-006)
  input  logic                       mxe_desc_error, // pulse: descriptor rejected (§3)

  // status to CSR
  output logic                       desc_error,     // forwarded 1-cycle pulse
  output logic                       q_empty,
  output logic                       q_full,
  output logic [$clog2(QDEPTH+1)-1:0] q_count,
  output logic                       busy,
  output logic                       aborting
);

  localparam int unsigned DW    = $bits(mxe_desc_t);
  localparam int unsigned PTR_W = $clog2(QDEPTH);
  localparam int unsigned CNT_W = $clog2(QDEPTH + 1);

  if (QDEPTH < 2 || (QDEPTH & (QDEPTH - 1)) != 0) begin : gen_qdepth_err
    $error("seq_walker: QDEPTH must be a power of 2, >= 2");
  end

  // ── input skid (§5: skid on every external stream boundary) ──────────────
  logic          in_m_valid, in_m_ready;
  logic [DW-1:0] in_m_data;

  stream_skid #(.WIDTH(DW)) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (ds_valid),
    .s_ready (ds_ready),
    .s_data  (DW'(ds_desc)),
    .m_valid (in_m_valid),
    .m_ready (in_m_ready),
    .m_data  (in_m_data)
  );

  // ── descriptor FIFO ────────────────────────────────────────────────────────
  logic [DW-1:0]    mem [QDEPTH];
  logic [PTR_W-1:0] wr_ptr, rd_ptr;
  logic [CNT_W-1:0] count;

  // ── walker FSM ─────────────────────────────────────────────────────────────
  typedef enum logic [1:0] { S_IDLE = 2'd0, S_REQ = 2'd1, S_WAIT = 2'd2 } state_e;
  state_e        state;
  logic [DW-1:0] head_q;          // presented descriptor hold register
  logic          abort_active;

  logic abort_eff, job_end, dispatch, push;
  assign abort_eff = abort_active || abort_req;
  assign job_end   = (state == S_WAIT) && (mxe_done || mxe_desc_error);
  assign dispatch  = (state == S_IDLE) && !abort_eff && enable && (count != '0);

  // ── abort-window skid poison tracking ─────────────────────────────────────
  // u_in_skid has 1 cycle of latency: a beat accepted on ds_* in the FINAL
  // cycle of the abort window emerges on in_m_* only AFTER abort_eff has
  // fallen. Gating push on abort_eff alone therefore leaks such beats into
  // the FIFO (audit finding, verif/audit_seq). Track skid occupancy and how
  // many resident beats belong to the abort window (skid_drop): while the
  // abort is in effect EVERY resident beat is poisoned; afterwards the
  // residue is discarded, in FIFO order, as it emerges. Invariants (by
  // construction of stream_skid): skid_occ ∈ {0,1,2}; skid_occ == 2 ⇒
  // ds_ready == 0; skid_occ == 0 ⇒ in_m_valid == 0; skid_drop <= skid_occ.
  logic [1:0] skid_occ;       // beats resident in u_in_skid
  logic [1:0] skid_drop;      // resident beats that must be discarded
  logic       s_acc, m_acc;
  logic [1:0] skid_occ_nxt;
  assign s_acc        = ds_valid && ds_ready;
  assign m_acc        = in_m_valid && in_m_ready;
  assign skid_occ_nxt = skid_occ + 2'(s_acc) - 2'(m_acc);

  // during an abort, arriving beats are accepted and discarded (never stall);
  // a post-abort poisoned residue drains unconditionally (deterministic)
  assign in_m_ready = abort_eff || (skid_drop != '0) || (count != CNT_W'(QDEPTH));
  assign push       = in_m_valid && in_m_ready && !abort_eff && (skid_drop == '0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      head_q       <= '0;
      wr_ptr       <= '0;
      rd_ptr       <= '0;
      count        <= '0;
      abort_active <= 1'b0;
      skid_occ     <= '0;
      skid_drop    <= '0;
    end else begin
      // skid occupancy / abort-window poison bookkeeping
      skid_occ <= skid_occ_nxt;
      if (abort_eff)                       skid_drop <= skid_occ_nxt;
      else if (m_acc && (skid_drop != '0)) skid_drop <= skid_drop - 2'd1;

      // FIFO push (mem write is unconditional-index, gated by push)
      if (push) begin
        mem[wr_ptr] <= in_m_data;
        wr_ptr      <= wr_ptr + PTR_W'(1);
      end

      // FSM + pop
      unique case (state)
        S_IDLE: if (dispatch) begin
          head_q <= mem[rd_ptr];
          rd_ptr <= rd_ptr + PTR_W'(1);
          state  <= S_REQ;
        end
        S_REQ:  if (md_ready) state <= S_WAIT;   // md_valid is high in S_REQ
        S_WAIT: if (mxe_done || mxe_desc_error) state <= S_IDLE;
        default: state <= S_IDLE;
      endcase

      // occupancy
      unique case ({push, dispatch})
        2'b10:   count <= count + CNT_W'(1);
        2'b01:   count <= count - CNT_W'(1);
        default: ;                                // 2'b00 and 2'b11: net 0
      endcase

      // abort: clear the queue now, drain the in-flight job, then release
      if (abort_eff) begin
        count  <= '0;
        wr_ptr <= '0;
        rd_ptr <= '0;
      end
      if (abort_req) abort_active <= 1'b1;
      else if (abort_active && (state == S_IDLE || job_end))
        abort_active <= 1'b0;
    end
  end

  // ── outputs ────────────────────────────────────────────────────────────────
  assign md_valid   = (state == S_REQ);
  assign md_desc    = mxe_desc_t'(head_q);
  assign desc_error = (state == S_WAIT) && mxe_desc_error;
  assign q_empty    = (count == '0);
  assign q_full     = (count == CNT_W'(QDEPTH));
  assign q_count    = count;
  assign aborting   = abort_active;
  assign busy       = (state != S_IDLE) || (count != '0) || in_m_valid
                    || abort_active;

endmodule
