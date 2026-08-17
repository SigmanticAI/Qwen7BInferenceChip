// apex_fuel_ctl.sv — IB-FUEL s3: tile-domain fuel controller. Owns the
// FUEL CSR state (decoded by the cl_apex bridge FSM, window 0x0 regs
// 0x20-0x30 — IB_FUEL.md §5), the 2-deep fuel_req ingress skid (the
// D-029-compatible reading of IB-WALK's FIFO>=2 reservation: ACCEPT-ahead,
// processing strictly serial), the tile half of the 4-phase crossing to
// apex_fuel_reader, the afifo read side toward the xw mux, and the sticky
// error bits.
//
// ── TWO RESETS (load-bearing — same rule as apex_afifo) ────────────────────
//   rst_glb_n : tile-domain GLOBAL reset only. Clocks the 4-phase toggle
//               state and the status synchronizers. NEVER gated by the
//               bridge's TILE_RST — resetting req_tgl while the shell's
//               ack parity survives would poison the handshake exactly
//               like resetting one side of the afifo.
//   rst_csr_n : rst_glb_n & ~TILE_RST. Clocks the CSR registers, the skid,
//               last_tag, and the sticky errors — per-file executor parity
//               (each regops file reprograms fuel from scratch). TILE_RST
//               while a record is in flight loses the skid contents by
//               design: quiesce first (FUEL_STAT idle) — the runbook rule.
//
// Record producers: (a) the host CSR GO (this file); (b) the IB-WALK D-029
// walker (wk_req_*, LANDED at the combine W-G3 flip): the tile walker's
// wf_* channel feeds the SAME 2-deep skid, per IB_FUEL.md §2.4 — records
// accepted ONLY in fuel mode (src=1), never during drain, atomically (one
// whole 64-bit record per push, so host-CSR GO and walker records can never
// interleave mid-record); on a same-cycle GO+walker collision (a host
// contract violation — the temporal split forbids driving both) the host
// CSR write wins the skid port deterministically and the walker simply
// holds valid (valid/ready, no record lost). Processing stays strictly
// serial and in-order — one burst in flight — exactly as before.
// Payload is FROZEN {tag[63:56], beats_64B[55:30], base_64B[29:0]}.
//
// Mode bits (src / ar_owner) are quasi-static route-level style: a write
// that CHANGES either while fuel is busy is REJECTED (old value kept) and
// flags FUEL_ERR.mode_switch. busy = skid nonempty | record engaged |
// reader busy | fifo nonempty.
//
// ABORT (FUEL_CTRL[9]): flushes the skid, aborts the in-flight record at
// the reader (which completes AXI protocol and acks), and drains the afifo
// (beats discarded, never presented to xw). Self-clears when everything is
// empty and idle.

`timescale 1ns/1ps

module apex_fuel_ctl (
  input  logic        clk,            // clk_tile
  input  logic        rst_glb_n,      // global tile reset (no TILE_RST)
  input  logic        rst_csr_n,      // global & ~TILE_RST

  // ── CSR access (from the cl_apex bridge FSM, 1-cycle strobes) ────────────
  input  logic        ctrl_wr,        // 0x20
  input  logic        base_wr,        // 0x24
  input  logic        beats_wr,       // 0x28
  input  logic        err_wr,         // 0x30 (W1C on [3:1])
  input  logic [31:0] wdata,
  output logic [31:0] ctrl_rd,
  output logic [31:0] base_rd,
  output logic [31:0] beats_rd,
  output logic [31:0] stat_rd,
  output logic [31:0] err_rd,

  // host_push detect (bridge: mailbox XW_PUSH write while fuel src)
  input  logic        host_push_err,

  // ── IB-WALK walker producer (combine W-G3; frozen walk2_freq_t payload
  //    {tag[63:56], beats_64B[55:30], base_64B[29:0]}) ─────────────────────
  input  logic        wk_req_valid,
  output logic        wk_req_ready,
  input  logic [63:0] wk_req,

  // ── mode levels out ──────────────────────────────────────────────────────
  output logic        fuel_src,       // 0 = host mailbox xw (RESET DEFAULT)
  output logic        ar_owner,       // 0 = PCIS loader owns DDR AR (R8)

  // ── 4-phase to apex_fuel_reader (shell domain) ───────────────────────────
  output logic        req_tgl_out,
  output logic        abort_tgl_out,
  output logic [29:0] fr_base_64B,    // stable while req in flight
  output logic [25:0] fr_beats_64B,
  input  logic        ack_tgl_in,

  // ── status levels from the shell domain (synced here) ────────────────────
  input  logic        rdr_busy_in,
  input  logic        ddr_ready_in,
  input  logic        axi_err_in,

  // ── afifo read side / xw mux ─────────────────────────────────────────────
  input  logic        f_rvalid,
  input  logic [63:0] f_rdata,
  output logic        f_rready,
  output logic        fuel_xw_valid,
  output logic [63:0] fuel_xw_data,
  input  logic        fuel_xw_ready
);

  // ── status synchronizers (2FF, global reset) ─────────────────────────────
  logic rdr_busy_meta, rdr_busy_sync;
  logic ddr_rdy_meta,  ddr_rdy_sync;
  logic axi_err_meta,  axi_err_sync;
  logic ack_meta, ack_sync, ack_seen;
  wire  ack_edge = (ack_sync != ack_seen);

  // ── CSR registers ────────────────────────────────────────────────────────
  logic        src_q, arown_q;
  logic [7:0]  tag_q;
  logic [29:0] base_q;
  logic [25:0] beats_q;
  logic [7:0]  last_tag_q;
  logic        desc_e_q, modesw_e_q, hpush_e_q;
  logic        drain_q;

  // ── 2-deep skid of frozen 64-bit records ─────────────────────────────────
  logic [63:0] skid_q [0:1];
  logic [1:0]  skid_cnt_q;
  wire         skid_full  = (skid_cnt_q == 2'd2);
  wire         skid_empty = (skid_cnt_q == 2'd0);

  // ── engagement state (4-phase tile half; global reset) ───────────────────
  logic        engaged_q;
  logic [7:0]  eng_tag_q;
  logic [63:0] eng_rec_q;              // payload regs — the stable crossing
  assign fr_base_64B  = eng_rec_q[29:0];
  assign fr_beats_64B = eng_rec_q[55:30];

  wire fifo_empty = !f_rvalid;
  wire busy = !skid_empty || engaged_q || rdr_busy_sync || !fifo_empty;

  // exported mode levels (the readbacks below use the same regs — an
  // undriven export here once read back "correctly" while the level never
  // left the module; the s3 DECERR probe caught it)
  assign fuel_src = src_q;
  assign ar_owner = arown_q;

  // ── readbacks ────────────────────────────────────────────────────────────
  assign ctrl_rd  = {8'b0, tag_q, 14'b0, arown_q, src_q};
  assign base_rd  = {2'b0, base_q};
  assign beats_rd = {6'b0, beats_q};
  assign stat_rd  = {16'b0, last_tag_q,
                     4'b0, ddr_rdy_sync, rdr_busy_sync,
                     (!skid_empty || engaged_q), fifo_empty};
  assign err_rd   = {28'b0, hpush_e_q, modesw_e_q, desc_e_q, axi_err_sync};

  // ── xw-side: drain overrides presentation ────────────────────────────────
  assign fuel_xw_valid = fuel_src && !drain_q && f_rvalid;
  assign fuel_xw_data  = f_rdata;
  assign f_rready      = drain_q ? 1'b1
                       : (fuel_src && !drain_q && fuel_xw_ready);

  // ── crossing state: GLOBAL reset only ────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_glb_n) begin
      rdr_busy_meta <= 1'b0; rdr_busy_sync <= 1'b0;
      ddr_rdy_meta  <= 1'b0; ddr_rdy_sync  <= 1'b0;
      axi_err_meta  <= 1'b0; axi_err_sync  <= 1'b0;
      ack_meta <= 1'b0; ack_sync <= 1'b0; ack_seen <= 1'b0;
      req_tgl_out   <= 1'b0;
      abort_tgl_out <= 1'b0;
      engaged_q     <= 1'b0;
      eng_tag_q     <= '0;
      eng_rec_q     <= '0;
    end else begin
      rdr_busy_meta <= rdr_busy_in; rdr_busy_sync <= rdr_busy_meta;
      ddr_rdy_meta  <= ddr_ready_in; ddr_rdy_sync <= ddr_rdy_meta;
      axi_err_meta  <= axi_err_in;  axi_err_sync  <= axi_err_meta;
      ack_meta <= ack_tgl_in; ack_sync <= ack_meta;

      if (engaged_q && ack_edge) begin
        ack_seen  <= ack_sync;
        engaged_q <= 1'b0;
      end else if (!engaged_q && !skid_empty && !drain_q) begin
        // engage the skid head: payload latched STABLE, then req toggles.
        // Blocked during drain — a GO issued mid-abort waits until the
        // drain self-clears rather than feeding beats into the discard.
        eng_rec_q   <= skid_q[0];
        eng_tag_q   <= skid_q[0][63:56];
        req_tgl_out <= ~req_tgl_out;
        engaged_q   <= 1'b1;
      end

      if (ctrl_wr && wdata[9]) abort_tgl_out <= ~abort_tgl_out;
    end
  end

  // ── CSR / skid / sticky state: TILE_RST-sensitive ────────────────────────
  wire mode_change = ctrl_wr && ((wdata[0] != src_q) || (wdata[1] != arown_q));
  wire go          = ctrl_wr && wdata[8];
  wire do_abort    = ctrl_wr && wdata[9];

  // walker producer accept (header note; IB_FUEL.md §2.4/§5): fuel mode
  // only, not during drain, skid room (or a same-cycle pop), and never on a
  // host-GO cycle — the atomic-record mux. The walker's geometry fences
  // guarantee beats != 0 (every fmt=1 tensor is nonempty; the d_ffn=0 case
  // is refused at the descriptor), so the GO path's desc guard has no
  // walker analogue.
  assign wk_req_ready = src_q && !drain_q && !go
                      && (!skid_full
                          || (engaged_q && ack_edge && !skid_empty));

  always_ff @(posedge clk) begin
    if (!rst_csr_n) begin
      src_q      <= 1'b0;              // host mailbox path: RESET DEFAULT
      arown_q    <= 1'b0;              // PCIS owns DDR AR: LOAD mode
      tag_q      <= '0;
      base_q     <= '0;
      beats_q    <= '0;
      last_tag_q <= '0;
      desc_e_q   <= 1'b0;
      modesw_e_q <= 1'b0;
      hpush_e_q  <= 1'b0;
      drain_q    <= 1'b0;
      skid_q[0]  <= '0;
      skid_q[1]  <= '0;
      skid_cnt_q <= '0;
    end else begin
      // skid pop (record completion) and push (GO or walker record)
      // resolved as ONE transition — sequential +/- assignments would let
      // the last one win and double-count on a same-cycle pop+push. GO and
      // walker pushes are mutually exclusive by wk_req_ready's !go term.
      logic pop_v, push_v, wk_push_v;
      logic [63:0] new_rec;
      pop_v     = engaged_q && ack_edge && !skid_empty;
      wk_push_v = wk_req_valid && wk_req_ready;
      push_v    = (go && (beats_q != 26'd0) && (!skid_full || pop_v))
                || wk_push_v;
      new_rec   = wk_push_v ? wk_req : {wdata[23:16], beats_q, base_q};

      if (pop_v) last_tag_q <= eng_tag_q;

      unique case ({pop_v, push_v})
        2'b10: begin skid_q[0] <= skid_q[1]; skid_cnt_q <= skid_cnt_q - 2'd1; end
        2'b01: begin skid_q[skid_cnt_q[0]] <= new_rec;
                     skid_cnt_q <= skid_cnt_q + 2'd1; end
        2'b11: begin                       // count unchanged, head advances
          if (skid_cnt_q == 2'd1) skid_q[0] <= new_rec;
          else begin skid_q[0] <= skid_q[1]; skid_q[1] <= new_rec; end
        end
        default: ;
      endcase

      if (base_wr)  base_q  <= wdata[29:0];
      if (beats_wr) beats_q <= wdata[25:0];

      if (ctrl_wr) begin
        tag_q <= wdata[23:16];
        if (mode_change) begin
          if (busy) modesw_e_q <= 1'b1;          // rejected, old values kept
          else begin
            src_q   <= wdata[0];
            arown_q <= wdata[1];
          end
        end
      end

      if (go && !push_v) desc_e_q <= 1'b1;       // dropped: full or beats==0

      if (do_abort) begin
        skid_cnt_q <= '0;                        // flush queued records
        drain_q    <= 1'b1;                      // discard fifo residue
      end else if (drain_q && fifo_empty && !engaged_q && !rdr_busy_sync)
        drain_q <= 1'b0;

      if (host_push_err) hpush_e_q <= 1'b1;

      if (err_wr) begin                          // W1C on [3:1]; [0] is the
        if (wdata[1]) desc_e_q   <= 1'b0;        // shell axi_err RO mirror
        if (wdata[2]) modesw_e_q <= 1'b0;
        if (wdata[3]) hpush_e_q  <= 1'b0;
      end
    end
  end

endmodule
