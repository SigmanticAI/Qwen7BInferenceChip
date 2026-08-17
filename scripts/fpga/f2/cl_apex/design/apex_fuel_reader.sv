// apex_fuel_reader.sv — IB-FUEL s3: sequential DDR burst reader, ENTIRELY
// in the shell clock domain (clk_main_a0). Consumes one frozen fuel record
// {tag[7:0], beats_64B[25:0], base_64B[29:0]} per 4-phase handshake from
// the tile-side controller (apex_fuel_ctl), streams the region as 64-bit
// lane8 beats into the apex_afifo write side, then acks. IB_FUEL.md §4:
// deliberately boring — single ID, INCR, AxSIZE=6, bursts trimmed to the
// 4 KiB boundary, ONE burst outstanding, zero data reformatting beyond
// slicing 512-bit words into eight lanes (the image IS the wire format).
//
// 4-phase (apex_ocl_cdc idiom): req_tgl/abort_tgl arrive as toggles and
// are 2FF-synced here; the payload {base,beats,tag} is latched STABLE by
// the tile side before req toggles (quasi-static crossing, max_delay in
// cl_synth_user.xdc). ack_tgl returns the same way.
//
// ABORT: completes the AXI protocol for any outstanding burst (discarding
// its beats), skips the rest of the record, and acks. Never leaves sh_ddr
// mid-burst.
//
// axi_err is a shell-domain STICKY set on any RRESP != OKAY (the record
// still completes protocol-wise and acks); it clears only on global reset
// — the tile CSR mirrors it read-only (documented in the FUEL_ERR map).

`timescale 1ns/1ps

module apex_fuel_reader (
  input  logic         clk,           // clk_main_a0
  input  logic         rst_n,         // shell-domain global reset

  // ── 4-phase request from apex_fuel_ctl (tile domain source) ──────────────
  input  logic         req_tgl_in,
  input  logic         abort_tgl_in,
  input  logic [29:0]  fr_base_64B,   // quasi-static while req in flight
  input  logic [25:0]  fr_beats_64B,
  output logic         ack_tgl_out,

  // ── status levels (synced on the tile side) ──────────────────────────────
  output logic         rdr_busy,
  output logic         axi_err,

  // ── DDR AXI4 read master (via the R8 AR mux in cl_apex) ──────────────────
  output logic [15:0]  m_arid,
  output logic [63:0]  m_araddr,
  output logic [7:0]   m_arlen,
  output logic [2:0]   m_arsize,
  output logic [1:0]   m_arburst,
  output logic         m_arvalid,
  input  logic         m_arready,
  input  logic [15:0]  m_rid,
  input  logic [511:0] m_rdata,
  input  logic [1:0]   m_rresp,
  input  logic         m_rlast,
  input  logic         m_rvalid,
  output logic         m_rready,

  // ── afifo write side (shell clock) ───────────────────────────────────────
  output logic         fw_valid,
  output logic [63:0]  fw_data,
  input  logic         fw_ready
);

  localparam logic [15:0] FUEL_ARID = 16'h00FE;

  // toggle synchronizers (ASYNC_REG pairs in cl_synth_user.xdc)
  logic req_meta, req_sync, req_seen;
  logic abt_meta, abt_sync, abt_seen;
  wire  req_edge = (req_sync != req_seen);
  wire  abt_edge = (abt_sync != abt_seen);

  typedef enum logic [2:0] {
    F_IDLE, F_ISSUE_AR, F_COLLECT, F_DRAIN, F_ACK
  } f_st_e;
  f_st_e        st_q;
  // perf/ingest F2: the ISSUE side (next AR) and the COLLECT side (beats
  // received) are split so the next burst's AR can be issued while the
  // current burst streams — at most TWO un-retired ARs (len_q depth),
  // single ID, INCR, strictly in-order, still ≤4 KiB each.
  logic [63:0]  ar_addr_q;           // next burst byte address to ISSUE
  logic [25:0]  ar_words_q;          // words not yet covered by an issued AR
  logic [25:0]  words_q;             // words not yet RECEIVED (record done)
  logic [7:0]   rbeats_q;            // R beats received in current burst
  logic         abort_q;             // discard the rest of the record
  // burst-length queue across the (≤2) in-flight bursts, in issue order:
  // len_q[0] = the burst being COLLECTED (rlast cross-check), len_q[1] =
  // the prefetched one
  logic [7:0]   len_q [0:1];
  logic [1:0]   len_cnt_q;
  wire          len_empty = (len_cnt_q == 2'd0);
  wire          len_full  = (len_cnt_q == 2'd2);

  // unpack stage: one 512-bit word -> eight 64-bit lane pushes
  logic [511:0] unpk_q;
  logic [3:0]   unpk_cnt_q;          // 0 = empty, else lanes remaining
  wire          unpk_empty = (unpk_cnt_q == 4'd0);
  // perf/ingest F1: the current word's LAST lane leaves on this edge, so
  // the next R beat may land on the same edge (the F_COLLECT load below is
  // written after the shift and wins the nonblocking race by design).
  // Removes the 1-in-9 accept bubble: steady state becomes 8 shell
  // cycles/64 B word — the afifo W=64 write-side bound.
  wire          unpk_last_push = (unpk_cnt_q == 4'd1) && fw_valid && fw_ready;

  // burst sizing: never cross 4 KiB — words to the boundary from the
  // ISSUE address (advances at AR accept, not per beat)
  wire [5:0]  word_in_page  = ar_addr_q[11:6];
  wire [6:0]  to_boundary   = 7'd64 - {1'b0, word_in_page};
  wire [25:0] burst_words   = (ar_words_q < {19'b0, to_boundary})
                              ? ar_words_q : {19'b0, to_boundary};

  assign m_arid    = FUEL_ARID;
  assign m_araddr  = ar_addr_q;
  // combinational: ar_addr_q/ar_words_q are stable between AR accepts, and
  // a registered copy would present a stale length on the entry cycle
  assign m_arlen   = 8'(burst_words - 26'd1);
  assign m_arsize  = 3'd6;
  assign m_arburst = 2'b01;
  // F2: issue from F_ISSUE_AR (record's first burst) OR while collecting
  // (the prefetched one); bounded by the len queue, never past the record,
  // never once an abort is latched
  assign m_arvalid = !abort_q && (ar_words_q != 26'd0) && !len_full
                     && ((st_q == F_ISSUE_AR) || (st_q == F_COLLECT));
  // accept an R beat with the unpack stage free or freeing on this edge
  // (or while discarding)
  assign m_rready  = ((st_q == F_COLLECT) && (unpk_empty || unpk_last_push))
                   || (st_q == F_DRAIN);

  assign fw_valid = !unpk_empty && !abort_q;
  assign fw_data  = unpk_q[63:0];

  assign rdr_busy = (st_q != F_IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      req_meta <= 1'b0; req_sync <= 1'b0; req_seen <= 1'b0;
      abt_meta <= 1'b0; abt_sync <= 1'b0; abt_seen <= 1'b0;
      ack_tgl_out <= 1'b0;
      st_q     <= F_IDLE;
      ar_addr_q  <= '0;
      ar_words_q <= '0;
      words_q  <= '0;
      len_q[0] <= '0;
      len_q[1] <= '0;
      len_cnt_q <= '0;
      rbeats_q <= '0;
      abort_q  <= 1'b0;
      unpk_q   <= '0;
      unpk_cnt_q <= '0;
      axi_err  <= 1'b0;
    end else begin
      req_meta <= req_tgl_in;  req_sync <= req_meta;
      abt_meta <= abort_tgl_in; abt_sync <= abt_meta;

      if (abt_edge) begin
        abt_seen <= abt_sync;
        // req_edge term closes the same-cycle race: a record whose req and
        // abort syncs land together must be consumed as ABORTED, not run
        if (st_q != F_IDLE || req_edge) abort_q <= 1'b1;
      end

      // unpack stage: shift one lane per accepted push
      if (fw_valid && fw_ready) begin
        unpk_q     <= {64'b0, unpk_q[511:64]};
        unpk_cnt_q <= unpk_cnt_q - 4'd1;
      end else if (abort_q && !unpk_empty) begin
        unpk_cnt_q <= '0;                      // discard residue on abort
      end

      // F2 issue-side bookkeeping — one transition for the len queue so a
      // same-cycle {rlast retire, AR accept} cannot double-count (the ctl
      // skid's own idiom). Runs in EVERY state; ar_push can only be true
      // where m_arvalid can (F_ISSUE_AR / F_COLLECT, pre-abort).
      begin : ar_issue
        logic ar_push, len_pop;
        ar_push = m_arvalid && m_arready;
        len_pop = m_rvalid && m_rready && m_rlast
                  && ((st_q == F_COLLECT) || (st_q == F_DRAIN));
        unique case ({len_pop, ar_push})
          2'b10: begin len_q[0] <= len_q[1];
                       len_cnt_q <= len_cnt_q - 2'd1; end
          2'b01: begin len_q[len_cnt_q[0]] <= 8'(burst_words - 26'd1);
                       len_cnt_q <= len_cnt_q + 2'd1; end
          2'b11: begin                       // count unchanged, head advances
            if (len_cnt_q == 2'd1) len_q[0] <= 8'(burst_words - 26'd1);
            else begin len_q[0] <= len_q[1];
                       len_q[1] <= 8'(burst_words - 26'd1); end
          end
          default: ;
        endcase
        if (ar_push) begin
          ar_addr_q  <= ar_addr_q + ({38'b0, burst_words} << 6);
          ar_words_q <= ar_words_q - burst_words;
        end
      end

      unique case (st_q)
        F_IDLE: if (req_edge) begin
          req_seen <= req_sync;
          ar_addr_q  <= {28'b0, fr_base_64B, 6'b0};
          ar_words_q <= fr_beats_64B;
          words_q  <= fr_beats_64B;
          rbeats_q <= '0;
          if (abort_q || (fr_beats_64B == 26'd0)) begin
            // aborted before start (or degenerate record blocked by ctl):
            // consume the handshake without touching DDR
            st_q <= F_ACK;
          end else begin
            st_q <= F_ISSUE_AR;
          end
        end

        F_ISSUE_AR: begin                    // record's FIRST burst
          if (m_arvalid && m_arready) begin
            rbeats_q <= '0;
            st_q     <= abort_q ? F_DRAIN : F_COLLECT;
          end
        end

        F_COLLECT: begin
          // (the prefetched next AR issues from here via m_arvalid; the
          // ar_issue block above owns the pointers and the len queue)
          if (m_rvalid && m_rready) begin
            if (m_rresp != 2'b00) axi_err <= 1'b1;
            unpk_q     <= m_rdata;
            unpk_cnt_q <= 4'd8;
            rbeats_q   <= rbeats_q + 8'd1;
            words_q    <= words_q - 26'd1;
            // rlast cross-check against the ISSUE-ORDER head (count is
            // authoritative; same ID + INCR keeps bursts strictly in order)
            if (m_rlast != (rbeats_q == len_q[0])) axi_err <= 1'b1;
            if (m_rlast) begin
              rbeats_q <= '0;                          // next burst's count
              if (abort_q)
                st_q <= (len_cnt_q == 2'd1) ? F_ACK : F_DRAIN;
              else if (words_q == 26'd1) st_q <= F_ACK;   // record complete
              // else stay in F_COLLECT: the next burst's beats follow
              // in-order (as early as next cycle with a pipelined slave)
            end
            if (abort_q && !m_rlast) st_q <= F_DRAIN;
          end else if (abort_q) begin
            // abort raised between beats: finish outstanding bursts
            // protocol-wise; nothing outstanding -> ack directly (a DRAIN
            // with no beats due would hang)
            st_q <= len_empty ? F_ACK : F_DRAIN;
          end
        end

        F_DRAIN: begin      // abort: finish EVERY outstanding burst, discard
          if (m_rvalid && m_rready) begin
            if (m_rresp != 2'b00) axi_err <= 1'b1;
            // len queue pops on each rlast in the ar_issue block above
            if (m_rlast && (len_cnt_q == 2'd1)) st_q <= F_ACK;
          end
        end

        F_ACK: begin
          // wait for the unpack stage to drain into the fifo (or abort
          // discard) so "record complete" means every beat is IN the fifo
          if (unpk_empty) begin
            ack_tgl_out <= ~ack_tgl_out;
            abort_q     <= 1'b0;
            st_q        <= F_IDLE;
          end
        end

        default: st_q <= F_IDLE;
      endcase
    end
  end

endmodule
