// baseline/mxe_ctrl.sv — PINNED VERBATIM COPY of rtl/mxe/mxe_ctrl.sv as
// it stood BEFORE the D-005 load-under-compute closure (sequential
// ingest -> [wload -> compute -> flush]xKB -> drain scheduler). Used ONLY by
// verif/mxe/perf as the perf-comparison baseline and as the permanent
// kept-failing overlap regression (run_base_reqov). Do NOT update this file
// when mxe_ctrl.sv changes — its whole value is being the pinned reference.
// mxe_ctrl.sv — MXE descriptor FSM.
//
// Implements: ARCHITECTURE.md §3 descriptor + conservative legality reject,
//             §5 / D-006 job semantics (done ⇒ final beat ACCEPTED post-skid;
//             desc_ready for job N+1 gated on job N's done — closes run1's
//             D-019 wart), D-003 (K ≤ 2048 legality), D-005 (per-pass bank-B
//             column loads with bank swap), C-2 (requant params exported to
//             the epilogue for the drain phase).
//
// ── Job flow (per accepted legal descriptor) ───────────────────────────────
//   IDLE   : desc_ready high. Legality checked combinationally on the offered
//            descriptor. ILLEGAL → descriptor consumed, desc_error 1-cycle
//            pulse + sticky bit, ZERO other state change (no busy, no done,
//            no stream consumption, accumulators untouched).
//   INGEST : accept M*ceil(K/8) activation beats into the activation buffer.
//            Beat order (contract for the producer): row-major, chunk-minor —
//            beat (m, p) carries A[m][8p+7 : 8p], lane k = A[m][8p+k];
//            written at address m*KB + p. Lanes beyond K on the last chunk
//            are DON'T-CARE (the matching weight rows are zero-masked).
//   Then per K-chunk p = 0 .. KB-1  (KB = ceil(K/8), multi-pass accumulate):
//   WLOAD  : MXE_N column-shift strobes into the NON-live weight bank:
//            N beats from the weight stream (ascending column order; beat
//            (p, c) lane r = B[8p+r][c]; rows ≥ K on the last chunk are
//            zero-masked here) followed by MXE_N-N internally generated zero
//            columns. On the final strobe the banks SWAP (live_sel toggles).
//   COMPUTE: stream the M buffered activation beats of chunk p into the
//            array back-to-back (1-cycle SRAM read latency pipelined), each
//            tagged {row m, clr}. clr = 1 only on pass 0 of an overwriting
//            job (WS always; OS with accumulate=0) — see mxe_array.sv OS
//            semantics.
//   FLUSH  : 2*MXE_N cycles to let the last wavefront reach the column
//            accumulators.
//   DRAIN  : emit M result beats (row m = accumulator row m; lanes ≥ N are
//            zero). Requant/bypass mux lives in mxe_top; this FSM only
//            presents row indices and valid/last.
//   WAIT_DONE: done pulses ONLY once all M beats are accepted POST-skid
//            (res_accepted taps the output skid's m_valid && m_ready).
//            desc_ready reasserts the cycle AFTER done.
//
// busy = (state ≠ IDLE) || (result beats sent into the skid but not yet
// accepted downstream) — §5 definition, skid occupancy included.
//
// NOTE (documented contract interpretation): the descriptor carries BOTH an
// opcode (OP_GEMM_WS/OP_GEMM_OS) and a mode_os flag with the same meaning.
// The OPCODE is authoritative here; desc.mode_os is ignored. desc.accumulate
// is honoured only for OP_GEMM_OS (ignored for WS, per the apex_pkg field
// comment "OS: retain accumulators").

module mxe_ctrl
  import apex_pkg::*;
  import mxe_cfg_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,          // synchronous, active-low

  // descriptor in (simple direct interface; fields registered on accept)
  input  logic       desc_valid,
  output logic       desc_ready,
  input  mxe_desc_t  desc,
  output logic       desc_error,        // 1-cycle pulse on illegal descriptor
  output logic       desc_error_sticky, // cleared only by reset (CSR later)
  output logic       busy,
  output logic       done,              // 1-cycle pulse, D-006 semantics

  // weight stream (post input skid)
  input  logic        w_valid,
  output logic        w_ready,
  input  lane8_beat_t w_beat,

  // activation stream (post input skid)
  input  logic        a_valid,
  output logic        a_ready,
  input  lane8_beat_t a_beat,

  // activation buffer (mxe_buf, 1-cycle read latency)
  output logic                 abuf_we,
  output logic [ABUF_AW-1:0]   abuf_waddr,
  output logic [8*MXE_N-1:0]   abuf_wdata,
  output logic [ABUF_AW-1:0]   abuf_raddr,
  input  logic [8*MXE_N-1:0]   abuf_rdata,

  // systolic array control
  output logic               arr_en,
  output logic [8*MXE_N-1:0] arr_act_lanes,
  output logic               arr_beat_valid,
  output logic               arr_beat_clear,
  output logic [ROW_W-1:0]   arr_beat_row,
  output logic               arr_live_sel,
  output logic               arr_wload_en,
  output logic [8*MXE_N-1:0] arr_wload_lanes,
  output logic [ROW_W-1:0]   arr_rd_row,

  // drain / result (pre output skid; data path assembled in mxe_top)
  output logic       r_valid,
  input  logic       r_ready,           // output skid s_ready
  output logic       r_last,
  input  logic       res_accepted,      // output skid m_valid && m_ready tap
  output logic       requant_en_q,
  output logic [15:0] rq_scale_q,
  output logic [4:0]  rq_shift_q,
  output logic [3:0]  n_active          // registered N (drain lane masking)
);

  typedef enum logic [2:0] {
    S_IDLE, S_INGEST, S_WLOAD, S_COMPUTE, S_FLUSH, S_DRAIN, S_WAIT_DONE
  } state_e;

  state_e state;

  // registered job parameters
  logic [6:0]        m_cnt;      // 1..64
  logic [3:0]        n_cnt;      // 1..8
  logic [PASS_W-1:0] kb_cnt;     // 1..256 K-chunks
  logic [3:0]        klast;      // valid rows in the final chunk, 1..8
  logic              clear_job;  // pass-0 beats carry clr
  logic [ACNT_W-1:0] total_a;    // M*KB ingest beats

  // phase counters
  logic [ACNT_W-1:0] ing_cnt;
  logic [PASS_W-1:0] pass_q;
  logic [3:0]        wl_cnt;
  logic [6:0]        comp_cyc;   // 0..M (issue M reads, then 1 inject cycle)
  logic [ABUF_AW-1:0] rd_addr_q;
  logic              inj_valid_q;
  logic [ROW_W-1:0]  inj_row_q;
  logic [4:0]        flush_cnt;
  logic [6:0]        drain_cnt;
  logic [6:0]        sent_cnt;   // beats accepted INTO the output skid
  logic [6:0]        accn_cnt;   // beats accepted POST-skid (D-006)
  logic              live_q;     // computing weight bank

  // ── legality (ARCHITECTURE.md §3, D-003; implemented ranges per
  //    mxe_cfg_pkg header: M ≤ 64, N ≤ 8; single buffer id 0 implemented) ──
  logic legal;
  assign legal =
       ((desc.opcode == OP_GEMM_WS) || (desc.opcode == OP_GEMM_OS))
    && (desc.k_dim != '0) && (desc.k_dim <= DIM_W'(K_MAX))
    && (desc.m_dim != '0) && (desc.m_dim <= DIM_W'(M_TILE_MAX))
    && (desc.n_dim != '0) && (desc.n_dim <= DIM_W'(MXE_N))
    && (desc.src_a_buf == 4'd0) && (desc.src_b_buf == 4'd0)
    && (desc.dst_buf   == 4'd0);
    // rq_shift ≤ 31 is enforced by its 5-bit field width (contract note)

  // KB = ceil(K/8) and total ingest beats M*KB, explicit widths
  logic [12:0]       k_pad;
  logic [9:0]        kb_new;      // up to 512 for the full 12-bit field range
  logic [ACNT_W-1:0] total_new;
  assign k_pad     = {1'b0, desc.k_dim} + 13'd7;
  assign kb_new    = k_pad[12:3];
  assign total_new = ACNT_W'(desc.m_dim) * ACNT_W'(kb_new);

  // ── WLOAD helpers ─────────────────────────────────────────────────────────
  logic       w_shift;                 // one column enters the chain
  logic [3:0] rows_valid;              // un-masked rows of the current chunk
  assign rows_valid = (pass_q == kb_cnt - PASS_W'(1)) ? klast : 4'(MXE_N);
  assign w_shift    = (state == S_WLOAD)
                      && ((wl_cnt < n_cnt) ? w_valid : 1'b1);

  // ── stream-facing outputs ─────────────────────────────────────────────────
  assign desc_ready = (state == S_IDLE) && !done;
  assign a_ready    = (state == S_INGEST);
  assign w_ready    = (state == S_WLOAD) && (wl_cnt < n_cnt);

  assign abuf_we    = (state == S_INGEST) && a_valid;
  assign abuf_waddr = ing_cnt[ABUF_AW-1:0];
  assign abuf_wdata = a_beat.data;
  assign abuf_raddr = rd_addr_q;

  // ── array-facing outputs ──────────────────────────────────────────────────
  assign arr_en         = (state == S_COMPUTE) || (state == S_FLUSH);
  assign arr_act_lanes  = inj_valid_q ? abuf_rdata : '0;
  assign arr_beat_valid = inj_valid_q;
  assign arr_beat_clear = clear_job && (pass_q == '0);
  assign arr_beat_row   = inj_row_q;
  assign arr_live_sel   = live_q;
  assign arr_wload_en   = w_shift;

  always_comb begin
    for (int r = 0; r < int'(MXE_N); r++) begin
      if ((wl_cnt < n_cnt) && (r < int'({28'd0, rows_valid})))
        arr_wload_lanes[8*r +: 8] = w_beat.data[8*r +: 8];
      else
        arr_wload_lanes[8*r +: 8] = 8'd0;
    end
  end

  // ── drain outputs ─────────────────────────────────────────────────────────
  assign arr_rd_row = drain_cnt[ROW_W-1:0];
  assign r_valid    = (state == S_DRAIN);
  assign r_last     = (drain_cnt == m_cnt - 7'd1);
  assign n_active   = n_cnt;

  assign busy = (state != S_IDLE) || (sent_cnt != accn_cnt);

  // ── FSM ───────────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state             <= S_IDLE;
      done              <= 1'b0;
      desc_error        <= 1'b0;
      desc_error_sticky <= 1'b0;
      live_q            <= 1'b0;
      inj_valid_q       <= 1'b0;
      inj_row_q         <= '0;
      m_cnt             <= '0;
      n_cnt             <= '0;
      kb_cnt            <= '0;
      klast             <= '0;
      clear_job         <= 1'b0;
      requant_en_q      <= 1'b0;
      rq_scale_q        <= '0;
      rq_shift_q        <= '0;
      total_a           <= '0;
      ing_cnt           <= '0;
      pass_q            <= '0;
      wl_cnt            <= '0;
      comp_cyc          <= '0;
      rd_addr_q         <= '0;
      flush_cnt         <= '0;
      drain_cnt         <= '0;
      sent_cnt          <= '0;
      accn_cnt          <= '0;
    end else begin
      done       <= 1'b0;
      desc_error <= 1'b0;

      // post-skid acceptance counter (any state; reset on job accept below)
      if (res_accepted) accn_cnt <= accn_cnt + 7'd1;

      // buffer-read → array-inject 1-cycle pipeline
      inj_valid_q <= (state == S_COMPUTE) && (comp_cyc < m_cnt);
      inj_row_q   <= comp_cyc[ROW_W-1:0];

      unique case (state)
        S_IDLE: begin
          if (desc_valid && desc_ready) begin
            if (legal) begin
              m_cnt        <= 7'(desc.m_dim);
              n_cnt        <= 4'(desc.n_dim);
              kb_cnt       <= kb_new[PASS_W-1:0];  // legality caps K ≤ 2048
              klast        <= (desc.k_dim[2:0] == 3'd0)
                              ? 4'd8 : {1'b0, desc.k_dim[2:0]};
              clear_job    <= !((desc.opcode == OP_GEMM_OS) && desc.accumulate);
              requant_en_q <= desc.requant_en;
              rq_scale_q   <= desc.rq_scale;
              rq_shift_q   <= desc.rq_shift;
              total_a      <= total_new;
              ing_cnt      <= '0;
              pass_q       <= '0;
              sent_cnt     <= '0;
              accn_cnt     <= '0;
              state        <= S_INGEST;
            end else begin
              // §3: consume descriptor, pulse + sticky, ZERO state change
              desc_error        <= 1'b1;
              desc_error_sticky <= 1'b1;
            end
          end
        end

        S_INGEST: begin
          if (a_valid) begin
            ing_cnt <= ing_cnt + ACNT_W'(1);
            if (ing_cnt == total_a - ACNT_W'(1)) begin
              wl_cnt <= '0;
              state  <= S_WLOAD;
            end
          end
        end

        S_WLOAD: begin
          if (w_shift) begin
            wl_cnt <= wl_cnt + 4'd1;
            if (wl_cnt == 4'(MXE_N - 1)) begin
              live_q    <= ~live_q;               // D-005 bank swap
              comp_cyc  <= '0;
              rd_addr_q <= ABUF_AW'(pass_q);      // addr(m=0, p) = p
              state     <= S_COMPUTE;
            end
          end
        end

        S_COMPUTE: begin
          comp_cyc <= comp_cyc + 7'd1;
          if (comp_cyc < m_cnt)
            rd_addr_q <= rd_addr_q + ABUF_AW'(kb_cnt);  // addr(m+1,p)
          if (comp_cyc == m_cnt) begin                  // last inject cycle
            flush_cnt <= '0;
            state     <= S_FLUSH;
          end
        end

        S_FLUSH: begin
          flush_cnt <= flush_cnt + 5'd1;
          if (flush_cnt == 5'(FLUSH_CYC - 1)) begin
            if (pass_q + PASS_W'(1) < kb_cnt) begin
              pass_q <= pass_q + PASS_W'(1);
              wl_cnt <= '0;
              state  <= S_WLOAD;
            end else begin
              drain_cnt <= '0;
              state     <= S_DRAIN;
            end
          end
        end

        S_DRAIN: begin
          if (r_valid && r_ready) begin
            sent_cnt  <= sent_cnt + 7'd1;
            drain_cnt <= drain_cnt + 7'd1;
            if (drain_cnt == m_cnt - 7'd1)
              state <= S_WAIT_DONE;
          end
        end

        S_WAIT_DONE: begin
          if (accn_cnt == m_cnt) begin   // final beat accepted post-skid
            done  <= 1'b1;               // D-006
            state <= S_IDLE;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  // informational / unimplemented descriptor bits, consumed for lint
  logic unused_fields;
  assign unused_fields = &{1'b0, desc.reserved, desc.mode_os,
                           a_beat.last, w_beat.last, k_pad[2:0]};

endmodule
