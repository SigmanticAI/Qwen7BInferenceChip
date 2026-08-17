// mxe_wfeed_w4b.sv — W4-B weight feeder: the ADOPTED G=32/(B) weight path
// (D-031, docs/design/W4B_FEEDER.md; B3_WEIGHT_PATH.md §8).
//
// packed-W4 stream in ── unpack (the (A) feeder's S1-S6 framing, verbatim
// contract) ──> per-element q8 = clamp(RNE(code4 · s_g / s8), -128, 127)
// ──> lane8 INT8 beats out. MXE still sees exactly one dtype (C-4).
//
//   * s_g  = fp16 GROUP scale (G K-rows x 1 column, G = elab param,
//            SHIP G=32), streamed in on the gs_* sideband in ascending
//            group_ids order: gid = c*ceil(K/G) + floor(k/G).
//   * s8   = fp16 HOST-computed stripe requant scale, a JOB PARAMETER
//            (the §8.1 D-021 delta: K-split stripes share one epilogue
//            factor only the host can compute).
//
// ARITHMETIC: w4b_fp_pkg::w4b_quant_front -> cq_rne_div_pipe (the fparith-
// certified S10 divider, one per lane) -> cq_fp_pkg::quant_back. The comb
// twin w4b_fp_pkg::w4b_quant_one is the TB oracle's per-element form; the
// pipeline composes the IDENTICAL functions (no pipe-private arithmetic),
// so the package-level exhaustive proof carries (cq_quant_pipe precedent).
// v0 datapath = 8 parallel divider pipes (full rate, II=1). A 4-pipe
// 16-entry-LUT-per-group variant would halve the dividers at G=32 (16
// fills per 32-element group = 4 fills/beat average); flagged as an area
// follow-on, NOT built — measure first (rmsnorm-probe pattern).
//
// GROUP <-> BEAT ALIGNMENT PROOF (why one scale per beat suffices): lane r
// of beat n carries flat element e = 8n + r, i.e. row k = 8p + r with
// p = n / N. G ∈ {16, 32} is a multiple of 8 and 8p mod G ∈ {0, 8, 16, 24},
// so rows 8p..8p+7 never straddle a G boundary: ALL 8 lanes of a beat share
// one gid = c*ng_k + (p >> log2(G/8)). One RAM read per beat.
//
// STREAM CONTRACT: identical to mxe_wfeed_w4 (S1-S6): job_beats counts
// EMITTED INT8 beats = ceil(K/8)*N; consumed packed beats =
// ceil(job_beats/2); odd job_beats drops the final packed beat's upper
// half. w4b_en is a route/CSR-level select sampled at job accept (NOT a
// descriptor field — apex_pkg stays frozen); w4b_en = 0 is a transparent
// 1:1 passthrough, so the module can sit unconditionally on the weight
// path. In passthrough the gs/s8 side is ignored entirely.
//
// JOB LEGALITY (§3-style, pulse + sticky, ZERO state change on reject):
//   1 <= N <= 8, 1 <= K <= K_MAX, job_beats == ceil(K/8)*N,
//   job_beats <= BEATS_MAX. (K/N are new job params vs (A): the gid walk
//   needs them. In passthrough mode K/N are ignored and only
//   1 <= job_beats <= BEATS_MAX is enforced — (A)-compatible.)
//
// GS SIDEBAND: ng_total = N*ceil(K/G) scales expected per W4B job, written
// in arrival order to the scale RAM (ascending gid by contract). Emission
// of a beat needing gid stalls until gs_wcnt > gid — late scales stall the
// stream (never corrupt it); early (eager) scales park in the skid + RAM.
// gs beats beyond ng_total are NOT accepted during the job (gs_ready
// deasserts) — a framing overrun therefore back-pressures the producer and
// is observable, not silent.
//
// D-006: done <=> every emitted beat ACCEPTED downstream (post-skid);
// job_ready gated until strictly after done. Mid-job reset: divider pipes
// take flush (kill all in-flight in one clock), matching cq_rne_div_pipe's
// contract.
//
// LATENCY: unpack(1) + gsread(1) + front(1) + divider(14) + back/pack(1)
// ~= 18; II=1 when gs is resident and the sink accepts.

`default_nettype none

module mxe_wfeed_w4b
  import apex_pkg::*;
#(
  parameter int unsigned G         = 32,     // K-rows per scale group (16|32)
  parameter int unsigned BEATS_MAX = 2048,  // KB*N <= 256*8 (K_MAX geometry)
  parameter int unsigned K_MAX_P   = 2048    // mirrors apex_pkg K_MAX
) (
  input  wire logic        clk,
  input  wire logic        rst_n,             // synchronous, active low

  // route/CSR-level mode select (sampled at job accept)
  input  wire logic        w4b_en,

  // job port (pulse; §3 legality reject on bad params)
  input  wire logic        job_valid,
  output logic             job_ready,
  input  wire logic [15:0] job_beats,         // EMITTED INT8 beats = KB*N
  input  wire logic [11:0] job_k,             // K (rows), W4B mode only
  input  wire logic [3:0]  job_n,             // N (cols 1..8), W4B mode only
  input  wire logic [15:0] job_s8,            // fp16 stripe requant scale
  output logic             job_error,         // 1-cycle pulse
  output logic             job_error_sticky,

  // packed-W4 / passthrough weight stream in (lane8 beats)
  input  wire logic        pw_valid,
  output logic             pw_ready,
  input  wire lane8_beat_t pw_beat,

  // fp16 group-scale sideband in (W4B mode; ascending gid)
  input  wire logic        gs_valid,
  output logic             gs_ready,
  input  wire logic [15:0] gs_data,

  // INT8 weight stream out (lane8 beats)
  output logic             xw_valid,
  input  wire logic        xw_ready,
  output lane8_beat_t      xw_beat,

  // job status (D-006)
  output logic             busy,
  output logic             done
);

  // ── derived geometry ──────────────────────────────────────────────────────
  localparam int unsigned S3     = (G == 16) ? 1 : 2;   // log2(G/8)
  localparam int unsigned NG_MAX = 8 * ((K_MAX_P + G - 1) / G); // 512 @ G=32
  localparam int unsigned GW     = $clog2(NG_MAX);      // gid width (9 @512)

  generate
    if (!(G == 16 || G == 32)) begin : g_cfg_g
      $error("mxe_wfeed_w4b: G must be 16 or 32 (D-031 freeze: ship 32)");
    end
  endgenerate

  // ── input skids (§5: every boundary crossing) ─────────────────────────────
  logic        ipw_valid, ipw_ready;
  lane8_beat_t ipw_beat;
  logic [$bits(lane8_beat_t)-1:0] ipw_vec;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_pw_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (pw_valid),
    .s_ready (pw_ready),
    .s_data  ($bits(lane8_beat_t)'(pw_beat)),
    .m_valid (ipw_valid),
    .m_ready (ipw_ready),
    .m_data  (ipw_vec)
  );
  assign ipw_beat = lane8_beat_t'(ipw_vec);

  logic        igs_valid, igs_ready;
  logic [15:0] igs_data;

  stream_skid #(.WIDTH(16)) u_gs_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (gs_valid),
    .s_ready (gs_ready),
    .s_data  (gs_data),
    .m_valid (igs_valid),
    .m_ready (igs_ready),
    .m_data  (igs_data)
  );

  // ── job state ─────────────────────────────────────────────────────────────
  typedef enum logic [1:0] { ST_IDLE, ST_RUN, ST_DRAIN } st_e;
  st_e st;

  logic        mode_q;                 // sampled w4b_en
  logic [15:0] beats_q;                // job_beats
  logic [15:0] s8_q;
  logic [11:0] k_q;
  logic [3:0]  n_q;
  logic [GW:0] ngk_q;                  // ceil(K/G)  (<= 128)
  logic [GW:0] ngtot_q;                // N*ceil(K/G) (<= NG_MAX)

  // emit-side counters
  logic [15:0] emit_cnt;               // INT8 beats issued into the pipe
  logic [15:0] out_cnt;                // beats accepted POST-SKID
  logic [3:0]  col_q;                  // c = beat mod N
  logic [12:0] prow_q;                 // p = beat div N (<= 2048/8*? p <= KB-1 <= 255... KB<=256, p<=255; 13b generous)
  logic        half_q;                 // which nibble half of the packed beat

  // gs write side
  logic [GW:0] gs_wcnt;

  // ── legality (evaluated on the job pulse; pure function of inputs) ───────
  logic [15:0] kb_calc;
  assign kb_calc = 16'((32'(job_k) + 7) >> 3);          // ceil(K/8)
  logic legal_w4b, legal_pass;
  assign legal_pass = (job_beats != 16'd0) && (32'(job_beats) <= BEATS_MAX);
  assign legal_w4b  = legal_pass
                   && (job_n >= 4'd1) && (job_n <= 4'd8)
                   && (job_k >= 12'd1) && (32'(job_k) <= K_MAX_P)
                   && (32'(job_beats) == 32'(kb_calc) * 32'(job_n));

  wire job_ok = w4b_en ? legal_w4b : legal_pass;

  assign job_ready = (st == ST_IDLE) && !done;   // D-006: strictly after done

  // ── group-scale RAM (NG_MAX x 16, 1W1R, sync read) ───────────────────────
  logic [15:0] gs_ram [NG_MAX];
  logic [GW-1:0] gid_rd;
  logic [15:0]   sg_rd_q;

  // accept gs only in W4B mode, while a job is loaded, below the expected
  // count — overrun back-pressures (observable, never silent)
  assign igs_ready = (st != ST_IDLE) && mode_q && (gs_wcnt < ngtot_q);
  wire gs_wr = igs_valid && igs_ready;

  always_ff @(posedge clk) begin
    if (gs_wr) gs_ram[gs_wcnt[GW-1:0]] <= igs_data;
    sg_rd_q <= gs_ram[gid_rd];
  end

  // ── unpack front ((A) S4 nibble map: element e at bits [4e +: 4]) ────────
  // per-lane 4-bit codes for the CURRENT half of the packed beat
  logic signed [3:0] code_l [8];
  for (genvar r = 0; r < 8; r++) begin : g_nib
    assign code_l[r] = half_q ? signed'(ipw_beat.data[4*r + 32 +: 4])
                              : signed'(ipw_beat.data[4*r      +: 4]);
  end

  // current beat's gid (all 8 lanes share it — header proof)
  logic [GW-1:0] gid_now;
  assign gid_now = GW'(32'(col_q) * 32'(ngk_q) + 32'(prow_q >> S3));
  assign gid_rd  = gid_now;

  // scale residency: emission of this beat may proceed only when its group
  // scale has been written (ascending-gid contract makes this one compare)
  wire sg_here = (32'(gs_wcnt) > 32'(gid_now));

  // ── issue / credit control ────────────────────────────────────────────────
  // divider pipes are free-running (no ready): gate issuance on output-FIFO
  // credit so nothing in flight can ever be dropped.
  // OF_DEPTH covers the ~16-cycle front+divider+back latency with margin
  localparam int unsigned OF_DEPTH = 32;
  localparam int unsigned OFW      = $clog2(OF_DEPTH);

  logic [OFW:0] credit_used;                    // in-flight + FIFO occupancy
  wire          credit_ok = (32'(credit_used) < OF_DEPTH);

  wire w4b_run   = (st == ST_RUN) && mode_q;
  wire pass_run  = (st == ST_RUN) && !mode_q;

  // a W4B issue consumes: the gs residency, one pipe slot, and (on half 1
  // or odd-tail half 0) the packed beat
  wire issue = w4b_run && ipw_valid && sg_here && credit_ok
               && (emit_cnt < beats_q);
  wire last_emit  = (emit_cnt == beats_q - 16'd1);
  // odd tail: the final packed beat's upper half is padding — consume the
  // packed beat at half 0 when it is the last emitted beat
  wire pw_consume = issue && (half_q || last_emit);
  assign ipw_ready = mode_q ? pw_consume
                            : (pass_run && credit_ok_pass);

  // ── divider pipes (8 lanes, lockstep; tag carries `last`) ────────────────
  // stage F: register the front (mirrors cq_quant_pipe's input stage)
  logic                     f_vld;
  logic                     f_last;
  cq_fp_pkg::quant_front_t  f_r [8];
  logic [15:0]              sg_use;

  // sg_rd_q is registered one cycle behind gid_rd; issue timing: gid_rd is
  // set combinationally from the CURRENT beat which stalls until sg_here,
  // and the RAM read registers in the SAME cycle the front registers uses
  // the bypass value — so feed the front from the RAM's write-through when
  // the scale arrived this very cycle, else the registered read.
  // Simpler and race-free: sg_use = gs_ram[gid_now] read combinationally is
  // not BRAM-friendly; instead delay issue by one cycle after sg_here first
  // becomes true for a NEW gid. Implementation: two-stage issue — stage A
  // (address) registers gid; stage B (front) uses sg_rd_q. The A/B skew is
  // folded into the existing half/beat walk: front consumes the A-stage
  // registers below.

  // A-stage registers (address/issue)
  logic              a_vld;
  logic              a_last;
  logic signed [3:0] a_code [8];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      a_vld  <= 1'b0;
      a_last <= 1'b0;
    end else begin
      a_vld  <= issue;
      a_last <= issue && last_emit;
      for (int r = 0; r < 8; r++) a_code[r] <= code_l[r];
    end
  end

  assign sg_use = sg_rd_q;                       // aligned with a_* by the RAM reg

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      f_vld  <= 1'b0;
      f_last <= 1'b0;
    end else begin
      f_vld  <= a_vld;
      f_last <= a_last;
      for (int r = 0; r < 8; r++)
        f_r[r] <= w4b_fp_pkg::w4b_quant_front(a_code[r], sg_use, s8_q);
    end
  end

  wire         pflush = (st == ST_IDLE);         // reset/abort kills in-flight
  // the cq pipes reset asynchronously; hand them their own registered copy
  // so the rst_n NET keeps a single (synchronous) style in this module
  logic rst_n_pipe;
  always_ff @(posedge clk) rst_n_pipe <= rst_n;
  logic        pv_out [8];
  logic [12:0] pmag   [8];
  logic [0:0]  ptag_o [8];

  for (genvar r = 0; r < 8; r++) begin : g_pipe
    cq_rne_div_pipe #(.TAG_W(1)) u_div (
      .clk       (clk),
      .rst_n     (rst_n_pipe),
      .flush     (pflush),
      .in_valid  (f_vld),
      .n         (f_r[r].n),
      .d         (f_r[r].d),
      .tag_in    (f_last ? 1'b1 : 1'b0),
      .out_valid (pv_out[r]),
      .q         (pmag[r]),
      .tag_out   (ptag_o[r])
    );
  end

  // neg/zero ride a matched-delay sideband (the pipes carry only n/d/tag);
  // depth covers front-to-out latency with margin
  localparam int unsigned SB_D = 24;
  logic [7:0]  sb_neg  [SB_D];
  logic [7:0]  sb_zero [SB_D];
  logic [$clog2(SB_D)-1:0] sb_wp, sb_rp;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sb_wp <= '0;
      sb_rp <= '0;
    end else begin
      if (pflush) begin
        sb_wp <= '0;
        sb_rp <= '0;
      end else begin
        if (f_vld) begin
          for (int r = 0; r < 8; r++) begin
            sb_neg [sb_wp][r] <= f_r[r].neg;
            sb_zero[sb_wp][r] <= f_r[r].zero;
          end
          sb_wp <= ($clog2(SB_D))'((32'(sb_wp) + 1) % SB_D);
        end
        if (pv_out[0])
          sb_rp <= ($clog2(SB_D))'((32'(sb_rp) + 1) % SB_D);
      end
    end
  end

  // back stage: certified clamp/negate, assemble the INT8 beat
  lane8_beat_t bk_beat;
  always_comb begin
    bk_beat = '0;
    for (int r = 0; r < 8; r++)
      bk_beat.data[8*r +: 8] = 8'(cq_fp_pkg::quant_back(
          pmag[r], sb_neg[sb_rp][r], sb_zero[sb_rp][r], 8));
  end

  // ── output FIFO (credit-backed; also carries passthrough beats) ──────────
  logic [$bits(lane8_beat_t):0] of_mem [OF_DEPTH];   // {last, beat}
  logic [OFW-1:0] of_wp, of_rp;
  logic [OFW:0]   of_cnt;

  // passthrough uses the same FIFO/credit (keeps ONE output path)
  wire credit_ok_pass = (32'(credit_used) < OF_DEPTH);
  wire pass_issue = pass_run && ipw_valid && credit_ok_pass
                    && (emit_cnt < beats_q);
  wire of_push_w4b  = mode_q && pv_out[0];
  wire of_push_pass = !mode_q && pass_issue;
  wire of_push = of_push_w4b || of_push_pass;
  wire [$bits(lane8_beat_t):0] of_wdata =
      of_push_w4b ? {ptag_o[0][0], $bits(lane8_beat_t)'(bk_beat)}
                  : {(emit_cnt == beats_q - 16'd1),
                     $bits(lane8_beat_t)'(ipw_beat)};

  logic        o_valid, o_ready;
  wire of_pop = o_valid && o_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      of_wp <= '0; of_rp <= '0; of_cnt <= '0;
    end else if (st == ST_IDLE) begin
      of_wp <= '0; of_rp <= '0; of_cnt <= '0;
    end else begin
      if (of_push) begin
        of_mem[of_wp] <= of_wdata;
        of_wp <= of_wp + OFW'(1);
      end
      if (of_pop) of_rp <= of_rp + OFW'(1);
      of_cnt <= of_cnt + (OFW+1)'(of_push) - (OFW+1)'(of_pop);
    end
  end

  assign o_valid = (of_cnt != '0);

  // credit: W4B counts issues (pipe in-flight + FIFO); passthrough counts
  // pushes directly. Decrement on pop.
  always_ff @(posedge clk) begin
    if (!rst_n)              credit_used <= '0;
    else if (st == ST_IDLE)  credit_used <= '0;
    else credit_used <= credit_used
                        + (OFW+1)'(mode_q ? 32'(issue) : 32'(pass_issue))
                        - (OFW+1)'(of_pop);
  end

  // output skid → xw
  logic [$bits(lane8_beat_t):0] o_data;
  assign o_data = of_mem[of_rp];
  logic [$bits(lane8_beat_t):0] m_bus;
  logic m_valid_i, m_last_unused;

  stream_skid #(.WIDTH($bits(lane8_beat_t)+1)) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  (o_data),
    .m_valid (m_valid_i),
    .m_ready (xw_ready),
    .m_data  (m_bus)
  );
  assign xw_valid = m_valid_i;
  assign xw_beat  = lane8_beat_t'(m_bus[$bits(lane8_beat_t)-1:0]);
  assign m_last_unused = m_bus[$bits(lane8_beat_t)];
  wire unused_ml = &{1'b0, m_last_unused};

  // ── FSM / counters ────────────────────────────────────────────────────────
  assign busy = (st != ST_IDLE);

  wire out_beat_acc = xw_valid && xw_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st               <= ST_IDLE;
      mode_q           <= 1'b0;
      beats_q          <= '0;
      s8_q             <= '0;
      k_q              <= '0;
      n_q              <= '0;
      ngk_q            <= '0;
      ngtot_q          <= '0;
      emit_cnt         <= '0;
      out_cnt          <= '0;
      col_q            <= '0;
      prow_q           <= '0;
      half_q           <= 1'b0;
      gs_wcnt          <= '0;
      done             <= 1'b0;
      job_error        <= 1'b0;
      job_error_sticky <= 1'b0;
    end else begin
      done      <= 1'b0;
      job_error <= 1'b0;
      if (out_beat_acc) out_cnt <= out_cnt + 16'd1;
      if (gs_wr)        gs_wcnt <= gs_wcnt + (GW+1)'(1);

      case (st)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (!job_ok) begin
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
            end else begin
              mode_q   <= w4b_en;
              beats_q  <= job_beats;
              s8_q     <= job_s8;
              k_q      <= job_k;
              n_q      <= job_n;
              ngk_q    <= (GW+1)'((32'(job_k) + G - 1) / G);
              ngtot_q  <= (GW+1)'(32'(job_n) * ((32'(job_k) + G - 1) / G));
              emit_cnt <= '0;
              out_cnt  <= '0;
              col_q    <= '0;
              prow_q   <= '0;
              half_q   <= 1'b0;
              gs_wcnt  <= '0;
              st       <= ST_RUN;
            end
          end
        end

        ST_RUN: begin
          if (mode_q ? issue : pass_issue) begin
            emit_cnt <= emit_cnt + 16'd1;
            half_q   <= mode_q ? ~half_q : half_q;
            // beat walk: c fast, p slow (beat n = p*N + c)
            if (32'(col_q) == 32'(n_q) - 1) begin
              col_q  <= '0;
              prow_q <= prow_q + 13'd1;
            end else begin
              col_q <= col_q + 4'd1;
            end
            if (emit_cnt == beats_q - 16'd1) st <= ST_DRAIN;
          end
        end

        ST_DRAIN: begin
          if (out_cnt == beats_q) begin
            done <= 1'b1;
            st   <= ST_IDLE;
          end
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

  // unused-field consumers (lint hygiene, house style)
  wire unused_meta = &{1'b0, ipw_beat.last, k_q, n_q};

endmodule

`default_nettype wire
