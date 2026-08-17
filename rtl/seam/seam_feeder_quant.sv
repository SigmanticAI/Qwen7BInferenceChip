// seam_feeder_quant.sv — D-021 seam block 1: per-token INT8 feeder requant.
//
// Implements ARCHITECTURE.md D-021 (K̂/V̂/Q feeder requant), C-1 quant rule,
// C-2 (RNE), D-010 (fp32 read bus), §5 stream contract, D-006 job semantics.
// Golden arbiter: golden/apex_golden/attention.py quant_rows_i8 (stages
// S-1/Q6/Q7) — this module is bit-exact against it (proof below + TB).
//
// FUNCTION (one job = one row-batch, D-006):
//   For each of job_rows rows of D fp32 elements (KVQ read-bus values or the
//   fp16-widened hidden vector):
//     pass 1: buffer the row, amax = max|x_i|            (IEEE-magnitude max)
//     scale : s = fp16_RNE( max(amax/127, EPS) ),  EPS = 2^-14   (C-1)
//     pass 2: q_i = clamp(RNE(x_i / s), -128, 127)                (C-1)
//   Emits the fp16 scale on the sideband FIRST, then the D INT8 codes packed
//   LSB-first into lane8_beat_t beats (element 8k+j -> byte j of beat k, the
//   MXE act-stream convention). out_beat.last marks the job's final beat;
//   scl_last marks the job's final row's scale.
//
// BIT-EXACTNESS PROOFS (vs the float64 golden, quant_rows_i8):
//   P1 (scale): golden computes RNE_f64(amax/127) then RNE_f16 of that
//     (double rounding). Exact ties amax/127 = (2m+1)*2^(E-11) are f64-exact
//     (<= 18 significant bits), so both paths agree there. A non-tie exact
//     quotient is >= 2^(E-23)-far from any f16 tie point (127*q*2^(11-E) is a
//     nonzero integer difference), while the f64 rounding error is
//     < 2^(E-42): the intermediate f64 rounding can never cross an f16 tie.
//     Hence directly rounding the exact quotient to f16 (done here with a
//     14-extra-bit division by 127 + sticky) equals the golden double round.
//   P2 (EPS floor): golden takes max(amax/127, EPS) BEFORE rounding. Exact
//     quotient < 2^-14 <=> its pre-round exponent E < -14; any such value
//     maps to EPS = 16'h0400. Values >= 2^-14 round to >= 2^-14 (monotone),
//     so the floor never applies post-rounding.
//   P3 (codes): golden computes RNE_int(RNE_f64(x/s)). The exact quotient
//     r = x/s = (2q+1)/2 exactly at a tie is f64-exact; a non-tie r is
//     >= 1/(2^(14-e+1)*ms) >= 2^-26-far from any half-integer while the f64
//     quotient error is <= 129*2^-53 < 2^-46, so the double rounding equals
//     exact-rational RNE — computed here as an integer division with
//     remainder (mx / (ms << (13-e))) and half-to-even on the remainder.
//   P4 (windows, mx/ms in (2^12, 2^14) with e = exp(x)-exp(s)):
//     e <= -2  => r < 2^-1        => code 0 (no tie reachable)
//     e >=  8  => r > 2^7 = 128   => saturate (defensive; see P5)
//     e in [-1, 7] => exact divide, quotient <= 255, mag <= 256.
//   P5 (clamp reachability): with s derived from the row's own amax,
//     |x|/s <= 127/(1 - 2^-12) < 127.04 < 127.5, so the +/-127/-128 clamps
//     and the e >= 8 window are DEFENSIVE (unreachable from in-contract
//     input). They are kept (never weaken checks) and documented as such in
//     the coverage report.
//   P6 (overflow): amax/127 rounding to > 65504 (f16 max, tie at 65520)
//     replicates golden: s = +inf (16'h7C00) and every code is 0 (x/inf).
//
// INPUT CONTRACT (SVA-checked in verif/seam): fp32 elements are finite
// (no inf/NaN). Subnormal and +/-0 inputs are fully supported (code 0 and
// the EPS floor make them exact).
//
// Job legality (§3-style): 1 <= job_rows <= ROWS_MAX, else job_error pulse
// (one cycle after the handshake) + sticky, NO state change, no busy/done.
//
// D-006: done <=> every beat AND every scale of the job has been ACCEPTED
// downstream (post-skid, both sidebands); job_ready gated until strictly
// after done. busy = (FSM != IDLE) || beats pending in either output skid.
//
// Latency/throughput: ~(D ingest + 4 + D*9/8 drain) cycles per row,
// unstalled (2 scale/push + 2 drain-pipeline prime); streams are 2-deep
// skid-buffered on ALL ends (§5, D-019).

module seam_feeder_quant
  import apex_pkg::*;
#(
  parameter int unsigned D        = 64,    // row length (D-021: 64 or 128)
  parameter int unsigned ROWS_MAX = 256    // rows per job upper bound
)(
  input  logic              clk,
  input  logic              rst_n,          // synchronous, active low

  // job interface (row-batch descriptor, D-006)
  input  logic              job_valid,
  output logic              job_ready,
  input  logic [DIM_W-1:0]  job_rows,
  output logic              job_error,
  output logic              job_error_sticky,
  output logic              busy,
  output logic              done,

  // fp32 element stream in (KVQ read bus / widened hidden), one element/beat
  input  logic              in_valid,
  output logic              in_ready,
  input  logic [31:0]       in_data,

  // INT8 row stream out (packed MXE lanes)
  output logic              out_valid,
  input  logic              out_ready,
  output lane8_beat_t       out_beat,

  // fp16 per-row scale sideband (emitted before the row's beats)
  output logic              scl_valid,
  input  logic              scl_ready,
  output logic [15:0]       scl_data,
  output logic              scl_last
);

  // ── parameter legality (elaboration-time) ─────────────────────────────────
  if (!(D == 64 || D == 128)) begin : g_chk_d
    $error("seam_feeder_quant: D must be 64 or 128 (D-021)");
  end
  if (ROWS_MAX == 0 || ROWS_MAX > 4095) begin : g_chk_rows
    $error("seam_feeder_quant: ROWS_MAX must be in [1, 4095]");
  end

  localparam int unsigned BPR   = D / 8;            // beats per row (power of 2)
  localparam int unsigned BW    = $clog2(BPR);      // row-boundary beat bits
  localparam int unsigned EW    = $clog2(D) + 1;    // element counters (0..D+2)
  localparam int unsigned CNT_W = 17;               // beat counters

  // ── input skid (§5: every boundary crossing) ──────────────────────────────
  logic        i_valid, i_ready;
  logic [31:0] i_data;

  stream_skid #(.WIDTH(32)) u_in_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (in_valid),
    .s_ready (in_ready),
    .s_data  (in_data),
    .m_valid (i_valid),
    .m_ready (i_ready),
    .m_data  (i_data)
  );

  // ── output skids ──────────────────────────────────────────────────────────
  logic                              o_valid, o_ready;
  lane8_beat_t                       o_beat;
  logic [$bits(lane8_beat_t)-1:0]    out_vec;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  ({o_beat.data, o_beat.last}),
    .m_valid (out_valid),
    .m_ready (out_ready),
    .m_data  (out_vec)
  );
  assign out_beat = lane8_beat_t'(out_vec);

  logic        sc_valid, sc_ready;
  logic [16:0] sc_data;
  logic [16:0] scl_vec;

  stream_skid #(.WIDTH(17)) u_scl_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (sc_valid),
    .s_ready (sc_ready),
    .s_data  (sc_data),
    .m_valid (scl_valid),
    .m_ready (scl_ready),
    .m_data  (scl_vec)
  );
  assign scl_data = scl_vec[15:0];
  assign scl_last = scl_vec[16];

  // ── C-1 scale: s = fp16_RNE(max(amax/127, EPS)) — proofs P1/P2/P6 ─────────
  function automatic logic [15:0] scale_from_amax(input logic [30:0] mag);
    logic [7:0]  aexp;
    logic [22:0] aman;
    logic [23:0] ma;
    logic [37:0] num;
    logic [31:0] q;
    logic [6:0]  r;
    logic [10:0] mant11;
    logic        rndb, stk, inc;
    logic [11:0] mant12;
    logic        unused_hid;
    int          es;
    aexp = mag[30:23];
    aman = mag[22:0];
    if (aexp == 8'h00) return 16'h0400;    // amax zero/subnormal -> EPS (P2)
    if (aexp == 8'hFF) return 16'h7C00;    // out-of-contract input (SVA)
    ma  = {1'b1, aman};
    num = {ma, 14'b0};                     // 14 extra quotient bits + sticky
    q   = 32'(num / 38'd127);              // quotient in [2^30, 2^32)
    r   = 7'(num % 38'd127);
    if (q[31]) begin                       // quotient in [2^31, 2^32)
      mant11 = q[31:21];
      rndb   = q[20];
      stk    = (|q[19:0]) || (r != 7'd0);
      es     = int'({24'b0, aexp}) - 127 - 6;
    end else begin                         // quotient in [2^30, 2^31)
      mant11 = q[30:20];
      rndb   = q[19];
      stk    = (|q[18:0]) || (r != 7'd0);
      es     = int'({24'b0, aexp}) - 127 - 7;
    end
    if (es < -14) return 16'h0400;         // exact quotient < 2^-14 -> EPS (P2)
    inc    = rndb && (stk || mant11[0]);   // round half to even (C-2)
    mant12 = {1'b0, mant11} + {11'b0, inc};
    unused_hid = mant12[10];               // hidden bit (implied downstream)
    if (mant12[11]) begin                  // mantissa carry: 2.0 -> 1.0 * 2^+1
      mant12 = 12'h400;
      es     = es + 1;
    end
    if (es > 15) return 16'h7C00;          // fp16 overflow -> +inf (P6)
    return {1'b0, 5'(es + 15), mant12[9:0]};
  endfunction

  // ── C-1 code: q = clamp(RNE(x/s)) — proofs P3/P4/P5 ───────────────────────
  // 2026-08-07 (the A0 62.5 MHz timing hunt): the one-cycle form — rbuf read
  // -> fp32 decode -> 25/25 combinational DIVIDE -> round -> pack — measured
  // 138 logic levels / 97 CARRY8 at post-route, endpoints pack_reg[*],
  // startpoint elem_idx_reg (build cl_apex.2026_08_07-175327; the tier after
  // apex_scale_quant's 39.2 ns cone, same disease). Split at the num/den
  // boundary into a 3-stage drain pipeline (read -> code_pre -> code_div),
  // the apex_scale_quant pattern. The composition code_div(code_pre(x)) is
  // the OLD code_from expression-for-expression; only a register sits
  // between them. The divide is an explicit 8-iteration restoring divider:
  // the quotient provably fits 8 bits (P4: e <= 7 => den >= 2^16 while
  // nn < 2^24 => qd <= 255), which the generic `/` operator cannot know —
  // it built the full-width array.
  function automatic logic [52:0] code_pre(input logic [31:0] xb,
                                           input logic [15:0] sb);
    logic        sx;
    logic [7:0]  xe;
    logic [22:0] xm;
    logic [10:0] ms;
    int          e;
    logic [24:0] den, nn;
    logic        zero, sat;
    sx   = xb[31];
    xe   = xb[30:23];
    xm   = xb[22:0];
    ms   = {1'b1, sb[9:0]};                // s >= EPS: always normal fp16
    e    = int'({24'b0, xe}) - 127 - (int'({27'b0, sb[14:10]}) - 15);
    zero = (sb == 16'h7C00)                // inf scale: x/inf -> 0 (P6)
           || (xe == 8'h00)                // zero/subnormal x: |x/s| < 2^-112
           || ((xe != 8'hFF) && (e <= -2));            // |x/s| < 1/2 (P4)
    sat  = !zero && ((xe == 8'hFF)         // out-of-contract (SVA)
                     || (e >= 8));         // |x/s| > 128 (P4, defensive)
    if (zero || sat) begin
      nn  = '0;                            // keep stage C benign
      den = 25'd1;
    end else begin
      den = 25'(ms) << (13 - e);           // shift in [6, 14] -> den < 2^25
      nn  = {2'b01, xm};                   // 24-bit x mantissa
    end
    return {nn, den, sx, zero, sat};
  endfunction

  function automatic logic signed [7:0] code_div(input logic [24:0] nn,
                                                 input logic [24:0] den,
                                                 input logic        sx,
                                                 input logic        zero,
                                                 input logic        sat);
    logic [31:0] r;                        // running remainder
    logic [31:0] dsh;
    logic [7:0]  qd;
    logic [24:0] rd;
    logic [25:0] r2, d2;
    logic        inc;
    logic [8:0]  mag;
    if (zero) return 8'sd0;
    if (sat)  return sx ? 8'sh80 : 8'sd127;
    r = 32'(nn);
    for (int i = 7; i >= 0; i--) begin     // restoring divide, 8 quotient
      dsh = 32'(den) << i;                 // bits (exact rational div, P3)
      if (r >= dsh) begin
        r     = r - dsh;
        qd[i] = 1'b1;
      end else qd[i] = 1'b0;
    end
    rd  = r[24:0];                         // nn % den
    r2  = {rd, 1'b0};
    d2  = {1'b0, den};
    inc = (r2 > d2) || ((r2 == d2) && qd[0]);    // round half to even (C-2)
    mag = {1'b0, qd} + {8'b0, inc};              // <= 256
    if (mag >= 9'd128) return sx ? 8'sh80 : 8'sd127;   // clamp (defensive, P5)
    return sx ? -$signed({1'b0, mag[6:0]}) : $signed({1'b0, mag[6:0]});
  endfunction

  // ── FSM / datapath state ──────────────────────────────────────────────────
  typedef enum logic [2:0] {
    ST_IDLE   = 3'd0,
    ST_INGEST = 3'd1,   // pass 1: buffer row, running amax
    ST_SCALE  = 3'd2,   // compute the C-1 fp16 scale
    ST_SPUSH  = 3'd3,   // offer the scale to its skid
    ST_DRAIN  = 3'd4,   // pass 2: quantize + pack + push beats
    ST_WAIT   = 3'd5    // D-006: wait for post-skid acceptance
  } state_e;

  state_e            state;
  logic [DIM_W-1:0]  rows_q;
  logic [DIM_W-1:0]  row_cnt;
  logic [EW-1:0]     elem_cnt;      // ingest index
  logic [EW-1:0]     elem_idx;      // drain READ index (0..D+2, stage A)
  logic [30:0]       amax_mag;
  logic [15:0]       sbits;
  logic [63:0]       pack;
  logic [3:0]        pack_cnt;      // 0..8
  logic [CNT_W-1:0]  beat_idx;      // pre-skid pushed beats
  logic [CNT_W-1:0]  total_beats;
  logic [CNT_W-1:0]  out_acc;       // POST-SKID accepted beats
  logic [CNT_W-1:0]  scl_acc;       // POST-SKID accepted scales
  logic [31:0]       rbuf [D];
  // drain pipeline (2026-08-07 A0 timing): A=read, B=code_pre, C=code_div
  logic [31:0]       p1_x;
  logic              p1_val;
  logic [24:0]       p2_nn;
  logic [24:0]       p2_den;
  logic              p2_sgn, p2_zero, p2_sat, p2_val;

  logic legal;
  assign legal = (job_rows != '0) && (job_rows <= DIM_W'(ROWS_MAX));

  // handshakes
  assign i_ready  = (state == ST_INGEST);
  assign o_valid  = (state == ST_DRAIN) && (pack_cnt == 4'd8);
  assign o_beat.data = pack;
  assign o_beat.last = (beat_idx == total_beats - CNT_W'(1));
  assign sc_valid = (state == ST_SPUSH);
  assign sc_data  = {(row_cnt == rows_q - DIM_W'(1)), sbits};

  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || out_valid || scl_valid;

  // drain read (stage A of the pipeline; one element per cycle sustained).
  // The address is mod-D ([EW-2:0], D a power of two): the 2 prefetch reads
  // past D-1 wrap to rbuf[0..1] — in bounds, and DISCARDED at the row
  // boundary (p1/p2 cleared), never packed.
  logic [31:0] dr_x;
  assign dr_x = rbuf[elem_idx[EW-2:0]];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state            <= ST_IDLE;
      done             <= 1'b0;
      job_error        <= 1'b0;
      job_error_sticky <= 1'b0;
      rows_q           <= '0;
      row_cnt          <= '0;
      elem_cnt         <= '0;
      elem_idx         <= '0;
      amax_mag         <= '0;
      sbits            <= '0;
      pack             <= '0;
      pack_cnt         <= '0;
      beat_idx         <= '0;
      total_beats      <= '0;
      out_acc          <= '0;
      scl_acc          <= '0;
      p1_x             <= '0;
      p1_val           <= 1'b0;
      p2_nn            <= '0;
      p2_den           <= '0;
      p2_sgn           <= 1'b0;
      p2_zero          <= 1'b0;
      p2_sat           <= 1'b0;
      p2_val           <= 1'b0;
    end else begin
      done      <= 1'b0;
      job_error <= 1'b0;
      if (out_valid && out_ready) out_acc <= out_acc + CNT_W'(1);
      if (scl_valid && scl_ready) scl_acc <= scl_acc + CNT_W'(1);

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state       <= ST_INGEST;
              rows_q      <= job_rows;
              total_beats <= CNT_W'(job_rows) * CNT_W'(BPR);
              row_cnt     <= '0;
              elem_cnt    <= '0;
              amax_mag    <= '0;
              beat_idx    <= '0;
              out_acc     <= '0;
              scl_acc     <= '0;
            end else begin                 // §3: pulse + sticky, no effects
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
            end
          end
        end

        ST_INGEST: begin
          if (i_valid && i_ready) begin
            rbuf[elem_cnt[EW-2:0]] <= i_data;
            if (i_data[30:0] > amax_mag) amax_mag <= i_data[30:0];
            if (elem_cnt == EW'(D - 1)) begin
              state    <= ST_SCALE;
              elem_cnt <= '0;
            end else begin
              elem_cnt <= elem_cnt + EW'(1);
            end
          end
        end

        ST_SCALE: begin
          sbits <= scale_from_amax(amax_mag);
          state <= ST_SPUSH;
        end

        ST_SPUSH: begin
          if (sc_ready) begin                // scale accepted into its skid
            state    <= ST_DRAIN;
            elem_idx <= '0;
            pack_cnt <= '0;
            p1_val   <= 1'b0;                // pipeline starts empty (2 prime
            p2_val   <= 1'b0;                // cycles before the first pack)
          end
        end

        ST_DRAIN: begin
          // 3-stage pipe: A reads, B pre-computes num/den, C divides+packs.
          // The ONLY stall is a full pack awaiting its skid; the stages HOLD
          // through it (their contents are the next pack's lead elements),
          // so sustained throughput stays one element per cycle.
          if (pack_cnt == 4'd8) begin
            if (o_ready) begin               // beat accepted into out skid
              beat_idx <= beat_idx + CNT_W'(1);
              pack_cnt <= '0;
              // row complete: beat_idx counts the job's beats from 0 and
              // every row is exactly BPR = 2^BW beats, so the row's final
              // beat is beat_idx == BPR-1 (mod BPR); the 2 prefetched tail
              // slots (wrapped reads) held in p1/p2 are DISCARDED here.
              if (&beat_idx[BW-1:0]) begin
                p1_val <= 1'b0;
                p2_val <= 1'b0;
                if (row_cnt == rows_q - DIM_W'(1)) begin
                  state <= ST_WAIT;
                end else begin
                  row_cnt  <= row_cnt + DIM_W'(1);
                  amax_mag <= '0;
                  elem_cnt <= '0;
                  state    <= ST_INGEST;
                end
              end
            end
          end else begin
            // stage C: pack the element staged two cycles ago
            if (p2_val) begin
              pack[8*pack_cnt[2:0] +: 8]
                <= code_div(p2_nn, p2_den, p2_sgn, p2_zero, p2_sat);
              pack_cnt <= pack_cnt + 4'd1;
            end
            // stage B: everything up to the divide, registered
            {p2_nn, p2_den, p2_sgn, p2_zero, p2_sat} <= code_pre(p1_x, sbits);
            p2_val <= p1_val;
            // stage A: the rbuf read
            p1_x     <= dr_x;
            p1_val   <= 1'b1;
            elem_idx <= elem_idx + EW'(1);
          end
        end

        ST_WAIT: begin                       // D-006: post-skid acceptance
          if ((out_acc == total_beats) && (scl_acc == CNT_W'(rows_q))) begin
            done  <= 1'b1;
            state <= ST_IDLE;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
