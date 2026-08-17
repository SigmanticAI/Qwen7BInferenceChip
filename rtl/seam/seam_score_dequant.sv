// seam_score_dequant.sv — D-021 seam block 2: per-element score dequant.
//
// Implements ARCHITECTURE.md D-021 / S-3 (golden/apex_golden/attention.py
// score_dequant_fx — the NORMATIVE arithmetic), C-2 (RNE), C-5 (INT32
// scores), §5 stream contract, D-006 job semantics.
//
// FUNCTION (one job = one score tile of job_cols columns):
//   1. The composite-scale sideband loads one fp32 constant per column:
//        comp[t] = f32( s_q * s_k[t] * 2^SCORE_FRAC / sqrt(D) )
//      (computed once per tile by the integration glue / SEQ — the single
//       f32 narrowing of S-3 happens THERE; this block consumes the bits).
//   2. For each INT32 accumulator lane (MXE OS result beats, 8 lanes/beat):
//        score[t] = RNE( acc[t] * comp[t] ),  asserted to be INT32
//      emitted as a serial 32-bit score stream (+ last) — the ASU
//      (asu_softmax s_score/s_last) input shape.
//
// BIT-EXACTNESS PROOF (vs the float64 golden, score_dequant_fx):
//   Golden computes rne( f64(acc) * f64(comp) ). The f64 multiply returns
//   the exact 56-bit product P = |acc| * mant24(comp) with its significand
//   rounded to 53 bits (RNE); np.rint then RNE-rounds that f64 value to an
//   integer. This module replicates BOTH roundings in exact integer
//   arithmetic: P is computed exactly (32x24 multiply), its significand is
//   pre-rounded to 53 bits half-to-even iff P >= 2^53 (at most 2 bits
//   dropped since P < 2^55), then the power-of-two exponent shift
//   (exp(comp) - 23) is applied with half-to-even on the shifted-out bits.
//   RNE is sign-symmetric, so magnitude arithmetic + sign is exact.
//   Domain: the ENTIRE int32 x finite-fp32 domain (subnormal comp included:
//   hidden bit 0, exponent -126). Only inf/NaN comp is out of contract
//   (SVA-checked in verif/seam); the S-3 composite is always a positive
//   normal fp32 (EPS^2 * 2^10 / sqrt(128) ~ 2^-21.5 at the floor).
//
// RANGE (the golden `assert |score| < 2^31` mirrored in hardware):
//   any element whose rounded magnitude is >= 2^31 raises a one-cycle
//   range_error pulse (+ sticky) at the element's accept and the output is
//   saturated to +/-(2^31 - 1). In-contract S-3 traffic never trips this.
//
// INPUT FRAMING (checked, not assumed): the acc stream must carry last=1 on
// exactly the job's final beat (the integration glue frames per score job;
// an MXE M=1 tile job naturally complies). Violations raise frame_error
// pulse + sticky. Tail lanes of the final beat (job_cols % 8 != 0) are
// don't-care.
//
// Job legality (§3-style): 1 <= job_cols <= N_MAX, else job_error pulse
// (one cycle after the handshake) + sticky, NO state change.
//
// D-006: done <=> every score of the job ACCEPTED downstream (post-skid);
// job_ready gated until strictly after done. busy = (FSM != IDLE) || score
// beats pending in the output skid. All ports skid-buffered (§5, D-019).

module seam_score_dequant
  import apex_pkg::*;
#(
  parameter int unsigned N_MAX = 1024   // max columns (tokens) per job
)(
  input  logic              clk,
  input  logic              rst_n,          // synchronous, active low

  // job interface (D-006)
  input  logic              job_valid,
  output logic              job_ready,
  input  logic [DIM_W-1:0]  job_cols,
  output logic              job_error,
  output logic              job_error_sticky,
  output logic              busy,
  output logic              done,

  // composite-scale sideband: one fp32 per column, loaded per job
  input  logic              cmp_valid,
  output logic              cmp_ready,
  input  logic [31:0]       cmp_data,

  // INT32 accumulator input (MXE OS result stream shape)
  input  logic              acc_valid,
  output logic              acc_ready,
  input  lane32_beat_t      acc_beat,

  // score output stream (ASU asu_softmax s_score/s_last shape)
  output logic              score_valid,
  input  logic              score_ready,
  output logic signed [31:0] score_data,
  output logic              score_last,

  // hardware mirrors of the golden checks
  output logic              range_error,        // |score| >= 2^31 (golden assert)
  output logic              range_error_sticky,
  output logic              frame_error,        // acc last mis-framed
  output logic              frame_error_sticky
);

  // ── parameter legality (elaboration-time) ─────────────────────────────────
  if (N_MAX < 8 || N_MAX > 4095) begin : g_chk_nmax
    $error("seam_score_dequant: N_MAX must be in [8, 4095]");
  end

  localparam int unsigned CNT_W = 13;   // column counters (N_MAX <= 4095)

  // ── input skids (§5) ──────────────────────────────────────────────────────
  logic        c_valid, c_ready;
  logic [31:0] c_data;

  stream_skid #(.WIDTH(32)) u_cmp_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (cmp_valid),
    .s_ready (cmp_ready),
    .s_data  (cmp_data),
    .m_valid (c_valid),
    .m_ready (c_ready),
    .m_data  (c_data)
  );

  logic                             a_valid, a_ready;
  logic [$bits(lane32_beat_t)-1:0]  a_vec;
  lane32_beat_t                     a_beat;

  stream_skid #(.WIDTH($bits(lane32_beat_t))) u_acc_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (acc_valid),
    .s_ready (acc_ready),
    .s_data  ({acc_beat.data, acc_beat.last}),
    .m_valid (a_valid),
    .m_ready (a_ready),
    .m_data  (a_vec)
  );
  assign a_beat = lane32_beat_t'(a_vec);

  // ── output skid ───────────────────────────────────────────────────────────
  logic        o_valid, o_ready;
  logic [32:0] o_data;
  logic [32:0] score_vec;

  stream_skid #(.WIDTH(33)) u_score_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  (o_data),
    .m_valid (score_valid),
    .m_ready (score_ready),
    .m_data  (score_vec)
  );
  assign score_data = $signed(score_vec[32:1]);
  assign score_last = score_vec[0];

  // ── S-3 element arithmetic: {ovf, score} = RNE(acc * comp) ────────────────
  function automatic logic [32:0] score_calc(input logic signed [31:0] a,
                                             input logic [31:0]        c);
    logic        sc, sgn;
    logic [7:0]  ce;
    logic [23:0] mc;
    logic [31:0] amag;
    logic [55:0] p;
    logic [56:0] p53, fl, magt;
    logic        inc;
    int          ceu, sh, t;
    logic [87:0] magw;
    logic [56:0] rem, half;
    logic [30:0] mag;
    logic        ovf;
    sc = c[31];
    ce = c[30:23];
    mc = {(ce != 8'h00), c[22:0]};             // subnormal comp: hidden bit 0
    if (a == 32'sd0 || mc == 24'd0) return 33'd0;
    sgn = a[31] ^ sc;
    if (ce == 8'hFF)                            // inf/NaN comp: out of contract
      return {1'b1, sgn ? 32'h8000_0001 : 32'h7FFF_FFFF};
    amag = a[31] ? (~$unsigned(a) + 32'd1) : $unsigned(a);   // |-2^31| ok

    // exact 56-bit product, then f64-significand pre-round (53 bits, RNE)
    p = 56'(amag) * 56'(mc);                    // < 2^55
    if (p[54]) begin
      inc = p[1] && (p[0] || p[2]);
      p53 = (({3'b0, p[55:2]}) + {56'b0, inc}) << 2;
    end else if (p[53]) begin
      inc = p[0] && p[1];
      p53 = (({2'b0, p[55:1]}) + {56'b0, inc}) << 1;
    end else begin
      p53 = {1'b0, p};
    end

    ceu = (ce == 8'h00) ? -126 : int'({24'b0, ce}) - 127;
    sh  = ceu - 23;                             // in [-149, 104]
    ovf = 1'b0;
    mag = '0;
    if (sh >= 0) begin
      if (sh >= 32) begin
        ovf = 1'b1;                             // p53 >= 1 here -> mag >= 2^32
      end else begin
        magw = {31'b0, p53} << sh;
        ovf  = |magw[87:31];
        mag  = magw[30:0];
      end
    end else begin
      t = -sh;                                  // in [1, 149]
      if (t >= 56) begin
        mag = '0;                               // value <= 0.5, tie -> even 0
      end else begin
        fl   = p53 >> t;
        rem  = p53 & ((57'd1 << t) - 57'd1);
        half = 57'd1 << (t - 1);
        inc  = (rem > half) || ((rem == half) && fl[0]);   // half to even
        magt = fl + {56'b0, inc};
        ovf  = |magt[56:31];
        mag  = magt[30:0];
      end
    end
    if (ovf) return {1'b1, sgn ? 32'h8000_0001 : 32'h7FFF_FFFF};
    return {1'b0, sgn ? (~{1'b0, mag} + 32'd1) : {1'b0, mag}};
  endfunction

  // ── FSM / state ───────────────────────────────────────────────────────────
  typedef enum logic [1:0] {
    ST_IDLE = 2'd0,
    ST_LOAD = 2'd1,   // load the per-column composite scales
    ST_PROC = 2'd2,   // dequant lanes, serialize scores
    ST_WAIT = 2'd3    // D-006: wait for post-skid acceptance
  } state_e;

  state_e            state;
  logic [DIM_W-1:0]  cols_q;
  logic [CNT_W-1:0]  load_cnt;
  logic [CNT_W-1:0]  col_idx;
  logic [CNT_W-1:0]  beat_no;      // consumed acc beats
  logic [CNT_W-1:0]  n_beats;      // ceil(cols/8)
  logic [CNT_W-1:0]  score_acc;    // POST-SKID accepted scores
  logic              beat_have;
  logic [255:0]      lanes;
  logic [31:0]       cmp_mem [N_MAX];

  logic legal;
  assign legal = (job_cols != '0) && (job_cols <= DIM_W'(N_MAX));

  // handshakes
  assign c_ready = (state == ST_LOAD);
  assign a_ready = (state == ST_PROC) && !beat_have;

  logic signed [31:0] lane_acc;
  logic [32:0]        calc;
  assign lane_acc = $signed(lanes[32*col_idx[2:0] +: 32]);
  assign calc     = score_calc(lane_acc, cmp_mem[col_idx[$clog2(N_MAX)-1:0]]);

  assign o_valid = (state == ST_PROC) && beat_have;
  assign o_data  = {calc[31:0], (col_idx == CNT_W'(cols_q) - CNT_W'(1))};

  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || score_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state              <= ST_IDLE;
      done               <= 1'b0;
      job_error          <= 1'b0;
      job_error_sticky   <= 1'b0;
      range_error        <= 1'b0;
      range_error_sticky <= 1'b0;
      frame_error        <= 1'b0;
      frame_error_sticky <= 1'b0;
      cols_q             <= '0;
      load_cnt           <= '0;
      col_idx            <= '0;
      beat_no            <= '0;
      n_beats            <= '0;
      score_acc          <= '0;
      beat_have          <= 1'b0;
      lanes              <= '0;
    end else begin
      done        <= 1'b0;
      job_error   <= 1'b0;
      range_error <= 1'b0;
      frame_error <= 1'b0;
      if (score_valid && score_ready) score_acc <= score_acc + CNT_W'(1);

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state     <= ST_LOAD;
              cols_q    <= job_cols;
              n_beats   <= CNT_W'((32'(job_cols) + 32'd7) >> 3);
              load_cnt  <= '0;
              col_idx   <= '0;
              beat_no   <= '0;
              score_acc <= '0;
              beat_have <= 1'b0;
            end else begin                 // §3: pulse + sticky, no effects
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
            end
          end
        end

        ST_LOAD: begin
          if (c_valid && c_ready) begin
            cmp_mem[load_cnt[$clog2(N_MAX)-1:0]] <= c_data;
            if (load_cnt == CNT_W'(cols_q) - CNT_W'(1)) begin
              state <= ST_PROC;
            end else begin
              load_cnt <= load_cnt + CNT_W'(1);
            end
          end
        end

        ST_PROC: begin
          if (!beat_have) begin
            if (a_valid && a_ready) begin
              lanes     <= a_beat.data;
              beat_have <= 1'b1;
              // input framing check: last exactly on the job's final beat
              if (a_beat.last != (beat_no == n_beats - CNT_W'(1))) begin
                frame_error        <= 1'b1;
                frame_error_sticky <= 1'b1;
              end
              beat_no <= beat_no + CNT_W'(1);
            end
          end else begin
            if (o_ready) begin               // score accepted into out skid
              if (calc[32]) begin            // golden range assert mirror
                range_error        <= 1'b1;
                range_error_sticky <= 1'b1;
              end
              if (col_idx == CNT_W'(cols_q) - CNT_W'(1)) begin
                state     <= ST_WAIT;
                beat_have <= 1'b0;
              end else begin
                if (col_idx[2:0] == 3'd7) beat_have <= 1'b0;
                col_idx <= col_idx + CNT_W'(1);
              end
            end
          end
        end

        ST_WAIT: begin                       // D-006: post-skid acceptance
          if (score_acc == CNT_W'(cols_q)) begin
            done  <= 1'b1;
            state <= ST_IDLE;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
