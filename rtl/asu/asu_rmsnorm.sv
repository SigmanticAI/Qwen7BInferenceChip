// asu_rmsnorm.sv — ASU RMSNorm lane: y = x * gamma / sqrt(mean(x^2) + eps).
//
// Implements ARCHITECTURE.md §6 (RMSNorm on the vendored prior work
// rsqrt_unit) and D-018 (31-cycle latency, serialized issue, cycle-counted
// capture — the unit has NO busy/ready and silently drops mid-op requests).
//
// ── FIXED-POINT CONTRACT (this module's normative definition) ─────────────
// Inputs : x_i  signed INT8 (raw activation integers), D = row length.
//                Legal D (framed by x-stream `last`):
//                  RMS_D_MAX <= 128 : POWER OF TWO in [1, RMS_D_MAX]
//                    (§4: production D = 64 / 128; pow-2 keeps mean(x^2) an
//                    exact arithmetic shift — no hidden division rounding).
//                  RMS_D_MAX  > 128 : additionally any MULTIPLE OF 128 up to
//                    RMS_D_MAX (C-RMSW wide envelope, docs/design/
//                    WIDE_RMSNORM.md — 7B hidden 3584 = 28·128).
//          g_i  signed 16-bit Q2.13 gamma (value = g_i / 2^13, range ±4).
// Output : y_i  signed 16-bit Q7.8 (value = y_i / 2^8), RNE-rounded (C-2),
//                saturated to [-32768, 32767].
// eps    : EPS_INT = 1 in mean-of-x^2 integer units (i.e. eps = 1.0 in x's
//          squared-LSB domain), which also guarantees norm2 >= 1 so the
//          rsqrt divider path (den = isqrt(norm2) >= 1) never saturates.
//
// GOLDEN MIRROR (exact pseudocode; implemented in
// verif/asu/smoke/gen_asu_vectors.py as rmsnorm_fx — the verifier's oracle):
//   def rmsnorm_fx(x: int8[D], g: int16[D]):        # D power of 2, <= 128
//       sum2 = sum(int(xi)**2 for xi in x)          # <= 128*128^2 = 2^21,
//                                                   #   held in 22 bits (proof
//                                                   #   below)
//       mean2 = sum2 >> log2(D)                     # exact floor (pow-2 D)
//       n2    = mean2 + 1                           # + EPS_INT
//       r     = (1 << 13) // isqrt(n2)              # EXACTLY the vendored
//                                                   #   rsqrt_unit: UQ2.13,
//                                                   #   floor(8192/floor-sqrt)
//       nrm   = min(isqrt(n2), 2**14 - 1)           # unit's norm_int output
//       for i in range(D):
//           p = int(x[i]) * int(g[i])               # Q2.13, |p| <= 2^22
//           q = p * r                               # frac 13+13=26, |q| < 2^36
//           t = q >> 18                             # floor shift to Q7.8
//           rem = q & (2**18 - 1)                   # non-negative remainder
//           t += (rem > 2**17) or (rem == 2**17 and (t & 1))   # RNE (C-2)
//           y[i] = clamp(t, -32768, 32767)          # sat16
//       return y, r, nrm
//
// WIDE MIRROR (C-RMSW, RMS_D_MAX > 128 elaborations): the arbiter is
// golden/apex_golden/attention.py rmsnorm_fx_wide — identical to the above
// except the mean shift becomes ONE scalar multiply,
//       mean2 = (sum2 * MU_D) >> (S_D + 16)
// with (MU_D, S_D) = _wide_mu(D): S = ceil(log2 D), MU = RNE(2^16·2^S/D).
// The (MU, S) pairs are baked in asu_wide_rms_params.svh, GENERATED from the
// golden by gen_wide_rms_params.py (provenance-gated, tables-check pattern).
// For pow-2 D, MU == 2^16 exactly and the μ path reduces bit-exactly to
// `sum2 >> log2(D)` — which is why this RTL keeps the frozen shift path for
// D <= 128 and the two elaborations agree on the overlap. n2/r/nrm and the
// per-element emission are UNCHANGED (rsqrt_unit reused verbatim).
//
// C-RMSW CHUNK COMPOSITION (R4, 2026-07-31 — wide rows on a NARROW build):
// a build whose RMS_D_MAX forbids the wide row (the only image the vendor
// bug aws-fpga#799 lets us fly) can still compute it EXACTLY, because the
// wide contract above is ALREADY a composition of <=128-chunk passes
// (golden rmsnorm_fx_wide's own 4 steps). Two quasi-static input levels,
// both 0 by default (every path is then the frozen behavior, bit-for-bit):
//   ext_sum_en (pass 1, golden step 1): each legal row is consumed through
//     `last` and its sum2 is EXPORTED (s2_push/s2_val) — no rsqrt, no
//     emission, no gamma, no done pulse (there is no output to drain; the
//     capture count kept by the glue is the completion observable). The
//     host (or an MXE x·x dot product) ACCUMULATES the k chunk sums —
//     golden step 2's "host (or a scalar adder)", pure INT32 addition.
//   ext_r_en (pass 2, golden steps 3-4): the rsqrt operand becomes
//       mean2 = (ext_sum2 · WIDE_RMS_MU[ext_k]) >> (WIDE_RMS_S[ext_k]+16)
//     (+1 for EPS_INT) — the SAME μ arithmetic, SAME generated table as
//     the wide elaboration — and each 128-chunk re-streams through the
//     UNCHANGED per-element datapath with the ONE broadcast r. r is
//     recomputed per chunk from the same registered operand, so all k
//     captures are identical by determinism — bit-exact to golden's single
//     broadcast r.
//   LEGALITY: ext_sum2/ext_k are validated LOUDLY at the CSR arm point in
//     apex_top (k in [2,64]; ext_sum2 <= k·2^21, the all-(−128) reachable
//     max, which pins mean2 <= 2^14 — refusal = LAYER_STATUS sticky +
//     err_code 8, and the arm REFUSES while busy so the levels are
//     quasi-static by construction). This module trusts those registered,
//     validated inputs — one refusal point, no second decoder.
//
// WIDTH PROOF (sum of squares): x_i^2 <= 128^2 = 2^14, so sum2 <=
// RMS_D_MAX·2^14 and SUM_W = clog2(RMS_D_MAX·2^14 + 1) holds it exactly:
// 22 bits at RMS_D_MAX=128 (sum2 <= 2^21, zero slack — unchanged from the
// frozen unit), 26 at 3584, 28 at 8192 (the all-(−128) row lands EXACTLY on
// 2^27 — legal, mirrors the golden accumulator bound). mean2 <= 2^14
// INDEPENDENT of D (mean of values <= 2^14; the μ product is shifted back
// into range, asserted per entry by the generator), so rsqrt_unit's 32-bit
// norm2 input takes mean2 + 1 <= 2^14 + 1 comfortably in every elaboration.
//
// ── rsqrt WRAPPER (D-018) ─────────────────────────────────────────────────
// in_valid pulses for EXACTLY one cycle (ST_ISSUE); the FSM then counts
// cycles and captures inv_norm/norm_int only when BOTH out_valid is high AND
// the counter equals apex_pkg::RSQRT_LATENCY (31: out_valid is registered
// 31 edges after the accepting edge). If the vendored unit ever deviated
// from D-018 the FSM would hang and the TB watchdog would $fatal — an honest
// failure, never a silent mis-sample. One row issues exactly one rsqrt op,
// so issue is trivially serialized.
//
// ── STREAMS & JOB SEMANTICS ───────────────────────────────────────────────
// Three §5 valid/ready streams, each behind a stream_skid (V0.5/D-019):
//   x  in  : {x[7:0], last} — `last` frames the row (defines D).
//   g  in  : gamma payload only. gamma is consumed DURING EMISSION, exactly
//            D beats per row; framing is carried by the x stream (`last` on
//            lane beats is informational per apex_pkg §5 comment).
//   y  out : {y[15:0], last}.
// D-006: busy = (FSM != IDLE), FSM holds ST_DRAIN until all D output beats
// are accepted POST-SKID; done = 1-cycle pulse after that (busy low at done).
// Legality reject (§3 style): D > RMS_D_MAX or D otherwise illegal (not a
// pow-2 when <= 128; not a multiple of 128 when wide) — len_error pulse +
// len_error_sticky, row consumed through `last`, no output, no done, no
// state change. The reject OBSERVABLE is identical in both elaborations.
//
// LATENCY: D (collect) + 1 (issue) + 32 (rsqrt wait incl. capture) +
// D (emit, 1/cycle when gamma present and no backpressure) + 2 (drain/done).
// dbg_norm exposes the captured norm_int (floor sqrt) for TB cross-checking.

module asu_rmsnorm
  import apex_pkg::*;
#(
  parameter int unsigned RMS_D_MAX = 128
) (
  input  logic               clk,
  input  logic               rst_n,     // synchronous, active low

  // x input stream: INT8 + last
  input  logic               s_valid,
  output logic               s_ready,
  input  logic signed [7:0]  s_x,
  input  logic               s_last,

  // gamma input stream: Q2.13, one beat per element, consumed at emission
  input  logic               g_valid,
  output logic               g_ready,
  input  logic signed [15:0] g_gamma,

  // y output stream: Q7.8 + last
  output logic               m_valid,
  input  logic               m_ready,
  output logic signed [15:0] m_y,
  output logic               m_last,

  // job / status (D-006, §3)
  output logic               busy,
  output logic               done,
  output logic               len_error,
  output logic               len_error_sticky,

  // captured rsqrt_unit norm_int (floor(sqrt(norm2))), for verification
  output logic        [13:0] dbg_norm,

  // C-RMSW chunk composition (R4, header): both levels quasi-static, 0 =
  // frozen behavior. ext_sum2/ext_k are validated at the apex_top arm.
  input  logic               ext_sum_en, // pass 1: rows are sum-capture only
  input  logic               ext_r_en,   // pass 2: rsqrt operand is external
  input  logic        [27:0] ext_sum2,   // accumulated wide-row sum2
  input  logic        [6:0]  ext_k,      // D_wide / 128, in [2, 64]
  output logic               s2_push,    // 1-cycle: s2_val captured (pass 1)
  output logic        [27:0] s2_val      // the completed row's own sum2
);

  generate
    if (RMS_D_MAX < 1 || RMS_D_MAX > 8192) begin : g_cfg_err_d
      // beyond 8192 the golden accumulator bound (sum2 <= 2^27) breaks
      $error("asu_rmsnorm: RMS_D_MAX outside [1,8192]");
    end
  endgenerate

  // width proofs in header; at RMS_D_MAX=128 these elaborate to the frozen
  // unit's exact values (SUM_W=22, DCNT_W=8)
  localparam int unsigned SUM_W  = $clog2(32'(RMS_D_MAX) * 32'd16384 + 32'd1);
  localparam int unsigned DCNT_W = $clog2(RMS_D_MAX + 1);   // counts 0..RMS_D_MAX
  localparam int unsigned IDX_W  = (RMS_D_MAX > 1) ? $clog2(RMS_D_MAX) : 1;

  // ── input skids ───────────────────────────────────────────────────────────
  logic              ix_valid, ix_ready;
  logic        [8:0] ix_bus;
  logic signed [7:0] ix_x;
  logic              ix_last;

  stream_skid #(.WIDTH(9)) u_x_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (s_valid),
    .s_ready (s_ready),
    .s_data  ({s_x, s_last}),
    .m_valid (ix_valid),
    .m_ready (ix_ready),
    .m_data  (ix_bus)
  );
  assign ix_x    = signed'(ix_bus[8:1]);
  assign ix_last = ix_bus[0];

  logic               ig_valid, ig_ready;
  logic        [15:0] ig_bus;
  logic signed [15:0] ig_gamma;

  stream_skid #(.WIDTH(16)) u_g_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (g_valid),
    .s_ready (g_ready),
    .s_data  (g_gamma),
    .m_valid (ig_valid),
    .m_ready (ig_ready),
    .m_data  (ig_bus)
  );
  assign ig_gamma = signed'(ig_bus);

  // ── state ─────────────────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    ST_IDLE, ST_COLLECT, ST_FLUSH, ST_ISSUE, ST_WAIT, ST_EMIT, ST_DRAIN
  } st_e;
  st_e st;

  logic [DCNT_W-1:0]  cnt;         // x beats accepted this row
  logic [DCNT_W-1:0]  n_len;       // final row length D
  logic [DCNT_W-1:0]  emit_idx;
  logic [DCNT_W-1:0]  out_cnt;     // POST-SKID accepted output beats
  logic [SUM_W-1:0]   sum2;        // sum of x^2 (width proof in header)
  logic [14:0]        r_q;         // captured inv_norm, UQ2.13
  logic [5:0]         rs_cnt;      // D-018 cycle counter
  logic signed [7:0]  x_buf [RMS_D_MAX];  // row buffer. Wide elab maps to
                                          // BRAM as-is (yosys absorbs the
                                          // registered read address; probe in
                                          // verif/asu/wide/RESULT.md)

  // ── collect datapath ──────────────────────────────────────────────────────
  logic signed [15:0] xx;                       // x^2: 0..16384, fits s16
  logic [DCNT_W-1:0]  n_new;
  assign xx    = 16'(ix_x) * 16'(ix_x);
  assign n_new = cnt + DCNT_W'(1);

  // log2 of the (one-hot) row length — exact mean shift for pow-2 D <= 128
  // (only consulted for such rows, so bits above 7 never reach the cast)
  function automatic logic [2:0] f_log2 (input logic [DCNT_W-1:0] v);
    f_log2 = 3'd0;
    for (int i = 0; i < DCNT_W; i++) begin
      if (v[i]) f_log2 = 3'(i);
    end
  endfunction

  // ── length legality + mean path (narrow = frozen shift; wide = C-RMSW μ) ──
  logic        len_ok;      // n_new is a legal row length (checked at `last`)
  logic [31:0] mean2p1;     // mean(x^2) + EPS_INT -> rsqrt operand

  generate
    if (RMS_D_MAX <= 128) begin : g_mean_narrow
      assign len_ok  = $onehot(n_new);
      assign mean2p1 = (32'(sum2) >> f_log2(n_len)) + 32'd1;     // mean2 + 1
    end else begin : g_mean_wide
      // C-RMSW: mean2 = (sum2 · MU[k]) >> (S[k] + 16), k = D/128 — golden
      // arbiter rmsnorm_fx_wide/_wide_mu; constants provenance-gated (header)
      `include "asu_wide_rms_params.svh"
      if (RMS_D_MAX % WIDE_RMS_CHUNK != 0
          || RMS_D_MAX > WIDE_RMS_CHUNK * WIDE_RMS_K_MAX) begin : g_cfg_err_w
        $error("asu_rmsnorm: wide RMS_D_MAX must be a multiple of 128 <= 8192");
      end

      localparam int unsigned PROD_W = SUM_W + 17;   // sum2 · MU, exact
      localparam int unsigned CHK_L2 = $clog2(WIDE_RMS_CHUNK);   // 7

      logic              use_mu;    // wide row (D > 128): μ path
      logic [6:0]        k_idx;     // D/128 in [1, 64]
      logic [16:0]       mu_v;
      logic [3:0]        s_v;
      logic [PROD_W-1:0] prod;
      logic [4:0]        sh_v;      // S + 16 in [23, 29]
      logic [31:0]       mean2_mu;

      // legal D: pow-2 <= 128 (frozen envelope, exact-shift path) OR any
      // multiple of 128 (μ path); <= RMS_D_MAX enforced by the COLLECT
      // overflow reject before `last` can arrive
      assign len_ok   = ($onehot(n_new) && (n_new <= DCNT_W'(WIDE_RMS_CHUNK)))
                        || ((n_new & DCNT_W'(WIDE_RMS_CHUNK - 1)) == '0);
      assign use_mu   = (n_len > DCNT_W'(WIDE_RMS_CHUNK));
      assign k_idx    = 7'(n_len >> CHK_L2);
      assign mu_v     = WIDE_RMS_MU[k_idx];
      assign s_v      = WIDE_RMS_S[k_idx];
      assign prod     = PROD_W'(sum2) * PROD_W'(mu_v);
      assign sh_v     = 5'(s_v) + 5'd16;
      assign mean2_mu = 32'(prod >> sh_v);           // <= 2^14 (header proof)
      assign mean2p1  = (use_mu ? mean2_mu
                                : (32'(sum2) >> f_log2(n_len))) + 32'd1;
    end
  endgenerate

  // ── C-RMSW ext operand (R4): the wide μ path on an EXTERNAL sum2 ─────────
  // Same generated table and same expression as g_mean_wide (the file is
  // included once per generate scope; scopes are independent). The arm point
  // (apex_top) enforces k in [2,64] and ext_sum2 <= k·2^21, so mean2 <= 2^14
  // (per-entry generator assertion) and the rsqrt operand range is unchanged.
  logic [31:0] ext_mean2;
  generate if (1) begin : g_ext_crmsw
    `include "asu_wide_rms_params.svh"
    // Drift guard: the 0x94 arm bound (apex_top R4: k in [2,64], sum2 <=
    // k·2^21) and the emitter mirror are written against THIS envelope; a
    // regenerated table with a different shape must re-audit both.
    if (WIDE_RMS_K_MAX != 64 || WIDE_RMS_CHUNK != 128) begin : g_cfg_err_ext
      $error("asu_rmsnorm R4: μ-table envelope changed — re-audit the arm");
    end
    localparam int unsigned XPROD_W = 28 + 17;      // ext_sum2 · MU, exact
    logic [XPROD_W-1:0] xprod;
    logic [4:0]         xsh;                        // S + 16 in [23, 29]
    assign xprod     = XPROD_W'(ext_sum2) * XPROD_W'(WIDE_RMS_MU[ext_k]);
    assign xsh       = 5'(WIDE_RMS_S[ext_k]) + 5'd16;
    assign ext_mean2 = 32'(xprod >> xsh);
  end endgenerate

  // ── vendored rsqrt_unit (D-018 wrapper: 1-cycle issue, counted capture) ──
  logic        rs_in_valid;
  logic [31:0] rs_norm2;
  logic        rs_out_valid;
  logic [14:0] rs_inv_norm;
  logic [13:0] rs_norm_int;

  assign rs_in_valid = (st == ST_ISSUE);
  // R4: ext_r_en swaps ONLY the rsqrt operand (mean2+EPS_INT); issue/wait/
  // capture and the whole emission datapath are the frozen unit verbatim.
  assign rs_norm2    = ext_r_en ? (ext_mean2 + 32'd1) : mean2p1;

  rsqrt_unit u_rsqrt (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_valid  (rs_in_valid),
    .norm2     (rs_norm2),
    .out_valid (rs_out_valid),
    .inv_norm  (rs_inv_norm),
    .norm_int  (rs_norm_int)
  );

  // ── emission datapath (golden mirror, per-element) ────────────────────────
  logic signed [23:0] p_xg;     // x * gamma, Q2.13, |p| <= 2^22
  logic signed [39:0] q_full;   // p * r, frac 26, |q| < 2^36
  logic signed [21:0] y_tr;     // floor(q >> 18), Q7.8
  logic        [17:0] rem_v;
  logic               rnd_up;
  logic signed [22:0] y_rne;
  logic signed [15:0] y_sat;

  assign p_xg   = 24'(x_buf[emit_idx[IDX_W-1:0]]) * 24'(ig_gamma);
  assign q_full = 40'(p_xg) * 40'(signed'({1'b0, r_q}));
  assign y_tr   = 22'(q_full >>> 18);
  assign rem_v  = q_full[17:0];
  assign rnd_up = (rem_v > 18'h20000) || ((rem_v == 18'h20000) && y_tr[0]);
  assign y_rne  = 23'(y_tr) + 23'(rnd_up);
  assign y_sat  = (y_rne > 23'sd32767)  ? 16'sd32767  :
                  (y_rne < -23'sd32768) ? -16'sd32768 : 16'(y_rne);

  // ── output skid / handshakes ──────────────────────────────────────────────
  logic o_valid, o_ready, o_last;
  logic in_xfer, m_xfer;

  assign o_last   = (emit_idx == n_len - DCNT_W'(1));
  assign o_valid  = (st == ST_EMIT) && ig_valid;   // gamma gates emission
  assign ig_ready = (st == ST_EMIT) && o_ready;
  assign ix_ready = (st == ST_IDLE) || (st == ST_COLLECT) || (st == ST_FLUSH);
  assign in_xfer  = ix_valid && ix_ready;
  assign m_xfer   = m_valid && m_ready;

  stream_skid #(.WIDTH(17)) u_out_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (o_valid),
    .s_ready (o_ready),
    .s_data  ({y_sat, o_last}),
    .m_valid (m_valid),
    .m_ready (m_ready),
    .m_data  ({m_y, m_last})
  );

  assign busy = (st != ST_IDLE);

  // ── FSM ───────────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st               <= ST_IDLE;
      cnt              <= '0;
      n_len            <= '0;
      emit_idx         <= '0;
      out_cnt          <= '0;
      sum2             <= '0;
      r_q              <= '0;
      rs_cnt           <= '0;
      dbg_norm         <= '0;
      done             <= 1'b0;
      len_error        <= 1'b0;
      len_error_sticky <= 1'b0;
      s2_push          <= 1'b0;
      s2_val           <= '0;
    end else begin
      done      <= 1'b0;
      len_error <= 1'b0;
      s2_push   <= 1'b0;
      if (m_xfer) out_cnt <= out_cnt + DCNT_W'(1);

      case (st)
        ST_IDLE: begin
          if (in_xfer) begin              // first x beat of a new row
            x_buf[0] <= ix_x;
            sum2     <= SUM_W'(unsigned'(xx));
            cnt      <= DCNT_W'(1);
            out_cnt  <= '0;
            if (ix_last) begin            // D = 1 (power of two: legal)
              if (ext_sum_en) begin       // R4 pass 1: capture, no emission
                s2_val  <= 28'(unsigned'(xx));
                s2_push <= 1'b1;
              end else begin
                n_len <= DCNT_W'(1);
                st    <= ST_ISSUE;
              end
            end else begin
              st <= ST_COLLECT;
            end
          end
        end

        ST_COLLECT: begin
          if (in_xfer) begin
            if (cnt == DCNT_W'(RMS_D_MAX)) begin
              // beat RMS_D_MAX+1: row too long — legality reject
              len_error        <= 1'b1;
              len_error_sticky <= 1'b1;
              st               <= ix_last ? ST_IDLE : ST_FLUSH;
            end else begin
              x_buf[cnt[IDX_W-1:0]] <= ix_x;
              sum2       <= sum2 + SUM_W'(unsigned'(xx));
              cnt        <= n_new;
              if (ix_last) begin
                if (!len_ok) begin           // illegal row length — reject
                  len_error        <= 1'b1;
                  len_error_sticky <= 1'b1;
                  st               <= ST_IDLE;
                end else if (ext_sum_en) begin
                  // R4 pass 1: export this row's sum2 (the last beat's
                  // square is folded here — the sum2 register only takes it
                  // on this same edge), then back to IDLE: no rsqrt, no
                  // emission, no gamma. Rejected rows never capture.
                  s2_val  <= 28'(sum2) + 28'(unsigned'(xx));
                  s2_push <= 1'b1;
                  st      <= ST_IDLE;
                end else begin
                  n_len <= n_new;
                  st    <= ST_ISSUE;
                end
              end
            end
          end
        end

        ST_FLUSH: begin                   // consume rejected row, no effects
          if (in_xfer && ix_last) st <= ST_IDLE;
        end

        ST_ISSUE: begin                   // exactly one in_valid cycle
          rs_cnt <= '0;
          st     <= ST_WAIT;
        end

        ST_WAIT: begin                    // D-018: count cycles, then capture
          rs_cnt <= rs_cnt + 6'd1;
          if (rs_out_valid && (rs_cnt == 6'(RSQRT_LATENCY))) begin
            r_q      <= rs_inv_norm;
            dbg_norm <= rs_norm_int;
            emit_idx <= '0;
            st       <= ST_EMIT;
          end
        end

        ST_EMIT: begin                    // one element per gamma beat
          if (o_valid && o_ready) begin
            if (o_last) st <= ST_DRAIN;
            else        emit_idx <= emit_idx + DCNT_W'(1);
          end
        end

        ST_DRAIN: begin                   // D-006: wait for post-skid accepts
          if (out_cnt == n_len) begin
            done <= 1'b1;
            st   <= ST_IDLE;
          end
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule
