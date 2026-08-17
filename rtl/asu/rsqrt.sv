// rsqrt.sv — APEX clean-room reciprocal-sqrt unit (module rsqrt_unit).
//
// CLEAN-ROOM PROVENANCE: this is an original implementation derived solely
// from (a) APEX's own golden contract — the normative fixed-point pseudocode
// in rtl/asu/asu_rmsnorm.sv's header and its golden mirror rmsnorm_fx in
// verif/asu/smoke/gen_asu_vectors.py, and (b) apex_golden.compute. It replaces
// the removed vendored rsqrt_unit. No prior/third-party RTL was consulted;
// the internal micro-architecture is my own. Only the PORT LIST and the
// NUMERIC BEHAVIOR (bit-exact vs the golden) are contractually fixed.
//
// ── FUNCTION (asu_rmsnorm.sv normative contract, per-op) ───────────────────
//   Given norm2 (already = mean(x^2)+eps in the wrapper; here treated as an
//   arbitrary 32-bit unsigned):
//       den      = isqrt(norm2)                  // floor integer sqrt
//       inv_norm = (1<<13) / den   (den != 0)    // UQ2.13, floor division
//       norm_int = min(den, 2^14 - 1)            // floor-sqrt, saturated
//   Out-of-contract input norm2 == 0 (den == 0) is handled deterministically:
//   inv_norm = 0, norm_int = 0 (no divide-by-zero). The wrapper guarantees
//   norm2 >= 1, so this case never arises in the RMSNorm datapath; it exists
//   only to keep the unit total for the exhaustive verification sweep.
//
// ── ALGORITHM (multiplier-free; two sequential digit-recurrences) ──────────
//   Phase A — isqrt (16 cycles): classic digit-by-digit (bit-set) integer
//     square root. Per iteration it = 15..0 with d = 1 << (2*it):
//         if (xr >= root + d) { xr -= root + d;  root = (root>>1) + d; }
//         else                {                  root =  root>>1;      }
//     After 16 iterations `root` == floor(sqrt(norm2)) exactly. Each step is
//     an add/compare/subtract and shifts — no multiplier. (Proven exact vs
//     math.isqrt across the whole 32-bit domain in the bring-up model.)
//   Phase B — reciprocal (14 cycles): restoring long division of the fixed
//     numerator N = 1<<13 = 8192 by `den`, MSB-first, 14 quotient bits
//     (i = 13..0). N has a single set bit (bit 13), so the dividend-bit feed
//     is just (i == 13). Yields floor(8192/den) with no multiplier.
//
// ── LATENCY (fixed, matches apex_pkg::RSQRT_LATENCY = 31) ───────────────────
//   in_valid is sampled for one cycle; internally cyc counts 0,1,2,... from
//   that accepting edge in exact lockstep with the D-018 wrapper's rs_cnt
//   (both are posedge FFs sampling the same one-cycle in_valid). Schedule:
//       cyc  0..15 : Phase A (16 sqrt iterations)
//       cyc 16..29 : Phase B (14 division iterations); outputs latched at the
//                    cyc==29 edge (available from cyc==30, stable thereafter)
//       cyc    30  : hold (results stable)
//       cyc    31  : out_valid asserted for exactly ONE cycle
//   asu_rmsnorm captures inv_norm/norm_int at the edge ending the cyc==31
//   cycle (its check `rs_out_valid && rs_cnt == RSQRT_LATENCY`). Total work is
//   30 cycles with 1 slack cycle, so results are settled well before capture.
//   Issue is fully serialized (one op per RMSNorm row); a fresh in_valid in
//   IDLE restarts cyc, giving clean back-to-back operation. in_valid during a
//   busy op is ignored (the wrapper never does this).
//
// ── WIDTH PROOFS (all registers non-overflowing; established by the bring-up
//    model rsqrt_model.py, exhaustive + 2M random) ───────────────────────────
//   root/xr/d/cpd : 32-bit. Max intermediate root = 2^30, max (root+d) =
//                   1342177280 < 2^31 — both fit 32-bit with headroom.
//   den           : 16-bit. isqrt(2^32-1) = 65535 = 16 bits exactly.
//   drem          : 16-bit. Restoring remainder never exceeds 8192<<1 = 16384.
//   quo/inv_norm  : quo 14-bit (max 8192 at den==1); inv_norm 15-bit (0-ext).
//
// House style: SYNCHRONOUS active-low reset. -Wall clean with ZERO functional
// waivers. The one exception below is purely cosmetic: the spec mandates the
// file be rtl/asu/rsqrt.sv while the module must be `rsqrt_unit` to match the
// kept instantiation in asu_rmsnorm.sv (`rsqrt_unit u_rsqrt`). That filename/
// module split makes Verilator's DECLFILENAME note unavoidable; it is a naming
// note only — no width, logic, or reset lint is suppressed anywhere.

/* verilator lint_off DECLFILENAME */
module rsqrt_unit (
  input  logic        clk,
  input  logic        rst_n,       // synchronous, active low

  input  logic        in_valid,    // 1-cycle issue pulse (no busy/ready)
  input  logic [31:0] norm2,       // operand (sampled at the accepting edge)

  output logic        out_valid,   // 1-cycle strobe, fixed latency (cyc==31)
  output logic [14:0] inv_norm,    // UQ2.13 = floor(8192 / isqrt(norm2))
  output logic [13:0] norm_int     // min(isqrt(norm2), 2^14 - 1)
);

  // ── sequencing state ──────────────────────────────────────────────────────
  typedef enum logic [1:0] { P_IDLE, P_SQRT, P_DIV, P_HOLD } phase_e;
  phase_e      phase;

  logic [5:0]  cyc;    // cycle counter from the accepting edge (lockstep w/ D-018)
  logic [3:0]  it;     // iteration index: sqrt 15..0, div 13..0

  // ── isqrt (Phase A) datapath ──────────────────────────────────────────────
  logic [31:0] root;   // 'c' accumulator -> floor(sqrt)
  logic [31:0] xr;     // running remainder

  logic [4:0]  sq_sh;      // 2*it (0..30)
  logic [31:0] sq_d;       // 1 << (2*it)
  logic [31:0] sq_cpd;     // root + d
  logic        sq_ge;      // xr >= root + d
  logic [31:0] sq_root_n;  // next root
  logic [31:0] sq_xr_n;    // next remainder

  assign sq_sh     = {it, 1'b0};
  assign sq_d      = 32'd1 << sq_sh;
  assign sq_cpd    = root + sq_d;
  assign sq_ge     = (xr >= sq_cpd);
  assign sq_root_n = sq_ge ? ((root >> 1) + sq_d) : (root >> 1);
  assign sq_xr_n   = sq_ge ? (xr - sq_cpd) : xr;

  // ── reciprocal (Phase B) datapath: restoring long division 8192 / den ─────
  logic [15:0] den;    // isqrt result captured at end of Phase A
  logic [15:0] drem;   // partial remainder
  logic [13:0] quo;    // quotient accumulator

  logic        dv_bit;      // dividend bit of N=8192 for this step (bit 13 only)
  logic [15:0] dv_drem_sh;  // (drem << 1) | dv_bit
  logic        dv_ge;       // drem_sh >= den
  logic [15:0] dv_drem_n;   // next remainder
  logic [13:0] dv_quo_n;    // next quotient

  assign dv_bit     = (it == 4'd13);
  assign dv_drem_sh = (drem << 1) | {15'd0, dv_bit};
  assign dv_ge      = (dv_drem_sh >= den);
  assign dv_drem_n  = dv_ge ? (dv_drem_sh - den) : dv_drem_sh;
  assign dv_quo_n   = dv_ge ? (quo | (14'd1 << it)) : quo;

  // ── control / output registers ────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      phase     <= P_IDLE;
      cyc       <= 6'd0;
      it        <= 4'd0;
      root      <= 32'd0;
      xr        <= 32'd0;
      den       <= 16'd0;
      drem      <= 16'd0;
      quo       <= 14'd0;
      out_valid <= 1'b0;
      inv_norm  <= 15'd0;
      norm_int  <= 14'd0;
    end else begin
      case (phase)
        // ── idle: accept a new operand on in_valid ──────────────────────────
        P_IDLE: begin
          out_valid <= 1'b0;
          if (in_valid) begin
            cyc   <= 6'd0;
            it    <= 4'd15;
            root  <= 32'd0;
            xr    <= norm2;      // operand sampled once, here
            den   <= 16'd0;
            drem  <= 16'd0;
            quo   <= 14'd0;
            phase <= P_SQRT;
          end
        end

        // ── Phase A: 16 sqrt iterations (it = 15..0) ─────────────────────────
        P_SQRT: begin
          cyc  <= cyc + 6'd1;
          root <= sq_root_n;
          xr   <= sq_xr_n;
          if (it == 4'd0) begin
            den   <= sq_root_n[15:0];   // isqrt result
            it    <= 4'd13;
            phase <= P_DIV;
          end else begin
            it <= it - 4'd1;
          end
        end

        // ── Phase B: 14 division iterations (it = 13..0) ─────────────────────
        P_DIV: begin
          cyc  <= cyc + 6'd1;
          drem <= dv_drem_n;
          quo  <= dv_quo_n;
          if (it == 4'd0) begin
            if (den == 16'd0) begin
              inv_norm <= 15'd0;
              norm_int <= 14'd0;
            end else begin
              inv_norm <= {1'b0, dv_quo_n};
              norm_int <= (den > 16'd16383) ? 14'h3FFF : den[13:0];
            end
            phase <= P_HOLD;
          end else begin
            it <= it - 4'd1;
          end
        end

        // ── hold to the fixed latency, then strobe out_valid for one cycle ──
        P_HOLD: begin
          cyc <= cyc + 6'd1;
          if (cyc == 6'd30) out_valid <= 1'b1;
          if (cyc == 6'd31) begin
            out_valid <= 1'b0;
            phase     <= P_IDLE;
          end
        end

        default: phase <= P_IDLE;
      endcase
    end
  end

endmodule
/* verilator lint_on DECLFILENAME */
