// apex_residual.sv — the C-6 residual unit (IB-LAYER S4; re-scoped from S2
// where its numeric core residual_add_fx was retired standalone).
//
//   row RAM  : [DM_MAX] fp16 — the resident layer row (X at layer entry,
//              r1 after the first residual job, r2 == the layer output
//              after the second). Host-loaded via the LAYER window
//              (lw_* port), host-read via rd_* (LAYER_RPTR/RDATA).
//   job      : cols beats of EXACT fp32 sublayer values from apex_layer_deq
//              (o8*graded composites — the D-030 lemma operands);
//              per element i:  row[i] <= f16( row[i] + b[i] )  — ONE RNE,
//              in-place update (r1 = f16(X+attn_proj), then r2 = f16(r1+
//              ffn_out) — transformer.py:464,:484).
//   base     : R3 (2026-07-30, lifts B-RES-BASE) — the job carries a 2-bit
//              WINDOW base (LAYER_JOB[15:14], stride 1024): stream element i
//              updates row[base*1024 + i]. A footprint past the RAM
//              (base*1024 + cols > DM_MAX) is REFUSED at accept — the
//              illegal-geometry frame_error class below — never wrapped.
//              With the 2040-element serializer frame (B-SER-FRAME), a
//              DM=3584 row is covered by 4 aligned 1024-windows.
//   egress   : E-1 (2026-07-31, E2E_TOY_LANE.md §4) — the row's INTERNAL
//              exit. Before this the ONLY reader was the host register path
//              (rd_*), so the running activation could not reach any
//              consumer inside the tile. ej_* is a second job port (same
//              R3 window-base addressing: base + cols, geometry refused at
//              accept with the SAME frame_error class, never wrapped) that
//              streams cols fp16 beats row[base*1024 + i] on ev/er/edata,
//              elast on the final beat, through a §5 skid. Exclusivity: an
//              egress job is accepted only while the add job is idle and
//              vice versa (one row RAM, no read-during-update ambiguity);
//              a held-valid push simply waits — never a silent overlap.
//              ebusy covers the FSM AND the output skid (post-skid, D-006).
//
// NUMERICS (frozen in IB_LAYER.md §3b, measured before freezing: worst
// window margin +19 bits over all 1,536 BUS_ON L4 sublayer operands,
// 0 refusals):
//   * the add runs EXACTLY on a 2^-38 grid in <=57 bits and narrows once
//     (f16_arith_pkg::f16_pack_real, exp=-38);
//   * b with value >= 2^17 (e8 >= 144) short-circuits to +-inf — provably
//     the RNE result for any fp16 row operand;
//   * b whose grid is finer than 2^-38 after trailing-zero normalization
//     REFUSES loudly (RESID_WINDOW: window_error pulse + sticky, job
//     aborts, remaining beats drained) — exact-or-refused, never silent
//     rounding. Zero-sign rule: exact-zero sum is -0 iff both operands -0.
//   * frame: ilast exactly at cols-1; early last drops the job, missing
//     last resyncs (the deq/rope_row shape). An aborted job leaves the
//     row PARTIALLY updated — the sticky is the host's truth; reload the
//     row before reuse.

module apex_residual
  import f16_arith_pkg::*;
#(
  parameter int unsigned DM_MAX = 128,
  localparam int unsigned AW = $clog2(DM_MAX),
  localparam int unsigned CW = 12
)(
  input  logic          clk,
  input  logic          rst_n,

  // LAYER window load / readback ports (glue-owned addressing)
  input  logic          lw_en,
  input  logic [AW-1:0] lw_addr,
  input  logic [15:0]   lw_data,
  input  logic [AW-1:0] rd_addr,
  output logic [15:0]   rd_data,

  // job + exact-fp32 sublayer stream (from apex_layer_deq)
  input  logic          jb_valid,
  output logic          jb_ready,
  input  logic [CW-1:0] jb_cols,
  input  logic [1:0]    jb_base,
  input  logic          iv,
  output logic          ir,
  input  logic [31:0]   idata,
  input  logic          ilast,

  // E-1 egress job + fp16 stream out (header): base + cols -> cols beats
  input  logic          ej_valid,
  output logic          ej_ready,
  input  logic [CW-1:0] ej_cols,
  input  logic [1:0]    ej_base,
  output logic          ev,
  input  logic          er,
  output logic [15:0]   edata,
  output logic          elast,
  output logic          ebusy,

  output logic          busy,
  output logic          done,
  output logic          frame_error,
  output logic          frame_error_sticky,
  output logic          window_error,
  output logic          window_error_sticky
);

  // ── input skid (§5) ───────────────────────────────────────────────────────
  logic        s_iv, s_ir, s_ilast;
  logic [31:0] s_idata;
  stream_skid #(.WIDTH(33)) u_in_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (iv), .s_ready (ir), .s_data ({ilast, idata}),
    .m_valid (s_iv), .m_ready (s_ir), .m_data ({s_ilast, s_idata})
  );

  // ── row RAM ───────────────────────────────────────────────────────────────
  logic [15:0] row [DM_MAX];
  assign rd_data = row[rd_addr];

  typedef enum logic [1:0] {ST_IDLE, ST_RUN, ST_DRAIN} st_e;
  st_e           st;
  logic [CW-1:0] cols_q, cnt;
  logic [1:0]    base_q;
  logic          err_fr, err_wn, stk_fr, stk_wn;

  // ── E-1 egress FSM + skid (header) ────────────────────────────────────────
  typedef enum logic {E_IDLE, E_RUN} est_e;
  est_e          est;
  logic [CW-1:0] e_cols_q, e_cnt;
  logic [1:0]    e_base_q;

  // R3 addressing reused verbatim: the accept-time refusal bounds
  // base*1024 + cols <= DM_MAX, so this add never exceeds DM_MAX-1.
  wire [CW-1:0] e_idx   = {e_base_q, 10'b0} + e_cnt;
  wire          eo_valid = (est == E_RUN);
  wire          eo_last  = (e_cnt == e_cols_q - CW'(1));
  logic         eo_ready;

  stream_skid #(.WIDTH(17)) u_e_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (eo_valid), .s_ready (eo_ready),
    .s_data  ({eo_last, row[e_idx[AW-1:0]]}),
    .m_valid (ev), .m_ready (er), .m_data ({elast, edata})
  );

  // exclusivity (header): all egress ROW READS are complete by E_IDLE (the
  // skid holds copies), so the add job needs only the FSM term; the egress
  // job additionally waits out a streaming add so it reads the POST-add row.
  assign ej_ready = (st == ST_IDLE) && (est == E_IDLE);
  assign ebusy    = (est != E_IDLE) || ev;

  // R3: effective row index = window base + beat count. The accept-time
  // refusal bounds base*1024 + cols <= DM_MAX, so this add never exceeds
  // DM_MAX-1 and the AW truncation below stays exact.
  wire [CW-1:0] eff_idx = {base_q, 10'b0} + cnt;

  // Config-dependent fold: at DM_MAX < 2^CW the window indices' high bits
  // are structurally unused (the accept-time refusals bound them below
  // DM_MAX). Consumed here so EVERY DM_MAX elaboration is -Wall clean
  // without waivers. (Found as a pre-existing R3 lint red at DM_MAX=128 —
  // eff_idx[11:7] — when E-1 relinted the narrow config.)
  logic unused_idx_ok;
  assign unused_idx_ok = &{1'b0, e_idx, eff_idx};

  // ── per-beat exact add (combinational on the accepted beat) ──────────────
  logic        b_sign, b_zero, b_inf_sc, b_fit;
  logic [7:0]  b_e8;
  logic [23:0] b_sig;
  int          sh_b;
  logic signed [F16_V24_W-1:0] a_v24;
  logic [F16_V24_W-1:0]        a_mag;
  logic [56:0] b_fix, a_fix;
  logic signed [57:0] acc;
  logic        a_sign, y_sign;
  logic [15:0] a_bits, y_bits;
  logic [56:0] mag;

  always_comb begin
    a_bits  = row[eff_idx[AW-1:0]];
    a_sign  = a_bits[15];
    b_sign  = s_idata[31];
    b_e8    = s_idata[30:23];
    b_sig   = {1'b1, s_idata[22:0]};
    b_zero  = (s_idata[30:0] == '0);
    b_inf_sc = (b_e8 >= 8'd144);              // |b| >= 2^17 -> sum == +-inf
    sh_b    = int'({24'b0, b_e8}) - 112;      // align 2^(e8-150) to 2^-38
    b_fit   = 1'b1;
    b_fix   = '0;
    if (!b_zero && !b_inf_sc) begin
      if (sh_b >= 0) begin
        b_fix = {33'b0, b_sig} << sh_b;       // sh_b <= 31 given e8 < 144
      end else if ((-sh_b) < 24
                   && ((b_sig & ((24'd1 << (-sh_b)) - 24'd1)) == '0)) begin
        b_fix = {33'b0, b_sig} >> (-sh_b);    // exact right shift
      end else begin
        b_fit = 1'b0;                         // grid finer than 2^-38: REFUSE
      end
    end
    a_v24 = f16_to_v24(a_bits);
    a_mag = a_sign ? F16_V24_W'(-a_v24) : F16_V24_W'(a_v24);
    a_fix = {{57 - F16_V24_W{1'b0}}, a_mag} << 14;   // fp16 onto the 2^-38 grid
    acc = (a_sign ? -$signed({1'b0, a_fix}) : $signed({1'b0, a_fix}))
        + (b_zero ? 58'sd0
                  : (b_sign ? -$signed({1'b0, b_fix})
                            : $signed({1'b0, b_fix})));
    if (acc == 0)
      y_sign = a_sign && b_sign;              // -0 only from (-0)+(-0)
    else
      y_sign = acc < 0;
    mag = (acc < 0) ? 57'(-acc) : 57'(acc);
    if (b_inf_sc)
      y_bits = {b_sign, 5'h1F, 10'd0};        // shortcut: always the RNE inf
    else
      y_bits = f16_pack_real(y_sign, mag, -38);
  end

  wire frame_ok = (s_ilast == (cnt == cols_q - CW'(1)));

  assign jb_ready = (st == ST_IDLE) && (est == E_IDLE);
  assign s_ir     = (st == ST_RUN) || (st == ST_DRAIN);
  assign busy     = (st != ST_IDLE);
  assign frame_error         = err_fr;
  assign window_error        = err_wn;
  assign frame_error_sticky  = stk_fr;
  assign window_error_sticky = stk_wn;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st       <= ST_IDLE;
      cols_q   <= '0;
      cnt      <= '0;
      base_q   <= '0;
      est      <= E_IDLE;
      e_cols_q <= '0;
      e_cnt    <= '0;
      e_base_q <= '0;
      err_fr <= 1'b0; err_wn <= 1'b0;
      stk_fr <= 1'b0; stk_wn <= 1'b0;
      done   <= 1'b0;
    end else begin
      err_fr <= 1'b0; err_wn <= 1'b0; done <= 1'b0;
      if (lw_en) row[lw_addr] <= lw_data;
      // ── E-1 egress (header): accept-time geometry refusal is the SAME
      // frame_error class as the add job's — never wrapped, never silent.
      unique case (est)
        E_IDLE: begin
          if (ej_valid && ej_ready) begin
            if ((ej_cols == '0)
                || (({20'b0, ej_cols} + {20'b0, ej_base, 10'b0})
                    > 32'(DM_MAX))) begin
              err_fr <= 1'b1;               // illegal geometry: refuse
              stk_fr <= 1'b1;
            end else begin
              e_cols_q <= ej_cols;
              e_base_q <= ej_base;
              e_cnt    <= '0;
              est      <= E_RUN;
            end
          end
        end
        E_RUN: begin
          if (eo_valid && eo_ready) begin
            if (eo_last) est   <= E_IDLE;
            else         e_cnt <= e_cnt + CW'(1);
          end
        end
        default: est <= E_IDLE;
      endcase
      unique case (st)
        ST_IDLE: begin
          if (jb_valid && jb_ready) begin
            if ((jb_cols == '0)
                || (({20'b0, jb_cols} + {20'b0, jb_base, 10'b0})
                    > 32'(DM_MAX))) begin
              err_fr <= 1'b1;                 // illegal geometry: reject
              stk_fr <= 1'b1;                 // (incl. R3 base+cols overrun)
            end else begin
              cols_q <= jb_cols;
              base_q <= jb_base;
              cnt    <= '0;
              st     <= ST_RUN;
            end
          end
        end
        ST_RUN: begin
          if (s_iv && s_ir) begin
            if (!frame_ok) begin
              err_fr <= 1'b1;
              stk_fr <= 1'b1;
              if (s_ilast) begin st <= ST_IDLE; done <= 1'b1; end
              else st <= ST_DRAIN;
            end else if (!b_fit) begin
              err_wn <= 1'b1;                 // RESID_WINDOW: abort job
              stk_wn <= 1'b1;
              if (s_ilast) begin st <= ST_IDLE; done <= 1'b1; end
              else st <= ST_DRAIN;
            end else begin
              row[eff_idx[AW-1:0]] <= y_bits; // in-place C-6 update
              if (s_ilast) begin
                st   <= ST_IDLE;
                done <= 1'b1;
              end else begin
                cnt <= cnt + CW'(1);
              end
            end
          end
        end
        ST_DRAIN: begin
          if (s_iv && s_ir && s_ilast) begin
            st   <= ST_IDLE;
            done <= 1'b1;
          end
        end
        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule
