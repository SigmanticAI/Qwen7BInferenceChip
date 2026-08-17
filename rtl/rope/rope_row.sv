// rope_row.sv — streaming RoPE row unit for the S-2 fp16 write bus.
//
// Consumes one D-beat fp16 row (the projection output for one K row or the
// q row, bias already folded upstream), applies the C-ROPE HF half-split
// rotation to every channel pair (i, i+half), and emits the rotated row in
// the SAME beat order — so it can sit between apex_scale_quant's F16 output
// and the KVQ s_axis (IB_LAYER.md §3) with rope-off routed around it in the
// stage-4 glue mux (this module always rotates; bypass is external).
//
// Phase codes come from an EXTERNAL phase RAM (glue-owned, host-loaded per
// C-ROPE: the per-(m,i) table is precomputed in float64 and quantized ONCE
// host-side; on-tile incremental phase generation is banned as not
// bit-exact — IB_LAYER.md §2 L3). This module reads pair j's code on a
// 1-cycle synchronous port; bank/position base is the glue's business.
//
// Dataflow per row (perf is signoff-secondary; ARCHITECTURE §0):
//   COLLECT  D beats -> row buffer (x_buf).  Framing: `last` must arrive
//            exactly on beat D-1. Early last  -> frame_error, row dropped.
//            Missing last -> frame_error, RESYNC (consume to next last).
//   EMIT     2 cycles/beat: EM_ADDR reads x_buf pair + drives ph_addr;
//            EM_OUT rotates (rope_pair_fx, combinational) and presents
//            y_i (beats 0..half-1) / y_ih (beats half..D-1) to the output
//            skid; ph_addr held stable while waiting so the external RAM's
//            registered read data stays valid.
//   V rows are NOT rotated — they simply never route through this module.
//
// §5 discipline: skid on both boundaries; m_valid/m_data stable until
// accepted (skid property); frame reject = pulse + sticky, no output, next
// row clean (§3 legality shape). Mid-op rst_n clears everything (skids
// included); mid-op soft-abort arrives with the stage-4 glue if needed.
//
// D is the per-head row length (head_dim build point: 64/128 — RoPE is
// per-head and never sees hidden-D rows, IB_LAYER.md §2 L3). D must be even.

module rope_row
  import f16_arith_pkg::*;
#(
  parameter int unsigned D = 64,
  localparam int unsigned HALF   = D / 2,
  localparam int unsigned CNT_W  = $clog2(D + 1),
  localparam int unsigned PAIR_W = $clog2(HALF)
)(
  input  logic              clk,
  input  logic              rst_n,          // synchronous, active-low

  // fp16 row stream in (S-2 bus grade)
  input  logic              s_valid,
  output logic              s_ready,
  input  logic [15:0]       s_data,
  input  logic              s_last,

  // rotated fp16 row stream out
  output logic              m_valid,
  input  logic              m_ready,
  output logic [15:0]       m_data,
  output logic              m_last,

  // external phase RAM read port (1-cycle registered read, glue-owned)
  output logic [PAIR_W-1:0] ph_addr,
  input  logic [13:0]       ph_data,

  output logic              busy,
  output logic              frame_error,        // 1-cycle pulse
  output logic              frame_error_sticky  // rst_n-only sticky
);

  if (D < 2 || (D % 2) != 0) begin : g_chk_d
    $error("rope_row: D must be even and >= 2");
  end

  // ── boundary skids (§5) ───────────────────────────────────────────────────
  logic        i_valid, i_ready, i_last;
  logic [15:0] i_data;
  stream_skid #(.WIDTH(17)) u_in_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (s_valid), .s_ready (s_ready), .s_data ({s_last, s_data}),
    .m_valid (i_valid), .m_ready (i_ready), .m_data ({i_last, i_data})
  );

  logic        o_valid, o_ready;
  logic [16:0] o_vec, m_vec;
  stream_skid #(.WIDTH(17)) u_out_skid (
    .clk (clk), .rst_n (rst_n),
    .s_valid (o_valid), .s_ready (o_ready), .s_data (o_vec),
    .m_valid (m_valid), .m_ready (m_ready), .m_data (m_vec)
  );
  assign m_data = m_vec[15:0];
  assign m_last = m_vec[16];

  // ── row buffer + FSM ──────────────────────────────────────────────────────
  typedef enum logic [1:0] {ST_COLLECT, ST_RESYNC, ST_EM_ADDR, ST_EM_OUT} st_e;
  st_e                st;
  logic [15:0]        x_buf [D];
  logic [CNT_W-1:0]   cnt;              // collect beat count
  logic [CNT_W-1:0]   k;                // emit beat index 0..D-1
  logic [15:0]        xa_q, xb_q;       // registered pair operands
  logic               err_q, sticky_q;

  wire [PAIR_W-1:0] pair_j =
      (k < CNT_W'(HALF)) ? PAIR_W'(k) : PAIR_W'(k - CNT_W'(HALF));
  wire              sel_hi = (k >= CNT_W'(HALF));

  // rotation core (combinational; operands registered in EM_ADDR, phase
  // data registered by the external RAM — both stable through EM_OUT)
  logic [15:0] y_i, y_ih;
  rope_pair_fx u_pair (
    .u_q (ph_data), .x_i (xa_q), .x_ih (xb_q), .y_i (y_i), .y_ih (y_ih)
  );

  assign ph_addr = pair_j;
  assign i_ready = (st == ST_COLLECT) || (st == ST_RESYNC);
  assign o_valid = (st == ST_EM_OUT);
  assign o_vec   = {(k == CNT_W'(D - 1)), (sel_hi ? y_ih : y_i)};
  assign busy    = (st != ST_COLLECT) || (cnt != '0);
  assign frame_error        = err_q;
  assign frame_error_sticky = sticky_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st       <= ST_COLLECT;
      cnt      <= '0;
      k        <= '0;
      xa_q     <= '0;
      xb_q     <= '0;
      err_q    <= 1'b0;
      sticky_q <= 1'b0;
    end else begin
      err_q <= 1'b0;
      unique case (st)
        ST_COLLECT: begin
          if (i_valid && i_ready) begin
            if (i_last && (cnt != CNT_W'(D - 1))) begin
              // early last: reject row, no output, clean restart (§3)
              err_q    <= 1'b1;
              sticky_q <= 1'b1;
              cnt      <= '0;
            end else if (!i_last && (cnt == CNT_W'(D - 1))) begin
              // missing last: reject + resync to the next last
              err_q    <= 1'b1;
              sticky_q <= 1'b1;
              cnt      <= '0;
              st       <= ST_RESYNC;
            end else begin
              x_buf[cnt[$clog2(D)-1:0]] <= i_data;
              if (cnt == CNT_W'(D - 1)) begin
                cnt <= '0;
                k   <= '0;
                st  <= ST_EM_ADDR;
              end else begin
                cnt <= cnt + CNT_W'(1);
              end
            end
          end
        end
        ST_RESYNC: begin
          if (i_valid && i_ready && i_last) st <= ST_COLLECT;
        end
        ST_EM_ADDR: begin
          // ph_addr = pair_j is live this cycle; RAM data valid next cycle
          xa_q <= x_buf[{1'b0, pair_j}];
          xb_q <= x_buf[{1'b0, pair_j} + $clog2(D)'(HALF)];
          st   <= ST_EM_OUT;
        end
        ST_EM_OUT: begin
          if (o_ready) begin
            if (k == CNT_W'(D - 1)) begin
              k  <= '0;
              st <= ST_COLLECT;
            end else begin
              k  <= k + CNT_W'(1);
              st <= ST_EM_ADDR;
            end
          end
        end
        default: st <= ST_COLLECT;
      endcase
    end
  end

endmodule
