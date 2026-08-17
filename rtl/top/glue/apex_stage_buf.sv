// apex_stage_buf.sv — v0.1 glue: row-batch staging/replay buffer for the MXE
// activation and weight streams.
//
// WHY (ARCHITECTURE.md §0 dataflow, host-sequenced v0.1 tile): the attention
// chain re-uses on-tile tensors across several MXE descriptors —
//   * h8 rows feed 3 projection matrices x 8 N-column jobs each (replay),
//   * k8 must be presented TRANSPOSED as the Q·K̂ᵀ weight stream,
//   * v8 must be presented column-block-tiled as the P·V̂ weight stream,
//   * q8 / c8 single rows are replayed once per N-column job.
// The stage buffer stores a row batch (INT8 codes, packed lane8 beats) once
// and re-emits it in the exact beat order the verified mxe_ctrl contract
// documents (mxe_ctrl.sv header: INGEST "row-major, chunk-minor"; WLOAD
// "beat (p,c) lane r = B[8p+r][c]"). Pure routing/staging — NO numerics.
//
// Jobs (one at a time, host-sequenced; D-006 semantics):
//   LOAD  (op=0): consume rows x nb lane8 beats from the load stream into
//          bank `bank`, row-major (row r beats 0..nb-1). done when the last
//          beat has been written (all beats consumed from the input skid).
//   EMIT  (op=1): emit one beat sequence from bank `bank` per `pat`:
//     PAT_ROW (0): nb beats — beat p = stored row `sel`, beats 0..nb-1.
//          This is the act stream of one M=1, K=8*nb MXE job (mxe_ctrl
//          INGEST order: beat (m=0, p) = A[0][8p+7:8p]).
//     PAT_T   (1): rows*nb beats in (p outer, c inner) order — beat (p,c) =
//          stored row c, beat p. With the stored matrix R[rows][8*nb] this
//          is the mxe_ctrl WLOAD stream of B = Rᵀ (beat (p,c) lane r =
//          B[8p+r][c] = R[c][8p+r] = stored byte r of row c beat p): the
//          Q·K̂ᵀ weight stream for one K=8*nb, N=rows OS job.
//     PAT_D   (2): MXE_N beats — beat c lane r = stored row r, beat `sel`,
//          byte c (rows > r zero-filled). With R[rows][D] this is the WLOAD
//          stream of B = R restricted to column block `sel` (beat c lane r =
//          B[8r'..][8*sel+c] = R[r][8*sel+c]): the P·V̂ weight stream for
//          one K=rows, N=MXE_N job (rows >= K lanes are zero — mxe_ctrl
//          zero-masks rows >= K itself, so the fill is belt-and-braces).
//   `last` is set on the final beat of the emission (informational per §5).
//
// Job legality (§3 style): rows in [1, R_MAX]; nb in [1, D/8]; PAT_ROW/LOAD:
// sel/rows within the bank; PAT_T: rows <= MXE_N (column count of one WLOAD
// chunk); PAT_D: sel < D/8 and rows <= MXE_N. Violation => job_error pulse
// + sticky, NO state change. bank is a 1-bit field (BANKS = 2).
//
// F-5 CLOSURE (2026-07-09, this file's two D=128 integration bugs):
//   F-5a  job_nb is NB_W = clog2(D/8)+1 bits, DERIVED from D so nb = BPR
//         itself is always expressible (D=128: BPR=16 needs 5 bits — the
//         original fixed [3:0] field truncated 16 to 0 = illegal job and
//         wedged the tile's phase-B pipeline; kept-passing regression:
//         verif/top/l3 case bug_d128_stagebuf_nb).
//   F-5b  PAT_D indexed the column block as 4'(sel_q[2:0]) — blocks 8..15
//         aliased 0..7 at D=128. Now the full legality-checked sel_q is
//         used (NB_W-cast; legality pins sel < BPR). Directed regression:
//         verif/top/l3/tb_stagebuf_patd.sv reads ALL D=128 column blocks
//         0..15 and fails on any aliasing.
//
// D-006: done <=> (LOAD) all beats consumed / (EMIT) all beats ACCEPTED
// downstream post-skid; job_ready gated until strictly after done.
// busy = (FSM != IDLE) || beats pending in the output skid. Both streams
// skid-buffered (§5, D-019). Contents persist across jobs and soft aborts
// (a pure data buffer; only rst_n clears the job FSM — stored bytes are
// don't-care after reset since any consumer re-loads first).

module apex_stage_buf
  import apex_pkg::*;
#(
  parameter int unsigned D     = 64,   // packed row width in INT8 elements
  parameter int unsigned R_MAX = 16,   // rows per bank
  // F-5a: beats-per-row field width — wide enough for the value BPR = D/8
  // itself (not just BPR-1), derived, never a magic number.
  localparam int unsigned NB_W = $clog2(D / 8) + 1
)(
  input  logic        clk,
  input  logic        rst_n,           // synchronous, active low

  // job interface (D-006)
  input  logic        job_valid,
  output logic        job_ready,
  input  logic        job_op,          // 0 = LOAD, 1 = EMIT
  input  logic        job_bank,
  input  logic [1:0]  job_pat,         // PAT_ROW/PAT_T/PAT_D (EMIT only)
  input  logic [4:0]  job_rows,
  input  logic [NB_W-1:0] job_nb,      // beats per row (F-5a: holds BPR itself)
  input  logic [4:0]  job_sel,         // row (PAT_ROW) / column block (PAT_D)
  output logic        job_error,
  output logic        job_error_sticky,
  output logic        busy,
  output logic        done,

  // load stream in (lane8 code beats)
  input  logic        ld_valid,
  output logic        ld_ready,
  input  lane8_beat_t ld_beat,

  // emit stream out (lane8 beats, MXE act/wgt shape)
  output logic        em_valid,
  input  logic        em_ready,
  output lane8_beat_t em_beat
);

  localparam int unsigned BPR   = D / 8;
  localparam int unsigned CNT_W = 10;               // <= R_MAX*BPR = 31*16 = 496 beats

  if (D != 64 && D != 128) begin : g_chk_d
    $error("apex_stage_buf: D must be 64 or 128");
  end
  if (R_MAX < 8 || R_MAX > 31) begin : g_chk_r
    $error("apex_stage_buf: R_MAX must be in [8, 31]");
  end

  localparam logic [1:0] PAT_ROW = 2'd0;
  localparam logic [1:0] PAT_T   = 2'd1;
  localparam logic [1:0] PAT_D   = 2'd2;

  // ── storage: 2 banks x R_MAX rows x BPR 64-bit beat words ────────────────
  logic [63:0] mem [2 * R_MAX * BPR];

  function automatic int unsigned midx(input logic bank,
                                       input logic [4:0] row,
                                       input logic [NB_W-1:0] beat);
    return ((32'(bank) * R_MAX + 32'(row)) * BPR) + 32'(beat);
  endfunction

  // ── input / output skids (§5) ─────────────────────────────────────────────
  logic                           li_valid, li_ready;
  logic [$bits(lane8_beat_t)-1:0] li_vec;
  lane8_beat_t                    li_beat;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_ld_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (ld_valid),
    .s_ready (ld_ready),
    .s_data  ({ld_beat.data, ld_beat.last}),
    .m_valid (li_valid),
    .m_ready (li_ready),
    .m_data  (li_vec)
  );
  assign li_beat = lane8_beat_t'(li_vec);

  logic                           eo_valid, eo_ready;
  lane8_beat_t                    eo_beat;
  logic [$bits(lane8_beat_t)-1:0] em_vec;

  stream_skid #(.WIDTH($bits(lane8_beat_t))) u_em_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (eo_valid),
    .s_ready (eo_ready),
    .s_data  ({eo_beat.data, eo_beat.last}),
    .m_valid (em_valid),
    .m_ready (em_ready),
    .m_data  (em_vec)
  );
  assign em_beat = lane8_beat_t'(em_vec);

  // ── FSM / job state ───────────────────────────────────────────────────────
  typedef enum logic [1:0] { ST_IDLE = 2'd0, ST_LOAD = 2'd1,
                             ST_EMIT = 2'd2, ST_WAIT = 2'd3 } state_e;
  state_e           state;
  logic             bank_q;
  logic [1:0]       pat_q;
  logic [4:0]       rows_q, sel_q;
  logic [NB_W-1:0]  nb_q;
  logic [4:0]       row_c;             // load row / emit c (PAT_T) counters
  logic [NB_W-1:0]  beat_c;            // load beat / emit p counters
  logic [CNT_W-1:0] emit_no, total_q, em_acc;

  logic legal;
  always_comb begin
    legal = (job_rows != 5'd0) && (job_rows <= 5'(R_MAX))
         && (job_nb != '0) && (job_nb <= NB_W'(BPR));   // F-5a: BPR expressible
    if (!job_op) begin
      // I-C IC-QPATH: LOAD's `sel` is the BASE ROW (was: ignored, always 0),
      // so a multi-head q staging pass can place head h's row at row h. Every
      // pre-I-C caller passes sel=0, for which this clause is exactly the
      // rows <= R_MAX already asserted above — byte-identical.
      legal = legal && ((6'({1'b0, job_sel}) + 6'({1'b0, job_rows}))
                        <= 6'(R_MAX));
    end
    if (job_op) begin
      unique case (job_pat)
        PAT_ROW: legal = legal && (job_sel < 5'(R_MAX));
        PAT_T:   legal = legal && (job_rows <= 5'(MXE_N));
        PAT_D:   legal = legal && (job_sel < 5'(BPR))
                               && (job_rows <= 5'(MXE_N));
        default: legal = 1'b0;
      endcase
    end
  end

  logic [CNT_W-1:0] total_new;
  always_comb begin
    if (!job_op)                  total_new = CNT_W'(job_rows) * CNT_W'(job_nb);
    else begin
      unique case (job_pat)
        PAT_ROW: total_new = CNT_W'(job_nb);
        PAT_T:   total_new = CNT_W'(job_rows) * CNT_W'(job_nb);
        default: total_new = CNT_W'(MXE_N);          // PAT_D
      endcase
    end
  end

  // ── emit datapath (combinational reads of the reg array) ─────────────────
  logic [63:0] em_word;
  always_comb begin
    em_word = '0;
    unique case (pat_q)
      PAT_ROW: em_word = mem[midx(bank_q, sel_q, beat_c)];
      PAT_T:   em_word = mem[midx(bank_q, row_c, beat_c)];
      default: begin                                  // PAT_D: byte gather
        for (int r = 0; r < int'(MXE_N); r++) begin
          if (r < int'({27'b0, rows_q}))
            // F-5b: full legality-checked sel_q (sel < BPR) — the original
            // sel_q[2:0] slice aliased column blocks 8..15 onto 0..7 at D=128
            em_word[8*r +: 8] = mem[midx(bank_q, 5'(r), NB_W'(sel_q))]
                                   [8*emit_no[2:0] +: 8];
        end
      end
    endcase
  end

  assign li_ready     = (state == ST_LOAD);
  assign eo_valid     = (state == ST_EMIT);
  assign eo_beat.data = em_word;
  assign eo_beat.last = (emit_no == total_q - CNT_W'(1));

  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || em_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state            <= ST_IDLE;
      done             <= 1'b0;
      job_error        <= 1'b0;
      job_error_sticky <= 1'b0;
      bank_q           <= 1'b0;
      pat_q            <= '0;
      rows_q           <= '0;
      sel_q            <= '0;
      nb_q             <= '0;
      row_c            <= '0;
      beat_c           <= '0;
      emit_no          <= '0;
      total_q          <= '0;
      em_acc           <= '0;
    end else begin
      done      <= 1'b0;
      job_error <= 1'b0;
      if (em_valid && em_ready) em_acc <= em_acc + CNT_W'(1);

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state   <= job_op ? ST_EMIT : ST_LOAD;
              bank_q  <= job_bank;
              pat_q   <= job_op ? job_pat : PAT_ROW;
              rows_q  <= job_rows;
              nb_q    <= job_nb;
              sel_q   <= job_sel;
              // LOAD writes rows sel..sel+rows-1 (I-C: per-head q staging);
              // EMIT's row_c is a PAT_T column counter and still starts at 0.
              row_c   <= job_op ? 5'd0 : job_sel;
              beat_c  <= '0;
              emit_no <= '0;
              em_acc  <= '0;
              total_q <= total_new;
            end else begin                 // §3: pulse + sticky, no effects
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
            end
          end
        end

        ST_LOAD: begin
          if (li_valid && li_ready) begin
            mem[midx(bank_q, row_c, beat_c)] <= li_beat.data;
            if (beat_c == nb_q - NB_W'(1)) begin
              beat_c <= '0;
              if (row_c == sel_q + rows_q - 5'd1) begin
                done  <= 1'b1;             // last beat consumed = written
                state <= ST_IDLE;
              end else begin
                row_c <= row_c + 5'd1;
              end
            end else begin
              beat_c <= beat_c + NB_W'(1);
            end
          end
        end

        ST_EMIT: begin
          if (eo_ready) begin              // beat accepted into out skid
            emit_no <= emit_no + CNT_W'(1);
            if (emit_no == total_q - CNT_W'(1)) begin
              state <= ST_WAIT;
            end else begin
              unique case (pat_q)
                PAT_ROW: beat_c <= beat_c + NB_W'(1);
                PAT_T: begin               // c inner (rows), p outer (nb)
                  if (row_c == rows_q - 5'd1) begin
                    row_c  <= '0;
                    beat_c <= beat_c + NB_W'(1);
                  end else begin
                    row_c <= row_c + 5'd1;
                  end
                end
                default: ;                 // PAT_D indexes via emit_no
              endcase
            end
          end
        end

        ST_WAIT: begin                     // D-006: post-skid acceptance
          if (em_acc == total_q) begin
            done  <= 1'b1;
            state <= ST_IDLE;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

  // load-stream `last` flags deliberately not interpreted (count-framed)
  logic unused_ok;
  assign unused_ok = li_beat.last;

endmodule
