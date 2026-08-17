// mxe_stub.sv — behavioral MXE model for the SEQ suite, derived from the
// D-006 / §3 / §5 CONTRACT (apex_pkg.sv job-semantics block + mxe_top.sv
// header), NOT from the MXE implementation. MXE itself is a verified black
// box (verif/mxe 9/9); SEQ only depends on the contract, so the stub
// implements exactly that:
//
//   - desc_ready only while IDLE (never between accept and done — D-006),
//     with randomized accept-latency duty per lat_mode;
//   - legal descriptor (mirror of the §3 conservative legality:
//     opcode ∈ {WS,OS}, 1 ≤ K ≤ K_MAX, 1 ≤ M ≤ 64, 1 ≤ N ≤ MXE_N,
//     all buffer ids == 0): busy, then M result beats on a §5 valid/ready
//     stream with randomized inter-beat gaps (data held stable while
//     valid && !ready, never retracted), `last` on the final beat, then a
//     1-cycle done pulse STRICTLY AFTER the final beat is accepted;
//   - illegal descriptor: consumed, 1-cycle desc_error pulse, NO done, NO
//     beats, no state change (ready to accept the next descriptor);
//   - busy = (state != IDLE).
//
// acc_pulse/acc_legal expose the accept event + the stub's own legality
// verdict so the TB can cross-check the generator's golden legality.
// force_stall holds desc_ready low (phase-targeted reset / md_valid-stall
// tests). Verification collateral only — not synthesizable style-audited,
// but -Wall clean like everything else in the suite.

module mxe_stub
  import apex_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,          // synchronous, active-low

  input  logic [1:0]   lat_mode,       // 0 fast / 1 random / 2 storm
  input  logic [31:0]  seed,
  input  logic         force_stall,    // hold desc_ready low

  // descriptor interface (mirrors mxe_top desc port group)
  input  logic         desc_valid,
  output logic         desc_ready,
  input  mxe_desc_t    desc,
  output logic         desc_error,     // 1-cycle pulse, rejected descriptor
  output logic         busy,
  output logic         done,           // 1-cycle pulse, after last beat ACCEPTED

  // result stream out (§5)
  output logic         res_valid,
  input  logic         res_ready,
  output lane32_beat_t res_beat,

  // TB cross-check taps
  output logic         acc_pulse,      // descriptor accepted this cycle
  output logic         acc_legal       // stub legality verdict for `desc`
);

  localparam int unsigned M_MAX = 64;  // §3 implemented range (mxe_ctrl mirror)

  typedef enum logic [2:0] {
    S_IDLE = 3'd0, S_ERR = 3'd1, S_LAT = 3'd2, S_BEAT = 3'd3, S_DONE = 3'd4
  } state_e;
  state_e state;

  // ── legality (contract mirror — MUST match gen_seq_vectors.py legal()) ────
  function automatic logic desc_legal(input mxe_desc_t d);
    logic unused_ok;
    unused_ok = &{1'b0, d.reserved, d.mode_os, d.accumulate, d.requant_en,
                  d.rq_shift, d.rq_scale};    // legality ignores these fields
    return ((d.opcode == 8'(OP_GEMM_WS)) || (d.opcode == 8'(OP_GEMM_OS)))
        && (d.k_dim != '0) && (32'(d.k_dim) <= K_MAX)
        && (d.m_dim != '0) && (32'(d.m_dim) <= M_MAX)
        && (d.n_dim != '0) && (32'(d.n_dim) <= MXE_N)
        && (d.src_a_buf == 4'd0) && (d.src_b_buf == 4'd0)
        && (d.dst_buf == 4'd0);
  endfunction

  // ── rng ────────────────────────────────────────────────────────────────────
  logic [31:0] lfsr;
  logic        rdy_duty;
  always_comb begin
    unique case (lat_mode)
      2'd0:    rdy_duty = 1'b1;
      2'd1:    rdy_duty = |lfsr[1:0];          // ~75%
      default: rdy_duty = (lfsr[3:0] == '0);   // ~6% storm
    endcase
  end

  assign desc_ready = (state == S_IDLE) && rdy_duty && !force_stall;
  assign acc_pulse  = desc_valid && desc_ready;
  assign acc_legal  = desc_legal(desc);
  assign busy       = (state != S_IDLE);

  // ── job engine ─────────────────────────────────────────────────────────────
  logic [11:0] m_q, beat_idx;
  logic [7:0]  lat_cnt, gap_cnt;

  function automatic logic [7:0] rnd_lat(input logic [31:0] r);
    logic unused_ok;
    unused_ok = &{1'b0, r[31:6]};
    unique case (lat_mode)
      2'd0:    rnd_lat = 8'd0;
      2'd1:    rnd_lat = {5'b0, r[2:0]};             // 0..7
      default: rnd_lat = {2'b0, r[5:0]};             // 0..63
    endcase
  endfunction

  function automatic logic [7:0] rnd_gap(input logic [31:0] r);
    logic unused_ok;
    unused_ok = &{1'b0, r[31:8]};
    unique case (lat_mode)
      2'd0:    rnd_gap = 8'd0;
      2'd1:    rnd_gap = (r[3:0] == '0) ? {5'b0, r[6:4]} : 8'd0;
      default: rnd_gap = (r[2:0] == '0) ? {3'b0, r[7:3]} : 8'd0;
    endcase
  endfunction

  function automatic logic [32*MXE_N-1:0] beat_data(input logic [11:0] idx);
    logic [32*MXE_N-1:0] v;
    for (int unsigned c = 0; c < MXE_N; c++)
      v[32*c +: 32] = {12'(c), 8'h5A, idx};
    return v;
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      lfsr       <= (seed ^ 32'hBEEF_5EED) | 32'h1;   // never all-zero
      m_q        <= '0;
      beat_idx   <= '0;
      lat_cnt    <= '0;
      gap_cnt    <= '0;
      desc_error <= 1'b0;
      done       <= 1'b0;
      res_valid  <= 1'b0;
      res_beat   <= '0;
    end else begin
      lfsr       <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      desc_error <= 1'b0;
      done       <= 1'b0;

      unique case (state)
        S_IDLE: if (acc_pulse) begin
          if (!acc_legal) begin
            state <= S_ERR;
          end else begin
            m_q      <= desc.m_dim;
            beat_idx <= '0;
            lat_cnt  <= rnd_lat(lfsr);
            state    <= S_LAT;
          end
        end

        S_ERR: begin
          desc_error <= 1'b1;        // consumed, no state change, no done (§3)
          state      <= S_IDLE;
        end

        S_LAT: begin
          if (lat_cnt == '0) begin
            res_valid <= 1'b1;
            res_beat  <= '{data: beat_data(12'd0), last: (m_q == 12'd1)};
            state     <= S_BEAT;
          end else begin
            lat_cnt <= lat_cnt - 8'd1;
          end
        end

        S_BEAT: begin
          if (res_valid && res_ready) begin
            if (beat_idx == m_q - 12'd1) begin
              res_valid <= 1'b0;
              state     <= S_DONE;   // done only AFTER the last beat accepted
            end else begin
              beat_idx <= beat_idx + 12'd1;
              gap_cnt  <= rnd_gap(lfsr);
              if (rnd_gap(lfsr) != '0) begin
                res_valid <= 1'b0;   // gap; beat re-presented after it
              end else begin
                res_beat <= '{data: beat_data(beat_idx + 12'd1),
                              last: (beat_idx + 12'd1 == m_q - 12'd1)};
              end
            end
          end else if (!res_valid) begin
            if (gap_cnt == '0) begin
              res_valid <= 1'b1;
              res_beat  <= '{data: beat_data(beat_idx),
                             last: (beat_idx == m_q - 12'd1)};
            end else begin
              gap_cnt <= gap_cnt - 8'd1;
            end
          end
          // res_valid && !res_ready: hold everything (§5 stability)
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  // unused descriptor fields (stub dispatches on dims/opcode/bufs only)
  logic unused_ok;
  assign unused_ok = &{1'b0, desc.reserved, desc.mode_os, desc.accumulate,
                       desc.requant_en, desc.rq_shift, desc.rq_scale};

endmodule
