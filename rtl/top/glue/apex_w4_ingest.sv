// apex_w4_ingest.sv — the W4 weight-ingest lane: tile-side glue that turns
// the external xw byte pipe into the ADOPTED W4 G=32/(B) weight path
// (D-031; docs/design/W4_DATAPATH.md is this lane's contract).
//
// Implements: W4B_FEEDER.md integration notes 1-2 (placement on the xw
//             path behind a route/CSR bit; gs scales ride the same byte
//             pipe as the packed beats), B3_WEIGHT_PATH.md §8 (host-
//             computed stripe s8, DIRECT-prep blobs), ARCHITECTURE C-4
//             (the MXE still sees exactly one dtype - INT8).
// Golden arbiter: golden/apex_golden/weight_codec.wfeed_w4b_to_i8 (per
//             element) + golden/apex_golden/w4_lane.py (this lane's wire
//             framing + job-level GEMM composition; TB verif/top/w4).
//
// ── What this module adds over the verified feeder ─────────────────────────
// mxe_wfeed_w4b (unit-verified: 6 regimes + 4.06M-point exhaustive sweep +
// 5/5 mutants) consumes THREE channels: a job pulse {beats,K,N,s8}, an fp16
// group-scale sideband gs_*, and the packed-W4 stream pw_*. The tile has
// ONE external weight pipe (xw). This glue is the demux between them:
//
//   xw beats ──► [PH_GS: ceil(ngtot/4) beats ── 4 fp16/beat ──► gs_*]
//               [PH_PW: ceil(beats/2) packed beats ──────────► pw_*]
//   u_w4b.xw  ──────────────────────────────────────────────► MXE wgt leg
//
// WIRE ORDER IS SCALES-FIRST (normative, and the one deliberate deviation
// from W4B_FEEDER.md note 2's "scales immediately after the packed beats"
// IMAGE layout): the feeder stalls emission of beat 0 until gid 0's scale
// is resident and holds only a 2-deep pw skid, so a packed-then-scales
// WIRE order deadlocks against a single in-order pipe by construction.
// Scales-first is legal by the feeder's own contract ("early (eager)
// scales park in the skid + RAM" - all ngtot of them), costs nothing, and
// the image layout stays free to keep scales after packed per job: the
// producer (host script order / a fuel gs-record fetched first) chooses
// fetch order, exactly like the walker's two-record descriptor plan.
//
// ── GS beat format (normative; mirrors apex_gam_unpack's int16 layout) ────
// One 64-bit xw beat carries FOUR fp16 group scales little-endian: scale
// slot i of beat b = bits [16i +: 16] = gid 4b+i, ascending-gid contract.
// The final beat's slots >= ngtot are PADDING - never pushed to the
// feeder (its gs_ready backpressures beyond ngtot by design, so pushing
// padding would wedge; the counter here stops exactly at ngtot).
//
// ── Job framing ───────────────────────────────────────────────────────────
// go pulses with {job_k, job_n, job_s8} stable (CSR-registered upstream).
// This glue derives beats = ceil(K/8)*N and ngtot = N*ceil(K/G) itself -
// the CSR window carries K/N/s8 only, so the feeder's beats==KB*N legality
// holds by construction. A go while busy, or an illegal {K,N}, pulses err
// + sets err_sticky and changes NOTHING else (§3 idiom). mode_pass=1 skips
// PH_GS (ngtot=0) and runs the feeder's INT8 passthrough - the framing
// debug mode; W4B dequant is mode_pass=0.
//
// ── Ingest-proof counters (the "doubled effective ingest" evidence) ───────
// cnt_pw / cnt_gs count xw beats consumed by each phase, cnt_out counts
// INT8 beats delivered toward the MXE; all cumulative, cleared only by
// reset. A W4B job consumes ceil(beats/2)+ceil(ngtot/4) xw beats where
// the INT8 path consumes beats: the weight payload rides 2 beats/packed
// beat (exactly 2x), and with G=32 scale overhead the whole-job ratio is
// 8/4.5 bits-per-weight (~1.78x) - both visible in these counters.
//
// D-006: done <=> the feeder's done (its own post-skid acceptance count);
// busy covers this glue's phases OR the feeder. Quasi-static rule: the
// upstream route level (lane_en, owned by apex_top's W4 window) changes
// only while this module reports !busy - the rt_* contract verbatim.

`ifndef APEX_W4_INGEST_SV
`define APEX_W4_INGEST_SV

module apex_w4_ingest
  import apex_pkg::*;
#(
  parameter int unsigned G         = 32,    // K-rows per scale group (16|32)
  parameter int unsigned BEATS_MAX = 2048,  // KB*N ceiling (K_MAX geometry)
  parameter int unsigned K_MAX_P   = 2048   // mirrors apex_pkg K_MAX
)(
  input  logic         clk,
  input  logic         rst_n,              // synchronous, active low

  // W4 CSR window side (apex_top owns the registers)
  input  logic         mode_pass,          // 1 = INT8 passthrough framing
  input  logic         go,                 // 1-cycle job kick
  input  logic [11:0]  job_k,
  input  logic [3:0]   job_n,
  input  logic [15:0]  job_s8,             // fp16 stripe requant scale

  // xw stream in (the external weight pipe, post gamma-window demux)
  input  logic         in_valid,
  output logic         in_ready,
  input  lane8_beat_t  in_beat,

  // INT8 weight stream out (-> the MXE weight-source mux xw leg)
  output logic         out_valid,
  input  logic         out_ready,
  output lane8_beat_t  out_beat,

  // status / proof surface (-> W4 CSR window)
  output logic         busy,
  output logic         done,               // 1-cycle, feeder D-006 done
  output logic         err,                // 1-cycle pulse
  output logic         err_sticky,         // W1C at the CSR window
  input  logic         err_clr,
  output logic [1:0]   phase_r,            // 0 idle / 1 gs / 2 pw / 3 drain
  output logic [15:0]  cnt_pw,             // packed xw beats consumed
  output logic [15:0]  cnt_gs,             // scale xw beats consumed
  output logic [31:0]  cnt_out             // INT8 beats emitted toward MXE
);

  // ── derived geometry (same arithmetic as the feeder's legality) ──────────
  localparam int unsigned NG_MAX = 8 * ((K_MAX_P + G - 1) / G);
  localparam int unsigned GW     = $clog2(NG_MAX);

  typedef enum logic [1:0] { PH_IDLE, PH_GS, PH_PW, PH_DRAIN } ph_e;
  ph_e ph;
  assign phase_r = 2'(ph);

  logic [15:0]  beats_c;                    // ceil(K/8) * N
  logic [GW:0]  ngtot_c;                    // N * ceil(K/G) (0 in passthrough)
  assign beats_c = 16'(((32'(job_k) + 7) >> 3) * 32'(job_n));
  assign ngtot_c = mode_pass ? '0
                 : (GW+1)'(32'(job_n) * ((32'(job_k) + G - 1) / G));

  logic legal_c;
  assign legal_c = (job_n >= 4'd1) && (job_n <= 4'd8)
                && (job_k >= 12'd1) && (32'(job_k) <= K_MAX_P)
                && (32'(beats_c) <= BEATS_MAX);

  // registered job geometry
  logic [15:0] pw_need_q;                   // packed beats to route
  logic [GW:0] gs_need_q;                   // scales to push (not beats)
  logic [15:0] pw_cnt_q;
  logic [GW:0] gs_cnt_q;                    // scales pushed so far

  // ── feeder job push (one registered pulse; feeder is IDLE by !busy) ──────
  logic        fj_valid_q;
  logic        fj_ready, fb_busy, fb_done, fb_err, fb_err_sticky;

  // ── gs serializer (apex_gam_unpack idiom: hold + shift, LE slots) ────────
  logic [63:0] gh_q;
  logic [2:0]  gh_have_q;                   // fp16 slots still to push (0..4)
  logic        gs_v;
  logic        gs_r;
  assign gs_v = (gh_have_q != 3'd0);

  logic [GW:0] gs_left;                     // scales not yet pushed
  assign gs_left = gs_need_q - gs_cnt_q;

  // xw routing per phase. PH_GS accepts a beat only when the hold register
  // is drained; PH_PW wires straight through to the feeder's pw skid.
  logic pw_v, pw_r;
  assign in_ready = (ph == PH_GS) ? (gh_have_q == 3'd0)
                  : (ph == PH_PW) ? pw_r
                                  : 1'b0;
  assign pw_v     = (ph == PH_PW) && in_valid && (pw_cnt_q < pw_need_q);

  wire in_acc  = in_valid && in_ready;
  wire gs_push = gs_v && gs_r;

  // ── FSM ──────────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ph         <= PH_IDLE;
      fj_valid_q <= 1'b0;
      pw_need_q  <= '0;
      gs_need_q  <= '0;
      pw_cnt_q   <= '0;
      gs_cnt_q   <= '0;
      gh_q       <= '0;
      gh_have_q  <= '0;
      err        <= 1'b0;
      err_sticky <= 1'b0;
    end else begin
      err <= 1'b0;
      if (err_clr) err_sticky <= 1'b0;      // W1C; same-cycle error wins below

      // feeder job pulse retires on its handshake
      if (fj_valid_q && fj_ready) fj_valid_q <= 1'b0;

      // go anywhere but a truly idle lane is REFUSED loudly (§3 pulse +
      // sticky, zero state change) — never silently dropped: the busy term
      // covers the one-cycle feeder-drain tail after ph returns to IDLE
      if (go && !((ph == PH_IDLE) && !fb_busy && !fj_valid_q)) begin
        err        <= 1'b1;
        err_sticky <= 1'b1;
      end

      unique case (ph)
        PH_IDLE: begin
          if (go && !fb_busy && !fj_valid_q) begin
            if (!legal_c) begin             // §3: pulse + sticky, no effects
              err        <= 1'b1;
              err_sticky <= 1'b1;
            end else begin
              fj_valid_q <= 1'b1;           // feeder samples mode at accept
              pw_need_q  <= mode_pass ? beats_c : 16'((beats_c + 16'd1) >> 1);
              gs_need_q  <= ngtot_c;
              pw_cnt_q   <= '0;
              gs_cnt_q   <= '0;
              gh_have_q  <= '0;
              ph         <= (ngtot_c != '0) ? PH_GS : PH_PW;
            end
          end
        end

        PH_GS: begin
          if (in_acc) begin                 // capture 4 (or the tail) scales
            gh_q      <= in_beat.data;
            gh_have_q <= (gs_left >= (GW+1)'(4)) ? 3'd4 : 3'(gs_left);
          end else if (gs_push) begin
            gh_q      <= {16'b0, gh_q[63:16]};
            gh_have_q <= gh_have_q - 3'd1;
            gs_cnt_q  <= gs_cnt_q + (GW+1)'(1);
            if ((gh_have_q == 3'd1) && (gs_cnt_q + (GW+1)'(1) == gs_need_q))
              ph <= PH_PW;
          end
        end

        PH_PW: begin
          if (pw_v && pw_r) begin
            pw_cnt_q <= pw_cnt_q + 16'd1;
            if (pw_cnt_q == pw_need_q - 16'd1) ph <= PH_DRAIN;
          end
        end

        PH_DRAIN: begin                     // feeder drains its pipe + skids
          if (fb_done) ph <= PH_IDLE;
        end

        default: ph <= PH_IDLE;
      endcase

      // the feeder's own reject is structurally unreachable after legal_c,
      // but never lose an error if it fires (framing drift would be loud)
      if (fb_err) begin
        err        <= 1'b1;
        err_sticky <= 1'b1;
        ph         <= PH_IDLE;
      end
    end
  end

  // ── the verified W4-B feeder (D-031; untouched sibling instance) ─────────
  mxe_wfeed_w4b #(
    .G         (G),
    .BEATS_MAX (BEATS_MAX),
    .K_MAX_P   (K_MAX_P)
  ) u_w4b (
    .clk              (clk),
    .rst_n            (rst_n),
    .w4b_en           (!mode_pass),
    .job_valid        (fj_valid_q),
    .job_ready        (fj_ready),
    .job_beats        (beats_c),
    .job_k            (job_k),
    .job_n            (job_n),
    .job_s8           (job_s8),
    .job_error        (fb_err),
    .job_error_sticky (fb_err_sticky),
    .pw_valid         (pw_v),
    .pw_ready         (pw_r),
    .pw_beat          (in_beat),
    .gs_valid         (gs_v),
    .gs_ready         (gs_r),
    .gs_data          (gh_q[15:0]),
    .xw_valid         (out_valid),
    .xw_ready         (out_ready),
    .xw_beat          (out_beat),
    .busy             (fb_busy),
    .done             (fb_done)
  );

  assign done = fb_done;
  assign busy = (ph != PH_IDLE) || fb_busy || fj_valid_q;

  // ── proof counters (cumulative; reset-only clear) ────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cnt_pw  <= '0;
      cnt_gs  <= '0;
      cnt_out <= '0;
    end else begin
      if (in_acc && (ph == PH_PW)) cnt_pw  <= cnt_pw + 16'd1;
      if (in_acc && (ph == PH_GS)) cnt_gs  <= cnt_gs + 16'd1;
      if (out_valid && out_ready)  cnt_out <= cnt_out + 32'd1;
    end
  end

  // job_k/job_n ride to the feeder directly; beats_c is the derived field.
  // The feeder's own sticky is folded into ours via fb_err; its level is
  // consumed here for lint (the pulse is the authoritative event).
  wire unused_ok = &{1'b0, fb_err_sticky, in_beat.last};

endmodule

`endif // APEX_W4_INGEST_SV
