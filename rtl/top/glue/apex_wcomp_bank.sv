// apex_wcomp_bank.sv — in-tile per-KV-HEAD COMPOSITE SCALE-CACHE BANK:
// N_ENG verified, UNMODIFIED seq_walker_comp instances (one per KV head)
// behind the SAME live engine select the S4b KVQ bank consumes. Sibling of
// rtl/top/glue/apex_kvq_gqa_bank.sv in construction and in contract; the
// two banks MUST be driven by one select, in the same cycle, or a record
// lands in engine g while its scale lands in cache g'.
//
// WHY THIS BLOCK EXISTS — LEVEL_C_INTEGRATION.md §9.1 F6 part (i)
// (2026-07-24, A/B-proven by the W-G3 executor lane, not asserted):
// seq_walker_comp's scale cache is indexed by RECORD ADDRESS ONLY (K at
// [0,T), V at [T,2T) — the L3 store map, WALK_SC_ENT = 2T entries). Under
// GQA banking (KVQ_GQA_NENG>1, kv_map=1) every KV group re-uses that same
// record space inside its OWN engine, so ONE cache instance has all H_kv
// groups colliding on one 2T array: the last group stored wins and every
// earlier group's composites are silently wrong. Evidence — same walker,
// vectors, scoreboard and seed, topology the only variable:
//   N_ENG caches      16,139 checks / 0 errors
//   one shared cache  16,139 checks / 980 errors — ALL 980 composite words
//                     (560/560 CS, 420/560 QS), the surviving QS being
//                     exactly heads 21-27 = KV group 3, the last staged.
//
// This bank is the TILE-SIDE form of a topology that is already validated at
// unit level: verif/seq_walker/tb_walker2_sb.sv:145-179 runs N_ENG real
// seq_walker_comp instances routed by the walker's kv_eng_sel, with the sel
// value checked per head and per KVQ write (IB_WALK.md §4 stage 2/3 — 7B:
// 28 heads, 4 engines, 6,821 checks, 0 errors). Capacity and isolation are
// realized by BANKING the verified unit, NEVER by editing it — the D-024 /
// S4b discipline.
//
// ROUTING CONTRACT (mirrors apex_kvq_gqa_bank's, deliberately):
//   eng_sel is a quasi-static LEVEL. The s_q tap, the composite request and
//   the response mux all route to instance[eng_sel] COMBINATIONALLY. The
//   store-time snoop is gated by the SAME select, then REGISTERED once in
//   the bank's A0 ingress stage (2026-08-13, below) — the association is
//   still decided combinationally in the gate cycle, delivered one cycle
//   later. The select may change only while the bank is quiescent
//   (no record mid-frame on the snooped stream, no composite in flight):
//   per STORE RUN and per head — the walker's poll-idle-before-WRITE_ADDR /
//   one-composite-at-a-time invariants satisfy this by construction, and the
//   KVQ bank next door is switched by the same net at the same instant.
//   Mid-record select switches are host contract violations (same class as
//   rt_* route levels / eng_sel on the KVQ bank).
//
//   snp_flush is BROADCAST, not routed: the D-020 soft reset is a tile-wide
//   abort, and a partial record left in a DESELECTED instance is exactly the
//   state that must not survive it.
//
//   err_frame / err_stale are OR-REDUCED over the whole bank: an error must
//   never be lost because the select moved off the instance that raised it
//   (the pulses are consumed one cycle later by the WALK sticky).
//
// The s_q tap is ROUTED, not broadcast, because s_q is PER HEAD and a head
// belongs to exactly one KV group (golden GQA slicing h // (H/H_kv), which
// lives in the SEQUENCER — this bank never computes a mapping). That is the
// same hook tb_walker2_sb models with push_sq(hgrp[head]).
//
// NECESSARY, NOT SUFFICIENT — say it here so no reader infers more: this
// bank closes F6 part (i) ONLY. Part (ii) is untouched and still blocks a
// multi-head fmt=1 walk: apex_top's hd_ready is a single sticky
// "s_q snooped, unconsumed" latch and the walk_en mode mux holds every host
// job port off during a walk, so no LATER head's q row can be staged (and
// therefore no fresh s_q can reach this bank) mid-walk. Each instance holds
// exactly one s_q at a time, exactly as the unit does; refreshing it per
// head is the segment-1 q-projection path's job (IB_LAYER §3c), not this
// block's.
//
// Every port of every instance is driven/consumed — full -Wall, no waivers.

`default_nettype none

module apex_wcomp_bank
  import seq_walker_pkg::*;
#(
  parameter int unsigned N_ENG = 4,     // per-KV-head caches (H_kv)
  parameter int unsigned CFG_D = 64,    // build geometry; 64 or 128 (D-021)
  localparam int unsigned ENG_W = (N_ENG > 1) ? $clog2(N_ENG) : 1
)(
  input  wire                       clk,
  input  wire                       rst_n,

  // live engine select (quasi-static; see routing contract above)
  input  wire  [ENG_W-1:0]          eng_sel,

  // store-time scale harvest (§A-2) — routed: the snooped record belongs to
  // the engine that is storing it
  input  wire                       snp_valid,
  input  wire  [15:0]               snp_data,
  input  wire                       snp_last,
  input  wire  [WALK_SC_AW-1:0]     snp_addr,
  input  wire                       snp_flush,   // broadcast (D-020 abort)

  // s_q tap — routed (per-head scale into the head's KV group)
  input  wire                       sq_valid,
  input  wire  [15:0]               sq_data,

  // composite request / response — routed + muxed
  input  wire                       req_valid,
  output logic                      req_ready,
  input  wire                       req_is_qs,
  input  wire  [WALK_SC_AW-1:0]     req_idx,
  output logic                      res_valid,
  input  wire                       res_ready,
  output logic [31:0]               res_data,

  // §3-style error pulses, OR-reduced over the bank
  output logic                      err_frame,
  output logic                      err_stale,
  // DBG-v5 (2026-08-08, passive): WHICH instance pulsed stale — the
  // OR-reduction above cannot say, and the walked-attention differential
  // needs the attribution. Purely observational; no behavior change.
  output logic [N_ENG-1:0]          dbg_stale_eng,
  output logic                      dbg_stale_sc,
  output logic                      dbg_stale_sq
);

  // per-instance wire bundles (index = KV-head engine index)
  logic        e_req_ready[N_ENG];
  logic        e_res_valid[N_ENG];
  logic [31:0] e_res_data [N_ENG];
  logic        e_err_frame[N_ENG];
  logic        e_err_stale[N_ENG];
  logic        e_dbg_sc[N_ENG];
  logic        e_dbg_sq[N_ENG];

  // input gating per instance
  (* keep = "true", dont_touch = "true" *)
  logic        sel        [N_ENG];
  (* keep = "true", dont_touch = "true" *)
  logic        e_snp_valid[N_ENG];
  logic        e_sq_valid [N_ENG];
  logic        e_req_valid[N_ENG];
  logic        e_res_ready[N_ENG];

  // defensive index: at non-power-of-two N_ENG the select space has codes
  // >= N_ENG (the walker's build-envelope fence refuses such geometries and
  // the l_kv_map host contract is 0..N_ENG-1) — clamp them to instance 0
  // anyway so the mux can never index out of range; at power-of-two N_ENG
  // (the H_kv=4 build) this constant-folds away. Fan-out and mux are
  // SEPARATE fine-grained assigns, the apex_kvq_gqa_bank idiom.
  // 2026-08-10 KEEP-CHAIN: the select/gating nets kept — the kill-shot
  // flight proved the write-enable chain dies between this boundary and
  // the units' flops even with the flops themselves dont_touch'd.
  (* keep = "true", dont_touch = "true" *)
  logic [ENG_W-1:0] idx;
  assign idx = (32'(eng_sel) >= N_ENG) ? '0 : eng_sel;

  generate
    for (genvar gs = 0; gs < int'(N_ENG); gs++) begin : g_sel
      assign sel[gs]         = (idx == ENG_W'(gs));
      assign e_snp_valid[gs] = snp_valid && sel[gs];
      assign e_sq_valid[gs]  = sq_valid  && sel[gs];
      assign e_req_valid[gs] = req_valid && sel[gs];
      assign e_res_ready[gs] = res_ready && sel[gs];
    end
  endgenerate

  // ── 2026-08-13 A0 INGRESS STAGE (A0 tier-2 path; CLOCK_LADDER.md §9
  //    addendum) ──────────────────────────────────────────────────────────
  // Both 2026-08-12 A0 builds put the SAME cone in every top-12 path slot:
  // the tile-level snoop flops -> g_comp[*].u_comp sc_mem BRAM write port,
  // WNS −8.68 then −8.41 after a SECOND tile-level stage (q2) — the ~8.4 ns
  // is PLACEMENT DISTANCE to the bank's BRAM pins, not logic depth, and the
  // D-033 dont_touch/keep_hierarchy fence (rightly) forbids retiming INTO
  // the bank, so no tile-level stage can ever shorten the hop. The fix the
  // §9 verdict names: the bank must OWN an ingress register — these flops
  // are placed WITH the bank, inside the keep_hierarchy region, so the
  // tile->bank route terminates at a fabric flop and the BRAM DIN/ADDR/WE
  // setup is paid on a short intra-bank hop.
  //
  // WHY THE GATED VALID IS STAGED PER INSTANCE (not the raw snp_valid): the
  // write-enable cone e_snp_valid = snp_valid && sel is THE net class this
  // bank's history is about — D-033 / the 2026-08-10 kill-shot proved
  // synthesis re-derives and const-folds exactly this conjunction per cone
  // even under keep. Registering it closes both problems at once: the
  // eng_sel->WE route also terminates at a flop (the same tier-2 path class,
  // one LUT earlier), and replication can only COPY a flop (same D net),
  // never re-derive its function — the sledgehammer upgrade of the
  // keep-chain below. The beat<->engine pairing is decided at the gate,
  // PRE-stage, from the identical cycle relationship the tile delivers
  // today (snoop bundle +2, kv_eng_sel_q +1), then carried latched — a
  // select move can no longer re-pair an in-flight beat even in principle.
  //
  // ALIGNMENT (the same argument the tile q and q2 stages already carry,
  // third application): the whole consumed bundle {valid, data, last, addr}
  // shifts ONE more cycle TOGETHER, and the units are UNMODIFIED — every
  // internal use (amax reduce, beat counter, sc_commit, sc_val decode,
  // err_frame) sees the staged copies together BY CONSTRUCTION. Commits
  // land one more cycle later and are consumed hundreds of cycles later
  // (the walker polls STATUS over AXIL between store and composite reads).
  //
  // snp_flush is DELIBERATELY NOT staged: D-020 is a broadcast tile-wide
  // abort and stays a direct hit, exactly as at tile level, where the
  // 1-cycle CSR pulse already leads the twice-staged beats — the flush can
  // only arrive EARLIER relative to a staged beat, i.e. only ever drop
  // MORE of the record being aborted, and the D-020 choreography (host
  // stops streaming, then writes CTRL over AXIL, many cycles later) keeps
  // beats and the pulse far apart regardless. Same skew class, +1.
  //
  // 2026-08-14 (A0 campaign, CLOCK_LADDER §9 addendum 4 lineage): this
  // ingress stage moved the A0 critical path INSIDE the fence (WNS −11.51,
  // snp_data_q_reg -> u_comp sc_mem BRAM DIN — the write CONE, not the
  // hop). The cone is now pipelined INSIDE seq_walker_comp itself (the
  // W1/W2/W3 write-port pipeline there): commits land 3 more cycles after
  // this stage delivers the last beat, sc_val staying decode-coupled to
  // the RAM write edge. Total staging vs the pre-A0 design: tile q + q2
  // (+2), this ingress (+1), the unit's write pipe (+3) — the alignment
  // argument is unchanged (the whole bundle shifts together, the units
  // consume their own staged copies), and the commit->request distance
  // argument above absorbs +3 with ≥ ~70 cycles of structural slack.
  (* dont_touch = "true" *)
  logic                  e_snp_valid_q[N_ENG];
  (* dont_touch = "true" *)
  logic [15:0]           snp_data_q;
  (* dont_touch = "true" *)
  logic                  snp_last_q;
  (* dont_touch = "true" *)
  logic [WALK_SC_AW-1:0] snp_addr_q;
  always_ff @(posedge clk) begin
    for (int unsigned i = 0; i < N_ENG; i++) begin
      if (!rst_n) e_snp_valid_q[i] <= 1'b0;
      else        e_snp_valid_q[i] <= e_snp_valid[i];
    end
    snp_data_q <= snp_data;
    snp_last_q <= snp_last;
    snp_addr_q <= snp_addr;
  end

  // output muxes (eng_sel is quasi-static; the unit registers everything)
  assign req_ready = e_req_ready[idx];
  assign res_valid = e_res_valid[idx];
  assign res_data  = e_res_data [idx];

  // OR-reductions over the instance array (loop form: N_ENG is a parameter)
  always_comb begin
    err_frame = 1'b0;
    err_stale = 1'b0;
    for (int unsigned i = 0; i < N_ENG; i++) begin
      err_frame |= e_err_frame[i];
      err_stale |= e_err_stale[i];
      dbg_stale_eng[i] = e_err_stale[i];
    end
    dbg_stale_sc = e_dbg_sc[idx];
    dbg_stale_sq = e_dbg_sq[idx];
  end

  generate
    for (genvar gi = 0; gi < int'(N_ENG); gi++) begin : g_comp
      seq_walker_comp #(.CFG_D(CFG_D)) u_comp (
        .clk       (clk),
        .rst_n     (rst_n),
        .snp_valid (e_snp_valid_q[gi]),  // A0 ingress stage (2026-08-13)
        .snp_data  (snp_data_q),
        .snp_last  (snp_last_q),
        .snp_addr  (snp_addr_q),
        .snp_flush (snp_flush),          // broadcast: tile-wide abort
        .sq_valid  (e_sq_valid[gi]),
        .sq_data   (sq_data),
        .req_valid (e_req_valid[gi]),
        .req_ready (e_req_ready[gi]),
        .req_is_qs (req_is_qs),
        .req_idx   (req_idx),
        .res_valid (e_res_valid[gi]),
        .res_ready (e_res_ready[gi]),
        .res_data  (e_res_data[gi]),
        .err_frame (e_err_frame[gi]),
        .err_stale (e_err_stale[gi]),
        .dbg_stale_sc (e_dbg_sc[gi]),
        .dbg_stale_sq (e_dbg_sq[gi])
      );
    end
  endgenerate

endmodule

`default_nettype wire
