// seq_walker_pkg.sv — B1 layer-walker local types (apex_pkg stays FROZEN).
//
// Implements: docs/design/B1_WALKER.md §3 (layer descriptor + WALK CSR window
//             at 0x5C-0x6C), §A-1 (CQ-8-only tier scope and its refusal gate),
//             §A-2 (scale source), §B-1 (tap mirror). Mirrors the D-024/S12
//             principle "runtime-selectable, structure frozen".
//             IB-WALK stage 1 (docs/design/IB_WALK.md §2, D-029 per
//             LEVEL_C_INTEGRATION.md §9.1 R2): descriptor VERSIONING — the
//             fmt nibble GEOM0[31:28] + the v1.1 hardening that REFUSES an
//             unknown format (before it, a future-format descriptor would
//             have been silently mis-walked as v1) — and the fmt=1 layer-
//             descriptor wire format (walk_desc2_t + word map + fetch-record
//             widths per §9.1 R1). fmt=1 is DECLARED here at stage 1; no RTL
//             consumes it until stage 2. Python mirror (layout single-source
//             cross-check): verif/seq_walker/seq_walker_fmt.py.
//
// WHY A SEPARATE PACKAGE: rtl/apex_pkg.sv is the frozen cross-block contract
// (APEX_VERSION 0x0001_0000, apex_pkg.sv:5-7 "Any change here is a contract
// change: bump APEX_VERSION and update the golden mirrors in the same
// commit"). B1 needs new types but no new cross-block contract, so they live
// here and apex_pkg is not touched — no version bump, no golden mirror churn,
// no collision with the B2/B3 lanes.
//
// SCOPE (§A-1, owner decision 2026-07-21): walker v1 walks the score+pv
// phases at tier KVQ_CQ8 ONLY. A descriptor naming any other tier must be
// REFUSED (WALK_ERR_TIER) — never walked with a relaxed equivalence. The
// grouped-tier commit-time amax pass is stage B1b.
`ifndef SEQ_WALKER_PKG_SV
`define SEQ_WALKER_PKG_SV

package seq_walker_pkg;

  import apex_pkg::*;
  // D-029 erratum (I-B): the walker EMITS mxe descriptors, so its job
  // decomposition is a function of the tile's IMPLEMENTED capacity, not of
  // the frozen field widths. M_TILE_MAX is imported (selectively, so no
  // wildcard can shadow apex_pkg) rather than restated — a restated copy is
  // exactly the kind of drift this erratum exists to remove. MXE_N and K_MAX
  // already arrive with apex_pkg. Every flist that compiles this package
  // already lists rtl/mxe/mxe_cfg_pkg.sv ahead of it.
  import mxe_cfg_pkg::M_TILE_MAX;

  // ── envelope (F-1: apex_top T_ROW_MAX = 128; D is a BUILD parameter) ───────
  localparam int unsigned WALK_T_MAX   = 128;              // rows per attention step
  localparam int unsigned WALK_D_MAX   = 128;              // CFG_D in {64,128}
  localparam int unsigned WALK_T_W     = $clog2(WALK_T_MAX + 1);   // 8
  localparam int unsigned WALK_D_W     = $clog2(WALK_D_MAX + 1);   // 8
  // 2*T scale-cache entries: K records at [0,T), V records at [T,2T)
  localparam int unsigned WALK_SC_ENT  = 2 * WALK_T_MAX;   // 256
  localparam int unsigned WALK_SC_AW   = $clog2(WALK_SC_ENT);       // 8

  // S-5: the score composite's 2^SCORE_FRAC factor. Pinned to apex_pkg's
  // ASU_IN_FRAC so a change to the verified f10 ASU config cannot silently
  // desynchronise the composite unit from the golden (attention.py:75
  // "SCORE_FRAC = 10 # S-5: matches ASU_IN_FRAC").
  localparam int unsigned WALK_SCORE_FRAC = ASU_IN_FRAC;   // 10
  // S-4 fold exponent: qs = s_v * 2^-OUT_FRAC (attention.py:76).
  localparam int unsigned WALK_OUT_FRAC   = ASU_OUT_FRAC;  // 15

  // ── descriptor format ids (D-029 versioning, IB_WALK.md §2.1) ─────────────
  // GEOM0[31:28] declares the format. Discovery is read-side: the glue
  // publishes a support mask in WALK_STATUS[15:12] (FMT_SUP; landed D-028
  // glue reads 0 there, which the host must decode as "fmt 0 only"). The
  // host never loads fmt=N unless FMT_SUP[N] reads 1.
  localparam logic [3:0] WALK_FMT_V1    = 4'd0;   // D-028 3-word, score+pv
  localparam logic [3:0] WALK_FMT_LAYER = 4'd1;   // fmt=1 full-layer (D-029)
  // FMT_SUP values (glue consumes at its stage-4 edit; declared here so the
  // encoding has one source): bit n set ⇔ fmt n accepted.
  localparam logic [3:0] WALK_FMT_SUP_V1    = 4'b0001;
  localparam logic [3:0] WALK_FMT_SUP_LAYER = 4'b0011;

  // ── WALK CSR window (tile glue, 0x5C-0x6C; csr_regs decodes only to 0x54
  //    and ERR_STICKY owns 0x58 — B1_WALKER.md §3) ──────────────────────────
  localparam logic [7:0] WALK_CTRL_ADDR   = 8'h5C;
  localparam logic [7:0] WALK_DPTR_ADDR   = 8'h60;
  localparam logic [7:0] WALK_DDATA_ADDR  = 8'h64;
  localparam logic [7:0] WALK_STATUS_ADDR = 8'h68;
  // WALK_RQ: the ONE descriptor field that genuinely changes every decode step
  // (the PV requant pair is per-step calibration, §2 correction 1). Giving it a
  // direct address instead of routing it through the DPTR/DDATA window is what
  // makes the steady-state cost 3 MMIO/step rather than 4 — the OPTIMIZATION.md:63
  // goal metric. Everything else in the descriptor is per-configuration.
  localparam logic [7:0] WALK_RQ_ADDR     = 8'h6C;
  // WALK_DBG (2026-08-06, the walked-attention differential): apex_top folds
  // BOTH composite-unit faults into WALK_ERR_SEQ (err_sticky[15] is pinned
  // reserved-0 by every L3 ESTK expectation), so the sticky alone cannot say
  // WHICH fired. This read-only register splits them: [0]=err_frame sticky
  // (tlast at the wrong beat), [1]=err_stale sticky (unwritten entry /
  // missing s_q). Cleared by the same WALK_STATUS[8] W1C as the walker
  // sticky. New address, no existing image or descriptor byte changes.
  localparam logic [7:0] WALK_DBG_ADDR    = 8'h98;

  // ── error codes (WALK_STATUS[11:9]) ───────────────────────────────────────
  // A refusal must say WHY: an unsupported configuration is not a fault.
  typedef enum logic [2:0] {
    WALK_ERR_NONE  = 3'd0,
    WALK_ERR_TIER  = 3'd1,   // §A-1: tier != KVQ_CQ8 — refused, host mode kept
    WALK_ERR_DESC  = 3'd2,   // descriptor field out of range (D/T/geometry)
    WALK_ERR_SEQ   = 3'd3,   // internal sequencing fault (SVA-guarded)
    WALK_ERR_ABORT = 3'd4    // terminated by CTRL.soft_reset mid-walk
  } walk_err_e;

  // ── walk phases (WALK_STATUS[7:1]) ────────────────────────────────────────
  // v1 walks SCORE and PV only. The other names exist so that walking the full
  // layer later is a descriptor extension, not a rewrite (B1_WALKER.md §1).
  typedef enum logic [2:0] {
    WPH_IDLE  = 3'd0,
    WPH_SCORE = 3'd1,
    WPH_PV    = 3'd2,
    WPH_DONE  = 3'd3
  } walk_phase_e;

  // ── layer descriptor ──────────────────────────────────────────────────────
  // Loaded ONCE through WALK_DPTR/WALK_DDATA, then a walk costs WALK_GO + a
  // WALK_STATUS done-poll (the ≤3 MMIO/step goal metric, OPTIMIZATION.md:63).
  //
  // rq_scale/rq_shift are LOADED, not derived: the PV descriptor's requant
  // pair comes from calib_requant(amax|acc_o|) where acc_o is the output of
  // the very GEMM the descriptor configures (attention.py:395-398) — causally
  // circular, so no on-tile unit can produce it in one pass. Naming it a
  // descriptor field is the honest encoding of that limit: the walker is
  // autonomous in SEQUENCING and carries exactly one per-step calibration
  // input it cannot derive. See B1_WALKER.md §2 correction 1.
  typedef struct packed {
    logic [3:0]          fmt;        // GEOM0[31:28] — D-029 versioning; v1 = 0
    logic [WALK_D_W-1:0] d_dim;      // D (must equal the build's CFG_D)
    logic [WALK_T_W-1:0] t_rows;     // T attention rows, 1..WALK_T_MAX
    kvq_tier_e           tier;       // §A-1: only KVQ_CQ8 may walk
    logic [3:0]          outlier_k;  // informational; CQ-4+ only, refused in v1
    logic [15:0]         rq_scale;   // PV requant, host-loaded (see above)
    logic [4:0]          rq_shift;   // PV requant, host-loaded
    logic                en_score;   // phase-enable mask
    logic                en_pv;
  } walk_desc_t;

  // Descriptor SRAM word map (WALK_DDATA auto-increments WALK_DPTR).
  localparam int unsigned WALK_DESC_WORDS = 3;
  localparam int unsigned WALK_DW_GEOM    = 0;   // {outlier_k, tier, T, D}
  localparam int unsigned WALK_DW_RQ      = 1;   // {rq_shift, rq_scale}
  localparam int unsigned WALK_DW_MASK    = 2;   // {en_pv, en_score}

  // Unpack a loaded descriptor word set. Pure function so the walker RTL and
  // its TB share one definition of the wire format.
  function automatic walk_desc_t walk_desc_unpack(
      input logic [31:0] w_geom, input logic [31:0] w_rq,
      input logic [31:0] w_mask);
    walk_desc_t d;
    logic       unused_hid;
    begin
      // reserved word bits: declared so the wire format can grow without a
      // layout change, deliberately not consumed yet. GEOM0[31:28] left this
      // set at stage 1 of IB-WALK: it is the fmt nibble (D-029).
      unused_hid  = &{1'b0, w_geom[27:24], w_geom[19:18], w_rq[31:21],
                      w_mask[31:2]};
      d.fmt       = w_geom[31:28];
      d.d_dim     = w_geom[WALK_D_W-1:0];
      d.t_rows    = w_geom[8 +: WALK_T_W];
      d.tier      = kvq_tier_e'(w_geom[17:16]);
      d.outlier_k = w_geom[23:20];
      d.rq_scale  = w_rq[15:0];
      d.rq_shift  = w_rq[20:16];
      d.en_score  = w_mask[0];
      d.en_pv     = w_mask[1];
      walk_desc_unpack = d;
    end
  endfunction

  // Descriptor legality (§3 style: checked BEFORE any state change; a
  // violation raises the error pulse + sticky and starts no walk).
  // cfg_d is the BUILD's CFG_D — a runtime D that disagrees with the compiled
  // datapath is WALK_ERR_DESC, never a silently mis-shaped walk.
  function automatic walk_err_e walk_desc_check(input walk_desc_t d,
                                                input int unsigned cfg_d);
    logic unused_hid;
    begin
      // legality is a function of geometry + tier only; the requant pair and
      // outlier_k are carried, not checked, here
      unused_hid      = &{1'b0, d.outlier_k, d.rq_scale, d.rq_shift};
      walk_desc_check = WALK_ERR_NONE;
      // v1.1 hardening (IB_WALK.md §2.1 item 3, D-029): an unknown format is
      // REFUSED before anything else — without this, a fmt!=0 descriptor's
      // bits would be silently misread as v1 fields and mis-walked. Checked
      // FIRST because a v2 descriptor's bits at the tier position mean
      // something else entirely; the honest code is DESC (bad descriptor),
      // never TIER.
      if (d.fmt != WALK_FMT_V1)                    walk_desc_check = WALK_ERR_DESC;
      else if (d.tier != KVQ_CQ8)                  walk_desc_check = WALK_ERR_TIER;
      else if (d.t_rows == '0
               || int'({24'b0, d.t_rows}) > int'(WALK_T_MAX))
                                                   walk_desc_check = WALK_ERR_DESC;
      else if (int'({24'b0, d.d_dim}) != int'(cfg_d))
                                                   walk_desc_check = WALK_ERR_DESC;
      else if (!d.en_score && !d.en_pv)            walk_desc_check = WALK_ERR_DESC;
    end
  endfunction

  // ════════════════════════════════════════════════════════════════════════
  // fmt=1 — FULL-LAYER descriptor wire format (D-029; IB_WALK.md §2.2-§2.3,
  // LEVEL_C_INTEGRATION.md §9.1 R1/R3). DECLARATIONS ONLY at stage 1: the
  // walker RTL consumes none of this until IB-WALK stage 2, and the glue
  // none until stage 4. The Python mirror seq_walker_fmt.py asserts these
  // layouts bit-for-bit; change them ONLY in both places in one commit.
  // ════════════════════════════════════════════════════════════════════════

  // ── envelope ──────────────────────────────────────────────────────────────
  localparam int unsigned WALK2_H_MAX      = 30;   // query heads (RQ table cap)
  localparam int unsigned WALK2_DESC_WORDS = 64;   // descriptor SRAM, fmt=1
  localparam int unsigned WALK2_DPTR_W     = 6;    // glue widens DPTR 2 -> 6 b

  // ── descriptor SRAM word map (IB_WALK.md §2.2) ────────────────────────────
  localparam int unsigned W2_GEOM0   = 0;   // fmt/outlier_k/tier/head_dim
  localparam int unsigned W2_MODEL0  = 1;   // {d_ffn[31:16], d_model[15:0]}
  localparam int unsigned W2_MODEL1  = 2;   // {H_kv[23:16], H[15:8], kv_map[1:0]}
  localparam int unsigned W2_MASK    = 3;   // step-enable bits [11:0]
  localparam int unsigned W2_WSCALE0 = 4;   // 7 fp32 per-tensor weight scales
  localparam int unsigned W2_TENS0   = 11;  // 10 DDR tensor-base words
  localparam int unsigned W2_STEP    = 21;  // {t_rows[15:8], pos_m[7:0]} per-step
  localparam int unsigned W2_RQ0     = 22;  // 32 requant-slot words, per-step
  localparam int unsigned WALK2_RQ_SLOTS = 32;  // [0..H-1] PV, [H] oproj, [H+1] down

  // ── JC table (stage 5; IB_LAYER.md §3b JOBC): per-step fp16-GRADE
  //    positive-normal f32 composites for the LAYER deq/swiglu jobs —
  //    host-loaded like the RQ pairs (same causal/precision logic; the
  //    canonical grade-narrowing reconciles against D-030 at combine) ──────
  localparam int unsigned W2_JC0          = 54;
  localparam int unsigned WALK2_JC_SLOTS  = 4;
  localparam int unsigned WALK2_JC_OPROJ  = 0;  // deq comp_a after o-proj
  localparam int unsigned WALK2_JC_DOWN   = 1;  // deq comp_a after down-proj
  localparam int unsigned WALK2_JC_GATE   = 2;  // swiglu comp_a (a=gate)
  localparam int unsigned WALK2_JC_UP     = 3;  // swiglu comp_b (b=up)

  // ── LAYER level word (walker-side image of the §3b LAYER_CTRL levels,
  //    l_kv_map baked 0 per the R3 carve; bit positions == the register's) ──
  //    [0] rope_en · [1] rope_bank · [3:2] ser_dst · [5:4] fsrc_ext[1:0] ·
  //    [6] resid_arm · [7] fsrc_ext[2] · [14:8] rope_pos · [19] nsrc
  //
  // E-3 (E2E_TOY_LANE.md §4, 2026-08-01) — fsrc_ext WIDENED 2 -> 3 BITS.
  // E-1 landed the residual's internal egress as feeder input source code 4
  // and put its high bit at LAYER_CTRL[7] (reserved-0 before), but the
  // WALKER-side level stayed 2 bits, so a walked program could not name
  // code 4 at all: apex_top zero-extended `{1'b0, wk_lw_fsrc_ext}`. This
  // function now carries the third bit AT THE REGISTER'S OWN POSITION —
  // bit 7, the slot the old `1'b0` filler occupied — so every legacy level
  // word (fsrc_ext <= 3, high bit 0) encodes to exactly the same 15 bits it
  // always did. The Python mirror seq_walker_fmt.lctl asserts this layout.
  //
  // E-4b (E2E_TOY_LANE.md §4, 2026-08-04) — the word GROWS 15 -> 20 BITS to
  // carry l_nsrc, again AT THE REGISTER'S OWN POSITION: LAYER_CTRL[19], the
  // E-2a norm-input-source level. E-4's measured finding (STEP_MATRIX.md):
  // l_nsrc was HOST-held across a walk (the l_bias_en idiom), so ONE kick
  // could not both stage q (feeder codes -> act stage, nsrc=0) and run
  // NFEED (feeder codes -> norm, nsrc=1) — walk_e4 needed two kicks with a
  // host CSR write between them. Bits [18:15] are the register's l_kv_map
  // [17:15] / l_bias_en [18] slots and stay reserved-0 here (kv_map rides
  // the walker's own kv_eng_sel port per R3; l_bias_en has no walker level
  // port — D-029 surface). Every legacy level word (nsrc 0) zero-extends to
  // exactly the 15 bits it always encoded, and %-4-minimum hex prints grow
  // only when nsrc is set, so legacy LCTL trace lines are byte-identical.
  localparam int unsigned WALK2_LCTL_W = 20;
  function automatic logic [WALK2_LCTL_W-1:0] walk2_lctl(
      input logic rope_en, input logic rope_bank, input logic [1:0] ser_dst,
      input logic [2:0] fsrc_ext, input logic resid_arm,
      input logic [6:0] rope_pos, input logic nsrc);
    begin
      walk2_lctl = {nsrc, 4'b0, rope_pos, fsrc_ext[2], resid_arm,
                    fsrc_ext[1:0], ser_dst, rope_bank, rope_en};
    end
  endfunction

  // feeder input source codes the walker names (apex_top's l_fsrc_ext mux)
  localparam logic [2:0] WALK2_FSRC_LEGACY = 3'd0;  // widen/KVQ read bus
  localparam logic [2:0] WALK2_FSRC_SWGP   = 3'd2;  // swiglu-p (down-proj)
  localparam logic [2:0] WALK2_FSRC_RESID  = 3'd4;  // E-1 residual EGRESS

  // LAYER_JOB unit ids (§3b 0x7C [13:12])
  localparam logic [1:0] WALK2_LU_DEQ    = 2'd0;
  localparam logic [1:0] WALK2_LU_SWIGLU = 2'd1;
  localparam logic [1:0] WALK2_LU_RESID  = 2'd2;
  // E-2b: unit 3 = the NORM/EGRESS job (residual row -> C-1 feeder -> the
  // norm's x port). apex_top REFUSES this push unless l_fsrc_ext is already
  // on code 4 and cols frames whole feeder rows (err_code 9, NFEED_ROUTE),
  // which is why the walker's NFEED step arms the level BEFORE it pushes.
  localparam logic [1:0] WALK2_LU_NORM   = 2'd3;

  // step-enable bit positions in W2_MASK (partial-layer bring-up is
  // first-class; the cross-step dependency matrix lands with the stage-2 ROM)
  localparam int unsigned W2_EN_NORM1   = 0;
  localparam int unsigned W2_EN_QKV     = 1;
  localparam int unsigned W2_EN_ROPE    = 2;
  localparam int unsigned W2_EN_STOREKV = 3;
  localparam int unsigned W2_EN_SCORE   = 4;
  localparam int unsigned W2_EN_PV      = 5;
  localparam int unsigned W2_EN_OPROJ   = 6;
  localparam int unsigned W2_EN_RES1    = 7;
  localparam int unsigned W2_EN_NORM2   = 8;
  localparam int unsigned W2_EN_FFN     = 9;
  localparam int unsigned W2_EN_RES2    = 10;
  // E-3: the NORM-FEED step (residual EGRESS -> C-1 feeder -> RMSNorm, all
  // in-tile). Bit 11 was the LAST free bit of the 12-bit mask field and was
  // reserved-0 in every emitted image (seq_walker_fmt.EN_ALL = (1<<11)-1),
  // so a legacy descriptor walks BYTE-IDENTICALLY: the step is skipped in
  // the one cycle S2_STEP spends on any disabled pc.
  localparam int unsigned W2_EN_NFEED   = 11;
  // E-3b (E2E_TOY_LANE.md §4, 2026-08-04): the Q-STAGING step — the walker
  // stages the rotated-q rows through the gap-A q_sink ITSELF (per head:
  // feeder job, MODE_F16 squant job + composite sideband, serializer job,
  // the k2-injection GEMM stream, act-bank-1 LOAD at row h) instead of the
  // host doing all of that before WALK_GO. The mask field GROWS 12 -> 13
  // bits for it: W2_MASK[12] was reserved (walk_desc2_check refused it
  // nonzero), so every legacy image is byte-identical, and a NEW image on
  // OLD RTL is refused loudly (WALK_ERR_DESC via the old resv clause) —
  // the E-3a reserved-bit discipline, one bit further out.
  localparam int unsigned W2_EN_QSTAGE  = 12;
  // E-4b (E2E_TOY_LANE.md §4, 2026-08-04): the NSRC ownership bit — the
  // walker OWNS the l_nsrc level (LAYER_CTRL[19]) for this walk: it drives
  // lw_nsrc 0 through its LVL/QSTAGE arms and 1 at the NFEED entry, and
  // apex_top's walk-mode mux copies the net into l_nsrc_q only while the
  // walker's lw_nsrc_own level is high. The mask field GROWS 13 -> 14 bits
  // for it: W2_MASK[13] was reserved (walk_desc2_check refused it nonzero),
  // so every legacy image is byte-identical — l_nsrc keeps its HOST-held
  // hold semantics there — and a NEW image on OLD RTL is refused loudly
  // (WALK_ERR_DESC via the old resv clause). The E-3a/E-3b reserved-bit
  // discipline, one bit further out. Cross-bit fences (an NSRC image must
  // name NFEED; a QSTAGE+NFEED image must name NSRC — the host cannot hold
  // one value that serves both) live at seq_layer_walker2's S2_CHECK, the
  // E-3b fence house: refuse, never degrade (§A-1).
  localparam int unsigned W2_EN_NSRC    = 13;
  // IB-FUEL x WALK convergence (E-5, 2026-08-04): the FUEL-FED PROJECTION
  // bit — the walked projection steps take their WEIGHT bytes from the
  // fuel reader's DDR stream instead of starving. Three measured blockers
  // fall together (STEP_MATRIX.md "what still blocks a fully-walked FULL
  // layer" items 1-3), and all three are gated on THIS bit so a legacy
  // image is byte-identical:
  //   (a) the walk-mode route pin rt_wgt_src=1 (seq_layer_walker2:792 via
  //       rt_f1_q[3], which tracks the wrapped v1 engine's h_rt_wsrc, and
  //       seq_layer_walker.sv:305 pins THAT to 1'b1) made the external xw
  //       stream — the fuel reader's only delivery port (apex_top.sv:2273
  //       mxe_wgt_valid mux) — unreachable from a walk. RT_FPROJ below is
  //       the per-step route override, the E-3b RT_QSTAGE template.
  //   (b) S2_PJOB pushed ds descriptors ONLY, so the MXE's activation
  //       stream (apex_top.sv:2215, the act stage buffer's em port) never
  //       got a beat. The FPROJ walk emits the PAT_ROW act family itself.
  //   (c) S2_FETCH parks forever when fuel mode is off (apex_fuel_ctl.sv:195
  //       wk_req_ready is src_q-gated) and D-020 forbids aborting out of a
  //       presented valid. The fence is therefore a PRE-ENTRY refusal at
  //       S2_CHECK, not an abort path — see the walker's fp_fence note.
  // The mask field GROWS 14 -> 15 bits: W2_MASK[14] was reserved (this
  // check refused it nonzero), so every legacy image is byte-identical and
  // a NEW image on OLD RTL is refused loudly (WALK_ERR_DESC, the old resv
  // clause) — the E-3a/E-3b/E-4b reserved-bit discipline, one bit further.
  localparam int unsigned W2_EN_FPROJ   = 14;
  // E-7 (CONVERGENCE session, 2026-08-05): the FETCHED-GAMMA bit — a NORM
  // step of an EN_FPROJ walk takes its gamma from the walker's OWN fuel
  // record (WALK2_TENS_G1/G2, laid in the DDR image since IB-FUEL but never
  // driven) instead of the host xg stream. The measured blocker it lifts
  // (STEP_MATRIX.md p_norm2 class, re-verified by execution 2026-08-05):
  // NO xw -> xg ROUTE EXISTED — apex_top's xw port had exactly one consumer
  // (the MXE weight mux, apex_top.sv:2279-2282) and asu_rmsnorm's g port
  // exactly one producer (the top-level xg_* stream), so a fetched gamma
  // dead-ended in the fuel FIFO ahead of the next step's weight beats —
  // stream poisoning, which is why NORM1/NORM2 were REFUSED under FPROJ.
  // This bit arms the walker's gamma WINDOW level (lw_gsrc): apex_top
  // steers xw beats into the new gamma unpacker (4 x int16 LE per beat,
  // make_weight_image.gamma_payload's exact layout) feeding the norm's g
  // port for exactly the fetch's d_model elements. The mask field GROWS
  // 15 -> 16 bits: W2_MASK[15] was reserved (this check refused it
  // nonzero), so every legacy image is byte-identical and a NEW image on
  // OLD RTL is refused loudly — the E-3a..E-5 reserved-bit discipline.
  // Cross-bit fences (seq_layer_walker2 S2_CHECK, §A-1): FGAM requires
  // FPROJ (a gamma fetch without fuel is the S2_FETCH park wedge) and at
  // least one NORM step (a window with no consumer); NORM1/NORM2 under
  // FPROJ still require FGAM (the poisoning refusal, kept).
  localparam int unsigned W2_EN_FGAM    = 15;
  // E-7b (same session): the DOWN-SEPARATION bit — the down projection's
  // pcs (PC_LVLD/PC_UDOWN/PC_WDF/PC_WDJ) were gated on W2_EN_FFN alone
  // ("mask-inseparable from FFN", E6_ON_SILICON.md), so the DOWN epilogue
  // — the SAME pc_hasrq class E-6 proved on OPROJ — could not be walked
  // without dragging in the FFN steps that asu_swiglu's phase alternation
  // starves. This bit enables exactly those four pcs WITHOUT FFN, paired
  // with RES2 (the OPROJ<->RES1 rule, same starvation argument). The DOWN
  // act family (the swiglu-product codes, host-staged — the disclosed E-6
  // o8->act seam class) stacks ABOVE the OPROJ family in act bank 1 and
  // its k = d_ffn must fit ONE k-split (walk2_k_job(FEED_DM)) and the
  // stage bank — at the 0.5B geometry d_ffn = 4864 > 1984 so a 0.5B DOWN
  // walk is REFUSED with that measured reason (76 stage rows cannot be
  // resident in a 31-row bank; per-k-chunk re-staging has no in-tile
  // replay source — real datapath work, fenced not faked). The mask field
  // GROWS 16 -> 17 bits at the reserved bit, same discipline as above.
  localparam int unsigned W2_EN_DOWN    = 16;
  localparam int unsigned W2_EN_W      = 17;   // en_mask field width

  // E-3b: the Q composite slot — ONE per-step S-2 composite word (fp32,
  // positive-normal fp16-grade like the JC slots), consumed W2_QC-repeated
  // by the QSTAGE step's MODE_F16 sideband. ONE word is the honest shape of
  // the fmt=1 model: weight scales are PER-TENSOR (W2_WSCALE0's design), so
  // the q projection's S-2 composite is one value per step. Subjects whose
  // q rows need per-element composites (the TB-side inject scaffolding's
  // general case) are OUT of this step's scope by construction. Word 58 was
  // reserved-0; it is consumed only when W2_EN_QSTAGE is set.
  localparam int unsigned W2_QC         = 58;

  // ── DDR tensor-base table order (tag values; IB_WALK.md §2.2 W11-W20) ─────
  // One word per tensor: [29:0] base in 64-BYTE units (64 GB DIMM), [31:30]
  // resv. Per-job offsets/lengths are DERIVED from geometry + the
  // decomposition rule below, never stored.
  localparam int unsigned WALK2_TENS_WQ    = 0;
  localparam int unsigned WALK2_TENS_WK    = 1;
  localparam int unsigned WALK2_TENS_WV    = 2;
  localparam int unsigned WALK2_TENS_WO    = 3;
  localparam int unsigned WALK2_TENS_WG    = 4;
  localparam int unsigned WALK2_TENS_WU    = 5;
  localparam int unsigned WALK2_TENS_WD    = 6;
  localparam int unsigned WALK2_TENS_G1    = 7;
  localparam int unsigned WALK2_TENS_G2    = 8;
  localparam int unsigned WALK2_TENS_PHASE = 9;   // RoPE phase rows (C-ROPE:
                                                  // quantized ONCE from f64 —
                                                  // MUST be table-fed to stay
                                                  // bit-exact; row = pos_m)
  localparam int unsigned WALK2_TENS_N     = 10;

  // ── walker->reader fetch request (§9.1 R1 + IB-FUEL s1 addendum) ──────────
  // FUEL froze the full 64-bit record layout (their s1, comp/ib-fuel
  // bbe5064/d62b230): base_64B at [29:0], beats_64B at [55:30], tag at
  // [63:56] — base/beats in 64-BYTE WORD units. 26-bit beats because the 7B
  // down-proj tensor is 18944*3584 B = 1,060,864 beats > 2^20 (the stage-0
  // 20-bit proposal could not address it). Tag value = WALK2_TENS_* index in
  // [7:0] (8 bits are FUEL's; only [3:0] carry information today).
  // FUEL-verified induced invariant their make_ddr_image.py hard-errors on:
  // every committed job's lane-beat count divisible by 8, every row-granular
  // tensor row a 64 B multiple (holds for every fmt=1 shape: weight-block k
  // is always a multiple of 64, gamma/phase rows are 64 B multiples at both
  // head_dims). One record per weight-consuming step; a step consuming
  // several tensors (QKV) issues one per tensor in template order. FIFO >= 2
  // keeps the interface issue-ahead capable (prefetch stays a deferred FSM
  // change, IB_WALK.md §5).
  localparam int unsigned WALK2_FR_BASE_W  = 30;
  localparam int unsigned WALK2_FR_BEATS_W = 26;
  localparam int unsigned WALK2_FR_TAG_W   = 8;
  // Packed-struct field order is MSB-first, so {tag, beats64, base64} lands
  // each field at FUEL's frozen bit positions: tag [63:56], beats [55:30],
  // base [29:0]. Do not reorder.
  typedef struct packed {
    logic [WALK2_FR_TAG_W-1:0]   tag;
    logic [WALK2_FR_BEATS_W-1:0] beats64;
    logic [WALK2_FR_BASE_W-1:0]  base64;
  } walk2_freq_t;

  // ── GEMM job decomposition (single source; IB_WALK.md §2.6) ───────────────
  // n-split-major, k-split-minor; job (n0,k0) = {m, k<=walk2_k_job(D),
  // n<=WALK2_N_MXE, accumulate=(k0>0)} — C-KSPLIT's on-tile realization via
  // the frozen descriptor's accumulate bit (apex_pkg [66]). The pre-swizzled
  // DDR image concatenates the per-job weight blocks in exactly this order.
  //
  // ── D-029 ERRATUM (I-B): TWO BOUNDS, NOT ONE ─────────────────────────────
  // These used to be ONE constant sized by the DESCRIPTOR FIELD WIDTH, and
  // that overload WAS the defect: every walked 7B projection descriptor
  // (n_total = 3584 / 4608 / 18944 -> n_cur = 4095) was rejected by the tile.
  // The two consumers have genuinely different limits:
  //
  //   WALK2_N_MXE — an MXE JOB's n_dim. Bounded by the IMPLEMENTED TILE
  //     CAPACITY, not the field: mxe_cfg_pkg.sv's header states "N <= MXE_N
  //     (8) — one array width per descriptor (no N tiling v0.1)", and
  //     mxe_ctrl.sv's `legal` term `(desc.n_dim != '0) && (desc.n_dim <=
  //     DIM_W'(MXE_N))` raises desc_error and REFUSES the job above it. A
  //     descriptor field 12 bits wide does not make a 4095-wide job legal.
  //
  //   WALK2_N_JOB — a LAYER_JOB's `cols` field (IB_LAYER.md §3b, 0x7C
  //     [11:0]). SECOND ERRATUM (2026-07-30, R1): this paragraph used to
  //     assert that the field maximum was a correct chunk bound for every
  //     LAYER unit ("a pure field-width bound — those units stream columns").
  //     That rationale was WRONG, the same way the original overload was:
  //     the chunk bound is the CONSUMING UNIT's implemented capacity, never
  //     the wire's width. apex_top hands asu_swiglu only $clog2(COLS_MAX+1)
  //     = 7 of the 12 bits (#(.COLS_MAX(WALK2_N_LU_SWIGLU)) at u_swg), so a
  //     field-bound chunk of 2564 ARRIVED AS 4 and was silently accepted —
  //     4 columns computed of the 2564 asked, no job_error. deq's
  //     COLS_MAX (4095) happens to equal the field maximum, and residual
  //     refuses cols > DM_MAX inside the unit; swiglu is the one whose
  //     capacity is narrower than the field. Chunking therefore goes
  //     through walk2_lu_bound() below — per unit, mirroring
  //     verif/seq_walker/seq_walker_fmt.py's LU_CHUNK table.
  //
  //   WALK2_M_MXE — an MXE JOB's m_dim, same class as N_MXE: the K/V
  //     projections emit m = t_rows, and mxe_ctrl rejects m_dim >
  //     M_TILE_MAX. WALK_T_MAX (128) exceeds it, so the walk envelope is
  //     FENCED in seq_layer_walker2's S2_CHECK (§A-1: refuse, never degrade).
  //   WALK2_K_JOB — the MXE's OWN k ceiling (mxe_ctrl.sv:162 `desc.k_dim <=
  //     DIM_W'(K_MAX)`). THIRD ERRATUM (W1, 2026-08-05): this is a CEILING,
  //     not the chunk bound — see walk2_k_job() below. The k side of the
  //     same blind spot, one consumer further down the pipe.
  localparam int unsigned WALK2_K_JOB = K_MAX;            // 2048  — mxe k_dim
  localparam int unsigned WALK2_N_MXE = MXE_N;            // 8     — mxe n_dim
  // FFN interleave (FFN_INTERLEAVE.md, fence-2 closure): the gate/up/swiglu
  // chunk width. asu_swiglu consumes <=64 cols per gate/up frame pair, so
  // the walker's FFN segment loops in 64-column chunks; d_ffn must be a
  // multiple of this (S2_CHECK envelope — 0.5B d_ffn=4864 = 76 chunks).
  localparam int unsigned WALK2_FFN_CHUNK = 64;
  localparam int unsigned WALK2_M_MXE = M_TILE_MAX;       // 64    — mxe m_dim
  localparam int unsigned WALK2_N_JOB = (1 << DIM_W) - 1; // 4095  — LAYER cols
  // asu_swiglu's IMPLEMENTED width — the swiglu LAYER_JOB chunk bound AND
  // the u_swg COLS_MAX parameter in apex_top (which reads THIS constant, so
  // walker, refusal check and unit cannot drift apart). Mirrors
  // seq_walker_fmt.py N_LU_SWIGLU; the mutants5 gate holds the spec side.
  localparam int unsigned WALK2_N_LU_SWIGLU = 64;

  // ── THIRD ERRATUM (W1, 2026-08-05): K-PER-JOB IS A FUNCTION OF D ─────────
  // WALK2_K_JOB = K_MAX said "the MXE accepts k = 2048" — true
  // (mxe_ctrl.sv:162) — and stopped one consumer short. A projection job's
  // ACTIVATION reaches the MXE act port through the ACT STAGE BUFFER
  // (apex_top.sv:2203 `apex_stage_buf #(.D(CFG_DM), .R_MAX(STAGE_R_MAX))
  // u_astage`, emitted per apex_top.sv:2215), and a k-wide contraction row
  // occupies ceil(k / CFG_DM) stage rows. That unit accepts rows in
  // [1, R_MAX] ONLY (apex_stage_buf.sv:167 `legal`, LOAD base+rows bound at
  // :174-175), its rows field is 5 bits (:78), and R_MAX itself is capped
  // at 31 by the elaboration guard (:103-104) — so a k > 31*CFG_DM act
  // family cannot be staged on ANY legal build. At CFG_DM = 128 the cap
  // (3968) sits above K_MAX and is invisible; at CFG_DM = 64 it bites:
  // 2048/64 = 32 rows > 31, so every K_MAX-sized k-chunk of a D=64 walk —
  // the 0.5B down projection, k_total = 4864 — was TILE-ILLEGAL. The host
  // lane hit the same wall on silicon tooling and encoded the same min():
  // tile_geom.rows_per_desc_for = min(K_MAX // d, STAGE_R_MAX). This
  // function is that rule at the walker's frozen-decomposition seam;
  // verif/seq_walker/seq_walker_fmt.py's k_job() mirrors it exactly.
  //
  // The cap is the ARCHITECTURAL 31, NOT a per-build STAGE_ROWS parameter,
  // ON PURPOSE: jobs()/walk2 decomposition order IS the pre-swizzled DDR
  // image byte order (IB_WALK §2.6), so it must be a pure function of the
  // image geometry (row width), never of one build's instantiation. A build
  // instantiated below the cap still refuses loudly at its own consumers
  // (apex_stage_buf job_error; the FPROJ family fence fp_rows > STAGE_ROWS
  // at seq_layer_walker2 S2_CHECK).
  localparam int unsigned WALK2_STAGE_R_CAP = 31;
  function automatic int unsigned walk2_k_job(input int unsigned row_w);
    int unsigned rows;
    begin
      rows = (K_MAX / row_w < WALK2_STAGE_R_CAP) ? (K_MAX / row_w)
                                                 : WALK2_STAGE_R_CAP;
      // whole stage rows per chunk: 31*64 = 1984 @ D=64, 16*128 = 2048 @ 128
      walk2_k_job = rows * row_w;
    end
  endfunction

  // k-split count at a given per-job bound (callers pass walk2_k_job(D) —
  // the third erratum made the bound D-dependent, so it is an argument now;
  // no RTL consumed the old single-constant form)
  function automatic int unsigned walk2_ksplits(input int unsigned k_total,
                                                input int unsigned k_job);
    walk2_ksplits = (k_total + k_job - 1) / k_job;
  endfunction

  // MXE-job n splits (bounded by the array width)
  function automatic int unsigned walk2_nsplits(input int unsigned n_total);
    walk2_nsplits = (n_total + WALK2_N_MXE - 1) / WALK2_N_MXE;
  endfunction

  // LAYER_JOB cols chunk bound — PER UNIT (second erratum, R1): the
  // CONSUMING unit's capacity, mirroring seq_walker_fmt.py's LU_CHUNK
  // table {deq: field max, swiglu: WALK2_N_LU_SWIGLU, resid: field max}.
  // Deliberately separate functions from the MXE n-split so a future edit
  // cannot silently re-merge the rules.
  // NOTE (E-3): WALK2_LU_NORM is deliberately NOT a client of this function.
  // A norm/egress job names a WINDOW of the residual row (base+cols), and
  // the walker LU channel carries no base — so a chunked egress would
  // re-read the SAME window per chunk, silently feeding the norm the first
  // chunk N times. The NFEED step therefore pushes ONE job of d_model
  // columns and FENCES d_model > WALK2_N_JOB at S2_CHECK instead of
  // splitting it (§A-1: refuse, never degrade).
  function automatic int unsigned walk2_lu_bound(input logic [1:0] unit);
    walk2_lu_bound = (unit == WALK2_LU_SWIGLU) ? WALK2_N_LU_SWIGLU
                                               : WALK2_N_JOB;
  endfunction

  function automatic int unsigned walk2_lu_chunks(input logic [1:0] unit,
                                                  input int unsigned cols);
    walk2_lu_chunks = (cols + walk2_lu_bound(unit) - 1)
                      / walk2_lu_bound(unit);
  endfunction

  // ── walk step ids, fmt=1 (v2 STATUS phase reporting; the landed 3-bit
  //    walk_phase_e stays untouched — glue reconciliation is a stage-4 item)
  typedef enum logic [3:0] {
    WS2_IDLE    = 4'd0,
    WS2_NORM1   = 4'd1,
    WS2_QKV     = 4'd2,
    WS2_ROPE    = 4'd3,
    WS2_STOREKV = 4'd4,
    WS2_SCORE   = 4'd5,
    WS2_PV      = 4'd6,
    WS2_OPROJ   = 4'd7,
    WS2_RES1    = 4'd8,
    WS2_NORM2   = 4'd9,
    WS2_FFN_GU  = 4'd10,
    WS2_SILU    = 4'd11,
    WS2_FFN_DN  = 4'd12,
    WS2_RES2    = 4'd13,
    WS2_DONE    = 4'd14,
    WS2_ERR     = 4'd15
  } walk_step_e;

  // ── fmt=1 scalar descriptor (the SRAM-resident TENS/RQ/WSCALE tables are
  //    accessed by word index, not carried in the struct) ────────────────────
  typedef struct packed {
    logic [3:0]          fmt;         // GEOM0[31:28], must be WALK_FMT_LAYER
    logic [3:0]          outlier_k;   // GEOM0[23:20]
    kvq_tier_e           tier;        // GEOM0[17:16]
    logic [WALK_D_W-1:0] head_dim;    // GEOM0[7:0] — attention D = build CFG_D
    logic [15:0]         d_model;     // MODEL0[15:0]
    logic [15:0]         d_ffn;       // MODEL0[31:16]
    logic [7:0]          n_heads;     // MODEL1[15:8]
    logic [7:0]          n_kv_heads;  // MODEL1[23:16]
    logic [1:0]          kv_map;      // MODEL1[1:0] — §9.1 R3 provisional:
                                      // 2'b01 = per-KV-head engine banks;
                                      // 2'b00 = flat; others refused (final
                                      // encoding pinned by IB-LAYER)
    logic [W2_EN_W-1:0]  en_mask;     // MASK[16:0] (E-3b, E-4b, E-5, then
                                      // E-7/E-7b widened; [12]..[16] were
                                      // each reserved-0 before their step
                                      // landed, so legacy images are
                                      // byte-identical at every widening)
    logic [WALK_T_W-1:0] t_rows;      // STEP[15:8]
    logic [7:0]          pos_m;       // STEP[7:0] — self-inclusive q position
  } walk_desc2_t;

  function automatic walk_desc2_t walk_desc2_unpack(
      input logic [31:0] w_geom0, input logic [31:0] w_model0,
      input logic [31:0] w_model1, input logic [31:0] w_mask,
      input logic [31:0] w_step);
    walk_desc2_t d;
    logic        unused_hid;
    begin
      unused_hid   = &{1'b0, w_geom0[27:24], w_geom0[19:18], w_geom0[15:8],
                       w_model1[31:24], w_model1[7:2], w_mask[31:17],
                       w_step[31:16]};
      d.fmt        = w_geom0[31:28];
      d.outlier_k  = w_geom0[23:20];
      d.tier       = kvq_tier_e'(w_geom0[17:16]);
      d.head_dim   = w_geom0[WALK_D_W-1:0];
      d.d_model    = w_model0[15:0];
      d.d_ffn      = w_model0[31:16];
      d.n_heads    = w_model1[15:8];
      d.n_kv_heads = w_model1[23:16];
      d.kv_map     = w_model1[1:0];
      d.en_mask    = w_mask[W2_EN_W-1:0];
      d.t_rows     = w_step[8 +: WALK_T_W];
      d.pos_m      = w_step[7:0];
      walk_desc2_unpack = d;
    end
  endfunction

  // fmt=1 legality — RAW words in, so nonzero reserved bits are refusable
  // (IB_WALK.md §2.2: "resv bits load as written but MUST be 0"). Checked
  // BEFORE any state change; clause order is the error-priority contract and
  // the Python mirror replicates it exactly.
  function automatic walk_err_e walk_desc2_check(
      input logic [31:0] w_geom0, input logic [31:0] w_model0,
      input logic [31:0] w_model1, input logic [31:0] w_mask,
      input logic [31:0] w_step, input int unsigned cfg_d);
    walk_desc2_t d;
    logic        resv_nz;
    logic        unused_hid;
    int          nh, nk;
    begin
      d       = walk_desc2_unpack(w_geom0, w_model0, w_model1, w_mask, w_step);
      // carried, not checked: outlier_k is informational; d_ffn has no
      // legality constraint of its own (any 16-bit value walks — the FFN
      // split loops derive from it, and 0 simply means the FFN steps are
      // mask-disabled or degenerate)
      unused_hid = &{1'b0, d.outlier_k, d.d_ffn};
      // E-3b: W2_MASK[12] left this set — it is the EN_QSTAGE bit now, so
      // a legacy check (which had [31:12] here) refuses a QSTAGE image
      // loudly and a legacy image (bit 12 = 0) checks identically.
      // E-4b: W2_MASK[13] left it the same way — the EN_NSRC bit. The
      // pre-E-4b check ([31:13] here) refuses an NSRC image loudly on old
      // RTL; a legacy image (bit 13 = 0) checks identically.
      // E-5: W2_MASK[14], the EN_FPROJ bit, same discipline again — the
      // pre-E-5 check ([31:14] here) refuses a fuel-fed-projection image
      // loudly on old RTL; a legacy image (bit 14 = 0) checks identically.
      // E-7/E-7b: W2_MASK[15] (EN_FGAM) and [16] (EN_DOWN) left the slice
      // the same way — the pre-E-7 check ([31:15] here) refuses either
      // image loudly on old RTL; legacy images check identically.
      resv_nz = |{w_geom0[27:24], w_geom0[19:18], w_geom0[15:8],
                  w_model1[31:24], w_model1[7:2], w_mask[31:17],
                  w_step[31:16]};
      nh      = int'({24'b0, d.n_heads});
      nk      = int'({24'b0, d.n_kv_heads});
      walk_desc2_check = WALK_ERR_NONE;
      if (d.fmt != WALK_FMT_LAYER)                 walk_desc2_check = WALK_ERR_DESC;
      else if (resv_nz)                            walk_desc2_check = WALK_ERR_DESC;
      else if (d.tier != KVQ_CQ8)                  walk_desc2_check = WALK_ERR_TIER;
      else if (int'({24'b0, d.head_dim}) != int'(cfg_d))
                                                   walk_desc2_check = WALK_ERR_DESC;
      else if (d.t_rows == '0
               || int'({24'b0, d.t_rows}) > int'(WALK_T_MAX))
                                                   walk_desc2_check = WALK_ERR_DESC;
      else if (int'({24'b0, d.pos_m}) >= int'(WALK_T_MAX))
                                                   walk_desc2_check = WALK_ERR_DESC;
      else if (nh == 0 || nk == 0 || nh > int'(WALK2_H_MAX))
                                                   walk_desc2_check = WALK_ERR_DESC;
      // 2026-08-07 (A0 timing, a0v2 report): these two clauses were the
      // walker's whole -8.68 ns cone — the int' casts made Vivado build a
      // 32-stage divider and a 32-bit multiply for operands that are 8 bits
      // (n_heads/n_kv_heads, both bounded above by WALK2_H_MAX and checked
      // nonzero one clause earlier). Same width, same verdicts, 4x
      // shallower: divisibility via an 8/8 modulo, the geometry product as
      // 8x8 -> 16 against d_model. The python twin (seq_walker_fmt.check2)
      // computes the same arithmetic on unbounded ints — identical results
      // over the legal field ranges, no twin change.
      else if ((d.n_heads % d.n_kv_heads) != 8'd0) walk_desc2_check = WALK_ERR_DESC;
      else if ((16'(d.n_heads) * 16'({8'b0, d.head_dim}))
               != d.d_model)                       walk_desc2_check = WALK_ERR_DESC;
      else if (d.en_mask == '0)                    walk_desc2_check = WALK_ERR_DESC;
      else if (d.kv_map[1])                        walk_desc2_check = WALK_ERR_DESC;
    end
  endfunction

endpackage

`endif // SEQ_WALKER_PKG_SV
