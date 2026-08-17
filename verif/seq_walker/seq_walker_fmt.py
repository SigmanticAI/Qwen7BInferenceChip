#!/usr/bin/env python3
"""seq_walker_fmt.py — Python mirror of the fmt=1 (D-029) descriptor layout.

Single-source rule (rtl/seq/seq_walker_pkg.sv, fmt=1 section): the SV package
and this file define the SAME wire format; change them only together in one
commit. Consumers: gen_layer_trace.py (the layer op-stream spec generator)
and, on the IB-FUEL side, their pre-swizzler — both build against these
functions rather than re-deriving bit positions.

Contract sources: docs/design/IB_WALK.md §2.1-§2.6, LEVEL_C_INTEGRATION.md
§9.1 R1 (fuel_req widths; the full 64-bit record layout is FUEL's s1 freeze:
base_64B [29:0], beats_64B [55:30], tag [63:56], 64-BYTE word units) and R3
(kv_map provisional encoding).

Selftest: `python3 seq_walker_fmt.py` — layout round-trips, check-mirror
positives/negatives, decomposition-vs-golden-chunking identity, and the R1
beat-width justification recomputed from the 7B shape.
"""
from __future__ import annotations

import sys

# ── error codes (walk_err_e) ─────────────────────────────────────────────────
ERR_NONE, ERR_TIER, ERR_DESC, ERR_SEQ, ERR_ABORT = 0, 1, 2, 3, 4

# ── format ids (D-029) ───────────────────────────────────────────────────────
FMT_V1 = 0
FMT_LAYER = 1
FMT_SUP_V1 = 0b0001          # WALK_STATUS[15:12] on a v1-only tile (reads 0
                             # on the LANDED D-028 glue: decode 0 as v1-only)
FMT_SUP_LAYER = 0b0011

# ── envelope ─────────────────────────────────────────────────────────────────
T_MAX = 128
H_MAX = 30
DESC_WORDS = 64
RQ_SLOTS = 32

# ── descriptor SRAM word map ─────────────────────────────────────────────────
W_GEOM0 = 0
W_MODEL0 = 1
W_MODEL1 = 2
W_MASK = 3
W_WSCALE0 = 4                # 7 words, resv-0 until IB-LAYER claims them (Q1)
W_TENS0 = 11                 # 10 words
W_STEP = 21                  # per-step
W_RQ0 = 22                   # 32 words, per-step

# step-enable bits (W_MASK)
EN_NORM1, EN_QKV, EN_ROPE, EN_STOREKV, EN_SCORE, EN_PV = 0, 1, 2, 3, 4, 5
EN_OPROJ, EN_RES1, EN_NORM2, EN_FFN, EN_RES2 = 6, 7, 8, 9, 10
# E-3: the norm-feed step (residual EGRESS -> C-1 -> RMSNorm, in-tile). Bit
# 11 was the last reserved bit of the mask field, so EN_ALL is UNCHANGED and
# every previously-emitted descriptor image is byte-identical.
EN_NFEED = 11
EN_ALL = (1 << 11) - 1
EN_ALL_NFEED = EN_ALL | (1 << EN_NFEED)
# E-3b: the Q-STAGING step (walker stages the rotated-q rows through the
# gap-A q_sink itself). Bit 12 GROWS the mask field 12 -> 13 bits; it was
# reserved (check2 refused it nonzero), so every legacy image is
# byte-identical and a QSTAGE image on old RTL refuses loudly.
EN_QSTAGE = 12
# E-4b: the NSRC ownership bit — the walker OWNS the l_nsrc level
# (LAYER_CTRL[19]) for this walk, arming it 1 at the NFEED entry and 0 in
# its LVL/QSTAGE arms, which is what lets ONE kick both stage q (codes ->
# act) and run NFEED (codes -> norm). Bit 13 GROWS the mask field 13 -> 14
# bits; it was reserved (check2 refused it nonzero), so every legacy image
# is byte-identical — l_nsrc stays HOST-held there — and an NSRC image on
# old RTL refuses loudly. Cross-bit fences (NSRC requires NFEED;
# QSTAGE+NFEED requires NSRC) are S2_CHECK fences, not check2 clauses —
# the E-3b fence house.
EN_NSRC = 13
# E-5 (IB-FUEL x WALK convergence): the FUEL-FED PROJECTION bit — the walked
# projection steps take their WEIGHT bytes from the fuel reader's DDR stream
# (route override RT_FPROJ pins rt_wgt_src=0 onto the external xw port, which
# cl_apex muxes to the fuel FIFO) and the walker emits the MXE's activation
# family itself. Bit 14 GROWS the mask field 14 -> 15 bits; it was reserved
# (check2 refused it nonzero), so every legacy image is byte-identical and an
# FPROJ image on old RTL refuses loudly. The envelope fences (FPROJ implies a
# QKV-only mask; fuel armed; the act family tiles d_model into whole stage
# rows and fits one bank; one k-split) are S2_CHECK fences, not check2
# clauses — the E-3b/E-4b fence house.
EN_FPROJ = 14
# E-7 (convergence session, 2026-08-05): the FETCHED-GAMMA bit — a NORM step
# of an EN_FPROJ walk takes its gamma from the walker's own G1/G2 fuel
# record through the new xw -> gamma-unpack -> norm-g route (apex_gam_unpack)
# instead of the host xg stream. Bit 15 GROWS the mask field 15 -> 16 bits;
# it was reserved (check2 refused it nonzero), so every legacy image is
# byte-identical and an FGAM image on old RTL refuses loudly. Cross-bit
# fences (FGAM requires FPROJ + a NORM step; NORMx under FPROJ requires
# FGAM) are S2_CHECK fences, not check2 clauses — the fence house.
EN_FGAM = 15
# E-7b: the DOWN-SEPARATION bit — enables exactly the DOWN pcs
# (LVLD/UDOWN/WDF/WDJ) without EN_FFN, paired with EN_RES2, so the DOWN
# requant epilogue (the E-6 pc_hasrq class) can walk where its k = d_ffn
# fits one k-split. Bit 16 GROWS the mask field 16 -> 17 bits, same
# reserved-bit discipline. The 0.5B refusal (d_ffn = 4864 > k_job(64) =
# 1984 — a 76-row act family against the 31-row stage bank, no in-tile
# re-staging source) is an S2_CHECK fence with a measured reason.
EN_DOWN = 16
EN_W = 17

# ── DDR tensor table order (= fetch tag values) ──────────────────────────────
TENS_WQ, TENS_WK, TENS_WV, TENS_WO = 0, 1, 2, 3
TENS_WG, TENS_WU, TENS_WD = 4, 5, 6
TENS_G1, TENS_G2, TENS_PHASE = 7, 8, 9
TENS_N = 10
TENS_NAMES = ["Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd", "g1", "g2", "phase"]

# ── fuel_req record (FUEL s1 freeze; 64-byte word units) ─────────────────────
FR_BASE_W, FR_BEATS_W, FR_TAG_W = 30, 26, 8
FR_BASE_LSB, FR_BEATS_LSB, FR_TAG_LSB = 0, 30, 56

# ── GEMM job decomposition constants ─────────────────────────────────────────
# D-029 ERRATUM (I-B) — TWO BOUNDS, NOT ONE. Mirrors seq_walker_pkg's
# WALK2_K_JOB / WALK2_N_MXE / WALK2_M_MXE / WALK2_N_JOB exactly; the SV side
# carries the full rationale. Short form:
#   * an MXE JOB's n/m are bounded by the IMPLEMENTED TILE CAPACITY
#     (mxe_cfg_pkg: N <= MXE_N = 8, M <= M_TILE_MAX = 64; mxe_ctrl.sv's
#     `legal` term refuses anything wider with desc_error),
#   * a LAYER_JOB's `cols` is bounded by its 12-bit FIELD (IB_LAYER §3b 0x7C).
# Sizing the n-split by the field width was the defect: every 7B projection
# descriptor (n = 3584/4608/18944 -> 4095) was illegal at the tile.
# ── the CONSUMER's acceptance rule (mxe_ctrl.sv `legal`, M/K/N terms) ────────
# These are the TILE's numbers, not the walker's choice, and they are kept
# in their own names ON PURPOSE. The blind-spot check below reads only these;
# the decomposition below reads only the WALK2_* bounds. If the two sets were
# one set — as they effectively were before this erratum — the check would
# move with the bug and stay green, which is precisely what happened.
MXE_N_MAX = 8                # apex_pkg MXE_N
MXE_M_MAX = 64               # mxe_cfg_pkg M_TILE_MAX
MXE_K_MAX = 2048             # apex_pkg K_MAX

# ── the ACT STAGE BUFFER's acceptance rule (its OWN elaboration bounds) ──────
# THIRD ERRATUM (W1, 2026-08-05) — the k side of the same blind spot, one
# consumer further down the pipe. A projection job's ACTIVATION reaches the
# MXE act port through the act stage buffer (apex_top.sv:2203 `apex_stage_buf
# #(.D(CFG_DM), .R_MAX(STAGE_R_MAX)) u_astage`), and a k-wide contraction row
# occupies ceil(k / CFG_DM) stage rows. The unit accepts rows in [1, R_MAX]
# ONLY (apex_stage_buf.sv:167 `legal`; LOAD base+rows bound :174-175), the
# rows field is 5 bits (:78), and R_MAX is capped at 31 by the elaboration
# guard (:103-104) — so a k > 31*CFG_DM act family cannot be staged on ANY
# legal build (the flying builds all instantiate the cap: tb_apex_l3.sv:187,
# tb_l4_compose.sv:83, cl_apex.sv:260 CL_ROWS_MAX). At CFG_DM=128 the cap
# (3968) sits above K_MAX and was invisible; at CFG_DM=64 it bites: a
# K_MAX-sized chunk is 32 rows > 31, so every 2048-chunk of a D=64 walk —
# the 0.5B Wd, k_total=4864 — was TILE-ILLEGAL. These are the CONSUMER's
# numbers, read ONLY by assert_act_stageable(); the decomposition below reads
# only K_ROWS/k_job(). Same independence discipline, same reason, as the MXE
# and LAYER pairs above — the `mutants6` gate proves it by moving K_ROWS
# alone and requiring the checker to fire.
SB_R_MAX = 31                # apex_stage_buf R_MAX cap @ :103-104 (rows [4:0])
SB_D_LEGAL = (64, 128)       # apex_stage_buf.sv:100-101 legal row widths

# ── the LAYER units' acceptance rule (their OWN instantiated COLS_MAX) ───────
# SECOND ERRATUM (2026-07-30 audit) — SAME MISTAKE, ONE LANE OVER. The
# LAYER_JOB `cols` field is 12 bits, but every LAYER unit is instantiated with
# its own COLS_MAX and apex_top hands each one a slice of that field:
#   apex_top.sv:1216  apex_layer_deq #(.COLS_MAX(4095)) .jb_cols(lj_cols_q)
#   apex_top.sv:1234  asu_swiglu     #(.COLS_MAX(64))   .jb_cols(7'(lj_cols_q))
#   apex_top.sv:1249  apex_residual  #(.DM_MAX(LAYER_DM_MAX)) .jb_cols(...)
# The swiglu port is SEVEN bits (CW = $clog2(64+1)), so a 12-bit cols wider
# than 127 arrives TRUNCATED. Chunking d_ffn=18944 at the field bound gives
# [4095,4095,4095,4095,2564]: the four 4095s truncate to 127 and trip
# asu_swiglu's `jb_cols > COLS_MAX` job_error (loud), but the tail 2564
# truncates to 2564 & 0x7F = 4 — INSIDE the unit's own legality rule, so it is
# accepted and quietly computes 4 columns of the 2564 that were asked for.
# These are the CONSUMERS' numbers, read ONLY by assert_layer_job_legal();
# the decomposition below reads only the LU_CHUNK bounds. Same independence
# discipline, same reason, as the MXE pair above.
SWG_COLS_MAX = 64            # rtl/asu/asu_swiglu.sv COLS_MAX @ apex_top:1234
DEQ_COLS_MAX = 4095          # rtl/top/glue/apex_layer_deq.sv COLS_MAX @ :1216
RES_COLS_MAX = 3584          # apex_residual DM_MAX = LAYER_DM_MAX; the CL sets
                             # it from APEX_CL_DM (3584 wide build, 128 default
                             # — cl_apex.sv:544-548), so this is the WIDE-build
                             # envelope the 7B spec is written against.
RESID_WIN = 1024             # R3 (2026-07-30): LAYER_JOB[15:14] is the
                             # RESIDUAL unit's window base, stride 1024
                             # (apex_residual jb_base). The unit refuses
                             # base*1024 + cols > DM_MAX loudly (illegal
                             # geometry, frame_error class) — this constant
                             # and the base rule below mirror that consumer
                             # rule at emission. Other units require base=0
                             # (apex_top ignores the bits for them, but an
                             # emission that sets them is a spec bug).

# ── the WALKER's decomposition bounds (what it CHOOSES to emit) ──────────────
K_JOB = 2048                 # apex_pkg K_MAX          — mxe k_dim CEILING.
                             # THIRD ERRATUM: a ceiling, not the chunk — the
                             # per-D chunk bound is k_job() below. K_JOB is
                             # kept as the D=128 chunk (k_job(128) == K_JOB)
                             # and the legacy default of jobs(cfg_d=None).
K_ROWS = 31                  # the walker's restated stage-row cap — mirrors
                             # seq_walker_pkg WALK2_STAGE_R_CAP, deliberately
                             # its own name (never SB_R_MAX; see the
                             # independence note above)
N_MXE = 8                    # apex_pkg MXE_N          — mxe n_dim chunk
N_JOB = 4095                 # (1 << DIM_W) - 1        — LAYER_JOB cols field
N_LU_SWIGLU = 64             # asu_swiglu's implemented width — swiglu cols
                             # chunk. NOT the 12-bit field: the field is what
                             # the wire can CARRY, this is what the unit can
                             # CONSUME, and only the smaller of the two is a
                             # legal job.


class MxeIllegalDesc(AssertionError):
    """Raised when a generated descriptor would be refused by mxe_ctrl."""


class ActUnstageable(AssertionError):
    """Raised when a descriptor's act family cannot be staged by
    apex_stage_buf on any legal build (third erratum)."""


def assert_act_stageable(k: int, cfg_d: int, what: str = "") -> None:
    """The ACT STAGE BUFFER's OWN acceptance rule, applied to the activation
    family of a k-deep job this spec is about to bless. Reads SB_* (the
    consumer's constants) and NEVER K_ROWS/k_job() — that independence is the
    whole value of the check (the D-029 lesson, third lane), and the
    `mutants6` gate proves it holds by moving K_ROWS alone and requiring this
    to fire."""
    if cfg_d not in SB_D_LEGAL:
        raise ActUnstageable(
            f"ACT-UNSTAGEABLE {what}: row width D={cfg_d} is not a legal "
            f"apex_stage_buf elaboration ({SB_D_LEGAL})")
    rows = -(-k // cfg_d)                       # ceil — partial last row OK
    if not (1 <= k and rows <= SB_R_MAX):
        raise ActUnstageable(
            f"ACT-UNSTAGEABLE {what}: k={k} needs {rows} stage rows at "
            f"D={cfg_d}, outside the stage buffer's 1<=rows<={SB_R_MAX} "
            f"(apex_stage_buf.sv:167 legal, :103-104 R_MAX cap) — the act "
            f"family of this job cannot be staged on any legal build")


def k_job(cfg_d: int) -> int:
    """The walker's per-D k-chunk bound — mirrors seq_walker_pkg::
    walk2_k_job exactly: the largest whole-stage-row k at row width `cfg_d`
    under the K_JOB ceiling and the K_ROWS row cap. 1984 at D=64 (31 rows);
    2048 at D=128 (16 rows, == the legacy K_JOB, so every D=128 stream is
    byte-identical). A pure function of the image geometry ON PURPOSE:
    jobs() order IS the pre-swizzled DDR image byte order, so it may not
    depend on one build's instantiated STAGE_R_MAX."""
    assert cfg_d in (64, 128), f"k_job: row width {cfg_d} not in (64, 128)"
    return cfg_d * min(K_JOB // cfg_d, K_ROWS)


def assert_mxe_legal(m: int, k: int, n: int, what: str = "") -> None:
    """The TILE's OWN legality rule, applied to a descriptor this spec is
    about to bless. Reads MXE_*_MAX (the consumer's constants) and NEVER the
    decomposition bounds — see the note above; that independence is the whole
    value of the check, and the `mutants4` gate proves it holds by moving
    N_MXE alone and requiring this to fire."""
    if not (1 <= m <= MXE_M_MAX and 1 <= k <= MXE_K_MAX and 1 <= n <= MXE_N_MAX):
        raise MxeIllegalDesc(
            f"MXE-ILLEGAL DESC {what}: m={m} k={k} n={n} violates the tile "
            f"rule 1<=m<={MXE_M_MAX}, 1<=k<={MXE_K_MAX}, 1<=n<={MXE_N_MAX} "
            f"(mxe_ctrl.sv legal / mxe_cfg_pkg)")

# ── stage 5: JC table + LAYER level word (IB_LAYER.md §3b, frozen) ───────────
W_JC0 = 54
JC_SLOTS = 4
JC_OPROJ, JC_DOWN, JC_GATE, JC_UP = 0, 1, 2, 3
# E-3b: the ONE per-step S-2 q composite (per-tensor weight-scale model —
# the W_WSCALE0 design), consumed repeated by the QSTAGE step's MODE_F16
# sideband. Word 58 was reserved-0; read only when EN_QSTAGE is set.
W_QC = 58
LU_DEQ, LU_SWIGLU, LU_RESID = 0, 1, 2
# E-2b/E-3: unit 3 = the NORM/EGRESS job (residual row -> C-1 feeder -> the
# norm's x port). Its consumer is apex_residual's EGRESS port, so its cols
# limit is the same DM_MAX the add job answers to — plus apex_top's own
# push-gate rule (route armed, cols a nonzero multiple of the feeder row),
# which assert_layer_job_legal below mirrors for this unit only.
LU_NORM = 3
LU_NAMES = {LU_DEQ: "deq", LU_SWIGLU: "swiglu", LU_RESID: "resid",
            LU_NORM: "norm"}

# the CONSUMERS' rule (what each unit accepts) and the WALKER's rule (what it
# chooses to emit) — two tables ON PURPOSE; see the erratum note above.
LU_COLS_MAX = {LU_DEQ: DEQ_COLS_MAX, LU_SWIGLU: SWG_COLS_MAX,
               LU_RESID: RES_COLS_MAX, LU_NORM: RES_COLS_MAX}
# LU_NORM is deliberately absent from LU_CHUNK: a norm/egress job names a
# WINDOW of the residual row and the walker LU channel carries no base, so
# chunking it would re-read window 0 per chunk. The walker fences oversize
# instead (seq_layer_walker2 S2_CHECK) — mirrored by lu_chunks() raising.
LU_CHUNK = {LU_DEQ: N_JOB, LU_SWIGLU: N_LU_SWIGLU, LU_RESID: N_JOB}

# the C-1 feeder's row width (seam_feeder_quant D, = apex_top CFG_DM). A
# unit-3 push whose cols is not a nonzero multiple of this is REFUSED by
# apex_top with err_code 9 (NFEED_ROUTE) — the feeder would starve mid-row.
FEED_DM = 128


class LayerJobIllegal(AssertionError):
    """Raised when a LAYER_JOB this spec blesses would be mis-consumed."""


def assert_layer_job_legal(unit: int, cols: int, what: str = "",
                           base: int = 0) -> None:
    """The LAYER UNIT's OWN acceptance rule, applied to a LAYER_JOB push this
    spec is about to bless. Reads LU_COLS_MAX (the consumers' constants) and
    NEVER LU_CHUNK — that independence is the whole value of the check, and
    `gen_layer_trace.py --mutate-lu-swiglu` proves it fires.

    `cols` is what the host writes into the 12-bit LAYER_JOB field; the port
    width the unit actually sees is $clog2(COLS_MAX+1), so the message spells
    out the value that would ARRIVE — the silent case is the one where the
    truncated value is itself legal."""
    lim = LU_COLS_MAX[unit]
    if not 0 <= base <= 3:
        raise LayerJobIllegal(
            f"LAYER_JOB-ILLEGAL {what}: base={base} does not fit the 2-bit "
            f"[15:14] window field")
    if base and unit not in (LU_RESID, LU_NORM):
        raise LayerJobIllegal(
            f"LAYER_JOB-ILLEGAL {what}: unit={LU_NAMES[unit]} base={base} — "
            f"the [15:14] window base is the RESIDUAL unit's (R3); every "
            f"other unit requires base=0")
    # E-2b/E-3: apex_top's unit-3 push gate. cols must frame WHOLE C-1 rows
    # or the feeder starves mid-row; the RTL refuses loudly (err_code 9,
    # NFEED_ROUTE), and a spec that emitted such a push would be blessing a
    # program the tile rejects.
    if unit == LU_NORM and (cols == 0 or cols % FEED_DM):
        raise LayerJobIllegal(
            f"LAYER_JOB-ILLEGAL {what}: unit=norm cols={cols} is not a "
            f"nonzero multiple of the feeder row FEED_DM={FEED_DM} — "
            f"apex_top refuses the push (err_code 9, NFEED_ROUTE)")
    if unit in (LU_RESID, LU_NORM) and base and base * RESID_WIN + cols > lim:
        raise LayerJobIllegal(
            f"LAYER_JOB-ILLEGAL {what}: unit={LU_NAMES[unit]} base={base} "
            f"cols={cols} footprint {base * RESID_WIN + cols} > DM_MAX "
            f"{lim} — apex_residual refuses it (illegal geometry)")
    if 1 <= cols <= lim:
        return
    cw = max(1, (lim).bit_length())
    seen = cols & ((1 << cw) - 1)
    silent = 1 <= seen <= lim
    raise LayerJobIllegal(
        f"LAYER_JOB-ILLEGAL {what}: unit={LU_NAMES[unit]} cols={cols} "
        f"violates the unit rule 1<=cols<={lim} "
        f"(apex_top jb_cols is {cw} bits wide: the unit would see {seen}"
        + (" — SILENTLY LEGAL, a wrong answer with no job_error)"
           if silent else " and raise job_error)"))


def lctl(rope_en=0, rope_bank=0, ser_dst=0, fsrc_ext=0, resid_arm=0,
         rope_pos=0, nsrc=0) -> int:
    """The 20-bit LAYER level word (LAYER_CTRL bit positions, kv_map baked 0).

    E-3 (2026-08-01): fsrc_ext is 3 BITS. E-1 placed l_fsrc_ext[2] at
    LAYER_CTRL[7] — the slot this word's old hard-coded `0` filler occupied —
    so codes 0-3 encode to exactly the bits they always did and every
    pre-E-3 level word is byte-identical. Code 4 = the residual row's
    internal egress, the source a WALKED program could not name until the
    walker's own level net was widened to match.

    E-4b (2026-08-04): the word grows 15 -> 20 bits to carry l_nsrc at the
    REGISTER'S own position, LAYER_CTRL[19] (the E-2a norm-input-source
    level, HOST-held across a walk until now). Bits [18:15] — the register's
    l_kv_map/l_bias_en slots — stay reserved-0 here. Every legacy word
    (nsrc 0) encodes to exactly the 15 bits it always did, and the trace
    print (%04x-minimum, both sides) grows only when nsrc is set, so every
    legacy LCTL line is byte-identical."""
    assert 0 <= rope_pos < 128 and 0 <= ser_dst < 4 and 0 <= fsrc_ext < 8
    return ((nsrc & 1) << 19) \
        | ((rope_pos & 0x7F) << 8) | (((fsrc_ext >> 2) & 1) << 7) \
        | ((resid_arm & 1) << 6) \
        | ((fsrc_ext & 3) << 4) | ((ser_dst & 3) << 2) \
        | ((rope_bank & 1) << 1) | (rope_en & 1)


def grade_f32(x: float) -> int:
    """D-030 CANONICAL grade (IB_LAYER.md §3b @ d4f9563, BINDING per the
    §9.1 combine agenda (a)): f64 exact chain → fp32 IEEE-RNE →
    positive-normal required → RNE the significand to 11 significant bits
    AT THE RETAINED fp32 EXPONENT: clear frac[12:0], round up (+0x2000,
    carry into the exponent is correct) iff frac[12:0] > 0x1000 or
    (== 0x1000 and bit 13 set — tie-to-even).

    Explicitly NOT an fp16 round-trip: the fp16 cast denormalizes below
    2^-14 and destroys significand bits (their distinguishing example:
    1.811241e-6 grades to 1.8114224076e-6; the fp16 trip returns
    1.7881393433e-6). This function replaced exactly that fp16-trip
    implementation — the divergence the §3b warning names."""
    import numpy as np
    b = int(np.float32(np.float64(x)).view(np.uint32))
    assert (b >> 31) == 0 and 0 < ((b >> 23) & 0xFF) < 255, \
        f"composite {x} not positive-normal fp32 pre-grade"
    low = b & 0x1FFF
    v = b & ~0x1FFF
    if low > 0x1000 or (low == 0x1000 and ((b >> 13) & 1)):
        v += 0x2000
    e8 = (v >> 23) & 0xFF
    assert (v & 0x1FFF) == 0 and (v >> 31) == 0 and 0 < e8 < 255, \
        f"composite {x} not graded positive-normal post-round"
    return v


# ── word pack/unpack ─────────────────────────────────────────────────────────

def pack_geom0(head_dim: int, tier: int = 0, outlier_k: int = 0,
               fmt: int = FMT_LAYER) -> int:
    assert 0 <= head_dim <= 0xFF and 0 <= tier <= 3 and 0 <= outlier_k <= 0xF
    return ((fmt & 0xF) << 28) | ((outlier_k & 0xF) << 20) \
        | ((tier & 0x3) << 16) | (head_dim & 0xFF)


def pack_model0(d_model: int, d_ffn: int) -> int:
    assert 0 <= d_model <= 0xFFFF and 0 <= d_ffn <= 0xFFFF
    return ((d_ffn & 0xFFFF) << 16) | (d_model & 0xFFFF)


def pack_model1(n_heads: int, n_kv_heads: int, kv_map: int = 1) -> int:
    # kv_map default 1 = §9.1 R3 provisional per-KV-head engine banks
    assert 0 <= n_heads <= 0xFF and 0 <= n_kv_heads <= 0xFF and 0 <= kv_map <= 3
    return ((n_kv_heads & 0xFF) << 16) | ((n_heads & 0xFF) << 8) | (kv_map & 0x3)


def pack_mask(en_mask: int) -> int:
    assert 0 <= en_mask < (1 << EN_W)
    return en_mask & ((1 << EN_W) - 1)


def pack_step(t_rows: int, pos_m: int) -> int:
    assert 0 <= t_rows <= 0xFF and 0 <= pos_m <= 0xFF
    return ((t_rows & 0xFF) << 8) | (pos_m & 0xFF)


def pack_rq(rq_scale: int, rq_shift: int) -> int:
    # identical packing to the D-028 RQ word: [20:16] shift, [15:0] scale
    assert 0 <= rq_scale <= 0xFFFF and 0 <= rq_shift <= 0x1F
    return ((rq_shift & 0x1F) << 16) | (rq_scale & 0xFFFF)


def pack_tens(base64: int) -> int:
    assert 0 <= base64 < (1 << 30), f"tensor base {base64} exceeds 30 bits"
    return base64 & 0x3FFF_FFFF


def unpack_scalar_words(w_geom0: int, w_model0: int, w_model1: int,
                        w_mask: int, w_step: int) -> dict:
    return {
        "fmt": (w_geom0 >> 28) & 0xF,
        "outlier_k": (w_geom0 >> 20) & 0xF,
        "tier": (w_geom0 >> 16) & 0x3,
        "head_dim": w_geom0 & 0xFF,
        "d_model": w_model0 & 0xFFFF,
        "d_ffn": (w_model0 >> 16) & 0xFFFF,
        "n_heads": (w_model1 >> 8) & 0xFF,
        "n_kv_heads": (w_model1 >> 16) & 0xFF,
        "kv_map": w_model1 & 0x3,
        "en_mask": w_mask & ((1 << EN_W) - 1),
        "t_rows": (w_step >> 8) & 0xFF,
        "pos_m": w_step & 0xFF,
    }


def check2(w_geom0: int, w_model0: int, w_model1: int, w_mask: int,
           w_step: int, cfg_d: int) -> int:
    """EXACT mirror of seq_walker_pkg::walk_desc2_check — same clause order,
    same codes. Divergence between this and the SV function is a stage-2
    equivalence-gate failure by definition."""
    d = unpack_scalar_words(w_geom0, w_model0, w_model1, w_mask, w_step)
    # E-5: the mask-resv slice starts at bit 15 now — bit 14 is EN_FPROJ.
    # The pre-E-5 check (>> 14 here) refuses an FPROJ image loudly on old
    # RTL; a legacy image (bit 14 = 0) checks identically. Same for the
    # E-4b boundary before it (bit 13 = EN_NSRC).
    resv_nz = ((w_geom0 >> 24) & 0xF) or ((w_geom0 >> 18) & 0x3) \
        or ((w_geom0 >> 8) & 0xFF) or ((w_model1 >> 24) & 0xFF) \
        or ((w_model1 >> 2) & 0x3F) or ((w_mask >> 17) & 0x7FFF) \
        or ((w_step >> 16) & 0xFFFF)
    nh, nk = d["n_heads"], d["n_kv_heads"]
    if d["fmt"] != FMT_LAYER:
        return ERR_DESC
    if resv_nz:
        return ERR_DESC
    if d["tier"] != 0:                       # KVQ_CQ8
        return ERR_TIER
    if d["head_dim"] != cfg_d:
        return ERR_DESC
    if d["t_rows"] == 0 or d["t_rows"] > T_MAX:
        return ERR_DESC
    if d["pos_m"] >= T_MAX:
        return ERR_DESC
    if nh == 0 or nk == 0 or nh > H_MAX:
        return ERR_DESC
    if nh % nk != 0:
        return ERR_DESC
    if nh * d["head_dim"] != d["d_model"]:
        return ERR_DESC
    if d["en_mask"] == 0:
        return ERR_DESC
    if d["kv_map"] & 0x2:
        return ERR_DESC
    return ERR_NONE


def check1(w_geom: int, w_rq: int, w_mask: int, cfg_d: int) -> int:
    """EXACT mirror of seq_walker_pkg::walk_desc_check (v1, including the
    v1.1 fmt-nibble hardening) — same clause order, same codes. w_rq is
    carried, not checked, exactly as in the SV."""
    _ = w_rq
    fmt_n = (w_geom >> 28) & 0xF
    tier = (w_geom >> 16) & 0x3
    t_rows = (w_geom >> 8) & 0xFF
    d_dim = w_geom & 0xFF
    en_score = w_mask & 1
    en_pv = (w_mask >> 1) & 1
    if fmt_n != FMT_V1:
        return ERR_DESC
    if tier != 0:                            # KVQ_CQ8
        return ERR_TIER
    if t_rows == 0 or t_rows > T_MAX:
        return ERR_DESC
    if d_dim != cfg_d:
        return ERR_DESC
    if not en_score and not en_pv:
        return ERR_DESC
    return ERR_NONE


def directed_check2_cases() -> list[tuple[tuple[int, int, int, int, int], int, int, str]]:
    """The directed check2 corpus: ((5 words), cfg_d, expected, name).
    Single source for the selftest, gen_check_vectors.py, and any future
    harness — the 7B-legal base plus one corruption per legality clause."""
    seven_b = (pack_geom0(128), pack_model0(3584, 18944),
               pack_model1(28, 4, 1), pack_mask(EN_ALL), pack_step(8, 7))
    cases = [(seven_b, 128, ERR_NONE, "7B legal")]
    bad = [
        ((pack_geom0(128, fmt=2),) + seven_b[1:], ERR_DESC, "unknown fmt"),
        ((seven_b[0] | (1 << 24),) + seven_b[1:], ERR_DESC, "geom resv bit"),
        ((seven_b[0] | (1 << 8),) + seven_b[1:], ERR_DESC, "t_rows-in-GEOM0 resv"),
        ((pack_geom0(128, tier=1),) + seven_b[1:], ERR_TIER, "grouped tier"),
        (seven_b[:4] + (pack_step(0, 0),), ERR_DESC, "t_rows=0"),
        (seven_b[:4] + (pack_step(8, 200),), ERR_DESC, "pos_m>=128"),
        (seven_b[:4] + (seven_b[4] | (1 << 16),), ERR_DESC, "step resv"),
        (seven_b[:2] + (pack_model1(28, 3, 1),) + seven_b[3:], ERR_DESC,
         "H % H_kv"),
        (seven_b[:2] + (pack_model1(27, 3, 1),) + seven_b[3:], ERR_DESC,
         "H*hd != d_model"),
        (seven_b[:2] + (pack_model1(28, 4, 2),) + seven_b[3:], ERR_DESC,
         "kv_map undefined encoding"),
        (seven_b[:3] + (pack_mask(0),) + seven_b[4:], ERR_DESC, "empty mask"),
        # E-7/E-7b boundary pins: bits 15/16 are EN_FGAM/EN_DOWN now
        # (pkg-LEGAL — their envelope rules are S2_CHECK fences,
        # deliberately not check2 clauses); bit 17 is the NEW first
        # reserved mask bit.
        (seven_b[:3] + (seven_b[3] | (1 << 17),) + seven_b[4:], ERR_DESC,
         "mask resv (bit 17, post-E-7 boundary)"),
    ]
    cases += [(w, 128, c, n) for (w, c, n) in bad]
    cases.append((seven_b[:3]
                  + (pack_mask(EN_ALL_NFEED | (1 << EN_NSRC)),)
                  + seven_b[4:], 128, ERR_NONE, "NSRC rides the mask (E-4b)"))
    cases.append((seven_b, 64, ERR_DESC, "head_dim != CFG_D"))
    return cases


def directed_check1_cases() -> list[tuple[tuple[int, int, int], int, int, str]]:
    """Directed v1 corpus: ((geom, rq, mask), cfg_d, expected, name)."""
    legal = ((64 & 0xFF) | (8 << 8), 0, 0x3)
    return [
        (legal, 64, ERR_NONE, "v1 legal"),
        ((legal[0] | (1 << 28), 0, 0x3), 64, ERR_DESC, "fmt nibble (v1.1)"),
        ((legal[0] | (1 << 16), 0, 0x3), 64, ERR_TIER, "grouped tier"),
        ((64, 0, 0x3), 64, ERR_DESC, "t_rows=0"),
        ((64 | (200 << 8), 0, 0x3), 64, ERR_DESC, "t_rows>128"),
        (legal, 128, ERR_DESC, "d_dim != CFG_D"),
        ((legal[0], 0, 0x0), 64, ERR_DESC, "empty mask"),
    ]


# ── fuel_req record ──────────────────────────────────────────────────────────

def pack_freq(base64: int, beats64: int, tag: int) -> int:
    """The frozen 64-bit fuel_req: base [29:0], beats [55:30], tag [63:56]."""
    assert 0 <= base64 < (1 << FR_BASE_W)
    assert 0 < beats64 < (1 << FR_BEATS_W), f"beats {beats64} out of range"
    assert 0 <= tag < (1 << FR_TAG_W)
    return (tag << FR_TAG_LSB) | (beats64 << FR_BEATS_LSB) | base64


def unpack_freq(rec: int) -> tuple[int, int, int]:
    return (rec & ((1 << FR_BASE_W) - 1),
            (rec >> FR_BEATS_LSB) & ((1 << FR_BEATS_W) - 1),
            (rec >> FR_TAG_LSB) & ((1 << FR_TAG_W) - 1))


# ── beat math + job decomposition (fixes the DDR image order) ────────────────

def beats64_of_bytes(nbytes: int) -> int:
    """64-byte beats of a tensor image; tensors are 64 B-aligned so the pad,
    when bytes % 64 != 0, sits at the tensor tail."""
    return (nbytes + 63) // 64


def jobs(k_total: int, n_total: int, n_job: int | None = None,
         cfg_d: int | None = None) -> list[tuple[int, int, int, int, int]]:
    """The decomposition rule (IB_WALK.md §2.6): n-split-major, k-split-minor.
    Returns [(n0, k0, k, n, accumulate)] in job order — the pre-swizzled DDR
    image concatenates the per-job weight blocks in exactly this order.

    n chunks at `N_MXE` (the ARRAY width, 8), k at `k_job(cfg_d)`. D-029
    erratum: n used to chunk at the 12-bit descriptor FIELD width
    (`N_JOB`=4095), which produced tile-illegal jobs — `mxe_ctrl.sv:164`
    refuses `n_dim > MXE_N`. `n_job` overrides the width for callers that
    need an explicit chunk (the W-G3 harness passes it); the DEFAULT is the
    corrected array width, so a caller that says nothing gets a legal
    stream.

    THIRD ERRATUM (W1, 2026-08-05): the k chunk is D-DEPENDENT — the act
    stage buffer holds at most 31 rows of `cfg_d` codes, so K_JOB (2048) is
    only a legal chunk at a 128-wide row; at 64 the bound is 1984 (see the
    SB_* consumer note above and k_job()). `cfg_d` names the build's stage
    row width (CFG_DM == CFG_D in every buildable image). None keeps the
    legacy K_JOB chunk, which equals k_job(128) — every existing caller is a
    D=128 flow and their streams (and the frozen DDR images cut from them)
    are byte-identical. A D=64 caller MUST pass cfg_d=64; the stageability
    sweep (assert_act_stageable, gen_layer_trace third sweep, mutants6)
    rejects the stream of one that does not."""
    n_job = N_MXE if n_job is None else n_job
    k_chunk = K_JOB if cfg_d is None else k_job(cfg_d)
    assert k_total >= 1 and n_total >= 1 and n_job >= 1
    out = []
    n_starts = list(range(0, n_total, n_job))
    k_starts = list(range(0, k_total, k_chunk))
    for n0, na in enumerate(n_starts):
        n = min(n_job, n_total - na)
        for k0, ka in enumerate(k_starts):
            k = min(k_chunk, k_total - ka)
            out.append((n0, k0, k, n, 1 if k0 > 0 else 0))
    return out


def lu_chunks(cols: int, unit: int | None = None) -> list[int]:
    """LAYER_JOB cols chunking — the OTHER bound (IB_LAYER §3b). Kept a
    separate function from jobs() so the two rules cannot re-merge.

    SECOND ERRATUM (2026-07-30 audit): the bound is per-UNIT, and it is the
    smaller of the 12-bit field and the unit's implemented COLS_MAX. It used
    to be the field width for every unit, which made every swiglu split of a
    7B FFN row wrong — and the LAST chunk of d_ffn=18944 wrong SILENTLY
    (2564 -> 4 at the 7-bit port). `unit` defaults to None = "legal at EVERY
    LAYER unit" (the tightest bound), so a caller that says nothing gets a
    stream no unit can mis-consume — the same default-safe rule jobs() uses."""
    assert cols >= 1
    # E-3: LU_NORM is not chunkable (see the LU_CHUNK note) — say so instead
    # of raising a bare KeyError from the table lookup.
    if unit == LU_NORM:
        raise LayerJobIllegal(
            "LAYER_JOB-ILLEGAL lu_chunks(unit=norm): a NORM/EGRESS job names "
            "a window of the residual row and the walker LU channel carries "
            "no base, so chunking would re-read window 0 per chunk; the "
            "walker fences oversize at S2_CHECK instead of splitting")
    bound = min(LU_CHUNK.values()) if unit is None else LU_CHUNK[unit]
    out = [min(bound, cols - c) for c in range(0, cols, bound)]
    for c in out:
        # the None sweep covers the CHUNKABLE units only — LU_NORM has its
        # own framing rule (whole feeder rows) that a generic chunk cannot
        # be expected to satisfy
        for u in (LU_CHUNK if unit is None else [unit]):
            assert_layer_job_legal(u, c, f"lu_chunks({cols}, unit={unit})")
    return out


def tensor_shapes(d_model: int, d_ffn: int, n_kv_heads: int, head_dim: int,
                  t_max: int = T_MAX) -> dict[int, tuple[int, int]]:
    """(K, N) per weight tensor + the byte-shaped aux tensors. Gamma rows are
    d_model int16; the RoPE phase table is t_max rows of head_dim/2 uint16
    codes (C-ROPE: quantized ONCE from float64 — table-fed to stay bit-exact,
    fetched one row per step at row index pos_m)."""
    kv_dim = n_kv_heads * head_dim
    return {
        TENS_WQ: (d_model, d_model),
        TENS_WK: (d_model, kv_dim),
        TENS_WV: (d_model, kv_dim),
        TENS_WO: (d_model, d_model),
        TENS_WG: (d_model, d_ffn),
        TENS_WU: (d_model, d_ffn),
        TENS_WD: (d_ffn, d_model),
        TENS_G1: (1, 2 * d_model),           # bytes = K*N for aux rows too
        TENS_G2: (1, 2 * d_model),
        TENS_PHASE: (t_max, head_dim),       # row = head_dim bytes (hd/2 u16)
    }


def tensor_bytes(shapes: dict[int, tuple[int, int]]) -> dict[int, int]:
    return {t: k * n for t, (k, n) in shapes.items()}


def image_bases(shapes: dict[int, tuple[int, int]]) -> dict[int, int]:
    """64 B-unit base per tensor, packed in TENS order from 0."""
    bases, cur = {}, 0
    for t in range(TENS_N):
        bases[t] = cur
        cur += beats64_of_bytes(tensor_bytes(shapes)[t])
    return bases


# ── selftest ─────────────────────────────────────────────────────────────────

def _selftest() -> int:
    import random
    rnd = random.Random(0xD029)

    # 1. scalar-word round-trip over randoms + the 7B point
    for _ in range(2000):
        f = {"head_dim": rnd.choice([64, 128]), "tier": 0,
             "outlier_k": rnd.randrange(16),
             "d_model": rnd.randrange(1 << 16), "d_ffn": rnd.randrange(1 << 16),
             "n_heads": rnd.randrange(256), "n_kv_heads": rnd.randrange(256),
             "kv_map": rnd.randrange(2), "en_mask": rnd.randrange(1 << EN_W),
             "t_rows": rnd.randrange(256), "pos_m": rnd.randrange(256)}
        w = (pack_geom0(f["head_dim"], f["tier"], f["outlier_k"]),
             pack_model0(f["d_model"], f["d_ffn"]),
             pack_model1(f["n_heads"], f["n_kv_heads"], f["kv_map"]),
             pack_mask(f["en_mask"]), pack_step(f["t_rows"], f["pos_m"]))
        u = unpack_scalar_words(*w)
        for k, v in f.items():
            assert u[k] == v, f"round-trip {k}: {u[k]} != {v}"
        assert u["fmt"] == FMT_LAYER

    # 2. check-mirror: the shared directed corpora (also consumed by
    #    gen_check_vectors.py — the SV-vs-mirror equivalence harness)
    d2c = directed_check2_cases()
    for words, cfgd, code, what in d2c:
        got = check2(*words, cfg_d=cfgd)
        assert got == code, f"check2 '{what}': got {got}, want {code}"
    d1c = directed_check1_cases()
    for words, cfgd, code, what in d1c:
        got = check1(*words, cfg_d=cfgd)
        assert got == code, f"check1 '{what}': got {got}, want {code}"
    n_neg = sum(1 for (_w, _d, code, _n) in d2c + d1c if code != ERR_NONE)

    # 3. fuel_req layout: frozen bit positions + round-trip
    rec = pack_freq(0x2AAA_AAAA, 0x155_5555, 0xA5)
    assert rec == (0xA5 << 56) | (0x155_5555 << 30) | 0x2AAA_AAAA
    assert unpack_freq(rec) == (0x2AAA_AAAA, 0x155_5555, 0xA5)
    assert FR_BASE_W + FR_BEATS_W + FR_TAG_W == 64

    # 4. decomposition identity vs the golden chunking pattern
    #    (gemm_i8_ksplit iterates range(0, K, k_chunk) / N by concatenation)
    n_mxe_chk = 0
    for k_total, n_total in [(3584, 3584), (3584, 512), (3584, 18944),
                             (18944, 3584), (128, 128), (344, 128)]:
        js = jobs(k_total, n_total)
        n_starts = list(range(0, n_total, N_MXE))
        k_starts = list(range(0, k_total, K_JOB))
        assert len(js) == len(n_starts) * len(k_starts)
        for (n0, k0, k, n, acc) in js:
            assert k == min(K_JOB, k_total - k_starts[k0])
            assert n == min(N_MXE, n_total - n_starts[n0])
            assert acc == (1 if k0 > 0 else 0)
        assert sum(k for (_, _, k, _, _) in js) \
            == len(n_starts) * k_total, "k coverage"
        # FUEL invariant: every job's lane-beat count divisible by 8
        for (_, _, k, n, _) in js:
            assert (k * n) % 8 == 0, f"lane-beat divisibility {k}x{n}"
        # D-029 erratum: every decomposed job must be legal AT THE TILE
        for (_, _, k, n, _) in js:
            assert_mxe_legal(1, k, n, f"jobs({k_total},{n_total})")
            n_mxe_chk += 1

    # 4a. the erratum's own negatives — the predicate must REJECT the shapes
    #     the pre-fix constant produced, and must not be silently disabled
    assert (MXE_N_MAX, MXE_M_MAX, MXE_K_MAX) == (8, 64, 2048), "tile-rule drift"
    assert (N_MXE, K_JOB, N_JOB) == (8, 2048, 4095), "decomposition drift"
    for (m, k, n, why) in [(1, 2048, 4095, "the pre-fix n-split"),
                           (1, 2048, 9, "n one past the array"),
                           (65, 128, 8, "m past M_TILE_MAX (T=65 K/V proj)"),
                           (1, 2049, 8, "k past K_MAX"),
                           (1, 128, 0, "n=0")]:
        try:
            assert_mxe_legal(m, k, n, why)
        except MxeIllegalDesc:
            pass
        else:
            raise AssertionError(f"assert_mxe_legal accepted {why}: "
                                 f"m={m} k={k} n={n}")
    # 4b'. THIRD ERRATUM — the k bound is D-DEPENDENT. Consumer-side pins
    #      first (the checker's own numbers), then the walker's bound, then
    #      the 0.5B down-projection decomposition the defect blocked.
    assert (SB_R_MAX, SB_D_LEGAL) == (31, (64, 128)), "stage-buf rule drift"
    assert K_ROWS == 31 and k_job(64) == 1984 and k_job(128) == K_JOB, \
        "k_job drift"
    #      D=128 byte-identity: cfg_d=128 must reproduce the legacy stream
    for k_total, n_total in [(3584, 3584), (3584, 512), (3584, 18944),
                             (18944, 3584), (128, 128), (344, 128)]:
        assert jobs(k_total, n_total, cfg_d=128) == jobs(k_total, n_total), \
            f"cfg_d=128 changed the {k_total}x{n_total} stream"
    #      the 0.5B Wd (k=4864, n=896) at D=64: 112 n-splits x [1984,1984,896]
    js = jobs(4864, 896, cfg_d=64)
    assert len(js) == 112 * 3, len(js)
    n_sb_chk = 0
    for (n0, k0, k, n, acc) in js:
        assert k == (896 if k0 == 2 else 1984) and n == 8, (k0, k, n)
        assert acc == (1 if k0 > 0 else 0)
    assert sum(k for (_, _, k, _, _) in js) == 112 * 4864, "k coverage"
    #      the whole 0.5B family decomposes tile-legal AND stageable at D=64
    for (kt, nt, nm) in [(896, 896, "Wq"), (896, 128, "Wk/Wv"),
                         (896, 4864, "Wg/Wu"), (4864, 896, "Wd")]:
        for (_, _, k, n, _) in jobs(kt, nt, cfg_d=64):
            assert_mxe_legal(1, k, n, f"0.5B {nm} @ D=64")
            assert_act_stageable(k, 64, f"0.5B {nm} @ D=64")
            n_sb_chk += 1
    #      the checker's own negatives — it must REJECT the pre-fix shapes,
    #      including the stream the legacy default cuts for Wd (the blind-spot
    #      shape: walker and spec agreeing on one wrong constant)
    n_sb_neg = 0
    for (k, d, why) in [(2048, 64, "the pre-fix D=64 chunk (32 rows)"),
                        (1985, 64, "one code past the 31-row family"),
                        (3969, 128, "32 rows at D=128"),
                        (64, 32, "illegal stage row width")]:
        try:
            assert_act_stageable(k, d, why)
        except ActUnstageable:
            n_sb_neg += 1
        else:
            raise AssertionError(f"assert_act_stageable accepted {why}")
    try:
        for (_, _, k, _, _) in jobs(4864, 896):     # cfg_d unnamed: K_JOB
            assert_act_stageable(k, 64, "legacy-default Wd @ D=64")
    except ActUnstageable:
        pass
    else:
        raise AssertionError("the checker blessed the pre-fix Wd stream")
    #      boundary sanity: the whole legal envelope is accepted
    assert_act_stageable(1984, 64, "31 whole rows @ 64")
    assert_act_stageable(2048, 128, "16 rows @ 128")
    assert_act_stageable(344, 128, "partial last row (the l128 Wd k)")
    # 4c. the LAYER_JOB cols path — its own rule, and it must NOT have
    #     followed n down to 8. SECOND ERRATUM (2026-07-30 audit): the bound
    #     is per-UNIT (min of the 12-bit field and the unit's COLS_MAX), not
    #     the field for everybody.
    assert (SWG_COLS_MAX, DEQ_COLS_MAX, RES_COLS_MAX) == (64, 4095, 3584), \
        "LAYER unit rule drift"
    assert (N_JOB, N_LU_SWIGLU) == (4095, 64), "LAYER chunk-bound drift"
    assert lu_chunks(18944, LU_SWIGLU) == [64] * 296
    assert sum(lu_chunks(18944, LU_SWIGLU)) == 18944, "swiglu cols coverage"
    assert lu_chunks(18944, LU_DEQ) == [4095, 4095, 4095, 4095, 2564]
    assert lu_chunks(3584, LU_DEQ) == [3584] and lu_chunks(3584, LU_RESID) \
        == [3584] and lu_chunks(4095, LU_DEQ) == [4095]
    assert lu_chunks(200) == [64, 64, 64, 8], "unit=None must be swiglu-safe"
    n_lu_neg = 0
    for (u, c, why) in [
            (LU_SWIGLU, 4095, "the pre-fix swiglu chunk (arrives as 127)"),
            (LU_SWIGLU, 2564, "the pre-fix TAIL chunk — arrives as 4, SILENT"),
            (LU_SWIGLU, 65, "one past asu_swiglu's COLS_MAX"),
            (LU_SWIGLU, 0, "cols=0"),
            (LU_DEQ, 4096, "one past the deq COLS_MAX / the 12-bit field"),
            (LU_RESID, 3585, "one past the wide-build residual DM_MAX")]:
        try:
            assert_layer_job_legal(u, c, why)
        except LayerJobIllegal:
            n_lu_neg += 1
        else:
            raise AssertionError(f"assert_layer_job_legal accepted {why}: "
                                 f"unit={u} cols={c}")
    # and the silent case must be NAMED as silent — that is the whole defect
    try:
        assert_layer_job_legal(LU_SWIGLU, 2564)
    except LayerJobIllegal as e:
        assert "SILENTLY LEGAL" in str(e) and "would see 4" in str(e), str(e)

    # 4d. E-3: the fsrc_ext WIDENING and the NORM/EGRESS unit.
    #     (i) every legacy level word is byte-identical — the third bit sits
    #     at LAYER_CTRL[7], which the old encoder hard-coded to 0.
    for rp in (0, 7, 127):
        for sd in range(4):
            for fe in range(4):
                for ra in (0, 1):
                    w = lctl(rope_en=1, rope_bank=0, ser_dst=sd, fsrc_ext=fe,
                             resid_arm=ra, rope_pos=rp)
                    assert (w >> 7) & 1 == 0, "legacy code set bit 7"
                    assert w == (((rp & 0x7F) << 8) | (ra << 6) | (fe << 4)
                                 | (sd << 2) | 1), "legacy lctl drift"
    #     (ii) code 4 is REACHABLE and lands at bit 7, nowhere else
    w4 = lctl(rope_en=1, ser_dst=1, resid_arm=1, rope_pos=7, fsrc_ext=4)
    assert (w4 >> 7) & 1 == 1 and (w4 >> 4) & 3 == 0, f"{w4:#06x}"
    assert w4 == lctl(rope_en=1, ser_dst=1, resid_arm=1, rope_pos=7) | 0x80
    assert w4 < (1 << 15), "every nsrc-free level word must stay 15 bits"
    #     (ii-b) E-4b: nsrc lands at bit 19 — the REGISTER's own position —
    #     and nowhere else; [18:15] (the kv_map/bias slots) stay reserved-0;
    #     and the minimum-width-4 hex print (both TB and spec sides) keeps
    #     every legacy word's trace line byte-identical.
    wn = lctl(rope_en=1, ser_dst=1, resid_arm=1, rope_pos=7, fsrc_ext=4,
              nsrc=1)
    assert wn == w4 | (1 << 19) and (wn >> 15) & 0xF == 0, f"{wn:#07x}"
    assert wn < (1 << 20), "level word must stay 20 bits"
    assert f"{w4:04x}" == f"{w4 & 0x7FFF:04x}" and len(f"{w4:04x}") == 4, \
        "legacy LCTL print drift"
    assert len(f"{wn:04x}") == 5, "nsrc LCTL print must grow, not truncate"
    #     (iii) the unit-3 push rules: whole feeder rows, window base allowed,
    #     never chunked
    assert_layer_job_legal(LU_NORM, FEED_DM, "one C-1 row")
    assert_layer_job_legal(LU_NORM, 28 * FEED_DM, "the 7B row family")
    n_nf_neg = 0
    for (u, c, bs, why) in [
            (LU_NORM, 96, 0, "cols not a whole feeder row"),
            (LU_NORM, 0, 0, "cols=0"),
            (LU_NORM, 5 * FEED_DM, 3, "window footprint past DM_MAX")]:
        try:
            assert_layer_job_legal(u, c, why, base=bs)
        except LayerJobIllegal:
            n_nf_neg += 1
        else:
            raise AssertionError(f"assert_layer_job_legal accepted {why}")
    assert n_nf_neg == 3
    try:
        lu_chunks(2 * FEED_DM, LU_NORM)
    except LayerJobIllegal as e:
        assert "no base" in str(e), str(e)
    else:
        raise AssertionError("lu_chunks chunked a NORM/EGRESS job")
    assert EN_ALL == (1 << 11) - 1 and not (EN_ALL >> EN_NFEED) & 1, \
        "EN_ALL must stay NFEED-free so legacy images are byte-identical"
    # E-4b: bit 13 is EN_NSRC; every legacy mask constant must be free of it
    # (byte-identity), and the pack path must carry it when named.
    assert EN_NSRC == 13 and EN_FPROJ == 14 and EN_FGAM == 15 \
        and EN_DOWN == 16 and EN_W == 17
    assert not (EN_ALL_NFEED >> EN_NSRC) & 1 and \
        not (EN_ALL_NFEED >> EN_QSTAGE) & 1 and \
        not (EN_ALL_NFEED >> EN_FPROJ) & 1 and \
        not (EN_ALL_NFEED >> EN_FGAM) & 1 and \
        not (EN_ALL_NFEED >> EN_DOWN) & 1, \
        "legacy mask constants must stay NSRC/QSTAGE/FPROJ/FGAM/DOWN-free"
    assert pack_mask(1 << EN_NSRC) == (1 << 13)
    assert pack_mask(1 << EN_FPROJ) == (1 << 14)
    assert pack_mask(1 << EN_FGAM) == (1 << 15)
    assert pack_mask(1 << EN_DOWN) == (1 << 16)

    # 4b. D-030 canonical grade conformance (IB_LAYER.md §3b @ d4f9563):
    #     their distinguishing example, the sub-2^-24 case, and idempotence
    import numpy as np
    g = grade_f32(1.811241e-6)
    gv = float(np.uint32(g).view(np.float32))
    assert abs(gv - 1.8114224076e-6) < 1e-15, f"distinguishing example: {gv!r}"
    assert ((g >> 23) & 0xFF) == 107, f"e8 = {(g >> 23) & 0xFF}, want 107"
    trip = float(np.float32(np.float16(gv)))
    assert abs(trip - 1.7881393433e-6) < 1e-15 and trip != gv, \
        "the fp16 round-trip must DIVERGE on the graded example"
    g2 = grade_f32(4.87e-8)                       # below 2^-24
    g2v = float(np.uint32(g2).view(np.float32))
    assert abs(g2v - 4.87e-8) / 4.87e-8 < 2e-3, f"sub-2^-24 grade: {g2v!r}"
    assert abs(float(np.float32(np.float16(g2v))) - 5.96e-8) < 1e-9, \
        "the fp16 trip must be off by the subnormal grid below 2^-24"
    rg = random.Random(0xD030)
    for _ in range(2000):
        x = rg.uniform(1e-9, 1e5) * (10.0 ** -rg.randrange(0, 6))
        v = grade_f32(x)
        assert v & 0x1FFF == 0 and 0 < ((v >> 23) & 0xFF) < 255
        assert grade_f32(float(np.uint32(v).view(np.float32))) == v, \
            "idempotence"

    # 5. R1 beat-width justification recomputed from the 7B shape
    sh = tensor_shapes(3584, 18944, 4, 128)
    b = {t: beats64_of_bytes(nb) for t, nb in tensor_bytes(sh).items()}
    assert b[TENS_WD] == 1_060_864, b[TENS_WD]
    assert b[TENS_WD] > (1 << 20), "the stage-0 20-bit field really was short"
    assert max(b.values()) < (1 << 26), "26-bit beats covers every fmt=1 tensor"
    assert b[TENS_G1] == 112 and b[TENS_PHASE] == 256
    # FUEL invariant: row-granular tensors have 64 B-multiple rows
    for t in (TENS_G1, TENS_G2, TENS_PHASE):
        _, row = sh[t]
        assert row % 64 == 0, f"row-granular tensor {TENS_NAMES[t]} row {row}"
    total = sum(b.values())
    assert image_bases(sh)[TENS_PHASE] + b[TENS_PHASE] == total
    assert total < (1 << 30)

    print(f"SEQ_WALKER_FMT SELFTEST: PASS "
          f"(2000 round-trips, {n_neg} check negatives incl. v1, fuel_req "
          f"layout pinned, decomposition == golden chunking on 6 shapes, "
          f"{n_mxe_chk} decomposed jobs MXE-legal + 5 directed illegals "
          f"rejected, k chunk D-AWARE (k_job 64->{k_job(64)} "
          f"128->{k_job(128)}, D=128 streams byte-identical): {n_sb_chk} "
          f"0.5B D=64 jobs stageable + {n_sb_neg} directed unstageables "
          f"rejected incl. the pre-fix 2048@64, LAYER cols chunk at the "
          f"per-UNIT bound (swiglu {N_LU_SWIGLU}, deq {N_JOB}) + {n_lu_neg} "
          f"directed LAYER_JOB illegals rejected, "
          f"7B image = {total} beats64 / Wd = {b[TENS_WD]})")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
