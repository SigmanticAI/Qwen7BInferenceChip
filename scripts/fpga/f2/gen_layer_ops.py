#!/usr/bin/env python3
# gen_layer_ops.py — MASTER TABLE rows B2-B5: the ONE emitter for a FULL
# decoder-layer step through the LAYER window (0x70-0x8C) of HEAD's apex_top.
#
# ══ WHAT THIS IS ═══════════════════════════════════════════════════════════
# C1 (breadth_step.py) proved the ATTENTION + PROJECTION op families of one
# real Qwen2.5-7B decode step on silicon. Everything else a decoder layer
# does — RoPE, the layer dequantizer, the C-1 feeder re-entry, residual-1,
# RMSNorm-2, SwiGLU, residual-2 — lives behind the LAYER CSR window, and
# NOTHING in the tree emitted a single write to it. This file does.
#
#   B2  the LAYER script tokens (LCTL/LPTR/LDATA/LUJOB/LJC/LRPTR/ERD/LPOLL)
#       + ERD, the SINGLE-READ produce-mode capture token (below)
#   B3  segment 1: o8 -> layer-deq -> residual-1 -> RDATA, golden-graded
#   B4  the tail: SwiGLU (per-unit 64-col chunks), wide RMSNorm-2,
#       residual-2, + the cap_decode extension (`rdata_f16`, marked "# B4:")
#   B5  the RoPE observability fix (below)
#
# ══ ERD: WHY RDATA NEEDS ITS OWN TOKEN ═════════════════════════════════════
# LAYER_RDATA (0x88) AUTO-INCREMENTS LAYER_RPTR on every read
# (apex_top.sv:1120 `else if (lr_rdata) l_rptr_q <= l_rptr_q + 16'd1`). The
# `--shadow-cap` bring-up idiom — emit a `cap` beside every `r` at the same
# address — therefore performs TWO reads of a POINTER-ADVANCING register and
# skews every later element by one. `ERD` emits EXACTLY ONE bus read, as a
# `cap` with NO expectation: produce-mode by construction, and the only
# read shape that is correct for this register. `--erd-check` flips it to a
# single checked read for bring-up (still one read).
# The skew is not asserted from prose — `--discriminate` reproduces it.
#
# ══ B5: THE RoPE OBSERVABILITY FIX ═════════════════════════════════════════
# Gap A (IB_LAYER §3c-1): `rope_row`'s output reached only the KV store path,
# so a RoPE'd q had NO host-visible egress and every "RoPE gate" was really a
# gate on the KV write port. HEAD added `l_fsrc_ext == 3` (`q_sink`,
# apex_top.sv:1186-1193): the S-2 seam's DESTINATION becomes the C-1 FEEDER.
# The fix here is host-side and needs no RTL: drive the seam with rope_en=1 +
# q_sink=1, then read the rotated row back as (fp16 row scale on the `fs`
# tap) x (INT8 codes recovered by an IDENTITY-weight GEMM over the staged
# act row). Both are produce-mode captures; both are graded against golden's
# own `quant_rows_i8(f16(q_rope_head))`.
#
# ══ SwiGLU STREAM ORDERING — RESOLVED BY READING THE CONSUMER ══════════════
# asu_swiglu.sv:106-282: ONE job = ST_GATE (cols beats, `last` on cols-1)
# THEN ST_UP (cols beats, `last` on cols-1), both arriving on the SAME
# serializer stream (apex_top.sv:1230-1231 wire gv and uv to the same
# ls_out_valid). So the order is GATE-then-UP **per 64-column job**, NOT
# whole-Wg-then-whole-Wu. Each phase is its OWN serializer frame (the frame
# IS the `last`), so a chunk costs two LJOBs. Chunking is B1's per-unit rule
# (`seq_walker_fmt.lu_chunks(d_ffn, LU_SWIGLU)` = 296 x 64), never the 12-bit
# LAYER_JOB field.
#
# ══ WHAT DOES **NOT** WORK ON TODAY'S RTL (measured, file:line) ════════════
# See BLOCKERS below and `--blockers`. Short form: the serializer's frame is
# 2040 elements (B-SER-FRAME, still true per job — but since R3 the residual
# carries a window base at LAYER_JOB[15:14], so a D_model=3584 residual IS
# done, as 4 aligned 1024-window jobs); the C-1 feeder's row length is
# frozen at 64/128, so a whole-row 3584-wide C-1 (what golden's
# `quant_rows_i8` does) is out of reach — the tile frames it as 28x128.
#
# ══ STATE (2026-07-30) AND WHERE A FOLLOW-UP PICKS IT UP ═══════════════════
# GREEN on the HEAD twin against the REAL S8 step (layer 0, step 10):
#   resid_r1   3584/3584 fp16 codes bit-exact vs the arbiter's r1 — FULL row
#              since R3 (base-windowed jobs; was 2040/2040 pre-R3)
#   resid_r2   3584/3584 bit-exact vs r2 (same)
#   resid_probe  R3 geometry-refusal guard: base+cols>DM_MAX must refuse
#              loudly (sticky + code 3); RED on pre-R3 RTL (bounded poll)
#   norm2      dm=3584 through the WIDE asu_rmsnorm; 3584/3584 C-1 codes AND
#              28/28 row scales bit-exact vs golden applied the same way
#   norm2c     (R4, 2026-07-31) the SAME dm=3584 norm on the NARROW
#              (RMS_D_MAX=128) twin — the image aws-fpga#799 lets us fly —
#              as golden rmsnorm_fx_wide's own chunk composition: 28 tile
#              sum2 exports (graded bit-exact per chunk), 27 host INT32
#              adds, then μ+rsqrt+all 3584 per-element ops ON THE TILE with
#              the broadcast r. 3584/3584 codes + 28/28 scales bit-exact.
#   norm2x     (R4) ext-only variant: sum2 arrives from OUTSIDE the norm
#              unit (sum2_mxe.py's proven MXE x·x product on the flight
#              path). Doubles as the red test: pre-R4 RTL acks the arm as a
#              reserved no-op and grades 719/3584 codes wrong (local r).
#   norm2_probe  (R4) refusal guard: illegal ext arm (k=1) must refuse
#              loudly — sticky + err_code 8, mode NOT armed, W1C clears
#   rope       128/128 codes + the row scale bit-exact vs golden's q_rope
#   swiglu     GREEN since R2 (2026-07-30): apex_top phase steering landed;
#              codes + scales bit-exact vs golden through the C-1 view
#   swiglu_probe  now the B-SWG-PHASE REGRESSION GUARD (expects NO error);
#              as the red test it went green->FAIL across the R2 fix, the
#              documented red/green evidence
# NEXT, in order:
#   1. (DONE, R2 2026-07-30) R-SWG: the B-SWG-PHASE fix — swg_up_q input-side
#      phase steering in apex_top; build_swiglu_stage went green unchanged
#      and swiglu_probe was flipped to expect_frame_error=False.
#   2. (DONE, R3 2026-07-30) R-RES: the residual base-offset landed —
#      LAYER_JOB[15:14] window base (stride 1024) in apex_residual/apex_top;
#      the resid stages emit 4 aligned window jobs and grade the FULL 3584
#      row. B-SER-FRAME stands (per-job cap), but no longer caps the CLAIM.
#   3. Wire the reused families in: attention via compute_job.build_compute_
#      job_full and the projections via gemm_job.ksplit_matvec (breadth_step
#      already samples them) so one plan covers the whole layer.
#   4. The wide C-1 (B-FEED-WIDTH) is IB_LAYER stage 6; until it lands, every
#      C-1 in this file is framed 128 wide and graded that way ON PURPOSE.
#
# CLI:
#   python3 scripts/fpga/f2/gen_layer_ops.py --selftest       # host only
#   python3 scripts/fpga/f2/gen_layer_ops.py --capture        # real 7B step
#   python3 scripts/fpga/f2/gen_layer_ops.py --smoke          # build+run+grade
#   python3 scripts/fpga/f2/gen_layer_ops.py --blockers       # the RTL fences
#
# API (the row-B2..B5 contract):
#   build_layer_step(layer_idx, step_inputs_npz, out_dir) -> dict
#   capture_layer_step(...)                               -> Path (.npz)

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif" / "top" / "l3"),
           str(REPO / "verif" / "seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_l3_vectors as g3                                    # noqa: E402
import trace_to_regops as t2r                                  # noqa: E402
import cap_decode as cd                                        # noqa: E402
import gemm_job as gj                                          # noqa: E402
import sum2_mxe as s2m                                         # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import transformer as tf                      # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

# ── the LAYER CSR window (apex_top.sv:934-945), through BAR0 window 0x1 ─────
L_CTRL, L_PTR, L_DATA, L_JOB = 0x70, 0x74, 0x78, 0x7C
L_STATUS, L_RPTR, L_RDATA, L_JOBC = 0x80, 0x84, 0x88, 0x8C
LA = {n: t2r.B_CSR + a for n, a in
      (("CTRL", L_CTRL), ("PTR", L_PTR), ("DATA", L_DATA), ("JOB", L_JOB),
       ("STATUS", L_STATUS), ("RPTR", L_RPTR), ("RDATA", L_RDATA),
       ("JOBC", L_JOBC))}

# LAYER_PTR banks (apex_top.sv:1073-1080 + :1157 res_lw_en)
BANK_PHK, BANK_PHQ, BANK_RESID = 0, 1, 2
# LAYER_JOB units (apex_top.sv:1099-1107)
LU_DEQ, LU_SWIGLU, LU_RESID = fmt.LU_DEQ, fmt.LU_SWIGLU, fmt.LU_RESID
# R3: residual window base (LAYER_JOB[15:14], stride 1024) — single source
# is the walker spec, same discipline as the LU tables.
RESID_WIN = fmt.RESID_WIN
# LAYER_STATUS bits (apex_top.sv:1128-1131)
LST_LDQ, LST_SWG, LST_RES, LST_ROPE = 0x1, 0x2, 0x4, 0x8
LST_ERR = 0x100

# ── R4 (C-RMSW chunk composition): the chunked-norm CSR pair ────────────────
# apex_top.sv R4 region (search "RMS_SUM2_ADDR"): 0x90 stages/exports sum2,
# 0x94 is the mode/arm register. The constants below MIRROR the consumer's
# refusal rule at emission (the R3 pattern: an illegal program raises HERE,
# not on the wire; LUJOBR's counterpart for deliberate refusal probes is
# nctl_raw). err_code 8 = NORM_EXT refusal.
RMS_SUM2, RMS_EXT = 0x90, 0x94
RMS_EXT_K_MIN, RMS_EXT_K_MAX = 2, 64      # μ-table envelope (asu_wide_rms_params.svh)
LERR_NORM_EXT = 8

D_TILE = gj.D_TILE                 # 128 — feeder/stage/MXE row width
BPR = gj.BPR                       # 16
MXE_N = gj.MXE_N                   # 8
ACT_BANK = gj.ACT_BANK             # 1 (bank0 row0 = the loader token)
F16_ONE = gj.F16_ONE

# ── the measured RTL fences (every one a file:line, none extrapolated) ─────
SER_BEATS_MAX = 255                # apex_lane32_ser.sv:34 job_beats is [7:0];
SER_LANES_MAX = 8                  # apex_top.sv:1675 BEATS_MAX(255), :261
SER_FRAME_MAX = SER_BEATS_MAX * SER_LANES_MAX          # = 2040 elements
FEEDER_D_LEGAL = (64, 128)         # seam_feeder_quant.sv:100-102

BLOCKERS = [
    dict(id="B-SER-FRAME", stage="deq / residual / swiglu",
         what="one serializer frame is at most 255 beats x 8 lanes = 2040 "
              "elements, so no LAYER unit can be handed a single job wider "
              "than 2040",
         where=["rtl/top/glue/apex_lane32_ser.sv:34  input logic [7:0] job_beats",
                "rtl/top/glue/apex_lane32_ser.sv:26  parameter BEATS_MAX = 255",
                "rtl/top/apex_top.sv:261             input logic [7:0] lj_beats",
                "rtl/top/apex_top.sv:1675            apex_lane32_ser #(.BEATS_MAX(255))"],
         needs="widen job_beats (and lj_beats) past 8 bits, or give the "
               "LAYER units a multi-frame job mode"),
    dict(id="B-RES-BASE", stage="residual-1 / residual-2 — FIXED IN RTL (R3, 2026-07-30)",
         what="FIXED: LAYER_JOB[15:14] is now the residual window base "
              "(stride 1024, apex_residual jb_base): element i updates "
              "row[base*1024+i], and base*1024+cols > DM_MAX REFUSES at "
              "accept (frame_error class, err_code 3 — resid_probe guards "
              "it). A DM=3584 residual = 4 aligned window jobs; the resid "
              "stages emit and grade the FULL row. THE DEFECT WAS: every "
              "job wrote from index 0 with no base field, so with "
              "B-SER-FRAME the reachable window was row[0:2040] in any "
              "number of passes.",
         where=["rtl/top/glue/apex_residual.sv  jb_base port + eff_idx",
                "rtl/top/apex_top.sv            l_push_base / lj_base_q",
                "verif/seq_walker/seq_walker_fmt.py  RESID_WIN + base rule"],
         needs="nothing — fixed. NOTE the walker LU channel carries no "
               "base: WALKED residual jobs stay window-0 by construction "
               "(apex_top pins l_push_base=0 in walk mode); host-path only."),
    dict(id="B-STAGE-ROWS", stage="SwiGLU readback (choreography, not a defect)",
         what="the act stage bank holds STAGE_R_MAX=16 rows and the aj "
              "descriptor's row field saturates near 31, so staging one "
              "feeder row per swiglu output row only works up to ~14 rows "
              "(28 chunks). MEASURED 2026-07-31 by the 296-chunk pre-flight "
              "gate: chunks=32 PASS, chunks=64 FAIL, and the 296-chunk run "
              "wedged at aj row 32. THE TRAP: below the ceiling the excess "
              "rows ALIAS (row field wraps into the bank), and because the "
              "identity readback aliases IDENTICALLY, the grade stayed GREEN "
              "— a silent wrong-answer shape that only shows up past the "
              "wrap. Fixed in the emitter by staging in GROUPS of 8 into "
              "fixed bank rows 2..9, reading each group back before reuse.",
         where=["rtl/top/apex_top.sv:136   STAGE_R_MAX = 16 (rows/bank)",
                "rtl/top/apex_top.sv:1885  apex_stage_buf #(.R_MAX(STAGE_R_MAX)) u_astage",
                "rtl/top/glue/apex_stage_buf.sv:64  parameter R_MAX = 16"],
         needs="nothing for the emitter (group-and-reuse). An RTL widening "
               "would only be needed if a single job had to stage >16 rows "
               "live at once, which no layer op requires."),
    dict(id="B-FEED-WIDTH", stage="RMSNorm-2 C-1 / SwiGLU C-1 / o-proj C-1",
         what="the C-1 feeder's row length is a parameter frozen at 64 or "
              "128, and apex_top hands it CFG_DM which cl_apex never "
              "overrides — so a whole-row 3584-wide C-1 (golden's "
              "quant_rows_i8 over the full row, ONE amax, ONE scale) is not "
              "reachable; the tile frames the same values as 28 x 128 rows "
              "with 28 independent scales",
         where=["rtl/seam/seam_feeder_quant.sv:67   parameter D = 64  (row length)",
                "rtl/seam/seam_feeder_quant.sv:100  $error(\"D must be 64 or 128\")",
                "rtl/top/apex_top.sv:1553           seam_feeder_quant #(.D(CFG_DM))",
                "rtl/top/apex_top.sv:22             CFG_DM default = CFG_D; cl_apex.sv:592 never sets it"],
         needs="IB_LAYER.md stage 6 wide-feeder elaboration (named in "
               "apex_top.sv:20-22 as the pending work)"),
    dict(id="B-STAGE-PORT", stage="SwiGLU / RoPE (choreography, not a defect)",
         what="the act stage buffer has ONE job FSM, so a stage cannot hold a "
              "LOAD job (sinking the feeder) while EMITting the rows that "
              "drive the serializer — the AJ EMIT job-fire never goes ready. "
              "seam_feeder_quant buffers a whole row before emitting, so the "
              "legal order is arm-feeder / inject / LOAD, per row. Measured "
              "as a `jf stall` on the AJ EMIT, then fixed in the emitter.",
         where=["rtl/top/glue/apex_stage_buf.sv (single job port)",
                "rtl/seam/seam_feeder_quant.sv:262  logic [31:0] rbuf [D]  (pass1 buffers the row)"],
         needs="nothing — the emitter interleaves; recorded so the next "
               "author does not re-derive it from a 2-million-cycle stall"),
    dict(id="B-FS-FIFO", stage="wide RMSNorm-2 (choreography, not a defect)",
         what="the mailbox fs capture FIFO is shallower than the 28 rows a "
              "dm=3584 C-1 produces. Pushing all 3584 gamma words before "
              "popping any scale wedges the tile at ~19 rows: the FIFO fills, "
              "seam_feeder_quant's ST_SPUSH blocks on scl_ready, and that "
              "back-pressures asu_rmsnorm's gamma port. MEASURED as a pw "
              "stall on XG element ~2451. Fixed in the emitter by draining "
              "one scale per 128 gamma words.",
         where=["rtl/seam/seam_feeder_quant.sv:294  ST_SPUSH: if (sc_ready) ...",
                "scripts/fpga/f2/cl_apex/design/apex_f2_mailbox.sv  fs capture FIFO"],
         needs="nothing — the emitter interleaves; recorded so the next "
               "author does not re-derive it from a stall"),
    dict(id="B-SWG-PHASE", stage="SwiGLU — FIXED IN RTL (R2, 2026-07-30)",
         what="FIXED: apex_top now steers each serializer beat to exactly "
              "the ACTIVE phase's skid (swg_up_q input-side phase tracker; "
              "serializer gated by that phase's ready only); the swiglu "
              "stage is bit-exact vs golden and swiglu_probe guards the fix. "
              "THE DEFECT WAS: asu_swiglu takes its GATE and UP phases on two SEPARATE "
              "skid-buffered ports, but apex_top drives both from the SAME "
              "serializer valid and gates the serializer with the OR of the "
              "two SKIDS' readies. stream_skid's s_ready is `~skid_valid` "
              "(registered, independent of m_ready), so whenever both skids "
              "have room BOTH latch the same beat: the UP skid swallows the "
              "first 2 GATE beats and the GATE skid the first 2 UP beats. "
              "The UP phase then sees 2 stale values first and its `last` "
              "lands 2 beats early -> frame_error, job aborted, the chunk's "
              "products are wrong AND lost. MEASURED on the HEAD twin: an "
              "exactly-framed gate(64)/up(64) chunk returns LAYER_STATUS "
              "0x700 = sticky + err_code 3 (FRAME); the same at cols=8; a "
              "gate frame 8 beats short hangs the tile instead (pw stall on "
              "the weight FIFO). `--probe-swiglu` reproduces all three.",
         where=["rtl/top/apex_top.sv:1230  assign swg_gv = ls_out_valid && (l_ser_dst_q == 2'd2);",
                "rtl/top/apex_top.sv:1231  assign swg_uv = ls_out_valid && (l_ser_dst_q == 2'd2);",
                "rtl/top/apex_top.sv:1918  2'd2: ls_out_ready = swg_gr | swg_ur;",
                "rtl/asu/asu_swiglu.sv:74-85   the two input skids (gr/ur are the SKIDS' readies)",
                "rtl/xbr/stream_skid.sv:81     s_ready = ~skid_valid (registered)"],
         needs="nothing — fixed in apex_top (R2). NOTE the fix is NOT the "
               "FSM-state export this row once suggested: the FSM runs up to "
               "2 skid beats behind the input boundary, so steering must "
               "track the INPUT-side phase (accepted `last` beats), which is "
               "what swg_up_q does. Recorded so the OR-of-readies gate is "
               "not reintroduced; swiglu_probe (expect NO error) guards it."),
    dict(id="B-SWG-EGRESS", stage="SwiGLU",
         what="asu_swiglu's fp16 product stream has exactly ONE consumer, "
              "the feeder; with l_fsrc_ext != 2 the unit stalls forever. So "
              "the fp16 products themselves have no egress: they are "
              "observable only through the C-1 view (exact fp16 row scale + "
              "INT8 codes)",
         where=["rtl/top/apex_top.sv:1232  assign swg_pr = (l_fsrc_ext_q == 2'd2) && fq_in_ready;"],
         needs="a swiglu-p tap on the mailbox, or an ls-style egress FIFO"),
    dict(id="B-ROPE-EGRESS", stage="RoPE",
         what="RESOLVED IN HOST SOFTWARE (B5). rope_row's output reaches "
              "only the KV write port unless l_fsrc_ext=3 (q_sink) steers "
              "the S-2 seam into the feeder; with q_sink the rotated row is "
              "recoverable as fs-scale x identity-GEMM codes. No RTL change.",
         where=["rtl/top/apex_top.sv:1186  assign q_sink = (l_fsrc_ext_q == 2'd3);",
                "rtl/top/apex_top.sv:1193  assign seam_ready = q_sink ? fq_in_ready : kv_s_tready;"],
         needs="nothing — this row is the FIX, recorded so it is not "
               "re-opened as a blocker"),
]

# ── produce-mode: which capture sites become `cap` (no expectation) ─────────
RO_SEMS = frozenset({"ro_meta"} | {f"ro_w{j}" for j in range(MXE_N)})
# The RO result lanes/meta and LAYER_RDATA are the addresses through which a
# stage's OWN OUTPUT leaves the tile: an expectation on either would put the
# answer in the program that computes it. The fs/ss scale taps are NOT on this
# list — gemm_job.py's rationale applies verbatim: outside a produce window
# they are INPUT-derived checks (the loader's real RMSNorm feeder scale, the
# staging rows' fp16-1.0 identity), and inside one the PMODE rewriting already
# turns them into expectation-free caps.
# R4: RMS_SUM2's read view is the norm unit's OWN sum-of-squares leaving the
# tile — an output address exactly like RDATA/the RO lanes (the arm WRITE
# side is host input and is not read back checked anywhere).
OUTPUT_ADDRS = frozenset(list(gj.RESULT_ADDRS)
                         + [LA["RDATA"], t2r.B_CSR + RMS_SUM2])

DEFAULT_OUT = REPO / "build" / "f2_layer"
S8_RUN = REPO / "docs/results/s8_7b_token/artifact_trace/run.json"
S8_WEIGHTS = REPO / "build/s8_weights/Qwen2.5-7B-4bit"


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ═══════════════ 1. the LAYER script tokens (row B2) ═══════════════════════

class LayerScript(g3.Script):
    """gen_l3_vectors.Script + the LAYER-window vocabulary.

    Every token is a THIN name for one BAR0 access; the choreography rules
    they encode are the frozen §3b ones, replayed from the walker spec
    (verif/seq_walker/gen_layer_trace.py steps_tail):
      rule 2  consumer-before-producer  — push the RESIDUAL job before the
              deq JOBC+JOB in a chained deq->residual flow
      rule 3  a JOB push RESETS the JOBC index (apex_top.sv:1097), so the
              JOBC words land IMMEDIATELY before THEIR push
    """

    def lctl(self, **kw):
        """LAYER_CTRL level word — built by seq_walker_fmt.lctl, never here."""
        w = fmt.lctl(**kw)
        self.emit("LCTL", f"{w:04x}")
        return w

    def lptr(self, bank, idx):
        self.emit("LPTR", f"{bank:x}", f"{idx:04x}")

    def ldata(self, vals):
        vals = [int(v) & 0xFFFF for v in np.asarray(vals).ravel()]
        for i in range(0, len(vals), 64):          # 64 per line: readable
            blk = vals[i:i + 64]
            self.emit("LDATAN", f"{len(blk):x}", *[f"{v:04x}" for v in blk])

    def lujob(self, unit, cols, base=0):
        # B1's rule, applied at EMISSION: the consuming unit's own COLS_MAX,
        # never the 12-bit field. A violation raises here, not on the wire.
        # R3: `base` is the residual window base ([15:14], stride 1024);
        # emitted as a 4th token only when nonzero so every pre-R3 stage
        # stays byte-identical.
        fmt.assert_layer_job_legal(unit, cols, "LayerScript.lujob", base=base)
        self.emit("LUJOB", f"{unit:x}", f"{cols:03x}",
                  *([f"{base:x}"] if base else []))

    def lujob_raw(self, unit, cols, base=0):
        """R3 refusal-probe ONLY: push a LAYER_JOB with NO emission-side
        legality check — the whole point is to watch the RTL refuse it."""
        self.emit("LUJOBR", f"{unit:x}", f"{cols:03x}", f"{base:x}")

    def lstatw(self, val):
        """W1C write to LAYER_STATUS (bit 8 clears the error sticky)."""
        self.emit("LSTATW", f"{val:08x}")

    def ljc(self, comp):
        assert (comp & 0x1FFF) == 0 and (comp >> 31) == 0 \
            and 0 < ((comp >> 23) & 0xFF) < 255, \
            f"JOBC composite {comp:#010x} is not positive-normal fp16-grade " \
            f"(apex_layer_deq.sv:90-92 / asu_swiglu.sv:97-104 would refuse)"
        self.emit("LJC", f"{comp:08x}")

    def lrptr(self, idx):
        self.emit("LRPTR", f"{idx:04x}")

    # ── R4 chunked-norm tokens (consumer rule: apex_top R4 region) ──────────
    def nsum2(self, v):
        """Stage the accumulated wide-row sum2 into RMS_SUM2 (0x90)."""
        v = int(v)
        assert 0 <= v < (1 << 28), \
            f"ext sum2 {v} does not fit [27:0] (apex_top refuses [31:28]!=0)"
        self.csrw(RMS_SUM2, v)

    def nctl(self, *, sum_en=0, ext=0, k=0, clr=0):
        """RMS_EXT (0x94) mode/arm word, legality mirrored at emission."""
        assert not (sum_en and ext), "sum_en and ext are exclusive modes"
        if ext:
            assert RMS_EXT_K_MIN <= k <= RMS_EXT_K_MAX, \
                f"ext k={k} outside [{RMS_EXT_K_MIN},{RMS_EXT_K_MAX}] " \
                f"(μ-table envelope) — apex_top would refuse (code 8)"
        else:
            assert k == 0, "k is only meaningful with ext=1"
        w = ((1 if clr else 0) << 9) | (int(k) << 2) \
            | ((1 if ext else 0) << 1) | (1 if sum_en else 0)
        self.csrw(RMS_EXT, w)

    def nctl_raw(self, w):
        """R4 refusal-probe ONLY (the LUJOBR idiom): push an RMS_EXT word
        with NO emission-side check — the point is to watch the RTL refuse."""
        self.csrw(RMS_EXT, int(w))

    def npoll_cnt(self, n):
        """Poll the pass-1 capture count (RMS_EXT[23:16]) to n."""
        assert 0 <= n <= 255
        self.csrp(RMS_EXT, 0x00FF0000, n << 16)

    def ns2cap(self):
        """ONE produce-mode capture of RMS_SUM2 — the chunk's sum2 export."""
        self.emit("NS2C")

    def erd(self, n):
        """n SINGLE-READ produce-mode captures of LAYER_RDATA (see header)."""
        self.emit("ERD", f"{n:x}")

    def lpoll(self, mask, exp):
        self.emit("LPOLL", f"{mask:08x}", f"{exp:08x}")

    def lstat(self, mask, exp):
        self.emit("LSTAT", f"{mask:08x}", f"{exp:08x}")
        self.n_expect += 1

    def xrow_n(self, row):
        """xa row of ANY length (last on the final element). The stock XR
        token frames at Xlate.D=128; the wide RMSNorm needs 3584."""
        r = [int(v) & 0xFF for v in np.asarray(row).ravel()]
        self.emit("XRN", f"{len(r):x}", *[f"{v:02x}" for v in r])

    def grow_n(self, g):
        r = [int(v) & 0xFFFF for v in np.asarray(g).ravel()]
        self.emit("GRN", f"{len(r):x}", *[f"{v:04x}" for v in r])

    def pmode(self, on):
        """Toggle produce-mode rewriting of the RESULT/scale-tap sites."""
        self.emit("PMODE", 1 if on else 0)


class LayerXlate(t2r.Xlate):
    """Xlate + the LAYER tokens + produce-mode capture rewriting.

    Produce mode (PMODE 1) rewrites the RO lanes/meta and the fs/ss scale
    taps into `cap` ops with NO expectation, exactly as
    compute_job/gemm_job do for their own results: a stage's OUTPUT may
    never be foreknown by the program that produces it. PMODE 0 restores
    the counted-check behaviour, which the loader token and the phase-A
    sanity block keep (they are INPUT-derived).
    """

    def __init__(self, D: int, *, erd_check: bool = False):
        super().__init__(D)
        self.pmode = False
        self.erd_check = erd_check
        self.n_capsites = 0
        self.n_erd = 0
        self.erd_expect: list[int] = []

    def r(self, a, m, e, sem="r"):
        if self.pmode and (sem in RO_SEMS or sem in ("fs", "ss")):
            self.cap(a, m, sem)
            self.n_capsites += 1
            return
        super().r(a, m, e, sem=sem)

    def run(self, lines):
        rest = []
        for ln in lines:
            t = ln.split()
            if not t or not t[0].isupper():
                rest.append(ln)
                continue
            op, args = t[0], t[1:]
            if op == "PMODE":
                self.pmode = bool(int(args[0]))
            elif op == "LCTL":
                self.w(LA["CTRL"], int(args[0], 16))
            elif op == "LPTR":
                self.w(LA["PTR"], (int(args[0], 16) << 28) | int(args[1], 16))
            elif op == "LDATAN":
                for v in args[1:]:
                    self.w(LA["DATA"], int(v, 16))
            elif op == "LUJOB":
                unit, cols = int(args[0], 16), int(args[1], 16)
                base = int(args[2], 16) if len(args) > 2 else 0
                fmt.assert_layer_job_legal(unit, cols, "LayerXlate.LUJOB",
                                           base=base)
                self.w(LA["JOB"], (base << 14) | (unit << 12) | cols)
            elif op == "LUJOBR":
                # R3 refusal probe: deliberately illegal push, no assert —
                # the checked LAYER_STATUS read after it IS the expectation.
                unit, cols, base = (int(a, 16) for a in args[:3])
                self.w(LA["JOB"], (base << 14) | (unit << 12) | cols)
            elif op == "LSTATW":
                self.w(LA["STATUS"], int(args[0], 16))
            elif op == "LJC":
                self.w(LA["JOBC"], int(args[0], 16))
            elif op == "LRPTR":
                self.w(LA["RPTR"], int(args[0], 16))
            elif op == "ERD":
                n = int(args[0], 16)
                for _ in range(n):
                    if self.erd_check:
                        # bring-up shape: still ONE read, but checked.
                        e = self.erd_expect[self.n_erd] \
                            if self.n_erd < len(self.erd_expect) else 0
                        self.ops.append({"op": "r", "a": LA["RDATA"],
                                         "m": 0xFFFF, "e": e & 0xFFFF})
                        self.n_checks += 1
                    else:
                        self.cap(LA["RDATA"], 0xFFFF, "lrd")
                    self.n_erd += 1
            elif op == "LPOLL":
                self.poll(LA["STATUS"], int(args[0], 16), int(args[1], 16))
            elif op == "LSTAT":
                self.r(LA["STATUS"], int(args[0], 16), int(args[1], 16),
                       sem="lstat")
            elif op == "XRN":
                n = int(args[0], 16)
                vals = args[1:1 + n]
                assert len(vals) == n, f"XRN {n} got {len(vals)}"
                for j, b in enumerate(vals):
                    last = 1 if j == n - 1 else 0
                    self.pw(t2r.B_MB + t2r.XA, (last << 8) | int(b, 16))
            elif op == "GRN":
                n = int(args[0], 16)
                for gt in args[1:1 + n]:
                    self.pw(t2r.B_MB + t2r.XG, int(gt, 16))
            elif op == "NS2C":
                # R4 pass-1 export: one read of RMS_SUM2, produce-mode by
                # construction (the ERD argument verbatim: a chunk's sum2 is
                # the tile's OWN output and may never be foreknown).
                self.cap(t2r.B_CSR + RMS_SUM2, 0x0FFFFFFF, "rs2")
                self.n_capsites += 1
            else:
                rest.append(ln)
                continue
            # a LAYER token consumed this line; flush anything queued first
            if rest:
                super().run(rest)
                rest = []
                # the LAYER op was already appended above, so re-order it to
                # the end of the queue-flushed ops
                self.ops.append(self.ops.pop(len(self.ops) - 1 - 0))
        if rest:
            super().run(rest)


def _translate(script: LayerScript, *, erd_check=False, note=None) -> LayerXlate:
    """Run the op script through the LAYER translator, IN ORDER.

    LayerXlate.run's interleave bookkeeping is subtle enough that the
    emitter never relies on it: it drives line-by-line instead, so the
    regops order is EXACTLY the script order.
    """
    x = LayerXlate(D_TILE, erd_check=erd_check)
    if note:
        x.note(note)
    for ln in script.lines:
        x.run([ln])
    return x


# ═══════════════ 2. injection: arbitrary INT32 onto the serializer ═════════
#
# Every LAYER unit consumes the SERIALIZER's INT32 element stream (apex_top
# .sv:1212/1230). The host's only ingress for that stream is a real MXE job,
# so a value v is produced as an actual GEMM: v = SUM_k a_k * w_k over the
# staged activation row a. Two staged rows are used, and the choice is
# RANGE-DRIVEN and asserted, never silent:
#
#   K2   the L3 loader token a = [127, 1, 0...] (gen_l3_vectors.loader_phase)
#        v = 127*w0 + w1                    exact for |v| <= 16256
#   K128 a 128-lane row a = [127, 1, 127 x126] staged through squant
#        MODE_QUANT (amax 127 => the identity, gemm_job's F1/F2 argument)
#        v = 127*(w0 + w2 + ... + w127) + w1  exact for |v| <= 2048446
#
# Anything wider REFUSES with the exact bound — a truncated accumulator is a
# silent wrong answer and this lane has had two of those already.

INJ_K2_MAX = 127 * 127 + 127                    # 16256 (w0,w1 in [-128,127])
INJ_K128_LANES = D_TILE - 1                     # 127 x 127-weighted lanes
INJ_K128_MAX = 127 * (127 * INJ_K128_LANES) + 63          # 2048446


class InjectRangeError(AssertionError):
    """A value the injection path cannot represent EXACTLY."""


def _k2_pair(v: int):
    w0 = int(round(v / 127.0))
    w0 = max(-128, min(127, w0))
    w1 = v - 127 * w0
    if not (-128 <= w1 <= 127):
        raise InjectRangeError(
            f"inject: |v|={abs(v)} exceeds the K=2 loader-row bound "
            f"{INJ_K2_MAX} (v={v}, w0={w0}, w1={w1})")
    return w0, w1


def _k128_col(v: int):
    """Weight column for the K=128 staged row a = [127, 1, 127 x126]."""
    q = int(round(v / 127.0))
    r = v - 127 * q
    if not (-127 <= r <= 127):                  # |r| <= 63 by construction
        raise InjectRangeError(f"inject: residue {r} out of int8")
    lanes = [0] * D_TILE
    lanes[1] = r
    rem, i = q, 0
    order = [0] + list(range(2, D_TILE))        # the 127-weighted lanes
    while rem != 0:
        if i >= len(order):
            raise InjectRangeError(
                f"inject: |v|={abs(v)} exceeds the K=128 staged-row bound "
                f"{INJ_K128_MAX}")
        step = max(-127, min(127, rem))
        lanes[order[i]] = step
        rem -= step
        i += 1
    return lanes


def inject_kind(values) -> str:
    a = int(np.max(np.abs(np.asarray(values, dtype=np.int64))), )
    if a <= INJ_K2_MAX:
        return "k2"
    if a <= INJ_K128_MAX:
        return "k128"
    raise InjectRangeError(
        f"inject: |v|max={a} exceeds every staged-row bound "
        f"(K=2 {INJ_K2_MAX}, K=128 {INJ_K128_MAX}). Widen the staged row "
        f"(a K<=2048 stage reaches {127 * 127 * 2047 + 63}) before claiming "
        f"this stage — do NOT truncate.")


def stage_inject_row(s: LayerScript, row: int = 1, bank: int = ACT_BANK):
    """Stage the K=128 injection row through squant MODE_QUANT.

    a = [127, 1, 127 x126]: amax is exactly 127, so gemm_job's F1/F2
    identity argument holds and the emitted row scale is fp16 1.0 — kept a
    COUNTED check (it is input-derived, not a result).
    """
    a = [127, 1] + [127] * (D_TILE - 2)
    q8, sb = at.quant_rows_i8(np.asarray(a, dtype=np.float64)[None, :])
    assert np.array_equal(np.asarray(q8[0], dtype=np.int64),
                          np.asarray(a, dtype=np.int64)), "inject row identity"
    assert int(sb[0]) == F16_ONE, "inject row scale != fp16 1.0"
    s.emit(f"// stage the K=128 injection row a=[127,1,127x126] -> "
           f"act bank{bank} row{row} (MODE_QUANT identity, scale fp16 1.0)")
    s.route(rdst=1, asrc=1)
    pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(
        np.asarray(a, dtype=np.float64))]
    g3.inject_jobs(s, pairs, 1)
    s.ess(F16_ONE, 1)
    s.aj(0, bank, 0, 1, s.BPR, row)
    return a


def inject_frame(s: LayerScript, values, *, kind=None, inj_row=1,
                 inj_bank=ACT_BANK):
    """Drive `values` INT32 elements onto the serializer output as ONE frame.

    ONE frame == one `last` == one LAYER-unit job's worth of data. Frame
    legality is the serializer's own (apex_lane32_ser.sv:110): 1 <= beats
    <= 255, 1 <= lanes <= 8.
    """
    v = [int(x) for x in np.asarray(values, dtype=np.int64).ravel()]
    n = len(v)
    if n % MXE_N:
        raise AssertionError(f"inject_frame: {n} values is not a multiple of "
                             f"{MXE_N} (one MXE result beat)")
    beats = n // MXE_N
    if not 1 <= beats <= SER_BEATS_MAX:
        raise AssertionError(
            f"inject_frame: {n} elements = {beats} beats exceeds the "
            f"serializer frame bound {SER_FRAME_MAX} (B-SER-FRAME)")
    kind = kind or inject_kind(v)
    s.emit(f"// serializer frame: {n} INT32 elements ({beats} beats x 8 "
           f"lanes), injection kind={kind}")
    s.ljob(beats, MXE_N)
    for j in range(beats):
        blk = v[MXE_N * j:MXE_N * (j + 1)]
        if kind == "k2":
            packed = []
            for x in blk:
                w0, w1 = _k2_pair(x)
                packed.append((w0 & 0xFF) | ((w1 & 0xFF) << 8))
            s.aj(1, 0, 0, 1, 1, 0)                    # loader row, 1 beat
            s.desc(g3.desc_words(g3.OP_GEMM_WS, 1, 2, MXE_N))
            s.wbeats(packed)
        else:
            W = np.zeros((D_TILE, MXE_N), dtype=np.int64)
            for c, x in enumerate(blk):
                W[:, c] = _k128_col(x)
            s.aj(1, inj_bank, 0, 1, s.BPR, inj_row)
            s.desc(g3.desc_words(g3.OP_GEMM_WS, 1, D_TILE, MXE_N))
            s.wbeats(g3.wgt_beats_ws(W, 0))
    return beats


def identity_readback(s: LayerScript, row, base, *, bank=ACT_BANK):
    """Read 8 INT8 codes of a staged act row back through an IDENTITY GEMM.

    acc[n] = SUM_k x8[k]*W[k][n] with W[k][n] = [k == base+n]  =>  acc[n] =
    x8[base+n], exactly. The RO lanes are the ONLY egress the act stage has,
    and in produce mode they are `cap`s — so the codes are never in the
    program that recovers them.
    """
    assert 0 <= base and base + MXE_N <= D_TILE
    W = np.zeros((D_TILE, MXE_N), dtype=np.int64)
    for n in range(MXE_N):
        W[base + n][n] = 1
    s.aj(1, bank, 0, 1, s.BPR, row)
    s.desc(g3.desc_words(g3.OP_GEMM_WS, 1, D_TILE, MXE_N))
    s.wbeats(g3.wgt_beats_ws(W, 0))
    s.ero([0] * MXE_N, 1)


# ═══════════════ 3. the stage programs ═════════════════════════════════════

def _prologue(s: LayerScript):
    g3.phase_a(s, tiers_used=(0,))
    g3.loader_phase(s)


def _drain_and_check(s: LayerScript, *, extra_stk=0):
    s.emit("// quiescence + LAYER error accounting (no unit may have "
           "refused: LAYER_STATUS[8] sticky must be clear)")
    s.csrp(g3.CSR["STATUS"], 0x1, 0x1)
    s.lpoll(LST_LDQ | LST_SWG | LST_RES | LST_ROPE, 0x0)
    s.lstat(LST_ERR, 0x0)
    g3.final_phase(s, 8, extra_stk=extra_stk)


def build_resid_stage(step, *, which: str, cols: int, out_dir: Path,
                      name: str, erd_check=False, off: int = 0):
    """SEGMENT 1/tail: o8 -> layer-deq -> residual -> RDATA (rows B3/B4).

    which='r1' : residual row preloaded with f16(X[T]), stream = the o-proj
                 o8 codes, JOBC = the graded s_out of the o-proj epilogue.
                 Golden expectation: r.r1.
    which='r2' : residual row preloaded with f16(r1), stream = the down-proj
                 o8 codes, JOBC = the graded s_out of the down epilogue.
                 Golden expectation: r.r2.
    """
    assert which in ("r1", "r2")
    # `off` (2026-07-31): the residual is ELEMENTWISE — row[i] <- f16(row[i]
    # + b[i]) with NO cross-element state (apex_residual: cnt indexes each
    # beat independently; golden transformer.py:23/:28 r = f16(x + y)). So
    # the full D_model row can be computed as independent aligned slices,
    # each BIT-IDENTICAL to what one wide pass would produce for those
    # elements. This is what lets a DM_MAX=128 image compute the FULL 3584
    # residual in 28 passes — the host only reloads the row slice between
    # passes (data movement, no arithmetic).
    sl = slice(off, off + cols)
    o8 = np.asarray(step[f"{which}_o8"], dtype=np.int64)[sl]
    comp = int(step[f"{which}_comp"])
    row0 = np.asarray(step[f"{which}_row_in_bits"], dtype=np.uint16)[sl]
    exp = np.asarray(step[f"{which}_bits"], dtype=np.uint16)[sl]
    assert len(o8) == cols, f"slice {sl} short: {len(o8)} != {cols}"
    # R3: one job per 1024-aligned window (base = LAYER_JOB[15:14]); every
    # window is <= RESID_WIN <= the 2040-element serializer frame, so the
    # full DM=3584 row is covered in 4 passes — B-RES-BASE lifted.
    # NB the window offsets below are WITHIN THE SLICE (the row RAM holds
    # exactly this slice); `off` is the slice's offset INTO THE D_model ROW
    # and must NOT be shadowed by them — it is what the grader slices by.
    wins = [(w // RESID_WIN, w, min(RESID_WIN, cols - w))
            for w in range(0, cols, RESID_WIN)]

    s = LayerScript(D_TILE)
    s.emit(f"// LAYER stage {name}: deq -> residual({which}) -> RDATA, "
           f"cols={cols} as {len(wins)} aligned window job(s) (R3 base "
           f"field). Choreography = the frozen §3b order "
           f"(verif/seq_walker/gen_layer_trace.py steps_tail), PER WINDOW: "
           f"residual job pushed BEFORE the deq JOBC+JOB "
           f"(consumer-before-producer), the JOBC word IMMEDIATELY before "
           f"ITS push (a push resets the index).")
    _prologue(s)
    s.emit(f"// preload the residual row RAM (LAYER_PTR bank 2 + LAYER_DATA, "
           f"auto-inc) with the fp16 row this residual adds onto")
    s.lptr(BANK_RESID, 0)
    s.ldata(row0)
    s.lctl(ser_dst=1, resid_arm=1)
    s.route(rdst=1)
    s.pmode(True)
    for base, woff, wcols in wins:
        s.emit(f"// window {base}: row elements "
               f"[{off + woff}:{off + woff + wcols}] of the D_model row "
               f"(slice-local [{woff}:{woff + wcols}]) — o8 stream: "
               f"MXE res -> serializer -> layer-deq -> residual")
        s.lujob(LU_RESID, wcols, base=base)
        s.ljc(comp)
        s.lujob(LU_DEQ, wcols)
        inject_frame(s, o8[woff:woff + wcols])
        s.lpoll(LST_LDQ | LST_RES, 0x0)
    s.emit(f"// read the FULL updated row back: ONE bus read per element "
           f"(LAYER_RDATA auto-increments LAYER_RPTR)")
    s.lrptr(0)
    s.erd(cols)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, erd_check=erd_check,
                   note=f"LAYER {name}: deq+residual({which}) cols={cols} "
                        f"({len(wins)} windows)")
    if erd_check:
        pass
    return _emit(x, out_dir, name, dict(
        stage=f"resid_{which}", kind="deq->residual->RDATA", cols=int(cols),
        off=int(off),
        windows=[[int(b), int(o), int(w)] for b, o, w in wins],
        jobc=f"{comp:#010x}", inject=inject_kind(o8),
        expect_sem="lrd", n_expect=int(cols))), exp


def build_resid_probe(step, *, out_dir: Path, name: str):
    """R3 GEOMETRY-REFUSAL GUARD: a residual job whose window footprint
    exceeds the row RAM (base=3, cols=1024 -> 3072+1024=4096 > DM_MAX=3584)
    must be REFUSED loudly at accept — frame_error class, LAYER_STATUS
    sticky + err_code 3 — and leave every unit idle. On pre-R3 RTL the base
    bits were ignored: the same push became a LEGAL cols=1024 job, silently
    accepted at window 0 (sticky clear, residual unit busy) — this stage's
    first checked LAYER_STATUS read goes RED there, which is the red half of
    the R3 red/green evidence. The W1C + re-read then prove the new error
    path clears like every other LAYER error."""
    del step                                  # geometry-only: no operands
    s = LayerScript(D_TILE)
    s.emit("// R3 refusal probe: LUJOBR resid cols=1024 base=3 — footprint "
           "4096 > DM_MAX. EXPECT: sticky + err_code 3, all units idle, "
           "then W1C clears it.")
    _prologue(s)
    s.lujob_raw(LU_RESID, RESID_WIN, base=3)
    s.emit("// wait for the refusal (bounded: POLL_LIMIT then loud fail — "
           "on pre-R3 RTL the sticky never sets and this stage goes RED "
           "here)")
    s.lpoll(LST_ERR, LST_ERR)
    s.lstat(0x1F00 | LST_LDQ | LST_SWG | LST_RES,
            (3 << 9) | LST_ERR)              # sticky+code 3, units idle
    s.lstatw(LST_ERR)                        # W1C
    # NOTE: only the STICKY clears on W1C — err_code holds its last value
    # until the next error (apex_top: l_err_code_q has no clear term), so
    # the final check masks the sticky bit alone.
    s.lstat(LST_ERR, 0x0)                    # sticky clear (code holds 3)
    s.emit("DONE")
    x = _translate(s, note=f"LAYER {name}: R3 geometry-refusal guard")
    return _emit(x, out_dir, name, dict(
        stage="resid_probe", kind="R3 geometry-refusal guard",
        cols=int(RESID_WIN), base=3,
        expect="LAYER_STATUS sticky + err_code 3 (FRAME class), then W1C"))


def build_swiglu_stage(step, *, chunks: int, out_dir: Path, name: str):
    """The SwiGLU tail (row B4).

    Per 64-column chunk (B1's per-unit rule): JOBC gate, JOBC up, LUJOB 1,
    then TWO serializer frames — GATE then UP — because asu_swiglu's ST_GATE
    phase ends on ITS frame's `last` and ST_UP on the next one
    (asu_swiglu.sv:236-271). The fp16 products leave through the C-1 feeder
    (the ONLY consumer, apex_top.sv:1232 — see B-SWG-EGRESS) and are
    recovered as (fs row scale) x (identity-GEMM INT8 codes).
    """
    C = fmt.SWG_COLS_MAX                                    # 64
    n = chunks * C
    assert n % D_TILE == 0, "swiglu readback frames the feeder at 128/row"
    acc_g = np.asarray(step["acc_g"], dtype=np.int64)[:n]
    acc_u = np.asarray(step["acc_u"], dtype=np.int64)[:n]
    comp_g, comp_u = int(step["comp_g"]), int(step["comp_u"])
    rows = n // D_TILE

    s = LayerScript(D_TILE)
    s.emit(f"// LAYER stage {name}: SwiGLU, {chunks} x {C}-column jobs "
           f"({n} of d_ffn={int(step['d_ffn'])}). cols is the CONSUMING "
           f"UNIT's bound (asu_swiglu COLS_MAX=64 @ apex_top.sv:1234), NOT "
           f"the 12-bit LAYER_JOB field — B1/D-029-erratum rule.")
    _prologue(s)
    stage_inject_row(s)
    s.emit(f"// ONE-JOB-PORT INTERLEAVE (measured, see B-STAGE-PORT): the act "
           f"stage has a single job FSM, so it cannot hold a LOAD (sinking "
           f"the feeder that eats swiglu-p) while the injection EMITs the "
           f"rows that drive the serializer. seam_feeder_quant buffers a "
           f"whole row before it emits (rbuf[D], pass1/pass2), so the legal "
           f"order is: arm the feeder, inject the row's products, THEN LOAD.")
    s.lctl(ser_dst=2, fsrc_ext=2)
    s.pmode(True)
    per_row = D_TILE // C                        # swiglu chunks per feeder row
    # ROW GROUPS (found by the 296-chunk pre-flight gate, 2026-07-31): the
    # act stage bank holds STAGE_R_MAX=16 rows and the descriptor row field
    # tops out at 31 — staging rows at 2+r stalled the 296-chunk run at
    # r=30 (aj row 32) after silently ALIASING rows past 15 in the smaller
    # runs (reads aliased identically, so grades stayed green — the worst
    # kind of luck). Stage in groups of G into the FIXED bank rows
    # [2 .. 2+G-1], read each group's codes back, then REUSE the rows.
    G = 8
    for g0 in range(0, rows, G):
        gn = min(G, rows - g0)
        s.emit(f"// ── row group {g0 // G}: feeder rows {g0}..{g0 + gn - 1} "
               f"-> bank rows 2..{1 + gn} (staged, scaled, read, reused)")
        s.route(rdst=1, asrc=0)
        for i in range(gn):
            r = g0 + i
            s.emit(f"// feeder row {r}: {per_row} swiglu chunks of {C}")
            s.fjob(1)
            for k in range(per_row):
                c = r * per_row + k
                g = acc_g[C * c:C * (c + 1)]
                u = acc_u[C * c:C * (c + 1)]
                s.ljc(comp_g)
                s.ljc(comp_u)
                s.lujob(LU_SWIGLU, C)
                inject_frame(s, g, kind="k128")
                inject_frame(s, u, kind="k128")
            s.aj(0, ACT_BANK, 0, 1, s.BPR, 2 + i)  # drain into the GROUP row
            s.efs(0, 0)                            # -> cap: the C-1 row scale
        s.lpoll(LST_SWG, 0x0)
        s.emit("// group readback: identity-GEMM codes for the staged rows "
               "BEFORE the next group overwrites them")
        s.route(rdst=0, wsrc=0, asrc=0)
        for i in range(gn):
            for b in range(0, D_TILE, MXE_N):
                identity_readback(s, 2 + i, b)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"LAYER {name}: swiglu {chunks}x{C}")
    return _emit(x, out_dir, name, dict(
        stage="swiglu", kind="serializer->asu_swiglu->feeder->act->RO",
        chunks=int(chunks), cols_per_job=int(C), n_values=int(n),
        feeder_rows=int(rows), jobc_gate=f"{comp_g:#010x}",
        jobc_up=f"{comp_u:#010x}", inject="k128"))


def build_swiglu_probe(step, *, out_dir: Path, name: str, cols: int = 64,
                       gate_len: int = 64, up_len: int = 64,
                       expect_frame_error: bool = False):
    """REGRESSION GUARD against reintroducing B-SWG-PHASE (fixed in R2,
    apex_top.sv swg_up_q phase steering): one exactly-framed gate/up chunk,
    then a CHECKED read of LAYER_STATUS asserting NO error sticky. Under the
    defect (both skids latching the same beat) this exact sequence returned
    sticky + err_code 3 (FRAME), so the guard goes RED the day the steering
    regresses. History: this was born as the blocker's RED TEST with
    expect_frame_error=True (green while the defect existed); it flipped
    polarity on 2026-07-30 when the fix landed and its FAIL — measured on
    the fixed twin, same twin the swiglu stage went bit-exact on — was the
    red-to-green evidence.
    """
    acc_g = np.asarray(step["acc_g"], dtype=np.int64)
    acc_u = np.asarray(step["acc_u"], dtype=np.int64)
    s = LayerScript(D_TILE)
    s.emit(f"// B-SWG-PHASE probe: cols={cols} gate_frame={gate_len} "
           f"up_frame={up_len}. "
           + ("The EXPECTATION IS THE DEFECT: LAYER_STATUS sticky set with "
              "err_code 3 (FRAME)." if expect_frame_error else
              "REGRESSION GUARD (defect fixed, R2): LAYER_STATUS must show "
              "NO error sticky."))
    _prologue(s)
    stage_inject_row(s)
    s.route(rdst=1, asrc=0)
    s.lctl(ser_dst=2, fsrc_ext=2)
    s.fjob(1)
    s.ljc(int(step["comp_g"]))
    s.ljc(int(step["comp_u"]))
    s.lujob(LU_SWIGLU, cols)
    inject_frame(s, acc_g[:gate_len], kind="k128")
    inject_frame(s, acc_u[:up_len], kind="k128")
    s.lpoll(LST_SWG, 0x0)
    if expect_frame_error:
        s.lstat(0x1F00, (3 << 9) | LST_ERR)
    else:
        s.lstat(LST_ERR, 0x0)
    s.emit("DONE")
    kind = ("B-SWG-PHASE red test" if expect_frame_error
            else "B-SWG-PHASE regression guard")
    x = _translate(s, note=f"LAYER {name}: {kind}")
    return _emit(x, out_dir, name, dict(
        stage="swiglu_probe", kind=kind, cols=int(cols),
        gate_frame=int(gate_len), up_frame=int(up_len),
        expect=("LAYER_STATUS err_code=3 (FRAME)" if expect_frame_error
                else "LAYER_STATUS error sticky CLEAR")))


def build_norm2_stage(step, *, dm: int, out_dir: Path, name: str):
    """Wide RMSNorm-2 (row B4): dm elements through the WIDE asu_rmsnorm.

    Requires a WIDE (RMS_D_MAX >= dm) twin; on the narrow flyable image use
    the R4 chunked stages below (build_norm2_chunked / build_norm2_ext),
    which compute the SAME row bit-exactly.

    The unit itself is width-parameterized (RMS_D_MAX = APEX_CL_DM) and takes
    the whole row: ONE xa frame of dm INT8 codes + dm gamma words. Its
    output re-enters through the C-1 feeder, which is frozen at 128 per row
    (B-FEED-WIDTH), so the tile frames the SAME values as dm/128 rows with
    dm/128 independent scales. That framing is graded against golden applied
    THE SAME WAY — the difference from golden's single whole-row scale IS
    the blocker, and it is reported, not hidden.
    """
    assert dm % D_TILE == 0
    rows = dm // D_TILE
    assert rows <= 31, f"feeder/stage rows {rows} exceed ROWS_MAX 31"
    r1_8 = np.asarray(step["r1_8"], dtype=np.int64)[:dm]
    g2 = np.asarray(step["gamma2"], dtype=np.int64)[:dm]

    s = LayerScript(D_TILE)
    s.emit(f"// LAYER stage {name}: RMSNorm-2 at dm={dm} through the WIDE "
           f"asu_rmsnorm (RMS_D_MAX=APEX_CL_DM). One xa frame of {dm} "
           f"codes; the C-1 re-entry is framed {rows} x {D_TILE} by the "
           f"feeder's frozen row length (B-FEED-WIDTH).")
    _prologue(s)
    s.route(rdst=0, asrc=0)
    s.aj(0, ACT_BANK, 0, rows, s.BPR, 0)
    s.fjob(rows)
    s.pmode(True)
    # xa/xg are HOST ports, not act-stage EMITs, so the LOAD may stay armed
    # here — the B-STAGE-PORT interleave is only needed when the drive itself
    # comes out of the act stage (swiglu, rope).
    s.xrow_n(r1_8)
    s.emit(f"// gamma in {rows} x {D_TILE} chunks, each followed by ITS "
           f"feeder-scale pop. MEASURED: pushing all {dm} gamma words first "
           f"wedges the tile at ~19 rows — the mailbox fs capture FIFO fills "
           f"and seam_feeder_quant's ST_SPUSH blocks on scl_ready, which "
           f"back-pressures asu_rmsnorm's gamma port. Draining per row keeps "
           f"the FIFO at depth 1.")
    for r in range(rows):
        s.grow_n(g2[r * D_TILE:(r + 1) * D_TILE])
        s.efs(0, 0)                       # placeholder; rewritten to `cap`
    s.emit("// codes back through identity GEMMs")
    s.route(rdst=0, wsrc=0, asrc=0)
    for r in range(rows):
        for b in range(0, D_TILE, MXE_N):
            identity_readback(s, r, b)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"LAYER {name}: wide RMSNorm-2 dm={dm}")
    return _emit(x, out_dir, name, dict(
        stage="norm2", kind="xa->asu_rmsnorm(wide)->widen->feeder->act->RO",
        dm=int(dm), feeder_rows=int(rows)))


# ═══ R4 (C-RMSW chunk composition): the FULL wide norm on a NARROW build ═══
#
# golden rmsnorm_fx_wide (attention.py:124) ALREADY defines the wide norm as
# a composition of <=128-chunk passes; R4 gives the tile the two missing
# verbs (asu_rmsnorm R4 header; apex_top CSR pair 0x90/0x94):
#   pass 1  each 128-chunk streams as its own legal narrow row with
#           sum-capture armed: the tile EXPORTS the chunk's sum2 (golden
#           step 1), and the host merely ADDS the k exports (golden step
#           2's "host (or a scalar adder)" — pure INT32 addition, dm/128 - 1
#           adds; sum2_mxe.py's MXE x·x dot product is the zero-add
#           alternative source for the same INT32).
#   pass 2  the total is staged (0x90) + armed (0x94, k = dm/128); the μ
#           correction, EPS, rsqrt AND the whole per-element datapath for
#           all dm elements run ON THE TILE, per chunk, with the one
#           broadcast r (golden steps 3-4).
# The C-1 re-entry stays framed rows x 128 (B-FEED-WIDTH) and is graded
# against golden applied the same way — exactly the norm2 stage's rule.

def _norm2_chunk_sums(step, dm: int):
    """The chunks and their sums, with golden's OWN width proofs mirrored."""
    assert dm % D_TILE == 0, f"dm={dm} is not a multiple of {D_TILE}"
    k = dm // D_TILE
    assert RMS_EXT_K_MIN <= k <= RMS_EXT_K_MAX, \
        f"k={k} outside the R4 arm envelope [{RMS_EXT_K_MIN},{RMS_EXT_K_MAX}]"
    r1_8 = np.asarray(step["r1_8"], dtype=np.int64)[:dm]
    assert r1_8.size == dm
    sums = []
    for c in range(k):
        s2c = int(np.sum(r1_8[c * D_TILE:(c + 1) * D_TILE] ** 2))
        assert s2c < 2 ** 22, "per-chunk width proof (golden attention.py:156)"
        sums.append(s2c)
    total = sum(sums)
    assert total <= (k << 21), \
        "total exceeds the reachable max k*2^21 — impossible for INT8 rows"
    return r1_8, sums, total


def build_norm2_chunked(step, *, dm: int, out_dir: Path, name: str):
    """R4 self-contained two-pass chunked norm (host arithmetic: k-1 adds)."""
    r1_8, sums, total = _norm2_chunk_sums(step, dm)
    rows = dm // D_TILE
    assert rows <= 31, f"feeder/stage rows {rows} exceed ROWS_MAX 31"
    g2 = np.asarray(step["gamma2"], dtype=np.int64)[:dm]

    s = LayerScript(D_TILE)
    s.emit(f"// LAYER stage {name} (R4): FULL dm={dm} RMSNorm-2 on a NARROW "
           f"(RMS_D_MAX=128) build, as golden rmsnorm_fx_wide's own "
           f"composition — {rows} chunk-sum exports, host ADDS them "
           f"({rows - 1} INT32 adds, the only host arithmetic), then the "
           f"armed tile runs μ+rsqrt+all {dm} per-element ops itself.")
    _prologue(s)
    s.pmode(True)
    s.emit("// ── pass 1: sum-capture rows (no rsqrt, no emission, no gamma)")
    s.nctl(sum_en=1, clr=1)
    for c in range(rows):
        s.xrow_n(r1_8[c * D_TILE:(c + 1) * D_TILE])
        s.npoll_cnt(c + 1)                 # completion observable
        s.ns2cap()                         # the chunk's sum2 export -> cap
    s.nctl()                               # disarm (unit idle: count seen)
    s.emit(f"// ── pass 2: total staged+armed (k={rows}); per-element "
           f"datapath per chunk with the ONE broadcast r. Feeder re-entry "
           f"framed {rows} x {D_TILE} (B-FEED-WIDTH), scale popped per row "
           f"(B-FS-FIFO).")
    s.nsum2(total)
    s.nctl(ext=1, k=rows)
    s.route(rdst=0, asrc=0)
    s.aj(0, ACT_BANK, 0, rows, s.BPR, 0)
    s.fjob(rows)
    for c in range(rows):
        s.xrow_n(r1_8[c * D_TILE:(c + 1) * D_TILE])
        s.grow_n(g2[c * D_TILE:(c + 1) * D_TILE])
        s.efs(0, 0)                        # placeholder; rewritten to `cap`
    s.emit("// codes back through identity GEMMs")
    s.route(rdst=0, wsrc=0, asrc=0)
    for r in range(rows):
        for b in range(0, D_TILE, MXE_N):
            identity_readback(s, r, b)
    s.nctl()                               # disarm ext
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"LAYER {name}: R4 chunked RMSNorm-2 dm={dm}")
    return _emit(x, out_dir, name, dict(
        stage="norm2c", kind="xa chunks->sum2 exports->host adds->ext arm->"
                             "asu_rmsnorm(narrow)->widen->feeder->act->RO",
        dm=int(dm), feeder_rows=int(rows), ext_sum2=int(total),
        host_arith=f"{rows - 1} INT32 adds (chunk-sum accumulation)"))


def build_norm2_ext(step, *, dm: int, out_dir: Path, name: str,
                    ext_sum2=None, sum2_source: str = "host"):
    """R4 pass-2 ONLY: the ext-armed emission with sum2 supplied from
    OUTSIDE the norm unit — the exact program shape the proven MXE x·x
    source (sum2_mxe.py, sum2 bit-exact on this twin) plugs into. Doubles
    as the grade-FAIL red test: pre-R4 RTL acks 0x90/0x94 as reserved
    no-ops, each chunk then normalizes by its LOCAL r, and the grade
    diverges from golden's broadcast-r row.

    sum2_source SELECTS WHO IS ALLOWED TO HAVE PRODUCED `ext_sum2`:
      "host" (stage norm2x, the default) — the emitter derived it from its
        own input row, so the REFUSAL GUARD applies: arming anything other
        than the row's own total would bake a wrong answer into a
        green-looking program, and that is refused here.
      "tile" (stage norm2m) — `ext_sum2` was MOVED verbatim off an MXE
        capture (build_chain_job's final accumulator). The refusal guard is
        deliberately NOT applied: re-deriving the total host-side and
        checking the tile against it would make the host the arbiter of the
        tile's own output — exactly the laundering this lane exists to
        avoid — and would make the perturbation discriminator unrunnable.
        The only thing enforced is the CSR field's own legality (nsum2's
        [27:0] bound, a range check, not arithmetic). THE GRADE DECIDES.
        For the record only, the manifest carries the host total under
        `host_total_for_reference_only`; nothing in the build path reads it.
    """
    assert sum2_source in ("host", "tile"), \
        f"sum2_source must be 'host' or 'tile', got {sum2_source!r}"
    r1_8, _, total = _norm2_chunk_sums(step, dm)
    rows = dm // D_TILE
    assert rows <= 31, f"feeder/stage rows {rows} exceed ROWS_MAX 31"
    g2 = np.asarray(step["gamma2"], dtype=np.int64)[:dm]
    if sum2_source == "tile":
        assert ext_sum2 is not None, \
            "sum2_source='tile' needs the captured accumulator; there is no " \
            "host default to fall back on"
        armed = int(ext_sum2)                 # MOVE: no arithmetic, no compare
    else:
        armed = int(total if ext_sum2 is None else ext_sum2)
        assert armed == total, \
            "REFUSE: arming a sum2 that is not the row's own total would " \
            "bake a wrong answer into a green-looking program"

    s = LayerScript(D_TILE)
    s.emit(f"// LAYER stage {name} (R4): ext-only chunked RMSNorm-2 dm={dm} "
           f"— sum2 comes from OUTSIDE the norm unit ("
           + ("MOVED verbatim off an MXE x·x capture: the host applied NO "
              "arithmetic to it"
              if sum2_source == "tile" else
              "here the emitter's input-derived total; on the flight path, "
              "the MXE dot product of the row with itself") + ").")
    _prologue(s)
    s.pmode(True)
    s.nsum2(armed)
    s.nctl(ext=1, k=rows)
    s.route(rdst=0, asrc=0)
    s.aj(0, ACT_BANK, 0, rows, s.BPR, 0)
    s.fjob(rows)
    for c in range(rows):
        s.xrow_n(r1_8[c * D_TILE:(c + 1) * D_TILE])
        s.grow_n(g2[c * D_TILE:(c + 1) * D_TILE])
        s.efs(0, 0)
    s.emit("// codes back through identity GEMMs")
    s.route(rdst=0, wsrc=0, asrc=0)
    for r in range(rows):
        for b in range(0, D_TILE, MXE_N):
            identity_readback(s, r, b)
    s.nctl()
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"LAYER {name}: R4 ext-only RMSNorm-2 dm={dm} "
                           f"(sum2 source: {sum2_source})")
    extra = dict(
        stage=("norm2m" if sum2_source == "tile" else "norm2x"),
        kind="ext sum2->arm->asu_rmsnorm(narrow)->widen->feeder->act->RO",
        dm=int(dm), feeder_rows=int(rows), ext_sum2=int(armed),
        sum2_source=sum2_source)
    if sum2_source == "tile":
        extra["host_arith"] = ("NONE — ext_sum2 was moved verbatim from an "
                               "MXE capture; no host addition, no host "
                               "comparison gated this build")
        extra["host_total_for_reference_only"] = int(total)
    return _emit(x, out_dir, name, extra)


# ── the MXE half of the pair: sum2 computed ON THE TILE, zero host adds ─────
#
# build_norm2_chunked's host arithmetic is dm/128 - 1 = 27 INT32 adds. The MXE
# can produce the SAME INT32 as ONE number (sum2_mxe.py proved the x·x dot
# product bit-exact, 120498), and gemm_job.build_chain_job now runs the whole
# K-split IN ONE PROGRAM with the accumulation ON THE TILE (OP_GEMM_OS
# accumulate=1). So the pair
#     build_sum2_chain  ->  [host MOVES one u32]  ->  build_norm2_ext(tile)
# has NO host arithmetic anywhere: the host reads a capture register and
# writes its bits to a CSR.

def build_sum2_chain(step, *, dm: int, out_dir: Path, name: str,
                     rows_per_desc: int = 1):
    """The norm-2 input row's sum-of-squares, computed ON THE TILE.

    The row is BOTH the activation and the single weight column (the identity
    sum2 = x·x, sum2_mxe.sum2_weight_block — reused here, including its
    refusal of a -128 code as a weight). Every descriptor after the first
    carries accumulate=1, so the LAST RO beat's lane 0 is the whole row's
    sum2 and the host never adds anything.
    """
    r1_8 = np.asarray(step["r1_8"], dtype=np.int64)[:dm]
    assert r1_8.size == dm, f"r1_8 has {r1_8.size} codes, need dm={dm}"
    path, man = gj.build_chain_job(s2m.sum2_weight_block(r1_8), r1_8,
                                   out_dir=out_dir, name=name,
                                   rows_per_desc=rows_per_desc)
    man = dict(man)
    man["stage"] = "sum2m"
    man["dm"] = int(dm)
    man["audit"] = audit_program(path)
    (Path(out_dir) / f"{name}.manifest.json").write_text(
        json.dumps({k: v for k, v in man.items() if k != "audit"}, indent=1))
    return man


def norm2_grade_sensitivity(step, dm: int) -> dict:
    """The SMALLEST sum2 error that can change what the norm2 grade compares.

    HONEST NECESSITY. golden's wide-norm tail is floor-heavy: sum2 enters only
    through the single integer r = 2^13 // isqrt(((sum2*mu) >> (s+16)) + 1),
    so every sum2 in one r-bucket produces the BIT-IDENTICAL row. sum2_mxe.py
    measured that bucket at 4942 wide on this row. A "+1" perturbation of the
    armed CSR therefore CANNOT go red — not because the grader is blind, but
    because golden itself does not distinguish the two inputs. A discriminator
    must move sum2 across an r boundary, and this function finds the first one:
      * r is monotone non-increasing in sum2 (mean2 and isqrt are monotone),
        so the smallest delta that changes r is found by exact bisection;
      * from there it walks r-buckets until the GRADED projection (golden's
        own quant_rows_i8 of y over the `rows` x 128 grading shape — the exact
        (codes, scales) pair grade_codes_scales compares) actually differs.
    Returns the measured delta plus the evidence that +1 does not.
    """
    r1_8 = np.asarray(step["r1_8"], dtype=np.int64)[:dm]
    g2 = np.asarray(step["gamma2"], dtype=np.int64)[:dm]
    rows = dm // D_TILE
    _, _, total = _norm2_chunk_sums(step, dm)

    def graded(sum2):
        y, r, _n = s2m.norm_from_sum2(int(sum2), r1_8, g2)
        c, sc = at.quant_rows_i8(np.asarray(y, dtype=np.float64)
                                 .reshape(rows, D_TILE) / 256.0)
        return (np.asarray(c, dtype=np.int64), np.asarray(sc, dtype=np.uint16),
                int(r))

    base_c, base_s, base_r = graded(total)
    # 1. exact bisection for the first delta that moves r (monotone)
    hi = 1
    while graded(total + hi)[2] == base_r:
        hi *= 2
        assert hi < (1 << 27), "no r change within the CSR's own [27:0] range"
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if graded(total + mid)[2] == base_r:
            lo = mid + 1
        else:
            hi = mid
    d = lo
    # 2. walk r-buckets until the GRADED pair really differs
    while True:
        c, sc, r = graded(total + d)
        if not np.array_equal(c, base_c) or not np.array_equal(sc, base_s):
            break
        d += 1
        assert d < (1 << 27), "no graded change within the CSR's own range"
    c1, s1, r1 = graded(total + 1)
    n_code_diff = int(np.sum(c != base_c))
    n_scale_diff = int(np.sum(sc != base_s))
    assert n_code_diff or n_scale_diff, "sensitivity search returned a no-op"
    return {"dm": int(dm), "rows": int(rows), "host_total": int(total),
            "delta_min": int(d), "r_base": base_r, "r_at_delta": int(r),
            "n_code_diff": n_code_diff, "n_scale_diff": n_scale_diff,
            "plus_one_absorbed": bool(np.array_equal(c1, base_c)
                                      and np.array_equal(s1, base_s)
                                      and r1 == base_r),
            "why": "sum2 reaches the graded row ONLY through the integer r; "
                   "deltas inside one r bucket are invisible to golden "
                   "itself, so the discriminator uses the first delta that "
                   "leaves it"}


def grade_sum2_chain(man, step, caps) -> dict:
    """Grade an on-tile sum2 chain: the LAST beat's lane 0 == golden's own
    per-chunk accumulation (rmsnorm_fx_wide step 1, replayed by
    _norm2_chunk_sums). Golden is computed HERE, after the run."""
    _, _, want = _norm2_chunk_sums(step, int(man["dm"]))
    try:
        ct = gj.chain_totals(caps, n_desc=int(man["n_desc"]))
    except ValueError as e:
        return {"equal": False, "n": 0, "error": f"{type(e).__name__}: {e}"}
    got = int(ct["total"][0])
    pad_zero = all(v == 0 for v in ct["total"][1:])
    return {"equal": bool(got == want and pad_zero),
            "n": int(man["K_real"]), "sum2_tile": got, "sum2_golden": want,
            "sum2_equal": bool(got == want), "pad_lanes_zero": bool(pad_zero),
            "n_beats": int(ct["n_beats"]),
            "raw_lane0_u32": int(ct["raw_lane0_u32"]),
            "host_adds": 0}


def build_norm2_probe(step, *, out_dir: Path, name: str):
    """R4 REFUSAL GUARD (the resid_probe idiom): an ext arm with k=1 —
    below the μ-table envelope's chunked floor — must be REFUSED loudly
    (LAYER_STATUS sticky + err_code 8) and must NOT arm: the mode readback
    stays zero. On pre-R4 RTL 0x94 is a reserved no-op, the sticky never
    sets, and the bounded poll goes RED — the red half of the evidence."""
    del step                                  # geometry-only: no operands
    s = LayerScript(D_TILE)
    s.emit("// R4 refusal probe: RMS_EXT write ext=1 k=1 (outside [2,64]) — "
           "EXPECT sticky + err_code 8, mode NOT armed, then W1C clears.")
    _prologue(s)
    s.nctl_raw((1 << 2) | 2)                  # ext=1, k=1: illegal
    s.lpoll(LST_ERR, LST_ERR)
    s.lstat(0x1F00 | LST_LDQ | LST_SWG | LST_RES,
            (LERR_NORM_EXT << 9) | LST_ERR)   # sticky + code 8, units idle
    s.emit("// the refused write must not have PARTIALLY landed: modes, k "
           "and the capture count all still zero (input-derived state)")
    s.csrr(RMS_EXT, 0x00FF7FFF, 0x0)
    s.lstatw(LST_ERR)                         # W1C (code holds by design)
    s.lstat(LST_ERR, 0x0)
    s.emit("DONE")
    x = _translate(s, note=f"LAYER {name}: R4 refusal guard")
    return _emit(x, out_dir, name, dict(
        stage="norm2_probe", kind="R4 ext-arm refusal guard",
        expect="LAYER_STATUS sticky + err_code 8 (NORM_EXT), mode unarmed, "
               "then W1C"))


def build_rope_stage(step, *, head: int, out_dir: Path, name: str):
    """B5 — the RoPE observability fix.

    Drive: the head's PRE-rope q row is injected as (v, comp) pairs through
    apex_scale_quant MODE_F16 (the L3 store_kv_phase idiom), the phase-q
    table is loaded into LAYER_PTR bank 1, rope_en=1 rotates it, and
    l_fsrc_ext=3 (q_sink) steers the seam into the C-1 feeder instead of the
    KV write port. Observe: the fs row scale + identity-GEMM INT8 codes.
    """
    hd = int(step["head_dim"])
    assert hd == D_TILE, f"rope stage assumes head_dim == {D_TILE}, got {hd}"
    q_pre = np.asarray(step["q_pre_bits"], dtype=np.uint16)[
        head * hd:(head + 1) * hd]
    phase = np.asarray(step["phase_q"], dtype=np.int64)

    s = LayerScript(D_TILE)
    s.emit(f"// LAYER stage {name} (B5): rope_row observability. The rotated "
           f"row is steered into the C-1 feeder by l_fsrc_ext=3 (q_sink, "
           f"apex_top.sv:1186) — the egress rope_row never had.")
    _prologue(s)
    s.emit(f"// phase-q table -> LAYER_PTR bank 1 ({len(phase)} pair codes)")
    s.lptr(BANK_PHQ, 0)
    s.ldata(phase)
    s.route(rdst=1, asrc=0)
    s.lctl(rope_en=1, rope_bank=1, fsrc_ext=3)
    s.pmode(True)
    s.emit("// arm the feeder, THEN inject (one act-stage job port — the "
           "B-STAGE-PORT interleave; the feeder buffers the whole row)")
    s.fjob(1)
    s.emit("// inject the PRE-rope q row through squant MODE_F16")
    pairs = [g3.decompose_f16(int(b))[:2] for b in q_pre]
    g3.inject_jobs(s, pairs, 0, waddr=None)
    s.aj(0, ACT_BANK, 0, 1, s.BPR, 0)      # drain the rotated row into act
    s.efs(0, 0)                            # -> cap: the rotated row's scale
    s.lpoll(LST_ROPE, 0x0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for b in range(0, D_TILE, MXE_N):
        identity_readback(s, 0, b)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"LAYER {name}: rope head {head} (B5)")
    return _emit(x, out_dir, name, dict(
        stage="rope", kind="squant F16->rope_row->q_sink->feeder->act->RO",
        head=int(head), head_dim=int(hd)))


def _emit(x: LayerXlate, out_dir: Path, name: str, extra: dict) -> dict:
    """Write the regops + manifest and run the zero-expectation audit."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.regops.jsonl"
    with path.open("w") as f:
        for o in x.ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    man = dict(name=name, regops=len(x.ops), checks=x.n_checks,
               caps=x.cap_seq, cap_sites_rewritten=x.n_capsites,
               erd_reads=x.n_erd, path=str(path), **extra)
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    man["audit"] = audit_program(path)
    return man


def audit_program(path) -> dict:
    """ZERO OUTPUT EXPECTATIONS, re-derived FROM THE FILE.

    No `cap` may carry an `e` and no surviving `r`/`rn`/`poll` may target a
    capture-window DATA address or LAYER_RDATA — the addresses through which
    a stage's own output leaves the tile.
    """
    n_cap = n_cap_e = 0
    bad = []
    for ln in Path(path).read_text().splitlines():
        o = json.loads(ln)
        if o["op"] == "cap":
            n_cap += 1
            if "e" in o:
                n_cap_e += 1
                bad.append(o)
        elif o["op"] in ("r", "rn", "poll") and o.get("a") in OUTPUT_ADDRS:
            bad.append(o)
    return {"caps": n_cap, "caps_with_e": n_cap_e, "violations": len(bad),
            "first": bad[0] if bad else None}


# ═══════════════ 4. the step capture (provenance, C1's discipline) ═════════

def capture_layer_step(*, layer: int, step: int, out: Path,
                       n_ffn: int = 512, verbose=True) -> Path:
    """Capture ONE real Qwen2.5-7B decode step's layer tensors into an .npz.

    Provenance follows breadth_step.py: the decode is run by the SAME
    machinery (run_tinynpu over the committed S8 prompt), and the layer's
    own X is taken from the call the model actually made. The Lane-B ARBITER
    is `decoder_layer_fx(..., bus=BUS_ON)` — the L4/walker arbiter
    (verif/top/l4/gen_l4_vectors.py:182, verif/seq_walker/gen_layer_trace.py)
    — because every LAYER unit REFUSES an ungraded composite
    (apex_layer_deq.sv:90-92). Both arbiters are run and their divergence is
    recorded in the npz, so the grade never silently swaps references.
    """
    import run_tinynpu as rt

    run = json.loads(S8_RUN.read_text())
    model = rt.GoldenModel(Path(S8_WEIGHTS))
    ids = [int(v) for v in run["prompt_ids"]]
    # run.json records the tier by its GOLDEN name ("CQ-8"); TIER_MAP keys
    # are the CLI aliases. decoder_layer_fx wants the golden name.
    tier, group = str(run["tier"]), int(run["group"])
    if verbose:
        eprint(f"[capture] layer {layer} step {step} of the committed S8 "
               f"prompt ({len(ids)} ids), tier={tier} G={group}")

    # FAST PREFIX (breadth_step.py's argument, verbatim): Session.hist[li] is
    # fed by layer li-1's output alone, so layers > `layer` cannot influence
    # layer `layer`'s inputs at any step. Evaluating 0..layer is EXACT for
    # everything captured here — it just cannot emit a token.
    Xs = {}
    hist = [[] for _ in range(layer + 1)]
    for si, tid in enumerate(ids):
        x = model.embed_row(tid)
        for li in range(layer + 1):
            hist[li].append(x)
            Xi = np.vstack(hist[li] + [x])
            ri = tf.decoder_layer_fx(Xi, model.layers[li], tier, G=group,
                                     q_pos=si)
            if li == layer and si == step:
                Xs["X"] = np.array(Xi, dtype=np.float64)
                Xs["q_pos"] = si
                Xs["r_off"] = ri
            x = ri.r2
        if verbose:
            eprint(f"  [prefix {si + 1}/{len(ids)}] layers 0..{layer} "
                   f"T={si + 1}")
        if si == step:
            break
    if "X" not in Xs:
        raise SystemExit(f"REFUSE: (step {step}, layer {layer}) never reached")

    X, q_pos, r_off = Xs["X"], Xs["q_pos"], Xs["r_off"]
    w = model.layers[layer]
    r = tf.decoder_layer_fx(X, w, tier, G=group, q_pos=q_pos, bus=tf.BUS_ON)
    _write_step_npz(out, X, w, r, r_off, layer=layer, step=step, tier=tier,
                    group=group, q_pos=q_pos, n_ffn=n_ffn)
    return out


def synthetic_layer_step(*, out: Path, H=4, head_dim=128, H_kv=1,
                         d_ffn=1024, T=8, seed=0x1B0B01, n_ffn=512) -> Path:
    """A SYNTHETIC step npz at tile-legal geometry — the debug loop.

    Same recipe as verif/seq_walker/gen_layer_trace.make_weights and the same
    BUS_ON arbiter, so the emitted choreography is byte-for-byte the shape the
    real 7B step produces; only the numbers are synthetic. Never used for a
    claim — every graded row in the smoke table names its npz's `source`.
    """
    import gen_layer_trace as glt
    rng = np.random.default_rng(seed)
    w = glt.make_weights(rng, H, head_dim, H_kv, d_ffn, False)
    D = H * head_dim
    Xc = rng.normal(0, 1, (T, D))
    X = np.vstack([Xc, Xc[-1]])
    r = tf.decoder_layer_fx(X, w, "CQ-8", G=128, q_pos=T - 1, bus=tf.BUS_ON)
    r_off = tf.decoder_layer_fx(X, w, "CQ-8", G=128, q_pos=T - 1)
    _write_step_npz(out, X, w, r, r_off, layer=0, step=T - 1, tier="CQ-8",
                    group=128, q_pos=T - 1, n_ffn=n_ffn,
                    source="SYNTHETIC (gen_layer_trace.make_weights recipe)")
    return out


def _rederive(X, w, r):
    """Re-derive, via PUBLIC golden functions only, the epilogue internals
    decoder_layer_fx discards but the LAYER units need: the o8 code streams
    and the fp16-graded JOBC composites of the o-proj and the down-proj.

    Asserted bit-exact against the arbiter's own attn_proj / ffn_out / r1 /
    r2 BEFORE anything is emitted — gen_l4_vectors.seg1_selfcheck's rule,
    one lane wider.
    """
    out = {}
    for tag, vec, W, s_w, ref in (
            ("r1", r.attn, w.Wo, w.s_wo, r.attn_proj),
            ("r2", r.swiglu, w.Wd, w.s_wd, r.ffn_out)):
        a8, s_a = at.quant_rows_i8(np.asarray(vec, dtype=np.float64)[None, :])
        acc = tf.gemm_i8_ksplit(a8, np.asarray(W, dtype=np.int64))[0]
        acc = acc.astype(np.int64)
        amax = int(np.max(np.abs(acc), initial=0)) or 1
        scale, shift = at.calib_requant(amax)
        o8 = at.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
        s_a_v = float(f16_bits_to_f64(np.array([s_a[0]]))[0])
        s_out = tf.f16_grade(s_a_v * float(s_w) * float(1 << shift)
                             / float(scale))
        proj = o8.astype(np.float64) * s_out
        assert np.array_equal(proj, ref), f"{tag}: epilogue replica != arbiter"
        assert int(np.max(np.abs(o8))) < 128, f"{tag}: o8 clip"
        comp = int(np.float64(s_out).astype(np.float32).view(np.uint32))
        assert (comp & 0x1FFF) == 0, f"{tag}: s_out not fp16-graded"
        out[f"{tag}_o8"] = o8
        out[f"{tag}_comp"] = comp
    # the residual row each job adds ONTO, and the row it must produce
    xrow = f64_to_f16_bits(np.asarray(X)[r.T])
    assert np.array_equal(f64_to_f16_bits(tf._f16(
        f16_bits_to_f64(xrow) + r.attn_proj)), f64_to_f16_bits(r.r1)), \
        "r1 re-derive != arbiter"
    out["r1_row_in_bits"] = xrow
    out["r1_bits"] = f64_to_f16_bits(r.r1)
    out["r2_row_in_bits"] = f64_to_f16_bits(r.r1)
    out["r2_bits"] = f64_to_f16_bits(r.r2)
    return out


def _write_step_npz(out: Path, X, w, r, r_off, *, layer, step, tier, group,
                    q_pos, n_ffn, source=None):
    d = _rederive(X, w, r)
    # RMSNorm-2 inputs: golden's own C-1 of r1, and gamma2
    r1_8, _ = at.quant_rows_i8(np.asarray(r.r1, dtype=np.float64)[None, :])
    # SwiGLU operands: the raw INT32 gate/up accumulators + graded composites
    h2_8, s_h2m = at.quant_rows_i8(r.h2.astype(np.float64)[None, :] / 256.0)
    s_h2 = float(f16_bits_to_f64(np.array([s_h2m[0]]))[0])
    acc_g = tf.gemm_i8_ksplit(h2_8, np.asarray(w.Wg, dtype=np.int64))[0]
    acc_u = tf.gemm_i8_ksplit(h2_8, np.asarray(w.Wu, dtype=np.int64))[0]
    comp_g = tf.f16_grade(s_h2 * w.s_wg)
    comp_u = tf.f16_grade(s_h2 * w.s_wu)
    assert np.array_equal(acc_g.astype(np.float64) * comp_g, r.gate_real), \
        "gate re-derive != arbiter"
    assert np.array_equal(acc_u.astype(np.float64) * comp_u, r.up_real), \
        "up re-derive != arbiter"
    gb = lambda x: int(np.float64(x).astype(np.float32).view(np.uint32))  # noqa
    # RoPE operands: the PRE-rope fp16 q row and the phase-q pair table
    q_pre = f64_to_f16_bits(np.asarray(r.q_real, dtype=np.float64))
    phase = _phase_q_codes(w, int(r.head_dim), int(q_pos if q_pos is not None
                                                   else r.T))
    nf = min(int(n_ffn), int(r.d_ffn))
    np.savez_compressed(
        out,
        meta_json=json.dumps(dict(
            layer=int(layer), step=int(step), tier=tier, group=int(group),
            q_pos=int(q_pos if q_pos is not None else r.T), T=int(r.T),
            D_model=int(r.D_model), d_ffn=int(r.d_ffn), H=int(r.H),
            H_kv=int(r.H_kv), head_dim=int(r.head_dim), n_ffn=int(nf),
            arbiter="decoder_layer_fx(bus=BUS_ON)",
            bus_off_r2_equal=bool(np.array_equal(
                f64_to_f16_bits(r_off.r2), f64_to_f16_bits(r.r2))),
            source=source or str(S8_RUN))),
        D_model=int(r.D_model), d_ffn=int(r.d_ffn), head_dim=int(r.head_dim),
        r1_o8=d["r1_o8"], r1_comp=d["r1_comp"],
        r1_row_in_bits=d["r1_row_in_bits"], r1_bits=d["r1_bits"],
        r2_o8=d["r2_o8"], r2_comp=d["r2_comp"],
        r2_row_in_bits=d["r2_row_in_bits"], r2_bits=d["r2_bits"],
        r1_8=r1_8[0].astype(np.int64), gamma2=np.asarray(w.gamma2),
        h2=r.h2.astype(np.int64), s_h2_bits=int(s_h2m[0]),
        acc_g=acc_g.astype(np.int64)[:nf], acc_u=acc_u.astype(np.int64)[:nf],
        comp_g=gb(comp_g), comp_u=gb(comp_u),
        swiglu_bits=f64_to_f16_bits(r.swiglu)[:nf],
        q_pre_bits=q_pre, q_rope_bits=f64_to_f16_bits(r.q_rope),
        phase_q=phase)
    return out


def _phase_q_codes(w, head_dim, pos):
    """The rope phase-q table for `pos`, from GOLDEN's own quantizer.

    `rope_phase_q` (transformer.py:244) IS the table the RTL's phase RAM is
    loaded with: one 14-bit Q0.14 turn-fraction code per channel PAIR, indexed
    by rope_row's `pair_j` (rope_row.sv:110). No trigonometry is re-derived
    here — a second implementation of the phase would be a second oracle.
    """
    base = float(getattr(w, "rope_theta_base", 10000.0))
    try:
        theta = tf.rope_theta(head_dim, base)
    except TypeError:
        theta = tf.rope_theta(head_dim)
    return np.asarray(tf.rope_phase_q(float(pos), theta), dtype=np.int64)


# ═══════════════ 5. the one entry point (rows B2-B5) ═══════════════════════

def build_layer_step(layer_idx: int, step_inputs_npz, out_dir=None, *,
                     resid_cols=None, swiglu_chunks=2, norm2_dm=None,
                     rope_head=0, erd_check=False, stages=None) -> dict:
    """Emit the regops program(s) for ONE full decoder-layer step.

    Returns {'stages': [manifest, ...], 'expect': {name: golden}, 'meta': ...}.
    Every stage is a SEPARATE regops file: each gets its own TILE_RST from
    the executors (f2_host_run.py:137), which is the per-file semantics the
    canonical 18-job replay already relies on.
    """
    z = np.load(step_inputs_npz, allow_pickle=False)
    step = {k: z[k] for k in z.files if k != "meta_json"}
    meta = json.loads(str(z["meta_json"]))
    step["d_ffn"] = meta["d_ffn"]
    step["head_dim"] = meta["head_dim"]
    out_dir = Path(out_dir or DEFAULT_OUT)
    dm = int(meta["D_model"])
    # R3: default = the FULL row. The window loop in build_resid_stage keeps
    # every job <= RESID_WIN <= SER_FRAME_MAX, so B-SER-FRAME no longer caps
    # the residual claim. (On a narrow DM_MAX=128 twin pass --resid-cols 128;
    # the unit refuses wider jobs loudly.)
    cols = int(resid_cols if resid_cols is not None else dm)
    n2 = int(norm2_dm if norm2_dm is not None else dm)
    want = set(stages or ("resid_probe", "resid_r1", "norm2", "norm2_probe",
                          "norm2c", "norm2x", "resid_r2",
                          "rope", "swiglu_probe", "swiglu"))
    tag = f"L{layer_idx:02d}_s{meta['step']:03d}"

    mans, expect = [], {}
    if cols > dm:
        raise AssertionError(f"resid_cols={cols} exceeds D_model={dm}")
    if "resid_probe" in want:
        mans.append(build_resid_probe(step, out_dir=out_dir,
                                      name=f"lay_{tag}_resid_probe"))
    if "resid_r1" in want:
        m, e = build_resid_stage(step, which="r1", cols=cols, out_dir=out_dir,
                                 name=f"lay_{tag}_resid_r1",
                                 erd_check=erd_check)
        mans.append(m)
        expect[m["name"]] = e
    if "swiglu_probe" in want:
        mans.append(build_swiglu_probe(step, out_dir=out_dir,
                                       name=f"lay_{tag}_swiglu_probe"))
    if "swiglu" in want:
        mans.append(build_swiglu_stage(step, chunks=swiglu_chunks,
                                       out_dir=out_dir,
                                       name=f"lay_{tag}_swiglu"))
    if "norm2" in want:
        mans.append(build_norm2_stage(step, dm=n2, out_dir=out_dir,
                                      name=f"lay_{tag}_norm2"))
    if "norm2_probe" in want:
        mans.append(build_norm2_probe(step, out_dir=out_dir,
                                      name=f"lay_{tag}_norm2_probe"))
    if "norm2c" in want:
        mans.append(build_norm2_chunked(step, dm=n2, out_dir=out_dir,
                                        name=f"lay_{tag}_norm2c"))
    if "norm2x" in want:
        mans.append(build_norm2_ext(step, dm=n2, out_dir=out_dir,
                                    name=f"lay_{tag}_norm2x"))
    if "resid_r2" in want:
        m, e = build_resid_stage(step, which="r2", cols=cols, out_dir=out_dir,
                                 name=f"lay_{tag}_resid_r2",
                                 erd_check=erd_check)
        mans.append(m)
        expect[m["name"]] = e
    if "rope" in want:
        mans.append(build_rope_stage(step, head=rope_head, out_dir=out_dir,
                                     name=f"lay_{tag}_rope"))
    res = {"stages": mans, "meta": meta, "layer": int(layer_idx),
           "resid_cols": cols, "norm2_dm": n2,
           "fences": {"serializer_frame_max": SER_FRAME_MAX,
                      "residual_window":
                          f"{-(-cols // RESID_WIN)} x {RESID_WIN}-aligned "
                          f"window jobs covering row[0:{cols}] of {dm} "
                          f"(R3 base field)",
                      "swiglu_cols_per_job": fmt.SWG_COLS_MAX}}
    (out_dir / f"lay_{tag}.plan.json").write_text(json.dumps(res, indent=1))
    return res


# ═══════════════ 6. grading ════════════════════════════════════════════════

def grade_resid(caps, expect_bits) -> dict:
    got = cd.rdata_f16(caps)
    n = len(expect_bits)
    bits = np.asarray([c["bits"] for c in got[:n]], dtype=np.uint16)
    if bits.size != n:
        return {"equal": False, "n": n, "got": int(bits.size),
                "note": "short capture"}
    eq = np.array_equal(bits, np.asarray(expect_bits, dtype=np.uint16))
    d = np.flatnonzero(bits != np.asarray(expect_bits, dtype=np.uint16))
    return {"equal": bool(eq), "n": n, "got": int(bits.size),
            "n_diff": int(d.size),
            "first_idx": int(d[0]) if d.size else -1,
            "value_equal": bool(np.array_equal(
                f16_bits_to_f64(bits),
                f16_bits_to_f64(np.asarray(expect_bits, dtype=np.uint16))))}


def grade_codes_scales(caps, ref_vals, *, rows, expect_scale_bits=None):
    """Grade a (fs scale, identity-GEMM INT8 codes) readback of `rows` x 128
    real values against golden's OWN C-1 applied the SAME way."""
    acc = cd.ro_lanes_to_i32(caps, strict=False)
    fs = [t for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    ref = np.asarray(ref_vals, dtype=np.float64).reshape(rows, D_TILE)
    g_codes, g_scales = at.quant_rows_i8(ref)
    n = rows * D_TILE
    got_codes = acc[:n].astype(np.int64) if acc.size >= n else acc.astype(np.int64)
    ok_codes = (got_codes.size == n
                and np.array_equal(got_codes.reshape(rows, D_TILE),
                                   np.asarray(g_codes, dtype=np.int64)))
    got_scales = np.asarray([t["bits"] for t in fs[:rows]], dtype=np.uint16)
    want_scales = (np.asarray(expect_scale_bits, dtype=np.uint16)
                   if expect_scale_bits is not None
                   else np.asarray(g_scales, dtype=np.uint16))
    ok_scales = (got_scales.size == rows
                 and np.array_equal(got_scales, want_scales))
    return {"equal": bool(ok_codes and ok_scales), "n": int(n),
            "codes_equal": bool(ok_codes), "scales_equal": bool(ok_scales),
            "got_codes": int(got_codes.size), "got_scales": int(got_scales.size),
            "n_code_diff": int(np.sum(
                got_codes.reshape(-1)[:min(n, got_codes.size)]
                != np.asarray(g_codes).reshape(-1)[:min(n, got_codes.size)]))}


# ═══════════════ 6b. discriminators (one per stage class) ═════════════════
#
# A grader that always passes proves nothing (cap_decode's own case 5). Each
# class of stage gets ONE perturbation that MUST turn it red:
#   resid       an injected o8 code moved by 1 -> the RDATA row must change
#               AT THAT INDEX (proves the row is the tile's arithmetic, not
#               the preload echoing back)
#   codes/scale one captured INT8 code, and separately one scale bit, moved
#               by 1 -> the grade must fail
#   ERD         the SAME readback done with the shadow-cap shape (two bus
#               reads per element) -> the recovered row must be SKEWED, which
#               is the whole reason ERD exists

def build_resid_discriminator(step, *, cols, out_dir, name, idx=3, delta=1,
                              erd_double=False):
    """resid_r1 with ONE injected o8 code moved by `delta` (or with the
    RDATA reads doubled, the shadow-cap shape)."""
    o8 = np.asarray(step["r1_o8"], dtype=np.int64)[:cols].copy()
    if not erd_double:
        o8[idx] = int(np.clip(o8[idx] + delta, -127, 127))
        if o8[idx] == int(np.asarray(step["r1_o8"])[idx]):
            o8[idx] -= delta                       # already at the clamp
    comp = int(step["r1_comp"])
    row0 = np.asarray(step["r1_row_in_bits"], dtype=np.uint16)[:cols]
    s = LayerScript(D_TILE)
    s.emit(f"// DISCRIMINATOR {name}: "
           + ("RDATA read TWICE per element (the shadow-cap shape) — the "
              "recovered row must come back SKEWED"
              if erd_double else
              f"injected o8[{idx}] moved by {delta}"))
    _prologue(s)
    s.lptr(BANK_RESID, 0)
    s.ldata(row0)
    s.lctl(ser_dst=1, resid_arm=1)
    s.lujob(LU_RESID, cols)
    s.ljc(comp)
    s.lujob(LU_DEQ, cols)
    s.route(rdst=1)
    s.pmode(True)
    inject_frame(s, o8)
    s.lpoll(LST_LDQ | LST_RES, 0x0)
    s.lrptr(0)
    for _ in range(cols):
        s.erd(2 if erd_double else 1)
    s.pmode(False)
    _drain_and_check(s)
    x = _translate(s, note=f"LAYER {name}: discriminator")
    return _emit(x, out_dir, name, dict(
        stage="discriminator", kind="resid_r1 perturbed", cols=int(cols),
        perturb=("erd_double" if erd_double else f"o8[{idx}]{delta:+d}")))


def discriminate(step, plan, rows, *, out_dir, runner) -> dict:
    """Run the perturbations and require every one of them to go RED."""
    # R3: the discriminators stay SINGLE-window (base=0) on purpose — one
    # job, one frame, perturbation semantics unchanged. Window-0 arithmetic
    # is the same datapath every window uses; the full-width coverage claim
    # belongs to the resid stages themselves.
    cols = min(int(plan["resid_cols"]), RESID_WIN)
    out = {}
    # (1) resid: a moved o8 code must move the RDATA row
    m = build_resid_discriminator(step, cols=cols, out_dir=out_dir,
                                  name="disc_resid_o8")
    r = runner(m["path"])
    g = grade_resid(r["captures"],
                    np.asarray(step["r1_bits"], dtype=np.uint16)[:cols])
    out["resid_o8_perturbation"] = {
        "caught": not g["equal"], "n_diff": g.get("n_diff"),
        "first_idx": g.get("first_idx"), "ran": bool(r["ok"])}
    # (2) ERD: the shadow-cap shape must skew the row
    m2 = build_resid_discriminator(step, cols=cols, out_dir=out_dir,
                                   name="disc_erd_double", erd_double=True)
    r2 = runner(m2["path"])
    # The shadow-cap shape emits a `cap` AND an `r` at the same address and
    # PAIRS them as one element. On an auto-incrementing register the pair is
    # row[2i] and row[2i+1], not row[i] twice — so the paired stream is
    # STRIDE-2 SKEWED. That is the exact failure ERD exists to prevent, and it
    # is asserted here on real captures, not argued.
    dec = cd.rdata_f16(r2["captures"])
    paired = np.asarray([d["bits"] for d in dec[0::2][:cols]], dtype=np.uint16)
    exp = np.asarray(step["r1_bits"], dtype=np.uint16)[:cols]
    n_ok = int(paired.size)
    skewed = not (n_ok == cols and np.array_equal(paired, exp))
    out["erd_double_read_skew"] = {
        "caught": skewed, "reads": len(dec), "ran": bool(r2["ok"]),
        "n_diff": int(np.sum(paired[:min(n_ok, cols)]
                             != exp[:min(n_ok, cols)])),
        "note": "shadow-cap pairs two reads as one element; RDATA auto-incs, "
                "so the paired stream is stride-2 skewed"}
    # (3b) R4 chunk-sum export: one captured sum2 moved by 1 must fail BOTH
    #      the export grade and (through sum_total_equal) the composition
    for row in rows:
        if row["stage"] == "norm2c" and row["grade"].get("equal"):
            bad = [dict(c) for c in row["_caps"]]
            for c in bad:
                if c.get("sem") == "rs2":
                    c["value"] = c["value"] + 1
                    break
            gs = _grade_stage(row["_man"], step, plan["meta"], bad, plan)
            out["norm2c_sum2_perturbation"] = {
                "caught": not gs.get("equal"),
                "sums_equal": gs.get("sums_equal"),
                "sum_total_equal": gs.get("sum_total_equal")}
            break
    # (3) codes/scales: host-side perturbation of a real capture
    for row in rows:
        if row["stage"] in ("rope", "norm2", "norm2c", "norm2x") \
                and row["grade"].get("equal"):
            caps = list(row["_caps"])
            tgt = next(c for c in caps if c.get("sem") == "ro_w0")
            bad = [dict(c) for c in caps]
            for c in bad:
                if c is not tgt and c["tag"] == tgt["tag"]:
                    pass
            for c in bad:
                if c["tag"] == tgt["tag"]:
                    c["value"] = (c["value"] + 1) & 0xFFFFFFFF
            g3d = _grade_stage(row["_man"], step, plan["meta"], bad, plan)
            out[f"{row['stage']}_code_perturbation"] = {
                "caught": not g3d.get("equal"), "detail": g3d}
            bad2 = [dict(c) for c in caps]
            for c in bad2:
                if c.get("sem") == "fs":
                    c["value"] = c["value"] ^ 1
                    break
            g4 = _grade_stage(row["_man"], step, plan["meta"], bad2, plan)
            out[f"{row['stage']}_scale_perturbation"] = {
                "caught": not g4.get("equal")}
            break
    return out


# ═══════════════ 7. selftest / blockers / smoke ════════════════════════════

def print_blockers():
    print("LANE-B RTL FENCES (measured, file:line — none extrapolated)\n")
    for b in BLOCKERS:
        print(f"[{b['id']}] stage: {b['stage']}")
        print(f"  what : {b['what']}")
        for wr in b["where"]:
            print(f"  where: {wr}")
        print(f"  needs: {b['needs']}\n")


def _selftest() -> int:
    fails = 0

    def chk(name, cond, extra=""):
        nonlocal fails
        print(f"  [{name}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    # 1. the token layer maps to the frozen §3b addresses
    s = LayerScript(D_TILE)
    s.lctl(rope_en=1, rope_bank=1, ser_dst=2, fsrc_ext=3, resid_arm=1,
           rope_pos=9)
    s.lptr(BANK_RESID, 5)
    s.ldata([0x3C00, 0xBC00])
    s.ljc(0x3F800000)
    s.lujob(LU_SWIGLU, 64)
    s.lrptr(0)
    s.erd(3)
    x = _translate(s)
    addrs = [(o["op"], o.get("a")) for o in x.ops]
    chk("1a LCTL -> 0x1070", addrs[0] == ("w", LA["CTRL"]), f"{addrs[0]}")
    chk("1b level word == seq_walker_fmt.lctl",
        x.ops[0]["d"] == fmt.lctl(rope_en=1, rope_bank=1, ser_dst=2,
                                  fsrc_ext=3, resid_arm=1, rope_pos=9),
        f"{x.ops[0]['d']:#06x}")
    chk("1c LPTR packs bank<<28", x.ops[1]["d"] == (2 << 28) | 5,
        f"{x.ops[1]['d']:#010x}")
    chk("1d LUJOB packs unit<<12",
        any(o["a"] == LA["JOB"] and o["d"] == (1 << 12) | 64 for o in x.ops))
    erds = [o for o in x.ops if o.get("a") == LA["RDATA"]]
    chk("1e ERD is ONE read per element, cap, no expectation",
        len(erds) == 3 and all(o["op"] == "cap" and "e" not in o
                               for o in erds), f"{len(erds)} reads")

    # 2. THE ERD DISCRIMINATOR: shadow-cap doubles the reads of an
    #    auto-incrementing register, so it CANNOT be used for RDATA.
    xs = LayerXlate(D_TILE)
    xs.shadow_cap = True
    xs.r(LA["RDATA"], 0xFFFF, 0x1234, sem="lrd")
    n_shadow = sum(1 for o in xs.ops if o.get("a") == LA["RDATA"])
    chk("2 shadow-cap emits TWO reads of RDATA (why ERD exists)",
        n_shadow == 2, f"{n_shadow} bus reads for one element")

    # 3. B1's per-unit chunk rule is enforced AT EMISSION
    ok = False
    try:
        LayerScript(D_TILE).lujob(LU_SWIGLU, 2564)
    except fmt.LayerJobIllegal as e:
        ok = "SILENTLY LEGAL" in str(e)
    chk("3a swiglu cols=2564 refused with the silent-legality message", ok)
    chk("3b lu_chunks(18944, swiglu) == 296 x 64",
        fmt.lu_chunks(18944, fmt.LU_SWIGLU) == [64] * 296)
    ok2 = False
    try:
        LayerScript(D_TILE).ljc(0x3DCCCCCD)          # ungraded 0.1f
    except AssertionError:
        ok2 = True
    chk("3c ungraded JOBC composite refused at emission", ok2)

    # 3d/3e/3f — R3 window-base rules, enforced AT EMISSION and packed right
    sb = LayerScript(D_TILE)
    sb.lujob(LU_RESID, 512, base=3)                  # 3072+512=3584 legal
    xb = _translate(sb)
    chk("3d LUJOB packs base<<14",
        any(o.get("a") == LA["JOB"]
            and o["d"] == (3 << 14) | (LU_RESID << 12) | 512
            for o in xb.ops))
    okb = False
    try:
        LayerScript(D_TILE).lujob(LU_RESID, RESID_WIN, base=3)   # 4096>3584
    except fmt.LayerJobIllegal as e:
        okb = "footprint" in str(e)
    chk("3e resid base=3 cols=1024 (footprint 4096) refused at emission", okb)
    okc = False
    try:
        LayerScript(D_TILE).lujob(LU_SWIGLU, 64, base=1)
    except fmt.LayerJobIllegal as e:
        okc = "RESIDUAL unit's" in str(e)
    chk("3f nonzero base on a non-resid unit refused", okc)

    # 4. the injection path is EXACT or it refuses
    for v in (0, 1, -1, 127, -128, 16256, -16256):
        w0, w1 = _k2_pair(v)
        assert 127 * w0 + w1 == v, v
    chk("4a K=2 injection exact over its whole range", True,
        f"|v| <= {INJ_K2_MAX}")
    for v in (16257, -16257, 1234567, -2048446, 2048446):
        lanes = _k128_col(v)
        a = [127, 1] + [127] * (D_TILE - 2)
        assert sum(int(x) * int(y) for x, y in zip(a, lanes)) == v, v
        assert all(-128 <= x <= 127 for x in lanes), v
    chk("4b K=128 injection exact past the K=2 bound", True,
        f"|v| <= {INJ_K128_MAX}")
    ref = False
    try:
        inject_kind([INJ_K128_MAX + 1])
    except InjectRangeError:
        ref = True
    chk("4c out-of-range value REFUSES (never truncates)", ref)

    # 5. the frame bound is the serializer's, and it is enforced
    s2 = LayerScript(D_TILE)
    ref2 = False
    try:
        inject_frame(s2, [0] * (SER_FRAME_MAX + 8))
    except AssertionError as e:
        ref2 = "B-SER-FRAME" in str(e)
    chk("5 frame > 2040 elements refused (B-SER-FRAME)", ref2)

    # 6. the audit catches a leaked expectation
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.jsonl"
        p.write_text(json.dumps({"op": "cap", "a": LA["RDATA"], "m": 0xFFFF,
                                 "tag": "lrd_0", "e": 1}) + "\n")
        a = audit_program(p)
        chk("6a leaked cap-with-e caught", a["violations"] == 1, f"{a}")
        p.write_text(json.dumps({"op": "r", "a": LA["RDATA"], "m": 0xFFFF,
                                 "e": 1}) + "\n")
        chk("6b checked read of RDATA caught",
            audit_program(p)["violations"] == 1)

    # 7. cap_decode's B4 extension round-trips
    caps = [{"tag": f"lrd_{i}", "sem": "lrd", "seq": i, "addr": LA["RDATA"],
             "mask": 0xFFFF, "value": b, "i": i}
            for i, b in enumerate([0x3C00, 0xBC00, 0x0000])]
    dec = cd.rdata_f16(caps)
    chk("7 rdata_f16 unpacks fp16 bits",
        [d["bits"] for d in dec] == [0x3C00, 0xBC00, 0x0000]
        and dec[0]["value"] == 1.0 and dec[1]["value"] == -1.0, f"{dec[0]}")

    # 8. THE FULL-ROW RESIDUAL'S OFFSET BOOKKEEPING (regression, 2026-07-31)
    #    build_resid_stage's `off=` is what _grade_stage slices GOLDEN by. It
    #    used to be shadowed by the window loop variable, so every slice
    #    recorded off=0 and slice k>0 would have been graded against golden's
    #    FIRST 128 elements — green only for slice 0, and silently wrong for
    #    the other 27. Build two slices of a synthetic row and prove (a) the
    #    offset survives into the manifest, (b) the returned expectation is
    #    the row AT that offset, (c) _grade_stage passes slice 1's own
    #    captures and (d) REJECTS slice 0's captures fed to slice 1.
    rng = np.random.default_rng(20260731)
    DM_T = 256
    st = {"r1_o8": rng.integers(-127, 128, DM_T, dtype=np.int64),
          "r1_comp": 0x3F800000,
          "r1_row_in_bits": rng.integers(0, 0x3C00, DM_T, dtype=np.uint16),
          "r1_bits": rng.integers(0, 0x3C00, DM_T, dtype=np.uint16)}
    with tempfile.TemporaryDirectory() as td:
        mans, exps = [], []
        for k in (0, 1):
            m, e = build_resid_stage(st, which="r1", cols=128, off=k * 128,
                                     out_dir=Path(td), name=f"st_off{k}")
            mans.append(m)
            exps.append(e)
        chk("8a off survives into the manifest",
            [m["off"] for m in mans] == [0, 128]
            and [m["cols"] for m in mans] == [128, 128],
            f"offs={[m['off'] for m in mans]}")
        chk("8b expectation is the row AT that offset",
            np.array_equal(exps[1], st["r1_bits"][128:256])
            and not np.array_equal(exps[1], exps[0]))

        def _caps(bits):
            return [{"tag": f"lrd_{i}", "sem": "lrd", "seq": i,
                     "addr": LA["RDATA"], "mask": 0xFFFF, "value": int(b),
                     "i": i} for i, b in enumerate(bits)]

        g_ok = _grade_stage(mans[1], st, {}, _caps(exps[1]), {})
        g_bad = _grade_stage(mans[1], st, {}, _caps(exps[0]), {})
        chk("8c slice-1 captures grade PASS at off=128", g_ok["equal"], f"{g_ok}")
        chk("8d slice-0 captures fed to slice 1 grade FAIL",
            not g_bad["equal"], f"n_diff={g_bad.get('n_diff')}")
        # 8e: the o8 STREAM must also come from the slice, not from row 0 —
        #     two slices of a random row cannot compile to the same program.
        chk("8e the two slice programs differ",
            Path(mans[0]["path"]).read_bytes()
            != Path(mans[1]["path"]).read_bytes())

    # 9. R4 chunked-norm tokens, legality mirror, and audit coverage
    s9 = LayerScript(D_TILE)
    s9.nsum2(120498)
    s9.nctl(ext=1, k=28)
    s9.npoll_cnt(28)
    s9.ns2cap()
    x9 = _translate(s9)
    chk("9a NSUM2/NCTL land on 0x1090/0x1094",
        x9.ops[0] == {"op": "w", "a": t2r.B_CSR + RMS_SUM2, "d": 120498}
        and x9.ops[1] == {"op": "w", "a": t2r.B_CSR + RMS_EXT,
                          "d": (28 << 2) | 2})
    chk("9b count poll masks RMS_EXT[23:16]",
        x9.ops[2] == {"op": "poll", "a": t2r.B_CSR + RMS_EXT,
                      "m": 0x00FF0000, "e": 28 << 16})
    chk("9c NS2C is ONE cap of RMS_SUM2, no expectation",
        x9.ops[3]["op"] == "cap" and x9.ops[3]["a"] == t2r.B_CSR + RMS_SUM2
        and "e" not in x9.ops[3] and x9.ops[3]["tag"].startswith("rs2"))
    n_ref = 0
    for bad_call in (lambda: LayerScript(D_TILE).nctl(ext=1, k=1),
                     lambda: LayerScript(D_TILE).nctl(ext=1, k=65),
                     lambda: LayerScript(D_TILE).nctl(sum_en=1, ext=1, k=4),
                     lambda: LayerScript(D_TILE).nsum2(1 << 28)):
        try:
            bad_call()
        except AssertionError:
            n_ref += 1
    chk("9d emission mirror refuses k=1/k=65/both-modes/29-bit sum2",
        n_ref == 4, f"{n_ref}/4 refused")
    with tempfile.TemporaryDirectory() as td:
        p9 = Path(td) / "x.jsonl"
        p9.write_text(json.dumps({"op": "r", "a": t2r.B_CSR + RMS_SUM2,
                                  "m": 0x0FFFFFFF, "e": 1}) + "\n")
        chk("9e checked read of RMS_SUM2 caught by the audit",
            audit_program(p9)["violations"] == 1)
        rng9 = np.random.default_rng(20260731)
        st9 = {"r1_8": rng9.integers(-127, 128, 256, dtype=np.int64),
               "gamma2": rng9.integers(-2000, 2000, 256, dtype=np.int64)}
        _, sums9, tot9 = _norm2_chunk_sums(st9, 256)
        chk("9f chunk sums mirror golden's own accumulation",
            sums9 == [int(np.sum(st9["r1_8"][:128] ** 2)),
                      int(np.sum(st9["r1_8"][128:] ** 2))]
            and tot9 == sum(sums9))
        m9 = build_norm2_chunked(st9, dm=256, out_dir=Path(td), name="st_n2c")
        chk("9g the armed total survives into the manifest and the program",
            m9["ext_sum2"] == tot9
            and any(json.loads(ln).get("a") == t2r.B_CSR + RMS_SUM2
                    and json.loads(ln).get("d") == tot9
                    for ln in Path(m9["path"]).read_text().splitlines())
            and m9["audit"]["violations"] == 0,
            f"ext_sum2={m9['ext_sum2']}")
        ok9 = False
        try:
            build_norm2_ext(st9, dm=256, out_dir=Path(td), name="st_bad",
                            ext_sum2=tot9 + 1)
        except AssertionError as e:
            ok9 = "REFUSE" in str(e)
        chk("9h ext stage refuses a total that is not the row's own", ok9)

        # ── 10. norm2m: the MOVED sum2 path (zero host arithmetic) ─────────
        # 10a the tile source must NOT be gated by a host recomputation (that
        #     is the whole point) — but it must still be a legal CSR value.
        m10 = build_norm2_ext(st9, dm=256, out_dir=Path(td), name="st_n2m",
                              ext_sum2=tot9 + 7, sum2_source="tile")
        wrote10 = [json.loads(ln) for ln in
                   Path(m10["path"]).read_text().splitlines()
                   if json.loads(ln).get("op") == "w"
                   and json.loads(ln).get("a") == t2r.B_CSR + RMS_SUM2]
        chk("10a tile-sourced sum2 is armed VERBATIM, not re-derived",
            m10["stage"] == "norm2m" and m10["ext_sum2"] == tot9 + 7
            and len(wrote10) == 1 and wrote10[0]["d"] == tot9 + 7
            and m10["host_total_for_reference_only"] == tot9
            and m10["audit"]["violations"] == 0,
            f"armed {m10['ext_sum2']} while the host total is {tot9} — the "
            f"grade, not the host, is the arbiter")
        # 10b an out-of-field value is still REFUSED (legality, not arithmetic)
        n_ref10 = 0
        for bad in (1 << 28, -1):
            try:
                build_norm2_ext(st9, dm=256, out_dir=Path(td), name="st_n2mb",
                                ext_sum2=bad, sum2_source="tile")
            except AssertionError:
                n_ref10 += 1
        chk("10b tile source still refuses an out-of-field CSR word",
            n_ref10 == 2, f"{n_ref10}/2 refused ([27:0], apex_top)")
        # 10c the sensitivity model: sum2 reaches the graded row ONLY via r,
        #     so +1 is invisible and delta_min is the first r-bucket exit.
        sens = norm2_grade_sensitivity(st9, 256)
        y_lo, r_lo, _ = s2m.norm_from_sum2(tot9 + sens["delta_min"] - 1,
                                           st9["r1_8"], st9["gamma2"])
        y_b, r_b, _ = s2m.norm_from_sum2(tot9, st9["r1_8"], st9["gamma2"])
        chk("10c grade sensitivity is measured, not assumed",
            sens["delta_min"] > 1 and sens["plus_one_absorbed"]
            and sens["r_at_delta"] != sens["r_base"]
            and (y_lo, r_lo) == (y_b, r_b),
            f"delta_min={sens['delta_min']} (r {sens['r_base']}->"
            f"{sens['r_at_delta']}, {sens['n_code_diff']} codes + "
            f"{sens['n_scale_diff']} scales move); delta_min-1 leaves "
            f"golden's own output bit-identical")
        # 10d the sum2-chain grader must go RED on a moved-by-one accumulator
        man10 = dict(dm=256, n_desc=2, K_real=256)

        def _rocaps(vals):
            out, i = [], 0
            for b, v in enumerate(vals):
                for j in range(MXE_N):
                    out.append({"tag": f"ro_w{j}_{b}", "sem": f"ro_w{j}",
                                "i": i, "addr": cd.RO_W0 + 4 * j,
                                "mask": 0xFFFFFFFF,
                                "value": int(v if j == 0 else 0) & 0xFFFFFFFF})
                    i += 1
                out.append({"tag": f"ro_meta_{b}", "sem": "ro_meta", "i": i,
                            "addr": cd.RO_META, "mask": 0x1, "value": 1})
                i += 1
            return out

        g_ok10 = grade_sum2_chain(man10, st9, _rocaps([sums9[0], tot9]))
        g_bad10 = grade_sum2_chain(man10, st9, _rocaps([sums9[0], tot9 + 1]))
        g_shrt = grade_sum2_chain(man10, st9, _rocaps([tot9]))
        chk("10d the sum2-chain grade is bit-exact and CAN go red",
            g_ok10["equal"] and not g_bad10["equal"] and not g_shrt["equal"]
            and g_ok10["sum2_tile"] == tot9 and g_ok10["host_adds"] == 0,
            f"total {tot9}: exact PASS, +1 FAIL, a short chain "
            f"({g_shrt.get('error', '')[:38]}…) FAIL")

    print(f"GENLAYEROPS SELFTEST: {'FAIL' if fails else 'PASS'} "
          f"(fails={fails})")
    return 1 if fails else 0


def _run(paths, *, binary, tile_div, timeout_s, cap_out):
    import tile_exec_bridge as bridge
    return bridge.run_job(paths, executor="sim", binary=binary,
                          tile_div=tile_div, timeout_s=timeout_s,
                          cap_out=cap_out)


def smoke(args) -> int:
    import tile_exec_bridge as bridge                          # noqa: F401
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = Path(args.npz) if args.npz else out / "layer_step.npz"
    if not npz.exists():
        if args.no_model:
            raise SystemExit(f"REFUSE: {npz} missing and --no-model given")
        capture_layer_step(layer=args.layer, step=args.step, out=npz,
                           n_ffn=args.n_ffn)
    plan = build_layer_step(args.layer, npz, out, resid_cols=args.resid_cols,
                            swiglu_chunks=args.swiglu_chunks,
                            norm2_dm=args.norm2_dm, rope_head=args.rope_head,
                            stages=args.stages.split(",") if args.stages
                            else None)
    z = np.load(npz, allow_pickle=False)
    step = {k: z[k] for k in z.files if k != "meta_json"}
    meta = plan["meta"]

    rows = []
    for man in plan["stages"]:
        t0 = time.time()
        r = _run([man["path"]], binary=args.binary, tile_div=args.tile_div,
                 timeout_s=args.timeout, cap_out=str(out / f"{man['name']}.cap.jsonl"))
        wall = time.time() - t0
        row = dict(name=man["name"], stage=man["stage"], ok=bool(r["ok"]),
                   rc=r["rc"], caps=len(r["captures"]), wall=wall,
                   regops=man["regops"], checks=man["checks"],
                   audit=man["audit"], summary=r.get("summary"))
        try:
            row["grade"] = _grade_stage(man, step, meta, r["captures"], plan)
        except Exception as e:                                  # noqa: BLE001
            row["grade"] = {"equal": False, "error": f"{type(e).__name__}: {e}"}
        row["blocked"] = man.get("blocked")
        row["_caps"] = r["captures"]
        row["_man"] = man
        rows.append(row)
        eprint(f"  [{man['name']}] rc={r['rc']} ok={r['ok']} "
               f"caps={len(r['captures'])} {wall:.1f}s "
               f"grade={row['grade'].get('equal')}")
    def runner(pth):
        return _run([pth], binary=args.binary, tile_div=args.tile_div,
                    timeout_s=args.timeout, cap_out=None)

    disc = {}
    if not args.no_discriminate:
        eprint("[disc] running the perturbations (each MUST go red)")
        disc = discriminate(step, plan, rows, out_dir=out, runner=runner)
    _report(args, plan, rows, disc)
    live = [x for x in rows if not x.get("blocked")]
    ok = all(x["ok"] and x["grade"].get("equal") for x in live) \
        and all(d.get("caught") for d in disc.values())
    return 0 if ok else 1


def _grade_stage(man, step, meta, caps, plan):
    st = man["stage"]
    if st in ("resid_r1", "resid_r2"):
        which = "r1" if st.endswith("r1") else "r2"
        o = int(man.get("off", 0))
        exp = np.asarray(step[f"{which}_bits"],
                         dtype=np.uint16)[o:o + man["cols"]]
        return grade_resid(caps, exp)
    if st == "swiglu":
        n = man["n_values"]
        sw = f16_bits_to_f64(np.asarray(step["swiglu_bits"],
                                        dtype=np.uint16)[:n])
        return grade_codes_scales(caps, sw, rows=man["feeder_rows"])
    if st == "norm2":
        dm = man["dm"]
        h2 = np.asarray(step["h2"], dtype=np.float64)[:dm] / 256.0
        return grade_codes_scales(caps, h2, rows=man["feeder_rows"])
    if st == "sum2m":
        # the MXE half of the norm2m pair: one INT32, accumulated ON the tile
        return grade_sum2_chain(man, step, caps)
    if st in ("norm2c", "norm2x", "norm2m"):
        # R4: the arbiter is rmsnorm_fx_wide via the step npz's own h2
        # (transformer.py:570) — the SAME reference the wide-build norm2
        # stage grades against; the composition earns equality, never a
        # second oracle.
        dm = man["dm"]
        h2 = np.asarray(step["h2"], dtype=np.float64)[:dm] / 256.0
        g = grade_codes_scales(caps, h2, rows=man["feeder_rows"])
        if st == "norm2c":
            # pass-1 exports: each captured sum2 must equal golden's OWN
            # per-chunk accumulation (rmsnorm_fx_wide step 1, replayed
            # line-for-line by _norm2_chunk_sums), and their sum must be
            # the value the program armed — closing the composition loop:
            # tile exports + host adds == the armed total.
            _, want_sums, want_total = _norm2_chunk_sums(step, dm)
            got = [c["value"] for c in caps if c.get("sem") == "rs2"]
            g["n_sums"] = len(got)
            g["sums_equal"] = bool(got == want_sums)
            g["sum_total_equal"] = bool(sum(got) == want_total
                                        == int(man["ext_sum2"]))
            g["equal"] = bool(g["equal"] and g["sums_equal"]
                              and g["sum_total_equal"])
        return g
    if st == "norm2_probe":
        return {"equal": True, "n": 0,
                "note": "checked guard: illegal ext arm REFUSED loudly "
                        "(sticky + code 8), mode not armed, W1C clears"}
    if st == "swiglu_probe":
        # the whole content of this stage is its CHECKED LAYER_STATUS
        # expectation, which the executor already enforced; there is no
        # capture to grade. `ok` from the run IS the verdict.
        return {"equal": True, "n": 0,
                "note": "checked guard: LAYER_STATUS error sticky CLEAR "
                        "(B-SWG-PHASE stays fixed)"}
    if st == "resid_probe":
        # same shape: the checked sticky+code read (and its W1C clear) IS
        # the content; the executor enforced it. Pre-R3 RTL fails the
        # bounded sticky-poll — the red half of the R3 evidence.
        return {"equal": True, "n": 0,
                "note": "checked guard: base+cols>DM_MAX REFUSED loudly "
                        "(R3 geometry refusal)"}
    if st == "rope":
        hd = man["head_dim"]
        h = man["head"]
        qr = f16_bits_to_f64(np.asarray(step["q_rope_bits"],
                                        dtype=np.uint16)[h * hd:(h + 1) * hd])
        return grade_codes_scales(caps, qr, rows=1)
    return {"equal": False, "error": f"no grader for stage {st}"}


def _report(args, plan, rows, disc=None):
    meta = plan["meta"]
    print()
    print("═" * 78)
    print("LANE B — ONE DECODER LAYER OF A REAL Qwen2.5-7B DECODE STEP "
          "THROUGH THE LAYER WINDOW")
    print("═" * 78)
    print(f"step     : layer {meta['layer']} step {meta['step']} "
          f"T={meta['T']} tier={meta['tier']} G={meta['group']} "
          f"q_pos={meta['q_pos']}")
    print(f"geometry : D_model={meta['D_model']} d_ffn={meta['d_ffn']} "
          f"H={meta['H']} H_kv={meta['H_kv']} head_dim={meta['head_dim']}")
    print(f"arbiter  : {meta['arbiter']}   source: {meta['source']}")
    print(f"executor : sim ({args.binary or 'default twin'}) "
          f"tile_div={args.tile_div}")
    print()
    print(f"{'stage':<26}{'regops':>8}{'chk':>6}{'caps':>7}{'run':>6}"
          f"{'grade':>8}{'wall':>8}")
    print("-" * 78)
    for r in rows:
        gcol = ("PASS" if r["grade"].get("equal")
                else ("BLOCKED" if r.get("blocked") else "FAIL"))
        print(f"{r['stage']:<26}{r['regops']:>8}{r['checks']:>6}"
              f"{r['caps']:>7}{'PASS' if r['ok'] else 'FAIL':>6}"
              f"{gcol:>8}{r['wall']:>7.1f}s")
    print("-" * 78)
    nv = sum(r["audit"]["violations"] for r in rows)
    nc = sum(r["audit"]["caps"] for r in rows)
    print(f"zero-output-expectation audit: {nc} caps, "
          f"{sum(r['audit']['caps_with_e'] for r in rows)} carry an "
          f"expectation, {nv} violations -> "
          f"{'PASS' if nv == 0 else 'FAIL'}")
    if disc:
        print()
        print("DISCRIMINATORS (every one MUST go red — a grader that always "
              "passes proves nothing)")
        for k, v in disc.items():
            print(f"  {k:<34} {'CAUGHT' if v.get('caught') else 'MISSED'}"
                  f"  {json.dumps({a: b for a, b in v.items() if a != 'caught'})[:90]}")
    print(f"fences applied: {json.dumps(plan['fences'])}")
    for r in rows:
        if not r["grade"].get("equal"):
            tagb = f" (BLOCKED by {r['blocked']})" if r.get("blocked") else ""
            print(f"  [{r['stage']}]{tagb} grade detail: {r['grade']}")
    print()
    print_blockers()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--blockers", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="build a SYNTHETIC step npz (debug loop, never a claim)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--npz")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--n-ffn", type=int, default=512)
    ap.add_argument("--resid-cols", type=int, default=None,
                    help="total residual cols (default: FULL D_model via R3 "
                         "window jobs; pass 128 on a narrow DM_MAX=128 twin)")
    ap.add_argument("--swiglu-chunks", type=int, default=2)
    ap.add_argument("--norm2-dm", type=int, default=None)
    ap.add_argument("--rope-head", type=int, default=0)
    ap.add_argument("--stages", default=None)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tile-div", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-discriminate", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.blockers:
        print_blockers()
        return 0
    if a.synthetic:
        out = Path(a.npz or (Path(a.out) / "layer_step_synth.npz"))
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic_layer_step(out=out, n_ffn=a.n_ffn)
        print(f"synthetic -> {out}")
        return 0
    if a.capture:
        out = Path(a.npz or (Path(a.out) / "layer_step.npz"))
        out.parent.mkdir(parents=True, exist_ok=True)
        capture_layer_step(layer=a.layer, step=a.step, out=out, n_ffn=a.n_ffn)
        print(f"captured -> {out}")
        return 0
    if a.smoke:
        return smoke(a)
    ap.error("give --selftest, --blockers, --synthetic, --capture or --smoke")


if __name__ == "__main__":
    sys.exit(main())
