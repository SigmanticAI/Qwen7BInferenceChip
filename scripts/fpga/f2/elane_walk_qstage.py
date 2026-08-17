#!/usr/bin/env python3
# elane_walk_qstage.py — E-lane E-3b demonstrator (E2E_TOY_LANE.md §4):
# the SEQUENCER stages its own rotated-q row through the gap-A q_sink.
#
# ══ WHAT THIS PROVES (and what it does not) ════════════════════════════════
# Before E-3b the fmt=1 walker never drove fsrc_ext=3: the rotated-q rows
# were staged by the HOST before WALK_GO (apex_top's own comment: "by
# design"), so a walk was not self-contained even at H=1. E-3b gives the
# walker the STAGING CHOREOGRAPHY ITSELF — mask bit EN_QSTAGE (W2_MASK[12],
# reserved-0 until now) arms PC_QSTAGE, which performs the host's exact
# verbs per head (gen_layer_ops.build_rope_stage + gen_l3_vectors.
# inject_jobs order): arm {rope-q, q_sink} + the production route
# {rdst=1, wsrc=0, asrc=0, fdst=0, qsrc=0}, push the C-1 feeder job, the
# MODE_F16 squant job, the per-step S-2 composite (descriptor word W2_QC,
# repeated D times), the serializer job (the NEW walker sj channel), then
# per 8 elements the k2-injection {loader-row act EMIT, OP_GEMM_WS m=1 k=2
# n=8 descriptor}, the act-bank-1 LOAD at row h, a whole-path drain wait,
# and a touch-one-field fsrc_ext restore at exit.
#
# THIS PROGRAM runs on the b64_05b twin (CFG_D=CFG_DM=64, QSTAGE_H_MAX=14):
#     host: load the rope phase-q table + prefill the xw FIFO with the k2
#           injection weight beats (DATA MOVEMENT — the host's legitimate
#           role; the values are the q row's k2 decomposition)
#     host: load the 64-word fmt=1 descriptor (mask = EN_QSTAGE only,
#           W2_QC = the one per-step S-2 composite), WALK_GO
#     WALKER: ...the whole staging choreography above, host silent...
#     host: drain — the staged row's C-1 scale (fs cap) and the act-bank-1
#           row 0 codes through identity GEMMs, graded vs GOLDEN:
#               codes == quant_rows_i8(rope_fx(f16(m * qc), pos)).codes
#               scale == .scale
#           with rope_fx/rope_phase_q/quant_rows_i8 all apex_golden's own.
#
# THE ONE-COMPOSITE FENCE, STATED HONESTLY: the fmt=1 model carries
# PER-TENSOR weight scales (W2_WSCALE0), so the q projection's S-2
# composite is ONE value per step — W2_QC. The demonstrator's subject is
# therefore a q row whose elements are acc*qc for integer acc (the k2
# injection's reachable set). Subjects needing per-element composites (the
# TB-side inject scaffolding's general case) are OUT of this step's scope
# by construction — that is the same host-loaded-calibration shape as the
# RQ and JC tables, not a hidden limitation.
#
# RED/GREEN: --smoke --binary <pre-E-3b twin> must go RED at the walk
# itself — the OLD walk_desc2_check refuses W2_MASK[12] (reserved-bit
# clause, WALK_ERR_DESC), the refused walker consumes NOTHING, and the
# host's xw stream-pacing poll therefore stalls loudly ("pw stall", the
# FIFO never drains) — measured 2026-08-04 on obj_b64_05b_pre_e3b. The
# walk-off discriminator (mask = EN_ROPE, same prep, same drain) must go
# RED on the NEW twin: the walker arms nothing, so the same stall fires
# and nothing golden is ever produced.
#
# E-4b RED/GREEN (2026-08-04, this delta): walk_e4 is ONE KICK now — mask
# {QSTAGE, SCORE, PV, NFEED, NSRC}, the walker owning l_nsrc (the E-4
# measured finding retired; see the E-4 section below). Its twin proof is
# `--redproof --binary <pre-E-4b twin>` (e.g. obj_b64_05b_e3b): the OLD
# check refuses W2_MASK[13] as a reserved bit, so the single-kick image
# must go RED there, loudly, before anything is consumed. On the NEW twin
# the smoke also proves the two fence refusals (e4_fence_nonsrc: the
# measured two-kick wedge shape {QSTAGE,SCORE,PV,NFEED} without NSRC;
# e4_fence_nsrconly: NSRC without NFEED) — both must be REFUSED with
# WALK_ERR_DESC and state unchanged.
#
# CLI:
#   python3 scripts/fpga/f2/elane_walk_qstage.py --selftest
#   python3 scripts/fpga/f2/elane_walk_qstage.py --build [--out DIR]
#   python3 scripts/fpga/f2/elane_walk_qstage.py --smoke [--binary PATH]
#       [--tile-div N] [--out DIR] [--no-discriminate]
#   python3 scripts/fpga/f2/elane_walk_qstage.py --redproof --binary PATH
#       [--tile-div N] [--out DIR]      # pre-E-4b twin: walk_e4 must be RED

from __future__ import annotations

import argparse
import json
import struct
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

import gen_layer_ops as glo                                    # noqa: E402
import gen_l3_vectors as g3                                    # noqa: E402
import cap_decode as cd                                        # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
import tile_geom as tg                                         # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import transformer as tf                      # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

from gen_layer_ops import (                                    # noqa: E402
    LayerScript, BANK_PHQ, L_CTRL,
    _translate, _emit, _prologue, _drain_and_check,
    identity_readback, eprint,
)
from elane_norm_feed import disc_ran                           # noqa: E402

W_CTRL, W_DPTR, W_DDATA, W_STATUS = 0x5C, 0x60, 0x64, 0x68
W_ST_BUSY, W_ST_ESTK, W_ST_ECODE = 0x0001, 0x0100, 0x0E00
W_ST_FMTSUP = 0xF000
FMT_SUP_LAYER = 0x3

D_TOY = 64
POS_M = 5                      # the rope position this walk stages at
DEFAULT_OUT = REPO / "build" / "f2_elane_qstage"


def _f32_val(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


# ═══════════════ the toy subject + golden references ═══════════════════════

def toy_qstep(d: int = D_TOY, pos: int = POS_M, seed: int = 20260804,
              rows: int = 1) -> dict:
    """The synthetic q row and its golden staging references.

    Subject: q[i] = f16(m[i] * qc), m integer in the k2-injection range,
    qc = the ONE per-step composite (2^-6, fp16-grade f32). References,
    ALL golden's own functions applied golden's own way:
      phase table   rope_phase_q(pos, rope_theta(d))     (the RAM image)
      rotated row   rope_fx(q_f16_vals, pos)             (fp16-narrowed)
      staged codes  quant_rows_i8(rotated)               (C-1)
    """
    rng = np.random.default_rng(seed)
    qc_bits = fmt.grade_f32(2.0 ** -6)
    qc = _f32_val(qc_bits)
    m = rng.integers(-16000, 16001, (rows, d), dtype=np.int64)
    q_bits = f64_to_f16_bits(m.astype(np.float64) * qc)
    q_vals = f16_bits_to_f64(np.asarray(q_bits, dtype=np.uint16))
    theta = tf.rope_theta(d)
    phase = np.asarray(tf.rope_phase_q(float(pos), theta), dtype=np.int64)
    rot = tf.rope_fx(q_vals, float(pos), theta)
    codes, scales = at.quant_rows_i8(np.asarray(rot, dtype=np.float64))
    out = dict(seed=int(seed), d=int(d), pos=int(pos), rows=int(rows),
               qc_bits=int(qc_bits),
               q_bits=np.asarray(q_bits, dtype=np.uint16),
               phase=phase,
               q_codes_rows=np.asarray(codes, dtype=np.int64),
               q_scales_rows=[int(v) for v in scales])
    # the H=1 view (every existing caller)
    out["m"] = m[0] if rows == 1 else m
    out["q_bits"] = (np.asarray(q_bits, dtype=np.uint16)[0] if rows == 1
                     else np.asarray(q_bits, dtype=np.uint16))
    out["q_codes"] = out["q_codes_rows"][0]
    out["q_scale"] = out["q_scales_rows"][0]
    return out


def qstage_descriptor(step: dict, *, qstage: bool = True) -> list[int]:
    """The 64-word fmt=1 image: H=1, head_dim == d_model == CFG_D, mask =
    EN_QSTAGE alone (the subject is the staging step). qstage=False is the
    walk-off discriminator: same LEGAL image, EN_ROPE stands in (the
    nfeed=False idiom of elane_walk_norm)."""
    d = step["d"]
    rows = int(step.get("rows", 1))
    n_kv = 2 if rows == 14 else 1        # the 0.5B GQA 14:2 geometry
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(d)
    w[fmt.W_MODEL0] = fmt.pack_model0(rows * d, 0)
    w[fmt.W_MODEL1] = fmt.pack_model1(rows, n_kv, 1)
    w[fmt.W_MASK] = fmt.pack_mask((1 << fmt.EN_QSTAGE) if qstage
                                  else (1 << fmt.EN_ROPE))
    w[fmt.W_STEP] = fmt.pack_step(1, step["pos"])
    w[fmt.W_QC] = step["qc_bits"]
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=d) == fmt.ERR_NONE
    return w


def load_descriptor(s: LayerScript, words: list[int]):
    s.csrw(W_DPTR, 0)
    for v in words:
        s.csrw(W_DDATA, v)


def _k2_beats(m: np.ndarray) -> list[list[int]]:
    """The k2-injection weight beats, group order == the walker's desc
    order (8 values per group, one packed halfword per column)."""
    m = np.asarray(m).reshape(-1)
    groups = []
    for g0 in range(0, len(m), 8):
        packed = []
        for v in m[g0:g0 + 8]:
            w0 = int(round(int(v) / 127.0))
            w1 = int(v) - 127 * w0
            assert -128 <= w0 <= 127 and -128 <= w1 <= 127
            packed.append((w0 & 0xFF) | ((w1 & 0xFF) << 8))
        groups.append(packed)
    return groups


# ═══════════════ the walked program ════════════════════════════════════════

def build_walk_qstage(step: dict, *, out_dir: Path, name: str,
                      qstage: bool = True) -> dict:
    """The E-3b walk: the SEQUENCER stages the rotated-q row.

    qstage=False = the walk-off discriminator: identical prep, identical
    drain, the step-enable bit swapped for EN_ROPE — the walker then arms
    the rope level and STAGES NOTHING, so the LAYER_CTRL readback (bank 0,
    q_sink never armed) and the staged-row grade both go red."""
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// E-LANE E-3b stage {name}: a WALKED fmt=1 program stages the "
           f"rotated-q row through the gap-A q_sink — feeder job, MODE_F16 "
           f"job, composite sideband, serializer job, k2 injection, act "
           f"bank-1 LOAD, all from the SEQUENCER. walk_en holds the host's "
           f"job/route/level ports off while it runs.")
    _prologue(s)

    s.emit("// [1] host loads the rope phase-q table (LAYER_PTR bank 1) — "
           "weights/inputs, its legitimate role")
    s.lptr(BANK_PHQ, 0)
    s.ldata(step["phase"])

    s.emit("// [2] the 64-word fmt=1 image (mask = EN_QSTAGE; W2_QC = the "
           "one per-step S-2 composite) + FMT_SUP check")
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, qstage_descriptor(step, qstage=qstage))

    s.emit("// [3] WALK_GO, then the host STREAMS the k2 injection weight "
           "beats into the xw FIFO while the walker runs — pure DATA "
           "MOVEMENT (the q row's k2 decomposition), paced by the FIFO-"
           "free poll: the walker's own descriptors are what consume them. "
           "The host issues NO job, level or route in this window — the "
           "sequencer alone drives (the selftest proves it on the emitted "
           "program).")
    s.pmode(True)
    s.csrw(W_CTRL, 0x3)
    for packed in _k2_beats(step["m"]):
        s.wbeats(packed)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit("// checked: the walk retired CLEAN — on the pre-E-3b twin this "
           "read goes RED instead (the old check refuses W2_MASK[12] as a "
           "reserved bit, WALK_ERR_DESC sticky)")
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)

    s.emit("// [5] the level the walker LEFT: rope-q armed, q_sink "
           "RESTORED to 0 (the touch-one-field exit) — a full-word check")
    s.csrr(L_CTRL, 0xFFFFF,
           fmt.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"]))

    s.emit("// [6] pop the staged rows' C-1 scales (the values the sq file "
           "also captured for the head interlock) and read the staged codes "
           "back from ACT BANK 1 rows 0..H-1 through identity GEMMs — the "
           "only act-stage egress; produce-mode caps, golden the arbiter")
    for _ in range(int(step.get("rows", 1))):
        s.efs(0, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for row in range(int(step.get("rows", 1))):
        for base in range(0, d, glo.MXE_N):
            identity_readback(s, row, base, bank=1)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"E-LANE E-3b {name}: WALKED q staging")
    man = _emit(x, out_dir, name, dict(
        stage="walk_qstage" if qstage else "walk_qstage_off",
        kind="WALKED: rope-q arm + q_sink + fj/qj/qs/sj + k2 inject + act "
             "bank1 LOAD; host: phase table + xw beats + drain",
        d=int(d), pos=step["pos"], qstage=bool(qstage),
        walk_mask=f"{qstage_descriptor(step, qstage=qstage)[fmt.W_MASK]:#06x}",
        qc=f"{step['qc_bits']:#010x}", seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    return man


# ═══════════════ E-4: the combined maximal walk ════════════════════════════
# ONE fmt=1 descriptor, ONE kick — mask {QSTAGE, SCORE, PV, NFEED, NSRC}
# (E-4b, 2026-08-04): the sequencer stages q, walks the whole attention
# (score+pv on the real datapath, o8 requantised on-tile), then feeds the
# resident row X to the RMSNorm in-tile — four of the layer's steps in one
# walk, the host writing NOTHING between walker start and completion (the
# paced xw DATA stream is the one disclosed exception, and it is data, not
# control). The previously-reserved mask bit EN_NSRC (W2_MASK[13]) makes
# the WALKER the owner of l_nsrc (LAYER_CTRL[19]) for the walk: QSTAGE arms
# it 0 (codes -> act stage), the NFEED entry arms it 1 (codes -> norm) on
# the same edge as the code-4 route — the two-kick shape this program
# measured first (l_nsrc HOST-held, one feeder output, no single value
# serves both) is retired, and a {QSTAGE, NFEED} image WITHOUT the bit is
# now REFUSED at S2_CHECK (WALK_ERR_DESC) instead of wedging at the staging
# drain. Golden the only arbiter:
#   staged q   quant_rows_i8(rope_fx(f16(m*qc), pos))     (E-3b's claim)
#   attention  at.attention_core(rot, K_f16, V_f16, CQ8)  (o8 BIT-EXACT,
#              s_k/s_v/s_q/s_c scale provenance)          (walk_job's claim
#                                                          shape, fmt=1)
#   norm feed  rmsnorm_fx(quant_rows_i8(X).codes, gamma)  (E-3a's claim at
#              the layer-entry row)
# Host-loaded calibration: RQ[0] = golden's own calib pair (gold.rq — the
# descriptor's documented host-loaded field, B1_WALKER.md §2 correction 1);
# W2_QC as in the staging walk.

def toy_e4(d: int = D_TOY, pos: int = POS_M, seed: int = 20260814) -> dict:
    rng = np.random.default_rng(seed)
    st = toy_qstep(d, pos, seed)
    theta = tf.rope_theta(d)
    rot = tf.rope_fx(f16_bits_to_f64(st["q_bits"]), float(pos), theta)
    K16 = f64_to_f16_bits(rng.normal(0.0, 0.8, (1, d)))
    V16 = f64_to_f16_bits(rng.normal(0.0, 0.8, (1, d)))
    gold = at.attention_core(np.asarray(rot, dtype=np.float64),
                             np.asarray(K16, dtype=np.uint16),
                             np.asarray(V16, dtype=np.uint16),
                             at.TIER_CQ8, G=128)
    x_bits = f64_to_f16_bits(rng.normal(0.0, 1.2, d))
    x_vals = f16_bits_to_f64(np.asarray(x_bits, dtype=np.uint16))
    x_codes, x_scales = at.quant_rows_i8(x_vals[None, :])
    g2 = rng.integers(-12288, 12289, d, dtype=np.int64)
    h2, _r, _n = at.rmsnorm_fx([int(c) for c in x_codes[0]],
                               [int(v) for v in g2])
    h2_codes, h2_scales = at.quant_rows_i8(
        np.asarray(h2, dtype=np.float64)[None, :] / 256.0)
    st.update(dict(
        K16=np.asarray(K16, dtype=np.uint16),
        V16=np.asarray(V16, dtype=np.uint16),
        gold_o8=np.asarray(gold.o8, dtype=np.int64),
        gold_sk=[int(v) for v in gold.s_k],
        gold_sv=[int(v) for v in gold.s_v],
        gold_sq=int(gold.s_q), gold_sc=int(gold.s_c),
        rq=(int(gold.rq[0]), int(gold.rq[1])),
        x_bits=np.asarray(x_bits, dtype=np.uint16),
        x_scale=int(x_scales[0]),
        gamma2=g2,
        h2_codes=np.asarray(h2_codes[0], dtype=np.int64),
        h2_scale=int(h2_scales[0])))
    return st


E4_MASK = ((1 << fmt.EN_QSTAGE) | (1 << fmt.EN_SCORE) | (1 << fmt.EN_PV)
           | (1 << fmt.EN_NFEED) | (1 << fmt.EN_NSRC))
# the two S2_CHECK fence shapes this lane must see REFUSED (pkg-legal, so
# the FENCE is what fires — the refuse2 idiom, on the real tile):
#   nonsrc    the pre-E-4b two-kick image with NFEED forced onto kick 1 —
#             the MEASURED wedge shape (host-held l_nsrc serves either
#             codes->act or codes->norm, never both)
#   nsrconly  NSRC without NFEED — ownership of the norm x-mux with nothing
#             to feed it (would pin l_nsrc 0 over a host arm)
E4_MASK_NONSRC = E4_MASK & ~(1 << fmt.EN_NSRC)
E4_MASK_NSRCONLY = E4_MASK & ~(1 << fmt.EN_NFEED)


def e4_descriptor(step: dict, *, mask: int = E4_MASK) -> list[int]:
    d = step["d"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(d)
    w[fmt.W_MODEL0] = fmt.pack_model0(d, 0)
    w[fmt.W_MODEL1] = fmt.pack_model1(1, 1, 1)
    # E-4b: ONE kick = {QSTAGE, SCORE, PV, NFEED, NSRC}. The first shape of
    # this program measured why NFEED could not ride the QSTAGE kick (lctl
    # 0x583 + LAYER_STATUS[5] stuck, attention already complete — l_nsrc
    # was HOST-held); EN_NSRC hands the level to the walker and fuses the
    # kicks. Every mask this builder accepts must pass the PKG check —
    # the fence images are S2_CHECK-refused, which is the point.
    w[fmt.W_MASK] = fmt.pack_mask(mask)
    w[fmt.W_STEP] = fmt.pack_step(1, step["pos"])
    w[fmt.W_RQ0] = fmt.pack_rq(*step["rq"])
    w[fmt.W_QC] = step["qc_bits"]
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=d) == fmt.ERR_NONE
    return w


def build_walk_e4(step: dict, *, out_dir: Path, name: str) -> dict:
    from elane_norm_feed import LST_NFEED
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// E-LANE E-4b stage {name}: ONE fmt=1 walk, ONE KICK = "
           f"QSTAGE + SCORE + PV + NFEED on the real datapath (toy {d}, "
           f"H=1, T=1). Host: phase table, KV records, X, calibration "
           f"slots, the paced xw data stream, and the post-walk drains — "
           f"NOTHING between walker start and completion. EN_NSRC makes "
           f"the WALKER own l_nsrc: 0 through staging/attention, 1 at the "
           f"NFEED entry — the level flip that used to be the one host "
           f"verb between kicks.")
    _prologue(s)
    s.emit("// [1] host loads: rope phase-q table; the K and V records "
           "(T=1) through squant MODE_F16 into the KVQ bank (the L3 store "
           "idiom — this is what populates the walker's scale caches); the "
           "layer-input row X into the residual RAM. l_nsrc is NOT armed "
           "here: the walker owns it (E-4b).")
    s.lptr(BANK_PHQ, 0)
    s.ldata(step["phase"])
    # store_kv_phase emits an L3-TB-side TAPF16 observability line that has
    # no regops translation (the tap is a TB pin, not a BAR0 access) —
    # dropped at emission, exactly as the trace replay path drops it.
    _tap = s.tap
    s.tap = lambda *a, **k: None
    g3.store_kv_phase(s, step["K16"], step["V16"])
    s.tap = _tap
    s.lptr(glo.BANK_RESID, 0)
    s.ldata(step["x_bits"])
    s.route(rdst=0, asrc=0)

    s.emit("// [2] the 64-word fmt=1 image: mask {QSTAGE, SCORE, PV, "
           "NFEED, NSRC} — the whole chain in one image; RQ[0] = the "
           "golden calib pair (host-loaded, the descriptor's documented "
           "per-step calibration); W2_QC. This program's FIRST shape "
           "measured why two kicks were needed (l_nsrc HOST-held, E-2a); "
           "the E-4b RTL delta hands the level to the walker via the "
           "previously-reserved W2_MASK[13] and retires the finding.")
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, e4_descriptor(step))

    s.emit("// [3] THE ONE KICK + the xw data stream (paced): the "
           "SEQUENCER stages q (E-3b), walks score+pv on the staged row "
           "(o8 requantised ON-TILE with RQ[0]), then arms l_nsrc=1 WITH "
           "code 4, frames the row, pushes unit 3 and waits nf_busy — the "
           "whole chain, no host verb inside.")
    s.pmode(True)
    s.csrw(W_CTRL, 0x3)
    for packed in _k2_beats(step["m"]):
        s.wbeats(packed)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit("// checked: the walk retired CLEAN — on a pre-E-4b binary "
           "this program goes RED at/before this read instead (W2_MASK[13] "
           "is a reserved bit there: WALK_ERR_DESC sticky, nothing "
           "consumes the xw stream)")
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    s.lstat(0x1F00 | LST_NFEED, 0x0)
    s.emit("// [3b] the level word the WALKER left, read on the real "
           "register: rope-q armed (QSTAGE), fsrc_ext = code 4 AND l_nsrc "
           "= 1 (the NFEED arm) — LAYER_CTRL[19] set by the SEQUENCER, the "
           "tile-level proof the walk-mode mux copied the owned level. A "
           "full-word check, bits [19:0].")
    s.csrr(L_CTRL, 0xFFFFF,
           fmt.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"],
                    fsrc_ext=4, nsrc=1))

    s.emit("// [4] drain the walk's egress, in FIFO order: 1 ss (s_c from "
           "the walked S-4; MEASURED: MODE_F16 emits no s-tap beat, so the "
           "staged q's scale provenance is the fs beat + the sq-file "
           "replay), 11 fs (staged-q scale, s_k, 8x s_v, the NFEED X "
           "scale), 8 RO beats (the PV o8 codes) — produce-mode caps, "
           "golden-graded")
    s.ess(0, 0)
    for _ in range(11):
        s.efs(0, 0)
    for _ in range(8):
        s.ero([0] * glo.MXE_N, 1)
    s.emit("// the walked score's TIP decision beat (block 0 — rt_tip_blk "
           "is a host route level, reset value): popped + blk-checked")
    s.emit("ETIP", "0")

    s.emit("// [5] gamma releases the norm's emission (the E-3a drain): "
           "levels to 0, act LOAD, feeder job, gamma, h2 scale + codes "
           "through identity GEMMs, X observability")
    s.csrw(L_CTRL, 0)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.fjob(1)
    s.grow_n(step["gamma2"])
    s.efs(0, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, d, glo.MXE_N):
        identity_readback(s, 0, base)
    s.lrptr(0)
    s.erd(d)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"E-LANE E-4b {name}: one-kick "
                           f"QSTAGE+SCORE+PV+NFEED walk (NSRC-owned)")
    man = _emit(x, out_dir, name, dict(
        stage="walk_e4",
        kind="ONE WALK, ONE KICK: q staged + score+pv (o8 on-tile requant) "
             "+ X->norm in-tile, l_nsrc walker-owned (EN_NSRC); host = "
             "data movement + calibration + drains",
        d=int(d), pos=step["pos"], rq=list(step["rq"]),
        walk_mask=f"{e4_descriptor(step)[fmt.W_MASK]:#06x}",
        seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    return man


def build_e4_refusal(step: dict, *, out_dir: Path, name: str,
                     mask: int, why: str) -> dict:
    """E-4b refusal fences ON THE REAL TILE (the refuse2 idiom): load a
    pkg-LEGAL image whose mask combination the S2_CHECK fence refuses, kick,
    and CHECK the refusal — busy fell, ESTK set, err_code = WALK_ERR_DESC
    (2), and the walker consumed nothing (no xw stream is offered, so a
    walker that WALKED would wedge and fail the busy poll loudly)."""
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// E-LANE E-4b fence {name}: {why} -> the walk must be "
           f"REFUSED at S2_CHECK (WALK_ERR_DESC), state unchanged, "
           f"nothing armed, nothing consumed.")
    _prologue(s)
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, e4_descriptor(step, mask=mask))
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit("// the refusal receipt: sticky + code 2 (DESC), busy down")
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE,
           W_ST_ESTK | (2 << 9))
    s.csrw(W_CTRL, 0x0)
    s.emit("// refusal leaves state unchanged: the levels never moved")
    s.csrr(L_CTRL, 0xFFFFF, 0x0)
    s.emit("DONE")
    x = _translate(s, note=f"E-LANE E-4b fence {name}")
    man = _emit(x, out_dir, name, dict(
        stage=name, kind=f"REFUSAL FENCE: {why}",
        d=int(d), walk_mask=f"{mask:#06x}", seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    return man


def grade_e4(caps, step: dict) -> dict:
    d = step["d"]
    out = {}
    # ss: the walked S-4's s_c (MODE_F16 emits no s-tap beat — measured;
    # the staged q's scale provenance is fs[0] + the sq-file replay, whose
    # correctness the walked SCORE's own composites prove)
    ss = [t for t in cd.fp16_bits(caps) if t["sem"] == "ss"]
    ss_bits = [int(t["bits"]) for t in ss]
    out["s_c"] = {"equal": len(ss_bits) >= 1
                  and ss_bits[0] == step["gold_sc"],
                  "got": ss_bits[:1], "want": step["gold_sc"]}
    # fs: staged-q scale, s_k, 8x s_v, X scale, then the h2 scale
    fs = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    want_fs = ([step["q_scale"]] + step["gold_sk"]
               + step["gold_sv"] * 8 + [step["x_scale"], step["h2_scale"]])
    out["fs"] = {"equal": fs == want_fs, "n": len(fs), "want_n": len(want_fs)}
    # RO: first 8 beats = the PV o8 codes; the rest = the h2 identity chain
    acc = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64)
    o8 = acc[:d]
    h2 = acc[d:2 * d]
    out["o8"] = {"equal": o8.size == d
                 and np.array_equal(o8, step["gold_o8"]),
                 "n_diff": int(np.sum(o8[:min(d, o8.size)]
                                      != step["gold_o8"][:min(d, o8.size)]))}
    out["h2_codes"] = {"equal": h2.size == d
                       and np.array_equal(h2, step["h2_codes"])}
    # X row observability
    from gen_layer_ops import grade_resid
    out["x_row"] = {"equal": bool(grade_resid(caps, step["x_bits"])["equal"])}
    out["equal"] = all(out[k]["equal"] for k in
                       ("s_c", "fs", "o8", "h2_codes", "x_row"))
    return out


# ═══════════════ E-6: the one-kick chain WITH the walked epilogue ══════════
# ONE fmt=1 descriptor, ONE kick — mask {SCORE, PV, OPROJ, RES1, NFEED,
# NSRC, FPROJ}: the sequencer walks the whole attention on the pre-staged q
# row (o8 requantised on-tile), then FETCHES Wo FROM DDR BY ITS OWN RECORD,
# runs the o-projection with the requant epilogue ON-TILE (RT_FPRQ: the o8'
# posts to the serializer), dequantises it through apex_layer_deq (the JC
# composite the descriptor carries), adds it onto the resident X row in
# apex_residual (r1 = f16(X + o8'*comp), IN the tile), and feeds THAT row —
# the epilogue's own product — to the RMSNorm via NFEED/NSRC. Host between
# WALK_GO and completion: NOTHING (not even a data-plane drain: the o8'
# stream never reaches the RO FIFO).
#
# HONESTY, measured and disclosed:
#   * q is HOST-PRE-STAGED (the l3 walker-mode idiom), NOT walker-staged:
#     the FPROJ fence requires fuel mode armed at S2_CHECK, and fuel_src=1
#     disconnects the mailbox xw stream QSTAGE's k2 injection rides
#     (cl_apex.sv:1074) — QSTAGE+FPROJ is a loud S2_CHECK refusal
#     (e6_fence_qstage below), so the E-4b walker-staged q and the E-6
#     fuel-fed epilogue cannot yet share one kick. Fusing them needs a
#     fuel-delivered staging path — named follow-on, not faked here.
#   * OPROJ's ACTIVATION is a host-staged COPY of the attention output (the
#     PV o8 codes, staged at act bank1 row 1 = the walker's fp_oproj_base
#     row, pre-walk): the tile's own o8 egresses on the RO lanes and is
#     graded bit-exact against the same values, but the o8 -> act link is
#     NOT an in-tile datapath. The V-flip discriminator MEASURES this seam:
#     it moves the walked o8 and NOT r1 (the copy was staged from the
#     unperturbed prediction) — reported as the seam detector, on purpose.
#   * Wo bytes are the FIRST 8 n-sub-blocks of the REAL resident L00_Wo
#     region (a d=64 walk consumes 4096 bytes of the 896-wide image, so
#     job j reads Wo[64j:64j+64, 0:8] — the image's own k-major sub-block
#     order); golden grades the SAME bytes the walker fetched.

E6C_MASK = ((1 << fmt.EN_SCORE) | (1 << fmt.EN_PV) | (1 << fmt.EN_OPROJ)
            | (1 << fmt.EN_RES1) | (1 << fmt.EN_NFEED) | (1 << fmt.EN_NSRC)
            | (1 << fmt.EN_FPROJ))
DEFAULT_IMAGE = REPO / "build" / "ddr_weights_05b"


def _wo_subblocks(image_dir: Path, d: int) -> tuple[int, list, float]:
    """Decode the first d/8 n-sub-blocks of the resident L00_Wo region —
    byte (p*8+c)*8+r == W[8p+r][c] (make_weight_image block order) — plus
    the real per-tensor scale from the weights meta."""
    man = json.loads((image_dir / "ddr_image.json").read_text())
    wo = next(t for t in man["tensors"]
              if t["layer"] == 0 and t["tensor"] == "Wo")
    img = (image_dir / "ddr_image.bin").read_bytes()
    base_b = wo["base_64B"] * 64
    blocks = []
    for j in range(d // 8):
        blk = img[base_b + 512 * j: base_b + 512 * (j + 1)]
        W = np.zeros((d, 8), dtype=np.int64)
        for p in range(8):
            for c in range(8):
                for r in range(8):
                    v = blk[(p * 8 + c) * 8 + r]
                    W[8 * p + r, c] = v - 256 if v >= 128 else v
        blocks.append(W)
    meta = json.loads((Path(man["weights_dir"]) / "meta.json").read_text())
    return int(wo["base_64B"]), blocks, float(meta["layers"][0]["s_Wo"])


def toy_e6c(d: int = D_TOY, pos: int = POS_M, seed: int = 20260815,
            image_dir: Path | None = None) -> dict:
    from apex_golden import compute as cpg
    from walk_fuel_proj import _deq_resid_fit
    st = toy_e4(d, pos, seed)
    image_dir = Path(image_dir or DEFAULT_IMAGE)
    wo_base, blocks, s_wo = _wo_subblocks(image_dir, d)
    # the STAGED activation is the C-1 quantisation of the o8 row — the
    # staging inject is MODE_QUANT, and the toy o8 row's amax is NOT 127
    # (the identity-scale shortcut only holds for c1_frames subjects), so
    # the stored codes are quant_rows_i8(o8_vals) with the row's own scale.
    # Golden composes on exactly those codes; the ess check carries the
    # actual scale bits (measured RED first: ess 0x3C00 vs the tile's
    # 0x3BF0 — the fence that caught the shortcut).
    o_vals = np.asarray(st["gold_o8"], dtype=np.float64)
    o_act_m, o_act_sb = at.quant_rows_i8(o_vals[None, :])
    o_act = np.asarray(o_act_m[0], dtype=np.int64)
    acc = np.concatenate([cpg.gemm_i8_ksplit(o_act[None, :], W)[0]
                          for W in blocks]).astype(np.int64)
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = at.calib_requant(amax)
    o8p = cpg.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    comp_bits = fmt.grade_f32(s_wo * float(1 << shift) / float(scale))
    comp = _f32_val(comp_bits)
    prods = o8p.astype(np.float64) * comp
    bad = [i for i, v in enumerate(prods) if not _deq_resid_fit(float(v))]
    if bad:
        raise SystemExit(f"REFUSE: e6 subject trips the residual window at "
                         f"idx {bad[0]} — would RESID_WINDOW, not compute")
    x_vals = f16_bits_to_f64(np.asarray(st["x_bits"], dtype=np.uint16))
    r1_bits = np.asarray(f64_to_f16_bits(x_vals + prods), dtype=np.uint16)
    r1_codes, r1_scales = at.quant_rows_i8(f16_bits_to_f64(r1_bits)[None, :])
    h2, _r, _n = at.rmsnorm_fx([int(c) for c in r1_codes[0]],
                               [int(v) for v in st["gamma2"]])
    h2_codes, h2_scales = at.quant_rows_i8(
        np.asarray(h2, dtype=np.float64)[None, :] / 256.0)
    st.update(dict(
        image_dir=str(image_dir), wo_base=wo_base, wo_blocks=blocks,
        o_act=o_act, o_act_scale=int(o_act_sb[0]),
        acc_o=acc, rq_o=(int(scale), int(shift)), o8p=o8p,
        comp_bits=int(comp_bits), comp=float(comp), r1_bits=r1_bits,
        r1_scale=int(r1_scales[0]),
        h2r_codes=np.asarray(h2_codes[0], dtype=np.int64),
        h2r_scale=int(h2_scales[0])))
    return st


def e6_descriptor(step: dict, *, mask: int = E6C_MASK) -> list[int]:
    w = e4_descriptor(step, mask=mask)
    w[fmt.W_TENS0 + fmt.TENS_WO] = step["wo_base"]
    w[fmt.W_RQ0 + 1] = fmt.pack_rq(*step["rq_o"])      # RQ[H], H=1
    w[fmt.W_JC0 + fmt.JC_OPROJ] = step["comp_bits"]    # the deq composite
    return w


def build_walk_e6c(step: dict, *, out_dir: Path, name: str,
                   mask: int = E6C_MASK) -> dict:
    from elane_norm_feed import LST_NFEED
    from walk_fuel_proj import FUEL_MARK, splice_fuel_arm
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// E-LANE E-6 chain {name}: ONE fmt=1 walk, ONE KICK = "
           f"SCORE + PV + OPROJ(FUEL-FED, o8 EPILOGUE ON-TILE) + RES1 + "
           f"NFEED (toy {d}, H=1, T=1). The o-projection's product lands "
           f"in the residual row THROUGH ser->deq->residual and THAT row "
           f"feeds the norm — the epilogue's activation never leaves the "
           f"tile. Host in the walk window: NOTHING.")
    _prologue(s)
    s.emit("// [1] host loads: phase-q table; K/V records; X; the o-act "
           "family (the PV o8 codes) at act bank1 ROW 1 (fp_oproj_base); "
           "then PRE-STAGES the rotated-q row at ROW 0 through the q_sink "
           "(the l3 walker-mode idiom — staging rides the mailbox xw "
           "stream, which the fuel arm below DISCONNECTS, so it must all "
           "happen first).")
    s.lptr(BANK_PHQ, 0)
    s.ldata(step["phase"])
    _tap = s.tap
    s.tap = lambda *a, **k: None
    g3.store_kv_phase(s, step["K16"], step["V16"])
    s.tap = _tap
    s.lptr(glo.BANK_RESID, 0)
    s.ldata(step["x_bits"])
    s.emit("// (a) the o-act staging -> bank1 row 1 (counted ess = the "
           "row's own C-1 scale; the stored codes are quant(o8 vals))")
    s.route(rdst=1, asrc=1)
    o_pairs = [g3.decompose_f16(int(b))[:2]
               for b in g3.to16(np.asarray(step["gold_o8"],
                                           dtype=np.float64))]
    g3.inject_jobs(s, o_pairs, 1)
    s.ess(step["o_act_scale"], 1)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 1)
    s.pmode(True)
    s.emit("// (b) the rotated-q staging -> bank1 row 0 (rope-q + q_sink; "
           "fs cap = the staged row's C-1 scale; then the q_sink restore)")
    s.route(rdst=1, asrc=0)
    s.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"], fsrc_ext=3)
    s.fjob(1)
    q_pairs = [g3.decompose_f16(int(b))[:2]
               for b in np.asarray(step["q_bits"], dtype=np.uint16)]
    g3.inject_jobs(s, q_pairs, 0, waddr=None)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.efs(0, 0)
    s.lpoll(glo.LST_ROPE, 0x0)
    s.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"])
    s.route(rdst=0, asrc=0)
    s.emit(f"// {FUEL_MARK}")
    s.emit("// [2] the 64-word fmt=1 image: mask {SCORE, PV, OPROJ, RES1, "
           "NFEED, NSRC, FPROJ} + Wo base + RQ[1] + JC[OPROJ] — the whole "
           "chain in one image; FMT_SUP check first")
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, e6_descriptor(step, mask=mask))
    s.emit("// [3] THE ONE KICK — and NOTHING else: no xw pacing (q was "
           "pre-staged), no RO drain (the epilogue's o8' never reaches the "
           "RO FIFO). The next host op is the completion poll.")
    s.emit("// CYCMARK e6go")
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit("// CYCMARK e6done")
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    from walk_fuel_proj import FUEL_UNMARK
    s.emit(f"// {FUEL_UNMARK}")
    s.emit("// epilogue consumers drained CLEAN; no LAYER error")
    s.lpoll(glo.LST_LDQ | glo.LST_RES, 0x0)
    s.lstat(0x1F00 | LST_NFEED | glo.LST_LDQ | glo.LST_RES, 0x0)
    s.emit("// [3b] the level word the WALKER left: LVLO's ser_dst=1 + "
           "resid_arm + rope re-arm (bank 0), then NFEED's code-4 + nsrc=1 "
           "— LAYER_CTRL[19] set by the SEQUENCER. Full-word check.")
    s.csrr(L_CTRL, 0xFFFFF,
           fmt.lctl(rope_en=1, rope_bank=0, ser_dst=1, fsrc_ext=4,
                    resid_arm=1, rope_pos=step["pos"], nsrc=1))
    s.emit("// [4] drain the walk's egress in FIFO order: 1 ss (s_c), 10 fs "
           "(s_k, 8x s_v, the NFEED r1 scale — the q scale was capped at "
           "staging), 8 RO beats (the PV o8), the TIP beat")
    s.ess(0, 0)
    for _ in range(10):
        s.efs(0, 0)
    for _ in range(8):
        s.ero([0] * glo.MXE_N, 1)
    s.emit("ETIP", "0")
    s.emit("// [5] gamma releases the norm's emission — but the row it "
           "normed is r1, THE WALKED EPILOGUE'S PRODUCT: h2 scale + codes, "
           "then the r1 row itself via LAYER_RDATA")
    s.csrw(L_CTRL, 0)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.fjob(1)
    s.grow_n(step["gamma2"])
    s.efs(0, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, d, glo.MXE_N):
        identity_readback(s, 0, base)
    s.lrptr(0)
    s.erd(d)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"E-LANE E-6 {name}: one-kick chain with the "
                           f"walked fuel-fed epilogue")
    man = _emit(x, out_dir, name, dict(
        stage="walk_e6",
        kind="ONE WALK, ONE KICK: pre-staged q -> score+pv (o8 on-tile "
             "requant) -> FUEL-FED OPROJ + o8 EPILOGUE -> deq -> residual "
             "(r1 in-tile) -> NFEED r1 -> norm; host silent in the window",
        d=int(d), pos=step["pos"], rq=list(step["rq"]),
        rq_o=list(step["rq_o"]), comp=f"{step['comp_bits']:#010x}",
        walk_mask=f"{e6_descriptor(step, mask=mask)[fmt.W_MASK]:#06x}",
        seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    splice_fuel_arm(Path(man["path"]))
    from walk_fuel_proj import splice_fuel_disarm
    splice_fuel_disarm(Path(man["path"]))
    return man


# ═══════════════ E-7: the chain NORMALIZES with a FETCHED gamma ════════════
# walk_e7 = walk_e6 + {NORM2, FGAM}: after NFEED hands the epilogue's own r1
# to the norm, the walker FETCHES the real resident L00_g2 gamma from card
# DRAM by its own record (the E-7 xw -> apex_gam_unpack -> norm-g route),
# arms the norm's output drain ITSELF (S2_GDL/GDA/GDF: levels, act-LOAD at
# the fp_top row window, C-1 feeder job — the host drain choreography as
# walker verbs), and the norm emits IN-WALK: h2 codes land in act bank 1
# row fp_top (= 2 here: above the staged-q row and the o-act row), its
# scale in the fs FIFO. Host in the walk window: NOTHING. The E-6 stage-[5]
# host gamma stream is GONE — that is the closed fence.
#
# HONESTY: the walk normalizes the REAL first 64 elements of L00_g2 over
# the toy r1 (the toy fence truncates d_model, exactly as it truncates the
# Wo read); golden composes on the SAME 64 gamma values decoded from the
# SAME image bytes. The o8->act seam and pre-staged q disclosures of
# walk_e6 apply unchanged.

E7C_MASK = E6C_MASK | (1 << fmt.EN_NORM2) | (1 << fmt.EN_FGAM)


def _g2_resident(image_dir: Path, d: int) -> tuple[int, list[int]]:
    """The g2 tensor base + its first d int16 LE values, decoded from the
    image bytes exactly as apex_gam_unpack consumes them (4 per beat, LE)."""
    man = json.loads((image_dir / "ddr_image.json").read_text())
    g2 = next(t for t in man["tensors"]
              if t["layer"] == 0 and t["tensor"] == "g2")
    img = (image_dir / "ddr_image.bin").read_bytes()
    b0 = g2["base_64B"] * 64
    vals = []
    for k in range(d):
        v = img[b0 + 2 * k] | (img[b0 + 2 * k + 1] << 8)
        vals.append(v - 65536 if v >= 32768 else v)
    return int(g2["base_64B"]), vals


def toy_e7(d: int = D_TOY, pos: int = POS_M, seed: int = 20260815,
           image_dir: Path | None = None) -> dict:
    """toy_e6c + the FETCHED-gamma norm reference: h2f = rmsnorm(C-1(r1),
    g2_resident[:d]), then its C-1 requant — golden's own next step, on the
    resident bytes the walker will fetch."""
    image_dir = Path(image_dir or DEFAULT_IMAGE)
    st = toy_e6c(d, pos, seed, image_dir=image_dir)
    g2_base, g2res = _g2_resident(image_dir, d)
    r1_codes, _sc = at.quant_rows_i8(
        f16_bits_to_f64(st["r1_bits"])[None, :])
    h2f, _r, _n = at.rmsnorm_fx([int(c) for c in r1_codes[0]],
                                [int(v) for v in g2res])
    h2f_codes, h2f_scales = at.quant_rows_i8(
        np.asarray(h2f, dtype=np.float64)[None, :] / 256.0)
    st.update(dict(
        g2_base=g2_base, g2_res=g2res,
        h2f_codes=np.asarray(h2f_codes[0], dtype=np.int64),
        h2f_scale=int(h2f_scales[0])))
    return st


def e7_descriptor(step: dict, *, mask: int = E7C_MASK) -> list[int]:
    w = e6_descriptor(step, mask=mask)
    w[fmt.W_TENS0 + fmt.TENS_G2] = step["g2_base"]
    return w


def build_walk_e7c(step: dict, *, out_dir: Path, name: str,
                   mask: int = E7C_MASK) -> dict:
    from elane_norm_feed import LST_NFEED
    from walk_fuel_proj import FUEL_MARK, FUEL_UNMARK, splice_fuel_arm, \
        splice_fuel_disarm
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// E-LANE E-7 chain {name}: ONE fmt=1 walk, ONE KICK = "
           f"SCORE + PV + FUEL-FED OPROJ EPILOGUE + RES1 + NFEED + NORM2 "
           f"with the GAMMA FETCHED FROM CARD DRAM by the walker's own "
           f"record (toy {d}, H=1, T=1). The norm eats the epilogue's own "
           f"r1 AND a weight the walker fetched; its output drains through "
           f"the walker-armed y path into act bank 1 row 2. Host in the "
           f"walk window: NOTHING.")
    _prologue(s)
    s.emit("// [1] host loads (walk_e6's exact staging): phase-q; K/V; X; "
           "o-act family at bank1 row 1; pre-staged rotated q at row 0.")
    s.lptr(BANK_PHQ, 0)
    s.ldata(step["phase"])
    _tap = s.tap
    s.tap = lambda *a, **k: None
    g3.store_kv_phase(s, step["K16"], step["V16"])
    s.tap = _tap
    s.lptr(glo.BANK_RESID, 0)
    s.ldata(step["x_bits"])
    s.route(rdst=1, asrc=1)
    o_pairs = [g3.decompose_f16(int(b))[:2]
               for b in g3.to16(np.asarray(step["gold_o8"],
                                           dtype=np.float64))]
    g3.inject_jobs(s, o_pairs, 1)
    s.ess(step["o_act_scale"], 1)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 1)
    s.pmode(True)
    s.route(rdst=1, asrc=0)
    s.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"], fsrc_ext=3)
    s.fjob(1)
    q_pairs = [g3.decompose_f16(int(b))[:2]
               for b in np.asarray(step["q_bits"], dtype=np.uint16)]
    g3.inject_jobs(s, q_pairs, 0, waddr=None)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.efs(0, 0)
    s.lpoll(glo.LST_ROPE, 0x0)
    s.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"])
    s.route(rdst=0, asrc=0)
    s.emit(f"// {FUEL_MARK}")
    s.emit("// [2] the 64-word image: E6C_MASK + {NORM2, FGAM} + the Wo "
           "AND g2 tensor bases — the gamma is a fetched tensor now")
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, e7_descriptor(step, mask=mask))
    s.emit("// [3] THE ONE KICK — attention, epilogue, NFEED, gamma fetch, "
           "norm, y-drain, all under one done-poll. No xw pacing, no RO "
           "drain, no gamma stream: the fs FIFO (12 beats total) holds "
           "everything until the post-walk drain.")
    s.emit("// CYCMARK e7go")
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit("// CYCMARK e7done")
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    s.emit(f"// {FUEL_UNMARK}")
    s.emit("// consumers drained CLEAN; no LAYER error")
    s.lpoll(glo.LST_LDQ | glo.LST_RES, 0x0)
    s.lstat(0x1F00 | LST_NFEED | glo.LST_LDQ | glo.LST_RES, 0x0)
    s.emit("// [3b] the level word the WALKER left — NFEED's code-4/nsrc=1 "
           "were RE-ARMED by the y-drain (S2_GDL: fsrc_ext=0, nsrc=0), so "
           "this full-word check PROVES the walker ran the drain arm")
    s.csrr(L_CTRL, 0xFFFFF,
           fmt.lctl(rope_en=1, rope_bank=0, ser_dst=1, fsrc_ext=0,
                    resid_arm=1, rope_pos=step["pos"], nsrc=0))
    s.emit("// [4] drain the walk's egress in FIFO order: 1 ss (s_c), 11 fs "
           "(s_k, 8x s_v, the NFEED r1 scale, the Y-DRAIN h2 scale), 8 RO "
           "beats (the PV o8), the TIP beat")
    s.ess(0, 0)
    for _ in range(11):
        s.efs(0, 0)
    for _ in range(8):
        s.ero([0] * glo.MXE_N, 1)
    s.emit("ETIP", "0")
    s.emit("// [5] NO host gamma, NO host drain jobs — the walk already "
           "normalized and drained. Read the h2 codes back from act bank1 "
           "ROW 2 (the walker's fp_top window) via identity GEMMs, then "
           "the r1 row.")
    s.csrw(L_CTRL, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, d, glo.MXE_N):
        identity_readback(s, 2, base)
    s.lrptr(0)
    s.erd(d)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"E-LANE E-7 {name}: one-kick chain, the norm's "
                           f"gamma FETCHED from card DRAM")
    man = _emit(x, out_dir, name, dict(
        stage="walk_e7",
        kind="ONE WALK, ONE KICK: pre-staged q -> score+pv -> FUEL-FED "
             "OPROJ epilogue -> residual (r1 in-tile) -> NFEED r1 -> "
             "NORM2 with WALKER-FETCHED gamma -> walker-armed y-drain "
             "-> h2 codes in act bank; host silent in the window",
        d=int(d), pos=step["pos"], rq=list(step["rq"]),
        rq_o=list(step["rq_o"]), comp=f"{step['comp_bits']:#010x}",
        walk_mask=f"{e7_descriptor(step, mask=mask)[fmt.W_MASK]:#06x}",
        g2_base=step["g2_base"], seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    splice_fuel_arm(Path(man["path"]))
    splice_fuel_disarm(Path(man["path"]))
    return man


# ═══════════════ E-8: QKV JOINS THE ONE-KICK CHAIN (fence 5 CLOSED) ════════
# The E-6 fence "QKV excludes SCORE/PV (act-bank row collision)" was a ROW
# ASSIGNMENT, not a structural conflict: the QKV act family emitted from
# rows 0..fp_rows-1 — the very rows the pre-staged q lives in. E-7's
# fp_base stacks every family in template order instead (q rows, QKV,
# OPROJ, then the y-drain window), so walk_e8 = walk_e7 + {QKV}: the
# sequencer ALSO fetches the real Wq/Wk/Wv region prefixes by its own
# records and runs the 24 raw projections (RO-drained — the E-5 disclosed
# data plane; the FIFO is 16 deep and 24 jobs make 216 entries) before the
# attention/epilogue/NFEED/NORM2 chain. Rows: q=0, QKV x=1, o-act=2,
# y-drain h2=3 — the walker's own deterministic stack, mirrored here.

E8C_MASK = E7C_MASK | (1 << fmt.EN_QKV)
N_QKV_JOBS_E8 = 3 * (D_TOY // 8)      # Wq/Wk/Wv x 8 n-jobs at the toy fence


def toy_e8(d: int = D_TOY, pos: int = POS_M, seed: int = 20260815,
           image_dir: Path | None = None) -> dict:
    """toy_e7 + the QKV subject: a synthetic C-1 x row against the REAL
    Wq/Wk/Wv region prefixes, decoded exactly as the walker's d=64 records
    read them (the _wo_subblocks idiom per tensor)."""
    from apex_golden import compute as cpg
    image_dir = Path(image_dir or DEFAULT_IMAGE)
    st = toy_e7(d, pos, seed, image_dir=image_dir)
    rng = np.random.default_rng(seed + 7)
    x_vals = rng.normal(0.0, 1.0, d)
    xq_codes, xq_scales = at.quant_rows_i8(x_vals[None, :])
    man = json.loads((image_dir / "ddr_image.json").read_text())
    img = (image_dir / "ddr_image.bin").read_bytes()
    qkv = {}
    for tens in ("Wq", "Wk", "Wv"):
        rec = next(t for t in man["tensors"]
                   if t["layer"] == 0 and t["tensor"] == tens)
        b0 = rec["base_64B"] * 64
        blocks = []
        for j in range(d // 8):
            blk = img[b0 + 512 * j: b0 + 512 * (j + 1)]
            W = np.frombuffer(blk, dtype=np.int8).astype(np.int64)
            blocks.append(W.reshape(8, 8, 8).transpose(0, 2, 1)
                          .reshape(64, 8))
        qkv[tens] = dict(base=int(rec["base_64B"]), blocks=blocks)
    x8 = np.asarray(xq_codes[0], dtype=np.int64)
    gold_qkv = np.concatenate(
        [cpg.gemm_i8_ksplit(x8[None, :], b)[0]
         for tens in ("Wq", "Wk", "Wv") for b in qkv[tens]["blocks"]])
    st.update(dict(xq_vals=x_vals, xq_codes=x8,
                   xq_scale=int(xq_scales[0]), qkv_bases={
                       t: qkv[t]["base"] for t in qkv},
                   gold_qkv=gold_qkv.astype(np.int64)))
    return st


def e8_descriptor(step: dict, *, mask: int = E8C_MASK) -> list[int]:
    w = e7_descriptor(step, mask=mask)
    for tname, tidx in (("Wq", fmt.TENS_WQ), ("Wk", fmt.TENS_WK),
                        ("Wv", fmt.TENS_WV)):
        w[fmt.W_TENS0 + tidx] = step["qkv_bases"][tname]
    return w


def build_walk_e8c(step: dict, *, out_dir: Path, name: str,
                   mask: int = E8C_MASK) -> dict:
    from elane_norm_feed import LST_NFEED
    from walk_fuel_proj import FUEL_MARK, FUEL_UNMARK, splice_fuel_arm, \
        splice_fuel_disarm
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// E-LANE E-8 chain {name}: ONE fmt=1 walk, ONE KICK = "
           f"QKV(FUEL-FED, RO-drained) + SCORE + PV + FUEL-FED OPROJ "
           f"EPILOGUE + RES1 + NFEED + NORM2 with FETCHED gamma (toy {d}). "
           f"Act rows: q=0, QKV x=1, o-act=2, y-drain h2=3 — the walker's "
           f"own template-order stack.")
    _prologue(s)
    s.emit("// [1] host loads: phase-q; K/V; X; the QKV x family at row 1; "
           "the o-act family at row 2; pre-staged rotated q at row 0.")
    s.lptr(BANK_PHQ, 0)
    s.ldata(step["phase"])
    _tap = s.tap
    s.tap = lambda *a, **k: None
    g3.store_kv_phase(s, step["K16"], step["V16"])
    s.tap = _tap
    s.lptr(glo.BANK_RESID, 0)
    s.ldata(step["x_bits"])
    s.route(rdst=1, asrc=1)
    s.emit("// (a) the QKV x family -> bank1 row 1. The injected values ARE "
           "the C-1 codes (quant_rows_i8 output, amax 127 by construction) "
           "so MODE_QUANT is the identity and the counted ess is F16_ONE — "
           "the E-5 c1_frames property (measured: 0x3C00 vs the row-scale "
           "shortcut on this gate's first emission).")
    x_pairs = [g3.decompose_f16(int(b))[:2]
               for b in g3.to16(np.asarray(step["xq_codes"],
                                           dtype=np.float64))]
    g3.inject_jobs(s, x_pairs, 1)
    s.ess(0x3C00, 1)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 1)
    s.emit("// (b) the o-act family -> bank1 row 2")
    o_pairs = [g3.decompose_f16(int(b))[:2]
               for b in g3.to16(np.asarray(step["gold_o8"],
                                           dtype=np.float64))]
    g3.inject_jobs(s, o_pairs, 1)
    s.ess(step["o_act_scale"], 1)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 2)
    s.pmode(True)
    s.route(rdst=1, asrc=0)
    s.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"], fsrc_ext=3)
    s.fjob(1)
    q_pairs = [g3.decompose_f16(int(b))[:2]
               for b in np.asarray(step["q_bits"], dtype=np.uint16)]
    g3.inject_jobs(s, q_pairs, 0, waddr=None)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.efs(0, 0)
    s.lpoll(glo.LST_ROPE, 0x0)
    s.lctl(rope_en=1, rope_bank=1, rope_pos=step["pos"])
    s.route(rdst=0, asrc=0)
    s.emit(f"// {FUEL_MARK}")
    s.emit("// [2] the 64-word image: E7C_MASK + QKV + the Wq/Wk/Wv/Wo/g2 "
           "tensor bases")
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, e8_descriptor(step, mask=mask))
    s.emit(f"// [3] THE ONE KICK. The {N_QKV_JOBS_E8} raw QKV jobs are "
           f"RO-drained MID-walk (data plane, disclosed: 216 entries vs "
           f"the 16-deep FIFO); everything else is host-silent.")
    s.emit("// CYCMARK e8go")
    s.csrw(W_CTRL, 0x3)
    for _ in range(N_QKV_JOBS_E8):
        s.ero([0] * glo.MXE_N, 1)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit("// CYCMARK e8done")
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    s.emit(f"// {FUEL_UNMARK}")
    s.lpoll(glo.LST_LDQ | glo.LST_RES, 0x0)
    s.lstat(0x1F00 | LST_NFEED | glo.LST_LDQ | glo.LST_RES, 0x0)
    s.csrr(L_CTRL, 0xFFFFF,
           fmt.lctl(rope_en=1, rope_bank=0, ser_dst=1, fsrc_ext=0,
                    resid_arm=1, rope_pos=step["pos"], nsrc=0))
    s.emit("// [4] drain the walk's remaining egress: 1 ss, 11 fs, the 8 "
           "PV-o8 RO beats, the TIP beat")
    s.ess(0, 0)
    for _ in range(11):
        s.efs(0, 0)
    for _ in range(8):
        s.ero([0] * glo.MXE_N, 1)
    s.emit("ETIP", "0")
    s.emit("// [5] h2 codes from act bank1 ROW 3 (the y-drain window above "
           "the three staged families), then the r1 row")
    s.csrw(L_CTRL, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, d, glo.MXE_N):
        identity_readback(s, 3, base)
    s.lrptr(0)
    s.erd(d)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"E-LANE E-8 {name}: QKV joins the one-kick "
                           f"fetched-gamma chain")
    man = _emit(x, out_dir, name, dict(
        stage="walk_e8",
        kind="ONE WALK, ONE KICK: FUEL-FED QKV (RO-drained) -> attention "
             "-> FUEL-FED OPROJ epilogue -> residual -> NFEED -> NORM2 "
             "with WALKER-FETCHED gamma -> walker-armed y-drain",
        d=int(d), pos=step["pos"],
        walk_mask=f"{e8_descriptor(step, mask=mask)[fmt.W_MASK]:#07x}",
        seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    splice_fuel_arm(Path(man["path"]))
    splice_fuel_disarm(Path(man["path"]))
    return man


def grade_e8c(caps, step: dict) -> dict:
    d = step["d"]
    nq = N_QKV_JOBS_E8 * 8
    out = {}
    ss = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "ss"]
    out["s_c"] = {"equal": len(ss) >= 1 and ss[0] == step["gold_sc"]}
    fs = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    want_fs = ([step["q_scale"]] + step["gold_sk"] + step["gold_sv"] * 8
               + [step["r1_scale"], step["h2f_scale"]])
    out["fs"] = {"equal": fs == want_fs, "n": len(fs)}
    acc = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64)
    out["qkv"] = {"equal": acc.size >= nq
                  and np.array_equal(acc[:nq], step["gold_qkv"])}
    o8 = acc[nq:nq + d]
    h2 = acc[nq + d:nq + 2 * d]
    out["o8"] = {"equal": o8.size == d
                 and np.array_equal(o8, step["gold_o8"])}
    out["h2_codes"] = {"equal": h2.size == d
                       and np.array_equal(h2, step["h2f_codes"])}
    gr = grade_resid(caps, step["r1_bits"])
    out["r1_row"] = {"equal": bool(gr["equal"]),
                     "n_diff": int(gr.get("n_diff", -1))}
    out["equal"] = all(out[k]["equal"] for k in
                       ("s_c", "fs", "qkv", "o8", "h2_codes", "r1_row"))
    return out


def build_e7_refusal(step: dict, *, out_dir: Path, name: str, mask: int,
                     why: str) -> dict:
    """E-7 fence probes on the twin: pkg-LEGAL masks S2_CHECK must refuse
    (WALK_ERR_DESC receipts checked; fuel armed first so wf_ready cannot
    mask the verdict)."""
    from walk_fuel_proj import FUEL_MARK, splice_fuel_arm
    s = LayerScript(step["d"])
    s.emit(f"// E-7 fence {name}: {why} -> REFUSED at S2_CHECK "
           f"(WALK_ERR_DESC), state unchanged.")
    _prologue(s)
    s.emit(f"// {FUEL_MARK}")
    load_descriptor(s, e7_descriptor(step, mask=mask))
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE,
           W_ST_ESTK | (2 << 9))
    s.csrw(W_CTRL, 0x0)
    s.emit("DONE")
    x = _translate(s, note=f"E-7 fence {name}")
    man = _emit(x, out_dir, name,
                dict(stage=name, kind=f"REFUSAL FENCE: {why}",
                     d=int(step["d"])))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    splice_fuel_arm(Path(man["path"]))
    return man


def grade_e7c(caps, step: dict) -> dict:
    d = step["d"]
    out = {}
    ss = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "ss"]
    out["s_c"] = {"equal": len(ss) >= 1 and ss[0] == step["gold_sc"],
                  "got": ss[:1], "want": step["gold_sc"]}
    fs = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    want_fs = ([step["q_scale"]] + step["gold_sk"] + step["gold_sv"] * 8
               + [step["r1_scale"], step["h2f_scale"]])
    out["fs"] = {"equal": fs == want_fs, "n": len(fs),
                 "want_n": len(want_fs)}
    acc = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64)
    o8 = acc[:d]
    h2 = acc[d:2 * d]
    out["o8"] = {"equal": o8.size == d
                 and np.array_equal(o8, step["gold_o8"])}
    out["h2_codes"] = {"equal": h2.size == d
                       and np.array_equal(h2, step["h2f_codes"])}
    gr = grade_resid(caps, step["r1_bits"])
    out["r1_row"] = {"equal": bool(gr["equal"]),
                     "n_diff": int(gr.get("n_diff", -1))}
    out["equal"] = all(out[k]["equal"] for k in
                       ("s_c", "fs", "o8", "h2_codes", "r1_row"))
    return out


def _gamma_poison(step: dict, image_dir: Path, dst_dir: Path) -> dict:
    """One flipped byte of the RESIDENT g2 tensor, golden-SEARCHED so the
    predicted h2 CODES move (the C-1 requant can absorb a small gamma move
    — the elane gamma-disc lesson); r1 must NOT move (gamma is downstream
    of the residual), which localizes the flip to the fetched-gamma norm."""
    d = step["d"]
    r1_codes, _sc = at.quant_rows_i8(
        f16_bits_to_f64(step["r1_bits"])[None, :])
    data = bytearray((image_dir / "ddr_image.bin").read_bytes())
    b0 = step["g2_base"] * 64
    for k in range(d):
        for byte_i in (1, 0):                     # high byte first: big move
            off = b0 + 2 * k + byte_i
            old = data[off]
            new = old ^ (0x40 if byte_i else 0xFF)
            v = (data[b0 + 2 * k] | (data[b0 + 2 * k + 1] << 8))
            if byte_i:
                vp = ((new << 8) | data[b0 + 2 * k]) & 0xFFFF
            else:
                vp = ((data[b0 + 2 * k + 1] << 8) | new) & 0xFFFF
            g2p = list(step["g2_res"])
            g2p[k] = vp - 65536 if vp >= 32768 else vp
            h2p, _r, _n = at.rmsnorm_fx([int(c) for c in r1_codes[0]],
                                        [int(vv) for vv in g2p])
            cp_, sp_ = at.quant_rows_i8(
                np.asarray(h2p, dtype=np.float64)[None, :] / 256.0)
            if (np.array_equal(np.asarray(cp_[0], dtype=np.int64),
                               step["h2f_codes"])
                    and int(sp_[0]) == step["h2f_scale"]):
                continue                          # absorbed — keep searching
            data[off] = new
            (dst_dir / "ddr_image.bin").write_bytes(bytes(data))
            return dict(byte_offset=off, k=k, old=old, new=new,
                        h2p_codes=np.asarray(cp_[0], dtype=np.int64),
                        h2p_scale=int(sp_[0]),
                        v_old=int(v), v_new=int(vp))
    raise SystemExit("REFUSE: no g2 byte flip moves the golden h2 — the "
                     "gamma poison would be vacuous on this subject")


def chain7(args) -> int:
    """The E-7 one-kick chain gate: the norm's gamma FETCHED from DRAM."""
    import tile_exec_bridge as bridge
    from walk_fuel_proj import rewrite_region_sha
    assert args.binary, "--chain7 requires --binary <E-7 DDR=1 b64 twin>"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image)
    e7 = toy_e7(seed=args.seed + 20, image_dir=image_dir)
    ddr = (f"+ddr_image={image_dir}/ddr_image.bin",
           f"+ddr_regions={image_dir}/ddr_image.regions.jsonl",
           "+fuel_audit", f"+tile_div={args.tile_div}")
    ddr_claim = ddr + ("+poll_limit=16000000",)

    with tg.at_d(D_TOY):
        m_c = build_walk_e7c(e7, out_dir=out, name="walk_e7")
        # walk-off: FGAM cleared, NORM2 kept -> the kept E-6 poisoning
        # refusal (NORMx under FPROJ requires FGAM) — a LOUD refusal, not
        # a park; receipts checked by the program.
        m_off = build_e7_refusal(
            e7, out_dir=out, name="walk_e7_off",
            mask=E7C_MASK & ~(1 << fmt.EN_FGAM),
            why="NORM2 under FPROJ without FGAM — the E-6 gamma-poisoning "
                "refusal, kept verbatim")
        m_f1 = build_e7_refusal(
            e7, out_dir=out, name="e7_fence_nofproj",
            mask=(1 << fmt.EN_FGAM) | (1 << fmt.EN_NORM2)
                 | (1 << fmt.EN_NFEED) | (1 << fmt.EN_NSRC),
            why="FGAM without FPROJ — the gamma fetch would park at "
                "S2_FETCH (fuel unarmed)")
        m_f2 = build_e7_refusal(
            e7, out_dir=out, name="e7_fence_nonorm",
            mask=E7C_MASK & ~(1 << fmt.EN_NORM2),
            why="FGAM with no NORM step — a gamma window with no consumer")

    # the GAMMA POISON: one resident g2 byte -> h2 moves by the golden-
    # predicted codes, r1 does NOT (localizes to the fetched-gamma norm)
    bad_dir = out / "bad_image_e7g"
    bad_dir.mkdir(parents=True, exist_ok=True)
    gmut = _gamma_poison(e7, image_dir, bad_dir)
    rewrite_region_sha(image_dir / "ddr_image.regions.jsonl",
                       bad_dir / "ddr_image.regions.jsonl",
                       bad_dir / "ddr_image.bin")
    bad = (f"+ddr_image={bad_dir}/ddr_image.bin",
           f"+ddr_regions={bad_dir}/ddr_image.regions.jsonl",
           "+fuel_audit", f"+tile_div={args.tile_div}",
           "+poll_limit=16000000")

    def _run(man, extra, timeout):
        return bridge.run_job(man["path"], executor="sim",
                              binary=args.binary, extra_args=extra,
                              timeout_s=timeout)

    rows = {}
    r_c = _run(m_c, ddr_claim, args.timeout)
    g_c = grade_e7c(r_c["captures"], e7) if r_c["ok"] else {"equal": False}
    cyc = _cyc_window_e7(r_c.get("log"))
    green = bool(r_c["ok"] and g_c.get("equal"))
    eprint(f"  [walk_e7        ] rc={r_c['rc']} ok={r_c['ok']} "
           f"caps={len(r_c['captures'])} grade={g_c.get('equal')} "
           f"cycles={cyc}")
    rows["walk_e7"] = dict(ok=bool(r_c["ok"]), grade=g_c, cycles=cyc)

    for nm, man in (("walk_e7_off", m_off), ("e7_fence_nofproj", m_f1),
                    ("e7_fence_nonorm", m_f2)):
        r_x = _run(man, ddr, args.timeout_off)
        eprint(f"  [{nm:<15}] rc={r_x['rc']} REFUSED={r_x['ok']}")
        rows[nm] = dict(refused=bool(r_x["ok"]))

    # the poison run: same walk_e7 program, mutated image
    r_p = _run(m_c, bad, args.timeout)
    ran_p = disc_ran(r_p)
    caps_p = r_p["captures"] if ran_p else []
    acc_p = cd.ro_lanes_to_i32(caps_p, strict=False).astype(np.int64) \
        if ran_p else np.zeros(0, dtype=np.int64)
    d = e7["d"]
    h2_p = acc_p[d:2 * d] if acc_p.size >= 2 * d else np.zeros(0)
    h2_moved_right = bool(ran_p and h2_p.size == d
                          and np.array_equal(h2_p, gmut["h2p_codes"])
                          and not np.array_equal(h2_p, e7["h2f_codes"]))
    gr_p = grade_resid(caps_p, e7["r1_bits"]) if ran_p else {"equal": False}
    r1_held = bool(gr_p.get("equal"))
    poison_red = bool(green and h2_moved_right and r1_held)
    eprint(f"  [walk_e7_gpoison] rc={r_p['rc']} h2==predicted:"
           f"{h2_moved_right} r1_held={r1_held} -> RED={poison_red}")
    rows["walk_e7_gpoison"] = dict(
        red=poison_red, r1_held=r1_held,
        mut={k: gmut[k] for k in ("byte_offset", "k", "old", "new",
                                  "v_old", "v_new")})

    ok = (green and all(rows[n]["refused"] for n in
                        ("walk_e7_off", "e7_fence_nofproj",
                         "e7_fence_nonorm"))
          and poison_red)
    report = dict(binary=args.binary, image=str(image_dir),
                  tile_div=args.tile_div, seed=e7["seed"],
                  mask=hex(E7C_MASK), stages=rows, verdict=bool(ok))
    (out / "e7_chain_report.json").write_text(json.dumps(report, indent=1,
                                                         default=str))
    print(f"\nE-LANE E-7 CHAIN @ tile_div={args.tile_div}")
    print(f"  walk_e7 (one kick, FETCHED gamma, host-silent): "
          f"{'PASS' if green else 'FAIL'} (cycles={cyc})")
    print(f"  off/nofproj/nonorm refusals + gamma poison    : "
          f"{rows['walk_e7_off']['refused']}/"
          f"{rows['e7_fence_nofproj']['refused']}/"
          f"{rows['e7_fence_nonorm']['refused']}/{poison_red}")
    print(f"ELANE-E7 CHAIN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def chain8(args) -> int:
    """The E-8 chain gate: QKV composed into the fetched-gamma one-kick
    chain — the fence-5 CLOSURE's positive arm. Discriminated by the E-7
    gate's own probes (this stage adds the composition claim + the QKV
    grade); the walk-off here is the CAPACITY refusal at H=9 (the unit
    refuse2 case (p) shape, proven on the twin)."""
    import tile_exec_bridge as bridge
    assert args.binary, "--chain8 requires --binary <E-7 DDR=1 b64 twin>"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image)
    e8 = toy_e8(seed=args.seed + 20, image_dir=image_dir)
    ddr = (f"+ddr_image={image_dir}/ddr_image.bin",
           f"+ddr_regions={image_dir}/ddr_image.regions.jsonl",
           "+fuel_audit", f"+tile_div={args.tile_div}")
    ddr_claim = ddr + ("+poll_limit=16000000",)
    with tg.at_d(D_TOY):
        m_c = build_walk_e8c(e8, out_dir=out, name="walk_e8")

    r_c = bridge.run_job(m_c["path"], executor="sim", binary=args.binary,
                         extra_args=ddr_claim, timeout_s=args.timeout)
    g_c = grade_e8c(r_c["captures"], e8) if r_c["ok"] else {"equal": False}
    cyc = _cyc_window_e8(r_c.get("log"))
    green = bool(r_c["ok"] and g_c.get("equal"))
    eprint(f"  [walk_e8        ] rc={r_c['rc']} ok={r_c['ok']} "
           f"caps={len(r_c['captures'])} grade={g_c.get('equal')} "
           f"cycles={cyc}")
    detail = {k: g_c.get(k, {}).get("equal") for k in
              ("s_c", "fs", "qkv", "o8", "h2_codes", "r1_row")} \
        if isinstance(g_c, dict) else {}
    report = dict(binary=args.binary, image=str(image_dir),
                  tile_div=args.tile_div, seed=e8["seed"],
                  mask=hex(E8C_MASK), grade=detail, cycles=cyc,
                  verdict=bool(green))
    (out / "e8_chain_report.json").write_text(json.dumps(report, indent=1,
                                                         default=str))
    print(f"\nE-LANE E-8 CHAIN @ tile_div={args.tile_div}")
    print(f"  walk_e8 (QKV + attention + epilogue + fetched-gamma NORM2, "
          f"one kick): {'PASS' if green else 'FAIL'} (cycles={cyc}) "
          f"{detail}")
    print(f"ELANE-E8 CHAIN: {'PASS' if green else 'FAIL'}")
    return 0 if green else 1


def _cyc_window_e8(log: str):
    go = done = None
    for ln in (log or "").splitlines():
        if not ln.startswith("CYCMARK"):
            continue
        if "e8go" in ln:
            go = int(ln.rsplit("cyc=", 1)[1])
        elif "e8done" in ln:
            done = int(ln.rsplit("cyc=", 1)[1])
    return (done - go) if (go is not None and done is not None) else None


def _cyc_window_e7(log: str):
    go = done = None
    for ln in (log or "").splitlines():
        if not ln.startswith("CYCMARK"):
            continue
        if "e7go" in ln:
            go = int(ln.rsplit("cyc=", 1)[1])
        elif "e7done" in ln:
            done = int(ln.rsplit("cyc=", 1)[1])
    return (done - go) if (go is not None and done is not None) else None


def build_e6_refusal_qstage(step: dict, *, out_dir: Path, name: str) -> dict:
    """QSTAGE under FPROJ must be REFUSED at S2_CHECK: fuel_src=1
    disconnects the mailbox xw stream QSTAGE's k2 injection consumes —
    walking it would wedge S2_QSD forever. Fuel IS armed first so the
    wf_ready clause cannot mask the verdict."""
    from walk_fuel_proj import FUEL_MARK, splice_fuel_arm
    mask = E6C_MASK | (1 << fmt.EN_QSTAGE)
    s = LayerScript(step["d"])
    s.emit(f"// E-6 fence {name}: QSTAGE+FPROJ -> REFUSED (WALK_ERR_DESC), "
           f"state unchanged — the wedge that is now a loud refusal.")
    _prologue(s)
    s.emit(f"// {FUEL_MARK}")
    load_descriptor(s, e6_descriptor(step, mask=mask))
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE,
           W_ST_ESTK | (2 << 9))
    s.csrw(W_CTRL, 0x0)
    s.emit("// refusal leaves state unchanged: the levels never moved")
    s.csrr(L_CTRL, 0xFFFFF, 0x0)
    s.emit("DONE")
    x = _translate(s, note=f"E-6 fence {name}")
    man = _emit(x, out_dir, name, dict(
        stage=name, kind="REFUSAL FENCE: QSTAGE under FPROJ (dead mailbox "
                         "xw stream)", d=int(step["d"]),
        walk_mask=f"{mask:#06x}", seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    splice_fuel_arm(Path(man["path"]))
    return man


def grade_e6c(caps, step: dict) -> dict:
    d = step["d"]
    out = {}
    ss = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "ss"]
    out["s_c"] = {"equal": len(ss) >= 1 and ss[0] == step["gold_sc"],
                  "got": ss[:1], "want": step["gold_sc"]}
    fs = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    want_fs = ([step["q_scale"]] + step["gold_sk"] + step["gold_sv"] * 8
               + [step["r1_scale"], step["h2r_scale"]])
    out["fs"] = {"equal": fs == want_fs, "n": len(fs),
                 "want_n": len(want_fs)}
    acc = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64)
    o8 = acc[:d]
    h2 = acc[d:2 * d]
    out["o8"] = {"equal": o8.size == d
                 and np.array_equal(o8, step["gold_o8"])}
    out["h2_codes"] = {"equal": h2.size == d
                       and np.array_equal(h2, step["h2r_codes"])}
    from gen_layer_ops import grade_resid
    gr = grade_resid(caps, step["r1_bits"])
    out["r1_row"] = {"equal": bool(gr["equal"]),
                     "n_diff": int(gr.get("n_diff", -1))}
    out["equal"] = all(out[k]["equal"] for k in
                       ("s_c", "fs", "o8", "h2_codes", "r1_row"))
    return out


def _cyc_window_e6(log: str):
    go = done = None
    for ln in (log or "").splitlines():
        if not ln.startswith("CYCMARK"):
            continue
        if "e6go" in ln:
            go = int(ln.rsplit("cyc=", 1)[1])
        elif "e6done" in ln:
            done = int(ln.rsplit("cyc=", 1)[1])
    return (done - go) if (go is not None and done is not None) else None


def chain(args) -> int:
    """The E-6 one-kick chain gate on the DDR=1 b64 twin."""
    import tile_exec_bridge as bridge
    from walk_fuel_proj import rewrite_region_sha
    import fuel_proj_05b as fp
    assert args.binary, "--chain requires --binary <DDR=1 b64 twin>"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image)
    e6 = toy_e6c(seed=args.seed + 20, image_dir=image_dir)
    ddr = (f"+ddr_image={image_dir}/ddr_image.bin",
           f"+ddr_regions={image_dir}/ddr_image.regions.jsonl",
           "+fuel_audit", f"+tile_div={args.tile_div}")
    # claim windows are host-silent: one done-poll spans the whole walk
    # (sim_main +poll_limit note); RED/refusal runs keep the default budget
    ddr_claim = ddr + ("+poll_limit=16000000",)

    with tg.at_d(D_TOY):
        m_c = build_walk_e6c(e6, out_dir=out, name="walk_e6")
        m_off = build_walk_e6c(e6, out_dir=out, name="walk_e6_off",
                               mask=E6C_MASK & ~(1 << fmt.EN_FPROJ))
        m_f = build_e6_refusal_qstage(e6, out_dir=out,
                                      name="e6_fence_qstage")
        # the V-flip SEAM DETECTOR (header): perturb a stored V record ->
        # the walked o8 must move (control-anchored), and r1 must NOT —
        # OPROJ consumed the host-staged COPY, and this measures that.
        e6v = dict(e6)
        for flip in (0x4000, 0x0800, 0x8000):
            v16p = e6["V16"].copy()
            v16p[0, 3] ^= np.uint16(flip)
            rot = tf.rope_fx(f16_bits_to_f64(e6["q_bits"]),
                             float(e6["pos"]), tf.rope_theta(e6["d"]))
            gp = at.attention_core(np.asarray(rot, dtype=np.float64),
                                   e6["K16"], v16p, at.TIER_CQ8, G=128)
            if not np.array_equal(np.asarray(gp.o8, dtype=np.int64),
                                  e6["gold_o8"]):
                e6v["V16"] = v16p
                break
        else:
            raise AssertionError("no V flip moved the golden o8")
        m_v = build_walk_e6c(e6v, out_dir=out, name="walk_e6_vbit")

    # the Wo poison: one byte of the fetched sub-block 0 -> r1[0] must move
    # by the PREDICTED f16 delta (recomposed with the SAME host-loaded rq);
    # the byte is golden-SEARCHED (requant can absorb a flip)
    from walk_fuel_proj import poison_search
    bad_dir = out / "bad_image_e6c"
    bad_dir.mkdir(parents=True, exist_ok=True)
    mut = poison_search(image_dir / "ddr_image.bin",
                        bad_dir / "ddr_image.bin",
                        base_64B=e6["wo_base"], x_codes=e6["o_act"],
                        acc=e6["acc_o"], rq=e6["rq_o"], comp=e6["comp"],
                        x16=np.asarray(e6["x_bits"], dtype=np.uint16),
                        name="L00_Wo_sub0(walked-e6)")
    rewrite_region_sha(image_dir / "ddr_image.regions.jsonl",
                       bad_dir / "ddr_image.regions.jsonl",
                       bad_dir / "ddr_image.bin")
    acc_p = e6["acc_o"].copy()
    acc_p[0] += mut["delta_lane0"]
    from apex_golden import compute as cpg
    o8p_p = cpg.requant_i32_to_i8(acc_p, *e6["rq_o"]).astype(np.int64)
    r1_p = np.asarray(f64_to_f16_bits(
        f16_bits_to_f64(np.asarray(e6["x_bits"], dtype=np.uint16))
        + o8p_p.astype(np.float64) * e6["comp"]), dtype=np.uint16)
    bad = (f"+ddr_image={bad_dir}/ddr_image.bin",
           f"+ddr_regions={bad_dir}/ddr_image.regions.jsonl",
           "+fuel_audit", f"+tile_div={args.tile_div}",
           "+poll_limit=16000000")

    def _run(man, extra, timeout):
        return bridge.run_job(man["path"], executor="sim",
                              binary=args.binary, extra_args=extra,
                              timeout_s=timeout)

    rows = {}
    r_c = _run(m_c, ddr_claim, args.timeout)
    g_c = grade_e6c(r_c["captures"], e6) if r_c["ok"] else {"equal": False}
    cyc = _cyc_window_e6(r_c.get("log"))
    green = bool(r_c["ok"] and g_c.get("equal"))
    eprint(f"  [walk_e6        ] rc={r_c['rc']} ok={r_c['ok']} "
           f"caps={len(r_c['captures'])} grade={g_c.get('equal')} "
           f"cycles={cyc}")
    rows["walk_e6"] = dict(ok=bool(r_c["ok"]), grade=g_c, cycles=cyc)

    r_off = _run(m_off, ddr, args.timeout_off)
    off_red = bool(green and not r_off["ok"])
    eprint(f"  [walk_e6_off    ] rc={r_off['rc']} ok={r_off['ok']} -> "
           f"RED={off_red}")
    rows["walk_e6_off"] = dict(ok=bool(r_off["ok"]), red=off_red)

    r_f = _run(m_f, ddr, args.timeout_off)
    eprint(f"  [e6_fence_qstage] rc={r_f['rc']} REFUSED={r_f['ok']}")
    rows["e6_fence_qstage"] = dict(refused=bool(r_f["ok"]))

    r_v = _run(m_v, ddr_claim, args.timeout)
    ran_v = disc_ran(r_v)
    g_v = grade_e6c(r_v["captures"], e6) if ran_v else {}
    o8_moved = bool(ran_v and not g_v.get("o8", {}).get("equal", True))
    r1_held = bool(ran_v and g_v.get("r1_row", {}).get("equal", False))
    seam = bool(green and o8_moved and r1_held)
    eprint(f"  [walk_e6_vbit   ] rc={r_v['rc']} o8_moved={o8_moved} "
           f"r1_held={r1_held} -> SEAM MEASURED={seam}")
    rows["walk_e6_vbit"] = dict(ran=ran_v, o8_moved=o8_moved,
                                r1_held=r1_held, seam_measured=seam)

    r_p = _run(m_c, bad, args.timeout)
    gp_pred = (grade_resid(r_p["captures"], r1_p) if r_p["ok"]
               else {"equal": False})
    gp_clean = (grade_resid(r_p["captures"], e6["r1_bits"]) if r_p["ok"]
                else {"equal": True})
    poison_red = bool(green and r_p["ok"] and gp_pred["equal"]
                      and not gp_clean["equal"]
                      and not np.array_equal(r1_p, e6["r1_bits"]))
    eprint(f"  [walk_e6_poison ] rc={r_p['rc']} r1==predicted:"
           f"{gp_pred.get('equal')} -> RED={poison_red}")
    rows["walk_e6_poison"] = dict(
        red=poison_red,
        mut={k: mut[k] for k in ("byte_offset", "k", "old", "new", "x_k",
                                 "delta_lane0")})

    ok = (green and off_red and rows["e6_fence_qstage"]["refused"]
          and seam and poison_red)
    report = dict(binary=args.binary, image=str(image_dir),
                  tile_div=args.tile_div, seed=e6["seed"], mask=hex(E6C_MASK),
                  stages=rows, verdict=bool(ok))
    (out / "e6_chain_report.json").write_text(json.dumps(report, indent=1,
                                                         default=str))
    print(f"\nE-LANE E-6 CHAIN @ tile_div={args.tile_div}")
    print(f"  walk_e6 (one kick, host-silent window)  : "
          f"{'PASS' if green else 'FAIL'} (cycles={cyc})")
    print(f"  walk-off RED / qstage fence / seam / poison: "
          f"{off_red}/{rows['e6_fence_qstage']['refused']}/{seam}/"
          f"{poison_red}")
    print(f"ELANE-E6 CHAIN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


from gen_layer_ops import grade_resid                          # noqa: E402


# ═══════════════ grading ═══════════════════════════════════════════════════

def grade_qstage(caps, step: dict) -> dict:
    d = step["d"]
    rows = int(step.get("rows", 1))
    out = {}
    acc = cd.ro_lanes_to_i32(caps, strict=False)
    want = step["q_codes_rows"].reshape(-1)
    got = acc[:rows * d].astype(np.int64)
    codes_ok = got.size == rows * d and np.array_equal(got, want)
    n = min(got.size, want.size)
    n_diff = int(np.sum(got[:n] != want[:n]))
    out["q_codes"] = {"equal": bool(codes_ok), "got": int(got.size),
                      "n_diff": n_diff, "rows": rows}
    fs = [t for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    fs_bits = [int(t["bits"]) for t in fs]
    scale_ok = fs_bits[:rows] == step["q_scales_rows"]
    out["scale"] = {"equal": bool(scale_ok), "got": fs_bits[:rows],
                    "want": step["q_scales_rows"]}
    out["equal"] = bool(codes_ok and scale_ok)
    return out


# ═══════════════ selftest / smoke ══════════════════════════════════════════

def _selftest() -> int:
    import tempfile
    ok = True

    def chk(name, cond, note=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {note}" if note and not cond else ""))

    print("[1] the subject is one-composite representable and the "
          "references are golden-derived")
    st = toy_qstep()
    qc = _f32_val(st["qc_bits"])
    chk("q row == f16(m*qc) exactly (the k2-reachable set)",
        np.array_equal(st["q_bits"],
                       f64_to_f16_bits(st["m"].astype(np.float64) * qc)))
    chk("determinism", toy_qstep()["q_scale"] == st["q_scale"])
    chk("k2 beats reconstruct m", all(
        127 * ((v := (p & 0xFF) if (p & 0xFF) < 128 else (p & 0xFF) - 256,
                w1 := ((p >> 8) & 0xFF) if ((p >> 8) & 0xFF) < 128
                else ((p >> 8) & 0xFF) - 256)[0]) + w1 == int(m)
        for grp, ms in zip(_k2_beats(st["m"]),
                           st["m"].reshape(-1, 8))
        for p, m in zip(grp, ms)))

    print("[2] the descriptor images")
    w = qstage_descriptor(st)
    chk("mask sets EN_QSTAGE (bit 12) and nothing else",
        w[fmt.W_MASK] == (1 << fmt.EN_QSTAGE))
    chk("bit 12 was reserved-0 in every legacy image",
        fmt.EN_QSTAGE == 12 and not (fmt.EN_ALL_NFEED >> 12) & 1)
    chk("W2_QC rides word 58 (reserved-0 until E-3b)", fmt.W_QC == 58)
    woff = qstage_descriptor(st, qstage=False)
    chk("the walk-off image is legal and QSTAGE-free",
        fmt.check2(woff[0], woff[1], woff[2], woff[3], woff[fmt.W_STEP],
                   cfg_d=st["d"]) == fmt.ERR_NONE
        and not (woff[fmt.W_MASK] >> fmt.EN_QSTAGE) & 1)
    e4 = toy_e4()
    chk("E-4b: the walk_e4 mask is the ONE-KICK set {QSTAGE,SCORE,PV,"
        "NFEED,NSRC}",
        e4_descriptor(e4)[fmt.W_MASK] == E4_MASK
        and E4_MASK == ((1 << fmt.EN_QSTAGE) | (1 << fmt.EN_SCORE)
                        | (1 << fmt.EN_PV) | (1 << fmt.EN_NFEED)
                        | (1 << fmt.EN_NSRC)))
    chk("E-4b: bit 13 (EN_NSRC) was reserved-0 in every legacy image",
        fmt.EN_NSRC == 13 and not (fmt.EN_ALL_NFEED >> 13) & 1)
    chk("E-4b: both fence images are pkg-LEGAL (the S2_CHECK fence, not "
        "the pkg check, must be what refuses them)",
        all(fmt.check2(x[0], x[1], x[2], x[3], x[fmt.W_STEP],
                       cfg_d=e4["d"]) == fmt.ERR_NONE
            for x in (e4_descriptor(e4, mask=E4_MASK_NONSRC),
                      e4_descriptor(e4, mask=E4_MASK_NSRCONLY))))

    print("[3] the programs build and audit clean")
    with tempfile.TemporaryDirectory() as td:
        with tg.at_d(D_TOY):
            m1 = build_walk_qstage(st, out_dir=Path(td), name="walk_qstage")
            m2 = build_walk_qstage(st, out_dir=Path(td),
                                   name="walk_qstage_off", qstage=False)
            m3 = build_walk_e4(e4, out_dir=Path(td), name="walk_e4")
            m4 = build_e4_refusal(e4, out_dir=Path(td),
                                  name="e4_fence_nonsrc",
                                  mask=E4_MASK_NONSRC,
                                  why="QSTAGE+NFEED without NSRC")
        for man in (m1, m2, m3, m4):
            a = man["audit"]
            chk(f"{man['name']}: audit clean (caps={a['caps']})",
                a["caps_with_e"] == 0 and a["violations"] == 0)
        prog = Path(m1["path"]).read_text()
        chk("the program never writes the q-sink level itself",
            not any(json.loads(ln).get("a") == 0x1000 + 0x70
                    and json.loads(ln)["op"] == "w"
                    and (json.loads(ln)["d"] & 0x30) == 0x30
                    for ln in prog.splitlines()))
        chk("the program pushes NO serializer/feeder/squant job between "
            "WALK_GO and the busy fall (the walker is the only sequencer)",
            _no_host_jobs_during_walk(prog))
        prog_e4 = Path(m3["path"]).read_text()
        chk("E-4b: walk_e4 kicks EXACTLY ONCE (one WALK_CTRL go edge)",
            sum(1 for ln in prog_e4.splitlines()
                if (o := json.loads(ln)).get("op") == "w"
                and o.get("a") == 0x1000 + W_CTRL
                and (o.get("d", 0) & 1) == 1) == 1)
        chk("E-4b: the host writes NOTHING between walker start and "
            "completion (xw DATA beats are the one disclosed exception)",
            _no_host_jobs_during_walk(prog_e4))
        chk("E-4b: the program never writes LAYER_CTRL between GO and the "
            "final level teardown (l_nsrc is the walker's, not the host's)",
            not any((o := json.loads(ln)).get("op") in ("w", "pw")
                    and o.get("a") == 0x1000 + 0x70
                    and o.get("d", 0) != 0
                    for ln in prog_e4.splitlines()))

    print("[4] the grader discriminates")
    caps = []
    i = 0

    def cap(sem, val):
        nonlocal i
        caps.append({"tag": f"{sem}_{i}", "i": i, "sem": sem,
                     "value": int(val) & 0xFFFFFFFF})
        i += 1
    cap("fs", st["q_scale"])
    for base in range(0, D_TOY, 8):
        for j in range(8):
            cap(f"ro_w{j}", int(st["q_codes"][base + j]))
    chk("all-correct capture set grades equal", grade_qstage(caps, st)["equal"])
    bad = [dict(c) for c in caps]
    for c in bad:
        if c["sem"] == "ro_w5":
            c["value"] = (c["value"] + 1) & 0xFFFFFFFF
            break
    chk("a moved staged code goes red", not grade_qstage(bad, st)["equal"])
    bad2 = [dict(c) for c in caps]
    bad2[0]["value"] ^= 1
    chk("a moved scale bit goes red", not grade_qstage(bad2, st)["equal"])

    print(f"ELANE-QSTAGE SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _no_host_jobs_during_walk(prog: str) -> bool:
    """Between the WALK_CTRL=3 write and the WALK_CTRL=0 write the program
    may only poll/read and stream XW DATA beats (0x3040/44/48 — the
    disclosed data movement; the FIFO-free poll paces it against the
    walker's own consumption). E-4b STRENGTHENED: any OTHER write there AT
    ALL — mailbox OR CSR: a job, a level (l_nsrc!), a route, a descriptor
    patch — would mean the host, not the walker, sequenced the chain. This
    is the predicate that states 'the host writes NOTHING between walker
    start and completion' about the emitted program itself."""
    xw_ok = {0x3040, 0x3044, 0x3048}
    in_walk = False
    for ln in prog.splitlines():
        o = json.loads(ln)
        if o.get("op") == "w" and o.get("a") == 0x1000 + W_CTRL:
            in_walk = (o["d"] & 1) == 1
            continue
        if in_walk and o.get("op") in ("w", "pw") \
                and o.get("a") not in xw_ok:
            return False
    return True


def smoke(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = toy_qstep(seed=args.seed)
    e4 = toy_e4(seed=args.seed + 10)
    with tg.at_d(D_TOY):
        plan = [build_walk_qstage(step, out_dir=out, name="walk_qstage"),
                build_walk_e4(e4, out_dir=out, name="walk_e4")]
        if not args.no_discriminate:
            plan.append(build_walk_qstage(step, out_dir=out,
                                          name="walk_qstage_off",
                                          qstage=False))
            # E-4b fence refusals ON THE TILE: both images are pkg-legal;
            # the S2_CHECK fence must refuse them (ESTK + code 2, state
            # unchanged) — checked reads, so ok==True IS the receipt.
            plan.append(build_e4_refusal(
                e4, out_dir=out, name="e4_fence_nonsrc",
                mask=E4_MASK_NONSRC,
                why="QSTAGE+NFEED without NSRC (the measured wedge shape)"))
            plan.append(build_e4_refusal(
                e4, out_dir=out, name="e4_fence_nsrconly",
                mask=E4_MASK_NSRCONLY,
                why="NSRC without NFEED (ownership with nothing to feed)"))
            # V-record discriminator: a stored V f16 element moved at build,
            # graded against the UNPERTURBED golden — o8 must go red, which
            # proves the tile's o8 really flowed through the stored records.
            # The delta is searched ON GOLDEN FIRST (the elane gamma-delta
            # idiom): CQ-8 quantization can absorb a small f16 move.
            e4v = dict(e4)
            for flip in (0x4000, 0x0800, 0x8000):
                v16p = e4["V16"].copy()
                v16p[0, 3] ^= np.uint16(flip)
                rot = tf.rope_fx(f16_bits_to_f64(e4["q_bits"]),
                                 float(e4["pos"]), tf.rope_theta(e4["d"]))
                gp = at.attention_core(np.asarray(rot, dtype=np.float64),
                                       e4["K16"], v16p, at.TIER_CQ8, G=128)
                if not np.array_equal(np.asarray(gp.o8, dtype=np.int64),
                                      e4["gold_o8"]):
                    e4v["V16"] = v16p
                    break
            else:
                raise AssertionError("no V flip moved the golden o8")
            plan.append(build_walk_e4(e4v, out_dir=out, name="walk_e4_vbit"))

    rows = []
    for man in plan:
        t0 = time.time()
        r = glo._run([man["path"]], binary=args.binary,
                     tile_div=args.tile_div, timeout_s=args.timeout,
                     cap_out=str(out / f"{man['name']}.cap.jsonl"))
        wall = time.time() - t0
        row = dict(name=man["name"], stage=man["stage"], ok=bool(r["ok"]),
                   rc=r["rc"], caps=len(r["captures"]), wall=round(wall, 1))
        if man["name"] == "walk_qstage":
            row["grade"] = grade_qstage(r["captures"], step)
        elif man["name"] == "walk_e4":
            row["grade"] = grade_e4(r["captures"], e4)
        elif man["name"] == "walk_e4_vbit":
            # graded against the UNPERTURBED golden: must be red at o8.
            # A verdict needs an actual gradeable run — with zero captures
            # the old `not g["o8"]["equal"]` was vacuously True (the
            # 2026-08-04 wrong-twin failure mode; see elane_norm_feed's
            # discriminator-verdict hygiene note).
            ran = disc_ran(r)
            g = grade_e4(r["captures"], e4) if ran else {}
            row["grade"] = {"equal": bool(g.get("equal")),
                            "ran": ran,
                            "o8_moved": bool(ran and not g["o8"]["equal"]),
                            "note": "v-bit discriminator"}
        elif man["name"].startswith("e4_fence_"):
            # E-4b fence: the refusal receipts ARE checked reads (ESTK +
            # code 2, busy down, levels unmoved), so ok==True means the
            # tile REFUSED exactly as fenced. GREEN expected.
            row["grade"] = {"equal": bool(r["ok"]),
                            "note": "refusal fence (checked reads)"}
        else:
            # the discriminator: same drain, nothing staged — the run MUST
            # go red AT THE DOCUMENTED SEAM: the walker arms nothing, so
            # the host's xw stream-pacing write stalls ('pw stall' — both
            # executors print that substring; measured 2026-08-04 on the
            # clean b64 twin, abort at the first wbeats pacing write). Any
            # OTHER failure (phase-A geometry, loader stall) is not this
            # discrimination — recorded so the verdict below can cite it.
            row["grade"] = {"equal": bool(r["ok"]),
                            "note": "walk-off must be red"}
            row["pw_stall"] = "pw stall" in (r.get("log") or "")
        rows.append(row)
        eprint(f"  [{man['name']:<16}] rc={r['rc']} ok={r['ok']} "
               f"caps={len(r['captures'])} {wall:.1f}s "
               f"grade={row['grade'].get('equal')}")

    report = dict(binary=args.binary or "(bridge default)",
                  tile_div=args.tile_div, seed=step["seed"],
                  stages=rows)
    (out / "report.json").write_text(json.dumps(report, indent=1))

    print(f"\nE-LANE E-3b walk_qstage @ tile_div={args.tile_div} "
          f"(binary={args.binary or 'bridge default'})")
    byname = {r["name"]: r for r in rows}
    g = byname["walk_qstage"]
    gq = g["grade"].get("q_codes", {})
    got_ok = 64 - gq.get("n_diff", 0) if gq.get("got", 0) == 64 else 0
    print(f"  walk_qstage      rc={g['rc']} ok={g['ok']} "
          f"grade={g['grade'].get('equal')} "
          f"(codes {got_ok}/64, "
          f"scale {g['grade'].get('scale', {}).get('equal')})")
    ok = g["ok"] and g["grade"].get("equal")
    e = byname["walk_e4"]
    eg = e["grade"]
    print(f"  walk_e4          rc={e['rc']} ok={e['ok']} "
          f"grade={eg.get('equal')} (o8 {eg.get('o8', {}).get('equal')}, "
          f"h2 {eg.get('h2_codes', {}).get('equal')}, "
          f"fs {eg.get('fs', {}).get('equal')}, "
          f"s_c {eg.get('s_c', {}).get('equal')})")
    ok = ok and e["ok"] and eg.get("equal")
    # discriminator verdicts are ANCHORED: red counts only against a green
    # control on the same binary, and only at the documented failure seam —
    # 'it failed somewhere' is not a bite (elane_norm_feed hygiene note).
    qs_green = bool(g["ok"] and g["grade"].get("equal"))
    e4_green = bool(e["ok"] and eg.get("equal"))
    if "walk_qstage_off" in byname:
        off = byname["walk_qstage_off"]
        off_red = bool(qs_green and not (off["ok"]
                                         and off["grade"].get("equal"))
                       and off.get("pw_stall"))
        print(f"  walk_qstage_off  rc={off['rc']} RED={off_red} "
              f"(pw_stall={off.get('pw_stall')}, control_green={qs_green})")
        ok = ok and off_red
    if "walk_e4_vbit" in byname:
        vb = byname["walk_e4_vbit"]
        caught = bool(e4_green and vb["grade"].get("ran")
                      and vb["grade"].get("o8_moved")
                      and not vb["grade"].get("equal"))
        print(f"  walk_e4_vbit     rc={vb['rc']} o8_moved="
              f"{vb['grade'].get('o8_moved')} RED={caught} "
              f"(ran={vb['grade'].get('ran')}, control_green={e4_green})")
        ok = ok and caught
    for fname in ("e4_fence_nonsrc", "e4_fence_nsrconly"):
        if fname in byname:
            fr = byname[fname]
            print(f"  {fname:<16} rc={fr['rc']} REFUSED={fr['ok']} "
                  f"(ESTK+DESC, levels unmoved)")
            ok = ok and fr["ok"]
    print(f"ELANE-QSTAGE SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def redproof(args) -> int:
    """E-4b pre-change-twin RED proof: run the SINGLE-KICK walk_e4 image on
    a pre-E-4b binary. Its W2_MASK[13] is a reserved bit there, so the old
    walk_desc2_check must REFUSE it (WALK_ERR_DESC sticky) — the program's
    clean-status/level checks then fail and the run is loudly red. PASS of
    this mode == the run went RED."""
    assert args.binary, "--redproof requires --binary <pre-E-4b twin>"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e4 = toy_e4(seed=args.seed + 10)
    with tg.at_d(D_TOY):
        man = build_walk_e4(e4, out_dir=out, name="walk_e4")
    r = glo._run([man["path"]], binary=args.binary, tile_div=args.tile_div,
                 timeout_s=args.timeout,
                 cap_out=str(out / "walk_e4.redproof.cap.jsonl"))
    red = not r["ok"]
    (out / "redproof.json").write_text(json.dumps(dict(
        binary=args.binary, rc=r["rc"], ok=bool(r["ok"]),
        caps=len(r["captures"]), red=red), indent=1))
    print(f"E-4b REDPROOF on {args.binary}: rc={r['rc']} ok={r['ok']} "
          f"caps={len(r['captures'])} -> "
          f"{'RED as required (PASS)' if red else 'GREEN — the twin is NOT pre-E-4b (FAIL)'}")
    return 0 if red else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--redproof", action="store_true",
                    help="run the single-kick walk_e4 on a PRE-E-4b binary "
                         "and require it to go RED (reserved-bit refusal)")
    ap.add_argument("--chain", action="store_true",
                    help="E-6: the one-kick chain WITH the walked fuel-fed "
                         "epilogue (needs --binary <DDR=1 b64 twin>)")
    ap.add_argument("--chain7", action="store_true",
                    help="E-7: the one-kick chain whose NORM2 runs with a "
                         "WALKER-FETCHED gamma (needs --binary <E-7 twin>)")
    ap.add_argument("--chain8", action="store_true",
                    help="E-8: QKV joins the one-kick fetched-gamma chain "
                         "(fence-5 positive; needs --binary <E-7 twin>)")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tile-div", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--timeout-off", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--no-discriminate", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.chain:
        return chain(args)
    if args.chain7:
        return chain7(args)
    if args.chain8:
        return chain8(args)
    if args.build:
        step = toy_qstep(seed=args.seed)
        with tg.at_d(D_TOY):
            for nm, qs in [("walk_qstage", True), ("walk_qstage_off", False)]:
                m = build_walk_qstage(step, out_dir=Path(args.out), name=nm,
                                      qstage=qs)
                print(f"  built {m['name']}: {m['path']} (audit={m['audit']})")
        return 0
    if args.smoke:
        return smoke(args)
    if args.redproof:
        return redproof(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
