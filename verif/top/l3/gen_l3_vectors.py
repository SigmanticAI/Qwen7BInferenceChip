#!/usr/bin/env python3
"""gen_l3_vectors.py — Layer-3 case scripts for tb_apex_l3 (the REAL
apex_top, host-sequenced per its v0.1 boundary contract).

Golden arbiter: golden/apex_golden/attention.py (attention_core for the
named tensor-replay cases, attention_fx for the full-chain random cases).
Final INT8 o8 outputs and EVERY mid-chain tensor (KVQ write-port fp16,
feeder scales, s_q/s_c, S-3 scores, ASU probabilities) are checked
BIT-EXACT against the golden run.

════════════════════════════════════════════════════════════════════════════
FEASIBILITY REGISTER (v0.2 state, 2026-07-09 — F-1 and F-5 CLOSED with
kept-PASSING regressions; F-2/F-3/F-4 remain as documented)
════════════════════════════════════════════════════════════════════════════
 F-1 CLOSED. T per attention row is now bounded by apex_top's T_ROW_MAX
     envelope parameter (=128, sizes seam_score_dequant N_MAX AND the
     apex_scale_quant per-job buffers for the T-column S-4 P-requant
     job). The named cases calib d64_T128 (T=128), calib d64_T70 (T=70),
     adv/outlier1000 (T=128) and calib d128_T100 (T=100, D=128) are
     replayed FULL LENGTH below — kept-passing F-1 regressions replacing
     the historical *_T64sub substitutes. When ceil(T/8) code beats
     exceed one stage-buffer row (T > 8*BPR, i.e. T=128 at D=64), the c8
     row is staged across act-stage bank1 + bank0 row0 (see
     stage_c8_load/emit; bank0 row0 is dead by that phase).
 F-2 CLOSED (2026-07-09, D-024). apex_top now instantiates the KVQ TIER
     BANK (apex_kvq_bank: 3 verified engines, CQ-8/CQ-4/CQ-4+) behind a
     live tier mux: TIER_CTRL host tier or the TIP-driven auto_tier map
     (tip_override — D-022 actuation), rt_kv_user routes grouped keys
     (tuser=0, WRITE_ADDR group base, D-008 CSR FLUSH for partial
     groups), and the b64 build carries a real OUTLIER_K=2/MASK_FILE
     mask ROM. Kept-passing regressions HERE: adv_T1_cq4p (T=1 group
     flush through the tile), adv_outlier1000_cq4p (T=128, 8 full key
     groups), tip_auto_mixed (per-block TIP-driven CQ-4/CQ-4+ map,
     bit-exact vs attention_core(tier_map=...)). Fail-first evidence:
     logs/prefix_f2_keptfail_before_fix.log (KVP INFO_TIER timeout —
     TIER_CTRL unwired on the pre-fix tile). CSR INFO_TIER now reports
     the build TRUTH ({mask?CQ4P:0, CQ4, CQ8}), not a hardwired 0x7.
 F-3 T > 8 replays raise the DOCUMENTED frame stickies by construction:
     MXE accepts N <= 8 per descriptor (mxe_cfg_pkg: no N tiling v0.1),
     so a T-column dequant job spans ceil(T/8) M=1 MXE jobs whose
     res.last=1 beats raise the sd frame_error sticky (err_sticky[8]);
     the TIP tap is BLOCK_M=1 x BLOCK_N=8, so any tile longer than 8
     aborts in TIP with the frame sticky (err_sticky[14]) — TIP still
     gets NO decision when T>8 (structural residual). CLOSED half
     (2026-07-09): the stickies are now CLEARABLE — every per-source
     sticky is latched in apex_top's ERR_STICKY window (0x58, W1C;
     bit 14 also pulses TIP frame_err_clear), and EVERY script here
     ends with the F-3 set->clear->verify sequence. Fail-first:
     logs/prefix_f3_keptfail_before_fix.log (ESTK stuck at 0x0001).
 F-4 KVQ eviction/capacity: KVQ_DEPTH is raised to 256 in the L3 builds
     (2*T = 256 records at the T=128 envelope fits exactly; §8's
     eviction-path obligation still has no tile-level test).
 F-5 CLOSED (was: D=128 structurally unsupported — 4-bit aj_nb/wj_nb/
     job_nb truncated the 16-beat D=128 row count to 0 and wedged the
     phase-B pipeline; latently, apex_stage_buf PAT_D indexed the column
     block as 4'(sel_q[2:0]), aliasing blocks 8..15 onto 0..7). Fixed
     2026-07-09: nb fields are now NB_W = clog2(D/8)+1 bits (derived,
     end-to-end: apex_stage_buf, apex_top ports, both TBs) and PAT_D
     uses the full legality-checked sel. Regressions, all kept-PASSING:
     (a) bug_d128_stagebuf_nb — the exact wedge case, now expected to
     complete bit-exact (a regression to the fail signature 'drive_g
     stall' means the fix broke); (b) tb_stagebuf_patd.sv — directed
     D=128 unit test reading ALL PAT_D column blocks 0..15 (fails on
     the pre-fix alias, verified by mutant run); (c) the full D=128
     attention matrix below (named d128_T100 + random D=128 cases).

Named-tensor replay path (host-sequenced INJECTION, no new RTL): a loader
token (x8 = all-ones, gamma = [32512, 256, 0...]) yields h8 = [127, 1,
0, ...] with s_h = 2^-5 through the REAL RMSNorm+feeder; then any INT32
row v (|v| <= 2047) is produced as MXE WS K=2 accumulators (acc = 127*W0
+ W1, host weights via xw_*), and apex_scale_quant's exact acc*f32
arithmetic turns (v = +/-fp16_mantissa, comp = 2^exponent) into EXACT
arbitrary fp16 K/V values on the KVQ write port (self-checked below) and
exact fp16-grid q rows into the Q7 quant. Every value still flows through
real tile datapaths only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden.attention import (  # noqa: E402
    SCORE_FRAC, TIER_CQ4, TIER_CQ4P, TIER_CQ8, attention_core, attention_fx,
    attention_ref_full, quant_rows_i8, rmsnorm_fx, score_dequant_fx,
)
from apex_golden.compute import online_softmax_fx  # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402

# TIP golden replica (verif/tip discipline — single source of truth for the
# decision + importance-update + tier-suggestion rules; the D-022 auto case
# derives its expected auto_tier map from it)
import importlib.util  # noqa: E402

_tip_spec = importlib.util.spec_from_file_location(
    "gen_tip_vectors", REPO / "verif/tip/smoke/gen_tip_vectors.py")
tipg = importlib.util.module_from_spec(_tip_spec)
_tip_spec.loader.exec_module(tipg)

SEED = 0xA9E7_1300
OP_GEMM_WS = 0x01
OP_GEMM_OS = 0x02

KV = {"CTRL": 0x00, "STATUS": 0x04, "INFO_TIER": 0x0C, "OCC": 0x24,
      "WADDR": 0x28, "RADDR": 0x2C, "IRQ_STATUS": 0x38, "INFO_OUTK": 0x3C,
      # D-027 (S12): loadable-outlier-mask window (engine AXI-Lite space —
      # same numerals as the tile WALK window, physically separate bus)
      "MASK0": 0x50, "MASK1": 0x54, "MASK2": 0x58, "MASK3": 0x5C,
      "MASK_CTRL": 0x60}
CSR = {"CTRL": 0x00, "STATUS": 0x04, "INFO_N": 0x08, "INFO_D": 0x0C,
       "INFO_G": 0x10, "INFO_TIER": 0x14, "INFO_VER": 0x18,
       "TIER_CTRL": 0x20, "THRESH": 0x24, "FLUSH": 0x28, "IMP": 0x2C,
       "PERF_CTRL": 0x30, "PERF_CYC": 0x34, "PERF_B0": 0x38,
       "ERR_STICKY": 0x58}
G_CFG = 16       # tile INFO_G / key-group size (F-2 CLOSED: keys group here)
OI_B64 = (5, 50)  # b64 build outlier mask channels (CQ-4+ engine ROM)
TIER_CODE = {TIER_CQ8: 0, TIER_CQ4: 1, TIER_CQ4P: 2}
VEC = REPO / "golden/vectors"


def to16(x):
    x = np.asarray(x, dtype=np.float64)
    return f64_to_f16_bits(x).reshape(x.shape)


def top2_outliers(K16):
    amax = np.max(np.abs(f16_bits_to_f64(K16).reshape(K16.shape)), axis=0)
    return np.sort(np.argsort(amax)[-2:])


def f32_bits_exact(x: float, what: str) -> int:
    b = np.float64(x).astype(np.float32)
    assert float(np.float64(b)) == float(x), f"{what}: not fp32-exact: {x}"
    bits = int(b.view(np.uint32))
    assert bits & 0x1FFF == 0, f"{what}: not fp16-grade: {bits:08x}"
    assert 0 < ((bits >> 23) & 0xFF) < 255 and bits >> 31 == 0, \
        f"{what}: not positive normal: {bits:08x}"
    return bits


def desc_words(opcode, m, k, n, *, mode_os=0, acc=0, rq_en=0,
               rq_scale=0, rq_shift=0):
    v = (opcode & 0xFF) | (m << 8) | (k << 20) | (n << 32) \
        | (rq_scale << 44) | (rq_shift << 60) | (rq_en << 65) \
        | (acc << 66) | (mode_os << 67)
    return [(v >> (32 * i)) & 0xFFFFFFFF for i in (3, 2, 1, 0)]


def wgt_beats_ws(W, j):
    """mxe_ctrl WLOAD stream for B = W[:, 8j:8j+8], K = W.shape[0]."""
    K = W.shape[0]
    beats = []
    for p in range(K // 8):
        for c in range(8):
            w = 0
            for r in range(8):
                w |= (int(W[8 * p + r][8 * j + c]) & 0xFF) << (8 * r)
            beats.append(w)
    return beats


def decompose_f16(bits16: int):
    """(v_int32, comp_f32_bits) with v*comp EXACTLY the fp16 value (after
    one RNE-to-f16 that is a no-op / sign-preserving for zero)."""
    s = (bits16 >> 15) & 1
    e = (bits16 >> 10) & 0x1F
    m = bits16 & 0x3FF
    assert e != 0x1F, f"inf/nan fp16 not injectable: {bits16:04x}"
    if e == 0 and m == 0:
        if s:                      # -0.0: v*comp = -2^-126, RNE-f16 -> -0.0
            return -1, (-126 + 127) << 23, True
        return 0, (-24 + 127) << 23, False
    mant = m if e == 0 else (1024 + m)
    k = -24 if e == 0 else e - 25
    v = -mant if s else mant
    comp = (k + 127) << 23
    # self-check: golden one-RNE narrowing of v*2^k reproduces the bits
    got = int(f64_to_f16_bits(np.array([float(v) * (2.0 ** k)]))[0])
    assert got == bits16, f"decompose_f16 self-check {bits16:04x} -> {got:04x}"
    return v, comp, False


def chunks(T):
    out = []
    c0 = 0
    while c0 < T:
        out.append((c0, min(8, T - c0)))
        c0 += min(8, T - c0)
    return out


class Script:
    def __init__(self, D):
        self.lines = []
        self.n_expect = 0
        self.D = D
        self.BPR = D // 8

    def emit(self, *toks):
        self.lines.append(" ".join(str(t) for t in toks))

    def csrw(self, a, d): self.emit("CSRW", f"{a:02x}", f"{d:08x}")

    def csrr(self, a, mask, exp):
        self.emit("CSRR", f"{a:02x}", f"{mask:08x}", f"{exp:08x}")
        self.n_expect += 1

    def csrp(self, a, mask, exp):
        self.emit("CSRP", f"{a:02x}", f"{mask:08x}", f"{exp:08x}")

    def csrn(self, a, mask):
        self.emit("CSRN", f"{a:02x}", f"{mask:08x}")
        self.n_expect += 1

    def kvw(self, a, d): self.emit("KVW", f"{a:02x}", f"{d:08x}")

    def kvp(self, a, mask, exp):
        self.emit("KVP", f"{a:02x}", f"{mask:08x}", f"{exp:08x}")

    def route(self, *, fsrc=0, fdst=0, asrc=0, wsrc=0, rdst=0, qsrc=0,
              kvu=1, blk=0, poll=True):
        # rt_* are LEVELS: change only while the touched paths are idle.
        # Quiescence is VERIFIED (STATUS.idle poll), never assumed — the
        # smoke phase-F wedge root cause (2026-07-09).
        # kvu (rt_kv_user) defaults 1 = VALUE path (historical hardwire);
        # kvu=0 routes grouped KEYS (F-2). poll=False is legal ONLY for a
        # blk/kvu-only change whose touched consumers are provably
        # quiescent (KVQ reads are idle-polled per record; TIP samples
        # s_blk on the tile's LAST beat only) — used by the per-chunk
        # tier-map routing of the D-022 auto case.
        if poll:
            self.csrp(CSR["STATUS"], 0x1, 0x1)
        r = fsrc | (fdst << 1) | (asrc << 2) | (wsrc << 3) | (rdst << 4) \
            | (qsrc << 6) | (kvu << 7) | (blk << 8)
        self.emit("ROUTE", f"{r:04x}")

    def imp(self, hi, lo):
        self.emit("IMP", f"{hi:04x}", f"{lo:04x}")

    def etipt(self, blk, tier, fp16):
        self.emit("ETIPT", f"{blk:02x}", f"{tier:x}", f"{fp16:x}")
        self.n_expect += 1

    def desc(self, words): self.emit("DESC", *[f"{w:08x}" for w in words])

    def xrow(self, row):
        self.emit("XR", *[f"{int(v) & 0xFF:02x}" for v in row])

    def grow(self, g):
        self.emit("GR", *[f"{int(v) & 0xFFFF:04x}" for v in g])

    def wbeats(self, beats):
        for b in beats:
            self.emit("WB", f"{b:016x}")

    def fjob(self, rows): self.emit("FJOB", f"{rows:x}")
    def qjob(self, mode, cols): self.emit("QJOB", mode, f"{cols:x}")
    def sjob(self, cols): self.emit("SJOB", f"{cols:x}")
    def ljob(self, beats, lanes): self.emit("LJOB", f"{beats:x}", f"{lanes:x}")

    def aj(self, op, bank, pat, rows, nb, sel):
        self.emit("AJ", op, bank, pat, f"{rows:x}", f"{nb:x}", f"{sel:x}")

    def wj(self, op, bank, pat, rows, nb, sel):
        self.emit("WJ", op, bank, pat, f"{rows:x}", f"{nb:x}", f"{sel:x}")

    def qs(self, bits): self.emit("QS", f"{bits:08x}")
    def cs(self, bits): self.emit("CS", f"{bits:08x}")

    def efs(self, bits, last):
        self.emit("EFS", f"{int(bits):04x}", int(last))
        self.n_expect += 1

    def ess(self, bits, last):
        self.emit("ESS", f"{int(bits):04x}", int(last))
        self.n_expect += 1

    def ero(self, lanes, last):
        self.emit("ERO", *[f"{int(v) & 0xFFFFFFFF:08x}" for v in lanes],
                  int(last))
        self.n_expect += 1

    def etip(self, blk):
        self.emit("ETIP", f"{blk:02x}")
        self.n_expect += 1

    def tap(self, kind, vals, width):
        self.emit(kind, f"{len(vals):x}")
        for v in vals:
            self.emit(f"{int(v) & ((1 << width) - 1):0{width // 4}x}")
        self.n_expect += len(vals)

    def estk(self, mask, exp):
        self.emit("ESTK", f"{mask:04x}", f"{exp:04x}")
        self.n_expect += 1


# ── shared phases ────────────────────────────────────────────────────────────

def stage_c8_load(s, nbT):
    """LOAD the S-4 c8 row (nbT lane8 beats) into the act stage buffer.

    F-1 staging: one stage-buffer row holds BPR beats, so a T=128 row at
    D=64 (nbT=16) spans two rows. LOAD jobs always start at row 0 of a
    bank, so the split uses bank1 (first min(nbT,BPR) beats) + bank0 row0
    (remainder) — bank0 row0 (loader token / h8 row 0) is dead by this
    phase in every script that calls this."""
    BPR = s.BPR
    s.aj(0, 1, 0, 1, min(nbT, BPR), 0)
    if nbT > BPR:
        assert nbT - BPR <= BPR, f"c8 needs >2 stage rows (nbT={nbT})"
        s.aj(0, 0, 0, 1, nbT - BPR, 0)


def stage_c8_emit(s, nbT, desc):
    """EMIT the staged c8 row as one MXE act stream (PAT_ROW; MXE ignores
    the informational beat `last`, so a split emission is transparent).
    Ordering (mxe_ctrl contract — INGEST consumes ALL act beats first,
    weights load concurrently): first EMIT job, then the descriptor (so
    MXE starts ingesting), then the tail EMIT — the tail's job_ready
    waits for the first's post-skid done, which needs MXE consuming."""
    BPR = s.BPR
    s.aj(1, 1, 0, 1, min(nbT, BPR), 0)
    s.desc(desc)
    if nbT > BPR:
        s.aj(1, 0, 0, 1, nbT - BPR, 0)


def phase_a(s, tiers_used=(0,)):
    """CSR sanity + illegal-descriptor reject + D-024 tier-bank init.

    tiers_used: kvq_tier_e codes this case stores/reads through — each
    engine is enabled through the tier-routed kv_* window and its OWN
    INFO_TIER is read back (proves the AXI tier mux per engine). Ends with
    TIER_CTRL = tiers_used[0] (host mode). INFO_TIER (CSR) checks the
    BUILD TRUTH: the b64 build carries the OUTLIER_K=2 mask (0x7), the
    b128 build does not (0x3 — CQ-4+ degenerates without a mask)."""
    s.emit("// phase A: CSR sanity + illegal-descriptor reject + tier init")
    s.csrr(CSR["INFO_VER"], 0xFFFFFFFF, 0x00010000)
    s.csrr(CSR["INFO_N"], 0xFFFFFFFF, 8)
    s.csrr(CSR["INFO_D"], 0xFFFFFFFF, s.D)
    s.csrr(CSR["INFO_G"], 0xFFFFFFFF, G_CFG)
    s.csrr(CSR["INFO_TIER"], 0xFFFFFFFF, 0x7 if s.D == 64 else 0x3)
    s.csrr(CSR["STATUS"], 0x3, 0x1)
    s.csrw(CSR["CTRL"], 0x1)
    s.csrw(CSR["PERF_CTRL"], 0x3)
    for tc in reversed(tiers_used):
        s.csrw(CSR["TIER_CTRL"], tc)
        s.csrr(CSR["TIER_CTRL"], 0xFFFFFFFF, tc)
        s.kvw(KV["CTRL"], 0x2)                    # enable engine[tc]
        s.kvp(KV["INFO_TIER"], 0xFFFFFFFF, tc)    # engine's own truth
    s.desc(desc_words(OP_GEMM_WS, 1, 0, 1))       # K=0: §3 reject
    s.csrp(CSR["STATUS"], 0x2, 0x2)
    s.csrw(CSR["STATUS"], 0x2)
    s.csrr(CSR["STATUS"], 0x2, 0x0)


def score_phase(s, r, T, blkmap=None):
    """Phase F: Q·K̂ᵀ in ceil(T/8) column chunks -> S-3 -> {TIP,ASU} -> S-4.

    blkmap (D-022 auto mode): rt_tip_blk per chunk — in auto mode the tier
    bank routes chunk c's K-record reads to engine[auto_tier[blkmap[c]]].
    The mid-phase route flips are poll-free but provably safe: every KVQ
    read is bracketed by its own engine-idle poll, and TIP samples s_blk
    only on the tile's LAST beat (mid-tile values are don't-care)."""
    D, BPR = s.D, s.BPR
    s.emit(f"// phase F: scores+softmax+P-requant, {len(chunks(T))} chunks")
    bm = blkmap if blkmap is not None else [0] * len(chunks(T))
    s.route(rdst=2, fsrc=1, fdst=1, asrc=1, wsrc=1, qsrc=1, blk=bm[0])
    s.tap("TAPSC", [int(v) for v in r.score_fx], 32)
    s.tap("TAPPR", [int(v) for v in r.p_q115], 16)
    s.sjob(T)
    s_q_val = float(f16_bits_to_f64(np.array([r.s_q]))[0])
    s_k_val = f16_bits_to_f64(np.asarray(r.s_k, dtype=np.uint16))
    comp64 = s_q_val * s_k_val * (float(1 << SCORE_FRAC) / math.sqrt(float(D)))
    comp32 = comp64.astype(np.float32).view(np.uint32)
    for t in range(T):
        s.cs(int(comp32[t]))
    s.qjob(1, T)                                   # S-4 P-requant fold
    for t in range(T):
        sv = float(f16_bits_to_f64(np.array([r.s_v[t]]))[0])
        s.qs(f32_bits_exact(sv * 2.0 ** -15, f"S-4 comp t={t}"))
    for ci, (c0, nc) in enumerate(chunks(T)):
        if blkmap is not None and bm[ci] != bm[max(ci - 1, 0)]:
            s.route(rdst=2, fsrc=1, fdst=1, asrc=1, wsrc=1, qsrc=1,
                    blk=bm[ci], poll=False)
        s.wj(0, 0, 0, nc, BPR, 0)                  # wstage LOAD k8 chunk
        s.fjob(nc)
        for t in range(c0, c0 + nc):
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["RADDR"], t)                  # K record t
            s.efs(r.s_k[t], t == c0 + nc - 1)      # scl_last per feeder job
        s.wj(1, 0, 1, nc, BPR, 0)                  # EMIT PAT_T (k8 chunk ᵀ)
        s.aj(1, 1, 0, 1, BPR, 0)                   # act: q8 (bank1 row 0)
        s.desc(desc_words(OP_GEMM_OS, 1, D, nc, mode_os=1))
    if T <= 8:
        s.etip(0)                                  # F-3: no decision if T>8
    s.ess(r.s_c, 1)
    stage_c8_load(s, (T + 7) // 8)                 # LOAD c8 (F-1 staging)


def pv_phase(s, r, T, blkmap=None):
    """Phase G: P·V̂ + C-2 epilogue, K=T contraction in v8 chunks per block.

    Order per block j (mxe_ctrl: INGEST first): c8 act emission wrapped
    around the descriptor (stage_c8_emit), then the v8 weight chunks.
    blkmap: as in score_phase — V-record reads route per token block."""
    D, BPR = s.D, s.BPR
    scale, shift = r.rq
    nbT = (T + 7) // 8
    s.emit("// phase G: P·V and epilogue")
    bm = blkmap if blkmap is not None else [0] * len(chunks(T))
    s.route(rdst=0, fsrc=1, fdst=1, asrc=1, wsrc=1, qsrc=1, blk=bm[0])
    for j in range(BPR):
        stage_c8_emit(s, nbT,                      # act: c8 row (F-1) + desc
                      desc_words(OP_GEMM_OS, 1, T, 8, mode_os=1, rq_en=1,
                                 rq_scale=int(scale), rq_shift=int(shift)))
        for ci, (c0, nc) in enumerate(chunks(T)):
            if blkmap is not None and (ci == 0 or bm[ci] != bm[ci - 1]):
                s.route(rdst=0, fsrc=1, fdst=1, asrc=1, wsrc=1, qsrc=1,
                        blk=bm[ci], poll=False)
            s.wj(0, 0, 0, nc, BPR, 0)              # wstage LOAD v8 chunk
            s.fjob(nc)
            for t in range(c0, c0 + nc):
                s.kvp(KV["STATUS"], 0x1, 0x1)
                s.kvw(KV["RADDR"], T + t)          # V record T+t
                s.efs(r.s_v[t], t == c0 + nc - 1)
            s.wj(1, 0, 2, nc, 1, j)                # EMIT PAT_D block j
        s.ero([int(v) for v in r.o8[8 * j:8 * j + 8]], 1)


def final_phase(s, T, extra_stk=0):
    s.emit("// final: quiescence + sticky accounting + PERF + F-3 clear")
    s.emit("ENDTAPS")
    s.n_expect += 3
    s.csrp(CSR["STATUS"], 0x1, 0x1)
    s.kvp(KV["STATUS"], 0x1, 0x1)
    if T > 8:
        # F-3: sd frame (chunked res.last) + TIP frame set STATUS[1]
        s.csrr(CSR["STATUS"], 0x2, 0x2)
        s.csrw(CSR["STATUS"], 0x2)
    s.csrr(CSR["STATUS"], 0x2, 0x0)
    s.csrn(CSR["PERF_CYC"], 0xFFFFFFFF)
    s.csrn(CSR["PERF_B0"] + 4 * 1, 0xFFFFFFFF)
    exp = 0x0001 | extra_stk                       # phase A illegal desc
    if T > 8:
        exp |= 0x0100 | 0x4000                     # sd frame + TIP frame
    s.estk(0xFFFF, exp)
    # F-3 CLOSED: set -> W1C-clear (0x58; bit 14 also pulses TIP
    # frame_err_clear) -> verify both the window readback and the pins
    s.csrw(CSR["ERR_STICKY"], 0x0000FFFF)
    s.csrr(CSR["ERR_STICKY"], 0xFFFFFFFF, 0x00000000)
    s.estk(0xFFFF, 0x0000)
    s.emit("DONE")


# ── named-tensor replay (attention_core) ─────────────────────────────────────

def loader_phase(s):
    """One loader token through the REAL RMSNorm+feeder: h8=[127,1,0...]."""
    D = s.D
    x8 = [1] * D
    gamma = [32512, 256] + [0] * (D - 2)
    y, _, _ = rmsnorm_fx(x8, gamma)
    h8, s_h = quant_rows_i8(np.array([y], dtype=np.float64) / 256.0)
    assert list(h8[0][:2]) == [127, 1] and not np.any(h8[0][2:]), "loader"
    s.emit("// loader token: h8=[127,1,0...] via real RMSNorm+feeder")
    s.route(fsrc=0, fdst=0, asrc=0, rdst=0)
    s.aj(0, 0, 0, 1, s.BPR, 0)
    s.fjob(1)
    s.xrow(x8)
    s.grow(gamma)
    s.efs(int(s_h[0]), 1)


def inject_jobs(s, vc_pairs, mode, waddr=None):
    """One squant job of D elements built from (v, comp) pairs; the v values
    are produced as K=2 WS accumulators acc = 127*W0 + W1 from the loader
    activation row. waddr=None with mode 0 = grouped-key member token: the
    engine is mid-group (ST_KACCEPT, not idle) and WRITE_ADDR was already
    programmed to the group base — no poll, no address write."""
    D, BPR = s.D, s.BPR
    if mode == 0 and waddr is not None:
        s.kvp(KV["STATUS"], 0x1, 0x1)
        s.kvw(KV["WADDR"], waddr)
    s.qjob(mode, D)
    for (_, comp) in vc_pairs:
        s.qs(comp)
    s.ljob(BPR, 8)
    for j in range(BPR):
        beats = []
        for c in range(8):
            v = vc_pairs[8 * j + c][0]
            w0 = int(round(v / 127.0))
            w1 = v - 127 * w0
            assert -128 <= w0 <= 127 and -128 <= w1 <= 127
            beats.append((w0 & 0xFF) | ((w1 & 0xFF) << 8))
        s.aj(1, 0, 0, 1, 1, 0)                     # act: loader beat 0
        s.desc(desc_words(OP_GEMM_WS, 1, 2, 8))
        s.wbeats(beats)


def store_kv_phase(s, K16, V16, tier=TIER_CQ8):
    """Inject the K/V fp16 rows through squant MODE_F16 into the KVQ bank.

    CQ-8: value path (kvu=1), per-token WRITE_ADDR (the historical tile
    wiring). CQ-4/CQ-4+ (F-2 CLOSED): keys take the GROUPED path — kvu=0,
    WRITE_ADDR = group base once per G_CFG-token group, tokens streamed
    back-to-back (the engine parks in ST_KACCEPT between tokens, so no
    per-token idle polls), a partial tail group closed by the D-008 CSR
    FLUSH; values per token on the value path."""
    T = K16.shape[0]
    key_grouped = tier != TIER_CQ8
    s.emit(f"// inject K/V fp16 rows through squant MODE_F16 -> KVQ ({tier})")
    kf = [int(v) for v in np.asarray(K16).reshape(-1)]
    vf = [int(v) for v in np.asarray(V16).reshape(-1)]
    s.tap("TAPF16", kf + vf, 16)
    if key_grouped:
        s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0, kvu=0)
        g0 = 0
        while g0 < T:
            g1 = min(g0 + G_CFG, T)
            s.emit(f"// key group tokens {g0}..{g1 - 1} (WADDR=base, F-2)")
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["WADDR"], g0)
            for t in range(g0, g1):
                pairs = [decompose_f16(int(b))[:2] for b in K16[t]]
                inject_jobs(s, pairs, 0)           # group member: no WADDR
            if g1 - g0 < G_CFG:
                # D-008 through the tile: tile idle => every f16 beat was
                # ACCEPTED (post-skid §5), the engine is in ST_KACCEPT
                s.emit("// partial key group -> CSR FLUSH (D-008)")
                s.csrp(CSR["STATUS"], 0x1, 0x1)
                s.csrw(CSR["FLUSH"], 0x1)
            s.kvp(KV["STATUS"], 0x1, 0x1)          # group records emitted
            g0 = g1
        s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0, kvu=1)
    else:
        s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0)
        for t in range(T):
            pairs = [decompose_f16(int(b))[:2] for b in K16[t]]
            inject_jobs(s, pairs, 0, waddr=t)
    for t in range(T):
        pairs = [decompose_f16(int(b))[:2] for b in V16[t]]
        inject_jobs(s, pairs, 0, waddr=T + t)
    s.kvp(KV["STATUS"], 0x1, 0x1)
    s.kvp(KV["OCC"], 0xFFFFFFFF, 2 * T)


def mask_load_phase(s, oi):
    """D-027 (S12 stage 4): stage + commit the outlier mask over the engine
    window (tier_sel is already CQ4P from phase_a, so the kv_* ops route to
    engine 2) and prove the LIVE tile INFO_TIER: phase_a checked the maskless
    build truth 0x3; a valid commit must flip it to 0x7 ("INFO_TIER never
    lies", now dynamic)."""
    words = [0, 0, 0, 0]
    for c in oi:
        words[c // 32] |= 1 << (c % 32)
    s.emit("// D-027 mask load: stage oi over CSR, commit, INFO_TIER 0x3->0x7")
    s.kvp(KV["MASK_CTRL"], 0x3, 0x0)             # maskless: !owned, !valid
    for w, val in enumerate(words):
        s.kvw(KV[f"MASK{w}"], val)
        s.kvp(KV[f"MASK{w}"], 0xFFFFFFFF, val)   # staged readback (§1)
    s.kvw(KV["MASK_CTRL"], 0x1)                  # commit at empty
    s.kvp(KV["MASK_CTRL"], 0x3, 0x3)             # owned + valid
    s.kvp(KV["IRQ_STATUS"], 0xC, 0x0)            # no MASK_ERR / MASK_SWAP
    s.csrr(CSR["INFO_TIER"], 0xFFFFFFFF, 0x7)    # tile truth went LIVE


def core_case(name, q, K16, V16, D, tier=TIER_CQ8, oi=(), mask_csr=False):
    T = K16.shape[0]
    assert T <= 128, f"{name}: T={T} beyond the T_ROW_MAX=128 envelope (F-1)"
    grouped = tier != TIER_CQ8
    r = attention_core(np.asarray(q, dtype=np.float64), K16, V16,
                       tier, G=G_CFG if grouped else 128, outlier_idx=oi)
    s = Script(D)
    s.emit(f"// L3 core replay: {name} (T={T} D={D} {tier})")
    phase_a(s, tiers_used=(TIER_CODE[tier],))
    if mask_csr:
        mask_load_phase(s, oi)                   # D-027: mask arrives by CSR
    loader_phase(s)

    store_kv_phase(s, K16, V16, tier)

    s.emit("// inject q row through squant MODE_QUANT (Q7 machinery)")
    s.route(rdst=1, asrc=1)
    q16 = to16(np.asarray(q, dtype=np.float64))
    qpairs = [decompose_f16(int(b))[:2] for b in q16]
    inject_jobs(s, qpairs, 1)
    s.ess(r.s_q, 1)
    s.aj(0, 1, 0, 1, s.BPR, 0)                     # LOAD q8 -> bank1

    score_phase(s, r, T)
    pv_phase(s, r, T)
    final_phase(s, T)
    return s, r


# ── full-chain random cases (attention_fx) ───────────────────────────────────

def full_case(name, rng, T, D):
    BPR = D // 8
    assert T + 1 <= 31, "phase-B staging bound (STAGE_R_MAX=31)"
    X = rng.normal(0.0, 1.0, (T + 1, D))
    Wq8 = rng.integers(-127, 128, (D, D), dtype=np.int64)
    Wk8 = rng.integers(-127, 128, (D, D), dtype=np.int64)
    Wv8 = rng.integers(-127, 128, (D, D), dtype=np.int64)
    gamma = rng.integers(4096, 12289, D, dtype=np.int64)
    S_W = 2.0 ** -6                                # power-of-two (S-2 exact)
    r = attention_fx(X, Wq8, Wk8, Wv8, S_W, S_W, S_W, gamma,
                     tier=TIER_CQ8, G=128)
    ref = attention_ref_full(X, Wq8, Wk8, Wv8, S_W, S_W, S_W, gamma)
    vscale = float(np.max(np.abs(ref.V_ref)))
    e2e = float(np.max(np.abs(r.out_hat - ref.y_ref))) / vscale
    for nm, acc in (("k", r.acc_k), ("v", r.acc_v), ("q", r.acc_q),
                    ("s", r.acc_s), ("o", r.acc_o)):
        assert np.max(np.abs(acc)) < 2 ** 24, f"{name}: acc_{nm} exceeds C1"

    s_h = f16_bits_to_f64(r.s_h)
    s = Script(D)
    s.emit(f"// L3 full chain: {name} (T={T} D={D} CQ-8) "
           f"e2e={100 * e2e:.2f}% of value scale")
    phase_a(s)

    s.emit("// phase B: hidden pipeline")
    s.route(fsrc=0, fdst=0, asrc=0)
    s.aj(0, 0, 0, T + 1, BPR, 0)
    s.fjob(T + 1)
    for t in range(T + 1):
        s.xrow(r.x8[t])
        s.grow(gamma)
        s.efs(r.s_h[t], t == T)

    s.emit("// phase C: K/V projections + S-2 + KVQ store")
    s.route(rdst=1)
    kf = [int(v) for v in np.asarray(r.K_f16).reshape(-1)]
    vf = [int(v) for v in np.asarray(r.V_f16).reshape(-1)]
    s.tap("TAPF16", kf + vf, 16)
    for W, base in ((Wk8, 0), (Wv8, T)):
        for t in range(T):
            comp = f32_bits_exact(float(s_h[t]) * S_W, f"S-2 t={t}")
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["WADDR"], base + t)
            s.qjob(0, D)
            for _ in range(D):
                s.qs(comp)
            s.ljob(BPR, 8)
            for j in range(BPR):
                s.aj(1, 0, 0, 1, BPR, t)
                s.desc(desc_words(OP_GEMM_WS, 1, D, 8))
                s.wbeats(wgt_beats_ws(W, j))
        if base == 0:
            s.emit("// mid-sequence illegal descriptor (K=0), D-006 reject")
            s.desc(desc_words(OP_GEMM_WS, 1, 0, 1))
            s.csrp(CSR["STATUS"], 0x2, 0x2)
            s.csrw(CSR["STATUS"], 0x2)
            s.csrr(CSR["STATUS"], 0x2, 0x0)
    s.kvp(KV["STATUS"], 0x1, 0x1)
    s.kvp(KV["OCC"], 0xFFFFFFFF, 2 * T)

    s.emit("// phase D: q projection + Q7")
    s.route(rdst=1, asrc=1)
    comp_q = f32_bits_exact(float(s_h[T]) * S_W, "Q7 comp")
    s.qjob(1, D)
    for _ in range(D):
        s.qs(comp_q)
    s.ljob(BPR, 8)
    for j in range(BPR):
        s.aj(1, 0, 0, 1, BPR, T)
        s.desc(desc_words(OP_GEMM_WS, 1, D, 8))
        s.wbeats(wgt_beats_ws(Wq8, j))
    s.ess(r.s_q, 1)
    s.aj(0, 1, 0, 1, BPR, 0)

    score_phase(s, r, T)
    pv_phase(s, r, T)
    final_phase(s, T)
    return s, r, e2e


# ── D-022 actuation: TIP-driven per-block tier map (auto mode) ───────────────

def tip_auto_case(name, D=64):
    """Two-pass decode demo of D-022 actuation, all values bit-exact:

    pass 1  store K/V at CQ-8 (host tier), then PROFILE: per 8-token block,
            one Q·K̂ᵀ score tile (sjob(8), rt_tip_blk=b) streams through the
            REAL score-dequant -> fork -> {ASU, TIP}; TIP's decisions (blk,
            tier, fp16 — checked bit-exact vs the verif/tip golden replica)
            write apex_top's auto_tier map; the profiling softmax/P-requant
            drains are ALSO golden-checked (8-row online softmax slices).
            The per-block map is then read back through IMPORTANCE_BASE.
    pass 2  TIER_CTRL.tip_override=1: the OUTLIER-BEARING blocks (planted
            hot channels {5,50}) restore at CQ-4+ (grouped keys + mask,
            engine 2), the benign blocks at CQ-4 (engine 1) — the tier is
            DRIVEN by TIP, per block, in hardware. The full attention then
            replays with per-chunk rt_tip_blk routing, checked bit-exact
            against attention_core(tier_map=...) — the golden extension
            whose uniform/composition properties are gated in
            golden/tests/test_attention.py section E."""
    T = 32
    NB = T // 8
    BEN, OUT = (0, 2), (1, 3)                    # benign / outlier blocks
    rng = np.random.default_rng(SEED + 0xD022)
    K = rng.normal(0.0, 0.05, (T, D))
    for b in OUT:
        K[8 * b:8 * b + 8, 5] = rng.normal(0, 16.0, 8)
        K[8 * b:8 * b + 8, 50] = rng.normal(0, 16.0, 8)
    V = rng.normal(0.0, 1.0, (T, D))
    q = rng.normal(0.0, 1.0, D)
    K16, V16 = to16(K), to16(V)
    q = f16_bits_to_f64(to16(q)).reshape(D)

    # pass-1 golden: CQ-8 profiling chain
    r8 = attention_core(q, K16, V16, TIER_CQ8, G=128, outlier_idx=[])
    s_v8val = f16_bits_to_f64(np.asarray(r8.s_v, dtype=np.uint16))
    s_qv = float(f16_bits_to_f64(np.array([r8.s_q]))[0])
    s_kv = f16_bits_to_f64(np.asarray(r8.s_k, dtype=np.uint16))
    comp64 = s_qv * s_kv * (float(1 << SCORE_FRAC) / math.sqrt(float(D)))
    comp32 = comp64.astype(np.float32).view(np.uint32)

    # TIP golden replica: decisions -> importance accs -> thresholds -> map
    fp16s, accs = [], []
    for b in range(NB):
        sc = [int(v) for v in r8.score_fx[8 * b:8 * b + 8]]
        f, mx = tipg.tip_decide_golden(sc, 1, 8)   # THRESH = 1 (reset value)
        fp16s.append(f)
        accs.append(tipg.imp_update_golden(0, f, mx))
    assert max(accs[b] for b in BEN) < min(accs[b] for b in OUT), \
        f"no TIP importance separation: accs={accs}"
    lo = (max(accs[b] for b in BEN) + min(accs[b] for b in OUT) + 1) // 2
    hi = 0xFFFF                                    # accs << hi: never CQ-8
    tiers = [tipg.tier_golden(a, hi, lo) for a in accs]
    assert tiers == [1, 2, 1, 2], f"unexpected TIP tier map: {tiers}"
    tmap = []
    for b in range(NB):
        tmap += [TIER_CQ4 if tiers[b] == 1 else TIER_CQ4P] * 8

    # pass-2 golden: the mixed-tier attention (D-022 load-bearing run)
    r = attention_core(q, K16, V16, TIER_CQ4, G=G_CFG, outlier_idx=OI_B64,
                       tier_map=tmap)
    assert int(r.s_q) == int(r8.s_q) and np.array_equal(r.q8, r8.q8)

    s = Script(D)
    s.emit(f"// L3 D-022 TIP-auto: {name} (T={T} D={D} tiers={tiers})")
    phase_a(s, tiers_used=(0, 1, 2))
    s.csrw(CSR["THRESH"], 1)                       # explicit (== reset value)
    s.imp(hi, lo)                                  # importance thresholds
    loader_phase(s)

    s.emit("// pass 1: CQ-8 store (host tier 0) for profiling")
    store_kv_phase(s, K16, V16, TIER_CQ8)
    s.emit("// q inject (shared by both passes; Q7 machinery)")
    s.route(rdst=1, asrc=1)
    qpairs = [decompose_f16(int(x))[:2] for x in to16(q)]
    inject_jobs(s, qpairs, 1)
    s.ess(r8.s_q, 1)
    s.aj(0, 1, 0, 1, s.BPR, 0)                     # LOAD q8 -> bank1

    s.emit("// pass 1 profiling: per-block score tiles -> TIP decisions")
    for b in range(NB):
        sc = [int(v) for v in r8.score_fx[8 * b:8 * b + 8]]
        p_prof = np.asarray(online_softmax_fx(sc, SCORE_FRAC), dtype=np.int64)
        c_prof = (p_prof.astype(np.float64) / (1 << 15)) \
            * s_v8val[8 * b:8 * b + 8]
        c8p, scp = quant_rows_i8(c_prof[None, :])
        _ = c8p                                    # codes parked, not read
        s.emit(f"// profile block {b}: 8-score tile, decision LOAD-BEARING")
        s.route(rdst=2, fsrc=1, fdst=1, asrc=1, wsrc=1, qsrc=1, blk=b)
        s.tap("TAPSC", sc, 32)
        s.tap("TAPPR", [int(v) for v in p_prof], 16)
        s.sjob(8)
        for t in range(8 * b, 8 * b + 8):
            s.cs(int(comp32[t]))
        s.qjob(1, 8)                               # drain the 8-row softmax
        for t in range(8 * b, 8 * b + 8):
            s.qs(f32_bits_exact(float(s_v8val[t]) * 2.0 ** -15,
                                f"prof S-4 t={t}"))
        s.wj(0, 0, 0, 8, s.BPR, 0)                 # wstage LOAD k8 block
        s.fjob(8)
        for t in range(8 * b, 8 * b + 8):
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["RADDR"], t)
            s.efs(r8.s_k[t], t == 8 * b + 7)
        s.wj(1, 0, 1, 8, s.BPR, 0)                 # EMIT PAT_T
        s.aj(1, 1, 0, 1, s.BPR, 0)                 # act: q8
        s.desc(desc_words(OP_GEMM_OS, 1, D, 8, mode_os=1))
        s.etipt(b, tiers[b], fp16s[b])             # FULL decision check
        s.ess(int(scp[0]), 1)
        s.aj(0, 0, 0, 1, 1, 0)                     # park c8 (bank0 scratch)

    s.emit("// IMPORTANCE_BASE window: the per-block map is IN the tile")
    for b in range(NB):
        s.csrw(CSR["IMP"], b)
        s.csrr(CSR["IMP"], 0x0003FFFF, (tiers[b] << 16) | accs[b])

    s.emit("// AUTO MODE ON: TIER_CTRL.tip_override — TIP drives the tier")
    s.csrw(CSR["TIER_CTRL"], 1 << 8)
    s.csrr(CSR["TIER_CTRL"], 0xFFFFFFFF, 1 << 8)
    # routing proof: kv_* window follows auto_tier[rt_tip_blk]
    s.route(rdst=1, blk=OUT[0])
    s.kvp(KV["INFO_TIER"], 0xFFFFFFFF, 2)          # outlier block -> CQ-4+
    s.route(rdst=1, blk=BEN[0])
    s.kvp(KV["INFO_TIER"], 0xFFFFFFFF, 1)          # benign block  -> CQ-4

    s.emit("// pass 2: TIP-DRIVEN per-block tier stores (D-022 actuation)")
    loader_phase(s)                                # profiling parked over it
    kf = [int(v) for v in np.asarray(K16).reshape(-1)]
    vf = [int(v) for v in np.asarray(V16).reshape(-1)]
    s.tap("TAPF16", kf + vf, 16)
    s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0, kvu=0, blk=0)
    for b in range(NB):
        s.emit(f"// block {b} keys -> engine[auto_tier[{b}]={tiers[b]}], "
               "grouped + FLUSH")
        if b > 0:
            s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0, kvu=0, blk=b)
        s.kvp(KV["STATUS"], 0x1, 0x1)
        s.kvw(KV["WADDR"], 8 * b)
        for t in range(8 * b, 8 * b + 8):
            pairs = [decompose_f16(int(x))[:2] for x in K16[t]]
            inject_jobs(s, pairs, 0)               # group member: no WADDR
        s.csrp(CSR["STATUS"], 0x1, 0x1)            # beats all accepted
        s.csrw(CSR["FLUSH"], 0x1)                  # partial group (8 < G)
        s.kvp(KV["STATUS"], 0x1, 0x1)
    for b in range(NB):
        s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0, kvu=1, blk=b)
        for t in range(8 * b, 8 * b + 8):
            pairs = [decompose_f16(int(x))[:2] for x in V16[t]]
            inject_jobs(s, pairs, 0, waddr=T + t)
    s.emit("// per-engine occupancy: 2 blocks x (8 K + 8 V) per tier")
    s.route(rdst=1, blk=BEN[0])
    s.kvp(KV["OCC"], 0xFFFFFFFF, 32)               # engine 1 (CQ-4)
    s.route(rdst=1, blk=OUT[0])
    s.kvp(KV["OCC"], 0xFFFFFFFF, 32)               # engine 2 (CQ-4+)

    s.emit("// pass 2 attention: mixed-tier readback, bit-exact vs tier_map")
    blkmap = list(range(NB))
    score_phase(s, r, T, blkmap=blkmap)
    pv_phase(s, r, T, blkmap=blkmap)
    final_phase(s, T)
    return s, r


# ── case assembly ─────────────────────────────────────────────────────────────

def main(build_dir: str) -> None:
    out = Path(build_dir) / "cases"
    out.mkdir(parents=True, exist_ok=True)
    # b64 build carries the CQ-4+ outlier mask ROM (channels {5,50} — the
    # adv recipe's planted-outlier channels); b128 has no mask, so its
    # INFO_TIER truthfully reads 0x3 (CQ-4+ degenerates without a mask)
    (Path(build_dir) / "mask_l3_b64.hex").write_text(
        "\n".join("01" if c in OI_B64 else "00" for c in range(64)) + "\n")
    manifest = {"builds": {"b64": {"CFG_D": 64, "G_CFG": G_CFG, "DEPTH": 256,
                                   "OUTK": 2, "MASK": list(OI_B64)},
                           "b128": {"CFG_D": 128, "G_CFG": G_CFG,
                                    "DEPTH": 256, "OUTK": 0}},
                "infeasible": [
                    {"case": "T > 128 attention rows", "why": "F-1 envelope: "
                     "apex_top T_ROW_MAX=128 (one score-dequant job == one "
                     "last-framed ASU row). Raising it further is sizing + "
                     "re-verification work (ASU SM_ROW_MAX=1024 headroom "
                     "exists); T<=128 is the verified v0.2 envelope."},
                    {"case": "KVQ eviction paths at tile level", "why":
                     "F-4: DEPTH=256 holds the full 2*T=256 records at "
                     "T=128; §8's eviction obligation remains block-level "
                     "only (verif/kvq sram_full)."},
                    {"case": "CQ-4+ at D=128 through the tile", "why":
                     "F-2 residual: MASK_FILE is one ROM per BUILD; the "
                     "b128 build ships without a mask (INFO_TIER=0x3, "
                     "CQ-4+ degenerates to CQ-4 there). D=128 CQ-4+ stays "
                     "covered at L2 chain (b) through the real kvq_engine "
                     "with its own calib mask."},
                    {"case": "TIP decision (and thus AUTO tier update) for "
                     "score tiles longer than 8", "why": "F-3 residual: "
                     "the TIP tap is BLOCK_M=1 x BLOCK_N=8 — a T>8 "
                     "last-framed tile frame-errors in TIP (sticky now "
                     "W1C-clearable via ERR_STICKY 0x58); auto-mode "
                     "profiling therefore uses per-block 8-score tiles "
                     "(tip_auto_mixed), never the full row."},
                    {"case": "mid-run tier switching (sub-transaction "
                     "granularity)", "why": "D-024 contract: the tier "
                     "select is quasi-static per KVQ transaction run — "
                     "per token (values), per group (keys), per record "
                     "(reads), per block (auto mode). The engines' TIER "
                     "stays a synthesis parameter; runtime control is by "
                     "BANKING verified engines, honestly documented."},
                ],
                "cases": []}

    def save(build, name, s, kind, T, D, wd=8_000_000, expect="pass",
             fail_sig=None):
        p = out / f"{name}.txt"
        p.write_text("\n".join(s.lines) + "\n")
        manifest["cases"].append({"name": name, "build": build,
                                  "kind": kind, "T": T, "D": D,
                                  "checks": s.n_expect, "watchdog": wd,
                                  "expect": expect, "fail_sig": fail_sig})
        print(f"case {name}: build={build} T={T} D={D} "
              f"checks={s.n_expect} expect={expect}")

    # ── named cases (CQ-8 through the REAL tile) ────────────────────────────
    arng = np.random.default_rng(0xA0D17)          # frozen adv recipe seed
    Da = 64
    qa = f16_bits_to_f64(to16(arng.normal(0, 1, Da))).reshape(Da)
    K1 = arng.normal(0, 1, (1, Da))
    K1 *= 2.0 / np.max(np.abs(K1))
    V1 = arng.normal(0, 1, (1, Da))
    V1 *= 2.0 / np.max(np.abs(V1))
    s, _ = core_case("adv_T1", qa, to16(K1), to16(V1), Da)
    save("b64", "adv_T1", s, "named-core", 1, Da)

    row_k = arng.normal(0, 1, Da)                  # keep the frozen draw order
    row_k *= 2.0 / np.max(np.abs(row_k))
    _ = np.tile(row_k, (64, 1))
    Vt = arng.normal(0, 1, (64, Da))
    Vt *= 2.0 / np.max(np.abs(Vt))
    To = 128
    Ko = arng.normal(0, 0.02, (To, Da))
    Ko[:, 5] = arng.normal(0, 20.0, To)
    Ko[:, 50] = arng.normal(0, 20.0, To)
    Vo = arng.normal(0, 1, (To, Da))
    Vo *= 2.0 / np.max(np.abs(Vo))
    qo = arng.normal(0, 1, Da)
    qo *= 1.0 / np.max(np.abs(qo))
    qo = f16_bits_to_f64(to16(qo)).reshape(Da)
    # F-1 kept-passing regression: full T=128 (was adv_outlier1000_T64sub;
    # same frozen tensors, no [:64] truncation)
    s, _ = core_case("adv_outlier1000", qo, to16(Ko), to16(Vo), Da)
    save("b64", "adv_outlier1000", s, "named-core", To, Da, wd=80_000_000)

    # ── F-2 CLOSED: CQ-4+ THROUGH THE TILE (kept-passing regressions) ──────
    # adv_T1 at CQ-4+: T=1 partial key group -> D-008 CSR FLUSH through the
    # tile; the {5,50} build mask is the static calibrated ROM.
    s, _ = core_case("adv_T1_cq4p", qa, to16(K1), to16(V1), Da,
                     tier=TIER_CQ4P, oi=OI_B64)
    save("b64", "adv_T1_cq4p", s, "named-core-cq4p", 1, Da)

    # adv_outlier1000 at CQ-4+: the planted-outlier tensors on their OWN
    # mask channels, T=128 = 8 FULL key groups at G=16 (no flush), grouped
    # per-channel INT4 keys + fp16 outlier lane, full attention bit-exact.
    s, _ = core_case("adv_outlier1000_cq4p", qo, to16(Ko), to16(Vo), Da,
                     tier=TIER_CQ4P, oi=OI_B64)
    save("b64", "adv_outlier1000_cq4p", s, "named-core-cq4p", To, Da,
         wd=80_000_000)

    # ── D-022 actuation: TIP-auto mixed-tier decode (kept-passing) ────────
    s, _ = tip_auto_case("tip_auto_mixed", 64)
    save("b64", "tip_auto_mixed", s, "tip-auto", 32, 64, wd=40_000_000)

    Vn = arng.normal(0, 1, (16, Da))
    Vn *= 2.0 / np.max(np.abs(Vn))
    Z16 = to16(np.zeros((16, Da)))
    _ = to16(np.zeros((16, Da)))                   # zeroK slot (draw order)
    s, _ = core_case("adv_zeroV", qa, to16(Vn), Z16, Da)
    save("b64", "adv_zeroV", s, "named-core", 16, Da, wd=16_000_000)

    # F-1 kept-passing regressions: the vendored calib tensors at FULL
    # length (were *_T64sub prefix substitutes)
    for nm, fn, Dc in (("calib_d64_T128", "d64_T128_G64__CQ4.npz", 64),
                       ("calib_d64_T70", "d64_T70_G64__CQ4.npz", 64),
                       ("calib_d128_T100", "d128_T100_G128__CQ4.npz", 128)):
        z = np.load(VEC / fn)
        K16 = np.ascontiguousarray(z["input_K"]).view(np.uint16)
        V16 = np.ascontiguousarray(z["input_V"]).view(np.uint16)
        assert K16.shape[1] == Dc
        qc = f16_bits_to_f64(K16[-1]).reshape(Dc)
        Tc = K16.shape[0]
        s, _ = core_case(nm, qc, K16, V16, Dc)
        save("b64" if Dc == 64 else "b128", nm, s, "named-core", Tc, Dc,
             wd=80_000_000)

    # F-5 fix regression (kept-PASSING): the exact case that wedged the
    # pre-fix tile ('drive_g stall' at the phase-B gamma stream, F-5a) now
    # runs the full D=128 chain bit-exact. Regressing to the wedge fails
    # this case loudly (watchdog / stall fatal).
    rngb = np.random.default_rng(SEED + 999)
    s, _, _ = full_case("bug_d128_stagebuf_nb", rngb, 4, 128)
    save("b128", "bug_d128_stagebuf_nb", s, "f5-fix-regression", 4, 128,
         wd=16_000_000)

    # ── >=15 seeded random full-chain jobs, T straddling G_CFG=16 (F-2
    #    caveat: CQ-8 never key-groups; straddle+flush proper is covered at
    #    L2 chain b) ────────────────────────────────────────────────────────
    for i, T in enumerate((1, 2, 3, 5, 6, 8, 9, 10, 12, 16, 17, 20, 24,
                           28, 30)):
        rng = np.random.default_rng(SEED + i)
        name = f"rand_d64_T{T}"
        s, _, e2e = full_case(name, rng, T, 64)
        save("b64", name, s, "random-full", T, 64,
             wd=8_000_000 + T * 400_000)

    # D=128 random full-chain cases (F-5 closure breadth: whole hidden
    # pipeline + PAT_D blocks 8..15 with random data, T straddling 8)
    for i, T in enumerate((5, 12)):
        rng = np.random.default_rng(SEED + 100 + i)
        name = f"rand_d128_T{T}"
        s, _, e2e = full_case(name, rng, T, 128)
        save("b128", name, s, "random-full", T, 128,
             wd=16_000_000 + T * 800_000)

    # ── S12/D-027: CQ-4+ at D=128 via the CSR-LOADED mask (retires the F-2
    #    residual — the b128 build ships maskless; the mask arrives at
    #    runtime). Planted-outlier recipe on channels {5,100} (100 proves
    #    MASK3-window-free bits AND a channel beyond D=64's words); T=64 =
    #    4 full key groups at G_CFG=16. Own frozen seed, appended LAST so
    #    every pre-existing rng stream is untouched.
    mrng = np.random.default_rng(0xD027)
    Dm, Tm = 128, 64
    OI_D128 = (5, 100)
    Km = mrng.normal(0, 0.02, (Tm, Dm))
    Km[:, OI_D128[0]] = mrng.normal(0, 20.0, Tm)
    Km[:, OI_D128[1]] = mrng.normal(0, 20.0, Tm)
    Vm = mrng.normal(0, 1, (Tm, Dm))
    Vm *= 2.0 / np.max(np.abs(Vm))
    qm = mrng.normal(0, 1, Dm)
    qm *= 1.0 / np.max(np.abs(qm))
    qm = f16_bits_to_f64(to16(qm)).reshape(Dm)
    s, _ = core_case("adv_outlier_d128_cq4p", qm, to16(Km), to16(Vm), Dm,
                     tier=TIER_CQ4P, oi=OI_D128, mask_csr=True)
    save("b128", "adv_outlier_d128_cq4p", s, "named-core-cq4p-maskcsr",
         Tm, Dm, wd=80_000_000)

    (Path(build_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    for e in manifest["infeasible"]:
        print(f"INFEASIBLE: {e['case']} -- {e['why']}")
    n64 = sum(1 for c in manifest["cases"] if c["build"] == "b64")
    n128 = sum(1 for c in manifest["cases"] if c["build"] == "b128")
    print(f"L3 cases: {len(manifest['cases'])} total ({n64} b64, {n128} b128), "
          f"{sum(c['checks'] for c in manifest['cases'])} scripted checks")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
