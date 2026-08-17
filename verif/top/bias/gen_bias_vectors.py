#!/usr/bin/env python3
"""gen_bias_vectors.py — IC-BIAS (gap B) vector generator.

Two artifacts, one arbiter:

  build/pbias_unit.vec   apex_proj_bias UNIT vectors — (v, composite, bias)
                         triples with the golden fp16 expectation and the
                         exact-or-refused verdict, plus a bias == +0 arm the
                         TB cross-checks against the REAL apex_scale_quant
                         MODE_F16 instance running the same stimulus.

  build/bias_tile.ops    the TILE op stream — the l4_bias case's q/K/V
                         projections driven through a real apex_top
                         (h8 -> GEMM_WS -> serializer -> apex_proj_bias ->
                         S-2 f16 seam) and checked beat-for-beat at the
                         dbg_f16_* tap.

THE ARBITER is decoder_layer_fx(bus=BUS_ON) — the D-030 owner-signed bus
composition (LEVEL_C_INTEGRATION.md §9.1). Golden is never adjusted; every
expectation below is either a LayerFx field or re-derived with PUBLIC golden
functions and asserted against a LayerFx field before any RTL runs.

Where the tap expectations come from (transformer.py):
  K : r.K_real is ALREADY the narrowed value under BUS_ON (:522 `_f16`
      applied AFTER the bias add at :512) -> expect f64_to_f16_bits(r.K_real)
  V : NOT narrowed in the record; its single S-2 narrowing is the attention
      core input at :541 -> expect f64_to_f16_bits(r.V_real)
  q : same as K (:521) -> expect f64_to_f16_bits(r.q_real)
All three are exactly "one RNE of (acc*composite + bias)", which is what the
tile's apex_proj_bias computes.

GEOMETRY NOTE (why all six l4_bias rows are drivable where §3c-3's O3'-1 was
1-of-6): the bias+narrow seam is PER ELEMENT — no head pairing — so it is run
with l_rope_en = 0 and GAP D (CFG_D overloaded as head_dim AND D_model) does
not bind. K/V rows are kv_dim = 128 = CFG_D and the q row is D_model = 128.
"""
from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l4"))
from l4_cases import TIER, build_case, tf                      # noqa: E402

from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

OP_GEMM_WS = 0x01
CSR = {"CTRL": 0x00, "STATUS": 0x04, "TIER_CTRL": 0x20, "ERR_STICKY": 0x58}
KV = {"CTRL": 0x00, "STATUS": 0x04, "WADDR": 0x28}
LAYER = {"CTRL": 0x70, "PTR": 0x74, "DATA": 0x78, "STATUS": 0x80}
L_BIAS_EN = 1 << 18                     # LAYER_CTRL[18] (IB_LAYER.md §3d)
L_BIAS_BANK = 3                         # LAYER_PTR[29:28] = 3

# ── the apex_proj_bias window, mirrored (RTL: PB_SX_MAX/PB_SB_MAX/PB_SPAN_MAX)
PB_SX_MAX, PB_SB_MAX, PB_SPAN_MAX = 21, 16, 53


# ═══════════════════════════ golden-side element model ══════════════════════
def comp_bits(x: float, what: str) -> int:
    """fp32 bits of a composite, asserting the apex_scale_quant C2 contract."""
    b = np.float64(x).astype(np.float32)
    assert float(np.float64(b)) == float(x), f"{what}: not fp32-exact: {x!r}"
    bits = int(b.view(np.uint32))
    assert bits & 0x1FFF == 0, f"{what}: not fp16-grade: {bits:08x}"
    assert 0 < ((bits >> 23) & 0xFF) < 255 and bits >> 31 == 0, \
        f"{what}: not positive normal: {bits:08x}"
    return bits


def exact_elem(v: int, cbits: int, b16: int):
    """(fp16 bits, window_ok) for ONE biased element, computed the way golden
    does and verified EXACT with Fractions.

    Returns the golden result of  f64_to_f16_bits(float64(v)*comp + b)  plus
    the RTL's exact-or-refused verdict, so the TB can require refusal exactly
    where golden's float64 add would have rounded (or the alignment window
    would have dropped a bit) and bit-exactness everywhere else.
    """
    ec = (cbits >> 23) & 0xFF
    m11 = 1024 + ((cbits >> 13) & 0x3FF)
    mag = abs(v)
    assert mag <= 1 << 24, "C1"
    p35 = mag * m11
    ex = ec - 137
    if ex >= -24:
        g, sx, sb = -24, ex + 24, 0
    else:
        g, sx, sb = ex, 0, -24 - ex
    okw = sx <= PB_SX_MAX and sb <= PB_SB_MAX
    bv = int(f16_bits_to_f64(np.array([b16], dtype=np.uint16))[0]
             * (1 << 24)) if b16 & 0x7FFF else 0
    # exact integer sum on the 2^g grid
    acc = (-p35 if v < 0 else p35) * (1 << sx) + bv * (1 << sb)
    span_ok = True
    if acc:
        a = abs(acc)
        hi = a.bit_length() - 1
        lo = (a & -a).bit_length() - 1
        span_ok = (hi - lo) < PB_SPAN_MAX
    ok = okw and span_ok
    # golden: exact f64 product, f64 add, ONE RNE to fp16
    xf = np.float64(v) * np.float64(np.float32(np.uint32(cbits).view(np.float32)))
    assert Fraction(float(xf)) == Fraction(v) * Fraction(
        float(np.uint32(cbits).view(np.float32))), "product not f64-exact"
    bf = float(f16_bits_to_f64(np.array([b16], dtype=np.uint16))[0])
    yf = float(xf) + bf
    if ok:   # inside the window golden's add is exact — assert it, never assume
        assert Fraction(yf) == Fraction(float(xf)) + Fraction(bf), \
            f"window says exact but f64 add rounded: v={v} c={cbits:08x} b={b16:04x}"
    return int(f64_to_f16_bits(np.array([yf]))[0]), ok


# ═══════════════════════════ unit vectors ═══════════════════════════════════
def unit_vectors(out: Path) -> dict:
    """Emit the apex_proj_bias unit stimulus. Line format:

        JOB  <cols> <zero_bias:0|1>       start a job (zero_bias arms the
                                          apex_scale_quant equivalence check)
        E    <v32> <c32> <b16> <y16> <ok> one element
        END                               finish

    The TB drives apex_proj_bias AND apex_scale_quant MODE_F16 with the SAME
    (v, c) stream; on a zero_bias job it additionally requires the two blocks'
    fp16 outputs to be IDENTICAL beat for beat.
    """
    rng = np.random.default_rng(0x1CB1A5)
    L, n_el, n_job, n_ref, n_zero = [], 0, 0, 0, 0
    n_ref_align, n_ref_span = 0, 0

    def emit_job(elems, zero_bias, note, scale_err=0):
        nonlocal n_el, n_job, n_ref, n_zero, n_ref_align, n_ref_span
        L.append(f"// {note}")
        L.append(f"JOB {len(elems):x} {1 if zero_bias else 0} {scale_err}")
        for (v, c, b) in elems:
            # the block decodes a sideband as its 11-bit truncation (the
            # apex_scale_quant C2 encoding), so the expectation uses the same
            # truncation — and the zero-bias arm cross-checks that against the
            # real apex_scale_quant, so this is never a self-consistent oracle
            y, ok = exact_elem(v, c & ~0x1FFF, b)
            L.append(f"E {v & 0xFFFFFFFF:08x} {c:08x} {b:04x} {y:04x} {int(ok)}")
            n_el += 1
            if not ok:
                n_ref += 1
                ex = ((c >> 23) & 0xFF) - 137
                if (max(ex + 24, 0) > PB_SX_MAX) or (max(-24 - ex, 0) > PB_SB_MAX):
                    n_ref_align += 1
                else:
                    n_ref_span += 1
        n_job += 1
        if zero_bias:
            n_zero += 1

    def rand_comp(lo=-30, hi=-4):
        e = int(rng.integers(lo + 127, hi + 127 + 1))
        m = int(rng.integers(0, 1024))
        return (e << 23) | (m << 13)

    def rand_bias(scale=0.05):
        return int(f64_to_f16_bits(np.array([rng.normal(0, scale)]))[0])

    # R1 — the operating regime: composites and biases at L4/7B magnitudes
    for k in range(6):
        el = [(int(rng.integers(-2_064_512, 2_064_513)), rand_comp(-24, -14),
               rand_bias()) for _ in range(128)]
        emit_job(el, False, f"R1.{k} operating regime (|acc|<=128*127*127)")

    # R2 — bias == +0 on the same regime: MUST equal apex_scale_quant MODE_F16
    for k in range(4):
        el = [(int(rng.integers(-2_064_512, 2_064_513)), rand_comp(-24, -14),
               0x0000) for _ in range(128)]
        emit_job(el, True, f"R2.{k} bias=+0 equivalence vs apex_scale_quant")

    # R3 — directed corners (signed zeros, subnormal + max fp16 bias, v==0,
    #      C1 clamp boundary, both window edges, RNE ties by construction)
    corner = []
    for b in (0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00):
        corner.append((0, rand_comp(-24, -14), b))
        corner.append((1, rand_comp(-24, -14), b))
        corner.append((-1, rand_comp(-24, -14), b))
        corner.append((1 << 24, (100 << 23), b))          # C1 boundary
        corner.append((-(1 << 24), (100 << 23), b))
    # exact half-way ties: v*c = 2^-11 * (2k+1) against a bias that lands the
    # sum exactly between two fp16s (keep parity in BOTH directions)
    for k in (1, 3, 5, 7):
        corner.append((k, (127 - 11) << 23, 0x3C00))      # 1.0 + k*2^-11
        corner.append((k, (127 - 11) << 23, 0xBC00))      # -1.0 + k*2^-11
    emit_job(corner, False, "R3 directed corners + constructed RNE ties")

    # R4 — window edges and refusals (exact-or-refused must fire, never round)
    edge = []
    for ex in (-42, -41, -40, -39, -5, -4, -3, -2):       # window = [-40,-3]
        edge.append((123_456, ((ex + 137) << 23) | (0x155 << 13), rand_bias()))
        edge.append((-7, ((ex + 137) << 23) | (0x2AA << 13), 0x0000))
    # a span > 53 bits inside the alignment window: tiny grid + big bias
    edge.append(((1 << 24) - 1, (97 << 23) | (0x3FF << 13), 0x7BFF))
    emit_job(edge, False, "R4 window edges + span refusals")

    # R5 — C2 contract monitor: NON-graded sideband significands. The block
    # decodes them exactly as apex_scale_quant does (11-bit truncation) and
    # raises scale_error; bias = +0 so the co-simulated apex_scale_quant is
    # the reference for the values.
    c2 = [(int(rng.integers(-2_064_512, 2_064_513)),
           rand_comp(-24, -14) | int(rng.integers(1, 0x2000)), 0x0000)
          for _ in range(64)]
    emit_job(c2, True, "R5 C2 monitor: non-graded sideband -> scale_error", 1)

    L.append("END")
    (out / "pbias_unit.vec").write_text("\n".join(L) + "\n")
    return {"unit_jobs": n_job, "unit_elems": n_el, "unit_refused": n_ref,
            "unit_refused_alignment": n_ref_align,
            "unit_refused_span53": n_ref_span,
            "unit_zero_bias_jobs": n_zero}


# ═══════════════════════════ tile op stream ═════════════════════════════════
def desc_words(opcode, m, k, n):
    v = (opcode & 0xFF) | (m << 8) | (k << 20) | (n << 32)
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


class Ops:
    def __init__(self, D):
        self.L, self.D, self.BPR, self.n_chk = [], D, D // 8, 0

    def e(self, *t):
        self.L.append(" ".join(str(x) for x in t))

    def csrw(self, a, d):  self.e("CSRW", f"{a:02x}", f"{d:08x}")
    def csrp(self, a, m, x): self.e("CSRP", f"{a:02x}", f"{m:08x}", f"{x:08x}")
    def kvw(self, a, d):   self.e("KVW", f"{a:02x}", f"{d:08x}")
    def kvp(self, a, m, x): self.e("KVP", f"{a:02x}", f"{m:08x}", f"{x:08x}")

    def csrr(self, a, m, x):
        self.e("CSRR", f"{a:02x}", f"{m:08x}", f"{x:08x}")
        self.n_chk += 1

    def idle(self):        self.csrp(CSR["STATUS"], 0x1, 0x1)
    def route(self, rdst=0): self.e("ROUTE", f"{0x80 | (rdst << 4):04x}")

    def stage_bias(self, vec):
        """LAYER_PTR bank 3 + LAYER_DATA: one fp16 per write, auto-increment."""
        self.csrp(LAYER["STATUS"], 0x10, 0x0)        # bias unit quiescent
        self.csrw(LAYER["PTR"], (L_BIAS_BANK << 28) | 0)
        for b in vec:
            self.csrw(LAYER["DATA"], int(b) & 0xFFFF)

    def tap(self, vals):
        self.e("TAPF16", f"{len(vals):x}")
        for v in vals:
            self.e(f"{int(v) & 0xFFFF:04x}")
        self.n_chk += len(vals)

    def proj_row(self, W, comp, waddr):
        """One projection row: KVQ record address, S-2 job + composites, the
        serializer job, then the BPR weight column-blocks (L3 phase-C order)."""
        self.kvp(KV["STATUS"], 0x1, 0x1)
        self.kvw(KV["WADDR"], waddr)
        self.e("QJOB", 0, f"{self.D:x}")
        for _ in range(self.D):
            self.e("QS", f"{comp:08x}")
        self.e("LJOB", f"{self.BPR:x}", "8")
        for j in range(self.BPR):
            self.e("AJ", 1, 0, 0, "1", f"{self.BPR:x}", f"{self.arow:x}")
            self.e("DESC", *[f"{w:08x}" for w in desc_words(OP_GEMM_WS, 1,
                                                            self.D, 8)])
            for b in wgt_beats_ws(W, j):
                self.e("WB", f"{b:016x}")


def tile_ops(out: Path) -> dict:
    name = "l4_bias"
    X, w, q_pos = build_case(name)
    r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
    D, T = r.D_model, r.T
    assert D == 128 and r.K_real.shape[1] == D and w.bq is not None
    x8, g1, s_h = selfcheck_front(name, X, w, r)
    s_h64 = f16_bits_to_f64(np.asarray(r.s_h, dtype=np.uint16))

    # composites: BUS_ON grades them (transformer.py:502-505) — read the
    # graded values back out of the arbiter's own definition, never re-invent
    comp_k = [comp_bits(float(tf.f16_grade(float(s_h64[t]) * w.s_wk)),
                        f"comp_k[{t}]") for t in range(T)]
    comp_v = [comp_bits(float(tf.f16_grade(float(s_h64[t]) * w.s_wv)),
                        f"comp_v[{t}]") for t in range(T)]
    comp_q = comp_bits(float(tf.f16_grade(float(s_h64[T]) * w.s_wq)), "comp_q")

    # PRE-RTL SELF-CHECK: the tile's element model must reproduce the arbiter
    # rows exactly, and every element must be inside the hardware window.
    K_exp = f64_to_f16_bits(r.K_real).astype(np.uint16)
    V_exp = f64_to_f16_bits(r.V_real).astype(np.uint16)
    q_exp = f64_to_f16_bits(r.q_real).astype(np.uint16)
    acc_k = tf.gemm_i8_ksplit(r.h8[:T], np.asarray(w.Wk, dtype=np.int64))
    acc_v = tf.gemm_i8_ksplit(r.h8[:T], np.asarray(w.Wv, dtype=np.int64))
    acc_q = tf.gemm_i8_ksplit(r.h8[T:], np.asarray(w.Wq, dtype=np.int64))[0]
    bq16 = f64_to_f16_bits(np.asarray(w.bq, dtype=np.float64)).astype(np.uint16)
    bk16 = f64_to_f16_bits(np.asarray(w.bk, dtype=np.float64)).astype(np.uint16)
    bv16 = f64_to_f16_bits(np.asarray(w.bv, dtype=np.float64)).astype(np.uint16)
    assert np.array_equal(f16_bits_to_f64(bq16), np.asarray(w.bq)), \
        "bq is not on the fp16 grid (golden contract transformer.py:348)"
    margin = []
    for (accs, comps, bias, exp, tagn) in (
            (acc_k, comp_k, bk16, K_exp, "K"), (acc_v, comp_v, bv16, V_exp, "V"),
            (acc_q[None, :], [comp_q], bq16, q_exp[None, :], "q")):
        for t in range(len(comps)):
            for j in range(D):
                y, ok = exact_elem(int(accs[t][j]), comps[t], int(bias[j]))
                assert ok, f"{tagn}[{t}][{j}] outside the hardware window"
                assert y == int(exp[t][j]), \
                    f"{tagn}[{t}][{j}] tile model {y:04x} != arbiter {int(exp[t][j]):04x}"
            ec = (comps[t] >> 23) & 0xFF
            margin.append((ec - 137 + 40, -3 - (ec - 137)))   # (low, high)
    lo_m = min(m[0] for m in margin)
    hi_m = min(m[1] for m in margin)
    # UNBIASED reference for the discrimination arm (l_bias_en = 0)
    q_unb = f64_to_f16_bits(acc_q.astype(np.float64)
                            * float(np.uint32(comp_q).view(np.float32)))
    n_diff = int(np.sum(q_unb.astype(np.uint16) != q_exp))

    s = Ops(D)
    s.arow = 0
    s.e(f"// IC-BIAS tile stream — case {name} (D={D} T={T} CQ-8, rope OFF)")
    s.e("// arbiter: decoder_layer_fx(BUS_ON). CHECK: dbg_f16 beats ==")
    s.e("//   f64_to_f16_bits(r.K_real / r.V_real / r.q_real)  [bias PRE-narrowing]")
    s.e(f"// window margins over every driven element: {lo_m} binades low, "
        f"{hi_m} high (refusal never reachable in contract)")
    s.csrw(CSR["CTRL"], 0x1)                              # tile enable
    s.csrw(CSR["TIER_CTRL"], 0x0)                         # CQ-8 engine
    s.kvw(KV["CTRL"], 0x2)                                # enable it
    s.kvp(KV["STATUS"], 0x1, 0x1)

    # ── phase B: stage h8 rows through the REAL RMSNorm + feeder ────────────
    s.e("// phase B: x8 -> rmsnorm -> feeder(C-1) -> act stage (T+1 rows)")
    s.idle()
    s.route()
    s.e("AJ", 0, 0, 0, f"{T + 1:x}", f"{D // 8:x}", 0)
    s.e("FJOB", f"{T + 1:x}")
    for t in range(T + 1):
        s.e("XR", *[f"{int(v) & 0xFF:02x}" for v in x8[t]])
        s.e("GR", *[f"{int(v) & 0xFFFF:04x}" for v in g1])
        s.e("EFS", f"{s_h[t]:04x}", 1 if t == T else 0)
        s.n_chk += 1

    # ── arm A1: biased K / V / q projections through apex_proj_bias ─────────
    s.e("// A1: LAYER_CTRL[18] l_bias_en = 1, bias staged via LAYER_PTR bank 3")
    s.idle()
    s.csrw(LAYER["CTRL"], L_BIAS_EN)
    s.csrr(LAYER["CTRL"], L_BIAS_EN, L_BIAS_EN)           # level read-back
    s.tap(list(K_exp.reshape(-1)) + list(V_exp.reshape(-1)) + list(q_exp))
    s.route(rdst=1)
    s.e("// K rows (bias bk), one KVQ record per token")
    s.e("TAG A1-K")
    s.stage_bias(bk16)
    for t in range(T):
        s.arow = t
        s.proj_row(w.Wk, comp_k[t], t)
    s.e("// V rows (bias bv)")
    s.e("TAG A1-V")
    s.idle()
    s.stage_bias(bv16)
    for t in range(T):
        s.arow = t
        s.proj_row(w.Wv, comp_v[t], T + t)
    s.e("// q row (bias bq) at the decode-token activation row T")
    s.e("TAG A1-q")
    s.idle()
    s.stage_bias(bq16)
    s.arow = T
    s.proj_row(w.Wq, comp_q, 2 * T)
    s.idle()
    s.csrp(LAYER["STATUS"], 0x100, 0x0)                   # no LAYER error
    s.csrr(LAYER["STATUS"], 0x1FF, 0x0)
    s.csrr(CSR["ERR_STICKY"], 0x0000FFFF, 0x0)            # B1 rule: [15] clean

    # ── arm A2: discrimination — the SAME q row with l_bias_en = 0 ──────────
    s.e(f"// A2 discrimination: l_bias_en=0 reproduces the UNBIASED narrowing")
    s.e(f"//   exactly; it differs from the biased arbiter in {n_diff}/{D} elements")
    s.idle()
    s.csrw(LAYER["CTRL"], 0x0)
    s.csrr(LAYER["CTRL"], L_BIAS_EN, 0x0)
    s.e("TAG A2")
    s.tap(list(q_unb))
    s.arow = T
    s.proj_row(w.Wq, comp_q, 2 * T + 1)
    s.idle()

    # ── arm A3: exact-or-refused, LOUD (poison element 0 -> zero beats) ─────
    s.e("// A3: exact-or-refused — a poisoned composite at element 0 aborts")
    s.e("//     the job: ZERO f16 beats leave the seam (the KVQ record is")
    s.e("//     never opened) and LAYER_STATUS reports code 6 BIAS_WINDOW.")
    s.e("TAG A3")
    s.idle()
    s.csrw(LAYER["CTRL"], L_BIAS_EN)
    s.stage_bias(bq16)
    for (poison, note) in (
            ((90 << 23) | (0x155 << 13), "ex=-47, below the alignment window"),
            ((140 << 23) | (0x2AA << 13), "ex=+3, above the alignment window")):
        y0, ok0 = exact_elem(int(acc_q[0]), poison, int(bq16[0]))
        assert not ok0, f"A3 poison {poison:08x} does not refuse (y={y0:04x})"
        s.e(f"// {note} -> LAYER_STATUS code 6")
        s.kvp(KV["STATUS"], 0x1, 0x1)
        s.kvw(KV["WADDR"], 2 * T + 2)
        s.e("QJOB", 0, f"{D:x}")
        s.e("QS", f"{poison:08x}")
        for _ in range(D - 1):
            s.e("QS", f"{comp_q:08x}")
        s.e("LJOB", f"{s.BPR:x}", "8")
        for j in range(s.BPR):
            s.e("AJ", 1, 0, 0, "1", f"{s.BPR:x}", f"{T:x}")
            s.e("DESC", *[f"{x:08x}" for x in desc_words(OP_GEMM_WS, 1, D, 8)])
            for b in wgt_beats_ws(w.Wq, j):
                s.e("WB", f"{b:016x}")
        s.idle()
        s.csrr(LAYER["STATUS"], 0x1F00, (6 << 9) | 0x100)
        s.csrw(LAYER["STATUS"], 0x100)                    # W1C
        s.csrr(LAYER["STATUS"], 0x100, 0x0)
    s.e("// an illegal S-2 job (cols=0) must be refused by the bias block too")
    s.e("QJOB", 0, "0")
    s.idle()
    s.csrr(LAYER["STATUS"], 0x1F00, (2 << 9) | 0x100)
    s.csrw(LAYER["STATUS"], 0x100)
    s.csrr(CSR["ERR_STICKY"], 0x0000FFFF, 0x0)            # still never [15]

    s.e("ENDTAPS")
    s.e("DONE")
    (out / "bias_tile.ops").write_text("\n".join(s.L) + "\n")
    return {"tile_case": name, "tile_T": T, "tile_D": D,
            "tap_checks": int(K_exp.size + V_exp.size + q_exp.size
                              + q_unb.size),
            "tile_checks": s.n_chk, "unbiased_diff_elems": n_diff,
            "window_margin_low_binades": lo_m,
            "window_margin_high_binades": hi_m}


def selfcheck_front(name, X, w, r):
    """The segment-2 front-half re-derive, verbatim in intent: x8 from the
    BUS_ON x_f16 narrowing, rmsnorm_fx_wide -> r.h, C-1 -> r.h8 / r.s_h. A
    failure is THIS generator's, never golden's."""
    Xf = tf._f16(np.asarray(X, dtype=np.float64))
    x8, _ = tf.quant_rows_i8(Xf)
    g1 = [int(v) for v in np.asarray(w.gamma1)]
    for t in range(r.T + 1):
        y, _, _ = tf.rmsnorm_fx_wide([int(v) for v in x8[t]], g1)
        assert np.array_equal(np.asarray(y, dtype=np.int64), r.h[t]), \
            f"{name}: h[{t}] re-derive"
    h8, s_h = tf.quant_rows_i8(r.h.astype(np.float64) / 256.0)
    assert np.array_equal(h8, r.h8) and np.array_equal(s_h, r.s_h), \
        f"{name}: h8/s_h re-derive"
    return x8, g1, [int(v) for v in r.s_h]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    m = {}
    m.update(unit_vectors(out))
    m.update(tile_ops(out))
    for k in sorted(m):
        print(f"  {k:32s} {m[k]}")
    print("IC-BIAS VECTORS: PASS (all generator self-checks vs the BUS_ON "
          "arbiter green)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
