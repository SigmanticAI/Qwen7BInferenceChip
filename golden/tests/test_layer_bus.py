#!/usr/bin/env python3
"""test_layer_bus.py — C-LBUS (D-030) gates: the hardware bus-grade
composition mode of decoder_layer_fx (IB-LAYER S1).

Sections:
  A  OFF bit-identity: bus=None / BUS_OFF / legacy call are byte-identical
     on EVERY LayerFx array and every per-head (o8, s_out) — the
     "flags default-OFF, banner byte-identical" half of the R4 ruling.
  B  Mode properties: graded composites are f16_grade-idempotent; BUS_ON
     r2 is fp16-grid; BUS_ON is deterministic; the chunked-merge guard
     refuses T > CHUNK_T_MAX loudly.
  C  RTL-realizability lemmas (exact-Fraction): under BUS_ON the float64
     intermediates feeding every single-RNE narrowing are EXACT — the
     per-head attn dequant o8*graded, both residual sums, and the SwiGLU
     silu*up product — so an integer datapath can be bit-exact against
     BUS_ON with one RNE per contract point. This is the lemma that
     dissolves the "residual/SwiGLU operand" narrowing question
     (IB_LAYER.md §2b): no extra narrowing flag exists because none is
     needed.
  D  Pinned-delta regression registers (B3-finding-3 style): the measured
     BUS_ON e2e deltas and yardstick errors per L4 case, and the measured
     rope-only absorption fact. Numbers produced by
     verif/top/l4/measure_bus_deltas.py; a drift forces a conscious
     re-measure + re-pin, never a silent slide.

OWNER SIGN-OFF: PENDING (LEVEL_C_INTEGRATION.md §9.1 R4). Until it is
recorded in docs/design/IB_LAYER.md, no RTL stage gates against BUS_ON.
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from apex_golden import attention as at  # noqa: E402
from apex_golden import transformer as tf  # noqa: E402
from apex_golden import fp  # noqa: E402
from apex_golden.weight_codec import f16_grade  # noqa: E402

TIER = at.TIER_CQ8

ARR_FIELDS = ("h", "h8", "s_h", "q_real", "K_real", "V_real", "q_rope",
              "K_rope", "attn", "attn_proj", "r1", "h2", "gate_real",
              "up_real", "swiglu", "ffn_out", "r2")

# ── the L4 case set (mirror of verif/top/l4/l4_cases.py — golden tests stay
#    self-contained; the mirror is asserted by that script's own use) ─────────
CASES = [
    ("l4_h2_hd64",     2, 64, 256,  8, 0x14B00001, 0, 1e4,  False, None),
    ("l4_h1_hd128",    1, 128, 344, 16, 0x14B00002, 0, 1e4,  False, None),
    ("l4_gqa_h2kv1",   2, 64, 256, 12, 0x14B00003, 1, 1e4,  False, None),
    ("l4_qwen_theta",  2, 64, 256,  8, 0x14B00004, 0, 1e6,  False, None),
    ("l4_bias",        2, 64, 256,  8, 0x14B00005, 0, 1e6,  True,  None),
    ("l4_selfinc",     2, 64, 256,  8, 0x14B00006, 0, 1e4,  False, "self"),
]


def make_weights(rng, H, hd, dff, H_kv=0, theta=1e4, bias=False):
    D = H * hd
    kvd = (H_kv or H) * hd

    def w8(shape):
        return rng.integers(-127, 128, shape).astype(np.int64)

    sp = 1.0 / (127.0 * math.sqrt(D))
    sd = 1.0 / (127.0 * math.sqrt(dff))
    gam = lambda: np.array(  # noqa: E731
        [int(round((0.7 + 0.6 * rng.random()) * 8192)) for _ in range(D)],
        dtype=np.int64)
    kw = {}
    if bias:
        kw = dict(bq=tf._f16(rng.normal(0, .05, D)),
                  bk=tf._f16(rng.normal(0, .05, kvd)),
                  bv=tf._f16(rng.normal(0, .05, kvd)))
    return tf.LayerWeights(
        Wq=w8((D, D)), Wk=w8((D, kvd)), Wv=w8((D, kvd)), Wo=w8((D, D)),
        Wg=w8((D, dff)), Wu=w8((D, dff)), Wd=w8((dff, D)),
        s_wq=sp * 1.01, s_wk=sp * 0.97, s_wv=sp * 1.0,
        s_wo=sp * 1.03, s_wg=sp * 0.99, s_wu=sp * 1.02, s_wd=sd * 1.01,
        gamma1=gam(), gamma2=gam(), H=H, head_dim=hd, H_kv=H_kv,
        rope_theta_base=theta, **kw)


def build_case(name):
    for (n, H, hd, dff, T, seed, H_kv, theta, bias, q_pos) in CASES:
        if n != name:
            continue
        rng = np.random.default_rng(seed)
        w = make_weights(rng, H, hd, dff, H_kv=H_kv, theta=theta, bias=bias)
        D = H * hd
        if q_pos == "self":
            Xc = rng.normal(0, 1, (T, D))
            return np.vstack([Xc, Xc[-1:]]), w, T - 1
        return rng.normal(0, 1, (T + 1, D)), w, None
    raise KeyError(name)


def maxrel(a, b):
    den = float(np.max(np.abs(b)))
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / (den or 1.0)


def bytes_of(r):
    out = [np.asarray(getattr(r, f)).tobytes() for f in ARR_FIELDS]
    for hd_ in r.heads:
        out.append(np.asarray(hd_.o8).tobytes())
        out.append(repr(float(hd_.s_out)).encode())
    return b"".join(out)


def frac_exact_sum(x, y):
    """True iff float64 x + y is EXACT (no rounding) — the lemma predicate."""
    return Fraction(x) + Fraction(y) == Fraction(float(x + y))


def frac_exact_prod(x, y):
    return Fraction(x) * Fraction(y) == Fraction(float(x * y))


# ── A: OFF bit-identity ──────────────────────────────────────────────────────
def sec_a():
    for name in ("l4_h2_hd64", "l4_bias", "l4_gqa_h2kv1", "l4_selfinc"):
        X, w, q_pos = build_case(name)
        legacy = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos)
        none_b = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=None)
        off_b = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_OFF)
        assert bytes_of(legacy) == bytes_of(none_b) == bytes_of(off_b), name
    print("A: OFF bit-identity — legacy/None/BUS_OFF byte-identical on "
          f"{len(ARR_FIELDS)} fields + per-head (o8, s_out), 4 cases")


# ── B: mode properties ───────────────────────────────────────────────────────
def sec_b():
    X, w, q_pos = build_case("l4_h2_hd64")
    # graded epilogue scale is f16_grade-idempotent (C2 form)
    _, s_out = tf._proj_epilogue(X[0], w.Wo, w.s_wo, grade=True)
    assert f16_grade(s_out) == s_out and s_out > 0
    _, s_raw = tf._proj_epilogue(X[0], w.Wo, w.s_wo)
    assert f16_grade(s_raw) != s_raw, "case must exercise grading (non-pow2 s_w)"
    r1 = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
    r2 = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
    assert bytes_of(r1) == bytes_of(r2), "BUS_ON determinism"
    assert np.array_equal(r1.r2, tf._f16(r1.r2)), "r2 on the fp16 grid"
    # chunked-merge guard: loud refusal, not a silent relaxed composition
    Xw = np.zeros((tf.CHUNK_T_MAX + 2, w.H * w.head_dim))
    try:
        tf.decoder_layer_fx(Xw, w, TIER, bus=tf.BUS_ON)
        raise SystemExit("B: chunked guard FAILED to fire")
    except AssertionError:
        pass
    print("B: graded s_out idempotent (and grading exercised), BUS_ON "
          "deterministic, r2 fp16-grid, chunked guard fires")


# ── C: RTL-realizability lemmas (exact-Fraction) ─────────────────────────────
def sec_c():
    checked = 0
    for name in ("l4_h2_hd64", "l4_h1_hd128", "l4_bias"):
        X, w, q_pos = build_case(name)
        r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
        hd = w.head_dim
        # per-head attn dequant: attn == o8 * f16_grade(s_out), elementwise
        # exact in float64 (o8 <= 8 bits x graded <= 11 bits)
        for h_i, core in enumerate(r.heads):
            g = f16_grade(core.s_out)
            for j, o in enumerate(core.o8):
                assert frac_exact_prod(float(o), g)
                assert r.attn[h_i * hd + j] == float(int(o) * Fraction(g))
                checked += 1
        # residual sums exact in float64 => single-RNE realizable
        xt = tf._f16(np.asarray(X, dtype=np.float64))[-1]
        for i in range(r.D_model):
            assert frac_exact_sum(float(xt[i]), float(r.attn_proj[i]))
            assert frac_exact_sum(float(r.r1[i]), float(r.ffn_out[i]))
            checked += 2
        # SwiGLU product exact in float64 (silu fp16-grid x up = acc*graded)
        silu = tf.silu_apply(r.gate_real)
        for i in range(r.d_ffn):
            assert frac_exact_prod(float(silu[i]), float(r.up_real[i]))
            checked += 1
        assert np.array_equal(r.swiglu, tf._f16(silu * r.up_real))
    print(f"C: realizability lemmas — {checked} exact-Fraction checks: "
          "attn o8*graded, both residual sums, silu*up all EXACT in f64 "
          "(single-RNE realizable); no extra narrowing flag needed")


# ── D: pinned-delta regression registers ─────────────────────────────────────
# (d_ON, err_legacy, err_BUS_ON) per case — measured 2026-07-22 by
# verif/top/l4/measure_bus_deltas.py on numpy defaults; integer GEMMs are
# exact and every float step is elementwise IEEE, so these reproduce
# bit-stably. RTOL forces a conscious re-pin on any drift.
PINS = {
    "l4_h2_hd64":   (3.386005e-03, 4.144319e-03, 3.673394e-03),
    "l4_h1_hd128":  (3.057554e-03, 3.524082e-03, 3.498405e-03),
    "l4_gqa_h2kv1": (2.906977e-03, 4.156051e-03, 4.080475e-03),
    "l4_qwen_theta": (3.825555e-03, 5.928302e-03, 3.663657e-03),
    "l4_bias":      (2.403846e-03, 3.056194e-03, 2.619128e-03),
    "l4_selfinc":   (3.224073e-03, 7.195027e-03, 6.376116e-03),
}
RTOL = 1e-6
# rope-only absorption fact (anchor case): q_rope/K_rope fp16 words that
# differ, and the attention output bit-identical (absorbed at the KVQ/feeder
# re-quantization) — measured, pinned; NOT claimed as a theorem.
ROPE_ABSORB_PIN = ("l4_h2_hd64", 33, 254)


def sec_d():
    for name, (p_don, p_el, p_eo) in PINS.items():
        X, w, q_pos = build_case(name)
        leg = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos)
        on = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
        d_on = maxrel(on.r2, leg.r2)
        err_l = maxrel(leg.r2, tf.decoder_layer_ref(X, w, q_pos=q_pos).y_ref)
        err_o = maxrel(on.r2,
                       tf.decoder_layer_ref(tf._f16(X), w, q_pos=q_pos).y_ref)
        for got, pin, what in ((d_on, p_don, "d_ON"), (err_l, p_el, "err_leg"),
                               (err_o, p_eo, "err_ON")):
            assert abs(got - pin) <= RTOL * pin, \
                f"{name} {what}: got {got:.6e} pinned {pin:.6e} — re-measure & re-pin"
        assert err_o <= err_l * 1.10, f"{name}: BUS_ON degraded vs yardstick"
    name, pin_q, pin_k = ROPE_ABSORB_PIN
    X, w, q_pos = build_case(name)
    leg = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos)
    ro = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos,
                             bus=tf.BusMode(rope_in_f16=True))
    nq = int(np.sum(fp.f64_to_f16_bits(ro.q_rope) != fp.f64_to_f16_bits(leg.q_rope)))
    nk = int(np.sum(fp.f64_to_f16_bits(ro.K_rope) != fp.f64_to_f16_bits(leg.K_rope)))
    assert (nq, nk) == (pin_q, pin_k), f"rope word-diff drift: {(nq, nk)}"
    assert np.asarray(ro.attn).tobytes() == np.asarray(leg.attn).tobytes()
    print(f"D: pinned registers — 6 cases x (d_ON, err_leg, err_ON) within "
          f"rtol {RTOL:g}; err_ON <= err_leg on all; rope-only absorption "
          f"pinned (q̂/K̂ words {pin_q}/{pin_k} differ, attn bit-identical)")


def main():
    sec_a()
    sec_b()
    sec_c()
    sec_d()
    print("APEX LAYER-BUS GOLDEN (C-LBUS/D-030): OFF BIT-IDENTICAL, "
          "LEMMAS EXACT, DELTAS PINNED — ALL PASS")


if __name__ == "__main__":
    main()
