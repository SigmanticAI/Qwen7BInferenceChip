"""test_transformer.py — Layer-3 golden gate for the FULL decoder layer
(RoPE + multi-head attention + SwiGLU FFN + residuals), extending the D-021
attention gate (tests/test_attention.py) to the whole layer.

Four sections, all mandatory:

  A. RoPE  — rope_fx vs rope_ref within a DERIVED budget, plus the exhaustive
     cos/sin LUT LEMMA (re-proven over the whole phase domain).
  B. SiLU  — silu_fx vs silu_ref within a DERIVED budget, plus the exhaustive
     SiLU LUT LEMMA (re-proven over the whole Q5.10 input grid).
  C. DECODER LAYER — decoder_layer_fx vs decoder_layer_ref over several small
     (H, head_dim, T) configs, gated per-sublayer (ISOLATED contract bounds)
     and end-to-end (a COMPOSITION budget built from the individually-gated
     sublayer errors — the b_comp discipline of test_attention).
  D. DETERMINISM — fixed seeds reproduce bit-identical fx intermediates.

════════════════════════════════════════════════════════════════════════════
ANALYTIC BUDGET DERIVATION (documented per the deliverable spec; the reused
primitives [Q]/[F16]/[RMS]/LEMMA-EXP/[E2E] are defined in test_attention.py)
════════════════════════════════════════════════════════════════════════════
[LEMMA-COS] cos/sin LUT (contract C-ROPE), re-proven exhaustively in A:
    over ALL phase codes u_q ∈ [0,2^14), |cos_fx(u_q)/2^14 − cos(2π u_q/2^14)|
    ≤ CS_LUT_ABS (chord h²/8 with h = 2π/256 = 7.5e-5, + Q2.14 table RNE
    + interp truncation).  Measured 1.65e-4; analytic cap CS_LUT_ABS = 1.9e-4.
    Including the phase reduction of a REAL angle φ (u = frac(φ/2π) quantized
    to Q0.14, ≤ π·2^-14 = 1.9e-4 rad ⇒ ≤1.9e-4 on cos/sin), the real-angle
    error is bounded by CS_ABS = 3.5e-4 (measured ≤ 3.06e-4).

[ROPE] rope_fx vs rope_ref, per output element (half-split pair (i, i+half)):
    x̂_lo = f16(x1·c_lut − x2·s_lut), ref = x1·c_true − x2·s_true. Hence
      |x̂_lo − ref| ≤ CS_ABS·(|x1|+|x2|) + |lo_lut|·2^-11
                   ≤ CS_ABS·(|x1|+|x2|) + (|ref|+CS_ABS·(|x1|+|x2|))·2^-11
    ([F16] one fp16 narrowing of the fp16 bus). Same for the hi half.

[LEMMA-SILU] SiLU LUT (contract C-SILU), re-proven exhaustively in B:
    over ALL x_q ∈ [−8·2^10, 8·2^10], |silu_fx(x_q)/2^12 − silu(x_q/2^10)|
    ≤ SILU_LUT_ABS (chord h²/8·max|silu''| = 0.0625²/8·0.5 = 2.44e-4, + Q4.12
    table RNE + interp truncation). Measured 4.53e-4; cap SILU_LUT_ABS = 5.0e-4.

[SILU] silu_apply(x) vs silu_ref(x) (real x), per element:
    input quant [Q]: |Δx| ≤ 0.5·2^-10 ⇒ ≤ max|silu'|·0.5·2^-10 = 5.37e-4
    (SILU_DERIV = 1.10); output fp16 [F16]: ≤ |silu(x)|·2^-11. Total
      |silu_apply − silu_ref| ≤ SILU_LUT_ABS + SILU_INQ + |silu_ref|·2^-11,
      SILU_INQ = SILU_DERIV·0.5·2^-10. Measured max 9.47e-4 ≤ cap.

[LAYER] decoder_layer_fx vs decoder_layer_ref. y = r1 + ffn_out, r2 = f16(y).
    ISOLATED per-sublayer gates (each fx sublayer vs a float ref from its OWN
    fx inputs, gated against its contract bound):
      rmsnorm1/2 : [RMS] per element (test_attention formula, D_ALL = 3.0).
      attention  : per head, D-023 5%-of-value-scale acceptance (core
                   yardstick attention_ref_core on the fx-roped q/K, fx V) —
                   reuses the proven D-021 chain gate; CQ-8 tier.
      out/down proj epilogues : |proj_fx − (in_fx @ W·s_w)| ≤ 0.57·s_a·‖W‖₁·s_w
                   + 0.5·s_out  (isolates INT8-feeder [Q] + C-2 epilogue RNE).
      silu       : [SILU]/[LEMMA-SILU] above.
    END-TO-END COMPOSITION budget B_layer (sound GIVEN the isolated gates —
    same construction as test_attention's b_comp), per output element d:
      E_attnproj_d = Σ_j |Wo[j,d]|·s_wo·(δattn_j + 0.57·s_a) + 0.5·s_out_o
        (δattn_j = MEASURED, isolated-gated per-head attention error incl. RoPE)
      E_r1_d       = E_attnproj_d + |r1_ref_d|·2^-11
      B_ffn_d      = Σ_k |Wd[k,d]|·s_wd·(δswiglu_k + 0.57·s_p) + 0.5·s_out_d
        δswiglu_k  = δsg_k·|up_fx_k| + |sg_loc_k|·δup_k + |swiglu_k|·2^-11
        δsg_k      = SILU_LUT_ABS + SILU_INQ + |sg_loc_k|·2^-11
                     + SILU_DERIV·δgate_k        (LUT + input-sens on δgate)
        δgate/up_k = Σ ( [RMS]_h2 + 0.57·s_h2 ) · |Wg/u|·s_wg/u   ([Q]+[RMS])
      sens_d       = |FFN_float(r1_fx) − FFN_float(r1_ref)|  (MEASURED r1-
                     sensitivity of the FFN branch; exact float, no fx)
      B_layer_d    = E_r1_d + B_ffn_d + sens_d + |y_ref_d|·2^-11
    Gate: max_d |r2_fx − y_ref|_d ≤ max_d B_layer_d, and the relative
    acceptance |r2_fx − y_ref|_max ≤ D023_FRAC·max|y_ref| (5% of the layer
    output signal scale). A blown budget FAILS with the offending stage named.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex_golden import attention as at                             # noqa: E402
from apex_golden import compute as cp                               # noqa: E402
from apex_golden import transformer as tf                           # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits, rne    # noqa: E402

# reused / new analytic constants
CS_LUT_ABS = 1.9e-4          # [LEMMA-COS] grid-exhaustive cap (meas 1.65e-4)
CS_ABS = 3.5e-4             # [ROPE] real-angle cos/sin cap (meas 3.06e-4)
SILU_LUT_ABS = 5.0e-4       # [LEMMA-SILU] grid-exhaustive cap (meas 4.53e-4)
SILU_DERIV = 1.10           # max|silu'| = 1.0998
SILU_INQ = SILU_DERIV * 0.5 * 2.0 ** -tf.SILU_IN_FRAC
Q8_ERR = 0.57               # [Q] INT8 hard per-element factor
D_ALL = 3.0                 # [RMS] denominator slack
D023_FRAC = 0.05            # 5% acceptance (per test_attention D-023)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ════════════════════════════ A. RoPE ════════════════════════════════════════

def section_a() -> None:
    print("[A] RoPE — cos/sin LUT lemma + rope_fx vs rope_ref")

    # A-1 LEMMA-COS: exhaustive over the whole phase domain (phase-exact grid)
    uq = np.arange(1 << tf.PH_FRAC, dtype=np.int64)
    ang = 2.0 * math.pi * uq / (1 << tf.PH_FRAC)
    ec = np.max(np.abs(tf.cos_fx(uq) / (1 << tf.CS_OUT_FRAC) - np.cos(ang)))
    es = np.max(np.abs(tf.sin_fx(uq) / (1 << tf.CS_OUT_FRAC) - np.sin(ang)))
    check(f"LEMMA-COS |Δcos| ≤ {CS_LUT_ABS} (all {len(uq)} codes)",
          bool(ec <= CS_LUT_ABS), f"max {ec:.3e}")
    check(f"LEMMA-COS |Δsin| ≤ {CS_LUT_ABS} (all {len(uq)} codes)",
          bool(es <= CS_LUT_ABS), f"max {es:.3e}")

    # A-2 real-angle cap CS_ABS (phase reduction folded in), swept positions
    rng = np.random.default_rng(0x2051)
    phi = rng.uniform(0, 200.0, 4_000_000)
    u = np.mod(phi / (2 * math.pi), 1.0)
    uq2 = rne(u * (1 << tf.PH_FRAC)) % (1 << tf.PH_FRAC)
    rc = np.max(np.abs(tf.cos_fx(uq2) / (1 << tf.CS_OUT_FRAC) - np.cos(phi)))
    rs = np.max(np.abs(tf.sin_fx(uq2) / (1 << tf.CS_OUT_FRAC) - np.sin(phi)))
    check(f"real-angle |Δcos|,|Δsin| ≤ CS_ABS={CS_ABS}",
          bool(max(rc, rs) <= CS_ABS), f"max {max(rc, rs):.3e}")

    # A-3 rope_fx vs rope_ref within [ROPE] budget, swept head_dim/positions
    worst = 0.0
    for head_dim in (64, 128):
        theta = tf.rope_theta(head_dim)
        half = head_dim // 2
        X = f16_bits_to_f64(f64_to_f16_bits(
            rng.normal(0, 2, (130, head_dim)))).reshape(130, head_dim)
        pos = np.arange(130)
        got = tf.rope_fx(X, pos, theta)
        ref = tf.rope_ref(X, pos, theta)
        x1, x2 = np.abs(X[:, :half]), np.abs(X[:, half:])
        base = CS_ABS * (x1 + x2)                     # per-pair, both halves
        base2 = np.concatenate([base, base], axis=1)
        bnd = base2 + (np.abs(ref) + base2) * 2.0 ** -11 + 1e-12
        ok = bool(np.all(np.abs(got - ref) <= bnd))
        worst = max(worst, float(np.max(np.abs(got - ref))))
        check(f"rope_fx within [ROPE] budget (head_dim={head_dim}, T=130)", ok)
    print(f"    RoPE worst abs err over sweep = {worst:.3e}")

    # A-4 convention sanity: HF half-split, pos 0 is identity; norm preserved
    theta = tf.rope_theta(64)
    x = f16_bits_to_f64(f64_to_f16_bits(rng.normal(0, 1, 64))).reshape(64)
    r0 = tf.rope_ref(x, 0, theta)
    check("rope_ref(pos=0) == identity", bool(np.allclose(r0, x, atol=1e-12)))
    rm = tf.rope_ref(x, 37, theta)
    check("rope_ref preserves per-pair norm (rotation)",
          bool(np.allclose(rm[:32] ** 2 + rm[32:] ** 2,
                           x[:32] ** 2 + x[32:] ** 2, atol=1e-9)))


# ════════════════════════════ B. SiLU ════════════════════════════════════════

def section_b() -> None:
    print("[B] SiLU — LUT lemma + silu_apply vs silu_ref")

    # B-1 LEMMA-SILU: exhaustive over the whole Q5.10 input grid (input-exact)
    xq = np.arange(-(8 << tf.SILU_IN_FRAC), (8 << tf.SILU_IN_FRAC) + 1,
                   dtype=np.int64)
    xr = xq.astype(np.float64) / (1 << tf.SILU_IN_FRAC)
    got = tf.silu_fx(xq) / (1 << tf.SILU_OUT_FRAC)
    ref = tf.silu_ref(xr)
    e = np.max(np.abs(got - ref))
    check(f"LEMMA-SILU |Δ| ≤ {SILU_LUT_ABS} (all {len(xq)} grid pts)",
          bool(e <= SILU_LUT_ABS), f"max {e:.3e}")
    check("silu_fx(0) == 0 exactly", int(tf.silu_fx(np.array([0]))[0]) == 0)

    # B-2 silu_apply vs silu_ref within [SILU] budget, swept real domain
    rng = np.random.default_rng(0x5170)
    x = rng.uniform(-8, 8, 4_000_000)
    g = tf.silu_apply(x)
    r = tf.silu_ref(x)
    bnd = SILU_LUT_ABS + SILU_INQ + np.abs(r) * 2.0 ** -11 + 1e-12
    check("silu_apply within [SILU] budget (4M pts over [-8,8])",
          bool(np.all(np.abs(g - r) <= bnd)),
          f"max {float(np.max(np.abs(g - r))):.3e}")

    # B-3 clamp behavior is the documented contract (saturates, not wraps)
    hi = tf.silu_fx(np.array([20 << tf.SILU_IN_FRAC]))[0]
    lo = tf.silu_fx(np.array([-(20 << tf.SILU_IN_FRAC)]))[0]
    check("silu_fx clamps |x|>8 to the domain edges (contract)",
          hi == tf.silu_fx(np.array([8 << tf.SILU_IN_FRAC]))[0]
          and lo == tf.silu_fx(np.array([-(8 << tf.SILU_IN_FRAC)]))[0])


# ═══════════════════════════ C. DECODER LAYER ════════════════════════════════

def rms_bound(x8_row: np.ndarray, gamma: np.ndarray, den: float) -> np.ndarray:
    """[RMS] per-element abs bound for one rmsnorm_fx row (test_attention form)."""
    g_abs = np.abs(np.asarray(gamma, dtype=np.float64)) / (1 << 13)
    u = math.sqrt(float(np.mean(x8_row.astype(np.float64) ** 2)))
    assert u >= 15, f"[RMS] validity: degenerate row (u={u:.2f})"
    u_lb = max(u - 2.0, 1.0)
    return (g_abs * (np.abs(x8_row) * (2.0 ** -13 + D_ALL / (den * u_lb))
                     + 0.57 / u_lb)) + 2.0 ** -9


def make_weights(rng, H, head_dim, d_ffn):
    D = H * head_dim
    def w(shape):
        return rng.integers(-127, 128, shape).astype(np.int64)
    s_proj = 1.0 / (127.0 * math.sqrt(D))
    s_ffn = 1.0 / (127.0 * math.sqrt(D))
    s_down = 1.0 / (127.0 * math.sqrt(d_ffn))
    gamma1 = np.array([int(round((0.7 + 0.6 * rng.random()) * 8192))
                       for _ in range(D)], dtype=np.int64)
    gamma2 = np.array([int(round((0.7 + 0.6 * rng.random()) * 8192))
                       for _ in range(D)], dtype=np.int64)
    return tf.LayerWeights(
        Wq=w((D, D)), Wk=w((D, D)), Wv=w((D, D)), Wo=w((D, D)),
        Wg=w((D, d_ffn)), Wu=w((D, d_ffn)), Wd=w((d_ffn, D)),
        s_wq=s_proj, s_wk=s_proj, s_wv=s_proj, s_wo=s_proj,
        s_wg=s_ffn, s_wu=s_ffn, s_wd=s_down,
        gamma1=gamma1, gamma2=gamma2, H=H, head_dim=head_dim)


def gate_layer(name: str, X, w, tier) -> dict:
    r = tf.decoder_layer_fx(X, w, tier)
    ref = tf.decoder_layer_ref(X, w)
    T, H, hd, D = r.T, w.H, w.head_dim, r.D_model
    d_ffn = r.d_ffn
    stage_fail: list[str] = []

    def gate(stage, meas, bound):
        if not (meas <= bound):
            stage_fail.append(f"{stage} meas={meas:.3e} > budget={bound:.3e}")

    # ── ISOLATED: rmsnorm1 (decode-token row T) ─────────────────────────────
    x8, _ = at.quant_rows_i8(X)
    _, _, nrm1 = at.rmsnorm_fx([int(v) for v in x8[T]],
                               [int(v) for v in w.gamma1])
    # den for the bound: recompute isqrt(mean2+1) (rmsnorm_fx internals)
    sum2 = int(np.sum(x8[T].astype(np.int64) ** 2))
    den1 = math.isqrt((sum2 >> int(math.log2(D))) + 1)
    h_row_fx = r.h[T].astype(np.float64) / 256.0
    h_row_ref = ref.h_ref[T]
    gate("rmsnorm1", float(np.max(np.abs(h_row_fx - h_row_ref))),
         float(np.max(rms_bound(x8[T], w.gamma1, den1))))

    # ── ISOLATED: per-head attention (D-023 core yardstick) ─────────────────
    vnorm_head = []
    for h_i in range(H):
        sl = slice(h_i * hd, (h_i + 1) * hd)
        core = r.heads[h_i]
        q_h = r.q_rope[sl]
        K_f16 = f64_to_f16_bits(r.K_rope[:, sl]).reshape(T, hd)
        V_f16 = f64_to_f16_bits(r.V_real[:, sl]).reshape(T, hd)
        refc = at.attention_ref_core(q_h, K_f16, V_f16)
        vmax = float(np.max(np.abs(refc.V_ref)))
        vnorm_head.append(vmax)
        gate(f"attn_head{h_i}", float(np.max(np.abs(core.out_hat - refc.y_ref))),
             D023_FRAC * vmax)

    # ── ISOLATED: output-projection epilogue ────────────────────────────────
    a8, s_a_b = at.quant_rows_i8(r.attn[None, :])
    s_a = float(f16_bits_to_f64(np.array([s_a_b[0]]))[0])
    attn_proj_deq = (a8[0].astype(np.float64) * s_a) @ (
        np.asarray(w.Wo, dtype=np.float64) * w.s_wo)
    s_out_o = _epilogue_scale(r.attn, w.Wo, w.s_wo)   # C-2 epilogue output scale
    b_proj = (Q8_ERR * s_a
              * float(np.max(np.abs(np.asarray(w.Wo, float)).sum(axis=0)))
              * w.s_wo) + 0.5 * s_out_o
    gate("out_proj_epi", float(np.max(np.abs(r.attn_proj - attn_proj_deq))),
         b_proj)

    # ── ISOLATED: rmsnorm2 (on r1_fx) ───────────────────────────────────────
    r1_8, _ = at.quant_rows_i8(r.r1[None, :])
    sum2b = int(np.sum(r1_8[0].astype(np.int64) ** 2))
    den2 = math.isqrt((sum2b >> int(math.log2(D))) + 1)
    h2_fx = r.h2.astype(np.float64) / 256.0
    h2_ref_local = cp.rmsnorm_ref(r.r1, np.asarray(w.gamma2, float) / (1 << 13))
    gate("rmsnorm2", float(np.max(np.abs(h2_fx - h2_ref_local))),
         float(np.max(rms_bound(r1_8[0], w.gamma2, den2))))

    # ── FFN branch: local ref from r1_fx, isolated + sensitivity ────────────
    Wg = np.asarray(w.Wg, float) * w.s_wg
    Wu = np.asarray(w.Wu, float) * w.s_wu
    Wd = np.asarray(w.Wd, float) * w.s_wd
    gate_loc = h2_ref_local @ Wg
    up_loc = h2_ref_local @ Wu
    sg_loc = tf.silu_ref(gate_loc)
    ffn_loc = (sg_loc * up_loc) @ Wd                     # FFN_float(r1_fx)
    assert np.max(np.abs(gate_loc)) <= tf.SILU_DOMAIN, \
        f"gate activation out of SiLU domain ({np.max(np.abs(gate_loc)):.2f})"

    # δgate/δup [Q]+[RMS] through the projections
    dh2 = rms_bound(r1_8[0], w.gamma2, den2)             # [D]
    s_h2 = float(f16_bits_to_f64(np.array([at.quant_rows_i8(
        r.h2.astype(np.float64)[None, :] / 256.0)[1][0]]))[0])
    dfeed = dh2 + Q8_ERR * s_h2
    dgate = dfeed @ np.abs(Wg)                            # [d_ffn]
    dup = dfeed @ np.abs(Wu)
    up_fx = r.up_real
    swiglu = r.swiglu
    dsg = SILU_LUT_ABS + SILU_INQ + np.abs(sg_loc) * 2.0 ** -11 \
        + SILU_DERIV * dgate
    dswiglu = dsg * np.abs(up_fx) + np.abs(sg_loc) * dup \
        + np.abs(swiglu) * 2.0 ** -11
    a8p, s_p_b = at.quant_rows_i8(swiglu[None, :])
    s_p = float(f16_bits_to_f64(np.array([s_p_b[0]]))[0])
    s_out_d = _epilogue_scale(swiglu, w.Wd, w.s_wd)
    B_ffn = (np.abs(Wd).T @ dswiglu) + Q8_ERR * s_p * np.abs(Wd).sum(axis=0) \
        + 0.5 * s_out_d                                  # [D]
    gate("ffn_isolated", float(np.max(np.abs(r.ffn_out - ffn_loc))),
         float(np.max(B_ffn)))

    # ── END-TO-END composition budget ───────────────────────────────────────
    delta_attn = np.abs(r.attn - ref.attn_ref)           # measured, gated
    E_attnproj = (np.abs(np.asarray(w.Wo, float)).T
                  @ (delta_attn + Q8_ERR * s_a)) * w.s_wo + 0.5 * s_out_o
    E_r1 = E_attnproj + np.abs(ref.r1_ref) * 2.0 ** -11
    # FFN r1-sensitivity (exact float, measured)
    ffn_ref = ref.ffn_out_ref
    sens = np.abs(ffn_loc - ffn_ref)
    B_layer = E_r1 + B_ffn + sens + np.abs(ref.y_ref) * 2.0 ** -11
    err = np.abs(r.r2 - ref.y_ref)
    m_max = float(np.max(err))
    gate("e2e_comp", m_max, float(np.max(B_layer)))
    ynorm = float(np.max(np.abs(ref.y_ref)))
    gate("e2e_accept_5pct", m_max, D023_FRAC * ynorm)

    for f in stage_fail:
        FAILS.append(f"{name}: {f}")
    return {
        "name": name, "tier": tier, "H": H, "hd": hd, "T": T, "d_ffn": d_ffn,
        "e2e_max": m_max, "e2e_rel": m_max / max(ynorm, 1e-12),
        "b_comp": float(np.max(B_layer)), "b_comp_rel": float(np.max(B_layer)) / max(ynorm, 1e-12),
        "fails": stage_fail,
    }


def _epilogue_scale(vec_real, W8, s_w) -> float:
    """The C-2 epilogue output scale s_out for proj_epilogue(vec_real, W8)."""
    a8, s_a = at.quant_rows_i8(np.asarray(vec_real, float)[None, :])
    acc = cp.gemm_i8(a8, np.asarray(W8, np.int64))[0].astype(np.int64)
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = at.calib_requant(amax)
    s_a_val = float(f16_bits_to_f64(np.array([s_a[0]]))[0])
    return s_a_val * float(s_w) * float(1 << shift) / float(scale)


def section_c() -> list[dict]:
    print("[C] decoder layer — decoder_layer_fx vs decoder_layer_ref")
    rows: list[dict] = []
    rng = np.random.default_rng(0x0DEC0DE)
    # D_model = H·head_dim is bounded by the FROZEN ASU rmsnorm envelope
    # (rmsnorm_fx: D pow-2, D ≤ 128). So D_model = 128: (H=2,hd=64) multi-head
    # and (H=1,hd=128) exercise both head_dim envelope points. d_ffn (down-proj
    # K ≤ 2048) is unconstrained by rmsnorm.
    configs = [(2, 64, 256), (1, 128, 344), (2, 64, 344)]   # (H, head_dim, d_ffn)
    for (H, hd, d_ffn) in configs:
        D = H * hd
        w = make_weights(rng, H, hd, d_ffn)
        for T in (8, 16, 64):
            X = rng.normal(0, 1, (T + 1, D))
            X *= 4.0 / np.max(np.abs(X))
            X = f16_bits_to_f64(f64_to_f16_bits(X)).reshape(T + 1, D)
            rows.append(gate_layer(f"H{H}/hd{hd}/dff{d_ffn}/T{T}", X, w,
                                   at.TIER_CQ8))
    return rows


def report_c(rows: list[dict]) -> None:
    print(f"  {'case':28s} {'e2e_max':>10s} {'e2e/|y|':>9s} "
          f"{'B_layer':>10s} {'B/|y|':>9s}  verdict")
    for r in rows:
        v = "PASS" if not r["fails"] else "FAIL<" + r["fails"][0].split()[0] + ">"
        print(f"  {r['name']:28s} {r['e2e_max']:10.3e} {r['e2e_rel']:9.2e} "
              f"{r['b_comp']:10.3e} {r['b_comp_rel']:9.2e}  {v}")
    print("  " + "-" * 84)
    print("  e2e/|y| = layer-output error relative to max|y_ref| (the layer "
          "signal scale); B_layer = derived composition budget.")


# ═══════════════════════════ D. DETERMINISM ══════════════════════════════════

def section_d() -> None:
    print("[D] determinism — fixed seeds reproduce bit-identical intermediates")
    rng = np.random.default_rng(0xD37E4)
    w = make_weights(rng, 2, 64, 256)
    X = rng.normal(0, 1, (33, 128))
    X *= 4.0 / np.max(np.abs(X))
    X = f16_bits_to_f64(f64_to_f16_bits(X)).reshape(33, 128)
    r1 = tf.decoder_layer_fx(X, w, at.TIER_CQ8)
    r2 = tf.decoder_layer_fx(X, w, at.TIER_CQ8)
    same = (np.array_equal(r1.r2, r2.r2) and np.array_equal(r1.h, r2.h)
            and np.array_equal(r1.q_rope, r2.q_rope)
            and all(np.array_equal(a.o8, b.o8)
                    for a, b in zip(r1.heads, r2.heads)))
    check("decoder_layer_fx is bit-deterministic (two runs identical)", same)
    # LUT tables are pure functions of the contract (regeneration-stable)
    cb, cs, sb, ss = tf._cos_sin_lut_tables()
    cb2, _, _, _ = tf._cos_sin_lut_tables()
    check("cos/sin LUT tables regenerate identically",
          np.array_equal(cb, cb2))
    b, s = tf._silu_lut_tables()
    b2, _ = tf._silu_lut_tables()
    check("SiLU LUT tables regenerate identically", np.array_equal(b, b2))


def main() -> int:
    section_a()
    section_b()
    rows = section_c()
    report_c(rows)
    section_d()
    print("=" * 88)
    if FAILS:
        print(f"FAIL: {len(FAILS)} checks/budgets blown:")
        for f in FAILS[:20]:
            print(f"  ✗ {f}")
        return 1
    print(f"APEX DECODER-LAYER GOLDEN: ALL RoPE/SiLU LEMMAS + {len(rows)} LAYER "
          f"CASES WITHIN DERIVED BUDGET, DETERMINISM OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
