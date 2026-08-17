"""test_7b_plumbing.py — S6 golden gate: GQA + the 7B plumbing contracts.

Covers the five 7B-plumbing items, each as a golden contract with
the house acceptance discipline (isolated gates + D-023 5%-of-signal-scale):

  A. C-KSPLIT  — gemm_i8_ksplit: host K-split INT32 accumulation (K > 2048;
     7B shapes 3584 / 18944), N-split at the DIM_W=12 field limit. Gate:
     BIT-IDENTICAL to a single wide integer matmul; INT32 prefix asserts;
     legality rejects.
  B. C-RMSW    — rmsnorm_fx_wide: wide-D RMSNorm (hidden 3584 = 28·128).
     Gate: BIT-IDENTICAL to rmsnorm_fx on the frozen D≤128 pow-2 envelope;
     |mean2 − floor(sum2/D)| ≤ 1 (the μ_D scalar-multiply divider
     replacement); vs rmsnorm_ref within the [RMS] analytic bound at D=3584.
  C. C-CHUNK   — attention_chunked: T > T_ROW_MAX=128 online-softmax chunk
     merge (exact merge identity on the ASU's (mmax, lsum) state, exp LUT
     rescale). Gate: D-023 (≤5% of max|V_ref|) vs attention_ref_core for
     CQ-8/CQ-4+ (CQ-4 reported, not gated — D-022 fragility register);
     merge-weight invariants; chunk-vs-unchunked fx delta reported.
  D. GQA       — decoder_layer_fx/ref with H_kv < H (incl. MQA H_kv=1) at
     D_model=128: isolated rmsnorm + per-query-head D-023 + e2e 5%
     acceptance. The GQA-semantics anchor: a GQA layer is asserted
     BIT-IDENTICAL to its repeat_kv-EXPANDED MHA form (KV-head columns
     duplicated per query head, H_kv=H — exactly HF's repeat_kv). The
     legacy-behavior anchor is the unchanged 9/9 test_transformer suite
     (same `make test`); the H_kv=0-vs-H check here only pins the default
     resolution. Qwen QKV biases: bias cases gated the same way; bias=None
     vs zero-bias asserted bit-identical. (Checkpoint fact, verified from
     the Qwen2.5-7B safetensors index: q/k/v_proj carry `.bias` tensors;
     o_proj/gate/up/down do NOT.)
  E. 7B GEOMETRY — one full decoder-layer decode step at the Qwen2.5-7B
     shape: D_model=3584, H=28, H_kv=4, head_dim=128, d_ffn=18944, QKV
     biases on, CQ-8, T=8. Exercises every S6 item at once (wide RMSNorm,
     K-split 3584 & 18944, N-split 18944, GQA 7:1, biases). Gates: isolated
     rmsnorm1, all 28 heads D-023, e2e 5% acceptance.
  F. DETERMINISM — the new paths reproduce bit-identically.
  G. Q-POS (S8) — the HF SELF-INCLUSIVE decode composition: token t attends
     to keys 0..t (itself included) with q at RoPE position t, realized as
     X = [x_0..x_t, x_t] (decode row duplicated) + q_pos=t. Gates:
     q_pos=None/T bit-identical to legacy; decoder_layer_ref under the
     composition equals an INDEPENDENTLY-WRITTEN HF-style float64 decode
     step (natural indexing, no duplication trick); the fx composition
     passes the same isolated + D-023 + e2e gates; the T=1 first-token job
     (self-attention only) is legal and gated.
  H. LAYER-CHUNK (S8) — decoder_layer_fx dispatches T > T_ROW_MAX heads to
     attention_chunked (per-chunk verified tile jobs + softmax merge,
     ChunkedHead). Gates: T ≤ 128 stays on attention_core (type-pinned);
     per-query-head D-023 vs attention_ref_core at T=200; e2e 5% vs
     decoder_layer_ref; the combined q_pos+chunk composition (the S8
     run_tinynpu streaming form) gated at T=140.

Composition-budget derivation note: the analytic B_layer machinery of
test_transformer.py is derived for the MHA cases and stays their gate; the
new GQA/7B cases are gated by the ISOLATED stage bounds plus the D-023
relative acceptance (the same absolute Layer-3 bar, ARCHITECTURE.md D-023).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex_golden import attention as at                             # noqa: E402
from apex_golden import compute as cp                               # noqa: E402
from apex_golden import transformer as tf                           # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits         # noqa: E402

D023_FRAC = 0.05
D_ALL = 3.0
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def _f16g(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return f16_bits_to_f64(f64_to_f16_bits(x)).reshape(x.shape)


def rms_bound(x8_row: np.ndarray, gamma: np.ndarray, den: float) -> np.ndarray:
    """[RMS] per-element bound (test_transformer form, D_ALL slack)."""
    g_abs = np.abs(np.asarray(gamma, dtype=np.float64)) / (1 << 13)
    u = math.sqrt(float(np.mean(x8_row.astype(np.float64) ** 2)))
    assert u >= 15, f"[RMS] validity: degenerate row (u={u:.2f})"
    u_lb = max(u - 2.0, 1.0)
    return (g_abs * (np.abs(x8_row) * (2.0 ** -13 + D_ALL / (den * u_lb))
                     + 0.57 / u_lb)) + 2.0 ** -9


# ═══════════════════ A. C-KSPLIT — host K-split GEMM ═════════════════════════

def section_a() -> None:
    print("[A] C-KSPLIT — gemm_i8_ksplit vs single wide integer matmul")
    rng = np.random.default_rng(0x7B0A)

    # A-1 single-chunk regime (K ≤ 2048): bit-identical to gemm_i8
    ok = True
    for K in (1, 7, 128, 2048):
        A = rng.integers(-128, 128, (3, K))
        B = rng.integers(-128, 128, (K, 5))
        g = cp.gemm_i8(A, B)
        s = cp.gemm_i8_ksplit(A, B)
        ok &= np.array_equal(g, s) and g.dtype == s.dtype
    check("K ≤ K_MAX: ksplit == gemm_i8 bit-identical (incl. dtype)", bool(ok))

    # A-2 K-split regime (7B shapes included): == exact int64 matmul
    ok = True
    for K in (2049, 3584, 4096, 6000, 18944):
        A = rng.integers(-128, 128, (2, K))
        B = rng.integers(-128, 128, (K, 3))
        direct = A.astype(np.int64) @ B.astype(np.int64)
        assert np.all(np.abs(direct) < 2 ** 31)
        ok &= np.array_equal(cp.gemm_i8_ksplit(A, B).astype(np.int64), direct)
    check("K-split (2049..18944): bit-identical to wide matmul", bool(ok))

    # A-3 N-split at the 12-bit descriptor limit (concat, exact)
    A = rng.integers(-128, 128, (2, 100))
    B = rng.integers(-128, 128, (100, 9000))
    direct = A.astype(np.int64) @ B.astype(np.int64)
    check("N-split (N=9000 > 4095): bit-identical",
          bool(np.array_equal(cp.gemm_i8_ksplit(A, B).astype(np.int64),
                              direct)))

    # A-4 chunk-size invariance (integer-add associativity)
    A = rng.integers(-128, 128, (1, 5000))
    B = rng.integers(-128, 128, (5000, 4))
    check("k_chunk 512 vs 2048: identical results",
          bool(np.array_equal(cp.gemm_i8_ksplit(A, B, k_chunk=512),
                              cp.gemm_i8_ksplit(A, B, k_chunk=2048))))

    # A-5 worst-case density at the 7B down-proj K: all −128·−128
    A = np.full((1, 18944), -128, dtype=np.int64)
    B = np.full((18944, 1), -128, dtype=np.int64)
    acc = cp.gemm_i8_ksplit(A, B)
    check("all-(−128)² @ K=18944: acc = 18944·2^14 inside INT32",
          int(acc[0, 0]) == 18944 * 16384 and int(acc[0, 0]) < 2 ** 31,
          f"acc={int(acc[0,0])}")

    # A-6 legality rejects (host job splitter mirrors the descriptor checker)
    try:
        cp.gemm_i8_ksplit(np.zeros((1, 0), int), np.zeros((0, 1), int))
        check("K=0 rejected", False)
    except (ValueError, AssertionError):
        check("K=0 rejected", True)
    K_bad = cp.K_TOTAL_MAX + 1
    try:
        cp.gemm_i8_ksplit(np.zeros((1, K_bad), int), np.zeros((K_bad, 1), int))
        check(f"K={K_bad} > K_TOTAL_MAX rejected", False)
    except ValueError:
        check(f"K={K_bad} > K_TOTAL_MAX rejected", True)


# ═══════════════════ B. C-RMSW — wide-D RMSNorm ══════════════════════════════

def section_b() -> None:
    print("[B] C-RMSW — rmsnorm_fx_wide (hidden 3584 strategy)")
    rng = np.random.default_rng(0x7B0B)

    # B-1 frozen-envelope bit-identity (the regression anchor)
    ok = True
    for D in (1, 2, 4, 8, 16, 32, 64, 128):
        for _ in range(8):
            x = [int(v) for v in rng.integers(-128, 128, D)]
            g = [int(round((0.7 + 0.6 * rng.random()) * 8192))
                 for _ in range(D)]
            ok &= at.rmsnorm_fx(x, g) == at.rmsnorm_fx_wide(x, g)
    check("D ≤ 128 pow-2: rmsnorm_fx_wide == rmsnorm_fx (y, r, nrm)", bool(ok))

    # B-2 the μ_D divider replacement: |mean2 − floor(sum2/D)| ≤ 1
    mu, s = at._wide_mu(3584)
    worst = 0
    for _ in range(20000):
        sum2 = int(rng.integers(0, 3584 * 16384))
        worst = max(worst, abs(((sum2 * mu) >> (s + 16)) - sum2 // 3584))
    check("D=3584: |mean2_μ − floor(sum2/D)| ≤ 1 (20k random sum2)",
          worst <= 1, f"worst={worst}, μ={mu}, s={s}")

    # B-3 D=3584 vs rmsnorm_ref within the [RMS] analytic bound
    worst_ratio = 0.0
    ok = True
    for _ in range(20):
        x8 = np.clip(np.rint(rng.normal(0, 45, 3584)), -128, 127).astype(int)
        g = np.array([int(round((0.7 + 0.6 * rng.random()) * 8192))
                      for _ in range(3584)], dtype=np.int64)
        y, r, nrm = at.rmsnorm_fx_wide([int(v) for v in x8],
                                       [int(v) for v in g])
        got = np.asarray(y, dtype=np.float64) / 256.0
        ref = cp.rmsnorm_ref(x8.astype(np.float64),
                             g.astype(np.float64) / (1 << 13))
        bnd = rms_bound(x8, g, float(nrm))
        err = np.abs(got - ref)
        ok &= bool(np.all(err <= bnd))
        worst_ratio = max(worst_ratio, float(np.max(err / bnd)))
    check("D=3584: within [RMS] bound (20 rows)", ok,
          f"worst err/bound = {worst_ratio:.3f}")

    # B-4 other wide sizes stay inside the bound too
    ok = True
    for D in (256, 512, 1024):
        x8 = np.clip(np.rint(rng.normal(0, 45, D)), -128, 127).astype(int)
        g = np.array([8192] * D, dtype=np.int64)
        y, r, nrm = at.rmsnorm_fx_wide([int(v) for v in x8],
                                       [int(v) for v in g])
        got = np.asarray(y, dtype=np.float64) / 256.0
        ref = cp.rmsnorm_ref(x8.astype(np.float64),
                             g.astype(np.float64) / (1 << 13))
        ok &= bool(np.all(np.abs(got - ref) <= rms_bound(x8, g, float(nrm))))
    check("D ∈ {256, 512, 1024}: within [RMS] bound", bool(ok))

    # B-5 legality: non-multiple-of-chunk D rejected
    try:
        at.rmsnorm_fx_wide([0] * 3585, [8192] * 3585)
        check("D=3585 (not a chunk multiple) rejected", False)
    except AssertionError:
        check("D=3585 (not a chunk multiple) rejected", True)

    # B-6 envelope corner: all-(−128) at D=8192 (sum2 == 2^27 exactly) is
    # LEGAL and lands on the same mean2 as the D=128 all-(−128) corner
    y, r, nrm = at.rmsnorm_fx_wide([-128] * 8192, [8192] * 8192)
    y1, r1, n1 = at.rmsnorm_fx([-128] * 128, [8192] * 128)
    check("D=8192 all-(−128) corner legal; (r, nrm) == D=128 corner",
          (r, nrm) == (r1, n1), f"r={r} nrm={nrm}")


# ═══════════════════ C. C-CHUNK — T>128 online-softmax merge ═════════════════

def _top2_outliers(K16: np.ndarray) -> np.ndarray:
    """CQ-4+ static lanes = true top-2 |K| amax channels (house recipe)."""
    amax = np.max(np.abs(f16_bits_to_f64(K16).reshape(K16.shape)), axis=0)
    return np.sort(np.argsort(amax)[-2:])


def _gen_gauss(rng, T, D):
    """test_attention gen_gauss recipe: K,V max-abs 3.0; q max-abs 2.0."""
    K = rng.normal(0, 1, (T, D))
    V = rng.normal(0, 1, (T, D))
    q = rng.normal(0, 1, D)
    K *= 3.0 / np.max(np.abs(K))
    V *= 3.0 / np.max(np.abs(V))
    q *= 2.0 / np.max(np.abs(q))
    return (f64_to_f16_bits(K).reshape(T, D), f64_to_f16_bits(V).reshape(T, D),
            _f16g(q))


def section_c() -> None:
    print("[C] C-CHUNK — attention_chunked vs attention_ref_core (D-023)")
    rng = np.random.default_rng(0x7B0C)
    for D in (64, 128):
        for T in (129, 200, 384):
            K_f16, V_f16, q = _gen_gauss(rng, T, D)
            oi = _top2_outliers(K_f16)
            refc = at.attention_ref_core(q, K_f16, V_f16)
            vmax = float(np.max(np.abs(refc.V_ref)))
            for tier, gated in ((at.TIER_CQ8, True), (at.TIER_CQ4P, True),
                                (at.TIER_CQ4, False)):
                y, cores, mg = at.attention_chunked(
                    q, K_f16, V_f16, tier, outlier_idx=oi)
                err = float(np.max(np.abs(y - refc.y_ref)))
                if gated:
                    check(f"D{D}/T{T}/{tier}: chunked ≤ 5%·V-scale",
                          err <= D023_FRAC * vmax,
                          f"err={err:.3e} bar={D023_FRAC*vmax:.3e} "
                          f"chunks={len(cores)}")
                else:
                    print(f"  REPORT D{D}/T{T}/{tier}: err={err:.3e} "
                          f"(5% bar {D023_FRAC*vmax:.3e}; CQ-4 not gated, "
                          f"D-022)")
                if tier == at.TIER_CQ8:
                    # merge invariants + unchunked cross-check
                    un = at.attention_core(q, K_f16, V_f16, tier)
                    d_un = float(np.max(np.abs(y - un.out_hat)))
                    wsum = sum(mg["w"])
                    check(f"D{D}/T{T}: |Σw−1| ≤ 1e-3 ∧ m* = max(m_c)",
                          abs(wsum - 1.0) <= 1e-3
                          and mg["m_star"] == max(c.sm_m for c in cores),
                          f"Σw={wsum:.6f}, Δ(chunk,unchunked)={d_un:.3e}")
    # non-default chunk size (4 sub-jobs; groups per chunk base, D-008 runs)
    K_f16, V_f16, q = _gen_gauss(rng, 200, 64)
    refc = at.attention_ref_core(q, K_f16, V_f16)
    y, cores, _ = at.attention_chunked(q, K_f16, V_f16, at.TIER_CQ8, chunk=64)
    check("chunk=64 (4 jobs, T=200): ≤ 5%·V-scale",
          float(np.max(np.abs(y - refc.y_ref)))
          <= D023_FRAC * float(np.max(np.abs(refc.V_ref))),
          f"chunks={len(cores)}")


# ═══════════════════ D. GQA + Qwen QKV biases (D_model = 128) ════════════════

def make_weights_gqa(rng, H, head_dim, d_ffn, H_kv=0, biases=False):
    D = H * head_dim
    kv_dim = (H_kv or H) * head_dim
    def w(shape):
        return rng.integers(-127, 128, shape).astype(np.int64)
    s_proj = 1.0 / (127.0 * math.sqrt(D))
    s_down = 1.0 / (127.0 * math.sqrt(d_ffn))
    def gam():
        return np.array([int(round((0.7 + 0.6 * rng.random()) * 8192))
                         for _ in range(D)], dtype=np.int64)
    b = {}
    if biases:  # fp16-grid real biases, Qwen-style (q/k/v only)
        b = {"bq": _f16g(rng.normal(0, 0.25, D)),
             "bk": _f16g(rng.normal(0, 0.25, kv_dim)),
             "bv": _f16g(rng.normal(0, 0.25, kv_dim))}
    return tf.LayerWeights(
        Wq=w((D, D)), Wk=w((D, kv_dim)), Wv=w((D, kv_dim)), Wo=w((D, D)),
        Wg=w((D, d_ffn)), Wu=w((D, d_ffn)), Wd=w((d_ffn, D)),
        s_wq=s_proj, s_wk=s_proj, s_wv=s_proj, s_wo=s_proj,
        s_wg=s_proj, s_wu=s_proj, s_wd=s_down,
        gamma1=gam(), gamma2=gam(), H=H, head_dim=head_dim, H_kv=H_kv, **b)


def gate_gqa_layer(name: str, X, w, tier, q_pos=None) -> None:
    """Isolated rmsnorm1 + per-query-head D-023 + e2e 5% acceptance."""
    r = tf.decoder_layer_fx(X, w, tier, q_pos=q_pos)
    ref = tf.decoder_layer_ref(X, w, q_pos=q_pos)
    T, H, hd = r.T, w.H, w.head_dim
    group = H // (w.H_kv or H)

    x8, _ = at.quant_rows_i8(X)
    _, _, nrm1 = at.rmsnorm_fx_wide([int(v) for v in x8[T]],
                                    [int(v) for v in w.gamma1])
    h_fx = r.h[T].astype(np.float64) / 256.0
    e_rms = float(np.max(np.abs(h_fx - ref.h_ref[T])))
    b_rms = float(np.max(rms_bound(x8[T], w.gamma1, float(nrm1))))
    check(f"{name}: rmsnorm1 within [RMS]", e_rms <= b_rms,
          f"meas={e_rms:.3e} bound={b_rms:.3e}")

    worst_head = 0.0
    ok = True
    for h_i in range(H):
        sl_q = slice(h_i * hd, (h_i + 1) * hd)
        sl_kv = slice((h_i // group) * hd, (h_i // group + 1) * hd)
        K_f16 = f64_to_f16_bits(r.K_rope[:, sl_kv]).reshape(T, hd)
        V_f16 = f64_to_f16_bits(r.V_real[:, sl_kv]).reshape(T, hd)
        refc = at.attention_ref_core(r.q_rope[sl_q], K_f16, V_f16)
        e = float(np.max(np.abs(r.heads[h_i].out_hat - refc.y_ref)))
        bar = D023_FRAC * float(np.max(np.abs(refc.V_ref)))
        ok &= e <= bar
        worst_head = max(worst_head, e / bar)
    check(f"{name}: all {H} heads ≤ 5%·V-scale (D-023)", ok,
          f"worst head err/bar = {worst_head:.3f}")

    e2e = float(np.max(np.abs(r.r2 - ref.y_ref)))
    ynorm = float(np.max(np.abs(ref.y_ref)))
    check(f"{name}: e2e ≤ 5%·|y| acceptance", e2e <= D023_FRAC * ynorm,
          f"e2e={e2e:.3e} rel={e2e/ynorm:.2e}")


def section_d() -> None:
    print("[D] GQA decoder layer + Qwen QKV biases (D_model = 128)")
    rng = np.random.default_rng(0x7B0D)

    for (H, H_kv, hd, dff, T) in ((2, 1, 64, 256, 8), (2, 1, 64, 256, 32),
                                  (4, 2, 32, 256, 16)):
        w = make_weights_gqa(rng, H, hd, dff, H_kv=H_kv)
        X = rng.normal(0, 1, (T + 1, H * hd))
        X *= 4.0 / np.max(np.abs(X))
        X = _f16g(X)
        gate_gqa_layer(f"H{H}/kv{H_kv}/hd{hd}/T{T}", X, w, at.TIER_CQ8)

    # GQA == repeat_kv-expanded MHA (the HF-semantics anchor): duplicate each
    # KV head's Wk/Wv columns (and bk/bv entries) once per query head, run as
    # H_kv=H — must reproduce the GQA result bit-for-bit.
    import dataclasses
    H, H_kv, hd = 4, 2, 32
    wg = make_weights_gqa(rng, H, hd, 256, H_kv=H_kv, biases=True)
    rep = H // H_kv
    col = np.concatenate([np.arange((h // rep) * hd, (h // rep + 1) * hd)
                          for h in range(H)])
    wx = dataclasses.replace(wg, Wk=wg.Wk[:, col], Wv=wg.Wv[:, col],
                             bk=wg.bk[col], bv=wg.bv[col], H_kv=H)
    X = _f16g(rng.normal(0, 1, (9, H * hd)) * 2.0)
    rg = tf.decoder_layer_fx(X, wg, at.TIER_CQ8)
    rx = tf.decoder_layer_fx(X, wx, at.TIER_CQ8)
    check("GQA == repeat_kv-expanded MHA (r2/attn/q_rope + per-head o8 "
          "bit-identical)",
          np.array_equal(rg.r2, rx.r2) and np.array_equal(rg.attn, rx.attn)
          and np.array_equal(rg.q_rope, rx.q_rope)
          and all(np.array_equal(a.o8, b.o8)
                  for a, b in zip(rg.heads, rx.heads)))

    # default resolution only: H_kv=0 resolves to H (same instructions)
    w0 = make_weights_gqa(rng, 2, 64, 256, H_kv=0)
    w2 = dataclasses.replace(w0, H_kv=2)
    X2 = _f16g(rng.normal(0, 1, (9, 128)) * 2.0)
    check("H_kv=0 default resolves to H_kv=H",
          np.array_equal(tf.decoder_layer_fx(X2, w0, at.TIER_CQ8).r2,
                         tf.decoder_layer_fx(X2, w2, at.TIER_CQ8).r2))

    # Qwen-style biases: gated case + bias=None == zero-bias bit-identity
    wb = make_weights_gqa(rng, 2, 64, 256, H_kv=1, biases=True)
    Xb = _f16g(rng.normal(0, 1, (17, 128)) * 3.0)
    gate_gqa_layer("bias/H2/kv1/hd64/T16", Xb, wb, at.TIER_CQ8)
    wz = dataclasses.replace(wb, bq=np.zeros(128), bk=np.zeros(64),
                             bv=np.zeros(64))
    wn = dataclasses.replace(wb, bq=None, bk=None, bv=None)
    rz = tf.decoder_layer_fx(Xb, wz, at.TIER_CQ8)
    rn = tf.decoder_layer_fx(Xb, wn, at.TIER_CQ8)
    check("bias=None == zero-bias (bit-identical r2)",
          np.array_equal(rz.r2, rn.r2))
    print("  NOTE checkpoint fact (Qwen2.5-7B safetensors index): "
          "q/k/v_proj.bias present; o_proj/gate/up/down have NO bias.")


# ═══════════════════ E. 7B geometry — one full decode step ═══════════════════

def section_e() -> None:
    print("[E] 7B geometry — D=3584, H=28/kv4, hd=128, d_ffn=18944, biases, "
          "rope_theta 1e6 (Qwen2.5 config), CQ-8, T=8")
    rng = np.random.default_rng(0x7B0E)
    t0 = time.time()
    import dataclasses
    w = make_weights_gqa(rng, 28, 128, 18944, H_kv=4, biases=True)
    w = dataclasses.replace(w, rope_theta_base=1e6)   # Qwen2/2.5 rope_theta
    X = rng.normal(0, 1, (9, 3584))
    X *= 4.0 / np.max(np.abs(X))
    X = _f16g(X)
    gate_gqa_layer("7B/H28/kv4/hd128/dff18944/T8", X, w, at.TIER_CQ8)
    print(f"    (7B-geometry case ran in {time.time()-t0:.1f}s)")


# ═══════════════════ F. determinism ══════════════════════════════════════════

def section_f() -> None:
    print("[F] determinism — new paths reproduce bit-identically")
    rng = np.random.default_rng(0x7B0F)
    x = [int(v) for v in rng.integers(-128, 128, 3584)]
    g = [8192] * 3584
    check("rmsnorm_fx_wide deterministic",
          at.rmsnorm_fx_wide(x, g) == at.rmsnorm_fx_wide(x, g))
    K = f64_to_f16_bits(rng.normal(0, 2, (200, 64))).reshape(200, 64)
    V = f64_to_f16_bits(rng.normal(0, 2, (200, 64))).reshape(200, 64)
    q = _f16g(rng.normal(0, 2, 64))
    y1, _, m1 = at.attention_chunked(q, K, V, at.TIER_CQ8)
    y2, _, m2 = at.attention_chunked(q, K, V, at.TIER_CQ8)
    check("attention_chunked deterministic",
          np.array_equal(y1, y2) and m1 == m2)
    w = make_weights_gqa(rng, 2, 64, 256, H_kv=1, biases=True)
    X = _f16g(rng.normal(0, 1, (9, 128)) * 2.0)
    r1 = tf.decoder_layer_fx(X, w, at.TIER_CQ8)
    r2 = tf.decoder_layer_fx(X, w, at.TIER_CQ8)
    check("GQA+bias decoder_layer_fx deterministic",
          np.array_equal(r1.r2, r2.r2))


# ═══════════════════ G. q_pos — HF self-inclusive decode (S8) ════════════════

def _hf_decode_ref(rows: np.ndarray, w) -> np.ndarray:
    """Independently-written HF/Qwen float64 decode step (the G anchor).

    rows: [t+1, D] activations of ALL tokens so far, current token last.
    Natural indexing, NO duplicated-row trick: q from rows[t] at RoPE
    position t; K/V from rows 0..t (self-inclusive causal window).
    Same dequantized weights and float64 stages as decoder_layer_ref.
    """
    rows = np.asarray(rows, dtype=np.float64)
    t = rows.shape[0] - 1
    H, hd = w.H, w.head_dim
    H_kv = w.H_kv or H
    group = H // H_kv
    g1 = np.asarray(w.gamma1, dtype=np.float64) / (1 << 13)
    h = cp.rmsnorm_ref(rows, g1)
    q = h[t] @ (np.asarray(w.Wq, np.float64) * w.s_wq)
    K = h @ (np.asarray(w.Wk, np.float64) * w.s_wk)
    V = h @ (np.asarray(w.Wv, np.float64) * w.s_wv)
    if w.bq is not None:
        q = q + np.asarray(w.bq, np.float64)
    if w.bk is not None:
        K = K + np.asarray(w.bk, np.float64)[None, :]
    if w.bv is not None:
        V = V + np.asarray(w.bv, np.float64)[None, :]
    theta = tf.rope_theta(hd, w.rope_theta_base)
    attn = np.empty(H * hd, dtype=np.float64)
    for h_i in range(H):
        sl_q = slice(h_i * hd, (h_i + 1) * hd)
        sl_kv = slice((h_i // group) * hd, (h_i // group + 1) * hd)
        q_h = tf.rope_ref(q[sl_q], t, theta)
        K_h = tf.rope_ref(K[:, sl_kv], np.arange(t + 1), theta)
        p = cp.softmax_ref(K_h @ q_h / math.sqrt(float(hd)))
        attn[sl_q] = p @ V[:, sl_kv]
    r1 = rows[t] + attn @ (np.asarray(w.Wo, np.float64) * w.s_wo)
    g2 = np.asarray(w.gamma2, dtype=np.float64) / (1 << 13)
    h2 = cp.rmsnorm_ref(r1, g2)
    gate = h2 @ (np.asarray(w.Wg, np.float64) * w.s_wg)
    up = h2 @ (np.asarray(w.Wu, np.float64) * w.s_wu)
    ffn = (tf.silu_ref(gate) * up) @ (np.asarray(w.Wd, np.float64) * w.s_wd)
    return r1 + ffn


def section_g() -> None:
    print("[G] q_pos — HF self-inclusive decode composition (S8)")
    rng = np.random.default_rng(0x7B06)

    # G-1 default neutrality: q_pos=None ≡ explicit q_pos=T, bit-identical
    w = make_weights_gqa(rng, 2, 64, 256, H_kv=1, biases=True)
    X = _f16g(rng.normal(0, 1, (9, 128)) * 2.0)
    rn = tf.decoder_layer_fx(X, w, at.TIER_CQ8)
    rt = tf.decoder_layer_fx(X, w, at.TIER_CQ8, q_pos=8)
    check("q_pos=None == q_pos=T (bit-identical r2/attn/q_rope)",
          np.array_equal(rn.r2, rt.r2) and np.array_equal(rn.attn, rt.attn)
          and np.array_equal(rn.q_rope, rt.q_rope))

    # G-2 the composition anchor: decoder_layer_ref on [rows; rows[-1]] with
    # q_pos=t equals the independently-written HF decode step (float64)
    ok = True
    worst = 0.0
    for t in (0, 3, 15):
        rows = _f16g(rng.normal(0, 1, (t + 1, 128)) * 2.0)
        X_hf = np.vstack([rows, rows[-1:]])
        y_comp = tf.decoder_layer_ref(X_hf, w, q_pos=t).y_ref
        y_hf = _hf_decode_ref(rows, w)
        d = float(np.max(np.abs(y_comp - y_hf)))
        scale = float(np.max(np.abs(y_hf)))
        ok &= d <= 1e-12 * scale
        worst = max(worst, d / scale)
    check("duplicated-row + q_pos=t == independent HF decode ref "
          "(t ∈ {0,3,15})", ok, f"worst rel Δ = {worst:.2e}")

    # G-3 the fx composition passes the standard gates (incl. GQA + biases)
    rows = _f16g(rng.normal(0, 1, (17, 128)) * 2.0)
    X_hf = np.vstack([rows, rows[-1:]])
    gate_gqa_layer("hf-decode/H2/kv1/hd64/t16", X_hf, w, at.TIER_CQ8,
                   q_pos=16)

    # G-4 first token: T=1 self-attention-only job is legal and gated
    rows0 = _f16g(rng.normal(0, 1, (1, 128)) * 2.0)
    X0 = np.vstack([rows0, rows0[-1:]])
    r0 = tf.decoder_layer_fx(X0, w, at.TIER_CQ8, q_pos=0)
    p0 = r0.heads[0].p_q115
    check("t=0: single-key softmax saturates (p == 2^15 − ε, T=1)",
          len(p0) == 1 and abs(int(p0[0]) - (1 << 15)) <= 1,
          f"p={int(p0[0])}")
    y0 = tf.decoder_layer_ref(X0, w, q_pos=0).y_ref
    e0 = float(np.max(np.abs(r0.r2 - y0)))
    check("t=0: e2e ≤ 5%·|y|", e0 <= D023_FRAC * float(np.max(np.abs(y0))),
          f"e2e={e0:.3e}")


# ═══════════════════ H. C-CHUNK in the decoder layer (S8) ════════════════════

def section_h() -> None:
    print("[H] layer-chunk — decoder_layer_fx dispatch to attention_chunked")
    rng = np.random.default_rng(0x7B08)
    w = make_weights_gqa(rng, 2, 64, 256, H_kv=1, biases=True)

    # H-1 envelope boundary: T = 128 stays on the unchunked attention_core
    X128 = _f16g(rng.normal(0, 1, (129, 128)) * 2.0)
    r128 = tf.decoder_layer_fx(X128, w, at.TIER_CQ8)
    check("T=128: heads are unchunked AttnCore (legacy path type-pinned)",
          all(isinstance(h, at.AttnCore) for h in r128.heads))

    # H-2 T=200: ChunkedHead per head, per-head D-023 + e2e 5% acceptance
    X = _f16g(rng.normal(0, 1, (201, 128)) * 2.0)
    r = tf.decoder_layer_fx(X, w, at.TIER_CQ8)
    ref = tf.decoder_layer_ref(X, w)
    check("T=200: heads are ChunkedHead with 2 chunk jobs each",
          all(isinstance(h, tf.ChunkedHead) and len(h.cores) == 2
              for h in r.heads))
    ok, worst = True, 0.0
    group = w.H // w.H_kv
    for h_i in range(2):
        sl_q = slice(h_i * 64, (h_i + 1) * 64)
        sl_kv = slice((h_i // group) * 64, (h_i // group + 1) * 64)
        K_f16 = f64_to_f16_bits(r.K_rope[:, sl_kv]).reshape(200, 64)
        V_f16 = f64_to_f16_bits(r.V_real[:, sl_kv]).reshape(200, 64)
        refc = at.attention_ref_core(r.q_rope[sl_q], K_f16, V_f16)
        e = float(np.max(np.abs(r.heads[h_i].out_hat - refc.y_ref)))
        bar = D023_FRAC * float(np.max(np.abs(refc.V_ref)))
        ok &= e <= bar
        worst = max(worst, e / bar)
    check("T=200: all heads ≤ 5%·V-scale (D-023, chunk-merged)", ok,
          f"worst head err/bar = {worst:.3f}")
    e2e = float(np.max(np.abs(r.r2 - ref.y_ref)))
    ynorm = float(np.max(np.abs(ref.y_ref)))
    check("T=200: e2e ≤ 5%·|y| acceptance", e2e <= D023_FRAC * ynorm,
          f"e2e={e2e:.3e} rel={e2e/ynorm:.2e}")
    wsum = sum(r.heads[0].merge["w"])
    check("T=200: merge invariant |Σw−1| ≤ 1e-3", abs(wsum - 1.0) <= 1e-3,
          f"Σw={wsum:.6f}")

    # H-3 the S8 streaming form: q_pos + chunking combined (t=139, T=140)
    rows = _f16g(rng.normal(0, 1, (140, 128)) * 2.0)
    X_hf = np.vstack([rows, rows[-1:]])
    gate_gqa_layer("chunk+q_pos/H2/kv1/hd64/t139", X_hf, w, at.TIER_CQ8,
                   q_pos=139)


def main() -> int:
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    section_h()
    print("=" * 88)
    if FAILS:
        print(f"FAIL: {len(FAILS)} checks blown:")
        for f in FAILS[:20]:
            print(f"  ✗ {f}")
        return 1
    print("APEX 7B-PLUMBING GOLDEN (S6+S8): C-KSPLIT + C-RMSW + C-CHUNK + "
          "GQA + QKV-BIAS + 7B-GEOMETRY + Q-POS ALL WITHIN GATES, "
          "DETERMINISM OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
