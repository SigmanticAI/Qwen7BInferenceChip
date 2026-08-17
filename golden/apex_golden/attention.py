"""attention.py — golden END-TO-END attention chain (ARCHITECTURE.md §0 + §9b).

Implements the EXACT fixed-point decode-step chain, D-021 (the attention
numeric seam), composing ONLY already-proven primitives:

  x  --(Q0: C-1 per-token INT8 quant)-->  x8
     --(Q1: rmsnorm_fx, the ASU NORMATIVE contract)-->  h  Q7.8
     --(Q2: C-1 per-token INT8 quant of h/2^8)-->  h8, s_h
     --(Q3: MXE gemm_i8 WS projections, INT32 acc)--> q/k/v accumulators
     --(Q4: KV write-port dequant acc*s_h*s_w -> fp16 RNE)--> K_f16, V_f16
     --(Q5: KVQ compress/decompress, D-001 contract)--> K_hat, V_hat (fp32 bus)
     --(Q6: D-021 feeder requant: per-token C-1 INT8)--> k8/s_k, v8/s_v
     --(Q7: Q feeder quant of the raw q row, same machinery)--> q8, s_q
     --(Q8: MXE-OS gemm_i8 Q.K_hat^T, INT32 acc)--> acc_s
     --(Q9: D-021 score-dequant: acc*(s_q*s_k[t])/sqrt(D) -> Q.SCORE_FRAC)-->
     --(Q10: ASU online_softmax_fx)--> p Q1.15
     --(Q11: D-021 P requant: c[t]=p[t]*s_v[t], per-row C-1 INT8)--> c8, s_c
     --(Q12: MXE-OS gemm_i8 P.V_hat, INT32 acc)--> acc_o
     --(Q13: C-2 requant epilogue INT32->INT8)--> o8, out scale s_out

Every step is bit-deterministic: pure integers, or float64 arithmetic on
exact fp16/fp32 values with single documented RNE narrowings (fp.py rules).

SEAM RESOLUTIONS documented here (the golden model is the arbiter; each is a
candidate for promotion into ARCHITECTURE.md when apex_top integration lands):

  S-1  RMSNorm->MXE: the Q7.8 hidden vector is per-token C-1 INT8 quantized
       (amax -> fp16 scale s_h, RNE, clamp [-128,127]) at the activation
       feeder. Same machinery as the D-021 K_hat feeder.
  S-2  K/V projections run with requant_en=0 (raw INT32 accumulators); the
       KV-cache WRITE PORT dequantizes acc*(s_h[t]*s_w) to fp16 (one RNE
       narrowing) because the KVQ codec's contract input is fp16 (D-001).
       No gratuitous INT8 epilogue in front of the codec's own quantizer.
  S-3  Score-dequant (D-021 stage) normative arithmetic: one fp32 composite
       constant per column, c_t = f32(s_q * s_k[t] * 2^SCORE_FRAC / sqrt(D)),
       then score_fx[t] = RNE(acc[t] * c_t) saturating-asserted to INT32.
       1/sqrt(D) (softmax temperature) is folded into the composite.
  S-4  P.V_hat scale folding: per-token V_hat scales s_v[t] sit on the
       CONTRACTION axis of P.V_hat, so they cannot be applied in the INT32
       epilogue. D-021's "output descale folds both into the epilogue" is
       realized by folding s_v[t] into the probability BEFORE P-requant:
       c[t] = p[t]*s_v[t]; c is per-row C-1 INT8 quantized (scale s_c); the
       single epilogue scale s_c then carries both the P and V descales.
       (The literal per-column-epilogue reading is numerically impossible
       for contraction-axis scales — this fold is the faithful resolution.)
  S-5  SCORE_FRAC = 10 (= apex_pkg ASU_IN_FRAC, the verified f10 ASU config):
       the exp LUT interpolation gets full resolution, SH = 0.

The float64 reference (`attention_ref`) is the yardstick: same inputs, same
DEQUANTIZED weights (w8*s_w exactly), rmsnorm_ref/softmax_ref — so measured
error isolates the D-021 chain (activation/KV quantization, codec, softmax
fixed point), not weight quantization of some pretrained model.

rmsnorm_fx below REPLICATES the normative golden mirror from
rtl/asu/asu_rmsnorm.sv's header (mirrored in verif/asu/smoke/
gen_asu_vectors.py); tests/test_attention.py cross-checks the two
implementations for exact equality — divergence is a build failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cq_codec import (
    EPS, compress_keys, compress_values, decompress_keys, decompress_values,
)
from .compute import (
    FRAC_BITS as EXP_FRAC_BITS, _clamp_z, exp_fx, gemm_i8, gemm_i8_ksplit,
    online_softmax_fx, requant_i32_to_i8, rmsnorm_ref, softmax_ref,
)
from .fp import f16_bits_to_f64, f32_bits_to_f64, f64_to_f16_bits, rne

SCORE_FRAC = 10          # S-5: matches ASU_IN_FRAC (SH = 0), verified f10 config
OUT_FRAC = 15            # ASU Q1.15 probabilities (apex_pkg ASU_OUT_FRAC)
CHUNK_T_MAX = 128        # F-1 T_ROW_MAX: the per-job attention-row envelope


# ── Q1: RMSNorm fixed-point (NORMATIVE replica — see module docstring) ───────

def rmsnorm_fx(x: list[int], g: list[int]) -> tuple[list[int], int, int]:
    """EXACT replica of the asu_rmsnorm.sv header pseudocode / the verifier's
    oracle in verif/asu/smoke/gen_asu_vectors.py. Do not 'improve'."""
    d = len(x)
    assert 1 <= d <= 128 and (d & (d - 1)) == 0, "D must be pow-2 in [1,128]"
    assert all(-128 <= xi <= 127 for xi in x)
    assert all(-32768 <= gi <= 32767 for gi in g) and len(g) == d
    sum2 = sum(int(xi) ** 2 for xi in x)
    assert sum2 < 2 ** 22                       # width proof (<= 2^21)
    mean2 = sum2 >> int(math.log2(d))           # exact floor (pow-2 D)
    n2 = mean2 + 1                              # EPS_INT = 1
    den = math.isqrt(n2)                        # rsqrt_unit phase A (floor)
    r = (1 << 13) // den                        # phase B (floor div), UQ2.13
    nrm = min(den, 2 ** 14 - 1)                 # norm_int output
    y = []
    for i in range(d):
        p = int(x[i]) * int(g[i])               # Q2.13
        q = p * r                               # frac 26
        t = q >> 18                             # floor to Q7.8
        rem = q & (2 ** 18 - 1)
        t += 1 if (rem > 2 ** 17 or (rem == 2 ** 17 and (t & 1))) else 0
        y.append(max(-32768, min(32767, t)))    # sat16
    return y, r, nrm


# ── C-RMSW (S6): wide-D RMSNorm strategy — hidden = 3584 = 28·128 ────────────

def _wide_mu(d: int) -> tuple[int, int]:
    """C-RMSW scalar constants: mean2 = (sum2·mu) >> (s+16).

    s = ceil(log2(d)) (so 2^s is the next power of two); mu = RNE(2^16·2^s/d)
    corrects the pow-2 shift to the true divisor with ONE 16-bit scalar
    multiply on the sum path — no divider, gamma untouched. For pow-2 d,
    mu = 2^16 exactly and the arithmetic REDUCES BIT-EXACTLY to rmsnorm_fx's
    `sum2 >> log2(d)`. mu rounding perturbs mean2 by ≤ 2^-17 relative —
    orders below the r = 2^13//den truncation already in the contract.
    """
    s = max(0, (int(d) - 1).bit_length())
    mu = int(rne(np.array([(1 << 16) * float(1 << s) / float(d)]))[0])
    return mu, s


def rmsnorm_fx_wide(x: list[int], g: list[int], chunk: int = 128
                    ) -> tuple[list[int], int, int]:
    """Wide-D RMSNorm (C-RMSW): the S6 strategy for hidden sizes beyond the
    frozen D≤128 pow-2 envelope (7B hidden = 3584 = 28 chunks of 128).

    Composition, hardware-honest:
      1. the row is cut into pow-2 chunks of ≤128; each chunk's sum-of-squares
         obeys the SAME 22-bit width proof as asu_rmsnorm.sv (asserted);
      2. the host (or a scalar adder) accumulates chunk sums — INT32-bounded
         for D ≤ 8192 (asserted: sum2 ≤ 2^27, the all-(−128) D=8192 corner);
      3. mean2 = (sum2·mu_D) >> (s+16) (see _wide_mu) — one scalar multiply
         replaces the divider; mean2 lands in the SAME [0, 2^14] range the
         31-cycle rsqrt_unit already takes, so phases A/B are reused as-is;
      4. the per-element Q2.13·r multiply + >>18 RNE + sat16 phase is the
         UNCHANGED rmsnorm_fx element datapath, run per chunk with the one
         broadcast r.
    For D ≤ 128 pow-2 this is BIT-IDENTICAL to rmsnorm_fx (tested). RTL
    deltas implied (backlog, not S6): export sum2 / accept external r.
    """
    d = len(x)
    if d <= 128 and (d & (d - 1)) == 0:
        chunks = [x]                            # the frozen envelope, unchanged
    else:
        assert 1 <= chunk <= 128 and (chunk & (chunk - 1)) == 0, \
            "chunk must be pow-2 in [1,128]"
        assert d % chunk == 0, f"wide D={d} must be a multiple of chunk={chunk}"
        chunks = [x[a:a + chunk] for a in range(0, d, chunk)]
    assert all(-128 <= xi <= 127 for xi in x)
    assert all(-32768 <= gi <= 32767 for gi in g) and len(g) == d
    sum2 = 0
    for c in chunks:
        s2c = sum(int(xi) ** 2 for xi in c)
        assert s2c < 2 ** 22                    # per-chunk width proof (22b)
        sum2 += s2c
    assert sum2 <= 2 ** 27, "C-RMSW host accumulator bound (D ≤ 8192)"
    # (<= : the all-(−128) row at D=8192 lands EXACTLY on 2^27 — legal; still
    #  16x inside INT32, and mean2 then equals the D=128 all-(−128) corner)
    mu, s = _wide_mu(d)
    mean2 = (sum2 * mu) >> (s + 16)             # == sum2 >> log2(d) for pow-2 d
    n2 = mean2 + 1                              # EPS_INT = 1 (same rsqrt range)
    den = math.isqrt(n2)
    r = (1 << 13) // den                        # UQ2.13, unchanged phase B
    nrm = min(den, 2 ** 14 - 1)
    y = []
    for i in range(d):
        p = int(x[i]) * int(g[i])               # Q2.13
        q = p * r                               # frac 26
        t = q >> 18                             # floor to Q7.8
        rem = q & (2 ** 18 - 1)
        t += 1 if (rem > 2 ** 17 or (rem == 2 ** 17 and (t & 1))) else 0
        y.append(max(-32768, min(32767, t)))    # sat16
    return y, r, nrm


# ── C-1 per-token INT8 quantization (feeder machinery, D-021) ────────────────

def quant_rows_i8(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8: amax -> fp16 scale (RNE), q = clamp(RNE(x/s)).

    X: [T, D] float64 EXACT values (fp16/fp32-representable or fixed-point).
    Returns (codes int64 [T,D] in [-128,127], scales uint16 fp16 bits [T]).

    For fp16-grid inputs this is BIT-IDENTICAL to cq_codec.compress_values
    (bits=8) codes+scales — proven by tests/test_attention.py section A. The
    fp32->fp16 amax path (feeder from the KVQ fp32 bus) takes the exact f64
    amax into the same s = f16(max(amax/127, EPS)) rule.
    """
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]
    codes = np.empty(X.shape, dtype=np.int64)
    scales = np.empty(T, dtype=np.uint16)
    for t in range(T):
        amax = float(np.max(np.abs(X[t]), initial=0.0))
        s = max(amax / 127.0, EPS)
        sb = np.uint16(f64_to_f16_bits(np.array([s]))[0])
        sv = float(f16_bits_to_f64(np.array([sb]))[0])
        codes[t] = np.clip(rne(X[t] / sv), -128, 127)
        scales[t] = sb
    return codes, scales


# ── C-2 epilogue calibration (deterministic scale/shift pick) ────────────────

def calib_requant(amax_acc: int, target: int = 126) -> tuple[int, int]:
    """Pick (rq_scale, rq_shift) mapping |acc| = amax_acc to ~`target` counts.

    Maximizes rq_shift (precision) subject to rq_scale <= 65535, shift <= 31.
    target = 126 (not 127) leaves RNE headroom so sat8 never clips (asserted
    by the caller): amax*scale/2^shift <= 126*(1+0.5/scale) < 127.5.
    """
    assert amax_acc >= 1
    ratio = target / float(amax_acc)
    shift = min(31, int(math.floor(math.log2(65535.0 / ratio))))
    scale = int(rne(np.array([ratio * (1 << shift)]))[0])
    scale = max(1, min(65535, scale))
    return scale, shift


# ── Q9: D-021 score-dequant stage (S-3 normative arithmetic) ─────────────────

def score_dequant_fx(acc: np.ndarray, s_q_f16: np.uint16, s_k_f16: np.ndarray,
                     D: int, score_frac: int = SCORE_FRAC) -> np.ndarray:
    """score_fx[t] = RNE(acc[t] * f32(s_q*s_k[t]*2^frac/sqrt(D))), INT32-checked.

    One fp32 composite constant per column (computed once per tile in RTL);
    the int32 x fp32 product + RNE-to-integer is the stage's only rounding.
    """
    acc = np.asarray(acc, dtype=np.int64)
    s_q = float(f16_bits_to_f64(np.array([s_q_f16]))[0])
    s_k = f16_bits_to_f64(np.asarray(s_k_f16, dtype=np.uint16))
    comp64 = s_q * s_k * (float(1 << score_frac) / math.sqrt(float(D)))
    comp = comp64.astype(np.float32).astype(np.float64)   # single f32 narrowing
    score = rne(acc.astype(np.float64) * comp)
    assert np.all(np.abs(score) < 2 ** 31), "score-dequant INT32 overflow"
    return score


# ── KVQ tier dispatch (D-001 codec, vendored-proven) ─────────────────────────

TIER_CQ8, TIER_CQ4, TIER_CQ4P = "CQ-8", "CQ-4", "CQ-4+"
TIERS = (TIER_CQ8, TIER_CQ4, TIER_CQ4P)


def kvq_roundtrip(K_f16: np.ndarray, V_f16: np.ndarray, tier: str, G: int,
                  outlier_idx) -> tuple[np.ndarray, np.ndarray, object, object]:
    """Compress+decompress K,V per tier -> (K_hat, V_hat f64 values, kb, vb)."""
    if tier == TIER_CQ8:
        kb = compress_values(K_f16, 8)
        vb = compress_values(V_f16, 8)
        K_hat = f32_bits_to_f64(decompress_values(kb)).reshape(K_f16.shape)
    elif tier in (TIER_CQ4, TIER_CQ4P):
        oi = np.asarray(outlier_idx if tier == TIER_CQ4P else [], dtype=np.int64)
        kb = compress_keys(K_f16, 4, G, oi)
        vb = compress_values(V_f16, 4)
        K_hat = f32_bits_to_f64(decompress_keys(kb)).reshape(K_f16.shape)
    else:
        raise ValueError(tier)
    V_hat = f32_bits_to_f64(decompress_values(vb)).reshape(V_f16.shape)
    return K_hat, V_hat, kb, vb


def kvq_roundtrip_tiermap(K_f16: np.ndarray, V_f16: np.ndarray,
                          tier_map: list[str], G: int, outlier_idx
                          ) -> tuple[np.ndarray, np.ndarray, list, list]:
    """D-022 actuation golden: PER-ROW tier map (len T, entries in TIERS).

    Contiguous runs of equal tier are compressed INDEPENDENTLY: each run is
    its own key-group sequence starting at the run base (grouped in G-token
    groups; a partial tail group freezes at its size — §3.1 / D-008 flush
    semantics). Values quantize per token at the row's tier (identical for
    CQ-4 and CQ-4+; INT8 for CQ-8). This mirrors the apex_top tier-bank
    programming model exactly: one kvq_engine per tier, a run realized as
    WRITE_ADDR = run base + back-to-back key tokens + FLUSH on a partial
    tail; a tier switch is only legal at a run boundary (engine idle).

    The map therefore has RUN granularity for keys — key-group scales never
    cross a tier boundary — and per-token granularity for values.
    Returns (K_hat, V_hat, [kb per run], [vb per run]).
    """
    T, D = K_f16.shape
    assert len(tier_map) == T and all(t in TIERS for t in tier_map)
    K_hat = np.empty((T, D), dtype=np.float64)
    V_hat = np.empty((T, D), dtype=np.float64)
    kbs, vbs = [], []
    a = 0
    while a < T:
        e = a
        while e < T and tier_map[e] == tier_map[a]:
            e += 1
        kh, vh, kb, vb = kvq_roundtrip(K_f16[a:e], V_f16[a:e],
                                       tier_map[a], G, outlier_idx)
        K_hat[a:e], V_hat[a:e] = kh, vh
        kbs.append(kb)
        vbs.append(vb)
        a = e
    return K_hat, V_hat, kbs, vbs


# ── results ──────────────────────────────────────────────────────────────────

@dataclass
class AttnCore:
    """All intermediates of the KV-side chain (stages Q5..Q13)."""
    tier: str
    D: int
    T: int
    G: int
    K_f16: np.ndarray = None        # [T,D] fp16 bits (codec input)
    V_f16: np.ndarray = None
    kb: object = None               # KeyBlob/ValueBlob (actual codec scales)
    vb: object = None
    K_hat: np.ndarray = None        # [T,D] f64 of the fp32 read bus
    V_hat: np.ndarray = None
    k8: np.ndarray = None           # feeder INT8 + fp16 scales
    s_k: np.ndarray = None
    v8: np.ndarray = None
    s_v: np.ndarray = None
    q: np.ndarray = None            # raw q row (f64) fed to the Q feeder
    q8: np.ndarray = None
    s_q: np.uint16 = None
    acc_s: np.ndarray = None        # INT32 Q.K_hat^T
    score_fx: np.ndarray = None     # Q.SCORE_FRAC
    p_q115: np.ndarray = None       # ASU output
    sm_m: int = None                # ASU end-of-row mmax (C-CHUNK merge state)
    sm_l: int = None                # ASU end-of-row lsum, Q1.15
    c: np.ndarray = None            # p*s_v (real), pre-quant
    c8: np.ndarray = None
    s_c: np.uint16 = None
    acc_o: np.ndarray = None        # INT32 P.V_hat
    rq: tuple = None                # (scale, shift) of the output epilogue
    o8: np.ndarray = None
    s_out: float = None
    out_hat: np.ndarray = None      # o8 * s_out (the chain's answer)


@dataclass
class AttnFull(AttnCore):
    """Full chain adds the front end (stages Q0..Q4)."""
    x8: np.ndarray = None           # [T+1,D] INT8 token activations
    s_x: np.ndarray = None
    h: np.ndarray = None            # [T+1,D] Q7.8 int
    r_fx: np.ndarray = None         # per-token UQ2.13 rsqrt
    den_fx: np.ndarray = None       # per-token floor-sqrt (norm_int)
    h8: np.ndarray = None
    s_h: np.ndarray = None
    acc_k: np.ndarray = None        # raw projection accumulators (S-2)
    acc_v: np.ndarray = None
    acc_q: np.ndarray = None


def attention_core(q_f64: np.ndarray, K_f16: np.ndarray, V_f16: np.ndarray,
                   tier: str, G: int = 128, outlier_idx=(),
                   score_frac: int = SCORE_FRAC,
                   res: AttnCore | None = None,
                   tier_map: list[str] | None = None,
                   kv_pre: tuple | None = None) -> AttnCore:
    """Stages Q5..Q13: KVQ -> feeders -> scores -> ASU -> P.V_hat -> epilogue.

    tier_map (optional, D-022 actuation): per-row tier list overriding the
    single `tier` for the KVQ roundtrip — see kvq_roundtrip_tiermap. All
    downstream stages (feeders, scores, softmax, P·V̂) are per-row and are
    unchanged; `tier` then only labels the result.

    kv_pre (optional, GQA host-speed reuse): the (K_hat, V_hat, kb, vb, k8,
    s_k, v8, s_v) tuple a PRIOR attention_core call produced for the SAME
    K_f16/V_f16 bits, tier, G and outliers. Q5/Q6 are pure functions of
    those inputs, so reusing a sibling query head's stage outputs is
    bit-identical to recomputing them (decoder_layer_fx shares one KV
    head's chain across its GQA group this way). Default None recomputes —
    today's path, and the RTL-replay arbiter used by --verify-trace."""
    T, D = K_f16.shape
    r = res if res is not None else AttnCore(tier=tier, D=D, T=T, G=G)
    r.tier, r.D, r.T, r.G = tier, D, T, G
    r.K_f16, r.V_f16 = K_f16, V_f16

    if kv_pre is not None:
        (r.K_hat, r.V_hat, r.kb, r.vb, r.k8, r.s_k, r.v8, r.s_v) = kv_pre
    else:
        if tier_map is not None:
            r.K_hat, r.V_hat, r.kb, r.vb = kvq_roundtrip_tiermap(
                K_f16, V_f16, tier_map, G, outlier_idx)
        else:
            r.K_hat, r.V_hat, r.kb, r.vb = kvq_roundtrip(K_f16, V_f16, tier,
                                                         G, outlier_idx)

        r.k8, r.s_k = quant_rows_i8(r.K_hat)                  # Q6 feeder
        r.v8, r.s_v = quant_rows_i8(r.V_hat)
    r.q = np.asarray(q_f64, dtype=np.float64)
    q8m, s_qa = quant_rows_i8(r.q[None, :])                   # Q7
    r.q8, r.s_q = q8m[0], np.uint16(s_qa[0])

    r.acc_s = gemm_i8(r.q8[None, :], r.k8.T)[0].astype(np.int64)     # Q8
    r.score_fx = score_dequant_fx(r.acc_s, r.s_q, r.s_k, D, score_frac)  # Q9
    p, r.sm_m, r.sm_l = online_softmax_fx(r.score_fx, score_frac,
                                          return_state=True)  # Q10
    r.p_q115 = np.asarray(p, dtype=np.int64)

    s_v_val = f16_bits_to_f64(r.s_v)
    r.c = (r.p_q115.astype(np.float64) / (1 << OUT_FRAC)) * s_v_val   # Q11/S-4
    c8m, s_ca = quant_rows_i8(r.c[None, :])
    r.c8, r.s_c = c8m[0], np.uint16(s_ca[0])

    r.acc_o = gemm_i8(r.c8[None, :], r.v8)[0].astype(np.int64)       # Q12
    amax_o = int(np.max(np.abs(r.acc_o), initial=0)) or 1
    scale, shift = calib_requant(amax_o)                      # Q13
    r.rq = (scale, shift)
    r.o8 = requant_i32_to_i8(r.acc_o, scale, shift).astype(np.int64)
    assert np.max(np.abs(r.o8)) < 128 and (
        np.max(np.abs(r.acc_o)) * scale) >> shift <= 127, "epilogue clip"
    s_c_val = float(f16_bits_to_f64(np.array([r.s_c]))[0])
    r.s_out = s_c_val * float(1 << shift) / float(scale)
    r.out_hat = r.o8.astype(np.float64) * r.s_out
    return r


def attention_fx(X: np.ndarray, Wq8, Wk8, Wv8,
                 s_wq: float, s_wk: float, s_wv: float,
                 gamma_q213: np.ndarray, tier: str, G: int = 128,
                 outlier_idx=(), score_frac: int = SCORE_FRAC) -> AttnFull:
    """Full chain. X: [T+1, D] float64 token activations — rows 0..T-1 are the
    cached context tokens, row T is the decode (query) token. Weight scales
    s_w* are exact float64 per-tensor dequant scales."""
    X = np.asarray(X, dtype=np.float64)
    Tp1, D = X.shape
    T = Tp1 - 1
    r = AttnFull(tier=tier, D=D, T=T, G=G)

    r.x8, r.s_x = quant_rows_i8(X)                            # Q0
    g = [int(v) for v in np.asarray(gamma_q213)]
    r.h = np.empty((Tp1, D), dtype=np.int64)                  # Q1
    r.r_fx = np.empty(Tp1, dtype=np.int64)
    r.den_fx = np.empty(Tp1, dtype=np.int64)
    for t in range(Tp1):
        y, rr, nrm = rmsnorm_fx([int(v) for v in r.x8[t]], g)
        r.h[t], r.r_fx[t], r.den_fx[t] = y, rr, nrm

    r.h8, r.s_h = quant_rows_i8(r.h.astype(np.float64) / 256.0)      # Q2/S-1

    r.acc_k = gemm_i8(r.h8[:T], np.asarray(Wk8, dtype=np.int64))     # Q3
    r.acc_v = gemm_i8(r.h8[:T], np.asarray(Wv8, dtype=np.int64))
    r.acc_q = gemm_i8(r.h8[T:], np.asarray(Wq8, dtype=np.int64))[0]

    s_h_val = f16_bits_to_f64(r.s_h)
    K_real = r.acc_k.astype(np.float64) * (s_h_val[:T, None] * s_wk)  # Q4/S-2
    V_real = r.acc_v.astype(np.float64) * (s_h_val[:T, None] * s_wv)
    K_f16 = f64_to_f16_bits(K_real).reshape(T, D)
    V_f16 = f64_to_f16_bits(V_real).reshape(T, D)
    q_real = r.acc_q.astype(np.float64) * (float(s_h_val[T]) * s_wq)

    attention_core(q_real, K_f16, V_f16, tier, G, outlier_idx, score_frac, res=r)
    return r


# ── C-CHUNK (S6): T > T_ROW_MAX host chunk merge (online-softmax algebra) ────

def softmax_merge_weights(states: list[tuple[int, int]],
                          score_frac: int = SCORE_FRAC
                          ) -> tuple[list[float], int, int]:
    """Merge per-chunk online-softmax states [(m_c, l_c)] into chunk weights.

    The exact online-softmax merge identity: with per-chunk running max m_c
    and rescaled sum l_c, the global result is
        m* = max_c m_c;   l̃_c = l_c·e^(m_c − m*);   l* = Σ_c l̃_c
        y  = Σ_c (l̃_c / l*) · y_c
    where y_c is chunk c's own softmax-normalized attention output. Fixed
    point: e^(m_c − m*) is the SAME exp LUT the ASU rescale path uses
    (exp_fx on the Q6.10 clamp of the score-domain delta — deltas below −16
    real units underflow to weight 0, matching the streaming rescale);
    l̃_c = l_c·α_c is kept as the EXACT int64 product (Q.30, no truncation);
    the final weights w_c = l̃_c/l* are an exact-integer ratio evaluated in
    host float64 (the merge runs on the host, which has fp64; an fp32 host
    would perturb w_c by ≤ 2^-24 relative — noted, not modeled).

    Chunk semantics match the cache reality: each ≤T_ROW_MAX job compresses
    its keys from its own base (chunk = G keeps groups aligned with the
    unchunked cache layout; a partial tail is the D-008 flush case).
    RTL delta implied (backlog, not S6): asu_softmax must export mmax/lsum
    (they exist as registers, ports absent).
    """
    assert states and all(l_c > 0 for _, l_c in states)
    m_star = max(m_c for m_c, _ in states)
    lt = []
    for m_c, l_c in states:
        d = _clamp_z((m_c - m_star) << (EXP_FRAC_BITS - score_frac)
                     if score_frac <= EXP_FRAC_BITS else
                     (m_c - m_star) >> (score_frac - EXP_FRAC_BITS))
        lt.append(int(l_c) * int(exp_fx(np.array([d]))[0]))   # exact Q.30
    l_star = sum(lt)
    assert l_star > 0, "all chunks underflowed — degenerate score row"
    return [t / l_star for t in lt], m_star, l_star


def attention_chunked(q_f64: np.ndarray, K_f16: np.ndarray, V_f16: np.ndarray,
                      tier: str, G: int = 128, chunk: int = CHUNK_T_MAX,
                      outlier_idx=(), score_frac: int = SCORE_FRAC
                      ) -> tuple[np.ndarray, list[AttnCore], dict]:
    """T > T_ROW_MAX attention: per-chunk attention_core jobs + C-CHUNK merge.

    Each chunk of ≤`chunk` cache rows runs the FULL verified per-job chain
    (KVQ → feeders → scores → ASU → P·V̂ → epilogue) exactly as a T≤128 job;
    the host merges chunk outputs with softmax_merge_weights. The merged
    output stays real float64 (it feeds the W_O C-1 feeder, same as the
    unchunked out_hat — no extra narrowing is introduced by the merge).
    """
    T, D = K_f16.shape
    assert 1 <= chunk <= CHUNK_T_MAX, "chunk exceeds the T_ROW_MAX envelope"
    cores: list[AttnCore] = []
    for a in range(0, T, chunk):
        e = min(a + chunk, T)
        cores.append(attention_core(q_f64, K_f16[a:e], V_f16[a:e], tier, G,
                                    outlier_idx, score_frac))
    w, m_star, l_star = softmax_merge_weights(
        [(c.sm_m, c.sm_l) for c in cores], score_frac)
    y = np.zeros(D, dtype=np.float64)
    for wc, c in zip(w, cores):
        y += wc * c.out_hat
    return y, cores, {"w": w, "m_star": m_star, "l_star": l_star}


# ── float64 reference chains (the yardstick) ─────────────────────────────────

@dataclass
class AttnRef:
    h_ref: np.ndarray = None        # [T+1,D] (full chain only)
    q_ref: np.ndarray = None        # [D]
    K_ref: np.ndarray = None        # [T,D]
    V_ref: np.ndarray = None
    scores_ref: np.ndarray = None   # [T] (post 1/sqrt(D))
    p_ref: np.ndarray = None
    y_ref: np.ndarray = None        # [D] float64 attention output


def attention_ref_core(q_f64, K_f16, V_f16) -> AttnRef:
    """Yardstick for core cases: exact f64 values of the codec-input tensors."""
    ref = AttnRef()
    ref.q_ref = np.asarray(q_f64, dtype=np.float64)
    ref.K_ref = f16_bits_to_f64(K_f16).reshape(K_f16.shape)
    ref.V_ref = f16_bits_to_f64(V_f16).reshape(V_f16.shape)
    D = ref.K_ref.shape[1]
    ref.scores_ref = ref.K_ref @ ref.q_ref / math.sqrt(float(D))
    ref.p_ref = softmax_ref(ref.scores_ref)
    ref.y_ref = ref.p_ref @ ref.V_ref
    return ref


def attention_ref_full(X, Wq8, Wk8, Wv8, s_wq, s_wk, s_wv,
                       gamma_q213) -> AttnRef:
    """Yardstick for full-chain cases: float64 all the way, same DEQUANTIZED
    weights (w8*s_w exact), rmsnorm_ref (eps = 2^-14), softmax_ref."""
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0] - 1
    D = X.shape[1]
    ref = AttnRef()
    g_real = np.asarray(gamma_q213, dtype=np.float64) / (1 << 13)
    ref.h_ref = rmsnorm_ref(X, g_real)
    Wq = np.asarray(Wq8, dtype=np.float64) * s_wq
    Wk = np.asarray(Wk8, dtype=np.float64) * s_wk
    Wv = np.asarray(Wv8, dtype=np.float64) * s_wv
    ref.q_ref = ref.h_ref[T] @ Wq
    ref.K_ref = ref.h_ref[:T] @ Wk
    ref.V_ref = ref.h_ref[:T] @ Wv
    ref.scores_ref = ref.K_ref @ ref.q_ref / math.sqrt(float(D))
    ref.p_ref = softmax_ref(ref.scores_ref)
    ref.y_ref = ref.p_ref @ ref.V_ref
    return ref
