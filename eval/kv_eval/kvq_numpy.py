# kvq_numpy.py — VECTORIZED twin of the golden KVQ codec, for model-scale evals.
#
# The golden model (golden/apex_golden/cq_codec.py) is the arbiter and is
# deliberately scalar/slow. This twin re-implements ONLY the quantize->
# dequantize round trip with vectorized numpy, using the same float64
# intermediates and IEEE RNE narrowings (fp.py contract C-1). It is certified
# BIT-EXACT against the golden compress->decompress path by
# test_twin_bitexact.py, which MUST pass before any eval result is reported.
# If the golden codec changes, the gate fails and this twin must be re-proven.

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "golden"))
from apex_golden.cq_codec import EPS, qmax_of, qmin_of  # noqa: E402  (the arbiter's own constants)


def _scales_f64(amax_bits: np.ndarray, bits: int) -> np.ndarray:
    """Vector scale_from_amax: s = max(amax/qmax, EPS), RNE-cast fp16, -> f64."""
    amax = amax_bits.astype(np.uint16).view(np.float16).astype(np.float64)
    s = np.maximum(amax / qmax_of(bits), EPS)
    return s.astype(np.float16).astype(np.float64)  # fp16 RNE, exact value back


def _amax_bits(x_bits: np.ndarray, axis: int) -> np.ndarray:
    """Sign-cleared fp16 magnitude winner along axis (== golden amax_bits)."""
    return (x_bits & np.uint16(0x7FFF)).max(axis=axis)


def qdq_values(V_bits: np.ndarray, bits: int) -> np.ndarray:
    """[T, D] fp16 bit patterns -> [T, D] float32 (== golden compress+decompress)."""
    x = V_bits.astype(np.uint16).view(np.float16).astype(np.float64)
    s = _scales_f64(_amax_bits(V_bits, axis=1), bits)          # [T] per-token
    codes = np.clip(np.rint(x / s[:, None]),
                    qmin_of(bits), qmax_of(bits)).astype(np.int64)  # int like rne()
    return (codes * s[:, None]).astype(np.float32)


def qdq_keys(K_bits: np.ndarray, bits: int, G: int, outlier_idx) -> np.ndarray:
    """[T, D] fp16 bits -> [T, D] float32. Per-channel grouped scales (partial
    final group scales over g, not G — §3.1); outlier channels replay fp16."""
    T, D = K_bits.shape
    out = np.empty((T, D), dtype=np.float32)
    outlier = np.sort(np.asarray(outlier_idx, dtype=np.int64))
    keep = np.setdiff1d(np.arange(D), outlier)
    x = K_bits.astype(np.uint16).view(np.float16).astype(np.float64)
    bounds = [(a, min(a + G, T)) for a in range(0, T, G)] if G > 0 else [(0, T)]
    for a, e in bounds:
        xk = x[a:e][:, keep]                                   # [g, nk]
        s = _scales_f64(_amax_bits(K_bits[a:e][:, keep], axis=0), bits)  # [nk]
        codes = np.clip(np.rint(xk / s[None, :]),
                        qmin_of(bits), qmax_of(bits)).astype(np.int64)
        out[a:e, keep] = (codes * s[None, :]).astype(np.float32)
    if len(outlier):
        out[:, outlier] = K_bits[:, outlier].view(np.float16).astype(np.float32)
    return out


def qdq_kv(K_bits, V_bits, tier: str, G: int = 128, outlier_idx=()):
    """tier in {kvq8, kvq4, kvq4p}: returns (K_f32, V_f32) through the codec.
    KVQ8 keys use the per-token value path (TIER==0 in RTL); KVQ4/KVQ4+ keys
    are per-channel grouped INT4 (KVQ4+ adds fp16 outlier lanes)."""
    if tier == "kvq8":
        return qdq_values(K_bits, 8), qdq_values(V_bits, 8)
    if tier == "kvq4":
        return qdq_keys(K_bits, 4, G, ()), qdq_values(V_bits, 4)
    if tier == "kvq4p":
        return qdq_keys(K_bits, 4, G, outlier_idx), qdq_values(V_bits, 4)
    raise ValueError(f"unknown tier {tier}")
