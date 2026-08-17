"""compute.py — golden models for APEX's own compute contract.

Implements ARCHITECTURE.md C-2 (RNE requant epilogue), C-3 (INT32
accumulators, K ≤ 2048), §6 (fixed-point online softmax + RMSNorm).
These define the contract for new RTL — validated by property tests
(tests/test_compute.py), not frozen vectors.
"""
from __future__ import annotations

import os

import numpy as np

from .fp import rne

K_MAX = 2048  # C-3: 21-bit worst-case product sum + 11 bits = exactly INT32
DIM_MAX = 4095  # apex_pkg DIM_W=12: M/K/N descriptor fields are 12-bit
K_TOTAL_MAX = (1 << 17) - 1  # C-KSPLIT: K·2^14 ≤ 2^31 − 2^14 — the ±128·±128
#   all-K worst case provably fits signed INT32 (2^17 exactly would hit +2^31)


# ── C-3: INT8×INT8 → INT32 GEMM ──────────────────────────────────────────────

def gemm_i8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A[M,K] int8 × B[K,N] int8 → int32 accumulators. Rejects K > K_MAX.

    Overflow-impossible by construction: |acc| ≤ K·128·128 = 2048·2¹⁴ = 2²⁵
    — comfortably inside INT32; the analytic worst case (±128×±128 all-K)
    is asserted, mirroring the RTL legality checker + SVA.
    """
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    K = A.shape[1]
    if K > K_MAX or K == 0:
        raise ValueError(f"illegal K={K} (legal: 1..{K_MAX})")  # desc reject
    acc = A @ B
    assert np.all(np.abs(acc) < 2**31), "C-3 violated — impossible by design"
    return acc.astype(np.int32)


# ── host-speed EXACT evaluation of the int8 operand class (bit-identical) ────
#
# gemm_i8_ksplit below carries a float64-BLAS fast path for operands PROVEN
# to be int8-range integers. It changes NO value, ever:
#   * both operands hold integers with |v| <= 128 (int8 dtype guarantees it;
#     other integer dtypes are value-checked), so every product is an integer
#     with |p| <= 2^14;
#   * every PARTIAL sum over K <= K_TOTAL_MAX = 2^17 - 1 terms is an integer
#     with |s| <= K·2^14 <= 2^31 - 2^14 << 2^53 — exactly representable in
#     float64. IEEE-754 mul/add/FMA on exactly-representable integer operands
#     whose true results are exactly representable is EXACT (nothing to
#     round), for ANY summation order or blocking a BLAS kernel chooses;
#   * float64 matmul therefore equals the k-chunked int64 accumulation
#     (integer adds are associative) bit for bit, and the per-prefix INT32
#     assertions cannot fire on this operand class — |prefix| <= K·2^14 is
#     the SAME analytic C-KSPLIT ceiling the docstring below cites. The
#     final INT32 assert is kept verbatim;
#   * belt-and-braces: the integer-valuedness of the BLAS result is asserted
#     elementwise before the cast — a precondition violation is a loud
#     refusal, never a silent value change.
# Float/other non-integer dtypes are NEVER certified (the int64 path's
# truncating asarray is their committed behaviour, unchanged).
#
# _f64_of_i8 can additionally cache the float64 mirror of IMMUTABLE
# int8-dtyped tensors across steps, capped by APEX_GOLD_WCACHE_GB. The
# DEFAULT is 0 == OFF: the full 0.5B token-loop mirror set is ~4 GiB
# resident, and measured on the 18 GB dev host that residency loses more to
# memory pressure than the cached conversion saves (the per-call int8->f64
# widening is chunk-sized and allocator-recycled). Set the env knob on a
# quiet, big-memory host if the conversion shows up in a profile. When
# caching: only non-writeable arrays are cached (golden weights are
# np.load(mmap_mode="r") maps — numpy refuses writes, so a cached mirror can
# never go stale); the key is the view's (data pointer, shape, strides,
# dtype) — the exact bytes read — because np.asarray re-views ndarray
# subclasses (memmaps) into fresh objects per call; the cached strong
# reference pins the buffer so the address can never be recycled while the
# entry lives.

_WCACHE: dict[tuple, tuple] = {}
_WCACHE_BYTES = 0
_WCACHE_LIMIT = int(float(os.environ.get("APEX_GOLD_WCACHE_GB", "0"))
                    * (1 << 30))
_CERT_MAX_ELEMS = 1 << 22        # value-check cap for non-int8 integer dtypes


def _f64_of_i8(W: np.ndarray) -> np.ndarray:
    """Cached float64 mirror of an immutable int8 tensor (see block above)."""
    global _WCACHE_BYTES
    if W.flags.writeable:
        return np.asarray(W, dtype=np.float64)   # mutable: never cached
    key = (W.__array_interface__["data"][0], W.shape, W.strides, W.dtype.str)
    hit = _WCACHE.get(key)
    if hit is not None:
        return hit[1]
    Wf = np.asarray(W, dtype=np.float64)
    if _WCACHE_BYTES + Wf.nbytes <= _WCACHE_LIMIT:
        _WCACHE[key] = (W, Wf)                   # pins the buffer/address
        _WCACHE_BYTES += Wf.nbytes
    return Wf


def _certified_f64(M: np.ndarray):
    """float64 image of an operand PROVEN int8-range integer, else None.

    int8 dtype certifies by type (mirror cached). Other signed-integer
    dtypes certify by a value check, capped at _CERT_MAX_ELEMS so the check
    stays negligible next to the matmul (bigger arrays: int64 path)."""
    if M.dtype == np.int8:
        return _f64_of_i8(M)
    if M.dtype in (np.dtype(np.int16), np.dtype(np.int32),
                   np.dtype(np.int64)) and M.size <= _CERT_MAX_ELEMS \
            and bool(np.all(np.abs(M) <= 128)):
        return np.asarray(M, dtype=np.float64)
    return None


# ── C-KSPLIT (S6): host job-splitting GEMM for K > K_MAX / N > DIM_MAX ───────

def gemm_i8_ksplit(A: np.ndarray, B: np.ndarray, k_chunk: int = K_MAX,
                   n_chunk: int = DIM_MAX) -> np.ndarray:
    """Host-split GEMM: reduction K > K_MAX runs as ceil(K/k_chunk) sub-jobs
    whose INT32 partial accumulators are summed — host-side, or on-tile via
    the frozen descriptor's `accumulate` flag (apex_pkg bit [66]); the integer
    math is identical either way. EVERY prefix sum is asserted to sit in
    INT32, because retained accumulators live in the INT32 array registers.
    Analytic ceiling: K_total ≤ K_TOTAL_MAX = 2^17 − 1 keeps the ±128·±128
    all-K worst case at ≤ 2^31 − 2^14 — provably inside signed INT32 (the
    C-3 argument, extended; 2^17 exactly would hit +2^31). N > DIM_MAX
    (12-bit descriptor field) splits by concatenation — exact, no numerics.
    Bit-identical to a single wide integer matmul by associativity of the
    integer adds; 7B shapes: K = 3584 (hidden) and 18944 (FFN down-proj).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    K = A.shape[1]
    assert B.shape[0] == K, "shape mismatch"
    assert 1 <= k_chunk <= K_MAX and 1 <= n_chunk <= DIM_MAX
    if K > K_TOTAL_MAX or K == 0:
        raise ValueError(f"illegal total K={K} (legal: 1..{K_TOTAL_MAX})")
    Af = _certified_f64(A)
    Bf = _certified_f64(B) if Af is not None else None
    if Bf is not None:
        # exact host-speed path (proof block above) — bit-identical to the
        # chunked int64 accumulation below for this operand class.
        acc = np.matmul(Af, Bf)
        acc64 = acc.astype(np.int64)
        assert np.array_equal(acc, acc64), \
            "f64 GEMM left the exact-integer domain"
        assert np.all(np.abs(acc64) < 2**31), \
            "C-KSPLIT violated: INT32 prefix-sum overflow"
        return acc64.astype(np.int32)
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    if B.shape[1] > n_chunk:
        return np.concatenate(
            [gemm_i8_ksplit(A, B[:, a:a + n_chunk], k_chunk, n_chunk)
             for a in range(0, B.shape[1], n_chunk)], axis=1)
    acc = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for a in range(0, K, k_chunk):
        acc += gemm_i8(A[:, a:a + k_chunk], B[a:a + k_chunk, :]).astype(np.int64)
        assert np.all(np.abs(acc) < 2**31), \
            "C-KSPLIT violated: INT32 prefix-sum overflow"
    return acc.astype(np.int32)


# ── C-2: requant epilogue INT32 → INT8, round-half-to-even ──────────────────

def requant_i32_to_i8(acc: np.ndarray, scale: int, shift: int) -> np.ndarray:
    """y = sat8(rne((acc * scale) / 2^shift)).

    scale: int16 unsigned magnitude (0..65535), shift: 0..31. The product
    acc×scale is exact in int64; the RNE division-by-power-of-two is done
    in exact integer arithmetic (no float) so RTL equivalence is testable
    exhaustively on the operand domain.
    """
    if not (0 <= scale < 2**16 and 0 <= shift < 32):
        raise ValueError("illegal requant params")
    p = np.asarray(acc, dtype=np.int64) * int(scale)
    if shift == 0:
        q = p
    else:
        half = np.int64(1) << (shift - 1)
        mask = (np.int64(1) << shift) - 1
        q = p >> shift                     # floor division (arithmetic shift)
        rem = p & mask                     # non-negative remainder
        round_up = (rem > half) | ((rem == half) & ((q & 1) == 1))
        q = q + round_up.astype(np.int64)
    return np.clip(q, -128, 127).astype(np.int8)


# ── §6: RMSNorm (float64 reference; fixed-point twin lands with the RTL) ─────

def rmsnorm_ref(x: np.ndarray, gamma: np.ndarray, eps: float = 2.0**-14) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5 * gamma


# ── §6: online softmax — float64 reference + fixed-point LUT model ──────────

def softmax_ref(scores: np.ndarray) -> np.ndarray:
    """Numerically-stable float64 softmax (the quality yardstick)."""
    s = np.asarray(scores, dtype=np.float64)
    m = s.max(axis=-1, keepdims=True)
    e = np.exp(s - m)
    return e / e.sum(axis=-1, keepdims=True)


# Fixed-point exp LUT (the reference design vector unit concept, APEX §6, amended by D-014):
#   input:  z = x_scaled in [-16, 0) as Q6.10 (16-bit signed, 10 frac bits)
#   exp(z) via 256-entry LUT on top-8 magnitude bits + linear interpolation,
#   entries Q1.15. Max abs error budget ≤ 2^-10 (verified exhaustively).
#   D-014: a 64-entry LUT is calibrated for fp16 SIMD lanes; a fixed-point
#   chord interpolation at h=16/64 has inherent error h²/8 ≈ 7.8e-3 — measured
#   6.9e-3, over budget. 256 entries (h=0.0625) → ~4.9e-4. ROM cost 1 KB.
LUT_BITS = 8
FRAC_BITS = 10          # Q6.10 input
OUT_FRAC = 15           # Q1.15 output
_SEG = 16.0 / (1 << LUT_BITS)   # segment width in real units = 0.25


def _lut_tables() -> tuple[np.ndarray, np.ndarray]:
    """256 (base, slope) pairs; base Q1.15 at segment start, slope Q1.15/seg."""
    edges = -16.0 + _SEG * np.arange((1 << LUT_BITS) + 1)
    vals = np.exp(edges)
    base = rne(vals[:-1] * (1 << OUT_FRAC))
    slope = rne((vals[1:] - vals[:-1]) * (1 << OUT_FRAC))
    return base.astype(np.int64), slope.astype(np.int64)


_BASE, _SLOPE = _lut_tables()


def exp_fx(z_q: np.ndarray) -> np.ndarray:
    """exp() of Q6.10 inputs in [-16·2^10, 0] → Q1.15. Bit-exact LUT model.

    index = top LUT_BITS of the offset magnitude; frac = remaining bits;
    y = base[i] + (slope[i] * frac) >> frac_rem  (truncating, per C-2 note).
    """
    z = np.asarray(z_q, dtype=np.int64)
    assert np.all((z <= 0) & (z >= -(16 << FRAC_BITS))), "exp_fx domain"
    off = z + (16 << FRAC_BITS)                    # 0 .. 16·2^10
    frac_rem = FRAC_BITS + 4 - LUT_BITS            # bits below the index = 6
    idx = np.minimum(off >> frac_rem, (1 << LUT_BITS) - 1)
    frac = off - (idx << frac_rem)
    return _BASE[idx] + ((_SLOPE[idx] * frac) >> frac_rem)


def online_softmax_fx(scores_q: np.ndarray, score_frac: int,
                      return_state: bool = False):
    """Fixed-point online (streaming) softmax over one row of integer scores.

    Mirrors the ASU dataflow: single pass maintaining running max m and
    rescaled running sum l; second pass (in RTL: the P·V̂ consumer applies
    the same rescale) emits p_i = exp(x_i − m) / l as Q1.15.
    Scores: int64 with `score_frac` fractional bits (INT32 scores → frac 0).

    return_state=True additionally returns (m, l) — the ASU's mmax/lsum
    registers at end-of-row. These are the C-CHUNK merge state (S6): the RTL
    holds them (asu_softmax.sv mmax/lsum) but does not yet export them; the
    golden merge contract (attention.softmax_merge_weights) defines what the
    export ports must carry.
    """
    x = np.asarray(scores_q, dtype=np.int64)
    m = np.int64(-2**62)
    l = np.int64(0)                                # Q1.15 accumulated sum
    for xi in x:                                   # streaming pass
        if xi > m:
            if m != -2**62:
                d = _clamp_z((m - xi) << (FRAC_BITS - score_frac)
                             if score_frac <= FRAC_BITS else
                             (m - xi) >> (score_frac - FRAC_BITS))
                l = (l * exp_fx(np.array([d]))[0]) >> OUT_FRAC   # rescale old sum
            m = xi
        d = _clamp_z((xi - m) << (FRAC_BITS - score_frac)
                     if score_frac <= FRAC_BITS else
                     (xi - m) >> (score_frac - FRAC_BITS))
        l += exp_fx(np.array([d]))[0]
    # emission pass: p_i = exp(x_i − m) · (2^15 / l), computed per element
    out = np.empty(len(x), dtype=np.int64)
    for i, xi in enumerate(x):
        d = _clamp_z((xi - m) << (FRAC_BITS - score_frac)
                     if score_frac <= FRAC_BITS else
                     (xi - m) >> (score_frac - FRAC_BITS))
        e = exp_fx(np.array([d]))[0]
        out[i] = (e << OUT_FRAC) // l if l > 0 else 0
    if return_state:
        return out, int(m), int(l)                  # Q1.15, mmax, lsum
    return out                                      # Q1.15 probabilities


def _clamp_z(d: np.int64) -> np.int64:
    lo = np.int64(-(16 << FRAC_BITS))
    return np.int64(max(lo, min(np.int64(0), np.int64(d))))
