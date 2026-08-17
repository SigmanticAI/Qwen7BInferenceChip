"""test_compute.py — property tests for APEX's own compute contract
(C-2 requant, C-3 GEMM legality, §6 exp LUT + online softmax + RMSNorm).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex_golden import compute as cp                            # noqa: E402

RNG = np.random.default_rng(0xA9E1)
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("[C-3 gemm_i8]")
A = RNG.integers(-128, 128, size=(16, 2048), dtype=np.int64)
B = RNG.integers(-128, 128, size=(2048, 16), dtype=np.int64)
acc = cp.gemm_i8(A, B)
check("matches int64 matmul at K=K_MAX", np.array_equal(acc, (A @ B).astype(np.int32)))
worst = cp.gemm_i8(np.full((1, 2048), -128, np.int64), np.full((2048, 1), -128, np.int64))
check("worst-case |acc| fits INT32", worst[0, 0] == 2048 * 128 * 128)
try:
    cp.gemm_i8(np.zeros((1, 2049), np.int64), np.zeros((2049, 1), np.int64))
    check("K>2048 rejected", False)
except ValueError:
    check("K>2048 rejected", True)

print("[C-2 requant_i32_to_i8]")
# exact RNE reference computed in float via Fraction-free integer math cross-check
acc_s = RNG.integers(-2**31, 2**31, size=20000, dtype=np.int64)
for scale, shift in [(1, 0), (3, 1), (77, 7), (65535, 31), (256, 8), (1, 31)]:
    got = cp.requant_i32_to_i8(acc_s, scale, shift)
    # reference: exact rational rounding via numpy float128-free integer path
    p = acc_s * scale
    q, r = np.divmod(p, np.int64(1) << shift)
    half2 = 2 * r                     # compare 2r vs 2^shift avoids fractions
    up = (half2 > (np.int64(1) << shift)) | ((half2 == (np.int64(1) << shift)) & (q % 2 == 1))
    ref = np.clip(q + up, -128, 127).astype(np.int8)
    check(f"RNE sat scale={scale} shift={shift}", np.array_equal(got, ref))
# tie cases explicitly (the RNE-vs-half-up divergence points)
ties = np.array([2, 6, -2, -6, 10, -10], dtype=np.int64)  # (x*1)/4 → .5 ties
got = cp.requant_i32_to_i8(ties, 1, 2)
check("half-to-even on exact ties", np.array_equal(got, np.array([0, 2, 0, -2, 2, -2], np.int8)),
      f"got {got.tolist()}")

print("[§6 exp LUT]")
z = np.arange(-(16 << cp.FRAC_BITS), 1, dtype=np.int64)   # exhaustive Q6.10 domain
y = cp.exp_fx(z) / (1 << cp.OUT_FRAC)
ref = np.exp(z / (1 << cp.FRAC_BITS))
maxerr = np.abs(y - ref).max()
check(f"exhaustive domain max|err|={maxerr:.2e} ≤ 2^-10", maxerr <= 2**-10)

print("[§6 online softmax fx]")
worst_l1 = 0.0
for trial in range(200):
    n = int(RNG.integers(2, 64))
    scores = RNG.integers(-2**20, 2**20, size=n)
    p_fx = cp.online_softmax_fx(scores, score_frac=10) / (1 << cp.OUT_FRAC)
    p_ref = cp.softmax_ref(scores / (1 << 10))
    worst_l1 = max(worst_l1, np.abs(p_fx - p_ref).sum())
check(f"200 random rows, worst L1 err={worst_l1:.4f} ≤ 0.02", worst_l1 <= 0.02)
mono = cp.online_softmax_fx(np.array([0, 0, 0, 0]), 0)
check("uniform scores → equal probs", len(set(mono.tolist())) == 1)

print("[§6 rmsnorm ref sanity]")
x = RNG.normal(size=(4, 64))
g = np.ones(64)
y = cp.rmsnorm_ref(x, g)
rms = np.sqrt(np.mean(y * y, axis=-1))
check("output RMS ≈ 1", np.allclose(rms, 1.0, atol=1e-3))

print("=" * 64)
if FAILS:
    print(f"FAIL: {len(FAILS)} properties: {FAILS}")
    raise SystemExit(1)
print("APEX COMPUTE GOLDEN: ALL PROPERTIES PASS")
