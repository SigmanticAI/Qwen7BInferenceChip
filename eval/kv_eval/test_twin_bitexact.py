#!/usr/bin/env python3
# test_twin_bitexact.py — GATE: the vectorized eval twin must be bit-exact
# against the golden arbiter's full compress->decompress round trip (packing
# included) before any model eval result may be reported. Anti-fabrication
# rule applies: run this, paste the output.

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apex_golden import cq_codec as cc  # noqa: E402
import kvq_numpy as tw  # noqa: E402

rng = np.random.default_rng(20260713)
checks = fails = 0


def data(T, D, kind):
    if kind == "normal":
        x = rng.standard_normal((T, D))
    elif kind == "heavy":
        x = rng.standard_t(2, (T, D)) * 10          # outlier-bearing
    elif kind == "tiny":
        x = rng.standard_normal((T, D)) * 2.0**-13  # subnormal-adjacent
    else:  # mixed zeros
        x = rng.standard_normal((T, D))
        x[rng.random((T, D)) < 0.3] = 0.0
    return x.astype(np.float16).view(np.uint16)


def eq(name, got_f32, want_f32bits):
    global checks, fails
    n = got_f32.size
    bad = int((got_f32.view(np.uint32) != want_f32bits.astype(np.uint32)).sum())
    checks += n
    if bad:
        fails += bad
        print(f"  FAIL {name}: {bad}/{n} fp32 bit mismatches")


for kind in ["normal", "heavy", "tiny", "zeros"]:
    for T in [1, 7, 64, 128, 130, 257]:
        for D in [64, 128]:
            V = data(T, D, kind)
            K = data(T, D, kind)
            for bits in [4, 8]:
                want = cc.decompress_values(cc.compress_values(V, bits))
                eq(f"val b{bits} {kind} T{T} D{D}", tw.qdq_values(V, bits), want)
            for k_out in [0, 2, 5]:
                idx = np.arange(k_out)
                want = cc.decompress_keys(cc.compress_keys(K, 4, 128, idx))
                eq(f"key k{k_out} {kind} T{T} D{D}", tw.qdq_keys(K, 4, 128, idx), want)

print(f"TWIN BIT-EXACT GATE: checks={checks} fails={fails}")
if fails:
    sys.exit(1)
print("TWIN BIT-EXACT GATE: PASS — eval twin == golden compress->decompress")
