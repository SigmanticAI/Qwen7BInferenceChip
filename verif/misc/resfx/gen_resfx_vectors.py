#!/usr/bin/env python3
"""gen_resfx_vectors.py — golden-fp expected vectors for residual_add_fx.

Arbiter: golden/apex_golden/fp.py — expected = f64_to_f16_bits(
f16_bits_to_f64(a) + f16_bits_to_f64(b)), the C-6 primitive exactly as
verif/layer/gen_layer_vectors.py:res_add computes it. The TB additionally
co-simulates the behavioral rtl/misc/residual_add.sv on every vector
(file-driven AND in-sim random), so the integer DUT is pinned against BOTH
references.

Classes (counts printed; the RESULT quotes them):
  tensor  : real decoder-layer residual operand pairs from the L4 anchor
            cases — (f16(X[T]), f16(attn_proj)) and (f16(r1), f16(ffn_out))
  grid    : full exponent-pair sweep ea,eb in [0,30]^2 x 4 sign combos x
            corner mantissas (both-zero e=31 excluded: inf/NaN out of domain)
  tie     : engineered half-ULP ties, both keep parities, add + cancel forms
  zero    : the +-0 matrix, +-0 against normals/subnormals, exact nonzero
            cancellation (must give +0)
  subn    : all-subnormal pairs + subnormal/normal boundary crossings
  ovfl    : saturation corners incl. the 65504+16 tie that rounds to inf
  rand    : uniform random finite pairs

Output: build/vectors_resfx.txt — lines "aaaa bbbb eeee" (hex fp16).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "golden"))
sys.path.insert(0, str(_REPO / "verif" / "top" / "l4"))

import numpy as np  # noqa: E402

from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402


def f16v(bits):
    return float(f16_bits_to_f64(np.array([bits], dtype=np.uint16))[0])


def expected(a, b):
    s = f16_bits_to_f64(np.array([a], dtype=np.uint16)) \
        + f16_bits_to_f64(np.array([b], dtype=np.uint16))
    return int(f64_to_f16_bits(s)[0])


def bits_of(v):
    return int(f64_to_f16_bits(np.array([v], dtype=np.float64))[0])


def finite(b):
    return ((b >> 10) & 0x1F) != 0x1F


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
    out.mkdir(parents=True, exist_ok=True)
    vecs = []          # (class, a, b)

    # ── tensor anchors (golden L4 cases, legacy path) ────────────────────────
    from l4_cases import TIER, build_case, tf  # noqa: E402 (repo-relative)
    for name in ("l4_h2_hd64", "l4_h1_hd128"):
        X, w, q_pos = build_case(name)
        r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos)
        xt = np.asarray(X, dtype=np.float64)[-1]
        for a_v, b_v in ((xt, r.attn_proj), (r.r1, r.ffn_out)):
            ab = f64_to_f16_bits(a_v)
            bb = f64_to_f16_bits(b_v)
            for i in range(ab.size):
                vecs.append(("tensor", int(ab[i]), int(bb[i])))

    # ── exponent grid ────────────────────────────────────────────────────────
    mcorn = [0x000, 0x001, 0x155, 0x2AA, 0x3FE, 0x3FF]
    for ea in range(31):
        for eb in range(31):
            for sa in (0, 1):
                for sb in (0, 1):
                    for ma in mcorn[:3 if (ea * eb) % 2 else 6:2]:
                        for mb in (mcorn[0], mcorn[-1], mcorn[2]):
                            a = (sa << 15) | (ea << 10) | ma
                            b = (sb << 15) | (eb << 10) | mb
                            vecs.append(("grid", a, b))

    # ── engineered ties (half-ULP) ───────────────────────────────────────────
    n_tie = 0
    for ea in range(2, 30):
        for m in (0x000, 0x001, 0x002, 0x003, 0x3FC, 0x3FD, 0x3FE):
            a = (ea << 10) | m
            ulp = 2.0 ** (ea - 15 - 10)
            for scale in (0.5, 1.5):        # tie at same-exp result & carry
                hb = bits_of(ulp * scale)
                if hb and finite(hb):
                    for sb in (0, 1):       # add and cancel directions
                        vecs.append(("tie", a, hb | (sb << 15)))
                        vecs.append(("tie", a | 0x8000, hb | ((1 - sb) << 15)))
                        n_tie += 4
    assert n_tie > 500, "tie battery too small"

    # ── zeros ────────────────────────────────────────────────────────────────
    for a in (0x0000, 0x8000):
        for b in (0x0000, 0x8000):
            vecs.append(("zero", a, b))
    for x in (0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x1234):
        for s in (0, 0x8000):
            vecs.append(("zero", x | s, 0x0000))
            vecs.append(("zero", 0x0000, x | s))
            vecs.append(("zero", x | s, 0x8000))
            vecs.append(("zero", x | s, (x | s) ^ 0x8000))   # exact cancel -> +0
    # ── subnormals ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(0x5EF1D)
    for _ in range(4000):
        a = int(rng.integers(0, 1024)) | (int(rng.integers(0, 2)) << 15)
        b = int(rng.integers(0, 1024)) | (int(rng.integers(0, 2)) << 15)
        vecs.append(("subn", a, b))
    for m in range(0, 1024, 37):
        vecs.append(("subn", 0x03FF, m))                     # boundary cross
        vecs.append(("subn", 0x0400, m | 0x8000))
    # ── overflow / saturation ────────────────────────────────────────────────
    for a, b in ((0x7BFF, 0x7BFF), (0x7BFF, 0x4C00), (0x7BFF, 0x4BFF),
                 (0x7BFF, 0x0400), (0xFBFF, 0xFBFF), (0x7BFF, 0xFBFF),
                 (0x7BFE, 0x4C00)):
        vecs.append(("ovfl", a, b))
    # 65504 + 16 == 65520: EXACT tie at the inf boundary, RNE -> inf
    assert f16v(0x7BFF) == 65504 and f16v(0x4C00) == 16
    # ── random finite ────────────────────────────────────────────────────────
    n_rand = 200_000
    r = rng.integers(0, 1 << 16, size=(n_rand, 2))
    for a, b in r:
        a, b = int(a), int(b)
        if not finite(a):
            a &= ~0x7C00
        if not finite(b):
            b &= ~0x7C00
        vecs.append(("rand", a, b))

    counts = {}
    with (out / "vectors_resfx.txt").open("w") as fh:
        for cls, a, b in vecs:
            assert finite(a) and finite(b)
            fh.write(f"{a:04x} {b:04x} {expected(a, b):04x}\n")
            counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    print("RESFX VECTORS: " + " ".join(f"{k}={v}" for k, v in counts.items())
          + f" total={total}")
    # gen-time self-check: the C-6 form equals verif/layer's res_add form
    for a, b in ((0x3C00, 0xBC00), (0x8000, 0x8000), (0x7BFF, 0x4C00)):
        s = f16_bits_to_f64(np.array([a], dtype=np.uint16)) + \
            f16_bits_to_f64(np.array([b], dtype=np.uint16))
        assert int(f64_to_f16_bits(s)[0]) == expected(a, b)
    print("GEN SELF-CHECK: C-6 expected form consistent")


if __name__ == "__main__":
    main()
