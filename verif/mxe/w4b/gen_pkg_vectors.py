#!/usr/bin/env python3
"""gen_pkg_vectors.py — W4B package-gate vectors (D-031).

Expected values from the golden semantics per element:
q = clip(rne(code * f16(sg) / f16(s8)), -128, 127) — the exact per-element
twin of weight_codec.wfeed_w4b_to_i8 (float64 arithmetic of exact values).
Corners: every code x {min-subnormal, max-subnormal, EPS, 1.0, max-normal,
just-above-EPS, mid, large} scale classes both sides, plus 30k random
positive-finite fp16 pairs (covers the sh>=23 saturation shortcut)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
import numpy as np
from apex_golden.fp import f16_bits_to_f64, rne

rng = np.random.default_rng(0xD030B)
rows = []


def add(code, sg, s8):
    sgv = float(f16_bits_to_f64(np.array([np.uint16(sg)]))[0])
    s8v = float(f16_bits_to_f64(np.array([np.uint16(s8)]))[0])
    if s8v == 0:
        return
    exp = int(np.clip(rne(np.array([code * sgv / s8v]))[0], -128, 127))
    rows.append(f"{code & 0xF:x} {sg:04x} {s8:04x} {exp & 0x1FF:03x}")


classes = [0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x0401, 0x1000, 0x63D0]
for c in range(-8, 8):
    for sg in classes:
        for s8 in classes:
            add(c, sg, s8)
for _ in range(30000):
    add(int(rng.integers(-8, 8)),
        int(rng.integers(0x0001, 0x7C00)), int(rng.integers(0x0001, 0x7C00)))
out = Path(__file__).parent / "build"
out.mkdir(exist_ok=True)
(out / "pkg_vectors.txt").write_text("\n".join(rows) + "\n")
print(f"{len(rows)} vectors")
