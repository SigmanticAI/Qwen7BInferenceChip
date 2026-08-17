"""fp.py — bit-exact float16/float32 helpers + golden-vector hex I/O.

Contract basis (ARCHITECTURE.md C-1): all intermediate arithmetic in float64
(exact superset of fp16/fp32), round-half-to-even at every narrowing. NumPy's float64 is an exact superset
of fp16/fp32, so the model is bit-exact against the SV RTL and golden vectors.
NumPy's float casts use IEEE round-to-nearest-even, matching real_to_f16/f32.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ── bits ↔ value ────────────────────────────────────────────────────────────

def f16_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    """uint16 IEEE-754 half bit patterns → exact float64 values."""
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint16))
    return b.view(np.float16).astype(np.float64)


def f64_to_f16_bits(x: np.ndarray) -> np.ndarray:
    """float64 → fp16 (RNE) → uint16 bit patterns."""
    return np.asarray(x, dtype=np.float64).astype(np.float16).view(np.uint16)


def f32_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint32))
    return b.view(np.float32).astype(np.float64)


def f64_to_f32_bits(x: np.ndarray) -> np.ndarray:
    """float64 → fp32 (RNE) → uint32 bit patterns."""
    return np.asarray(x, dtype=np.float64).astype(np.float32).view(np.uint32)


# ── round-half-to-even on float64 (numpy.rint == banker's rounding) ─────────

def rne(x: np.ndarray) -> np.ndarray:
    """Round half to even, elementwise, → int64. Mirrors srint_ll."""
    return np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)


# ── frozen golden-vector hex I/O (INDEX.md: one word/line, MSB-first) ────────

_WIDTH = {"u8": 2, "f16": 4, "f32": 8}
_DTYPE = {"u8": np.uint8, "f16": np.uint16, "f32": np.uint32}


def load_hex(path: str | Path, kind: str) -> np.ndarray:
    """Load a `$readmemh`-style hex image → raw-bits ndarray (u8/f16/f32)."""
    w, dt = _WIDTH[kind], _DTYPE[kind]
    vals = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        assert len(line) == w, f"{path}: bad word width {len(line)} != {w}"
        vals.append(int(line, 16))
    return np.asarray(vals, dtype=dt)
