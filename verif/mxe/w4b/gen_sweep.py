#!/usr/bin/env python3
"""gen_sweep.py — the REQUIRED exhaustive operand sweep (W4B stage 3,
B3_WEIGHT_PATH.md §8.4, fparith pattern).

Domain: EVERY positive finite fp16 group scale (subnormals included:
0x0001..0x7BFF, 31,743 values) x all 16 codes x a fixed panel of s8
values spanning the contract (EPS, 1.0, max normal, min subnormal, two
mid-grid, one tile-typical, one just-under-EPS). Expected value per
element from the golden per-element rule (float64 exact); emitted as a
BINARY expectation file (one int8 per row, row index = implicit operand
order) so the TB enumerates the identical order and compares 4.06M
results with zero parsing overhead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
import numpy as np
from apex_golden.fp import f16_bits_to_f64, rne

S8_PANEL = [0x0400, 0x3C00, 0x7BFF, 0x0001, 0x1000, 0x63D0, 0x2E66, 0x03FF]

def main():
    out = Path(__file__).parent / "build"
    out.mkdir(exist_ok=True)
    sg_bits = np.arange(0x0001, 0x7C00, dtype=np.uint16)      # 31,743
    sg_val = f16_bits_to_f64(sg_bits)
    total = 0
    exp = np.empty(len(S8_PANEL) * 16 * len(sg_bits), dtype=np.int8)
    i = 0
    for s8b in S8_PANEL:
        s8v = float(f16_bits_to_f64(np.array([np.uint16(s8b)]))[0])
        for code in range(-8, 8):
            q = np.clip(rne(code * sg_val / s8v), -128, 127).astype(np.int8)
            exp[i:i + len(sg_bits)] = q
            i += len(sg_bits)
            total += len(sg_bits)
    (out / "sweep_expect.bin").write_bytes(exp.tobytes())
    (out / "sweep_meta.txt").write_text(
        " ".join(f"{b:04x}" for b in S8_PANEL) + f"\n{len(sg_bits)}\n")
    print(f"sweep: {total} expectations "
          f"({len(S8_PANEL)} s8 x 16 codes x {len(sg_bits)} scales)")

if __name__ == "__main__":
    main()
