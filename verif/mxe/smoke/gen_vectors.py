#!/usr/bin/env python3
"""gen_vectors.py — MXE smoke-TB vector generator (provenance record).

The constants embedded in tb_mxe_smoke.sv were produced by THIS script from
the golden arbiter apex_golden.compute (gemm_i8 + requant_i32_to_i8, C-2/C-3).
Re-run it and diff against the TB if the vectors are ever in doubt:

    python3 \
        gen_vectors.py

Test 1: WS GEMM M=4, K=8, N=8, requant_en=1 (scale/shift chosen so the
        operand set contains at least one exact RNE tie, plus saturation).
Test 3: OS accumulate chain — full K=16 problem split into two K=8 OS
        descriptors (accumulate=0 then accumulate=1), raw INT32 drain.

Beat packing (mirrors mxe_ctrl.sv contract):
  act beat (m,p):  lane k = A[m][8p+k]      -> bits [8k+7:8k]
  wgt beat (p,c):  lane r = B[8p+r][c]      -> bits [8r+7:8r]
  res beat m:      lane c -> bits [32c+31:32c]; requant lanes sign-extended.
"""
import sys

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
from apex_golden.compute import gemm_i8, requant_i32_to_i8  # noqa: E402


def pack8(vals):
    """8 int8 lane values -> 64-bit SV hex literal."""
    w = 0
    for k, v in enumerate(vals):
        w |= (int(v) & 0xFF) << (8 * k)
    return f"64'h{w:016x}"


def pack32(vals):
    """8 int32 lane values -> 256-bit SV hex literal."""
    w = 0
    for c, v in enumerate(vals):
        w |= (int(v) & 0xFFFF_FFFF) << (32 * c)
    return f"256'h{w:064x}"


def sv_array(name, words, width):
    lines = [f"  localparam logic [{width-1}:0] {name} [{len(words)}] = '{{"]
    for i, w in enumerate(words):
        sep = "," if i < len(words) - 1 else ""
        lines.append(f"    {w}{sep}")
    lines.append("  };")
    return "\n".join(lines)


def has_rne_tie(acc, scale, shift):
    p = acc.astype(np.int64) * int(scale)
    if shift == 0:
        return False
    half = np.int64(1) << (shift - 1)
    mask = (np.int64(1) << shift) - 1
    return bool(np.any((p & mask) == half))


rng = np.random.default_rng(20260707)

# ── Test 1: WS + requant ────────────────────────────────────────────────────
M, K, N = 4, 8, 8
A1 = rng.integers(-128, 128, size=(M, K), dtype=np.int64).astype(np.int8)
B1 = rng.integers(-128, 128, size=(K, N), dtype=np.int64).astype(np.int8)
acc1 = gemm_i8(A1, B1)

# pick scale/shift with an exact RNE tie and some saturation
choice = None
for shift in range(6, 15):
    for scale in range(1, 400):
        y = requant_i32_to_i8(acc1, scale, shift)
        if has_rne_tie(acc1, scale, shift) and np.any(np.abs(y.astype(int)) == 127):
            choice = (scale, shift)
            break
    if choice:
        break
assert choice, "no tie-generating (scale, shift) found"
T1_SCALE, T1_SHIFT = choice
y1 = requant_i32_to_i8(acc1, T1_SCALE, T1_SHIFT)

print(f"// Test 1: scale={T1_SCALE} shift={T1_SHIFT}  "
      f"(RNE tie present: {has_rne_tie(acc1, T1_SCALE, T1_SHIFT)}, "
      f"saturated lanes: {int(np.sum(np.abs(y1.astype(int)) == 127))})")
print(sv_array("T1_ACT", [pack8(A1[m]) for m in range(M)], 64))
print(sv_array("T1_WGT", [pack8(B1[:, c]) for c in range(N)], 64))
print(sv_array("T1_EXP", [pack32(y1[m].astype(np.int64)) for m in range(M)], 256))
print(f"  localparam logic [15:0] T1_SCALE = 16'd{T1_SCALE};")
print(f"  localparam logic [4:0]  T1_SHIFT = 5'd{T1_SHIFT};")

# ── Test 3: OS accumulate chain, K=16 as 8+8, raw INT32 drain ───────────────
K3 = 16
A3 = rng.integers(-128, 128, size=(M, K3), dtype=np.int64).astype(np.int8)
B3 = rng.integers(-128, 128, size=(K3, N), dtype=np.int64).astype(np.int8)
acc_c0 = gemm_i8(A3[:, :8], B3[:8, :])          # after descriptor 1
acc_all = gemm_i8(A3, B3)                       # after descriptor 2 (chain)
assert np.array_equal(acc_all, acc_c0 + gemm_i8(A3[:, 8:], B3[8:, :]))

print(sv_array("T3_ACT_C0", [pack8(A3[m, :8]) for m in range(M)], 64))
print(sv_array("T3_ACT_C1", [pack8(A3[m, 8:]) for m in range(M)], 64))
print(sv_array("T3_WGT_C0", [pack8(B3[:8, c]) for c in range(N)], 64))
print(sv_array("T3_WGT_C1", [pack8(B3[8:, c]) for c in range(N)], 64))
print(sv_array("T3_EXP_C0", [pack32(acc_c0[m].astype(np.int64)) for m in range(M)], 256))
print(sv_array("T3_EXP_C1", [pack32(acc_all[m].astype(np.int64)) for m in range(M)], 256))

# ── Test 4: WS multi-pass in ONE descriptor — K=13 (KB=2, klast=5), N=5 ─────
# Exercises: within-descriptor multi-pass accumulate, K-tail weight-lane
# masking (rows 5..7 of chunk 1), N<8 (zero shift-chain columns 5..7, zero
# drain lanes 5..7). Padding lanes carry GARBAGE (0xAA acts / 0x55 weights)
# to prove the DUT's masking, not the vectors', makes the result correct.
M4, K4, N4 = 4, 13, 5
A4 = rng.integers(-128, 128, size=(M4, K4), dtype=np.int64).astype(np.int8)
B4 = rng.integers(-128, 128, size=(K4, N4), dtype=np.int64).astype(np.int8)
acc4 = gemm_i8(A4, B4)

def pack8_pad(vals, pad):
    w = 0
    for k in range(8):
        v = int(vals[k]) if k < len(vals) else pad
        w |= (v & 0xFF) << (8 * k)
    return f"64'h{w:016x}"

# act beats: row-major, chunk-minor: (m,p0),(m,p1)
t4_act = []
for m in range(M4):
    t4_act.append(pack8_pad(A4[m, :8], 0xAA))
    t4_act.append(pack8_pad(A4[m, 8:], 0xAA))
# wgt beats: chunk-major, ascending column, N4 beats per chunk
t4_wgt = []
for p in range(2):
    rows = B4[8*p:8*p+8, :]
    for c in range(N4):
        t4_wgt.append(pack8_pad(rows[:, c], 0x55))
# expected: lanes >= N4 are zero
t4_exp = []
for m in range(M4):
    lanes = [int(acc4[m, c]) if c < N4 else 0 for c in range(8)]
    t4_exp.append(pack32(lanes))

print(sv_array("T4_ACT", t4_act, 64))
print(sv_array("T4_WGT", t4_wgt, 64))
print(sv_array("T4_EXP", t4_exp, 256))
