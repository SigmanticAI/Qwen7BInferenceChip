#!/usr/bin/env python
"""gen_fparith_vectors.py — GOLDEN-derived vectors for the exhaustive fp-arith
bit-exactness TB (verif/kvq/fparith/tb_fparith.sv).

The arbiter for every expected value is golden/apex_golden (cq_codec.py /
fp.py) — the same float64->narrow chain that gates every other KVQ suite.
Nothing here uses int_model.py: the TB proves the synthesizable integer SV
against the GOLDEN, not against its own Python mirror.

Emitted into build/vec/ ($readmemh format, one word per line):

  scale_exp_b4.hex / scale_exp_b8.hex   fp16 expected scale for every finite
                                        fp16 amax pattern (FINITE_ALL order,
                                        63488 lines; negative amax included —
                                        golden gives EPS for those)
  widen_exp.hex                         fp32 expected f16->f32 widen for every
                                        finite fp16 pattern (= the CQ-4+
                                        outlier lane, dequant code=+1)
  qscale_b4.hex / qscale_b8.hex         the DENSE quant scale sets: every
                                        distinct scale the golden can produce
                                        for that qmax over all finite
                                        magnitudes, plus boundary/EPS/
                                        subnormal/negative extras
  qsum_b4.hex / qsum_b8.hex             per-scale order-sensitive 64-bit
                                        checksum of the golden quant codes
                                        over ALL finite fp16 x (see CHECKSUM)
  dsum.hex                              per-scale checksum of the golden
                                        dequant fp32 bits over codes
                                        -256..255, scale sweeping ALL finite
                                        fp16 patterns (FINITE_ALL order,
                                        incl. +-0 / subnormal / negative)
  fparith_sizes.svh                     localparam sizes for the TB

CHECKSUM (order-sensitive, wrap mod 2^64; identical in tb_fparith.sv):
    acc = 0; w = 1
    for each value v (uint64) in sweep order:
        acc += (v + K) * w ;  w *= R          (all mod 2^64)
    K = 0x9E3779B97F4A7C15, R = 0xD1B54A32D192ED03 (odd => w never collapses)

quant v = the int64 code as two's-complement uint64; dequant v = the fp32
bits zero-extended. A checksum equality over an exhaustive sweep pins every
word: any single-bit deviation at any position changes the sum (weights are
distinct odd multipliers), which the seeded-error runs demonstrate.

inf/nan (exp==31) inputs are OUT OF CONTRACT (the codec never produces them;
the golden's own int cast on them is undefined) and are excluded, exactly as
in prove.py. +-0 scales are excluded from the QUANT sweep only (golden x/0
is undefined); they ARE in the dequant sweep.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "golden"))
import numpy as np
from apex_golden import cq_codec as C
from apex_golden import fp as FP

OUT = os.path.join(os.path.dirname(__file__), "build", "vec")
os.makedirs(OUT, exist_ok=True)

K = np.uint64(0x9E3779B97F4A7C15)
R = np.uint64(0xD1B54A32D192ED03)

FINITE_ALL = np.array([b for b in range(0x10000)
                       if (b & 0x7C00) != 0x7C00], dtype=np.uint16)
FINITE_MAG = np.arange(0x0000, 0x7C00, dtype=np.uint16)
NF = FINITE_ALL.size                          # 63488

# defensive boundary scales (kept in sync with prove.py BOUNDARY_S)
BOUNDARY_S = [0x0400, 0x0001, 0x03FF, 0x0401, 0x7BFF, 0x3C00, 0x3C01, 0x2400,
              0x8400, 0x8001, 0x83FF, 0xBC00, 0xFBFF]


def wr(name, vals, width):
    with open(os.path.join(OUT, name), "w") as f:
        for v in vals:
            f.write(f"{int(v) & ((1 << (4 * width)) - 1):0{width}x}\n")
    print(f"  wrote {name}: {len(vals)} words")


def weights(n):
    w = np.empty(n, dtype=np.uint64)
    w[0] = np.uint64(1)
    if n > 1:
        w[1:] = np.cumprod(np.full(n - 1, R, dtype=np.uint64))
    return w


def checksum(vals_u64, w):
    return np.uint64(np.sum((vals_u64 + K) * w, dtype=np.uint64))


# ── scale_from_amax: exhaustive explicit expected (both qmax) ───────────────
for bits in (4, 8):
    exp = [int(C.scale_from_amax(np.uint16(a), bits))
           for a in FINITE_ALL.tolist()]
    wr(f"scale_exp_b{bits}.hex", exp, 4)

# ── widen f16->f32 (outlier lane, dequant code=+1): exhaustive explicit ─────
widen = FP.f64_to_f32_bits(FP.f16_bits_to_f64(FINITE_ALL))
wr("widen_exp.hex", widen.tolist(), 8)

# ── quant: dense golden-producible scale sets + per-scale checksums ─────────
X64 = FP.f16_bits_to_f64(FINITE_ALL)          # exact values, computed once
WX = weights(NF)
nqs = {}
for bits in (4, 8):
    sset = set(int(C.scale_from_amax(np.uint16(a), bits))
               for a in FINITE_MAG.tolist())
    sset |= set(BOUNDARY_S)
    sset = sorted(s for s in sset if (s & 0x7FFF) != 0)   # no +-0 (x/0 UB)
    nqs[bits] = len(sset)
    wr(f"qscale_b{bits}.hex", sset, 4)

    lo, hi = C.qmin_of(bits), C.qmax_of(bits)
    sums = np.empty(len(sset), dtype=np.uint64)
    for i, s in enumerate(sset):
        sval = float(FP.f16_bits_to_f64(np.array([s], dtype=np.uint16))[0])
        # identical chain to C.quant_codes (rne + clip on the exact f64 x/s);
        # X64 is hoisted out of the loop for speed only
        codes = np.clip(FP.rne(X64 / sval), lo, hi)
        sums[i] = checksum(codes.view(np.uint64), WX)
    wr(f"qsum_b{bits}.hex", sums.tolist(), 16)
    print(f"  quant bits={bits}: {len(sset)} scales x {NF} x-values")

# ── dequant: codes -256..255 x ALL finite scale patterns, checksums ─────────
CODES = np.arange(-256, 256, dtype=np.int64)
WD = weights(CODES.size)
dsums = np.empty(NF, dtype=np.uint64)
for i, s in enumerate(FINITE_ALL.tolist()):
    g = C.dequant_f32(CODES, np.uint16(s))
    dsums[i] = checksum(g.astype(np.uint64), WD)
wr("dsum.hex", dsums.tolist(), 16)
print(f"  dequant: {NF} scales x {CODES.size} codes")

# ── TB size include ─────────────────────────────────────────────────────────
with open(os.path.join(OUT, "fparith_sizes.svh"), "w") as f:
    f.write("// generated by gen_fparith_vectors.py — do not edit\n")
    f.write(f"localparam int N_QSCALE4 = {nqs[4]};\n")
    f.write(f"localparam int N_QSCALE8 = {nqs[8]};\n")
print("  wrote fparith_sizes.svh")
print("GEN OK")
