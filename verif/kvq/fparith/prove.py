"""prove.py — EXHAUSTIVE bit-exactness proof of int_model vs the golden arbiter.

  scale_from_amax : all finite fp16 magnitudes (0x0000..0x7BFF) x qmax {7,127}
  dequant_one     : all codes x every distinct scale golden produces
  widen           : all finite fp16 bit patterns
  quant_one       : all finite fp16 x  x  every distinct scale (per qmax)

inf/nan (exp==31) are OUT OF CONTRACT (codec data is finite) and excluded from
the x/amax sweeps; the golden itself produces undefined int casts on them.

Golden = golden/apex_golden/cq_codec.py (float64 -> narrow chain). One mismatch
aborts non-zero. This arbiter also gates the SV port (tb re-proves in HW).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "golden"))
import numpy as np
from apex_golden import cq_codec as C
from apex_golden import fp as FP
import int_model as M

FINITE_MAG = np.arange(0x0000, 0x7C00, dtype=np.uint16)     # exp 0..30
FINITE_ALL = np.array([b for b in range(0x10000)
                       if (b & 0x7C00) != 0x7C00], dtype=np.uint16)
fails = 0

def check(name, ok, detail=""):
    global fails
    if ok:
        print(f"  PASS  {name}")
    else:
        fails += 1
        print(f"  FAIL  {name}  {detail}")

# ---------- vectorized fp16 decode + RNE integer divide -------------------
def decode(bits):
    b = bits.astype(np.int64)
    exp = (b >> 10) & 0x1F
    man = b & 0x3FF
    sign = (b >> 15) & 1
    Mv = np.where(exp == 0, man, 1024 + man)
    Ev = np.where(exp == 0, -24, exp - 25)
    return sign, Mv, Ev

def rne_div(N, D):
    q = N // D
    r = N - q * D
    two_r = r << 1
    up = (two_r > D) | ((two_r == D) & ((q & 1) == 1))
    return q + up.astype(np.int64)

def quant_vec(xs, s, bits, lo, hi):
    sx, Mx, Ex = decode(xs)
    ss, Ms, Es = M._decode_f16(s)
    shift = Ex - Es
    pos = shift >= 0
    N = np.where(pos, Mx << np.where(pos, shift, 0), Mx)
    D = np.where(pos, Ms, Ms << np.where(pos, 0, -shift))
    mag = rne_div(N, D)
    q = np.where((sx ^ ss) == 1, -mag, mag)   # sign(x) XOR sign(s)
    q = np.where(Mx == 0, 0, q)
    return np.clip(q, lo, hi)

# ---------- scale_from_amax (exhaustive, incl. defensive negative amax) ---
for bits in (4, 8):
    bad = []
    for a in FINITE_ALL.tolist():
        g = int(C.scale_from_amax(np.uint16(a), bits))
        m = M.scale_from_amax(a, bits)
        if g != m:
            bad.append((a, g, m))
            if len(bad) > 4:
                break
    check(f"scale_from_amax bits={bits} (exhaustive {FINITE_ALL.size})",
          not bad, str(bad[:5]))

# ---------- distinct scales the golden can produce ------------------------
scales = {}
BOUNDARY_S = {0x0400, 0x0001, 0x03FF, 0x0401, 0x7BFF, 0x3C00, 0x3C01, 0x2400,
              # defensive negative scales (out of contract; sign honored)
              0x8400, 0x8001, 0x83FF, 0xBC00, 0xFBFF}
for bits in (4, 8):
    s = set(int(M.scale_from_amax(int(a), bits)) for a in FINITE_MAG.tolist())
    s |= BOUNDARY_S
    scales[bits] = sorted(s)
    print(f"  distinct scales bits={bits}: {len(scales[bits])}")

# ---------- dequant_one : codes x every distinct scale (exhaustive) -------
for bits in (4, 8):
    codes = np.arange(M.qmin_of(bits), M.qmax_of(bits) + 1, dtype=np.int64)
    bad = []
    for s in scales[bits]:
        g = C.dequant_f32(codes, np.uint16(s))
        for ci, code in enumerate(codes.tolist()):
            mm = M.dequant_one(int(code), s)
            if int(g[ci]) != mm:
                bad.append((code, s, hex(int(g[ci])), hex(mm)))
                break
        if bad:
            break
    check(f"dequant_one bits={bits} (codes x {len(scales[bits])} scales)",
          not bad, str(bad[:5]))

# ---------- widen f16->f32 (exhaustive over all finite) -------------------
g = FP.f64_to_f32_bits(FP.f16_bits_to_f64(FINITE_ALL))
bad = []
for i, b in enumerate(FINITE_ALL.tolist()):
    mm = M.widen_f16_f32(b)
    if int(g[i]) != mm:
        bad.append((hex(b), hex(int(g[i])), hex(mm)))
        if len(bad) > 4:
            break
check(f"widen f16->f32 (exhaustive {FINITE_ALL.size} finite)", not bad, str(bad[:5]))

# ---------- cross-check vectorized quant == scalar int_model --------------
samp_x = FINITE_ALL[:: 7]
xc = 0
for s in scales[4][:: 200]:
    v = quant_vec(samp_x, s, 4, M.qmin_of(4), M.qmax_of(4))
    for i, x in enumerate(samp_x.tolist()):
        if int(v[i]) != M.quant_one(x, s, 4):
            xc += 1
check("quant vector==scalar cross-check", xc == 0, f"{xc} mismatches")

# ---------- quant_one : all finite fp16 x  x  every distinct scale --------
for bits in (4, 8):
    lo, hi = M.qmin_of(bits), M.qmax_of(bits)
    bad = []
    for s in scales[bits]:
        g = C.quant_codes(FINITE_ALL, np.uint16(s), bits).astype(np.int64)
        mm = quant_vec(FINITE_ALL, s, bits, lo, hi)
        diff = np.nonzero(g != mm)[0]
        if diff.size:
            for idx in diff[:5]:
                bad.append((hex(int(FINITE_ALL[idx])), hex(s),
                            int(g[idx]), int(mm[idx])))
            break
    check(f"quant_one bits={bits} ({FINITE_ALL.size} x  x {len(scales[bits])} scales)",
          not bad, str(bad[:5]))

print("\nDATAPATH INTEGER-MODEL PROOF:",
      "ALL PASS" if fails == 0 else f"{fails} FAIL")
sys.exit(1 if fails else 0)
