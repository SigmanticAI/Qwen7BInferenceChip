#!/usr/bin/env python3
"""gen_swiglu_vectors.py — golden vectors for apex_layer_deq + asu_swiglu
(IB-LAYER S3).

Arbiters:
  deq    : plain IEEE — expected fp32 = float32(acc * comp) where the f64
           product is EXACT and fp32-representable (asserted per legal
           element); inexact/overrange elements are REFUSAL cases.
  swiglu : golden/apex_golden/transformer.py — expected p[i] =
           f64_to_f16_bits( silu_apply(acc_g[i]*comp_g) * (acc_u[i]*comp_u) )
           — silu_apply IS the golden C-SILU chain (Q5.10 RNE + LUT + one
           f16 RNE), and both dequants are float64-EXACT because the
           composites are fp16-graded (D-030 lemma).

Job classes (counts printed): see the per-file writers below. Every graded
composite is produced by weight_codec.f16_grade (the C2 primitive).

Files:
  build/vectors_deq.txt     JOB <cols> <comp8> <mode>  then acc lines,
                            then EXP lines (n_exp of them)
                            mode: OK / REFUSE_JOB / REFUSE_AT_<i> /
                                  FRAMEE_<pos> / FRAMEM_<extra>
  build/vectors_swiglu.txt  JOB <cols> <comp_g8> <comp_u8> <mode>
                            then G lines, U lines, EXP lines
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "golden"))

import numpy as np  # noqa: E402

from apex_golden import transformer as tf  # noqa: E402
from apex_golden.fp import f64_to_f16_bits  # noqa: E402
from apex_golden.weight_codec import f16_grade  # noqa: E402


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def f32_val(bits: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", bits))[0])


def graded_bits(v: float) -> int:
    g = f16_grade(v)
    b = f32_bits(g)
    assert (b & 0x1FFF) == 0 and 0 < ((b >> 23) & 0xFF) < 0xFF and b >> 31 == 0
    return b


def k_e8(bits: int):
    return ((0x800000 | (bits & 0x7FFFFF)) >> 13), ((bits >> 23) & 0xFF)


def exact_f32(acc: int, comp_bits: int):
    """(is_exact, bits) mirroring the deq contract (span + exponent range)."""
    if acc == 0:
        return True, 0
    k, e8 = k_e8(comp_bits)
    prod = abs(acc) * k
    p = prod.bit_length() - 1
    l = (prod & -prod).bit_length() - 1
    biased = p + e8 - 10
    ok = (p - l) < 24 and 1 <= biased <= 254
    if not ok:
        return False, 0
    v = np.float32(np.float64(acc) * np.float64(f32_val(comp_bits)))
    assert float(v) == float(np.float64(acc) * np.float64(f32_val(comp_bits)))
    return True, f32_bits(float(v))


def swiglu_expected(acc_g, acc_u, cg_bits, cu_bits):
    g = np.float64(acc_g) * np.float64(f32_val(cg_bits))
    u = np.float64(acc_u) * np.float64(f32_val(cu_bits))
    p = tf.silu_apply(np.array([g]))[0] * u
    return int(f64_to_f16_bits(np.array([p]))[0])


def q510_tie(acc, cg_bits) -> bool:
    """RNE(gate*2^10) lands exactly on a half-LSB (integer mirror)."""
    if acc == 0:
        return False
    k, e8 = k_e8(cg_bits)
    prod = abs(acc) * k
    sh = 127 - e8
    return sh >= 1 and (prod & ((1 << sh) - 1)) == (1 << (sh - 1))


def prod_tie(silu_bits, acc_u, cu_bits) -> tuple[bool, bool]:
    """(normal_tie, subnormal_tie) of the silu*up product RNE."""
    if acc_u == 0 or (silu_bits & 0x7FFF) == 0:
        return False, False
    ks, es = ((silu_bits >> 10) & 0x1F), silu_bits & 0x3FF
    sig_s = es if ks == 0 else (1024 + es)
    e_s = -24 if ks == 0 else ks - 25
    ku, e8u = k_e8(cu_bits)
    prod = sig_s * abs(acc_u) * ku
    if prod == 0:
        return False, False
    exp = e_s + (e8u - 137)
    p = prod.bit_length() - 1
    E = p + exp
    if E >= -14:
        sh = p - 10
        if sh <= 0:
            return False, False
        return (prod & ((1 << sh) - 1)) == (1 << (sh - 1)), False
    sh = -(exp + 24)
    if sh <= 0 or sh > 60:
        return False, False
    return False, (prod & ((1 << sh) - 1)) == (1 << (sh - 1))


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0x5316)

    # ── deq vectors ──────────────────────────────────────────────────────────
    counts = {}
    with (out / "vectors_deq.txt").open("w") as fh:
        def job(cols, comp, mode, accs, exps):
            fh.write(f"JOB {cols} {comp:08x} {mode}\n")
            for a in accs:
                fh.write(f"{a & 0xFFFFFFFF:08x}\n")
            for e in exps:
                fh.write(f"{e:08x}\n")
            counts[mode.split("_")[0]] = counts.get(mode.split("_")[0], 0) + 1

        # exhaustive o8 x graded-comp sweep (every code, 16 comps across
        # the exponent range) — always exact (19-bit span, D-030 lemma)
        comps = [graded_bits(v) for v in
                 (2**-24 * 1.37, 2**-13 * 1.01, 0.0009, 0.031, 0.4437, 1.0,
                  1.7, 13.11, 977.0, 2.0**11 * 1.23, 2.0**19 * 1.61,
                  2.0**36 * 1.11, 2.0**-30 * 1.9, 6.1e-5, 3.2e-3, 90.51)]
        for c in comps:
            accs = list(range(-128, 128))
            exps = []
            for a in accs:
                ok, b = exact_f32(a, c)
                assert ok
                exps.append(b)
            job(256, c, "OK", accs, exps)
        # exact wide accs (span<=24 via trailing zeros)
        accs = [int(a) << int(s) for a, s in
                zip(rng.integers(-4096, 4096, 40), rng.integers(0, 12, 40))]
        c = graded_bits(0.00317)
        exps = []
        for a in accs:
            ok, b = exact_f32(a, c)
            assert ok, a
            exps.append(b)
        job(40, c, "OK", accs, exps)
        # refusal: inexact at element 0 (wide odd acc) and mid-job
        c = graded_bits(0.11)
        bad = (1 << 26) - 1                       # 26-bit span, odd
        assert not exact_f32(bad, c)[0]
        job(4, c, "REFUSE_AT_0", [bad, 1, 2, 3], [])
        pre = [7, -9]
        pre_e = [exact_f32(a, c)[1] for a in pre]
        job(5, c, "REFUSE_AT_2", pre + [bad, 4, 5], pre_e)
        # exponent-range refusal: OVER-range (biased > 254 — value beyond
        # f32 max). Under-range is structurally unreachable: k >= 2^10 and
        # e8 >= 1 force biased = p + e8 - 10 >= 1; the RTL's >10 check is
        # defensive for that side.
        c_huge = graded_bits(2.0**126 * 1.5)
        assert not exact_f32(1 << 20, c_huge)[0]
        job(2, c_huge, "REFUSE_AT_0", [1 << 20, 1], [])
        # grade refusals (no inputs driven)
        for cb in (f32_bits(0.1),                 # ungraded
                   0x80000000 | graded_bits(0.5),  # negative
                   0x00000000,                    # zero
                   0x7F800000):                   # inf
            job(3, cb, "REFUSE_JOB", [], [])
        # frame errors — violation engineered at beat 0 so the expected
        # output count is exactly zero (violations later in a job emit the
        # legal prefix; that shape is covered by REFUSE_AT_2 above)
        c = graded_bits(1.0)
        job(6, c, "FRAMEE_0", [1], [])
        job(1, c, "FRAMEM_3", [1, 2, 3, 4], [])
    print("DEQ VECTORS: " + " ".join(f"{k}={v}" for k, v in counts.items()))

    # ── swiglu vectors ───────────────────────────────────────────────────────
    counts = {}
    with (out / "vectors_swiglu.txt").open("w") as fh:
        def job(cols, cg, cu, mode, g, u, e):
            fh.write(f"JOB {cols} {cg:08x} {cu:08x} {mode}\n")
            for a in g:
                fh.write(f"{a & 0xFFFFFFFF:08x}\n")
            for a in u:
                fh.write(f"{a & 0xFFFFFFFF:08x}\n")
            for b in e:
                fh.write(f"{b:04x}\n")
            counts[mode.split("_")[0]] = counts.get(mode.split("_")[0], 0) + 1

        def mk(cols, cg, cu, g, u, mode="OK"):
            e = [swiglu_expected(g[i], u[i], cg, cu) for i in range(cols)]
            job(cols, cg, cu, mode, g, u, e)

        # golden-composition mass: realistic scales, mixed accs (in-domain
        # AND clamp-exercising magnitudes)
        # scales chosen so gate values land mostly IN the SiLU domain
        # (clamped gates hide rounding behavior — measured: the first cut
        # of this generator saturated most mass elements and let the
        # RNE->floor mutant survive)
        for i in range(10):
            cols = int(rng.integers(3, 65))
            cg = graded_bits(float(2.0 ** rng.uniform(-24, -19)
                                   * (1 + rng.random())))
            cu = graded_bits(float(2.0 ** rng.uniform(-24, -19)
                                   * (1 + rng.random())))
            g = [int(v) for v in rng.integers(-(1 << 22), 1 << 22, cols)]
            u = [int(v) for v in rng.integers(-(1 << 22), 1 << 22, cols)]
            mk(cols, cg, cu, g, u)
        # zeros battery
        cg, cu = graded_bits(0.001), graded_bits(0.002)
        mk(8, cg, cu, [0, 0, 5, -5, 100, -100, 0, 1],
                       [3, -3, 0, 0, -7, 7, 0, -1])
        # clamp battery: gate exactly +-8.0, straddles, far out of domain
        cg = graded_bits(2.0 ** -10)              # comp = 2^-10: acc==xq grid
        acc_at_8 = 8192                           # gate == 8.0 exactly
        mk(10, cg, cu, [acc_at_8, -acc_at_8, 8191, -8191, 8193, -8193,
                        50000, -50000, 1, -1],
                       [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                        1000, 1000])
        # Q5.10 RNE ties (both parities) — CONSTRUCTED, not searched:
        # acc*k === 2^(sh-1) (mod 2^sh) with k odd via modular inverse
        ties = []
        parities = set()
        for e8 in (112, 114, 116):
            cgt = graded_bits(float((1 + 513 / 1024) * 2.0 ** (e8 - 127)))
            k, e8r = k_e8(cgt)
            assert k % 2 == 1 and e8r == e8
            sh = 127 - e8
            inv = pow(k, -1, 1 << sh)
            for m in (0, 1):
                a0 = (((2 * m + 1) << (sh - 1)) * inv) % (1 << sh)
                a0 += 1 << sh                 # keep magnitude sane
                # a0 and a0 + 2^sh tie identically but flip keep parity
                # (k odd), so tie-to-even and tie-away differ on one of
                # each pair — the RNE->floor/away kill is GUARANTEED
                for a in (a0, a0 + (1 << sh)):
                    keep = (a * k) >> sh
                    parities.add(keep & 1)
                    for sgn in (1, -1):
                        assert q510_tie(sgn * a, cgt), (a, e8)
                        ties.append((sgn * a, cgt))
        assert parities == {0, 1}, "tie battery must cover both keep parities" 
        for i in range(0, len(ties), 8):
            ch = ties[i:i + 8]
            mk(len(ch), ch[0][1], cu, [a for a, _ in ch],
               [int(rng.integers(1, 4000)) for _ in ch])
        # product ties — CONSTRUCTED: with silu forced to exactly 1.0
        # (sig 2^10) and ku = 2^10:
        #   normal tie   : au = 2^11 + 1  -> prod = 2^31 + 2^20, msb-11 rem
        #   subnormal tie: au in {1, 3}   -> prod = au*2^20, sh = 21 via e8
        cg10 = graded_bits(2.0 ** -10)    # xq grid == acc
        xq_one = None
        for xq in range(1, 8193):         # silu_fx(xq) == 4096 -> exactly 1.0
            if int(tf.silu_fx(np.array([xq]))[0]) == 4096:
                xq_one = xq
                break
        assert xq_one is not None, "no exact-1.0 silu point"
        sb = int(f64_to_f16_bits(tf.silu_apply(np.array([xq_one * 2.0 ** -10])))[0])
        assert sb == 0x3C00
        cu_n = graded_bits(2.0 ** (120 - 127))          # k=1024, e8=120
        # 2049 -> keep 1024 (even, tie stays); 2051 -> keep 1025
        # (odd, tie rounds up) — both RNE parities
        tn_au = [2049, -2049, 2051, -2051]
        for au in tn_au:
            assert prod_tie(sb, au, cu_n)[0], au
        mk(4, cg10, cu_n, [xq_one] * 4, tn_au, mode="tieN")
        cu_s = graded_bits(2.0 ** (102 - 127))          # k=1024, e8=102
        ts_au = [1, -1, 3, -3]
        for au in ts_au:
            assert prod_tie(sb, au, cu_s)[1], au
        mk(4, cg10, cu_s, [xq_one] * 4, ts_au, mode="tieS")
        # overflow behavior: product E >= 16 -> fp16 inf (both signs)
        cu_big = graded_bits(2.0 ** 30 * 1.31)
        e = [swiglu_expected(a, u_, cg, cu_big)
             for a, u_ in ((4000, 1 << 22), (-4000, 1 << 22))]
        assert all((x & 0x7FFF) == 0x7C00 for x in e), "ovfl battery not inf"
        job(2, cg, cu_big, "OVFL", [4000, -4000], [1 << 22, 1 << 22], e)
        # legality refusals
        for cgb, cub in ((f32_bits(0.1), graded_bits(0.2)),
                         (graded_bits(0.2), f32_bits(0.3)),
                         (0x7F800000, graded_bits(0.2))):
            job(3, cgb, cub, "REFUSE_JOB", [], [], [])
        # frame errors (gate phase early-last / up phase missing-last)
        cg2, cu2 = graded_bits(0.01), graded_bits(0.02)
        job(6, cg2, cu2, "FRAMEG_0", [1], [], [])
        job(1, cg2, cu2, "FRAMEU_2", [1], [5, 6, 7], [])
    print("SWIGLU VECTORS: " + " ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
