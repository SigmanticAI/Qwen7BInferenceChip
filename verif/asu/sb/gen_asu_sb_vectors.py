#!/usr/bin/env python3
"""gen_asu_sb_vectors.py — INDEPENDENT ASU verification vector generator.

Verifier-owned (does NOT reuse the implementer's smoke vectors). Numerics come
from the golden arbiter golden/apex_golden/compute.py:
  * softmax rows      -> apex_golden.compute.online_softmax_fx (the arbiter)
  * exp LUT error     -> independently re-measured vs numpy exp (D-014 budget)
  * softmax quality   -> independently measured vs softmax_ref (ARCH §6
                         acceptance: <= 2 ULP of Q1.15 per element)
RMSNorm has no fixed-point twin in the arbiter yet (compute.py: "fixed-point
twin lands with the RTL"), so the oracle is THIS FILE's independent
re-implementation of the implementer's documented pseudocode
(rtl/asu/asu_rmsnorm.sv header). It is cross-checked here against
(a) the implementer's own mirror (transcription-error guard) and
(b) the float64 arbiter rmsnorm_ref (quality sanity, reported not gated).

Outputs (in --outdir):
  vectors_sm_directed.txt    softmax: directed rows + rejects   (SCORE_FRAC=0)
  vectors_sm_random.txt      softmax: >=520 seeded random rows + rejects
  vectors_sm_reset.txt       softmax: phase-targeted mid-op resets
  vectors_sm_f10.txt         softmax: SCORE_FRAC=10 config (LUT interpolation
                             exercised INSIDE the softmax datapath)
  vectors_rms_directed.txt   rmsnorm: directed rows (ties, extremes) + rejects
  vectors_rms_random.txt     rmsnorm: >=520 seeded random rows + rejects
  vectors_rms_reset.txt      rmsnorm: phase-targeted mid-op resets

Record formats (parsed by tb_asu_softmax_sb.sv / tb_asu_rmsnorm_sb.sv):
  softmax:  R <n> <nresc> <flags>   then n 'S <8hex>' + n 'P <4hex>' lines
            O <n>                   oversize reject row (TB synthesizes data)
            X <n> <phase> <abort>   mid-op reset job (phase 0 = cycle-count)
  rmsnorm:  N <d> <flags>           then d 'D <2hex>' + d 'G <4hex>' +
                                    d 'Y <4hex>' lines + 'Q <r> <nrm>'
            B <d>                   non-pow-2 length reject (TB synthesizes)
            O <d>                   oversize reject (>128)
            Z <d> <phase> <abort>   mid-op reset job

softmax R flags: b0 dup_max, b1 has_clamp(diff<=-17), b2 all_equal,
                 b3 clamp_boundary(diff==-16 exactly), b4 int32_extreme,
                 b5 has_negative, b6 lut_frac_nonzero
rmsnorm N flags: b0 tie_odd, b1 tie_even, b2 x_extreme(+-128/127),
                 b3 all_zero_x, b4 g_zero_lane, b5 g_extreme
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
import numpy as np  # noqa: E402

from apex_golden.compute import (  # noqa: E402
    FRAC_BITS, OUT_FRAC, exp_fx, online_softmax_fx, rmsnorm_ref, softmax_ref,
)

SM_ROW_MAX = 1024
RMS_D_MAX = 128
I32_MIN, I32_MAX = -(2**31), 2**31 - 1


# ── independent D-014 / §6 quality audits ────────────────────────────────────

def audit_exp_lut() -> float:
    """Max abs error of exp_fx vs float64 exp over the full contract domain."""
    z = np.arange(-(16 << FRAC_BITS), 1, dtype=np.int64)
    fx = exp_fx(z).astype(np.float64) / (1 << OUT_FRAC)
    ref = np.exp(z.astype(np.float64) / (1 << FRAC_BITS))
    return float(np.max(np.abs(fx - ref)))


def softmax_ulp(row: list[int], frac: int, p_fx: list[int]) -> int:
    """Max |p_fx - rne(softmax_ref * 2^15)| for one row (ARCH §6: <= 2)."""
    ref = softmax_ref(np.array(row, dtype=np.float64) / (1 << frac))
    tgt = np.rint(ref * (1 << OUT_FRAC)).astype(np.int64)
    return int(np.max(np.abs(np.array(p_fx, dtype=np.int64) - tgt)))


# ── RMSNorm oracle: independent re-implementation of the documented contract ─

def rmsnorm_fx_iv(x: list[int], g: list[int]) -> tuple[list[int], int, int]:
    """Verifier's own transcription of the asu_rmsnorm.sv header pseudocode."""
    d = len(x)
    assert 1 <= d <= RMS_D_MAX and (d & (d - 1)) == 0
    sum2 = sum(int(v) * int(v) for v in x)
    assert sum2 <= 2**21                    # width proof from the header
    mean2 = sum2 >> int(math.log2(d))       # exact pow-2 floor shift
    n2 = mean2 + 1                          # EPS_INT = 1
    den = math.isqrt(n2)                    # rsqrt_unit phase A: floor sqrt
    r = (1 << 13) // den                    # phase B: floor division, UQ2.13
    nrm = min(den, 2**14 - 1)
    y = []
    for i in range(d):
        q = int(x[i]) * int(g[i]) * r       # frac 13+13 = 26
        t = q >> 18                         # floor to Q7.8
        rem = q & (2**18 - 1)
        if rem > 2**17 or (rem == 2**17 and (t & 1)):
            t += 1                          # RNE (C-2)
        y.append(max(-32768, min(32767, t)))
    return y, r, nrm


# ── emit helpers ─────────────────────────────────────────────────────────────

def hx(v: int, width_bits: int) -> str:
    return f"{int(v) & ((1 << width_bits) - 1):0{width_bits // 4}x}"


def sm_flags(row: list[int], frac: int) -> tuple[int, int]:
    m = max(row)
    nresc = 0
    cur = None
    for v in row:
        if cur is None:
            cur = v
        elif v > cur:
            nresc += 1
            cur = v
    fl = 0
    if row.count(m) >= 2:
        fl |= 1
    sh = FRAC_BITS - frac
    diffs = [(v - m) << sh for v in row]
    if any(dv < -16384 for dv in diffs):
        fl |= 2
    if len(set(row)) == 1 and len(row) > 1:
        fl |= 4
    if any(dv == -16384 for dv in diffs):
        fl |= 8
    if any(abs(v) >= 2**30 for v in row):
        fl |= 16
    if any(v < 0 for v in row):
        fl |= 32
    if any((dv + (16 << FRAC_BITS)) & 63 for dv in diffs if dv >= -16384):
        fl |= 64
    return nresc, fl


class SmWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.rows = 0
        self.rejects = 0
        self.max_ulp = 0
        self.ulp_viol: list[tuple[int, int]] = []

    def row(self, row: list[int], frac: int) -> None:
        assert 1 <= len(row) <= SM_ROW_MAX
        assert all(I32_MIN <= v <= I32_MAX for v in row)
        p = [int(v) for v in online_softmax_fx(np.array(row, np.int64), frac)]
        assert all(0 <= v <= 32768 for v in p)
        u = softmax_ulp(row, frac, p)
        self.max_ulp = max(self.max_ulp, u)
        if u > 2:
            self.ulp_viol.append((self.rows, u))
        nresc, fl = sm_flags(row, frac)
        self.lines.append(f"R {len(row)} {nresc} {fl}")
        self.lines += [f"S {hx(v, 32)}" for v in row]
        self.lines += [f"P {hx(v, 16)}" for v in p]
        self.rows += 1

    def reject(self, n: int) -> None:
        assert n > SM_ROW_MAX
        self.lines.append(f"O {n}")
        self.rejects += 1

    def reset(self, n: int, phase: int, abort: int) -> None:
        self.lines.append(f"X {n} {phase} {abort}")

    def save(self, path: Path) -> None:
        path.write_text("# generated by gen_asu_sb_vectors.py (verifier)\n"
                        + "\n".join(self.lines) + "\n")


class RmsWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.rows = 0
        self.rejects = 0
        self.max_abs_t = 0
        self.max_float_dev = 0.0

    def row(self, x: list[int], g: list[int], flags_extra: int = 0) -> None:
        y, r, nrm = rmsnorm_fx_iv(x, g)
        # honest bound audit: does the golden ever saturate? (see RESULT.md)
        raw_t = [int(x[i]) * int(g[i]) * r >> 18 for i in range(len(x))]
        self.max_abs_t = max(self.max_abs_t, max(abs(t) for t in raw_t))
        # float-arbiter sanity (reported, not gated): compare vs rmsnorm_ref
        # with the SAME integer-domain eps=1.0 the RTL contract defines
        xf = np.array(x, dtype=np.float64)
        if np.any(xf != 0):
            ref = xf * (np.mean(xf * xf) + 1.0) ** -0.5 \
                * (np.array(g, np.float64) / 8192.0)
            dev = float(np.max(np.abs(np.array(y) / 256.0 - ref)))
            scale = max(1.0, float(np.max(np.abs(ref))))
            self.max_float_dev = max(self.max_float_dev, dev / scale)
        fl = flags_extra
        if any(v in (-128, 127) for v in x):
            fl |= 4
        if all(v == 0 for v in x):
            fl |= 8
        if any(v == 0 for v in g) and any(v != 0 for v in x):
            fl |= 16
        if any(v in (-32768, 32767) for v in g):
            fl |= 32
        self.lines.append(f"N {len(x)} {fl}")
        self.lines += [f"D {hx(v, 8)}" for v in x]
        self.lines += [f"G {hx(v, 16)}" for v in g]
        self.lines += [f"Y {hx(v, 16)}" for v in y]
        self.lines.append(f"Q {r} {nrm}")
        self.rows += 1

    def bad_len(self, d: int) -> None:
        assert d <= RMS_D_MAX and (d & (d - 1)) != 0
        self.lines.append(f"B {d}")
        self.rejects += 1

    def oversize(self, d: int) -> None:
        assert d > RMS_D_MAX
        self.lines.append(f"O {d}")
        self.rejects += 1

    def reset(self, d: int, phase: int, abort: int) -> None:
        self.lines.append(f"Z {d} {phase} {abort}")

    def save(self, path: Path) -> None:
        path.write_text("# generated by gen_asu_sb_vectors.py (verifier)\n"
                        + "\n".join(self.lines) + "\n")


# ── RNE tie engineering (independent of the implementer's search) ────────────

def find_tie_gamma(x: list[int], r: int, want_odd: bool,
                   skip: set[int]) -> tuple[int, int] | None:
    """Find (lane, gamma) with x*g*r == 2^17 (mod 2^18) and floor-quotient
    parity as requested — discriminates RNE from truncate AND from
    round-half-up (C-2 mutation targets)."""
    for j in range(len(x)):
        if j in skip or x[j] == 0:
            continue
        for g in range(-32768, 32768):
            q = x[j] * g * r
            if (q & (2**18 - 1)) == 2**17 and ((q >> 18) & 1) == int(want_odd):
                return j, g
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0xA5D_2026)

    # ── audit: D-014 LUT error budget (independent re-measurement) ─────────
    lut_err = audit_exp_lut()
    print(f"AUDIT exp LUT: max |exp_fx - exp| = {lut_err:.6e} "
          f"(budget 2^-10 = {2**-10:.6e}) -> "
          f"{'OK' if lut_err <= 2**-10 else 'VIOLATION'}")
    assert lut_err <= 2**-10, "D-014 error budget violated"

    # ═════════════════ softmax: directed (SCORE_FRAC = 0) ═════════════════
    sm = SmWriter()
    sm.row([42], 0)                                # n=1 -> p=0x8000
    sm.row([-123456789], 0)                        # n=1 negative
    sm.row([7, 7], 0)                              # n=2 equal -> 16384 each
    sm.row([I32_MIN, I32_MAX], 0)                  # n=2 extreme spread
    sm.row([I32_MIN, I32_MIN, I32_MAX, I32_MAX], 0)
    sm.row([I32_MAX, I32_MIN], 0)                  # descending extreme order
    sm.row([0] * 100, 0)                           # all-equal, all-zero
    sm.row(list(range(1000, 1064)), 0)             # ascending: 63 rescales
    sm.row(list(range(1063, 999, -1)), 0)          # descending: 0 rescales
    sm.row([1000, 984, 983, 1000], 0)              # -16 boundary + -17 clamp
    sm.row([500, 484, 483, 482, 500, 500], 0)      # triplicate max + clamps
    sm.row([-5, -3, -21, -3, -4], 0)               # negatives, dup max
    sm.row([1000] + [970] * 30, 0)                 # deep-tail row (p=0 tail)
    base = int(rng.integers(-1000, 1000))
    sm.row([base + int(v) for v in rng.integers(-15, 1, size=SM_ROW_MAX)], 0)
    sm.reject(SM_ROW_MAX + 1)                      # reject, last ON overflow
    sm.row([3, 1, 2], 0)                           # clean row right after
    sm.reject(SM_ROW_MAX + 40)                     # reject via FLUSH path
    sm.row([9, 9, 9, 8], 0)
    sm.save(out / "vectors_sm_directed.txt")
    print(f"sm_directed: {sm.rows} rows, {sm.rejects} rejects, "
          f"max ULP vs float64 = {sm.max_ulp}")
    d_ulp, d_viol = sm.max_ulp, list(sm.ulp_viol)

    # ═════════════════ softmax: >=520 seeded random rows ══════════════════
    smr = SmWriter()
    n_random = 520
    for i in range(n_random):
        pick = rng.random()
        if pick < 0.60:
            n = int(rng.integers(1, 17))
        elif pick < 0.90:
            n = int(rng.integers(17, 65))
        elif pick < 0.98:
            n = int(rng.integers(65, 301))
        else:
            n = int(rng.integers(990, SM_ROW_MAX + 1))
        style = rng.random()
        if style < 0.70:      # clustered (non-trivial probabilities)
            b = int(rng.integers(I32_MIN + 50, I32_MAX - 50))
            row = [b + int(v) for v in rng.integers(-40, 2, size=n)]
        elif style < 0.90:    # medium spread (mostly clamped tail)
            b = int(rng.integers(-10**6, 10**6))
            row = [b + int(v) for v in rng.integers(-2000, 1, size=n)]
        else:                 # wild full-range INT32
            row = [int(v) for v in
                   rng.integers(I32_MIN, I32_MAX, size=n, dtype=np.int64)]
        row = [min(I32_MAX, max(I32_MIN, v)) for v in row]
        smr.row(row, 0)
        if i % 26 == 25:      # 20 interleaved oversize rejects
            smr.reject(SM_ROW_MAX + 1 + int(rng.integers(0, 2)) * 37)
    smr.save(out / "vectors_sm_random.txt")
    print(f"sm_random  : {smr.rows} rows, {smr.rejects} rejects, "
          f"max ULP vs float64 = {smr.max_ulp}")

    # ═════════════════ softmax: SCORE_FRAC = 10 config ════════════════════
    smf = SmWriter()
    smf.row([0, -512, -1024, -1536, -333, -77], 10)      # sub-LSB structure
    smf.row([16384, 16000, 15873, 511, 0, -16384], 10)
    smf.row([1] * 8, 10)
    for _ in range(150):
        n = int(rng.integers(1, 49))
        b = int(rng.integers(-2**24, 2**24))
        row = [b + int(v) for v in rng.integers(-16384, 1, size=n)]
        row = [min(I32_MAX, max(I32_MIN, v)) for v in row]
        smf.row(row, 10)
    smf.save(out / "vectors_sm_f10.txt")
    print(f"sm_f10     : {smf.rows} rows, {smf.rejects} rejects, "
          f"max ULP vs float64 = {smf.max_ulp}")

    # ═════════════════ softmax: phase-targeted mid-op resets ══════════════
    smx = SmWriter()
    # phases: 1=COLLECT 2=FLUSH 3=LOAD 4=DIV 5=PUSH 6=DRAIN (0=cycle-count)
    for phase in (1, 3, 4, 5, 6):
        for n in (8, 33):
            smx.reset(n, phase, 0)
            smx.row([100 + i for i in range(5)], 0)     # clean row after
    smx.reset(SM_ROW_MAX + 60, 2, 0)                    # FLUSH-phase reset
    smx.row([1, 2, 3, 2], 0)
    for a in (3, 17, 61):                               # random-cycle aborts
        smx.reset(24, 0, a)
        smx.row([50, 49, 48], 0)
    smx.save(out / "vectors_sm_reset.txt")
    print(f"sm_reset   : {smx.rows} verify rows, 11 targeted + 3 cycle resets")

    # ═════════════════ rmsnorm: directed ══════════════════════════════════
    rms = RmsWriter()
    rms.row([0], [32767])                          # all-zero x, D=1
    rms.row([0], [-32768])
    rms.row([-128], [-32768])                      # extreme product +2^22
    rms.row([127], [32767])
    rms.row([-128], [12288])                       # implementer's T4 shape
    rms.row([0] * 128, [int(v) for v in rng.integers(-32768, 32768, 128)])
    rms.row([7, -7], [0, 16384])                   # zero-gamma lane, x != 0
    # every legal power-of-two length
    for d in (1, 2, 4, 8, 16, 32, 64, 128):
        x = [int(v) for v in rng.integers(-128, 128, size=d)]
        g = [int(v) for v in rng.integers(-32768, 32768, size=d)]
        rms.row(x, g)
    # max-ratio rows (honest saturation-reachability probe, both signs)
    xs = [0] * 128
    xs[7] = 127
    gs = [0] * 128
    gs[7] = 32767
    rms.row(xs, gs)
    xs2 = [0] * 128
    xs2[100] = -128
    gs2 = [0] * 128
    gs2[100] = -32768
    rms.row(xs2, gs2)
    # engineered RNE ties (independent search; discriminates trunc AND
    # round-half-up mutants of the C-2 rule)
    x_t = [int(v) for v in rng.integers(-128, 128, size=64)]
    x_t[0], x_t[1] = -128, 127
    _, r_t, _ = rmsnorm_fx_iv(x_t, [0] * 64)
    t_odd = find_tie_gamma(x_t, r_t, True, {0, 1})
    t_even = find_tie_gamma(x_t, r_t, False,
                            {0, 1} | ({t_odd[0]} if t_odd else set()))
    assert t_odd and t_even, "no RNE tie lanes found"
    g_t = [int(v) for v in rng.integers(-32768, 32768, size=64)]
    g_t[t_odd[0]], g_t[t_even[0]] = t_odd[1], t_even[1]
    y_t, _, _ = rmsnorm_fx_iv(x_t, g_t)
    q_o = x_t[t_odd[0]] * t_odd[1] * r_t
    q_e = x_t[t_even[0]] * t_even[1] * r_t
    assert y_t[t_odd[0]] == (q_o >> 18) + 1, "odd tie must round up"
    assert y_t[t_even[0]] == (q_e >> 18), "even tie must hold"
    rms.row(x_t, g_t, flags_extra=(1 | 2))
    print(f"rms ties   : lanes {t_odd[0]} (odd->up) / {t_even[0]} (even->hold)"
          f" r={r_t}")
    # framing rejects
    rms.bad_len(3)
    rms.row([5, -5], [8192, 8192])                 # clean row after reject
    rms.bad_len(127)
    rms.bad_len(12)
    rms.oversize(129)                              # last ON the overflow beat
    rms.oversize(160)                              # FLUSH path
    rms.row([1, 2, 3, 4], [8192] * 4)
    rms.save(out / "vectors_rms_directed.txt")
    print(f"rms_directed: {rms.rows} rows, {rms.rejects} rejects, "
          f"max|t|={rms.max_abs_t} (sat16 needs >32767), "
          f"max rel dev vs float ref = {rms.max_float_dev:.4f}")
    rms_maxt_d = rms.max_abs_t

    # ═════════════════ rmsnorm: >=520 seeded random rows ══════════════════
    rmr = RmsWriter()
    pow2 = [1, 2, 4, 8, 16, 32, 64, 128]
    for i in range(520):
        d = pow2[int(rng.integers(0, len(pow2)))]
        style = rng.random()
        if style < 0.70:
            x = [int(v) for v in rng.integers(-128, 128, size=d)]
        elif style < 0.85:
            x = [int(v) for v in rng.integers(-4, 5, size=d)]     # tiny norms
        else:
            x = [int(rng.choice([-128, 127]))
                 if rng.random() < 0.5 else int(rng.integers(-128, 128))
                 for _ in range(d)]
        g = [int(v) for v in rng.integers(-32768, 32768, size=d)]
        rmr.row(x, g)
        if i % 40 == 39:                           # 13 interleaved rejects
            if rng.random() < 0.5:
                bad = int(rng.integers(3, RMS_D_MAX))
                while (bad & (bad - 1)) == 0:
                    bad += 1
                rmr.bad_len(bad)
            else:
                rmr.oversize(RMS_D_MAX + 1 + int(rng.integers(0, 40)))
    rmr.save(out / "vectors_rms_random.txt")
    print(f"rms_random : {rmr.rows} rows, {rmr.rejects} rejects, "
          f"max|t|={rmr.max_abs_t}, "
          f"max rel dev vs float ref = {rmr.max_float_dev:.4f}")

    # ═════════════════ rmsnorm: phase-targeted mid-op resets ══════════════
    rmz = RmsWriter()
    # phases: 1=COLLECT 2=FLUSH 3=ISSUE 4=WAIT 5=EMIT 6=DRAIN (0=cycle-count)
    for phase in (1, 3, 4, 5, 6):
        for d in (16, 64):
            rmz.reset(d, phase, 0)
            x = [int(v) for v in rng.integers(-128, 128, size=8)]
            g = [int(v) for v in rng.integers(-32768, 32768, size=8)]
            rmz.row(x, g)                          # clean row after
    rmz.reset(RMS_D_MAX + 30, 2, 0)                # FLUSH-phase reset
    rmz.row([9, -9], [8192, -8192])
    for a in (5, 23, 71):                          # random-cycle aborts
        rmz.reset(32, 0, a)
        rmz.row([1, -1, 2, -2], [16384] * 4)
    rmz.save(out / "vectors_rms_reset.txt")
    print(f"rms_reset  : {rmz.rows} verify rows, 11 targeted + 3 cycle resets")

    # ═════ cross-check: verifier's rmsnorm oracle vs implementer's mirror ═══
    # (transcription-error guard only; both implement the documented contract)
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "impl_gen", Path(__file__).parent.parent / "smoke/gen_asu_vectors.py")
    impl = ilu.module_from_spec(spec)
    spec.loader.exec_module(impl)
    for _ in range(200):
        d = pow2[int(rng.integers(0, len(pow2)))]
        x = [int(v) for v in rng.integers(-128, 128, size=d)]
        g = [int(v) for v in rng.integers(-32768, 32768, size=d)]
        assert rmsnorm_fx_iv(x, g) == impl.rmsnorm_fx(x, g), \
            f"oracle divergence at D={d}"
    print("AUDIT rmsnorm oracle: 200 random rows identical to the "
          "implementer's documented mirror (independent transcription)")

    # ═════════════════ summary / gates ═════════════════════════════════════
    all_ulp = max(d_ulp, smr.max_ulp, smf.max_ulp)
    print(f"AUDIT softmax quality: max ULP vs float64 across all rows = "
          f"{all_ulp} (ARCH §6 acceptance: <= 2)")
    viol = d_viol + smr.ulp_viol + smf.ulp_viol
    if viol:
        print(f"AUDIT softmax quality VIOLATIONS: {viol[:10]}")
    maxt = max(rms_maxt_d, rmr.max_abs_t)
    print(f"AUDIT rmsnorm saturation: max |pre-sat t| observed = {maxt} "
          f"(sat16 at 32768+; structural bound ~11.9k at D<=128 -> "
          f"sat buckets unreachable, clamp is defensive)")
    if all_ulp > 2:
        print("GEN: WARNING - softmax ULP budget exceeded (see RESULT.md)")
    print("GEN: all vector files written")


if __name__ == "__main__":
    main()
