#!/usr/bin/env python3
"""gen_seam_vectors.py — golden-vector generator for the D-021 seam suite.

Stimulus AND bit-exact expected results come from the apex_golden package
(the arbiter): quant_rows_i8 (feeder, stages S-1/Q6/Q7) and score_dequant_fx
(score dequant, stage S-3/Q9) of golden/apex_golden/attention.py. The TBs
only drive, collect and compare.

Files written to <build_dir>:
  vectors_fq_{directed,random,storm,reset}_d{64,128}.txt   (feeder)
  vectors_sd_{directed,random,storm,reset}.txt             (score dequant)

Feeder record format:
  F <rows> <fz> <feps> <stie> <sbnd> <sinf> <ctie> <fsub> <fnegz>
      legal job, then rows*D  "X <8-hex fp32>" input element lines
      (row-major), rows "S <4-hex fp16>" expected scale lines, and
      rows*D/8 "E <16-hex>" expected packed-INT8 beat lines (element 8k+j
      -> byte j of beat k, the MXE act convention).
  I <rows>                          illegal job (0 or > ROWS_MAX)
  R <rows> <abortcyc> <tphase>      mid-op reset job + rows*D X lines

Score record format:
  J <cols> <rerr> <fz> <fext> <ftie> <fp53> <fund> <fsub> <fone>
      legal job, then cols "C <8-hex fp32>" composite-scale lines,
      ceil(cols/8) "A <8x 8-hex>" INT32 acc beat lines (lane t%8 = word t%8,
      pad lanes carry pseudo-random garbage), and cols "E <8-hex>" expected
      score lines (saturated +/-(2^31-1) for the rerr range-error elements).
  I <cols>                          illegal job
  R <cols> <abortcyc> <tphase>      mid-op reset job + C/A lines

The random score set is the S-3 EQUIVALENCE PROOF required by D-021: >= 1M
(acc, comp) operand pairs drawn from the real S-3 distribution (q/K rows ->
quant_rows_i8 feeders -> gemm_i8 -> composite scales), all compared bit-exact
in the TB. The directed set adds the analytic discriminators (double-rounding
cases where a no-pre-round implementation provably diverges from the f64
golden — constructed, then ASSERTED to discriminate).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden.attention import SCORE_FRAC, quant_rows_i8, score_dequant_fx  # noqa: E402
from apex_golden.cq_codec import EPS  # noqa: E402
from apex_golden.compute import gemm_i8  # noqa: E402
from apex_golden.fp import (  # noqa: E402
    f16_bits_to_f64, f32_bits_to_f64, f64_to_f16_bits, f64_to_f32_bits, rne,
)

ROWS_TB_MAX = 64      # feeder TB DUT ROWS_MAX
N_TB_MAX = 1024       # score TB DUT N_MAX
SEED = 0x5EA10


def f32b(x: np.ndarray) -> np.ndarray:
    """float64 -> fp32 bits, ASSERTED exactly representable (no rounding)."""
    b = f64_to_f32_bits(np.asarray(x, dtype=np.float64))
    back = f32_bits_to_f64(b)
    assert np.array_equal(back, np.asarray(x, dtype=np.float64)), \
        "value not fp32-exact"
    return b


def snap_f32(x: np.ndarray) -> np.ndarray:
    """Snap arbitrary float64 onto the fp32 grid (the KVQ read-bus domain)."""
    return f32_bits_to_f64(f64_to_f32_bits(x))


# ── feeder vector writer ─────────────────────────────────────────────────────

class FeederVec:
    def __init__(self, d: int, rng: np.random.Generator):
        self.d = d
        self.rng = rng
        self.lines: list[str] = []
        self.n_jobs = self.n_rows = self.n_elems = 0
        self.n_illegal = self.n_reset = 0

    def job(self, xbits: np.ndarray, stie=0, sbnd=0, ctie=0, fsub=0, fnegz=0):
        """xbits: uint32 [rows, D] fp32 bit patterns (finite)."""
        rows, d = xbits.shape
        assert d == self.d and 1 <= rows <= ROWS_TB_MAX
        assert np.all(((xbits >> 23) & 0xFF) != 0xFF), "inf/NaN out of contract"
        xf = f32_bits_to_f64(xbits.reshape(-1)).reshape(rows, d)
        codes, scales = quant_rows_i8(xf)                 # GOLDEN arbiter
        amax = np.max(np.abs(xf), axis=1)
        fz = int(np.any(amax == 0.0))
        feps = int(np.any((amax > 0.0) & (scales == 0x0400)))
        sinf = int(np.any(scales == 0x7C00))
        self.lines.append(
            f"F {rows} {fz} {feps} {stie} {sbnd} {sinf} {ctie} {fsub} {fnegz}")
        for r in range(rows):
            for e in range(d):
                self.lines.append(f"X {int(xbits[r, e]):08x}")
        for r in range(rows):
            self.lines.append(f"S {int(scales[r]):04x}")
        for r in range(rows):
            for b in range(d // 8):
                w = 0
                for j in range(8):
                    w |= (int(codes[r, 8 * b + j]) & 0xFF) << (8 * j)
                self.lines.append(f"E {w:016x}")
        self.n_jobs += 1
        self.n_rows += rows
        self.n_elems += rows * d

    def illegal(self, rows: int):
        self.lines.append(f"I {rows}")
        self.n_illegal += 1

    def reset_job(self, xbits: np.ndarray, abortcyc: int, tphase: int):
        rows, d = xbits.shape
        self.lines.append(f"R {rows} {abortcyc} {tphase}")
        for r in range(rows):
            for e in range(d):
                self.lines.append(f"X {int(xbits[r, e]):08x}")
        self.n_reset += 1

    def write(self, path: Path, tag: str):
        path.write_text(
            f"# feeder vectors D={self.d} ({tag}): {self.n_jobs} jobs, "
            f"{self.n_rows} rows, {self.n_elems} elements, "
            f"{self.n_illegal} illegal, {self.n_reset} resets\n"
            + "\n".join(self.lines) + "\n")
        print(f"  {path.name}: {self.n_jobs} jobs / {self.n_rows} rows / "
              f"{self.n_elems} elems, {self.n_illegal} illegal, "
              f"{self.n_reset} resets")


def rand_row(rng: np.random.Generator, d: int) -> np.ndarray:
    """One random fp32-grid row spanning the S-1/Q6 operand space."""
    mag = 2.0 ** rng.uniform(-18.0, 15.0)
    x = rng.standard_normal(d) * mag
    if rng.random() < 0.25:                       # heavy-tail outliers
        n = int(rng.integers(1, 4))
        idx = rng.integers(0, d, size=n)
        x[idx] *= 2.0 ** rng.uniform(3.0, 8.0)
    if rng.random() < 0.3:                        # sprinkle exact zeros
        x[rng.random(d) < 0.2] = 0.0
    if rng.random() < 0.05:                       # all-zero row
        x[:] = 0.0
    if rng.random() < 0.1:                        # EPS-floor row
        x = rng.standard_normal(d) * (2.0 ** rng.uniform(-24.0, -18.0))
    return snap_f32(x)


def feeder_directed(d: int, rng: np.random.Generator) -> FeederVec:
    v = FeederVec(d, rng)

    # 1. all-zero row + negative-zero row + zero/negzero mix
    z = np.zeros((3, d))
    zb = f32b(z)
    zb[1, :] |= 0x8000_0000                        # -0.0 everywhere
    zb[2, ::2] |= 0x8000_0000
    v.job(zb, fnegz=1)

    # 2. EPS-floor rows (amax/127 below/at/around 2^-14), subnormal rows
    eps_rows = []
    for amax in [127.0 * EPS,                      # exactly 2^-14 (boundary)
                 127.0 * EPS * (1.0 - 2.0 ** -20), # just below -> floor
                 127.0 * EPS * 0.5,                # E = -15 region
                 127.0 * (2.0 ** -15),
                 2.0 ** -30, 2.0 ** -100]:
        amax32 = float(f32_bits_to_f64(f64_to_f32_bits(np.array([amax])))[0])
        row = snap_f32(rng.uniform(-amax32, amax32, size=d) * 0.5)
        row[0] = amax32
        eps_rows.append(row)
    v.job(f32b(np.stack(eps_rows)))
    sub = np.zeros((2, d))
    subb = f32b(sub)
    subb[0, :8] = np.arange(1, 9, dtype=np.uint32)          # +subnormals
    subb[1, :8] = np.arange(1, 9, dtype=np.uint32) | 0x8000_0000
    subb[1, 8] = f32b(np.array([1.0]))[0]                   # normal amax
    v.job(subb, fsub=1)

    # 3. scale ties: amax = 127*(2^11 + 2f + 1)*2^(E-11) — exact f16 tie point
    tie_rows = []
    for (f, e2) in [(0, 0), (1, 0), (2, -5), (511, 3), (1022, -14),
                    (1023, -10), (346, 7), (345, 7)]:
        amax = 127.0 * (2048 + 2 * f + 1) * (2.0 ** (e2 - 11))
        row = np.zeros(d)
        row[0] = amax
        row[1:5] = snap_f32(rng.uniform(0, amax * 0.5, size=4))
        tie_rows.append(row)
    v.job(f32b(np.stack(tie_rows)), stie=1)
    # +/- 1 fp32 ulp around a tie
    near = []
    amax = 127.0 * (2048 + 2 * 100 + 1) * (2.0 ** -11)
    ab = int(f32b(np.array([amax]))[0])
    for bits in (ab - 1, ab, ab + 1):
        row = np.zeros(d)
        row[0] = float(f32_bits_to_f64(np.array([bits], dtype=np.uint32))[0])
        near.append(row)
    v.job(f32b(np.stack(near)), stie=1)

    # 4. fp16 boundary rows: max normal, tie-to-inf, overflow -> inf scale
    bnd = []
    for amax in [127.0 * 65504.0,                  # s = 0x7BFF (f16 max)
                 127.0 * 65520.0,                  # exact tie -> +inf
                 127.0 * 65519.0,                  # just below the tie
                 127.0 * 65536.0,                  # overflow -> +inf
                 3.0e38]:                          # huge fp32 -> +inf
        amax32 = float(f32_bits_to_f64(f64_to_f32_bits(np.array([amax])))[0])
        row = np.zeros(d)
        row[0] = amax32
        row[1:4] = snap_f32(np.array([1.0, -65504.0, 511.4375]))
        bnd.append(row)
    v.job(f32b(np.stack(bnd)), sbnd=1)

    # 5. code ties: elements exactly at s*(k + 0.5) for even/odd k
    ct = []
    for a0_raw in [1.0, 3.5, 300.0, 0.0117]:
        a0 = float(snap_f32(np.array([a0_raw]))[0])
        probe = np.zeros((1, d))
        probe[0, 0] = a0
        _, sb = quant_rows_i8(probe)
        sv = float(f16_bits_to_f64(np.array([sb[0]]))[0])
        row = np.zeros(d)
        row[0] = a0
        ks = [0, 1, 2, 3, 50, 51, 125, 126]
        for i, k in enumerate(ks):
            x = sv * (k + 0.5)
            if x < a0:
                row[1 + i] = x if (k % 2 == 0) else -x
        row_s = snap_f32(row)
        assert np.array_equal(row_s, row), "code-tie element not fp32-exact"
        ct.append(row)
    v.job(f32b(np.stack(ct)), ctie=1)

    # 6. size extremes + a general random block
    one = np.stack([rand_row(rng, d)])
    v.job(f32b(one))
    v.job(f32b(np.stack([rand_row(rng, d) for _ in range(ROWS_TB_MAX)])))

    # 7. illegal jobs (§3): rows = 0 and rows > ROWS_MAX
    v.illegal(0)
    v.illegal(ROWS_TB_MAX + 1)
    v.illegal(4095)
    # legal job right after the rejects (D-019 regression)
    v.job(f32b(np.stack([rand_row(rng, d) for _ in range(2)])))
    return v


def feeder_random(d: int, rng: np.random.Generator, target_elems: int) -> FeederVec:
    v = FeederVec(d, rng)
    while v.n_elems < target_elems:
        rows = int(rng.integers(1, ROWS_TB_MAX + 1))
        v.job(f32b(np.stack([rand_row(rng, d) for _ in range(rows)])))
        if v.n_jobs % 37 == 0:
            v.illegal(int(rng.choice([0, ROWS_TB_MAX + 1, 2000])))
    return v


def feeder_storm(d: int, rng: np.random.Generator) -> FeederVec:
    v = FeederVec(d, rng)
    for _ in range(6):
        rows = int(rng.integers(1, 9))
        v.job(f32b(np.stack([rand_row(rng, d) for _ in range(rows)])))
    return v


def feeder_reset(d: int, rng: np.random.Generator) -> FeederVec:
    v = FeederVec(d, rng)
    # tphase: 0 = cycle-count abort, 1..5 = target FSM state
    # (1 INGEST, 2 SCALE, 3 SPUSH, 4 DRAIN, 5 WAIT)
    for tphase in [1, 2, 3, 4, 5, 0, 0, 0, 0]:
        rows = int(rng.integers(2, 6))
        xb = f32b(np.stack([rand_row(rng, d) for _ in range(rows)]))
        abortcyc = int(rng.integers(5, rows * d * 2))
        v.reset_job(xb, abortcyc, tphase)
        # a full legal job after each recovery
        v.job(f32b(np.stack([rand_row(rng, d) for _ in range(2)])))
    return v


# ── score-dequant vector writer ──────────────────────────────────────────────

def score_expect(acc: np.ndarray, comp_bits: np.ndarray):
    """EXACT mirror of score_dequant_fx's arithmetic given the comp bits
    (acc*f64(comp) -> np.rint), extended with the RTL's documented
    saturation for the golden-assert (|score| < 2^31) violations."""
    comp = f32_bits_to_f64(np.asarray(comp_bits, dtype=np.uint32))
    prod = np.asarray(acc, dtype=np.float64) * comp
    s = rne(prod)
    ovf = np.abs(s) >= 2 ** 31
    sat = np.where(prod < 0, -(2 ** 31 - 1), 2 ** 31 - 1)
    return np.where(ovf, sat, s), ovf


def comp_bits_from(s_q: np.uint16, s_k: np.ndarray, dd: int) -> np.ndarray:
    """The S-3 composite constant, the SAME single f32 narrowing as golden."""
    sq = float(f16_bits_to_f64(np.array([s_q], dtype=np.uint16))[0])
    sk = f16_bits_to_f64(np.asarray(s_k, dtype=np.uint16))
    comp64 = sq * sk * (float(1 << SCORE_FRAC) / math.sqrt(float(dd)))
    return f64_to_f32_bits(comp64)


class ScoreVec:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.lines: list[str] = []
        self.n_jobs = self.n_ops = self.n_illegal = self.n_reset = 0
        self.n_rerr = 0

    def job(self, acc: np.ndarray, comp_bits: np.ndarray,
            fone=0, allow_rerr=False):
        acc = np.asarray(acc, dtype=np.int64)
        cb = np.asarray(comp_bits, dtype=np.uint32)
        cols = len(acc)
        assert 1 <= cols <= N_TB_MAX and len(cb) == cols
        assert np.all(np.abs(acc) <= 2 ** 31) and np.all(acc >= -(2 ** 31))
        assert np.all(((cb >> 23) & 0xFF) != 0xFF), "inf/NaN comp"
        exp, ovf = score_expect(acc, cb)
        rerr = int(ovf.sum())
        assert allow_rerr or rerr == 0, "unexpected range-error case"
        comp = f32_bits_to_f64(cb)
        prod = acc.astype(np.float64) * comp
        frac = prod - np.floor(prod)
        ftie = int(np.any((frac == 0.5) & (prod != 0)))
        mc = np.where((cb >> 23) & 0xFF, (cb & 0x7FFFFF) | 0x800000,
                      cb & 0x7FFFFF).astype(np.int64)
        fp53 = int(np.any(np.abs(acc) * mc >= 2 ** 53))
        fz = int(np.any(acc == 0))
        fext = int(np.any(np.abs(acc) >= 2 ** 30))
        fund = int(np.any((exp == 0) & (acc != 0) & (mc != 0)))
        fsub = int(np.any(((cb >> 23) & 0xFF) == 0))
        self.lines.append(f"J {cols} {rerr} {fz} {fext} {ftie} {fp53} "
                          f"{fund} {fsub} {fone}")
        for t in range(cols):
            self.lines.append(f"C {int(cb[t]):08x}")
        for b in range((cols + 7) // 8):
            ws = []
            for j in range(8):
                t = 8 * b + j
                w = int(acc[t]) & 0xFFFFFFFF if t < cols \
                    else int(self.rng.integers(0, 2 ** 32))
                ws.append(f"{w:08x}")
            self.lines.append("A " + " ".join(ws))
        for t in range(cols):
            self.lines.append(f"E {int(exp[t]) & 0xFFFFFFFF:08x}")
        self.n_jobs += 1
        self.n_ops += cols
        self.n_rerr += rerr

    def illegal(self, cols: int):
        self.lines.append(f"I {cols}")
        self.n_illegal += 1

    def reset_job(self, acc: np.ndarray, comp_bits: np.ndarray,
                  abortcyc: int, tphase: int):
        cols = len(acc)
        self.lines.append(f"R {cols} {abortcyc} {tphase}")
        for t in range(cols):
            self.lines.append(f"C {int(comp_bits[t]):08x}")
        for b in range((cols + 7) // 8):
            ws = []
            for j in range(8):
                t = 8 * b + j
                w = int(acc[t]) & 0xFFFFFFFF if t < cols \
                    else int(self.rng.integers(0, 2 ** 32))
                ws.append(f"{w:08x}")
            self.lines.append("A " + " ".join(ws))
        self.n_reset += 1

    def write(self, path: Path, tag: str):
        path.write_text(
            f"# score-dequant vectors ({tag}): {self.n_jobs} jobs, "
            f"{self.n_ops} operands, {self.n_rerr} range-error elems, "
            f"{self.n_illegal} illegal, {self.n_reset} resets\n"
            + "\n".join(self.lines) + "\n")
        print(f"  {path.name}: {self.n_jobs} jobs / {self.n_ops} operands, "
              f"{self.n_rerr} rerr, {self.n_illegal} illegal, "
              f"{self.n_reset} resets")


def s3_job(rng: np.random.Generator, cols: int, dd: int):
    """One S-3-distribution job: real q/K rows -> feeders -> gemm -> comp."""
    kmag = 2.0 ** rng.uniform(-8.0, 8.0, size=(cols, 1))
    K = rng.standard_normal((cols, dd)) * kmag
    if rng.random() < 0.3:                      # outlier channels
        K[:, rng.integers(0, dd)] *= 2.0 ** rng.uniform(3.0, 7.0)
    if rng.random() < 0.2:                      # all-zero rows -> EPS scales
        K[rng.random(cols) < 0.05] = 0.0
    q = rng.standard_normal(dd) * (2.0 ** rng.uniform(-6.0, 6.0))
    K = snap_f32(K)
    q = snap_f32(q)
    k8, s_k = quant_rows_i8(K)
    q8m, s_qa = quant_rows_i8(q[None, :])
    q8, s_q = q8m[0], np.uint16(s_qa[0])
    acc = gemm_i8(q8[None, :], k8.T)[0].astype(np.int64)
    cb = comp_bits_from(s_q, s_k, dd)
    exp_direct = score_dequant_fx(acc, s_q, s_k, dd)      # GOLDEN arbiter
    exp_mirror, ovf = score_expect(acc, cb)
    assert not ovf.any() and np.array_equal(exp_direct, exp_mirror), \
        "comp-bits mirror diverged from score_dequant_fx"
    return acc, cb


def modinv(a: int, m: int) -> int:
    return pow(a, -1, m)


def double_round_discriminators():
    """Constructed (acc, comp) pairs where the exact 56-bit product P sits so
    that the f64 significand pre-round (P -> P53) crosses an integer-rounding
    boundary of the final shift: a no-pre-round implementation provably
    diverges from the f64 golden. t = 24 (comp exponent -1 - 23)."""
    t = 24
    half = 1 << (t - 1)
    cases = []
    for mc in (0x9A0F27, 0xC13579, 0xFFFFFF):
        # (target P mod 2^t, required quotient parity): rem = half-1 with the
        # pre-round nudging P UP onto the tie needs A odd; rem = half+1 with
        # the pre-round nudging P DOWN onto the tie needs A even.
        for target_rem, a_par in ((half - 1, 1), (half + 1, 0)):
            base = target_rem * modinv(mc, 1 << t) % (1 << t)
            for k in range(1 << 12):
                amag = base + (k << t)
                if amag >= 2 ** 31:
                    break
                P = amag * mc
                if P < 2 ** 54 or P >= 2 ** 55:
                    continue
                A = P >> t
                if (A & 1) != a_par:
                    continue
                # golden path: pre-round P (2 dropped bits, RNE) then shift-RNE
                inc = (P >> 1 & 1) and ((P & 1) or (P >> 2 & 1))
                P53 = ((P >> 2) + inc) << 2
                rem = P53 & ((1 << t) - 1)
                g = (P53 >> t) + (1 if (rem > half or
                     (rem == half and ((P53 >> t) & 1))) else 0)
                # naive path: exact rational RNE of P/2^t (no pre-round)
                remn = P & ((1 << t) - 1)
                n = (P >> t) + (1 if (remn > half or
                     (remn == half and ((P >> t) & 1))) else 0)
                if g != n and g < 2 ** 31:
                    # confirm numpy f64 (the golden's engine) agrees with g
                    comp_v = mc * (2.0 ** -t)
                    assert int(np.rint(np.float64(amag) * comp_v)) == g, \
                        "f64 model mismatch in discriminator construction"
                    cases.append((amag, mc))
                    break
    assert len(cases) >= 3, "discriminator construction failed"
    accs, cbs = [], []
    for amag, mc in cases:
        cb = ((126 << 23) | (mc & 0x7FFFFF))    # exponent -1 => shift t = 24
        for sgn_a in (1, -1):
            accs.append(sgn_a * amag)
            cbs.append(cb)
    return np.array(accs, dtype=np.int64), np.array(cbs, dtype=np.uint64)


def score_directed(rng: np.random.Generator) -> ScoreVec:
    v = ScoreVec(rng)
    one = 0x3F800000                            # comp = 1.0

    # size extremes
    v.job(np.array([0]), np.array([one], dtype=np.uint32), fone=1)
    a, cb = s3_job(rng, 8, 64)
    v.job(a, cb)
    a, cb = s3_job(rng, 9, 64)                  # lane tail
    v.job(a, cb)
    a, cb = s3_job(rng, N_TB_MAX, 128)
    v.job(a, cb)

    # INT32 extremes through comp = 1.0 (identity: score == acc, exact)
    accs = np.array([2 ** 31 - 1, -(2 ** 31 - 1), 1, -1, 0,
                     2 ** 30 + 1, -(2 ** 30) - 1, 12345678], dtype=np.int64)
    v.job(accs, np.full(len(accs), one, dtype=np.uint32), fone=1)

    # integer-rounding ties: comp = 0.5 / 2^-31, odd accumulators
    half = 0x3F000000
    accs = np.array([1, -1, 3, -3, 5, 2 ** 25 + 1, -(2 ** 25 + 3)],
                    dtype=np.int64)
    v.job(accs, np.full(len(accs), half, dtype=np.uint32))
    c31 = ((127 - 31) << 23)                    # comp = 2^-31
    accs = np.array([2 ** 30, -(2 ** 30), 3 * 2 ** 29, 2 ** 30 + 1, 7],
                    dtype=np.int64)
    v.job(accs, np.full(len(accs), c31, dtype=np.uint32))

    # f64 double-rounding discriminators (the pre-round proof, header claim)
    accs, cbs = double_round_discriminators()
    v.job(accs, cbs.astype(np.uint32))

    # subnormal comp: hidden-bit-0 lane, underflow to 0
    cbs = np.array([1, 0x7FFFFF, 0x000400, 0x400000], dtype=np.uint32)
    accs = np.array([2 ** 31 - 1, -(2 ** 31 - 1), 123456, -7], dtype=np.int64)
    v.job(accs, cbs)

    # negative comp bits (sign-symmetry of RNE)
    a, cb = s3_job(rng, 16, 64)
    v.job(a, np.array([int(x) | 0x80000000 for x in cb], dtype=np.uint32))

    # range-error job: golden `assert |score| < 2^31` violations
    accs = np.array([-(2 ** 31), 2 ** 31 - 1, -(2 ** 31 - 1), 100],
                    dtype=np.int64)
    cbs = np.array([one, 0x40000000, 0x40000000, one], dtype=np.uint32)
    v.job(accs, cbs, fone=1, allow_rerr=True)

    # illegal jobs + clean job after (D-019 regression)
    v.illegal(0)
    v.illegal(N_TB_MAX + 1)
    a, cb = s3_job(rng, 24, 64)
    v.job(a, cb)
    return v


def score_random(rng: np.random.Generator, target_ops: int) -> ScoreVec:
    v = ScoreVec(rng)
    while v.n_ops < target_ops:
        dd = int(rng.choice([64, 128]))
        cols = N_TB_MAX if rng.random() < 0.9 else int(rng.integers(1, N_TB_MAX))
        a, cb = s3_job(rng, cols, dd)
        v.job(a, cb)
        if v.n_jobs % 53 == 0:
            v.illegal(int(rng.choice([0, N_TB_MAX + 1])))
    return v


def score_storm(rng: np.random.Generator) -> ScoreVec:
    v = ScoreVec(rng)
    for _ in range(8):
        a, cb = s3_job(rng, int(rng.integers(8, 128)), 64)
        v.job(a, cb)
    return v


def score_reset(rng: np.random.Generator) -> ScoreVec:
    v = ScoreVec(rng)
    # tphase: 0 = cycle-count abort, 1 LOAD, 2 PROC, 3 WAIT
    for tphase in [1, 2, 3, 0, 0, 0]:
        cols = int(rng.integers(16, 64))
        a, cb = s3_job(rng, cols, 64)
        v.reset_job(a, cb, int(rng.integers(5, cols * 3)), tphase)
        a, cb = s3_job(rng, 16, 64)
        v.job(a, cb)
    return v


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    build = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
    build.mkdir(parents=True, exist_ok=True)
    quick = "--quick" in sys.argv
    fq_target = 40_000 if quick else 520_000
    sd_target = 40_000 if quick else 1_050_000

    for d in (64, 128):
        rng = np.random.default_rng(SEED + d)
        print(f"feeder D={d}:")
        feeder_directed(d, rng).write(
            build / f"vectors_fq_directed_d{d}.txt", "directed")
        feeder_random(d, rng, fq_target).write(
            build / f"vectors_fq_random_d{d}.txt", "random")
        feeder_storm(d, rng).write(
            build / f"vectors_fq_storm_d{d}.txt", "storm")
        feeder_reset(d, rng).write(
            build / f"vectors_fq_reset_d{d}.txt", "reset")

    rng = np.random.default_rng(SEED + 3)
    print("score dequant:")
    score_directed(rng).write(build / "vectors_sd_directed.txt", "directed")
    score_random(rng, sd_target).write(build / "vectors_sd_random.txt",
                                       "random (S-3 >=1M proof)")
    score_storm(rng).write(build / "vectors_sd_storm.txt", "storm")
    score_reset(rng).write(build / "vectors_sd_reset.txt", "reset")
    print("VECTOR GENERATION DONE (golden arbiter: apex_golden.attention)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
