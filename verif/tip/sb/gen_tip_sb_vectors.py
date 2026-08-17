#!/usr/bin/env python3
"""gen_tip_sb_vectors.py — TIP independent-verification vector generator.

INDEPENDENT golden model (written from ARCHITECTURE.md §1 TIP row / C-5 /
D-011 / D-017 and the implementer's documented contracts in rtl/tip/*.sv —
NOT copied from verif/tip/smoke/gen_tip_vectors.py):

  decision rule (raw pre-softmax INT32 scores, |.| two's-complement
  magnitude so |INT32_MIN| = 2^31):
      FP16  <=>  max(|s|) * N  >  T_eff * sum(|s|)     (strict >, tie -> INT8)
      T_eff = threshold & 0x1F, clamped 0 -> 1
  computed here with UNBOUNDED Python integers, then cross-checked against
  the RTL's masked fixed-width arithmetic (SUM_W = 32+log2(N),
  CMP_W = SUM_W+5, 5-tap shift-add RHS) — the dual-oracle discipline. For
  tiles of <= N scores the two must agree (tip_decide.sv bound proof).

  importance rule (tip_importance.sv header, D-011 TIP-lite):
      bucket(m) = 0 if m == 0 else floor(log2(m)) + 1
      inc       = bucket * 2 if FP16 else bucket
      acc[blk]  = min(acc[blk] + inc, 0xFFFF)        (16-bit saturating)
      tier      = CQ8(0) if acc >= hi else CQ4P(2) if acc >= lo else CQ4(1)
  the per-decision expected tier is the POST-update suggestion.

Vector file format (text records, streamed by tb_tip_sb.sv):
  T <thr> <blk> <len> <fp16> <tier> <acc_hex4> <flags_hex>   then <len> S lines
  S <hex8>                       one INT32 score (two's-complement)
  F <thr> <blk> <len>            overrun tile (len > N): expect ONE frame_err
                                 pulse, NO decision, NO importance update;
                                 TB then clears + re-checks the sticky bit
  C                              drain; full 128-block CSR compare against the
                                 following 128 K lines; pulse imp_clear;
                                 verify cleared
  R <phase>                      drain; TB self-drives sacrificial stimulus,
                                 mid-op reset targeted at <phase> (1..6), then
                                 verifies clean state (model acc zeroed here)
  K <hex4>                       one importance snapshot word (x128 after C/E)
  E                              end: drain, final counters + full CSR compare
                                 against the following 128 K lines

flags bits: 0 exact LHS==RHS tie, 1 off-by-one from the tie, 2 all-zero tile,
  3 contains INT32_MIN, 4 contains INT32_MAX, 5 inc==0 (bucket 0),
  6 acc saturates to 0xFFFF at this tile, 7 acc == imp_lo exactly (post-upd),
  8 acc == imp_hi exactly (post-upd)

<name>.args files carry +vectors/+imp_hi/+imp_lo for the Makefile.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)

W = 32
T_MAX = 31
ACC_W = 16
BLOCKS = 128
IMIN = -(1 << 31)
IMAX = (1 << 31) - 1
M31 = 1 << 31
CQ8, CQ4, CQ4P = 0, 1, 2   # apex_pkg kvq_tier_e encoding

FLAG_TIE, FLAG_OFF1, FLAG_ZERO, FLAG_MIN, FLAG_MAX = 1, 2, 4, 8, 16
FLAG_INC0, FLAG_SAT, FLAG_EQLO, FLAG_EQHI = 32, 64, 128, 256


# ---------------------------------------------------------------------------
# independent golden model
# ---------------------------------------------------------------------------
def t_eff(thr: int) -> int:
    t = thr & T_MAX
    return 1 if t == 0 else t


def mag(v: int) -> int:
    """|v| as the RTL's two's-complement magnitude (|IMIN| = 2^31)."""
    assert IMIN <= v <= IMAX
    return -v if v < 0 else v


def decide(scores, thr: int, n: int):
    """(fp16, max_mag) by unbounded math, cross-checked vs RTL fixed widths."""
    t = t_eff(thr)
    mags = [mag(int(v)) for v in scores]
    mx, sm = max(mags), sum(mags)
    fp16 = 1 if mx * n > t * sm else 0

    # fixed-width cross-check (dual oracle; must agree for len <= n)
    log2n = n.bit_length() - 1
    assert (1 << log2n) == n
    assert len(scores) <= n
    sum_w, cmp_w = W + log2n, W + log2n + 5
    smk = 0
    for a in mags:
        smk = (smk + a) & ((1 << sum_w) - 1)
    lhs = (mx << log2n) & ((1 << cmp_w) - 1)
    rhs = 0
    for i in range(5):
        if (t >> i) & 1:
            rhs = (rhs + ((smk << i) & ((1 << cmp_w) - 1))) & ((1 << cmp_w) - 1)
    fixed = 1 if lhs > rhs else 0
    assert fixed == fp16, (
        f"DUAL-ORACLE SPLIT: math={fp16} fixed={fixed} t={t} n={n} "
        f"len={len(scores)} — in-contract wrap?!")
    return fp16, mx


def bucket(m: int) -> int:
    return 0 if m == 0 else m.bit_length()


def tier(acc: int, hi: int, lo: int) -> int:
    return CQ8 if acc >= hi else (CQ4P if acc >= lo else CQ4)


# ---------------------------------------------------------------------------
# emitter
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, name, n, hi, lo):
        self.name, self.n, self.hi, self.lo = name, n, hi, lo
        self.lines = []
        self.acc = [0] * BLOCKS
        self.tiles = self.frames = self.resets = 0
        self.tiers_seen = set()
        self.thr_seen = set()

    def tile(self, thr, blk, scores, tie=0, off1=0):
        n = self.n
        assert 1 <= len(scores) <= n
        fp16, mx = decide(scores, thr, n)
        inc = bucket(mx) * (2 if fp16 else 1)
        pre = self.acc[blk]
        post = min(pre + inc, (1 << ACC_W) - 1)
        self.acc[blk] = post
        tr = tier(post, self.hi, self.lo)
        flags = (FLAG_TIE if tie else 0) | (FLAG_OFF1 if off1 else 0)
        if all(int(v) == 0 for v in scores):
            flags |= FLAG_ZERO
        if any(int(v) == IMIN for v in scores):
            flags |= FLAG_MIN
        if any(int(v) == IMAX for v in scores):
            flags |= FLAG_MAX
        if inc == 0:
            flags |= FLAG_INC0
        if post == 0xFFFF and pre + inc >= 0x10000:
            flags |= FLAG_SAT
        if post == self.lo:
            flags |= FLAG_EQLO
        if post == self.hi:
            flags |= FLAG_EQHI
        self.lines.append(f"T {thr} {blk} {len(scores)} {fp16} {tr} "
                          f"{post:04x} {flags:x}")
        self.lines += [f"S {int(v) & 0xFFFFFFFF:08x}" for v in scores]
        self.tiles += 1
        self.tiers_seen.add(tr)
        self.thr_seen.add(thr & T_MAX if (thr & T_MAX) else 0)
        return fp16

    def frame(self, thr, blk, length, rng):
        assert length > self.n
        self.lines.append(f"F {thr} {blk} {length}")
        vals = rng.integers(-(1 << 20), 1 << 20, size=length)
        self.lines += [f"S {int(v) & 0xFFFFFFFF:08x}" for v in vals]
        self.frames += 1

    def snapshot(self):
        self.lines += [f"K {a:04x}" for a in self.acc]

    def clear(self):
        self.lines.append("C")
        self.snapshot()
        self.acc = [0] * BLOCKS

    def reset(self, phase):
        self.lines.append(f"R {phase}")
        self.acc = [0] * BLOCKS
        self.resets += 1

    def end(self):
        self.lines.append("E")
        self.snapshot()
        path = OUT / f"vectors_{self.name}.txt"
        path.write_text("\n".join(self.lines) + "\n")
        (OUT / f"{self.name}.args").write_text(
            f"+vectors=build/vectors_{self.name}.txt "
            f"+imp_hi={self.hi} +imp_lo={self.lo}\n")
        print(f"{self.name}: N={self.n} tiles={self.tiles} "
              f"frames={self.frames} resets={self.resets} "
              f"thr_values={len(self.thr_seen)} tiers={sorted(self.tiers_seen)}")


# ---------------------------------------------------------------------------
# tile constructors
# ---------------------------------------------------------------------------
def fill(total, count, cap):
    """count magnitudes in [0, cap] summing exactly to total."""
    assert 0 <= total <= count * cap, (total, count, cap)
    base, rem = divmod(total, count)
    assert base < cap or (base == cap and rem == 0)
    vals = [base + 1] * rem + [base] * (count - rem)
    return vals


def signed_vals(mags, rng):
    out = []
    for m in mags:
        if m == M31:
            out.append(IMIN)
        else:
            out.append(int(m) if int(rng.integers(0, 2)) else -int(m))
    return out


def sum_target_tile(m, total, length, rng):
    """max magnitude m (first element) + length-1 fillers, |sum| == total."""
    assert m <= total <= length * m
    rest = fill(total - m, length - 1, m) if length > 1 else []
    vals = [IMIN if m == M31 else -m] + signed_vals(rest, rng)
    assert max(mag(v) for v in vals) == m
    assert sum(mag(v) for v in vals) == total
    rng.shuffle(vals)
    return vals


# ---------------------------------------------------------------------------
# directed set (N=64)
# ---------------------------------------------------------------------------
def build_directed(rng):
    r = Run("directed", 64, hi=64000, lo=6400)
    n = 64

    # threshold sweep 0..31: exact tie / tie-1 (FP16) / tie+1 for every value
    for thr in range(0, T_MAX + 1):
        t = t_eff(thr)
        m = 777 * t                      # max a multiple of t -> exact tie
        s_eq = m * n // t
        r.tile(thr, thr % BLOCKS, sum_target_tile(m, s_eq, n, rng), tie=1)
        r.tile(thr, thr % BLOCKS, sum_target_tile(m, s_eq - 1, n, rng), off1=1)
        r.tile(thr, thr % BLOCKS, sum_target_tile(m, min(s_eq + 1, m * n),
                                                  n, rng), off1=1)
        # one random tile per threshold
        r.tile(thr, int(rng.integers(0, BLOCKS)),
               list(rng.integers(IMIN, IMAX + 1, size=n, dtype=np.int64)))

    # extreme magnitudes: 2^31 (INT32_MIN) exact ties for power-of-two T
    for t in (1, 2, 4, 8, 16):
        s_eq = M31 * n // t
        r.tile(t, 100, sum_target_tile(M31, s_eq, n, rng), tie=1)
        if s_eq - 1 >= M31:
            r.tile(t, 100, sum_target_tile(M31, s_eq - 1, n, rng), off1=1)
    # INT32_MAX floor boundary straddle
    for t in (3, 10, 31):
        s_fl = (IMAX * n) // t
        r.tile(t, 101, sum_target_tile(IMAX, s_fl, n, rng))
        r.tile(t, 101, sum_target_tile(IMAX, min(s_fl + 1, IMAX * n), n, rng))

    # pathological constants
    r.tile(1, 102, [0] * n)                       # all-zero: INT8, inc 0
    r.tile(1, 102, [IMIN] * n, tie=1)             # T=1 exact tie at max mag
    r.tile(2, 102, [IMAX] * n)
    r.tile(5, 102, [IMIN if i % 2 else IMAX for i in range(n)])
    r.tile(31, 103, [IMIN] + [0] * (n - 1))       # spike: FP16 at any T<=31
    r.tile(31, 103, [0] * (n - 1) + [1])          # 1-spike on zero background

    # short tiles: len < N still compares against the free shift by log2(N)
    r.tile(10, 104, [IMIN])                       # len=1: 64 > 10 -> FP16
    r.tile(10, 104, [0])                          # len=1 zero -> INT8, inc 0
    r.tile(7, 104, [3, -3, 3])
    for t in (1, 2, 4, 8, 16):                    # all-equal len = N/t: tie
        ln = n // t
        r.tile(t, 105, [-5] * ln, tie=1)
    r.tile(2, 105, [1] * 63)

    # frame errors interleaved with clean ties (state-clean-after proof)
    r.frame(10, 3, n + 1, rng)                    # overrun ON the last beat
    r.tile(10, 3, sum_target_tile(10 * 99, 99 * n, n, rng), tie=1)
    r.frame(10, 3, n + 2, rng)                    # abort + 1-beat drain
    r.frame(10, 3, n + 33, rng)                   # abort + long drain
    r.tile(10, 3, sum_target_tile(10 * 99, 99 * n - 1, n, rng), off1=1)

    # importance: INT8-weighted (x1) increments and inc=0 holds
    for _ in range(5):
        r.tile(4, 79, [1] * n)                    # INT8 at T=4, bucket 1 -> +1
    r.tile(4, 79, [0] * n)                        # inc 0, acc holds at 5

    # mid-run C: full 128-block CSR compare + clear
    r.clear()

    # saturation + exact tier-boundary equalities on block 77:
    # len-1 spike [IMIN] -> FP16 (64 > T), bucket 32 -> inc 64/tile.
    # lo=6400 -> acc==lo at tile 100; hi=64000 -> acc==hi at tile 1000;
    # 0xFFFF sat at tile 1024 (65472+64=65536 clamps); hold 8 more.
    for _ in range(1032):
        r.tile(1, 77, [IMIN])
    r.tile(1, 78, [1] * n)     # post-clear low-acc tile: CQ4 again at the end

    r.end()
    return r


# ---------------------------------------------------------------------------
# random set (N=64) — >=500 seeded randomized tiles (we emit ~1216)
# ---------------------------------------------------------------------------
def rand_tile(rng, thr, n):
    kind = int(rng.integers(0, 5))
    lsel = int(rng.integers(0, 10))
    length = 1 if lsel == 0 else (int(rng.integers(2, n)) if lsel < 3 else n)
    if kind == 0:                                  # full-range int32
        return list(rng.integers(IMIN, IMAX + 1, size=length, dtype=np.int64))
    if kind == 1:                                  # small values
        return list(rng.integers(-(1 << 12), 1 << 12, size=length,
                                 dtype=np.int64))
    if kind == 2:                                  # log-magnitude spread
        mags = np.exp2(rng.uniform(0, 31.9, size=length)).astype(np.int64)
        return signed_vals([int(min(m, IMAX)) for m in mags], rng)
    if kind == 3:                                  # small + huge spikes
        t = list(rng.integers(-255, 256, size=length, dtype=np.int64))
        for _ in range(int(rng.integers(1, 3))):
            t[int(rng.integers(0, length))] = int(
                rng.integers(IMIN, IMAX + 1))
        return t
    # kind 4: boundary-targeted near the exact tie for this thr
    t = t_eff(thr)
    m = int(rng.integers(max(t, 1), IMAX)) or 1
    s_eq = m * 64 // t
    target = s_eq + int(rng.integers(-2, 3))
    target = max(m, min(target, length * m))
    return sum_target_tile(m, target, length, rng)


def build_random(rng):
    r = Run("random", 64, hi=1500, lo=300)
    thr_seq = []
    for thr in range(0, T_MAX + 1):
        thr_seq += [thr] * 38                     # grouped: fewer TB drains
    hot = list(rng.integers(0, BLOCKS, size=12))
    k = 0
    for thr in thr_seq:
        blk = int(hot[int(rng.integers(0, 12))]) if rng.random() < 0.7 \
            else int(rng.integers(0, BLOCKS))
        r.tile(thr, blk, rand_tile(rng, thr, 64))
        k += 1
        if k % 50 == 17:                          # interleaved malformed tiles
            r.frame(thr, blk, 64 + int(rng.integers(1, 17)), rng)
        if k in (400, 900):
            r.clear()
    assert r.tiers_seen == {CQ8, CQ4, CQ4P}, r.tiers_seen
    assert len(r.thr_seen) == 32
    r.end()
    return r


# ---------------------------------------------------------------------------
# storm set (N=64) — sized for backpressure-storm / stalled-feed runs
# ---------------------------------------------------------------------------
def build_storm(rng):
    r = Run("storm", 64, hi=1500, lo=300)
    for k in range(260):
        thr = int(rng.integers(0, 32))
        blk = int(rng.integers(0, 16))
        r.tile(thr, blk, rand_tile(rng, thr, 64))
        if k % 40 == 11:
            r.frame(thr, blk, 65 + int(rng.integers(0, 8)), rng)
    r.end()
    return r


# ---------------------------------------------------------------------------
# mid-op reset set (N=64) — phase-targeted (the Layer-1 hole the smoke left)
# ---------------------------------------------------------------------------
def build_reset(rng):
    r = Run("reset", 64, hi=1000, lo=100)
    n = 64
    for rnd in range(4):
        for phase in (1, 2, 3, 4, 5, 6):
            r.reset(phase)
            # post-reset correctness: exact tie + spike + random, fresh accs
            t = (rnd * 6 + phase) % T_MAX + 1
            m = 777 * t
            r.tile(t, 5, sum_target_tile(m, m * n // t, n, rng), tie=1)
            r.tile(31, 6, [IMIN] + [0] * (n - 1))
            r.tile(t, 7, rand_tile(rng, t, n))
    r.end()
    return r


# ---------------------------------------------------------------------------
# N=4096 set — big-N build + independent frozen-143 cross-check
# ---------------------------------------------------------------------------
FROZEN = Path(__file__).parent / "frozen_v03"  # upstream-derived vectors,
# deliberately CURATED OUT of the public tree (competitor-lineage data).


def load_frozen():
    for f in ("scores.hex", "expected.hex", "labels.txt"):
        sha = hashlib.sha256((FROZEN / f).read_bytes()).hexdigest()
        print(f"  frozen {f}: sha256 {sha}")
    raw = (FROZEN / "scores.hex").read_text().split()
    exp = [int(x, 16) for x in (FROZEN / "expected.hex").read_text().split()]
    assert len(exp) == 143 and len(raw) == 143 * 4096
    sc = []
    for h in raw:
        v = int(h, 16) & 0xFF
        sc.append(v - 256 if v & 0x80 else v)
    return [sc[i * 4096:(i + 1) * 4096] for i in range(143)], exp


def build_n4096(rng):
    r = Run("n4096", 4096, hi=2000, lo=200)
    n = 4096

    if FROZEN.exists():
        tiles, exp = load_frozen()
        # INDEPENDENT re-proof of the implementer's equivalence claim: my
        # golden at T=10 must reproduce the frozen verdict for ALL 143 tiles
        for i, tl in enumerate(tiles):
            fp16, _ = decide(tl, 10, n)
            assert fp16 == exp[i], (
                f"frozen tile {i}: mine={fp16} upstream={exp[i]} — T=10 "
                "equivalence expected for all 143 tiles")
        # RTL round-trip a 24-tile subset (full 143 = smoke's replay run)
        for i in range(0, 143, 6):
            r.tile(10, (i * 3 + 5) % BLOCKS, tiles[i])
    else:
        # frozen_v03 is ABSENT in the public tree (curated out — upstream-
        # derived data). The upstream-equivalence cross-check is RETIRED
        # with it; N=4096 config coverage is kept via seeded tiles, and the
        # arbiter-correctness proof at N=4096 lives in smoke's selfcheck_t10.
        print("  frozen_v03 absent (curated out) — upstream-equivalence "
              "cross-check RETIRED; N=4096 round-trip uses seeded tiles")
        for i in range(24):
            m = int(rng.integers(500, 60000))
            s_eq = m * n // 10
            s = int(max(m, min(m * n, s_eq + int(rng.integers(-3, 4)))))
            r.tile(10, (i * 3 + 5) % BLOCKS, sum_target_tile(m, s, n, rng))

    # directed at N=4096
    for t in (1, 10, 31):
        m = 777 * t
        s_eq = m * n // t
        r.tile(t, 40, sum_target_tile(m, s_eq, n, rng), tie=1)
        r.tile(t, 40, sum_target_tile(m, s_eq - 1, n, rng), off1=1)
    r.tile(1, 41, [0] * n)                        # all-zero full tile
    r.tile(31, 41, [IMIN] + [0] * (n - 1))        # spike
    r.tile(10, 41, [IMIN])                        # len=1 at big N
    r.tile(10, 41, [7] * 100)                     # short tile
    r.frame(10, 42, n + 1, rng)                   # overrun at 4097
    r.frame(10, 42, n + 9, rng)                   # abort + drain
    r.tile(10, 42, sum_target_tile(7770, 7770 * n // 10, n, rng), tie=1)
    for _ in range(4):
        r.tile(int(rng.integers(1, 32)), int(rng.integers(0, BLOCKS)),
               list(rng.integers(IMIN, IMAX + 1, size=n, dtype=np.int64)))
    r.end()
    return r


def main():
    rng = np.random.default_rng(20260707)
    runs = [build_directed(rng), build_random(rng), build_storm(rng),
            build_reset(rng), build_n4096(rng)]
    total = sum(r.tiles for r in runs)
    print(f"TOTAL tiles={total} frames={sum(r.frames for r in runs)} "
          f"resets={sum(r.resets for r in runs)}")
    print("dual-oracle (unbounded math vs RTL fixed-width) agreed on every tile")


if __name__ == "__main__":
    sys.exit(main())
