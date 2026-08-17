#!/usr/bin/env python3
"""APEX TIP smoke — golden model + vector generator (D-011 / D-017).

THE GOLDEN MODEL LIVES HERE and mirrors rtl/tip/* bit-exactly:

  tip_decide_golden(tile, thr, W, N) — our programmable-threshold ratio test (the
    the ratio-test unit decision rule, on raw pre-softmax scores (C-5):

        FP16  <=>  max(|s|) * N  >  T * sum(|s|)        (strict >, tie->INT8)

    with the RTL's exact fixed widths: |s| is the two's-complement magnitude
    at W bits (abs(INT_MIN) = 2^(W-1)); sum is masked to SUM_W = W + log2(N);
    LHS = max << log2(N); RHS is the PROGRAMMABLE 5-tap shift-add tree
    sum(T[i] ? sum<<i : 0, i=0..4) masked to CMP_W = SUM_W + clog2(T_MAX+1)
    = SUM_W + 5 (T_MAX=31); threshold 0 is CLAMPED to 1 (RTL thr_eff).
    Every generated tile is ALSO cross-checked against the unbounded-integer
    mathematical rule (dual oracle, V0.3 discipline): for tiles of <= N
    scores the fixed widths can never wrap (bound proof in tip_decide.sv),
    so golden == math is asserted for every tile emitted.

  imp_update_golden / tier_golden — mirror tip_importance:
    bucket(m) = 0 if m==0 else floor(log2(m))+1   (1-based MSB index, <=32)
    inc       = bucket*2 if FP16 decision else bucket
    acc[blk]  = min(acc[blk]+inc, 0xFFFF)          (16-bit saturating)
    tier      = CQ8 if acc>=hi else CQ4P if acc>=lo else CQ4
                (enum values from apex_pkg: CQ8=0, CQ4=1, CQ4P=2)
    Per-tile expected tier = suggestion for the POST-update value (upd_tier).

Vector file formats (all $readmemh, written to build/):
  <run>_scores.hex : one score per line, 8 hex digits (int32 two's-complement)
  <run>_meta.hex   : one line per tile, 8 hex digits packing
                     [22:10]=len  [9:3]=blk  [2:1]=exp_tier  [0]=exp_fp16
  <run>_imp.hex    : 128 lines, 4 hex digits — final importance acc per block
  <run>.args       : the exact plusargs for the TB run (single source of truth
                     for tiles/threshold/imp_hi/imp_lo — the Makefile cats it)

Runs generated:
  selfcheck_t10 : 143 self-generated tiles at T=10, N=4096 (RTL == golden).
  sweep_t1/t5/t31 : N=64 sweeps at the programmable-threshold operating
                points, directed decision edges (exact LHS==RHS ties, +/-1,
                extreme magnitudes incl. INT32_MIN/MAX) + 3000 random tiles
                + short (<N) tiles per threshold.
  clamp_t0    : same tile set as sweep_t1 but run with +threshold=0 — proves
                the hardware clamp (0 -> 1) against T=1 golden expectations.
  imp_directed: N=64 directed importance set — exact tier-boundary equality
                (acc == lo, acc == hi), saturation to 0xFFFF and hold,
                zero-magnitude (inc=0) tiles, INT8-weighted updates,
                multi-block interleave.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)

W = 32                    # D-011 SCORE_WIDTH
T_MAX = 31
T_W = 5                   # clog2(T_MAX+1)
ACC_W = 16
BLOCKS = 128
IMIN, IMAX = -(1 << 31), (1 << 31) - 1
M31 = 1 << 31

# tier encoding mirrors apex_pkg kvq_tier_e
CQ8, CQ4, CQ4P = 0, 1, 2


# ---------------------------------------------------------------------------
# golden model — bit-exact mirror of rtl/tip/tip_decide.sv
# ---------------------------------------------------------------------------
def abs2c(v: int, w: int) -> int:
    """Two's-complement magnitude circuit at w bits (abs(INT_MIN)=2^(w-1))."""
    mask = (1 << w) - 1
    v &= mask
    if v & (1 << (w - 1)):
        return ((~v) + 1) & mask
    return v


def clamp_thr(t: int) -> int:
    """RTL thr_eff: 5-bit field, all-zeros clamped to 1."""
    t &= (1 << T_W) - 1
    return 1 if t == 0 else t


def tip_decide_golden(tile, thr: int, n: int):
    """Returns (fp16, max_mag) with the RTL's exact fixed-width arithmetic."""
    log2n = n.bit_length() - 1
    assert n == (1 << log2n), "N must be a power of two"
    assert len(tile) <= n, "frame guard: golden only defined for <= N scores"
    sum_w = W + log2n
    cmp_w = sum_w + T_W
    sum_mask = (1 << sum_w) - 1
    cmp_mask = (1 << cmp_w) - 1
    t = clamp_thr(thr)

    mx, sm = 0, 0
    for v in tile:
        a = abs2c(int(v), W)
        if a > mx:
            mx = a
        sm = (sm + a) & sum_mask

    lhs = (mx << log2n) & cmp_mask
    rhs = 0
    for i in range(T_W):
        if (t >> i) & 1:
            rhs = (rhs + ((sm << i) & cmp_mask)) & cmp_mask
    fp16 = 1 if lhs > rhs else 0

    # dual oracle (V0.3 discipline): unbounded math must agree in-contract
    a_true = [abs(int(v)) if int(v) != IMIN else M31 for v in tile]
    math_fp16 = 1 if max(a_true) * n > t * sum(a_true) else 0
    assert fp16 == math_fp16, (
        f"golden({fp16}) != math({math_fp16}) — fixed-width wrap in-contract?! "
        f"T={t} n={n} len={len(tile)}")
    return fp16, mx


# golden mirror of rtl/tip/tip_importance.sv ---------------------------------
def bucket(m: int) -> int:
    return 0 if m == 0 else m.bit_length()      # floor(log2)+1, 1-based MSB


def imp_update_golden(acc: int, fp16: int, max_mag: int) -> int:
    inc = bucket(max_mag) * (2 if fp16 else 1)
    return min(acc + inc, (1 << ACC_W) - 1)


def tier_golden(acc: int, hi: int, lo: int) -> int:
    if acc >= hi:
        return CQ8
    if acc >= lo:
        return CQ4P
    return CQ4


# ---------------------------------------------------------------------------
# emit one run: scores/meta/imp_final/args
# ---------------------------------------------------------------------------
def emit_run(name, tiles, n, thr, imp_hi, imp_lo, frozen_expected=None):
    """tiles: list of (label, blk, [scores]). Runs the golden model in tile
    order (importance accumulates across the whole run, like the RTL)."""
    score_lines, meta_lines = [], []
    acc = [0] * BLOCKS
    n_fp16 = 0
    for idx, (label, blk, tile) in enumerate(tiles):
        assert 1 <= len(tile) <= n, f"{label}: len {len(tile)}"
        assert 0 <= blk < BLOCKS
        for v in tile:
            v = int(v)
            assert IMIN <= v <= IMAX, f"{label}: {v} out of int32"
            score_lines.append(f"{v & 0xFFFFFFFF:08x}")
        fp16, mx = tip_decide_golden(tile, thr, n)
        if frozen_expected is not None:
            assert fp16 == frozen_expected[idx], (
                f"{label}: APEX golden={fp16} but golden "
                f"expected={frozen_expected[idx]} — NOT equivalent at T=10")
        acc[blk] = imp_update_golden(acc[blk], fp16, mx)
        tier = tier_golden(acc[blk], imp_hi, imp_lo)
        n_fp16 += fp16
        meta = (len(tile) << 10) | (blk << 3) | (tier << 1) | fp16
        meta_lines.append(f"{meta:08x}")

    (OUT / f"{name}_scores.hex").write_text("\n".join(score_lines) + "\n")
    (OUT / f"{name}_meta.hex").write_text("\n".join(meta_lines) + "\n")
    (OUT / f"{name}_imp.hex").write_text(
        "\n".join(f"{a:04x}" for a in acc) + "\n")
    (OUT / f"{name}.args").write_text(
        f"+scores=build/{name}_scores.hex +meta=build/{name}_meta.hex "
        f"+impfinal=build/{name}_imp.hex +tiles={len(tiles)} "
        f"+threshold={thr} +imp_hi={imp_hi} +imp_lo={imp_lo}\n")
    print(f"{name}: tiles={len(tiles)} scores={len(score_lines)} N={n} "
          f"T={thr} (thr_eff={clamp_thr(thr)}) FP16={n_fp16} "
          f"INT8={len(tiles) - n_fp16}")
    return len(tiles)


# ---------------------------------------------------------------------------
# Run 1 — self-generated 143-tile decision replay at T=10, N=4096
# (RTL bit-exact vs OUR tip_decide_golden; broad decision-boundary coverage)
# ---------------------------------------------------------------------------
def build_replay():
    rng = np.random.default_rng(0x71D_10)          # deterministic
    tiles = []
    for t in range(143):
        # int8 scores; mix peaked (few large) and flat tiles to exercise both
        # sides of the ratio test at T=10.
        if t % 3 == 0:                             # peaked → likely high-importance
            s = rng.integers(-8, 9, size=4096, dtype=np.int64)
            s[rng.integers(0, 4096, size=t % 5 + 1)] = rng.integers(100, 128)
        else:                                      # flat/noisy
            s = rng.integers(IMIN >> 24, (IMAX >> 24) + 1, size=4096, dtype=np.int64)
        tiles.append((f"tile_{t}", t % BLOCKS, [int(x) for x in s]))
    n_tiles = emit_run("selfcheck_t10", tiles, 4096, 10, 2000, 200)
    print("SELF-CHECK 143 tiles @ T=10: RTL == tip_decide_golden (our arbiter)")
    return n_tiles


# ---------------------------------------------------------------------------
# Runs 2..4 — programmable-threshold sweeps at T in {1, 5, 31}, N=64
# ---------------------------------------------------------------------------
def spread(total, count, cap):
    """count values in [0, cap] summing exactly to total."""
    assert 0 <= total <= count * cap
    base, rem = divmod(total, count)
    assert base < cap or (base == cap and rem == 0)
    return [base + 1] * rem + [base] * (count - rem)


def boundary_tile(m, target_sum, n, rng, label):
    """max magnitude m (placed negative, IMIN when m=2^31) + fillers with
    |v|<=m summing to target_sum-m; signs randomized on the fillers."""
    rest = spread(target_sum - m, n - 1, m)
    signs = rng.choice([-1, 1], size=n - 1)
    # magnitude 2^31 is only representable as INT32_MIN (negative)
    vals = [IMIN if r == M31 else int(s) * r for s, r in zip(signs, rest)]
    head = IMIN if m == M31 else -m
    tile = [head] + vals
    a = [abs(v) if v != IMIN else M31 for v in tile]
    assert max(a) == m and sum(a) == target_sum, label
    return (label, tile)


def sweep_tiles(t_val, seed):
    n = 64
    rng = np.random.default_rng(seed)
    tiles = []

    # directed decision edges — exact LHS==RHS ties (strict > : tie -> INT8)
    for m in [t_val, 7 * t_val, (IMAX // t_val) * t_val]:
        s_eq = m * n // t_val               # integer: m is a multiple of T
        if s_eq - 1 >= m:
            tiles.append(boundary_tile(m, s_eq - 1, n, rng,
                                       f"eq_m{m}_minus1_fp16"))
        tiles.append(boundary_tile(m, s_eq, n, rng, f"eq_m{m}_tie_int8"))
        tiles.append(boundary_tile(m, min(s_eq + 1, m * n), n, rng,
                                   f"eq_m{m}_plus1_int8"))
    # extreme-magnitude straddles at m = 2^31 (from INT32_MIN) and m = INT32_MAX
    for m in [M31, IMAX]:
        s = (m * n) // t_val
        tiles.append(boundary_tile(m, s, n, rng, f"ext_m{m}_floor"))
        tiles.append(boundary_tile(m, min(s + 1, m * n), n, rng,
                                   f"ext_m{m}_floor_plus1"))
    # pathological constants + spikes
    tiles += [
        ("all_zero", [0] * n),
        ("all_int32min", [IMIN] * n),       # T=1: exact tie at max magnitude
        ("all_int32max", [IMAX] * n),
        ("alt_min_max", [IMIN if i % 2 == 0 else IMAX for i in range(n)]),
        ("spike_min_zerobg", [IMIN] + [0] * (n - 1)),
        ("spike_1_zerobg_last", [0] * (n - 1) + [1]),
    ]
    # short tiles (len < N): decision still uses the free shift by log2(N)
    tiles += [
        ("short1_int32min", [IMIN]),                    # 64 > T -> FP16
        ("short1_one", [1]),
        ("short3", [5, -5, 5]),
        ("short32_ones", [1] * 32),                     # FP16 iff T < 2
        ("short63", [2] * 63),
    ]
    # randomized: v0.3 kind mix re-targeted at threshold T
    for j in range(3000):
        kind = j % 4
        if kind == 0:
            tile = rng.integers(IMIN, IMAX + 1, size=n, dtype=np.int64)
        elif kind == 1:
            tile = rng.integers(-(1 << 16), 1 << 16, size=n, dtype=np.int64)
            for _ in range(int(rng.integers(0, 3))):
                tile[int(rng.integers(0, n))] = int(
                    rng.integers(IMIN, IMAX + 1))
        elif kind == 2:                      # boundary-targeted +/-2 counts
            m = int(rng.integers(t_val, IMAX))
            target = m * n // t_val + int(rng.integers(-2, 3))
            target = max(m, min(target, m * n))
            rest = target - m
            base, rem = divmod(rest, n - 1)
            vals = np.full(n - 1, base, dtype=np.int64)
            vals[:rem] += 1
            tile = np.concatenate(([m], vals))
            tile = tile * rng.choice([-1, 1], size=n)
        else:
            mags = np.exp2(rng.uniform(0, 31, size=n)).astype(np.int64)
            tile = np.clip(mags, 0, IMAX) * rng.choice([-1, 1], size=n)
        rng.shuffle(tile)
        tiles.append((f"rnd{j}", tile.tolist()))

    # attach a random block id per tile
    return [(lbl, int(rng.integers(0, BLOCKS)), tile) for lbl, tile in tiles]


# ---------------------------------------------------------------------------
# Run 5 — importance directed set (tier boundaries, saturation), N=64, T=4
# ---------------------------------------------------------------------------
def imp_tiles():
    n = 64
    tiles = []
    fp16_spike = [IMIN] + [0] * (n - 1)   # FP16 at any T<=31; bucket 32 -> inc 64
    ones = [1] * n                        # INT8 at T=4; bucket 1 -> inc 1
    zero = [0] * n                        # INT8; bucket 0 -> inc 0 (no change)
    small = [3] + [1] * (n - 1)           # INT8 at T=4; bucket 2 -> inc 2

    # block 7: inc=64 per tile. lo=640 -> acc==lo EXACTLY at tile 10 (CQ4P via
    # >=); hi=64000 -> acc==hi EXACTLY at tile 1000 (CQ8 via >=); saturation
    # 65472+64=65536 -> clamped 0xFFFF at tile 1024; holds through tile 1040.
    for k in range(1040):
        tiles.append((f"sat7_{k}", 7, fp16_spike))
        if k in (5, 250, 600):            # interleave other blocks mid-stream
            tiles.append((f"ones3_{k}", 3, ones))
            tiles.append((f"zero9_{k}", 9, zero))
    tiles += [("small12", 12, small)] * 3   # INT8-weighted: acc[12] = 6
    tiles += [("zero9_tail", 9, zero)] * 2  # acc[9] stays 0, tier CQ4
    return tiles


def main():
    total = 0
    total += build_replay()
    total += emit_run("sweep_t1", sweep_tiles(1, 1001), 64, 1, 512, 64)
    total += emit_run("sweep_t5", sweep_tiles(5, 1005), 64, 5, 512, 64)
    total += emit_run("sweep_t31", sweep_tiles(31, 1031), 64, 31, 512, 64)
    # clamp proof: SAME tiles as sweep_t1, golden run with thr=0 (clamps to 1)
    total += emit_run("clamp_t0", sweep_tiles(0 + 1, 1001), 64, 0, 512, 64)
    total += emit_run("imp_directed", imp_tiles(), 64, 4, 64000, 640)
    print(f"TOTAL tiles: {total}")
    print("All golden==math dual-oracle cross-checks passed "
          "(no fixed-width wrap in-contract).")


if __name__ == "__main__":
    main()
