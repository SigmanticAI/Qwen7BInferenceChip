#!/usr/bin/env python3
"""test_w4_lane.py — gates for the W4 INGEST LANE golden (w4_lane.py).

Everything numeric defers to the landed arbiters (wfeed_w4_to_i8 /
wfeed_w4b_to_i8 / gemm_i8 / requant_i32_to_i8); these gates pin the NEW
surface: the DIRECT prep recipe, the stripe s8/segmentation semantics,
the tile wire framing, and the job-level GEMM composition.

  [A] DIRECT prep + reduction identity: prep_w4_direct's s8 IS the
      tile-amax scale — wfeed_w4b_to_i8(blob, s8) == wfeed_w4_to_i8(
      blob, "B") bit-exactly, over shapes x G in {16,32} x seeds.
  [B] stripe segmentation: G-aligned segments reassemble the whole-stripe
      W8 exactly, and segment-wise GEMM partial sums equal the full-K
      GEMM (the D-021 shared-s8 stripe semantics, matrix form).
  [C] wire framing round-trip: gs_words unpack back to the scales
      (ascending gid, LE slots, zero tail) and pw_words to the packed
      beats; beat counts match wire_beats_per_job; the packed phase's
      density is exactly 2 emitted per consumed beat (odd-tail exact
      via S6: emitted == 2*consumed - (KB*N & 1)).
  [D] composite grade: prep.composite == f16_grade(s8 * s_w) recomputed,
      and it survives the f16_grade idempotence check.
  [E] the C-2 requant composition matches an independent recomputation.
  [F] determinism: byte-identical repeat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apex_golden import w4_lane as wl  # noqa: E402
from apex_golden.compute import (gemm_i8, gemm_i8_ksplit,  # noqa: E402
                                 requant_i32_to_i8)
from apex_golden.fp import f16_bits_to_f64  # noqa: E402
from apex_golden.weight_codec import (f16_grade, n_wgt_beats,  # noqa: E402
                                      wfeed_w4_to_i8, wfeed_w4b_to_i8)

n_pass = 0
n_fail = 0


def check(name, ok):
    global n_pass, n_fail
    if ok:
        n_pass += 1
    else:
        n_fail += 1
        print(f"FAIL {name}")


SHAPES = [(64, 8), (128, 8), (41, 8), (8, 5), (96, 3), (256, 8), (33, 1)]


def main():
    rng = np.random.default_rng(0xD031)

    # [A] DIRECT prep + reduction identity
    for K, N in SHAPES:
        for G in (16, 32):
            for seed in range(3):
                r2 = np.random.default_rng(seed * 7919 + K + N + G)
                W = r2.normal(0, 0.05, (K, N))
                p = wl.prep_w4_direct(W, G=G)
                ref, s8_ref = wfeed_w4_to_i8(p.blob, "B")
                got = wfeed_w4b_to_i8(p.blob, p.s8)
                check(f"A ident K{K}N{N}G{G}s{seed}",
                      int(p.s8) == int(s8_ref) and np.array_equal(ref, got))

    # [B] stripe segmentation (G-aligned splits; shared s8)
    for Ktot, split in [(96, 64), (256, 128), (160, 32), (2048 + 32, 2048)]:
        W = rng.normal(0, 0.05, (Ktot, 8))
        p = wl.prep_w4_direct(W, G=32, k_split=split)
        check(f"B segk {Ktot}/{split}", sum(p.seg_k) == Ktot
              and all(k % 32 == 0 for k in p.seg_k[:-1]))
        Wseg = np.vstack([wl.expected_w8(b, p.s8) for b in p.segs])
        check(f"B reasm {Ktot}/{split}",
              np.array_equal(Wseg, wl.expected_w8(p.blob, p.s8)))
        A = rng.integers(-128, 128, (3, Ktot), dtype=np.int64)
        acc = np.zeros((3, 8), dtype=np.int64)
        k0 = 0
        for b, kk in zip(p.segs, p.seg_k):
            acc = acc + wl.gemm_w4b(A[:, k0:k0 + kk], b, p.s8)
            k0 += kk
        # whole-stripe reference: gemm_w4b models ONE MXE job (K <= K_MAX);
        # beyond that the golden K-split GEMM is the arbiter
        ref = gemm_i8_ksplit(A, wl.expected_w8(p.blob, p.s8))
        check(f"B ksplit {Ktot}/{split}", np.array_equal(acc, ref))

    # [C] wire framing round-trip + counts + packed density
    for K, N in SHAPES:
        W = rng.normal(0, 0.05, (K, N))
        p = wl.prep_w4_direct(W, G=32)
        gw, pw = wl.gs_words(p.blob), wl.pw_words(p.blob)
        gsb, pwb, base = wl.wire_beats_per_job(K, N, 32)
        check(f"C counts K{K}N{N}", len(gw) == gsb and len(pw) == pwb
              and base == n_wgt_beats(K, N))
        sc = []
        for w in gw:
            for i in range(4):
                sc.append((int(w) >> (16 * i)) & 0xFFFF)
        ok = sc[:p.blob.n_groups] == [int(x) for x in p.blob.scales]
        ok &= all(v == 0 for v in sc[p.blob.n_groups:])
        check(f"C gs rt K{K}N{N}", ok)
        pb = np.zeros((pwb, 8), dtype=np.uint8)
        for j, w in enumerate(pw):
            for r in range(8):
                pb[j, r] = (int(w) >> (8 * r)) & 0xFF
        check(f"C pw rt K{K}N{N}", np.array_equal(pb, p.blob.packed))
        check(f"C S6 K{K}N{N}", base == 2 * pwb - (base & 1))

    # [D] composite grade
    for s_w in (1.0, 0.0123, 7.5e-4):
        W = rng.normal(0, 0.05, (64, 8))
        p = wl.prep_w4_direct(W, G=32, s_w=s_w)
        s8v = float(f16_bits_to_f64(np.array([np.uint16(p.s8)]))[0])
        check(f"D comp sw{s_w}", p.composite == f16_grade(s8v * s_w)
              and p.composite == f16_grade(p.composite))

    # [E] requant composition vs independent recomputation
    W = rng.normal(0, 0.08, (64, 8))
    p = wl.prep_w4_direct(W, G=32)
    A = rng.integers(-128, 128, (4, 64), dtype=np.int64)
    acc = gemm_i8(A, wl.expected_w8(p.blob, p.s8))
    o = wl.gemm_w4b(A, p.blob, p.s8, rq=(54505, 24))
    check("E requant", np.array_equal(
        o, requant_i32_to_i8(acc, 54505, 24).astype(np.int64)))

    # [F] determinism
    W = rng.normal(0, 0.05, (96, 8))
    p1 = wl.prep_w4_direct(W, G=32, k_split=64)
    p2 = wl.prep_w4_direct(W, G=32, k_split=64)
    check("F determinism",
          np.array_equal(p1.blob.packed, p2.blob.packed)
          and np.array_equal(p1.blob.scales, p2.blob.scales)
          and int(p1.s8) == int(p2.s8)
          and all(np.array_equal(a.packed, b.packed)
                  for a, b in zip(p1.segs, p2.segs)))

    print(f"W4LANE RESULT: checks={n_pass + n_fail} fails={n_fail} -> "
          f"{'PASS' if n_fail == 0 else 'FAIL'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
