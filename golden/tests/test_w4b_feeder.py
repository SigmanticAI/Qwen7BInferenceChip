#!/usr/bin/env python3
"""test_w4b_feeder.py — W4-B given-scale feeder oracle gates (D-031).

Gates wfeed_w4b_to_i8 (docs/design/W4B_FEEDER.md stage 0):
  A. REDUCTION IDENTITY — with s8 = the tile-amax scale, the given-scale
     oracle is bit-identical to the landed arbiter wfeed_w4_to_i8("B")
     (which derives that scale internally). This is what licenses the RTL
     to verify against ONE oracle in both host modes.
  B. STRIPE-SHARED-SCALE — K-split segments requantized independently
     under one host scale equal the whole-stripe result (the §8.1 D-021
     delta this oracle exists for).
  C. GIVEN-SCALE CORNERS — all-zero tiles, saturating small scales
     (both clamp rails), washing-out large scales, EPS.
  D. DETERMINISM.
House idiom: check() lines, verbatim-parsable, nonzero exit on any FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from apex_golden.weight_codec import (  # noqa: E402
    compress_weights_w4, wfeed_w4_to_i8, wfeed_w4b_to_i8)
from apex_golden.fp import f64_to_f16_bits, f16_bits_to_f64  # noqa: E402

n_pass = n_fail = 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global n_pass, n_fail
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name}" + (f" — {extra}" if extra else ""))
    n_pass += ok
    n_fail += (not ok)


def rnd_w8(rng, K, N):
    return rng.integers(-128, 128, size=(K, N)).astype(np.int64)


def main() -> int:
    rng = np.random.default_rng(0xD030)

    print("[A] reduction identity: w4b(s8=tile scale) == realization 'B'")
    ok = True
    shapes = [(8, 1), (24, 5), (33, 3), (64, 8), (100, 7), (256, 8),
              (2048, 8), (48, 2)]
    for G in (16, 32):
        for (K, N) in shapes:
            for _ in range(3):
                blob = compress_weights_w4(rnd_w8(rng, K, N), group=G)
                iB, s8 = wfeed_w4_to_i8(blob, "B")
                got = wfeed_w4b_to_i8(blob, s8)
                ok &= bool(np.array_equal(got, iB))
    check("identity over 8 shapes x G in {16,32} x 3 seeds", ok)

    print("[B] stripe-shared-scale: segments under one host scale == whole")
    ok = True
    for G in (16, 32):
        for segK in ([64, 64, 64], [96, 32, 128], [2048, 2048]):
            N = 8
            Wseg = [rnd_w8(rng, k, N) for k in segK]
            blobs = [compress_weights_w4(w, group=G) for w in Wseg]
            # host scale: tile-amax rule over the CONCATENATED stripe reals
            whole = compress_weights_w4(np.vstack(Wseg), group=G)
            _, s8 = wfeed_w4_to_i8(whole, "B")
            per_seg = np.concatenate(
                [wfeed_w4b_to_i8(b, s8) for b in blobs])
            got_whole = wfeed_w4b_to_i8(whole, s8)
            # compare on live elements only (per-segment streams carry
            # their own KB-tail padding; the whole-stripe stream pads once)
            def live(vals, blobs_or_blob):
                if isinstance(blobs_or_blob, list):
                    out, off = [], 0
                    for b in blobs_or_blob:
                        from apex_golden.weight_codec import (
                            n_wgt_beats, stream_index)
                        n = n_wgt_beats(b.K, b.N) * 8
                        r, _ = stream_index(b.K, b.N)
                        out.append(vals[off:off + n][r < b.K])
                        off += n
                    return np.concatenate(out)
                from apex_golden.weight_codec import stream_index
                r, _ = stream_index(blobs_or_blob.K, blobs_or_blob.N)
                return vals[r < blobs_or_blob.K]
            ok &= bool(np.array_equal(live(per_seg, blobs),
                                      live(got_whole, whole)))
    check("3 stripe splits x G in {16,32}: live elements identical", ok)

    print("[C] given-scale corners")
    blob0 = compress_weights_w4(np.zeros((64, 8), dtype=np.int64), group=32)
    any_s = np.uint16(f64_to_f16_bits(np.array([1.0]))[0])
    check("all-zero tile -> all-zero codes at any scale",
          bool(np.all(wfeed_w4b_to_i8(blob0, any_s) == 0)))
    blob = compress_weights_w4(rnd_w8(rng, 64, 8), group=32)
    tiny = np.uint16(f64_to_f16_bits(np.array([2.0 ** -14]))[0])
    q_tiny = wfeed_w4b_to_i8(blob, tiny)
    check("tiny scale saturates BOTH rails",
          bool(q_tiny.max() == 127 and q_tiny.min() == -128),
          f"max={q_tiny.max()} min={q_tiny.min()}")
    huge = np.uint16(f64_to_f16_bits(np.array([60000.0]))[0])
    check("huge scale washes to zero",
          bool(np.all(wfeed_w4b_to_i8(blob, huge) == 0)))
    _, s8n = wfeed_w4_to_i8(blob, "B")
    sv = float(f16_bits_to_f64(np.array([s8n]))[0])
    check("nominal scale keeps codes in [-128,127] with both signs live",
          bool(wfeed_w4b_to_i8(blob, s8n).max() > 0 >
               wfeed_w4b_to_i8(blob, s8n).min()), f"s={sv:.3e}")

    print("[D] determinism")
    a = wfeed_w4b_to_i8(blob, s8n)
    b = wfeed_w4b_to_i8(blob, s8n)
    check("bit-identical on repeat", bool(np.array_equal(a, b)))

    print("=" * 72)
    verdict = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILURE(S)"
    print(f"W4B FEEDER ORACLE (D-031): {n_pass} checks, {verdict}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
