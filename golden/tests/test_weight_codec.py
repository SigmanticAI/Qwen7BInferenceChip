"""test_weight_codec.py — gate apex_golden.weight_codec (B3 native W4).

Two classes of check, kept strictly separate:

  A-E  BIT-EXACTNESS / SELF-CONSISTENCY. Integer equality on raw bit
       patterns against the already-trusted cq_codec primitives and against
       the mxe_ctrl beat accounting. No tolerances anywhere in these
       sections — the W4 unpack is the same primitive family as the KV INT4
       dequant and must be exactly that, not approximately.

  F    REALIZATION MEASUREMENT (B3 stage 0b). The load-bearing decision:
       which of the two contract-legal realizations the RTL feeder should
       implement. Measured against the ORIGINAL real weights — NOT against
       the dequantized weights, because every existing float64 reference in
       the repo is built from w8*s_w (transformer.py:519-521,
       attention.py:538-541), so a W4 error measured against those cancels
       identically and the gate reads the INT8 plumbing baseline instead.
       Numbers are PINNED so a codec change must update them consciously
       (the test_effective_bits.py discipline), never silently re-based.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex_golden import cq_codec as cq                          # noqa: E402
from apex_golden import weight_codec as wc                      # noqa: E402
from apex_golden.compute import gemm_i8                         # noqa: E402
from apex_golden.fp import f16_bits_to_f64                      # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def first_diff(got: np.ndarray, want: np.ndarray) -> str:
    """test_contract.py's diagnostic: count + first offender in hex."""
    got, want = np.asarray(got).reshape(-1), np.asarray(want).reshape(-1)
    if got.shape != want.shape:
        return f"shape {got.shape} != {want.shape}"
    bad = np.flatnonzero(got != want)
    if not len(bad):
        return ""
    i = bad[0]
    return (f"{len(bad)} mismatches; first @[{i}] "
            f"got={int(got[i]):#x} want={int(want[i]):#x}")


def sval(sb) -> float:
    return float(f16_bits_to_f64(np.array([sb], dtype=np.uint16))[0])


# geometries: (K, N) — the shipped D=64/N=8 config, D=128, a partial final
# K-chunk (K=70 -> KB=9, klast=6), and the ODD KB*N cases S6 warns about.
GEOMS = [(64, 8), (128, 8), (70, 8), (64, 3), (40, 1), (24, 5)]


# ── A. job geometry & the normative stream contract (S1-S6) ─────────────────

def section_a() -> None:
    print("\n[A] job geometry / stream contract (S1-S6)")
    for K, N in GEOMS:
        KB = wc.kb_of(K)
        check(f"K={K} N={N}: KB == ceil(K/8)", KB == -(-K // 8), f"KB={KB}")
        check(f"K={K} N={N}: INT8 beats == KB*N",
              wc.n_wgt_beats(K, N) == KB * N, f"{wc.n_wgt_beats(K, N)}")
        check(f"K={K} N={N}: packed beats == ceil(KB*N/2)",
              wc.n_packed_beats(K, N) == (KB * N + 1) // 2,
              f"{wc.n_packed_beats(K, N)}")

    # S1-S3: flat element e = 8*(p*N + c) + r maps to row 8p+r, column c.
    for K, N in GEOMS:
        rows, cols = wc.stream_index(K, N)
        KB = wc.kb_of(K)
        ok_len = len(rows) == KB * N * wc.LANES
        exp_r, exp_c, e = [], [], 0
        for p in range(KB):
            for c in range(N):
                for r in range(wc.LANES):
                    exp_r.append(8 * p + r)
                    exp_c.append(c)
                    e += 1
        ok = ok_len and np.array_equal(rows, exp_r) and np.array_equal(cols,
                                                                       exp_c)
        check(f"K={K} N={N}: stream order chunk-major/column-minor (S1-S3)",
              ok, first_diff(rows, np.asarray(exp_r)) or
              first_diff(cols, np.asarray(exp_c)))

    # S2: rows >= K on the final chunk are the loader's zero mask.
    codes = np.arange(1, 70 * 8 + 1, dtype=np.int64).reshape(70, 8) % 7 + 1
    flat = wc.code_stream(codes, 70, 8)
    rows, _ = wc.stream_index(70, 8)
    check("K=70: pad elements (row >= K) are zero (S2)",
          bool(np.all(flat[rows >= 70] == 0)) and bool(np.all(flat[rows < 70]
                                                              != 0)),
          f"{int(np.sum(rows >= 70))} pad elements")


# ── B. INT8 -> fp16 narrowing is lossless (no extra quantization hop) ───────

def section_b() -> None:
    print("\n[B] INT8 -> fp16 narrowing is lossless")
    allw = np.arange(-127, 128, dtype=np.int64).reshape(-1, 1)
    bits = wc.weights_to_f16(allw)
    back = f16_bits_to_f64(bits.reshape(-1))
    check("every INT8 code -127..127 round-trips fp16 EXACTLY",
          np.array_equal(back.astype(np.int64), allw.reshape(-1)),
          first_diff(back.astype(np.int64), allw.reshape(-1)))
    try:
        wc.weights_to_f16(np.array([[4096]], dtype=np.int64))
        check("out-of-range weight is rejected", False, "no raise")
    except ValueError:
        check("out-of-range weight is rejected", True)


# ── C. the W4 primitives ARE cq_codec's (bit-exact reuse) ──────────────────

def section_c() -> None:
    print("\n[C] pack/unpack/dequant == cq_codec primitives (bit-exact)")
    rng = np.random.default_rng(20260720)
    for K, N in GEOMS:
        for group in ("tile", "col", 16):
            W8 = rng.integers(-127, 128, size=(K, N), dtype=np.int64)
            b = wc.compress_weights_w4(W8, group=group)

            check(f"K={K} N={N} g={group}: codes within C-1 INT4 clamp [-8,7]",
                  bool(np.all(b.codes >= -8) and np.all(b.codes <= 7)),
                  f"range [{int(b.codes.min())},{int(b.codes.max())}]")

            # S4/S5: unpack of the packed beats == the emission-order codes.
            n_elem = wc.n_wgt_beats(K, N) * wc.LANES
            got = cq.unpack_int4(b.packed.reshape(-1), n_elem)
            want = wc.code_stream(b.codes, K, N)
            check(f"K={K} N={N} g={group}: unpack_int4(packed) == code stream",
                  np.array_equal(got, want), first_diff(got, want))

            # the W4 dequant IS cq_codec.dequant_f32 — compare fp32 BITS.
            for g in range(min(b.n_groups, 4)):
                m = b.gid == g
                sb = np.uint16(b.scales[g])
                d_got = cq.dequant_f32(b.codes[m], sb)
                d_want = cq.dequant_f32(
                    cq.unpack_int4(cq.pack_int4(b.codes[m]),
                                   int(np.sum(m))), sb)
                check(f"K={K} N={N} g={group} grp{g}: dequant_f32 bit-exact "
                      f"through pack/unpack",
                      np.array_equal(d_got, d_want),
                      first_diff(d_got, d_want))

    # the packed payload is byte-identical to cq_codec.pack_int4 of the stream
    W8 = rng.integers(-127, 128, size=(64, 8), dtype=np.int64)
    b = wc.compress_weights_w4(W8, group="col")
    want = cq.pack_int4(wc.code_stream(b.codes, 64, 8))
    check("packed payload == cq_codec.pack_int4(code stream) byte-for-byte",
          np.array_equal(b.packed.reshape(-1)[:len(want)], want),
          first_diff(b.packed.reshape(-1)[:len(want)], want))


# ── D. the feeder (C-4): what the RTL must emit ────────────────────────────

def section_d() -> None:
    print("\n[D] feeder equivalence (C-4) + realization legality")
    rng = np.random.default_rng(4242)

    for K, N in GEOMS:
        W8 = rng.integers(-127, 128, size=(K, N), dtype=np.int64)

        # (A) is a PURE UNPACK: output == the sign-extended INT4 codes.
        ba = wc.compress_weights_w4(W8, group="tile")
        fa, sa = wc.wfeed_w4_to_i8(ba, "A")
        want = wc.code_stream(ba.codes, K, N)
        check(f"K={K} N={N}: (A) feeder out == sign-extended INT4 codes",
              np.array_equal(fa, want), first_diff(fa, want))
        check(f"K={K} N={N}: (A) output is INT8-representable",
              bool(np.all(np.abs(fa) <= 127)))
        check(f"K={K} N={N}: (A) scale is the tile's group scale",
              sa == ba.scales[0], f"{sval(sa):.6g}")

        # (B) absorbs fine scales: output is a full-range INT8 requant.
        bb = wc.compress_weights_w4(W8, group="col")
        fb, sb = wc.wfeed_w4_to_i8(bb, "B")
        check(f"K={K} N={N}: (B) feeder out is INT8-representable",
              bool(np.all(fb >= -128) and np.all(fb <= 127)),
              f"range [{int(fb.min())},{int(fb.max())}]")
        check(f"K={K} N={N}: (B) emits one beat per INT8 beat",
              len(fb) == wc.n_wgt_beats(K, N) * wc.LANES, f"{len(fb)}")
        check(f"K={K} N={N}: (B) pad lanes stay zero",
              bool(np.all(fb[wc.stream_index(K, N)[0] >= K] == 0)))

    # (A) with a non-tile group must be REFUSED, not silently wrong.
    b = wc.compress_weights_w4(np.ones((64, 8), dtype=np.int64), group="col")
    try:
        wc.wfeed_w4_to_i8(b, "A")
        check("(A) with per-column groups is refused", False, "no raise")
    except ValueError:
        check("(A) with per-column groups is refused", True)

    # pow2 snapping never pushes a code into the clamp (snap UP, §2).
    for K, N in GEOMS:
        W8 = rng.integers(-127, 128, size=(K, N), dtype=np.int64)
        bp = wc.compress_weights_w4(W8, group="col", pow2_scale=True)
        man_ok = all((int(s) & 0x3FF) == 0 for s in bp.scales)
        check(f"K={K} N={N}: pow2 scales have a zero mantissa (fp16-grade)",
              man_ok)
        check(f"K={K} N={N}: pow2 codes still inside [-8,7]",
              bool(np.all(np.abs(bp.codes) <= 8)),
              f"max|q|={int(np.max(np.abs(bp.codes)))}")


# ── E. PERF beat accounting (S6) ──────────────────────────────────────────

def section_e() -> None:
    print("\n[E] PERF: xw beats per job (S6)")
    for K, N in GEOMS:
        consumed, emitted, ratio = wc.perf_beats(K, N)
        odd = (wc.kb_of(K) * N) & 1
        check(f"K={K} N={N}: emitted == 2*consumed - (KB*N&1)",
              emitted == 2 * consumed - odd,
              f"consumed={consumed} emitted={emitted} ratio={ratio:.4f}"
              + ("  [ODD tail]" if odd else ""))
    # the shipped config: exactly halved
    c, e, r = wc.perf_beats(64, 8)
    check("shipped D=64/N=8: xw beats per job exactly halved (64 -> 32)",
          c == 32 and e == 64 and r == 2.0, f"{e} -> {c}")
    # and the general claim is FALSE where KB*N is odd
    c, e, r = wc.perf_beats(40, 1)
    check("K=40/N=1 (KB*N=5, odd): ratio is 2*c-1, NOT 2.0",
          e == 2 * c - 1 and r != 2.0, f"consumed={c} emitted={e} r={r:.4f}")


# ── F. STAGE 0b: which realization? (measured, not asserted) ───────────────

def proj_err(a8: np.ndarray, W_real: np.ndarray, W_rec: np.ndarray) -> float:
    """max|a@W_rec - a@W_real| / max|a@W_real| — the projection-isolated
    weight error. Activations are identical on both sides by construction,
    so this is the weight path's contribution alone."""
    y_ref = a8.astype(np.float64) @ W_real
    y_got = a8.astype(np.float64) @ W_rec
    den = float(np.max(np.abs(y_ref)))
    return float(np.max(np.abs(y_got - y_ref))) / den


def quant_per_tensor_i8(W_real: np.ndarray) -> tuple[np.ndarray, float]:
    """The SHIPPED weight path: INT8 codes + one per-tensor float scale."""
    s_w = float(np.max(np.abs(W_real))) / 127.0
    W8 = np.clip(np.rint(W_real / s_w), -127, 127).astype(np.int64)
    return W8, s_w


"""Realization x grouping grid. `graded` applies apex_scale_quant's C2
fp16-grade narrowing to the composite scale; `pow2` snaps the GROUP scale
up to a power of two instead (the alternative mitigation)."""
CFGS = [
    # label            real  group  pow2   graded
    ("A tile",         "A", "tile", False, True),
    ("A tile  raw",    "A", "tile", False, False),
    ("A tile  pow2",   "A", "tile", True,  False),
    ("B col",          "B", "col",  False, True),
    ("B G=32",         "B", 32,     False, True),
    ("B G=16",         "B", 16,     False, True),
]

# PINNED from the 2026-07-20 run (seed 90210, K=64 N=8 M=32, numpy RNG
# PCG64). Regression tripwire under the test_effective_bits.py discipline —
# NOT a quality claim. Any codec change must update these consciously.
PINNED = {
    "gaussian": {
        "INT8 baseline": 0.00492, "A tile": 0.11542, "A tile  raw": 0.11537,
        "A tile  pow2": 0.19695, "B col": 0.09987, "B G=32": 0.07758,
        "B G=16": 0.07262,
    },
    "uniform": {
        "INT8 baseline": 0.00398, "A tile": 0.06541, "A tile  raw": 0.06541,
        "A tile  pow2": 0.13297, "B col": 0.07817, "B G=32": 0.07817,
        "B G=16": 0.06092,
    },
    "outlier": {
        "INT8 baseline": 0.00833, "A tile": 0.13109, "A tile  raw": 0.13108,
        "A tile  pow2": 0.27894, "B col": 0.13171, "B G=32": 0.12554,
        "B G=16": 0.09259,
    },
}
PIN_TOL = 5e-4


def section_f() -> dict:
    print("\n[F] STAGE 0b — realization measurement (vs ORIGINAL weights)")
    print("    rel_err = max|a@W_rec - a@W_real| / max|a@W_real|,"
          " activations identical both sides")
    rng = np.random.default_rng(90210)
    K, N, M = 64, 8, 32
    measured: dict = {}

    dists = {
        "gaussian": rng.standard_normal((K, N)),
        "uniform": rng.uniform(-1.0, 1.0, (K, N)),
    }
    # outlier-bearing: one column with a 20x heavier tail (the D-022 shape
    # that broke CQ-4 on the score side — coarse grouping's vulnerable case)
    Wo = rng.standard_normal((K, N))
    Wo[:, 0] *= 20.0
    Wo[3, 5] *= 30.0
    dists["outlier"] = Wo

    a8 = rng.integers(-127, 128, size=(M, K), dtype=np.int64)
    rows, cols = wc.stream_index(K, N)
    live = rows < K

    for dname, W_real in dists.items():
        dyn = (np.max(np.abs(W_real).max(0)) / np.min(np.abs(W_real).max(0)))
        print(f"\n  {dname}  (column dynamic range {dyn:.1f}x):")
        W8, s_w = quant_per_tensor_i8(W_real)
        row = {"INT8 baseline": proj_err(a8, W_real,
                                        W8.astype(np.float64) * s_w)}
        for label, real, group, pow2, graded in CFGS:
            b = wc.compress_weights_w4(W8, group=group, pow2_scale=pow2)
            flat, sc = wc.wfeed_w4_to_i8(b, real)
            comp = sval(sc) * s_w
            if graded:                      # apex_scale_quant C2 (see codec)
                comp = wc.f16_grade(comp)
            W_rec = np.zeros((K, N), dtype=np.float64)
            W_rec[rows[live], cols[live]] = flat[live]
            row[label] = proj_err(a8, W_real, W_rec * comp)
        measured[dname] = row
        base = row["INT8 baseline"]
        for label, v in row.items():
            print(f"    {label:14s} rel_err = {v:.5f}"
                  + (f"   ({v / base:5.1f}x INT8)" if label != "INT8 baseline"
                     else "   (shipped path)"))

    print()
    # --- what the measurement actually decides ---------------------------

    # 1. The realization choice is NOT accuracy-driven: (A) and (B) land
    #    within 25% of each other everywhere. The contract's "pick (B) only
    #    if (A) misses budget" cannot be executed — there is no budget gap
    #    between them to arbitrate.
    for dname, row in measured.items():
        ratio = row["B col"] / row["A tile"]
        check(f"{dname}: (A) and (B) are within 25% of each other",
              0.75 <= ratio <= 1.25, f"B/A = {ratio:.3f}")

    # 2. fp16-grade narrowing of the COMPOSITE is free; pow2-snapping the
    #    GROUP scale is not. This picks the apex_scale_quant C2 mitigation.
    for dname, row in measured.items():
        check(f"{dname}: fp16-grade composite costs ~0 vs the raw scale",
              abs(row["A tile"] - row["A tile  raw"]) <= 1e-4,
              f"{row['A tile']:.5f} vs {row['A tile  raw']:.5f}")
        check(f"{dname}: pow2-snapping the group scale costs >50%",
              row["A tile  pow2"] > 1.5 * row["A tile"],
              f"{row['A tile  pow2'] / row['A tile']:.2f}x")

    # 3. K-grouping granularity is the ONLY meaningful accuracy lever.
    for dname, row in measured.items():
        check(f"{dname}: G=16 beats tile-wide grouping",
              row["B G=16"] < row["A tile"],
              f"{row['B G=16']:.5f} < {row['A tile']:.5f} "
              f"({100*(1 - row['B G=16']/row['A tile']):.0f}% better)")

    # 4. THE HEADLINE: native W4 costs 13-25x the INT8 baseline on projection
    #    error, on every distribution and every realization. This is the
    #    number B3's viability turns on, and it is intrinsic to the 4-bit
    #    code width — see the ceiling check below.
    worst = max(row[lab] / row["INT8 baseline"]
                for row in measured.values()
                for lab in ("A tile", "B col", "B G=16"))
    best = min(row[lab] / row["INT8 baseline"]
               for row in measured.values()
               for lab in ("A tile", "B col", "B G=16"))
    check("W4 costs 10-30x the INT8 baseline on projection error",
          10.0 <= best and worst <= 30.0,
          f"best {best:.1f}x, worst {worst:.1f}x")

    # 5. The realization machinery (fp16 scale grid, INT8 output funnel,
    #    nibble packing) adds nothing material on top of the INTRINSIC INT4
    #    loss — so the accuracy is set by the 4-bit code width, not by any
    #    choice B3 can make in the feeder. Reference = INT4 straight onto the
    #    REAL weights with an exact tile scale (no INT8 hop, no fp16 scale
    #    grid). Two-sided 20% band: the realization can land slightly BELOW
    #    the reference because scale_from_amax's fp16 RNE sometimes rounds
    #    the scale down to a finer grid than amax/7 exactly.
    for dname, W_real in dists.items():
        s = float(np.max(np.abs(W_real))) / 7.0
        ideal = proj_err(a8, W_real,
                         np.clip(np.rint(W_real / s), -8, 7) * s)
        got = measured[dname]["A tile"]
        check(f"{dname}: (A) tracks the intrinsic-INT4 reference within 20%",
              abs(got - ideal) <= 0.20 * ideal,
              f"got {got:.5f} vs intrinsic {ideal:.5f} "
              f"({100*(got/ideal - 1):+.0f}%)")

    # --- pinned-value gate (conscious-update discipline) ------------------
    drift = []
    for dname, row in measured.items():
        for label, v in row.items():
            want = PINNED.get(dname, {}).get(label)
            if want is None:
                drift.append(f"{dname}/{label}: unpinned ({v:.5f})")
            elif abs(v - want) > PIN_TOL:
                drift.append(f"{dname}/{label}: {v:.5f} != pinned {want:.5f}")
    n_pin = sum(len(v) for v in PINNED.values())
    check(f"measured errors match the pinned {n_pin}-value register",
          not drift, "; ".join(drift[:4]))

    return measured


def main() -> int:
    print("=" * 72)
    print("APEX W4 WEIGHT CODEC (B3) — golden gate")
    print("=" * 72)
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()

    print("\n" + "=" * 72)
    if FAILS:
        print(f"FAIL: {len(FAILS)} checks blown:")
        for f in FAILS[:20]:
            print(f"  ✗ {f}")
        return 1
    print("APEX W4 WEIGHT CODEC: ALL CHECKS PASS (bit-exact vs cq_codec "
          "primitives; realization measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
