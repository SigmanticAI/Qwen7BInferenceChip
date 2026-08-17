#!/usr/bin/env python3
"""gen_comp_vectors.py — B1 stage-3 vectors: composite unit vs the oracle.

Stage 2 proved the walker emits the right SEQUENCE; stage 3 proves the
composite ARITHMETIC (B1_WALKER.md §5 row 3) against
verif/top/l3/walker_composite_golden.py — the same oracle stage 0 validated
bit-exactly against the live L3 op stream.

Vectors drive the composite unit through its REAL input path: each scale
enters the cache via the snoop (a record of D identical fp16 elements, whose
amax is that element, reduced by cq_fp_pkg::scale_from_amax exactly as the
KVQ engine does). Sweeping the element over every positive-normal fp16
reaches 23,377 distinct scales — near-full coverage of the reachable domain —
so the file carries only (s_q, element, expected_cs, expected_qs) and the TB
reconstructs the record itself. No scale is back-doored into the cache.

Coverage rationale (notes §2 item 7):
  - qs is exercised over every reachable s_v (exhaustive in the element sweep);
  - cs pairs each swept s_k against a rotating s_q covering exponent and
    mantissa corners, so the E32 arithmetic sees its full [105,166] range and
    the carry path;
  - directed corners force the round-up decision and the f32-vs-f64 constant
    divergence, which is what kills the m2 mutant. That divergence is
    observable ONLY at D=128 (§2 P3/P4), so both builds are generated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L3 = HERE.parent / "top" / "l3"
REPO = HERE.parents[2]
sys.path.insert(0, str(L3))
sys.path.insert(0, str(REPO / "golden"))

from walker_composite_golden import score_composite, p_requant_composite  # noqa: E402
from apex_golden.cq_codec import compress_values                          # noqa: E402

F16_MIN_NORM = 0x0400          # EPS = 2^-14, the floor every scale producer uses
F16_MAX_NORM = 0x7BFF

# s_q rotation: exponent corners + mantissa corners + a few interior values
SQ_SET = [0x0400, 0x0401, 0x07FF, 0x1C00, 0x3C00, 0x3C01, 0x3FFF,
          0x4900, 0x5555, 0x6000, 0x7BFF, 0x2AAA]


def scales_for(elements: np.ndarray, D: int) -> np.ndarray:
    """The fp16 record scale the KVQ stores (== the feeder's read-time scale
    at CQ-8) for records of D identical elements."""
    rows = np.repeat(elements.reshape(-1, 1), D, axis=1).astype(np.uint16)
    return np.asarray(compress_values(rows, 8).scales).reshape(-1).astype(np.uint16)


def build(D: int, stride: int) -> list[str]:
    elems = np.arange(F16_MIN_NORM, F16_MAX_NORM + 1, stride, dtype=np.uint16)
    sc = scales_for(elems, D)
    out = [f"CFGD {D}"]
    n_skip = 0
    for i, (e, s) in enumerate(zip(elems, sc)):
        s = int(s)
        if not (F16_MIN_NORM <= s <= F16_MAX_NORM):
            n_skip += 1                     # out of the positive-normal contract
            continue
        s_q = SQ_SET[i % len(SQ_SET)]
        out.append(f"V {s_q:04x} {int(e):04x} "
                   f"{score_composite(s_q, s, D):08x} "
                   f"{p_requant_composite(s):08x}")
    if n_skip:
        out.append(f"# skipped {n_skip} elements whose scale left the "
                   f"positive-normal contract domain")
    return out


def directed(D: int) -> list[str]:
    """Corners that discriminate the rounding decision and the constant width.

    At D=128 the constant is 64*sqrt(2) and the RTL must carry its FULL f64
    significand: an f32-rounded constant mismatches on 86,592 of the 419,629
    reachable product significands. These vectors put several of those in the
    suite explicitly so the m2 mutant cannot survive a thinned sweep.
    """
    out = [f"CFGD {D}"]
    CM64 = 0x16A09E667F3BCC
    CM32 = 0xB504F3 << 29
    found = 0
    for e in range(F16_MIN_NORM, F16_MAX_NORM + 1, 7):
        s = int(scales_for(np.array([e], dtype=np.uint16), D)[0])
        if not (F16_MIN_NORM <= s <= F16_MAX_NORM):
            continue
        for s_q in SQ_SET:
            if D == 128:
                p = (0x400 | (s_q & 0x3FF)) * (0x400 | (s & 0x3FF))
                # does the f32-grade constant round differently?
                def rn24(m):
                    nb = m.bit_length()
                    sh = nb - 24
                    hi, g = m >> sh, (m >> (sh - 1)) & 1
                    st = 1 if (m & ((1 << (sh - 1)) - 1)) else 0
                    return hi + (g & (st | (hi & 1)))
                if rn24(p * CM64) == rn24(p * CM32):
                    continue
            out.append(f"V {s_q:04x} {e:04x} "
                       f"{score_composite(s_q, s, D):08x} "
                       f"{p_requant_composite(s):08x}")
            found += 1
            break
        if found >= 64:
            break
    return out


def main(stride: int = 1) -> int:
    outdir = HERE / "build"
    outdir.mkdir(parents=True, exist_ok=True)
    for D in (64, 128):
        body = build(D, stride)
        n = sum(1 for x in body if x.startswith("V "))
        (outdir / f"comp_sweep_d{D}.txt").write_text("\n".join(body) + "\n")
        print(f"comp_sweep_d{D}.txt : {n} vectors (element stride {stride})")
        dbody = directed(D)
        dn = sum(1 for x in dbody if x.startswith("V "))
        (outdir / f"comp_dir_d{D}.txt").write_text("\n".join(dbody) + "\n")
        print(f"comp_dir_d{D}.txt   : {dn} directed corner vectors"
              + (" (f32-vs-f64 constant divergence)" if D == 128 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
