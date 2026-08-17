#!/usr/bin/env python3
"""gen_wcomp_vectors.py — F6(i) per-KV-head composite-cache BANK vectors.

Scope, stated honestly: the composite ARITHMETIC of `seq_walker_comp` is
already proven exhaustively by verif/seq_walker (comp sweeps, 30,720 vectors
per D, both geometries, mutant-gated). This suite proves the thing that unit
CANNOT prove — that `apex_wcomp_bank` keeps H_kv independent caches and
routes snoop / s_q / request / response / flush / error by the SAME engine
select the S4b KVQ bank consumes (LEVEL_C_INTEGRATION.md §9.1 F6 part (i)).

So every expected word here still comes from the ONE arbiter — the stage-0
oracle verif/top/l3/walker_composite_golden.py (score_composite /
p_requant_composite), with the cached scale itself derived through the REAL
path: a record of D identical fp16 elements whose amax the KVQ reduces with
golden/apex_golden/cq_codec.compress_values. Nothing is re-derived here and
nothing is back-doored into a cache.

Vector file (one directive per line; the TB replays them in order):

  CFGD <D>                              build geometry guard
  NENG <n>                              engine count guard
  A <g> <rec> <s_q> <elem> <cs> <qs>    isolation grid: group g stores <elem>
                                        at record <rec> with q-scale <s_q>;
                                        after ALL stores, group g must still
                                        answer <cs>/<qs> for <rec>
  S <g_written> <g_probe> <rec> <s_q> <elem> <cs> <qs>
                                        stale probe: <rec> is written ONLY in
                                        <g_written>; a request from <g_probe>
                                        must raise err_stale and return NO
                                        response, and <g_written> must then
                                        still answer <cs>/<qs>
  F <g> <rec> <s_q> <big> <small> <cs> <qs>
                                        flush probe: a PARTIAL record of <big>
                                        beats is left in group <g>, the select
                                        moves away, snp_flush is pulsed
                                        (BROADCAST), and a full record of
                                        <small> must then compose to <cs>/<qs>
                                        with no err_frame

COVERAGE SELF-CHECK (the gate that makes the grid discriminating): at every
record index the N_ENG groups' cached scales — and both composite words —
must be PAIRWISE DISTINCT. If they were not, a bank that collapsed all four
groups onto one cache would still "pass", which is exactly the failure mode
this suite exists to catch. The generator refuses to emit a grid that does
not satisfy it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L3 = HERE.parent / "l3"
REPO = HERE.parents[2]
sys.path.insert(0, str(L3))
sys.path.insert(0, str(REPO / "golden"))

from walker_composite_golden import score_composite, p_requant_composite  # noqa: E402
from apex_golden.cq_codec import compress_values                          # noqa: E402

# record indices: both halves of the L3 store map (K at [0,T), V at [T,2T))
RECS = [0, 1, 7, 63, 127, 128, 129, 255]

# one distinct q-scale per group (exponent + mantissa corners)
SQ_G = [0x3C00, 0x0401, 0x5555, 0x7BFF]

# elements are chosen so that scale(elem) differs per group at every record
ELEM_BASE = [0x3C00, 0x4900, 0x5400, 0x6100]
ELEM_STEP = [0x0011, 0x0023, 0x0037, 0x0041]

STALE_REC = 200          # written in group 0 only
FLUSH_REC = 100          # partial-record + broadcast-flush probe (group 2)
FLUSH_BIG = 0x7000       # left half-written, must NOT survive the flush
FLUSH_SMALL = 0x3800     # the record that must define the cached scale


def scale_of(elem: int, D: int) -> int:
    """The fp16 scale the KVQ stores for a record of D identical elements —
    the golden packer, not a re-derivation."""
    row = np.full((1, D), elem, dtype=np.uint16)
    return int(np.asarray(compress_values(row, 8).scales).reshape(-1)[0])


def elem_for(g: int, i: int) -> int:
    return (ELEM_BASE[g % len(ELEM_BASE)]
            + ELEM_STEP[g % len(ELEM_STEP)] * (i + 1)) & 0x7BFF


def build(D: int, n_eng: int) -> list[str]:
    out = [f"CFGD {D}", f"NENG {n_eng}"]
    n_pairs = 0

    for i, rec in enumerate(RECS):
        scales, css, qss = [], [], []
        for g in range(n_eng):
            elem = elem_for(g, i)
            s = scale_of(elem, D)
            assert 0x0400 <= s <= 0x7BFF, (
                f"g{g} rec{rec}: scale {s:04x} left the positive-normal domain")
            cs = score_composite(SQ_G[g], s, D)
            qs = p_requant_composite(s)
            scales.append(s)
            css.append(cs)
            qss.append(qs)
            out.append(f"A {g} {rec} {SQ_G[g]:04x} {elem:04x} {cs:08x} {qs:08x}")
        # coverage self-check: a collapsed bank must be VISIBLE at this record
        for a in range(n_eng):
            for b in range(a + 1, n_eng):
                assert scales[a] != scales[b], (
                    f"rec {rec}: groups {a},{b} share cached scale "
                    f"{scales[a]:04x} — the grid would not discriminate")
                assert css[a] != css[b], f"rec {rec}: groups {a},{b} share cs"
                assert qss[a] != qss[b], f"rec {rec}: groups {a},{b} share qs"
                n_pairs += 1

    # stale probe: written in group 0, requested from group 1
    s_elem = 0x4A11
    s_sc = scale_of(s_elem, D)
    out.append(f"S 0 1 {STALE_REC} {SQ_G[0]:04x} {s_elem:04x} "
               f"{score_composite(SQ_G[0], s_sc, D):08x} "
               f"{p_requant_composite(s_sc):08x}")

    # flush probe (broadcast): group 2, partial <big> then full <small>
    f_sc = scale_of(FLUSH_SMALL, D)
    assert scale_of(FLUSH_BIG, D) != f_sc, "flush probe elements must differ"
    out.append(f"F 2 {FLUSH_REC} {SQ_G[2]:04x} {FLUSH_BIG:04x} "
               f"{FLUSH_SMALL:04x} "
               f"{score_composite(SQ_G[2], f_sc, D):08x} "
               f"{p_requant_composite(f_sc):08x}")

    print(f"grid: {len(RECS)} records x {n_eng} groups, "
          f"{n_pairs} pairwise-distinct checks held")
    return out


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
    n_eng = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    (outdir / "vec").mkdir(parents=True, exist_ok=True)
    for D in (64, 128):
        lines = build(D, n_eng)
        path = outdir / "vec" / f"wcomp_d{D}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path} ({len(lines)} lines)")
    print("WCOMP VECTORS OK (arbiter: walker_composite_golden + cq_codec)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
