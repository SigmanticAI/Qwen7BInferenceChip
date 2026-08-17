#!/usr/bin/env python3
"""gen_walkfmt2_desc.py — W-G2 first rung: the fmt=1 image for the FIRST
WALKABLE single-engine fmt=1 tile-level case (+walkfmt2 probe).

A NEW SIBLING of gen_walker_desc.py (the B1 §7 rule: spec generators are
never edited). It compiles ONE walkable CQ-8 L3 case's already-golden-gated
v1 descriptor into the D-029 fmt=1 wire format, using the walker suite's own
layout mirror (verif/seq_walker/seq_walker_fmt.py — the single source the SV
pkg is equivalence-gated against, CHECKEQ 12,020 rows), and asserts at
generation time that the compiled image

  (a) passes the mirror's walk_desc2_check (fmt=1 legality — so the tile
      MUST walk it, not refuse it), and
  (b) is per-head-EQUIVALENT to the case's v1 descriptor: walker2's
      S2_HLOAD synthesis {8'b0,4'b0,tier=CQ8,t_rows,CFG_D} / RQ[h] /
      {en_pv,en_score} reproduces the v1 GEOM/RQ/MASK words BIT-EXACTLY
      for the l64-class single-engine geometry (n_heads=1, n_kv_heads=1,
      head_dim=CFG_D=64, d_model=64) — "a descriptor extension, not a
      rewrite" (B1_WALKER.md §1) enforced as an equality, not a slogan.

The tile-level checks are then the case's OWN full L3 scoreboard (bit-exact
vs golden results) — the walked fmt=1 stream must reproduce the v1 walk's
tile outputs exactly, which is the W-G2 subset claim for the attention
steps. The attention-subset mask is en_score|en_pv only: QKV/ROPE/STOREKV/
OPROJ/FFN/RES stay host-side exactly as in the D-028 choreography (the L3
store/inject phases), which is what today's tile admits — per-head q
staging mid-walk is IB-LAYER S5 segment-1 (IB_LAYER.md §3c).

Emits build/walker/<case>.desc2:
    DW2 <idx> <word>   the six meaningful image words (0,1,2,3,21,22) —
                       word 21 (STEP) and word 22 (RQ[0]) are DEEP SRAM
                       words (>= 3): the walk reading them is the first
                       DIRECT deep-word read (the stage-4 note's named
                       residual; until now only the wrapped write pointer
                       was proven, via the refusal landing at word 0)
    TROWS <n>          informational, for the run log
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "verif" / "seq_walker"))

import seq_walker_fmt as fmt  # noqa: E402

CASE = "rand_d64_T8"          # the walkfmt probe's case: walkable CQ-8, D=64
CFG_D = 64


def main(build_dir: str = "build") -> int:
    b = Path(build_dir)
    desc = b / "walker" / f"{CASE}.desc"
    if not desc.exists():
        print(f"missing {desc} — run `make walkdesc` first")
        return 2

    kv = {}
    for line in desc.read_text().splitlines():
        f = line.split()
        if len(f) == 2:
            kv[f[0]] = f[1]
    walkable = int(kv["WALKABLE"])
    geom = int(kv["GEOM"], 16)
    rq = int(kv["RQ"], 16)
    mask = int(kv["MASK"], 16)
    trows = int(kv["TROWS"])

    assert walkable == 1, f"{CASE}: probe needs a WALKABLE (CQ-8) case"
    tier = (geom >> 16) & 0x3
    d_dim = geom & 0xFF
    assert tier == 0 and d_dim == CFG_D and (geom >> 8 & 0xFF) == trows, \
        f"{CASE}: v1 GEOM fields unexpected: {geom:08x}"
    assert mask == 0x3, f"{CASE}: probe needs score+pv both enabled"

    # ── the fmt=1 image: l64-class single-engine geometry ────────────────────
    # n_heads=1, n_kv_heads=1 -> d_model = head_dim = 64; the N_ENG=1 fence
    # admits it and today's single live-tier KVQ engine IS its engine 0.
    # kv_map=0 (flat base_g = g*2T) is the tile-truthful record addressing.
    # pos_m = T-1 (the S8 self-inclusive q position; unused by the attention
    # subset - ROPE is mask-disabled - but kept canonical).
    words = {
        fmt.W_GEOM0:  fmt.pack_geom0(CFG_D, tier=tier),
        fmt.W_MODEL0: fmt.pack_model0(CFG_D, 0),      # d_ffn=0: FFN disabled
        fmt.W_MODEL1: fmt.pack_model1(1, 1, kv_map=0),
        fmt.W_MASK:   fmt.pack_mask((1 << fmt.EN_SCORE) | (1 << fmt.EN_PV)),
        fmt.W_STEP:   fmt.pack_step(trows, trows - 1),
        fmt.W_RQ0:    rq,                             # RQ[0] = head-0 PV pair
    }

    # (a) the image must be LEGAL fmt=1 (the tile must WALK it)
    err = fmt.check2(words[fmt.W_GEOM0], words[fmt.W_MODEL0],
                     words[fmt.W_MODEL1], words[fmt.W_MASK],
                     words[fmt.W_STEP], cfg_d=CFG_D)
    assert err == fmt.ERR_NONE, f"{CASE}: fmt=1 image illegal (err={err})"

    # (b) per-head equivalence: walker2's synthesized v1 words == the case's
    # own v1 descriptor, bit-exact (S2_HLOAD formulas, seq_layer_walker2.sv)
    h_geom = (trows << 8) | CFG_D            # {resv0, tier=CQ8, t_rows, D}
    h_rq = words[fmt.W_RQ0]
    h_mask = 0x3                             # {en_pv, en_score}
    assert h_geom == geom, f"synth GEOM {h_geom:08x} != v1 {geom:08x}"
    assert h_rq == rq, f"synth RQ {h_rq:08x} != v1 {rq:08x}"
    assert h_mask == mask, f"synth MASK {h_mask:08x} != v1 {mask:08x}"

    out = b / "walker" / f"{CASE}.desc2"
    with out.open("w") as fh:
        for idx in sorted(words):
            fh.write(f"DW2 {idx} {words[idx]:08x}\n")
        fh.write(f"TROWS {trows}\n")
    print(f"walkfmt2 descriptor: {CASE} -> {out.name} "
          f"(6 words; deep words 21/22; fmt=1 legal; v1-equivalent per head)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
