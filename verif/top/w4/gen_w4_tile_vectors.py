#!/usr/bin/env python3
"""gen_w4_tile_vectors.py — tile-level W4 INGEST LANE case for tb_apex_l3
(built with -GW4EN=1; docs/design/W4_DATAPATH.md).

One case file, nine sub-jobs, every MXE result beat checked BIT-EXACT
against the golden composition golden/apex_golden/w4_lane.py (which itself
defers every element to the landed arbiters: wfeed_w4b_to_i8, gemm_i8,
requant_i32_to_i8, calib_requant):

  e0  illegal GO (k=0)      -> W4_STAT err sticky + W1C, zero side effects
  j1  WS K=64  N=8 raw      -> the basic W4B GEMM
  j2  WS K=128 N=8 raw      -> two-bank act row, 4 scale groups/col;
                               + a GO-while-busy refuse mid-job (sticky+W1C)
  j3  WS K=41  N=8 raw      -> G-tail (ngk=2, ragged) AND lane-tail (41%8=1)
  j4  WS K=8   N=5 raw      -> ODD job_beats=5: packed odd-tail padding drop
  j5  WS K=64  N=8 rq       -> the C-2 epilogue on W4B weights
  j6  OS K=96 = 64+32 acc   -> the D-021 K-SPLIT STRIPE: two lane jobs, ONE
                               shared s8, partial raw drain + final requant
  j7  passthrough K=64 N=8  -> mode_pass INT8 framing smoke
  j8  INT8 baseline (lane_en=0) with j1's EXPECTED W8 image -> the legacy
      path produces j1's exact result (lane == INT8-image equivalence) and
      proves the identity mux with the lane elaborated but off

The W4_CNTI/W4_CNTO counters are checked against the wire-beat accounting
(the doubled-effective-ingest evidence): the packed phase carries 2 INT8
beats per xw beat; scales add G=32's 0.5 b/w so the whole-job wire ratio
is 8/4.5. The gen prints the table; the TB checks the counters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "l3"))
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "golden"))

from gen_l3_vectors import (CSR, OP_GEMM_OS, OP_GEMM_WS,  # noqa: E402
                            Script, desc_words)
from apex_golden.attention import (calib_requant, quant_rows_i8,  # noqa: E402
                                   rmsnorm_fx)
from apex_golden import w4_lane as wl  # noqa: E402

D = 64
BPR = D // 8
G = 32

# W4 CSR window (apex_top.sv W4_*_ADDR; the tile lane's contract)
W4_CTRL = 0x9C
W4_JOB = 0xA0
W4_STAT = 0xA4
W4_CNTI = 0xA8
W4_CNTO = 0xAC

CTRL_EN = 0x001          # lane_en
CTRL_PASS = 0x002        # mode_pass
CTRL_GO = 0x100
CTRL_ERRC = 0x200        # W1C err_sticky
CTRL_DONEC = 0x400       # W1C done_seen

rng = np.random.default_rng(0xB4B4)

# cumulative counter model (mirrors apex_w4_ingest's reset-only counters)
cnt = {"pw": 0, "gs": 0, "out": 0}


def stage_row(s, x8, gamma, bank):
    """One dense feeder row through the REAL RMSNorm+feeder into a stage
    bank (loader_phase's exact choreography and golden arithmetic)."""
    y, _, _ = rmsnorm_fx(x8, gamma)
    h8, s_h = quant_rows_i8(np.array([y], dtype=np.float64) / 256.0)
    s.aj(0, bank, 0, 1, BPR, 0)
    s.fjob(1)
    s.xrow(x8)
    s.grow(gamma)
    s.efs(int(s_h[0]), 1)
    return h8[0].astype(np.int64)


def int8_words(W, K, N):
    """S1/S2 framing of an INT8 [K,N] tile: beat (p,c) lane r = W[8p+r][c]."""
    KB = (K + 7) // 8
    words = []
    for p in range(KB):
        for c in range(N):
            w = 0
            for r in range(8):
                k = 8 * p + r
                v = int(W[k][c]) & 0xFF if k < K else 0
                w |= v << (8 * r)
            words.append(w)
    return words


def w4_kick(s, k, n, s8):
    """Program + kick one lane job (lane already idle by the caller's poll)."""
    s.csrw(W4_JOB, (int(s8) << 16) | (k << 4) | n)
    s.csrw(W4_CTRL, CTRL_GO | CTRL_EN)


def lane_job(s, blob, s8, *, desc, emits, ero_rows, busy_probe=False):
    """One MXE job with weights through the W4 lane (W4B mode).

    emits: [(bank, nb, sel)] act EMITs (first EMIT, then desc, then the
    tail EMITs — the stage_c8_emit ordering); wire beats AFTER the desc so
    the interpreter can never stall the descriptor behind backpressure."""
    K, N = blob.K, blob.N
    s.csrp(W4_STAT, 0x1, 0x0)                     # lane idle
    s.csrw(W4_JOB, (int(s8) << 16) | (K << 4) | N)
    s.csrw(W4_CTRL, CTRL_GO | CTRL_EN)
    if busy_probe:                                # GO while busy: refused loud
        s.csrw(W4_CTRL, CTRL_GO | CTRL_EN)
        s.csrr(W4_STAT, 0x2, 0x2)
        s.csrw(W4_CTRL, CTRL_ERRC | CTRL_EN)
        s.csrr(W4_STAT, 0x2, 0x0)
    bank0, nb0, sel0 = emits[0]
    s.aj(1, bank0, 0, 1, nb0, sel0)
    s.desc(desc)
    for bank, nb, sel in emits[1:]:
        s.aj(1, bank, 0, 1, nb, sel)
    s.wbeats([int(w) for w in wl.wire_words(blob)])
    for lanes, last in ero_rows:
        s.ero(lanes, last)
    gsb, pwb, base = wl.wire_beats_per_job(K, N, G)
    cnt["gs"] += gsb
    cnt["pw"] += pwb
    cnt["out"] += base


def ero_raw(acc_row, n):
    return [int(v) for v in acc_row] + [0] * (8 - n)


def main(build_dir: str) -> None:
    out = Path(build_dir) / "cases"
    out.mkdir(parents=True, exist_ok=True)

    s = Script(D)
    s.emit("// W4 INGEST LANE tile case (D-031 G=32/(B); W4_DATAPATH.md)")

    # ── phase A-lite: CSR sanity + W4 window presence/reset state ────────
    s.csrr(CSR["INFO_VER"], 0xFFFFFFFF, 0x00010000)
    s.csrw(CSR["CTRL"], 0x1)
    s.csrr(W4_STAT, 0x80000000, 0x80000000)       # lane present (W4EN build)
    s.csrr(W4_STAT, 0x00000007, 0x00000000)       # !busy !err !done_seen
    s.csrr(W4_CNTI, 0xFFFFFFFF, 0x00000000)
    s.csrr(W4_CNTO, 0xFFFFFFFF, 0x00000000)

    # e0: illegal GO (k=0) — pulse+sticky, zero side effects, W1C
    s.emit("// e0: illegal GO (k=0) refused loudly")
    s.csrw(W4_CTRL, CTRL_EN)                      # lane on (route level)
    s.csrw(W4_JOB, (0x3C00 << 16) | (0 << 4) | 8)
    s.csrw(W4_CTRL, CTRL_GO | CTRL_EN)
    s.csrr(W4_STAT, 0x3, 0x2)                     # err sticky, not busy
    s.csrw(W4_CTRL, CTRL_ERRC | CTRL_EN)
    s.csrr(W4_STAT, 0x3, 0x0)

    # ── act staging: two dense rows -> bank0/bank1 row 0 ─────────────────
    s.emit("// act staging: two dense feeder rows")
    s.route(fsrc=0, fdst=0, asrc=0, wsrc=0, rdst=0)
    x1 = [int(v) for v in rng.integers(-90, 91, D)]
    x2 = [int(v) for v in rng.integers(-90, 91, D)]
    gam = [int(v) for v in rng.integers(-3000, 3001, D)]
    h1 = stage_row(s, x1, gam, 0)
    h2 = stage_row(s, x2, gam, 1)
    A128 = np.concatenate([h1, h2])

    def prep(K, N, scale=0.05, k_split=2048):
        W = rng.normal(0, scale, (K, N))
        return wl.prep_w4_direct(W, G=G, k_split=k_split)

    # j1: WS K=64 N=8 raw
    s.emit("// j1: WS K=64 N=8 raw")
    p1 = prep(64, 8)
    acc1 = wl.gemm_w4b(A128[None, :64], p1.blob, p1.s8)
    lane_job(s, p1.blob, p1.s8,
             desc=desc_words(OP_GEMM_WS, 1, 64, 8),
             emits=[(0, BPR, 0)],
             ero_rows=[(ero_raw(acc1[0], 8), 1)])
    s.csrp(W4_STAT, 0x1, 0x0)
    s.csrr(W4_STAT, 0x4, 0x4)                     # done_seen
    s.csrw(W4_CTRL, CTRL_DONEC | CTRL_EN)
    s.csrr(W4_STAT, 0x4, 0x0)
    s.csrr(W4_JOB, 0xFFFFFFFF,
           (int(p1.s8) << 16) | (64 << 4) | 8)    # job readback

    # j2: WS K=128 N=8 raw, two-bank act + GO-while-busy refuse
    s.emit("// j2: WS K=128 N=8 raw + busy GO refuse")
    p2 = prep(128, 8)
    acc2 = wl.gemm_w4b(A128[None, :], p2.blob, p2.s8)
    lane_job(s, p2.blob, p2.s8,
             desc=desc_words(OP_GEMM_WS, 1, 128, 8),
             emits=[(0, BPR, 0), (1, BPR, 0)],
             ero_rows=[(ero_raw(acc2[0], 8), 1)],
             busy_probe=True)

    # j3: WS K=41 N=8 raw (G-tail + lane-tail)
    s.emit("// j3: WS K=41 N=8 raw (ragged tails)")
    p3 = prep(41, 8)
    acc3 = wl.gemm_w4b(A128[None, :41], p3.blob, p3.s8)
    lane_job(s, p3.blob, p3.s8,
             desc=desc_words(OP_GEMM_WS, 1, 41, 8),
             emits=[(0, 6, 0)],
             ero_rows=[(ero_raw(acc3[0], 8), 1)])

    # j4: WS K=8 N=5 raw (odd job_beats -> packed odd-tail padding)
    s.emit("// j4: WS K=8 N=5 raw (odd beats)")
    p4 = prep(8, 5)
    acc4 = wl.gemm_w4b(A128[None, :8], p4.blob, p4.s8)
    lane_job(s, p4.blob, p4.s8,
             desc=desc_words(OP_GEMM_WS, 1, 8, 5),
             emits=[(0, 1, 0)],
             ero_rows=[(ero_raw(acc4[0], 5), 1)])

    # j5: WS K=64 N=8 with the C-2 requant epilogue
    s.emit("// j5: WS K=64 N=8 requant")
    p5 = prep(64, 8, scale=0.08)
    acc5 = wl.gemm_w4b(A128[None, :64], p5.blob, p5.s8)
    rq5 = calib_requant(int(np.max(np.abs(acc5))))
    o5 = wl.gemm_w4b(A128[None, :64], p5.blob, p5.s8, rq=rq5)
    lane_job(s, p5.blob, p5.s8,
             desc=desc_words(OP_GEMM_WS, 1, 64, 8, rq_en=1,
                             rq_scale=int(rq5[0]), rq_shift=int(rq5[1])),
             emits=[(0, BPR, 0)],
             ero_rows=[(ero_raw(o5[0], 8), 1)])

    # j6: the K-SPLIT STRIPE — K=96 as 64+32 OS-accumulate, ONE shared s8
    s.emit("// j6: OS K=96 stripe (64+32), shared s8, partial + final")
    p6 = prep(96, 8, k_split=64)
    assert p6.seg_k == [64, 32]
    A96 = np.concatenate([h1, h2[:32]])
    acc6a = wl.gemm_w4b(A96[None, :64], p6.segs[0], p6.s8)
    acc6 = wl.gemm_w4b(A96[None, :], p6.blob, p6.s8)
    rq6 = calib_requant(int(np.max(np.abs(acc6))))
    o6 = wl.gemm_w4b(A96[None, :], p6.blob, p6.s8, rq=rq6)
    lane_job(s, p6.segs[0], p6.s8,
             desc=desc_words(OP_GEMM_OS, 1, 64, 8, mode_os=1),
             emits=[(0, BPR, 0)],
             ero_rows=[(ero_raw(acc6a[0], 8), 1)])
    s.csrp(W4_STAT, 0x1, 0x0)
    lane_job(s, p6.segs[1], p6.s8,
             desc=desc_words(OP_GEMM_OS, 1, 32, 8, mode_os=1, acc=1,
                             rq_en=1, rq_scale=int(rq6[0]),
                             rq_shift=int(rq6[1])),
             emits=[(1, 4, 0)],
             ero_rows=[(ero_raw(o6[0], 8), 1)])

    # j7: passthrough framing smoke (mode_pass, INT8 image through the lane)
    s.emit("// j7: passthrough K=64 N=8")
    W7 = rng.integers(-128, 128, (64, 8), dtype=np.int64)
    acc7 = np.asarray(A128[None, :64] @ W7)
    s.csrp(W4_STAT, 0x1, 0x0)
    s.csrw(W4_JOB, (0 << 16) | (64 << 4) | 8)     # s8 ignored in passthrough
    s.csrw(W4_CTRL, CTRL_GO | CTRL_EN | CTRL_PASS)
    s.aj(1, 0, 0, 1, BPR, 0)
    s.desc(desc_words(OP_GEMM_WS, 1, 64, 8))
    s.wbeats(int8_words(W7, 64, 8))
    s.ero(ero_raw(acc7[0], 8), 1)
    cnt["pw"] += 64
    cnt["out"] += 64

    # j8: INT8 BASELINE with j1's expected W8 image, lane OFF — the legacy
    # path must produce j1's exact result (identity mux + lane==image proof)
    s.emit("// j8: INT8 baseline, lane off, j1's W8 image")
    s.csrp(W4_STAT, 0x1, 0x0)
    s.csrw(W4_CTRL, 0x0)                          # lane_en=0 (idle: legal)
    W8_1 = wl.expected_w8(p1.blob, p1.s8)
    s.aj(1, 0, 0, 1, BPR, 0)
    s.desc(desc_words(OP_GEMM_WS, 1, 64, 8))
    s.wbeats(int8_words(W8_1, 64, 8))
    s.ero(ero_raw(acc1[0], 8), 1)                 # same lanes as j1

    # ── final: counters (the ingest proof), quiescence, stickies ─────────
    s.emit("// final: counter proof + quiescence")
    s.csrp(CSR["STATUS"], 0x1, 0x1)
    s.csrr(W4_CNTI, 0xFFFFFFFF, (cnt["gs"] << 16) | cnt["pw"])
    s.csrr(W4_CNTO, 0xFFFFFFFF, cnt["out"])
    s.csrr(W4_STAT, 0x3, 0x0)                     # no residual err, idle
    s.estk(0xFFFF, 0x0000)
    s.emit("DONE")

    p = out / "w4_gemm_b64.txt"
    p.write_text("\n".join(s.lines) + "\n")

    w4b_base = cnt["out"] - 64 - 64               # exclude j7 pass + j8
    w4b_wire = (cnt["pw"] - 64) + cnt["gs"]
    print(f"case w4_gemm_b64: checks={s.n_expect}")
    print(f"INGEST ACCOUNTING (checked in-RTL via W4_CNTI/W4_CNTO):")
    print(f"  W4B jobs: int8-equivalent beats={w4b_base} "
          f"wire beats={w4b_wire} (packed {cnt['pw'] - 64} + "
          f"scales {cnt['gs']})")
    print(f"  packed-phase density: {w4b_base}/{cnt['pw'] - 64} = "
          f"{w4b_base / (cnt['pw'] - 64):.3f}x per beat")
    print(f"  whole-job wire ratio: {w4b_base}/{w4b_wire} = "
          f"{w4b_base / w4b_wire:.3f}x (4-bit + fp16/G={G} ~= 4.5 b/w)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
