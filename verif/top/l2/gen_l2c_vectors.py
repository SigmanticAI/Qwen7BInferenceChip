#!/usr/bin/env python3
"""gen_l2c_vectors.py — golden-driven scripts for tb_l2c_chain (Layer-2
chain c: seam_score_dequant -> apex_score_fork -> {asu_softmax, tip_top}).

Golden arbiters:
  scores : RNE(acc * f32comp) mirror (seam_score_dequant contract)
  probs  : apex_golden.compute.online_softmax_fx(scores, 10)
  TIP    : tip_decide_golden / imp_update_golden / tier_golden — the exact
           fixed-width mirrors from verif/tip/smoke/gen_tip_vectors.py,
           replicated at the apex_top operating point (W=32, N=BLOCK_M*
           BLOCK_N=8, T_MAX=31, ACC_W=16)

Emits directed / random / reset scripts.  The directed set includes the
committed-apex_top L3 reality check: a tile longer than 8 scores makes TIP
abort with the frame sticky (no decision) while the SAME forked stream's
ASU row stays bit-exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden.compute import online_softmax_fx  # noqa: E402
from apex_golden.fp import rne  # noqa: E402

SCORE_FRAC = 10
SEED = 0x12C0_0001
W, T_W, N_TIP, ACC_W = 32, 5, 8, 16
IMIN = -(1 << 31)
CQ8, CQ4, CQ4P = 0, 1, 2
ONE_F32 = 0x3F800000


# ── TIP golden mirrors (verif/tip/smoke/gen_tip_vectors.py, N=8) ────────────

def abs2c(v):
    mask = (1 << W) - 1
    v &= mask
    return ((~v) + 1) & mask if v & (1 << (W - 1)) else v


def clamp_thr(t):
    t &= (1 << T_W) - 1
    return 1 if t == 0 else t


def tip_decide_golden(tile, thr):
    log2n = 3
    sum_w = W + log2n
    cmp_w = sum_w + T_W
    sum_mask = (1 << sum_w) - 1
    cmp_mask = (1 << cmp_w) - 1
    t = clamp_thr(thr)
    mx, sm = 0, 0
    for v in tile:
        a = abs2c(int(v))
        mx = max(mx, a)
        sm = (sm + a) & sum_mask
    lhs = (mx << log2n) & cmp_mask
    rhs = 0
    for i in range(T_W):
        if (t >> i) & 1:
            rhs = (rhs + ((sm << i) & cmp_mask)) & cmp_mask
    return (1 if lhs > rhs else 0), mx


def bucket(m):
    return 0 if m == 0 else m.bit_length()


def imp_update(acc, fp16, mx):
    return min(acc + bucket(mx) * (2 if fp16 else 1), (1 << ACC_W) - 1)


def tier_of(acc, hi, lo):
    return CQ8 if acc >= hi else (CQ4P if acc >= lo else CQ4)


def score_mirror(acc_flat, comp_bits):
    """Golden S-3 arithmetic + the RTL saturation contract. Returns
    (scores, saturated_flag) — saturation raises the sd_range sticky."""
    comp = np.array(comp_bits, dtype=np.uint32).view(np.float32).astype(np.float64)
    out = []
    sat = False
    for a, c in zip(acc_flat, comp):
        v = rne(np.array([float(a) * float(c)]))[0]
        if v >= 2**31:
            out.append(np.int64(2**31 - 1))
            sat = True
        elif v <= -(2**31):
            out.append(np.int64(-(2**31 - 1)))
            sat = True
        else:
            out.append(np.int64(v))
    return np.array(out, dtype=np.int64), sat


class Script:
    def __init__(self):
        self.lines = []
        self.n_expect = 0

    def emit(self, *toks):
        self.lines.append(" ".join(str(t) for t in toks))

    def epr(self, probs):
        self.emit("EPR", f"{len(probs):x}")
        for i, v in enumerate(probs):
            self.emit(f"{int(v) & 0xFFFF:04x}", int(i == len(probs) - 1))
        self.n_expect += len(probs)

    def etip(self, blk, tier, fp16):
        self.emit("ETIP", f"{blk:02x}", f"{tier:x}", f"{fp16:x}")
        self.n_expect += 1

    def stk(self, mask, exp):
        self.emit("STK", f"{mask:02x}", f"{exp:02x}")
        self.n_expect += 1

    def imprd(self, addr, data, tier):
        self.emit("IMPRD", f"{addr:02x}", f"{data:08x}", f"{tier:x}")
        self.n_expect += 2


class TipState:
    def __init__(self):
        self.imp = {}
        self.thr = 1
        self.hi = 0xFFFF
        self.lo = 0x0000
        self.stk = 0            # {tip_frame, asu_row, sd_frame, sd_range, sd_job}


def emit_tile(s, st, acc_lanes, blk, comp_bits=None, frame_break=False):
    """One dequant job = one forked tile. cols = len(acc_lanes).
    frame_break: cols > 8 -> TIP aborts (sticky), ASU row still checked."""
    cols = len(acc_lanes)
    comp = comp_bits if comp_bits is not None else [ONE_F32] * cols
    scores, sat = score_mirror(acc_lanes, comp)
    if sat:
        st.stk |= 0x02          # sd_range sticky (clears only on reset)
    if frame_break:
        st.stk |= 0x10          # tip_frame sticky
    probs = online_softmax_fx(scores, SCORE_FRAC)
    s.emit(f"// tile blk={blk} cols={cols} frame_break={int(frame_break)}")
    s.epr(probs)
    s.emit("DJ", f"{cols:x}")
    for c in comp:
        s.emit("CMP", f"{c:08x}")
    nb = (cols + 7) // 8
    for b in range(nb):
        lanes = [0] * 8
        for j in range(8):
            if 8 * b + j < cols:
                lanes[j] = int(acc_lanes[8 * b + j])
        s.emit("ACB", *[f"{v & 0xFFFFFFFF:08x}" for v in lanes],
               int(b == nb - 1))
    if not frame_break:
        # ETIP is a BLOCKING expectation: it must follow the tile stimulus
        fp16, mx = tip_decide_golden(scores.tolist(), st.thr)
        st.imp[blk] = imp_update(st.imp.get(blk, 0), fp16, mx)
        s.etip(blk, tier_of(st.imp[blk], st.hi, st.lo), fp16)
    s.emit("IDLE")


def main(build_dir: str) -> None:
    out = Path(build_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cov = {"tiles": 0, "fp16": 0, "int8": 0, "frame": 0, "intmin": 0,
           "zero_tile": 0, "single": 0, "thr_extreme": 0, "imprd": 0,
           "blk_hi": 0, "reset": 0}

    # ── directed ─────────────────────────────────────────────────────────────
    s = Script()
    st = TipState()
    s.emit("// l2c directed")
    s.stk(0x1F, st.stk)

    s.emit("BLK", "00")
    # max-dominant tile -> FP16 decision
    emit_tile(s, st, [1 << 20, 3, -2, 1, 0, 2, -1, 4], 0)
    cov["tiles"] += 1
    cov["fp16"] += 1
    # flat tile -> INT8 decision
    emit_tile(s, st, [100, 101, -102, 100, 99, -100, 101, 100], 0)
    cov["tiles"] += 1
    cov["int8"] += 1
    # INT_MIN acc: dequant SATURATES to -(2^31-1) + sd_range sticky — the
    # real chain can never hand TIP an INT_MIN score (documented finding)
    emit_tile(s, st, [IMIN, 5, -5, 7, 1, 2, 3, 4], 0)
    cov["tiles"] += 1
    cov["intmin"] += 1
    # all-zero tile (bucket(0) = 0, sum = 0, tie -> INT8)
    emit_tile(s, st, [0] * 8, 0)
    cov["tiles"] += 1
    cov["zero_tile"] += 1
    # single-score tile on a different block
    s.emit("IDLE")
    s.emit("BLK", "05")
    emit_tile(s, st, [12345], 5)
    cov["tiles"] += 1
    cov["single"] += 1
    # importance window readout for both touched blocks
    s.imprd(0, st.imp.get(0, 0), tier_of(st.imp.get(0, 0), st.hi, st.lo))
    s.imprd(5, st.imp.get(5, 0), tier_of(st.imp.get(5, 0), st.hi, st.lo))
    cov["imprd"] += 2

    # threshold extremes (quasi-static: only changed at IDLE points)
    s.emit("THR", "1f")
    st.thr = 31
    emit_tile(s, st, [1 << 24, 1, 1, 1, 1, 1, 1, 1], 5)
    cov["tiles"] += 1
    cov["thr_extreme"] += 1
    s.emit("THR", "0")                      # all-zeros clamps to 1 (D-017)
    st.thr = 0
    emit_tile(s, st, [1 << 24, 1, 1, 1, 1, 1, 1, 1], 5)
    cov["tiles"] += 1
    cov["thr_extreme"] += 1
    s.emit("THR", "4")
    st.thr = 4

    # importance thresholds: force CQ8 / CQ4P / CQ4 suggestions
    s.emit("BLK", "09")
    s.emit("IHI", "0001")
    s.emit("ILO", "0000")
    st.hi, st.lo = 1, 0
    emit_tile(s, st, [9, 8, 7, 6, 5, 4, 3, 2], 9)      # acc>=1 -> CQ8
    cov["tiles"] += 1
    s.emit("BLK", "0a")
    s.emit("IHI", "ffff")
    s.emit("ILO", "0001")
    st.hi, st.lo = 0xFFFF, 1
    emit_tile(s, st, [9, 8, 7, 6, 5, 4, 3, 2], 10)     # lo<=acc<hi -> CQ4P
    cov["tiles"] += 1
    s.emit("BLK", "0b")
    s.emit("IHI", "ffff")
    s.emit("ILO", "ffff")
    st.hi, st.lo = 0xFFFF, 0xFFFF
    emit_tile(s, st, [0, 0, 0, 0, 0, 0, 0, 0], 11)     # acc<lo -> CQ4
    cov["tiles"] += 1

    # frame-break tile: 9..16 scores -> TIP aborts + sticky, ASU bit-exact
    s.emit("// frame guard: 12-score tile, TIP aborts, ASU row unaffected")
    acc12 = [int(v) for v in rng.integers(-(1 << 16), 1 << 16, 12)]
    emit_tile(s, st, acc12, 11, frame_break=True)
    cov["tiles"] += 1
    cov["frame"] += 1
    s.stk(0x1F, st.stk)
    # recovery: clean tile after the aborted one
    emit_tile(s, st, [55, -3, 8, 0, 2, 1, 9, -7], 11)
    cov["tiles"] += 1
    s.stk(0x1F, st.stk)

    # high block id (rt_tip_blk width sweep)
    s.emit("IDLE")
    s.emit("BLK", "7f")
    emit_tile(s, st, [77, -6, 2, 4], 127)
    cov["tiles"] += 1
    cov["blk_hi"] += 1
    s.imprd(127, st.imp.get(127, 0),
            tier_of(st.imp.get(127, 0), st.hi, st.lo))
    cov["imprd"] += 1

    s.emit("ENDTAB")
    s.n_expect += 2
    s.emit("DONE")
    (out / "vectors_l2c_directed.txt").write_text("\n".join(s.lines) + "\n")
    n_directed = s.n_expect

    # ── random ───────────────────────────────────────────────────────────────
    s = Script()
    st = TipState()
    s.emit("// l2c random")
    s.emit("THR", "3")
    st.thr = 3
    s.emit("IHI", "0040")
    s.emit("ILO", "0010")
    st.hi, st.lo = 0x40, 0x10
    blk = 0
    for i in range(64):
        if i % 8 == 0:
            blk = int(rng.integers(0, 128))
            s.emit("IDLE")
            s.emit("BLK", f"{blk:02x}")
        cols = int(rng.integers(1, 9))
        mag = int(rng.integers(4, 30))
        acc = [int(v) for v in rng.integers(-(1 << mag), 1 << mag, cols)]
        comp = None
        if rng.random() < 0.3:
            c = np.float32(2.0 ** float(rng.integers(-8, 4)))
            comp = [int(c.view(np.uint32))] * cols
        emit_tile(s, st, acc, blk, comp_bits=comp)
        cov["tiles"] += 1
    s.stk(0x1F, st.stk)
    s.emit("ENDTAB")
    s.n_expect += 2
    s.emit("DONE")
    (out / "vectors_l2c_random.txt").write_text("\n".join(s.lines) + "\n")
    n_random = s.n_expect

    # ── mid-operation reset ──────────────────────────────────────────────────
    s = Script()
    st = TipState()
    s.emit("// l2c reset: tile mid-flight in dequant+fork+TIP, then rst_n")
    s.emit("BLK", "03")
    s.emit("DJ", "10")                      # 16-col job, only 1 of 2 beats
    for _ in range(16):
        s.emit("CMP", f"{ONE_F32:08x}")
    s.emit("ACB", *[f"{(i + 1):08x}" for i in range(8)], 0)
    s.emit("RST")
    s.n_expect += 2
    s.emit("// post-reset clean set")
    s.stk(0x1F, st.stk)
    for i in range(6):
        cols = [8, 5, 1, 8, 3, 8][i]
        acc = [int(v) for v in rng.integers(-(1 << 20), 1 << 20, cols)]
        emit_tile(s, st, acc, 3)
        cov["tiles"] += 1
    cov["reset"] = 1
    s.stk(0x1F, st.stk)
    s.emit("ENDTAB")
    s.n_expect += 2
    s.emit("DONE")
    (out / "vectors_l2c_reset.txt").write_text("\n".join(s.lines) + "\n")

    print(f"l2c vectors: directed={n_directed} random={n_random} "
          f"reset={s.n_expect} scripted checks")
    for k, v in cov.items():
        print(f"COVGEN l2c {k} {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
