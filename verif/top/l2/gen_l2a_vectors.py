#!/usr/bin/env python3
"""gen_l2a_vectors.py — golden-driven scripts for tb_l2a_chain (Layer-2
chain a: mxe_top -> seam_score_dequant -> asu_softmax).

Golden arbiter chain (all bit-exact mirrors, per module headers):
  acc     = apex_golden.compute.gemm_i8(A, B)      (INT32 accumulators)
  flat    = MXE drain-lane flattening (lanes >= N are structural zeros)
  score   = RNE(f64(acc) * f64(f32(comp))) with the RTL's saturation to
            +/-(2^31 - 1) on overflow (seam_score_dequant range contract;
            in-contract stimulus never saturates except the directed
            range_error case)
  probs   = apex_golden.compute.online_softmax_fx(score, SCORE_FRAC=10)

Emits:
  vectors_l2a_directed.txt  directed integration set: single-job framing,
                            N<8 structural-zero columns, K tails with
                            garbage in masked lanes (proves masking),
                            the L3 chunked-job framing reality (multi
                            M=1 MXE jobs -> one dequant job; documented
                            frame_error sticky + bit-exact numerics),
                            range_error saturation, illegal job/descriptor
                            rejects (§3), sticky accounting
  vectors_l2a_random.txt    seeded random clean jobs (the volume set)
  vectors_l2a_reset.txt     mid-operation reset: job in flight in every
                            block, rst_n dropped, full clean set replayed

Coverage facts about the emitted stimulus are printed as COVGEN lines and
gated by coverage_report.py against the run logs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden.compute import gemm_i8, online_softmax_fx  # noqa: E402
from apex_golden.fp import rne  # noqa: E402

SCORE_FRAC = 10
OP_GEMM_WS = 0x01
OP_GEMM_OS = 0x02
SEED = 0x12A0_0001

SAT_POS = 0x7FFF_FFFF
SAT_NEG = 0x8000_0001  # -(2^31 - 1) two's complement


def desc_words(opcode, m, k, n, *, mode_os=0, acc=0, rq_en=0,
               rq_scale=0, rq_shift=0):
    v = (opcode & 0xFF) | (m << 8) | (k << 20) | (n << 32) \
        | (rq_scale << 44) | (rq_shift << 60) | (rq_en << 65) \
        | (acc << 66) | (mode_os << 67)
    return [(v >> (32 * i)) & 0xFFFFFFFF for i in (3, 2, 1, 0)]


def act_beats(A, K, garbage):
    """INGEST order: beat (m, p) = A[m][8p+7:8p]; lanes >= K on the last
    chunk carry deterministic garbage (must be ignored by the DUT)."""
    M = A.shape[0]
    KB = (K + 7) // 8
    beats = []
    for m in range(M):
        for p in range(KB):
            w = 0
            for r in range(8):
                kk = 8 * p + r
                v = int(A[m][kk]) if kk < K else int(garbage())
                w |= (v & 0xFF) << (8 * r)
            beats.append(w)
    return beats


def wgt_beats(B, K, N, garbage):
    """WLOAD order: per chunk p, N beats c=0..N-1; beat (p,c) lane r =
    B[8p+r][c]; rows >= K carry garbage (zero-masked by the DUT)."""
    KB = (K + 7) // 8
    beats = []
    for p in range(KB):
        for c in range(N):
            w = 0
            for r in range(8):
                kk = 8 * p + r
                v = int(B[kk][c]) if kk < K else int(garbage())
                w |= (v & 0xFF) << (8 * r)
            beats.append(w)
    return beats


def score_mirror(acc_flat, comp_bits):
    """Golden S-3 arithmetic + the RTL saturation contract, elementwise."""
    comp = np.array(comp_bits, dtype=np.uint32).view(np.float32).astype(np.float64)
    out = []
    for a, c in zip(acc_flat, comp):
        v = rne(np.array([float(a) * float(c)]))[0]
        if v >= 2**31:
            out.append(np.int64(np.uint64(SAT_POS)))
        elif v <= -(2**31):
            out.append(np.int64(-(2**31 - 1)))
        else:
            out.append(np.int64(v))
    return np.array(out, dtype=np.int64)


class Script:
    def __init__(self):
        self.lines = []
        self.n_expect = 0

    def emit(self, *toks):
        self.lines.append(" ".join(str(t) for t in toks))

    def desc(self, words):
        self.emit("DESC", *[f"{w:08x}" for w in words])

    def esc(self, scores):
        self.emit("ESC", f"{len(scores):x}")
        for v in scores:
            self.emit(f"{int(v) & 0xFFFFFFFF:08x}")
        self.n_expect += len(scores)

    def epr(self, probs):
        self.emit("EPR", f"{len(probs):x}")
        for i, v in enumerate(probs):
            self.emit(f"{int(v) & 0xFFFF:04x}", int(i == len(probs) - 1))
        self.n_expect += len(probs)

    def stk(self, mask, exp):
        self.emit("STK", f"{mask:02x}", f"{exp:02x}")
        self.n_expect += 1


def gen_comp(rng, K):
    """A positive normal fp32 composite sized so |score| stays interesting
    but within INT32: |acc| <= K*127*127; pick comp ~ 2^t / amax_bound."""
    amax_bound = K * 127 * 127
    target = float(rng.integers(1 << 8, 1 << 14))
    c = np.float32(target / amax_bound * (0.5 + rng.random()))
    b = int(c.view(np.uint32))
    assert 0 < ((b >> 23) & 0xFF) < 255
    return b


def emit_job(s, rng, M, K, N, cols, sticky_exp, chunked=False,
             comp_override=None):
    """One dequant row of `cols` scores. chunked=False: one MXE M=M job.
    chunked=True: cols/8 M=1 MXE jobs (the L3 T>8 pattern; raises the
    documented sd_frame sticky on every non-final mid-stream last)."""
    A = rng.integers(-128, 128, (M, K), dtype=np.int64)
    B = rng.integers(-128, 128, (K, N), dtype=np.int64)
    acc = gemm_i8(A, B).astype(np.int64)          # [M, N]
    garbage = lambda: rng.integers(-128, 128)     # noqa: E731

    # MXE drain flattening: beat m lane j = acc[m][j] (j<N) else 0
    flat = np.zeros(M * 8, dtype=np.int64)
    for m in range(M):
        flat[8 * m:8 * m + N] = acc[m]
    flat = flat[:cols]

    comp = ([comp_override] * cols if comp_override is not None
            else [gen_comp(rng, K) for _ in range(cols)])
    scores = score_mirror(flat, comp)
    probs = online_softmax_fx(scores, SCORE_FRAC)

    s.emit(f"// job M={M} K={K} N={N} cols={cols} chunked={int(chunked)}")
    s.esc(scores)
    s.epr(probs)
    s.emit("DJ", f"{cols:x}")
    for c in comp:
        s.emit("CMP", f"{c:08x}")
    if not chunked:
        s.desc(desc_words(OP_GEMM_WS if rng.random() < 0.5 else OP_GEMM_OS,
                          M, K, N, mode_os=int(rng.random() < 0.5)))
        ab = act_beats(A, K, garbage)
        for i, w in enumerate(ab):
            s.emit("AB", f"{w:016x}", int(i == len(ab) - 1))
        for w in wgt_beats(B, K, N, garbage):
            s.emit("WB", f"{w:016x}")
    else:
        assert N == 8 and cols % 8 == 0 and cols // 8 == M
        for m in range(M):
            s.desc(desc_words(OP_GEMM_OS, 1, K, 8, mode_os=1))
            ab = act_beats(A[m:m + 1], K, garbage)
            for i, w in enumerate(ab):
                s.emit("AB", f"{w:016x}", int(i == len(ab) - 1))
            for w in wgt_beats(B, K, 8, garbage):
                s.emit("WB", f"{w:016x}")
        sticky_exp["sd_frame"] = 1                # documented v0.1 wart
    s.emit("IDLE")
    return scores


def stk_word(st):
    return (st["mxe_desc"] | (st["sd_job"] << 1) | (st["sd_range"] << 2)
            | (st["sd_frame"] << 3) | (st["asu_row"] << 4))


def main(build_dir: str) -> None:
    out = Path(build_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cov = {"m1": 0, "m8": 0, "tail_cols": 0, "n_lt8": 0, "cols64": 0,
           "chunked": 0, "ktail": 0, "sat": 0, "illegal": 0, "jobs": 0,
           "neg_ext": 0, "reset": 0}

    # ── directed set ────────────────────────────────────────────────────────
    s = Script()
    st = {"mxe_desc": 0, "sd_job": 0, "sd_range": 0, "sd_frame": 0,
          "asu_row": 0}
    s.emit("// l2a directed: clean singles first, sticky-raising cases last")
    s.stk(0x1F, 0x00)

    for (M, K, N, cols) in ((1, 8, 8, 8), (1, 64, 8, 8), (8, 64, 8, 64),
                            (4, 17, 8, 29), (2, 8, 5, 16), (8, 96, 8, 61),
                            (1, 1, 1, 1), (3, 33, 8, 24)):
        sc = emit_job(s, rng, M, K, N, cols, st)
        cov["jobs"] += 1
        cov["m1"] += int(M == 1)
        cov["m8"] += int(M == 8)
        cov["tail_cols"] += int(cols % 8 != 0)
        cov["n_lt8"] += int(N < 8)
        cov["cols64"] += int(cols == 64)
        cov["ktail"] += int(K % 8 != 0)
        cov["neg_ext"] += int(np.min(sc) < -(1 << 20))
    # score-extreme job: comp = 2^9 pushes |score| toward 2^29 (no overflow;
    # exercises the ASU running-max/clamp path with huge magnitudes)
    sc = emit_job(s, rng, 8, 64, 8, 64, st, comp_override=0x44000000)
    cov["jobs"] += 1
    cov["neg_ext"] += int(np.min(sc) < -(1 << 20))
    assert np.min(sc) < -(1 << 20), "extreme job failed to produce extremes"
    s.stk(0x1F, stk_word(st))

    # illegal dequant job (cols=0): §3 pulse+sticky, no state change
    s.emit("// illegal DJ cols=0 -> sd_job sticky; chain stays clean")
    s.emit("DJ", "0")
    st["sd_job"] = 1
    cov["illegal"] += 1
    s.stk(0x1F, stk_word(st))
    # illegal MXE descriptor (K=0): desc_error sticky, no side effects
    s.emit("// illegal DESC K=0 -> mxe desc sticky")
    s.desc(desc_words(OP_GEMM_WS, 1, 0, 1))
    st["mxe_desc"] = 1
    cov["illegal"] += 1
    s.stk(0x1F, stk_word(st))
    # a clean job right after the rejects (reject leaves no state)
    emit_job(s, rng, 2, 16, 8, 16, st)
    cov["jobs"] += 1
    s.stk(0x1F, stk_word(st))

    # chunked framing: one dequant job spanning 3 M=1 MXE jobs (L3 T>8)
    s.emit("// chunked: 3x M=1 MXE jobs -> one DJ cols=24 (frame sticky)")
    emit_job(s, rng, 3, 64, 8, 24, st, chunked=True)
    cov["jobs"] += 1
    cov["chunked"] += 1
    s.stk(0x1F, stk_word(st))

    # range_error saturation: giant composite -> |score| >= 2^31
    s.emit("// range_error: comp 2^40 saturates scores; probs still golden")
    M, K, N, cols = 1, 8, 8, 8
    A = rng.integers(100, 128, (M, K), dtype=np.int64)
    B = rng.integers(100, 128, (K, N), dtype=np.int64)
    acc = gemm_i8(A, B).astype(np.int64)
    comp = [0x53800000] * cols                    # 2^40, positive normal
    scores = score_mirror(acc[0][:cols], comp)
    assert int(np.max(scores)) == SAT_POS
    probs = online_softmax_fx(scores, SCORE_FRAC)
    s.esc(scores)
    s.epr(probs)
    s.emit("DJ", f"{cols:x}")
    for c in comp:
        s.emit("CMP", f"{c:08x}")
    s.desc(desc_words(OP_GEMM_WS, M, K, N))
    g = lambda: 0                                 # noqa: E731
    ab = act_beats(A, K, g)
    for i, w in enumerate(ab):
        s.emit("AB", f"{w:016x}", int(i == len(ab) - 1))
    for w in wgt_beats(B, K, N, g):
        s.emit("WB", f"{w:016x}")
    s.emit("IDLE")
    st["sd_range"] = 1
    cov["sat"] += 1
    cov["jobs"] += 1
    s.stk(0x1F, stk_word(st))

    s.emit("ENDTAB")
    s.n_expect += 2
    s.emit("DONE")
    (out / "vectors_l2a_directed.txt").write_text("\n".join(s.lines) + "\n")
    n_directed = s.n_expect

    # ── random volume set (clean jobs only) ─────────────────────────────────
    s = Script()
    st = {"mxe_desc": 0, "sd_job": 0, "sd_range": 0, "sd_frame": 0,
          "asu_row": 0}
    s.emit("// l2a random volume set")
    for _ in range(48):
        M = int(rng.integers(1, 9))
        K = int(rng.integers(1, 97))
        N = int(rng.integers(1, 9))
        lo = 8 * (M - 1) + 1
        cols = int(rng.integers(lo, 8 * M + 1))
        emit_job(s, rng, M, K, N, cols, st)
        cov["jobs"] += 1
        cov["m1"] += int(M == 1)
        cov["m8"] += int(M == 8)
        cov["tail_cols"] += int(cols % 8 != 0)
        cov["n_lt8"] += int(N < 8)
        cov["ktail"] += int(K % 8 != 0)
    s.stk(0x1F, 0x00)
    s.emit("ENDTAB")
    s.n_expect += 2
    s.emit("DONE")
    (out / "vectors_l2a_random.txt").write_text("\n".join(s.lines) + "\n")
    n_random = s.n_expect

    # ── mid-operation reset set ─────────────────────────────────────────────
    s = Script()
    st = {"mxe_desc": 0, "sd_job": 0, "sd_range": 0, "sd_frame": 0,
          "asu_row": 0}
    s.emit("// l2a mid-op reset: sacrificial partial job, reset, clean rerun")
    # a job in flight in every block: dequant mid-LOAD, MXE mid-INGEST
    s.emit("DJ", "10")
    for _ in range(6):                            # partial composite load
        s.emit("CMP", f"{gen_comp(rng, 16):08x}")
    s.desc(desc_words(OP_GEMM_WS, 2, 16, 8))
    s.emit("AB", f"{int(rng.integers(0, 1 << 63)):016x}", 0)
    s.emit("AB", f"{int(rng.integers(0, 1 << 63)):016x}", 0)
    s.emit("RST")
    s.n_expect += 2                               # RST drain checks
    cov["reset"] += 1
    s.emit("// post-reset: stickies clear, full clean set")
    s.stk(0x1F, 0x00)
    for (M, K, N, cols) in ((1, 8, 8, 8), (8, 64, 8, 64), (4, 17, 8, 29)):
        emit_job(s, rng, M, K, N, cols, st)
        cov["jobs"] += 1
    s.stk(0x1F, 0x00)
    s.emit("ENDTAB")
    s.n_expect += 2
    s.emit("DONE")
    (out / "vectors_l2a_reset.txt").write_text("\n".join(s.lines) + "\n")

    print(f"l2a vectors: directed={n_directed} random={n_random} "
          f"reset={s.n_expect} scripted checks")
    for k, v in cov.items():
        print(f"COVGEN l2a {k} {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
