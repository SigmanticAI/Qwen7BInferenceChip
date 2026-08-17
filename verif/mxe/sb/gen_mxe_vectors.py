#!/usr/bin/env python3
"""gen_mxe_vectors.py — MXE scoreboard-TB vector generator (V0 pattern).

Golden arbiter: apex_golden.compute (gemm_i8 + requant_i32_to_i8), per
ARCHITECTURE.md C-2/C-3. This script maintains a bit-accurate model of the
MXE's RESIDENT accumulator state (mxe_array.sv OS semantics: 64x8 INT32
register file, clr on pass 0 of overwriting jobs, non-destructive drain,
lanes >= N masked to zero at drain, INT32 two's-complement wraparound on
cross-descriptor accumulate chains) and emits per-job stimulus + expected
result beats for tb_mxe_sb.sv.

Output files (into argv[1] directory):
  vectors_directed.txt  directed cases: identity, all -128 worst case at
                        K=K_MAX, saturation both signs, RNE half-to-even tie
                        points, M/K/N edges incl. 1, OS accumulate chains,
                        illegal descriptors interleaved (D-019 regression)
  vectors_random.txt    >=500 seeded random descriptor jobs + interleaved
                        illegal descriptors
  vectors_storm.txt     random jobs sized for backpressure-storm / stalled-
                        feed protocol runs
  vectors_reset.txt     mid-operation reset jobs (X lines with abort cycle)
                        each followed by a legal job proving clean recovery

File format (one token line per record, '#' = comment):
  J op m k n acc rq scale shift bufa bufb bufd ftie fsatp fsatn   legal job
  I op m k n acc rq scale shift bufa bufb bufd                    illegal desc
  X op m k n acc rq scale shift bufa bufb bufd abortcyc           mid-op reset
  A <16 hex>      one activation beat (row-major, chunk-minor; mxe_ctrl order)
  W <16 hex>      one weight beat (chunk-major, ascending column)
  E <8x 8-hex>    one expected result beat, lane c = word c (32-bit)
Padding lanes (k tail, unused act lanes) carry pseudo-random GARBAGE so the
DUT's masking — not benign vectors — is what is proven.
"""
import os
import sys

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
from apex_golden.compute import gemm_i8, requant_i32_to_i8  # noqa: E402

MXE_N = 8
M_MAX = 64
K_MAX = 2048
OP_NOP, OP_WS, OP_OS = 0, 1, 2


def wrap32(x):
    """int64 array -> INT32 two's-complement wraparound (RTL acc_t semantics)."""
    return ((x.astype(np.int64) + 2**31) % 2**32) - 2**31


class Env:
    """Vector emitter + resident-accumulator model for one output file."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.res = np.zeros((M_MAX, MXE_N), dtype=np.int64)  # resident acc
        self.lines = []
        self.n_legal = 0
        self.n_illegal = 0
        self.n_reset = 0

    def emit(self, s):
        self.lines.append(s)

    def _pack_act(self, row, p, k):
        w = 0
        for j in range(8):
            idx = 8 * p + j
            v = int(row[idx]) if idx < k else int(self.rng.integers(0, 256))
            w |= (v & 0xFF) << (8 * j)
        return f"{w:016x}"

    def _pack_wgt(self, B, p, c, k):
        w = 0
        for r in range(8):
            idx = 8 * p + r
            v = int(B[idx, c]) if idx < k else int(self.rng.integers(0, 256))
            w |= (v & 0xFF) << (8 * r)
        return f"{w:016x}"

    def _emit_data(self, A, B, m, k, n):
        kb = (k + 7) // 8
        for mm in range(m):
            for p in range(kb):
                self.emit("A " + self._pack_act(A[mm], p, k))
        for p in range(kb):
            for c in range(n):
                self.emit("W " + self._pack_wgt(B, p, c, k))

    def _apply_job_model(self, op, m, k, n, acc, A, B):
        """Update the resident model; return the m x 8 drain view (int64)."""
        C = gemm_i8(A, B).astype(np.int64)          # golden arbiter (C-3)
        Cf = np.zeros((m, MXE_N), dtype=np.int64)
        Cf[:, :n] = C
        clr = not (op == OP_OS and acc)             # WS always clears; OS acc=0
        if clr:
            self.res[:m, :] = Cf                    # in-INT32 by golden assert
        else:
            self.res[:m, :] = wrap32(self.res[:m, :] + Cf)
        drain = self.res[:m, :].copy()
        drain[:, n:] = 0                            # n_active lane mask (RTL)
        return drain

    def legal(self, op, m, k, n, acc=0, rq=0, scale=0, shift=0, A=None, B=None):
        rng = self.rng
        if A is None:
            A = rng.integers(-128, 128, size=(m, k), dtype=np.int64).astype(np.int8)
        if B is None:
            B = rng.integers(-128, 128, size=(k, n), dtype=np.int64).astype(np.int8)
        A = np.asarray(A, dtype=np.int8).reshape(m, k)
        B = np.asarray(B, dtype=np.int8).reshape(k, n)
        drain = self._apply_job_model(op, m, k, n, acc, A, B)

        ftie = fsp = fsn = 0
        if rq:
            # pre-clip q for saturation/tie flags (mirrors golden internals)
            p = drain * int(scale)
            if shift > 0:
                half = np.int64(1) << (shift - 1)
                mask = (np.int64(1) << shift) - 1
                q = p >> shift
                rem = p & mask
                ftie = int(bool(np.any(rem == half)))
                q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1))).astype(np.int64)
            else:
                q = p
            fsp = int(bool(np.any(q > 127)))
            fsn = int(bool(np.any(q < -128)))
            y = requant_i32_to_i8(drain.astype(np.int32), scale, shift)  # arbiter
            assert np.array_equal(np.clip(q, -128, 127), y.astype(np.int64)), \
                "internal flag model diverged from golden requant"
            outw = y.astype(np.int64) & 0xFFFFFFFF   # sign-extended INT8 lane
        else:
            outw = drain & 0xFFFFFFFF                # raw INT32 lane

        self.emit(f"J {op} {m} {k} {n} {int(bool(acc))} {int(bool(rq))} "
                  f"{scale} {shift} 0 0 0 {ftie} {fsp} {fsn}")
        self._emit_data(A, B, m, k, n)
        for mm in range(m):
            self.emit("E " + " ".join(f"{int(v):08x}" for v in outw[mm]))
        self.n_legal += 1

    def illegal(self, op, m, k, n, ba=0, bb=0, bd=0):
        self.emit(f"I {op} {m} {k} {n} 0 0 1 0 {ba} {bb} {bd}")
        self.n_illegal += 1

    def reset_job(self, op, m, k, n, abortcyc=0, phase=0, acc=0):
        """phase != 0: TB waits until the ctrl FSM reaches that state (1=INGEST
        .. 6=WAIT_DONE) and resets there — exact phase targeting. phase == 0:
        reset after `abortcyc` cycles (randomized point)."""
        rng = self.rng
        A = rng.integers(-128, 128, size=(m, k), dtype=np.int64).astype(np.int8)
        B = rng.integers(-128, 128, size=(k, n), dtype=np.int64).astype(np.int8)
        self.emit(f"X {op} {m} {k} {n} {int(bool(acc))} 0 0 0 0 0 0 "
                  f"{abortcyc} {phase}")
        self._emit_data(A, B, m, k, n)
        self.res[:, :] = 0                           # DUT reset zeroes the acc
        self.n_reset += 1

    def save(self, path, title):
        hdr = [f"# {title}",
               f"# jobs={self.n_legal} illegal={self.n_illegal} resets={self.n_reset}",
               "# generated by gen_mxe_vectors.py — golden arbiter apex_golden.compute"]
        with open(path, "w") as f:
            f.write("\n".join(hdr + self.lines) + "\n")
        print(f"wrote {path}: {self.n_legal} legal, {self.n_illegal} illegal, "
              f"{self.n_reset} reset jobs, {len(self.lines)} lines")


# ── directed set ─────────────────────────────────────────────────────────────

def build_directed(path):
    e = Env(seed=0xD1EC7ED)

    # identity: B = I(8), raw INT32 drain must equal sign-extended A
    e.legal(OP_WS, 8, 8, 8, B=np.eye(8, dtype=np.int8))

    # all -128 worst case at K = K_MAX (C-3 corner: acc = 2048*16384 = 2^25)
    e.legal(OP_WS, 4, K_MAX, 8,
            A=np.full((4, K_MAX), -128, dtype=np.int8),
            B=np.full((K_MAX, 8), -128, dtype=np.int8))
    # same magnitude, negative accumulators (-128 * 127 * 2048)
    e.legal(OP_WS, 2, K_MAX, 8,
            A=np.full((2, K_MAX), -128, dtype=np.int8),
            B=np.full((K_MAX, 8), 127, dtype=np.int8))

    # saturation both signs through the requant epilogue (scale=1 shift=0
    # on the +/-2^25 accumulators: q = +/-2^25 -> clip to +127 / -128)
    e.legal(OP_WS, 2, K_MAX, 8, rq=1, scale=1, shift=0,
            A=np.full((2, K_MAX), -128, dtype=np.int8),
            B=np.full((K_MAX, 8), -128, dtype=np.int8))
    e.legal(OP_WS, 2, K_MAX, 8, rq=1, scale=1, shift=0,
            A=np.full((2, K_MAX), -128, dtype=np.int8),
            B=np.full((K_MAX, 8), 127, dtype=np.int8))

    # RNE half-to-even tie points (C-2): K=1 so acc == A*B exactly.
    # shift=1: acc odd -> rem == half. Ties on both parities and both signs:
    #   +1 -> q=0 (even, keep), +3 -> q=1 (odd, +1), -1 -> q=-1 (odd, +1),
    #   -3 -> q=-2 (even, keep)
    e.legal(OP_WS, 1, 1, 8, rq=1, scale=1, shift=1,
            A=np.array([[1]], dtype=np.int8),
            B=np.array([[1, 3, -1, -3, 5, -5, 7, -7]], dtype=np.int8))
    # shift=6 ties at rem == 32: acc in {32, 96, -32, -96}
    e.legal(OP_WS, 1, 1, 8, rq=1, scale=1, shift=6,
            A=np.array([[1]], dtype=np.int8),
            B=np.array([[32, 96, -32, -96, 31, 33, -31, -33]], dtype=np.int8))
    # scale-driven ties: acc*scale ties at shift=4 (half=8): 8=8*1, 24=8*3
    e.legal(OP_WS, 1, 1, 8, rq=1, scale=8, shift=4,
            A=np.array([[1]], dtype=np.int8),
            B=np.array([[1, 3, -1, -3, 2, -2, 127, -128]], dtype=np.int8))

    # M/K/N edge sizes incl. 1
    e.legal(OP_WS, 1, 1, 1)
    e.legal(OP_WS, 1, 1, 8)
    e.legal(OP_WS, 64, 1, 1)
    e.legal(OP_WS, 64, 8, 8)
    e.legal(OP_WS, 1, 2048, 1)
    e.legal(OP_WS, 2, 2048, 3, rq=1, scale=3, shift=11)
    e.legal(OP_WS, 64, 2048, 8)          # full-capacity tile (abuf all 16384)
    # K tails (klast = 1..7) with multi-pass
    for (m, k, n) in [(4, 9, 8), (4, 15, 5), (3, 17, 2), (5, 7, 4),
                      (2, 23, 8), (6, 33, 6), (1, 1023, 7)]:
        e.legal(OP_WS, m, k, n)
    # N sweep 1..8
    for n in range(1, 9):
        e.legal(OP_WS, 2, 8, n)
    # M sweep edges
    for m in [1, 2, 63, 64]:
        e.legal(OP_WS, m, 16, 8, rq=1, scale=7, shift=9)

    # OS accumulate chains (raw INT32 drain after EACH descriptor)
    e.legal(OP_OS, 8, 8, 8, acc=0)
    e.legal(OP_OS, 8, 8, 8, acc=1)
    e.legal(OP_OS, 8, 8, 8, acc=1)
    e.legal(OP_OS, 8, 8, 8, acc=1)       # chain of 4
    # chain with K tails and requant only on the final link
    e.legal(OP_OS, 4, 13, 5, acc=0)
    e.legal(OP_OS, 4, 11, 5, acc=1)
    e.legal(OP_OS, 4, 9, 5, acc=1, rq=1, scale=5, shift=8)
    # OS acc=1 directly after a WS job (documented residency retention)
    e.legal(OP_WS, 4, 8, 8)
    e.legal(OP_OS, 4, 8, 8, acc=1)

    # illegal descriptors interleaved between legal jobs (run1/D-019
    # regression: the NEXT legal job must be bit-exact, desc_ready gated)
    e.legal(OP_WS, 4, 8, 8, rq=1, scale=2, shift=5)
    e.illegal(OP_NOP, 4, 8, 8)           # bad opcode (OP_NOP)
    e.illegal(0x7F, 4, 8, 8)             # bad opcode (undefined)
    e.legal(OP_WS, 4, 8, 8, rq=1, scale=2, shift=5)
    e.illegal(OP_WS, 4, 0, 8)            # K = 0
    e.illegal(OP_WS, 4, 2049, 8)         # K = K_MAX + 1
    e.illegal(OP_WS, 4, 4095, 8)         # K = field max
    e.legal(OP_OS, 4, 16, 8, acc=0)
    e.illegal(OP_WS, 0, 8, 8)            # M = 0
    e.illegal(OP_WS, 65, 8, 8)           # M > M_TILE_MAX
    e.illegal(OP_WS, 4095, 8, 8)         # M = field max
    e.legal(OP_OS, 4, 16, 8, acc=1)      # chain across the rejects
    e.illegal(OP_WS, 4, 8, 0)            # N = 0
    e.illegal(OP_WS, 4, 8, 9)            # N > MXE_N
    e.illegal(OP_WS, 4, 8, 8, ba=1)      # unimplemented src A buffer
    e.illegal(OP_WS, 4, 8, 8, bb=2)      # unimplemented src B buffer
    e.illegal(OP_WS, 4, 8, 8, bd=3)      # unimplemented dst buffer
    e.legal(OP_WS, 4, 8, 8, rq=1, scale=2, shift=5)

    e.save(path, "MXE directed vectors")
    return e


# ── random set ───────────────────────────────────────────────────────────────

def _rand_dims(rng):
    m = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 63, 64],
                       p=[.10, .09, .08, .09, .06, .06, .06, .10,
                          .06, .06, .05, .05, .04, .05, .05]))
    kpool = ([1, 2, 3, 5, 7, 8, 9, 13, 15, 16, 17, 24, 31, 33, 40, 55, 64]
             + [int(rng.integers(65, 300))])
    k = int(rng.choice(kpool))
    n = int(rng.integers(1, 9))
    return m, k, n


def build_random(path, n_jobs=520, n_illegal=32, seed=0xA9EC2026):
    e = Env(seed=seed)
    rng = e.rng
    ill_slots = set(rng.choice(n_jobs, size=n_illegal, replace=False).tolist())
    for i in range(n_jobs):
        if i in ill_slots:
            kind = int(rng.integers(0, 8))
            if kind == 0:
                e.illegal(int(rng.choice([0, 3, 0x55, 0xFF])), 4, 8, 8)
            elif kind == 1:
                e.illegal(OP_WS, 4, 0, 8)
            elif kind == 2:
                e.illegal(OP_OS, 4, int(rng.integers(2049, 4096)), 8)
            elif kind == 3:
                e.illegal(OP_WS, 0, 8, 8)
            elif kind == 4:
                e.illegal(OP_OS, int(rng.integers(65, 4096)), 8, 8)
            elif kind == 5:
                e.illegal(OP_WS, 4, 8, 0)
            elif kind == 6:
                e.illegal(OP_WS, 4, 8, int(rng.integers(9, 16)))
            else:
                e.illegal(OP_WS, 4, 8, 8, ba=int(rng.integers(1, 16)))
        m, k, n = _rand_dims(rng)
        if i % 97 == 0:
            k = 2048                       # a few full-K jobs
            m = min(m, 4)
        op = OP_OS if rng.random() < 0.35 else OP_WS
        acc = int(op == OP_OS and rng.random() < 0.5)
        rq = int(rng.random() < 0.5)
        scale = int(rng.integers(0, 65536)) if rng.random() < 0.9 else \
            int(rng.choice([0, 1, 2, 65535]))
        shift = int(rng.integers(0, 32)) if rng.random() < 0.3 else \
            int(rng.integers(0, 14))
        e.legal(op, m, k, n, acc=acc, rq=rq, scale=scale, shift=shift)
    e.save(path, f"MXE random vectors (seed={seed:#x})")
    return e


# ── storm set (protocol-adversarial runs reuse smaller jobs) ────────────────

def build_storm(path, n_jobs=90, seed=0x570A21):
    e = Env(seed=seed)
    rng = e.rng
    for i in range(n_jobs):
        if i % 9 == 4:
            e.illegal(OP_NOP, 2, 8, 4)
        m = int(rng.integers(1, 9))
        k = int(rng.choice([1, 3, 7, 8, 9, 13, 16, 17, 24]))
        n = int(rng.integers(1, 9))
        op = OP_OS if rng.random() < 0.4 else OP_WS
        acc = int(op == OP_OS and rng.random() < 0.5)
        rq = int(rng.random() < 0.5)
        e.legal(op, m, k, n, acc=acc, rq=rq,
                scale=int(rng.integers(0, 65536)), shift=int(rng.integers(0, 14)))
    e.save(path, "MXE storm vectors (backpressure/stall protocol runs)")
    return e


# ── mid-operation reset set ─────────────────────────────────────────────────
# Phase-targeted resets (phase 1..6 = S_INGEST..S_WAIT_DONE): the TB observes
# the ctrl FSM and resets exactly in the requested state, so every phase is
# provably hit. A cycle-count abort per shape adds an untargeted random point.
# The TB records the ACTUAL phase at reset for the coverage table.

def build_reset(path, seed=0x2E5E7):
    e = Env(seed=seed)
    rng = e.rng

    shapes = [(16, 16, 8), (32, 8, 5), (8, 24, 8), (24, 16, 3), (16, 13, 6)]
    for (m, k, n) in shapes:
        kb = (k + 7) // 8
        ing = m * kb
        # each FSM phase, exactly targeted (1=INGEST .. 6=WAIT_DONE)
        for ph in range(1, 7):
            e.reset_job(OP_WS, m, k, n, phase=ph)
            e.legal(OP_WS, m, k, n, rq=1, scale=9, shift=9)
        # untargeted random abort point, OS accumulate job interrupted
        e.reset_job(OP_OS, m, k, n, acc=1,
                    abortcyc=int(rng.integers(2, 2 * ing + kb * (m + 30))))
        e.legal(OP_OS, m, k, n, acc=0)
        e.legal(OP_OS, m, k, n, acc=1)   # chain proves acc state fully clean
    e.save(path, "MXE mid-operation reset vectors (run with +stall_mode=0)")
    return e


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "build"
    os.makedirs(outdir, exist_ok=True)
    build_directed(os.path.join(outdir, "vectors_directed.txt"))
    build_random(os.path.join(outdir, "vectors_random.txt"))
    build_storm(os.path.join(outdir, "vectors_storm.txt"))
    build_reset(os.path.join(outdir, "vectors_reset.txt"))
