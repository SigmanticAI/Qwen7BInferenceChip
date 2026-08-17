#!/usr/bin/env python3
"""gen_seq_vectors.py — SEQ (seq_walker) vector generator with an
INDEPENDENT golden queue model, written from ARCHITECTURE.md §1 SEQ row /
§3 / §5 / D-006 and the abort contract documented in rtl/seq/seq_walker.sv.

Golden queue model: descriptors accepted on the ds stream are walked IN
ORDER, one job at a time (D-006); every legal descriptor eventually yields
exactly one done, every illegal one exactly one desc_error and no done; a
soft abort discards everything not yet dispatched and lets the in-flight
job finish. The model therefore predicts, for every drain point (N record),
the cumulative done/err counts since the last abort/reset sync, and the TB
checks dispatch ORDER and CONTENT beat-exactly against the pushed sequence.

Descriptor legality here is the §3 conservative mirror and MUST match
mxe_stub.sv desc_legal(): opcode ∈ {1,2}, 1 <= K <= 2048, 1 <= M <= 64,
1 <= N <= 8, all buffer ids == 0. The TB cross-checks this flag against
the stub's own verdict on every accept (dual-oracle discipline).

Vector record format (streamed by tb_seq_sb.sv):
  D <hex32> <legal> <cat>   push one 128-bit descriptor (cat = coverage id)
  N <done> <err>            drain; check cumulative done/err since last sync
  A <kind>                  soft abort: 0 wait-for-in-flight, 1 queued-only
                            (dispatch disabled), 2 empty; counters resync
  G <0|1>                   CTRL.enable level
  Q <full> <count> <undisp> settled queue-state check (use with enable=0)
  R <phase>                 mid-op HARD reset, phase-targeted (1..4); resync
  E                         end-of-run checks

cats: 0 legal_plain, 1 legal_k1, 2 legal_k2048, 3 legal_m1, 4 bad_opcode,
      5 k_zero, 6 k_big, 7 m_zero, 8 n_zero, 9 m_big, 10 n_big, 11 buf_bad
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)

K_MAX, M_MAX, N_MAX = 2048, 64, 8
OP_WS, OP_OS = 1, 2

CATS = ["legal_plain", "legal_k1", "legal_k2048", "legal_m1", "bad_opcode",
        "k_zero", "k_big", "m_zero", "n_zero", "m_big", "n_big", "buf_bad"]


def pack(op, m, k, n, scale=0, shift=0, rq=0, acc=0, os=0,
         sa=0, sb=0, dst=0, rsv=0):
    """mxe_desc_t layout (apex_pkg.sv, FROZEN): LSB-first fields."""
    assert 0 <= op < 256 and 0 <= m < 4096 and 0 <= k < 4096 and 0 <= n < 4096
    v = (op | (m << 8) | (k << 20) | (n << 32) | ((scale & 0xFFFF) << 44)
         | ((shift & 0x1F) << 60) | ((rq & 1) << 65) | ((acc & 1) << 66)
         | ((os & 1) << 67) | ((sa & 0xF) << 68) | ((sb & 0xF) << 72)
         | ((dst & 0xF) << 76) | ((rsv & ((1 << 48) - 1)) << 80))
    return v


def legal(op, m, k, n, sa=0, sb=0, dst=0):
    return (op in (OP_WS, OP_OS) and 1 <= k <= K_MAX and 1 <= m <= M_MAX
            and 1 <= n <= N_MAX and sa == 0 and sb == 0 and dst == 0)


class Emitter:
    """golden queue model + record emitter (counters resync on A/R)."""

    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.done = 0
        self.err = 0
        self.pushed = 0        # since last drain/sync (for Q records)
        self.n_descs = 0

    def raw(self, line):
        self.f.write(line + "\n")

    def desc(self, op, m, k, n, cat, **kw):
        v = pack(op, m, k, n, **kw)
        lg = legal(op, m, k, n, kw.get("sa", 0), kw.get("sb", 0),
                   kw.get("dst", 0))
        self.raw(f"D {v:032x} {int(lg)} {cat}")
        if lg:
            self.done += 1
        else:
            self.err += 1
        self.pushed += 1
        self.n_descs += 1

    def drain(self):
        self.raw(f"N {self.done} {self.err}")
        self.pushed = 0

    def abort(self, kind):
        self.raw(f"A {kind}")
        self.done = 0
        self.err = 0
        self.pushed = 0

    def gate(self, v):
        self.raw(f"G {int(v)}")

    def qcheck(self, full, count, undisp):
        self.raw(f"Q {int(full)} {count} {undisp}")

    def reset(self, phase):
        self.raw(f"R {phase}")
        self.done = 0
        self.err = 0
        self.pushed = 0

    def end(self):
        self.raw("E")
        self.f.close()


def rand_legal(e, rng, cat=0, m_hi=12):
    op = rng.choice((OP_WS, OP_OS))
    m = rng.randint(1, m_hi)
    k = rng.randint(1, K_MAX)
    n = rng.randint(1, N_MAX)
    if cat == 1:
        k = 1
    if cat == 2:
        k = K_MAX
    if cat == 3:
        m = 1
    e.desc(op, m, k, n, cat, scale=rng.getrandbits(16),
           shift=rng.getrandbits(5), rq=rng.getrandbits(1),
           acc=rng.getrandbits(1), os=int(op == OP_OS),
           rsv=rng.getrandbits(48))


def rand_illegal(e, rng, cat):
    op, m, k, n, sa, sb, dst = (rng.choice((OP_WS, OP_OS)),
                                rng.randint(1, 12), rng.randint(1, K_MAX),
                                rng.randint(1, N_MAX), 0, 0, 0)
    if cat == 4:
        op = rng.choice((0, 3, 0x7F, 0xFF))
    elif cat == 5:
        k = 0
    elif cat == 6:
        k = rng.choice((K_MAX + 1, 3000, 4095))
    elif cat == 7:
        m = 0
    elif cat == 8:
        n = 0
    elif cat == 9:
        m = rng.choice((M_MAX + 1, 100, 4095))
    elif cat == 10:
        n = rng.choice((N_MAX + 1, 12, 4095))
    elif cat == 11:
        which = rng.randint(0, 2)
        val = rng.randint(1, 15)
        sa, sb, dst = ((val, 0, 0), (0, val, 0), (0, 0, val))[which]
    e.desc(op, m, k, n, cat, sa=sa, sb=sb, dst=dst, rsv=rng.getrandbits(48))


def directed(e, rng):
    # single legal job, then a burst
    rand_legal(e, rng)
    e.drain()
    for _ in range(3):
        rand_legal(e, rng)
    e.drain()

    # every illegal category (rejected: desc_error, NO done, NO beats)
    for cat in (4, 5, 6, 7, 8, 9, 10, 11):
        rand_illegal(e, rng, cat)
        e.drain()

    # legality boundaries: K=1, K=2048, M=1, plus M=64/N=8 corners
    rand_legal(e, rng, cat=1)
    rand_legal(e, rng, cat=2)
    rand_legal(e, rng, cat=3)
    e.desc(OP_WS, M_MAX, 7, N_MAX, 0)          # M=64: longest beat stream
    e.desc(OP_OS, 2, K_MAX, N_MAX, 2, os=1)
    e.drain()

    # mixed legal/illegal interleave (reject must not disturb the walk)
    for cat in (0, 5, 0, 4, 3, 6, 0, 11):
        (rand_legal if cat in (0, 1, 2, 3) else rand_illegal)(e, rng, cat)
    e.drain()

    # queue-full: dispatch disabled, 10 pushes = 8 FIFO + 2 skid
    e.gate(0)
    for _ in range(10):
        rand_legal(e, rng, m_hi=6)
    e.qcheck(1, 8, 10)
    e.gate(1)
    e.drain()

    # aborts: empty / queued-only / with a job in flight
    e.abort(2)
    rand_legal(e, rng)
    e.drain()

    e.gate(0)
    for _ in range(5):
        rand_legal(e, rng, m_hi=6)
    e.abort(1)                                  # nothing dispatched: all dropped
    e.gate(1)
    rand_legal(e, rng)
    rand_legal(e, rng)
    e.drain()

    for _ in range(6):
        e.desc(rng.choice((OP_WS, OP_OS)), rng.randint(24, 48),
               rng.randint(1, K_MAX), rng.randint(1, N_MAX), 0)
    e.abort(0)                                  # in-flight job must FINISH
    for _ in range(3):
        rand_legal(e, rng)
    e.drain()

    # enable gating mid-stream
    rand_legal(e, rng)
    e.gate(0)
    rand_legal(e, rng)
    e.gate(1)
    e.drain()
    e.end()


def randomized(e, rng, n_batches):
    for b in range(n_batches):
        if b % 9 == 4:
            e.gate(0)
            for _ in range(rng.randint(4, 10)):
                rand_legal(e, rng, m_hi=6)
            if rng.random() < 0.5:
                e.abort(1)
                e.gate(1)
            else:
                e.gate(1)
                e.drain()
            continue
        for _ in range(rng.randint(6, 16)):
            r = rng.random()
            if r < 0.62:
                rand_legal(e, rng, cat=rng.choice((0, 0, 0, 1, 2, 3)))
            else:
                rand_illegal(e, rng, rng.randint(4, 11))
        if b % 7 == 5:
            e.abort(0)
        else:
            e.drain()
    e.drain()
    e.end()


def reset_seq(e, rng):
    for phase in (1, 2, 3, 4):
        e.reset(phase)
        for _ in range(3):
            rand_legal(e, rng)
        rand_illegal(e, rng, rng.randint(4, 11))
        e.drain()                               # full recovery after each phase
    e.end()


def main():
    rng = random.Random(0x53E_2026)
    e = Emitter(OUT / "vectors_directed.txt")
    e.raw("# seq directed vectors (golden: gen_seq_vectors.py)")
    directed(e, rng)
    print(f"directed: {e.n_descs} descriptors -> build/vectors_directed.txt")

    e = Emitter(OUT / "vectors_random.txt")
    e.raw("# seq randomized vectors, seed 0x53E2026")
    randomized(e, rng, 40)
    if e.n_descs < 200:
        raise SystemExit("randomized descriptor count fell below 200")
    print(f"random: {e.n_descs} descriptors -> build/vectors_random.txt")

    e = Emitter(OUT / "vectors_reset.txt")
    e.raw("# seq mid-op reset vectors")
    reset_seq(e, rng)
    print(f"reset: {e.n_descs} descriptors -> build/vectors_reset.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
