#!/usr/bin/env python3
"""gen_csr_vectors.py — CSR (tile register window) vector generator with an
INDEPENDENT golden register model, written from ARCHITECTURE.md §7 and the
documented bus/register contract in rtl/csr/csr_regs.sv (mirrors the FROZEN
apex_pkg constants; not derived from the RTL implementation).

Golden model scope: every architectural register (CTRL / STATUS / INFO_* /
TIER_CTRL / THRESHOLD_REG / FLUSH / IMPORTANCE_BASE / PERF_CTRL) including
W1C + sticky-set-wins races, reserved/unaligned 0xDEADBEEF (KVQ ISA
convention), field masking, tier-code clamping, and reset defaults. The
PERF_* cycle counters are cycle-accurate objects and are instead checked by
the TB against its own bus-level mirror (V records) — the golden model here
decides WHEN and HOW they are exercised.

Vector record format (streamed by tb_csr_sb.sv):
  W aa dddddddd            write strobe
  R aa dddddddd            read strobe + expected rdata
  S aa dddddddd eeeeeeee   simultaneous read+write: expect PRE-write value e
  Y a1 e1 a2 e2            two back-to-back (pipelined) reads
  B bb                     drive block_busy_i level
  P                        1-cycle desc_error_i pulse
  Z dddddddd               STATUS write RACING a desc_error pulse (set wins)
  T n                      idle n cycles
  V aa                     perf-counter read, checked against the TB mirror
  X                        mid-run hard reset (busy driven to 0 first)
  E                        end-of-run checks
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)

# ── frozen contract mirrors (apex_pkg.sv / csr_regs.sv parameters) ──────────
MXE_N = 8
APEX_VERSION = 0x0001_0000
CFG_D = 64
CFG_G = 128
TIER_BITMAP = 0x7          # CQ8|CQ4|CQ4P supported
N_BLOCKS = 8
IMP_AW = 7
KVQ_CQ8, KVQ_CQ4, KVQ_CQ4P = 0, 1, 2

A_CTRL, A_STATUS = 0x00, 0x04
A_INFO_N, A_INFO_D, A_INFO_G, A_INFO_TIER, A_INFO_VER = 0x08, 0x0C, 0x10, 0x14, 0x18
A_TIER_CTRL, A_THRESH, A_FLUSH, A_IMP = 0x20, 0x24, 0x28, 0x2C
A_PERF_CTRL, A_PERF_CYC = 0x30, 0x34
A_PERF_EVT = [0x38 + 4 * i for i in range(N_BLOCKS)]
PERF_ADDRS = set([A_PERF_CYC] + A_PERF_EVT)
DEFINED = [A_CTRL, A_STATUS, A_INFO_N, A_INFO_D, A_INFO_G, A_INFO_TIER,
           A_INFO_VER, A_TIER_CTRL, A_THRESH, A_FLUSH, A_IMP, A_PERF_CTRL]
RESERVED_SAMPLES = [0x1C, 0x58, 0x7C, 0xA0, 0xFC]
DEADBEEF = 0xDEADBEEF


def imp_data(a: int) -> int:
    """fake TIP importance window — MUST match tb_csr_sb.sv imp_data()."""
    return ((a * 0x9E37) ^ 0x1234) & 0xFFFF


def imp_tier(a: int) -> int:
    return a % 3


class CsrModel:
    """independent architectural register model (non-counter registers)."""

    def __init__(self):
        self.reset()
        self.busy = 0            # TB-driven block_busy_i level

    def reset(self):
        self.enable = 0
        self.sticky = 0
        self.tier = KVQ_CQ4      # §4 production default
        self.override = 0
        self.thresh = 1
        self.imp_addr = 0
        self.perf_en = 0

    def read(self, addr: int) -> int:
        if addr & 3:
            return DEADBEEF
        if addr == A_CTRL:
            return self.enable            # soft_reset bit reads 0
        if addr == A_STATUS:
            idle = 1 if self.busy == 0 else 0
            return (self.busy << 8) | (self.sticky << 1) | idle
        if addr == A_INFO_N:
            return MXE_N
        if addr == A_INFO_D:
            return CFG_D
        if addr == A_INFO_G:
            return CFG_G
        if addr == A_INFO_TIER:
            return TIER_BITMAP
        if addr == A_INFO_VER:
            return APEX_VERSION
        if addr == A_TIER_CTRL:
            return (self.override << 8) | self.tier
        if addr == A_THRESH:
            return self.thresh
        if addr == A_FLUSH:
            return 0
        if addr == A_IMP:
            return (imp_tier(self.imp_addr) << 16) | imp_data(self.imp_addr)
        if addr == A_PERF_CTRL:
            return self.perf_en           # clear bit reads 0
        if addr in PERF_ADDRS:
            raise RuntimeError("perf counters are V-record territory")
        return DEADBEEF

    def write(self, addr: int, d: int):
        if addr & 3:
            return
        if addr == A_CTRL:
            self.enable = d & 1
        elif addr == A_STATUS:
            if d & 2:
                self.sticky = 0
        elif addr == A_TIER_CTRL:
            t = d & 3
            self.tier = KVQ_CQ4 if t == 3 else t     # reserved code clamps
            self.override = (d >> 8) & 1
        elif addr == A_THRESH:
            self.thresh = d & 0x1F
        elif addr == A_IMP:
            self.imp_addr = d & ((1 << IMP_AW) - 1)
        elif addr == A_PERF_CTRL:
            self.perf_en = d & 1
        # STATUS other bits / INFO / FLUSH / reserved: no stored state


class Emitter:
    def __init__(self, path: Path, model: CsrModel):
        self.f = open(path, "w")
        self.m = model
        self.n = 0

    def raw(self, line: str):
        self.f.write(line + "\n")
        self.n += 1

    def wr(self, a, d):
        self.raw(f"W {a:02x} {d:08x}")
        self.m.write(a, d)

    def rd(self, a):
        self.raw(f"R {a:02x} {self.m.read(a):08x}")

    def simul(self, a, d):
        pre = DEADBEEF if (a & 3) else self.m.read(a)
        self.raw(f"S {a:02x} {d:08x} {pre:08x}")
        self.m.write(a, d)

    def b2b(self, a1, a2):
        self.raw(f"Y {a1:02x} {self.m.read(a1):08x} "
                 f"{a2:02x} {self.m.read(a2):08x}")

    def busy(self, lvl):
        self.raw(f"B {lvl:02x}")
        self.m.busy = lvl & 0xFF

    def derr(self):
        self.raw("P")
        self.m.sticky = 1

    def race(self, d):
        self.raw(f"Z {d:08x}")
        self.m.sticky = 1              # set wins over the same-cycle W1C

    def idle(self, n):
        self.raw(f"T {n}")

    def perf_rd(self, a):
        self.raw(f"V {a:02x}")

    def hard_reset(self):
        self.busy(0)
        self.idle(2)
        self.raw("X")
        self.m.reset()

    def end(self):
        self.raw("E")
        self.f.close()


def full_sweep(e: Emitter):
    for a in DEFINED + RESERVED_SAMPLES:
        e.rd(a)
    for a in (0x01, 0x02, 0x03, 0x06, 0x27, 0xFD):   # unaligned -> DEADBEEF
        e.rd(a)


def directed(e: Emitter):
    full_sweep(e)                                     # post-reset defaults

    # CTRL: enable level + soft_reset pulse (self-clearing, reads 0)
    e.wr(A_CTRL, 0x1); e.rd(A_CTRL)
    e.wr(A_CTRL, 0x3); e.rd(A_CTRL)                   # pulse; enable stays 1
    e.wr(A_CTRL, 0x2); e.rd(A_CTRL)                   # pulse; enable -> 0
    e.wr(A_CTRL, 0xFFFF_FFFC); e.rd(A_CTRL)           # field masking

    # TIER_CTRL: all 4 codes (3 clamps to CQ4) x override
    for t in (0, 1, 2, 3):
        e.wr(A_TIER_CTRL, t); e.rd(A_TIER_CTRL)
    e.wr(A_TIER_CTRL, 0x102); e.rd(A_TIER_CTRL)       # override + CQ4P
    e.wr(A_TIER_CTRL, 0xFFFF_FEFC); e.rd(A_TIER_CTRL)  # masking (code 0)
    e.wr(A_TIER_CTRL, 0x001); e.rd(A_TIER_CTRL)       # back to default

    # THRESHOLD_REG: every value + masking
    for t in range(32):
        e.wr(A_THRESH, t); e.rd(A_THRESH)
    e.wr(A_THRESH, 0xFFFF_FFE5); e.rd(A_THRESH)       # -> 5
    e.wr(A_THRESH, 1)

    # IMPORTANCE_BASE window
    for a in (0, 1, 42, 127):
        e.wr(A_IMP, a); e.rd(A_IMP)
    e.wr(A_IMP, 0xFFFF_FF80 | 5); e.rd(A_IMP)         # masking -> addr 5

    # STATUS: sticky set / W1C / re-clear / no-clear-without-bit1 / race
    e.derr(); e.rd(A_STATUS)
    e.wr(A_STATUS, 0x2); e.rd(A_STATUS)               # W1C clears
    e.wr(A_STATUS, 0x2); e.rd(A_STATUS)               # W1C on clear: still 0
    e.derr(); e.wr(A_STATUS, 0xFFFF_FFFD); e.rd(A_STATUS)  # bit1=0: no clear
    e.wr(A_STATUS, 0x2); e.rd(A_STATUS)
    e.race(0x2); e.rd(A_STATUS)                       # set wins
    e.wr(A_STATUS, 0x2); e.rd(A_STATUS)
    # writes to RO bits / RO registers: no effect
    e.wr(A_STATUS, 0xFFFF_FF01); e.rd(A_STATUS)
    e.wr(A_INFO_VER, 0); e.rd(A_INFO_VER)
    e.wr(A_INFO_N, 0xFFFF_FFFF); e.rd(A_INFO_N)
    e.wr(RESERVED_SAMPLES[1], 0x1234_5678); e.rd(RESERVED_SAMPLES[1])
    e.wr(0x02, 0xFFFF_FFFF); e.rd(A_CTRL)             # unaligned wr: ignored

    # busy bits / idle honesty
    e.busy(0xFF); e.rd(A_STATUS)
    e.busy(0x0F); e.rd(A_STATUS)
    e.busy(0xA5); e.rd(A_STATUS)
    e.busy(0x00); e.rd(A_STATUS)

    # FLUSH pulses (counted by the TB), WO reads 0
    e.wr(A_FLUSH, 0); e.wr(A_FLUSH, 0xFFFF_FFFF); e.rd(A_FLUSH)

    # PERF: enable, accumulate under several busy patterns, snapshot, clear,
    # clear+enable in one write, disabled hold
    e.wr(A_PERF_CTRL, 0x1); e.rd(A_PERF_CTRL)
    e.busy(0xFF); e.idle(37)
    e.busy(0x81); e.idle(23)
    e.busy(0x00); e.idle(11)
    for a in sorted(PERF_ADDRS):
        e.perf_rd(a)
    e.wr(A_PERF_CTRL, 0x2); e.rd(A_PERF_CTRL)         # clear + disable
    for a in sorted(PERF_ADDRS):
        e.perf_rd(a)
    e.busy(0x3C); e.idle(19)                          # disabled: must hold 0
    for a in sorted(PERF_ADDRS):
        e.perf_rd(a)
    e.wr(A_PERF_CTRL, 0x3)                            # clear + enable
    e.busy(0xFF)
    e.idle(13)
    for a in sorted(PERF_ADDRS):
        e.perf_rd(a)
    e.wr(A_PERF_CTRL, 0x0)                            # disable WITHOUT clear
    e.busy(0x55)
    e.idle(9)
    for a in sorted(PERF_ADDRS):
        e.perf_rd(a)                                  # nonzero, HELD
    e.busy(0x00)

    # simultaneous read+write (read-before-write) + pipelined reads
    e.simul(A_CTRL, 0x1); e.rd(A_CTRL)
    e.simul(A_THRESH, 0x1F); e.rd(A_THRESH)
    e.simul(A_TIER_CTRL, 0x100); e.rd(A_TIER_CTRL)
    e.simul(RESERVED_SAMPLES[0], 0xCAFE_F00D)
    e.b2b(A_INFO_N, A_INFO_VER)
    e.b2b(A_CTRL, A_STATUS)
    e.b2b(A_THRESH, RESERVED_SAMPLES[2])

    # mid-run hard reset -> defaults sweep again
    e.derr()
    e.wr(A_CTRL, 0x1)
    e.wr(A_THRESH, 17)
    e.wr(A_PERF_CTRL, 0x1)
    e.hard_reset()
    full_sweep(e)
    for a in sorted(PERF_ADDRS):                      # counters cleared
        e.perf_rd(a)


def randomized(e: Emitter, rng: random.Random, n_ops: int):
    wr_addrs = DEFINED + RESERVED_SAMPLES
    rd_addrs = [a for a in DEFINED] + RESERVED_SAMPLES
    for _ in range(n_ops):
        op = rng.choices(
            ["W", "R", "S", "Y", "B", "P", "Z", "T", "V"],
            weights=[24, 30, 4, 4, 8, 5, 3, 8, 14])[0]
        if op == "W":
            a = rng.choice(wr_addrs)
            d = rng.getrandbits(32)
            if a == A_PERF_CTRL and rng.random() < 0.5:
                d &= ~2                      # don't clear counters every time
            e.wr(a, d)
        elif op == "R":
            a = rng.choice(rd_addrs)
            if rng.random() < 0.1:
                a += rng.randint(1, 3)       # unaligned
            e.rd(a)
        elif op == "S":
            e.simul(rng.choice(wr_addrs), rng.getrandbits(32))
        elif op == "Y":
            e.b2b(rng.choice(rd_addrs), rng.choice(rd_addrs))
        elif op == "B":
            e.busy(rng.getrandbits(8))
        elif op == "P":
            e.derr()
            if rng.random() < 0.6:
                e.rd(A_STATUS)
        elif op == "Z":
            e.race(rng.getrandbits(32))
            e.rd(A_STATUS)
        elif op == "T":
            e.idle(rng.randint(1, 20))
        elif op == "V":
            e.perf_rd(rng.choice(sorted(PERF_ADDRS)))
    # settle: known busy, drain sticky, final sweep
    e.busy(0)
    e.wr(A_STATUS, 0x2)
    full_sweep(e)


def saturation(e: Emitter):
    """PERF_W=8 build: counters must PEG at 0xFF, never wrap."""
    e.wr(A_PERF_CTRL, 0x3)
    e.busy(0xFF)
    e.idle(300)                                       # >> 2^8 cycles
    e.busy(0x00)
    for a in sorted(PERF_ADDRS):
        e.perf_rd(a)                                  # all pegged
    e.wr(A_PERF_CTRL, 0x3)                            # clear restarts from 0
    e.busy(0x01)
    e.idle(20)
    e.busy(0x00)
    for a in sorted(PERF_ADDRS):
        e.perf_rd(a)
    e.wr(A_PERF_CTRL, 0x0)


def main():
    # directed
    m = CsrModel()
    e = Emitter(OUT / "vectors_directed.txt", m)
    e.raw("# csr directed vectors (golden: gen_csr_vectors.py)")
    directed(e)
    e.end()
    print(f"directed: {e.n} records -> build/vectors_directed.txt")

    # randomized (fresh model; deterministic seed)
    rng = random.Random(0xC5A_2026)
    m = CsrModel()
    e = Emitter(OUT / "vectors_random.txt", m)
    e.raw("# csr randomized vectors, seed 0xC5A2026")
    randomized(e, rng, 2500)
    e.hard_reset()
    full_sweep(e)
    randomized(e, rng, 1000)
    e.end()
    print(f"random: {e.n} records -> build/vectors_random.txt")

    # PERF_W=8 saturation
    m = CsrModel()
    e = Emitter(OUT / "vectors_sat.txt", m)
    e.raw("# csr perf-saturation vectors (PERF_W=8 build only)")
    saturation(e)
    e.end()
    print(f"sat: {e.n} records -> build/vectors_sat.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
