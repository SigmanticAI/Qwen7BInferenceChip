#!/usr/bin/env python3
"""gen_check_vectors.py — IB-WALK stage 3: SV-vs-mirror check equivalence.

The descriptor legality functions exist twice by design (seq_walker_pkg.sv
walk_desc_check/walk_desc2_check and seq_walker_fmt.py check1/check2 — the
single-source-in-two-places rule of IB_WALK.md §2.2). This generator makes
that equivalence a DYNAMIC gate: it emits the shared directed corpora plus
biased-random word tuples with the MIRROR's verdict attached, and
tb_check_sb replays them through the SV functions — any clause-order or
field-extraction divergence between the two implementations fails loudly.

Random strategy: uniform words almost always trip the first clauses (fmt /
resv), so most vectors start from a LEGAL base and corrupt 1-3 fields —
that reaches the deep clauses (geometry products, divisibility, kv_map,
mask) that uniform noise never exercises. A produced-code histogram is
asserted: every reachable code {NONE, TIER, DESC} must appear for BOTH
checks, so silent coverage collapse fails the generator itself.

Line grammar (build/check_vectors.txt):
    C2 <w0> <w1> <w2> <w3> <w21> <cfg_d> <exp>
    C1 <geom> <rq> <mask> <cfg_d> <exp>
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import seq_walker_fmt as fmt  # noqa: E402

N_MUTATED = 5000
N_UNIFORM = 1000


def v2_bases() -> list[tuple[tuple[int, ...], int]]:
    b7 = ((fmt.pack_geom0(128), fmt.pack_model0(3584, 18944),
           fmt.pack_model1(28, 4, 1), fmt.pack_mask(fmt.EN_ALL),
           fmt.pack_step(8, 7)), 128)
    bt = ((fmt.pack_geom0(64), fmt.pack_model0(128, 256),
           fmt.pack_model1(2, 1, 1), fmt.pack_mask(0x272),
           fmt.pack_step(8, 7)), 64)
    return [b7, bt]


def mutate_v2(rnd: random.Random, words: tuple[int, ...], cfgd: int):
    w = list(words)
    for _ in range(rnd.randint(1, 3)):
        op = rnd.randrange(8)
        if op == 0:                                   # random fmt nibble
            w[0] = (w[0] & 0x0FFF_FFFF) | (rnd.randrange(16) << 28)
        elif op == 1:                                 # flip one random bit
            i = rnd.randrange(5)
            w[i] ^= 1 << rnd.randrange(32)
        elif op == 2:                                 # tier bits
            w[0] = (w[0] & ~(0x3 << 16)) | (rnd.randrange(4) << 16)
        elif op == 3:                                 # t_rows / pos bytes
            w[4] = fmt.pack_step(rnd.randrange(256), rnd.randrange(256))
        elif op == 4:                                 # H / H_kv bytes
            w[2] = (w[2] & ~0x00FF_FF00) \
                | (rnd.randrange(256) << 16) | (rnd.randrange(256) << 8)
        elif op == 5:                                 # kv_map
            w[2] = (w[2] & ~0x3) | rnd.randrange(4)
        elif op == 6:                                 # mask word
            w[3] = rnd.randrange(1 << 13)
        else:                                         # build mismatch
            cfgd = 192 - cfgd                         # 64 <-> 128
    return tuple(w), cfgd


def main() -> int:
    rnd = random.Random(0xC4EC)
    out = HERE / "build"
    out.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    hist2 = {0: 0, 1: 0, 2: 0}
    hist1 = {0: 0, 1: 0, 2: 0}

    for words, cfgd, exp, _name in fmt.directed_check2_cases():
        L.append(f"C2 {words[0]:08x} {words[1]:08x} {words[2]:08x} "
                 f"{words[3]:08x} {words[4]:08x} {cfgd} {exp}")
        hist2[exp] += 1
    for words, cfgd, exp, _name in fmt.directed_check1_cases():
        L.append(f"C1 {words[0]:08x} {words[1]:08x} {words[2]:08x} "
                 f"{cfgd} {exp}")
        hist1[exp] += 1

    for _ in range(N_MUTATED):
        base, cfgd = rnd.choice(v2_bases())
        w, cfgd = mutate_v2(rnd, base, cfgd)
        exp = fmt.check2(*w, cfg_d=cfgd)
        hist2[exp] += 1
        L.append(f"C2 {w[0]:08x} {w[1]:08x} {w[2]:08x} {w[3]:08x} "
                 f"{w[4]:08x} {cfgd} {exp}")
    for _ in range(N_UNIFORM):
        w = tuple(rnd.randrange(1 << 32) for _ in range(5))
        cfgd = rnd.choice([64, 128])
        exp = fmt.check2(*w, cfg_d=cfgd)
        hist2[exp] += 1
        L.append(f"C2 {w[0]:08x} {w[1]:08x} {w[2]:08x} {w[3]:08x} "
                 f"{w[4]:08x} {cfgd} {exp}")

    v1_legal = (64 | (8 << 8), 0, 0x3)
    for _ in range(N_MUTATED):
        g, r, m = v1_legal
        cfgd = 64
        for _ in range(rnd.randint(1, 3)):
            op = rnd.randrange(6)
            if op == 0:
                g = (g & 0x0FFF_FFFF) | (rnd.randrange(16) << 28)
            elif op == 1:
                g ^= 1 << rnd.randrange(32)
            elif op == 2:
                g = (g & ~(0x3 << 16)) | (rnd.randrange(4) << 16)
            elif op == 3:
                g = (g & ~0xFF00) | (rnd.randrange(256) << 8)
            elif op == 4:
                m = rnd.randrange(4)
            else:
                cfgd = 192 - cfgd
        exp = fmt.check1(g, r, m, cfgd)
        hist1[exp] += 1
        L.append(f"C1 {g:08x} {r:08x} {m:08x} {cfgd} {exp}")
    for _ in range(N_UNIFORM):
        g, r, m = (rnd.randrange(1 << 32) for _ in range(3))
        cfgd = rnd.choice([64, 128])
        exp = fmt.check1(g, r, m, cfgd)
        hist1[exp] += 1
        L.append(f"C1 {g:08x} {r:08x} {m:08x} {cfgd} {exp}")

    # coverage gate: every reachable code must actually be produced
    assert all(hist2[c] > 0 for c in (0, 1, 2)), f"check2 coverage: {hist2}"
    assert all(hist1[c] > 0 for c in (0, 1, 2)), f"check1 coverage: {hist1}"

    (out / "check_vectors.txt").write_text("\n".join(L) + "\n")
    print(f"check_vectors.txt: {len(L)} rows "
          f"(check2 code histogram {hist2}, check1 {hist1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
