#!/usr/bin/env python
"""gen_cores_vectors.py — FRESH clean-room vectors for the KVQ datapath cores
(verif/kvq/cores). Golden arbiter: golden/apex_golden/cq_codec.py — every
expected scale / payload / code / fp32 x_hat is produced by the golden model
here, NOT lifted from any pre-existing vendored vector set.

Emits one directory per config; each holds:
  cfg.txt   : "D TIER G K_OUT NV NG"
  mask.u8.hex : per-channel outlier mask (D lines), only if K_OUT>0
  val.txt   : NV value tokens — one line each:
                raw(D f16) | s(f16) | pay(PAY_BYTES bytes) | code(D bytes) | hat(D f32)
  key.txt   : NG key groups (INT4 grouped, §3/§4). Per group a header line
                "g full maskflag" then, all whitespace-separated one item/line:
                g×raw(Df16), 1×field(Df16 keep-scale/0), g×storecode(Dbytes),
                g×storefield(Df16), g×hat(Df32)
The testbench streams these into cq_value_path / cq_key_path and checks
bit-exact, covering D∈{64,128}, both INT tiers, full+partial(flush) groups,
and the -0.0 / +0.0 outlier identity lane (D-010 / B-4).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))
from apex_golden import cq_codec as cq                       # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f32_bits      # noqa: E402


def f16(x):
    return int(np.float64(x).astype(np.float16).view(np.uint16))


def widen_f16(u16):
    return int(f64_to_f32_bits(f16_bits_to_f64(np.array([u16], np.uint16)))[0])


class Gen:
    def __init__(self, name, D, TIER, G, K_OUT, seed, outdir):
        self.name, self.D, self.TIER, self.G, self.K_OUT = name, D, TIER, G, K_OUT
        self.bits = 8 if TIER == 0 else 4
        self.rng = np.random.default_rng(seed)
        self.out = Path(outdir); self.out.mkdir(parents=True, exist_ok=True)
        self.outlier = sorted(self.rng.choice(D, K_OUT, replace=False).tolist()) \
            if K_OUT > 0 else []
        self.is_out = np.zeros(D, bool); self.is_out[self.outlier] = True
        self.val_lines = []
        self.key_lines = []
        self.ng = 0

    def rand_tok(self, allow_zero=True):
        r = self.rng
        sign = r.integers(0, 2, self.D).astype(np.uint16) << 15
        expo = r.integers(0, 31, self.D).astype(np.uint16) << 10
        man = r.integers(0, 1024, self.D).astype(np.uint16)
        v = (sign | expo | man).astype(np.uint16)
        if allow_zero and r.random() < 0.2:
            v[r.integers(0, self.D)] = 0x0000
        return v

    def directed(self):
        D = self.D
        toks = []
        toks.append(np.zeros(D, np.uint16))                       # all zero
        sub = np.full(D, 0x0001, np.uint16); sub[1 % D] = 0x8001  # subnormals
        toks.append(sub)
        mx = np.full(D, f16(65504.0), np.uint16); mx[::2] |= 0x8000
        toks.append(mx)                                           # +/- maxnorm
        qm = float(cq.qmax_of(self.bits))
        tv = np.array([f16(v) for v in
                       ([qm, 2.5, 3.5, -2.5, -3.5, 0.5, 1.5, -0.5] * D)[:D]], np.uint16)
        tv[0] = f16(qm)                                           # exact ties
        toks.append(tv)
        na = self.rng.integers(0, 0x3C00, D).astype(np.uint16)
        na[D // 2] = f16(-1000.0)                                 # negative amax winner
        toks.append(na)
        return toks

    # ── value path ───────────────────────────────────────────────────────────
    def add_value(self, vec):
        s = int(cq.scale_from_amax(cq.amax_bits(vec), self.bits))
        codes = cq.quant_codes(vec, np.uint16(s), self.bits)
        pay = cq.pack_int4(codes) if self.bits == 4 else cq.pack_int8(codes)
        hat = cq.dequant_f32(codes, np.uint16(s))
        # positional (no separators): raw(D) s pay(PAY_BYTES) code(D) hat(D)
        raw_s = " ".join(f"{int(w):04x}" for w in vec)
        pay_s = " ".join(f"{int(b):02x}" for b in pay)
        code_s = " ".join(f"{int(c) & 0xFF:02x}" for c in codes)
        hat_s = " ".join(f"{int(h):08x}" for h in hat)
        self.val_lines.append(f"{raw_s} {s:04x} {pay_s} {code_s} {hat_s}")

    # ── key path (INT4 grouped) ──────────────────────────────────────────────
    def add_key_group(self, toks, full):
        g = len(toks); D = self.D
        mat = np.stack(toks)                                       # [g, D] u16
        field = np.zeros(D, np.uint16)                            # keep scale / 0
        code = np.zeros((g, D), np.int64)
        for c in range(D):
            if self.is_out[c]:
                continue
            s = cq.scale_from_amax(cq.amax_bits(mat[:, c]), 4)
            field[c] = s
            code[:, c] = cq.quant_codes(mat[:, c], s, 4)
        self.key_lines.append(str(g))
        self.key_lines.append(str(1 if full else 0))
        for t in range(g):                                        # raw tokens
            self.key_lines.append(" ".join(f"{int(w):04x}" for w in mat[t]))
        self.key_lines.append(" ".join(f"{int(field[c]):04x}" for c in range(D)))
        for t in range(g):                                        # storecode/token
            row = []
            for c in range(D):
                row.append(1 if self.is_out[c] else (int(code[t, c]) & 0xFF))
            self.key_lines.append(" ".join(f"{v:02x}" for v in row))
        for t in range(g):                                        # storefield/token
            row = []
            for c in range(D):
                row.append(int(mat[t, c]) if self.is_out[c] else int(field[c]))
            self.key_lines.append(" ".join(f"{v:04x}" for v in row))
        for t in range(g):                                        # hat/token
            row = []
            for c in range(D):
                if self.is_out[c]:
                    row.append(widen_f16(int(mat[t, c])))
                else:
                    row.append(int(cq.dequant_f32(np.array([code[t, c]]),
                                                  np.uint16(field[c]))[0]))
            self.key_lines.append(" ".join(f"{v:08x}" for v in row))
        self.ng += 1

    def build(self):
        for v in self.directed():
            self.add_value(v)
        for _ in range(20):
            self.add_value(self.rand_tok())
        nv = len(self.val_lines)
        if self.TIER != 0:
            G = self.G
            # full group
            self.add_key_group([self.rand_tok() for _ in range(G)], full=True)
            # partial groups via flush: g=1 and g=G-1
            self.add_key_group([self.rand_tok()], full=False)
            if G > 2:
                self.add_key_group([self.rand_tok() for _ in range(G - 1)], full=False)
            # directed-tie full group
            self.add_key_group(self.directed()[:G] if G <= 5 else
                               (self.directed() + [self.rand_tok()
                                for _ in range(G - 5)]), full=True)
            if self.K_OUT > 0:
                # -0.0 / +0.0 / neg-subnormal in the OUTLIER lanes (identity)
                t0 = self.rand_tok(allow_zero=False)
                t0[self.outlier[0]] = 0x8000                       # -0.0
                if len(self.outlier) > 1:
                    t0[self.outlier[1]] = 0x8001                  # -subnormal
                t1 = self.rand_tok(allow_zero=False)
                t1[self.outlier[0]] = 0x0000                       # +0.0
                grp = [t0, t1] + [self.rand_tok() for _ in range(self.G - 2)]
                self.add_key_group(grp, full=True)
        self.write(nv)

    def write(self, nv):
        (self.out / "cfg.txt").write_text(
            f"{self.D} {self.TIER} {self.G} {self.K_OUT} {nv} {self.ng}\n")
        (self.out / "val.txt").write_text("\n".join(self.val_lines) + "\n")
        if self.TIER != 0:
            (self.out / "key.txt").write_text("\n".join(self.key_lines) + "\n")
        if self.K_OUT > 0:
            with open(self.out / "mask.u8.hex", "w") as f:
                for c in range(self.D):
                    f.write(f"{1 if self.is_out[c] else 0:02x}\n")
        print(f"[{self.name}] D={self.D} tier={self.TIER} G={self.G} "
              f"K={self.K_OUT} NV={nv} NG={self.ng} outliers={self.outlier}")


CONFIGS = [
    # name          D    TIER  G   K   seed
    ("d64_cq4",     64,  1,    8,  0,  10001),
    ("d128_cq4",    128, 1,    8,  0,  10002),
    ("d64_cq8",     64,  0,    2,  0,  10003),
    ("d128_cq8",    128, 0,    2,  0,  10004),
    ("d64_cq4p",    64,  2,    4,  2,  10005),
    ("d128_cq4p",   128, 2,    8,  2,  10006),
]


def main(outroot):
    outroot = Path(outroot)
    for name, D, TIER, G, K, seed in CONFIGS:
        Gen(name, D, TIER, G, K, seed, outroot / name).build()
    print("CORES GEN OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
