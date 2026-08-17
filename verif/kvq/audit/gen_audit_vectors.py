#!/usr/bin/env python
"""gen_audit_vectors.py — adversarial re-audit vectors for the D-020-fixed KVQ
(verif/kvq/audit; attacks the B-1..B-4 fixes specifically; record images are
the A2/D-026 layout).

Golden arbiter: golden/apex_golden (cq_codec.py, D-001/D-013/D-026) — same
arbiter as verif/kvq/sb, fresh generator. Record/bank images come STRAIGHT
from the golden packers (cq_codec.pack_value_records / pack_key_records /
sram_row_bits — imported, never re-derived here):

  * KEY record (D-026): [tag {ssid[6:0],1'b1}][OUTLIER_K x fp16 lanes]
    [D x int4 codes, outlier ch -> sentinel 4'd1][pad64]. exp_rec.hex stores
    the image with ssid0=0 (tag 8'h01); the TB masks tag[7:1] on compare and
    checks the LIVE ssid against dut.alloc_ptr separately (ssid is a runtime
    commit-sequence property, not a pool property).
  * per-group SCALE BANK row -> exp_bank.f16.hex (one D x 16b row per key job,
    channel c at bits [16c +: 16]; outlier lanes pinned 16'h0000 by the golden
    packer).
  * VALUE record unchanged: [tag 8'h00][fp16 scale][D x BPV][pad64].

Emits, per configuration, a POOL of fully-precomputed clean jobs (value tokens
and key groups) with bit-exact §4/D-026 record images and fp32 readback
expectations. The TB (tb_kvq_audit.sv):

  * reset-attack mode (+mode=0): >= N randomized soft-reset injections across
    every reachable FSM state (hierarchical landing-state binning), spanning
    pending-beat / mid-token / mid-group / beat-coincident / 1-cycle-state
    collision conditions; EVERY reset is followed by a full clean job (value
    write + record peek + fp32 readback, all bit-exact vs this golden pool),
    every 3rd also by a full key-group clean job.
  * directed mode (+mode=1): runs every pool job once (used for the -0.0
    outlier / non-outlier directed content below).

Pool layout convention (shared with the TB — indices are computed, no script):
  tokens[0 .. NV-1]                : value jobs (ridx/hidx = i)
  tokens[NV .. NV+NK*G-1]          : key jobs (job k token j -> ridx NV+k*G+j)
  token  NV+NK*G                   : read-victim value job (ridx NV+NK*G)
  tokens NV+NK*G+1 .. +G           : victim tokens (no expectations; aborted)

Directed -0.0 content (attacks the B-4 fix + the non-outlier -0.0 INT4 path):
  * value jobs with -0.0 elements, an ALL -0.0 token (amax=0 -> EPS scale),
    +/-0.0 + subnormal mixes  (D=16 and D=64)
  * key groups with a whole non-outlier channel column of +/-0.0 (EPS scale,
    codes 0) and single -0.0 non-outlier hits           (D=16 and D=64)
  * CQ-4+ only: -0.0 / +0.0 / 0x8001 / -2048.0 in the OUTLIER lanes, incl.
    -0.0 outlier + -0.0 non-outlier in the SAME token   (D=16 and D=64)
All expectations come from the golden model: outlier lanes are the identity
fp16->fp32 widen (so -0.0 -> 0x8000_0000); non-outlier -0.0 quantizes to code
0 and reads back as code*scale (+0.0, 0x0000_0000).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))
from apex_golden import cq_codec as cq                    # noqa: E402


def f16(x: float) -> int:
    return int(np.float64(x).astype(np.float16).view(np.uint16))


class Cfg:
    def __init__(self, name, D, TIER, G, K_OUT, DEPTH, seed, outdir):
        self.name, self.D, self.TIER, self.G = name, D, TIER, G
        self.K_OUT, self.DEPTH = K_OUT, DEPTH
        self.bits = 8 if TIER == 0 else 4
        self.keyg = TIER != 0
        self.rng = np.random.default_rng(seed)
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        # D-026: row width comes from the golden packer — imported, NOT re-derived
        self.SRAMW = cq.sram_row_bits(D, self.bits, K_OUT, self.keyg)
        self.outlier = sorted(self.rng.choice(D, K_OUT, replace=False).tolist()) \
            if K_OUT > 0 else []
        self.is_out = np.zeros(D, bool)
        self.is_out[self.outlier] = True
        self.stim, self.recs, self.hats = [], [], []
        self.banks = []                      # D-026: one D x 16b bank row per key job
        self.cov = Counter()
        self.dsuf = f"_d{D}"

    def add_tok(self, vec) -> int:
        v = np.asarray(vec, np.uint16)
        assert v.shape == (self.D,)
        self.stim.append(v)
        return len(self.stim) - 1

    def rand_tok(self) -> int:
        r = self.rng
        sign = r.integers(0, 2, self.D).astype(np.uint16) << 15
        expo = r.integers(0, 31, self.D).astype(np.uint16) << 10
        man = r.integers(0, 1024, self.D).astype(np.uint16)
        v = sign | expo | man
        if r.random() < 0.15:
            v[r.integers(0, self.D)] = 0x0000
        return self.add_tok(v)

    # golden packers (D-026: cq_codec is the single source of truth) ----------
    def add_value_exp(self, tok):
        blob = cq.compress_values(
            np.asarray(self.stim[tok], np.uint16)[None, :], self.bits)
        rec = cq.pack_value_records(blob, self.SRAMW)[0]
        self.recs.append(int.from_bytes(rec.tobytes(), "little"))
        self.hats.append(cq.decompress_values(blob)[0])
        return len(self.recs) - 1

    def add_key_group_exp(self, toks):
        mat = np.stack([self.stim[t] for t in toks]).astype(np.uint16)
        blob = cq.compress_keys(mat, 4, self.G, self.outlier)
        # one group per call (g <= G): ssid0=0 so exp_rec tags read 8'h01; the
        # TB masks tag[7:1] and checks the live ssid against dut.alloc_ptr
        recs, bank, ssids = cq.pack_key_records(blob, self.SRAMW, ssid0=0)
        assert bank.shape[0] == 1 and ssids == [0], "one group per call"
        hats = cq.decompress_keys(blob)
        idxs = []
        for t in range(mat.shape[0]):
            self.recs.append(int.from_bytes(recs[t].tobytes(), "little"))
            self.hats.append(hats[t])
            idxs.append(len(self.recs) - 1)
        self.banks.append(bank[0])
        return idxs

    # directed -0.0 builders ---------------------------------------------------
    def negz_value_tokens(self):
        """Value jobs 0..2: -0.0 through the (INT4/INT8) value quant path."""
        D = self.rng, self.D
        r, D = D
        v = (r.integers(0x0400, 0x7000, D).astype(np.uint16))
        v[:: max(D // 6, 1)] = 0x8000                     # sprinkle -0.0
        self.add_tok(v)
        self.cov[f"negz_value{self.dsuf}"] += 1
        self.add_tok(np.full(D, 0x8000, np.uint16))       # ALL -0.0: amax=0
        self.cov[f"negz_value_all{self.dsuf}"] += 1
        m = np.zeros(D, np.uint16)
        m[0::4] = 0x8000
        m[1::4] = 0x0000
        m[2::4] = 0x8001                                  # neg subnormal
        m[3::4] = f16(1.0)
        self.add_tok(m)
        self.cov[f"negz_value_mix{self.dsuf}"] += 1

    def negz_key_tokens(self, n_groups):
        """First key jobs: -0.0 in non-outlier channels (and outlier lanes at
        CQ-4+). Returns list of token-index lists (one per group)."""
        D, G = self.D, self.G
        keep = [c for c in range(D) if not self.is_out[c]]
        groups = []
        # group 0: whole non-outlier channel column of +/-0.0 (EPS scale) plus
        # a single -0.0 hit in another channel of token 0
        toks = []
        for t in range(G):
            v = self.rng.integers(0x0400, 0x7000, D).astype(np.uint16)
            v[keep[0]] = 0x8000 if (t % 2 == 0) else 0x0000
            if t == 0:
                v[keep[1]] = 0x8000
            if self.K_OUT > 0:
                v[self.outlier[0]] = 0x8000 if t == 0 else 0x0000
                if self.K_OUT > 1:
                    v[self.outlier[1]] = [0x8001, f16(-2048.0), 0x8000,
                                          0x0001][t % 4]
            toks.append(self.add_tok(v))
        groups.append(toks)
        self.cov[f"negz_col_all_zero{self.dsuf}"] += 1
        self.cov[f"negz_key_nonoutlier{self.dsuf}"] += 1
        if self.K_OUT > 0:
            self.cov[f"negz_outlier{self.dsuf}"] += 1
            self.cov[f"posz_outlier{self.dsuf}"] += 1
            self.cov[f"negsub_outlier{self.dsuf}"] += 1
        for _ in range(n_groups - 1):
            toks = []
            for t in range(G):
                v = self.rng.integers(0x0400, 0x7000, D).astype(np.uint16)
                if self.rng.random() < 0.6:
                    v[int(self.rng.integers(0, D))] = 0x8000
                if self.K_OUT > 0 and self.rng.random() < 0.7:
                    v[self.outlier[int(self.rng.integers(0, self.K_OUT))]] = 0x8000
                    self.cov[f"negz_outlier{self.dsuf}"] += 1
                toks.append(self.add_tok(v))
            groups.append(toks)
        return groups

    def write_files(self, NV, NK):
        with open(self.out / "stim.f16.hex", "w") as f:
            for v in self.stim:
                for w in v:
                    f.write(f"{int(w):04x}\n")
        wid = self.SRAMW // 4
        with open(self.out / "exp_rec.hex", "w") as f:
            for r in self.recs:
                assert r < (1 << self.SRAMW)
                f.write(f"{r:0{wid}x}\n")
        with open(self.out / "exp_hat.f32.hex", "w") as f:
            for h in self.hats:
                for w in h:
                    f.write(f"{int(w):08x}\n")
        if self.banks:
            # D-026 bank rows: one D x 16b row per key job, ch c at [16c +: 16]
            with open(self.out / "exp_bank.f16.hex", "w") as f:
                for row in self.banks:
                    v = 0
                    for c, s in enumerate(row):
                        v |= int(s) << (16 * c)
                    f.write(f"{v:0{self.D * 4}x}\n")
        if self.K_OUT > 0:
            with open(self.out / "mask.u8.hex", "w") as f:
                for c in range(self.D):
                    f.write(f"{1 if self.is_out[c] else 0:02x}\n")
        with open(self.out / "cov_gen.txt", "w") as f:
            for k in sorted(self.cov):
                f.write(f"COVGEN {k} {self.cov[k]}\n")
        assert len(self.recs) == NV + NK * self.G + 1
        assert len(self.stim) == NV + NK * self.G + 1 + self.G
        assert len(self.banks) == (NK if self.keyg else 0)
        print(f"[{self.name}] NV={NV} NK={NK} toks={len(self.stim)} "
              f"recs={len(self.recs)} banks={len(self.banks)} "
              f"SRAMW={self.SRAMW} outliers={self.outlier}")


def build(name, D, TIER, G, K_OUT, DEPTH, NV, NK, seed, outroot):
    g = Cfg(name, D, TIER, G, K_OUT, DEPTH, seed, outroot / f"vec_{name}")
    # ---- value jobs: 3 directed -0.0 jobs first, rest random -----------------
    g.negz_value_tokens()
    for _ in range(NV - 3):
        g.rand_tok()
    for i in range(NV):
        g.add_value_exp(i)
    # ---- key jobs ------------------------------------------------------------
    if NK > 0:
        assert g.keyg
        n_directed = min(3, NK)
        groups = g.negz_key_tokens(n_directed)
        for _ in range(NK - n_directed):
            groups.append([g.rand_tok() for _ in range(G)])
        # tokens for key job k MUST be contiguous starting at NV + k*G
        for k, toks in enumerate(groups):
            assert toks == list(range(NV + k * G, NV + (k + 1) * G)), \
                f"key job {k} tokens not contiguous"
            g.add_key_group_exp(toks)
    # ---- read-victim job + victim tokens --------------------------------------
    rv = g.rand_tok()
    assert rv == NV + NK * G
    g.add_value_exp(rv)
    for _ in range(G):
        g.rand_tok()
    g.write_files(NV, NK)


def main(outroot):
    outroot = Path(outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    #      name        D  TIER G  K_OUT DEPTH NV  NK  seed
    build("rst_cq4",  16, 1,  4, 0,    64,   24,  8, 20260711, outroot)
    build("rst_cq4p", 16, 2,  4, 2,    64,   12,  8, 20260712, outroot)
    build("rst_d64",  64, 1,  8, 0,    32,    8,  4, 20260713, outroot)
    build("rst_cq8",  16, 0,  2, 0,    64,   12,  0, 20260714, outroot)
    build("negz_d64p", 64, 2, 8, 4,    32,    6,  4, 20260715, outroot)
    print("GEN OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
