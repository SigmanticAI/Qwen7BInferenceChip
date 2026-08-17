#!/usr/bin/env python
"""gen_kvq_sb_vectors.py — independent KVQ scoreboard vectors (verif/kvq/sb).

Golden arbiter: golden/apex_golden (cq_codec.py — V0-proven bit-exact vs
the frozen vendored vectors, ARCHITECTURE.md D-001/D-013). This generator
simulates the kvq_engine storage state (addr -> §4 byte-aligned record image)
and emits, per configuration:

  stim.f16.hex      all token vectors (tok-major, D fp16 words per token)
  exp_rec.hex       expected FULL SRAM record image per checked record (SRAMW bits)
  exp_hat.f32.hex   expected m_axis fp32 readback words per checked record
  exp_scalebank.hex expected persistent scale-bank rows (D x 16b, keyed cfgs only)
  script_main.txt   scoreboard transaction script (directed + seeded random)
  script_reset.txt  mid-operation reset choreography (cq4 config only)
  script_framing.txt malformed-framing robustness script (cq4 config only)
  cov_gen.txt       numeric-condition coverage counted here (COVGEN lines)
  mask.u8.hex       outlier mask ROM (cq4p / k5 configs only)

A2/D-026 KV-REC-DEDUP model (records + persistent scale bank): the record
images, bank rows and geometry come from the GOLDEN packers
apex_golden.cq_codec.{sram_row_bits, pack_value_records, pack_key_records,
compress_keys, decompress_keys} — never re-derived here.
  * KEY record: [tag {ssid[6:0],1'b1}][OUTLIER_K fp16 lanes][D x int4][pad64];
    ssid = per-engine group COMMIT sequence % SCALE_SETS since hard reset.
    This generator mirrors that allocator per SCRIPT (each script runs on a
    fresh DUT) in self.commit_seq[sc]; hard resets inside the reset script
    (X kind 0) reset the counter and zero the bank model, soft resets do NOT
    (D-020: bank rows persist with the records; aborted groups never commit).
  * A group's D x 16b scale row (outlier lanes forced 0) lands in the
    persistent bank set named by its records' ssid. With SCALE_SETS=4 a 5th+
    live group OVERWRITES the oldest set: later reads of the older records
    dequant with the NEW row (staleness is architectural, D-026). Expected
    hats for randomized/sweep reads are therefore computed LAZILY at R
    emission time from the live bank model (COVGEN stale_key_read counts the
    reads that hit an overwritten set). Reads emitted immediately after a
    commit keep their commit-time hats (set provably fresh).

Script alphabet (one command per line, parsed by tb_kvq_sb.sv):
  A <addr>                       write REG_WRITE_ADDR
  V <tok>                        stream value token <tok> (tuser=1), wait idle
  K <tok> <w>                    stream key token (tuser=0); w=1 wait idle after
                                 (group_last), w=2 no wait (KEMIT poke follows)
  F <m>                          flush_req: m=0 pulse now + wait idle,
                                 m=1 arm mid-token pulse for next K (partial),
                                 m=3 arm mid-token pulse for next K (full grp),
                                 m=2 bare pulse (no wait; no-op / KEMIT poke)
  G                              wait idle (used after K ... 2 / F 2)
  R <addr> <hidx>                read burst, compare D fp32 beats vs exp_hat[hidx]
  P <addr> <ridx>                peek SRAM record, compare vs exp_rec[ridx]
  B <set> <bidx>                 peek persistent scale bank set <set>
                                 (dut.g_key.u_bank_store.mem), compare the full
                                 D x 16b row vs exp_scalebank[bidx]
  Q <exp> <w1c>                  IRQ_STATUS[1] (SB_OVWR, D-026) must read <exp>;
                                 w1c=1 writes the W1C bit first
  E <addr> <mask>                read of UNWRITTEN addr: expect 0 beats, RD_ERR
                                 sticky + STATUS[1] + irq==(mask&1), then W1C
  O <n>                          REG_OCCUPANCY must read <n>
  S <b>                          STATUS[3] (sram_full) must read <b>
  X <kind> <phase> <tok> <vtok> <addr> <ridx> <hidx>
                                 mid-op reset test: kind 0=hard rst_n,
                                 1=soft parked, 2=soft colliding with ST_KFEED,
                                 3=soft colliding with ST_RLOAD; <tok> = op
                                 token (key phases use tok..tok+G-1), <vtok>
                                 recovery value token verified at <addr> vs
                                 exp_rec[ridx]/exp_hat[hidx]
  Y <tokbase> <addr> <r0> <h0>   stream a FULL key group immediately (post-soft-
                                 reset datapath-quiesce probe), check G records

Reachability proof (printed as COVGEN, used by coverage_report.py):
exhaustive sweep over ALL finite fp16 amax magnitudes shows rne(x/s) with the
engine-computed scale s never exceeds +/-qmax, so the quant saturation clamp
and qmin (-8/-128) are UNREACHABLE through the kvq_engine top for both tiers.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))
from apex_golden import cq_codec as cq                    # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f32_bits   # noqa: E402


def widen_f16(u16: int) -> int:
    """fp16 bits -> exact fp32 bits (outlier identity lane)."""
    return int(f64_to_f32_bits(f16_bits_to_f64(np.array([u16], np.uint16)))[0])


def f16(x: float) -> int:
    return int(np.float64(x).astype(np.float16).view(np.uint16))


class ConfigGen:
    def __init__(self, name, D, TIER, G, K_OUT, DEPTH, seed, outdir):
        self.name, self.D, self.TIER, self.G = name, D, TIER, G
        self.K_OUT, self.DEPTH = K_OUT, DEPTH
        self.bits = 8 if TIER == 0 else 4
        self.keyg = TIER != 0
        self.rng = np.random.default_rng(seed)
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        # §4/D-026 geometry straight from the golden packer (single source of
        # truth — kvq_engine.sv REC_RAW/SRAM_WIDTH mirror the same function)
        self.SRAMW = cq.sram_row_bits(D, self.bits, K_OUT, self.keyg)
        self.outlier = sorted(self.rng.choice(D, K_OUT, replace=False).tolist()) \
            if K_OUT > 0 else []
        self.is_out = np.zeros(D, bool)
        self.is_out[self.outlier] = True
        self.stim = []          # list of np.uint16[D]
        self.recs = []          # ints (SRAMW-bit record images)
        self.hats = []          # list of np.uint32[D]
        self.banks = []         # ints (D*16-bit expected bank rows, B targets)
        self.lines = {"main": [], "reset": [], "framing": []}
        self.written = set()    # addrs with valid records (soft state only)
        # D-026 persistent-bank model, per script (each script = fresh DUT):
        # commit_seq mirrors the engine's commit-time ssid allocator (wraps
        # mod SCALE_SETS, reset only by hard reset); bank_model holds the
        # current row per set; live maps addr -> record entry for lazy
        # (stale-aware) readback hats.
        self.commit_seq = Counter()
        self.bank_model = {s: [np.zeros(D, np.uint16)
                               for _ in range(cq.SCALE_SETS)]
                           for s in ("main", "reset", "framing")}
        self.live = {s: {} for s in ("main", "reset", "framing")}
        self.cov = Counter()
        self.txn = Counter()

    # ── token bookkeeping ────────────────────────────────────────────────────
    def add_tok(self, vec) -> int:
        v = np.asarray(vec, np.uint16)
        assert v.shape == (self.D,)
        self.stim.append(v)
        return len(self.stim) - 1

    def rand_tok(self) -> int:
        """random finite fp16: exponent 0..30, any sign/mantissa; sprinkle
        exact zeros and repeats so degenerate patterns appear."""
        r = self.rng
        sign = r.integers(0, 2, self.D).astype(np.uint16) << 15
        expo = r.integers(0, 31, self.D).astype(np.uint16) << 10
        man = r.integers(0, 1024, self.D).astype(np.uint16)
        v = sign | expo | man
        if r.random() < 0.15:
            v[r.integers(0, self.D)] = 0x0000
        if r.random() < 0.10:
            v[:] = v[0]
        return self.add_tok(v)

    # ── golden math + numeric coverage ───────────────────────────────────────
    def _cov_numeric(self, x_f16, s_u16):
        x = f16_bits_to_f64(x_f16)
        s = float(f16_bits_to_f64(np.array([s_u16]))[0])
        q = x / s
        frac = np.abs(q - np.floor(q))
        self.cov["tie_hits"] += int(np.sum(frac == 0.5))
        if s == cq.EPS:
            self.cov["eps_clamp"] += 1

    def compress_value(self, vec):
        s = cq.scale_from_amax(cq.amax_bits(vec), self.bits)
        codes = cq.quant_codes(vec, s, self.bits)
        self._cov_numeric(vec, s)
        amax_pos = int(np.argmax(np.asarray(vec, np.uint16) & 0x7FFF))
        if vec[amax_pos] & 0x8000:
            self.cov["neg_amax_winner"] += 1
        return s, codes

    def val_rec_image(self, s, codes) -> int:
        """§4 value record image via the GOLDEN packer (unchanged by D-026)."""
        b = cq.ValueBlob(T=1, D=self.D, bits=self.bits,
                         scales=np.array([s], np.uint16),
                         codes=np.asarray(codes, np.int64).reshape(1, self.D))
        b.payload = (cq.pack_int4(b.codes.reshape(-1)) if self.bits == 4
                     else cq.pack_int8(b.codes.reshape(-1)))
        row = cq.pack_value_records(b, self.SRAMW)[0]
        return int.from_bytes(row.tobytes(), "little")

    def val_hat(self, s, codes):
        return cq.dequant_f32(codes, np.uint16(s))

    # ── script emission helpers ──────────────────────────────────────────────
    def emit(self, sc, line):
        self.lines[sc].append(line)

    def value_write(self, sc, addr, tok, check=True):
        """A addr; V tok; optional P+R. Returns (ridx, hidx)."""
        s, codes = self.compress_value(self.stim[tok])
        self.recs.append(self.val_rec_image(s, codes))
        self.hats.append(self.val_hat(s, codes))
        # recs and hats grow at DIFFERENT rates (emit_read appends lazy
        # stale-aware hats with no record) — never conflate the two indices,
        # or an overwrite after the first lazy read binds the live entry to
        # a stale hat row (the old exp_hat at len(recs)-1, key-typed).
        ridx = len(self.recs) - 1
        hidx = len(self.hats) - 1
        self.live[sc][addr] = ("V", hidx)   # value hats never go stale
        self.emit(sc, f"A {addr}")
        self.emit(sc, f"V {tok}")
        self.txn["V"] += 1
        if addr in self.written:
            self.cov["overwrite"] += 1
        self.written.add(addr)
        if check:
            self.emit(sc, f"P {addr} {ridx}")
            self.emit(sc, f"R {addr} {hidx}")
            self.txn["R"] += 1
        return ridx, hidx

    def key_group_exp(self, sc, toks):
        """Golden D-026 packing of ONE committed key group: record images via
        cq.pack_key_records (tag = {ssid,1} with ssid = this script's commit
        sequence % SCALE_SETS — commit-time allocation, mirroring the engine),
        fresh readback hats via cq.decompress_keys, and the group's D x 16b
        scale row into the persistent bank model. Returns a per-token list of
        (rec_image, hat, live_entry) plus the committed ssid."""
        g = len(toks)
        mat = np.stack([self.stim[t] for t in toks]).astype(np.uint16)
        blob = cq.compress_keys(mat, 4, 0, self.outlier)   # G=0: one group of g
        for ci, c in enumerate(blob.keep):                 # numeric coverage
            self._cov_numeric(mat[:, c], np.uint16(blob.scales[ci]))
        ssid = self.commit_seq[sc] % cq.SCALE_SETS
        recs_u8, bank, ssids = cq.pack_key_records(blob, self.SRAMW, ssid0=ssid)
        assert list(ssids) == [ssid]
        self.commit_seq[sc] += 1
        self.bank_model[sc][ssid] = bank[0].copy()
        hats = cq.decompress_keys(blob)                    # [g, D] fp32 bits
        nk = len(blob.keep)
        codes = cq.unpack_int4(blob.payload, g * nk).reshape(g, nk)
        out = []
        for t in range(g):
            full = np.zeros(self.D, np.int64)
            full[blob.keep] = codes[t]
            rec = int.from_bytes(recs_u8[t].tobytes(), "little")
            # live entry: keep codes + raw fp16 row (outlier lanes) + ssid +
            # a snapshot of the commit-time bank row + the fresh golden hat
            # (fresh-read cross-check in emit_read)
            ent = ("K", full, mat[t].copy(), ssid, bank[0].copy(),
                   hats[t].astype(np.uint32))
            out.append((rec, hats[t].astype(np.uint32), ent))
        return out, ssid

    def emit_bank_check(self, sc, ssid):
        """B command: compare the DUT's persistent bank set vs the model row."""
        row = self.bank_model[sc][ssid]
        val = 0
        for c in range(self.D):
            val |= int(row[c]) << (16 * c)
        self.banks.append(val)
        self.emit(sc, f"B {ssid} {len(self.banks) - 1}")

    def bank_sweep(self, sc):
        """End-of-script B sweep over ALL sets (never-committed sets must
        still read as the hard-reset zeros — the model rows match)."""
        for s_i in range(cq.SCALE_SETS):
            self.emit_bank_check(sc, s_i)

    def emit_read(self, sc, addr):
        """R command with a hat computed against the CURRENT bank state:
        value records keep their commit-time hat; key records dequant their
        stored codes with the row NOW in their ssid's set (D-026 staleness —
        outlier lanes replay identity from the record and never go stale)."""
        ent = self.live[sc][addr]
        if ent[0] == "V":
            hidx = ent[1]
        else:
            _, full, raw, ssid, row0, hat0 = ent
            row = self.bank_model[sc][ssid]
            stale = not np.array_equal(row, row0)
            if stale:
                self.cov["stale_key_read"] += 1
            hat = np.zeros(self.D, np.uint32)
            for c in range(self.D):
                if self.is_out[c]:
                    hat[c] = widen_f16(int(raw[c]))
                else:
                    hat[c] = cq.dequant_f32(full[c:c + 1],
                                            np.uint16(row[c]))[0]
            # self-check: with the commit-time row still in place, the lazy
            # per-channel dequant must reproduce cq.decompress_keys exactly
            assert stale or np.array_equal(hat, hat0), \
                f"lazy hat model drifted from decompress_keys at addr {addr}"
            self.hats.append(hat)
            hidx = len(self.hats) - 1
        self.emit(sc, f"R {addr} {hidx}")
        self.txn["R"] += 1

    def key_group_write(self, sc, base, toks, flush=None, check=True):
        """Stream a key group. flush: None=full G group; 'tail'=pulse in
        KACCEPT after last tok; 'mid'=arm mid-token flush on last tok;
        'midfull'=arm mid-token flush on the G-th token (group completes FULL,
        flush_pending must be discarded)."""
        g = len(toks)
        self.emit(sc, f"A {base}")
        if flush is None or flush == "midfull":
            assert g == self.G
        else:
            assert g < self.G
        for i, t in enumerate(toks):
            last_of_group = (i == g - 1) and (g == self.G)
            if flush in ("mid", "midfull") and i == g - 1:
                self.emit(sc, f"F {1 if flush == 'mid' else 3}")
            if flush in ("mid",) and i == g - 1:
                w = 1                      # emit follows this token's KFEED
            else:
                w = 1 if last_of_group else 0
            self.emit(sc, f"K {t} {w}")
            self.txn["K"] += 1
        if flush == "tail":
            self.emit(sc, "F 0")
        exp, ssid = self.key_group_exp(sc, toks)
        idxs = []
        for i, (rec, hat, ent) in enumerate(exp):
            self.recs.append(rec)
            self.hats.append(hat)
            ri = len(self.recs) - 1
            hi = len(self.hats) - 1   # hats index != recs index (lazy appends)
            idxs.append(ri)
            self.live[sc][base + i] = ent
            if (base + i) in self.written:
                self.cov["overwrite"] += 1
            self.written.add(base + i)
            if check:
                # immediate R: the set was committed THIS group — fresh hat
                self.emit(sc, f"P {base + i} {ri}")
                self.emit(sc, f"R {base + i} {hi}")
                self.txn["R"] += 1
        if check:
            self.emit_bank_check(sc, ssid)   # score the committed bank row
        if flush is None:
            self.cov["k_group_full"] += 1
        elif flush == "midfull":
            self.cov["flush_on_full"] += 1
        elif flush == "mid":
            self.cov["flush_midtoken"] += 1
        elif g == 1:
            self.cov["k_flush_g1"] += 1
        elif g == self.G - 1:
            self.cov["k_flush_gm1"] += 1
        else:
            self.cov["k_flush_mid"] += 1
        return idxs

    # ── file output ──────────────────────────────────────────────────────────
    def write_files(self):
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
            bwid = self.D * 16 // 4
            with open(self.out / "exp_scalebank.hex", "w") as f:
                for r in self.banks:
                    assert r < (1 << (self.D * 16))
                    f.write(f"{r:0{bwid}x}\n")
        for sc, lines in self.lines.items():
            if lines:
                with open(self.out / f"script_{sc}.txt", "w") as f:
                    f.write(f"# {self.name} {sc}: generated by gen_kvq_sb_vectors.py\n")
                    f.write("\n".join(lines) + "\n")
        if self.K_OUT > 0:
            with open(self.out / "mask.u8.hex", "w") as f:
                for c in range(self.D):
                    f.write(f"{1 if self.is_out[c] else 0:02x}\n")
        with open(self.out / "cov_gen.txt", "w") as f:
            for k in sorted(self.cov):
                f.write(f"COVGEN {k} {self.cov[k]}\n")
        print(f"[{self.name}] toks={len(self.stim)} recs={len(self.recs)} "
              f"banks={len(self.banks)} "
              f"txn: V={self.txn['V']} K={self.txn['K']} R={self.txn['R']} "
              f"SRAMW={self.SRAMW} outliers={self.outlier}")
        return dict(self.txn)


# ── directed token builders ───────────────────────────────────────────────────

def directed_tokens(g: ConfigGen):
    D = g.D
    toks = {}
    toks["zero"] = g.add_tok(np.zeros(D, np.uint16))
    g.cov["zero_tok"] += 1
    sub = np.full(D, 0x0001, np.uint16)     # smallest positive subnormal
    sub[1] = 0x8001                         # and its negation
    toks["subnormal"] = g.add_tok(sub)
    g.cov["subnormal_tok"] += 1
    mx = np.full(D, f16(65504.0), np.uint16)
    mx[::2] |= 0x8000
    toks["maxnorm"] = g.add_tok(mx)
    g.cov["maxnorm_tok"] += 1
    # exact RNE tie token: amax = qmax exactly -> scale = 1.0 exactly
    qm = float(cq.qmax_of(g.bits))
    tie_vals = [qm, 2.5, 3.5, -2.5, -3.5, 0.5, 1.5, -0.5, 4.5, -4.5, 5.5, 6.5]
    tv = np.array([f16(tie_vals[i % len(tie_vals)]) for i in range(D)], np.uint16)
    tv[0] = f16(qm)
    toks["ties"] = g.add_tok(tv)
    # negative amax winner
    na = (g.rng.integers(0, 0x3C00, D).astype(np.uint16))
    na[D // 2] = f16(-1000.0)
    toks["negamax"] = g.add_tok(na)
    return toks


def build_cq4(outroot):
    g = ConfigGen("cq4", D=16, TIER=1, G=4, K_OUT=0, DEPTH=64, seed=20260707,
                  outdir=outroot / "vec_cq4")
    sc = "main"
    t = directed_tokens(g)
    # directed values (addr 0..4)
    for i, name in enumerate(["zero", "subnormal", "maxnorm", "ties", "negamax"]):
        g.value_write(sc, i, t[name])
    g.emit(sc, "O 5")
    g.cov["occ_check"] += 1
    # overwrite addr 0 with a random token; occupancy must not double-count
    g.value_write(sc, 0, g.rand_tok())
    g.emit(sc, "O 5")
    g.cov["occ_check"] += 1
    # flush no-op while idle
    g.emit(sc, "F 2")
    g.emit(sc, "G")
    g.cov["flush_noop_idle"] += 1
    # directed key groups
    g.key_group_write(sc, 8, [g.rand_tok() for _ in range(4)])              # full
    g.key_group_write(sc, 12, [g.rand_tok()], flush="tail")                 # g=1
    g.key_group_write(sc, 14, [g.rand_tok(), g.rand_tok()], flush="tail")   # g=2
    g.key_group_write(sc, 16, [g.rand_tok() for _ in range(3)], flush="tail")  # g=G-1
    # D-026 SB_OVWR choreography: 4 commits so far -> all sets fresh, bit clear
    g.emit(sc, "Q 0 0")
    g.key_group_write(sc, 20, [g.rand_tok(), g.rand_tok()], flush="mid")    # mid-token
    # 5th commit reused set 0 (still live) -> sticky SB_OVWR; then W1C clears
    g.emit(sc, "Q 1 0")
    g.emit(sc, "Q 0 1")
    g.key_group_write(sc, 24, [g.rand_tok() for _ in range(4)], flush="midfull")
    # 6th commit reused set 1 -> SB_OVWR re-raised (sticky re-arm proof)
    g.emit(sc, "Q 1 0")
    # flush pulse during KEMIT must be a no-op (group already frozen)
    base = 28
    toks = [g.rand_tok() for _ in range(4)]
    g.emit(sc, f"A {base}")
    for i, tk in enumerate(toks):
        g.emit(sc, f"K {tk} {2 if i == 3 else 0}")
        g.txn["K"] += 1
    g.emit(sc, "F 2")     # engine is in ST_KEMIT (D-cycle quant walk per token)
    g.emit(sc, "G")
    g.cov["flush_during_emit"] += 1
    exp, ssid = g.key_group_exp(sc, toks)
    for i, (rec, hat, ent) in enumerate(exp):
        g.recs.append(rec)
        g.hats.append(hat)
        ri = len(g.recs) - 1
        hi = len(g.hats) - 1
        g.live[sc][base + i] = ent
        g.written.add(base + i)
        g.emit(sc, f"P {base + i} {ri}")
        g.emit(sc, f"R {base + i} {hi}")
        g.txn["R"] += 1
    g.emit_bank_check(sc, ssid)
    # RD_ERR flows: masked then unmasked
    g.emit(sc, "E 63 1")
    g.cov["rderr"] += 1
    g.cov["w1c"] += 1
    g.emit(sc, "E 62 0")
    g.cov["rderr"] += 1
    g.cov["w1c"] += 1
    g.cov["irq_masked"] += 1
    # key record overwritten by a value record (tag flip at same addr)
    g.value_write(sc, 8, g.rand_tok())
    # ── seeded randomized phase ──────────────────────────────────────────────
    latest = {}
    for _ in range(80):
        op = g.rng.random()
        if op < 0.55:
            a = int(g.rng.integers(0, 56))
            r, h = g.value_write(sc, a, g.rand_tok(), check=False)
            latest[a] = r
            g.cov["rand_v"] += 1
        elif op < 0.80:
            a = int(g.rng.integers(0, 52))
            idxs = g.key_group_write(sc, a, [g.rand_tok() for _ in range(4)],
                                     check=False)
            for i, ri in enumerate(idxs):
                latest[a + i] = ri
            g.cov["rand_k"] += 1
        else:
            gsz = int(g.rng.integers(1, 4))
            a = int(g.rng.integers(0, 52))
            idxs = g.key_group_write(sc, a,
                                     [g.rand_tok() for _ in range(gsz)],
                                     flush="tail", check=False)
            for i, ri in enumerate(idxs):
                latest[a + i] = ri
            g.cov["rand_k"] += 1
        if g.rng.random() < 0.5 and latest:
            a = int(g.rng.choice(sorted(latest)))
            g.emit_read(sc, a)          # hat vs the CURRENT bank row (D-026)
    # final sweep: check EVERY live address (records must persist; key hats
    # score against whatever row their ssid's set holds NOW — staleness model)
    for a in sorted(latest):
        g.emit(sc, f"P {a} {latest[a]}")
        g.emit_read(sc, a)
    # fill the SRAM completely and observe sram_full via STATUS[3]
    for a in range(g.DEPTH):
        if a not in g.written:
            g.value_write(sc, a, g.rand_tok(), check=False)
    g.emit(sc, f"O {g.DEPTH}")
    g.cov["occ_check"] += 1
    g.emit(sc, "S 1")
    g.cov["sram_full_seen"] += 1
    # persistent-bank end state: all 4 sets vs the model
    g.bank_sweep(sc)

    # ── reset script (fresh SRAM state per run; reuses stim/exp arrays) ─────
    sc = "reset"
    # value tokens for recovery round-trips: one per test, distinct addrs
    def xline(kind, phase, optok, addr):
        if kind == 0:
            # hard reset: the engine's ssid allocator, set_live and bank rows
            # all clear (rst_n) — mirror that. Soft resets (kind 1/2/3/4) do
            # NOT touch the model: aborted groups never commit (no ssid), and
            # committed bank rows persist (D-020/D-026).
            g.commit_seq[sc] = 0
            g.bank_model[sc] = [np.zeros(g.D, np.uint16)
                                for _ in range(cq.SCALE_SETS)]
        vtok = g.rand_tok()
        s, codes = g.compress_value(g.stim[vtok])
        g.recs.append(g.val_rec_image(s, codes))
        g.hats.append(g.val_hat(s, codes))
        ri = len(g.recs) - 1
        hi = len(g.hats) - 1
        g.emit(sc, f"X {kind} {phase} {optok} {vtok} {addr} {ri} {hi}")
    vt = g.rand_tok()
    kt = [g.rand_tok() for _ in range(4)]   # key tokens (base kt[0], contiguous)
    assert kt == list(range(kt[0], kt[0] + 4))
    for phase in range(1, 12):              # hard reset: every FSM phase
        xline(0, phase, kt[0] if phase >= 7 else vt, 2 * phase)
    for phase in (1, 2, 6, 7, 9, 11):       # soft reset, parked states
        xline(1, phase, kt[0] if phase >= 7 else vt, 30 + phase)
    xline(2, 8, kt[0], 56)                  # soft reset colliding with ST_KFEED
    xline(3, 4, vt, 58)                     # soft reset colliding with ST_RLOAD
    # LAST (may wedge the DUT; the TB owns a bounded-wait + hard-reset rescue):
    # soft reset mid-KEMIT, then IMMEDIATELY a new full key group — probes that
    # a soft reset leaves the (un-reset) cq_key_path datapath mid-walk with no
    # host-visible busy indication.
    xline(4, 10, kt[0], 45)
    ytoks = [g.rand_tok() for _ in range(4)]
    assert ytoks == list(range(ytoks[0], ytoks[0] + 4))
    # the kind-4 group above was ABORTED mid-KEMIT (no commit, no ssid), so
    # the Y group is the FIRST commit since the last hard reset -> ssid 0
    assert g.commit_seq[sc] == 0
    yexp, yssid = g.key_group_exp(sc, ytoks)
    assert yssid == 0
    r0 = len(g.recs)
    h0 = len(g.hats)
    for rec, hat, _ent in yexp:
        g.recs.append(rec)
        g.hats.append(hat)
    g.emit(sc, f"Y {ytoks[0]} 50 {r0} {h0}")
    # the Y commit must have (re)written bank set 0 — score the row
    g.emit_bank_check(sc, yssid)

    # ── framing script ───────────────────────────────────────────────────────
    sc = "framing"
    for mode, name in ((1, "framing_early"), (2, "framing_notlast"),
                       (3, "framing_late")):
        tok = g.rand_tok()
        s, codes = g.compress_value(g.stim[tok])
        g.recs.append(g.val_rec_image(s, codes))
        g.hats.append(g.val_hat(s, codes))
        ri = len(g.recs) - 1
        hi = len(g.hats) - 1
        g.emit(sc, f"M {tok} {mode} {mode * 2} {ri} {hi}")
        g.cov[name] += 1
    # a clean value op at the end proves the engine is still fully usable
    tok = g.rand_tok()
    s, codes = g.compress_value(g.stim[tok])
    g.recs.append(g.val_rec_image(s, codes))
    g.hats.append(g.val_hat(s, codes))
    ri = len(g.recs) - 1
    hi = len(g.hats) - 1
    g.emit(sc, "A 20")
    g.emit(sc, f"V {tok}")
    g.emit(sc, f"P 20 {ri}")
    g.emit(sc, f"R 20 {hi}")
    return g.write_files()


def build_cq8(outroot):
    g = ConfigGen("cq8", D=16, TIER=0, G=2, K_OUT=0, DEPTH=64, seed=20260708,
                  outdir=outroot / "vec_cq8")
    sc = "main"
    t = directed_tokens(g)
    for i, name in enumerate(["zero", "subnormal", "maxnorm", "ties", "negamax"]):
        g.value_write(sc, i, t[name])
    latest = {}
    for _ in range(120):
        a = int(g.rng.integers(0, g.DEPTH))
        r, h = g.value_write(sc, a, g.rand_tok(), check=False)
        latest[a] = (r, h)
        g.cov["rand_v"] += 1
        if g.rng.random() < 0.6:
            g.emit(sc, f"R {a} {h}")
            g.txn["R"] += 1
    # TIER-0 "keys": tuser=0 tokens take the per-token value path (KEYG=0);
    # stored as value-format records. Streamed via K to exercise tuser=0.
    for _ in range(60):
        a = int(g.rng.integers(0, g.DEPTH))
        tok = g.rand_tok()
        s, codes = g.compress_value(g.stim[tok])
        g.recs.append(g.val_rec_image(s, codes))
        g.hats.append(g.val_hat(s, codes))
        ri = len(g.recs) - 1
        hi = len(g.hats) - 1
        g.emit(sc, f"A {a}")
        g.emit(sc, f"K {tok} 1")
        g.txn["K"] += 1
        if a in g.written:
            g.cov["overwrite"] += 1
        g.written.add(a)
        latest[a] = (ri, hi)
        g.cov["rand_k"] += 1
    for a in sorted(latest):
        g.emit(sc, f"P {a} {latest[a][0]}")
        g.emit(sc, f"R {a} {latest[a][1]}")
        g.txn["R"] += 1
    g.emit(sc, "E 63 1") if 63 not in g.written else g.emit(sc, "O %d" % len(g.written))
    if 63 not in g.written:
        g.cov["rderr"] += 1
        g.cov["w1c"] += 1
    return g.write_files()


def build_cq4p(outroot):
    g = ConfigGen("cq4p", D=16, TIER=2, G=4, K_OUT=2, DEPTH=64, seed=20260709,
                  outdir=outroot / "vec_cq4p")
    sc = "main"
    t = directed_tokens(g)
    for i, name in enumerate(["zero", "maxnorm", "ties"]):
        g.value_write(sc, i, t[name])
    # directed outlier corners: -0.0, +0.0, negative subnormal in the outlier
    # lanes (identity FP16 replay, D-010 — a -0.0 outlier must read back -0.0)
    oc = np.zeros(16, np.uint16)
    oc[:] = g.rng.integers(0x0400, 0x3C00, 16).astype(np.uint16)
    oc[g.outlier[0]] = 0x8000    # -0.0
    oc[g.outlier[1]] = 0x8001    # smallest negative subnormal
    tok_negz = g.add_tok(oc)
    g.cov["negzero_outlier"] += 1
    oc2 = oc.copy()
    oc2[g.outlier[0]] = 0x0000   # +0.0
    oc2[g.outlier[1]] = f16(-2048.0)
    tok_posz = g.add_tok(oc2)
    grp = [tok_negz, tok_posz, g.rand_tok(), g.rand_tok()]
    g.key_group_write(sc, 8, grp)
    latest = {}
    for _ in range(9):
        a = int(g.rng.integers(0, 52))
        if g.rng.random() < 0.5:
            idxs = g.key_group_write(sc, a, [g.rand_tok() for _ in range(4)],
                                     check=False)
        else:
            gsz = int(g.rng.integers(1, 4))
            idxs = g.key_group_write(sc, a, [g.rand_tok() for _ in range(gsz)],
                                     flush="tail", check=False)
        for i, ri in enumerate(idxs):
            latest[a + i] = ri
        g.cov["rand_k"] += 1
    for _ in range(20):
        a = int(g.rng.integers(0, 52))
        r, h = g.value_write(sc, a, g.rand_tok(), check=False)
        latest[a] = r
        g.cov["rand_v"] += 1
    for a in sorted(latest):
        g.emit(sc, f"P {a} {latest[a]}")
        g.emit_read(sc, a)              # stale-aware (10 commits, 4 sets)
    g.bank_sweep(sc)
    return g.write_files()


def build_d64(outroot):
    g = ConfigGen("d64", D=64, TIER=1, G=8, K_OUT=0, DEPTH=32, seed=20260710,
                  outdir=outroot / "vec_d64")
    sc = "main"
    t = directed_tokens(g)
    for i, name in enumerate(["zero", "ties", "maxnorm"]):
        g.value_write(sc, i, t[name])
    latest = {}
    for _ in range(8):
        a = int(g.rng.integers(0, 8))
        r, h = g.value_write(sc, a, g.rand_tok(), check=False)
        latest[a] = r
        g.cov["rand_v"] += 1
    g.key_group_write(sc, 8, [g.rand_tok() for _ in range(8)])       # full G=8
    g.key_group_write(sc, 16, [g.rand_tok() for _ in range(3)],
                      flush="tail")                                  # partial
    g.key_group_write(sc, 20, [g.rand_tok() for _ in range(7)],
                      flush="tail")                                  # g = G-1
    for a in sorted(latest):
        g.emit(sc, f"P {a} {latest[a]}")
        g.emit_read(sc, a)              # values only in latest here
    g.emit(sc, "E 31 1")
    g.cov["rderr"] += 1
    g.cov["w1c"] += 1
    g.bank_sweep(sc)
    return g.write_files()


def build_k5(outroot):
    """A2/D-026 k=5 point (the 6.125 b/v tier at shipped G=128): K_OUT=5 at
    D=64 gives the 5-outlier-lane record geometry (KEY_REC_RAW = 8+80+256 =
    344 -> 384b row, dominating VAL_RAW 280) RTL coverage; G=8 keeps sim time
    bounded. The 5-outlier mask hex is generated here (write_files)."""
    g = ConfigGen("k5", D=64, TIER=2, G=8, K_OUT=5, DEPTH=32, seed=20260714,
                  outdir=outroot / "vec_k5")
    sc = "main"
    t = directed_tokens(g)
    for i, name in enumerate(["zero", "maxnorm", "ties"]):
        g.value_write(sc, i, t[name])
    # outlier corners spread across ALL 5 lanes: -0.0, +/- smallest subnormal,
    # a large negative, +0.0 (identity FP16 replay incl. sign, D-010)
    oc = g.rng.integers(0x0400, 0x3C00, 64).astype(np.uint16)
    oc[g.outlier[0]] = 0x8000    # -0.0
    oc[g.outlier[1]] = 0x8001    # smallest negative subnormal
    oc[g.outlier[2]] = 0x0001    # smallest positive subnormal
    oc[g.outlier[3]] = f16(-2048.0)
    oc[g.outlier[4]] = 0x0000    # +0.0
    tok_oc = g.add_tok(oc)
    g.cov["negzero_outlier"] += 1
    g.cov["outlier_lanes5"] += 1
    grp = [tok_oc] + [g.rand_tok() for _ in range(7)]
    g.key_group_write(sc, 8, grp)                                    # full G=8
    g.key_group_write(sc, 16, [g.rand_tok() for _ in range(3)],
                      flush="tail")                                  # partial
    g.key_group_write(sc, 20, [g.rand_tok() for _ in range(7)],
                      flush="tail")                                  # g = G-1
    latest = {}
    for _ in range(6):
        a = int(g.rng.integers(0, 24))
        if g.rng.random() < 0.5:
            idxs = g.key_group_write(sc, a, [g.rand_tok() for _ in range(8)],
                                     check=False)
        else:
            gsz = int(g.rng.integers(1, 8))
            idxs = g.key_group_write(sc, a,
                                     [g.rand_tok() for _ in range(gsz)],
                                     flush="tail", check=False)
        for i, ri in enumerate(idxs):
            latest[a + i] = ri
        g.cov["rand_k"] += 1
    for _ in range(10):
        a = int(g.rng.integers(0, 28))
        r, h = g.value_write(sc, a, g.rand_tok(), check=False)
        latest[a] = r
        g.cov["rand_v"] += 1
    for a in sorted(latest):
        g.emit(sc, f"P {a} {latest[a]}")
        g.emit_read(sc, a)              # stale-aware (9 commits, 4 sets)
    g.emit(sc, "E 31 1")
    g.cov["rderr"] += 1
    g.cov["w1c"] += 1
    g.bank_sweep(sc)
    return g.write_files()


def reachability_proof():
    """Exhaustive fp16 sweep: quant clamp / qmin unreachable through the top."""
    notes = Counter()
    for bits in (4, 8):
        qmax = cq.qmax_of(bits)
        mags = np.arange(1, 0x7C00, dtype=np.uint16)
        v = mags.view(np.float16).astype(np.float64)
        s = np.maximum(v / qmax, cq.EPS).astype(np.float16).astype(np.float64)
        q = np.rint(v / s)
        assert q.max() <= qmax and (-q).min() >= -qmax, \
            f"reachability sweep violated at bits={bits}"
    notes["satpos_unreachable_proof"] = 1
    notes["qmin_unreachable_proof"] = 1
    print("REACHABILITY: exhaustive fp16 amax sweep (2x31743 values): "
          "rne(x/s) never exceeds +/-qmax with engine-computed scales -> "
          "saturation clamp and qmin are UNREACHABLE through kvq_engine")
    return notes


def main(outroot):
    outroot = Path(outroot)
    total = Counter()
    for fn in (build_cq4, build_cq8, build_cq4p, build_d64, build_k5):
        total.update(fn(outroot))
    notes = reachability_proof()
    with open(outroot / "cov_proofs.txt", "w") as f:
        for k, n in notes.items():
            f.write(f"COVGEN {k} {n}\n")
    n = total["V"] + total["K"] + total["R"]
    print(f"TOTAL randomized/directed transactions: V={total['V']} "
          f"K={total['K']} R={total['R']} -> {n}")
    assert n >= 500, "transaction budget not met"
    print("GEN OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
