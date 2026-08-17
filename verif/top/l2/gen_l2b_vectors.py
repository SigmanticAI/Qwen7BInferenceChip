#!/usr/bin/env python3
"""gen_l2b_vectors.py — golden-driven scripts + mask ROMs for tb_l2b_chain
(Layer-2 chain b: kvq_engine -> seam_feeder_quant -> mxe_top).

Golden arbiter chain: apex_golden.cq_codec (compress/decompress per
tier — the D-001 vector-proven codec model), apex_golden.attention
quant_rows_i8 (D-021 feeder), apex_golden.compute.gemm_i8 (MXE).

Configs (each is a separate Verilator build; TIER/G/OUTLIER_K/MASK_FILE are
kvq_engine synthesis parameters — the same reason apex_top v0.1 is a
CQ-8-only tile, see the L3 feasibility notes):
  t0      TIER=0 CQ-8  G=16 : K and V through the VALUE path (tuser=1),
          exactly the apex_top wiring; T in {5,16,23} (G unused by CQ-8)
  t1      TIER=1 CQ-4  G=16 : K through the KEY path in G-groups, partial
          groups closed by the D-008 FLUSH; T in {5,16,23,37} straddling G
          (partial-only / exact / full+partial / 2 full+partial)
  t2      TIER=2 CQ-4+ G=16 OUTLIER_K=2 mask={5,50}: same straddle set with
          x14 planted outliers on the masked channels, plus all-zero V rows
          (adv/zeroV recipe) and an all-zero K row
  adv_T1      TIER=2 G=128 mask={1,2} : the golden adv/T1 case  (T=1 flush)
  adv_out1000 TIER=2 G=128 mask={5,50}: golden adv/outlier1000  (T=128)
  calib_T128  TIER=2 G=64 mask=top2(K): frozen calib d64_T128_G64 tensors
  calib_T70   TIER=2 G=64 mask=top2(K): frozen calib d64_T70_G64 tensors
          (named cases run at CQ-4+ HERE because the committed apex_top
          hard-wires s_axis tuser=1 / OUTLIER_K=0 — see verif/top/l3)
  t1_reset    mid-op rst_n + KVQ D-020 CTRL soft reset, then clean re-run

Every readback row's fp16 feeder scale (EFS) is checked; every k8/v8 code
plane is checked THROUGH the MXE via INT8 identity weights; the golden
acc_s (q8 . k8ᵀ score accumulators) is checked with the golden q8 as the
weight column for the named cases; random full-B GEMMs mix all codes.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden.attention import kvq_roundtrip, quant_rows_i8  # noqa: E402
from apex_golden.compute import gemm_i8  # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402

TIER_NAME = {0: "CQ-8", 1: "CQ-4", 2: "CQ-4+"}
OP_GEMM_WS = 0x01
SEED = 0x12B0_0001
VEC = REPO / "golden/vectors"

KV = {"CTRL": 0x00, "STATUS": 0x04, "OCC": 0x24, "WADDR": 0x28, "RADDR": 0x2C}


def to16(x):
    return f64_to_f16_bits(np.asarray(x, dtype=np.float64)).reshape(np.asarray(x).shape)


def top2_outliers(K16):
    amax = np.max(np.abs(f16_bits_to_f64(K16).reshape(K16.shape)), axis=0)
    return np.sort(np.argsort(amax)[-2:])


def desc_words(opcode, m, k, n):
    v = (opcode & 0xFF) | (m << 8) | (k << 20) | (n << 32)
    return [(v >> (32 * i)) & 0xFFFFFFFF for i in (3, 2, 1, 0)]


def wgt_beats(B, K, N):
    KB = (K + 7) // 8
    beats = []
    for p in range(KB):
        for c in range(N):
            w = 0
            for r in range(8):
                kk = 8 * p + r
                v = int(B[kk][c]) if kk < K else 0
                w |= (v & 0xFF) << (8 * r)
            beats.append(w)
    return beats


class Script:
    def __init__(self):
        self.lines = []
        self.n_expect = 0

    def emit(self, *toks):
        self.lines.append(" ".join(str(t) for t in toks))

    def kvw(self, a, d):
        self.emit("KVW", f"{a:02x}", f"{d:08x}")

    def kvp(self, a, m, e):
        self.emit("KVP", f"{a:02x}", f"{m:08x}", f"{e:08x}")

    def tok(self, kind, vals):
        self.emit(kind, *[f"{int(v) & 0xFFFF:04x}" for v in vals])

    def desc(self, words):
        self.emit("DESC", *[f"{w:08x}" for w in words])

    def efs(self, scales, last_idx):
        self.emit("EFS", f"{len(scales):x}")
        for i, v in enumerate(scales):
            self.emit(f"{int(v) & 0xFFFF:04x}", int(i == last_idx))
        self.n_expect += len(scales)

    def ero(self, lanes, last):
        self.emit("ERO", *[f"{int(v) & 0xFFFFFFFF:08x}" for v in lanes],
                  int(last))
        self.n_expect += 1

    def stk(self, mask, exp):
        self.emit("STK", f"{mask:02x}", f"{exp:02x}")
        self.n_expect += 1


class Case:
    """One K/V tensor pair pushed through the engine + feeder + MXE."""

    def __init__(self, name, K16, V16, tier, G, oi, q=None):
        self.name = name
        T, D = K16.shape
        self.T, self.D = T, D
        self.K16, self.V16 = K16, V16
        self.K_hat, self.V_hat, _, _ = kvq_roundtrip(K16, V16,
                                                     TIER_NAME[tier], G, oi)
        self.k8, self.s_k = quant_rows_i8(self.K_hat)
        self.v8, self.s_v = quant_rows_i8(self.V_hat)
        self.q8 = None
        self.acc_s = None
        if q is not None:
            q8m, s_qa = quant_rows_i8(np.asarray(q, dtype=np.float64)[None, :])
            self.q8 = q8m[0]
            self.acc_s = gemm_i8(self.q8[None, :], self.k8.T)[0]
        self.G = G


def write_case(s, rng, case, kbase, vbase, occ_before, cov, full_identity):
    """KVQ writes (per tier path), then chunked readback->feeder->MXE."""
    T, D, G = case.T, case.D, case.G
    tier_key_path = case.name_tier != 0
    # ── writes ───────────────────────────────────────────────────────────────
    if tier_key_path:
        g0 = 0
        while g0 < T:
            g1 = min(g0 + G, T)
            s.emit(f"// {case.name}: key group tokens {g0}..{g1 - 1}")
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["WADDR"], kbase + g0)
            for t in range(g0, g1):
                s.tok("KT", case.K16[t])
            if g1 - g0 < G:
                s.emit("// partial group -> D-008 FLUSH")
                s.emit("FLUSH")
                cov["flush"] += 1
            s.kvp(KV["STATUS"], 0x1, 0x1)
            g0 = g1
        cov["straddle_" + ("lt" if T < G else "eq" if T == G else "gt")] += 1
    else:
        for t in range(T):
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["WADDR"], kbase + t)
            s.tok("VT", case.K16[t])
    for t in range(T):
        s.kvp(KV["STATUS"], 0x1, 0x1)
        s.kvw(KV["WADDR"], vbase + t)
        s.tok("VT", case.V16[t])
    s.kvp(KV["STATUS"], 0x1, 0x1)
    s.kvp(KV["OCC"], 0xFFFFFFFF, occ_before + 2 * T)

    # ── chunked readback -> feeder -> MXE ───────────────────────────────────
    def pass_over(rows, base, scales, B, acc, tag):
        nc = len(rows)
        KB = D // 8
        s.emit(f"// {case.name}: {tag} rows {rows[0]}..{rows[-1]}")
        s.emit("FJ", f"{nc:x}")
        s.desc(desc_words(OP_GEMM_WS, nc, D, 8))
        s.efs([scales[t] for t in rows], nc - 1)
        for t in rows:
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.emit("RD", f"{base + t:x}")
        for w in wgt_beats(B, D, 8):
            s.emit("WB", f"{w:016x}")
        for m in range(nc):
            s.ero(acc[m], m == nc - 1)
        s.emit("IDLE")

    for plane, base, codes, scales in (("k", kbase, case.k8, case.s_k),
                                       ("v", vbase, case.v8, case.s_v)):
        c0 = 0
        ident_j = 0
        while c0 < T:
            c1 = min(c0 + 8, T)
            rows = list(range(c0, c1))
            chunk = codes[c0:c1]
            # score-shaped job: golden q8 column (named cases) or random B
            if plane == "k" and case.q8 is not None:
                B = np.zeros((D, 8), dtype=np.int64)
                B[:, 0] = case.q8
                acc = np.zeros((len(rows), 8), dtype=np.int64)
                acc[:, 0] = case.acc_s[c0:c1]
                pass_over(rows, base, scales, B, acc, "k acc_s (q8 col)")
                cov["acc_s"] += 1
            else:
                B = rng.integers(-128, 128, (D, 8), dtype=np.int64)
                acc = gemm_i8(chunk, B).astype(np.int64)
                pass_over(rows, base, scales, B, acc, f"{plane} random-B")
            # identity dumps: full sweep (small cases) or rotating block
            jlist = range(D // 8) if full_identity else [ident_j % (D // 8)]
            for j in jlist:
                B = np.zeros((D, 8), dtype=np.int64)
                for c in range(8):
                    B[8 * j + c][c] = 1
                acc = np.zeros((len(rows), 8), dtype=np.int64)
                acc[:, :] = chunk[:, 8 * j:8 * j + 8]
                pass_over(rows, base, scales, B, acc, f"{plane}8 identity j={j}")
                cov["identity"] += 1
            ident_j += 1
            c0 = c1
    cov["cases"] += 1


def gauss16(rng, T, D, out_ch=(), out_gain=14.0, zero_v=False):
    K = rng.normal(0, 1, (T, D))
    for c in out_ch:
        K[:, c] *= out_gain
    V = np.zeros((T, D)) if zero_v else rng.normal(0, 1, (T, D))
    return to16(K), to16(V)


def build_config(out, cfg, tier, G, oi, cases, reset=False):
    rng = np.random.default_rng(SEED ^ zlib.crc32(cfg.encode()))
    s = Script()
    cov = {"cases": 0, "flush": 0, "identity": 0, "acc_s": 0,
           "straddle_lt": 0, "straddle_eq": 0, "straddle_gt": 0, "reset": 0}
    s.emit(f"// l2b config {cfg}: TIER={tier} G={G} oi={list(oi)}")
    s.kvw(KV["CTRL"], 0x2)
    s.stk(0x3, 0x0)

    if reset:
        # mid-op rst_n: half a value token in flight, then hard reset
        case0 = cases[0]
        half = case0.D // 2
        s.emit("// sacrificial half token, then mid-op rst_n")
        s.kvw(KV["WADDR"], 0)
        s.emit("PT", f"{half:x}",
               *[f"{int(v) & 0xFFFF:04x}" for v in case0.V16[0][:half]])
        s.emit("RST")
        s.n_expect += 2
        s.kvw(KV["CTRL"], 0x2)
        # D-020 CTRL soft reset mid-token, engine must go idle and stay sane
        s.emit("// D-020 soft reset mid-token")
        s.kvw(KV["WADDR"], 0)
        s.emit("PT", f"{half:x}",
               *[f"{int(v) & 0xFFFF:04x}" for v in case0.V16[0][:half]])
        s.kvw(KV["CTRL"], 0x3)
        s.kvp(KV["STATUS"], 0x1, 0x1)
        cov["reset"] = 1

    kbase = 0
    occ = 0
    for case in cases:
        case.name_tier = tier
        vbase = kbase + case.T
        full_ident = case.T <= 16
        write_case(s, rng, case, kbase, vbase, occ, cov, full_ident)
        occ += 2 * case.T
        kbase = vbase + case.T

    s.stk(0x3, 0x0)
    s.emit("ENDTAB")
    s.n_expect += 2
    s.emit("DONE")
    (out / f"vectors_l2b_{cfg}.txt").write_text("\n".join(s.lines) + "\n")
    print(f"l2b {cfg}: {s.n_expect} scripted checks, cov={cov}")
    for k, v in cov.items():
        print(f"COVGEN l2b_{cfg} {k} {v}")


def write_mask(out, cfg, D, oi):
    p = out / f"mask_{cfg}.hex"
    p.write_text("\n".join("01" if c in set(int(x) for x in oi) else "00"
                           for c in range(D)) + "\n")
    return p


def main(build_dir: str) -> None:
    out = Path(build_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = 64

    # t0 — CQ-8, apex_top wiring (value path for K and V)
    cases = []
    for T in (5, 16, 23):
        K16, V16 = gauss16(rng, T, D)
        cases.append(Case(f"t0/T{T}", K16, V16, 0, 16, []))
    build_config(out, "t0", 0, 16, [], cases)

    # t1 — CQ-4 key groups + FLUSH straddle set
    cases = []
    for T in (5, 16, 23, 37):
        K16, V16 = gauss16(rng, T, D)
        cases.append(Case(f"t1/T{T}", K16, V16, 1, 16, []))
    build_config(out, "t1", 1, 16, [], cases)

    # t1_reset — same tier config, mid-op resets + one clean case
    K16, V16 = gauss16(rng, 23, D)
    build_config(out, "t1_reset", 1, 16, [],
                 [Case("t1r/T23", K16, V16, 1, 16, [])], reset=True)

    # t2 — CQ-4+ with planted outliers on the masked channels + zero rows
    oi = (5, 50)
    write_mask(out, "t2", D, oi)
    cases = []
    for T in (5, 16, 23, 37):
        K16, V16 = gauss16(rng, T, D, out_ch=oi)
        cases.append(Case(f"t2/T{T}", K16, V16, 2, 16, oi))
    K16, V16 = gauss16(rng, 8, D, out_ch=oi, zero_v=True)   # adv/zeroV recipe
    cases.append(Case("t2/zeroV", K16, V16, 2, 16, oi))
    Kz = to16(np.zeros((8, D)))                             # adv/zeroK recipe
    _, Vn = gauss16(rng, 8, D)
    cases.append(Case("t2/zeroK", Kz, Vn, 2, 16, oi))
    build_config(out, "t2", 2, 16, oi, cases)

    # named golden cases at CQ-4+ (the tier the committed apex_top cannot
    # build — verified here through the real engine + feeder + MXE instead)
    arng = np.random.default_rng(0xA0D17)                   # frozen adv seed
    qa = f16_bits_to_f64(to16(arng.normal(0, 1, D))).reshape(D)
    K1 = arng.normal(0, 1, (1, D))
    K1 *= 2.0 / np.max(np.abs(K1))
    V1 = arng.normal(0, 1, (1, D))
    V1 *= 2.0 / np.max(np.abs(V1))
    write_mask(out, "adv_T1", D, (1, 2))
    build_config(out, "adv_T1", 2, 128, (1, 2),
                 [Case("adv/T1", to16(K1), to16(V1), 2, 128, (1, 2), q=qa)])

    # regenerate adv/outlier1000 exactly as golden/tests/test_attention.py
    # (same frozen generator draw ORDER: T1 pair, tied pair, then outliers)
    row_k = arng.normal(0, 1, D)
    row_k *= 2.0 / np.max(np.abs(row_k))
    _ = np.tile(row_k, (64, 1))
    Vt = arng.normal(0, 1, (64, D))
    Vt *= 2.0 / np.max(np.abs(Vt))
    To = 128
    Ko = arng.normal(0, 0.02, (To, D))
    Ko[:, 5] = arng.normal(0, 20.0, To)
    Ko[:, 50] = arng.normal(0, 20.0, To)
    Vo = arng.normal(0, 1, (To, D))
    Vo *= 2.0 / np.max(np.abs(Vo))
    qo = arng.normal(0, 1, D)
    qo *= 1.0 / np.max(np.abs(qo))
    qo = f16_bits_to_f64(to16(qo)).reshape(D)
    write_mask(out, "adv_out1000", D, (5, 50))
    build_config(out, "adv_out1000", 2, 128, (5, 50),
                 [Case("adv/outlier1000", to16(Ko), to16(Vo), 2, 128,
                       (5, 50), q=qo)])

    # D=128 named case (the committed apex_top cannot run D=128 AT ALL —
    # L3 finding F-5: 4-bit aj_nb/wj_nb/job_nb fields + PAT_D sel[2:0]
    # truncation; this chain has no stage buffer, so D=128 KVQ->feeder->MXE
    # gets its real-RTL coverage HERE, at CQ-4+ on the frozen calib tensors)
    z = np.load(VEC / "d128_T100_G128__CQ4.npz")
    K16 = np.ascontiguousarray(z["input_K"]).view(np.uint16)
    V16 = np.ascontiguousarray(z["input_V"]).view(np.uint16)
    qc = f16_bits_to_f64(K16[-1]).reshape(K16.shape[1])
    oi = top2_outliers(K16)
    write_mask(out, "calib_d128", K16.shape[1], oi)
    build_config(out, "calib_d128", 2, 128, tuple(int(x) for x in oi),
                 [Case("calib/d128_T100_G128", K16, V16, 2, 128, oi, q=qc)])

    # committed calibration tensors (golden/vectors, from golden/gen_vectors.py)
    for cfg, fn, G in (("calib_T128", "d64_T128_G64__CQ4.npz", 64),
                       ("calib_T70", "d64_T70_G64__CQ4.npz", 64)):
        z = np.load(VEC / fn)
        K16 = np.ascontiguousarray(z["input_K"]).view(np.uint16)
        V16 = np.ascontiguousarray(z["input_V"]).view(np.uint16)
        qc = f16_bits_to_f64(K16[-1]).reshape(K16.shape[1])
        oi = top2_outliers(K16)
        write_mask(out, cfg, K16.shape[1], oi)
        build_config(out, cfg, 2, G, tuple(int(x) for x in oi),
                     [Case(f"calib/{fn[:-4]}", K16, V16, 2, G, oi, q=qc)])

    print("l2b vector generation complete")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
