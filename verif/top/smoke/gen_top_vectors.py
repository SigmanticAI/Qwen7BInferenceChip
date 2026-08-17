#!/usr/bin/env python3
"""gen_top_vectors.py — full-chain smoke script generator for apex_top v0.1.

THE first RTL-vs-golden full-chain point: one directed attention micro-case
per tile width (D=64 AND D=128 — F-5 closure, 2026-07-09; T=8 context
tokens + 1 query, tier CQ-8, G=128) is driven phase by phase through the
REAL tile streams and checked bit-exact against
golden/apex_golden/attention.py (attention_fx — the NORMATIVE chain, §9b).
The D=128 case is a kept-passing F-5 regression: its phase-E/G stage-buffer
jobs use nb = BPR = 16 (F-5a) and its phase-G PAT_D weight emits read
column blocks 8..15 (F-5b) with golden-checked o8.

The generator RUNS the golden chain and emits a flat command script the TB
interprets. Everything the host would compute in v0.1 (weights in MXE beat
order, descriptors, S-2/S-3/S-4 fp32 composites, epilogue calibration) is
derived HERE from the same golden run; every scale composite is asserted to
be (a) exactly representable in fp32 and (b) fp16-grade (mantissa[12:0]==0)
— the apex_scale_quant bit-exactness contract. Weight scales are powers of
two so the S-2 composites s_h*s_w stay exact (documented v0.1 host contract).

Checked bit-exact against the golden run, at the tile boundary:
  s_h[0..8]   feeder scale tap        (phase B — RMSNorm + h feeder)
  K_f16,V_f16 KVQ write-port tap      (phase C — S-2 conversion, 1024 values)
  s_q         scale_quant tap         (phase D — Q7 feeder-from-acc)
  s_k,s_v     feeder scale tap        (phase E — KVQ readback feeders)
  score_fx    score tap               (phase F — S-3 dequant, 8 values)
  p_q115      probability tap         (phase F — ASU online softmax)
  s_c         scale_quant tap         (phase F — S-4 P-requant fold)
  o8          result stream           (phase G — P·V̂ + C-2 epilogue, 64 vals)
Plus CSR sanity (INFO_*, VERSION, idle, W1C) and PERF non-zero checks.

Command grammar: see tb_apex_smoke.sv header (whitespace-separated tokens).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden.attention import (  # noqa: E402
    SCORE_FRAC, TIER_CQ4P, TIER_CQ8, attention_fx, attention_ref_full,
)
from apex_golden.fp import f16_bits_to_f64  # noqa: E402

T = 8            # context tokens; +1 query row
SEED = 0xA9E7_0001
OI = (5, 50)     # CQ-4+ smoke: static outlier mask channels (build ROM)

# power-of-two weight scales: keeps every S-2/Q7 composite s_h*s_w exact in
# fp32 with an fp16-grade mantissa (apex_scale_quant contract C2)
S_WQ = 2.0 ** -6
S_WK = 2.0 ** -6
S_WV = 2.0 ** -6

OP_GEMM_WS = 0x01
OP_GEMM_OS = 0x02

KV = {  # kvq_engine AXI-Lite map (kvq_engine.sv; window routes per tier)
    "CTRL": 0x00, "STATUS": 0x04, "INFO_TIER": 0x0C, "OCC": 0x24,
    "WADDR": 0x28, "RADDR": 0x2C, "INFO_OUTK": 0x3C,
}
CSR = {  # csr_regs.sv map (§7)
    "CTRL": 0x00, "STATUS": 0x04, "INFO_N": 0x08, "INFO_D": 0x0C,
    "INFO_G": 0x10, "INFO_TIER": 0x14, "INFO_VER": 0x18, "TIER_CTRL": 0x20,
    "THRESH": 0x24, "FLUSH": 0x28, "IMP": 0x2C, "PERF_CTRL": 0x30,
    "PERF_CYC": 0x34, "PERF_B0": 0x38,
}


def f32_bits_exact(x: float, what: str) -> int:
    """fp32 bits of x, asserting exactness + fp16-grade mantissa (C2)."""
    b = np.float64(x).astype(np.float32)
    assert float(np.float64(b)) == float(x), f"{what}: not fp32-exact: {x}"
    bits = int(b.view(np.uint32))
    assert bits & 0x1FFF == 0, f"{what}: not fp16-grade: {bits:08x}"
    assert 0 < ((bits >> 23) & 0xFF) < 255 and bits >> 31 == 0, \
        f"{what}: not positive normal: {bits:08x}"
    return bits


def desc_words(opcode: int, m: int, k: int, n: int, *, mode_os: int = 0,
               acc: int = 0, rq_en: int = 0, rq_scale: int = 0,
               rq_shift: int = 0) -> list[int]:
    """128-bit mxe_desc_t (apex_pkg.sv, layout FROZEN) as 4x32b, MSW first."""
    v = (opcode & 0xFF) | (m << 8) | (k << 20) | (n << 32) \
        | (rq_scale << 44) | (rq_shift << 60) | (rq_en << 65) \
        | (acc << 66) | (mode_os << 67)          # buf ids = 0 (implemented)
    return [(v >> (32 * i)) & 0xFFFFFFFF for i in (3, 2, 1, 0)]


def wgt_beats_ws(W: np.ndarray, j: int) -> list[int]:
    """mxe_ctrl WLOAD stream for B = W[:, 8j:8j+8], K = W.shape[0] (mult of 8):
    beat (p, c) lane r = B[8p+r][c]."""
    K = W.shape[0]
    beats = []
    for p in range(K // 8):
        for c in range(8):
            w = 0
            for r in range(8):
                w |= (int(W[8 * p + r][8 * j + c]) & 0xFF) << (8 * r)
            beats.append(w)
    return beats


class Script:
    def __init__(self):
        self.lines: list[str] = []
        self.n_expect = 0

    def emit(self, *toks):
        self.lines.append(" ".join(str(t) for t in toks))

    # ── bus / poll ─────────────────────────────────────────────────────────
    def csrw(self, a, d): self.emit("CSRW", f"{a:02x}", f"{d:08x}")

    def csrr(self, a, mask, exp):
        self.emit("CSRR", f"{a:02x}", f"{mask:08x}", f"{exp:08x}")
        self.n_expect += 1

    def csrp(self, a, mask, exp):
        self.emit("CSRP", f"{a:02x}", f"{mask:08x}", f"{exp:08x}")

    def csrn(self, a, mask):
        self.emit("CSRN", f"{a:02x}", f"{mask:08x}")
        self.n_expect += 1

    def kvw(self, a, d): self.emit("KVW", f"{a:02x}", f"{d:08x}")

    def kvp(self, a, mask, exp):
        self.emit("KVP", f"{a:02x}", f"{mask:08x}", f"{exp:08x}")

    def route(self, *, fsrc=0, fdst=0, asrc=0, wsrc=0, rdst=0, qsrc=0,
              kvu=1, blk=0):
        # rt_* are LEVELS: "change only while the touched paths are idle"
        # (apex_top port contract). Quiescence is VERIFIED, not assumed: poll
        # whole-tile STATUS.idle first. This is load-bearing — the EFS/ESS
        # scale taps are pushed BEFORE the code beats drain (both quant
        # blocks emit scale in ST_SPUSH, codes in ST_DRAIN), so scripted
        # expectations can pass while tail beats are still in flight; an
        # unpolled route flip severs those streams mid-transfer (the phase-F
        # 'drive_qj stall' root cause, 2026-07-09).
        # kvu (rt_kv_user) defaults 1 = VALUE path (the historical hardwire);
        # kvu=0 routes grouped KEYS at CQ-4/CQ-4+ (F-2).
        self.csrp(CSR["STATUS"], 0x1, 0x1)
        r = fsrc | (fdst << 1) | (asrc << 2) | (wsrc << 3) | (rdst << 4) \
            | (qsrc << 6) | (kvu << 7) | (blk << 8)
        self.emit("ROUTE", f"{r:04x}")

    def desc(self, words): self.emit("DESC", *[f"{w:08x}" for w in words])

    # ── streams / jobs ─────────────────────────────────────────────────────
    def xrow(self, row):
        self.emit("XR", *[f"{int(v) & 0xFF:02x}" for v in row])

    def grow(self, g):
        self.emit("GR", *[f"{int(v) & 0xFFFF:04x}" for v in g])

    def wbeats(self, beats):
        for b in beats:
            self.emit("WB", f"{b:016x}")

    def fjob(self, rows): self.emit("FJOB", f"{rows:x}")
    def qjob(self, mode, cols): self.emit("QJOB", mode, f"{cols:x}")
    def sjob(self, cols): self.emit("SJOB", f"{cols:x}")
    def ljob(self, beats, lanes): self.emit("LJOB", f"{beats:x}", f"{lanes:x}")

    def aj(self, op, bank, pat, rows, nb, sel):
        self.emit("AJ", op, bank, pat, f"{rows:x}", f"{nb:x}", f"{sel:x}")

    def wj(self, op, bank, pat, rows, nb, sel):
        self.emit("WJ", op, bank, pat, f"{rows:x}", f"{nb:x}", f"{sel:x}")

    def qs(self, bits): self.emit("QS", f"{bits:08x}")
    def cs(self, bits): self.emit("CS", f"{bits:08x}")

    # ── expectations ───────────────────────────────────────────────────────
    def efs(self, bits, last):
        self.emit("EFS", f"{int(bits):04x}", int(last))
        self.n_expect += 1

    def ess(self, bits, last):
        self.emit("ESS", f"{int(bits):04x}", int(last))
        self.n_expect += 1

    def ero(self, lanes, last):
        self.emit("ERO", *[f"{int(v) & 0xFFFFFFFF:08x}" for v in lanes],
                  int(last))
        self.n_expect += 1

    def etip(self, blk):
        self.emit("ETIP", f"{blk:02x}")
        self.n_expect += 1

    def tap(self, kind, vals, width):
        self.emit(kind, f"{len(vals):x}")
        for v in vals:
            self.emit(f"{int(v) & ((1 << width) - 1):0{width // 4}x}")
        self.n_expect += len(vals)

    def estk(self, mask, exp):
        self.emit("ESTK", f"{mask:04x}", f"{exp:04x}")
        self.n_expect += 1


def main(build_dir: str) -> None:
    build_case(build_dir, 64, "vectors_smoke.txt")
    build_case(build_dir, 128, "vectors_smoke_d128.txt")
    # F-2 closure smoke: the SAME full chain at CQ-4+ THROUGH THE TILE —
    # grouped keys (tuser=0, WRITE_ADDR group base, D-008 FLUSH on the
    # 8-token partial group at G=128) + the static outlier mask ROM.
    mask = Path(build_dir) / "mask_smoke_cq4p.hex"
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_text("\n".join("01" if c in OI else "00"
                              for c in range(64)) + "\n")
    build_case(build_dir, 64, "vectors_smoke_cq4p.txt", tier=TIER_CQ4P)


def build_case(build_dir: str, D: int, fname: str,
               tier: str = TIER_CQ8) -> None:
    BPR = D // 8
    rng = np.random.default_rng(
        SEED if D == 64 else SEED + 0x80) if tier == TIER_CQ8 \
        else np.random.default_rng(SEED + 0x40)
    X = rng.normal(0.0, 1.0, (T + 1, D))
    Wq8 = rng.integers(-127, 128, (D, D), dtype=np.int64)
    Wk8 = rng.integers(-127, 128, (D, D), dtype=np.int64)
    Wv8 = rng.integers(-127, 128, (D, D), dtype=np.int64)
    gamma = rng.integers(4096, 12289, D, dtype=np.int64)   # Q2.13 in [0.5,1.5]
    oi = ()
    if tier == TIER_CQ4P:
        # plant key-channel outliers on the masked channels so the fp16
        # identity lane is load-bearing: K = h8·Wk, so shrink every
        # NON-masked Wk column ~8x — channels {5,50} dominate the K amax
        # (asserted below), the per-channel INT4 codec outlier pattern.
        oi = OI
        keep = np.ones(D, dtype=bool)
        keep[list(oi)] = False
        Wk8[:, keep] = Wk8[:, keep] // 8
        # temper the query projection so softmax is NOT one-hot: a near-
        # one-hot p at T=8 pins e2e at the INT4-V C-1 floor (~7% — the
        # documented D023-T1 phenomenon), which is a data property, not a
        # tile property; the smoke case must sit INSIDE the D-023 budget
        Wq8 //= 16

    r = attention_fx(X, Wq8, Wk8, Wv8, S_WQ, S_WK, S_WV, gamma,
                     tier=tier, G=128, outlier_idx=oi)
    ref = attention_ref_full(X, Wq8, Wk8, Wv8, S_WQ, S_WK, S_WV, gamma)
    if tier == TIER_CQ4P:
        amax = np.max(np.abs(f16_bits_to_f64(r.K_f16).reshape(T, D)), axis=0)
        assert set(np.argsort(amax)[-2:]) == set(oi), \
            "planted outliers are not the top-2 K channels"

    # golden self-report (informational; the D-023 gate lives in golden CI)
    vscale = float(np.max(np.abs(ref.V_ref)))
    e2e = float(np.max(np.abs(r.out_hat - ref.y_ref))) / vscale
    print(f"golden: e2e max err = {100.0 * e2e:.3f}% of value scale "
          f"(tier {tier}; D-023 budget 5%)")
    assert e2e < 0.05, "golden chain out of D-023 budget — bad stimulus draw"

    # tile-domain contract checks (apex_scale_quant C1, score INT32)
    for name, acc in (("acc_k", r.acc_k), ("acc_v", r.acc_v),
                      ("acc_q", r.acc_q), ("acc_s", r.acc_s),
                      ("acc_o", r.acc_o)):
        assert np.max(np.abs(acc)) < 2 ** 24, f"{name} exceeds C1 (2^24)"

    s_h = f16_bits_to_f64(r.s_h)
    s_v = f16_bits_to_f64(r.s_v)

    s = Script()
    tier_code = {TIER_CQ8: 0, TIER_CQ4P: 2}[tier]
    # INFO_TIER TRUTH (F-2): CQ4P bit only when the outlier mask is built in
    tier_truth = 0x7 if tier == TIER_CQ4P else 0x3

    # ── phase A: CSR sanity (version, INFO, idle, W1C) ────────────────────
    s.emit(f"// phase A: CSR sanity (D={D} tile, {tier})")
    s.csrr(CSR["INFO_VER"], 0xFFFFFFFF, 0x00010000)
    s.csrr(CSR["INFO_N"], 0xFFFFFFFF, 8)
    s.csrr(CSR["INFO_D"], 0xFFFFFFFF, D)
    s.csrr(CSR["INFO_G"], 0xFFFFFFFF, 128)
    s.csrr(CSR["INFO_TIER"], 0xFFFFFFFF, tier_truth)       # build truth
    s.csrr(CSR["STATUS"], 0x0000_0003, 0x0000_0001)        # idle, no error
    s.csrw(CSR["CTRL"], 0x1)                               # enable (SEQ)
    s.csrw(CSR["PERF_CTRL"], 0x3)                          # perf en + clear
    # host tier select (TIER_CTRL resets to CQ4 — set BEFORE touching kv_*)
    s.csrw(CSR["TIER_CTRL"], tier_code)
    s.csrr(CSR["TIER_CTRL"], 0xFFFFFFFF, tier_code)
    s.kvw(KV["CTRL"], 0x2)                                 # enable engine[tier]
    # per-engine truth via the routed window (proves the D-024 tier mux)
    s.kvp(KV["INFO_TIER"], 0xFFFFFFFF, tier_code)
    if tier == TIER_CQ4P:
        s.kvp(KV["INFO_OUTK"], 0xFFFFFFFF, 2)
    # illegal descriptor (K=0) -> desc_error -> STATUS[1] sticky -> W1C
    s.desc(desc_words(OP_GEMM_WS, 1, 0, 1))
    s.csrp(CSR["STATUS"], 0x2, 0x2)
    s.csrw(CSR["STATUS"], 0x2)                             # W1C
    s.csrr(CSR["STATUS"], 0x2, 0x0)

    # ── phases B..G: one full attention pass (run twice: clean + post-reset)
    run_attention_pass(s, r, gamma, Wq8, Wk8, Wv8, s_h, s_v, D, tier)

    # ── D-020 at tile level: soft reset + full clean re-run (same sim) ────
    # v0.1 boundary (apex_top header): CSR CTRL.soft_reset drives ONLY the
    # SEQ abort contract; KVQ D-020 reset is its own CTRL register. Test:
    # dirty the SEQ queue with UNDISPATCHED legal descriptors (enable=0
    # holds the queue), then abort + KVQ soft reset, verify idle/no-error,
    # and re-run the ENTIRE attention pass bit-exact. If the abort leaked a
    # queued descriptor, SEQ would dispatch it on re-enable and the dataless
    # MXE job would wedge round 2 (honest watchdog fail, never silent).
    s.emit("// D-020 tile soft reset: abort queued descs + KVQ reset + rerun")
    s.csrp(CSR["STATUS"], 0x1, 0x1)                        # quiescent first
    s.csrw(CSR["CTRL"], 0x0)                               # enable=0: hold q
    s.desc(desc_words(OP_GEMM_WS, 8, 64, 8))               # queued, legal,
    s.desc(desc_words(OP_GEMM_OS, 1, 8, 8, mode_os=1))     #  must be DISCARDED
    s.csrw(CSR["CTRL"], 0x3)                               # enable + abort
    s.kvw(KV["CTRL"], 0x3)                                 # KVQ D-020 reset
    s.csrp(CSR["STATUS"], 0x1, 0x1)                        # tile idle again
    s.kvp(KV["STATUS"], 0x1, 0x1)                          # KVQ idle again
    s.csrr(CSR["STATUS"], 0x2, 0x0)                        # abort is no error
    s.emit("// round 2: full clean re-run after soft reset")
    run_attention_pass(s, r, gamma, Wq8, Wk8, Wv8, s_h, s_v, D, tier)

    # ── final: quiescence, no new errors, PERF alive ──────────────────────
    s.emit("// final: quiescence + error/PERF checks")
    s.emit("ENDTAPS")
    s.n_expect += 3
    s.csrp(CSR["STATUS"], 0x1, 0x1)                        # tile idle
    s.kvp(KV["STATUS"], 0x1, 0x1)                          # KVQ idle
    s.csrr(CSR["STATUS"], 0x2, 0x0)                        # no new error
    s.csrn(CSR["PERF_CYC"], 0xFFFFFFFF)
    s.csrn(CSR["PERF_B0"] + 4 * 1, 0xFFFFFFFF)             # MXE busy lane
    # only the DELIBERATE illegal-descriptor stickies may be set (phase A +
    # one mid-sequence per round — same bit); then F-3: W1C-clear the tile
    # ERR_STICKY window and verify (set -> clear -> verify)
    s.estk(0xFFFF, 0x0001)
    s.csrw(0x58, 0x0000FFFF)                               # ERR_STICKY W1C
    s.csrr(0x58, 0xFFFFFFFF, 0x00000000)                   # readback clear
    s.estk(0xFFFF, 0x0000)                                 # pins clear too
    s.emit("DONE")

    out = Path(build_dir) / fname
    out.write_text("\n".join(s.lines) + "\n")
    kf = 2 * T * D  # per-round KVQ f16 tap count (K + V)
    print(f"wrote {out}: {len(s.lines)} lines, {s.n_expect} scripted checks "
          f"(D={D}, 2 rounds; +{2 * (kf + 2 * T)}-value tap tables inside)")
    scale, shift = r.rq
    print(f"golden digest: s_q={int(r.s_q):04x} s_c={int(r.s_c):04x} "
          f"rq=({scale},{shift}) o8[0:4]={[int(v) for v in r.o8[:4]]}")


def run_attention_pass(s: Script, r, gamma, Wq8, Wk8, Wv8, s_h, s_v,
                       D: int, tier: str = TIER_CQ8) -> None:
    """Phases B..G — one complete attention micro-case, every mid-chain
    tensor checked bit-exact. Emitted twice (clean + post-soft-reset).
    tier selects the KVQ store discipline in phase C: CQ-8 stores K and V
    per token through the value path (kvu=1); CQ-4+ stores K as ONE grouped
    key run (kvu=0, WRITE_ADDR = group base, T=8 < G=128 partial group
    closed by the D-008 CSR FLUSH), values per token (kvu=1)."""
    BPR = D // 8
    key_grouped = tier != TIER_CQ8
    # ── phase B: x8 -> RMSNorm -> widen -> feeder -> act stage bank0 ──────
    s.emit("// phase B: hidden pipeline (Q0..Q2/S-1)")
    s.route(fsrc=0, fdst=0, asrc=0)
    s.aj(0, 0, 0, T + 1, BPR, 0)                           # load bank0, 9 rows
    s.fjob(T + 1)
    for t in range(T + 1):
        s.xrow(r.x8[t])
        s.grow(gamma)
        s.efs(r.s_h[t], t == T)

    # ── phase C: K/V projections (Q3) + S-2 write port (Q4) + KVQ store ───
    s.emit(f"// phase C: K/V projections + KVQ compress (Q3/Q4/S-2, {tier})")
    s.route(rdst=1, kvu=0 if key_grouped else 1)
    kf = [int(v) for v in np.asarray(r.K_f16).reshape(-1)]
    vf = [int(v) for v in np.asarray(r.V_f16).reshape(-1)]
    s.tap("TAPF16", kf + vf, 16)
    for W, s_w, base in ((Wk8, S_WK, 0), (Wv8, S_WV, T)):
        is_key_run = key_grouped and base == 0
        if is_key_run:
            # ONE key group run: base address programmed once; the engine
            # stays non-idle (ST_KACCEPT) between tokens by design, so the
            # per-token idle polls are skipped inside the run.
            s.emit("// grouped KEY run: WADDR=group base, tuser=0 (F-2)")
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["WADDR"], 0)
        if key_grouped and base == T:
            s.route(rdst=1, kvu=1)                          # value path now
        for t in range(T):
            comp = f32_bits_exact(float(s_h[t]) * s_w, f"S-2 comp t={t}")
            if not is_key_run:
                s.kvp(KV["STATUS"], 0x1, 0x1)
                s.kvw(KV["WADDR"], base + t)
            s.qjob(0, D)
            for _ in range(D):
                s.qs(comp)
            s.ljob(BPR, 8)
            for j in range(BPR):
                s.aj(1, 0, 0, 1, BPR, t)                   # act: bank0 row t
                s.desc(desc_words(OP_GEMM_WS, 1, D, 8))
                s.wbeats(wgt_beats_ws(W, j))
        if is_key_run:
            # D-008 through the tile: the 8-token group is partial (G=128).
            # Tile idle (poll) implies every f16 beat was ACCEPTED by the
            # engine (post-skid §5 semantics), so the engine is parked in
            # ST_KACCEPT and the CSR FLUSH freezes the group at 8 tokens.
            s.emit("// D-008: CSR FLUSH closes the partial key group (8<G)")
            s.csrp(CSR["STATUS"], 0x1, 0x1)
            s.csrw(CSR["FLUSH"], 0x1)
            s.kvp(KV["STATUS"], 0x1, 0x1)                  # group emitted
        if base == 0:
            # mid-sequence illegal-descriptor reject (§3): K=0, queued
            # between the K-projection and V-projection legal jobs — sticky
            # sets, W1C clears, and the chain continues unpolluted (D-006:
            # the reject cannot disturb the in-flight/neighbouring jobs)
            s.emit("// phase C: mid-sequence illegal descriptor (K=0)")
            s.desc(desc_words(OP_GEMM_WS, 1, 0, 1))
            s.csrp(CSR["STATUS"], 0x2, 0x2)
            s.csrw(CSR["STATUS"], 0x2)                     # W1C
            s.csrr(CSR["STATUS"], 0x2, 0x0)

    # ── phase D: q projection (Q3) + Q7 feeder-from-acc -> act bank1 ──────
    s.emit("// phase D: q projection + Q7 quant")
    s.route(rdst=1, asrc=1)
    comp_q = f32_bits_exact(float(s_h[T]) * S_WQ, "Q7 comp")
    s.qjob(1, D)
    for _ in range(D):
        s.qs(comp_q)
    s.ljob(BPR, 8)
    for j in range(BPR):
        s.aj(1, 0, 0, 1, BPR, T)                           # act: bank0 row T
        s.desc(desc_words(OP_GEMM_WS, 1, D, 8))
        s.wbeats(wgt_beats_ws(Wq8, j))
    s.ess(r.s_q, 1)
    s.aj(0, 1, 0, 1, BPR, 0)                               # load bank1 <- q8

    # ── phase E: KVQ readback (Q5) -> feeders (Q6) -> wgt stage ───────────
    s.emit("// phase E: KVQ readback + K/V feeders (Q5/Q6)")
    s.route(fsrc=1, fdst=1)
    for bank, base, scl in ((0, 0, r.s_k), (1, T, r.s_v)):
        s.wj(0, bank, 0, T, BPR, 0)
        s.fjob(T)
        for t in range(T):
            s.kvp(KV["STATUS"], 0x1, 0x1)
            s.kvw(KV["RADDR"], base + t)
            s.efs(scl[t], t == T - 1)

    # ── phase F: Q·K̂ᵀ (Q8) -> S-3 (Q9) -> TIP + softmax (Q10) -> S-4 (Q11)
    s.emit("// phase F: scores + softmax + P-requant (Q8..Q11/S-3/S-4)")
    # asrc=1 is REQUIRED here: the S-4 c8 codes drain scale_quant -> act
    # stage bank1 (the phase-G act source). With asrc=0 that load would
    # never see a beat AND the act-stage load mux would instead swallow any
    # feeder traffic (as_ld_valid falls back to fq_out when fdst=0) —
    # exactly the phase-F wedge/misroute found 2026-07-09.
    s.route(rdst=2, asrc=1, qsrc=1, wsrc=1, blk=0)
    s.tap("TAPSC", [int(v) for v in r.score_fx], 32)
    s.tap("TAPPR", [int(v) for v in r.p_q115], 16)
    s.sjob(T)
    s_q_val = float(f16_bits_to_f64(np.array([r.s_q]))[0])
    s_k_val = f16_bits_to_f64(np.asarray(r.s_k, dtype=np.uint16))
    comp64 = s_q_val * s_k_val * (float(1 << SCORE_FRAC) / math.sqrt(float(D)))
    comp32 = comp64.astype(np.float32).view(np.uint32)     # S-3 single narrow
    for t in range(T):
        s.cs(int(comp32[t]))
    s.qjob(1, T)                                           # S-4 P-requant
    for t in range(T):
        s.qs(f32_bits_exact(float(s_v[t]) * 2.0 ** -15, f"S-4 comp t={t}"))
    s.wj(1, 0, 1, T, BPR, 0)                               # wgt: k8ᵀ (PAT_T)
    s.aj(1, 1, 0, 1, BPR, 0)                               # act: q8 (bank1)
    s.desc(desc_words(OP_GEMM_OS, 1, D, T, mode_os=1))
    s.etip(0)
    s.ess(r.s_c, 1)
    s.aj(0, 1, 0, 1, 1, 0)                                 # load bank1 <- c8

    # ── phase G: P·V̂ (Q12) + C-2 epilogue (Q13) -> o8 out ────────────────
    s.emit("// phase G: P·V and epilogue (Q12/Q13)")
    s.route(rdst=0, wsrc=1)
    scale, shift = r.rq
    for j in range(BPR):
        s.aj(1, 1, 0, 1, 1, 0)                             # act: c8 (1 beat)
        s.wj(1, 1, 2, T, BPR, j)                           # wgt: v8 block j
        s.desc(desc_words(OP_GEMM_OS, 1, T, 8, mode_os=1, rq_en=1,
                          rq_scale=int(scale), rq_shift=int(shift)))
        s.ero([int(v) for v in r.o8[8 * j:8 * j + 8]], 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
