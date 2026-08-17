#!/usr/bin/env python3
"""attack_attention.py — INDEPENDENT adversarial audit of the golden
attention chain (golden/apex_golden/attention.py) and of the budget gates
in golden/tests/test_attention.py. Audit collateral: nothing here modifies
the deliverable; it imports it and attacks it.

Attacks (holes the shipped distributions do not cover):
  1. T=1 single token                      (their T grid starts at 16)
  2. exactly-tied scores (identical K rows)
  3. near-tie scores (1 fp16 ulp apart)
  4. steeply ASCENDING scores              (max online-softmax rescale count)
  5. dominant score (others underflow the exp LUT to 0) -> p==1.0 exactly?
  6. extreme outlier channels x1000, CQ-4 UNMASKED (worst legal codec case)
  7. all-zero K / all-zero V               (EPS scale path)
  8. partial-group straddles T in {65,127,129}, G in {64,128}
  9. score-dequant INT32 envelope: legal fp16 magnitudes that blow the
     documented assert (operating envelope probe, expected to raise)
 10. Q1.15 probability ceiling: does p_q115 reach 32768 (== 1.0, which does
     NOT fit a signed 16-bit Q1.15 lane)?
 11. brute-force re-check of the [Q] hard factor 0.57s (INT8) / 0.51s (INT4)

Every case is also pushed through THEIR derive_and_gate so a budget-gate
failure on legal in-contract data is exposed (that would mean the analytic
derivation is incomplete).
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "golden"))
sys.path.insert(0, str(ROOT / "golden" / "tests"))

from apex_golden import attention as at                      # noqa: E402
from apex_golden import cq_codec as cq                   # noqa: E402
from apex_golden import compute as cp                        # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f32_bits_to_f64, f64_to_f16_bits  # noqa: E402
import test_attention as ta                                  # noqa: E402

FAILS = []
NOTES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(f"{name}: {detail}")


def note(msg):
    print(f"  NOTE  {msg}")
    NOTES.append(msg)


def to16(x):
    return f64_to_f16_bits(np.asarray(x, dtype=np.float64)).reshape(np.asarray(x).shape)


def run_case(name, q, K16, V16, tier, G, oi=()):
    """Run chain + reference + THEIR gates; return (row, r, ref)."""
    r = at.attention_core(np.asarray(q, dtype=np.float64), K16, V16, tier, G=G,
                          outlier_idx=list(oi))
    refc = at.attention_ref_core(q, K16, V16)
    base = len(ta.FAILS)
    try:
        row = ta.derive_and_gate(f"audit/{name}/{tier}", r, refc)
    except ZeroDivisionError as e:
        del ta.FAILS[base:]
        note(f"{name}/{tier}: THEIR derive_and_gate CRASHES ({e!r}) on legal "
             "input with y_ref==0 (ynorm division unguarded)")
        return None, r, refc
    del ta.FAILS[base:]
    gate_fails = row["fails"]
    vnorm = max(float(np.max(np.abs(refc.V_ref))), 1e-12)
    err = float(np.max(np.abs(r.out_hat - refc.y_ref)))
    pmax = int(np.max(r.p_q115))
    psum = int(np.sum(r.p_q115))
    print(f"    {name}/{tier}: T={r.T} D={r.D} e2e_abs={err:.3e} "
          f"e2e/|V|={err / vnorm:.3e} pmax={pmax} psum={psum} "
          f"their_gates={'CLEAN' if not gate_fails else gate_fails}")
    check(f"{name}/{tier} their budget gates hold on adversarial data",
          not gate_fails, str(gate_fails))
    return row, r, refc


def main():
    rng = np.random.default_rng(0xA0D17)

    # ── 1/10. single token ───────────────────────────────────────────────────
    print("[1] single token T=1")
    D = 64
    K = rng.normal(0, 1, (1, D)); K *= 2.0 / np.max(np.abs(K))
    V = rng.normal(0, 1, (1, D)); V *= 2.0 / np.max(np.abs(V))
    q = f16_bits_to_f64(to16(rng.normal(0, 1, D))).reshape(D)
    for tier in at.TIERS:
        row, r, refc = run_case("T1", q, to16(K), to16(V), tier, 128, (1, 2))
        pm = int(np.max(r.p_q115))
        check(f"T1/{tier} p fits UQ1.15 (<= 32768; asu_softmax.sv contract)",
              pm <= 32768, f"pmax={pm}")
        if pm == 32768:
            note(f"T1/{tier}: p reaches 32768 (=1.0). asu_softmax.sv documents "
                 "UNSIGNED Q1.15 0..32768 so RTL holds it, but apex_pkg says "
                 "only 'Q1.15' — P-requant seam must accept 32768; never "
                 "exercised by the shipped T>=16 grid")

    # ── 2. exactly tied scores ───────────────────────────────────────────────
    print("[2] exactly tied scores (identical K rows), T=64")
    row = rng.normal(0, 1, D); row *= 2.0 / np.max(np.abs(row))
    K = np.tile(row, (64, 1))
    V = rng.normal(0, 1, (64, D)); V *= 2.0 / np.max(np.abs(V))
    for tier in at.TIERS:
        _, r, refc = run_case("tied", q, to16(K), to16(V), tier, 128, (0, 1))
        psum = int(np.sum(r.p_q115))
        check(f"tied/{tier} sum(p) within [2^15-T, 2^15]",
              2 ** 15 - r.T <= psum <= 2 ** 15, f"sum={psum}")

    # ── 3. near-tie (1 ulp) ──────────────────────────────────────────────────
    print("[3] near-tie: half the rows 1 fp16-ulp higher")
    K16 = to16(np.tile(row, (64, 1)))
    K16[32:] = (K16[32:].astype(np.int64) + np.where(K16[32:] & 0x8000, -1, 1)
                ).astype(np.uint16)  # +1 ulp toward +inf
    for tier in at.TIERS:
        run_case("neartie", q, K16, to16(V), tier, 128, (0, 1))

    # ── 4. steeply ascending scores (max rescale churn) ─────────────────────
    print("[4] ascending scores: T=130 rescale-every-token")
    T = 130
    base_row = rng.normal(0, 1, D); base_row /= np.linalg.norm(base_row)
    K = np.array([base_row * (0.02 * (t + 1)) for t in range(T)])
    K *= 4.0 / np.max(np.abs(K))
    V = rng.normal(0, 1, (T, D)); V *= 2.0 / np.max(np.abs(V))
    qq = f16_bits_to_f64(to16(base_row * 2.0)).reshape(D)
    for tier in at.TIERS:
        run_case("ascend", qq, to16(K), to16(V), tier, 128, (0, 1))
    print("[4b] descending scores")
    for tier in at.TIERS:
        run_case("descend", qq, to16(K[::-1].copy()), to16(V), tier, 128, (0, 1))

    # ── 5. dominant score: all others below the exp LUT floor ───────────────
    print("[5] dominant score, rest underflow exp LUT (T=32)")
    T = 32
    K = np.tile(-base_row * 4.0, (T, 1))
    K[7] = base_row * 4.0                       # one aligned row
    V = rng.normal(0, 1, (T, D)); V *= 2.0 / np.max(np.abs(V))
    for tier in at.TIERS:
        _, r, refc = run_case("dominant", qq, to16(K), to16(V), tier, 128, (0, 1))
        check(f"dominant/{tier} p fits SIGNED Q1.15",
              int(np.max(r.p_q115)) <= 32767,
              f"pmax={int(np.max(r.p_q115))}")

    # ── 6. extreme outliers x1000, CQ-4 UNMASKED ─────────────────────────────
    print("[6] outlier channels x1000 (legal fp16), CQ-4 unmasked vs CQ-4+ masked")
    T = 128
    K = rng.normal(0, 0.02, (T, D))
    K[:, 5] = rng.normal(0, 20.0, T)            # x1000 scale channels
    K[:, 50] = rng.normal(0, 20.0, T)
    V = rng.normal(0, 1, (T, D)); V *= 2.0 / np.max(np.abs(V))
    qo = rng.normal(0, 1, D); qo *= 1.0 / np.max(np.abs(qo))
    qo = f16_bits_to_f64(to16(qo)).reshape(D)
    res = {}
    for tier in at.TIERS:
        _, r, refc = run_case("outlier1000", qo, to16(K), to16(V), tier, 128, (5, 50))
        vnorm = max(float(np.max(np.abs(refc.V_ref))), 1e-12)
        res[tier] = float(np.max(np.abs(r.out_hat - refc.y_ref))) / vnorm
    note(f"outlier1000 e2e/|V|: CQ-8={res['CQ-8']:.3e} CQ-4={res['CQ-4']:.3e} "
         f"CQ-4+={res['CQ-4+']:.3e} (CQ-4 unmasked degradation factor "
         f"{res['CQ-4'] / max(res['CQ-8'], 1e-12):.1f}x)")

    # ── 7. all-zero K / V (EPS path) ─────────────────────────────────────────
    print("[7] all-zero K, then all-zero V")
    Z = np.zeros((16, D))
    Vn = rng.normal(0, 1, (16, D)); Vn *= 2.0 / np.max(np.abs(Vn))
    for tier in at.TIERS:
        run_case("zeroK", q, to16(Z), to16(Vn), tier, 128, (0, 1))
        run_case("zeroV", q, to16(Vn), to16(Z), tier, 128, (0, 1))

    # ── 8. partial-group straddles ───────────────────────────────────────────
    print("[8] partial-group straddles")
    for (T, G) in ((65, 64), (127, 128), (129, 128), (3, 128)):
        K = rng.normal(0, 1, (T, D)); K *= 3.0 / np.max(np.abs(K))
        V = rng.normal(0, 1, (T, D)); V *= 2.0 / np.max(np.abs(V))
        for tier in at.TIERS:
            run_case(f"straddle_T{T}_G{G}", q, to16(K), to16(V), tier, G, (0, 1))

    # ── 9. score INT32 envelope probe (legal fp16 magnitudes) ────────────────
    print("[9] score-dequant INT32 envelope (legal fp16 inputs, |K|~2000)")
    T = 16
    K = np.full((T, D), 2000.0) + rng.normal(0, 10.0, (T, D))   # aligned rows
    V = rng.normal(0, 1, (T, D))
    qb = f16_bits_to_f64(to16(np.full(D, 2000.0))).reshape(D)
    try:
        at.attention_core(qb, to16(K), to16(V), at.TIER_CQ8, G=128)
        note("large-magnitude case did NOT overflow (envelope wider than probed)")
    except AssertionError as e:
        note(f"score-dequant INT32 assert TRIPS on legal fp16 magnitudes: '{e}' "
             "-> RTL saturation semantics + max-|K|,|q| envelope are UNDERIVED")

    # ── 11. brute-force the [Q] hard factors ─────────────────────────────────
    print("[11] brute-force [Q] hard factors (0.57s INT8 / 0.51s INT4)")
    worst8 = 0.0
    for _ in range(2000):
        X = rng.normal(0, rng.uniform(0.01, 100), (1, 64))
        Xv = f16_bits_to_f64(to16(X)).reshape(1, 64)
        c8, s8 = at.quant_rows_i8(Xv)
        sv = float(f16_bits_to_f64(np.array([s8[0]]))[0])
        worst8 = max(worst8, float(np.max(np.abs(c8[0] * sv - Xv[0]))) / sv)
    check("[Q] INT8 factor 0.57 sound (measured worst |x̂−x|/s)",
          worst8 <= 0.57, f"worst={worst8:.4f}")
    worst4 = 0.0
    for _ in range(2000):
        X = rng.normal(0, rng.uniform(0.01, 100), (4, 64))
        X16 = to16(X)
        vb = cq.compress_values(X16, 4)
        Xh = f32_bits_to_f64(cq.decompress_values(vb)).reshape(4, 64)
        Xv = f16_bits_to_f64(X16).reshape(4, 64)
        sc = f16_bits_to_f64(vb.scales)
        worst4 = max(worst4, float(np.max(np.abs(Xh - Xv) / sc[:, None])))
    check("[Q] INT4 factor 0.51 sound (measured worst codec |x̂−x|/s)",
          worst4 <= 0.51, f"worst={worst4:.4f}")

    print("=" * 78)
    for n in NOTES:
        print(f"NOTE: {n}")
    if FAILS:
        print(f"AUDIT: {len(FAILS)} adversarial check(s) FAILED:")
        for f in FAILS:
            print(f"  ✗ {f}")
        return 1
    print("AUDIT: all adversarial checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
