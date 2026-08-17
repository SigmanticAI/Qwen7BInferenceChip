#!/usr/bin/env python3
"""gen_asu_vectors.py — ASU smoke vector generator (provenance record).

Everything the smoke TBs compare against is produced HERE, from the golden
arbiter apex_golden.compute (imported, never retyped):

  build/exp_expected.hex   exp_fx(clamp(z)) for ALL 65536 16-bit input
                           patterns (address = unsigned bit pattern of z_q),
                           4 hex digits/line — the exhaustive-domain oracle
                           for asu_exp_lut (below -16384 clamps to the floor
                           entry, above 0 clamps to exp(0) = 32768).
  build/asu_vectors.svh    softmax rows (inputs + online_softmax_fx expected)
                           and RMSNorm vectors (x, gamma, rmsnorm_fx expected
                           + expected r / norm_int), as SV localparams.

rmsnorm_fx below is the NORMATIVE golden mirror of rtl/asu/asu_rmsnorm.sv
(the exact pseudocode from that module's header): sum of x^2, mean by exact
pow-2 shift, +1 eps, r = 8192 // isqrt(n2) (the vendored rsqrt_unit contract:
floor division by the FLOOR integer sqrt), per-element x*gamma*r with
RNE(>>18) to Q7.8 and sat16.

Run:
    python3 \
        gen_asu_vectors.py --outdir build
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
import numpy as np  # noqa: E402

from apex_golden.compute import (  # noqa: E402
    FRAC_BITS, OUT_FRAC, exp_fx, online_softmax_fx,
)


# ── RMSNorm golden mirror (normative — mirrors asu_rmsnorm.sv header) ────────

def rmsnorm_fx(x: list[int], g: list[int]) -> tuple[list[int], int, int]:
    d = len(x)
    assert 1 <= d <= 128 and (d & (d - 1)) == 0, "D must be pow-2 in [1,128]"
    assert all(-128 <= xi <= 127 for xi in x)
    assert all(-32768 <= gi <= 32767 for gi in g) and len(g) == d
    sum2 = sum(int(xi) ** 2 for xi in x)
    assert sum2 < 2 ** 22                       # width proof (<= 2^21)
    mean2 = sum2 >> int(math.log2(d))           # exact floor (pow-2 D)
    n2 = mean2 + 1                              # EPS_INT = 1
    den = math.isqrt(n2)                        # rsqrt_unit phase A (floor)
    r = (1 << 13) // den                        # phase B (floor div), UQ2.13
    nrm = min(den, 2 ** 14 - 1)                 # norm_int output
    y = []
    for i in range(d):
        p = int(x[i]) * int(g[i])               # Q2.13
        q = p * r                               # frac 26
        t = q >> 18                             # floor to Q7.8
        rem = q & (2 ** 18 - 1)
        t += 1 if (rem > 2 ** 17 or (rem == 2 ** 17 and (t & 1))) else 0
        y.append(max(-32768, min(32767, t)))    # sat16
    return y, r, nrm


# ── SV emit helpers ──────────────────────────────────────────────────────────

def sv_arr(name: str, vals: list[int], width: int, signed_mask: bool) -> str:
    lines = [f"localparam logic [{width-1}:0] {name} [{len(vals)}] = '{{"]
    for i, v in enumerate(vals):
        h = (int(v) & ((1 << width) - 1)) if signed_mask else int(v)
        sep = "," if i < len(vals) - 1 else ""
        lines.append(f"  {width}'h{h:0{(width + 3) // 4}x}{sep}")
    lines.append("};")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ── exp LUT exhaustive oracle: all 65536 16-bit patterns ────────────────
    u = np.arange(65536, dtype=np.int64)
    z = np.where(u >= 32768, u - 65536, u)              # signed value
    z_cl = np.clip(z, -(16 << FRAC_BITS), 0)            # module's domain clamp
    exp_exp = exp_fx(z_cl)
    assert int(exp_exp.max()) == 32768 and int(exp_exp.min()) == 0
    with open(out / "exp_expected.hex", "w") as f:
        f.write("".join(f"{int(v):04x}\n" for v in exp_exp))

    # ── softmax rows (SCORE_FRAC = 0, raw INT32 — C-5) ──────────────────────
    # Row 0: n=24, values clustered within +/-15 of the max so probabilities
    # are non-trivial; max ascends twice (exercises the l-rescale), the final
    # max is DUPLICATED (equal-to-max path), and one score sits 16+ below the
    # final max (clamp -> exp = 0). Deterministic seed.
    rng = np.random.default_rng(20260707)
    row0 = list(rng.integers(985, 1000, size=20))
    row0.insert(3, 1004)          # first max step (rescale #1)
    row0.insert(9, 1009)          # second max step (rescale #2)
    row0.append(1009)             # duplicate of the final max
    row0.append(980)              # 29 below max -> clamped, p = 0
    row0 = [int(v) for v in row0]
    assert len(row0) == 24
    exp0 = [int(v) for v in online_softmax_fx(np.array(row0, dtype=np.int64), 0)]
    assert exp0[-1] == 0, "clamped element should quantize to 0"

    # Row 1: single element -> p = 32768 (0x8000, exactly 1.0)
    row1 = [-123456789]
    exp1 = [int(v) for v in online_softmax_fx(np.array(row1, dtype=np.int64), 0)]
    assert exp1 == [32768]

    # Row 2: INT32 extremes — full 33-bit diff arithmetic + rescale-to-clamp
    row2 = [-(2 ** 31), -(2 ** 31), 2 ** 31 - 1, 2 ** 31 - 1]
    exp2 = [int(v) for v in online_softmax_fx(np.array(row2, dtype=np.int64), 0)]
    assert exp2 == [0, 0, 16384, 16384]

    # ── RMSNorm vectors ─────────────────────────────────────────────────────
    # V0: D=64, full-range INT8 x (forced +/-128 extremes), gamma ~ +/-1.5 in
    # Q2.13 with sign flips. V1: D=1 edge case (x = -128).
    x0 = [int(v) for v in rng.integers(-128, 128, size=64)]
    x0[0], x0[1] = -128, 127                            # extremes
    g0 = [int(v) for v in rng.integers(-3 << 13, (3 << 13) - 1, size=64)]
    g0[0] = 1 << 13                                     # gamma = 1.0 lane
    y0, r0, nrm0 = rmsnorm_fx(x0, g0)

    # Engineer exact RNE ties (rem == 2^17) into two lanes: one with an ODD
    # floor quotient (RNE rounds up — catches truncation mutants) and one
    # with an EVEN floor quotient (RNE holds — catches round-half-up
    # mutants). gamma is a free parameter (r depends only on x), so brute-
    # force gamma per lane. Without these, the smoke row cannot distinguish
    # the C-2 tie rule (verified by mutation during bring-up).
    def find_tie(want_odd: bool, skip: set[int]) -> tuple[int, int]:
        """Return (lane, gamma) whose product hits rem == 2^17 with the
        requested quotient parity. A tie needs x*g*r == 2^17 (mod 2^18),
        solvable within int16 gamma only for some lanes — so scan lanes."""
        for j in range(2, len(x0)):                 # keep lanes 0/1 (extremes)
            if j in skip or x0[j] == 0:
                continue
            for g in range(-32768, 32768):
                q = x0[j] * g * r0
                if (q & (2 ** 18 - 1)) == 2 ** 17 \
                        and ((q >> 18) & 1) == int(want_odd):
                    return j, g
        raise AssertionError(f"no tie lane found (odd={want_odd})")

    j_odd, g_odd = find_tie(want_odd=True, skip=set())
    j_even, g_even = find_tie(want_odd=False, skip={j_odd})
    g0[j_odd], g0[j_even] = g_odd, g_even
    y0, r0b, nrm0b = rmsnorm_fx(x0, g0)
    assert (r0b, nrm0b) == (r0, nrm0)               # r depends only on x
    # prove the ties discriminate the C-2 rule from its mutants
    q_o = x0[j_odd] * g_odd * r0
    q_e = x0[j_even] * g_even * r0
    assert y0[j_odd] == (q_o >> 18) + 1, "odd tie must round up under RNE"
    assert y0[j_even] == (q_e >> 18), "even tie must hold under RNE"
    print(f"rmsnorm    : RNE ties engineered at lanes {j_odd} (odd, rounds up)"
          f" and {j_even} (even, holds)")

    x1 = [-128]
    g1 = [3 << 12]                                      # gamma = 1.5
    y1, r1, nrm1 = rmsnorm_fx(x1, g1)

    # ── emit per-TB .svh (split so each TB uses every localparam — -Wall) ───
    hdr = [
        "// GENERATED by gen_asu_vectors.py from the golden arbiter",
        "// apex_golden.compute (+ rmsnorm_fx mirror). DO NOT EDIT.",
        "",
    ]
    sm = hdr + [
        f"localparam int unsigned SM_N0 = {len(row0)};",
        sv_arr("SM_IN0", row0, 32, True),
        sv_arr("SM_EXP0", exp0, 16, False),
        f"localparam int unsigned SM_N1 = {len(row1)};",
        sv_arr("SM_IN1", row1, 32, True),
        sv_arr("SM_EXP1", exp1, 16, False),
        f"localparam int unsigned SM_N2 = {len(row2)};",
        sv_arr("SM_IN2", row2, 32, True),
        sv_arr("SM_EXP2", exp2, 16, False),
        "",
    ]
    (out / "asu_vectors_softmax.svh").write_text("\n".join(sm))

    rms = hdr + [
        f"localparam int unsigned RMS_D0 = {len(x0)};",
        sv_arr("RMS_X0", x0, 8, True),
        sv_arr("RMS_G0", g0, 16, True),
        sv_arr("RMS_EXP0", y0, 16, True),
        f"localparam logic [14:0] RMS_R0   = 15'd{r0};",
        f"localparam logic [13:0] RMS_NRM0 = 14'd{nrm0};",
        f"localparam int unsigned RMS_D1 = {len(x1)};",
        sv_arr("RMS_X1", x1, 8, True),
        sv_arr("RMS_G1", g1, 16, True),
        sv_arr("RMS_EXP1", y1, 16, True),
        f"localparam logic [14:0] RMS_R1   = 15'd{r1};",
        f"localparam logic [13:0] RMS_NRM1 = 14'd{nrm1};",
        "",
    ]
    (out / "asu_vectors_rmsnorm.svh").write_text("\n".join(rms))

    print(f"exp oracle : 65536 entries -> {out/'exp_expected.hex'}")
    print(f"softmax    : rows n={len(row0)},{len(row1)},{len(row2)} "
          f"(sum p0 = {sum(exp0)}/32768)")
    print(f"rmsnorm    : D={len(x0)} r={r0} nrm={nrm0}; D={len(x1)} r={r1} "
          f"nrm={nrm1}")


if __name__ == "__main__":
    main()
