#!/usr/bin/env python3
"""gen_resid_vectors.py — apex_residual DATAPATH closure vectors (IB-LAYER
S5, closing the S4 deferral).

Arbiter: golden decoder_layer_fx(bus=BUS_ON) — D-030 signed. The unit
consumes EXACTLY the operand pairs the tile will: the resident fp16 row
(BUS_ON X[T] on the fp16 grid) plus the EXACT-fp32 sublayer stream
(attn_proj = o8*graded, ffn_out likewise), and must land r1 = f16(X+attn_proj)
then r2 = f16(r1+ffn_out) BIT-EXACT — chained in-place, per case.

Classes:
  tensor : all six L4 cases, both residual jobs chained (r1 then r2),
           every row element checked (the datapath closure proper)
  corner : +-0 sign rules, window-refusal b (grid finer than 2^-38, via
           tiny-exponent f32), inf shortcut (e8 >= 144, both signs),
           window-boundary exact hits (sh_b + tz == 0), frame errors,
           cols-0 / cols>DM_MAX geometry rejects

File format (build/vectors_resid.txt):
  CASE <name> <cols>
  ROW  <cols fp16 hex lines>          initial row RAM load
  JOB  <mode>                          then <cols> fp32 b lines (except
                                       REFUSE_GEO*, which drives nothing);
                                       EXP: <cols> fp16 lines for OK jobs,
                                       none for refusal/frame modes
       modes: OK | WREFUSE_<i> (window refusal at beat i; beats before i
       applied in place) | FRAMEE (early last, 1 beat) | GEO_ZERO |
       GEO_BIG (cols pushed = DM_MAX+1, no beats)
  ENDCASE
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "golden"))
sys.path.insert(0, str(_REPO / "verif" / "top" / "l4"))

import numpy as np  # noqa: E402

from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402
from l4_cases import CASES, TIER, build_case, tf  # noqa: E402


def f32b(v):
    b = struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]
    assert float(np.float32(v)) == float(v), "operand not fp32-exact"
    return b


def f16b(v):
    return int(f64_to_f16_bits(np.array([v], dtype=np.float64))[0])


def f16v(bits):
    return float(f16_bits_to_f64(np.array([bits], dtype=np.uint16))[0])


def res_expect(row_bits, b_f64):
    return f16b(f16v(row_bits) + b_f64)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
    out.mkdir(parents=True, exist_ok=True)
    n_case = n_job = n_elem = 0
    with (out / "vectors_resid.txt").open("w") as fh:
        # ── tensor closure: the six L4 cases, r1 then r2 chained ─────────────
        for (name, *_r) in [(c[0],) for c in CASES]:
            X, w, q_pos = build_case(name)
            r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
            cols = r.D_model
            xrow = [int(v) for v in f64_to_f16_bits(np.asarray(X)[-1])]
            fh.write(f"CASE {name} {cols}\n")
            fh.write("ROW\n")
            for bts in xrow:
                fh.write(f"{bts:04x}\n")
            # job 1: attn_proj -> r1 (checked vs the ARBITER r.r1)
            fh.write("JOB OK\n")
            exp1 = []
            for i in range(cols):
                fh.write(f"{f32b(r.attn_proj[i]):08x}\n")
                e = res_expect(xrow[i], float(r.attn_proj[i]))
                assert e == f16b(float(r.r1[i])), f"{name} r1[{i}] arbiter miss"
                exp1.append(e)
            fh.write("EXP\n")
            for e in exp1:
                fh.write(f"{e:04x}\n")
            # job 2 (chained, in place): ffn_out -> r2 (vs the ARBITER r.r2)
            fh.write("JOB OK\n")
            exp2 = []
            for i in range(cols):
                fh.write(f"{f32b(r.ffn_out[i]):08x}\n")
                e = res_expect(exp1[i], float(r.ffn_out[i]))
                assert e == f16b(float(r.r2[i])), f"{name} r2[{i}] arbiter miss"
                exp2.append(e)
            fh.write("EXP\n")
            for e in exp2:
                fh.write(f"{e:04x}\n")
            fh.write("ENDCASE\n")
            n_case += 1
            n_job += 2
            n_elem += 2 * cols

        # ── corners ──────────────────────────────────────────────────────────
        fh.write("CASE corners 8\n")
        fh.write("ROW\n")
        crow = [0x8000, 0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x7BFF, 0x03FF]
        for bts in crow:
            fh.write(f"{bts:04x}\n")
        # zero-sign battery: b = -0, +0, tiny; exact cancel; window-boundary
        fh.write("JOB OK\n")
        bvals = [struct.unpack("<f", struct.pack("<I", 0x80000000))[0],  # -0
                 0.0,                                                    # +0
                 0.0,                     # (-0)+(+0) -> +0 (IEEE add)
                 -1.0,                                                   # cancel -> +0
                 float(np.float32(2.0 ** -35)),   # sh_b+tz boundary-legal
                 float(np.float32(2.0 ** -38)),   # exact window floor
                 float(np.float32(64.0)),
                 float(np.float32(-2.0 ** -24))]
        exp = []
        for i, bv in enumerate(bvals):
            fh.write(f"{f32b(bv):08x}\n")
            exp.append(res_expect(crow[i], bv))
        # pin the zero-sign truths (the module contract, IEEE add rules)
        assert exp[0] == 0x8000, "(-0)+(-0) must be -0"
        assert exp[1] == 0x0000 and exp[2] == 0x0000
        assert exp[3] == 0x0000, "exact cancel must be +0"
        fh.write("EXP\n")
        for e in exp:
            fh.write(f"{e:04x}\n")
        # window refusal at beat 3 (grid 2^-60: unrepresentable on 2^-38)
        fh.write("JOB WREFUSE_3\n")
        pre = [1.0, 2.0, -0.5]
        for v in pre:
            fh.write(f"{f32b(v):08x}\n")
        fh.write(f"{f32b(float(np.float32(2.0 ** -60 * 1.5))):08x}\n")
        for v in (3.0, 4.0, 5.0, 6.0):
            fh.write(f"{f32b(v):08x}\n")
        fh.write("EXP\n")                        # the applied prefix only
        for i, v in enumerate(pre):
            fh.write(f"{res_expect(exp[i], v):04x}\n")
        fh.write("ENDCASE\n")
        n_case += 1
        n_job += 2
        n_elem += 11

        # inf shortcut + frame/geometry rejects
        fh.write("CASE guards 4\n")
        fh.write("ROW\n")
        grow = [0x3C00, 0xFBFF, 0xFBFF, 0x8000]
        for bts in grow:
            fh.write(f"{bts:04x}\n")
        fh.write("JOB OK\n")
        gb = [float(np.float32(2.0 ** 20 * 1.31)),      # e8=147 -> +inf
              float(np.float32(-(2.0 ** 17))),          # e8=144 boundary -> -inf
              float(np.float32(2.0 ** 16)),             # e8=143: window path —
                                                        # -65504 + 65536 = 32,
                                                        # FINITE iff no shortcut
              float(np.float32(-0.75))]
        gexp = []
        for i, bv in enumerate(gb):
            fh.write(f"{f32b(bv):08x}\n")
            gexp.append(res_expect(grow[i], bv))
        assert gexp[0] == 0x7C00 and gexp[1] == 0xFC00, "inf shortcut wrong"
        assert gexp[2] == f16b(32.0), "e8=143 must take the window path (finite 32.0)"
        fh.write("EXP\n")
        for e in gexp:
            fh.write(f"{e:04x}\n")
        fh.write("JOB FRAMEE\n")                 # early last at beat 0 (cols 4)
        fh.write(f"{f32b(1.0):08x}\n")
        fh.write("JOB GEO_ZERO\n")
        fh.write("JOB GEO_BIG\n")
        fh.write("ENDCASE\n")
        n_case += 1
        n_job += 4
        n_elem += 4

    print(f"RESID VECTORS: cases={n_case} jobs={n_job} elements={n_elem} "
          f"(tensor jobs arbiter-asserted r1/r2 bit-exact at gen time)")


if __name__ == "__main__":
    main()
