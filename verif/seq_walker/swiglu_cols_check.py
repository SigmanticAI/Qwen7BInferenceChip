#!/usr/bin/env python3
"""swiglu_cols_check.py — the SwiGLU cols-truncation regression (2026-07-30
adversarial audit, "LIVE SILENT-WRONG-ANSWER DEFECT").

THE DEFECT (quoted from the audit):

    apex_top.sv:983    logic [11:0] lj_cols_q;        <- 12-bit LAYER_JOB cols
    apex_top.sv:1234   asu_swiglu #(.COLS_MAX(64))    <- CW = clog2(65) = 7 b
    apex_top.sv:1237   .jb_cols (7'(lj_cols_q)),      <- 12 -> 7 bit TRUNCATION

    "The walker chunks SwiGLU at the FIELD bound (seq_walker_fmt.py:88
    N_JOB = 4095), not the CONSUMER bound (64). lu_chunks(18944) =
    [4095,4095,4095,4095,2564]: the first four truncate to 127 (>64 ->
    job_error, loud), but the tail chunk 2564 truncates to 4 — silently
    legal. asu_swiglu would compute 4 columns instead of 2564 and raise no
    error."

WHAT THIS CHECKS. One real Qwen2.5-7B FFN row (d_ffn = 18944) is pushed
through the host's OWN chunking rule, then through a model of the delivery
path the audit names — the 12-bit LAYER_JOB field, apex_top's 7-bit slice,
and asu_swiglu's `jb_cols == 0 || jb_cols > COLS_MAX -> job_error` arm
(asu_swiglu.sv:221) — and the products the unit would emit are compared
element-by-element against the GOLDEN SwiGLU chain
(apex_golden.transformer.silu_apply * up, one f16 RNE — the same arbiter
verif/asu/swiglu/gen_swiglu_vectors.py uses).

The per-element arithmetic is NOT re-derived here; the S3 suite gates that
exhaustively. What is gated here is WHICH elements the layer computes at
all, which is exactly what a truncated cols destroys, and the silence with
which it destroys them.

    --bound consumer  (default) the fixed rule: chunk at asu_swiglu's own
                      COLS_MAX. 296 x 64, full coverage, bit-exact -> PASS
    --bound field     the pre-fix rule: chunk at the 12-bit field. 4 loud
                      refusals + ONE SILENTLY ACCEPTED truncation that
                      computes 4 of 2564 columns -> FAIL (this is the
                      red-before half of the regression; the exit code is
                      1 and the banner names the silent job)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "golden"))

import numpy as np  # noqa: E402

import seq_walker_fmt as fmt  # noqa: E402
from apex_golden import transformer as tf  # noqa: E402
from apex_golden.fp import f64_to_f16_bits  # noqa: E402
from apex_golden.weight_codec import f16_grade  # noqa: E402

D_FFN = 18944                     # Qwen2.5-7B intermediate size
LJ_FIELD_W = 12                   # apex_top.sv:983  logic [11:0] lj_cols_q
SWG_PORT_W = 7                    # apex_top.sv:1237 .jb_cols (7'(lj_cols_q))
SWG_COLS_MAX = 64                 # apex_top.sv:1234 asu_swiglu #(.COLS_MAX(64))


def f32_val(bits: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", bits))[0])


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def golden_products(acc_g, acc_u, cg_bits, cu_bits) -> np.ndarray:
    """p[i] = f16( silu_apply(acc_g[i]*comp_g) * (acc_u[i]*comp_u) ) — the
    gen_swiglu_vectors.py arbiter, vectorized. Both dequants are float64-
    EXACT because the composites are fp16-graded (the D-030 lemma), which is
    asserted rather than assumed."""
    cg, cu = f32_val(cg_bits), f32_val(cu_bits)
    g = acc_g.astype(np.float64) * np.float64(cg)
    u = acc_u.astype(np.float64) * np.float64(cu)
    for a, c, prod in ((acc_g, cg, g), (acc_u, cu, u)):
        chk = np.array([float(np.float64(int(x)) * np.float64(c))
                        for x in a[:256]], dtype=np.float64)
        assert np.array_equal(chk, prod[:256]), "dequant not f64-exact"
    return f64_to_f16_bits(tf.silu_apply(g) * u)


def deliver(cols: int) -> tuple[int, bool]:
    """The apex_top -> asu_swiglu delivery path for one LAYER_JOB push.

    Returns (cols_the_unit_computes, silently_wrong). The host writes `cols`
    into the 12-bit field; apex_top hands asu_swiglu the low SWG_PORT_W bits;
    asu_swiglu refuses 0 or >COLS_MAX with job_error (loud) and otherwise
    accepts — including when what it accepted is a truncated stranger."""
    assert 1 <= cols < (1 << LJ_FIELD_W), f"cols {cols} does not fit the field"
    seen = cols & ((1 << SWG_PORT_W) - 1)
    if seen == 0 or seen > SWG_COLS_MAX:
        return 0, False                      # job_error pulse + sticky: LOUD
    return seen, seen != cols                # accepted; silent iff truncated


def run(bound: str) -> int:
    rng = np.random.default_rng(0x5761C10)
    acc_g = rng.integers(-(1 << 18), 1 << 18, size=D_FFN, dtype=np.int64)
    acc_u = rng.integers(-(1 << 18), 1 << 18, size=D_FFN, dtype=np.int64)
    cg_bits = f32_bits(f16_grade(2.7136e-4))     # s_h2 * s_wg, C2-graded
    cu_bits = f32_bits(f16_grade(3.0517e-5))     # s_h2 * s_wu
    for b in (cg_bits, cu_bits):
        assert (b & 0x1FFF) == 0 and 0 < ((b >> 23) & 0xFF) < 0xFF

    exp = golden_products(acc_g, acc_u, cg_bits, cu_bits)

    if bound == "consumer":
        chunks = fmt.lu_chunks(D_FFN, fmt.LU_SWIGLU)
    else:                                        # the pre-fix rule, verbatim
        chunks = [min(fmt.N_JOB, D_FFN - c)
                  for c in range(0, D_FFN, fmt.N_JOB)]
    print(f"host chunking ({bound} bound): {len(chunks)} LAYER_JOB pushes, "
          f"first={chunks[0]} last={chunks[-1]} sum={sum(chunks)}")

    got = np.full(D_FFN, -1, dtype=np.int64)
    ptr, refused, silent = 0, 0, []
    for i, c in enumerate(chunks):
        n, quiet = deliver(c)
        if n == 0:
            refused += 1
            continue                             # no stream consumed
        if quiet:
            silent.append((i, c, n))
        take = min(n, D_FFN - ptr)
        got[ptr:ptr + take] = exp[ptr:ptr + take]   # the unit's own math
        ptr += take

    covered = int((got >= 0).sum())
    mism = int((got[:covered] != exp[:covered]).sum())
    print(f"delivery: {refused} jobs refused with job_error (loud), "
          f"{len(silent)} jobs SILENTLY truncated, {covered}/{D_FFN} columns "
          f"computed, {mism} mismatches vs golden over the computed ones")
    for (i, c, n) in silent:
        print(f"  SILENT: push[{i}] cols={c} -> asu_swiglu sees "
              f"{c} & 0x{(1 << SWG_PORT_W) - 1:02X} = {n}: it computes {n} of "
              f"{c} columns and raises NO error")

    fails = []
    if covered != D_FFN:
        fails.append(f"{D_FFN - covered} of {D_FFN} FFN columns never "
                     f"computed")
    if mism:
        fails.append(f"{mism} computed columns differ from golden")
    if silent:
        fails.append(f"{len(silent)} LAYER_JOB push(es) accepted a TRUNCATED "
                     f"cols — a wrong answer with no job_error")
    if fails:
        print("SWIGLU COLS CHECK: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1

    # the fixed rule must also be exact and complete, not merely legal
    assert np.array_equal(got, exp) and len(chunks) == D_FFN // SWG_COLS_MAX
    print(f"SWIGLU COLS CHECK: PASS (d_ffn={D_FFN} split {len(chunks)}x"
          f"{SWG_COLS_MAX} at asu_swiglu's own COLS_MAX; every chunk survives "
          f"the {LJ_FIELD_W}->{SWG_PORT_W} bit port unchanged; {covered}/"
          f"{D_FFN} columns computed, bit-exact vs golden silu_apply*up, "
          f"0 silent truncations)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bound = "consumer"
    while argv:
        a = argv.pop(0)
        if a == "--bound":
            bound = argv.pop(0)
            assert bound in ("consumer", "field"), bound
        else:
            print(f"unknown argument {a!r}")
            return 2
    return run(bound)


if __name__ == "__main__":
    sys.exit(main())
