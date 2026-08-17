#!/usr/bin/env python3
"""gen_l4_vectors.py — L4 (full decoder-layer through the REAL apex_top,
host mode) vector generator. IB-LAYER S1 SKELETON.

WHAT IT DOES TODAY (S1): runs the golden arbiter decoder_layer_fx with
bus=BUS_ON (C-LBUS/D-030) on the l4_cases.py case set and emits, per case,
the bit-exact CHECKPOINT tensors the L4 TB will compare at the tile
boundaries (IB_LAYER.md §5), plus a manifest with per-case checkpoint-word
counts and a determinism self-check (two runs, byte-compared).

WHAT IT DOES **NOT** DO YET (lands at S4/S5, flagged loudly here so nobody
mistakes the skeleton for the suite):
  * emit the HOST OP STREAM (CSRW/ROUTE/job/LAYER-window writes) — the op
    vocabulary firms up with the stage-4 CSR window shape (IB_LAYER.md §3);
    the emitter slot is `emit_ops()` below, currently a stub that emits a
    single header line and counts nothing.
  * schedule C-KSPLIT sub-jobs (K > K_MAX never occurs at tile scale).

SEGMENT-1 (o-proj -> r1) — LANDED (Option 1, coordinator-confirmed 2026-07-26):
the o-proj epilogue internals decoder_layer_fx discards (o8, rq scale/shift,
graded s_out) are RE-DERIVED via PUBLIC golden functions (rederive_oproj) with
ZERO golden edits, and seg1_selfcheck ASSERTS the re-derived chain reproduces
r.attn_proj AND r.r1 bit-exactly BEFORE any RTL runs. Per case a `.seg1`
reference (a8/o8/xrow/r1 + scale/shift/s_out) is emitted for tb_l4_compose's
deq->residual->RDATA tail. NOTE the tile-drive of the o-proj GEMM needs the
attention back-half to supply its `attn` input (the feeder has no host act
inject); see IB_LAYER.md §3b L4-harness addendum.

GATE STATUS: BUS_ON is design-approved (R4) with OWNER SIGN-OFF PENDING —
these vectors must not gate RTL until docs/design/IB_LAYER.md records the
sign-off.

Usage:  python3 gen_l4_vectors.py [--outdir build]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from l4_cases import CASES, TIER, build_case, tf

from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402  (path via l4_cases)


def rederive_oproj(attn, W8, s_w):
    """Re-derive the o-proj epilogue internals decoder_layer_fx DISCARDS
    (transformer.py:563 keeps only attn_proj), via PUBLIC golden functions —
    an exact inline replica of transformer._proj_epilogue(..., grade=True),
    ZERO golden edits (IB_LAYER.md §3b L4 harness spec, Option 1). Exposes
    a8/s_a (the C-1 activation feed), (scale,shift) (the MXE requant descriptor
    fields), o8 (the requantized serializer codes) and the fp16-graded s_out
    (the layer-deq JOBC composite)."""
    a8, s_a = tf.quant_rows_i8(np.asarray(attn, dtype=np.float64)[None, :])
    acc = tf.gemm_i8_ksplit(a8, np.asarray(W8, dtype=np.int64))[0].astype(np.int64)
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = tf.calib_requant(amax)
    o8 = tf.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    s_a_val = float(f16_bits_to_f64(np.array([s_a[0]]))[0])
    s_out = tf.f16_grade(s_a_val * float(s_w) * float(1 << shift) / float(scale))
    return {"a8": a8[0], "s_a": int(s_a[0]), "scale": int(scale),
            "shift": int(shift), "o8": o8, "s_out": s_out,
            "attn_proj": o8.astype(np.float64) * s_out}


def seg1_selfcheck(name, X, w, r):
    """MANDATORY pre-RTL gate (coordinator directive): the re-derived o-proj
    chain MUST reproduce r.attn_proj AND r.r1 bit-exactly from the LayerFx
    record. A failure means THIS generator is wrong, never golden. Returns
    the segment-1 reference dict the tb_l4_compose tail checks RDATA against."""
    d = rederive_oproj(r.attn, w.Wo, w.s_wo)
    assert np.array_equal(d["attn_proj"], r.attn_proj), f"{name}: attn_proj != arbiter"
    ap_fn, s_out_fn = tf._proj_epilogue(r.attn, w.Wo, w.s_wo, grade=True)
    assert np.array_equal(ap_fn, r.attn_proj) and s_out_fn == d["s_out"], \
        f"{name}: _proj_epilogue cross-check"
    xrow_bits = f64_to_f16_bits(np.asarray(X)[r.T])          # residual row = f16(X[T])
    r1 = tf._f16(f16_bits_to_f64(xrow_bits) + d["attn_proj"])
    assert np.array_equal(f64_to_f16_bits(r1), f64_to_f16_bits(r.r1)), \
        f"{name}: r1 != arbiter"
    # deq-exactness premises: o8 in [-127,127]; s_out positive-normal fp16-grade
    assert int(np.max(np.abs(d["o8"]))) < 128, f"{name}: o8 clip"
    sb = int(np.float64(d["s_out"]).astype(np.float32).view(np.uint32))
    assert (sb & 0x1FFF) == 0 and 0 < ((sb >> 23) & 0xFF) < 255, f"{name}: s_out not graded"
    d["s_out_bits"] = sb
    d["xrow_bits"] = [int(v) for v in xrow_bits]
    d["r1_bits"] = [int(v) for v in f64_to_f16_bits(r.r1)]
    return d


def seg2_selfcheck(name, X, w, r):
    """SEGMENT 2 (front-half xa->rmsnorm->feeder->astage) MANDATORY pre-RTL
    gate: re-derive x8 = quant_rows_i8(f16(X)) (the RMSNorm-1 input codes
    decoder_layer_fx computes at :485; LayerFx exposes no x8 — same re-derive
    discipline as segment 1, PUBLIC fns, zero golden edits) and assert the
    re-derived chain (x8 -> rmsnorm_fx_wide -> C-1 quant) reproduces r.h and
    r.s_h bit-exactly. s_h is the tile-checkable feeder-scale checkpoint
    (fs_* bus, EFS). Returns (x8[Tp1][D], gamma1[D], s_h_bits[Tp1])."""
    Xf = tf._f16(np.asarray(X, dtype=np.float64))          # BUS_ON x_f16 narrowing
    x8, _s_x = tf.quant_rows_i8(Xf)
    g1 = [int(v) for v in np.asarray(w.gamma1)]
    for t in range(r.T + 1):
        y, _, _ = tf.rmsnorm_fx_wide([int(v) for v in x8[t]], g1)
        assert np.array_equal(np.asarray(y, dtype=np.int64), r.h[t]), f"{name}: h[{t}] re-derive"
    h8_rd, s_h_rd = tf.quant_rows_i8(r.h.astype(np.float64) / 256.0)
    assert np.array_equal(s_h_rd, r.s_h), f"{name}: s_h re-derive"
    assert np.array_equal(h8_rd, r.h8), f"{name}: h8 re-derive"
    return x8, g1, [int(v) for v in r.s_h]


def write_compose_seg2(out: Path) -> dict:
    """Emit the tb_l4_compose SEGMENT-2 op stream (the full L4 harness: op
    stream + real apex_top + checkpoint compare). One combined file drives all
    cases' RMSNorm-1 front-halves through ONE CFG_D=128 tile; the EFS op checks
    the feeder scale == r.s_h per row. h8 bit-exactness is DEFERRED to segment 1
    (the o-proj GEMM consumes the h8 codes vs the re-derived o8, so any h8 error
    necessarily breaks the o8/r1 match — a real indirect check; IB_LAYER.md §3c)."""
    L = ["// tb_l4_compose SEGMENT 2 — RMSNorm-1 front-half xa->rmsnorm->feeder->astage",
         "// arbiter decoder_layer_fx(BUS_ON); CHECK: feeder scale fs == r.s_h (EFS).",
         "// h8 bit-exactness DEFERRED to segment 1 (disclosed carve-out, §3c).",
         "CSRW 00 00000001"]                                # tile enable (CTRL[0])
    n_chk = n_case = 0
    for (name, *_r) in [(c[0],) for c in CASES]:
        X, w, q_pos = build_case(name)
        r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
        assert r.D_model == 128, f"{name}: seg2 tile is CFG_D=128 (D_model={r.D_model})"
        x8, g1, s_h = seg2_selfcheck(name, X, w, r)
        Tp1, BPR = r.T + 1, r.D_model // 8
        L.append(f"// case {name}: T+1={Tp1} rows x D={r.D_model} (theta={w.rope_theta_base:g})")
        L.append("CSRP 04 00000001 00000001")              # tile quiescent
        L.append("ROUTE 0080")                             # fsrc=0 fdst=0 asrc=0 kvu=1
        L.append(f"AJ 0 0 0 {Tp1:x} {BPR:x} 0")            # act-stage LOAD (drains feeder codes)
        L.append(f"FJOB {Tp1:x}")                          # feeder job: Tp1 rows
        for t in range(Tp1):
            L.append("XR " + " ".join(f"{int(v) & 0xFF:02x}" for v in x8[t]))
            L.append("GR " + " ".join(f"{int(v) & 0xFFFF:04x}" for v in g1))
            L.append(f"EFS {s_h[t]:04x} {1 if t == Tp1 - 1 else 0}")
            n_chk += 1
        L.append("CSRP 04 00000001 00000001")              # back to quiescent
        n_case += 1
    L.append("DONE")
    (out / "compose_seg2.ops").write_text("\n".join(L) + "\n")
    return {"cases": n_case, "efs_checks": n_chk}


def emit_ops(case_name: str, fh) -> int:
    """S4 STUB — the host op stream emitter (CSR/route/job/LAYER window).

    Returns the number of emitted ops: 0 today. The L3-style drive
    vocabulary plus the IB_LAYER.md §3 LAYER-window ops land here once the
    stage-4 glue shape is frozen; until then the TB has nothing to replay
    and the suite CANNOT run — by design, the skeleton refuses to pretend.
    """
    fh.write(f"# OPS {case_name}: S4 stub — op stream not yet emitted\n")
    return 0


def hex16(fh, name, bits):
    fh.write(f"# {name} n={bits.size}\n")
    for b in np.asarray(bits, dtype=np.uint16).ravel():
        fh.write(f"{int(b):04x}\n")


def hex64(fh, name, vals):
    v = np.asarray(vals, dtype=np.float64)
    fh.write(f"# {name} n={v.size} (f64 bits)\n")
    for b in v.ravel().view(np.uint64):
        fh.write(f"{int(b):016x}\n")


def hexi(fh, name, vals, width):
    v = np.asarray(vals, dtype=np.int64)
    mask = (1 << (4 * width)) - 1
    fh.write(f"# {name} n={v.size} (two's-complement {4 * width}b)\n")
    for x in v.ravel():
        fh.write(f"{int(x) & mask:0{width}x}\n")


def run_case(name, outdir: Path) -> dict:
    X, w, q_pos = build_case(name)
    r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
    p = outdir / f"{name}.chk"
    with p.open("w") as fh:
        fh.write(f"# L4 CHECKPOINTS {name} (arbiter: decoder_layer_fx BUS_ON)\n")
        fh.write(f"# D_model={r.D_model} d_ffn={r.d_ffn} H={r.H} "
                 f"H_kv={r.H_kv} hd={r.head_dim} T={r.T} tier={r.tier} "
                 f"theta={w.rope_theta_base:g} q_pos={q_pos}\n")
        hexi(fh, "h_q78", r.h, 4)                       # RMSNorm-1 rows
        hexi(fh, "h8", r.h8, 2)                         # C-1 codes
        hex16(fh, "s_h", r.s_h)                         # C-1 fp16 row scales
        hex16(fh, "q_rope", f64_to_f16_bits(r.q_rope))  # S-2 bus, rotated
        hex16(fh, "K_rope", f64_to_f16_bits(r.K_rope))
        hex16(fh, "V_f16", f64_to_f16_bits(r.V_real))   # S-2 bus, NOT rotated
        for i, hd_ in enumerate(r.heads):
            hexi(fh, f"head{i}_o8", hd_.o8, 2)          # PV epilogue codes
        hex64(fh, "attn_bus", r.attn)                   # o8*graded (exact f64)
        hex64(fh, "attn_proj", r.attn_proj)
        hex16(fh, "r1", f64_to_f16_bits(r.r1))          # C-6 residual bus
        hexi(fh, "h2_q78", r.h2, 4)
        hex64(fh, "gate_real", r.gate_real)             # exact acc*graded
        hex64(fh, "up_real", r.up_real)
        hex16(fh, "swiglu", f64_to_f16_bits(r.swiglu))
        hex64(fh, "ffn_out", r.ffn_out)
        hex16(fh, "r2", f64_to_f16_bits(r.r2))          # the layer output
    ops = outdir / f"{name}.ops"
    with ops.open("w") as fh:
        n_ops = emit_ops(name, fh)
    # ── segment-1 (o-proj -> r1) reference + MANDATORY self-check ──────────────
    d = seg1_selfcheck(name, X, w, r)
    seg1 = outdir / f"{name}.seg1"
    with seg1.open("w") as fh:
        fh.write(f"# SEG1 {name} o-proj->r1 (arbiter: decoder_layer_fx BUS_ON; "
                 f"re-derived via public golden fns, r.attn_proj/r.r1 asserted)\n")
        fh.write(f"# D_model={r.D_model} scale={d['scale']} shift={d['shift']} "
                 f"s_out_bits={d['s_out_bits']:08x} o8_amax={int(np.max(np.abs(d['o8'])))}\n")
        hexi(fh, "a8", d["a8"], 2)                     # o-proj C-1 activation codes
        hexi(fh, "o8", d["o8"], 2)                     # requantized serializer codes
        hex16(fh, "xrow", np.asarray(d["xrow_bits"], dtype=np.uint16))   # residual row f16(X[T])
        hex16(fh, "r1", np.asarray(d["r1_bits"], dtype=np.uint16))       # RDATA expectation
    words = sum(1 for ln in p.open() if not ln.startswith("#"))
    return {"case": name, "chk_words": words, "ops": n_ops,
            "scale": d["scale"], "shift": d["shift"], "s_out_bits": d["s_out_bits"],
            "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [run_case(c[0], out) for c in CASES]
    # determinism self-check: regenerate into a shadow dir, byte-compare
    shadow = out / "_det"
    shadow.mkdir(exist_ok=True)
    for c in CASES:
        r2 = run_case(c[0], shadow)
        r1 = next(x for x in rows if x["case"] == c[0])
        assert r1["sha"] == r2["sha"], f"non-deterministic: {c[0]}"

    with (out / "manifest.txt").open("w") as fh:
        for r in rows:
            fh.write(f"{r['case']} chk_words={r['chk_words']} "
                     f"ops={r['ops']} sha={r['sha']}\n")
    seg2 = write_compose_seg2(out)                 # segment-2 op stream + self-check
    total = sum(r["chk_words"] for r in rows)
    print(f"L4 GEN: {len(rows)} cases, {total} checkpoint words, determinism OK.")
    print(f"SEGMENT-1 SELF-CHECK PASSED {len(rows)}/{len(rows)} — re-derived o-proj "
          f"reproduces r.attn_proj AND r.r1 bit-exactly (public golden fns, zero "
          f"golden edits).")
    print(f"SEGMENT-2 SELF-CHECK PASSED {seg2['cases']}/{len(rows)} — re-derived "
          f"RMSNorm-1 front-half reproduces r.h AND r.s_h bit-exactly; "
          f"compose_seg2.ops emitted ({seg2['efs_checks']} EFS s_h checkpoints). "
          f"h8 deferred to segment 1 (§3c carve-out).")


if __name__ == "__main__":
    main()
