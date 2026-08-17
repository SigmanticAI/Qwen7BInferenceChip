#!/usr/bin/env python3
"""gen_layer_vectors.py — FULL decoder-layer composition oracle for tb_layer.

Runs the golden arbiter apex_golden.transformer.decoder_layer_fx on small
self-check configs and emits, IN LAYER-DATAFLOW ORDER, the bit-exact stage
vectors that drive the three NEW RTL blocks built this session:

  * rtl/rope/rope.sv       — the attention-half RoPE rotation (per HF half-split
                             channel pair, decode-token q at m=T and every cache
                             key row at m=t).
  * rtl/asu/asu_silu.sv    — the SwiGLU FFN SiLU gate (per d_ffn element).
  * rtl/misc/residual_add.sv — the two C-6 fp16 residual adds (r1 = X[T]+attn,
                             r2 = r1+ffn), on the fp16 residual bus.

The GEMM / RMSNorm / attention-core / KVQ stages between these are ALREADY
RTL-verified bit-exact elsewhere (verif/mxe, verif/asu, verif/top/l3 replay the
REAL apex_top host-sequenced); here they are supplied from the SAME golden
decoder_layer_fx intermediates so this TB is a *layer-level composition* of the
new ops on real layer tensors, not a re-verification of the whole tile.

fp16-bus boundary (faithful, documented): the hardware Q/K write bus is fp16
(S-2 KV-write dequant), so the RoPE inputs are the projection outputs narrowed
to fp16; expected outputs are rope_fx() of those fp16 values (bit-exact with
rope.sv). residual_add likewise takes fp16 operands; its expected is
res_add = f16(f16(a)+f16(b)) — the fp16 residual primitive (NOT decoder_layer_fx
.r1, which adds pre-fp16 reals; the two differ by <=1 ULP double-rounding — this
TB checks the hardware primitive on the layer's real operands).

Emits into build/:
  layer_rope_{u_q,x_i,x_ih,exp_i,exp_ih}.hex   RoPE channel-pair records
  layer_silu_{xq,exp}.hex                       SiLU element records
  layer_res_{a,b,exp}.hex                       residual-add records
  layer_nvec.svh                                localparams: the three counts
  rope_cos_base/slope, rope_sin_base/slope,     cos/sin ROMs  (Icarus path)
  silu_lut_base/slope                           SiLU  ROMs    (Icarus path)
  layer_cases.txt                               human-readable per-case manifest
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "golden"))
import numpy as np  # noqa: E402

from apex_golden import attention as at  # noqa: E402
from apex_golden import transformer as tf  # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits, rne  # noqa: E402

# ── layer self-check configs (both head_dim envelope points) ─────────────────
#   (H, head_dim, d_ffn, T, seed)   — D_model = H*head_dim = 128 (rmsnorm envelope)
CASES = [
    (2, 64, 256, 8, 0x0DEC0DE0),   # multi-head, head_dim=64
    (1, 128, 344, 16, 0x0DEC0DE1),  # single-head, head_dim=128
]
TIER = at.TIER_CQ8


def make_weights(rng, H, head_dim, d_ffn):
    """Same construction as golden tests/test_transformer.make_weights."""
    D = H * head_dim

    def w(shape):
        return rng.integers(-127, 128, shape).astype(np.int64)

    s_proj = 1.0 / (127.0 * math.sqrt(D))
    s_ffn = 1.0 / (127.0 * math.sqrt(D))
    s_down = 1.0 / (127.0 * math.sqrt(d_ffn))
    gamma1 = np.array([int(round((0.7 + 0.6 * rng.random()) * 8192))
                       for _ in range(D)], dtype=np.int64)
    gamma2 = np.array([int(round((0.7 + 0.6 * rng.random()) * 8192))
                       for _ in range(D)], dtype=np.int64)
    return tf.LayerWeights(
        Wq=w((D, D)), Wk=w((D, D)), Wv=w((D, D)), Wo=w((D, D)),
        Wg=w((D, d_ffn)), Wu=w((D, d_ffn)), Wd=w((d_ffn, D)),
        s_wq=s_proj, s_wk=s_proj, s_wv=s_proj, s_wo=s_proj,
        s_wg=s_ffn, s_wu=s_ffn, s_wd=s_down,
        gamma1=gamma1, gamma2=gamma2, H=H, head_dim=head_dim)


def res_add(a_bits: int, b_bits: int) -> int:
    a = f16_bits_to_f64(np.array([a_bits], dtype=np.uint16))
    b = f16_bits_to_f64(np.array([b_bits], dtype=np.uint16))
    return int(f64_to_f16_bits(a + b)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rope_u, rope_xi, rope_xih, rope_ei, rope_eih = [], [], [], [], []
    silu_xq, silu_exp = [], []
    res_a, res_b, res_exp = [], [], []
    manifest = []

    for (H, hd, dff, T, seed) in CASES:
        D = H * hd
        rng = np.random.default_rng(seed)
        w = make_weights(rng, H, hd, dff)
        X = rng.normal(0, 1, (T + 1, D))
        r = tf.decoder_layer_fx(X, w, TIER)

        theta = tf.rope_theta(hd)
        half = hd // 2
        pos_k = np.arange(T)

        # ── RoPE: fp16 bus in, rope_fx(fp16) out, per HF half-split pair ───────
        n_rope0 = len(rope_u)
        # decode-token q (m = T) and every cache key row (m = t) per head
        for h_i in range(H):
            sl = slice(h_i * hd, (h_i + 1) * hd)
            # decode query row
            q_bus = f16_bits_to_f64(f64_to_f16_bits(r.q_real[sl]))
            q_y = tf.rope_fx(q_bus, T, theta)
            qb = f64_to_f16_bits(q_bus)
            yb = f64_to_f16_bits(q_y)
            u_q = tf.rope_phase_q(T, theta)              # [half]
            for i in range(half):
                rope_u.append(int(u_q[i]) % (1 << tf.PH_FRAC))
                rope_xi.append(int(qb[i]))
                rope_xih.append(int(qb[i + half]))
                rope_ei.append(int(yb[i]))
                rope_eih.append(int(yb[i + half]))
            # cache key rows
            K_bus = f16_bits_to_f64(f64_to_f16_bits(r.K_real[:, sl])).reshape(T, hd)
            K_y = tf.rope_fx(K_bus, pos_k, theta)
            Kb = f64_to_f16_bits(K_bus).reshape(T, hd)
            Kyb = f64_to_f16_bits(K_y).reshape(T, hd)
            u_qk = tf.rope_phase_q(pos_k, theta)          # [T, half]
            for t in range(T):
                for i in range(half):
                    rope_u.append(int(u_qk[t, i]) % (1 << tf.PH_FRAC))
                    rope_xi.append(int(Kb[t, i]))
                    rope_xih.append(int(Kb[t, i + half]))
                    rope_ei.append(int(Kyb[t, i]))
                    rope_eih.append(int(Kyb[t, i + half]))
        n_rope = len(rope_u) - n_rope0

        # ── SiLU: Q5.10 gate feeder in, silu_fx out (per d_ffn element) ───────
        n_silu0 = len(silu_xq)
        xq = rne(np.asarray(r.gate_real, dtype=np.float64) * (1 << tf.SILU_IN_FRAC))
        xq = xq.astype(np.int64)
        assert np.all(np.abs(xq) < (1 << 15)), "gate Q5.10 code exceeds signed 16b"
        yq = tf.silu_fx(xq)
        for k in range(dff):
            silu_xq.append(int(xq[k]))
            silu_exp.append(int(yq[k]))
        n_silu = len(silu_xq) - n_silu0

        # ── residual: the two C-6 fp16 residual adds on the layer's operands ──
        n_res0 = len(res_a)
        # r1 = residual_add( f16(X[T]) , f16(attn_proj) )   per element
        xT_bits = f64_to_f16_bits(X[T])
        ap_bits = f64_to_f16_bits(r.attn_proj)
        for d in range(D):
            a, b = int(xT_bits[d]), int(ap_bits[d])
            res_a.append(a); res_b.append(b); res_exp.append(res_add(a, b))
        # r2 = residual_add( f16(r1) , f16(ffn_out) )       per element
        r1_bits = f64_to_f16_bits(r.r1)
        ff_bits = f64_to_f16_bits(r.ffn_out)
        for d in range(D):
            a, b = int(r1_bits[d]), int(ff_bits[d])
            res_a.append(a); res_b.append(b); res_exp.append(res_add(a, b))
        n_res = len(res_a) - n_res0

        manifest.append(
            f"H{H}/hd{hd}/dff{dff}/T{T} tier=CQ8 D_model={D}: "
            f"rope_pairs={n_rope} silu_elems={n_silu} res_adds={n_res}")

    # ── write vector streams ─────────────────────────────────────────────────
    def wr(name, vals, mask, width):
        with open(out / name, "w") as f:
            f.write("".join(f"{int(v) & mask:0{width}x}\n" for v in vals))

    wr("layer_rope_u_q.hex", rope_u, 0x3FFF, 4)
    wr("layer_rope_x_i.hex", rope_xi, 0xFFFF, 4)
    wr("layer_rope_x_ih.hex", rope_xih, 0xFFFF, 4)
    wr("layer_rope_exp_i.hex", rope_ei, 0xFFFF, 4)
    wr("layer_rope_exp_ih.hex", rope_eih, 0xFFFF, 4)
    wr("layer_silu_xq.hex", silu_xq, 0xFFFF, 4)
    wr("layer_silu_exp.hex", silu_exp, 0xFFFF, 4)
    wr("layer_res_a.hex", res_a, 0xFFFF, 4)
    wr("layer_res_b.hex", res_b, 0xFFFF, 4)
    wr("layer_res_exp.hex", res_exp, 0xFFFF, 4)

    with open(out / "layer_nvec.svh", "w") as f:
        f.write("// GENERATED by gen_layer_vectors.py — do not edit.\n")
        f.write(f"localparam int LAYER_ROPE_NVEC = {len(rope_u)};\n")
        f.write(f"localparam int LAYER_SILU_NVEC = {len(silu_xq)};\n")
        f.write(f"localparam int LAYER_RES_NVEC  = {len(res_a)};\n")

    # ── ROMs for the Icarus $readmemh path (SAME golden tables) ──────────────
    cb, cs, sb, ss = tf._cos_sin_lut_tables()
    wr("rope_cos_base.hex", cb, 0xFFFF, 4)
    wr("rope_cos_slope.hex", cs, 0x07FF, 3)
    wr("rope_sin_base.hex", sb, 0xFFFF, 4)
    wr("rope_sin_slope.hex", ss, 0x07FF, 3)
    sbase, sslope = tf._silu_lut_tables()
    wr("silu_lut_base.hex", sbase, 0xFFFF, 4)
    wr("silu_lut_slope.hex", sslope, 0x03FF, 3)

    with open(out / "layer_cases.txt", "w") as f:
        f.write("\n".join(manifest) + "\n")

    print("LAYER oracle (decoder_layer_fx, tier CQ8):")
    for m in manifest:
        print(f"  {m}")
    print(f"  TOTAL: rope_pairs={len(rope_u)} silu_elems={len(silu_xq)} "
          f"res_adds={len(res_a)} -> {out}")


if __name__ == "__main__":
    main()
