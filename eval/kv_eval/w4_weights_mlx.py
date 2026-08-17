# w4_weights_mlx.py — B3 stage-6 ADOPTION GATE: weight-tier installer.
#
# Installs the B3 weight-path tiers into an MLX Qwen2.5 model by constructing
# the model's own quantized-weight representation (q, scales, biases) DIRECTLY
# from the golden chain — no re-quantization hop:
#
#   base       stock mlx-4bit g64 weights (no transform; THE baseline)
#   w-int8     S8 golden feed: per-tensor symmetric INT8 (bytes read from the
#              committed build/s8_weights cache; UNfolded per-tensor s_w from
#              its meta.json — gamma folds are caller-side and pow2-exact, so
#              cache scales are the clean per-tensor s_w)
#   w4-tile-a  B3 as landed: INT8 → weight_codec realization (A), one fp16
#              scale per job tile (8 output columns × full contraction K —
#              required by (A): K-split partials sum raw INT32 before the one
#              epilogue factor), codes sign-extended
#   w4-g32-b   the lever: INT8 → G=32 K-groups → realization (B) dequant +
#   w4-g16-b   one INT8 requant per 8-column × full-K stripe (quant_rows_i8
#              rule). ⚠️ REALIZATION DISCLOSURE: at this model's K (3584 /
#              18944 > K_MAX 2048) every tensor is a multi-job K-split chain,
#              so a D-021-style feeder-DERIVED per-job scale cannot ride the
#              single raw-INT32 epilogue factor. What these tiers model is a
#              HOST-COMPUTED stripe-global requant scale consumed as a
#              sideband (a design delta from D-021; no (B) RTL exists — the
#              built stage-1–3 feeder is realization (A) only). A per-job-
#              scale variant (host dequant per K-split segment) would use
#              FINER scales, i.e. same-or-better accuracy — this reading is
#              the conservative one for the finer-grouping lever.
#
# Numerics: this module re-implements the golden chain VECTORIZED (the golden
# weight_codec is deliberately scalar). Certification is two-layer, both
# hard gates counted into the result JSON:
#   1. weight_twin_selftest(): synthetic tiles across shapes/distributions,
#      vectorized chain vs golden weight_codec, EXACT equality.
#   2. install-time golden stripe gate: per real tensor, sampled stripes
#      (incl. the tensor's amax stripe) re-derived through the actual golden
#      compress_weights_w4/wfeed_w4_to_i8 and compared EXACTLY.
#
# Representation exactness: MLX affine dequant is w = scales*q + biases per
# group of 64 along K. We set q = code + 2^(bits-1), scales = fp16(composite),
# biases = -2^(bits-1)*scales (exact: pow2 multiple). The ONLY narrowing vs
# the golden composite is the f64→fp16 RNE cast of the composite scale
# (≤2^-12 relative, measured per tensor and recorded) — the same 11-bit
# grade the hardware's f16_grade applies to the epilogue composite (§2 of
# B3_WEIGHT_PATH.md; measured ~free in test_weight_codec section F). All
# tiers INCLUDING base then live on the same fp16 dequant grid, so tier
# deltas are format-symmetric.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden import weight_codec as WC  # noqa: E402  (the B3 arbiter)
from apex_golden import cq_codec as CQ      # noqa: E402  (C-1 primitives)
from apex_golden import attention as AT     # noqa: E402  (quant_rows_i8)
from apex_golden.cq_codec import EPS        # noqa: E402  (contract §1)
from apex_golden.fp import (f16_bits_to_f64, f32_bits_to_f64,  # noqa: E402
                            f64_to_f16_bits)

STRIPE = 8            # job tile = 8 output columns (MXE_N=8)
MLX_GS = 64           # MLX group size of the shipped checkpoint (asserted)
WTIERS = ("base", "w-int8", "w4-tile-a", "w4-g32-b", "w4-g16-b",
          "w4-direct-g32-b", "w4-direct-g16-b")
# w4-direct-g32-b: the SAME G=32/(B) chain, but sourced straight from the
# mlx-4bit-dequant real weights (fp16 grid) instead of the per-tensor INT8
# intermediate — a host-prep variant, feeder contract unchanged. The
# stage-6 full set showed the INT8 hop costs −1.8 pts on its own; this tier
# measures what W4 costs WITHOUT that hop. cq_codec primitives take fp16
# bit patterns, and both integer codes ≤127 and mlx-dequant values are
# fp16-grid — the golden chain applies unchanged. No s_w: the composite is
# the per-stripe requant scale alone (already fp16, zero cast error).

# cache tag → (mlx module path pattern, transpose_cache_to_mlx)
LAYER_TAGS = {"Wq": "self_attn.q_proj", "Wk": "self_attn.k_proj",
              "Wv": "self_attn.v_proj", "Wo": "self_attn.o_proj",
              "Wg": "mlp.gate_proj", "Wu": "mlp.up_proj",
              "Wd": "mlp.down_proj"}


# ── vectorized golden chain (certified by the gates below) ───────────────────

def _f16_rne(x64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f64 → fp16 RNE; returns (exact f64 value of the fp16, fp16 bits)."""
    s16 = np.asarray(x64, dtype=np.float64).astype(np.float16)
    return s16.astype(np.float64), s16.view(np.uint16)


def w4_quant_tile(w: np.ndarray):
    """w [C, 8, K] int64 → (codes [C,8,K] in [-8,7], s64 [C], sbits [C]).

    Golden rule per tile: s = f16(max(amax/7, EPS)); codes = clip(rne(w/s)).
    Weight ints ≤127 are fp16-exact, so f64 amax == golden's fp16-bits amax.
    """
    amax = np.abs(w).reshape(w.shape[0], -1).max(axis=1).astype(np.float64)
    s64, sbits = _f16_rne(np.maximum(amax / 7.0, EPS))
    codes = np.clip(np.rint(w / s64[:, None, None]), -8, 7).astype(np.int64)
    return codes, s64, sbits


def w4_quant_groups(w: np.ndarray, G: int):
    """w [C, 8, K] int64 → per-(column, G-K-block) INT4: (codes, s64, sbits).

    Same partition as golden group_ids(K, N, G): G contraction rows × 1
    output column."""
    C, R, K = w.shape
    assert K % G == 0, (K, G)
    g = w.reshape(C, R, K // G, G)
    amax = np.abs(g).max(axis=3).astype(np.float64)
    s64, sbits = _f16_rne(np.maximum(amax / 7.0, EPS))          # [C,R,K//G]
    codes = np.clip(np.rint(g / s64[..., None]), -8, 7).astype(np.int64)
    return codes.reshape(C, R, K), s64, sbits


def requant_rows_i8(x: np.ndarray):
    """x [C, M] exact f64 → per-row INT8 (attention.quant_rows_i8 rule):
    s = f16(max(amax/127, EPS)); codes = clip(rne(x/s), -128, 127)."""
    amax = np.abs(x).max(axis=1)
    s64, sbits = _f16_rne(np.maximum(amax / 127.0, EPS))
    codes = np.clip(np.rint(x / s64[:, None]), -128, 127).astype(np.int64)
    return codes, s64, sbits


def tier_codes_for_stripes(w: np.ndarray, wtier: str):
    """w [C, 8, K] int64 (INT8 feed bytes) → (out_codes [C,8,K], s64 [C],
    sbits [C], bits, fine) for the tier's INT-domain feed + per-stripe scale.
    fine = (codes4, sbits) of the intermediate fine-group W4 (B tiers only)."""
    C, R, K = w.shape
    if wtier == "w-int8":
        return w.astype(np.int64), np.ones(C), None, 8, None
    if wtier == "w4-tile-a":
        codes, s64, sbits = w4_quant_tile(w)
        return codes, s64, sbits, 4, None
    if wtier in ("w4-g32-b", "w4-g16-b", "w4-direct-g32-b",
                 "w4-direct-g16-b"):
        G = 16 if wtier in ("w4-g16-b", "w4-direct-g16-b") else 32
        c4, sg64, sgbits = w4_quant_groups(w, G)
        # realization (B): exact dequant (int·fp16 product, exact in f64),
        # then ONE per-tile INT8 requant — the seam_feeder_quant chain.
        real = c4.reshape(C, R, K // G, G) * sg64[..., None]
        i8, s64, sbits = requant_rows_i8(real.reshape(C, R * K))
        return i8.reshape(C, R, K), s64, sbits, 8, (c4, sgbits)
    raise ValueError(f"unknown weight tier {wtier!r}")


# ── golden stripe gate (the arbiter run on real bytes) ───────────────────────

def golden_check_stripe(w8_stripe: np.ndarray, wtier: str,
                        my_codes: np.ndarray, my_sbits, my_fine) -> int:
    """One stripe [8, K] (MLX orientation) through the ACTUAL golden
    weight_codec; exact equality vs the vectorized results. Returns #checks."""
    tile = np.ascontiguousarray(w8_stripe.T)              # golden [K, N=8]
    K, N = tile.shape
    n = 0
    assert wtier != "w4-direct-g32-b", \
        "direct tier gates via golden_check_direct_stripe (this path's " \
        "compress_weights_w4 input adapter int-truncates real values)"
    if wtier == "w4-tile-a":
        blob = WC.compress_weights_w4(tile, group="tile")
        assert np.array_equal(blob.codes, my_codes.T), "A codes mismatch"
        n += blob.codes.size
        assert np.uint16(blob.scales[0]) == np.uint16(my_sbits), "A scale"
        n += 1
        flat, s = WC.wfeed_w4_to_i8(blob, "A")            # the feeder bytes
        rows, cols = WC.stream_index(K, N)
        back = np.zeros_like(tile)
        back[rows, cols] = flat                            # K%8==0: no pads
        assert np.array_equal(back, my_codes.T), "A feeder stream mismatch"
        n += flat.size
        assert np.uint16(s) == np.uint16(my_sbits)
        n += 1
    elif wtier in ("w4-g32-b", "w4-g16-b"):
        G = 32 if wtier == "w4-g32-b" else 16
        blob = WC.compress_weights_w4(tile, group=G)
        fine_codes, _ = my_fine
        assert np.array_equal(blob.codes, fine_codes.T), "B fine codes"
        n += blob.codes.size
        i8_flat, s8 = WC.wfeed_w4_to_i8(blob, "B")
        rows, cols = WC.stream_index(K, N)
        back = np.zeros_like(tile)
        back[rows, cols] = i8_flat
        assert np.array_equal(back, my_codes.T), "B requant stream mismatch"
        n += i8_flat.size
        assert np.uint16(s8) == np.uint16(my_sbits), "B tile scale"
        n += 1
    elif wtier == "w-int8":
        n += 0                                             # bytes ARE the feed
    return n


def golden_check_direct_stripe(w_stripe: np.ndarray, my_codes: np.ndarray,
                               my_sbits, my_fine, G: int = 32) -> int:
    """Direct-tier golden gate: reference built from the cq_codec C-1
    primitives on the fp16 bit patterns (amax_bits → scale_from_amax →
    quant_codes per G-group, exact dequant, quant_rows_i8 tile requant) —
    the same primitive family the KV codec is certified on.
    compress_weights_w4 does NOT apply here: its weights_to_f16 input
    adapter is the INT8-narrowing step and int-truncates real values."""
    tile = np.ascontiguousarray(w_stripe.T)               # [K, 8] f64
    K, N = tile.shape
    bits16 = f64_to_f16_bits(tile).reshape(K, N)
    assert np.array_equal(
        f16_bits_to_f64(bits16.reshape(-1)).reshape(K, N), tile), \
        "direct-tier input is not on the fp16 grid"
    n = K * N                                             # grid exactness
    gid, ngroups = WC.group_ids(K, N, G)
    fine_codes = my_fine[0].T                             # [K, 8]
    real = np.zeros((K, N))
    for g in range(ngroups):
        m = gid == g
        s = CQ.scale_from_amax(CQ.amax_bits(bits16[m]), 4)
        codes = CQ.quant_codes(bits16[m], s, 4)
        assert np.array_equal(codes, fine_codes[m]), "direct fine codes"
        n += codes.size
        real[m] = f32_bits_to_f64(CQ.dequant_f32(codes, np.uint16(s)))
    i8, s8 = AT.quant_rows_i8(real.T.reshape(1, -1))      # one tile stream
    assert np.array_equal(i8[0].reshape(N, K), my_codes), "direct requant"
    n += i8.size
    assert np.uint16(s8[0]) == np.uint16(my_sbits), "direct tile scale"
    return n + 1


def weight_twin_selftest(seed: int = 0x84EF) -> int:
    """Vectorized chain vs the golden arbiter on synthetic tiles — the
    weight-path analogue of test_twin_bitexact. Exact equality, every tile,
    every tier. Returns total check count (raises on any mismatch)."""
    rng = np.random.default_rng(seed)
    total = 0
    for K in (64, 128, 192, 2048):
        for dist in ("uniform", "gauss", "outlier", "zeros", "amax127",
                     "realgauss"):
            if dist == "realgauss":     # fp16-grid REAL values (direct tier)
                w8 = rng.normal(0, 0.05, size=(K, STRIPE)) \
                    .astype(np.float16).astype(np.float64)
            elif dist == "uniform":
                w8 = rng.integers(-127, 128, size=(K, STRIPE))
            elif dist == "gauss":
                w8 = np.clip(np.rint(rng.normal(0, 30, size=(K, STRIPE))),
                             -127, 127).astype(np.int64)
            elif dist == "outlier":
                w8 = np.clip(np.rint(rng.normal(0, 6, size=(K, STRIPE))),
                             -127, 127).astype(np.int64)
                w8[rng.integers(0, K, 8), rng.integers(0, STRIPE, 8)] = 127
            elif dist == "zeros":
                w8 = np.zeros((K, STRIPE), dtype=np.int64)   # EPS scale path
            else:
                w8 = np.full((K, STRIPE), -127, dtype=np.int64)
            stripe = np.ascontiguousarray(w8.T)[None]        # [1, 8, K]
            if dist == "realgauss":      # real fp16-grid → direct tiers only
                for dt, dg in (("w4-direct-g32-b", 32),
                               ("w4-direct-g16-b", 16)):
                    codes, _, sbits, _, fine = tier_codes_for_stripes(
                        stripe, dt)
                    total += golden_check_direct_stripe(
                        stripe[0], codes[0], sbits[0],
                        (fine[0][0], fine[1][0]), G=dg)
                continue
            for wtier in ("w4-tile-a", "w4-g32-b", "w4-g16-b"):
                codes, _, sbits, _, fine = tier_codes_for_stripes(stripe, wtier)
                fine_s = (fine[0][0], fine[1][0]) if fine else None
                total += golden_check_stripe(stripe[0], wtier, codes[0],
                                             sbits[0], fine_s)
    return total


# ── MLX packed representation (bit order proven at runtime) ──────────────────

def pack_mlx(q: np.ndarray, bits: int) -> np.ndarray:
    """q [N, K] ints in [0, 2^bits-1] → uint32 [N, K*bits/32], LSB-first."""
    per = 32 // bits
    v = q.reshape(q.shape[0], -1, per).astype(np.uint32)
    shifts = (np.arange(per, dtype=np.uint32) * np.uint32(bits))
    return np.bitwise_or.reduce(v << shifts, axis=2).astype(np.uint32)


def unpack_mlx(wq: np.ndarray, bits: int, K: int) -> np.ndarray:
    per = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    v = (wq[..., None] >> (np.arange(per, dtype=np.uint32) * np.uint32(bits)))
    return (v & mask).reshape(wq.shape[0], K).astype(np.int64)


def mlx_pack_selftest() -> int:
    """Prove the nibble/byte order + affine convention against mx.quantize/
    mx.dequantize on this MLX build. Returns #checks; raises on mismatch."""
    import mlx.core as mx
    rng = np.random.default_rng(7)
    n = 0
    for bits in (4, 8):
        W = rng.normal(0, 1, size=(16, 128)).astype(np.float32)
        wq, sc, bi = mx.quantize(mx.array(W), MLX_GS, bits)
        wq_np = np.array(wq, copy=True).view(np.uint32).reshape(16, -1)
        q = unpack_mlx(wq_np, bits, 128)
        assert np.array_equal(pack_mlx(q, bits), wq_np), "pack∘unpack != id"
        n += q.size
        deq = np.array(mx.dequantize(wq, sc, bi, MLX_GS, bits)
                       .astype(mx.float32), copy=True).astype(np.float64)
        s64 = np.array(sc, copy=True).astype(np.float64)
        b64 = np.array(bi, copy=True).astype(np.float64)
        manual = (q.reshape(16, 128 // MLX_GS, MLX_GS)
                  * s64[..., None] + b64[..., None]).reshape(16, 128)
        # dequant grid: fp16 RNE of the affine form (kernel may hold more
        # precision internally; ≤1 fp16 ulp either way — assert that band)
        m16 = manual.astype(np.float16).astype(np.float64)
        ulp = np.abs(deq - m16)
        tol = np.maximum(np.abs(m16), 2.0 ** -14) * 2.0 ** -10
        assert np.all(ulp <= tol + 1e-30), \
            f"dequant off the affine grid by >1 ulp (bits={bits})"
        n += deq.size
    return n


# ── installer ────────────────────────────────────────────────────────────────

def _cache_targets(model, meta: dict, cache_dir: Path):
    """Yield (label, module, npy_path, s_w, transpose) for every B3-covered
    tensor: 7 projections × n_layers + lm_head. Embedding untouched."""
    assert not meta["tie_word_embeddings"], "tied embeddings unsupported"
    for li in range(meta["n_layers"]):
        lrec = meta["layers"][li]
        blk = model.model.layers[li]
        for tag, path in LAYER_TAGS.items():
            obj = blk
            for part in path.split("."):
                obj = getattr(obj, part)
            yield (f"L{li:02d}.{tag}", obj,
                   cache_dir / f"L{li:02d}_{tag}.npy",
                   float(lrec[f"s_{tag}"]), True)
    yield ("lm_head", model.lm_head, cache_dir / "lm_head_w8.npy",
           float(meta["s_head"]), False)


def install_tier(model, wtier: str, cache_dir: Path,
                 sample_seed: int = 0xB3, n_sample_stripes: int = 2,
                 expect_model: str | None = None, log=print) -> dict:
    """Transform every covered tensor of `model` in place. Returns the gate
    record (counts + measured cast errors) for the result JSON."""
    import mlx.core as mx
    import mlx.nn as nn

    assert wtier in WTIERS and wtier != "base"
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    if expect_model is not None:
        assert meta["model"] == expect_model, \
            f"cache is for {meta['model']!r}, eval model is {expect_model!r}"
    sq = meta["source_quant"]
    assert (sq["bits"], sq["group_size"]) == (4, MLX_GS), sq
    rng = np.random.default_rng(sample_seed)
    direct = wtier in ("w4-direct-g32-b", "w4-direct-g16-b")
    direct_G = 16 if wtier == "w4-direct-g16-b" else 32

    rec = {"wtier": wtier, "cache": str(cache_dir),
           "cache_format": meta.get("format"), "cache_model": meta["model"],
           "tensors": 0,
           "golden_stripe_checks": 0, "golden_stripes": 0,
           "composite_cast_max_rel": 0.0, "subnormal_scales": 0,
           "install_dequant_probe_max_ulp": 0.0}
    if wtier == "w-int8":
        rec["note"] = ("identity feed: the cache bytes ARE the golden "
                       "w-int8 feed, so golden_stripe_checks=0 is expected "
                       "(nothing for the arbiter to re-derive); the fp16 "
                       "composite cast + packed form are still measured "
                       "via composite_cast_max_rel and the dequant probe")
    elif wtier in ("w4-direct-g32-b", "w4-direct-g16-b"):
        rec["note"] = (f"host-prep variant (G={direct_G}): the SAME (B) "
                       "chain sourced straight from the checkpoint's "
                       "mlx-4bit-dequant weights (fp16 grid) — NO per-tensor "
                       "INT8 intermediate, no s_w; composite = the per-stripe "
                       "requant scale alone (fp16, zero cast error). Feeder "
                       "contract unchanged. Golden stripe gate samples "
                       "first+last+random stripes (no materialized tensor to "
                       "argmax). Same host-computed stripe-global-scale "
                       "realization disclosure as w4-g32-b applies.")
    elif wtier in ("w4-g32-b", "w4-g16-b"):
        rec["note"] = ("realization (B) modeled with a HOST-COMPUTED "
                       "stripe-global requant scale (K > K_MAX=2048 makes "
                       "the D-021 feeder-derived per-job scale unable to "
                       "ride the single K-split epilogue factor; no (B) RTL "
                       "exists). Conservative for the lever: a per-job-"
                       "scale design would use finer scales. See module "
                       "docstring.")

    for label, module, npy, s_w, transpose in _cache_targets(model, meta,
                                                             cache_dir):
        assert isinstance(module, nn.QuantizedLinear), (label, type(module))
        assert module.bits == 4 and module.group_size == MLX_GS, label
        if direct:
            # source = the loaded checkpoint's own mlx-4bit weights,
            # dequantized chunk-wise (fp16 grid); no INT8 hop, no s_w
            s_w = 1.0
            N, Kg = module.scales.shape
            K = Kg * MLX_GS
        else:
            w8 = np.load(npy, mmap_mode="r")
            Wt = w8.T if transpose else w8                   # [N_out, K]
            N, K = Wt.shape
        assert N % STRIPE == 0 and K % MLX_GS == 0, (label, N, K)
        S = N // STRIPE
        bits = 4 if wtier == "w4-tile-a" else 8
        off = 1 << (bits - 1)

        q_all = np.empty((N, K * bits // 32), dtype=np.uint32)
        comp16_all = np.empty(S, dtype=np.float16)
        # sampled stripes: n random + the tensor's amax stripe (worst case);
        # the direct tier has no materialized tensor to argmax over, so it
        # samples first + last + random instead (recorded in the JSON)
        sample = set(rng.integers(0, S, n_sample_stripes).tolist())
        if direct:
            sample |= {0, S - 1}
        else:
            amax_flat = int(np.argmax(np.abs(np.asarray(w8))))
            amax_col = (amax_flat % w8.shape[1]) if transpose \
                else (amax_flat // w8.shape[1])
            sample.add(int(amax_col) // STRIPE)
        probe_rows, probe_exp = [], []

        chunk = max(1, (1 << 21) // K)
        for a in range(0, S, chunk):
            b = min(a + chunk, S)
            if direct:
                r0, r1 = a * STRIPE, b * STRIPE
                w = np.array(mx.dequantize(
                    module.weight[r0:r1], module.scales[r0:r1],
                    module.biases[r0:r1], MLX_GS, 4).astype(mx.float32),
                    copy=True).astype(np.float64) \
                    .reshape(b - a, STRIPE, K)
            else:
                w = np.ascontiguousarray(Wt[a * STRIPE:b * STRIPE]) \
                    .astype(np.int64).reshape(b - a, STRIPE, K)
            codes, s64, sbits, cbits, fine = tier_codes_for_stripes(w, wtier)
            assert cbits == bits
            comp64 = s64 * s_w
            comp16 = comp64.astype(np.float16)
            rec["composite_cast_max_rel"] = max(
                rec["composite_cast_max_rel"],
                float(np.max(np.abs(comp16.astype(np.float64) - comp64)
                             / np.maximum(comp64, 1e-300))))
            rec["subnormal_scales"] += int(np.sum(
                (comp16.astype(np.float64) < 2.0 ** -14) & (comp64 > 0)))
            assert np.all(np.isfinite(comp16.astype(np.float64))) and \
                np.all(comp16.astype(np.float64) > 0), f"{label}: dead scale"
            comp16_all[a:b] = comp16
            q = (codes + off).reshape((b - a) * STRIPE, K)
            assert q.min() >= 0 and q.max() < (1 << bits), label
            q_all[a * STRIPE:b * STRIPE] = pack_mlx(q, bits)
            for si in sorted(sample):
                if a <= si < b:
                    i = si - a
                    fine_s = (fine[0][i], fine[1][i]) if fine else None
                    if direct:
                        rec["golden_stripe_checks"] += \
                            golden_check_direct_stripe(w[i], codes[i],
                                                       sbits[i], fine_s,
                                                       G=direct_G)
                    else:
                        rec["golden_stripe_checks"] += golden_check_stripe(
                            w[i], wtier, codes[i],
                            sbits[i] if sbits is not None else None, fine_s)
                    rec["golden_stripes"] += 1
                    r = si * STRIPE
                    probe_rows.append(r)
                    probe_exp.append(codes[i][0].astype(np.float64)
                                     * comp16[i].astype(np.float64))
            del w, codes, s64, fine

        scales = np.repeat(comp16_all[:, None], STRIPE, axis=1) \
            .reshape(N, 1).repeat(K // MLX_GS, axis=1)
        biases = (-float(off) * scales.astype(np.float64)).astype(np.float16)
        assert np.array_equal(biases.astype(np.float64),
                              -float(off) * scales.astype(np.float64)), \
            f"{label}: bias not fp16-exact"

        module.weight = mx.array(q_all)
        module.scales = mx.array(scales)
        module.biases = mx.array(biases)
        module.bits = bits
        module.group_size = MLX_GS
        mx.eval(module.weight, module.scales, module.biases)

        # informational probe: MLX's own dequant of sampled rows vs the
        # golden effective values on the fp16 grid (≤1 ulp band expected)
        if probe_rows:
            pr = mx.array(np.array(probe_rows, dtype=np.int32))
            deq = np.array(mx.dequantize(
                module.weight[pr], module.scales[pr], module.biases[pr],
                MLX_GS, bits).astype(mx.float32), copy=True).astype(np.float64)
            exp16 = np.stack(probe_exp).astype(np.float16).astype(np.float64)
            denom = np.maximum(np.abs(exp16), 2.0 ** -14) * 2.0 ** -10
            rec["install_dequant_probe_max_ulp"] = max(
                rec["install_dequant_probe_max_ulp"],
                float(np.max(np.abs(deq - exp16) / denom)))
        rec["tensors"] += 1
        mx.clear_cache()
        if rec["tensors"] % 28 == 0:
            log(f"[install {wtier}] {rec['tensors']} tensors done "
                f"(last {label})")

    assert rec["tensors"] == meta["n_layers"] * len(LAYER_TAGS) + 1
    return rec
