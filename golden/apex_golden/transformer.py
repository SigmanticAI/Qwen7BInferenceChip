"""transformer.py — golden FULL decoder-layer chain (Llama/Qwen family).

Extends the proven APEX attention decode-step (apex_golden.attention, the
D-021 seam) into a COMPLETE modern transformer decoder layer by ADDING two
new fixed-point primitives — RoPE (rotary position encoding) and SwiGLU's
SiLU gate — and wiring them, the multi-head attention, the output projection,
two residual adds and a post-attention RMSNorm together.  Nothing here
reinvents a proven stage: attention_core / gemm_i8 / online_softmax_fx /
rmsnorm_fx / quant_rows_i8 / requant_i32_to_i8 / calib_requant / the KVQ codec
are IMPORTED and composed.

Layer graph (single decode step, HF Llama/Qwen convention):

  x  --(RMSNorm-1, rmsnorm_fx, gamma1)-->  h
     --(C-1 INT8 quant of h/2^8)-->  h8
     --(per-head Q/K/V gemm_i8 WS projections, S-2 KV-write dequant)-->
         q_real[hd], K_real[t,hd], V_real[t,hd]  (fp16 bus)
     --(ROPE_FX on q at pos T and each K row at pos t; HF half-split)-->
         q̂, K̂  (fp16 bus)  [V is NOT rotated]
     --(attention_core per head: KVQ -> feeders -> Q·K̂ᵀ -> softmax -> P·V̂
        -> C-2 epilogue)-->  head outputs -> concat = attn[D_model]
     --(W_O gemm_i8 + C-2 epilogue)-->  attn_proj  (real)
     --(RESIDUAL-1, C-6 contract below)-->  r1 = f16(x + attn_proj)
     --(RMSNorm-2, rmsnorm_fx, gamma2)-->  h2
     --(SwiGLU FFN: gate=gemm_i8(h2,Wg); up=gemm_i8(h2,Wu);
        p = f16( SILU_FX(gate) * up ); down = gemm_i8(p,Wd) + C-2 epilogue)-->
         ffn_out  (real)
     --(RESIDUAL-2)-->  r2 = f16(r1 + ffn_out)   = the layer output

════════════════════════════════════════════════════════════════════════════
NEW FIXED-POINT CONTRACTS (mirror the ASU exp LUT design; RTL byte-mirrors
these via golden/gen_rope_lut_tables.py and golden/gen_silu_lut_tables.py)
════════════════════════════════════════════════════════════════════════════

RoPE (C-ROPE)
  Convention: HuggingFace Llama/Qwen — HALF-SPLIT rotate_half (NOT interleaved).
    For head_dim d, half = d/2, pair (i, i+half):
      x̂[i]      = x[i]·cos(φ) − x[i+half]·sin(φ)
      x̂[i+half] = x[i+half]·cos(φ) + x[i]·sin(φ)
    with φ = m·θ_i, θ_i = base^(−2i/d), base = 10000 (ROPE_THETA_BASE),
    m the token position (decode token at m = T, cache token t at m = t).
  Phase: the real angle φ = m·θ_i is reduced to a turn fraction
      u = frac(φ / 2π) ∈ [0,1) and quantized to Q0.PH_FRAC (PH_FRAC = 14):
      u_q = RNE(u · 2^14) mod 2^14.  (rope_phase_q — the RTL precomputes the
      identical per-(m,i) phase table; θ_i·m folding is done in float64 and
      quantized ONCE, deterministically.)  This is the ONLY phase rounding.
  cos/sin LUT (cos_fx / sin_fx): 256 entries over a FULL turn, linear
      interpolation, EXACT mirror of the asu_exp_lut structure:
        idx  = u_q >> CS_FRAC_REM            (top CS_LUT_BITS = 8 bits, 0..255)
        frac = u_q − (idx << CS_FRAC_REM)     (CS_FRAC_REM = 14−8 = 6 bits)
        y    = BASE[idx] + ((SLOPE[idx]·frac) >>> CS_FRAC_REM)   (SIGNED
               arithmetic shift — SLOPEs are signed; floor toward −∞, matching
               numpy/Python int64 >>; RTL uses $signed(...) >>>)
      Output SIGNED Q2.14 (CS_OUT_FRAC = 14): value = y / 2^14 ∈ [−1, 1].
      Tables: cos/sin base ∈ [−16384, 16384] (signed 16b), slope ∈ [−402, 402]
      (signed 11b).  No input clamp needed — the phase is periodic within
      [0, 2^14); u_q is already reduced mod 2^14.
  rope_fx multiply: x is an fp16-grid value; x·(cos/2^14) and x·(sin/2^14) are
      EXACT in float64; the two-product sum is narrowed to fp16 with ONE RNE
      (matching the S-2 KV-write fp16 bus).  rope_ref uses true math.cos/sin
      and no fp16 narrowing (the yardstick isolates LUT + phase + fp16 error).

SiLU (C-SILU) — SwiGLU gate  SiLU(x) = x·sigmoid(x)
  Input Q-format Q5.SILU_IN_FRAC (SILU_IN_FRAC = 10); domain [−8, 8] (span 16,
      identical count geometry to the exp LUT).  Inputs are CLAMPED to
      [−8, 8]: below −8 → SiLU(−8) ≈ −0.00268; above +8 → SiLU(8) ≈ 7.997.
      Both clamps are part of the contract; the FFN generator asserts the gate
      activations lie inside the domain so the clamp is never exercised in a
      budgeted case (out-of-domain error grows linearly and is NOT budgeted).
  LUT (silu_fx): 256 entries, segment 16/256 = 0.0625, linear interpolation:
        off  = clamp(x_q, −8·2^10, 8·2^10) + 8·2^10          (0..16384)
        idx  = min(off >> SILU_FRAC_REM, 255)  (SILU_FRAC_REM = 4+10−8 = 6)
        frac = off − (idx << SILU_FRAC_REM)
        y    = BASE[idx] + ((SLOPE[idx]·frac) >>> SILU_FRAC_REM)  (signed)
      Output SIGNED Q4.SILU_OUT_FRAC (SILU_OUT_FRAC = 12): value = y / 2^12.
      Tables: base ∈ [−1140, 32500] (signed 16b), slope ∈ [−26, 282] (signed
      10b).  silu_fx(0) == 0 exactly.

RESIDUAL (C-6, proposed contract — documented here, NOT a frozen apex_pkg
  struct change): the residual stream is carried as float64 EXACT of the fp16
  bus.  Each sublayer output is dequantized to real units (its C-2 epilogue
  scale) and added to the incoming residual; the SUM is narrowed to fp16 with
  ONE RNE (f16(a+b)) — the same single-narrowing discipline as the KV write
  port.  The fp16 residual is then per-token C-1 INT8 quantized at the next
  RMSNorm feeder (RMSNorm is scale-covariant, so the C-1 scale cancels in the
  norm — only its quantization noise enters, exactly as in attention_fx).

The float64 yardstick decoder_layer_ref runs the identical graph in full
precision (same DEQUANTIZED weights w8·s_w, rmsnorm_ref, softmax_ref,
rope_ref, silu_ref, NO fp16 narrowings) so measured error isolates THIS
layer's fixed-point stages, not weight quantization of a pretrained model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .attention import (
    CHUNK_T_MAX, AttnCore, attention_chunked, attention_core,
    attention_ref_core, calib_requant, quant_rows_i8, rmsnorm_fx,
    rmsnorm_fx_wide,
)
from .compute import gemm_i8, gemm_i8_ksplit, requant_i32_to_i8, rmsnorm_ref, softmax_ref
from .fp import f16_bits_to_f64, f64_to_f16_bits, rne
from .weight_codec import f16_grade

# ── RoPE cos/sin LUT contract constants (C-ROPE) ─────────────────────────────
ROPE_THETA_BASE = 10000.0
CS_LUT_BITS = 8                 # 256-entry cos/sin table
PH_FRAC = 14                    # phase Q0.14 (turn fraction), u_q ∈ [0, 2^14)
CS_OUT_FRAC = 14                # cos/sin output signed Q2.14 ∈ [-1, 1]
CS_FRAC_REM = PH_FRAC - CS_LUT_BITS      # 6 interp-fraction bits per segment

# ── SiLU LUT contract constants (C-SILU) ─────────────────────────────────────
SILU_LUT_BITS = 8               # 256-entry SiLU table
SILU_IN_FRAC = 10               # input Q5.10, domain [-8, 8]
SILU_OUT_FRAC = 12              # output signed Q4.12
SILU_DOMAIN = 8.0               # clamp |x| <= 8
SILU_FRAC_REM = 4 + SILU_IN_FRAC - SILU_LUT_BITS   # 6

# fp16 bus narrowing (one RNE), the residual/RoPE bus discipline
def _f16(x: np.ndarray) -> np.ndarray:
    """Narrow float64 -> fp16 (RNE) -> exact float64 of the fp16 value."""
    x = np.asarray(x, dtype=np.float64)
    return f16_bits_to_f64(f64_to_f16_bits(x)).reshape(x.shape)


# ═══════════ C-LBUS (D-030) — hardware bus-grade composition mode ═══════════
#
# STATUS: design-approved (LEVEL_C_INTEGRATION.md §9.1 R4); OWNER SIGN-OFF
# PENDING at the IB-LAYER S1 gate — no RTL stage may gate against BUS_ON
# before that sign-off is recorded in docs/design/IB_LAYER.md. Measured
# deltas: docs/results/ib_layer_bus/RESULTS.md (pinned in
# tests/test_layer_bus.py).
#
# WHY: decoder_layer_fx composes some stages on float64 operands no bus-
# realizable datapath can reproduce bit-exactly — per-tensor weight scales
# are arbitrary float64, while the tile's sideband contract requires every
# composite to be fp16-GRADE fp32 (<= 11 significant bits; apex_scale_quant
# C2, `scale_error` otherwise), and the S-2 write bus / residual bus carry
# fp16. C-LBUS names the exact narrowing points so the RTL integration
# (docs/design/IB_LAYER.md §2b) has ONE arbiter instead of per-TB footnotes
# (the delta verif/layer/gen_layer_vectors.py:21-27 documented locally).
#
# The three flags (all default OFF == today's behaviour, bit-identical):
#   x_f16          NP-x: the layer-input rows enter on the fp16 bus. In real
#                  decode this is a no-op in steady state (the input IS the
#                  previous layer's fp16 r2); it only affects synthetic
#                  float64 test rows.
#   rope_in_f16    NP-r: q/K projection rows (bias already added, per the
#                  HF rope(Wx+b) order) are narrowed to the S-2 fp16 bus
#                  BEFORE rope_fx — the hardware order: projection ->
#                  S-2 dequant/narrow -> rotate -> store. V is untouched
#                  here (its S-2 narrowing already exists at the
#                  attention-core input, unchanged).
#   scale_f16grade NP-s: every host-computed dequant composite is
#                  f16_grade()d (weight_codec.f16_grade — the SAME C2
#                  primitive B3 measured at <= 5e-5 absolute cost): the
#                  Q/K/V dequant composites s_h*s_w?, the per-head attn
#                  output dequant o8*s_out, the projection-epilogue s_out
#                  (Wo, Wd), and the FFN gate/up composites s_h2*s_w[gu].
#
# NO separate flag exists for the residual operands or the SwiGLU product:
# with scale_f16grade set, o8*graded-composite products and their fp16-bus
# sums/products are EXACT in float64 (<= 53 significand bits), so the
# single-RNE narrowings already in decoder_layer_fx are directly
# RTL-realizable. That is a tested lemma (test_layer_bus.py section C),
# not prose.
@dataclass(frozen=True)
class BusMode:
    x_f16: bool = False
    rope_in_f16: bool = False
    scale_f16grade: bool = False


BUS_OFF = BusMode()
BUS_ON = BusMode(x_f16=True, rope_in_f16=True, scale_f16grade=True)


def _grade_rows(comp: np.ndarray) -> np.ndarray:
    """f16_grade over a per-row composite column vector (shape [T,1])."""
    c = np.asarray(comp, dtype=np.float64)
    return np.array([[f16_grade(float(v))] for v in c[:, 0]], dtype=np.float64)


# ═══════════════════════ RoPE cos/sin LUT (C-ROPE) ═══════════════════════════

def _cos_sin_lut_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """256 (base, slope) pairs each for cos and sin over a FULL turn.

    base[k]  = RNE(f(2π·k/256)·2^CS_OUT_FRAC)          at segment start
    slope[k] = RNE((f(2π·(k+1)/256) − f(2π·k/256))·2^CS_OUT_FRAC)
    for f ∈ {cos, sin}.  Signed (cos/sin range [-1,1]).
    """
    n = 1 << CS_LUT_BITS
    edges = 2.0 * math.pi * np.arange(n + 1) / n
    out = []
    for f in (np.cos, np.sin):
        v = f(edges)
        base = rne(v[:-1] * (1 << CS_OUT_FRAC))
        slope = rne((v[1:] - v[:-1]) * (1 << CS_OUT_FRAC))
        out += [base.astype(np.int64), slope.astype(np.int64)]
    return tuple(out)  # cos_base, cos_slope, sin_base, sin_slope


_COS_BASE, _COS_SLOPE, _SIN_BASE, _SIN_SLOPE = _cos_sin_lut_tables()


def _cs_interp(u_q: np.ndarray, base: np.ndarray, slope: np.ndarray) -> np.ndarray:
    u = np.asarray(u_q, dtype=np.int64) % (1 << PH_FRAC)     # periodic reduction
    idx = np.minimum(u >> CS_FRAC_REM, (1 << CS_LUT_BITS) - 1)
    frac = u - (idx << CS_FRAC_REM)
    return base[idx] + ((slope[idx] * frac) >> CS_FRAC_REM)  # signed arith. shift


def cos_fx(u_q: np.ndarray) -> np.ndarray:
    """cos of a Q0.14 turn-fraction phase code → signed Q2.14. Bit-exact LUT."""
    return _cs_interp(u_q, _COS_BASE, _COS_SLOPE)


def sin_fx(u_q: np.ndarray) -> np.ndarray:
    """sin of a Q0.14 turn-fraction phase code → signed Q2.14. Bit-exact LUT."""
    return _cs_interp(u_q, _SIN_BASE, _SIN_SLOPE)


def rope_theta(head_dim: int, base: float = ROPE_THETA_BASE) -> np.ndarray:
    """θ_i = base^(−2i/head_dim), i = 0..head_dim/2−1 (float64 constants).

    base is a model parameter, folded in float64 at phase-table build time
    (C-ROPE: the per-(m,i) table is precomputed and quantized ONCE, so the
    base never touches the cos/sin LUT or any RTL contract). Default 10000
    (Llama convention, the frozen test anchor); Qwen2/2.5 ships 1000000
    (mlx/HF config rope_theta) — set LayerWeights.rope_theta_base for parity
    with real Qwen weights.
    """
    assert head_dim % 2 == 0 and head_dim >= 2
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    return float(base) ** (-2.0 * i / float(head_dim))


def rope_phase_q(m, theta: np.ndarray) -> np.ndarray:
    """Per-(position, channel-pair) phase code u_q ∈ [0, 2^PH_FRAC).

    m may be a scalar or a [T]/[T,1] array; theta is [half]. Result broadcasts
    to m ⊗ theta shape. u = frac(m·θ / 2π); u_q = RNE(u·2^14) mod 2^14.
    """
    m = np.asarray(m, dtype=np.float64)
    ang = np.multiply.outer(m, theta) if m.ndim else m * theta
    u = np.mod(ang / (2.0 * math.pi), 1.0)
    return rne(u * (1 << PH_FRAC)) % (1 << PH_FRAC)


def rope_fx(x: np.ndarray, m, theta: np.ndarray | None = None) -> np.ndarray:
    """Fixed-point RoPE on a row (or rows) x[..., head_dim] at position(s) m.

    HF half-split convention (C-ROPE). Returns the rotated row(s) narrowed to
    the fp16 bus (float64 of the fp16 value). For 2-D x, m is per-row [T].
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    half = d // 2
    if theta is None:
        theta = rope_theta(d)
    u_q = rope_phase_q(m, theta)
    c = cos_fx(u_q).astype(np.float64) / (1 << CS_OUT_FRAC)
    s = sin_fx(u_q).astype(np.float64) / (1 << CS_OUT_FRAC)
    x1 = x[..., :half]
    x2 = x[..., half:]
    lo = x1 * c - x2 * s
    hi = x2 * c + x1 * s
    return _f16(np.concatenate([lo, hi], axis=-1))


def rope_ref(x: np.ndarray, m, theta: np.ndarray | None = None) -> np.ndarray:
    """Float64 RoPE yardstick: true cos/sin, no fp16 narrowing."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    half = d // 2
    if theta is None:
        theta = rope_theta(d)
    m = np.asarray(m, dtype=np.float64)
    ang = np.multiply.outer(m, theta) if m.ndim else m * theta
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


# ═══════════════════════════ SiLU LUT (C-SILU) ═══════════════════════════════

def _silu_lut_tables() -> tuple[np.ndarray, np.ndarray]:
    """256 (base, slope) pairs for SiLU over [-8, 8]. Signed Q4.12."""
    n = 1 << SILU_LUT_BITS
    seg = (2.0 * SILU_DOMAIN) / n
    edges = -SILU_DOMAIN + seg * np.arange(n + 1)
    v = edges / (1.0 + np.exp(-edges))                       # x·sigmoid(x)
    base = rne(v[:-1] * (1 << SILU_OUT_FRAC))
    slope = rne((v[1:] - v[:-1]) * (1 << SILU_OUT_FRAC))
    return base.astype(np.int64), slope.astype(np.int64)


_SILU_BASE, _SILU_SLOPE = _silu_lut_tables()


def silu_fx(x_q: np.ndarray) -> np.ndarray:
    """SiLU of Q5.10 inputs → signed Q4.12. Bit-exact LUT (clamped [-8,8])."""
    x = np.asarray(x_q, dtype=np.int64)
    lim = SILU_DOMAIN * (1 << SILU_IN_FRAC)                  # 8192
    xcl = np.clip(x, -int(lim), int(lim))
    off = xcl + int(lim)                                     # 0..16384
    idx = np.minimum(off >> SILU_FRAC_REM, (1 << SILU_LUT_BITS) - 1)
    frac = off - (idx << SILU_FRAC_REM)
    return _SILU_BASE[idx] + ((_SILU_SLOPE[idx] * frac) >> SILU_FRAC_REM)


def silu_apply(x_real: np.ndarray) -> np.ndarray:
    """Real-valued SiLU via the Q5.10 feeder + LUT → float64 (fp16-grid).

    Quantizes x to Q5.10 (RNE), applies silu_fx, dequantizes /2^12, narrows to
    fp16 (the SwiGLU product bus). The single documented input rounding.
    """
    xq = rne(np.asarray(x_real, dtype=np.float64) * (1 << SILU_IN_FRAC))
    y = silu_fx(xq).astype(np.float64) / (1 << SILU_OUT_FRAC)
    return _f16(y)


def silu_ref(x_real: np.ndarray) -> np.ndarray:
    """Float64 SiLU yardstick = x·sigmoid(x)."""
    x = np.asarray(x_real, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


# ═══════════════════════════ decoder layer ═══════════════════════════════════

@dataclass
class LayerWeights:
    """INT8 weights + per-tensor float64 dequant scales for one decoder layer.

    Projection weights are stored [in_dim, out_dim] (gemm_i8 second operand).
    Wq: [D_model, D_model]; Wk/Wv: [D_model, H_kv·head_dim] (GQA, S6 — MHA is
    the H_kv == H special case); Wo: [D_model, D_model]; Wg/Wu: [D_model,
    d_ffn]; Wd: [d_ffn, D_model].  gamma1/gamma2: [D_model] RMSNorm gains,
    Q2.13 int (as rmsnorm_fx expects).  H query heads, head_dim = D_model / H;
    H_kv KV heads (0 → = H); query head h reads KV head h // (H // H_kv).

    bq/bk/bv (S6, Qwen2/2.5): OPTIONAL projection biases in REAL units on the
    fp16 grid — Qwen puts biases on q/k/v only (o/gate/up/down have none).
    They are added to the dequantized projection output BEFORE the single
    fp16 bus narrowing (K/V write port) / before RoPE (q), matching HF's
    rope(Wx + b) order. None → no bias (all pre-Qwen cases unchanged).
    """
    Wq: np.ndarray; Wk: np.ndarray; Wv: np.ndarray; Wo: np.ndarray
    Wg: np.ndarray; Wu: np.ndarray; Wd: np.ndarray
    s_wq: float; s_wk: float; s_wv: float; s_wo: float
    s_wg: float; s_wu: float; s_wd: float
    gamma1: np.ndarray; gamma2: np.ndarray
    H: int; head_dim: int
    H_kv: int = 0
    bq: np.ndarray = None; bk: np.ndarray = None; bv: np.ndarray = None
    rope_theta_base: float = ROPE_THETA_BASE  # Qwen2/2.5: 1e6 (config value)


@dataclass
class ChunkedHead:
    """One query head's C-CHUNK result (S8): T > T_ROW_MAX ran as per-chunk
    attention_core jobs merged by softmax_merge_weights. Presents the same
    out_hat interface as AttnCore; the per-chunk AttnCore jobs (each a
    T ≤ 128 tile job, individually RTL-replayable) are in `cores`."""
    out_hat: np.ndarray
    cores: list
    merge: dict


@dataclass
class LayerFx:
    """All fixed-point intermediates of one decoder-layer decode step.

    GQA (S6): K_real/V_real/K_rope span [T, H_kv·head_dim] — the KV heads —
    while q_real/q_rope/attn span [D_model] (query heads).
    C-CHUNK (S8): for T > CHUNK_T_MAX each heads[] entry is a ChunkedHead
    (per-chunk AttnCore jobs + merge state) instead of an AttnCore."""
    D_model: int = 0
    d_ffn: int = 0
    H: int = 0
    H_kv: int = 0
    head_dim: int = 0
    T: int = 0
    tier: str = ""
    h: np.ndarray = None            # [T+1, D_model] Q7.8 (RMSNorm-1)
    h8: np.ndarray = None
    s_h: np.ndarray = None
    q_real: np.ndarray = None       # [D_model] projected query (fp16 bus)
    K_real: np.ndarray = None       # [T, D_model]
    V_real: np.ndarray = None
    q_rope: np.ndarray = None       # [D_model] RoPE'd (fp16)
    K_rope: np.ndarray = None       # [T, D_model] RoPE'd (fp16)
    heads: list = field(default_factory=list)   # per-head AttnCore
    attn: np.ndarray = None         # [D_model] concat head outputs
    attn_proj: np.ndarray = None    # [D_model] after W_O + epilogue (real)
    r1: np.ndarray = None           # [D_model] residual-1 (fp16 bus)
    h2: np.ndarray = None           # [D_model] Q7.8 (RMSNorm-2)
    gate_real: np.ndarray = None    # [d_ffn]
    up_real: np.ndarray = None
    swiglu: np.ndarray = None       # [d_ffn] silu(gate)*up (fp16)
    ffn_out: np.ndarray = None      # [D_model] after W_d + epilogue (real)
    r2: np.ndarray = None           # [D_model] layer output (fp16 bus)


def _proj_epilogue(vec_real: np.ndarray, W8: np.ndarray, s_w: float,
                   grade: bool = False) -> tuple[np.ndarray, float]:
    """Quantize a real vector (C-1), gemm_i8 against W8, C-2 requant epilogue.

    Returns (out_real [N], s_out) where out_real = o8·s_out. Mirrors the
    attention output-projection epilogue in attention_core (calib_requant +
    requant_i32_to_i8, no-clip asserted).

    grade (C-LBUS NP-s): narrow s_out to the fp16-GRADE fp32 the tile's
    sideband contract requires (apex_scale_quant C2) BEFORE the dequant, so
    o8·s_out is exact in fp32/float64 and bus-realizable. Default False ==
    today's float64 s_out, bit-identical.
    """
    a8, s_a = quant_rows_i8(np.asarray(vec_real, dtype=np.float64)[None, :])
    acc = gemm_i8_ksplit(a8, np.asarray(W8, dtype=np.int64))[0].astype(np.int64)
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = calib_requant(amax)
    o8 = requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    assert np.max(np.abs(o8)) < 128, "projection epilogue clip"
    s_a_val = float(f16_bits_to_f64(np.array([s_a[0]]))[0])
    # real = acc·s_a·s_w and acc ≈ o8·2^shift/scale ⇒ s_out folds BOTH the
    # activation feeder scale s_a AND the per-tensor weight scale s_w.
    s_out = s_a_val * float(s_w) * float(1 << shift) / float(scale)
    if grade:
        s_out = f16_grade(s_out)
    return o8.astype(np.float64) * s_out, s_out


def decoder_layer_fx(X: np.ndarray, w: LayerWeights, tier: str,
                     G: int = 128, outlier_idx=(),
                     q_pos: int | None = None,
                     bus: BusMode | None = None) -> LayerFx:
    """Full fixed-point decoder-layer decode step (graph in the module docstring).

    X: [T+1, D_model] float64 token activations — rows 0..T-1 are the cached
    context, row T is the decode token (position m = T). Produces the layer
    output for the decode token (r2, [D_model]).

    bus (C-LBUS, D-030 — see the BusMode block above): hardware bus-grade
    composition flags. Default None == BUS_OFF == bit-identical legacy
    behaviour (gated by tests/test_layer_bus.py section A). BUS_ON is the
    IB-LAYER L4 arbiter once the owner sign-off is recorded. v1 limit:
    scale_f16grade requires T <= CHUNK_T_MAX (the tile envelope) — the
    chunked-merge bus composition is explicitly out of C-LBUS v1 scope.

    q_pos (S8, default None → T): RoPE position of the decode-token query.
    The legacy convention (q at m = T over T cached rows) EXCLUDES the decode
    token from its own attention window. HF/Qwen decode is SELF-INCLUSIVE:
    token t attends to keys 0..t with q at m = t. That step is composed
    WITHOUT new plumbing by duplicating the current token as the decode row —
    X = [x_0..x_t, x_t] (T = t+1 context rows INCLUDING x_t) with q_pos = t —
    so K/V cover positions 0..t and q sits at its true position. Gated by
    test_7b_plumbing section G against an independently-written HF-style
    reference; q_pos=None/T is bit-identical to the pre-S8 behavior.
    """
    bus = bus or BUS_OFF
    X = np.asarray(X, dtype=np.float64)
    if bus.x_f16:
        X = _f16(X)                                          # C-LBUS NP-x
    Tp1, D_model = X.shape
    T = Tp1 - 1
    assert not (bus.scale_f16grade and T > CHUNK_T_MAX), \
        "C-LBUS v1: chunked-merge bus composition out of scope (T <= 128)"
    H, hd = w.H, w.head_dim
    H_kv = w.H_kv or H                                       # GQA (S6); 0 → MHA
    assert H * hd == D_model, "H·head_dim must equal D_model"
    assert H % H_kv == 0, "H must be a multiple of H_kv (GQA groups)"
    kv_dim = H_kv * hd
    assert w.Wk.shape == (D_model, kv_dim) and w.Wv.shape == (D_model, kv_dim)
    d_ffn = w.Wg.shape[1]
    r = LayerFx(D_model=D_model, d_ffn=d_ffn, H=H, H_kv=H_kv, head_dim=hd,
                T=T, tier=tier)

    # ── RMSNorm-1 (C-RMSW wide composition; == rmsnorm_fx for D ≤ 128 pow-2) ─
    x8, _s_x = quant_rows_i8(X)
    g1 = [int(v) for v in np.asarray(w.gamma1)]
    r.h = np.empty((Tp1, D_model), dtype=np.int64)
    for t in range(Tp1):
        y, _, _ = rmsnorm_fx_wide([int(v) for v in x8[t]], g1)
        r.h[t] = y
    r.h8, r.s_h = quant_rows_i8(r.h.astype(np.float64) / 256.0)
    s_h = f16_bits_to_f64(r.s_h)

    # ── Q/K/V projections (C-KSPLIT jobs; S-2 KV-write dequant to fp16 bus;
    #    Qwen biases added in real units BEFORE the single narrowing) ─────────
    acc_q = gemm_i8_ksplit(r.h8[T:], np.asarray(w.Wq, dtype=np.int64))[0]
    acc_k = gemm_i8_ksplit(r.h8[:T], np.asarray(w.Wk, dtype=np.int64))
    acc_v = gemm_i8_ksplit(r.h8[:T], np.asarray(w.Wv, dtype=np.int64))
    comp_q = float(s_h[T]) * w.s_wq
    comp_k = s_h[:T, None] * w.s_wk
    comp_v = s_h[:T, None] * w.s_wv
    if bus.scale_f16grade:                                   # C-LBUS NP-s
        comp_q = f16_grade(comp_q)
        comp_k = _grade_rows(comp_k)
        comp_v = _grade_rows(comp_v)
    r.q_real = acc_q.astype(np.float64) * comp_q
    r.K_real = acc_k.astype(np.float64) * comp_k
    r.V_real = acc_v.astype(np.float64) * comp_v
    if w.bq is not None:
        r.q_real = r.q_real + np.asarray(w.bq, dtype=np.float64)
    if w.bk is not None:
        r.K_real = r.K_real + np.asarray(w.bk, dtype=np.float64)[None, :]
    if w.bv is not None:
        r.V_real = r.V_real + np.asarray(w.bv, dtype=np.float64)[None, :]
    if bus.rope_in_f16:                                      # C-LBUS NP-r
        # the S-2 write-bus narrowing, applied AFTER the bias add (HF
        # rope(Wx+b) order) and BEFORE rotation — the hardware path
        # projection -> S-2 dequant/narrow -> rope_row -> KVQ store.
        # V is untouched: its S-2 narrowing already sits at the
        # attention-core input below, unchanged in both modes.
        r.q_real = _f16(r.q_real)
        r.K_real = _f16(r.K_real)

    # ── RoPE (q per query head, K per KV head) + per-head attention_core ─────
    theta = rope_theta(hd, w.rope_theta_base)
    pos_k = np.arange(T)
    m_q = T if q_pos is None else int(q_pos)
    r.q_rope = np.empty(D_model, dtype=np.float64)
    r.K_rope = np.empty((T, kv_dim), dtype=np.float64)
    r.attn = np.empty(D_model, dtype=np.float64)
    group = H // H_kv
    for g_i in range(H_kv):
        sl = slice(g_i * hd, (g_i + 1) * hd)
        r.K_rope[:, sl] = rope_fx(r.K_real[:, sl], pos_k, theta)  # [T,hd] fp16
    for h_i in range(H):
        sl_q = slice(h_i * hd, (h_i + 1) * hd)
        sl_kv = slice((h_i // group) * hd, (h_i // group + 1) * hd)
        q_h = rope_fx(r.q_real[sl_q], m_q, theta)            # [hd] fp16
        r.q_rope[sl_q] = q_h
        K_f16 = f64_to_f16_bits(r.K_rope[:, sl_kv]).reshape(T, hd)  # exact
        V_f16 = f64_to_f16_bits(r.V_real[:, sl_kv]).reshape(T, hd)
        if T > CHUNK_T_MAX:
            # C-CHUNK (S8): per-chunk T ≤ T_ROW_MAX tile jobs + host merge
            # (softmax_merge_weights); chunk = default 128 = G keeps key
            # groups aligned with the unchunked cache layout.
            y, cores, mg = attention_chunked(q_h, K_f16, V_f16, tier, G=G,
                                             outlier_idx=list(outlier_idx))
            r.heads.append(ChunkedHead(out_hat=y, cores=cores, merge=mg))
            r.attn[sl_q] = y
        else:
            core = attention_core(q_h, K_f16, V_f16, tier, G=G,
                                  outlier_idx=list(outlier_idx))
            r.heads.append(core)
            if bus.scale_f16grade:                           # C-LBUS NP-s
                # the per-head output leaves the attention step as o8 codes;
                # the LAYER-side dequant composite (o8 -> the o-proj C-1
                # feeder) is fp16-graded. attention_core itself is untouched.
                r.attn[sl_q] = core.o8.astype(np.float64) * f16_grade(core.s_out)
            else:
                r.attn[sl_q] = core.out_hat

    # ── output projection W_O + residual-1 ───────────────────────────────────
    r.attn_proj, _ = _proj_epilogue(r.attn, w.Wo, w.s_wo,
                                    grade=bus.scale_f16grade)
    r.r1 = _f16(X[T] + r.attn_proj)                          # C-6 residual

    # ── RMSNorm-2 ────────────────────────────────────────────────────────────
    r1_8, _ = quant_rows_i8(r.r1[None, :])
    g2 = [int(v) for v in np.asarray(w.gamma2)]
    h2, _, _ = rmsnorm_fx_wide([int(v) for v in r1_8[0]], g2)
    r.h2 = np.asarray(h2, dtype=np.int64)
    h2_8m, s_h2m = quant_rows_i8(r.h2.astype(np.float64)[None, :] / 256.0)
    h2_8 = h2_8m
    s_h2 = float(f16_bits_to_f64(np.array([s_h2m[0]]))[0])

    # ── SwiGLU FFN ───────────────────────────────────────────────────────────
    acc_g = gemm_i8_ksplit(h2_8, np.asarray(w.Wg, dtype=np.int64))[0]
    acc_u = gemm_i8_ksplit(h2_8, np.asarray(w.Wu, dtype=np.int64))[0]
    comp_g = s_h2 * w.s_wg
    comp_u = s_h2 * w.s_wu
    if bus.scale_f16grade:                                   # C-LBUS NP-s
        comp_g = f16_grade(comp_g)
        comp_u = f16_grade(comp_u)
    r.gate_real = acc_g.astype(np.float64) * comp_g
    r.up_real = acc_u.astype(np.float64) * comp_u
    r.swiglu = _f16(silu_apply(r.gate_real) * r.up_real)     # SiLU gate * up
    r.ffn_out, _ = _proj_epilogue(r.swiglu, w.Wd, w.s_wd,
                                  grade=bus.scale_f16grade)

    # ── residual-2 = layer output ────────────────────────────────────────────
    r.r2 = _f16(r.r1 + r.ffn_out)
    return r


# ═══════════════════════ float64 yardstick (the reference) ═══════════════════

@dataclass
class LayerRef:
    h_ref: np.ndarray = None        # [T+1, D_model] RMSNorm-1
    attn_ref: np.ndarray = None     # [D_model] concat head outputs
    attn_proj_ref: np.ndarray = None
    r1_ref: np.ndarray = None
    h2_ref: np.ndarray = None
    gate_ref: np.ndarray = None
    up_ref: np.ndarray = None
    ffn_out_ref: np.ndarray = None
    y_ref: np.ndarray = None        # [D_model] layer output


def decoder_layer_ref(X: np.ndarray, w: LayerWeights,
                      q_pos: int | None = None) -> LayerRef:
    """Float64 full-precision decoder layer: same DEQUANTIZED weights, true
    rmsnorm/softmax/rope/silu, no fp16 narrowings (the yardstick).

    q_pos mirrors decoder_layer_fx (S8): RoPE position of the decode-token
    query, default None → T (legacy)."""
    X = np.asarray(X, dtype=np.float64)
    Tp1, D_model = X.shape
    T = Tp1 - 1
    H, hd = w.H, w.head_dim
    H_kv = w.H_kv or H                                       # GQA (S6)
    ref = LayerRef()

    g1 = np.asarray(w.gamma1, dtype=np.float64) / (1 << 13)
    ref.h_ref = rmsnorm_ref(X, g1)
    Wq = np.asarray(w.Wq, dtype=np.float64) * w.s_wq
    Wk = np.asarray(w.Wk, dtype=np.float64) * w.s_wk
    Wv = np.asarray(w.Wv, dtype=np.float64) * w.s_wv
    q = ref.h_ref[T] @ Wq
    K = ref.h_ref[:T] @ Wk
    V = ref.h_ref[:T] @ Wv
    if w.bq is not None:
        q = q + np.asarray(w.bq, dtype=np.float64)
    if w.bk is not None:
        K = K + np.asarray(w.bk, dtype=np.float64)[None, :]
    if w.bv is not None:
        V = V + np.asarray(w.bv, dtype=np.float64)[None, :]

    theta = rope_theta(hd, w.rope_theta_base)
    pos_k = np.arange(T)
    m_q = T if q_pos is None else int(q_pos)
    group = H // H_kv
    ref.attn_ref = np.empty(D_model, dtype=np.float64)
    for h_i in range(H):
        sl_q = slice(h_i * hd, (h_i + 1) * hd)
        sl_kv = slice((h_i // group) * hd, (h_i // group + 1) * hd)
        q_h = rope_ref(q[sl_q], m_q, theta)                  # [hd]
        K_h = rope_ref(K[:, sl_kv], pos_k, theta)             # [T,hd]
        scores = K_h @ q_h / math.sqrt(float(hd))
        p = softmax_ref(scores)
        ref.attn_ref[sl_q] = p @ V[:, sl_kv]

    Wo = np.asarray(w.Wo, dtype=np.float64) * w.s_wo
    ref.attn_proj_ref = ref.attn_ref @ Wo
    ref.r1_ref = X[T] + ref.attn_proj_ref

    g2 = np.asarray(w.gamma2, dtype=np.float64) / (1 << 13)
    ref.h2_ref = rmsnorm_ref(ref.r1_ref, g2)
    Wg = np.asarray(w.Wg, dtype=np.float64) * w.s_wg
    Wu = np.asarray(w.Wu, dtype=np.float64) * w.s_wu
    Wd = np.asarray(w.Wd, dtype=np.float64) * w.s_wd
    ref.gate_ref = ref.h2_ref @ Wg
    ref.up_ref = ref.h2_ref @ Wu
    ref.ffn_out_ref = (silu_ref(ref.gate_ref) * ref.up_ref) @ Wd
    ref.y_ref = ref.r1_ref + ref.ffn_out_ref
    return ref
