"""weight_codec.py — APEX's bit-exact native-W4 weight-path model (B3).

The reference (arbiter) for the W4 weight feeder: packed 4-bit projection
weights unpacked + dequantized to the INT8 operand stream the MXE array
already understands. Per ARCHITECTURE.md C-4 the array sees exactly ONE
dtype (INT8); W4 weights unpack+dequant AT THE FEEDER, never inside PEs.

Built only on already-trusted primitives — the C-1 INT4 machinery of
cq_codec (scale_from_amax / quant_codes / pack_int4 / unpack_int4 /
dequant_f32) and, for realization (B), the D-021 seam feeder's INT8 requant
rule (attention.quant_rows_i8). Nothing here re-derives a numeric rule.

═══ NORMATIVE STREAM CONTRACT (this module is the arbiter) ═══════════════════

The MXE weight loader consumes, per job, exactly KB*N INT8 lane8 beats where
KB = ceil(K/8) and N = desc.n_dim (mxe_ctrl.sv:46-59, w_ready :190, lane
assembly :206-213):

  S1. Beat order is chunk-major, column-minor: beat n = p*N + c for
      p = 0..KB-1, c = 0..N-1.
  S2. Within beat (p,c), lane r carries B[8p+r][c] at data[8r +: 8].
      Rows 8p+r >= K on the final chunk are ZERO (the loader zero-masks
      them via `rows_valid`; golden emits the same zeros so the byte
      streams compare directly).
  S3. Flat element index of (beat n, lane r) is e = 8n + r. The flat
      element stream therefore has KB*N*8 entries — always a multiple of 8.
  S4. Packed-W4 beat j carries flat elements 16j..16j+15, i.e. exactly the
      two INT8 beats 2j and 2j+1. Byte i of the packed beat (at
      data[8i +: 8]) is cq_codec.pack_int4's rule applied to that flat
      order: low nibble = element 16j+2i, high nibble = element 16j+2i+1.
  S5. Packed beats per job = ceil(KB*N / 2). When KB*N is ODD the final
      packed beat's bytes 4..7 are PADDING (emitted as zero here,
      don't-care for the feeder, never turned into output beats).
  S6. Therefore  emitted_i8_beats == 2*consumed_packed_beats - (KB*N & 1).
      The unqualified "emitted == 2 x consumed" is FALSE whenever KB*N is
      odd, which is legal (N ranges 1..8, KB 1..256). It holds at the
      shipped D=64/N=8 config only because KB*N is even there.

═══ GROUP GEOMETRY & THE TWO REALIZATIONS ════════════════════════════════════

A quantization group is a contiguous run of K rows within ONE output column
(scales cannot vary along a column's K extent unless the feeder absorbs
them — see below). Three geometries:

  "tile"  one scale for the whole K x N job tile
  "col"   one scale per output column, full K
  int G   G K-rows x 1 column

Which geometries are legal depends on where the scale is applied:

  (A) SCALE-IN-EPILOGUE — feeder output = the INT4 code sign-extended to
      INT8; the group's fp16 scale rides downstream as a factor of the
      composite, exactly where the per-tensor s_w rides today
      (transformer.py:369, s_out = s_a * s_w * 2^shift / scale).
      For a scale to leave the contraction sum
          acc[m][c] = SUM_k A[m][k] * B[k][c]
      as a single downstream factor it must be constant over BOTH k and c.
      => realization (A) REQUIRES group == "tile". Per-column grouping
      needs N=1 tiling; per-K-chunk grouping cannot work at all (K-chunks
      accumulate inside one job, mxe_ctrl.sv:334-363).

  (B) DEQUANT-THEN-REQUANT — the feeder dequants code*s_group to real and
      requantizes the tile to INT8 against ONE per-tile fp16 scale, which
      then rides downstream. This is the D-021 seam_feeder_quant chain
      reused for weights. Any group geometry is legal because the fine
      scales are absorbed into the codes before the array sees them.

NOTE on scale grade (found while reading the shipped path): the downstream
consumer apex_scale_quant requires sideband scales to be fp32 with an
fp16-GRADE mantissa — frac[12:0] == 0, <= 11 significant bits — else
scale_error pulse+sticky (apex_scale_quant.sv:29-31, :227). Today's flows
satisfy that only because s_w is a power of two (gen_l3_vectors.py:589,
S_W = 2**-6). An fp16 W4 scale multiplied into s_a*s_w makes the composite
~22 significant bits and TRIPS that error. `pow2_scale=True` snaps the
group scale up to the next power of two, which keeps the composite
fp16-grade with no change to apex_scale_quant; its accuracy cost is
measured in tests/test_weight_codec.py section F.

INT8 -> fp16 narrowing is LOSSLESS here and adds no quantization hop: the
stored weights are INT8 codes in [-127,127] and every such integer is
exactly representable in fp16 (11-bit mantissa covers |x| <= 2048). The
per-tensor s_w is never narrowed — it stays a float64 factor of the
composite, exactly as today.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attention import quant_rows_i8
from .cq_codec import (amax_bits, dequant_f32, pack_int4, quant_codes,
                       scale_from_amax, unpack_int4)
from .fp import f16_bits_to_f64, f32_bits_to_f64, f64_to_f16_bits, rne

LANES = 8          # lane8_beat_t lanes (apex_pkg.sv:57-60)
W4_BITS = 4        # C-1 symmetric signed INT4, clamp [-8,7]
W8_BITS = 8        # the dtype the MXE array sees (C-4)


# ── §0 job geometry (mirrors mxe_ctrl's beat accounting) ─────────────────────

def kb_of(K: int) -> int:
    """K-chunks per job = ceil(K/8) (mxe_ctrl.sv:169-175)."""
    return (K + LANES - 1) // LANES


def n_wgt_beats(K: int, N: int) -> int:
    """INT8 weight beats per job = ceil(K/8)*N (mxe_ctrl.sv:46-59)."""
    return kb_of(K) * N


def n_packed_beats(K: int, N: int) -> int:
    """Packed-W4 beats per job = ceil(KB*N / 2)  (S5)."""
    return (n_wgt_beats(K, N) + 1) // 2


def stream_index(K: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat emission order -> (row, col) per S1-S3.

    Returns two [KB*N*8] int64 arrays. row >= K marks a zero-masked pad
    element on the final K-chunk (S2).
    """
    KB = kb_of(K)
    p = np.repeat(np.arange(KB, dtype=np.int64), N * LANES)
    c = np.tile(np.repeat(np.arange(N, dtype=np.int64), LANES), KB)
    r = np.tile(np.arange(LANES, dtype=np.int64), KB * N)
    return p * LANES + r, c


# ── §1 grouping ──────────────────────────────────────────────────────────────

def group_ids(K: int, N: int, group) -> tuple[np.ndarray, int]:
    """Group index per weight element -> ([K,N] int64, n_groups).

    group: "tile" | "col" | int G (G K-rows x 1 column).
    """
    if group == "tile":
        return np.zeros((K, N), dtype=np.int64), 1
    if group == "col":
        return np.tile(np.arange(N, dtype=np.int64), (K, 1)), N
    G = int(group)
    if G <= 0:
        raise ValueError(f"illegal group {group!r}")
    ng_k = (K + G - 1) // G
    kidx = (np.arange(K, dtype=np.int64) // G)[:, None]
    cidx = np.arange(N, dtype=np.int64)[None, :]
    return cidx * ng_k + kidx, N * ng_k


# ── §2 scale grade (apex_scale_quant C2 — see module docstring) ──────────────

def snap_pow2_f16(sb: np.uint16) -> np.uint16:
    """Snap a positive fp16 scale UP to the next power of two.

    Snapping UP (never down) keeps |x/s| <= qmax, so no code that was in
    range can be pushed into the clamp. Infinities/zero pass through.

    NOTE: measurement (tests section F) shows this is the WRONG mitigation
    for the apex_scale_quant fp16-grade constraint — it throws away up to a
    full bit of INT4 range and costs 1.7-2.1x in projection error. Use
    f16_grade() on the COMPOSITE instead, which costs ~0. Kept because it
    is the only mitigation that needs no host-side change, and because the
    measured cost is what rules it out.
    """
    b = int(sb)
    exp, man = (b >> 10) & 0x1F, b & 0x3FF
    if exp == 0x1F or man == 0:        # inf/NaN, or already a power of two
        return np.uint16(b)
    if exp == 0:                       # subnormal -> smallest normal
        return np.uint16(1 << 10)
    return np.uint16((exp + 1) << 10)  # exp+1 == 0x1F gives +inf, correct


def f16_grade(x: float) -> float:
    """Narrow a composite scale to apex_scale_quant's fp16-GRADE fp32 form.

    C2 (apex_scale_quant.sv:29-31, :227) requires the fp32 sideband scale to
    carry <= 11 significant bits (frac[12:0] == 0). Today's composites
    satisfy it because s_w is a power of two (gen_l3_vectors.py:589); a W4
    group scale breaks that. Rounding the COMPOSITE to 11 significant bits
    (RNE) restores the contract at a relative cost of ~2^-12, which section
    F measures as negligible next to W4's own error. fp32 EXPONENT range is
    retained, so this is not an fp16 cast.
    """
    b = int(np.asarray(np.float64(x), dtype=np.float32).view(np.uint32))
    exp = (b >> 23) & 0xFF
    if exp == 0 or exp == 0xFF:        # zero/subnormal/inf/NaN: leave alone
        return float(np.uint32(b).view(np.float32))
    low = b & 0x1FFF                   # the 13 bits C2 requires to be zero
    b &= ~0x1FFF
    if low > 0x1000 or (low == 0x1000 and (b >> 13) & 1):   # RNE (C-2)
        b += 0x2000                    # carry into the exponent is correct
    return float(np.uint32(b).view(np.float32))


# ── §3 compression: INT8 weights -> INT4 codes + fp16 group scales ──────────

@dataclass
class WeightBlob:
    """One job's weight tile in the native-W4 form."""
    K: int
    N: int
    group: object
    pow2: bool
    n_groups: int
    gid: np.ndarray = field(default=None)      # [K, N] group index
    scales: np.ndarray = field(default=None)   # [n_groups] fp16 bits
    codes: np.ndarray = field(default=None)    # [K, N] int64 in [-8, 7]
    packed: np.ndarray = field(default=None)   # [n_packed, 8] uint8 beats


def weights_to_f16(W8: np.ndarray) -> np.ndarray:
    """INT8 weight codes -> fp16 bit patterns. LOSSLESS (see docstring).

    Raises if any input is outside the exactly-representable range, so the
    no-extra-hop property is asserted rather than assumed.
    """
    W = np.asarray(W8, dtype=np.int64)
    if W.size and int(np.max(np.abs(W))) > 2048:
        raise ValueError("weights_to_f16: |w| > 2048 is not fp16-exact")
    bits = f64_to_f16_bits(W.astype(np.float64))
    return bits.reshape(W.shape)


def compress_weights_w4(W8: np.ndarray, group="tile",
                        pow2_scale: bool = False) -> WeightBlob:
    """[K,N] INT8 weight tile -> W4Blob (codes + fp16 group scales + packed).

    Per group: the C-1 INT4 chain from cq_codec — amax_bits ->
    scale_from_amax(bits=4) -> quant_codes(bits=4) — i.e. the SAME
    [-8,7] / EPS=2^-14 / round-half-to-even contract as the KV keys.
    """
    W8 = np.asarray(W8, dtype=np.int64)
    K, N = W8.shape
    W_f16 = weights_to_f16(W8)
    gid, n_groups = group_ids(K, N, group)

    b = WeightBlob(K=K, N=N, group=group, pow2=bool(pow2_scale),
                   n_groups=n_groups, gid=gid)
    b.scales = np.empty(n_groups, dtype=np.uint16)
    b.codes = np.zeros((K, N), dtype=np.int64)

    for g in range(n_groups):
        m = gid == g
        s = scale_from_amax(amax_bits(W_f16[m]), W4_BITS)
        if pow2_scale:
            s = snap_pow2_f16(s)
        b.scales[g] = s
        b.codes[m] = quant_codes(W_f16[m], s, W4_BITS)

    b.packed = pack_stream(b.codes, K, N)
    return b


# ── §4 packing to the lane8 beat stream (S3-S5) ─────────────────────────────

def code_stream(codes: np.ndarray, K: int, N: int) -> np.ndarray:
    """[K,N] element array -> the flat emission-order stream (S1-S3).

    Elements whose row >= K are the loader's zero-masked pad (S2).
    """
    rows, cols = stream_index(K, N)
    out = np.zeros(rows.shape, dtype=np.int64)
    live = rows < K
    out[live] = np.asarray(codes, dtype=np.int64)[rows[live], cols[live]]
    return out


def pack_stream(codes: np.ndarray, K: int, N: int) -> np.ndarray:
    """[K,N] INT4 codes -> [n_packed, 8] uint8 packed-W4 beats (S4-S5)."""
    flat = code_stream(codes, K, N)
    payload = pack_int4(flat)                      # cq_codec's nibble rule
    npb = n_packed_beats(K, N)
    buf = np.zeros(npb * LANES, dtype=np.uint8)    # S5 padding = 0
    buf[:len(payload)] = payload
    return buf.reshape(npb, LANES)


def beat_words(beats: np.ndarray) -> np.ndarray:
    """[n, 8] uint8 lane array -> [n] uint64 beat words, lane r at bits 8r."""
    b = np.asarray(beats, dtype=np.uint64)
    return np.sum(b << (np.uint64(8) * np.arange(LANES, dtype=np.uint64)),
                  axis=1, dtype=np.uint64)


# ── §5 the feeder: packed W4 in -> INT8 beats out (C-4) ─────────────────────

def wfeed_w4_to_i8(blob: WeightBlob, realization: str = "B"
                   ) -> tuple[np.ndarray, np.uint16]:
    """THE EXACT BYTES THE RTL FEEDER MUST EMIT.

    Returns (i8_stream, scale_f16) where i8_stream is the flat INT8
    emission-order element stream of length KB*N*8 (reshape to [-1, 8] for
    beats, S3) and scale_f16 is the single fp16 scale that rides downstream
    in place of the per-tensor s_w. The real weight is recovered as
        w_real ~= i8_stream * f16(scale_f16) * s_w
    with s_w the unchanged per-tensor float64 factor.

    realization "A": sign-extend the INT4 code to INT8; the group scale
        rides the epilogue. Requires group == "tile" (see docstring).
    realization "B": dequant code*s_group (exact, cq_codec.dequant_f32) then
        requant the tile to one INT8 scale (D-021 quant_rows_i8 rule).
    """
    K, N = blob.K, blob.N
    n_elem = n_wgt_beats(K, N) * LANES
    flat = unpack_int4(blob.packed.reshape(-1), n_elem)   # cq_codec (S4)

    if realization == "A":
        if blob.group != "tile":
            raise ValueError(
                "realization (A) requires group == 'tile': a scale leaves the "
                "contraction as one downstream factor only if it is constant "
                "over both k and c (module docstring)")
        return flat.astype(np.int64), np.uint16(blob.scales[0])

    if realization != "B":
        raise ValueError(f"unknown realization {realization!r}")

    # (B): exact dequant per group, then one per-tile INT8 requant.
    rows, cols = stream_index(K, N)
    live = rows < K
    gid_s = np.full(rows.shape, -1, dtype=np.int64)
    gid_s[live] = blob.gid[rows[live], cols[live]]

    real = np.zeros(rows.shape, dtype=np.float64)
    for g in range(blob.n_groups):
        m = gid_s == g
        if m.any():
            real[m] = f32_bits_to_f64(
                dequant_f32(flat[m], np.uint16(blob.scales[g])))

    q8, s8 = quant_rows_i8(real[None, :])
    return q8[0].astype(np.int64), np.uint16(s8[0])


def wfeed_w4b_to_i8(blob: WeightBlob, s8_bits: np.uint16) -> np.ndarray:
    """W4-B FEEDER oracle (D-031, docs/design/W4B_FEEDER.md): the (B) chain
    with the requant scale GIVEN.

    Same exact per-group dequant as wfeed_w4_to_i8(realization="B"), then
        q8 = clamp(RNE(real / f16(s8_bits)), -128, 127)
    — the D-021 quant_rows_i8 requant rule with the scale as an INPUT: the
    RTL feeder consumes a HOST-COMPUTED stripe-global sideband scale
    (B3_WEIGHT_PATH.md §8.1 — K-split stripes share one epilogue factor, so
    the scale cannot be feeder-derived per job). With s8_bits equal to the
    tile-amax scale this is bit-identical to wfeed_w4_to_i8(blob, "B")
    (the reduction gate in tests/test_w4b_feeder.py). The dequant block is
    deliberately DUPLICATED from wfeed_w4_to_i8 rather than refactored, so
    the landed arbiter's code path stays byte-untouched; the identity gate
    keeps the two from drifting.
    """
    K, N = blob.K, blob.N
    n_elem = n_wgt_beats(K, N) * LANES
    flat = unpack_int4(blob.packed.reshape(-1), n_elem)

    rows, cols = stream_index(K, N)
    live = rows < K
    gid_s = np.full(rows.shape, -1, dtype=np.int64)
    gid_s[live] = blob.gid[rows[live], cols[live]]

    real = np.zeros(rows.shape, dtype=np.float64)
    for g in range(blob.n_groups):
        m = gid_s == g
        if m.any():
            real[m] = f32_bits_to_f64(
                dequant_f32(flat[m], np.uint16(blob.scales[g])))

    sv = float(f16_bits_to_f64(np.array([np.uint16(s8_bits)]))[0])
    return np.clip(rne(real / sv), -128, 127).astype(np.int64)


def feeder_beats(blob: WeightBlob, realization: str = "B"
                 ) -> tuple[np.ndarray, np.uint16]:
    """wfeed_w4_to_i8 shaped as [KB*N, 8] INT8 beats (lane-major, S2)."""
    flat, s = wfeed_w4_to_i8(blob, realization)
    return flat.reshape(-1, LANES), s


# ── §6 PERF accounting (S6) ─────────────────────────────────────────────────

def perf_beats(K: int, N: int) -> tuple[int, int, float]:
    """(consumed_packed, emitted_i8, emitted/consumed) for one job.

    The honest headline: emitted == 2*consumed - (KB*N & 1). The ratio is
    exactly 2.0 only when KB*N is even.
    """
    emitted = n_wgt_beats(K, N)
    consumed = n_packed_beats(K, N)
    return consumed, emitted, emitted / consumed
