"""w4_lane.py — the W4 INGEST LANE's golden: DIRECT host prep (the frozen
D-031 recipe), the tile wire framing, and job-level GEMM composition.

This module is the arbiter for what the combine session added AROUND the
verified W4-B feeder (docs/design/W4_DATAPATH.md):

  * ``prep_w4_direct``  — the ADOPTED host prep: quantize DIRECT from the
    (fp16-grid) source reals, G=32 group scales, per-STRIPE s8 by the
    tile-amax rule, epilogue composite f16_grade(s8 x s_w). The INT8-hop
    prep (quantize to per-tensor INT8 first) is DEPRECATED (-1.7 pt,
    B3_WEIGHT_PATH.md §5 row 6) and deliberately not implemented here.
  * ``gs_words`` / ``pw_words`` — the xw wire framing consumed by
    rtl/top/glue/apex_w4_ingest.sv: SCALES FIRST (4 fp16 LE per 64-bit
    beat, ascending gid, zero-padded tail), then the packed-W4 beats
    (weight_codec S4/S5, element e at bits [4e +: 4]).
  * ``expected_w8`` / ``gemm_w4b`` — the job-level composition: the
    landed per-element oracle ``wfeed_w4b_to_i8`` re-framed to the [K,N]
    matrix the MXE contracts on, then compute.gemm_i8 (+ the C-2
    requant when enabled). Nothing numeric is reimplemented: every
    element flows through the landed arbiters (cq_codec primitives,
    wfeed_w4b_to_i8, gemm_i8, requant_i32_to_i8).

Wire order rationale (normative for the tile lane): the feeder stalls
emission until the current beat's group scale is resident and holds only
a 2-deep pw skid, so packed-before-scales on ONE in-order pipe would
deadlock; scales-first is legal by the feeder's own eager-scale contract
and is what apex_w4_ingest frames. The DDR image may still store scales
after packed per job — fetch order is the producer's choice (two fuel
records), exactly like the walker's descriptor plan (W4B_FEEDER.md
integration note 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cq_codec import amax_bits, quant_codes, scale_from_amax
from .compute import gemm_i8, requant_i32_to_i8
from .fp import f16_bits_to_f64, f64_to_f16_bits
from .weight_codec import (LANES, W4_BITS, WeightBlob, beat_words, f16_grade,
                           group_ids, n_packed_beats, n_wgt_beats, pack_stream,
                           stream_index, wfeed_w4_to_i8, wfeed_w4b_to_i8)

GS_PER_BEAT = 4          # fp16 scales per 64-bit xw beat (LE slots)


def n_gs(K: int, N: int, G: int) -> int:
    """Group scales per job = N * ceil(K/G) (the feeder's ngtot)."""
    return N * ((K + G - 1) // G)


def n_gs_beats(K: int, N: int, G: int) -> int:
    """Scale beats per job = ceil(ngtot / 4)."""
    return (n_gs(K, N, G) + GS_PER_BEAT - 1) // GS_PER_BEAT


# ── DIRECT host prep (the frozen recipe) ─────────────────────────────────────

@dataclass
class W4Prep:
    """One weight STRIPE (full contraction K, one 8-column job tile),
    prepped by the frozen recipe and split into <= K_MAX job segments."""
    K: int
    N: int
    G: int
    blob: WeightBlob = field(default=None)     # whole-stripe blob
    s8: np.uint16 = field(default=None)        # stripe requant scale (fp16)
    s_w: float = 1.0                           # source scale (1.0 = direct)
    composite: float = field(default=None)     # f16_grade(s8 x s_w)
    seg_k: list = field(default_factory=list)  # per-segment K
    segs: list = field(default_factory=list)   # per-segment WeightBlob


def _direct_blob(W_real: np.ndarray, G: int) -> WeightBlob:
    """fp16-grid source reals -> W4 blob via the C-1 INT4 chain per group.

    DIRECT rule: the quantizer sees the (fp16-grid) source values, never a
    per-tensor INT8 intermediate. Same primitives, same [-8,7] / EPS=2^-14 /
    RNE contract as compress_weights_w4 — only the input domain differs
    (reals instead of weights_to_f16(int8)), which is the whole point.
    """
    W_real = np.asarray(W_real, dtype=np.float64)
    K, N = W_real.shape
    W16 = f64_to_f16_bits(W_real).reshape(K, N)
    gid, ng = group_ids(K, N, int(G))

    b = WeightBlob(K=K, N=N, group=int(G), pow2=False, n_groups=ng, gid=gid)
    b.scales = np.empty(ng, dtype=np.uint16)
    b.codes = np.zeros((K, N), dtype=np.int64)
    for g in range(ng):
        m = gid == g
        s = scale_from_amax(amax_bits(W16[m]), W4_BITS)
        b.scales[g] = s
        b.codes[m] = quant_codes(W16[m], s, W4_BITS)
    b.packed = pack_stream(b.codes, K, N)
    return b


def _seg_slice(whole: WeightBlob, G: int, k0: int, k1: int) -> WeightBlob:
    """Slice rows [k0,k1) of a whole-stripe blob into a per-job blob.

    k0 must be G-aligned so the segment's groups are exactly a subset of
    the stripe's groups (scales re-indexed, never re-derived)."""
    assert k0 % G == 0, f"segment start {k0} not {G}-aligned"
    Kseg = k1 - k0
    N = whole.N
    ngk_tot = (whole.K + G - 1) // G
    ngk_seg = (Kseg + G - 1) // G
    g0 = k0 // G

    gid, ng = group_ids(Kseg, N, int(G))
    b = WeightBlob(K=Kseg, N=N, group=int(G), pow2=False, n_groups=ng,
                   gid=gid)
    b.codes = np.asarray(whole.codes)[k0:k1, :].copy()
    b.scales = np.empty(ng, dtype=np.uint16)
    for c in range(N):
        for gl in range(ngk_seg):
            b.scales[c * ngk_seg + gl] = whole.scales[c * ngk_tot + g0 + gl]
    b.packed = pack_stream(b.codes, Kseg, N)
    return b


def prep_w4_direct(W_real: np.ndarray, G: int = 32, s_w: float = 1.0,
                   k_split: int = 2048) -> W4Prep:
    """The frozen-recipe host prep for one stripe (B3 §8 obligations).

    W_real: [K, N<=8] source reals (the mlx-dequantized values; any float
    input is snapped to the fp16 grid first, which is where mlx dequant
    already lives). G: 32 (D-031 ship freeze; 16 stays a validated
    option). s_w: the source tensor scale the epilogue composite carries
    (1.0 when the reals ARE the weights). k_split: job segmentation bound
    (K_MAX; interior segments are rounded down to a G-multiple so group
    boundaries stay aligned).

    Returns codes/scales for the whole stripe, the STRIPE s8 (tile-amax
    rule over the whole stripe — the one scale every K-split job shares,
    B3 §8.1), the graded epilogue composite f16_grade(s8 x s_w) (§8.3),
    and the per-job segment blobs.
    """
    W_real = np.asarray(W_real, dtype=np.float64)
    K, N = W_real.shape
    G = int(G)
    prep = W4Prep(K=K, N=N, G=G, s_w=float(s_w))
    prep.blob = _direct_blob(W_real, G)

    # stripe s8 = the tile-amax scale of the WHOLE stripe's dequant — taken
    # from the landed (B) arbiter itself, so the reduction identity
    # wfeed_w4b_to_i8(blob, s8) == wfeed_w4_to_i8(blob, "B") holds by
    # construction (test_w4b_feeder gate A semantics).
    _, prep.s8 = wfeed_w4_to_i8(prep.blob, realization="B")
    s8_val = float(f16_bits_to_f64(np.array([np.uint16(prep.s8)]))[0])
    prep.composite = f16_grade(s8_val * float(s_w))

    # K-split segmentation (interior segments G-aligned)
    step = (int(k_split) // G) * G
    assert step >= G, f"k_split {k_split} below one group {G}"
    k0 = 0
    while k0 < K:
        k1 = min(k0 + step, K)
        prep.seg_k.append(k1 - k0)
        prep.segs.append(_seg_slice(prep.blob, G, k0, k1)
                         if not (k0 == 0 and k1 == K) else prep.blob)
        k0 = k1
    return prep


# ── the tile wire framing (apex_w4_ingest contract) ─────────────────────────

def gs_words(blob: WeightBlob) -> np.ndarray:
    """Scale phase beats: 4 fp16 LE per u64 word, ascending gid, zero tail.

    Slot i of beat b carries gid 4b+i at bits [16i +: 16] — byte-identical
    to an x86 memcpy of the fp16 array (the image byte-pipe rule)."""
    sc = np.asarray(blob.scales, dtype=np.uint64)
    nb = (len(sc) + GS_PER_BEAT - 1) // GS_PER_BEAT
    buf = np.zeros(nb * GS_PER_BEAT, dtype=np.uint64)
    buf[:len(sc)] = sc
    buf = buf.reshape(nb, GS_PER_BEAT)
    sh = np.uint64(16) * np.arange(GS_PER_BEAT, dtype=np.uint64)
    return np.sum(buf << sh, axis=1, dtype=np.uint64)


def pw_words(blob: WeightBlob) -> np.ndarray:
    """Packed phase beats: weight_codec S4/S5 beats as u64 words."""
    return beat_words(blob.packed)


def wire_words(blob: WeightBlob) -> np.ndarray:
    """The full per-job xw stream in tile wire order: gs || pw."""
    return np.concatenate([gs_words(blob), pw_words(blob)])


# ── job-level composition ───────────────────────────────────────────────────

def expected_w8(blob: WeightBlob, s8_bits) -> np.ndarray:
    """The [K,N] INT8 matrix the MXE loads for one W4-B job.

    wfeed_w4b_to_i8's flat emission stream inverse-framed by the S1-S3
    contract (beat n = p*N + c, lane r = B[8p+r][c]; rows >= K are the
    loader's zero mask and drop out)."""
    stream = wfeed_w4b_to_i8(blob, np.uint16(s8_bits))
    rows, cols = stream_index(blob.K, blob.N)
    live = rows < blob.K
    W = np.zeros((blob.K, blob.N), dtype=np.int64)
    W[rows[live], cols[live]] = stream[live]
    return W


def gemm_w4b(A8: np.ndarray, blob: WeightBlob, s8_bits,
             rq: tuple | None = None) -> np.ndarray:
    """One MXE job on W4-B weights: [M,K] int8 x expected_w8 -> [M,N].

    rq = None -> raw INT32 accumulators (requant_en=0); rq = (scale,
    shift) -> the C-2 epilogue (sign-extended INT8 values)."""
    A8 = np.asarray(A8, dtype=np.int64)
    acc = gemm_i8(A8, expected_w8(blob, s8_bits))
    if rq is None:
        return acc
    scale, shift = rq
    return requant_i32_to_i8(acc, int(scale), int(shift)).astype(np.int64)


# ── ingest accounting (the doubled-ingest headline, honestly stated) ────────

def wire_beats_per_job(K: int, N: int, G: int) -> tuple[int, int, int]:
    """(gs_beats, pw_beats, int8_baseline_beats) for one W4B job.

    The packed phase alone is exactly 2x denser than the INT8 stream
    (2 emitted beats per consumed beat, S6 odd-tail caveat aside); with
    G=32 scale overhead the whole-job wire ratio is 8/4.5 b/w ~= 1.78x.
    """
    return n_gs_beats(K, N, G), n_packed_beats(K, N), n_wgt_beats(K, N)
