"""cq_codec.py — APEX's bit-exact per-channel INT4 KV-cache codec model.

The reference (arbiter) for the KVQ hardware: a per-channel INT4 key /
per-token INT4 value quantizer with an FP16 outlier lane, in the published
KV-cache-quantization style (cf. KIVI, KVQuant). Implements the APEX numerics
contract — ARCHITECTURE.md §2 (C-1..C-5): symmetric signed INT4, EPS=2^-14,
clamp [-8,7]/[-128,127], round-half-to-even, FP16 scales, fp32 dequant bus.

Conformance is established by golden/tests/test_contract.py, which drives
stimulus through this model and checks round-trip + packing invariants; the
KVQ RTL is then verified bit-exact against this model. Validated against an
independent fixed-point reference during development.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fp import f16_bits_to_f64, f64_to_f16_bits, f64_to_f32_bits, rne

EPS = 2.0 ** -14  # contract §1, FINAL


def qmax_of(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def qmin_of(bits: int) -> int:
    return -(1 << (bits - 1))


# ── §1 primitives (mirror cq_units.sv via an independent fixed-point reference.hpp) ──────────────

def amax_bits(x_f16: np.ndarray) -> np.uint16:
    """Max-abs over fp16 bit patterns → winner's bits, sign cleared.

    fp16 magnitude compare == unsigned compare on sign-cleared bits (the
    cq_value_path/cq_key_path MAG_MASK trick), so the max of the cleared bit patterns IS the answer.
    """
    mags = np.asarray(x_f16, dtype=np.uint16) & np.uint16(0x7FFF)
    return np.uint16(mags.max(initial=np.uint16(0)))


def scale_from_amax(amax_f16: np.uint16, bits: int) -> np.uint16:
    """s = max(amax/qmax, EPS), RNE-cast to fp16 bits (cq_scale_unit)."""
    s = float(f16_bits_to_f64(np.array([amax_f16]))[0]) / qmax_of(bits)
    if s < EPS:
        s = EPS
    return np.uint16(f64_to_f16_bits(np.array([s]))[0])


def quant_codes(x_f16: np.ndarray, scale_f16: np.uint16, bits: int) -> np.ndarray:
    """q = clamp(rne(x/s), qmin, qmax) → int64 codes (cq_quant_unit)."""
    x = f16_bits_to_f64(x_f16)
    s = float(f16_bits_to_f64(np.array([scale_f16]))[0])
    return np.clip(rne(x / s), qmin_of(bits), qmax_of(bits))


def dequant_f32(codes: np.ndarray, scale_f16: np.uint16) -> np.ndarray:
    """x_hat = code * s → fp32 bits (cq_dequant_unit; fp32 read bus, D-010)."""
    s = float(f16_bits_to_f64(np.array([scale_f16]))[0])
    return f64_to_f32_bits(np.asarray(codes, dtype=np.float64) * s)


# ── §5 packing ────────────────────────────────────────────────────────────────

def pack_int4(codes: np.ndarray) -> np.ndarray:
    """Two's-complement nibbles: element 2i → low nibble, 2i+1 → high."""
    c = np.asarray(codes, dtype=np.int64) & 0x0F
    if len(c) & 1:
        c = np.concatenate([c, [0]])
    return (c[0::2] | (c[1::2] << 4)).astype(np.uint8)


def unpack_int4(payload: np.ndarray, count: int) -> np.ndarray:
    b = np.asarray(payload, dtype=np.uint8)
    nib = np.empty(len(b) * 2, dtype=np.int64)
    nib[0::2] = b & 0x0F
    nib[1::2] = (b >> 4) & 0x0F
    nib = nib[:count]
    return np.where(nib >= 8, nib - 16, nib)


def pack_int8(codes: np.ndarray) -> np.ndarray:
    return (np.asarray(codes, dtype=np.int64) & 0xFF).astype(np.uint8)


def unpack_int8(payload: np.ndarray) -> np.ndarray:
    b = np.asarray(payload, dtype=np.int64)
    return np.where(b >= 128, b - 256, b)


# ── §2 values: per-token scale over D dims (streaming, no buffer) ────────────

@dataclass
class ValueBlob:
    T: int
    D: int
    bits: int
    scales: np.ndarray = field(default=None)   # [T] fp16 bits
    codes: np.ndarray = field(default=None)    # [T, D] int64
    payload: np.ndarray = field(default=None)  # packed uint8 stream


def compress_values(V_f16: np.ndarray, bits: int) -> ValueBlob:
    """V_f16: [T, D] fp16 bit patterns."""
    T, D = V_f16.shape
    b = ValueBlob(T=T, D=D, bits=bits)
    b.scales = np.empty(T, dtype=np.uint16)
    b.codes = np.empty((T, D), dtype=np.int64)
    for t in range(T):
        s = scale_from_amax(amax_bits(V_f16[t]), bits)
        b.scales[t] = s
        b.codes[t] = quant_codes(V_f16[t], s, bits)
    flat = b.codes.reshape(-1)
    b.payload = pack_int4(flat) if bits == 4 else pack_int8(flat)
    return b


def decompress_values(b: ValueBlob) -> np.ndarray:
    """→ [T, D] fp32 bit patterns."""
    out = np.empty((b.T, b.D), dtype=np.uint32)
    for t in range(b.T):
        out[t] = dequant_f32(b.codes[t], np.uint16(b.scales[t]))
    return out


# ── §3/§4 keys: per-channel grouped scale + optional FP16 outlier lane ────────

@dataclass
class KeyBlob:
    T: int
    D: int
    bits: int
    G: int
    groups: list = field(default_factory=list)      # [(a, b), ...]
    keep: np.ndarray = field(default=None)          # non-outlier channels
    outlier: np.ndarray = field(default=None)       # sorted outlier channels
    scales: np.ndarray = field(default=None)        # concat per-group [nk] fp16
    payload: np.ndarray = field(default=None)       # concat per-group int4 bytes
    sidecar: np.ndarray = field(default=None)       # [T, k] fp16 bits, t-major


def _group_bounds(T: int, G: int) -> list[tuple[int, int]]:
    if G <= 0:
        return [(0, T)]
    return [(s, min(s + G, T)) for s in range(0, T, G)]


def compress_keys(K_f16: np.ndarray, bits: int, G: int,
                  outlier_idx: np.ndarray | list) -> KeyBlob:
    """K_f16: [T, D] fp16 bits. Partial final group (g<G) handled per §3.1."""
    T, D = K_f16.shape
    b = KeyBlob(T=T, D=D, bits=bits, G=G)
    b.outlier = np.sort(np.asarray(outlier_idx, dtype=np.int64))
    is_out = np.zeros(D, dtype=bool)
    is_out[b.outlier] = True
    b.keep = np.flatnonzero(~is_out)
    nk = len(b.keep)

    b.groups = _group_bounds(T, G)
    scales, payload = [], []
    for a, e in b.groups:
        g = e - a
        codes = np.empty((g, nk), dtype=np.int64)
        for ci, cc in enumerate(b.keep):
            col = K_f16[a:e, cc]                       # g tokens of channel cc
            s = scale_from_amax(amax_bits(col), bits)  # amax over g, not G (§3.1)
            scales.append(s)
            codes[:, ci] = quant_codes(col, s, bits)
        payload.append(pack_int4(codes.reshape(-1)))   # per-group byte-aligned
    b.scales = np.asarray(scales, dtype=np.uint16)
    b.payload = (np.concatenate(payload) if payload
                 else np.empty(0, dtype=np.uint8))
    b.sidecar = K_f16[:, b.outlier].astype(np.uint16)  # identity FP16, t-major
    return b


def decompress_keys(b: KeyBlob) -> np.ndarray:
    """→ [T, D] fp32 bit patterns (outliers replay FP16 widened to fp32)."""
    out = np.zeros((b.T, b.D), dtype=np.uint32)
    nk = len(b.keep)
    sc_base = 0
    byte_base = 0
    for a, e in b.groups:
        g = e - a
        gn = g * nk
        nbytes = (gn + 1) // 2
        codes = unpack_int4(b.payload[byte_base:byte_base + nbytes], gn)
        codes = codes.reshape(g, nk)
        byte_base += nbytes
        for ci, cc in enumerate(b.keep):
            out[a:e, cc] = dequant_f32(codes[:, ci],
                                       np.uint16(b.scales[sc_base + ci]))
        sc_base += nk
    if len(b.outlier):
        widened = f64_to_f32_bits(f16_bits_to_f64(b.sidecar.reshape(-1)))
        out[:, b.outlier] = widened.reshape(b.T, len(b.outlier))
    return out


# ── §4/D-026 record packing — A2 KV-REC-DEDUP (scales in a persistent bank) ──
#
# D-026 supersedes D-009's KEY record COMPOSITION only; the byte-aligned
# LSB-first, pad-to-64b rule stays normative for records AND bank rows.
#
#   KEY record (per token, CQ-4/CQ-4+ tiers), one padded SRAM row:
#     [ tag: 8b = {ssid[6:0], 1'b1} ]                       bits [7:0]
#     [ outlier lanes: OUTLIER_K x fp16 ]                   bits [8 +: k*16]
#         lane j = raw fp16 of the j-th mask-set channel (ascending channel
#         order == KeyBlob.sidecar column order)
#     [ int4 codes: D x 4b ]                                bits [8+k*16 +: D*4]
#         keep channel c -> quant code nibble; outlier c -> sentinel 4'd1
#     [ pad to 64b multiple ]
#   VALUE record: unchanged  [tag 8'h00][fp16 scale][D x BPV codes][pad64].
#   SCALE BANK row (per committed group): D x 16b, keep channel c -> the
#     group's per-channel scale, outlier channels forced 16'h0000.
#   ssid = group emission sequence (ssid0 + gi) mod SCALE_SETS; the engine's
#     allocator is commit-time and wrap-with-fault (SB_OVWR, IRQ_STATUS[1]).

SCALE_SETS = 4   # persistent scale-bank sets per engine (D-026 default)


def pad64(bits: int) -> int:
    return 64 * ((bits + 63) // 64)


def key_rec_raw_bits(D: int, k: int) -> int:
    return 8 + 16 * k + 4 * D


def val_rec_raw_bits(D: int, bits: int) -> int:
    return 8 + 16 + bits * D


def sram_row_bits(D: int, bits: int, k: int, keyed: bool) -> int:
    """Unified row width: pad64(max(key_raw, val_raw)) on keyed (CQ-4/CQ-4+)
    engines; value-only engines (CQ-8, KEYG=0) pad the value record alone.
    Mirrors kvq_engine.sv REC_RAW/SRAM_WIDTH — import this, never re-derive."""
    raw = (max(key_rec_raw_bits(D, k), val_rec_raw_bits(D, bits))
           if keyed else val_rec_raw_bits(D, bits))
    return pad64(raw)


def _bits_to_row_bytes(val: int, row_bits: int) -> np.ndarray:
    return np.frombuffer(val.to_bytes(row_bits // 8, "little"), dtype=np.uint8)


def pack_value_records(b: ValueBlob, row_bits: int) -> np.ndarray:
    """[T, row_bytes] uint8 — the §4 value-record image per token."""
    nib = 2 if b.bits == 4 else 1                    # payload bytes per elem⁻¹
    per_tok = (b.D * b.bits) // 8
    out = np.empty((b.T, row_bits // 8), dtype=np.uint8)
    for t in range(b.T):
        v = 0                                        # tag 8'h00
        v |= int(b.scales[t]) << 8
        pay = b.payload[t * per_tok:(t + 1) * per_tok]
        for i, byte in enumerate(pay):
            v |= int(byte) << (24 + 8 * i)
        out[t] = _bits_to_row_bytes(v, row_bits)
    return out


def pack_key_records(b: KeyBlob, row_bits: int, ssid0: int = 0,
                     scale_sets: int = SCALE_SETS):
    """→ (records uint8 [T, row_bytes], bank uint16 [n_groups, D], ssids).
    The arbiter image of D-026 key records + persistent scale-bank rows."""
    k = len(b.outlier)
    nk = len(b.keep)
    lanes_lo = 8
    codes_lo = 8 + 16 * k
    recs = np.empty((b.T, row_bits // 8), dtype=np.uint8)
    bank = np.zeros((len(b.groups), b.D), dtype=np.uint16)
    ssids = []
    sc_base = 0
    byte_base = 0
    for gi, (a, e) in enumerate(b.groups):
        g = e - a
        gn = g * nk
        nbytes = (gn + 1) // 2
        codes = unpack_int4(b.payload[byte_base:byte_base + nbytes], gn)
        codes = codes.reshape(g, nk)
        byte_base += nbytes
        ssid = (ssid0 + gi) % scale_sets
        ssids.append(ssid)
        for ci, cc in enumerate(b.keep):
            bank[gi, cc] = b.scales[sc_base + ci]    # outlier lanes stay 0
        for t in range(a, e):
            v = (ssid << 1) | 1                      # tag {ssid[6:0], 1'b1}
            for j in range(k):                       # lanes, sidecar order
                v |= int(b.sidecar[t, j]) << (lanes_lo + 16 * j)
            ki = 0
            for cc in range(b.D):
                if cc in set(b.outlier):
                    nibble = 0x1                     # sentinel 4'd1
                else:
                    nibble = int(codes[t - a, ki]) & 0xF
                    ki += 1
                v |= nibble << (codes_lo + 4 * cc)
            recs[t] = _bits_to_row_bytes(v, row_bits)
        sc_base += nk
    return recs, bank, ssids
