#!/usr/bin/env python3
# test_effective_bits.py — the honest KV-compression accounting gate.
#
# ARCHITECTURE.md §4 promises: "Effective bits/value after padding is
# recomputed and asserted in the golden model." This is that assertion
# (it did not exist before 2026-07-13 — the claim-detox audit created it).
#
# THREE accountings, every one printed and pinned:
#   RECORD  — what the shipped RTL actually stores per token record
#             (mirrors rtl/kvq/kvq_engine.sv "Derived geometry" localparams:
#             tag + fields + payload, padded to 64b). This is the number a
#             skeptic can compute from the SRAM; it is the only accounting
#             allowed in external claims about "realized in RTL".
#   CODEC   — the method-level cost measured from real cq_codec blobs
#             (payload + scales + outlier sidecar, per-channel scales
#             amortized over the G-token group). This is the ceiling the
#             record layout approaches post-A2/D-026 (landed 2026-07-14).
#   CSR     — what the hardware's INFO_CR registers advertise (amortized,
#             no tag/pad). Pinned here so the discrepancy is a tracked fact,
#             not a surprise.
#
# Any change to the record layout (e.g. A2) or the codec MUST update the
# pinned constants below — that is the point: numbers change consciously,
# with this gate as the receipt.

import sys
import numpy as np

sys.path.insert(0, sys.path[0] + "/..")
from apex_golden import cq_codec as cc  # noqa: E402

FP16_BITS = 16
TAG_BITS = 8
SCALE_WIDTH = 16
G = 128  # tile default KEY_GROUP (rtl/top/apex_top.sv KVQ_G)

fails = 0


def check(name, got, exp, tol=5e-4):
    global fails
    ok = abs(got - exp) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got:.4f} (pinned {exp:.4f})")
    if not ok:
        fails += 1


def record_row_bits(tier, D, k):
    """§4/D-026 unified SRAM row width — imported from the packer module
    (cc.sram_row_bits), never re-derived: pad64(max(key_raw, val_raw)) on
    keyed engines. Post-A2 the key record is [tag {ssid,1}][k fp16 lanes]
    [D int4 codes][pad]; group scales live in the persistent scale bank."""
    bpv = 8 if tier == "KVQ8" else 4
    return cc.sram_row_bits(D, bpv, k, keyed=(tier != "KVQ8"))


def stored_bv(tier, D, k):
    """Whole-KV bits/value as stored: one row per K record + one per V record
    (both SRAM_WIDTH wide) per token, plus the scale-bank row amortized over
    the group (D*16/G bits per token, keyed tiers only)."""
    row = record_row_bits(tier, D, k)
    bank = (D * FP16_BITS / G) if tier != "KVQ8" else 0.0
    return (2 * row + bank) / (2 * D)


def csr_advertised(tier, D):
    """Mirror of kvq_engine.sv INFO_CR fixed-point formulas (x256, floor)."""
    bpv = 8 if tier == "KVQ8" else 4
    val_den = D * bpv + SCALE_WIDTH
    key_den = val_den if tier == "KVQ8" else (D * 4 + (SCALE_WIDTH * D) // G)
    cr_v = (D * FP16_BITS * 256 // val_den) / 256
    cr_k = (D * FP16_BITS * 256 // key_den) / 256
    return cr_v, cr_k


def codec_bits_per_value(tier, D, T=256):
    """Measured from real golden blobs: payload + scales + sidecar, K and V."""
    rng = np.random.default_rng(20260713)
    K = rng.standard_normal((T, D)).astype(np.float16).view(np.uint16)
    V = rng.standard_normal((T, D)).astype(np.float16).view(np.uint16)
    bits = 8 if tier == "KVQ8" else 4
    k_out = 2 if tier == "KVQ4+" else 0
    vb = cc.compress_values(V, bits)
    v_bits = vb.payload.size * 8 + vb.scales.size * FP16_BITS
    if tier == "KVQ8":
        kb2 = cc.compress_values(K, bits)  # KVQ8 keys are per-token, value path
        k_bits = kb2.payload.size * 8 + kb2.scales.size * FP16_BITS
    else:
        kb = cc.compress_keys(K, bits, G, np.arange(k_out))
        k_bits = kb.payload.size * 8 + kb.scales.size * FP16_BITS \
            + kb.sidecar.size * FP16_BITS
    return (k_bits + v_bits) / (2 * T * D)


print("EFFECTIVE BITS/VALUE — honest accounting gate (FP16 baseline = 16 b/v)")
print(f"  record geometry: tag={TAG_BITS}b, scale={SCALE_WIDTH}b, pad-to-64b, G={G}")

# ── RECORD accounting (shipped RTL, post-A2/D-026) — pinned expectations ────
# Unified-slot convention (the engine's SRAM truth: every row is SRAM_WIDTH
# wide) + scale bank amortized over G. Pre-A2 (RETIRED 2026-07-14): key rows
# 1344/2624 b, whole-KV 13.0/12.5 b/v = 1.2308/1.2800x — keys EXPANDED.
#              tier      D    k   row    K+V b/v   ratio
RECORD_PIN = [("KVQ4",   64,  0,  320,   5.0625,   3.1605),
              ("KVQ4",  128,  0,  576,   4.5625,   3.5068),
              ("KVQ4+",  64,  2,  320,   5.0625,   3.1605),   # shipped k=2
              ("KVQ4+", 128,  2,  576,   4.5625,   3.5068),
              ("KVQ4+",  64,  5,  384,   6.0625,   2.6392),   # k=5 headroom
              ("KVQ4+", 128,  5,  640,   5.0625,   3.1605),
              ("KVQ8",   64,  0,  576,   9.0000,   1.7778),
              ("KVQ8",  128,  0, 1088,   8.5000,   1.8824)]

print("\nRECORD (as stored by rtl/kvq/kvq_engine.sv, A2/D-026 layout + scale bank):")
for tier, D, k, exp_row, exp_bv, exp_r in RECORD_PIN:
    row = record_row_bits(tier, D, k)
    bv = stored_bv(tier, D, k)
    if row != exp_row:
        print(f"  FAIL  {tier} D={D} k={k}: row width {row} != pinned {exp_row}")
        fails += 1
    check(f"{tier:5s} D={D:3d} k={k} K+V stored", bv, exp_bv)
    check(f"{tier:5s} D={D:3d} k={k} stored ratio", FP16_BITS / bv, exp_r)

# ── CODEC accounting (method ceiling) — pinned expectations ─────────────────
CODEC_PIN = [("KVQ4",   64, 4.1875), ("KVQ4",  128, 4.1250),
             ("KVQ4+",  64, 4.3730), ("KVQ4+", 128, 4.2178),
             ("KVQ8",   64, 8.2500), ("KVQ8",  128, 8.1250)]

print("\nCODEC (measured from golden blobs; scales amortized over G — the A2 target):")
for tier, D, exp in CODEC_PIN:
    bv = codec_bits_per_value(tier, D)
    check(f"{tier:5s} D={D:3d} codec b/v (ratio {FP16_BITS/bv:.3f}x)", bv, exp)

# ── CSR advertisement — pinned so the discrepancy is tracked ────────────────
CSR_PIN = [("KVQ4", 64, 3.7617, 3.8750), ("KVQ4", 128, 3.8750, 3.8750)]
print("\nCSR INFO_CR (hardware-advertised; amortized, NO tag/pad — do not quote externally):")
for tier, D, exp_v, exp_k in CSR_PIN:
    cr_v, cr_k = csr_advertised(tier, D)
    check(f"{tier:5s} D={D:3d} CR_V", cr_v, exp_v)
    check(f"{tier:5s} D={D:3d} CR_K", cr_k, exp_k)

print()
if fails:
    print(f"EFFECTIVE-BITS GATE: {fails} FAILURES")
    sys.exit(1)
print("EFFECTIVE-BITS GATE: ALL PINNED ACCOUNTINGS MATCH")
print("  -> headline-safe (A2/D-026, shipped k<=2): whole-KV STORED 3.16x (D=64) / 3.51x (D=128),")
print("     all overheads counted (tag, lanes, pad, scale bank). k=5 point: 2.64x (D=64).")
print("  -> method ceiling (codec): 4.13-4.37 b/v = 3.66-3.88x. A ratio >4x is IMPOSSIBLE here.")
print("  -> record-vs-codec gap = tag + pad + unified-slot rounding; CSR INFO_CR still amortized,")
print("     never quote CR_* externally.")
