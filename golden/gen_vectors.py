#!/usr/bin/env python
"""gen_vectors.py — generate APEX's committed, self-contained golden vector set
FROM golden/apex_golden/cq_codec.py (our own bit-exact codec model).

WHY THIS EXISTS
---------------
The KVQ golden gates (golden/tests/test_contract.py, test_attention.py) and the
RTL suites (verif/kvq/smoke, verif/top/l2, verif/top/l3) all need frozen fp16
K/V stimulus plus the codec's expected outputs. Historically those vectors were
read from a third-party checkout that is NOT shipped with this repo, so a fresh
clone could not build or verify. This script re-derives an equivalent set from
OUR model and commits it under golden/vectors/, making the repo standalone.

The set is deterministic (fixed seeds) and outlier-bearing: each key tensor
plants two dominant channels so the per-channel INT4 codec's tier behavior is
exercised end to end — CQ-4 (no outlier lane) blows the D-023 5% budget on the
planted channels, while CQ-4+ (top-2 masked) and the TIP per-block mix recover.
This is the property test_attention.py's D-021/D-022/D-023 gates assert.

WHAT IT WRITES (under golden/vectors/)
--------------------------------------
  <cfg>/                          one dir per (D,T,G,tier) contract config
    input_k.f16.hex input_v.f16.hex   fp16 stimulus (shared across the tiers
                                      of one D,T,G base)
    outlier_mask.u8.hex               D bytes; bit0=1 marks an outlier channel
                                      (CQ-4+ only; all-zero otherwise)
    val_scales.f16.hex val_payload.u8.hex expected_v_hat.f32.hex   value lane
    key_scales.f16.hex key_payload.u8.hex expected_k_hat.f32.hex   key lane
    sidecar.f16.hex                   CQ-4/CQ-4+ outlier fp16 sidecar
  <cfg>.npz  (the three CQ-4 calib configs only)   input_K/input_V float16
             tensors, consumed by test_attention / L2 / L3 as calibration data

ROLE (honest): this is a SELF-CONSISTENCY + RTL-vs-model vector set. The model
is the arbiter (golden/tests/test_contract.py drives stimulus through it and
checks round-trip + packing invariants); the RTL is then checked bit-exact
against these committed vectors. Any historical "validate our model against an
external oracle" role is retired — that oracle is not part of this repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "golden"))
from apex_golden import cq_codec as cq  # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402

OUT = REPO / "golden" / "vectors"

# ── contract configs (name → D, T, G, tier)  tier: 0=CQ-8 1=CQ-4 2=CQ-4+ ─────
# The nine test_contract configs, plus d64_T128_G128__CQ4 (the same d64_T128
# stimulus re-grouped at G=128 — the smoke suite's g128_cq4 key golden).
CONFIGS = {
    "d64_T128_G64__CQ8":       (64, 128, 64, 0),
    "d64_T128_G64__CQ4":       (64, 128, 64, 1),
    "d64_T128_G64__CQ4plus":   (64, 128, 64, 2),
    "d64_T70_G64__CQ8":        (64, 70, 64, 0),
    "d64_T70_G64__CQ4":        (64, 70, 64, 1),
    "d64_T70_G64__CQ4plus":    (64, 70, 64, 2),
    "d128_T100_G128__CQ8":     (128, 100, 128, 0),
    "d128_T100_G128__CQ4":     (128, 100, 128, 1),
    "d128_T100_G128__CQ4plus": (128, 100, 128, 2),
    "d64_T128_G128__CQ4":      (64, 128, 128, 1),
}

# npz calibration tensors emitted for the attention / L2 / L3 gates (CQ-4 name)
NPZ_CONFIGS = ("d64_T128_G64__CQ4", "d64_T70_G64__CQ4", "d128_T100_G128__CQ4")

# planted outlier channels per D (also the CQ-4+ mask / the top-2 amax channels)
OUTLIER_CH = {64: (5, 50), 128: (5, 100)}

# stimulus recipe constants (tuned so CQ-4 exceeds the D-023 5% value-scale
# budget on the outlier channels while CQ-4+/CQ-8 and the TIP mix stay inside)
SEED_BASE = 0x_A9E_2026
K_STD = 1.0
V_STD = 1.0
OUTLIER_GAIN = 16.0
K_FS = 8.0     # full-scale target for K after planting (fp16 headroom)
V_FS = 3.0     # value full-scale target (keeps per-token INT4 V error modest)


def _base_key(D: int, T: int) -> int:
    return SEED_BASE ^ (D * 1_000_003) ^ (T * 9_176)


def gen_inputs(D: int, T: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Deterministic fp16 K (2 planted outlier channels) + V, per (D,T) base.

    Shared across the CQ-8/CQ-4/CQ-4+ configs of one D,T,G so the .npz K that
    test_attention masks is the same tensor the hex dirs quantize.
    """
    rng = np.random.default_rng(_base_key(D, T))
    oi = OUTLIER_CH[D]
    K = rng.normal(0.0, K_STD, (T, D))
    for c in oi:
        K[:, c] = rng.normal(0.0, K_STD * OUTLIER_GAIN, T)
    K *= K_FS / np.max(np.abs(K))              # fit fp16 range, keep outliers hot
    V = rng.normal(0.0, V_STD, (T, D))
    V *= V_FS / np.max(np.abs(V))
    K16 = f64_to_f16_bits(K).reshape(T, D)
    V16 = f64_to_f16_bits(V).reshape(T, D)
    # guarantee the planted channels are exactly the top-2 amax (top2_outliers)
    amax = np.max(np.abs(f16_bits_to_f64(K16).reshape(T, D)), axis=0)
    top2 = tuple(int(c) for c in np.sort(np.argsort(amax)[-2:]))
    assert top2 == tuple(sorted(oi)), (D, T, top2, oi)
    return K16, V16, oi


def write_hex(path: Path, arr, width: int) -> None:
    path.write_text("".join(f"{int(v):0{width}x}\n" for v in np.asarray(arr).reshape(-1)))


def emit_config(name: str, D: int, T: int, G: int, tier: int) -> dict:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    K16, V16, oi = gen_inputs(D, T)

    mask = np.zeros(D, dtype=np.uint8)
    if tier == 2:
        for c in oi:
            mask[c] = 1
    outlier_idx = np.flatnonzero(mask) if tier == 2 else np.array([], dtype=np.int64)

    write_hex(out / "input_k.f16.hex", K16, 4)
    write_hex(out / "input_v.f16.hex", V16, 4)
    write_hex(out / "outlier_mask.u8.hex", mask, 2)

    # value lane: per-token INT8 (CQ-8) / INT4 (CQ-4/+)
    vbits = 8 if tier == 0 else 4
    vb = cq.compress_values(V16, vbits)
    write_hex(out / "val_scales.f16.hex", vb.scales, 4)
    write_hex(out / "val_payload.u8.hex", vb.payload, 2)
    write_hex(out / "expected_v_hat.f32.hex", cq.decompress_values(vb).reshape(-1), 8)

    # key lane: CQ-8 keys ride the value path; CQ-4/+ use per-channel grouping
    if tier == 0:
        kb = cq.compress_values(K16, 8)
        write_hex(out / "key_scales.f16.hex", kb.scales, 4)
        write_hex(out / "key_payload.u8.hex", kb.payload, 2)
        write_hex(out / "expected_k_hat.f32.hex", cq.decompress_values(kb).reshape(-1), 8)
        khat = cq.decompress_values(kb)
    else:
        kb = cq.compress_keys(K16, 4, G, outlier_idx)
        write_hex(out / "key_scales.f16.hex", kb.scales, 4)
        write_hex(out / "key_payload.u8.hex", kb.payload, 2)
        write_hex(out / "sidecar.f16.hex", kb.sidecar.reshape(-1), 4)
        khat = cq.decompress_keys(kb)
        write_hex(out / "expected_k_hat.f32.hex", khat.reshape(-1), 8)

    # §4/D-026 record + scale-bank images (the arbiter's SRAM truth): key
    # records carry {ssid,1} tags + outlier lanes + codes; group scales live
    # in the persistent bank rows. CQ-8 keys ride the value-record format.
    k_out = int(mask.sum())
    row = cq.sram_row_bits(D, vbits, k_out, keyed=(tier != 0))
    write_hex(out / "val_records.u8.hex", cq.pack_value_records(vb, row), 2)
    if tier == 0:
        write_hex(out / "key_records.u8.hex", cq.pack_value_records(kb, row), 2)
    else:
        krecs, bank, _ = cq.pack_key_records(kb, row)
        write_hex(out / "key_records.u8.hex", krecs, 2)
        write_hex(out / "scale_bank.f16.hex", bank, 4)

    # commit the fp16 calibration tensors for the CQ-4 configs the attention /
    # L2 / L3 gates load as .npz (float16 arrays; consumers .view(uint16))
    if name in NPZ_CONFIGS:
        np.savez(OUT / f"{name}.npz",
                 input_K=f16_bits_to_f64(K16).reshape(T, D).astype(np.float16),
                 input_V=f16_bits_to_f64(V16).reshape(T, D).astype(np.float16))

    return {"name": name, "D": D, "T": T, "G": G, "tier": tier,
            "groups": len(kb.groups) if tier != 0 else T,
            "scales": int(kb.scales.size), "payload": int(kb.payload.size),
            "khat": int(np.asarray(khat).size), "outliers": int(mask.sum())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"gen_vectors: writing committed golden vectors -> {OUT}")
    for name, (D, T, G, tier) in CONFIGS.items():
        info = emit_config(name, D, T, G, tier)
        print(f"  {name:26s} D={info['D']:3d} T={info['T']:3d} G={info['G']:3d} "
              f"tier={info['tier']} groups={info['groups']:3d} "
              f"scales={info['scales']:5d} payload={info['payload']:6d}B "
              f"outliers={info['outliers']}")
    print(f"gen_vectors: {len(CONFIGS)} configs + {len(NPZ_CONFIGS)} npz "
          f"calibration tensors written")
    print("GEN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
