"""test_contract.py — validate apex_golden.cq_codec BIT-EXACTLY against
the committed golden-vector set (the arbiter, per ARCHITECTURE.md D-001/D-013).

The vectors live in golden/vectors/, generated from THIS model by
golden/gen_vectors.py; this is a self-consistency check (the model is the
arbiter and the RTL is then checked bit-exact against the same vectors). For
each vector dir: recompress the fp16 inputs and demand byte-identical payloads,
scale banks, and sidecar; then decompress and demand bit-identical fp32 x_hat
words. Any mismatch prints the first offending index and fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex_golden import cq_codec as cq                      # noqa: E402
from apex_golden.fp import load_hex                             # noqa: E402

VEC = Path(__file__).resolve().parents[1] / "vectors"

# name → (D, T, G, tier)   [G applies to keys; tier: 0=CQ-8, 1=CQ-4, 2=CQ-4+]
CONFIGS = {
    "d64_T128_G64__CQ8":      (64, 128, 64, 0),
    "d64_T128_G64__CQ4":      (64, 128, 64, 1),
    "d64_T128_G64__CQ4plus":  (64, 128, 64, 2),
    "d64_T70_G64__CQ8":       (64, 70, 64, 0),
    "d64_T70_G64__CQ4":       (64, 70, 64, 1),
    "d64_T70_G64__CQ4plus":   (64, 70, 64, 2),
    "d128_T100_G128__CQ8":    (128, 100, 128, 0),
    "d128_T100_G128__CQ4":    (128, 100, 128, 1),
    "d128_T100_G128__CQ4plus": (128, 100, 128, 2),
}


def check(name: str, D: int, T: int, G: int, tier: int) -> list[str]:
    d = VEC / name
    errs: list[str] = []

    def expect(label, got, want):
        got, want = np.asarray(got).reshape(-1), np.asarray(want).reshape(-1)
        if got.shape != want.shape:
            errs.append(f"{label}: shape {got.shape} != {want.shape}")
            return
        bad = np.flatnonzero(got != want)
        if len(bad):
            i = bad[0]
            errs.append(f"{label}: {len(bad)} mismatches; first @[{i}] "
                        f"got={got[i]:#x} want={want[i]:#x}")

    K_in = load_hex(d / "input_k.f16.hex", "f16").reshape(T, D)
    V_in = load_hex(d / "input_v.f16.hex", "f16").reshape(T, D)
    mask = load_hex(d / "outlier_mask.u8.hex", "u8")
    outlier_idx = np.flatnonzero(mask) if tier == 2 else np.array([], dtype=np.int64)

    # values: per-token INT8 (CQ-8) / INT4 (CQ-4/+)
    vbits = 8 if tier == 0 else 4
    vb = cq.compress_values(V_in, vbits)
    expect("val_payload", vb.payload, load_hex(d / "val_payload.u8.hex", "u8"))
    expect("val_scales", vb.scales, load_hex(d / "val_scales.f16.hex", "f16"))
    expect("v_hat_f32", cq.decompress_values(vb),
           load_hex(d / "expected_v_hat.f32.hex", "f32"))

    # keys: CQ-8 → per-token INT8 (value path); CQ-4/+ → per-channel grouped
    if tier == 0:
        kb = cq.compress_values(K_in, 8)
        expect("key_payload", kb.payload, load_hex(d / "key_payload.u8.hex", "u8"))
        expect("key_scales", kb.scales, load_hex(d / "key_scales.f16.hex", "f16"))
        expect("k_hat_f32", cq.decompress_values(kb),
               load_hex(d / "expected_k_hat.f32.hex", "f32"))
    else:
        kb = cq.compress_keys(K_in, 4, G, outlier_idx)
        expect("key_payload", kb.payload, load_hex(d / "key_payload.u8.hex", "u8"))
        expect("key_scales", kb.scales, load_hex(d / "key_scales.f16.hex", "f16"))
        sidecar_want = load_hex(d / "sidecar.f16.hex", "f16")
        expect("sidecar", kb.sidecar, sidecar_want)
        expect("k_hat_f32", cq.decompress_keys(kb),
               load_hex(d / "expected_k_hat.f32.hex", "f32"))

    # §4/D-026 record + scale-bank images (self-consistency vs the packers)
    k_out = int(mask.sum())
    row = cq.sram_row_bits(D, vbits, k_out, keyed=(tier != 0))
    expect("val_records", cq.pack_value_records(vb, row),
           load_hex(d / "val_records.u8.hex", "u8").reshape(T, row // 8))
    if tier == 0:
        expect("key_records", cq.pack_value_records(kb, row),
               load_hex(d / "key_records.u8.hex", "u8").reshape(T, row // 8))
    else:
        krecs, bank, _ = cq.pack_key_records(kb, row)
        expect("key_records", krecs,
               load_hex(d / "key_records.u8.hex", "u8").reshape(T, row // 8))
        expect("scale_bank", bank,
               load_hex(d / "scale_bank.f16.hex", "f16").reshape(-1, D))

    return errs


def main() -> int:
    total_err = 0
    for name, (D, T, G, tier) in CONFIGS.items():
        errs = check(name, D, T, G, tier)
        tag = "PASS" if not errs else "FAIL"
        print(f"{name:26s} D={D:3d} T={T:3d} G={G:3d} tier={tier} : {tag}")
        for e in errs:
            print(f"    ✗ {e}")
        total_err += len(errs)
    print("=" * 64)
    if total_err == 0:
        print("APEX GOLDEN vs FROZEN VECTORS: ALL 9 CONFIGS BIT-EXACT")
        return 0
    print(f"APEX GOLDEN: {total_err} mismatching fields — NON-CONFORMANT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
