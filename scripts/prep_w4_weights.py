#!/usr/bin/env python3
"""prep_w4_weights.py — pack weights for the tile's W4 INGEST LANE with the
FROZEN recipe (D-031: G=32 group scales, (B) chain, DIRECT host prep).

  python3 scripts/prep_w4_weights.py --in W.npy --out build/w4_pack \\
      [--s-w 1.0] [--g 32] [--k-split 2048] [--memh] [--check]

Input: one weight matrix .npy, float (the mlx-DEQUANTIZED source values —
snapped to the fp16 grid internally, which is where mlx dequant already
lives) or a .npz of named tensors. Orientation [K, N_total]: K is the GEMM
contraction (in_dim), columns are output channels — run_tinynpu.py's cache
orientation. The INT8-hop prep (quantizing to per-tensor INT8 first) is
DEPRECATED (-1.7 pt, B3_WEIGHT_PATH.md §5 row 6) and deliberately not
offered.

Per 8-column JOB TILE and per K_MAX job SEGMENT this emits the tile wire
stream (docs/design/W4_DATAPATH.md): the SCALE beats (4 fp16 LE per 64-bit
beat, ascending gid) followed by the PACKED beats (weight_codec S4/S5).
The little-endian u64 sequence is byte-identical to what an x86 memcpy of
the .bin lands on the DDR byte pipe — the "image IS the wire format" rule;
a fuel producer that stores scales after packed per job simply fetches the
scale record first (fetch order is the producer's choice, the wire order
is the tile lane's contract).

Outputs, per tensor:
  <name>.w4.bin       all jobs' wire words, job-major (LE u64 stream)
  <name>.w4.json      manifest: per job-tile/segment {col0, n, k0, k,
                      beats, gs_beats, pw_beats, word_off, word_cnt,
                      s8_f16, w4_job_word}, per stripe {s8_f16,
                      composite_f32 = f16_grade(s8 x s_w)} (§8.3), sha256
                      of the bin, and the ingest accounting (4-bit +
                      fp16/G b/w vs the INT8 stream)
  <name>.w4.memh      (--memh) one u64 hex per line, TB-consumable

--check replays every job's INT8 image through the golden feeder oracle
(wfeed_w4b_to_i8) and re-asserts the stripe reduction identity — slow on
big tensors, priceless on a new integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden import w4_lane as wl  # noqa: E402
from apex_golden.fp import f16_bits_to_f64  # noqa: E402
from apex_golden.weight_codec import (n_wgt_beats,  # noqa: E402
                                      wfeed_w4_to_i8, wfeed_w4b_to_i8)

MXE_N = 8


def pack_tensor(name: str, W: np.ndarray, out_dir: Path, *, G: int,
                s_w: float, k_split: int, memh: bool, check: bool) -> dict:
    K, N_total = W.shape
    words_all: list[int] = []
    jobs = []
    stripes = []

    for col0 in range(0, N_total, MXE_N):
        n = min(MXE_N, N_total - col0)
        prep = wl.prep_w4_direct(W[:, col0:col0 + n], G=G, s_w=s_w,
                                 k_split=k_split)
        stripes.append({"col0": col0, "n": n,
                        "s8_f16": int(prep.s8),
                        "composite_f32": float(prep.composite)})
        if check:
            ref, s8_ref = wfeed_w4_to_i8(prep.blob, "B")
            assert int(s8_ref) == int(prep.s8), f"{name}: s8 drift"
            assert np.array_equal(ref, wfeed_w4b_to_i8(prep.blob, prep.s8)), \
                f"{name}: reduction identity broke"
        k0 = 0
        for blob, kk in zip(prep.segs, prep.seg_k):
            ww = [int(w) for w in wl.wire_words(blob)]
            gsb, pwb, base = wl.wire_beats_per_job(kk, n, G)
            jobs.append({
                "col0": col0, "n": n, "k0": k0, "k": kk,
                "beats": base, "gs_beats": gsb, "pw_beats": pwb,
                "word_off": len(words_all), "word_cnt": len(ww),
                "s8_f16": int(prep.s8),
                # the ready-to-write W4_JOB register value
                "w4_job_word": (int(prep.s8) << 16) | (kk << 4) | n,
            })
            words_all.extend(ww)
            k0 += kk
    arr = np.asarray(words_all, dtype=np.uint64)
    bin_path = out_dir / f"{name}.w4.bin"
    arr.tofile(bin_path)
    if memh:
        with open(out_dir / f"{name}.w4.memh", "w") as f:
            for w in words_all:
                f.write(f"{w:016x}\n")

    int8_beats = sum(j["beats"] for j in jobs)
    wire_beats = sum(j["gs_beats"] + j["pw_beats"] for j in jobs)
    man = {
        "format": "w4-lane-1",
        "recipe": {"G": G, "chain": "B", "prep": "direct", "s_w": s_w,
                   "decision": "D-031"},
        "K": K, "N_total": N_total,
        "jobs": jobs, "stripes": stripes,
        "ingest": {
            "int8_beats": int8_beats, "wire_beats": wire_beats,
            "wire_ratio": round(int8_beats / wire_beats, 4),
            "bits_per_weight": round(64.0 * wire_beats /
                                     (n_wgt_beats(K, 1) * 8 * N_total), 4),
        },
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
    with open(out_dir / f"{name}.w4.json", "w") as f:
        json.dump(man, f, indent=1)
    print(f"{name}: K={K} N={N_total} jobs={len(jobs)} "
          f"words={len(words_all)} int8_beats={int8_beats} "
          f"wire_beats={wire_beats} ratio={man['ingest']['wire_ratio']}x "
          f"b/w={man['ingest']['bits_per_weight']}"
          + (" CHECK-OK" if check else ""))
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True,
                    help=".npy (one matrix) or .npz (named tensors), float")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--g", type=int, default=32,
                    help="scale group (D-031 ship freeze: 32)")
    ap.add_argument("--s-w", type=float, default=1.0,
                    help="source tensor scale for the epilogue composite")
    ap.add_argument("--k-split", type=int, default=2048,
                    help="job K bound (K_MAX; interior segments G-aligned)")
    ap.add_argument("--memh", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="golden-replay every stripe (slow, thorough)")
    args = ap.parse_args()

    if args.g != 32:
        print(f"NOTE: G={args.g} is a validated option, NOT the D-031 "
              f"ship freeze (32)", file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.inp)
    if src.suffix == ".npz":
        z = np.load(src)
        tensors = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    else:
        tensors = {src.stem: np.asarray(np.load(src), dtype=np.float64)}

    for name, W in tensors.items():
        if W.ndim != 2:
            print(f"skip {name}: ndim={W.ndim}", file=sys.stderr)
            continue
        pack_tensor(name, W, out_dir, G=args.g, s_w=args.s_w,
                    k_split=args.k_split, memh=args.memh, check=args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
