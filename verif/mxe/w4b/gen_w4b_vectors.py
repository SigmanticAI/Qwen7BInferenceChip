#!/usr/bin/env python3
"""gen_w4b_vectors.py — stimulus + bit-exact expected results for the W4-B
feeder TB (D-031), from the golden arbiter apex_golden.weight_codec
(wfeed_w4b_to_i8 — the given-scale (B) chain; contract docs/design/
W4B_FEEDER.md; framing S1-S6 shared with the (A) suite).

The TB only drives, collects and compares — every expected byte here is
produced by the golden model, never by the RTL and never by hand.

RECORD FORMAT (one record per job; '#' lines are comments)

  F <beats> <w4b_en> <K> <N> <s8hex> <odd> <cmin> <cmax> <czero> <cneg> <satr>
      S <hex16>  x N*ceil(K/G) if w4b_en else 0        (gs sideband, gid asc)
      P <hex64>  x ceil(beats/2) if w4b_en else beats  (packed-W4 input)
      E <hex64>  x beats                               (expected INT8 out)
  I <beats> <K> <N>                                    (illegal job)
  R <beats> <w4b_en> <K> <N> <s8hex> <abortcyc> <tphase>
      S/P as for F                                     (mid-op reset job)

  beats = EMITTED INT8 beats = ceil(K/8)*N. satr = a clamp rail (+127 or
  -128) present in the expected output (small-s8 saturation coverage).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
from apex_golden import weight_codec as wc                     # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits   # noqa: E402

LANES = wc.LANES
G = int(__import__("os").environ.get("W4B_G", "32"))          # ship 32


def words_from_i8(elems: np.ndarray) -> list[int]:
    e = np.asarray(elems, dtype=np.int64).reshape(-1, LANES) & 0xFF
    return [int(sum(int(e[b, r]) << (8 * r) for r in range(LANES)))
            for b in range(e.shape[0])]


def f16(v: float) -> int:
    return int(f64_to_f16_bits(np.array([float(v)]))[0])


def w4b_job(rng, K: int, N: int, zero: bool = False,
            s8_mode: str = "tile") -> dict:
    """One (B) job. s8_mode: 'tile' = the tile-amax scale (reduction case),
    'stripe' = a deliberately different host scale (2x tile — the stripe
    semantics the sideband exists for), 'tiny' = rail saturation."""
    W8 = (np.zeros((K, N), dtype=np.int64) if zero
          else rng.integers(-127, 128, size=(K, N), dtype=np.int64))
    blob = wc.compress_weights_w4(W8, group=G)
    _, s8_tile = wc.wfeed_w4_to_i8(blob, "B")
    if s8_mode == "tile":
        s8 = int(s8_tile)
    elif s8_mode == "stripe":
        s8 = f16(float(f16_bits_to_f64(
            np.array([np.uint16(s8_tile)]))[0]) * 2.0)
        if s8 == 0 or (s8 & 0x7C00) == 0x7C00:
            s8 = int(s8_tile)
    else:  # tiny
        s8 = 0x0400                                            # EPS: rails
    flat = wc.wfeed_w4b_to_i8(blob, np.uint16(s8))
    beats = wc.n_wgt_beats(K, N)
    out = words_from_i8(flat)
    packed = packed_words(blob, K, N)
    codes = unpacked_codes(blob, K, N)
    exp = np.asarray(flat, dtype=np.int64)
    return dict(
        beats=beats, K=K, N=N, s8=s8,
        scales=[int(s) for s in blob.scales],
        packed=packed, out=out,
        odd=int(beats & 1),
        cmin=int(np.any(codes == -8)), cmax=int(np.any(codes == 7)),
        czero=int(np.any(np.all(exp.reshape(-1, LANES) == 0, axis=1))),
        cneg=int(np.any(codes < 0)),
        satr=int(np.any(exp == 127) or np.any(exp == -128)))


def packed_words(blob, K, N) -> list[int]:
    pw = blob.packed.reshape(-1, LANES)
    return [int(sum(int(pw[b, r]) << (8 * r) for r in range(LANES)))
            for b in range(pw.shape[0])]


def unpacked_codes(blob, K, N):
    n_elem = wc.n_wgt_beats(K, N) * LANES
    from apex_golden.cq_codec import unpack_int4
    return unpack_int4(blob.packed.reshape(-1), n_elem)


def w4b_job_extreme(rng) -> dict:
    """Hand-built blob OUTSIDE compress's reachable envelope: forced -8
    codes (cmin — unreachable from amax-derived scales) and max-normal
    group scales against the min-subnormal s8 (sh up to 39 — arms the
    MW1 shortcut-removal mutant, whose overflow only exists at sh >= 28).
    The oracle handles it exactly; only the stimulus is hand-shaped."""
    K, N = 64, 8
    gid, ng = wc.group_ids(K, N, G)
    codes = rng.integers(-8, 8, size=(K, N)).astype(np.int64)
    codes[0, :] = -8                                   # cmin guaranteed
    codes[1, :] = 7
    panel = [0x7BFF, 0x0001, 0x0400, 0x63D0, 0x3C00, 0x03FF, 0x1000, 0x7800]
    scales = np.array([panel[g % len(panel)] for g in range(ng)],
                      dtype=np.uint16)
    packed = wc.pack_stream(codes, K, N)
    blob = wc.WeightBlob(K=K, N=N, group=G, pow2=False, n_groups=ng,
                         gid=gid, scales=scales, codes=codes, packed=packed)
    s8 = 0x0001                                        # min subnormal
    flat = wc.wfeed_w4b_to_i8(blob, np.uint16(s8))
    beats = wc.n_wgt_beats(K, N)
    exp = np.asarray(flat, dtype=np.int64)
    return dict(
        beats=beats, K=K, N=N, s8=s8,
        scales=[int(s) for s in scales],
        packed=packed_words(blob, K, N), out=words_from_i8(flat),
        odd=int(beats & 1),
        cmin=1, cmax=1,
        czero=int(np.any(np.all(exp.reshape(-1, LANES) == 0, axis=1))),
        cneg=1,
        satr=int(np.any(exp == 127) or np.any(exp == -128)))


def emit_f(L, j, en=1):
    L.append(f"F {j['beats']} {en} {j['K']} {j['N']} {j['s8']:04x} "
             f"{j['odd']} {j['cmin']} {j['cmax']} {j['czero']} {j['cneg']} "
             f"{j['satr']}")
    if en:
        L += [f"S {s:04x}" for s in j["scales"]]
        L += [f"P {w:016x}" for w in j["packed"]]
    else:
        # passthrough: input beats ARE the expected output beats
        L += [f"P {w:016x}" for w in j["out"]]
    L += [f"E {w:016x}" for w in j["out"]]


def main() -> None:
    out_dir = Path(__file__).parent / "build"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0xD031)

    # ── directed ────────────────────────────────────────────────────────────
    L = ["# gen_w4b_vectors.py directed (golden wfeed_w4b_to_i8, G=%d)" % G]
    for (K, N) in [(8, 1), (16, 2), (32, 8), (33, 3), (64, 8), (100, 7),
                   (256, 8), (2048, 8), (40, 5), (48, 2)]:
        emit_f(L, w4b_job(rng, K, N))                      # tile-scale s8
    for (K, N) in [(64, 8), (96, 4), (2048, 8)]:
        emit_f(L, w4b_job(rng, K, N, s8_mode="stripe"))    # stripe s8
    emit_f(L, w4b_job(rng, 64, 8, s8_mode="tiny"))         # rail saturation
    emit_f(L, w4b_job(rng, 64, 8, zero=True))              # all-zero tile
    emit_f(L, w4b_job_extreme(rng))                        # cmin + sh=39
    emit_f(L, w4b_job(rng, 128, 8), en=0)                  # passthrough
    # illegal jobs: beats mismatch, K=0 shape, N=9
    L.append("I 63 63 8")          # beats != ceil(K/8)*N
    L.append("I 0 0 0")            # zero everything
    L.append("I 2049 2049 1")      # beats > BEATS_MAX geometry
    emit_f(L, w4b_job(rng, 24, 3))                         # clean after
    (out_dir / "vectors_w4b_directed.txt").write_text("\n".join(L) + "\n")
    print(f"directed: {sum(1 for x in L if x.startswith('F '))} jobs, "
          f"3 illegal")

    # ── random ──────────────────────────────────────────────────────────────
    L = ["# random"]
    njobs = 0
    for i in range(120):
        K = int(rng.integers(1, 2049))
        N = int(rng.integers(1, 9))
        mode = ["tile", "stripe", "tiny"][int(rng.integers(0, 3))] \
            if i % 3 else "tile"
        emit_f(L, w4b_job(rng, K, N, s8_mode=mode))
        njobs += 1
        if i % 17 == 16:
            bad_k = int(rng.integers(1, 2049))
            bad_n = int(rng.integers(1, 9))
            good = ((bad_k + 7) // 8) * bad_n
            L.append(f"I {good + 1} {bad_k} {bad_n}")
    (out_dir / "vectors_w4b_random.txt").write_text("\n".join(L) + "\n")
    print(f"random: {njobs} jobs + interleaved illegals")

    # ── mid-op resets ───────────────────────────────────────────────────────
    L = ["# resets  (R <beats> <en> <K> <N> <s8> <abortcyc> <tphase>)"]
    for phase in (1, 2):                                   # RUN, DRAIN
        for (K, N) in [(64, 8), (256, 8)]:
            j = w4b_job(rng, K, N)
            L.append(f"R {j['beats']} 1 {K} {N} {j['s8']:04x} 0 {phase}")
            L += [f"S {s:04x}" for s in j["scales"]]
            L += [f"P {w:016x}" for w in j["packed"]]
            jc = w4b_job(rng, 16, 2)
            emit_f(L, jc)                                  # clean job after
    for a in (7, 33, 101):                                 # cycle aborts
        j = w4b_job(rng, 128, 4)
        L.append(f"R {j['beats']} 1 128 4 {j['s8']:04x} {a} 0")
        L += [f"S {s:04x}" for s in j["scales"]]
        L += [f"P {w:016x}" for w in j["packed"]]
        jc = w4b_job(rng, 8, 1)
        emit_f(L, jc)
    (out_dir / "vectors_w4b_reset.txt").write_text("\n".join(L) + "\n")
    print("resets: 4 targeted + 3 cycle aborts")


if __name__ == "__main__":
    main()
