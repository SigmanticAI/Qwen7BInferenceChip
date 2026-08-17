"""gen_w4_vectors.py — stimulus + bit-exact expected results for the B3
native-W4 weight feeder TB, from the golden arbiter
apex_golden.weight_codec (wfeed_w4_to_i8, realization "A", contract S1-S6).

The TB only drives, collects and compares — every expected byte in these
files is produced here by the golden model, never by the RTL and never by
hand (ARCHITECTURE.md:203-205).

RECORD FORMAT (one record per job; '#' lines are comments)

  F <beats> <w4_en> <odd> <cmin> <cmax> <czero> <cneg>
      P <hex64>  x ceil(beats/2) if w4_en else beats   (packed-W4 input)
      E <hex64>  x beats                               (expected INT8 out)
  I <beats>                                            (illegal job)
  R <beats> <w4_en> <abortcyc> <tphase>
      P <hex64>  x consumed                            (mid-op reset job)

  beats = EMITTED INT8 beats = ceil(K/8)*N — the unit pinned in
          weight_codec.py and mxe_wfeed_w4.sv. Consumed packed beats are
          ceil(beats/2), so `odd` marks the tail where the final packed
          beat's upper half is padding and emitted == 2*consumed - 1.
  Coverage flags: cmin/cmax = code -8 / +7 present, czero = an all-zero
          emitted beat, cneg = any negative code (sign-extension live).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
from apex_golden import weight_codec as wc                     # noqa: E402

LANES = wc.LANES
BEATS_MAX_TB = 256          # matches BEATS_MAX_TB in tb_wfeed_w4_sb.sv


def words_from_i8(elems: np.ndarray) -> list[int]:
    """Flat INT8 element stream -> 64-bit beat words, lane r at bits [8r]."""
    e = np.asarray(elems, dtype=np.int64).reshape(-1, LANES) & 0xFF
    return [int(sum(int(e[b, r]) << (8 * r) for r in range(LANES)))
            for b in range(e.shape[0])]


def w4_job(rng, K: int, N: int, zero: bool = False) -> dict:
    """One unpacking job: golden-compressed weights -> (packed in, INT8 out).

    zero=True forces an all-zero weight tile: amax 0 -> the EPS scale floor
    (cq_codec.py:47-48) and every code 0, so every emitted beat is zero.
    That is the w4_code_zero coverage bucket, and it is the case where a
    dropped sign-extension mutant is INVISIBLE — so it must coexist with the
    min/max code cases, never replace them.
    """
    W8 = (np.zeros((K, N), dtype=np.int64) if zero
          else rng.integers(-127, 128, size=(K, N), dtype=np.int64))
    blob = wc.compress_weights_w4(W8, group="tile")
    flat, _scale = wc.wfeed_w4_to_i8(blob, "A")
    beats = wc.n_wgt_beats(K, N)
    out = words_from_i8(flat)
    assert len(out) == beats
    packed = [int(w) for w in wc.beat_words(blob.packed)]
    assert len(packed) == wc.n_packed_beats(K, N)
    codes = flat.reshape(-1, LANES)
    return {
        "beats": beats, "w4_en": 1, "odd": beats & 1,
        "P": packed, "E": out,
        "cmin": int(np.any(flat == -8)), "cmax": int(np.any(flat == 7)),
        "czero": int(np.any(np.all(codes == 0, axis=1))),
        "cneg": int(np.any(flat < 0)),
        "K": K, "N": N,
    }


def pass_job(rng, beats: int) -> dict:
    """One passthrough job (w4_en=0): input beats emitted verbatim, 1:1."""
    e = rng.integers(-128, 128, size=(beats, LANES), dtype=np.int64)
    words = words_from_i8(e.reshape(-1))
    flat = e.reshape(-1)
    return {
        "beats": beats, "w4_en": 0, "odd": 0,
        "P": words, "E": words,
        "cmin": int(np.any(flat == -8)), "cmax": int(np.any(flat == 7)),
        "czero": int(np.any(np.all(e == 0, axis=1))),
        "cneg": int(np.any(flat < 0)),
        "K": 0, "N": 0,
    }


def emit(fh, job: dict, kind: str = "F", extra: str = "") -> None:
    if kind == "F":
        fh.write(f"F {job['beats']} {job['w4_en']} {job['odd']} "
                 f"{job['cmin']} {job['cmax']} {job['czero']} {job['cneg']}"
                 f"   # K={job['K']} N={job['N']}\n")
    else:
        fh.write(f"R {job['beats']} {job['w4_en']} {extra}\n")
    for w in job["P"]:
        fh.write(f"P {w:016x}\n")
    if kind == "F":
        for w in job["E"]:
            fh.write(f"E {w:016x}\n")


# (K, N) geometries. KB*N even -> exact 2x; odd -> the S6 tail case.
DIRECTED_GEOMS = [
    (64, 8),    # the shipped D=64 config, KB*N = 64 (even)
    (128, 8),   # D=128, KB*N = 128
    (70, 8),    # partial final K-chunk (KB=9, klast=6)
    (64, 3),    # narrow N
    (8, 1),     # smallest job: KB*N = 1 beat, ODD -> half a packed beat
    (40, 1),    # KB*N = 5, ODD tail
    (24, 5),    # KB*N = 15, ODD tail
    (2048, 1),  # K_MAX: KB*N = 256 = BEATS_MAX_TB
]


def write_directed(path: Path) -> int:
    rng = np.random.default_rng(11071)
    n = 0
    with path.open("w") as fh:
        fh.write("# directed: every geometry class + both modes + illegal\n")
        for K, N in DIRECTED_GEOMS:
            if wc.n_wgt_beats(K, N) > BEATS_MAX_TB:
                continue
            emit(fh, w4_job(rng, K, N)); n += 1
        # all-zero tiles: EPS scale floor + the w4_code_zero bucket, on both
        # an even and an odd beat count
        for K, N in ((64, 8), (40, 1)):
            emit(fh, w4_job(rng, K, N, zero=True)); n += 1
        # passthrough mode at the same beat counts
        for beats in (1, 5, 15, 64, BEATS_MAX_TB):
            emit(fh, pass_job(rng, beats)); n += 1
        # a passthrough job of all-zero beats (zero bucket in the other mode)
        z = pass_job(rng, 16)
        z["P"] = [0] * 16; z["E"] = [0] * 16
        z["cmin"] = 0; z["cmax"] = 0; z["czero"] = 1; z["cneg"] = 0
        emit(fh, z); n += 1
        # illegal: zero beats, and one past BEATS_MAX
        fh.write("I 0\n")
        fh.write(f"I {BEATS_MAX_TB + 1}\n")
        # a legal job AFTER the rejects — D-019: reject must not wedge the FSM
        emit(fh, w4_job(rng, 64, 8)); n += 1
    return n


def write_random(path: Path, n_jobs: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    n = 0
    with path.open("w") as fh:
        fh.write(f"# random: {n_jobs} jobs, mixed modes and geometries\n")
        for _ in range(n_jobs):
            if rng.random() < 0.75:
                N = int(rng.integers(1, 9))
                kb = int(rng.integers(1, BEATS_MAX_TB // N + 1))
                K = kb * 8 - int(rng.integers(0, 8))   # exercises klast
                K = max(K, 1)
                if wc.n_wgt_beats(K, N) > BEATS_MAX_TB:
                    continue
                emit(fh, w4_job(rng, K, N))
            else:
                emit(fh, pass_job(rng, int(rng.integers(1, 65))))
            n += 1
    return n


def write_reset(path: Path) -> int:
    """Mid-operation reset (ARCHITECTURE.md:189-190, REQUIRED per block).

    tphase targets the DUT FSM state: 1 = ST_RUN, 2 = ST_WAIT; 0 = abort
    after <abortcyc> cycles (random phase, incl. mid-packed-beat).
    """
    rng = np.random.default_rng(50505)
    n = 0
    with path.open("w") as fh:
        fh.write("# mid-op reset: targeted FSM phases + random abort points\n")
        for tphase in (1, 2):
            for K, N in ((64, 8), (40, 1)):
                emit(fh, w4_job(rng, K, N), kind="R", extra=f"0 {tphase}")
                n += 1
        for abortcyc in (1, 2, 3, 5, 8, 13, 21, 34):
            emit(fh, w4_job(rng, 64, 8), kind="R", extra=f"{abortcyc} 0")
            n += 1
        for abortcyc in (1, 4, 9):
            emit(fh, pass_job(rng, 32), kind="R", extra=f"{abortcyc} 0")
            n += 1
    return n


def main(outdir: str) -> int:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    nd = write_directed(d / "vectors_w4_directed.txt")
    nr = write_random(d / "vectors_w4_random.txt", 220, 30303)
    ns = write_random(d / "vectors_w4_storm.txt", 40, 40404)
    nx = write_reset(d / "vectors_w4_reset.txt")
    print(f"gen_w4_vectors: directed={nd} random={nr} storm={ns} reset={nx}")
    print(f"gen_w4_vectors: golden arbiter = apex_golden.weight_codec "
          f"(realization A, S1-S6); wrote 4 vector sets to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
