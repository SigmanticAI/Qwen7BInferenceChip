#!/usr/bin/env python3
# cap_decode.py — decode captured tile values, run the host epilogue, grade
# against golden (audit component 5 / gap N4, docs/design/PROMPT_DEMO_AUDIT.md).
#
# Input is whatever tile_exec_bridge.run_job() returned in ['captures']:
# normalized {'tag','sem','seq','addr','mask','value','i'} records, `value`
# already masked, ALWAYS an unsigned u32.
#
# ══ THE TWO DECODING TRAPS (N4) — both handled here, both discriminated in
#    the selftest ═════════════════════════════════════════════════════════════
#  1. RO lanes are U32-MASKED on the wire (gen_l3_vectors.py:285 emits
#     `int(v) & 0xFFFFFFFF`) while every golden/manifest reference is SIGNED
#     (gen_wg3_fuel_case.py:267 `acc_i32`). A missing u32 -> int32 sign
#     extension reports mismatches on a perfectly correct tile: -16 comes back
#     as 4294967280. ro_lanes_to_i32() sign-extends; nothing else may.
#  2. FS/SS words are FP16 BITS packed {last, data16} under mask 0x1FFFF
#     (trace_to_regops.py:216-222), i.e. value = (last << 16) | f16_bits — a
#     raw compare against a float scale is meaningless. fp16_bits() unpacks.
#
# ══ WHY THE EPILOGUE LIVES HERE ═════════════════════════════════════════════
# Milestone B needs a job whose expectations are NOT golden-derived, and the
# only no-RTL route is rq_en=0 (raw INT32 accumulators out; the wg3 precedent,
# gen_wg3_fuel_case.py:135). The tile then returns `acc_o` and the HOST must
# reproduce golden's Q13 epilogue bit-exactly or token identity dies. So the
# functions are IMPORTED FROM GOLDEN (calib_requant, requant_i32_to_i8) and
# called in exactly the shape attention_core uses (attention.py:395-403) —
# re-deriving the arithmetic here would be a second, ungated implementation.
#
# CLI: python3 cap_decode.py --selftest    (synthetic captures + golden; no
#                                           executor, no FPGA, no build)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))

# The golden arbiter's own functions — never reimplemented.
from apex_golden.attention import (                            # noqa: E402
    calib_requant, requant_i32_to_i8,
)
from apex_golden.fp import f16_bits_to_f64                     # noqa: E402

# Mailbox capture-window addresses (trace_to_regops.py:47-61, B_MB = 0x3000).
B_MB = 0x3000
RO_STATUS, RO_W0, RO_META, RO_ADV = 0x3200, 0x3204, 0x3224, 0x3228
FS_W, SS_W, TD_W = 0x3244, 0x3254, 0x3264
RO_LANES = 8                     # 8 x u32 per RO beat
FS_SS_MASK = 0x1FFFF             # {last, data16}


# ── u32 -> signed ───────────────────────────────────────────────────────────
def sign_extend(v: int, bits: int = 32) -> int:
    """Two's-complement widen: 0xFFFFFFF0 -> -16 (bits=32)."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


# ── RO lanes -> INT32 accumulators ──────────────────────────────────────────
def _ro_lane_index(rec: dict, w0_addr: int, sem_prefix: str):
    """Lane index of an RO-word capture, or None if this record is not one.

    Prefers the EMITTER's semantic tag ('ro_w3' -> 3) and falls back to
    address arithmetic, so the decoder survives a tag-scheme change without
    silently mis-ordering lanes (which would produce a wrong-but-plausible
    accumulator vector — the worst possible failure mode here).
    """
    sem = rec.get("sem", "")
    if sem.startswith(sem_prefix):
        suf = sem[len(sem_prefix):]
        if suf.isdigit():
            return int(suf)
    a = rec.get("addr")
    if a is not None and w0_addr <= a < w0_addr + 4 * RO_LANES \
            and (a - w0_addr) % 4 == 0:
        return (a - w0_addr) // 4
    return None


def ro_beats(captures, *, lanes: int = RO_LANES, w0_addr: int = RO_W0,
             meta_addr: int = RO_META, sem_prefix: str = "ro_w",
             meta_sem: str = "ro_meta", bits: int = 32,
             strict: bool = True) -> list[dict]:
    """Group RO-word captures into per-beat records, SIGN-EXTENDED.

    Returns [{'lanes': [int32 …], 'last': int|None, 'i0': int, 'tags': [...]}]
    in execution order. A new beat starts on a lane index that repeats or goes
    backwards, or right after a beat's meta word (mask 0x1 = `last`) is seen —
    so interleaved fs/ss/td captures between lanes are harmless.

    strict=True (default) raises on a ragged beat: a dropped lane would shift
    every downstream value by one and still grade "plausible".
    """
    caps = sorted(captures, key=lambda c: c["i"])
    beats: list[dict] = []
    cur = None

    def close():
        nonlocal cur
        if cur is not None:
            beats.append(cur)
            cur = None

    for c in caps:
        if c.get("sem") == meta_sem or c.get("addr") == meta_addr:
            if cur is not None:
                cur["last"] = c["value"] & 0x1
            close()
            continue
        j = _ro_lane_index(c, w0_addr, sem_prefix)
        if j is None:
            continue                          # not an RO word — ignore
        if cur is None or j in cur["_seen"] or j < cur["_seen"][-1]:
            close()
            cur = {"lanes": [None] * lanes, "last": None, "i0": c["i"],
                   "tags": [], "_seen": []}
        if j >= lanes:
            raise ValueError(f"RO lane {j} beyond lanes={lanes} (tag {c['tag']})")
        cur["lanes"][j] = sign_extend(c["value"], bits)
        cur["_seen"].append(j)
        cur["tags"].append(c["tag"])
    close()
    for b in beats:
        b.pop("_seen", None)
        if strict and any(v is None for v in b["lanes"]):
            miss = [i for i, v in enumerate(b["lanes"]) if v is None]
            raise ValueError(
                f"ragged RO beat at i={b['i0']}: lanes {miss} never captured — "
                f"a missing lane shifts every later value (tags {b['tags']})")
    return beats


def ro_lanes_to_i32(captures, *, lanes: int = RO_LANES, w0_addr: int = RO_W0,
                    meta_addr: int = RO_META, sem_prefix: str = "ro_w",
                    meta_sem: str = "ro_meta", bits: int = 32,
                    strict: bool = True, expect_n: int | None = None):
    """RO captures -> flat np.int64 vector of SIGNED accumulators, beat-major.

    This is the inverse of `Script.ero` (gen_l3_vectors.py:284) whose lanes are
    `int(v) & 0xFFFFFFFF`: rq_en=1 attention jobs carry o8 there, rq_en=0
    compute-mode jobs carry the raw INT32 acc_o. expect_n asserts the length
    (e.g. D=128), turning a silently short capture into a loud failure.
    """
    bs = ro_beats(captures, lanes=lanes, w0_addr=w0_addr, meta_addr=meta_addr,
                  sem_prefix=sem_prefix, meta_sem=meta_sem, bits=bits,
                  strict=strict)
    flat = [v for b in bs for v in b["lanes"] if v is not None]
    if expect_n is not None and len(flat) != expect_n:
        raise ValueError(f"decoded {len(flat)} RO values, expected {expect_n} "
                         f"({len(bs)} beats x {lanes} lanes)")
    return np.asarray(flat, dtype=np.int64)


# ── FS/SS words -> fp16 ─────────────────────────────────────────────────────
def fp16_bits(captures, *, sems=("fs", "ss"), addrs=(FS_W, SS_W),
              mask: int = FS_SS_MASK) -> list[dict]:
    """Unpack FS/SS scale-tap captures: value = (last << 16) | f16_bits.

    Returns [{'sem','tag','i','bits':u16,'last':0|1,'value':float}] in
    execution order. `bits` is the fp16 SCALE code the tile emitted (s_k / s_v
    / s_c per EFS/ESS, trace_to_regops.py:215-222); `value` is its exact f64.
    """
    out = []
    for c in sorted(captures, key=lambda c: c["i"]):
        if c.get("sem") not in sems and c.get("addr") not in tuple(addrs):
            continue
        v = c["value"] & mask
        b = np.uint16(v & 0xFFFF)
        out.append({"sem": c.get("sem"), "tag": c["tag"], "i": c["i"],
                    "bits": int(b), "last": (v >> 16) & 0x1,
                    "value": float(f16_bits_to_f64(np.array([b]))[0])})
    return out


# B4: LAYER_RDATA captures -> fp16 ──────────────────────────────────────────
# The LAYER window's residual row RAM is read through LAYER_RDATA (0x1088),
# which returns {16'b0, row[rptr]} — a RAW fp16 CODE, not the {last,data16}
# packing fp16_bits() unpacks for the fs/ss taps. It also AUTO-INCREMENTS
# LAYER_RPTR on every read (apex_top.sv:1120), which is why its emitter token
# (gen_layer_ops.LayerScript.erd) is a SINGLE-read `cap` and why --shadow-cap
# must never be used on it. Kept a separate function from fp16_bits() so the
# two packings cannot re-merge — the same independence discipline as
# ro_lanes_to_i32 vs fp16_bits above.
LAYER_RDATA = 0x1088


def rdata_f16(captures, *, sem: str = "lrd", addr: int = LAYER_RDATA,
              mask: int = 0xFFFF) -> list[dict]:
    """LAYER_RDATA captures -> [{'tag','i','bits':u16,'value':float}], in
    execution order (which IS row order: the pointer auto-increments)."""
    out = []
    for c in sorted(captures, key=lambda c: c["i"]):
        if c.get("sem") != sem and c.get("addr") != addr:
            continue
        b = np.uint16(c["value"] & mask)
        out.append({"tag": c["tag"], "i": c["i"], "bits": int(b),
                    "value": float(f16_bits_to_f64(np.array([b]))[0])})
    return out


# ── the host epilogue (golden's Q13, called in golden's shape) ──────────────
def epilogue(acc_i32, s_c, *, target: int = 126, strict: bool = True) -> dict:
    """Raw INT32 accumulators + the C-scale -> the layer's `out_hat`.

    Byte-for-byte the tail of golden `attention_core` (attention.py:395-403):

        amax_o = int(np.max(np.abs(r.acc_o), initial=0)) or 1
        scale, shift = calib_requant(amax_o)
        r.o8 = requant_i32_to_i8(r.acc_o, scale, shift).astype(np.int64)
        s_out = f16(s_c) * float(1 << shift) / float(scale)
        r.out_hat = r.o8.astype(np.float64) * s_out

    s_c: the fp16 BITS of the c-quantization scale (uint16, as AttnCore.s_c and
    the ESS tap carry it) — or a float, which is taken as the already-decoded
    scale value.

    NOTE ON NAMING: golden's `out_hat` is the FLOAT64 dequantized vector
    (`o8 * s_out`) and it is what re-enters the layer (transformer.py:560).
    The INT8 codes are returned separately as 'o8'. Both are provided; grade
    whichever the caller's golden reference holds.
    """
    acc = np.asarray(acc_i32, dtype=np.int64)
    if np.any(np.abs(acc) >= (1 << 31)):
        raise ValueError("acc_i32 exceeds INT32 — sign-extension or lane "
                         "grouping is wrong (see ro_lanes_to_i32)")
    amax_o = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = calib_requant(amax_o, target)
    o8 = requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    if strict:
        assert np.max(np.abs(o8)) < 128 and (
            np.max(np.abs(acc)) * scale) >> shift <= 127, "epilogue clip"
    if isinstance(s_c, (int, np.integer)):
        s_c_val = float(f16_bits_to_f64(np.array([np.uint16(s_c)]))[0])
    else:
        s_c_val = float(s_c)
    s_out = s_c_val * float(1 << shift) / float(scale)
    return {"o8": o8, "out_hat": o8.astype(np.float64) * s_out,
            "rq": (scale, shift), "scale": scale, "shift": shift,
            "s_out": s_out, "s_c": s_c_val, "amax_acc": amax_o}


# ── grading ─────────────────────────────────────────────────────────────────
def grade(out_hat, golden_out_hat, *, atol: float = 0.0) -> dict:
    """Compare a decoded result against golden. atol=0 -> BIT-EXACT.

    {'equal', 'n', 'n_diff', 'worst', 'first_idx'} — works on the float64
    out_hat and equally on the int8 o8 codes.

    GRADE BOTH LEVELS. The requant quantum is 2^shift/scale accumulator counts
    (~152 in the selftest's case), so an acc error smaller than that leaves
    out_hat identical: `out_hat` equality alone does NOT prove the tile's raw
    INT32 accumulators were right. Grade the decoded acc_i32 against golden
    `acc_o` as well — that is the bit-exactness claim; out_hat equality is the
    token-identity claim. Both are asserted in _selftest cases 4/5.
    """
    a = np.asarray(out_hat)
    b = np.asarray(golden_out_hat)
    if a.shape != b.shape:
        return {"equal": False, "n": int(a.size), "n_diff": -1,
                "worst": float("inf"), "first_idx": -1,
                "shape": (tuple(a.shape), tuple(b.shape))}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    bad = d > atol
    idx = np.flatnonzero(bad)
    return {"equal": bool(not bad.any()), "n": int(a.size),
            "n_diff": int(bad.sum()),
            "worst": float(d.max()) if d.size else 0.0,
            "first_idx": int(idx[0]) if idx.size else -1}


# ── selftest: synthetic captures + the real golden functions ────────────────
def _fake_ro_captures(vals, *, lanes: int = RO_LANES, with_meta: bool = True,
                      start_i: int = 0) -> list[dict]:
    """Emit the JSON-line records an executor WOULD write for these signed
    values: u32-masked exactly as `Script.ero` does."""
    caps, i = [], start_i
    vals = list(vals)
    nbeat = (len(vals) + lanes - 1) // lanes
    for b in range(nbeat):
        blk = vals[b * lanes:(b + 1) * lanes]
        for j, v in enumerate(blk):
            caps.append({"tag": f"ro_w{j}_{i}", "sem": f"ro_w{j}", "seq": i,
                         "addr": RO_W0 + 4 * j, "mask": 0xFFFFFFFF,
                         "value": int(v) & 0xFFFFFFFF, "i": i})
            i += 1
        if with_meta:
            last = 1 if b == nbeat - 1 else 0
            caps.append({"tag": f"ro_meta_{i}", "sem": "ro_meta", "seq": i,
                         "addr": RO_META, "mask": 0x1, "value": last, "i": i})
            i += 1
    return caps


def _selftest() -> int:
    from apex_golden.attention import attention_core, TIER_CQ8
    from apex_golden.fp import f64_to_f16_bits
    fails = 0

    def chk(name, cond, extra=""):
        nonlocal fails
        print(f"  [{name}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    # ── 1. sign extension: a NEGATIVE accumulator must round-trip ───────────
    ref = [-16, 1, -1, 0, 2 ** 31 - 1, -(2 ** 31), 12345, -98765]
    caps = _fake_ro_captures(ref)
    got = ro_lanes_to_i32(caps, expect_n=len(ref))
    chk("1a sign-extend round-trip", got.tolist() == ref, f"{got.tolist()}")
    raw = [c["value"] for c in caps if c["sem"].startswith("ro_w")]
    chk("1b the trap is real", raw != ref and raw[0] == 0xFFFFFFF0,
        f"un-extended lane0 = {raw[0]} ({raw[0]:#x}) vs {ref[0]}")
    chk("1c meta `last`", [b["last"] for b in ro_beats(caps)] == [1],
        f"beats={len(ro_beats(caps))}")
    # grouping must also work with NO meta word (lane-wrap only)
    ref2 = list(range(-8, 8))
    chk("1d lane-wrap grouping",
        ro_lanes_to_i32(_fake_ro_captures(ref2, with_meta=False),
                        expect_n=16).tolist() == ref2)
    try:
        bad = [c for c in _fake_ro_captures(ref) if c["sem"] != "ro_w3"]
        ro_lanes_to_i32(bad)
        chk("1e ragged beat rejected", False, "a dropped lane was accepted")
    except ValueError as e:
        chk("1e ragged beat rejected", True, f"{str(e)[:52]}…")

    # ── 2. fp16 tap unpacking ──────────────────────────────────────────────
    sb = int(f64_to_f16_bits(np.array([0.0123456]))[0])
    fs = [{"tag": "fs_9", "sem": "fs", "seq": 9, "addr": FS_W,
           "mask": FS_SS_MASK, "value": (1 << 16) | sb, "i": 9}]
    d = fp16_bits(fs)[0]
    chk("2 fp16 {last,data16}",
        d["bits"] == sb and d["last"] == 1
        and d["value"] == float(f16_bits_to_f64(np.array([np.uint16(sb)]))[0]),
        f"bits={sb:#06x} last={d['last']} value={d['value']}")

    # ── 3. epilogue == golden attention_core's own Q13, on a real core ──────
    rng = np.random.default_rng(20260729)
    D, T = 128, 5
    K = f64_to_f16_bits(rng.normal(0, 1, (T, D))).astype(np.uint16)
    V = f64_to_f16_bits(rng.normal(0, 1, (T, D))).astype(np.uint16)
    core = attention_core(rng.normal(0, 1, D), K, V, TIER_CQ8, G=128)
    ep = epilogue(core.acc_o, core.s_c)
    chk("3a rq pair", ep["rq"] == tuple(core.rq), f"{ep['rq']} vs {tuple(core.rq)}")
    chk("3b o8 bit-exact", np.array_equal(ep["o8"], core.o8),
        grade(ep["o8"], core.o8)["equal"] and "")
    g = grade(ep["out_hat"], core.out_hat)
    chk("3c out_hat bit-exact", g["equal"], f"{g}")
    # …and against the golden primitives called directly (no attention_core)
    amax = int(np.max(np.abs(core.acc_o), initial=0)) or 1
    sc, sh = calib_requant(amax)
    o8d = requant_i32_to_i8(core.acc_o, sc, sh).astype(np.int64)
    s_out_d = float(f16_bits_to_f64(np.array([core.s_c]))[0]) \
        * float(1 << sh) / float(sc)
    chk("3d direct-golden recompute",
        np.array_equal(ep["o8"], o8d)
        and grade(ep["out_hat"], o8d.astype(np.float64) * s_out_d)["equal"],
        f"scale={sc} shift={sh}")

    # ── 4. END TO END: what a compute-mode (rq_en=0) run will actually do ───
    #     tile returns acc_o on the RO lanes -> decode -> epilogue -> grade
    caps = _fake_ro_captures([int(v) for v in core.acc_o])
    acc = ro_lanes_to_i32(caps, expect_n=D)
    chk("4a acc_o decoded", np.array_equal(acc, core.acc_o),
        f"{len(caps)} captures -> {acc.size} values, "
        f"{int(np.sum(core.acc_o < 0))} negative")
    e2e = grade(epilogue(acc, core.s_c)["out_hat"], core.out_hat)
    chk("4b token-path value bit-exact", e2e["equal"], f"{e2e}")

    # ── 5. grade DISCRIMINATES (a grader that always passes proves nothing) ─
    # 5a is the CALIBRATION of the grader, and it is a real caveat: the requant
    # quantum is 2^shift/scale accumulator counts, so an acc error SMALLER than
    # that is invisible in out_hat. A harness that grades only out_hat is
    # therefore WEAKER than one that also grades the raw acc_i32 — grade both.
    sub = core.acc_o.copy()
    sub[7] += 1                                    # one acc LSB = sub-quantum
    gsub = grade(epilogue(sub, core.s_c)["out_hat"], core.out_hat)
    quantum = (1 << ep["shift"]) // ep["scale"] + 1
    chk("5a sub-quantum acc error invisible in out_hat (documented caveat)",
        gsub["equal"], f"quantum≈{quantum} counts -> grade acc_i32 too")
    chk("5a2 raw acc grading DOES catch it",
        not grade(sub, core.acc_o)["equal"], f"{grade(sub, core.acc_o)}")
    big = core.acc_o.copy()
    big[7] += quantum                              # one OUTPUT code
    gp = grade(epilogue(big, core.s_c)["out_hat"], core.out_hat)
    chk("5a3 one-output-code acc error caught", not gp["equal"], f"{gp}")
    unext = np.asarray([c["value"] for c in caps if c["sem"].startswith("ro_w")],
                       dtype=np.int64)
    gu = grade(unext, core.acc_o)
    chk("5b un-sign-extended capture caught", not gu["equal"],
        f"n_diff={gu['n_diff']} worst={gu['worst']:.0f}")
    chk("5c shape mismatch caught", not grade(acc[:-1], core.acc_o)["equal"])

    print(f"CAPDECODE SELFTEST: {'FAIL' if fails else 'PASS'} (fails={fails}; "
          f"golden attention_core D={D} T={T} CQ-8, "
          f"rq={tuple(core.rq)}, {len(caps)} synthetic captures)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cap-file", help="decode a cap-out JSONL and dump RO/FS")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.cap_file:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tile_exec_bridge import parse_cap_file
        caps = parse_cap_file(a.cap_file)
        acc = ro_lanes_to_i32(caps, strict=False)
        print(f"{len(caps)} captures; RO -> {acc.size} int32: {acc[:8].tolist()}…")
        for t in fp16_bits(caps)[:8]:
            print(f"  {t['sem']} tag={t['tag']} bits={t['bits']:#06x} "
                  f"last={t['last']} value={t['value']}")
        return 0
    ap.error("give --selftest or --cap-file")


if __name__ == "__main__":
    sys.exit(main())
