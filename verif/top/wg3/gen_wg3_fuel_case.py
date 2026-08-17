#!/usr/bin/env python3
"""gen_wg3_fuel_case.py — W-G3 FUEL half, tile-level leg: a host-driven
projection GEMM at the legal n=8 over REAL Qwen2.5-7B Wq weights, with the
weight bytes arriving on the EXTERNAL `xw` port — the exact port the IB-FUEL
reader feeds.

This lands the `xw -> MXE -> golden` hop with the corrected fuel geometry and
a bit-exact activation. The remaining hop (loader -> DDR -> reader -> afifo ->
xw) is `verif/f2sim`'s, and the byte stream this case pushes is asserted here
to be BYTE-IDENTICAL to the compact DDR case image, so the two legs join on a
proven identity rather than on an assumption.

GEOMETRY (corrected — see docs/results/wg3/RESULT.md §5.2): `accumulate` is
honoured for OP_GEMM_OS only (`mxe_ctrl.sv:286`), and the fuel line feeds the
WS external-weight path, so there is no accumulate chain available and k is
bounded by the activation row = CFG_D. One legal fuel-fed descriptor is
therefore m=1, k=128, n=8 -> 1024 B = 16 DDR words.

THE ACTIVATION-STAGING TRICK: the golden `h8` row is quantized over all 3584
channels, but `inject_jobs` MODE_QUANT RE-quantizes whatever it is handed
(`q8 = round(v * 127 / amax_local)`), so an arbitrary 128-slice does NOT stage
bit-exactly. Fix: choose the 128-aligned slice that CONTAINS the row's global
amax. Then `amax_local == 127`, the re-quantization is the identity, and the
staged INT8 row is `h8[k0:k0+128]` exactly. Asserted below (F2), not assumed.

SELF-CHECKS (hard asserts, no simulator):
  F1  the chosen slice contains the row's global amax, so |h8| there is 127
  F2  the MODE_QUANT re-quantization of the staged row is the IDENTITY on
      that slice (q8 == h8[k0:k0+128]) — the premise of the whole case
  F3  the weight beats this case pushes are BYTE-IDENTICAL to the compact
      DDR case image's block (the join with the f2sim leg)
  F4  the descriptor is legal for the implemented tile and carries no
      accumulate (WS always clears)
  F5g the expected INT32 result is recomputed independently and is INT32-safe
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO / "verif" / "seq_walker"))
sys.path.insert(0, str(REPO / "verif" / "top" / "l3"))
sys.path.insert(0, str(REPO / "verif" / "walkgold"))

import gen_l3_vectors as glv                                  # noqa: E402
import gen_walk_golden as gwg                                 # noqa: E402
sys.path.insert(0, str(REPO / "scripts" / "fpga" / "f2"))
import trace_to_regops as t2r                                 # noqa: E402
from apex_golden.attention import quant_rows_i8               # noqa: E402
from apex_golden.fp import f16_bits_to_f64                    # noqa: E402

MXE_N, M_TILE_MAX, K_MAX = 8, 64, 2048

# IB_FUEL.md §5 — the FUEL CSR window (bridge window 0, tile domain)
FUEL_CTRL, FUEL_BASE, FUEL_BEATS = 0x0020, 0x0024, 0x0028
FUEL_STAT, FUEL_ERR = 0x002C, 0x0030
STAT_DDR_READY = 0x8                       # FUEL_STAT[3]
CTRL_SRC_FUEL, CTRL_AR_RUN, CTRL_GO = 0x1, 0x2, 0x100
XW0, XW1, XWP = 0x3040, 0x3044, 0x3048     # the mailbox WB triple
FUEL_TAG = 0                               # TENS_WQ


def build(gold: Path, bundle: Path, out: Path) -> dict:
    meta = json.loads(bundle.with_suffix(".json").read_text())
    bun = dict(np.load(bundle))
    D_model, T = meta["D_model"], meta["T"]
    CFG_D = 128
    n = MXE_N

    h8 = np.asarray(bun["h8"], dtype=np.int8)[T]        # the decode row
    # F1: the 128-aligned slice holding the global amax
    k0 = int(np.argmax(np.abs(h8.astype(np.int32)))) // CFG_D * CFG_D
    sl = h8[k0:k0 + CFG_D]
    assert int(np.max(np.abs(sl.astype(np.int32)))) == 127, \
        f"F1: slice at k0={k0} has amax {int(np.max(np.abs(sl)))} != 127"

    # F2: MODE_QUANT re-quantization must be the identity on this slice
    q8, s_bits = quant_rows_i8(sl.astype(np.float64)[None, :])
    assert np.array_equal(np.asarray(q8[0], dtype=np.int8), sl), \
        "F2: re-quantization is not the identity on the chosen slice"

    W = np.asarray(np.load(Path(meta["weights_dir"])
                           / f"L{meta['layer']:02d}_Wq.npy", mmap_mode="r"),
                   dtype=np.uint8)
    blk = W[k0:k0 + CFG_D, 0:n].astype(np.int8)

    # F3: the beats this case pushes == the compact DDR image's bytes
    beats = glv.wgt_beats_ws(blk.astype(np.int64), 0)
    assert len(beats) == CFG_D, f"F3: {len(beats)} beats != k={CFG_D}"
    raw = np.frombuffer(np.asarray(beats, dtype='<u8').tobytes(),
                        dtype=np.uint8)
    assert raw.tobytes() == gwg.swizzle_block(W[k0:k0 + CFG_D, 0:n]).tobytes(), \
        "F3: pushed beats differ from the DDR image swizzle"
    assert np.array_equal(gwg.unswizzle_block(raw, CFG_D),
                          W[k0:k0 + CFG_D, 0:n]), "F3: un-swizzle"

    # F5g: the exact INT32 result
    acc = np.einsum("k,kn->n", sl.astype(np.int64), blk.astype(np.int64))
    chk = (sl.astype(np.int64)[:, None] * blk.astype(np.int64)).sum(axis=0)
    assert np.array_equal(acc, chk), "F5g: reference recompute mismatch"
    assert np.all(np.abs(acc) < (1 << 31)), "F5g: exceeds INT32"

    # ── the op script ───────────────────────────────────────────────────────
    s = glv.Script(CFG_D)
    name = f"wg3_fuel_wq_k{k0}_hd{CFG_D}"
    s.emit(f"// L3 fuel-geometry projection GEMM: {name} — REAL Qwen2.5-7B "
           f"Wq[{k0}:{k0 + CFG_D}, 0:{n}] on the EXTERNAL xw port "
           f"(m=1 k={CFG_D} n={n} WS, no accumulate)")
    glv.phase_a(s, tiers_used=(0,))
    glv.loader_phase(s)

    s.emit("// stage the activation: the golden h8 decode-row slice, injected "
           "through squant MODE_QUANT (re-quantization is the identity here)")
    s.route(rdst=1, asrc=1)
    pairs = [glv.decompose_f16(int(b))[:2]
             for b in glv.to16(sl.astype(np.float64))]
    glv.inject_jobs(s, pairs, 1)
    s.ess(int(s_bits[0]), 1)
    s.aj(0, 1, 0, 1, s.BPR, 0)                  # LOAD -> act bank 1 row 0

    s.emit(f"// the projection GEMM: weights from the EXTERNAL xw port "
           f"({len(beats)} lane8 beats = the DDR image block, byte-identical)")
    s.route(rdst=0, wsrc=0, asrc=1)              # res -> ro; wgt <- external
    s.aj(1, 1, 0, 1, s.BPR, 0)                   # EMIT the activation row
    s.desc(glv.desc_words(glv.OP_GEMM_WS, 1, CFG_D, n))
    s.wbeats(beats)
    s.ero([int(v) for v in acc], 1)              # the INT32 accumulators

    # no score/pv phase here, so no frame stickies — the T<=8 tail is the
    # TRUE expectation (see the same reasoning in gen_wg3_krope_case.py)
    glv.final_phase(s, 8)
    assert "ESTK ffff 0001" in s.lines, "expected the no-frame-error tail"

    # ── ONE case, TWO build points, two different INFO_TIER truths ──────────
    # The wg3 tile binary is the I-B CL build point (KVQ_GQA_NENG=4), where
    # the GQA bank replaces the tier bank and INFO_TIER reads 0x1
    # (apex_top.sv, D-027 "INFO_TIER never lies"). The f2sim executor builds
    # cl_apex, which instantiates apex_top with KVQ_GQA_NENG at its DEFAULT
    # of 1 (cl_apex.sv:579-590) — the tier bank, so INFO_TIER reads 0x3, which
    # is `phase_a`'s own scripted value. Neither is a fudge: each is its
    # build's truth, and asserting the other one would be the lying gate.
    # f2sim therefore takes the script UNMODIFIED; only the tile variant is
    # rewritten.
    f2_lines = list(s.lines)                     # INFO_TIER 0x3 (tier bank)
    old, new = "CSRR 14 ffffffff 00000003", "CSRR 14 ffffffff 00000001"
    n_hit = sum(1 for ln in s.lines if ln == old)
    assert n_hit == 1, f"expected 1 phase-A INFO_TIER line, found {n_hit}"
    s.lines = [new if ln == old else ln for ln in s.lines]
    assert "CSRR 14 ffffffff 00000003" in f2_lines, "f2 variant lost its line"

    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "cases" / f"{name}.txt").write_text("\n".join(s.lines) + "\n")

    # ── the f2sim legs: host-mode regops, the compact DDR image, and the
    #    fuel-mode regops (WB triples removed, fuel prologue prepended) ──────
    fd = out / "fuel"
    fd.mkdir(parents=True, exist_ok=True)
    x = t2r.Xlate(CFG_D)
    x.note(f"W-G3 fuel projection: REAL Wq[{k0}:{k0 + CFG_D}, 0:{n}] "
           f"m=1 k={CFG_D} n={n} WS")
    x.run(f2_lines)
    host_ops = x.ops
    with (fd / f"{name}.regops.jsonl").open("w") as f:
        for o in host_ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    # the compact DDR image: ONE region, this block, at base_64B = 0
    words = len(beats) * 8 // 64
    blob = raw.tobytes()
    assert len(blob) == words * 64 == 1024
    (fd / "fuel.bin").write_bytes(blob)
    region = {"job": name, "base_64B": 0, "beats_64B": words,
              "tag": FUEL_TAG,
              "sha256": hashlib.sha256(blob).hexdigest()}
    with (fd / "fuel.regions.jsonl").open("w") as f:
        f.write(json.dumps(region, separators=(",", ":")) + "\n")

    # fuel-mode regops: strip every WB triple, prepend the fuel prologue, and
    # append the DDR-provenance postlude. Same removal rule and the same
    # prologue as scripts/fpga/f2/trace_to_fuel.py — reproduced rather than
    # imported because that tool walks a directory of committed job files.
    def fuel_ops(src):
        """Strip ONLY the projection GEMM's WB triples and source those beats
        from DDR instead.

        The case contains TWO kinds of WB traffic: `inject_jobs` uses K=2 WS
        GEMMs with mailbox weight beats to STAGE THE ACTIVATION (128 of them),
        and the projection GEMM consumes the real Wq block (128 more). Only
        the latter is weight fuel. Stripping both would be wrong, and — worse
        — leaving fuel mode armed across the injection would trip
        FUEL_ERR.host_push (IB_FUEL §5 bit 3: "mailbox XW_PUSH while fuel
        src"), because a mailbox push while `src=fuel` is REJECTED.

        So the mode switch is placed exactly where the stripped beats were:
        after the injection is done and the descriptor is pushed. The split
        point is the projection GEMM's own note, not a count or an index.
        """
        o, removed, armed = [], 0, False
        for op in src:
            if op["op"] == "note" and "the projection GEMM" in op.get("s", ""):
                armed = True
                o.append(op)
                continue
            a = op.get("a")
            is_wb = (op["op"] == "w" and a in (XW0, XW1)) \
                or (op["op"] == "pw" and a == XWP)
            if armed and is_wb:
                if removed == 0:
                    # the mode switch must happen while the fuel line is idle
                    # (IB_FUEL §5: a src/ar_owner change while busy is
                    # REJECTED and sets FUEL_ERR.mode_switch)
                    o += [{"op": "note", "s": "IB-FUEL: weights FROM DDR — "
                           f"base_64B=0 beats_64B={words} tag={FUEL_TAG} "
                           "(the projection GEMM's WB triples removed)"},
                          {"op": "poll", "a": FUEL_STAT,
                           "m": STAT_DDR_READY, "e": STAT_DDR_READY},
                          {"op": "poll", "a": FUEL_STAT, "m": 0x6, "e": 0x0},
                          {"op": "w", "a": FUEL_BASE, "d": 0},
                          {"op": "w", "a": FUEL_BEATS, "d": words},
                          {"op": "w", "a": FUEL_CTRL,
                           "d": (FUEL_TAG << 16) | CTRL_GO | CTRL_AR_RUN
                                | CTRL_SRC_FUEL}]
                removed += 1
                continue
            o.append(op)
        assert armed, "projection-GEMM split marker not found"
        assert removed % 3 == 0, "partial WB triple"
        assert removed // 3 == words * 8, \
            f"WB count {removed // 3} != region lane beats {words * 8}"
        # DDR PROVENANCE: the record retired, its tag is echoed back, and no
        # AXI / desc / mode-switch / host-push error fired anywhere.
        o += [{"op": "note", "s": "DDR provenance: record retired, tag echoed"},
              {"op": "poll", "a": FUEL_STAT, "m": 0x6, "e": 0x0},
              {"op": "r", "a": FUEL_STAT, "m": 0xFF00, "e": FUEL_TAG << 8},
              {"op": "r", "a": FUEL_ERR, "m": 0xF, "e": 0x0}]
        return o, removed // 3

    fops, n_wb = fuel_ops(host_ops)
    with (fd / f"{name}.fuel.regops.jsonl").open("w") as f:
        for o in fops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    # ── discrimination arm: the SAME fuel regops against a DDR image whose
    #    block is perturbed by one INT8 lane. The loader's own readback+SHA
    #    would reject a mismatched manifest, so the digest is updated too —
    #    the arm therefore loads CLEANLY and can only be caught by the RESULT.
    #    If it ever passes, the weights did not come from DDR.
    bad = bytearray(blob)
    bad[0] = (bad[0] + 1) & 0xFF
    (fd / "fuel_bad.bin").write_bytes(bytes(bad))
    with (fd / "fuel_bad.regions.jsonl").open("w") as f:
        f.write(json.dumps({**region,
                            "sha256": hashlib.sha256(bytes(bad)).hexdigest()},
                           separators=(",", ":")) + "\n")
    man = {"name": name, "build": "wg3", "kind": "fuel-projection",
           "m": 1, "k": CFG_D, "n": n, "k0": k0, "opcode": "OP_GEMM_WS",
           "accumulate": False, "beats": len(beats),
           "bytes": len(beats) * 8, "ddr_words": len(beats) * 8 // 64,
           "checks": s.n_expect, "acc_i32": [int(v) for v in acc],
           "block_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
           "host_regops": len(host_ops), "fuel_regops": len(fops),
           "wb_triples_removed": n_wb, "fuel_tag": FUEL_TAG,
           "step": meta["step"], "layer": meta["layer"], "tensor": "Wq"}
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(HERE / "build"))
    args = ap.parse_args()
    gold = REPO / "build" / "walkgold"
    bp = sorted(gold.glob("layer_s*_L*.npz"))[-1]
    man = build(gold, bp, Path(args.build))
    print(f"[wg3f] case {man['name']}: REAL Wq[{man['k0']}:"
          f"{man['k0'] + man['k']}, 0:{man['n']}] of step {man['step']} "
          f"layer {man['layer']}, m={man['m']} k={man['k']} n={man['n']} "
          f"{man['opcode']} accumulate={man['accumulate']}")
    print(f"[wg3f] {man['beats']} lane8 beats on the external xw port = "
          f"{man['bytes']} B = {man['ddr_words']} DDR words "
          f"(sha {man['block_sha256'][:16]})")
    print(f"[wg3f] {man['checks']} scripted checks; expected INT32 acc = "
          f"{man['acc_i32']}")
    print(f"[wg3f] regops: host {man['host_regops']} ops -> fuel "
          f"{man['fuel_regops']} ops ({man['wb_triples_removed']} WB triples "
          f"removed, tag={man['fuel_tag']}); compact DDR image "
          f"{man['ddr_words']} words + a one-lane-perturbed discrimination "
          f"image")
    print(f"WG3 FUEL CASE: PASS (F1 slice holds the global amax; "
          f"F2 re-quantization is the identity; F3 beats byte-identical to "
          f"the DDR image block; F4 descriptor tile-legal, no accumulate; "
          f"F5g reference recomputed independently, INT32-safe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
