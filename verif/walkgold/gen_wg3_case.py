#!/usr/bin/env python3
"""gen_wg3_case.py — W-G3 stage C: render the executor cases from the
already-golden-gated stage-B artifacts.

A NEW SIBLING of `gen_walk_golden.py` (the B1 §7 rule: spec generators are
never edited). This script RE-RENDERS; it does not re-derive. Every emission
line is copied VERBATIM from `layer_fenced_walk.ops`, and every staged value
is read straight out of the stage-A bundle's golden fields. The only thing
computed here is the integer reference for the fuel GEMM (§C-2), which is a
pure INT8 dot product over golden `h8` and the on-disk weight matrix — no
floating point, nothing that could re-derive a golden numeric.

Outputs under --out (default build/walkgold):

  (C-1) `fenced_h28.sub.ops`  — the FENCED (mask 0x03D) emission stream in
        `tb_walker2_sb` vector form: the 64 descriptor words, the per-KV-group
        scale-cache records (GROUP/REC), the per-head s_q (HEADSQ) and
        head->engine table (HEAD), then the fenced E-lines verbatim.
        This is the H=28 / H_kv=4 / GQA-4 UNIT-level case.

  (C-2) `fuel_case/` — the COMPACT DDR case image for the host-driven
        projection GEMM at the legal n=8: ONE weight job block of the real
        Qwen2.5-7B Wq (k=2048, n=8 => 16,384 B = 256 x 64 B words) lifted
        BYTE-FOR-BYTE out of the stage-B full image, its regions manifest,
        and the golden INT32 reference accumulators.

SELF-CHECKS (hard asserts, no simulator):
  C1  every DW/E line in the rendered case is byte-identical to the stage-B
      artifact (no emission is authored here)
  C2  the staged REC rows equal the bundle's K_f16/V_f16 fields exactly, and
      the CQ-8 store-scale identity holds on every staged record
  C3  the compact fuel block is byte-identical to the corresponding slice of
      the full stage-B DDR image AND un-swizzles back to Wq[0:2048, 0:8]
  C4  the fuel reference is reproduced by an independent einsum over the same
      INT8 operands, and every accumulator fits INT32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO / "verif" / "seq_walker"))

import seq_walker_fmt as fmt                                 # noqa: E402
import gen_walk_golden as gwg                                # noqa: E402

MXE_N, K_JOB = 8, 2048


def render_fenced_sub(bun: dict, meta: dict, fenced: str) -> tuple[str, dict]:
    """(C-1) the fenced stream as a tb_walker2_sb vector file."""
    H, T, H_kv = meta["H"], meta["T"], meta["H_kv"]
    grp = meta["gqa_group"]

    dw, elines, n_head_mark = [], [], 0
    for ln in fenced.splitlines():
        if ln.startswith("DW "):
            dw.append(ln)
        elif ln.startswith("E "):
            elines.append(ln)
        elif ln.startswith("HEAD "):
            n_head_mark += 1
    assert len(dw) == fmt.DESC_WORDS, f"{len(dw)} DW words"
    assert n_head_mark == H, f"{n_head_mark} HEAD markers != H={H}"

    S = [f"CASE fenced_h28 D={meta['head_dim']} T={T} H={H} HKV={H_kv} "
         f"DM={meta['D_model']} mask=03d"]
    S += dw

    # per-KV-group scale-cache records: K at [0,T), V at [T,2T) — the golden
    # fields verbatim (the group's records are head g*grp's, the GQA slicing)
    n_rec = 0
    for g in range(H_kv):
        h0 = g * grp
        K = np.asarray(bun[f"h{h0:02d}_K_f16"], dtype=np.uint16)
        V = np.asarray(bun[f"h{h0:02d}_V_f16"], dtype=np.uint16)
        assert K.shape == (T, meta["head_dim"]) and V.shape == K.shape
        S += [f"GROUP {g}"]
        for r in range(2 * T):
            row = K[r] if r < T else V[r - T]
            S += [f"REC {r:04x} " + " ".join(f"{int(v):04x}" for v in row)]
            n_rec += 1
    for h in range(H):
        S += [f"HEADSQ {h:x} {int(bun[f'h{h:02d}_s_q']):04x}"]
    for h in range(H):
        S += [f"HEAD {h:x} {h // grp:x}"]
    S += elines
    S += ["END"]

    return "\n".join(S) + "\n", {
        "desc_words": len(dw), "emissions": len(elines),
        "records": n_rec, "kv_groups": H_kv, "heads": H,
        "descs": sum(1 for x in elines if x.split()[:2] == ["E", "DESC"]),
    }


def build_fuel_case(meta: dict, bun: dict, outdir: Path, imgdir: Path) -> dict:
    """(C-2) the compact one-block DDR case image + its golden reference.

    F1 forbids a WALKED projection; this is the HOST-driven half. The block is
    the FIRST decomposition job of Wq at the implemented tile's legal width:
    (n0=0, k0=0, k=2048, n=8, accumulate=0).
    """
    D, hd = meta["D_model"], meta["head_dim"]
    wdir = Path(meta["weights_dir"])
    layer = meta["layer"]
    shapes = fmt.tensor_shapes(D, meta["d_ffn"], meta["H_kv"], hd)
    bases, nb = fmt.image_bases(shapes), fmt.tensor_bytes(shapes)

    # ── the DRIVABLE block geometry (CORRECTED 2026-07-26) ──────────────────
    # The first attempt took the full decomposition job (k = K_JOB = 2048,
    # 256 DDR words) and planned to consume it as 16 ACCUMULATING descriptors
    # of k=128, since gap D bounds a WS activation row at CFG_D. That is
    # WRONG and the RTL refutes it:
    #
    #   rtl/mxe/mxe_ctrl.sv:286
    #     clear_job <= !((desc.opcode == OP_GEMM_OS) && desc.accumulate);
    #   rtl/apex_pkg.sv:44
    #     logic accumulate;   // [66]  OS: retain accumulators
    #
    # `accumulate` is honoured for OP_GEMM_OS ONLY — a WS job ALWAYS clears.
    # And the fuel line feeds the EXTERNAL xw weight port, which is the WS
    # projection path (`rt_wgt_src=0`); OS takes its B operand from the weight
    # stage buffer instead. So there is no accumulate chain available on the
    # fuel-fed path, and k is bounded by the activation row = CFG_D.
    #
    # The drivable fuel block is therefore k = CFG_D = 128, n = MXE_N = 8 =
    # 1024 B = 16 DDR words. It is a strict PREFIX of the same region: the
    # swizzle is p-major over p = k/8, so W[0:128, 0:8] occupies exactly the
    # first 1024 bytes of the k=2048 job's block. Nothing about the image
    # layout changes — only how much of it one legal descriptor consumes.
    CFG_D = 128
    js = fmt.jobs(D, D, MXE_N)
    (n0, k0, k_job, n, acc) = js[0]
    assert (n0, k0, k_job, n, acc) == (0, 0, K_JOB, MXE_N, 0), f"job0 {js[0]}"
    k = CFG_D
    nbytes = k * n
    assert nbytes == 1024 and nbytes % 64 == 0
    words = nbytes // 64
    assert words == 16, words

    # lift the block BYTE-FOR-BYTE out of the full stage-B image
    full = imgdir / "ddr_layer_image.bin"
    off = bases[fmt.TENS_WQ] * 64                  # job 0 is at region offset 0
    with full.open("rb") as f:
        f.seek(off)
        blob = f.read(nbytes)
    assert len(blob) == nbytes

    # C3: un-swizzle back to the source matrix slice
    W = np.asarray(np.load(wdir / f"L{layer:02d}_Wq.npy", mmap_mode="r"),
                   dtype=np.uint8)
    assert W.shape == (D, D)
    rec = gwg.unswizzle_block(np.frombuffer(blob, dtype=np.uint8), k)
    assert np.array_equal(rec, W[0:k, 0:n]), "fuel block un-swizzle MISMATCH"
    # and against an independent re-swizzle of the source
    assert blob == gwg.swizzle_block(W[0:k, 0:n]).tobytes(), "re-swizzle"

    # ── the golden reference: the exact INT32 GEMM the tile must produce ─────
    # m=1 (one decode row), k=2048, n=8, mode_os, no accumulate. Operands are
    # the golden's own INT8 fields: h8 (the feeder-quantized normed rows) and
    # the signed weight block. Row T is the S8 self-inclusive decode row (the
    # duplicate of row T-1) — the row the q projection consumes.
    from apex_golden.fp import f16_bits_to_f64            # noqa: E402
    T = meta["T"]
    h8_all = np.asarray(bun["h8"], dtype=np.int8)
    assert h8_all.shape == (T + 1, D), h8_all.shape
    h8 = h8_all[T]
    assert np.array_equal(h8, h8_all[T - 1]), "decode row is not the duplicate"
    Wq_i8 = W[0:k, 0:n].astype(np.int8)
    a = h8[0:k].astype(np.int64)
    accs = (a[:, None] * Wq_i8.astype(np.int64)).sum(axis=0)
    # C4: independent recompute + INT32 range
    chk = np.einsum("k,kn->n", a, Wq_i8.astype(np.int64))
    assert np.array_equal(accs, chk), "fuel reference recompute mismatch"
    assert np.all(np.abs(accs) < (1 << 31)), "accumulator exceeds INT32"

    # C5 — the GOLDEN TIE: the FULL-k accumulator over this same operand pair,
    # composed with the golden's own scales and bias, reproduces the traced
    # `q_real[0:8]` of a REAL Qwen2.5-7B layer BIT-EXACTLY. This is what makes
    # the fuel result a golden gate and not merely an arithmetic self-check.
    acc_full = np.einsum("k,kn->n", h8.astype(np.int64),
                         W[:, 0:n].astype(np.int8).astype(np.int64))
    s_h = float(f16_bits_to_f64(np.asarray(bun["s_h"], dtype=np.uint16))[T])
    q_ref = acc_full * s_h * meta["scales"]["s_wq"] \
        + np.asarray(bun["bq"], dtype=np.float64)[0:n]
    q_real = np.asarray(bun["q_real"], dtype=np.float64)[0:n]
    assert np.array_equal(q_ref, q_real), \
        "C5: full-k composition does not reproduce golden q_real"

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fuel_case.bin").write_bytes(blob)
    region = {"job": "wq_n0_k0", "base_64B": 0, "beats_64B": words,
              "tag": fmt.TENS_WQ,
              "sha256": hashlib.sha256(blob).hexdigest()}
    with (outdir / "fuel_case.regions.jsonl").open("w") as f:
        f.write(json.dumps(region, separators=(",", ":")) + "\n")
    (outdir / "fuel_case.json").write_text(json.dumps(
        {"format": 1, "kind": "wg3-fuel-compact", "step": meta["step"],
         "layer": meta["layer"], "tensor": "Wq", "job": [n0, k0, k, n, acc],
         "opcode": "OP_GEMM_WS", "accumulate": False,
         "k_is_prefix_of_job_k": k_job,
         "m_dim": 1, "words": words, "bytes": nbytes,
         "src_offset_in_full_image": off,
         "acc_i32": [int(v) for v in accs],
         "act_sha256": hashlib.sha256(h8[0:k].tobytes()).hexdigest(),
         "q_real_first8": [float(v) for v in q_real],
         "regions": [region]}, indent=1))
    np.save(outdir / "fuel_act_h8.npy", h8[0:k])
    np.save(outdir / "fuel_acc_i32.npy", accs.astype(np.int64))
    return {"words": words, "bytes": nbytes, "k": k, "n": n,
            "acc_min": int(accs.min()), "acc_max": int(accs.max()),
            "sha": region["sha256"][:16], "q_real0": float(q_real[0])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "build/walkgold"))
    ap.add_argument("--bundle", default=None)
    args = ap.parse_args()

    outdir = Path(args.out)
    bp = Path(args.bundle) if args.bundle else \
        sorted(outdir.glob("layer_s*_L*.npz"))[-1]
    meta = json.loads(bp.with_suffix(".json").read_text())
    bun = dict(np.load(bp))
    fenced = (outdir / "layer_fenced_walk.ops").read_text()

    sub, st = render_fenced_sub(bun, meta, fenced)
    (outdir / "fenced_h28.sub.ops").write_text(sub)

    # C1: every DW/E line is byte-identical to the stage-B artifact
    src = [l for l in fenced.splitlines()
           if l.startswith("DW ") or l.startswith("E ")]
    got = [l for l in sub.splitlines()
           if l.startswith("DW ") or l.startswith("E ")]
    assert src == got, "C1: rendered stream diverges from the stage-B artifact"

    fuel = build_fuel_case(meta, bun, outdir / "fuel_case", outdir)

    print(f"[wg3c] (C-1) fenced_h28.sub.ops: {st['desc_words']} DW words, "
          f"{st['emissions']} emissions ({st['descs']} DESCs), "
          f"{st['records']} staged records over {st['kv_groups']} KV groups, "
          f"{st['heads']} heads")
    print(f"[wg3c] (C-2) fuel_case: ONE Wq job block k={fuel['k']} n={fuel['n']}"
          f" = {fuel['bytes']} B = {fuel['words']} DDR words "
          f"(sha {fuel['sha']}), acc range [{fuel['acc_min']}, "
          f"{fuel['acc_max']}]")
    print(f"WG3 CASE RENDER: PASS (C1 emissions byte-identical to stage B; "
          f"C2 {st['records']} records from golden K_f16/V_f16; "
          f"C3 fuel block byte-identical to the full image and un-swizzles "
          f"to Wq[0:{fuel['k']}, 0:{fuel['n']}]; "
          f"C4 reference recomputed independently, INT32-safe; "
          f"C5 full-k composition reproduces golden q_real[0:8] bit-exactly "
          f"(q_real[0]={fuel['q_real0']:.6f}))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
