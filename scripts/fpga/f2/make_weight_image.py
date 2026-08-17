#!/usr/bin/env python3
# make_weight_image.py — WEIGHTS RESIDENT IN CARD MEMORY: pack the REAL
# golden-prepared model tensors into the DDR image the fuel reader and the
# walker's fmt=1 tensor table expect.
#
#   python3 scripts/fpga/f2/make_weight_image.py build \
#       --weights-dir build/s8_weights/Qwen2.5-0.5B-Instruct-4bit \
#       --layers 0,1 --out build/ddr_weights_05b
#   python3 scripts/fpga/f2/make_weight_image.py check --out build/ddr_weights_05b
#   python3 scripts/fpga/f2/make_weight_image.py plan  --out build/ddr_weights_05b
#   python3 scripts/fpga/f2/make_weight_image.py selftest        # seconds, offline
#
# ══ WHAT THIS IS ═══════════════════════════════════════════════════════════
# `make_ddr_image.py` (IB-FUEL s2) builds a JOB-granular image: it scrapes the
# xw beats a compiled regops file would have pushed over BAR0 and lays them
# out per job. That is the right artifact for replaying the 18-job gate from
# DDR, and it is useless for a card that holds its weights: the regions are
# keyed by JOB, they exist only for jobs that were already compiled, and a
# job's stream is (via gemm_job.stage_plan) a function of its ACTIVATION.
#
# This tool builds the TENSOR-granular image instead — the one the walker's
# fmt=1 descriptor addresses (`verif/seq_walker/seq_walker_fmt.py`
# TENS_WQ..TENS_PHASE + `image_bases`), from the REAL golden-prepared weight
# files in `build/s8_weights/<model>/`. It invents NO quantization: every
# weight byte is a byte of `L<nn>_{Wq,Wk,Wv,Wo,Wg,Wu,Wd}.npy` /
# `L<nn>_gamma{1,2}.npy` exactly as golden prepared them.
#
# ══ THE LAYOUT (normative sources, not invented here) ══════════════════════
#  * IB_FUEL.md §2.1-§2.2 — "the DDR image IS the wire format": a region is
#    the exact lane8 beat sequence the tile consumes, little-endian u64,
#    64 B-word granular; the reader reformats nothing.
#  * seq_walker_fmt.jobs() — "n-split-major, k-split-minor ... the
#    pre-swizzled DDR image concatenates the per-job weight blocks in exactly
#    this order" (IB_WALK §2.6). n chunks at N_MXE=8; k chunks at the
#    D-AWARE fmt.k_job(d_tile) (third erratum, W1 2026-08-05: 1984 at D=64,
#    2048 at D=128 — the stage buffer's 31-row cap).
#  * gen_l3_vectors.wgt_beats_ws — the per-block beat encoding (mxe_ctrl
#    WLOAD stream): beat index p*8+c packs W[8p+r][c] into byte r.
#
# So tensor T's payload is
#     for (n0, k0) in jobs(K, N, cfg_d=d_tile):     # n-major, k-minor
#         wgt_beats_ws(W[k0*k_job(d_tile) : ..., n0*8 : n0*8+8], 0)
# which is exactly K*N bytes when N % 8 == 0 (every 0.5B/7B tensor).
# MEASURED PROPERTY (2026-08-05 rebuild): because the k-cuts fall on
# multiples of 8 rows and k-minor blocks of one n-block are CONTIGUOUS
# k-ranges, the payload BYTES are invariant under the k-chunk width — the
# 0.5B image rebuilt at k_job(64)=1984 is sha-identical, region for region,
# to the K_JOB=2048 one. Only the per-(n0,k0) block_record ADDRESSING (and
# the manifest's K_JOB) changes.
#
# The fast path here is a numpy reshape/transpose; it is CROSS-CHECKED against
# gen_l3_vectors.wgt_beats_ws itself on sampled blocks of every tensor, so the
# frozen emitter — not this file — remains the definition of the encoding.
#
# ══ REGION ALIGNMENT ═══════════════════════════════════════════════════════
# Each tensor is its own 4 KiB-aligned region (IB_FUEL §2.3 RECOMMENDED, and
# REQUIRED by the in-sim loader: sim_main.cpp:358 refuses a region base that
# is not 4 KiB-aligned). That is a superset of `image_bases`'s dense 64 B
# packing: a fuel record may still address any 64 B-granular sub-block inside
# a tensor, which is what a per-(n0,k0) projection job does.
#
# ══ WHAT IS GATED AND WHAT IS NOT (say it, do not bury it) ═════════════════
#  * Wq/Wk/Wv/Wo/Wg/Wu/Wd  — the MXE xw stream. Byte-for-byte gated: the
#    fuel-fed projection proof (`fuel_proj_05b.py`) replays real blocks of
#    these tensors out of this image through the fuel reader into the tile and
#    grades the accumulators against golden.
#  * g1/g2 (gamma, int16) and phase (RoPE u_q, uint16) — carried as
#    little-endian 16-bit words. NO gate in this tree consumes them from DDR:
#    the walker's gamma/phase fetch path has never been driven end to end
#    (IB_WALK fmt=1 is emission-gated only), so their byte ORDER is this
#    tool's choice, not a verified contract. Marked `"gated": false` in the
#    manifest. Do not quote them as proven.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif/top/l3"),
           str(REPO / "verif/seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_l3_vectors as g3                                    # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
from apex_golden import transformer as tf                      # noqa: E402

ALIGN = 4096
# cl_apex.sv:260 CL_ROWS_MAX -> apex_top STAGE_R_MAX -> apex_stage_buf R_MAX,
# whose own elaboration guard is R_MAX in [8, 31].
STAGE_R_MAX = 31
WEIGHT_TENSORS = (fmt.TENS_WQ, fmt.TENS_WK, fmt.TENS_WV, fmt.TENS_WO,
                  fmt.TENS_WG, fmt.TENS_WU, fmt.TENS_WD)
NPY_OF = {fmt.TENS_WQ: "Wq", fmt.TENS_WK: "Wk", fmt.TENS_WV: "Wv",
          fmt.TENS_WO: "Wo", fmt.TENS_WG: "Wg", fmt.TENS_WU: "Wu",
          fmt.TENS_WD: "Wd", fmt.TENS_G1: "gamma1", fmt.TENS_G2: "gamma2"}


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ══════════════════ 1. the block encoder (fast + cross-checked) ════════════

def block_bytes(block: np.ndarray) -> bytes:
    """The wgt_beats_ws byte stream of one [k, 8] INT8 weight block.

    beat index p*8+c packs W[8p+r][c] into byte r, and a beat is stored
    little-endian, so file byte (p*8+c)*8+r == W[8p+r][c]. As an array that
    is (p, r, c) -> (p, c, r).
    """
    b = np.ascontiguousarray(block, dtype=np.int8)
    k, n = b.shape
    assert n == 8, f"block_bytes: n={n}, the MXE array width is 8"
    assert k % 8 == 0, f"block_bytes: k={k} is not a multiple of 8"
    return np.ascontiguousarray(
        b.reshape(k // 8, 8, 8).transpose(0, 2, 1)).tobytes()


def _crosscheck_block(block: np.ndarray, where: str) -> None:
    """The frozen emitter, not this file, defines the encoding."""
    import struct
    beats = g3.wgt_beats_ws(np.asarray(block, dtype=np.int64), 0)
    want = struct.pack(f"<{len(beats)}Q", *beats)
    got = block_bytes(block)
    if got != want:
        raise AssertionError(
            f"{where}: the vectorised block encoder disagrees with "
            f"gen_l3_vectors.wgt_beats_ws — refusing to write an image whose "
            f"bytes are not the frozen wire format")


def weight_payload(W: np.ndarray, *, name: str, rng: np.random.Generator,
                   d_tile: int, n_crosschecks: int = 2) -> bytes:
    """Tensor T's DDR payload: the fmt.jobs() n-major/k-minor concatenation.

    THIRD ERRATUM (W1, 2026-08-05): the k chunk is D-DEPENDENT
    (fmt.k_job(d_tile): 1984 at D=64, 2048 at D=128 — the stage buffer's
    31-row cap), so the image byte order is a function of the tile row width
    the image targets. A D=128 image is byte-identical to its pre-erratum
    self; a D=64 image re-lays any tensor with K > 1984 (the 0.5B Wd)."""
    W = np.ascontiguousarray(W, dtype=np.int8)
    K, N = W.shape
    assert N % fmt.N_MXE == 0, (
        f"{name}: N={N} is not a multiple of the MXE array width "
        f"{fmt.N_MXE}; a padded last n-block would make the tensor's image "
        f"longer than K*N and break the fmt=1 tensor_bytes contract")
    order = fmt.jobs(K, N, cfg_d=d_tile)
    kb = fmt.k_job(d_tile)
    parts, checked = [], 0
    pick = set(rng.choice(len(order), size=min(n_crosschecks, len(order)),
                          replace=False).tolist()) | {0, len(order) - 1}
    for i, (n0, k0, k, n, _acc) in enumerate(order):
        ka, na = k0 * kb, n0 * fmt.N_MXE
        blk = W[ka:ka + k, na:na + n]
        if i in pick:
            _crosscheck_block(blk, f"{name} job[{i}] n0={n0} k0={k0}")
            checked += 1
        parts.append(block_bytes(blk))
    payload = b"".join(parts)
    assert len(payload) == K * N, \
        f"{name}: payload {len(payload)} B != K*N = {K * N} B"
    return payload


def gamma_payload(g: np.ndarray, *, name: str) -> bytes:
    """int16 gamma row, little-endian. NOT gated — see the header."""
    a = np.asarray(g, dtype=np.int64)
    lo, hi = int(a.min()), int(a.max())
    assert -32768 <= lo and hi <= 32767, \
        f"{name}: gamma {lo}..{hi} outside int16"
    return np.ascontiguousarray(a.astype("<i2")).tobytes()


def phase_payload(head_dim: int, t_max: int, theta_base: float) -> bytes:
    """t_max rows of head_dim/2 uint16 RoPE phase codes (golden's own
    rope_phase_q, C-ROPE: quantized ONCE from float64). NOT gated."""
    theta = tf.rope_theta(head_dim, base=theta_base)
    u = tf.rope_phase_q(np.arange(t_max, dtype=np.float64), theta)
    u = np.asarray(u, dtype=np.int64)
    assert u.shape == (t_max, head_dim // 2), f"phase shape {u.shape}"
    assert u.min() >= 0 and u.max() < (1 << tf.PH_FRAC)
    return np.ascontiguousarray(u.astype("<u2")).tobytes()


# ══════════════════════════ 2. the image builder ═══════════════════════════

def _pad64(b: bytes) -> bytes:
    r = len(b) % 64
    return b if r == 0 else b + bytes(64 - r)


def build(weights_dir: Path, layers: list[int], outdir: Path, *,
          t_max: int = fmt.T_MAX, seed: int = 0xF0E1,
          tensors: tuple[int, ...] | None = None,
          d_tile: int | None = None) -> dict:
    meta = json.loads((weights_dir / "meta.json").read_text())
    d_model, d_ffn = int(meta["D_model"]), int(meta["d_ffn"])
    H, H_kv, hd = int(meta["H"]), int(meta["H_kv"]), int(meta["head_dim"])
    theta_base = float(meta["rope_theta"])
    shapes = fmt.tensor_shapes(d_model, d_ffn, H_kv, hd, t_max=t_max)
    nbytes = fmt.tensor_bytes(shapes)
    want = tuple(range(fmt.TENS_N)) if tensors is None else tuple(tensors)
    d_tile = int(d_tile or hd)      # the tile row width these bytes target
    rng = np.random.default_rng(seed)

    outdir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[int, bytes]] = []          # (base_bytes, padded payload)
    records: list[dict] = []
    cur = 0
    for li in layers:
        assert 0 <= li < int(meta["n_layers"]), f"layer {li} outside the model"
        for t in range(fmt.TENS_N):
            if t not in want:
                continue
            K, N = shapes[t]
            if t in WEIGHT_TENSORS:
                W = np.load(weights_dir / f"L{li:02d}_{NPY_OF[t]}.npy",
                            mmap_mode="r")
                assert tuple(W.shape) == (K, N), (
                    f"L{li:02d} {fmt.TENS_NAMES[t]}: file shape {W.shape} != "
                    f"the fmt=1 tensor_shapes {(K, N)} — the descriptor's "
                    f"tensor table and the weight file disagree")
                assert W.dtype == np.int8, \
                    f"L{li:02d} {fmt.TENS_NAMES[t]}: dtype {W.dtype} != int8"
                payload = weight_payload(np.asarray(W),
                                         name=f"L{li:02d}_{fmt.TENS_NAMES[t]}",
                                         rng=rng, d_tile=d_tile)
                gated = True
            elif t in (fmt.TENS_G1, fmt.TENS_G2):
                g = np.load(weights_dir / f"L{li:02d}_{NPY_OF[t]}.npy")
                assert g.shape == (d_model,), f"gamma shape {g.shape}"
                payload = gamma_payload(g,
                                        name=f"L{li:02d}_{fmt.TENS_NAMES[t]}")
                gated = False
            else:                                  # TENS_PHASE
                payload = phase_payload(hd, t_max, theta_base)
                gated = False
            assert len(payload) == nbytes[t], (
                f"L{li:02d} {fmt.TENS_NAMES[t]}: payload {len(payload)} B != "
                f"fmt.tensor_bytes {nbytes[t]} B")
            base = (cur + ALIGN - 1) // ALIGN * ALIGN
            padded = _pad64(payload)
            beats64 = len(padded) // 64
            assert beats64 == fmt.beats64_of_bytes(nbytes[t])
            assert 0 < beats64 < (1 << fmt.FR_BEATS_W), \
                f"{fmt.TENS_NAMES[t]}: {beats64} words exceeds the 26-bit " \
                f"fuel_req beats field"
            assert base // 64 < (1 << fmt.FR_BASE_W), \
                "image outgrew the 30-bit fuel_req base field (64 GiB)"
            chunks.append((base, padded))
            # Is every k-chunk of this tensor's decomposition STAGEABLE on a
            # D=`d_tile` tile? THIRD ERRATUM (W1, 2026-08-05): fmt.jobs now
            # chunks k at the D-AWARE fmt.k_job(d_tile) (1984 at D=64 —
            # 31 whole stage rows; 2048 at D=128), so this audit is a
            # REGRESSION FENCE, not a disclosure: an unstageable chunk means
            # the decomposition rule drifted, and writing such an image
            # would relay the bytes at an order no build can walk. It is
            # checked through seq_walker_fmt.assert_act_stageable — the
            # stage buffer's OWN acceptance rule, independent of the
            # decomposition constants — and hard-errors, never warns.
            kmax = max((j[2] for j in fmt.jobs(K, N, cfg_d=d_tile)),
                       default=0) if t in WEIGHT_TENSORS else 0
            if t in WEIGHT_TENSORS:
                for (jn0, jk0, jk, jn, _a) in fmt.jobs(K, N, cfg_d=d_tile):
                    fmt.assert_act_stageable(
                        jk, d_tile,
                        f"L{li:02d}_{fmt.TENS_NAMES[t]} n{jn0}_k{jk0}")
            records.append({
                "job": f"L{li:02d}_{fmt.TENS_NAMES[t]}",
                "layer": li, "tensor": fmt.TENS_NAMES[t], "t": t,
                "base_64B": base // 64, "beats_64B": beats64,
                "tag": t, "bytes": nbytes[t], "shape": [K, N],
                "gated": gated, "max_k_chunk": kmax,
                "stage_rows_at_d": (-(-kmax // d_tile)) if kmax else 0,
                "stageable_at_d": True,
                "sha256": hashlib.sha256(padded).hexdigest()})
            cur = base + len(padded)

    image = bytearray(cur)
    for b, blob in chunks:
        image[b:b + len(blob)] = blob
    (outdir / "ddr_image.bin").write_bytes(image)

    manifest = {
        "format": 1,
        "kind": "tensor-granular weight image (walker fmt=1 tensor table)",
        "model": meta.get("model"),
        "weights_dir": str(weights_dir),
        "geometry": {"D_model": d_model, "d_ffn": d_ffn, "H": H, "H_kv": H_kv,
                     "head_dim": hd, "t_max": t_max,
                     "rope_theta": theta_base,
                     # K_JOB is the EFFECTIVE per-D chunk this image is laid
                     # out at (third erratum: fmt.k_job(d_tile), not the
                     # frozen 2048 ceiling) — block_record derives offsets
                     # from it and check() re-derives payloads with it.
                     "K_JOB": fmt.k_job(d_tile), "N_MXE": fmt.N_MXE,
                     "d_tile": d_tile, "STAGE_R_MAX": STAGE_R_MAX},
        "layers": sorted(layers),
        "align": ALIGN,
        "image_bytes": cur,
        "image_sha256": hashlib.sha256(bytes(image)).hexdigest(),
        "tensors": records,
        "disclosure": "g1/g2/phase are carried little-endian 16-bit and are "
                      "NOT consumed by any gate in this tree (gated=false); "
                      "their byte order is this tool's choice.",
    }
    (outdir / "ddr_image.json").write_text(json.dumps(manifest, indent=1))
    with (outdir / "ddr_image.regions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return manifest


# ═════════════════════════════ 3. the checker ══════════════════════════════

def check(outdir: Path, weights_dir: Path | None = None) -> int:
    man = json.loads((outdir / "ddr_image.json").read_text())
    image = (outdir / "ddr_image.bin").read_bytes()
    wdir = Path(weights_dir or man["weights_dir"])
    g = man["geometry"]
    shapes = fmt.tensor_shapes(g["D_model"], g["d_ffn"], g["H_kv"],
                               g["head_dim"], t_max=g["t_max"])
    # third erratum: re-derive payloads at the tile row width the image
    # targets. NOTE (measured, 2026-08-05, W1 verification pass): check()
    # still PASSES on a pre-erratum image — the k-cuts fall on 8-row
    # boundaries, so the payload BYTES are chunk-invariant (the MEASURED
    # PROPERTY note at the top; ddr_image.bin sha-identical across the
    # 2048->1984 relayout) and only the per-block ADDRESSING moved. The
    # loud pre-erratum refusal is block_record's manifest K_JOB assert,
    # where stale addressing would actually bite.
    d_tile = int(g.get("d_tile") or g["head_dim"])
    rng = np.random.default_rng(0xC4EC)
    fails = 0
    if hashlib.sha256(image).hexdigest() != man["image_sha256"]:
        print("CHECK FAIL: whole-image sha256 mismatch")
        fails += 1
    if len(image) != man["image_bytes"]:
        print(f"CHECK FAIL: image {len(image)} B != manifest "
              f"{man['image_bytes']} B")
        fails += 1
    for r in man["tensors"]:
        t, li = r["t"], r["layer"]
        b0, nb = r["base_64B"] * 64, r["beats_64B"] * 64
        sl = image[b0:b0 + nb]
        ok_align = b0 % ALIGN == 0
        ok_sha = hashlib.sha256(sl).hexdigest() == r["sha256"]
        if t in WEIGHT_TENSORS:
            W = np.load(wdir / f"L{li:02d}_{NPY_OF[t]}.npy", mmap_mode="r")
            want = _pad64(weight_payload(np.asarray(W), name=r["job"],
                                         rng=rng, d_tile=d_tile))
        elif t in (fmt.TENS_G1, fmt.TENS_G2):
            want = _pad64(gamma_payload(
                np.load(wdir / f"L{li:02d}_{NPY_OF[t]}.npy"), name=r["job"]))
        else:
            want = _pad64(phase_payload(g["head_dim"], g["t_max"],
                                        g["rope_theta"]))
        ok_bytes = sl == want
        ok_len = len(want) == nb and r["bytes"] == fmt.tensor_bytes(shapes)[t]
        if not (ok_align and ok_sha and ok_bytes and ok_len):
            fails += 1
            print(f"[{r['job']}] CHECK FAIL align={ok_align} sha={ok_sha} "
                  f"bytes={ok_bytes} len={ok_len}")
        else:
            print(f"[{r['job']}] ok: {r['bytes']} B = {r['beats_64B']} words "
                  f"@ 0x{b0:x} tag={r['tag']} gated={r['gated']}")
    print(f"WEIGHTIMG CHECK: tensors={len(man['tensors'])} fails={fails} -> "
          f"{'FAIL' if fails else 'PASS'}")
    return fails


# ══════════════ 4. the CARD loader plan (what the host must do) ════════════
#
# INVESTIGATED, stated plainly (see docs/design/IB_FUEL.md §0 and the kit):
#  * The CL exposes DDR at PCIS byte 0x0, 64 GiB (apex_pcis_slv.sv:108,
#    `in_range` = addr[63:36]==0). PCIS is the AppPF BAR4 memory space
#    (AWS_Shell_Interface_Specification.md: BAR4 = 64-bit prefetchable,
#    128 GiB, mapped onto PCIS). Writes are ALWAYS PCIS-owned; reads are
#    blocked with DECERR while FUEL_CTRL.ar_owner=1 (RUN mode).
#  * The kit's host API for that space already exists —
#    sdk/userspace/include/fpga_pci.h: fpga_pci_attach(slot, pf, APP_PF_BAR4,
#    BURST_CAPABLE) + fpga_pci_write_burst(handle, offset, dwords, len) +
#    fpga_pci_peek64 — and the cython bindings expose all three
#    (fpga_pci_wrapper.pyx: pci_attach/pci_write_burst/pci_peek64).
#  * What does NOT exist is any USE of it in this repo: f2_host_run.py
#    attaches BAR0 only (`pci_attach(slot, 0, 0, 0)`) and drives peek/poke.
#    So the card-side loader is a NEW host script over EXISTING kit calls —
#    not a new hardware path. `f2_ddr_load.py` is that script; this
#    subcommand emits the burst plan it (and any reviewer) works from.

def plan(outdir: Path, burst_dwords: int = 1024) -> dict:
    man = json.loads((outdir / "ddr_image.json").read_text())
    assert burst_dwords % 16 == 0, \
        "a burst must be a whole number of 64 B DDR words (16 dwords)"
    bursts, total = [], 0
    for r in man["tensors"]:
        b0, nb = r["base_64B"] * 64, r["beats_64B"] * 64
        for off in range(0, nb, burst_dwords * 4):
            n = min(burst_dwords * 4, nb - off)
            bursts.append({"tensor": r["job"], "bar4_offset": b0 + off,
                           "bytes": n, "dwords": n // 4,
                           "file_offset": b0 + off})
            total += n
    out = {"bar": "AppPF BAR4 (PCIS) — DDR at PCIS byte 0x0",
           "attach": "fpga_pci_attach(slot, pf=0, bar=4, flags=BURST_CAPABLE)",
           "write": "fpga_pci_write_burst(handle, bar4_offset, dwords)",
           "verify": "fpga_pci_peek64 sweep + per-tensor sha256 vs the "
                     "manifest; FUEL_CTRL.ar_owner must be 0 (LOAD) or PCIS "
                     "reads answer DECERR by design",
           "precondition": "poll FUEL_STAT[3] ddr_ready (DDR calibration) "
                           "before the first write",
           "burst_dwords": burst_dwords,
           "n_bursts": len(bursts), "total_bytes": total,
           "image_sha256": man["image_sha256"],
           "bursts": bursts}
    (outdir / "ddr_load_plan.json").write_text(json.dumps(out, indent=1))
    print(f"DDRPLAN: bursts={len(bursts)} bytes={total} "
          f"({burst_dwords} dwords each) -> {outdir / 'ddr_load_plan.json'}")
    return out


# ═══════════════════ 5. fuel records for a projection block ════════════════

def tensor_record(man: dict, layer: int, tensor: str) -> dict:
    for r in man["tensors"]:
        if r["layer"] == layer and r["tensor"] == tensor:
            return r
    raise KeyError(f"L{layer:02d} {tensor} is not in this image "
                   f"(layers {man['layers']})")


def block_record(man: dict, layer: int, tensor: str, n0: int, k0: int) -> dict:
    """The fuel record (base_64B, beats_64B, tag) of ONE fmt.jobs() block.

    This is the whole point of the tensor-granular image: a projection job's
    weight stream is a CONTIGUOUS run inside the resident tensor, at the
    offset the walker's own decomposition puts it at — no re-swizzle, no
    per-job region.
    """
    r = tensor_record(man, layer, tensor)
    K, N = r["shape"]
    # third erratum: the image is laid out at the D-AWARE chunk — read the
    # tile row width the image was built for, never the frozen ceiling
    g = man["geometry"]
    d_tile = int(g.get("d_tile") or g["head_dim"])
    kb = fmt.k_job(d_tile)
    assert int(g.get("K_JOB", kb)) == kb, (
        f"L{layer:02d} {tensor}: manifest K_JOB={g.get('K_JOB')} != "
        f"fmt.k_job({d_tile})={kb} — a pre-erratum image; rebuild it "
        f"(make_weight_image.py build)")
    order = fmt.jobs(K, N, cfg_d=d_tile)
    off_beats = 0                                   # in lane8 beats
    for (jn0, jk0, k, n, _a) in order:
        if (jn0, jk0) == (n0, k0):
            assert off_beats % 8 == 0, "block start is not 64 B aligned"
            assert k % 8 == 0, f"block k={k} breaks the mod-8 invariant"
            return {"job": f"L{layer:02d}_{tensor}_n{n0}_k{k0}",
                    "base_64B": r["base_64B"] + off_beats // 8,
                    "beats_64B": k // 8, "tag": r["t"],
                    "k": k, "n": n, "k_off": jk0 * kb,
                    "n_off": jn0 * fmt.N_MXE,
                    "byte_off_in_tensor": off_beats * 8,
                    "tensor_base_64B": r["base_64B"]}
        off_beats += k
    raise KeyError(f"no block n0={n0} k0={k0} in L{layer:02d} {tensor}")


# ════════════════════════════════ selftest ═════════════════════════════════

def selftest() -> int:
    import struct
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("MAKE_WEIGHT_IMAGE SELFTEST (offline, no executor, no card)")
    rng = np.random.default_rng(7)

    # 1. the vectorised encoder IS gen_l3_vectors.wgt_beats_ws
    ok = True
    for k in (8, 64, 128, 896):
        b = rng.integers(-127, 128, (k, 8), dtype=np.int64)
        beats = g3.wgt_beats_ws(b, 0)
        if block_bytes(b) != struct.pack(f"<{len(beats)}Q", *beats):
            ok = False
    chk("1 block_bytes == wgt_beats_ws for k in {8,64,128,896}", ok)

    # 2. byte (p*8+c)*8+r is W[8p+r][c] — the encoding stated directly
    b = rng.integers(-127, 128, (64, 8), dtype=np.int64)
    raw = block_bytes(b)
    ok = all(raw[(p * 8 + c) * 8 + r] == (int(b[8 * p + r][c]) & 0xFF)
             for p in range(8) for c in range(8) for r in range(8))
    chk("2 file byte (8p+c)*8+r == W[8p+r][c]", ok)

    # 3. a tensor payload is exactly K*N bytes and n-major/k-minor
    K, N = 256, 24
    W = rng.integers(-127, 128, (K, N), dtype=np.int64).astype(np.int8)
    pay = weight_payload(W, name="synth", rng=rng, d_tile=64)
    chk("3 payload length == K*N", len(pay) == K * N, f"{len(pay)}")
    off, ok = 0, True
    kb64 = fmt.k_job(64)
    for (n0, k0, k, n, _a) in fmt.jobs(K, N, cfg_d=64):
        want = block_bytes(W[k0 * kb64:k0 * kb64 + k,
                             n0 * fmt.N_MXE:n0 * fmt.N_MXE + n])
        if pay[off:off + len(want)] != want:
            ok = False
        off += len(want)
    chk("4 payload == concat of fmt.jobs() blocks, in fmt.jobs() order", ok)

    # 5. a non-multiple-of-8 N must REFUSE (would break tensor_bytes)
    try:
        weight_payload(rng.integers(-9, 9, (16, 12), dtype=np.int64
                                    ).astype(np.int8), name="bad", rng=rng,
                       d_tile=64)
        chk("5 N % 8 != 0 is REFUSED", False, "no refusal")
    except AssertionError as e:
        chk("5 N % 8 != 0 is REFUSED", "MXE array width" in str(e))

    # 6-10. a full tiny image round-trips through build/check/plan/block_record
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wd, od = td / "w", td / "img"
        wd.mkdir()
        # third erratum: hd=64 (a legal stage-row width — d_tile = hd)
        # and dff=4864 (the 0.5B d_ffn), so the tiny image exercises the
        # D-AWARE multi-k-split layout: Wd (4864 x 64) lays out at
        # [1984, 1984, 896] per n-block, 31/31/14 stage rows.
        dm, dff, hkv, hd, H, L = 64, 4864, 2, 64, 4, 2
        meta = {"model": "synth", "n_layers": L, "D_model": dm, "d_ffn": dff,
                "H": H, "H_kv": hkv, "head_dim": hd, "rope_theta": 1e6}
        (wd / "meta.json").write_text(json.dumps(meta))
        sh = fmt.tensor_shapes(dm, dff, hkv, hd, t_max=16)
        for li in range(L):
            for t in WEIGHT_TENSORS:
                k, n = sh[t]
                np.save(wd / f"L{li:02d}_{NPY_OF[t]}.npy",
                        rng.integers(-127, 128, (k, n), dtype=np.int64
                                     ).astype(np.int8))
            for t in (fmt.TENS_G1, fmt.TENS_G2):
                np.save(wd / f"L{li:02d}_{NPY_OF[t]}.npy",
                        rng.integers(-4000, 4000, dm, dtype=np.int64
                                     ).astype(np.int16))
        man = build(wd, [0, 1], od, t_max=16)
        chk("6 every tensor of every layer is in the image",
            len(man["tensors"]) == 2 * fmt.TENS_N, str(len(man["tensors"])))
        chk("7 every region base is 4 KiB aligned",
            all(r["base_64B"] * 64 % ALIGN == 0 for r in man["tensors"]))
        chk("8 check() re-derives the whole image from the weight files",
            check(od, wd) == 0)
        p = plan(od, burst_dwords=64)
        chk("9 the burst plan covers every region byte exactly once",
            p["total_bytes"] == sum(r["beats_64B"] * 64
                                    for r in man["tensors"]),
            f"{p['total_bytes']}")
        # 10: a block record points at the real bytes of the real tensor
        img = (od / "ddr_image.bin").read_bytes()
        W = np.load(wd / "L01_Wq.npy")
        n0 = 3
        br = block_record(man, 1, "Wq", n0, 0)
        got = img[br["base_64B"] * 64: br["base_64B"] * 64
                  + br["beats_64B"] * 64]
        want = block_bytes(W[:, n0 * 8:n0 * 8 + 8])
        chk("10 block_record addresses the block's real bytes in the image",
            got == want and br["beats_64B"] * 64 == len(want),
            f"n0={n0} base_64B={br['base_64B']} beats={br['beats_64B']}")
        # 10b (third erratum): a MID-CHAIN k-split block of the D=64-laid
        # Wd (K=4864 -> [1984, 1984, 896]) — the block class the D-blind
        # chunk made unaddressable — resolves to the real tensor bytes at
        # the D-aware offset.
        Wd = np.load(wd / "L01_Wd.npy")
        kb = fmt.k_job(hd)
        for kk0, (ka, kw) in enumerate([(0, kb), (kb, kb),
                                        (2 * kb, 4864 - 2 * kb)]):
            br = block_record(man, 1, "Wd", 2, kk0)
            got = img[br["base_64B"] * 64: br["base_64B"] * 64
                      + br["beats_64B"] * 64]
            want = block_bytes(Wd[ka:ka + kw, 16:24])
            chk(f"10b block_record Wd n0=2 k0={kk0} (k={kw}) addresses the "
                f"D-aware k-split block",
                got == want and br["k"] == kw and br["k_off"] == ka
                and -(-kw // hd) <= STAGE_R_MAX,
                f"base_64B={br['base_64B']} beats={br['beats_64B']}")
        # 11: a one-byte perturbation of the image must break check()
        bad = od / "bad"
        bad.mkdir()
        (bad / "ddr_image.json").write_text((od / "ddr_image.json").read_text())
        mut = bytearray(img)
        idx = br["base_64B"] * 64 + 7
        mut[idx] ^= 0xFF
        (bad / "ddr_image.bin").write_bytes(bytes(mut))
        chk("11 a single flipped image byte turns check() RED",
            check(bad, wd) > 0, f"byte {idx}")

    print("=" * 60)
    if fails:
        print(f"MAKE_WEIGHT_IMAGE SELFTEST FAIL: {fails}")
        return 1
    print("MAKE_WEIGHT_IMAGE SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pack real golden-prepared model tensors into the DDR "
                    "weight image the fuel reader + walker fmt=1 expect.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--weights-dir", default=str(
        REPO / "build/s8_weights/Qwen2.5-0.5B-Instruct-4bit"))
    b.add_argument("--layers", default="0,1")
    b.add_argument("--out", default=str(REPO / "build/ddr_weights_05b"))
    b.add_argument("--t-max", type=int, default=fmt.T_MAX)
    b.add_argument("--tensors", default="",
                   help="comma-separated TENS names (default: all 10)")
    b.add_argument("--d-tile", type=int, default=None,
                   help="tile row width to report stageability against "
                        "(default: the model head_dim)")

    c = sub.add_parser("check")
    c.add_argument("--out", default=str(REPO / "build/ddr_weights_05b"))
    c.add_argument("--weights-dir", default=None)

    p = sub.add_parser("plan")
    p.add_argument("--out", default=str(REPO / "build/ddr_weights_05b"))
    p.add_argument("--burst-dwords", type=int, default=1024)

    sub.add_parser("selftest")
    args = ap.parse_args()

    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "check":
        return 1 if check(Path(args.out), args.weights_dir) else 0
    if args.cmd == "plan":
        plan(Path(args.out), args.burst_dwords)
        return 0

    layers = [int(v) for v in args.layers.split(",") if v != ""]
    tsel = None
    if args.tensors:
        names = [s.strip() for s in args.tensors.split(",") if s.strip()]
        tsel = tuple(fmt.TENS_NAMES.index(n) for n in names)
    man = build(Path(args.weights_dir), layers, Path(args.out),
                t_max=args.t_max, tensors=tsel, d_tile=args.d_tile)
    for r in man["tensors"]:
        print(f"[{r['job']}] base=0x{r['base_64B'] * 64:x} "
              f"words={r['beats_64B']} bytes={r['bytes']} tag={r['tag']} "
              f"shape={r['shape']} gated={r['gated']} "
              f"stageable={r['stageable_at_d']}")
    print(f"WEIGHTIMG: model={man['model']} layers={man['layers']} "
          f"tensors={len(man['tensors'])} bytes={man['image_bytes']} "
          f"({man['image_bytes'] / 2**20:.1f} MiB) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
