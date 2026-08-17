#!/usr/bin/env python3
"""gen_walk_golden.py — W-G3 stage B: compile a traced REAL Qwen2.5-7B layer
step into the three replay artifacts, all golden-gated before any RTL runs.

Consumes the bundle from `trace_layer_golden.py` and emits, under --out:

  (a) `<stem>.ops` / `<stem>.sub.ops`  — the fmt=1 64-word descriptor image
      and the expected walker emission stream, produced by the EXISTING
      compiler (`verif/seq_walker/gen_layer_trace.py::build_case`) through
      its additive `inject=` hook, so the wire format has ONE source.
  (b) `ddr_layer_image.bin` + `.json` + `.regions.jsonl` — the pre-swizzled
      DDR weight image: per-tensor regions at the `seq_walker_fmt.image_bases`
      offsets, each tensor laid out as the concatenation of its per-job
      `wgt_beats_ws` blocks in decomposition order (IB_WALK §2.6: "the image
      IS the wire format"), in the IB_FUEL §2 region/manifest format.
  (c) `layer_step.mmio.jsonl` — the per-step host calibration stream
      (IB_WALK §2.5: DPTR->W21, STEP, H+2 RQ words, DPTR->W54, 4 JC words,
      GO, STATUS poll = H+11 = 39 MMIO at H=28), plus `layer_expect.json`
      with the golden-gated observables (KVQ records, per-head epilogue
      outputs, r1/r2 end checkpoints).

SELF-CHECKS (hard asserts — the "golden-side first" gate; nothing here needs
a simulator):
  S1  descriptor passes the `check2` mirror; every per-step word round-trips
  S2  epilogue replicas reproduce `fx.attn_proj` / `fx.ffn_out` BIT-EXACT
      (inherited from the gen_layer_trace compiler) and the RQ table equals
      the golden per-head `rq` pairs
  S3  JC composites conform to the D-030 CANONICAL grade (idempotent,
      frac[12:0]==0, positive-normal) and equal the graded golden chains
  S4  DDR image ROUND-TRIP: every tensor region, re-read from the emitted
      image and un-swizzled, reproduces the source weight matrix bit-exactly
  S5  decomposition tiles each matrix EXACTLY (full coverage, no overlap)
  S6  IB_FUEL §2.3 divisibility: every region's lane-beat count % 8 == 0
  S7  every emitted MXE descriptor is LEGAL for the implemented tile
      (`mxe_ctrl.sv:164` / `mxe_cfg_pkg.sv`: 1<=n<=MXE_N=8, 1<=m<=64,
      1<=k<=K_MAX) — the W-G3 finding, see README.md
  S8  fuel_req records fit the frozen field widths and address the image
  S9  the per-step MMIO stream is exactly H+11 words
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

import gen_layer_trace as glt                                # noqa: E402
import seq_walker_fmt as fmt                                 # noqa: E402
from apex_golden import transformer as tf                    # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402

# implemented MXE tile limits (rtl/mxe/mxe_cfg_pkg.sv, enforced mxe_ctrl.sv:164)
MXE_N, M_TILE_MAX, K_MAX = 8, 64, 2048

# WALK CSR window (docs/design/IB_WALK.md §2.1; apex_top glue 0x5C-0x6C)
WALK_CTRL, WALK_DPTR, WALK_DDATA, WALK_STATUS = 0x5C, 0x60, 0x64, 0x68
CSR_BASE = 0x1000                       # cl_apex BAR0 CSR window
GO_BIT = 0x1


def swizzle_block(B: np.ndarray) -> np.ndarray:
    """The VERIFIED WS weight-load order as bytes, vectorized.

    Scalar reference (`verif/top/l3/gen_l3_vectors.py::wgt_beats_ws`): for
    block B = W[k0:k0+k, 8j:8j+8], for each 8-row pass p, for each of the 8
    columns c, emit one 64-bit beat whose byte r is row 8p+r — i.e. lane r =
    data[8r+7:8r] = image byte 8*beat+r (IB_FUEL §2.2). So the byte stream is
    exactly B viewed as (p, r, c) and transposed to (p, c, r).

    `_selftest_swizzle` proves this identical to the scalar loop."""
    k, n = B.shape
    assert k % 8 == 0 and n == MXE_N, f"block {B.shape}"
    return (np.ascontiguousarray(B, dtype=np.uint8)
            .reshape(k // 8, 8, MXE_N)          # (p, r, c)
            .transpose(0, 2, 1)                 # (p, c, r)
            .reshape(-1))


def unswizzle_block(raw: np.ndarray, k: int) -> np.ndarray:
    """Inverse of `swizzle_block` — the reader's view, used by the S4 gate."""
    return (np.ascontiguousarray(raw, dtype=np.uint8)
            .reshape(k // 8, MXE_N, 8)          # (p, c, r)
            .transpose(0, 2, 1)                 # (p, r, c)
            .reshape(k, MXE_N))


def tensor_bytes_stream(W: np.ndarray, n_job: int
                        ) -> tuple[np.ndarray, list[tuple]]:
    """The tensor's full pre-swizzled wire stream = its per-job blocks
    concatenated in decomposition order (n-split-major, k-split-minor)."""
    k_total, n_total = W.shape
    parts, jobs, off = [], [], 0
    for (n0, k0, k, n, acc) in fmt.jobs(k_total, n_total, n_job):
        assert k % 8 == 0, f"k-split {k} not a multiple of 8 (whole-word rule)"
        ka = k0 * K_MAX
        for jj in range(0, n, MXE_N):
            c0 = n0 * n_job + jj
            parts.append(swizzle_block(W[ka:ka + k, c0:c0 + MXE_N]))
        jobs.append((n0, k0, k, n, acc, off, k * n))
        off += k * n
    return (np.concatenate(parts) if parts
            else np.zeros(0, dtype=np.uint8)), jobs


def _selftest_swizzle() -> int:
    """The vectorized swizzle vs the scalar L3 reference, on shapes that
    exercise both k-splits and column groups."""
    rng = np.random.default_rng(0x76133)
    n_chk = 0
    for (k, n) in ((16, 8), (64, 16), (2048, 8), (24, 32)):
        W = rng.integers(-128, 128, (k, n)).astype(np.int8)
        for j in range(n // MXE_N):
            want = []
            for p in range(k // 8):
                for c in range(8):
                    v = 0
                    for r in range(8):
                        v |= (int(W[8 * p + r][8 * j + c]) & 0xFF) << (8 * r)
                    want.append(v)
            got = swizzle_block(W[:, j * MXE_N:(j + 1) * MXE_N])
            assert np.array_equal(
                np.frombuffer(got.tobytes(), dtype='<u8'),
                np.asarray(want, dtype=np.uint64)), f"swizzle {k}x{n} j={j}"
            assert np.array_equal(unswizzle_block(got, k),
                                  np.asarray(W[:, j * MXE_N:(j + 1) * MXE_N],
                                             dtype=np.uint8)), "unswizzle"
            n_chk += 1
    return n_chk


def pack_u16_bytes(vals: np.ndarray) -> list[int]:
    """Row-granular aux tensors (gammas, phase rows) as little-endian u16
    packed into 64-bit lane beats — 4 codes per beat."""
    v = np.asarray(vals, dtype=np.uint16).reshape(-1)
    assert v.size % 4 == 0, f"u16 count {v.size} not a multiple of 4"
    return [int(v[i]) | (int(v[i + 1]) << 16) | (int(v[i + 2]) << 32)
            | (int(v[i + 3]) << 48) for i in range(0, v.size, 4)]


def phase_table(head_dim: int, theta_base: float, t_max: int) -> np.ndarray:
    """C-ROPE: the per-(position, pair) phase code table, quantized ONCE from
    float64 (`transformer.rope_phase_q`) — an on-tile generator cannot
    reproduce frac(m*theta/2pi) in f64, so bit-exactness REQUIRES this table
    (IB_WALK §2.6). Row m = the codes the walker fetches at pos_m."""
    theta = tf.rope_theta(head_dim, theta_base)
    return np.asarray([tf.rope_phase_q(float(m), theta)
                       for m in range(t_max)], dtype=np.uint16)


def build_images(bun: dict, meta: dict, n_job: int, outdir: Path) -> dict:
    """(b) the DDR image + regions manifest, with the S4/S5/S6 self-checks."""
    D, dffn = meta["D_model"], meta["d_ffn"]
    hd, H_kv = meta["head_dim"], meta["H_kv"]
    wdir = Path(meta["weights_dir"])
    layer = meta["layer"]

    shapes = fmt.tensor_shapes(D, dffn, H_kv, hd)
    nbytes = fmt.tensor_bytes(shapes)
    bases = fmt.image_bases(shapes)
    total_words = bases[fmt.TENS_PHASE] + fmt.beats64_of_bytes(
        nbytes[fmt.TENS_PHASE])

    ph = phase_table(hd, meta["rope_theta"], fmt.T_MAX)
    src = {
        fmt.TENS_G1: bun["gamma1"], fmt.TENS_G2: bun["gamma2"],
        fmt.TENS_PHASE: ph,
    }
    wtag = {fmt.TENS_WQ: "Wq", fmt.TENS_WK: "Wk", fmt.TENS_WV: "Wv",
            fmt.TENS_WO: "Wo", fmt.TENS_WG: "Wg", fmt.TENS_WU: "Wu",
            fmt.TENS_WD: "Wd"}

    image = bytearray(total_words * 64)
    regions, ndesc, roundtrips = [], 0, 0
    for t in range(fmt.TENS_N):
        base_w = bases[t]
        if t in wtag:
            W = np.load(wdir / f"L{layer:02d}_{wtag[t]}.npy", mmap_mode="r")
            assert tuple(W.shape) == shapes[t], \
                f"{wtag[t]} shape {W.shape} != spec {shapes[t]}"
            raw, jobs = tensor_bytes_stream(W, n_job)
            # S5: the decomposition tiles the matrix exactly
            cov = np.zeros(shapes[t], dtype=np.uint8)
            for (n0, k0, k, n, _a, _o, _c) in jobs:
                cov[k0 * K_MAX:k0 * K_MAX + k,
                    n0 * n_job:n0 * n_job + n] += 1
            assert np.all(cov == 1), f"{wtag[t]}: decomposition not a tiling"
            ndesc += len(jobs)
        else:
            raw = np.frombuffer(
                struct.pack(f"<{len(pack_u16_bytes(src[t]))}Q",
                            *pack_u16_bytes(src[t])), dtype=np.uint8)
            jobs = []
        n_lane = raw.size // 8
        # S6: whole-word streaming invariant (IB_FUEL §2.3)
        assert n_lane % 8 == 0, \
            (f"{fmt.TENS_NAMES[t]}: lane-beat count {n_lane} % 8 != 0 — "
             "IB_FUEL.md §2.3 invariant; escalate, do NOT pad")
        assert raw.size == nbytes[t], \
            f"{fmt.TENS_NAMES[t]}: {raw.size} bytes != spec {nbytes[t]}"
        blob = raw.tobytes()
        image[base_w * 64:base_w * 64 + len(blob)] = blob
        beats64 = n_lane // 8
        # S8: the frozen fuel_req record must hold this (base, beats, tag)
        fmt.pack_freq(base_w, beats64, t)
        regions.append({"tensor": fmt.TENS_NAMES[t], "tag": t,
                        "base_64B": base_w, "beats_64B": beats64,
                        "lane_beats": n_lane, "jobs": len(jobs),
                        "sha256": hashlib.sha256(blob).hexdigest()})

    (outdir / "ddr_layer_image.bin").write_bytes(image)
    man = {"format": 1, "kind": "wg3-layer-image",
           "step": meta["step"], "layer": meta["layer"],
           "n_job": n_job, "words": total_words, "bytes": total_words * 64,
           "regions": regions}
    (outdir / "ddr_layer_image.json").write_text(json.dumps(man, indent=1))
    with (outdir / "ddr_layer_image.regions.jsonl").open("w") as f:
        for r in regions:
            f.write(json.dumps({"job": r["tensor"], "base_64B": r["base_64B"],
                                "beats_64B": r["beats_64B"], "tag": r["tag"],
                                "sha256": r["sha256"]},
                               separators=(",", ":")) + "\n")

    # S4: ROUND-TRIP — re-read the emitted image and un-swizzle every weight
    # tensor back to the source matrix. This is the gate that proves the
    # image really IS the wire format and the reader needs zero reformatting.
    img = np.fromfile(outdir / "ddr_layer_image.bin", dtype=np.uint8)
    for t, tag in wtag.items():
        W = np.asarray(np.load(wdir / f"L{layer:02d}_{tag}.npy",
                               mmap_mode="r"), dtype=np.uint8)
        k_total, n_total = W.shape
        b0 = bases[t] * 64
        raw = img[b0:b0 + nbytes[t]]
        rec = np.zeros((k_total, n_total), dtype=np.uint8)
        pos = 0
        for (n0, k0, k, n, _acc) in fmt.jobs(k_total, n_total, n_job):
            ka = k0 * K_MAX
            for jj in range(0, n, MXE_N):
                c0 = n0 * n_job + jj
                rec[ka:ka + k, c0:c0 + MXE_N] = \
                    unswizzle_block(raw[pos:pos + k * MXE_N], k)
                pos += k * MXE_N
        assert pos == nbytes[t], f"{tag}: round-trip consumed {pos} bytes"
        assert np.array_equal(rec, W), f"{tag}: DDR image round-trip MISMATCH"
        roundtrips += 1

    man["selfcheck"] = {"roundtrip_tensors": roundtrips,
                        "proj_descriptors": ndesc}
    return man


def build_mmio(bun: dict, meta: dict, words: list[int], outdir: Path) -> dict:
    """(c) the per-step calibration stream — IB_WALK §2.5's H+11 exactly."""
    H, T = meta["H"], meta["T"]
    ops: list[dict] = []

    def w(a, d, note=None):
        ops.append({"op": "w", "a": CSR_BASE + a, "d": int(d) & 0xFFFFFFFF,
                    **({"note": note} if note else {})})

    w(WALK_DPTR, fmt.W_STEP, "point the auto-inc window at W21 (STEP)")
    w(WALK_DDATA, words[fmt.W_STEP], f"STEP: t_rows={T} pos_m={T-1}")
    for i in range(H + 2):                       # per-head PV + oproj + down
        w(WALK_DDATA, words[fmt.W_RQ0 + i],
          f"RQ[{i}] " + (f"head {i}" if i < H else
                         ("o-proj" if i == H else "down-proj")))
    w(WALK_DPTR, fmt.W_JC0, "re-point at W54 (JC table)")
    for i in range(fmt.JC_SLOTS):
        w(WALK_DDATA, words[fmt.W_JC0 + i],
          f"JC[{i}] {['oproj','down','gate','up'][i]} (D-030 canonical grade)")
    w(WALK_CTRL, GO_BIT, "walk GO")
    ops.append({"op": "poll", "a": CSR_BASE + WALK_STATUS, "m": 0x1, "e": 0x0,
                "note": "busy falls"})

    # S9: the derived per-step cost, asserted not asserted-by-comment
    n_mmio = len(ops)
    assert n_mmio == H + 11, f"per-step MMIO {n_mmio} != H+11 ({H+11})"
    with (outdir / "layer_step.mmio.jsonl").open("w") as f:
        for o in ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    return {"mmio_ops": n_mmio, "formula": "H+11", "H": H}


def build_expect(bun: dict, meta: dict, outdir: Path) -> dict:
    """Golden-gated expectations for the observables the tile exposes."""
    H, T, hd, H_kv = meta["H"], meta["T"], meta["head_dim"], meta["H_kv"]
    grp = meta["gqa_group"]
    exp = {"step": meta["step"], "layer": meta["layer"], "T": T,
           "q_pos": meta["q_pos"], "tier": meta["tier"],
           "heads": [], "kvq_groups": [], "checkpoints": {}}
    for h in range(H):
        exp["heads"].append({
            "head": h, "kv_group": h // grp,
            "s_q": int(bun[f"h{h:02d}_s_q"]),
            "rq_scale": int(bun[f"h{h:02d}_rq"][0]),
            "rq_shift": int(bun[f"h{h:02d}_rq"][1]),
            "o8_sha": hashlib.sha256(
                np.asarray(bun[f"h{h:02d}_o8"], dtype=np.int8)
                .tobytes()).hexdigest()[:16],
            "s_out": float(bun[f"h{h:02d}_s_out"]),
        })
    # one KVQ record set per KV group (the walker stores K@T-1, V@2T-1)
    for g in range(H_kv):
        h0 = g * grp
        exp["kvq_groups"].append({
            "group": g, "rows": 2 * T,
            "k_waddr": T - 1, "v_waddr": 2 * T - 1,
            "K_sha": hashlib.sha256(np.asarray(
                bun[f"h{h0:02d}_K_f16"], dtype=np.uint16).tobytes()
            ).hexdigest()[:16],
            "V_sha": hashlib.sha256(np.asarray(
                bun[f"h{h0:02d}_V_f16"], dtype=np.uint16).tobytes()
            ).hexdigest()[:16],
        })
    for nm in ("r1", "r2"):
        v = np.asarray(bun[nm], dtype=np.float64)
        bits = f64_to_f16_bits(v).astype(np.uint16)
        assert np.array_equal(f16_bits_to_f64(bits), v), \
            f"{nm} is not on the fp16 bus grid"
        exp["checkpoints"][nm] = {
            "len": int(v.size),
            "f16_sha": hashlib.sha256(bits.tobytes()).hexdigest()[:16],
            "first8": [int(x) for x in bits[:8]],
            "last8": [int(x) for x in bits[-8:]]}
        np.save(outdir / f"expect_{nm}_f16.npy", bits)
    (outdir / "layer_expect.json").write_text(json.dumps(exp, indent=1))
    return exp


def build_fenced_walk(bun: dict, meta: dict, words: list[int],
                      bases: dict, nbytes: dict, outdir: Path) -> dict:
    """The WALKED half (lead ruling 1): the step-enable fence that admits
    NORM1 | ROPE | STOREKV | SCORE | PV at real 7B geometry.

    QKV / OPROJ / FFN / RES1 / RES2 / NORM2 are fenced OUT because their
    projection descriptors are refused by the implemented MXE today (README
    F1, D-029 erratum, owned by the walker lane) and because RES/NORM2 lose
    their producer with OPROJ fenced. The fence is UNPOLICED (README F3):
    `walk_desc2_check` has no dependency clauses, so admitting a coherent
    subset is the HOST's responsibility — that is why every enabled step's
    inputs are staged and gated here.

    K-ROPE COVERAGE (lead ruling 2): K is staged PRE-RoPE and the tile
    rotates it on the S-2 store path at l_rope_pos. The arbiter is golden's
    `rope_in_f16` composition (D-030), NOT the legacy `K_f16` — see F5."""
    H, T, hd, H_kv = meta["H"], meta["T"], meta["head_dim"], meta["H_kv"]
    D, grp = meta["D_model"], meta["gqa_group"]
    theta = tf.rope_theta(hd, meta["rope_theta"])

    mask = ((1 << fmt.EN_NORM1) | (1 << fmt.EN_ROPE) | (1 << fmt.EN_STOREKV)
            | (1 << fmt.EN_SCORE) | (1 << fmt.EN_PV))
    w = list(words)
    w[fmt.W_MASK] = fmt.pack_mask(mask)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=hd) == fmt.ERR_NONE, "fenced descriptor refused"

    def f16b(x):
        return f64_to_f16_bits(np.asarray(x, dtype=np.float64)).astype(np.uint16)

    # K staging + the golden-gated post-rope expectation, per KV group
    Kr = np.asarray(bun["K_real"], dtype=np.float64)
    kin, kexp, n_rows = {}, {}, 0
    for g in range(H_kv):
        sl = slice(g * hd, (g + 1) * hd)
        stage = np.zeros((T, hd), dtype=np.uint16)
        want = np.zeros((T, hd), dtype=np.uint16)
        for t in range(T):
            row_f16 = f16_bits_to_f64(f16b(Kr[t, sl]))   # the S-2 bus value
            stage[t] = f16b(Kr[t, sl])
            want[t] = f16b(tf.rope_fx(row_f16, t, theta))
            n_rows += 1
        kin[g], kexp[g] = stage, want
        np.save(outdir / f"stage_K_g{g}_f16.npy", stage)
        np.save(outdir / f"expect_Krope_g{g}_f16.npy", want)

    # the fenced emission stream, reusing the D-028 emitters (single source)
    lv = fmt.lctl(rope_en=1, rope_pos=T - 1)
    L = [f"CASE {meta['step']}_{meta['layer']} FENCED fmt=1 D={hd} T={T} "
         f"H={H} HKV={H_kv} DM={D} pos={T-1} tier=CQ8 mask={mask:03x}"]
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += [f"E FETCH {fmt.TENS_G1:x} {bases[fmt.TENS_G1]:x} "
          f"{fmt.beats64_of_bytes(nbytes[fmt.TENS_G1]):x}"]
    L += [f"E LJOB {D:x}", f"E LCTL {lv:04x}"]
    for g in range(H_kv):
        L += [f"E KVWA {g:x} {T-1:x}", f"E KVWA {g:x} {2*T-1:x}"]
    n_att = 0
    for h in range(H):
        L += [f"HEAD {h:x} ENG {h//grp:x} RQSLOT {h:x}"]
        sc = glt.score_ops(hd, T, bun[f"h{h:02d}_s_q"],
                           bun[f"h{h:02d}_s_k"], bun[f"h{h:02d}_s_v"])
        pv = glt.pv_ops(hd, T, *[int(v) for v in bun[f"h{h:02d}_rq"]])
        f_sc, f_pv = glt.drive_formulas(hd, T)
        assert len(sc) == f_sc and len(pv) == f_pv, f"h{h}: emitter formulas"
        L += [glt.desc9(x) if x.startswith("E DESC") else x for x in sc + pv]
        n_att += len(sc) + len(pv)
    L += ["END"]
    (outdir / "layer_fenced_walk.ops").write_text("\n".join(L) + "\n")

    # every descriptor in the FENCED stream must be tile-legal
    for ln in L:
        f = ln.split()
        if f[:2] == ["E", "DESC"]:
            m_, k_, n_ = int(f[3], 16), int(f[4], 16), int(f[5], 16)
            assert 1 <= n_ <= MXE_N and 1 <= m_ <= M_TILE_MAX \
                and 1 <= k_ <= K_MAX, f"fenced DESC illegal: {ln}"
    n_desc = sum(1 for ln in L if ln.split()[:2] == ["E", "DESC"])
    return {"mask": mask, "mask_steps": "NORM1|ROPE|STOREKV|SCORE|PV",
            "emissions": sum(1 for x in L if x.startswith("E ")),
            "att_emissions": n_att, "descs": n_desc,
            "k_rows_staged": n_rows, "kv_groups": H_kv,
            "krope_arbiter": "golden rope_fx on the f16 S-2 bus value "
                             "== BusMode(rope_in_f16=True) (D-030)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None,
                    help="the trace_layer_golden .npz (default: newest)")
    ap.add_argument("--out", default=str(REPO / "build/walkgold"))
    ap.add_argument("--n-job", type=int, default=MXE_N,
                    help="N-split width; default MXE_N=8 (the IMPLEMENTED "
                         "tile limit). The fmt spec's WALK2_N_JOB=4095 is "
                         "the descriptor FIELD limit — see README.md")
    args = ap.parse_args()

    n_swz = _selftest_swizzle()          # S4 premise: the vectorized swizzle
    print(f"[wg3] swizzle selftest: {n_swz} column groups match the scalar "
          "gen_l3_vectors reference (and un-swizzle round-trips)")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    bp = Path(args.bundle) if args.bundle else \
        sorted(outdir.glob("layer_s*_L*.npz"))[-1]
    meta = json.loads(bp.with_suffix(".json").read_text())
    bun = dict(np.load(bp))
    print(f"[wg3] bundle {bp.name}: step {meta['step']} layer {meta['layer']} "
          f"T={meta['T']} {meta['tier']} ({meta['model']})")

    # ── (a) descriptor image + expected emissions, through the ONE compiler ──
    wdir = Path(meta["weights_dir"])
    layer = meta["layer"]
    lw = tf.LayerWeights(
        **{t: np.load(wdir / f"L{layer:02d}_{t}.npy", mmap_mode="r")
           for t in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")},
        **{f"s_w{t[1:].lower()}": meta["scales"][f"s_w{t[1:].lower()}"]
           for t in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")},
        gamma1=np.asarray(bun["gamma1"], dtype=np.int64),
        gamma2=np.asarray(bun["gamma2"], dtype=np.int64),
        H=meta["H"], head_dim=meta["head_dim"], H_kv=meta["H_kv"],
        rope_theta_base=meta["rope_theta"],
        **{n: np.asarray(bun[n], dtype=np.float64)
           for n in ("bq", "bk", "bv") if meta["has_bias"][n]})
    X = np.asarray(bun["X"], dtype=np.float64)
    fx = tf.decoder_layer_fx(X, lw, meta["tier"], G=meta["G"],
                             q_pos=meta["q_pos"])
    # the re-run must reproduce the traced layer exactly (bundle integrity)
    for nm in ("attn", "attn_proj", "r1", "ffn_out", "r2"):
        assert np.array_equal(np.asarray(getattr(fx, nm), dtype=np.float64),
                              np.asarray(bun[nm], dtype=np.float64)), \
            f"bundle re-run mismatch on {nm}"

    stem = bp.stem
    text, sub, st = glt.build_case(
        stem, meta["H"], meta["head_dim"], meta["H_kv"], meta["d_ffn"],
        meta["T"], 0, any(meta["has_bias"].values()),
        inject=(lw, X, fx), n_job=args.n_job)
    (outdir / f"{stem}.ops").write_text(text)
    (outdir / f"{stem}.sub.ops").write_text(sub)
    words = [int(ln.split()[2], 16) for ln in text.splitlines()
             if ln.startswith("DW ")]
    assert len(words) == fmt.DESC_WORDS
    # S1: the descriptor the tile will be handed is legal per the mirror
    assert fmt.check2(words[0], words[1], words[2], words[3],
                      words[fmt.W_STEP], cfg_d=meta["head_dim"]) \
        == fmt.ERR_NONE, "descriptor refused by the check2 mirror"
    # S2/S3 rode inside build_case (epilogue replicas, store-scale identity,
    # emitter formulas); re-assert the RQ table against the golden here.
    for h in range(meta["H"]):
        want = fmt.pack_rq(*[int(v) for v in bun[f"h{h:02d}_rq"]])
        assert words[fmt.W_RQ0 + h] == want, f"RQ slot {h} != golden rq"
    for i in range(fmt.JC_SLOTS):
        v = words[fmt.W_JC0 + i]
        assert (v & 0x1FFF) == 0 and 0 < ((v >> 23) & 0xFF) < 255, \
            f"JC[{i}] not D-030 graded"
        assert fmt.grade_f32(float(np.uint32(v).view(np.float32))) == v, \
            f"JC[{i}] not idempotent under the canonical grade"

    # S7: every emitted MXE descriptor must be legal for the IMPLEMENTED tile
    illegal = []
    for ln in sub.splitlines():
        f = ln.split()
        if len(f) >= 6 and f[:2] == ["E", "DESC"]:
            m, k, n = int(f[3], 16), int(f[4], 16), int(f[5], 16)
            if not (1 <= n <= MXE_N and 1 <= m <= M_TILE_MAX
                    and 1 <= k <= K_MAX):
                illegal.append((m, k, n))
    n_desc = sum(1 for ln in sub.splitlines()
                 if ln.split()[:2] == ["E", "DESC"])
    if illegal:
        u = sorted(set(illegal))[:4]
        raise SystemExit(
            f"S7 FAIL: {len(illegal)}/{n_desc} descriptors exceed the "
            f"IMPLEMENTED tile limits (1<=n<={MXE_N}, 1<=m<={M_TILE_MAX}, "
            f"1<=k<={K_MAX}) — e.g. (m,k,n)={u}. mxe_ctrl.sv:164 rejects "
            "these. Re-run with --n-job 8, or see README.md finding F1.")

    man = build_images(bun, meta, args.n_job, outdir)
    mm = build_mmio(bun, meta, words, outdir)
    exp = build_expect(bun, meta, outdir)
    shapes = fmt.tensor_shapes(meta["D_model"], meta["d_ffn"], meta["H_kv"],
                               meta["head_dim"])
    fen = build_fenced_walk(bun, meta, words, fmt.image_bases(shapes),
                            fmt.tensor_bytes(shapes), outdir)

    print(f"[wg3] (a) descriptor + emissions: {len(words)} DW words, "
          f"{st['sub_emissions']} expected emissions, {n_desc} MXE DESCs "
          f"(all legal: n<={MXE_N}, m<={M_TILE_MAX}, k<={K_MAX})")
    print(f"[wg3] (b) DDR image: {man['words']} words "
          f"({man['bytes']/1e6:.1f} MB), {len(man['regions'])} tensor "
          f"regions, {man['selfcheck']['proj_descriptors']} weight jobs, "
          f"round-trip verified on {man['selfcheck']['roundtrip_tensors']}/7 "
          f"weight tensors")
    for r in man["regions"]:
        print(f"        {r['tensor']:>5}  base_64B={r['base_64B']:<9} "
              f"beats_64B={r['beats_64B']:<9} lane_beats={r['lane_beats']}")
    print(f"[wg3] (d) FENCED WALK (mask {fen['mask']:#05x} = "
          f"{fen['mask_steps']}): {fen['emissions']} emissions "
          f"({fen['att_emissions']} attention), {fen['descs']} DESCs all "
          f"tile-legal; K-rope: {fen['k_rows_staged']} rows staged PRE-RoPE "
          f"over {fen['kv_groups']} KV groups, arbiter = {fen['krope_arbiter']}")
    print(f"[wg3] (c) per-step MMIO: {mm['mmio_ops']} ops "
          f"({mm['formula']} at H={mm['H']}), expectations: "
          f"{len(exp['heads'])} heads, {len(exp['kvq_groups'])} KVQ groups, "
          f"r1/r2 checkpoints on the fp16 grid")
    print(f"WG3 GOLDEN COMPILE: PASS (S1 descriptor legal; S2 epilogue "
          f"replicas + RQ table golden-exact; S3 JC canonical-graded; "
          f"S4 image round-trip 7/7; S5 decomposition tiles exactly; "
          f"S6 divisibility held on {len(man['regions'])} regions; "
          f"S7 {n_desc} descriptors tile-legal; S8 fuel_req widths ok; "
          f"S9 MMIO == H+11)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
