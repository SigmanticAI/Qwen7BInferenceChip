#!/usr/bin/env python3
# fuel_proj_05b.py — THE FUEL PROOF AT 0.5B/D=64: projection GEMMs whose
# weight bytes come out of CARD MEMORY (behavioural DDR) instead of the host
# mailbox, graded bit-exact against golden, with a one-byte DDR discriminator.
#
#   python3 scripts/fpga/f2/fuel_proj_05b.py run                 # the gate
#   python3 scripts/fpga/f2/fuel_proj_05b.py selftest            # offline
#
# ══ THE CLAIM, EXACTLY ═════════════════════════════════════════════════════
# For each job below, ALL of this is true and measured, never assumed:
#   * the WEIGHTS are real golden-prepared Qwen2.5-0.5B INT8 tensors, resident
#     in the (behavioural) DDR as ONE tensor-granular image built by
#     make_weight_image.py — 20 tensors, 2 real layers, loaded and SHA-verified
#     over PCIS by the executor before any job runs;
#   * the job's fuel record addresses a CONTIGUOUS sub-block INSIDE the
#     resident tensor at the offset the WALKER's own n-major/k-minor
#     decomposition puts it at (seq_walker_fmt.jobs) — not a per-job region
#     cut to fit;
#   * the program contains ZERO projection-weight pushes: all 896 of its weight
#     beats are removed (§2 — the TAIL translation; the activation-staging
#     pushes stay on the mailbox, and that is checked, not assumed) and the
#     weight beats arrive
#     PCIS -> sh_ddr -> apex_fuel_reader -> apex_afifo -> xw mux -> MXE;
#   * the 8 INT32 accumulators are `cap` egress (no baked expectation anywhere
#     in the program) and are graded AFTER the run against golden's own
#     compute.gemm_i8_ksplit on the same INT8 operands;
#   * a ONE-BYTE flip in the DDR image turns the fuel run RED and leaves the
#     host-mode run GREEN — so the tile really ate the DDR bytes.
#
# ══ THE ACTIVATION FRAMING, DISCLOSED (this is the load-bearing choice) ════
# gemm_job.stage_plan makes squant's MODE_QUANT the identity by putting the
# activation's GLOBAL amax lane (k*) in lane 0 of every stage row. That
# permutation is a function of the ACTIVATION, so the weight stream it induces
# (plan.Wst) is activation-dependent — which is exactly why a resident,
# activation-independent weight tensor cannot feed it.
#
# This driver stages the row the way THIS IMAGE's own C-1 feeder does instead:
# seam_feeder_quant is elaborated at CFG_DM=64, so the tile frames a D_model
# row as D_model/64 independently-scaled 64-wide frames (layer05b.py calls
# this B-FEED-WIDTH and prints it as a standing cost). Golden's own
# quant_rows_i8 on those 64-wide frames yields codes whose per-row amax is
# 127 BY CONSTRUCTION — so MODE_QUANT is the identity with NO permutation, and
# the weight stream is the natural tensor order. The reframing is applied to
# the ACTIVATION only; not one weight byte is re-quantized.
#
# Consequence worth reading twice: the per-64-frame C-1 framing is what makes
# weights-resident possible on this tile today. The full-row-amax staging that
# layer05b/gemm_job use for host-fed projections cannot be fuelled from a
# resident tensor without a per-job re-swizzle.

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif/top/l3"),
           str(REPO / "verif/seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tinynpu as rt                                       # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import compute as cp                          # noqa: E402

import gemm_job as gj                                          # noqa: E402
import make_ddr_image as mdi                                   # noqa: E402
import make_weight_image as mwi                                # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402
import tile_geom as tg                                         # noqa: E402
import trace_to_fuel as t2f                                    # noqa: E402

D_TILE_05B = 64
DEFAULT_WEIGHTS = REPO / "build/s8_weights/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_IMAGE = REPO / "build/ddr_weights_05b"
DEFAULT_WORK = REPO / "build/fuel_proj_05b"
# The committed 0.5B prompt (build/layer05b_exact/layer05b_result.json,
# docs/results/p2_multilayer/RESULTS.md): "The capital of France is" -> " Paris"
PROMPT_IDS = [785, 6722, 315, 9625, 374]
CAPS_PER_JOB = gj.MXE_N + 1                       # 8 RO lanes + the meta word

# which activation each weight tensor contracts (golden's own graph)
ACT_OF = {"Wq": "h", "Wk": "h", "Wv": "h", "Wg": "h2", "Wu": "h2"}


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═════════════ 1. the real activation, framed the way the tile does ════════

def c1_frames(row: np.ndarray, d: int = D_TILE_05B) -> tuple[np.ndarray, list]:
    """The b64 image's C-1 feeder framing of one real row.

    Golden's OWN quant_rows_i8, applied to dm/d frames of `d` values — which
    is what seam_feeder_quant #(.D(64)) does to the same stream. Returns the
    concatenated INT8 codes and the per-frame fp16 scale bits.
    """
    v = np.asarray(row, dtype=np.float64).reshape(-1, d)
    q8, sb = at.quant_rows_i8(v)
    q8 = np.asarray(q8, dtype=np.int64)
    for r in range(q8.shape[0]):
        a = int(np.max(np.abs(q8[r])))
        if a != 127:
            raise SystemExit(
                f"REFUSE: C-1 frame {r} has amax {a} != 127, so squant "
                f"MODE_QUANT would NOT be the identity and the staged codes "
                f"would not be these codes. (An all-zero frame does this.)")
    return q8.ravel(), [int(s) for s in np.asarray(sb).ravel()]


def golden_prefill(model, ids: list[int], tier: str, group: int) -> dict:
    """Run golden's own Session over the prompt; keep the LAST step's LayerFx.

    rt.Session.step already exposes a `collect(li, fx)` hook — no rebinding of
    golden, no reimplementation of the composition.
    """
    sess = rt.Session(model, tier, group)
    per_step: list[dict] = []
    for tid in ids:
        cur: dict = {}
        sess.step(model.embed_row(tid), collect=lambda li, r: cur.__setitem__(li, r))
        per_step.append(cur)
    return per_step[-1]                       # the last prefill step


def activation_for(fx, kind: str) -> np.ndarray:
    """The REAL row a projection contracts, before the C-1 framing.

    'h'  : RMSNorm-1 output of the decode row (Q7.8) — what Wq/Wk/Wv see
           (transformer.decoder_layer_fx: h8 = quant_rows_i8(h/256)).
    'h2' : RMSNorm-2 output (Q7.8) — what Wg/Wu see.
    """
    if kind == "h":
        return np.asarray(fx.h, dtype=np.float64)[fx.T] / 256.0
    if kind == "h2":
        return np.asarray(fx.h2, dtype=np.float64) / 256.0
    raise KeyError(kind)


# ══════════════ 2. the TAIL fuel translation (and why it is a tail) ════════
#
# trace_to_fuel.py replaces EVERY WB triple in a job with one fuel record.
# That is right for the 18 attention jobs, whose whole xw stream is weights.
# It is WRONG here, and the reason is worth stating: a projection program's
# xw stream is not all weights. `gemm_job.build_script` stages each activation
# row with `gen_l3_vectors.inject_jobs`, which drives the SAME external xw
# port with k=2 WS jobs — measured on this tree at D=64/K=896: 1792 WB triples
# of which 896 are the staged activation and only the LAST 896 (contiguous,
# emitted by `s.wbeats`) are the projection's weights.
#
# So this driver fuels the TAIL: the activation staging stays on the host
# mailbox exactly as today, FUEL_CTRL switches to src=fuel while the reader is
# idle and the FIFO is empty (legal — §5's mode_switch sticky only fires when
# busy), the record streams the weight block, and the FUEL_ERR postlude checks
# `host_push` too, so a stray mailbox push after the switch would be caught.
# That is also the honest shape of the era-2 transition: weights resident,
# activations still streamed.

def fuel_tail_translate(job_path: Path, rec: dict, out_path: Path,
                        n_beats: int) -> dict:
    """Remove the LAST `n_beats` WB triples; arm the fuel record in their
    place. Every other op, in order, byte-identical."""
    ops = [json.loads(l) for l in
           Path(job_path).read_text().splitlines() if l.strip()]
    idx = [i for i, o in enumerate(ops)
           if (o["op"] == "w" and o.get("a") in (mdi.XW0, mdi.XW1))
           or (o["op"] == "pw" and o.get("a") == mdi.XWP)]
    if len(idx) % 3:
        raise SystemExit(f"{job_path}: {len(idx)} WB ops is not whole triples")
    tail = idx[-3 * n_beats:]
    if len(tail) != 3 * n_beats or tail != list(range(tail[0],
                                                      tail[0] + 3 * n_beats)):
        raise SystemExit(
            f"{job_path}: the last {n_beats} weight beats are not a "
            f"contiguous WB run — the emitter's ordering changed and this "
            f"translation would move ops past each other")
    if any(i > tail[-1] for i in idx):
        raise SystemExit(f"{job_path}: a WB push follows the weight block")
    head_beats = (len(idx) - len(tail)) // 3
    pro = [
        {"op": "note", "s": f"IB-FUEL tail: the LAST {n_beats} weight beats "
                            f"come FROM DDR (base_64B={rec['base_64B']} "
                            f"beats_64B={rec['beats_64B']} tag={rec['tag']}); "
                            f"the {head_beats} activation-staging beats above "
                            f"stay on the BAR0 mailbox"},
        {"op": "poll", "a": t2f.FUEL_STAT, "m": t2f.STAT_DDR_READY,
         "e": t2f.STAT_DDR_READY},
        {"op": "w", "a": t2f.FUEL_BASE, "d": rec["base_64B"]},
        {"op": "w", "a": t2f.FUEL_BEATS, "d": rec["beats_64B"]},
        {"op": "w", "a": t2f.FUEL_CTRL,
         "d": (rec["tag"] << 16) | t2f.CTRL_GO | t2f.CTRL_AR_RUN
              | t2f.CTRL_SRC_FUEL},
    ]
    new = ops[:tail[0]] + pro + ops[tail[-1] + 1:]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w") as f:
        for o in new:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    return {"kept": len(new), "removed_beats": n_beats,
            "host_beats_kept": head_beats}


# ═══════════════════════════ 2b. one fuelled job ═══════════════════════════

def build_job(man: dict, weights_dir: Path, layer: int, tensor: str, n0: int,
              x_codes: np.ndarray, work: Path, image_bytes: bytes) -> dict:
    """Emit one projection job + its fuel record, and PROVE the provenance."""
    rec = mwi.block_record(man, layer, tensor, n0, 0)
    W = np.load(weights_dir / f"L{layer:02d}_{tensor}.npy", mmap_mode="r")
    K, N = W.shape
    assert rec["k"] == K, f"{rec['job']}: k={rec['k']} != tensor K={K}"
    assert x_codes.shape[0] == K, \
        f"{rec['job']}: activation {x_codes.shape[0]} != K={K}"
    blk = np.asarray(W[:, rec["n_off"]:rec["n_off"] + rec["n"]],
                     dtype=np.int64)

    name = rec["job"]
    with tg.at_d(D_TILE_05B):
        path, jman = gj.build_gemm_job_full(blk, x_codes, work / "host", name)
    tg.audit_geometry(path, D_TILE_05B)
    tg.retarget_info_tier(path, tg.IMG_05B)

    # ── PROVENANCE: the program's PROJECTION weight stream IS the resident
    #    tensor's bytes at the record's address. Re-extracted from the EMITTED
    #    FILE (make_ddr_image's own RTL-faithful WB replay), never from the
    #    arrays — and taken as the TAIL, per the note above.
    import struct
    all_beats = mdi.extract_beats(path)
    n_w = int(jman["weight_beats"])
    assert n_w == K, f"{name}: manifest weight_beats {n_w} != K={K}"
    beats = all_beats[-n_w:]
    wire = struct.pack(f"<{len(beats)}Q", *beats)
    b0, nb = rec["base_64B"] * 64, rec["beats_64B"] * 64
    resident = image_bytes[b0:b0 + nb]
    if len(wire) != nb or wire != resident:
        raise SystemExit(
            f"REFUSE: {name}: the program's {len(wire)} projection weight "
            f"bytes are NOT the {nb} bytes resident at DDR 0x{b0:x}. The "
            f"image layout and the emitter's wire format have diverged — "
            f"fuelling this job would prove nothing.")

    fuel_dir = work / "fuel"
    fuel_dir.mkdir(parents=True, exist_ok=True)
    fpath = fuel_dir / f"{name}.fuel.regops.jsonl"
    tr = fuel_tail_translate(path, rec, fpath, n_w)
    kept, nbeats = tr["kept"], tr["removed_beats"]

    golden = cp.gemm_i8_ksplit(x_codes[None, :].astype(np.int64), blk)[0]
    assert np.array_equal(golden, np.einsum("k,kn->n", x_codes.astype(np.int64),
                                            blk)), \
        f"{name}: golden's C-KSPLIT disagrees with the plain integer product"
    return {"name": name, "layer": layer, "tensor": tensor, "n0": n0,
            "k": int(K), "n": int(rec["n"]), "n_off": rec["n_off"],
            "host_path": str(path), "fuel_path": str(fpath),
            "base_64B": rec["base_64B"], "beats_64B": rec["beats_64B"],
            "tag": rec["tag"], "tensor_base_64B": rec["tensor_base_64B"],
            "byte_off_in_tensor": rec["byte_off_in_tensor"],
            "wire_bytes": len(wire), "weight_beats": nbeats,
            "host_staging_beats_kept": tr["host_beats_kept"],
            "ops_host": jman["regops"], "ops_fuel": kept,
            "golden": [int(v) for v in golden],
            "x_codes": x_codes, "W_block": blk}


# ═══════════════════════════ 3. run + grade ════════════════════════════════

def run_and_grade(jobs: list[dict], key: str, binary: Path, work: Path,
                  tag: str, *, extra=(), timeout_s: int = 3600) -> dict:
    paths = [j[key] for j in jobs]
    cap_out = work / f"caps_{tag}.jsonl"
    t0 = time.perf_counter()
    r = bridge.run_job(paths, executor="sim", binary=str(binary),
                       cap_out=str(cap_out), tile_div=bridge.TILE_DIV_DEFAULT,
                       extra_args=tuple(extra), timeout_s=timeout_s)
    wall = time.perf_counter() - t0
    caps = r["captures"]
    want = CAPS_PER_JOB * len(jobs)
    out = {"tag": tag, "rc": r["rc"], "ok": r["ok"], "wall": round(wall, 1),
           "n_caps": len(caps), "want_caps": want, "log": r["log"],
           "grades": [], "argv": r["argv"]}
    if len(caps) != want:
        out["error"] = (f"{len(caps)} caps, expected {want} "
                        f"({len(jobs)} jobs x {CAPS_PER_JOB})")
        return out
    for i, j in enumerate(jobs):
        acc = gj.decode_acc(caps[i * CAPS_PER_JOB:(i + 1) * CAPS_PER_JOB])
        g = np.asarray(j["golden"], dtype=np.int64)
        eq = bool(np.array_equal(np.asarray(acc, dtype=np.int64), g))
        out["grades"].append({
            "job": j["name"], "bit_exact": eq,
            "tile": [int(v) for v in acc], "golden": [int(v) for v in g],
            "n_lane_diff": int(np.sum(np.asarray(acc, dtype=np.int64) != g))})
    out["all_bit_exact"] = all(x["bit_exact"] for x in out["grades"])
    out["fuel_audit_ok"] = out["log"].count("-> ok")
    out["fuel_audit_fail"] = out["log"].count("FUELAUDIT") - out["fuel_audit_ok"]
    return out


def mutate_image(src: Path, dst: Path, job: dict) -> dict:
    """Flip ONE byte of ONE fuelled block — chosen so the flip MUST change the
    accumulator (its activation lane is non-zero), and say which byte."""
    img = bytearray(Path(src).read_bytes())
    x, blk = job["x_codes"], job["W_block"]
    k = int(np.argmax(np.abs(np.asarray(x, dtype=np.int64))))   # |x[k]| is max
    assert int(x[k]) != 0, "every activation code is zero — pick another job"
    c = 0                                       # RO lane 0 of this block
    p, r = k // 8, k % 8
    off = job["base_64B"] * 64 + (p * 8 + c) * 8 + r
    old = img[off]
    img[off] = old ^ 0xFF
    Path(dst).write_bytes(bytes(img))
    new_i8 = img[off] - 256 if img[off] >= 128 else img[off]
    old_i8 = old - 256 if old >= 128 else old
    return {"byte_offset": off, "job": job["name"], "k": k, "lane": c,
            "old": old_i8, "new": new_i8, "x_k": int(x[k]),
            "delta_lane0": int(x[k]) * (new_i8 - old_i8),
            "tensor_byte": job["byte_off_in_tensor"] + (p * 8 + c) * 8 + r}


# ══════════════════════════════ 4. the gate ════════════════════════════════

def run(args) -> int:
    work = Path(args.work)
    if work.exists() and args.clean:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    wdir, idir = Path(args.weights_dir), Path(args.image)
    image = tg.IMG_05B
    binary = Path(args.binary) if args.binary else \
        REPO / "verif/f2sim" / args.obj / "f2sim"
    if not binary.exists():
        eprint(f"REFUSE: the DDR=1 {image.name} twin is not built at {binary}.\n"
               f"  cd verif/f2sim && make build D={image.cfg_d} DDR=1 "
               f"OBJ={args.obj} VFLAGS_EXTRA=\"{image.defines()}\"")
        return 2

    # [1] the weight image (real tensors, real layers)
    layers = [int(v) for v in args.layers.split(",")]
    if args.rebuild_image or not (idir / "ddr_image.json").exists():
        eprint(f"[1] building the DDR weight image for layers {layers} …")
        mwi.build(wdir, layers, idir)
    man = json.loads((idir / "ddr_image.json").read_text())
    if sorted(man["layers"]) != sorted(layers):
        eprint(f"REFUSE: the image at {idir} holds layers {man['layers']}, "
               f"not {layers}. Re-run with --rebuild-image.")
        return 2
    eprint(f"[1] checking the image against the weight files …")
    if mwi.check(idir, wdir):
        eprint("REFUSE: the weight image does not re-derive from the tensors")
        return 2
    image_bytes = (idir / "ddr_image.bin").read_bytes()

    # [2] the real activations
    eprint(f"[2] golden prefill of {PROMPT_IDS} …")
    model = rt.GoldenModel(wdir)
    dm = int(model.meta["D_model"])
    assert int(model.meta["head_dim"]) == D_TILE_05B, \
        f"model head_dim {model.meta['head_dim']} != {D_TILE_05B}"
    tier = rt.TIER_MAP[args.tier]
    fxs = golden_prefill(model, PROMPT_IDS, tier, args.group)
    acts: dict[tuple[int, str], tuple[np.ndarray, list]] = {}
    for li in layers:
        for kind in ("h", "h2"):
            acts[(li, kind)] = c1_frames(activation_for(fxs[li], kind),
                                         D_TILE_05B)

    # [3] the jobs
    tensors = [t.strip() for t in args.tensors.split(",") if t.strip()]
    for t in tensors:
        if t not in ACT_OF:
            eprint(f"REFUSE: {t} is not a k={dm} projection this driver "
                   f"grades (choose from {sorted(ACT_OF)})")
            return 2
    jobs = []
    for li in layers:
        for t in tensors:
            for n0 in range(args.blocks):
                x, _sb = acts[(li, ACT_OF[t])]
                jobs.append(build_job(man, wdir, li, t, n0, x, work,
                                      image_bytes))
    eprint(f"[3] {len(jobs)} jobs emitted; every weight stream verified "
           f"byte-identical to its resident tensor block")

    ddr_args = (f"+ddr_image={idir / 'ddr_image.bin'}",
                f"+ddr_regions={idir / 'ddr_image.regions.jsonl'}",
                "+fuel_audit")

    # [4] control: host-fed (the weights go over BAR0, as today)
    eprint("[4] control run: HOST-fed (BAR0 mailbox WB) …")
    ctrl = run_and_grade(jobs, "host_path", binary, work, "host")

    # [5] the claim: fuel-fed from the resident image
    eprint("[5] FUEL run: weights from DDR …")
    fuel = run_and_grade(jobs, "fuel_path", binary, work, "fuel",
                         extra=ddr_args)

    # [6] the discriminator: ONE flipped DDR byte
    eprint("[6] discriminator: one flipped DDR byte …")
    bad_dir = work / "bad_image"
    bad_dir.mkdir(parents=True, exist_ok=True)
    mut = mutate_image(idir / "ddr_image.bin", bad_dir / "ddr_image.bin",
                       jobs[0])
    # the loader's per-region sha256 would catch the flip before the tile ever
    # ran, which would prove the LOADER works, not the FUEL path — so the
    # mutant run gets a regions file with that one region's digest updated to
    # the mutated bytes. Everything else is untouched.
    import hashlib
    with (bad_dir / "ddr_image.regions.jsonl").open("w") as f:
        badimg = (bad_dir / "ddr_image.bin").read_bytes()
        for ln in (idir / "ddr_image.regions.jsonl").read_text().splitlines():
            r = json.loads(ln)
            b0, nb = r["base_64B"] * 64, r["beats_64B"] * 64
            r["sha256"] = hashlib.sha256(badimg[b0:b0 + nb]).hexdigest()
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    bad_args = (f"+ddr_image={bad_dir / 'ddr_image.bin'}",
                f"+ddr_regions={bad_dir / 'ddr_image.regions.jsonl'}",
                "+fuel_audit")
    red = run_and_grade(jobs, "fuel_path", binary, work, "fuel_mutant",
                        extra=bad_args)
    # and the other half of the discriminator: the SAME mutated image loaded
    # while the jobs are HOST-fed must stay GREEN.
    eprint("[6b] mutant-image control: host-fed jobs, mutated DDR loaded …")
    red_ctrl = run_and_grade(jobs, "host_path", binary, work,
                             "host_mutant", extra=bad_args)

    return report(args, man, jobs, ctrl, fuel, red, red_ctrl, mut, binary,
                  idir, work, model)


# ═══════════════════════════════ the ledger ════════════════════════════════

def report(args, man, jobs, ctrl, fuel, red, red_ctrl, mut, binary, idir,
           work, model) -> int:
    W = 78
    gated = [r for r in man["tensors"] if r["gated"]]
    print("\n" + "=" * W)
    print("FUEL-FED PROJECTION GEMMs — Qwen2.5-0.5B WEIGHTS RESIDENT IN CARD "
          "MEMORY")
    print("=" * W)
    print(f"  model            : {man['model']}")
    print(f"  image            : {idir}  {man['image_bytes']} B "
          f"({man['image_bytes'] / 2**20:.1f} MiB), layers {man['layers']}, "
          f"{len(man['tensors'])} tensors ({len(gated)} weight + "
          f"{len(man['tensors']) - len(gated)} aux)")
    print(f"  image sha256     : {man['image_sha256']}")
    print(f"  twin             : {binary}  (DDR_PRESENT=1, D=64 GQA=2 DM=896 "
          f"QSTAGE=14 DMODEL=64)")
    print(f"  prompt           : ids {PROMPT_IDS} (last prefill step, "
          f"T={len(PROMPT_IDS)})")
    print(f"  activation       : golden quant_rows_i8 on {model.meta['D_model']}"
          f"/{D_TILE_05B} = {model.meta['D_model'] // D_TILE_05B} C-1 frames "
          f"(the CFG_DM=64 feeder framing) — WEIGHTS UNTOUCHED")
    print()
    print("  JOBS — every weight beat's address inside the resident tensor")
    print(f"  {'job':<22} {'k':>5} {'n':>3} {'DDR base':>12} {'words':>6} "
          f"{'tag':>4} {'@tensor+':>10}")
    for j in jobs:
        print(f"  {j['name']:<22} {j['k']:>5} {j['n']:>3} "
              f"0x{j['base_64B'] * 64:>10x} {j['beats_64B']:>6} "
              f"{j['tag']:>4} {j['byte_off_in_tensor']:>10}")
    print(f"  weight bytes per job: {jobs[0]['wire_bytes']} — every one "
          f"verified byte-identical to the resident tensor BEFORE the run")
    print()
    print("  RUNS (golden = compute.gemm_i8_ksplit on the same INT8 operands, "
          "computed after)")
    for r in (ctrl, fuel, red_ctrl, red):
        label = {"host": "control  HOST-fed (BAR0 WB)",
                 "fuel": "CLAIM    FUEL-fed (from DDR)",
                 "host_mutant": "control  HOST-fed + mutated DDR",
                 "fuel_mutant": "RED ARM  FUEL-fed + mutated DDR"}[r["tag"]]
        if "error" in r:
            print(f"    {label:<34} EXECUTOR ERROR: {r['error']}")
            continue
        n_ok = sum(1 for g in r["grades"] if g["bit_exact"])
        print(f"    {label:<34} rc={r['rc']} caps={r['n_caps']} "
              f"bit-exact {n_ok}/{len(r['grades'])}  "
              f"fuel_audit ok={r['fuel_audit_ok']} fail={r['fuel_audit_fail']}"
              f"  {r['wall']}s")
    print()
    print("  DISCRIMINATOR — one byte, named")
    print(f"    DDR byte 0x{mut['byte_offset']:x} (= {mut['job']} + "
          f"tensor byte {mut['tensor_byte']}): {mut['old']} -> {mut['new']}")
    print(f"    it is weight W[k={mut['k']}][lane {mut['lane']}] and the "
          f"activation there is {mut['x_k']}, so lane 0 MUST move by "
          f"{mut['delta_lane0']}")
    got = None
    if "error" not in red and red["grades"]:
        d = red["grades"][0]
        got = d["tile"][0] - d["golden"][0]
        print(f"    measured lane-0 move on the mutant run: {got}")
    print("-" * W)

    ok_ctrl = ("error" not in ctrl) and ctrl["all_bit_exact"] and ctrl["ok"]
    ok_fuel = ("error" not in fuel) and fuel["all_bit_exact"] and fuel["ok"] \
        and fuel["fuel_audit_ok"] == len(jobs) and fuel["fuel_audit_fail"] == 0
    ok_redctl = ("error" not in red_ctrl) and red_ctrl["all_bit_exact"]
    red_fired = ("error" not in red) and (not red["all_bit_exact"]) \
        and got == mut["delta_lane0"]
    verdict = ok_ctrl and ok_fuel and ok_redctl and red_fired
    print(f"  control HOST-fed bit-exact          : "
          f"{'PASS' if ok_ctrl else 'FAIL'}")
    print(f"  FUEL-fed bit-exact + FUEL_ERR clean : "
          f"{'PASS' if ok_fuel else 'FAIL'}")
    print(f"  mutated DDR + host-fed stays GREEN  : "
          f"{'PASS' if ok_redctl else 'FAIL'}   (the flip only matters when "
          f"the tile reads DDR)")
    print(f"  mutated DDR + FUEL-fed goes RED     : "
          f"{'PASS' if red_fired else 'FAIL'}   (by exactly the predicted "
          f"delta)")
    print(f"  FUEL PROJECTION GATE                : "
          f"{'PASS' if verdict else 'FAIL'}")
    print("=" * W)

    out = work / "fuel_proj_05b_result.json"
    out.write_text(json.dumps({
        "model": man["model"], "image": str(idir),
        "image_sha256": man["image_sha256"],
        "image_bytes": man["image_bytes"], "layers": man["layers"],
        "n_tensors": len(man["tensors"]), "binary": str(binary),
        "prompt_ids": PROMPT_IDS, "d": D_TILE_05B,
        "jobs": [{k: v for k, v in j.items()
                  if k not in ("x_codes", "W_block")} for j in jobs],
        "runs": {r["tag"]: {k: v for k, v in r.items() if k != "log"}
                 for r in (ctrl, fuel, red_ctrl, red)},
        "discriminator": mut,
        "measured_lane0_move": got,
        "verdict": {"control": ok_ctrl, "fuel": ok_fuel,
                    "mutant_host_green": ok_redctl,
                    "mutant_fuel_red": red_fired, "pass": verdict},
        "git": rt.git_rev(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if verdict else 1


# ══════════════════════════════ selftest ═══════════════════════════════════

def selftest() -> int:
    import struct
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("FUEL_PROJ_05B SELFTEST (offline: no executor, no card)")
    wdir = DEFAULT_WEIGHTS
    if not (wdir / "meta.json").exists():
        print(f"  SKIP: {wdir} absent")
        return 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idir, work = td / "img", td / "work"
        man = mwi.build(wdir, [0, 1], idir)
        img = (idir / "ddr_image.bin").read_bytes()

        # 1: the fuel record of a block lands inside its tensor's region
        r = mwi.tensor_record(man, 1, "Wq")
        b = mwi.block_record(man, 1, "Wq", 5, 0)
        chk("1 block record lies inside its tensor region",
            r["base_64B"] <= b["base_64B"]
            and b["base_64B"] + b["beats_64B"]
            <= r["base_64B"] + r["beats_64B"],
            f"tensor {r['base_64B']}+{r['beats_64B']}, block "
            f"{b['base_64B']}+{b['beats_64B']}")

        # 2: the resident bytes decode back to the real weight columns
        W = np.load(wdir / "L01_Wq.npy", mmap_mode="r")
        sl = img[b["base_64B"] * 64: (b["base_64B"] + b["beats_64B"]) * 64]
        beats = list(struct.unpack(f"<{len(sl) // 8}Q", sl))
        with tg.at_d(D_TILE_05B):
            back = gj._beats_to_block(beats, b["k"])
        chk("2 the resident bytes decode back to W[:, 40:48]",
            np.array_equal(back, np.asarray(W[:, 40:48], dtype=np.int64)))

        # 3: the real activation frames satisfy the MODE_QUANT identity
        model = rt.GoldenModel(wdir)
        fxs = golden_prefill(model, PROMPT_IDS, rt.TIER_MAP["kvq8"], 128)
        x, sb = c1_frames(activation_for(fxs[0], "h"))
        chk("3 every 64-wide C-1 frame of the real row has amax 127",
            x.shape[0] == model.meta["D_model"]
            and all(int(np.max(np.abs(x[i:i + 64]))) == 127
                    for i in range(0, x.shape[0], 64)),
            f"{len(sb)} frames")

        # 4: the emitted program's weight stream IS the resident bytes
        j = build_job(man, wdir, 0, "Wq", 3, c1_frames(
            activation_for(fxs[0], "h"))[0], work, img)
        chk("4 build_job's provenance check accepts the real block",
            j["wire_bytes"] == j["beats_64B"] * 64 == b["beats_64B"] * 64)

        # 5: the fuel program pushes ZERO projection weight beats (the
        #    activation-staging pushes remain, by design — see §2)
        ops = [json.loads(l) for l in
               Path(j["fuel_path"]).read_text().splitlines()]
        host_ops = [json.loads(l) for l in
                    Path(j["host_path"]).read_text().splitlines()]
        wb_f = sum(1 for o in ops if o.get("a") in (mdi.XW0, mdi.XW1, mdi.XWP))
        wb_h = sum(1 for o in host_ops
                   if o.get("a") in (mdi.XW0, mdi.XW1, mdi.XWP))
        chk("5 exactly the projection's weight beats left the program",
            wb_h - wb_f == 3 * j["k"]
            and wb_f == 3 * j["host_staging_beats_kept"],
            f"{wb_h // 3} -> {wb_f // 3} WB triples "
            f"({j['host_staging_beats_kept']} staging beats kept)")

        # 6: it carries exactly one fuel record, with this block's address
        ctrl = [o for o in ops if o.get("op") == "w"
                and o.get("a") in (t2f.FUEL_BASE, t2f.FUEL_BEATS, t2f.FUEL_CTRL)]
        base = [o["d"] for o in ctrl if o["a"] == t2f.FUEL_BASE]
        bts = [o["d"] for o in ctrl if o["a"] == t2f.FUEL_BEATS]
        chk("6 one fuel record naming the resident block",
            base == [j["base_64B"]] and bts == [j["beats_64B"]],
            f"base={base} beats={bts}")

        # 7: a provenance violation REFUSES (not warns)
        bad = bytearray(img)
        bad[j["base_64B"] * 64 + 3] ^= 0xFF
        try:
            build_job(man, wdir, 0, "Wq", 3,
                      c1_frames(activation_for(fxs[0], "h"))[0], work,
                      bytes(bad))
            chk("7 a byte-mismatched image is REFUSED", False, "no refusal")
        except SystemExit as e:
            chk("7 a byte-mismatched image is REFUSED", "REFUSE" in str(e))

        # 8: the discriminator's predicted delta is non-zero and derived
        m = mutate_image(idir / "ddr_image.bin", td / "bad.bin", j)
        chk("8 the flipped byte is predicted to move lane 0",
            m["delta_lane0"] != 0 and m["old"] != m["new"], str(m))

        # 9: the flipped byte really is inside this job's fuel region
        chk("9 the flipped byte lies inside the job's fuel record",
            j["base_64B"] * 64 <= m["byte_offset"]
            < (j["base_64B"] + j["beats_64B"]) * 64)

        # 10: golden is the plain integer product, from golden's own function
        blk = np.asarray(np.load(wdir / "L00_Wq.npy",
                                 mmap_mode="r")[:, 24:32], dtype=np.int64)
        xx = c1_frames(activation_for(fxs[0], "h"))[0]
        chk("10 golden == compute.gemm_i8_ksplit == einsum",
            np.array_equal(j["golden"],
                           [int(v) for v in np.einsum("k,kn->n", xx, blk)]))

    print("=" * 60)
    if fails:
        print(f"FUEL_PROJ_05B SELFTEST FAIL: {fails}")
        return 1
    print("FUEL_PROJ_05B SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Projection GEMMs fed from DDR-resident real 0.5B "
                    "weights, graded bit-exact against golden.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    r.add_argument("--image", default=str(DEFAULT_IMAGE))
    r.add_argument("--work", default=str(DEFAULT_WORK))
    r.add_argument("--layers", default="0,1")
    r.add_argument("--tensors", default="Wq,Wg",
                   help=f"k=D_model projections to fuel {sorted(ACT_OF)}")
    r.add_argument("--blocks", type=int, default=2,
                   help="how many 8-column n-blocks per tensor")
    r.add_argument("--obj", default="obj_b64_05b_ddr1")
    r.add_argument("--binary", default=None)
    r.add_argument("--tier", default="kvq8", choices=list(rt.TIER_MAP))
    r.add_argument("--group", type=int, default=128)
    r.add_argument("--rebuild-image", action="store_true")
    r.add_argument("--clean", action="store_true")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
