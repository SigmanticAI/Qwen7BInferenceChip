#!/usr/bin/env python3
# fatproof.py — the red/green for the FAT projection transport, on the b64
# silicon twin, plus the measured old-vs-new job-rate comparison.
#
#   python3 scripts/fpga/f2/fatproof.py --blocks 16          # ~2 min in sim
#   python3 scripts/fpga/f2/fatproof.py --blocks 4 --quick
#
# ══ WHAT IS BEING CLAIMED, AND WHAT WOULD FALSIFY IT ═══════════════════════
# The 2026-08-03 hw run (build/p05b_hw_check2.log) drove one 8-column block
# per tile program: 2,190 programs and 470 s of executor per layer. Three
# host-side changes shrink that WITHOUT changing a single descriptor field:
#
#   (1) FAT PROGRAMS — stage the activation ONCE per (row, K-chunk) and fire
#       one OP_GEMM_WS descriptor per column block against those same staged
#       rows (gemm_job.build_gemm_multiblock_job). The re-staging was ~62% of
#       every program's BAR0 ops.
#   (2) PUSH BURSTS — one "FIFO is empty" poll per DEPTH=16 pushes instead of
#       a free-slot PEEK before every push (gemm_job.BurstXlate). A peek is a
#       blocking PCIe round trip; a poke is posted.
#   (3) MULTI-ROW DESCRIPTORS, per image — rows_per_desc from tile_geom's
#       base-row (38ec95c) allowlist instead of the blanket
#       `1 if executor=='hw'` rule.
#
# Every one of those is a claim about TRANSPORT, so the thing that has to be
# proven is that the NUMBERS do not move. This file proves it by running the
# same operands three ways on the same twin and comparing INT32 accumulators,
# and it proves the proof can fail by running two arms that MUST go red:
#
#   RED-1  the same fat program emitted the way a PRE-38ec95c tile behaved
#          (every LOAD to base row 0) — must NOT match golden. If it did,
#          multi-row staging would be doing nothing and the allowlist would
#          be theatre.
#   RED-2  the mailbox drop audit — >DEPTH blind pushes into a FIFO nothing
#          is draining must FAIL the program. That audit is the net under
#          the burst, so a net that cannot fire is not a net.
#
# NO HARDWARE. Sim only, by construction: the executor is the verilated b64
# twin and there is no hw path in this file.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cap_decode as cd                                          # noqa: E402
import gemm_job as gj                                            # noqa: E402
import tile_exec_bridge as bridge                                # noqa: E402
import tile_geom as tg                                           # noqa: E402
import trace_to_regops as t2r                                    # noqa: E402
from apex_golden import attention as at                          # noqa: E402

W05B = REPO / "build/s8_weights/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_OUT = REPO / "build" / "fatproof"


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═══════════════════════ the operands (real 0.5B) ══════════════════════════

def real_operands(n_blocks: int, tensor: str = "Wq", layer: int = 0):
    """A real Qwen2.5-0.5B layer-0 projection: golden's own INT8 activation
    row and `n_blocks` 8-column slices of the real weight matrix.

    The activation is re-derived with golden's own three lines (the same
    provenance gemm_job.real_case documents for 7B), so both sides of every
    comparison below are the model's numbers, not a random draw.
    """
    emb = np.load(W05B / "embed.npy", mmap_mode="r")
    gamma1 = [int(v) for v in np.load(W05B / f"L{layer:02d}_gamma1.npy")]
    X = np.asarray(emb[9625], dtype=np.float64)[None, :]     # ' France'
    x8, _ = at.quant_rows_i8(X)
    h, _, _ = at.rmsnorm_fx_wide([int(v) for v in x8[0]], gamma1)
    h8m, _ = at.quant_rows_i8(np.asarray(h, dtype=np.float64)[None, :] / 256.)
    h8 = h8m[0].astype(np.int64)
    W = np.asarray(np.load(W05B / f"L{layer:02d}_{tensor}.npy",
                           mmap_mode="r")[:, :8 * n_blocks], dtype=np.int64)
    return h8, W, {"tensor": tensor, "layer": layer, "K": int(h8.shape[0]),
                   "blocks": n_blocks, "amax_code": int(np.max(np.abs(h8))),
                   "weights": str((W05B / f"L{layer:02d}_{tensor}.npy")
                                  .relative_to(REPO))}


def synth_operands(n_blocks: int, K: int = 896, seed: int = 20260804):
    rng = np.random.default_rng(seed)
    x8 = rng.integers(-100, 101, K, dtype=np.int64)
    x8[K // 3] = 127
    W = rng.integers(-9, 9, (K, 8 * n_blocks), dtype=np.int64)
    return x8, W, {"tensor": "synthetic", "layer": -1, "K": K,
                   "blocks": n_blocks, "amax_code": 127, "weights": "rng"}


# ═════════════════════════════ the executor ════════════════════════════════

class Twin:
    """The verilated b64 twin, with this lane's two per-program obligations
    (geometry audit + the disclosed INFO_TIER retarget) applied exactly as
    layer05b.Runner05B applies them — so what runs here is what the driver
    would run."""

    def __init__(self, binary: str, work: Path, d: int = 64,
                 image: tg.Image = tg.IMG_05B, timeout_s: int = 3600):
        self.binary, self.work, self.d = binary, Path(work), d
        self.image, self.timeout_s = image, timeout_s
        self.n_programs = 0
        self.n_caps = 0
        self.wall = 0.0

    def run(self, paths, tag: str, *, expect_ok: bool = True) -> list:
        paths = [str(p) for p in paths]
        for p in paths:
            tg.audit_geometry(p, self.d)
            tg.retarget_info_tier(p, self.image)
        out, t0 = [], time.perf_counter()
        for i, p in enumerate(paths):
            r = bridge.run_job(p, executor="sim", binary=self.binary,
                               cap_out=str(self.work / f"{tag}_{i:03d}.cap.jsonl"),
                               timeout_s=self.timeout_s)
            if expect_ok and not r["ok"]:
                raise RuntimeError(f"{tag}[{i}]: executor not ok rc={r['rc']} "
                                   f"notes={r['notes']}\n{r['log'][-1500:]}")
            self.n_caps += len(r["captures"])
            out.append(r)
        self.wall += time.perf_counter() - t0
        self.n_programs += len(paths)
        return out


def _cost(paths) -> dict:
    tot = {"ops": 0, "peeks": 0, "pokes": 0, "bytes": 0, "programs": 0}
    for p in paths:
        c = gj.program_cost(p)
        for k in ("ops", "peeks", "pokes", "bytes"):
            tot[k] += c[k]
        tot["programs"] += 1
    return tot


def _gz_bytes(paths) -> int:
    """What the wire actually carries after remote_hw_exec's gzipped tar."""
    import tarfile
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        name = f.name
    try:
        with tarfile.open(name, "w:gz") as t:
            for p in paths:
                t.add(str(p), arcname=Path(p).name)
        return Path(name).stat().st_size
    finally:
        Path(name).unlink(missing_ok=True)


# ══════════════════════════════ the arms ═══════════════════════════════════

def arm_thin(twin, plan, W, blocks, out, name, rows_per_desc):
    """The 2026-08-03 shape: one program per (block, K-chunk), per-push peeks."""
    paths, per_block = [], []
    for bi in range(blocks):
        Wst = plan.restage(W[:, 8 * bi:8 * bi + 8])
        mine = []
        for ci, (r0, nr, K) in enumerate(plan.chunks(rows_per_desc)):
            off = r0 * gj.D_TILE
            p, _ = gj.build_gemm_job_full(Wst[off:off + K],
                                          plan.xst[off:off + K], out,
                                          f"{name}_b{bi:03d}_k{ci}")
            mine.append(str(p))
        per_block.append(mine)
        paths.extend(mine)
    t0 = time.perf_counter()
    res = twin.run(paths, name)
    wall = time.perf_counter() - t0
    caps = {p: r["captures"] for p, r in zip(paths, res)}
    accs = [np.asarray(gj.accumulate_partials(
        [gj.decode_acc(caps[p]) for p in mine]), dtype=np.int64)
        for mine in per_block]
    return {"accs": accs, "wall_s": wall, "paths": paths, **_cost(paths)}


def arm_fat(twin, plan, W, blocks, out, name, rows_per_desc, *, burst=True,
            blocks_per_program=128, base_row_override=None, expect_ok=True):
    """One program per (K-chunk, group of blocks): staged once, swept."""
    chunks = plan.chunks(rows_per_desc)
    Wst = [plan.restage(W[:, 8 * bi:8 * bi + 8]) for bi in range(blocks)]
    paths, layout = [], []
    for ci, (r0, nr, K) in enumerate(chunks):
        off = r0 * gj.D_TILE
        for gi in range(0, blocks, blocks_per_program):
            grp = list(range(gi, min(gi + blocks_per_program, blocks)))
            p, man = gj.build_gemm_multiblock_job(
                [Wst[b][off:off + K] for b in grp], plan.xst[off:off + K],
                out, f"{name}_k{ci}_g{gi:03d}", burst=burst,
                load_base_row_override=base_row_override)
            paths.append(str(p))
            layout.append((grp, len(grp)))
    t0 = time.perf_counter()
    res = twin.run(paths, name, expect_ok=expect_ok)
    wall = time.perf_counter() - t0
    parts: dict = {}
    for (grp, nb), r in zip(layout, res):
        for b, acc in zip(grp, gj.decode_multiblock(r["captures"], nb)):
            parts.setdefault(b, []).append(acc)
    accs = [np.asarray(gj.accumulate_partials(parts[b]), dtype=np.int64)
            for b in range(blocks)]
    return {"accs": accs, "wall_s": wall, "paths": paths, **_cost(paths)}


def arm_drop_audit(twin, out: Path) -> dict:
    """RED-2: does the mailbox drop audit actually bite?

    Blind pushes into the xw FIFO with NO descriptor pushed — nothing is
    draining it — then a read of MB_STATUS expecting "no drop". At or below
    DEPTH the program must PASS; past it the mailbox drops the push, latches
    ovfl_in (apex_f2_mailbox.sv:258-263) and the read FAILS. That read is in
    the `DONE` tail of every program this lane emits, bursted or not.
    """
    rows = []
    for n, want_fail in ((gj.MB_FIFO_DEPTH, False),
                         (gj.MB_FIFO_DEPTH + 8, True)):
        ops = [{"op": "note", "s": f"MB drop probe: {n} blind xw pushes with "
                                   f"no descriptor — nothing drains the FIFO"}]
        for i in range(n):
            ops += [{"op": "w", "a": t2r.B_MB + t2r.XW0, "d": i},
                    {"op": "w", "a": t2r.B_MB + t2r.XW1, "d": 0},
                    {"op": "w", "a": t2r.B_MB + t2r.XWP, "d": 0}]
        ops.append({"op": "cap", "a": t2r.B_MB + t2r.XWP, "m": 0xFFFFFFFF,
                    "tag": "xw_free_0"})
        ops.append({"op": "r", "a": t2r.B_MB + t2r.MB_STATUS, "m": 0x3,
                    "e": 0x0})
        p = out / f"dropprobe_{n}.regops.jsonl"
        p.write_text("\n".join(json.dumps(o, separators=(",", ":"))
                               for o in ops) + "\n")
        r = bridge.run_job(str(p), executor="sim", binary=twin.binary,
                           cap_out=str(out / f"dropprobe_{n}.cap.jsonl"),
                           timeout_s=300, require_summary=False)
        fails = 0
        for ln in r["log"].splitlines():
            if " fails" in ln and ln.startswith("["):
                fails = int(ln.split("checks,")[1].split("fails")[0])
        free = [(c["value"] >> 9) & 0x7FFF for c in r["captures"]]
        rows.append({"blind_pushes": n, "fails": fails,
                     "free_slots_after": free, "want_fail": want_fail,
                     "ok": (fails > 0) == want_fail})
    return {"rows": rows, "pass": all(r["ok"] for r in rows),
            "depth": gj.MB_FIFO_DEPTH}


# ══════════════ the hw cost model (PREDICTION, fitted to 2 runs) ═══════════
#
# Nothing below is measured on hardware BY THIS FILE — no hw is touched here.
# It fits a 2-parameter transport model to the two committed silicon runs and
# then evaluates it on the emitted program set. Every input is cited; the
# output is labelled PREDICTION everywhere it is printed.
#
#   wall ≈ C_inv·invocations + a·programs + b·peeks
#
#   C_inv = 4.8 s   the batched-invocation setup measured on silicon
#                   (BATCHING_STUDY.md §e: "~4.8 s setup + N×0.21 s")
#   a               per PROGRAM inside a batch: f2_host_run.py:146-149 toggles
#                   TILE_RST with two 10 ms sleeps per file, plus the per-file
#                   json parse — and batch_exec interleaves one marker FILE
#                   per job, so a job costs TWO of those.
#   b               per PEEK: `pw`/`jf`/`r`/`poll` each block on a BAR0 read.
#
# The two equations:
#   S3 sweep     29,696 programs, 68 invocations, 36.0 min
#                (PROJ_SWEEP_RESULT.md, agfi-0ae06ea568e5667ba)
#   P2 stage 3   2,190 programs, 13 projection batches, 470.3 s of executor
#                for ONE layer (build/p05b_hw_check2.log, agfi-0ecab46b8a8376b21)
C_INV_S = 4.8
HW_POINTS = {
    "s3_sweep": {"programs": 29696, "invocations": 68, "seconds": 36.0 * 60,
                 "source": "docs/results/prompt_on_chip/PROJ_SWEEP_RESULT.md"},
    # 43 invocations = 13 projection batches (112/80/80/112/608/608/560
    # programs at --batch-size 256) + one per non-projection program (14
    # RoPE + 14 attention + 2 residual): layer_offload's wrappers need each
    # result before the next op, so each is its own runner.run().
    "p05b_layer": {"programs": 2190, "invocations": 43, "seconds": 470.28,
                   "source": "build/p05b_hw_check2.log (layer 0)"},
}


def fit_hw_model(peeks_s3: int, peeks_p05b: int) -> dict:
    """Solve the 2x2 for (a, b). Returns the fit and its residual shape."""
    rows = []
    for k, pk in (("s3_sweep", peeks_s3), ("p05b_layer", peeks_p05b)):
        p = HW_POINTS[k]
        rows.append((p["programs"], pk,
                     p["seconds"] - C_INV_S * p["invocations"]))
    (n1, k1, y1), (n2, k2, y2) = rows
    det = n1 * k2 - n2 * k1
    a = (y1 * k2 - y2 * k1) / det
    b = (n1 * y2 - n2 * y1) / det
    return {"per_program_s": a, "per_peek_s": b, "c_inv_s": C_INV_S,
            "inputs": {"s3_peeks": peeks_s3, "p05b_peeks": peeks_p05b},
            "points": HW_POINTS, "well_posed": a > 0 and b > 0}


def predict_wall(model: dict, *, programs: int, peeks: int,
                 invocations: int) -> float:
    return (model["c_inv_s"] * invocations + model["per_program_s"] * programs
            + model["per_peek_s"] * peeks)


# ═══════════════════════════════ the run ═══════════════════════════════════

def s3_reference_peeks(out: Path) -> dict:
    """Peeks in ONE job of the S3 sweep's class, emitted here and counted.

    The sweep flew `rows_per_desc=1` at D=128 on agfi-0ae06ea568e5667ba:
    K=128, one staged row, 29,696 of them. gemm_job.real_case() rebuilds the
    exact operand class (real Qwen2.5-7B Wq + golden's own activation row).
    """
    c = gj.real_case(gj.MXE_N, 0)
    plan = gj.stage_plan(c["x8"], c["W"])
    r0, nr, K = plan.chunks(1)[0]
    off = r0 * gj.D_TILE
    p, man = gj.build_gemm_job_full(plan.Wst[off:off + K],
                                    plan.xst[off:off + K], out, "s3_ref")
    cost = gj.program_cost(p)
    return {"k": K, "rows_per_desc": 1, "d": gj.D_TILE, **cost}


def _layer_cost(res: dict) -> dict:
    """Per-LAYER projection transport cost out of a prompt05b result JSON."""
    t = res["transport"]["projection_cost"]
    n_layers = max(len(res.get("layers_served") or res["layers"]), 1)
    per = {k: v / n_layers for k, v in t.items()}
    # the non-projection programs (rope/attn/resid) are one invocation each
    npj = sum(o["jobs"] for li, ops in res["per_layer_ops"].items()
              for o in ops if o["op"] != "proj" and o["n_served"]) / n_layers
    # EXECUTOR INVOCATIONS, from the run's own per-call program counts: one
    # runner.run() per projection call (batched at --batch-size) and one per
    # non-projection program, because its result is needed before the next op.
    bs = res["transport"]["batch_size"]
    inv = sum(int(np.ceil(o["jobs"] / bs))
              for li, ops in res["per_layer_ops"].items()
              for o in ops if o["op"] == "proj" and o["jobs"]) / n_layers
    out = {"programs": per["programs"], "peeks": per["peeks"],
           "ops": per["ops"], "bytes": per["bytes"], "blocks": per["blocks"],
           "nonproj_programs": npj, "n_layers": n_layers,
           "proj_invocations": inv,
           "fat": res["transport"]["fat"], "burst": res["transport"]["burst"],
           "rows_per_desc": res["transport"]["rows_per_desc"],
           "batch_size": res["transport"]["batch_size"]}
    # ── MEASURED transport, when the run recorded it ──────────────────────
    # Since the collapse (layer_offload.Runner's capture pool) the driver
    # counts EVERY entry into an executor and every program that entry
    # carried, per layer. A measured count always beats the derivation
    # above, which assumes the old "one invocation per non-projection op"
    # shape and would silently over-report a collapsed run.
    pl = res.get("per_layer") or {}
    m_inv = [b["invocations"] for b in pl.values()
             if isinstance(b, dict) and b.get("invocations")]
    m_prog = [b["jobs"] for b in pl.values()
              if isinstance(b, dict) and b.get("jobs")]
    m_pass = [b.get("passes", 1) for b in pl.values() if isinstance(b, dict)]
    out["collapse"] = bool(res["transport"].get("collapse"))
    out["measured_invocations"] = (sum(m_inv) / len(m_inv)) if m_inv else None
    out["measured_programs"] = (sum(m_prog) / len(m_prog)) if m_prog else None
    out["measured_passes"] = (max(m_pass) if m_pass else 1)
    return out


def _inv(c) -> int:
    """Executor entries per layer.

    The derivation below — one batch per projection call plus ONE ENTRY PER
    non-projection program — is what the pre-collapse transport actually did,
    and it is what the committed prediction was evaluated on. A COLLAPSED run
    breaks that assumption completely (the entries are per flush, not per op),
    so for those the MEASURED count the driver recorded is used instead.
    """
    if c.get("collapse") and c.get("measured_invocations"):
        return int(round(c["measured_invocations"]))
    return int(round(c["proj_invocations"] + c["nonproj_programs"]))


def _programs(c) -> float:
    """Programs an executor ran per layer.

    A replayed layer runs a few programs a later pass supersedes (the
    attention programs built on golden's RoPE row, before the tile's own C-1
    reconstruction existed). They cost real per-program and per-peek time, and
    the measured count includes them — so a collapsed run is charged for its
    own waste rather than credited with the smaller final program set.
    """
    if c.get("collapse") and c.get("measured_programs"):
        return float(c["measured_programs"])
    return c["programs"] + c["nonproj_programs"]


def predict(a) -> int:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path(a.old).read_text())
    new = json.loads(Path(a.new).read_text())
    lo, ln = _layer_cost(old), _layer_cost(new)
    s3 = s3_reference_peeks(out)
    peeks_s3 = s3["peeks"] * HW_POINTS["s3_sweep"]["programs"]
    peeks_p05b = lo["peeks"] + lo["nonproj_programs"] * s3["peeks"]
    model = fit_hw_model(int(peeks_s3), int(peeks_p05b))

    inv = _inv
    w_old = predict_wall(model, programs=_programs(lo),
                         peeks=peeks_p05b, invocations=inv(lo))
    # Programs the measured count carries beyond the records' own (proj +
    # non-proj) tally are superseded replay programs; they are charged the
    # same per-peek proxy as any other non-projection program.
    peeks_new = int(ln["peeks"] + ln["nonproj_programs"] * s3["peeks"]
                    + max(_programs(ln) - ln["programs"]
                          - ln["nonproj_programs"], 0) * s3["peeks"])
    w_new = predict_wall(model, programs=_programs(ln),
                         peeks=peeks_new, invocations=inv(ln))
    W = 78
    print("=" * W)
    print("PREDICTED HW PER-LAYER WALL — A PREDICTION, NOT A MEASUREMENT")
    print("=" * W)
    print("  No hardware was touched by this file. The model is fitted to the")
    print("  two committed silicon runs and evaluated on the program sets the")
    print("  sim runs above actually emitted.")
    print(f"  model : wall = {model['c_inv_s']}s x invocations "
          f"+ {model['per_program_s'] * 1e3:.1f} ms x programs "
          f"+ {model['per_peek_s'] * 1e6:.1f} us x PEEKS")
    for k, p in HW_POINTS.items():
        print(f"          fit point {k}: {p['programs']:,} programs, "
              f"{p['invocations']} invocations, {p['seconds']:.0f}s "
              f"({p['source']})")
    print(f"          S3-class reference program (emitted here): {s3['ops']:,}"
          f" ops / {s3['peeks']:,} peeks at D={s3['d']} k={s3['k']}")
    print(f"          well-posed (both coefficients positive): "
          f"{model['well_posed']}")
    print("-" * W)
    for nm, c, pk, w in (("OLD (measured shape)", lo, peeks_p05b, w_old),
                         ("NEW (this change)", ln, peeks_new, w_new)):
        print(f"  {nm:<22} {c['programs']:.0f} proj programs + "
              f"{c['nonproj_programs']:.0f} other, {inv(c)} invocations, "
              f"{pk:,.0f} peeks, {c['bytes'] / 1e6:.0f} MB")
        print(f"  {'':<22} -> PREDICTED {w:.0f} s per layer")
        if c.get("measured_invocations"):
            print(f"  {'':<22} (that run MEASURED "
                  f"{c['measured_invocations']:.0f} executor entries and "
                  f"{c['measured_programs']:.0f} programs per layer over "
                  f"{c['measured_passes']} replay pass(es); collapse="
                  f"{c['collapse']}"
                  + ("" if c["collapse"] else
                     " so the DERIVED counts above are used, exactly as the "
                     "committed prediction did") + ")")
    print("-" * W)
    print(f"  The OLD row is NOT a validation: two parameters fitted to two")
    print(f"  points reproduce both points by construction ({w_old:.0f} s vs "
          f"the {HW_POINTS['p05b_layer']['seconds']:.0f} s measured executor")
    print(f"  time it was fitted to; the measured per-layer WALL was 575-578 s"
          f", the")
    print(f"  difference being host-side emit+grade, which this change also "
          f"cuts).")
    print(f"  What IS checkable: the fitted "
          f"{model['per_peek_s'] * 1e6:.1f} us/peek is the same order as the")
    print(f"  ~0.8 ms/job MMIO figure in MASTER_TABLE.md:44-46 divided by a "
          f"job's peeks,")
    print(f"  and both fit points come from DIFFERENT images, drivers and "
          f"sessions.")
    print(f"  PREDICTED speedup on the hw per-layer wall: "
          f"{w_old / w_new:.1f}x  (PREDICTION)")
    print(f"  PREDICTED new per-layer wall: {w_new:.0f} s  (PREDICTION)")
    # ── where the predicted residual actually sits ────────────────────────
    parts = [("invocation setup", model["c_inv_s"] * inv(ln)),
             ("per-program (TILE_RST + parse + marker)",
              model["per_program_s"] * _programs(ln)),
             ("BAR0 peeks", model["per_peek_s"] * peeks_new)]
    print("  the PREDICTED 'new' wall, by term:")
    for nm, v in parts:
        print(f"      {nm:<42} {v:7.1f} s  ({100 * v / w_new:.0f}%)")
    if ln.get("collapse"):
        print(f"  {inv(ln)} invocations = one flush per layer REPLAY "
              f"({ln['measured_passes']} passes; the last one serves every "
              f"program from the")
        print("  capture pool and enters no executor at all), each flush "
              "size-bounded at")
        print("  --max-batch-mb. The per-head RoPE/attention and the residual "
              "windows no")
        print("  longer cost an entry each: they ride the same flush as the "
              "projections.")
    else:
        print(f"  {inv(ln)} invocations = {ln['proj_invocations']:.0f} "
              f"projection batches + {ln['nonproj_programs']:.0f} "
              f"non-projection programs (RoPE/attention/residual), each of "
              f"which layer_offload needs")
        print("  before it can call the next op — one executor entry apiece.")
    print("  SENSITIVITY to the one term this change attacks but cannot "
          "measure")
    print("  without a card — C_inv is 5 ssh/scp connections + remote python "
          "+ SDK")
    print("  attach; remote_hw_exec now multiplexes all 5 onto ONE ssh "
          "connection:")
    sens = {}
    for c in (model["c_inv_s"], 2.5, 1.5, 1.0):
        m2 = dict(model, c_inv_s=c)
        w2 = predict_wall(m2, programs=_programs(ln),
                          peeks=peeks_new, invocations=inv(ln))
        sens[c] = w2
        print(f"      C_inv = {c:4.1f} s  ->  {w2:6.1f} s per layer  "
              f"({w_old / w2:5.1f}x)"
              + ("   [MEASURED, pre-multiplexing]"
                 if c == model["c_inv_s"] else "   [PREDICTION]"))
    print("  Not modelled, and therefore NOT in the number above: the upload")
    print("  bytes (this change cuts them "
          f"{lo['bytes'] / max(ln['bytes'], 1):.1f}x before gzip and ~10x")
    print("  again with the gzipped tar), the ssh-multiplexing saving on the")
    print("  5 connections per invocation, and the host-side emit time. All")
    print("  three move the same way, so the prediction is a FLOOR on the")
    print("  improvement only if the fitted model is the whole story — which")
    print("  is exactly what the card session will decide.")
    rec = {"model": model, "s3_reference": s3, "old": lo, "new": ln,
           "peeks_old": peeks_p05b, "peeks_new": peeks_new,
           "predicted_old_s": w_old, "predicted_new_s": w_new,
           "predicted_speedup": w_old / w_new,
           "predicted_new_terms": {k: v for k, v in parts},
           "sensitivity_c_inv_s": {str(k): v for k, v in sens.items()},
           "invocations": {"old": inv(lo), "new": inv(ln)},
           "measured_old_s": 578.5,
           "labelled": "PREDICTION — fitted to committed hw runs, evaluated "
                       "on sim-emitted program sets; no hardware touched",
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    jp = Path(a.json) if a.json else out / "hw_prediction.json"
    jp.write_text(json.dumps(rec, indent=1, default=str))
    print(f"  record -> {jp}")
    print("=" * W)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", type=int, default=16,
                    help="8-column blocks to prove over")
    ap.add_argument("--tensor", default="Wq")
    ap.add_argument("--synthetic", action="store_true",
                    help="random INT8 operands instead of the 0.5B weights")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="skip the thin arm (no old-vs-new rate number)")
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--predict", action="store_true",
                    help="fit the hw transport model to the two committed "
                         "silicon runs and evaluate it on two prompt05b "
                         "result JSONs (--old/--new). PREDICTION, no hw.")
    ap.add_argument("--old", help="prompt05b_result.json from a --no-fat run")
    ap.add_argument("--new", help="prompt05b_result.json from a fat run")
    a = ap.parse_args()

    if a.predict:
        if not (a.old and a.new):
            ap.error("--predict needs --old and --new result JSONs")
        return predict(a)

    image = tg.IMG_05B
    binary = a.binary or str(image.binary)
    if not Path(binary).exists():
        eprint(f"REFUSE: the {image.name} twin is not built ({binary}).\n"
               f"  cd verif/f2sim && make build D={image.cfg_d} DDR=0 "
               f"OBJ={image.obj} VFLAGS_EXTRA=\"{image.defines()}\"")
        return 2
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fails: list = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    if a.synthetic or not W05B.is_dir():
        x8, W, meta = synth_operands(a.blocks)
    else:
        x8, W, meta = real_operands(a.blocks, a.tensor)
    d = image.cfg_d
    rpd_sim = tg.rows_per_desc_for(image, d, executor="sim")
    rpd_hw = tg.rows_per_desc_for(image, d, executor="hw")
    rpd_legacy = tg.rows_per_desc_for(tg.IMG_B128, 128, executor="hw",
                                      agfi="agfi-0ae06ea568e5667ba")

    print("=" * 78)
    print("FAT PROJECTION TRANSPORT — RED/GREEN ON THE b64 TWIN (SIM ONLY)")
    print("=" * 78)
    print(f"  operands        : {meta['tensor']} L{meta['layer']:02d}, K="
          f"{meta['K']}, {meta['blocks']} x 8 columns  ({meta['weights']})")
    print(f"  image           : {image.name} / {image.agfi}  CFG_D={d}")
    print(f"  twin            : {binary}")
    print(f"  rows_per_desc   : sim {rpd_sim}, this image on hw {rpd_hw}, "
          f"the LEGACY agfi-0ae06ea568e5667ba {rpd_legacy}")
    print(f"                    {tg.rows_per_desc_note(image, d, executor='hw')}")
    print("-" * 78)

    twin = Twin(binary, out, d=d, image=image, timeout_s=a.timeout_s)
    rec: dict = {"meta": meta, "image": image.name, "agfi": image.agfi,
                 "rows_per_desc": {"sim": rpd_sim, "hw": rpd_hw,
                                   "legacy_agfi": rpd_legacy},
                 "binary": binary, "arms": {}}

    with tg.at_d(d):
        amax = int(np.max(np.abs(x8)))
        xs, Wa = x8, W
        if amax != 127:
            xs = np.concatenate([x8, [127]])
            Wa = np.concatenate([W, np.zeros((1, W.shape[1]), dtype=np.int64)])
        plan = gj.stage_plan(xs, Wa[:, :8])
        gold = [np.einsum("k,kn->n", x8, W[:, 8 * b:8 * b + 8])
                for b in range(a.blocks)]

        # ── GREEN: fat + burst, the shipping shape ─────────────────────────
        fat = arm_fat(twin, plan, Wa, a.blocks, out, "fat", rpd_sim)
        eq_fat = all(np.array_equal(x, g) for x, g in zip(fat["accs"], gold))
        chk(f"FAT+BURST: {a.blocks}/{a.blocks} blocks bit-exact vs golden",
            eq_fat, f"{fat['programs']} program(s), {fat['ops']:,} BAR0 ops, "
                    f"{fat['peeks']:,} peeks, {fat['wall_s']:.1f}s in the twin")
        rec["arms"]["fat_burst"] = {k: v for k, v in fat.items()
                                    if k not in ("accs", "paths")}

        # ── GREEN: fat WITHOUT the burst — isolates the two changes ────────
        fnb = arm_fat(twin, plan, Wa, a.blocks, out, "fatnb", rpd_sim,
                      burst=False)
        chk("FAT without the push burst gives the SAME accumulators "
            "(the burst is a transport change, not a numeric one)",
            all(np.array_equal(x, y) for x, y in zip(fat["accs"], fnb["accs"])),
            f"peeks {fnb['peeks']:,} -> {fat['peeks']:,} "
            f"({fnb['peeks'] / max(fat['peeks'], 1):.1f}x fewer)")
        rec["arms"]["fat_nonburst"] = {k: v for k, v in fnb.items()
                                       if k not in ("accs", "paths")}

        # ── GREEN: the THIN path — the old shape, same operands ────────────
        thin = None
        if not a.quick:
            thin = arm_thin(twin, plan, Wa, a.blocks, out, "thin",
                            gj.ROWS_PER_DESC)
            chk("THIN (one program per block, per-push peeks) gives the SAME "
                "accumulators — fat is a pure transport change",
                all(np.array_equal(x, y)
                    for x, y in zip(fat["accs"], thin["accs"])),
                f"{thin['programs']} programs, {thin['ops']:,} BAR0 ops, "
                f"{thin['peeks']:,} peeks, {thin['wall_s']:.1f}s")
            rec["arms"]["thin"] = {k: v for k, v in thin.items()
                                   if k not in ("accs", "paths")}

        # ── RED-1: the pre-38ec95c base row ───────────────────────────────
        red = arm_fat(twin, plan, Wa, a.blocks, out, "redbase", rpd_sim,
                      base_row_override=0, expect_ok=True)
        n_same = sum(1 for x, g in zip(red["accs"], gold)
                     if np.array_equal(x, g))
        chk("RED-1 a PRE-38ec95c base row (every LOAD -> row 0) does NOT "
            "reproduce golden — the multi-row staging is load-bearing",
            n_same == 0, f"{n_same}/{a.blocks} blocks matched (want 0); "
                         f"first block off by "
                         f"{int(np.max(np.abs(red['accs'][0] - gold[0])))}")
        rec["arms"]["red_base_row"] = {"blocks_matching_golden": n_same,
                                       "max_abs_delta_block0":
                                           int(np.max(np.abs(red["accs"][0]
                                                             - gold[0])))}

        # ── RED-2: the drop audit under the burst ─────────────────────────
        drop = arm_drop_audit(twin, out)
        chk("RED-2 the mailbox drop audit BITES past the FIFO depth (and not "
            "at or below it) — the net under the burst is live",
            drop["pass"],
            "; ".join(f"{r['blind_pushes']} pushes -> {r['fails']} fails "
                      f"(free {r['free_slots_after']})" for r in drop["rows"]))
        rec["arms"]["red_drop_audit"] = drop

        # ── the compile-time burst invariant, over every fat program ──────
        worst = []
        for p in fat["paths"]:
            ops = [json.loads(ln) for ln in Path(p).read_text().splitlines()
                   if ln.strip()]
            worst.append(gj.audit_push_bursts(ops))
        chk("every emitted fat program passes the compile-time burst walk "
            "(no blind push can outrun a FIFO-empty poll)",
            bool(worst) and all(w["blind_pushes"] > 0 for w in worst),
            f"{sum(w['empty_polls'] for w in worst):,} empty-polls cover "
            f"{sum(w['blind_pushes'] for w in worst):,} blind pushes at "
            f"depth {gj.MB_FIFO_DEPTH}")

    # ── the rate comparison ───────────────────────────────────────────────
    print("-" * 78)
    print("  MEASURED, SAME BLOCK SET, SAME TWIN (sim wall is NOT the hw wall)")
    print(f"  {'arm':<16} {'progs':>6} {'BAR0 ops':>12} {'peeks':>10} "
          f"{'MB regops':>10} {'gz MB':>7} {'sim s':>8}")
    arms = [("thin (old)", thin), ("fat, no burst", fnb), ("fat+burst", fat)]
    for nm, arm in arms:
        if arm is None:
            continue
        gzb = _gz_bytes(arm["paths"])
        arm["gz_bytes"] = gzb
        if nm == "fat+burst":
            rec["arms"]["fat_burst"]["gz_bytes"] = gzb
        elif nm == "fat, no burst":
            rec["arms"]["fat_nonburst"]["gz_bytes"] = gzb
        elif thin is not None:
            rec["arms"]["thin"]["gz_bytes"] = gzb
        print(f"  {nm:<16} {arm['programs']:>6} {arm['ops']:>12,} "
              f"{arm['peeks']:>10,} {arm['bytes'] / 1e6:>10.1f} "
              f"{gzb / 1e6:>7.2f} {arm['wall_s']:>8.1f}")
    if thin is not None:
        r = {"programs": thin["programs"] / fat["programs"],
             "ops": thin["ops"] / fat["ops"],
             "peeks": thin["peeks"] / fat["peeks"],
             "bytes": thin["bytes"] / fat["bytes"],
             "gz_bytes": thin["gz_bytes"] / fat["gz_bytes"],
             "sim_wall": thin["wall_s"] / fat["wall_s"]}
        rec["ratios_thin_over_fat"] = r
        print(f"  {'RATIO old/new':<16} {r['programs']:>6.1f}x "
              f"{r['ops']:>11.2f}x {r['peeks']:>9.1f}x {r['bytes']:>9.2f}x "
              f"{r['gz_bytes']:>6.2f}x {r['sim_wall']:>7.2f}x")
        print(f"  gzip on the emitted programs: fat "
              f"{fat['bytes'] / fat['gz_bytes']:.1f}x, thin "
              f"{thin['bytes'] / thin['gz_bytes']:.1f}x "
              f"(remote_hw_exec now ships a gzipped tar)")

    rec["verdict"] = "PASS" if not fails else "FAIL"
    rec["failed"] = fails
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        rec["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=str(REPO), capture_output=True,
                                    text=True).stdout.strip()
    except Exception:
        rec["git"] = None
    jp = Path(a.json) if a.json else out / "fatproof_result.json"
    jp.write_text(json.dumps(rec, indent=1, default=str))
    print("-" * 78)
    print(f"  twin totals     : {twin.n_programs} programs, {twin.n_caps} "
          f"captures, {twin.wall:.1f}s")
    print(f"  record          -> {jp}")
    print(f"FATPROOF: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
