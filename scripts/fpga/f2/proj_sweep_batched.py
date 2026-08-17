#!/usr/bin/env python3
# proj_sweep_batched.py — the FULL projection sweep (all 1024 8-column blocks
# of Wq/Wk/Wv/Wo for one real decode step), executed BATCHED via
# batch_exec.run_jobs_batched.
#
# WHY THIS FILE EXISTS. breadth_step --full runs one executor invocation per
# tile job. On silicon the image-parity rule (rows_per_desc=1: the flying
# 05efb2a image predates the stage-buffer base-row change) makes one block
# ~28 K-split jobs, so --full is ~29k jobs ≈ 30 h at the measured ~3.6 s per
# separate invocation — the in-repo "2.19 h" figure (breadth_step selftest 9)
# assumed the SIM chunking of 2 jobs/block. Batched, the marginal in-batch
# job cost measured on silicon 2026-07-30 is ~0.2 s (attrib_proof.json), so
# the same sweep fits in ~2 h. This driver REUSES the verified primitives —
# breadth_step.capture_step / projection_specs / audit_program and
# gemm_job.stage_plan / build_gemm_job_full / decode_acc /
# accumulate_partials — and adds ONLY the two-phase orchestration
# (build-everything, then run in ~512-file batches with proven per-file
# attribution). Golden is computed by projection_specs BEFORE any hardware
# runs; the grade is bit-exact INT32 equality per block.
#
# Claim discipline: every job is produce-mode (requant_en=0 raw INT32 read
# back, zero output expectations — audited per file before running); the
# per-invocation clock gate of remote_hw_exec applies to EVERY batch.
#
# Sim proof (required before any hw session):
#   python3 scripts/fpga/f2/proj_sweep_batched.py --tensors Wv --blocks 0,63 \
#       --executor sim --cross-check
# The --cross-check flag additionally runs block 0 through the UNBATCHED
# gemm_job.ksplit_matvec and asserts identical INT32 accumulators.
#
# Full silicon sweep:
#   APEX_F2_HOST=ubuntu@<ip> APEX_F2_KEY=~/.ssh/apex-f2.pem \
#   python3 scripts/fpga/f2/proj_sweep_batched.py --executor hw --prune

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import breadth_step as bs                                      # noqa: E402
import batch_exec as bx                                        # noqa: E402
import gemm_job as gj                                          # noqa: E402
import cap_decode as cd                                        # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402
import remote_hw_exec                                          # noqa: E402

MXE_N = gj.MXE_N
D_TILE = gj.D_TILE


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def build_block_jobs(sp, bi, out_dir: Path, *, rows_per_desc: int):
    """All K-split jobs for one 8-column block. Returns [(path, ji)] in
    chunk order plus the block's golden INT32 slice."""
    Wfull = np.asarray(sp["W"])
    n0 = bi * MXE_N
    W = np.asarray(Wfull[:, n0:n0 + MXE_N], dtype=np.int64)
    plan = gj.stage_plan(np.asarray(sp["x8"], dtype=np.int64), W)
    jobs = []
    for ji, (r0, nr, K) in enumerate(plan.chunks(rows_per_desc)):
        a = r0 * D_TILE
        name = f"sw_{sp['tensor']}_n{n0:04d}_k{ji}"
        path, _man = gj.build_gemm_job_full(plan.Wst[a:a + K],
                                            plan.xst[a:a + K], out_dir, name)
        jobs.append((str(path), ji))
    gold = np.asarray(sp["acc"][n0:n0 + MXE_N], dtype=np.int64)
    return jobs, gold


def grade_block(file_results, gold):
    """file_results: demuxed per-file dicts of ONE block, chunk order.
    Same checks as gemm_job.ksplit_matvec, from the same primitives."""
    problems = []
    partials = []
    for fr in file_results:
        caps = fr["captures"]
        if not fr.get("complete") or not fr.get("attribution_ok"):
            problems.append(f"{Path(fr['path']).name}: incomplete/unattributed")
            continue
        meta = [c for c in caps if c["sem"] == "ro_meta"]
        if len(meta) != 1 or (meta[0]["value"] & 1) != 1:
            problems.append(f"{Path(fr['path']).name}: RO meta last not set")
        partials.append(gj.decode_acc(caps))
    if len(partials) != len(file_results):
        return {"equal": False, "problems": problems, "acc": None}
    acc = gj.accumulate_partials(partials)
    g = cd.grade(acc, gold)
    return {"equal": bool(g["equal"] and not problems),
            "problems": problems, "acc": [int(v) for v in acc]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tensors", default=None,
                    help="comma list (default: all four projections)")
    ap.add_argument("--blocks", default=None,
                    help="comma list of per-tensor block indices (default all)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--tier", default="kvq8", choices=list(bs.rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--weights-dir", default=str(bs.S8_WEIGHTS))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip jobs already recorded in <out>/batches.jsonl")
    ap.add_argument("--cross-check", action="store_true",
                    help="also run the first selected block through the "
                         "UNBATCHED ksplit_matvec and assert equal acc")
    args = ap.parse_args()
    remote_hw_exec.attach(bridge, args)      # env-gated, no-op for sim

    work = Path(args.out_dir or (bs.REPO / "build" / "hw_s2_sweep" / "sweep"))
    work.mkdir(parents=True, exist_ok=True)
    jobs_dir = work / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    # image parity: rows_per_desc=1 on hw (05efb2a predates 38ec95c) —
    # the same rule breadth_step.run_proj_blocks:450 enforces.
    rpd = 1 if args.executor == "hw" else gj.ROWS_PER_DESC

    model = bs.rt.GoldenModel(Path(args.weights_dir))
    run = json.loads(bs.S8_RUN.read_text())
    ids = [int(v) for v in run["prompt_ids"]]
    step = args.step if args.step is not None else len(ids) - 1
    eprint(f"[sweep] capture: layer {args.layer} step {step} (fast prefix)")
    t0 = time.time()
    capt = bs.capture_step(model, ids, layer=args.layer, step=step,
                           tier=bs.rt.TIER_MAP[args.tier], group=args.group,
                           fast_prefix=True)
    capture_s = time.time() - t0
    specs = bs.projection_specs(capt)
    for sp in specs:
        assert sp["recompose_ok"], f"{sp['tensor']}: spec recompose failed"
    want_t = set((args.tensors or "Wq,Wk,Wv,Wo").split(","))
    specs = [sp for sp in specs if sp["tensor"] in want_t]

    # ── phase 1: build everything, audit every program ─────────────────────
    t0 = time.time()
    blocks = []          # {tensor, bi, n0, paths[...], gold}
    n_jobs = 0
    audit_viol = 0
    for sp in specs:
        nb = np.asarray(sp["W"]).shape[1] // MXE_N
        picks = ([int(b) for b in args.blocks.split(",")] if args.blocks
                 else list(range(nb)))
        for bi in picks:
            jobs, gold = build_block_jobs(sp, bi, jobs_dir,
                                          rows_per_desc=rpd)
            for p, _ in jobs:
                audit_viol += bs.audit_program(
                    p, gj.RESULT_ADDRS)["output_expectations"]
            blocks.append({"tensor": sp["tensor"], "bi": bi,
                           "n0": bi * MXE_N,
                           "paths": [p for p, _ in jobs],
                           "gold": [int(v) for v in gold]})
            n_jobs += len(jobs)
        eprint(f"[sweep] built {sp['tensor']}: {len(picks)} blocks")
    build_s = time.time() - t0
    eprint(f"[sweep] {len(blocks)} blocks, {n_jobs} jobs, "
           f"{audit_viol} audit violations, build {build_s:.0f}s")
    if audit_viol:
        eprint("REFUSE: output expectations survived in generated programs")
        return 2

    # ── optional unbatched cross-check of the first selected block ─────────
    if args.cross_check:
        sp = specs[0]
        bi = blocks[0]["bi"]
        n0 = bi * MXE_N
        W = np.asarray(sp["W"])[:, n0:n0 + MXE_N]
        r = gj.ksplit_matvec(np.asarray(W, dtype=np.int64),
                             np.asarray(sp["x8"], dtype=np.int64),
                             out_dir=work / "xcheck", name="xchk",
                             executor=args.executor, binary=args.binary,
                             rows_per_desc=rpd, timeout_s=args.timeout_s,
                             verbose=False)
        blocks[0]["_xcheck_acc"] = [int(v) for v in r["acc_i32"]]
        eprint(f"[sweep] cross-check block ready (unbatched acc recorded, "
               f"ok={r['ok']})")

    # ── phase 2: run in batches, demux, persist, grade per block ───────────
    # Every batch appends its per-file results (captures included) to
    # batches.jsonl, so a killed sweep resumes with --resume instead of
    # re-spending the card. A batch that executed NOTHING (e.g. the
    # 2026-07-30 upload timeout: rc=124, 0/513 markers) aborts the sweep
    # loudly instead of burning hours of meter on a dead transport.
    done_file = work / "batches.jsonl"
    path2file = {}
    all_paths = [p for b in blocks for p in b["paths"]]
    if args.resume and done_file.exists():
        for line in done_file.read_text().splitlines():
            rec = json.loads(line)
            for f in rec["files"]:
                if f["complete"] and f["attribution_ok"]:
                    path2file[f["path"]] = f
        eprint(f"[sweep] resume: {len(path2file)} jobs already recorded")
    todo = [p for p in all_paths if p not in path2file]
    t0 = time.time()
    n_batches = 0
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        # One retry on transport exceptions (e.g. the 2026-07-30 mid-session
        # ISP IP rotation: the Mac fell out of the security group and the
        # cap fetch raised CaptureEgressError at batch 40/58). State is
        # persisted per batch, so even a second failure only costs a
        # --resume, never the sweep.
        try:
            r = bx.run_jobs_batched(chunk, executor=args.executor,
                                    binary=args.binary,
                                    timeout_s=args.timeout_s)
        except Exception as e:                                 # noqa: BLE001
            eprint(f"[sweep] batch raised {type(e).__name__}: {e} — "
                   f"retrying once in 30s")
            time.sleep(30)
            try:
                r = bx.run_jobs_batched(chunk, executor=args.executor,
                                        binary=args.binary,
                                        timeout_s=args.timeout_s)
            except Exception as e2:                            # noqa: BLE001
                eprint(f"[sweep] ABORT: retry also raised "
                       f"{type(e2).__name__}: {e2} — state persisted, rerun "
                       f"with --resume once the transport is fixed")
                return 3
        n_batches += 1
        ran_any = any(f["n_observed"] for f in r["files"])
        if not ran_any:
            # a clean nothing-ran return (e.g. transient scp rc=255) gets
            # the same single retry as an exception before we give up
            for n in r["batch"]["notes"]:
                eprint(f"[sweep]   exec note: {n}")
            eprint("[sweep] batch executed nothing — retrying once in 30s")
            time.sleep(30)
            r = bx.run_jobs_batched(chunk, executor=args.executor,
                                    binary=args.binary,
                                    timeout_s=args.timeout_s)
            ran_any = any(f["n_observed"] for f in r["files"])
        if not r["ok"]:
            for n in r["batch"]["notes"]:
                eprint(f"[sweep]   exec note: {n}")
            for n in r["notes"]:
                eprint(f"[sweep]   batch note: {n}")
        if not ran_any:
            eprint(f"[sweep] ABORT after batch {n_batches}: the batch "
                   f"executed NOTHING twice (transport failure) — refusing "
                   f"to continue; fix the transport and rerun with --resume")
            return 3
        with done_file.open("a") as fh:
            fh.write(json.dumps({
                "batch": n_batches, "ok": r["ok"],
                "wall_s": round(r["wall_s"], 2),
                "files": [{"path": f["path"], "captures": f["captures"],
                           "n_observed": f["n_observed"],
                           "n_expected": f["n_expected"],
                           "complete": f["complete"],
                           "attribution_ok": f["attribution_ok"]}
                          for f in r["files"]]}) + "\n")
        for f in r["files"]:
            path2file[f["path"]] = f
        done = len(path2file)
        eprint(f"[sweep] batch {n_batches}: {done}/{len(all_paths)} jobs, "
               f"batch ok={r['ok']} wall={r['wall_s']:.1f}s "
               f"({r['wall_s'] / max(1, len(chunk)):.2f}s/job)")
        if args.prune:
            for p in chunk:
                for suf in ("", ".manifest.json"):
                    try:
                        Path(p if not suf else
                             p.replace(".regops.jsonl", suf)).unlink()
                    except OSError:
                        pass
    run_s = time.time() - t0

    n_green = 0
    for b in blocks:
        res = grade_block([path2file[p] for p in b["paths"]],
                          np.asarray(b["gold"], dtype=np.int64))
        b["acc"] = res["acc"]
        b["equal"] = res["equal"]
        b["problems"] = res["problems"]
        b.pop("paths")
        if res["equal"]:
            n_green += 1
        if b.get("_xcheck_acc") is not None:
            b["xcheck_equal"] = (b["_xcheck_acc"] == b["acc"])
    xchk = [b for b in blocks if "_xcheck_acc" in b]

    ok = (n_green == len(blocks)) and all(b.get("xcheck_equal") for b in xchk)
    summary = {"executor": args.executor, "n_blocks": len(blocks),
               "n_green": n_green, "n_jobs": n_jobs,
               "n_batches": n_batches, "batch_size": args.batch_size,
               "rows_per_desc": rpd, "capture_s": round(capture_s, 1),
               "build_s": round(build_s, 1), "run_s": round(run_s, 1),
               "s_per_job": round(run_s / max(1, n_jobs), 3),
               "xcheck": ({"ran": bool(xchk),
                           "equal": all(b.get("xcheck_equal")
                                        for b in xchk)} if xchk
                          else {"ran": False}),
               "ok": ok, "blocks": blocks}
    out = work / "sweep_result.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"PROJ SWEEP ({args.executor}): blocks {n_green}/{len(blocks)} "
          f"bit-exact, {n_jobs} jobs in {n_batches} batches, "
          f"run {run_s / 60:.1f} min ({run_s / max(1, n_jobs):.2f}s/job) "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  record -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
