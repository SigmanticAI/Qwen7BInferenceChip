#!/usr/bin/env python3
# batch_exec.py — submit N regops jobs in ONE executor invocation and
# demultiplex the capture egress back to per-file results.
#
# ══ WHY (the measured economics this file attacks) ══════════════════════════
# On silicon one job costs ~3.8 s wall of which only ~0.78 ms is real MMIO
# (docs/design/MASTER_TABLE.md:44-46; per-job walls: docs/results/
# prompt_on_chip/hw_breadth_result.json projections[].job_wall_s, median
# 3.73 s over 232 jobs). ~99.98% of a job is therefore per-INVOCATION
# overhead — ssh handshakes, remote python start, SDK attach, our own
# per-call setup — not the tile. Both executors ALREADY take many regops
# files per invocation (f2_host_run.py:78 `nargs="+"`; sim_main.cpp argv
# loop) and both isolate state per file (f2_host_run.py:146 toggles
# TILE_RST 0x000C before EVERY file; sim_main.cpp calls reset() before every
# file). So N jobs in one invocation is contractually safe — what was
# missing is (1) a caller that does it and (2) a CORRECT way to attribute
# the merged capture stream back to its source files.
#
# ══ ATTRIBUTION (the part that needs care) ══════════════════════════════════
# The cap records carry only `tag` (unique PER JOB, not across jobs — every
# emitter restarts its `<sem>_<seq>` counter at 0) and a global monotonic
# `i`. Counting caps per file from each program almost works, but breaks
# silently the moment any file aborts mid-run (a stall executes fewer caps
# than programmed and every later file inherits the shift). This file uses
# BOTH mechanisms, so an abort is detected instead of mis-attributed:
#
#   1. BOUNDARY MARKERS: a one-line regops file is interleaved before every
#      job and once after the last —
#          {"op":"cap","a":0,"m":4294967295,"tag":"__bm_<nonce>_<k>__"}
#      BAR0 0x0000 is the bridge ID register (reads 0x41394558 "A9EX",
#      scripts/fpga/f2/host_smoke.py:9) — read-only, side-effect-free, and
#      present in both executors. The markers are SEPARATE files, so the
#      job files themselves stay byte-identical to their canonical
#      artifacts (evidence integrity), at the cost of one extra per-file
#      reset each (~cheap: the marker file executes in ~330 sim cycles).
#      Captures between marker k and marker k+1 belong to file k, whatever
#      happened inside it.
#   2. PER-FILE MANIFESTS: each job file is parsed for its programmed `cap`
#      ops (ordered tag/addr/mask). After demux, segment k must equal its
#      manifest exactly (complete) or be a strict prefix of it (aborted →
#      flagged partial, never silently attributed). Any other shape is an
#      attribution FAILURE and the batch result says so.
#
# NOTE on tags: across a batch, tags DUPLICATE by design (per-job counters).
# tile_exec_bridge.audit_captures() will therefore emit its duplicate-tag
# note on the merged stream — expected in batch mode; the demuxed per-file
# segments restore uniqueness and are re-audited individually here.
#
# ══ WHAT THIS FILE DOES NOT DO ══════════════════════════════════════════════
# It never talks to hardware itself: it drives tile_exec_bridge.run_job()
# (or any injected run_job-shaped callable, e.g. the remote_hw_exec attach()
# shim), so the sim and hw paths inherit every existing paranoia check
# (summary line required, stale cap files rejected, n cross-checked).
#
# CLI:
#   python3 batch_exec.py --selftest                 # fake executor, no build
#   python3 batch_exec.py --prove  [--binary F2SIM]  # REAL sim attribution proof
#   python3 batch_exec.py --bench 1,4,16,64 [--binary F2SIM]   # timing study
#   python3 batch_exec.py job1.jsonl job2.jsonl ... [--executor sim|hw]

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tile_exec_bridge as bridge                              # noqa: E402
from tile_exec_bridge import audit_captures                    # noqa: E402

REPO = bridge.REPO

MARKER_ADDR = 0x0000            # cl_apex bridge ID register — RO, no side effects
MARKER_MASK = 0xFFFFFFFF
BRIDGE_ID = 0x41394558          # "A9EX" (host_smoke.py:18)
MARKER_STEM = "__bm"


class BatchAttributionError(RuntimeError):
    """The merged capture stream could not be attributed to its files."""


# ── markers ─────────────────────────────────────────────────────────────────
def _marker_tag(nonce: str, k: int) -> str:
    return f"{MARKER_STEM}_{nonce}_{k}__"


def _parse_marker_tag(tag: str, nonce: str):
    """Return k for this batch's marker tags, else None."""
    pre = f"{MARKER_STEM}_{nonce}_"
    if tag.startswith(pre) and tag.endswith("__"):
        mid = tag[len(pre):-2]
        if mid.isdigit():
            return int(mid)
    return None


def _jline(d: dict) -> str:
    """COMPACT separators are the emitter contract: sim_main.cpp's flat-JSON
    extractor matches the literal pattern `"key":` — a space after the colon
    (json.dumps default) makes it SKIP the line silently (2 counted ops, 0
    executed). trace_to_regops.py emits compact; so must we."""
    return json.dumps(d, separators=(",", ":"))


def write_marker_file(workdir: Path, nonce: str, k: int) -> Path:
    p = Path(workdir) / f"bm_{nonce}_{k:04d}.regops.jsonl"
    rec = {"op": "note",
           "s": f"batch_exec boundary marker {k} (nonce {nonce}) — the cap "
                f"below reads the RO bridge ID at 0x0000; it belongs to the "
                f"BATCHING MACHINERY, never to a job"}
    cap = {"op": "cap", "a": MARKER_ADDR, "m": MARKER_MASK,
           "tag": _marker_tag(nonce, k)}
    p.write_text(_jline(rec) + "\n" + _jline(cap) + "\n")
    return p


# ── manifests ───────────────────────────────────────────────────────────────
CAP_MARK = '"op":"cap"'          # trace_to_regops emits COMPACT separators


def cap_manifest(path) -> list[tuple]:
    """Ordered (tag, addr, mask) of every `cap` op programmed in a regops
    file. Mask default when absent mirrors the executors: 0
    (f2_host_run.py:209 'sim default when absent: 0').

    The `CAP_MARK` pre-filter is a pure speed guard, not a semantic one: a
    FAT projection program is tens of MB of `w`/`pw` lines around ~1k caps,
    and json.loads on every line costs more than the executor spends on the
    file. The marker is the emitter's own compact spelling (_jline /
    trace_to_regops both use separators=(",", ":")); any line that could be
    a cap contains it verbatim, and every line that contains it is still
    parsed and validated below.
    """
    out = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or CAP_MARK not in line:
            continue
        d = json.loads(line)
        if d.get("op") == "cap":
            if "tag" not in d:
                raise BatchAttributionError(
                    f"{path}:{lineno}: cap without tag — the executors abort "
                    f"the file here; refusing to batch it")
            out.append((str(d["tag"]), int(d.get("a", 0)), int(d.get("m", 0))))
    return out


# ── planning ────────────────────────────────────────────────────────────────
def plan_batch(paths, workdir=None) -> dict:
    """Interleave boundary-marker files around the job files.

    Returns {'exec_paths': [m0, p0, m1, p1, …, mN], 'paths': […],
             'nonce': str, 'manifests': [ [(tag,addr,mask),…] per file ],
             'workdir': str}.
    Job files are passed through UNTOUCHED (byte-identical artifacts).
    """
    paths = [str(Path(p)) for p in paths]
    if not paths:
        raise ValueError("no regops files given")
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"regops file not found: {p}")
    names = [Path(p).name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError(f"regops basenames collide (the hw executor rejects "
                         f"them on the remote dir): {names}")
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="apex_batch_")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex[:8]
    manifests = [cap_manifest(p) for p in paths]
    # a job whose OWN tags look like this batch's markers would corrupt the
    # demux — the nonce makes that astronomically unlikely, but check anyway
    for p, man in zip(paths, manifests):
        clash = [t for (t, _a, _m) in man
                 if _parse_marker_tag(t, nonce) is not None]
        if clash:
            raise BatchAttributionError(f"{p} programs marker-shaped tags "
                                        f"{clash} — refusing")
    exec_paths = []
    for k, p in enumerate(paths):
        exec_paths.append(str(write_marker_file(workdir, nonce, k)))
        exec_paths.append(p)
    exec_paths.append(str(write_marker_file(workdir, nonce, len(paths))))
    return {"exec_paths": exec_paths, "paths": paths, "nonce": nonce,
            "manifests": manifests, "workdir": str(workdir)}


# ── demux ───────────────────────────────────────────────────────────────────
def demux_captures(caps: list[dict], plan: dict) -> dict:
    """Split the merged capture stream into per-file results and PROVE the
    attribution against each file's programmed manifest.

    Returns {'files': [ {path, captures, n_expected, n_observed, complete,
                         attribution_ok, notes} ], 'markers_seen': int,
             'ok': bool, 'notes': [batch-level notes]}
    Raises BatchAttributionError only for streams that cannot be split at
    all (orphan captures / out-of-order markers); per-file shortfalls are
    REPORTED, not raised, so a partial batch is still inspectable.
    """
    nonce, paths, manifests = plan["nonce"], plan["paths"], plan["manifests"]
    n_files = len(paths)
    segments = [[] for _ in range(n_files)]
    notes: list = []
    cur = None                     # index of the file currently accumulating
    next_marker = 0
    markers_seen = 0
    for c in sorted(caps, key=lambda c: c["i"]):
        k = _parse_marker_tag(c["tag"], nonce)
        if k is not None:
            if k != next_marker:
                raise BatchAttributionError(
                    f"marker {k} arrived when {next_marker} was expected — "
                    f"the executor ran files out of order or dropped one; "
                    f"attribution impossible")
            if c["value"] != (BRIDGE_ID & MARKER_MASK):
                notes.append(f"marker {k} read {c['value']:#010x}, expected "
                             f"bridge ID {BRIDGE_ID:#010x} — different CL or "
                             f"broken bridge; values may still attribute but "
                             f"treat this run with suspicion")
            markers_seen += 1
            next_marker += 1
            cur = k if k < n_files else None
        else:
            if cur is None:
                raise BatchAttributionError(
                    f"capture {c['tag']!r} (i={c['i']}) arrived outside any "
                    f"file window (before the first marker or after the "
                    f"terminal one) — attribution impossible")
            segments[cur].append(c)

    if markers_seen != n_files + 1:
        notes.append(f"only {markers_seen}/{n_files + 1} boundary markers "
                     f"executed — the batch stopped early; files after "
                     f"marker {markers_seen - 1} never ran")

    files = []
    all_ok = markers_seen == n_files + 1
    for k, (path, man, seg) in enumerate(zip(paths, manifests, segments)):
        got = [(c["tag"], c["addr"], c["mask"]) for c in seg]
        fnotes = []
        if got == man:
            complete, attribution_ok = True, True
        elif len(got) < len(man) and got == man[:len(got)]:
            complete, attribution_ok = False, True
            fnotes.append(f"PARTIAL: {len(got)}/{len(man)} programmed caps "
                          f"executed — the file aborted mid-run (stall/"
                          f"unknown op); trailing caps are missing, the ones "
                          f"present are correctly attributed")
        else:
            complete, attribution_ok, all_ok = False, False, False
            first_bad = next((i for i, (g, m) in enumerate(zip(got, man))
                              if g != m), min(len(got), len(man)))
            fnotes.append(f"ATTRIBUTION FAILURE: observed caps do not match "
                          f"the programmed manifest (first divergence at "
                          f"cap {first_bad}: got "
                          f"{got[first_bad] if first_bad < len(got) else None} "
                          f"want "
                          f"{man[first_bad] if first_bad < len(man) else None})"
                          )
        if not complete:
            all_ok = False
        fnotes += audit_captures(seg)      # per-file audit (tags unique again)
        files.append({"path": path, "captures": seg, "n_expected": len(man),
                      "n_observed": len(seg), "complete": complete,
                      "attribution_ok": attribution_ok, "notes": fnotes})
    return {"files": files, "markers_seen": markers_seen, "ok": all_ok,
            "notes": notes}


# ── the batched runner ──────────────────────────────────────────────────────
def run_jobs_batched(paths, executor: str = "sim", binary=None, cap_out=None,
                     timeout_s: int = 1800, *,
                     tile_div: int = bridge.TILE_DIV_DEFAULT, slot: int = 0,
                     extra_args=(), cwd=None, env=None,
                     require_summary: bool = True, workdir=None,
                     run_job_fn=None) -> dict:
    """Execute N regops files in ONE executor invocation; return per-file
    results with proven attribution.

    run_job_fn defaults to tile_exec_bridge.run_job — which is exactly what
    remote_hw_exec.attach() shims, so `executor='hw'` batches over ssh with
    ONE clock probe / mkdir / upload / python start / SDK attach for the
    whole batch instead of one per job. (Sim-only sessions never touch hw.)

    Returns {'batch': <run_job result dict, verbatim>,
             'files': <demux 'files' list>, 'ok': bool, 'nonce': str,
             'notes': [...], 'n_files': int, 'wall_s': float}
    `ok` = the executor invocation was ok AND every file's captures were
    completely and provably attributed.
    """
    plan = plan_batch(paths, workdir=workdir)
    fn = run_job_fn or bridge.run_job
    t0 = time.perf_counter()
    res = fn(plan["exec_paths"], executor=executor, binary=binary,
             cap_out=cap_out, timeout_s=timeout_s, tile_div=tile_div,
             slot=slot, extra_args=extra_args, cwd=cwd, env=env,
             require_summary=require_summary)
    wall = time.perf_counter() - t0
    dm = demux_captures(res["captures"], plan)
    notes = list(dm["notes"])
    # the merged-stream duplicate-tag note is EXPECTED in batch mode: explain
    # it rather than letting it read like a defect.
    dup = [n for n in res["notes"] if "duplicate tag" in n]
    if dup:
        notes.append("merged-stream duplicate tags are EXPECTED in a batch "
                     "(per-job counters restart); per-file segments are "
                     "re-audited individually above")
    return {"batch": res, "files": dm["files"],
            "ok": bool(res["ok"]) and dm["ok"], "nonce": plan["nonce"],
            "notes": notes, "n_files": len(plan["paths"]), "wall_s": wall}


def run_jobs_separate(paths, executor: str = "sim", binary=None,
                      timeout_s: int = 1800, *,
                      tile_div: int = bridge.TILE_DIV_DEFAULT, slot: int = 0,
                      workdir=None, run_job_fn=None) -> dict:
    """The baseline: one executor invocation PER file (what every caller
    does today). Same return granularity as run_jobs_batched for diffing."""
    fn = run_job_fn or bridge.run_job
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="apex_sep_")
    files, walls = [], []
    ok = True
    t0 = time.perf_counter()
    for k, p in enumerate(paths):
        co = Path(workdir) / f"sep_{k:04d}.cap.jsonl"
        t1 = time.perf_counter()
        r = fn([str(p)], executor=executor, binary=binary, cap_out=str(co),
               timeout_s=timeout_s, tile_div=tile_div, slot=slot)
        w = time.perf_counter() - t1
        walls.append(w)
        ok = ok and r["ok"]
        files.append({"path": str(p), "captures": r["captures"],
                      "ok": r["ok"], "wall_s": w})
    return {"files": files, "ok": ok, "wall_s": time.perf_counter() - t0,
            "per_file_wall_s": walls}


def captures_equal(a: list[dict], b: list[dict]) -> bool:
    """Same captured stream up to the global `i` renumbering."""
    ka = [(c["tag"], c["addr"], c["mask"], c["value"]) for c in a]
    kb = [(c["tag"], c["addr"], c["mask"], c["value"]) for c in b]
    return ka == kb


# ═════════════════ selftest (fake executor — no Verilator) ══════════════════
# Unlike tile_exec_bridge._FAKE (which ignores its regops), this fake
# INTERPRETS the files: note/w/cap against a register map with the bridge ID
# at 0x0000 — so markers, per-file values and aborts behave like the real
# executors. APEX_BFAKE_ABORT="<basename>:<n>" stops that file after n caps.
_FAKE = r'''
import json, os, sys
cap = None
files = []
for i, a in enumerate(sys.argv[1:], 1):
    if a.startswith("+cap_out="): cap = a.split("=", 1)[1]
    elif not a.startswith("+"): files.append(a)
abort = os.environ.get("APEX_BFAKE_ABORT", "")
ab_name, ab_n = (abort.split(":", 1) + ["0"])[:2] if abort else ("", "0")
regs = {0x0000: 0x41394558}
fp = open(cap, "w") if cap else None
gi = 0
for path in files:
    regs = {0x0000: 0x41394558}          # per-file reset
    ncap = 0
    for line in open(path):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        op = d["op"]
        if op == "note": continue
        if op == "w": regs[d["a"]] = d["d"] & 0xFFFFFFFF
        elif op == "cap":
            if os.path.basename(path) == ab_name and ncap >= int(ab_n):
                break                     # simulate a mid-file stall/abort
            m = d.get("m", 0)
            v = regs.get(d["a"], 0xDEAD0000 | (d["a"] & 0xFFFF)) & m
            if fp:
                fp.write(json.dumps({"tag": d["tag"], "addr": "0x%08x" % d["a"],
                                     "mask": "0x%08x" % m, "value": v, "i": gi},
                                    separators=(",", ":")) + "\n")
            gi += 1
            ncap += 1
print("F2SIM RESULT: files=%d checks=0 fails=0 -> PASS" % len(files))
if fp: fp.close()
if cap is not None:
    print("F2SIM CAPTURES: n=%d out=%s" % (gi, cap))
sys.exit(0)
'''


def _mkjob(d: Path, name: str, value: int, tagp: str, n: int = 2) -> Path:
    """A job that writes `value` to the scratch reg 0x0008 and captures it
    back n times — two such jobs give DIFFERING capture streams."""
    p = d / name
    lines = [_jline({"op": "note", "s": f"selftest job {name}"}),
             _jline({"op": "w", "a": 0x0008, "d": value})]
    for i in range(n):
        lines.append(_jline({"op": "cap", "a": 0x0008, "m": 0xFFFFFFFF,
                             "tag": f"{tagp}_{i}"}))
    p.write_text("\n".join(lines) + "\n")
    return p


def _selftest() -> int:
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="apex_batch_st_"))
    fails = 0

    def bad(tag, msg):
        nonlocal fails
        fails += 1
        print(f"  [{tag}] FAIL: {msg}")

    def ok(tag, msg):
        print(f"  [{tag}] ok  {msg}")

    try:
        fake = tmp / "fake_exec.py"
        fake.write_text("#!/usr/bin/env python3\n" + _FAKE)
        os.chmod(fake, 0o755)
        A = _mkjob(tmp, "jobA.regops.jsonl", 0x11111111, "scr", 2)
        B = _mkjob(tmp, "jobB.regops.jsonl", 0x22222222, "scr", 3)
        C = _mkjob(tmp, "jobC.regops.jsonl", 0x33333333, "scr", 2)
        a_bytes_before = A.read_bytes()

        # [1] attribution with DIFFERING captures — the required proof shape
        r = run_jobs_batched([A, B], binary=str(fake), workdir=tmp / "w1")
        va = [c["value"] for c in r["files"][0]["captures"]]
        vb = [c["value"] for c in r["files"][1]["captures"]]
        if not r["ok"]:
            bad(1, f"batch not ok: {r['notes']} {[f['notes'] for f in r['files']]}")
        elif va != [0x11111111] * 2 or vb != [0x22222222] * 3:
            bad(1, f"values misattributed: A={va} B={vb}")
        elif not (r["files"][0]["complete"] and r["files"][1]["complete"]):
            bad(1, "complete flags wrong")
        else:
            ok(1, f"A -> {[hex(v) for v in va]}, B -> {[hex(v) for v in vb]} "
                  f"(same tags, different files, correctly split)")

        # [2] identical TAGS across files (the real-world hazard: per-job
        #     counters restart) still attribute by position
        seq = [c["tag"] for c in r["files"][0]["captures"]]
        if seq != ["scr_0", "scr_1"]:
            bad(2, f"file A tags wrong: {seq}")
        else:
            ok(2, "duplicate cross-file tags handled (attribution is by "
                  "marker window, never by tag)")

        # [3] mid-file abort: file B stops after 1 of 3 caps -> B is PARTIAL
        #     and C (after it) still attributes completely
        os.environ["APEX_BFAKE_ABORT"] = "jobB.regops.jsonl:1"
        r = run_jobs_batched([A, B, C], binary=str(fake), workdir=tmp / "w2")
        os.environ.pop("APEX_BFAKE_ABORT", None)
        fA, fB, fC = r["files"]
        if fA["complete"] and (not fB["complete"]) and fB["attribution_ok"] \
                and fB["n_observed"] == 1 and fC["complete"] \
                and [c["value"] for c in fC["captures"]] == [0x33333333] * 2 \
                and not r["ok"]:
            ok(3, f"abort detected: B partial 1/3 (note: "
                  f"{fB['notes'][0][:40]}…), C still bit-correct, batch "
                  f"ok=False")
        else:
            bad(3, f"abort handling wrong: "
                   f"{[(f['complete'], f['n_observed']) for f in r['files']]} "
                   f"ok={r['ok']}")

        # [4] a wrong stream must FAIL attribution, not mis-attribute:
        #     tamper a captured value's tag inside the demux input
        plan = plan_batch([A, B], workdir=tmp / "w3")
        res = bridge.run_job(plan["exec_paths"], executor="sim",
                             binary=str(fake), cap_out=tmp / "w3" / "c.jsonl")
        caps = res["captures"]
        victim = next(c for c in caps if _parse_marker_tag(c["tag"],
                                                           plan["nonce"]) is None)
        victim["tag"] = "not_the_programmed_tag_0"
        dm = demux_captures(caps, plan)
        if dm["ok"] or all(f["attribution_ok"] for f in dm["files"]):
            bad(4, "tampered stream was attributed as ok")
        else:
            ok(4, "tampered stream flagged: "
                  + next(n for f in dm["files"] for n in f["notes"]
                         if "ATTRIBUTION FAILURE" in n)[:60] + "…")

        # [5] orphan capture (executor dropped a marker file) -> hard error
        caps2 = [c for c in res["captures"]
                 if _parse_marker_tag(c["tag"], plan["nonce"]) != 0]
        try:
            demux_captures(caps2, plan)
            bad(5, "stream with a missing marker was demuxed")
        except BatchAttributionError as e:
            ok(5, f"missing marker raised: {str(e)[:60]}…")

        # [6] batched == separate, capture-for-capture (the equivalence
        #     that makes batching a pure transport change)
        rb = run_jobs_batched([A, B, C], binary=str(fake), workdir=tmp / "w4")
        rs = run_jobs_separate([A, B, C], binary=str(fake),
                               workdir=tmp / "w4s")
        eq = [captures_equal(x["captures"], y["captures"])
              for x, y in zip(rb["files"], rs["files"])]
        if all(eq) and rb["ok"] and rs["ok"]:
            ok(6, f"batched == separate for {len(eq)}/{len(eq)} files "
                  f"(tag/addr/mask/value; `i` renumbering excepted)")
        else:
            bad(6, f"equivalence broken: {eq}")

        # [7] marker files leave the job files untouched
        if A.read_bytes() == a_bytes_before:
            ok(7, "job files byte-identical after batching (markers are "
                  "separate files)")
        else:
            bad(7, "job file was modified")

        # [8] the cap_manifest fast path skips only NON-cap lines: a cap
        #     without a tag must still be refused (the executors abort there)
        D = tmp / "jobD.regops.jsonl"
        D.write_text(_jline({"op": "w", "a": 0x0008, "d": 1}) + "\n"
                     + _jline({"op": "cap", "a": 0x0008, "m": 0xFFFFFFFF})
                     + "\n")
        try:
            cap_manifest(D)
            bad(8, "a cap without a tag survived the fast path")
        except BatchAttributionError as e:
            ok(8, f"tagless cap still refused after the CAP_MARK prefilter: "
                  f"{str(e)[:52]}…")
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)
        os.environ.pop("APEX_BFAKE_ABORT", None)
    print(f"BATCH_EXEC SELFTEST: {'FAIL' if fails else 'PASS'} (fails={fails})")
    return 1 if fails else 0


# ═════════════ real-sim attribution proof + timing bench ════════════════════

def _poff_jobs() -> list[Path]:
    d = REPO / "build" / "prompt_offload"
    return sorted(d.glob("poff_*.compute.regops.jsonl"))


def _prove(binary=None) -> int:
    """Attribution proof on the REAL verilated executor, twice over:
    (1) two synthetic scratch jobs whose captured VALUES differ;
    (2) two real produce-mode jobs whose capture streams differ in length
        and content — each checked batched-vs-separate, capture-for-capture.
    """
    tmp = Path(tempfile.mkdtemp(prefix="apex_batch_prove_"))
    fails = 0
    try:
        A = _mkjob(tmp, "provA.regops.jsonl", 0x11111111, "scr", 2)
        B = _mkjob(tmp, "provB.regops.jsonl", 0x22222222, "scr", 2)
        rb = run_jobs_batched([A, B], binary=binary, workdir=tmp / "b")
        va = [c["value"] for c in rb["files"][0]["captures"]]
        vb = [c["value"] for c in rb["files"][1]["captures"]]
        print(f"  [scratch] batched: ok={rb['ok']} A={[hex(v) for v in va]} "
              f"B={[hex(v) for v in vb]}")
        if not (rb["ok"] and va == [0x11111111] * 2 and vb == [0x22222222] * 2):
            print("  [scratch] FAIL: misattributed"); fails += 1

        jobs = _poff_jobs()[:2]
        if len(jobs) == 2:
            rb = run_jobs_batched(jobs, binary=binary, workdir=tmp / "r")
            rs = run_jobs_separate(jobs, binary=binary, workdir=tmp / "rs")
            for k, (x, y) in enumerate(zip(rb["files"], rs["files"])):
                same = captures_equal(x["captures"], y["captures"])
                print(f"  [real {Path(jobs[k]).name}] batched n="
                      f"{x['n_observed']}/{x['n_expected']} complete="
                      f"{x['complete']}  == separate: {same}")
                if not (same and x["complete"]):
                    fails += 1
            cross = captures_equal(rb["files"][0]["captures"],
                                   rb["files"][1]["captures"])
            print(f"  [real] the two files' capture streams differ from each "
                  f"other: {not cross} (required for a meaningful proof)")
            if cross:
                fails += 1
        else:
            print("  [real] SKIP: build/prompt_offload has <2 poff jobs")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"BATCH_EXEC PROOF (real sim): {'FAIL' if fails else 'PASS'}")
    return 1 if fails else 0


def _bench(sizes, binary=None, out_json=None) -> int:
    """Time N jobs one-at-a-time vs batched on the REAL sim executor and
    extract the per-invocation constant. Jobs = the committed produce-mode
    poff regops, cycled (copied under unique names) up to max(sizes)."""
    src = _poff_jobs()
    if not src:
        print("BENCH ABORT: no poff jobs under build/prompt_offload")
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="apex_batch_bench_"))
    rows = []
    try:
        nmax = max(sizes)
        pool = []
        for k in range(nmax):
            s = src[k % len(src)]
            d = tmp / f"j{k:03d}_{s.name}"
            d.write_text(s.read_text())
            pool.append(d)
        for n in sizes:
            jobs = pool[:n]
            rs = run_jobs_separate(jobs, binary=binary, workdir=tmp / f"s{n}")
            rb = run_jobs_batched(jobs, binary=binary, workdir=tmp / f"b{n}")
            eq = all(captures_equal(x["captures"], y["captures"])
                     for x, y in zip(rb["files"], rs["files"]))
            c_inv = ((rs["wall_s"] - rb["wall_s"]) / (n - 1)) if n > 1 else None
            rows.append({"n": n, "sep_s": rs["wall_s"], "bat_s": rb["wall_s"],
                         "speedup": rs["wall_s"] / rb["wall_s"],
                         "sep_ok": rs["ok"], "bat_ok": rb["ok"],
                         "captures_equal": eq,
                         "sep_per_job_s": rs["wall_s"] / n,
                         "bat_per_job_s": rb["wall_s"] / n,
                         "c_inv_est_s": c_inv})
            print(f"  N={n:3d}  separate={rs['wall_s']:8.2f}s "
                  f"({rs['wall_s']/n:6.3f}s/job)   batched={rb['wall_s']:7.2f}s "
                  f"({rb['wall_s']/n:6.3f}s/job)   speedup={rows[-1]['speedup']:5.2f}x   "
                  f"ok={rs['ok']}/{rb['ok']}  captures_equal={eq}")
        ests = [r["c_inv_est_s"] for r in rows if r["c_inv_est_s"]]
        if ests:
            print(f"  per-invocation constant (sim, est from sep-vs-batch "
                  f"deltas): median {statistics.median(ests)*1e3:.1f} ms "
                  f"(spread {min(ests)*1e3:.1f}–{max(ests)*1e3:.1f} ms)")
        if out_json:
            Path(out_json).write_text(json.dumps(
                {"rows": rows, "binary": str(binary or "auto"),
                 "jobs_pool": [p.name for p in pool],
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
            print(f"  record -> {out_json}")
        bad = [r for r in rows
               if not (r["sep_ok"] and r["bat_ok"] and r["captures_equal"])]
        print(f"BATCH_EXEC BENCH: {'FAIL' if bad else 'PASS'}")
        return 1 if bad else 0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("regops", nargs="*", help="regops.jsonl job files")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--cap-out", default=None)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prove", action="store_true",
                    help="attribution proof on the real sim executor")
    ap.add_argument("--bench", metavar="N1,N2,…",
                    help="time one-at-a-time vs batched on the real sim")
    ap.add_argument("--bench-json", default=None)
    a = ap.parse_args()
    import remote_hw_exec                       # hw routing (no-op unless
    remote_hw_exec.attach(bridge, a)            # $APEX_F2_HOST is set) —
    # the breadth_step:1001 idiom: with the shim attached, --executor hw
    # batches over ONE ssh/upload/SDK-attach for the whole invocation.
    if a.selftest:
        return _selftest()
    if a.prove:
        return _prove(binary=a.binary)
    if a.bench:
        return _bench([int(x) for x in a.bench.split(",")], binary=a.binary,
                      out_json=a.bench_json)
    if not a.regops:
        ap.error("give regops files, --selftest, --prove or --bench")
    r = run_jobs_batched(a.regops, executor=a.executor, binary=a.binary,
                         cap_out=a.cap_out, timeout_s=a.timeout_s,
                         tile_div=a.tile_div, slot=a.slot)
    for f in r["files"]:
        print(f"BATCH FILE: {Path(f['path']).name} caps="
              f"{f['n_observed']}/{f['n_expected']} complete={f['complete']} "
              f"attributed={f['attribution_ok']}")
        for n in f["notes"]:
            print(f"  note: {n}")
    for n in r["notes"]:
        print(f"BATCH NOTE: {n}")
    print(f"BATCH: files={r['n_files']} ok={r['ok']} wall={r['wall_s']:.2f}s "
          f"(one invocation)")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
