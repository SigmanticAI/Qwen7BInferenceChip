#!/usr/bin/env python3
# tile_exec_bridge.py — the Python -> executor bridge (audit component 5, N3).
#
# One function, two executors: run_job() hands a regops.jsonl to EITHER the
# verilated sim (verif/f2sim/obj_*/f2sim) or the silicon host runner
# (scripts/fpga/f2/f2_host_run.py) and returns the values the TILE PRODUCED as
# ordinary Python ints. Until this file existed nothing in the tree could read
# a captured value back: sim_main.cpp:444-459 discards it and f2_host_run.py's
# `all_captured` is dead (docs/design/PROMPT_DEMO_AUDIT.md §3 N1/N3).
#
# ══ CAPTURE EGRESS FORMAT (the contract both executors implement) ═══════════
#   * CLI/plusarg:   f2_host_run.py --cap-out <path>   |   f2sim +cap_out=<path>
#   * One JSON line per EXECUTED `cap` op, appended in execution order:
#       {"tag":"<str>","addr":"0x<hex>","mask":"0x<hex>","value":<u32 int>,
#        "i":<seq int>}
#     `value` is ALREADY masked (v & mask) — the executors mask on read.
#   * The file is TRUNCATED at run start.
#   * End-of-run stdout summary, printed WHENEVER --cap-out/+cap_out is given,
#     even for n=0:
#       "F2SIM CAPTURES: n=<N> out=<path>"     (sim)
#       "F2HOST CAPTURES: n=<N> out=<path>"    (host)
#   * Per-site unique tags are the EMITTER's job (trace_to_regops.py):
#     tag = "<sem>_<seq>", sem naming the site (ro_w3, ro_meta, fs, ss, …) and
#     seq a global per-job counter. Address-only tags collapse 25,008 captures
#     onto one name (N2) and are NOT usable for decoding.
#
# ══ WHY THIS FILE IS PARANOID ABOUT THE SUMMARY LINE ═══════════════════════
# `cap` mismatches never enter total_fails, so the executor's EXIT CODE STAYS 0
# whatever happens to the capture path (sim_main.cpp:453-458 vs :578,
# f2_host_run.py:233). An rc-or-grep-FAIL check therefore cannot tell a real
# capture run from a run where the flag was forgotten, the emitter emitted no
# `cap`, or the file was never written. So run_job():
#   1. deletes any pre-existing cap-out file before launch (a stale file must
#      never be mistaken for this run's output),
#   2. REQUIRES the summary line and cross-checks its n against the number of
#      lines actually parsed (require_summary=True, the default: raise),
#   3. reports rc separately from ok — never conflate them.
#
# CLI: python3 tile_exec_bridge.py --selftest   (fake executor, no build needed)
#      python3 tile_exec_bridge.py <regops.jsonl> [--executor sim|hw] [...]

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue as _queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
F2SIM_DIR = REPO / "verif" / "f2sim"
HOST_RUNNER = REPO / "scripts" / "fpga" / "f2" / "f2_host_run.py"

# "F2SIM CAPTURES: n=27996 out=/tmp/cap.jsonl" — either executor's prefix.
SUMMARY_RE = re.compile(
    r"^F2(?:SIM|HOST) CAPTURES:\s*n=(?P<n>\d+)\s+out=(?P<out>\S+)\s*$",
    re.MULTILINE)

TILE_DIV_DEFAULT = 5            # verif/f2sim/Makefile:31 (25 MHz sim ratio)
REQUIRED_KEYS = ("tag", "addr", "mask", "value", "i")


class CaptureEgressError(RuntimeError):
    """The executor did not honour the capture-egress contract above."""


# ── binary discovery ────────────────────────────────────────────────────────
def resolve_sim_binary(binary=None, d: int = 128, ddr=None) -> Path:
    """Locate the verilated executor.

    Order: explicit arg -> $APEX_F2SIM_BIN -> obj_d<D>_ddr0 (the SILICON TWIN;
    the AFI has no DDR decode) -> obj_d<D>_ddr1 (the Makefile default) -> any
    obj_d*/f2sim. Never builds: a multi-minute Verilator run must be the
    caller's explicit choice, and `make run` re-verilates every time because
    `build` is .PHONY (Makefile:40,58).
    """
    if binary:
        p = Path(binary)
        if not p.is_file():
            raise FileNotFoundError(f"f2sim binary not found: {p}")
        return p
    env = os.environ.get("APEX_F2SIM_BIN", "")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"APEX_F2SIM_BIN={env} is not a file")
        return p
    cands = []
    if ddr is not None:
        cands.append(F2SIM_DIR / f"obj_d{d}_ddr{int(ddr)}" / "f2sim")
    else:
        cands += [F2SIM_DIR / f"obj_d{d}_ddr0" / "f2sim",
                  F2SIM_DIR / f"obj_d{d}_ddr1" / "f2sim"]
    cands += sorted(F2SIM_DIR.glob("obj_d*/f2sim"))
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "no f2sim binary; build it with:  make -C verif/f2sim build "
        f"AWS_FPGA=<kit> D={d} DDR=0   (searched {[str(c) for c in cands]})")


# ── cap-out parsing ─────────────────────────────────────────────────────────
def _split_tag(tag: str):
    """'ro_w3_1742' -> ('ro_w3', 1742). Legacy 'r@0x3244' -> ('r@0x3244', None)."""
    sem, _, suf = tag.rpartition("_")
    if sem and suf.isdigit():
        return sem, int(suf)
    return tag, None


def parse_cap_file(path, strict: bool = True) -> list[dict]:
    """Read the JSON-lines cap-out file into normalized records.

    Returns [{'tag','sem','seq','addr':int,'mask':int,'value':int,'i':int}],
    in FILE ORDER (= execution order). addr/mask arrive as "0x…" strings per
    the contract and are converted here; ints are also accepted so a future
    emitter tweak does not break the reader.
    """
    p = Path(path)
    if not p.exists():
        if strict:
            raise CaptureEgressError(
                f"cap-out file was never written: {p} — the executor ignored "
                f"the flag, or no `cap` op was emitted (see N1/N3)")
        return []
    out = []
    for lineno, line in enumerate(p.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise CaptureEgressError(f"{p}:{lineno}: bad JSON line: {e}")
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise CaptureEgressError(
                f"{p}:{lineno}: capture record missing {missing}: {line[:120]}")
        addr = d["addr"] if isinstance(d["addr"], int) else int(d["addr"], 16)
        mask = d["mask"] if isinstance(d["mask"], int) else int(d["mask"], 16)
        val = int(d["value"])
        if not 0 <= val <= 0xFFFFFFFF:
            raise CaptureEgressError(
                f"{p}:{lineno}: value {val} is not a u32 (the executors mask "
                f"on read; a signed value here means somebody sign-extended "
                f"in the WRONG layer — see cap_decode.ro_lanes_to_i32)")
        sem, seq = _split_tag(str(d["tag"]))
        rec = {"tag": str(d["tag"]), "sem": sem, "seq": seq, "addr": addr,
               "mask": mask, "value": val, "i": int(d["i"])}
        if "e" in d:                      # bring-up gate expectation, if any
            rec["e"] = int(d["e"]) if isinstance(d["e"], int) \
                else int(str(d["e"]), 16)
        out.append(rec)
    return out


def audit_captures(caps: list[dict]) -> list[str]:
    """Cheap structural notes (never raises): duplicate tags, non-monotonic i,
    all-one-tag collapse (the N2 symptom), masked-value/mask disagreement."""
    notes = []
    tags = [c["tag"] for c in caps]
    if caps and len(set(tags)) == 1 and len(caps) > 1:
        notes.append(f"ALL {len(caps)} captures share tag {tags[0]!r} — "
                     f"address-only tagging (N2); values are undecodable")
    dups = len(tags) - len(set(tags))
    if dups:
        notes.append(f"{dups} duplicate tag(s) — per-site uniqueness broken")
    seqs = [c["i"] for c in caps]
    if seqs != sorted(seqs):
        notes.append("`i` is not monotonic in file order")
    if len(set(seqs)) != len(seqs):
        notes.append("`i` has duplicates — the global counter is not global")
    bad = [c["tag"] for c in caps if c["value"] & ~c["mask"] & 0xFFFFFFFF]
    if bad:
        notes.append(f"{len(bad)} capture(s) carry bits outside their mask "
                     f"(first: {bad[0]}) — executor did not mask on read")
    return notes


# ── the persistent hw executor (kills the per-invocation fixed cost) ────────
# Every hw run_job used to be a FRESH `python3 f2_host_run.py` subprocess —
# python startup + module imports + the clkgen preflight (one
# fpga-describe-clkgen subprocess per invocation, f2_host_run's LC-1 rule) +
# pci_attach, paid per call, i.e. per TOKEN on the batched token loop. The
# daemon path starts ONE long-lived `f2_host_run.py --server --slot N`
# process on the first hw job (attach paid once), then sends each job as a
# JSON line ({"argv": [...]} — the exact single-shot argv tail) and reads
# one JSON reply ({"rc": ..., "log": ...}). The run_job() CONTRACT IS
# UNCHANGED: the server truncates/append the same cap-out file, the reply
# log carries the same F2HOST summary lines, and all parsing below is
# shared with the subprocess path.
#   * revert: APEX_NO_DAEMON=1 (env), --no-daemon (CLI), run_job(daemon=False)
#   * wedge:  a job timeout kills the server and reports EXACTLY what a
#     subprocess timeout reported (rc=124, timed_out=True, partial cap file)
#   * death:  a server that dies mid-job is reported LOUDLY on stderr and
#     restarted ONCE; a second death raises CaptureEgressError
#   * lifecycle: engines call shutdown_daemons() at finish; atexit covers
#     the rest. One daemon per (interp, slot, cwd, env) key.
DAEMON_ENV_OFF = "APEX_NO_DAEMON"
SERVER_READY_TIMEOUT_S = float(os.environ.get("APEX_DAEMON_READY_S", "120"))

_DAEMONS: dict = {}


class _HostDaemon:
    """One live `f2_host_run.py --server` process + its protocol pipe.
    A reader thread pumps stdout lines into a queue so requests can time
    out without blocking; server stderr stays inherited (operator-visible,
    never part of the protocol)."""

    def __init__(self, interp: str, slot: int, cwd, env):
        self.argv = [interp, str(HOST_RUNNER), "--server",
                     "--slot", str(slot)]
        self.p = subprocess.Popen(self.argv, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=None,
                                  text=True, bufsize=1,
                                  cwd=str(cwd) if cwd else None, env=env)
        self.q: _queue.Queue = _queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        try:
            hello = self._recv(SERVER_READY_TIMEOUT_S)
        except (_queue.Empty, BrokenPipeError) as e:
            self.kill()
            raise CaptureEgressError(
                f"hw server never became ready ({e.__class__.__name__}) — "
                f"argv={self.argv}")
        if not hello.get("ready"):
            self.kill()
            raise CaptureEgressError(
                f"hw server refused to start: {hello} — argv={self.argv} "
                f"(see its stderr above; APEX_NO_DAEMON=1 reverts to "
                f"one-shot subprocesses)")

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put(None)                       # EOF sentinel

    def _recv(self, timeout_s):
        line = self.q.get(timeout=timeout_s)   # queue.Empty on timeout
        if line is None:
            raise BrokenPipeError("hw server closed its stdout")
        return json.loads(line)

    def request(self, argv_tail, timeout_s):
        self.p.stdin.write(json.dumps({"argv": [str(x) for x in argv_tail]})
                           + "\n")
        self.p.stdin.flush()
        return self._recv(timeout_s)

    def kill(self):
        try:
            if self.p.poll() is None:
                self.p.kill()
            self.p.wait(timeout=5)
        except OSError:
            pass

    def stop(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.write(json.dumps({"op": "exit"}) + "\n")
                self.p.stdin.flush()
                self.p.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.kill()


def shutdown_daemons() -> None:
    """Stop every persistent hw executor this process started. Called by
    the engines at finish() and by atexit; idempotent, safe with none."""
    for key in list(_DAEMONS):
        _DAEMONS.pop(key).stop()


def _daemon_on(daemon) -> bool:
    if daemon is not None:
        return bool(daemon)
    return not os.environ.get(DAEMON_ENV_OFF)


def _daemon_request(interp: str, slot: int, cwd, env, argv_tail,
                    timeout_s) -> tuple:
    """One job through the persistent server. Returns (rc, log, timed_out)
    — the exact observables subprocess.run produced. Lazy start; wedge =
    kill + the subprocess-timeout result; death mid-job = loud stderr +
    ONE restart (safe: every job starts with per-file TILE_RST and its own
    cap-out truncation), then raise."""
    key = (interp, int(slot), str(cwd) if cwd else None,
           tuple(sorted(env.items())) if env else None)
    for attempt in (1, 2):
        d = _DAEMONS.get(key)
        if d is None or d.p.poll() is not None:
            d = _HostDaemon(interp, int(slot), cwd, env)
            _DAEMONS[key] = d
            atexit.register(shutdown_daemons)   # idempotent shutdown
        try:
            rep = d.request(argv_tail, timeout_s)
            return int(rep["rc"]), str(rep.get("log", "")), False
        except _queue.Empty:
            print(f"BRIDGE: hw server wedged (no reply in {timeout_s}s) — "
                  f"killed; reporting the subprocess-timeout result",
                  file=sys.stderr)
            d.kill()
            _DAEMONS.pop(key, None)
            return 124, "", True
        except (BrokenPipeError, OSError, ValueError) as e:
            d.kill()
            _DAEMONS.pop(key, None)
            print(f"BRIDGE: hw server DIED mid-job ({e}) — "
                  f"{'restarting once' if attempt == 1 else 'giving up'}",
                  file=sys.stderr)
            if attempt == 2:
                raise CaptureEgressError(
                    f"hw server died twice on the same job — "
                    f"argv_tail={list(map(str, argv_tail))}: {e}")


# ── the bridge ──────────────────────────────────────────────────────────────
def run_job(regops_path, executor: str = "sim", binary=None, cap_out=None,
            timeout_s: int = 1800, *, tile_div: int = TILE_DIV_DEFAULT,
            slot: int = 0, extra_args=(), cwd=None, env=None,
            require_summary: bool = True, keep_cap_out: bool = True,
            daemon: bool | None = None) -> dict:
    """Execute regops on one executor and return the tile's captured values.

    regops_path : one path, or a sequence of paths (executors take many).
    executor    : 'sim' (verilated cl_apex) | 'hw' (f2_host_run.py on F2).
    binary      : 'sim' -> f2sim path (default: resolve_sim_binary());
                  'hw'  -> python interpreter for f2_host_run.py (default:
                  sys.executable). The runner script itself is fixed.
    cap_out     : where the executor writes JSON-lines (default: a temp file;
                  kept unless keep_cap_out=False).
    daemon      : hw only — route the job through the persistent
                  `f2_host_run.py --server` (see the daemon block above).
                  None (default) = on unless $APEX_NO_DAEMON is set;
                  False = the historical one-shot subprocess. The returned
                  dict is identical either way.

    Returns {'captures': [...], 'rc': int, 'log': str,  # the contract keys
             'ok': bool, 'n_reported': int|None, 'summary': str|None,
             'cap_out': str, 'argv': [...], 'executor': str, 'notes': [...],
             'timed_out': bool}

    `ok` means: process exited 0 AND the summary line was printed AND its n
    equals the number of records parsed. rc alone proves nothing about the
    capture path (see the header).
    """
    if executor not in ("sim", "hw"):
        raise ValueError(f"executor must be 'sim' or 'hw', got {executor!r}")
    regops = [regops_path] if isinstance(regops_path, (str, os.PathLike)) \
        else list(regops_path)
    regops = [str(Path(r)) for r in regops]
    if not regops:
        raise ValueError("no regops files given")
    for r in regops:
        if not Path(r).is_file():
            raise FileNotFoundError(f"regops file not found: {r}")

    tmp_made = False
    if cap_out is None:
        fd, cap_out = tempfile.mkstemp(prefix="apex_cap_", suffix=".jsonl")
        os.close(fd)
        tmp_made = True
    cap_out = str(Path(cap_out).expanduser())
    Path(cap_out).parent.mkdir(parents=True, exist_ok=True)
    # A stale file must never be read as this run's output. The executors
    # truncate too; this is belt-and-braces for the case where they never
    # open it at all.
    try:
        os.unlink(cap_out)
    except FileNotFoundError:
        pass

    use_daemon = False
    if executor == "sim":
        argv = [str(resolve_sim_binary(binary)), f"+tile_div={tile_div}",
                f"+cap_out={cap_out}", *map(str, extra_args), *regops]
    else:
        if not HOST_RUNNER.is_file():
            raise FileNotFoundError(f"host runner missing: {HOST_RUNNER}")
        argv_tail = [*regops, "--slot", str(slot), "--cap-out", cap_out,
                     *map(str, extra_args)]
        argv = [str(binary or sys.executable), str(HOST_RUNNER), *argv_tail]
        use_daemon = _daemon_on(daemon)

    timed_out = False
    if use_daemon:
        rc, log, timed_out = _daemon_request(argv[0], slot, cwd, env,
                                             argv_tail, timeout_s)
    else:
        try:
            pr = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout_s,
                                cwd=str(cwd) if cwd else None, env=env)
            rc, log = pr.returncode, (pr.stdout or "") + (pr.stderr or "")
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc = 124
            log = ((e.stdout or "") if isinstance(e.stdout, str)
                   else (e.stdout or b"").decode("utf8", "replace")) + \
                  ((e.stderr or "") if isinstance(e.stderr, str)
                   else (e.stderr or b"").decode("utf8", "replace"))

    m = None
    for m in SUMMARY_RE.finditer(log):      # last one wins (per-run summary)
        pass
    n_reported = int(m.group("n")) if m else None
    summary = m.group(0).strip() if m else None

    # A timeout leaves a PARTIAL file, which is useful on multi-hour Verilator
    # runs — parse it, but never demand the summary that cannot exist.
    caps = parse_cap_file(cap_out, strict=not timed_out and require_summary)
    notes = audit_captures(caps)
    if timed_out:
        notes.insert(0, f"TIMEOUT after {timeout_s}s — captures are PARTIAL")
    if m and m.group("out") != cap_out:
        notes.append(f"summary out={m.group('out')} != requested {cap_out}")
    if n_reported is not None and n_reported != len(caps):
        notes.append(f"summary n={n_reported} != {len(caps)} records parsed")

    if summary is None and require_summary and not timed_out:
        raise CaptureEgressError(
            f"executor printed no 'F2SIM/F2HOST CAPTURES:' line — the "
            f"capture-egress contract is unimplemented in {argv[0]}, or the "
            f"flag was dropped. rc={rc} (rc is NOT evidence: cap mismatches "
            f"never reach total_fails). argv={argv}")

    ok = (rc == 0 and summary is not None and n_reported == len(caps)
          and not timed_out)
    res = {"captures": caps, "rc": rc, "log": log, "ok": ok,
           "n_reported": n_reported, "summary": summary, "cap_out": cap_out,
           "argv": argv, "executor": executor, "notes": notes,
           "timed_out": timed_out}
    if tmp_made and not keep_cap_out:
        try:
            os.unlink(cap_out)
        except OSError:
            pass
    return res


def captures_by_sem(caps: list[dict]) -> dict:
    """{sem: [records in i-order]} — the handle cap_decode's decoders want."""
    out: dict = {}
    for c in sorted(caps, key=lambda c: c["i"]):
        out.setdefault(c["sem"], []).append(c)
    return out


# ── selftest: a FAKE executor, so no Verilator build and no FPGA needed ─────
_FAKE = r'''
import json, sys, os
cap = mode = None
for a in sys.argv[1:]:
    if a.startswith("+cap_out="): cap = a.split("=", 1)[1]
    elif a == "--cap-out": cap = sys.argv[sys.argv.index(a) + 1]
    elif a.startswith("--mode="): mode = a.split("=", 1)[1]
n = 0
if cap and mode != "nofile":
    with open(cap, "w") as f:                      # TRUNCATE at run start
        for i in range(3):
            for w in range(2):
                f.write(json.dumps({
                    "tag": f"ro_w{w}_{i * 2 + w}",
                    "addr": f"{0x3204 + 4 * w:#x}", "mask": "0xffffffff",
                    "value": (0xFFFFFFF0 + i) if w else (i + 1), "i": i * 2 + w,
                }, separators=(",", ":")) + "\n")
                n += 1
print("[fake] 6 ops, 0 checks, 0 fails")
print("F2SIM RESULT: files=1 checks=0 fails=0 -> PASS")
if mode == "nosummary":
    sys.exit(0)
print(f"F2SIM CAPTURES: n={n if mode != 'badn' else n + 1} out={cap}")
sys.exit(0)
'''


def _selftest() -> int:
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="apex_bridge_st_"))
    fails = 0
    try:
        fake = tmp / "fake_exec.py"
        fake.write_text("#!/usr/bin/env python3\n" + _FAKE)
        os.chmod(fake, 0o755)
        regops = tmp / "job_fake.regops.jsonl"
        regops.write_text(json.dumps({"op": "note", "s": "selftest"}) + "\n")
        capf = tmp / "cap.jsonl"

        # 1. happy path — captures reach a Python variable
        r = run_job(regops, executor="sim", binary=str(fake), cap_out=capf)
        assert r["rc"] == 0 and r["ok"], r
        assert len(r["captures"]) == 6 and r["n_reported"] == 6, r["captures"]
        c0 = r["captures"][0]
        assert (c0["tag"], c0["sem"], c0["seq"]) == ("ro_w0_0", "ro_w0", 0), c0
        assert c0["addr"] == 0x3204 and c0["value"] == 1, c0
        assert r["captures"][1]["value"] == 0xFFFFFFF0, r["captures"][1]
        assert r["notes"] == [], r["notes"]
        print(f"  [1] ok  n={len(r['captures'])} summary={r['summary']!r}")

        # 2. stale cap-out must not be inherited: pre-fill, then run a fake
        #    that writes NO file -> the reader must raise, not return stale data
        capf.write_text(json.dumps(
            {"tag": "stale_0", "addr": "0x3204", "mask": "0xffffffff",
             "value": 7, "i": 0}) + "\n")
        try:
            run_job(regops, executor="sim", binary=str(fake), cap_out=capf,
                    extra_args=["--mode=nofile"])
            print("  [2] FAIL: stale cap-out was accepted"); fails += 1
        except CaptureEgressError as e:
            print(f"  [2] ok  stale file rejected: {str(e)[:64]}…")

        # 3. missing summary line is a HARD error (the rc==0 vacuity trap)
        try:
            run_job(regops, executor="sim", binary=str(fake), cap_out=capf,
                    extra_args=["--mode=nosummary"])
            print("  [3] FAIL: missing summary accepted"); fails += 1
        except CaptureEgressError as e:
            print(f"  [3] ok  no-summary rejected: {str(e)[:64]}…")

        # 4. n mismatch: rc 0, summary present, but the count lies -> not ok
        r = run_job(regops, executor="sim", binary=str(fake), cap_out=capf,
                    extra_args=["--mode=badn"])
        assert r["rc"] == 0 and not r["ok"] and r["notes"], r
        print(f"  [4] ok  rc=0 but ok=False; notes={r['notes']}")

        # 5. by-sem grouping
        g = captures_by_sem(r["captures"])
        assert sorted(g) == ["ro_w0", "ro_w1"] and len(g["ro_w0"]) == 3, g
        print(f"  [5] ok  sems={sorted(g)}")

        # 6. hw daemon: run_job through a PERSISTENT fake-pci server must
        #    equal the one-shot subprocess result on every contract key
        #    that does not embed the cap-out path, and the second call must
        #    REUSE the server (same pid, no respawn).
        J = json.dumps
        hwregs = tmp / "job_hw.regops.jsonl"
        hwregs.write_text("\n".join([
            J({"op": "w", "a": 0x1000, "d": 1}),
            J({"op": "w", "a": 0x2000, "d": 2}),
            J({"op": "w", "a": 0x2004, "d": 3}),
            J({"op": "cap", "a": 0x3204, "m": 0xFFFFFFFF,
               "tag": "ro_w0_0"}),
            J({"op": "cap", "a": 0x3208, "m": 0xFFFFFFFF,
               "tag": "ro_w1_1"}),
            J({"op": "r", "a": 0x3204, "m": 0xFFFF, "e": 0xBEEF}),
        ]) + "\n")
        spec = tmp / "fake_pci.json"
        spec.write_text(J({"regs": {"0x3204": 0xBEEF, "0x3208": 0x40}}))
        env = {**os.environ, "APEX_F2HOST_FAKE_PCI": str(spec)}
        r0 = run_job(hwregs, executor="hw", cap_out=tmp / "hw0.jsonl",
                     env=env, daemon=False)
        r1 = run_job(hwregs, executor="hw", cap_out=tmp / "hw1.jsonl",
                     env=env, daemon=True)
        pid1 = next(iter(_DAEMONS.values())).p.pid
        r2 = run_job(hwregs, executor="hw", cap_out=tmp / "hw2.jsonl",
                     env=env, daemon=True)
        pid2 = next(iter(_DAEMONS.values())).p.pid
        shutdown_daemons()
        assert not _DAEMONS, "shutdown_daemons left a daemon registered"
        assert r0["ok"] and r1["ok"] and r2["ok"], \
            (r0["rc"], r1["rc"], r2["rc"], r1["log"][-400:])
        assert pid1 == pid2, "second hw job did not reuse the server"
        for k in ("captures", "rc", "ok", "n_reported", "notes",
                  "timed_out"):
            assert r1[k] == r0[k] == r2[k], (k, r0[k], r1[k])
        assert r1["summary"].startswith("F2HOST CAPTURES: n=2"), \
            r1["summary"]
        print(f"  [6] ok  hw daemon == one-shot on contract keys; server "
              f"reused (pid {pid1}); captures={len(r1['captures'])}")
    finally:
        shutdown_daemons()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"BRIDGE SELFTEST: {'FAIL' if fails else 'PASS'} (fails={fails})")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("regops", nargs="*", help="regops.jsonl file(s)")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--cap-out", default=None)
    ap.add_argument("--tile-div", type=int, default=TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--no-require-summary", action="store_true")
    ap.add_argument("--no-daemon", action="store_true",
                    help="hw: one-shot f2_host_run subprocess per job "
                         "(the pre-daemon behavior; APEX_NO_DAEMON=1 is "
                         "the env equivalent)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.regops:
        ap.error("give regops file(s) or --selftest")
    r = run_job(a.regops, executor=a.executor, binary=a.binary,
                cap_out=a.cap_out, timeout_s=a.timeout_s, tile_div=a.tile_div,
                slot=a.slot, require_summary=not a.no_require_summary,
                daemon=False if a.no_daemon else None)
    print(r["log"].rstrip())
    print(f"BRIDGE: rc={r['rc']} ok={r['ok']} captures={len(r['captures'])} "
          f"reported={r['n_reported']} -> {r['cap_out']}")
    for n in r["notes"]:
        print(f"BRIDGE NOTE: {n}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
