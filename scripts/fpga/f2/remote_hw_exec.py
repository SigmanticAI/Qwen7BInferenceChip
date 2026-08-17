#!/usr/bin/env python3
# remote_hw_exec.py — Milestone D transport: execute one regops JOB on the REAL
# F2 instance from this Mac, over ssh, and return the tile's captured values in
# EXACTLY the dict shape tile_exec_bridge.run_job() returns.
#
# ══ WHY A REMOTE EXECUTOR AT ALL (the split-host constraint) ════════════════
# The 7.6 GB Qwen2.5-7B weight cache, the tokenizer and the mlx/HF venv live on
# this Mac; the PCI device lives on an f2.6xlarge. Per-OP round trips are
# impossible (one job is ~27k–170k BAR0 ops), per-JOB round trips are cheap
# (~5 jobs per prompt). So the unit shipped over the wire is a whole
# regops.jsonl — compiled here by compute_job.py, executed there by
# f2_host_run.py, with the capture file (docs/design/PROMPT_ON_CHIP.md §0
# egress) scp'd back and parsed HERE by tile_exec_bridge.parse_cap_file, which
# is imported and reused, never reimplemented.
#
# ══ THE TRAP THIS FILE EXISTS TO DEFEAT (read this before trusting a run) ═══
# f2_host_run.py's MMCM-lock preflight (f2_host_run.py:49-73) accepts ANY
# output containing the substring "lock" — and `fpga-describe-clkgen`'s own
# table prints "Clock Group A ...", so the preflight PASSES VACUOUSLY, always
# (docs/results/f2_stage2_hw/RESULT.md:64-68). Meanwhile an AFI load RESETS the
# clkgen MMCMs to the DEFAULT recipe A1, i.e. clk_extra_a1 = 125.00 MHz
# (RESULT.md:38-44), while the tile only closes timing at recipe A2 =
# 15.625 MHz (docs/design/LEVEL_C_INTEGRATION.md:116). A host that forgets
# `sudo fpga-load-clkgen-recipe -S 0 -a 2` therefore runs the tile ~8× over its
# closed clock: it computes GARBAGE while every preflight and every rc says OK,
# and `cap` values (which carry no expectation in Milestone B) would be
# silently substituted into the model.
#
# So verify_clock=True (the default) reads the frequency BEFORE anything else
# and REFUSES to ship or execute the job unless clk_extra_a1 parses to the
# EXPECTED value ± 0.2 MHz — IMAGE-KEYED via clock_key.py when the caller
# passes agfi= / $APEX_F2_AGFI / --agfi (A2 images: 15.625; an A0 image:
# 62.5), else the unkeyed A2 default 15.625. Refusal is loud on stderr,
# rc=78, ok=False, and notes[0] explains it — the caller (prompt_offload)
# turns that into a RuntimeError.
#
# ══ CONTRACT MIRRORING (what "SAME dict shape" means here) ══════════════════
#   run_job_remote() returns the 11 keys run_job() documents
#   (tile_exec_bridge.py:199-207): captures, rc, log, ok, n_reported, summary,
#   cap_out, argv, executor, notes, timed_out — same meanings, same types —
#   plus remote-only extras (host, remote_*, clkgen_a1_mhz, refused).
#   `ok` = rc==0 AND the "F2HOST CAPTURES:" summary was printed AND its n
#   equals the number of records parsed AND no timeout. rc alone proves
#   nothing: `cap` mismatches never enter total_fails (f2_host_run.py:266,288),
#   so a run that captured nothing still exits 0.
#   A missing summary line RAISES CaptureEgressError, exactly as in the sim
#   path (tile_exec_bridge.py:275-280) — the one deliberate asymmetry is the
#   clock refusal, which RETURNS a dict so the note survives into the record.
#
#   `rc` is the RUNNER's exit status, recovered from an "APEX_REMOTE_RC=$?"
#   sentinel the remote shell echoes. ssh's own rc cannot be used: the command
#   ends in that echo, so ssh reports 0 whatever the runner did. If the
#   sentinel is missing, the transport/setup failed and rc is forced non-zero.
#
# ══ WIRING prompt_offload.py TO THIS FILE — the exact 2-line change ═════════
# NOT APPLIED HERE (prompt_offload.py is another agent's file). In
# prompt_offload.main(), immediately after `args = ap.parse_args()`
# (prompt_offload.py:842), insert:
#
#     import remote_hw_exec                                      # + line 1
#     remote_hw_exec.attach(bridge, args)                        # + line 2
#
# `bridge` is already imported there (prompt_offload.py:81) and HERE is already
# on sys.path (prompt_offload.py:71-73), so no other edit is needed.
# attach() is a NO-OP unless $APEX_F2_HOST is set, so `--executor sim` and
# every existing selftest keep byte-identical behaviour. With it set,
# `--executor hw` routes prompt_offload.py:416's bridge.run_job(executor='hw')
# to run_job_remote() and the rest of that function (captures -> cap_decode /
# compute_job.grade_compute_job -> substitution) is untouched.
#
#   export APEX_F2_HOST=ubuntu@<f2-ip>     APEX_F2_KEY=~/.ssh/apex-f2.pem
#   [optional] APEX_F2_REMOTE_DIR=~/apexrun  APEX_F2_PY=python3
#              APEX_F2_NO_SUDO=1  APEX_F2_RUNNER=<path on the instance>
#              APEX_F2_SSH_OPTS='-o …'      APEX_F2_USER=ubuntu
#              APEX_F2_REMOTE_ENV='AWS_FPGA_REPO_DIR=/opt/aws-fpga'
#              APEX_F2_NO_CLOCK_CHECK=1   # do not use for a real claim
#
# ══ CLI ════════════════════════════════════════════════════════════════════
#   python3 remote_hw_exec.py --selftest            # no instance, no spend
#   python3 remote_hw_exec.py --check-clock --host H --key K
#   python3 remote_hw_exec.py job.regops.jsonl --host H --key K --cap-out C

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                      # direct CLI use
    sys.path.insert(0, str(HERE))

import tile_exec_bridge as bridge                              # noqa: E402
from tile_exec_bridge import (                                 # noqa: E402
    SUMMARY_RE, CaptureEgressError, audit_captures, parse_cap_file)

import clock_key                                               # noqa: E402

HOST_RUNNER = bridge.HOST_RUNNER          # scripts/fpga/f2/f2_host_run.py

# ── the tile clock (DECISION-LC-1, image-keyed since the A0 track) ──────────
# recipe A2 = 1250/80 = 15.625 MHz; `fpga-describe-clkgen` prints it rounded to
# 15.62 (docs/results/f2_stage2_hw/clkgen_final.txt), hence a tolerance rather
# than equality. The failure mode this must catch is the DEFAULT recipe A1
# (125.00) and a dead clock (0.00) — both far outside ±0.2.
#
# TWO legal frequencies exist since the A0 (62.5 MHz) track opened
# (docs/design/CLOCK_LADDER.md): the expectation is therefore IMAGE-KEYED via
# clock_key.expected_clock(agfi) — pass `agfi=` (or $APEX_F2_AGFI / --agfi)
# and the gate binds to the recipe that image's netlist was constrained at.
# The constants below remain the UNKEYED default: A2, the value every image
# flown before 2026-08-05 was constrained at. An A0 card meeting an unkeyed
# gate refuses (62.50 != 15.625) — the safe direction; it can never make an
# A2 image run fast.
TILE_CLK_MHZ = clock_key.RECIPE_A_A1_MHZ[2]        # 15.625
TILE_CLK_TOL = clock_key.A1_TOL                    # 0.2
CLKGEN_A1 = "clk_extra_a1"
RECIPE_FIX = "sudo fpga-load-clkgen-recipe -S 0 -a 2"

RC_CLOCK_REFUSED = 78          # sysexits EX_CONFIG — never executed anything
RC_TIMEOUT = 124               # same spelling as run_job's timeout rc
RC_TRANSPORT = 255             # ssh's own "could not talk to the instance"

SENTINEL = "APEX_REMOTE_RC="
SENTINEL_RE = re.compile(r"^" + SENTINEL + r"(\d+)\s*$", re.MULTILINE)

# BatchMode: never hang on a password/passphrase prompt in an automated run.
# accept-new: an ephemeral devbox IP has no known_hosts entry yet (TOFU); set
# APEX_F2_SSH_OPTS / ssh_opts= to tighten it if you pre-seed known_hosts.
DEFAULT_SSH_OPTS = ("-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=15",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ServerAliveInterval=30",
                    "-o", "ServerAliveCountMax=20")

# ── connection multiplexing (the per-invocation constant) ───────────────────
# EVERY run_job_remote makes FIVE separate ssh/scp connections: the clkgen
# probe, mkdir, the upload, the job, the capture fetch. Each one pays a fresh
# TCP handshake + key exchange + auth against us-west-2 — from a residential
# uplink that is ~0.4-0.9 s apiece, and it is most of the 3.7 s per-invocation
# constant the batching study measured (BATCHING_STUDY.md §a). OpenSSH solves
# this with ControlMaster: the first connection opens a control socket and
# every later ssh/scp to the same host rides it, paying one round trip.
#
# Nothing about WHAT runs changes — same commands, same clock gate, same
# per-file TILE_RST. The socket lives in a private 0700 dir, is keyed by the
# process id so two sessions never share one, and ControlPersist reaps it.
# $APEX_F2_NO_SSH_MUX=1 turns it off (a host whose sshd sets MaxSessions=1, or
# a debugging session that wants every connection to stand on its own).
SSH_MUX_PERSIST = "600"          # seconds the master lingers after last use
_MUX_DIR: Path | None = None


def ssh_mux_opts(enable: bool = True) -> tuple:
    """The ControlMaster options, or () when multiplexing is off."""
    global _MUX_DIR
    if not enable or os.environ.get("APEX_F2_NO_SSH_MUX"):
        return ()
    if _MUX_DIR is None:
        # NOT tempfile.mkdtemp(): on macOS $TMPDIR is /var/folders/<deep...>,
        # and unix control sockets are capped at 104 bytes — ssh refuses with
        # "ControlPath too long" and every muxed call dies rc=255 (measured
        # 2026-08-04; the clock gate caught it). Use a short, fixed, 0700 dir
        # under $HOME; %C keeps one socket per destination without embedding
        # the hostname in the path.
        d = Path.home() / ".apexmux"
        d.mkdir(mode=0o700, exist_ok=True)
        d.chmod(0o700)
        _MUX_DIR = d
    # %C is a hash of (host, port, user, proxy) — one socket per destination.
    return ("-o", "ControlMaster=auto",
            "-o", f"ControlPath={_MUX_DIR}/%C",
            "-o", f"ControlPersist={SSH_MUX_PERSIST}")


def ssh_mux_close(host: str, key=None, user: str = "ubuntu",
                  ssh: str = "ssh", ssh_opts=()) -> bool:
    """Tear the master connection down (end of session / after a transport
    error). Best effort: a dead socket is not an error worth propagating."""
    if _MUX_DIR is None:
        return False
    argv = [str(ssh), *[str(o) for o in ssh_opts], *ssh_mux_opts(),
            "-O", "exit"]
    if key:
        argv += ["-i", str(Path(str(key)).expanduser())]
    argv.append(_userhost(host, user))
    try:
        subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return True
    except Exception:
        return False

CONTRACT_KEYS = ("captures", "rc", "log", "ok", "n_reported", "summary",
                 "cap_out", "argv", "executor", "notes", "timed_out")


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ══════════════════════ remote path / argv plumbing ═════════════════════════

def _rq(path: str) -> str:
    """shlex.quote for a REMOTE SHELL word, keeping a leading '~/' expandable.

    shlex.quote('~/apexrun') -> "'~/apexrun'", which the remote shell does NOT
    expand — the job would land in a literal directory named '~'.
    """
    p = str(path)
    if p == "~":
        return "~"
    if p.startswith("~/"):
        return "~/" + shlex.quote(p[2:])
    return shlex.quote(p)


def _scp_path(path: str) -> str:
    """Spelling of a remote path for scp's file argument.

    OpenSSH 9's scp is SFTP-backed and does not reliably expand '~', but a
    RELATIVE path is defined against the login directory — the same place. So
    '~/apexrun/x' is sent as 'apexrun/x'.
    """
    p = str(path)
    if p == "~":
        return "."
    if p.startswith("~/"):
        return p[2:]
    return p


def _rjoin(d: str, name: str) -> str:
    return f"{str(d).rstrip('/')}/{name}"


def build_remote_argv(regops_names, *, slot: int = 0,
                      cap_name: str = "./cap.jsonl",
                      python: str = "python3",
                      runner: str = "./f2_host_run.py",
                      extra_args=(), skip_clkgen_wait: bool = False) -> list:
    """The f2_host_run.py command line, in run_job's exact order.

    MIRRORS tile_exec_bridge.py:240-242 —
        [python, runner, *regops, "--slot", str(slot), "--cap-out", cap, *extra]
    against f2_host_run.py's real parser (f2_host_run.py:77-89):
        regops  nargs="+"  positional      :78
        --slot  int                        :79
        --stop-on-fail  store_true         :80   (via extra_args)
        --skip-clkgen-wait  store_true     :82   (via skip_clkgen_wait)
        --cap-out  PATH                    :85
    Interleaving optionals after an nargs="+" positional is what the sim path
    already does, and selftest case [0] proves the REAL parser accepts it by
    executing it.
    """
    argv = [str(python), str(runner), *[str(r) for r in regops_names],
            "--slot", str(int(slot)), "--cap-out", str(cap_name)]
    if skip_clkgen_wait:
        argv.append("--skip-clkgen-wait")
    argv += [str(a) for a in extra_args]
    return argv


def _ssh_argv(ssh, key, ssh_opts, userhost, command, *, mux=True) -> list:
    argv = [str(ssh), *[str(o) for o in ssh_opts], *ssh_mux_opts(mux)]
    if key:
        argv += ["-i", str(Path(str(key)).expanduser())]
    argv += [str(userhost), str(command)]
    return argv


def _scp_argv(scp, key, ssh_opts, srcs, dst, *, mux=True) -> list:
    argv = [str(scp), *[str(o) for o in ssh_opts], *ssh_mux_opts(mux)]
    if key:
        argv += ["-i", str(Path(str(key)).expanduser())]
    argv += [str(s) for s in srcs] + [str(dst)]
    return argv


def _userhost(host: str, user: str = "ubuntu") -> str:
    h = str(host)
    return h if "@" in h else f"{user}@{h}"


def _run_local(argv, timeout_s, label, log_parts, verbose=False):
    """Run a local process (ssh/scp), append a traceable block to log_parts.

    Returns (rc, combined_output, timed_out).
    """
    if verbose:
        eprint(f"[remote_hw_exec] {label}: "
               + " ".join(shlex.quote(a) for a in argv))
    log_parts.append(f"[remote_hw_exec] $ "
                     + " ".join(shlex.quote(a) for a in argv))
    try:
        pr = subprocess.run(argv, capture_output=True, text=True,
                            timeout=timeout_s)
        out = (pr.stdout or "") + (pr.stderr or "")
        log_parts.append(out.rstrip("\n"))
        return pr.returncode, out, False
    except subprocess.TimeoutExpired as e:
        def _s(x):
            if x is None:
                return ""
            return x if isinstance(x, str) else x.decode("utf8", "replace")
        out = _s(e.stdout) + _s(e.stderr)
        log_parts.append(out.rstrip("\n"))
        log_parts.append(f"[remote_hw_exec] TIMEOUT after {timeout_s}s: {label}")
        return RC_TIMEOUT, out, True
    except FileNotFoundError as e:
        log_parts.append(f"[remote_hw_exec] {label}: {e}")
        return RC_TRANSPORT, str(e), False


# ═══════════════════════ the anti-vacuity clock gate ════════════════════════

def parse_clkgen_a1(text: str) -> float | None:
    """clk_extra_a1's MHz out of `fpga-describe-clkgen -S 0`, or None.

    Real output (docs/results/f2_stage2_hw/clkgen_final.txt):

        Clock Group A Frequency (Mhz)
        | clk_extra_a1 | clk_extra_a2 | clk_extra_a3 |
        |--------------|--------------|--------------|
        |      15.62   |     125.00   |      62.50   |

    Column position is taken from the HEADER row (not assumed), so extra or
    reordered columns cannot silently return a2's frequency. None means "the
    tool did not report a1" — which is a REFUSAL, never a pass.
    """
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if CLKGEN_A1 not in line or "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if CLKGEN_A1 not in cells:
            continue
        col = cells.index(CLKGEN_A1)
        for nxt in lines[i + 1:]:
            if "|" not in nxt:
                break                       # blank line: this table ended
            vals = [c.strip() for c in nxt.split("|")]
            if col >= len(vals):
                continue
            cell = vals[col]
            if not cell or set(cell) <= set("-+ "):
                continue                    # the |------| separator row
            try:
                return float(cell)
            except ValueError:
                return None
    return None


def resolve_clock_expectation(*, agfi=None, run_recipe=None, mhz=None,
                              tol=None, slot: int = 0) -> dict:
    """{'mhz','tol','fix','keyed','why'} — the expectation the gate enforces.

    Precedence (each rung REFUSES rather than guessing, via ClockKeyRefusal):
      * agfi given            -> clock_key.expected_clock (image-keyed; an
                                 explicit mhz that disagrees is a refusal —
                                 conflicting expectations mean caller
                                 confusion, not a preference).
      * run_recipe WITHOUT agfi -> refusal. The overclock legality check
                                 needs the image's constrained recipe; a bare
                                 recipe override is exactly the knob that
                                 would fly an A2 image at 62.5.
      * neither               -> the unkeyed A2 default (every pre-2026-08-05
                                 image), disclosed as unkeyed in `why`.
    """
    if agfi:
        e = clock_key.expected_clock(agfi, run_recipe=run_recipe)
        if mhz is not None and abs(float(mhz) - e["mhz"]) > 1e-9:
            raise clock_key.ClockKeyRefusal(
                f"conflicting clock expectations: caller passed "
                f"mhz={mhz} but image {agfi} keys to {e['mhz']:g} MHz "
                f"({e['why']}) — refusing to guess which one is meant")
        return {"mhz": e["mhz"], "tol": e["tol"] if tol is None else tol,
                "fix": clock_key.recipe_cmd(slot, e["recipe_cmd_a"],
                                            sudo=True),
                "keyed": True, "why": e["why"]}
    if run_recipe is not None:
        raise clock_key.ClockKeyRefusal(
            f"run_recipe={run_recipe!r} was given WITHOUT an image key "
            f"(agfi) — the overclock-legality check is impossible without "
            f"knowing what the netlist was constrained at. Pass --agfi / "
            f"$APEX_F2_AGFI as well.")
    return {"mhz": TILE_CLK_MHZ if mhz is None else float(mhz),
            "tol": TILE_CLK_TOL if tol is None else tol,
            "fix": RECIPE_FIX, "keyed": False,
            "why": ("UNKEYED default expectation: recipe A2 = "
                    f"{TILE_CLK_MHZ:g} MHz (every image flown before "
                    "2026-08-05). An A0 image must be flown keyed: pass "
                    "agfi= / --agfi / $APEX_F2_AGFI "
                    "(docs/design/CLOCK_LADDER.md §7.2)")}


def check_tile_clock(host, key=None, *, user: str = "ubuntu", slot: int = 0,
                     ssh: str = "ssh", ssh_opts=DEFAULT_SSH_OPTS,
                     sudo: bool = True, timeout_s: int = 120,
                     mhz: float | None = None, tol: float | None = None,
                     agfi: str | None = None, run_recipe=None,
                     verbose: bool = False) -> dict:
    """Read the tile clock on the instance. Returns
    {'ok','a1_mhz','why','rc','log','argv','fix','expect_mhz'}.

    THIS, not f2_host_run.py's substring-"lock" preflight, is the real check.
    The expectation is image-keyed when `agfi` is given (clock_key.py); a
    clock-key refusal returns ok=False WITHOUT contacting the instance —
    nothing can be attested against an unknown expectation.
    """
    try:
        exp = resolve_clock_expectation(agfi=agfi, run_recipe=run_recipe,
                                        mhz=mhz, tol=tol, slot=slot)
    except clock_key.ClockKeyRefusal as e:
        return {"ok": False, "a1_mhz": None, "rc": None, "log": "",
                "argv": [], "fix": RECIPE_FIX, "expect_mhz": None,
                "why": f"clock expectation NOT ESTABLISHED (nothing was "
                       f"contacted): {e}"}
    mhz, tol, fix = exp["mhz"], exp["tol"], exp["fix"]
    cmd = f"fpga-describe-clkgen -S {int(slot)}"
    if sudo:
        cmd = "sudo -n " + cmd
    log_parts: list = []
    argv = _ssh_argv(ssh, key, ssh_opts, _userhost(host, user), cmd)
    rc, out, timed_out = _run_local(argv, timeout_s, "clkgen probe", log_parts,
                                    verbose)
    a1 = parse_clkgen_a1(out)
    log = "\n".join(log_parts) + "\n"
    base = {"rc": rc, "log": log, "argv": argv, "fix": fix,
            "expect_mhz": mhz}
    if timed_out:
        return {**base, "ok": False, "a1_mhz": None,
                "why": f"`{cmd}` did not answer within {timeout_s}s — the "
                       f"instance is unreachable or wedged"}
    if a1 is None:
        return {**base, "ok": False, "a1_mhz": None,
                "why": f"`{cmd}` printed no parsable {CLKGEN_A1} frequency "
                       f"(rc={rc}) — cannot attest the tile clock. Output: "
                       f"{out.strip()[:240]!r}"}
    if abs(a1 - mhz) > tol:
        ratio = (a1 / mhz) if mhz else float("inf")
        return {**base, "ok": False, "a1_mhz": a1,
                "why": f"{CLKGEN_A1} = {a1:.2f} MHz, want {mhz:.3f} ± {tol} "
                       f"({ratio:.2f}× the expected clock; {exp['why']}). "
                       f"An AFI load resets the MMCMs to the A1 default "
                       f"(125 MHz). Program it: {fix}"}
    return {**base, "ok": True, "a1_mhz": a1,
            "why": f"{CLKGEN_A1} = {a1:.2f} MHz (want {mhz:.3f} ± {tol}; "
                   f"{exp['why']})"}


def _refusal_banner(why: str, host: str, fix: str = RECIPE_FIX) -> str:
    bar = "=" * 72
    return (f"\n{bar}\n"
            f"REFUSED TO EXECUTE ON HARDWARE — tile clock not verified\n"
            f"  instance : {host}\n"
            f"  reason   : {why}\n"
            f"  why it matters: over its closed clock the tile COMPUTES\n"
            f"    GARBAGE while every gate stays green — f2_host_run.py's\n"
            f"    MMCM preflight (f2_host_run.py:49-73) matches the substring\n"
            f"    \"lock\", which `fpga-describe-clkgen`'s own \"Clock Group\"\n"
            f"    table satisfies, so it passes VACUOUSLY\n"
            f"    (docs/results/f2_stage2_hw/RESULT.md:64-68), and a `cap`\n"
            f"    without a baked expectation never fails a check\n"
            f"    (f2_host_run.py:218-236). NOTHING was shipped or run.\n"
            f"  fix      : {fix}   (re-run after EVERY AFI load)\n"
            f"{bar}")


# ═════════════════════════════ the remote executor ══════════════════════════

def _result(**kw) -> dict:
    """Assemble the run_job contract dict (+ remote extras)."""
    res = {"captures": [], "rc": RC_TRANSPORT, "log": "", "ok": False,
           "n_reported": None, "summary": None, "cap_out": "", "argv": [],
           "executor": "hw", "notes": [], "timed_out": False,
           # remote-only extras (namespaced; run_job never sets these)
           "remote": True, "host": "", "remote_argv": [], "remote_cmd": "",
           "remote_cap_out": "", "clkgen_a1_mhz": None, "refused": False}
    res.update(kw)
    missing = [k for k in CONTRACT_KEYS if k not in res]
    assert not missing, f"result is missing contract keys {missing}"
    return res


def run_job_remote(regops_path, host, key, remote_dir: str = "~/apexrun",
                   slot: int = 0, cap_out=None, timeout_s: int = 900,
                   python: str = "python3", verify_clock: bool = True,
                   *, user: str = "ubuntu", sudo: bool = True,
                   remote_runner=None, ship_runner: bool = True,
                   extra_args=(), skip_clkgen_wait: bool = False,
                   remote_env=None, ssh: str = "ssh", scp: str = "scp",
                   ssh_opts=DEFAULT_SSH_OPTS, require_summary: bool = True,
                   ctl_timeout_s: int = 120, clock_mhz: float | None = None,
                   clock_tol: float | None = None, agfi: str | None = None,
                   run_recipe=None, keep_remote: bool = True,
                   verbose: bool = False) -> dict:
    """Execute regops on the REAL FPGA over ssh; return run_job's dict shape.

    regops_path : one path, or a sequence of paths (f2_host_run takes many;
                  `i` in the cap file is global over the whole run).
    host        : 'ubuntu@1.2.3.4' or '1.2.3.4' (then `user` is prepended).
    key         : ssh private key (…/apex-f2.pem); None uses the agent.
    remote_dir  : working dir ON the instance ('~/…' is fine); created.
    cap_out     : LOCAL landing path for the fetched capture file (default: a
                  temp file). The remote file is <remote_dir>/<its basename>.
    timeout_s   : wall clock for the job itself; ctl_timeout_s bounds the
                  clock probe / uploads / fetch.
    remote_runner : path to an f2_host_run.py ALREADY on the instance. Default
                  None ships THIS tree's copy (ship_runner=True) so the
                  executed runner is the one with the per-file TILE_RST fix
                  and --cap-out; an older clone silently lacks both.
    sudo        : BAR0 needs root; the runner is invoked as `sudo -n …` (-n so
                  a sudoers password requirement fails loudly, never hangs).
    remote_env  : {K: V} rendered as `env K=V …` INSIDE the sudo call. Needed
                  only if the SDK is not at /home/ubuntu/aws-fpga: under sudo
                  `~` is /root and sudo's env_reset drops AWS_FPGA_REPO_DIR,
                  so f2_host_run.py:95-101 would fall through to its hardcoded
                  /home/ubuntu/aws-fpga and then ABORT on the bindings import.
    agfi        : the image that is flying — KEYS the clock gate's expected
                  a1 MHz to the recipe that image was constrained at
                  (clock_key.py; env $APEX_F2_AGFI via remote_config). Without
                  it the gate enforces the unkeyed A2 default and says so.
    run_recipe  : deliberate underclock arm (e.g. 2 to fly an A0 image at
                  15.625 for the §7.3 A/B) — needs agfi; an overclock or a
                  bare run_recipe REFUSES before any contact.

    Raises CaptureEgressError if the runner printed no "F2HOST CAPTURES:" line
    (same as the sim path). Returns ok=False with a note if the clock check
    refuses, if rc!=0, if n disagrees, or on timeout.
    """
    # $APEX_F2_CTL_TIMEOUT overrides the control-op bound (mkdir/upload/
    # fetch/cleanup) without touching every caller.
    ctl_timeout_s = int(os.environ.get("APEX_F2_CTL_TIMEOUT", ctl_timeout_s))
    # ── argument normalization (mirrors run_job:210-217) ────────────────────
    regops = [regops_path] if isinstance(regops_path, (str, os.PathLike)) \
        else list(regops_path)
    regops = [str(Path(r)) for r in regops]
    if not regops:
        raise ValueError("no regops files given")
    for r in regops:
        if not Path(r).is_file():
            raise FileNotFoundError(f"regops file not found: {r}")
    names = [Path(r).name for r in regops]
    if len(set(names)) != len(names):
        raise ValueError(f"regops basenames collide on the remote dir: {names}")

    tmp_made = False
    if cap_out is None:
        fd, cap_out = tempfile.mkstemp(prefix="apex_cap_hw_", suffix=".jsonl")
        os.close(fd)
        tmp_made = True
    cap_out = str(Path(str(cap_out)).expanduser())
    Path(cap_out).parent.mkdir(parents=True, exist_ok=True)
    # A stale LOCAL file must never be read as this run's output (run_job:226).
    try:
        os.unlink(cap_out)
    except FileNotFoundError:
        pass

    uh = _userhost(host, user)
    cap_name = Path(cap_out).name
    remote_cap = _rjoin(remote_dir, cap_name)
    runner_word = str(remote_runner) if remote_runner \
        else "./" + HOST_RUNNER.name
    remote_argv = build_remote_argv(
        [f"./{n}" for n in names], slot=slot, cap_name=f"./{cap_name}",
        python=python, runner=runner_word, extra_args=extra_args,
        skip_clkgen_wait=skip_clkgen_wait)
    log_parts: list = []
    notes: list = []

    # ── 1. THE CLOCK GATE — before a single byte is shipped ─────────────────
    a1 = None
    if verify_clock:
        clk = check_tile_clock(host, key, user=user, slot=slot, ssh=ssh,
                               ssh_opts=ssh_opts, sudo=sudo,
                               timeout_s=ctl_timeout_s, mhz=clock_mhz,
                               tol=clock_tol, agfi=agfi,
                               run_recipe=run_recipe, verbose=verbose)
        log_parts.append(clk["log"].rstrip("\n"))
        a1 = clk["a1_mhz"]
        if not clk["ok"]:
            fix = clk.get("fix", RECIPE_FIX)
            eprint(_refusal_banner(clk["why"], uh, fix))
            note = (f"REFUSED (clock gate): {clk['why']} — nothing was "
                    f"shipped or executed. f2_host_run.py's own MMCM "
                    f"preflight would have passed vacuously here "
                    f"(f2_host_run.py:49-73). Fix: {fix}")
            return _result(
                captures=[], rc=RC_CLOCK_REFUSED,
                log="\n".join(log_parts) + "\n"
                    + _refusal_banner(clk["why"], uh, fix) + "\n",
                ok=False, n_reported=None, summary=None, cap_out=cap_out,
                argv=clk["argv"], notes=[note], timed_out=False, host=uh,
                remote_argv=remote_argv, remote_cap_out=remote_cap,
                clkgen_a1_mhz=a1, refused=True)
        log_parts.append(f"[remote_hw_exec] clock gate PASS: {clk['why']}")
        # The runner's OWN preflight (f2_host_run.py:49-73) then shells out to
        # `fpga-describe-clkgen` again, once per invocation, and accepts any
        # output containing the substring "lock" — which "Clock Group A"
        # satisfies, so it passes VACUOUSLY (RESULT.md:64-68). We have just
        # read the SAME tool and checked the frequency BY NUMBER, which is
        # strictly stronger and covers the same precondition (a locked MMCM
        # is what makes 15.62 MHz readable at all). So skip the duplicate:
        # one fewer remote subprocess per invocation, no check lost.
        # $APEX_F2_KEEP_CLKGEN_WAIT=1 restores it.
        if not os.environ.get("APEX_F2_KEEP_CLKGEN_WAIT"):
            skip_clkgen_wait = True
            notes.append(
                "the runner's substring-'lock' clkgen preflight was skipped: "
                "this invocation's NUMERIC clock gate already read "
                f"{CLKGEN_A1} = {a1} MHz from the same tool "
                "(APEX_F2_KEEP_CLKGEN_WAIT=1 restores the duplicate)")
            remote_argv = build_remote_argv(
                [f"./{n}" for n in names], slot=slot, cap_name=f"./{cap_name}",
                python=python, runner=runner_word, extra_args=extra_args,
                skip_clkgen_wait=True)
    else:
        notes.append("tile clock NOT verified (verify_clock=False) — a run at "
                     "the default recipe A1 (125 MHz) computes garbage and "
                     "still exits 0; do not publish this as evidence")

    # ── 2. make the remote dir, then upload (regops + the runner itself) ────
    up = list(regops)
    if remote_runner is None and ship_runner:
        if not HOST_RUNNER.is_file():
            raise FileNotFoundError(f"host runner missing: {HOST_RUNNER}")
        up.append(str(HOST_RUNNER))
    up_bytes = sum(os.path.getsize(p) for p in up)
    up_timeout = max(ctl_timeout_s, 60 + len(up) // 4,
                     int(up_bytes / 250_000))     # >= 250 kB/s of headroom
    # Large batches upload as ONE tar stream. scp's per-file protocol
    # exchange costs a WAN round trip per file: a 512-job batch (1025 files)
    # blew the 120 s control timeout on 2026-07-30 and NOTHING ran (caught
    # by batch_exec's zero-marker refusal, reported as UPLOAD FAILED rc=124).
    # Refusal semantics are identical: any nonzero rc below refuses to
    # execute. APEX_F2_SCP_UPLOAD=1 forces the old path.
    use_tar = len(up) > 16 and not os.environ.get("APEX_F2_SCP_UPLOAD")
    # The tar lands in the LOGIN dir and its extraction command carries the
    # `mkdir -p` itself, so the tar path needs no separate mkdir round trip:
    # 4 connections per invocation instead of 5. The scp path still needs the
    # directory to exist before scp writes into it.
    rc = 0
    if not use_tar:
        rc, _, _ = _run_local(
            _ssh_argv(ssh, key, ssh_opts, uh, f"mkdir -p {_rq(remote_dir)}"),
            ctl_timeout_s, "mkdir", log_parts, verbose)
    if rc != 0 and ssh_mux_opts():
        # A STALE control socket is the one failure multiplexing can add:
        # tear it down and retry ONCE on a fresh connection before deciding
        # the instance is unreachable. (Only the mkdir gets this: it is the
        # first op through the socket, so a live socket here means a live
        # socket for the upload/job/fetch that follow.)
        log_parts.append("[remote_hw_exec] mkdir failed with ssh multiplexing "
                         "on — closing the control socket and retrying once")
        ssh_mux_close(host, key, user=user, ssh=ssh, ssh_opts=ssh_opts)
        rc, _, _ = _run_local(
            _ssh_argv(ssh, key, ssh_opts, uh, f"mkdir -p {_rq(remote_dir)}",
                      mux=False),
            ctl_timeout_s, "mkdir(retry, no mux)", log_parts, verbose)
    if rc != 0:
        notes.append(f"could not create {remote_dir} on {uh} (rc={rc})")
        return _result(rc=rc or RC_TRANSPORT, log="\n".join(log_parts) + "\n",
                       cap_out=cap_out, argv=[], notes=notes, host=uh,
                       remote_argv=remote_argv, remote_cap_out=remote_cap,
                       clkgen_a1_mhz=a1)

    dst = (f"{uh}:" if use_tar
           else f"{uh}:{_scp_path(remote_dir).rstrip('/')}/")
    if use_tar:
        # GZIP, not a bare tar. regops JSONL is the most compressible thing
        # this system produces — a repeating {"op":"w","a":..,"d":..} skeleton
        # around hex weight beats — and the wire, not the CPU, is the scarce
        # resource from this Mac to us-west-2: a full-width 0.5B layer emits
        # hundreds of MB of programs. gzip -6 is ~50 MB/s locally and the
        # measured ratio on real projection programs is ~6-8x
        # (fatproof.py reports it per run). $APEX_F2_NO_GZIP=1 falls back.
        import tarfile
        gz = not os.environ.get("APEX_F2_NO_GZIP")
        sfx = ".apexup.tar.gz" if gz else ".apexup.tar"
        tfd = tempfile.NamedTemporaryFile(suffix=sfx, delete=False)
        tfd.close()
        try:
            with tarfile.open(tfd.name, "w:gz" if gz else "w") as t:
                seen = set()
                for p in up:
                    bn = os.path.basename(p)
                    assert bn not in seen, \
                        f"duplicate upload basename {bn} — the flat remote " \
                        f"dir cannot hold both"
                    seen.add(bn)
                    t.add(p, arcname=bn)
            tar_bytes = os.path.getsize(tfd.name)
            log_parts.append(
                f"[remote_hw_exec] upload: {len(up)} files, "
                f"{up_bytes / 1e6:.1f} MB -> {tar_bytes / 1e6:.1f} MB "
                f"{'gzipped ' if gz else ''}tar "
                f"({up_bytes / max(tar_bytes, 1):.1f}x)")
            rc, _, _ = _run_local(
                _scp_argv(scp, key, ssh_opts, [tfd.name], dst),
                up_timeout, "upload(tar)", log_parts, verbose)
            if rc == 0:
                tarname = os.path.basename(tfd.name)
                rc, _, _ = _run_local(
                    _ssh_argv(ssh, key, ssh_opts, uh,
                              f"mkdir -p {_rq(remote_dir)} && "
                              f"tar x{'z' if gz else ''}f "
                              f"{shlex.quote(tarname)} "
                              f"-C {_rq(remote_dir)} && "
                              f"rm -f {shlex.quote(tarname)}"),
                    up_timeout, "mkdir+untar", log_parts, verbose)
        finally:
            os.unlink(tfd.name)
    else:
        rc, _, _ = _run_local(_scp_argv(scp, key, ssh_opts, up, dst),
                              up_timeout, "upload", log_parts, verbose)
    if rc != 0:
        # Never run on a stale input: an old file of the same name may sit in
        # remote_dir from a previous job.
        notes.append(f"UPLOAD FAILED (rc={rc}) — refusing to execute, the "
                     f"remote dir may still hold a PREVIOUS job's "
                     f"{names[0]}; nothing was run")
        return _result(rc=rc or RC_TRANSPORT, log="\n".join(log_parts) + "\n",
                       cap_out=cap_out, argv=[], notes=notes, host=uh,
                       remote_argv=remote_argv, remote_cap_out=remote_cap,
                       clkgen_a1_mhz=a1)

    # ── 3. execute. The remote shell echoes the runner's rc: the command ends
    #      in `echo`, so ssh's own rc is 0 by construction and is NOT usable.
    inner = " ".join(shlex.quote(a) for a in remote_argv)
    if remote_env:
        inner = "env " + " ".join(
            f"{k}={shlex.quote(str(v))}" for k, v in remote_env.items()
        ) + " " + inner
    if sudo:
        inner = "sudo -n " + inner
    command = (f"cd {_rq(remote_dir)} && rm -f {shlex.quote('./' + cap_name)} "
               f"&& {{ {inner}; echo \"{SENTINEL}$?\"; }}")
    argv = _ssh_argv(ssh, key, ssh_opts, uh, command)
    ssh_rc, run_out, timed_out = _run_local(argv, timeout_s, "job", log_parts,
                                            verbose)

    m = SENTINEL_RE.search(run_out)
    sent = int(m.group(1)) if m else None
    if sent is None:
        rc = ssh_rc if ssh_rc else RC_TRANSPORT
        if not timed_out:
            notes.append(f"no {SENTINEL} sentinel in the remote output "
                         f"(ssh rc={ssh_rc}) — the ssh transport or the "
                         f"remote setup failed, so the runner's own exit "
                         f"status is unknown; treating the run as FAILED")
    else:
        rc = sent
        if ssh_rc != 0:
            notes.append(f"ssh exited {ssh_rc} although the runner reported "
                         f"rc={sent} — the session was truncated; treating "
                         f"the run as FAILED")
            rc = ssh_rc
    if timed_out:
        rc = RC_TIMEOUT

    # ── 4. fetch the capture file ───────────────────────────────────────────
    fetch_rc, _, _ = _run_local(
        _scp_argv(scp, key, ssh_opts, [f"{uh}:{_scp_path(remote_cap)}"],
                  cap_out), ctl_timeout_s, "fetch cap", log_parts, verbose)
    if fetch_rc != 0:
        notes.append(f"could not fetch {remote_cap} from {uh} (scp rc="
                     f"{fetch_rc}) — the runner never wrote it, or the "
                     f"transport failed")

    if not keep_remote:
        _run_local(_ssh_argv(ssh, key, ssh_opts, uh,
                             "rm -f " + " ".join(
                                 _rq(_rjoin(remote_dir, n)) for n in
                                 names + [cap_name])),
                   ctl_timeout_s, "cleanup", log_parts, verbose)

    # ── 5. the same paranoia the sim path applies (run_job:258-283) ─────────
    log = "\n".join(p for p in log_parts if p is not None) + "\n"
    sm = None
    for sm in SUMMARY_RE.finditer(log):        # last one wins
        pass
    n_reported = int(sm.group("n")) if sm else None
    summary = sm.group(0).strip() if sm else None

    caps = parse_cap_file(cap_out, strict=not timed_out and require_summary)
    notes += audit_captures(caps)
    if timed_out:
        notes.insert(0, f"TIMEOUT after {timeout_s}s — captures are PARTIAL "
                        f"and the remote runner may STILL BE RUNNING; the "
                        f"next job's per-file TILE_RST (f2_host_run.py:146) "
                        f"is what makes a retry safe")
    if sm and sm.group("out") not in (f"./{cap_name}", remote_cap, cap_name):
        # NB: compare against the REMOTE path — the runner never saw cap_out.
        notes.append(f"summary out={sm.group('out')} != the requested remote "
                     f"path ./{cap_name} in {remote_dir}")
    if n_reported is not None and n_reported != len(caps):
        notes.append(f"summary n={n_reported} != {len(caps)} records parsed")

    if summary is None and require_summary and not timed_out:
        raise CaptureEgressError(
            f"the remote runner printed no 'F2HOST CAPTURES:' line — "
            f"--cap-out was dropped, the runner on {uh} is an OLD copy "
            f"without the capture-egress channel, or it aborted before the "
            f"summary. rc={rc} (rc is NOT evidence: cap mismatches never "
            f"reach total_fails, f2_host_run.py:266). remote argv="
            f"{remote_argv}\n{log[-2000:]}")

    ok = (rc == 0 and summary is not None and n_reported == len(caps)
          and not timed_out)
    return _result(captures=caps, rc=rc, log=log, ok=ok,
                   n_reported=n_reported, summary=summary, cap_out=cap_out,
                   argv=argv, notes=notes, timed_out=timed_out, host=uh,
                   remote_argv=remote_argv, remote_cmd=command,
                   remote_cap_out=remote_cap, clkgen_a1_mhz=a1,
                   tmp_cap_out=tmp_made)


# ═════════════ the tiny helper prompt_offload dispatches through ════════════
# See the header for the exact 2-line change. attach() is deliberately a no-op
# when no host is configured, so nothing about the sim path can change.

ENV = {"host": "APEX_F2_HOST", "key": "APEX_F2_KEY",
       "remote_dir": "APEX_F2_REMOTE_DIR", "user": "APEX_F2_USER",
       "python": "APEX_F2_PY", "remote_runner": "APEX_F2_RUNNER",
       "agfi": "APEX_F2_AGFI", "run_recipe": "APEX_F2_RUN_RECIPE"}


def remote_config(args=None, **over) -> dict | None:
    """Gather the remote config from $APEX_F2_* (and optional argparse attrs).

    Returns None when no host is configured — the signal to leave the sim path
    completely alone.
    """
    cfg = {"host": None, "key": None, "remote_dir": "~/apexrun",
           "user": "ubuntu", "python": "python3", "remote_runner": None,
           "sudo": True, "verify_clock": True, "ssh_opts": DEFAULT_SSH_OPTS,
           "remote_env": None, "agfi": None, "run_recipe": None}
    for k, envname in ENV.items():
        v = os.environ.get(envname)
        if v:
            cfg[k] = v
    if os.environ.get("APEX_F2_REMOTE_ENV"):
        cfg["remote_env"] = dict(
            kv.split("=", 1)
            for kv in shlex.split(os.environ["APEX_F2_REMOTE_ENV"]))
    if os.environ.get("APEX_F2_NO_SUDO"):
        cfg["sudo"] = False
    if os.environ.get("APEX_F2_NO_CLOCK_CHECK"):
        cfg["verify_clock"] = False
    if os.environ.get("APEX_F2_SSH_OPTS"):
        cfg["ssh_opts"] = tuple(shlex.split(os.environ["APEX_F2_SSH_OPTS"]))
    for k in list(cfg):                       # argparse attrs win over env
        v = getattr(args, "hw_" + k, None) if args is not None else None
        if v not in (None, ""):
            cfg[k] = v
    cfg.update({k: v for k, v in over.items() if v is not None})
    return cfg if cfg["host"] else None


def attach(bridge_module=None, args=None, **over) -> bool:
    """Route bridge.run_job(executor='hw') to run_job_remote().

    Returns True if the shim was installed. Idempotent. Everything except
    executor='hw' falls through to the original run_job untouched.
    """
    mod = bridge_module or bridge
    cfg = remote_config(args, **over)
    if cfg is None:
        return False
    orig = getattr(mod, "run_job")
    if getattr(orig, "_apex_remote_shim", False):
        orig._apex_cfg.update(cfg)            # re-point, do not double-wrap
        return True

    def _dispatch(regops_path, executor: str = "sim", binary=None,
                  cap_out=None, timeout_s: int = 1800, *,
                  tile_div: int = bridge.TILE_DIV_DEFAULT, slot: int = 0,
                  extra_args=(), cwd=None, env=None,
                  require_summary: bool = True, keep_cap_out: bool = True):
        if executor != "hw":
            return orig(regops_path, executor=executor, binary=binary,
                        cap_out=cap_out, timeout_s=timeout_s,
                        tile_div=tile_div, slot=slot, extra_args=extra_args,
                        cwd=cwd, env=env, require_summary=require_summary,
                        keep_cap_out=keep_cap_out)
        c = _dispatch._apex_cfg
        # tile_div is a SIM plusarg (+tile_div) and has no hardware meaning:
        # on silicon the tile runs at the clkgen recipe, which is exactly what
        # the clock gate verifies. Dropped on purpose.
        return run_job_remote(
            regops_path, c["host"], c["key"], remote_dir=c["remote_dir"],
            slot=slot, cap_out=cap_out, timeout_s=timeout_s,
            python=(str(binary) if binary else c["python"]),
            verify_clock=c["verify_clock"], user=c["user"], sudo=c["sudo"],
            remote_runner=c["remote_runner"], extra_args=extra_args,
            remote_env=c["remote_env"], ssh_opts=c["ssh_opts"],
            agfi=c.get("agfi"), run_recipe=c.get("run_recipe"),
            require_summary=require_summary)

    _dispatch._apex_remote_shim = True
    _dispatch._apex_cfg = dict(cfg)
    _dispatch._apex_orig = orig
    mod.run_job = _dispatch
    eprint(f"[remote_hw_exec] executor 'hw' -> "
           f"{_userhost(cfg['host'], cfg['user'])}:{cfg['remote_dir']} "
           f"(clock gate {'ON' if cfg['verify_clock'] else 'OFF — UNSAFE'}"
           + (f", keyed to {cfg['agfi']}" if cfg.get("agfi")
              else ", UNKEYED: A2 default") + ")")
    return True


def detach(bridge_module=None) -> bool:
    mod = bridge_module or bridge
    f = getattr(mod, "run_job", None)
    if getattr(f, "_apex_remote_shim", False):
        mod.run_job = f._apex_orig
        return True
    return False


# ══════════════════════════════ selftest ════════════════════════════════════
# No instance, no spend, no Verilator: `ssh`/`scp` are replaced by local stubs
# and f2_host_run.py by a stub that mirrors its argparse spec and its capture
# egress. The stubs execute the remote command in a REAL shell (`sh -c`), so
# the quoting, the `cd`, the `rm -f` and the rc sentinel are exercised for
# real. Case [0] is the authority on argv/interface parity: it runs the argv
# my builder produces through the REAL f2_host_run.py.

_STUB_RUNNER = r'''
import argparse, json, os, sys
# argparse spec MIRRORS f2_host_run.py:78-88. (The real parser is exercised by
# selftest case [0]; this stub only emulates the runner's EFFECTS.)
ap = argparse.ArgumentParser()
ap.add_argument("regops", nargs="+")
ap.add_argument("--slot", type=int, default=0)
ap.add_argument("--stop-on-fail", action="store_true")
ap.add_argument("--skip-clkgen-wait", action="store_true")
ap.add_argument("--cap-out", metavar="PATH")
a = ap.parse_args()
mode = os.environ.get("APEX_STUB_MODE", "ok")
json.dump({"regops": a.regops, "slot": a.slot, "cap_out": a.cap_out,
           "stop_on_fail": a.stop_on_fail, "skip": a.skip_clkgen_wait,
           "cwd": os.getcwd(), "argv": sys.argv, "uid": os.getuid()},
          open(os.environ["APEX_STUB_LOG"], "w"))
for r in a.regops:                       # the upload really has to have landed
    if not os.path.isfile(r):
        print("STUB ABORT: regops file absent: %s (cwd=%s)" % (r, os.getcwd()))
        sys.exit(3)
n = 0
if a.cap_out and mode != "nofile":
    with open(a.cap_out, "w") as f:      # TRUNCATE at run start
        for i in range(3):
            for w in range(2):
                f.write(json.dumps({
                    "tag": "ro_w%d_%d" % (w, i * 2 + w),
                    "addr": "0x%08x" % (0x3204 + 4 * w), "mask": "0xffffffff",
                    "value": (0xFFFFFFF0 + i) if w else (i + 1),
                    "i": i * 2 + w}, separators=(",", ":")) + "\n")
                n += 1
print("[%s] 6 ops, 0 checks, %d fails" % (a.regops[0], 1 if mode == "fail" else 0))
print("F2HOST RESULT: files=%d checks=0 fails=%d -> %s"
      % (len(a.regops), 1 if mode == "fail" else 0,
         "FAIL" if mode == "fail" else "PASS"))
if mode != "nosummary":
    print("F2HOST CAPTURES: n=%d out=%s"
          % (n + (1 if mode == "badn" else 0), a.cap_out))
sys.exit(1 if mode == "fail" else 0)
'''

_FAKE_SSH = r'''
import json, os, subprocess, sys
argv = sys.argv[1:]
cmd = argv[-1] if argv else ""
with open(os.environ["APEX_FAKE_SSH_LOG"], "a") as f:
    f.write(json.dumps({"argv": argv, "cmd": cmd}) + "\n")
if "fpga-describe-clkgen" in cmd:
    a1 = os.environ.get("APEX_FAKE_A1", "15.62")
    if a1 == "garbage":                      # tool present, output unusable
        print("Error: (5) FPGA slot 0 has no clkgen")
        sys.exit(1)
    print("Clock Group A Frequency (Mhz)")
    print("| clk_extra_a1 | clk_extra_a2 | clk_extra_a3 |")
    print("|--------------|--------------|--------------|")
    print("|      %s   |     125.00   |      62.50   |" % a1)
    print("")
    print("Clock Group B Frequency (Mhz)")
    print("| clk_extra_b0 | clk_extra_b1 |")
    print("|--------------|--------------|")
    print("|       0.00   |       0.00   |")
    sys.exit(0)
# mkdir / the job / cleanup: run it in a REAL shell so quoting, cd, rm -f and
# the rc sentinel are all exercised, STARTING IN THE LOGIN DIR like a real
# ssh session (that is where an uploaded tar lands). No root here, so drop
# the sudo prefix.
home = os.environ.get("APEX_FAKE_HOME") or os.getcwd()
os.makedirs(home, exist_ok=True)
sys.exit(subprocess.run(["sh", "-c", cmd.replace("sudo -n ", "", 1)],
                        cwd=home).returncode)
'''

_FAKE_SCP = r'''
import os, shutil, sys
OPTS_WITH_ARG = ("-i", "-o", "-F", "-l", "-P", "-c", "-J", "-S")
files, args, i = [], sys.argv[1:], 0
while i < len(args):
    a = args[i]
    if a.startswith("-"):
        i += 2 if a in OPTS_WITH_ARG else 1
        continue
    files.append(a); i += 1
def loc(p):
    h, sep, rest = p.partition(":")          # 'host:/abs' -> '/abs'
    if sep and "/" not in h:                 # 'host:' -> the LOGIN dir
        return rest or (os.environ.get("APEX_FAKE_HOME") or ".")
    return p
dst, srcs = loc(files[-1]), [loc(s) for s in files[:-1]]
for s in srcs:
    if not os.path.exists(s):
        sys.stderr.write("fake_scp: %s: No such file or directory\n" % s)
        sys.exit(1)
    if not (os.path.isdir(dst) or dst.endswith("/")) and len(srcs) > 1:
        sys.stderr.write("fake_scp: %s: Not a directory\n" % dst)
        sys.exit(1)
    shutil.copy(s, dst)
sys.exit(0)
'''


def _selftest() -> int:                                   # noqa: C901
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="apex_remote_st_"))
    fails = 0

    def bad(tag, msg):
        nonlocal fails
        fails += 1
        print(f"  [{tag}] FAIL: {msg}")

    def ok(tag, msg):
        print(f"  [{tag}] ok  {msg}")

    try:
        rroot = tmp / "remote"                 # stands in for ~/apexrun
        (tmp / "local").mkdir()
        stub = tmp / "stub_runner.py"
        stub.write_text("#!/usr/bin/env python3\n" + _STUB_RUNNER)
        fssh = tmp / "fake_ssh.py"
        fssh.write_text("#!/usr/bin/env python3\n" + _FAKE_SSH)
        fscp = tmp / "fake_scp.py"
        fscp.write_text("#!/usr/bin/env python3\n" + _FAKE_SCP)
        for f in (stub, fssh, fscp):
            os.chmod(f, 0o755)
        sshlog = tmp / "ssh.log"
        stublog = tmp / "stub.json"
        os.environ["APEX_FAKE_SSH_LOG"] = str(sshlog)
        os.environ["APEX_STUB_LOG"] = str(stublog)
        # the stand-in for the instance's LOGIN dir: where `scp x host:`
        # lands and where an ssh command starts.
        (tmp / "home").mkdir()
        os.environ["APEX_FAKE_HOME"] = str(tmp / "home")
        regops = tmp / "local" / "poff_s000_L00_h00.regops.jsonl"
        regops.write_text(json.dumps({"op": "note", "s": "selftest"}) + "\n")
        capf = tmp / "local" / "poff_s000_L00_h00.cap.jsonl"

        def reset(a1="15.62", mode="ok"):
            for p in (sshlog, stublog, capf):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
            os.environ["APEX_FAKE_A1"] = a1
            os.environ["APEX_STUB_MODE"] = mode

        def sshlines():
            if not sshlog.exists():
                return []
            return [json.loads(x) for x in sshlog.read_text().splitlines() if x]

        def go(**kw):
            base = dict(regops_path=regops, host="ubuntu@10.0.0.9",
                        key=None, remote_dir=str(rroot), cap_out=str(capf),
                        ssh=str(fssh), scp=str(fscp),   # executable stubs
                        ssh_opts=("-o", "BatchMode=yes"),
                        remote_runner=str(stub), python=sys.executable,
                        timeout_s=120, ctl_timeout_s=60)
            base.update(kw)
            return run_job_remote(**base)

        # ── [0] INTERFACE PARITY: the argv my builder makes, through the REAL
        #        f2_host_run.py. argparse would exit 2 with "usage:" on a bad
        #        flag; instead it must get past the parser and stop at the kit
        #        bindings import (f2_host_run.py:102-107).
        av = build_remote_argv([f"./{regops.name}"], slot=0,
                               cap_name="./probe.cap.jsonl",
                               python=sys.executable,
                               runner=str(HOST_RUNNER))
        pr = subprocess.run(av, capture_output=True, text=True, timeout=120,
                            cwd=str(regops.parent))
        out = (pr.stdout or "") + (pr.stderr or "")
        if "usage:" in out or "unrecognized arguments" in out \
                or "invalid" in out.lower():
            bad(0, f"the REAL f2_host_run.py rejected my argv: {out[:300]}")
        elif "fpga_pci bindings not built" not in out:
            bad(0, f"unexpected output from the real runner (argparse passed "
                   f"but then?): rc={pr.returncode} {out[:300]}")
        else:
            ok(0, f"REAL f2_host_run.py accepted argv "
                  f"{' '.join(av[1:])!r} -> rc={pr.returncode} "
                  f"'{out.strip().splitlines()[0][:52]}…'")

        # ── [1] happy path: captures reach a Python variable
        reset()
        r = go()
        # the ONE expected note is the disclosed clkgen-preflight elision:
        # the numeric gate above replaces the runner's substring check.
        extra = [n for n in r["notes"] if "clkgen preflight" not in n]
        if not (r["ok"] and r["rc"] == 0 and len(r["captures"]) == 6
                and r["n_reported"] == 6 and extra == []
                and len(r["notes"]) == 1):
            bad(1, f"ok={r['ok']} rc={r['rc']} n={len(r['captures'])} "
                   f"reported={r['n_reported']} notes={r['notes']}")
        elif "--skip-clkgen-wait" not in r["remote_argv"]:
            bad(1, f"the elision was announced but not sent: {r['remote_argv']}")
        else:
            c0, c1 = r["captures"][0], r["captures"][1]
            if (c0["tag"], c0["sem"], c0["seq"], c0["addr"], c0["value"]) != \
                    ("ro_w0_0", "ro_w0", 0, 0x3204, 1) \
                    or c1["value"] != 0xFFFFFFF0:
                bad(1, f"capture decode wrong: {c0} {c1}")
            else:
                ok(1, f"n=6 rc=0 summary={r['summary']!r} "
                      f"a1={r['clkgen_a1_mhz']} MHz")
        # the stub really saw the argv I built, in the dir I cd'd to
        s = json.loads(stublog.read_text())
        exp = [f"./{regops.name}"]
        if s["regops"] != exp or s["slot"] != 0 \
                or s["cap_out"] != f"./{capf.name}" \
                or Path(s["cwd"]).resolve() != rroot.resolve():
            bad("1b", f"remote argv/cwd wrong: {s}")
        else:
            ok("1b", f"runner argv={s['argv'][1:]} cwd={s['cwd']}")
        cmds = [x["cmd"] for x in sshlines()]
        job = [c for c in cmds if "stub_runner" in c]
        if not any("sudo -n " in c for c in cmds):
            bad("1c", f"BAR0 needs root but no `sudo -n` was sent: {cmds}")
        elif not job or f"rm -f ./{capf.name} &&" not in job[0]:
            bad("1c", f"stale REMOTE cap file not removed before the run: "
                      f"{job or cmds}")
        elif SENTINEL + "$?" not in job[0]:
            bad("1c", f"no rc sentinel — ssh's own rc is 0 by construction "
                      f"here: {job[0]}")
        else:
            ok("1c", "sudo -n sent; remote cap rm -f'd before the run; rc "
                     "sentinel present")

        # ── [2] THE VACUOUS-PREFLIGHT TRAP: default recipe A1 = 125 MHz must
        #        REFUSE, and must not ship or execute anything.
        reset(a1="125.00")
        r = go()
        cmds = [x["cmd"] for x in sshlines()]
        ran = [c for c in cmds if "f2_host_run" in c or "stub_runner" in c]
        if r["ok"] or not r["refused"] or r["rc"] != RC_CLOCK_REFUSED:
            bad(2, f"125 MHz was ACCEPTED: ok={r['ok']} rc={r['rc']}")
        elif ran:
            bad(2, f"refused but still executed the job: {ran}")
        elif r["captures"] or capf.exists():
            bad(2, "refused but captures/cap file materialized")
        elif "15.6" not in r["notes"][0] or RECIPE_FIX not in r["notes"][0]:
            bad(2, f"refusal note is not actionable: {r['notes']}")
        else:
            ok(2, f"a1=125.00 REFUSED rc={r['rc']} (probe only, {len(cmds)} "
                  f"ssh call) note={r['notes'][0][:64]}…")

        # ── [3] dead clock (0.00) and unparsable output must also refuse
        for tag, a1v in (("3a", "0.00"), ("3b", "garbage")):
            reset(a1=a1v)
            r = go()
            if r["ok"] or not r["refused"]:
                bad(tag, f"a1={a1v} accepted: ok={r['ok']}")
            else:
                ok(tag, f"a1={a1v!r} refused: {r['notes'][0][:60]}…")

        # ── [4] runner FAIL (rc=1) must not be ok, and rc must be the
        #        RUNNER's, not ssh's (the command ends in `echo`, so ssh's rc
        #        is 0 by construction)
        reset(mode="fail")
        r = go()
        if r["ok"] or r["rc"] != 1:
            bad(4, f"expected ok=False rc=1, got ok={r['ok']} rc={r['rc']}")
        else:
            ok(4, f"runner rc=1 recovered through the sentinel (ok={r['ok']})")

        # ── [5] no "F2HOST CAPTURES:" line -> HARD error (rc==0 vacuity trap)
        reset(mode="nosummary")
        try:
            go()
            bad(5, "a run with no capture summary was accepted")
        except CaptureEgressError as e:
            ok(5, f"no-summary raised: {str(e)[:60]}…")

        # ── [6] n mismatch: rc 0, summary present, count lies -> not ok
        reset(mode="badn")
        r = go()
        if r["ok"] or not any("!=" in n for n in r["notes"]):
            bad(6, f"n mismatch accepted: ok={r['ok']} notes={r['notes']}")
        else:
            ok(6, f"rc=0 but ok=False; notes={r['notes']}")

        # ── [7] stale LOCAL cap file must never be returned as this run's
        #        output (the runner writes nothing -> the fetch fails)
        reset(mode="nofile")
        capf.write_text(json.dumps({"tag": "stale_0", "addr": "0x3204",
                                    "mask": "0xffffffff", "value": 7,
                                    "i": 0}) + "\n")
        try:
            r = go()
            bad(7, f"stale local cap file accepted: {r['captures']}")
        except CaptureEgressError as e:
            ok(7, f"stale local cap rejected: {str(e)[:56]}…")

        # ── [8] shipping THIS tree's f2_host_run.py (remote_runner=None):
        #        the upload must land, and a runner that cannot reach the
        #        bindings must NOT look like a pass.
        reset()
        try:
            r = go(remote_runner=None, sudo=False)
            bad(8, f"bindings-less real runner looked ok: ok={r['ok']}")
        except CaptureEgressError as e:
            shipped = rroot / HOST_RUNNER.name
            if not shipped.is_file():
                bad(8, "runner was not shipped to the remote dir")
            elif shipped.read_text() != HOST_RUNNER.read_text():
                bad(8, "shipped runner differs from this tree's copy")
            else:
                ok(8, f"shipped {HOST_RUNNER.name} verbatim; its "
                      f"bindings ABORT raised instead of passing: "
                      f"{str(e)[:44]}…")
        if any("sudo -n" in x["cmd"] for x in sshlines()
               if "stub_runner" in x["cmd"] or "f2_host_run" in x["cmd"]):
            bad("8b", "sudo=False still sent `sudo -n`")
        else:
            ok("8b", "sudo=False sends no sudo")

        # ── [9] verify_clock=False: no probe, loud note, job still runs.
        #        Also the remote_env escape hatch (sudo drops
        #        AWS_FPGA_REPO_DIR; f2_host_run.py:95-101).
        reset()
        r = go(verify_clock=False,
               remote_env={"AWS_FPGA_REPO_DIR": "/opt/aws fpga"})
        probes = [x for x in sshlines() if "clkgen" in x["cmd"]]
        job = [x["cmd"] for x in sshlines() if "stub_runner" in x["cmd"]]
        if probes:
            bad(9, "verify_clock=False still probed the clock")
        elif not r["ok"] or not any("NOT verified" in n for n in r["notes"]):
            bad(9, f"ok={r['ok']} notes={r['notes']}")
        elif "sudo -n env AWS_FPGA_REPO_DIR='/opt/aws fpga' " not in job[0]:
            bad(9, f"remote_env not rendered inside sudo: {job[0]}")
        else:
            ok(9, f"no probe, ok=True, `sudo -n env K=V` rendered, "
                  f"note={r['notes'][0][:40]}…")

        # ── [10] DICT SHAPE PARITY with tile_exec_bridge.run_job, proven by
        #         running the bridge's own fake executor and diffing keys.
        reset()
        r = go()
        if hasattr(bridge, "_FAKE"):
            fake = tmp / "bridge_fake.py"
            fake.write_text("#!/usr/bin/env python3\n" + bridge._FAKE)
            os.chmod(fake, 0o755)
            sim = bridge.run_job(regops, executor="sim", binary=str(fake),
                                 cap_out=str(tmp / "sim.cap.jsonl"))
            miss = sorted(set(sim) - set(r))
            if miss:
                bad(10, f"remote result is missing run_job keys: {miss}")
            elif {k: type(sim[k]) for k in sim} != \
                    {k: type(r[k]) for k in sim}:
                bad(10, "key TYPES differ: "
                        + str({k: (type(sim[k]).__name__, type(r[k]).__name__)
                               for k in sim if type(sim[k]) is not type(r[k])}))
            else:
                ok(10, f"same {len(sim)} keys + same types as run_job; "
                       f"extras={sorted(set(r) - set(sim))}")
        else:
            bad(10, "bridge._FAKE vanished — cannot diff the dict shape")

        # ── [11] the attach() shim: hw -> remote, sim -> untouched
        import types
        holder = types.SimpleNamespace(
            run_job=lambda *a, **k: {"sentinel": "ORIGINAL", "args": a,
                                     "kw": k},
            TILE_DIV_DEFAULT=5)
        if attach(holder) is not False:
            bad(11, "attach installed a shim with no $APEX_F2_HOST set")
        else:
            os.environ["APEX_F2_HOST"] = "ubuntu@10.0.0.9"
            os.environ["APEX_F2_REMOTE_DIR"] = str(rroot)
            os.environ["APEX_F2_NO_CLOCK_CHECK"] = "1"
            # the shim path uses the REAL ssh (that is the point: it proves
            # the route). 10.0.0.9 is unroutable -> fail fast, deterministically.
            os.environ["APEX_F2_SSH_OPTS"] = \
                "-o BatchMode=yes -o ConnectTimeout=2"
            try:
                if not attach(holder):
                    bad(11, "attach refused with a host configured")
                elif holder.run_job(regops, executor="sim",
                                    tile_div=5)["sentinel"] != "ORIGINAL":
                    bad(11, "executor='sim' did not fall through")
                else:
                    reset()
                    got = holder.run_job(regops, executor="hw",
                                         cap_out=str(capf), tile_div=5,
                                         slot=0, timeout_s=120)
                    # the shim cannot pass the stubs, so it uses real ssh:
                    # what matters is that it ROUTED and kept the shape.
                    if got.get("remote") is not True:
                        bad(11, f"executor='hw' was not routed: {got}")
                    elif attach(holder) is not True or not detach(holder):
                        bad(11, "attach is not idempotent / detach failed")
                    elif holder.run_job(regops, executor="hw")["sentinel"] \
                            != "ORIGINAL":
                        bad(11, "detach did not restore run_job")
                    else:
                        ok(11, f"sim falls through; hw routed "
                               f"(rc={got['rc']} ok={got['ok']}); "
                               f"attach idempotent; detach restores")
            except CaptureEgressError:
                bad(11, "hw route raised instead of returning a dict "
                        "(a real ssh must fail as ok=False here)")
            finally:
                for k in ("APEX_F2_HOST", "APEX_F2_REMOTE_DIR",
                          "APEX_F2_NO_CLOCK_CHECK", "APEX_F2_SSH_OPTS"):
                    os.environ.pop(k, None)

        # ── [12] pure unit checks on the two parsers/quoters
        cases = [
            (Path("docs/results/f2_stage2_hw/clkgen_final.txt"), 15.62),
        ]
        real = None
        for rel, want in cases:
            p = bridge.REPO / rel
            if p.is_file():
                real = parse_clkgen_a1(p.read_text())
                if real != want:
                    bad(12, f"{rel} parsed a1={real}, want {want}")
        swapped = ("Clock Group A Frequency (Mhz)\n"
                   "| clk_extra_a3 | clk_extra_a1 |\n"
                   "|--------------|--------------|\n"
                   "|      62.50   |      15.62   |\n")
        if parse_clkgen_a1(swapped) != 15.62:
            bad(12, "column position is not taken from the header")
        elif parse_clkgen_a1("Clock Group A\n| clk_extra_a2 |\n|  15.62  |\n") \
                is not None:
            bad(12, "returned a frequency with no clk_extra_a1 column")
        elif _rq("~/apexrun/a b") != "~/'apexrun/a b'" \
                or _scp_path("~/apexrun/x") != "apexrun/x" \
                or _rq("/tmp/x") != "/tmp/x":
            bad(12, f"remote quoting wrong: {_rq('~/apexrun/a b')} "
                    f"{_scp_path('~/apexrun/x')}")
        else:
            ok(12, f"clkgen_final.txt -> a1={real} MHz; header-indexed; "
                   f"'~/' survives quoting")

        # ── [13] TRANSPORT: ssh multiplexing is ON by default and reaches
        #         BOTH ssh and scp; $APEX_F2_NO_SSH_MUX disables it.
        mo = ssh_mux_opts()
        a_ssh = _ssh_argv("ssh", None, (), "u@h", "true")
        a_scp = _scp_argv("scp", None, (), ["x"], "u@h:")
        if not (mo and "ControlMaster=auto" in mo):
            bad(13, f"multiplexing off by default: {mo}")
        elif not (set(mo) <= set(a_ssh) and set(mo) <= set(a_scp)):
            bad(13, f"mux opts missing from ssh/scp argv: {a_ssh} {a_scp}")
        else:
            os.environ["APEX_F2_NO_SSH_MUX"] = "1"
            off_opts = ssh_mux_opts()
            os.environ.pop("APEX_F2_NO_SSH_MUX", None)
            if off_opts != ():
                bad(13, "APEX_F2_NO_SSH_MUX did not disable multiplexing")
            elif ssh_mux_opts(False) != ():
                bad(13, "mux=False did not disable multiplexing")
            else:
                ok(13, f"ControlMaster/ControlPath/ControlPersist on both ssh "
                       f"and scp; opt-out honoured ({len(mo)} opts, socket "
                       f"dir {Path(mo[3].split('=')[1]).parent})")

        # ── [14] the upload is a GZIPPED tar (>16 files) and the remote side
        #         is told to unzip it — the two halves must agree.
        reset()
        many = [tmp / "local" / f"up_{i:03d}.regops.jsonl" for i in range(20)]
        for i, m in enumerate(many):
            m.write_text("\n".join(
                json.dumps({"op": "w", "a": 0x3040, "d": i},
                           separators=(",", ":")) for _ in range(400)) + "\n")
        r = go(regops_path=many, cap_out=str(capf))
        cmds = [x["cmd"] for x in sshlines()]
        untar = [c for c in cmds if "tar x" in c]
        landed = sorted((rroot / m.name).is_file() for m in many)
        line = [l for l in r["log"].splitlines() if "upload:" in l]
        if not untar or "tar xzf" not in untar[0]:
            bad(14, f"remote side does not gunzip: {untar or cmds}")
        elif not all(landed) or len(landed) != len(many):
            bad(14, f"not every file landed: {landed}")
        elif not line:
            bad(14, "no upload size line in the log")
        elif [c for c in cmds if c.startswith("mkdir -p") and "tar x" not in c]:
            bad(14, "the tar path still spends a separate mkdir round trip")
        elif "mkdir -p" not in untar[0]:
            bad(14, f"the tar path dropped the mkdir entirely: {untar[0]}")
        else:
            ok(14, f"{line[0].split('upload: ')[1]}; all {len(many)} files "
                   f"landed and are byte-identical: "
                   f"{all((rroot / m.name).read_bytes() == m.read_bytes() for m in many)}")
        os.environ["APEX_F2_NO_GZIP"] = "1"
        reset()
        try:
            go(regops_path=many, cap_out=str(capf))
            cmds = [x["cmd"] for x in sshlines()]
            untar = [c for c in cmds if "tar x" in c]
            if not untar or "tar xf" not in untar[0]:
                bad("14b", f"APEX_F2_NO_GZIP still gzipped: {untar}")
            else:
                ok("14b", "APEX_F2_NO_GZIP=1 falls back to a plain tar, and "
                          "the remote `tar xf` matches it")
        finally:
            os.environ.pop("APEX_F2_NO_GZIP", None)

        # ── [15] THE IMAGE-KEYED CLOCK GATE (clock_key.py). Two legal
        #        frequencies exist since the A0 track; the gate must bind the
        #        expectation to the AGFI and refuse every cross-wiring:
        #        an A2 image on a 62.5 card, an A0 image on a 15.625 card
        #        (unless the underclock arm is EXPLICIT), any overclock, and
        #        any unknown image.
        A2_IMG = "agfi-0ecab46b8a8376b21"        # registered A2 (b64_05b)
        A0_IMG = "agfi-0feedfacefeedface"        # fake A0, injected below
        assert A0_IMG not in clock_key.IMAGE_RECIPE
        clock_key.IMAGE_RECIPE[A0_IMG] = (0, "selftest-only fake A0 image")
        try:
            # [15a] keyed A2 image on an A2 card: PASS, keying disclosed
            reset(a1="15.62")
            r = go(agfi=A2_IMG)
            if not (r["ok"] and r["clkgen_a1_mhz"] == 15.62
                    and f"image {A2_IMG} constrained at recipe A2"
                    in r["log"]):
                bad("15a", f"keyed A2 pass broken: ok={r['ok']} "
                           f"a1={r['clkgen_a1_mhz']}")
            else:
                ok("15a", "A2 image keyed, 15.62 accepted, keying in log")
            # [15b] the SILENT-GARBAGE direction: A2 image, card programmed
            #       to 62.5 (note a3 ALSO prints 62.50 — the column trap)
            reset(a1="62.50")
            r = go(agfi=A2_IMG)
            ran = [c for c in [x["cmd"] for x in sshlines()]
                   if "stub_runner" in c]
            if r["ok"] or not r["refused"] or r["rc"] != RC_CLOCK_REFUSED \
                    or ran:
                bad("15b", f"A2 image at 62.5 NOT refused: ok={r['ok']} "
                           f"rc={r['rc']} ran={ran}")
            else:
                ok("15b", f"A2 image on a 62.5 card REFUSED rc={r['rc']}, "
                          f"nothing executed")
            # [15c] A0 image on a 62.5 card: PASS; on a 15.625 card: REFUSE
            #       (an A0 image is not silently accepted at the old clock)
            reset(a1="62.50")
            r = go(agfi=A0_IMG)
            reset(a1="15.62")
            r2 = go(agfi=A0_IMG)
            if not r["ok"] or r["clkgen_a1_mhz"] != 62.50:
                bad("15c", f"A0 image at 62.5 did not pass: {r['notes']}")
            elif r2["ok"] or not r2["refused"]:
                bad("15c", f"A0 image at 15.62 accepted WITHOUT the "
                           f"explicit underclock arm: ok={r2['ok']}")
            elif "-a 0" not in r2["notes"][0]:
                bad("15c", f"the refusal's fix does not program A0: "
                           f"{r2['notes'][0][:90]}")
            else:
                ok("15c", "A0 image: 62.50 passes; 15.62 refused with an "
                          "'-a 0' fix (no implicit underclock)")
            # [15d] the EXPLICIT underclock arm (§7.3 A/B): A0 image,
            #       run_recipe=2, card at 15.625 -> PASS, loudly disclosed
            reset(a1="15.62")
            r = go(agfi=A0_IMG, run_recipe=2)
            if not (r["ok"] and "UNDERCLOCK" in r["log"]):
                bad("15d", f"underclock arm broken: ok={r['ok']}")
            else:
                ok("15d", "A0 image at run_recipe=2 passes at 15.62 and "
                          "says UNDERCLOCK in the log")
            # [15e] OVERCLOCK and unknown/bare keys refuse BEFORE contact
            for tag, kw in (("15e-over", dict(agfi=A2_IMG, run_recipe=0)),
                            ("15e-unknown",
                             dict(agfi="agfi-0000000000000dead")),
                            ("15e-bare", dict(run_recipe=0))):
                reset(a1="15.62")
                r = go(**kw)
                probes = sshlines()
                if r["ok"] or not r["refused"]:
                    bad(tag, f"not refused: ok={r['ok']} kw={kw}")
                elif probes:
                    bad(tag, f"refused but still contacted the host: "
                             f"{probes}")
                else:
                    ok(tag, f"refused pre-contact: {r['notes'][0][:58]}…")
            # [15f] env keying reaches remote_config (the attach()/driver
            #       path used by prompt05b/batch_exec without code changes)
            os.environ["APEX_F2_HOST"] = "ubuntu@10.0.0.9"
            os.environ["APEX_F2_AGFI"] = A0_IMG
            os.environ["APEX_F2_RUN_RECIPE"] = "2"
            try:
                cfgk = remote_config()
                if cfgk["agfi"] != A0_IMG or cfgk["run_recipe"] != "2":
                    bad("15f", f"env keying lost: {cfgk}")
                else:
                    ok("15f", "APEX_F2_AGFI / APEX_F2_RUN_RECIPE reach "
                              "remote_config (attach path keyed)")
            finally:
                for k in ("APEX_F2_HOST", "APEX_F2_AGFI",
                          "APEX_F2_RUN_RECIPE"):
                    os.environ.pop(k, None)
        finally:
            clock_key.IMAGE_RECIPE.pop(A0_IMG, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for k in ("APEX_FAKE_SSH_LOG", "APEX_STUB_LOG", "APEX_FAKE_A1",
                  "APEX_STUB_MODE", "APEX_FAKE_HOME"):
            os.environ.pop(k, None)
    print(f"REMOTE_HW_EXEC SELFTEST: {'FAIL' if fails else 'PASS'} "
          f"(fails={fails})")
    return 1 if fails else 0


# ═══════════════════════════════════ CLI ════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Execute a regops job on a real F2 instance over ssh.")
    ap.add_argument("regops", nargs="*", help="regops.jsonl file(s)")
    ap.add_argument("--host", default=os.environ.get("APEX_F2_HOST"),
                    help="ubuntu@<ip> ($APEX_F2_HOST)")
    ap.add_argument("--key", default=os.environ.get("APEX_F2_KEY"),
                    help="ssh private key ($APEX_F2_KEY)")
    ap.add_argument("--user", default=os.environ.get("APEX_F2_USER", "ubuntu"))
    ap.add_argument("--remote-dir",
                    default=os.environ.get("APEX_F2_REMOTE_DIR", "~/apexrun"))
    ap.add_argument("--remote-runner", default=os.environ.get("APEX_F2_RUNNER"),
                    help="f2_host_run.py already on the instance (default: "
                         "ship this tree's copy)")
    ap.add_argument("--python", default=os.environ.get("APEX_F2_PY", "python3"))
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--cap-out", default=None)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--no-verify-clock", action="store_true",
                    help="DANGEROUS: skips the only real tile-clock check")
    ap.add_argument("--agfi", default=os.environ.get("APEX_F2_AGFI"),
                    help="the image that is flying — keys the clock gate's "
                         "expected a1 MHz to that image's constrained recipe "
                         "(clock_key.py; $APEX_F2_AGFI). Without it the gate "
                         "enforces the unkeyed A2 default 15.625 MHz")
    ap.add_argument("--run-recipe", type=int, default=(
                        int(os.environ["APEX_F2_RUN_RECIPE"])
                        if os.environ.get("APEX_F2_RUN_RECIPE") else None),
                    help="deliberate UNDERCLOCK arm: group-A recipe index to "
                         "run the keyed image at (e.g. 2 to fly an A0 image "
                         "at 15.625 for the A/B). Overclock refuses; needs "
                         "--agfi ($APEX_F2_RUN_RECIPE)")
    ap.add_argument("--skip-clkgen-wait", action="store_true",
                    help="pass --skip-clkgen-wait to f2_host_run.py (its own "
                         "preflight is vacuous anyway; this gate replaces it)")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="extra f2_host_run.py arg (e.g. --stop-on-fail)")
    ap.add_argument("--remote-env", action="append", default=[],
                    metavar="K=V",
                    help="env var set inside the sudo call (e.g. "
                         "AWS_FPGA_REPO_DIR=/opt/aws-fpga)")
    ap.add_argument("--check-clock", action="store_true",
                    help="only read the tile clock and report it")
    ap.add_argument("--no-require-summary", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.host:
        ap.error("--host (or $APEX_F2_HOST) is required")
    if a.check_clock:
        c = check_tile_clock(a.host, a.key, user=a.user, slot=a.slot,
                             sudo=not a.no_sudo, agfi=a.agfi,
                             run_recipe=a.run_recipe, verbose=True)
        print(c["log"].rstrip())
        print(f"CLOCK GATE: {'PASS' if c['ok'] else 'REFUSE'} — {c['why']}")
        return 0 if c["ok"] else RC_CLOCK_REFUSED
    if not a.regops:
        ap.error("give regops file(s), --check-clock, or --selftest")

    r = run_job_remote(a.regops, a.host, a.key, remote_dir=a.remote_dir,
                       slot=a.slot, cap_out=a.cap_out,
                       timeout_s=a.timeout_s, python=a.python,
                       verify_clock=not a.no_verify_clock, user=a.user,
                       sudo=not a.no_sudo, remote_runner=a.remote_runner,
                       extra_args=a.extra_arg,
                       remote_env=dict(kv.split("=", 1)
                                       for kv in a.remote_env) or None,
                       skip_clkgen_wait=a.skip_clkgen_wait,
                       agfi=a.agfi, run_recipe=a.run_recipe,
                       require_summary=not a.no_require_summary, verbose=True)
    print(r["log"].rstrip())
    print(f"REMOTE: host={r['host']} rc={r['rc']} ok={r['ok']} "
          f"captures={len(r['captures'])} reported={r['n_reported']} "
          f"a1={r['clkgen_a1_mhz']} MHz -> {r['cap_out']}")
    for n in r["notes"]:
        print(f"REMOTE NOTE: {n}")
    return 0 if r["ok"] else (r["rc"] or 1)


if __name__ == "__main__":
    sys.exit(main())
