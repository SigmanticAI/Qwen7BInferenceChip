#!/usr/bin/env python3
# smoke_bridge_sim.py — close the two never-run seams around `cap`:
#
#   SEAM 1 (bridge assumption A1).  tile_exec_bridge.run_job(executor="sim")
#   was only ever exercised against the FAKE executor in its own --selftest
#   (a Python script that prints the contract's summary line by construction).
#   Nothing had ever proved the argv it builds — [f2sim, +tile_div=N,
#   +cap_out=PATH, <regops>...] — is accepted by the REAL verilated binary,
#   that the binary's summary line matches SUMMARY_RE, or that ok==True comes
#   out the far end.
#
#   SEAM 2 (Stage A's gate).  trace_to_regops.py --shadow-cap and the
#   executors' cap-egress exist, but the gate they were built for had never
#   been run: take a job whose expectations are already validated by the `r`
#   compare path, ALSO capture every one of those sites, and assert the value
#   the tile HANDED BACK equals the baked expectation for the SAME site.
#   That is the only evidence that read-and-return reads what compare-and-
#   discard has been checking all along.
#
# The verdict here is deliberately NOT the executor's own self-report. The
# sim prints `F2SIM CAPGATE: ... -> PASS`, but a gate that trusts the thing
# under test proves nothing: this script re-derives every expectation from
# the regops file it fed in and compares against the egress JSONL itself.
# The executor's line is cross-checked as a SECOND opinion, never as the
# verdict.
#
# Why n=0 on the canonical set is not a pass: build/f2_regops/*.regops.jsonl
# contain ZERO `cap` ops (they are compiled without --shadow-cap), so running
# them with +cap_out= yields "n=0" — summary present, ok==True, and not one
# byte of the capture-record path executed. Seam 1 needs that run (it proves
# argv acceptance); seam 2 needs shadow regops, which is why this script
# recompiles the job.
#
# N10 (docs/design/PROMPT_DEMO_AUDIT.md): the shadow regops MUST NOT be
# written into build/f2_regops — that directory is the canonical 18-job set
# the gates replay, and --shadow-cap output would silently redefine it.
# _guard_outdir() refuses, rather than documenting, that mistake.
#
# CLI:  python3 scripts/fpga/f2/smoke_bridge_sim.py            # both seams
#       ... --job job_s019_L19_h03 --scratch DIR --binary PATH --clean
#       ... --skip-canonical          # skip the seam-1 (n=0) canonical run

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import tile_exec_bridge as bridge                        # noqa: E402

TRACE_DIR = REPO / "docs" / "results" / "s8_7b_token" / "artifact_trace"
CANON_DIR = REPO / "build" / "f2_regops"
EMITTER = HERE / "trace_to_regops.py"
DEFAULT_JOB = "job_s019_L19_h03"        # smallest canonical job (T=20)


class GateFail(RuntimeError):
    pass


# --quiet buffers the evidence instead of dropping it: a gate should be silent
# when green (one verdict line, as `make capgate` wants) and MAXIMALLY loud
# when red, so the buffer is flushed on the failure path. Diagnostics that
# only exist in a passing run are diagnostics nobody reads.
_QUIET = False
_BUF: list[str] = []


def _say(tag: str, msg: str) -> None:
    line = f"[capgate] {tag:<10} {msg}"
    _BUF.append(line)
    if not _QUIET:
        print(line, flush=True)


def _flush() -> None:
    if _QUIET:
        for line in _BUF:
            print(line, flush=True)
    _BUF.clear()


# ── N10 guard ───────────────────────────────────────────────────────────────
def _guard_outdir(outdir: Path) -> Path:
    """Refuse to emit shadow regops anywhere that could shadow the canonical
    set. Compared on RESOLVED paths so a symlink or `..` cannot slip past."""
    out = Path(outdir).expanduser().resolve()
    canon = CANON_DIR.resolve() if CANON_DIR.exists() else CANON_DIR
    if out == canon or canon in out.parents:
        raise SystemExit(
            f"REFUSED: --scratch {out} is inside the canonical regops dir "
            f"{canon}. Shadow-cap output there would redefine the 18-job "
            f"gate set (audit N10). Pick a scratch dir outside build/.")
    return out


def _fresh_dir(d: Path) -> Path:
    """Create d and remove only the artifacts THIS script writes — never an
    rmtree of a caller-supplied path."""
    d.mkdir(parents=True, exist_ok=True)
    for f in list(d.glob("*.regops.jsonl")) + list(d.glob("manifest.json")):
        f.unlink()
    return d


# ── regops-side truth: what the emitter baked ───────────────────────────────
def read_cap_ops(regops: Path) -> list[dict]:
    """The `cap` ops in FILE ORDER = the order the executor must produce them.

    Returns [{'tag','a','m','e'|None,'line'}]. `e` is optional in the vocab;
    a cap WITHOUT it is uncheckable and is reported, not silently skipped.
    """
    out = []
    with regops.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or '"op":"cap"' not in line.replace(" ", ""):
                continue
            d = json.loads(line)
            if d.get("op") != "cap":
                continue
            out.append({"tag": d["tag"], "a": int(d["a"]), "m": int(d["m"]),
                        "e": (int(d["e"]) if "e" in d else None),
                        "line": lineno})
    return out


def assert_caps_match(cap_ops: list[dict], caps: list[dict]) -> dict:
    """THE GATE. Every captured value == the baked expectation for the SAME
    site, matched BOTH positionally (execution order == emission order) and
    by tag. Raises GateFail on the first structural problem; value mismatches
    are collected so the report shows the shape of the failure, not just one.
    """
    if not cap_ops:
        raise GateFail(
            "the regops file contains no `cap` op — this job was compiled "
            "WITHOUT --shadow-cap, so a green run proves nothing about the "
            "capture path")
    if len(caps) != len(cap_ops):
        raise GateFail(f"captured {len(caps)} values but the regops file has "
                       f"{len(cap_ops)} `cap` ops")
    tags = [c["tag"] for c in cap_ops]
    if len(set(tags)) != len(tags):
        dup = next(t for t in tags if tags.count(t) > 1)
        raise GateFail(f"emitter produced duplicate cap tag {dup!r} — tags "
                       f"are the value's only identity (N2)")

    by_tag = {c["tag"]: c for c in cap_ops}
    ordered = sorted(caps, key=lambda c: c["i"])
    mismatches, unchecked, misordered = [], [], []
    for k, (want, got) in enumerate(zip(cap_ops, ordered)):
        if got["tag"] != want["tag"]:
            misordered.append((k, want["tag"], got["tag"]))
            want = by_tag.get(got["tag"])          # fall back to tag identity
            if want is None:
                raise GateFail(f"capture #{k} carries tag {got['tag']!r}, "
                               f"which the regops file never emitted")
        if got["addr"] != want["a"] or got["mask"] != (want["m"] & 0xFFFFFFFF):
            raise GateFail(
                f"{got['tag']}: egress addr/mask 0x{got['addr']:04x}/"
                f"0x{got['mask']:08x} != emitted 0x{want['a']:04x}/"
                f"0x{want['m'] & 0xFFFFFFFF:08x}")
        if want["e"] is None:
            unchecked.append(got["tag"])
            continue
        exp = want["e"] & want["m"] & 0xFFFFFFFF     # BOTH sides masked
        if got["value"] != exp:
            mismatches.append((got["tag"], f"0x{got['addr']:04x}",
                               got["value"], exp, want["line"]))
    if misordered:
        k, w, g = misordered[0]
        raise GateFail(f"{len(misordered)} capture(s) out of emission order "
                       f"(first: slot {k} expected {w!r}, got {g!r}) — `i` "
                       f"does not track execution order")
    if mismatches:
        lines = [f"    {t} @{a} captured 0x{v:08x} != baked 0x{e:08x} "
                 f"(regops L{ln})" for t, a, v, e, ln in mismatches[:8]]
        raise GateFail(f"{len(mismatches)}/{len(cap_ops)} captured values "
                       f"!= baked expectation:\n" + "\n".join(lines))
    return {"checked": len(cap_ops) - len(unchecked), "unchecked": unchecked}


# ── seam 1: does the bridge drive the REAL binary? ──────────────────────────
def seam1_canonical(job: str, scratch: Path, binary, tile_div: int,
                    timeout_s: int) -> dict:
    """Run the UNMODIFIED canonical regops through run_job(). Proves argv and
    plusarg acceptance, summary-line parsing and ok==True against the real
    executor. Expects n=0 (canonical jobs carry no `cap`) — asserted, so a
    surprise nonzero count means the canonical set drifted."""
    canon = CANON_DIR / f"{job}.regops.jsonl"
    if not canon.is_file():
        raise GateFail(f"canonical regops missing: {canon}")
    r = bridge.run_job(canon, executor="sim", binary=binary,
                       cap_out=str(scratch / "seam1_cap.jsonl"),
                       timeout_s=timeout_s, tile_div=tile_div)
    _say("seam1", f"argv={' '.join(Path(a).name if i == 0 else a for i, a in enumerate(r['argv']))}")
    _say("seam1", f"rc={r['rc']} ok={r['ok']} summary={r['summary']!r}")
    if r["notes"]:
        raise GateFail(f"bridge notes on the canonical run: {r['notes']}")
    if not r["ok"]:
        raise GateFail(f"run_job(executor='sim') not ok on the canonical job: "
                       f"rc={r['rc']} summary={r['summary']!r}")
    if r["captures"]:
        raise GateFail(f"canonical job yielded {len(r['captures'])} captures; "
                       f"the canonical set is supposed to carry no `cap` op "
                       f"(has build/f2_regops been rebuilt with "
                       f"--shadow-cap? that is audit N10)")
    if "F2SIM RESULT:" not in r["log"] or "-> PASS" not in r["log"]:
        raise GateFail("canonical replay did not report PASS")
    return r


# ── seam 2: the shadow-cap read-back gate ───────────────────────────────────
def compile_shadow(job: str, scratch: Path, shadow: bool = True) -> Path:
    """Recompile ONE job with (or without) --shadow-cap into scratch.

    trace_to_regops.py has no per-job selector (only --jobs N over the sorted
    glob), so the job is isolated by pointing --trace at a scratch dir holding
    a single symlink. --out is guarded by _guard_outdir().
    """
    npz = TRACE_DIR / f"{job}.npz"
    if not npz.is_file():
        raise GateFail(f"trace job not found: {npz}")
    tdir = _fresh_dir(scratch / ("tr_shadow" if shadow else "tr_plain"))
    link = tdir / npz.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(npz)
    outdir = _fresh_dir(_guard_outdir(scratch / ("shadow" if shadow else "plain")))
    argv = [sys.executable, str(EMITTER), "--trace", str(tdir),
            "--out", str(outdir)] + (["--shadow-cap"] if shadow else [])
    pr = subprocess.run(argv, capture_output=True, text=True)
    if pr.returncode != 0:
        raise GateFail(f"trace_to_regops.py failed (rc={pr.returncode}):\n"
                       f"{(pr.stdout + pr.stderr)[-1200:]}")
    out = outdir / f"{job}.regops.jsonl"
    if not out.is_file():
        raise GateFail(f"emitter wrote no {out}")
    _say("emit", (pr.stdout.strip().splitlines() or ["(no output)"])[0])
    return out


def seam2_shadow(job: str, scratch: Path, binary, tile_div: int,
                 timeout_s: int, canon_check: bool) -> dict:
    shadow = compile_shadow(job, scratch, shadow=True)
    cap_ops = read_cap_ops(shadow)
    _say("emit", f"{len(cap_ops)} `cap` ops in {shadow.name}, "
                 f"{sum(1 for c in cap_ops if c['e'] is None)} without `e`")

    # Is the shadow file the SAME JOB the canonical set encodes? If the
    # emitter has drifted, a green shadow gate would certify a different
    # workload than the one the gates replay. Cheap, so on by default.
    if canon_check:
        plain = compile_shadow(job, scratch, shadow=False)
        canon = CANON_DIR / f"{job}.regops.jsonl"
        same = plain.read_bytes() == canon.read_bytes()
        _say("canon", f"plain regen {'==' if same else '!='} "
                      f"build/f2_regops/{canon.name}")
        if not same:
            raise GateFail(
                "the emitter no longer reproduces the canonical regops for "
                "this job — the shadow gate would certify a DIFFERENT "
                "workload than build/f2_regops replays. Investigate before "
                "trusting either.")
        stripped = [ln for ln in shadow.read_text().splitlines()
                    if json.loads(ln).get("op") != "cap"]
        if "\n".join(stripped) + "\n" != plain.read_text():
            raise GateFail("shadow regops differ from plain regops by more "
                           "than the added `cap` ops — --shadow-cap is not "
                           "expectation-preserving")
        _say("canon", "shadow-minus-cap == plain (shadow adds caps only)")

    r = bridge.run_job(shadow, executor="sim", binary=binary,
                       cap_out=str(scratch / "shadow_cap.jsonl"),
                       timeout_s=timeout_s, tile_div=tile_div)
    _say("seam2", f"rc={r['rc']} ok={r['ok']} n_reported={r['n_reported']} "
                  f"parsed={len(r['captures'])}")
    for n in r["notes"]:
        _say("seam2", f"NOTE {n}")
    if not r["ok"]:
        raise GateFail(f"run_job not ok on the shadow job: rc={r['rc']} "
                       f"summary={r['summary']!r} notes={r['notes']}")
    if r["notes"]:
        raise GateFail(f"bridge structural notes on the shadow run: {r['notes']}")
    if "CAPMISMATCH" in r["log"]:
        bad = [l for l in r["log"].splitlines() if "CAPMISMATCH" in l]
        raise GateFail(f"executor reported {len(bad)} CAPMISMATCH: {bad[0]}")
    if "F2SIM RESULT:" not in r["log"] or "-> PASS" not in r["log"]:
        raise GateFail("shadow replay did not report PASS — a cap-with-`e` "
                       "mismatch now counts as a failure (D-P1)")

    # THE GATE — re-derived from the regops we fed in, not from the log.
    stats = assert_caps_match(cap_ops, r["captures"])
    _say("seam2", f"independent check: {stats['checked']}/{len(cap_ops)} "
                  f"captured values == baked expectation")
    if stats["unchecked"]:
        _say("seam2", f"{len(stats['unchecked'])} cap(s) had no `e` "
                      f"(uncheckable, disclosed): {stats['unchecked'][:4]}")

    # Second opinion only: the executor's own self-report must AGREE.
    selfline = [l for l in r["log"].splitlines() if "F2SIM CAPGATE:" in l]
    if not selfline:
        raise GateFail("executor printed no 'F2SIM CAPGATE:' line although "
                       "caps carried expectations")
    _say("seam2", f"executor self-report: {selfline[-1].strip()}")
    if "-> PASS" not in selfline[-1]:
        raise GateFail(f"executor self-report is red: {selfline[-1].strip()}")
    got_checked = int(selfline[-1].split("captured=")[1].split()[0])
    if got_checked != stats["checked"]:
        raise GateFail(f"executor checked {got_checked} caps, the regops file "
                       f"carries {stats['checked']} with expectations")
    r["gate"] = stats
    r["n_cap_ops"] = len(cap_ops)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", default=DEFAULT_JOB)
    ap.add_argument("--scratch", default=None,
                    help="scratch dir (default: a fresh temp dir). NEVER "
                         "build/f2_regops — refused (audit N10)")
    ap.add_argument("--binary", default=None,
                    help="f2sim path (default: bridge resolution order, "
                         "which prefers the DDR=0 silicon twin)")
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--skip-canonical", action="store_true",
                    help="skip seam 1 (the n=0 canonical argv-acceptance run)")
    ap.add_argument("--no-canon-check", action="store_true",
                    help="skip the plain-regen == build/f2_regops comparison")
    ap.add_argument("--clean", action="store_true",
                    help="delete the scratch dir on success")
    ap.add_argument("--quiet", action="store_true",
                    help="print ONLY the verdict line when green; the full "
                         "evidence is still dumped if the gate fails")
    a = ap.parse_args()

    global _QUIET
    _QUIET = a.quiet
    scratch = Path(a.scratch).expanduser() if a.scratch else \
        Path(tempfile.mkdtemp(prefix="apex_capgate_"))
    _guard_outdir(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    _say("job", f"{a.job}  tile_div={a.tile_div}")
    _say("scratch", str(scratch))
    try:
        binary = bridge.resolve_sim_binary(a.binary)
        _say("binary", str(binary))
    except FileNotFoundError as e:
        _flush()
        print(f"CAPGATE: FAIL (job={a.job}) — {e}")
        return 1

    n_caps = 0
    try:
        if not a.skip_canonical:
            seam1_canonical(a.job, scratch, str(binary), a.tile_div,
                            a.timeout_s)
            _say("seam1", "PASS — bridge argv/plusargs accepted by the real "
                          "binary, summary parsed, ok=True (n=0 as expected)")
        r = seam2_shadow(a.job, scratch, str(binary), a.tile_div, a.timeout_s,
                         canon_check=not a.no_canon_check)
        n_caps = r["gate"]["checked"]
    except (GateFail, bridge.CaptureEgressError) as e:
        _flush()
        print(f"CAPGATE: FAIL (job={a.job}) — {e}")
        print(f"CAPGATE: evidence kept in {scratch}")
        return 1
    if a.clean:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"CAPGATE: PASS (job={a.job} caps={n_caps} "
          f"values_matched={n_caps}/{n_caps} tile_div={a.tile_div} "
          f"executor=sim:{binary.parent.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
