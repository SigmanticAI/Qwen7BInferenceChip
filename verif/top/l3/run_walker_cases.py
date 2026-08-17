#!/usr/bin/env python3
"""run_walker_cases.py — B1 stage-5 gate: the L3 suite in WALKER mode.

A NEW sibling of run_cases.py, not an edit to it: the host-mode runner stays
byte-identical so the compat proof keeps its meaning (B1_WALKER.md §7).

Same mechanical pass criterion as run_cases.py ("L3 PASS" in stdout and
rc == 0), plus the two §A-1 obligations that make the coverage limit legible
rather than inferable:
  - every CQ-8 case must actually WALK (a case that silently fell back to host
    mode would otherwise look like a pass);
  - grouped-tier cases must be REFUSED with WALK_ERR_TIER, then pass in host
    mode; and the run must PRINT the tier split.

It also compares each case's check count against the host-mode run, so a
walker-mode pass that quietly LOST checks fails the gate — the §B-1 rule.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

BIN = {"b64": "obj_b64/Vtb_apex_l3", "b128": "obj_b128/Vtb_apex_l3"}
RE_RES = re.compile(r"L3 RESULT: cycles=(\d+) checks=(\d+) errors=(\d+)")
RE_SPLIT = re.compile(r"WALKER SPLIT: walked=(\d+) refused=(\d+)")


def host_checks(build: Path) -> dict[str, int]:
    """Per-case check counts from the host-mode run, if it has been done."""
    f = build / "run_all.txt"
    if not f.exists():
        return {}
    out = {}
    for ln in f.read_text().splitlines():
        m = re.search(r"\] (\S+): L3 RESULT: cycles=\d+ checks=(\d+)", ln)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def main(build_dir: str = "build") -> int:
    b = Path(build_dir)
    man = json.loads((b / "manifest.json").read_text())
    href = host_checks(b)
    bad = 0
    n_walk = n_ref = 0

    for c in man["cases"]:
        name = c["name"]
        desc = b / "walker" / f"{name}.desc"
        if not desc.exists():
            print(f"[FAIL] {name}: no walker descriptor "
                  f"(run gen_walker_desc.py)")
            bad += 1
            continue
        walkable = "WALKABLE 1" in desc.read_text()

        r = subprocess.run(
            [str(b / BIN[c["build"]]),
             f"+vectors={b}/cases/{name}.txt",
             f"+watchdog={c['watchdog'] * 2}",      # walker paces differently
             "+walker=1", f"+walkdesc={desc}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (b / f"run_walker_{name}.log").write_text(r.stdout)
        passed = "L3 PASS" in r.stdout and r.returncode == 0

        walked = "WALKER: walking" in r.stdout
        refused = "WALKER: REFUSED" in r.stdout
        m = RE_RES.search(r.stdout)
        checks = int(m.group(2)) if m else -1

        problems = []
        if not passed:
            problems.append("did not pass")
        # §A-1: a CQ-8 case that silently fell back to host mode is NOT a pass
        if walkable and not walked:
            problems.append("CQ-8 case did not walk (silent host fallback)")
        if (not walkable) and not refused:
            problems.append("grouped-tier case was not refused")
        # §B-1: walker mode must not LOSE checks vs host mode
        if name in href and checks < href[name]:
            problems.append(f"check count dropped {href[name]} -> {checks}")

        if problems:
            print(f"[FAIL] {name}: {'; '.join(problems)}")
            bad += 1
        else:
            tag = "WALKED " if walked else "REFUSED"
            delta = f" (host {href[name]})" if name in href else ""
            print(f"[{tag}] {name}: checks={checks}{delta}")
        n_walk += 1 if walked else 0
        n_ref += 1 if refused else 0

    total = len(man["cases"])
    print()
    print(f"WALKER SPLIT: {n_walk}/{total} cases walked (CQ-8), "
          f"{n_ref}/{total} refused (grouped tier, B1b)")
    if bad:
        print(f"WALKER-MODE L3: FAIL ({bad} case(s))")
        return 1
    print("WALKER-MODE L3: all cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
