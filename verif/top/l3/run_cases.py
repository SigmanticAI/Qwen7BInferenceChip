#!/usr/bin/env python3
"""run_cases.py — run every manifest case on its build's binary, one log per
case (build/run_<case>.log). Runs ALL cases even after a failure (the
coverage gate needs the full picture) but exits nonzero if any failed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BIN = {"b64": "obj_b64/Vtb_apex_l3", "b128": "obj_b128/Vtb_apex_l3"}


def main(build: str) -> int:
    b = Path(build)
    man = json.loads((b / "manifest.json").read_text())
    bad = 0
    for c in man["cases"]:
        log = b / f"run_{c['name']}.log"
        cmd = [str(b / BIN[c["build"]]),
               f"+vectors={b}/cases/{c['name']}.txt",
               f"+watchdog={c['watchdog']}"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        log.write_text(r.stdout)
        passed = "L3 PASS" in r.stdout and r.returncode == 0
        if c.get("expect") == "bug":
            # EXPECT_BUG discipline: the case DOCUMENTS a confirmed tile
            # bug and must keep failing with its signature; a silent pass
            # means the RTL changed under us -> loud gate failure.
            if not passed and c["fail_sig"] in r.stdout:
                print(f"[CONFIRMED-BUG] {c['name']}: still fails with "
                      f"'{c['fail_sig']}' (F-5 register)")
                continue
            print(f"[ALARM] {c['name']}: expected the F-5 bug signature "
                  f"'{c['fail_sig']}' but got "
                  f"{'a PASS' if passed else 'a different failure'} — "
                  f"re-audit required")
            bad += 1
            continue
        verdict = "PASS" if passed else "FAIL"
        tail = [ln for ln in r.stdout.splitlines() if "L3 RESULT" in ln]
        print(f"[{verdict}] {c['name']}: "
              f"{tail[-1] if tail else '(no RESULT line)'}")
        if verdict == "FAIL":
            bad += 1
            for ln in r.stdout.splitlines():
                if "%Error" in ln or "%Fatal" in ln or "[E" in ln:
                    print(f"    {ln}")
                    break
    if bad:
        print(f"L3 RUNS: {bad}/{len(man['cases'])} cases FAILED")
        return 1
    print(f"L3 RUNS: all {len(man['cases'])} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
