#!/usr/bin/env python3
"""coverage_report.py — Layer-3 coverage aggregation + GATE.

Anti-fabrication tie: every case in build/manifest.json must have a fresh
run log whose CHECK COUNT EQUALS the generator's scripted expectation
(plus passive-tap checks, which only ever ADD) and errors == 0. The case
matrix (named cores incl. the FULL-LENGTH F-1 regressions T=128/T=100/T=70,
the F-5 fix regression + D=128 breadth, random T sweep incl. T>8 chunked
and T%8!=0, T=1) is gated structurally from the manifest.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main(build: str) -> int:
    b = Path(build)
    man = json.loads((b / "manifest.json").read_text())
    fails = []
    total = 0

    print("== Layer-3 case gate ==")
    for c in man["cases"]:
        log = b / f"run_{c['name']}.log"
        if not log.exists():
            fails.append(f"{c['name']}: not run")
            print(f"  [MISS] {c['name']}: NOT RUN")
            continue
        txt = log.read_text()
        if c.get("expect") == "bug":
            ok = ("L3 PASS" not in txt) and (c["fail_sig"] in txt)
            print(f"  [{'ok ' if ok else 'FAIL'}] {c['name']}: "
                  f"CONFIRMED-BUG regression (F-5), sig='{c['fail_sig']}'")
            if not ok:
                fails.append(f"{c['name']}: F-5 bug signature missing")
            continue
        m = re.search(r"L3 RESULT: cycles=(\d+) checks=(\d+) errors=(\d+)",
                      txt)
        if not m or "L3 PASS" not in txt:
            fails.append(f"{c['name']}: no PASS")
            print(f"  [FAIL] {c['name']}: no PASS/RESULT")
            continue
        checks, errors = int(m.group(2)), int(m.group(3))
        total += checks
        ok = errors == 0 and checks >= c["checks"]
        print(f"  [{'ok ' if ok else 'FAIL'}] {c['name']}: checks={checks} "
              f"(scripted {c['checks']}) errors={errors}")
        if not ok:
            fails.append(f"{c['name']}: checks/errors mismatch")

    kinds = {}
    for c in man["cases"]:
        kinds.setdefault(c["kind"], []).append(c)
    named = [c for c in man["cases"] if c["kind"].startswith("named")]
    rand = kinds.get("random-full", [])
    reqs = [
        ("named cases >= 6", len(named) >= 6),
        ("random full-chain jobs >= 15", len(rand) >= 15),
        ("F-5 fix regression (D=128 full chain, kept-passing) present",
         any(c["kind"] == "f5-fix-regression" and c["D"] == 128
             for c in man["cases"])),
        ("D=128 cases >= 4 (F-5 closure breadth)",
         sum(1 for c in man["cases"] if c["D"] == 128) >= 4),
        ("T=1 present", any(c["T"] == 1 for c in man["cases"])),
        ("T>8 chunked cases >= 6",
         sum(1 for c in man["cases"] if c["T"] > 8) >= 6),
        ("T %% 8 != 0 cases >= 4",
         sum(1 for c in man["cases"] if c["T"] % 8 != 0) >= 4),
        ("full-length T=128 cases >= 2 (F-1 envelope edge)",
         sum(1 for c in man["cases"] if c["T"] == 128) >= 2),
        ("T=100 at D=128 present (F-1 x F-5, partial-group tensor)",
         any(c["T"] == 100 and c["D"] == 128 for c in man["cases"])),
        ("T=70 full-length present (F-1 split-row staging, nbT%%BPR!=0)",
         any(c["T"] == 70 for c in man["cases"])),
        ("F-2 CLOSED: CQ-4+ tile cases >= 2 (grouped keys + mask)",
         sum(1 for c in man["cases"]
             if c["kind"] == "named-core-cq4p") >= 2),
        ("F-2/D-008: tile-level partial-group FLUSH case present (T=1 "
         "CQ-4+)",
         any(c["kind"] == "named-core-cq4p" and c["T"] == 1
             for c in man["cases"])),
        ("F-2: full-group CQ-4+ case present (T=128 = 8 full G=16 groups)",
         any(c["kind"] == "named-core-cq4p" and c["T"] == 128
             for c in man["cases"])),
        ("D-022 actuation: TIP-auto mixed-tier case present",
         any(c["kind"] == "tip-auto" for c in man["cases"])),
        ("F-2/F-3 residuals documented in the infeasibility register",
         any("F-2 residual" in e["why"] for e in man["infeasible"])
         and any("F-3 residual" in e["why"] for e in man["infeasible"])),
    ]
    print("== case-matrix buckets ==")
    for what, ok in reqs:
        print(f"  [{'ok ' if ok else 'MISS'}] {what}")
        if not ok:
            fails.append(f"matrix: {what}")

    print("== documented-infeasible register ==")
    for e in man["infeasible"]:
        print(f"  DOC {e['case']}: {e['why']}")

    print(f"total checks: {total}")
    if fails:
        print("L3 COVERAGE GATE: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("L3 COVERAGE GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
