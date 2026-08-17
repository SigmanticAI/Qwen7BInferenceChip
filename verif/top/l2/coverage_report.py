#!/usr/bin/env python3
"""coverage_report.py — Layer-2 chain coverage aggregation + GATE.

Reads the generator logs (COVGEN stimulus-fact lines — deterministic, the
vectors ARE what ran) and every required run log (PASS + errors=0 + check
count sanity). Exits nonzero if any required run is missing / failed or any
required coverage bucket is empty: `make all` cannot lie green.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_RUNS = [
    # (log, expected RESULT tag)
    ("run_l2a_directed.log", "L2A"), ("run_l2a_fast.log", "L2A"),
    ("run_l2a_random.log", "L2A"), ("run_l2a_storm.log", "L2A"),
    ("run_l2a_reset.log", "L2A"),
    ("run_l2b_t0.log", "L2B"), ("run_l2b_t0_storm.log", "L2B"),
    ("run_l2b_t1.log", "L2B"), ("run_l2b_t1_storm.log", "L2B"),
    ("run_l2b_t1_reset.log", "L2B"), ("run_l2b_t2.log", "L2B"),
    ("run_l2b_adv_T1.log", "L2B"), ("run_l2b_adv_out1000.log", "L2B"),
    ("run_l2b_calib_T128.log", "L2B"), ("run_l2b_calib_T70.log", "L2B"),
    ("run_l2b_calib_d128.log", "L2B"),
    ("run_l2c_directed.log", "L2C"), ("run_l2c_fast.log", "L2C"),
    ("run_l2c_storm.log", "L2C"), ("run_l2c_random.log", "L2C"),
    ("run_l2c_reset.log", "L2C"),
]

# COVGEN buckets that must be nonzero (key -> gen log scope)
REQUIRED_BUCKETS = {
    "l2a": ["jobs", "m1", "m8", "tail_cols", "n_lt8", "cols64", "chunked",
            "ktail", "sat", "illegal", "neg_ext", "reset"],
    "l2b_t0": ["cases"],
    "l2b_t1": ["cases", "flush", "straddle_lt", "straddle_eq", "straddle_gt",
               "identity"],
    "l2b_t1_reset": ["reset"],
    "l2b_t2": ["cases", "flush", "identity"],
    "l2b_adv_T1": ["cases", "flush", "acc_s"],
    "l2b_adv_out1000": ["cases", "acc_s"],
    "l2b_calib_T128": ["cases", "acc_s"],
    "l2b_calib_T70": ["cases", "flush", "acc_s"],
    "l2b_calib_d128": ["cases", "flush", "acc_s"],
    "l2c": ["tiles", "fp16", "int8", "frame", "intmin", "zero_tile",
            "single", "thr_extreme", "imprd", "blk_hi", "reset"],
}


def main(build: str) -> int:
    b = Path(build)
    fails = []

    cov: dict[str, dict[str, int]] = {}
    for g in sorted(b.glob("gen_l2*.log")):
        for line in g.read_text().splitlines():
            mm = re.match(r"COVGEN (\S+) (\S+) (\d+)", line)
            if mm:
                cov.setdefault(mm.group(1), {})[mm.group(2)] = int(mm.group(3))

    print("== Layer-2 coverage gate ==")
    for scope, keys in REQUIRED_BUCKETS.items():
        got = cov.get(scope, {})
        for k in sorted(set(keys)):
            v = got.get(k, 0)
            mark = "ok " if v > 0 else "MISS"
            print(f"  [{mark}] {scope}.{k} = {v}")
            if v <= 0:
                fails.append(f"bucket {scope}.{k} empty")

    total_checks = 0
    for log, tag in REQUIRED_RUNS:
        p = b / log
        if not p.exists():
            fails.append(f"missing run log {log}")
            print(f"  [MISS] {log}: NOT RUN")
            continue
        txt = p.read_text()
        m = re.search(rf"{tag} RESULT: cycles=(\d+) checks=(\d+) errors=(\d+)",
                      txt)
        if not m or f"{tag} PASS" not in txt:
            fails.append(f"{log}: no PASS/RESULT")
            print(f"  [FAIL] {log}: no PASS")
            continue
        checks, errors = int(m.group(2)), int(m.group(3))
        total_checks += checks
        ok = errors == 0 and checks > 0
        print(f"  [{'ok ' if ok else 'FAIL'}] {log}: checks={checks} "
              f"errors={errors}")
        if not ok:
            fails.append(f"{log}: errors={errors}")

    print(f"total checks across required runs: {total_checks}")
    if fails:
        print("L2 COVERAGE GATE: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("L2 COVERAGE GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
