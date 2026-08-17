#!/usr/bin/env python3
"""perf_compare.py — D-005 perf GATE.

Parses the PERF lines emitted by tb_mxe_perf.sv from two builds of the SAME
testbench:
  argv[1]  run_new.log   rtl/mxe/mxe_ctrl.sv        (load-under-compute)
  argv[2]  run_base.log  baseline/mxe_ctrl_seq.sv   (pinned sequential ctrl)

Gate (exit 1 on any violation):
  - both logs contain the same job set and both TBs passed bit-exact;
  - every multi-chunk job: new overlap_cycles > 0 (load-under-compute is
    actually happening) and baseline overlap_cycles == 0 (the baseline really
    is the sequential scheduler — keeps the comparison honest);
  - every job: new total cycles < baseline total cycles (the overlap buys
    real cycles, not just a reshuffle).

Prints the speedup table that TRACEABILITY.md quotes for D-005.
"""
import re
import sys

RX = re.compile(r"PERF job=(\S+) m=(\d+) k=(\d+) n=(\d+) wgap=(\d+) "
                r"cycles=(\d+) overlap=(\d+)")


def parse(path):
    jobs = {}
    passed = False
    for line in open(path):
        m = RX.search(line)
        if m:
            jobs[m.group(1)] = {
                "m": int(m.group(2)), "k": int(m.group(3)),
                "n": int(m.group(4)), "wgap": int(m.group(5)),
                "cycles": int(m.group(6)), "overlap": int(m.group(7)),
            }
        if "PERF TB PASS" in line:
            passed = True
    return jobs, passed


def main():
    new_log, base_log = sys.argv[1], sys.argv[2]
    new, new_ok = parse(new_log)
    base, base_ok = parse(base_log)
    fails = []

    if not new_ok:
        fails.append(f"{new_log}: TB did not pass")
    if not base_ok:
        fails.append(f"{base_log}: TB did not pass")
    if set(new) != set(base) or not new:
        fails.append(f"job sets differ or empty: {sorted(new)} vs {sorted(base)}")

    print(f"{'job':<18} {'base cyc':>9} {'new cyc':>9} {'saved':>7} "
          f"{'speedup':>8} {'overlap':>8}")
    print("-" * 66)
    for name in sorted(new):
        if name not in base:
            continue
        nn, bb = new[name], base[name]
        multi_chunk = (nn["k"] + 7) // 8 > 1
        speedup = bb["cycles"] / nn["cycles"] if nn["cycles"] else float("inf")
        print(f"{name:<18} {bb['cycles']:>9} {nn['cycles']:>9} "
              f"{bb['cycles'] - nn['cycles']:>7} {speedup:>7.3f}x "
              f"{nn['overlap']:>8}")
        if multi_chunk and nn["overlap"] <= 0:
            fails.append(f"{name}: new ctrl overlap_cycles == 0 (no "
                         f"load-under-compute)")
        if bb["overlap"] != 0:
            fails.append(f"{name}: baseline overlap_cycles = {bb['overlap']} "
                         f"!= 0 (baseline is not the sequential scheduler)")
        if nn["cycles"] >= bb["cycles"]:
            fails.append(f"{name}: no cycle win ({nn['cycles']} >= "
                         f"{bb['cycles']})")
    print("-" * 66)

    if fails:
        for f in fails:
            print(f"PERF GATE VIOLATION: {f}")
        print("PERF GATE: FAILED")
        return 1
    print("PERF GATE: PASS — overlap_cycles > 0 on every multi-chunk job and "
          "total cycles strictly below the sequential baseline (D-005)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
