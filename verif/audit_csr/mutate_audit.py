#!/usr/bin/env python3
"""mutate_audit.py — audit-checker credibility mutation (scratch copy only).

perf_double: PERF_BUSY lanes count by 2 per busy cycle — must be caught by
the audit TB's cycle-counting monitor (CA-4/CA-5).
"""
import shutil
import sys
from pathlib import Path

FILES = ["apex_pkg.sv", "csr/csr_regs.sv"]

MUTS = {
    "perf_double": (
        "csr/csr_regs.sv",
        "          if (block_busy_i[i] && perf_cnt[i] != PERF_MAX)\n"
        "            perf_cnt[i] <= perf_cnt[i] + 1'b1;",
        "          if (block_busy_i[i] && perf_cnt[i] != PERF_MAX)\n"
        "            perf_cnt[i] <= perf_cnt[i] + PERF_W'(2);",
    ),
}


def main():
    name, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    mfile, old, new = MUTS[name]
    if dst.exists():
        shutil.rmtree(dst)
    for f in FILES:
        d = dst / f
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f, d)
    p = dst / mfile
    text = p.read_text()
    assert old in text, f"mutation anchor not found in {mfile}"
    p.write_text(text.replace(old, new, 1))
    print(f"MUTANT {name}: {mfile} patched")


if __name__ == "__main__":
    main()
