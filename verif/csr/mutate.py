#!/usr/bin/env python3
"""mutate.py — checker-credibility mutations for the CSR suite (house
pattern: mutate a SCRATCH COPY of the RTL, never the tree; the suite must
FAIL on every mutant WITH THAT MUTANT'S OWN FAILURE SIGNATURE, or the
checkers are decorative).

Usage: mutate.py <mode> <mutant> [args]
  patch <name> <src_rtl_dir(repo rtl/)> <dst_dir>
                        write the patched scratch copy
  check <name> <runlog> assert the run FAILED with the expected signature
                        (exit 0 iff properly caught)

check() follows verif/top/l3/mutate.py: a kill requires (a) no PASS banner,
(b) real failure evidence, AND (c) the mutant-specific signature. A run that
merely dies with an unrelated %Error/%Fatal (missing vector file, watchdog,
bad plusargs) is an INFRA FAILURE and fails the gate — it is not a kill.
"""
import shutil
import sys
from pathlib import Path

FILES = [
    "apex_pkg.sv",
    "csr/csr_regs.sv",
]

MUTS = {
    # sticky-race priority flipped: a W1C in the same cycle as a desc_error
    # pulse LOSES the error (set no longer wins) — must be caught by the Z
    # (race) records and the ap_sticky_set SVA
    "sticky_race": {
        "file": "csr/csr_regs.sv",
        "old": (
            "      if (desc_error_i)  desc_error_sticky <= 1'b1;\n"
            "      else if (w1c_fire) desc_error_sticky <= 1'b0;"
        ),
        "new": (
            "      if (w1c_fire)           desc_error_sticky <= 1'b0;\n"
            "      else if (desc_error_i)  desc_error_sticky <= 1'b1;"
        ),
        # ap_sticky_set SVA, or the STATUS (@0x04) value check after a Z race
        "sig": ["desc_error pulse did not set the sticky bit",
                "]: read @04 "],
        "kind": "race",
    },
    # PERF enable gate removed: counters run while perf_en=0 — must be
    # caught by the V-record mirror compares (the run2-lesson counters must
    # be trustworthy, not merely present)
    "perf_nogate": {
        "file": "csr/csr_regs.sv",
        "old": "      end else if (perf_en) begin",
        "new": "      end else begin",
        # V-record bus-level mirror compare
        "sig": ["[V]: perf @"],
        "kind": "perf-gate",
    },
}


def patch(name: str, src: Path, dst: Path) -> int:
    mut = MUTS[name]
    if dst.exists():
        shutil.rmtree(dst)
    for f in FILES:
        d = dst / f
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f, d)
    p = dst / mut["file"]
    text = p.read_text()
    assert mut["old"] in text, \
        f"mutation anchor not found in {mut['file']}: {mut['old']!r}"
    p.write_text(text.replace(mut["old"], mut["new"], 1))
    print(f"MUTANT {name}: {mut['file']} patched")
    return 0


def check(name: str, runlog: str) -> int:
    mut = MUTS[name]
    txt = Path(runlog).read_text(errors="replace")
    if "TB PASS" in txt:
        print(f"MUTANT {name}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    fatal = ("%Fatal" in txt) or ("%Error" in txt) or ("TB FAIL" in txt)
    if not fatal:
        print(f"MUTANT {name}: run neither passed nor failed?! Gate FAIL.")
        return 1
    hit = [s for s in mut["sig"] if s in txt]
    if not hit:
        print(f"MUTANT {name}: run FAILED but without the expected signature "
              f"{mut['sig']} — infra failure, not a kill. Gate FAIL.")
        return 1
    print(f"MUTANT {name}: CAUGHT ({mut['kind']}; signature {hit})")
    return 0


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    if mode == "patch":
        sys.exit(patch(name, Path(sys.argv[3]), Path(sys.argv[4])))
    if mode == "check":
        sys.exit(check(name, sys.argv[3]))
    sys.exit(f"unknown mode {mode!r} (expected patch|check)")
