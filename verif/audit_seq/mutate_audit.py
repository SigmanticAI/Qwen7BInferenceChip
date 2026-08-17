#!/usr/bin/env python3
"""mutate_audit.py — audit-checker credibility mutation (scratch copy only).

busy_skid: busy no longer covers a beat held only in the input skid
(count==0, FSM==IDLE) — must be caught by the audit TB's AT-5 check.

Usage: mutate_audit.py <mode> <mutant> [args]
  patch <name> <src_rtl_dir(repo rtl/)> <dst_dir>
                        write the patched scratch copy
  check <name> <runlog> assert the run FAILED with the expected signature
                        (exit 0 iff properly caught)

check() follows verif/top/l3/mutate.py: a kill requires (a) no PASS banner,
(b) real failure evidence, AND (c) the mutant-specific signature. A run that
merely dies with an unrelated %Error/%Fatal (watchdog, drain timeout, bad
plusargs) is an INFRA FAILURE and fails the gate — it is not a kill.
"""
import shutil
import sys
from pathlib import Path

FILES = ["apex_pkg.sv", "xbr/stream_skid.sv", "seq/seq_walker.sv"]

MUTS = {
    "busy_skid": {
        "file": "seq/seq_walker.sv",
        "old": (
            "  assign busy       = (state != S_IDLE) || (count != '0) || in_m_valid\n"
            "                    || abort_active;"
        ),
        "new": (
            "  assign busy       = (state != S_IDLE) || (count != '0)\n"
            "                    || abort_active;"
        ),
        # the AT-5 busy-honesty check is the designated catcher
        "sig": ["[AT-5]: accepted descriptor pending"],
        "kind": "busy-honesty",
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
    assert mut["old"] in text, f"mutation anchor not found in {mut['file']}"
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
