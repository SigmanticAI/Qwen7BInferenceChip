#!/usr/bin/env python3
"""mutate.py — checker-credibility mutations for the SEQ suite (house
pattern: mutate a SCRATCH COPY of the RTL, never the tree; the suite must
FAIL on every mutant or the checkers are decorative).

Usage:
  mutate.py <mutant_name> <src_rtl_dir(repo rtl/)> <dst_dir>
      write the patched scratch copy
  mutate.py check <mutant_name> <runlog>
      assert the mutant run FAILED with the expected signature (exit 0 iff
      properly caught — house verif/top/l3 mutate.py check() shape). A run
      that merely crashed (missing vectors, watchdog, segfault) is NOT a
      kill: no TB PASS banner alone is not enough, the mutant-specific
      checker signature must be in the log.
"""
import shutil
import sys
from pathlib import Path

FILES = [
    "apex_pkg.sv",
    "xbr/stream_skid.sv",
    "seq/seq_walker.sv",
]

MUTS = {
    # D-006 violation: the walker stops waiting for done/desc_error and
    # returns to IDLE immediately after the descriptor is accepted, so the
    # NEXT descriptor is presented while a job is in flight — must be caught
    # by the bound ap_d006_serial SVA (the exact run1 failure class D-006
    # exists to kill)
    "d006_early": (
        "seq/seq_walker.sv",
        "        S_WAIT: if (mxe_done || mxe_desc_error) state <= S_IDLE;",
        "        S_WAIT: state <= S_IDLE;",
    ),
    # FIFO walk-order corruption: the read pointer skips an entry on every
    # dispatch — must be caught by the in-order bit-exact dispatch scoreboard
    "fifo_skip": (
        "seq/seq_walker.sv",
        "          rd_ptr <= rd_ptr + PTR_W'(1);",
        "          rd_ptr <= rd_ptr + PTR_W'(2);",
    ),
}


# expected kill signatures per mutant (any one suffices); these are the
# checker messages the mutation was DESIGNED to trip (see MUTS comments) —
# tb_seq_sb.sv scoreboard "FAIL cyc=" classes and seq_sb_sva.svh $error text
SIGS = {
    # killed by ap_d006_serial (or the busy-honesty SVA the same early-IDLE
    # transition necessarily breaks)
    "d006_early": ["D-006 violated", "busy hole: job in flight"],
    # killed by the in-order bit-exact dispatch scoreboard
    "fifo_skip":  ["ORPHAN dispatch", "dispatch #"],
}


def check(name: str, runlog: str) -> int:
    sigs = SIGS[name]
    txt = Path(runlog).read_text()
    if "TB PASS" in txt:
        print(f"{name}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    hit = [s for s in sigs if s in txt]
    fatal = ("%Fatal" in txt) or ("TB FAIL" in txt) or ("%Error" in txt)
    if not fatal:
        print(f"{name}: run neither passed nor failed?! Gate FAIL.")
        return 1
    if not hit:
        print(f"{name}: FAILED but without the expected signature "
              f"{sigs} — inspect (infra failure is not a kill). Gate FAIL.")
        return 1
    print(f"MUTANT {name} CAUGHT (signature {hit})")
    return 0


def main():
    if sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
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
    assert old in text, f"mutation anchor not found in {mfile}: {old!r}"
    p.write_text(text.replace(old, new, 1))
    print(f"MUTANT {name}: {mfile} patched")


if __name__ == "__main__":
    main()
