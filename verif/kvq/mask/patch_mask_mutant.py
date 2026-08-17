#!/usr/bin/env python
"""patch_mask_mutant.py <bypass|stuck> <mutant_rtl_dir> — break one D-027
mask-commit property in a scratch copy of rtl/kvq (mutation checks,
verif/kvq/mask; same discipline as verif/kvq/audit/patch_mutant.py).

Each patch is an exact-string replacement that must apply EXACTLY ONCE;
anything else exits non-zero (the Makefile then marks the mutation failed).
The real rtl/kvq/ is never touched — the Makefile copies it into build/.
"""
import sys
from pathlib import Path

PATCHES = {
    # commit-check bypass: drop the popcount==OUTLIER_K legality gate from
    # the commit arm — an illegal commit now takes effect. Caught by C2
    # (owned must stay 0 after a rejected commit; the mutant makes it 1).
    # MASK_ERR itself still raises (separate wire) — the mutant isolates the
    # "illegal commit must not take effect" property, not the flag.
    "bypass": ("kvq_engine.sv",
               "                    REG_MASK_CTRL: if (mask_commit_fire && mask_commit_legal) begin",
               "                    REG_MASK_CTRL: if (mask_commit_fire) begin  // MUTANT: legality bypassed"),
    # stuck live-mask: the commit sets ownership but never loads the staged
    # value — the live mask stays the build default. Caught by C3
    # (mask_valid is COMPUTED from the live bus: popcount(0) != 2 -> the
    # owned+valid readback shows 0x1... wait: owned=1, valid=0 -> 0x2? No:
    # {owned,valid} bit1=owned bit0=valid -> got 0x2 exp 0x3) and by every
    # C4 record/bank/readback expectation (encoded under the zero mask).
    "stuck": ("kvq_engine.sv",
              "                        mask_live      <= mask_stage;",
              "                        mask_live      <= mask_live;  // MUTANT: live mask stuck"),
}


def main():
    which, root = sys.argv[1], Path(sys.argv[2])
    fname, old, new = PATCHES[which]
    p = root / fname
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        print("patch_mask_mutant: anchor for '%s' found %d times (need exactly 1)"
              % (which, n))
        sys.exit(1)
    p.write_text(text.replace(old, new))
    print("patch_mask_mutant: applied '%s' to %s" % (which, p))


if __name__ == "__main__":
    main()
