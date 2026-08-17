#!/usr/bin/env python
"""patch_fparith_mutant.py <mut_tie|mut_eps> <pkg_copy.sv> — seed one numeric
error into a SCRATCH COPY of cq_fp_pkg.sv (seeded-error checks that the
exhaustive fparith TB MUST catch; the real rtl/kvq is never touched).

  mut_tie  flip the single RNE tie decision to round-half-UP (affects both
           scale_from_amax and quant_one via the shared rne_div_bounded)
  mut_eps  remove the EPS = 2^-14 floor from scale_from_amax

Each patch is an exact-string replacement that must apply EXACTLY ONCE;
anything else exits non-zero.
"""
import sys
from pathlib import Path

PATCHES = {
    "mut_tie": (
        "        else if ((r2 == dd) && q[0]) q = q + 13'd1;"
        "   // RNE tie -> nearest even",
        "        else if (r2 == dd)           q = q + 13'd1;"
        "   // MUTANT: tie half-up",
    ),
    "mut_eps": (
        "        if (e <= -15) begin",
        "        if (e <= -15 && bits < 0) begin  // MUTANT: EPS floor removed",
    ),
}


def main():
    which, target = sys.argv[1], Path(sys.argv[2])
    old, new = PATCHES[which]
    text = target.read_text()
    n = text.count(old)
    if n != 1:
        print(f"PATCH FAIL {which}: pattern found {n} times (expected 1)")
        sys.exit(1)
    target.write_text(text.replace(old, new))
    print(f"PATCH OK {which}: seeded in {target}")


if __name__ == "__main__":
    main()
