#!/usr/bin/env python3
"""gen_exp_lut_tables.py — emit rtl/asu/asu_exp_lut_tables.svh from the arbiter.

The 256-entry (base, slope) exp tables come from
apex_golden.compute._lut_tables() (ARCHITECTURE.md §6, D-014) — IMPORTED from
the golden model, never retyped. Regenerate and diff whenever in doubt:

    python3 \
        gen_exp_lut_tables.py [--out PATH]

`make tables-check` in verif/asu/smoke regenerates into build/ and diffs
against the checked-in copy (a provenance gate, D-013 spirit).

Width proofs asserted below (the RTL relies on them):
  base  entries fit  [14:0] (max = rne(exp(-1/16)*2^15) = 30783 < 2^15)
  slope entries fit  [10:0] (max = last-segment delta = 1985 < 2^11, all >= 0)
  exp_fx(0) == 32768 (Q1.15 "1.0"; the asu_softmax rescale adds this constant)
  exp_fx(-16<<10) == base[0] (floor entry, the below-domain clamp target)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "golden"))
from apex_golden.compute import (  # noqa: E402
    FRAC_BITS, LUT_BITS, OUT_FRAC, _lut_tables, exp_fx,
)

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "asu_exp_lut_tables.svh"))
    args = ap.parse_args()

    assert LUT_BITS == 8 and FRAC_BITS == 10 and OUT_FRAC == 15, "contract drift"
    base, slope = _lut_tables()
    n = 1 << LUT_BITS
    assert len(base) == n and len(slope) == n
    assert int(base.min()) >= 0 and int(base.max()) < (1 << 15), \
        f"base out of [14:0]: max {base.max()}"
    assert int(slope.min()) >= 0 and int(slope.max()) < (1 << 11), \
        f"slope out of [10:0]: max {slope.max()}"
    # RTL invariants the ASU depends on:
    assert int(exp_fx(np.array([0]))[0]) == 1 << OUT_FRAC, "exp_fx(0) != 32768"
    assert int(exp_fx(np.array([-(16 << FRAC_BITS)]))[0]) == int(base[0]), \
        "floor entry mismatch"

    def fmt(name: str, width: int, vals: np.ndarray) -> str:
        lines = [f"localparam logic [{width-1}:0] {name} [{n}] = '{{"]
        for i in range(0, n, 8):
            row = ", ".join(f"{width}'d{int(v)}" for v in vals[i:i + 8])
            sep = "," if i + 8 < n else ""
            lines.append(f"  {row}{sep}")
        lines.append("};")
        return "\n".join(lines)

    hdr = f"""\
// asu_exp_lut_tables.svh — GENERATED FILE, DO NOT EDIT.
//
// Source of truth: apex_golden.compute._lut_tables() (golden arbiter),
// emitted by rtl/asu/gen_exp_lut_tables.py. Regenerate + diff:
//   python3 \\
//       gen_exp_lut_tables.py
// Contract: ARCHITECTURE.md §6 / D-014 — 256-entry exp LUT, segment width
// 1/16 real (64 Q6.10 counts), base = rne(exp(edge)·2^15) at segment start,
// slope = rne((exp(edge+h) − exp(edge))·2^15) (all slopes >= 0, max 1985;
// max base 30783; base[255] + slope[255] == 32768 == exp_fx(0)).
"""
    body = fmt("EXP_LUT_BASE", 15, base) + "\n\n" + fmt("EXP_LUT_SLOPE", 11, slope) + "\n"
    Path(args.out).write_text(hdr + "\n" + body)
    print(f"wrote {args.out}: 256 base (max {int(base.max())}), "
          f"256 slope (max {int(slope.max())})")


if __name__ == "__main__":
    main()
