#!/usr/bin/env python3
"""gen_silu_lut_tables.py — emit rtl/asu/silu_lut_tables.svh from the arbiter.

The 256-entry (base, slope) SiLU tables come from
apex_golden.transformer._silu_lut_tables() (contract C-SILU) — IMPORTED from
the golden model, never retyped.  Mirrors golden/gen_exp_lut_tables.py.
Regenerate + diff whenever in doubt:

    python3 \
        gen_silu_lut_tables.py [--out PATH]

Width proofs asserted below (the RTL relies on them):
  base  fit signed [15:0]  (base max = 32500 < 2^15;  min = -1140)
  slope fit signed [ 9:0]  (slope max = 282 < 512;    min = -26)
  silu_fx(0) == 0;  silu_fx(8·2^10) == silu(8)·2^12  (upper clamp target)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
from apex_golden.transformer import (  # noqa: E402
    SILU_DOMAIN, SILU_FRAC_REM, SILU_IN_FRAC, SILU_LUT_BITS, SILU_OUT_FRAC,
    _silu_lut_tables, silu_fx,
)

import numpy as np  # noqa: E402


def _sv(width: int, v: int) -> str:
    return f"{width}'sd{v}" if v >= 0 else f"-{width}'sd{-v}"


def _fmt(name: str, width: int, vals: np.ndarray, n: int) -> str:
    lines = [f"localparam logic signed [{width-1}:0] {name} [{n}] = '{{"]
    for i in range(0, n, 8):
        row = ", ".join(_sv(width, int(v)) for v in vals[i:i + 8])
        sep = "," if i + 8 < n else ""
        lines.append(f"  {row}{sep}")
    lines.append("};")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parents[1] / "rtl/asu/silu_lut_tables.svh"))
    args = ap.parse_args()

    assert (SILU_LUT_BITS == 8 and SILU_IN_FRAC == 10 and SILU_OUT_FRAC == 12
            and SILU_FRAC_REM == 6 and SILU_DOMAIN == 8.0), "C-SILU contract drift"
    base, slope = _silu_lut_tables()
    n = 1 << SILU_LUT_BITS
    assert len(base) == n and len(slope) == n
    assert -(1 << 15) <= int(base.min()) and int(base.max()) < (1 << 15), \
        f"base out of signed 16b: [{base.min()},{base.max()}]"
    assert -(1 << 9) <= int(slope.min()) and int(slope.max()) < (1 << 9), \
        f"slope out of signed 10b: [{slope.min()},{slope.max()}]"
    # RTL invariants the SwiGLU datapath depends on:
    assert int(silu_fx(np.array([0]))[0]) == 0, "silu_fx(0) != 0"
    lim = int(SILU_DOMAIN * (1 << SILU_IN_FRAC))
    top = int(silu_fx(np.array([lim]))[0])
    assert top == int(base[-1] + slope[-1]), "upper-clamp interp mismatch"

    hdr = f"""\
// silu_lut_tables.svh — GENERATED FILE, DO NOT EDIT.
//
// Source of truth: apex_golden.transformer._silu_lut_tables() (golden
// arbiter), emitted by golden/gen_silu_lut_tables.py.  Regenerate + diff:
//   python3 \\
//       golden/gen_silu_lut_tables.py
// Contract: C-SILU (golden/apex_golden/transformer.py) — 256-entry SiLU LUT
// over [-8,8], input Q5.{SILU_IN_FRAC} (clamped), output SIGNED Q4.{SILU_OUT_FRAC},
// segment 1/16, {SILU_FRAC_REM} interp bits, SIGNED arith interp shift.
//   x_cl = clamp(x_q, -{lim}, {lim});  off = x_cl + {lim};
//   idx  = min(off >> {SILU_FRAC_REM}, 255);  frac = off - (idx << {SILU_FRAC_REM});
//   y    = BASE[idx] + ($signed(SLOPE[idx]) * frac) >>> {SILU_FRAC_REM};
// base in [{int(base.min())},{int(base.max())}]; slope in [{int(slope.min())},{int(slope.max())}].
"""
    body = (_fmt("SILU_LUT_BASE", 16, base, n) + "\n\n"
            + _fmt("SILU_LUT_SLOPE", 10, slope, n) + "\n")
    Path(args.out).write_text(hdr + "\n" + body)
    print(f"wrote {args.out}: 256 base ([{int(base.min())},{int(base.max())}]), "
          f"256 slope ([{int(slope.min())},{int(slope.max())}])")


if __name__ == "__main__":
    main()
