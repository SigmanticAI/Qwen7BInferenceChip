#!/usr/bin/env python3
"""gen_rope_lut_tables.py — emit rtl/rope/rope_lut_tables.svh from the arbiter.

The 256-entry (base, slope) cos/sin tables come from
apex_golden.transformer._cos_sin_lut_tables() (contract C-ROPE) — IMPORTED
from the golden model, never retyped.  Mirrors golden/gen_exp_lut_tables.py.
Regenerate + diff whenever in doubt:

    python3 \
        gen_rope_lut_tables.py [--out PATH]

Width proofs asserted below (the RTL relies on them):
  cos/sin base  fit signed [15:0]  (|base| max = 16384 = 2^14  ->  needs 16b)
  cos/sin slope fit signed [10:0]  (|slope| max = 402 < 1024)
  cos_fx(0) == 16384 (== 1.0 in Q2.14);  sin_fx(0) == 0
  cos_fx(2^12) == 0 (quarter turn),  sin_fx(2^12) == 16384
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
from apex_golden.transformer import (  # noqa: E402
    CS_FRAC_REM, CS_LUT_BITS, CS_OUT_FRAC, PH_FRAC,
    _cos_sin_lut_tables, cos_fx, sin_fx,
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
        Path(__file__).resolve().parents[1] / "rtl/rope/rope_lut_tables.svh"))
    args = ap.parse_args()

    assert (CS_LUT_BITS == 8 and PH_FRAC == 14 and CS_OUT_FRAC == 14
            and CS_FRAC_REM == 6), "C-ROPE contract drift"
    cb, cs, sb, ss = _cos_sin_lut_tables()
    n = 1 << CS_LUT_BITS
    for nm, t in (("cos_base", cb), ("cos_slope", cs),
                  ("sin_base", sb), ("sin_slope", ss)):
        assert len(t) == n, nm
    for nm, t in (("base", cb), ("base", sb)):
        assert int(np.abs(t).max()) < (1 << 15), f"{nm} out of signed 16b"
    for nm, t in (("slope", cs), ("slope", ss)):
        assert int(np.abs(t).max()) < (1 << 10), f"{nm} out of signed 11b"
    # RTL invariants the RoPE datapath depends on:
    assert int(cos_fx(np.array([0]))[0]) == (1 << CS_OUT_FRAC), "cos_fx(0)!=1.0"
    assert int(sin_fx(np.array([0]))[0]) == 0, "sin_fx(0)!=0"
    assert int(cos_fx(np.array([1 << (PH_FRAC - 2)]))[0]) == 0, "cos(¼turn)!=0"
    assert int(sin_fx(np.array([1 << (PH_FRAC - 2)]))[0]) == (1 << CS_OUT_FRAC), \
        "sin(¼turn)!=1.0"

    hdr = f"""\
// rope_lut_tables.svh — GENERATED FILE, DO NOT EDIT.
//
// Source of truth: apex_golden.transformer._cos_sin_lut_tables() (golden
// arbiter), emitted by golden/gen_rope_lut_tables.py.  Regenerate + diff:
//   python3 \\
//       golden/gen_rope_lut_tables.py
// Contract: C-ROPE (golden/apex_golden/transformer.py) — 256-entry cos/sin
// LUT over a FULL turn, phase Q0.{PH_FRAC} (u_q), output SIGNED Q2.{CS_OUT_FRAC},
// segment 1/256 turn, {CS_FRAC_REM} interp bits, SIGNED arith interp shift.
//   idx  = u_q >> {CS_FRAC_REM};  frac = u_q - (idx << {CS_FRAC_REM});
//   y    = BASE[idx] + ($signed(SLOPE[idx]) * frac) >>> {CS_FRAC_REM};
// base |.| max 16384 (2^14, = 1.0); slope |.| max 402.
"""
    body = (_fmt("ROPE_COS_BASE", 16, cb, n) + "\n\n"
            + _fmt("ROPE_COS_SLOPE", 11, cs, n) + "\n\n"
            + _fmt("ROPE_SIN_BASE", 16, sb, n) + "\n\n"
            + _fmt("ROPE_SIN_SLOPE", 11, ss, n) + "\n")
    Path(args.out).write_text(hdr + "\n" + body)
    print(f"wrote {args.out}: 256 cos/sin base+slope "
          f"(base max {int(max(np.abs(cb).max(), np.abs(sb).max()))}, "
          f"slope max {int(max(np.abs(cs).max(), np.abs(ss).max()))})")


if __name__ == "__main__":
    main()
