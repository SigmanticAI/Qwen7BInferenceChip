#!/usr/bin/env python3
"""gen_silu_vectors.py — exhaustive SiLU oracle for the asu_silu smoke TB.

Everything the TB compares against is produced HERE from the golden arbiter
apex_golden.transformer.silu_fx (imported, never retyped), mirroring
verif/asu/smoke/gen_asu_vectors.py's exp oracle:

  build/silu_expected.hex   silu_fx(clamp(x)) for ALL 65536 16-bit input
                            patterns (address = unsigned bit pattern of x_q),
                            4 hex digits/line (signed Q4.12, two's complement).
                            Below -8192 clamps to the floor entry; above +8192
                            clamps to idx=255/frac=64 (SiLU(8) ~= 7.997).

silu_fx's contract range over the whole 16-bit domain is [-1140, +32757],
which fits signed 16b exactly (asserted below), so the DUT port and this
oracle are both signed [15:0].

Run:
    python3 \
        gen_silu_vectors.py --outdir build
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "golden"))
import numpy as np  # noqa: E402

from apex_golden.transformer import (  # noqa: E402
    SILU_DOMAIN, SILU_IN_FRAC, _silu_lut_tables, silu_fx,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # all 65536 16-bit patterns -> signed value -> silu_fx (which itself
    # clamps to [-8192, +8192]); the DUT clamps identically.
    u = np.arange(65536, dtype=np.int64)
    x = np.where(u >= 32768, u - 65536, u)
    y = silu_fx(x)                                   # signed Q4.12

    lim = int(SILU_DOMAIN * (1 << SILU_IN_FRAC))     # 8192
    assert int(y.min()) == -1140 and int(y.max()) == 32757, \
        f"C-SILU range drift: [{int(y.min())},{int(y.max())}]"
    assert -(1 << 15) <= int(y.min()) and int(y.max()) < (1 << 15), \
        "silu_fx output does not fit signed 16b"
    # documented invariants (redundant with the sweep, but provenance-explicit)
    assert int(silu_fx(np.array([0]))[0]) == 0, "silu_fx(0) != 0"
    floor_v = int(silu_fx(np.array([-lim]))[0])
    assert int(silu_fx(np.array([-lim - 5000]))[0]) == floor_v, "floor-clamp drift"
    assert int(silu_fx(np.array([lim]))[0]) == 32757, "ceil-clamp drift"
    assert int(silu_fx(np.array([lim + 5000]))[0]) == 32757, "ceil-clamp drift"

    with open(out / "silu_expected.hex", "w") as f:
        f.write("".join(f"{int(v) & 0xFFFF:04x}\n" for v in y))

    # ── ROM hex for the Icarus ($readmemh) path — SAME golden tables as the
    # canonical rtl/asu/silu_lut_tables.svh localparam arrays (never retyped).
    base, slope = _silu_lut_tables()
    assert len(base) == 256 and len(slope) == 256
    with open(out / "silu_lut_base.hex", "w") as f:
        f.write("".join(f"{int(v) & 0xFFFF:04x}\n" for v in base))
    with open(out / "silu_lut_slope.hex", "w") as f:
        f.write("".join(f"{int(v) & 0x3FF:03x}\n" for v in slope))

    print(f"silu oracle: 65536 entries -> {out/'silu_expected.hex'} "
          f"(range [{int(y.min())},{int(y.max())}], lim={lim})")
    print(f"silu ROM   : 256 base + 256 slope -> {out}/silu_lut_{{base,slope}}.hex "
          f"(Icarus $readmemh path)")


if __name__ == "__main__":
    main()
