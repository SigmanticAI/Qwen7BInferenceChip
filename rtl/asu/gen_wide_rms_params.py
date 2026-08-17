#!/usr/bin/env python3
"""gen_wide_rms_params.py — emit rtl/asu/asu_wide_rms_params.svh from the arbiter.

The C-RMSW wide-D RMSNorm mean path (docs/design/WIDE_RMSNORM.md §3) replaces
the pow-2 mean shift with `mean2 = (sum2 · MU_d) >> (S_d + 16)`. The (MU, S)
pairs come from apex_golden.attention._wide_mu (the golden arbiter) — IMPORTED,
never retyped. One entry per legal wide row length d = k·CHUNK, k = 1..K_MAX;
K_MAX = 64 (d = 8192) matches the golden host-accumulator proof bound
(rmsnorm_fx_wide: sum2 <= 2^27 for D <= 8192). Regenerate and diff whenever in
doubt:

    python3 gen_wide_rms_params.py [--out PATH]

`make params-check` in verif/asu/wide regenerates into build/ and diffs
against the checked-in copy (a provenance gate, tables-check house pattern).

Width/range proofs asserted below (the RTL relies on them):
  MU entries fit [16:0]  (mu = RNE(2^16·2^s/d) < 2^17 since 2^s < 2d)
  S  entries fit [3:0]   (s = (d-1).bit_length() <= 13 at d = 8192)
  pow-2 d  =>  MU == 2^16 exactly (the μ path reduces bit-exactly to the
               legacy `sum2 >> log2(d)` shift — the D<=128 identity keystone)
  D=3584 pin: (MU, S) == (74898, 12) (contract drift alarm)
  rsqrt range: mean2 at sum2 = d·2^14 (the all-(±128) row) stays <= 2^14 for
               every k, so `n2 = mean2 + 1` lands in the same 32-bit rsqrt_unit
               operand range as today (WIDE_RMSNORM.md §1 invariant).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "golden"))
from apex_golden.attention import _wide_mu  # noqa: E402

CHUNK = 128
K_MAX = 64          # d up to 8192 = the golden sum2 <= 2^27 accumulator bound
MU_W = 17
S_W = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "asu_wide_rms_params.svh"))
    args = ap.parse_args()

    assert CHUNK == 128 and (CHUNK & (CHUNK - 1)) == 0, "contract drift"
    mus, ss = [0], [0]                      # index k = d/CHUNK; [0] unused
    for k in range(1, K_MAX + 1):
        d = k * CHUNK
        mu, s = _wide_mu(d)
        assert 0 < mu < (1 << MU_W), f"MU out of [{MU_W-1}:0] at d={d}: {mu}"
        assert 0 <= s < (1 << S_W), f"S out of [{S_W-1}:0] at d={d}: {s}"
        if d & (d - 1) == 0:                # pow-2 d: exact-reduction keystone
            assert mu == 1 << 16, f"pow-2 d={d}: mu {mu} != 2^16"
        # rsqrt operand range: max sum2 = d·2^14 (all-(±128) row) must give
        # mean2 <= 2^14 so the 32-bit norm2 range is unchanged vs D<=128.
        mean2_max = (d * (1 << 14) * mu) >> (s + 16)
        assert mean2_max <= 1 << 14, f"d={d}: mean2_max {mean2_max} > 2^14"
        mus.append(mu)
        ss.append(s)
    assert (mus[3584 // CHUNK], ss[3584 // CHUNK]) == (74898, 12), \
        "D=3584 pin drifted vs _wide_mu"

    def fmt(name: str, width: int, vals: list[int]) -> str:
        n = len(vals)
        lines = [f"localparam logic [{width-1}:0] {name} [{n}] = '{{"]
        for i in range(0, n, 8):
            row = ", ".join(f"{width}'d{v}" for v in vals[i:i + 8])
            sep = "," if i + 8 < n else ""
            lines.append(f"  {row}{sep}")
        lines.append("};")
        return "\n".join(lines)

    hdr = f"""\
// asu_wide_rms_params.svh — GENERATED FILE, DO NOT EDIT.
//
// Source of truth: apex_golden.attention._wide_mu (golden arbiter, C-RMSW),
// emitted by rtl/asu/gen_wide_rms_params.py. Regenerate + diff:
//   python3 gen_wide_rms_params.py
// Contract: docs/design/WIDE_RMSNORM.md §3 — wide mean path
//   mean2 = (sum2 · WIDE_RMS_MU[k]) >> (WIDE_RMS_S[k] + 16),  k = d / CHUNK.
// Index k = d/128, entry [0] unused (d=0 is not a row). K_MAX = 64 (d = 8192)
// = the golden accumulator proof bound (sum2 <= 2^27). For pow-2 d the entry
// is exactly 2^16, so the μ path reduces bit-exactly to `sum2 >> log2(d)`.
// Product width: sum2 (<= 2^27) · MU (< 2^17) < 2^44 — a 44-bit unsigned
// product; after >> (S+16) the result mean2 <= 2^14 (asserted per entry by
// the generator), so the rsqrt_unit 32-bit operand range is unchanged.
localparam int unsigned WIDE_RMS_CHUNK = {CHUNK};
localparam int unsigned WIDE_RMS_K_MAX = {K_MAX};
"""
    body = (fmt("WIDE_RMS_MU", MU_W, mus) + "\n\n" +
            fmt("WIDE_RMS_S", S_W, ss) + "\n")
    Path(args.out).write_text(hdr + "\n" + body)
    print(f"wrote {args.out}: k = 1..{K_MAX} (d = {CHUNK}..{K_MAX * CHUNK}), "
          f"MU[28] = {mus[28]}, S[28] = {ss[28]} (D = 3584)")


if __name__ == "__main__":
    main()
