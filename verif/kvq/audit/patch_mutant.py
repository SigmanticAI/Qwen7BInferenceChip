#!/usr/bin/env python
"""patch_mutant.py <b1|b2|b4> <mutant_rtl_dir> — REVERT one D-020 fix in a
scratch copy of rtl/kvq (fix-revert mutation checks, verif/kvq/audit).

Each patch is an exact-string replacement that must apply EXACTLY ONCE;
anything else exits non-zero (the Makefile then marks the mutation failed).
The real rtl/kvq/ is never touched — the Makefile copies it into build/.
"""
import sys
from pathlib import Path

PATCHES = {
    # B-1 revert: drop the ctrl_reset priority guard on the FSM case, so any
    # case branch that assigns `state` overrides the reset again (last NBA
    # wins) — the original lost-soft-reset bug.
    "b1": ("kvq_engine.sv",
           "            if (!ctrl_reset || state == ST_OUTPUT || state == ST_OFLUSH)\n"
           "            case (state)",
           "            case (state)"),
    # B-2 revert: tie off the datapath abort — cq_value_path/cq_key_path and
    # the shared quant divider keep walking through a soft reset. (D-026 note:
    # scale_bank_store deliberately has NO clear port, so this mutant cannot
    # and must not touch the bank — the bank-survival bucket guards the
    # opposite failure, a bank wrongly WIRED to dp_clear.)
    "b2": ("kvq_engine.sv",
           "    wire dp_clear = ctrl_reset;",
           "    wire dp_clear = 1'b0;  // MUTANT: B-2 reverted"),
    # B-4 revert: -0.0 outlier reads back +0.0 again. RETARGETED for A2/D-026:
    # the old mutant (dead-branch the whole kp_dec_hat_sx force) is now a
    # functional NO-OP — the clean-room cq_fp_pkg::dequant_one preserves the
    # signed zero ({sign, 31'd0}, sign = code[8]^s[15]), so dequant(+1, -0.0)
    # is already 32'h8000_0000 and the engine force (bit31 from key_cur_scale,
    # the stored raw fp16 lane, ex-key_field_rd) is redundant armor; verified
    # empirically 2026-07-14: the dead-branch mutant PASSES the full negz
    # directed run. The retargeted mutant instead drops the forced sign
    # exactly when the dequant magnitude is zero — reintroducing the original
    # B-4 symptom (and nothing else): an exact -0.0 outlier lane reads back
    # 32'h0000_0000. Caught by the mode=1 negz directed content
    # (got 00000000 exp 80000000, the original finding's signature).
    "b4": ("kvq_engine.sv",
           "                                         ? {key_cur_scale[COORD_WIDTH-1], kp_dec_hat[30:0]}",
           "                                         ? {key_cur_scale[COORD_WIDTH-1] && (|kp_dec_hat[30:0]), kp_dec_hat[30:0]}  // MUTANT: B-4 reverted"),
}


def main():
    which, root = sys.argv[1], Path(sys.argv[2])
    fname, old, new = PATCHES[which]
    p = root / fname
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        print(f"PATCH FAIL {which}: pattern found {n} times (expected 1)")
        sys.exit(1)
    p.write_text(text.replace(old, new))
    print(f"PATCH OK {which}: reverted in {p}")


if __name__ == "__main__":
    main()
