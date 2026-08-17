#!/usr/bin/env python3
"""mutate.py — checker-credibility mutations for the TIP suite (house
pattern: mutate a SCRATCH COPY of the RTL, never the tree; the suite must
FAIL on every mutant or the checkers are decorative).

Usage: mutate.py <mutant_name> <src_rtl_dir(repo rtl/)> <dst_dir>
       mutate.py check <mutant_name> <runlog>

Patch mode copies the 5 TIP-build RTL files into <dst_dir> and applies
exactly one mutation by exact-string replacement (asserts the pattern
exists).

Check mode (house shape: verif/top/l3/mutate.py check()) asserts the mutant
run FAILED **with the mutant-specific signature** (exit 0 iff properly
caught). A run that merely exits non-zero — missing vector file, harness
crash, watchdog — is an infra failure, NOT a caught mutant, and fails the
gate.
"""
import shutil
import sys
from pathlib import Path

FILES = [
    "apex_pkg.sv",
    "xbr/stream_skid.sv",
    "tip/tip_decide.sv",
    "tip/tip_importance.sv",
    "tip/tip_top.sv",
]

MUTS = {
    # (b) TIP-specific contract violation: threshold/tie off-by-one — the
    # strict > of the ratio test becomes >=, so exact ties flip to FP16
    "tie_ge": (
        "tip/tip_decide.sv",
        "d_fp16    <= (lhs > rhs);",
        "d_fp16    <= (lhs >= rhs);",
    ),
    # (a) data corruption in the stream fabric: the skid's direct path flips
    # payload bit 0 (score LSB on the input skid; d_fp16 on the output skid)
    "skid_xor": (
        "xbr/stream_skid.sv",
        "                    m_data  <= s_data;",
        "                    m_data  <= s_data ^ {{(WIDTH-1){1'b0}}, 1'b1};",
    ),
    # (a') importance data corruption: saturation clamp removed -> 16-bit wrap
    "sat_wrap": (
        "tip/tip_importance.sv",
        "assign acc_post = acc_sum[ACC_W] ? {ACC_W{1'b1}} : acc_sum[ACC_W-1:0];",
        "assign acc_post = acc_sum[ACC_W-1:0];",
    ),
    # (c) §5 busy contract hole: post-skid pending beat no longer reported —
    # must be caught by the bound SVA (ap_busy_dvalid / ap_idle_is_idle)
    "busy_hole": (
        "tip/tip_top.sv",
        "assign busy = in_m_valid || eng_tile_active || dec_pend || hold_valid\n             || d_valid;",
        "assign busy = in_m_valid || eng_tile_active || dec_pend || hold_valid;",
    ),
}


# expected failure signature per mutant (any-of), anchored on the exact
# checker message that must catch it:
#   tb_tip_sb.sv decision scoreboard .... "got fp16="
#   tb_tip_sb.sv CSR importance readout .. "]: importance["
#   tip_sb_sva.svh ap_busy_dvalid ........ "output beat pending (post-skid) but busy low"
#   tip_sb_sva.svh ap_idle_is_idle ....... "busy low with work in flight"
SIGS = {
    "tie_ge":    ["got fp16="],
    "skid_xor":  ["got fp16=", "]: importance["],
    "sat_wrap":  ["]: importance[", "got fp16="],
    "busy_hole": ["output beat pending (post-skid) but busy low",
                  "busy low with work in flight"],
}
assert set(SIGS) == set(MUTS), "every mutant needs a failure signature"

PASS_BANNER = "TB PASS"
FAIL_MARKS = ("TB FAIL", "%Error", "%Fatal")


def check(name: str, runlog: str) -> int:
    sigs = SIGS[name]
    txt = Path(runlog).read_text()
    if PASS_BANNER in txt:
        print(f"MUTANT {name}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    if not any(m in txt for m in FAIL_MARKS):
        print(f"MUTANT {name}: run neither passed nor failed?! Gate FAIL.")
        return 1
    hit = [s for s in sigs if s in txt]
    if not hit:
        print(f"MUTANT {name}: FAILED but without the expected signature "
              f"{sigs} — infra failure, not a catch. Inspect. Gate FAIL.")
        return 1
    print(f"MUTANT {name} CAUGHT (signature {hit})")
    return 0


def main():
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
    print(f"MUTANT {name}: {mfile}: {old.strip()!r} -> {new.strip()!r}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
    main()
