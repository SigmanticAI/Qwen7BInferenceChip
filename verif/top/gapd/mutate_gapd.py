#!/usr/bin/env python3
"""mutate_gapd.py — GAP-D signature-required mutation gate (l3 discipline).

Each mutant is a PATCHED COPY of rtl/top/apex_top.sv (the RTL is never
edited); each is CAUGHT only on its OWN specific FAIL signature, never on
any nonzero exit. All three run the SPLIT build (CFG_D=64, CFG_DM=128) on
the gapd_l4_h2_hd64 stream and prove the new CFG_DM parameter is
LOAD-BEARING in both directions:

  mDM1  feeder forced back to the PER-HEAD width (.D(CFG_DM) -> .D(CFG_D)):
        the C-1 h-row quantization frames 64-element rows, so the FIRST
        in-window EFS check mismatches with the exact scale the generator
        pre-computed (manifest mdm1.got/exp) — the model-wide family MUST
        ride CFG_DM.
  mDM2  act stage forced back to the per-head width (D and its nb port
        truncated to WK_NB_W, interface kept legal): the phase-B LOAD job's
        nb = CFG_DM/8 = 16 truncates to 0 -> illegal job refused -> the
        front-half stream wedges: "drive_x stall" with NO EFS mismatch —
        the resurrected F-5a failure, now impossible in the healthy split.
  mDM3  rope family forced onto the MODEL width (rope_row .D and the phase
        RAM L_HALF -> CFG_DM): the per-head 64-beat S-2 row hits rope_row's
        early-last frame check -> FRAME error (code 3) in LAYER_STATUS at
        the generator's t==0 probe: "[CSRR 80] got 00000700 exp 00000000" —
        the per-head family MUST stay CFG_D (§3c-3's head-pairing defect,
        pinned as a signature).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOP = REPO / "rtl" / "top" / "apex_top.sv"

MUTS = {
    "mDM1": [
        ("seam_feeder_quant #(.D(CFG_DM), .ROWS_MAX(FEED_ROWS_MAX)) u_feeder (",
         "seam_feeder_quant #(.D(CFG_D), .ROWS_MAX(FEED_ROWS_MAX)) u_feeder ("),
    ],
    "mDM2": [
        ("apex_stage_buf #(.D(CFG_DM), .R_MAX(STAGE_R_MAX)) u_astage (",
         "apex_stage_buf #(.D(CFG_D), .R_MAX(STAGE_R_MAX)) u_astage ("),
        # the F-5a wedge resurrected: nb values needing the widened bit
        # vanish to 0 (illegal job). Every bit stays consumed (-Wall).
        (".job_nb           (w_aj_nb),",
         ".job_nb           (w_aj_nb[STAGE_NB_W-1] ? WK_NB_W'(0)"
         " : WK_NB_W'(w_aj_nb)),"),
    ],
    "mDM3": [
        ("rope_row #(.D(CFG_D)) u_rope_row (",
         "rope_row #(.D(CFG_DM)) u_rope_row ("),
        ("localparam int unsigned L_HALF   = CFG_D / 2;",
         "localparam int unsigned L_HALF   = CFG_DM / 2;"),
    ],
}


def patch(mut: str, outdir: str) -> int:
    src = TOP.read_text()
    for old, new in MUTS[mut]:
        n = src.count(old)
        assert n == 1, f"{mut}: anchor not unique ({n}x): {old!r}"
        src = src.replace(old, new)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "apex_top.sv").write_text(src)
    print(f"{mut}: patched copy -> {out}/apex_top.sv "
          f"({len(MUTS[mut])} replacement(s))")
    return 0


def check(mut: str, logf: str, manifest: str) -> int:
    log = Path(logf).read_text(errors="replace")
    man = json.loads(Path(manifest).read_text())
    ok_pass = "L3 PASS" in log
    if ok_pass:
        print(f"{mut}: MISSED — the mutant run PASSED")
        return 1
    if mut == "mDM1":
        sig = (f"[EFS] got {man['mdm1']['got']:04x}/0 "
               f"exp {man['mdm1']['exp']:04x}/0")
        hit = sig in log
        why = f"first in-window EFS mismatch '{sig}'"
    elif mut == "mDM2":
        # the resurrected F-5a wedge: nb=CFG_DM/8 truncates to 0, the LOAD
        # is refused and the front-half INPUT stream (x or gamma — whichever
        # the RMSNorm backpressure reaches first) stalls; the EFS values
        # that do emerge are still correct (the feeder itself is intact).
        hit = (("drive_x stall" in log) or ("drive_g stall" in log)) \
            and ("[EFS] got" not in log)
        why = "front-half stall (drive_x/drive_g) with zero EFS mismatches"
    else:  # mDM3
        sig = "[CSRR 80] got 00000700 exp 00000000"
        hit = sig in log
        why = f"LAYER_STATUS FRAME (code 3) probe '{sig}'"
    if hit:
        print(f"{mut}: CAUGHT on its own signature — {why}")
        return 0
    print(f"{mut}: MISSED — died without the required signature ({why})")
    tail = "\n".join(log.splitlines()[-8:])
    print(f"{mut}: log tail:\n{tail}")
    return 1


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "patch":
        return patch(sys.argv[2], sys.argv[3])
    if cmd == "check":
        return check(sys.argv[2], sys.argv[3], sys.argv[4])
    print("usage: mutate_gapd.py patch <mut> <outdir> | "
          "check <mut> <log> <manifest>")
    return 2


if __name__ == "__main__":
    sys.exit(main())
