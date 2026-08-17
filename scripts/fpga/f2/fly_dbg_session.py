#!/usr/bin/env python3
"""The WALK_DBG differential card session — one command, four arms.

Flies the 2026-08-07 probe set on the A2+DBG image (WALK_DBG @ 0x98):

  arm 1  walk_e7ng_dbg0        clean control — must PASS with DBG==0
  arm 2  walk_e6_silicon_dbg   THE probe: the real failing walk with an
                               ESTK-gated WALK_DBG read; the run fails at
                               the done-poll (expected) but the walk_dbg
                               capture says WHICH composite fault fired:
                               1=err_frame, 2=err_stale, 3=both, 0=NEITHER
                               (0 would mean the SEQ came from somewhere
                               the differential has not mapped)
  arm 3  hostattn_fuelarm      host attention under fuel-arm — PASS
                               exonerates fuel arming; FAIL implicates it
  arm 4  walk_e6 (plain)       signature re-check on this image

Prereqs: AFI loaded + clkgen verified by remote_hw_exec's gate (the AGFI
must be registered in clock_key.py), DDR weights loaded (use
f2_ddr_load.py --load --verify first, or --skip-ddr-check here at your
peril — the probe REQUIRES resident weights: walk_e6 is fuel-fed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

PROBES = REPO / "docs" / "results" / "prompt_on_chip"


def main() -> int:
    import tile_exec_bridge as bridge                      # noqa: PLC0415
    import remote_hw_exec                                  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "build" / "dbg_session"))
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not remote_hw_exec.attach(bridge, args):
        print("[dbg-session] REFUSE: no remote config "
              "(APEX_F2_HOST/APEX_F2_KEY/APEX_F2_AGFI)", file=sys.stderr)
        return 2

    res = {}

    def fly(name, path, expect_ok):
        t0 = time.time()
        r = bridge.run_job(str(path), executor="hw", binary=None,
                           extra_args=(), timeout_s=args.timeout,
                           cap_out=str(out / f"{name}.hw.cap.jsonl"))
        wall = round(time.time() - t0, 1)
        row = dict(rc=r["rc"], ok=r["ok"], caps=len(r["captures"]),
                   wall_s=wall, as_expected=(r["ok"] == expect_ok))
        # pull the walk_dbg capture if the program took one
        dbg = [c for c in r["captures"] if c.get("tag") == "walk_dbg"]
        if dbg:
            row["walk_dbg"] = int(dbg[0]["value"]) & 3
        for ln in (r.get("log") or "").splitlines():
            if re.search(r"stall|FAIL", ln):
                row.setdefault("lines", []).append(ln.strip()[:140])
        res[name] = row
        print(f"[{name:<22}] rc={r['rc']} ok={r['ok']} caps="
              f"{len(r['captures'])} dbg={row.get('walk_dbg', '-')} "
              f"{wall}s", flush=True)
        return r

    fly("walk_e7ng_dbg0", PROBES / "walk_e7ng_dbg0.regops.jsonl", True)
    fly("walk_e6_silicon_dbg", PROBES / "walk_e6_silicon_dbg.regops.jsonl",
        False)
    fly("hostattn_fuelarm", PROBES / "hostattn_fuelarm.regops.jsonl", True)
    fly("walk_e6_plain", REPO / "build" / "e6c_bisect"
        / "walk_e6.regops.jsonl", False)

    dbg = res.get("walk_e6_silicon_dbg", {}).get("walk_dbg")
    verdict = {
        None: "NO walk_dbg capture — the ESTK poll never satisfied "
              "(walk completed?! or died pre-fault) — read the caps",
        0: "DBG=0: WALK_ERR_SEQ without a composite error — the "
           "differential's composite localization is WRONG; re-map",
        1: "err_frame: record framing misaligned at the walked score "
           "(tlast in the wrong beat slot)",
        2: "err_stale: the walked score found an unwritten scale-cache "
           "entry / missing s_q — matches the sim empty-cache repro",
        3: "BOTH frame and stale latched — multi-fault",
    }[dbg]
    ctl_ok = bool(res.get("walk_e7ng_dbg0", {}).get("ok"))
    fuel_ok = bool(res.get("hostattn_fuelarm", {}).get("ok"))
    summary = dict(
        arms=res, walk_dbg=dbg, verdict=verdict,
        control_clean=ctl_ok,
        fuel_arm_exonerated=fuel_ok,
        note=("control failed — image/bring-up suspect, do not interpret "
              "the probe" if not ctl_ok else
              "fuel-arm implicated: host attention faults under fuel-arm "
              "on this silicon" if not fuel_ok else
              "walk-mode composite interaction confirmed as the seam"))
    (out / "dbg_session_verdict.json").write_text(
        json.dumps(summary, indent=1, default=str))
    print(f"\nDBG SESSION: control={'OK' if ctl_ok else 'FAIL'} "
          f"fuel_arm={'exonerated' if fuel_ok else 'IMPLICATED'} "
          f"walk_dbg={dbg}\n  -> {verdict}")
    return 0 if ctl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
