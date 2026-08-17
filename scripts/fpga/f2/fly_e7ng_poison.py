#!/usr/bin/env python3
"""The E-7ng gamma-poison arm ON HARDWARE — one flipped byte of the RESIDENT
g2 tensor in card DRAM. RED = h2 moves by the golden-predicted codes while
r1 holds (gamma is downstream of the residual), localizing the flip to the
walker-FETCHED gamma. The card's DRAM is restored and full-verified after.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import elane_walk_qstage as eq                                 # noqa: E402
import cap_decode as cd                                        # noqa: E402
import tile_geom as tg                                         # noqa: E402
import fly_e7ng as ng                                          # noqa: E402
from fly_e7_hw import ddr_reload                               # noqa: E402
from walk_fuel_proj import rewrite_region_sha                  # noqa: E402


def main() -> int:
    import tile_exec_bridge as bridge                          # noqa: PLC0415
    import remote_hw_exec                                      # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(eq.DEFAULT_IMAGE))
    ap.add_argument("--out", default=str(REPO / "build" / "e7ng"))
    ap.add_argument("--host", default=os.environ.get("APEX_F2_HOST"))
    ap.add_argument("--key", default=os.environ.get("APEX_F2_KEY"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    out = Path(args.out)
    image_dir = Path(args.image)
    if not remote_hw_exec.attach(bridge, args):
        print("[e7ng-poison] REFUSE: no remote config", file=sys.stderr)
        return 2

    e7 = eq.toy_e7(seed=args.seed + 20, image_dir=image_dir)
    with tg.at_d(eq.D_TOY):
        m = ng.build_walk_e7ng(e7, out_dir=out, name="walk_e7ng")

    # the mutated image: one g2 byte, golden-searched so h2 CODES move
    bad_dir = out / "bad_image_g2"
    bad_dir.mkdir(parents=True, exist_ok=True)
    gmut = eq._gamma_poison(e7, image_dir, bad_dir)
    rewrite_region_sha(image_dir / "ddr_image.regions.jsonl",
                       bad_dir / "ddr_image.regions.jsonl",
                       bad_dir / "ddr_image.bin")
    print(f"[e7ng-poison] byte {gmut['byte_offset']}: "
          f"{gmut['old']}->{gmut['new']} (g2[{gmut['k']}])")

    ld = ddr_reload(args.host, args.key, bad_dir, "ngbad",
                    manifest_from=image_dir)
    print(f"[e7ng-poison] bad image loaded: verify="
          f"{ld.get('verify', ld.get('verdict'))}")

    r = bridge.run_job(m["path"], executor="hw", binary=None,
                       extra_args=(), timeout_s=args.timeout,
                       cap_out=str(out / "walk_e7ng_gpoison.hw.cap.jsonl"))
    ran = eq.disc_ran(r)
    d = e7["d"]
    caps = r["captures"] if ran else []
    acc = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64) \
        if ran else np.zeros(0, dtype=np.int64)
    h2 = acc[:d]
    h2_moved_right = bool(ran and h2.size == d
                          and np.array_equal(h2, gmut["h2p_codes"])
                          and not np.array_equal(h2, e7["h2f_codes"]))
    gr = eq.grade_resid(caps, e7["r1_bits"]) if ran else {"equal": False}
    r1_held = bool(gr.get("equal"))
    red = bool(h2_moved_right and r1_held)
    print(f"[e7ng-poison] ran={ran} h2==predicted:{h2_moved_right} "
          f"r1_held={r1_held} -> RED={red}")

    print("[e7ng-poison] RESTORE: good image back into card DRAM")
    ld2 = ddr_reload(args.host, args.key, image_dir, "nggood",
                     full_verify=True)
    print(f"[e7ng-poison] restore verify={ld2.get('verify', ld2.get('verdict'))}")

    (out / "e7ng_poison_report.json").write_text(json.dumps(dict(
        mut={k: gmut[k] for k in ("byte_offset", "k", "old", "new",
                                  "v_old", "v_new")},
        ran=bool(ran), h2_moved_right=h2_moved_right, r1_held=r1_held,
        red=red, restore=ld2), indent=1, default=str))
    print(f"E7NG-POISON: {'RED (as designed)' if red else 'NOT CONCLUSIVE'}")
    return 0 if red else 1


if __name__ == "__main__":
    raise SystemExit(main())
