#!/usr/bin/env python3
"""E-7ng — the FETCHED-GAMMA chain WITHOUT attention (the no-score arm).

The 2026-08-06 differential proved walked SCORE+PV faults on silicon while
every walked chain without it passes. E-7's actual novelty — the walker
FETCHING the norm's gamma from card DRAM (FGAM) and running NORM2 on the
epilogue's own r1 — does not need attention: OPROJ consumes the host-staged
o-act (walk_oproj is silicon-proven in exactly that shape). This program is
`build_walk_e7c` with SCORE|PV masked out and the egress drain re-derived:

    mask   E7C_MASK & ~(SCORE|PV)  =  OPROJ RES1 NORM2 NFEED NSRC FPROJ FGAM
    egress no s_c ss, no s_k/8xs_v fs, no PV o8 RO, no TIP —
           fs = [q_scale (pre-walk staging), r1_scale, h2_scale]
    grade  h2 codes (identity readbacks) + r1 row, SAME golden as E-7
           (r1/h2 derive from the staged o-act path, untouched by score)

GREEN on silicon = the fetched-gamma machinery is silicon-proven and E-7's
only blocker is the walked-attention defect. RED = the defect is broader
than attention. Either way the claim sharpens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import elane_walk_qstage as eq                                 # noqa: E402
import cap_decode as cd                                        # noqa: E402
import tile_geom as tg                                         # noqa: E402

fmt = eq.fmt
E7NG_MASK = eq.E7C_MASK & ~((1 << fmt.EN_SCORE) | (1 << fmt.EN_PV))


def build_walk_e7ng(step: dict, *, out_dir: Path, name: str) -> dict:
    from elane_norm_feed import LST_NFEED                      # noqa: PLC0415
    from walk_fuel_proj import FUEL_MARK, FUEL_UNMARK, \
        splice_fuel_arm, splice_fuel_disarm                    # noqa: PLC0415
    g3, glo = eq.g3, eq.glo
    d = step["d"]
    s = eq.LayerScript(d)
    s.emit(f"// E-7ng {name}: ONE fmt=1 walk, NO ATTENTION — FUEL-FED "
           f"OPROJ EPILOGUE + RES1 + NFEED + NORM2 with the GAMMA FETCHED "
           f"FROM CARD DRAM (toy {d}, H=1, T=1). The no-score arm of the "
           f"2026-08-06 walked-attention differential.")
    eq._prologue(s)
    # [1] staging WITHOUT the attention family. fp_r_attn = 0 when neither
    # SCORE nor PV is enabled (seq_layer_walker2.sv:686,707), so the row
    # stack shifts: o-act at ROW 0, the y-drain fp_top window at ROW 1.
    # No q pre-stage (it would collide with the o-act at row 0), no rope.
    s.lptr(glo.BANK_RESID, 0)
    s.ldata(step["x_bits"])
    s.route(rdst=1, asrc=1)
    o_pairs = [g3.decompose_f16(int(b))[:2]
               for b in g3.to16(np.asarray(step["gold_o8"],
                                           dtype=np.float64))]
    g3.inject_jobs(s, o_pairs, 1)          # MODE_QUANT (arg 3 is mode)
    s.ess(step["o_act_scale"], 1)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)  # ...but the act ROW is 0 now
    s.pmode(True)
    s.route(rdst=0, asrc=0)
    s.emit(f"// {FUEL_MARK}")
    # [2] the descriptor: E-7's image with SCORE|PV cleared
    s.csrr(eq.W_STATUS, eq.W_ST_FMTSUP, eq.FMT_SUP_LAYER << 12)
    eq.load_descriptor(s, eq.e7_descriptor(step, mask=E7NG_MASK))
    # [3] the one kick
    s.emit("// CYCMARK e7go")
    s.csrw(eq.W_CTRL, 0x3)
    s.csrp(eq.W_STATUS, eq.W_ST_BUSY, 0x0)
    s.emit("// CYCMARK e7done")
    s.csrr(eq.W_STATUS,
           eq.W_ST_BUSY | eq.W_ST_ESTK | eq.W_ST_ECODE, 0x0)
    s.csrw(eq.W_CTRL, 0x0)
    s.emit(f"// {FUEL_UNMARK}")
    s.lpoll(glo.LST_LDQ | glo.LST_RES, 0x0)
    s.lstat(0x1F00 | LST_NFEED | glo.LST_LDQ | glo.LST_RES, 0x0)
    # the level word the WALKER leaves: rope_en/rope_pos come from the
    # DESCRIPTOR (pos_m), not host staging — measured 0x545 and decoded
    # back through fmt.lctl to exactly the E-7 expectation
    s.csrr(eq.L_CTRL, 0xFFFFF,
           fmt.lctl(rope_en=1, rope_bank=0, ser_dst=1, fsrc_ext=0,
                    resid_arm=1, rope_pos=step["pos"], nsrc=0))
    # [4] egress WITHOUT attention: only the NFEED r1 scale and the
    # y-drain h2 scale ride the fs FIFO; nothing else was produced.
    for _ in range(2):
        s.efs(0, 0)
    # [5] h2 codes from the y-drain window — ROW 1 in the shifted stack
    s.csrw(eq.L_CTRL, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, d, glo.MXE_N):
        eq.identity_readback(s, 1, base)
    s.lrptr(0)
    s.erd(d)
    s.pmode(False)
    eq._drain_and_check(s)

    x = eq._translate(s, note=f"E-7ng {name}: fetched-gamma chain, "
                              f"NO attention")
    man = eq._emit(x, out_dir, name, dict(
        stage="walk_e7ng",
        kind="ONE WALK, ONE KICK, NO SCORE/PV: FUEL-FED OPROJ epilogue -> "
             "residual -> NFEED r1 -> NORM2 with WALKER-FETCHED gamma -> "
             "y-drain; host silent in the window",
        d=int(d), pos=step["pos"], rq_o=list(step["rq_o"]),
        comp=f"{step['comp_bits']:#010x}",
        walk_mask=f"{E7NG_MASK:#06x}", g2_base=step["g2_base"],
        seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    splice_fuel_arm(Path(man["path"]))
    splice_fuel_disarm(Path(man["path"]))
    return man


def grade_e7ng(caps, step: dict) -> dict:
    d = step["d"]
    out = {}
    fs = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    want_fs = [step["r1_scale"], step["h2f_scale"]]
    out["fs"] = {"equal": fs == want_fs, "n": len(fs), "want": want_fs,
                 "got": fs[:6]}
    acc = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64)
    h2 = acc[:d]
    out["h2_codes"] = {"equal": h2.size == d
                       and np.array_equal(h2, step["h2f_codes"])}
    gr = eq.grade_resid(caps, step["r1_bits"])
    out["r1_row"] = {"equal": bool(gr["equal"]),
                     "n_diff": int(gr.get("n_diff", -1))}
    out["equal"] = all(out[k]["equal"] for k in ("fs", "h2_codes", "r1_row"))
    return out


def main() -> int:
    import tile_exec_bridge as bridge                          # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(eq.DEFAULT_IMAGE))
    ap.add_argument("--out", default=str(REPO / "build" / "e7ng"))
    ap.add_argument("--executor", default="sim", choices=("sim", "hw"))
    ap.add_argument("--binary", default=str(
        REPO / "verif/f2sim/obj_e7fly_b64_ddr1/f2sim"))
    ap.add_argument("--tile-div", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image)
    e7 = eq.toy_e7(seed=args.seed + 20, image_dir=image_dir)
    with tg.at_d(eq.D_TOY):
        m = build_walk_e7ng(e7, out_dir=out, name="walk_e7ng")

    extra = ()
    if args.executor == "sim":
        extra = (f"+ddr_image={image_dir}/ddr_image.bin",
                 f"+ddr_regions={image_dir}/ddr_image.regions.jsonl",
                 "+fuel_audit", f"+tile_div={args.tile_div}",
                 "+poll_limit=16000000")
    else:
        import remote_hw_exec                                  # noqa: PLC0415
        if not remote_hw_exec.attach(bridge, args):
            print("[e7ng] REFUSE: no remote config", file=sys.stderr)
            return 2
        print("[e7ng] remote hw shim attached (clock gate ON)",
              file=sys.stderr)

    t0 = time.time()
    r = bridge.run_job(m["path"], executor=args.executor,
                       binary=args.binary if args.executor == "sim" else None,
                       extra_args=extra, timeout_s=args.timeout,
                       cap_out=str(out / f"walk_e7ng.{args.executor}"
                                         f".cap.jsonl"))
    wall = round(time.time() - t0, 1)
    g = grade_e7ng(r["captures"], e7) if r["ok"] else {"equal": False}
    green = bool(r["ok"] and g.get("equal"))
    print(f"[walk_e7ng @{args.executor}] rc={r['rc']} ok={r['ok']} "
          f"caps={len(r['captures'])} grade={g.get('equal')} {wall}s")
    if not r["ok"]:
        for ln in (r.get("log") or "").splitlines():
            if "stall" in ln or "FAIL" in ln:
                print("   ", ln[:150])
    (out / f"e7ng_{args.executor}_report.json").write_text(json.dumps(
        dict(executor=args.executor, mask=hex(E7NG_MASK), caps=len(
            r["captures"]), grade={k: (v if not isinstance(v, dict) else
                                       {kk: (vv if not isinstance(
                                           vv, np.ndarray) else "arr")
                                        for kk, vv in v.items()})
                                   for k, v in g.items()},
             wall_s=wall, verdict=green), indent=1, default=str))
    print(f"E7NG @{args.executor}: {'PASS' if green else 'FAIL'}")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
