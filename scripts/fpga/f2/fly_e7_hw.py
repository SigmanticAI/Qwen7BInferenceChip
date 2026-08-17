#!/usr/bin/env python3
"""E-7 / E-8 ON THE FPGA — the fetched-gamma one-kick chains on real silicon.

The sim gates (`elane_walk_qstage.py --chain7/--chain8`) drive the twin's
BEHAVIORAL DDR model through `+ddr_image` plusargs. On the card the weights
are PHYSICALLY resident in the DIMM (loaded + full-verified by
`f2_ddr_load.py` before this driver runs), so this driver:

  * builds the IDENTICAL programs from the same builders as the sim gate,
  * routes them through `remote_hw_exec` to the F2 with NO ddr plusargs,
  * grades them against the SAME golden,
  * and runs the gamma-poison arm by PHYSICALLY reloading the card's DRAM
    with the mutated image (then restoring the good one).

Refuse-loudly, everywhere: a missing `remote_hw_exec.attach` silently
returns ZERO captures (measured twice, 2026-07-31), so a failed attach
aborts. A poison arm that cannot prove the DDR reload actually happened is
reported as INCONCLUSIVE, never as RED.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import elane_walk_qstage as eq                                # noqa: E402
import cap_decode as cd                                       # noqa: E402
import tile_geom as tg                                        # noqa: E402
from seq_walker_fmt import __dict__ as _fmt_ns                 # noqa: E402,F401


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def _ssh_base(host: str, key: str) -> list[str]:
    # APEX_F2_HOST is the bare IP (remote_hw_exec keeps `user` separate);
    # ssh/scp need user@host, and an un-prefixed host silently resolves to
    # the LOCAL username on this Mac.
    if "@" not in host:
        host = f"{os.environ.get('APEX_F2_USER', 'ubuntu')}@{host}"
    return ["ssh", "-i", os.path.expanduser(key),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={os.path.expanduser('~/.apexmux')}/%C",
            "-o", "ControlPersist=300", host]


def ddr_reload(host: str, key: str, local_dir: Path, tag: str,
               *, full_verify: bool = False, manifest_from: Path = None) -> dict:
    """Ship a DDR image dir to the card and load it into the real DIMM.

    Returns the loader's own JSON verdict. Raises on any transport failure —
    a poison arm must never be graded off a DDR load that did not happen.
    """
    ssh = _ssh_base(host, key)
    uhost = ssh[-1]                       # user@host, normalized above
    scp = ["scp", "-i", os.path.expanduser(key),
           "-o", "StrictHostKeyChecking=no"]
    rdir = f"~/ddrswap_{tag}"
    subprocess.run(ssh + [f"mkdir -p {rdir}"], check=True, timeout=120)
    files = [str(local_dir / "ddr_image.bin")]
    for nm in ("ddr_image.json", "ddr_image.regions.jsonl"):
        p = local_dir / nm
        if not p.exists() and nm == "ddr_image.json" and manifest_from:
            # a poison dir carries only bin + regions; f2_ddr_load reads
            # ddr_image.json AND gates on its whole-image sha256 at load
            # plus PER-TENSOR sha256s at --full-verify, so the source
            # manifest must be re-hashed against the mutated bin on BOTH
            # levels — the integrity gate stays armed rather than bypassed
            # (measured: the un-rehashed manifest is REFUSED, 2026-08-06;
            # same discipline as the E-5 poison, SELF_RUNNING_CARD §notes).
            import hashlib                                     # noqa: PLC0415
            img = (local_dir / "ddr_image.bin").read_bytes()
            man = json.loads((manifest_from / nm).read_text())
            man["image_sha256"] = hashlib.sha256(img).hexdigest()
            for t in man.get("tensors", ()):
                b0, nb = t["base_64B"] * 64, t["beats_64B"] * 64
                t["sha256"] = hashlib.sha256(img[b0:b0 + nb]).hexdigest()
            p = local_dir / nm
            p.write_text(json.dumps(man, indent=1))
        if p.exists():
            files.append(str(p))
        elif nm == "ddr_image.json":
            raise RuntimeError(f"no ddr_image.json for {local_dir} — "
                               f"f2_ddr_load cannot load without a manifest")
    subprocess.run(scp + files + [f"{uhost}:{rdir}/"], check=True,
                   timeout=1800)
    vflag = " --full-verify" if full_verify else ""
    cmd = ("cd ~/aws-fpga && source sdk_setup.sh >/dev/null 2>&1; "
           f"sudo python3 ~/apexddr/scripts/fpga/f2/f2_ddr_load.py "
           f"--image {rdir} --load --verify{vflag} --out {rdir}/load.json "
           f"&& cat {rdir}/load.json")
    rr = subprocess.run(ssh + [cmd], capture_output=True, text=True,
                        timeout=3600)
    if rr.returncode != 0:
        raise RuntimeError(f"DDR load failed rc={rr.returncode}: "
                           f"{rr.stderr[-800:]}")
    txt = rr.stdout[rr.stdout.find("{"):]
    try:
        return json.loads(txt)
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(f"DDR load produced no verdict JSON: {exc}\n"
                           f"{rr.stdout[-800:]}") from exc


def fly(args) -> int:
    import tile_exec_bridge as bridge                          # noqa: PLC0415
    import remote_hw_exec                                      # noqa: PLC0415

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image)

    if not remote_hw_exec.attach(bridge, args):
        eprint("[e7hw] REFUSE: no remote config (APEX_F2_HOST / APEX_F2_KEY)")
        return 2
    eprint("[e7hw] remote hw shim attached (clock gate ON); weights assumed "
           "RESIDENT — f2_ddr_load full-verify must have passed")

    fmt = eq.fmt
    e7 = eq.toy_e7(seed=args.seed + 20, image_dir=image_dir)
    with tg.at_d(eq.D_TOY):
        m_c = eq.build_walk_e7c(e7, out_dir=out, name="walk_e7")
        m_off = eq.build_e7_refusal(
            e7, out_dir=out, name="walk_e7_off",
            mask=eq.E7C_MASK & ~(1 << fmt.EN_FGAM),
            why="NORM2 under FPROJ without FGAM — the E-6 gamma-poisoning "
                "refusal, kept verbatim")
        m_f1 = eq.build_e7_refusal(
            e7, out_dir=out, name="e7_fence_nofproj",
            mask=(1 << fmt.EN_FGAM) | (1 << fmt.EN_NORM2)
                 | (1 << fmt.EN_NFEED) | (1 << fmt.EN_NSRC),
            why="FGAM without FPROJ — the gamma fetch would park at S2_FETCH")
        m_f2 = eq.build_e7_refusal(
            e7, out_dir=out, name="e7_fence_nonorm",
            mask=eq.E7C_MASK & ~(1 << fmt.EN_NORM2),
            why="FGAM with no NORM step — a gamma window with no consumer")

    def _run(man, name, timeout):
        t0 = time.time()
        r = bridge.run_job(man["path"], executor="hw", binary=None,
                           extra_args=(), timeout_s=timeout,
                           cap_out=str(out / f"{name}.hw.cap.jsonl"))
        r["wall_s"] = round(time.time() - t0, 1)
        return r

    rows = {}

    # ---- the CLAIM run -------------------------------------------------
    eprint("[e7hw] CLAIM: walk_e7 — one kick, gamma FETCHED from card DRAM")
    r_c = _run(m_c, "walk_e7", args.timeout)
    g_c = eq.grade_e7c(r_c["captures"], e7) if r_c["ok"] else {"equal": False}
    green = bool(r_c["ok"] and g_c.get("equal"))
    eprint(f"  [walk_e7        ] rc={r_c['rc']} ok={r_c['ok']} "
           f"caps={len(r_c['captures'])} grade={g_c.get('equal')} "
           f"{r_c['wall_s']}s")
    rows["walk_e7"] = dict(ok=bool(r_c["ok"]), caps=len(r_c["captures"]),
                           grade=g_c, wall_s=r_c["wall_s"])

    # A claim run that never EXECUTED (clock gate refusal, dead transport)
    # makes every downstream arm meaningless — and the poison arm would
    # rewrite the card's DRAM for nothing. Stop here, loudly.
    if not r_c["ok"] and len(r_c["captures"]) == 0:
        eprint(f"[e7hw] ABORT: the claim run did not execute "
               f"(rc={r_c['rc']}, 0 captures). Not running the refusals, "
               f"and NOT touching the card's DRAM.")
        (out / "e7_hw_report.json").write_text(json.dumps(
            dict(where="fpga", host=args.host, agfi=args.agfi,
                 aborted="claim run did not execute", rc=r_c["rc"],
                 verdict=False), indent=1, default=str))
        return 3

    # ---- the three refusals -------------------------------------------
    for nm, man in (("walk_e7_off", m_off), ("e7_fence_nofproj", m_f1),
                    ("e7_fence_nonorm", m_f2)):
        r_x = _run(man, nm, args.timeout_off)
        eprint(f"  [{nm:<16}] rc={r_x['rc']} REFUSED={r_x['ok']} "
               f"caps={len(r_x['captures'])} {r_x['wall_s']}s")
        rows[nm] = dict(refused=bool(r_x["ok"]), caps=len(r_x["captures"]))

    # ---- the GAMMA POISON, with a PHYSICAL DDR reload ------------------
    poison_red = False
    if not args.skip_poison:
        from walk_fuel_proj import rewrite_region_sha              # noqa: PLC0415
        bad_dir = out / "bad_image_e7g"
        bad_dir.mkdir(parents=True, exist_ok=True)
        gmut = eq._gamma_poison(e7, image_dir, bad_dir)
        rewrite_region_sha(image_dir / "ddr_image.regions.jsonl",
                           bad_dir / "ddr_image.regions.jsonl",
                           bad_dir / "ddr_image.bin")
        eprint(f"[e7hw] POISON: reloading card DRAM with the mutated image "
               f"(byte {gmut['byte_offset']}: {gmut['old']}->{gmut['new']})")
        ld_bad = ddr_reload(args.host, args.key, bad_dir, "e7bad",
                            manifest_from=image_dir)
        eprint(f"  ddr reload (bad): {ld_bad.get('verdict', ld_bad)}")

        r_p = _run(m_c, "walk_e7_gpoison", args.timeout)
        ran_p = eq.disc_ran(r_p)
        caps_p = r_p["captures"] if ran_p else []
        acc_p = cd.ro_lanes_to_i32(caps_p, strict=False).astype(np.int64) \
            if ran_p else np.zeros(0, dtype=np.int64)
        d = e7["d"]
        h2_p = acc_p[d:2 * d] if acc_p.size >= 2 * d else np.zeros(0)
        h2_moved_right = bool(ran_p and h2_p.size == d
                              and np.array_equal(h2_p, gmut["h2p_codes"])
                              and not np.array_equal(h2_p, e7["h2f_codes"]))
        gr_p = eq.grade_resid(caps_p, e7["r1_bits"]) if ran_p else \
            {"equal": False}
        r1_held = bool(gr_p.get("equal"))
        poison_red = bool(green and h2_moved_right and r1_held)
        eprint(f"  [walk_e7_gpoison] rc={r_p['rc']} ran={ran_p} "
               f"h2==predicted:{h2_moved_right} r1_held={r1_held} "
               f"-> RED={poison_red}")
        rows["walk_e7_gpoison"] = dict(
            red=poison_red, ran=bool(ran_p), r1_held=r1_held,
            h2_moved_right=h2_moved_right,
            ddr_reload=ld_bad.get("verdict"),
            mut={k: gmut[k] for k in ("byte_offset", "k", "old", "new",
                                      "v_old", "v_new")})

        # RESTORE — every later run on this card must see the true weights.
        eprint("[e7hw] RESTORE: reloading the GOOD image into card DRAM")
        ld_good = ddr_reload(args.host, args.key, image_dir, "e7good",
                             full_verify=True)
        eprint(f"  ddr reload (good): {ld_good.get('verdict', ld_good)}")
        rows["ddr_restore"] = ld_good.get("verdict")
        if not ld_good.get("verdict"):
            eprint("[e7hw] WARNING: good-image restore did not verify — "
                   "the card's DRAM is NOT trustworthy for later runs")

    ok = (green
          and all(rows[n]["refused"] for n in
                  ("walk_e7_off", "e7_fence_nofproj", "e7_fence_nonorm"))
          and (poison_red or args.skip_poison))
    report = dict(where="fpga", host=args.host, image=str(image_dir),
                  agfi=args.agfi, seed=e7["seed"], mask=hex(eq.E7C_MASK),
                  stages=rows, verdict=bool(ok))
    (out / "e7_hw_report.json").write_text(json.dumps(report, indent=1,
                                                      default=str))
    print(f"\nE-7 ON THE FPGA  (agfi={args.agfi})")
    print(f"  walk_e7 (one kick, gamma FETCHED from card DRAM): "
          f"{'PASS' if green else 'FAIL'}")
    print(f"  refusals off/nofproj/nonorm: "
          f"{rows['walk_e7_off']['refused']}/"
          f"{rows['e7_fence_nofproj']['refused']}/"
          f"{rows['e7_fence_nonorm']['refused']}")
    if not args.skip_poison:
        print(f"  gamma poison (PHYSICAL DDR reload) RED: {poison_red}")
    print(f"E7-HW: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def fly_e8(args) -> int:
    """The E-8 arm: QKV composed into the fetched-gamma one-kick chain.

    Discriminated by the E-7 gate's own probes (which run in the same card
    session); this arm adds the COMPOSITION claim and its own grade.
    """
    import tile_exec_bridge as bridge                          # noqa: PLC0415
    import remote_hw_exec                                      # noqa: PLC0415

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not remote_hw_exec.attach(bridge, args):
        eprint("[e8hw] REFUSE: no remote config")
        return 2
    eprint("[e8hw] remote hw shim attached (clock gate ON)")

    e8 = eq.toy_e8(seed=args.seed + 20, image_dir=Path(args.image))
    with tg.at_d(eq.D_TOY):
        m_c = eq.build_walk_e8c(e8, out_dir=out, name="walk_e8")

    t0 = time.time()
    r_c = bridge.run_job(m_c["path"], executor="hw", binary=None,
                         extra_args=(), timeout_s=args.timeout,
                         cap_out=str(out / "walk_e8.hw.cap.jsonl"))
    wall = round(time.time() - t0, 1)
    g_c = eq.grade_e8c(r_c["captures"], e8) if r_c["ok"] else {"equal": False}
    green = bool(r_c["ok"] and g_c.get("equal"))
    detail = {k: g_c.get(k, {}).get("equal") for k in
              ("s_c", "fs", "qkv", "o8", "h2_codes", "r1_row")} \
        if isinstance(g_c, dict) else {}
    eprint(f"  [walk_e8] rc={r_c['rc']} ok={r_c['ok']} "
           f"caps={len(r_c['captures'])} grade={g_c.get('equal')} {wall}s")
    report = dict(where="fpga", host=args.host, agfi=args.agfi,
                  image=str(args.image), seed=e8["seed"],
                  mask=hex(eq.E8C_MASK), caps=len(r_c["captures"]),
                  grade=detail, wall_s=wall, verdict=bool(green))
    (out / "e8_hw_report.json").write_text(json.dumps(report, indent=1,
                                                      default=str))
    print(f"\nE-8 ON THE FPGA  (agfi={args.agfi})")
    print(f"  walk_e8 (QKV + attention + epilogue + FETCHED-gamma NORM2, "
          f"ONE kick): {'PASS' if green else 'FAIL'} {detail}")
    print(f"E8-HW: {'PASS' if green else 'FAIL'}")
    return 0 if green else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e8", action="store_true",
                    help="fly the E-8 composition arm instead of E-7")
    ap.add_argument("--image", default=str(eq.DEFAULT_IMAGE))
    ap.add_argument("--out", default=str(REPO / "build" / "e7_hw"))
    ap.add_argument("--host", default=os.environ.get("APEX_F2_HOST"))
    ap.add_argument("--key", default=os.environ.get("APEX_F2_KEY"))
    ap.add_argument("--agfi", default=os.environ.get("APEX_F2_AGFI", "?"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--timeout-off", type=int, default=900)
    ap.add_argument("--skip-poison", action="store_true")
    args = ap.parse_args()
    if not args.host or not args.key:
        eprint("[e7hw] REFUSE: --host/--key (or APEX_F2_HOST/APEX_F2_KEY) "
               "are required")
        return 2
    return fly_e8(args) if args.e8 else fly(args)


if __name__ == "__main__":
    raise SystemExit(main())
