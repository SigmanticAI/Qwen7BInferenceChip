#!/usr/bin/env python3
# elane_walk_norm.py — E-lane E-3 demonstrator (E2E_TOY_LANE.md §4):
# the SEQUENCER, not the host, arms the in-tile residual -> norm chain.
#
# ══ WHAT THIS PROVES (and what it does not) ════════════════════════════════
# E-1/E-2 (commit 7c0d4a5) closed the residual->norm seam INSIDE the tile:
# apex_residual grew an egress job + fp16 stream, feeder input source code 4
# widens it into the C-1 quantizer, l_nsrc puts the resulting INT8 codes on
# asu_rmsnorm's x port, and LAYER_JOB unit 3 is the start verb. But ONLY THE
# HOST CSR PATH could arm it. seq_layer_walker2's level net for fsrc_ext was
# 2 BITS — apex_top zero-extended it (`{1'b0, wk_lw_fsrc_ext}`) — so a WALKED
# program physically could not name code 4, and no walker step pushed the
# unit-3 job. The layer sequencer had therefore never driven this datapath:
# every run to date was host-driven.
#
# E-3 closes that. RTL: the level is 3 bits end to end (seq_walker_pkg
# walk2_lctl -> seq_layer_walker2 lw_fsrc_ext -> apex_top's walk-mode mux,
# which now copies all three), and PC_NFEED — gated by the previously
# reserved mask bit W2_EN_NFEED — issues the host demonstrator's exact verbs
# in the same order: arm code 4, push the C-1 feeder job, push LAYER_JOB
# unit 3, then hold until nf_busy (LAYER_STATUS[5]) falls.
#
# THIS PROGRAM runs, on the 128-wide toy configuration (E2E_TOY_LANE.md §2):
#     host: preload X, produce r1 = f16(X + o8*comp) via the host CSR path
#     host: arm l_nsrc, load the 64-word fmt=1 descriptor, WALK_GO
#     WALKER: LCTL fsrc_ext=4 -> FJOB rows -> LAYER_JOB unit 3 -> wait nf_busy
#             ... r1 (row RAM) -> C-1 feeder -> RMSNorm, host untouched ...
#     host: drain y through widen -> feeder -> act stage -> identity GEMM
# and grades the SAME references, computed by the SAME golden functions, as
# the host-driven demonstrator — elane_norm_feed.toy_step / grade_nfeed are
# IMPORTED, never re-derived, so a walked run and a host run are graded by
# one arbiter and any divergence is the walk's.
#
# HOST ROLE THAT REMAINS while the walker is driving (stated precisely, since
# that is the honest content of the claim):
#   * before WALK_GO: load X and gamma2 (weights/inputs), produce r1 (the
#     RES1 step is NOT in this fenced scope), arm l_nsrc — a build-shaped
#     route selector apex_top deliberately HOLDS across a walk (the
#     l_bias_en idiom) — set the rt_* routes the post-walk drain needs, load
#     the descriptor image, write WALK_CTRL.
#   * DURING the walk: nothing. The host's job/route/level ports are all
#     held not-ready by the walk_en mode mux; the three verbs that arm and
#     start the chain come from the sequencer.
#   * after the walk: drain the norm's OUTPUT (the y -> projections seam is
#     the next lane's, not this one's) and read back r1 for observability.
#
# NEW FILE ON PURPOSE: gen_layer_ops.py / narrow_flight.py / layer_offload.py
# / apex_repl.py are concurrently owned. This emitter IMPORTS the frozen
# vocabulary and the E-1/E-2 references and adds only the WALK-window words.
# Zero baked output expectations (audit_program enforced), golden the only
# arbiter, discriminators that must go red.
#
# CLI:
#   python3 scripts/fpga/f2/elane_walk_norm.py --selftest   # host only
#   python3 scripts/fpga/f2/elane_walk_norm.py --build      # emit programs
#   python3 scripts/fpga/f2/elane_walk_norm.py --smoke      # build+run+grade
#     [--binary PATH] [--tile-div N] [--out DIR] [--no-discriminate]
#   (RED evidence = --smoke --binary <pre-E-3 twin> --out <red dir>: the
#    walked route-arm poll can never be satisfied there, because a 2-bit
#    walker level cannot encode code 4.)

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif" / "top" / "l3"),
           str(REPO / "verif" / "seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_layer_ops as glo                                    # noqa: E402
import gen_l3_vectors as g3                                    # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
import elane_norm_feed as enf                                  # noqa: E402
from apex_golden import attention as at                        # noqa: E402

from gen_layer_ops import (                                    # noqa: E402
    LayerScript, D_TILE, MXE_N, BANK_RESID, LU_RESID, LU_DEQ,
    LST_LDQ, LST_RES, LST_SWG, L_CTRL,
    _translate, _emit, _prologue, _drain_and_check, inject_frame,
    identity_readback, eprint,
)
from elane_norm_feed import (                                  # noqa: E402
    LCTL_NSRC, LCTL_FSRC_EXT2, LST_NFEED, toy_step, grade_nfeed,
    disc_ran, stalled_at_poll, DISC_WITHHELD,
)

# ── the WALK CSR window (rtl/seq/seq_walker_pkg.sv:74-83) ───────────────────
W_CTRL, W_DPTR, W_DDATA, W_STATUS = 0x5C, 0x60, 0x64, 0x68
W_ST_BUSY = 0x0001          # WALK_STATUS[0]   walker busy
W_ST_ESTK = 0x0100          # WALK_STATUS[8]   error sticky
W_ST_ECODE = 0x0E00         # WALK_STATUS[11:9] walk_err_e
W_ST_FMTSUP = 0xF000        # WALK_STATUS[15:12] FMT_SUP nibble
FMT_SUP_LAYER = 0x3         # fmt 0 AND fmt 1 walk on this tile

DEFAULT_OUT = REPO / "build" / "f2_elane_walk"


# ═══════════════ the fmt=1 descriptor image ════════════════════════════════

def nfeed_descriptor(*, nfeed: bool = True) -> list[int]:
    """The 64-word fmt=1 image for a NORM-FEED-ONLY walk on the toy tile.

    Geometry: H=1, head_dim=128 == D_model == the build's CFG_D and CFG_DM,
    so `head_dim == D_model` is a CONSISTENT single-head layer (§2) and the
    walker's division-free row count (d_model / FEED_DM == n_heads == 1) is
    exact. Every other step is mask-disabled: this program's subject is the
    seam, not the layer, and a disabled pc costs the walker one cycle.

    `nfeed=False` builds the SAME image with the step-enable bit cleared —
    the walk-did-it discriminator. It must still be a LEGAL descriptor
    (en_mask != 0 is a refusal clause), so RES1 stands in as a step that
    pushes nothing without a deq stream behind it.
    """
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(D_TILE)             # fmt=1, tier CQ8
    w[fmt.W_MODEL0] = fmt.pack_model0(D_TILE, 0)        # d_model, d_ffn unused
    w[fmt.W_MODEL1] = fmt.pack_model1(1, 1, 1)          # H=1, H_kv=1, kv_map=01
    w[fmt.W_MASK] = fmt.pack_mask((1 << fmt.EN_NFEED) if nfeed
                                  else (1 << fmt.EN_RES1))
    w[fmt.W_STEP] = fmt.pack_step(1, 0)                 # t_rows=1, pos_m=0
    # legality is the WALKER's rule, mirrored here so an illegal image raises
    # at emission rather than being refused on the wire (the R3 discipline)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=D_TILE) == fmt.ERR_NONE
    return w


def load_descriptor(s: LayerScript, words: list[int]):
    """DPTR=0 then a 64-word DDATA burst (the pointer auto-increments)."""
    s.csrw(W_DPTR, 0)
    for v in words:
        s.csrw(W_DDATA, v)


# ═══════════════ the walked program ════════════════════════════════════════

def build_walk_stage(step: dict, *, out_dir: Path, name: str,
                     nfeed: bool = True, perturb: str | None = None) -> dict:
    """The E-3 chain: a WALKED program arms and completes residual -> norm.

    nfeed=False  the walk-did-it discriminator: identical program, the
                 step-enable bit cleared. The walker then arms nothing and
                 the route-arm poll must stall — the chain cannot happen by
                 host leftovers, because during walk_en the host's job and
                 level ports are all held not-ready.
    perturb='x'      one X element moved 1 fp16 ulp — r1 moves.
    perturb='gamma'  one gamma2 word moved — the NORM OUTPUT moves and r1
                     does NOT, which localizes the perturbation to the norm
                     and proves the captured codes really came through the
                     walked feed rather than from anything r1-shaped. The
                     delta is searched ON GOLDEN FIRST (the C-1 requant can
                     absorb a small gamma move and make the discriminator
                     vacuous), exactly as elane_norm_feed does.
    """
    x_bits = step["x_bits"].copy()
    g2 = step["gamma2"].copy()
    if perturb == "x":
        x_bits[5] = np.uint16(int(x_bits[5]) ^ 1)
    elif perturb == "gamma":
        for delta in (512, 2048, 8192, -8192, 16384):
            g2p = g2.copy()
            g2p[7] = int(np.clip(int(g2p[7]) + delta, -32768, 32767))
            h2p, _, _ = at.rmsnorm_fx([int(c) for c in step["r1_codes"]],
                                      [int(v) for v in g2p])
            cp, sp = at.quant_rows_i8(
                np.asarray(h2p, dtype=np.float64)[None, :] / 256.0)
            if (not np.array_equal(np.asarray(cp[0], dtype=np.int64),
                                   step["h2_codes"])
                    or int(sp[0]) != step["h2_scale"]):
                g2 = g2p
                break
        else:
            raise AssertionError("no gamma delta moved the golden reference")
    else:
        assert perturb is None, perturb

    s = LayerScript(D_TILE)
    s.emit(f"// E-LANE E-3 stage {name}: a WALKED fmt=1 program arms feeder "
           f"source code 4 and runs residual EGRESS -> C-1 -> RMSNorm with "
           f"the activation never leaving the tile. walk_en=1 throughout the "
           f"chain; the host's job/level/route ports are held not-ready by "
           f"the walk-mode mux while it runs.")
    _prologue(s)

    s.emit("// [1] host loads the LAYER INPUT row X into the residual RAM "
           "(LAYER_PTR bank 2, auto-inc) — its legitimate role")
    s.lptr(BANK_RESID, 0)
    s.ldata(x_bits)

    s.emit("// [2] host CSR path PRODUCES r1 = f16(X + o8*comp). RES1 is not "
           "in E-3's fenced scope: the subject here is the seam the walker "
           "could not reach, so r1 arrives the way it already did.")
    s.lctl(ser_dst=1, resid_arm=1)
    s.route(rdst=1)
    s.pmode(True)
    s.lujob(LU_RESID, D_TILE)
    s.ljc(step["comp"])
    s.lujob(LU_DEQ, D_TILE)
    inject_frame(s, step["o8"])
    s.lpoll(LST_LDQ | LST_RES, 0x0)

    s.emit("// [3] pre-walk staging. The rt_* routes for the LATER drain are "
           "set NOW: the route token guards on whole-tile idleness, and from "
           "the norm-feed start until gamma arrives the norm is legitimately "
           "busy holding the row (that guard would park forever). This poll "
           "also drains the loader phase's slow feeder tail-off, so the "
           "walker's own feeder-job push finds the unit idle. l_nsrc is a "
           "build-shaped route selector apex_top HOLDS across a walk (the "
           "l_bias_en idiom), so the host arms it here — and nothing else: "
           "the level word's fsrc_ext field is written 0, so the ONLY way "
           "code 4 can appear below is the walker driving it.")
    s.route(rdst=0, asrc=0)
    s.csrw(L_CTRL, LCTL_NSRC)
    s.csrr(L_CTRL, LCTL_NSRC | 0xFF, LCTL_NSRC)

    s.emit("// [4] load the 64-word fmt=1 descriptor image and confirm the "
           "tile publishes FMT_SUP for it (WALK_STATUS[15:12])")
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, nfeed_descriptor(nfeed=nfeed))

    s.emit("// [5] WALK_GO. From this write until [7], the SEQUENCER drives: "
           "walk_en holds every host job port not-ready, and the walker's "
           "PC_NFEED step arms the level, pushes the C-1 feeder job, pushes "
           "LAYER_JOB unit 3 and waits on nf_busy.")
    s.csrw(W_CTRL, 0x3)                        # walk_en=1 | walk_go=1

    s.emit("// [6] the walked chain, observed exactly the way the host path "
           "observes it. (a) the ROUTE-ARM observable: LAYER_CTRL[7] is "
           "l_fsrc_ext[2], and in walk mode that register tracks the "
           "WALKER's level net every cycle — so this bit reading 1 is the "
           "walker having selected the residual egress. It is a POLL, not a "
           "read, because the walk is asynchronous to the bus; on a walker "
           "whose level net cannot encode code 4 it can never be satisfied "
           "and the executor aborts loudly. (b) nf_busy falls when the whole "
           "row has reached the norm. (c) the walk retires.")
    s.csrp(L_CTRL, LCTL_FSRC_EXT2, LCTL_FSRC_EXT2)
    s.lpoll(LST_NFEED, 0x0)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit("// checked: the walk retired CLEAN (no sticky, no code) and the "
           "route it armed is still the residual egress")
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrr(L_CTRL, LCTL_NSRC | LCTL_FSRC_EXT2 | 0x30,
           LCTL_NSRC | LCTL_FSRC_EXT2)
    s.lstat(0x1F00 | LST_NFEED | LST_LDQ | LST_SWG | LST_RES, 0x0)

    s.emit("// [7] leave walk mode and pop the r1 C-1 row scale — the value "
           "golden computes and DISCARDS at the NORM2 entry (RMSNorm is "
           "scale-invariant), graded here as provenance of the C-1 the norm "
           "actually consumed")
    s.csrw(W_CTRL, 0x0)
    s.efs(0, 0)                                # -> cap: quant_rows_i8(r1).scale

    s.emit("// [8] drain the norm's OUTPUT through the legacy host path — "
           "golden's own next step, h2_8 = quant_rows_i8(h2/256). Levels "
           "back to 0 (both touched paths quiescent after the [6] polls), "
           "act LOAD armed, feeder job, then gamma releases the emission.")
    s.csrw(L_CTRL, 0)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.fjob(1)
    s.grow_n(g2)
    s.efs(0, 0)                                # -> cap: quant_rows_i8(h2/256)

    s.emit("// [9] the h2 codes back through identity GEMMs (the ONLY act-"
           "stage egress; produce-mode caps). The route token's idle guard "
           "IS the synchronization point.")
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, D_TILE, MXE_N):
        identity_readback(s, 0, base)

    s.emit("// [10] r1 observability AFTER the walked feed (egress does not "
           "mutate the row): one bus read per element")
    s.lrptr(0)
    s.erd(D_TILE)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"E-LANE E-3 {name}: WALKED resid->C1->norm")
    return _emit(x, out_dir, name, dict(
        stage="walk_nfeed" if nfeed else "walk_nfeed_off",
        kind="host deq->residual; WALKED resid-EGRESS->feeder->rmsnorm; "
             "host widen->feeder->act->RO",
        cols=int(D_TILE), nfeed=bool(nfeed), perturb=perturb or "",
        walk_mask=f"{nfeed_descriptor(nfeed=nfeed)[fmt.W_MASK]:#06x}",
        jobc=f"{step['comp']:#010x}", seed=step["seed"]))


# ── WHERE THE E-3 REFUSAL EVIDENCE LIVES (deliberately NOT here) ───────────
# The new loud failure mode is seq_layer_walker2's NFEED envelope fence: a
# descriptor whose norm-feed step would ask the C-1 feeder for more rows than
# it accepts is refused at S2_CHECK with WALK_ERR_DESC, before any state
# changes — instead of being refused DOWNSTREAM (job_error) after the walker
# had taken the push handshake and committed to waiting on nf_busy, which
# would be a silent wedge.
#
# That fence is NOT REACHABLE ON THIS CL, and the honest thing is to say so
# rather than ship a probe that cannot fire: the walker frames d_model as
# n_heads feeder rows, WALK2_H_MAX caps n_heads at 30, and cl_apex
# instantiates FEED_ROWS_MAX = 31 (cl_apex.sv:260). A descriptor that could
# trip it is refused by walk_desc2_check's own H_MAX clause first.
#
# It IS reachable — and is proven RED/GREEN — at the unit level, where the
# TB's default instantiation has FEED_ROWS = 16: verif/seq_walker's
# `refuse2.ops` case (d), H=17 with EN_NFEED set, pkg-legal so the FENCE is
# what fires. `make -C verif/seq_walker run_refuse2`.


# ═══════════════ selftest / smoke ══════════════════════════════════════════

def _selftest() -> int:
    import tempfile
    ok = True

    def chk(name, cond, note=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" +
              (f" — {note}" if note and not cond else ""))

    print("[1] the descriptor image is walker-LEGAL and names the new step")
    w = nfeed_descriptor()
    chk("check2 accepts it",
        fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                   cfg_d=D_TILE) == fmt.ERR_NONE)
    chk("mask sets EN_NFEED and nothing else",
        w[fmt.W_MASK] == (1 << fmt.EN_NFEED), f"{w[fmt.W_MASK]:#x}")
    chk("EN_NFEED is bit 11 — the mask bit every legacy image left 0",
        fmt.EN_NFEED == 11 and not (fmt.EN_ALL >> fmt.EN_NFEED) & 1)
    chk("d_model == n_heads * head_dim == the build's C-1 row (the walker's "
        "division-free row count is exact here)",
        w[fmt.W_MODEL0] & 0xFFFF == D_TILE
        and (w[fmt.W_MODEL1] >> 8) & 0xFF == 1)
    woff = nfeed_descriptor(nfeed=False)
    chk("the walk-off discriminator image is legal and NFEED-free",
        fmt.check2(woff[0], woff[1], woff[2], woff[3], woff[fmt.W_STEP],
                   cfg_d=D_TILE) == fmt.ERR_NONE
        and not (woff[fmt.W_MASK] >> fmt.EN_NFEED) & 1)
    chk("the two images differ ONLY in the mask word",
        [i for i in range(fmt.DESC_WORDS) if w[i] != woff[i]]
        == [fmt.W_MASK])

    print("[2] the level word this walk must produce")
    chk("code 4 lives at LAYER_CTRL[7] and nowhere else",
        fmt.lctl(fsrc_ext=4) == LCTL_FSRC_EXT2)
    chk("no legacy code can set that bit",
        all(not (fmt.lctl(fsrc_ext=c) & LCTL_FSRC_EXT2) for c in range(4)))

    print("[3] the programs build and pass the zero-expectation audit")
    st = toy_step()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        mans = [build_walk_stage(st, out_dir=out, name="walk_nfeed"),
                build_walk_stage(st, out_dir=out, name="walk_off",
                                 nfeed=False)]
        for m in mans:
            a = m["audit"]
            chk(f"{m['name']}: audit clean (caps={a['caps']}, "
                f"caps_with_e={a['caps_with_e']}, "
                f"violations={a['violations']})",
                a["caps_with_e"] == 0 and a["violations"] == 0)
        chk("walked stage captures: 2 fs + 16*(MXE_N+1) RO + 128 ERD",
            mans[0]["caps"] == 2 + 16 * (MXE_N + 1) + D_TILE,
            f"got {mans[0]['caps']}")
        prog = Path(mans[0]["path"]).read_text()
        chk("the program writes WALK_CTRL and 64 descriptor words",
            prog.count(f'"a":{0x1000 + W_DDATA}') == fmt.DESC_WORDS
            and prog.count(f'"a":{0x1000 + W_CTRL}') == 2)
        chk("the program NEVER writes feeder source code 4 itself "
            "(the walker is the only source of that bit)",
            not any(json.loads(ln).get("a") == 0x1000 + 0x70
                    and json.loads(ln)["op"] == "w"
                    and json.loads(ln)["d"] & LCTL_FSRC_EXT2
                    for ln in prog.splitlines()))
        chk("the program pushes NO LAYER_JOB unit 3 itself",
            not any(json.loads(ln).get("a") == 0x1000 + 0x7C
                    and json.loads(ln)["op"] == "w"
                    and ((json.loads(ln)["d"] >> 12) & 3) == 3
                    for ln in prog.splitlines()))

    print("[4] the grader (imported from the host-driven demonstrator) "
          "discriminates")
    caps = []
    i = 0

    def cap(sem, val):
        nonlocal i
        caps.append({"tag": f"{sem}_{i}", "i": i, "sem": sem,
                     "value": int(val) & 0xFFFFFFFF})
        i += 1
    cap("fs", st["r1_scale"])
    for base in range(0, D_TILE, MXE_N):
        for j in range(MXE_N):
            cap(f"ro_w{j}", int(st["h2_codes"][base + j]))
    cap("fs", st["h2_scale"])
    for b in st["r1_bits"]:
        cap("lrd", int(b))
    chk("all-correct capture set grades equal", grade_nfeed(caps, st)["equal"])
    bad = [dict(c) for c in caps]
    for c in bad:
        if c["sem"] == "ro_w3":
            c["value"] = (c["value"] + 1) & 0xFFFFFFFF
            break
    chk("a moved norm-output code goes red", not grade_nfeed(bad, st)["equal"])

    print(f"ELANE-WALK SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_all(out_dir: Path, step: dict) -> list[dict]:
    return [build_walk_stage(step, out_dir=out_dir, name="walk_nfeed")]


def smoke(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = toy_step(args.seed)
    plan = build_all(out, step)

    # EXECUTOR (2026-08-02): glo._run hardcodes executor="sim", so the hw path
    # cannot go through it — remote_hw_exec.attach only intercepts
    # executor="hw". Call the bridge directly with the selected executor and
    # attach the remote shim first; a missing attach silently returns ZERO
    # captures (measured, twice, on 2026-07-31), so REFUSE rather than run.
    import tile_exec_bridge as bridge                            # noqa: PLC0415
    if args.executor == "hw":
        import remote_hw_exec                                    # noqa: PLC0415
        if not remote_hw_exec.attach(bridge, args):
            eprint("[elane] REFUSE: --executor hw but no remote config "
                   "(set APEX_F2_HOST / APEX_F2_KEY)")
            return 2
        eprint("[elane] remote hw shim attached (clock gate ON)")

    def runner(pth, capname):
        return bridge.run_job([pth], executor=args.executor,
                              binary=args.binary, tile_div=args.tile_div,
                              timeout_s=args.timeout,
                              cap_out=str(out / f"{capname}.cap.jsonl"))

    rows = []
    for man in plan:
        t0 = time.time()
        r = runner(man["path"], man["name"])
        wall = time.time() - t0
        row = dict(name=man["name"], stage=man["stage"], ok=bool(r["ok"]),
                   rc=r["rc"], caps=len(r["captures"]), wall=round(wall, 1),
                   audit=man["audit"], summary=r.get("summary"))
        row["_caps"] = r["captures"]
        if man["stage"] == "walk_nfeed":
            try:
                row["grade"] = grade_nfeed(r["captures"], step)
            except Exception as e:                             # noqa: BLE001
                row["grade"] = {"equal": False,
                                "error": f"{type(e).__name__}: {e}"}
        else:
            row["grade"] = {"equal": bool(r["ok"]),
                            "note": "checked refusal guard"}
        rows.append(row)
        eprint(f"  [{man['name']}] rc={r['rc']} ok={r['ok']} "
               f"caps={len(r['captures'])} {wall:.1f}s "
               f"grade={row['grade'].get('equal')}")

    disc = {}
    if not args.no_discriminate:
        # a perturbation verdict is only evidence against a GREEN control on
        # the SAME binary. MEASURED (2026-08-04): this suite pointed at a
        # D=64 twin aborted every stage in phase A with zero captures, and
        # the old fallbacks then claimed walk_off/x_ulp/cap_* caught=True
        # (vacuous — nothing ran) and gamma caught=False. See
        # elane_norm_feed's discriminator-verdict hygiene note.
        base = next(x for x in rows if x["stage"] == "walk_nfeed")
        if not (base["ok"] and base["grade"].get("equal")):
            eprint("[disc] WITHHELD: the unperturbed walk_nfeed control is "
                   "not green on this binary — perturbation runs would "
                   "prove nothing (the 2026-08-04 wrong-twin failure mode)")
            disc["withheld"] = dict(DISC_WITHHELD)
        else:
            eprint("[disc] running the perturbations (each MUST go red)")
            # (1) THE WALK IS WHAT DID IT: same program, step-enable bit
            #     cleared. The walker then arms nothing, and since walk_en
            #     holds every host job/level port not-ready there is no
            #     other way for the chain to run — the route-arm poll must
            #     stall. The verdict cites the failure POINT, not just rc:
            #     the executor must have aborted ON the LAYER_CTRL route-arm
            #     poll (any other failure is not this discrimination).
            m = build_walk_stage(step, out_dir=out, name="disc_walk_off",
                                 nfeed=False)
            r = runner(m["path"], m["name"])
            stalled = stalled_at_poll(r["log"], 0x1000 + L_CTRL)
            disc["walk_off"] = {"caught": bool(not r["ok"] and stalled),
                                "rc": r["rc"],
                                "stalled_at_route_arm": stalled}
            # (2) one X element moved 1 fp16 ulp: r1 AND the norm chain move
            m2 = build_walk_stage(step, out_dir=out, name="disc_x_ulp",
                                  perturb="x")
            r2 = runner(m2["path"], m2["name"])
            ran2 = disc_ran(r2)
            gx = grade_nfeed(r2["captures"], step) if ran2 else {}
            disc["x_ulp"] = {"caught": bool(ran2 and not gx["equal"]),
                             "ran": ran2,
                             "r1_equal": gx.get("r1", {}).get("equal"),
                             "codes_equal": gx.get("h2_codes", {}).get("equal")}
            # (2b) one gamma2 word moved: ONLY the norm output moves (r1
            #      must still grade equal), which localizes the perturbation
            #      to the norm the WALKED feed fed — the x-ulp case above is
            #      absorbed by the C-1 requant on this row and moves r1
            #      alone.
            m3 = build_walk_stage(step, out_dir=out, name="disc_gamma",
                                  perturb="gamma")
            r3 = runner(m3["path"], m3["name"])
            ran3 = disc_ran(r3)
            gg = grade_nfeed(r3["captures"], step) if ran3 else {}
            disc["gamma"] = {"caught": bool(ran3 and not gg["equal"]
                                            and gg["r1"].get("equal")),
                             "ran": ran3,
                             "r1_equal": gg.get("r1", {}).get("equal"),
                             "codes_equal": gg.get("h2_codes", {}).get("equal")}
            # (3) host-side: a captured code / scale bit moved must go red
            #     (the control is green, so the capture set is real+graded)
            caps = base["_caps"]
            for key, sem in (("cap_code", "ro_w0"), ("cap_scale", "fs")):
                bad = [dict(c) for c in caps]
                hit = False
                for c in bad:
                    if c.get("sem") == sem:
                        c["value"] = (c["value"] + 1) & 0xFFFFFFFF
                        hit = True
                        break
                disc[key] = {"caught": bool(
                    hit and not grade_nfeed(bad, step)["equal"])}

    report = dict(binary=args.binary or "(bridge default)",
                  tile_div=args.tile_div, seed=step["seed"],
                  stages=[{k: v for k, v in r.items()
                           if not k.startswith("_")} for r in rows],
                  discriminators=disc)
    (out / "report.json").write_text(json.dumps(report, indent=1))

    print(f"\nE-LANE E-3 walked norm-feed @ tile_div={args.tile_div} "
          f"(binary={args.binary or 'bridge default'})")
    for r in rows:
        print(f"  {r['name']:<16} rc={r['rc']} ok={r['ok']} "
              f"caps={r['caps']:<4} grade={r['grade'].get('equal')}")
    for k, v in disc.items():
        print(f"  disc {k:<12} caught={v.get('caught')}")
    ok = all(r["ok"] and r["grade"].get("equal") for r in rows) \
        and all(v.get("caught") for v in disc.values())
    print(f"ELANE-WALK SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--binary", default=None,
                    help="f2sim twin (default: the bridge's resolution)")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim",
                    help="hw routes through remote_hw_exec (needs "
                         "APEX_F2_HOST/APEX_F2_KEY; clock verified per call)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--tile-div", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--no-discriminate", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.build:
        step = toy_step(args.seed)
        for m in build_all(Path(args.out), step):
            print(f"  built {m['name']}: {m['path']} "
                  f"(regops={m['regops']}, caps={m['caps']}, "
                  f"audit={m['audit']})")
        return 0
    if args.smoke:
        return smoke(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
