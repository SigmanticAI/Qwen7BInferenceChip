#!/usr/bin/env python3
"""FFN-WALK RUNG 1 — the degenerate one-chunk toy (d_ffn = 64, NCHUNK = 1).

The first PROGRAM ever to run the walked FFN machinery (fence-2 closure,
seq_layer_walker2 @1d09270): under ONE fmt=1 descriptor the walker runs
LVLG -> USWI (swiglu job + the JC_GATE/JC_UP composites from the
DESCRIPTOR) -> WGF/WGJ (fuel-fetched Wg chunk, 8 gate jobs, S2_PSJ-framed
serializer stream into asu_swiglu's gate phase) -> WUF/WUJ (up phase) ->
the per-chunk product stage (S2_CSL..S2_CSW: fsrc_ext=SWGP, C-1 feeder,
act LOAD at row fp_dwn_base+0 = row 0) -> done. Host in the window:
NOTHING.

Scope fence, stated: NCHUNK=1 ONLY. At NCHUNK>=2 the product rows collide
with the x row in this minimal mask (fp_dwn_base == the FFN act base when
attention/QKV/OPROJ are absent) — the row-model extension (an FFN-x family
in the fp stack) is the named next design step, not skirted here. At
NCHUNK=1 the x row is fully consumed before the stage overwrites it — the
overwrite is the DESIGN (DOWN reads the product from these rows).

Golden (layer_offload.py:1315-1326, the proven host composition):
  comp_g = f16_grade(s_x * s_Wg)        comp_u = f16_grade(s_x * s_Wu)
  product = f16(silu(acc_g * comp_g) * (acc_u * comp_u))
  staged  = quant_rows_i8(product)      (codes + ONE C-1 row scale)
graded against the tile's OWN staged row (identity readbacks) + fs scale.
"""
from __future__ import annotations

import argparse
import json
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
import trace_to_fuel as t2f                                    # noqa: E402

fmt = eq.fmt
D_TOY = eq.D_TOY
FFN_MASK = (1 << fmt.EN_FPROJ) | (1 << fmt.EN_FFN)
# rung 3: the full composed back-half — FFN + DOWN(+epilogue) + RES2.
# NCHUNK=1 only: the proven host contract quantizes DOWN's act under ONE
# uniform scale (layer_offload treats the whole product vector as a single
# C-1 row); the walker's per-chunk staging matches it exactly when the
# vector IS one chunk. The NCHUNK>=2 uniform-scale design is rung 4.
FFN_DOWN_MASK = FFN_MASK | (1 << fmt.EN_DOWN) | (1 << fmt.EN_RES2)


def _tens_subblocks(image_dir: Path, d: int, name: str, chunk: int = 0):
    """The _wo_subblocks idiom for any weight tensor: decode the first
    d*d bytes of the resident L00_<name> region as d/8 WS sub-blocks —
    the SAME reinterpretation the walker's ktot x ntot record fetch and
    the MXE's WS ingest apply, which is the consistency the grade needs
    (the E-6 toys graded bit-exact on silicon with exactly this)."""
    man = json.loads((image_dir / "ddr_image.json").read_text())
    t = next(x for x in man["tensors"]
             if x["layer"] == 0 and x["tensor"] == name)
    img = (image_dir / "ddr_image.bin").read_bytes()
    base_b = t["base_64B"] * 64 + chunk * d * 64   # per-chunk carve
    blocks = []
    for j in range(d // 8):
        blk = img[base_b + 512 * j: base_b + 512 * (j + 1)]
        W = np.zeros((d, 8), dtype=np.int64)
        for p in range(8):
            for c in range(8):
                for r in range(8):
                    v = blk[(p * 8 + c) * 8 + r]
                    W[8 * p + r, c] = v - 256 if v >= 128 else v
        blocks.append(W)
    meta = json.loads((Path(man["weights_dir"]) / "meta.json").read_text())
    return int(t["base_64B"]), blocks, float(meta["layers"][0][f"s_{name}"])


def toy_ffn(d: int = D_TOY, seed: int = 7, image_dir: Path | None = None,
            d_ffn: int = 64, x_shift: int = 12):
    from apex_golden import compute as cpg                     # noqa: PLC0415
    at, tf = eq.at, eq.tf
    image_dir = Path(image_dir or eq.DEFAULT_IMAGE)
    rng = np.random.default_rng(seed)

    # the x row: modest f16 values -> C-1 codes + scale (the staged input).
    # x_shift is the E-6-style golden-search knob: the rung-3 caller walks
    # it up until the DOWN epilogue's deq stream fits the residual window.
    x_vals = rng.integers(-12000, 12001, d).astype(np.float64)              * (2.0 ** -x_shift)
    x_bits = eq.f64_to_f16_bits(x_vals)
    x_f = eq.f16_bits_to_f64(np.asarray(x_bits, dtype=np.uint16))
    x_codes_m, x_sb = at.quant_rows_i8(x_f[None, :])
    x_codes = np.asarray(x_codes_m[0], dtype=np.int64)
    s_x = float(eq.f16_bits_to_f64(np.array([x_sb[0]], dtype=np.uint16))[0])

    nchunk = d_ffn // 64
    accs_g, accs_u = [], []
    for c in range(nchunk):
        wg_base, wg_blocks, s_wg = _tens_subblocks(image_dir, d, "Wg", c)
        wu_base, wu_blocks, s_wu = _tens_subblocks(image_dir, d, "Wu", c)
        accs_g.append(np.concatenate(
            [cpg.gemm_i8_ksplit(x_codes[None, :], W)[0]
             for W in wg_blocks]).astype(np.int64))
        accs_u.append(np.concatenate(
            [cpg.gemm_i8_ksplit(x_codes[None, :], W)[0]
             for W in wu_blocks]).astype(np.int64))
    wg_base, _, s_wg = _tens_subblocks(image_dir, d, "Wg", 0)
    wu_base, _, s_wu = _tens_subblocks(image_dir, d, "Wu", 0)
    acc_g = np.concatenate(accs_g)
    acc_u = np.concatenate(accs_u)

    comp_g = tf.f16_grade(s_x * s_wg)
    comp_u = tf.f16_grade(s_x * s_wu)
    prod = eq.f16_bits_to_f64(eq.f64_to_f16_bits(
        tf.silu_apply(acc_g.astype(np.float64) * comp_g)
        * (acc_u.astype(np.float64) * comp_u)))
    # ONE C-1 row per chunk (the walker stages chunk c into row 1+c)
    p_codes_m, p_sb = at.quant_rows_i8(prod.reshape(nchunk, 64))
    st = dict(
        d=int(d), seed=int(seed), image_dir=str(image_dir),
        x_bits=np.asarray(x_bits, dtype=np.uint16), x_codes=x_codes,
        x_scale=int(x_sb[0]),
        wg_base=wg_base, wu_base=wu_base,
        comp_g=int(np.float32(comp_g).view(np.uint32)),
        comp_u=int(np.float32(comp_u).view(np.uint32)),
        d_ffn=int(d_ffn), nchunk=int(nchunk),
        p_codes=np.asarray(p_codes_m, dtype=np.int64).reshape(-1),
        p_scales=[int(v) for v in p_sb])
    if d_ffn == 64:
        # rung 3 golden: DOWN + epilogue + RES2, mirroring toy_e6c's
        # proven OPROJ-epilogue composition with Wd and the product row.
        from walk_fuel_proj import _deq_resid_fit             # noqa: PLC0415
        wd_base, wd_blocks, s_wd = _tens_subblocks(image_dir, d, "Wd")
        pc = st["p_codes"]
        acc_d = np.concatenate([cpg.gemm_i8_ksplit(pc[None, :], W)[0]
                                for W in wd_blocks]).astype(np.int64)
        amax = int(np.max(np.abs(acc_d), initial=0)) or 1
        scale, shift = at.calib_requant(amax)
        d8 = cpg.requant_i32_to_i8(acc_d, scale, shift).astype(np.int64)
        s_p = float(eq.f16_bits_to_f64(
            np.array([st["p_scales"][0]], dtype=np.uint16))[0])
        # C2: JOBC composites are fp16-GRADED fp32 — apex_layer_deq refuses
        # an ungraded word at job accept (comp_legal, err_jb sticky) and the
        # walk wedges with no deq consumer (RUNG3_WEDGE.md: the rung-3 seam
        # was exactly this line missing the grade both proven epilogue
        # paths apply — walk_fuel_proj grade_f32 at :749/:1242).
        comp_d = np.float32(tf.f16_grade(
            s_p * s_wd * float(1 << shift) / float(scale)))
        deq = d8.astype(np.float64) * float(comp_d)
        bad = [i for i, v in enumerate(deq) if not _deq_resid_fit(float(v))]
        if bad:
            st["down_unfit"] = True          # caller searches x_shift up
            return st
        r1_vals = rng.integers(-8000, 8001, d).astype(np.float64) * 2.0 ** -12
        r1_bits = eq.f64_to_f16_bits(r1_vals)
        r1_f = eq.f16_bits_to_f64(np.asarray(r1_bits, dtype=np.uint16))
        r2_bits = eq.f64_to_f16_bits(r1_f + deq)
        st.update(dict(down=True, wd_base=wd_base,
                       comp_d=int(comp_d.view(np.uint32)),
                       rq_d=(int(scale), int(shift)),
                       r1_bits=np.asarray(r1_bits, dtype=np.uint16),
                       r2_bits=np.asarray(r2_bits, dtype=np.uint16)))
    return st


def ffn_descriptor(step: dict) -> list[int]:
    d = step["d"]
    down = bool(step.get("down"))
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(d)
    w[fmt.W_MODEL0] = fmt.pack_model0(d, step["d_ffn"])
    w[fmt.W_MODEL1] = fmt.pack_model1(1, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask(FFN_DOWN_MASK if down else FFN_MASK)
    w[fmt.W_STEP] = fmt.pack_step(1, 0)
    w[fmt.W_TENS0 + fmt.TENS_WG] = step["wg_base"]
    w[fmt.W_TENS0 + fmt.TENS_WU] = step["wu_base"]
    w[fmt.W_JC0 + fmt.JC_GATE] = step["comp_g"]
    w[fmt.W_JC0 + fmt.JC_UP] = step["comp_u"]
    if down:
        w[fmt.W_TENS0 + fmt.TENS_WD] = step["wd_base"]
        w[fmt.W_JC0 + fmt.JC_DOWN] = step["comp_d"]
        # RQ slot map (seq_layer_walker2 rq_word): DOWN at W_RQ0+n_heads+1
        w[fmt.W_RQ0 + 2] = fmt.pack_rq(*step["rq_d"])
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=d) == fmt.ERR_NONE
    return w


def build_walk_ffn(step: dict, *, out_dir: Path, name: str) -> dict:
    from walk_fuel_proj import FUEL_MARK, FUEL_UNMARK, \
        splice_fuel_arm, splice_fuel_disarm                    # noqa: PLC0415
    g3, glo = eq.g3, eq.glo
    d = step["d"]
    s = eq.LayerScript(d)
    s.emit(f"// FFN-WALK RUNG 1 {name}: ONE fmt=1 walk, ONE KICK = LVLG + "
           f"USWI(desc JC) + fuel Wg chunk + 8 gate jobs (S2_PSJ-framed) + "
           f"fuel Wu chunk + 8 up jobs + the walker-staged swiglu product "
           f"(S2_CS*, fsrc=SWGP, act row 0). Host in the window: NOTHING.")
    eq._prologue(s)
    if step.get("down"):
        s.emit("// rung 3: pre-load r1 into the resid row — RES2 adds the "
               "DOWN epilogue's deq stream onto it (the E-6 idiom)")
        s.lptr(glo.BANK_RESID, 0)
        s.ldata(step["r1_bits"])
    s.emit("// [1] host stages the x row at act bank1 ROW 0 (the FFN act "
           "family base at this mask) — the o-act idiom")
    s.route(rdst=1, asrc=1)
    x_pairs = [g3.decompose_f16(int(b))[:2]
               for b in np.asarray(step["x_bits"], dtype=np.uint16)]
    g3.inject_jobs(s, x_pairs, 1)
    s.ess(step["x_scale"], 1)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.pmode(True)
    s.route(rdst=0, asrc=0)
    s.emit(f"// {FUEL_MARK}")
    s.csrr(eq.W_STATUS, eq.W_ST_FMTSUP, eq.FMT_SUP_LAYER << 12)
    eq.load_descriptor(s, ffn_descriptor(step))
    s.emit("// CYCMARK ffngo")
    s.csrw(eq.W_CTRL, 0x3)
    s.csrp(eq.W_STATUS, eq.W_ST_BUSY, 0x0)
    s.emit("// CYCMARK ffndone")
    s.csrr(eq.W_STATUS,
           eq.W_ST_BUSY | eq.W_ST_ESTK | eq.W_ST_ECODE, 0x0)
    s.csrw(eq.W_CTRL, 0x0)
    s.emit(f"// {FUEL_UNMARK}")
    s.emit("// [2] egress: ONE fs beat per chunk (the CS stage row "
           "scales), then the staged product rows — the fp stack places "
           "them at rows 1..nchunk (AFTER the x family).")
    for _ in range(step["nchunk"]):
        s.efs(0, 0)
    s.csrw(eq.L_CTRL, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for c in range(step["nchunk"]):
        for base in range(0, d, glo.MXE_N):
            eq.identity_readback(s, 1 + c, base)
    if step.get("down"):
        s.emit("// rung 3: read the r2 row back — the composed back-half's "
               "final product")
        s.lrptr(0)
        s.erd(d)
    s.pmode(False)
    eq._drain_and_check(s)

    x = eq._translate(s, note=f"FFN-WALK rung1 {name}")
    man = eq._emit(x, out_dir, name, dict(
        stage="walk_ffn1",
        kind="ONE WALK: fuel gate/up chunk + on-tile swiglu + walker-staged "
             "product; NCHUNK=1 (degenerate loop)",
        d=int(d), mask=f"{FFN_MASK:#x}",
        wg_base=step["wg_base"], wu_base=step["wu_base"],
        comp_g=f"{step['comp_g']:#010x}", comp_u=f"{step['comp_u']:#010x}",
        seed=step["seed"]))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    splice_fuel_arm(Path(man["path"]))
    splice_fuel_disarm(Path(man["path"]))
    return man


def grade_ffn(caps, step: dict) -> dict:
    d = step["d"]
    out = {}
    n = step["nchunk"] * d
    fs = [int(t["bits"]) for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    want_fs = list(step["p_scales"])
    out["fs"] = {"equal": fs == want_fs, "got": fs[:4], "want": want_fs}
    acc = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64)
    got = acc[:n]
    out["p_codes"] = {"equal": got.size == n
                      and np.array_equal(got, step["p_codes"]),
                      "n_diff": int(np.sum(got != step["p_codes"][:got.size]))
                      if got.size else -1}
    keys = ["fs", "p_codes"]
    if step.get("down"):
        gr = eq.grade_resid(caps, step["r2_bits"])
        out["r2_row"] = {"equal": bool(gr["equal"]),
                         "n_diff": int(gr.get("n_diff", -1))}
        keys.append("r2_row")
    out["equal"] = all(out[k]["equal"] for k in keys)
    return out


def toy_ffn_swapped(step: dict):
    """The JC-SWAP poison's golden: comp_g and comp_u exchanged in the
    DESCRIPTOR — the tile computes silu(g*comp_u)*(u*comp_g). Predicted
    from the same accs; the poison run must land EXACTLY here (localizes
    the descriptor JC path), and NOT on the healthy codes."""
    at, tf = eq.at, eq.tf
    cg = float(np.uint32(step["comp_g"]).view(np.float32))
    cu = float(np.uint32(step["comp_u"]).view(np.float32))
    prod = eq.f16_bits_to_f64(eq.f64_to_f16_bits(
        tf.silu_apply(step["acc_g"].astype(np.float64) * cu)
        * (step["acc_u"].astype(np.float64) * cg)))
    codes, sb = at.quant_rows_i8(prod.reshape(step["nchunk"], 64))
    return (np.asarray(codes, dtype=np.int64).reshape(-1),
            [int(v) for v in sb])


def run_discriminators(args, step, out, bridge, extra) -> bool:
    """walk-off (a mask-fence refusal) + the JC-swap poison — REQUIRED for
    the rung-1 claim. Under --down (rung 3) the arms compose with the
    epilogue: walk-off clears EN_DOWN but KEEPS RES2 (the E-7b pairing
    fence must refuse — either alone is the starvation class), the swap
    golden extends through the swapped products' epilogue under the
    HOST-LOADED rq_d/comp_d, and a third arm poisons JC_DOWN with a
    graded-but-wrong composite (the C2 seam this rung debugged: r2 must
    land EXACTLY on the poisoned prediction while fs/p_codes stay clean)."""
    ok = True
    down = bool(step.get("down"))
    # arm 1: walk-off — the identical program minus one load-bearing mask
    # bit. Rung 1: clear EN_FFN (no fuel family remains). Rung 3: clear
    # EN_DOWN, keep RES2 (fp_mask_ok pairing fence).
    off = dict(step)
    with tg.at_d(D_TOY):
        m_off = build_walk_ffn(off, out_dir=out, name="walk_ffn1_off")
    import json as _j
    on_mask = fmt.pack_mask(FFN_DOWN_MASK if down else FFN_MASK)
    off_mask = fmt.pack_mask((FFN_MASK | (1 << fmt.EN_RES2)) if down
                             else (1 << fmt.EN_FPROJ))
    rows = [_j.loads(l) for l in open(m_off["path"])]
    fixed = []
    kicked = False
    for r in rows:
        if (r.get("op") == "w" and r.get("d") == on_mask and not kicked):
            r = dict(r, d=off_mask)
            kicked = True
        fixed.append(r)
    assert kicked, "mask word not found for the off arm"
    # the off arm REFUSES at the kick: busy never rises, ESTK carries DESC.
    # Trim the program to kick + status expectation (the walk-off idiom).
    trim = []
    for r in fixed:
        trim.append(r)
        if r.get("op") == "w" and r.get("a") == 4188 and r.get("d") == 3:
            trim.append({"op": "r", "a": 4200, "m": 0xF01, "e": 0x500})
            break
    open(m_off["path"], "w").write(
        "\n".join(_j.dumps(x, separators=(",", ":")) for x in trim) + "\n")
    r_off = bridge.run_job(m_off["path"], executor=args.executor,
                           binary=args.binary if args.executor == "sim"
                           else None, extra_args=extra, timeout_s=600)
    off_ok = bool(r_off["ok"])
    print(f"  disc walk_off  : REFUSED={off_ok} (DESC sticky)")
    ok &= off_ok
    # arm 2: JC swap — descriptor composites exchanged
    swp = dict(step, comp_g=step["comp_u"], comp_u=step["comp_g"])
    with tg.at_d(D_TOY):
        m_swp = build_walk_ffn(swp, out_dir=out, name="walk_ffn1_jswap")
    r_s = bridge.run_job(m_swp["path"], executor=args.executor,
                         binary=args.binary if args.executor == "sim"
                         else None, extra_args=extra, timeout_s=args.timeout)
    p_codes, p_scales = toy_ffn_swapped(step)
    gstep = dict(step, p_codes=p_codes, p_scales=p_scales)
    moved = not np.array_equal(p_codes, step["p_codes"])
    fit = True
    if down:
        # the swapped products feed the UNCHANGED epilogue: the tile
        # requants Wd @ p_codes_swapped under the host-loaded rq_d and
        # deqs under the descriptor's comp_d — predict r2 exactly. An
        # unfit prediction (residual-window violation) is the tile's own
        # loud-refusal class instead; both shapes are RED.
        r2s, fit = _down_epilogue_predict(step, p_codes)
        if fit:
            gstep["r2_bits"] = r2s
            moved = moved or not np.array_equal(r2s, step["r2_bits"])
    if fit:
        g = grade_ffn(r_s["captures"], gstep)
        swap_red = bool(r_s["ok"] and g["equal"] and moved)
        print(f"  disc jc_swap   : predicted-move={moved} "
              f"landed-on-prediction={g['equal']} -> RED={swap_red}")
    else:
        # the tile refuses the unfit swapped subject instead of grading it
        swap_red = bool(not r_s["ok"])
        print(f"  disc jc_swap   : swapped epilogue unfit -> "
              f"REFUSED={swap_red} -> RED={swap_red}")
    ok &= swap_red
    if down:
        ok &= _run_dpoison(args, step, out, bridge, extra)
    return ok


def _down_epilogue_predict(step: dict, p_codes: np.ndarray,
                           comp_bits: int | None = None):
    """r2 of the DOWN+RES2 epilogue on arbitrary staged codes under the
    HOST-LOADED rq_d and a descriptor composite (default: the step's own):
    the tile's exact composition, golden's own functions. Returns
    (r2_bits, fits_window)."""
    from apex_golden import compute as cpg                     # noqa: PLC0415
    from walk_fuel_proj import _deq_resid_fit                  # noqa: PLC0415
    image_dir = Path(step["image_dir"])
    d = step["d"]
    _, wd_blocks, _ = _tens_subblocks(image_dir, d, "Wd")
    acc = np.concatenate([cpg.gemm_i8_ksplit(
        np.asarray(p_codes, dtype=np.int64)[None, :], W)[0]
        for W in wd_blocks]).astype(np.int64)
    scale, shift = step["rq_d"]
    d8 = cpg.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    cd = float(np.uint32(comp_bits if comp_bits is not None
                         else step["comp_d"]).view(np.float32))
    deq = d8.astype(np.float64) * cd
    fit = all(_deq_resid_fit(float(v)) for v in deq)
    r1_f = eq.f16_bits_to_f64(np.asarray(step["r1_bits"], dtype=np.uint16))
    r2 = np.asarray(eq.f64_to_f16_bits(r1_f + deq), dtype=np.uint16)
    return r2, fit


def _run_dpoison(args, step, out, bridge, extra) -> bool:
    """Rung-3 arm: JC_DOWN poisoned GRADED (x2, else /2 — first factor
    whose prediction fits the residual window and moves r2). The FFN half
    must stay CLEAN (fs + p_codes on the healthy golden) while r2 lands
    EXACTLY on the poisoned prediction — the C2 composite seam, localized.
    An ungraded word is the wedge class this rung closed; it is refused at
    the deq unit and cannot fly as a graded arm."""
    cd0 = float(np.uint32(step["comp_d"]).view(np.float32))
    pick = None
    for f in (2.0, 0.5):
        bits = int(np.float32(eq.tf.f16_grade(cd0 * f)).view(np.uint32))
        r2p, fit = _down_epilogue_predict(step, step["p_codes"], bits)
        if fit and not np.array_equal(r2p, step["r2_bits"]):
            pick = (f, bits, r2p)
            break
    if pick is None:
        raise SystemExit("REFUSE: no fitting graded comp_d poison "
                         "(x2 and /2 both unfit or unmoved)")
    f, bits, r2p = pick
    psn = dict(step, comp_d=bits)
    with tg.at_d(D_TOY):
        m_p = build_walk_ffn(psn, out_dir=out, name="walk_ffn1_dpoison")
    r_p = bridge.run_job(m_p["path"], executor=args.executor,
                         binary=args.binary if args.executor == "sim"
                         else None, extra_args=extra, timeout_s=args.timeout)
    g = grade_ffn(r_p["captures"], dict(step, r2_bits=r2p))
    red = bool(r_p["ok"] and g["equal"])
    print(f"  disc d_comp    : factor={f} landed-on-poisoned-r2="
          f"{g.get('r2_row', {}).get('equal')} ffn-half-clean="
          f"{g.get('p_codes', {}).get('equal')} -> RED={red}")
    return red


def main() -> int:
    import tile_exec_bridge as bridge                          # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(eq.DEFAULT_IMAGE))
    ap.add_argument("--out", default=str(REPO / "build" / "ffn_toy"))
    ap.add_argument("--executor", default="sim", choices=("sim", "hw"))
    ap.add_argument("--binary", default=str(
        REPO / "verif/f2sim/obj_ffn_b64_ddr1/f2sim"))
    ap.add_argument("--tile-div", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dffn", type=int, default=64)
    ap.add_argument("--down", action="store_true",
                    help="rung 3: compose DOWN+RES2 (NCHUNK=1 only)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--discriminate", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = toy_ffn(seed=args.seed, image_dir=Path(args.image),
                   d_ffn=args.dffn)
    if args.down:
        assert args.dffn == 64, "rung 3 is NCHUNK=1 only (uniform-scale)"
        for xs in range(12, 22):
            step = toy_ffn(seed=args.seed, image_dir=Path(args.image),
                           d_ffn=args.dffn, x_shift=xs)
            if step.get("down") and not step.get("down_unfit"):
                print(f"[rung3] subject fits at x_shift={xs}")
                break
        else:
            raise SystemExit("REFUSE: no fitting rung-3 subject in range")
    else:
        step.pop("down", None)
    with tg.at_d(D_TOY):
        m = build_walk_ffn(step, out_dir=out, name="walk_ffn1")

    extra = ()
    if args.executor == "sim":
        extra = (f"+ddr_image={args.image}/ddr_image.bin",
                 f"+ddr_regions={args.image}/ddr_image.regions.jsonl",
                 "+fuel_audit", f"+tile_div={args.tile_div}",
                 "+poll_limit=16000000")
    else:
        import remote_hw_exec                                  # noqa: PLC0415
        if not remote_hw_exec.attach(bridge, args):
            print("[ffn1] REFUSE: no remote config", file=sys.stderr)
            return 2

    t0 = time.time()
    r = bridge.run_job(m["path"], executor=args.executor,
                       binary=args.binary if args.executor == "sim" else None,
                       extra_args=extra, timeout_s=args.timeout,
                       cap_out=str(out / f"walk_ffn1.{args.executor}"
                                         f".cap.jsonl"))
    wall = round(time.time() - t0, 1)
    g = grade_ffn(r["captures"], step) if r["ok"] else {"equal": False}
    green = bool(r["ok"] and g.get("equal"))
    print(f"[walk_ffn1 @{args.executor}] rc={r['rc']} ok={r['ok']} "
          f"caps={len(r['captures'])} grade={g.get('equal')} {wall}s")
    if not green:
        for ln in (r.get("log") or "").splitlines():
            if "stall" in ln or "FAIL" in ln:
                print("   ", ln[:150])
        if r["ok"]:
            print("   grade detail:", {k: v for k, v in g.items()
                                       if k != "equal"})
    (out / f"ffn1_{args.executor}_report.json").write_text(json.dumps(
        dict(executor=args.executor, verdict=green, wall_s=wall,
             grade=str(g)), indent=1, default=str))
    if green and args.discriminate:
        # the accs ride along for the swap golden
        from apex_golden import compute as cpg                 # noqa: PLC0415
        wg_b, wg_blocks, _ = _tens_subblocks(Path(args.image), D_TOY, "Wg")
        wu_b, wu_blocks, _ = _tens_subblocks(Path(args.image), D_TOY, "Wu")
        ag, au = [], []
        for c in range(step["nchunk"]):
            _, gb, _ = _tens_subblocks(Path(args.image), D_TOY, "Wg", c)
            _, ub, _ = _tens_subblocks(Path(args.image), D_TOY, "Wu", c)
            ag.append(np.concatenate(
                [cpg.gemm_i8_ksplit(step["x_codes"][None, :], W)[0]
                 for W in gb]).astype(np.int64))
            au.append(np.concatenate(
                [cpg.gemm_i8_ksplit(step["x_codes"][None, :], W)[0]
                 for W in ub]).astype(np.int64))
        step["acc_g"] = np.concatenate(ag)
        step["acc_u"] = np.concatenate(au)
        green &= run_discriminators(args, step, out, bridge, extra)
    print(f"FFN-WALK RUNG 1 @{args.executor}: {'PASS' if green else 'FAIL'}")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
