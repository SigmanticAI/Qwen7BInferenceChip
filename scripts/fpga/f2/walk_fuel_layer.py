#!/usr/bin/env python3
# walk_fuel_layer.py — E-6: the WALKED o8 EPILOGUE, and the chained walk.
#
#   python3 scripts/fpga/f2/walk_fuel_layer.py selftest          # offline
#   python3 scripts/fpga/f2/walk_fuel_layer.py run               # the gate
#
# ══ WHAT THIS PROVES, EXACTLY ══════════════════════════════════════════════
# E-5 (walk_fuel_proj.py) proved a walked, FUEL-FED projection — fenced to
# QKV, requant_en=0, raw INT32 on the RO cap lanes, the HOST draining them.
# The o8 requant epilogue (pc_hasrq: PC_WOJ/PC_WDJ) was explicitly excluded
# and stated unproven. This driver closes that fence for OPROJ:
#
# CHAIN A ("oproj", mask exactly {FPROJ, OPROJ, RES1}):
#   ONE fmt=1 descriptor in which the SEQUENCER fetches the real 0.5B Wo
#   from card DRAM (its own fuel record), emits the activation family and
#   112 MXE descriptors WITH the requant epilogue (requant_en=1, the
#   RQ[H]-slot pair), pushes the serializer frame job (S2_PSJ — the
#   host-mode leg's own `ljob` verb), and the o8 codes leave on the TILE'S
#   OWN consumption path: MXE requant -> serializer -> LAYER deq (JC_OPROJ
#   composite) -> armed residual -> r1 in the row RAM. Between WALK_GO and
#   the done-poll the host issues NOTHING — no control write, no data-plane
#   op, not even an RO drain (the epilogue has none to drain). That is
#   STRONGER than E-5's silence claim, and predicate-checked on the
#   artefact.
#
# CHAIN B ("qkv_oproj", mask exactly {FPROJ, QKV, OPROJ, RES1}):
#   BOTH projection classes under ONE descriptor — 4 walker-issued fetch
#   records (Wq, Wk, Wv, Wo), 144 raw QKV jobs graded on the RO lanes
#   (E-5's own grade, host RO drain = disclosed data plane) THEN the o8
#   epilogue chain, with the per-step route override flipping RT_FPROJ ->
#   RT_FPRQ between the classes.
#
# ══ THE FENCE — WHAT IS AND IS NOT CLAIMED ═════════════════════════════════
#   * DOWN/FFN are NOT walked. PC_WDJ is the same pc class (pc_hasrq) and
#     inherits the S2_PSJ/RT_FPRQ machinery by construction, but the FFN
#     mask also walks gate/up — swiglu LAYER jobs no walked step can frame —
#     and the 0.5B DOWN act family is d_ffn/CFG_D = 76 stage rows, which no
#     31-row act bank holds (seq_layer_walker2.sv FP_M_* note, the exact-set
#     mask fence at S2_CHECK). Refused loudly, not degraded.
#   * NFEED/NORM2 are NOT in these chains: at d_model=896 > FEED_DM=64 the
#     C-1 feeder frames the fed row with PER-FRAME scales (B-FEED-WIDTH,
#     STEP_MATRIX blocker 5), so the walked norm would not be golden's
#     norm. The RTL admits {FPROJ,OPROJ,RES1,NFEED,NSRC} for the geometry
#     where it IS exact (d_model == FEED_DM); proving that walk is the toy
#     lane's follow-on, not this driver's claim.
#   * THE ACTIVATION IS HOST-STAGED PRE-WALK (act bank 1), exactly E-5's
#     documented host-loaded surface. Chain A's staged rows are the C-1
#     framed REAL attention output of 0.5B layer-0 at the committed prompt
#     (golden's own quant per 64-frame, per-frame amax 127 by construction
#     => squant MODE_QUANT identity — walk_fuel_proj's framing argument
#     verbatim). Chain B stages TWO real families — the 'h' row at bank-1
#     rows 0..13 for QKV and the SAME real attention row at rows 14..27
#     (HEAD's fp_oproj_base = fp_rows when QKV is in the mask) — so BOTH
#     projection classes contract their real operands in one walk.
#   * The epilogue calibration (RQ pair, JC composite) is HOST-COMPUTED and
#     descriptor-loaded — the fmt=1 model's own per-step calibration slots
#     (causally circular on-tile: calib_requant needs max|acc| of the very
#     GEMM the descriptor configures; seq_walker_pkg rq_scale note).
#   * The r1 grade is against golden's OWN epilogue functions on the SAME
#     operands, computed AFTER the run: acc = gemm_i8_ksplit, (scale,shift)
#     = calib_requant(max|acc|), o8 = requant_i32_to_i8, comp =
#     f16_grade(s_wo * 2^shift / scale) (the staged frames' scale is 1.0 by
#     the amax-127 framing, so ONE per-tensor composite is EXACT — the
#     fmt=1 W2_JC design's own scope), r1 = f16(X[T] + o8*comp). Zero baked
#     expectations: every r1 element leaves the tile as a produce-mode cap.
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif/top/l3"),
           str(REPO / "verif/seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_l3_vectors as g3                                    # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402
import tile_geom as tg                                         # noqa: E402
import trace_to_fuel as t2f                                    # noqa: E402
import cap_decode as cd                                        # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import compute as cp                          # noqa: E402
from apex_golden import transformer as tfm                     # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402
import gen_layer_ops as glo                                    # noqa: E402
from gen_layer_ops import LayerScript, _translate, _emit       # noqa: E402
import fuel_proj_05b as fp                                     # noqa: E402
import walk_fuel_proj as wfp                                   # noqa: E402

D_TILE = 64
MXE_N = 8
ACT_BANK = 1
F16_ONE = 0x3C00
W_CTRL, W_DPTR, W_DDATA, W_STATUS = wfp.W_CTRL, wfp.W_DPTR, wfp.W_DDATA, \
    wfp.W_STATUS
W_ST_BUSY, W_ST_ESTK, W_ST_ECODE = wfp.W_ST_BUSY, wfp.W_ST_ESTK, wfp.W_ST_ECODE

DEFAULT_IMAGE = REPO / "build/ddr_weights_05b"
DEFAULT_WORK = REPO / "build/walk_fuel_layer"
# Build hygiene (walk_fuel_proj's measured rule): a NEW --Mdir for the E-6
# RTL. Recipe: make -C verif/f2sim build D=64 DDR=1 OBJ=obj_e6r_b64_ddr1 \
#   VFLAGS_EXTRA="$(python3 -c 'import sys; sys.path.insert(0,"scripts/fpga/f2");
#                  import tile_geom as tg; print(tg.IMG_05B.defines())')"
DEFAULT_OBJ = "obj_e6r_b64_ddr1"

N_HEADS, N_KV_HEADS, HEAD_DIM = 14, 2, 64
D_MODEL = N_HEADS * HEAD_DIM            # 896
KV_DIM = N_KV_HEADS * HEAD_DIM          # 128
QKV = (("Wq", D_MODEL), ("Wk", KV_DIM), ("Wv", KV_DIM))

MASK_A = ((1 << fmt.EN_FPROJ) | (1 << fmt.EN_OPROJ) | (1 << fmt.EN_RES1))
MASK_B = MASK_A | (1 << fmt.EN_QKV)


# ═══════════════════ 1. the golden epilogue subject ════════════════════════

def golden_epilogue(a8: np.ndarray, Wo: np.ndarray, s_wo: float,
                    xrow_bits: np.ndarray) -> dict:
    """Golden's own o8 -> deq -> residual chain on the FRAMED operands.

    Public golden functions only, the gen_layer_ops._rederive recipe with
    the framed subject's activation scale (1.0 — every staged frame's amax
    is 127, so quant_rows_i8's s = f16(127/127) = 1.0 for EVERY frame, and
    the per-tensor composite is exact, not approximate).
    """
    acc = cp.gemm_i8_ksplit(a8[None, :].astype(np.int64),
                            np.asarray(Wo, dtype=np.int64))[0].astype(np.int64)
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = at.calib_requant(amax)
    o8 = at.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    assert int(np.max(np.abs(o8))) < 128, "o8 clip (calib target violated)"
    s_out = tfm.f16_grade(float(s_wo) * float(1 << shift) / float(scale))
    comp = int(np.float64(s_out).astype(np.float32).view(np.uint32))
    assert (comp & 0x1FFF) == 0 and 0 < ((comp >> 23) & 0xFF) < 255, \
        f"composite {comp:#010x} not positive-normal fp16-grade"
    r1_bits = f64_to_f16_bits(tfm._f16(
        f16_bits_to_f64(np.asarray(xrow_bits, dtype=np.uint16))
        + o8.astype(np.float64) * s_out))
    return {"acc": acc, "scale": int(scale), "shift": int(shift), "o8": o8,
            "s_out": float(s_out), "comp": comp,
            "r1_bits": np.asarray(r1_bits, dtype=np.uint16)}


def golden_subject(idir: Path, layer: int) -> dict:
    """The REAL 0.5B layer-`layer` rows this walk is about, framed the way
    the D=64 tile frames them (fuel_proj_05b.c1_frames)."""
    man = json.loads((idir / "ddr_image.json").read_text())
    wdir = Path(man["weights_dir"])
    model = fp.rt.GoldenModel(wdir)
    fxs = fp.golden_prefill(model, fp.PROMPT_IDS, fp.rt.TIER_MAP["kvq8"], 128)
    fx = fxs[layer]
    # layer input row X[T] (the row RES1 adds onto): layer 0's is the
    # embedding of the LAST prompt token; deeper layers' is r2 of the
    # previous layer (the session's own inter-layer bus).
    if layer == 0:
        xrow = np.asarray(model.embed_row(fp.PROMPT_IDS[-1]), dtype=np.float64)
    else:
        xrow = np.asarray(fxs[layer - 1].r2, dtype=np.float64)
    xrow_bits = np.asarray(f64_to_f16_bits(xrow), dtype=np.uint16)
    attn8, _ = fp.c1_frames(np.asarray(fx.attn, dtype=np.float64), D_TILE)
    h8, _ = fp.c1_frames(fp.activation_for(fx, "h"), D_TILE)
    Wo = np.load(wdir / f"L{layer:02d}_Wo.npy")
    s_wo = float(model.layers[layer].s_wo)
    bases = {t["tensor"]: t["base_64B"] for t in man["tensors"]
             if t["layer"] == layer}
    blocks = {n: np.asarray(np.load(wdir / f"L{layer:02d}_{n}.npy"),
                            dtype=np.int64) for n, _ in QKV}
    return {"man": man, "wdir": wdir, "bases": bases, "xrow_bits": xrow_bits,
            "attn8": np.asarray(attn8, dtype=np.int64),
            "h8": np.asarray(h8, dtype=np.int64),
            "Wo": np.asarray(Wo, dtype=np.int64), "s_wo": s_wo,
            "blocks": blocks}


# ═══════════════════ 2. the fmt=1 descriptor ═══════════════════════════════

def walk_descriptor(bases: dict, *, mask: int, rq: tuple[int, int],
                    comp: int, fproj: bool = True) -> list[int]:
    """fproj=False is DISCRIMINATOR A: the identical image with W2_MASK[14]
    cleared — the legacy walk path, whose measured class is the fetch park /
    operand starvation (STEP_MATRIX p_oproj/p_res1): no epilogue, no r1."""
    scale, shift = rq
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(HEAD_DIM)
    w[fmt.W_MODEL0] = fmt.pack_model0(D_MODEL, 0)
    w[fmt.W_MODEL1] = fmt.pack_model1(N_HEADS, N_KV_HEADS, 1)
    m = mask if fproj else (mask & ~(1 << fmt.EN_FPROJ))
    w[fmt.W_MASK] = fmt.pack_mask(m)
    w[fmt.W_STEP] = fmt.pack_step(1, 0)
    for t, name in ((fmt.TENS_WQ, "Wq"), (fmt.TENS_WK, "Wk"),
                    (fmt.TENS_WV, "Wv"), (fmt.TENS_WO, "Wo")):
        w[fmt.W_TENS0 + t] = bases[name]
    # the o8 epilogue's host-loaded calibration: RQ[H] (o-proj slot) and
    # the JC_OPROJ composite — the documented per-step slots.
    assert 1 <= scale <= 0xFFFF and 0 <= shift <= 31
    w[fmt.W_RQ0 + N_HEADS] = (shift << 16) | scale
    w[fmt.W_JC0 + fmt.JC_OPROJ] = comp
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=HEAD_DIM) == fmt.ERR_NONE, "illegal walk image"
    return w


def expected_qkv_jobs() -> list[tuple[str, int]]:
    out = []
    for name, n_total in QKV:
        for n0 in range((n_total + MXE_N - 1) // MXE_N):
            out.append((name, n0))
    return out


# ═══════════════════ 3. the program ════════════════════════════════════════

# CYCSTAMP: the sim executor prints its shell-cycle counter at any note
# containing this token (verif/f2sim/sim_main.cpp, the E-6 hook) — purely
# observational, so the walk window can be MEASURED without adding a single
# host op inside it. MEASURED FINDING (2026-08-05, the H-bisect): the
# host-free epilogue walk at the 0.5B width runs LONGER than the executor's
# 2,000,000-shell-cycle single-poll budget at tile_div=5 (E-5 never saw
# this: its RO-drain polls each got a fresh budget) — H<=12 fits, H>=13
# does not. This driver therefore runs at tile_div=2 (DECISION-LC-1:
# bit-exactness must hold at ALL ratios; 2 is the recipe-B1 sim ratio),
# which keeps the ONE done-poll inside the budget with margin.
GO_MARK = "CYCMARK E6R-PREGO"
DONE_MARK = "CYCMARK E6R-WALKDONE"
TILE_DIV = 2


def build_program(sub: dict, desc: list[int], out_dir: Path, name: str, *,
                  act8: np.ndarray, qkv_drain: bool,
                  act8_oproj: np.ndarray | None = None) -> dict:
    with tg.at_d(D_TILE):
        return _build_at_d(sub, desc, out_dir, name, act8=act8,
                           qkv_drain=qkv_drain, act8_oproj=act8_oproj)


def _build_at_d(sub, desc, out_dir, name, *, act8, qkv_drain,
                act8_oproj=None):
    rows = D_MODEL // D_TILE
    s = LayerScript(D_TILE)
    s.emit("// E-6 WALKED o8 EPILOGUE: one fmt=1 descriptor; the sequencer "
           "fetches DDR weights, drives the GEMMs WITH requant, frames the "
           "o8 stream, and the tile consumes it (deq -> residual -> r1).")
    g3.phase_a(s, tiers_used=(0,))
    g3.loader_phase(s)

    s.emit(f"// [host, pre-walk] stage the {rows} C-1 activation frames "
           f"into act bank {ACT_BANK} (amax-127 identity staging)")
    s.route(rdst=1, asrc=1)
    fams = [(0, act8)]
    if act8_oproj is not None:
        # HEAD's fp_oproj_base: with QKV in the mask the OPROJ act family
        # lives at bank-1 rows fp_rows.. — so a QKV+OPROJ walk carries TWO
        # real operand families (the 'h' row AND the attention row), and
        # the OPROJ grade needs no synthetic-operand caveat.
        fams.append((rows, act8_oproj))
    for base_r, fam in fams:
        for r in range(rows):
            rowv = fam[r * D_TILE:(r + 1) * D_TILE].astype(np.float64)
            pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(rowv)]
            g3.inject_jobs(s, pairs, 1)
            s.ess(F16_ONE, 1)
            s.aj(0, ACT_BANK, 0, 1, s.BPR, base_r + r)

    s.emit("// [host, pre-walk] preload the residual row RAM with the f16 "
           "LAYER INPUT row X[T] — the row RES1 adds onto (its legitimate "
           "role; LAYER_PTR bank 2, auto-inc). ser_dst / resid_arm / the "
           "route are NOT touched: the WALKER arms all of them (PC_LVLO + "
           "RT_FPRQ).")
    s.lptr(glo.BANK_RESID, 0)
    s.ldata(sub["xrow_bits"])

    s.emit(f"// {wfp.FUEL_MARK}")

    s.emit("// load the fmt=1 descriptor (64 words incl. the RQ[H]/JC_OPROJ "
           "calibration slots) and KICK.")
    s.csrw(W_DPTR, 0)
    for v in desc:
        s.csrw(W_DDATA, v)
    s.emit(f"// {GO_MARK}")
    s.csrw(W_CTRL, 0x3)

    if qkv_drain:
        njobs = len(expected_qkv_jobs())
        s.emit(f"// RO drain, {njobs} raw QKV jobs (data plane, E-5's "
               f"disclosed class — the 16-deep RO FIFO must be paced). The "
               f"OPROJ epilogue that follows produces NO RO traffic.")
        for _ in range(njobs):
            s.ero([0] * MXE_N, 1)

    s.emit("// the walk retired, and it retired CLEAN")
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.emit(f"// {DONE_MARK}")
    s.csrr(W_STATUS, W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    s.emit("// [host, post-walk] LAYER hygiene: no unit refused, all quiet "
           "(the walker's own S2_PJW already waited on LST_LDQ|LST_RES)")
    s.lpoll(glo.LST_LDQ | glo.LST_SWG | glo.LST_RES | glo.LST_ROPE, 0x0)
    s.lstat(glo.LST_ERR, 0x0)
    s.emit(f"// [host, post-walk] read r1 back: {D_MODEL} produce-mode caps "
           f"(LAYER_RDATA auto-inc). The walk's OUTPUT — never foreknown.")
    s.lrptr(0)
    s.pmode(True)
    s.erd(D_MODEL)
    s.pmode(False)
    g3.final_phase(s, 8)

    x = _translate(s, note=f"E-6 walked o8 epilogue: {name}")
    man = _emit(x, out_dir, name, dict(stage="walk_fuel_layer", d=D_TILE,
                                       qkv_drain=bool(qkv_drain),
                                       act_rows=rows))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    return man


def parse_cycles(log: str) -> dict:
    """The walk window from the executor's CYCSTAMP lines: shell cycles
    between the pre-GO stamp and the post-done-poll stamp (the GO write +
    the poll's final read ride inside — a handful of AXI ops)."""
    st = {m.group(1): int(m.group(2)) for m in
          re.finditer(r"CYCMARK CYCMARK (\S+) cyc=(\d+)", log)}
    out = {"stamps": st}
    if "E6R-PREGO" in st and "E6R-WALKDONE" in st:
        shell = st["E6R-WALKDONE"] - st["E6R-PREGO"]
        out["walk_shell_cycles"] = shell
        out["walk_tile_cycles"] = shell // (2 * TILE_DIV)
    return out


# ═══════════════════ 4. predicates and grading ═════════════════════════════

def hostfree_predicate(path: Path) -> dict:
    """CHAIN A's silence claim, checked on the artefact: between the WALK_GO
    write and the done-poll there is NOTHING — no write, no read, no poll,
    no cap. Strictly stronger than E-5's control-silence."""
    ops = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    go = [i for i, o in enumerate(ops)
          if o.get("op") == "w" and o.get("a") == (0x1000 | W_CTRL)
          and o.get("d") == 0x3]
    if len(go) != 1:
        raise SystemExit(f"REFUSE: {len(go)} WALK_GO writes in {path}")
    end = [i for i, o in enumerate(ops)
           if i > go[0] and o.get("op") == "poll"
           and o.get("a") == (0x1000 | W_STATUS)]
    if not end:
        raise SystemExit("REFUSE: no walk-done poll after WALK_GO")
    window = [o for o in ops[go[0] + 1:end[0]] if o.get("op") != "note"]
    return {"window_ops": len(window), "hostfree": not window,
            "offenders": window[:4]}


def search_poison(sub: dict, ge: dict) -> dict:
    """DISC B's byte flip, SEARCHED ON GOLDEN FIRST (the same vacuity rule
    as the rq delta): fuel_proj's argmax-|x| flip moves the raw INT32 by
    x_k*delta, but the o8 EPILOGUE quantizes acc at ~amax/126 per code —
    ~130k counts at the 896-deep contraction — so a single-byte acc move
    (<= 127*255) is often ABSORBED by the requant. Search k (by |x_k| desc)
    for the first flip whose delta crosses a rounding boundary of lane 0's
    o8 under the DESCRIPTOR's own (scale, shift); refuse if none does."""
    x, acc0 = sub["attn8"], int(ge["acc"][0])
    order = np.argsort(-np.abs(np.asarray(x)))
    Wo_col0 = sub["Wo"][:, 0]
    for k in (int(i) for i in order):
        if int(x[k]) == 0:
            break
        old_i8 = int(Wo_col0[k])
        new_u8 = (old_i8 & 0xFF) ^ 0xFF
        new_i8 = new_u8 - 256 if new_u8 >= 128 else new_u8
        acc_bad0 = acc0 + int(x[k]) * (new_i8 - old_i8)
        o8_bad0 = int(at.requant_i32_to_i8(
            np.array([acc_bad0], dtype=np.int64),
            ge["scale"], ge["shift"])[0])
        if o8_bad0 == int(ge["o8"][0]):
            continue
        # ... and the moved o8 must survive the f16 residual rounding too,
        # or the observable (r1) would still be unmoved.
        r1b0 = int(f64_to_f16_bits(tfm._f16(
            f16_bits_to_f64(np.asarray([sub["xrow_bits"][0]],
                                       dtype=np.uint16))
            + np.float64(o8_bad0) * ge["s_out"]))[0])
        if r1b0 != int(ge["r1_bits"][0]):
            return {"k": k, "lane": 0, "old": old_i8, "new": new_i8,
                    "x_k": int(x[k]),
                    "delta_acc": int(x[k]) * (new_i8 - old_i8),
                    "o8_lane0": int(ge["o8"][0]), "o8_bad_lane0": o8_bad0}
    raise SystemExit("REFUSE: no single-byte Wo flip in lane 0 moves the "
                     "golden o8 through the requant — disc B would be "
                     "vacuous at this calibration")


def flip_wo_byte(src: Path, dst: Path, base_64B: int, k: int, lane: int):
    """Apply the searched flip at the weight-stationary byte address —
    fuel_proj_05b.mutate_image's own layout arithmetic: byte (p*8+c)*8+r
    of the (n0=0, k0=0) block, p=k//8, r=k%8, c=lane."""
    img = bytearray(Path(src).read_bytes())
    p, r = k // 8, k % 8
    off = base_64B * 64 + (p * 8 + lane) * 8 + r
    img[off] ^= 0xFF
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(bytes(img))
    return off


def search_rq_delta(ge: dict) -> int:
    """DISC C's perturbation, SEARCHED ON GOLDEN FIRST (the elane gamma-disc
    rule: a delta the RNE absorbs would make the discriminator vacuous).
    Returns the smallest candidate rq_scale delta that MOVES the golden o8
    without clipping; refuses if none does."""
    for dlt in (max(1, ge["scale"] // 128), max(1, ge["scale"] // 32),
                max(1, ge["scale"] // 8), max(1, ge["scale"] // 4)):
        o8p = at.requant_i32_to_i8(ge["acc"], ge["scale"] + dlt, ge["shift"])
        if int(np.max(np.abs(o8p))) < 128 \
                and not np.array_equal(ge["o8"], o8p):
            return int(dlt)
    raise SystemExit("REFUSE: no searched rq_scale delta moves the golden "
                     "o8 without clipping — disc C would be vacuous")


def split_caps(caps: list) -> tuple[list, list]:
    """(ro_caps, lrd_caps) by capture tag."""
    ro = [c for c in caps if str(c.get("tag", "")).startswith("ro_")]
    lrd = [c for c in caps if str(c.get("tag", "")).startswith("lrd")]
    return ro, lrd


def grade_r1(lrd_caps: list, r1_bits: np.ndarray) -> dict:
    got = [int(c["value"]) & 0xFFFF for c in lrd_caps]
    want = [int(v) for v in np.asarray(r1_bits).ravel()]
    if len(got) != len(want):
        return {"error": f"{len(got)} r1 caps, expected {len(want)}"}
    bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
    return {"n": len(want), "bad": len(bad), "bit_exact": not bad,
            "first_bad": bad[:8],
            "got_head": got[:8], "want_head": want[:8]}


# ═══════════════════ 5. selftest ═══════════════════════════════════════════

def selftest() -> int:
    ok = []

    def chk(cond, msg):
        ok.append(bool(cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {len(ok)} {msg}")

    man = json.loads((DEFAULT_IMAGE / "ddr_image.json").read_text())
    bases = {t["tensor"]: t["base_64B"] for t in man["tensors"]
             if t["layer"] == 0}
    d = walk_descriptor(bases, mask=MASK_A, rq=(1234, 7), comp=0x3F000000)
    chk(fmt.check2(d[0], d[1], d[2], d[3], d[fmt.W_STEP],
                   cfg_d=HEAD_DIM) == fmt.ERR_NONE,
        "the chain-A descriptor is fmt=1 LEGAL at the twin's CFG_D")
    chk(d[fmt.W_MASK] == MASK_A
        and MASK_A == (1 << 14) | (1 << 6) | (1 << 7),
        "chain-A mask is EXACTLY {FPROJ, OPROJ, RES1} (an RTL exact set)")
    db = walk_descriptor(bases, mask=MASK_B, rq=(1234, 7), comp=0x3F000000)
    chk(db[fmt.W_MASK] == MASK_B == MASK_A | (1 << fmt.EN_QKV),
        "chain-B mask is EXACTLY {FPROJ, QKV, OPROJ, RES1}")
    off = walk_descriptor(bases, mask=MASK_A, rq=(1234, 7),
                          comp=0x3F000000, fproj=False)
    chk(fmt.check2(off[0], off[1], off[2], off[3], off[fmt.W_STEP],
                   cfg_d=HEAD_DIM) == fmt.ERR_NONE
        and off[fmt.W_MASK] == MASK_A & ~(1 << 14),
        "the walk-off discriminator differs by EXACTLY W2_MASK[14]")
    chk(d[fmt.W_RQ0 + N_HEADS] == (7 << 16) | 1234,
        "the RQ pair lands in slot RQ[H] (the o-proj slot the RTL reads)")
    chk(d[fmt.W_JC0 + fmt.JC_OPROJ] == 0x3F000000,
        "the composite lands in the JC_OPROJ slot")
    chk(D_MODEL // MXE_N == 112 and D_MODEL // MXE_N <= 255,
        "the o8 stream is 112 serializer beats — ONE frame (<= 255)")
    chk(D_MODEL <= 1024, "ONE base-0 residual window holds the row")
    # the golden epilogue chain on a synthetic subject is self-consistent
    rng = np.random.default_rng(0xE6)
    a8 = rng.integers(-127, 128, D_MODEL, dtype=np.int64)
    a8[0] = 127                     # every frame amax-127 not needed here:
    W = rng.integers(-127, 128, (D_MODEL, D_MODEL), dtype=np.int64)
    ge = golden_epilogue(a8, W, 0.01,
                         np.zeros(D_MODEL, dtype=np.uint16))
    o8_alt = at.requant_i32_to_i8(ge["acc"], ge["scale"], ge["shift"])
    chk(np.array_equal(ge["o8"], o8_alt) and ge["r1_bits"].shape == (D_MODEL,),
        "golden_epilogue: requant reproducible, r1 is one f16 row")
    # a perturbed rq pair must MOVE r1 (the discriminator's premise)
    dlt = search_rq_delta(ge)
    o8p = at.requant_i32_to_i8(ge["acc"], ge["scale"] + dlt, ge["shift"])
    chk(dlt >= 1 and not np.array_equal(ge["o8"], o8p),
        f"an rq_scale delta (+{dlt}, searched on golden first) moves the "
        f"golden o8 (disc C premise holds)")
    print("=" * 62)
    print(f"WALK_FUEL_LAYER SELFTEST: {'ALL PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


# ═══════════════════ 6. the gate ═══════════════════════════════════════════

def _run_files(files, binary, ddr_args, timeout, log_to: Path | None = None):
    r = bridge.run_job(files, executor="sim", binary=str(binary),
                       extra_args=ddr_args, timeout_s=timeout,
                       tile_div=TILE_DIV)
    if log_to is not None:
        # persist the executor's stdout (CYCMARK + INGESTMON lines live
        # there; the bridge only returns it in memory)
        log_to.write_text(r.get("log") or "")
    return r


def run(args) -> int:
    idir, work = Path(args.image), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    binary = REPO / "verif/f2sim" / args.obj / "f2sim"
    ddr_args = (f"+ddr_image={idir}/ddr_image.bin",
                f"+ddr_regions={idir}/ddr_image.regions.jsonl", "+fuel_audit")
    if args.mon:
        # perf/ingest-lane: arm the harness ingest monitor (needs an
        # APEX_INGEST_MON build; a plain binary refuses the plusarg — fail
        # loud). Pure observation: the gate's checks are unchanged.
        ddr_args += ("+ingest_mon",)

    print("[1] golden prefill + the epilogue subject …")
    sub = golden_subject(idir, args.layer)
    geA = golden_epilogue(sub["attn8"], sub["Wo"], sub["s_wo"],
                          sub["xrow_bits"])
    print(f"    rq=({geA['scale']},{geA['shift']}) comp={geA['comp']:#010x} "
          f"(chain B's OPROJ family is the SAME real attention row, staged "
          f"at bank-1 rows fp_rows.. per HEAD's fp_oproj_base)")

    # ── CHAIN A ─────────────────────────────────────────────────────────────
    print("[2] CHAIN A — walked fuel-fed OPROJ + o8 epilogue, host-free "
          "window …")
    dA = walk_descriptor(sub["bases"], mask=MASK_A,
                         rq=(geA["scale"], geA["shift"]), comp=geA["comp"])
    mA = build_program(sub, dA, work, "walk_oproj", act8=sub["attn8"],
                       qkv_drain=False)
    spA = wfp.splice_fuel_arm(Path(mA["path"]))
    hf = hostfree_predicate(Path(mA["path"]))
    print(f"    fuel arm spliced ({spA['inserted']} ops); walk window has "
          f"{hf['window_ops']} host ops -> "
          f"{'HOST-FREE' if hf['hostfree'] else 'NOT FREE'}")
    rA = _run_files([mA["path"]], binary, ddr_args, args.timeout,
                    log_to=work / "walk_oproj.sim.log")
    _, lrdA = split_caps(rA["captures"])
    resA = grade_r1(lrdA, geA["r1_bits"]) if rA["ok"] else \
        {"error": f"rc={rA['rc']}"}
    cycA = parse_cycles(rA["log"])
    audA = rA["log"].count("-> ok")
    print(f"    rc={rA['rc']} r1 bit-exact={resA.get('bit_exact')} "
          f"bad={resA.get('bad')} fuel_audit={audA} cyc={cycA}")

    # ── CHAIN B ─────────────────────────────────────────────────────────────
    print("[3] CHAIN B — {QKV + OPROJ} under ONE descriptor …")
    jobs = expected_qkv_jobs()
    dB = walk_descriptor(sub["bases"], mask=MASK_B,
                         rq=(geA["scale"], geA["shift"]), comp=geA["comp"])
    mB = build_program(sub, dB, work, "walk_qkv_oproj", act8=sub["h8"],
                       qkv_drain=True, act8_oproj=sub["attn8"])
    wfp.capture_ro(Path(mB["path"]))
    wfp.splice_fuel_arm(Path(mB["path"]))
    silB = wfp.silence_predicate(Path(mB["path"]))
    rB = _run_files([mB["path"]], binary, ddr_args, args.timeout,
                    log_to=work / "walk_qkv_oproj.sim.log")
    roB, lrdB = split_caps(rB["captures"])
    resBq = wfp.grade(roB, jobs, sub["h8"], sub["blocks"]) if rB["ok"] else \
        {"error": f"rc={rB['rc']}"}
    resBo = grade_r1(lrdB, geA["r1_bits"]) if rB["ok"] else \
        {"error": f"rc={rB['rc']}"}
    cycB = parse_cycles(rB["log"])
    print(f"    rc={rB['rc']} QKV {resBq.get('jobs')} jobs "
          f"bit-exact={resBq.get('all_bit_exact')} bad={resBq.get('bad')}; "
          f"OPROJ r1 bit-exact={resBo.get('bit_exact')}; control-silence "
          f"{silB['control_writes']} writes / {silB['ro_advances']} RO "
          f"advances; cyc={cycB}")

    # ── DISCRIMINATOR A: walk-off ───────────────────────────────────────────
    print("[4] DISC A — walk-off (W2_MASK[14] cleared): the legacy path "
          "must produce NO r1 …")
    dOff = walk_descriptor(sub["bases"], mask=MASK_A,
                           rq=(geA["scale"], geA["shift"]),
                           comp=geA["comp"], fproj=False)
    mOff = build_program(sub, dOff, work, "walk_oproj_off",
                         act8=sub["attn8"], qkv_drain=False)
    wfp.splice_fuel_arm(Path(mOff["path"]))
    rOff = _run_files([mOff["path"]], binary, ddr_args, args.timeout_off,
                      log_to=work / "walk_oproj_off.sim.log")
    off_red = not rOff["ok"]
    print(f"    rc={rOff['rc']} ok={rOff['ok']} -> "
          f"{'RED (legacy park/starvation class, as measured)' if off_red else 'GREEN — UNEXPECTED'}")

    # ── DISCRIMINATOR B: one poisoned DDR byte in Wo ────────────────────────
    print("[5] DISC B — one poisoned byte in the SEQUENCER-fetched Wo, "
          "SEARCHED on golden so the requant cannot absorb it …")
    mut = search_poison(sub, geA)
    off = flip_wo_byte(idir / "ddr_image.bin",
                       work / "bad_image" / "ddr_image.bin",
                       sub["bases"]["Wo"], mut["k"], mut["lane"])
    mut["byte_offset"] = off
    wfp.rewrite_region_sha(idir / "ddr_image.regions.jsonl",
                           work / "bad_image" / "ddr_image.regions.jsonl",
                           work / "bad_image" / "ddr_image.bin")
    # predict r1 THROUGH the epilogue: same rq/comp (descriptor-loaded from
    # the GOOD calibration), poisoned acc on the touched lane only.
    Wo_bad = sub["Wo"].copy()
    Wo_bad[mut["k"], mut["lane"]] += (mut["new"] - mut["old"])
    geBad = {"acc": cp.gemm_i8_ksplit(sub["attn8"][None, :],
                                      Wo_bad)[0].astype(np.int64)}
    o8_bad = at.requant_i32_to_i8(geBad["acc"], geA["scale"], geA["shift"])
    r1_bad = f64_to_f16_bits(tfm._f16(
        f16_bits_to_f64(sub["xrow_bits"])
        + o8_bad.astype(np.float64) * geA["s_out"]))
    pred_moves = int(np.count_nonzero(
        np.asarray(r1_bad) != np.asarray(geA["r1_bits"])))
    bad_args = (f"+ddr_image={work}/bad_image/ddr_image.bin",
                f"+ddr_regions={work}/bad_image/ddr_image.regions.jsonl",
                "+fuel_audit")
    rP = _run_files([mA["path"]], binary, bad_args, args.timeout,
                    log_to=work / "walk_oproj_poison.sim.log")
    _, lrdP = split_caps(rP["captures"])
    resP = grade_r1(lrdP, np.asarray(r1_bad)) if rP["ok"] else \
        {"error": "run failed"}
    resP_vs_good = grade_r1(lrdP, geA["r1_bits"]) if rP["ok"] else {}
    poison_red = bool(rP["ok"] and resP.get("bit_exact")
                      and not resP_vs_good.get("bit_exact")
                      and pred_moves >= 1)
    print(f"    predicted {pred_moves} moved r1 element(s); measured == "
          f"prediction: {resP.get('bit_exact')} ; != good r1: "
          f"{not resP_vs_good.get('bit_exact')} -> "
          f"{'RED by the predicted delta' if poison_red else 'NOT the predicted RED'}")

    # ── DISCRIMINATOR C: the epilogue calibration is LIVE on-tile ───────────
    print("[6] DISC C — a searched rq_scale delta in the descriptor: r1 "
          "must move by golden's OWN requant prediction …")
    rq_dlt = search_rq_delta(geA)
    o8_rq = at.requant_i32_to_i8(geA["acc"], geA["scale"] + rq_dlt,
                                 geA["shift"])
    r1_rq = f64_to_f16_bits(tfm._f16(
        f16_bits_to_f64(sub["xrow_bits"])
        + o8_rq.astype(np.float64) * geA["s_out"]))
    rq_moves = int(np.count_nonzero(
        np.asarray(r1_rq) != np.asarray(geA["r1_bits"])))
    if rq_moves == 0:
        raise SystemExit(f"REFUSE: rq_scale+{rq_dlt} moves o8 but not r1 — "
                         f"the discriminator would be vacuous")
    dC = walk_descriptor(sub["bases"], mask=MASK_A,
                         rq=(geA["scale"] + rq_dlt, geA["shift"]),
                         comp=geA["comp"])
    mC = build_program(sub, dC, work, "walk_oproj_rq", act8=sub["attn8"],
                       qkv_drain=False)
    wfp.splice_fuel_arm(Path(mC["path"]))
    rC = _run_files([mC["path"]], binary, ddr_args, args.timeout,
                    log_to=work / "walk_oproj_rq.sim.log")
    _, lrdC = split_caps(rC["captures"])
    resC = grade_r1(lrdC, np.asarray(r1_rq)) if rC["ok"] else \
        {"error": "run failed"}
    resC_vs_good = grade_r1(lrdC, geA["r1_bits"]) if rC["ok"] else {}
    rq_red = bool(rC["ok"] and resC.get("bit_exact")
                  and not resC_vs_good.get("bit_exact"))
    print(f"    delta +{rq_dlt}; predicted {rq_moves} moved element(s); measured == "
          f"prediction: {resC.get('bit_exact')} -> "
          f"{'RED by the predicted delta (epilogue params LIVE)' if rq_red else 'NOT the predicted RED'}")

    verdict = bool(rA["ok"] and resA.get("bit_exact") and hf["hostfree"]
                   and audA >= 1
                   and rB["ok"] and resBq.get("all_bit_exact")
                   and resBo.get("bit_exact") and silB["silent"]
                   and off_red and poison_red and rq_red)
    print("-" * 78)
    print(f"  A: walked o8 epilogue r1 bit-exact  : "
          f"{'PASS' if resA.get('bit_exact') else 'FAIL'}")
    print(f"  A: walk window is HOST-FREE         : "
          f"{'PASS' if hf['hostfree'] else 'FAIL'}")
    print(f"  B: QKV+OPROJ one-descriptor walk    : "
          f"{'PASS' if (resBq.get('all_bit_exact') and resBo.get('bit_exact')) else 'FAIL'}")
    print(f"  walk-off (mask bit cleared) is RED  : "
          f"{'PASS' if off_red else 'FAIL'}")
    print(f"  poisoned Wo byte is RED via r1      : "
          f"{'PASS' if poison_red else 'FAIL'}")
    print(f"  rq-pair perturbation is RED via r1  : "
          f"{'PASS' if rq_red else 'FAIL'}")
    print(f"  E-6 WALKED o8 EPILOGUE GATE         : "
          f"{'PASS' if verdict else 'FAIL'}")
    out = work / "walk_fuel_layer_result.json"
    out.write_text(json.dumps(
        {"verdict": verdict,
         "chain_a": {"r1": resA, "hostfree": hf, "cycles": cycA,
                     "rq": [geA["scale"], geA["shift"]],
                     "comp": f"{geA['comp']:#010x}"},
         "chain_b": {"qkv": {k: v for k, v in resBq.items()
                             if k != "per_job"},
                     "r1": resBo, "silence": silB, "cycles": cycB},
         "disc": {"walk_off_red": off_red,
                  "poison": {"mut": mut, "pred_moves": pred_moves,
                             "red": poison_red},
                  "rq": {"pred_moves": rq_moves, "red": rq_red}}},
        indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if verdict else 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--image", default=str(DEFAULT_IMAGE))
    r.add_argument("--work", default=str(DEFAULT_WORK))
    r.add_argument("--obj", default=DEFAULT_OBJ)
    r.add_argument("--layer", type=int, default=0)
    r.add_argument("--timeout", type=int, default=5400)
    r.add_argument("--timeout-off", type=int, default=600)
    r.add_argument("--mon", action="store_true",
                   help="arm the f2sim ingest monitor (+ingest_mon; "
                        "APEX_INGEST_MON build only)")
    a = p.parse_args()
    return selftest() if a.cmd == "selftest" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
