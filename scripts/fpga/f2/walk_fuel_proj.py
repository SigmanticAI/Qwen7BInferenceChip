#!/usr/bin/env python3
# walk_fuel_proj.py — E-5, THE CONVERGENCE: a WALKED, FUEL-FED projection.
#
#   python3 scripts/fpga/f2/walk_fuel_proj.py selftest    # offline
#   python3 scripts/fpga/f2/walk_fuel_proj.py run         # the gate
#
# ══ WHAT THIS PROVES, EXACTLY ══════════════════════════════════════════════
# IB-FUEL proved the tile can eat DDR-resident weights when THE HOST drives
# the GEMM (fuel_proj_05b.py). IB-WALK proved the sequencer can drive the
# datapath when THE HOST supplies the operands (E-3b/E-4b). Neither crossed:
# STEP_MATRIX.md measured every walked projection parking at S2_FETCH.
#
# This driver runs ONE fmt=1 descriptor in which the SEQUENCER does all of:
#   * issues the fuel record for Wq/Wk/Wv from its OWN descriptor tensor
#     table (W2_TENS0), one per weight-consuming step — no host GO;
#   * drives the MXE's activation port itself (the act-row family EMITs that
#     S2_PJOB never pushed — STEP_MATRIX blocker 3);
#   * routes the MXE weight port onto the fuel stream (RT_FPROJ pins
#     rt_wgt_src=0 — STEP_MATRIX blocker 1);
# while the weight bytes travel DDR -> apex_fuel_reader -> apex_afifo -> the
# xw mux -> MXE, and the INT32 accumulators egress on the RO lanes and are
# graded, after the fact, against golden's own compute.gemm_i8_ksplit.
#
# ══ THE FENCE — WHAT IS AND IS NOT CLAIMED ═════════════════════════════════
#   * The walk is QKV-only (RTL fence: an EN_FPROJ mask must be exactly
#     {FPROJ, QKV}). OPROJ/FFN/DOWN are NOT walked here: their pc programs
#     push LAYER-unit jobs that no walked step can feed (the measured
#     RES1/OPROJ data-starvation classes).
#   * NO o8 EPILOGUE. The QKV steps carry pc_hasrq = 0, so the descriptors
#     are requant_en=0 and the RO lanes carry RAW INT32 accumulators — which
#     is exactly what the host-mode fuel proof grades, and what golden's
#     gemm_i8_ksplit computes. The o8 requant epilogue belongs to the OPROJ
#     and DOWN steps, which this fence excludes. It is NOT proven here.
#   * HOST-WRITE SILENCE is claimed for CONTROL, not for the data plane:
#     between WALK_GO and walk-done the program contains ZERO writes to any
#     CSR / LAYER level / job / mailbox-descriptor register. It DOES drain
#     the RO FIFO (poll + 9 reads + the advance write per job), because that
#     FIFO is 16 deep (apex_f2_mailbox.sv:153) and one MXE job produces 9
#     entries — the same data-plane class as E-4's "paced xw data stream",
#     and disclosed, not hidden. The predicate below CHECKS this by
#     inspecting the emitted regops, so it cannot rot.
#   * Fuel mode is armed BEFORE walk_en (a documented host-loaded field, like
#     the RQ/QC calibration slots). The RTL fences a walk whose fuel is not
#     armed at S2_CHECK; a fuel-mode drop MID-walk still wedges S2_FETCH and
#     is fenced, not fixed (seq_layer_walker2 S2_CHECK note).
#
# ══ THE ACTIVATION FRAMING ═════════════════════════════════════════════════
# Identical to fuel_proj_05b's, and for the identical reason: the C-1 feeder
# is elaborated at CFG_DM=64, so a 896-wide row is 14 independently-scaled
# 64-wide frames whose per-frame amax is 127 BY CONSTRUCTION, which makes
# squant's MODE_QUANT the identity with NO activation-dependent permutation.
# A resident, activation-independent weight tensor cannot feed the
# full-row-amax staging gemm_job.stage_plan uses. The reframing touches the
# ACTIVATION only; not one weight byte is re-quantized.

from __future__ import annotations

import argparse
import json
import struct
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
import make_weight_image as mwi                                # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402
import tile_geom as tg                                         # noqa: E402
import trace_to_fuel as t2f                                    # noqa: E402
import cap_decode as cd                                        # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import compute as cp                          # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402
from gen_layer_ops import (                                    # noqa: E402
    LayerScript, _translate, _emit,
    BANK_RESID, LST_LDQ, LST_RES, LST_ERR, grade_resid,
)
import fuel_proj_05b as fp                                     # noqa: E402

D_TILE = 64                       # the twin's CFG_D == CFG_DM
MXE_N = 8
ACT_BANK = 1                      # gemm_job.ACT_BANK — the staged-row bank
F16_ONE = 0x3C00

# WALK CSR window (rtl/seq/seq_walker_pkg.sv)
W_CTRL, W_DPTR, W_DDATA, W_STATUS = 0x5C, 0x60, 0x64, 0x68
W_ST_BUSY, W_ST_ESTK, W_ST_ECODE = 0x0001, 0x0100, 0x0E00

DEFAULT_IMAGE = REPO / "build/ddr_weights_05b"
DEFAULT_WORK = REPO / "build/walk_fuel_proj"
# BUILD HYGIENE, MEASURED THE HARD WAY (2026-08-04): rebuilding IN PLACE over
# an existing obj dir that was verilated with a DIFFERENT define set leaves
# stale objects behind — the resulting obj_b64_05b_ddr1 advertised INFO_TIER=3
# where a clean build of the SAME RTL and the SAME defines advertises 1, and
# 4 elane step-matrix probes went red on it while the identical RTL at DDR=0
# and the pre-change RTL at DDR=1 both stayed 10/10. Always verilate a NEW
# --Mdir for a new configuration; never trust an in-place rebuild.
DEFAULT_OBJ = "obj_e5_b64_ddr1"

# the walk's geometry: the real 0.5B decode shape at the twin's CFG_D
N_HEADS, N_KV_HEADS, HEAD_DIM = 14, 2, 64
D_MODEL = N_HEADS * HEAD_DIM            # 896
KV_DIM = N_KV_HEADS * HEAD_DIM          # 128
QKV = (("Wq", D_MODEL), ("Wk", KV_DIM), ("Wv", KV_DIM))


# ═══════════════════════ 1. the fmt=1 descriptor ═══════════════════════════

def walk_descriptor(bases: dict, *, fproj: bool = True,
                    extra_mask: int = 0) -> list[int]:
    """The 64-word fmt=1 image this walk runs.

    fproj=False is the WALK-OFF DISCRIMINATOR: the identical descriptor with
    W2_MASK[14] cleared. The RTL then takes the LEGACY projection path, whose
    measured behaviour is the S2_FETCH park (STEP_MATRIX) — no compute, no
    caps. extra_mask exercises the QKV-only fence.
    """
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(HEAD_DIM)
    w[fmt.W_MODEL0] = fmt.pack_model0(D_MODEL, 0)
    w[fmt.W_MODEL1] = fmt.pack_model1(N_HEADS, N_KV_HEADS, 1)
    mask = (1 << fmt.EN_QKV) | extra_mask
    if fproj:
        mask |= (1 << fmt.EN_FPROJ)
    w[fmt.W_MASK] = fmt.pack_mask(mask)
    w[fmt.W_STEP] = fmt.pack_step(1, 0)       # t_rows=1 -> every job is m=1
    for t, name in ((fmt.TENS_WQ, "Wq"), (fmt.TENS_WK, "Wk"),
                    (fmt.TENS_WV, "Wv")):
        w[fmt.W_TENS0 + t] = bases[name]
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=HEAD_DIM) == fmt.ERR_NONE, "illegal walk image"
    return w


def golden_activation(wdir: Path, layer: int, *, tier: str = "kvq8",
                      group: int = 128) -> tuple[np.ndarray, list]:
    """The REAL row Wq/Wk/Wv contract at `layer`, in this image's C-1 framing.

    Golden's own Session, golden's own activation_for, golden's own
    quant_rows_i8 per 64-wide frame — all borrowed verbatim from the
    host-mode proof so the two arms grade the SAME operand.
    """
    model = fp.rt.GoldenModel(wdir)
    fxs = fp.golden_prefill(model, fp.PROMPT_IDS, fp.rt.TIER_MAP[tier], group)
    return fp.c1_frames(fp.activation_for(fxs[layer], "h"), D_TILE)


def rewrite_region_sha(src_regions: Path, dst_regions: Path, bad_img: Path):
    """Re-digest every region against the MUTATED image.

    Verbatim in intent from fuel_proj_05b: the loader's per-region sha256
    would catch the flip before the tile ever ran, which would prove the
    LOADER works, not the FUEL path. Only digests change.
    """
    import hashlib
    dst_regions.parent.mkdir(parents=True, exist_ok=True)
    img = bad_img.read_bytes()
    with dst_regions.open("w") as f:
        for ln in src_regions.read_text().splitlines():
            r = json.loads(ln)
            b0, nb = r["base_64B"] * 64, r["beats_64B"] * 64
            r["sha256"] = hashlib.sha256(img[b0:b0 + nb]).hexdigest()
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def expected_jobs() -> list[tuple[str, int]]:
    """(tensor, n0) in the order the walker's pc program retires them —
    PC_WQF/WQJ, PC_WKF/WKJ, PC_WVF/WVJ, each n-major at N_MXE=8."""
    out = []
    for name, n_total in QKV:
        for n0 in range((n_total + MXE_N - 1) // MXE_N):
            out.append((name, n0))
    return out


# ═══════════════════════ 2. the program ════════════════════════════════════

FUEL_MARK = "FUELARM-SPLICE-POINT"
FUEL_UNMARK = "FUELDISARM-SPLICE-POINT"


def splice_fuel_disarm(path: Path) -> dict:
    """Insert the post-walk fuel DISARM at its marker: ONE ctrl write back
    to src=mailbox (reader idle + FIFO empty after a completed walk — the
    §5-legal mode switch). Needed by any program whose POST-walk drains
    ride the mailbox xw stream (the toy chain's identity readbacks):
    measured without it, the first post-walk pw push stalls dead and
    FUEL_ERR latches the host-push-during-fuel code."""
    ops = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    at = [i for i, o in enumerate(ops)
          if o.get("op") == "note" and FUEL_UNMARK in str(o.get("s", ""))]
    if len(at) != 1:
        raise SystemExit(f"REFUSE: {len(at)} fuel-disarm markers in {path}")
    i = at[0]
    arm = [
        {"op": "note", "s": "IB-FUEL disarm: src back to MAILBOX (reader "
                            "idle, FIFO empty — the walk consumed its "
                            "records whole)"},
        {"op": "w", "a": t2f.FUEL_CTRL, "d": 0},
    ]
    new = ops[:i + 1] + arm + ops[i + 1:]
    with path.open("w") as f:
        for o in new:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    return {"inserted": len(arm), "at": i + 1}


def build_program(x_codes: np.ndarray, desc: list[int], out_dir: Path,
                  name: str, *, njobs: int, drain: bool = True) -> dict:
    """Host prologue + activation staging + fuel arm + ONE walk + RO drain.

    Everything before the WALK_GO is the documented host-loaded surface. From
    WALK_GO to the busy poll the host issues NO control write — only the RO
    data drain (see the header fence).
    """
    # tile_geom.at_d rebinds gen_layer_ops/gemm_job's MODULE-LEVEL D_TILE,
    # BPR and ROWS_PER_DESC to this image's row width. Without it _translate
    # builds a LayerXlate at the default 128 and the loader's gamma push
    # stalls the xg FIFO (measured: "pw stall" at the GR beat) — the elane
    # rule, elane_walk_steps.py:495.
    with tg.at_d(D_TILE):
        return _build_program_at_d(x_codes, desc, out_dir, name,
                                   njobs=njobs, drain=drain)


def _build_program_at_d(x_codes, desc, out_dir, name, *, njobs, drain):
    rows = D_MODEL // D_TILE
    s = LayerScript(D_TILE)
    s.emit(f"// E-5 WALKED FUEL-FED PROJECTION: one fmt=1 descriptor, QKV, "
           f"weights from DDR via the walker's OWN fetch records.")
    g3.phase_a(s, tiers_used=(0,))
    g3.loader_phase(s)

    s.emit(f"// stage the {rows} C-1 activation frames into act bank "
           f"{ACT_BANK} (host-loaded, PRE-walk — the walker replays them)")
    s.route(rdst=1, asrc=1)
    for r in range(rows):
        rowv = x_codes[r * D_TILE:(r + 1) * D_TILE].astype(np.float64)
        pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(rowv)]
        g3.inject_jobs(s, pairs, 1)
        s.ess(F16_ONE, 1)                     # identity scale — a COUNTED check
        s.aj(0, ACT_BANK, 0, 1, s.BPR, r)     # LOAD -> act bank1 row r

    # the fuel arm is spliced in here as RAW regops (window 0x0 has no
    # LayerScript vocabulary) — see splice_fuel_arm
    s.emit(f"// {FUEL_MARK}")

    s.emit("// load the fmt=1 descriptor (64 words) and KICK. Nothing below "
           "this line writes a control register until the walk retires.")
    s.csrw(W_DPTR, 0)
    for v in desc:
        s.csrw(W_DDATA, v)
    s.csrw(W_CTRL, 0x3)                       # walk_en | walk_go

    if drain:
        s.emit(f"// RO drain, {njobs} jobs x (poll + 8 lanes + meta + adv). "
               f"DATA plane only: the RO FIFO is 16 deep and one job is 9 "
               f"entries, so it must be paced (header fence).")
        for _ in range(njobs):
            s.ero([0] * MXE_N, 1)             # -> caps; placeholders, not values

    s.emit("// the walk retired, and it retired CLEAN")
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.csrr(W_STATUS, W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    g3.final_phase(s, 8)

    x = _translate(s, note=f"E-5 walked fuel projection: {name}")
    man = _emit(x, out_dir, name, dict(stage="walk_fuel_proj", d=D_TILE,
                                       njobs=njobs, act_rows=rows))
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    return man


def capture_ro(path: Path) -> dict:
    """Rewrite every RO result READ into a `cap` — pure egress, no `e`.

    LayerXlate emits `ero` as CHECKED reads against the placeholder lanes the
    script passed (`[0]*8`), which would be a BAKED EXPECTATION of zero on
    the tile's own accumulators — the one thing this repo never does. This is
    GemmXlate.r's rewrite (gemm_job.py:274-279) applied as an explicit,
    auditable post-pass, because the LAYER vocabulary this walk needs and the
    result-capturing Xlate live in different translators.

    Tags follow the frozen contract `f"{sem}_{seq}"` with a per-FILE counter,
    so no two capture sites can collide (trace_to_regops.Xlate.cap).
    """
    lane_addr = {cd.RO_W0 + 4 * i: f"ro_w{i}" for i in range(cd.RO_LANES)}
    lane_addr[cd.RO_META] = "ro_meta"
    ops = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    seq, n = 0, 0
    for o in ops:
        if o.get("op") == "r" and o.get("a") in lane_addr:
            addr = o["a"]
            sem = lane_addr[addr]
            mask = o.get("m", 0xFFFFFFFF)
            o.clear()
            o.update({"op": "cap", "a": addr, "m": mask,
                      "tag": f"{sem}_{seq}"})
            seq += 1
            n += 1
    # the audit that makes this structural rather than hopeful
    bad = [o for o in ops if o.get("op") in ("r", "rn")
           and o.get("a") in lane_addr]
    if bad:
        raise SystemExit(f"REFUSE: {len(bad)} RO result expectation(s) "
                         f"survived, first {bad[0]} — the tile's accumulators "
                         f"must be captured, never asserted")
    with path.open("w") as f:
        for o in ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    return {"captures": n}


def splice_fuel_arm(path: Path) -> dict:
    """Insert the FUEL arm at the marker: poll ddr_ready, then ONE ctrl write
    that sets src=1 (fuel) + ar_owner=1 (the reader owns the DDR AR port).

    NO base/beats/GO here — that is the whole point. The host arms the MODE;
    the WALKER supplies every record from its descriptor tensor table.
    """
    ops = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    at = [i for i, o in enumerate(ops)
          if o.get("op") == "note" and FUEL_MARK in str(o.get("s", ""))]
    if len(at) != 1:
        raise SystemExit(f"REFUSE: {len(at)} fuel-arm markers in {path}")
    i = at[0]
    arm = [
        {"op": "note", "s": "IB-FUEL arm: mode only (src=1, ar_owner=1). "
                            "The walker issues the records itself."},
        {"op": "poll", "a": t2f.FUEL_STAT, "m": t2f.STAT_DDR_READY,
         "e": t2f.STAT_DDR_READY},
        {"op": "w", "a": t2f.FUEL_CTRL,
         "d": t2f.CTRL_AR_RUN | t2f.CTRL_SRC_FUEL},
    ]
    new = ops[:i + 1] + arm + ops[i + 1:]
    with path.open("w") as f:
        for o in new:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    return {"inserted": len(arm), "at": i + 1, "total": len(new)}


# ═══════════════ 3. the host-write-silence predicate ═══════════════════════

# every address a CONTROL write could land on. The RO advance (0x3228) and
# the note/poll/read ops are the disclosed data plane.
RO_ADV = 0x3228


def silence_predicate(path: Path) -> dict:
    """Between the WALK_GO write and the busy poll, assert that the ONLY
    writes are RO advances. This is the E-4b idiom, checked on the artefact
    rather than asserted in prose."""
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
    window = ops[go[0] + 1:end[0]]
    writes = [o for o in window if o.get("op") in ("w", "pw")]
    bad = [o for o in writes if o.get("a") != RO_ADV]
    return {"window_ops": len(window), "writes": len(writes),
            "ro_advances": len(writes) - len(bad),
            "control_writes": len(bad),
            "silent": not bad,
            "offenders": bad[:4]}


# ═══════════════════════ 4. grading ════════════════════════════════════════

def grade(caps: list, jobs: list, x_codes: np.ndarray, blocks: dict) -> dict:
    per, nbad = [], 0
    want = len(jobs) * (MXE_N + 1)
    if len(caps) != want:
        return {"error": f"{len(caps)} caps, expected {want}"}
    for i, (tensor, n0) in enumerate(jobs):
        acc = cd.ro_lanes_to_i32(caps[i * (MXE_N + 1):(i + 1) * (MXE_N + 1)],
                                 expect_n=MXE_N)
        blk = blocks[tensor][:, n0 * MXE_N:(n0 + 1) * MXE_N]
        gold = cp.gemm_i8_ksplit(x_codes[None, :].astype(np.int64), blk)[0]
        ok = bool(np.array_equal(np.asarray(acc, dtype=np.int64), gold))
        nbad += 0 if ok else 1
        per.append({"job": f"{tensor}_n{n0}", "ok": ok,
                    "acc": [int(v) for v in acc], "golden": [int(v) for v in gold]})
    return {"jobs": len(jobs), "bad": nbad, "all_bit_exact": nbad == 0,
            "per_job": per}


# ═══════════════════════ 5. selftest ═══════════════════════════════════════

def selftest() -> int:
    ok = []

    def chk(cond, msg):
        ok.append(bool(cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {len(ok)} {msg}")

    man = json.loads((DEFAULT_IMAGE / "ddr_image.json").read_text())
    bases = {t["tensor"]: t["base_64B"] for t in man["tensors"]
             if t["layer"] == 0}
    d = walk_descriptor(bases)
    chk(fmt.check2(d[0], d[1], d[2], d[3], d[fmt.W_STEP],
                   cfg_d=HEAD_DIM) == fmt.ERR_NONE,
        "the walk descriptor is fmt=1 LEGAL at the twin's CFG_D")
    chk((d[fmt.W_MASK] >> fmt.EN_FPROJ) & 1 and (d[fmt.W_MASK] >> fmt.EN_QKV) & 1
        and d[fmt.W_MASK] == ((1 << fmt.EN_QKV) | (1 << fmt.EN_FPROJ)),
        "the mask is EXACTLY {FPROJ, QKV} (the RTL fence)")
    off = walk_descriptor(bases, fproj=False)
    chk(fmt.check2(off[0], off[1], off[2], off[3], off[fmt.W_STEP],
                   cfg_d=HEAD_DIM) == fmt.ERR_NONE
        and not (off[fmt.W_MASK] >> fmt.EN_FPROJ) & 1,
        "the walk-off discriminator differs by EXACTLY W2_MASK[14]")
    chk(fmt.EN_FPROJ == 14 and fmt.EN_FGAM == 15 and fmt.EN_DOWN == 16
        and fmt.EN_W == 17,
        "the mask field grew 14 -> 15 (E-5) -> 17 (E-7/E-7b), each at the "
        "reserved bit")
    # bit 17 is the NEW first reserved bit — a post-E-7 image must refuse
    resv = list(d)
    resv[fmt.W_MASK] |= (1 << 17)
    chk(fmt.check2(resv[0], resv[1], resv[2], resv[3], resv[fmt.W_STEP],
                   cfg_d=HEAD_DIM) == fmt.ERR_DESC,
        "W2_MASK[17] is still reserved and refused")
    jobs = expected_jobs()
    chk(len(jobs) == 112 + 16 + 16,
        f"the pc program retires {len(jobs)} jobs (Wq 112 + Wk 16 + Wv 16)")
    chk(D_MODEL % D_TILE == 0 and D_MODEL // D_TILE == 14,
        "the act family is 14 whole stage rows (the C-1 framing)")
    chk(D_MODEL <= fmt.K_JOB,
        f"d_model {D_MODEL} <= K_JOB {fmt.K_JOB} — ONE k-split, accumulate=0")
    wdir = Path(man["weights_dir"])
    W = np.load(wdir / "L00_Wq.npy", mmap_mode="r")
    chk(W.shape == (D_MODEL, D_MODEL),
        f"the resident Wq really is the real {W.shape} 0.5B tensor")
    xc, _sb = golden_activation(wdir, 0)
    chk(xc.shape[0] == D_MODEL and int(np.max(np.abs(xc))) == 127,
        "the C-1 framed activation is 896 codes with per-frame amax 127")

    # ── E-6: the walked epilogue ─────────────────────────────────────────
    d_o = walk_descriptor_e6(bases)
    chk(d_o[fmt.W_MASK] == ((1 << fmt.EN_OPROJ) | (1 << fmt.EN_RES1)
                            | (1 << fmt.EN_FPROJ)),
        "E-6: the epilogue mask is {OPROJ, RES1, FPROJ}")
    d_b = walk_descriptor_e6(bases, qkv=True)
    chk((d_b[fmt.W_MASK] >> fmt.EN_QKV) & 1
        and (d_b[fmt.W_MASK] >> fmt.EN_OPROJ) & 1,
        "E-6: the one-kick QKV+OPROJ mask is pkg-legal")
    chk(fmt.W_RQ0 + N_HEADS == 36 and fmt.W_JC0 + fmt.JC_OPROJ == 54,
        "E-6: the RQ[H] and JC_OPROJ slots land at words 36 / 54")
    chk(N_JOBS_O == 112, "E-6: the o-projection is 112 n-jobs")
    chk(N_JOBS_O <= 255 and D_MODEL <= 2040,
        "E-6: the o8 serializer frame fits the 8-bit beats field")
    for nm, kw in (("nores1", dict(oproj=True, res1=False)),
                   ("ffn", dict(qkv=True, oproj=False,
                                extra_mask=1 << fmt.EN_FFN)),
                   ("qkvattn", dict(qkv=True, oproj=False,
                                    extra_mask=(1 << fmt.EN_SCORE)
                                    | (1 << fmt.EN_PV)))):
        df = walk_descriptor_e6(bases, **kw)
        chk(fmt.check2(df[0], df[1], df[2], df[3], df[fmt.W_STEP],
                       cfg_d=HEAD_DIM) == fmt.ERR_NONE,
            f"E-6 fence probe '{nm}' is pkg-LEGAL (S2_CHECK must refuse it)")
    chk(_deq_resid_fit(1.0) and _deq_resid_fit(0.0)
        and not _deq_resid_fit(float(2.0 ** -40))
        and not _deq_resid_fit(float(2.0 ** 18)),
        "E-6: the deq/residual exact-or-refused replica discriminates")
    Wo = np.load(wdir / "L00_Wo.npy", mmap_mode="r")
    chk(Wo.shape == (D_MODEL, D_MODEL),
        f"the resident Wo really is the real {Wo.shape} 0.5B tensor")
    print("=" * 62)
    print(f"WALK_FUEL_PROJ SELFTEST: {'ALL PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


# ═══════════════════════ 6. the gate ═══════════════════════════════════════

def run(args) -> int:
    idir, work = Path(args.image), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    man = json.loads((idir / "ddr_image.json").read_text())
    bases = {t["tensor"]: t["base_64B"] for t in man["tensors"]
             if t["layer"] == args.layer}
    wdir = Path(man["weights_dir"])
    blocks = {n: np.asarray(np.load(wdir / f"L{args.layer:02d}_{n}.npy"),
                            dtype=np.int64) for n, _ in QKV}

    print("[1] golden prefill + the C-1 activation framing …")
    x_codes, _sb = golden_activation(wdir, args.layer)

    jobs = expected_jobs()
    print(f"[2] emitting ONE walked descriptor — {len(jobs)} MXE jobs "
          f"(Wq/Wk/Wv), {D_MODEL // D_TILE} act rows …")
    desc = walk_descriptor(bases)
    m = build_program(x_codes, desc, work, "walk_fuel_qkv", njobs=len(jobs))
    cp_ = capture_ro(Path(m["path"]))
    sp = splice_fuel_arm(Path(m["path"]))
    print(f"    {cp_['captures']} RO capture sites (pure egress, no baked "
          f"expectations)")
    sil = silence_predicate(Path(m["path"]))
    print(f"    fuel arm spliced ({sp['inserted']} ops); host-write silence: "
          f"{sil['control_writes']} control writes, {sil['ro_advances']} RO "
          f"advances in the walk window")

    binary = REPO / "verif/f2sim" / args.obj / "f2sim"
    ddr_args = (f"+ddr_image={idir}/ddr_image.bin",
                f"+ddr_regions={idir}/ddr_image.regions.jsonl", "+fuel_audit")
    # HW mode (2026-08-05): the +ddr_* plusargs are the SIM twin's mechanism;
    # on hardware the weights are physically resident (loaded+verified by
    # f2_ddr_load BEFORE this driver runs) and the poison arm PHYSICALLY
    # reloads DDR between runs. Same programs, same grades, same golden.
    if args.executor == "hw":
        import remote_hw_exec                                 # noqa: PLC0415
        if not remote_hw_exec.attach(bridge, args):
            print("[hw] REFUSE: no remote config (APEX_F2_HOST/APEX_F2_KEY)")
            return 2
        ddr_args = ()
        print("[hw] remote shim attached (clock gate ON); weights assumed "
              "RESIDENT — f2_ddr_load full-verify must have passed")

    print("[3] CLAIM: the walked, fuel-fed run …")
    r = bridge.run_job(m["path"], executor=args.executor, binary=str(binary),
                       extra_args=ddr_args, timeout_s=args.timeout)
    res = grade(r["captures"], jobs, x_codes, blocks) if r["ok"] else \
        {"error": f"rc={r['rc']} ok={r['ok']}"}
    audit_ok = r["log"].count("-> ok") if args.executor == "sim" else "n/a(hw)"
    print(f"    rc={r['rc']} caps={len(r['captures'])} "
          f"bit-exact={res.get('all_bit_exact')} bad={res.get('bad')} "
          f"fuel_audit={audit_ok}")

    print("[4] DISCRIMINATOR A — walk-off (W2_MASK[14] cleared) …")
    off = walk_descriptor(bases, fproj=False)
    mo = build_program(x_codes, off, work, "walk_fuel_qkv_off",
                       njobs=len(jobs))
    capture_ro(Path(mo["path"]))
    splice_fuel_arm(Path(mo["path"]))
    ro = bridge.run_job(mo["path"], executor=args.executor,
                        binary=str(binary),
                        extra_args=ddr_args, timeout_s=args.timeout_off)
    off_red = not ro["ok"]
    print(f"    rc={ro['rc']} ok={ro['ok']} -> "
          f"{'RED (as measured: the legacy path parks at S2_FETCH)' if off_red else 'GREEN — UNEXPECTED'}")

    print("[5] DISCRIMINATOR B — one poisoned DDR byte …")
    # the poison targets job Wq_n0 — the FIRST block the walker's own Wq
    # fetch record streams, so a RED here is a RED on bytes the SEQUENCER
    # asked DDR for. base_64B is the tensor base because n0=0,k0=0.
    jr = {"base_64B": bases["Wq"], "x_codes": x_codes,
          "W_block": blocks["Wq"][:, :MXE_N],
          "name": "L00_Wq_n0_k0(walked)", "byte_off_in_tensor": 0}
    (work / "bad_image").mkdir(parents=True, exist_ok=True)
    mut = fp.mutate_image(idir / "ddr_image.bin",
                          work / "bad_image" / "ddr_image.bin", jr)
    rewrite_region_sha(idir / "ddr_image.regions.jsonl",
                       work / "bad_image" / "ddr_image.regions.jsonl",
                       work / "bad_image" / "ddr_image.bin")
    bad_args = (f"+ddr_image={work}/bad_image/ddr_image.bin",
                f"+ddr_regions={work}/bad_image/ddr_image.regions.jsonl",
                "+fuel_audit")
    if args.executor == "hw":
        import subprocess, os                                  # noqa: PLC0415
        host = args.host or os.environ.get("APEX_F2_HOST")
        keyf = args.key or os.environ.get("APEX_F2_KEY")
        def _ddr_swap(local_dir, tag):
            ssh = ["ssh", "-i", os.path.expanduser(keyf),
                   "-o", "ConnectTimeout=15", host]
            scp = ["scp", "-i", os.path.expanduser(keyf)]
            subprocess.run(ssh + [f"mkdir -p ~/ddrswap_{tag}"], check=True)
            subprocess.run(scp + [f"{local_dir}/ddr_image.bin",
                                  f"{local_dir}/ddr_image.regions.jsonl",
                                  f"{host}:~/ddrswap_{tag}/"], check=True)
            # regions.jsonl is this repo's manifest name on the swap dirs;
            # f2_ddr_load reads ddr_image.json — ship the matching manifest
            subprocess.run(scp + [f"{idir}/ddr_image.json",
                                  f"{host}:~/ddrswap_{tag}/"], check=True)
            rr = subprocess.run(ssh + ["cd ~/aws-fpga && source sdk_setup.sh "
                 ">/dev/null 2>&1; sudo python3 "
                 "~/apexddr/scripts/fpga/f2/f2_ddr_load.py --image "
                 f"~/ddrswap_{tag} --load --verify"],
                 capture_output=True, text=True)
            okk = "-> PASS" in rr.stdout
            print(f"    [hw] DDR swap {tag}: "
                  f"{'loaded+verified' if okk else 'FAILED'}")
            if not okk:
                raise SystemExit(f"[hw] REFUSE: DDR swap {tag} failed:\n"
                                 + rr.stdout[-500:])
        _ddr_swap(work / "bad_image", "bad")
        bad_args = ()
    rb = bridge.run_job(m["path"], executor=args.executor,
                        binary=str(binary),
                        extra_args=bad_args, timeout_s=args.timeout)
    if args.executor == "hw":
        _ddr_swap(idir, "good")
    resb = grade(rb["captures"], jobs, x_codes, blocks) if rb["ok"] else \
        {"error": "run failed"}
    moved = None
    if rb["ok"] and resb.get("per_job"):
        moved = resb["per_job"][0]["acc"][0] - res["per_job"][0]["acc"][0] \
            if res.get("per_job") else None
    red_fired = bool(rb["ok"] and not resb.get("all_bit_exact")
                     and moved == mut["delta_lane0"])
    print(f"    predicted lane-0 move {mut['delta_lane0']}, measured {moved} "
          f"-> {'RED by the predicted delta' if red_fired else 'NOT the predicted RED'}")

    verdict = bool(r["ok"] and res.get("all_bit_exact")
                   and audit_ok >= 1 and sil["silent"]
                   and off_red and red_fired)
    print("-" * 78)
    print(f"  walked FUEL-fed bit-exact vs golden : "
          f"{'PASS' if res.get('all_bit_exact') else 'FAIL'}")
    print(f"  host-write silence in the walk      : "
          f"{'PASS' if sil['silent'] else 'FAIL'}")
    print(f"  walk-off (mask bit cleared) is RED  : "
          f"{'PASS' if off_red else 'FAIL'}")
    print(f"  poisoned DDR byte is RED by delta   : "
          f"{'PASS' if red_fired else 'FAIL'}")
    print(f"  E-5 WALKED FUEL PROJECTION GATE     : "
          f"{'PASS' if verdict else 'FAIL'}")
    out = work / "walk_fuel_proj_result.json"
    out.write_text(json.dumps(
        {"verdict": verdict, "claim": res, "silence": sil,
         "walk_off_red": off_red, "poison": {"mut": mut, "moved": moved,
                                             "red_fired": red_fired},
         "fuel_audit": audit_ok, "jobs": [f"{t}_n{n}" for t, n in jobs]},
        indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if verdict else 1


# ═══════════════════════ 7. E-6 — THE WALKED EPILOGUE ══════════════════════
# E-5 fenced OUT the o8 epilogue: its QKV walks egress RAW INT32 on the RO
# lanes and the host drains them. E-6 un-fences it for OPROJ: ONE fmt=1
# descriptor, mask {OPROJ, RES1, FPROJ} (optionally + QKV), in which the
# SEQUENCER
#   * arms the o8 consumer chain ITSELF, in template order: the residual
#     add job (PC_URES1), the deq JOBC+job (PC_UOPRJ), and — new, S2_PSJ —
#     the serializer frame for the whole o8 stream;
#   * fetches Wo from DDR by its own record and runs the projection with
#     requant_en=1 on the descriptor (RQ[H], the pj_desc encoding that had
#     NEVER executed), route RT_FPRQ = rdst 1: the o8 codes post to the
#     serializer, NOT the RO lanes;
#   * whose product lands, THROUGH apex_layer_deq (x the JC composite) and
#     apex_residual (+ the host-preloaded X row), as r1 = f16(X + o8*comp)
#     IN THE RESIDUAL ROW RAM — the activation the next op consumes, never
#     leaving the tile.
# The walk window contains ZERO host operations beyond the busy poll — the
# E-5 RO drain is GONE because there is nothing to drain. Golden is the only
# arbiter: gemm_i8_ksplit -> calib_requant/requant_i32_to_i8 -> the C-6
# f16(X + o8*comp) composition, all golden's own functions on the same
# operands, graded on the post-walk LAYER_RDATA readback of the row.
#
# HONESTY FENCES (each measured, with the RTL cite):
#   * the activation is the REAL attention-output row of the layer, C-1
#     framed exactly as E-5 frames its rows (the header's activation-framing
#     note applies unchanged — per-frame scales, disclosed);
#   * DOWN's epilogue is the same pc class (pc_hasrq) but stays REFUSED:
#     it is mask-inseparable from FFN (seq_layer_walker2 PC_WDF/WDJ gate on
#     W2_EN_FFN) and the FFN template order starves asu_swiglu (fp_mask_ok
#     note); at D=64 its k = d_ffn = 4864 also exceeds the 31-row stage
#     bank (make_weight_image.py stageable_at_d) — the frozen-spec K_JOB
#     erratum, out of this driver's scope;
#   * NORM1/NORM2 stay REFUSED under FPROJ: with fuel armed their gamma
#     fetches would ISSUE and poison the fuel FIFO (no xw consumer exists
#     for a gamma — STEP_MATRIX.md p_norm2).

N_JOBS_O = D_MODEL // MXE_N          # 112 OPROJ n-jobs (Wo is 896 x 896)


def _f32_of(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _deq_resid_fit(v: float) -> bool:
    """Exact-or-refused pre-check of one deq product against the tile's own
    legality: apex_layer_deq emits it as exact fp32; apex_residual accepts a
    b operand only if it is normal, below the 2^17 inf shortcut, and on the
    2^-38 grid (apex_residual.sv b_fit). Mirrored here so a subject that
    would trip RESID_WINDOW is REFUSED at build, never run into a sticky."""
    if v == 0.0:
        return True
    b = struct.unpack("<I", struct.pack("<f", abs(v)))[0]
    if struct.unpack("<f", struct.pack("<f", v))[0] != v:
        return False                             # not fp32-exact
    e8 = (b >> 23) & 0xFF
    if e8 == 0 or e8 >= 144:
        return False
    sh = e8 - 112
    if sh >= 0:
        return True
    sig = (1 << 23) | (b & 0x7FFFFF)
    return (-sh) < 24 and (sig & ((1 << (-sh)) - 1)) == 0


def golden_oproj(wdir: Path, layer: int, *, tier: str = "kvq8",
                 group: int = 128) -> dict:
    """The REAL o-projection operand set at `layer`: the attention-output
    row (golden's own fx.attn), C-1 framed like every fuel activation, and
    the layer's residual-entry row X (fp16-grid) for the C-6 add."""
    model = fp.rt.GoldenModel(wdir)
    fxs = fp.golden_prefill(model, fp.PROMPT_IDS, fp.rt.TIER_MAP[tier], group)
    fx = fxs[layer]
    o_real = np.asarray(fx.attn, dtype=np.float64)
    o_codes, o_scales = fp.c1_frames(o_real, D_TILE)
    x_real = (model.embed_row(fp.PROMPT_IDS[-1]) if layer == 0
              else np.asarray(fxs[layer - 1].r2, dtype=np.float64))
    x16 = np.asarray(f64_to_f16_bits(np.asarray(x_real, dtype=np.float64)),
                     dtype=np.uint16)
    return dict(o_codes=o_codes, o_scales=[int(v) for v in o_scales],
                x16=x16, s_wo=float(model.layers[layer].s_wo))


def oproj_epilogue_golden(o_codes: np.ndarray, Wo: np.ndarray, *,
                          s_wo: float, s_a_bits: int, x16: np.ndarray) -> dict:
    """Golden's OWN epilogue composition on the walk's exact operands:
    per-n-job gemm_i8_ksplit, ONE calib_requant pair over the whole row
    (host-loaded, like golden _proj_epilogue), requant_i32_to_i8, the
    fp16-graded fold composite, and r1 = f16(X + o8*comp) elementwise —
    transformer._f16(X[T] + attn_proj)'s exact f64 pattern.

    The composite folds the FRAME-0 activation scale (the C-1 framing gives
    14 per-frame scales and the deq job carries ONE composite — the same
    B-FEED-WIDTH honesty note as the E-5 activation framing; golden grades
    the SAME composition on the SAME operands, so the claim is datapath
    bit-exactness, not 0.5B model semantics at DM > CFG_DM)."""
    acc = np.concatenate([
        cp.gemm_i8_ksplit(o_codes[None, :].astype(np.int64),
                          np.asarray(Wo[:, n0 * MXE_N:(n0 + 1) * MXE_N],
                                     dtype=np.int64))[0]
        for n0 in range(N_JOBS_O)])
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = at.calib_requant(amax)
    o8 = cp.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    assert int(np.max(np.abs(o8))) < 128, "epilogue clip (calib target 126)"
    s_a = float(f16_bits_to_f64(np.array([s_a_bits], dtype=np.uint16))[0])
    comp_bits = fmt.grade_f32(s_a * s_wo * float(1 << shift) / float(scale))
    comp = _f32_of(comp_bits)
    prods = o8.astype(np.float64) * comp
    bad = [i for i, v in enumerate(prods) if not _deq_resid_fit(float(v))]
    if bad:
        raise SystemExit(f"REFUSE: {len(bad)} deq product(s) violate the "
                         f"tile's exact-or-refused window (first idx "
                         f"{bad[0]}) — this subject would trip "
                         f"RESID_WINDOW/exact_error, not compute")
    r1_bits = np.asarray(
        f64_to_f16_bits(f16_bits_to_f64(x16) + prods), dtype=np.uint16)
    return dict(acc=acc, rq=(int(scale), int(shift)), o8=o8,
                comp_bits=int(comp_bits), comp=comp, r1_bits=r1_bits)


def poison_search(img: Path, dst: Path, *, base_64B: int, x_codes, acc, rq,
                  comp: float, x16, name: str) -> dict:
    """ONE flipped byte of the fetched block 0 — SEARCHED ON GOLDEN FIRST
    (the elane gamma-delta idiom): the requant epilogue can ABSORB a flip
    (measured: the first argmax-|x| flip moved acc[0] but requant mapped it
    to the same o8 count — a vacuous discriminator), so the byte is chosen
    so the predicted o8[0] AND the predicted r1[0] BOTH move."""
    data = bytearray(Path(img).read_bytes())
    x = np.asarray(x_codes, dtype=np.int64)
    o80 = int(cp.requant_i32_to_i8(np.asarray([acc[0]]), *rq)[0])
    x0v = float(f16_bits_to_f64(np.asarray(x16[:1], dtype=np.uint16))[0])
    r10 = int(f64_to_f16_bits(np.asarray([x0v + o80 * comp]))[0])
    for k in (int(i) for i in np.argsort(-np.abs(x))):
        if x[k] == 0:
            break
        off = base_64B * 64 + ((k // 8) * 8 + 0) * 8 + (k % 8)
        old = data[off]
        new = old ^ 0xFF
        old_i8 = old - 256 if old >= 128 else old
        new_i8 = new - 256 if new >= 128 else new
        delta = int(x[k]) * (new_i8 - old_i8)
        o8p0 = int(cp.requant_i32_to_i8(np.asarray([acc[0] + delta]),
                                        *rq)[0])
        if o8p0 == o80:
            continue
        r1p0 = int(f64_to_f16_bits(np.asarray([x0v + o8p0 * comp]))[0])
        if r1p0 == r10:
            continue
        data[off] = new
        Path(dst).write_bytes(bytes(data))
        return {"byte_offset": off, "k": k, "old": old_i8, "new": new_i8,
                "x_k": int(x[k]), "delta_lane0": delta, "name": name}
    raise SystemExit("REFUSE: no single-byte flip moves r1[0] — the poison "
                     "discriminator would be vacuous on this subject")


def walk_descriptor_e6(bases: dict, *, qkv: bool = False, oproj: bool = True,
                       res1: bool | None = None, fproj: bool = True,
                       rq: tuple = (1, 0), comp_bits: int = 0x3F800000,
                       extra_mask: int = 0) -> list[int]:
    """The E-6 fmt=1 image. res1 defaults to oproj (the paired fence);
    passing them unequal, or extra_mask bits, builds FENCE-PROBE images —
    pkg-LEGAL on purpose, so S2_CHECK is what refuses them."""
    res1 = oproj if res1 is None else res1
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(HEAD_DIM)
    w[fmt.W_MODEL0] = fmt.pack_model0(D_MODEL, 0)
    w[fmt.W_MODEL1] = fmt.pack_model1(N_HEADS, N_KV_HEADS, 1)
    mask = extra_mask
    if qkv:
        mask |= (1 << fmt.EN_QKV)
    if oproj:
        mask |= (1 << fmt.EN_OPROJ)
    if res1:
        mask |= (1 << fmt.EN_RES1)
    if fproj:
        mask |= (1 << fmt.EN_FPROJ)
    w[fmt.W_MASK] = fmt.pack_mask(mask)
    w[fmt.W_STEP] = fmt.pack_step(1, 0)
    for t, name in ((fmt.TENS_WQ, "Wq"), (fmt.TENS_WK, "Wk"),
                    (fmt.TENS_WV, "Wv"), (fmt.TENS_WO, "Wo")):
        w[fmt.W_TENS0 + t] = bases[name]
    w[fmt.W_RQ0 + N_HEADS] = fmt.pack_rq(*rq)      # the OPROJ epilogue slot
    w[fmt.W_JC0 + fmt.JC_OPROJ] = comp_bits        # the deq composite slot
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=HEAD_DIM) == fmt.ERR_NONE, "illegal walk image"
    return w


def _stage_act_rows(s, codes: np.ndarray, row0: int):
    """Host-stage one C-1 code family into act bank 1 at rows row0.. —
    the E-5 staging verbs verbatim, with a row offset."""
    rows = len(codes) // D_TILE
    for r in range(rows):
        rowv = codes[r * D_TILE:(r + 1) * D_TILE].astype(np.float64)
        pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(rowv)]
        g3.inject_jobs(s, pairs, 1)
        s.ess(F16_ONE, 1)                 # identity scale — a COUNTED check
        s.aj(0, ACT_BANK, 0, 1, s.BPR, row0 + r)


def build_e6_program(x16, act_families, desc, out_dir: Path, name: str, *,
                     njobs_ro: int = 0, meta: dict | None = None) -> dict:
    """Host prologue + X preload + act staging + fuel arm + ONE walk +
    (RO drain iff QKV rides along) + post-walk r1 readback.

    act_families: [(codes, row0), ...] — row0 mirrors the walker's own
    deterministic fp_oproj_base rule (QKV at 0, OPROJ above it)."""
    with tg.at_d(D_TILE):
        s = LayerScript(D_TILE)
        s.emit("// E-6 THE WALKED EPILOGUE: one fmt=1 descriptor; the "
               "sequencer fetches Wo from DDR, runs the requant epilogue "
               "on-tile, and its o8 lands in the residual row THROUGH the "
               "tile's own ser->deq->residual chain.")
        g3.phase_a(s, tiers_used=(0,))
        g3.loader_phase(s)
        s.emit("// host loads X — the residual-entry row the epilogue adds "
               "onto (fp16 bits, LAYER window bank 2)")
        s.lptr(BANK_RESID, 0)
        s.ldata(x16)
        s.emit("// stage the C-1 activation families (host-loaded PRE-walk; "
               "the walker replays them per n-job)")
        s.route(rdst=1, asrc=1)
        for codes, row0 in act_families:
            _stage_act_rows(s, codes, row0)
        s.emit(f"// {FUEL_MARK}")
        s.emit("// load the fmt=1 descriptor (64 words) and KICK")
        s.csrw(W_DPTR, 0)
        for v in desc:
            s.csrw(W_DDATA, v)
        s.emit("// CYCMARK e6go")
        s.csrw(W_CTRL, 0x3)
        if njobs_ro:
            s.emit(f"// RO drain, {njobs_ro} raw QKV jobs x 9 — the E-5 "
                   f"disclosed data plane; the OPROJ epilogue that follows "
                   f"needs NO drain (its o8 never reaches the RO FIFO)")
            for _ in range(njobs_ro):
                s.ero([0] * MXE_N, 1)
        s.csrp(W_STATUS, W_ST_BUSY, 0x0)
        s.emit("// CYCMARK e6done")
        s.csrr(W_STATUS, W_ST_ESTK | W_ST_ECODE, 0x0)
        s.csrw(W_CTRL, 0x0)
        s.emit("// the epilogue consumers drained CLEAN: deq idle, residual "
               "idle, no LAYER error sticky — the host-mode readback idiom")
        s.lpoll(LST_LDQ | LST_RES, 0x0)
        s.lstat(LST_ERR | LST_LDQ | LST_RES, 0x0)
        s.emit("// r1 readback: the row the TILE built (produce-mode caps), "
               "golden the only arbiter")
        s.lrptr(0)
        s.erd(D_MODEL)
        g3.final_phase(s, 8)
        x = _translate(s, note=f"E-6 walked epilogue: {name}")
        man = _emit(x, out_dir, name,
                    dict(stage="walk_fuel_e6", d=D_TILE, **(meta or {})))
        tg.retarget_info_tier(man["path"], tg.IMG_05B)
        # arm fuel MODE at the marker (src=1, ar_owner=1 — no base/beats/GO):
        # without it the S2_CHECK wf_ready clause refuses the walk, LOUDLY —
        # measured on the first emission of this very program.
        splice_fuel_arm(Path(man["path"]))
        return man


def build_e6_refusal(bases: dict, out_dir: Path, name: str, *, why: str,
                     **desc_kw) -> dict:
    """S2_CHECK fence probes ON THE TWIN (the e4_fence idiom): a pkg-LEGAL
    image whose mask combination the E-6 fence refuses — kick, then CHECK
    the refusal receipts (busy down, ESTK, code DESC, nothing consumed).
    Fuel IS armed first, so the wf_ready clause cannot mask the verdict."""
    desc = walk_descriptor_e6(bases, **desc_kw)
    with tg.at_d(D_TILE):
        s = LayerScript(D_TILE)
        s.emit(f"// E-6 fence {name}: {why} -> REFUSED at S2_CHECK "
               f"(WALK_ERR_DESC), state unchanged.")
        g3.phase_a(s, tiers_used=(0,))
        g3.loader_phase(s)
        s.emit(f"// {FUEL_MARK}")
        s.csrw(W_DPTR, 0)
        for v in desc:
            s.csrw(W_DDATA, v)
        s.csrw(W_CTRL, 0x3)
        s.csrp(W_STATUS, W_ST_BUSY, 0x0)
        s.emit("// the refusal receipt: sticky + code 2 (DESC), busy down. "
               "NO final_phase: a refused walk consumed NOTHING, so the "
               "activity counters final_phase asserts nonzero stay 0 — the "
               "e4_fence idiom (measured RED first with final_phase in)")
        s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE,
               W_ST_ESTK | (2 << 9))
        s.csrw(W_CTRL, 0x0)
        s.emit("DONE")
        x = _translate(s, note=f"E-6 fence {name}")
        man = _emit(x, out_dir, name,
                    dict(stage=name, kind=f"REFUSAL FENCE: {why}", d=D_TILE))
        tg.retarget_info_tier(man["path"], tg.IMG_05B)
        splice_fuel_arm(Path(man["path"]))
        return man


def silence_e6(path: Path, *, allow_ro: bool) -> dict:
    """The E-6 window predicate. allow_ro=False asserts the STRONGER claim:
    between WALK_GO and the busy poll the program contains NO write AT ALL
    — the E-5 RO drain is gone because the epilogue's o8 never reaches the
    RO FIFO. Checked on the emitted artefact, the E-4b/E-5 idiom."""
    sil = silence_predicate(path)
    sil["strict"] = sil["silent"] and (allow_ro or sil["ro_advances"] == 0)
    return sil


def _cyc_window(log: str) -> int | None:
    """The measured walk-window cycles from the CYCMARK sim hook."""
    go = done = None
    for ln in (log or "").splitlines():
        if not ln.startswith("CYCMARK"):
            continue
        if "e6go" in ln:
            go = int(ln.rsplit("cyc=", 1)[1])
        elif "e6done" in ln:
            done = int(ln.rsplit("cyc=", 1)[1])
    return (done - go) if (go is not None and done is not None) else None


def rune6(args) -> int:
    idir, work = Path(args.image), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    man = json.loads((idir / "ddr_image.json").read_text())
    bases = {t["tensor"]: t["base_64B"] for t in man["tensors"]
             if t["layer"] == args.layer}
    wdir = Path(man["weights_dir"])
    Wo = np.asarray(np.load(wdir / f"L{args.layer:02d}_Wo.npy"),
                    dtype=np.int64)

    print("[1] golden prefill: the REAL attention-output row + X + the "
          "epilogue composition …")
    go = golden_oproj(wdir, args.layer)
    ep = oproj_epilogue_golden(go["o_codes"], Wo, s_wo=go["s_wo"],
                               s_a_bits=go["o_scales"][0], x16=go["x16"])
    print(f"    rq=({ep['rq'][0]},{ep['rq'][1]}) comp={ep['comp']:.3e} "
          f"o8 amax={int(np.max(np.abs(ep['o8'])))}")

    binary = REPO / "verif/f2sim" / args.obj / "f2sim"
    ddr_args = (f"+ddr_image={idir}/ddr_image.bin",
                f"+ddr_regions={idir}/ddr_image.regions.jsonl", "+fuel_audit")
    # the CLAIM windows are HOST-SILENT: the whole fuel-fed projection sits
    # under ONE done-poll, so the sim's per-poll stall budget must cover it
    # (sim_main +poll_limit note — measured: ~21.7k cyc/job at tile_div=5,
    # so 112 epilogue jobs alone need ~2.5M > the 2M default). Walk-off and
    # the fences keep the DEFAULT budget: their RED/refusal is the stall.
    ddr_claim = ddr_args + ("+poll_limit=16000000",)

    results, all_ok = {}, True

    def _run_stage(mn, *, extra=ddr_claim, timeout=None):
        return bridge.run_job(mn["path"], executor="sim", binary=str(binary),
                              extra_args=extra,
                              timeout_s=timeout or args.timeout)

    print("[2] CLAIM A — the walked epilogue alone: mask {OPROJ, RES1, "
          "FPROJ} …")
    d_o = walk_descriptor_e6(bases, rq=ep["rq"], comp_bits=ep["comp_bits"])
    m_o = build_e6_program(go["x16"], [(go["o_codes"], 0)], d_o, work,
                           "walk_fuel_oproj",
                           meta=dict(mask=f"{d_o[fmt.W_MASK]:#06x}"))
    sil_o = silence_e6(Path(m_o["path"]), allow_ro=False)
    print(f"    window writes={sil_o['writes']} (STRICT silence: "
          f"{sil_o['strict']})")
    r_o = _run_stage(m_o)
    g_o = grade_resid(r_o["captures"], ep["r1_bits"]) if r_o["ok"] else \
        {"equal": False, "error": f"rc={r_o['rc']}"}
    cyc_o = _cyc_window(r_o.get("log"))
    print(f"    rc={r_o['rc']} caps={len(r_o['captures'])} "
          f"r1_bit_exact={g_o.get('equal')} n_diff={g_o.get('n_diff')} "
          f"walk_cycles={cyc_o}")
    results["oproj"] = dict(ok=bool(r_o["ok"]), grade=g_o, silence=sil_o,
                            cycles=cyc_o)
    all_ok &= bool(r_o["ok"] and g_o.get("equal") and sil_o["strict"])

    print("[3] CLAIM B — TWO fueled projections, ONE kick: mask {QKV, "
          "OPROJ, RES1, FPROJ} …")
    x_codes, _sb = golden_activation(wdir, args.layer)
    blocks = {n: np.asarray(np.load(wdir / f"L{args.layer:02d}_{n}.npy"),
                            dtype=np.int64) for n, _ in QKV}
    jobs = expected_jobs()
    d_b = walk_descriptor_e6(bases, qkv=True, rq=ep["rq"],
                             comp_bits=ep["comp_bits"])
    m_b = build_e6_program(go["x16"],
                           [(x_codes, 0), (go["o_codes"], D_MODEL // D_TILE)],
                           d_b, work, "walk_fuel_qkv_oproj",
                           njobs_ro=len(jobs),
                           meta=dict(mask=f"{d_b[fmt.W_MASK]:#06x}"))
    capture_ro(Path(m_b["path"]))
    sil_b = silence_e6(Path(m_b["path"]), allow_ro=True)
    r_b = _run_stage(m_b)
    ro_caps = [c for c in r_b["captures"]
               if str(c.get("tag", "")).startswith("ro_")]
    rd_caps = [c for c in r_b["captures"]
               if not str(c.get("tag", "")).startswith("ro_")]
    g_qkv = grade(ro_caps, jobs, x_codes, blocks) if r_b["ok"] else \
        {"all_bit_exact": False, "error": f"rc={r_b['rc']}"}
    g_r1 = grade_resid(rd_caps, ep["r1_bits"]) if r_b["ok"] else \
        {"equal": False}
    cyc_b = _cyc_window(r_b.get("log"))
    print(f"    rc={r_b['rc']} caps={len(r_b['captures'])} "
          f"qkv_bit_exact={g_qkv.get('all_bit_exact')} "
          f"(bad={g_qkv.get('bad')}) r1_bit_exact={g_r1.get('equal')} "
          f"walk_cycles={cyc_b}")
    results["qkv_oproj"] = dict(ok=bool(r_b["ok"]), qkv=
                                {k: g_qkv.get(k) for k in
                                 ("jobs", "bad", "all_bit_exact", "error")
                                 if k in g_qkv},
                                r1=g_r1, silence=sil_b, cycles=cyc_b)
    all_ok &= bool(r_b["ok"] and g_qkv.get("all_bit_exact")
                   and g_r1.get("equal") and sil_b["silent"])

    print("[4] DISCRIMINATOR A — walk-off (FPROJ cleared; the measured "
          "p_oproj park class) …")
    d_off = walk_descriptor_e6(bases, fproj=False, rq=ep["rq"],
                               comp_bits=ep["comp_bits"])
    m_off = build_e6_program(go["x16"], [(go["o_codes"], 0)], d_off, work,
                             "walk_fuel_oproj_off")
    r_off = _run_stage(m_off, extra=ddr_args, timeout=args.timeout_off)
    off_red = not r_off["ok"]
    print(f"    rc={r_off['rc']} ok={r_off['ok']} -> "
          f"{'RED (no walked epilogue without the bit)' if off_red else 'GREEN — UNEXPECTED'}")
    results["walk_off"] = dict(ok=bool(r_off["ok"]), red=off_red)
    all_ok &= off_red

    print("[5] DISCRIMINATOR B — one poisoned Wo byte -> r1 moves by the "
          "PREDICTED f16 delta …")
    (work / "bad_image_e6").mkdir(parents=True, exist_ok=True)
    mut = poison_search(idir / "ddr_image.bin",
                        work / "bad_image_e6" / "ddr_image.bin",
                        base_64B=bases["Wo"], x_codes=go["o_codes"],
                        acc=ep["acc"], rq=ep["rq"], comp=ep["comp"],
                        x16=go["x16"],
                        name=f"L{args.layer:02d}_Wo_n0(walked-epilogue)")
    rewrite_region_sha(idir / "ddr_image.regions.jsonl",
                       work / "bad_image_e6" / "ddr_image.regions.jsonl",
                       work / "bad_image_e6" / "ddr_image.bin")
    # the PREDICTION, composed with the SAME host-loaded rq/comp: only
    # acc[0] (job n0=0, lane 0) moves, by x_k * (new - old)
    acc_p = ep["acc"].copy()
    acc_p[0] += mut["delta_lane0"]
    o8_p = cp.requant_i32_to_i8(acc_p, *ep["rq"]).astype(np.int64)
    r1_p = np.asarray(f64_to_f16_bits(
        f16_bits_to_f64(go["x16"]) + o8_p.astype(np.float64) * ep["comp"]),
        dtype=np.uint16)
    bad_args = (f"+ddr_image={work}/bad_image_e6/ddr_image.bin",
                f"+ddr_regions={work}/bad_image_e6/ddr_image.regions.jsonl",
                "+fuel_audit", "+poll_limit=16000000")
    r_p = _run_stage(m_o, extra=bad_args)
    gp_clean = grade_resid(r_p["captures"], ep["r1_bits"]) if r_p["ok"] \
        else {"equal": True}
    gp_pred = grade_resid(r_p["captures"], r1_p) if r_p["ok"] else \
        {"equal": False}
    red_fired = bool(r_p["ok"] and not gp_clean["equal"]
                     and gp_pred["equal"]
                     and not np.array_equal(r1_p, ep["r1_bits"]))
    print(f"    o8[0] {int(ep['o8'][0])} -> {int(o8_p[0])} predicted; "
          f"measured==predicted: {gp_pred.get('equal')} -> "
          f"{'RED by the predicted f16 delta' if red_fired else 'NOT the predicted RED'}")
    results["poison"] = dict(mut={k: mut[k] for k in
                                  ("byte_offset", "k", "old", "new", "x_k",
                                   "delta_lane0")},
                             red=red_fired)
    all_ok &= red_fired

    print("[6] the E-6 fence refusals (pkg-legal images; S2_CHECK must be "
          "what refuses) …")
    fences = [
        ("e6_fence_nores1", dict(oproj=True, res1=False),
         "OPROJ without RES1 — the o8 stream would starve on an unarmed "
         "residual"),
        ("e6_fence_ffn", dict(qkv=True, oproj=False,
                              extra_mask=(1 << fmt.EN_FFN)),
         "FFN under FPROJ — gamma-class fetch poisoning + the swiglu "
         "gate/up starvation (and DOWN with it)"),
        # E-7 CLOSED the old qkvattn fence: {QKV, SCORE, PV, FPROJ} now
        # COMPOSES (the families stack in template order — fp_base), so the
        # refusal probe is the CAPACITY bound instead: adding OPROJ at the
        # 0.5B geometry stacks 14 (q rows) + 14 (QKV) + 14 (OPROJ) = 42
        # rows against the 31-row bank — refused, with a measured reason.
        ("e6_fence_qkvattn_cap", dict(qkv=True, oproj=True,
                                      extra_mask=(1 << fmt.EN_SCORE)
                                      | (1 << fmt.EN_PV)),
         "QKV + attention + OPROJ under FPROJ at H=14 — the stacked act "
         "families (42 rows) exceed the 31-row stage bank"),
    ]
    for fname, kw, why in fences:
        mf = build_e6_refusal(bases, work, fname, why=why,
                              rq=ep["rq"], comp_bits=ep["comp_bits"], **kw)
        rf = _run_stage(mf, extra=ddr_args, timeout=args.timeout_off)
        print(f"    {fname:<18} rc={rf['rc']} REFUSED={rf['ok']}")
        results[fname] = dict(refused=bool(rf["ok"]))
        all_ok &= bool(rf["ok"])

    print("-" * 78)
    print(f"  walked EPILOGUE r1 bit-exact vs golden     : "
          f"{'PASS' if results['oproj']['grade'].get('equal') else 'FAIL'}")
    print(f"  STRICT walk-window silence (no drain left) : "
          f"{'PASS' if results['oproj']['silence']['strict'] else 'FAIL'}")
    print(f"  QKV+OPROJ one-kick both bit-exact          : "
          f"{'PASS' if (results['qkv_oproj']['qkv'].get('all_bit_exact') and results['qkv_oproj']['r1'].get('equal')) else 'FAIL'}")
    print(f"  walk-off RED / poison RED / 3 fences       : "
          f"{'PASS' if (results['walk_off']['red'] and results['poison']['red'] and all(results[f]['refused'] for f, _, _ in fences)) else 'FAIL'}")
    print(f"  E-6 WALKED EPILOGUE GATE                   : "
          f"{'PASS' if all_ok else 'FAIL'}")
    out = work / "walk_fuel_e6_result.json"
    out.write_text(json.dumps(dict(verdict=bool(all_ok), stages=results),
                              indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if all_ok else 1


# ═══════════════ 8. E-7b — THE SEPARATED, WALKED DOWN EPILOGUE ═════════════
# DOWN was "mask-inseparable from FFN" (its pcs gated on W2_EN_FFN) and its
# 0.5B k = d_ffn = 4864 exceeds one k-split. E-7b's W2_EN_DOWN bit enables
# exactly the DOWN pcs, paired with RES2, and this gate proves the DOWN
# requant epilogue — the E-6 pc_hasrq class — WALKED at a k-LEGAL geometry:
# d_ffn = 1984 (= walk2_k_job(64), the largest one-split contraction this
# row width can stage: 31 whole stage rows), n = the REAL d_model = 896.
#
# HONESTY FENCES:
#   * the weight bytes are the FIRST 1984x896 of the REAL resident L00_Wd
#     region, decoded exactly as the walker's own k=1984 decomposition
#     reads them (job n0 = the image's n0-th 1984x8 block — image byte
#     order, not model semantics; the claim is datapath bit-exactness on
#     bytes the SEQUENCER fetched from card DRAM);
#   * the activation family (the swiglu-product codes a full layer would
#     produce) is SYNTHETIC and HOST-staged — the disclosed E-6 o8->act
#     seam class; the in-tile producer is the FFN interleave (fence 2,
#     OPEN, real work);
#   * the 0.5B d_ffn = 4864 walk stays REFUSED and the refusal is PROVEN
#     ON THE TWIN here (d7_fence_05b): 76 stage rows cannot be resident in
#     the 31-row bank and S2_PAE has no re-staging source — fence 3's
#     honest remnant, measured not stylistic.

D_FFN_D7 = 1984                       # walk2_k_job(64): 31 whole stage rows
N_JOBS_D7 = D_MODEL // MXE_N          # 112 n-jobs (n = d_model = 896)


def wd_blocks_as_fetched(idir: Path, layer: int, k: int) -> list[np.ndarray]:
    """Decode the first k*d_model bytes of the resident Wd region exactly as
    the walker's one-k-split decomposition consumes them: job n0 = the n0-th
    k x 8 block, byte (p*8+c)*8+r == W[8p+r][c] (make_weight_image order)."""
    man = json.loads((idir / "ddr_image.json").read_text())
    wd = next(t for t in man["tensors"]
              if t["layer"] == layer and t["tensor"] == "Wd")
    img = (idir / "ddr_image.bin").read_bytes()
    b0 = wd["base_64B"] * 64
    blocks = []
    for j in range(N_JOBS_D7):
        blk = img[b0 + k * 8 * j: b0 + k * 8 * (j + 1)]
        W = np.frombuffer(bytes(blk), dtype=np.int8).astype(np.int64)
        # byte (p*8+c)*8+r -> W[8p+r][c]
        W = W.reshape(k // 8, 8, 8).transpose(0, 2, 1).reshape(k, 8)
        blocks.append(W)
    return blocks


def walk_descriptor_d7(bases: dict, *, d_ffn: int = D_FFN_D7,
                       down: bool = True, res2: bool = True,
                       fproj: bool = True, rq: tuple = (1, 0),
                       comp_bits: int = 0x3F800000,
                       extra_mask: int = 0) -> list[int]:
    """The E-7b image: mask {DOWN, RES2, FPROJ} (perturbable into the fence
    probes), the REAL Wd base, RQ[H+1] and JC_DOWN."""
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(HEAD_DIM)
    w[fmt.W_MODEL0] = fmt.pack_model0(D_MODEL, d_ffn)
    w[fmt.W_MODEL1] = fmt.pack_model1(N_HEADS, N_KV_HEADS, 1)
    mask = extra_mask
    if down:
        mask |= (1 << fmt.EN_DOWN)
    if res2:
        mask |= (1 << fmt.EN_RES2)
    if fproj:
        mask |= (1 << fmt.EN_FPROJ)
    w[fmt.W_MASK] = fmt.pack_mask(mask)
    w[fmt.W_STEP] = fmt.pack_step(1, 0)
    w[fmt.W_TENS0 + fmt.TENS_WD] = bases["Wd"]
    w[fmt.W_RQ0 + N_HEADS + 1] = fmt.pack_rq(*rq)   # the DOWN epilogue slot
    w[fmt.W_JC0 + fmt.JC_DOWN] = comp_bits
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=HEAD_DIM) == fmt.ERR_NONE, "illegal walk image"
    return w


def down_epilogue_golden(h_codes: np.ndarray, blocks: list, *,
                         r1_16: np.ndarray) -> dict:
    """Golden's own DOWN epilogue on the walk's exact operands: per-n-job
    gemm over the AS-FETCHED blocks, one calib_requant pair, the fp16-grade
    composite (identity act scale — the staging's counted `ess` identity),
    r2 = f16(r1 + down8*comp) elementwise."""
    acc = np.concatenate([
        cp.gemm_i8_ksplit(h_codes[None, :].astype(np.int64), b)[0]
        for b in blocks])
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = at.calib_requant(amax)
    o8 = cp.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    comp_bits = fmt.grade_f32(float(1 << shift) / float(scale) * (2.0 ** -7))
    comp = _f32_of(comp_bits)
    prods = o8.astype(np.float64) * comp
    bad = [i for i, v in enumerate(prods) if not _deq_resid_fit(float(v))]
    if bad:
        raise SystemExit(f"REFUSE: d7 subject trips the residual window at "
                         f"idx {bad[0]}")
    r2_bits = np.asarray(
        f64_to_f16_bits(f16_bits_to_f64(r1_16) + prods), dtype=np.uint16)
    return dict(acc=acc, rq=(int(scale), int(shift)), o8=o8,
                comp_bits=int(comp_bits), comp=comp, r2_bits=r2_bits)


def rund7(args) -> int:
    idir, work = Path(args.image), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    man = json.loads((idir / "ddr_image.json").read_text())
    bases = {t["tensor"]: t["base_64B"] for t in man["tensors"]
             if t["layer"] == args.layer}
    binary = REPO / "verif/f2sim" / args.obj / "f2sim"
    ddr_args = (f"+ddr_image={idir}/ddr_image.bin",
                f"+ddr_regions={idir}/ddr_image.regions.jsonl", "+fuel_audit")
    ddr_claim = ddr_args + ("+poll_limit=16000000",)

    print("[1] the operands: as-fetched Wd blocks (REAL resident bytes), a "
          "synthetic staged act family, the r1 row …")
    rng = np.random.default_rng(20260805)
    h_codes = rng.integers(-127, 128, D_FFN_D7).astype(np.int64)
    # per-64-frame amax = 127 BY CONSTRUCTION, so the staging's MODE_QUANT
    # is the identity and the counted `ess` really is F16_ONE — the E-5
    # c1_frames property, forced here because the codes are synthetic (the
    # E-6 chain's 0x3BF0-vs-0x3C00 lesson, re-learnt by execution on the
    # first emission of this gate).
    for _i in range(D_FFN_D7 // D_TILE):
        h_codes[_i * D_TILE] = 127
    blocks = wd_blocks_as_fetched(idir, args.layer, D_FFN_D7)
    r1_16 = np.asarray(f64_to_f16_bits(rng.normal(0.0, 1.0, D_MODEL)),
                       dtype=np.uint16)
    ep = down_epilogue_golden(h_codes, blocks, r1_16=r1_16)
    print(f"    rq=({ep['rq'][0]},{ep['rq'][1]}) comp={ep['comp']:.3e} "
          f"down8 amax={int(np.max(np.abs(ep['o8'])))}")

    print("[2] CLAIM — the walked DOWN epilogue: mask {DOWN, RES2, FPROJ}, "
          f"k = d_ffn = {D_FFN_D7} (one k-split, 31 stage rows), "
          f"{N_JOBS_D7} n-jobs …")
    d_w = walk_descriptor_d7(bases, rq=ep["rq"], comp_bits=ep["comp_bits"])
    m_w = build_e6_program(r1_16, [(h_codes, 0)], d_w, work, "walk_fuel_down",
                           meta=dict(mask=f"{d_w[fmt.W_MASK]:#07x}",
                                     d_ffn=D_FFN_D7))
    sil = silence_e6(Path(m_w["path"]), allow_ro=False)
    r_w = bridge.run_job(m_w["path"], executor="sim", binary=str(binary),
                         extra_args=ddr_claim, timeout_s=args.timeout)
    g_w = grade_resid(r_w["captures"], ep["r2_bits"]) if r_w["ok"] else \
        {"equal": False, "error": f"rc={r_w['rc']}"}
    cyc = _cyc_window(r_w.get("log"))
    print(f"    rc={r_w['rc']} caps={len(r_w['captures'])} "
          f"r2_bit_exact={g_w.get('equal')} n_diff={g_w.get('n_diff')} "
          f"strict_silence={sil['strict']} walk_cycles={cyc}")
    green = bool(r_w["ok"] and g_w.get("equal") and sil["strict"])

    print("[3] DISCRIMINATOR — one poisoned Wd byte -> r2 moves by the "
          "predicted f16 delta …")
    (work / "bad_image_d7").mkdir(parents=True, exist_ok=True)
    mut = poison_search(idir / "ddr_image.bin",
                        work / "bad_image_d7" / "ddr_image.bin",
                        base_64B=bases["Wd"], x_codes=h_codes,
                        acc=ep["acc"], rq=ep["rq"], comp=ep["comp"],
                        x16=r1_16,
                        name=f"L{args.layer:02d}_Wd_n0(walked-down)")
    rewrite_region_sha(idir / "ddr_image.regions.jsonl",
                       work / "bad_image_d7" / "ddr_image.regions.jsonl",
                       work / "bad_image_d7" / "ddr_image.bin")
    acc_p = ep["acc"].copy()
    acc_p[0] += mut["delta_lane0"]
    o8_p = cp.requant_i32_to_i8(acc_p, *ep["rq"]).astype(np.int64)
    r2_p = np.asarray(f64_to_f16_bits(
        f16_bits_to_f64(r1_16) + o8_p.astype(np.float64) * ep["comp"]),
        dtype=np.uint16)
    bad_args = (f"+ddr_image={work}/bad_image_d7/ddr_image.bin",
                f"+ddr_regions={work}/bad_image_d7/ddr_image.regions.jsonl",
                "+fuel_audit", "+poll_limit=16000000")
    r_p = bridge.run_job(m_w["path"], executor="sim", binary=str(binary),
                         extra_args=bad_args, timeout_s=args.timeout)
    gp_pred = grade_resid(r_p["captures"], r2_p) if r_p["ok"] else \
        {"equal": False}
    gp_clean = grade_resid(r_p["captures"], ep["r2_bits"]) if r_p["ok"] \
        else {"equal": True}
    poison_red = bool(r_p["ok"] and gp_pred["equal"]
                      and not gp_clean["equal"]
                      and not np.array_equal(r2_p, ep["r2_bits"]))
    print(f"    predicted r2[0] delta applied; measured==predicted: "
          f"{gp_pred.get('equal')} -> "
          f"{'RED by the predicted delta' if poison_red else 'NOT RED'}")

    print("[4] the E-7b fence refusals (pkg-legal; S2_CHECK refuses) …")
    fences = [
        ("d7_fence_05b", dict(d_ffn=4864),
         "the REAL 0.5B d_ffn=4864 > walk2_k_job(64)=1984 — 76 stage rows "
         "vs the 31-row bank, no in-tile re-staging source (fence 3's "
         "honest remnant)"),
        ("d7_fence_nores2", dict(res2=False),
         "DOWN without RES2 — the starvation pair rule"),
        ("d7_fence_ffn", dict(extra_mask=(1 << fmt.EN_FFN)),
         "DOWN with FFN — a double enable of the same pcs"),
        ("d7_fence_nofproj", dict(fproj=False),
         "DOWN without FPROJ — the Wd fetch park wedge"),
    ]
    ref_ok = True
    for fname, kw, why in fences:
        desc = walk_descriptor_d7(bases, rq=ep["rq"],
                                  comp_bits=ep["comp_bits"], **kw)
        mf = _build_d7_refusal(desc, work, fname, why=why)
        rf = bridge.run_job(mf["path"], executor="sim", binary=str(binary),
                            extra_args=ddr_args, timeout_s=args.timeout_off)
        print(f"    {fname:<18} rc={rf['rc']} REFUSED={rf['ok']}")
        ref_ok &= bool(rf["ok"])

    verdict = bool(green and poison_red and ref_ok)
    print("-" * 78)
    print(f"  walked DOWN epilogue r2 bit-exact (k=1984)  : "
          f"{'PASS' if g_w.get('equal') else 'FAIL'}")
    print(f"  STRICT walk-window silence                  : "
          f"{'PASS' if sil['strict'] else 'FAIL'}")
    print(f"  Wd poison RED / 4 fences (incl. 0.5B d_ffn) : "
          f"{'PASS' if (poison_red and ref_ok) else 'FAIL'}")
    print(f"  E-7b WALKED DOWN GATE                       : "
          f"{'PASS' if verdict else 'FAIL'}")
    out = work / "walk_fuel_d7_result.json"
    out.write_text(json.dumps(
        dict(verdict=verdict, grade=g_w, silence=sil, cycles=cyc,
             poison=dict(red=poison_red,
                         mut={k: mut[k] for k in ("byte_offset", "k", "old",
                                                  "new", "x_k",
                                                  "delta_lane0")}),
             fences_refused=ref_ok), indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if verdict else 1


def _build_d7_refusal(desc: list[int], out_dir: Path, name: str, *,
                      why: str) -> dict:
    with tg.at_d(D_TILE):
        s = LayerScript(D_TILE)
        s.emit(f"// E-7b fence {name}: {why} -> REFUSED at S2_CHECK "
               f"(WALK_ERR_DESC), state unchanged.")
        g3.phase_a(s, tiers_used=(0,))
        g3.loader_phase(s)
        s.emit(f"// {FUEL_MARK}")
        s.csrw(W_DPTR, 0)
        for v in desc:
            s.csrw(W_DDATA, v)
        s.csrw(W_CTRL, 0x3)
        s.csrp(W_STATUS, W_ST_BUSY, 0x0)
        s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE,
               W_ST_ESTK | (2 << 9))
        s.csrw(W_CTRL, 0x0)
        s.emit("DONE")
        x = _translate(s, note=f"E-7b fence {name}")
        man = _emit(x, out_dir, name,
                    dict(stage=name, kind=f"REFUSAL FENCE: {why}", d=D_TILE))
        tg.retarget_info_tier(man["path"], tg.IMG_05B)
        splice_fuel_arm(Path(man["path"]))
        return man


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--executor", choices=("sim", "hw"), default="sim",
                   help="hw: weights must ALREADY be resident (f2_ddr_load "
                        "--load --full-verify PASS on the card); the poison "
                        "arm physically re-loads DDR via ssh")
    r.add_argument("--host", default=None)
    r.add_argument("--key", default=None)
    r.add_argument("--image", default=str(DEFAULT_IMAGE))
    r.add_argument("--work", default=str(DEFAULT_WORK))
    r.add_argument("--obj", default=DEFAULT_OBJ)
    r.add_argument("--layer", type=int, default=0)
    r.add_argument("--timeout", type=int, default=5400)
    r.add_argument("--timeout-off", type=int, default=300)
    # E-6: the walked-epilogue gate (sim-only in this driver; the hw shape
    # rides the E-5 machinery once the sim gate is green and an AFI exists)
    e6 = sub.add_parser("rune6")
    e6.add_argument("--image", default=str(DEFAULT_IMAGE))
    e6.add_argument("--work", default=str(DEFAULT_WORK))
    e6.add_argument("--obj", default="obj_e6_b64_ddr1")
    e6.add_argument("--layer", type=int, default=0)
    e6.add_argument("--timeout", type=int, default=7200)
    e6.add_argument("--timeout-off", type=int, default=600)
    # E-7b: the walked DOWN epilogue at the k-legal geometry (sim gate)
    d7 = sub.add_parser("rund7")
    d7.add_argument("--image", default=str(DEFAULT_IMAGE))
    d7.add_argument("--work", default=str(REPO / "build/walk_fuel_d7"))
    d7.add_argument("--obj", default="obj_e7_b64_ddr1")
    d7.add_argument("--layer", type=int, default=0)
    d7.add_argument("--timeout", type=int, default=10800)
    d7.add_argument("--timeout-off", type=int, default=600)
    a = p.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "rund7":
        return rund7(a)
    return rune6(a) if a.cmd == "rune6" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
