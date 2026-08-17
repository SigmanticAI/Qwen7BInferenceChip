#!/usr/bin/env python3
# tile_geom.py — run the FROZEN layer/GEMM emitters at a tile geometry other
# than D=128, and say out loud where that is NOT enough.
#
# ══ WHY THIS FILE EXISTS ═══════════════════════════════════════════════════
# gen_layer_ops.py and gemm_job.py are the verified emitters for the six op
# families, and both are owned/frozen. Their CHOREOGRAPHY is geometry-neutral
# — every script is built as `LayerScript(D_TILE)` / `g3.Script(D_TILE)` and
# every token frames off `self.D` — but their module-level `D_TILE = 128`
# leaks into the array slicing, the injection-row bounds and the descriptor
# k fields. So there are exactly two honest moves for a D=64 image:
#
#   IMPORT WHAT GENERALIZES — `at_d(64)` rebinds those module globals for the
#     duration of an emission, the same "rebind a module global, never edit
#     the arbiter" technique layer_offload.py uses on golden. Rebinding is
#     VERIFIED, not assumed: `at_d` asserts the derived constants agree with
#     D on entry and restores them on every exit path, and every emitted
#     program is audited (`audit_geometry`) against its OWN build-identity
#     block: every emitter builds its script as `Script(D_TILE)`, so the
#     INFO_D expectation phase A bakes IS the width the program was emitted
#     at. A rebind that silently failed would emit a 128-framed program into
#     a 64-wide tile, so it is checked at compile time, not trusted.
#
#   RE-DERIVE WHAT DOES NOT — `build_norm2_chunked_d` is a real re-derivation,
#     not a rebind, because the R4 chunked-norm arm has TWO different
#     "number of chunks" and the frozen emitter conflates them (correctly, at
#     D=128, where they coincide):
#       * the STREAMED chunk count = dm / D          (14 at dm=896, D=64) —
#         set by the C-1 feeder's frame, seam_feeder_quant #(.D(CFG_DM)).
#       * the μ-table index  ext_k = dm / 128        (7  at dm=896) —
#         FROZEN at 128 by the generated table itself: asu_wide_rms_params.svh
#         `WIDE_RMS_CHUNK = 128`, and asu_rmsnorm.sv:300-302 ELABORATION-ERRORS
#         if that ever changes. The arm computes
#         mean2 = (ext_sum2·MU[ext_k]) >> (S[ext_k]+16), i.e. a divide by
#         128·ext_k — so arming k = dm/D = 14 would normalize a 896-wide row
#         by 1792 and hand back a wrong (but green-looking-to-the-RTL) answer.
#     gen_layer_ops.py:1071/:1114 uses `k = dm // D_TILE` for BOTH. At
#     D_TILE=128 that is right by coincidence; at 64 it is wrong. Hence a
#     separate builder here rather than a rebind of that one.
#
# ══ WHY THE 0.5B IMAGE IS CFG_DM = CFG_D = 64 (the 14x64 vs 7x128 choice) ══
# Qwen2.5-0.5B's model row is 896. It can be framed for the C-1 feeder as
# 7 x 128 or 14 x 64, and ONLY 14 x 64 is reachable on an image that also
# serves head_dim=64:
#   * apex_top elaborates the feeder at CFG_DM (apex_top.sv:1866
#     `seam_feeder_quant #(.D(CFG_DM))`), and that module accepts D of 64 or
#     128 ONLY (seam_feeder_quant.sv:100-102). So the frame width IS CFG_DM.
#   * a SPLIT build CFG_D=64 / CFG_DM=128 passes apex_top's g_chk_dm
#     (:1854) — but the RoPE q_sink egress (apex_top.sv:1186) and the
#     KVQ-readback requant route push PER-HEAD (64-wide) rows through that
#     same feeder, which cannot frame them at 128. apex_top.sv:1858-1864
#     records exactly this: those routes are "NOT drivable at CFG_DM !=
#     CFG_D" until the stage-6 wide-feeder work lands.
#   * RMSNorm itself is happy either way: with RMS_D_MAX=896 the legal row
#     lengths are pow-2 <= 128 (so 64) or any multiple of 128
#     (asu_rmsnorm.sv:274-276), and in ext mode the streamed row length feeds
#     only framing/legality — the rsqrt operand comes from ext_k.
# => the 0.5B image is CFG_D = CFG_DM = 64, the norm streams 14 x 64, and the
#    arm carries ext_k = 7. Both numbers are asserted below.
#
# ══ THE DISCLOSED BUILD-IDENTITY RETARGET ══════════════════════════════════
# Every emitted program opens with g3.phase_a, which checks the tile's build
# identity — including INFO_TIER, expected as 0x7 at D=64 (the L3 d64
# REFERENCE build, gen_l3_vectors.py:349). apex_top.sv:2540-2543 defines the
# truth as {(KVQ_OUTLIER_K>0)&&mask_valid, (KVQ_GQA_NENG==1), 1'b1}:
#     plain b64 CL (OUTLIER_K=0, GQA=1) -> 0x3   (the p2 gate's retarget)
#     0.5B image   (OUTLIER_K=0, GQA=2) -> 0x1   (GQA build is CQ-8-only)
# `retarget_info_tier` rewrites that ONE expectation per program and REFUSES
# if a program does not contain exactly one — the point of the check is to
# catch a wrong image, so a retarget that quietly matched zero or many sites
# would destroy it. No datapath expectation is ever touched.

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif" / "top" / "l3"),
           str(REPO / "verif" / "seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_l3_vectors as g3                                    # noqa: E402
import trace_to_regops as t2r                                  # noqa: E402
import gemm_job as gj                                          # noqa: E402
import gen_layer_ops as gl                                     # noqa: E402

# The μ-table chunk the R4 arm's k field is denominated in. NOT a tunable:
# asu_wide_rms_params.svh pins WIDE_RMS_CHUNK = 128 and asu_rmsnorm.sv:300-302
# elaboration-errors if a regenerated table changes it.
RMS_MU_CHUNK = 128
# seam_feeder_quant.sv:100-102 — the only legal C-1 feeder row lengths.
FEEDER_D_LEGAL = (64, 128)


@dataclass(frozen=True)
class Image:
    """A cl_apex build's geometry + the build-identity truth it reads back."""
    name: str
    cfg_d: int              # CFG_D — per-head / KVQ-record / rope row
    cfg_dm: int             # CFG_DM — model-wide family = the C-1 frame
    gqa_neng: int           # KVQ_GQA_NENG
    dm_max: int             # RMS_D_MAX == LAYER_DM_MAX (APEX_CL_DM)
    qstage: int             # QSTAGE_H_MAX
    obj: str                # verif/f2sim Mdir holding the twin binary
    agfi: str = ""          # the SILICON image this twin stands for ("" = sim only)

    @property
    def info_tier(self) -> int:
        """apex_top.sv:2540-2543, with cl_apex pinning KVQ_OUTLIER_K = 0."""
        return 0x1 | (0x2 if self.gqa_neng == 1 else 0x0)

    @property
    def binary(self) -> Path:
        return REPO / "verif" / "f2sim" / self.obj / "f2sim"

    def defines(self) -> str:
        return (f"+define+APEX_CL_DM={self.dm_max} "
                f"+define+APEX_CL_GQA={self.gqa_neng} "
                f"+define+APEX_CL_QSTAGE={self.qstage} "
                f"+define+APEX_CL_DMODEL={self.cfg_dm}")

    def rows_per_desc(self, d: int, *, executor: str = "sim") -> int:
        """How many `d`-wide staged rows one descriptor may contract over.

        See §4 below. `executor='sim'` asks THIS TREE's RTL (checked, not
        assumed); `executor='hw'` asks the per-AGFI allowlist.
        """
        return rows_per_desc_for(self, d, executor=executor)


# The image ITEM A adds knobs for, and the image this driver runs against.
# agfi: the b64v4 silicon image the P2 stage-3 hw run flew
# (build/p05b_hw_check2.log, 2 layers, 27,392/27,392 accumulators bit-exact).
IMG_05B = Image(name="b64_05b", cfg_d=64, cfg_dm=64, gqa_neng=2, dm_max=896,
                qstage=14, obj="obj_b64_05b",
                agfi="agfi-0ecab46b8a8376b21")
# The stock D=128 silicon twin, for reference/regression.
IMG_B128 = Image(name="b128", cfg_d=128, cfg_dm=128, gqa_neng=1, dm_max=128,
                 qstage=1, obj="obj_d128_ddr0",
                 agfi="agfi-0ae06ea568e5667ba")


# ═════ 4. THE BASE-ROW (38ec95c) ALLOWLIST — rows_per_desc, per image ══════
#
# ══ WHAT THE RULE WAS, AND WHY IT WAS RIGHT ════════════════════════════════
# `rows_per_desc=1 when executor=='hw'` is written into sum2_mxe.py:42-46,
# breadth_step.py:447-450 and proj_sweep_batched.py:132-134 as "the 05efb2a
# image-parity rule". It is not superstition: it is a MEASURED silicon
# failure. On `agfi-0ae06ea568e5667ba` (built from 05efb2a, 2026-07-22) a
# 16-row descriptor came back WRONG on real hardware —
# docs/results/prompt_on_chip/hw_projection_k2048_FAILED_image_parity_lesson.log
# records `bit_exact=False`, every one of the 8 lanes off, while the same
# operands at rows_per_desc=1 (K=128) graded green
# (hw_projection_k128.log). The cause is in the RTL: before 38ec95c,
# apex_stage_buf's LOAD IGNORED `sel` and wrote from row 0, so all 16 staged
# rows landed on top of each other and 15/16 of the contraction was stale
# data. 38ec95c ("apex_stage_buf LOAD now treats `sel` as the BASE ROW")
# fixed it; every image built after it carries the fix.
#
# ══ WHY THIS IS A PER-IMAGE PROPERTY, NOT A PER-EXECUTOR ONE ═══════════════
# The old rule keys on the EXECUTOR ('hw' -> 1), which was a safe proxy while
# exactly one AFI existed. It is now wrong in both directions: it forbids
# multi-row descriptors on four images that DO have the fix, and it would
# allow them in sim against a twin built from a pre-fix tree. So the gate
# keys on the IMAGE — the AGFI on silicon, the tree's own RTL in sim.
#
# ══ THE EVIDENCE FOR EACH ENTRY ════════════════════════════════════════════
#   * 38ec95c is an ancestor of 9b993bf (`git merge-base --is-ancestor`), and
#     9b993bf is the S3 sweep commit; every image in the ALLOW list below was
#     built from a tree at or after it.
#   * `base_row_fix_in_tree()` re-reads rtl/top/glue/apex_stage_buf.sv and
#     checks the two clauses the fix introduced, so "this tree has it" is a
#     test, not a memory.
#   * agfi-0ecab46b8a8376b21 additionally has the strongest evidence there
#     is: it FLEW 15-row descriptors, 2,160 of them per layer, and returned
#     27,392/27,392 accumulators bit-exact (build/p05b_hw_check2.log). A
#     pre-fix image cannot do that.
BASE_ROW_FIX_COMMIT = "38ec95c"

IMAGES_WITH_BASE_ROW_FIX = {
    "agfi-0ecab46b8a8376b21":
        "b64v4 / b64_05b — the 0.5B D=64 image. FLEW 15-row descriptors "
        "bit-exact, 27,392/27,392 accumulators over 2 layers "
        "(build/p05b_hw_check2.log, 2026-08-03).",
    "agfi-006b1314fcbbb3505":
        "narrow image #3 (E-lane walker, @47a5a3a) — tree contains 38ec95c.",
    "agfi-09e947873048a6877":
        "narrow image #2 (R4) — tree contains 38ec95c.",
    "agfi-0cc7aa798fe3abce2":
        "narrow image #1 (layer window) — tree contains 38ec95c.",
}

IMAGES_WITHOUT_BASE_ROW_FIX = {
    "agfi-0ae06ea568e5667ba":
        "the C1/D image, built from 05efb2a (2026-07-22) — PREDATES 38ec95c "
        "(2026-07-26). A 16-row descriptor measured bit_exact=False on this "
        "image: docs/results/prompt_on_chip/"
        "hw_projection_k2048_FAILED_image_parity_lesson.log",
    "agfi-0f7c93ffa798ecc3f":
        "the D=64 first-light control image — same era, same absence.",
}

_STAGE_BUF_RTL = REPO / "rtl" / "top" / "glue" / "apex_stage_buf.sv"


def base_row_fix_in_tree(path: Path = _STAGE_BUF_RTL) -> dict:
    """Is 38ec95c's base-row LOAD in THIS tree's apex_stage_buf.sv?

    Checked against the two clauses the commit introduced, both of which a
    pre-fix file lacks:
      * the LOAD legality clause `sel + rows <= R_MAX`
      * the write pointer init `row_c <= job_op ? 5'd0 : job_sel`
    A rename or a rewrite lands here as "unknown", never as a silent yes.
    """
    src = Path(path).read_text()
    legality = "6'({1'b0, job_sel}) + 6'({1'b0, job_rows})" in src
    base_ptr = "row_c   <= job_op ? 5'd0 : job_sel;" in src
    return {"path": str(path), "legality_clause": legality,
            "base_row_pointer": base_ptr,
            "present": bool(legality and base_ptr)}


def has_base_row_fix(agfi: str) -> dict:
    """{'fix': bool|None, 'why': str} for a SILICON image, by AGFI.

    An AGFI on neither list returns fix=None — UNKNOWN, which callers must
    treat as "no" (the conservative direction: a wrong answer here is silent
    numerical garbage on the card, exactly the 2026-07-30 incident).
    """
    a = (agfi or "").strip()
    if a in IMAGES_WITH_BASE_ROW_FIX:
        return {"fix": True, "why": IMAGES_WITH_BASE_ROW_FIX[a], "agfi": a}
    if a in IMAGES_WITHOUT_BASE_ROW_FIX:
        return {"fix": False, "why": IMAGES_WITHOUT_BASE_ROW_FIX[a], "agfi": a}
    return {"fix": None, "agfi": a,
            "why": f"AGFI {a!r} is on neither the with- nor the without-fix "
                   f"list in tile_geom.py. Treating it as PRE-FIX "
                   f"(rows_per_desc=1): a multi-row descriptor on an image "
                   f"without {BASE_ROW_FIX_COMMIT} returns silently wrong "
                   f"numbers. Add the AGFI to IMAGES_WITH_BASE_ROW_FIX once "
                   f"its build tree is known to contain that commit."}


def rows_per_desc_for(image: Image, d: int, *, executor: str = "sim",
                      agfi: str | None = None) -> int:
    """The legal staged-rows-per-descriptor for `image` at row width `d`.

    The ceiling is the smaller of the two hardware bounds:
      * K_MAX / d      — the frozen descriptor contract (gemm_job.K_MAX=2048)
      * STAGE_R_MAX    — rows resident in one stage-buffer bank (31)
    and it collapses to 1 without the base-row fix.
    """
    ceiling = min(gj.K_MAX // int(d), gj.STAGE_R_MAX)
    if executor == "hw":
        return ceiling if has_base_row_fix(agfi or image.agfi)["fix"] else 1
    # sim / golden: the twin is built from THIS tree
    return ceiling if base_row_fix_in_tree()["present"] else 1


def rows_per_desc_note(image: Image, d: int, *, executor: str = "sim",
                       agfi: str | None = None) -> str:
    """One ledger line saying WHICH rule fired and on what evidence."""
    rpd = rows_per_desc_for(image, d, executor=executor, agfi=agfi)
    if executor == "hw":
        v = has_base_row_fix(agfi or image.agfi)
        state = {True: "HAS", False: "LACKS", None: "UNKNOWN for"}[v["fix"]]
        return (f"rows_per_desc={rpd} (image {v['agfi'] or '<unset>'} {state} "
                f"the {BASE_ROW_FIX_COMMIT} base-row LOAD; {v['why']})")
    t = base_row_fix_in_tree()
    return (f"rows_per_desc={rpd} (sim twin built from this tree; "
            f"{BASE_ROW_FIX_COMMIT} base-row LOAD present in "
            f"{Path(t['path']).name}: {t['present']})")


# ═════════════════════ 1. the geometry rebind (verified) ═══════════════════

# module -> {attribute: f(D) -> value}. Every entry is a constant the frozen
# emitters DERIVE from D_TILE at import time, so rebinding D_TILE alone would
# leave a stale, silently-wrong bound behind.
_REBIND = {
    gj: {
        "D_TILE": lambda d: d,
        "BPR": lambda d: d // 8,
        "ROWS_PER_DESC": lambda d: gj.K_MAX // d,
    },
    gl: {
        "D_TILE": lambda d: d,
        "BPR": lambda d: d // 8,
        "INJ_K128_LANES": lambda d: d - 1,
        "INJ_K128_MAX": lambda d: 127 * (127 * (d - 1)) + 63,
    },
}


def _check_derivations() -> None:
    """The rebind table must reproduce the modules' OWN import-time values at
    their own D_TILE — otherwise it is not a rebind, it is a redefinition."""
    for mod, spec in _REBIND.items():
        d0 = int(mod.D_TILE)
        for attr, fn in spec.items():
            want, got = fn(d0), getattr(mod, attr)
            if want != got:
                raise AssertionError(
                    f"tile_geom: rebind table disagrees with {mod.__name__}."
                    f"{attr} at its own D_TILE={d0}: table says {want}, module "
                    f"has {got}. The frozen module changed — re-read it before "
                    f"trusting any geometry this file emits.")


@contextmanager
def at_d(d: int):
    """Emit with gen_layer_ops / gemm_job reframed at row width `d`."""
    if d not in FEEDER_D_LEGAL:
        raise AssertionError(
            f"tile_geom.at_d({d}): the C-1 feeder accepts only "
            f"{FEEDER_D_LEGAL} (seam_feeder_quant.sv:100-102), and every op "
            f"family here egresses through it")
    _check_derivations()
    saved = {(m, a): getattr(m, a) for m, s in _REBIND.items() for a in s}
    try:
        for mod, spec in _REBIND.items():
            for attr, fn in spec.items():
                setattr(mod, attr, fn(d))
        yield d
    finally:
        for (mod, attr), v in saved.items():
            setattr(mod, attr, v)


INFO_D_ADDR = t2r.B_CSR + g3.CSR["INFO_D"]      # 0x100C
T_ROW_MAX = 128                                  # apex_top.sv:156 envelope


def audit_geometry(regops_path, d: int) -> dict:
    """PROVE the rebind reached the wire, at COMPILE time.

    The decisive site is phase A's INFO_D read (gen_l3_vectors.py:347,
    `s.csrr(CSR["INFO_D"], .., s.D)`): every emitter in this lane builds its
    script as `Script(D_TILE)`, so the expectation baked there IS the frame
    width the whole program was emitted at. If the rebind had not taken, a
    program emitted at 128 would carry INFO_D==128 — and would then fail on a
    D=64 tile at run time. Catching it here means never running it.

    A descriptor-k histogram is collected too, but as EVIDENCE, not as the
    gate: a `k` is legitimately the loader row (2), the phase-A reject (0), a
    staged-row contraction (a multiple of d — 64 for one row, 960 for the
    15-row projection stage) or the attention P·V contraction over T
    (<= T_ROW_MAX). Anything else means the emitter framed something this
    audit does not understand, and that refuses.
    """
    ops = [json.loads(ln) for ln in
           Path(regops_path).read_text().splitlines() if ln.strip()]
    seen = [int(o["e"]) for o in ops
            if o.get("op") == "r" and o.get("a") == INFO_D_ADDR]
    if len(seen) != 1 or seen[0] != d:
        raise AssertionError(
            f"{regops_path}: INFO_D expectations {seen}, wanted exactly one "
            f"== {d}. The program was NOT emitted at D={d} (the geometry "
            f"rebind did not reach the emitter), or its build-identity block "
            f"is not the one this audit was written against.")
    ks, bad = [], []
    for o in gj.decode_desc_pushes(ops):
        k = int(o["k"])
        ks.append(k)
        if not (k in (0, 2) or (k % d == 0 and k > 0) or 0 < k <= T_ROW_MAX):
            bad.append(k)
    if bad:
        raise AssertionError(
            f"{regops_path}: descriptor(s) framed k={sorted(set(bad))} on a "
            f"D={d} image — not a loader row, not a multiple of {d}, not a "
            f"T<={T_ROW_MAX} contraction. Refusing to run a frame this audit "
            f"cannot account for.")
    return {"descriptors": len(ks), "k_values": sorted(set(ks)), "info_d": d}


# ══════════════════ 2. the disclosed INFO_TIER retarget ════════════════════

INFO_TIER_ADDR = t2r.B_CSR + 0x14           # BAR0 CSR window + INFO_TIER


def retarget_info_tier(path, image: Image, *, l3_expect: int | None = None):
    """Rewrite the ONE build-identity expectation, in place. Refuses on 0 or
    2+ sites: the check exists to catch a wrong image, and a retarget that
    silently matched nothing would leave that hole open with a green log."""
    p = Path(path)
    want_from = l3_expect if l3_expect is not None else (
        0x7 if image.cfg_d == 64 else 0x3)         # gen_l3_vectors.py:349
    want_to = image.info_tier
    if want_from == want_to:
        return {"retargeted": 0, "note": "image truth == L3 expectation"}
    out, n = [], 0
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        o = json.loads(ln)
        if (o.get("op") == "r" and o.get("a") == INFO_TIER_ADDR
                and o.get("e") == want_from):
            o["e"] = want_to
            n += 1
            out.append(json.dumps(
                {"op": "note",
                 "s": f"INFO_TIER expect retargeted {want_from:#x}->"
                      f"{want_to:#x} (image {image.name}: KVQ_OUTLIER_K=0, "
                      f"KVQ_GQA_NENG={image.gqa_neng} -> CQ-8-only; "
                      f"apex_top.sv:2540-2543). Jobs are CQ-8; NO datapath "
                      f"expectation touched."}, separators=(",", ":")))
        out.append(json.dumps(o, separators=(",", ":")))
    if n != 1:
        raise AssertionError(
            f"{p}: expected exactly ONE INFO_TIER expectation of "
            f"{want_from:#x}, found {n}. Refusing to retarget — a "
            f"build-identity check that matches 0 or 2+ sites is not the "
            f"check this retarget was audited against.")
    p.write_text("\n".join(out) + "\n")
    return {"retargeted": n, "from": want_from, "to": want_to}


# ══════════════ 3. the RE-DERIVED chunked RMSNorm-2 (ext_k fix) ════════════

def norm2_plan(dm: int, d: int) -> dict:
    """The two chunk counts, kept apart on purpose. See the header."""
    if dm % d:
        raise AssertionError(f"dm={dm} is not a whole number of {d}-wide "
                             f"C-1 frames")
    if dm % RMS_MU_CHUNK:
        raise AssertionError(
            f"dm={dm} is not a multiple of {RMS_MU_CHUNK}: the R4 arm's k "
            f"field is denominated in {RMS_MU_CHUNK}-element chunks "
            f"(asu_wide_rms_params.svh WIDE_RMS_CHUNK), so a row that is not "
            f"a whole number of them has NO exact μ-table entry")
    ext_k = dm // RMS_MU_CHUNK
    if not gl.RMS_EXT_K_MIN <= ext_k <= gl.RMS_EXT_K_MAX:
        raise AssertionError(
            f"ext_k={ext_k} outside the arm envelope "
            f"[{gl.RMS_EXT_K_MIN},{gl.RMS_EXT_K_MAX}] (apex_top R4)")
    return {"dm": dm, "d": d, "rows": dm // d, "ext_k": ext_k}


def norm2_chunk_sums_d(step, dm: int, d: int):
    """Per-C-1-frame sums + the row total, with golden's own width proofs."""
    plan = norm2_plan(dm, d)
    r1_8 = np.asarray(step["r1_8"], dtype=np.int64)[:dm]
    assert r1_8.size == dm, f"r1_8 short: {r1_8.size} < {dm}"
    sums = [int(np.sum(r1_8[c * d:(c + 1) * d] ** 2)) for c in range(plan["rows"])]
    total = sum(sums)
    # apex_top R4 arm bound, mirrored at emission with the k WE WILL ARM.
    assert total <= (plan["ext_k"] << 21), (
        f"sum2 {total} exceeds the arm's k·2^21 = {plan['ext_k'] << 21} — "
        f"apex_top would refuse (err_code 8)")
    return r1_8, sums, total, plan


def build_norm2_chunked_d(step, *, dm: int, d: int, out_dir: Path, name: str):
    """R4 two-pass chunked RMSNorm-2 at an arbitrary C-1 frame width.

    Identical in choreography to gen_layer_ops.build_norm2_chunked; the ONE
    difference is that the streamed chunk width (d, the feeder frame) and the
    armed μ index (dm/128) are computed separately. See this file's header.
    """
    r1_8, sums, total, plan = norm2_chunk_sums_d(step, dm, d)
    rows, ext_k = plan["rows"], plan["ext_k"]
    assert rows <= 31, f"feeder/stage rows {rows} exceed ROWS_MAX 31"
    g2 = np.asarray(step["gamma2"], dtype=np.int64)[:dm]

    s = gl.LayerScript(d)
    s.emit(f"// LAYER stage {name} (R4, re-derived at D={d}): FULL dm={dm} "
           f"RMSNorm-2 as golden rmsnorm_fx_wide's own composition. "
           f"{rows} chunk-sum exports of {d} elements (the C-1 feeder frame, "
           f"seam_feeder_quant #(.D(CFG_DM={d}))), host ADDS them "
           f"({rows - 1} INT32 adds), then the tile runs μ+rsqrt+all {dm} "
           f"per-element ops. THE ARM CARRIES k={ext_k} = {dm}/"
           f"{RMS_MU_CHUNK}, NOT the {rows} streamed chunks: the μ table is "
           f"indexed in {RMS_MU_CHUNK}-element units by construction "
           f"(asu_wide_rms_params.svh WIDE_RMS_CHUNK, guarded at "
           f"asu_rmsnorm.sv:300-302), so k={rows} would normalize this row "
           f"by {rows * RMS_MU_CHUNK} instead of {dm}.")
    gl._prologue(s)
    s.pmode(True)
    s.emit("// ── pass 1: sum-capture rows (no rsqrt, no emission, no gamma)")
    s.nctl(sum_en=1, clr=1)
    for c in range(rows):
        s.xrow_n(r1_8[c * d:(c + 1) * d])
        s.npoll_cnt(c + 1)
        s.ns2cap()
    s.nctl()
    s.emit(f"// ── pass 2: total staged + armed (ext_k={ext_k}); the ONE "
           f"broadcast r, then the per-element datapath per chunk. Feeder "
           f"re-entry framed {rows} x {d} (B-FEED-WIDTH), scale popped per "
           f"row (B-FS-FIFO).")
    s.nsum2(total)
    s.nctl(ext=1, k=ext_k)
    s.route(rdst=0, asrc=0)
    s.aj(0, gl.ACT_BANK, 0, rows, s.BPR, 0)
    s.fjob(rows)
    for c in range(rows):
        s.xrow_n(r1_8[c * d:(c + 1) * d])
        s.grow_n(g2[c * d:(c + 1) * d])
        s.efs(0, 0)
    s.emit("// codes back through identity GEMMs")
    s.route(rdst=0, wsrc=0, asrc=0)
    for r in range(rows):
        for b in range(0, d, gl.MXE_N):
            gl.identity_readback(s, r, b)
    s.nctl()
    s.pmode(False)
    gl._drain_and_check(s)

    x = gl._translate(s, note=f"LAYER {name}: R4 chunked RMSNorm-2 dm={dm} "
                              f"at D={d} (ext_k={ext_k})")
    return gl._emit(x, out_dir, name, dict(
        stage="norm2c", kind="xa chunks->sum2 exports->host adds->ext arm->"
                             "asu_rmsnorm->widen->feeder->act->RO",
        dm=int(dm), frame_d=int(d), feeder_rows=int(rows), ext_k=int(ext_k),
        ext_sum2=int(total), chunk_sums=[int(v) for v in sums],
        host_arith=f"{rows - 1} INT32 adds (chunk-sum accumulation)"))


# ══════════════════════════════ selftest ═══════════════════════════════════

def selftest() -> int:
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("TILE_GEOM SELFTEST")
    # --- the rebind ---------------------------------------------------------
    d0 = (gl.D_TILE, gj.D_TILE, gl.INJ_K128_LANES, gj.ROWS_PER_DESC)
    with at_d(64):
        chk("at_d(64) reframes BOTH frozen emitters and their derived bounds",
            (gl.D_TILE, gj.D_TILE, gl.BPR, gj.BPR, gl.INJ_K128_LANES,
             gj.ROWS_PER_DESC) == (64, 64, 8, 8, 63, 32))
        chk("a Script built inside the context frames at 64",
            gl.LayerScript(gl.D_TILE).BPR == 8)
    chk("every rebound global restored on exit",
        (gl.D_TILE, gj.D_TILE, gl.INJ_K128_LANES, gj.ROWS_PER_DESC) == d0,
        str(d0))
    try:
        with at_d(64):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    chk("restored even when the body raises",
        (gl.D_TILE, gj.D_TILE) == (d0[0], d0[1]))
    try:
        with at_d(96):
            pass
        chk("an illegal feeder width is REFUSED", False, "no refusal")
    except AssertionError as e:
        chk("an illegal feeder width is REFUSED", "seam_feeder_quant" in str(e))

    # --- the norm plan: the whole point of the re-derivation ----------------
    p = norm2_plan(896, 64)
    chk("0.5B norm: 14 streamed 64-wide C-1 frames, armed ext_k = 7",
        (p["rows"], p["ext_k"]) == (14, 7), str(p))
    p128 = norm2_plan(896, 128)
    chk("at D=128 the two counts COINCIDE (why the frozen emitter is right "
        "there)", p128["rows"] == p128["ext_k"] == 7)
    chk("frozen gen_layer_ops would have armed k=14 at D=64 (the bug this "
        "re-derivation exists for)", 896 // 64 == 14)
    try:
        norm2_plan(896 + 64, 64)
        chk("a row that is not a whole number of μ chunks is REFUSED", False)
    except AssertionError as e:
        chk("a row that is not a whole number of μ chunks is REFUSED",
            "WIDE_RMS_CHUNK" in str(e))
    # the arm bound is EXACTLY tight for 896 all-(-128) — check we did not
    # write a bound that is accidentally loose
    step = {"r1_8": np.full(896, -128, dtype=np.int64),
            "gamma2": np.zeros(896, dtype=np.int64)}
    _r, _s, tot, pl = norm2_chunk_sums_d(step, 896, 64)
    chk("the all-(-128) row sits EXACTLY on the arm's k·2^21 bound",
        tot == (pl["ext_k"] << 21) == 14680064, f"{tot}")
    bad = dict(step)
    try:
        norm2_chunk_sums_d(bad, 896, 64)
        over = False
    except AssertionError:
        over = True
    chk("...and is accepted, not refused (the bound is <=, not <)", not over)

    # --- image identity -----------------------------------------------------
    chk("INFO_TIER truth: plain b64 CL (GQA=1) reads 0x3",
        Image("x", 64, 64, 1, 128, 1, "o").info_tier == 0x3)
    chk("INFO_TIER truth: the 0.5B image (GQA=2) reads 0x1",
        IMG_05B.info_tier == 0x1)

    # --- the base-row allowlist (rows_per_desc, per image) ------------------
    t = base_row_fix_in_tree()
    chk(f"{BASE_ROW_FIX_COMMIT}'s base-row LOAD is in THIS tree's "
        f"apex_stage_buf.sv (both clauses)", t["present"], str(t))
    chk("the LEGACY image is refused multi-row descriptors "
        "(agfi-0ae06ea568e5667ba: the measured bit_exact=False case)",
        rows_per_desc_for(IMG_B128, 128, executor="hw",
                          agfi="agfi-0ae06ea568e5667ba") == 1)
    chk("the b64v4 image (which FLEW 15-row descriptors bit-exact) gets the "
        "full ceiling at D=64",
        rows_per_desc_for(IMG_05B, 64, executor="hw") == 31,
        f"min(K_MAX/64={gj.K_MAX // 64}, STAGE_R_MAX={gj.STAGE_R_MAX})")
    chk("...and at D=128 the K_MAX bound is the binding one",
        rows_per_desc_for(IMG_05B, 128, executor="hw") == 16)
    chk("an UNKNOWN AGFI is treated as PRE-FIX, not as fine",
        rows_per_desc_for(IMG_05B, 64, executor="hw",
                          agfi="agfi-000000000000000ff") == 1
        and has_base_row_fix("agfi-000000000000000ff")["fix"] is None)
    chk("sim asks the TREE, not the AGFI list",
        rows_per_desc_for(IMG_05B, 64, executor="sim") == 31)
    chk("the two lists are disjoint",
        not (set(IMAGES_WITH_BASE_ROW_FIX) & set(IMAGES_WITHOUT_BASE_ROW_FIX)))
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        fake = Path(_td) / "prefix_stage_buf.sv"
        fake.write_text(Path(_STAGE_BUF_RTL).read_text().replace(
            "row_c   <= job_op ? 5'd0 : job_sel;", "row_c   <= 5'd0;"))
        chk("a PRE-FIX stage buffer is detected (the mutant: LOAD writes from "
            "row 0)", base_row_fix_in_tree(fake)["present"] is False)

    # --- a real emission end to end (host only) -----------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(7)
        step = {"r1_8": rng.integers(-127, 128, 896),
                "gamma2": rng.integers(-4000, 4000, 896)}
        with at_d(64):
            man = build_norm2_chunked_d(step, dm=896, d=64, out_dir=td,
                                        name="t_norm")
        chk("emitted a dm=896 norm program at D=64",
            man["feeder_rows"] == 14 and man["ext_k"] == 7, str(man["path"]))
        a = audit_geometry(man["path"], 64)
        chk("the program's own INFO_D expectation is 64", a["info_d"] == 64,
            str(a))
        try:
            audit_geometry(man["path"], 128)
            chk("auditing the SAME program as D=128 REFUSES", False)
        except AssertionError as e:
            chk("auditing the SAME program as D=128 REFUSES",
                "INFO_D expectations" in str(e))
        r = retarget_info_tier(man["path"], IMG_05B)
        chk("the INFO_TIER retarget hit exactly one site (0x7 -> 0x1)",
            r["retargeted"] == 1 and r["to"] == 0x1, str(r))
        try:
            retarget_info_tier(man["path"], IMG_05B)
            chk("a second retarget REFUSES (0 sites left)", False)
        except AssertionError as e:
            chk("a second retarget REFUSES (0 sites left)",
                "exactly ONE" in str(e))

    print("=" * 60)
    if fails:
        print(f"TILE_GEOM SELFTEST FAIL: {fails}")
        return 1
    print("TILE_GEOM SELFTEST: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
