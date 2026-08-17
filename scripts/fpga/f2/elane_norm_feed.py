#!/usr/bin/env python3
# elane_norm_feed.py — E-lane E-1/E-2 demonstrator (E2E_TOY_LANE.md §4):
# an activation PRODUCED BY THE RESIDUAL reaches the NORM without leaving
# the tile, graded bit-exact against the golden arbiter.
#
# ══ WHAT THIS PROVES (and what it does not) ════════════════════════════════
# Until 2026-07-31 the decoder layer's chain was tile-internal everywhere
# EXCEPT one seam it crossed twice (layer entry X -> NORM1, mid-layer
# r1 -> NORM2): the residual row RAM had no internal egress (its only reader
# was rd_data -> the CSR path) and asu_rmsnorm's x input was a TOP-LEVEL
# PORT of apex_top — structurally host-fed. The E-1/E-2 RTL closes both:
#   E-1   apex_residual egress job (LAYER_JOB unit 3, R3 window-base
#         addressing) + fp16 stream -> the C-1 feeder as l_fsrc_ext code 4
#         (LAYER_CTRL[7] is l_fsrc_ext[2]).
#   E-2a  l_nsrc (LAYER_CTRL[19]): the feeder's INT8 codes, unpacked
#         lane8 -> serial (apex_lane8_unpack), drive the norm's x port
#         instead of xa_*.
#   E-2b  the unit-3 push is the norm-feed START verb; LAYER_STATUS[5]
#         (nf_busy) is the whole-path completion poll; a push with the
#         route unarmed is REFUSED loudly (err_code 9, NFEED_ROUTE).
#
# The nfeed stage here runs, on the 128-wide toy configuration (the shape
# where head_dim == D_model == 128 is consistent and the C-1 quantizer is
# natively full-row — E2E_TOY_LANE.md §2):
#     preload X ─┐                                        (host: layer input)
#     o8+comp  ──┴─> serializer -> layer-deq -> RESIDUAL  r1 = f16(X+o8·comp)
#     r1 (row RAM) ──E-1──> C-1 feeder ──E-2a──> RMSNorm  ENTIRELY IN-TILE
#     y ──> widen ──> C-1 feeder ──> act stage ──> identity GEMM + fs tap
# and grades, all against golden's OWN functions applied golden's own way
# (transformer.py:568-572 — the r1 -> NORM2 entry):
#     r1 row bits          == f16(X + o8·comp)            (ERD readback)
#     fs cap 1             == quant_rows_i8(r1).scale     (golden computes
#                             and DISCARDS it — RMSNorm is scale-invariant;
#                             we grade it as provenance of the C-1 the norm
#                             actually consumed)
#     identity-GEMM codes  == quant_rows_i8(rmsnorm_fx(quant_rows_i8(r1)
#                             .codes, g2)/256).codes      (fs cap 2 == its
#                             scale)
# The host's remaining role in the demonstrated chain: load the layer input
# X and gamma (weights/inputs), write control-plane levels and job pushes,
# and DRAIN THE NORM'S OUTPUT (the y -> projections seam is not this lane's;
# the claim is exactly the residual -> norm seam, host-free).
#
# NEW FILE ON PURPOSE: gen_layer_ops.py / narrow_flight.py / prompt_offload.py
# are concurrently owned by other sessions. This emitter IMPORTS the frozen
# vocabulary (LayerScript tokens, inject/readback idioms, graders, runner)
# and adds ONLY the E-1/E-2 words. Zero baked output expectations
# (audit_program enforced), discriminators that must go red (--smoke runs
# them), golden the only arbiter.
#
# CLI:
#   python3 scripts/fpga/f2/elane_norm_feed.py --selftest   # host only
#   python3 scripts/fpga/f2/elane_norm_feed.py --build      # emit programs
#   python3 scripts/fpga/f2/elane_norm_feed.py --smoke      # build+run+grade
#     [--binary PATH] [--tile-div N] [--out DIR] [--no-discriminate]
#   (RED evidence = --smoke --binary <pre-E1 twin> --out <red dir>: the
#    probes' checked reads and the nfeed grade must FAIL there.)

from __future__ import annotations

import argparse
import json
import struct
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
import cap_decode as cd                                        # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

from gen_layer_ops import (                                    # noqa: E402
    LayerScript, D_TILE, MXE_N, BANK_RESID, LU_RESID, LU_DEQ,
    LST_LDQ, LST_SWG, LST_RES, LST_ROPE, LST_ERR, L_CTRL, L_JOB,
    _translate, _emit, _prologue, _drain_and_check, inject_frame,
    identity_readback, grade_resid, eprint,
)

# ── the E-1/E-2 consumer rule, MIRRORED at emission (the R3 discipline:
#    an illegal program raises HERE, not on the wire; the probes use the
#    raw path on purpose). Consumer: rtl/top/apex_top.sv E-1/E-2 region.
LCTL_FSRC_EXT2 = 1 << 7        # l_fsrc_ext[2] -> {[7],[5:4]} = 3'b100 = code 4
LCTL_NSRC = 1 << 19            # E-2a: feeder codes -> the norm's x mux
LCTL_NFEED = LCTL_FSRC_EXT2 | LCTL_NSRC
LU_NORM = 3                    # E-2b: LAYER_JOB unit 3 = the NORM/EGRESS job
LST_NFEED = 0x20               # LAYER_STATUS[5] nf_busy (whole feed path)
LERR_NFEED_ROUTE = 9           # push-gate refusal code (route unarmed)
LERR_FRAME = 3                 # in-unit geometry refusal class (R3's)

DEFAULT_OUT = REPO / "build" / "f2_elane"


def nfjob(s: LayerScript, cols: int, base: int = 0, *, armed_word: int,
          dm_max: int = D_TILE, raw: bool = False):
    """Push the E-2b NORM/EGRESS job (LAYER_JOB unit 3).

    Legality mirrored from the consumer unless raw=True (the LUJOBR idiom —
    refusal probes exist to watch the RTL refuse):
      * the route must be armed (l_fsrc_ext code 4 — apex_top refuses with
        err_code 9 otherwise), tracked here via the level word the program
        last wrote;
      * cols must be a nonzero multiple of the feeder row D_TILE (same
        refusal);
      * base*1024 + cols must fit DM_MAX (apex_residual refuses in-unit,
        frame class code 3).
    """
    if not raw:
        assert (armed_word & LCTL_NFEED) == LCTL_NFEED and \
            (armed_word & 0x30) == 0, \
            f"nfeed push with the route unarmed (LCTL={armed_word:#x}) — " \
            f"apex_top would refuse (err_code {LERR_NFEED_ROUTE})"
        assert cols and cols % D_TILE == 0, \
            f"cols={cols} does not frame whole {D_TILE}-element C-1 rows " \
            f"— apex_top would refuse (err_code {LERR_NFEED_ROUTE})"
        assert base * 1024 + cols <= dm_max, \
            f"footprint {base * 1024 + cols} > DM_MAX={dm_max} — " \
            f"apex_residual would refuse (frame class, code {LERR_FRAME})"
    s.csrw(L_JOB, ((base & 3) << 14) | (LU_NORM << 12) | (cols & 0xFFF))


# ═══════════════ toy step (E-4: subject synthetic, arbiter GOLDEN) ═════════

def _f32_val(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def toy_step(seed: int = 20260731) -> dict:
    """The synthetic 128-dim single-head layer slice this lane demonstrates
    on. Every reference below is computed by apex_golden functions applied
    exactly the way transformer.py applies them at the r1 -> NORM2 seam
    (:568-572); nothing here re-implements an op."""
    rng = np.random.default_rng(seed)
    # layer-entry row X on the fp16 grid (the residual RAM's resident row)
    x_bits = f64_to_f16_bits(rng.normal(0.0, 1.5, D_TILE))
    x_vals = f16_bits_to_f64(np.asarray(x_bits, dtype=np.uint16))
    # the sublayer stream: o8 codes x one fp16-graded fp32 composite —
    # the D-030 operand shape apex_layer_deq reconstructs EXACTLY
    o8 = rng.integers(-127, 128, D_TILE, dtype=np.int64)
    comp_bits = fmt.grade_f32(2.0 ** -6)
    comp_val = _f32_val(comp_bits)
    b = o8.astype(np.float64) * comp_val
    # r1 = f16(X + b) — golden's C-6 (transformer.py `_f16(X[T] + ...)`),
    # ONE RNE onto the fp16 grid
    r1_bits = f64_to_f16_bits(x_vals + b)
    r1_vals = f16_bits_to_f64(np.asarray(r1_bits, dtype=np.uint16))
    # golden NORM2 entry (:568-570): C-1 the fp16 row, norm the CODES
    # (the row scale is discarded there — kept here as an fs-tap reference)
    r1_codes, r1_scales = at.quant_rows_i8(r1_vals[None, :])
    g2 = rng.integers(-12288, 12289, D_TILE, dtype=np.int64)   # Q2.13, |g|<=1.5
    h2, _r, _nrm = at.rmsnorm_fx([int(c) for c in r1_codes[0]],
                                 [int(v) for v in g2])
    # golden :572: the norm output re-enters through C-1 at /256 (Q7.8)
    h2_vals = np.asarray(h2, dtype=np.float64) / 256.0
    h2_codes, h2_scales = at.quant_rows_i8(h2_vals[None, :])
    return dict(
        seed=int(seed),
        x_bits=np.asarray(x_bits, dtype=np.uint16),
        o8=o8, comp=int(comp_bits),
        r1_bits=np.asarray(r1_bits, dtype=np.uint16),
        r1_codes=np.asarray(r1_codes[0], dtype=np.int64),
        r1_scale=int(r1_scales[0]),
        gamma2=g2,
        h2=np.asarray(h2, dtype=np.int64),
        h2_codes=np.asarray(h2_codes[0], dtype=np.int64),
        h2_scale=int(h2_scales[0]),
    )


# ═══════════════ the stage programs ════════════════════════════════════════

def build_nfeed_stage(step: dict, *, out_dir: Path, name: str,
                      perturb: str | None = None) -> dict:
    """The E-1/E-2 chain: residual-PRODUCED r1 -> C-1 -> norm, in-tile.

    perturb (discriminators; each must turn the grade red):
      'x'      one X element moved 1 fp16 ulp  -> r1 AND the norm chain move
      'gamma'  one gamma word moved            -> the norm output codes move
    """
    x_bits = step["x_bits"].copy()
    g2 = step["gamma2"].copy()
    if perturb == "x":
        x_bits[5] = np.uint16(int(x_bits[5]) ^ 1)          # 1 ulp
    elif perturb == "gamma":
        # pick a delta that PROVABLY moves the GOLDEN reference — the C-1
        # re-quantization can absorb a small gamma move (measured: +64 at
        # index 7 left all 128 output codes AND the scale unchanged on the
        # toy row, which would make the discriminator vacuous, not red)
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
    s.emit(f"// E-LANE stage {name}: residual -> C-1 feeder -> RMSNorm with "
           f"the activation NEVER leaving the tile (E-1/E-2). Toy 128-wide "
           f"configuration (E2E_TOY_LANE.md §2). Choreography rules are the "
           f"frozen §3b ones + the E-2b poll (LAYER_STATUS[5]).")
    _prologue(s)
    s.emit("// [1] host loads the LAYER INPUT row X (its legitimate role) "
           "into the residual RAM (LAYER_PTR bank 2, auto-inc)")
    s.lptr(BANK_RESID, 0)
    s.ldata(x_bits)
    s.emit("// [2] the tile PRODUCES r1 = f16(X + o8*comp): consumer-before-"
           "producer residual job, JOBC immediately before its deq push")
    s.lctl(ser_dst=1, resid_arm=1)
    s.route(rdst=1)
    s.pmode(True)
    s.lujob(LU_RESID, D_TILE)
    s.ljc(step["comp"])
    s.lujob(LU_DEQ, D_TILE)
    inject_frame(s, step["o8"])
    s.lpoll(LST_LDQ | LST_RES, 0x0)
    s.emit("// [3] E-1/E-2: r1 -> C-1 -> norm ENTIRELY IN-TILE. Routes for "
           "the LATER drain are set NOW — the route token guards on "
           "whole-tile idleness, and from the norm-feed start until gamma "
           "arrives the norm is legitimately busy holding the row (its "
           "STATUS lane would park that guard forever). Then: arm the "
           "levels (fsrc_ext=4 + nsrc), feeder job for the r1 row, the "
           "unit-3 NORM/EGRESS job; poll LAYER_STATUS[5] (nf_busy) until "
           "the whole row has reached the norm; pop the r1 C-1 row scale "
           "(golden computes and discards it — graded here as provenance)")
    s.route(rdst=0, asrc=0)
    s.csrw(L_CTRL, LCTL_NFEED)
    s.fjob(1)
    nfjob(s, D_TILE, 0, armed_word=LCTL_NFEED)
    s.lpoll(LST_NFEED, 0x0)
    s.efs(0, 0)                          # -> cap: quant_rows_i8(r1).scale
    s.emit("// [4] drain the norm's output through the LEGACY path — "
           "golden's own next step, h2_8 = quant_rows_i8(h2/256): levels "
           "back to 0 (both touched paths quiescent after the [3] poll; a "
           "plain CSR write, no idle guard), act LOAD armed (xa/xg-driven "
           "flows may hold it — the norm2 idiom), feeder job, then gamma "
           "releases the emission")
    s.csrw(L_CTRL, 0)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.fjob(1)
    s.grow_n(g2)
    s.efs(0, 0)                          # -> cap: quant_rows_i8(h2/256).scale
    s.emit("// [5] the h2 codes back through identity GEMMs (the ONLY act-"
           "stage egress; produce-mode caps). The route token's idle guard "
           "IS the synchronization point: it passes once the norm has "
           "emitted, the feeder has drained and the act LOAD completed.")
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, D_TILE, MXE_N):
        identity_readback(s, 0, base)
    s.emit("// [6] r1 observability AFTER the feed (egress does not mutate "
           "the row): one bus read per element")
    s.lrptr(0)
    s.erd(D_TILE)
    s.pmode(False)
    _drain_and_check(s)

    x = _translate(s, note=f"E-LANE {name}: resid->C1->norm in-tile")
    return _emit(x, out_dir, name, dict(
        stage="nfeed",
        kind="deq->residual; resid-EGRESS->feeder->rmsnorm (in-tile); "
             "widen->feeder->act->RO",
        cols=int(D_TILE), perturb=perturb or "",
        jobc=f"{step['comp']:#010x}", seed=step["seed"]))


def build_route_probe(*, out_dir: Path, name: str) -> dict:
    """E-2b ROUTE-ARM REFUSAL GUARD: a unit-3 push with the feeder input mux
    NOT on code 4 would stream the row into a dead mux — the RTL must refuse
    AT PUSH, LAYER_STATUS sticky + err_code 9 (NFEED_ROUTE), all units idle,
    W1C clears. On pre-E1 RTL unit 3 is the JOB-error default (code 1), so
    the checked code read goes RED there — half of the red/green evidence."""
    s = LayerScript(D_TILE)
    s.emit("// E-2b refusal probe: unit-3 push with levels 0 (route unarmed)"
           " — EXPECT sticky + err_code 9, no unit busy, then W1C.")
    _prologue(s)
    s.lctl()                                       # all levels 0
    nfjob(s, D_TILE, 0, armed_word=0, raw=True)    # deliberately illegal
    s.lpoll(LST_ERR, LST_ERR)
    s.lstat(0x1F00 | LST_NFEED | LST_LDQ | LST_SWG | LST_RES,
            (LERR_NFEED_ROUTE << 9) | LST_ERR)
    s.lstatw(LST_ERR)                              # W1C (code holds by design)
    s.lstat(LST_ERR, 0x0)
    s.emit("DONE")
    x = _translate(s, note=f"E-LANE {name}: route-arm refusal guard")
    return _emit(x, out_dir, name, dict(
        stage="nfeed_probe_route", kind="E-2b route-arm refusal guard",
        expect=f"LAYER_STATUS sticky + err_code {LERR_NFEED_ROUTE} "
               f"(NFEED_ROUTE), then W1C"))


def build_geom_probe(*, out_dir: Path, name: str) -> dict:
    """E-1 GEOMETRY REFUSAL GUARD (the R3 resid_probe shape, egress side):
    with the route ARMED, an egress whose window footprint exceeds the row
    RAM (base=1, cols=128 -> 1024+128 > DM_MAX=128 on the toy twin) must be
    refused AT ACCEPT by apex_residual — frame class, sticky + err_code 3 —
    and consume the push (no wedge, nf_busy falls). Pre-E1 RTL errors the
    push itself with code 1, so the checked code read goes RED there."""
    s = LayerScript(D_TILE)
    s.emit("// E-1 refusal probe: unit-3 push base=1 cols=128 (footprint "
           "1152 > DM_MAX=128) with the route armed — EXPECT sticky + "
           "err_code 3 (frame class), nf_busy clear, then W1C. NO feeder "
           "job is armed: fq_busy is a TERM of nf_busy while code 4 is "
           "armed (by design — the poll covers the whole path), so an "
           "armed-but-unfed feeder would hold LAYER_STATUS[5] high and "
           "the refusal is at the RESIDUAL's accept, feeder irrelevant.")
    _prologue(s)
    s.emit("// wait out the loader phase's drain BEFORE arming code 4 — "
           "fq_busy is an nf_busy term while code 4 is armed, and the "
           "loader's feeder tail-off is slow (measured ~39k cycles)")
    s.csrp(g3.CSR["STATUS"], 0x1, 0x1)
    s.csrw(L_CTRL, LCTL_NFEED)
    nfjob(s, D_TILE, 1, armed_word=LCTL_NFEED, raw=True)
    s.lpoll(LST_ERR, LST_ERR)
    s.emit("// the refused push must be CONSUMED, path idle (bounded poll — "
           "a wedge here is a loud rc, never a hang)")
    s.lpoll(LST_NFEED, 0x0)
    s.lstat(0x1F00 | LST_NFEED | LST_LDQ | LST_SWG | LST_RES,
            (LERR_FRAME << 9) | LST_ERR)
    s.lstatw(LST_ERR)
    s.lstat(LST_ERR, 0x0)
    s.csrw(L_CTRL, 0)
    s.emit("DONE")
    x = _translate(s, note=f"E-LANE {name}: egress geometry refusal guard")
    return _emit(x, out_dir, name, dict(
        stage="nfeed_probe_geom", kind="E-1 geometry refusal guard",
        expect=f"LAYER_STATUS sticky + err_code {LERR_FRAME} (frame class), "
               f"then W1C"))


# ═══════════════ grading (golden the only arbiter) ═════════════════════════

def grade_nfeed(caps, step: dict) -> dict:
    """Grade the nfeed captures against the golden references in `step`
    (all of which toy_step computed with apex_golden functions only)."""
    out = {}
    # (a) the tile-produced r1 row (ERD readback, post-feed)
    g_r1 = grade_resid(caps, step["r1_bits"])
    out["r1"] = g_r1
    # (b) the norm-output C-1 codes (identity GEMM lanes, in order)
    acc = cd.ro_lanes_to_i32(caps, strict=False)
    got_codes = acc[:D_TILE].astype(np.int64)
    codes_ok = (got_codes.size == D_TILE
                and np.array_equal(got_codes, step["h2_codes"]))
    n_diff = int(np.sum(got_codes[:min(D_TILE, got_codes.size)]
                        != step["h2_codes"][:min(D_TILE, got_codes.size)]))
    out["h2_codes"] = {"equal": bool(codes_ok), "got": int(got_codes.size),
                       "n_diff": n_diff}
    # (c) the two fs scales, in capture order: r1's C-1 scale (the value
    # golden discards), then h2's C-1 scale
    fs = [t for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    fs_bits = [int(t["bits"]) for t in fs]
    scales_ok = (len(fs_bits) == 2
                 and fs_bits[0] == step["r1_scale"]
                 and fs_bits[1] == step["h2_scale"])
    out["scales"] = {"equal": bool(scales_ok), "got": fs_bits,
                     "want": [step["r1_scale"], step["h2_scale"]]}
    out["equal"] = bool(g_r1["equal"] and codes_ok and scales_ok)
    out["n"] = int(2 * D_TILE + 2)
    return out


# ── discriminator-verdict hygiene (2026-08-04) ─────────────────────────────
# MEASURED failure mode (build/f2_elane_walk report, 2026-08-04): the walk
# suite was pointed at a D=64 twin, every stage aborted in phase A with ZERO
# captures — and the old verdict fallbacks then reported x_ulp/cap_* as
# caught=True (a vacuous RED: nothing ran) and gamma as caught=False (its
# empty-capture fallback was {"equal": True}). A perturbation verdict is
# EVIDENCE only when (a) the UNPERTURBED control graded green on the same
# binary — otherwise every run fails alike and "red" proves nothing — and
# (b) the perturbed run actually executed and produced gradeable captures.
# The helpers below make both preconditions explicit. A discriminator that
# cannot prove its bite must FAIL the gate, never pass it.

def disc_ran(r) -> bool:
    """The perturbed run is gradeable: executor ok AND captures present."""
    return bool(r["ok"]) and bool(r["captures"])


def stalled_at_poll(log: str, addr: int) -> bool:
    """True iff the executor aborted on a POLL of BAR0 `addr` — the failure-
    POINT receipt for stall-shaped discriminators (walk_off). Both executors
    print the address on the stall line: f2sim '[1070]' (sim_main.cpp:537),
    f2_host_run '@0x1070' (:190). Any OTHER failure (phase-A geometry check,
    loader stall, AXI wedge) does not match — 'it failed somewhere' is not
    'it failed at the discriminated seam'."""
    a = f"{addr:04x}"
    return any("poll stall" in ln and (f"[{a}]" in ln or f"0x{a}" in ln)
               for ln in (log or "").splitlines())


DISC_WITHHELD = {"caught": False,
                 "why": "control stage not green on this binary — a "
                        "perturbation verdict would be vacuous (nothing "
                        "biteable ran); fix the run before reading discs"}


# ═══════════════ selftest / smoke ══════════════════════════════════════════

def _selftest() -> int:
    import tempfile
    ok = True

    def chk(name, cond, note=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" +
              (f" — {note}" if note and not cond else ""))

    print("[1] toy references are golden-derived and self-consistent")
    st = toy_step()
    chk("r1 = f16(X + o8*comp) round-trips the fp16 grid",
        np.array_equal(
            st["r1_bits"],
            f64_to_f16_bits(f16_bits_to_f64(st["r1_bits"]))))
    chk("norm consumes CODES (scale-invariance): rmsnorm_fx(r1_codes) "
        "matches the stored h2", np.array_equal(
            np.asarray(at.rmsnorm_fx([int(c) for c in st["r1_codes"]],
                                     [int(v) for v in st["gamma2"]])[0],
                       dtype=np.int64), st["h2"]))
    chk("determinism", int(toy_step()["h2_scale"]) == st["h2_scale"])

    print("[2] emission-side legality mirrors the consumer rule")
    s = LayerScript(D_TILE)
    for bad, kw in [("route unarmed", dict(armed_word=0)),
                    ("cols not a row multiple",
                     dict(armed_word=LCTL_NFEED)),
                    ("footprint past DM_MAX",
                     dict(armed_word=LCTL_NFEED))]:
        try:
            if bad == "cols not a row multiple":
                nfjob(s, 96, 0, **kw)
            elif bad == "footprint past DM_MAX":
                nfjob(s, D_TILE, 1, **kw)
            else:
                nfjob(s, D_TILE, 0, **kw)
            chk(f"refuses: {bad}", False)
        except AssertionError:
            chk(f"refuses: {bad}", True)

    print("[3] the programs build and pass the zero-expectation audit")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        m1 = build_nfeed_stage(st, out_dir=out, name="nfeed")
        m2 = build_route_probe(out_dir=out, name="nfeed_probe_route")
        m3 = build_geom_probe(out_dir=out, name="nfeed_probe_geom")
        for m in (m1, m2, m3):
            a = m["audit"]
            chk(f"{m['name']}: audit clean "
                f"(caps={a['caps']}, caps_with_e={a['caps_with_e']}, "
                f"violations={a['violations']})",
                a["caps_with_e"] == 0 and a["violations"] == 0)
        chk("nfeed captures: 2 fs + 128 RO lanes*? + 128 ERD",
            m1["caps"] == 2 + 16 * (MXE_N + 1) + D_TILE,
            f"got {m1['caps']}")

    print("[4] the grader discriminates on host-perturbed captures")
    # a fabricated all-correct capture set must pass; a moved code must fail
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
    # (fs cap 2 lands between per the program; order within sems is what
    #  the decoders key on, so grading tolerates this flat layout)
    cap("fs", st["h2_scale"])
    for b in st["r1_bits"]:
        cap("lrd", int(b))
    g = grade_nfeed(caps, st)
    chk("all-correct capture set grades equal", g["equal"], json.dumps(g))
    bad = [dict(c) for c in caps]
    for c in bad:
        if c["sem"] == "ro_w3":
            c["value"] = (c["value"] + 1) & 0xFFFFFFFF
            break
    chk("a moved code goes red", not grade_nfeed(bad, st)["equal"])
    bad2 = [dict(c) for c in caps]
    for c in bad2:
        if c["sem"] == "lrd":
            c["value"] ^= 1
            break
    chk("a moved r1 bit goes red", not grade_nfeed(bad2, st)["equal"])

    print("[5] discriminator-verdict hygiene (the 2026-08-04 wrong-twin "
          "failure mode stays fixed)")
    chk("disc_ran refuses an ok run with ZERO captures (the vacuous-RED "
        "shape)", not disc_ran({"ok": True, "captures": []}))
    chk("disc_ran refuses a failed run", not disc_ran({"ok": False,
                                                       "captures": [1]}))
    chk("disc_ran accepts ok + captures", disc_ran({"ok": True,
                                                    "captures": [1]}))
    chk("stalled_at_poll matches the sim executor's stall line",
        stalled_at_poll("poll stall L983: [1070] last 00080000 want "
                        "00000080/00000080", 0x1070))
    chk("stalled_at_poll matches the hw executor's stall line",
        stalled_at_poll("poll stall L983 @0x1070: last 0x00080000 want "
                        "0x80/0x80", 0x1070))
    chk("a phase-A checked-read FAIL is NOT a route-arm stall",
        not stalled_at_poll("FAIL L6: [100c] got 00000040 want 00000080 "
                            "(mask ffffffff)", 0x1070))
    chk("a pw (pacing-write) stall is NOT a poll stall",
        not stalled_at_poll("pw stall L243", 0x1070))
    chk("a poll stall at a DIFFERENT address is not this seam",
        not stalled_at_poll("poll stall L10: [1068] last 00000001 want "
                            "00000000/00000001", 0x1070))
    chk("the withheld verdict can never read as caught",
        DISC_WITHHELD["caught"] is False)

    print(f"ELANE SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_all(out_dir: Path, step: dict) -> list[dict]:
    return [
        build_nfeed_stage(step, out_dir=out_dir, name="nfeed"),
        build_route_probe(out_dir=out_dir, name="nfeed_probe_route"),
        build_geom_probe(out_dir=out_dir, name="nfeed_probe_geom"),
    ]


def smoke(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = toy_step(args.seed)
    plan = build_all(out, step)

    def runner(pth, capname):
        return glo._run([pth], binary=args.binary, tile_div=args.tile_div,
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
        if man["stage"] == "nfeed":
            try:
                row["grade"] = grade_nfeed(r["captures"], step)
            except Exception as e:                             # noqa: BLE001
                row["grade"] = {"equal": False,
                                "error": f"{type(e).__name__}: {e}"}
        else:
            # the probes' whole content is their CHECKED sticky/code reads —
            # the executor enforced them; `ok` IS the verdict (probe idiom)
            row["grade"] = {"equal": bool(r["ok"]),
                            "note": "checked refusal guard"}
        rows.append(row)
        eprint(f"  [{man['name']}] rc={r['rc']} ok={r['ok']} "
               f"caps={len(r['captures'])} {wall:.1f}s "
               f"grade={row['grade'].get('equal')}")

    disc = {}
    if not args.no_discriminate:
        # a perturbation verdict is only evidence against a GREEN control on
        # the SAME binary (see the hygiene note above grade_nfeed's helpers)
        base = next(x for x in rows if x["stage"] == "nfeed")
        if not (base["ok"] and base["grade"].get("equal")):
            eprint("[disc] WITHHELD: the unperturbed nfeed control is not "
                   "green on this binary — perturbation runs would prove "
                   "nothing (the 2026-08-04 wrong-twin failure mode)")
            disc["withheld"] = dict(DISC_WITHHELD)
        else:
            eprint("[disc] running the perturbations (each MUST go red)")
            # (1) one X element moved 1 fp16 ulp: r1 AND the norm chain move
            m = build_nfeed_stage(step, out_dir=out, name="disc_x_ulp",
                                  perturb="x")
            r = runner(m["path"], m["name"])
            ran = disc_ran(r)
            gx = grade_nfeed(r["captures"], step) if ran else {}
            disc["x_ulp"] = {"caught": bool(ran and not gx["equal"]),
                             "ran": ran,
                             "r1_equal": gx.get("r1", {}).get("equal"),
                             "codes_equal": gx.get("h2_codes", {}).get("equal")}
            # (2) one gamma word moved: only the norm output moves (r1 must
            #     still grade equal — localizes the perturbation to the norm)
            m2 = build_nfeed_stage(step, out_dir=out, name="disc_gamma",
                                   perturb="gamma")
            r2 = runner(m2["path"], m2["name"])
            ran2 = disc_ran(r2)
            gg = grade_nfeed(r2["captures"], step) if ran2 else {}
            disc["gamma"] = {"caught": bool(ran2 and not gg["equal"]
                                            and gg["r1"].get("equal")),
                             "ran": ran2,
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

    print(f"\nE-LANE nfeed @ tile_div={args.tile_div} "
          f"(binary={args.binary or 'bridge default'})")
    for r in rows:
        print(f"  {r['name']:<20} rc={r['rc']} ok={r['ok']} "
              f"caps={r['caps']:<4} grade={r['grade'].get('equal')}")
    for k, v in disc.items():
        print(f"  disc {k:<15} caught={v.get('caught')}")
    ok = all(r["ok"] and r["grade"].get("equal") for r in rows) \
        and all(v.get("caught") for v in disc.values())
    print(f"ELANE SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--binary", default=None,
                    help="f2sim twin (default: the bridge's resolution)")
    ap.add_argument("--tile-div", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--no-discriminate", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.build:
        step = toy_step(args.seed)
        plan = build_all(Path(args.out), step)
        for m in plan:
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
