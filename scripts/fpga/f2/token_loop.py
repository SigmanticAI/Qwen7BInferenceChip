#!/usr/bin/env python3
# token_loop.py — THE TOKEN LOOP: multi-token greedy decode of Qwen2.5-0.5B
# with a PLUGGABLE per-layer engine, graded token-for-token against the
# host-golden pipeline. This is the driver behind the project's first honest
# tokens/second measurement.
#
#   python3 scripts/fpga/f2/token_loop.py selftest                # offline
#   python3 scripts/fpga/f2/token_loop.py run --engine host-golden \
#       --tokens 3                                                # host only
#   python3 scripts/fpga/f2/token_loop.py run --engine sim-walked \
#       --tokens 1 --binary verif/f2sim/obj_tokenloop/f2sim       # Verilator
#   python3 scripts/fpga/f2/token_loop.py run --engine sim-walked \
#       --prompt <16 ids> --tokens 1 --probe-steps 15             # deep-T probe
#
# ══ WHAT THIS IS ═══════════════════════════════════════════════════════════
# walk_layers_05b.py proved the walked E-6 chain for all 24 layers of ONE
# decode step. prompt_offload.decode / run_tinynpu.run_generate own the
# token semantics (greedy, EOS, self-inclusive q_pos composition). This file
# is the loop ABOVE both: per GENERATED token,
#
#   (1) the host computes the embedding of the current token id;
#   (2) every layer of that decode step runs through a pluggable per-layer
#       ENGINE (see the Engine contract below) — "host-golden" is the pure
#       golden reference, "sim-walked" additionally drives each layer's
#       walked {FPROJ, QKV, OPROJ, RES1} chain (wfl.MASK_B, the E-6 chain
#       walk_layers_05b runs) on a Verilator twin and grades every walked
#       value bit-exact against golden;
#   (3) the host computes final-norm + lm_head + greedy argmax;
#   (4) the token is appended and the loop repeats, --tokens times.
#
# KV STATE is run_tinynpu.Session's own: the per-layer input-row history
# (golden recomputes K/V from the row history each step — the committed
# 0.5B composition; prompt05b/prompt_offload carry state the same way).
#
# ══ THE GRADE (bit-exact discipline; golden is the ONLY arbiter) ═══════════
#   * per layer, per step: the engine's returned layer-output row must be
#     BIT-IDENTICAL (f16 bits) to golden's r2 for that layer — REFUSED
#     loudly otherwise, before any token is sampled from it;
#   * per token: the sampled token id under ANY engine must equal the
#     host-golden engine's token id. For engine != host-golden the harness
#     FIRST runs a pure host-golden reference decode (fresh Session, same
#     ids) and compares at every token, refusing loudly at the first
#     divergence. For engine == host-golden the same two-pass structure is
#     the run-to-run consistency check (ids AND logit bits must agree);
#   * sim-walked, per layer: 144 QKV RO jobs (raw INT32 accumulators) and
#     the 896 r1 f16 caps are graded against apex_golden functions on the
#     same operands, computed AFTER the run (wfp.grade / wfl.grade_r1 —
#     reused, not reimplemented). Any mismatch REFUSES the whole run.
#
# ══ THE INTER-LAYER / INTER-TOKEN SEAM, STATED HONESTLY ════════════════════
# Exactly walk_layers_05b's seam, extended across tokens: the walked step
# set is {QKV, OPROJ, RES1}; NORM1, attention, NORM2, FFN, RES2, the
# embedding, the final norm, the lm_head and the argmax are the HOST's
# (golden's own arithmetic). The state carried between layers and between
# tokens is golden's composition — the continuation of walked values that
# were verified bit-exact first, not a replacement for the unwalked steps.
# The report therefore carries a measured MAC census (prompt05b.MacCensus —
# reused) and labels exactly which fraction of each token's arithmetic ran
# on the walked path. Nothing in this file makes the unwalked steps
# disappear; the JSON says so in as many words.
#
# --ffn (walked engines: sim-walked AND, since 2026-08-13, hw-walked) adds
# a SECOND walked window per (step, layer): the FFN→DOWN slice of the
# rung-1/2/3 machinery — see the FFN block below for the slice bound and
# the grade semantics. The token composition is UNCHANGED: the FFN stays
# host-side in the carry; the window is a graded probe on live operands,
# and the MAC split counts only its graded MACs.
#
# ══ TIMING, LABELLED ═══════════════════════════════════════════════════════
# host-golden: tokens/s is a MEASUREMENT of the golden pipeline on this
#   host (wall clock). It is NOT a hardware number.
# sim-walked: the Verilator wall is a property of the simulator, never a
#   hardware claim. What IS carried per layer is the CYCMARK-measured walk
#   window in tile cycles; the per-token aggregate is handed to
#   walk_layers_05b.predict (reused), whose output is labelled PREDICTION
#   at the flown 15.625 MHz / analyzed 62.5 MHz clocks and covers the
#   walked steps ONLY.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif/top/l3"),
           str(REPO / "verif/seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tinynpu as rt                                       # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

import seq_walker_fmt as fmt                                   # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402
import tile_geom as tg                                         # noqa: E402
import fuel_proj_05b as fp                                     # noqa: E402
import walk_fuel_proj as wfp                                   # noqa: E402
import walk_fuel_layer as wfl                                  # noqa: E402
import walk_layers_05b as wl                                   # noqa: E402
import gen_layer_ops as glo                                    # noqa: E402
import gen_l3_vectors as g3                                    # noqa: E402
import cap_decode as cd                                        # noqa: E402
import trace_to_regops as t2r                                  # noqa: E402
import emit_cache as ec                                        # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import compute as cpg                         # noqa: E402
from apex_golden import transformer as tfm                     # noqa: E402
from prompt05b import MacCensus                                # noqa: E402

DEFAULT_WEIGHTS = REPO / "build/s8_weights/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_IMAGE = REPO / "build/ddr_weights_05b_24L"
DEFAULT_WORK = REPO / "build/token_loop"
# ══ THE SESSION BUDGETS (prompt + new tokens), SPLIT 2026-08-11 ═════════════
# T_SIM_MAX is a WALL-TIME budget and nothing else: a FULL sim-walked decode
# is 24 Verilator flights per generated token (~10 min/chain on the dev
# host), so unprobed sim sessions stay short. It is NOT a hardware or RTL
# limit — deep-T sim evidence goes through --probe-steps, which walks only
# the named (step, layer) chains and is therefore fenced by T_HW_MAX below.
T_SIM_MAX = 8                    # sim-walked FULL-decode budget (wall time)
# T_HW_MAX is the tile's WALK-ENVELOPE budget: the session range inside
# which every walked step this loop composes — and the attention lane it is
# scheduled to compose (B1/B1b) — is descriptor-legal on the RTL. Each
# ceiling is cited so raising one is a deliberate RTL/host change, never a
# constant edit. As of comp/prompt-b-c @ 8740a90:
#   64  — MXE job m-height, the BINDING ceiling. mxe_ctrl refuses m_dim >
#         M_TILE_MAX = 64 (rtl/mxe/mxe_cfg_pkg.sv:22), and the walker's
#         S2_CHECK refuses any EN_QKV descriptor with t_rows > WALK2_M_MXE
#         up front (rtl/seq/seq_layer_walker2.sv:1423-1428, the D-029
#         erratum clause: K/V projection jobs carry m = t_rows). Today's
#         E-6 decode chain pins t_rows=1/pos_m=0 (walk_fuel_layer.
#         walk_descriptor's fmt.pack_step(1, 0)), so the tile never sees
#         the session T directly — 64 is the envelope every walked shape
#         (decode m=1 today, prefill-class m=t_rows next) stays legal in.
#   128 — walker descriptor legality: t_rows <= WALK_T_MAX = 128 and
#         pos_m < WALK_T_MAX (rtl/seq/seq_walker_pkg.sv:44 and
#         walk_desc2_check's clauses at :692-696).
#   128 — walker scale cache: the L3 store map is K records at [0,T), V at
#         [T,2T), so 2T <= WALK_SC_ENT = 256 entries (seq_walker_pkg.sv:49;
#         sc_mem/sc_val in rtl/seq/seq_walker_comp.sv:131,140 — one
#         UNMODIFIED 256-entry instance per KV engine,
#         rtl/top/glue/apex_wcomp_bank.sv:174).
#   128 — KVQ CQ-8 store: the same 2T record map into KVQ_DEPTH = 256
#         records/engine on the I-B CL (rtl/top/glue/apex_kvq_gqa_bank.sv:
#         12,53; rtl/top/apex_top.sv:165).
#   384 — the host session range rt.T_SESSION_MAX (run_tinynpu.py:79),
#         enforced separately below for EVERY engine.
T_HW_MAX = 64                    # hw-walked session budget = min(above)
WALKED_STEPS = "{FPROJ, QKV, OPROJ, RES1} (wfl.MASK_B — the E-6 chain)"
HOST_STEPS = ("NORM1, attention (score/softmax/PV), NORM2, FFN "
              "(gate/up/swiglu/down), RES2, embedding, final norm, "
              "lm_head, argmax")

# ══ the walked FFN→DOWN slice window (--ffn, walked engines) ════════════════
# The rung-1/2/3 machinery (walk_ffn_toy.py; seq_layer_walker2 fence-2 +
# the 2026-08-08 FFN-interleave row model) applied per (decode step, layer)
# with LIVE subjects: mask exactly {FPROJ, FFN, DOWN, RES2}, the chunked
# PC_USWI..PC_WUJ loop fuel-fetching the layer's own Wg/Wu chunk slabs,
# on-tile swiglu + per-chunk S2_CS* product staging (RT_FFN route), then the
# DOWN projection + o8 epilogue + RES2 onto the layer's own r1 — ONE kick,
# host-silent window (the swiglu window is never tile_idle-fenced by the
# host: the walker arms the consumer at PC_USWI, before the up phase).
#
# THE SLICE, STATED — two independent bounds, both measured on this tree:
#   (1) stage rows: the walker stages the FFN x family (h2, d_model/64 = 14
#       rows) plus ONE product row per chunk in the same act bank, and
#       S2_CHECK bounds fp_top <= STAGE_ROWS = 31 (seq_layer_walker2.sv;
#       the full-width 0.5B DOWN — 76 rows — is the RTL's own REFUSED
#       class, E-7b note). 31 - 14 = 17 chunks fit.
#   (2) the fs capture FIFO: each chunk's S2_CS* stage pushes ONE fp16 row
#       scale into the mailbox fs FIFO (apex_f2_mailbox.sv MB_INFIFO(fs),
#       DEPTH = 16), and a host-silent window drains NOTHING — at chunk 17
#       the push backpressures squant, the product stream stalls, and
#       S2_CSW parks on tile_idle forever (measured on the W2DBG twin:
#       16 clean chunk loops, the 17th parks). 16 chunks fit.
# One descriptor therefore holds min(17, 16) = 16 chunks = 1024 of the 4864
# d_ffn columns. --ffn walks ONE maximal slice per layer — real weights,
# real h2, ~2.75M graded MACs/layer — and the report never calls it more.
#
# THE GRADE, STATED: like the walked QKV accumulators and r1, the window is
# graded against golden's OWN functions applied to the SAME framed operands
# (h2 C-1 frames staged amax-127-identity; composites C2 f16-graded, gate/up
# folding golden's own row scale s_h2; the DOWN composite folding the
# uniform-contract product scale = the max per-chunk C-1 scale). The walked
# r2 is the probe's own composed value; the token stream continues from
# golden's fx exactly as without --ffn.
FFN_FS_FIFO_DEPTH = 16          # apex_f2_mailbox.sv:153 (bound 2 above)
FFN_CHUNKS_MAX = min(31 - (wfl.D_MODEL // wfl.D_TILE),   # 17: stage rows
                     FFN_FS_FIFO_DEPTH)                  # 16: fs capture
FFN_TENS = ("Wg", "Wu", "Wd")
FFN_MASK_DOWN = ((1 << fmt.EN_FPROJ) | (1 << fmt.EN_FFN)
                 | (1 << fmt.EN_DOWN) | (1 << fmt.EN_RES2))
FFN_REPOINT_WORDS = tuple(sorted(
    [fmt.W_TENS0 + fmt.TENS_WG, fmt.W_TENS0 + fmt.TENS_WU,
     fmt.W_TENS0 + fmt.TENS_WD,
     fmt.W_JC0 + fmt.JC_GATE, fmt.W_JC0 + fmt.JC_UP,
     fmt.W_JC0 + fmt.JC_DOWN,
     fmt.W_RQ0 + wfl.N_HEADS + 1]))
FFN_SEAM_NOTE = (
    "PROBE semantics: the FFN→DOWN window walks d_ffn[0:S] of the model's "
    "d_ffn (the stage-bank bound) on the layer's real Wg/Wu/Wd and live h2, "
    "graded bit-exact against golden's own functions on the framed "
    "operands (per-chunk product scales; composites C2 f16-graded). It "
    "does NOT replace the host FFN in the token composition — fx carries "
    "the state, exactly as the E-6 chain's seam already states.")


def ffn_walked_steps(s_walk: int, d_ffn: int) -> str:
    return (WALKED_STEPS + f" + per-layer FFN→DOWN slice {{gate, up, "
            f"swiglu, product-stage, DOWN, RES2}} over d_ffn[0:{s_walk}] "
            f"of {d_ffn}")


# ── the hw arm of the FFN lane (2026-08-13) ────────────────────────────────
# The FFN slice image is a SEPARATE DDR artifact — the walker's chunk
# arithmetic (base + chunk*d_model beats64) cannot address Wd inside the
# MAIN image's k_job=1984 n-major/k-minor layout (the FfnLane substrate
# note), so on silicon the slice image must be RESIDENT ALONGSIDE the main
# image. hw mode therefore builds it REBASED at FFN_HW_DDR_BASE via
# make_weight_image.build's base_bytes arm: every region base, every
# manifest/regions base_64B word AND the bin itself carry the offset, so
# the UNTOUCHED f2_ddr_load loads and verifies it at the rebased physical
# addresses, and the descriptor bases come straight from that manifest —
# no in-flight address math anywhere. The operator loads it like the main
# image and attests with --ffn-ddr-attested (recorded, REFUSED if absent).
# 512 MiB clears the 24L main image (341.6 MiB, fenced at prepare()) and
# keeps every rebased base_64B far inside the 30-bit fuel_req base field
# (make_weight_image.build's own assert stays live on the rebased bases).
FFN_HW_DDR_BASE = 0x2000_0000            # 512 MiB, 4 KiB aligned


def check_ffn_composition(engine: str, mask: str, probe_steps,
                          ffn: bool) -> None:
    """The --ffn composition fences, in one testable place (run() calls
    this; the selftest exercises every arm offline). hw-walked ARMED
    2026-08-13: the same window/grade machinery flies through the hw
    executor seam — see FfnLane.run_one's hw branch."""
    if not ffn:
        return
    if engine not in ("sim-walked", "hw-walked"):
        raise SystemExit("REFUSE: --ffn needs a walked engine (sim-walked "
                         "or hw-walked) — there is no tile to fly the "
                         "FFN→DOWN window on under " + repr(engine))
    if mask == "e7":
        raise SystemExit("REFUSE: --ffn composes with --mask b only — the "
                         "e7+ffn pairing has never been flown or measured "
                         "on this tree; run the two machineries separately")
    if probe_steps is not None:
        raise SystemExit("REFUSE: --ffn with --probe-steps is not a "
                         "composed mode — the --ffn gate requires a walked "
                         "FFN window at EVERY layer of every sampled step, "
                         "which probe skipping contradicts by design")


# ── the E-7 composition at the 0.5B geometry (--mask e7) ───────────────────
# docs/design/E7_TOKEN_LOOP.md. TWO kicks per (step, layer): the one-kick
# walk_e7/walk_e8 shape does not fit the 31-row act bank at H=14 (q 14 +
# o-act 14 + y 14 = 42 > 31, the e6_fence_qkvattn_cap refusal), and the
# fetched-gamma NORM2 at d_model=896 is value-blocked (B-FEED-WIDTH: the
# C-1 feeder frames r1 with PER-FRAME scales, so the walked norm would not
# be golden's norm) — fenced, not degraded. MEASURED 2026-08-11 on the
# obj_tokenloop twin: a {FPROJ, QKV} walk with W_STEP.t_rows > 1 WEDGES at
# QKV job 123 (mid-Wk; t_rows=1 runs all 144, RQ words innocent — the
# fueled projection template is T-aware and starves on a T-row act family
# no 31-row bank can hold), so the projection kick keeps the FLOWN T-free
# MASK_B image and attention walks in its own kick at the live T.
# ── the walked STOREKV composition (E7-at-live-T rework, 2026-08-12) ────────
# E7_LIVE_T_DEFECT.md: the flown ATT builder selected the KVQ engine for its
# host staging by flipping L_CTRL[17:15] (l_kv_map) NINE times — the host
# branch of apex_top's `kv_eng_sel = walk_en_q ? wk_kv_eng_sel : l_kv_map_q`
# mux, a D-033-class synthesis-casualty suspect on the flying image (it sits
# at tile level, OUTSIDE the dont_touch bank instances). This mask adds
# EN_STOREKV so the WALKER runs its store step (seq_layer_walker2.sv
# S2_SKPAR..S2_SKB, :2000-2024): per engine g, idle-poll then WRITE_ADDR
# K@T-1 / V@2T-1 through the walker's OWN sk_g select (kv_eng_sel's
# in_store arm, :1302) — every walk-side KVQ address/health op rides the
# walker branch, and a non-idle/errored engine FAILS the walk loudly
# (WALK_ERR_SEQ, kvm_rdata[1] at :2003-2006).
#
# MEASURED MECHANISM LIMIT (the step-1 study, stated honestly): the walked
# STOREKV step moves NO data — it is address pacing for records the walk
# itself produces (the QKV→ser→squant seam, rt_kv_user), and a live-T walk
# cannot produce them (a t_rows>1 {FPROJ, QKV} image wedges — the header's
# measured finding). Nor can the host feed the seam mid-walk: walk_en_q
# holds EVERY host job port off (apex_top.sv:680-706 — fj/qj/dj/aj/wj/ds
# ready forced 0; the mailbox job-fire poll would park), and the CQ-8
# engine has no auto-advancing store address (kvq_engine KEYG = (TIER != 0)
# = 0, so each record needs its own host WRITE_ADDR). The record DATA
# therefore stays HOST-staged pre-walk under ONE quasi-static l_kv_map
# level per engine (the §3b carve, 2 select writes + restore vs the flown
# 9) — the one remaining host-branch dependence, disclosed in e7_fences;
# the l_kv_map RTL hardening (flop + dont_touch, the wc_snp_* pattern) is
# what makes it silicon-true on the next image.
MASK_E7A = ((1 << fmt.EN_STOREKV) | (1 << fmt.EN_SCORE)
            | (1 << fmt.EN_PV))
WALKED_STEPS_E7 = ("{STOREKV, SCORE, PV} @ live T + {FPROJ, QKV, OPROJ, "
                   "RES1} (wfl.MASK_B, the flown chain) — the E-7 "
                   "composition as two kicks/layer; "
                   "docs/design/E7_TOKEN_LOOP.md")
HOST_STEPS_E7 = ("NORM1, NORM2 (fetched-gamma FENCED at 896 — "
                 "B-FEED-WIDTH), FFN (gate/up/swiglu/down), RES2, "
                 "embedding, final norm, lm_head, argmax")
W_ST_FMTSUP, FMT_SUP_LAYER = 0xF000, 0x3
E7A_GO_MARK = "CYCMARK E7A-PREGO"
E7A_DONE_MARK = "CYCMARK E7A-WALKDONE"
# words an ATT descriptor may move between two layers of ONE step: the
# Wq/Wk/Wv/Wo bases (Wo written for uniformity, unused by this mask) + the
# 14 per-head PV requant pairs. Across a step boundary, additionally W_STEP.
ATT_REPOINT_WORDS = tuple(sorted(
    [fmt.W_TENS0 + t for t in (fmt.TENS_WQ, fmt.TENS_WK, fmt.TENS_WV,
                               fmt.TENS_WO)]
    + [fmt.W_RQ0 + h for h in range(wfl.N_HEADS)]))
# window writes that are DATA PLANE in the ATT kick: the four capture-FIFO
# advance registers (RO for the QKV/PV drains, FS/SS for the scale ladders,
# TD for the TIP beat) — everything else is a control write and REFUSES.
E7A_DATA_ADVANCES = frozenset(
    t2r.B_MB + t2r.CAP[w][3] for w in ("ro", "fs", "ss", "td"))


def e7_att_descriptor(bases: dict, head_rqs: list, T: int,
                      pos: int) -> list[int]:
    """The ATT kick's 64-word fmt=1 image: mask {STOREKV, SCORE, PV} at
    the 0.5B geometry, W_STEP = (live T, q_pos) — the S8 self-inclusive
    composition; the walked STOREKV step derives its per-engine WRITE_ADDR
    pair (K@T-1, V@2T-1) from t_rows, so W_STEP also addresses the store —
    and the 14 per-head PV requant pairs in RQ[0..13] (host-loaded
    calibration, the B1/D-028 documented scope). The tensor bases are
    written for image uniformity; a no-FPROJ mask never reads them.
    Envelope: no EN_QKV, so the D-029 t_rows<=M_TILE fence does not bind
    and no S2_CHECK cross-bit clause names STOREKV
    (seq_layer_walker2.sv:1423-1590 — swept, none fires on this mask)."""
    if not (1 <= T <= T_SIM_MAX):
        raise SystemExit(f"REFUSE: T={T} outside the CQ-8 window "
                         f"(1..{T_SIM_MAX}) — the walked score is fenced "
                         f"to the B1 scope")
    if len(head_rqs) != wfl.N_HEADS:
        raise SystemExit(f"REFUSE: {len(head_rqs)} head rq pairs, "
                         f"need {wfl.N_HEADS}")
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(wfl.HEAD_DIM)
    w[fmt.W_MODEL0] = fmt.pack_model0(wfl.D_MODEL, 0)
    w[fmt.W_MODEL1] = fmt.pack_model1(wfl.N_HEADS, wfl.N_KV_HEADS, 1)
    w[fmt.W_MASK] = fmt.pack_mask(MASK_E7A)
    w[fmt.W_STEP] = fmt.pack_step(T, pos)
    for t, name in ((fmt.TENS_WQ, "Wq"), (fmt.TENS_WK, "Wk"),
                    (fmt.TENS_WV, "Wv"), (fmt.TENS_WO, "Wo")):
        w[fmt.W_TENS0 + t] = bases[name]
    for h, rq in enumerate(head_rqs):
        scale, shift = int(rq[0]), int(rq[1])
        assert 1 <= scale <= 0xFFFF and 0 <= shift <= 31, \
            f"head {h} rq {rq} outside the RQ word"
        w[fmt.W_RQ0 + h] = fmt.pack_rq(scale, shift)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=wfl.HEAD_DIM) == fmt.ERR_NONE, \
        "illegal ATT walk image"
    return w


def _quiesce(s) -> None:
    """CSR + KVQ + LAYER idle — the qmh pre-verb fence (gen_qmh_case)."""
    s.csrp(g3.CSR["STATUS"], 0x1, 0x1)
    s.kvp(g3.KV["STATUS"], 0x1, 0x1)
    s.lpoll(0xF, 0x0)


def _build_att_at_d(sub: dict, desc: list[int], out_dir: Path, name: str):
    """The ATT kick program: live-T KV records staged in the walked-
    consumption form (one quasi-static engine level per store run — the
    MASK_E7A header carries the measured mechanism study), 14 rope-staged
    q rows through ONE sink arming, one fmt=1 {STOREKV, SCORE, PV} kick —
    the walker re-addresses/health-checks both engines through its OWN
    sk_g select before the heads walk — and the in-window data-plane
    drain in production order. No fuel (nothing fetches), no QKV family —
    the projections walk in the companion MASK_B kick with the flown
    T-free image (a {FPROJ, QKV} image at t_rows > 1 wedges: the header's
    measured finding)."""
    rows = wfl.D_MODEL // wfl.D_TILE
    T, pos = int(sub["T"]), int(sub["pos"])
    bpr8 = wfl.D_TILE // wfl.MXE_N
    s = glo.LayerScript(wfl.D_TILE)
    s.emit(f"// E-7 TOKEN-LOOP ATT KICK {name}: ONE fmt=1 walk, mask "
           f"{{STOREKV, SCORE, PV}} at the 0.5B geometry (H=14, GQA 2, "
           f"T={T}, q_pos={pos}). Host pre-walk: phase table, live K/V "
           f"records into both engines (ONE select level per store run), "
           f"14 rope-staged q rows at rows 0..{rows - 1}. In-walk: the "
           f"STOREKV step idle-polls + re-addresses each engine through "
           f"the walker's own sk_g select, then the heads. In-window "
           f"host ops: data-plane drains only.")
    g3.phase_a(s, tiers_used=(0,))
    g3.loader_phase(s)

    s.emit(f"// [1] q phase table for THIS step (pos={pos}) -> LAYER RAM "
           f"bank {glo.BANK_PHQ}")
    s.lptr(glo.BANK_PHQ, 0)
    s.ldata(sub["phase"])

    s.emit("// [2] the LIVE K/V records, walked-consumption form: engine g "
           "holds KV head g's 2T records — K row t at KVQ addr t, V row t "
           "at addr T+t (the T-relative layout: every V address moves when "
           "T grows, which is why the loop restages per step). ONE "
           "quasi-static l_kv_map level per store run (the §3b carve; the "
           "flown builder's 9-flip select choreography and its separate "
           "reset/OCC passes are GONE — E7_LIVE_T_DEFECT.md). CTRL=0x2 is "
           "the ENABLE write (bit1 — kvq_engine.sv:1024; engines reset "
           "disabled, :997; phase_a reaches engine 0 only).")
    _tap = s.tap
    s.tap = lambda *a, **k: None
    for g_i in range(wfl.N_KV_HEADS):
        _quiesce(s)
        s.emit(f"// KV group {g_i} -> GQA engine {g_i} ({T} K + {T} V rows)")
        s.csrw(glo.L_CTRL, (g_i & 7) << 15)
        s.kvw(g3.KV["CTRL"], 0x2)
        g3.store_kv_phase(s, sub["K16"][g_i], sub["V16"][g_i])
    s.tap = _tap

    s.emit(f"// [3] the 14 q rows through the gap-A sink: ONE arming (the "
           f"rising edge resets the s_q file pointer — gen_qmh_case), then "
           f"rows back to back; entry h <- head h's C-1 scale (fs cap), "
           f"act bank 1 row h <- head h's rotated codes")
    s.pmode(True)
    _quiesce(s)
    s.csrw(glo.L_CTRL, 0)
    s.route(rdst=1, asrc=0)
    s.lctl(rope_en=1, rope_bank=1, rope_pos=pos, fsrc_ext=3)
    for h in range(wfl.N_HEADS):
        s.fjob(1)
        q_pairs = [g3.decompose_f16(int(b))[:2]
                   for b in sub["q16"][h * wfl.D_TILE:(h + 1) * wfl.D_TILE]]
        g3.inject_jobs(s, q_pairs, 0, waddr=None)
        s.efs(0, 1)
        s.aj(0, wfl.ACT_BANK, 0, 1, s.BPR, h)
    s.emit("// disarm the sink (walker owns every level in the walk); the "
           "sink-leak occupancy check moved POST-WALK ([6b]) — no "
           "pre-walk l_kv_map readback loop")
    _quiesce(s)
    s.csrw(glo.L_CTRL, 0)
    s.route(rdst=0, asrc=0)

    s.emit("// [4] FMT_SUP + the 64-word image + the ONE kick")
    s.csrr(wfl.W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    s.csrw(wfl.W_DPTR, 0)
    for v in desc:
        s.csrw(wfl.W_DDATA, v)
    s.emit(f"// {E7A_GO_MARK}")
    s.csrw(wfl.W_CTRL, 0x3)

    s.emit(f"// [5] in-window data-plane drain, production order, per "
           f"head: {T} s_k fs, 1 s_c ss, "
           f"{bpr8} x ({T} s_v fs + 1 o8 RO block), 1 TIP beat (block 0 — "
           f"single importance block at T <= {T_SIM_MAX})")
    for h in range(wfl.N_HEADS):
        for t in range(T):
            s.efs(0, 1 if t == T - 1 else 0)
        s.ess(0, 1)
        for _j in range(bpr8):
            for t in range(T):
                s.efs(0, 1 if t == T - 1 else 0)
            s.ero([0] * wfl.MXE_N, 1)
        s.emit("ETIP", "0")

    s.emit("// [6] the walk retired, and it retired CLEAN — the ESTK/ECODE "
           "read below is ALSO the walked engine-health verdict: the "
           "STOREKV step idle-polled BOTH engines through the walker's "
           "sk_g select and errors the walk (WALK_ERR_SEQ) on a sick one")
    s.csrp(wfl.W_STATUS, wfl.W_ST_BUSY, 0x0)
    s.emit(f"// {E7A_DONE_MARK}")
    s.csrr(wfl.W_STATUS, wfl.W_ST_ESTK | wfl.W_ST_ECODE, 0x0)
    s.csrw(wfl.W_CTRL, 0x0)
    s.emit(f"// [6b] post-walk occupancy: engine 0 (L_CTRL=0) holds EXACTLY "
           f"its 2T={2 * T} records — the q-sink did not leak into the KVQ "
           f"and the walk added nothing. ENGINE 0 ONLY, on purpose: a host "
           f"per-engine OCC readback needs the l_kv_map CSR-window select — "
           f"the D-033-class suspect branch (E7_LIVE_T_DEFECT.md) — and a "
           f"folded select would alias engine 0 and pass WRONGLY; there is "
           f"no walked-mode readback (walk_en muxes the host KV window off, "
           f"apex_top.sv:734-739). Engine 1's write-path health is the "
           f"walked STOREKV's own idle/err poll, verdicted at [6].")
    s.kvp(g3.KV["STATUS"], 0x1, 0x1)
    s.kvp(g3.KV["OCC"], 0xFFFFFFFF, 2 * T)
    s.lpoll(glo.LST_LDQ | glo.LST_SWG | glo.LST_RES | glo.LST_ROPE, 0x0)
    s.lstat(glo.LST_ERR, 0x0)
    s.pmode(False)
    g3.final_phase(s, 8)

    x = glo._translate(s, note=f"E-7 token-loop ATT kick: {name}")
    man = glo._emit(x, out_dir, name, dict(
        stage="walk_e7_att", d=wfl.D_TILE, T=T, pos=pos,
        walk_mask=f"{MASK_E7A:#06x}", act_rows=rows))
    wfl.tg.retarget_info_tier(man["path"], wfl.tg.IMG_05B)
    return man


def build_att_program(sub: dict, desc: list[int], out_dir: Path, name: str):
    with wfl.tg.at_d(wfl.D_TILE):
        return _build_att_at_d(sub, desc, out_dir, name)


def att_silence(path: Path) -> dict:
    """The ATT kick's window discipline: between WALK_GO and the done-poll
    the only WRITES are capture-FIFO advances (RO + FS + SS + TD — the
    disclosed data plane of a walked-attention drain). wfp.silence_predicate
    verbatim, with the advance set widened for the attention FIFOs."""
    ops = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    go = [i for i, o in enumerate(ops)
          if o.get("op") == "w" and o.get("a") == (0x1000 | wfl.W_CTRL)
          and o.get("d") == 0x3]
    if len(go) != 1:
        raise SystemExit(f"REFUSE: {len(go)} WALK_GO writes in {path}")
    end = [i for i, o in enumerate(ops)
           if i > go[0] and o.get("op") == "poll"
           and o.get("a") == (0x1000 | wfl.W_STATUS)]
    if not end:
        raise SystemExit("REFUSE: no walk-done poll after WALK_GO")
    window = ops[go[0] + 1:end[0]]
    writes = [o for o in window if o.get("op") in ("w", "pw")]
    bad = [o for o in writes if o.get("a") not in E7A_DATA_ADVANCES]
    return {"window_ops": len(window), "writes": len(writes),
            "fifo_advances": len(writes) - len(bad),
            "control_writes": len(bad), "silent": not bad,
            "offenders": bad[:4]}


def _count_caps(path: Path) -> int:
    n = 0
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and json.loads(ln).get("op") == "cap":
                n += 1
    return n


def att_cap_layout(T: int) -> dict:
    """The ATT file's capture ledger, from the drain arithmetic (§5 of the
    design doc) — used by the grade to slice the concatenated stream and by
    the selftest to pin the arithmetic."""
    bpr8 = wfl.D_TILE // wfl.MXE_N
    n_fs = wfl.N_HEADS + wfl.N_HEADS * (T + bpr8 * T)   # q scales + ladders
    n_ss = wfl.N_HEADS
    n_ro = wfl.N_HEADS * bpr8 * (wfl.MXE_N + 1)         # the PV o8 blocks
    return {"fs": n_fs, "ss": n_ss, "ro": n_ro,
            "total": n_fs + n_ss + n_ro}


def predict_e7(rows: list[dict], n_layers: int) -> dict:
    """The e7 measured sums + the labelled walked-window prediction —
    wl.predict's shape with the e7 covers string (its 256-job basis and
    'QKV, OPROJ, RES1 ONLY' label do not describe an attention-bearing
    window, so the numbers are derived here from the combined windows)."""
    meas = [r for r in rows if r.get("cycles", {}).get("walk_tile_cycles")]
    out = {"measured": {}, "prediction": {}}
    if not meas:
        out["measured"]["note"] = ("no CYCMARK stamps — walls only")
        out["measured"]["wall_s_per_layer"] = [r["wall_s"] for r in rows]
        return out
    tc = [r["cycles"]["walk_tile_cycles"] for r in meas]
    walls = [r["wall_s"] for r in meas]
    mean_tc = sum(tc) / len(tc)
    tok_tc = mean_tc * n_layers
    covers = ("24 x walked {QKV, STOREKV, SCORE, PV} + {OPROJ, RES1} (two "
              "kicks) — 7 of 13 step families; NOT a full token")
    out["measured"] = {
        "n_layers_measured": len(meas), "tile_div": wfl.TILE_DIV,
        "walk_tile_cycles": tc,
        "att_tile_cycles": [r["cycles"].get("att", {})
                            .get("walk_tile_cycles") for r in meas],
        "epi_tile_cycles": [r["cycles"].get("epi", {})
                            .get("walk_tile_cycles") for r in meas],
        "tile_cycles_mean": round(mean_tc, 1),
        "tile_cycles_min_max": [min(tc), max(tc)],
        "sim_wall_s_per_layer": walls,
        "sim_wall_s_total": round(sum(walls), 1),
        "conversion": "tile = shell / (2*TILE_DIV)  [cl_apex.sv:82]"}
    out["prediction"] = {
        "LABEL": "PREDICTION — sim-derived, hardware-unvalidated; the "
                 "walk windows ONLY (pre-walk staging/readback excluded). "
                 "NOT hardware measurements.",
        "walked_steps_only_per_token": {
            "tile_cycles": round(tok_tc),
            "wall_ms_at_15p625MHz_flown": round(
                wl.tile_ms(tok_tc, wl.FLOWN_MHZ), 1),
            "wall_ms_at_62p5MHz_analyzed": round(
                wl.tile_ms(tok_tc, wl.ANALYZED_MHZ), 1),
            "covers": covers}}
    return out


def parse_cycles_e7(log: str) -> dict:
    """Both kicks' walk windows from one shared executor log: the ATT marks
    (E7A-*) and the EPI file's E-6 marks (E6R-*) key distinct stamps."""
    out = {"att": {}, "epi": wfl.parse_cycles(log)}
    st = {m.group(1): int(m.group(2)) for m in
          re.finditer(r"CYCMARK CYCMARK (\S+) cyc=(\d+)", log)}
    if "E7A-PREGO" in st and "E7A-WALKDONE" in st:
        shell = st["E7A-WALKDONE"] - st["E7A-PREGO"]
        out["att"] = {"walk_shell_cycles": shell,
                      "walk_tile_cycles": shell // (2 * wfl.TILE_DIV)}
    a = out["att"].get("walk_tile_cycles")
    b = out["epi"].get("walk_tile_cycles")
    if a is not None and b is not None:
        out["walk_tile_cycles"] = a + b
        out["walk_shell_cycles"] = (out["att"]["walk_shell_cycles"]
                                    + out["epi"]["walk_shell_cycles"])
    return out


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def bits16(x: np.ndarray) -> np.ndarray:
    return np.asarray(f64_to_f16_bits(np.asarray(x, dtype=np.float64)),
                      dtype=np.uint16)


def parse_ids(spec) -> list[int]:
    """'785,6722,315' -> [785, 6722, 315]. Refuses junk, empties, negatives."""
    if isinstance(spec, (list, tuple)):
        vals = list(spec)
    else:
        parts = [p.strip() for p in str(spec).split(",")]
        try:
            vals = [int(p) for p in parts if p != ""]
        except ValueError:
            raise SystemExit(f"REFUSE: --prompt {spec!r} is not a "
                             f"comma-separated token-id list")
    if not vals:
        raise SystemExit("REFUSE: --prompt selected no token ids")
    if any(v < 0 for v in vals):
        raise SystemExit(f"REFUSE: negative token id in {vals}")
    return vals


def parse_probes(args) -> tuple[set[int] | None, set[int] | None]:
    """--probe-steps/--probe-layers -> (steps|None, layers|None).

    Probe mode walks ONLY the named (step, layer) chains; every other
    layer slot is host-vouched golden and RECORDED as skipped — a probe
    run can never read as more coverage than it bought. Range checks
    (steps against this run's sampling-relevant window, layers against
    the model) live in run(), where prompt/tokens/model are known."""
    ps = getattr(args, "probe_steps", None)
    pl = getattr(args, "probe_layers", None)
    if ps is None and pl is None:
        return None, None
    if ps is None:
        raise SystemExit("REFUSE: --probe-layers without --probe-steps — "
                         "name the steps the probe is about")
    if getattr(args, "engine", None) not in ("sim-walked", "hw-walked"):
        raise SystemExit(f"REFUSE: --probe-steps is a walked-engine mode; "
                         f"engine={getattr(args, 'engine', None)!r} walks "
                         f"nothing to probe")
    steps = set(parse_ids(ps))
    layers = set(parse_ids(pl)) if pl is not None else None
    return steps, layers


def t_budget(engine: str, probing: bool) -> tuple[int, str]:
    """The session fence (prompt + new tokens) for one engine run.

    sim-walked FULL decodes are wall-budgeted (24 Verilator flights per
    token — T_SIM_MAX). A --probe-steps run's wall scales with the probe
    list instead of with T, so it gets the walk-envelope budget T_HW_MAX,
    the same fence hw-walked sessions get. rt.T_SESSION_MAX is enforced
    separately for every engine."""
    if engine == "hw-walked" or (engine == "sim-walked" and probing):
        return T_HW_MAX, "T_HW_MAX (the tile's walk envelope)"
    if engine == "sim-walked":
        return T_SIM_MAX, "T_SIM_MAX (Verilator wall budget)"
    return rt.T_SESSION_MAX, "T_SESSION_MAX (host session range)"


# ═══════════════ --fast N: the EXPLICIT demo/measurement mode ══════════════
# The DEFAULT run is the two-pass GRADING discipline: a full host-golden
# reference decode (pass 1) + per-value grades on EVERY walked chain — that
# is the right default and --fast changes NONE of it when absent (fast_gate
# returns None before consulting any other flag; every fast branch in run()
# and decode_pass() is behind that None).
#
# --fast N (walked engines only; REFUSED without --ddr-attested) trades
# verification coverage for wall clock, and the report says so in as many
# words:
#   * pass 1 (the host-golden reference decode) is SKIPPED entirely —
#     token identity vs the reference is therefore NOT CHECKABLE and the
#     run makes NO identity claim;
#   * walked-value grades run on every Nth generated token only (token 1,
#     1+N, … — full per-value grades there); the other tokens' chains
#     still FLY (program, fences, silence census, audit, cycles all real)
#     but no golden compare runs — they carry NO bit-exactness claim and
#     count ZERO walked MACs (the walked-MAC ledger counts only
#     graded-bit-exact work);
#   * what CANNOT be skipped stays: sess.step runs every layer of every
#     step because the engine subjects derive from golden's LayerFx
#     (_subject/_prep_one — staging operands, and the descriptor's
#     rq/comp calibration comes from golden_epilogue's own accumulators;
#     for --mask e7 the D-030 cores also feed the ATT descriptor's RQ
#     words). The SKIPPABLE golden work is exactly the post-flight
#     grading (wfp.grade's per-job golden recompute, wfl.grade_r1, the
#     e7 fs/ss/o8 compares) and the pass-1 reference decode.
FAST_UNGRADED_NOTE = ("fast: chain WALKED, grades SKIPPED (--fast) — NO "
                      "bit-exactness claim for this layer's walked values")


def _fast_ungraded(i: int, fast_n: int | None) -> bool:
    """True iff generated token i (0-based) is walked-but-UNGRADED under
    --fast N. fast_n=None (the default, --fast absent) never marks anything
    ungraded — the two-pass discipline grades every token."""
    return fast_n is not None and (i % fast_n) != 0


def fast_label(graded_tokens: list[int]) -> str:
    return (f"FAST MODE: reference pass SKIPPED, walked grades on tokens "
            f"{sorted(graded_tokens)}; UNGRADED tokens carry NO "
            f"bit-exactness claim; token identity vs the host-golden "
            f"reference is NOT checkable without pass 1 — this run makes "
            f"NO identity claim")


def fast_gate(args) -> int | None:
    """--fast validation. Returns N, or None when --fast is absent — and
    when absent it returns BEFORE consulting any other flag, so the
    default path can never trip a fast-mode refusal."""
    n = getattr(args, "fast", None)
    if n is None:
        return None
    n = int(n)
    if n < 1:
        raise SystemExit(f"REFUSE: --fast {n} — N must be >= 1 (N=1 grades "
                         f"every token and only drops the reference pass)")
    if getattr(args, "engine", None) not in ("sim-walked", "hw-walked"):
        raise SystemExit("REFUSE: --fast is a walked-engine mode — "
                         "host-golden's two-pass consistency check IS the "
                         "run; there is nothing walked to fast-grade")
    if not getattr(args, "ddr_attested", False):
        raise SystemExit(
            "REFUSE: --fast without --ddr-attested — the fast path drops "
            "the reference pass and most per-value grades, so the "
            "operator's DDR attestation is the substrate claim that "
            "remains; make it explicitly (f2_ddr_load --load --verify "
            "--full-verify, then pass the flag)")
    if getattr(args, "ffn", False):
        raise SystemExit("REFUSE: --fast with --ffn — the --ffn gate "
                         "requires a graded, bit-exact FFN window at EVERY "
                         "layer of every sampled step, which fast grading "
                         "contradicts by design")
    if getattr(args, "probe_steps", None) is not None:
        raise SystemExit("REFUSE: --fast with --probe-steps — probe mode "
                         "reduces WALK coverage, fast mode reduces GRADE "
                         "coverage; composing them makes the report's "
                         "coverage statement ambiguous. Run them "
                         "separately")
    return n


# ═══════════════ the Engine contract (the pluggable seam) ══════════════════
#
# start(ctx)                     once, after the model loads. ctx carries
#                                model/args/work/manifest/binary; the engine
#                                installs its own caches and REFUSES here if
#                                its substrate is missing (binary, image, a
#                                weights-dir identity mismatch).
# layer(ctx, step, li, fx, xrow) once per (decode step, layer). fx is
#                                golden's LayerFx for THIS layer at THIS
#                                step (the arbiter's own value, already
#                                computed); xrow is the layer's input row
#                                (fp16-grid f64). Returns a dict with AT
#                                LEAST:
#                                  r2_bits     uint16[D_model] — the layer
#                                              output the engine vouches
#                                              for. The harness REFUSES
#                                              unless it equals golden's r2
#                                              bits.
#                                  walked_macs int — MACs the TILE actually
#                                              produced AND that were graded
#                                              bit-exact (0 for host-only).
#                                  walk        dict|None — per-layer walk
#                                              record (grades, cycles,
#                                              walls) or None.
#                                  refused     str|None — a LOUD per-layer
#                                              refusal (operand not
#                                              walkable); the layer then
#                                              counted fully host-side.
#                                A grade mismatch must raise SystemExit, not
#                                return.
# finish(ctx)                    once; returns an engine summary dict.


def emit_cache_mode(args) -> str:
    """--emit-cache resolution. Default: ON for hw-walked (emission ≈1.2
    s/token is the measured per-token bottleneck there), OFF for sim-walked
    (the proven from-scratch path stays the default for the twin gates).
    host-golden emits nothing — an explicit flag there REFUSES; the absent
    default resolves to 'off' harmlessly."""
    mode = getattr(args, "emit_cache", None)
    eng = getattr(args, "engine", None)
    if mode is None:
        return "on" if eng == "hw-walked" else "off"
    if eng not in ("sim-walked", "hw-walked"):
        raise SystemExit(f"REFUSE: --emit-cache is a walked-engine flag — "
                         f"engine={eng!r} emits no walk programs to cache")
    return mode


def _emit_program(sub, desc, work, name, li, idir, emit_cache=None):
    """Program emission + static checks for ONE layer — module-level and
    self-free so a process pool can run 24 of these concurrently.

    emit_cache=None (the default, and --emit-cache off): today's full
    build, byte-path-identical — _emit_program_full below, unchanged.

    With an armed config ({mode, dir, run} — SimWalkedEngine._arm_emit_cache
    / emit_cache.py): the FIRST build for layer li runs the full path AND
    records the emitted bytes as the layer's template (from the recorded
    emission, never a re-run); every later (step, li) PATCHES the template's
    step-varying payload slots only (descriptor words, staged-activation
    QS/XW0 beats, the xrow preload row, the name note) and re-runs the
    CHEAP fences on the patched output: control-silence + census, the
    cap-count ledger, and the descriptor drift fence vs the template
    (wl.REPOINT_WORDS). mode 'verify' additionally full-builds every
    patched program and REFUSES unless byte-identical (the gate mode;
    emit_s excludes that overhead — verify_s carries it, disclosed)."""
    t0 = time.monotonic()
    if emit_cache is None or emit_cache.get("mode", "off") == "off":
        em = _emit_program_full(sub, desc, work, name, li, idir)
        em.update(emit_mode="full", emit_s=round(time.monotonic() - t0, 4))
        return em
    got = ec.patch_program(emit_cache, li, sub, desc, Path(work), name)
    if got is None:
        em = _emit_program_full(sub, desc, work, name, li, idir)
        ec.record_template(emit_cache, li, Path(em["path"]), name, desc,
                           sub, em["n_caps"])
        em.update(emit_mode="template",
                  emit_s=round(time.monotonic() - t0, 4))
        return em
    got["regions"] = wl.layer_regions(Path(idir), Path(work), li)
    got.update(emit_mode="patched",
               emit_s=round(time.monotonic() - t0, 4))
    if emit_cache["mode"] == "verify":
        t_v = time.monotonic()
        vdir = Path(emit_cache["dir"]) / "verify" / name
        em_full = _emit_program_full(sub, desc, vdir, name, li, idir)
        ec.assert_byte_identical(got["path"], em_full["path"], name)
        got.update(verify_ok=True,
                   verify_s=round(time.monotonic() - t_v, 4))
    return got


def _emit_program_full(sub, desc, work, name, li, idir):
    """The from-scratch emission — the proven path, verbatim. The
    semantics are byte-identical to the sequential path (same builders,
    same checks); only WHERE it runs changes."""
    man = wfl.build_program(sub, desc, Path(work), name,
                            act8=sub["h8"], qkv_drain=True,
                            act8_oproj=sub["attn8"])
    path = Path(man["path"])
    wfp.capture_ro(path)
    wfp.splice_fuel_arm(path)
    sil = wfp.silence_predicate(path)
    cen = wl.census(path)
    if cen["window_control_writes"] != 0 or not sil["silent"]:
        raise SystemExit(f"REFUSE: {name} walk window is not "
                         f"control-silent: {cen} / {sil}")
    regions = wl.layer_regions(Path(idir), Path(work), li)
    n_caps = 0
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and json.loads(ln).get("op") == "cap":
                n_caps += 1
    return {"path": str(path), "sil": sil, "regions": regions,
            "n_caps": n_caps}


def _grade_payload(caps, jobs, h8, blocks, r1_bits):
    """The pure grading computation for ONE layer — module-level so the
    batch pool can fan 24 of these out. REFUSE decisions stay with the
    caller (the engine), which inspects the returned results."""
    ro, lrd = wfl.split_caps(caps)
    res_q = wfp.grade(ro, jobs, h8, blocks)
    res_r1 = wfl.grade_r1(lrd, r1_bits)
    return res_q, res_r1


def _emit_program_e7(sub, desc_att, desc_epi, work, name, li, idir,
                     emit_cache=None):
    """The e7 pair for ONE layer — ATT kick (new builder, {STOREKV, SCORE,
    PV} at the live T) + the MASK_B kick VERBATIM (_emit_program — the flown
    per-layer program, byte-path-identical to --mask b). Module-level and
    self-free like _emit_program so the process pool can run these
    concurrently. Static checks: the ATT window admits only capture-FIFO
    advances; the MASK_B file keeps its own control-silence census.

    emit_cache covers the MASK_B (EPI) kick ONLY — the ATT kick's op
    structure is T-dependent (K/V record staging, the phase table, the
    drain ledger all move with T), so it is full-built every step."""
    t0 = time.monotonic()
    man_a = build_att_program(sub, desc_att, Path(work), f"{name}_att")
    pa = Path(man_a["path"])
    wfp.capture_ro(pa)                    # audit: no surviving RO expectation
    sil_a = att_silence(pa)
    if not sil_a["silent"]:
        raise SystemExit(f"REFUSE: {name} ATT walk window is not "
                         f"data-plane-only: {sil_a}")
    att_s = time.monotonic() - t0
    em_b = _emit_program(sub, desc_epi, work, f"{name}_qkv", li, idir,
                         emit_cache)
    n_a = _count_caps(pa)
    lay = att_cap_layout(int(sub["T"]))
    if n_a != lay["total"]:
        raise SystemExit(f"REFUSE: {name} ATT file bakes {n_a} caps, the "
                         f"drain arithmetic says {lay['total']} — the "
                         f"builder and the ledger disagree")
    return {"paths": [str(pa), em_b["path"]], "sil": sil_a,
            "sil_b": em_b["sil"], "regions": em_b["regions"],
            "n_caps": [n_a, em_b["n_caps"]],
            "emit_mode": em_b.get("emit_mode", "full"),
            "emit_s": round(att_s + float(em_b.get("emit_s") or 0.0), 4),
            "verify_s": em_b.get("verify_s"),
            "verify_ok": em_b.get("verify_ok")}


def build_ffn_descriptor(bases: dict, *, d_ffn_walk: int, comp_g: int,
                         comp_u: int, comp_d: int,
                         rq_d: tuple[int, int]) -> list[int]:
    """The fmt=1 FFN→DOWN slice descriptor — walk_ffn_toy.ffn_descriptor at
    the REAL 0.5B geometry: model0 carries (d_model=896, d_ffn=SLICE), the
    DOWN RQ pair sits at RQ[n_heads+1] (seq_layer_walker2 rq_word), and the
    three composites are the C2 f16-graded JC words."""
    scale, shift = rq_d
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(wfl.HEAD_DIM)
    w[fmt.W_MODEL0] = fmt.pack_model0(wfl.D_MODEL, d_ffn_walk)
    w[fmt.W_MODEL1] = fmt.pack_model1(wfl.N_HEADS, wfl.N_KV_HEADS, 1)
    w[fmt.W_MASK] = fmt.pack_mask(FFN_MASK_DOWN)
    w[fmt.W_STEP] = fmt.pack_step(1, 0)
    for t, name in ((fmt.TENS_WG, "Wg"), (fmt.TENS_WU, "Wu"),
                    (fmt.TENS_WD, "Wd")):
        w[fmt.W_TENS0 + t] = bases[name]
    w[fmt.W_JC0 + fmt.JC_GATE] = comp_g
    w[fmt.W_JC0 + fmt.JC_UP] = comp_u
    w[fmt.W_JC0 + fmt.JC_DOWN] = comp_d
    w[fmt.W_RQ0 + wfl.N_HEADS + 1] = fmt.pack_rq(scale, shift)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=wfl.HEAD_DIM) == fmt.ERR_NONE, \
        "illegal FFN slice walk image"
    return w


class FfnLane:
    """The walked engines' --ffn machinery: one walked FFN→DOWN slice
    window per (step, layer), independent of (and additional to) the E-6
    chain flight. Substrate note: the MAIN image lays Wd at fmt.jobs'
    k_job=1984 k-chunks (n-major/k-minor), so a d_ffn=SLICE fetch from it
    would misalign the byte→job stream; this lane builds a dedicated SLICE
    image through the UNTOUCHED make_weight_image.build against a sliced
    weights VIEW — every shape assert and block crosscheck of the tool
    stays live, and the walker's own chunk-offset arithmetic (base +
    chunk*d_model beats64) addresses it exactly.

    EXECUTOR SEAM (hw arm, 2026-08-13): program builder, fences and the
    bit-exact grade (walk_ffn_toy.grade_ffn on the bridge's capture
    stream) are executor-agnostic; only run_one's flight differs —
    sim flies the slice image through the twin's +ddr_image/+ddr_regions
    model, hw flies bridge.run_job(executor='hw') against the REBASED
    slice image the operator loaded and attested (FFN_HW_DDR_BASE header;
    no region-restriction arg — physical DDR holds the verified image).
    Like the E-6 chain on hw: no sim '+fuel_audit' printer and no CYCMARK
    window exist, so the audit is the attestation chain and no
    ffn_tile_cycles are claimed."""

    def __init__(self, chunks: int):
        if not 1 <= int(chunks) <= FFN_CHUNKS_MAX:
            raise SystemExit(
                f"REFUSE: --ffn-chunks {chunks} outside [1, "
                f"{FFN_CHUNKS_MAX}] — min(31 - "
                f"{wfl.D_MODEL // wfl.D_TILE} stage rows, "
                f"{FFN_FS_FIFO_DEPTH} fs-FIFO slots per silent window)")
        self.chunks = int(chunks)
        self.S = self.chunks * wfl.D_TILE
        self.d_ffn = None                # the model's own (from meta)
        self.idir = None
        self.man_sha = None
        self.bases = {}                  # li -> {"Wg","Wu","Wd"} base_64B
        self.regions = {}                # li -> per-layer regions file
        self.tensors = {}                # ONE-slot cache {li: {...}}
        self.desc_prev = None
        self.n_walked = 0
        self.n_refused = 0
        self.grade_s = 0.0               # cumulative grade_ffn wall
        self.hw = False                  # armed in prepare() from the engine
        import walk_ffn_toy as wft       # noqa: PLC0415 — the rung graders
        self.wft = wft

    # ── substrate: the slice image, built one-time and identity-fenced ─────
    def prepare(self, ctx) -> None:
        args, model, work = ctx["args"], ctx["model"], ctx["work"]
        self.hw = getattr(args, "engine", None) == "hw-walked"
        self.d_ffn = int(model.meta["d_ffn"])
        if self.S > self.d_ffn:
            raise SystemExit(f"REFUSE: slice {self.S} > model d_ffn "
                             f"{self.d_ffn}")
        base_bytes = FFN_HW_DDR_BASE if self.hw else 0
        wdir = Path(args.weights_dir).resolve()
        idir = Path(args.ffn_image) if args.ffn_image else \
            work / ("ffn_slice_image_hw" if self.hw else "ffn_slice_image")
        view = work / "ffn_view"
        stamp_p = idir / "ffn_view_stamp.json"
        meta_sha = hashlib.sha256((wdir / "meta.json").read_bytes()) \
            .hexdigest()
        stamp = {"parent_weights_dir": str(wdir),
                 "parent_meta_sha256": meta_sha,
                 "d_ffn_walk": self.S, "n_layers": model.n_layers,
                 "ddr_base_bytes": base_bytes}
        have = None
        if stamp_p.is_file():
            try:
                have = json.loads(stamp_p.read_text())
            except (OSError, json.JSONDecodeError):
                have = None
        rebuilt = have != stamp
        if rebuilt:
            self._build_slice_image(wdir, view, idir, model.n_layers,
                                    stamp, stamp_p, base_bytes)
        man = json.loads((idir / "ddr_image.json").read_text())
        if Path(man["weights_dir"]).resolve() != view.resolve():
            raise SystemExit(
                f"REFUSE: ffn image {idir} was built from "
                f"{man['weights_dir']}, not this run's view {view} — the "
                f"walked fetches would be about different tensors")
        geo = man["geometry"]
        if (geo["d_ffn"] != self.S or geo["D_model"] != wfl.D_MODEL
                or geo["d_tile"] != wfl.D_TILE):
            raise SystemExit(f"REFUSE: ffn image geometry {geo} != "
                             f"(D_model={wfl.D_MODEL}, d_ffn={self.S}, "
                             f"d_tile={wfl.D_TILE})")
        if not set(range(model.n_layers)) <= set(man["layers"]):
            raise SystemExit(f"REFUSE: ffn image holds layers "
                             f"{man['layers']}; the loop needs "
                             f"{model.n_layers}")
        if int(man.get("base_bytes", 0)) != base_bytes:
            raise SystemExit(
                f"REFUSE: ffn image {idir} is based at "
                f"{int(man.get('base_bytes', 0)):#x}, this "
                f"{'hw' if self.hw else 'sim'} run needs {base_bytes:#x} — "
                f"a sim image cannot be flown on silicon (it would alias "
                f"the main image) and vice versa")
        if self.hw:
            lo = min(t["base_64B"] for t in man["tensors"]) * 64
            if lo != FFN_HW_DDR_BASE:
                raise SystemExit(
                    f"REFUSE: ffn hw image's first region sits at "
                    f"{lo:#x}, not FFN_HW_DDR_BASE={FFN_HW_DDR_BASE:#x} — "
                    f"the manifest is not the rebased build this lane "
                    f"addresses")
            main_top = int(ctx["man"]["image_bytes"])
            if main_top > FFN_HW_DDR_BASE:
                raise SystemExit(
                    f"REFUSE: main image tops out at {main_top:#x} > "
                    f"FFN_HW_DDR_BASE={FFN_HW_DDR_BASE:#x} — the resident "
                    f"images would overlap; raise the constant (a "
                    f"deliberate layout change) and rebuild+reload the "
                    f"ffn image")
            load_cmd = (f"python3 scripts/fpga/f2/f2_ddr_load.py --image "
                        f"{idir} --load --verify --full-verify --slot "
                        f"{getattr(args, 'slot', 0)}")
            if rebuilt:
                raise SystemExit(
                    f"REFUSE: the ffn hw slice image was (re)built by THIS "
                    f"run, so it cannot already be resident in the card's "
                    f"DDR. Load it now, then re-run with "
                    f"--ffn-ddr-attested:\n  {load_cmd}")
            if not getattr(args, "ffn_ddr_attested", False):
                raise SystemExit(
                    f"REFUSE: hw-walked --ffn needs --ffn-ddr-attested — "
                    f"run\n  {load_cmd}\non THIS card first, then pass the "
                    f"flag; the report records the attestation you are "
                    f"making (the slice image is a SEPARATE DDR artifact "
                    f"at {FFN_HW_DDR_BASE:#x}, alongside the main image)")
        for t in man["tensors"]:
            self.bases.setdefault(t["layer"], {})[t["tensor"]] = t["base_64B"]
        src = [ln for ln in (idir / "ddr_image.regions.jsonl")
               .read_text().splitlines() if ln.strip()]
        for li in range(model.n_layers):
            keep = [ln for ln in src if json.loads(ln).get("layer") == li]
            if len(keep) != len(FFN_TENS):
                raise SystemExit(f"REFUSE: {len(keep)} ffn regions for "
                                 f"layer {li} (want {len(FFN_TENS)})")
            dst = work / f"ffn_regions_L{li:02d}.jsonl"
            dst.write_text("\n".join(keep) + "\n")
            self.regions[li] = dst
        self.idir = idir
        self.man_sha = man["image_sha256"]
        eprint(f"[{'hw' if self.hw else 'sim'}-walked/ffn] slice image "
               f"{idir} sha={self.man_sha[:16]}… "
               f"d_ffn_walk={self.S}/{self.d_ffn} "
               f"({self.chunks} chunks; stage rows "
               f"{wfl.D_MODEL // wfl.D_TILE}+{self.chunks} <= 31"
               + (f"; DDR base {FFN_HW_DDR_BASE:#x}, operator-attested "
                  f"full verify)" if self.hw else ")"))

    def _build_slice_image(self, wdir, view, idir, n_layers, stamp,
                           stamp_p, base_bytes: int = 0) -> None:
        import make_weight_image as mwi  # noqa: PLC0415 — resident via fp
        eprint(f"[{'hw' if self.hw else 'sim'}-walked/ffn] building slice "
               f"image (one-time): view={view} image={idir}"
               + (f" base={base_bytes:#x}" if base_bytes else ""))
        t0 = time.monotonic()
        view.mkdir(parents=True, exist_ok=True)
        meta = json.loads((wdir / "meta.json").read_text())
        if int(meta["d_ffn"]) < self.S:
            raise SystemExit(f"REFUSE: parent d_ffn {meta['d_ffn']} < "
                             f"slice {self.S}")
        for li in range(n_layers):
            for name in FFN_TENS:
                W = np.load(wdir / f"L{li:02d}_{name}.npy", mmap_mode="r")
                sl = W[:, :self.S] if name in ("Wg", "Wu") else \
                    W[:self.S, :]
                np.save(view / f"L{li:02d}_{name}.npy",
                        np.ascontiguousarray(sl, dtype=np.int8))
        (view / "meta.json").write_text(
            json.dumps(dict(meta, d_ffn=self.S), indent=1))
        man = mwi.build(view, list(range(n_layers)), idir,
                        tensors=(fmt.TENS_WG, fmt.TENS_WU, fmt.TENS_WD),
                        d_tile=wfl.D_TILE, base_bytes=base_bytes)
        stamp_p.write_text(json.dumps(stamp, indent=1))
        eprint(f"[{'hw' if self.hw else 'sim'}-walked/ffn] slice image "
               f"built: {man['image_bytes'] / 1e6:.1f} MB in "
               f"{time.monotonic() - t0:.0f}s")

    # ── per-layer live operands ────────────────────────────────────────────
    def _tensors(self, ctx, li: int) -> dict:
        if li not in self.tensors:
            self.tensors.clear()         # one-slot: layers walk in order
            wdir, model = Path(ctx["args"].weights_dir), ctx["model"]
            lw = model.layers[li]
            self.tensors[li] = {
                "Wg": np.asarray(np.load(wdir / f"L{li:02d}_Wg.npy",
                                         mmap_mode="r")[:, :self.S],
                                 dtype=np.int64),
                "Wu": np.asarray(np.load(wdir / f"L{li:02d}_Wu.npy",
                                         mmap_mode="r")[:, :self.S],
                                 dtype=np.int64),
                "Wd": np.asarray(np.load(wdir / f"L{li:02d}_Wd.npy",
                                         mmap_mode="r")[:self.S, :],
                                 dtype=np.int64),
                "s_wg": float(lw.s_wg), "s_wu": float(lw.s_wu),
                "s_wd": float(lw.s_wd)}
        return self.tensors[li]

    def _subject(self, ctx, li: int, fx) -> dict:
        """Live operands: h2 C-1 frames (identity staging — SystemExit here
        is the amax!=127 REFUSAL class), golden's own h2 row scale (folded
        into the gate/up composites so the probe swiglu runs at physical
        magnitude; == the max C-1 frame scale, same amax), the layer's own
        r1 bits, and the slice tensors + per-tensor scales."""
        tn = self._tensors(ctx, li)
        h2_real = fp.activation_for(fx, "h2")
        h2_8, _ = fp.c1_frames(h2_real, wfl.D_TILE)
        _, sb = at.quant_rows_i8(
            np.asarray(h2_real, dtype=np.float64)[None, :])
        s_h2 = float(f16_bits_to_f64(
            np.array([sb[0]], dtype=np.uint16))[0])
        return dict(tn, h2_8=np.asarray(h2_8, dtype=np.int64), s_h2=s_h2,
                    r1_bits=bits16(fx.r1))

    def _golden(self, sub: dict) -> dict:
        """Golden's OWN functions on the framed slice operands — the
        walk_ffn_toy recipe (toy_ffn + its rung-3 tail) at slice scale.
        SystemExit = a REFUSAL class (non-C2 composite, o8 clip, deq
        outside the residual window)."""
        nch = self.chunks
        x = sub["h2_8"][None, :]
        acc_g = cpg.gemm_i8_ksplit(x, sub["Wg"])[0].astype(np.int64)
        acc_u = cpg.gemm_i8_ksplit(x, sub["Wu"])[0].astype(np.int64)
        comp_g = tfm.f16_grade(sub["s_h2"] * sub["s_wg"])
        comp_u = tfm.f16_grade(sub["s_h2"] * sub["s_wu"])
        prod = f16_bits_to_f64(f64_to_f16_bits(
            tfm.silu_apply(acc_g.astype(np.float64) * comp_g)
            * (acc_u.astype(np.float64) * comp_u)))
        p_codes_m, p_sb = at.quant_rows_i8(prod.reshape(nch, wfl.D_TILE))
        p_codes = np.asarray(p_codes_m, dtype=np.int64).reshape(-1)
        acc_d = cpg.gemm_i8_ksplit(p_codes[None, :],
                                   sub["Wd"])[0].astype(np.int64)
        amax = int(np.max(np.abs(acc_d), initial=0)) or 1
        scale, shift = at.calib_requant(amax)
        d8 = at.requant_i32_to_i8(acc_d, scale, shift).astype(np.int64)
        if int(np.max(np.abs(d8), initial=0)) >= 128:
            raise SystemExit("FFN DOWN o8 clip (calib target violated)")
        # the uniform-contract product scale: max per-chunk C-1 scale ==
        # the ONE scale the proven host contract (layer_offload's single
        # C-1 row) would give this slice vector — well-defined at any nchunk
        s_p = float(np.max(f16_bits_to_f64(
            np.asarray(p_sb, dtype=np.uint16))))
        comp_d = np.float32(tfm.f16_grade(
            s_p * sub["s_wd"] * float(1 << shift) / float(scale)))
        comps = {"gate": float(comp_g), "up": float(comp_u),
                 "down": float(comp_d)}
        for name, c in comps.items():
            b = int(np.float32(c).view(np.uint32))
            if (b & 0x1FFF) != 0 or not 0 < ((b >> 23) & 0xFF) < 255:
                raise SystemExit(f"FFN composite {name}={b:#010x} not "
                                 f"positive-normal fp16-grade (C2)")
        deq = d8.astype(np.float64) * float(comp_d)
        bad = [i for i, v in enumerate(deq)
               if not wfp._deq_resid_fit(float(v))]
        if bad:
            raise SystemExit(f"FFN deq stream outside the residual window "
                             f"at {len(bad)} lanes (first {bad[:4]}) — "
                             f"this live subject cannot fly the epilogue")
        r1_f = f16_bits_to_f64(np.asarray(sub["r1_bits"], dtype=np.uint16))
        r2_bits = np.asarray(f64_to_f16_bits(r1_f + deq), dtype=np.uint16)
        return {"p_codes": p_codes, "p_scales": [int(v) for v in p_sb],
                "comp_g": int(np.float32(comp_g).view(np.uint32)),
                "comp_u": int(np.float32(comp_u).view(np.uint32)),
                "comp_d": int(comp_d.view(np.uint32)),
                "rq_d": (int(scale), int(shift)), "r2_bits": r2_bits}

    # ── the program (walk_ffn_toy.build_walk_ffn at the real geometry) ─────
    def _build_at_d(self, sub, ge, desc, out_dir: Path, name: str) -> dict:
        rows = wfl.D_MODEL // wfl.D_TILE
        s = glo.LayerScript(wfl.D_TILE)
        s.emit(f"// FFN→DOWN SLICE WALK {name}: ONE fmt=1 kick = "
               f"{self.chunks}x(USWI + fuel Wg chunk + 8 gate jobs + fuel "
               f"Wu chunk + 8 up jobs + S2_CS* product stage) then LVLD + "
               f"RES2 arm + JC_DOWN + fuel Wd slice + {wfl.D_MODEL // 8} "
               f"DOWN jobs with the o8 epilogue onto the r1 row. Host in "
               f"the window: NOTHING.")
        glo._prologue(s)
        s.emit("// [host, pre-walk] preload the resid row with the layer's "
               "own r1 — RES2 adds the DOWN epilogue's deq stream onto it "
               "(the E-6 idiom)")
        s.lptr(glo.BANK_RESID, 0)
        s.ldata(sub["r1_bits"])
        s.emit(f"// [host, pre-walk] stage the {rows} C-1 h2 frames into "
               f"act bank {wfl.ACT_BANK} (amax-127 identity staging — the "
               f"E-6 staging idiom)")
        s.route(rdst=1, asrc=1)
        for r in range(rows):
            rowv = sub["h2_8"][r * wfl.D_TILE:(r + 1) * wfl.D_TILE] \
                .astype(np.float64)
            pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(rowv)]
            g3.inject_jobs(s, pairs, 1)
            s.ess(wfl.F16_ONE, 1)
            s.aj(0, wfl.ACT_BANK, 0, 1, s.BPR, r)
        s.pmode(True)
        s.route(rdst=0, asrc=0)
        s.emit(f"// {wfp.FUEL_MARK}")
        s.csrr(wfl.W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
        s.csrw(wfl.W_DPTR, 0)
        for v in desc:
            s.csrw(wfl.W_DDATA, v)
        s.emit("// CYCMARK FFN-PREGO")
        s.csrw(wfl.W_CTRL, 0x3)
        s.csrp(wfl.W_STATUS, wfl.W_ST_BUSY, 0x0)
        s.emit("// CYCMARK FFN-WALKDONE")
        s.csrr(wfl.W_STATUS,
               wfl.W_ST_BUSY | wfl.W_ST_ESTK | wfl.W_ST_ECODE, 0x0)
        s.csrw(wfl.W_CTRL, 0x0)
        s.emit(f"// {wfp.FUEL_UNMARK}")
        s.emit(f"// [host, post-walk] egress: {self.chunks} fs beats (the "
               f"CS stage row scales), the staged product rows at rows "
               f"{rows}..{rows + self.chunks - 1} (identity readbacks), "
               f"then the r2 row")
        for _ in range(self.chunks):
            s.efs(0, 0)
        s.csrw(glo.L_CTRL, 0)
        s.route(rdst=0, wsrc=0, asrc=0)
        for c in range(self.chunks):
            for base in range(0, wfl.D_TILE, wfl.MXE_N):
                glo.identity_readback(s, rows + c, base)
        s.lrptr(0)
        s.erd(wfl.D_MODEL)
        s.pmode(False)
        glo._drain_and_check(s)
        x = glo._translate(s, note=f"FFN→DOWN slice walk: {name}")
        man = glo._emit(x, out_dir, name, dict(
            stage="token_loop_ffn",
            kind=f"ONE WALK: chunked gate/up + on-tile swiglu + "
                 f"walker-staged products + DOWN epilogue + RES2; "
                 f"d_ffn_walk={self.S} of {self.d_ffn}",
            d=wfl.D_TILE, mask=f"{FFN_MASK_DOWN:#x}", chunks=self.chunks))
        tg.retarget_info_tier(man["path"], tg.IMG_05B)
        return man

    def _build_program(self, sub, ge, desc, work: Path, name: str) -> dict:
        with tg.at_d(wfl.D_TILE):
            man = self._build_at_d(sub, ge, desc, Path(work), name)
        p = Path(man["path"])
        wfp.splice_fuel_arm(p)
        wfp.splice_fuel_disarm(p)
        sil = wfp.silence_predicate(p)
        cen = wl.census(p)
        if (cen["window_control_writes"] != 0
                or cen["window_ro_advances"] != 0 or not sil["silent"]):
            raise SystemExit(f"REFUSE: {name} FFN walk window is not "
                             f"host-silent: {cen} / {sil}")
        n_caps = 0
        with open(p) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln and json.loads(ln).get("op") == "cap":
                    n_caps += 1
        return {"path": str(p), "sil": sil, "n_caps": n_caps}

    @staticmethod
    def _cycles(log: str) -> dict:
        st = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"CYCMARK CYCMARK (\S+) cyc=(\d+)", log)}
        out = {"stamps": {k: v for k, v in st.items()
                          if k.startswith("FFN-")}}
        if "FFN-PREGO" in st and "FFN-WALKDONE" in st:
            shell = st["FFN-WALKDONE"] - st["FFN-PREGO"]
            out["ffn_shell_cycles"] = shell
            out["ffn_tile_cycles"] = shell // (2 * wfl.TILE_DIV)
        return out

    # ── one (step, layer): build → fence → fly → grade ─────────────────────
    def run_one(self, ctx, step: int, li: int, fx) -> dict:
        args = ctx["args"]
        tag = f"t{step:03d}_L{li:02d}"
        t0 = time.monotonic()
        try:
            sub = self._subject(ctx, li, fx)
            ge = self._golden(sub)
        except SystemExit as e:
            self.n_refused += 1
            eprint(f"    [{tag}] FFN REFUSED (subject): {e}")
            return {"refused": str(e), "macs_total": 0,
                    "slice": {"d_ffn_walk": self.S, "d_ffn": self.d_ffn,
                              "chunks": self.chunks}}
        desc = build_ffn_descriptor(self.bases[li], d_ffn_walk=self.S,
                                    comp_g=ge["comp_g"],
                                    comp_u=ge["comp_u"],
                                    comp_d=ge["comp_d"], rq_d=ge["rq_d"])
        if self.desc_prev is not None:
            moved = [w for w in range(fmt.DESC_WORDS)
                     if desc[w] != self.desc_prev[w]]
            if not set(moved) <= set(FFN_REPOINT_WORDS):
                raise SystemExit(
                    f"REFUSE: {tag} FFN descriptor moved words {moved} — "
                    f"more than the FFN re-point set "
                    f"{list(FFN_REPOINT_WORDS)}")
        self.desc_prev = desc
        em = self._build_program(sub, ge, desc, ctx["work"], f"ffn_{tag}")
        t_build = time.monotonic() - t0
        t0 = time.monotonic()
        if self.hw:
            # hw: the walk fetches from PHYSICAL DDR — the rebased slice
            # image the operator loaded and attested (prepare()'s fences).
            # No region restriction (the full verified image is resident),
            # no +poll_limit (f2_host_run bounds each poll by wall clock).
            r = bridge.run_job([em["path"]], executor="hw",
                               slot=getattr(args, "slot", 0),
                               timeout_s=args.timeout)
        else:
            # +poll_limit: the FFN slice window (~2.9 MB fuel + 384 jobs)
            # runs longer than the executor's default 2,000,000-shell-cycle
            # single poll budget at tile_div=2 — walk_ffn_toy's own knob
            # and value.
            r = bridge.run_job([em["path"]], executor="sim",
                               binary=str(ctx["binary"]),
                               extra_args=(
                                   f"+ddr_image={self.idir}/ddr_image.bin",
                                   f"+ddr_regions={self.regions[li]}",
                                   "+fuel_audit", "+poll_limit=16000000"),
                               timeout_s=args.timeout,
                               tile_div=wfl.TILE_DIV)
        wall = time.monotonic() - t0
        if not r["ok"]:
            raise SystemExit(f"REFUSE: {tag} FFN executor not ok "
                             f"(rc={r['rc']}, timed_out={r['timed_out']}) "
                             f"— log tail:\n" + r["log"][-2000:])
        if self.hw:
            # No sim '+fuel_audit' printer on hw (HwWalkedEngine._audit's
            # documented stance): the attestation chain is the operator's
            # full-verified DDR load (--ffn-ddr-attested, fenced at
            # prepare()) + run_job's ok-contract + the bit-exact grade
            # below. Record 0 sim-audit lines.
            aud = 0
        else:
            aud = r["log"].count("-> ok")
            if aud < 1:
                raise SystemExit(f"REFUSE: {tag} FFN no fuel-audit '-> ok' "
                                 f"in the executor log — the fetch path is "
                                 f"unattested")
        gstep = {"d": wfl.D_TILE, "nchunk": self.chunks, "down": True,
                 "p_codes": ge["p_codes"], "p_scales": ge["p_scales"],
                 "r2_bits": ge["r2_bits"]}
        t_g = time.monotonic()
        g = self.wft.grade_ffn(r["captures"], gstep)
        self.grade_s += time.monotonic() - t_g
        if not g["equal"]:
            raise SystemExit(f"REFUSE: {tag} walked FFN NOT bit-exact — "
                             f"fs={g['fs']} p_codes={g['p_codes']} "
                             f"r2={g.get('r2_row')} — the tile diverged "
                             f"from golden on the framed slice operands")
        cyc = self._cycles(r["log"])
        self.n_walked += 1
        macs = {"gate": wfl.D_MODEL * self.S, "up": wfl.D_MODEL * self.S,
                "down": self.S * wfl.D_MODEL}
        eprint(f"    [{tag}] FFN {self.chunks}-chunk slice: fs "
               f"{self.chunks}/{self.chunks} + p_codes {self.S}/{self.S} "
               f"+ r2 {wfl.D_MODEL}/{wfl.D_MODEL} BIT-EXACT  tile_cyc="
               f"{cyc.get('ffn_tile_cycles')}  build={t_build:.1f}s "
               f"exec={wall:.1f}s")
        return {"refused": None, "macs_total": sum(macs.values()),
                "slice": {"d_ffn_walk": self.S, "d_ffn": self.d_ffn,
                          "chunks": self.chunks},
                "fs": g["fs"], "p_codes": g["p_codes"],
                "r2_row": g["r2_row"], "silence": em["sil"],
                "fuel_audit_ok": aud, "cycles": cyc,
                "rq_d": list(ge["rq_d"]),
                "comp": {"gate": f"{ge['comp_g']:#010x}",
                         "up": f"{ge['comp_u']:#010x}",
                         "down": f"{ge['comp_d']:#010x}"},
                "walked_macs": macs,
                "build_s": round(t_build, 2),
                "sim_wall_s": round(wall, 2),
                "grade_seam": FFN_SEAM_NOTE}


class HostGoldenEngine:
    """The pure golden reference. The 'compute' IS the arbiter's own
    decoder_layer_fx call the harness already made; this engine vouches for
    exactly that value and nothing else runs."""
    name = "host-golden"

    def start(self, ctx) -> None:
        pass

    def layer(self, ctx, step: int, li: int, fx, xrow) -> dict:
        return {"r2_bits": bits16(fx.r2), "walked_macs": 0,
                "walk": None, "refused": None}

    def finish(self, ctx) -> dict:
        return {"engine": self.name, "walked_steps": "none — pure host"}


class SimWalkedEngine:
    """Each layer's walked {FPROJ, QKV, OPROJ, RES1} chain on a Verilator
    twin — walk_layers_05b's per-layer machinery (descriptor, program,
    fuel arm, silence census, grades) reused verbatim, re-subjected at
    every decode step of the token loop instead of the one prefill step.

    PROBE MODE (--probe-steps [+ --probe-layers]): walk ONLY the named
    (step, layer) chains — the deep-T evidence mode, where the session T
    exceeds the FULL-decode wall budget but the Verilator wall stays
    bounded by the probe count. Every unprobed layer slot is host-vouched
    (r2 gate still applies) and recorded as skipped, never as walked."""
    name = "sim-walked"

    def __init__(self):
        self.jobs = wfl.expected_qkv_jobs()
        self.tensors = {}                    # li -> {"Wo","s_wo","blocks"}
        self.bases = {}                      # li -> tensor-table bases
        self.desc_prev = None
        self.desc_att_prev = None            # e7: (step, descriptor words)
        self.n_walked = 0
        self.n_refused = 0
        self.parity_done = False
        self.ffn = None                      # --ffn: the FfnLane
        self.probe_steps = None              # None => walk every (step, li)
        self.probe_layers = None             # None => all layers at a probe
        self.n_skipped = 0                   # layer slots skipped by probe
        self.grade_s = 0.0                   # cumulative post-flight grading
        self.n_ungraded = 0                  # --fast: walked, NOT graded
        self.emit_cache = None               # --emit-cache config (start())
        self._emit_stat = {"s": 0.0, "verify_s": 0.0, "modes": Counter()}
        self.emit_steps = []                 # per-step emission ledger
        self.emit_verify = [0, 0]            # [byte-identical, compared]

    def _arm_emit_cache(self, ctx) -> None:
        """--emit-cache: resolve the mode and stake a RUN-SCOPED template
        dir. Templates are never shared across runs (the meta carries the
        run token and _load REFUSES a stale one) — a template must come
        from THIS run's own recorded emission."""
        mode = emit_cache_mode(ctx["args"])
        if mode == "off":
            self.emit_cache = None
            return
        run_tok = f"{int(time.time())}_{os.getpid()}"
        cdir = Path(ctx["work"]) / "emit_cache" / run_tok
        cdir.mkdir(parents=True, exist_ok=True)
        self.emit_cache = {"mode": mode, "dir": str(cdir), "run": run_tok}
        eprint(f"[{self.name}] EMIT-CACHE {mode}: per-layer MASK_B program "
               f"templates in {cdir} — first build per layer records the "
               f"emission; later steps patch it, with the cheap fences "
               f"(silence census, cap count, descriptor drift) re-run on "
               f"every patched file"
               + ("; verify: every patched file is ALSO full-built and "
                  "byte-compared, REFUSING on any difference"
                  if mode == "verify" else ""))

    def _tally_emit(self, em) -> None:
        if self.emit_cache is None:
            return
        st = self._emit_stat
        st["s"] += float(em.get("emit_s") or 0.0)
        st["verify_s"] += float(em.get("verify_s") or 0.0)
        st["modes"][em.get("emit_mode", "full")] += 1
        if em.get("verify_ok") is not None:
            self.emit_verify[1] += 1
            if em.get("verify_ok"):
                self.emit_verify[0] += 1

    def _print_emit_step(self, step) -> None:
        """Close one step's emission ledger (gate d's evidence): the
        per-step emission seconds, labelled by mode, with verify overhead
        split out (it is verification, not emission)."""
        if self.emit_cache is None:
            return
        st = self._emit_stat
        n = sum(st["modes"].values())
        if n == 0:
            return
        modes = dict(st["modes"])
        line = (f"[emit-cache] step {step}: emission {st['s']:.3f}s over "
                f"{n} program(s) {modes}")
        if st["verify_s"]:
            line += (f" + verify overhead {st['verify_s']:.3f}s (full "
                     f"rebuild + byte compare; excluded from emission)")
        eprint(line)
        self.emit_steps.append({"step": int(step),
                                "emit_s": round(st["s"], 3),
                                "verify_s": round(st["verify_s"], 3),
                                "programs": n, "modes": modes})
        self._emit_stat = {"s": 0.0, "verify_s": 0.0, "modes": Counter()}

    def flush_step(self, ctx, step) -> None:
        """sim engine: the per-step hook decode_pass already calls for
        batched engines — used here ONLY to close the emission ledger; no
        execution or grading happens (the sim path stays per-layer)."""
        self._print_emit_step(step)

    def _arm_probes(self, args) -> None:
        self.probe_steps, self.probe_layers = parse_probes(args)
        if self.probe_steps is not None:
            lay = ("ALL" if self.probe_layers is None
                   else sorted(self.probe_layers))
            eprint(f"[{self.name}] PROBE mode: steps "
                   f"{sorted(self.probe_steps)} x layers {lay} — every "
                   f"other layer slot is host-vouched (recorded skipped)")

    def _probe_skip(self, step: int, li: int, fx) -> dict | None:
        """None => this (step, layer) is walked. Otherwise the host-vouched
        skip record (bit gate still applies to it downstream)."""
        if self.probe_steps is None:
            return None
        if step in self.probe_steps and (self.probe_layers is None
                                         or li in self.probe_layers):
            return None
        self.n_skipped += 1
        return {"r2_bits": bits16(fx.r2), "walked_macs": 0, "walk": None,
                "refused": None, "skipped": "probe"}

    # ── start: substrate + identity fences ─────────────────────────────────
    def start(self, ctx) -> None:
        args, model = ctx["args"], ctx["model"]
        if (getattr(args, "mask", "b") == "e7"
                and getattr(args, "tier", "kvq8") != "kvq8"):
            raise SystemExit("REFUSE: --mask e7 walks score/pv at CQ-8 "
                             "only (the B1 fenced scope) — --tier kvq8")
        idir = Path(args.image)
        man_p = idir / "ddr_image.json"
        if not man_p.is_file():
            raise SystemExit(f"REFUSE: no DDR image manifest at {man_p} — "
                             f"build it (make_weight_image.py build "
                             f"--layers 0..23)")
        man = json.loads(man_p.read_text())
        have = set(man["layers"])
        need = set(range(model.n_layers))
        if not need <= have:
            raise SystemExit(f"REFUSE: image {idir} holds layers "
                             f"{sorted(have)}; the loop needs {sorted(need)}")
        # the DDR-resident weights MUST be the model's own weights — the
        # walk fetches from the image while golden reads the weights dir.
        wdir_img = Path(man["weights_dir"]).resolve()
        wdir_mod = Path(args.weights_dir).resolve()
        if wdir_img != wdir_mod:
            raise SystemExit(
                f"REFUSE: image weights_dir {wdir_img} != --weights-dir "
                f"{wdir_mod} — the walked fetches and the golden grades "
                f"would be about different tensors")
        binary = Path(args.binary) if args.binary else \
            REPO / "verif/f2sim" / args.obj / "f2sim"
        if not binary.is_file():
            raise SystemExit(
                f"REFUSE: sim binary missing: {binary} — build it:\n"
                f"  cd verif/f2sim && make build D=64 DDR=1 "
                f"OBJ=obj_tokenloop VFLAGS_EXTRA='{wfl.tg.IMG_05B.defines()}'")
        for t in man["tensors"]:
            self.bases.setdefault(t["layer"], {})[t["tensor"]] = t["base_64B"]
        ctx["man"], ctx["idir"], ctx["binary"] = man, idir, binary
        self._arm_probes(args)
        self._arm_emit_cache(ctx)
        eprint(f"[sim-walked] image {idir} sha={man['image_sha256'][:16]}… "
               f"binary {binary}")
        if getattr(args, "ffn", False):
            self.ffn = FfnLane(getattr(args, "ffn_chunks", FFN_CHUNKS_MAX))
            self.ffn.prepare(ctx)

    def _tensors(self, ctx, li: int) -> dict:
        if li not in self.tensors:
            wdir, model = Path(ctx["args"].weights_dir), ctx["model"]
            self.tensors[li] = {
                "Wo": np.asarray(np.load(wdir / f"L{li:02d}_Wo.npy"),
                                 dtype=np.int64),
                "s_wo": float(model.layers[li].s_wo),
                "blocks": {n: np.asarray(np.load(wdir / f"L{li:02d}_{n}.npy"),
                                         dtype=np.int64)
                           for n, _ in wfl.QKV}}
        return self.tensors[li]

    def _subject(self, ctx, li: int, fx, xrow, step: int | None = None) -> dict:
        """The walk subject at THIS (step, layer) — walk_layers_05b's
        subject_for re-sourced from the live session instead of the fixed
        prefill. Same field shape; parity-fenced below. --mask e7 extends
        it with the attention operands (per-engine live K/V records, the
        pre-rope q row, the step's phase table) — all golden's own arrays,
        picklable for the emission pool; the AttnCore grades stay on fx."""
        tn = self._tensors(ctx, li)
        attn8, _ = fp.c1_frames(np.asarray(fx.attn, dtype=np.float64),
                                wfl.D_TILE)
        h8, _ = fp.c1_frames(fp.activation_for(fx, "h"), wfl.D_TILE)
        sub = {"bases": self.bases[li], "xrow_bits": bits16(xrow),
               "attn8": np.asarray(attn8, dtype=np.int64),
               "h8": np.asarray(h8, dtype=np.int64),
               "Wo": tn["Wo"], "s_wo": tn["s_wo"], "blocks": tn["blocks"]}
        if getattr(ctx["args"], "mask", "b") != "e7":
            return sub
        heads = fx.heads
        if len(heads) != wfl.N_HEADS:
            raise RuntimeError(f"REFUSE: golden carries {len(heads)} heads, "
                             f"the walk needs {wfl.N_HEADS}")
        T = int(fx.T)
        if step is not None and T != step + 1:
            raise RuntimeError(f"REFUSE: fx.T={T} != step+1={step + 1} — the "
                             f"self-inclusive composition drifted")
        group = wfl.N_HEADS // wfl.N_KV_HEADS
        K16, V16 = [], []
        for g_i in range(wfl.N_KV_HEADS):
            k_ref = np.asarray(heads[g_i * group].K_f16, dtype=np.uint16)
            v_ref = np.asarray(heads[g_i * group].V_f16, dtype=np.uint16)
            for j in range(1, group):
                hd = heads[g_i * group + j]
                if not (np.array_equal(k_ref, np.asarray(hd.K_f16))
                        and np.array_equal(v_ref, np.asarray(hd.V_f16))):
                    raise RuntimeError(
                        f"REFUSE: KV group {g_i} heads disagree on K/V — "
                        f"golden's GQA composition drifted")
            if k_ref.shape != (T, wfl.HEAD_DIM):
                raise RuntimeError(f"REFUSE: K16[{g_i}] shape {k_ref.shape} "
                                 f"!= ({T}, {wfl.HEAD_DIM})")
            K16.append(k_ref)
            V16.append(v_ref)
        # ── the D-030 arbiter composition (erratum F5, gen_qmh_case) ───────
        # The session's golden is a BUS-OFF composition: fx.q_real is NOT
        # on the fp16 grid, and fx.heads' cores rotated that UN-narrowed
        # row. The tile can only rotate what it stages — f16(q_real),
        # MODE_F16's one RNE — so the walk's arbiter is attention_core
        # RE-COMPOSED in the tile's own order (`_f16` -> `rope_fx` -> core)
        # on the staged operands, the qmh D-030 idiom verbatim
        # (verif/top/qpath/gen_qmh_case.py:98-146). The layer output the
        # loop consumes stays golden's own composition (fx.r2, the r2 bit
        # gate); the walked attention is a bit-exact PROBE on the staged
        # operands, exactly like the QKV probes on their c1-framed rows.
        pos = T - 1
        w_l = ctx["model"].layers[li]
        theta = tfm.rope_theta(wfl.HEAD_DIM,
                               float(getattr(w_l, "rope_theta_base",
                                             10000.0)))
        q_real = np.asarray(fx.q_real, dtype=np.float64)
        q16 = np.empty(wfl.D_MODEL, dtype=np.uint16)
        cores = []
        for h in range(wfl.N_HEADS):
            sl = slice(h * wfl.HEAD_DIM, (h + 1) * wfl.HEAD_DIM)
            pre_f64 = tfm._f16(q_real[sl])           # the ONE NP-r narrowing
            pre = np.asarray(f64_to_f16_bits(pre_f64), dtype=np.uint16)
            if not np.array_equal(
                    np.asarray(f64_to_f16_bits(f16_bits_to_f64(pre)),
                               dtype=np.uint16), pre):
                raise RuntimeError(f"REFUSE: head {h} staged q row not "
                                   f"fp16-idempotent (qmh M2)")
            q16[sl] = pre
            rot = tfm.rope_fx(pre_f64, float(pos), theta)
            cores.append(at.attention_core(
                np.asarray(rot, dtype=np.float64), K16[h // group],
                V16[h // group], at.TIER_CQ8,
                G=int(getattr(ctx["args"], "group", 128)), outlier_idx=[]))
        sub.update({
            "T": T, "pos": pos, "q16": q16, "K16": K16, "V16": V16,
            "head_rqs": [tuple(c.rq) for c in cores],
            "cores": cores,
            "phase": glo._phase_q_codes(w_l, wfl.HEAD_DIM, pos)})
        return sub

    def _parity_fence(self, ctx, sub0: dict) -> None:
        """First walked step, layer 0, on the committed prompt: my live
        subject must equal walk_fuel_layer.golden_subject's own — the
        builder-drift fence walk_layers_05b runs, applied to THIS builder."""
        ref = wfl.golden_subject(ctx["idir"], 0)
        for k in ("xrow_bits", "attn8", "h8", "Wo"):
            if not np.array_equal(sub0[k], ref[k]):
                raise SystemExit(
                    f"REFUSE: token-loop subject[{k}] != "
                    f"wfl.golden_subject[{k}] at the committed prompt's "
                    f"last prefill step — the live subject builder drifted "
                    f"from the flown one")
        if sub0["bases"] != ref["bases"] or sub0["s_wo"] != ref["s_wo"]:
            raise SystemExit("REFUSE: layer-0 bases/s_wo drifted from "
                             "wfl.golden_subject")
        eprint("[sim-walked] subject parity vs wfl.golden_subject: OK")

    def grade_seconds(self) -> float:
        """Cumulative wall spent in post-flight golden grading — the
        decode_pass measurement hook reads the delta per token so phases_s
        can separate engine (build+fly) from grade time."""
        return self.grade_s + (self.ffn.grade_s if self.ffn is not None
                               else 0.0)

    def _ungraded_record(self, e, log, wall_s: float, aud,
                         e7: bool) -> dict:
        """--fast, ungraded token: the chain FLEW (program, fences, silence
        census, audit and cycles are all real) but NO golden compare ran.
        walked_macs is 0 — the walked-MAC ledger counts only work that
        graded bit-exact — and the walk record says UNGRADED in as many
        words. Cycles/walls are kept: they are measurements, not grades."""
        cyc = parse_cycles_e7(log) if e7 else wfl.parse_cycles(log)
        self.n_ungraded += 1
        walk = {"ungraded": FAST_UNGRADED_NOTE, "fuel_audit_ok": aud,
                "cycles": cyc, "build_s": round(e["t_build"], 2),
                "sim_wall_s": round(wall_s, 2)}
        if e7:
            walk["silence_att"], walk["silence"] = e["sil"], e["sil_b"]
        else:
            walk["silence"] = e["sil"]
        eprint(f"    [{e['tag']}] walked UNGRADED (--fast) tile_cyc="
               f"{cyc.get('walk_tile_cycles')}  build={e['t_build']:.1f}s "
               f"exec={wall_s:.1f}s")
        return {"walked_macs": 0, "walk": walk}

    # ── executor hooks (the ONLY executor-specific seams) ──────────────────
    def _execute(self, ctx, args, path, regions) -> dict:
        """sim: the verilated twin, DDR modelled from the image file.
        `path` is one file or the e7 [ATT, EPI] pair — one invocation
        either way (per-file TILE_RST is the executor contract)."""
        files = [str(p) for p in
                 (path if isinstance(path, (list, tuple)) else [path])]
        return bridge.run_job(files, executor="sim",
                              binary=str(ctx["binary"]),
                              extra_args=(f"+ddr_image={ctx['idir']}/"
                                          f"ddr_image.bin",
                                          f"+ddr_regions={regions}",
                                          "+fuel_audit"),
                              timeout_s=args.timeout, tile_div=wfl.TILE_DIV)

    def _audit(self, r, tag) -> int:
        """sim: the +fuel_audit plusarg prints '-> ok' per attested fetch."""
        aud = r["log"].count("-> ok")
        if aud < 1:
            raise SystemExit(f"REFUSE: {tag} no fuel-audit '-> ok' in the "
                             f"executor log — the fetch path is unattested")
        return aud

    # ── per (step, layer): build → fence → run → grade ─────────────────────
    def _prep_one(self, ctx, step: int, li: int, fx, xrow):
        """subject → parity fence → epilogue → descriptor → drift fence.
        STRICTLY sequential (the drift fence compares consecutive
        descriptors). Returns (entry-without-program, None) or
        (None, refusal_record)."""
        args, work = ctx["args"], ctx["work"]
        e7 = getattr(args, "mask", "b") == "e7"
        tag = f"t{step:03d}_L{li:02d}"
        try:
            sub = self._subject(ctx, li, fx, xrow, step)
        except SystemExit as e:
            # an operand this framing cannot identity-stage (amax != 127
            # frame). LOUD, reported, counted fully host-side.
            self.n_refused += 1
            eprint(f"    [{tag}] REFUSED (subject): {e}")
            return None, {"r2_bits": bits16(fx.r2), "walked_macs": 0,
                          "walk": None, "refused": str(e)}
        if (not self.parity_done and li == 0
                and list(ctx["ids"]) == list(fp.PROMPT_IDS)
                and step == len(ctx["ids"]) - 1):
            self._parity_fence(ctx, sub)
            self.parity_done = True

        t0 = time.monotonic()
        ge = wfl.golden_epilogue(sub["attn8"], sub["Wo"], sub["s_wo"],
                                 sub["xrow_bits"])
        desc = wfl.walk_descriptor(sub["bases"], mask=wfl.MASK_B,
                                   rq=(ge["scale"], ge["shift"]),
                                   comp=ge["comp"])
        if self.desc_prev is not None:
            moved = [w for w in range(fmt.DESC_WORDS)
                     if desc[w] != self.desc_prev[w]]
            if not set(moved) <= set(wl.REPOINT_WORDS):
                raise SystemExit(
                    f"REFUSE: {tag} descriptor moved words {moved} — more "
                    f"than the re-point set {list(wl.REPOINT_WORDS)}")
        self.desc_prev = desc
        entry = {"tag": tag, "li": li, "fx": fx, "sub": sub, "ge": ge,
                 "desc": desc, "work": str(work), "idir": str(ctx["idir"])}
        if e7:
            # the D-030 cores are the grade's arbiter — kept on the ENTRY
            # (main process), never pickled through the emission pool.
            entry["cores"] = sub.pop("cores")
            desc_att = e7_att_descriptor(sub["bases"], sub["head_rqs"],
                                         sub["T"], sub["pos"])
            if self.desc_att_prev is not None:
                prev_step, prev = self.desc_att_prev
                moved = [w for w in range(fmt.DESC_WORDS)
                         if desc_att[w] != prev[w]]
                allowed = set(ATT_REPOINT_WORDS)
                if step != prev_step:
                    allowed.add(fmt.W_STEP)
                if not set(moved) <= allowed:
                    raise SystemExit(
                        f"REFUSE: {tag} ATT descriptor moved words {moved} "
                        f"— more than the re-point set "
                        f"{sorted(allowed)} (step boundary: "
                        f"{step != prev_step})")
            self.desc_att_prev = (step, desc_att)
            entry["desc_att"] = desc_att
        entry["t_build"] = time.monotonic() - t0
        return entry, None

    def _build_one(self, ctx, step: int, li: int, fx, xrow):
        """prep + program emission, sequential — the per-layer path."""
        e, refusal = self._prep_one(ctx, step, li, fx, xrow)
        if refusal is not None:
            return None, refusal
        t0 = time.monotonic()
        if getattr(ctx["args"], "mask", "b") == "e7":
            em = _emit_program_e7(e["sub"], e["desc_att"], e["desc"],
                                  e["work"], f"walk_{e['tag']}", li,
                                  e["idir"], self.emit_cache)
            e.update(paths=[Path(p) for p in em["paths"]], sil=em["sil"],
                     sil_b=em["sil_b"], regions=em["regions"],
                     n_caps=em["n_caps"],
                     t_build=e["t_build"] + (time.monotonic() - t0))
            self._tally_emit(em)
            return e, None
        em = _emit_program(e["sub"], e["desc"], e["work"],
                           f"walk_{e['tag']}", li, e["idir"],
                           self.emit_cache)
        e.update(path=Path(em["path"]), sil=em["sil"],
                 regions=em["regions"], n_caps=em["n_caps"],
                 t_build=e["t_build"] + (time.monotonic() - t0))
        self._tally_emit(em)
        return e, None

    def _grade_one(self, e, captures, log, wall_s: float, aud) -> dict:
        """One layer's grade against golden — REFUSES on any mismatch.
        Identical semantics for per-layer and batched execution."""
        res_q, res_r1 = _grade_payload(captures, self.jobs,
                                       e["sub"]["h8"], e["sub"]["blocks"],
                                       e["ge"]["r1_bits"])
        return self._apply_grade(e, res_q, res_r1, log, wall_s, aud)

    def _apply_grade(self, e, res_q, res_r1, log, wall_s: float,
                     aud) -> dict:
        tag, sub, ge = e["tag"], e["sub"], e["ge"]
        if not res_q.get("all_bit_exact"):
            raise SystemExit(f"REFUSE: {tag} walked QKV NOT bit-exact "
                             f"({res_q.get('bad')} of {res_q.get('jobs')} "
                             f"jobs bad) — the tile diverged from golden")
        if not res_r1.get("bit_exact"):
            raise SystemExit(f"REFUSE: {tag} walked r1 NOT bit-exact "
                             f"({res_r1.get('bad')} of {res_r1.get('n')} "
                             f"elements bad) — the tile diverged from golden")
        cyc = wfl.parse_cycles(log)
        self.n_walked += 1
        # walked MACs: only accumulators the tile produced AND that graded
        # bit-exact — derived from the graded tensors' own shapes.
        qkv_macs = sum(int(b.shape[0]) * int(b.shape[1])
                       for b in sub["blocks"].values())
        oproj_macs = int(sub["Wo"].shape[0]) * int(sub["Wo"].shape[1])
        eprint(f"    [{tag}] QKV {res_q['jobs']}/144 jobs + r1 "
               f"{res_r1['n']}/896 BIT-EXACT  tile_cyc="
               f"{cyc.get('walk_tile_cycles')}  build={e['t_build']:.1f}s "
               f"exec={wall_s:.1f}s")
        return {"walked_macs": qkv_macs + oproj_macs,
                "walk": {"qkv": {k: v for k, v in res_q.items()
                                 if k != "per_job"},
                         "r1": {k: v for k, v in res_r1.items()
                                if k not in ("got_head", "want_head")},
                         "silence": e["sil"], "fuel_audit_ok": aud,
                         "cycles": cyc, "rq": [ge["scale"], ge["shift"]],
                         "comp": f"{ge['comp']:#010x}",
                         "build_s": round(e["t_build"], 2),
                         "sim_wall_s": round(wall_s, 2),
                         "walked_macs": {"qkv": qkv_macs,
                                         "oproj": oproj_macs},
                         "res1_f16_adds": int(sub["Wo"].shape[1])}}

    def _grade_one_e7(self, e, caps_att, caps_epi, log, wall_s: float,
                      aud) -> dict:
        """The e7 pair's grade — the §6 table of E7_TOKEN_LOOP.md, then the
        UNCHANGED QKV/r1 grades. The attention arbiter is the D-030
        recomposition (attention_core on the STAGED operands — see
        _subject); the layer output consumed stays fx.r2 through the
        unchanged bit gate. REFUSES on any mismatch."""
        tag, sub, ge = e["tag"], e["sub"], e["ge"]
        heads, T = e["cores"], int(sub["T"])
        lay = att_cap_layout(T)
        bpr8 = wfl.D_TILE // wfl.MXE_N
        # ── the fs/ss ladders, in production order ─────────────────────────
        fs = [int(t["bits"]) for t in cd.fp16_bits(caps_att)
              if t["sem"] == "fs"]
        ss = [int(t["bits"]) for t in cd.fp16_bits(caps_att)
              if t["sem"] == "ss"]
        want_fs = [int(h.s_q) for h in heads]
        for h in heads:
            want_fs += [int(v) for v in h.s_k]
            for _j in range(bpr8):
                want_fs += [int(v) for v in h.s_v]
        want_ss = [int(h.s_c) for h in heads]
        if fs != want_fs:
            bad = next(i for i, (a, b) in enumerate(zip(fs, want_fs))
                       if a != b) if len(fs) == len(want_fs) else "len"
            raise SystemExit(f"REFUSE: {tag} walked attention fs ladder NOT "
                             f"bit-exact (n={len(fs)}/{len(want_fs)}, first "
                             f"bad index {bad}) — s_q/s_k/s_v provenance "
                             f"diverged from golden")
        if ss != want_ss:
            raise SystemExit(f"REFUSE: {tag} walked s_c ladder NOT "
                             f"bit-exact ({ss[:4]}… vs {want_ss[:4]}…)")
        # ── ATT RO: the 14 heads' PV o8 blocks ─────────────────────────────
        ro = [c for c in caps_att if str(c.get("tag", "")).startswith("ro_")]
        if len(ro) != lay["ro"]:
            raise SystemExit(f"REFUSE: {tag} ATT file returned {len(ro)} RO "
                             f"caps, ledger says {lay['ro']}")
        o8_acc = cd.ro_lanes_to_i32(ro, strict=False).astype(np.int64)
        o8_bad = []
        for h_i, h in enumerate(heads):
            got = o8_acc[h_i * wfl.HEAD_DIM:(h_i + 1) * wfl.HEAD_DIM]
            if not np.array_equal(got, np.asarray(h.o8, dtype=np.int64)):
                o8_bad.append(h_i)
        if o8_bad:
            raise SystemExit(f"REFUSE: {tag} walked attention o8 NOT "
                             f"bit-exact (heads {o8_bad}) — the walked "
                             f"score/pv diverged from attention_core")
        # ── the MASK_B kick: QKV + r1, graded exactly as --mask b does ─────
        ro_b, lrd = wfl.split_caps(caps_epi)
        res_q = wfp.grade(ro_b, self.jobs, sub["h8"], sub["blocks"])
        if not res_q.get("all_bit_exact"):
            raise SystemExit(f"REFUSE: {tag} walked QKV NOT bit-exact "
                             f"({res_q.get('bad')} of {res_q.get('jobs')} "
                             f"jobs bad) — the tile diverged from golden")
        res_r1 = wfl.grade_r1(lrd, ge["r1_bits"])
        if not res_r1.get("bit_exact"):
            raise SystemExit(f"REFUSE: {tag} walked r1 NOT bit-exact "
                             f"({res_r1.get('bad')} of {res_r1.get('n')} "
                             f"elements bad) — the tile diverged from golden")
        cyc = parse_cycles_e7(log)
        self.n_walked += 1
        qkv_macs = sum(int(b.shape[0]) * int(b.shape[1])
                       for b in sub["blocks"].values())
        oproj_macs = int(sub["Wo"].shape[0]) * int(sub["Wo"].shape[1])
        # score (T x 64) + PV (64 x T) per head — graded THROUGH the o8
        # composition (acc_s -> softmax -> acc_o -> requant), the B1 rule.
        attn_macs = 2 * wfl.HEAD_DIM * T * wfl.N_HEADS
        eprint(f"    [{tag}] QKV {res_q['jobs']}/144 + attn o8 "
               f"{wfl.N_HEADS}x{wfl.HEAD_DIM} (T={T}) + r1 "
               f"{res_r1['n']}/896 BIT-EXACT  att_cyc="
               f"{cyc['att'].get('walk_tile_cycles')} epi_cyc="
               f"{cyc['epi'].get('walk_tile_cycles')} build="
               f"{e['t_build']:.1f}s exec={wall_s:.1f}s")
        return {"walked_macs": qkv_macs + oproj_macs + attn_macs,
                "walk": {"qkv": {k: v for k, v in res_q.items()
                                 if k != "per_job"},
                         "attn": {"bit_exact": True, "heads": wfl.N_HEADS,
                                  "T": T, "o8_graded": wfl.N_HEADS
                                  * wfl.HEAD_DIM,
                                  "fs_graded": len(fs),
                                  "ss_graded": len(ss),
                                  "rq_source": "host-loaded RQ[h] from "
                                               "the D-030 core (B1 scope)"},
                         "r1": {k: v for k, v in res_r1.items()
                                if k not in ("got_head", "want_head")},
                         "silence_att": e["sil"], "silence": e["sil_b"],
                         "fuel_audit_ok": aud,
                         "cycles": cyc, "rq": [ge["scale"], ge["shift"]],
                         "comp": f"{ge['comp']:#010x}",
                         "build_s": round(e["t_build"], 2),
                         "sim_wall_s": round(wall_s, 2),
                         "walked_macs": {"qkv": qkv_macs,
                                         "oproj": oproj_macs,
                                         "attn_score_pv": attn_macs},
                         "res1_f16_adds": int(sub["Wo"].shape[1])}}

    def layer(self, ctx, step: int, li: int, fx, xrow) -> dict:
        skip = self._probe_skip(step, li, fx)
        if skip is not None:
            return skip
        args = ctx["args"]
        e, refusal = self._build_one(ctx, step, li, fx, xrow)
        if refusal is not None:
            # NOTE (--ffn): an E-6-refused subject refuses the whole layer;
            # the FFN window is not flown for it either (counted host-side,
            # exactly as today). No such refusal occurs on the committed
            # prompt path.
            return refusal
        e7 = getattr(args, "mask", "b") == "e7"
        t0 = time.monotonic()
        r = self._execute(ctx, args, e["paths"] if e7 else e["path"],
                          e["regions"])
        t_exec = time.monotonic() - t0
        if not r["ok"]:
            raise SystemExit(f"REFUSE: {e['tag']} executor not ok "
                             f"(rc={r['rc']}, timed_out={r['timed_out']}) — "
                             f"log tail:\n" + r["log"][-2000:])
        aud = self._audit(r, e["tag"])
        graded = not ctx.get("fast_ungraded", False)
        if e7:
            n_a, n_b = e["n_caps"]
            caps = r["captures"]
            if len(caps) != n_a + n_b:
                raise SystemExit(f"REFUSE: {e['tag']} pair returned "
                                 f"{len(caps)} captures, programs bake "
                                 f"{n_a}+{n_b}")
            if graded:
                t_g = time.monotonic()
                g = self._grade_one_e7(e, caps[:n_a], caps[n_a:], r["log"],
                                       t_exec, aud)
                self.grade_s += time.monotonic() - t_g
            else:
                g = self._ungraded_record(e, r["log"], t_exec, aud, e7=True)
        elif graded:
            t_g = time.monotonic()
            g = self._grade_one(e, r["captures"], r["log"], t_exec, aud)
            self.grade_s += time.monotonic() - t_g
        else:
            g = self._ungraded_record(e, r["log"], t_exec, aud, e7=False)
        out = {"r2_bits": bits16(fx.r2), "walked_macs": g["walked_macs"],
               "walk": g["walk"], "refused": None}
        if self.ffn is not None:
            fr = self.ffn.run_one(ctx, step, li, fx)
            out["walked_macs"] += fr["macs_total"]
            out["walk"] = dict(out["walk"], ffn=fr)
        return out

    def finish(self, ctx) -> dict:
        e7 = getattr(ctx.get("args"), "mask", "b") == "e7"
        out = {"engine": self.name,
               "walked_steps": WALKED_STEPS_E7 if e7 else WALKED_STEPS,
               "layers_walked": self.n_walked,
               "layers_refused": self.n_refused,
               "binary": str(ctx.get("binary")),
               "image": str(ctx.get("idir")),
               "image_sha256": ctx.get("man", {}).get("image_sha256")}
        if e7:
            out["e7_fences"] = (
                "fetched-gamma NORM2 NOT walked at 896 (B-FEED-WIDTH "
                "per-frame feeder scales; capacity 42>31 with attention) — "
                "docs/design/E7_TOKEN_LOOP.md §2; per-head RQ host-loaded "
                "(B1 scope); o8->OPROJ-act seam is the disclosed "
                "host-staged copy; EN_STOREKV IS composed — the walker "
                "idle-polls + re-addresses both engines (K@T-1/V@2T-1) "
                "through its own sk_g select and refuses the walk on a "
                "sick engine — but the walked store step moves NO data "
                "(it paces records the walk itself would produce, and a "
                "live-T QKV walk wedges; walk mode holds every host job "
                "port off; the CQ-8 engine store address never "
                "auto-advances), so KV record DATA stays host-staged per "
                "step under ONE quasi-static l_kv_map level per engine — "
                "the remaining host-branch dependence the l_kv_map RTL "
                "hardening (next image) makes silicon-true; post-walk OCC "
                "verified on engine 0 only (a folded host select would "
                "alias engine 0 and green a per-engine readback wrongly)")
        if self.ffn is not None:
            out["walked_steps"] = ffn_walked_steps(self.ffn.S,
                                                   self.ffn.d_ffn)
            out["ffn"] = {
                "slice": {"d_ffn_walk": self.ffn.S,
                          "d_ffn": self.ffn.d_ffn,
                          "chunks": self.ffn.chunks},
                "ffn_windows_walked": self.ffn.n_walked,
                "ffn_windows_refused": self.ffn.n_refused,
                "image": str(self.ffn.idir),
                "image_sha256": self.ffn.man_sha,
                "seam": FFN_SEAM_NOTE,
                **({"hw_ddr_base_bytes": FFN_HW_DDR_BASE,
                    "hw_ddr_attestation":
                        "operator: f2_ddr_load --load --verify "
                        "--full-verify on the REBASED slice image at "
                        f"{FFN_HW_DDR_BASE:#x}, alongside the main image "
                        "(--ffn-ddr-attested)",
                    "hw_fuel_audit":
                        "hw: no sim '+fuel_audit' printer — chain is "
                        "full-verified slice-image DDR + program baked "
                        "checks + bit-exact grades",
                    "hw_timing_label":
                        "no CYCMARK on hw — no ffn_tile_cycles window is "
                        "claimed; sim_wall_s keys are hw wall clock"}
                   if self.ffn.hw else {})}
        if self.emit_cache is not None:
            tpl = [r["emit_s"] for r in self.emit_steps
                   if r["modes"].get("template")]
            pat = [r["emit_s"] for r in self.emit_steps
                   if r["modes"].get("patched")
                   and not r["modes"].get("template")]
            ratio = (round((sum(tpl) / len(tpl)) / (sum(pat) / len(pat)), 2)
                     if tpl and pat and sum(pat) > 0 else None)
            out["emit_cache"] = {
                "mode": self.emit_cache["mode"],
                "dir": self.emit_cache["dir"],
                "per_step_emission": self.emit_steps,
                "template_step_vs_patched_step_emission_ratio": ratio,
                "verify_byte_identical": (
                    f"{self.emit_verify[0]}/{self.emit_verify[1]}"
                    if self.emit_cache["mode"] == "verify" else None),
                "note": "emission = program build wall (emit_s); verify "
                        "overhead (full rebuild + byte compare) is "
                        "recorded separately and excluded from the ratio"}
            if self.emit_cache["mode"] == "verify":
                print(f"EMIT-CACHE VERIFY: {self.emit_verify[0]}/"
                      f"{self.emit_verify[1]} byte-identical")
            if ratio is not None:
                print(f"EMIT-CACHE RATIO: template-step emission / "
                      f"patched-step emission = {ratio}x")
        if self.probe_steps is not None:
            out["probe"] = {
                "steps": sorted(self.probe_steps),
                "layers": ("all" if self.probe_layers is None
                           else sorted(self.probe_layers)),
                "layer_slots_skipped": self.n_skipped,
                "note": "probe mode: ONLY the named (step, layer) chains "
                        "were walked; skipped slots are host-vouched "
                        "golden, counted in no walked total"}
        if getattr(ctx.get("args"), "fast", None) is not None:
            out["fast"] = {
                "n": int(ctx["args"].fast),
                "layers_walked_ungraded": self.n_ungraded,
                "note": "FAST MODE (--fast): layers_walked counts GRADED "
                        "chains only; the ungraded chains FLEW but carry "
                        "NO bit-exactness claim, and no reference pass "
                        "ran (token identity NOT checkable)"}
        return out


class HwWalkedEngine(SimWalkedEngine):
    """The SAME per-layer walked chain, executed on the REAL tile: run this
    ON the F2 instance (the hw executor attaches slot 0 directly — see
    tile_exec_bridge.run_job(executor='hw')). Program builders, fences and
    bit-exact grades are SimWalkedEngine's verbatim; only the executor seam
    and its attestations differ:

      * DDR: the walk fetches from PHYSICAL DDR. The image must have been
        loaded and FULL-VERIFIED on this card first
        (f2_ddr_load.py --image <image> --load --verify --full-verify) —
        this engine re-checks the manifest identity but takes the load
        attestation from the operator via --ddr-attested (recorded, and
        REFUSED if absent, so a report can never silently omit it).
      * fuel audit: there is no sim-side '+fuel_audit' printer on hw. The
        attestation chain is: full-verified DDR at load + the program's own
        baked checks (run_job 'ok' requires the executor summary) + every
        walked value bit-exact against golden. Recorded in as many words.
      * cycles: no CYCMARK on hw — parse_cycles returns no window. What IS
        real on this engine is the WALL CLOCK: sim_wall_s is renamed
        hw_wall_s in spirit (the JSON key stays for schema parity; the
        finish() record labels it), and the harness's tokens/s is a
        MEASUREMENT of this hybrid (walked steps on silicon @ the loaded
        clock recipe + host steps in python on the card's CPU).
    """
    name = "hw-walked"

    def start(self, ctx) -> None:
        args, model = ctx["args"], ctx["model"]
        idir = Path(args.image)
        man_p = idir / "ddr_image.json"
        if not man_p.is_file():
            raise SystemExit(f"REFUSE: no DDR image manifest at {man_p}")
        man = json.loads(man_p.read_text())
        have, need = set(man["layers"]), set(range(model.n_layers))
        if not need <= have:
            raise SystemExit(f"REFUSE: image {idir} holds layers "
                             f"{sorted(have)}; the loop needs {sorted(need)}")
        wdir_img = Path(man["weights_dir"]).resolve()
        wdir_mod = Path(args.weights_dir).resolve()
        if wdir_img != wdir_mod:
            raise SystemExit(
                f"REFUSE: image weights_dir {wdir_img} != --weights-dir "
                f"{wdir_mod} — the walked fetches and the golden grades "
                f"would be about different tensors")
        if not getattr(args, "ddr_attested", False):
            raise SystemExit(
                "REFUSE: hw-walked needs --ddr-attested — run "
                "f2_ddr_load.py --image <image> --load --verify "
                "--full-verify on THIS card first, then pass the flag; "
                "the report records the attestation you are making")
        for t in man["tensors"]:
            self.bases.setdefault(t["layer"], {})[t["tensor"]] = t["base_64B"]
        ctx["man"], ctx["idir"], ctx["binary"] = man, idir, None
        self._arm_probes(args)
        self._arm_emit_cache(ctx)
        eprint(f"[hw-walked] image {idir} sha={man['image_sha256'][:16]}… "
               f"slot {getattr(args, 'slot', 0)} (physical DDR, "
               f"operator-attested full verify)")
        if getattr(args, "ffn", False):
            # the SAME lane the sim engine arms — prepare() switches its
            # executor seam to hw (rebased slice image + its own
            # attestation) off args.engine; builder/fences/grade unchanged.
            self.ffn = FfnLane(getattr(args, "ffn_chunks", FFN_CHUNKS_MAX))
            self.ffn.prepare(ctx)

    def __init__(self):
        super().__init__()
        self._pending = []          # built-not-yet-flown entries (batch mode)
        self._pool = None           # persistent emission/grading pool

    def _get_pool(self, n: int):
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=min(12, max(n, 1)))
        return self._pool

    def _execute(self, ctx, args, path, regions) -> dict:
        # regions is a sim-side fetch restriction; physical DDR holds the
        # whole verified image — nothing to restrict.
        files = [str(p) for p in
                 (path if isinstance(path, (list, tuple)) else [path])]
        return bridge.run_job(files, executor="hw",
                              slot=getattr(args, "slot", 0),
                              timeout_s=args.timeout)

    # ── batched transport: 24 layer programs, ONE executor invocation ──────
    # Sound because the walked chains of one decode step are independent
    # probes: layer li+1's subject comes from GOLDEN's composition (fx),
    # never from the walked outputs — see the seam statement in the header.
    # The grades still complete BEFORE any token is sampled: decode_pass
    # calls flush_step() after the layer loop and before head/argmax, and
    # any mismatch REFUSES the run right there.
    def layer(self, ctx, step: int, li: int, fx, xrow) -> dict:
        skip = self._probe_skip(step, li, fx)
        if skip is not None:
            return skip
        if getattr(ctx["args"], "no_batch", False):
            return super().layer(ctx, step, li, fx, xrow)
        e, refusal = self._prep_one(ctx, step, li, fx, xrow)
        if refusal is not None:
            return refusal
        rec = {"r2_bits": bits16(fx.r2), "walked_macs": 0,
               "walk": None, "refused": None}
        e["rec"] = rec
        self._pending.append(e)
        return rec

    def flush_step(self, ctx, step: int) -> None:
        if not self._pending:
            self._print_emit_step(step)
            return
        args = ctx["args"]
        e7 = getattr(args, "mask", "b") == "e7"
        pend, self._pending = self._pending, []
        # program emission fans out across CPUs — order-independent by
        # construction (every fence that needs order ran in _prep_one).
        # emit_cache rides along: each worker loads/patches the run's
        # per-layer template from the run-scoped cache dir (emit_cache._MEM
        # is per-process; the disk template is the shared truth).
        t0 = time.monotonic()
        pool = self._get_pool(len(pend))
        if e7:
            ems = list(pool.map(
                _emit_program_e7,
                [e["sub"] for e in pend], [e["desc_att"] for e in pend],
                [e["desc"] for e in pend], [e["work"] for e in pend],
                [f"walk_{e['tag']}" for e in pend],
                [e["li"] for e in pend], [e["idir"] for e in pend],
                [self.emit_cache] * len(pend)))
        else:
            ems = list(pool.map(
                _emit_program,
                [e["sub"] for e in pend], [e["desc"] for e in pend],
                [e["work"] for e in pend],
                [f"walk_{e['tag']}" for e in pend],
                [e["li"] for e in pend], [e["idir"] for e in pend],
                [self.emit_cache] * len(pend)))
        t_emit = time.monotonic() - t0
        for e, em in zip(pend, ems):
            if e7:
                e.update(paths=[Path(p) for p in em["paths"]], sil=em["sil"],
                         sil_b=em["sil_b"], regions=em["regions"],
                         n_caps=em["n_caps"])
            else:
                e.update(path=Path(em["path"]), sil=em["sil"],
                         regions=em["regions"], n_caps=em["n_caps"])
            e["t_build"] += t_emit / len(pend)
            self._tally_emit(em)
        files = ([str(p) for e in pend for p in e["paths"]] if e7
                 else [str(e["path"]) for e in pend])
        t0 = time.monotonic()
        r = bridge.run_job(files, executor="hw",
                           slot=getattr(args, "slot", 0),
                           timeout_s=args.timeout)
        wall = time.monotonic() - t0
        if not r["ok"]:
            raise SystemExit(f"REFUSE: step {step} batched executor not ok "
                             f"(rc={r['rc']}, timed_out={r['timed_out']}) — "
                             f"log tail:\n" + r["log"][-2000:])
        caps, off, slices = r["captures"], 0, []
        for e in pend:
            n_e = sum(e["n_caps"]) if e7 else e["n_caps"]
            sl = caps[off:off + n_e]
            if len(sl) != n_e:
                raise SystemExit(
                    f"REFUSE: {e['tag']} batch produced {len(sl)} captures, "
                    f"program bakes {n_e} — a file aborted mid-batch "
                    f"(log tail):\n" + r["log"][-2000:])
            off += n_e
            aud = self._audit(r, e["tag"])
            graded = not ctx.get("fast_ungraded", False)
            # TODO(lever-2b): re-pool this grading over the persistent pool
            # (the pre-e7 flush pooled _grade_payload; worth ~0.1-0.2 s/tok)
            if not graded:
                g = self._ungraded_record(e, r["log"], wall / len(pend),
                                          aud, e7=e7)
            elif e7:
                n_a = e["n_caps"][0]
                t_g = time.monotonic()
                g = self._grade_one_e7(e, sl[:n_a], sl[n_a:], r["log"],
                                       wall / len(pend), aud)
                self.grade_s += time.monotonic() - t_g
            else:
                t_g = time.monotonic()
                g = self._grade_one(e, sl, r["log"], wall / len(pend), aud)
                self.grade_s += time.monotonic() - t_g
            e["rec"]["walked_macs"] = g["walked_macs"]
            e["rec"]["walk"] = dict(g["walk"], batched=len(pend))
        if off != len(caps):
            raise SystemExit(f"REFUSE: step {step} batch has {len(caps)-off} "
                             f"unattributed captures beyond the {off} the "
                             f"programs bake — splitter and programs disagree")
        if self.ffn is not None:
            # --ffn: the walked FFN→DOWN window per layer — one ADDITIONAL
            # executor invocation each, AFTER the batched E-6 flush and its
            # grades (the sim engine's per-layer ordering, kept: chain
            # graded first, then the FFN window on the same fx). Still
            # strictly before decode_pass samples a token from this step —
            # flush_step() runs pre-head/argmax, and any mismatch REFUSES
            # here. A subject the E-6 prep refused never reached _pending,
            # so its FFN window is not flown either (counted host-side,
            # exactly the sim NOTE in layer()).
            for e in pend:
                fr = self.ffn.run_one(ctx, step, e["li"], e["fx"])
                e["rec"]["walked_macs"] += fr["macs_total"]
                e["rec"]["walk"] = dict(e["rec"]["walk"], ffn=fr)
        self._print_emit_step(step)

    def _audit(self, r, tag) -> int:
        # No '-> ok' printer on hw. run_job's ok-contract (summary n ==
        # records) + the program's own baked checks already gated above;
        # the DDR attestation is start()'s. Record 0 sim-audit lines.
        return 0

    def finish(self, ctx) -> dict:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if hasattr(bridge, "shutdown_daemons"):
            # release the persistent hw executor (one PCI attach + one
            # python process) with the engine; atexit is the backstop.
            bridge.shutdown_daemons()
        out = super().finish(ctx)
        out["engine"] = self.name
        out["executor"] = "hw (f2_host_run on this instance)"
        out["ddr_attestation"] = ("operator: f2_ddr_load --load --verify "
                                  "--full-verify on this card before the run "
                                  "(--ddr-attested)")
        out["fuel_audit"] = ("hw: no sim '+fuel_audit' printer — chain is "
                             "full-verified DDR + program baked checks + "
                             "bit-exact grades")
        out["timing_label"] = ("tokens/s here is a MEASUREMENT of the hybrid "
                               "(walked {FPROJ,QKV,OPROJ,RES1} on silicon at "
                               "the loaded clock recipe; all other steps "
                               "python-on-card-CPU). No CYCMARK on hw; no "
                               "tile-cycle window is claimed.")
        return out


ENGINES = {"host-golden": HostGoldenEngine, "sim-walked": SimWalkedEngine,
           "hw-walked": HwWalkedEngine}


# ═══════════════════ the harness's own bit gate ════════════════════════════

def check_engine_bits(tag: str, engine_bits: np.ndarray,
                      golden_bits: np.ndarray) -> None:
    """The layer-output gate: whatever an engine vouches for must BE
    golden's value, bit for bit, before anything downstream consumes it."""
    e = np.asarray(engine_bits, dtype=np.uint16).ravel()
    g = np.asarray(golden_bits, dtype=np.uint16).ravel()
    if e.shape != g.shape or not np.array_equal(e, g):
        bad = (int(np.count_nonzero(e != g)) if e.shape == g.shape
               else "shape")
        raise SystemExit(f"REFUSE: {tag}: engine layer output != golden r2 "
                         f"({bad} bits16 differ) — bit-exact discipline "
                         f"violated; no token is sampled from this value")


# ═══════════════════ the decode loop (one engine pass) ═════════════════════

def decode_pass(model, ids: list[int], n_tokens: int, tier: str, group: int,
                eos: set, engine, ctx, census: MacCensus,
                expect_ids: list[int] | None, quiet: bool = False,
                fast_n: int | None = None) -> dict:
    """Greedy decode mirroring prompt_offload.decode's token semantics,
    with the engine seam at every sampling-relevant step.

    Steps consuming ids[:-1] build context (host-golden by definition —
    they produce no token). Every step from the last prompt token onward
    IS sampling-relevant: its hidden state becomes a token. Those steps
    run through the engine, layer by layer.

    fast_n (default None = today's path, untouched): --fast N. Tokens with
    i % N != 0 are flagged ungraded via ctx['fast_ungraded'] — the walked
    engines then skip the post-flight golden compares for those tokens'
    chains (the chains still fly). golden's sess.step is NEVER skipped:
    the engine subjects derive from its LayerFx per layer.
    """
    L = model.n_layers
    sess = rt.Session(model, tier, group)
    t_start = time.monotonic()
    prefill_s = []
    for tid in ids[:-1]:
        t0 = time.monotonic()
        with census:
            sess.step(model.embed_row(tid))
        prefill_s.append(round(time.monotonic() - t0, 3))
    cur = int(ids[-1])
    gen: list[int] = []
    tokens: list[dict] = []
    for i in range(n_tokens):
        t_tok = time.monotonic()
        ungraded = _fast_ungraded(i, fast_n)
        if fast_n is not None:
            ctx["fast_ungraded"] = ungraded
        # (1) embedding of the current token id (host)
        t0 = time.monotonic()
        x = model.embed_row(cur)
        embed_s = time.monotonic() - t0
        # (2) the decode step — golden composition + per-layer collection
        step = sess.pos
        fxs: dict[int, object] = {}
        marks: list[float] = []

        def _coll(li, r):
            fxs[li] = r
            marks.append(time.monotonic())

        t0 = time.monotonic()
        with census:
            hidden = sess.step(x, collect=_coll)
        golden_step_s = time.monotonic() - t0
        if sorted(fxs) != list(range(L)):
            raise SystemExit(f"REFUSE: step {step} collected layers "
                             f"{sorted(fxs)}, not all {L}")
        golden_layer_s = [round(b - a, 4) for a, b in
                          zip([t0] + marks[:-1], marks)]
        # the engine seam, layer by layer (outside the census: engine-side
        # golden grading must not contaminate the measured denominator)
        xrow = np.asarray(x, dtype=np.float64)
        layer_recs = []
        # the grade split (the measurement hook): the engine accumulates
        # its post-flight grading wall in grade_seconds(); the delta over
        # this token separates engine (build+fly) from grade time so a
        # --fast gain is attributable. 0 for engines without the hook.
        _gs = getattr(engine, "grade_seconds", None)
        g_before = _gs() if _gs is not None else 0.0
        t0 = time.monotonic()
        for li in range(L):
            rec = engine.layer(ctx, step, li, fxs[li], xrow)
            check_engine_bits(f"step {step} layer {li}", rec["r2_bits"],
                              bits16(fxs[li].r2))
            layer_recs.append(rec)
            xrow = np.asarray(fxs[li].r2, dtype=np.float64)
        if hasattr(engine, "flush_step"):
            # batched engines execute + grade the whole step HERE — still
            # strictly before any token is sampled from these values.
            engine.flush_step(ctx, step)
        engine_s = time.monotonic() - t0
        grade_s = (_gs() if _gs is not None else 0.0) - g_before
        check_engine_bits(f"step {step} hidden", layer_recs[-1]["r2_bits"],
                          bits16(hidden))
        # (3) final norm + lm_head + greedy argmax (host)
        t0 = time.monotonic()
        with census:
            logits = model.head_logits(hidden)
        tok = int(np.argmax(logits))
        head_s = time.monotonic() - t0
        # the cross-engine token gate
        if expect_ids is not None:
            if i >= len(expect_ids) or tok != int(expect_ids[i]):
                raise SystemExit(
                    f"REFUSE: token {i + 1} under engine {engine.name!r} is "
                    f"{tok} but host-golden sampled "
                    f"{expect_ids[i] if i < len(expect_ids) else '<none>'} "
                    f"— token identity violated")
        gen.append(tok)
        wall = time.monotonic() - t_tok
        rec_t = {
            "index": i + 1, "consumed_id": cur, "token_id": tok,
            "step": step, "T": step + 1,
            "logits_argmax_bits": int(np.float64(logits[tok])
                                      .view(np.int64)),
            "wall_s": round(wall, 3),
            # engine and grade are DISJOINT: engine = the seam's wall minus
            # the post-flight golden grading measured inside the engine.
            "phases_s": {"embed": round(embed_s, 4),
                         "golden_layers": round(golden_step_s, 3),
                         "engine": round(engine_s - grade_s, 3),
                         "grade": round(grade_s, 3),
                         "head_sample": round(head_s, 3)},
            "golden_layer_s": golden_layer_s,
            "layers": [{"layer": li,
                        "walked_macs": r["walked_macs"],
                        "refused": r["refused"],
                        "skipped": r.get("skipped"),
                        "walk": r["walk"]}
                       for li, r in enumerate(layer_recs)],
        }
        if fast_n is not None:
            rec_t["fast"] = {"graded": not ungraded}
        tokens.append(rec_t)
        if not quiet:
            eprint(f"  [token {i + 1}/{n_tokens}] step={step} T={step + 1} "
                   f"consumed={cur} -> sampled={tok}  wall={wall:.2f}s "
                   f"(golden {golden_step_s:.2f}s + engine "
                   f"{engine_s - grade_s:.2f}s + grade {grade_s:.2f}s "
                   f"+ head {head_s:.2f}s)"
                   + ("  [FAST: UNGRADED — no bit-exactness claim]"
                      if ungraded else ""))
        if tok in eos:
            if not quiet:
                eprint("  [token] EOS — stopping")
            break
        cur = tok
    return {"ids": gen, "tokens": tokens, "prefill_s": prefill_s,
            "wall_s": round(time.monotonic() - t_start, 3)}


# ═══════════════════ the measured MAC split ════════════════════════════════

def verify_census(census: MacCensus, model, steps: list[int],
                  n_head_calls: int) -> None:
    """The census is an observation; this is the independent second opinion
    (prompt05b.MacCensus.verify_against's rule, restated for a MULTI-token
    run: that method assumes exactly one lm_head call). Each counted step
    must show exactly 7 outermost projection gemms per layer with the
    model's own tensor shapes, and the outside-layer total must be exactly
    n_head_calls x vocab x D_model. REFUSES on any disagreement."""
    order = ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")
    for step in steps:
        seen: dict[int, list] = {}
        for r in census.gemm:
            if r["where"] is not None and r["where"][0] == step:
                seen.setdefault(r["where"][1], []).append(r)
        if sorted(seen) != list(range(model.n_layers)):
            raise SystemExit(f"REFUSE: census step {step} saw layers "
                             f"{sorted(seen)}, not all {model.n_layers}")
        for li, rows in seen.items():
            if len(rows) != len(order):
                raise SystemExit(f"REFUSE: census step {step} layer {li} "
                                 f"made {len(rows)} outermost gemms, "
                                 f"expected {len(order)}")
            for tag, r in zip(order, rows):
                sh = tuple(np.asarray(getattr(model.layers[li], tag)).shape)
                if (r["K"], r["N"]) != sh:
                    raise SystemExit(f"REFUSE: census step {step} layer "
                                     f"{li} counted {tag} as K={r['K']} "
                                     f"N={r['N']}, tensor is {sh}")
    vocab, dm = int(np.asarray(model.head_w8).shape[0]), \
        int(model.meta["D_model"])
    want = n_head_calls * vocab * dm
    if census.outside_layer_macs != want:
        raise SystemExit(f"REFUSE: lm_head census "
                         f"{census.outside_layer_macs} != {n_head_calls} "
                         f"calls x {vocab}x{dm} = {want}")


def mac_split(census: MacCensus, model, tok_rec: dict,
              mask: str = "b", walked_label: str | None = None) -> dict:
    """One token's honest MAC split. Two denominators, both stated:

    census   — every MAC golden ACTUALLY performed for this step + one
               lm_head call (measured by rebinding golden's own globals;
               includes golden's per-step K/V recompute over the row
               history — the committed composition's own shape).
    kv-cached— the same step if K/V for context rows were cached (the
               deployment shape): current-row QKV + attention + OPROJ +
               FFN + lm_head, derived from the model's own tensor shapes
               and the census's measured attention MACs.
    """
    step = tok_rec["step"]
    sm = census.step_macs(step)
    vocab, dm = int(np.asarray(model.head_w8).shape[0]), \
        int(model.meta["D_model"])
    head = vocab * dm
    census_total = sm["proj"] + sm["attn"] + head
    per_layer_1row = {}
    for tag in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"):
        sh = tuple(np.asarray(getattr(model.layers[0], tag)).shape)
        per_layer_1row[tag] = sh[0] * sh[1]
    kv_cached_total = (model.n_layers * sum(per_layer_1row.values())
                      + sm["attn"] + head)
    walked = sum(l["walked_macs"] for l in tok_rec["layers"])
    walked_layers = sum(1 for l in tok_rec["layers"]
                        if l["walked_macs"] > 0)
    return {
        "walked_macs": walked,
        "walked_layers": walked_layers,
        "refused_layers": [l["layer"] for l in tok_rec["layers"]
                           if l["refused"]],
        "census": {"proj": sm["proj"], "attn": sm["attn"],
                   "lm_head": head, "total": census_total,
                   "note": "measured by MacCensus (golden's own calls); "
                           "includes golden's per-step K/V recompute over "
                           "the row history"},
        "kv_cached": {"total": kv_cached_total,
                      "note": "shape-derived deployment denominator: "
                              "current-row projections only + measured "
                              "attention + lm_head"},
        "walked_frac_of_census": round(walked / census_total, 6),
        "walked_frac_of_kv_cached": round(walked / kv_cached_total, 6),
        "host_side": HOST_STEPS_E7 if mask == "e7" else HOST_STEPS,
        "walked_side": ((walked_label if walked_label is not None
                         else (WALKED_STEPS_E7 if mask == "e7"
                               else WALKED_STEPS))
                        if walked else "none"),
    }


# ═══════════════════════════════ run ═══════════════════════════════════════

def run(args) -> int:
    if args.engine not in ENGINES:
        raise SystemExit(f"REFUSE: unknown engine {args.engine!r} — have "
                         f"{sorted(ENGINES)}")
    ffn_on = bool(getattr(args, "ffn", False))
    check_ffn_composition(args.engine, getattr(args, "mask", "b"),
                          getattr(args, "probe_steps", None), ffn_on)
    fast_n = fast_gate(args)
    ids = parse_ids(args.prompt)
    n_total = len(ids) + args.tokens
    if args.tokens < 1:
        raise SystemExit("REFUSE: --tokens must be >= 1")
    probe_steps, probe_layers = parse_probes(args)
    if probe_steps is not None:
        # sampling-relevant steps are len(ids)-1 .. len(ids)-1+tokens-1;
        # anything below is host prefill of ids[:-1] — a probe there would
        # silently never walk, which is exactly the failure mode probe mode
        # exists to avoid.
        lo, hi = len(ids) - 1, len(ids) - 1 + args.tokens - 1
        bad = sorted(s for s in probe_steps if s < lo or s > hi)
        if bad:
            raise SystemExit(
                f"REFUSE: --probe-steps {bad} outside this run's "
                f"sampling-relevant steps [{lo}..{hi}] (steps < {lo} are "
                f"host prefill of ids[:-1] and never reach the engine)")
    if args.engine in ("sim-walked", "hw-walked"):
        limit, label = t_budget(args.engine, probe_steps is not None)
        if n_total > limit:
            hint = ("a FULL sim-walked decode is 24 Verilator flights per "
                    "token; deep-T sim evidence goes through --probe-steps "
                    "(fenced by T_HW_MAX instead)"
                    if limit == T_SIM_MAX else
                    "the walk-envelope budget — the cited RTL ceilings "
                    "live at T_HW_MAX's definition at the top of this file")
            raise SystemExit(f"REFUSE: prompt({len(ids)}) + tokens"
                             f"({args.tokens}) = {n_total} > {limit} "
                             f"[{label}] — {hint}")
    if n_total > rt.T_SESSION_MAX:
        raise SystemExit(f"REFUSE: {n_total} > T_SESSION_MAX="
                         f"{rt.T_SESSION_MAX}")
    tier = rt.TIER_MAP[args.tier]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    model = rt.GoldenModel(Path(args.weights_dir))
    if probe_layers is not None:
        badl = sorted(li for li in probe_layers if li >= model.n_layers)
        if badl:
            raise SystemExit(f"REFUSE: --probe-layers {badl} out of range "
                             f"— model has layers 0..{model.n_layers - 1}")
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    vocab = int(np.asarray(model.head_w8).shape[0])
    tok = None
    try:
        tok = rt.load_tokenizer(model.meta["model"])
    except Exception as e:                                   # noqa: BLE001
        eprint(f"[tokenizer] unavailable ({type(e).__name__}) — token ids "
               f"only, no text decode")

    print("=" * 78)
    print("TOKEN LOOP — Qwen2.5-0.5B greedy decode, pluggable per-layer "
          "engine")
    print(f"  engine={args.engine}  mask={getattr(args, 'mask', 'b')}  "
          f"prompt={ids} (+{args.tokens} new = "
          f"T<={n_total})  tier={tier} G={args.group}  L={model.n_layers} "
          f"vocab={vocab}")
    if tok:
        print(f"  prompt text: {tok.decode(ids)!r}")
    print("=" * 78)

    # ── pass 1: host-golden reference (the DEFAULT; --fast N skips it) ─────
    # For engine != host-golden this is the arbiter sequence every sampled
    # token is compared against. For engine == host-golden it makes the run
    # a two-independent-sessions consistency check. fast_n is None on every
    # run without --fast, so the reference pass ALWAYS runs there —
    # today's path, unchanged.
    ref = None
    if fast_n is None:
        eprint("[pass 1/2] host-golden reference decode …")
        ref_census = MacCensus(model.n_layers)
        ref_engine = HostGoldenEngine()
        ref_ctx = {"model": model, "args": args, "work": work, "ids": ids}
        ref_engine.start(ref_ctx)
        ref = decode_pass(model, ids, args.tokens, tier, args.group, eos,
                          ref_engine, ref_ctx, ref_census, None, quiet=True)
        eprint(f"[pass 1/2] reference ids: {ref['ids']} "
               f"({ref['wall_s']}s)")
    else:
        eprint(f"[FAST MODE] --fast {fast_n}: pass-1 host-golden reference "
               f"decode SKIPPED — token identity vs the reference is NOT "
               f"checkable in this run")

    # ── pass 2: the requested engine, graded against pass 1 ────────────────
    eprint(f"[pass 2/2] engine={args.engine} decode …")
    census = MacCensus(model.n_layers)
    engine = ENGINES[args.engine]()
    ctx = {"model": model, "args": args, "work": work, "ids": ids}
    engine.start(ctx)
    res = decode_pass(model, ids, args.tokens, tier, args.group, eos,
                      engine, ctx, census,
                      expect_ids=(ref["ids"] if ref is not None else None),
                      fast_n=fast_n)
    if ref is not None and res["ids"] != ref["ids"]:
        raise SystemExit(f"REFUSE: engine ids {res['ids']} != host-golden "
                         f"ids {ref['ids']}")
    if probe_steps is not None:
        ran = {t["step"] for t in res["tokens"]}
        missed = sorted(set(probe_steps) - ran)
        if missed:
            raise SystemExit(
                f"REFUSE: probe steps {missed} never ran — the decode "
                f"ended at steps {sorted(ran)} (EOS before the probe); "
                f"re-aim --probe-steps or change the prompt")
    # run-to-run consistency (both passes are golden compositions): the
    # sampled logits bits must agree token for token, not just the argmax.
    # (--fast: no reference pass exists, so there is nothing to compare —
    # the fast label states the unmet check in as many words.)
    if ref is not None:
        for a, b in zip(ref["tokens"], res["tokens"]):
            if a["logits_argmax_bits"] != b["logits_argmax_bits"]:
                raise SystemExit(f"REFUSE: token {a['index']} sampled-logit "
                                 f"bits differ between the two passes — the "
                                 f"pipeline is not deterministic")
    engine_summary = engine.finish(ctx)

    # ── the measured census, verified against the model's own shapes ───────
    steps = [t["step"] for t in res["tokens"]]
    verify_census(census, model,
                  steps=list(range(len(ids) - 1)) + steps,
                  n_head_calls=len(res["tokens"]))
    walked_label = None
    if ffn_on:
        walked_label = ffn_walked_steps(
            int(args.ffn_chunks) * wfl.D_TILE, int(model.meta["d_ffn"]))
    splits = [mac_split(census, model, t, mask=getattr(args, "mask", "b"),
                        walked_label=walked_label)
              for t in res["tokens"]]

    # ── timing + (sim-walked) the labelled prediction ──────────────────────
    n_gen = len(res["ids"])
    decode_wall = sum(t["wall_s"] for t in res["tokens"])
    golden_wall = sum(t["phases_s"]["golden_layers"]
                      + t["phases_s"]["head_sample"]
                      + t["phases_s"]["embed"] for t in res["tokens"])
    timing = {
        "prefill_s": res["prefill_s"],
        "per_token_wall_s": [t["wall_s"] for t in res["tokens"]],
        "decode_wall_s": round(decode_wall, 3),
        "host_golden_pipeline_wall_s": round(golden_wall, 3),
        "tokens_per_s_host_wall": {
            "value": round(n_gen / golden_wall, 4) if golden_wall else None,
            "LABEL": "MEASURED on this host — the golden PIPELINE's wall "
                     "(embed + 24 golden layers + head per token). A host "
                     "software number, NOT a hardware/tile claim."},
    }
    if args.engine == "sim-walked":
        timing["tokens_per_s_including_sim"] = {
            "value": round(n_gen / decode_wall, 5) if decode_wall else None,
            "LABEL": "Verilator wall included — a property of the "
                     "SIMULATOR on this host. Meaningless as a hardware "
                     "number; recorded only for run bookkeeping."}
        per_tok_pred = []
        for t in res["tokens"]:
            rows = [{"cycles": l["walk"]["cycles"],
                     "wall_s": l["walk"]["sim_wall_s"]}
                    for l in t["layers"] if l["walk"]]
            if getattr(args, "mask", "b") == "e7":
                per_tok_pred.append(predict_e7(rows, model.n_layers))
            else:
                per_tok_pred.append(wl.predict(rows, model.n_layers))
        timing["walked_window_prediction_per_token"] = per_tok_pred
        if per_tok_pred and per_tok_pred[0].get("prediction"):
            w = per_tok_pred[0]["prediction"]["walked_steps_only_per_token"]
            print(f"  walked window/token: {w['tile_cycles']} tile cyc = "
                  f"{w['wall_ms_at_15p625MHz_flown']} ms @ flown "
                  f"15.625 MHz ({w['covers']})")
        if ffn_on:
            timing["ffn_window_tile_cycles_per_token"] = {
                "value": [sum(l["walk"]["ffn"]["cycles"]
                              .get("ffn_tile_cycles") or 0
                              for l in t["layers"]
                              if l["walk"] and l["walk"].get("ffn")
                              and not l["walk"]["ffn"].get("refused"))
                          for t in res["tokens"]],
                "LABEL": "sim-MEASURED tile cycles summed over the layers' "
                         "FFN slice windows (CYCMARK FFN-PREGO..WALKDONE). "
                         "NOT part of walk_layers_05b.predict — that model "
                         "covers the E-6 chain only."}

    # ── the ledger ─────────────────────────────────────────────────────────
    text = tok.decode(res["ids"]) if tok else None
    # all_walk_green = every GRADED walk green. --fast ungraded records
    # short-circuit (they carry no grade to be green about — the fast
    # label owns the no-claim); non-fast runs never carry "ungraded".
    all_walk_green = all(l["walk"] is None
                         or bool(l["walk"].get("ungraded"))
                         or (l["walk"]["qkv"]["all_bit_exact"]
                             and l["walk"]["r1"]["bit_exact"]
                             and l["walk"].get("attn",
                                               {}).get("bit_exact", True))
                         for t in res["tokens"] for l in t["layers"])
    n_refused = sum(len(s["refused_layers"]) for s in splits)
    ffn_green, ffn_stats = True, None
    if ffn_on:
        # the --ffn gate: EVERY layer of every sampled step must carry a
        # walked, bit-exact FFN window — a refusal or a missing window
        # fails the run's verdict, in as many words.
        recs = [(l["walk"] or {}).get("ffn")
                for t in res["tokens"] for l in t["layers"]]
        n_missing = sum(1 for f in recs if f is None)
        n_ffn_ref = sum(1 for f in recs if f and f.get("refused"))
        n_ok = sum(1 for f in recs if f and not f.get("refused")
                   and f["fs"]["equal"] and f["p_codes"]["equal"]
                   and f["r2_row"]["equal"])
        ffn_green = (n_missing == 0 and n_ffn_ref == 0
                     and n_ok == len(recs))
        ffn_stats = {"windows": len(recs), "bit_exact": n_ok,
                     "refused": n_ffn_ref, "missing": n_missing}
    graded_tokens = None
    if fast_n is not None:
        graded_tokens = [t["index"] for j, t in enumerate(res["tokens"])
                         if not _fast_ungraded(j, fast_n)]
    # --fast: no reference exists — identity is NOT part of the verdict
    # (and the label says so); the verdict is the graded checks only.
    ident_ok = (res["ids"] == ref["ids"]) if ref is not None else True
    verdict = bool(ident_ok and all_walk_green and ffn_green)
    print("-" * 78)
    print(f"  generated ids        : {res['ids']}"
          + (f"  text={text!r}" if text is not None else ""))
    if fast_n is not None:
        n_g_walk = sum(1 for t in res["tokens"] for l in t["layers"]
                       if l["walk"] and not l["walk"].get("ungraded"))
        n_u_walk = sum(1 for t in res["tokens"] for l in t["layers"]
                       if l["walk"] and l["walk"].get("ungraded"))
        print(f"  {fast_label(graded_tokens)}")
        print(f"  token identity       : NOT CHECKED — the reference pass "
              f"was skipped (--fast); this run makes NO identity claim")
        print(f"  walked layer-chains  : {n_g_walk} graded (bit-exact -> "
              f"{'PASS' if all_walk_green else 'FAIL'}) + {n_u_walk} "
              f"walked UNGRADED — NO bit-exactness claim"
              + (f"  (REFUSED subjects: {n_refused})" if n_refused else ""))
    else:
        print(f"  token identity       : engine == host-golden at every "
              f"token -> {'PASS' if res['ids'] == ref['ids'] else 'FAIL'}")
    if args.engine in ("sim-walked", "hw-walked"):
        n_walk = sum(1 for t in res["tokens"] for l in t["layers"]
                     if l["walk"])
        e7_run = getattr(args, "mask", "b") == "e7"
        if fast_n is None:
            print(f"  walked layer-chains  : {n_walk} run"
                  + (" (incl. attention score/pv per layer)" if e7_run
                     else "")
                  + f", all bit-exact -> "
                  f"{'PASS' if all_walk_green else 'FAIL'}"
                  + (f"  (REFUSED subjects: {n_refused})" if n_refused
                     else ""))
        if ffn_on:
            s_walk = int(args.ffn_chunks) * wfl.D_TILE
            print(f"  walked FFN windows   : {ffn_stats['bit_exact']}/"
                  f"{ffn_stats['windows']} bit-exact @ d_ffn[0:{s_walk}] "
                  f"of {int(model.meta['d_ffn'])} "
                  f"({3 * wfl.D_MODEL * s_walk:,} MACs/layer) -> "
                  f"{'PASS' if ffn_green else 'FAIL'}"
                  + (f"  (refused {ffn_stats['refused']}, missing "
                     f"{ffn_stats['missing']})"
                     if (ffn_stats['refused'] or ffn_stats['missing'])
                     else ""))
        for i, s in enumerate(splits):
            print(f"  token {i + 1} MAC split    : walked {s['walked_macs']:,} "
                  f"= {100 * s['walked_frac_of_census']:.2f}% of the "
                  f"measured step+head ({s['census']['total']:,}); "
                  f"{100 * s['walked_frac_of_kv_cached']:.2f}% of the "
                  f"kv-cached denominator ({s['kv_cached']['total']:,})")
    probe_rec = None
    if probe_steps is not None:
        walked_at_T = sorted({t["T"] for t in res["tokens"]
                              for l in t["layers"] if l["walk"]})
        probe_rec = {"steps": sorted(probe_steps),
                     "layers": ("all" if probe_layers is None
                                else sorted(probe_layers)),
                     "walked_at_T": walked_at_T,
                     "layer_slots_skipped": getattr(engine, "n_skipped", 0),
                     "note": "probe mode walks ONLY the named (step, layer) "
                             "chains; skipped slots are host-vouched golden"}
        print(f"  probe mode           : steps {sorted(probe_steps)} x "
              f"layers {probe_rec['layers']} — walked chains at "
              f"T={walked_at_T}; {probe_rec['layer_slots_skipped']} layer "
              f"slots host-vouched (NOT walked)")
    print(f"  per-token wall       : {[t['wall_s'] for t in res['tokens']]} s"
          f"  ({timing['tokens_per_s_host_wall']['value']} tok/s host "
          f"golden pipeline — see LABEL in the record)")
    print(f"  TOKEN LOOP GATE      : {'PASS' if verdict else 'FAIL'}")

    out = work / (f"token_loop_{args.engine}.ffn.json" if ffn_on
                  else f"token_loop_{args.engine}.json")
    out.write_text(json.dumps({
        "verdict": verdict,
        "engine": args.engine,
        "engine_summary": engine_summary,
        "prompt_ids": ids,
        "prompt_text": tok.decode(ids) if tok else None,
        "generated_ids": res["ids"],
        "generated_text": text,
        "reference_ids_host_golden": (ref["ids"] if ref is not None
                                      else None),
        "token_identity": ((res["ids"] == ref["ids"]) if ref is not None
                           else "NOT CHECKABLE (--fast: reference pass "
                                "skipped; no claim is made)"),
        "fast": (None if fast_n is None else {
            "n": fast_n,
            "graded_tokens": graded_tokens,
            "LABEL": fast_label(graded_tokens)}),
        "eos_hit": bool(res["ids"] and res["ids"][-1] in eos),
        "tier": tier, "group": args.group,
        "weights_dir": str(Path(args.weights_dir).resolve()),
        "seam": {
            "walked_steps": (
                (WALKED_STEPS_E7
                 if args.engine in ("sim-walked", "hw-walked")
                 else "none — pure host")
                if getattr(args, "mask", "b") == "e7"
                else ((walked_label if walked_label is not None
                       else WALKED_STEPS)
                      if args.engine in ("sim-walked", "hw-walked")
                      else "none — pure host")),
            "host_steps": (HOST_STEPS_E7
                           if getattr(args, "mask", "b") == "e7"
                           else HOST_STEPS),
            "carry": "inter-layer and inter-token state is golden's own "
                     "composition — the continuation of walked values "
                     "verified bit-exact first (walk_layers_05b's seam, "
                     "extended across tokens); prefill of ids[:-1] is "
                     "host-golden context build",
            **({"ffn_windows": ffn_stats, "ffn_seam": FFN_SEAM_NOTE}
               if ffn_on else {}),
        },
        "probe": probe_rec,
        "timing": timing,
        "mac_split_per_token": splits,
        "tokens": res["tokens"],
        "git": rt.git_rev(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if verdict else 1


# ═══════════════════ selftest (offline: no sim, no model) ══════════════════

def selftest() -> int:
    ok = []

    def chk(cond, msg):
        ok.append(bool(cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {len(ok)} {msg}")

    # 1. id parsing refuses junk and empties
    chk(parse_ids("785, 6722,315") == [785, 6722, 315],
        "parse_ids parses a comma list")
    for bad in ("", "a,b", "12,-3"):
        try:
            parse_ids(bad)
            chk(False, f"parse_ids({bad!r}) is REFUSED")
        except SystemExit:
            chk(True, f"parse_ids({bad!r}) is REFUSED")

    # 2. the engine registry is exactly the three contracts
    chk(sorted(ENGINES) == ["host-golden", "hw-walked", "sim-walked"],
        "engine registry = {host-golden, hw-walked, sim-walked}")
    # 2b. hw-walked REFUSES without the operator's DDR attestation, and its
    #     executor hooks are the only overrides of the sim seam
    chk(HwWalkedEngine.__bases__ == (SimWalkedEngine,),
        "hw-walked subclasses the sim engine (fences/grades shared)")
    _hw_need = {"start", "_execute", "_audit", "finish", "__init__",
                "layer", "flush_step"}
    _hw_own = {n for n in _hw_need if n in HwWalkedEngine.__dict__}
    chk(_hw_own == _hw_need,
        "hw-walked overrides exactly the executor+batch seam")
    chk("_build_one" not in HwWalkedEngine.__dict__
        and "_grade_one" not in HwWalkedEngine.__dict__,
        "hw-walked does NOT override _build_one/_grade_one — the build and "
        "grade semantics are the sim engine's verbatim")

    # 3. the bit gate REFUSES a single flipped bit and passes identity
    g = np.arange(896, dtype=np.uint16)
    try:
        check_engine_bits("t", g.copy(), g)
        chk(True, "bit gate passes identical bits")
    except SystemExit:
        chk(False, "bit gate passes identical bits")
    e = g.copy()
    e[313] ^= 1
    try:
        check_engine_bits("t", e, g)
        chk(False, "bit gate REFUSES one flipped bit")
    except SystemExit as exc:
        chk("REFUSE" in str(exc), "bit gate REFUSES one flipped bit")

    # 4. walked-MAC arithmetic from the walker's own decomposition
    jobs = wfl.expected_qkv_jobs()
    n_out = len(jobs) * 8
    chk(len(jobs) == 144 and n_out == wfl.D_MODEL + 2 * wfl.KV_DIM,
        f"walked QKV = 144 jobs = {n_out} accumulators (D+2*kv)")
    walked = n_out * wfl.D_MODEL + wfl.D_MODEL * wfl.D_MODEL
    chk(walked == 1_835_008,
        f"walked MACs/layer = QKV {n_out}x{wfl.D_MODEL} + OPROJ "
        f"{wfl.D_MODEL}^2 = {walked}")

    # 5. verify_census refuses a fabricated census (wrong layer count)
    class _M:                                     # minimal stand-in
        n_layers = 2

    c = MacCensus(2)
    c.gemm = [{"where": (0, 0), "M": 1, "K": 4, "N": 4, "macs": 16}]
    try:
        verify_census(c, _M(), steps=[0], n_head_calls=0)
        chk(False, "verify_census REFUSES a partial step")
    except (SystemExit, AttributeError):
        chk(True, "verify_census REFUSES a partial step")

    # 6. the SPLIT T budgets and their selection arithmetic (enforced in
    #    run(); the constants' RTL citations live at their definitions)
    chk(T_SIM_MAX == 8, "sim-walked FULL-decode budget is T<=8 (wall time)")
    chk(T_HW_MAX == 64, "hw-walked envelope budget is T<=64 (MXE m ceiling)")
    chk(T_SIM_MAX < T_HW_MAX <= rt.T_SESSION_MAX,
        f"budget order T_SIM_MAX < T_HW_MAX <= T_SESSION_MAX "
        f"({T_SIM_MAX} < {T_HW_MAX} <= {rt.T_SESSION_MAX})")
    chk(t_budget("sim-walked", False)[0] == T_SIM_MAX,
        "FULL sim decode is fenced at T_SIM_MAX")
    chk(t_budget("sim-walked", True)[0] == T_HW_MAX,
        "sim --probe-steps run is fenced at T_HW_MAX")
    chk(t_budget("hw-walked", False)[0] == T_HW_MAX
        and t_budget("hw-walked", True)[0] == T_HW_MAX,
        "hw-walked is fenced at T_HW_MAX, probes or not")
    chk(t_budget("host-golden", False)[0] == rt.T_SESSION_MAX,
        "host-golden is fenced at T_SESSION_MAX")

    # 6b. probe parsing + the probe skip filter (offline arithmetic)
    class _A:                                    # argparse stand-in
        engine = "sim-walked"
        probe_steps = "15,23"
        probe_layers = "0,23"

    chk(parse_probes(_A()) == ({15, 23}, {0, 23}),
        "parse_probes parses steps x layers")
    _A.probe_layers = None
    chk(parse_probes(_A()) == ({15, 23}, None),
        "--probe-layers defaults to all layers")
    _A.probe_steps, _A.probe_layers = None, None
    chk(parse_probes(_A()) == (None, None), "no probe flags -> no probing")
    _A.probe_layers = "3"
    try:
        parse_probes(_A())
        chk(False, "--probe-layers without --probe-steps is REFUSED")
    except SystemExit:
        chk(True, "--probe-layers without --probe-steps is REFUSED")
    _A.probe_steps, _A.probe_layers = "5", None
    _A.engine = "host-golden"
    try:
        parse_probes(_A())
        chk(False, "probing a non-walked engine is REFUSED")
    except SystemExit:
        chk(True, "probing a non-walked engine is REFUSED")

    class _Fx:                                   # fx stand-in for the skip
        r2 = np.zeros(4)

    eng = SimWalkedEngine()
    eng.probe_steps, eng.probe_layers = {15}, {0, 23}
    chk(eng._probe_skip(15, 0, _Fx()) is None
        and eng._probe_skip(15, 23, _Fx()) is None,
        "probed (step, layer) chains are walked")
    s = eng._probe_skip(15, 7, _Fx())
    chk(s is not None and s["skipped"] == "probe" and s["walk"] is None
        and s["walked_macs"] == 0,
        "unprobed layer at a probed step is skipped, walks nothing")
    chk(eng._probe_skip(14, 0, _Fx()) is not None,
        "unprobed step is skipped entirely")
    chk(eng.n_skipped == 2, "the skip counter counts every skipped slot")
    eng.probe_steps = None
    chk(eng._probe_skip(3, 3, _Fx()) is None,
        "no probe config -> every (step, layer) is walked")

    # 7. the e7 composition (docs/design/E7_TOKEN_LOOP.md) — masks, rows,
    #    descriptor slots, drift sets, drain ledger, MAC formula. Offline:
    #    synthetic bases/rq, no image, no model.
    chk(MASK_E7A == (1 << 3) | (1 << 4) | (1 << 5),
        "MASK_E7A is EXACTLY {STOREKV, SCORE, PV} — attention at the live "
        "T with the WALKER owning the per-engine store addressing/health "
        "(sk_g — E7_LIVE_T_DEFECT.md); the projections keep the FLOWN "
        "T-free MASK_B image (a {FPROJ, QKV} image at t_rows>1 wedges at "
        "the Wk family — measured 2026-08-11)")
    chk(fmt.EN_STOREKV == 3 and (MASK_E7A >> fmt.EN_STOREKV) & 1 == 1,
        "the ATT image names EN_STOREKV (W2_MASK[3]) — the walked store "
        "step is composed, not host-only")
    chk(wfl.MASK_B == (1 << 14) | (1 << 1) | (1 << 6) | (1 << 7),
        "the e7 projection kick reuses wfl.MASK_B verbatim")
    rows = wfl.D_MODEL // wfl.D_TILE
    chk(rows == 14 and 2 * rows == 28 <= 31 and 3 * rows == 42 > 31,
        "the split obeys the 31-row act bank (ATT q rows 14; MASK_B kick "
        "28; the one-kick E-8 shape at 42 is the refused capacity)")
    b0 = {"Wq": 0x100, "Wk": 0x200, "Wv": 0x300, "Wo": 0x400}
    b1 = {"Wq": 0x110, "Wk": 0x210, "Wv": 0x310, "Wo": 0x410}
    rq0 = [(1000 + h, 3) for h in range(wfl.N_HEADS)]
    rq1 = [(2000 + h, 4) for h in range(wfl.N_HEADS)]
    dA0 = e7_att_descriptor(b0, rq0, T=6, pos=5)
    dA1 = e7_att_descriptor(b1, rq1, T=6, pos=5)
    moved = [w for w in range(fmt.DESC_WORDS) if dA0[w] != dA1[w]]
    chk(set(moved) <= set(ATT_REPOINT_WORDS),
        f"L->L+1 ATT descriptor delta is within the re-point set "
        f"({sorted(moved)})")
    dA2 = e7_att_descriptor(b1, rq1, T=7, pos=6)
    moved2 = [w for w in range(fmt.DESC_WORDS) if dA1[w] != dA2[w]]
    chk(set(moved2) == {fmt.W_STEP},
        "step->step+1 ATT delta (same layer) is EXACTLY W_STEP (T, q_pos)")
    chk(dA0[fmt.W_STEP] == fmt.pack_step(6, 5)
        and dA0[fmt.W_RQ0 + 13] == fmt.pack_rq(1013, 3)
        and dA0[fmt.W_RQ0 + wfl.N_HEADS] == 0,
        "ATT slots: W_STEP=(T, q_pos); RQ[h] per head; RQ[H] (o-proj) "
        "untouched by the ATT image")
    try:
        e7_att_descriptor(b0, rq0, T=9, pos=8)
        chk(False, "T>8 (outside CQ-8) is REFUSED")
    except SystemExit:
        chk(True, "T>8 (outside CQ-8) is REFUSED")
    lay = att_cap_layout(6)
    chk(lay["fs"] == wfl.N_HEADS * (1 + 6 + 8 * 6)
        and lay["ss"] == wfl.N_HEADS
        and lay["ro"] == 14 * 8 * 9,
        f"ATT drain ledger at T=6: fs={lay['fs']} ss={lay['ss']} "
        f"ro={lay['ro']} (q scales + s_k/s_v ladders + the PV o8 blocks)")
    chk(2 * wfl.HEAD_DIM * wfl.N_HEADS == 1792,
        "attention adds 1792*T walked MACs/layer (score T*64 + PV 64*T "
        "per head, graded through the o8 composition)")
    walked_e7 = 1_835_008 + 1792 * 6
    chk(walked_e7 == 1_845_760,
        f"walked MACs/layer at T=6 = {walked_e7} (MASK_B's 1,835,008 + "
        f"attention)")
    chk(E7A_DATA_ADVANCES == frozenset({0x3228, 0x3248, 0x3258, 0x3268}),
        "the ATT window's data plane is exactly the RO/FS/SS/TD advances")
    chk(set(wl.REPOINT_WORDS) == set(
        [fmt.W_TENS0 + t for t in (fmt.TENS_WQ, fmt.TENS_WK,
                                   fmt.TENS_WV, fmt.TENS_WO)]
        + [fmt.W_RQ0 + wfl.N_HEADS, fmt.W_JC0 + fmt.JC_OPROJ]),
        "EPI drift fence reuses wl.REPOINT_WORDS verbatim (4 bases + "
        "RQ[H] + JC_OPROJ)")

    # 8. --ffn: the slice descriptor is fmt=1 LEGAL at the real geometry,
    #    with the slots the RTL reads (offline — synthetic bases/comps)
    s_walk = FFN_CHUNKS_MAX * wfl.D_TILE
    dw = build_ffn_descriptor({"Wg": 0, "Wu": 1064, "Wd": 2128},
                              d_ffn_walk=s_walk, comp_g=0x38800000,
                              comp_u=0x38800000, comp_d=0x38800000,
                              rq_d=(1234, 7))
    chk(fmt.check2(dw[0], dw[1], dw[2], dw[3], dw[fmt.W_STEP],
                   cfg_d=wfl.HEAD_DIM) == fmt.ERR_NONE,
        "the FFN slice descriptor is fmt=1 LEGAL at the twin's CFG_D")
    chk(dw[fmt.W_MASK] == FFN_MASK_DOWN
        and FFN_MASK_DOWN == (1 << 14) | (1 << 9) | (1 << 16) | (1 << 10),
        "the FFN mask is EXACTLY {FPROJ, FFN, DOWN, RES2}")
    chk(dw[fmt.W_MODEL0] == (s_walk << 16) | wfl.D_MODEL,
        f"model0 carries (d_model={wfl.D_MODEL}, d_ffn_walk={s_walk})")
    chk(dw[fmt.W_RQ0 + wfl.N_HEADS + 1] == (7 << 16) | 1234,
        "the DOWN RQ pair lands at RQ[H+1] (seq_layer_walker2 rq_word)")

    # 9. --ffn: the slice bounds and the walked-MAC arithmetic
    chk(FFN_CHUNKS_MAX == 16
        and wfl.D_MODEL // wfl.D_TILE + FFN_CHUNKS_MAX <= 31
        and FFN_CHUNKS_MAX <= FFN_FS_FIFO_DEPTH,
        "slice bound: min(31 - 14 stage rows = 17, 16 fs-FIFO slots) == 16 "
        "chunks per silent window")
    ffn_macs = 2 * wfl.D_MODEL * s_walk + s_walk * wfl.D_MODEL
    chk(ffn_macs == 2_752_512,
        f"FFN walked MACs/layer = gate+up {2 * wfl.D_MODEL * s_walk:,} + "
        f"down {s_walk * wfl.D_MODEL:,} = {ffn_macs:,} (~1.5x the E-6 "
        f"chain's 1,835,008)")
    chk(set(FFN_REPOINT_WORDS) <= set(range(fmt.DESC_WORDS))
        and len(FFN_REPOINT_WORDS) == 7,
        "the FFN re-point set = 3 tensor bases + 3 JC words + RQ[H+1]")

    # 10. --fast (the EXPLICIT demo/measurement mode): flag surface, gate
    #     refusals, graded-token arithmetic, label honesty, and the
    #     default path UNTOUCHED (asserted on run()'s own source).
    import inspect                       # noqa: PLC0415 — selftest-only
    pp = build_parser()
    a0 = pp.parse_args(["run"])
    chk(getattr(a0, "fast", "MISSING") is None,
        "--fast defaults to ABSENT (None) — the two-pass graded run")
    a3 = pp.parse_args(["run", "--engine", "sim-walked", "--fast", "3",
                        "--ddr-attested"])
    chk(a3.fast == 3 and a3.ddr_attested,
        "--fast N parses as an int period alongside --ddr-attested")

    class _F0:                           # NOTHING but fast=None
        fast = None

    chk(fast_gate(_F0()) is None,
        "fast_gate(--fast absent) returns None before consulting ANY "
        "other flag — the default path takes no fast branch")

    class _F:                            # argparse stand-in
        engine = "sim-walked"
        fast = 3
        ddr_attested = False
        ffn = False
        probe_steps = None

    try:
        fast_gate(_F())
        chk(False, "--fast without --ddr-attested is REFUSED")
    except SystemExit as exc:
        chk("--ddr-attested" in str(exc),
            "--fast without --ddr-attested is REFUSED")
    _F.ddr_attested = True
    chk(fast_gate(_F()) == 3,
        "--fast 3 + --ddr-attested on a walked engine passes the gate")
    _F.engine = "host-golden"
    try:
        fast_gate(_F())
        chk(False, "--fast on host-golden is REFUSED (nothing walked)")
    except SystemExit:
        chk(True, "--fast on host-golden is REFUSED (nothing walked)")
    _F.engine, _F.ffn = "hw-walked", True
    try:
        fast_gate(_F())
        chk(False, "--fast with --ffn is REFUSED (the --ffn gate)")
    except SystemExit:
        chk(True, "--fast with --ffn is REFUSED (the --ffn gate)")
    _F.ffn, _F.probe_steps = False, "5"
    try:
        fast_gate(_F())
        chk(False, "--fast with --probe-steps is REFUSED")
    except SystemExit:
        chk(True, "--fast with --probe-steps is REFUSED")
    _F.probe_steps, _F.fast = None, 0
    try:
        fast_gate(_F())
        chk(False, "--fast 0 is REFUSED (N >= 1)")
    except SystemExit:
        chk(True, "--fast 0 is REFUSED (N >= 1)")

    chk([i for i in range(7) if not _fast_ungraded(i, 3)] == [0, 3, 6],
        "--fast 3 grades tokens 1, 4, 7 (every Nth, anchored at the first)")
    chk(all(not _fast_ungraded(i, None) for i in range(8)),
        "fast absent -> EVERY token graded (the default discipline)")
    chk(all(not _fast_ungraded(i, 1) for i in range(4)),
        "--fast 1 grades every token (drops only the reference pass)")

    lbl = fast_label([1, 4])
    chk("FAST MODE: reference pass SKIPPED, walked grades on tokens "
        "[1, 4]; UNGRADED tokens carry NO bit-exactness claim" in lbl
        and "NOT checkable" in lbl,
        "the fast label names the skipped pass, the graded tokens, the "
        "no-claim, and the uncheckable token identity")
    chk("NO bit-exactness claim" in FAST_UNGRADED_NOTE,
        "the per-layer ungraded note carries the no-claim in words")

    # the default (non-fast) path is UNTOUCHED — pinned on the source:
    chk(inspect.signature(decode_pass).parameters["fast_n"].default is None,
        "decode_pass(fast_n=None) — callers that don't name it get "
        "today's path")
    src_run = inspect.getsource(run)
    chk("ref = None" in src_run and "if fast_n is None:" in src_run
        and (src_run.index("if fast_n is None:")
             < src_run.index("[pass 1/2] host-golden")),
        "run(): the reference pass is gated ONLY by fast_n (None -> "
        "pass 1 ALWAYS runs — the default two-pass discipline)")
    chk('expect_ids=(ref["ids"] if ref is not None else None)' in src_run,
        "run(): pass 2's token expectation comes from the reference pass "
        "alone (absent only under --fast)")
    src_dp = inspect.getsource(decode_pass)
    chk("if fast_n is not None:" in src_dp
        and 'ctx["fast_ungraded"]' in src_dp
        and (src_dp.index("if fast_n is not None:")
             < src_dp.index('ctx["fast_ungraded"]')),
        "decode_pass: the ungraded flag is written ONLY under --fast — "
        "non-fast ctx never carries it")
    eng_f = SimWalkedEngine()
    chk(eng_f.grade_s == 0.0 and eng_f.n_ungraded == 0
        and 'ctx.get("fast_ungraded", False)' in
        inspect.getsource(SimWalkedEngine.layer),
        "engine default: no fast flag in ctx -> graded (today's path); "
        "grade_s/n_ungraded start at zero")

    # 11. --ffn composition fences (check_ffn_composition — run()'s own
    #     arbiter): hw-walked is ARMED (2026-08-13); the non-walked engine
    #     and the never-flown pairings still REFUSE (--fast+--ffn is
    #     fast_gate's own refusal, pinned in section 10)
    for eng, mask, probes, f_on, want_ok in (
            ("sim-walked", "b", None, True, True),
            ("hw-walked", "b", None, True, True),
            ("host-golden", "b", None, True, False),
            ("sim-walked", "e7", None, True, False),
            ("hw-walked", "e7", None, True, False),
            ("sim-walked", "b", {5}, True, False),
            ("hw-walked", "b", {5}, True, False),
            ("host-golden", "e7", {5}, False, True)):
        try:
            check_ffn_composition(eng, mask, probes, f_on)
            got_ok = True
        except SystemExit:
            got_ok = False
        chk(got_ok == want_ok,
            f"--ffn fence: engine={eng} mask={mask} "
            f"probes={'y' if probes else 'n'} ffn={f_on} -> "
            f"{'ALLOWED' if want_ok else 'REFUSED'}")

    # 11b. the hw DDR layout arithmetic: the rebased slice image must be
    #      4 KiB-region-legal at the constant and stay inside the 30-bit
    #      fuel_req base field with the maximal slice resident
    s_bytes_l = 3 * wfl.D_MODEL * s_walk               # Wg + Wu + Wd, int8
    hw_top = FFN_HW_DDR_BASE + 24 * ((s_bytes_l + 3 * 4095) // 4096) * 4096
    chk(FFN_HW_DDR_BASE % 4096 == 0
        and hw_top // 64 < (1 << fmt.FR_BASE_W),
        f"FFN_HW_DDR_BASE={FFN_HW_DDR_BASE:#x} is 4 KiB aligned and the "
        f"24L maximal slice image tops out at {hw_top:#x} << the 30-bit "
        f"fuel_req base field")
    b_lo = {"Wg": FFN_HW_DDR_BASE // 64,
            "Wu": FFN_HW_DDR_BASE // 64 + 14336,
            "Wd": FFN_HW_DDR_BASE // 64 + 28672}
    b_hi = {k: v + 43008 for k, v in b_lo.items()}     # the next layer
    dh0 = build_ffn_descriptor(b_lo, d_ffn_walk=s_walk, comp_g=0x38800000,
                               comp_u=0x38800000, comp_d=0x38800000,
                               rq_d=(1234, 7))
    dh1 = build_ffn_descriptor(b_hi, d_ffn_walk=s_walk, comp_g=0x38800000,
                               comp_u=0x38800000, comp_d=0x38800000,
                               rq_d=(1234, 7))
    chk(fmt.check2(dh0[0], dh0[1], dh0[2], dh0[3], dh0[fmt.W_STEP],
                   cfg_d=wfl.HEAD_DIM) == fmt.ERR_NONE
        and dh0[fmt.W_TENS0 + fmt.TENS_WG] == FFN_HW_DDR_BASE // 64,
        "a REBASED (hw) FFN descriptor is fmt=1 LEGAL and carries the "
        "offset base words verbatim (no in-flight address math)")
    moved_hw = [w for w in range(fmt.DESC_WORDS) if dh0[w] != dh1[w]]
    chk(set(moved_hw) <= set(FFN_REPOINT_WORDS),
        f"hw L->L+1 FFN descriptor delta stays within the re-point set "
        f"({sorted(moved_hw)})")

    # 12. hw+ffn DRY CONSTRUCTION (the hw arm's static gate — no executor):
    #     one layer's FFN program through the SAME builder the hw flight
    #     uses (FfnLane._build_program is executor-agnostic BY DESIGN — the
    #     hw/sim split lives only in run_one's bridge call), on a synthetic
    #     subject with rebased bases. Asserts the walk window is host-
    #     silent and the file bakes EXACTLY the cap ledger the grade
    #     consumes: D_model r2 caps + per chunk (1 fs beat + 8 identity
    #     readbacks x 9 caps) — measured 2026-08-13 on this builder at
    #     chunks in {1, 2, 16} (affine in chunks).
    import tempfile                                    # noqa: PLC0415
    lane = FfnLane(2)
    lane.d_ffn = 4864
    lane.hw = True                                     # the hw arm's lane
    _rng = np.random.default_rng(7)
    _sub = {"h2_8": _rng.integers(-127, 128,
                                  size=wfl.D_MODEL).astype(np.int64),
            "r1_bits": np.asarray(
                f64_to_f16_bits(_rng.standard_normal(wfl.D_MODEL)),
                dtype=np.uint16)}
    _desc = build_ffn_descriptor(
        {"Wg": FFN_HW_DDR_BASE // 64, "Wu": FFN_HW_DDR_BASE // 64 + 3584,
         "Wd": FFN_HW_DDR_BASE // 64 + 7168},
        d_ffn_walk=lane.S, comp_g=0x38800000, comp_u=0x38800000,
        comp_d=0x38800000, rq_d=(1234, 7))
    with tempfile.TemporaryDirectory() as _td:
        _em = lane._build_program(_sub, None, _desc, Path(_td),
                                  "selftest_hw_ffn_dry")
        _want = wfl.D_MODEL + lane.chunks * (1 + 9 * (wfl.D_TILE
                                                      // wfl.MXE_N))
        chk(_em["sil"]["silent"],
            "hw+ffn dry construction: the walk window is host-silent "
            "(census + silence predicate enforced inside the builder)")
        chk(_em["n_caps"] == _want,
            f"hw+ffn dry construction bakes exactly the grade's cap "
            f"ledger: {_em['n_caps']} == {wfl.D_MODEL} r2 + "
            f"{lane.chunks} x 73 = {_want}")

    # 13. --emit-cache (the per-layer MASK_B emission template cache):
    #     flag surface + per-engine default resolution, the payload
    #     transform pinned against inject_jobs' OWN emission through the
    #     LayerScript translator, and every fence surface — descriptor
    #     drift, cap-count ledger, verify-mode byte identity. Offline.
    a13 = pp.parse_args(["run"])
    chk(getattr(a13, "emit_cache", "MISSING") is None,
        "--emit-cache defaults to ABSENT (None) — resolved per engine")

    class _EC:                           # argparse stand-in
        engine = "sim-walked"
        emit_cache = None

    chk(emit_cache_mode(_EC()) == "off",
        "emit-cache default is OFF for sim-walked — the proven "
        "from-scratch emission stays the sim default")
    _EC.engine = "hw-walked"
    chk(emit_cache_mode(_EC()) == "on",
        "emit-cache default is ON for hw-walked (the measured ~1.2 "
        "s/token emission bottleneck)")
    _EC.engine, _EC.emit_cache = "sim-walked", "verify"
    chk(emit_cache_mode(_EC()) == "verify",
        "an explicit --emit-cache value passes through unchanged")
    _EC.engine = "host-golden"
    try:
        emit_cache_mode(_EC())
        chk(False, "--emit-cache on host-golden is REFUSED")
    except SystemExit:
        chk(True, "--emit-cache on host-golden is REFUSED")
    _EC.emit_cache = None
    chk(emit_cache_mode(_EC()) == "off",
        "host-golden without the flag resolves off (harmless)")

    # the payload transform vs the REAL builder emission (inject_jobs
    # through the LayerScript translator — the code path build_program's
    # staging loop drives)
    with wfl.tg.at_d(wfl.D_TILE):
        _s13 = glo.LayerScript(wfl.D_TILE)
        _r13 = np.random.default_rng(13)
        _fam13 = _r13.integers(-127, 128,
                               size=2 * wfl.D_TILE).astype(np.int64)
        for _r in range(2):
            _rowv = _fam13[_r * wfl.D_TILE:(_r + 1) * wfl.D_TILE] \
                .astype(np.float64)
            _prs = [g3.decompose_f16(int(_b))[:2]
                    for _b in g3.to16(_rowv)]
            g3.inject_jobs(_s13, _prs, 1)
        _x13 = glo._translate(_s13)
    _qs_d = [o["d"] for o in _x13.ops
             if o.get("op") == "pw" and o.get("a") == ec.QS_A]
    _xw_d = [o["d"] for o in _x13.ops
             if o.get("op") == "w" and o.get("a") == ec.XW0_A]
    _q13, _w13 = ec.act_payloads([_fam13])
    chk(_qs_d == _q13 and _xw_d == _w13 and len(_q13) == 2 * wfl.D_TILE,
        "emit-cache payload transform reproduces inject_jobs' QS/XW0 "
        "streams bit-for-bit (the patcher writes what the builder writes)")
    chk(all(o["d"] == 0 for o in _x13.ops
            if o.get("op") == "w" and o.get("a") == ec.XW1_A),
        "every XW1 beat word is 0 (the K=2 encoding the template pins)")

    # the descriptor drift fence: same-layer rq/comp re-point passes;
    # any non-re-point word REFUSES the patch
    _b13 = {"Wq": 0x100, "Wk": 0x200, "Wv": 0x300, "Wo": 0x400}
    _dT = wfl.walk_descriptor(_b13, mask=wfl.MASK_B, rq=(1111, 5),
                              comp=0x3F000000)
    _dP = wfl.walk_descriptor(_b13, mask=wfl.MASK_B, rq=(2222, 6),
                              comp=0x3E800000)
    _mv = ec.check_desc_drift(_dT, _dP, tag="selftest")
    chk(_mv and set(_mv) <= set(wl.REPOINT_WORDS),
        f"same-layer rq/comp re-point passes the drift fence ({_mv})")
    _dBad = list(_dP)
    _dBad[fmt.W_MASK] ^= 1
    try:
        ec.check_desc_drift(_dT, _dBad, tag="selftest")
        chk(False, "a non-re-point descriptor move REFUSES the patch")
    except SystemExit as exc:
        chk("REPOINT" in str(exc) or "re-point" in str(exc),
            "a non-re-point descriptor move REFUSES the patch")

    # the cap-count fence
    try:
        ec.check_cap_fence(2191, 2192, tag="selftest")
        chk(False, "cap-count fence REFUSES a drifted capture ledger")
    except SystemExit as exc:
        chk("REFUSE" in str(exc),
            "cap-count fence REFUSES a drifted capture ledger")
    try:
        ec.check_cap_fence(2192, 2192, tag="selftest")
        chk(True, "cap-count fence passes an identical ledger")
    except SystemExit:
        chk(False, "cap-count fence passes an identical ledger")

    # the verify-mode byte-identity assertion surface
    with tempfile.TemporaryDirectory() as _td13:
        _pa = Path(_td13) / "a.regops.jsonl"
        _pb = Path(_td13) / "b.regops.jsonl"
        _pa.write_text('{"op":"w","a":1,"d":2}\n')
        _pb.write_text('{"op":"w","a":1,"d":2}\n')
        chk(ec.assert_byte_identical(_pa, _pb, "selftest")
            == len(_pa.read_bytes()),
            "byte-identity assertion passes identical files")
        _pb.write_text('{"op":"w","a":1,"d":3}\n')
        try:
            ec.assert_byte_identical(_pa, _pb, "selftest")
            chk(False, "one differing byte REFUSES with VERIFY named")
        except SystemExit as exc:
            chk("EMIT-CACHE VERIFY" in str(exc),
                "one differing byte REFUSES with VERIFY named")

    # the default path is UNTOUCHED — pinned on the source (the --fast
    # idiom): no cache config -> _emit_program_full only, and the flag
    # default keeps sim-walked on the proven path
    chk(inspect.signature(_emit_program).parameters["emit_cache"].default
        is None,
        "_emit_program(emit_cache=None) — callers that don't name it get "
        "today's path")
    src_ep = inspect.getsource(_emit_program)
    chk("emit_cache is None" in src_ep and "_emit_program_full" in src_ep
        and (src_ep.index("emit_cache is None")
             < src_ep.index("ec.patch_program")),
        "_emit_program: the cache branch is gated ONLY by an armed config "
        "— absent/off falls through to the full build first")
    _eng13 = SimWalkedEngine()
    chk(_eng13.emit_cache is None and _eng13.emit_verify == [0, 0],
        "engine default: no emit-cache config until start() arms one")

    print("=" * 62)
    print(f"TOKEN_LOOP SELFTEST: {'ALL PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """The CLI surface — module-level so the selftest can pin the flag
    surface (notably that --fast defaults to ABSENT). main() is a pure
    parse_args + dispatch over this parser; semantics unchanged."""
    p = argparse.ArgumentParser(
        description="Multi-token greedy decode of Qwen2.5-0.5B with a "
                    "pluggable per-layer engine, graded token-for-token "
                    "against host-golden (the only arbiter).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--engine", default="host-golden",
                   choices=sorted(ENGINES))
    r.add_argument("--mask", default="b", choices=("b", "e7"),
                   help="walked chain per (step, layer): 'b' = wfl.MASK_B "
                        "{FPROJ, QKV, OPROJ, RES1} (the flown default, "
                        "unchanged); 'e7' = the E-7 composition at 0.5B — "
                        "+{SCORE, PV} with live-T KV staging, two kicks "
                        "(docs/design/E7_TOKEN_LOOP.md)")
    r.add_argument("--prompt", default=",".join(map(str, fp.PROMPT_IDS)),
                   help="comma-separated token ids (default: the committed "
                        "0.5B prompt 'The capital of France is')")
    r.add_argument("--tokens", type=int, default=3,
                   help="tokens to generate (FULL sim-walked decode: "
                        "prompt+tokens <= T_SIM_MAX=8, a Verilator wall "
                        "budget; hw-walked and sim --probe-steps runs: "
                        "<= T_HW_MAX=64, the tile's walk envelope)")
    r.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    r.add_argument("--image", default=str(DEFAULT_IMAGE),
                   help="sim-walked: the 24-layer resident weight image")
    r.add_argument("--binary", default=None,
                   help="sim-walked: f2sim twin (default: "
                        "verif/f2sim/<--obj>/f2sim)")
    r.add_argument("--obj", default="obj_tokenloop")
    r.add_argument("--no-batch", action="store_true",
                   help="hw-walked: one executor invocation PER LAYER "
                        "(the flown 2026-08-11 transport) instead of the "
                        "batched per-step invocation")
    r.add_argument("--emit-cache", default=None,
                   choices=("off", "on", "verify"),
                   help="walked engines: per-layer TEMPLATE CACHE for the "
                        "MASK_B walk-program emission (emit_cache.py). "
                        "off = build every (step, layer) program from "
                        "scratch (the proven path; the sim-walked "
                        "DEFAULT). on = the first build per layer records "
                        "the emitted bytes as a template; later steps "
                        "PATCH only the step-varying payloads (descriptor "
                        "words, staged activation beats, xrow row, name "
                        "note) and re-run the cheap fences — silence "
                        "census, cap-count ledger, descriptor drift "
                        "(wl.REPOINT_WORDS) — on the patched output, "
                        "REFUSING on any mismatch (the hw-walked DEFAULT; "
                        "emission is the measured ~1.2 s/token hw "
                        "bottleneck). verify = additionally full-build "
                        "every patched program and REFUSE unless "
                        "byte-identical (the gate mode). The e7 ATT kick "
                        "is never cached (its op structure moves with T)")
    r.add_argument("--ffn", action="store_true",
                   help="walked engines (sim-walked, hw-walked): ALSO walk "
                        "one FFN→DOWN slice window per (step, layer) — the "
                        "rung-1/2/3 machinery (walk_ffn_toy) on live "
                        "subjects; d_ffn[0:chunks*64] per descriptor (the "
                        "stage-bank bound). Roughly doubles the walked-MAC "
                        "share AND the sim wall. hw-walked additionally "
                        "needs the REBASED slice image loaded+full-verified "
                        "on the card and --ffn-ddr-attested.")
    r.add_argument("--ffn-chunks", type=int, default=FFN_CHUNKS_MAX,
                   help=f"--ffn: slice width in 64-col chunks (1.."
                        f"{FFN_CHUNKS_MAX}; default the maximal fit — "
                        f"min(31 - 14 stage rows, {FFN_FS_FIFO_DEPTH} "
                        f"fs-FIFO slots per silent window))")
    r.add_argument("--ffn-image", default=None,
                   help="--ffn: slice-image dir (default <--work>/"
                        "ffn_slice_image; built one-time from "
                        "--weights-dir via make_weight_image.build "
                        "against a sliced view)")
    r.add_argument("--slot", type=int, default=0,
                   help="hw-walked: FPGA slot for the direct-attach executor")
    r.add_argument("--ddr-attested", action="store_true",
                   help="hw-walked: I ran f2_ddr_load --load --verify "
                        "--full-verify for --image on THIS card")
    r.add_argument("--ffn-ddr-attested", action="store_true",
                   help="hw-walked --ffn: I ran f2_ddr_load --load "
                        "--verify --full-verify for the REBASED ffn slice "
                        "image (FFN_HW_DDR_BASE) on THIS card, after the "
                        "main image")
    r.add_argument("--fast", type=int, default=None, metavar="N",
                   help="EXPLICIT opt-in demo/measurement mode (walked "
                        "engines only; REFUSED without --ddr-attested): "
                        "SKIP the pass-1 host-golden reference decode and "
                        "grade the walked values of every Nth generated "
                        "token only (token 1, 1+N, …). UNGRADED tokens' "
                        "chains still fly but carry NO bit-exactness "
                        "claim, and token identity vs the reference is "
                        "NOT checkable (no reference exists) — the report "
                        "labels all of this. Absent (the default): "
                        "today's two-pass fully graded run, unchanged")
    r.add_argument("--probe-steps", default=None,
                   help="walked engines: comma-separated decode STEP "
                        "indices (the printed step=N) whose layers are "
                        "walked; every other layer slot is host-vouched "
                        "and recorded as skipped. Bounds the Verilator "
                        "wall by the probe count instead of T, so the "
                        "session fence becomes T_HW_MAX — the deep-T "
                        "evidence mode")
    r.add_argument("--probe-layers", default=None,
                   help="with --probe-steps: layer indices to walk at "
                        "the probed steps (default: all layers)")
    r.add_argument("--work", default=str(DEFAULT_WORK))
    r.add_argument("--tier", default="kvq8", choices=sorted(rt.TIER_MAP))
    r.add_argument("--group", type=int, default=128)
    r.add_argument("--timeout", type=int, default=5400,
                   help="sim-walked: per-layer executor timeout (s)")
    return p


def main() -> int:
    a = build_parser().parse_args()
    return selftest() if a.cmd == "selftest" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
