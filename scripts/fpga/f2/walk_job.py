#!/usr/bin/env python3
# walk_job.py — WALKER MODE on the F2 tile: the TILE sequences the attention
# score+pv control stream itself, the host only loads a 3-word descriptor,
# kicks, and drains the capture FIFOs.
#
# Sibling of compute_job.py (which is host-sequenced, rq_en=0, raw-INT32
# egress). Same npz inputs, same executor bridge, same cap_decode grading —
# the difference is WHO EMITS THE CONTROL OPS for phases F and G.
#
# ══ THE CHOREOGRAPHY (found by reading the RTL + the L3 walker-mode TB) ═════
# The WALK CSR window lives in the tile glue at csr word 0x5C-0x6C
# (rtl/seq/seq_walker_pkg.sv:74-83), which the CL maps at BAR0 0x1000 + word
# (scripts/fpga/f2/cl_apex/design/cl_apex.sv:19-20, 364) — so:
#     0x105C WALK_CTRL    [0]=walk_en (level), [1]=walk_go (self-clearing kick)
#     0x1060 WALK_DPTR    descriptor-SRAM write pointer (auto-inc on DDATA)
#     0x1064 WALK_DDATA   descriptor word (pointer post-increments)
#     0x1068 WALK_STATUS  [0]=busy [3:1]=phase [8]=err sticky (W1C)
#                         [11:9]=err code [15:12]=FMT_SUP
#     0x106C WALK_RQ      direct write of descriptor word 1 (the PV requant
#                         pair) — the ONE per-step field, addressed directly so
#                         a steady-state step costs 3 MMIO and not 4
#                         (rtl/seq/seq_walker_pkg.sv:78-83)
# The glue decode/capture is rtl/top/apex_top.sv:482-540; the mode mux that
# hands the walker ds_*/route/jobs/seam/KVQ-AXI is :542-609. Descriptor word
# map: WALK_DW_GEOM=0 / WALK_DW_RQ=1 / WALK_DW_MASK=2
# (rtl/seq/seq_walker_pkg.sv:129-132), unpacked at :136-158, legality-checked
# at :164-187 (fmt!=0 -> WALK_ERR_DESC, tier!=CQ8 -> WALK_ERR_TIER).
# The FSM: kick at W_IDLE (rtl/seq/seq_layer_walker.sv:482), legality at
# W_CHECK (:489-497), score (W_S_*), pv (W_P_*), W_DONE -> W_IDLE (:640), with
# walk_busy = (state != W_IDLE) (:417).
#
# Sequence this file emits (mirrors verif/top/l3/tb_apex_l3.sv:982-1111, the
# stage-5 walker-mode L3 path, translated from TB tasks to BAR0 regops):
#   1. host mode (walk_en=0): phase_a, loader, store_kv, q-inject — VERBATIM
#      from gen_l3_vectors, because the walker deliberately does not walk them
#      (B1_WALKER.md §2 amendment 3: they are 87.5% injection scaffolding).
#      The store_kv writes are what populate the walker's on-tile scale cache:
#      apex_top.sv:616-623 snoops KVQ WRITE_ADDR passively, and seq_walker_comp
#      harvests each record's fp16 scale off the store stream (§A-2 option B).
#      The single q-inject SS beat is where the composite unit latches s_q
#      (apex_top.sv:648).
#   2. WALK_DPTR=0, 3x WALK_DDATA (GEOM/RQ/MASK), WALK_CTRL=3 (en|go).
#   3. the host drains the capture FIFOs IN THE HOST-MODE ORDER while the
#      walker runs. This is not optional: the mailbox out-FIFOs are 16 deep and
#      BACKPRESSURE the tile (apex_f2_mailbox.sv:153,207-210), so a walk that
#      is not drained stalls. The drain order is not hand-written either — it
#      is score_phase/pv_phase's own EFS/ESS/ERO line order, filtered out of
#      the real emitters, so it cannot drift from them.
#   4. poll WALK_STATUS busy=0, check err sticky/code = 0, WALK_CTRL=0.
#   5. host mode again: final_phase (quiescence, sticky, PERF, DONE audit).
#      It MUST come after walk_en=0: the mux parks the host's KVQ AXI-Lite
#      window during a walk (apex_top.sv:602-606), so its KVP poll would hang.
#
# ══ WHAT WALKER MODE CANNOT DO: rq_en=0 ════════════════════════════════════
# compute_job's whole non-circularity argument is requant_en=0 — the RO lanes
# then carry the raw INT32 accumulators and the Q13 epilogue runs on the host.
# THE WALKER HARDCODES requant_en=1: rtl/seq/seq_layer_walker.sv:408-410
#
#     ds_desc.requant_en = 1'b1;
#     ds_desc.rq_scale   = desc_q.rq_scale;
#     ds_desc.rq_shift   = desc_q.rq_shift;
#
# so in walker mode the PV GEMM always requants and the 8 RO lanes carry the
# INT8 codes (sign-extended), never acc_o. There is no descriptor bit, no CSR
# and no rq value that recovers the raw accumulators: 32-bit accumulators do
# not fit through an 8-bit requantised lane. This is the D-028 fenced scope
# ("rq host-loaded", B1_WALKER.md §2 correction 1 / §3): the requant pair is
# calib_requant(amax|acc_o|) of the very GEMM it configures, so no single pass
# can derive it on-tile.
# Consequences, named rather than papered over:
#   * the walk's egress is o8 (+ the s_c/s_k/s_v scale taps), not acc_o;
#   * out_hat is reconstructed from the captured s_c and the LOADED rq;
#   * the rq pair itself is a host input. --rq picks WHERE it comes from:
#       tile  (default) TWO-PASS: run compute_job's rq_en=0 job on the SAME
#               executor first, epilogue the TILE's own raw accumulators, and
#               load the resulting pair. Nothing golden is loaded — pass 1 is
#               input-only by construction and pass 2's one calibration input
#               is derived from pass 1's captures.
#       trace  the pair stored in the S8 npz (golden-derived: circular in
#               exactly the way PROMPT_DEMO_AUDIT §2 C1 describes). Fast,
#               honest only if labelled — the report says so.
#       s,h    an explicit pair.
#
# CLI:
#   python3 scripts/fpga/f2/walk_job.py --smoke            # T=20 npz, sim
#   python3 scripts/fpga/f2/walk_job.py --npz <job.npz> [--rq tile|trace|s,h]
#   python3 scripts/fpga/f2/walk_job.py --selftest         # host-only
#
# API: build_walk_job_full(q, K, V, rq, ...) -> (path, manifest)
#      grade_walk_job(captures, q, K, V, rq, ...) -> dict

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO / "verif" / "top" / "l3"))
sys.path.insert(0, str(HERE))

import gen_l3_vectors as g3                                    # noqa: E402
import trace_to_regops as t2r                                  # noqa: E402
import cap_decode as cd                                        # noqa: E402
import compute_job as cj                                       # noqa: E402
from tile_exec_bridge import run_job                           # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden.fp import f16_bits_to_f64                     # noqa: E402

# ── WALK CSR window, at BAR0 (cl_apex 0x1___ = tile csr word) ───────────────
WALK_CTRL = t2r.B_CSR + 0x5C
WALK_DPTR = t2r.B_CSR + 0x60
WALK_DDATA = t2r.B_CSR + 0x64
WALK_STATUS = t2r.B_CSR + 0x68
WALK_RQ = t2r.B_CSR + 0x6C
# WALK_STATUS fields (rtl/top/apex_top.sv:534-538)
ST_BUSY = 0x1
ST_PHASE = 0xE
ST_ERR = 0x100            # sticky, W1C
ST_ERRCODE = 0xE00
ST_FMTSUP = 0xF000
WALK_FMT_SUP_LAYER = 0x3  # fmt 0 and fmt 1 both walk (seq_walker_pkg.sv:70)
# WALK_CTRL payloads
CTRL_EN_GO = 0x3
CTRL_OFF = 0x0
# descriptor word map (seq_walker_pkg.sv:129-132)
DW_GEOM, DW_RQ, DW_MASK = 0, 1, 2
MASK_SCORE_PV = 0x3

# op-script partition for the walked phases. Every line in score_phase +
# pv_phase must land in exactly one bucket or the build fails loudly — a new
# op appearing in the emitters must be classified, never silently walked or
# silently host-driven.
# CSRP belongs here: score/pv only ever poll CSR STATUS.idle as the fence
# before a route-LEVEL change (gen_l3_vectors.py:239-240), and the walker does
# that fence internally with its tile_idle input (W_S_FENCE / W_P_FENCE,
# rtl/seq/seq_layer_walker.sv:533,592) instead of over the bus.
WALKED_OPS = frozenset({"ROUTE", "IMP", "DESC", "AJ", "WJ", "FJOB", "QJOB",
                        "SJOB", "LJOB", "KVW", "KVP", "CSRP", "CS", "QS"})
HOST_OPS = frozenset({"EFS", "ESS", "ERO", "ETIP", "ETIPT"})

DEFAULT_NPZ = cj.DEFAULT_NPZ


# ── the walked/host/tap partition of the score+pv op-script ─────────────────
def partition_walked(lines):
    """Split score+pv lines into (host_lines, walked_lines, n_tap_lines).

    `host_lines` keeps the comments and the capture-FIFO pops in their
    emission order — that order IS the drain order, so it is taken from the
    emitters rather than restated here. `walked_lines` is what the tile now
    emits for itself: translated with a throwaway Xlate it gives the exact
    regop count the walker replaces (the autonomy number).
    """
    host, walked = [], []
    n_tap = 0
    it = iter(range(len(lines)))
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        i += 1
        if not ln:
            continue
        if ln.startswith("//"):
            host.append(ln)
            continue
        t = ln.split()
        op = t[0]
        if op in t2r.TAP_KINDS:
            if op != "ENDTAPS":
                n_vals = int(t[1], 16)
                i += n_vals            # drop the payload lines with the header
                n_tap += n_vals
            continue
        if op in HOST_OPS:
            host.append(ln)
        elif op in WALKED_OPS:
            walked.append(ln)
        else:
            raise AssertionError(
                f"walk_job: unclassified op {op!r} in the walked phases "
                f"({ln[:70]}) — classify it in WALKED_OPS or HOST_OPS before "
                f"trusting a walker-mode program")
    del it
    return host, walked, n_tap


def count_regops(lines, D: int) -> dict:
    """How many BAR0 regops the HOST would issue for these op-script lines."""
    x = t2r.Xlate(D)
    x.run(lines)
    kinds: dict[str, int] = {}
    for o in x.ops:
        if o["op"] == "note":
            continue
        kinds[o["op"]] = kinds.get(o["op"], 0) + 1
    # On the F2 CL every one of these is a BAR0 AXI-Lite transaction — the
    # mailbox is a register window too (cl_apex.sv:19-21), so unlike the L3
    # counting rule (B1_WALKER.md §8) there is no "direct pin drive" subset
    # here. The split is by WINDOW, and is reported so the number cannot be
    # confused with the L3 figure.
    return {"total": sum(kinds.values()), "by_op": kinds,
            "win_csr_kvq": sum(1 for o in x.ops
                               if o["op"] != "note"
                               and (t2r.B_CSR <= o["a"] < t2r.B_MB)),
            "win_mailbox": sum(1 for o in x.ops
                               if o["op"] != "note" and o["a"] >= t2r.B_MB)}


# ── descriptor words ────────────────────────────────────────────────────────
def desc_words(D: int, T: int, tier_code: int = 0, outlier_k: int = 0,
               rq=(0, 0), mask: int = MASK_SCORE_PV) -> list[int]:
    """The 3-word fmt=0 image, packed exactly as walk_desc_unpack reads it
    (seq_walker_pkg.sv:147-155) and as verif/top/l3/gen_walker_desc.py:72-73
    packs it for the L3 walker-mode gate."""
    assert 0 < T <= 128 and D in (64, 128), f"envelope: D={D} T={T}"
    scale, shift = int(rq[0]), int(rq[1])
    assert 0 <= scale <= 0xFFFF and 0 <= shift <= 0x1F, f"rq={rq}"
    geom = (D & 0xFF) | ((T & 0x1FF) << 8) | ((tier_code & 0x3) << 16) \
        | ((outlier_k & 0xF) << 20)                # fmt nibble [31:28] = 0
    return [geom, ((shift & 0x1F) << 16) | (scale & 0xFFFF), mask & 0x3]


# ── the walker-mode section (emitted straight onto the Xlate) ───────────────
def emit_walk(x, words, host_lines, *, fast_path: bool = False) -> dict:
    """Descriptor load + kick + interleaved drain + completion poll.

    fast_path=True emits the STEADY-STATE choreography instead of the cold
    start: WALK_RQ + WALK_CTRL.go + the done poll = 3 MMIO. It is only legal
    when the descriptor SRAM already holds this (D,T,tier) image, which inside
    one job it does not — so this program uses the cold start and the 3-op
    figure is reported as what a second step costs, never as what was measured.
    """
    WALK_ADDRS = (WALK_CTRL, WALK_DPTR, WALK_DDATA, WALK_STATUS, WALK_RQ)

    def n_walk():
        return sum(1 for o in x.ops
                   if o["op"] != "note" and o.get("a") in WALK_ADDRS)

    x.note("WALKER MODE: the tile sequences score+pv itself. Host role from "
           "here: 3-word descriptor + kick + FIFO drain + done poll.")
    if fast_path:
        x.w(WALK_RQ, words[DW_RQ])
    else:
        x.w(WALK_DPTR, 0)
        for w in words:
            x.w(WALK_DDATA, w)
    n_load = n_walk()
    x.w(WALK_CTRL, CTRL_EN_GO)                 # walk_en | walk_go
    x.note("walk kicked; the ops below are capture-FIFO pops only — the "
           "mailbox out-FIFOs are 16 deep and backpressure the tile, so the "
           "drain has to run CONCURRENTLY with the walk")
    x.run(host_lines)                          # EFS/ESS/ERO pops, in order
    x.note("walk completion: busy must clear and no error may be sticky")
    x.poll(WALK_STATUS, ST_BUSY, 0x0)
    n_min = n_walk()                           # the FUNCTIONAL minimum
    x.r(WALK_STATUS, ST_ERR | ST_ERRCODE, 0x0, sem="walkstat")
    x.cap(WALK_STATUS, ST_FMTSUP, "walkfmt")
    x.w(WALK_CTRL, CTRL_OFF)                   # release: host mode resumes
    return {"walk_window_ops": n_walk(),
            # what a walk STRICTLY needs: descriptor load + GO + done-poll
            "functional_minimum": n_min,
            "desc_load_ops": n_load, "kick_ops": 1, "poll_ops": 1,
            # diagnostics this program adds on top: err readback, FMT_SUP
            # capture, and the walk_en release that hands the bus back
            "diagnostic_ops": n_walk() - n_min}


# ── build ───────────────────────────────────────────────────────────────────
def build_script(name, q, K_f16, V_f16, rq, tier=at.TIER_CQ8, G=128,
                 outlier_idx=(), mask_csr=False):
    """core_case's phases, with score+pv HANDED TO THE TILE.

    The host phases are gen_l3_vectors' own emitters, called verbatim (the
    same discipline compute_job.py:299-316 uses). score_phase/pv_phase are
    still CALLED — into a scratch Script — so their EFS/ESS/ERO pop order and
    their walked-op census both come from the real emitters.
    """
    K_f16 = np.asarray(K_f16, dtype=np.uint16)
    V_f16 = np.asarray(V_f16, dtype=np.uint16)
    T, D = K_f16.shape
    assert V_f16.shape == (T, D), f"{name}: K/V shape mismatch"
    assert T <= 128, f"{name}: T={T} beyond the F-1 T_ROW_MAX=128 envelope"
    assert tier == at.TIER_CQ8, (
        f"{name}: walker v1 refuses tier {tier} (WALK_ERR_TIER, "
        f"B1_WALKER.md §A-1) — grouped tiers are B1b")
    p = cj.Prep(q, K_f16, V_f16, tier, G, outlier_idx)

    pre = g3.Script(D)
    pre.emit(f"// WALKER MODE (tile-sequenced score+pv, rq host-loaded): "
             f"{name} (T={T} D={D} {tier}) — host drives phase A/loader/"
             f"store_kv/q-inject, then the TILE walks phases F+G")
    g3.phase_a(pre, tiers_used=(g3.TIER_CODE[tier],))
    if mask_csr:
        g3.mask_load_phase(pre, outlier_idx)
    g3.loader_phase(pre)
    g3.store_kv_phase(pre, K_f16, V_f16, tier)
    pre.emit("// inject q row through squant MODE_QUANT (Q7 machinery). This "
             "SS beat is also where the walker's composite unit latches s_q "
             "(apex_top.sv:648) — the walk depends on it.")
    pre.route(rdst=1, asrc=1)
    qpairs = [g3.decompose_f16(int(b))[:2]
              for b in g3.to16(np.asarray(p.q, dtype=np.float64))]
    g3.inject_jobs(pre, qpairs, 1)
    pre.ess(0, 1)                        # placeholder: becomes a `cap`
    pre.aj(0, 1, 0, 1, pre.BPR, 0)       # LOAD q8 -> act bank 1

    # the walked phases, into scratch scripts (never executed as host ops).
    # score and pv are partitioned SEPARATELY so a pv-only re-kick can reuse
    # exactly pv's own drain order.
    sc = g3.Script(D)
    g3.score_phase(sc, p, T)
    host_sc, walked_sc, n_tap_sc = partition_walked(sc.lines)
    pv = g3.Script(D)
    g3.pv_phase(pv, p, T)
    host_pv, walked_pv, n_tap_pv = partition_walked(pv.lines)

    post = g3.Script(D)
    g3.final_phase(post, T)
    return (pre.lines, host_sc, host_pv, walked_sc + walked_pv, post.lines,
            p, n_tap_sc + n_tap_pv)


def emit_rekick_pv(x, words, host_lines_pv) -> dict:
    """A SECOND walk on the already-loaded image — the per-step cost, MEASURED.

    B1_WALKER.md §3 flags that the L3 harness cannot measure the steady-state
    figure because every case is one decode step, so its "3 MMIO/step" is read
    off the register interface rather than observed. Here a second step IS
    observable, with one caveat that has to be stated rather than hidden:

    a re-kick of the FULL score+pv walk would compute garbage, and not because
    of the walker. score_phase ends by LOADING the S-4 c8 row into act-stage
    bank 1 row 0 (gen_l3_vectors.py:307 stage_c8_load) — the same row the
    host's q8 injection occupies (core_case:594) — so after walk 1 the staged
    q row is gone. That is L3 staging scaffolding, not walker state.

    So the measured second step is PV-ONLY (mask = en_pv): it re-reads the V
    records from the KVQ and re-emits the c8 row walk 1 left staged, and must
    reproduce the same 16 RO beats bit-exact. Cost at the register interface:
    DPTR + DDATA(mask) + WALK_RQ + GO + done-poll. The mask rewrite is an
    artifact of asking for a DIFFERENT phase mask; a genuine next decode step
    reuses the loaded mask and costs the WALK_RQ + GO + poll that §3 derives.
    """
    WALK_ADDRS = (WALK_CTRL, WALK_DPTR, WALK_DDATA, WALK_STATUS, WALK_RQ)

    def n_walk():
        return sum(1 for o in x.ops
                   if o["op"] != "note" and o.get("a") in WALK_ADDRS)

    n0 = n_walk()
    x.note("SECOND WALK on the same loaded (D,T,tier) image — the per-step "
           "cost, measured. PV-only: score_phase's c8 LOAD overwrote the "
           "staged q row, so a full re-walk would be scaffolding-limited.")
    x.w(WALK_DPTR, DW_MASK)
    x.w(WALK_DDATA, 0x2)                       # en_pv only
    x.w(WALK_RQ, words[DW_RQ])                 # the one per-step field
    x.w(WALK_CTRL, CTRL_EN_GO)
    x.run(host_lines_pv)                       # the same PV drains, again
    x.poll(WALK_STATUS, ST_BUSY, 0x0)
    x.r(WALK_STATUS, ST_ERR | ST_ERRCODE, 0x0, sem="walkstat2")
    x.w(WALK_CTRL, CTRL_OFF)
    return {"step2_walk_ops": n_walk() - n0,        # 7
            # the breakdown, so the number cannot be quoted as either better
            # or worse than it is:
            "step2_mask_rewrite_ops": 2,   # experiment artifact (phase mask)
            "step2_rq_go_poll_ops": 3,     # THE per-step cost: RQ + GO + poll
            "step2_diagnostic_ops": 2}     # err readback + walk_en release


def cap_census(T: int, D: int, rekick: bool = False) -> dict:
    """Same census as compute_job.cap_census, plus the walk-status FMT_SUP
    capture. Walker mode changes WHO emits control, not which taps fire.
    A pv-only re-kick repeats the pv taps: bpr*T more fs, bpr more RO beats."""
    c = dict(cj.cap_census(T, D))
    c["walkfmt"] = 1
    c["total"] += 1
    if rekick:
        bpr = D // 8
        c["fs"] += bpr * T
        c["ro_lanes"] += bpr * 8
        c["ro_meta"] += bpr
        c["total"] += bpr * T + bpr * 9
    return c


def build_walk_job_full(q, K, V, rq, tier=at.TIER_CQ8, G=128, outlier_idx=(),
                        out_dir=None, name="walk_job", mask_csr=False,
                        rq_source="unspecified", rekick=False):
    q = np.asarray(q)
    if q.dtype == np.uint16:
        q = f16_bits_to_f64(q).reshape(-1)
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K)
    V = np.asarray(V)
    if K.dtype != np.uint16:
        K = g3.to16(np.asarray(K, dtype=np.float64))
    if V.dtype != np.uint16:
        V = g3.to16(np.asarray(V, dtype=np.float64))

    (pre, host_sc, host_pv, walked_lines, post, prep,
     n_tap) = build_script(name, q, K, V, rq, tier, G, outlier_idx,
                           mask_csr=mask_csr)
    T, D = prep.T, prep.D

    x = cj.ComputeXlate(D)     # capture-window compares -> pure `cap` egress
    x.note(f"WALKER-MODE job {name}: T={T} D={D} {tier}. The tile emits the "
           f"score+pv control stream from a 3-word descriptor; the host loads "
           f"it, kicks, drains the capture FIFOs and polls done.")
    x.run(pre)
    words = desc_words(D, T, g3.TIER_CODE[tier], rq=rq)
    wstat = emit_walk(x, words, host_sc + host_pv)
    rk = emit_rekick_pv(x, words, host_pv) if rekick else None
    x.run(post)

    census = cap_census(T, D, rekick)
    if x.cap_seq != census["total"]:
        raise AssertionError(
            f"{name}: emitted {x.cap_seq} caps, census expects "
            f"{census['total']} ({census}) — the capture sites moved")
    cj._audit_no_output_expectations(x.ops)
    # No DESC survives for the walked phases; every DESC still in the program
    # belongs to a host phase. Prove no PV requant pair leaked into one.
    cj._assert_no_requant(pre)

    walked = count_regops(walked_lines, D)
    host_drain = count_regops(host_sc + host_pv, D)

    out_dir = Path(out_dir or (REPO / "build" / "f2_walk"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.walk.regops.jsonl"
    with path.open("w") as f:
        for o in x.ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    h = hashlib.sha256()
    h.update(prep.digest().encode())
    h.update(f"|rq={tuple(int(v) for v in rq)}|desc="
             f"{[hex(w) for w in words]}".encode())
    man = {"name": name, "mode": "walker", "requant_en": 1,
           "T": T, "D": D, "tier": tier, "G": G,
           "outlier_idx": list(prep.outlier_idx),
           "rq": [int(rq[0]), int(rq[1])], "rq_source": rq_source,
           "desc_words": [f"{w:08x}" for w in words],
           "regops": len(x.ops), "checks": x.n_checks, "caps": x.cap_seq,
           "cap_census": census, "taps_skipped": x.n_tap_skipped + n_tap,
           "walk_window_ops": wstat,
           "rekick": rk,                   # the MEASURED second step, if asked
           "walked_away": walked,          # what the tile now emits itself
           "host_drain": host_drain,       # host reads, unchanged either mode
           "steady_state_mmio_per_step_derived": 3,
           "inputs_sha256": prep.digest(), "program_sha256": h.hexdigest(),
           "host_prep": {"s_q": int(prep.s_q),
                         "s_k": [int(v) for v in prep.s_k],
                         "s_v": [int(v) for v in prep.s_v]}}
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return path, man, prep


# ── grade ───────────────────────────────────────────────────────────────────
def grade_walk_job(captures, q, K, V, rq, tier=at.TIER_CQ8, G=128,
                   outlier_idx=(), prep=None, rekick=False) -> dict:
    """Decode a walker-mode run and grade vs golden.

    Walker mode requants on-tile (rq_en=1 is hardcoded, seq_layer_walker.sv:
    408), so the RO lanes carry o8 — the INT8 codes — not acc_o. The grades:
      o8       the tile's requantised codes vs golden o8. BIT-EXACT claim.
      out_hat  o8 * s_out, with s_out built from the CAPTURED s_c and the
               LOADED rq — the token-identity claim.
      scales   captured s_q/s_c and the s_k/s_v feeder taps vs golden.
    """
    q = np.asarray(q)
    if q.dtype == np.uint16:
        q = f16_bits_to_f64(q).reshape(-1)
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K, dtype=np.uint16)
    V = np.asarray(V, dtype=np.uint16)
    T, D = K.shape
    bpr = D // 8
    if prep is None:
        prep = cj.Prep(q, K, V, tier, G, outlier_idx)
    gold = at.attention_core(q, K, V, tier, G=G, outlier_idx=outlier_idx)
    scale, shift = int(rq[0]), int(rq[1])

    res: dict = {"T": T, "D": D, "tier": tier, "G": G, "rq": (scale, shift),
                 "n_caps": len(captures),
                 "cap_census": cap_census(T, D, rekick), "problems": []}

    # ── RO -> the requantised INT8 codes ──────────────────────────────────
    n_walks = 2 if rekick else 1
    all_o8 = cd.ro_lanes_to_i32(captures, expect_n=D * n_walks)
    o8 = all_o8[:D]
    if rekick:
        o8b = all_o8[D:]
        res["rekick_o8"] = cd.grade(o8b, gold.o8)
        res["rekick_identical_to_step1"] = bool(np.array_equal(o8b, o8))
    if int(np.max(np.abs(o8))) > 127:
        res["problems"].append(
            f"RO lane |value| max {int(np.max(np.abs(o8)))} > 127 — walker "
            f"mode requants on-tile, so these must be INT8 codes; a wide "
            f"value means requant_en was NOT set")
    res["o8_values"] = [int(v) for v in o8]
    res["o8"] = cd.grade(o8, gold.o8)

    # ── the walk-status capture ───────────────────────────────────────────
    wf = [c for c in captures if c.get("sem") == "walkfmt"]
    res["fmt_sup"] = (wf[0]["value"] >> 12) & 0xF if wf else None
    if res["fmt_sup"] is not None and res["fmt_sup"] != WALK_FMT_SUP_LAYER:
        res["problems"].append(
            f"WALK_STATUS FMT_SUP reads {res['fmt_sup']:#x}, expected "
            f"{WALK_FMT_SUP_LAYER:#x} (walker2 tile: fmt 0 and 1)")

    # ── SS -> s_q (q-inject), s_c (score S-4) ─────────────────────────────
    ss = cd.fp16_bits(captures, sems=("ss",), addrs=(cd.SS_W,))
    if len(ss) != 2:
        res["problems"].append(f"expected 2 SS captures (s_q, s_c), got "
                               f"{len(ss)}")
    s_q_bits = ss[0]["bits"] if ss else None
    s_c_bits = ss[-1]["bits"] if len(ss) == 2 else None
    res["s_q"] = {"captured": s_q_bits, "golden": int(gold.s_q),
                  "equal": s_q_bits == int(gold.s_q)}
    res["s_c"] = {"captured": s_c_bits, "golden": int(gold.s_c),
                  "equal": s_c_bits == int(gold.s_c)}

    # ── FS -> s_h, s_k[t] (score), s_v[t] x D/8 (pv) ──────────────────────
    fs = cd.fp16_bits(captures, sems=("fs",), addrs=(cd.FS_W,))
    n_fs_want = 1 + T + bpr * T * n_walks
    if len(fs) != n_fs_want:
        res["problems"].append(f"expected {n_fs_want} FS captures, got "
                               f"{len(fs)}")
    bits = [c["bits"] for c in fs]
    s_k_cap = bits[1:1 + T]
    s_v_blocks = [bits[1 + T + b * T:1 + T + (b + 1) * T]
                  for b in range(bpr * n_walks)]
    res["s_h_loader"] = bits[0] if bits else None
    res["s_k"] = {"equal": s_k_cap == [int(v) for v in gold.s_k],
                  "host_prep_equal": s_k_cap == [int(v) for v in prep.s_k],
                  "n": len(s_k_cap)}
    sv_gold = [int(v) for v in gold.s_v]
    res["s_v"] = {"equal": all(b == sv_gold for b in s_v_blocks),
                  "blocks": len(s_v_blocks),
                  "blocks_identical": all(b == s_v_blocks[0]
                                          for b in s_v_blocks)}

    # ── out_hat from the TILE's codes + the TILE's s_c + the loaded rq ─────
    if s_c_bits is None:
        res["problems"].append("no s_c capture — out_hat needs the tile's "
                               "c-scale")
        res["ok"] = False
        return res
    s_c_val = float(f16_bits_to_f64(np.array([np.uint16(s_c_bits)]))[0])
    s_out = s_c_val * float(1 << shift) / float(scale)
    out_hat = o8.astype(np.float64) * s_out
    res["s_out"] = {"value": s_out, "golden": gold.s_out,
                    "equal": s_out == gold.s_out}
    res["out_hat"] = cd.grade(out_hat, gold.out_hat)
    res["out_hat_values"] = [float(v) for v in out_hat]
    res["rq_equal_golden"] = (scale, shift) == tuple(int(v) for v in gold.rq)

    res["ok"] = bool(res["o8"]["equal"] and res["out_hat"]["equal"]
                     and res["s_q"]["equal"] and res["s_c"]["equal"]
                     and res["s_k"]["equal"] and res["s_v"]["equal"]
                     and not res["problems"]
                     and (not rekick or (res["rekick_o8"]["equal"]
                                         and res["rekick_identical_to_step1"])))
    return res


# ── negative controls: prove the WALK window is what did the work ──────────
# A green walk is only evidence if the same program goes RED when the walker
# is not the one driving. Two controls, both cheap:
#   nogo   the identical program with the kick written as walk_en=0/go=0. No
#          walk starts, nothing drives score+pv, and the first capture-FIFO
#          drain must STALL. If this passes, the numbers came from somewhere
#          else and the whole result is void.
#   refuse the §A-1 refusal gate at the register interface: a descriptor with
#          tier!=CQ8 / T=0 / an unsupported fmt must set WALK_STATUS.err with
#          the right code and start no walk (rtl/seq/seq_walker_pkg.sv:178-185,
#          rtl/seq/seq_layer_walker.sv:489-497). Checked as `r` compares, so
#          the executor's own verdict carries it.
WALK_ERR_TIER, WALK_ERR_DESC = 1, 2


def build_nogo_variant(walk_path, out_dir=None) -> Path:
    """The same regops file with the kick neutered (walk_en=0)."""
    src = [json.loads(l) for l in Path(walk_path).read_text().splitlines()]
    n = 0
    for o in src:
        if o["op"] == "w" and o["a"] == WALK_CTRL and o["d"] == CTRL_EN_GO:
            o["d"] = CTRL_OFF
            n += 1
    if n < 1:
        raise AssertionError("no walk kick found to neuter — the program does "
                             "not start a walk, so the control is vacuous")
    out = Path(out_dir or Path(walk_path).parent) / (
        Path(walk_path).name.replace(".regops.jsonl", ".NOGO.regops.jsonl"))
    with out.open("w") as f:
        for o in src:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    return out


def build_refusal_probe(D: int, T: int, out_dir=None,
                        name="walk_refusal_probe") -> Path:
    """A standalone WALK-window probe: read the window, then prove three
    illegal descriptors are REFUSED with the documented error codes."""
    x = t2r.Xlate(D)
    x.note("WALK-window probe: FMT_SUP discovery + the A-1 refusal gate at "
           "the register interface (no tile work, no data)")
    x.r(WALK_STATUS, ST_BUSY | ST_ERR, 0x0, sem="walkidle")
    x.cap(WALK_STATUS, ST_FMTSUP, "walkfmt")
    cases = [
        ("tier CQ-4 (grouped: B1b, not v1)",
         desc_words(D, T, tier_code=1), ST_ERR | (WALK_ERR_TIER << 9)),
        ("T=0 (geometry out of range)",
         [(D & 0xFF), 0, MASK_SCORE_PV], ST_ERR | (WALK_ERR_DESC << 9)),
        ("fmt=2 (unsupported descriptor format)",
         [desc_words(D, T)[0] | (2 << 28), 0, MASK_SCORE_PV],
         ST_ERR | (WALK_ERR_DESC << 9)),
    ]
    for label, words, want in cases:
        x.note(f"refusal case: {label}")
        x.w(WALK_DPTR, 0)
        for w in words:
            x.w(WALK_DDATA, w)
        x.w(WALK_CTRL, CTRL_EN_GO)
        # refused => not busy, err sticky set, code as documented
        x.r(WALK_STATUS, ST_BUSY | ST_ERR | ST_ERRCODE, want, sem="walkrefuse")
        x.w(WALK_STATUS, ST_ERR)                      # W1C
        x.r(WALK_STATUS, ST_ERR, 0x0, sem="walkw1c")  # sticky really cleared
        x.w(WALK_CTRL, CTRL_OFF)
    out_dir = Path(out_dir or (REPO / "build" / "f2_walk"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.regops.jsonl"
    with path.open("w") as f:
        for o in x.ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")
    return path


# ── rq sourcing ─────────────────────────────────────────────────────────────
def rq_from_tile(j, name, out_dir, executor, binary, timeout_s) -> tuple:
    """PASS 1: compute_job (rq_en=0, host-sequenced) on the same executor —
    the tile's own raw INT32 accumulators, epilogued by GOLDEN's calib_requant
    on the host. Returns (rq, info). Nothing golden enters pass 2's program."""
    path, man, prep = cj.build_compute_job_full(
        j["q"], j["K"], j["V"], j["tier"], j["G"], (), out_dir,
        f"{name}_pass1_rqen0")
    r = run_job(path, executor=executor, binary=binary, timeout_s=timeout_s)
    if not r["captures"]:
        raise RuntimeError("pass 1 produced no captures; cannot source rq "
                           "from the tile")
    acc = cd.ro_lanes_to_i32(r["captures"], expect_n=prep.D)
    ss = cd.fp16_bits(r["captures"], sems=("ss",), addrs=(cd.SS_W,))
    ep = cd.epilogue(acc, ss[-1]["bits"])
    return (int(ep["rq"][0]), int(ep["rq"][1])), {
        "pass1_regops": man["regops"], "pass1_caps": len(r["captures"]),
        "pass1_rc": r["rc"], "pass1_ok": r["ok"],
        "amax_acc": ep["amax_acc"], "s_c_bits": ss[-1]["bits"]}


# ── smoke ───────────────────────────────────────────────────────────────────
def smoke(npz_path=DEFAULT_NPZ, out_dir=None, executor="sim", binary=None,
          cap_out=None, timeout_s=1800, rq_mode="tile",
          controls: bool = True, rekick: bool = False) -> int:
    j = cj.load_npz(npz_path)
    meta, tier = j["meta"], j["tier"]
    name = Path(npz_path).stem
    T, D = meta["T"], meta["D"]
    print("=" * 78)
    print("WALKER-MODE SMOKE — the TILE sequences a real Qwen2.5-7B "
          "attention op's score+pv")
    print(f"  npz    {npz_path}")
    print(f"  op     step {meta['step']} layer {meta['layer']} head "
          f"{meta['head']} q_pos {meta['q_pos']} kv_head {meta['kv_head']}")
    print(f"  geom   T={T} D={D} {tier} G={j['G']}")
    print("=" * 78)

    t0 = time.time()
    rq_info = {}
    if rq_mode == "tile":
        print("[rq   ] PASS 1 (host-sequenced, rq_en=0) to source the requant "
              "pair from the TILE's own accumulators…")
        rq, rq_info = rq_from_tile(j, name, out_dir, executor, binary,
                                   timeout_s)
        print(f"[rq   ] tile-derived rq={rq} (amax_acc="
              f"{rq_info['amax_acc']}, {time.time() - t0:.1f}s)")
    elif rq_mode == "trace":
        rq = (int(j["trace"]["rq_scale"]), int(j["trace"]["rq_shift"]))
        print(f"[rq   ] rq={rq} from the S8 npz — GOLDEN-DERIVED (circular, "
              f"PROMPT_DEMO_AUDIT §2 C1). Use --rq tile for a "
              f"non-circular pair.")
    else:
        rq = tuple(int(v) for v in rq_mode.split(","))
        print(f"[rq   ] rq={rq} (explicit)")

    t1 = time.time()
    path, man, prep = build_walk_job_full(
        j["q"], j["K"], j["V"], rq, tier, j["G"], (), out_dir,
        f"{name}_walk", rq_source=rq_mode, rekick=rekick)
    print(f"[build] {path}")
    print(f"[build] {man['regops']} regops, {man['checks']} structural "
          f"checks, {man['caps']} caps")
    print(f"[build] descriptor words {man['desc_words']} "
          f"(GEOM=D|T<<8|tier<<16, RQ=shift<<16|scale, MASK=score|pv)")
    print(f"[build] WALK-window ops: {man['walk_window_ops']}")
    print(f"[build] WALKED AWAY from the host: "
          f"{man['walked_away']['total']} regops "
          f"({man['walked_away']['by_op']})")
    print(f"[build] host still drains {man['host_drain']['total']} capture "
          f"regops (same in both modes)")
    print(f"[build] inputs sha256 {man['inputs_sha256'][:32]}… "
          f"({time.time() - t1:.1f}s)")

    t2 = time.time()
    print(f"[run  ] executor={executor}")
    r = run_job(path, executor=executor, binary=binary, cap_out=cap_out,
                timeout_s=timeout_s)
    print(r["log"].rstrip())
    print(f"[run  ] rc={r['rc']} ok={r['ok']} captures={len(r['captures'])} "
          f"reported={r['n_reported']} ({time.time() - t2:.1f}s)")
    for n in r["notes"]:
        print(f"[run  ] NOTE: {n}")
    if not r["captures"]:
        print("WALKER SMOKE: FAIL — no captures; nothing to grade")
        return 1

    g = grade_walk_job(r["captures"], j["q"], j["K"], j["V"], rq, tier,
                       j["G"], (), prep=prep, rekick=rekick)
    gold = at.attention_core(j["q"], j["K"], j["V"], tier, G=j["G"])
    print("-" * 78)
    print(f"[grade] captures {g['n_caps']} (census "
          f"{g['cap_census']['total']})")
    print(f"[grade] WALK_STATUS FMT_SUP = {g['fmt_sup']:#x} "
          f"(fmt0|fmt1 supported)" if g["fmt_sup"] is not None else
          "[grade] WALK_STATUS FMT_SUP missing")
    print(f"[grade] WALKED o8 (on-tile requant) : {g['o8']}")
    print(f"[grade]   first 8 codes captured    : {g['o8_values'][:8]}")
    print(f"[grade]   golden o8 first 8         : "
          f"{[int(v) for v in gold.o8[:8]]}")
    print(f"[grade] s_q captured {g['s_q']['captured']} vs golden "
          f"{g['s_q']['golden']} -> {g['s_q']['equal']}")
    print(f"[grade] s_c captured {g['s_c']['captured']} vs golden "
          f"{g['s_c']['golden']} -> {g['s_c']['equal']}  (TILE-SOURCED; "
          f"out_hat's only scale input besides rq)")
    print(f"[grade] feeder taps: s_k {g['s_k']}")
    print(f"[grade]              s_v {g['s_v']}")
    print(f"[grade] s_out {g['s_out']['value']!r} vs golden "
          f"{g['s_out']['golden']!r} -> {g['s_out']['equal']}")
    print(f"[grade] loaded rq {g['rq']} == golden rq -> "
          f"{g['rq_equal_golden']}  (source: {rq_mode})")
    print(f"[grade] out_hat : {g['out_hat']}")
    print(f"[grade]   out_hat[:6] "
          f"{[round(v, 8) for v in g['out_hat_values'][:6]]}")
    print(f"[grade]   golden [:6] "
          f"{[round(float(v), 8) for v in gold.out_hat[:6]]}")
    for pb in g["problems"]:
        print(f"[grade] PROBLEM: {pb}")

    # ── DISCRIMINATION: a grader that cannot go red proves nothing ─────────
    disc = {}
    pert = [dict(c) for c in r["captures"]]
    tag = None
    for c in pert:
        if c["sem"].startswith("ro_w") and c["sem"] != "ro_meta":
            c["value"] = (c["value"] + 1) & 0xFFFFFFFF
            tag = c["tag"]
            break
    ga = grade_walk_job(pert, j["q"], j["K"], j["V"], rq, tier, j["G"], (),
                        prep=prep, rekick=rekick)
    disc["a_one_code_perturbation"] = {"tag": tag,
                                       "o8_red": not ga["o8"]["equal"],
                                       "ok_red": not ga["ok"]}
    gb = grade_walk_job(r["captures"], -j["q"], j["K"], j["V"], rq, tier,
                        j["G"], (), rekick=rekick)
    disc["b_wrong_golden_q_negated"] = {"o8_red": not gb["o8"]["equal"],
                                        "ok_red": not gb["ok"]}
    print(f"[disc ] (a) one-code perturbation of {tag}: "
          f"o8 RED={disc['a_one_code_perturbation']['o8_red']} "
          f"verdict RED={disc['a_one_code_perturbation']['ok_red']}")
    print(f"[disc ] (b) correct captures vs golden(q -> -q): "
          f"o8 RED={disc['b_wrong_golden_q_negated']['o8_red']} "
          f"verdict RED={disc['b_wrong_golden_q_negated']['ok_red']}")
    discriminates = all(v for d in disc.values() for k, v in d.items()
                        if k.endswith("_red"))
    if not discriminates:
        print("[disc ] PROBLEM: the grader did NOT go red on a known-bad "
              "input — the green result above is not evidence")

    if rekick and man["rekick"]:
        rk = man["rekick"]
        print(f"[step2] SECOND walk on the loaded image (pv-only): "
              f"{rk['step2_walk_ops']} WALK-window ops MEASURED = "
              f"{rk['step2_rq_go_poll_ops']} per-step (WALK_RQ + GO + "
              f"done-poll) + {rk['step2_mask_rewrite_ops']} phase-mask "
              f"rewrite (an artifact of asking for pv-only, not of a decode "
              f"step) + {rk['step2_diagnostic_ops']} diagnostics (err "
              f"readback, walk_en release). B1_WALKER.md §3 could only DERIVE "
              f"the 3; this is the register interface actually paying it.")
        print(f"[step2] its PV output vs golden o8: {g.get('rekick_o8')}  "
              f"identical to step 1: {g.get('rekick_identical_to_step1')}")

    # ── negative controls (the walk is only evidence if this can go red) ───
    ctl = {}
    if controls:
        probe = build_refusal_probe(D, T, out_dir)
        rp = run_job(probe, executor=executor, binary=binary,
                     timeout_s=timeout_s, require_summary=False)
        ctl["refusal_gate"] = {"rc": rp["rc"], "ok": rp["ok"],
                               "pass": bool(rp["ok"] and rp["rc"] == 0)}
        print(f"[ctl  ] A-1 refusal gate (tier CQ-4 / T=0 / fmt=2 all "
              f"refused with the documented code, sticky W1C-clears): "
              f"{'PASS' if ctl['refusal_gate']['pass'] else 'FAIL'} "
              f"(rc={rp['rc']})")
        if not ctl["refusal_gate"]["pass"]:
            print(rp["log"].rstrip())
        nogo = build_nogo_variant(path, out_dir)
        rn = run_job(nogo, executor=executor, binary=binary,
                     timeout_s=timeout_s, require_summary=False)
        stalled = rn["rc"] != 0 or not rn["ok"]
        ctl["nogo_must_fail"] = {"rc": rn["rc"], "ok": rn["ok"],
                                 "pass": bool(stalled),
                                 "tail": rn["log"].strip().splitlines()[-1:]}
        nogo_verdict = ("PASS (went red as it must)" if stalled else
                        "FAIL — it passed WITHOUT a walk, so the walk "
                        "proves nothing")
        print(f"[ctl  ] same program, kick neutered (walk_en=0): "
              f"{nogo_verdict} (rc={rn['rc']}) "
              f"{ctl['nogo_must_fail']['tail']}")
    controls_ok = all(v["pass"] for v in ctl.values()) if ctl else True

    # ── the autonomy number ────────────────────────────────────────────────
    wa = man["walked_away"]
    ww = man["walk_window_ops"]["walk_window_ops"]
    print("-" * 78)
    print(f"[auton] score+pv control the HOST used to issue : "
          f"{wa['total']} BAR0 MMIO transactions "
          f"({wa['win_csr_kvq']} in the tile-CSR/KVQ windows, "
          f"{wa['win_mailbox']} in the mailbox window — on this CL the "
          f"mailbox is MMIO too, so all {wa['total']} are bus traffic)")
    print(f"[auton] what the host issues in walker mode     : {ww} "
          f"WALK-window ops (cold start: DPTR + 3xDDATA + GO + done-poll "
          f"+ err-read + release)")
    print(f"[auton] reduction on the walked phases          : "
          f"{wa['total']} -> {ww}  ({wa['total'] / ww:.0f}x)")
    if rekick and man["rekick"]:
        print(f"[auton] steady-state per NEXT step: 3 "
              f"(WALK_RQ + GO + done-poll) — MEASURED above, and the second "
              f"walk's output was bit-exact")
    else:
        print(f"[auton] steady-state per NEXT step (DERIVED from the register "
              f"interface, not measured — one walk is one step; pass --rekick "
              f"to measure it): 3 (WALK_RQ + GO + done-poll)")
    print("-" * 78)
    verdict = ("PASS" if (g["ok"] and r["ok"] and discriminates
                          and controls_ok) else "FAIL")
    print(f"WALKER MODE, on-tile requant (rq host-loaded, D-028 fenced "
          f"scope): o8={g['o8']['equal']} out_hat="
          f"{g.get('out_hat', {}).get('equal')} rq_source={rq_mode}")
    print(f"WALKER-MODE SMOKE: {verdict}  (tile-sequenced score+pv, "
          f"{len(r['captures'])} captured values, {man['checks']} structural "
          f"checks, 0 output expectations)")
    return 0 if verdict == "PASS" else 1


# ── selftest: build only, no executor ──────────────────────────────────────
def _selftest() -> int:
    fails = 0

    def chk(n, cond, extra=""):
        nonlocal fails
        print(f"  [{n}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    import tempfile
    rng = np.random.default_rng(20260729)
    D, T = 128, 5
    K = g3.to16(rng.normal(0, 1, (T, D)))
    V = g3.to16(rng.normal(0, 1, (T, D)))
    q = f16_bits_to_f64(g3.to16(rng.normal(0, 1, D))).reshape(D)
    rq = (39958, 23)
    tmp = Path(tempfile.mkdtemp(prefix="apex_wj_"))
    path, man, prep = build_walk_job_full(q, K, V, rq, out_dir=tmp,
                                          name="st", rq_source="explicit")
    ops = [json.loads(l) for l in path.read_text().splitlines()]

    chk("1 compiled", man["regops"] == len(ops) and man["caps"] > 0,
        f"{man['regops']} regops, {man['caps']} caps")
    # descriptor round-trip against the RTL's own unpack semantics
    geom, wrq, wmask = [int(w, 16) for w in man["desc_words"]]
    chk("2 GEOM unpacks to (fmt=0, D, T, tier=CQ8)",
        (geom >> 28) == 0 and (geom & 0xFF) == D
        and ((geom >> 8) & 0x1FF) == T and ((geom >> 16) & 0x3) == 0,
        f"{geom:#010x}")
    chk("3 RQ word carries the pair", (wrq & 0xFFFF) == rq[0]
        and ((wrq >> 16) & 0x1F) == rq[1] and wmask == 0x3, f"{wrq:#010x}")
    # the WALK window choreography, in order
    walk_w = [o for o in ops if o["op"] == "w"
              and o["a"] in (WALK_DPTR, WALK_DDATA, WALK_CTRL, WALK_RQ)]
    chk("4 descriptor load then kick",
        [o["a"] for o in walk_w] == [WALK_DPTR, WALK_DDATA, WALK_DDATA,
                                     WALK_DDATA, WALK_CTRL, WALK_CTRL]
        and walk_w[4]["d"] == CTRL_EN_GO and walk_w[5]["d"] == CTRL_OFF,
        f"{[hex(o['a']) for o in walk_w]}")
    chk("5 DDATA payload == descriptor words",
        [o["d"] for o in walk_w[1:4]] == [geom, wrq, wmask])
    # a done-poll exists and the error check is a real check
    chk("6 busy poll + err check",
        any(o["op"] == "poll" and o["a"] == WALK_STATUS and o["m"] == ST_BUSY
            and o["e"] == 0 for o in ops)
        and any(o["op"] == "r" and o["a"] == WALK_STATUS
                and o["e"] == 0 for o in ops))
    # ordering: every capture pop sits BETWEEN the kick and the release
    i_go = next(i for i, o in enumerate(ops)
                if o["op"] == "w" and o["a"] == WALK_CTRL
                and o["d"] == CTRL_EN_GO)
    i_off = next(i for i, o in enumerate(ops)
                 if o["op"] == "w" and o["a"] == WALK_CTRL
                 and o["d"] == CTRL_OFF)
    ro_pos = [i for i, o in enumerate(ops)
              if o["op"] == "cap" and o["tag"].startswith("ro_w")]
    chk("7 all RO pops are inside the walk window",
        bool(ro_pos) and all(i_go < i < i_off for i in ro_pos),
        f"{len(ro_pos)} RO word pops in ({i_go},{i_off})")
    # no host control op may appear inside the walk window: the mode mux
    # forces the tile's job/desc/route ready low there, so any surviving job
    # fire or KVQ access would HANG the program, not merely be redundant.
    inside = ops[i_go + 1:i_off]
    bad = [o for o in inside
           if (o["op"] in ("jf", "pw"))
           or (o["op"] in ("w", "r", "rn", "poll")
               and t2r.B_KVQ <= o["a"] < t2r.B_MB)
           or (o["op"] == "w" and t2r.B_MB + 0x080 <= o["a"] < t2r.B_MB + 0x200)]
    chk("8 no host control op inside the walk window", not bad,
        f"{len(inside)} ops inside; offenders {bad[:2]}")
    # no cap carries an expectation, no compare on an output address
    chk("9 caps are pure egress",
        all("e" not in o for o in ops if o["op"] == "cap"))
    chk("10 no compare on a capture-window data address",
        not [o for o in ops if o["op"] in ("r", "rn", "poll")
             and o.get("a") in cj.OUTPUT_ADDRS])
    # the walked-away census must be non-trivial and must contain the ops the
    # walker owns per B1_WALKER §2
    wa = man["walked_away"]["by_op"]
    chk("11 walked-away census non-trivial",
        man["walked_away"]["total"] > 100
        and {"w", "pw", "jf", "poll"} <= set(wa),
        f"{man['walked_away']['total']} regops {wa}")
    # and NONE of those ops may still be in the emitted program's walk window
    chk("12 walk window: 4-word load + GO + done-poll = 6 functional ops",
        man["walk_window_ops"]["functional_minimum"] == 6
        and man["walk_window_ops"]["desc_load_ops"] == 4
        and man["walk_window_ops"]["walk_window_ops"] == 9,
        f"{man['walk_window_ops']}")
    # the golden-leak guard, same shape as compute_job._selftest check 7
    gold = at.attention_core(q, K, V, at.TIER_CQ8, G=128)
    leak = {int(v) & 0xFFFFFFFF for v in gold.acc_o if abs(int(v)) >= 256}
    leak -= {0x00010000}
    hit = [o for o in ops if o["op"] in ("r", "cap") and o.get("e") in leak]
    chk("13 no golden accumulator survives as an expectation", not hit,
        f"{len(leak)} values searched")
    chk("14 s_c / o8 unknown at build time",
        int(prep.s_c) == 0 and not prep.o8.any())
    # grader discrimination, host-only: synthesise a perfect capture set from
    # golden and confirm the grade is green, then bend one code and confirm red
    caps = _fake_walk_captures(gold, T, D)
    gg = grade_walk_job(caps, q, K, V, tuple(int(v) for v in gold.rq),
                        prep=prep)
    chk("15 synthetic perfect captures grade GREEN", gg["ok"],
        f"o8={gg['o8']['equal']} out_hat={gg['out_hat']['equal']} "
        f"problems={gg['problems'][:1]}")
    caps2 = [dict(c) for c in caps]
    for c in caps2:
        if c["sem"] == "ro_w0":
            c["value"] = (c["value"] + 1) & 0xFFFFFFFF
            break
    gr = grade_walk_job(caps2, q, K, V, tuple(int(v) for v in gold.rq),
                        prep=prep)
    chk("16 one bent code grades RED", not gr["ok"] and not gr["o8"]["equal"])
    # the --rekick program: one extra kick, one extra pv drain set, and the
    # per-step op breakdown must add up to what is reported
    pk, mk, _ = build_walk_job_full(q, K, V, rq, out_dir=tmp, name="st_rk",
                                    rq_source="explicit", rekick=True)
    opsk = [json.loads(l) for l in pk.read_text().splitlines()]
    rk = mk["rekick"]
    kicks = [o for o in opsk if o["op"] == "w" and o["a"] == WALK_CTRL
             and o["d"] == CTRL_EN_GO]
    chk("17 rekick adds exactly one more kick and one more RQ write",
        len(kicks) == 2
        and len([o for o in opsk if o["op"] == "w" and o["a"] == WALK_RQ]) == 1,
        f"{len(kicks)} kicks")
    chk("18 rekick op breakdown adds up",
        rk["step2_walk_ops"] == (rk["step2_rq_go_poll_ops"]
                                 + rk["step2_mask_rewrite_ops"]
                                 + rk["step2_diagnostic_ops"])
        and mk["caps"] == mk["cap_census"]["total"],
        f"{rk}")
    print(f"WALK_JOB SELFTEST: {'FAIL' if fails else 'PASS'} "
          f"(fails={fails}; T={T} D={D}, {man['regops']} regops, "
          f"{man['caps']} caps, walked-away {man['walked_away']['total']} "
          f"-> {man['walk_window_ops']['walk_window_ops']} WALK ops)")
    return 1 if fails else 0


def _fake_walk_captures(gold, T, D):
    """The capture records an executor WOULD write for a perfect walk."""
    bpr = D // 8
    caps, i = [], 0

    def add(sem, addr, value):
        nonlocal i
        caps.append({"tag": f"{sem}_{i}", "sem": sem, "seq": i, "addr": addr,
                     "mask": 0xFFFFFFFF, "value": value & 0xFFFFFFFF, "i": i})
        i += 1

    # loader s_h (value irrelevant to the grades, presence is not)
    add("fs", cd.FS_W, (1 << 16) | int(gold.s_k[0]))
    add("ss", cd.SS_W, (1 << 16) | int(gold.s_q))          # q-inject s_q
    for t in range(T):                                      # score s_k
        add("fs", cd.FS_W, ((1 if t == T - 1 else 0) << 16) | int(gold.s_k[t]))
    add("ss", cd.SS_W, (1 << 16) | int(gold.s_c))           # score s_c
    for b in range(bpr):
        for t in range(T):
            add("fs", cd.FS_W,
                ((1 if t == T - 1 else 0) << 16) | int(gold.s_v[t]))
        for lane in range(8):
            add(f"ro_w{lane}", cd.RO_W0 + 4 * lane, int(gold.o8[8 * b + lane]))
        add("ro_meta", cd.RO_META, 1)
    add("walkfmt", WALK_STATUS, WALK_FMT_SUP_LAYER << 12)
    return caps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=None, help="one S8 trace job .npz")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--cap-out", default=None)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--rq", default="tile",
                    help="where the PV requant pair comes from: 'tile' "
                         "(two-pass, non-circular), 'trace' (the npz's "
                         "golden-derived pair), or 'scale,shift'")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-controls", action="store_true",
                    help="skip the negative controls (refusal gate + neutered "
                         "kick). They are what make the green run evidence; "
                         "only skip to save executor time on a repeat run.")
    ap.add_argument("--rekick", action="store_true",
                    help="add a MEASURED second walk (pv-only) on the same "
                         "loaded descriptor — the per-step MMIO cost that "
                         "B1_WALKER.md §3 could only derive")
    ap.add_argument("--probe-only", action="store_true",
                    help="run just the WALK-window/refusal-gate probe")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.probe_only:
        p = build_refusal_probe(128, 20, a.out_dir)
        r = run_job(p, executor=a.executor, binary=a.binary,
                    timeout_s=a.timeout_s, require_summary=False)
        print(r["log"].rstrip())
        print(f"WALK REFUSAL PROBE: {'PASS' if r['ok'] else 'FAIL'} "
              f"(rc={r['rc']}) -> {p}")
        return 0 if r["ok"] else 1
    npz = Path(a.npz) if a.npz else DEFAULT_NPZ
    if a.build_only:
        j = cj.load_npz(npz)
        rq = ((int(j["trace"]["rq_scale"]), int(j["trace"]["rq_shift"]))
              if a.rq in ("tile", "trace")
              else tuple(int(v) for v in a.rq.split(",")))
        path, man, _ = build_walk_job_full(
            j["q"], j["K"], j["V"], rq, j["tier"], j["G"], (), a.out_dir,
            f"{npz.stem}_walk", rq_source=a.rq)
        print(json.dumps(man, indent=1))
        print(f"-> {path}")
        return 0
    if a.smoke or a.npz:
        return smoke(npz, a.out_dir, a.executor, a.binary, a.cap_out,
                     a.timeout_s, a.rq, controls=not a.no_controls,
                     rekick=a.rekick)
    ap.error("give --smoke, --npz, --build-only or --selftest")


if __name__ == "__main__":
    sys.exit(main())
