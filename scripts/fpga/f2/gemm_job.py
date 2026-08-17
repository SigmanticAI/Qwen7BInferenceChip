#!/usr/bin/env python3
# gemm_job.py — a PROJECTION-shaped GEMM (weights x activation) through the
# tile's MXE on the host-driven job interface: the first non-attention op to
# run on the silicon twin.
#
# ══ WHAT THIS IS, IN ONE LINE ══════════════════════════════════════════════
# acc[n] = SUM_k x8[k] * W[k][n]  — INT8 x INT8 -> raw INT32, K split host-side
# per golden's C-KSPLIT (golden/apex_golden/compute.py:39-71), N = one MXE
# array width (8). The tile's RAW INT32 accumulators come off the RO capture
# path (requant_en=0), the host sums the per-chunk partials, and the result is
# graded against golden's OWN wide integer matmul AFTERWARDS.
#
# ══ THE THREE THINGS THAT MADE THIS POSSIBLE WITHOUT NEW RTL ═══════════════
# 1. WEIGHT INGRESS IS ALREADY HOST-DRIVEN. `rt_wgt_src=0` muxes the MXE
#    weight port onto the EXTERNAL `xw` port (rtl/top/apex_top.sv:1887-1889),
#    and the mailbox's WB triple (XW0/XW1/XWP, apex_f2_mailbox) drives that
#    port straight from BAR0. verif/top/wg3/gen_wg3_fuel_case.py:129-134 is
#    the precedent — a REAL Qwen Wq block pushed as `WB` beats — and its own
#    F3 assertion proves those beats are BYTE-IDENTICAL to the DDR fuel
#    image. So the DDR/IB-FUEL leg (absent in the DDR=0 twin and in today's
#    AFI) is an OPTIMIZATION of this path, not a precondition. NOTHING here
#    needs a new image. §"FEASIBILITY" at the bottom of this header.
# 2. K > 128 PER DESCRIPTOR IS REACHABLE. wg3 concluded "k is bounded by the
#    activation row = CFG_D" (its docstring, GEOMETRY) — true for ONE
#    activation EMIT job, but mxe_ctrl's INGEST just counts M*ceil(K/8) act
#    beats (mxe_ctrl.sv:21-26) and ignores the informational beat `last`, so
#    CHAINED stage-buffer EMIT jobs concatenate into one descriptor. That is
#    exactly the trick gen_l3_vectors.stage_c8_emit:321-332 already uses for
#    a 2-row c8 row. Chaining 16 rows of BPR=16 beats gives the frozen
#    contract's full K_MAX = 2048 in ONE descriptor, with the tile's own
#    STAGE_R_MAX=31 rows/bank (cl_apex.sv:260) holding 31*128 = 3968 INT8
#    activation codes at once.
# 3. LOAD CAN ADDRESS A ROW. apex_stage_buf.sv:170-176/258-260 (I-C IC-QPATH)
#    made LOAD's `sel` the BASE ROW, so the 29 staged rows are placed
#    one-per-LOAD-job instead of all overwriting row 0.
#
# ══ THE ACTIVATION-STAGING PROBLEM, AND THE HONEST FIX ═════════════════════
# There is no external ingress for raw INT8 activation codes: the act stage
# buffer's load stream is muxed between the FEEDER (RMSNorm+quant) and SQUANT
# (apex_top.sv:1878-1884) — both RE-QUANTIZE. The squant route is the wg3
# mechanism (`inject_jobs` MODE_QUANT), and its re-quantization
# q8 = clamp(RNE(x/s)), s = f16(amax_row/127) is the IDENTITY iff the row's
# amax is EXACTLY 127. wg3 got that for free by picking the ONE 128-slice
# containing the row's global amax (its F1/F2). A 3584-wide K-split needs it
# on ALL 29 slices, and MODE_QUANT's `cols` is capped at D=128
# (apex_scale_quant.sv:74-76), so one wide 3584-column quant job is illegal.
#
# FIX (stage_plan): channel k* = argmax|x8| (|x8[k*]| == 127, ASSERTED) is
# placed in LANE 0 OF EVERY STAGED ROW. Every row's amax is then 127 by
# construction, s = fp16 1.0, and MODE_QUANT is provably the identity — the
# wg3 F2 argument, made unconditional. k* is a REAL channel and NO synthetic
# value is ever injected; the duplicate lanes are made harmless on the WEIGHT
# side instead: row 0 lane 0 carries the real W[k*] row, every later
# duplicate carries a ZERO weight row, and tail padding does the same. So
# every one of the 3584 real channels contributes EXACTLY ONCE with its real
# weight row. That identity is not asserted by prose:
#     assert xst @ Wst == x8 @ W        (exact int64, stage_plan F3)
# and the 29 fp16-1.0 row scales stay COUNTED CHECKS in the program (the ESS
# taps), so if the tile's re-quantization were not the identity the job goes
# red in simulation, before any grading.
#
# ══ WHAT IS A CHECK AND WHAT IS A CAPTURE ══════════════════════════════════
# The RESULT is pure egress: the 8 RO lane words + the RO meta word become
# `cap` ops with NO expectation (GemmXlate), exactly as
# scripts/fpga/f2/compute_job.py does for attention, so the program cannot
# contain the answer. Everything else stays a counted check: phase A's CSR
# sanity + illegal-descriptor reject, the loader's real RMSNorm feeder scale,
# the 29 staging scales (input-derived), the end-of-run FIFO-empty audit, the
# MB_STATUS drop audit, ERR_STICKY and PERF. _audit_no_result_expectations()
# proves the result claim BY ADDRESS.
#
# ══ FEASIBILITY ON TODAY'S AFI ═════════════════════════════════════════════
# Every op used here is in the 18-job canonical vocabulary that already ran on
# the F2 instance (docs/results/f2_stage2_hw): CSR r/w, KVW/KVP, ROUTE, DESC
# pushes, WB triples, AJ/QJOB/LJOB job fires, QS pushes, and the RO/FS/SS
# capture windows. No fuel CSR, no DDR, no new mailbox register, no RTL delta.
# The cost is host bandwidth: 3 BAR0 ops per weight beat, K beats per
# descriptor.
#
# CLI:
#   python3 scripts/fpga/f2/gemm_job.py --selftest        # host-only
#   python3 scripts/fpga/f2/gemm_job.py --smoke           # build+run+grade
#   python3 scripts/fpga/f2/gemm_job.py --smoke --rows 4  # short K, fast loop
#
# API:
#   stage_plan(x8, W)                        -> Plan
#   build_gemm_job(W_chunk, x_chunk, out_dir, name) -> Path
#   ksplit_matvec(W, x8, ...)                -> dict (partials, acc, grade)

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
import tile_exec_bridge                                        # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import compute as cp                          # noqa: E402

# tile geometry (cl_apex.sv: APEX_CL_D=128, APEX_CL_DM=128, CL_ROWS_MAX=31)
D_TILE = 128                     # stage-buffer row width in INT8 codes
BPR = D_TILE // 8                # lane8 beats per stage row = 16
MXE_N = 8                        # one array width per descriptor
K_MAX = cp.K_MAX                 # 2048 (frozen contract, mxe_ctrl legality)
ROWS_PER_DESC = K_MAX // D_TILE  # 16 staged rows per descriptor
STAGE_R_MAX = 31                 # rows per stage-buffer bank
ACT_BANK = 1                     # bank0 row0 holds the loader token
F16_ONE = 0x3C00                 # fp16 bits of 1.0 — the staging row scale

# the result sites: these become `cap` (pure egress, no expectation)
RESULT_SEMS = frozenset({"ro_meta"} | {f"ro_w{j}" for j in range(MXE_N)})
RESULT_ADDRS = frozenset(
    [t2r.B_MB + t2r.CAP["ro"][1] + 4 * j for j in range(MXE_N)]
    + [t2r.B_MB + t2r.CAP["ro"][2]])

S8_RUN = REPO / "docs/results/s8_7b_token/artifact_trace/run.json"
S8_WEIGHTS = REPO / "build/s8_weights/Qwen2.5-7B-4bit"
DEFAULT_OUT = REPO / "build" / "f2_gemm"


# ══ 1. the staging plan ════════════════════════════════════════════════════
class Plan:
    """The lane-exact staging of one K-long INT8 activation + its weights.

    chan[i] : which real channel lane i carries      (len = rows*128)
    use[i]  : whether lane i's REAL weight row is used (False -> zero row)
    xst     : the INT8 codes as staged, xst[i] = x8[chan[i]]
    Wst     : the INT8 weight rows as pushed, Wst[i] = W[chan[i]] or 0
    """

    def __init__(self, x8, W, kstar, chan, use):
        self.x8 = np.asarray(x8, dtype=np.int64)
        self.W = np.asarray(W, dtype=np.int64)
        self.kstar = int(kstar)
        self.chan = np.asarray(chan, dtype=np.int64)
        self.use = np.asarray(use, dtype=bool)
        self.xst = self.x8[self.chan].astype(np.int64)
        self.Wst = np.where(self.use[:, None], self.W[self.chan], 0)
        self.rows = len(self.chan) // D_TILE
        self.K_total = len(self.chan)

    def restage(self, W) -> np.ndarray:
        """The staged weight rows for ANOTHER column block of the SAME x8.

        The lane permutation (k* in lane 0 of every row, the others in fill
        order, tail pad) is a function of the ACTIVATION alone — so every
        column block of one projection call re-uses it instead of re-deriving
        it. F3 is re-asserted here, per block, against that block's own
        reference product: the permutation is shared, the proof is not.
        """
        W = np.asarray(W, dtype=np.int64)
        assert W.shape[0] == self.x8.shape[0], \
            f"restage: K mismatch {W.shape[0]} vs {self.x8.shape[0]}"
        assert 1 <= W.shape[1] <= MXE_N, f"restage: N={W.shape[1]}"
        assert np.all(np.abs(W) <= 127), "weight codes outside INT8-symmetric"
        Wst = np.where(self.use[:, None], W[self.chan], 0)
        assert np.array_equal(np.einsum("k,kn->n", self.xst, Wst),
                              np.einsum("k,kn->n", self.x8, W)), \
            "F3 (restage): the staged product is not the un-staged product"
        return Wst

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.x8.astype(np.int8)).tobytes())
        h.update(np.ascontiguousarray(self.W.astype(np.int8)).tobytes())
        h.update(f"kstar={self.kstar}|K={self.K_total}".encode())
        return h.hexdigest()

    def chunks(self, rows_per_desc: int = ROWS_PER_DESC):
        """[(row0, nrows, K), ...] — the C-KSPLIT chunking, row-aligned."""
        out = []
        r = 0
        while r < self.rows:
            nr = min(rows_per_desc, self.rows - r)
            out.append((r, nr, nr * D_TILE))
            r += nr
        return out


def stage_plan(x8, W) -> Plan:
    """Lay out x8[K] x W[K, <=8] onto 128-lane stage rows whose every amax is
    exactly 127, so squant MODE_QUANT is the identity — and prove the layout
    computes the same integer product as the un-staged operands.

    F1  |x8[k*]| == 127 for k* = argmax|x8|   (the wg3 F1 premise, globalized)
    F2  golden quant_rows_i8 of every staged row reproduces the row's codes
        with scale fp16 1.0                   (the wg3 F2 premise, per row)
    F3  xst @ Wst == x8 @ W, exactly, in int64
    F4  every real channel appears exactly once with a non-zero-masked row
    """
    x8 = np.asarray(x8, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    Kt = x8.shape[0]
    assert W.shape[0] == Kt, f"shape mismatch: x8[{Kt}] vs W{W.shape}"
    assert 1 <= W.shape[1] <= MXE_N, f"N={W.shape[1]} exceeds MXE_N={MXE_N}"
    assert np.all(np.abs(x8) <= 127), "activation codes outside INT8-symmetric"
    assert np.all(np.abs(W) <= 127), "weight codes outside INT8-symmetric"

    kstar = int(np.argmax(np.abs(x8)))
    amax = int(abs(int(x8[kstar])))
    assert amax == 127, (
        f"F1: max|x8| = {amax} != 127 at k*={kstar}. The MODE_QUANT identity "
        f"needs a row amax of exactly 127 (s = f16(amax/127) must be 1.0); "
        f"an activation whose global amax code is not 127 needs a synthetic "
        f"+127 sentinel lane with a zeroed weight row instead")

    others = [k for k in range(Kt) if k != kstar]
    chan, use, i = [], [], 0
    row = 0
    while True:
        chan.append(kstar)                       # lane 0: the amax anchor
        use.append(row == 0)                     # counted ONCE, in row 0
        for _ in range(D_TILE - 1):
            if i < len(others):
                chan.append(others[i]); use.append(True); i += 1
            else:
                chan.append(kstar); use.append(False)   # tail pad, zero row
        row += 1
        if i >= len(others):
            break
    p = Plan(x8, W, kstar, chan, use)

    # F4: exactly-once coverage
    used = p.chan[p.use]
    assert len(used) == Kt and len(set(int(v) for v in used)) == Kt, \
        f"F4: {len(used)} used lanes cover {len(set(used.tolist()))} distinct " \
        f"channels, need {Kt} each"
    # F3: the staged product IS the un-staged product
    ref = np.einsum("k,kn->n", x8, W)
    got = np.einsum("k,kn->n", p.xst, p.Wst)
    assert np.array_equal(ref, got), \
        f"F3: staged product {got} != reference {ref}"
    # F2: golden's own quantizer must reproduce every staged row verbatim
    for a in range(0, p.K_total, D_TILE):
        rowv = p.xst[a:a + D_TILE].astype(np.float64)
        q8, sb = at.quant_rows_i8(rowv[None, :])
        assert np.array_equal(np.asarray(q8[0], dtype=np.int64),
                              p.xst[a:a + D_TILE]), \
            f"F2: MODE_QUANT is not the identity on staged row {a // D_TILE}"
        assert int(sb[0]) == F16_ONE, \
            f"F2: staged row {a // D_TILE} scale {int(sb[0]):#06x} != fp16 1.0"
    return p


# ══ 2. the program ═════════════════════════════════════════════════════════
class GemmXlate(t2r.Xlate):
    """Xlate with ONLY the result sites turned into `cap` egress.

    compute_job.ComputeXlate also captures the fs/ss scale taps; here they
    stay COUNTED CHECKS on purpose — the ESS scales are the evidence that the
    tile's MODE_QUANT staging was the identity (fp16 1.0 per row), and the
    loader's EFS scale is the real RMSNorm feeder check inherited from the
    verified L3 path. Only the accumulators may not be foreknown.
    """

    def __init__(self, D: int):
        super().__init__(D)
        self.n_capsites = 0

    def r(self, a, m, e, sem="r"):
        if sem in RESULT_SEMS:
            self.cap(a, m, sem)          # NO `e`: cannot fail, cannot leak
            self.n_capsites += 1
            return
        super().r(a, m, e, sem=sem)


def _audit_no_result_expectations(ops) -> None:
    bad = [o for o in ops if o["op"] in ("r", "rn", "poll")
           and o.get("a") in RESULT_ADDRS]
    if bad:
        raise AssertionError(
            f"{len(bad)} result expectation(s) survived, first {bad[0]} — the "
            f"tile's accumulators must be captured, never asserted")


def _beats_to_block(beats, K: int) -> np.ndarray:
    """Inverse of gen_l3_vectors.wgt_beats_ws for j=0 — proves the WB stream
    carries exactly the INT8 weight rows we intended (the wg3 F3 idea, done
    against the beat encoding itself rather than the DDR image)."""
    out = np.zeros((K, MXE_N), dtype=np.int64)
    for p in range(K // 8):
        for c in range(MXE_N):
            b = int(beats[p * MXE_N + c])
            for r in range(8):
                v = (b >> (8 * r)) & 0xFF
                out[8 * p + r, c] = v - 256 if v >= 128 else v
    return out


def build_script(name, W_chunk, x_chunk):
    """The op script: phase A, loader token, staging, projection, tail.

    Every phase but the middle two is an L3 phase called VERBATIM. The
    projection block is gen_wg3_fuel_case.py:129-136 with the act EMIT chained
    over `rows` stage rows instead of one, and rq_en left at its default 0 so
    the RO lanes carry the RAW INT32 accumulators (mxe_top.sv:195-199).
    """
    W_chunk = np.asarray(W_chunk, dtype=np.int64)
    x_chunk = np.asarray(x_chunk, dtype=np.int64)
    K = x_chunk.shape[0]
    assert W_chunk.shape[0] == K, f"{name}: W/x K mismatch"
    assert K % D_TILE == 0, f"{name}: K={K} is not a whole number of stage rows"
    assert 1 <= K <= K_MAX, f"{name}: K={K} outside 1..{K_MAX} (C-3/D-003)"
    rows = K // D_TILE
    assert rows <= STAGE_R_MAX, f"{name}: {rows} rows exceed STAGE_R_MAX"
    # every staged row's amax must be exactly 127 (see stage_plan F1/F2)
    for r in range(rows):
        a = int(np.max(np.abs(x_chunk[r * D_TILE:(r + 1) * D_TILE])))
        assert a == 127, (
            f"{name}: staged row {r} amax {a} != 127 — MODE_QUANT would "
            f"re-scale it and the staged codes would NOT be x_chunk. Run the "
            f"operands through stage_plan() first")
    # N must be the full array width: wgt_beats_ws emits MXE_N beats per
    # K-chunk, which is the stream the loader expects only when N == MXE_N
    # (mxe_ctrl.sv:52-58 generates the MXE_N-N zero columns itself).
    Wp = np.zeros((K, MXE_N), dtype=np.int64)
    Wp[:, :W_chunk.shape[1]] = W_chunk

    s = g3.Script(D_TILE)
    s.emit(f"// PROJECTION GEMM (non-attention): {name} — m=1 k={K} n={MXE_N} "
           f"OP_GEMM_WS, requant_en=0. Weights arrive on the EXTERNAL xw port "
           f"as mailbox WB beats (no DDR/fuel leg); the {rows} activation "
           f"stage rows are CHAINED into ONE descriptor's INGEST.")
    g3.phase_a(s, tiers_used=(0,))
    g3.loader_phase(s)                       # loader token -> act bank0 row0

    s.emit(f"// stage {rows} activation rows through squant MODE_QUANT "
           f"(re-quantization is the IDENTITY: every row's amax is 127, so "
           f"the emitted scale is fp16 1.0 — a COUNTED check per row)")
    s.route(rdst=1, asrc=1)                  # MXE res -> lane32 -> squant -> act
    for r in range(rows):
        rowv = x_chunk[r * D_TILE:(r + 1) * D_TILE].astype(np.float64)
        pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(rowv)]
        g3.inject_jobs(s, pairs, 1)          # 16 K=2 WS jobs -> 128 INT32 v's
        s.ess(F16_ONE, 1)                    # the identity scale
        s.aj(0, ACT_BANK, 0, 1, s.BPR, r)    # LOAD -> act bank1 row r (I-C sel)

    s.emit(f"// the projection: {K} lane8 weight beats on the EXTERNAL xw "
           f"port, act = the {rows} staged rows chained into one INGEST")
    s.route(rdst=0, wsrc=0, asrc=1)          # res -> RO; wgt <- external xw
    beats = g3.wgt_beats_ws(Wp, 0)
    assert len(beats) == K, f"{name}: {len(beats)} beats != K={K}"
    assert np.array_equal(_beats_to_block(beats, K), Wp), \
        f"{name}: the WB beat stream does not decode back to the weight block"
    # mxe_ctrl INGEST-first ordering (stage_c8_emit:321-332): fire the first
    # EMIT, push the descriptor so MXE starts ingesting, then chain the rest —
    # EMIT r+1's job_ready waits on EMIT r's post-skid done, which needs MXE
    # consuming. Weights go LAST: INGEST completes on activation beats alone
    # (mxe_ctrl.sv:307-320), so pushing 2048 beats first would fill the xw
    # FIFO and wedge the `pw` free-space poll.
    s.aj(1, ACT_BANK, 0, 1, s.BPR, 0)
    s.desc(g3.desc_words(g3.OP_GEMM_WS, 1, K, MXE_N))
    for r in range(1, rows):
        s.aj(1, ACT_BANK, 0, 1, s.BPR, r)
    s.wbeats(beats)
    s.ero([0] * MXE_N, 1)                    # -> caps; placeholders, not values

    g3.final_phase(s, 8)                     # T<=8 tail: no frame stickies
    assert "ESTK ffff 0001" in s.lines, "expected the no-frame-error tail"
    return s, rows, len(beats)


# ══ 2b. ON-TILE K accumulation (OP_GEMM_OS accumulate=1) ═══════════════════
#
# ksplit_matvec above splits K host-side and the HOST sums the per-chunk INT32
# partials (golden C-KSPLIT's "host-side" arm). The SAME golden clause offers
# the other arm — "or on-tile via the accumulate flag" (compute.py:39-47) —
# and the array implements it verbatim:
#
#   rtl/mxe/mxe_array.sv:36-52   acc[c][m] <= (clr ? 0 : acc[c][m]) + bottom
#     * OP_GEMM_WS : pass-0 clr=1  -> every job OVERWRITES
#     * OP_GEMM_OS : accumulate=0 -> pass-0 clr=1 (overwrite)
#                    accumulate=1 -> pass-0 clr=0 (RETAIN) — "this is the
#                    cross-descriptor K-dim tiling primitive (split K > K_MAX
#                    ... into several OS descriptors, accumulate=1 on all but
#                    the first)"
#   rtl/mxe/mxe_ctrl.sv:286      clear_job <= !((opcode==OP_GEMM_OS) && accumulate)
#   verif/mxe/smoke/tb_mxe_smoke.sv:405-418  T3 exercises exactly this chain.
# Draining is NON-DESTRUCTIVE (mxe_array.sv:50), so every descriptor's beat can
# be read without disturbing the running total.
#
# ORDERING CONSTRAINT THAT SHAPES THE PROGRAM. Activation staging is done by
# `inject_jobs`, which is itself a stream of OP_GEMM_WS k=2 descriptors — and a
# WS job ALWAYS clears. So a staging job interleaved between two chained OS
# descriptors would WIPE the running accumulator. Hence: stage ALL rows first,
# then run the chain with nothing else touching the array. That is only
# possible because LOAD's `sel` is the base row (header note 3), which lets all
# `rows` staged rows be resident simultaneously (STAGE_R_MAX=31).

def build_chain_script(name, plan, rows_per_desc: int):
    """One op script: stage every row, then chain the K-split ON THE TILE.

    Returns (script, chunks, n_beats). Chunk ci's descriptor carries
    accumulate=(ci > 0), so after the LAST descriptor drains, RO lane n holds
    SUM over ALL K of act*W[.][n] — the host adds nothing.
    """
    rows = plan.rows
    assert rows <= STAGE_R_MAX, \
        f"{name}: {rows} staged rows exceed STAGE_R_MAX={STAGE_R_MAX} — the " \
        f"whole chain must be resident before the first OS descriptor fires"
    chunks = plan.chunks(rows_per_desc)
    for (_r0, _nr, K) in chunks:
        assert 1 <= K <= K_MAX, f"{name}: chunk K={K} outside 1..{K_MAX}"

    s = g3.Script(D_TILE)
    s.emit(f"// ON-TILE K-SPLIT GEMM: {name} — m=1 n={MXE_N}, K_staged="
           f"{plan.K_total} over {len(chunks)} OP_GEMM_OS descriptors "
           f"(accumulate=1 on all but the first). The HOST performs NO "
           f"addition: the LAST descriptor's RO beat already holds the total.")
    g3.phase_a(s, tiers_used=(0,))
    g3.loader_phase(s)

    s.emit(f"// stage ALL {rows} activation rows FIRST (squant MODE_QUANT is "
           f"the identity: every staged row's amax is 127 -> fp16 1.0, a "
           f"COUNTED check per row). Staging uses OP_GEMM_WS k=2 jobs, which "
           f"ALWAYS clear the array accumulators (mxe_ctrl.sv:286), so not "
           f"one of them may run once the OS chain has started.")
    s.route(rdst=1, asrc=1)                  # MXE res -> lane32 -> squant -> act
    for r in range(rows):
        rowv = plan.xst[r * D_TILE:(r + 1) * D_TILE].astype(np.float64)
        pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(rowv)]
        g3.inject_jobs(s, pairs, 1)
        s.ess(F16_ONE, 1)                    # the identity scale
        s.aj(0, ACT_BANK, 0, 1, s.BPR, r)    # LOAD -> act bank1 BASE ROW r

    s.emit(f"// the chain: {len(chunks)} descriptor(s), each re-EMITting its "
           f"own staged rows with its own weight beats. Every descriptor "
           f"drains one m=1 beat (non-destructive readout, mxe_array.sv:50), "
           f"so the caps are the RUNNING PREFIX and the last IS the total.")
    s.route(rdst=0, wsrc=0, asrc=1)          # res -> RO; wgt <- external xw
    n_beats = 0
    for ci, (r0, nr, K) in enumerate(chunks):
        a = r0 * D_TILE
        Wp = np.zeros((K, MXE_N), dtype=np.int64)
        Wp[:, :plan.Wst.shape[1]] = plan.Wst[a:a + K]
        beats = g3.wgt_beats_ws(Wp, 0)
        assert len(beats) == K, f"{name}: {len(beats)} beats != K={K}"
        assert np.array_equal(_beats_to_block(beats, K), Wp), \
            f"{name}: chunk {ci}'s WB stream does not decode back to Wst"
        n_beats += len(beats)
        # mxe_ctrl INGEST-first ordering, verbatim from build_script
        s.aj(1, ACT_BANK, 0, 1, s.BPR, r0)
        s.desc(g3.desc_words(g3.OP_GEMM_OS, 1, K, MXE_N, mode_os=1,
                             acc=1 if ci else 0))
        for r in range(r0 + 1, r0 + nr):
            s.aj(1, ACT_BANK, 0, 1, s.BPR, r)
        s.wbeats(beats)
        s.ero([0] * MXE_N, 1)                # -> caps; placeholders, not values

    g3.final_phase(s, 8)
    assert "ESTK ffff 0001" in s.lines, "expected the no-frame-error tail"
    return s, chunks, n_beats


def build_chain_job(W, x8, out_dir=None, name="chain_job", *,
                    rows_per_desc: int = 1):
    """Compile the WHOLE K (any length) into ONE program whose tile-side
    accumulator is the complete dot product. Returns (path, manifest).

    rows_per_desc=1 keeps every descriptor at the K=128 geometry that has run
    on silicon tens of thousands of times (the S3 projection sweep); larger
    values are legal up to K_MAX but buy nothing here.
    """
    plan = stage_plan(x8, W)
    s, chunks, n_beats = build_chain_script(name, plan, rows_per_desc)

    x = GemmXlate(D_TILE)
    x.note(f"ON-TILE K-SPLIT GEMM {name}: m=1 K_staged={plan.K_total} "
           f"n={MXE_N}, {len(chunks)} OP_GEMM_OS descriptors chained by "
           f"accumulate=1. The RO lanes are `cap` egress — no accumulator, "
           f"partial or total, appears in this program.")
    x.run(s.lines)
    _audit_no_result_expectations(x.ops)
    want_caps = len(chunks) * (MXE_N + 1)
    if x.cap_seq != want_caps:
        raise AssertionError(
            f"{name}: {x.cap_seq} caps emitted, expected {want_caps} "
            f"({len(chunks)} m=1 beats x (8 RO lanes + meta))")

    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.regops.jsonl"
    with path.open("w") as f:
        for o in x.ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    man = {"name": name, "kind": "on-tile-ksplit-gemm", "opcode": "OP_GEMM_OS",
           "m": 1, "n": MXE_N, "requant_en": 0, "accumulate_chain": True,
           "K_real": int(np.asarray(x8).shape[0]), "K_staged": plan.K_total,
           "staged_rows": plan.rows, "kstar": plan.kstar,
           "n_desc": len(chunks), "chunk_k": [int(c[2]) for c in chunks],
           "weight_beats": n_beats, "weight_bytes": n_beats * 8,
           "wgt_src": "external xw (mailbox WB)",
           "regops": len(x.ops), "checks": x.n_checks, "caps": x.cap_seq,
           "cap_sites_rewritten": x.n_capsites,
           "taps_skipped": x.n_tap_skipped,
           "host_arith": "NONE — the tile accumulates across descriptors "
                         "(OP_GEMM_OS accumulate=1); the host reads the last "
                         "beat and adds nothing",
           "acc_i32_expected_by_host_AFTER_the_run": None,
           "inputs_sha256": plan.digest(),
           "path": str(path)}
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return path, man


DS_W_ADDRS = tuple(t2r.B_MB + a for a in t2r.DS_W)


def decode_desc_pushes(ops) -> list[dict]:
    """Every MXE descriptor a compiled program pushes, rebuilt FROM THE FILE.

    trace_to_regops writes DS_W[3..0] = w127_96 … w31_0 then pushes, so four
    consecutive DS_W writes reassemble the 128-bit descriptor. Returns
    [{'i': op index of the last word, 'opcode', 'm', 'k', 'n', 'rq_en',
      'acc', 'os'}] in program order — the shape a structural audit needs
    (e.g. "no weight-stationary job runs inside an accumulate chain").
    """
    out, cur = [], []
    for i, o in enumerate(ops):
        if o["op"] != "w" or o.get("a") not in DS_W_ADDRS:
            continue
        cur.append(int(o["d"]) & 0xFFFFFFFF)
        if len(cur) == 4:
            v = 0
            for w in cur:
                v = (v << 32) | w
            out.append({"i": i, "opcode": v & 0xFF, "m": (v >> 8) & 0xFFF,
                        "k": (v >> 20) & 0xFFF, "n": (v >> 32) & 0xFFF,
                        "rq_en": (v >> 65) & 1, "acc": (v >> 66) & 1,
                        "os": (v >> 67) & 1})
            cur = []
    assert not cur, f"{len(cur)} trailing DS_W write(s): a torn descriptor"
    return out


def chain_totals(captures, *, n_desc: int | None = None) -> dict:
    """Decode an on-tile-chain job's RO captures into the per-descriptor beats.

    Returns {'beats': [[int32 x 8], ...], 'total': [int32 x 8],
             'raw_lane0_u32': u32, 'n_beats': n} where `total` is the LAST
    beat VERBATIM — no addition is performed here. `raw_lane0_u32` is the
    UNTOUCHED 32-bit word the executor read out of the final beat's lane-0
    capture register (no sign extension, no arithmetic): the exact bits a
    caller may MOVE somewhere else.
    """
    bs = cd.ro_beats(captures)
    if n_desc is not None and len(bs) != n_desc:
        raise ValueError(f"decoded {len(bs)} RO beats, expected {n_desc} "
                         f"(one per chained descriptor)")
    if not bs:
        raise ValueError("no RO beats captured")
    lane0 = [c for c in sorted(captures, key=lambda c: c["i"])
             if c.get("sem") == "ro_w0"]
    if len(lane0) != len(bs):
        raise ValueError(f"{len(lane0)} lane-0 captures for {len(bs)} beats")
    return {"beats": [[int(v) for v in b["lanes"]] for b in bs],
            "total": [int(v) for v in bs[-1]["lanes"]],
            "raw_lane0_u32": int(lane0[-1]["value"]) & 0xFFFFFFFF,
            "n_beats": len(bs)}


# ══ 2c. FAT PROGRAMS — stage ONCE, sweep MANY column blocks ════════════════
#
# ══ WHY (the measured economics, from this repo's own silicon runs) ════════
# `build_gemm_job_full` compiles ONE 8-column block. A 0.5B layer-step is
# 2,160 such blocks, and EVERY ONE of them re-stages the SAME activation row
# from scratch: at D=64 a K=896 contraction stages 15 rows through 120 K=2
# squant jobs = 960 injected weight beats + 960 XA pushes + 120 descriptors,
# ~4,830 of the job's 7,742 BAR0 ops (62%) — before a single weight of the
# projection is pushed. Measured on the b64 image (build/p05b_hw_check2.log):
# 2,190 programs, 470 s of executor per layer.
#
# The stage buffer was BUILT for this: apex_stage_buf.sv's own header opens
# with "the attention chain re-uses on-tile tensors across several MXE
# descriptors — h8 rows feed 3 projection matrices x 8 N-column jobs each
# (replay)", and ":Contents persist across jobs and soft aborts". So one
# program can stage the rows ONCE and then fire one OP_GEMM_WS descriptor per
# column block against them. Every WS descriptor clears the array at pass 0
# (mxe_ctrl.sv:286 `clear_job <= !((opcode==OP_GEMM_OS) && accumulate)`), so
# blocks cannot bleed into each other, and each descriptor drains its own m=1
# result beat, which becomes that block's 9 `cap` records.
#
# THE PRECONDITION IS THE BASE-ROW FIX (38ec95c, "apex_stage_buf LOAD now
# treats `sel` as the BASE ROW"). Without it every LOAD lands on row 0 and a
# multi-row staging silently returns garbage — that is exactly what
# docs/results/prompt_on_chip/hw_projection_k2048_FAILED_image_parity_lesson.log
# recorded on agfi-0ae06ea568e5667ba (bit_exact=False on a 16-row descriptor,
# green with rows_per_desc=1). tile_geom.Image.rows_per_desc is the per-image
# gate; this builder REFUSES to emit multi-row staging without it.
#
# ══ THE PUSH-BURST (the second measured cost) ══════════════════════════════
# Both executors implement `pw` as "PEEK the free-slot count, then POKE"
# (f2_host_run.py:165-173, sim_main.cpp:453-460). A PCIe peek is a blocking
# round trip; a poke is posted. Every weight beat costs a peek, and a layer
# pushes ~2.1 M beats. `BurstXlate` replaces DEPTH per-push peeks with ONE
# `poll` for "this FIFO is EMPTY" (free == DEPTH) followed by DEPTH plain
# writes. Overflow is then impossible BY CONSTRUCTION — nothing but the host
# pushes into that FIFO, so pushing exactly DEPTH into a FIFO observed empty
# cannot exceed it — and if that reasoning were wrong anyway the mailbox
# DROPS the push and latches MB_STATUS (apex_f2_mailbox.sv:46-48), which
# every program already reads as a COUNTED CHECK in its `DONE` tail. Two
# independent nets, neither of them prose.

MB_FIFO_DEPTH = 16               # apex_f2_mailbox.sv:153  localparam DEPTH
# mb_rdata for every push register: {8'h0, 15'(DEPTH - cnt), nonempty, 8'h0}
MB_FREE_SHIFT = 9
MB_FREE_MASK = 0x7FFF << MB_FREE_SHIFT
MB_FREE_EMPTY = MB_FIFO_DEPTH << MB_FREE_SHIFT
PUSH_ADDRS = tuple(t2r.B_MB + a for a in
                   (t2r.XA, t2r.XG, t2r.QS, t2r.CS, t2r.XWP, t2r.DS_PUSH))


class BurstXlate(GemmXlate):
    """GemmXlate whose stream pushes are BURSTED behind one emptiness poll.

    `pw a d` = peek-until-free-then-poke. This subclass emits, per FIFO
    address, ONE `poll a (free==DEPTH)` and then up to DEPTH plain `w a d`.
    Same beats, same order, same tile behaviour — strictly fewer PEEKS.

    The budget is per address and is spent only by pushes, because nothing
    else in this vocabulary writes into a mailbox input FIFO. A poll that
    never sees an empty FIFO is a hard failure in BOTH executors (sim_main
    increments fails and aborts the file; f2_host_run.py:186-196 the same),
    so a wedge is loud rather than silent.
    """

    def __init__(self, D: int, depth: int = MB_FIFO_DEPTH,
                 allow_overflow: bool = False):
        super().__init__(D)
        self.depth = int(depth)
        assert 1 <= self.depth <= MB_FIFO_DEPTH or allow_overflow, (
            f"burst depth {self.depth} > the mailbox FIFO depth "
            f"{MB_FIFO_DEPTH} — a burst longer than the FIFO WILL drop beats. "
            f"(allow_overflow=True is the DISCRIMINATOR path: it exists to "
            f"prove the MB_STATUS drop audit bites.)")
        self._budget: dict = {}
        self.n_burst_polls = 0
        self.n_pushes = 0

    def pw(self, a, d):
        if self._budget.get(a, 0) <= 0:
            self.poll(a, MB_FREE_MASK, MB_FREE_EMPTY)
            self.n_burst_polls += 1
            self._budget[a] = self.depth
        self._budget[a] -= 1
        self.n_pushes += 1
        self.w(a, d)


def audit_push_bursts(ops, depth: int = MB_FIFO_DEPTH) -> dict:
    """COMPILE-TIME proof that no burst can overflow a mailbox FIFO.

    Walks the program and, per push address, requires that every blind `w`
    push is covered by a preceding "free == DEPTH" poll with budget left.
    `depth` blind pushes into a FIFO observed EMPTY cannot exceed DEPTH
    entries, because nothing but the host pushes into it — so this walk IS
    the safety argument, checked against the emitted file rather than
    asserted in a comment. (A `pw` self-polls, so it is always legal; it
    resets the budget because it only guarantees ONE free slot.)
    """
    budget: dict = {}
    n_polls = n_blind = 0
    for i, o in enumerate(ops):
        a = o.get("a")
        if a not in PUSH_ADDRS:
            continue
        if o["op"] == "poll":
            if o.get("m") == MB_FREE_MASK and o.get("e") == MB_FREE_EMPTY:
                budget[a] = depth
                n_polls += 1
            continue
        if o["op"] == "pw":
            budget[a] = 0            # self-polled: exactly one slot promised
            continue
        if o["op"] == "w":
            b = budget.get(a)
            if b is None:
                raise AssertionError(
                    f"op {i}: blind push to {a:#06x} with no preceding "
                    f"FIFO-empty poll — the mailbox would drop it on a full "
                    f"FIFO (apex_f2_mailbox.sv:258-263)")
            if b <= 0:
                raise AssertionError(
                    f"op {i}: blind push #{depth + 1} to {a:#06x} since the "
                    f"last FIFO-empty poll — a burst longer than DEPTH="
                    f"{MB_FIFO_DEPTH} WILL drop beats")
            budget[a] = b - 1
            n_blind += 1
    return {"empty_polls": n_polls, "blind_pushes": n_blind, "depth": depth}


def program_cost(ops) -> dict:
    """What a compiled program COSTS the transport, by op class.

    PEEKS are the expensive half: `r`/`rn`/`cap`/`poll` are one BAR0 read
    each, `pw`/`jf` are a read AND a write (both executors poll before the
    push/fire). Pokes are posted writes. Bytes are the JSONL the host must
    ship to the instance. This is a measurement instrument, not an estimate:
    it counts the file.
    """
    if isinstance(ops, (str, Path)):
        text = Path(ops).read_text()
        ops = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        nbytes = len(text)
    else:
        nbytes = sum(len(json.dumps(o, separators=(",", ":"))) + 1 for o in ops)
    peeks = pokes = 0
    per_op: dict = {}
    for o in ops:
        k = o["op"]
        per_op[k] = per_op.get(k, 0) + 1
        if k in ("r", "rn", "cap", "poll"):
            peeks += 1
        elif k in ("pw", "jf"):
            peeks += 1
            pokes += 1
        elif k == "w":
            pokes += 1
    return {"ops": len(ops), "peeks": peeks, "pokes": pokes, "bytes": nbytes,
            "by_op": per_op}


def build_multiblock_script(name, x_chunk, Wblocks, *, allow_multirow=True,
                            load_base_row_override=None):
    """ONE op script: stage `rows` activation rows, then one OP_GEMM_WS
    descriptor per column block against those same rows.

    x_chunk : the STAGED activation codes of this K chunk (every D_TILE-wide
              row's amax exactly 127 — run the operands through stage_plan).
    Wblocks : a sequence of [K, N<=8] INT8 weight blocks, all sharing K.

    load_base_row_override : DISCRIMINATOR ONLY. Forces every LOAD's `sel` to
        this value, which is what a PRE-38ec95c tile did to ALL of them (sel
        ignored, every row landing on row 0). A program emitted with 0 here
        MUST grade RED; if it did not, the base-row fix would not be
        load-bearing and this whole fat path would be resting on nothing.

    Returns (script, rows, n_beats).
    """
    x_chunk = np.asarray(x_chunk, dtype=np.int64)
    K = int(x_chunk.shape[0])
    assert K % D_TILE == 0, f"{name}: K={K} is not a whole number of stage rows"
    assert 1 <= K <= K_MAX, f"{name}: K={K} outside 1..{K_MAX} (C-3/D-003)"
    rows = K // D_TILE
    assert rows <= STAGE_R_MAX, f"{name}: {rows} rows exceed STAGE_R_MAX"
    if rows > 1 and not allow_multirow:
        raise AssertionError(
            f"{name}: {rows} staged rows need apex_stage_buf's BASE-ROW LOAD "
            f"(38ec95c). This image is on the pre-fix list, where every LOAD "
            f"lands on row 0 and the result is silently wrong "
            f"(hw_projection_k2048_FAILED_image_parity_lesson.log). Refusing "
            f"to emit.")
    assert len(Wblocks) >= 1, f"{name}: no weight blocks"
    for r in range(rows):
        a = int(np.max(np.abs(x_chunk[r * D_TILE:(r + 1) * D_TILE])))
        assert a == 127, (
            f"{name}: staged row {r} amax {a} != 127 — MODE_QUANT would "
            f"re-scale it and the staged codes would NOT be x_chunk. Run the "
            f"operands through stage_plan() first")
    Wp = []
    for bi, W in enumerate(Wblocks):
        W = np.asarray(W, dtype=np.int64)
        assert W.shape[0] == K, f"{name}: block {bi} K={W.shape[0]} != {K}"
        assert 1 <= W.shape[1] <= MXE_N, f"{name}: block {bi} N={W.shape[1]}"
        P = np.zeros((K, MXE_N), dtype=np.int64)
        P[:, :W.shape[1]] = W
        Wp.append(P)

    s = g3.Script(D_TILE)
    s.emit(f"// FAT PROJECTION PROGRAM {name}: ONE staging of {rows} "
           f"activation rows (k={K}), then {len(Wp)} OP_GEMM_WS descriptors "
           f"m=1 n={MXE_N} requant_en=0 — one per 8-column block — against "
           f"those SAME staged rows. The stage buffer is a pure data buffer "
           f"whose contents persist across jobs (apex_stage_buf.sv header), "
           f"and every WS descriptor clears the array at pass 0 "
           f"(mxe_ctrl.sv:286), so the blocks are independent.")
    g3.phase_a(s, tiers_used=(0,))
    g3.loader_phase(s)                       # loader token -> act bank0 row0

    s.emit(f"// stage {rows} activation rows ONCE through squant MODE_QUANT "
           f"(the identity: every row's amax is 127 -> emitted scale fp16 "
           f"1.0, a COUNTED check per row). LOAD's `sel` is the BASE ROW "
           f"(38ec95c) — row r of this staging IS stage-buffer row r.")
    s.route(rdst=1, asrc=1)                  # MXE res -> lane32 -> squant -> act
    for r in range(rows):
        rowv = x_chunk[r * D_TILE:(r + 1) * D_TILE].astype(np.float64)
        pairs = [g3.decompose_f16(int(b))[:2] for b in g3.to16(rowv)]
        g3.inject_jobs(s, pairs, 1)
        s.ess(F16_ONE, 1)                    # the identity scale
        base = r if load_base_row_override is None \
            else int(load_base_row_override)
        s.aj(0, ACT_BANK, 0, 1, s.BPR, base)  # LOAD -> act bank1 base row r

    s.emit(f"// {len(Wp)} column blocks, each re-EMITting the SAME {rows} "
           f"staged rows with its OWN {K} weight beats on the external xw "
           f"port. Ordering per block is build_script's, verbatim: first "
           f"EMIT, descriptor, remaining EMITs, weights last (INGEST "
           f"completes on activation beats alone, mxe_ctrl.sv:307-320).")
    s.route(rdst=0, wsrc=0, asrc=1)          # res -> RO; wgt <- external xw
    n_beats = 0
    for bi, P in enumerate(Wp):
        beats = g3.wgt_beats_ws(P, 0)
        assert len(beats) == K, f"{name}: block {bi}: {len(beats)} != K={K}"
        assert np.array_equal(_beats_to_block(beats, K), P), \
            f"{name}: block {bi}'s WB stream does not decode back to W"
        n_beats += len(beats)
        s.aj(1, ACT_BANK, 0, 1, s.BPR, 0)
        s.desc(g3.desc_words(g3.OP_GEMM_WS, 1, K, MXE_N))
        for r in range(1, rows):
            s.aj(1, ACT_BANK, 0, 1, s.BPR, r)
        s.wbeats(beats)
        s.ero([0] * MXE_N, 1)                # -> caps; placeholders, not values

    g3.final_phase(s, 8)                     # T<=8 tail: no frame stickies
    assert "ESTK ffff 0001" in s.lines, "expected the no-frame-error tail"
    return s, rows, n_beats


def build_gemm_multiblock_job(Wblocks, x_chunk, out_dir=None,
                              name="gemm_fat", *, burst: bool = True,
                              burst_depth: int = MB_FIFO_DEPTH,
                              allow_multirow: bool = True,
                              load_base_row_override=None,
                              allow_burst_overflow: bool = False):
    """Compile ONE K<=K_MAX chunk x MANY 8-column blocks -> (path, manifest).

    The captures come back as `len(Wblocks)` m=1 RO beats in block order;
    `decode_multiblock` splits them. Bit-for-bit the same accumulators as
    `build_gemm_job_full` per block — that equivalence is not asserted here,
    it is MEASURED by fatproof.py on the real twin.
    """
    s, rows, n_beats = build_multiblock_script(
        name, x_chunk, Wblocks, allow_multirow=allow_multirow,
        load_base_row_override=load_base_row_override)
    K = int(np.asarray(x_chunk).shape[0])
    nb = len(Wblocks)

    x = (BurstXlate(D_TILE, burst_depth, allow_burst_overflow) if burst
         else GemmXlate(D_TILE))
    x.note(f"FAT PROJECTION {name}: {nb} blocks x (m=1 k={K} n={MXE_N} WS, "
           f"requant_en=0) against ONE staging of {rows} act rows. The RO "
           f"lanes are `cap` egress — no accumulator, partial or total, "
           f"appears in this program."
           + (f" Stream pushes are BURSTED: one poll for FIFO-empty per "
              f"{burst_depth} pushes instead of a peek per push; a dropped "
              f"push would latch MB_STATUS, which the tail reads as a "
              f"counted check." if burst else ""))
    x.run(s.lines)
    _audit_no_result_expectations(x.ops)
    burst_audit = None
    if burst and not allow_burst_overflow:
        burst_audit = audit_push_bursts(x.ops, min(int(burst_depth),
                                                   MB_FIFO_DEPTH))
    want_caps = nb * (MXE_N + 1)
    if x.cap_seq != want_caps:
        raise AssertionError(
            f"{name}: {x.cap_seq} caps emitted, expected {want_caps} "
            f"({nb} blocks x (8 RO lanes + meta))")
    # the drop audit MUST be in the tail — it is the net under the burst
    if not [o for o in x.ops if o["op"] == "r"
            and o.get("a") == t2r.B_MB + t2r.MB_STATUS]:
        raise AssertionError(
            f"{name}: no MB_STATUS drop audit in the program — refusing to "
            f"emit bursted pushes without the check that catches a drop")

    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.regops.jsonl"
    with path.open("w") as f:
        for o in x.ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    cost = program_cost(x.ops)
    man = {"name": name, "kind": "fat-projection-gemm", "opcode": "OP_GEMM_WS",
           "m": 1, "k": K, "n": MXE_N, "requant_en": 0, "accumulate": False,
           "blocks": nb, "act_rows_staged": rows,
           "act_rows_chained_per_block": rows,
           "weight_beats": n_beats, "weight_bytes": n_beats * 8,
           "wgt_src": "external xw (mailbox WB)",
           "burst": bool(burst), "burst_depth": int(burst_depth) if burst else 0,
           "load_base_row_override": load_base_row_override,
           "burst_polls": getattr(x, "n_burst_polls", 0),
           "stream_pushes": getattr(x, "n_pushes", 0),
           "burst_audit": burst_audit,
           "regops": len(x.ops), "checks": x.n_checks, "caps": x.cap_seq,
           "cap_sites_rewritten": x.n_capsites,
           "taps_skipped": x.n_tap_skipped,
           "cost": cost,
           "acc_i32_expected_by_host_AFTER_the_run": None,
           "inputs_sha256": hashlib.sha256(
               np.ascontiguousarray(
                   np.asarray(x_chunk, dtype=np.int64).astype(np.int8)).tobytes()
               + b"".join(np.ascontiguousarray(
                   np.asarray(W, dtype=np.int64).astype(np.int8)).tobytes()
                   for W in Wblocks)).hexdigest(),
           "path": str(path)}
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return path, man


def decode_multiblock(captures, n_blocks: int) -> list:
    """The per-block raw INT32 accumulators of a fat program, in block order.

    REFUSES a short or ragged stream: `cd.ro_beats` is strict, and a beat
    count that is not exactly `n_blocks` means a descriptor did not drain —
    which must never be silently attributed to the wrong column block.
    """
    bs = cd.ro_beats(captures)
    if len(bs) != n_blocks:
        raise ValueError(f"decoded {len(bs)} RO beats, expected {n_blocks} "
                         f"(one m=1 beat per column block)")
    out = []
    for b in bs:
        lanes = b["lanes"]
        if any(v is None for v in lanes):
            raise ValueError(f"ragged RO beat at i0={b['i0']}: {lanes}")
        out.append(np.asarray(lanes, dtype=np.int64))
    return out


def build_gemm_job(W_chunk, x_chunk, out_dir=None, name="gemm_job") -> Path:
    """Compile ONE K<=2048 projection chunk -> the regops.jsonl PATH.

    W_chunk : int8 [K, N<=8]     x_chunk : int8 [K], K a multiple of 128 whose
    every 128-slice has amax exactly 127 (use stage_plan). The manifest lands
    beside the regops as <name>.manifest.json.
    """
    return build_gemm_job_full(W_chunk, x_chunk, out_dir, name)[0]


def build_gemm_job_full(W_chunk, x_chunk, out_dir=None, name="gemm_job"):
    """As build_gemm_job, returning (path, manifest)."""
    W_chunk = np.asarray(W_chunk, dtype=np.int64)
    x_chunk = np.asarray(x_chunk, dtype=np.int64)
    s, rows, n_beats = build_script(name, W_chunk, x_chunk)
    K = x_chunk.shape[0]

    x = GemmXlate(D_TILE)
    x.note(f"PROJECTION GEMM {name}: m=1 k={K} n={MXE_N} WS, requant_en=0, "
           f"weights on the external xw port, {rows} chained act stage rows. "
           f"The RO lanes are `cap` egress — the raw INT32 accumulators are "
           f"NOT in this program.")
    x.run(s.lines)
    _audit_no_result_expectations(x.ops)
    want_caps = MXE_N + 1                    # one m=1 result beat: 8 lanes+meta
    if x.cap_seq != want_caps:
        raise AssertionError(
            f"{name}: {x.cap_seq} caps emitted, expected {want_caps} "
            f"(8 RO lanes + the meta word of one m=1 result beat)")

    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.regops.jsonl"
    with path.open("w") as f:
        for o in x.ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    man = {"name": name, "kind": "projection-gemm", "opcode": "OP_GEMM_WS",
           "m": 1, "k": K, "n": MXE_N, "requant_en": 0, "accumulate": False,
           "act_rows_chained": rows, "weight_beats": n_beats,
           "weight_bytes": n_beats * 8, "wgt_src": "external xw (mailbox WB)",
           "regops": len(x.ops), "checks": x.n_checks, "caps": x.cap_seq,
           "cap_sites_rewritten": x.n_capsites,
           "taps_skipped": x.n_tap_skipped,
           "acc_i32_expected_by_host_AFTER_the_run": None,
           "inputs_sha256": hashlib.sha256(
               np.ascontiguousarray(x_chunk.astype(np.int8)).tobytes()
               + np.ascontiguousarray(W_chunk.astype(np.int8)).tobytes()
           ).hexdigest()}
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return path, man


# ══ 3. decode + host K-split accumulation ══════════════════════════════════
def decode_acc(captures) -> np.ndarray:
    """The 8 raw INT32 accumulators of one projection job, SIGN-EXTENDED."""
    return cd.ro_lanes_to_i32(captures, expect_n=MXE_N)


def accumulate_partials(partials) -> np.ndarray:
    """Sum INT32 partials per C-KSPLIT, asserting every PREFIX sits in INT32
    (compute.py:68-70 — retained accumulators live in INT32 registers)."""
    acc = np.zeros(MXE_N, dtype=np.int64)
    for p in partials:
        acc = acc + np.asarray(p, dtype=np.int64)
        assert np.all(np.abs(acc) < 2 ** 31), \
            "C-KSPLIT violated: INT32 prefix-sum overflow"
    return acc


def ksplit_matvec(W, x8, out_dir=None, name="ksplit", executor="sim",
                  binary=None, timeout_s=3600, rows_per_desc=ROWS_PER_DESC,
                  verbose=True) -> dict:
    """x8[K] x W[K, N<=8] -> raw INT32, K split into <=2048 tile jobs.

    Builds ceil(rows/16) SELF-CONTAINED regops jobs, runs each through
    tile_exec_bridge.run_job, decodes the RO captures, sums the partials per
    C-KSPLIT and grades against golden's own wide matmul. Golden is computed
    HERE, after the runs, from the same INT8 operands.
    """
    plan = stage_plan(x8, W)
    N = np.asarray(W).shape[1]
    res = {"K_real": int(np.asarray(x8).shape[0]), "N": int(N),
           "K_staged": plan.K_total, "rows": plan.rows, "kstar": plan.kstar,
           "inputs_sha256": plan.digest(), "jobs": [], "problems": []}
    partials, wall = [], []
    for ji, (r0, nr, K) in enumerate(plan.chunks(rows_per_desc)):
        a = r0 * D_TILE
        jn = f"{name}_k{ji}"
        t0 = time.time()
        path, man = build_gemm_job_full(plan.Wst[a:a + K],
                                       plan.xst[a:a + K], out_dir, jn)
        t_build = time.time() - t0
        if verbose:
            print(f"[build] {jn}: rows {r0}..{r0 + nr - 1} k={K} "
                  f"{man['regops']} regops {man['checks']} checks "
                  f"{man['caps']} caps {man['weight_beats']} weight beats "
                  f"({t_build:.1f}s) -> {path}")
        t0 = time.time()
        r = tile_exec_bridge.run_job(path, executor=executor, binary=binary,
                    timeout_s=timeout_s)
        t_run = time.time() - t0
        wall.append(t_run)
        line = [l for l in r["log"].splitlines() if "ops," in l or
                "RESULT" in l]
        if verbose:
            for l in line:
                print(f"[run  ] {l}")
            print(f"[run  ] {jn}: rc={r['rc']} ok={r['ok']} "
                  f"captures={len(r['captures'])} ({t_run:.1f}s wall)")
        for n in r["notes"]:
            res["problems"].append(f"{jn}: bridge note: {n}")
        if not r["ok"]:
            res["problems"].append(f"{jn}: executor not ok (rc={r['rc']})")
        if not r["captures"]:
            res["problems"].append(f"{jn}: no captures")
            res["ok"] = False
            return res
        acc = decode_acc(r["captures"])
        meta = [c for c in r["captures"] if c["sem"] == "ro_meta"]
        if len(meta) != 1 or (meta[0]["value"] & 1) != 1:
            res["problems"].append(
                f"{jn}: RO meta `last` not asserted on the single m=1 beat "
                f"({[m['value'] for m in meta]})")
        partials.append(acc)
        res["jobs"].append({"job": jn, "row0": r0, "rows": nr, "k": K,
                            "regops": man["regops"], "checks": man["checks"],
                            "weight_beats": man["weight_beats"],
                            "acc_i32": [int(v) for v in acc],
                            "wall_s": round(t_run, 2), "rc": r["rc"],
                            "ok": bool(r["ok"]), "log_tail": line[-2:]})

    acc = accumulate_partials(partials)
    # ── golden, computed AFTER the runs, from the same INT8 operands ────────
    Wi = np.asarray(W, dtype=np.int64)
    xi = np.asarray(x8, dtype=np.int64)
    gold_ks = cp.gemm_i8_ksplit(xi[None, :], Wi)[0].astype(np.int64)
    gold_wide = np.einsum("k,kn->n", xi, Wi)          # independent recompute
    assert np.array_equal(gold_ks, gold_wide), \
        "golden disagrees with itself — gemm_i8_ksplit vs a wide int64 matmul"
    res["partials"] = [[int(v) for v in p] for p in partials]
    res["acc_i32"] = [int(v) for v in acc]
    res["golden_i32"] = [int(v) for v in gold_wide]
    res["grade"] = cd.grade(acc[:N], gold_wide[:N])
    res["pad_lanes_zero"] = bool(np.all(acc[N:] == 0)) if N < MXE_N else True
    res["wall_s_per_job"] = [round(w, 2) for w in wall]
    res["ok"] = bool(res["grade"]["equal"] and res["pad_lanes_zero"]
                     and not res["problems"])
    return res


# ══ 4. the REAL case: Qwen2.5-7B layer-0 Wq projection ═════════════════════
def real_case(n_cols: int = MXE_N, col0: int = 0, layer: int = 0,
              rows: int | None = None) -> dict:
    """The layer-0 Wq projection of the committed S8 run's FIRST prompt token.

    PROVENANCE, stated exactly. The S8 trace .npz records hold per-HEAD
    attention tensors (q/K/V at D=128), NOT the 3584-wide hidden row a
    projection contracts over — so the activation is RE-DERIVED, not read
    back. It is re-derived with golden's OWN three lines, the ones
    transformer.decoder_layer_fx:481-489 runs:

        x8, _ = quant_rows_i8(X)                       # the fp16 embedding row
        h , _, _ = rmsnorm_fx_wide(x8[0], gamma1)
        h8, s_h  = quant_rows_i8(h / 256.0)

    with X = embed[prompt_ids[0]] — and for LAYER 0 at STEP 0 that embedding
    row IS the layer input, with no earlier layer to reproduce (Session.step:
    hidden = embed_row(tok) for the first layer of the first step). Token id
    and weight cache both come from the committed artifacts:
    docs/results/s8_7b_token/artifact_trace/run.json and build/s8_weights.
    """
    run = json.loads(S8_RUN.read_text())
    tok = int(run["prompt_ids"][0])
    emb = np.load(S8_WEIGHTS / "embed.npy", mmap_mode="r")
    X = np.asarray(emb[tok], dtype=np.float64)[None, :]      # fp16 grid
    gamma1 = [int(v) for v in np.load(S8_WEIGHTS / f"L{layer:02d}_gamma1.npy")]
    x8, _ = at.quant_rows_i8(X)
    h, _, _ = at.rmsnorm_fx_wide([int(v) for v in x8[0]], gamma1)
    h8m, s_h = at.quant_rows_i8(np.asarray(h, dtype=np.float64)[None, :] / 256.)
    h8 = h8m[0].astype(np.int64)

    wp = S8_WEIGHTS / f"L{layer:02d}_Wq.npy"
    Wfull = np.load(wp, mmap_mode="r")
    W = np.asarray(Wfull[:, col0:col0 + n_cols], dtype=np.int64)
    if rows is not None:                     # short-K debug slice
        # keep k* inside the slice so the F1 premise still holds
        kstar = int(np.argmax(np.abs(h8)))
        keep = min(rows * (D_TILE - 1), h8.shape[0])
        sel = np.array(sorted(set([kstar]) | set(range(keep - 1))),
                       dtype=np.int64)[:keep]
        h8, W = h8[sel], W[sel]
    return {"token": tok, "prompt": run.get("prompt"), "layer": layer,
            "tensor": "Wq", "col0": col0, "n_cols": n_cols,
            "x8": h8, "W": W, "s_h": int(s_h[0]),
            "D_model": int(h8.shape[0]),
            "weights": str(wp.relative_to(REPO)),
            "amax_code": int(np.max(np.abs(h8))),
            "kstar": int(np.argmax(np.abs(h8)))}


def smoke(out_dir=None, executor="sim", binary=None, n_cols=MXE_N, col0=0,
          rows=None, timeout_s=3600, rows_per_desc=ROWS_PER_DESC) -> int:
    c = real_case(n_cols, col0, rows=rows)
    x8, W = c["x8"], c["W"]
    print("=" * 78)
    print("FIRST NON-ATTENTION OP ON THE SILICON TWIN — a projection GEMM")
    print(f"  op      L{c['layer']:02d} {c['tensor']}[:, "
          f"{c['col0']}:{c['col0'] + c['n_cols']}] x h8  "
          f"(K={x8.shape[0]}, N={c['n_cols']})")
    print(f"  weights {c['weights']}  (real Qwen2.5-7B INT8)")
    print(f"  act     h8 of token {c['token']} = prompt_ids[0] of the "
          f"committed S8 run, prompt {c['prompt']!r}")
    print(f"  act     re-derived with golden's own quant/rmsnorm_fx_wide; "
          f"s_h={c['s_h']:#06x} amax code {c['amax_code']} at k*={c['kstar']}")
    print("=" * 78)

    t0 = time.time()
    r = ksplit_matvec(W, x8, out_dir=out_dir, name="qwen_l00_wq",
                      executor=executor, binary=binary, timeout_s=timeout_s,
                      rows_per_desc=rows_per_desc)
    t_all = time.time() - t0

    print("-" * 78)
    print(f"[plan ] K_real={r['K_real']} -> K_staged={r['K_staged']} in "
          f"{r['rows']} stage rows of 128 (lane 0 of every row = channel "
          f"k*={r['kstar']}, |code|=127, so MODE_QUANT is the identity; the "
          f"{r['K_staged'] - r['K_real']} duplicate/pad lanes carry ZERO "
          f"weight rows)")
    print(f"[plan ] C-KSPLIT: {len(r['jobs'])} tile jobs, "
          f"k = {[j['k'] for j in r['jobs']]} (K_MAX={K_MAX})")
    for j in r["jobs"]:
        print(f"[part ] {j['job']}: k={j['k']} rows {j['row0']}.."
              f"{j['row0'] + j['rows'] - 1}  INT32 partial {j['acc_i32']}")
    print(f"[sum  ] host-accumulated INT32 : {r['acc_i32']}")
    print(f"[gold ] golden wide matmul     : {r['golden_i32']}")
    print(f"[grade] {r['grade']}")
    if 'pad_lanes_zero' in r:
        print(f"[grade] zero-padded output lanes read 0: {r['pad_lanes_zero']}")
    print(f"[time ] per-job wall clock {r['wall_s_per_job']} s "
          f"(total {t_all:.1f}s incl. build+golden)")
    for p in r["problems"]:
        print(f"[grade] PROBLEM: {p}")

    # ── DISCRIMINATION: a grader that cannot go red proves nothing ─────────
    disc = {}
    bad = np.asarray(r["acc_i32"], dtype=np.int64).copy()
    bad[0] += 1
    disc["a_one_count"] = not cd.grade(
        bad[:c["n_cols"]],
        np.asarray(r["golden_i32"], dtype=np.int64)[:c["n_cols"]])["equal"]
    Wb = W.copy()
    Wb[0, 0] = Wb[0, 0] + 1 if Wb[0, 0] < 127 else Wb[0, 0] - 1
    gold_b = np.einsum("k,kn->n", x8, Wb)
    disc["b_wrong_golden_one_weight"] = not cd.grade(
        np.asarray(r["acc_i32"], dtype=np.int64)[:c["n_cols"]],
        gold_b[:c["n_cols"]])["equal"]
    print(f"[disc ] (a) +1 on one accumulated lane -> RED: "
          f"{disc['a_one_count']}")
    print(f"[disc ] (b) correct captures vs golden with ONE weight byte "
          f"changed -> RED: {disc['b_wrong_golden_one_weight']}")
    discriminates = all(disc.values())
    if not discriminates:
        print("[disc ] PROBLEM: the grader did not go red on a known-bad "
              "input — the result above is not evidence")

    verdict = "PASS" if (r["ok"] and discriminates) else "FAIL"
    print("-" * 78)
    print(f"PROJECTION GEMM ON THE TWIN: bit_exact={r['grade']['equal']} "
          f"({r['K_real']} INT8 channels contracted, "
          f"{sum(j['weight_beats'] for j in r['jobs'])} weight beats pushed "
          f"on the external xw port, "
          f"{sum(j['checks'] for j in r['jobs'])} structural checks, "
          f"0 result expectations)")
    print(f"GEMM_JOB SMOKE: {verdict}")
    return 0 if verdict == "PASS" else 1


# ══ 5. host-only selftest ══════════════════════════════════════════════════
def _selftest() -> int:
    import tempfile
    fails = 0

    def chk(n, cond, extra=""):
        nonlocal fails
        print(f"  [{n}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    rng = np.random.default_rng(20260729)
    tmp = Path(tempfile.mkdtemp(prefix="apex_gemm_"))

    # 1-4: the staging plan on a synthetic 3584-wide operand pair
    Kt, N = 3584, 8
    x8 = rng.integers(-100, 101, Kt, dtype=np.int64)
    x8[1234] = 127                                     # the amax anchor
    W = rng.integers(-8, 8, (Kt, N), dtype=np.int64)
    p = stage_plan(x8, W)
    chk("1 rows", p.rows == 29 and p.K_total == 29 * 128,
        f"{p.rows} rows, K_staged={p.K_total}, k*={p.kstar}")
    chk("2 identity", np.array_equal(np.einsum("k,kn->n", p.xst, p.Wst),
                                     np.einsum("k,kn->n", x8, W)))
    chk("3 amax", all(int(np.max(np.abs(p.xst[a:a + 128]))) == 127
                      for a in range(0, p.K_total, 128)))
    chk("4 coverage", sorted(int(v) for v in p.chan[p.use]) == list(range(Kt)),
        f"{int(p.use.sum())} used lanes, "
        f"{p.K_total - int(p.use.sum())} zero-weight lanes")

    # 5: the F1 premise must be ENFORCED, not assumed
    try:
        stage_plan(np.clip(x8, -100, 100), W)
        chk("5 F1 enforced", False, "an amax != 127 activation was accepted")
    except AssertionError as e:
        chk("5 F1 enforced", "F1" in str(e), f"{str(e)[:56]}…")

    # 6-10: one compiled chunk
    r0, nr, K = p.chunks()[0]
    a = r0 * 128
    path, man = build_gemm_job_full(p.Wst[a:a + K], p.xst[a:a + K], tmp, "st")
    ops = [json.loads(l) for l in path.read_text().splitlines()]
    chk("6 compiled", man["regops"] == len(ops) and man["k"] == K_MAX,
        f"{man['regops']} regops, k={man['k']}, {man['checks']} checks")
    chk("7 caps", man["caps"] == 9 and all("e" not in o for o in ops
                                           if o["op"] == "cap"),
        f"{man['caps']} caps (8 lanes + meta), none carries an expectation")
    chk("8 no result expectation",
        not [o for o in ops if o["op"] in ("r", "rn", "poll")
             and o.get("a") in RESULT_ADDRS])
    descs = [o for o in ops if o["op"] == "w" and o["a"] == t2r.B_MB + 0x88]
    chk("9 requant off in every descriptor push",
        all((o["d"] >> 1) & 1 == 0 for o in descs), f"{len(descs)} DS_W[2]")
    n_wb = sum(1 for o in ops if o["op"] == "pw"
               and o["a"] == t2r.B_MB + t2r.XWP)
    chk("10 weight beats", man["weight_beats"] == K
        and n_wb == K + 8 * (K // 128) * 16,
        f"{man['weight_beats']} projection beats; {n_wb} XWP pushes total "
        f"(projection + the {K // 128}x16 K=2 staging jobs' 8 beats each)")

    # 11: the act EMIT chain must deliver exactly mxe_ctrl's total_a
    emits = [o for o in ops if o["op"] == "jf"
             and o["a"] == t2r.B_MB + t2r.JOB["AJ"]]
    nbw = (D_TILE // 8 - 1).bit_length() + 1
    chained = [o for o in emits
               if (o["d"] & 1) == 1 and ((o["d"] >> 1) & 1) == ACT_BANK
               and ((o["d"] >> 4) & 0x1F) == 1
               and ((o["d"] >> 9) & ((1 << nbw) - 1)) == BPR]
    chk("11 INGEST beat count", len(chained) * BPR == K // 8,
        f"{len(chained)} chained EMIT jobs x {BPR} beats = {len(chained) * BPR}"
        f" act beats, mxe total_a = M*ceil(K/8) = {K // 8}")

    # 12: the beat encoding round-trips to the real weight rows
    chk("12 beat round-trip",
        np.array_equal(_beats_to_block(g3.wgt_beats_ws(p.Wst[a:a + K], 0), K),
                       p.Wst[a:a + K]))

    # 13: C-KSPLIT accumulation is the wide matmul, on the plan's own chunks
    parts = [np.einsum("k,kn->n", p.xst[c[0] * 128:c[0] * 128 + c[2]],
                       p.Wst[c[0] * 128:c[0] * 128 + c[2]])
             for c in p.chunks()]
    chk("13 C-KSPLIT sum", np.array_equal(accumulate_partials(parts),
                                          np.einsum("k,kn->n", x8, W)),
        f"{len(parts)} partials")

    # 14: the REAL operands load and satisfy F1 (no simulator involved)
    try:
        c = real_case()
        rp = stage_plan(c["x8"], c["W"])
        chk("14 real case", c["D_model"] == 3584 and rp.rows == 29
            and c["amax_code"] == 127,
            f"token {c['token']} L00 Wq, D_model={c['D_model']}, "
            f"k*={rp.kstar}, {rp.rows} rows")
    except FileNotFoundError as e:
        chk("14 real case", False, f"weights missing: {e}")

    # ── 15-19: the ON-TILE K-split chain (OP_GEMM_OS accumulate=1) ──────────
    # A pure-host proof of the two invariants the whole zero-host-add claim
    # rests on: the accumulate flags are right, and NOTHING that clears the
    # array runs once the chain has started.
    cpath, cman = build_chain_job(W, x8, tmp, "chain", rows_per_desc=1)
    cops = [json.loads(l) for l in cpath.read_text().splitlines()]
    ds = decode_desc_pushes(cops)
    os_ds = [d for d in ds if d["opcode"] == 2]
    ws_ds = [d for d in ds if d["opcode"] == 1]
    chk("15 chain descriptors", len(os_ds) == cman["n_desc"] == p.rows
        and all(d["k"] == D_TILE for d in os_ds),
        f"{len(os_ds)} OP_GEMM_OS descriptors, k={os_ds[0]['k']} each")
    chk("16 accumulate flags", os_ds[0]["acc"] == 0
        and all(d["acc"] == 1 for d in os_ds[1:])
        and all(d["os"] == 1 for d in os_ds),
        "first OVERWRITES (clr=1), every later one RETAINS "
        "(mxe_ctrl.sv:286)")
    # the killer: a WS job between two chained OS descriptors would clear the
    # array (WS always clears), silently zeroing the running total.
    first_os = min(d["i"] for d in os_ds)
    late_ws = [d for d in ws_ds if d["i"] > first_os]
    chk("17 no clearing job inside the chain", not late_ws,
        f"{len(ws_ds)} staging WS jobs, all before the chain "
        f"(last at op {max(d['i'] for d in ws_ds)} < {first_os})")
    chk("18 chain caps", cman["caps"] == cman["n_desc"] * 9
        and all("e" not in o for o in cops if o["op"] == "cap")
        and not [o for o in cops if o["op"] in ("r", "rn", "poll")
                 and o.get("a") in RESULT_ADDRS],
        f"{cman['caps']} caps = {cman['n_desc']} beats x 9, none carries an "
        f"expectation")
    # 19: the decode takes the LAST beat verbatim — it never adds
    synth = []
    pref = np.cumsum([int(np.einsum("k,kn->n", p.xst[r * 128:(r + 1) * 128],
                                    p.Wst[r * 128:(r + 1) * 128])[0])
                      for r in range(p.rows)])
    for b, v in enumerate(pref):
        for j in range(MXE_N):
            synth.append({"tag": f"ro_w{j}_{b}", "sem": f"ro_w{j}",
                          "i": len(synth), "addr": cd.RO_W0 + 4 * j,
                          "mask": 0xFFFFFFFF,
                          "value": int(v if j == 0 else 0) & 0xFFFFFFFF})
        synth.append({"tag": f"ro_meta_{b}", "sem": "ro_meta", "i": len(synth),
                      "addr": cd.RO_META, "mask": 0x1, "value": 1})
    ct = chain_totals(synth, n_desc=p.rows)
    chk("19 last beat IS the total", ct["total"][0] == int(pref[-1])
        == int(np.einsum("k,kn->n", x8, W)[0])
        and ct["raw_lane0_u32"] == (int(pref[-1]) & 0xFFFFFFFF)
        and ct["beats"][0][0] == int(pref[0]),
        f"tile total {ct['total'][0]} == wide matmul lane 0; raw u32"
        f" {ct['raw_lane0_u32']} is those bits untouched (unsigned view — a "
        f"NEGATIVE accumulator would be >= 2^31 and RMS_SUM2's [27:0] arm "
        f"would refuse it loudly); no host addition applied")

    # ── 20-27: FAT programs (stage once, sweep N blocks) + push bursts ──────
    # Structural only — no executor. The RUN-level fat-vs-thin equivalence
    # and the two red arms are fatproof.py's job, on the real twin.
    NB = 5
    blocks = [p.restage(rng.integers(-9, 9, (Kt, MXE_N), dtype=np.int64))
              for _ in range(NB)]
    r0, nr, K = p.chunks()[0]
    off = r0 * D_TILE
    fpath, fman = build_gemm_multiblock_job(
        [b[off:off + K] for b in blocks], p.xst[off:off + K], tmp, "fat")
    fops = [json.loads(l) for l in fpath.read_text().splitlines()]
    fds = decode_desc_pushes(fops)
    proj = [d for d in fds if d["opcode"] == 1 and d["k"] == K]
    chk("20 fat descriptors", len(proj) == NB and fman["blocks"] == NB
        and all(d["n"] == MXE_N and d["m"] == 1 and d["rq_en"] == 0
                for d in proj),
        f"{len(proj)} OP_GEMM_WS k={K} n={MXE_N} descriptors in ONE program")
    def _load_jobs(ops):
        """AJ jobs with op=LOAD — i.e. one per staged activation row."""
        return [o for o in ops if o["op"] == "jf"
                and o["a"] == t2r.B_MB + t2r.JOB["AJ"] and (o["d"] & 1) == 0]
    # the economics, MEASURED against the thin path on the SAME operands
    tcost = {"ops": 0, "peeks": 0, "bytes": 0}
    t_loads = 0
    for bi, b in enumerate(blocks):
        tp, _ = build_gemm_job_full(b[off:off + K], p.xst[off:off + K], tmp,
                                    f"thin{bi}")
        c = program_cost(tp)
        t_loads += len(_load_jobs(
            [json.loads(l) for l in tp.read_text().splitlines()]))
        for k in tcost:
            tcost[k] += c[k]
    fcost = program_cost(fpath)
    f_loads = len(_load_jobs(fops))
    chk("21 the activation is staged ONCE for all blocks",
        f_loads * NB == t_loads and f_loads == K // D_TILE + 1,
        f"{f_loads} LOAD jobs in the fat program ({K // D_TILE} staged rows "
        f"+ the loader token) vs {t_loads} across the {NB} thin ones")
    chk("22 fat caps == 9 per block, none carrying an expectation",
        fman["caps"] == NB * 9
        and all("e" not in o for o in fops if o["op"] == "cap")
        and not [o for o in fops if o["op"] in ("r", "rn", "poll")
                 and o.get("a") in RESULT_ADDRS],
        f"{fman['caps']} caps")
    chk("23 fat costs strictly fewer BAR0 ops AND fewer PEEKS than the same "
        "blocks thin", fcost["ops"] < tcost["ops"]
        and fcost["peeks"] < tcost["peeks"],
        f"ops {tcost['ops']:,} -> {fcost['ops']:,} "
        f"({tcost['ops'] / fcost['ops']:.2f}x), peeks {tcost['peeks']:,} -> "
        f"{fcost['peeks']:,} ({tcost['peeks'] / fcost['peeks']:.1f}x), "
        f"files {NB} -> 1")
    chk("24 ...and pushes EXACTLY the same weight beats (the tile sees the "
        "same stream)",
        fman["weight_beats"] == NB * K,
        f"{fman['weight_beats']} = {NB} x {K}")
    chk("25 the burst can never overflow a mailbox FIFO (compile-time walk)",
        fman["burst_audit"]["blind_pushes"] > 0
        and fman["burst_audit"]["empty_polls"] * MB_FIFO_DEPTH
        >= fman["burst_audit"]["blind_pushes"],
        str(fman["burst_audit"]))
    bad = list(fops)
    ins = next(i for i, o in enumerate(bad)
               if o["op"] == "w" and o.get("a") == t2r.B_MB + t2r.XWP)
    bad.insert(ins, dict(bad[ins]))          # one push too many in a burst
    try:
        audit_push_bursts(bad)
        chk("26 ...and that walk BITES on one extra push", False, "no refusal")
    except AssertionError as e:
        chk("26 ...and that walk BITES on one extra push", "burst longer" in
            str(e) or "blind push" in str(e), f"{str(e)[:64]}…")
    try:
        build_multiblock_script("prefix", p.xst[off:off + K],
                                [b[off:off + K] for b in blocks],
                                allow_multirow=False)
        chk("27 a PRE-38ec95c image is REFUSED multi-row staging", False,
            "no refusal")
    except AssertionError as e:
        chk("27 a PRE-38ec95c image is REFUSED multi-row staging",
            "BASE-ROW" in str(e), f"{str(e)[:60]}…")

    print(f"GEMM_JOB SELFTEST: {'FAIL' if fails else 'PASS'} (fails={fails})")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--n-cols", type=int, default=MXE_N)
    ap.add_argument("--col0", type=int, default=0)
    ap.add_argument("--rows", type=int, default=None,
                    help="short-K debug slice: stage only this many rows")
    ap.add_argument("--rows-per-desc", type=int, default=ROWS_PER_DESC)
    ap.add_argument("--timeout-s", type=int, default=3600)
    a = ap.parse_args()
    import remote_hw_exec                      # hw routing (no-op unless
    remote_hw_exec.attach(tile_exec_bridge, a) # $APEX_F2_HOST is set)
    if a.selftest:
        return _selftest()
    if a.build_only:
        c = real_case(a.n_cols, a.col0, rows=a.rows)
        p = stage_plan(c["x8"], c["W"])
        for ji, (r0, nr, K) in enumerate(p.chunks(a.rows_per_desc)):
            o = r0 * D_TILE
            path, man = build_gemm_job_full(p.Wst[o:o + K], p.xst[o:o + K],
                                            a.out_dir, f"qwen_l00_wq_k{ji}")
            print(json.dumps(man, indent=1))
            print(f"-> {path}")
        return 0
    if a.smoke:
        return smoke(a.out_dir, a.executor, a.binary, a.n_cols, a.col0,
                     a.rows, a.timeout_s, a.rows_per_desc)
    ap.error("give --smoke, --build-only or --selftest")


if __name__ == "__main__":
    sys.exit(main())
