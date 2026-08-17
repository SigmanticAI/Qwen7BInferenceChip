#!/usr/bin/env python3
# layer_offload.py — MILESTONE C2: EVERY OP TYPE OF ONE DECODER LAYER, SERVED
# BY THE TILE, INSIDE A REAL PROMPT RUN.
#
#   python3 scripts/fpga/f2/layer_offload.py --prompt "The capital of France is" \
#       --layer 0 --max-tokens 1 --executor sim [--offload-step S] [--dry-run]
#
# ══ WHAT THIS IS ═══════════════════════════════════════════════════════════
# prompt_offload.py took ONE operation (attention_core, one head) away from the
# golden Qwen2.5-7B decode and let the tile compute it, then proved the emitted
# token was unchanged. Since then the tile has grown five more op families
# (docs/results/prompt_on_chip/SIX_OF_SIX_RESULT.md). This file takes ALL SIX
# away — for ONE chosen layer of the SAME real prompt run:
#
#     projections   gemm_i8_ksplit          -> gemm_job.ksplit_matvec jobs
#     RoPE          rope_fx (the q row)     -> gen_layer_ops.build_rope_stage
#     attention     attention_core          -> compute_job.build_compute_job
#     residual      _f16 at the two residual call sites
#                                           -> gen_layer_ops.build_resid_stage
#     RMSNorm-2     rmsnorm_fx_wide(gamma2) -> gen_layer_ops.build_norm2_chunked
#     SwiGLU        _f16 at the swiglu site -> gen_layer_ops.build_swiglu_stage
#
# and then asks the only question that matters: does the model still emit the
# SAME TOKEN as a pure-host run?
#
# ══ THE SEAM: REBINDING, NEVER EDITING ═════════════════════════════════════
# golden/ is the arbiter and is never modified. Every op above is reached by
# REBINDING a MODULE GLOBAL that decoder_layer_fx resolves at call time
# (prompt_offload.py's established technique, one lane wider):
#
#   tf.decoder_layer_fx   bookkeeping wrapper: (step, layer) + arm/disarm
#   tf.gemm_i8_ksplit     all 7 projection GEMMs (q,k,v,o,gate,up,down)
#   tf.rope_fx            the per-head RoPE
#   tf.attention_core / at.attention_core     the per-head attention
#   tf.rmsnorm_fx_wide    RMSNorm-1 and RMSNorm-2 (told apart by gamma)
#   tf._proj_epilogue     NOT substituted — read-only, to learn the o8 code
#                         stream + graded composite the residual job needs
#   tf._f16               the three narrowing sites inside decoder_layer_fx:
#                         residual-1, the SwiGLU product, residual-2
#
# `_f16` is used by golden in several places, so the wrapper identifies the
# CALL SITE by the caller's line number, and those line numbers are located at
# __enter__ by SEARCHING transformer.py's own source for the three statements
# (`_locate_f16_sites`). If golden moves or renames them the harness REFUSES
# rather than silently offloading the wrong narrowing. Every interception is
# then re-verified against operands this file tracked independently (the
# residual sum, the SwiGLU product), so a mis-identified site cannot pass.
#
# ══ WHY THE OFFLOADED LAYER RUNS IN C-LBUS BUS_ON ══════════════════════════
# apex_layer_deq.sv:90-92 REFUSES a job composite whose fp32 mantissa carries
# bits below fp16 grade (`c_m[12:0] == 13'h0`), and asu_swiglu applies the same
# rule to its gate/up composites. golden's default composition (BUS_OFF) leaves
# the epilogue scales as arbitrary float64, which the tile is entitled to
# refuse — so the residual and SwiGLU ops are only REACHABLE when the layer
# composes with fp16-graded composites. That is exactly transformer.BUS_ON
# (C-LBUS, D-030), the same arbiter capture_layer_step / gen_l4_vectors /
# gen_layer_trace already use. The target layer therefore runs with bus=BUS_ON.
#
# This is a REAL change to that layer's host arithmetic, so it is measured, not
# waved past: run 3 of the A/B is host-only WITH BUS_ON, which separates "the
# tile changed the token" from "the bus mode changed the token".
#
# ══ WHAT THE TILE'S EGRESS CAN AND CANNOT CARRY ════════════════════════════
# Four of the six ops leave the tile EXACTLY:
#   projections  raw INT32 accumulators on the RO lanes
#   attention    raw INT32 P.V accumulators + the captured s_c
#   residual     the updated row's fp16 words on LAYER_RDATA
#   RoPE         the rotated row's C-1 view (INT8 codes + fp16 row scale).
#                head_dim == 128 == the C-1 frame, and golden's very next step
#                (attention_core Q7) re-quantizes the row with the SAME
#                quant_rows_i8 — which is IDEMPOTENT on a code x scale
#                reconstruction. That idempotence is CHECKED at run time
#                (`rope_requant_identical`), not assumed, so the substituted
#                row is exact where the model actually consumes it.
# Two cannot, and this file says so instead of hiding it:
#   RMSNorm-2 and SwiGLU re-enter the host through the C-1 feeder, whose row
#   length is frozen at 128 (B-FEED-WIDTH, seam_feeder_quant.sv:67/100). The
#   tile's egress is therefore 28 (resp. 148) independently-scaled INT8 rows,
#   while golden's next stage wants ONE whole-row C-1. The substituted value is
#   the tile's reconstruction, and the ledger prints the measured delta from
#   golden's own value AND the effect on the code stream the model consumes.
#
# ══ SCOPE IS A PARAMETER (and every ledger line is measured) ═══════════════
# A full-width 7B layer is ~300k tile GEMM jobs; at the measured 0.38 s/job in
# simulation that is days. So the projection width (--proj-cols/--proj-rows) is
# a knob, and the ledger reports served/total for EVERY op — a partial but
# honest coverage number, never a rounded-up claim. The five non-GEMM ops run
# at FULL 7B width by default (28/28 heads, 3584/3584 residual elements,
# 3584/3584 norm codes, 18944/18944 SwiGLU columns).
#
# ══ TRANSPORT: ONE INVOCATION PER PASS, NOT ONE PER OP ═════════════════════
# Measured on silicon (docs/results/prompt_on_chip/fat_hw_prediction.json): a
# job costs ~3.8 s wall of which ~0.78 ms is MMIO, so ~4.8 s of every executor
# ENTRY is ssh + remote python + SDK attach. A 0.5B layer-step served one op
# at a time is 37 entries — and 30 of them are single-program calls for the
# per-head RoPE, the per-head attention and the per-slice residual.
#
# Those cannot simply be gathered: golden calls rope and attention inside ONE
# per-head loop and consumes each value before the next operand exists
# (transformer.py:535-553). So the layer is REPLAYED instead. `Runner.acquire`
# answers one question — "is this exact program's capture set already in
# hand?" — and on a miss QUEUES the program and leaves that op on the host for
# the pass. `_layer` replays until a pass serves everything it asks for; each
# replay costs ONE flush, and a flush is one executor invocation per
# --max-batch-mb of regops.
#
# The pool is keyed by `program_key` = the program's NAME and its exact BYTES,
# so reusing an entry asserts only that the executor ran a byte-identical
# program under the same per-file TILE_RST discipline. Every decode, grade,
# substitution and consumption check is re-run from those captures on the pass
# that is kept — nothing is remembered except the tile's own captured words,
# and a pass that could not serve everything is DISCARDED rather than reported
# as partial coverage. `--no-collapse` restores one-invocation-per-op so the
# before/after is measured on the same job set by the same code.
#
# ══ HONESTY OF EACH MODE ═══════════════════════════════════════════════════
#   --dry-run      every tile call is replaced by golden's own value: proves
#                  the whole seam (rebind, site identification, substitution,
#                  consumption, A/B, poison) with no executor. Provenance
#                  reads `GOLDEN (dry-run)`.
#   --executor sim the verilated cl_apex computes the values (this file's
#                  proof mode; `--make-tiny-weights` gives a tile-legal random
#                  model so the WHOLE six-op path runs end to end in minutes).
#   --executor hw  the same, on an F2 card (NOT run here).
#   --poison K     re-runs with every tile-served value scaled by K. If the
#                  logits do not move, the substitutions were never
#                  load-bearing and every PASS above is vacuous.
#
# CLI: --selftest  tiny random model, no executor — exercises the rebinds,
#                  the three `_f16` sites, substitution, consumption, token
#                  identity and the poison discriminator in seconds.

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tinynpu as rt                                        # noqa: E402
from apex_golden import attention as at                         # noqa: E402
from apex_golden import compute as cp                           # noqa: E402
from apex_golden import transformer as tf                       # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits, rne  # noqa: E402

import batch_exec as bx                                         # noqa: E402
import cap_decode as cd                                         # noqa: E402
import compute_job as cj                                        # noqa: E402
import gemm_job as gj                                           # noqa: E402
import gen_layer_ops as gl                                      # noqa: E402
import prompt_offload as po                                     # noqa: E402
import tile_exec_bridge as bridge                               # noqa: E402

TOKENS_MAX = at.CHUNK_T_MAX          # 128 — the T_ROW_MAX per-job envelope
D_TILE = gj.D_TILE                   # 128 — feeder / stage / MXE row width
MXE_N = gj.MXE_N                     # 8
DEFAULT_WORK = REPO / "build" / "layer_offload"
OP_TYPES = ("proj", "rope", "attn", "resid", "norm", "swiglu")

# the three `_f16` call sites inside decoder_layer_fx, by their source text
F16_SITES = {
    "r1":     "r.r1 = _f16(",
    "swiglu": "r.swiglu = _f16(",
    "r2":     "r.r2 = _f16(",
}


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def _f32_bits(v: float) -> int:
    """The fp32 bit pattern of a float64 — the LAYER_JOBC composite word."""
    return int(np.float64(v).astype(np.float32).view(np.uint32))


def _is_graded(bits: int) -> bool:
    """apex_layer_deq.sv:90-92 comp_legal: positive, normal, fp16-grade."""
    return (bits >> 31) == 0 and 0 < ((bits >> 23) & 0xFF) < 255 \
        and (bits & 0x1FFF) == 0


def _exact_f16(x: np.ndarray) -> bool:
    """Is every element already ON the fp16 grid (so f64->f16 loses nothing)?"""
    x = np.asarray(x, dtype=np.float64)
    return bool(np.array_equal(
        f16_bits_to_f64(f64_to_f16_bits(x)).reshape(x.shape), x))


def _locate_f16_sites() -> dict:
    """Absolute transformer.py line numbers of the three narrowing sites.

    Found by SEARCHING golden's own source, never hardcoded: if a statement
    moved, was renamed, or appears twice, this raises instead of letting the
    wrapper offload some other narrowing.
    """
    src, first = inspect.getsourcelines(tf.decoder_layer_fx)
    out = {}
    for tag, needle in F16_SITES.items():
        hits = [first + i for i, ln in enumerate(src) if needle in ln]
        if len(hits) != 1:
            raise SystemExit(
                f"REFUSE: {len(hits)} occurrence(s) of {needle!r} in "
                f"{tf.__file__}:decoder_layer_fx — the `_f16` call-site map "
                f"this harness rebinds against is no longer valid. Golden is "
                f"the arbiter and is never edited, so fix THIS file's site "
                f"table (F16_SITES) after re-reading golden.")
        out[tag] = hits[0]
    return out


# ══════════════════════════ scope + provenance ═════════════════════════════

@dataclass
class Scope:
    """How much of each op the tile serves. Every field lands in the ledger."""
    ops: tuple = OP_TYPES
    proj_cols: int = MXE_N          # output columns per projection call
    proj_rows: int = 1              # activation rows per projection call
    heads: int = -1                 # attention heads (-1 = all)
    rope_heads: int = -1            # q-row rotations (-1 = all)
    resid_cols: int = -1            # residual elements (-1 = the full row)
    resid_slice: int = D_TILE       # per-job window (the narrow image's DM_MAX)
    norm_dm: int = -1               # RMSNorm-2 elements (-1 = the full row)
    norm1_rows: int = 0             # RMSNorm-1 rows (default: host)
    swiglu_chunks: int = -1         # 64-column SwiGLU jobs (-1 = the full d_ffn)

    def on(self, op: str) -> bool:
        return op in self.ops


@dataclass
class OpRec:
    """One tile-served (or deliberately host-left) op instance."""
    op: str
    name: str
    source: str
    n_served: int = 0
    n_total: int = 0
    jobs: int = 0
    caps: int = 0
    seconds: float = 0.0
    exact: bool | None = None       # is the SUBSTITUTED value == golden's?
    grade_ok: bool | None = None    # is the tile's egress == golden's view?
    detail: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ══════════════════════════ the executor runner ════════════════════════════

# The largest regops payload ALREADY carried by ONE invocation of the flown
# shape (the Wg projection call: 5 fat programs, 60.1 MB — build/p05b_fat2.log
# and build/p05b_fat_hw2.log, 2026-08-04). Collapsing invocations must not
# quietly turn into "one 214 MB upload"; narrow_flight.py learned that lesson
# the expensive way (its --max-batch-mb note). Bounding a batch at the biggest
# payload that has actually flown keeps the fitted per-invocation constant
# applicable instead of extrapolated.
MAX_BATCH_MB_DEFAULT = 64
# The longest chain a replay has to walk is the layer's own dependency depth
# whenever a family's substituted value is NOT golden's: rope's C-1
# reconstruction, and (in a --poison arm) every poisoned family. Measured
# depths: 3 passes for the four exact-egress families, 5 for a poisoned arm.
# The ceiling is a REFUSAL, not a budget — it exists so a set that never
# stabilises stops the run instead of looping.
MAX_PASSES_DEFAULT = 12


def program_key(path) -> str:
    """The identity of an emitted program: its NAME and its exact BYTES.

    This is the key the capture pool is addressed by, so "a pooled capture set
    may be reused" means precisely: *the program this pass emitted is
    byte-identical to the program the executor ran*. The name is folded in on
    purpose — two ops that happened to emit identical bytes under different
    names stay distinct keys, which keeps "programs audited == programs
    executed" an equality a reader does not have to reason around.
    """
    h = hashlib.sha256()
    h.update(Path(path).name.encode())
    h.update(b"\0")
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def operand_key(*parts) -> str:
    """A digest over the EXACT operand bytes an emitter is a pure function of.

    Used only where re-emitting to compare program bytes would cost more than
    the transport it saves (the full-width projections: ~28 s of staging and
    214 MB of regops per layer). Every value the emitter reads goes in —
    operand bytes, shapes, dtypes, and every transport/geometry knob — and the
    values a hit returns are RE-GRADED against golden on the spot, so a stale
    or mismatched entry cannot survive undetected.
    """
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.dtype).encode())
            h.update(str(p.shape).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b"\0")
    return h.hexdigest()


class Runner:
    """Builds nothing; runs regops and hands back per-file captures.

    ══ THE TRANSPORT PROBLEM THIS CLASS OWNS ══════════════════════════════
    On silicon a job costs ~3.8 s wall of which ~0.78 ms is real MMIO — the
    rest is per-INVOCATION overhead (ssh, remote python start, SDK attach).
    The fitted model (docs/results/prompt_on_chip/fat_hw_prediction.json)
    puts that constant at 4.8 s. A 0.5B layer-step served one op at a time
    is 37 invocations — 7 batched projection calls plus ONE INVOCATION EACH
    for 14 RoPE rows, 14 attention heads and 2 residual windows — so ~178 s
    of the measured 193.7 s/layer is transport, not tile.

    Those 30 single-program invocations cannot simply be gathered: golden
    calls rope/attention per head INSIDE one loop and consumes each value
    before emitting the next operand (transformer.py:535-553), so at the
    first head there is nothing to gather yet.

    ══ THE COLLAPSE: a CAPTURE POOL plus a fixed point ════════════════════
    `acquire()` is `run()` with a memory. It emits nothing and decides
    nothing; it answers ONE question — "is this exact program's capture set
    already in hand?" — and on a miss it QUEUES the program and says so, so
    the op wrapper leaves that value on the host for this pass. The layer is
    then replayed (LayerOffloader._layer) until a whole pass answers "yes"
    everywhere; each replay costs one `flush()`, and a flush is ONE executor
    invocation per <= --max-batch-mb of regops.

    For the four exact-egress families the fixed point is reached in three
    passes: pass 1 discovers every program (the golden fallback trajectory is
    the tile trajectory, because those families are bit-exact), pass 2
    re-discovers only the attention programs (their q row is now the tile's
    C-1 reconstruction, which is a DIFFERENT operand and therefore honestly a
    different program), pass 3 is served entirely from the pool. Two
    invocations per layer instead of 37.

    ══ WHY A POOLED CAPTURE IS THE SAME EVIDENCE ══════════════════════════
    The pool is keyed by `program_key` — the program's name and its exact
    bytes. Reusing an entry therefore asserts nothing beyond "the executor
    ran a byte-identical program, in this session, on this image, under the
    same per-file TILE_RST discipline both executors apply before every file
    (f2_host_run.py:146; sim_main.cpp reset-per-file)". The decode, the
    grading, the substitution and every consumption check run again from
    those captures on EVERY pass — nothing is remembered except the tile's
    own captured words.

    `collapse=False` restores the pre-collapse behaviour exactly (one
    invocation per `acquire`), which is how the before/after in the ledger is
    measured on the same job set by the same code.
    """

    def __init__(self, mode: str, work: Path, *, binary=None,
                 tile_div: int = bridge.TILE_DIV_DEFAULT, slot: int = 0,
                 timeout_s: int = 3600, collapse: bool = False,
                 max_batch_mb: int = MAX_BATCH_MB_DEFAULT):
        self.mode, self.work, self.binary = mode, Path(work), binary
        self.tile_div, self.slot, self.timeout_s = tile_div, slot, timeout_s
        self.n_jobs = 0
        self.n_caps = 0
        self.wall = 0.0
        # ── the collapse ───────────────────────────────────────────────────
        self.collapse = bool(collapse)
        self.max_batch_bytes = max(1, int(max_batch_mb)) * 1024 * 1024
        self.n_invocations = 0          # EVERY entry into an executor
        self.pool: dict = {}            # program_key -> captures
        self.memo: dict = {}            # operand_key -> {"keys","paths","meta"}
        self.pending: list = []         # [(key, path, tag)] awaiting a flush
        self._pending_keys: set = set()
        self._prepared: dict = {}       # path -> program_key AFTER prepare()
        self.audited_keys: set = set()  # distinct programs prepare() has seen
        self.executed_keys: set = set()  # distinct programs an executor ran
        self.prunable: set = set()      # paths deletable once pooled
        self.keep_paths: set = set()     # paths kept as evidence
        self.flushes: list = []
        self.n_hits = 0
        self.n_misses = 0
        self.pool_bytes = 0
        self.pruned_bytes = 0

    # ── per-program obligations (subclass hook) ────────────────────────────
    def prepare(self, path: str) -> None:
        """Whatever must happen to a program BEFORE it can be keyed or run.

        layer05b.Runner05B puts the geometry audit and the disclosed INFO_TIER
        retarget here, so both still happen exactly once per distinct program
        — including programs that are only ever served from the pool.
        """

    def _prepare_key(self, path) -> str:
        """The key of a program, with `prepare` guaranteed to have run on it.

        A replayed pass RE-EMITS the cheap families, and `prepare` rewrites
        the file in place (the disclosed INFO_TIER retarget), so a freshly
        emitted file never matches the post-prepare key of the previous pass
        and is prepared again. That is deliberate: the audit and the retarget
        are obligations on the BYTES that reach the tile, and re-emitted bytes
        get them again. `audited_keys` is what "distinct programs audited"
        means, and it is a superset of what the executor ever ran.
        """
        p = str(path)
        d0 = program_key(p)
        prev = self._prepared.get(p)
        if prev is not None and prev == d0:
            return prev                       # already prepared, unchanged
        self.prepare(p)                       # may rewrite the file in place
        k = program_key(p)
        self._prepared[p] = k
        self.audited_keys.add(k)
        return k

    # ── pool-aware fetch ───────────────────────────────────────────────────
    def acquire(self, paths, tag: str):
        """-> [captures per path] if every program is pooled, else None.

        A None is not a failure: it means "this pass cannot serve this op",
        and the caller must leave the value on the host for this pass. The
        missing programs are queued for the next flush.
        """
        paths = [str(p) for p in paths]
        if not self.collapse:
            return self.run(paths, tag)
        keys = [self._prepare_key(p) for p in paths]
        missing = [(k, p) for k, p in zip(keys, paths) if k not in self.pool]
        if missing:
            for k, p in missing:
                if k not in self._pending_keys:
                    self._pending_keys.add(k)
                    self.pending.append((k, p, tag))
            self.n_misses += 1
            return None
        self.n_hits += 1
        return [self.pool[k] for k in keys]

    # ── the operand memo (expensive emitters only) ─────────────────────────
    def memo_lookup(self, fp: str):
        """-> (captures per program, meta) for a memoised emission, or None."""
        e = self.memo.get(fp)
        if e is None:
            return None
        if not all(k in self.pool for k in e["keys"]):
            return None                       # queued but not yet flushed
        self.n_hits += 1
        return [self.pool[k] for k in e["keys"]], e["meta"]

    def memo_store(self, fp: str, paths, meta) -> None:
        self.memo[fp] = {"keys": [self._prepared[str(p)] for p in paths],
                         "paths": [str(p) for p in paths], "meta": meta}

    def mark_prunable(self, paths) -> None:
        """These files may be deleted once their captures are pooled."""
        for p in paths:
            if str(p) not in self.keep_paths:
                self.prunable.add(str(p))

    # ── flush: ONE invocation per <= max_batch_mb of regops ────────────────
    def flush(self, tag: str) -> int:
        """Run every queued program. -> number of executor invocations."""
        if not self.pending:
            return 0
        batches, cur, cur_b = [], [], 0
        for (k, p, t) in self.pending:
            try:
                b = os.path.getsize(p)
            except OSError as e:
                raise RuntimeError(
                    f"{tag}: a queued program vanished before its flush: {p} "
                    f"({e}) — refusing to run a partial batch") from e
            if cur and cur_b + b > self.max_batch_bytes:
                batches.append(cur)
                cur, cur_b = [], 0
            cur.append((k, p, t))
            cur_b += b
        if cur:
            batches.append(cur)
        n_inv = 0
        for i, batch in enumerate(batches):
            paths = [p for _k, p, _t in batch]
            nb = sum(os.path.getsize(p) for p in paths)
            capsets = self._execute(paths, f"{tag}_f{i:02d}")
            for (k, _p, _t), caps in zip(batch, capsets):
                self.pool[k] = caps
            self.pool_bytes += nb
            self.flushes.append({"tag": f"{tag}_f{i:02d}",
                                 "programs": len(paths),
                                 "bytes": nb,
                                 "caps": sum(len(c) for c in capsets)})
            n_inv += 1
            for p in paths:                   # evidence kept, bulk reclaimed
                if p in self.prunable:
                    for q in (p, p.replace(".regops.jsonl", ".manifest.json")):
                        try:
                            self.pruned_bytes += os.path.getsize(q)
                            os.unlink(q)
                        except OSError:
                            pass
                    self.prunable.discard(p)
        self.pending.clear()
        self._pending_keys.clear()
        return n_inv

    # ── the executor itself (unchanged contract) ───────────────────────────
    def run(self, paths, tag: str) -> list:
        """-> [captures per path], in the order given. Raises on any refusal.

        The immediate path: prepare, then run NOW. Used when the collapse is
        off, and by callers that are not inside the layer replay (the
        layer05b RMSNorm-k red arm).
        """
        paths = [str(p) for p in paths]
        for p in paths:
            self._prepare_key(p)
        return self._execute(paths, tag)

    def _execute(self, paths, tag: str) -> list:
        paths = [str(p) for p in paths]
        for p in paths:                       # what the executor actually ran
            k = self._prepared.get(p)
            if k is None:
                raise RuntimeError(
                    f"{tag}: {p} reached the executor without prepare() — "
                    f"the per-program obligations (geometry audit, disclosed "
                    f"retarget) would have been skipped")
            self.executed_keys.add(k)
        t0 = time.perf_counter()
        self.n_invocations += 1
        if len(paths) == 1:
            r = bridge.run_job(paths[0], executor=self.mode,
                               binary=self.binary,
                               cap_out=str(self.work / f"{tag}.cap.jsonl"),
                               tile_div=self.tile_div, slot=self.slot,
                               timeout_s=self.timeout_s)
            if not r["ok"]:
                raise RuntimeError(
                    f"{tag}: executor did not honour the capture contract — "
                    f"rc={r['rc']} ok={r['ok']} notes={r['notes']}\n"
                    f"{r['log'][-2000:]}")
            out = [r["captures"]]
        else:
            b = bx.run_jobs_batched(paths, executor=self.mode,
                                    binary=self.binary,
                                    cap_out=str(self.work / f"{tag}.cap.jsonl"),
                                    tile_div=self.tile_div, slot=self.slot,
                                    timeout_s=self.timeout_s,
                                    workdir=str(self.work / f"{tag}_batch"))
            if not b["ok"]:
                raise RuntimeError(
                    f"{tag}: batched executor run not ok — notes={b['notes']}"
                    f"\n{b['batch']['log'][-2000:]}")
            byp = {f["path"]: f for f in b["files"]}
            out = []
            for p in paths:
                f = byp.get(p) or byp.get(str(Path(p).resolve()))
                if f is None:
                    raise RuntimeError(f"{tag}: no captures attributed to {p}")
                out.append(f["captures"])
        dt = time.perf_counter() - t0
        self.wall += dt
        self.n_jobs += len(paths)
        self.n_caps += sum(len(c) for c in out)
        return out


def _codes_scales(caps, rows: int):
    """(INT8 codes [rows*128], fp16 scale bits [rows]) of a C-1 readback."""
    codes = cd.ro_lanes_to_i32(caps, strict=False).astype(np.int64)
    fs = [t["bits"] for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    n = rows * D_TILE
    if codes.size < n or len(fs) < rows:
        raise RuntimeError(
            f"short C-1 readback: {codes.size}/{n} codes, {len(fs)}/{rows} "
            f"scales — the tile did not produce the whole row")
    return codes[:n], np.asarray(fs[:rows], dtype=np.uint16)


def _c1_reconstruct(codes, scales, rows: int) -> np.ndarray:
    """codes x scale, per 128-element C-1 frame — the value the C-1 egress
    can carry back (B-FEED-WIDTH: the frame is 128 wide, not the whole row)."""
    sv = f16_bits_to_f64(np.asarray(scales, dtype=np.uint16))
    return (np.asarray(codes, dtype=np.float64).reshape(rows, D_TILE)
            * np.asarray(sv, dtype=np.float64)[:, None]).reshape(-1)


# ══════════════════════════ the offload seam ═══════════════════════════════

class LayerOffloader:
    """Rebinds golden's per-op functions for ONE (layer, step) and substitutes
    the tile's values back into the model's own state."""

    def __init__(self, *, n_layers: int, layer: int, mode: str, work: Path,
                 scope: Scope, runner: Runner, step: int | None = None,
                 poison: float | None = None, verbose: bool = True,
                 max_passes: int = MAX_PASSES_DEFAULT):
        self.n_layers, self.t_layer, self.t_step = n_layers, layer, step
        self.mode, self.work, self.scope, self.runner = mode, Path(work), scope, runner
        self.poison, self.verbose = poison, verbose
        self.layer_calls = 0
        self.armed = False
        self.records: list[OpRec] = []
        self.checks: list[tuple] = []          # (name, ok, detail)
        self._guard: set = set()
        self.cur = None                        # per-layer state
        self.sites = {}
        # ── the layer replay (see Runner's docstring) ──────────────────────
        self.max_passes = max(1, int(max_passes))
        self.pass_no = 0
        self._say_buf: list = []
        self.passes: dict = {}                 # layer -> passes it needed
        self.invocations: dict = {}            # layer -> invocations it cost

    # ── rebinding (context manager; ALWAYS restored) ───────────────────────
    def __enter__(self):
        self.sites = _locate_f16_sites()
        self._o_layer = tf.decoder_layer_fx
        self._o_gemm = tf.gemm_i8_ksplit
        self._o_rope = tf.rope_fx
        self._o_norm = tf.rmsnorm_fx_wide
        self._o_epi = tf._proj_epilogue
        self._o_f16 = tf._f16
        self._o_attn_tf = tf.attention_core
        self._o_attn_at = at.attention_core
        assert self._o_attn_tf is self._o_attn_at, \
            "transformer/attention already resolve different attention_core"
        self._attn_sig = inspect.signature(self._o_attn_at)
        tf.decoder_layer_fx = self._layer
        tf.gemm_i8_ksplit = self._gemm
        tf.rope_fx = self._rope
        tf.rmsnorm_fx_wide = self._norm
        tf._proj_epilogue = self._projepi
        tf._f16 = self._f16
        tf.attention_core = self._attn
        at.attention_core = self._attn
        return self

    def __exit__(self, *exc):
        tf.decoder_layer_fx = self._o_layer
        tf.gemm_i8_ksplit = self._o_gemm
        tf.rope_fx = self._o_rope
        tf.rmsnorm_fx_wide = self._o_norm
        tf._proj_epilogue = self._o_epi
        tf._f16 = self._o_f16
        tf.attention_core = self._o_attn_tf
        at.attention_core = self._o_attn_at
        return False

    # ── re-entrancy: our own calls into golden must be TRANSPARENT ─────────
    # A rebind is a module global, so golden code we call for grading (or to
    # get the object we substitute into) lands back in these wrappers. The
    # guard is per-op, NOT global, because _proj_epilogue's inner GEMM is an
    # op we DO want offloaded while the epilogue itself is only observed.
    @contextmanager
    def _own(self, *ops):
        added = [o for o in ops if o not in self._guard]
        self._guard.update(added)
        try:
            yield
        finally:
            self._guard.difference_update(added)

    def _busy(self, op: str) -> bool:
        return op in self._guard

    def _serving(self, op: str) -> bool:
        return self.armed and self.scope.on(op)

    def chk(self, name: str, ok: bool, detail=None):
        self.checks.append((name, bool(ok), detail))
        if not ok:
            self.say(f"[layer_offload] CHECK FAILED: {name}: {detail}")

    def say(self, msg: str) -> None:
        """A progress line for THIS pass.

        Buffered, because a pass whose programs were not all in the pool is
        discarded — printing its lines would report coverage the run did not
        have. The buffer is flushed only by the pass that is kept.
        """
        if self.verbose:
            self._say_buf.append(msg)

    def _pass_begin(self, r0: int, c0: int) -> None:
        """A new pass over the SAME layer call: drop what the discarded pass
        appended. A subclass that streams `records` to a live ledger keeps a
        print cursor; clamp it rather than leave it past the end."""
        del self.records[r0:]
        del self.checks[c0:]
        self._say_buf = []
        cur = getattr(self, "_shown", None)
        if isinstance(cur, int) and cur > len(self.records):
            self._shown = len(self.records)

    def _say_flush(self) -> None:
        for m in self._say_buf:
            eprint(m)
        self._say_buf = []

    # ── (step, layer) bookkeeping + arm/disarm ─────────────────────────────
    @property
    def step(self) -> int:
        return self.layer_calls // self.n_layers

    @property
    def layer(self) -> int:
        return self.layer_calls % self.n_layers

    def _layer(self, X, w, tier, *a, **kw):
        X = np.asarray(X, dtype=np.float64)
        T = X.shape[0] - 1
        layer, step = self.layer, self.step
        target = (layer == self.t_layer
                  and (self.t_step is None or step == self.t_step))
        if not target:
            try:
                return self._o_layer(X, w, tier, *a, **kw)
            finally:
                self.layer_calls += 1
        if T > TOKENS_MAX:
            raise SystemExit(
                f"REFUSE: T={T} > {TOKENS_MAX} — the head would become a "
                f"ChunkedHead whose merge needs per-chunk sm_m/sm_l, which "
                f"the mailbox does not expose. An offloaded chunk is not "
                f"possible (audit N8). Reduce --max-tokens / the prompt.")
        # C-LBUS BUS_ON: the LAYER deq/swiglu units REFUSE an ungraded fp32
        # composite (apex_layer_deq.sv:90-92), so the residual and SwiGLU ops
        # are only reachable in the tile's own bus-grade composition mode.
        kw = dict(kw)
        kw["bus"] = tf.BUS_ON
        tag = f"lo_s{step:03d}_L{layer:02d}"
        r0, c0 = len(self.records), len(self.checks)
        n_pass = max(self.max_passes if self.runner.collapse else 1, 1)
        inv0 = self.runner.n_invocations
        t0 = time.perf_counter()
        try:
            for p in range(1, n_pass + 1):
                self.pass_no = p
                # a pass that could not serve every program it asked for is
                # DISCARDED: those ops were left on the host for that pass, so
                # its ledger would understate coverage. Only the pass that
                # served everything is kept.
                self._pass_begin(r0, c0)
                self.cur = _LayerState(X=X, w=w, T=T, step=step, layer=layer,
                                       tag=tag)
                self.cur.x_row = self._o_f16(X)[T]   # the row resid-1 adds to
                self.armed = True
                try:
                    r = self._o_layer(X, w, tier, *a, **kw)
                finally:
                    self.armed = False
                if not self.runner.pending:
                    self._say_flush()
                    break
                n_queued = len(self.runner.pending)
                n_inv = self.runner.flush(f"{tag}_p{p}")
                if self.verbose:
                    eprint(f"[layer_offload] {tag} pass {p}: {n_queued} "
                           f"program(s) the pool could not serve -> {n_inv} "
                           f"executor invocation(s)")
            else:
                raise SystemExit(
                    f"REFUSE: {tag} did not reach a stable program set in "
                    f"{n_pass} passes — every replay is still asking for "
                    f"programs the pool has never run "
                    f"({len(self.runner.pending)} queued). The layer's "
                    f"operands are not converging, so no ledger this harness "
                    f"printed could be trusted. Raise --max-passes only if "
                    f"you can name the op that is still moving.")
        finally:
            self.layer_calls += 1
        self.cur.seconds = time.perf_counter() - t0
        self.cur.passes = self.pass_no
        self.passes[layer] = self.pass_no
        self.invocations[layer] = self.runner.n_invocations - inv0
        self._layer_done(tag)
        self._post_checks(r)
        return r

    def _layer_done(self, tag: str) -> None:
        """Hook: the layer reached its fixed point and its ledger is final."""

    # ── did the tile's values actually reach the model's own state? ────────
    def _post_checks(self, r):
        c = self.cur
        if c.q_rope:
            ok = all(np.array_equal(r.q_rope[h * c.hd:(h + 1) * c.hd], v)
                     for h, v in c.q_rope.items())
            self.chk("rope: substituted q rows are the model's r.q_rope", ok,
                     {"heads": sorted(c.q_rope)})
        if c.attn_cores:
            same = all(r.heads[h] is core for h, core in c.attn_cores.items())
            consumed = all(
                np.array_equal(r.attn[h * c.hd:(h + 1) * c.hd],
                               core.o8.astype(np.float64)
                               * tf.f16_grade(core.s_out))
                for h, core in c.attn_cores.items())
            self.chk("attention: tile o8/s_out are the model's r.attn",
                     same and consumed,
                     {"core_identity": same, "consumed": consumed})
        if c.acc.get("Wq") is not None and c.proj_served.get("Wq"):
            s_h = f16_bits_to_f64(r.s_h)
            comp = tf.f16_grade(float(s_h[c.T]) * float(c.w.s_wq))
            want = c.acc["Wq"][0].astype(np.float64) * comp
            if c.w.bq is not None:
                want = want + np.asarray(c.w.bq, dtype=np.float64)
            want = self._o_f16(want)          # C-LBUS NP-r: the S-2 write bus
            self.chk("projections: the spliced Wq accumulators are r.q_real",
                     np.array_equal(r.q_real, want),
                     {"n_diff": int(np.sum(r.q_real != want))})
        for tag, val in (("r1", c.r1), ("r2", c.r2)):
            if val is not None:
                self.chk(f"residual {tag}: the tile row IS the model's r.{tag}",
                         np.array_equal(getattr(r, tag), val))
        if c.h2 is not None:
            self.chk("RMSNorm-2: the tile row IS the model's r.h2",
                     np.array_equal(r.h2, np.asarray(c.h2, dtype=np.int64)))
        if c.swiglu is not None:
            self.chk("SwiGLU: the tile row IS the model's r.swiglu",
                     np.array_equal(r.swiglu, c.swiglu))

    # ── projections: gemm_i8_ksplit ────────────────────────────────────────
    def _gemm(self, A, B, *a, **kw):
        if self._busy("gemm") or not self.armed:
            return self._o_gemm(A, B, *a, **kw)
        with self._own("gemm"):
            gold = self._o_gemm(A, B, *a, **kw)
        c = self.cur
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        label = c.next_proj(B)                 # verified against w's own tensor
        rec = OpRec(op="proj", name=f"{c.tag}_{label}",
                    source="HOST (golden)", n_total=int(A.shape[0] * B.shape[1]))
        if not self._serving("proj") or self.scope.proj_cols <= 0:
            rec.notes.append("not in --ops / --proj-cols 0")
            self.records.append(rec)
            c.acc[label] = np.asarray(gold, dtype=np.int64)
            return gold
        out = np.asarray(gold, dtype=np.int64).copy()
        ncol = min(self.scope.proj_cols, B.shape[1])
        ncol -= ncol % MXE_N
        nrow = min(self.scope.proj_rows, A.shape[0])
        t0 = time.perf_counter()
        served = jobs = 0
        eq = True
        deferred = 0
        for m in range(nrow):
            for c0 in range(0, ncol, MXE_N):
                n = min(MXE_N, ncol - c0)
                nm = f"{c.tag}_{label}_m{m}_c{c0}"
                try:
                    got = self._tile_matvec(B[:, c0:c0 + n], A[m], nm)
                except (AssertionError, gl.InjectRangeError) as e:
                    rec.notes.append(f"m{m} c{c0}: staging REFUSED: {e}")
                    continue
                if got is None:            # queued for this pass's flush
                    deferred += 1
                    continue
                acc, nj, ncap = got
                jobs += nj
                rec.caps += ncap
                out[m, c0:c0 + n] = acc[:n]
                served += n
                eq = eq and bool(np.array_equal(acc[:n], gold[m, c0:c0 + n]))
        if deferred:
            rec.notes.append(f"{deferred} block(s) queued for this pass's "
                             f"flush — left on the host for this pass")
        rec.seconds = round(time.perf_counter() - t0, 2)
        rec.jobs, rec.n_served = jobs, served
        rec.source = (f"TILE ({self.mode})" if served else "HOST (golden)")
        rec.grade_ok = eq if served else None
        rec.exact = eq if served else None
        rec.detail = {"A": list(A.shape), "B": list(B.shape),
                      "rows_served": nrow if served else 0, "cols_served": ncol}
        self.records.append(rec)
        c.acc[label] = out
        c.proj_served[label] = served
        if served:
            self.say(f"[layer_offload] proj {label}: {served}/{rec.n_total} "
                     f"values from the tile, bit-exact={eq} ({rec.seconds}s)")
        return out.astype(np.int32)

    def _tile_matvec(self, W, x8, name):
        """x8[K] x W[K,N<=8] -> INT32, computed by the tile (K-split jobs).

        -> (acc, n_programs, n_caps), or None when this pass could not serve
        the job from the capture pool (the programs are queued for the flush
        at the end of the pass and the caller leaves the value on the host).

        The staging contract (gemm_job.stage_plan F1) needs the activation
        row's amax code to be exactly 127; golden's C-1 does not guarantee it,
        so a +127 SENTINEL lane with a ZERO weight row is appended when
        needed — F3 (staged product == un-staged product) is asserted inside
        stage_plan, so the sentinel provably contributes nothing.
        """
        W = np.asarray(W, dtype=np.int64)
        x8 = np.asarray(x8, dtype=np.int64)
        if self.mode == "golden":
            return cp.gemm_i8_ksplit(x8[None, :], W)[0].astype(np.int64), 0, 0
        amax = int(np.max(np.abs(x8), initial=0))
        assert amax <= 127, (
            f"activation code magnitude {amax} > 127 — the INT8-symmetric "
            f"staging bound (a -128 code cannot be an amax anchor)")
        if amax != 127:
            x8 = np.concatenate([x8, [127]])
            W = np.concatenate([W, np.zeros((1, W.shape[1]), dtype=np.int64)])
        plan = gj.stage_plan(x8, W)
        paths = []
        for ji, (r0, nr, K) in enumerate(plan.chunks()):
            a = r0 * D_TILE
            p, _ = gj.build_gemm_job_full(plan.Wst[a:a + K], plan.xst[a:a + K],
                                          self.work, f"{name}_k{ji}")
            paths.append(p)
        caps = self.runner.acquire(paths, name)
        if caps is None:
            return None
        acc = gj.accumulate_partials([gj.decode_acc(c) for c in caps])
        return (np.asarray(acc, dtype=np.int64), len(paths),
                sum(len(c) for c in caps))

    # ── RoPE ───────────────────────────────────────────────────────────────
    def _rope(self, x, m, theta=None):
        if self._busy("rope") or not self.armed:
            return self._o_rope(x, m, theta)
        with self._own("rope", "f16"):
            gold = self._o_rope(x, m, theta)
        c = self.cur
        x = np.asarray(x, dtype=np.float64)
        hd = x.shape[-1]
        if x.ndim != 1:
            c.rope_host_rows += int(x.shape[0])
            return gold
        head = c.rope_calls
        c.rope_calls += 1
        lim = self.scope.rope_heads
        if not self._serving("rope") or (lim >= 0 and head >= lim):
            return gold
        rec = OpRec(op="rope", name=f"{c.tag}_h{head:02d}",
                    source="HOST (golden)", n_total=int(hd))
        if hd != D_TILE:
            rec.notes.append(f"head_dim {hd} != {D_TILE}: rope_row's frame")
            self.records.append(rec)
            return gold
        if not _exact_f16(x):
            rec.notes.append("pre-rope row is not on the fp16 grid (the S-2 "
                             "bus the RTL rotates); left on the host")
            self.records.append(rec)
            return gold
        th = theta if theta is not None else tf.rope_theta(hd)
        step = {"head_dim": np.int64(hd),
                "q_pre_bits": f64_to_f16_bits(x).astype(np.uint16),
                "phase_q": np.asarray(tf.rope_phase_q(float(m), th),
                                      dtype=np.int64)}
        t0 = time.perf_counter()
        if self.mode == "golden":
            codes, scales = at.quant_rows_i8(gold[None, :])
            codes, scales = codes[0].astype(np.int64), scales.astype(np.uint16)
            caps, jobs = [], 0
            grade = {"equal": True, "note": "dry-run: golden's own C-1 view"}
        else:
            man = gl.build_rope_stage(step, head=0, out_dir=self.work,
                                      name=rec.name)
            got = self.runner.acquire([man["path"]], rec.name)
            if got is None:
                return gold          # queued; this pass leaves it on the host
            caps = got[0]
            codes, scales = _codes_scales(caps, rows=1)
            grade = gl.grade_codes_scales(caps, gold, rows=1)
            jobs = 1
        recon = _c1_reconstruct(codes, scales, rows=1)
        # the model's next consumer (attention_core Q7) re-quantizes this row
        # with the SAME quant_rows_i8 — CHECK that the reconstruction survives
        # it unchanged, rather than assuming C-1 idempotence.
        rc, rs = at.quant_rows_i8(recon[None, :])
        ident = bool(np.array_equal(rc[0].astype(np.int64), codes)
                     and int(rs[0]) == int(scales[0]))
        gc, gs = at.quant_rows_i8(gold[None, :])
        same_as_golden = bool(np.array_equal(rc[0].astype(np.int64),
                                             gc[0].astype(np.int64))
                              and int(rs[0]) == int(gs[0]))
        rec.seconds = round(time.perf_counter() - t0, 2)
        rec.jobs, rec.caps, rec.n_served = jobs, len(caps), int(hd)
        rec.source = ("GOLDEN (dry-run)" if self.mode == "golden"
                      else f"TILE ({self.mode})")
        rec.grade_ok = bool(grade.get("equal"))
        rec.exact = bool(np.array_equal(recon, gold))
        rec.detail = {"grade": grade, "requant_identical": ident,
                      "downstream_codes_match_golden": same_as_golden,
                      "max_abs_delta": float(np.max(np.abs(recon - gold)))}
        self.chk(f"rope h{head:02d}: C-1 reconstruction re-quantizes to the "
                 f"tile's own codes (idempotent)", ident)
        self.chk(f"rope h{head:02d}: the row the model consumes quantizes "
                 f"exactly as golden's", same_as_golden)
        self.records.append(rec)
        val = recon * float(self.poison) if self.poison is not None else recon
        c.q_rope[head] = val
        return val

    # ── attention ──────────────────────────────────────────────────────────
    def _attn(self, *a, **kw):
        if self._busy("attn") or not self.armed:
            return self._o_attn_at(*a, **kw)
        with self._own("attn"):
            core = self._o_attn_at(*a, **kw)
        c = self.cur
        head = c.attn_calls
        c.attn_calls += 1
        lim = self.scope.heads
        if not self._serving("attn") or (lim >= 0 and head >= lim):
            return core
        b = self._attn_sig.bind(*a, **kw)
        b.apply_defaults()
        A = b.arguments
        rec = OpRec(op="attn", name=f"{c.tag}_h{head:02d}",
                    source="HOST (golden)", n_total=int(core.D))
        assert core.T <= TOKENS_MAX, "chunk core reached the offload seam (N8)"
        t0 = time.perf_counter()
        if self.mode == "golden":
            acc, s_c = np.asarray(core.acc_o, dtype=np.int64), core.s_c
            caps, jobs = [], 0
            grade = {"ok": True, "note": "dry-run: golden's own accumulators"}
        else:
            q = np.asarray(A["q_f64"], dtype=np.float64)
            K = np.asarray(A["K_f16"], dtype=np.uint16)
            V = np.asarray(A["V_f16"], dtype=np.uint16)
            path, _man, prep = cj.build_compute_job_full(
                q, K, V, A["tier"], A["G"], list(A.get("outlier_idx") or ()),
                self.work, rec.name)
            got = self.runner.acquire([path], rec.name)
            if got is None:
                return core          # queued; this pass leaves it on the host
            caps = got[0]
            with self._own("attn"):
                grade = cj.grade_compute_job(
                    caps, q, K, V, A["tier"], A["G"],
                    list(A.get("outlier_idx") or ()), prep=prep)
            acc = np.asarray(grade["acc_i32"], dtype=np.int64)
            s_c = grade["s_c"]["captured"]
            if s_c is None:
                raise RuntimeError(f"{rec.name}: no s_c capture — the host "
                                   f"epilogue has no tile c-scale to use")
            jobs = 1
        ep = cd.epilogue(acc, s_c)
        out_hat = np.asarray(ep["out_hat"], dtype=np.float64)
        if self.poison is not None:
            out_hat = out_hat * float(self.poison)
        rec.seconds = round(time.perf_counter() - t0, 2)
        rec.jobs, rec.caps, rec.n_served = jobs, len(caps), int(core.D)
        rec.source = ("GOLDEN (dry-run)" if self.mode == "golden"
                      else f"TILE ({self.mode})")
        rec.grade_ok = bool(np.array_equal(acc, np.asarray(core.acc_o,
                                                           dtype=np.int64)))
        rec.exact = bool(np.array_equal(ep["out_hat"], core.out_hat))
        rec.detail = {"T": int(core.T), "D": int(core.D), "tier": core.tier,
                      "acc_grade": cd.grade(acc, core.acc_o),
                      "o8_grade": cd.grade(ep["o8"], core.o8),
                      "rq_tile": tuple(int(v) for v in ep["rq"]),
                      "rq_golden": tuple(int(v) for v in core.rq),
                      "compute_job_ok": grade.get("ok")}
        self.records.append(rec)
        # THE SUBSTITUTION: tile-derived values re-enter the model's own object
        core.acc_o = acc
        core.rq = ep["rq"]
        core.o8 = np.asarray(ep["o8"], dtype=np.int64)
        core.s_out = float(ep["s_out"])
        core.out_hat = out_hat
        c.attn_cores[head] = core
        return core

    # ── RMSNorm (told apart by which gamma golden handed us) ───────────────
    def _norm(self, x, g, chunk: int = 128):
        if self._busy("norm") or not self.armed:
            return self._o_norm(x, g, chunk)
        with self._own("norm"):
            gold = self._o_norm(x, g, chunk)
        c = self.cur
        gi = np.asarray(g, dtype=np.int64)
        which = None
        if gi.size == np.asarray(c.w.gamma2).size and np.array_equal(
                gi, np.asarray(c.w.gamma2, dtype=np.int64)):
            which = "norm2"
        elif np.array_equal(gi, np.asarray(c.w.gamma1, dtype=np.int64)):
            which = "norm1"
        row = c.norm1_calls if which == "norm1" else 0
        if which == "norm1":
            c.norm1_calls += 1
        if which is None:
            return gold
        if which == "norm2":
            # SwiGLU's composites are built from h2 — track it even when the
            # norm itself is left on the host, so --ops swiglu stands alone.
            c.h2 = [int(v) for v in gold[0]]
        if not self._serving("norm"):
            return gold
        if which == "norm1" and row >= self.scope.norm1_rows:
            return gold
        dm_all = int(np.asarray(x).size)
        dm = dm_all if self.scope.norm_dm < 0 else min(self.scope.norm_dm,
                                                       dm_all)
        dm -= dm % D_TILE
        rec = OpRec(op="norm", name=f"{c.tag}_{which}"
                                    + (f"_r{row}" if which == "norm1" else ""),
                    source="HOST (golden)", n_total=dm_all)
        if dm < 2 * D_TILE:
            rec.notes.append(f"dm={dm}: the R4 chunk arm needs k in "
                             f"[{gl.RMS_EXT_K_MIN},{gl.RMS_EXT_K_MAX}]")
            self.records.append(rec)
            return gold
        y = np.asarray(gold[0], dtype=np.int64)
        step = {"r1_8": np.asarray(x, dtype=np.int64),
                "gamma2": gi}
        t0 = time.perf_counter()
        if self.mode == "golden":
            codes, scales = at.quant_rows_i8(
                y[:dm].astype(np.float64).reshape(-1, D_TILE) / 256.0)
            codes = codes.reshape(-1).astype(np.int64)
            scales = np.asarray(scales, dtype=np.uint16)
            caps, jobs = [], 0
            grade = {"equal": True, "note": "dry-run: golden's own C-1 view"}
        else:
            man = gl.build_norm2_chunked(step, dm=dm, out_dir=self.work,
                                         name=rec.name)
            got = self.runner.acquire([man["path"]], rec.name)
            if got is None:
                return gold          # queued; this pass leaves it on the host
            caps = got[0]
            codes, scales = _codes_scales(caps, rows=dm // D_TILE)
            grade = gl.grade_codes_scales(caps, y[:dm].astype(np.float64)
                                          / 256.0, rows=dm // D_TILE)
            _, want_sums, want_total = gl._norm2_chunk_sums(step, dm)
            got = [c2["value"] for c2 in caps if c2.get("sem") == "rs2"]
            grade["sums_equal"] = bool(got == want_sums)
            grade["sum_total_equal"] = bool(sum(got) == want_total
                                            == int(man["ext_sum2"]))
            grade["equal"] = bool(grade["equal"] and grade["sums_equal"]
                                  and grade["sum_total_equal"])
            jobs = 1
        recon = _c1_reconstruct(codes, scales, rows=dm // D_TILE) * 256.0
        out = y.copy()
        out[:dm] = np.asarray(rne(recon), dtype=np.int64)
        # the model consumes h2 ONLY through a whole-row C-1 (transformer.py
        # :572) — report what the 128-framed egress costs THERE, which is the
        # measurement B-FEED-WIDTH actually implies.
        c8_t, s8_t = at.quant_rows_i8(out.astype(np.float64)[None, :] / 256.0)
        c8_g, s8_g = at.quant_rows_i8(y.astype(np.float64)[None, :] / 256.0)
        rec.seconds = round(time.perf_counter() - t0, 2)
        rec.jobs, rec.caps, rec.n_served = jobs, len(caps), dm
        rec.source = ("GOLDEN (dry-run)" if self.mode == "golden"
                      else f"TILE ({self.mode})")
        rec.grade_ok = bool(grade.get("equal"))
        rec.exact = bool(np.array_equal(out, y))
        rec.detail = {
            "which": which, "dm": dm, "rows": dm // D_TILE, "grade": grade,
            "max_abs_delta_q78": int(np.max(np.abs(out - y), initial=0)),
            "downstream_c1_codes_n_diff":
                int(np.sum(c8_t[0] != c8_g[0])),
            "downstream_c1_scale_equal": bool(int(s8_t[0]) == int(s8_g[0])),
            "egress": "C-1 view, 128-wide frames (B-FEED-WIDTH)"}
        self.records.append(rec)
        vals = [int(v) for v in out]
        if which == "norm2":
            if self.poison is not None:
                vals = [int(rne(np.array([v * float(self.poison)]))[0])
                        for v in vals]
            c.h2 = vals
        return (vals, gold[1], gold[2])

    # ── the projection epilogue: OBSERVED, never substituted ───────────────
    def _projepi(self, *a, **kw):
        """The C-2 requant epilogue stays on the host (the standing fence).
        This wrapper only learns the o8 code stream and the graded composite
        that the residual's tile job consumes — its GEMM is still offloaded,
        because `gemm` is deliberately NOT guarded here."""
        with self._own("projepi"):
            out_real, s_out = self._o_epi(*a, **kw)
        if not self.armed:
            return out_real, s_out
        c = self.cur
        which = "r1" if c.epi_calls == 0 else "r2"
        c.epi_calls += 1
        o8 = np.rint(np.asarray(out_real, dtype=np.float64) / float(s_out))
        if not np.array_equal(o8 * float(s_out), out_real):
            raise SystemExit(
                f"REFUSE: could not recover the {which} epilogue's INT8 code "
                f"stream from (out_real, s_out) exactly — the residual job "
                f"would be fed operands this file GUESSED. Golden changed.")
        c.epi[which] = (o8.astype(np.int64), float(s_out))
        return out_real, s_out

    # ── the three `_f16` narrowing sites ───────────────────────────────────
    def _f16(self, x):
        if self._busy("f16") or not self.armed:
            return self._o_f16(x)
        site = sys._getframe(1).f_lineno
        if site == self.sites["r1"]:
            return self._residual("r1", x)
        if site == self.sites["r2"]:
            return self._residual("r2", x)
        if site == self.sites["swiglu"]:
            return self._swiglu(x)
        return self._o_f16(x)

    def _residual(self, which: str, arg):
        with self._own("f16"):
            gold = self._o_f16(arg)
        c = self.cur
        if which not in c.epi or not self._serving("resid"):
            return gold
        o8, s_out = c.epi[which]
        row_in = c.x_row if which == "r1" else c.r1
        if row_in is None:
            return gold
        # the interception is VERIFIED against operands tracked independently
        # of the call site: a mis-identified `_f16` cannot pass this.
        if not np.array_equal(np.asarray(arg, dtype=np.float64),
                              row_in + o8.astype(np.float64) * s_out):
            raise SystemExit(
                f"REFUSE: the `_f16` call at the {which} site did not receive "
                f"row + o8*s_out — the call-site map is wrong or golden's "
                f"residual composition changed. Refusing to offload a "
                f"narrowing whose operands this file cannot name.")
        comp = _f32_bits(s_out)
        rec = OpRec(op="resid", name=f"{c.tag}_{which}", source="HOST (golden)",
                    n_total=int(np.asarray(gold).size))
        if not _is_graded(comp):
            rec.notes.append(f"composite {comp:#010x} is not fp16-grade — "
                             f"apex_layer_deq.sv:90-92 would refuse it")
            self.records.append(rec)
            return gold
        if not _exact_f16(row_in):
            rec.notes.append("the row this residual adds onto is not on the "
                             "fp16 grid (the C-6 residual bus)")
            self.records.append(rec)
            return gold
        n_all = int(np.asarray(gold).size)
        cols = n_all if self.scope.resid_cols < 0 else min(self.scope.resid_cols,
                                                           n_all)
        sl = min(self.scope.resid_slice, cols)
        cols -= cols % sl
        exp_bits = f64_to_f16_bits(np.asarray(gold, dtype=np.float64))
        step = {f"{which}_o8": o8,
                f"{which}_comp": comp,
                f"{which}_row_in_bits": f64_to_f16_bits(row_in),
                f"{which}_bits": exp_bits}
        t0 = time.perf_counter()
        offs = list(range(0, cols, sl))
        if self.mode == "golden":
            got = np.asarray(exp_bits[:cols], dtype=np.uint16)
            caps_n, jobs, per_slice = 0, 0, [{"equal": True} for _ in offs]
        else:
            paths, exps = [], []
            for off in offs:
                man, exp = gl.build_resid_stage(
                    step, which=which, cols=sl, off=off, out_dir=self.work,
                    name=f"{rec.name}_{off:04d}")
                paths.append(man["path"])
                exps.append(exp)
            capsets = self.runner.acquire(paths, rec.name)
            if capsets is None:
                return gold          # queued; this pass leaves it on the host
            per_slice = [gl.grade_resid(cs, e) for cs, e in zip(capsets, exps)]
            got = np.concatenate(
                [np.asarray([t["bits"] for t in cd.rdata_f16(cs)[:sl]],
                            dtype=np.uint16) for cs in capsets])
            caps_n = sum(len(cs) for cs in capsets)
            jobs = len(paths)
        if got.size != cols:
            raise RuntimeError(f"{rec.name}: reassembled {got.size} fp16 words, "
                               f"expected {cols}")
        out = np.asarray(gold, dtype=np.float64).copy()
        out[:cols] = f16_bits_to_f64(got).reshape(-1)
        rec.seconds = round(time.perf_counter() - t0, 2)
        rec.jobs, rec.caps, rec.n_served = jobs, caps_n, int(cols)
        rec.source = ("GOLDEN (dry-run)" if self.mode == "golden"
                      else f"TILE ({self.mode})")
        rec.grade_ok = bool(all(g["equal"] for g in per_slice)
                            and np.array_equal(got, exp_bits[:cols]))
        rec.exact = bool(np.array_equal(out, gold))
        rec.detail = {"slices": len(offs), "slice": sl,
                      "reassembled_equal_golden":
                          bool(np.array_equal(got, exp_bits[:cols])),
                      "per_slice_pass": int(sum(g["equal"] for g in per_slice)),
                      "jobc": f"{comp:#010x}", "egress": "LAYER_RDATA fp16"}
        self.records.append(rec)
        if self.poison is not None:
            out = out * float(self.poison)
        setattr(c, which, out)
        self.say(f"[layer_offload] resid {which}: {cols}/{n_all} elements "
                 f"from the tile in {len(offs)} slices, "
                 f"bit-exact={rec.grade_ok} ({rec.seconds}s)")
        return out

    def _swiglu(self, arg):
        with self._own("f16"):
            gold = self._o_f16(arg)
        c = self.cur
        if not self._serving("swiglu") or c.acc.get("Wg") is None \
                or c.acc.get("Wu") is None or c.h2 is None:
            return gold
        w = c.w
        h2 = np.asarray(c.h2, dtype=np.float64)
        _h8, s_h2b = at.quant_rows_i8(h2[None, :] / 256.0)
        s_h2 = float(f16_bits_to_f64(np.array([s_h2b[0]]))[0])
        comp_g = tf.f16_grade(s_h2 * float(w.s_wg))
        comp_u = tf.f16_grade(s_h2 * float(w.s_wu))
        acc_g = c.acc["Wg"][0].astype(np.int64)
        acc_u = c.acc["Wu"][0].astype(np.int64)
        with self._own("f16"):
            want = (tf.silu_apply(acc_g.astype(np.float64) * comp_g)
                    * (acc_u.astype(np.float64) * comp_u))
        if not np.array_equal(np.asarray(arg, dtype=np.float64), want):
            raise SystemExit(
                "REFUSE: the `_f16` call at the SwiGLU site did not receive "
                "silu(gate)*up built from the accumulators and composites "
                "this file tracked — the call-site map is wrong or golden's "
                "SwiGLU composition changed.")
        n_all = int(acc_g.size)
        rec = OpRec(op="swiglu", name=f"{c.tag}_swiglu", source="HOST (golden)",
                    n_total=n_all)
        for tag, cw in (("gate", comp_g), ("up", comp_u)):
            if not _is_graded(_f32_bits(cw)):
                rec.notes.append(f"{tag} composite not fp16-grade — "
                                 f"asu_swiglu would refuse it")
                self.records.append(rec)
                return gold
        chunks_all = n_all // gl.fmt.SWG_COLS_MAX
        chunks = chunks_all if self.scope.swiglu_chunks < 0 \
            else min(self.scope.swiglu_chunks, chunks_all)
        per = D_TILE // gl.fmt.SWG_COLS_MAX               # 2 chunks per C-1 row
        chunks -= chunks % per
        if chunks < per:
            rec.notes.append("fewer than one full C-1 feeder row of chunks")
            self.records.append(rec)
            return gold
        n = chunks * gl.fmt.SWG_COLS_MAX
        rows = n // D_TILE
        step = {"acc_g": acc_g, "acc_u": acc_u, "comp_g": _f32_bits(comp_g),
                "comp_u": _f32_bits(comp_u), "d_ffn": np.int64(n_all)}
        t0 = time.perf_counter()
        if self.mode == "golden":
            codes, scales = at.quant_rows_i8(
                np.asarray(gold, dtype=np.float64)[:n].reshape(rows, D_TILE))
            codes = codes.reshape(-1).astype(np.int64)
            scales = np.asarray(scales, dtype=np.uint16)
            caps, jobs = [], 0
            grade = {"equal": True, "note": "dry-run: golden's own C-1 view"}
        else:
            try:
                man = gl.build_swiglu_stage(step, chunks=chunks,
                                            out_dir=self.work, name=rec.name)
            except (gl.InjectRangeError, AssertionError) as e:
                rec.notes.append(f"SwiGLU job REFUSED at emission: {e}")
                self.records.append(rec)
                return gold
            got = self.runner.acquire([man["path"]], rec.name)
            if got is None:
                return gold          # queued; this pass leaves it on the host
            caps = got[0]
            codes, scales = _codes_scales(caps, rows=rows)
            grade = gl.grade_codes_scales(
                caps, np.asarray(gold, dtype=np.float64)[:n], rows=rows)
            jobs = 1
        recon = _c1_reconstruct(codes, scales, rows=rows)
        out = np.asarray(gold, dtype=np.float64).copy()
        out[:n] = recon
        c8_t, s8_t = at.quant_rows_i8(out[None, :])
        c8_g, s8_g = at.quant_rows_i8(np.asarray(gold, dtype=np.float64)[None, :])
        rec.seconds = round(time.perf_counter() - t0, 2)
        rec.jobs, rec.caps, rec.n_served = jobs, len(caps), int(n)
        rec.source = ("GOLDEN (dry-run)" if self.mode == "golden"
                      else f"TILE ({self.mode})")
        rec.grade_ok = bool(grade.get("equal"))
        rec.exact = bool(np.array_equal(out, gold))
        rec.detail = {"chunks": chunks, "chunks_total": chunks_all,
                      "feeder_rows": rows, "grade": grade,
                      "max_abs_delta": float(np.max(np.abs(out - gold))),
                      "downstream_c1_codes_n_diff":
                          int(np.sum(c8_t[0] != c8_g[0])),
                      "downstream_c1_scale_equal":
                          bool(int(s8_t[0]) == int(s8_g[0])),
                      "egress": "C-1 view, 128-wide frames (B-FEED-WIDTH)"}
        self.records.append(rec)
        if self.poison is not None:
            out = out * float(self.poison)
        c.swiglu = out
        self.say(f"[layer_offload] swiglu: {n}/{n_all} columns from the "
                 f"tile ({chunks} x 64-col jobs), C-1 view bit-exact="
                 f"{rec.grade_ok} ({rec.seconds}s)")
        return out


@dataclass
class _LayerState:
    """Everything the wrappers must know about the layer call in flight."""
    X: np.ndarray
    w: object
    T: int
    step: int
    layer: int
    tag: str
    x_row: np.ndarray = None
    seconds: float = 0.0
    passes: int = 1
    acc: dict = field(default_factory=dict)
    proj_served: dict = field(default_factory=dict)
    epi: dict = field(default_factory=dict)
    epi_calls: int = 0
    rope_calls: int = 0
    rope_host_rows: int = 0
    attn_calls: int = 0
    norm1_calls: int = 0
    q_rope: dict = field(default_factory=dict)
    attn_cores: dict = field(default_factory=dict)
    r1: np.ndarray = None
    r2: np.ndarray = None
    h2: list = None
    swiglu: np.ndarray = None
    _proj_i: int = 0

    @property
    def hd(self) -> int:
        return int(self.w.head_dim)

    def next_proj(self, B) -> str:
        """Name the projection tensor this GEMM contracts against, and PROVE
        it: the call order inside decoder_layer_fx is fixed, and the operand
        is compared to the layer's own weight tensor."""
        order = ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")
        if self._proj_i >= len(order):
            raise SystemExit(
                f"REFUSE: decoder_layer_fx made more than {len(order)} "
                f"gemm_i8_ksplit calls — the projection call-order map is "
                f"stale (golden changed).")
        lab = order[self._proj_i]
        self._proj_i += 1
        want = np.asarray(getattr(self.w, lab), dtype=np.int64)
        if want.shape != B.shape or not np.array_equal(want, B):
            raise SystemExit(
                f"REFUSE: projection call #{self._proj_i} was expected to be "
                f"{lab}{want.shape} but got a {B.shape} operand that is not "
                f"that tensor — the call-order map is stale (golden changed).")
        return lab


# ═══════════════════════════ the decode (A/B) ══════════════════════════════

class BusOnly:
    """Host-only run with the target layer composed in C-LBUS BUS_ON.

    The diagnostic that separates 'the tile changed the token' from 'the bus
    mode changed the token' — it rebinds decoder_layer_fx and NOTHING else.
    """

    def __init__(self, n_layers: int, layer: int, step: int | None):
        self.n_layers, self.t_layer, self.t_step = n_layers, layer, step
        self.calls = 0
        self.fired = 0

    def __enter__(self):
        self._orig = tf.decoder_layer_fx
        tf.decoder_layer_fx = self._layer
        return self

    def __exit__(self, *exc):
        tf.decoder_layer_fx = self._orig
        return False

    def _layer(self, X, w, tier, *a, **kw):
        layer = self.calls % self.n_layers
        step = self.calls // self.n_layers
        if layer == self.t_layer and (self.t_step is None
                                      or step == self.t_step):
            kw = dict(kw)
            kw["bus"] = tf.BUS_ON
            self.fired += 1
        try:
            return self._orig(X, w, tier, *a, **kw)
        finally:
            self.calls += 1


def run_ab(model, tok, args, work: Path) -> int:
    ids = tok.encode(args.prompt) if tok else list(args.ids)
    n_total = len(ids) + args.max_tokens
    if n_total > TOKENS_MAX:
        eprint(f"REFUSE: prompt({len(ids)}) + max_tokens({args.max_tokens}) = "
               f"{n_total} > {TOKENS_MAX}. Beyond 128 tokens a head becomes a "
               f"ChunkedHead and the C-CHUNK host merge needs per-chunk "
               f"sm_m/sm_l, which the mailbox does not expose "
               f"(apex_f2_mailbox.sv:33-45) — a chunked head CANNOT be "
               f"offloaded (audit N8). Shorten the prompt or --max-tokens.")
        return 2
    L, H = model.n_layers, model.meta["H"]
    if not 0 <= args.layer < L:
        eprint(f"REFUSE: --layer {args.layer} outside [0,{L})")
        return 2
    tier = rt.TIER_MAP[args.tier]
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    mode = "golden" if args.dry_run else args.executor
    scope = Scope(ops=tuple(args.ops), proj_cols=args.proj_cols,
                  proj_rows=args.proj_rows, heads=args.heads,
                  rope_heads=args.rope_heads, resid_cols=args.resid_cols,
                  resid_slice=args.resid_slice, norm_dm=args.norm_dm,
                  norm1_rows=args.norm1_rows,
                  swiglu_chunks=args.swiglu_chunks)
    work.mkdir(parents=True, exist_ok=True)

    eprint(f"[C2] model={model.meta['model']} L={L} H={H} tier={tier} "
           f"G={args.group}; prompt={len(ids)} tok + {args.max_tokens} new "
           f"= {n_total} <= {TOKENS_MAX} OK")
    eprint(f"[C2] target: layer {args.layer}"
           + (f" step {args.offload_step}" if args.offload_step is not None
              else " (every step)")
           + f"  mode={mode}  ops={','.join(scope.ops)}")

    runner = Runner(mode, work, binary=args.binary, tile_div=args.tile_div,
                    slot=args.slot, timeout_s=args.timeout_s,
                    collapse=args.collapse, max_batch_mb=args.max_batch_mb)
    eprint(f"[C2] transport: collapse={args.collapse} "
           f"max_batch_mb={args.max_batch_mb} max_passes={args.max_passes}")
    eprint("[C2] === run 1/3: OFFLOAD ON ===")
    with LayerOffloader(n_layers=L, layer=args.layer, mode=mode, work=work,
                        scope=scope, runner=runner, step=args.offload_step,
                        max_passes=args.max_passes) \
            as off:
        on = po.decode(model, ids, args.max_tokens, tier, args.group, eos, None)
    on["records"] = off.records
    on["checks"] = off.checks
    on["passes"] = dict(off.passes)
    on["invocations"] = dict(off.invocations)

    eprint("[C2] === run 2/3: OFFLOAD OFF (pure golden, default bus) ===")
    assert tf.attention_core is at.attention_core, "rebind not restored"
    assert tf.decoder_layer_fx.__module__.endswith("apex_golden.transformer"), \
        "decoder_layer_fx rebind not restored"
    offr = po.decode(model, ids, args.max_tokens, tier, args.group, eos, None)

    eprint("[C2] === run 3/3: HOST ONLY, target layer in C-LBUS BUS_ON ===")
    with BusOnly(L, args.layer, args.offload_step) as bo:
        bus = po.decode(model, ids, args.max_tokens, tier, args.group, eos,
                        None)
    bus["fired"] = bo.fired

    poison = None
    if args.poison is not None:
        eprint(f"[C2] === discriminator: every tile value x{args.poison} ===")
        with LayerOffloader(n_layers=L, layer=args.layer, mode=mode, work=work,
                            scope=scope, runner=runner, step=args.offload_step,
                            poison=args.poison,
                            max_passes=args.max_passes) as off2:
            poison = po.decode(model, ids, args.max_tokens, tier, args.group,
                               eos, None)
        poison["records"] = off2.records

    return report(model, tok, args, ids, on, offr, bus, poison, mode, scope,
                  runner, work)


# ═══════════════════════════════ the ledger ════════════════════════════════

OP_LABEL = {
    "proj": "projections q/k/v/o/g/u/d",
    "rope": "RoPE (decode-token q)",
    "attn": "attention (score + PV)",
    "resid": "residual (r1, r2)",
    "norm": "RMSNorm-2",
    "swiglu": "SwiGLU",
}
EGRESS = {
    "proj": "RO lanes, raw INT32                 EXACT",
    "rope": "C-1 view (codes + row scale)        EXACT where consumed*",
    "attn": "RO lanes INT32 + captured s_c       EXACT",
    "resid": "LAYER_RDATA fp16 words              EXACT",
    "norm": "C-1 view, 128-wide frames           LOSSY (B-FEED-WIDTH)",
    "swiglu": "C-1 view, 128-wide frames           LOSSY (B-FEED-WIDTH)",
}


def transport_record(runner: Runner, on: dict | None = None) -> dict:
    """The machine-readable half of `transport_lines`."""
    return {
        "collapse": runner.collapse,
        "invocations": runner.n_invocations,
        "programs": runner.n_jobs,
        "distinct_programs": len(runner.executed_keys),
        "audited_programs": len(runner.audited_keys),
        "pool_hits": runner.n_hits, "pool_misses": runner.n_misses,
        "uploaded_bytes": runner.pool_bytes,
        "max_batch_mb": runner.max_batch_bytes // (1024 * 1024),
        "flushes": list(runner.flushes),
        "passes_per_layer": dict((on or {}).get("passes") or {}),
        "invocations_per_layer": dict((on or {}).get("invocations") or {}),
        "executor_s": round(runner.wall, 2),
    }


def transport_lines(runner: Runner, on: dict | None = None) -> list:
    """The transport ledger: how many times an executor was ENTERED, and why.

    Printed on every run, collapse on or off, because the whole point of the
    collapse is a number — and a number that is not printed every run is a
    claim, not a measurement.
    """
    if runner.mode == "golden":
        return ["executor       : none (--dry-run) — no invocation to count"]
    out = [f"invocations    : {runner.n_invocations} entries into the "
           f"executor for {runner.n_jobs} programs "
           f"(collapse {'ON' if runner.collapse else 'OFF'}"
           + (f", <= {runner.max_batch_bytes // (1024 * 1024)} MB per batch"
              if runner.collapse else "") + ")"]
    if runner.collapse:
        passes = (on or {}).get("passes") or {}
        inv = (on or {}).get("invocations") or {}
        if passes:
            out.append("layer replay   : " + ", ".join(
                f"L{k:02d} {v} pass(es) / {inv.get(k, 0)} invocation(s)"
                for k, v in sorted(passes.items())))
        out.append(f"capture pool   : {runner.n_hits} op fetch(es) served "
                   f"from the pool, {runner.n_misses} queued; "
                   f"{len(runner.executed_keys)} distinct programs executed, "
                   f"{runner.pool_bytes / 1e6:.1f} MB of regops uploaded")
        out.append("                 (a pooled capture set is reused ONLY for "
                   "a program whose NAME and BYTES are identical — "
                   "layer_offload.program_key — and every decode, grade and "
                   "consumption check is re-run from those captures on every "
                   "pass)")
        if runner.flushes:
            big = max(f["bytes"] for f in runner.flushes)
            out.append(f"largest batch  : {big / 1e6:.1f} MB in one "
                       f"invocation ({len(runner.flushes)} flushes)")
        out.append("                 READ THE PER-OP 'seconds' BELOW AS THE "
                   "KEPT PASS ONLY: with the")
        out.append("                 collapse an op's executor time is paid "
                   "in the flushes above, not")
        out.append("                 inside the op call. The per-layer "
                   "'exec s' and this run's total")
        out.append("                 executor seconds are the numbers that "
                   "account for all of it.")
    return out


def _agg(records, op):
    rs = [r for r in records if r.op == op]
    served = [r for r in rs if r.n_served > 0]
    return {
        "instances": len(rs), "served_instances": len(served),
        "values": sum(r.n_served for r in served),
        "values_total": sum(r.n_total for r in rs),
        "jobs": sum(r.jobs for r in rs), "caps": sum(r.caps for r in rs),
        "seconds": round(sum(r.seconds for r in rs), 2),
        "grade_ok": (all(r.grade_ok for r in served) if served else None),
        "exact": (all(r.exact for r in served) if served else None),
        "notes": sorted({n for r in rs for n in r.notes}),
        "source": (served[0].source if served else "HOST (golden)"),
        "detail": [r.detail for r in served[:2]],
    }


def report(model, tok, args, ids, on, offr, bus, poison, mode, scope, runner,
           work) -> int:
    recs = on["records"]
    checks = on["checks"]
    ident = on["ids"] == offr["ids"]
    bus_ident = bus["ids"] == offr["ids"]
    fired = any(r.n_served for r in recs)
    agg = {op: _agg(recs, op) for op in OP_TYPES}
    served_ops = [op for op in OP_TYPES if agg[op]["values"] > 0]
    checks_ok = all(ok for _n, ok, _d in checks)
    grades_ok = all(agg[op]["grade_ok"] is not False for op in OP_TYPES)

    def txt(v):
        return tok.decode(v) if tok else str(v)

    W = 78
    print("\n" + "=" * W)
    print("MILESTONE C2 — ONE DECODER LAYER OF A REAL PROMPT, EVERY OP TYPE "
          "ON THE TILE")
    print("=" * W)
    print(f"  prompt          : {args.prompt!r}")
    print(f"  prompt ids      : {ids} ({len(ids)} tokens)")
    print(f"  fence           : {len(ids)}+{args.max_tokens} = "
          f"{len(ids)+args.max_tokens} <= {TOKENS_MAX} tokens  OK "
          f"(chunked heads cannot be offloaded — N8)")
    print(f"  model           : {model.meta['model']}  L={model.n_layers} "
          f"H={model.meta['H']} head_dim={model.meta['head_dim']} "
          f"d_ffn={model.meta.get('d_ffn')}")
    print(f"  offloaded layer : {args.layer}"
          + (f"  step {args.offload_step}" if args.offload_step is not None
             else "  (every step)")
          + f"   composition: C-LBUS BUS_ON (apex_layer_deq.sv:90-92 refuses "
            f"an ungraded composite)")
    print(f"  executor        : {mode}"
          + ("   (--dry-run: every tile call replaced by golden — the seam "
             "only)" if mode == "golden" else ""))
    print(f"  tile jobs       : {runner.n_jobs} programs, {runner.n_caps} "
          f"capture records, {runner.wall:.1f}s in the executor")
    for line in transport_lines(runner, on):
        print(f"  {line}")
    print()
    print("  PER-OP LEDGER — who computed each op type of the offloaded layer")
    print("  " + "-" * (W - 2))
    print(f"  {'op type':<26} {'who':<16} {'served/total values':<21} exact?")
    for op in OP_TYPES:
        a = agg[op]
        who = a["source"] if a["values"] else "HOST (golden)"
        cov = f"{a['values']}/{a['values_total']}"
        ex = ("—" if not a["values"] else
              ("BIT-EXACT" if a["exact"] else "reconstructed"))
        print(f"  {OP_LABEL[op]:<26} {who:<16} {cov:<21} {ex}")
        print(f"      egress      : {EGRESS[op]}")
        if a["values"]:
            print(f"      instances   : {a['served_instances']}/"
                  f"{a['instances']} op calls served, {a['jobs']} tile "
                  f"programs, {a['caps']} caps, {a['seconds']}s")
            print(f"      tile vs golden (same view): "
                  f"{'bit-exact' if a['grade_ok'] else 'DIFFERS'}")
            for d in a["detail"][:1]:
                for k in ("max_abs_delta", "max_abs_delta_q78",
                          "downstream_c1_codes_n_diff",
                          "downstream_c1_scale_equal", "requant_identical",
                          "downstream_codes_match_golden",
                          "reassembled_equal_golden", "per_slice_pass",
                          "chunks", "rows", "slices"):
                    if k in d:
                        print(f"        {k:<32}: {d[k]}")
        for n in a["notes"]:
            print(f"      NOTE: {n}")
    print("  * RoPE: the C-1 reconstruction is re-quantized by golden's very "
          "next stage")
    print("    (attention_core Q7, quant_rows_i8) — measured identical to "
          "golden's own codes,")
    print("    so the substituted row is exact WHERE THE MODEL CONSUMES IT.")
    print()
    print("  SUBSTITUTION / CONSUMPTION CHECKS (the tile's value reached the "
          "model)")
    for name, ok, detail in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  {detail}" if detail and not ok else ""))
    if not checks:
        print("    (none — no op was served)")
    print()
    print(f"  token OFFLOAD ON  : ids={on['ids']} text={txt(on['ids'])!r} "
          f"({on['seconds']:.0f}s)")
    print(f"  token PURE HOST   : ids={offr['ids']} text={txt(offr['ids'])!r} "
          f"({offr['seconds']:.0f}s)")
    print(f"  token HOST+BUS_ON : ids={bus['ids']} text={txt(bus['ids'])!r} "
          f"({bus['seconds']:.0f}s, layer composed BUS_ON {bus['fired']}x) "
          f"-> {'same as pure host' if bus_ident else 'DIFFERS from pure host'}")
    if poison is not None:
        dl = (float(np.max(np.abs(poison["logits"] - on["logits"])))
              if poison["logits"] is not None and on["logits"] is not None
              else float("nan"))
        verdict = ("substitutions ARE load-bearing" if dl > 0 else
                   "VACUOUS — the tile values never reached the token")
        print(f"  discriminator     : tile values x{args.poison} -> "
              f"ids={poison['ids']}, max|dlogit|={dl:.4g} ({verdict})")
    ok = bool(ident and fired and checks_ok and grades_ok)
    print("-" * W)
    if not fired:
        print("  FAIL: no op was served by the tile — token identity proves "
              "nothing.")
    if not checks_ok:
        print("  FAIL: a tile-derived value was not the value the model "
              "consumed (see the checks above).")
    if not grades_ok:
        print("  FAIL: a tile egress differs from golden's own view of the "
              "same op.")
    if not ident and bus_ident:
        print("  NOTE: the emitted token changed, and the BUS_ON-only run did "
              "NOT change it — so the difference came from a tile-served "
              "value, not from the composition mode.")
    if not ident and not bus_ident:
        print("  NOTE: the BUS_ON-only host run ALSO changes the token, so "
              "the composition mode (not the tile) is implicated.")
    print(f"  OP TYPES SERVED BY THE TILE: {len(served_ops)}/6 "
          f"({', '.join(served_ops) or 'none'})")
    print(f"  TOKEN IDENTITY: {'PASS' if ident else 'FAIL'}"
          f"   (offload-on == pure-host)")
    print(f"  MILESTONE C2  : {'PASS' if ok else 'FAIL'}")
    print("=" * W)
    print("  WHAT THE HOST STILL DID (in the offloaded layer and outside it)")
    for line in host_ledger(model, args, agg):
        print(f"    - {line}")
    print("=" * W)

    out = work / "layer_offload_result.json"
    out.write_text(json.dumps({
        "prompt": args.prompt, "prompt_ids": ids, "mode": mode,
        "layer": args.layer, "step": args.offload_step,
        "scope": scope.__dict__,
        "tokens_on": on["ids"], "tokens_host": offr["ids"],
        "tokens_bus_on": bus["ids"],
        "text_on": txt(on["ids"]), "text_host": txt(offr["ids"]),
        "token_identity": ident, "bus_on_token_identity": bus_ident,
        "op_types_served": served_ops, "milestone_c2": ok,
        "ledger": agg,
        "checks": [{"name": n, "ok": o, "detail": str(d)}
                   for n, o, d in checks],
        "ops": [{"op": r.op, "name": r.name, "source": r.source,
                 "n_served": r.n_served, "n_total": r.n_total, "jobs": r.jobs,
                 "caps": r.caps, "seconds": r.seconds, "exact": r.exact,
                 "grade_ok": r.grade_ok, "notes": r.notes}
                for r in recs],
        "seconds": {"on": on["seconds"], "host": offr["seconds"],
                    "bus_on": bus["seconds"], "executor": round(runner.wall, 1)},
        "transport": transport_record(runner, on),
        "tile_jobs": runner.n_jobs, "tile_caps": runner.n_caps,
        "git": rt.git_rev(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if ok else 1


def host_ledger(model, args, agg) -> list:
    """Exactly what stayed on the host. Written from the code, not from hope."""
    out = [
        f"tokenizer, embedding lookup, the lm_head logits and the argmax",
        f"every other decoder layer: {model.n_layers - 1} of "
        f"{model.n_layers} layers ran entirely on the host",
        "all data movement: weights, the KV rows, every operand staged into "
        "the tile and every value read back",
    ]
    n_add = max(agg["norm"]["values"] // D_TILE - 1, 0)
    if agg["norm"]["values"] == 0 or agg["norm"]["instances"] > \
            agg["norm"]["served_instances"]:
        out.append("RMSNorm-1 (the pre-attention norm, T+1 rows) — same unit "
                   "as the served RMSNorm-2; host by default (--norm1-rows)")
    out += [
        "the C-2 requant epilogues (calib_requant + requant_i32_to_i8) of the "
        "attention output, the o-projection and the down-projection — the "
        "standing fence: the tile returns raw INT32, the host calibrates",
        "the attention path's score-dequant composites (s_q*s_k/sqrt(D), "
        "compute_job.py build_script score_phase) are host-computed and "
        "pushed as CSRs; the online softmax itself runs on the tile's ASU",
        "the per-token C-1 quantizations that feed each op (quant_rows_i8) — "
        "the tile's own C-1 feeder is frozen at 128 (B-FEED-WIDTH), so the "
        "whole-row C-1 golden composes with is a host step",
        f"the {n_add} INT32 adds that accumulate the RMSNorm chunk sums "
        f"(golden rmsnorm_fx_wide's own step 2 — the tile exported all "
        f"{n_add + 1} chunk sums and every one is graded)",
    ]
    if agg["rope"]["values"]:
        out.append("the RoPE of the CACHED K rows: the rotated K must reach "
                   "the KVQ codec as exact fp16 bits, and the tile's q_sink "
                   "egress is a C-1 view (B-ROPE-EGRESS) — decode-token q "
                   "rows only")
    if agg["proj"]["values"] < agg["proj"]["values_total"]:
        out.append(f"the projection columns outside --proj-cols "
                   f"{args.proj_cols} / --proj-rows {args.proj_rows}: "
                   f"{agg['proj']['values_total'] - agg['proj']['values']} of "
                   f"{agg['proj']['values_total']} accumulators")
    return out


# ═══════════════════════════ selftest (tiny model) ═════════════════════════

def _tiny(wdir: Path):
    """A tile-legal tiny random model: head_dim 128 (rope_row's frame), a
    D_model and d_ffn that are multiples of 128 with >= 2 norm chunks."""
    return po._tiny_model(wdir, D=256, H=2, H_kv=1, hd=128, dff=256,
                          vocab=97, L=2)


class _FakeRunner(Runner):
    """A Runner whose executor is a pure function of the program's bytes.

    Everything above `_execute` — prepare-once, the byte key, the pool, the
    size-bounded flush, the invocation count — is the REAL code; only the
    tile is replaced, so the transport machinery is tested rather than
    described.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prepared_calls = 0
        self.exec_batches: list = []

    def prepare(self, path: str) -> None:
        self.prepared_calls += 1

    def _execute(self, paths, tag: str) -> list:
        paths = [str(p) for p in paths]
        for p in paths:
            if self._prepared.get(p) is None:
                raise RuntimeError(f"{tag}: {p} reached the executor without "
                                   f"prepare()")
            self.executed_keys.add(self._prepared[p])
        self.exec_batches.append(list(paths))
        self.n_invocations += 1
        self.n_jobs += len(paths)
        out = []
        for p in paths:
            v = int(hashlib.sha256(Path(p).read_bytes()).hexdigest()[:6], 16)
            out.append([{"tag": Path(p).name, "addr": 0, "mask": 0xFFFFFFFF,
                         "value": v, "i": 0}])
        self.n_caps += len(out)
        return out


class _PassProbe(LayerOffloader):
    """The rope -> attention dependency, reduced to two programs.

    Op A's program depends only on the head index; op B's program EMBEDS the
    word A returned — which is exactly why attention cannot be gathered with
    RoPE in one sweep (transformer.py:535-553). Driven through the REAL
    `_layer` replay loop, so what is under test is the fixed point, not a
    reimplementation of it.
    """
    unstable = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.probe_served = 0

    def _rope(self, x, m, theta=None):
        if self._busy("rope") or not self.armed:
            return self._o_rope(x, m, theta)
        with self._own("rope", "f16"):
            gold = self._o_rope(x, m, theta)
        h = self.cur.rope_calls
        self.cur.rope_calls += 1
        pa = self.work / f"probe_A_h{h}.regops.jsonl"
        pa.write_text('{"op":"note","s":"A%d"}\n' % h)
        ca = self.runner.acquire([str(pa)], f"A{h}")
        if ca is None:
            return gold
        va = ca[0][0]["value"]
        pb = self.work / f"probe_B_h{h}.regops.jsonl"
        pb.write_text('{"op":"note","s":"B%d_%d_%d"}\n'
                      % (h, va, self.pass_no if self.unstable else 0))
        cb = self.runner.acquire([str(pb)], f"B{h}")
        if cb is None:
            return gold
        self.probe_served += 1
        return gold


def _selftest_transport(chk, td) -> None:
    """The capture pool, the size-bounded flush and the layer fixed point."""
    work = td / "tp"
    work.mkdir(parents=True, exist_ok=True)

    def prog(name, body):
        p = work / f"{name}.regops.jsonl"
        p.write_text('{"op":"note","s":"%s"}\n' % body)
        return str(p)

    r = _FakeRunner("sim", work, collapse=True)
    p1, p2, p3 = prog("p1", "x" * 10), prog("p2", "y" * 10), prog("p3", "z")
    chk("acquire on an unseen program returns None and queues it",
        r.acquire([p1], "t") is None and len(r.pending) == 1
        and r.n_invocations == 0)
    r.acquire([p2, p3], "t")
    chk("a second miss adds to the SAME queue (no invocation yet)",
        len(r.pending) == 3 and r.n_invocations == 0)
    n_inv = r.flush("t")
    chk("one flush of 3 small programs is ONE executor invocation",
        n_inv == 1 and r.n_invocations == 1 and len(r.pool) == 3,
        f"{n_inv} invocation(s), pool {len(r.pool)}")
    got = r.acquire([p1, p2], "t")
    chk("a pooled program is served with NO further invocation",
        got is not None and len(got) == 2 and r.n_invocations == 1)
    v1 = got[0][0]["value"]
    Path(p1).write_text('{"op":"note","s":"CHANGED"}\n')
    chk("rewriting a program's BYTES makes it a different key — the stale "
        "captures are NOT reused",
        r.acquire([p1], "t") is None and len(r.pending) == 1)
    r.flush("t2")
    chk("the rewritten program executed and returned a different value",
        r.acquire([p1], "t")[0][0]["value"] != v1 and r.n_invocations == 2)
    chk("prepare() ran once per DISTINCT program content, never per read-back",
        r.prepared_calls == 4, f"{r.prepared_calls} prepare() calls for "
                               f"4 distinct program contents")

    # the size budget: 3 x ~4 KB with a 1 KB budget must be 3 invocations
    rb = _FakeRunner("sim", work, collapse=True, max_batch_mb=1)
    rb.max_batch_bytes = 4096
    big = [prog(f"big{i}", "q" * 4000) for i in range(3)]
    for b in big:
        rb.acquire([b], "b")
    chk("the flush splits at --max-batch-mb instead of one giant upload",
        rb.flush("b") == 3 and rb.n_invocations == 3,
        f"{len(rb.exec_batches)} batches of "
        f"{[len(x) for x in rb.exec_batches]}")

    # an unprepared path must never reach an executor
    r2 = _FakeRunner("sim", work, collapse=True)
    try:
        r2._execute([prog("unprep", "u")], "t")
        chk("an unprepared program is REFUSED at the executor", False,
            "no refusal")
    except RuntimeError as e:
        chk("an unprepared program is REFUSED at the executor",
            "prepare()" in str(e))

    # ── the fixed point, through the real _layer replay ────────────────────
    model = _tiny(td / "wp")
    ids = [3, 1, 4, 1, 5]
    tstep, L, H = len(ids) - 1, model.n_layers, model.meta["H"]
    tier = at.TIER_CQ8

    def run_probe(collapse, unstable=False, max_passes=MAX_PASSES_DEFAULT):
        rr = _FakeRunner("golden", work / f"c{int(collapse)}{int(unstable)}",
                         collapse=collapse)
        (work / f"c{int(collapse)}{int(unstable)}").mkdir(parents=True,
                                                          exist_ok=True)
        pr = _PassProbe(n_layers=L, layer=1, mode="golden",
                        work=work / f"c{int(collapse)}{int(unstable)}",
                        scope=Scope(resid_slice=D_TILE), runner=rr,
                        step=tstep, verbose=False, max_passes=max_passes)
        pr.unstable = unstable
        with pr:
            po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
        return pr, rr

    pc, rc = run_probe(True)
    pn, rn = run_probe(False)
    chk("the layer replay reaches a fixed point in 3 passes (A discovered, "
        "then B built on A's value, then all pooled)",
        pc.passes.get(1) == 3, f"passes={pc.passes}")
    n_op = pn.probe_served                   # q heads + the K-row rotation
    chk("collapsing the same job set serves EVERY op with strictly fewer "
        "executor invocations",
        pc.probe_served == n_op >= H and rn.n_invocations == 2 * n_op
        and rc.n_invocations == 2,
        f"served {pc.probe_served}/{n_op} ops; invocations "
        f"{rc.n_invocations} collapsed vs {rn.n_invocations} per-op")
    chk("both transports ran the SAME distinct programs",
        set(rc.pool) and len(rc.executed_keys) == len(rn.executed_keys),
        f"{len(rc.executed_keys)} vs {len(rn.executed_keys)}")
    chk("a discarded planning pass leaves NO records behind",
        len(pc.records) == len(pn.records),
        f"{len(pc.records)} collapsed vs {len(pn.records)} per-op")
    try:
        run_probe(True, unstable=True, max_passes=4)
        chk("a program set that never stabilises is REFUSED", False,
            "no refusal")
    except SystemExit as e:
        chk("a program set that never stabilises is REFUSED",
            "REFUSE" in str(e) and "stable program set" in str(e),
            str(e)[:60] + "…")


def selftest(args) -> int:
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model = _tiny(td / "w")
        ids, work = [3, 1, 4, 1, 5], td / "work"
        tier, L, H = at.TIER_CQ8, model.n_layers, model.meta["H"]
        scope = Scope(resid_slice=D_TILE)
        runner = Runner("golden", work)

        sites = _locate_f16_sites()
        chk("golden's three `_f16` narrowing sites located in transformer.py",
            len(sites) == 3, str(sites))

        # the LAST prefill step: the step whose layer output actually reaches
        # the emitted token (a poison at an earlier step of the LAST layer is
        # provably invisible — Session.step keeps no history of it).
        # (--max-tokens 1, so the logits under comparison are the ones that
        # step produced; with 2+ tokens the returned logits come from a LATER
        # step that the last layer's output never reaches.)
        tstep = len(ids) - 1
        base = po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
        with LayerOffloader(n_layers=L, layer=1, mode="golden", work=work,
                            scope=scope, runner=runner, step=tstep,
                            verbose=False) as off:
            on = po.decode(model, ids, 1, tier, 128, set(), None,
                           verbose=False)
        chk("every golden rebind restored after the context manager",
            tf.attention_core is at.attention_core
            and tf.decoder_layer_fx.__module__.endswith("apex_golden.transformer")
            and tf._f16.__module__.endswith("apex_golden.transformer")
            and tf.gemm_i8_ksplit.__module__.endswith("apex_golden.compute"))
        chk("token identity (dry-run offload vs pure golden)",
            on["ids"] == base["ids"], f"{on['ids']} vs {base['ids']}")
        got = {r.op for r in off.records if r.n_served}
        chk("all six op types were served in the target layer",
            got == set(OP_TYPES), f"served: {sorted(got)}")
        chk("every substitution/consumption check passed",
            all(ok for _n, ok, _d in off.checks),
            f"{sum(1 for _n, ok, _d in off.checks if ok)}/{len(off.checks)}")
        # the EXACT-egress ops must round-trip bit-exactly; the two C-1-view
        # ops must NOT be claimed exact — B-FEED-WIDTH is a measured cost, and
        # a harness that reported them exact would be lying.
        chk("exact-egress ops (projections, attention, residual) are bit-exact",
            all(r.exact for r in off.records
                if r.n_served and r.op in ("proj", "attn", "resid")))
        chk("C-1-view ops (RMSNorm-2, SwiGLU) are reported as reconstructions, "
            "with a measured delta",
            all(r.exact is False and "max_abs_delta" in str(r.detail)
                for r in off.records if r.n_served and r.op in ("norm",
                                                                "swiglu")))
        chk("RoPE's C-1 reconstruction is exact WHERE THE MODEL CONSUMES IT",
            all(r.detail["downstream_codes_match_golden"]
                and r.detail["requant_identical"]
                for r in off.records if r.n_served and r.op == "rope"))
        chk("the offload fired at exactly the targeted (layer, step)",
            all(r.name.startswith(f"lo_s{tstep:03d}_L01") for r in off.records),
            str(sorted({r.name.split('_h')[0] for r in off.records})[:3]))

        # coverage at full width for the five non-GEMM ops
        by = {op: _agg(off.records, op) for op in OP_TYPES}
        chk("residual covers the FULL D_model row, twice",
            by["resid"]["values"] == 2 * model.meta["D_model"],
            f"{by['resid']['values']} of {2 * model.meta['D_model']}")
        chk("RMSNorm-2 covers the FULL row",
            by["norm"]["values"] == model.meta["D_model"])
        chk("SwiGLU covers the FULL d_ffn",
            by["swiglu"]["values"] == model.meta["d_ffn"],
            f"{by['swiglu']['values']} of {model.meta['d_ffn']}")
        chk("attention + RoPE cover ALL heads",
            by["attn"]["served_instances"] == H
            and by["rope"]["served_instances"] == H)

        # the DISCRIMINATOR: if poisoning changes nothing, this is all vacuous
        with LayerOffloader(n_layers=L, layer=1, mode="golden", work=work,
                            scope=scope, runner=runner, step=tstep, poison=0.5,
                            verbose=False):
            po_run = po.decode(model, ids, 1, tier, 128, set(), None,
                               verbose=False)
        dl = float(np.max(np.abs(po_run["logits"] - on["logits"])))
        chk("poisoned tile values perturb the logits (substitutions are "
            "load-bearing)", dl > 0, f"max|dlogit|={dl:.4g}")

        # a wrong call-site map must REFUSE, not silently offload
        with LayerOffloader(n_layers=L, layer=1, mode="golden", work=work,
                            scope=scope, runner=runner, step=tstep,
                            verbose=False) as off3:
            off3.sites = dict(off3.sites)
            off3.sites["r1"] = off3.sites["swiglu"]     # deliberately wrong
            try:
                po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
                chk("a wrong `_f16` call-site map is REFUSED", False,
                    "no refusal")
            except SystemExit as e:
                chk("a wrong `_f16` call-site map is REFUSED",
                    "REFUSE" in str(e), str(e)[:60] + "…")

        # an unreachable target must not silently pass
        with LayerOffloader(n_layers=L, layer=L + 5, mode="golden", work=work,
                            scope=scope, runner=runner, verbose=False) as off4:
            po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
        chk("unreachable target produces zero served ops (report must FAIL)",
            not any(r.n_served for r in off4.records))

        # the T > 128 fence, exercised through the real layer wrapper
        sess = rt.Session(model, tier, 128)
        rows = [tf._f16(v) for v in
                np.random.default_rng(1).normal(0, 1, (129, 256))]
        for li in range(L):
            sess.hist[li] = list(rows)
        sess.pos = 129
        with LayerOffloader(n_layers=L, layer=0, mode="golden", work=work,
                            scope=scope, runner=runner, verbose=False):
            try:
                sess.step(model.embed_row(7))
                chk("T>128 refused at the seam", False, "no refusal")
            except SystemExit as e:
                chk("T>128 refused at the seam", "ChunkedHead" in str(e),
                    str(e)[:60] + "…")

        # the TRANSPORT: capture pool, size-bounded flush, layer fixed point
        _selftest_transport(chk, td)

    print("=" * 60)
    if fails:
        print(f"LAYER_OFFLOAD SELFTEST FAIL: {fails}")
        return 1
    print("LAYER_OFFLOAD SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Milestone C2: serve EVERY op type of one decoder layer "
                    "from the tile during a real prompt run, and prove the "
                    "emitted token is unchanged.")
    ap.add_argument("--prompt")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--offload-step", type=int, default=None,
                    help="offload only at this decode step (default: every)")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--dry-run", action="store_true",
                    help="replace every tile call with golden — exercises the "
                         "whole seam without an executor")
    ap.add_argument("--poison", type=float, default=None, metavar="K",
                    help="extra run with every tile value scaled by K (the "
                         "discriminator: proves the values are load-bearing)")
    ap.add_argument("--ops", default=",".join(OP_TYPES),
                    help=f"comma list of op types to offload "
                         f"({','.join(OP_TYPES)})")
    ap.add_argument("--proj-cols", type=int, default=MXE_N,
                    help="output columns per projection call served by the "
                         "tile (multiple of 8; 0 = none)")
    ap.add_argument("--proj-rows", type=int, default=1,
                    help="activation rows per projection call (K/V projections "
                         "have T rows)")
    ap.add_argument("--heads", type=int, default=-1,
                    help="attention heads to offload (-1 = all)")
    ap.add_argument("--rope-heads", type=int, default=-1,
                    help="q-row RoPE rotations to offload (-1 = all)")
    ap.add_argument("--resid-cols", type=int, default=-1,
                    help="residual elements per row (-1 = the full D_model)")
    ap.add_argument("--resid-slice", type=int, default=D_TILE,
                    help="residual elements per tile job (the narrow image's "
                         "DM_MAX is 128)")
    ap.add_argument("--norm-dm", type=int, default=-1,
                    help="RMSNorm-2 elements (-1 = the full row)")
    ap.add_argument("--norm1-rows", type=int, default=0,
                    help="RMSNorm-1 rows to offload as well (default 0)")
    ap.add_argument("--swiglu-chunks", type=int, default=-1,
                    help="64-column SwiGLU jobs (-1 = the full d_ffn)")
    ap.add_argument("--no-collapse", dest="collapse", action="store_false",
                    help="one executor invocation per op (the pre-collapse "
                         "transport) — the A/B baseline, same job set")
    ap.add_argument("--max-batch-mb", type=int, default=MAX_BATCH_MB_DEFAULT,
                    help="regops per executor invocation when collapsing "
                         "(default: the largest payload already flown)")
    ap.add_argument("--max-passes", type=int, default=MAX_PASSES_DEFAULT,
                    help="replays of a layer allowed before the harness "
                         "REFUSES for lack of a stable program set")
    ap.add_argument("--tier", default="kvq8", choices=list(rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--weights-dir",
                    default=str(REPO / "build/s8_weights/Qwen2.5-7B-4bit"))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--binary", default=None,
                    help="f2sim binary (default: the D=128 silicon twin)")
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="bypass the tokenizer (debug): raw prompt ids")
    ap.add_argument("--make-tiny-weights", metavar="DIR",
                    help="write a TILE-LEGAL tiny RANDOM golden weight cache "
                         "(D=256, head_dim=128, d_ffn=256) and exit — lets "
                         "the whole six-op --executor sim path run in "
                         "minutes. NOT a real model.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import remote_hw_exec                      # hw routing (no-op without env)
    remote_hw_exec.attach(bridge, args)

    if args.selftest:
        return selftest(args)
    if args.make_tiny_weights:
        m = _tiny(Path(args.make_tiny_weights))
        print(f"tile-legal tiny random weights -> {args.make_tiny_weights} "
              f"(L={m.n_layers} H={m.meta['H']} head_dim={m.meta['head_dim']} "
              f"D_model={m.meta['D_model']} d_ffn={m.meta['d_ffn']} "
              f"vocab={m.meta['vocab']}) — NOT a real model")
        return 0
    if args.prompt is None and not args.ids:
        ap.error("need --prompt (or --ids for a tokenizer-free run)")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be >= 1")
    bad = [o for o in args.ops.split(",") if o and o not in OP_TYPES]
    if bad:
        ap.error(f"unknown op type(s) {bad}; choose from {OP_TYPES}")
    args.ops = tuple(o for o in args.ops.split(",") if o)
    work = Path(args.work_dir)
    model = rt.GoldenModel(Path(args.weights_dir))
    tok = None
    if args.ids:
        args.prompt = args.prompt or f"<ids {args.ids}>"
    else:
        tok = rt.load_tokenizer(model.meta["model"])
    return run_ab(model, tok, args, work)


if __name__ == "__main__":
    sys.exit(main())
