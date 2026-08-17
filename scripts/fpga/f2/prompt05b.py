#!/usr/bin/env python3
# prompt05b.py — P2 stage 3: MANY LAYERS of a real Qwen2.5-0.5B prompt served
# by the D=64 tile, with a measured FLOP share.
#
#   python3 scripts/fpga/f2/prompt05b.py --prompt "The capital of France is" \
#       --layers 0-5 --executor sim --poison 0.5
#   python3 scripts/fpga/f2/prompt05b.py --selftest        # host only, seconds
#
# ══ WHAT THIS IS ═══════════════════════════════════════════════════════════
# layer05b.py proved ONE 0.5B decoder layer's six op families on the D=64
# tile (docs/results/p2_b64_cl/RESULTS.md). This file is the same machinery
# over a SET of layers of the same decode step, so the sentence the demo
# wants — "type a question, most of the model's arithmetic runs on the tile,
# and the model's own answer comes back" — has a number attached to it.
#
# Nothing about the seam is re-invented: layer_offload.LayerOffloader is the
# arbiter-safe rebind harness, layer05b.Offloader05B / .Runner05B add the
# D=64 obligations (geometry rebind + per-program INFO_D audit + the
# disclosed INFO_TIER retarget + the re-derived RMSNorm arm), and this file
# adds exactly four things:
#
#   (1) MANY LAYERS — `_layer` re-points the inherited single-layer target at
#       every call, so any subset of layers arms at the same decode step. Per
#       layer the wrappers take their operands from THAT layer's own
#       LayerWeights (golden hands them in), and `_LayerState.next_proj`
#       already REFUSES if a staged operand is not the tensor it claims to
#       be — which is what makes per-layer weight staging checked rather than
#       assumed.
#
#   (2) FULL-WIDTH PROJECTIONS, BATCHED — `--proj-cols -1` (the default) means
#       every accumulator of every projection, and `_gemm` builds all of a
#       projection call's K-split jobs first, then runs them through
#       batch_exec in one invocation per `--batch-size`. Batching is a pure
#       transport change (batch_exec's own capture-for-capture equivalence
#       proof), and `--cross-check` re-runs one block UNBATCHED through the
#       inherited `_tile_matvec` and asserts identical INT32 accumulators.
#
#   (3) A MEASURED FLOP SHARE — not a shape guess. A read-only census
#       (`MacCensus`) wraps golden's own `gemm_i8_ksplit` / `attention_core` /
#       `decoder_layer_fx` during the PURE-HOST arm and counts every
#       multiply-accumulate the step performs, per (step, layer, tensor),
#       plus the lm_head. The numerator counts only accumulators the tile
#       actually produced (`n_served`, not `n_total`), so a sampled
#       projection scope shows up as a smaller share and never as an
#       aspiration. §FLOP SHARE in the ledger prints the whole derivation.
#
#   (3b) TRANSPORT: ONE INVOCATION PER PASS, NOT ONE PER OP — the batching in
#       (2) collapsed the projections to 7 executor entries per layer, but the
#       per-head RoPE, the per-head attention and the two residual windows
#       still cost ONE ENTRY EACH: 30 of the 37 entries a layer-step made, at
#       a fitted ~4.8 s of ssh/attach constant apiece. `layer_offload.Runner`
#       now carries a CAPTURE POOL keyed by the program's own bytes, and
#       `LayerOffloader._layer` REPLAYS the layer until every program it asks
#       for is already in that pool — so a layer costs one flush per replay
#       (measured: 3 replays, 2 invocations) instead of 37 entries. This file
#       adds the one thing the replay would otherwise make expensive: the
#       full-width projections are ~28 s of staging and ~214 MB of regops per
#       layer, so their emission is memoised on an OPERAND digest and the
#       captures a hit returns are re-decoded and RE-GRADED against golden for
#       that pass's own operands. `--no-collapse` restores the old transport
#       for the A/B.
#
#   (4) THE OWNER'S FAMILY POLICY — the default op set is the FOUR
#       EXACT-EGRESS families (proj, rope, attn, resid), for which token
#       identity must hold and is a PASS condition. RMSNorm-2 and SwiGLU
#       re-enter the host through the 64-wide C-1 feeder (B-FEED-WIDTH), and
#       at 0.5B that disclosed reconstruction cost is comparable to the
#       model's top-1 margins (stage 2: 2 of 4 prompts flipped). They are
#       reachable ONLY behind `--include-reconstructed`, which prints a
#       plain warning, drops token identity from the PASS condition and
#       grades the run at the LOGIT level instead.
#
#   (5) THE LM_HEAD — `--lmhead-rows -1` serves the vocabulary projection,
#       the token's LAST GEMM, from the tile. Until now it was the ceiling on
#       the whole claim: with all 24 layers offloaded the tile could still
#       only reach 73.6% of a decode step's arithmetic, because the head is
#       136,134,656 of the token's 516,196,352 MACs (26.4%). It is NOT a new
#       op type — golden's own `head_logits` is already an INT8 GEMM
#       (run_tinynpu.py:277-292), so the tile substitutes raw INT32
#       accumulators exactly as it does for q/k/v/o/g/u/d, at N = 151,936
#       instead of N = 896, and every float in the head (both C-1
#       quantizations, the final RMSNorm, the dequant, the argmax) stays
#       golden's. See LmHeadOffloader for the exactness status in full.
#
# ══ WHAT IS STILL THE HOST'S ═══════════════════════════════════════════════
# Everything the single-layer ledger already listed, plus: the layers outside
# --layers, the embedding, the C-2 requant epilogues, the whole-row C-1
# quantizations, and — unless --lmhead-rows says otherwise — the lm_head. The
# ledger prints it every run, with this run's own numbers.

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
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

import gemm_job as gj                                           # noqa: E402
import gen_layer_ops as gl                                      # noqa: E402
import layer05b as l05                                          # noqa: E402
import layer_offload as lof                                     # noqa: E402
import prompt_offload as po                                     # noqa: E402
import remote_hw_exec as rhe                                    # noqa: E402
import tile_exec_bridge as bridge                               # noqa: E402
import tile_geom as tg                                          # noqa: E402

OP_TYPES = lof.OP_TYPES
EXACT_FAMILIES = ("proj", "rope", "attn", "resid")
RECON_FAMILIES = ("norm", "swiglu")
FULL = 1 << 30                       # "every column / every row" sentinel
DEFAULT_WORK = REPO / "build" / "prompt05b"


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═══════════════════════ the layer set (a real parser) ═════════════════════

def parse_layers(spec: str, n_layers: int) -> tuple:
    """'all' | '0-5' | '0,3,7' | '0-3,7' -> a sorted tuple. Refuses junk and
    out-of-range indices rather than silently offloading nothing."""
    s = str(spec).strip().lower()
    if s in ("all", "*"):
        return tuple(range(n_layers))
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                raise SystemExit(f"REFUSE: --layers: {part!r} is not a range")
            if lo > hi:
                raise SystemExit(f"REFUSE: --layers: empty range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(part))
            except ValueError:
                raise SystemExit(f"REFUSE: --layers: {part!r} is not a layer")
    bad = sorted(v for v in out if not 0 <= v < n_layers)
    if bad:
        raise SystemExit(f"REFUSE: --layers {bad} outside [0,{n_layers})")
    if not out:
        raise SystemExit("REFUSE: --layers selected no layer")
    return tuple(sorted(out))


# ═════════════════ the MAC census (measured, not modelled) ═════════════════

class MacCensus:
    """Count every multiply-accumulate the decode performs, by observing
    golden's own calls. READ-ONLY: every wrapper calls the original and
    returns its value untouched.

    Why this and not a shape formula: the denominator of the FLOP-share line
    is the whole step, including the 20-odd layers the tile never touches and
    the lm_head, and a formula would be a claim about golden's structure
    rather than a measurement of it. Wrapping the same three module globals
    layer_offload rebinds costs nothing and makes the denominator this run's
    own observation.

      gemm_i8_ksplit(A[M,K], B[K,N])  -> M*N*K MACs   (all 7 projections and
                                                       the lm_head)
      attention_core(...)             -> 2*T*D MACs   (Q·K̂ᵀ then P·V̂, both
                                                       gemm_i8 inside golden)
      rmsnorm_fx_wide(x, g)           -> len(x) element-touches (NOT MACs;
                                                       counted separately)

    RE-ENTRANCY IS NOT OPTIONAL HERE. compute.gemm_i8_ksplit splits
    `N > DIM_MAX` (4095) by CONCATENATION and recurses through its own module
    global (compute.py:64-67) — which is the very name this census rebinds.
    Without a depth guard a 0.5B gate/up projection (N = 4864) is counted
    three times (the outer 4864 plus the inner 4095 and 769), the step total
    comes out ~55% too large and the FLOP share too small. Counted once, at
    the outermost call. `verify_against` then re-derives the same shapes from
    the model's own weight tensors and REFUSES on any disagreement, so this
    is checked at run time rather than remembered.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.calls = 0
        self.cur = None                      # (step, layer) while inside one
        self.depth = 0                       # gemm_i8_ksplit recursion depth
        self.gemm: list[dict] = []
        self.attn: list[dict] = []
        self.elem: list[dict] = []
        self.inner_calls = 0                 # recursions skipped (evidence)

    # ── rebind ─────────────────────────────────────────────────────────────
    def __enter__(self):
        self._o_layer = tf.decoder_layer_fx
        self._o_gemm_tf, self._o_gemm_cp = tf.gemm_i8_ksplit, cp.gemm_i8_ksplit
        self._o_attn_tf, self._o_attn_at = tf.attention_core, at.attention_core
        self._o_norm = tf.rmsnorm_fx_wide
        assert self._o_gemm_tf is self._o_gemm_cp, (
            "transformer and compute already resolve different gemm_i8_ksplit "
            "— the census would double count or miss calls")
        assert self._o_attn_tf is self._o_attn_at, \
            "transformer and attention already resolve different attention_core"
        tf.decoder_layer_fx = self._layer
        tf.gemm_i8_ksplit = cp.gemm_i8_ksplit = self._gemm
        tf.attention_core = at.attention_core = self._attn
        tf.rmsnorm_fx_wide = self._norm
        return self

    def __exit__(self, *exc):
        tf.decoder_layer_fx = self._o_layer
        tf.gemm_i8_ksplit, cp.gemm_i8_ksplit = self._o_gemm_tf, self._o_gemm_cp
        tf.attention_core, at.attention_core = self._o_attn_tf, self._o_attn_at
        tf.rmsnorm_fx_wide = self._o_norm
        return False

    # ── wrappers ───────────────────────────────────────────────────────────
    def _layer(self, X, w, tier, *a, **kw):
        step, layer = self.calls // self.n_layers, self.calls % self.n_layers
        prev, self.cur = self.cur, (step, layer)
        try:
            return self._o_layer(X, w, tier, *a, **kw)
        finally:
            self.cur = prev
            self.calls += 1

    def _gemm(self, A, B, *a, **kw):
        A = np.asarray(A)
        B = np.asarray(B)
        if self.depth == 0:
            self.gemm.append({"where": self.cur, "M": int(A.shape[0]),
                              "K": int(A.shape[1]), "N": int(B.shape[1]),
                              "macs": int(A.shape[0]) * int(A.shape[1])
                              * int(B.shape[1])})
        else:
            self.inner_calls += 1
        self.depth += 1
        try:
            return self._o_gemm_tf(A, B, *a, **kw)
        finally:
            self.depth -= 1

    def _attn(self, *a, **kw):
        core = self._o_attn_at(*a, **kw)
        self.attn.append({"where": self.cur, "T": int(core.T), "D": int(core.D),
                          "macs": 2 * int(core.T) * int(core.D)})
        return core

    def _norm(self, x, g, chunk: int = 128):
        self.elem.append({"where": self.cur, "kind": "rmsnorm",
                          "n": int(np.asarray(x).size)})
        return self._o_norm(x, g, chunk)

    # ── queries ────────────────────────────────────────────────────────────
    def step_macs(self, step: int) -> dict:
        """Every MAC of one decode step, split the way the ledger prints it."""
        per_layer, per_tensor = {}, {}
        order = ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")
        for r in self.gemm:
            if r["where"] is None or r["where"][0] != step:
                continue
            li = r["where"][1]
            per_layer.setdefault(li, {"proj": 0, "attn": 0, "calls": 0})
            per_layer[li]["proj"] += r["macs"]
            i = per_layer[li]["calls"]
            per_layer[li]["calls"] = i + 1
            if i < len(order):
                t = per_tensor.setdefault(order[i], {"macs": 0, "M": r["M"],
                                                     "K": r["K"], "N": r["N"],
                                                     "layers": 0})
                t["macs"] += r["macs"]
                t["layers"] += 1
        for r in self.attn:
            if r["where"] is None or r["where"][0] != step:
                continue
            li = r["where"][1]
            per_layer.setdefault(li, {"proj": 0, "attn": 0, "calls": 0})
            per_layer[li]["attn"] += r["macs"]
        elem = sum(r["n"] for r in self.elem
                   if r["where"] is not None and r["where"][0] == step)
        return {
            "per_layer": per_layer, "per_tensor": per_tensor,
            "proj": sum(v["proj"] for v in per_layer.values()),
            "attn": sum(v["attn"] for v in per_layer.values()),
            "layers_seen": len(per_layer),
            "elem_touches_norm": elem,
        }

    @property
    def outside_layer_macs(self) -> int:
        """The lm_head (and anything else golden GEMMs outside a layer)."""
        return sum(r["macs"] for r in self.gemm if r["where"] is None)

    def verify_against(self, model, step: int) -> dict:
        """Re-derive the denominator from the MODEL's own tensors and REFUSE
        on any disagreement.

        The census is an observation of golden's call stream; this is the
        independent second opinion on the same number. Each layer must have
        made exactly 7 outermost ksplit calls, in the order layer_offload's
        `next_proj` already enforces, and each call's (K, N) must be the
        shape of the weight tensor it claims. The lm_head must come out at
        vocab x D_model. A recursion leak, a missed layer, or a golden
        change in call order all land here as a refusal, not as a number.
        """
        order = ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")
        seen = {}
        for r in self.gemm:
            if r["where"] is None or r["where"][0] != step:
                continue
            seen.setdefault(r["where"][1], []).append(r)
        if sorted(seen) != list(range(model.n_layers)):
            raise SystemExit(
                f"REFUSE: the MAC census saw layers {sorted(seen)} at step "
                f"{step}, not all {model.n_layers}. The denominator would not "
                f"be the whole step.")
        want_total = 0
        for li, rows in seen.items():
            if len(rows) != len(order):
                raise SystemExit(
                    f"REFUSE: layer {li} made {len(rows)} outermost "
                    f"gemm_i8_ksplit calls, expected {len(order)} "
                    f"({', '.join(order)}). Either golden's projection call "
                    f"order changed or the census counted a recursion.")
            w = model.layers[li]
            for tag, r in zip(order, rows):
                sh = tuple(np.asarray(getattr(w, tag)).shape)
                if (r["K"], r["N"]) != sh:
                    raise SystemExit(
                        f"REFUSE: layer {li} call #{order.index(tag)} was "
                        f"counted as K={r['K']} N={r['N']} but {tag} is "
                        f"{sh}. The census is not measuring what it names.")
                want_total += r["M"] * sh[0] * sh[1]
        got = self.step_macs(step)["proj"]
        if got != want_total:
            raise SystemExit(
                f"REFUSE: census projection total {got} != the model's own "
                f"shapes {want_total} for step {step}.")
        vocab = int(np.asarray(model.head_w8).shape[0])
        dm = int(model.meta["D_model"])
        if self.outside_layer_macs != vocab * dm:
            raise SystemExit(
                f"REFUSE: lm_head counted {self.outside_layer_macs} MACs, the "
                f"model's head is {vocab}x{dm} = {vocab * dm}.")
        return {"layers": len(seen), "proj_macs": want_total,
                "lm_head_macs": vocab * dm,
                "recursions_skipped": self.inner_calls}


# ══════════════════════════ the multi-layer seam ═══════════════════════════

class MultiOffloader05B(l05.Offloader05B):
    """layer05b's offloader, armed for a SET of layers at one decode step.

    The inherited `_layer` arms on `self.t_layer`; this one re-points that
    target at every call from the layer the bookkeeping says is in flight, so
    the whole inherited body (the BUS_ON composition, the T fence, the
    per-layer `_LayerState`, the post-hoc substitution checks) runs unchanged
    for every selected layer and is skipped for the rest.
    """

    def __init__(self, *a, layers: tuple, batch_size: int = 256,
                 poison_layers: tuple = (), cross_check: bool = False,
                 prune: bool = True, fat: bool = True,
                 blocks_per_program: int = 128, burst: bool = True,
                 rows_per_desc: int = 1, **kw):
        super().__init__(*a, layer=layers[0], **kw)
        self.t_layers = tuple(sorted(set(layers)))
        self.batch_size = max(1, int(batch_size))
        self.poison_all = self.poison           # what the CLI asked for
        self.poison_layers = (tuple(sorted(set(poison_layers)))
                              if poison_layers else self.t_layers)
        self.cross_check = bool(cross_check)
        self.prune = bool(prune)
        # ── the transport shape (see _gemm) ────────────────────────────────
        self.fat = bool(fat)                    # stage once, sweep N blocks
        self.blocks_per_program = max(1, int(blocks_per_program))
        self.burst = bool(burst)                # poll-empty push bursts
        self.rows_per_desc = max(1, int(rows_per_desc))
        self.by_layer: dict = {}                # layer -> per-layer summary
        self.cross_checks: list = []
        self.cost = {"ops": 0, "peeks": 0, "pokes": 0, "bytes": 0,
                     "programs": 0, "blocks": 0}
        self._pruned_bytes = 0
        self._kept_evidence = False

    # ── arm per layer, and account per layer ───────────────────────────────
    def _layer(self, X, w, tier, *a, **kw):
        li = self.layer
        self.t_layer = li if li in self.t_layers else -1
        # poison is per-layer: the discriminator can bite ONE layer of a
        # multi-layer run without re-running the whole thing.
        self.poison = (self.poison_all
                       if (self.poison_all is not None
                           and li in self.poison_layers) else None)
        # A selected layer is only ARMED at the target step (the inherited
        # `_layer` tests both), so only then is there anything to account.
        # Accounting on every step would overwrite the armed step's numbers
        # with zeros for any --offload-step that is not the last one.
        armed = self.t_layer == li and (self.t_step is None
                                        or self.step == self.t_step)
        if not armed:
            return super()._layer(X, w, tier, *a, **kw)
        r0, c0 = len(self.records), len(self.checks)
        j0, p0, w0 = self.runner.n_jobs, self.runner.n_caps, self.runner.wall
        i0 = self.runner.n_invocations
        t0 = time.perf_counter()
        try:
            return super()._layer(X, w, tier, *a, **kw)
        finally:
            recs = self.records[r0:]
            chks = self.checks[c0:]
            fam = {}
            for op in OP_TYPES:
                ag = lof._agg(recs, op)
                fam[op] = {"served": ag["values"], "total": ag["values_total"],
                           "instances": ag["served_instances"],
                           "jobs": ag["jobs"], "caps": ag["caps"],
                           "seconds": ag["seconds"], "exact": ag["exact"],
                           "grade_ok": ag["grade_ok"]}
            self.by_layer[li] = {
                "layer": li,
                "records": recs,
                "checks": chks,
                "family": fam,
                "checks_pass": sum(1 for _n, ok, _d in chks if ok),
                "checks_total": len(chks),
                "jobs": self.runner.n_jobs - j0,
                "caps": self.runner.n_caps - p0,
                "executor_s": round(self.runner.wall - w0, 2),
                "wall_s": round(time.perf_counter() - t0, 2),
                "invocations": self.runner.n_invocations - i0,
                "passes": self.passes.get(li, 1),
                "poisoned": li in self.poison_layers
                            and self.poison_all is not None,
                "families": sorted({r.op for r in recs if r.n_served}),
            }
            if self.verbose:
                b = self.by_layer[li]
                eprint(f"[P2-C] layer {li:2d} done: {b['jobs']} jobs, "
                       f"{b['caps']} caps, {b['invocations']} executor "
                       f"invocation(s) over {b['passes']} pass(es), executor "
                       f"{b['executor_s']}s, wall {b['wall_s']}s, families "
                       f"{b['families']}, checks "
                       f"{b['checks_pass']}/{b['checks_total']}")

    # ── projections: FAT programs, then batch ──────────────────────────────
    def _plan_programs(self, A, B, c, label, nrow, ncol):
        """Emit this projection call's tile programs. -> (programs, notes)

        A `program` is {'path', 'units': [{'m','c0','n'}…], 'blocks': n}.
        THIN (the 2026-08-03 shape): one program per (m, block, K-chunk) —
        7,742 BAR0 ops each, of which ~62% is re-staging the SAME activation
        row that the previous 111 programs already staged.
        FAT: one program per (m, K-chunk, group of <= blocks_per_program
        blocks) — the staging is paid ONCE and every block after it costs
        only its own weight beats. Same operands, same descriptors, same
        gradings; strictly fewer BAR0 ops and strictly fewer files.
        """
        programs, notes = [], []
        for m in range(nrow):
            x8 = A[m]
            amax = int(np.max(np.abs(x8), initial=0))
            assert amax <= 127, (
                f"activation code magnitude {amax} > 127 — the INT8-symmetric "
                f"staging bound (a -128 code cannot be an amax anchor)")
            # the +127 sentinel lane, if this row's own amax is not 127
            pad = amax != 127
            xs = np.concatenate([x8, [127]]) if pad else x8
            zrow = np.zeros((1, lof.MXE_N), dtype=np.int64)

            def block_w(c0, n):
                W = B[:, c0:c0 + n]
                return np.concatenate([W, zrow[:, :n]]) if pad else W

            # ONE staging plan per activation row: the lane permutation is a
            # function of the activation alone (Plan.restage re-asserts F3
            # per block against that block's own reference product).
            try:
                plan = gj.stage_plan(xs, block_w(0, min(lof.MXE_N, ncol)))
            except (AssertionError, gl.InjectRangeError) as e:
                notes.append(f"m{m}: staging REFUSED: {e}")
                continue
            staged, skipped = [], []
            for c0 in range(0, ncol, lof.MXE_N):
                n = min(lof.MXE_N, ncol - c0)
                try:
                    staged.append((c0, n, plan.restage(block_w(c0, n))))
                except (AssertionError, gl.InjectRangeError) as e:
                    skipped.append(c0)
                    notes.append(f"m{m} c{c0}: staging REFUSED: {e}")
            if not staged:
                continue
            # ONE knob for both shapes, so an A/B differs only in the thing
            # being measured. The 2026-08-03 flown shape is exactly
            # `--no-fat --no-burst --rows-per-desc 16`.
            chunks = plan.chunks(self.rows_per_desc)
            for ci, (r0, nr, K) in enumerate(chunks):
                off = r0 * lof.D_TILE
                xst = plan.xst[off:off + K]
                if not self.fat:
                    for (c0, n, Wst) in staged:
                        nm = f"{c.tag}_{label}_m{m}_c{c0}_k{ci}"
                        p, _ = gj.build_gemm_job_full(Wst[off:off + K], xst,
                                                      self.work, nm)
                        programs.append({"path": str(p), "blocks": 1,
                                         "units": [{"m": m, "c0": c0, "n": n}]})
                    continue
                for gi in range(0, len(staged), self.blocks_per_program):
                    grp = staged[gi:gi + self.blocks_per_program]
                    nm = (f"{c.tag}_{label}_m{m}_k{ci}"
                          f"_g{gi // self.blocks_per_program:03d}")
                    p, man = gj.build_gemm_multiblock_job(
                        [Wst[off:off + K] for (_c0, _n, Wst) in grp], xst,
                        self.work, nm, burst=self.burst,
                        allow_multirow=self.rows_per_desc > 1 or K <= lof.D_TILE)
                    programs.append({
                        "path": str(p), "blocks": len(grp),
                        "units": [{"m": m, "c0": c0, "n": n}
                                  for (c0, n, _W) in grp]})
        return programs, notes

    def _gemm(self, A, B, *a, **kw):
        """Full-width projections without one executor invocation per block.

        The inherited `_gemm` runs `_tile_matvec` per 8-column block, and
        each of those is its own executor call. A 0.5B layer-step is 13,696
        accumulators = 1,712 blocks = 2,160 K-split jobs, so the per-call
        constant is paid 2,160 times. This override keeps the SAME staging
        (gemm_job.stage_plan, sentinel included) and the SAME grading, and
        changes only the TRANSPORT: how many programs the same descriptors
        are packed into, when the executor is entered, and how many BAR0
        PEEKS the pushes cost. batch_exec proves the merged capture stream is
        attributed per file, and `--cross-check` re-runs a block through the
        inherited unbatched THIN path and asserts identical INT32
        accumulators — the fat-vs-thin equivalence, measured per run.
        """
        if self._busy("gemm") or not self.armed:
            return self._o_gemm(A, B, *a, **kw)
        if self.mode == "golden":
            return super()._gemm(A, B, *a, **kw)      # no executor at all
        with self._own("gemm"):
            gold = self._o_gemm(A, B, *a, **kw)
        c = self.cur
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        label = c.next_proj(B)                        # REFUSES a wrong tensor
        rec = lof.OpRec(op="proj", name=f"{c.tag}_{label}",
                        source="HOST (golden)",
                        n_total=int(A.shape[0] * B.shape[1]))
        rec.detail = {"A": list(A.shape), "B": list(B.shape),
                      "K": int(A.shape[1]), "rows_served": 0, "cols_served": 0}
        if not self._serving("proj") or self.scope.proj_cols <= 0:
            rec.notes.append("not in --ops / --proj-cols 0")
            self.records.append(rec)
            c.acc[label] = np.asarray(gold, dtype=np.int64)
            return gold
        ncol = min(self.scope.proj_cols, B.shape[1])
        ncol -= ncol % lof.MXE_N
        nrow = min(self.scope.proj_rows, A.shape[0])
        out = np.asarray(gold, dtype=np.int64).copy()
        t0 = time.perf_counter()

        # ── phase 1: stage + emit this call's tile programs ────────────────
        # Emission is the expensive half here — ~28 s of staging and ~214 MB
        # of regops per layer-step (measured) — so a REPLAYED pass must not
        # pay it again. The memo key is a digest of every value
        # `_plan_programs` is a pure function of: the exact operand bytes,
        # their shapes and dtypes, and every transport/geometry knob it
        # reads. A hit skips emission ONLY; the captures it returns are then
        # decoded, spliced and RE-GRADED against golden's own accumulators
        # for THIS pass's operands (phase 3), so a stale or mis-keyed entry
        # shows up as `bit-exact=False` and fails the run rather than
        # passing quietly.
        memo_fp = lof.operand_key(
            A, B, label, c.tag, nrow, ncol, self.fat, self.burst,
            self.rows_per_desc, self.blocks_per_program, self.batch_size,
            lof.D_TILE, lof.MXE_N, gj.D_TILE, gl.fmt.SWG_COLS_MAX)
        hit = self.runner.memo_lookup(memo_fp) if self.runner.collapse else None
        if hit is not None:
            capsets, meta = hit
            programs, cost = meta["programs"], meta["cost"]
            n_blocks = meta["blocks"]
            rec.notes.extend(meta["notes"])
            rec.detail["emission"] = "operand memo (not re-emitted)"
            caps_by_path = dict(zip([p["path"] for p in programs], capsets))
        else:
            programs, notes = self._plan_programs(A, B, c, label, nrow, ncol)
            rec.notes.extend(notes)
            if not programs:
                rec.seconds = round(time.perf_counter() - t0, 2)
                self.records.append(rec)
                c.acc[label] = out
                return out.astype(np.int32)
            paths = [p["path"] for p in programs]
            cost = {"ops": 0, "peeks": 0, "pokes": 0, "bytes": 0}
            for p in paths:
                k = gj.program_cost(p)
                for key in cost:
                    cost[key] += k[key]
            n_blocks = sum(p["blocks"] for p in programs)
            for key in cost:
                self.cost[key] += cost[key]
            self.cost["programs"] += len(programs)
            self.cost["blocks"] += n_blocks
            rec.detail["emission"] = "emitted"

            # ── phase 2: run (audit + retarget happen in Runner.prepare) ──
            if self.runner.collapse:
                if not self._kept_evidence:
                    self._kept_evidence = True
                    self.runner.keep_paths.update(paths)
                    eprint(f"[P2-C] keeping {rec.name}'s {len(paths)} job "
                           f"files as evidence (later calls are reclaimed "
                           f"after their flush; --keep-jobs disables it)")
                capsets = self.runner.acquire(paths, rec.name)
                if self.prune:
                    self.runner.mark_prunable(paths)
                self.runner.memo_store(memo_fp, paths,
                                       {"programs": programs, "cost": cost,
                                        "blocks": n_blocks, "notes": notes})
                if capsets is None:
                    # queued for this pass's flush: the accumulators stay
                    # golden's for this (discarded) pass.
                    rec.notes.append(f"{len(paths)} program(s) queued for "
                                     f"this pass's flush")
                    rec.seconds = round(time.perf_counter() - t0, 2)
                    self.records.append(rec)
                    c.acc[label] = out
                    return out.astype(np.int32)
                caps_by_path = dict(zip(paths, capsets))
            else:
                caps_by_path = {}
                for i in range(0, len(paths), self.batch_size):
                    chunk = paths[i:i + self.batch_size]
                    capsets = self.runner.run(
                        chunk, f"{rec.name}_b{i // self.batch_size:03d}")
                    caps_by_path.update(dict(zip(chunk, capsets)))
        paths = [p["path"] for p in programs]

        # ── phase 3: decode, splice, grade ────────────────────────────────
        # partials[(m, c0)] accumulate across this call's K chunks; a fat
        # program hands back one m=1 RO beat per block, IN BLOCK ORDER, and
        # decode_multiblock refuses any other beat count rather than letting
        # a missing beat shift every later block's value.
        partials: dict = {}
        widths: dict = {}
        for prog in programs:
            caps = caps_by_path[prog["path"]]
            rec.caps += len(caps)
            accs = (gj.decode_multiblock(caps, prog["blocks"])
                    if prog["blocks"] > 1 or self.fat
                    else [gj.decode_acc(caps)])
            for u, acc in zip(prog["units"], accs):
                key = (u["m"], u["c0"])
                partials.setdefault(key, []).append(np.asarray(acc,
                                                               dtype=np.int64))
                widths[key] = u["n"]
        served, eq = 0, True
        for (m, c0), parts in sorted(partials.items()):
            n = widths[(m, c0)]
            acc = np.asarray(gj.accumulate_partials(parts), dtype=np.int64)
            out[m, c0:c0 + n] = acc[:n]
            served += n
            eq = eq and bool(np.array_equal(
                acc[:n], np.asarray(gold)[m, c0:c0 + n]))
        rec.jobs = len(paths)
        rec.n_served = served
        rec.seconds = round(time.perf_counter() - t0, 2)
        rec.source = f"TILE ({self.mode})" if served else "HOST (golden)"
        rec.grade_ok = eq if served else None
        rec.exact = eq if served else None
        rec.detail.update({"rows_served": nrow if served else 0,
                           "cols_served": ncol, "blocks": n_blocks,
                           "batched": True, "batch_size": self.batch_size,
                           "fat": self.fat, "burst": self.burst,
                           "rows_per_desc": self.rows_per_desc,
                           "programs": len(programs),
                           "blocks_per_program": (self.blocks_per_program
                                                  if self.fat else 1),
                           "bar0_ops": cost["ops"], "bar0_peeks": cost["peeks"],
                           "program_bytes": cost["bytes"]})

        # ── FAT vs THIN: the same block, through the OLD path, must match ──
        # `_tile_matvec` is layer_offload's inherited one-block-per-program
        # path (ROWS_PER_DESC K-split, per-push peeks, its own staging). If a
        # fat program's accumulators were not the thin path's accumulators,
        # every speed number below would be worthless — so one block of every
        # armed call is re-run the old way and compared, INT32 for INT32.
        if self.cross_check and programs:
            u = programs[0]["units"][0]
            nm = f"{c.tag}_{label}_m{u['m']}_c{u['c0']}"
            got = self._tile_matvec(
                B[:, u["c0"]:u["c0"] + u["n"]], A[u["m"]], f"{nm}_xcheck")
            if got is None:
                # the thin reference is queued for this pass's flush; it is
                # graded on the pass that serves it, never skipped.
                rec.notes.append("fat-vs-thin cross-check queued for this "
                                 "pass's flush")
            else:
                ref, nj, _nc = got
                same = bool(np.array_equal(
                    np.asarray(ref)[:u["n"]],
                    out[u["m"], u["c0"]:u["c0"] + u["n"]]))
                self.cross_checks.append({"name": nm, "equal": same,
                                          "jobs": nj, "fat": self.fat,
                                          "burst": self.burst,
                                          "rows_per_desc": self.rows_per_desc})
                self.chk(f"fat-vs-thin cross-check {nm}: fat program acc == "
                         f"UNBATCHED thin _tile_matvec acc", same)
                rec.detail["cross_checked"] = same

        self.records.append(rec)
        c.acc[label] = out
        c.proj_served[label] = served
        if self.prune and not self.runner.collapse:
            self._prune(paths, rec.name)
        self.say(f"[P2-C] L{c.layer:02d} proj {label}: {served}/"
                 f"{rec.n_total} accumulators from the tile in "
                 f"{len(paths)} programs / {n_blocks} blocks "
                 f"({cost['ops']:,} BAR0 ops, {cost['peeks']:,} peeks), "
                 f"bit-exact={eq} ({rec.seconds}s)")
        return out.astype(np.int32)

    def _prune(self, paths, tag: str) -> None:
        """Delete this call's emitted job files once they are graded.

        A full-width layer-step emits ~2,160 regops files of ~240 KB (~520 MB
        per layer); a many-layer run would fill the disk. The FIRST projection
        call of the run is kept as evidence, and every job's captures live on
        in the per-batch cap files either way.
        """
        if not self._kept_evidence:
            self._kept_evidence = True
            eprint(f"[P2-C] keeping {tag}'s {len(paths)} job files as evidence"
                   f" (later calls are pruned; --keep-jobs disables pruning)")
            return
        for p in paths:
            for q in (Path(p), Path(str(p).replace(".regops.jsonl",
                                                   ".manifest.json"))):
                try:
                    self._pruned_bytes += q.stat().st_size
                    q.unlink()
                except OSError:
                    pass
        for d in self.work.glob(f"{tag}_b*_batch"):
            shutil.rmtree(d, ignore_errors=True)


class BusOnlyMulti:
    """lof.BusOnly for a SET of layers — the diagnostic that separates 'the
    tile changed the token' from 'the C-LBUS composition changed the token'.
    Rebinds decoder_layer_fx and nothing else."""

    def __init__(self, n_layers: int, layers: tuple, step: int | None):
        self.n_layers, self.t_layers, self.t_step = n_layers, set(layers), step
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
        layer, step = self.calls % self.n_layers, self.calls // self.n_layers
        if layer in self.t_layers and (self.t_step is None
                                       or step == self.t_step):
            kw = dict(kw)
            kw["bus"] = tf.BUS_ON
            self.fired += 1
        try:
            return self._orig(X, w, tier, *a, **kw)
        finally:
            self.calls += 1


# ═══════════════ the lm_head: the last GEMM of the whole token ═════════════

class LmHeadOffloader:
    """Serve golden's OWN lm_head GEMM — the vocabulary projection — from the
    tile, so the token's last 136 M MACs stop being the host's.

    ══ THE EXACTNESS STATUS, ESTABLISHED BY READING GOLDEN, NOT BY WISHING ══
    Nothing here invents a quantization. golden's `GoldenModel.head_logits`
    (run_tinynpu.py:277-292) is ALREADY an integer kernel:

        x8, _  = at.quant_rows_i8(y)                 # C-1, the model's own
        h,_,_  = at.rmsnorm_fx_wide(x8, final_gamma) # the final norm
        h8,s_h = at.quant_rows_i8(h / 256.0)         # C-1 again
        for a in range(0, vocab, DIM_MAX):           # 12-bit M field
            acc = cp.gemm_i8_ksplit(head_w8[a:b], h8[:, None])   # INT8 x INT8
        return acc * (s_h * s_head * final_fold)     # host dequant

    So the lm_head is the SAME op the tile has been proving all along: an
    INT8 x INT8 GEMM whose raw INT32 accumulators the host then scales. The
    operands are golden's own — `head_w8` as prepared (tie_word_embeddings:
    the head IS the embedding, quantized per tensor at prepare time) and the
    activation `h8` golden itself derived. This class substitutes the tile's
    INT32 accumulators for golden's, and leaves EVERY float in golden's
    hands: the two C-1 quantizations, the final RMSNorm, the dequant
    multiply and the argmax all run untouched on the host.

    THE CLAIM IS THEREFORE: **BIT-EXACT INT32 ACCUMULATORS**, graded per call
    against `cp.gemm_i8_ksplit`'s own output on the same operands, with the
    logits and the token then formed by golden's untouched epilogue. It is
    the identical claim the q/k/v/o/g/u/d projections carry, at a bigger N.
    It is NOT a claim that the tile computed the logits in float, and it is
    NOT a re-quantization of anything.

    ══ THE SHAPE ══════════════════════════════════════════════════════════
    Golden's call is `A[M, 896] x B[896, 1]` — M vocabulary rows against ONE
    activation column. The tile's primitive is `x8[K] x W[K, <=8]`, so the
    activation is `B[:, 0]` and each 8-column block is EIGHT VOCABULARY ROWS
    transposed: `W[k, j] = A[v0 + j, k]`. At 0.5B that is 151,936 rows =
    18,997 blocks (golden's own 4095-row chunking leaves a ragged tail block
    per call), K = 896 -> 15 staged 64-wide rows -> k = 960, ONE descriptor
    per block. Structurally identical to the Wq/Wo programs the ledger above
    already flies; only the block count is bigger.

    ══ THREE PASSES OVER THE SAME HEAD CALL ═══════════════════════════════
    `head_logits` is a LEAF: its output feeds the argmax and nothing else. So
    the A/B is run at the head itself rather than by re-decoding the model:
      pass 1  serving OFF -> golden's logits. THE ARBITER.
      pass 2  serving ON  -> the tile's logits. This is the array returned to
              the model, and the one `int(np.argmax(...))` consumes.
      pass 3  the discriminator (--lmhead-poison K), served from pass 2's own
              decoded accumulators — no second entry into the executor — with
              ONE accumulator scaled by K. A UNIFORM scale cannot flip an
              argmax (x -> Kx with K > 0 is order preserving), so a uniform
              poison would be a vacuous discriminator here; the one that
              actually discriminates targets the row the argmax picks, and
              the ledger names the row it touched.
    """

    LABEL = "lm_head (vocab projection)"
    EGRESS = "RO lanes, raw INT32                  EXACT"

    def __init__(self, model, *, mode: str, work: Path, runner, rows: int = 0,
                 poison: float | None = None, blocks_per_program: int = 128,
                 burst: bool = True, rows_per_desc: int = 1,
                 prune: bool = True, verbose: bool = True):
        self.model = model
        self.mode, self.work, self.runner = mode, Path(work), runner
        self.dm = int(model.meta["D_model"])
        self.vocab = int(np.asarray(model.head_w8).shape[0])
        self.rows = self.vocab if rows is None or rows < 0 else int(rows)
        self.poison = poison
        self.blocks_per_program = max(1, int(blocks_per_program))
        self.burst = bool(burst)
        self.rows_per_desc = max(1, int(rows_per_desc))
        self.prune, self.verbose = bool(prune), bool(verbose)
        # ── state ──────────────────────────────────────────────────────────
        self.armed = self.rows > 0
        self.records: list = []
        self.checks: list = []
        self.cost = {"ops": 0, "peeks": 0, "pokes": 0, "bytes": 0,
                     "programs": 0, "blocks": 0}
        self.served_rows = 0
        self._rows_this_pass = 0             # the row budget of ONE pass
        self.n_calls = 0                     # head_logits calls seen
        self.n_invocations = 0               # entries into the executor
        self.n_gemm_calls = 0                # the ksplit calls INSIDE one head
        self.result: dict | None = None      # the A/B, filled by _head
        self._serve_on = False
        self._recording = True
        self._depth = 0
        self._cursor = 0                     # next vocabulary row expected
        self._plan_cache: tuple | None = None
        self._acc_memo: dict = {}
        self._poison_row = None
        self._kept_evidence = False
        self._pruned_bytes = 0
        self._seconds = 0.0                  # pass 2 (the tile's) wall
        self._gold_seconds = 0.0             # pass 1 (golden's own) wall

    # ── rebinding (context manager; ALWAYS restored) ───────────────────────
    def __enter__(self):
        off = self
        self._o_head = rt.GoldenModel.head_logits
        self._o_gemm = cp.gemm_i8_ksplit

        def _head(model_self, y):                    # a real descriptor, so
            return off._head(model_self, y)          # `self` still arrives

        def _gemm(A, B, *a, **kw):
            return off._gemm(A, B, *a, **kw)

        rt.GoldenModel.head_logits = _head
        cp.gemm_i8_ksplit = _gemm
        return self

    def __exit__(self, *exc):
        rt.GoldenModel.head_logits = self._o_head
        cp.gemm_i8_ksplit = self._o_gemm
        return False

    def chk(self, name: str, ok: bool, detail=None):
        self.checks.append((name, bool(ok), detail))
        if not ok:
            eprint(f"[lm_head] CHECK FAILED: {name}: {detail}")

    def say(self, msg: str) -> None:
        if self.verbose:
            eprint(msg)

    # ── the head call itself ───────────────────────────────────────────────
    def _head(self, m, y):
        """golden's head_logits, run up to three times over the same operands."""
        self.n_calls += 1
        if not self.armed or m is not self.model:
            return self._o_head(m, y)
        self.armed = False              # only the FIRST head call is offloaded

        # pass 1 — the arbiter, with nothing of ours in the loop
        self._serve_on = False
        t_gold = time.perf_counter()
        gold_logits = np.asarray(self._o_head(m, y), dtype=np.float64).ravel()
        self._gold_seconds = time.perf_counter() - t_gold

        # pass 2 — the tile's own accumulators, spliced into golden's epilogue
        self._cursor = self._rows_this_pass = 0
        self._serve_on, self._recording = True, True
        t0 = time.perf_counter()
        tile_logits = np.asarray(self._o_head(m, y), dtype=np.float64).ravel()
        self._seconds = time.perf_counter() - t0
        self._serve_on = False

        served = self.served_rows
        dlogit = float(np.max(np.abs(tile_logits - gold_logits))) \
            if gold_logits.size == tile_logits.size else float("inf")
        arg_g, arg_t = int(np.argmax(gold_logits)), int(np.argmax(tile_logits))
        srt = np.sort(gold_logits)[::-1]
        margin = float(srt[0] - srt[1]) if srt.size > 1 else 0.0

        # pass 3 — the discriminator, on the tile's OWN captured values
        pois = None
        if self.poison is not None and served:
            row = arg_g if arg_g < served else int(
                np.argmax(gold_logits[:served]))
            self._poison_row = row
            self._cursor = self._rows_this_pass = 0
            self._serve_on, self._recording = True, False
            pl = np.asarray(self._o_head(m, y), dtype=np.float64).ravel()
            self._serve_on, self._recording = False, True
            self._poison_row = None
            pois = {"k": float(self.poison), "row": row,
                    "row_was_top1": row == arg_g,
                    "argmax": int(np.argmax(pl)),
                    "logit_before": float(tile_logits[row]),
                    "logit_after": float(pl[row]),
                    "max_abs_dlogit_vs_tile": float(np.max(np.abs(
                        pl - tile_logits))),
                    "logit_moved": float(pl[row]) != float(tile_logits[row]),
                    "token_flipped": int(np.argmax(pl)) != arg_t,
                    "top1_row_served": arg_g < served,
                    "logits": pl}

        self.result = {
            "vocab": self.vocab, "rows_served": served,
            "rows_total": self.vocab,
            "sampled": served < self.vocab,
            "full_width": served == self.vocab,
            "K": self.dm, "mode": self.mode,
            "argmax_golden": arg_g, "argmax_tile": arg_t,
            "argmax_equal": arg_g == arg_t,
            "max_abs_dlogit_tile_vs_golden": dlogit,
            "top1_margin_golden": margin,
            "logit_range_golden": float(gold_logits.max() - gold_logits.min()),
            "seconds": round(self._seconds, 2),
            "host_seconds_golden_arm": round(self._gold_seconds, 2),
            "poison": ({k: v for k, v in pois.items() if k != "logits"}
                       if pois else None),
            "cost": dict(self.cost),
            "gemm_calls": self.n_gemm_calls,
            "invocations": self.n_invocations,
            "predicted_hw_wall_s": self.predicted_hw_wall(),
        }
        # ── the checks this stage lives or dies by ─────────────────────────
        exact = all(r.exact for r in self.records if r.n_served)
        self.chk("lm_head: the tile's INT32 accumulators ARE golden's own "
                 "gemm_i8_ksplit accumulators, per call, bit for bit",
                 bool(exact) and served > 0,
                 {"rows_served": served, "calls": len(self.records)})
        self.chk("lm_head: the logits the model argmaxed are the TILE's "
                 "(max|dlogit| vs golden's own logits)",
                 dlogit == 0.0, {"max_abs_dlogit": dlogit})
        self.chk("lm_head: argmax over TILE-computed logits == argmax over "
                 "golden's logits",
                 arg_g == arg_t, {"golden": arg_g, "tile": arg_t,
                                  "top1_margin": margin})
        if pois is not None:
            det = {k: v for k, v in pois.items() if k != "logits"}
            # What a poison can PROVE depends on the scope it ran in. Moving
            # the logit proves the tile's accumulator reached the logits;
            # flipping the token proves it reached the token — and only a
            # scope that actually served golden's top-1 row can be asked for
            # the second, because demoting a row that was never winning
            # cannot change an argmax. The check names which one it is.
            self.chk(f"lm_head DISCRIMINATOR: scaling the tile accumulator of "
                     f"ONE vocabulary row (#{pois['row']}) by {pois['k']} "
                     f"MOVES that logit", pois["logit_moved"], det)
            if pois["row_was_top1"]:
                self.chk(f"lm_head DISCRIMINATOR: ...and row #{pois['row']} "
                         f"IS golden's top-1, so the emitted token FLIPS "
                         f"(the tile's value is what the argmax consumed)",
                         pois["token_flipped"], det)
            else:
                eprint(f"[lm_head] NOTE: --lmhead-rows {self.rows} does not "
                       f"cover golden's top-1 row {arg_g}, so the poisoned "
                       f"row #{pois['row']} is not the winner and demoting it "
                       f"CANNOT flip the token. The token-level "
                       f"discriminator needs a scope that serves row "
                       f"{arg_g} (--lmhead-rows -1).")
        self.say(f"[lm_head] {served}/{self.vocab} vocabulary rows from the "
                 f"tile in {self.cost['programs']} programs / "
                 f"{self.cost['blocks']} blocks "
                 f"({self.cost['ops']:,} BAR0 ops, {self.cost['peeks']:,} "
                 f"peeks), bit-exact={exact}, argmax {arg_t} "
                 f"{'==' if arg_g == arg_t else '!='} golden's {arg_g} "
                 f"({self._seconds:.1f}s)")
        return tile_logits

    # ── the GEMM inside it ─────────────────────────────────────────────────
    def _gemm(self, A, B, *a, **kw):
        """compute.gemm_i8_ksplit, rebound.

        EVERY ksplit in the run lands here — including the N > DIM_MAX
        recursion of the gate/up projections, which goes through compute.py's
        own module global (compute.py:62-65), i.e. through this very name. The
        `_serve_on` gate is what keeps this a lm_head wrapper and not a
        second, unaccounted projection seam.
        """
        if self._depth or not self._serve_on:
            return self._o_gemm(A, B, *a, **kw)
        self._depth += 1
        try:
            return self._serve(np.asarray(A, dtype=np.int64),
                               np.asarray(B, dtype=np.int64), a, kw)
        finally:
            self._depth -= 1

    def _expect_head_slice(self, A) -> int:
        """REFUSE an operand that is not the next slice of the model's own
        head tensor. The projections have `_LayerState.next_proj`; this is the
        same obligation for the head, and it is why "the tile computed the
        lm_head" is checked rather than assumed."""
        v0, M = self._cursor, int(A.shape[0])
        if A.shape[1] != self.dm:
            raise SystemExit(
                f"REFUSE: a GEMM inside head_logits contracted over "
                f"{A.shape[1]}, not D_model={self.dm} — this wrapper is not "
                f"looking at the lm_head it claims to serve.")
        if v0 + M > self.vocab:
            raise SystemExit(
                f"REFUSE: the head's GEMM calls cover rows {v0}..{v0 + M}, "
                f"past the {self.vocab}-row head tensor.")
        want = np.asarray(self.model.head_w8[v0:v0 + M], dtype=np.int64)
        if not np.array_equal(A, want):
            raise SystemExit(
                f"REFUSE: the operand of head GEMM call #{self.n_gemm_calls} "
                f"is not head_w8[{v0}:{v0 + M}]. The tile would be computing "
                f"something other than the vocabulary projection.")
        self._cursor = v0 + M
        return v0

    def _serve(self, A, B, a, kw):
        v0 = self._expect_head_slice(A)
        M, K = int(A.shape[0]), int(A.shape[1])
        self.n_gemm_calls += 1
        gold = np.asarray(self._o_gemm(A, B, *a, **kw), dtype=np.int64)
        if B.shape != (K, 1):
            raise SystemExit(
                f"REFUSE: head GEMM B is {B.shape}, expected ({K}, 1) — one "
                f"activation column is what makes an 8-row block one MXE "
                f"descriptor.")
        rec = lof.OpRec(op="lmhead", name=f"lmhead_v{v0:06d}",
                        source="HOST (golden)", n_total=M)
        rec.detail = {"v0": v0, "M": M, "K": K, "rows_served": 0}
        # The row budget is PER PASS, not per run: pass 3 replays the same
        # head call, so a budget carried over from pass 2 would silently serve
        # it nothing and the discriminator would poison an empty set.
        nrow = max(0, min(M, self.rows - self._rows_this_pass))
        self._rows_this_pass += nrow
        if not nrow:
            rec.notes.append("outside --lmhead-rows")
            if self._recording:
                self.records.append(rec)
            return gold.astype(np.int32)

        memo_fp = lof.operand_key(A[:nrow], B, v0, nrow, self.rows_per_desc,
                                  self.blocks_per_program, self.burst,
                                  lof.D_TILE, gj.D_TILE)
        hit = self._acc_memo.get(memo_fp)
        out = gold.copy()
        t0 = time.perf_counter()
        if hit is not None:
            out[:nrow, 0] = hit
            eq = bool(np.array_equal(out[:nrow, 0], gold[:nrow, 0]))
            n_prog = n_blocks = 0
            rec.detail["emission"] = "accumulator memo (pass 3)"
        else:
            out, eq, n_prog, n_blocks, notes = self._run_call(
                A, B, gold, v0, nrow, rec)
            rec.notes.extend(notes)
            self._acc_memo[memo_fp] = out[:nrow, 0].copy()

        if self._recording:
            self.served_rows += nrow
            rec.n_served = nrow
            rec.jobs = n_prog
            rec.seconds = round(time.perf_counter() - t0, 2)
            rec.source = f"TILE ({self.mode})"
            rec.exact = rec.grade_ok = eq
            rec.detail.update({"rows_served": nrow, "blocks": n_blocks,
                               "programs": n_prog})
            self.records.append(rec)

        # the discriminator: ONE vocabulary row of the tile's own answer
        if self._poison_row is not None and v0 <= self._poison_row < v0 + nrow:
            j = self._poison_row - v0
            out[j, 0] = int(np.rint(float(out[j, 0]) * float(self.poison)))
        return out.astype(np.int32)

    def _run_call(self, A, B, gold, v0, nrow, rec):
        """Emit, run and decode ONE head GEMM call. -> (out, eq, progs, blocks,
        notes). Mirrors MultiOffloader05B._plan_programs: the SAME staging
        (gemm_job.stage_plan with the +127 sentinel), the SAME fat multiblock
        descriptors, the SAME K-split chunking."""
        notes = []
        out = gold.copy()
        x8 = np.asarray(B[:, 0], dtype=np.int64)
        amax = int(np.max(np.abs(x8), initial=0))
        assert amax <= 127, (
            f"lm_head activation code magnitude {amax} > 127 — the "
            f"INT8-symmetric staging bound")
        pad = amax != 127
        xs = np.concatenate([x8, [127]]) if pad else x8
        zrow = np.zeros((1, lof.MXE_N), dtype=np.int64)

        def block_w(j, n):
            # EIGHT VOCABULARY ROWS, transposed into one 8-column MXE block:
            # W[k, i] = head_w8[v0 + j + i, k].
            W = np.ascontiguousarray(A[j:j + n].T)             # [K, n]
            return np.concatenate([W, zrow[:, :n]]) if pad else W

        if self._plan_cache is None or not np.array_equal(
                self._plan_cache[0], xs):
            self._plan_cache = (xs.copy(),
                                gj.stage_plan(xs, block_w(0, min(lof.MXE_N,
                                                                 nrow))))
        plan = self._plan_cache[1]

        staged = []
        for j in range(0, nrow, lof.MXE_N):
            n = min(lof.MXE_N, nrow - j)
            staged.append((j, n, plan.restage(block_w(j, n))))
        if self.mode == "golden":
            # the dry run: the seam, the ledger and the share with no executor
            return out, True, 0, len(staged), ["--dry-run: golden's own values"]

        partials, widths = {}, {}
        n_prog = 0
        for ci, (r0, _nr, K) in enumerate(plan.chunks(self.rows_per_desc)):
            off = r0 * lof.D_TILE
            xst = plan.xst[off:off + K]
            programs = []
            for gi in range(0, len(staged), self.blocks_per_program):
                grp = staged[gi:gi + self.blocks_per_program]
                nm = (f"lmhead_v{v0:06d}_k{ci}"
                      f"_g{gi // self.blocks_per_program:03d}")
                p, _man = gj.build_gemm_multiblock_job(
                    [W[off:off + K] for (_j, _n, W) in grp], xst,
                    self.work, nm, burst=self.burst,
                    allow_multirow=self.rows_per_desc > 1 or K <= lof.D_TILE)
                programs.append({"path": str(p), "blocks": len(grp),
                                 "units": [{"j": j, "n": n}
                                           for (j, n, _W) in grp]})
            paths = [p["path"] for p in programs]
            for p in paths:
                k = gj.program_cost(p)
                for key in ("ops", "peeks", "pokes", "bytes"):
                    self.cost[key] += k[key]
            self.cost["programs"] += len(programs)
            self.cost["blocks"] += sum(p["blocks"] for p in programs)
            n_prog += len(programs)
            caps_by_path = self._run_bounded(paths, f"lmhead_v{v0:06d}_k{ci}")
            for prog in programs:
                caps = caps_by_path[prog["path"]]
                rec.caps += len(caps)
                accs = gj.decode_multiblock(caps, prog["blocks"])
                for u, acc in zip(prog["units"], accs):
                    partials.setdefault(u["j"], []).append(
                        np.asarray(acc, dtype=np.int64))
                    widths[u["j"]] = u["n"]
            self._reclaim(paths)

        eq = True
        for j, parts in sorted(partials.items()):
            n = widths[j]
            acc = np.asarray(gj.accumulate_partials(parts), dtype=np.int64)
            out[j:j + n, 0] = acc[:n]
            eq = eq and bool(np.array_equal(acc[:n], gold[j:j + n, 0]))
        return out, eq, n_prog, len(staged), notes

    def _run_bounded(self, paths, tag) -> dict:
        """Run these programs in <= --max-batch-mb invocations.

        The head is NOT inside the layer replay — it is called once, its
        output is consumed by the argmax and by nothing else — so there is no
        fixed point to reach and no capture pool to fill: `Runner.run` (which
        still prepares, audits and retargets every program) is the honest
        path. The byte bound is the same one the collapse uses, so an
        invocation here costs what an invocation there costs.
        """
        out, cur, cur_b, bi = {}, [], 0, 0
        limit = self.runner.max_batch_bytes
        for p in paths:
            b = Path(p).stat().st_size
            if cur and cur_b + b > limit:
                out.update(zip(cur, self.runner.run(cur, f"{tag}_f{bi:02d}")))
                self.n_invocations += 1
                bi, cur, cur_b = bi + 1, [], 0
            cur.append(p)
            cur_b += b
        if cur:
            out.update(zip(cur, self.runner.run(cur, f"{tag}_f{bi:02d}")))
            self.n_invocations += 1
        return out

    def predicted_hw_wall(self) -> dict | None:
        """What this program set would cost on the CARD — A PREDICTION.

        No hardware is touched by this file. The coefficients are the COMMITTED
        fit (docs/results/prompt_on_chip/fat_hw_prediction.json, fatproof.py's
        two-point solve against the s3_sweep and p05b_layer silicon runs);
        only the program/peek/invocation counts below are this run's own
        measurement. Returns None if the committed fit is not in the tree,
        because a prediction with invented coefficients is worse than none.
        """
        f = REPO / "docs/results/prompt_on_chip/fat_hw_prediction.json"
        if not f.exists() or not self.cost["programs"]:
            return None
        try:
            m = json.loads(f.read_text())["model"]
        except (ValueError, KeyError):
            return None
        wall = (m["c_inv_s"] * self.n_invocations
                + m["per_program_s"] * self.cost["programs"]
                + m["per_peek_s"] * self.cost["peeks"])
        return {"seconds": round(wall, 1),
                "invocation_setup_s": round(m["c_inv_s"]
                                            * self.n_invocations, 1),
                "per_program_s": round(m["per_program_s"]
                                       * self.cost["programs"], 1),
                "peek_s": round(m["per_peek_s"] * self.cost["peeks"], 1),
                "model": {k: m[k] for k in ("c_inv_s", "per_program_s",
                                            "per_peek_s")},
                "source": str(f.relative_to(REPO)),
                "measured_here": {"programs": self.cost["programs"],
                                  "peeks": self.cost["peeks"],
                                  "invocations": self.n_invocations},
                "label": "PREDICTION, NOT A MEASUREMENT — committed fit x "
                         "this run's own program set"}

    def _reclaim(self, paths) -> None:
        """Full width is ~1.9 GB of regops; keep the first program of the run
        as evidence and delete the rest once their captures are decoded."""
        if not self.prune:
            return
        keep = None
        if not self._kept_evidence and paths:
            self._kept_evidence, keep = True, paths[0]
            eprint(f"[lm_head] keeping {Path(keep).name} as evidence "
                   f"(later programs are reclaimed after decode; "
                   f"--keep-jobs disables it)")
        for p in paths:
            if p == keep:
                continue
            for q in (Path(p), Path(str(p).replace(".regops.jsonl",
                                                   ".manifest.json"))):
                try:
                    self._pruned_bytes += q.stat().st_size
                    q.unlink()
                except OSError:
                    pass
        for d in self.work.glob("lmhead_v*_f*_batch"):
            shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════ the FLOP-share model ══════════════════════════

def flop_share(census: MacCensus, step: int, records: list) -> dict:
    """What fraction of the decode step's arithmetic the tile actually did.

    NUMERATOR — only what the tile produced, from the per-op records:
      proj   n_served accumulators x K (the record's own measured contraction)
      attn   2*T*D per SERVED head (Q·K̂ᵀ + P·V̂, the two gemm_i8 inside
             attention_core), from the record's own measured T and D
      lmhead n_served vocabulary rows x K (D_model) — the last GEMM of the
             token, counted ONLY for rows the tile actually produced
    RoPE, residual, RMSNorm and SwiGLU are elementwise; they are NOT
    multiply-accumulate arithmetic and are excluded from BOTH sides, with the
    elementwise touch count reported so the exclusion can be checked.

    DENOMINATOR — the census's observation of the same step, plus the lm_head
    the token needs, so the share is a share of the whole thing.

    TWO SHARES, KEPT APART. `share_of_step` is the 24 decoder layers only, and
    its numerator NEVER contains lm_head MACs — the lm_head is not part of the
    step's denominator either. `share_of_token` is the number the demo cares
    about: (layers the tile did + lm_head rows the tile did) / (whole step +
    whole lm_head). With the lm_head on the host the two differ by exactly the
    26.4% the head costs; that gap closing is the whole point of this stage.
    """
    st = census.step_macs(step)
    tile_proj = tile_attn = tile_head = 0
    proj_served = proj_total = 0
    head_served = head_total = 0
    attn_heads = 0
    head_mode = "HOST (golden)"
    for r in records:
        if r.op == "proj" and r.n_served:
            K = int(r.detail.get("K") or (r.detail.get("A") or [0, 0])[1])
            tile_proj += int(r.n_served) * K
            proj_served += int(r.n_served)
        if r.op == "proj":
            proj_total += int(r.n_total)
        if r.op == "attn" and r.n_served:
            T, D = int(r.detail.get("T", 0)), int(r.detail.get("D", 0))
            tile_attn += 2 * T * D
            attn_heads += 1
        if r.op == "lmhead":
            head_total += int(r.n_total)
            if r.n_served:
                tile_head += int(r.n_served) * int(r.detail.get("K", 0))
                head_served += int(r.n_served)
                head_mode = r.source
    step_total = st["proj"] + st["attn"]
    head_macs = census.outside_layer_macs
    token_total = step_total + head_macs
    tile = tile_proj + tile_attn
    tile_token = tile + tile_head
    return {
        "step": step,
        "layers_in_step": st["layers_seen"],
        "denominator_step_macs": step_total,
        "denominator_proj_macs": st["proj"],
        "denominator_attn_macs": st["attn"],
        "lm_head_macs": head_macs,
        "denominator_token_macs": token_total,
        "tile_macs": tile,
        "tile_proj_macs": tile_proj,
        "tile_attn_macs": tile_attn,
        "tile_lmhead_macs": tile_head,
        "tile_token_macs": tile_token,
        "share_of_step": (100.0 * tile / step_total) if step_total else 0.0,
        "share_of_token": ((100.0 * tile_token / token_total)
                           if token_total else 0.0),
        "share_of_token_layers_only": ((100.0 * tile / token_total)
                                       if token_total else 0.0),
        "lm_head_share_of_token": ((100.0 * head_macs / token_total)
                                   if token_total else 0.0),
        "proj_accumulators_served": proj_served,
        "proj_accumulators_total": proj_total,
        "proj_sampled": bool(proj_total and proj_served < proj_total),
        "lm_head_rows_served": head_served,
        "lm_head_rows_total": head_total,
        "lm_head_on_tile": bool(head_served),
        "lm_head_sampled": bool(head_served and head_total
                                and head_served < head_total),
        "lm_head_mode": ("HOST (golden)" if not head_served else
                         (f"{head_mode}, "
                          + ("FULL WIDTH" if head_served >= head_total
                             else f"SAMPLED {head_served}/{head_total} rows"))),
        "attn_heads_served": attn_heads,
        "per_tensor": st["per_tensor"],
        "per_layer": {str(k): v for k, v in st["per_layer"].items()},
        "elem_touches_norm_in_step": st["elem_touches_norm"],
    }


# ══════════════════════════════ the run ════════════════════════════════════

def _decode(model, ids, args, tier, eos):
    return po.decode(model, ids, args.max_tokens, tier, args.group, eos, None,
                     verbose=False)


def _offload_run(model, ids, args, tier, eos, work, runner, layers, *,
                 scope, step, poison=None, poison_layers=(), cross_check=False,
                 tag="on", fat=None, lmhead_rows=0, lmhead_poison=None):
    """One OFFLOAD-ON decode. Returns (result dict, offloader, head offloader).

    The lm_head offloader nests INSIDE the layer offloader and rebinds a
    disjoint pair of names (rt.GoldenModel.head_logits and
    compute.gemm_i8_ksplit; the layer seam owns transformer.gemm_i8_ksplit),
    so the two seams cannot shadow each other. It is armed only where the
    caller asks for it — the control/poison arms of the layer discriminator
    do not pay for a 151,936-row head they are not measuring.
    """
    t0 = time.perf_counter()
    with l05.d64_emission(args.d):
        with MultiOffloader05B(
                n_layers=model.n_layers, mode=args.mode, work=work,
                scope=scope, runner=runner, step=step, layers=layers,
                poison=poison, poison_layers=poison_layers,
                batch_size=args.batch_size, cross_check=cross_check,
                prune=not args.keep_jobs,
                fat=(args.fat if fat is None else fat), burst=args.burst,
                blocks_per_program=args.blocks_per_program,
                rows_per_desc=args.rows_per_desc,
                max_passes=args.max_passes) as off:
            with LmHeadOffloader(
                    model, mode=args.mode, work=work, runner=runner,
                    rows=lmhead_rows, poison=lmhead_poison,
                    blocks_per_program=args.lmhead_blocks_per_program,
                    burst=args.burst, rows_per_desc=args.rows_per_desc,
                    prune=not args.keep_jobs) as head:
                r = _decode(model, ids, args, tier, eos)
    r["records"] = off.records + head.records
    r["checks"] = off.checks + head.checks
    r["passes"], r["invocations"] = dict(off.passes), dict(off.invocations)
    r["lm_head"] = head.result
    r["wall"] = time.perf_counter() - t0
    r["tag"] = tag
    return r, off, head


def run_multi(model, tok, args, work: Path) -> int:
    ids = tok.encode(args.prompt) if tok else list(args.ids)
    n_total = len(ids) + args.max_tokens
    if n_total > lof.TOKENS_MAX:
        eprint(f"REFUSE: prompt({len(ids)}) + max_tokens({args.max_tokens}) = "
               f"{n_total} > {lof.TOKENS_MAX} — beyond that a head becomes a "
               f"ChunkedHead whose merge needs per-chunk sm_m/sm_l, which the "
               f"mailbox does not expose (audit N8).")
        return 2
    L = model.n_layers
    hd = int(model.meta["head_dim"])
    if hd != args.d:
        eprint(f"REFUSE: model head_dim {hd} != --d {args.d}")
        return 2
    layers = parse_layers(args.layers, L)
    tier = rt.TIER_MAP[args.tier]
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    args.mode = "golden" if args.dry_run else args.executor
    image = tg.IMG_05B

    if args.mode == "hw":
        # The missing-attach trap: without the shim the bridge drives BAR0
        # from THIS machine, every batch comes back with no cap file at all,
        # and the run looks like a clean zero-capture pass
        # (docs/results/prompt_on_chip/NARROW_LANE_RESULT.md).
        if not rhe.attach(bridge, args):
            eprint("REFUSE: --executor hw but no remote config — set "
                   "APEX_F2_HOST / APEX_F2_KEY (or --hw-host/--hw-key). "
                   "Without the attach shim every job returns ZERO captures "
                   "and the run would look green while touching no tile.")
            return 2
        eprint("[P2-C] remote hw shim attached (per-invocation clock gate on)")
    if args.mode == "sim" and not (args.binary or image.binary.exists()):
        eprint(f"REFUSE: the {image.name} twin is not built.\n"
               f"  cd verif/f2sim && make build D={image.cfg_d} DDR=0 "
               f"OBJ={image.obj} VFLAGS_EXTRA=\"{image.defines()}\"")
        return 2

    step_target = (args.offload_step if args.offload_step is not None
                   else len(ids) - 1)
    if not 0 <= step_target < len(ids) + args.max_tokens:
        eprint(f"REFUSE: --offload-step {step_target} outside the run")
        return 2

    ops = tuple(args.ops)
    recon_on = tuple(o for o in ops if o in RECON_FAMILIES)
    dm = int(model.meta["D_model"])
    scope = lof.Scope(ops=ops, proj_cols=args.proj_cols,
                      proj_rows=args.proj_rows, heads=args.heads,
                      rope_heads=args.rope_heads, resid_cols=args.resid_cols,
                      resid_slice=(dm if args.resid_slice < 0
                                   else args.resid_slice),
                      norm_dm=args.norm_dm, norm1_rows=args.norm1_rows,
                      swiglu_chunks=args.swiglu_chunks)
    work.mkdir(parents=True, exist_ok=True)

    eprint(f"[P2-C] model={model.meta['model']}  L={L} H={model.meta['H']} "
           f"H_kv={model.meta.get('H_kv')} head_dim={hd} D_model={dm} "
           f"d_ffn={model.meta.get('d_ffn')}")
    eprint(f"[P2-C] layers={layers} step={step_target} mode={args.mode} "
           f"ops={','.join(ops)} proj_cols="
           f"{'FULL' if args.proj_cols >= FULL else args.proj_cols}")

    # ── the transport shape, decided PER IMAGE, printed before anything runs
    # rows_per_desc: how many staged rows one descriptor may contract over.
    # It is NOT a free knob — an image without 38ec95c's base-row LOAD
    # returns silently wrong numbers above 1 (tile_geom §4).
    pol_exec = "hw" if args.mode == "hw" else "sim"
    ceiling = tg.rows_per_desc_for(image, args.d, executor=pol_exec)
    if args.rows_per_desc < 0:
        args.rows_per_desc = ceiling
    elif args.rows_per_desc > ceiling:
        eprint(f"REFUSE: --rows-per-desc {args.rows_per_desc} > the legal "
               f"ceiling {ceiling} for this image. "
               f"{tg.rows_per_desc_note(image, args.d, executor=pol_exec)}")
        return 2
    eprint(f"[P2-C] transport: fat={args.fat} "
           f"blocks_per_program={args.blocks_per_program} burst={args.burst} "
           f"(FIFO depth {gj.MB_FIFO_DEPTH}); "
           f"{tg.rows_per_desc_note(image, args.d, executor=pol_exec)}"
           f" -> using {args.rows_per_desc}")
    if recon_on:
        eprint(f"[P2-C] WARNING: --include-reconstructed is ON "
               f"({','.join(recon_on)}): those families re-enter the host "
               f"through the 64-wide C-1 feeder, the emitted token MAY "
               f"legitimately differ from pure host, and this run is graded "
               f"at the LOGIT level.")

    eprint(f"[P2-C] transport: collapse={args.collapse} "
           f"(<= {args.max_batch_mb} MB per executor invocation, <= "
           f"{args.max_passes} layer replays)"
           + ("" if args.collapse else
              "  — ONE INVOCATION PER OP: the pre-collapse baseline"))

    # ── the lm_head scope, decided and printed before anything runs ───────
    vocab = int(np.asarray(model.head_w8).shape[0])
    if args.lmhead_rows > vocab:
        eprint(f"REFUSE: --lmhead-rows {args.lmhead_rows} > the model's "
               f"{vocab}-row head")
        return 2
    lm_rows = vocab if args.lmhead_rows < 0 else int(args.lmhead_rows)
    if lm_rows:
        eprint(f"[P2-C] lm_head: {lm_rows}/{vocab} vocabulary rows on the "
               f"tile ({'FULL WIDTH' if lm_rows == vocab else 'SAMPLED'}), "
               f"K={dm}, >= {-(-lm_rows // lof.MXE_N)} 8-row blocks "
               f"(golden's own {cp.DIM_MAX}-row chunking leaves a ragged tail "
               f"block per call, so the MEASURED count in the ledger is a few "
               f"higher), {args.lmhead_blocks_per_program} blocks/program. "
               f"golden's own INT8 head GEMM (run_tinynpu.py:277-292); the "
               f"tile returns raw INT32 and every float stays golden's.")
    else:
        eprint("[P2-C] lm_head: HOST (golden) — --lmhead-rows 0")

    runner = l05.Runner05B(
        args.mode, work, image=image, d=args.d,
        binary=(args.binary or (str(image.binary)
                                if args.mode == "sim" else None)),
        tile_div=args.tile_div, slot=args.slot, timeout_s=args.timeout_s,
        collapse=args.collapse, max_batch_mb=args.max_batch_mb)

    eprint(f"[P2-C] === run 1/3: OFFLOAD ON ({len(layers)} layers"
           + (f" + lm_head {lm_rows}/{vocab}" if lm_rows else "") + ") ===")
    on, off, head = _offload_run(
        model, ids, args, tier, eos, work, runner, layers,
        scope=scope, step=step_target, cross_check=args.cross_check, tag="on",
        lmhead_rows=lm_rows, lmhead_poison=args.lmhead_poison)

    eprint("[P2-C] === run 2/3: PURE HOST (golden, default bus) + MAC census ===")
    assert tf.attention_core is at.attention_core, "rebind not restored"
    assert tf.gemm_i8_ksplit is cp.gemm_i8_ksplit, \
        "the lm_head seam did not restore compute.gemm_i8_ksplit — the " \
        "pure-host arm would not be a pure-host arm"
    assert rt.GoldenModel.head_logits.__qualname__.startswith("GoldenModel"), \
        "the lm_head seam did not restore GoldenModel.head_logits"
    assert (lof.D_TILE, gl.D_TILE) == l05._FROZEN_FRAMES \
        and gl.build_norm2_chunked is l05._FROZEN_NORM, \
        "the geometry rebind leaked out of the emission scope — the pure-host " \
        "arm of the A/B would not be a pure-host arm"
    with MacCensus(L) as census:
        host = _decode(model, ids, args, tier, eos)
    ver = census.verify_against(model, step_target)
    eprint(f"[P2-C] MAC census verified against the model's own tensors: "
           f"{ver['layers']} layers, {ver['proj_macs']:,} projection MACs, "
           f"{ver['lm_head_macs']:,} lm_head MACs, "
           f"{ver['recursions_skipped']} inner ksplit recursions counted once")

    eprint("[P2-C] === run 3/3: HOST ONLY, target layers in C-LBUS BUS_ON ===")
    with BusOnlyMulti(L, layers, step_target) as bo:
        bus = _decode(model, ids, args, tier, eos)
    bus["fired"] = bo.fired

    # ── the discriminator: a matched control/poison pair on one layer ──────
    pois = ctrl = None
    if args.poison is not None:
        players = (parse_layers(args.poison_layers, L)
                   if args.poison_layers else (layers[0],))
        pscope = lof.Scope(**{**scope.__dict__,
                             "proj_cols": args.poison_proj_cols})
        eprint(f"[P2-C] === discriminator on layer(s) {players} "
               f"(proj_cols={args.poison_proj_cols}) ===")
        ctrl, _c, _ch = _offload_run(model, ids, args, tier, eos, work, runner,
                                     players, scope=pscope, step=step_target,
                                     tag="control")
        pois, _p, _ph = _offload_run(model, ids, args, tier, eos, work, runner,
                                     players, scope=pscope, step=step_target,
                                     poison=args.poison, poison_layers=players,
                                     tag="poison")
        pois["layers"] = players
    else:
        players = ()

    return report(model, tok, args, ids, layers, on, host, bus, ctrl, pois,
                  players, off, head, runner, census, image, scope,
                  step_target, work)


# ═══════════════════════════════ the ledger ════════════════════════════════

def _margin(host, on, bus=None) -> dict:
    """The two deltas a multi-layer run has to keep apart.

    `dlogit` (offload vs PURE host) mixes two causes: the tile's values AND
    the C-LBUS BUS_ON composition every offloaded layer must run in
    (apex_layer_deq.sv:90-92 refuses an ungraded fp32 composite, so the
    residual op is unreachable in golden's default BUS_OFF). Over many layers
    the composition change alone accumulates. `dlogit_vs_bus` (offload vs the
    HOST run in the SAME composition) isolates the tile: for exact-egress
    families it is the number that must be 0.
    """
    h = np.asarray(host["logits"], dtype=np.float64).ravel()
    o = np.asarray(on["logits"], dtype=np.float64).ravel()
    s = np.sort(h)[::-1]
    out = {"host_margin": float(s[0] - s[1]),
           "host_range": float(h.max() - h.min()),
           "dlogit": float(np.max(np.abs(o - h))) if o.size == h.size
           else float("nan"), "dlogit_vs_bus": float("nan"),
           "dlogit_bus_vs_host": float("nan")}
    if bus is not None and bus.get("logits") is not None:
        b = np.asarray(bus["logits"], dtype=np.float64).ravel()
        if b.size == o.size:
            out["dlogit_vs_bus"] = float(np.max(np.abs(o - b)))
            out["dlogit_bus_vs_host"] = float(np.max(np.abs(b - h)))
    return out


def _fmt_int(v) -> str:
    return f"{int(v):,}"


def report(model, tok, args, ids, layers, on, host, bus, ctrl, pois, players,
           off, lmh, runner, census, image, scope, step, work) -> int:
    recs, checks = on["records"], on["checks"]
    ident = on["ids"] == host["ids"]           # tile vs PURE host
    ident_bus = on["ids"] == bus["ids"]        # tile vs host in the SAME bus
    bus_ident = bus["ids"] == host["ids"]      # the bus mode's own effect
    fired = any(r.n_served for r in recs)
    agg = {op: lof._agg(recs, op) for op in OP_TYPES}
    served_ops = [op for op in OP_TYPES if agg[op]["values"] > 0]
    checks_ok = all(ok for _n, ok, _d in checks)
    n_ok_pre = sum(1 for _n, ok, _d in checks if ok)
    # the per-LAYER table totals only the layer seam's checks; the lm_head's
    # live outside any layer and are counted in the run-wide line below.
    n_ok_layer = sum(1 for _n, ok, _d in off.checks if ok)
    grades_ok = all(agg[op]["grade_ok"] is not False for op in OP_TYPES)
    unaudited = runner.unaudited_executed()
    geom_ok = (args.mode == "golden") or (
        len(runner.audited_keys) >= len(runner.executed_keys) > 0
        and unaudited == 0)
    recon_on = [o for o in scope.ops if o in RECON_FAMILIES]
    fs = flop_share(census, step, recs)
    margin = _margin(host, on, bus)

    def txt(v):
        return tok.decode(v) if tok else str(v)

    W = 78
    P = print
    P("\n" + "=" * W)
    P("P2 STAGE 3 — MANY Qwen2.5-0.5B DECODER LAYERS SERVED BY THE D=64 TILE")
    P("=" * W)
    P(f"  prompt          : {args.prompt!r}")
    P(f"  prompt ids      : {ids} ({len(ids)} tokens)")
    P(f"  model           : {model.meta['model']}")
    P(f"                    L={model.n_layers} H={model.meta['H']} "
      f"H_kv={model.meta.get('H_kv')} head_dim={model.meta['head_dim']} "
      f"D_model={model.meta['D_model']} d_ffn={model.meta.get('d_ffn')}")
    P(f"  image           : {image.name}  CFG_D={image.cfg_d} "
      f"CFG_DM={image.cfg_dm} KVQ_GQA_NENG={image.gqa_neng} "
      f"RMS/LAYER_DM_MAX={image.dm_max} QSTAGE_H_MAX={image.qstage}")
    P(f"  offloaded layers: {list(layers)}  ({len(layers)} of "
      f"{model.n_layers})  decode step {step}  composition C-LBUS BUS_ON")
    P(f"  lm_head         : {fs['lm_head_mode']}"
      + (f"   ({_fmt_int(fs['lm_head_macs'])} MACs = "
         f"{fs['lm_head_share_of_token']:.1f}% of the token)"))
    P(f"  op families     : {','.join(scope.ops)}"
      + ("   [FOUR EXACT-EGRESS FAMILIES — token identity is a PASS "
         "condition]" if not recon_on else ""))
    if recon_on:
        P(f"  *** --include-reconstructed IS ON ({','.join(recon_on)}). Those "
          f"families re-enter")
        P(f"      the host through the {image.cfg_dm}-wide C-1 feeder "
          f"(B-FEED-WIDTH): golden composes ONE")
        P(f"      whole-row C-1 over {model.meta['D_model']} elements, the "
          f"tile can only hand back")
        P(f"      independently-scaled {image.cfg_dm}-wide frames. THE "
          f"EMITTED TOKEN MAY LEGITIMATELY")
        P(f"      DIFFER from the pure-host run — stage 2 measured a flip on "
          f"2 of 4 prompts")
        P(f"      (docs/results/p2_b64_cl/RESULTS.md §5). This run is "
          f"therefore graded at the")
        P(f"      LOGIT level: token identity is REPORTED, not required.")
    P(f"  executor        : {args.mode}"
      + ("   (--dry-run: every tile call replaced by golden — seam only)"
         if args.mode == "golden" else f"   binary {runner.binary}"))
    P(f"  tile jobs       : {runner.n_jobs} programs, {runner.n_caps} capture "
      f"records, {runner.wall:.1f}s in the executor")
    for line in lof.transport_lines(runner, on):
        P(f"  {line}")
    if args.mode != "golden":
        P(f"  geometry audit  : {len(runner.audited_keys)} distinct programs "
          f"audited ({runner.n_audited} audit passes over re-emitted files), "
          f"{len(runner.executed_keys)} distinct programs executed, "
          f"{unaudited} executed WITHOUT an audit — each audited program "
          f"carries INFO_D == {args.d}; descriptor k values seen = "
          f"{sorted(runner.desc_k_seen)}  -> "
          f"{'PASS' if geom_ok else 'FAIL'}")
        P(f"  disclosed retarget: {runner.n_retargeted} INFO_TIER "
          f"expectations rewritten 0x7 -> {image.info_tier:#x}, exactly one "
          f"per audited program (apex_top.sv:2540-2543). No datapath "
          f"expectation touched.")
    if off.cross_checks:
        for x in off.cross_checks:
            P(f"  fat-vs-thin     : {x['name']} fat acc == UNBATCHED thin "
              f"_tile_matvec acc -> {'EQUAL' if x['equal'] else 'DIFFERS'}")
    # ── the transport ledger: what the tile cost the HOST to drive ────────
    ck = off.cost
    if ck["programs"]:
        pol_exec = "hw" if args.mode == "hw" else "sim"
        P(f"  transport       : fat={off.fat} burst={off.burst} "
          f"rows_per_desc={off.rows_per_desc} "
          f"blocks_per_program={off.blocks_per_program if off.fat else 1}")
        P(f"                    {tg.rows_per_desc_note(image, args.d, executor=pol_exec)}")
        P(f"  projection cost : {ck['programs']:,} programs, {ck['blocks']:,} "
          f"8-column blocks, {ck['ops']:,} BAR0 ops of which "
          f"{ck['peeks']:,} PEEKS, {ck['bytes'] / 1e6:.1f} MB of regops")
        P(f"                    (MEASURED by counting the emitted programs. "
          f"Both executors poll before every push/fire, so a PEEK is the "
          f"expensive half: f2_host_run.py:165-173, sim_main.cpp:453-460.)")
    hk = lmh.cost
    if hk["programs"]:
        P(f"  lm_head cost    : {hk['programs']:,} programs, {hk['blocks']:,} "
          f"8-row blocks, {hk['ops']:,} BAR0 ops of which {hk['peeks']:,} "
          f"PEEKS, {hk['bytes'] / 1e6:.1f} MB of regops")
        P(f"                    (same fat multiblock descriptors as the "
          f"projections, {lmh.blocks_per_program} blocks behind ONE staging "
          f"of the head's single activation row)")

    # ── per-layer ledger ──────────────────────────────────────────────────
    P("")
    P("  PER-LAYER x PER-FAMILY LEDGER — served/total values, one row per")
    P("  offloaded layer. Every layer's operands are staged from THAT layer's")
    P("  own LayerWeights and verified per GEMM by _LayerState.next_proj,")
    P("  which refuses an operand that is not the tensor it claims to be.")
    P("  '*' marks a reconstruction (C-1 view); everything else is bit-exact")
    P("  against golden's own view of that op. '.' = left on the host.")
    cols = [op for op in OP_TYPES
            if any(off.by_layer.get(li, {}).get("family", {})
                   .get(op, {}).get("served") for li in layers)]
    head = (f"  {'layer':>5} "
            + " ".join(f"{lof.OP_LABEL[op].split(' ')[0][:13]:>14}"
                       for op in cols)
            + f" {'jobs':>6} {'caps':>8} {'exec s':>7} {'inv':>4} "
              f"{'pass':>4} {'checks':>8}")
    P("  " + "-" * (len(head) - 2))
    P(head)
    tot_jobs = tot_caps = tot_exec = tot_inv = 0
    tot_fam = {op: [0, 0] for op in cols}
    for li in layers:
        b = off.by_layer.get(li)
        if b is None:
            P(f"  {li:>5}  (layer never ran — the target step was not reached)")
            continue
        cells = []
        for op in cols:
            f = b["family"][op]
            tot_fam[op][0] += f["served"]
            tot_fam[op][1] += f["total"]
            if not f["served"]:
                cells.append(f"{'.':>14}")
            else:
                mark = "" if f["exact"] else "*"
                cells.append(f"{f['served']}/{f['total']}{mark}".rjust(14))
        P(f"  {li:>5} " + " ".join(cells)
          + f" {b['jobs']:>6} {b['caps']:>8} {b['executor_s']:>7.1f} "
            f"{b.get('invocations', 0):>4} {b.get('passes', 1):>4} "
            f"{b['checks_pass']}/{b['checks_total']:<6}")
        tot_jobs += b["jobs"]
        tot_caps += b["caps"]
        tot_exec += b["executor_s"]
        tot_inv += b.get("invocations", 0)
    P("  " + "-" * (len(head) - 2))
    P(f"  {'TOTAL':>5} "
      + " ".join(f"{tot_fam[op][0]}/{tot_fam[op][1]}".rjust(14)
                 for op in cols)
      + f" {tot_jobs:>6} {tot_caps:>8} {tot_exec:>7.1f} {tot_inv:>4} "
        f"{'':>4} {n_ok_layer}/{len(off.checks):<6}")
    P(f"  'inv' = entries into the executor (ssh/process start + upload + "
      f"attach), the")
    P(f"  cost the fitted hw model charges {4.8:.1f} s each; 'pass' = replays "
      f"of the layer")
    P(f"  needed before every program it asks for was already in the capture "
      f"pool.")
    if off.by_layer:
        per = [b["wall_s"] for b in off.by_layer.values()]
        P(f"  per-layer wall  : min {min(per):.1f}s  median "
          f"{sorted(per)[len(per) // 2]:.1f}s  max {max(per):.1f}s  "
          f"(MEASURED, {len(per)} layers, emit + audit + retarget + execute "
          f"+ grade)")
        P(f"  24-layer cost   : {24 * sorted(per)[len(per) // 2] / 3600:.2f} h "
          f"at the median above — EXTRAPOLATION, NOT RUN")

    # ── per-family ledger, summed over every offloaded layer ──────────────
    P("")
    P("  PER-FAMILY LEDGER — summed over every offloaded layer")
    P("  " + "-" * (W - 2))
    P(f"  {'op family':<26} {'who':<16} {'served/total values':<21} exact?")
    for op in OP_TYPES:
        a = agg[op]
        who = a["source"] if a["values"] else "HOST (golden)"
        cov = f"{a['values']}/{a['values_total']}"
        ex = ("—" if not a["values"] else
              ("BIT-EXACT" if a["exact"] else "reconstructed"))
        P(f"  {lof.OP_LABEL[op]:<26} {who:<16} {cov:<21} {ex}")
        P(f"      egress      : {l05.EGRESS[op]}")
        if a["values"]:
            P(f"      instances   : {a['served_instances']}/{a['instances']} "
              f"op calls served, {a['jobs']} tile programs, {a['caps']} caps, "
              f"{a['seconds']}s")
            P(f"      tile vs golden (same view): "
              f"{'bit-exact' if a['grade_ok'] else 'DIFFERS'}")
            for d in a["detail"][:1]:
                for k in ("max_abs_delta", "max_abs_delta_q78",
                          "downstream_c1_codes_n_diff",
                          "downstream_c1_scale_equal", "requant_identical",
                          "downstream_codes_match_golden",
                          "reassembled_equal_golden", "per_slice_pass",
                          "chunks", "rows", "slices", "blocks", "K"):
                    if k in d:
                        P(f"        {k:<32}: {d[k]}")
        for n in a["notes"][:4]:
            P(f"      NOTE: {n}")
    ha = lof._agg(recs, "lmhead")
    hres = lmh.result
    h_who = ha["source"] if ha["values"] else "HOST (golden)"
    h_cov = f"{ha['values']}/{ha['values_total'] or lmh.vocab}"
    h_ex = "—" if not ha["values"] else ("BIT-EXACT" if ha["exact"]
                                         else "DIFFERS")
    P(f"  {LmHeadOffloader.LABEL:<26} {h_who:<16} {h_cov:<21} {h_ex}")
    P(f"      egress      : {LmHeadOffloader.EGRESS}")
    if ha["values"]:
        P(f"      instances   : {ha['served_instances']}/{ha['instances']} "
          f"GEMM calls served, {ha['jobs']} tile programs, {ha['caps']} caps, "
          f"{ha['seconds']}s")
        P(f"      tile vs golden (same view): "
          f"{'bit-exact' if ha['grade_ok'] else 'DIFFERS'}")

    # ── the lm_head, in full ──────────────────────────────────────────────
    if hres is not None:
        P("")
        P("  LM-HEAD LEDGER — the last GEMM of the token")
        P("  " + "-" * (W - 2))
        P(f"  golden's own path : head_logits = C-1 quantize -> final RMSNorm "
          f"-> C-1 quantize")
        P(f"                      -> gemm_i8_ksplit(head_w8[{hres['vocab']}, "
          f"{hres['K']}], h8) -> x (s_h * s_head * fold)")
        P(f"                      (run_tinynpu.py:277-292). The lm_head is "
          f"ALREADY an INT8 GEMM in")
        P(f"                      golden, so NOTHING is re-quantized here: "
          f"the tile substitutes the")
        P(f"                      raw INT32 accumulators and every float — "
          f"both C-1 quantizations,")
        P(f"                      the norm, the dequant and the argmax — "
          f"stays golden's.")
        P(f"  exactness status  : BIT-EXACT INT32 accumulators, graded per "
          f"call against")
        P(f"                      cp.gemm_i8_ksplit on the same operands "
          f"(NOT a float claim)")
        P(f"  rows served       : {_fmt_int(hres['rows_served'])}/"
          f"{_fmt_int(hres['rows_total'])} vocabulary rows"
          + ("   [FULL WIDTH]" if hres["full_width"]
             else "   [SAMPLED — see --lmhead-rows]"))
        P(f"  operand check     : every GEMM operand verified to BE "
          f"head_w8[v0:v0+M] before staging")
        P(f"  tile programs     : {hk['programs']} fat programs, "
          f"{hk['blocks']} 8-row blocks, {ha['caps']} caps, "
          f"{hres['invocations']} executor invocation(s), "
          f"{hres['seconds']}s (emit + run + decode + grade; the same head "
          f"on the host costs {hres['host_seconds_golden_arm']}s)")
        hp = hres.get("predicted_hw_wall_s")
        if hp:
            P(f"  predicted hw wall : {hp['seconds']:.0f} s for this exact "
              f"program set — A PREDICTION, NOT A MEASUREMENT.")
            P(f"                      wall = {hp['model']['c_inv_s']}s x "
              f"{hp['measured_here']['invocations']} invocations + "
              f"{hp['model']['per_program_s'] * 1e3:.1f} ms x "
              f"{hp['measured_here']['programs']} programs +")
            P(f"                      {hp['model']['per_peek_s'] * 1e6:.1f} us "
              f"x {hp['measured_here']['peeks']:,} peeks  "
              f"(= {hp['invocation_setup_s']:.0f} + {hp['per_program_s']:.0f} "
              f"+ {hp['peek_s']:.0f} s)")
            P(f"                      coefficients from the COMMITTED "
              f"two-point silicon fit ({hp['source']});")
            P(f"                      only the counts are this run's own.")
        P(f"  logits            : max|dlogit| tile vs golden = "
          f"{hres['max_abs_dlogit_tile_vs_golden']:.4g}")
        P(f"  argmax            : tile {hres['argmax_tile']} vs golden "
          f"{hres['argmax_golden']} -> "
          f"{'SAME TOKEN' if hres['argmax_equal'] else 'DIFFERENT TOKEN'} "
          f"(top-1 margin {hres['top1_margin_golden']:.4g} over a "
          f"{hres['logit_range_golden']:.4g} range, on THIS arm's own hidden "
          f"state)")
        if hres["poison"] is not None:
            pz = hres["poison"]
            P(f"  discriminator     : the tile's accumulator for vocabulary "
              f"row {pz['row']}"
              + (" (golden's own top-1)" if pz["row_was_top1"] else "")
              + f" x{pz['k']}")
            P(f"                      logit {pz['logit_before']:.4g} -> "
              f"{pz['logit_after']:.4g}, argmax {hres['argmax_tile']} -> "
              f"{pz['argmax']} -> "
              f"{'TOKEN FLIPPED' if pz['token_flipped'] else 'NO FLIP'}")
            if not pz["row_was_top1"]:
                P(f"                      NOTE: this scope does not serve "
                  f"golden's top-1 row {hres['argmax_golden']}, so demoting "
                  f"the best")
                P(f"                      SERVED row cannot flip the token. "
                  f"What this arm proves is the")
                P(f"                      LOGIT-level claim; the token-level "
                  f"one needs --lmhead-rows -1.")
            P(f"                      (a UNIFORM scale cannot flip an argmax — "
              f"x -> Kx, K>0, is order-")
            P(f"                      preserving — so the discriminator that "
              f"MEANS something at the")
            P(f"                      head targets one row, and the row it "
              f"touched is named above.")
            P(f"                      Served from pass 2's own decoded "
              f"captures: no second executor")
            P(f"                      entry, the same tile values, one of "
              f"them scaled.)")

    # ── the FLOP share, with its derivation ──────────────────────────────
    P("")
    P("  FLOP SHARE — how much of this decode step's arithmetic the tile did")
    P("  " + "-" * (W - 2))
    P(f"  DENOMINATOR (measured by observing golden's own calls during the")
    P(f"  pure-host arm — MacCensus wraps gemm_i8_ksplit / attention_core /")
    P(f"  decoder_layer_fx and counts, it never substitutes):")
    for t, v in fs["per_tensor"].items():
        P(f"      {t}: {v['M']}x{v['K']}x{v['N']} per layer x {v['layers']} "
          f"layers = {_fmt_int(v['macs'])} MACs")
    P(f"      attention (Q·K̂ᵀ + P·V̂, all heads, all layers) = "
      f"{_fmt_int(fs['denominator_attn_macs'])} MACs")
    P(f"      step total ({fs['layers_in_step']} layers)            = "
      f"{_fmt_int(fs['denominator_step_macs'])} MACs")
    P(f"      lm_head (vocab x D_model, needed for the token) = "
      f"{_fmt_int(fs['lm_head_macs'])} MACs")
    P(f"      step + lm_head                                 = "
      f"{_fmt_int(fs['denominator_token_macs'])} MACs")
    P(f"  NUMERATOR (only accumulators the tile actually produced):")
    P(f"      projections: {_fmt_int(fs['proj_accumulators_served'])} of "
      f"{_fmt_int(fs['proj_accumulators_total'])} accumulators over "
      f"{len(layers)} layers")
    P(f"                   x each call's own K = "
      f"{_fmt_int(fs['tile_proj_macs'])} MACs"
      + ("   [SAMPLED — see --proj-cols]" if fs["proj_sampled"]
         else "   [FULL WIDTH]"))
    P(f"      attention  : {fs['attn_heads_served']} heads served x 2·T·D = "
      f"{_fmt_int(fs['tile_attn_macs'])} MACs")
    P(f"      layer total (the STEP numerator)               = "
      f"{_fmt_int(fs['tile_macs'])} MACs")
    P(f"      lm_head    : {_fmt_int(fs['lm_head_rows_served'])} of "
      f"{_fmt_int(fs['lm_head_rows_total'] or lmh.vocab)} vocabulary rows "
      f"x K={model.meta['D_model']}")
    P(f"                 = {_fmt_int(fs['tile_lmhead_macs'])} MACs"
      + ("   [HOST]" if not fs["lm_head_on_tile"] else
         ("   [SAMPLED — see --lmhead-rows]" if fs["lm_head_sampled"]
          else "   [FULL WIDTH]")))
    P(f"      token total (the STEP+lm_head numerator)       = "
      f"{_fmt_int(fs['tile_token_macs'])} MACs")
    P(f"  >>> TILE SHARE OF THE DECODE STEP        : "
      f"{fs['share_of_step']:.2f}%   "
      f"({'FULL-WIDTH' if not fs['proj_sampled'] else 'SAMPLED'} projections, "
      f"{len(layers)}/{model.n_layers} layers; the lm_head is NOT in this "
      f"ratio, either side)")
    P(f"  >>> TILE SHARE OF STEP + lm_head         : "
      f"{fs['share_of_token']:.2f}%   (lm_head: {fs['lm_head_mode']})")
    if fs["lm_head_on_tile"]:
        P(f"      of which the lm_head contributes "
          f"{100.0 * fs['tile_lmhead_macs'] / max(fs['denominator_token_macs'], 1):.2f}"
          f"pp; the layers contribute "
          f"{fs['share_of_token_layers_only']:.2f}pp")
    P(f"  Excluded from BOTH sides: elementwise work (RMSNorm, RoPE, SiLU,")
    P(f"  residual, softmax) is not multiply-accumulate arithmetic. Measured")
    P(f"  RMSNorm element-touches in this step: "
      f"{_fmt_int(fs['elem_touches_norm_in_step'])} "
      f"({100.0 * fs['elem_touches_norm_in_step'] / max(fs['denominator_step_macs'], 1):.3f}% "
      f"the size of the MAC count), so the exclusion cannot move the share")
    P(f"  by a meaningful amount in either direction.")
    if len(layers) < model.n_layers:
        per_layer_share = fs["share_of_step"] / max(len(layers), 1)
        all_layers_tok = (per_layer_share * model.n_layers
                          * fs["denominator_step_macs"]
                          / max(fs["denominator_token_macs"], 1))
        head_pp = (100.0 * fs["tile_lmhead_macs"]
                   / max(fs["denominator_token_macs"], 1))
        P(f"  EXTRAPOLATION (LABELLED AS SUCH, not measured): at this run's")
        P(f"  measured per-layer share of {per_layer_share:.2f}%, all "
          f"{model.n_layers} layers would be")
        P(f"  {per_layer_share * model.n_layers:.1f}% of the step and "
          f"{all_layers_tok:.1f}% of step+lm_head")
        P(f"  — plus this run's MEASURED lm_head contribution of "
          f"{head_pp:.1f}pp = {all_layers_tok + head_pp:.1f}% of the whole")
        P(f"  token. NOT RUN. (The lm_head half of that sum IS measured here; "
          f"the 24-layer half is not.)")

    # ── the checks ────────────────────────────────────────────────────────
    P("")
    n_ok = n_ok_pre
    P(f"  SUBSTITUTION / CONSUMPTION CHECKS: {n_ok}/{len(checks)} PASS "
      f"(the tile's value IS the value the model consumed)")
    for name, ok, detail in checks:
        if not ok:
            P(f"    FAIL  {name}  {detail}")
    if not checks:
        P("    (none — no op was served)")

    # ── tokens ────────────────────────────────────────────────────────────
    P("")
    P(f"  logit geometry    : pure-host top-1 margin {margin['host_margin']:.4g}"
      f" over a {margin['host_range']:.4g} logit range "
      f"({100 * margin['host_margin'] / margin['host_range']:.2f}% of range)")
    P(f"      max|dlogit| offload vs PURE host           = "
      f"{margin['dlogit']:.4g}   (tile + BUS_ON composition, mixed)")
    P(f"      max|dlogit| HOST+BUS_ON vs PURE host       = "
      f"{margin['dlogit_bus_vs_host']:.4g}   (the composition mode ALONE, no "
      f"tile)")
    P(f"      max|dlogit| offload vs HOST+BUS_ON         = "
      f"{margin['dlogit_vs_bus']:.4g}   (THE TILE ALONE — same composition "
      f"both sides)")
    P(f"  token OFFLOAD ON  : ids={on['ids']} text={txt(on['ids'])!r} "
      f"({on['seconds']:.0f}s)")
    P(f"  token PURE HOST   : ids={host['ids']} text={txt(host['ids'])!r} "
      f"({host['seconds']:.0f}s)")
    P(f"  token HOST+BUS_ON : ids={bus['ids']} text={txt(bus['ids'])!r} "
      f"({bus['seconds']:.0f}s, {bus['fired']} layer(s) composed BUS_ON) -> "
      f"{'same as pure host' if bus_ident else 'DIFFERS from pure host'}")
    dl = dctrl = None
    if pois is not None:
        pl = np.asarray(pois["logits"], dtype=np.float64).ravel()
        cl = np.asarray(ctrl["logits"], dtype=np.float64).ravel()
        dl = float(np.max(np.abs(pl - cl)))
        dctrl = float(np.max(np.abs(
            cl - np.asarray(host["logits"], dtype=np.float64).ravel())))
        verdict = ("substitutions ARE load-bearing" if dl > 0 else
                   "VACUOUS — the tile values never reached the token")
        P(f"  discriminator     : layer(s) {list(players)}, proj_cols="
          f"{args.poison_proj_cols}")
        P(f"      control (same layers offloaded, NOT poisoned): "
          f"ids={ctrl['ids']}, max|dlogit| vs pure host = {dctrl:.4g}")
        P(f"      poison  (every tile value x{args.poison}): "
          f"ids={pois['ids']}, max|dlogit| vs control = {dl:.4g}  ({verdict})")

    # ── verdict ───────────────────────────────────────────────────────────
    # THE TILE'S OWN QUESTION is `ident_bus`: does the tile reproduce the
    # token the host produces in the SAME composition? Every offloaded layer
    # must run C-LBUS BUS_ON to make the residual op reachable at all, and
    # that is a real (arbiter-blessed, D-030) change to the layer's host
    # arithmetic — run 3 measures it with no tile in the loop. Requiring
    # `ident` (vs the DEFAULT-bus host) when run 3 already shows the bus mode
    # alone moved the token would blame the tile for the composition.
    token_required = not recon_on
    bus_explains = (not ident) and (not bus_ident) and ident_bus
    # the lm_head's own gate: served rows must have been bit-exact AND the
    # argmax over the tile-computed logits must be golden's argmax. Its
    # substitution checks are already in `checks` (and therefore in
    # `checks_ok`); this is the named, printed condition.
    head_ok = (True if hres is None else
               bool(hres["rows_served"] and hres["argmax_equal"]
                    and hres["max_abs_dlogit_tile_vs_golden"] == 0.0
                    and ha["exact"] is not False
                    and (hres["poison"] is None
                         or (hres["poison"]["logit_moved"]
                             and (hres["poison"]["token_flipped"]
                                  or not hres["poison"]["row_was_top1"])))))
    ok = bool(fired and checks_ok and grades_ok and geom_ok and head_ok
              and (dl is None or dl > 0)
              and all(x["equal"] for x in off.cross_checks)
              and (not token_required or ident_bus)
              and (not token_required or ident or bus_explains))
    P("-" * W)
    if not fired:
        P("  FAIL: no op was served by the tile — token identity proves "
          "nothing.")
    if not checks_ok:
        P("  FAIL: a tile-derived value was not the value the model consumed.")
    if not grades_ok:
        P("  FAIL: a tile egress differs from golden's own view of the same "
          "op.")
    if not geom_ok:
        P("  FAIL: a program reached the tile framed at the wrong D.")
    if not head_ok:
        P("  FAIL: the lm_head stage did not hold — see the LM-HEAD LEDGER "
          "above. The")
        P("        accumulators must be bit-exact against golden's own "
          "gemm_i8_ksplit AND")
        P("        the argmax over the tile-computed logits must be golden's "
          "argmax.")
    if token_required and not ident_bus:
        P("  FAIL: the tile's token differs from the HOST run in the SAME "
          "composition,")
        P("        and every offloaded family here has EXACT egress — that "
          "is a tile defect,")
        P("        not the composition mode and not the disclosed C-1 cost.")
    if bus_explains:
        P("  NOTE: offload-on != pure-host, BUT the host-only BUS_ON run "
          "produces the")
        P("        SAME token as the offload run and the tile-vs-BUS_ON "
          f"logit delta is")
        P(f"        {margin['dlogit_vs_bus']:.4g}. The change came from the "
          f"C-LBUS composition every")
        P("        offloaded layer must use (apex_layer_deq.sv:90-92 refuses "
          "an ungraded")
        P("        fp32 composite, so the residual op is unreachable in "
          "golden's default")
        P(f"        BUS_OFF), not from a tile value. The pure-host top-1 "
          f"margin here is")
        P(f"        {margin['host_margin']:.4g} "
          f"({100 * margin['host_margin'] / margin['host_range']:.2f}% of "
          f"range) against a "
          f"{margin['dlogit_bus_vs_host']:.4g} composition-only delta.")
    if not ident and not token_required:
        P("  NOTE: the token changed. --include-reconstructed is on, so the "
          "C-1 re-entry")
        P("        of RMSNorm-2/SwiGLU (B-FEED-WIDTH) is in play as well as "
          "the composition")
        P("        mode: every family above is still bit-exact against "
          "golden's own view of")
        P("        that op, so no tile value is wrong. Graded at the logit "
          "level, as declared.")
    P(f"  OP FAMILIES SERVED BY THE TILE: {len(served_ops)}/6 "
      f"({', '.join(served_ops) or 'none'})")
    P(f"  LAYERS SERVED BY THE TILE    : {len(off.by_layer)}/"
      f"{model.n_layers}  {sorted(off.by_layer)}")
    P(f"  TILE SHARE OF THE STEP       : {fs['share_of_step']:.2f}% of "
      f"multiply-accumulates")
    P(f"  TILE SHARE OF STEP + lm_head : {fs['share_of_token']:.2f}%   "
      f"(lm_head: {fs['lm_head_mode']})")
    if hres is not None:
        P(f"  LM_HEAD ON THE TILE          : "
          f"{'PASS' if head_ok else 'FAIL'}   "
          f"({_fmt_int(hres['rows_served'])}/{_fmt_int(hres['rows_total'])} "
          f"vocabulary rows, BIT-EXACT INT32 accumulators, argmax "
          f"{'==' if hres['argmax_equal'] else '!='} golden's)")
    P(f"  TOKEN == HOST IN SAME BUS    : {'PASS' if ident_bus else 'FAIL'}"
      f"   (the tile's own question)"
      + ("" if token_required else "   [reported, NOT a pass condition]"))
    P(f"  TOKEN == PURE HOST           : {'PASS' if ident else 'FAIL'}"
      f"   (offload-on == default-bus golden)"
      + ("" if token_required else "   [reported, NOT a pass condition]"))
    P(f"  P2 STAGE 3                   : {'PASS' if ok else 'FAIL'}")
    P("=" * W)
    P("  WHAT THE HOST STILL DID")
    for line in host_ledger(model, args, agg, image, layers, fs):
        P(f"    - {line}")
    P("=" * W)

    out = work / "prompt05b_result.json"
    out.write_text(json.dumps({
        "prompt": args.prompt, "prompt_ids": ids, "mode": args.mode,
        "image": image.__dict__ | {"info_tier": image.info_tier},
        "d": args.d, "layers": list(layers), "step": step,
        "scope": scope.__dict__,
        "include_reconstructed": bool(recon_on),
        "model": {k: model.meta.get(k) for k in
                  ("model", "H", "H_kv", "head_dim", "D_model", "d_ffn")},
        "n_layers": model.n_layers,
        "tokens_on": on["ids"], "tokens_host": host["ids"],
        "tokens_bus_on": bus["ids"],
        "text_on": txt(on["ids"]), "text_host": txt(host["ids"]),
        "token_identity": ident, "token_identity_required": token_required,
        "token_identity_vs_bus_on": ident_bus,
        "bus_on_token_identity": bus_ident,
        "bus_mode_explains_difference": bus_explains,
        "op_families_served": served_ops,
        "layers_served": sorted(off.by_layer),
        "stage3_pass": ok,
        "lm_head": (None if hres is None else
                    hres | {"pass": head_ok, "rows_requested": args.lmhead_rows,
                            "blocks_per_program": lmh.blocks_per_program,
                            "exactness": "BIT-EXACT INT32 accumulators vs "
                                         "cp.gemm_i8_ksplit on the same "
                                         "operands; every float (both C-1 "
                                         "quantizations, the final RMSNorm, "
                                         "the dequant, the argmax) stays "
                                         "golden's"}),
        "flop_share": fs,
        "per_layer": {str(li): {k: v for k, v in b.items()
                                if k not in ("records", "checks")}
                      for li, b in off.by_layer.items()},
        "per_layer_ops": {
            str(li): [{"op": r.op, "name": r.name, "n_served": r.n_served,
                       "n_total": r.n_total, "jobs": r.jobs, "caps": r.caps,
                       "exact": r.exact, "grade_ok": r.grade_ok}
                      for r in b["records"]]
            for li, b in off.by_layer.items()},
        "per_layer_wall_s": {str(li): b["wall_s"]
                             for li, b in off.by_layer.items()},
        "geometry_audit": {"audit_passes": runner.n_audited,
                           "programs": len(runner.audited_keys),
                           "executed_distinct": len(runner.executed_keys),
                           "unaudited_executed": unaudited,
                           "desc_k_seen": sorted(runner.desc_k_seen),
                           "pass": geom_ok},
        "info_tier_retargets": runner.n_retargeted,
        "batching_cross_checks": off.cross_checks,
        "transport": {"fat": off.fat, "burst": off.burst,
                      "rows_per_desc": off.rows_per_desc,
                      "blocks_per_program": off.blocks_per_program,
                      "batch_size": off.batch_size,
                      "image_agfi": image.agfi,
                      "rows_per_desc_rule": tg.rows_per_desc_note(
                          image, args.d,
                          executor=("hw" if args.mode == "hw" else "sim")),
                      "projection_cost": off.cost,
                      **lof.transport_record(runner, on)},
        "poison": ({"k": args.poison, "layers": list(players),
                    "proj_cols": args.poison_proj_cols,
                    "ids_control": ctrl["ids"], "ids_poison": pois["ids"],
                    "max_abs_dlogit_poison_vs_control": dl,
                    "max_abs_dlogit_control_vs_host": dctrl}
                   if pois is not None else None),
        "logit_geometry": margin,
        "ledger": agg,
        "checks_pass": n_ok, "checks_total": len(checks),
        "checks_failed": [{"name": n, "detail": str(d)}
                          for n, ok_, d in checks if not ok_],
        "ops": [{"op": r.op, "name": r.name, "source": r.source,
                 "n_served": r.n_served, "n_total": r.n_total, "jobs": r.jobs,
                 "caps": r.caps, "seconds": r.seconds, "exact": r.exact,
                 "grade_ok": r.grade_ok, "notes": r.notes} for r in recs],
        "seconds": {"on": on["seconds"], "host": host["seconds"],
                    "bus_on": bus["seconds"], "on_wall": on["wall"],
                    "executor": round(runner.wall, 1)},
        "tile_jobs": runner.n_jobs, "tile_caps": runner.n_caps,
        "pruned_bytes": off._pruned_bytes + runner.pruned_bytes,
        "git": rt.git_rev(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))
    P(f"  record -> {out}")
    return 0 if ok else 1


def host_ledger(model, args, agg, image, layers, fs) -> list:
    d = image.cfg_dm
    rest = model.n_layers - len(layers)
    if fs["lm_head_on_tile"]:
        head_line = (
            "the lm_head's HOST HALF: the tokenizer, the embedding lookup, "
            "both C-1 quantizations feeding the head, the final RMSNorm, the "
            "s_h*s_head*fold dequant of the tile's INT32 accumulators and the "
            f"argmax itself. The {_fmt_int(fs['lm_head_macs'])}-MAC GEMM "
            f"({100.0 * fs['lm_head_macs'] / max(fs['denominator_token_macs'], 1):.1f}% "
            f"of the token's arithmetic) is the tile's for "
            f"{_fmt_int(fs['lm_head_rows_served'])} of "
            f"{_fmt_int(fs['lm_head_rows_total'])} vocabulary rows")
    else:
        head_line = (
            "tokenizer, embedding lookup, the lm_head logits and the argmax "
            "(tie_word_embeddings: the head IS the embedding, handled at "
            f"prepare) — the lm_head alone is "
            f"{_fmt_int(fs['lm_head_macs'])} MACs, "
            f"{100.0 * fs['lm_head_macs'] / max(fs['denominator_token_macs'], 1):.1f}% "
            f"of the token's arithmetic")
    out = [
        head_line,
        f"the {rest} decoder layer(s) outside --layers, entirely on the host",
        "every decode step other than the offloaded one",
        "all data movement: weights, the KV rows, every operand staged into "
        "the tile and every value read back",
        "the C-2 requant epilogues (calib_requant + requant_i32_to_i8) of the "
        "attention output, the o-projection and the down-projection — the "
        "standing fence: the tile returns raw INT32, the host calibrates",
        f"the attention path's score-dequant composites (s_q*s_k/sqrt(D), "
        f"exact at D={image.cfg_d}) are host-computed and pushed as CSRs; the "
        f"online softmax itself runs on the tile's ASU",
        f"the per-token whole-row C-1 quantizations that feed each op "
        f"(quant_rows_i8 over all {model.meta['D_model']} elements, ONE amax) "
        f"— the tile's C-1 feeder is elaborated at CFG_DM={d}, so it frames "
        f"the same values as {model.meta['D_model'] // d} x {d} rows",
    ]
    if agg["norm"]["values"] == 0:
        out.append("BOTH RMSNorms of every offloaded layer (--ops does not "
                   "include 'norm': its C-1 re-entry is the B-FEED-WIDTH cost)")
    if agg["swiglu"]["values"] == 0:
        out.append("the SwiGLU product of every offloaded layer (--ops does "
                   "not include 'swiglu', same reason)")
    if agg["rope"]["values"]:
        out.append("the RoPE of the CACHED K rows: the rotated K must reach "
                   "the KVQ codec as exact fp16 bits and the tile's q_sink "
                   "egress is a C-1 view (B-ROPE-EGRESS) — decode-token q "
                   "rows only")
    if fs["proj_sampled"]:
        out.append(f"the projection accumulators outside --proj-cols "
                   f"{args.proj_cols} / --proj-rows {args.proj_rows}: "
                   f"{fs['proj_accumulators_total'] - fs['proj_accumulators_served']} "
                   f"of {fs['proj_accumulators_total']}")
    if fs["lm_head_sampled"]:
        out.append(f"the lm_head rows outside --lmhead-rows "
                   f"{args.lmhead_rows}: "
                   f"{fs['lm_head_rows_total'] - fs['lm_head_rows_served']} "
                   f"of {fs['lm_head_rows_total']} vocabulary rows")
    return out


# ═══════════════════════════ selftest (tiny model) ═════════════════════════

def selftest(args) -> int:
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("PROMPT05B SELFTEST (tiny head_dim=64 GQA model, no executor)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model = l05._tiny(td / "w")
        ids, work = [3, 1, 4, 1, 5], td / "work"
        tier, L = at.TIER_CQ8, model.n_layers
        dm, dff = model.meta["D_model"], model.meta["d_ffn"]
        H = model.meta["H"]
        tstep = len(ids) - 1

        # --- the layer-set parser ------------------------------------------
        chk("--layers 'all' spans the model", parse_layers("all", 24)
            == tuple(range(24)))
        chk("--layers '0-3,7' parses to a sorted set",
            parse_layers("0-3,7", 24) == (0, 1, 2, 3, 7))
        for bad, why in (("99", "out of range"), ("3-1", "empty range"),
                         ("x", "not a layer")):
            try:
                parse_layers(bad, 24)
                chk(f"--layers {bad!r} is REFUSED ({why})", False, "no refusal")
            except SystemExit as e:
                chk(f"--layers {bad!r} is REFUSED ({why})", "REFUSE" in str(e))

        # --- the MAC census, against an independently computed truth -------
        with MacCensus(L) as census:
            base = po.decode(model, ids, 1, tier, 128, set(), None,
                             verbose=False)
        st = census.step_macs(tstep)
        chk("the census saw every layer of the step", st["layers_seen"] == L,
            f"{st['layers_seen']} of {L}")
        # golden's own shapes: q/o/g/u/d are 1 row, k/v are T rows, and
        # X is [t+2, D] so T = tstep + 1.
        T = tstep + 1
        want_proj = L * (1 * dm * dm + T * dm * (dm // H * model.meta["H_kv"])
                         * 2 + 1 * dm * dm + 2 * dm * dff + dff * dm)
        chk("per-layer projection MACs == the shapes golden actually used",
            st["proj"] == want_proj, f"census {st['proj']} vs {want_proj}")
        want_head = int(model.head_w8.shape[0]) * dm
        chk("the lm_head is counted OUTSIDE the layers",
            census.outside_layer_macs == want_head,
            f"{census.outside_layer_macs} vs {want_head}")
        chk("attention MACs are 2·T·D per head per layer",
            st["attn"] == L * H * 2 * T * (dm // H),
            f"{st['attn']} vs {L * H * 2 * T * (dm // H)}")
        chk("the census is READ-ONLY (same token as an unwrapped decode)",
            base["ids"] == po.decode(model, ids, 1, tier, 128, set(), None,
                                     verbose=False)["ids"])
        chk("the census cross-checks against the model's own weight shapes",
            census.verify_against(model, tstep)["proj_macs"] == want_proj)

        # --- the recursion trap: N > DIM_MAX splits by concatenation THROUGH
        #     compute.py's own module global, i.e. through this very wrapper.
        big = MacCensus(L)
        M, K, N = 2, cp.K_MAX + 7, cp.DIM_MAX + 601
        A8 = np.ones((M, K), dtype=np.int64)
        B8 = np.ones((K, N), dtype=np.int64)
        with big:
            ref = cp.gemm_i8_ksplit(A8, B8)
        chk("a K>K_MAX, N>DIM_MAX GEMM is counted ONCE at M*N*K "
            "(the ksplit recursion trap)",
            len(big.gemm) == 1 and big.gemm[0]["macs"] == M * N * K
            and big.inner_calls >= 2,
            f"rows={len(big.gemm)} macs={big.gemm[0]['macs'] if big.gemm else 0}"
            f" vs {M * N * K}, inner={big.inner_calls}")
        chk("...and the wrapped call still returns golden's own result",
            ref.shape == (M, N) and int(ref[0, 0]) == K)

        # --- a multi-layer dry run -----------------------------------------
        scope = lof.Scope(ops=EXACT_FAMILIES, resid_slice=dm,
                          proj_cols=FULL, proj_rows=FULL)
        runner = l05.Runner05B("golden", work, d=64)
        layers = (0, 1)
        with l05.d64_emission(64):
            with MultiOffloader05B(n_layers=L, mode="golden", work=work,
                                   scope=scope, runner=runner, step=tstep,
                                   layers=layers, verbose=False) as off:
                on = po.decode(model, ids, 1, tier, 128, set(), None,
                               verbose=False)
        chk("every geometry rebind restored on exit",
            (lof.D_TILE, gl.D_TILE) == l05._FROZEN_FRAMES
            and gl.build_norm2_chunked is l05._FROZEN_NORM)
        chk("every golden rebind restored on exit",
            tf.attention_core is at.attention_core
            and tf.gemm_i8_ksplit is cp.gemm_i8_ksplit)
        chk("BOTH selected layers were armed and accounted",
            sorted(off.by_layer) == list(layers), str(sorted(off.by_layer)))
        chk("token identity for the four exact families over 2 layers",
            on["ids"] == base["ids"], f"{on['ids']} vs {base['ids']}")
        served = {op for r in off.records if r.n_served for op in [r.op]}
        chk("exactly the four exact-egress families were served",
            served == set(EXACT_FAMILIES), str(sorted(served)))
        chk("no reconstructed family ran without --include-reconstructed",
            not any(r.n_served for r in off.records
                    if r.op in RECON_FAMILIES))
        chk("every substitution/consumption check passed over both layers",
            all(ok for _n, ok, _d in off.checks),
            f"{sum(1 for _n, ok, _d in off.checks if ok)}/{len(off.checks)}")
        per_layer_recs = {li: b["records"] for li, b in off.by_layer.items()}
        chk("records are attributed to the layer that produced them",
            all(f"L{li:02d}" in r.name for li, rs in per_layer_recs.items()
                for r in rs), "record names carry their layer")
        a0 = lof._agg(per_layer_recs[0], "proj")
        chk("--proj-cols FULL serves EVERY projection accumulator of a layer",
            a0["values"] == a0["values_total"] and a0["values"] > 0,
            f"{a0['values']}/{a0['values_total']}")

        # --- accounting must attach to the ARMED step, not the last one ----
        early = 1
        assert early != tstep
        with l05.d64_emission(64):
            with MultiOffloader05B(n_layers=L, mode="golden", work=work,
                                   scope=scope, runner=runner, step=early,
                                   layers=layers, verbose=False) as offe:
                po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
        chk("per-layer accounting attaches to the ARMED step, not the last "
            "step of the run",
            sorted(offe.by_layer) == list(layers)
            and all(b["records"] and b["checks"]
                    for b in offe.by_layer.values()),
            {li: len(b["records"]) for li, b in offe.by_layer.items()})
        chk("...and the records it holds are that step's",
            all(f"_s{early:03d}_" in r.name
                for b in offe.by_layer.values() for r in b["records"]))

        # --- the FLOP share on that dry run --------------------------------
        fs = flop_share(census, tstep, off.records)
        chk("the FLOP-share denominator is the whole step + lm_head",
            fs["denominator_token_macs"] == st["proj"] + st["attn"] + want_head)
        chk("the numerator counts only the offloaded layers",
            fs["tile_proj_macs"] == want_proj // L * len(layers),
            f"{fs['tile_proj_macs']} vs {want_proj // L * len(layers)}")
        chk("full-width projections are NOT flagged sampled",
            fs["proj_sampled"] is False)
        chk("the share is layers/L of the layer arithmetic (sanity)",
            abs(fs["share_of_step"]
                - 100.0 * (want_proj // L * len(layers)
                           + fs["tile_attn_macs"])
                / (st["proj"] + st["attn"])) < 1e-9,
            f"{fs['share_of_step']:.3f}%")

        # --- a SAMPLED scope must show up as sampled, with a smaller share --
        scope8 = lof.Scope(ops=EXACT_FAMILIES, resid_slice=dm, proj_cols=8,
                           proj_rows=1)
        with l05.d64_emission(64):
            with MultiOffloader05B(n_layers=L, mode="golden", work=work,
                                   scope=scope8, runner=runner, step=tstep,
                                   layers=layers, verbose=False) as off8:
                po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
        fs8 = flop_share(census, tstep, off8.records)
        chk("a sampled projection scope is REPORTED as sampled",
            fs8["proj_sampled"] is True)
        chk("...and its share is strictly smaller than full width",
            fs8["share_of_step"] < fs["share_of_step"],
            f"{fs8['share_of_step']:.3f}% vs {fs['share_of_step']:.3f}%")

        # --- per-layer poison ----------------------------------------------
        with l05.d64_emission(64):
            with MultiOffloader05B(n_layers=L, mode="golden", work=work,
                                   scope=scope, runner=runner, step=tstep,
                                   layers=layers, poison=0.5,
                                   poison_layers=(1,), verbose=False) as offp:
                pr = po.decode(model, ids, 1, tier, 128, set(), None,
                               verbose=False)
        dl = float(np.max(np.abs(pr["logits"] - on["logits"])))
        chk("poisoning ONE layer of a multi-layer run perturbs the logits",
            dl > 0, f"max|dlogit|={dl:.4g}")
        chk("...and the poison was applied to that layer only",
            offp.by_layer[1]["poisoned"] and not offp.by_layer[0]["poisoned"])

        # --- include-reconstructed reaches the two C-1 families ------------
        scope6 = lof.Scope(ops=OP_TYPES, resid_slice=dm, proj_cols=8)
        with l05.d64_emission(64):
            with MultiOffloader05B(n_layers=L, mode="golden", work=work,
                                   scope=scope6, runner=runner, step=tstep,
                                   layers=layers, verbose=False) as off6:
                po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
        got6 = {r.op for r in off6.records if r.n_served}
        chk("--include-reconstructed serves all six families over both layers",
            got6 == set(OP_TYPES), str(sorted(got6)))
        by6 = {op: lof._agg(off6.records, op) for op in OP_TYPES}
        chk("RMSNorm-2 covers the full row of BOTH layers",
            by6["norm"]["values"] == 2 * dm, f"{by6['norm']['values']}")
        chk("SwiGLU covers the full d_ffn of BOTH layers",
            by6["swiglu"]["values"] == 2 * dff)

        # --- THE LM_HEAD ---------------------------------------------------
        # golden must actually HAVE the integer head path this stage claims
        # to be substituting into; if run_tinynpu ever computes the logits in
        # float, everything below is a different claim and must not pass.
        import inspect as _insp
        src = _insp.getsource(rt.GoldenModel.head_logits)
        chk("golden's OWN lm_head is an INT8 GEMM (gemm_i8_ksplit over "
            "head_w8), so the tile substitutes accumulators rather than "
            "inventing a quantization",
            "gemm_i8_ksplit" in src and "head_w8" in src
            and "quant_rows_i8" in src,
            "run_tinynpu.GoldenModel.head_logits")
        vocab = int(np.asarray(model.head_w8).shape[0])

        def _head_run(rows, poison=None, mode="golden"):
            r2 = l05.Runner05B(mode, work, d=64)
            with l05.d64_emission(64):
                with LmHeadOffloader(model, mode=mode, work=work / "lmh",
                                     runner=r2, rows=rows, poison=poison,
                                     verbose=False) as h:
                    d = po.decode(model, ids, 1, tier, 128, set(), None,
                                  verbose=False)
            return d, h

        d_full, h_full = _head_run(-1)
        chk("the lm_head seam restores BOTH names it rebinds",
            cp.gemm_i8_ksplit is tf.gemm_i8_ksplit
            and rt.GoldenModel.head_logits.__qualname__
            == "GoldenModel.head_logits")
        chk("--lmhead-rows -1 serves EVERY vocabulary row of the head",
            h_full.result["rows_served"] == vocab
            and h_full.result["full_width"],
            f"{h_full.result['rows_served']}/{vocab}")
        chk("the head's accumulators are bit-exact and the token is unchanged",
            h_full.result["max_abs_dlogit_tile_vs_golden"] == 0.0
            and h_full.result["argmax_equal"]
            and d_full["ids"] == base["ids"],
            f"max|dlogit|={h_full.result['max_abs_dlogit_tile_vs_golden']}")
        chk("every head check passed",
            all(ok for _n, ok, _d in h_full.checks),
            f"{sum(1 for _n, ok, _d in h_full.checks if ok)}/"
            f"{len(h_full.checks)}")
        chk("the head GEMM's operands are verified to BE head_w8 slices "
            "(every call, before staging)",
            h_full._cursor == vocab and h_full.n_gemm_calls > 0,
            f"cursor {h_full._cursor}, {h_full.n_gemm_calls} calls")

        d_off, h_off = _head_run(0)
        chk("--lmhead-rows 0 leaves the whole head on the host",
            h_off.result is None and not h_off.records
            and d_off["ids"] == base["ids"])

        n_s = 8 * (vocab // 16)
        d_s, h_s = _head_run(n_s)
        chk("a SAMPLED head serves exactly the rows asked for and says so",
            h_s.result["rows_served"] == n_s and h_s.result["sampled"]
            and d_s["ids"] == base["ids"],
            f"{h_s.result['rows_served']}/{vocab}")

        fs_h = flop_share(census, tstep, off.records + h_full.records)
        fs_nh = flop_share(census, tstep, off.records)
        chk("the lm_head enters the STEP+lm_head numerator and NOT the STEP "
            "one",
            fs_h["share_of_step"] == fs_nh["share_of_step"]
            and fs_h["share_of_token"] > fs_nh["share_of_token"],
            f"step {fs_h['share_of_step']:.2f}% (unchanged), token "
            f"{fs_nh['share_of_token']:.2f}% -> {fs_h['share_of_token']:.2f}%")
        chk("...counted at rows x D_model, matching the census's own lm_head "
            "denominator",
            fs_h["tile_lmhead_macs"] == vocab * dm == fs_h["lm_head_macs"],
            f"{fs_h['tile_lmhead_macs']} vs {vocab * dm}")
        fs_s = flop_share(census, tstep, off.records + h_s.records)
        chk("a sampled head is REPORTED sampled and scores strictly lower",
            fs_s["lm_head_sampled"] and not fs_h["lm_head_sampled"]
            and fs_s["share_of_token"] < fs_h["share_of_token"],
            f"{fs_s['lm_head_mode']}")

        # the discriminator, and WHY it targets one row
        _dg, h_p = _head_run(-1, poison=0.5)
        pz = h_p.result["poison"]
        chk("the lm_head discriminator flips the emitted token (the tile's "
            "accumulator IS what the argmax consumed)",
            pz["token_flipped"] and pz["row_was_top1"],
            f"row {pz['row']} x{pz['k']}: argmax "
            f"{h_p.result['argmax_tile']} -> {pz['argmax']}")
        gl_ = np.asarray(d_full["logits"], dtype=np.float64).ravel()
        chk("...and a UNIFORM scale would NOT: x -> Kx (K>0) preserves the "
            "argmax, which is why the head's poison targets a row",
            int(np.argmax(gl_ * 0.5)) == int(np.argmax(gl_)))

        # --- the TRANSPORT SHAPE: fat vs thin emission (host only) ---------
        # `_gemm` short-circuits to golden in dry-run mode, so the emitters
        # are exercised here directly: same operands, same scope, both
        # shapes, and the costs are COUNTED off the emitted files.
        import types as _t
        rng = np.random.default_rng(20260804)
        K, NB = 896, 6
        xa = rng.integers(-100, 101, (1, K), dtype=np.int64)
        xa[0, 11] = 127
        Wb = rng.integers(-9, 9, (K, 8 * NB), dtype=np.int64)
        fake_c = _t.SimpleNamespace(tag="st_s000_L00")

        def _emit(fat, burst, rpd, bpp=128):
            o = MultiOffloader05B(n_layers=L, mode="golden", work=work / "tx",
                                  scope=scope, runner=runner, step=tstep,
                                  layers=layers, verbose=False, fat=fat,
                                  burst=burst, rows_per_desc=rpd,
                                  blocks_per_program=bpp)
            tag = f"f{int(fat)}b{int(burst)}r{rpd}"
            fake_c.tag = f"st_{tag}_L00"
            with l05.d64_emission(64):
                progs, notes = o._plan_programs(xa, Wb, fake_c, "Wq",
                                                1, 8 * NB)
            cost = {"ops": 0, "peeks": 0, "bytes": 0}
            for p in progs:
                c = gj.program_cost(p["path"])
                for k in cost:
                    cost[k] += c[k]
            return progs, cost, notes

        p_fat, c_fat, n_fat = _emit(True, True, 31)
        p_thin, c_thin, n_thin = _emit(False, False, 31)
        chk("the FAT emission is ONE program covering every block; the THIN "
            "one is a program per block",
            len(p_fat) == 1 and p_fat[0]["blocks"] == NB
            and len(p_thin) == NB and not (n_fat or n_thin),
            f"fat {len(p_fat)} program(s), thin {len(p_thin)}")
        p_leg, c_leg, _ = _emit(False, False, 1)
        with l05.d64_emission(64):
            n_rows = gj.stage_plan(xa[0], Wb[:, :8]).rows
        chk("rows_per_desc=1 (what a PRE-38ec95c image is held to) costs one "
            "program per staged ROW per block",
            len(p_leg) == NB * n_rows and len(p_leg) > len(p_thin),
            f"{len(p_leg)} programs = {NB} blocks x {n_rows} single-row "
            f"K-chunks (vs {len(p_thin)} at the allowed ceiling)")
        chk("...and the blocks are attributed in the SAME order in both",
            [u["c0"] for u in p_fat[0]["units"]]
            == [p["units"][0]["c0"] for p in p_thin]
            == [8 * i for i in range(NB)])
        chk("fat costs fewer BAR0 ops and far fewer PEEKS for the same blocks",
            c_fat["ops"] < c_thin["ops"] and c_fat["peeks"] < c_thin["peeks"],
            f"ops {c_thin['ops']:,} -> {c_fat['ops']:,} "
            f"({c_thin['ops'] / c_fat['ops']:.2f}x), peeks "
            f"{c_thin['peeks']:,} -> {c_fat['peeks']:,} "
            f"({c_thin['peeks'] / c_fat['peeks']:.1f}x)")
        p_nb, c_nb, _ = _emit(True, False, 31)
        chk("the push burst is what removes the peeks (fat alone does not)",
            c_nb["peeks"] > c_fat["peeks"] and c_nb["ops"] <= c_thin["ops"],
            f"fat-no-burst {c_nb['peeks']:,} peeks vs fat+burst "
            f"{c_fat['peeks']:,}")
        p_grp, _c, _n = _emit(True, True, 31, bpp=2)
        chk("--blocks-per-program bounds the program size",
            len(p_grp) == 3 and all(p["blocks"] == 2 for p in p_grp),
            f"{len(p_grp)} programs of {[p['blocks'] for p in p_grp]} blocks")
        chk("the per-image rows_per_desc ceiling is the smaller of K_MAX/d "
            "and STAGE_R_MAX, and the LEGACY image still gets 1",
            tg.rows_per_desc_for(tg.IMG_05B, 64, executor="hw") == 31
            and tg.rows_per_desc_for(tg.IMG_B128, 128, executor="hw",
                                     agfi="agfi-0ae06ea568e5667ba") == 1)

        # --- an unattached hw run must REFUSE, not run empty ---------------
        import argparse as _ap
        ns = _ap.Namespace(executor="hw", dry_run=False, layers="0",
                           max_tokens=1, d=64, ids=ids, prompt="x",
                           offload_step=tstep, tier="kvq8", group=128,
                           ops=EXACT_FAMILIES, proj_cols=8, proj_rows=1,
                           heads=-1, rope_heads=-1, resid_cols=-1,
                           resid_slice=-1, norm_dm=-1, norm1_rows=0,
                           swiglu_chunks=-1, poison=None, poison_layers=None,
                           poison_proj_cols=8, batch_size=8,
                           cross_check=False, keep_jobs=False, binary=None,
                           tile_div=5, slot=0, timeout_s=60,
                           hw_host=None, hw_key=None,
                           fat=True, burst=True, blocks_per_program=128,
                           rows_per_desc=-1, collapse=True,
                           lmhead_rows=0, lmhead_blocks_per_program=128,
                           lmhead_poison=None,
                           max_batch_mb=lof.MAX_BATCH_MB_DEFAULT,
                           max_passes=lof.MAX_PASSES_DEFAULT)
        import os as _os
        saved = {k: _os.environ.pop(k, None)
                 for k in ("APEX_F2_HOST", "APEX_F2_KEY")}
        try:
            rc = run_multi(model, None, ns, work / "hwrefuse")
        finally:
            for k, v in saved.items():
                if v is not None:
                    _os.environ[k] = v
        chk("--executor hw without a remote config REFUSES (the "
            "zero-captures trap)", rc == 2, f"rc={rc}")

    print("=" * 60)
    if fails:
        print(f"PROMPT05B SELFTEST FAIL: {fails}")
        return 1
    print("PROMPT05B SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="P2 stage 3: serve MANY Qwen2.5-0.5B decoder layers from "
                    "the D=64 tile during a real prompt run, with a measured "
                    "FLOP share.")
    ap.add_argument("--prompt")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--layers", default="0-3",
                    help="'all', '0-23', '0,3,7' or a mix")
    ap.add_argument("--offload-step", type=int, default=None,
                    help="decode step to offload (default: the last prefill "
                         "step, whose output reaches the emitted logits)")
    ap.add_argument("--d", type=int, default=l05.D_TILE_05B)
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-reconstructed", action="store_true",
                    help="ALSO offload RMSNorm-2 and SwiGLU. They re-enter "
                         "through the 64-wide C-1 feeder (B-FEED-WIDTH), so "
                         "the token MAY legitimately differ and the run is "
                         "graded at the logit level.")
    ap.add_argument("--ops", default=None,
                    help="explicit op-family list (overrides the default "
                         "four exact-egress families)")
    ap.add_argument("--proj-cols", type=int, default=-1,
                    help="output columns per projection call (-1 = FULL "
                         "WIDTH, the default)")
    ap.add_argument("--proj-rows", type=int, default=-1,
                    help="activation rows per projection call (-1 = all)")
    ap.add_argument("--no-collapse", dest="collapse", action="store_false",
                    help="ONE EXECUTOR INVOCATION PER OP — the pre-collapse "
                         "transport, kept so the before/after is measured on "
                         "the same job set by the same code")
    ap.add_argument("--max-batch-mb", type=int,
                    default=lof.MAX_BATCH_MB_DEFAULT,
                    help="regops carried by one executor invocation when "
                         "collapsing (default: the largest payload the flown "
                         "shape already carried in one invocation)")
    ap.add_argument("--max-passes", type=int, default=lof.MAX_PASSES_DEFAULT,
                    help="replays of a layer allowed before the harness "
                         "REFUSES for lack of a stable program set")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="programs per executor invocation (batch_exec)")
    ap.add_argument("--cross-check", action="store_true",
                    help="re-run one projection block through the OLD thin "
                         "path and assert identical INT32 accumulators")
    # ── transport shape (host-side only; no descriptor semantics change) ──
    ap.add_argument("--no-fat", dest="fat", action="store_false",
                    help="the 2026-08-03 shape: one program per 8-column "
                         "block, re-staging the activation every time")
    ap.add_argument("--blocks-per-program", type=int, default=128,
                    help="8-column blocks packed behind ONE activation "
                         "staging (fat path)")
    ap.add_argument("--no-burst", dest="burst", action="store_false",
                    help="emit a per-push free-slot PEEK instead of one "
                         "FIFO-empty poll per mailbox burst")
    ap.add_argument("--rows-per-desc", type=int, default=-1,
                    help="staged rows per descriptor (-1 = the per-image "
                         "ceiling from tile_geom's base-row allowlist)")
    # ── the lm_head: the token's last 136 M MACs ──────────────────────────
    ap.add_argument("--lmhead-rows", type=int, default=0, metavar="N",
                    help="vocabulary rows of the lm_head GEMM the tile serves. "
                         "0 (default) leaves the whole head on the host; -1 is "
                         "FULL WIDTH (all 151,936 rows at 0.5B); N serves the "
                         "first N rows and the share line says SAMPLED. Full "
                         "width is ~19k 8-row blocks / ~1.9 GB of regops, so "
                         "this is opt-in and the ledger prints what it cost.")
    ap.add_argument("--lmhead-blocks-per-program", type=int, default=128,
                    help="8-row vocabulary blocks packed behind ONE staging of "
                         "the head's single activation row (fat path)")
    ap.add_argument("--lmhead-poison", type=float, default=None, metavar="K",
                    help="discriminator: scale the TILE's accumulator for ONE "
                         "vocabulary row (golden's own top-1) by K and report "
                         "whether the emitted token flips. A uniform scale "
                         "cannot flip an argmax, so the head's discriminator "
                         "targets a row.")
    ap.add_argument("--poison", type=float, default=None, metavar="K")
    ap.add_argument("--poison-layers", default=None,
                    help="layers the discriminator poisons (default: the "
                         "first offloaded layer)")
    ap.add_argument("--poison-proj-cols", type=int, default=8,
                    help="projection width for the control/poison pair — the "
                         "discriminator does not need full width, and the "
                         "ledger prints what it used")
    ap.add_argument("--heads", type=int, default=-1)
    ap.add_argument("--rope-heads", type=int, default=-1)
    ap.add_argument("--resid-cols", type=int, default=-1)
    ap.add_argument("--resid-slice", type=int, default=-1)
    ap.add_argument("--norm-dm", type=int, default=-1)
    ap.add_argument("--norm1-rows", type=int, default=0)
    ap.add_argument("--swiglu-chunks", type=int, default=-1)
    ap.add_argument("--tier", default="kvq8", choices=list(rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--weights-dir", default=str(l05.DEFAULT_WEIGHTS))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--keep-jobs", action="store_true",
                    help="do not delete graded projection job files "
                         "(~520 MB per full-width layer)")
    ap.add_argument("--binary", default=None)
    # remote_hw_exec.remote_config reads these as `args.hw_*`, argparse
    # winning over $APEX_F2_*; both are None by default, so the sim path is
    # untouched and attach() stays a no-op.
    ap.add_argument("--hw-host", default=None,
                    help="user@ip of the F2 instance (else $APEX_F2_HOST)")
    ap.add_argument("--hw-key", default=None,
                    help="ssh key for --hw-host (else $APEX_F2_KEY)")
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=14400)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args)
    if args.prompt is None and not args.ids:
        ap.error("need --prompt (or --ids for a tokenizer-free run)")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be >= 1")
    if args.ops:
        bad = [o for o in args.ops.split(",") if o and o not in OP_TYPES]
        if bad:
            ap.error(f"unknown op family/families {bad}; choose from "
                     f"{OP_TYPES}")
        args.ops = tuple(o for o in args.ops.split(",") if o)
        if any(o in RECON_FAMILIES for o in args.ops) \
                and not args.include_reconstructed:
            ap.error(f"--ops names {RECON_FAMILIES} but "
                     f"--include-reconstructed was not given. Those families "
                     f"re-enter through the 64-wide C-1 feeder and the token "
                     f"may legitimately differ; the flag is how you say you "
                     f"accept that.")
    else:
        args.ops = OP_TYPES if args.include_reconstructed else EXACT_FAMILIES
    if args.proj_cols < 0:
        args.proj_cols = FULL
    if args.proj_rows < 0:
        args.proj_rows = FULL
    work = Path(args.work_dir)
    model = rt.GoldenModel(Path(args.weights_dir))
    tok = None
    if args.ids:
        args.prompt = args.prompt or f"<ids {args.ids}>"
    else:
        tok = rt.load_tokenizer(model.meta["model"])
    return run_multi(model, tok, args, work)


if __name__ == "__main__":
    sys.exit(main())
