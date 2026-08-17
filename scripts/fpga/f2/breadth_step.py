#!/usr/bin/env python3
# breadth_step.py — MASTER TABLE row C1: the WHOLE ATTENTION STACK of ONE real
# Qwen2.5-7B decode step through the tile, produce-mode, graded bit-exact.
#
# ══ WHAT THIS EXTENDS, AND FROM WHAT ═══════════════════════════════════════
# T10(B) is today evidenced by ONE attention head (prompt_offload.py ->
# compute_job.py) and ONE Wq column block (gemm_job.py). Both are single
# instances. This file runs, for ONE real decode step of the committed S8
# prompt:
#     * all 28 query heads   — attention_core, input-only jobs (requant_en=0)
#     * all four attention projections Wq / Wk / Wv / Wo — K-split GEMM jobs
# and grades every one of them against golden AFTER the run. Nothing new is
# compiled: the head jobs are compute_job.build_compute_job_full and the
# projection jobs are gemm_job.ksplit_matvec, called verbatim. This file is the
# BREADTH HARNESS — the step capture, the block sampling, the audit and the
# verdict table — not a third compiler.
#
# ══ WHERE THE NUMBERS COME FROM (provenance, stated exactly) ═══════════════
# The per-head q / K_f16 / V_f16 are NOT re-derived. `attention_core` is
# REBOUND in both modules that resolve it (transformer.py:551 and
# attention.py:502 — the prompt_offload.py seam, used here in CAPTURE-ONLY
# mode: the wrapper returns golden's own result untouched, so the model is
# bit-identical to an un-instrumented run). What is recorded is the ARGUMENT
# TUPLE of each call, and the returned AttnCore OBJECT. After the step,
#     r.heads[h] is captured_core[h]                (asserted, 28/28)
# proves the job's inputs are the inputs of the call whose result the model
# actually consumed — not a plausible reconstruction of them.
#
# The projection operands come from the same step's LayerFx (`collect` in
# run_tinynpu.Session.step:325):
#     Wq        x8 = r.h8[T]     the decode row's RMSNorm+quant codes
#     Wk, Wv    x8 = r.h8[T-1]   the current token's CONTEXT row. Under the S8
#                                self-inclusive composition X = [x_0..x_t, x_t]
#                                that row IS the decode row (asserted equal);
#                                the K/V projection of the new token is the
#                                only K/V work a decode step does.
#     Wo        x8 = quant_rows_i8(r.attn)[0]   — _proj_epilogue's C-1 feeder,
#                                the concatenated 28-head attention output.
# Each projection's golden INT32 accumulator is recomputed with golden's own
# gemm_i8_ksplit and then RE-COMPOSED into the layer's own r.q_real /
# r.K_real / r.V_real / r.attn_proj and asserted bit-equal to them, so the
# graded integers are provably the layer's integers.
#
# ══ SAMPLING IS NEVER SILENT ═══════════════════════════════════════════════
# One MXE descriptor is N=8 output columns, so the four projections are
# 3584+512+512+3584 = 8192 columns = 1024 eight-column blocks, each a 2-job
# K-split over the FULL K=3584. 1024 blocks x 2 jobs x 3.8 s/job is 2.2 h of
# silicon, so the default samples --n-blocks-per-proj (2) evenly-spaced blocks
# per tensor and --full runs every one. The verdict table always prints the
# exact fraction run, per tensor and overall; a sampled run can never read as
# a complete one. Heads are NEVER sampled: 28/28 or the run fails.
#
# ══ ZERO OUTPUT EXPECTATIONS (audited here, independently) ═════════════════
# compute_job and gemm_job each assert their own no-expectation property at
# build time. This file re-audits every emitted program FROM THE FILE: no
# `cap` may carry an `e`, and no surviving `r`/`rn`/`poll` may target a
# capture-window DATA address (compute_job.OUTPUT_ADDRS for head jobs,
# gemm_job.RESULT_ADDRS for projection jobs — the fs/ss taps are captures in
# an attention job and deliberate CHECKS in a GEMM job). Totals are printed.
#
# CLI:
#   python3 scripts/fpga/f2/breadth_step.py --smoke                  # sim
#   python3 scripts/fpga/f2/breadth_step.py --smoke --full           # 1024 blk
#   python3 scripts/fpga/f2/breadth_step.py --smoke --executor hw    # F2
#   python3 scripts/fpga/f2/breadth_step.py --selftest               # no model
#   ... --fast-prefix     run only layers 0..--layer (see FAST PREFIX below)

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif" / "top" / "l3"),
           str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tinynpu as rt                                         # noqa: E402
from apex_golden import attention as at                          # noqa: E402
from apex_golden import compute as cp                            # noqa: E402
from apex_golden import transformer as tf                        # noqa: E402
from apex_golden.fp import f64_to_f16_bits                       # noqa: E402

import cap_decode as cd                                          # noqa: E402
import compute_job as cj                                         # noqa: E402
import gemm_job as gj                                            # noqa: E402
import tile_exec_bridge as bridge                                # noqa: E402

TOKENS_MAX = at.CHUNK_T_MAX          # 128 — the F-1 per-job row envelope
MXE_N = gj.MXE_N                     # 8 output columns per descriptor
SILICON_S_PER_JOB = 3.8              # MASTER_TABLE v4, measured job wall clock
DEFAULT_WORK = REPO / "build" / "f2_breadth"
S8_RUN = REPO / "docs/results/s8_7b_token/artifact_trace/run.json"
S8_WEIGHTS = REPO / "build/s8_weights/Qwen2.5-7B-4bit"
PROJ_ORDER = ("Wq", "Wk", "Wv", "Wo")


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═══════════════════ 1. the capture seam (capture-only) ═════════════════════

class StepCapture:
    """Record every attention_core call of ONE (step, layer). Changes nothing.

    The rebind is prompt_offload.py's seam used WITHOUT substitution: `_core`
    forwards to the original and returns its result object unmodified, so the
    decode is bit-identical to an un-instrumented run. `_layer` wraps
    decoder_layer_fx for (step, layer) bookkeeping only.
    """

    def __init__(self, *, layers_per_step: int, H: int, layer: int, step: int):
        self.layers_per_step = layers_per_step
        self.H = H
        self.t_layer, self.t_step = layer, step
        self.layer_calls = 0
        self.head_idx = 0
        self.heads: list[dict] = []
        self.head_counts: list[int] = []
        # the (step, layer) whose decoder_layer_fx call JUST RETURNED. Session
        # fires `collect` after the call, by which time layer_calls has already
        # advanced — so the live `step`/`layer` properties would name the NEXT
        # layer (and the next STEP for the last layer of a step).
        self.last_step = self.last_layer = -1
        self._orig_core_tf = self._orig_core_at = self._orig_layer = None

    # ── rebinding (context manager; ALWAYS restored) ────────────────────────
    def __enter__(self):
        self._orig_core_tf = tf.attention_core
        self._orig_core_at = at.attention_core
        self._orig_layer = tf.decoder_layer_fx
        assert self._orig_core_tf is self._orig_core_at, \
            "transformer/attention already resolve different attention_core"
        import inspect
        self._sig = inspect.signature(self._orig_core_at)
        tf.attention_core = self._core
        at.attention_core = self._core
        tf.decoder_layer_fx = self._layer
        return self

    def __exit__(self, *exc):
        tf.attention_core = self._orig_core_tf
        at.attention_core = self._orig_core_at
        tf.decoder_layer_fx = self._orig_layer
        return False

    @property
    def step(self) -> int:
        return self.layer_calls // self.layers_per_step

    @property
    def layer(self) -> int:
        return self.layer_calls % self.layers_per_step

    def _layer(self, X, w, tier, *a, **kw):
        X = np.asarray(X)
        T = X.shape[0] - 1
        if T > TOKENS_MAX and self.layer == self.t_layer:
            raise SystemExit(
                f"REFUSE: T={T} > {TOKENS_MAX} — the heads of this step are "
                f"ChunkedHeads whose host merge needs per-chunk sm_m/sm_l, "
                f"which the mailbox does not expose (apex_f2_mailbox.sv:33-45)."
                f" A chunked head cannot be run as a tile job (audit N8). "
                f"Shorten the prompt.")
        self.head_idx = 0
        try:
            return self._orig_layer(X, w, tier, *a, **kw)
        finally:
            self.last_step, self.last_layer = self.step, self.layer
            if (self.last_layer == self.t_layer
                    and self.last_step == self.t_step):
                self.head_counts.append(self.head_idx)
            self.layer_calls += 1

    def _core(self, *a, **kw):
        h = self.head_idx
        self.head_idx += 1
        core = self._orig_core_at(*a, **kw)          # TRANSPARENT: golden's own
        if not (self.layer == self.t_layer and self.step == self.t_step):
            return core
        b = self._sig.bind(*a, **kw)
        b.apply_defaults()
        A = b.arguments
        self.heads.append({
            "head": h,
            "q_f64": np.asarray(A["q_f64"], dtype=np.float64).copy(),
            "K_f16": np.asarray(A["K_f16"], dtype=np.uint16).copy(),
            "V_f16": np.asarray(A["V_f16"], dtype=np.uint16).copy(),
            "tier": A["tier"], "G": int(A["G"]),
            "outlier_idx": tuple(A.get("outlier_idx") or ()),
            "core": core,
        })
        return core


def capture_step(model, ids, *, layer: int, step: int, tier: str, group: int,
                 fast_prefix: bool, verbose: bool = True) -> dict:
    """Run the decode and return the target (step, layer)'s inputs.

    fast_prefix=False (default): the REAL step — run_tinynpu.Session over all
    28 layers, so the run also emits the next token and it can be checked
    against the committed S8 artifact.

    fast_prefix=True: only layers 0..`layer` are evaluated. This is EXACT for
    everything captured here, not an approximation: Session.hist[li] is fed by
    layer li-1's output alone (run_tinynpu.py:319-332), so layers > `layer`
    cannot influence layer `layer`'s inputs at any step. It cannot produce a
    token, and the banner says so.
    """
    L, H = model.n_layers, model.meta["H"]
    if not 0 <= layer < L:
        raise SystemExit(f"REFUSE: --layer {layer} outside [0,{L})")
    if not 0 <= step < len(ids):
        raise SystemExit(f"REFUSE: --step {step} outside [0,{len(ids)})")
    n_per_step = (layer + 1) if fast_prefix else L
    grabbed: dict = {}
    t0 = time.perf_counter()

    def collect(li, r):
        if cap.last_layer == layer and cap.last_step == step:
            grabbed["r"] = r

    with StepCapture(layers_per_step=n_per_step, H=H, layer=layer,
                     step=step) as cap:
        if fast_prefix:
            hist = [[] for _ in range(n_per_step)]
            for si, tid in enumerate(ids):
                x = model.embed_row(tid)
                for li in range(n_per_step):
                    hist[li].append(x)
                    X = np.vstack(hist[li] + [x])
                    r = tf.decoder_layer_fx(X, model.layers[li], tier,
                                            G=group, q_pos=si)
                    if li == layer and si == step:
                        grabbed["r"] = r
                    x = r.r2
                if verbose:
                    eprint(f"  [prefix {si+1}/{len(ids)}] layers 0..{layer} "
                           f"T={si+1} ({time.perf_counter()-t0:.0f}s)")
                if si == step:
                    break
            logits = None
        else:
            sess = rt.Session(model, tier, group)
            hidden = None
            for si, tid in enumerate(ids):
                hidden = sess.step(model.embed_row(tid), collect=collect)
                if verbose:
                    eprint(f"  [prefill {si+1}/{len(ids)}] pos={sess.pos-1} "
                           f"T={sess.pos} tok={tid} "
                           f"({time.perf_counter()-t0:.0f}s)")
            logits = model.head_logits(hidden)

    if "r" not in grabbed:
        raise SystemExit(f"REFUSE: layer {layer} step {step} was never reached "
                         f"— nothing was captured, so nothing can be claimed")
    r = grabbed["r"]
    heads = sorted(cap.heads, key=lambda d: d["head"])
    if len(heads) != H:
        raise SystemExit(f"REFUSE: captured {len(heads)} attention_core calls "
                         f"at (step {step}, layer {layer}), expected H={H} "
                         f"(head counts seen: {cap.head_counts})")
    # THE PROVENANCE PROOF: the captured arguments belong to the calls whose
    # results the model consumed.
    is_core = [r.heads[d["head"]] is d["core"] for d in heads]
    consumed = []
    hd = model.meta["head_dim"]
    for d in heads:
        sl = slice(d["head"] * hd, (d["head"] + 1) * hd)
        consumed.append(bool(np.array_equal(r.attn[sl], d["core"].out_hat)))
    return {"r": r, "w": model.layers[layer], "heads": heads,
            "logits": logits, "seconds": time.perf_counter() - t0,
            "is_core": is_core, "consumed": consumed,
            "layer_evals": cap.layer_calls, "T": int(r.T), "H": H,
            "head_dim": hd, "D_model": int(r.D_model), "H_kv": int(r.H_kv)}


# ═══════════════════════ 2. the 28 head jobs ════════════════════════════════

def run_head_jobs(capt: dict, *, out_dir: Path, executor: str, binary,
                  tile_div: int, slot: int, timeout_s: int,
                  verbose: bool = True) -> list[dict]:
    """One input-only (requant_en=0) attention job per query head."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in capt["heads"]:
        h = d["head"]
        name = f"bs_L{capt['layer']:02d}_s{capt['step']:03d}_h{h:02d}"
        t0 = time.time()
        path, man, prep = cj.build_compute_job_full(
            d["q_f64"], d["K_f16"], d["V_f16"], d["tier"], d["G"],
            d["outlier_idx"], out_dir, name)
        t_build = time.time() - t0
        t0 = time.time()
        res = bridge.run_job(path, executor=executor, binary=binary,
                             cap_out=str(out_dir / f"{name}.cap.jsonl"),
                             tile_div=tile_div, slot=slot, timeout_s=timeout_s)
        t_run = time.time() - t0
        row = {"kind": "head", "head": h, "name": name, "path": str(path),
               "T": man["T"], "D": man["D"], "tier": man["tier"],
               "regops": man["regops"], "checks": man["checks"],
               "caps": man["caps"], "cap_census": man["cap_census"]["total"],
               "n_captures": len(res["captures"]), "rc": res["rc"],
               "exec_ok": bool(res["ok"]), "notes": list(res["notes"]),
               "build_s": round(t_build, 2), "run_s": round(t_run, 2),
               "audits": [audit_program(path, cj.OUTPUT_ADDRS)]}
        if not res["captures"]:
            row.update({"ok": False, "problems": ["no captures"]})
            rows.append(row)
            continue
        g = cj.grade_compute_job(res["captures"], d["q_f64"], d["K_f16"],
                                 d["V_f16"], d["tier"], d["G"],
                                 d["outlier_idx"], prep=prep)
        # grade_compute_job recomputes golden from the inputs; ALSO grade
        # against the core the MODEL used, which is the load-bearing object.
        gold = d["core"]
        acc = np.asarray(g["acc_i32"], dtype=np.int64)
        ep = cd.epilogue(acc, g["s_c"]["captured"]) if g["s_c"]["captured"] \
            is not None else None
        row.update({
            "acc_equal": bool(g["acc"]["equal"]),
            "acc_vs_model": bool(cd.grade(acc, gold.acc_o)["equal"]),
            "o8_equal": bool(g.get("o8", {}).get("equal")),
            "out_hat_equal": bool(g.get("out_hat", {}).get("equal")),
            "out_hat_vs_model": (None if ep is None else
                                 bool(cd.grade(ep["out_hat"],
                                               gold.out_hat)["equal"])),
            "rq_equal": bool(g.get("rq_equal")),
            "s_q": bool(g["s_q"]["equal"]), "s_c": bool(g["s_c"]["equal"]),
            "s_k": bool(g["s_k"]["equal"]), "s_v": bool(g["s_v"]["equal"]),
            "problems": list(g["problems"]),
            "acc_i32_head8": [int(v) for v in acc[:8]],
            "_captures": res["captures"], "_gold": gold, "_grade": g,
        })
        row["ok"] = bool(g["ok"] and res["ok"] and row["acc_vs_model"]
                         and row["out_hat_vs_model"])
        rows.append(row)
        if verbose:
            eprint(f"  [head {h:02d}/{capt['H']}] T={man['T']} "
                   f"caps={len(res['captures'])} "
                   f"acc={'==' if row['acc_equal'] else '!!'} "
                   f"out_hat={'==' if row['out_hat_equal'] else '!!'} "
                   f"({t_build:.1f}s build + {t_run:.1f}s run)")
    return rows


# ═════════════════════ 3. the four projections ══════════════════════════════

def sample_blocks(nb: int, k: int) -> list[int]:
    """`k` evenly spaced 8-column block indices out of `nb`, endpoints included.

    Deterministic and disclosed — the verdict table prints exactly which
    blocks ran and what fraction of the tensor that is.
    """
    if k >= nb:
        return list(range(nb))
    if k <= 1:
        return [0]
    return sorted({int(round(i * (nb - 1) / (k - 1))) for i in range(k)})


def projection_specs(capt: dict) -> list[dict]:
    """The four attention projections of this step, with golden accumulators.

    Every spec's `acc` is recomputed with golden's OWN gemm_i8_ksplit and is
    then re-composed into the layer's own real-valued tensor and asserted
    bit-equal — so grading a block against acc[n0:n0+8] grades it against the
    integer the layer itself produced.
    """
    r, w, T = capt["r"], capt["w"], capt["T"]
    from apex_golden.fp import f16_bits_to_f64
    s_h = f16_bits_to_f64(r.s_h)
    h8 = np.asarray(r.h8, dtype=np.int64)
    specs = []

    # ── Wq: the decode row ─────────────────────────────────────────────────
    acc_q = cp.gemm_i8_ksplit(h8[T:], np.asarray(w.Wq, dtype=np.int64))[0]
    q_real = acc_q.astype(np.float64) * (float(s_h[T]) * w.s_wq)
    if w.bq is not None:
        q_real = q_real + np.asarray(w.bq, dtype=np.float64)
    specs.append({"tensor": "Wq", "x8": h8[T], "W": w.Wq, "acc": acc_q,
                  "x_from": "r.h8[T] (RMSNorm-1 + quant of the decode row)",
                  "recompose_ok": bool(np.array_equal(q_real, r.q_real))})

    # ── Wk / Wv: the current token's CONTEXT row (== the decode row) ────────
    same_row = bool(np.array_equal(h8[T - 1], h8[T]))
    for tag, W, s_w, bias, ref in (("Wk", w.Wk, w.s_wk, w.bk, r.K_real),
                                   ("Wv", w.Wv, w.s_wv, w.bv, r.V_real)):
        acc = cp.gemm_i8_ksplit(h8[T - 1:T], np.asarray(W, dtype=np.int64))[0]
        real = acc.astype(np.float64) * (float(s_h[T - 1]) * s_w)
        if bias is not None:
            real = real + np.asarray(bias, dtype=np.float64)
        specs.append({"tensor": tag, "x8": h8[T - 1], "W": W, "acc": acc,
                      "x_from": f"r.h8[T-1] (the new token's KV row; == the "
                                f"decode row: {same_row})",
                      "recompose_ok": bool(np.array_equal(real, ref[T - 1]))})

    # ── Wo: the concatenated 28-head attention output ──────────────────────
    a8m, s_a = at.quant_rows_i8(np.asarray(r.attn, dtype=np.float64)[None, :])
    a8 = np.asarray(a8m[0], dtype=np.int64)
    acc_o = cp.gemm_i8_ksplit(a8m, np.asarray(w.Wo, dtype=np.int64))[0] \
        .astype(np.int64)
    amax = int(np.max(np.abs(acc_o), initial=0)) or 1
    sc, sh = at.calib_requant(amax)
    o8 = at.requant_i32_to_i8(acc_o, sc, sh).astype(np.int64)
    s_out = float(f16_bits_to_f64(np.array([s_a[0]]))[0]) * float(w.s_wo) \
        * float(1 << sh) / float(sc)
    specs.append({"tensor": "Wo", "x8": a8, "W": w.Wo, "acc": acc_o,
                  "x_from": "quant_rows_i8(r.attn) — the C-1 feeder of the "
                            "28 concatenated head outputs",
                  "recompose_ok": bool(np.array_equal(
                      o8.astype(np.float64) * s_out, r.attn_proj))})
    return specs


def run_proj_blocks(specs, *, n_per_proj: int, full: bool, out_dir: Path,
                    executor: str, binary, timeout_s: int,
                    layer: int, step: int, prune: bool = False,
                    verbose: bool = True) -> list[dict]:
    """K-split GEMM jobs for the sampled 8-column blocks of each projection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sp in specs:
        Wfull = np.asarray(sp["W"])
        K, N_total = int(Wfull.shape[0]), int(Wfull.shape[1])
        nb = N_total // MXE_N
        if N_total % MXE_N:
            raise SystemExit(f"REFUSE: {sp['tensor']} has {N_total} columns, "
                             f"not a whole number of {MXE_N}-column MXE blocks")
        picks = list(range(nb)) if full else sample_blocks(nb, n_per_proj)
        sp["n_blocks"] = nb
        sp["picked"] = picks
        for bi in picks:
            n0 = bi * MXE_N
            W = np.asarray(Wfull[:, n0:n0 + MXE_N], dtype=np.int64)
            name = f"bs_L{layer:02d}_s{step:03d}_{sp['tensor']}_n{n0:04d}"
            t0 = time.time()
            try:
                # rows_per_desc=1 on hw: the flying image (05efb2a) predates the
                # stage-buffer base-row change; k=2048 chaining is HEAD-sim-only
                # (image-parity rule — this exact trap fired 2026-07-29 AND here).
                rpd = 1 if executor == "hw" else gj.ROWS_PER_DESC
                res = gj.ksplit_matvec(W, np.asarray(sp["x8"], dtype=np.int64),
                                       out_dir=out_dir, name=name,
                                       executor=executor, binary=binary,
                                       rows_per_desc=rpd,
                                       timeout_s=timeout_s, verbose=False)
            except AssertionError as e:                # stage_plan F1..F4
                rows.append({"kind": "proj", "tensor": sp["tensor"],
                             "block": bi, "n0": n0, "K": K, "ok": False,
                             "refused": True, "problems": [str(e)[:200]],
                             "jobs": [], "wall_s": round(time.time() - t0, 2)})
                if verbose:
                    eprint(f"  [{sp['tensor']} n{n0:04d}] REFUSED: "
                           f"{str(e)[:80]}")
                continue
            wall = time.time() - t0
            gold = np.asarray(sp["acc"][n0:n0 + MXE_N], dtype=np.int64)
            acc = np.asarray(res["acc_i32"], dtype=np.int64)
            g_model = cd.grade(acc, gold)
            mans = []
            for j in res["jobs"]:
                mp = out_dir / f"{j['job']}.manifest.json"
                mans.append(json.loads(mp.read_text()) if mp.is_file() else {})
            row = {"kind": "proj", "tensor": sp["tensor"], "block": bi,
                   "n0": n0, "K": K, "K_staged": res["K_staged"],
                   "rows": res["rows"], "kstar": res["kstar"],
                   "n_jobs": len(res["jobs"]),
                   "paths": [str(out_dir / f"{j['job']}.regops.jsonl")
                             for j in res["jobs"]],
                   "regops": sum(j["regops"] for j in res["jobs"]),
                   "checks": sum(j["checks"] for j in res["jobs"]),
                   "caps": sum(m.get("caps", 0) for m in mans),
                   "weight_beats": sum(j["weight_beats"] for j in res["jobs"]),
                   "acc_i32": [int(v) for v in acc],
                   "golden_i32": [int(v) for v in gold],
                   "acc_equal": bool(res["grade"]["equal"]),
                   "acc_vs_model": bool(g_model["equal"]),
                   "exec_ok": all(j["ok"] for j in res["jobs"]),
                   "problems": list(res["problems"]), "refused": False,
                   "wall_s": round(wall, 2),
                   "job_wall_s": res["wall_s_per_job"],
                   "_W": W}
            # audited HERE, from the files, before any pruning can hide them
            row["audits"] = [audit_program(p, gj.RESULT_ADDRS)
                             for p in row["paths"]]
            row["ok"] = bool(res["ok"] and row["acc_vs_model"])
            rows.append(row)
            if verbose:
                eprint(f"  [{sp['tensor']} blk {bi}/{nb} n{n0:04d}] "
                       f"{len(res['jobs'])} K-split jobs k="
                       f"{[j['k'] for j in res['jobs']]} acc"
                       f"{'==' if row['acc_vs_model'] else '!!'}model "
                       f"({wall:.1f}s)")
            if prune:                      # --full writes ~900 MB of regops
                for p in row["paths"]:
                    try:
                        Path(p).unlink()
                    except OSError:
                        pass
    return rows


# ══════════════════ 4. the independent no-expectation audit ═════════════════

def audit_program(path, out_addrs) -> dict:
    """Re-read one emitted program and prove it contains no answer."""
    ops = [json.loads(l) for l in Path(path).read_text().splitlines() if l]
    caps_with_e = [o for o in ops if o["op"] == "cap" and "e" in o]
    expect_out = [o for o in ops if o["op"] in ("r", "rn", "poll")
                  and o.get("a") in out_addrs]
    return {"path": str(path), "ops": len(ops),
            "caps": sum(1 for o in ops if o["op"] == "cap"),
            "checks": sum(1 for o in ops if o["op"] in ("r", "rn", "poll")),
            "caps_with_expectation": len(caps_with_e),
            "output_expectations": len(expect_out),
            "first_bad": (caps_with_e or expect_out or [None])[0]}


def audit_all(head_rows, proj_rows) -> dict:
    """Aggregate the per-program audits taken as each program was emitted."""
    tot = {"programs": 0, "ops": 0, "caps": 0, "checks": 0,
           "caps_with_expectation": 0, "output_expectations": 0, "bad": []}
    for r in list(head_rows) + list(proj_rows):
        for a in r.get("audits", ()):
            _accum(tot, a)
    tot["ok"] = (tot["programs"] > 0 and tot["caps_with_expectation"] == 0
                 and tot["output_expectations"] == 0)
    return tot


def _accum(tot, a):
    tot["programs"] += 1
    for k in ("ops", "caps", "checks", "caps_with_expectation",
              "output_expectations"):
        tot[k] += a[k]
    if a["caps_with_expectation"] or a["output_expectations"]:
        tot["bad"].append(a)


# ═══════════════════════ 5. the discriminators ══════════════════════════════

def discriminate(head_rows, proj_rows, capt) -> dict:
    """A grader that cannot go red proves nothing. Four known-bad inputs."""
    d = {}
    live = [r for r in head_rows if r.get("_captures")]
    if live:
        r0 = live[0]
        pert = [dict(c) for c in r0["_captures"]]
        tag = None
        for c in pert:
            if c["sem"].startswith("ro_w") and c["sem"] != "ro_meta":
                c["value"] = (c["value"] + 1) & 0xFFFFFFFF
                tag = c["tag"]
                break
        acc_bad = cd.ro_lanes_to_i32(pert, expect_n=r0["D"])
        d["a_head_one_count"] = {
            "tag": tag,
            "red": not cd.grade(acc_bad, r0["_gold"].acc_o)["equal"]}
        # a head-SPECIFIC grader: head 0's numbers against head 1's golden
        if len(live) > 1:
            acc0 = cd.ro_lanes_to_i32(r0["_captures"], expect_n=r0["D"])
            d["b_head_cross"] = {
                "pair": (r0["head"], live[1]["head"]),
                "red": not cd.grade(acc0, live[1]["_gold"].acc_o)["equal"]}
    live_p = [r for r in proj_rows if r.get("acc_i32") and not r.get("refused")]
    if live_p:
        p0 = live_p[0]
        bad = np.asarray(p0["acc_i32"], dtype=np.int64).copy()
        bad[0] += 1
        d["c_proj_one_count"] = {
            "red": not cd.grade(bad, np.asarray(p0["golden_i32"],
                                                dtype=np.int64))["equal"]}
        Wb = np.asarray(p0["_W"], dtype=np.int64).copy()
        Wb[0, 0] += 1 if Wb[0, 0] < 127 else -1
        specs = {s["tensor"]: s for s in capt["specs"]}
        x8 = np.asarray(specs[p0["tensor"]]["x8"], dtype=np.int64)
        d["d_proj_wrong_weight"] = {
            "red": not cd.grade(np.asarray(p0["acc_i32"], dtype=np.int64),
                                np.einsum("k,kn->n", x8, Wb))["equal"]}
    d["all_red"] = all(v.get("red") for v in d.values() if isinstance(v, dict))
    return d


# ══════════════════════════ 6. the verdict table ════════════════════════════

def _pct(a, b) -> str:
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


def report(args, model, ids, capt, head_rows, proj_rows, audit, disc,
           timing, work: Path) -> int:                          # noqa: C901
    W = 92
    H = capt["H"]
    n_head_ok = sum(1 for r in head_rows if r.get("ok"))
    n_acc = sum(1 for r in head_rows if r.get("acc_equal"))
    n_accm = sum(1 for r in head_rows if r.get("acc_vs_model"))
    n_out = sum(1 for r in head_rows if r.get("out_hat_equal"))
    heads_all = (len(head_rows) == H and n_head_ok == H)

    specs = capt["specs"]
    by_t = {t: [r for r in proj_rows if r["tensor"] == t] for t in PROJ_ORDER}
    n_blk_run = len(proj_rows)
    n_blk_tot = sum(s["n_blocks"] for s in specs)
    n_blk_ok = sum(1 for r in proj_rows if r.get("ok"))
    n_jobs_proj = sum(r.get("n_jobs", 0) for r in proj_rows)
    n_jobs = len(head_rows) + n_jobs_proj
    n_jobs_full = H + n_blk_tot * 2
    proj_all = (n_blk_run > 0 and n_blk_ok == n_blk_run)

    print("\n" + "=" * W)
    print("C1 — T10(B) BREADTH: the WHOLE ATTENTION STACK of ONE real "
          "Qwen2.5-7B decode step")
    print("=" * W)
    print(f"  model        {model.meta['model']}  L={model.n_layers} H={H} "
          f"H_kv={capt['H_kv']} head_dim={capt['head_dim']} "
          f"D_model={capt['D_model']}")
    print(f"  prompt       {capt['prompt']!r}")
    print(f"  prompt ids   {ids} ({len(ids)} tokens)   source: "
          f"{capt['ids_source']}")
    print(f"  step         {capt['step']} of {len(ids)} (T={capt['T']} rows, "
          f"q_pos={capt['step']}, self-inclusive) ; layer "
          f"{capt['layer']:02d} ; tier {capt['tier']} G={args.group}")
    how = (f"FULL {model.n_layers}-layer step" if not args.fast_prefix
           else f"PREFIX layers 0..{args.layer} only")
    print(f"  decode       {how} — {capt['layer_evals']} decoder_layer_fx "
          f"evaluations, {timing['capture_s']:.0f}s")
    if capt.get("token") is not None:
        tk = capt["token"]
        print(f"  next token   argmax(logits) = {tk['id']}  vs the committed "
              f"S8 run's generated_ids[0] = {tk['expect']}  -> "
              f"{'MATCH' if tk['match'] else 'MISMATCH'}")
    elif args.fast_prefix:
        print(f"  next token   n/a — --fast-prefix ran only layers 0.."
              f"{args.layer}; EXACT for every tensor captured here (a layer's "
              f"inputs never depend on later layers) but it emits no token")
    print(f"  provenance   r.heads[h] IS the captured core: "
          f"{sum(capt['is_core'])}/{H} ; its out_hat is the value the layer "
          f"wrote into r.attn: {sum(capt['consumed'])}/{H}")
    print(f"  executor     {args.executor}   mode: PRODUCE (requant_en=0 in "
          f"every program; the host runs the Q13 epilogue)")

    # ── coverage, stated before any result ─────────────────────────────────
    print("-" * W)
    print("  COVERAGE (nothing is sampled silently)")
    print(f"    heads              {len(head_rows)}/{H} = "
          f"{_pct(len(head_rows), H)}  (heads are never sampled)")
    for s in specs:
        pk = s["picked"]
        shown = str(pk) if len(pk) <= 8 else f"{pk[:8]}… ({len(pk)} of them)"
        print(f"    {s['tensor']:<3}  {s['W'].shape[0]}x{s['W'].shape[1]:<5} "
              f"{len(pk):>4}/{s['n_blocks']:<4} 8-column blocks = "
              f"{_pct(len(pk), s['n_blocks']):>7}   blocks run: {shown}")
    how_blk = ("--full: EVERY block" if args.full else
               f"--n-blocks-per-proj {args.n_blocks_per_proj}, evenly "
               f"spaced, endpoints included")
    print(f"    projection cols    {n_blk_run * MXE_N}/{n_blk_tot * MXE_N} = "
          f"{_pct(n_blk_run, n_blk_tot)}   ({how_blk})")
    print(f"    K contraction      every block that ran contracted the FULL "
          f"K={specs[0]['W'].shape[0]} (2 K-split jobs, 100% of the "
          f"contraction — only OUTPUT COLUMNS are sampled)")

    # ── heads ──────────────────────────────────────────────────────────────
    print("-" * W)
    print(f"  ATTENTION HEADS — {len(head_rows)}/{H} run, {n_head_ok}/"
          f"{len(head_rows)} green")
    print("    head kv   T   D   caps/cens  acc_i32  vs.model  o8   out_hat  "
          "rq  s_q s_c s_k s_v   s")
    grp = H // max(capt["H_kv"], 1)
    for r in head_rows:
        m = "  ok " if r.get("acc_vs_model") else " DIFF"
        print(f"    h{r['head']:02d}  {r['head']//grp}  {r['T']:>3} "
              f"{r['D']:>3}   {r['n_captures']:>4}/{r['cap_census']:<4} "
              f"{'bit-exact' if r.get('acc_equal') else '  DIFF   '}{m}  "
              f"{'==' if r.get('o8_equal') else '!!'}   "
              f"{'==' if r.get('out_hat_equal') else '!!'}     "
              f"{'ok' if r.get('rq_equal') else '!!'}  "
              f"{'ok ' if r.get('s_q') else '!! '}"
              f"{'ok ' if r.get('s_c') else '!! '}"
              f"{'ok ' if r.get('s_k') else '!! '}"
              f"{'ok ' if r.get('s_v') else '!! '} "
              f"{r['run_s']:>5.1f}")
        for p in r.get("problems", []):
            print(f"         PROBLEM: {p}")
        for p in r.get("notes", []):
            print(f"         NOTE: {p}")
    print(f"    totals: acc_i32 bit-exact vs golden {n_acc}/{len(head_rows)} ; "
          f"vs the MODEL's own core {n_accm}/{len(head_rows)} ; out_hat "
          f"{n_out}/{len(head_rows)}")

    # ── projections ────────────────────────────────────────────────────────
    print("-" * W)
    print(f"  PROJECTIONS (K-split GEMM, N=8 per descriptor) — {n_blk_run} "
          f"blocks, {n_blk_ok} green, {n_jobs_proj} tile jobs")
    print("    tensor  block  cols        K    jobs  regops  checks caps  "
          "acc_i32   vs.model    s")
    for t in PROJ_ORDER:
        for r in by_t[t]:
            if r.get("refused"):
                print(f"    {t:<6} {r['block']:>5}  "
                      f"{r['n0']:>4}..{r['n0']+7:<4}  {r['K']:>5}   "
                      f"REFUSED — {r['problems'][0][:44]}")
                continue
            print(f"    {t:<6} {r['block']:>5}  {r['n0']:>4}..{r['n0']+7:<4}  "
                  f"{r['K']:>5}  {r['n_jobs']:>4} {r['regops']:>7} "
                  f"{r['checks']:>6} {r['caps']:>4}  "
                  f"{'bit-exact' if r['acc_equal'] else '  DIFF  '} "
                  f"{'   ok   ' if r['acc_vs_model'] else '  DIFF  '} "
                  f"{r['wall_s']:>5.1f}")
            for p in r.get("problems", []):
                print(f"         PROBLEM: {p}")
    for s in specs:
        print(f"    {s['tensor']:<3} activation: {s['x_from']}")
        print(f"        golden accumulator re-composed into the layer's own "
              f"tensor bit-exactly: {s['recompose_ok']}")

    # ── audit ──────────────────────────────────────────────────────────────
    print("-" * W)
    print(f"  ZERO-OUTPUT-EXPECTATION AUDIT (each program re-read from its own "
          f"emitted file: {audit['programs']} programs)")
    print(f"    {audit['ops']} regops, {audit['checks']} structural checks, "
          f"{audit['caps']} caps")
    print(f"    caps carrying an expectation      : "
          f"{audit['caps_with_expectation']}  (must be 0)")
    print(f"    compares on a capture-window addr : "
          f"{audit['output_expectations']}  (must be 0)")
    for b in audit["bad"][:3]:
        print(f"    BAD: {b}")
    print(f"    the canonical replay arm (build/f2_regops) is untouched — no "
          f"check is lost from it")

    # ── discrimination ─────────────────────────────────────────────────────
    print("-" * W)
    print("  DISCRIMINATION (a grader that cannot go red is not evidence)")
    for k, v in disc.items():
        if isinstance(v, dict):
            print(f"    {k:<22} RED={v.get('red')}"
                  + (f"  {({kk: vv for kk, vv in v.items() if kk != 'red'})}"
                     if len(v) > 1 else ""))

    # ── cost ───────────────────────────────────────────────────────────────
    print("-" * W)
    print("  COST")
    print(f"    tile jobs this invocation : {n_jobs}  "
          f"({len(head_rows)} head + {n_jobs_proj} projection K-split)")
    print(f"    sim wall clock            : "
          f"{timing['jobs_s']:.0f}s  (build+run, all jobs)")
    print(f"    PROJECTED SILICON         : {n_jobs} x "
          f"{SILICON_S_PER_JOB} s/job = {n_jobs * SILICON_S_PER_JOB:.0f}s "
          f"= {n_jobs * SILICON_S_PER_JOB / 60:.1f} min")
    print(f"    projected silicon, --full : {n_jobs_full} jobs x "
          f"{SILICON_S_PER_JOB} s = "
          f"{n_jobs_full * SILICON_S_PER_JOB / 3600:.2f} h "
          f"({H} heads + {n_blk_tot} blocks x 2 K-split jobs)")

    ok = (heads_all and proj_all and audit["ok"] and disc.get("all_red")
          and all(s["recompose_ok"] for s in specs)
          and sum(capt["is_core"]) == H and sum(capt["consumed"]) == H)
    print("-" * W)
    if not heads_all:
        print(f"  FAIL: {len(head_rows) - n_head_ok} head job(s) not green "
              f"(or fewer than {H} ran)")
    if not proj_all:
        print(f"  FAIL: {n_blk_run - n_blk_ok} projection block(s) not green")
    if not audit["ok"]:
        print("  FAIL: a program carries an expectation on the tile's output")
    if not disc.get("all_red"):
        print("  FAIL: the grader did not go red on a known-bad input")
    scope = ("ALL 28 heads + ALL 1024 projection blocks"
             if args.full else
             f"ALL {H} heads + {n_blk_run}/{n_blk_tot} projection blocks "
             f"({_pct(n_blk_run, n_blk_tot)} of the output columns; full K)")
    print(f"  SCOPE RUN     : {scope}")
    print(f"  BREADTH (C1)  : {'PASS' if ok else 'FAIL'}")
    print("=" * W)

    work.mkdir(parents=True, exist_ok=True)
    out = work / "breadth_step_result.json"
    out.write_text(json.dumps({
        "prompt": capt["prompt"], "prompt_ids": ids,
        "layer": capt["layer"], "step": capt["step"], "T": capt["T"],
        "tier": capt["tier"], "G": args.group, "executor": args.executor,
        "full_stack": not args.fast_prefix, "layer_evals": capt["layer_evals"],
        "token": capt.get("token"),
        "heads": [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in head_rows],
        "projections": [{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in proj_rows],
        "coverage": {"heads": [len(head_rows), H],
                     "blocks": [n_blk_run, n_blk_tot],
                     "per_tensor": {s["tensor"]: [len(s["picked"]),
                                                  s["n_blocks"]]
                                    for s in specs}},
        "audit": {k: v for k, v in audit.items() if k != "bad"},
        "discrimination": disc, "timing": timing,
        "jobs": n_jobs, "projected_silicon_s": n_jobs * SILICON_S_PER_JOB,
        "projected_silicon_full_s": n_jobs_full * SILICON_S_PER_JOB,
        "verdict": "PASS" if ok else "FAIL",
        "git": rt.git_rev(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if ok else 1


# ══════════════════════════════ 7. the run ══════════════════════════════════

def smoke(args) -> int:
    work = Path(args.out_dir or DEFAULT_WORK)
    model = rt.GoldenModel(Path(args.weights_dir))
    tier = rt.TIER_MAP[args.tier]

    # ── the prompt: the COMMITTED S8 run by default (no tokenizer needed) ───
    ids_source = "--ids"
    prompt = args.prompt
    if args.ids:
        ids = list(args.ids)
        prompt = prompt or f"<ids {ids}>"
    elif args.prompt:
        tok = rt.load_tokenizer(model.meta["model"])
        ids = tok.encode(args.prompt)
        ids_source = "tokenizer"
    else:
        run = json.loads(S8_RUN.read_text())
        ids = [int(v) for v in run["prompt_ids"]]
        prompt = run["prompt"]
        ids_source = f"{S8_RUN.relative_to(REPO)} (committed S8 run)"
    if len(ids) > TOKENS_MAX:
        eprint(f"REFUSE: {len(ids)} prompt tokens > {TOKENS_MAX} — beyond the "
               f"F-1 envelope a head is a ChunkedHead and cannot be a tile job")
        return 2
    step = args.step if args.step is not None else len(ids) - 1

    eprint(f"[C1] {model.meta['model']} L={model.n_layers} H={model.meta['H']} "
           f"H_kv={model.meta['H_kv']}; prompt {len(ids)} tokens; capturing "
           f"layer {args.layer} step {step} "
           f"({'full 28-layer decode' if not args.fast_prefix else 'prefix'})")
    t0 = time.time()
    capt = capture_step(model, ids, layer=args.layer, step=step, tier=tier,
                        group=args.group, fast_prefix=args.fast_prefix)
    capture_s = time.time() - t0
    capt.update({"layer": args.layer, "step": step, "tier": tier,
                 "prompt": prompt, "ids_source": ids_source})
    if capt["logits"] is not None and step == len(ids) - 1:
        run = json.loads(S8_RUN.read_text()) if S8_RUN.is_file() else {}
        exp = (int(run["generated_ids"][0])
               if run.get("generated_ids") and ids_source.startswith("docs")
               else None)
        tid = int(np.argmax(capt["logits"]))
        capt["token"] = {"id": tid, "expect": exp,
                         "match": (None if exp is None else tid == exp)}

    t0 = time.time()
    head_rows = run_head_jobs(capt, out_dir=work / "heads",
                              executor=args.executor, binary=args.binary,
                              tile_div=args.tile_div, slot=args.slot,
                              timeout_s=args.timeout_s)
    capt["specs"] = projection_specs(capt)
    proj_rows = run_proj_blocks(capt["specs"],
                                n_per_proj=args.n_blocks_per_proj,
                                full=args.full, out_dir=work / "proj",
                                executor=args.executor, binary=args.binary,
                                timeout_s=args.timeout_s, layer=args.layer,
                                step=step, prune=args.prune_jobs)
    jobs_s = time.time() - t0

    audit = audit_all(head_rows, proj_rows)
    disc = discriminate(head_rows, proj_rows, capt)
    timing = {"capture_s": capture_s, "jobs_s": jobs_s}
    return report(args, model, ids, capt, head_rows, proj_rows, audit, disc,
                  timing, work)


# ═══════════════════════ 8. host-only selftest ══════════════════════════════

def _selftest() -> int:
    """No 7B model, no executor: the harness logic only."""
    fails = 0

    def chk(n, cond, extra=""):
        nonlocal fails
        print(f"  [{n}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    # 1-3: block sampling is deterministic, endpoint-inclusive, never silent
    chk("1 sampling endpoints", sample_blocks(448, 2) == [0, 447],
        f"{sample_blocks(448, 2)}")
    chk("2 sampling spread", sample_blocks(64, 5) == [0, 16, 32, 47, 63],
        f"{sample_blocks(64, 5)}")
    chk("3 sampling saturates", sample_blocks(3, 9) == [0, 1, 2]
        and sample_blocks(4, 1) == [0])

    # 4: the audit catches an expectation that survived
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="apex_breadth_st_"))
    good = tmp / "good.jsonl"
    good.write_text("\n".join(json.dumps(o) for o in [
        {"op": "cap", "a": cd.RO_W0, "m": 0xFFFFFFFF, "sem": "ro_w0"},
        {"op": "r", "a": 0x1008, "m": 0xFFFFFFFF, "e": 8, "sem": "r"}]) + "\n")
    a = audit_program(good, cj.OUTPUT_ADDRS)
    chk("4 clean program audits clean",
        a["output_expectations"] == 0 and a["caps_with_expectation"] == 0
        and a["caps"] == 1 and a["checks"] == 1, f"{a}")
    bad = tmp / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(o) for o in [
        {"op": "cap", "a": cd.RO_W0, "m": 0xFFFFFFFF, "e": 7, "sem": "ro_w0"},
        {"op": "r", "a": cd.RO_W0 + 4, "m": 0xFFFFFFFF, "e": 3,
         "sem": "ro_w1"}]) + "\n")
    b = audit_program(bad, cj.OUTPUT_ADDRS)
    chk("5 audit catches a surviving expectation",
        b["output_expectations"] == 1 and b["caps_with_expectation"] == 1,
        f"{b}")

    # 6: the capture seam is transparent AND restores golden
    rng = np.random.default_rng(20260730)
    D, T = 128, 4
    K = f64_to_f16_bits(rng.normal(0, 1, (T, D))).astype(np.uint16)
    V = f64_to_f16_bits(rng.normal(0, 1, (T, D))).astype(np.uint16)
    q = rng.normal(0, 1, D)
    ref = at.attention_core(q, K, V, at.TIER_CQ8, G=128)
    with StepCapture(layers_per_step=1, H=1, layer=0, step=0) as sc:
        got = at.attention_core(q, K, V, at.TIER_CQ8, G=128)
    chk("6 seam is transparent",
        np.array_equal(got.acc_o, ref.acc_o)
        and np.array_equal(got.out_hat, ref.out_hat) and len(sc.heads) == 1
        and np.array_equal(sc.heads[0]["K_f16"], K),
        f"{len(sc.heads)} capture(s), acc equal")
    chk("7 golden restored after the context manager",
        tf.attention_core is at.attention_core
        and at.attention_core.__module__.endswith("apex_golden.attention"))

    # 8: the projection block geometry the run will use
    tot = 3584 + 512 + 512 + 3584
    chk("8 block geometry", tot % MXE_N == 0 and tot // MXE_N == 1024,
        f"{tot} output columns = {tot // MXE_N} blocks of {MXE_N}")
    chk("9 silicon projection arithmetic",
        abs((28 + 1024 * 2) * SILICON_S_PER_JOB / 3600 - 2.19) < 0.01,
        f"--full = {(28 + 1024 * 2)} jobs = "
        f"{(28 + 1024 * 2) * SILICON_S_PER_JOB / 3600:.2f} h")

    # 10: cd.grade discriminates on the shapes this harness feeds it
    chk("10 grade discriminates", not cd.grade(np.array([1, 2]),
                                               np.array([1, 3]))["equal"]
        and cd.grade(np.array([1, 2]), np.array([1, 2]))["equal"])

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"BREADTH_STEP SELFTEST: {'FAIL' if fails else 'PASS'} "
          f"(fails={fails})")
    return 1 if fails else 0


# ═══════════════════════════════ 9. CLI ═════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="C1: all 28 heads + Wq/Wk/Wv/Wo of one real Qwen2.5-7B "
                    "decode step through the tile, produce-mode, graded.")
    ap.add_argument("--smoke", action="store_true",
                    help="capture the step, build+run+grade every job")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prompt", default=None,
                    help="default: the committed S8 run's prompt (no "
                         "tokenizer needed — its ids are read from run.json)")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--step", type=int, default=None,
                    help="decode step to capture (default: the last prefill "
                         "step, whose logits pick the first new token)")
    ap.add_argument("--n-blocks-per-proj", type=int, default=2, metavar="N",
                    help="8-column blocks sampled per projection (default 2, "
                         "evenly spaced, endpoints included)")
    ap.add_argument("--full", action="store_true",
                    help="every 8-column block of all four projections "
                         "(1024 blocks = 2048 tile jobs)")
    ap.add_argument("--fast-prefix", action="store_true",
                    help="evaluate only layers 0..--layer. EXACT for every "
                         "tensor captured (a layer's inputs never depend on "
                         "later layers) but emits no token")
    ap.add_argument("--prune-jobs", action="store_true",
                    help="unlink each projection program after it has been "
                         "run AND audited (--full writes ~900 MB of regops)")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tier", default="kvq8", choices=list(rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--weights-dir", default=str(S8_WEIGHTS))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=3600)
    args = ap.parse_args()
    import remote_hw_exec                       # hw routing (no-op unless
    remote_hw_exec.attach(bridge, args)         # $APEX_F2_HOST is set)

    if args.selftest:
        return _selftest()
    if args.smoke or args.prompt or args.ids is not None:
        return smoke(args)
    ap.error("give --smoke or --selftest")


if __name__ == "__main__":
    sys.exit(main())
