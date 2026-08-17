#!/usr/bin/env python3
# prompt_offload.py — MILESTONE C, the PROMPT SEAM (audit component 2 + N5/N8).
#
#   python3 scripts/fpga/f2/prompt_offload.py --prompt "The capital of France is" \
#       --max-tokens 1 [--offload-layer 0 --offload-head 0] [--executor sim] [--dry-run]
#
# WHAT THIS DOES
#   Streams a REAL prompt through the REAL golden Qwen2.5-7B decode
#   (run_tinynpu.py's own GoldenModel/Session — imported, never edited) with ONE
#   attention operation of ONE (layer, head) taken away from golden and computed
#   by the TILE, then proves the emitted token is the same token the pure-golden
#   path emits.
#
#   The offloaded value is SUBSTITUTED, not compared: the wrapper returns an
#   AttnCore whose `out_hat` (and o8 / s_out / rq / acc_o) came from the tile,
#   and that array is the one `transformer.py:560` writes into `r.attn[sl_q]`
#   (verified per capture — see `consumed` in the banner). Golden is still run
#   for the same inputs, but only to GRADE afterwards (PROMPT_ON_CHIP.md §1 B).
#
# THE SEAM IS IN golden/, NOT IN run_tinynpu.py (PROMPT_DEMO_AUDIT.md §2 C2).
#   `Session.step` passes no kwargs and `collect` fires after the layer returns,
#   so the only no-frozen-edit route is REBINDING the module global. It is
#   rebound in BOTH modules that resolve it:
#       transformer.attention_core   (transformer.py:551, the unchunked head)
#       attention.attention_core     (attention.py:502, the chunk cores)
#   plus a PASS-THROUGH wrapper on transformer.decoder_layer_fx used only for
#   (step, layer) bookkeeping and for the `consumed` check — it changes no
#   arithmetic. Nothing under golden/ is modified on disk.
#
# HARD FENCE (N8 — a CORRECTNESS requirement, not a nicety).
#   prompt + generated tokens must be <= 128. Beyond that a head becomes a
#   ChunkedHead whose host merge needs per-chunk sm_m/sm_l, and the mailbox
#   exposes no such window (apex_f2_mailbox.sv:33-45) — so a chunked head
#   CANNOT be offloaded. `Session` itself permits T up to 384, so this file
#   refuses loudly rather than silently offloading a chunk.
#
# HONESTY OF EACH MODE
#   --dry-run        : the whole path runs — rebind, capture, epilogue via
#                      cap_decode, substitution, A/B, banner — with the tile
#                      call replaced by golden. Proves the PLUMBING only; the
#                      banner says `acc source = GOLDEN (dry-run)`.
#   --executor sim   : the job is built by compute_job.py (the input-only
#                      rq_en=0 compiler), executed by tile_exec_bridge.run_job
#                      on the verilated cl_apex, and the returned INT32
#                      accumulators go through the host epilogue.
#   --executor hw    : same, on the F2 instance via f2_host_run.py.
#   --poison K       : re-runs the ON decode with the tile value scaled by K.
#                      The DISCRIMINATOR: if the logits do not change, the
#                      substitution was never load-bearing and every PASS above
#                      is vacuous. Used by --selftest.
#
# CLI: --selftest    tiny random model, no MLX/HF/executor — exercises rebind,
#                    per-(layer,head) targeting, substitution, consumption,
#                    A/B identity and the poison discriminator in seconds.

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tinynpu as rt                                       # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import transformer as tf                      # noqa: E402
from apex_golden.fp import f64_to_f16_bits                     # noqa: E402

import cap_decode as cd                                        # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402

TOKENS_MAX = at.CHUNK_T_MAX          # 128 — the T_ROW_MAX per-job envelope
DEFAULT_WORK = REPO / "build" / "prompt_offload"


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═══════════════ compute_job adapter (the parallel lane's file) ══════════════
# compute_job.py is owned by another agent in this sprint. Its exact signatures
# are not frozen, so it is bound by INTROSPECTION and any mismatch is reported
# verbatim instead of guessed at. Nothing here reimplements the compiler: if it
# cannot be bound, the real-tile path is declared BLOCKED and says why.

SYN = {
    "q_f64":  ("q_f64", "q", "q_row", "q_real", "q_h"),
    "q_f16":  ("q_f16", "q_bits", "q_f16_bits"),
    "K_f16":  ("K_f16", "K", "K_bits", "K_f16_bits", "k_f16"),
    "V_f16":  ("V_f16", "V", "V_bits", "V_f16_bits", "v_f16"),
    "tier":   ("tier", "tier_s"),
    "D":      ("D", "d", "head_dim"),
    "T":      ("T", "t", "n_rows"),
    "G":      ("G", "group", "kvq_g"),
    "name":   ("name", "job_name", "stem", "label"),
    "out":    ("out", "out_dir", "outdir", "dest", "out_path", "regops_path",
               "path", "dir"),
    "outlier_idx": ("outlier_idx", "outliers", "oi"),
    "prep":   ("prep", "host_prep"),
    "captures": ("captures", "caps", "cap_records", "records"),
    "job":    ("job", "spec", "info", "build", "manifest", "jobinfo"),
    "golden": ("golden", "core", "golden_core", "ref", "expect", "expected"),
    "s_c":    ("s_c", "s_c_bits", "sc"),
}
_REQ = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _adapt(fn, offered: dict):
    """Bind `offered` (canonical name -> value) onto fn's real parameters."""
    sig = inspect.signature(fn)
    kw, used = {}, set()
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        for canon, names in SYN.items():
            if canon in offered and pname in names and canon not in used:
                kw[pname] = offered[canon]
                used.add(canon)
                break
    missing = [n for n, p in sig.parameters.items()
               if p.default is p.empty and p.kind in _REQ and n not in kw]
    if missing:
        raise TypeError(
            f"cannot bind {fn.__module__}.{fn.__name__}{sig}: no value for "
            f"required parameter(s) {missing}. Offered canonical keys: "
            f"{sorted(offered)} (extend SYN in prompt_offload.py, or align the "
            f"compute_job signature)")
    return fn(**kw)


def _dig(obj, keys):
    """First present key/attribute of a dict/object, else None."""
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
        if hasattr(obj, k):
            return getattr(obj, k)
    return None


def _prep_of(built):
    """The compiler's host-prep object, if the build handed one back."""
    got = _dig(built, ("prep", "host_prep"))
    if got is not None:
        return got
    if isinstance(built, (list, tuple)):
        for x in built:                    # (path, manifest, prep) tuples
            if not isinstance(x, (str, Path, dict)) and hasattr(x, "s_q"):
                return x
    return None


def _scalar(v):
    """A grader may report a scale as {'captured':bits, 'golden':…} — the
    epilogue needs the NUMBER the tile produced, never the whole verdict dict."""
    if isinstance(v, dict):
        for k in ("captured", "bits", "value", "tile"):
            if v.get(k) is not None:
                return v[k]
        return None
    return v


def _regops_of(built):
    """Pull regops path(s) out of whatever build_compute_job returned."""
    if isinstance(built, (str, Path)):
        return [str(built)]
    if isinstance(built, (list, tuple)):
        out = []
        for x in built:
            if isinstance(x, (str, Path)) and str(x).endswith(".jsonl"):
                out.append(str(x))
        if out:
            return out
        for x in built:                       # (path, meta) style tuples
            got = _regops_of(x) if not isinstance(x, (str, Path)) else None
            if got:
                return got
        return []
    got = _dig(built, ("regops", "regops_path", "regops_file", "path", "file",
                       "out", "jsonl", "files", "regops_files"))
    if got is None:
        return []
    return _regops_of(got)


class ComputeJobMissing(RuntimeError):
    """compute_job.py (the input-only rq_en=0 compiler) is not usable yet."""


def load_compute_job(path=None):
    """Import scripts/fpga/f2/compute_job.py, or explain what is missing.

    `path` (--compute-job / $APEX_COMPUTE_JOB) overrides the location. It exists
    so the --executor sim/hw CODE PATH can be exercised against the real
    executor with a stand-in compiler while the real compute_job.py is still
    being written in the parallel lane — a stand-in cannot produce a correct
    attention result, and the banner's `acc source` / grade lines say so.
    """
    p = Path(path or os.environ.get("APEX_COMPUTE_JOB")
             or HERE / "compute_job.py")
    if not p.is_file():
        raise ComputeJobMissing(
            f"{p} does not exist — the input-only (rq_en=0) job compiler is "
            f"the parallel lane's deliverable (audit component 1). The real "
            f"tile path cannot run without it; --dry-run exercises everything "
            f"else.")
    spec = importlib.util.spec_from_file_location("compute_job", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                    # noqa: BLE001 — report verbatim
        raise ComputeJobMissing(f"{p} failed to import: {type(e).__name__}: {e}")
    if not hasattr(mod, "build_compute_job"):
        raise ComputeJobMissing(
            f"{p} has no build_compute_job (found: "
            f"{[n for n in dir(mod) if not n.startswith('_')][:12]})")
    return mod


# ═══════════════════════════ the offload seam ════════════════════════════════

class Offloader:
    """Rebinds golden's attention_core and substitutes ONE (layer, head)."""

    def __init__(self, *, n_layers: int, H: int, layer: int, head: int,
                 mode: str, work: Path, step: int | None = None,
                 tile_div: int = bridge.TILE_DIV_DEFAULT, slot: int = 0,
                 poison: float | None = None, timeout_s: int = 1800,
                 verbose: bool = True, compute_job=None):
        self.n_layers, self.H = n_layers, H
        self.t_layer, self.t_head, self.t_step = layer, head, step
        self.mode = mode                       # 'golden' | 'sim' | 'hw'
        self.work = Path(work)
        self.tile_div, self.slot, self.timeout_s = tile_div, slot, timeout_s
        self.poison = poison
        self.verbose = verbose
        self.compute_job = compute_job
        self.layer_calls = 0                   # -> (step, layer)
        self.head_idx = 0                      # within the current layer
        self.records: list[dict] = []
        self._cj = None
        self._pending = None                   # capture awaiting the `consumed`
        self._inside = False                   # re-entrancy guard (see _core)
        self._orig_core_tf = self._orig_core_at = self._orig_layer = None

    # ── rebinding (context manager; ALWAYS restored) ────────────────────────
    def __enter__(self):
        self._orig_core_tf = tf.attention_core
        self._orig_core_at = at.attention_core
        self._orig_layer = tf.decoder_layer_fx
        assert self._orig_core_tf is self._orig_core_at, \
            "transformer/attention already resolve different attention_core"
        self._sig = inspect.signature(self._orig_core_at)
        tf.attention_core = self._core          # transformer.py:551
        at.attention_core = self._core          # attention.py:502 (chunk cores)
        tf.decoder_layer_fx = self._layer       # bookkeeping only
        return self

    def __exit__(self, *exc):
        tf.attention_core = self._orig_core_tf
        at.attention_core = self._orig_core_at
        tf.decoder_layer_fx = self._orig_layer
        return False

    # ── (step, layer) bookkeeping + the consumption proof ──────────────────
    @property
    def step(self) -> int:
        return self.layer_calls // self.n_layers

    @property
    def layer(self) -> int:
        return self.layer_calls % self.n_layers

    def _layer(self, X, w, tier, *a, **kw):
        X = np.asarray(X)
        T = X.shape[0] - 1
        if T > TOKENS_MAX:
            raise SystemExit(
                f"REFUSE: T={T} > {TOKENS_MAX} — the head would become a "
                f"ChunkedHead whose merge needs per-chunk sm_m/sm_l, which the "
                f"mailbox does not expose. An offloaded chunk is not possible "
                f"(audit N8). Reduce --max-tokens / the prompt.")
        self.head_idx = 0
        self._pending = None
        try:
            r = self._orig_layer(X, w, tier, *a, **kw)
        finally:
            layer, step = self.layer, self.step
            self.layer_calls += 1
        # the substitution actually reached the model's attention vector
        if self._pending is not None:
            rec, hd = self._pending, w.head_dim
            sl = slice(rec["head"] * hd, (rec["head"] + 1) * hd)
            rec["consumed"] = bool(np.array_equal(r.attn[sl], rec["_out_hat"]))
            rec["head_is_core"] = r.heads[rec["head"]] is rec["_core"]
            self._pending = None
        if self.head_idx and self.head_idx != self.H:
            eprint(f"[offload] WARNING: layer {layer} step {step} made "
                   f"{self.head_idx} attention_core calls, expected H={self.H}"
                   f" — (layer, head) indices may be misaligned")
        return r

    # ── the wrapper that replaces golden's attention_core ──────────────────
    def _core(self, *a, **kw):
        # RE-ENTRANCY. The rebind is a MODULE GLOBAL, so anything that calls
        # golden's attention_core while we are handling an offload lands here
        # too — and compute_job.grade_compute_job does exactly that (it
        # computes golden AFTER the run, from the same inputs, to grade). Left
        # unguarded those calls consume head indices and silently shift every
        # later head in the layer. Inside our own handling we are transparent.
        if self._inside:
            return self._orig_core_at(*a, **kw)
        h = self.head_idx
        self.head_idx += 1
        b = self._sig.bind(*a, **kw)
        b.apply_defaults()
        A = b.arguments
        selected = (self.layer == self.t_layer and h == self.t_head
                    and (self.t_step is None or self.step == self.t_step))
        if not selected:
            return self._orig_core_at(*a, **kw)

        self._inside = True
        try:
            core = self._orig_core_at(*a, **kw)  # golden — for grading only
            assert core.T <= TOKENS_MAX, \
                "chunk core reached the offload seam (N8)"
            name = f"poff_s{self.step:03d}_L{self.layer:02d}_h{h:02d}"
            t0 = time.perf_counter()
            tile = self._tile_value(A, core, name)
            dt = time.perf_counter() - t0
        finally:
            self._inside = False

        acc = np.asarray(tile["acc_i32"], dtype=np.int64)
        ep = cd.epilogue(acc, tile["s_c"])
        out_hat = np.asarray(ep["out_hat"], dtype=np.float64)
        if self.poison is not None:
            out_hat = out_hat * float(self.poison)

        rec = {
            "name": name, "step": self.step, "layer": self.layer, "head": h,
            "T": int(core.T), "D": int(core.D), "tier": core.tier,
            "G": int(core.G), "mode": self.mode, "seconds": round(dt, 3),
            "acc_source": tile["acc_source"], "s_c_source": tile["s_c_source"],
            "n_captures": tile["n_captures"], "executor_ok": tile.get("ok"),
            "executor_rc": tile.get("rc"), "notes": tile.get("notes", []),
            "compute_job_ok": tile.get("cj_ok"),
            "grade_acc": cd.grade(acc, core.acc_o),
            "grade_out_hat": cd.grade(ep["out_hat"], core.out_hat),
            "grade_o8": cd.grade(ep["o8"], core.o8),
            "rq_tile": tuple(int(v) for v in ep["rq"]),
            "rq_golden": tuple(int(v) for v in core.rq),
            "poisoned": self.poison,
            "_out_hat": out_hat, "_core": core,
        }

        # ── THE SUBSTITUTION: tile-derived values re-enter the model ────────
        core.acc_o = acc
        core.rq = ep["rq"]
        core.o8 = np.asarray(ep["o8"], dtype=np.int64)
        core.s_out = float(ep["s_out"])
        core.out_hat = out_hat
        rec["substituted"] = core.out_hat is out_hat
        self.records.append(rec)
        self._pending = rec
        if self.verbose:
            g = rec["grade_out_hat"]
            eprint(f"[offload] {name} T={core.T} {core.tier} "
                   f"{self.mode}: acc<-{rec['acc_source']} "
                   f"caps={rec['n_captures']} "
                   f"out_hat {'==' if g['equal'] else '!='} golden "
                   f"({dt:.2f}s)")
        return core

    # ── where the numbers come from ────────────────────────────────────────
    def _tile_value(self, A: dict, core, name: str) -> dict:
        """Return {'acc_i32','s_c', provenance…} for this attention op."""
        if self.mode == "golden":               # --dry-run
            return {"acc_i32": core.acc_o, "s_c": core.s_c,
                    "acc_source": "GOLDEN (dry-run)",
                    "s_c_source": "GOLDEN (dry-run)", "n_captures": 0}
        if self._cj is None:
            self._cj = load_compute_job(self.compute_job)
        cj = self._cj
        self.work.mkdir(parents=True, exist_ok=True)
        q = np.asarray(A["q_f64"], dtype=np.float64)
        K = np.asarray(A["K_f16"], dtype=np.uint16)
        V = np.asarray(A["V_f16"], dtype=np.uint16)
        offered = {
            "q_f64": q, "q_f16": f64_to_f16_bits(q).astype(np.uint16),
            "K_f16": K, "V_f16": V, "tier": A["tier"],
            "D": int(core.D), "T": int(core.T), "G": int(A["G"]),
            "outlier_idx": list(A.get("outlier_idx") or ()),
            "name": name, "out": str(self.work),
        }
        built = _adapt(cj.build_compute_job, offered)
        regops = _regops_of(built)
        if not regops:
            raise ComputeJobMissing(
                f"build_compute_job returned {type(built).__name__} with no "
                f"regops path I can find: {str(built)[:200]}")
        cap_out = self.work / f"{name}.cap.jsonl"
        res = bridge.run_job(regops, executor=self.mode,
                             cap_out=str(cap_out), tile_div=self.tile_div,
                             slot=self.slot, timeout_s=self.timeout_s)
        caps = res["captures"]
        if not res["ok"]:
            raise RuntimeError(
                f"{name}: executor did not honour the capture contract — "
                f"rc={res['rc']} ok={res['ok']} n={len(caps)} "
                f"notes={res['notes']}\n{res['log'][-2000:]}")

        # Prefer the compiler's own grader/decoder (it knows its own capture
        # layout); cap_decode is the fallback so a grader-less build still runs.
        acc = s_c = None
        acc_src = sc_src = None
        notes = list(res["notes"])
        cj_grade = None
        grader = getattr(cj, "grade_compute_job", None)
        if grader is not None:
            cj_grade = _adapt(grader, {**offered, "captures": caps,
                                       "job": built, "golden": core,
                                       "prep": _prep_of(built)})
            acc = _dig(cj_grade, ("acc_i32", "acc_int32"))
            s_c = _scalar(_dig(cj_grade, ("s_c", "s_c_bits", "sc")))
            if acc is not None:
                acc_src = f"TILE via compute_job.grade_compute_job ({self.mode})"
            if s_c is not None:
                sc_src = "TILE (captured s_c, via grade_compute_job)"
            for p in (_dig(cj_grade, ("problems",)) or []):
                notes.append(f"compute_job: {p}")
        if acc is None:                          # fall back to cap_decode (N4)
            acc = cd.ro_lanes_to_i32(caps, expect_n=int(core.D))
            acc_src = f"TILE via cap_decode.ro_lanes_to_i32 ({self.mode})"
        if s_c is None:
            taps = [t for t in cd.fp16_bits(caps) if t["sem"] == "ss"]
            if taps:
                s_c, sc_src = np.uint16(taps[-1]["bits"]), "TILE (ss tap)"
            else:
                s_c, sc_src = core.s_c, "GOLDEN (no ss tap captured)"
        return {"acc_i32": acc, "s_c": s_c, "acc_source": acc_src,
                "s_c_source": sc_src, "n_captures": len(caps),
                "ok": res["ok"], "rc": res["rc"], "notes": notes,
                "cap_out": res["cap_out"], "regops": regops,
                "cj_ok": (None if cj_grade is None
                          else bool(_dig(cj_grade, ("ok",))))}


# ═══════════════════════════ the decode (A/B) ════════════════════════════════

def decode(model, tok_ids: list[int], max_tokens: int, tier: str, group: int,
           eos: set, offloader: Offloader | None, verbose: bool = True) -> dict:
    """Greedy decode mirroring run_tinynpu.run_generate's token semantics."""
    sess = rt.Session(model, tier, group)
    hidden, gen = None, []
    t0 = time.perf_counter()
    for i, tid in enumerate(tok_ids):
        hidden = sess.step(model.embed_row(tid))
        if verbose:
            eprint(f"  [prefill {i+1}/{len(tok_ids)}] pos={sess.pos-1} "
                   f"T={sess.pos} tok={tid} {time.perf_counter()-t0:.0f}s")
    logits = None
    for i in range(max_tokens):
        logits = model.head_logits(hidden)
        tid = int(np.argmax(logits))
        gen.append(tid)
        if tid in eos or i == max_tokens - 1:
            break
        hidden = sess.step(model.embed_row(tid))
    return {"ids": gen, "logits": logits, "seconds": time.perf_counter() - t0,
            "records": list(offloader.records) if offloader else []}


def run_ab(model, tok, args, work: Path) -> int:
    ids = tok.encode(args.prompt) if tok else list(args.ids)
    n_total = len(ids) + args.max_tokens
    if n_total > TOKENS_MAX:
        eprint(f"REFUSE: prompt({len(ids)}) + max_tokens({args.max_tokens}) = "
               f"{n_total} > {TOKENS_MAX}.\n"
               f"        Beyond 128 tokens a head becomes a ChunkedHead and "
               f"the C-CHUNK host merge needs per-chunk sm_m/sm_l, which the "
               f"mailbox does not expose (apex_f2_mailbox.sv:33-45) — a "
               f"chunked head CANNOT be offloaded to the tile (audit N8). "
               f"Shorten the prompt or --max-tokens.")
        return 2
    H, L = model.meta["H"], model.n_layers
    if not 0 <= args.offload_layer < L:
        eprint(f"REFUSE: --offload-layer {args.offload_layer} outside [0,{L})")
        return 2
    if not 0 <= args.offload_head < H:
        eprint(f"REFUSE: --offload-head {args.offload_head} outside [0,{H})")
        return 2
    tier = rt.TIER_MAP[args.tier]
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    mode = "golden" if args.dry_run else args.executor

    eprint(f"[C] model={model.meta['model']} L={L} H={H} tier={tier} "
           f"G={args.group}; prompt={len(ids)} tok + {args.max_tokens} new "
           f"= {n_total} <= {TOKENS_MAX} OK")
    eprint(f"[C] offload target: layer {args.offload_layer} head "
           f"{args.offload_head}"
           + (f" step {args.offload_step}" if args.offload_step is not None
              else " (every step)")
           + f"  mode={mode}")

    # ── run 1: OFFLOAD ON ──────────────────────────────────────────────────
    eprint("[C] === run 1/2: OFFLOAD ON ===")
    with Offloader(n_layers=L, H=H, layer=args.offload_layer,
                   head=args.offload_head, mode=mode, work=work,
                   step=args.offload_step, tile_div=args.tile_div,
                   slot=args.slot, timeout_s=args.timeout_s,
                   compute_job=args.compute_job) as off:
        on = decode(model, ids, args.max_tokens, tier, args.group, eos, off)

    # ── run 2: OFFLOAD OFF (pure golden — no rebind at all) ────────────────
    eprint("[C] === run 2/2: OFFLOAD OFF (pure golden) ===")
    assert tf.attention_core is at.attention_core, "rebind not restored"
    offr = decode(model, ids, args.max_tokens, tier, args.group, eos, None)

    poison = None
    if args.poison is not None:
        eprint(f"[C] === discriminator: OFFLOAD ON, tile value x{args.poison} ===")
        with Offloader(n_layers=L, H=H, layer=args.offload_layer,
                       head=args.offload_head, mode=mode, work=work,
                       step=args.offload_step, tile_div=args.tile_div,
                       slot=args.slot, timeout_s=args.timeout_s,
                       poison=args.poison,
                       compute_job=args.compute_job) as off2:
            poison = decode(model, ids, args.max_tokens, tier, args.group,
                            eos, off2)

    return report(model, tok, args, ids, on, offr, poison, mode, work)


def report(model, tok, args, ids, on, offr, poison, mode, work) -> int:
    recs = on["records"]
    ident = on["ids"] == offr["ids"]
    fired = len(recs) > 0
    subs = all(r["substituted"] for r in recs)
    consumed = all(r.get("consumed") for r in recs)
    heads_ok = all(r.get("head_is_core") for r in recs)
    eq_out = [r["grade_out_hat"]["equal"] for r in recs]
    eq_acc = [r["grade_acc"]["equal"] for r in recs]
    cj_ok = all(r.get("compute_job_ok") is not False for r in recs)
    ex_ok = all(r.get("executor_ok") is not False for r in recs)

    def txt(v):
        return tok.decode(v) if tok else str(v)

    W = 78
    print("\n" + "=" * W)
    print("MILESTONE C — PROMPT SEAM: one 7B attention op offloaded to the tile")
    print("=" * W)
    print(f"  prompt          : {args.prompt!r}")
    print(f"  prompt ids      : {ids} ({len(ids)} tokens)")
    print(f"  fence           : {len(ids)}+{args.max_tokens} = "
          f"{len(ids)+args.max_tokens} <= {TOKENS_MAX} tokens  OK "
          f"(chunked heads cannot be offloaded — N8)")
    print(f"  model           : {model.meta['model']}  L={model.n_layers} "
          f"H={model.meta['H']} head_dim={model.meta['head_dim']}")
    print(f"  offloaded op    : attention_core  layer={args.offload_layer} "
          f"head={args.offload_head}"
          + (f" step={args.offload_step}" if args.offload_step is not None
             else " (every step)"))
    print(f"  executor        : {mode}"
          + ("   (--dry-run: tile call replaced by golden — plumbing only)"
             if mode == "golden" else ""))
    print(f"  captures        : {len(recs)} offloaded op(s)"
          + (f", {sum(r['n_captures'] for r in recs)} cap records"
             if mode != "golden" else ""))
    for r in recs:
        print(f"    - {r['name']}  T={r['T']} D={r['D']} {r['tier']} "
              f"G={r['G']}  {r['seconds']}s")
        print(f"      acc source   : {r['acc_source']}")
        print(f"      s_c source   : {r['s_c_source']}")
        print(f"      substituted  : {r['substituted']}   consumed by model: "
              f"{r.get('consumed')}   core object kept: {r.get('head_is_core')}")
        print(f"      tile vs golden: acc_o {_g(r['grade_acc'])} | "
              f"o8 {_g(r['grade_o8'])} | out_hat {_g(r['grade_out_hat'])}")
        print(f"      rq (scale,shift): tile={r['rq_tile']} "
              f"golden={r['rq_golden']}")
        if r["mode"] != "golden":
            print(f"      executor ok   : {r['executor_ok']} (rc="
                  f"{r['executor_rc']}) ; compute_job grade ok: "
                  f"{r['compute_job_ok']}")
        for n in r["notes"]:
            print(f"      NOTE: {n}")
    print(f"  tile vs golden  : out_hat bit-exact {sum(eq_out)}/{len(recs)}"
          f" ; acc_o bit-exact {sum(eq_acc)}/{len(recs)}")
    print(f"  token OFFLOAD ON : ids={on['ids']} text={txt(on['ids'])!r} "
          f"({on['seconds']:.0f}s)")
    print(f"  token OFFLOAD OFF: ids={offr['ids']} text={txt(offr['ids'])!r} "
          f"({offr['seconds']:.0f}s)")
    if poison is not None:
        dl = (float(np.max(np.abs(poison["logits"] - on["logits"])))
              if poison["logits"] is not None and on["logits"] is not None
              else float("nan"))
        verdict = ("substitution IS load-bearing" if dl > 0 else
                   "VACUOUS — the returned value never reached the token")
        print(f"  discriminator   : tile value x{args.poison} -> "
              f"ids={poison['ids']}, max|dlogit|={dl:.4g} ({verdict})")
    ok = (ident and fired and subs and consumed and heads_ok and all(eq_out)
          and cj_ok and ex_ok)
    print("-" * W)
    if not fired:
        print("  FAIL: the target (layer, head) was never reached — no offload "
              "happened, so token identity proves nothing.")
    if fired and not (subs and consumed and heads_ok):
        print("  FAIL: the tile-derived value was not the value the model "
              "consumed (substituted/consumed/head_is_core above).")
    if not all(eq_out):
        print("  FAIL: tile out_hat differs from golden out_hat.")
    if not (cj_ok and ex_ok):
        print(f"  FAIL: executor_ok={ex_ok} compute_job_grade_ok={cj_ok} — see "
              f"the NOTE lines above.")
    print(f"  TOKEN IDENTITY: {'PASS' if ident else 'FAIL'}"
          f"   (offload-on == pure-golden)")
    print(f"  MILESTONE C   : {'PASS' if ok else 'FAIL'}")
    print("=" * W)

    work.mkdir(parents=True, exist_ok=True)
    out = work / "prompt_offload_result.json"
    out.write_text(json.dumps({
        "prompt": args.prompt, "prompt_ids": ids, "mode": mode,
        "offload": {"layer": args.offload_layer, "head": args.offload_head,
                    "step": args.offload_step},
        "tokens_on": on["ids"], "tokens_off": offr["ids"],
        "text_on": txt(on["ids"]), "text_off": txt(offr["ids"]),
        "token_identity": ident, "milestone_c": ok,
        "seconds": {"on": on["seconds"], "off": offr["seconds"]},
        "captures": [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in recs],
        "git": rt.git_rev(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if ok else 1


def _g(gr) -> str:
    return ("bit-exact" if gr["equal"]
            else f"DIFF n={gr['n_diff']} worst={gr['worst']:.6g}")


# ═══════════════════════════ selftest (tiny model) ═══════════════════════════

def _tiny_model(wdir: Path, D=128, H=2, H_kv=1, hd=64, dff=256, vocab=97, L=2):
    """A random tiny GoldenModel — same layout run_tinynpu.selftest builds."""
    rng = np.random.default_rng(0xC0FFEE)
    w = Path(wdir)
    w.mkdir(parents=True, exist_ok=True)
    meta = {"model": "selftest-tiny", "format": "apex-s8-golden-weights-v1",
            "D_model": D, "H": H, "H_kv": H_kv, "head_dim": hd, "d_ffn": dff,
            "n_layers": L, "vocab": vocab, "rope_theta": 1e6,
            "eos_token_id": None, "tie_word_embeddings": False,
            "source_quant": None, "disclosure": "selftest random weights",
            "layers": [], "created": "-", "git": rt.git_rev()}
    for li in range(L):
        lr = {}
        for tag, sh in {"Wq": (D, D), "Wk": (D, H_kv * hd),
                        "Wv": (D, H_kv * hd), "Wo": (D, D), "Wg": (D, dff),
                        "Wu": (D, dff), "Wd": (dff, D)}.items():
            w8, s = rt.quant_w_i8(rng.normal(0, 1, sh) / np.sqrt(sh[0]))
            np.save(w / f"L{li:02d}_{tag}.npy", w8)
            lr[f"s_{tag}"] = s
        for tag in ("gamma1", "gamma2"):
            gq, k, ncl = rt.quant_gamma_q213(0.7 + 0.6 * rng.random(D))
            np.save(w / f"L{li:02d}_{tag}.npy", gq)
            lr.update({f"{tag}_fold_k": k, f"{tag}_clamped": ncl,
                       f"{tag}_amax": 0.0})
        meta["layers"].append(lr)
    gq, kf, _ = rt.quant_gamma_q213(np.full(D, 1.1))
    np.save(w / "final_gamma.npy", gq)
    meta["final_gamma"] = {"fold_k": kf, "clamped": 0, "amax": 1.1}
    h8, s_head = rt.quant_w_i8(rng.normal(0, 0.5, (vocab, D)))
    np.save(w / "lm_head_w8.npy", h8)
    meta["s_head"] = s_head
    np.save(w / "embed.npy", rng.normal(0, 1, (vocab, D)).astype(np.float16))
    (w / "meta.json").write_text(json.dumps(meta))
    return rt.GoldenModel(w)


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
        model = _tiny_model(td / "w")
        ids, work = [3, 1, 4, 1, 5], td / "work"
        tier, L, H = at.TIER_CQ8, model.n_layers, model.meta["H"]

        base = decode(model, ids, 2, tier, 128, set(), None, verbose=False)
        with Offloader(n_layers=L, H=H, layer=1, head=1, mode="golden",
                       work=work, verbose=False) as off:
            on = decode(model, ids, 2, tier, 128, set(), off,
                        verbose=False)
        chk("golden attention_core restored after the context manager",
            tf.attention_core is at.attention_core
            and tf.attention_core.__module__.endswith("apex_golden.attention"))
        chk("token identity (dry-run offload vs pure golden)",
            on["ids"] == base["ids"], f"{on['ids']} vs {base['ids']}")
        n_steps = len(ids) + 1                       # 2 tokens: 1 extra step
        chk("offload fired once per step at the target head",
            len(on["records"]) == n_steps,
            f"{len(on['records'])} records, expected {n_steps}")
        chk("targeting hit exactly (layer 1, head 1)",
            all(r["layer"] == 1 and r["head"] == 1 for r in on["records"]),
            str([(r['step'], r['layer'], r['head']) for r in on['records']]))
        chk("T grows with position (self-inclusive)",
            [r["T"] for r in on["records"]] == list(range(1, n_steps + 1)),
            str([r["T"] for r in on["records"]]))
        chk("substituted + consumed by the model on every capture",
            all(r["substituted"] and r["consumed"] and r["head_is_core"]
                for r in on["records"]))
        chk("host epilogue reproduces golden out_hat bit-exact",
            all(r["grade_out_hat"]["equal"] and r["grade_o8"]["equal"]
                and r["grade_acc"]["equal"] for r in on["records"]))

        # the DISCRIMINATOR: if poisoning the returned value changes nothing,
        # the whole harness is vacuous.
        with Offloader(n_layers=L, H=H, layer=1, head=1, mode="golden",
                       work=work, poison=0.5, verbose=False) as off2:
            po = decode(model, ids, 2, tier, 128, set(), off2,
                        verbose=False)
        dl = float(np.max(np.abs(po["logits"] - on["logits"])))
        chk("poisoned tile value perturbs the logits (substitution is "
            "load-bearing)", dl > 0, f"max|dlogit|={dl:.4g}, ids={po['ids']}")

        # re-entrancy: a golden call made from INSIDE our own handling (which is
        # what compute_job.grade_compute_job does) must not consume a head index
        with Offloader(n_layers=L, H=H, layer=1, head=1, mode="golden",
                       work=work, verbose=False) as off4:
            rng = np.random.default_rng(3)
            Kf = f64_to_f16_bits(rng.normal(0, 1, (2, 64))).astype(np.uint16)
            Vf = f64_to_f16_bits(rng.normal(0, 1, (2, 64))).astype(np.uint16)
            off4.head_idx, off4._inside = 3, True
            r_in = at.attention_core(rng.normal(0, 1, 64), Kf, Vf, tier, G=128)
            chk("re-entrant golden call is transparent (no head index eaten)",
                off4.head_idx == 3 and r_in.out_hat is not None,
                f"head_idx={off4.head_idx}")
            off4._inside = False

        # a target that never fires must not silently pass
        with Offloader(n_layers=L, H=H, layer=L + 5, head=0, mode="golden",
                       work=work, verbose=False) as off3:
            miss = decode(model, ids, 1, tier, 128, set(), off3,
                          verbose=False)
        chk("unreachable target produces zero captures (report must FAIL)",
            len(miss["records"]) == 0)

        # the T>128 fence, exercised through the real layer wrapper
        sess = rt.Session(model, tier, 128)
        rows = [tf._f16(v) for v in
                np.random.default_rng(1).normal(0, 1, (129, 128))]
        for li in range(L):
            sess.hist[li] = list(rows)
        sess.pos = 129
        with Offloader(n_layers=L, H=H, layer=0, head=0, mode="golden",
                       work=work, verbose=False):
            try:
                sess.step(model.embed_row(7))
                chk("T>128 refused at the seam", False, "no refusal")
            except SystemExit as e:
                chk("T>128 refused at the seam", "ChunkedHead" in str(e),
                    str(e)[:70] + "…")

        # compute_job absence must be a clear, specific error
        try:
            load_compute_job()
            chk("compute_job present -> real tile path importable", True,
                "build_compute_job found")
        except ComputeJobMissing as e:
            chk("compute_job absence reported precisely", True,
                str(e).split("—")[0].strip()[:60] + "…")

    print("=" * 60)
    if fails:
        print(f"PROMPT_OFFLOAD SELFTEST FAIL: {fails}")
        return 1
    print("PROMPT_OFFLOAD SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Milestone C: offload one 7B attention op to the tile and "
                    "prove token identity.")
    ap.add_argument("--prompt")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--offload-layer", type=int, default=0)
    ap.add_argument("--offload-head", type=int, default=0)
    ap.add_argument("--offload-step", type=int, default=None,
                    help="offload only at this decode step (default: every)")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--dry-run", action="store_true",
                    help="replace the tile call with golden — exercises the "
                         "whole path without an executor")
    ap.add_argument("--poison", type=float, default=None, metavar="K",
                    help="extra run with the tile value scaled by K (the "
                         "discriminator: proves the value is load-bearing)")
    ap.add_argument("--tier", default="kvq8", choices=list(rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--weights-dir",
                    default=str(REPO / "build/s8_weights/Qwen2.5-7B-4bit"))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="bypass the tokenizer (debug): raw prompt ids")
    ap.add_argument("--compute-job", default=None, metavar="PATH",
                    help="override the job-compiler module (also "
                         "$APEX_COMPUTE_JOB). For exercising the executor code "
                         "path with a stand-in before compute_job.py lands.")
    ap.add_argument("--make-tiny-weights", metavar="DIR",
                    help="write a tiny RANDOM golden weight cache to DIR and "
                         "exit — lets the full CLI (including --executor sim) "
                         "run in seconds instead of minutes. NOT a real model.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import remote_hw_exec                      # Milestone D: hw routing
    remote_hw_exec.attach(bridge, args)        # no-op unless $APEX_F2_HOST set

    if args.selftest:
        return selftest(args)
    if args.make_tiny_weights:
        m = _tiny_model(Path(args.make_tiny_weights))
        print(f"tiny random weights -> {args.make_tiny_weights} "
              f"(L={m.n_layers} H={m.meta['H']} head_dim={m.meta['head_dim']} "
              f"vocab={m.meta['vocab']}) — NOT a real model")
        return 0
    if args.prompt is None and not args.ids:
        ap.error("need --prompt (or --ids for a tokenizer-free run)")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be >= 1")
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
