#!/usr/bin/env python3
# apex_repl.py — the LIVE, INTERACTIVE prompt demo (sprint: prompt-on-FPGA).
#
#   python3 scripts/fpga/f2/apex_repl.py --executor sim            # no card
#   python3 scripts/fpga/f2/apex_repl.py --executor hw \
#       --host ubuntu@<f2-ip> --key ~/.ssh/apex-f2.pem             # real card
#
# WHAT THIS IS
#   A demo-quality wrapper around capability that ALREADY EXISTS and is proven
#   (docs/results/prompt_on_chip/RESULT.md + C2_PROMPT_ALL_OPS_RESULT.md). The
#   owner types a prompt, presses enter, and WATCHES a real Qwen2.5-7B decode
#   run with tile-served ops, then sees the emitted token plus an explicit
#   identity check against the pure-golden reference run.
#
#   TWO OFFLOAD MODES, selected at startup:
#     --offload attention  (default)  ONE attention op (layer, head) per decode
#                                     step -> the tile          [Milestone C]
#     --offload layer                 ALL SIX op families of ONE decoder layer
#                                     (proj / rope / attn / resid / norm /
#                                     swiglu) at ONE decode step -> the tile
#                                                                [Milestone C2]
#
#   NOTHING here computes: every arithmetic decision lives in the imported,
#   unmodified machinery —
#     prompt_offload.Offloader / prompt_offload.decode   (the proven seam)
#     layer_offload.LayerOffloader / Scope / Runner      (the six-op seam)
#     compute_job.build_compute_job / grade_compute_job  (produce-mode jobs)
#     gen_layer_ops / gemm_job (via layer_offload)       (the other five ops)
#     tile_exec_bridge.run_job                           (sim executor)
#     remote_hw_exec (attach + numeric clock gate)       (hw executor)
#   This file only adds: a warm session, a live ledger, a verdict, a footer.
#
# WARM SESSION
#   hw : bring the card up ONCE at startup — load the AFI, program the clkgen
#        recipe KEYED TO THAT IMAGE (clock_key.py; an AFI load RESETS the
#        MMCMs to 125 MHz — mandatory), then VERIFY the tile clock
#        NUMERICALLY against the image-keyed expectation (A2 images:
#        15.625 ± 0.2 MHz; an A0 image: 62.5 ± 0.2) and print the measured
#        reading. An AGFI not registered in clock_key.IMAGE_RECIPE REFUSES at
#        startup. Refuse to enter the REPL otherwise. Every job is ALSO
#        re-gated by remote_hw_exec's per-invocation clock check.
#   sim: resolve the Verilated cl_apex binary and the job compiler once.
#   Model weights + tokenizer load once; per-prompt latency is compute only.
#
# HONESTY RULES (enforced in the output, not just claimed)
#   - The verdict is ON-vs-OFF token identity, both computed live in this
#     process. Zero baked output expectations anywhere.
#   - Every op line names WHO computed it (TILE vs HOST vs TILE-SLOT dry-run).
#   - The footer states exactly what the host still did, every run.
#   - Elapsed times are labeled correctness-clock wall time (hw) or simulation
#     wall time (sim) — NEVER presented as throughput/performance.
#   - A refused job, a missing capture, or a tile-vs-golden mismatch is a LOUD
#     on-screen failure, never a soft skip.
#
# CLI: --selftest runs in seconds with a tiny random model, no executor.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tinynpu as rt                                       # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import transformer as tf                      # noqa: E402

import clock_key                                               # noqa: E402
import layer_offload as lof                                    # noqa: E402
import prompt_offload as pof                                   # noqa: E402
import remote_hw_exec as rhe                                   # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402

# The live image (narrow + R4, all six op types proven on silicon).
DEFAULT_AGFI = "agfi-09e947873048a6877"

# The session's clock expectation — set by bringup_hw() from the image key,
# read by the honest footer and the session banner. The default is the A2
# discipline every pre-2026-08-05 image was constrained at; a session that
# never runs bringup_hw (sim / dry-run / selftest) keeps it and never prints
# it (the footer's hw branch is the only reader).
SESSION_CLOCK = {"mhz": clock_key.RECIPE_A_A1_MHZ[2], "recipe": 2,
                 "underclocked": False}
DEFAULT_WORK = REPO / "build" / "apex_repl"
W = 78


def say(*a) -> None:
    """stdout, flushed — keeps ordering sane when stderr is merged in."""
    print(*a, flush=True)


# ═══════════════════════════ the live ledger ═════════════════════════════════

class Ledger:
    """Plain-terminal live ledger: permanent op lines + one status line with a
    running elapsed timer. No curses. A non-tty stderr degrades to plain
    lines (no \\r spam), so piped transcripts stay readable."""

    def __init__(self, tty: bool | None = None, out=None):
        self.out = out if out is not None else sys.stderr
        self.tty = self.out.isatty() if tty is None else tty
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()
        self.lines: list[str] = []          # every permanent line (for tests)
        self._status = ""
        self._stopev = threading.Event()
        self._thread = None

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def _draw_status(self) -> None:        # caller holds the lock
        if self.tty and self._status:
            self.out.write("\r\x1b[2K[%7.1fs] %s" % (self.elapsed(),
                                                     self._status))
            self.out.flush()

    def start(self) -> None:
        """Reset the clock; in a tty, spin the elapsed-timer redraw thread."""
        self.t0 = time.perf_counter()
        self._status = ""
        if self.tty and self._thread is None:
            self._stopev.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def _spin(self) -> None:
        while not self._stopev.wait(0.25):
            with self.lock:
                self._draw_status()

    def stop(self) -> None:
        if self._thread is not None:
            self._stopev.set()
            self._thread.join(timeout=2)
            self._thread = None
        with self.lock:
            if self.tty:
                self.out.write("\r\x1b[2K")
                self.out.flush()
            self._status = ""

    def op(self, text: str) -> None:
        """A permanent ledger line, stamped with elapsed time."""
        with self.lock:
            line = "[%7.1fs] %s" % (self.elapsed(), text)
            self.lines.append(line)
            if self.tty:
                self.out.write("\r\x1b[2K" + line + "\n")
                self._draw_status()
            else:
                self.out.write(line + "\n")
            self.out.flush()

    def tick(self, status: str) -> None:
        """Transient status line (tty only; quiet between op lines when
        piped)."""
        with self.lock:
            self._status = status
            self._draw_status()


# ══════════════ ledger hooks around the PROVEN offload seam ═════════════════

class LedgerOffloader(pof.Offloader):
    """prompt_offload.Offloader + live ledger. Arithmetic untouched: every
    compute/substitution decision stays in the parent class; this subclass
    only observes (counts calls, prints lines)."""

    def __init__(self, *a, ledger: Ledger, tag: str = "[ON ]",
                 n_steps: int = 0, step_tokens=(), **kw):
        super().__init__(*a, verbose=False, **kw)
        self.ledger = ledger
        self.tag = tag
        self.n_steps = n_steps
        self.step_tokens = list(step_tokens)
        self.host_attn = self.tile_attn = 0
        self._step_host = self._step_tile = 0

    def _lbl(self, st: int) -> str:
        if st < len(self.step_tokens):
            return " tok=%s" % self.step_tokens[st]
        return " (generated token)"

    def _layer(self, X, w, tier, *args, **kw):
        st, ly = self.step, self.layer
        if ly == 0:
            self._step_host = self._step_tile = 0
        self.ledger.tick("%s step %d/%d · layer %2d/%d · proj+attn+MLP+norms"
                         % (self.tag, st + 1, self.n_steps, ly + 1,
                            self.n_layers))
        r = super()._layer(X, w, tier, *args, **kw)
        if ly == self.n_layers - 1:
            self.ledger.op(
                "%s step %d/%d%s done · %d layers · attention heads: "
                "%d HOST, %d TILE · proj/MLP/norms/residuals: HOST"
                % (self.tag, st + 1, self.n_steps, self._lbl(st),
                   self.n_layers, self._step_host, self._step_tile))
        return r

    def _core(self, *a, **kw):
        if self._inside:            # grading re-entry: stay transparent
            return super()._core(*a, **kw)
        n0 = len(self.records)
        r = super()._core(*a, **kw)
        if len(self.records) > n0:
            rec = self.records[-1]
            self.tile_attn += 1
            self._step_tile += 1
            if self.mode == "golden":
                who = "TILE-SLOT (dry-run: value from GOLDEN — no tile ran)"
            else:
                who = "TILE"
            g = rec["grade_out_hat"]
            self.ledger.op(
                "%s   >> %s attention_core L%02d/h%02d T=%d %s caps=%d "
                "acc<-%s · out_hat %s golden · %.2fs"
                % (self.tag, who, rec["layer"], rec["head"], rec["T"],
                   rec["tier"], rec["n_captures"], rec["acc_source"],
                   "==" if g["equal"] else "!=", rec["seconds"]))
            if not g["equal"]:
                self.ledger.op("%s   !!! TILE VALUE DIFFERS FROM GOLDEN — "
                               "the verdict below will FAIL" % self.tag)
        else:
            self.host_attn += 1
            self._step_host += 1
        return r


class HostTicker:
    """Progress ticks for the pure-golden reference run: a pass-through
    wrapper on tf.decoder_layer_fx — bookkeeping only, no arithmetic touched
    (the same trick Offloader._layer uses), always restored."""

    def __init__(self, n_layers: int, n_steps: int, ledger: Ledger,
                 tag: str = "[OFF]", step_tokens=()):
        self.n_layers, self.n_steps = n_layers, n_steps
        self.ledger, self.tag = ledger, tag
        self.step_tokens = list(step_tokens)
        self.calls = 0
        self._orig = None

    def __enter__(self):
        self._orig = tf.decoder_layer_fx

        def wrapped(X, w, tier, *a, **kw):
            st, ly = divmod(self.calls, self.n_layers)
            self.ledger.tick("%s step %d/%d · layer %2d/%d · ALL ops HOST "
                             "(reference)" % (self.tag, st + 1, self.n_steps,
                                              ly + 1, self.n_layers))
            r = self._orig(X, w, tier, *a, **kw)
            self.calls += 1
            if ly == self.n_layers - 1:
                lbl = (" tok=%s" % self.step_tokens[st]
                       if st < len(self.step_tokens)
                       else " (generated token)")
                self.ledger.op("%s step %d/%d%s done · everything HOST "
                               "(pure-golden reference)"
                               % (self.tag, st + 1, self.n_steps, lbl))
            return r

        tf.decoder_layer_fx = wrapped
        return self

    def __exit__(self, *exc):
        tf.decoder_layer_fx = self._orig
        return False


# ═══════════ ledger hooks around the PROVEN six-op layer seam ════════════════

def _who(source: str, mode: str) -> str:
    """Display name for a record's provenance. In dry-run NOTHING ran on a
    tile, so every served record reads GOLDEN (dry-run): layer_offload labels
    its projection records `TILE (golden)`, and on this screen that must never
    be able to look like silicon. Display only — rec.source is what lands in
    the JSON."""
    if mode == "golden" and source.startswith("TILE"):
        return "GOLDEN (dry-run)"
    return source


class LedgerLayerOffloader(lof.LayerOffloader):
    """layer_offload.LayerOffloader + live ledger.

    layer_offload.py is IMPORT-ONLY and untouched: every rebind, every job
    builder, every substitution and every check below stays in the parent
    class. This subclass calls super() first and then PRINTS whatever the
    parent appended to `records` — it observes, it never decides.
    """

    def __init__(self, *a, ledger: Ledger, tag: str = "[ON ]",
                 n_steps: int = 0, step_tokens=(), **kw):
        super().__init__(*a, verbose=False, **kw)
        self.ledger = ledger
        self.tag = tag
        self.n_steps = n_steps
        self.step_tokens = list(step_tokens)
        self.cov = {op: [0, 0] for op in lof.OP_TYPES}   # [served, total]
        self.families = set()                            # families with values
        self._shown = 0                                  # records already
        self.host_layer_calls = 0                        # printed
        self.offloaded_steps: list = []
        self.dest = ("the GOLDEN stand-in (dry-run: no tile ran)"
                     if self.mode == "golden" else "the TILE (%s)" % self.mode)

    # ── printing helpers (no arithmetic anywhere in here) ──────────────────
    def _lbl(self, st: int) -> str:
        if st < len(self.step_tokens):
            return " tok=%s" % self.step_tokens[st]
        return " (generated token)"

    def _cov_str(self) -> str:
        return " ".join("%s %d/%d" % (op, self.cov[op][0], self.cov[op][1])
                        for op in lof.OP_TYPES if self.cov[op][1])

    def _tick(self, what: str) -> None:
        self.ledger.tick("%s step %d/%d · L%02d · %s · coverage so far: %s"
                         % (self.tag, self.step + 1, self.n_steps,
                            self.t_layer, what, self._cov_str() or "-"))

    def _emit_new(self) -> None:
        """One ledger line per op record the parent just appended."""
        while self._shown < len(self.records):
            rec = self.records[self._shown]
            self._shown += 1
            self.cov[rec.op][0] += rec.n_served
            self.cov[rec.op][1] += rec.n_total
            if rec.n_served:
                self.families.add(rec.op)
                exact = ("bit-exact vs golden" if rec.exact
                         else "reconstructed (C-1 view)")
                self.ledger.op(
                    "%s   >> %-16s %-6s %-22s %6d/%-6d values · %2d jobs · "
                    "%5d caps · %s · %.2fs"
                    % (self.tag, _who(rec.source, self.mode), rec.op,
                       rec.name, rec.n_served, rec.n_total, rec.jobs,
                       rec.caps, exact, rec.seconds))
                if rec.grade_ok is False:
                    self.ledger.op(
                        "%s   !!! TILE EGRESS DIFFERS FROM GOLDEN'S VIEW of "
                        "%s — the verdict below will FAIL" % (self.tag,
                                                              rec.name))
            else:
                self.ledger.op(
                    "%s   -- %-16s %-6s %-22s left on the HOST"
                    % (self.tag, rec.source, rec.op, rec.name))
            for n in rec.notes:
                self.ledger.op("%s      NOTE %s: %s" % (self.tag, rec.name, n))

    # ── the wrappers: super() first, then print ────────────────────────────
    def _layer(self, X, w, tier, *a, **kw):
        step, layer = self.step, self.layer
        target = (layer == self.t_layer
                  and (self.t_step is None or step == self.t_step))
        if target:
            self.ledger.op(
                "%s step %d/%d%s · layer %d (of %d) -> OFFLOADED LAYER: all "
                "six op families (%s) to %s"
                % (self.tag, step + 1, self.n_steps, self._lbl(step), layer,
                   self.n_layers, "/".join(lof.OP_TYPES), self.dest))
            self._tick("arming the six-op seam")
        else:
            self.host_layer_calls += 1
            self.ledger.tick("%s step %d/%d · layer %2d/%d · HOST (this layer "
                             "is not offloaded)"
                             % (self.tag, step + 1, self.n_steps, layer + 1,
                                self.n_layers))
        r = super()._layer(X, w, tier, *a, **kw)
        if target:
            self.offloaded_steps.append(step)
            self._emit_new()
            self.ledger.op(
                "%s   layer %d served by %s: %d/6 op families · %s · "
                "%d programs · %d caps · %.1fs"
                % (self.tag, layer, self.dest, len(self.families),
                   self._cov_str(), self.runner.n_jobs, self.runner.n_caps,
                   self.cur.seconds if self.cur else 0.0))
        if layer == self.n_layers - 1:
            if step in self.offloaded_steps:
                self.ledger.op(
                    "%s step %d/%d%s done · layer %d served by %s (%d/6 op "
                    "families) · the other %d layer(s) ran on the HOST"
                    % (self.tag, step + 1, self.n_steps, self._lbl(step),
                       self.t_layer, self.dest, len(self.families),
                       self.n_layers - 1))
            else:
                self.ledger.op(
                    "%s step %d/%d%s done · ALL %d layers HOST — the offload "
                    "is armed only at step %s"
                    % (self.tag, step + 1, self.n_steps, self._lbl(step),
                       self.n_layers,
                       "every step" if self.t_step is None
                       else "%d/%d" % (self.t_step + 1, self.n_steps)))
        return r

    def _gemm(self, A, B, *args, **kw):
        if self.armed and not self._busy("gemm"):
            self._tick("projections q/k/v/o/g/u/d -> tile GEMM jobs")
        r = super()._gemm(A, B, *args, **kw)
        if self.armed:
            self._emit_new()
        return r

    def _rope(self, x, m, theta=None):
        if self.armed and not self._busy("rope"):
            self._tick("RoPE (decode-token q rows)")
        r = super()._rope(x, m, theta)
        if self.armed:
            self._emit_new()
        return r

    def _attn(self, *a, **kw):
        if self.armed and not self._busy("attn"):
            self._tick("attention (score + PV) per head")
        r = super()._attn(*a, **kw)
        if self.armed:
            self._emit_new()
        return r

    def _norm(self, x, g, chunk: int = 128):
        if self.armed and not self._busy("norm"):
            self._tick("RMSNorm")
        r = super()._norm(x, g, chunk)
        if self.armed:
            self._emit_new()
        return r

    def _residual(self, which: str, arg):
        self._tick("residual %s (LAYER_RDATA fp16 slices)" % which)
        r = super()._residual(which, arg)
        self._emit_new()
        return r

    def _swiglu(self, arg):
        self._tick("SwiGLU (64-column jobs)")
        r = super()._swiglu(arg)
        self._emit_new()
        return r


# ═══════════════════════ verdict + honest footer ═════════════════════════════

def build_verdict(on, off, recs, tok, mode):
    """Mirror prompt_offload.report()'s gates, compactly. Returns (ok, lines).
    The DIFFERENT case is deliberately loud."""
    def txt(v):
        return tok.decode(v) if tok else str(v)

    ident = on["ids"] == off["ids"]
    fired = len(recs) > 0
    subs = all(r["substituted"] for r in recs)
    consumed = all(r.get("consumed") for r in recs)
    heads_ok = all(r.get("head_is_core") for r in recs)
    eq_out = [r["grade_out_hat"]["equal"] for r in recs]
    eq_acc = [r["grade_acc"]["equal"] for r in recs]
    cj_ok = all(r.get("compute_job_ok") is not False for r in recs)
    ex_ok = all(r.get("executor_ok") is not False for r in recs)
    ok = (ident and fired and subs and consumed and heads_ok and all(eq_out)
          and cj_ok and ex_ok)

    L = ["=" * W, "VERDICT", "-" * W]
    L.append("  emitted token ON  : ids=%s text=%r  (%.1fs)"
             % (on["ids"], txt(on["ids"]), on["seconds"]))
    L.append("  emitted token OFF : ids=%s text=%r  (%.1fs)"
             % (off["ids"], txt(off["ids"]), off["seconds"]))
    if ident:
        L.append("  TOKEN IDENTITY    : IDENTICAL — offload-on == pure-golden")
    else:
        L.append("  TOKEN IDENTITY    : *** DIFFERENT *** — the offloaded op "
                 "CHANGED the token")
        L.append("  !!! LOUD FAILURE — do not present this run; the tile or "
                 "the plumbing is wrong")
    if not fired:
        L.append("  FAIL: the target (layer, head) was never reached — no "
                 "offload happened, so")
        L.append("        token identity above proves NOTHING about the tile")
    else:
        L.append("  tile vs golden    : out_hat bit-exact %d/%d · acc_o "
                 "bit-exact %d/%d"
                 % (sum(eq_out), len(recs), sum(eq_acc), len(recs)))
        L.append("  substitution      : substituted %d/%d · consumed by the "
                 "model %d/%d · core kept %d/%d"
                 % (sum(1 for r in recs if r["substituted"]), len(recs),
                    sum(1 for r in recs if r.get("consumed")), len(recs),
                    sum(1 for r in recs if r.get("head_is_core")), len(recs)))
        if not all(eq_out):
            L.append("  FAIL: tile out_hat differs from golden out_hat")
        if not (subs and consumed and heads_ok):
            L.append("  FAIL: the tile-derived value was NOT the value the "
                     "model consumed")
        if not (cj_ok and ex_ok):
            L.append("  FAIL: executor_ok=%s compute_job_grade_ok=%s — see "
                     "the NOTE lines" % (ex_ok, cj_ok))
        for r in recs:
            for n in r.get("notes") or []:
                L.append("  NOTE %s: %s" % (r["name"], n))
    L.append("  PROMPT VERDICT    : %s"
             % ("PASS" if ok else "*** FAIL — see above ***"))
    L.append("=" * W)
    return ok, L


def build_footer(mode, n_layers, H, layer, head, n_steps, n_recs,
                 on_s, off_s, a1_mhz=None, off_reused=False):
    """The per-run honesty block. Printed EVERY run, no exceptions."""
    f = ["-" * W, "HONEST FOOTER — what actually ran where"]
    f.append("  host did   : tokenizer + detokenizer; embedding, LM head + "
             "argmax; all %d" % n_layers)
    f.append("               layers' projections, MLP, RMSNorms, residuals; "
             "the other %d" % (H - 1))
    f.append("               attention heads of layer %d and ALL attention "
             "of the other %d" % (layer, n_layers - 1))
    f.append("               layers; q/K/V staging between ops; the "
             "softmax/requant-scale")
    f.append("               epilogue (calib_requant / requant_i32_to_i8) "
             "applied to the")
    f.append("               tile's raw acc_o + s_c; and the whole "
             "OFFLOAD-OFF reference run.")
    if mode == "golden":
        f.append("  tile did   : NOTHING — DRY-RUN: the tile slot was filled "
                 "by golden itself.")
        f.append("               This rehearses plumbing only; no tile (sim "
                 "or hw) ran.")
    else:
        f.append("  tile did   : the attention-core raw INT32 accumulators "
                 "(acc_o) + s_c for")
        f.append("               ONE head (L%02d/h%02d) at each of the %d "
                 "decode step(s) — %d" % (layer, head, n_steps, n_recs))
        f.append("               op(s), produce-mode, requant_en=0, ZERO "
                 "baked output")
        f.append("               expectations in the job.")
    f += _exec_footer(mode, on_s, off_s, a1_mhz, off_reused)
    f.append("-" * W)
    return f


def _exec_footer(mode, on_s, off_s, a1_mhz=None, off_reused=False):
    """Executor + wall-time honesty lines — identical in both offload modes."""
    f = []
    if mode == "hw":
        clk = ("measured %.2f MHz at session start" % a1_mhz
               if a1_mhz else "NOT VERIFIED — do not present this run")
        f.append("  clock      : %g MHz correctness clock (recipe A%d%s, "
                 "image-keyed; %s;"
                 % (SESSION_CLOCK["mhz"], SESSION_CLOCK["recipe"],
                    " UNDERCLOCK ARM" if SESSION_CLOCK["underclocked"]
                    else "", clk))
        f.append("               re-verified numerically before every job). "
                 "Elapsed times are")
        f.append("               correctness-clock wall time — NOT a "
                 "performance number; no")
        f.append("               throughput claim is made or implied.")
    elif mode == "sim":
        f.append("  executor   : Verilated cl_apex simulation — no silicon, "
                 "no clock claim.")
        f.append("               Elapsed times are simulation wall time on "
                 "this host — NOT a")
        f.append("               performance number; no throughput claim is "
                 "made or implied.")
    else:
        f.append("  executor   : DRY-RUN (golden stand-in) — nothing above "
                 "demonstrates the tile.")
        f.append("               Elapsed times are host wall time of a "
                 "rehearsal — NOT a")
        f.append("               performance number; no throughput claim is "
                 "made or implied.")
    f.append("  wall time  : OFFLOAD-ON %.1fs · OFFLOAD-OFF (reference) "
             "%.1fs — labels above" % (on_s, off_s))
    if off_reused:
        f.append("               apply; the OFF reference was REUSED from an "
                 "earlier prompt of this")
        f.append("               session (identical ids/tier/group/max-"
                 "tokens, pure golden is")
        f.append("               deterministic) — its %.1fs was NOT spent "
                 "again in this run." % off_s)
    else:
        f.append("               apply; the OFF run is host compute, part of "
                 "the proof, not overhead.")
    return f


# ══════════════════ layer mode: verdict + honest footer ══════════════════════

def _check_families(checks):
    """Collapse per-head check names ('rope h07: ...') into families. Only the
    HEAD INDEX is collapsed — every other word of the check's own name is
    printed as layer_offload wrote it."""
    fam: dict = {}
    for name, ok, _d in checks:
        key = re.sub(r"\bh\d+\b", "h**", name)
        c = fam.setdefault(key, [0, 0])
        c[0] += 1
        c[1] += 1 if ok else 0
    return fam


def build_layer_verdict(on, off, bus, recs, checks, agg, tok, mode, layer,
                        step, n_steps, runner):
    """Mirror layer_offload.report()'s gates for the REPL. -> (ok, lines)."""
    def txt(v):
        return tok.decode(v) if tok else str(v)

    ident = on["ids"] == off["ids"]
    fired = any(r.n_served for r in recs)
    served = [op for op in lof.OP_TYPES if agg[op]["values"] > 0]
    checks_ok = all(ok for _n, ok, _d in checks)
    grades_ok = all(agg[op]["grade_ok"] is not False for op in lof.OP_TYPES)
    caps_ok = (mode == "golden") or runner.n_caps > 0
    # this mode's claim is ALL SIX op families of the layer: if one was left
    # on the host (refused composite, empty scope, …) the run does not get to
    # PASS under that headline — it says which family, and why.
    families_ok = len(served) == len(lof.OP_TYPES)
    ok = bool(ident and fired and checks_ok and grades_ok and caps_ok
              and families_ok and checks)

    L = ["=" * W,
         "VERDICT — FULL-LAYER OFFLOAD (every op family of one decoder layer)",
         "-" * W]
    L.append("  offloaded layer %d at decode step %d/%d · composition: C-LBUS "
             "BUS_ON" % (layer, step + 1, n_steps))
    L.append("  " + "-" * (W - 2))
    L.append("  %-26s %-16s %-21s %s"
             % ("op family", "who", "served/total values", "exact?"))
    for op in lof.OP_TYPES:
        a = agg[op]
        who = _who(a["source"], mode) if a["values"] else "HOST (golden)"
        cov = "%d/%d" % (a["values"], a["values_total"])
        ex = ("—" if not a["values"]
              else ("BIT-EXACT" if a["exact"] else "reconstructed"))
        L.append("  %-26s %-16s %-21s %s" % (lof.OP_LABEL[op], who, cov, ex))
        L.append("      egress      : %s" % lof.EGRESS[op])
        if a["values"]:
            L.append("      instances   : %d/%d op calls served, %d tile "
                     "programs, %d caps, %ss"
                     % (a["served_instances"], a["instances"], a["jobs"],
                        a["caps"], a["seconds"]))
            L.append("      tile vs golden (same view): %s"
                     % ("bit-exact" if a["grade_ok"] else "*** DIFFERS ***"))
            for d in a["detail"][:1]:
                for k in ("max_abs_delta", "max_abs_delta_q78",
                          "downstream_c1_codes_n_diff",
                          "downstream_c1_scale_equal", "requant_identical",
                          "downstream_codes_match_golden",
                          "reassembled_equal_golden", "per_slice_pass",
                          "chunks", "rows", "slices"):
                    if k in d:
                        L.append("        %-32s: %s" % (k, d[k]))
        for n in a["notes"]:
            L.append("      NOTE: %s" % n)
    L.append("  * RoPE's C-1 reconstruction is re-quantized by golden's very "
             "next stage")
    L.append("    (attention_core Q7, quant_rows_i8) — measured identical to "
             "golden's own")
    L.append("    codes, so that row is exact WHERE THE MODEL CONSUMES IT.")
    L.append("  " + "-" * (W - 2))
    n_ok = sum(1 for _n, o, _d in checks if o)
    L.append("  substitution / consumption checks: %d/%d PASS"
             % (n_ok, len(checks)))
    for key, (n, good) in sorted(_check_families(checks).items()):
        L.append("      %s  %3dx  %s" % ("PASS" if good == n else "FAIL",
                                         n, key))
    for name, o, detail in checks:
        if not o:
            L.append("      FAILED CHECK: %s  %s" % (name, detail))
    if not checks:
        L.append("      (none — nothing was substituted, so nothing is "
                 "verified)")
    L.append("  " + "-" * (W - 2))
    L.append("  emitted token ON  : ids=%s text=%r  (%.1fs)"
             % (on["ids"], txt(on["ids"]), on["seconds"]))
    L.append("  emitted token OFF : ids=%s text=%r  (%.1fs%s)"
             % (off["ids"], txt(off["ids"]), off["seconds"],
                ", REUSED from an earlier prompt this session"
                if off.get("reused") else ""))
    if bus is not None:
        L.append("  token HOST+BUS_ON : ids=%s text=%r  (%.1fs, layer "
                 "composed BUS_ON %dx) -> %s"
                 % (bus["ids"], txt(bus["ids"]), bus["seconds"], bus["fired"],
                    "same as pure host" if bus["ids"] == off["ids"]
                    else "DIFFERS from pure host"))
    if ident:
        L.append("  TOKEN IDENTITY    : IDENTICAL — offload-on == pure-golden")
    else:
        L.append("  TOKEN IDENTITY    : *** DIFFERENT *** — the offloaded "
                 "layer CHANGED the token")
        L.append("  !!! LOUD FAILURE — do not present this run; the tile or "
                 "the plumbing is wrong")
        if bus is not None and bus["ids"] == off["ids"]:
            L.append("  NOTE: the host-only BUS_ON run did NOT change the "
                     "token, so the difference")
            L.append("        came from a TILE-SERVED VALUE, not from the "
                     "composition mode.")
        elif bus is not None:
            L.append("  NOTE: the host-only BUS_ON run ALSO changes the "
                     "token, so the composition")
            L.append("        mode (not the tile) is implicated.")
        else:
            L.append("  NOTE: no host-only BUS_ON run was made, so whether "
                     "the TILE or the BUS_ON")
            L.append("        composition changed the token is NOT "
                     "established by this run.")
    if not fired:
        L.append("  FAIL: no op of layer %d was served by the tile — token "
                 "identity above" % layer)
        L.append("        proves NOTHING about the tile")
    if not caps_ok:
        L.append("  FAIL: the executor returned ZERO capture records — there "
                 "is no evidence")
        L.append("        any tile ran; a run with no captures cannot pass")
    if not checks_ok:
        L.append("  FAIL: a tile-derived value was NOT the value the model "
                 "consumed (see above)")
    if not grades_ok:
        L.append("  FAIL: a tile egress differs from golden's own view of the "
                 "same op")
    if fired and not families_ok:
        L.append("  FAIL: only %d/6 op families were served — this mode's "
                 "claim is ALL SIX." % len(served))
        L.append("        Left on the HOST: %s (see the NOTE lines above)"
                 % ", ".join(op for op in lof.OP_TYPES if op not in served))
    L.append("  OP FAMILIES SERVED BY THE TILE: %d/6 (%s)"
             % (len(served), ", ".join(served) or "none"))
    L.append("  tile programs     : %d programs, %d capture records, %.1fs in "
             "the executor" % (runner.n_jobs, runner.n_caps, runner.wall))
    L.append("  LAYER VERDICT     : %s"
             % ("PASS" if ok else "*** FAIL — see above ***"))
    L.append("=" * W)
    return ok, L


def build_layer_footer(model, mode, layer, step, n_steps, agg, checks, runner,
                       on_s, off_s, bus, bus_check, proj_cols, off_ids,
                       a1_mhz=None, off_reused=False):
    """The per-run honesty block for layer mode. The host-side items are
    layer_offload.host_ledger()'s own list — printed verbatim, so the REPL
    cannot quietly claim more than the proven tool does."""
    f = ["-" * W, "HONEST FOOTER — what actually ran where"]
    if mode == "golden":
        f.append("  tile did   : NOTHING — DRY-RUN: every tile call was "
                 "filled by golden itself.")
        f.append("               This rehearses the six-op seam only; no tile "
                 "(sim or hw) ran.")
    else:
        f.append("  tile did   : ALL SIX op families of decoder layer %d at "
                 "decode step %d/%d —" % (layer, step + 1, n_steps))
        for op in lof.OP_TYPES:
            a = agg[op]
            f.append("               %-8s %7d/%-7d values · %d programs · %d "
                     "caps" % (op, a["values"], a["values_total"], a["jobs"],
                               a["caps"]))
        f.append("               %d tile programs, %d capture records, %.1fs "
                 "in the executor;" % (runner.n_jobs, runner.n_caps,
                                       runner.wall))
        f.append("               produce-mode jobs, ZERO baked output "
                 "expectations.")
        f.append("  re-entry   : RMSNorm-2, SwiGLU and RoPE come back through "
                 "the tile's 128-wide")
        f.append("               C-1 view (B-FEED-WIDTH), so those "
                 "substituted values are")
        f.append("               RECONSTRUCTIONS, not raw bits — the measured "
                 "deltas are in the")
        f.append("               ledger above. RoPE's is exact where the "
                 "model consumes it")
        f.append("               (golden re-quantizes it to identical "
                 "codes+scale — checked).")
    f.append("  composition: the offloaded layer runs in C-LBUS BUS_ON "
             "(apex_layer_deq.sv:90-92")
    f.append("               refuses an ungraded fp32 composite) — a REAL "
             "change to that layer's")
    if bus is not None:
        f.append("               host arithmetic. Host-only BUS_ON "
                 "diagnostic: ids=%s -> %s"
                 % (bus["ids"], "SAME token as pure host (the bus mode alone "
                    "changes nothing)" if bus["ids"] == off_ids
                    else "DIFFERENT token than pure host"))
    else:
        f.append("               host arithmetic. No host-only BUS_ON "
                 "diagnostic ran this prompt")
        f.append("               (--bus-check %s; 'auto' runs it only when "
                 "the offloaded token" % bus_check)
        f.append("               DIFFERS from pure host, which is when the "
                 "attribution matters).")
    f.append("  host did   : (this list is layer_offload.host_ledger's own, "
             "printed verbatim)")
    ns = argparse.Namespace(proj_cols=proj_cols, proj_rows=1)
    for line in lof.host_ledger(model, ns, agg):
        f.append("               - " + line)
    f.append("               - the whole OFFLOAD-OFF (pure-golden) reference "
             "run")
    f.append("  checks     : %d/%d substitution + consumption checks passed "
             "(printed above)"
             % (sum(1 for _n, o, _d in checks if o), len(checks)))
    f += _exec_footer(mode, on_s, off_s, a1_mhz, off_reused)
    f.append("-" * W)
    return f


# ═══════════════════════════ one prompt, end to end ══════════════════════════

ERRS = (pof.ComputeJobMissing, bridge.CaptureEgressError, RuntimeError,
        FileNotFoundError, ValueError, SystemExit)

# Golden's own functions, captured at import (before any rebind can exist).
_GOLDEN_SNAPSHOT = tuple(
    ("transformer.%s" % n, tf, n, getattr(tf, n))
    for n in ("decoder_layer_fx", "attention_core", "gemm_i8_ksplit",
              "rope_fx", "rmsnorm_fx_wide", "_proj_epilogue", "_f16")
) + (("attention.attention_core", at, "attention_core", at.attention_core),)


def run_one(model, tok, ids, prompt_text, args, mode, ledger, work,
            seq: int, a1_mhz=None, off_cache=None) -> dict:
    """Shared fence, then dispatch to the attention-op or full-layer runner."""
    kind = getattr(args, "offload", "attention")
    n_total = len(ids) + args.max_tokens
    res: dict = {"seq": seq, "prompt": prompt_text, "ids": ids, "mode": mode,
                 "offload": kind, "ok": False, "refused": False, "error": None,
                 "record": None}
    if n_total > pof.TOKENS_MAX:
        say("REFUSE: prompt(%d) + max_tokens(%d) = %d > %d tokens."
            % (len(ids), args.max_tokens, n_total, pof.TOKENS_MAX))
        say("        Beyond %d a head becomes a ChunkedHead whose host merge "
            "needs per-chunk" % pof.TOKENS_MAX)
        say("        sm_m/sm_l, which the mailbox does not expose — a chunked "
            "head CANNOT be")
        say("        offloaded (audit N8). Shorten the prompt or :tokens.")
        res["refused"] = True
        return res
    if kind == "layer":
        return run_one_layer(model, tok, ids, prompt_text, args, mode, ledger,
                             work, seq, res, a1_mhz, off_cache)
    return run_one_attention(model, tok, ids, prompt_text, args, mode, ledger,
                             work, seq, res, a1_mhz, off_cache)


def _off_reference(model, ids, args, tier, eos, ledger, n_steps, step_tokens,
                   off_cache, n_runs: str = "2/2"):
    """The pure-golden reference decode — recomputed, or REUSED from this
    session when every operand that determines it is identical (pure golden
    is deterministic and no rebind survives a run: both are asserted)."""
    key = (tuple(ids), args.tier, args.group, args.max_tokens)
    if off_cache is not None and key in off_cache:
        off = dict(off_cache[key])
        off["reused"] = True
        ledger.op("[OFF] run %s OFFLOAD OFF — reference REUSED from this "
                  "session (identical ids/" % n_runs)
        ledger.op("[OFF]   tier/group/max-tokens; pure golden is "
                  "deterministic) · ids=%s · originally %.1fs"
                  % (off["ids"], off["seconds"]))
        return off
    ledger.op("[OFF] run %s OFFLOAD OFF — pure-golden reference, no rebind, "
              "everything HOST" % n_runs)
    with HostTicker(model.n_layers, n_steps, ledger, "[OFF]", step_tokens):
        off = pof.decode(model, ids, args.max_tokens, tier, args.group,
                         eos, None, verbose=False)
    off["reused"] = False
    ledger.op("[OFF] LM head + argmax -> HOST · token ids=%s" % off["ids"])
    if off_cache is not None:
        off_cache[key] = off
    return off


def _assert_golden_restored() -> None:
    """Every global that prompt_offload / layer_offload rebind must be the
    SAME OBJECT golden had at import time before a reference run starts — a
    stale rebind would make the 'pure host' run anything but. Compared by
    identity against the snapshot taken when this module was imported, not by
    module name (golden re-exports several of these across modules)."""
    for label, mod, name, orig in _GOLDEN_SNAPSHOT:
        assert getattr(mod, name) is orig, \
            "%s rebind NOT restored — the reference run would not be golden" \
            % label
    assert tf.attention_core is at.attention_core, \
        "transformer/attention resolve different attention_core"


def run_one_attention(model, tok, ids, prompt_text, args, mode, ledger, work,
                      seq, res, a1_mhz=None, off_cache=None) -> dict:
    """OFFLOAD-ON decode, OFFLOAD-OFF reference decode, verdict, footer,
    JSON record. Returns a result dict; never raises for run errors (they
    are printed LOUDLY and recorded)."""
    n_total = len(ids) + args.max_tokens
    L_n, H = model.n_layers, model.meta["H"]
    if not (0 <= args.offload_layer < L_n and 0 <= args.offload_head < H):
        say("REFUSE: target L%d/h%d outside layer [0,%d) / head [0,%d)"
            % (args.offload_layer, args.offload_head, L_n, H))
        res["refused"] = True
        return res

    tier = rt.TIER_MAP[args.tier]
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    n_steps = len(ids) + args.max_tokens - 1
    step_tokens = [repr(tok.decode([t])) if tok else str(t) for t in ids]

    t_all = time.perf_counter()
    ledger.start()
    ledger.op("prompt %r -> ids %s (%d tok) · fence %d+%d=%d <= %d OK"
              % (prompt_text, ids, len(ids), len(ids), args.max_tokens,
                 n_total, pof.TOKENS_MAX))
    ledger.op("[ON ] run 1/2 OFFLOAD ON — attention L%02d/h%02d -> %s at "
              "every step"
              % (args.offload_layer, args.offload_head,
                 "TILE-SLOT (dry-run)" if mode == "golden"
                 else "TILE (%s)" % mode))
    on = off = None
    recs: list = []
    err = None
    try:
        with LedgerOffloader(
                n_layers=L_n, H=H, layer=args.offload_layer,
                head=args.offload_head, mode=mode, work=work,
                tile_div=args.tile_div, slot=args.slot,
                timeout_s=args.timeout_s, compute_job=args.compute_job,
                ledger=ledger, tag="[ON ]", n_steps=n_steps,
                step_tokens=step_tokens) as offl:
            on = pof.decode(model, ids, args.max_tokens, tier, args.group,
                            eos, offl, verbose=False)
        recs = on["records"]
        _assert_golden_restored()
        ledger.op("[ON ] LM head + argmax -> HOST · token ids=%s" % on["ids"])
        off = _off_reference(model, ids, args, tier, eos, ledger, n_steps,
                             step_tokens, off_cache)
    except ERRS as e:
        err = e
    finally:
        ledger.stop()

    if err is not None:
        say("=" * W)
        say("*** RUN FAILED — %s ***" % type(err).__name__)
        say("  the executor/tile did NOT deliver this prompt's offload:")
        for ln in str(err).splitlines() or ["(no message)"]:
            say("  " + ln)
        say("  No verdict is possible. This is a FAILURE, not a soft skip.")
        say("=" * W)
        res["error"] = "%s: %s" % (type(err).__name__, err)
        res["record"] = str(_write_record(work, seq, res, recs))
        return res

    ok, vlines = build_verdict(on, off, recs, tok, mode)
    for ln in vlines:
        say(ln)
    for ln in build_footer(mode, L_n, H, args.offload_layer,
                           args.offload_head, n_steps, len(recs),
                           on["seconds"], off["seconds"], a1_mhz,
                           bool(off.get("reused"))):
        say(ln)

    res.update({
        "ok": ok, "ids_on": on["ids"], "ids_off": off["ids"],
        "text_on": tok.decode(on["ids"]) if tok else str(on["ids"]),
        "text_off": tok.decode(off["ids"]) if tok else str(off["ids"]),
        "token_identity": on["ids"] == off["ids"],
        "n_records": len(recs), "off_reused": bool(off.get("reused")),
        "seconds": {"on": on["seconds"], "off": off["seconds"],
                    "total": time.perf_counter() - t_all},
        "target": {"layer": args.offload_layer, "head": args.offload_head},
        "clkgen_a1_mhz": a1_mhz,
    })
    res["record"] = str(_write_record(work, seq, res, recs))
    say("  record -> %s" % res["record"])
    return res


def run_one_layer(model, tok, ids, prompt_text, args, mode, ledger, work,
                  seq, res, a1_mhz=None, off_cache=None) -> dict:
    """FULL-LAYER offload: every op family of ONE decoder layer at ONE decode
    step is served by the tile (layer_offload.LayerOffloader, imported and
    unmodified), then the emitted token is checked against a pure-golden
    run."""
    n_total = len(ids) + args.max_tokens
    L_n = model.n_layers
    n_steps = len(ids) + args.max_tokens - 1
    if not 0 <= args.offload_layer < L_n:
        say("REFUSE: --offload-layer %d outside [0,%d)"
            % (args.offload_layer, L_n))
        res["refused"] = True
        return res
    step = (n_steps - 1 if args.offload_step is None else args.offload_step)
    if not 0 <= step < n_steps:
        say("REFUSE: --offload-step %s outside [0,%d) for this prompt "
            "(%d prompt tokens + %d new = %d decode steps)."
            % (args.offload_step, n_steps, len(ids), args.max_tokens, n_steps))
        say("        Nothing was offloaded, so nothing would be proven.")
        res["refused"] = True
        return res
    if args.proj_cols < lof.MXE_N or args.proj_cols % lof.MXE_N:
        say("REFUSE: --proj-cols %d — layer mode's claim is that ALL SIX op "
            "families are" % args.proj_cols)
        say("        served by the tile, so the projection family needs at "
            "least one whole")
        say("        MXE block: a multiple of %d, >= %d. (Use --offload "
            "attention for a" % (lof.MXE_N, lof.MXE_N))
        say("        narrower claim.)")
        res["refused"] = True
        return res

    tier = rt.TIER_MAP[args.tier]
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    step_tokens = [repr(tok.decode([t])) if tok else str(t) for t in ids]
    scope = lof.Scope(ops=lof.OP_TYPES, proj_cols=args.proj_cols)
    runner = lof.Runner(mode, work, tile_div=args.tile_div, slot=args.slot,
                        timeout_s=args.timeout_s)

    t_all = time.perf_counter()
    ledger.start()
    ledger.op("prompt %r -> ids %s (%d tok) · fence %d+%d=%d <= %d OK"
              % (prompt_text, ids, len(ids), len(ids), args.max_tokens,
                 n_total, pof.TOKENS_MAX))
    ledger.op("[ON ] run 1/2 OFFLOAD ON — ALL SIX op families (%s) of layer "
              "%d at decode" % ("/".join(lof.OP_TYPES), args.offload_layer))
    ledger.op("[ON ]   step %d/%d -> %s · projections sampled at --proj-cols "
              "%d, the other five at FULL width"
              % (step + 1, n_steps,
                 "GOLDEN stand-in (dry-run: no tile ran)" if mode == "golden"
                 else "TILE (%s)" % mode, args.proj_cols))
    ledger.op("[ON ] the offloaded layer composes in C-LBUS BUS_ON "
              "(apex_layer_deq.sv:90-92 refuses")
    ledger.op("[ON ]   an ungraded composite) — a REAL change to that layer's "
              "host arithmetic")
    on = off = bus = None
    recs: list = []
    checks: list = []
    err = None
    try:
        with LedgerLayerOffloader(
                n_layers=L_n, layer=args.offload_layer, mode=mode, work=work,
                scope=scope, runner=runner, step=step, ledger=ledger,
                tag="[ON ]", n_steps=n_steps,
                step_tokens=step_tokens) as offl:
            on = pof.decode(model, ids, args.max_tokens, tier, args.group,
                            eos, None, verbose=False)
            recs, checks = list(offl.records), list(offl.checks)
        _assert_golden_restored()
        ledger.op("[ON ] LM head + argmax -> HOST · token ids=%s" % on["ids"])
        off = _off_reference(model, ids, args, tier, eos, ledger, n_steps,
                             step_tokens, off_cache)
        need_bus = (args.bus_check == "always"
                    or (args.bus_check == "auto" and on["ids"] != off["ids"]))
        if need_bus:
            ledger.op("[BUS] extra run — HOST ONLY with layer %d composed in "
                      "C-LBUS BUS_ON: separates" % args.offload_layer)
            ledger.op("[BUS]   'the tile changed the token' from 'the bus "
                      "mode changed the token'")
            with lof.BusOnly(L_n, args.offload_layer, step) as bo:
                bus = pof.decode(model, ids, args.max_tokens, tier, args.group,
                                 eos, None, verbose=False)
            bus["fired"] = bo.fired
            _assert_golden_restored()
            ledger.op("[BUS] token ids=%s (layer composed BUS_ON %dx)"
                      % (bus["ids"], bus["fired"]))
    except ERRS as e:
        err = e
    finally:
        ledger.stop()

    ser = [{"op": r.op, "name": r.name, "source": r.source,
            "n_served": r.n_served, "n_total": r.n_total, "jobs": r.jobs,
            "caps": r.caps, "seconds": r.seconds, "exact": r.exact,
            "grade_ok": r.grade_ok, "notes": r.notes} for r in recs]
    if err is not None:
        say("=" * W)
        say("*** RUN FAILED — %s ***" % type(err).__name__)
        say("  the executor/tile did NOT deliver this prompt's layer offload:")
        for ln in str(err).splitlines() or ["(no message)"]:
            say("  " + ln)
        say("  No verdict is possible. This is a FAILURE, not a soft skip.")
        say("=" * W)
        res["error"] = "%s: %s" % (type(err).__name__, err)
        res["record"] = str(_write_record(work, seq, res, ser))
        return res

    agg = {op: lof._agg(recs, op) for op in lof.OP_TYPES}
    ok, vlines = build_layer_verdict(on, off, bus, recs, checks, agg, tok,
                                     mode, args.offload_layer, step, n_steps,
                                     runner)
    for ln in vlines:
        say(ln)
    for ln in build_layer_footer(model, mode, args.offload_layer, step,
                                 n_steps, agg, checks, runner, on["seconds"],
                                 off["seconds"], bus, args.bus_check,
                                 args.proj_cols, off["ids"], a1_mhz,
                                 bool(off.get("reused"))):
        say(ln)

    res.update({
        "ok": ok, "ids_on": on["ids"], "ids_off": off["ids"],
        "text_on": tok.decode(on["ids"]) if tok else str(on["ids"]),
        "text_off": tok.decode(off["ids"]) if tok else str(off["ids"]),
        "token_identity": on["ids"] == off["ids"],
        "ids_bus_on": (bus["ids"] if bus else None),
        "bus_on_token_identity": (bus["ids"] == off["ids"] if bus else None),
        "n_records": len(recs), "off_reused": bool(off.get("reused")),
        "op_families_served": [op for op in lof.OP_TYPES
                               if agg[op]["values"] > 0],
        "ledger": agg,
        "checks": [{"name": n, "ok": o, "detail": str(d)}
                   for n, o, d in checks],
        "tile_jobs": runner.n_jobs, "tile_caps": runner.n_caps,
        "executor_seconds": round(runner.wall, 1),
        "seconds": {"on": on["seconds"], "off": off["seconds"],
                    "bus_on": (bus["seconds"] if bus else None),
                    "total": time.perf_counter() - t_all},
        "target": {"layer": args.offload_layer, "step": step,
                   "proj_cols": args.proj_cols},
        "clkgen_a1_mhz": a1_mhz,
    })
    res["record"] = str(_write_record(work, seq, res, ser))
    say("  record -> %s" % res["record"])
    return res


def _write_record(work: Path, seq: int, res: dict, recs) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    data = dict(res)
    data["captures"] = [{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in recs]
    data["git"] = rt.git_rev()
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p = work / ("repl_%03d_%s.json" % (seq, time.strftime("%H%M%S")))
    p.write_text(json.dumps(data, indent=1, default=str))
    return p


# ═══════════════════════════ hw warm-session bring-up ════════════════════════

def hw_bringup_cmds(slot: int, agfi: str, sudo: bool = True,
                    run_recipe=None) -> dict:
    """The exact remote command lines, KEYED to the image (clock_key.py).
    Split out so --selftest can check them without an instance.

    Refuses (SystemExit) a malformed AGFI, an AGFI not registered in
    clock_key.IMAGE_RECIPE (its constrained clock is unknown — flying it
    ungated is the silent-garbage class), and any overclocking run_recipe.
    """
    if not re.fullmatch(r"agfi-[0-9a-f]+", agfi):
        raise SystemExit("REFUSE: %r does not look like an AGFI id" % agfi)
    try:
        exp = clock_key.expected_clock(agfi, run_recipe=run_recipe)
    except clock_key.ClockKeyRefusal as e:
        raise SystemExit("REFUSE (clock key): %s" % e)
    pre = "sudo -n " if sudo else ""
    return {
        "image": "%sfpga-load-local-image -S %d -I %s" % (pre, slot, agfi),
        "recipe": clock_key.recipe_cmd(slot, exp["recipe_cmd_a"], sudo=sudo),
        "recipe_idx": exp["recipe_cmd_a"],
        "expect_mhz": exp["mhz"],
        "expect_tol": exp["tol"],
        "underclocked": exp["underclocked"],
        "clock_why": exp["why"],
    }


def _ssh_run(cfg, command: str, timeout_s: int, label: str):
    logp: list = []
    argv = rhe._ssh_argv("ssh", cfg["key"], cfg["ssh_opts"],
                         rhe._userhost(cfg["host"], cfg["user"]), command)
    rc, out, timed_out = rhe._run_local(argv, timeout_s, label, logp)
    return rc, out, timed_out


def bringup_hw(args) -> dict:
    """Image + clock recipe + NUMERIC clock verification, once, at startup.
    Refuses (SystemExit) rather than entering the REPL on a cold/wrong card."""
    cfg = rhe.remote_config(args)
    if cfg is None:
        raise SystemExit(
            "REFUSE: --executor hw needs --host (or $APEX_F2_HOST) and "
            "usually --key (or $APEX_F2_KEY). Nothing was contacted.")
    uh = rhe._userhost(cfg["host"], cfg["user"])
    say("[bringup] card session: %s slot %d" % (uh, args.slot))
    cmds = hw_bringup_cmds(args.slot, args.agfi, sudo=cfg["sudo"],
                           run_recipe=getattr(args, "run_recipe", None))
    say("[bringup] clock key: %s" % cmds["clock_why"])
    if cmds["underclocked"]:
        say("[bringup]   DELIBERATE UNDERCLOCK ARM (--run-recipe %d): the "
            "image is signed off faster;" % cmds["recipe_idx"])
        say("[bringup]   this arm is setup-safe and will be said so in "
            "every footer.")

    if args.skip_image_load:
        say("[bringup] image load SKIPPED (--skip-image-load) — using "
            "whatever is flying;")
        say("[bringup]   the numeric clock gate below still decides.")
    else:
        say("[bringup] loading image %s ..." % args.agfi)
        say("[bringup]   (an AFI load RESETS the clkgen MMCMs to the 125 MHz "
            "default — the")
        say("[bringup]   recipe step after this is MANDATORY, not a ritual)")
        rc, out, to = _ssh_run(cfg, cmds["image"], 600, "image load")
        if rc != 0 or to:
            raise SystemExit(
                "BRING-UP FAILED at image load (rc=%s%s):\n%s\n"
                "hint: if 'command not found', run "
                "'source ~/aws-fpga/sdk_setup.sh' once on the instance; "
                "if sudo asked for a password, fix sudoers (-n never prompts)."
                % (rc, " TIMEOUT" if to else "", out[-1500:]))
        say("[bringup] image load rc=0")

    label = "clkgen recipe A%d" % cmds["recipe_idx"]
    rc, out, to = _ssh_run(cfg, cmds["recipe"], 300, label)
    if rc != 0 or to:
        raise SystemExit("BRING-UP FAILED at %s (rc=%s%s):\n%s"
                         % (label, rc, " TIMEOUT" if to else "",
                            out[-1500:]))
    say("[bringup] %s programmed (rc=0)" % label)

    clk = rhe.check_tile_clock(cfg["host"], cfg["key"], user=cfg["user"],
                               slot=args.slot, ssh_opts=cfg["ssh_opts"],
                               sudo=cfg["sudo"], agfi=args.agfi,
                               run_recipe=getattr(args, "run_recipe", None))
    if not clk["ok"]:
        raise SystemExit(
            "BRING-UP FAILED — the tile clock did not verify numerically:\n"
            "  %s\nNOT entering the REPL: over its closed clock the tile "
            "computes garbage\nwhile every rc stays 0." % clk["why"])
    say("[bringup] tile clock VERIFIED numerically: %s" % clk["why"])
    SESSION_CLOCK.update(mhz=cmds["expect_mhz"], recipe=cmds["recipe_idx"],
                         underclocked=cmds["underclocked"])
    return {"a1_mhz": clk["a1_mhz"], "host": uh,
            "expect_mhz": cmds["expect_mhz"],
            "recipe_idx": cmds["recipe_idx"]}


# ═══════════════════════════════ the REPL ════════════════════════════════════

REPL_HELP = """commands:
  <text>            run this prompt (OFFLOAD-ON vs pure-golden + verdict)
  :ids 1 2 3        run raw token ids (tokenizer-free; tiny models)
  :mode attention   offload ONE attention op (layer, head) per decode step
  :mode layer       offload ALL SIX op families of one layer at one step
  :target L H       attention mode: change the offloaded (layer, head)
  :target L         layer mode: change the offloaded layer
  :step N | last    layer mode: which decode step is offloaded
  :tokens N         change max new tokens (fence: prompt+N <= 128)
  :help             this text
  :quit  (Ctrl-D)   leave — clean; nothing is left running remotely"""


def _exit_msg(mode: str) -> str:
    m = "session closed cleanly."
    if mode == "hw":
        m += (" No persistent remote processes are held; if a job was "
              "mid-flight, the next job's per-file TILE_RST makes a retry "
              "safe (f2_host_run.py).")
    return m


def repl(model, tok, args, mode, work, hw_info, off_cache=None) -> int:
    ledger = Ledger()
    seq = 0
    worst = 0
    echo = not sys.stdin.isatty()
    say("")
    say("type a prompt and press enter (:help for commands, Ctrl-D to quit)")
    while True:
        try:
            line = input("apex> ").strip()
        except EOFError:
            say("\n[exit] Ctrl-D — " + _exit_msg(mode))
            return worst
        except KeyboardInterrupt:
            say("\n[exit] Ctrl-C — " + _exit_msg(mode))
            return worst
        if echo and line:
            say(line)                     # piped stdin: show what was "typed"
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            say("[exit] " + _exit_msg(mode))
            return worst
        if line == ":help":
            say(REPL_HELP)
            continue
        if line.startswith(":mode"):
            try:
                m = line.split()[1]
                assert m in ("attention", "layer")
                args.offload = m
                say("offload mode -> %s" % m)
                if m == "layer":
                    say("  layer %d · ALL SIX op families (%s) at decode step "
                        "%s" % (args.offload_layer, "/".join(lof.OP_TYPES),
                                "last" if args.offload_step is None
                                else args.offload_step))
                else:
                    say("  attention L%02d/h%02d at every decode step"
                        % (args.offload_layer, args.offload_head))
            except (IndexError, AssertionError):
                say("usage: :mode attention | :mode layer")
            continue
        if line.startswith(":step"):
            try:
                a = line.split()[1]
                args.offload_step = None if a == "last" else int(a)
                assert args.offload_step is None or args.offload_step >= 0
                say("layer-mode offload step -> %s"
                    % ("last decode step" if args.offload_step is None
                       else args.offload_step))
            except (IndexError, ValueError, AssertionError):
                say("usage: :step N  (0-based)  |  :step last")
            continue
        if line.startswith(":target"):
            parts = line.split()
            try:
                if len(parts) == 2:
                    lt = int(parts[1])
                    assert 0 <= lt < model.n_layers
                    args.offload_layer = lt
                    say("offload target -> layer %d (head unchanged: %d)"
                        % (lt, args.offload_head))
                else:
                    _, l_s, h_s = parts
                    lt, ht = int(l_s), int(h_s)
                    assert 0 <= lt < model.n_layers \
                        and 0 <= ht < model.meta["H"]
                    args.offload_layer, args.offload_head = lt, ht
                    say("offload target -> layer %d head %d" % (lt, ht))
            except (ValueError, AssertionError):
                say("usage: :target L H  with 0<=L<%d, 0<=H<%d  (or :target L)"
                    % (model.n_layers, model.meta["H"]))
            continue
        if line.startswith(":tokens"):
            try:
                args.max_tokens = max(1, int(line.split()[1]))
                say("max new tokens -> %d" % args.max_tokens)
            except (ValueError, IndexError):
                say("usage: :tokens N")
            continue
        if line.startswith(":ids"):
            try:
                ids = [int(x) for x in line.split()[1:]]
                assert ids
            except (ValueError, AssertionError):
                say("usage: :ids 3 1 4 1 5")
                continue
            text = "<ids %s>" % ids
        elif line.startswith(":"):
            say("unknown command %r — :help" % line)
            continue
        else:
            if tok is None:
                say("no tokenizer in this session — use ':ids ...' "
                    "(raw token ids)")
                continue
            ids = tok.encode(line)
            text = line
        seq += 1
        try:
            r = run_one(model, tok, list(ids), text, args, mode, ledger,
                        work, seq, a1_mhz=hw_info.get("a1_mhz"),
                        off_cache=off_cache)
        except KeyboardInterrupt:
            ledger.stop()
            say("\n[abort] Ctrl-C during the run — golden rebinds are "
                "restored by the context")
            say("[abort] managers; " + _exit_msg(mode))
            return 130
        if not r.get("ok") and not r.get("refused"):
            worst = 1


# ═══════════════════════════════ selftest ════════════════════════════════════

def selftest() -> int:
    """Seconds, no executor, no card: tiny random model through the whole
    REPL machinery in dry-run, plus the pure display/refusal logic."""
    import contextlib
    import io
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                              " — " + extra if extra else ""))
        if not cond:
            fails.append(name)

    orig_layer = tf.decoder_layer_fx
    orig_core = tf.attention_core
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        model = pof._tiny_model(tdp / "w")
        ns = argparse.Namespace(
            max_tokens=2, offload_layer=1, offload_head=1, tier="kvq8",
            group=128, tile_div=bridge.TILE_DIV_DEFAULT, slot=0,
            timeout_s=60, compute_job=None)
        led = Ledger(tty=False, out=io.StringIO())
        buf = io.StringIO()
        led.start()
        with contextlib.redirect_stdout(buf):
            r = run_one(model, None, [3, 1, 4, 1, 5], "<ids>", ns, "golden",
                        led, tdp / "work", 1)
        out = buf.getvalue()
        n_steps = 5 + 2 - 1
        chk("dry-run prompt verdict PASS", r["ok"], str(r.get("error")))
        chk("token identity ON==OFF", r["ids_on"] == r["ids_off"],
            "%s vs %s" % (r.get("ids_on"), r.get("ids_off")))
        chk("offload fired once per step", r["n_records"] == n_steps,
            "%s records, expected %d" % (r["n_records"], n_steps))
        chk("ledger printed one TILE-SLOT line per step",
            sum("TILE-SLOT" in ln for ln in led.lines) >= n_steps)
        chk("ledger names HOST work explicitly",
            any("HOST" in ln for ln in led.lines))
        chk("footer says dry-run proves nothing about the tile",
            "DRY-RUN" in out and "NOTHING" in out)
        chk("footer refuses performance claims",
            "NOT a" in out and "performance" in out)
        chk("verdict prints IDENTICAL", "IDENTICAL" in out)
        chk("json record written",
            r["record"] and Path(r["record"]).is_file())
        chk("golden rebinds restored after the run",
            tf.decoder_layer_fx is orig_layer
            and tf.attention_core is orig_core
            and tf.attention_core is at.attention_core)

        # fence refusal — never reaches decode
        buf2 = io.StringIO()
        ns2 = argparse.Namespace(**{**vars(ns), "max_tokens": 128})
        with contextlib.redirect_stdout(buf2):
            r2 = run_one(model, None, [1] * 5, "<ids>", ns2, "golden",
                         Ledger(tty=False, out=io.StringIO()),
                         tdp / "w2", 2)
        chk("T>128 fence refuses loudly",
            r2["refused"] and "ChunkedHead" in buf2.getvalue())

        # bad target refusal
        buf3 = io.StringIO()
        ns3 = argparse.Namespace(**{**vars(ns), "offload_layer": 99})
        with contextlib.redirect_stdout(buf3):
            r3 = run_one(model, None, [1, 2], "<ids>", ns3, "golden",
                         Ledger(tty=False, out=io.StringIO()),
                         tdp / "w3", 3)
        chk("out-of-range target refused", r3["refused"])

        # a DIFFERENT token must be a loud failure (pure verdict logic)
        okd, lines = build_verdict({"ids": [1], "seconds": 0.0},
                                   {"ids": [2], "seconds": 0.0},
                                   [], None, "sim")
        chk("DIFFERENT verdict fails loudly",
            not okd and any("DIFFERENT" in ln for ln in lines)
            and any("LOUD FAILURE" in ln for ln in lines))
        chk("zero-capture run cannot pass",
            any("proves NOTHING" in ln for ln in lines))

        # ══════════ LAYER MODE: the six-op seam, same REPL machinery ═══════
        print("  --- layer mode (--offload layer, dry-run) ---")
        lmodel = lof._tiny(tdp / "wl")           # D=256 H=2 hd=128 dff=256 L=2
        lids = [3, 1, 4, 1, 5]
        lns = argparse.Namespace(
            max_tokens=1, offload="layer", offload_layer=1, offload_head=0,
            offload_step=None, proj_cols=lof.MXE_N, bus_check="auto",
            tier="kvq8", group=128, tile_div=bridge.TILE_DIV_DEFAULT, slot=0,
            timeout_s=60, compute_job=None)
        lled = Ledger(tty=False, out=io.StringIO())
        lbuf = io.StringIO()
        cache: dict = {}
        with contextlib.redirect_stdout(lbuf):
            lr = run_one(lmodel, None, list(lids), "<ids>", lns, "golden",
                         lled, tdp / "lwork", 1, off_cache=cache)
        lout = lbuf.getvalue()
        n_lsteps = len(lids) + lns.max_tokens - 1
        chk("layer: dry-run verdict PASS", lr["ok"], str(lr.get("error")))
        chk("layer: token identity ON==OFF", lr["token_identity"],
            "%s vs %s" % (lr.get("ids_on"), lr.get("ids_off")))
        chk("layer: all six op families served",
            lr.get("op_families_served") == list(lof.OP_TYPES),
            str(lr.get("op_families_served")))
        chk("layer: ledger shows a dispatch line for every op family",
            all(any(">>" in ln and (" %s " % op) in ln for ln in lled.lines)
                for op in lof.OP_TYPES),
            str([op for op in lof.OP_TYPES
                 if not any(">>" in ln and (" %s " % op) in ln
                            for ln in lled.lines)]))
        chk("layer: ledger names who computed each op (TILE/GOLDEN vs HOST)",
            any("GOLDEN (dry-run)" in ln for ln in lled.lines)
            and any("HOST" in ln for ln in lled.lines))
        chk("layer: ledger says the host runs the other layers at the "
            "offloaded step",
            any("the other 1 layer(s) ran on the HOST" in ln
                for ln in lled.lines))
        chk("layer: ledger says the host runs everything between offloaded "
            "steps",
            sum("ALL 2 layers HOST" in ln for ln in lled.lines)
            == n_lsteps - 1,
            "%d of %d non-offloaded steps"
            % (sum("ALL 2 layers HOST" in ln for ln in lled.lines),
               n_lsteps - 1))
        chk("layer: ledger prints per-family coverage counts + op-family "
            "tally",
            any("6/6 op families" in ln
                and all(("%s " % op) in ln for op in lof.OP_TYPES)
                for ln in lled.lines))
        chk("layer: verdict prints the per-family coverage table",
            all(lof.OP_LABEL[op] in lout for op in lof.OP_TYPES))
        chk("layer: verdict prints IDENTICAL", "IDENTICAL" in lout)
        chk("layer: substitution/consumption checks ran and all passed",
            lr["checks"] and all(c2["ok"] for c2 in lr["checks"]),
            "%d checks" % len(lr["checks"]))
        chk("layer: C-1-view ops are called reconstructions, not bit-exact",
            "reconstructed" in lout)
        chk("layer: footer prints layer_offload.host_ledger's own items "
            "verbatim",
            "host_ledger" in lout
            and "1 of 2 layers ran entirely on the host" in lout
            and "requant epilogues" in lout
            and "B-FEED-WIDTH" in lout)
        chk("layer: footer discloses the C-LBUS BUS_ON composition",
            "BUS_ON" in lout and "apex_layer_deq" in lout)
        chk("layer: footer says dry-run proves nothing about the tile",
            "DRY-RUN" in lout and "NOTHING" in lout)
        chk("layer: footer refuses performance claims",
            "NOT a" in lout and "performance" in lout
            and "throughput" in lout)
        chk("layer: json record written with the per-op ledger",
            lr["record"] and Path(lr["record"]).is_file()
            and "ledger" in json.loads(Path(lr["record"]).read_text()))
        try:
            _assert_golden_restored()
            chk("layer: every golden rebind restored after the run", True)
        except AssertionError as e:
            chk("layer: every golden rebind restored after the run", False,
                str(e))

        # the OFF reference is reused across prompts — and SAYS so
        lled2 = Ledger(tty=False, out=io.StringIO())
        lbuf2 = io.StringIO()
        with contextlib.redirect_stdout(lbuf2):
            lr2 = run_one(lmodel, None, list(lids), "<ids>", lns, "golden",
                          lled2, tdp / "lwork", 2, off_cache=cache)
        chk("layer: OFF reference reused across prompts, and labelled",
            lr2["off_reused"] and lr2["ids_off"] == lr["ids_off"]
            and any("REUSED" in ln for ln in lled2.lines)
            and "REUSED" in lbuf2.getvalue(), str(lr2.get("off_reused")))

        # layer-mode refusals
        buf4 = io.StringIO()
        ns4 = argparse.Namespace(**{**vars(lns), "offload_step": 99})
        with contextlib.redirect_stdout(buf4):
            r4 = run_one(lmodel, None, list(lids), "<ids>", ns4, "golden",
                         Ledger(tty=False, out=io.StringIO()), tdp / "l4", 4)
        chk("layer: out-of-range --offload-step refused",
            r4["refused"] and "outside" in buf4.getvalue())
        buf5 = io.StringIO()
        ns5 = argparse.Namespace(**{**vars(lns), "offload_layer": 99})
        with contextlib.redirect_stdout(buf5):
            r5 = run_one(lmodel, None, list(lids), "<ids>", ns5, "golden",
                         Ledger(tty=False, out=io.StringIO()), tdp / "l5", 5)
        chk("layer: out-of-range --offload-layer refused", r5["refused"])
        buf6 = io.StringIO()
        ns6 = argparse.Namespace(**{**vars(lns), "max_tokens": 130})
        with contextlib.redirect_stdout(buf6):
            r6 = run_one(lmodel, None, list(lids), "<ids>", ns6, "golden",
                         Ledger(tty=False, out=io.StringIO()), tdp / "l6", 6)
        chk("layer: T>128 fence refuses loudly",
            r6["refused"] and "ChunkedHead" in buf6.getvalue())
        buf7 = io.StringIO()
        ns7 = argparse.Namespace(**{**vars(lns), "proj_cols": 4})
        with contextlib.redirect_stdout(buf7):
            r7 = run_one(lmodel, None, list(lids), "<ids>", ns7, "golden",
                         Ledger(tty=False, out=io.StringIO()), tdp / "l7", 7)
        chk("layer: a --proj-cols that cannot serve a whole MXE block is "
            "refused",
            r7["refused"] and "ALL SIX" in buf7.getvalue())

        # pure verdict logic: the layer-mode gates, with no run at all
        empty_agg = {op: lof._agg([], op) for op in lof.OP_TYPES}
        rn = lof.Runner("sim", tdp / "lz")
        okz, lz = build_layer_verdict({"ids": [1], "seconds": 0.0},
                                      {"ids": [1], "seconds": 0.0}, None, [],
                                      [], empty_agg, None, "sim", 0, 0, 1, rn)
        chk("layer: a run that served no op cannot pass",
            not okz and any("proves NOTHING" in ln for ln in lz))
        chk("layer: a run with ZERO captures cannot pass",
            any("ZERO capture records" in ln for ln in lz))
        bad = [lof.OpRec(op="attn", name="x", source="TILE (sim)", n_served=1,
                         n_total=1, jobs=1, caps=1, exact=False,
                         grade_ok=False)]
        okg, lg = build_layer_verdict(
            {"ids": [1], "seconds": 0.0}, {"ids": [1], "seconds": 0.0}, None,
            bad, [("c", True, None)],
            {op: lof._agg(bad, op) for op in lof.OP_TYPES}, None, "golden",
            0, 0, 1, rn)
        chk("layer: a tile egress that differs from golden cannot pass",
            not okg and any("differs from golden" in ln for ln in lg))
        good1 = [lof.OpRec(op="attn", name="x", source="TILE (sim)",
                           n_served=1, n_total=1, jobs=1, caps=1, exact=True,
                           grade_ok=True)]
        okf, lf = build_layer_verdict(
            {"ids": [1], "seconds": 0.0}, {"ids": [1], "seconds": 0.0}, None,
            good1, [("c", True, None)],
            {op: lof._agg(good1, op) for op in lof.OP_TYPES}, None, "golden",
            0, 0, 1, rn)
        chk("layer: fewer than 6/6 op families cannot pass as a full layer",
            not okf and any("only 1/6 op families" in ln for ln in lf)
            and any("Left on the HOST" in ln for ln in lf))
        okd2, ld2 = build_layer_verdict(
            {"ids": [2], "seconds": 0.0}, {"ids": [1], "seconds": 0.0},
            {"ids": [1], "seconds": 0.0, "fired": 1}, bad,
            [("c", True, None)],
            {op: lof._agg(bad, op) for op in lof.OP_TYPES}, None, "golden",
            0, 0, 1, rn)
        chk("layer: a DIFFERENT token fails loudly and is attributed",
            not okd2 and any("DIFFERENT" in ln for ln in ld2)
            and any("LOUD FAILURE" in ln for ln in ld2)
            and any("TILE-SERVED VALUE" in ln for ln in ld2))
        okd3, ld3 = build_layer_verdict(
            {"ids": [2], "seconds": 0.0}, {"ids": [1], "seconds": 0.0}, None,
            bad, [("c", True, None)],
            {op: lof._agg(bad, op) for op in lof.OP_TYPES}, None, "golden",
            0, 0, 1, rn)
        chk("layer: without the BUS_ON run, a token change is NOT attributed",
            not okd3 and any("NOT established" in ln for ln in ld3))
        print("  --- shared / hw plumbing ---")

        # hw bring-up command lines (no instance contacted) — IMAGE-KEYED
        c = hw_bringup_cmds(0, DEFAULT_AGFI)
        chk("bringup: image cmd",
            c["image"] == "sudo -n fpga-load-local-image -S 0 -I %s"
            % DEFAULT_AGFI)
        chk("bringup: A2 image keys clkgen recipe -a 2 / 15.625",
            c["recipe"] == "sudo -n fpga-load-clkgen-recipe -S 0 -a 2"
            and c["recipe_idx"] == 2 and c["expect_mhz"] == 15.625
            and not c["underclocked"])

        def refused(name, fn):
            try:
                fn()
                chk(name, False)
            except SystemExit as e:
                chk(name, True, str(e)[:60])
        refused("bringup: malformed AGFI refused",
                lambda: hw_bringup_cmds(0, "not-an-agfi"))
        refused("bringup: UNREGISTERED AGFI refused (unknown clock)",
                lambda: hw_bringup_cmds(0, "agfi-000000000000000ff"))
        refused("bringup: OVERCLOCK run_recipe refused on an A2 image",
                lambda: hw_bringup_cmds(0, DEFAULT_AGFI, run_recipe=0))

        # a (fake) A0-registered image keys -a 0 / 62.5, and its explicit
        # underclock arm keys -a 2 / 15.625 with the flag set; an A2 image
        # can NEVER be flown as A0 (previous check) nor an A0 one silently
        # flown as A2 (the arm must be explicit).
        fake_a0 = "agfi-0feedfacefeedface"
        assert fake_a0 not in clock_key.IMAGE_RECIPE
        clock_key.IMAGE_RECIPE[fake_a0] = (0, "apex_repl selftest fake")
        try:
            c0 = hw_bringup_cmds(0, fake_a0)
            chk("bringup: A0 image keys clkgen recipe -a 0 / 62.5",
                c0["recipe"] == "sudo -n fpga-load-clkgen-recipe -S 0 -a 0"
                and c0["expect_mhz"] == 62.5 and not c0["underclocked"])
            cu = hw_bringup_cmds(0, fake_a0, run_recipe=2)
            chk("bringup: A0 image underclock arm = -a 2 / 15.625, flagged",
                cu["recipe"].endswith("-a 2") and cu["expect_mhz"] == 15.625
                and cu["underclocked"] and "UNDERCLOCK" in cu["clock_why"])
            # the footer names the SESSION's keyed clock, not a constant
            old = dict(SESSION_CLOCK)
            try:
                SESSION_CLOCK.update(mhz=62.5, recipe=0, underclocked=False)
                f_hw = "\n".join(_exec_footer("hw", 1.0, 1.0, 62.50))
                chk("footer: hw clock line is image-keyed (62.5 / A0)",
                    "62.5 MHz correctness clock (recipe A0" in f_hw)
                SESSION_CLOCK.update(mhz=15.625, recipe=2,
                                     underclocked=True)
                f_uc = "\n".join(_exec_footer("hw", 1.0, 1.0, 15.62))
                chk("footer: the underclock arm is disclosed",
                    "UNDERCLOCK ARM" in f_uc
                    and "15.625 MHz correctness clock (recipe A2" in f_uc)
            finally:
                SESSION_CLOCK.update(old)
        finally:
            clock_key.IMAGE_RECIPE.pop(fake_a0, None)

        # ledger elapsed stamps + status machinery don't crash in tty mode
        sio = io.StringIO()
        sio.isatty = lambda: True                       # type: ignore
        lt = Ledger(out=sio)
        lt.start()
        lt.tick("status")
        lt.op("line")
        lt.stop()
        chk("tty ledger writes CR updates", "\r" in sio.getvalue()
            and "line" in sio.getvalue())

    print("=" * 60)
    if fails:
        print("APEX_REPL SELFTEST FAIL: %s" % fails)
        return 1
    print("APEX_REPL SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live interactive prompt demo: type a prompt, watch the "
                    "tile serve one attention op per decode step (--offload "
                    "attention) or EVERY op family of one decoder layer "
                    "(--offload layer), see the token + identity verdict + "
                    "honest footer.")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--dry-run", action="store_true",
                    help="tile slot filled by golden — plumbing rehearsal "
                         "with NO executor at all")
    ap.add_argument("--host", default=None,
                    help="hw: ubuntu@<f2-ip> (or $APEX_F2_HOST)")
    ap.add_argument("--key", default=None,
                    help="hw: ssh key, e.g. ~/.ssh/apex-f2.pem (or "
                         "$APEX_F2_KEY)")
    ap.add_argument("--agfi", default=DEFAULT_AGFI,
                    help="hw: image to load at startup (default: the live "
                         "narrow+R4 image). Must be registered in "
                         "clock_key.IMAGE_RECIPE — the clkgen recipe AND the "
                         "numeric clock gate are keyed to it")
    ap.add_argument("--run-recipe", type=int, default=None,
                    help="hw: deliberate UNDERCLOCK arm — group-A recipe "
                         "index to run the keyed image at (e.g. 2 to fly an "
                         "A0 image at 15.625 for the A/B ladder). An "
                         "overclock refuses at bring-up")
    ap.add_argument("--skip-image-load", action="store_true",
                    help="hw: card already carries the image; still programs "
                         "the image-keyed recipe and still verifies the "
                         "clock numerically")
    ap.add_argument("--offload", choices=("attention", "layer"),
                    default="attention",
                    help="attention: ONE attention op (layer, head) per "
                         "decode step (Milestone C). layer: ALL SIX op "
                         "families "
                         "(%s) of --offload-layer at ONE decode step, via the "
                         "imported layer_offload.py (Milestone C2)."
                         % "/".join(lof.OP_TYPES))
    ap.add_argument("--offload-layer", type=int, default=0)
    ap.add_argument("--offload-head", type=int, default=0,
                    help="attention mode only")
    ap.add_argument("--offload-step", type=int, default=None,
                    help="layer mode: offload only at this 0-based decode "
                         "step (default: the LAST step, the one whose layer "
                         "output reaches the emitted token). A latency guard: "
                         "it changes no claim, only how many steps run.")
    ap.add_argument("--proj-cols", type=int, default=lof.MXE_N,
                    help="layer mode: output columns per projection call "
                         "served by the tile (multiple of 8; full width is "
                         "~300k GEMM jobs, so projections are SAMPLED and the "
                         "ledger says so). The other five run at FULL width.")
    ap.add_argument("--bus-check", choices=("auto", "always", "never"),
                    default="auto",
                    help="layer mode: extra HOST-ONLY run with the target "
                         "layer composed in C-LBUS BUS_ON, which separates "
                         "'the tile changed the token' from 'the bus mode "
                         "changed it'. auto = only when the tokens differ.")
    ap.add_argument("--no-reuse-reference", action="store_true",
                    help="recompute the pure-golden OFFLOAD-OFF reference "
                         "for every prompt, even when the ids/tier/group/"
                         "max-tokens match an earlier prompt this session")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--tier", default="kvq8", choices=list(rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--weights-dir",
                    default=str(REPO / "build/s8_weights/Qwen2.5-7B-4bit"))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--compute-job", default=None, metavar="PATH",
                    help="override the job-compiler module (also "
                         "$APEX_COMPUTE_JOB)")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="skip tokenizer load (tiny/random weights): "
                         "prompts via ':ids ...' only")
    ap.add_argument("--once", default=None, metavar="PROMPT",
                    help="run one prompt non-interactively and exit "
                         "(rehearsal/automation)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.host:
        os.environ["APEX_F2_HOST"] = args.host
    if args.key:
        os.environ["APEX_F2_KEY"] = args.key
    if not args.dry_run and args.executor == "hw":
        # Key the PER-JOB clock gate (remote_hw_exec via attach) to the same
        # image the bring-up loads — one source of truth for the whole
        # session, same asymmetric under/overclock rules.
        os.environ["APEX_F2_AGFI"] = args.agfi
        if args.run_recipe is not None:
            os.environ["APEX_F2_RUN_RECIPE"] = str(args.run_recipe)
    mode = "golden" if args.dry_run else args.executor
    work = Path(args.work_dir)

    say("=" * W)
    say("APEX PROMPT REPL — live FPGA-offload demo (warm session)")
    say("=" * W)

    hw_info: dict = {}
    if mode == "hw":
        hw_info = bringup_hw(args)               # SystemExit on any failure
        if not rhe.attach(bridge, args):
            raise SystemExit("REFUSE: remote executor did not attach — "
                             "$APEX_F2_HOST vanished?")
        say("[session] executor hw: every job re-verifies the clock "
            "numerically before it ships")
    elif mode == "sim":
        sim_bin = bridge.resolve_sim_binary()    # raises loudly if absent
        say("[session] executor sim: %s" % sim_bin)
        say("[session]   (Verilated cl_apex — the silicon twin; no card, "
            "no spend)")
        pof.load_compute_job(args.compute_job)   # fail NOW, not mid-demo
        say("[session] job compiler: compute_job.build_compute_job "
            "(produce-mode, requant_en=0)")
    else:
        say("[session] DRY-RUN: the tile slot is filled by golden — plumbing "
            "rehearsal ONLY;")
        say("[session]   nothing below demonstrates the tile")

    t0 = time.perf_counter()
    model = rt.GoldenModel(Path(args.weights_dir))
    say("[session] model: %s  L=%d H=%d head_dim=%d  (weights mapped in "
        "%.1fs)" % (model.meta["model"], model.n_layers, model.meta["H"],
                    model.meta["head_dim"], time.perf_counter() - t0))
    tok = None
    if args.no_tokenizer:
        say("[session] tokenizer: SKIPPED (--no-tokenizer) — ':ids ...' "
            "prompts only")
    else:
        try:
            t1 = time.perf_counter()
            tok = rt.load_tokenizer(model.meta["model"])
            say("[session] tokenizer: loaded (%.1fs)"
                % (time.perf_counter() - t1))
        except Exception as e:                    # noqa: BLE001 — report it
            say("[session] tokenizer FAILED to load: %s: %s" %
                (type(e).__name__, e))
            say("[session]   continuing in ':ids ...' mode — this is a "
                "degraded session, not a pass")
    if hw_info.get("a1_mhz") is not None:
        say("[session] tile clock: MEASURED %.2f MHz (want %g ± 0.2, keyed "
            "to %s, recipe A%d) —"
            % (hw_info["a1_mhz"], hw_info.get("expect_mhz", 15.625),
               args.agfi, hw_info.get("recipe_idx", 2)))
        say("[session]   correctness clock, never presented as performance")
    if args.offload == "layer":
        say("[session] offload mode: LAYER — ALL SIX op families (%s) of "
            "layer %d," % ("/".join(lof.OP_TYPES), args.offload_layer))
        say("[session]   at decode step %s · projections SAMPLED at "
            "--proj-cols %d, the other"
            % ("the last one" if args.offload_step is None
               else str(args.offload_step), args.proj_cols))
        say("[session]   five at FULL width · the offloaded layer composes in "
            "C-LBUS BUS_ON")
        say("[session]   (seam: layer_offload.LayerOffloader, imported "
            "unmodified)")
    else:
        say("[session] offload mode: ATTENTION — one attention op per decode "
            "step")
    say("[session] offload target: layer %d head %d · max new tokens %d"
        % (args.offload_layer, args.offload_head, args.max_tokens))
    say("[session] fence: prompt+new <= %d tokens (a chunked head cannot be "
        "offloaded)" % pof.TOKENS_MAX)
    off_cache = None if args.no_reuse_reference else {}
    say("[session] OFF reference: %s"
        % ("recomputed for every prompt (--no-reuse-reference)"
           if off_cache is None else
           "reused across prompts when ids/tier/group/max-tokens match "
           "(said so in the ledger)"))

    if args.once is not None:
        ledger = Ledger()
        ids = tok.encode(args.once) if tok else None
        if ids is None:
            raise SystemExit("--once needs a tokenizer (drop --no-tokenizer)")
        r = run_one(model, tok, list(ids), args.once, args, mode, ledger,
                    work, 1, a1_mhz=hw_info.get("a1_mhz"),
                    off_cache=off_cache)
        say("[exit] --once done — " + _exit_msg(mode))
        return 0 if r.get("ok") else 1

    return repl(model, tok, args, mode, work, hw_info, off_cache)


if __name__ == "__main__":
    sys.exit(main())
