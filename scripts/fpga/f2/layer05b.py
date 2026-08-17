#!/usr/bin/env python3
# layer05b.py — P2 stage 2, ITEM B: ALL SIX OP FAMILIES OF ONE Qwen2.5-0.5B
# DECODER LAYER, SERVED BY THE D=64 TILE, INSIDE A REAL PROMPT RUN.
#
#   python3 scripts/fpga/f2/layer05b.py --prompt "The capital of France is" \
#       --layer 0 --executor sim --poison 0.5
#   python3 scripts/fpga/f2/layer05b.py --selftest          # host only, seconds
#
# ══ WHAT THIS IS, AND WHAT IT IS NOT ═══════════════════════════════════════
# layer_offload.py proved MILESTONE C2 for Qwen2.5-7B on the D=128 image: one
# decoder layer's six op families served by the tile, token unchanged. It is
# owned/frozen and hardcodes D_TILE=128 (`gemm_job.py:115`), as does
# gen_layer_ops.py. This file is the SAME claim at 0.5B geometry on the D=64
# image whose build knobs ITEM A added:
#
#   hidden 896   24 layers   H=14 / H_kv=2 (GQA group 7)   head_dim 64
#   d_ffn 4864 = 76 x 64     rope_theta 1e6   tie_word_embeddings
#
# It is NOT a rewrite. The seam, the substitution/consumption checks, the
# A/B and the poison discriminator are layer_offload's — imported and
# SUBCLASSED. What this file supplies is exactly the three things that do not
# generalize from 128 to 64, each named and each proven:
#
#   (1) GEOMETRY — every emission runs inside `tile_geom.at_d(64)`, which
#       reframes the frozen emitters and is verified per program by
#       `audit_geometry` (the program's own INFO_D expectation must be 64).
#       `layer_offload.D_TILE` is rebound in the same scope, which is what
#       makes the inherited RoPE frame check, the C-1 reconstruction, the
#       norm chunk count and the SwiGLU chunks-per-row correct at 64.
#
#   (2) THE R4 ARM'S k — the ONE piece of real re-derivation.
#       gen_layer_ops.build_norm2_chunked arms k = dm/D_TILE for both the
#       streamed chunk count AND the μ-table index; at D=64 those are 14 and
#       7 and only 7 is right (tile_geom header, asu_wide_rms_params.svh
#       WIDE_RMS_CHUNK=128). `tile_geom.build_norm2_chunked_d` is swapped in
#       HERE, explicitly, so the swap is visible at the call site. The
#       difference is measured, not asserted: `--norm-k-red` re-runs the norm
#       with the frozen emitter's k and the ledger prints how far it lands
#       from golden.
#
#   (3) BUILD IDENTITY — the 0.5B image sets KVQ_GQA_NENG=2, so INFO_TIER
#       truthfully reads 0x1, not the 0x7 the L3 d64 choreography bakes
#       (apex_top.sv:2540-2543; gen_l3_vectors.py:349). Every program gets
#       `tile_geom.retarget_info_tier`, which rewrites that ONE expectation
#       and REFUSES if a program does not carry exactly one. Disclosed in the
#       ledger on every run, with the count.
#
# ══ THE 0.5B SHAPE FACTS THIS DRIVER DEPENDS ON ════════════════════════════
#   residual   896 <= RESID_WIN(1024) -> ONE window job per residual, base 0
#              (apex_residual R3; LAYER_DM_MAX=896 from +define+APEX_CL_DM).
#              7B needed 4 windows; 0.5B needs 1.
#   norm       896 = 14 x 64 streamed C-1 frames (the feeder IS CFG_DM=64),
#              armed ext_k = 896/128 = 7. See tile_geom's header for why
#              7 x 128 is unreachable on an image that also serves hd=64.
#   swiglu     4864 = 76 x 64 chunks; at D=64 that is exactly ONE swiglu
#              chunk per C-1 feeder row (per_row = D_TILE // SWG_COLS_MAX).
#   attention  head_dim 64: the score composite sqrt(D) = 2^7 is EXACT at
#              D=64 (seq_walker_comp.sv:45), and a D=128 image would REFUSE
#              these descriptors outright (head_dim != cfg_d -> WALK_ERR_DESC).
#   proj       K=896 contracts; stage_plan lays it on 64-wide rows (15 rows,
#              K_total 960 <= K_MAX 2048) -> one k-split job per matvec.
#
# ══ HONESTY ════════════════════════════════════════════════════════════════
# The inherited rules are unchanged and still bind: RMSNorm-2 and SwiGLU
# re-enter the host through the C-1 feeder, so their substituted values are
# RECONSTRUCTIONS with a measured delta, never claimed bit-exact
# (B-FEED-WIDTH); the projections are SAMPLED unless --proj-cols says
# otherwise; 23 of 24 layers run on the host. The ledger prints all of it.

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tinynpu as rt                                        # noqa: E402
from apex_golden import attention as at                         # noqa: E402
from apex_golden import transformer as tf                       # noqa: E402

import gen_layer_ops as gl                                      # noqa: E402
import layer_offload as lof                                     # noqa: E402
import prompt_offload as po                                     # noqa: E402
import tile_exec_bridge as bridge                               # noqa: E402
import tile_geom as tg                                          # noqa: E402

OP_TYPES = lof.OP_TYPES
D_TILE_05B = 64
# The frozen modules' own frame widths, captured BEFORE any rebind, so the
# "did the geometry scope leak?" assertion compares against what the modules
# actually shipped with rather than a hardcoded 128.
_FROZEN_FRAMES = (lof.D_TILE, gl.D_TILE)
DEFAULT_WORK = REPO / "build" / "layer05b"
DEFAULT_WEIGHTS = REPO / "build/s8_weights/Qwen2.5-0.5B-Instruct-4bit"


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═══════════════════ the D=64 emission scope (explicit) ════════════════════

@contextmanager
def d64_emission(d: int = D_TILE_05B):
    """Everything an emission needs to be a D=`d` emission, in one place.

    * tile_geom.at_d      reframes gen_layer_ops / gemm_job (verified).
    * layer_offload.D_TILE is the frame width the INHERITED wrappers reason
      with — the RoPE `hd != D_TILE` gate, `_codes_scales`/`_c1_reconstruct`,
      the norm's `dm % D_TILE`, the SwiGLU chunks-per-C-1-row. Left at 128 it
      would silently leave RoPE on the host and mis-shape every C-1 readback.
    * gen_layer_ops.build_norm2_chunked is REPLACED by the re-derived builder
      for the duration — see this file's header (2). The swap is here, at the
      call site, rather than hidden inside tile_geom, so a reader of this
      driver cannot miss that the frozen norm emitter is NOT the one running.
    """
    o_d, o_norm = lof.D_TILE, gl.build_norm2_chunked

    def _norm_d(step, *, dm, out_dir, name):
        return tg.build_norm2_chunked_d(step, dm=dm, d=d, out_dir=out_dir,
                                        name=name)

    with tg.at_d(d):
        lof.D_TILE = d
        gl.build_norm2_chunked = _norm_d
        try:
            yield d
        finally:
            lof.D_TILE = o_d
            gl.build_norm2_chunked = o_norm


# ═══════════════════════ the runner: retarget + audit ══════════════════════

class Runner05B(lof.Runner):
    """layer_offload.Runner + the two per-program obligations of this image:
    the disclosed INFO_TIER retarget and the geometry audit. Both happen
    between emission and execution, so nothing runs unretargeted or
    un-audited — and both REFUSE rather than warn.

    They hang off `prepare`, which the base Runner calls once per DISTINCT
    program CONTENT — before the program is keyed, queued, executed or served
    from the capture pool. So a program that is only ever served from the pool
    was still audited and retargeted, and a program a replay RE-EMITS gets
    both again on the new bytes.

    `n_audited` therefore counts AUDIT PASSES (>= the number of distinct
    programs, because a re-emitted file is audited again); the set of distinct
    programs audited is `audited_keys`, and `unaudited_executed()` is the gate
    that must read 0.
    """

    def __init__(self, *a, image: tg.Image = tg.IMG_05B, d: int = D_TILE_05B,
                 **kw):
        super().__init__(*a, **kw)
        self.image, self.d = image, d
        self.n_retargeted = 0
        self.n_audited = 0
        self.desc_k_seen: set = set()

    def prepare(self, path: str) -> None:
        if self.mode == "golden":
            return
        a = tg.audit_geometry(path, self.d)
        self.desc_k_seen.update(a["k_values"])
        self.n_audited += 1
        r = tg.retarget_info_tier(path, self.image)
        self.n_retargeted += r["retargeted"]

    def unaudited_executed(self) -> int:
        """Distinct programs an executor ran that `prepare` never saw.

        Must be 0. `_execute` already REFUSES an unprepared path, so this is
        a structural zero — it is recomputed from the two sets and printed
        anyway, because a gate nobody evaluates is not a gate.
        """
        if self.mode == "golden":
            return 0
        return sum(1 for k in self.executed_keys if k not in self.audited_keys)


# ═══════════════════════════ the offloader ═════════════════════════════════

class Offloader05B(lof.LayerOffloader):
    """layer_offload.LayerOffloader with the 0.5B/D=64 fences made explicit.

    Every op wrapper, every substitution check, the arm/disarm bookkeeping and
    the poison path are INHERITED verbatim — they are geometry-neutral once
    layer_offload.D_TILE is reframed. What is added: a refusal if the model
    handed to it is not the geometry this driver was audited for, and the
    norm-arm provenance recorded per run.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.norm2_operands = None

    def _layer(self, X, w, tier, *a, **kw):
        hd = int(w.head_dim)
        if hd != lof.D_TILE:
            raise SystemExit(
                f"REFUSE: head_dim={hd} but the emission scope is framed at "
                f"{lof.D_TILE}. This driver is the D={lof.D_TILE} image's "
                f"driver; a head_dim={hd} model belongs to the D={hd} one "
                f"(layer_offload.py for 128). Refusing to emit descriptors "
                f"whose k does not match the head the model will consume.")
        return super()._layer(X, w, tier, *a, **kw)

    def _norm(self, x, g, chunk: int = 128):
        """Inherited verbatim — except the operands are RECORDED, so the red
        arm below can re-emit the SAME row with the frozen emitter's k."""
        if self.armed and not self._busy("norm"):
            gi = np.asarray(g, dtype=np.int64)
            if self.cur is not None and gi.size == np.asarray(
                    self.cur.w.gamma2).size and np.array_equal(
                    gi, np.asarray(self.cur.w.gamma2, dtype=np.int64)):
                self.norm2_operands = {
                    "r1_8": np.asarray(x, dtype=np.int64).copy(),
                    "gamma2": gi.copy()}
        return super()._norm(x, g, chunk)


# ══════════════════════════ the red arm (norm k) ═══════════════════════════

_FROZEN_NORM = gl.build_norm2_chunked          # captured at import, pre-swap


def norm_k_red(step, dm: int, d: int, work: Path, image, runner) -> dict:
    """Measure what the FROZEN emitter's k would have produced on this image.

    Not a rhetorical claim: it emits the SAME row through
    gen_layer_ops.build_norm2_chunked (captured at import, before
    d64_emission's swap), which at D_TILE=d arms k = dm/d, runs it on the
    same twin and grades it against the same golden row. If this came back
    bit-exact the re-derivation in tile_geom would be pointless — and this
    driver's headline would be resting on nothing.
    """
    y, _, _ = at.rmsnorm_fx_wide(
        [int(v) for v in np.asarray(step["r1_8"]).ravel()[:dm]],
        [int(v) for v in np.asarray(step["gamma2"]).ravel()[:dm]])
    y = np.asarray(y, dtype=np.int64)
    del image                            # the runner owns the retarget
    with tg.at_d(d):                     # geometry ONLY — no norm swap here
        m = _FROZEN_NORM(step, dm=dm, out_dir=work, name="norm_k_red")
        caps = runner.run([m["path"]], "norm_k_red")[0]
        g = gl.grade_codes_scales(caps, y.astype(np.float64) / 256.0,
                                  rows=dm // d)
    return {"armed_k": dm // d, "correct_k": dm // tg.RMS_MU_CHUNK,
            "bit_exact": bool(g.get("equal")),
            "n_code_diff": int(g.get("n_code_diff", -1)),
            "scales_equal": bool(g.get("scales_equal"))}


# ═══════════════════════════ the decode (A/B) ══════════════════════════════

def run_ab(model, tok, args, work: Path) -> int:
    ids = tok.encode(args.prompt) if tok else list(args.ids)
    n_total = len(ids) + args.max_tokens
    if n_total > lof.TOKENS_MAX:
        eprint(f"REFUSE: prompt({len(ids)}) + max_tokens({args.max_tokens}) = "
               f"{n_total} > {lof.TOKENS_MAX} — beyond that a head becomes a "
               f"ChunkedHead whose merge needs per-chunk sm_m/sm_l, which the "
               f"mailbox does not expose (audit N8).")
        return 2
    L, H = model.n_layers, model.meta["H"]
    hd, dm = int(model.meta["head_dim"]), int(model.meta["D_model"])
    dff = int(model.meta["d_ffn"])
    if hd != args.d:
        eprint(f"REFUSE: model head_dim {hd} != --d {args.d}")
        return 2
    if not 0 <= args.layer < L:
        eprint(f"REFUSE: --layer {args.layer} outside [0,{L})")
        return 2
    tier = rt.TIER_MAP[args.tier]
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    mode = "golden" if args.dry_run else args.executor
    image = tg.IMG_05B
    if mode != "golden" and not image.binary.exists():
        eprint(f"REFUSE: the {image.name} twin is not built.\n"
               f"  cd verif/f2sim && make build D={image.cfg_d} DDR=0 "
               f"OBJ={image.obj} VFLAGS_EXTRA=\"{image.defines()}\"")
        return 2

    step_target = args.offload_step
    if step_target is None:
        # the LAST prefill step — the one whose layer output actually reaches
        # the emitted logits (layer_offload's selftest rationale, and what
        # makes the poison discriminator meaningful).
        step_target = len(ids) - 1

    scope = lof.Scope(ops=tuple(args.ops), proj_cols=args.proj_cols,
                      proj_rows=args.proj_rows, heads=args.heads,
                      rope_heads=args.rope_heads, resid_cols=args.resid_cols,
                      resid_slice=(dm if args.resid_slice < 0
                                   else args.resid_slice),
                      norm_dm=args.norm_dm, norm1_rows=args.norm1_rows,
                      swiglu_chunks=args.swiglu_chunks)
    work.mkdir(parents=True, exist_ok=True)

    eprint(f"[P2-B] model={model.meta['model']}  L={L} H={H} H_kv="
           f"{model.meta.get('H_kv')} head_dim={hd} D_model={dm} d_ffn={dff}")
    eprint(f"[P2-B] image={image.name}: CFG_D={image.cfg_d} "
           f"CFG_DM={image.cfg_dm} GQA={image.gqa_neng} DM_MAX={image.dm_max} "
           f"QSTAGE={image.qstage}  INFO_TIER truth {image.info_tier:#x}")
    eprint(f"[P2-B] target: layer {args.layer} step {step_target}  mode={mode}"
           f"  ops={','.join(scope.ops)}  resid_slice={scope.resid_slice}")

    runner = Runner05B(mode, work, image=image, d=args.d, binary=(
        args.binary or (str(image.binary) if mode == "sim" else None)),
        tile_div=args.tile_div, slot=args.slot, timeout_s=args.timeout_s,
        collapse=args.collapse, max_batch_mb=args.max_batch_mb)
    eprint(f"[P2-B] transport: collapse={args.collapse} "
           f"max_batch_mb={args.max_batch_mb} max_passes={args.max_passes}")

    eprint("[P2-B] === run 1/3: OFFLOAD ON ===")
    t0 = time.perf_counter()
    with d64_emission(args.d):
        with Offloader05B(n_layers=L, layer=args.layer, mode=mode, work=work,
                          scope=scope, runner=runner, step=step_target,
                          max_passes=args.max_passes) as off:
            on = po.decode(model, ids, args.max_tokens, tier, args.group, eos,
                           None)
    on["records"], on["checks"] = off.records, off.checks
    on["passes"], on["invocations"] = dict(off.passes), dict(off.invocations)
    on["wall"] = time.perf_counter() - t0

    eprint("[P2-B] === run 2/3: OFFLOAD OFF (pure golden, default bus) ===")
    assert tf.attention_core is at.attention_core, "rebind not restored"
    assert (lof.D_TILE, gl.D_TILE) == _FROZEN_FRAMES \
        and gl.build_norm2_chunked is _FROZEN_NORM, \
        "the geometry rebind leaked out of the emission scope — the pure-host " \
        "arm of the A/B would not be a pure-host arm"
    offr = po.decode(model, ids, args.max_tokens, tier, args.group, eos, None)

    eprint("[P2-B] === run 3/3: HOST ONLY, target layer in C-LBUS BUS_ON ===")
    with lof.BusOnly(L, args.layer, step_target) as bo:
        bus = po.decode(model, ids, args.max_tokens, tier, args.group, eos,
                        None)
    bus["fired"] = bo.fired

    poison = None
    if args.poison is not None:
        eprint(f"[P2-B] === discriminator: every tile value x{args.poison} ===")
        with d64_emission(args.d):
            with Offloader05B(n_layers=L, layer=args.layer, mode=mode,
                              work=work, scope=scope, runner=runner,
                              step=step_target, poison=args.poison,
                              max_passes=args.max_passes) as off2:
                poison = po.decode(model, ids, args.max_tokens, tier,
                                   args.group, eos, None)
        poison["records"] = off2.records

    red = None
    if args.norm_k_red and mode != "golden":
        if off.norm2_operands is None:
            eprint("[P2-B] --norm-k-red: the RMSNorm-2 operands were never "
                   "seen (was 'norm' in --ops?); skipping the red arm.")
        else:
            eprint("[P2-B] === red arm: the FROZEN emitter's norm k ===")
            red = norm_k_red(off.norm2_operands, dm, args.d, work, image,
                             runner)

    return report(model, tok, args, ids, on, offr, bus, poison, red, mode,
                  scope, runner, image, work, step_target)


# ═══════════════════════════════ the ledger ════════════════════════════════

OP_LABEL = lof.OP_LABEL
EGRESS = {
    "proj": "RO lanes, raw INT32                  EXACT",
    "rope": "C-1 view (64 codes + row scale)      EXACT where consumed*",
    "attn": "RO lanes INT32 + captured s_c        EXACT",
    "resid": "LAYER_RDATA fp16, ONE 896 window     EXACT",
    "norm": "C-1 view, 14 x 64 frames             LOSSY (B-FEED-WIDTH)",
    "swiglu": "C-1 view, 76 x 64 frames             LOSSY (B-FEED-WIDTH)",
}


def report(model, tok, args, ids, on, offr, bus, poison, red, mode, scope,
           runner, image, work, step_target) -> int:
    recs, checks = on["records"], on["checks"]
    ident = on["ids"] == offr["ids"]
    bus_ident = bus["ids"] == offr["ids"]
    fired = any(r.n_served for r in recs)
    agg = {op: lof._agg(recs, op) for op in OP_TYPES}
    served_ops = [op for op in OP_TYPES if agg[op]["values"] > 0]
    checks_ok = all(ok for _n, ok, _d in checks)
    grades_ok = all(agg[op]["grade_ok"] is not False for op in OP_TYPES)
    unaudited = runner.unaudited_executed()
    geom_ok = (mode == "golden") or (
        len(runner.audited_keys) >= len(runner.executed_keys) > 0
        and unaudited == 0)

    def txt(v):
        return tok.decode(v) if tok else str(v)

    W = 78
    print("\n" + "=" * W)
    print("P2 ITEM B — ONE Qwen2.5-0.5B DECODER LAYER, EVERY OP FAMILY ON THE "
          "D=64 TILE")
    print("=" * W)
    print(f"  prompt          : {args.prompt!r}")
    print(f"  prompt ids      : {ids} ({len(ids)} tokens)")
    print(f"  model           : {model.meta['model']}")
    print(f"                    L={model.n_layers} H={model.meta['H']} "
          f"H_kv={model.meta.get('H_kv')} head_dim={model.meta['head_dim']} "
          f"D_model={model.meta['D_model']} d_ffn={model.meta.get('d_ffn')}")
    print(f"  image           : {image.name}  CFG_D={image.cfg_d} "
          f"CFG_DM={image.cfg_dm} KVQ_GQA_NENG={image.gqa_neng} "
          f"RMS/LAYER_DM_MAX={image.dm_max} QSTAGE_H_MAX={image.qstage}")
    print(f"                    build: make build D={image.cfg_d} DDR=0 "
          f"OBJ={image.obj} VFLAGS_EXTRA=\"{image.defines()}\"")
    print(f"  offloaded layer : {args.layer}  step {step_target}  "
          f"composition C-LBUS BUS_ON")
    print(f"  executor        : {mode}"
          + ("   (--dry-run: every tile call replaced by golden — seam only)"
             if mode == "golden" else f"   binary {runner.binary}"))
    print(f"  tile jobs       : {runner.n_jobs} programs, {runner.n_caps} "
          f"capture records, {runner.wall:.1f}s in the executor")
    for line in lof.transport_lines(runner, on):
        print(f"  {line}")
    if mode != "golden":
        print(f"  geometry audit  : {len(runner.audited_keys)} distinct "
              f"programs audited ({runner.n_audited} audit passes over "
              f"re-emitted files), {len(runner.executed_keys)} distinct "
              f"programs executed, {unaudited} executed WITHOUT an audit — "
              f"each audited program carries INFO_D == {args.d} (so each was "
              f"emitted at D={args.d}); descriptor k values seen = "
              f"{sorted(runner.desc_k_seen)}  -> "
              f"{'PASS' if geom_ok else 'FAIL'}")
        print(f"  disclosed retarget: {runner.n_retargeted} INFO_TIER "
              f"expectations rewritten 0x7 -> {image.info_tier:#x}, exactly "
              f"one per audited program (apex_top.sv:2540-2543: KVQ_GQA_NENG="
              f"{image.gqa_neng} is CQ-8-only). No datapath expectation "
              f"touched.")
    print()
    print("  PER-OP LEDGER — who computed each op family of the offloaded layer")
    print("  " + "-" * (W - 2))
    print(f"  {'op family':<26} {'who':<16} {'served/total values':<21} exact?")
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
                          "chunks", "rows", "slices", "dm"):
                    if k in d:
                        print(f"        {k:<32}: {d[k]}")
        for n in a["notes"]:
            print(f"      NOTE: {n}")
    print("  * RoPE: the C-1 reconstruction is re-quantized by golden's very "
          "next stage")
    print("    (attention_core Q7, quant_rows_i8) — measured identical to "
          "golden's own codes.")
    print()
    print("  SUBSTITUTION / CONSUMPTION CHECKS (the tile's value reached the "
          "model)")
    for name, ok, detail in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  {detail}" if detail and not ok else ""))
    if not checks:
        print("    (none — no op was served)")
    print()
    if red is not None:
        print("  RED ARM — the frozen emitter's RMSNorm k on this image")
        print(f"    armed k = dm/D = {red['armed_k']} (gen_layer_ops.py:1114) "
              f"vs the correct μ index dm/128 = {red['correct_k']}")
        print(f"    result: bit-exact vs golden = {red['bit_exact']}, "
              f"{red['n_code_diff']} of {model.meta['D_model']} codes differ, "
              f"row scales equal = {red['scales_equal']}")
        print(f"    -> the re-derivation in tile_geom.py is LOAD-BEARING "
              f"({'PROVEN' if not red['bit_exact'] else 'VACUOUS'})")
        print()
    # HOW FAR did the tile move the logits, and how far was there to move?
    # Without both numbers a token flip is unreadable: it could mean the tile
    # is wrong, or it could mean the top-1 was decided by less than the C-1
    # reconstruction noise the ledger already disclosed. Say which.
    margin = _margin(offr, on)
    print(f"  logit geometry    : pure-host top-1 margin "
          f"{margin['host_margin']:.4g} over a {margin['host_range']:.4g} "
          f"logit range ({100 * margin['host_margin'] / margin['host_range']:.2f}"
          f"% of range); the tile-served layer moved the logits by "
          f"max|dlogit| = {margin['dlogit']:.4g}")
    print(f"  token OFFLOAD ON  : ids={on['ids']} text={txt(on['ids'])!r} "
          f"({on['seconds']:.0f}s)")
    print(f"  token PURE HOST   : ids={offr['ids']} text={txt(offr['ids'])!r} "
          f"({offr['seconds']:.0f}s)")
    print(f"  token HOST+BUS_ON : ids={bus['ids']} text={txt(bus['ids'])!r} "
          f"({bus['seconds']:.0f}s, layer composed BUS_ON {bus['fired']}x) "
          f"-> {'same as pure host' if bus_ident else 'DIFFERS from pure host'}")
    dl = None
    if poison is not None:
        dl = (float(np.max(np.abs(poison["logits"] - on["logits"])))
              if poison["logits"] is not None and on["logits"] is not None
              else float("nan"))
        verdict = ("substitutions ARE load-bearing" if dl > 0 else
                   "VACUOUS — the tile values never reached the token")
        print(f"  discriminator     : tile values x{args.poison} -> "
              f"ids={poison['ids']}, max|dlogit|={dl:.4g} ({verdict})")
    ok = bool(ident and fired and checks_ok and grades_ok and geom_ok
              and (dl is None or dl > 0))
    print("-" * W)
    if not fired:
        print("  FAIL: no op was served by the tile — token identity proves "
              "nothing.")
    if not checks_ok:
        print("  FAIL: a tile-derived value was not the value the model "
              "consumed.")
    if not grades_ok:
        print("  FAIL: a tile egress differs from golden's own view of the "
              "same op.")
    if not geom_ok:
        print("  FAIL: a program reached the tile framed at the wrong D.")
    if not ident and bus_ident:
        print("  NOTE: the token changed and the BUS_ON-only run did NOT "
              "change it — the difference came from a tile-served value.")
        lossy = [op for op in ("norm", "swiglu") if agg[op]["values"]]
        if lossy and margin["dlogit"] < margin["host_margin"] * 20:
            print(f"        Every family's tile output is BIT-EXACT vs "
                  f"golden's own view of that op (column 'tile vs golden' "
                  f"above), so no tile value is wrong. What moved is the "
                  f"C-1 RE-ENTRY of {'/'.join(lossy)}: the feeder is "
                  f"elaborated at CFG_DM={image.cfg_dm}, so golden's "
                  f"whole-row C-1 (ONE amax over "
                  f"{model.meta['D_model']}) comes back as independently "
                  f"scaled {image.cfg_dm}-wide frames (B-FEED-WIDTH). That "
                  f"is a DISCLOSED, MEASURED cost, not a defect — and here "
                  f"it is larger than a {margin['host_margin']:.3g} top-1 "
                  f"margin. The same substitution did NOT flip 7B's token "
                  f"(C2_PROMPT_ALL_OPS_RESULT.md); this prompt's margin is "
                  f"simply thinner. Closing it needs the wide feeder "
                  f"(IB_LAYER.md stage 6), not a host change.")
    if not ident and not bus_ident:
        print("  NOTE: the BUS_ON-only host run ALSO changes the token, so "
              "the composition mode (not the tile) is implicated.")
    print(f"  OP FAMILIES SERVED BY THE TILE: {len(served_ops)}/6 "
          f"({', '.join(served_ops) or 'none'})")
    print(f"  TOKEN IDENTITY: {'PASS' if ident else 'FAIL'}"
          f"   (offload-on == pure-host)")
    print(f"  P2 ITEM B     : {'PASS' if ok else 'FAIL'}")
    print("=" * W)
    print("  WHAT THE HOST STILL DID")
    for line in host_ledger(model, args, agg, image):
        print(f"    - {line}")
    print("=" * W)

    out = work / "layer05b_result.json"
    out.write_text(json.dumps({
        "prompt": args.prompt, "prompt_ids": ids, "mode": mode,
        "image": image.__dict__ | {"info_tier": image.info_tier},
        "d": args.d, "layer": args.layer, "step": step_target,
        "scope": scope.__dict__,
        "model": {k: model.meta.get(k) for k in
                  ("model", "H", "H_kv", "head_dim", "D_model", "d_ffn")},
        "tokens_on": on["ids"], "tokens_host": offr["ids"],
        "tokens_bus_on": bus["ids"],
        "text_on": txt(on["ids"]), "text_host": txt(offr["ids"]),
        "token_identity": ident, "bus_on_token_identity": bus_ident,
        "op_families_served": served_ops, "item_b_pass": ok,
        "geometry_audit": {"audit_passes": runner.n_audited,
                           "programs": len(runner.audited_keys),
                           "executed_distinct": len(runner.executed_keys),
                           "unaudited_executed": unaudited,
                           "desc_k_seen": sorted(runner.desc_k_seen),
                           "pass": geom_ok},
        "transport": lof.transport_record(runner, on),
        "info_tier_retargets": runner.n_retargeted,
        "norm_k_red": red,
        "poison": ({"k": args.poison, "ids": poison["ids"],
                    "max_abs_dlogit": dl} if poison is not None else None),
        "logit_geometry": margin,
        "ledger": agg,
        "checks": [{"name": n, "ok": o, "detail": str(d)}
                   for n, o, d in checks],
        "ops": [{"op": r.op, "name": r.name, "source": r.source,
                 "n_served": r.n_served, "n_total": r.n_total, "jobs": r.jobs,
                 "caps": r.caps, "seconds": r.seconds, "exact": r.exact,
                 "grade_ok": r.grade_ok, "notes": r.notes} for r in recs],
        "seconds": {"on": on["seconds"], "host": offr["seconds"],
                    "bus_on": bus["seconds"],
                    "executor": round(runner.wall, 1)},
        "tile_jobs": runner.n_jobs, "tile_caps": runner.n_caps,
        "git": rt.git_rev(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))
    print(f"  record -> {out}")
    return 0 if ok else 1


def _margin(host, on) -> dict:
    """The pure-host top-1 margin and how far the offloaded run moved things.

    A token flip means one of two very different things and the ledger has to
    tell them apart: EITHER a tile value is wrong (dlogit large, margin
    comfortable) OR the top-1 was decided by a gap smaller than the C-1
    reconstruction noise this ledger already printed as a measured cost
    (dlogit small, margin smaller). Both numbers, always, flip or no flip.
    """
    h = np.asarray(host["logits"], dtype=np.float64).ravel()
    o = np.asarray(on["logits"], dtype=np.float64).ravel()
    s = np.sort(h)[::-1]
    return {"host_margin": float(s[0] - s[1]),
            "host_range": float(h.max() - h.min()),
            "dlogit": float(np.max(np.abs(o - h))) if o.size == h.size
            else float("nan")}


def host_ledger(model, args, agg, image) -> list:
    """Exactly what stayed on the host, with THIS image's numbers.

    Not layer_offload's — that one derives the RMSNorm chunk-add count and
    the C-1 frame width from its own D_TILE=128 and would print 6 adds and a
    "frozen at 128" feeder for a run whose feeder is 64 and whose row took 13
    adds.
    """
    d = image.cfg_dm
    out = [
        "tokenizer, embedding lookup, the lm_head logits and the argmax "
        "(tie_word_embeddings: the head IS the embedding, handled at prepare)",
        f"every other decoder layer: {model.n_layers - 1} of "
        f"{model.n_layers} layers ran entirely on the host",
        "all data movement: weights, the KV rows, every operand staged into "
        "the tile and every value read back",
        "the C-2 requant epilogues (calib_requant + requant_i32_to_i8) of the "
        "attention output, the o-projection and the down-projection — the "
        "standing fence: the tile returns raw INT32, the host calibrates",
        "the attention path's score-dequant composites (s_q*s_k/sqrt(D), "
        f"exact at D={image.cfg_d}: sqrt(64) = 2^3 and the composite's 2^7 "
        "is exact — seq_walker_comp.sv:45) are host-computed and pushed as "
        "CSRs; the online softmax itself runs on the tile's ASU",
        f"the per-token whole-row C-1 quantizations that feed each op "
        f"(quant_rows_i8 over all {model.meta['D_model']} elements, ONE amax, "
        f"ONE scale) — the tile's C-1 feeder is elaborated at CFG_DM={d} "
        f"(seam_feeder_quant #(.D(CFG_DM)); legal 64/128 only), so it frames "
        f"the same values as {model.meta['D_model'] // d} x {d} rows with "
        f"{model.meta['D_model'] // d} independent scales",
    ]
    if agg["norm"]["values"] == 0 or agg["norm"]["instances"] > \
            agg["norm"]["served_instances"]:
        out.append("RMSNorm-1 (the pre-attention norm, T+1 rows) — same unit "
                   "as the served RMSNorm-2; host by default (--norm1-rows)")
    n_add = max(agg["norm"]["values"] // d - 1, 0)
    if n_add:
        out.append(f"the {n_add} INT32 adds that accumulate the RMSNorm "
                   f"chunk sums (golden rmsnorm_fx_wide's own step 2 — the "
                   f"tile exported all {n_add + 1} chunk sums and every one "
                   f"is graded)")
    if agg["rope"]["values"]:
        out.append("the RoPE of the CACHED K rows: the rotated K must reach "
                   "the KVQ codec as exact fp16 bits and the tile's q_sink "
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
    """A tile-legal tiny model with the 0.5B SHAPE INVARIANTS: head_dim 64
    (so the D=64 frames are real), GQA (H_kv < H), a D_model that is >= 2
    μ-chunks so the R4 arm is exercised, and a d_ffn that is a whole number
    of 64-column SwiGLU chunks."""
    return po._tiny_model(wdir, D=256, H=4, H_kv=2, hd=64, dff=256,
                          vocab=97, L=2)


def selftest(args) -> int:
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("LAYER05B SELFTEST (tiny head_dim=64 GQA model, no executor)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model = _tiny(td / "w")
        ids, work = [3, 1, 4, 1, 5], td / "work"
        tier, L, H = at.TIER_CQ8, model.n_layers, model.meta["H"]
        dm, dff = model.meta["D_model"], model.meta["d_ffn"]
        tstep = len(ids) - 1

        chk("the tiny model carries the 0.5B shape invariants "
            "(head_dim 64, GQA, >=2 mu-chunks, 64-col SwiGLU chunks)",
            model.meta["head_dim"] == 64 and model.meta["H_kv"] < H
            and dm % tg.RMS_MU_CHUNK == 0 and dm // tg.RMS_MU_CHUNK >= 2
            and dff % 64 == 0, f"D_model={dm} d_ffn={dff}")

        base = po.decode(model, ids, 1, tier, 128, set(), None, verbose=False)
        scope = lof.Scope(resid_slice=dm)
        runner = Runner05B("golden", work, d=64)
        with d64_emission(64):
            chk("inside the emission scope every frame width is 64",
                (lof.D_TILE, gl.D_TILE, gl.gj.D_TILE) == (64, 64, 64))
            chk("the frozen norm emitter is swapped for the re-derived one",
                gl.build_norm2_chunked is not _FROZEN_NORM)
            with Offloader05B(n_layers=L, layer=1, mode="golden", work=work,
                              scope=scope, runner=runner, step=tstep,
                              verbose=False) as off:
                on = po.decode(model, ids, 1, tier, 128, set(), None,
                               verbose=False)
        chk("every geometry rebind restored on exit",
            (lof.D_TILE, gl.D_TILE) == _FROZEN_FRAMES
            and gl.build_norm2_chunked is _FROZEN_NORM)
        chk("every golden rebind restored on exit",
            tf.attention_core is at.attention_core
            and tf.decoder_layer_fx.__module__.endswith(
                "apex_golden.transformer"))
        chk("token identity (dry-run offload vs pure golden)",
            on["ids"] == base["ids"], f"{on['ids']} vs {base['ids']}")
        got = {r.op for r in off.records if r.n_served}
        chk("all six op families served at head_dim=64",
            got == set(OP_TYPES), f"served: {sorted(got)}")
        chk("every substitution/consumption check passed",
            all(ok for _n, ok, _d in off.checks),
            f"{sum(1 for _n, ok, _d in off.checks if ok)}/{len(off.checks)}")
        chk("exact-egress families (proj, attn, resid) are bit-exact",
            all(r.exact for r in off.records
                if r.n_served and r.op in ("proj", "attn", "resid")))
        chk("C-1-view families (norm, swiglu) are reported as reconstructions",
            all(r.exact is False and "max_abs_delta" in str(r.detail)
                for r in off.records
                if r.n_served and r.op in ("norm", "swiglu")))
        chk("RoPE is served at head_dim 64 (it would be REFUSED as "
            "'head_dim != 128' without the frame rebind)",
            any(r.op == "rope" and r.n_served for r in off.records))
        by = {op: lof._agg(off.records, op) for op in OP_TYPES}
        chk("residual covers the FULL D_model row, twice, in ONE window each",
            by["resid"]["values"] == 2 * dm
            and all(d.get("slices") == 1 for d in by["resid"]["detail"]),
            f"{by['resid']['values']} of {2 * dm}")
        chk("RMSNorm-2 covers the FULL row as dm/64 C-1 frames",
            by["norm"]["values"] == dm
            and by["norm"]["detail"][0]["rows"] == dm // 64,
            str(by["norm"]["detail"][0].get("rows")))
        chk("SwiGLU covers the FULL d_ffn", by["swiglu"]["values"] == dff)
        chk("attention + RoPE cover ALL heads",
            by["attn"]["served_instances"] == H
            and by["rope"]["served_instances"] == H)

        # the discriminator
        with d64_emission(64):
            with Offloader05B(n_layers=L, layer=1, mode="golden", work=work,
                              scope=scope, runner=runner, step=tstep,
                              poison=0.5, verbose=False):
                po_run = po.decode(model, ids, 1, tier, 128, set(), None,
                                   verbose=False)
        dl = float(np.max(np.abs(po_run["logits"] - on["logits"])))
        chk("poisoned tile values perturb the logits", dl > 0,
            f"max|dlogit|={dl:.4g}")

        # a head_dim mismatch must REFUSE, not silently run at the wrong frame
        m128 = po._tiny_model(td / "w128", D=256, H=2, H_kv=1, hd=128,
                              dff=256, vocab=97, L=2)
        with d64_emission(64):
            with Offloader05B(n_layers=2, layer=1, mode="golden", work=work,
                              scope=lof.Scope(resid_slice=256), runner=runner,
                              step=tstep, verbose=False):
                try:
                    po.decode(m128, ids, 1, tier, 128, set(), None,
                              verbose=False)
                    chk("a head_dim != frame model is REFUSED", False,
                        "no refusal")
                except SystemExit as e:
                    chk("a head_dim != frame model is REFUSED",
                        "REFUSE" in str(e), str(e)[:70] + "…")

        # the norm plan the driver will arm
        p = tg.norm2_plan(dm, 64)
        chk("the norm arm uses the μ index dm/128, not the streamed chunk "
            "count dm/64", p["ext_k"] == dm // 128 and p["rows"] == dm // 64,
            str(p))

    print("=" * 60)
    if fails:
        print(f"LAYER05B SELFTEST FAIL: {fails}")
        return 1
    print("LAYER05B SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="P2 ITEM B: serve every op family of one Qwen2.5-0.5B "
                    "decoder layer from the D=64 tile during a real prompt "
                    "run, and prove the emitted token is unchanged.")
    ap.add_argument("--prompt")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--offload-step", type=int, default=None,
                    help="decode step to offload (default: the last prefill "
                         "step, whose output reaches the emitted logits)")
    ap.add_argument("--d", type=int, default=D_TILE_05B,
                    help="tile row width (must equal the model's head_dim)")
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poison", type=float, default=None, metavar="K")
    ap.add_argument("--norm-k-red", action="store_true",
                    help="also run the RMSNorm with the FROZEN emitter's k "
                         "(dm/D) and report how far it lands from golden")
    ap.add_argument("--ops", default=",".join(OP_TYPES))
    ap.add_argument("--proj-cols", type=int, default=lof.MXE_N)
    ap.add_argument("--proj-rows", type=int, default=1)
    ap.add_argument("--heads", type=int, default=-1)
    ap.add_argument("--rope-heads", type=int, default=-1)
    ap.add_argument("--resid-cols", type=int, default=-1)
    ap.add_argument("--resid-slice", type=int, default=-1,
                    help="residual elements per tile job (-1 = the whole "
                         "D_model row: 896 <= RESID_WIN, so ONE window)")
    ap.add_argument("--norm-dm", type=int, default=-1)
    ap.add_argument("--norm1-rows", type=int, default=0)
    ap.add_argument("--swiglu-chunks", type=int, default=-1)
    ap.add_argument("--no-collapse", dest="collapse", action="store_false",
                    help="one executor invocation per op (the pre-collapse "
                         "transport) — the A/B baseline, same job set")
    ap.add_argument("--max-batch-mb", type=int,
                    default=lof.MAX_BATCH_MB_DEFAULT,
                    help="regops per executor invocation when collapsing")
    ap.add_argument("--max-passes", type=int, default=lof.MAX_PASSES_DEFAULT,
                    help="replays of a layer allowed before the harness "
                         "REFUSES for lack of a stable program set")
    ap.add_argument("--tier", default="kvq8", choices=list(rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tile-div", type=int, default=bridge.TILE_DIV_DEFAULT)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=7200)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args)
    if args.prompt is None and not args.ids:
        ap.error("need --prompt (or --ids for a tokenizer-free run)")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be >= 1")
    bad = [o for o in args.ops.split(",") if o and o not in OP_TYPES]
    if bad:
        ap.error(f"unknown op family/families {bad}; choose from {OP_TYPES}")
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
