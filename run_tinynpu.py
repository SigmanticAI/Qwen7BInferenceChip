#!/usr/bin/env python3
"""run_tinynpu.py — S8 GOLDEN-7B-TOKEN: stream Qwen2.5-7B tokens through the
golden fixed-point APEX pipeline (the strongest honest 7B artifact).

    python run_tinynpu.py --prepare                       # one-time (needs mlx)
    python run_tinynpu.py --prompt "..." --max-new-tokens 4 \
        --trace-dir docs/results/s8_7b_token/trace --trace-jobs 8
    python run_tinynpu.py --verify-trace <dir>            # bit-exact replay
    python run_tinynpu.py --selftest                      # tiny model, no MLX

WHAT RUNS WHERE (honesty contract):
  * Every decoder layer is golden decoder_layer_fx — the S6-gated fixed-point
    graph (C-RMSW wide RMSNorm, C-KSPLIT INT8 GEMMs, RoPE/SiLU LUTs, KVQ
    codec, fixed-point online softmax, C-2 epilogues, fp16 residual bus).
    HF/Qwen SELF-INCLUSIVE decode (token t attends to keys 0..t, q at RoPE
    position t) uses the S8 q_pos composition gated by plumbing7b section G.
  * Prefill is NOT batched: every prompt token is a full M=1 decode step —
    exactly the tile's host-sequenced job contract (perf-model semantics).
  * Final norm runs golden rmsnorm_fx_wide; the LM head runs gemm_i8_ksplit
    INT8×INT8 with host INT32→real dequant (raw-accumulator read, S-2 style).
    Token pick is greedy argmax — bit-deterministic end to end.

WEIGHTS (disclosed double quantization):
  Source = the S5-cached MLX 4-bit checkpoint (mlx-community/Qwen2.5-7B-4bit;
  fp16 does not fit this 18 GB machine). --prepare dequantizes tensor-by-
  tensor and re-quantizes PER-TENSOR symmetric INT8 (the only granularity the
  frozen LayerWeights/C-2 contract carries). So weights are 4-bit-then-INT8;
  activations/KV follow the golden contracts exactly. Norm gammas quantize to
  Q2.13; if a gamma exceeds the Q2.13 range a power-of-2 pre-scale is folded
  out and compensated EXACTLY through the downstream per-tensor weight scales
  (host calibration data, no golden change; folds recorded in meta.json).

T ENVELOPE: context T ≤ 128 (T_ROW_MAX per-job attention envelope). Total
tokens (prompt + generated) ≤ 128 until C-CHUNK is wired into the layer
composition (S8 phase 2).

RTL REPLAY TRACE: --trace-dir samples (step, layer, head) attention jobs and
dumps EXACT tile-boundary stimulus + every golden intermediate:
  inputs : q row (fp16 bits, post-RoPE), K/V cache rows (fp16 bits, K
           post-RoPE — the KVQ write-port values), tier, G, outlier_idx.
  outputs: codec blob scales/payloads, feeder codes+scales (k8/s_k, v8/s_v,
           q8/s_q), INT32 score accumulators, S-3 score_fx, ASU p/mmax/lsum,
           c8/s_c, INT32 P·V̂ acc, C-2 (scale,shift), o8, s_out, out_hat.
  attention_core is a pure function of the inputs, so --verify-trace re-runs
  it and asserts EVERY stored array bit-exact. The same records drive RTL
  replay: verif/top/l3 named-tensor injection (arbitrary fp16 K/V/q through
  the real tile) and verif/kvq TBs (K/V rows → record/decode checks). Note
  the group size G is a build/CSR parameter on the RTL side — replay with a
  build matching the traced G (default here: 128, the S4/S5 cache-at-rest
  convention), or re-trace with --group matching the build.

ENV: generation needs numpy (+ transformers/huggingface_hub for the
tokenizer); --prepare additionally needs mlx (use ~/.venvs/apex-eval).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "golden"))

from apex_golden import attention as at                              # noqa: E402
from apex_golden import compute as cp                                # noqa: E402
from apex_golden import transformer as tf                            # noqa: E402
from apex_golden.cq_codec import KeyBlob, ValueBlob                  # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits, rne     # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-4bit"
TIER_MAP = {"kvq8": at.TIER_CQ8, "kvq4": at.TIER_CQ4}
T_SESSION_MAX = 384                  # C-CHUNK D-023-gated range (plumbing §C/§H)
Q213_MAX = 32767
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj")


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True,
                              text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# ═══════════════════════ weight preparation (host calibration) ═══════════════

def quant_w_i8(W: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8: s = amax/127 (float64), w8 = RNE, ±127."""
    W = np.asarray(W, dtype=np.float64)
    amax = float(np.max(np.abs(W), initial=0.0))
    s = amax / 127.0 if amax > 0 else 1.0
    w8 = np.clip(rne(W / s), -127, 127).astype(np.int8)
    return w8, s


def quant_gamma_q213(g: np.ndarray) -> tuple[np.ndarray, int, int]:
    """RMSNorm gain → Q2.13 int16 with a pow2 pre-scale fold (see module doc).

    Returns (g_q213 int16, fold_k, n_clamped). fold_k ≥ 0: the stored gamma
    is g/2^fold_k; the CALLER must multiply the scales of every weight that
    consumes this norm's output by 2^fold_k (exact float64 compensation).
    """
    g = np.asarray(g, dtype=np.float64)
    amax = float(np.max(np.abs(g), initial=0.0))
    k = 0
    while amax * (1 << 13) / (1 << k) > Q213_MAX:
        k += 1
    q = rne(g / (1 << k) * (1 << 13))
    n_cl = int(np.sum((q > Q213_MAX) | (q < -Q213_MAX - 1)))
    return np.clip(q, -Q213_MAX - 1, Q213_MAX).astype(np.int16), k, n_cl


def prepare(args) -> None:
    """Stream the MLX 4-bit checkpoint → golden weight cache (mmap .npy)."""
    import mlx.core as mx
    from huggingface_hub import snapshot_download

    t0 = time.time()
    src = Path(snapshot_download(args.model, local_files_only=True))
    cfg = json.loads((src / "config.json").read_text())
    qz = cfg.get("quantization") or {}
    gs, qbits = qz.get("group_size", 64), qz.get("bits", 4)
    n_layers = cfg["num_hidden_layers"]
    outdir = Path(args.weights_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    eprint(f"[prepare] {args.model} → {outdir}  (mlx q{qbits}/g{gs}, "
           f"{n_layers} layers)")

    weights: dict = {}
    for f in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(f)))

    def dq(name: str) -> np.ndarray:
        w = weights[f"{name}.weight"]
        if f"{name}.scales" in weights:
            w = mx.dequantize(w, weights[f"{name}.scales"],
                              weights[f"{name}.biases"], gs, qbits)
        return np.array(w.astype(mx.float32))

    meta = {
        "model": args.model, "format": "apex-s8-golden-weights-v1",
        "D_model": cfg["hidden_size"], "H": cfg["num_attention_heads"],
        "H_kv": cfg["num_key_value_heads"],
        "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
        "d_ffn": cfg["intermediate_size"], "n_layers": n_layers,
        "vocab": cfg["vocab_size"], "rope_theta": cfg["rope_theta"],
        "eos_token_id": cfg.get("eos_token_id"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
        "source_quant": {"bits": qbits, "group_size": gs},
        "disclosure": "weights = MLX 4-bit dequant → PER-TENSOR symmetric "
                      "INT8 (double quantization; the golden LayerWeights/C-2 "
                      "contract carries one scale per tensor). rms_norm_eps "
                      f"config={cfg.get('rms_norm_eps')} vs golden EPS_INT=1 "
                      "fixed-point contract (disclosed, gated).",
        "layers": [], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_rev(),
    }

    # embedding (stored [vocab, D] float16 — exact fp16-bus token inputs)
    emb = dq("model.embed_tokens").astype(np.float16)
    np.save(outdir / "embed.npy", emb)
    eprint(f"[prepare] embed {emb.shape} saved ({emb.nbytes/1e6:.0f} MB)")

    # final norm + LM head (head stored [vocab, D] int8, native row-major)
    g_fin = np.array(weights["model.norm.weight"].astype(mx.float32))
    gq, kf, ncl = quant_gamma_q213(g_fin)
    np.save(outdir / "final_gamma.npy", gq)
    meta["final_gamma"] = {"fold_k": kf, "clamped": ncl,
                           "amax": float(np.max(np.abs(g_fin)))}
    head_name = ("model.embed_tokens" if meta["tie_word_embeddings"]
                 else "lm_head")
    h8, s_head = quant_w_i8(dq(head_name))
    np.save(outdir / "lm_head_w8.npy", h8)
    meta["s_head"] = s_head
    eprint(f"[prepare] head {h8.shape} s={s_head:.3e}  final-gamma "
           f"amax={meta['final_gamma']['amax']:.3f} fold_k={kf} clamp={ncl}")

    hf_names = {"Wq": "self_attn.q_proj", "Wk": "self_attn.k_proj",
                "Wv": "self_attn.v_proj", "Wo": "self_attn.o_proj",
                "Wg": "mlp.gate_proj", "Wu": "mlp.up_proj",
                "Wd": "mlp.down_proj"}
    for li in range(n_layers):
        base = f"model.layers.{li}"
        lrec = {}
        for tag, nm in hf_names.items():
            w8, s = quant_w_i8(dq(f"{base}.{nm}").T)   # → [in_dim, out_dim]
            np.save(outdir / f"L{li:02d}_{tag}.npy",
                    np.ascontiguousarray(w8))
            lrec[f"s_{tag}"] = s
        for tag, nm in (("gamma1", "input_layernorm"),
                        ("gamma2", "post_attention_layernorm")):
            g = np.array(weights[f"{base}.{nm}.weight"].astype(mx.float32))
            gq, k, ncl = quant_gamma_q213(g)
            np.save(outdir / f"L{li:02d}_{tag}.npy", gq)
            lrec[f"{tag}_fold_k"] = k
            lrec[f"{tag}_clamped"] = ncl
            lrec[f"{tag}_amax"] = float(np.max(np.abs(g)))
        for tag, nm in (("bq", "self_attn.q_proj"),
                        ("bk", "self_attn.k_proj"),
                        ("bv", "self_attn.v_proj")):
            bname = f"{base}.{nm}.bias"
            if bname in weights:
                np.save(outdir / f"L{li:02d}_{tag}.npy",
                        np.array(weights[bname].astype(mx.float32))
                        .astype(np.float16))
                lrec[f"has_{tag}"] = True
        meta["layers"].append(lrec)
        mx.clear_cache()                     # release Metal buffer cache
        eprint(f"[prepare] layer {li:2d}/{n_layers}  "
               f"g1(amax={lrec['gamma1_amax']:.2f},k={lrec['gamma1_fold_k']}) "
               f"g2(amax={lrec['gamma2_amax']:.2f},k={lrec['gamma2_fold_k']})")

    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    total = sum(f.stat().st_size for f in outdir.glob("*.npy"))
    eprint(f"[prepare] DONE in {time.time()-t0:.0f}s — {total/1e9:.2f} GB "
           f"under {outdir}")


# ═══════════════════════ model load + decode step ════════════════════════════

class GoldenModel:
    """mmap-backed golden weight set + per-layer decode-step state."""

    def __init__(self, wdir: Path):
        self.dir = Path(wdir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        m = self.meta
        self.n_layers = m["n_layers"]
        self.layers: list[tf.LayerWeights] = []
        for li, lr in enumerate(m["layers"]):
            def ld(tag, req=True):
                p = self.dir / f"L{li:02d}_{tag}.npy"
                return np.load(p, mmap_mode="r") if (req or p.exists()) \
                    else None
            k1 = 1 << lr["gamma1_fold_k"]        # exact pow2 compensation of
            k2 = 1 << lr["gamma2_fold_k"]        # the Q2.13 gamma pre-scale
            self.layers.append(tf.LayerWeights(
                Wq=ld("Wq"), Wk=ld("Wk"), Wv=ld("Wv"), Wo=ld("Wo"),
                Wg=ld("Wg"), Wu=ld("Wu"), Wd=ld("Wd"),
                s_wq=lr["s_Wq"] * k1, s_wk=lr["s_Wk"] * k1,
                s_wv=lr["s_Wv"] * k1, s_wo=lr["s_Wo"],
                s_wg=lr["s_Wg"] * k2, s_wu=lr["s_Wu"] * k2, s_wd=lr["s_Wd"],
                gamma1=np.asarray(ld("gamma1"), dtype=np.int64),
                gamma2=np.asarray(ld("gamma2"), dtype=np.int64),
                H=m["H"], head_dim=m["head_dim"], H_kv=m["H_kv"],
                bq=(None if not lr.get("has_bq") else
                    np.asarray(ld("bq", False), dtype=np.float64)),
                bk=(None if not lr.get("has_bk") else
                    np.asarray(ld("bk", False), dtype=np.float64)),
                bv=(None if not lr.get("has_bv") else
                    np.asarray(ld("bv", False), dtype=np.float64)),
                rope_theta_base=float(m["rope_theta"]),
            ))
        self.embed = np.load(self.dir / "embed.npy", mmap_mode="r")
        self.final_gamma = [int(v)
                            for v in np.load(self.dir / "final_gamma.npy")]
        self.final_fold = 1 << m["final_gamma"]["fold_k"]
        self.head_w8 = np.load(self.dir / "lm_head_w8.npy", mmap_mode="r")
        self.s_head = m["s_head"]

    def embed_row(self, tok: int) -> np.ndarray:
        return np.asarray(self.embed[tok], dtype=np.float64)  # fp16-grid

    def head_logits(self, y: np.ndarray) -> np.ndarray:
        """Final rmsnorm_fx_wide + INT8 LM head (raw-acc host dequant)."""
        x8, _ = at.quant_rows_i8(y[None, :])
        h, _, _ = at.rmsnorm_fx_wide([int(v) for v in x8[0]],
                                     self.final_gamma)
        h8, s_h = at.quant_rows_i8(np.asarray(h, np.float64)[None, :] / 256.0)
        s_h_val = float(f16_bits_to_f64(np.array([s_h[0]]))[0])
        V, vocab = self.head_w8, self.head_w8.shape[0]
        out = np.empty(vocab, dtype=np.float64)
        step = cp.DIM_MAX                        # 12-bit M descriptor field
        # The M-chunks are materialized ONCE as id-stable int8 arrays (byte-
        # identical slices of head_w8) so gemm_i8_ksplit's exact host-speed
        # path can cache their float64 mirrors across tokens instead of
        # re-widening the whole head every decode step. Same call structure,
        # same operand values, same accumulators — bit-identical.
        if getattr(self, "_head_chunks", None) is None:
            chunks = [np.ascontiguousarray(V[a:min(a + step, vocab)])
                      for a in range(0, vocab, step)]
            for c in chunks:
                c.flags.writeable = False        # immutable -> mirror-cached
            self._head_chunks = chunks
        col = np.asarray(h8[0], dtype=np.int64)[:, None]
        for i, a in enumerate(range(0, vocab, step)):
            b = min(a + step, vocab)
            acc = cp.gemm_i8_ksplit(self._head_chunks[i], col)
            out[a:b] = acc[:, 0].astype(np.float64)
        return out * (s_h_val * self.s_head * self.final_fold)


class Session:
    """Streaming decode state: per-layer input-row history (fp16-grid f64)."""

    def __init__(self, model: GoldenModel, tier: str, group: int):
        self.m = model
        self.tier = tier
        self.group = group
        self.hist: list[list[np.ndarray]] = [[] for _ in model.layers]
        # per-layer host-speed history state (decoder_layer_fx `state`):
        # caches the per-row norm-1/projection/RoPE-K values of the
        # append-only row history — bit-identical reuse, see transformer.py.
        self._lstate: list[dict] = [{} for _ in model.layers]
        self.pos = 0
        self.silu_clamp = 0                      # |gate| > 8 events (C-SILU)
        self.silu_total = 0

    def step(self, x_row: np.ndarray, collect=None,
             ref_out: dict | None = None) -> np.ndarray:
        """One full decode step (all layers) for the token at self.pos.

        HF self-inclusive composition (plumbing7b section G): the current
        token is context row t AND the duplicated decode row, q_pos = t.
        """
        t = self.pos
        assert t + 1 <= T_SESSION_MAX, \
            f"T={t+1} exceeds the C-CHUNK D-023-gated range " \
            f"T_SESSION_MAX={T_SESSION_MAX} (plumbing7b §C/§H)"
        x = x_row
        for li, w in enumerate(self.m.layers):
            self.hist[li].append(x)
            X = np.vstack(self.hist[li] + [x])   # [t+2, D]: rows 0..t + dup
            r = tf.decoder_layer_fx(X, w, self.tier, G=self.group, q_pos=t,
                                    state=self._lstate[li])
            self.silu_clamp += int(np.sum(np.abs(r.gate_real) > 8.0))
            self.silu_total += r.gate_real.size
            if collect is not None:
                collect(li, r)
            if ref_out is not None:
                ref = tf.decoder_layer_ref(X, w, q_pos=t)
                den = float(np.max(np.abs(ref.y_ref))) or 1.0
                ref_out.setdefault("layers", []).append(
                    float(np.max(np.abs(r.r2 - ref.y_ref))) / den)
            x = r.r2
        self.pos += 1
        return x


# ═══════════════════════ RTL-replay trace ════════════════════════════════════

def _blob_arrays(prefix: str, blob) -> dict:
    d = {f"{prefix}_scales": np.asarray(blob.scales, dtype=np.uint16),
         f"{prefix}_payload": np.asarray(blob.payload, dtype=np.uint8),
         f"{prefix}_bits": np.int64(blob.bits)}
    if isinstance(blob, KeyBlob):
        d[f"{prefix}_kind"] = np.int64(1)
        d[f"{prefix}_G"] = np.int64(blob.G)
        d[f"{prefix}_keep"] = np.asarray(blob.keep, dtype=np.int64)
        d[f"{prefix}_outlier"] = np.asarray(blob.outlier, dtype=np.int64)
        d[f"{prefix}_sidecar"] = np.asarray(blob.sidecar,
                                            dtype=np.uint16).reshape(blob.T, -1)
    else:
        assert isinstance(blob, ValueBlob)
        d[f"{prefix}_kind"] = np.int64(0)
    return d


def core_record(core: at.AttnCore, meta: dict) -> dict:
    """Everything needed to replay one attention/KVQ job bit-exact."""
    q_bits = f64_to_f16_bits(core.q).astype(np.uint16)
    rec = {
        "q_f16": q_bits,
        "K_f16": np.asarray(core.K_f16, dtype=np.uint16),
        "V_f16": np.asarray(core.V_f16, dtype=np.uint16),
        "k8": core.k8.astype(np.int8), "s_k": core.s_k,
        "v8": core.v8.astype(np.int8), "s_v": core.s_v,
        "q8": core.q8.astype(np.int8), "s_q": np.uint16(core.s_q),
        "acc_s": core.acc_s.astype(np.int64),
        "score_fx": core.score_fx.astype(np.int64),
        "p_q115": core.p_q115.astype(np.int64),
        "sm_m": np.int64(core.sm_m), "sm_l": np.int64(core.sm_l),
        "c8": core.c8.astype(np.int8), "s_c": np.uint16(core.s_c),
        "acc_o": core.acc_o.astype(np.int64),
        "rq_scale": np.int64(core.rq[0]), "rq_shift": np.int64(core.rq[1]),
        "o8": core.o8.astype(np.int8),
        "s_out": np.float64(core.s_out),
        "out_hat": core.out_hat.astype(np.float64),
        "meta_json": json.dumps({**meta, "tier": core.tier, "D": core.D,
                                 "T": core.T, "G": core.G,
                                 "score_frac": at.SCORE_FRAC}),
    }
    rec.update(_blob_arrays("kb", core.kb))
    rec.update(_blob_arrays("vb", core.vb))
    return rec


def job_digest(rec: dict) -> str:
    h = hashlib.sha256()
    for k in ("q_f16", "K_f16", "V_f16"):
        h.update(rec[k].tobytes())
    h.update(rec["meta_json"].encode())
    return h.hexdigest()[:16]


class Tracer:
    def __init__(self, outdir: Path, n_jobs: int, max_steps: int,
                 n_layers: int, n_heads: int, seed: int):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        grid = max_steps * n_layers * n_heads
        pick = rng.choice(grid, size=min(n_jobs, grid), replace=False)
        self.targets: dict[tuple[int, int], list[int]] = {}
        for p in sorted(int(v) for v in pick):
            s, rem = divmod(p, n_layers * n_heads)
            l, h = divmod(rem, n_heads)
            self.targets.setdefault((s, l), []).append(h)
        self.seed = seed
        self.jobs: list[dict] = []

    def collect(self, step: int, layer: int, r: tf.LayerFx) -> None:
        for h in self.targets.get((step, layer), []):
            head = r.heads[h]
            meta = {"step": step, "layer": layer, "head": h, "q_pos": step,
                    "kv_head": h // (r.H // r.H_kv)}
            if isinstance(head, tf.ChunkedHead):
                # C-CHUNK: each per-chunk tile job is its own replayable
                # record; merge state travels in every chunk's meta.
                cores = [(ci, c, {**meta, "chunk": ci,
                                  "chunk_base": ci * at.CHUNK_T_MAX,
                                  "n_chunks": len(head.cores),
                                  "merge_w": head.merge["w"],
                                  "merge_m_star": head.merge["m_star"],
                                  "merge_l_star": head.merge["l_star"]})
                         for ci, c in enumerate(head.cores)]
            else:
                cores = [(None, head, meta)]
            for ci, core, m in cores:
                rec = core_record(core, m)
                name = (f"job_s{step:03d}_L{layer:02d}_h{h:02d}"
                        + (f"_c{ci}" if ci is not None else "") + ".npz")
                np.savez_compressed(self.dir / name, **rec)
                self.jobs.append({"file": name, "step": step, "layer": layer,
                                  "head": h, "T": core.T, "tier": core.tier,
                                  "chunk": ci, "digest": job_digest(rec)})

    def finalize(self, run_info: dict) -> None:
        (self.dir / "manifest.json").write_text(json.dumps({
            "seed": self.seed, "jobs": self.jobs, "run": run_info,
            "replay": "inputs (q_f16/K_f16/V_f16 fp16 bits + tier/G) fully "
                      "determine the job; --verify-trace re-runs golden "
                      "attention_core and asserts every stored array "
                      "bit-exact. RTL: verif/top/l3 named-tensor injection "
                      "or verif/kvq TBs with a build matching D/tier/G.",
        }, indent=1))
        eprint(f"[trace] {len(self.jobs)} jobs + manifest → {self.dir}")


def verify_trace(tdir: Path) -> int:
    tdir = Path(tdir)
    man = json.loads((tdir / "manifest.json").read_text())
    fails = 0
    for j in man["jobs"]:
        z = np.load(tdir / j["file"])
        meta = json.loads(str(z["meta_json"]))
        core = at.attention_core(
            f16_bits_to_f64(z["q_f16"]),
            z["K_f16"].astype(np.uint16), z["V_f16"].astype(np.uint16),
            meta["tier"], G=meta["G"],
            outlier_idx=list(z["kb_outlier"]) if "kb_outlier" in z else [],
            score_frac=meta["score_frac"])
        rec = core_record(core, {k: v for k, v in meta.items()
                                 if k not in ("tier", "D", "T", "G",
                                              "score_frac")})
        bad = [k for k in rec if k != "meta_json"
               and not np.array_equal(np.asarray(rec[k]), np.asarray(z[k]))]
        ok = not bad and job_digest(rec) == j["digest"]
        print(f"  {'PASS' if ok else 'FAIL'}  {j['file']}  T={meta['T']} "
              f"{meta['tier']}" + (f"  mismatch: {bad}" if bad else ""))
        fails += 0 if ok else 1
    n = len(man["jobs"])
    print(f"TRACE VERIFY: {n - fails}/{n} jobs replay bit-exact"
          + (" — FAIL" if fails else ""))
    return fails


# ═══════════════════════ generation ══════════════════════════════════════════

def load_tokenizer(model_id: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        snapshot_download(model_id, local_files_only=True))


def run_generate(args) -> int:
    tier = TIER_MAP[args.tier]
    model = GoldenModel(Path(args.weights_dir))
    tok = load_tokenizer(model.meta["model"])
    ids = tok.encode(args.prompt)
    n_total = len(ids) + args.max_new_tokens
    if n_total > T_SESSION_MAX:
        sys.exit(f"ABORT: prompt({len(ids)}) + max_new({args.max_new_tokens})"
                 f" = {n_total} > T_SESSION_MAX={T_SESSION_MAX} "
                 "C-CHUNK D-023-gated range (plumbing7b §C/§H)")
    eos = model.meta.get("eos_token_id")
    eos_set = set(eos if isinstance(eos, list) else [eos]) - {None}

    tracer = None
    if args.trace_dir:
        tracer = Tracer(Path(args.trace_dir), args.trace_jobs, n_total,
                        model.n_layers, model.meta["H"], args.trace_seed)
    sess = Session(model, tier, args.group)
    eprint(f"[run] {model.meta['model']} tier={tier} G={args.group} "
           f"prompt={len(ids)} tok, max_new={args.max_new_tokens}, "
           f"layers={model.n_layers}")

    timings, ref_checks = [], []
    hidden = None
    gen_ids: list[int] = []
    out_json = Path(args.trace_dir or ".") / "run.json"

    def snapshot(final: bool = False) -> dict:
        info = {
            "model": model.meta["model"],
            "weights_dir": str(args.weights_dir),
            "tier": tier, "group": args.group, "git": git_rev(),
            "prompt": args.prompt, "prompt_ids": ids,
            "generated_ids": gen_ids,
            "generated_text": tok.decode(gen_ids),
            "complete": final,
            "per_token_seconds": [round(t, 2) for t in timings],
            "silu_domain_exceed": {"events": sess.silu_clamp,
                                   "total": sess.silu_total,
                                   "frac": sess.silu_clamp
                                   / max(1, sess.silu_total)},
            "ref_checks": ref_checks,
            "weights_disclosure": model.meta["disclosure"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        out_json.write_text(json.dumps(info, indent=1))
        return info

    for phase in ("prefill", "decode"):
        n = len(ids) if phase == "prefill" else args.max_new_tokens
        for i in range(n):
            if phase == "prefill":
                tok_id = ids[i]
            else:
                logits = model.head_logits(hidden)
                tok_id = int(np.argmax(logits))
                gen_ids.append(tok_id)
                piece = tok.decode(gen_ids)
                sys.stdout.write(piece[len(tok.decode(gen_ids[:-1])):]
                                 if len(gen_ids) > 1 else piece)
                sys.stdout.flush()
                if tok_id in eos_set:
                    eprint("\n[run] EOS")
                    break
                if i == n - 1:
                    break                    # last token emitted; skip the
                                             # never-consumed final step
            t0 = time.perf_counter()
            step = sess.pos
            coll = ((lambda li, r: tracer.collect(step, li, r))
                    if tracer else None)
            ref = {} if (step < args.ref_check
                         or (args.ref_every
                             and step % args.ref_every == 0)) else None
            hidden = sess.step(model.embed_row(tok_id), collect=coll,
                               ref_out=ref)
            dt = time.perf_counter() - t0
            timings.append(dt)
            if ref:
                ref_checks.append({"step": step,
                                   "per_layer_e2e_rel": ref["layers"]})
                eprint(f"[ref] step {step}: worst layer e2e rel = "
                       f"{max(ref['layers']):.4f}")
            eprint(f"[{phase} {i+1}/{n}] pos={step} T={step+1} "
                   f"tok={tok_id} {dt:.1f}s")
            snapshot()                           # incremental run record
        else:
            continue
        break
    print()

    run_info = snapshot(final=True)
    if tracer:
        tracer.finalize(run_info)
    eprint(f"[run] DONE — {len(gen_ids)} tokens, "
           f"{sum(timings):.0f}s total, run record → {out_json}")
    eprint(f"[run] SiLU domain exceedances: {sess.silu_clamp}/"
           f"{sess.silu_total} ({run_info['silu_domain_exceed']['frac']:.2e})")
    return 0


# ═══════════════════════ selftest (tiny model, no MLX/HF) ════════════════════

def selftest(args) -> int:
    import tempfile
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    rng = np.random.default_rng(0x58E1F)
    D, H, H_kv, hd, dff, vocab, L = 128, 2, 1, 64, 256, 97, 2
    with tempfile.TemporaryDirectory() as td:
        wdir = Path(td) / "w"
        wdir.mkdir()
        meta = {"model": "selftest-tiny", "format": "apex-s8-golden-weights-v1",
                "D_model": D, "H": H, "H_kv": H_kv, "head_dim": hd,
                "d_ffn": dff, "n_layers": L, "vocab": vocab,
                "rope_theta": 1e6, "eos_token_id": None,
                "tie_word_embeddings": False, "source_quant": None,
                "disclosure": "selftest random weights", "layers": [],
                "created": "-", "git": git_rev()}
        for li in range(L):
            lrec = {}
            shapes = {"Wq": (D, D), "Wk": (D, H_kv * hd), "Wv": (D, H_kv * hd),
                      "Wo": (D, D), "Wg": (D, dff), "Wu": (D, dff),
                      "Wd": (dff, D)}
            for tag, sh in shapes.items():
                w8, s = quant_w_i8(rng.normal(0, 1, sh)
                                   / np.sqrt(sh[0]))
                np.save(wdir / f"L{li:02d}_{tag}.npy", w8)
                lrec[f"s_{tag}"] = s
            for tag in ("gamma1", "gamma2"):
                gq, k, ncl = quant_gamma_q213(0.7 + 0.6 * rng.random(D))
                np.save(wdir / f"L{li:02d}_{tag}.npy", gq)
                lrec.update({f"{tag}_fold_k": k, f"{tag}_clamped": ncl,
                             f"{tag}_amax": 0.0})
            for tag, n in (("bq", D), ("bk", H_kv * hd), ("bv", H_kv * hd)):
                np.save(wdir / f"L{li:02d}_{tag}.npy",
                        rng.normal(0, 0.1, n).astype(np.float16))
                lrec[f"has_{tag}"] = True
            meta["layers"].append(lrec)
        gq, kf, _ = quant_gamma_q213(np.full(D, 5.5))   # exercises the fold
        np.save(wdir / "final_gamma.npy", gq)
        meta["final_gamma"] = {"fold_k": kf, "clamped": 0, "amax": 5.5}
        check("gamma fold engages (amax 5.5 → k=1)", kf == 1, f"k={kf}")
        h8, s_head = quant_w_i8(rng.normal(0, 0.5, (vocab, D)))
        np.save(wdir / "lm_head_w8.npy", h8)
        meta["s_head"] = s_head
        np.save(wdir / "embed.npy",
                rng.normal(0, 1, (vocab, D)).astype(np.float16))
        (wdir / "meta.json").write_text(json.dumps(meta))

        model = GoldenModel(wdir)
        ids = [3, 1, 4, 1, 5]
        n_new = 3

        def run_once(tdir):
            tracer = Tracer(tdir, 6, len(ids) + n_new, L, H, seed=7)
            sess = Session(model, at.TIER_CQ8, 128)
            hidden, gen = None, []
            for tok_id in ids:
                hidden = sess.step(model.embed_row(tok_id),
                                   collect=lambda li, r, s=sess.pos:
                                   tracer.collect(s, li, r))
            for _ in range(n_new):
                nxt = int(np.argmax(model.head_logits(hidden)))
                gen.append(nxt)
                hidden = sess.step(model.embed_row(nxt),
                                   collect=lambda li, r, s=sess.pos:
                                   tracer.collect(s, li, r))
            tracer.finalize({"selftest": True})
            return gen, tracer

        gen1, tr1 = run_once(Path(td) / "t1")
        gen2, tr2 = run_once(Path(td) / "t2")
        check("greedy decode deterministic (two runs identical)",
              gen1 == gen2, f"gen={gen1}")
        check("trace digests deterministic",
              [j["digest"] for j in tr1.jobs] ==
              [j["digest"] for j in tr2.jobs])
        check("self-inclusive T: traced job at step s has T = s+1",
              all(j["T"] == j["step"] + 1 for j in tr1.jobs))
        fails_v = verify_trace(Path(td) / "t1")
        check("verify-trace replays bit-exact", fails_v == 0)

        # C-CHUNK crossing: fabricated history to T=130 → 2-chunk heads,
        # chunk trace records replay bit-exact
        sess = Session(model, at.TIER_CQ8, 128)
        rows = [tf._f16(v) for v in rng.normal(0, 1, (129, D))]
        for li in range(L):
            sess.hist[li] = list(rows)           # per-layer fp16-grid history
        sess.pos = 129
        tr3 = Tracer(Path(td) / "t3", 0, 1, L, H, seed=1)
        tr3.targets = {(129, 0): [0]}            # force one traced head
        sess.step(model.embed_row(7),
                  collect=lambda li, r: tr3.collect(129, li, r))
        tr3.finalize({"selftest": "chunk-crossing"})
        check("T=130 head traces as 2 chunk jobs",
              [j["chunk"] for j in tr3.jobs] == [0, 1]
              and [j["T"] for j in tr3.jobs] == [128, 2])
        check("chunked trace replays bit-exact",
              verify_trace(Path(td) / "t3") == 0)

        # session envelope guard (chunk-gated range)
        sess = Session(model, at.TIER_CQ8, 128)
        sess.pos = T_SESSION_MAX
        try:
            sess.step(model.embed_row(0))
            check("T_SESSION_MAX guard raises", False)
        except AssertionError:
            check("T_SESSION_MAX guard raises", True)

        # ref cross-check on the tiny model (health metric wiring)
        sess = Session(model, at.TIER_CQ8, 128)
        ref = {}
        sess.step(model.embed_row(3), ref_out=ref)
        worst = max(ref["layers"])
        check("ref-check wiring: layer e2e rel sane (< 0.05)",
              worst < 0.05, f"worst={worst:.4f}")

    print("=" * 60)
    if fails:
        print(f"SELFTEST FAIL: {fails}")
        return 1
    print("RUN_TINYNPU SELFTEST: ALL PASS")
    return 0


# ═══════════════════════ CLI ═════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--weights-dir",
                    default=str(REPO / "build/s8_weights/Qwen2.5-7B-4bit"))
    ap.add_argument("--prepare", action="store_true",
                    help="build the golden weight cache (needs mlx)")
    ap.add_argument("--prompt")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--tier", default="kvq8", choices=[*TIER_MAP, "kvq4p"])
    ap.add_argument("--group", type=int, default=128,
                    help="KVQ key-group size (S4/S5 cache-at-rest: 128)")
    ap.add_argument("--trace-dir", help="dump RTL-replayable job trace here")
    ap.add_argument("--trace-jobs", type=int, default=8)
    ap.add_argument("--trace-seed", type=int, default=0x58)
    ap.add_argument("--ref-check", type=int, default=0, metavar="N",
                    help="float64 yardstick per layer for the first N steps")
    ap.add_argument("--ref-every", type=int, default=0, metavar="K",
                    help="additionally yardstick every K-th step (late-T "
                         "health probe)")
    ap.add_argument("--verify-trace", metavar="DIR",
                    help="re-run golden on a trace, assert bit-exact")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.tier == "kvq4p":
        sys.exit("ABORT: kvq4p deferred — D=128 RTL build is maskless until "
                 "the loadable-mask task; use kvq8/kvq4.")
    if args.selftest:
        return selftest(args)
    if args.verify_trace:
        return 1 if verify_trace(Path(args.verify_trace)) else 0
    if args.prepare:
        prepare(args)
        return 0
    if args.prompt is None:
        sys.exit("need --prompt (or --prepare / --selftest / --verify-trace)")
    return run_generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
