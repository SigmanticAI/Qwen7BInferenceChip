#!/usr/bin/env python3
"""trace_layer_golden.py — W-G3 stage A: trace ONE REAL Qwen2.5-7B decoder
layer step out of the committed S8 run.

The S8 tracer (`run_tinynpu.py`, `Tracer.collect`) samples per-HEAD attention
jobs — inputs + every golden intermediate of one `attention_core`. W-G3 needs
the layer above that: the whole `decoder_layer_fx` step (all seven weight
tensors, both gammas, the RoPE phase row, the H+2 epilogue requant pairs and
every stage tap incl. r2) so the layer can be replayed loader->DDR->reader->
walker->tile.

ADDITIVE, per the W-G3 contract: `run_tinynpu.py` and `golden/` are NOT
edited. This script IMPORTS the S8 model/session machinery and re-runs the
committed token sequence, exactly the idiom `trace_to_regops.py` uses to
import `gen_l3_vectors`. The golden stays the one-directional arbiter.

PROVENANCE GATE (the reason this is trustworthy): the committed artifact
`docs/results/s8_7b_token/artifact_trace/job_s<step>_L<layer>_h<head>.npz`
holds the SAME head of the SAME (step, layer) of the SAME run. This script
asserts its re-derived per-head fields are BIT-EXACT vs that committed record
(q_f16/K_f16/V_f16/k8/v8/q8/s_k/s_v/s_q/acc_s/score_fx/c8/acc_o/rq/o8/s_out).
If the replay drifted by one token or one layer, those arrays would differ.

    python3 verif/walkgold/trace_layer_golden.py            # step 19, layer 19
    python3 verif/walkgold/trace_layer_golden.py --step 27 --layer 23

Writes <out>/layer_s<step>_L<layer>.npz + .json (the bundle the image and
op-stream compilers consume). Weight MATRICES are not copied into the bundle
(233 MB/layer); the bundle pins their path + sha256 and the compilers mmap
them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO))

from apex_golden import attention as at                      # noqa: E402
from apex_golden import transformer as tf                    # noqa: E402
from apex_golden.fp import f64_to_f16_bits                   # noqa: E402
import run_tinynpu as s8                                     # noqa: E402

S8_TRACE = REPO / "docs/results/s8_7b_token/artifact_trace"
# The S8 weight cache is a gitignored build artifact (~7.6 GB) that lives in
# whichever checkout produced it — REPO-relative is right for a clone, but a
# git WORKTREE has no copy, so allow an explicit override and fail loudly with
# the fix rather than silently on a missing path.
_W_REL = "build/s8_weights/Qwen2.5-7B-4bit"
DEF_WEIGHTS = Path(os.environ.get("APEX_S8_WEIGHTS", REPO / _W_REL))

# per-head fields cross-checked against the committed S8 record
XCHECK = ("q_f16", "K_f16", "V_f16", "k8", "v8", "q8", "s_k", "s_v", "s_q",
          "acc_s", "score_fx", "p_q115", "c8", "s_c", "acc_o", "rq_scale",
          "rq_shift", "o8", "s_out", "out_hat", "sm_m", "sm_l")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def token_sequence(run: dict) -> list[int]:
    """The committed run's fed-token order: prompt ids at steps 0..P-1, then
    the greedily generated ids. run_generate feeds generated_ids[i] at step
    P+i, so token_at_step[s] indexes straight into this list."""
    return list(run["prompt_ids"]) + list(run["generated_ids"])


def trace(step: int, layer: int, wdir: Path, tier: str, group: int):
    run = json.loads((S8_TRACE / "run.json").read_text())
    assert run["tier"] == tier and run["group"] == group, \
        f"run.json tier/group ({run['tier']}/{run['group']}) != requested"
    toks = token_sequence(run)
    assert step < len(toks), f"step {step} beyond the committed run"

    model = s8.GoldenModel(wdir)
    m = model.meta
    assert m["model"] == run["model"], "weight cache is not the run's model"
    print(f"[trace] {m['model']}  D={m['D_model']} H={m['H']} "
          f"H_kv={m['H_kv']} hd={m['head_dim']} d_ffn={m['d_ffn']} "
          f"layers={m['n_layers']} theta={m['rope_theta']}")
    print(f"[trace] target step={step} layer={layer} -> T={step+1}, "
          f"q_pos={step}; replaying {step+1} steps x layers 0..{layer}")

    # Session.step semantics, replayed verbatim but truncated at the target
    # layer (layers > target cannot influence layer <= target at any step).
    hist: list[list[np.ndarray]] = [[] for _ in range(layer + 1)]
    target = None
    t0 = time.time()
    for t in range(step + 1):
        x = model.embed_row(toks[t])
        for li in range(layer + 1):
            hist[li].append(x)
            X = np.vstack(hist[li] + [x])        # rows 0..t + the decode dup
            r = tf.decoder_layer_fx(X, model.layers[li], tier, G=group,
                                    q_pos=t)
            if t == step and li == layer:
                target = (X, r)
            x = r.r2
        print(f"[trace]   step {t:3d}/{step}  T={t+1}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    X, r = target
    assert r.T == step + 1, f"T {r.T} != step+1 {step+1}"
    return model, run, X, r


def crosscheck_committed(step: int, layer: int, r) -> tuple[int, int]:
    """Bit-exact comparison against every committed per-head record of this
    (step, layer). This is the provenance gate — see the module docstring."""
    n_files = n_fields = 0
    for p in sorted(S8_TRACE.glob(f"job_s{step:03d}_L{layer:02d}_h*.npz")):
        if p.stem.endswith(("_c0", "_c1")):
            raise SystemExit(f"{p.name}: chunked head — pick a step with "
                             "T <= CHUNK_T_MAX so heads are single-core")
        z = np.load(p)
        meta = json.loads(str(z["meta_json"]))
        h = meta["head"]
        core = r.heads[h]
        assert not isinstance(core, tf.ChunkedHead), f"head {h} chunked"
        got = s8.core_record(core, {})
        got["q_f16"] = f64_to_f16_bits(core.q).astype(np.uint16)
        bad = []
        for k in XCHECK:
            if not np.array_equal(np.asarray(got[k]), np.asarray(z[k])):
                bad.append(k)
            else:
                n_fields += 1
        if bad:
            raise SystemExit(f"PROVENANCE FAIL {p.name}: fields differ {bad} "
                             "— the replay is not the committed run's step")
        n_files += 1
        print(f"[xcheck] {p.name}: head {h} bit-exact on {len(XCHECK)} fields")
    if not n_files:
        raise SystemExit(f"no committed head record for step {step} layer "
                         f"{layer} — pick a traced (step, layer)")
    return n_files, n_fields


def build_bundle(model, run, X, r, step: int, layer: int, wdir: Path,
                 tier: str, group: int) -> tuple[dict, dict]:
    w = model.layers[layer]
    lr = model.meta["layers"][layer]
    H, hd, H_kv = r.H, r.head_dim, r.H_kv
    grp = H // H_kv

    arr: dict[str, np.ndarray] = {
        "X": np.asarray(X, dtype=np.float64),
        "h": np.asarray(r.h, dtype=np.int64),
        "h8": np.asarray(r.h8, dtype=np.int8),
        "s_h": np.asarray(r.s_h, dtype=np.uint16),
        "q_real": np.asarray(r.q_real, dtype=np.float64),
        "K_real": np.asarray(r.K_real, dtype=np.float64),
        "V_real": np.asarray(r.V_real, dtype=np.float64),
        "q_rope": np.asarray(r.q_rope, dtype=np.float64),
        "K_rope": np.asarray(r.K_rope, dtype=np.float64),
        "attn": np.asarray(r.attn, dtype=np.float64),
        "attn_proj": np.asarray(r.attn_proj, dtype=np.float64),
        "r1": np.asarray(r.r1, dtype=np.float64),
        "h2": np.asarray(r.h2, dtype=np.int64),
        "gate_real": np.asarray(r.gate_real, dtype=np.float64),
        "up_real": np.asarray(r.up_real, dtype=np.float64),
        "swiglu": np.asarray(r.swiglu, dtype=np.float64),
        "ffn_out": np.asarray(r.ffn_out, dtype=np.float64),
        "r2": np.asarray(r.r2, dtype=np.float64),
        "gamma1": np.asarray(w.gamma1, dtype=np.int64),
        "gamma2": np.asarray(w.gamma2, dtype=np.int64),
    }
    for nm in ("bq", "bk", "bv"):
        b = getattr(w, nm, None)
        if b is not None:
            arr[nm] = np.asarray(b, dtype=np.float64)

    # per-head attention state (the score/pv drive + KVQ records + RQ table)
    for h in range(H):
        c = r.heads[h]
        arr[f"h{h:02d}_q_f16"] = f64_to_f16_bits(c.q).astype(np.uint16)
        arr[f"h{h:02d}_K_f16"] = np.asarray(c.K_f16, dtype=np.uint16)
        arr[f"h{h:02d}_V_f16"] = np.asarray(c.V_f16, dtype=np.uint16)
        arr[f"h{h:02d}_s_k"] = np.asarray(c.s_k, dtype=np.uint16)
        arr[f"h{h:02d}_s_v"] = np.asarray(c.s_v, dtype=np.uint16)
        arr[f"h{h:02d}_s_q"] = np.uint16(c.s_q)
        arr[f"h{h:02d}_o8"] = np.asarray(c.o8, dtype=np.int8)
        arr[f"h{h:02d}_s_out"] = np.float64(c.s_out)
        arr[f"h{h:02d}_rq"] = np.asarray(c.rq, dtype=np.int64)
        arr[f"h{h:02d}_out_hat"] = np.asarray(c.out_hat, dtype=np.float64)

    wfiles = {tag: wdir / f"L{layer:02d}_{tag}.npy"
              for tag in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")}
    meta = {
        "format": "apex-wg3-layer-trace-v1",
        "source": "docs/results/s8_7b_token/artifact_trace (committed S8 run)",
        "model": model.meta["model"],
        "run_git": run["git"], "run_prompt": run["prompt"],
        "step": step, "layer": layer, "T": int(r.T), "q_pos": step,
        "tier": tier, "G": group,
        "token_at_step": token_sequence(run)[step],
        "D_model": int(r.D_model), "d_ffn": int(r.d_ffn), "H": H,
        "H_kv": H_kv, "head_dim": hd, "gqa_group": grp,
        "rope_theta": float(model.meta["rope_theta"]),
        "weights_dir": str(wdir),
        "scales": {"s_wq": w.s_wq, "s_wk": w.s_wk, "s_wv": w.s_wv,
                   "s_wo": w.s_wo, "s_wg": w.s_wg, "s_wu": w.s_wu,
                   "s_wd": w.s_wd},
        "gamma_folds": {"gamma1_fold_k": lr["gamma1_fold_k"],
                        "gamma2_fold_k": lr["gamma2_fold_k"]},
        "has_bias": {n: (getattr(w, n, None) is not None)
                     for n in ("bq", "bk", "bv")},
        "weight_files": {t: {"path": str(p), "shape": list(np.load(
            p, mmap_mode="r").shape), "sha256": sha256_file(p)}
            for t, p in wfiles.items()},
        "kv_head_of": [h // grp for h in range(H)],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "disclosure": model.meta["disclosure"],
    }
    return arr, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=19)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--weights-dir", default=str(DEF_WEIGHTS))
    ap.add_argument("--tier", default="CQ-8")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out", default=str(REPO / "build/walkgold"))
    args = ap.parse_args()

    wdir = Path(args.weights_dir)
    if not (wdir / "meta.json").exists():
        raise SystemExit(f"no prepared weight cache at {wdir} — S8 "
                         "--prepare output is required (gitignored, ~7.6 GB)")
    model, run, X, r = trace(args.step, args.layer, wdir, args.tier,
                             args.group)
    n_files, n_fields = crosscheck_committed(args.step, args.layer, r)
    arr, meta = build_bundle(model, run, X, r, args.step, args.layer, wdir,
                             args.tier, args.group)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"layer_s{args.step:03d}_L{args.layer:02d}"
    np.savez_compressed(out / f"{stem}.npz", **arr)
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    sz = (out / f"{stem}.npz").stat().st_size
    print(f"[trace] wrote {out}/{stem}.npz ({sz/1e6:.1f} MB) + .json")
    print(f"WG3 LAYER TRACE: PASS (step {args.step} layer {args.layer}, "
          f"T={meta['T']} q_pos={meta['q_pos']} {meta['tier']}; provenance "
          f"{n_files} committed head record(s), {n_fields} arrays bit-exact; "
          f"{meta['H']} heads, {meta['H_kv']} kv groups, 7 weight tensors "
          f"sha-pinned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
