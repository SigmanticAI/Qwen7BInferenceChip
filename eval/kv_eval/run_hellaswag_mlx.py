#!/usr/bin/env python3
# run_hellaswag_mlx.py — S5 EVAL-7B: HellaSwag with Qwen2.5-7B and the KV
# cache routed through the bit-exact-certified KVQ codec twin (same twin,
# same gate as S4). 7B fp16 weights do not fit this 18 GB machine, so the
# model runs on MLX with 4-bit weights; the baseline tier uses the SAME
# 4-bit weights through an identity hook on the identical code path, so
# KVQ deltas isolate the codec — weight precision is disclosed in the JSON.
#
#   python run_hellaswag_mlx.py --model mlx-community/Qwen2.5-7B-4bit --tier kvq4 --limit 1000
#
# tiers: base (identity-hooked 4-bit-weight baseline) | kvq8 | kvq4 | kvq4p
# GATE: refuses to run unless test_twin_bitexact.py passes first.
#
# Scoring: one full-sequence forward per (context, ending) with a FRESH cache
# and batch size 1 — the S4 semantics (cache-at-rest G=128 key groups aligned
# to position 0). Deliberately NOT mlx_lm's shared-prefix scoring path: prefix
# reuse would quantize the same position under two different partial-group
# scales in the two forwards.

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from kvq_cache_mlx import make_kvq_caches  # noqa: E402


def certify_twin():
    r = subprocess.run([sys.executable, str(HERE / "test_twin_bitexact.py")],
                       capture_output=True, text=True)
    # keep the counts line AND the verdict line so the check count is
    # reproducible from the committed record, not just the PASS
    line = " | ".join((r.stdout.strip().splitlines() or ["<no output>"])[-2:])
    if r.returncode != 0:
        sys.exit(f"ABORT: twin bit-exact gate failed:\n{r.stdout}\n{r.stderr}")
    print(f"[gate] {line}")
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-4bit")
    ap.add_argument("--tier", required=True,
                    choices=["base", "kvq8", "kvq4", "kvq4p"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--outdir", default=str(REPO / "docs/results/s5_eval7b"))
    args = ap.parse_args()

    gate_line = certify_twin()
    if args.tier == "kvq4p":
        sys.exit("ABORT: kvq4p deferred — D=128 RTL build is maskless until "
                 "the loadable-mask task; report kvq8/kvq4 first.")

    import lm_eval
    import mlx_lm
    from huggingface_hub import hf_hub_download
    from mlx_lm.evaluate import MLXLM
    from tqdm import tqdm

    class KVQMLXLM(MLXLM):
        tier = "identity"
        group = 128
        outliers = None

        def _score_full(self, toks):
            """Forward toks[:-1] with a fresh KVQ cache; per-position
            continuation logprobs and greedy hits for toks[1:]."""
            cache = make_kvq_caches(self._model, self.tier, self.group,
                                    self.outliers)
            inp = mx.array(toks[:-1])[None]
            logits = self._model(inp, cache=cache)
            logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
            targets = mx.array(toks[1:])[None]
            sc = mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
            ig = mx.argmax(logits, axis=-1) == targets
            mx.eval(sc, ig)
            mx.clear_cache()
            return np.array(sc[0]), np.array(ig[0])

        def loglikelihood(self, requests):
            out = []
            for req in tqdm(requests):
                ctx, cont = req.args
                prefix = self._tokenize([ctx])[0]
                full = self._tokenize([ctx + cont])[0]
                assert 0 < len(prefix) < len(full) <= 4096, \
                    (len(prefix), len(full))
                sc, ig = self._score_full(full)
                out.append((float(sc[len(prefix) - 1:].sum()),
                            bool(ig[len(prefix) - 1:].all())))
            return out

    KVQMLXLM.tier = "identity" if args.tier == "base" else args.tier
    KVQMLXLM.group = args.group
    lm = KVQMLXLM(args.model, batch_size=1, use_chat_template=False)

    cfg = json.loads(Path(hf_hub_download(args.model, "config.json")).read_text())
    res = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"],
                                  limit=args.limit, log_samples=True)
    h = res["results"]["hellaswag"]
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    record = {
        "model": args.model, "tier": args.tier, "task": "hellaswag",
        "limit": args.limit,
        "group": args.group, "device": "mlx-metal",
        "stderr_method": "lm-eval analytic binomial sqrt(p*(1-p)/(n-1))",
        "acc": h.get("acc,none"), "acc_stderr": h.get("acc_stderr,none"),
        "acc_norm": h.get("acc_norm,none"),
        "acc_norm_stderr": h.get("acc_norm_stderr,none"),
        "twin_gate": gate_line,
        "weights": {
            "quantization": cfg.get("quantization"),
            "disclosure": "WEIGHTS ARE 4-BIT (MLX affine, group_size 64), "
                          "not fp16 — 7B fp16 does not fit 18 GB unified "
                          "memory. The base tier runs the SAME 4-bit weights "
                          "with an identity KV hook on the identical code "
                          "path, so tier deltas isolate the KV codec.",
        },
        "model_shape": {k: cfg.get(k) for k in
                        ["num_hidden_layers", "num_attention_heads",
                         "num_key_value_heads", "hidden_size"]},
        "d128_note": "head_dim=128: kvq8/kvq4 fully covered by the twin gate "
                     "(D=128 cases included). KVQ4+ deferred: the D=128 RTL "
                     "build is maskless until S12.",
        "versions": {"mlx": mx.__version__,
                     "mlx_lm": mlx_lm.__version__,
                     "lm_eval": lm_eval.__version__,
                     "numpy": np.__version__},
        "timestamp": stamp,
        "method": "KV routed through bit-exact-certified twin of golden "
                  "cq_codec via KVCache.update_and_fetch (mlx-lm qwen2: keys "
                  "stored post-RoPE); raw fp16 storage, codec applied to the "
                  "full buffer at read (cache-at-rest); fp32 dequant cast to "
                  "fp16 (conservative); one full-sequence forward per ending "
                  "with a fresh cache, batch_size=1 (no padding pollution, "
                  "no shared-prefix path); baseline = identity hook on the "
                  "same path. Tokenization: mlx-lm whole-string convention "
                  "(context+ending encoded together, split at the context "
                  "token count) — absolutes not comparable to S4's HFLM "
                  "absolutes; deltas are the comparable quantity.",
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model.split('/')[-1]}_{args.tier}_n{args.limit}.json"
    (outdir / name).write_text(json.dumps(record, indent=1))
    # compact per-document outcomes: enables PAIRED tier-vs-base statistics
    # (the honest σ for deltas) and doc-level reproducibility checks
    samples = [{"doc_id": s["doc_id"], "acc": s["acc"], "acc_norm": s["acc_norm"]}
               for s in res["samples"]["hellaswag"]]
    (outdir / name.replace(".json", ".samples.json")).write_text(
        json.dumps(samples, separators=(",", ":")))
    print(f"RESULT {args.model} {args.tier} n={args.limit}: "
          f"acc={record['acc']:.4f} acc_norm={record['acc_norm']:.4f} "
          f"(stderr {record['acc_norm_stderr']:.4f}) -> {outdir / name}")


if __name__ == "__main__":
    main()
