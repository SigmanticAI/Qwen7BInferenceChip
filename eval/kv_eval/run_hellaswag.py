#!/usr/bin/env python3
# run_hellaswag.py — S4 head-to-head: HellaSwag with the KV cache routed
# through the bit-exact-certified KVQ codec twin. published ChannelQuant-family setup:
# Qwen2-0.5B / Qwen2-1.5B, acc_norm, n-limited. Raw JSON + log committed.
#
#   python run_hellaswag.py --model Qwen/Qwen2-0.5B --tier kvq4 --limit 1000
#
# tiers: fp16 (identity-hooked baseline) | kvq8 | kvq4 | kvq4p
# GATE: refuses to run unless test_twin_bitexact.py passes first.

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from kvq_cache import KVQCache, KeyAmaxRecorder, topk_outliers  # noqa: E402


def certify_twin():
    r = subprocess.run([sys.executable, str(HERE / "test_twin_bitexact.py")],
                       capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
    if r.returncode != 0:
        sys.exit(f"ABORT: twin bit-exact gate failed:\n{r.stdout}\n{r.stderr}")
    print(f"[gate] {line}")
    return line


def calibrate(lm, docs_n=32):
    """Static top-2 outlier channels per (layer, head) from |K| maxima over
    HellaSwag TRAIN contexts (no test leakage)."""
    import datasets
    ds = datasets.load_dataset("Rowan/hellaswag", split=f"train[:{docs_n}]")
    store = {}
    for doc in ds:
        text = doc["ctx"]
        enc = lm.tok_encode(text)
        inp = torch.tensor([enc], device=lm.device)
        with torch.no_grad():
            lm.model(inp, past_key_values=KeyAmaxRecorder(store), use_cache=True)
    return topk_outliers(store, k=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", required=True,
                    choices=["fp16", "kvq8", "kvq4", "kvq4p"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--outdir", default=str(REPO / "docs/results/s4_head2head"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    gate_line = certify_twin()

    import lm_eval
    import transformers
    from lm_eval.models.huggingface import HFLM

    class KVQHFLM(HFLM):
        cache_factory = None

        def _model_call(self, inps, attn_mask=None, labels=None):
            if self.cache_factory is None or attn_mask is not None or labels is not None:
                return super()._model_call(inps, attn_mask, labels)
            with torch.no_grad():
                return self.model(inps, past_key_values=self.cache_factory(),
                                  use_cache=True).logits

    # Load manually and hand the object to HFLM: HFLM's by-name path placed
    # the model on CPU (fp16-on-CPU grind) or segfaulted under transformers
    # 5.13 + MPS; AutoModel -> .to(mps) is the verified-fast path.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lm = KVQHFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    outliers = None
    if args.tier == "kvq4p":
        print("[calib] recording key amax over 32 HellaSwag TRAIN contexts…")
        outliers = calibrate(lm)

    tier = "identity" if args.tier == "fp16" else args.tier
    KVQHFLM.cache_factory = staticmethod(
        lambda: KVQCache(tier, G=args.group, outliers=outliers))

    res = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"],
                                  limit=args.limit, bootstrap_iters=1000)
    h = res["results"]["hellaswag"]
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    record = {
        "model": args.model, "tier": args.tier, "limit": args.limit,
        "group": args.group, "device": args.device,
        "acc": h.get("acc,none"), "acc_stderr": h.get("acc_stderr,none"),
        "acc_norm": h.get("acc_norm,none"),
        "acc_norm_stderr": h.get("acc_norm_stderr,none"),
        "twin_gate": gate_line,
        "outliers": outliers,
        "versions": {"transformers": transformers.__version__,
                     "lm_eval": lm_eval.__version__,
                     "torch": torch.__version__},
        "timestamp": stamp,
        "method": "KV routed through bit-exact-certified twin of golden "
                  "cq_codec via Cache.update(); post-RoPE keys; fp32 dequant "
                  "cast to fp16 (conservative); baseline uses identity hook "
                  "on the same code path; batch_size=1 (no padding pollution).",
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model.split('/')[-1]}_{args.tier}_n{args.limit}.json"
    (outdir / name).write_text(json.dumps(record, indent=1))
    print(f"RESULT {args.model} {args.tier} n={args.limit}: "
          f"acc={record['acc']:.4f} acc_norm={record['acc_norm']:.4f} "
          f"(stderr {record['acc_norm_stderr']:.4f}) -> {outdir / name}")


if __name__ == "__main__":
    main()
