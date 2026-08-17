#!/usr/bin/env python3
# test_mlx_identity.py — S5 pre-flight (the S4 "identity-hooked logits ==
# no-cache logits" check, MLX edition). Verifies on real prompts that:
#   1. identity-hooked KVQMLXCache logits == plain-KVCache logits == cache-free
#      logits, BITWISE — the hook path is transparent, so tier deltas isolate
#      the codec;
#   2. kvq8/kvq4 logits actually differ (the hook engages) and are finite.
# Loads the full model: run only when the machine is quiet (serialize rule).

import argparse
import sys

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import load

from kvq_cache_mlx import make_kvq_caches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-4bit")
args = ap.parse_args()

model, tokenizer = load(args.model)

PROMPTS = [
    "The capital of France is",
    "A man is sitting on a roof. He starts pulling up",
    "Members of the procession walk down the street holding small horn "
    "brass instruments. A drum line",
]

fails = 0
for text in PROMPTS:
    toks = tokenizer.encode(text)
    inp = mx.array(toks)[None]

    def logits_with(cache):
        out = np.array(model(inp, cache=cache).astype(mx.float16))
        mx.clear_cache()
        return out

    l_none = logits_with(None)
    l_kv = logits_with([KVCache() for _ in model.layers])
    l_id = logits_with(make_kvq_caches(model, "identity"))
    l_q8 = logits_with(make_kvq_caches(model, "kvq8"))
    l_q4 = logits_with(make_kvq_caches(model, "kvq4"))

    same_kv = np.array_equal(l_none.view(np.uint16), l_kv.view(np.uint16))
    same_id = np.array_equal(l_none.view(np.uint16), l_id.view(np.uint16))
    q8_diff = float(np.abs(l_q8 - l_none).max())
    q4_diff = float(np.abs(l_q4 - l_none).max())
    ok = (same_kv and same_id
          and q8_diff > 0 and np.isfinite(l_q8).all()
          and q4_diff > 0 and np.isfinite(l_q4).all())
    fails += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'} n_tok={len(toks):3d} "
          f"kvcache==nocache:{same_kv} identity==nocache:{same_id} "
          f"max|dlogit| kvq8={q8_diff:.4f} kvq4={q4_diff:.4f}  «{text[:40]}…»")

if fails:
    sys.exit(f"MLX IDENTITY CHECK: {fails}/{len(PROMPTS)} FAILED")
print("MLX IDENTITY CHECK: PASS — identity hook is bitwise-transparent; "
      "kvq tiers engage the codec")
