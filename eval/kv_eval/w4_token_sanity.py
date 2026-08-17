#!/usr/bin/env python3
# w4_token_sanity.py — B3 stage-6 token-level sanity: greedy generation on
# Qwen2.5-7B, stock weights vs the B3 weight tiers, via the SAME certified
# installer as the HellaSwag matrix (w4_weights_mlx).
#
# ⚠️ SCOPE DISCLOSURE (recorded in the output JSON): this streams tokens
# through MLX, not through the golden fixed-point pipeline. run_tinynpu.py
# cannot carry realization (A)'s per-stripe scales without editing golden —
# _proj_epilogue takes ONE scalar s_w per tensor (frozen C-2 contract), so
# the golden-pipeline W4 token stream is B3 stage-5 integration work (where
# the per-job scale plumbing lands). What THIS run shows is the token-level
# effect of the W4 weight VALUES on the real model: greedy divergence step
# and text coherence vs the stock stream, prompt matched to the committed S8
# artifact run (docs/results/s8_7b_token/).
#
#   python w4_token_sanity.py --tokens 150
#
# Install order matters: install_tier asserts the modules it replaces are
# still 4-bit/g64, and it rebuilds every tensor from the CACHE bytes (never
# from the model's current weights), so stacking installs is value-correct
# as long as each install starts from 4-bit modules. Order used: stock (no
# install) → w4-tile-a (modules stay 4-bit) → w-int8 (goes 8-bit, last).

import argparse
import datetime
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import w4_weights_mlx as W4  # noqa: E402

PROMPT = "Here are five interesting facts about the Moon:\n1."  # S8 artifact


def greedy(model, tokenizer, prompt: str, n: int) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache
    ids = tokenizer.encode(prompt)
    cache = make_prompt_cache(model)
    toks = list(ids)
    out = []
    y = mx.array(ids)[None]
    for _ in range(n):
        logits = model(y, cache=cache)
        nxt = int(mx.argmax(logits[0, -1]))
        out.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        y = mx.array([[nxt]])
        toks.append(nxt)
    mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-4bit")
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--weights-cache",
                    default=str(REPO / "build/s8_weights/Qwen2.5-7B-4bit"))
    ap.add_argument("--outdir",
                    default=str(REPO / "docs/results/b3_w4_adoption"))
    ap.add_argument("--tiers", default="base,w4-tile-a,w-int8",
                    help="comma list; 'base' first, at most one 8-bit tier "
                         "last (installs are one-way past 8-bit)")
    args = ap.parse_args()
    tiers = args.tiers.split(",")

    n_wtwin = W4.weight_twin_selftest()
    print(f"[gate] WEIGHT TWIN GATE: checks={n_wtwin} fails=0")
    n_pack = W4.mlx_pack_selftest()
    print(f"[gate] MLX PACK GATE: checks={n_pack} fails=0")

    from mlx_lm import load
    model, tokenizer = load(args.model)

    runs = {}
    gates = {}
    for wtier in tiers:
        if wtier != "base":
            print(f"[install] {wtier}")
            gates[wtier] = W4.install_tier(model, wtier,
                                           Path(args.weights_cache),
                                           expect_model=args.model)
            print(f"[gate] install {wtier}: "
                  f"stripe_checks={gates[wtier]['golden_stripe_checks']} "
                  f"fails=0")
        toks = greedy(model, tokenizer, PROMPT, args.tokens)
        runs[wtier] = {"tokens": toks, "text": tokenizer.decode(toks)}
        print(f"--- {wtier} ({len(toks)} tokens) ---")
        print(PROMPT + runs[wtier]["text"])
        print()

    base = runs["base"]["tokens"]
    for wtier in [t for t in tiers if t != "base"]:
        t = runs[wtier]["tokens"]
        div = next((i for i in range(min(len(base), len(t)))
                    if base[i] != t[i]), min(len(base), len(t)))
        runs[wtier]["first_divergence_step"] = div
        runs[wtier]["match_prefix_of_base"] = div
        print(f"DIVERGENCE {wtier}: first differing greedy token at step "
              f"{div} of {len(t)}")

    out = {
        "model": args.model, "prompt": PROMPT, "tokens": args.tokens,
        "weight_twin_gate": f"checks={n_wtwin} fails=0",
        "mlx_pack_gate": f"checks={n_pack} fails=0",
        "install_gates": gates,
        "runs": runs,
        "scope_disclosure":
            "MLX forward path (fp16 activations), NOT the golden "
            "fixed-point pipeline: golden _proj_epilogue carries ONE scalar "
            "s_w per tensor (frozen C-2), so per-stripe realization-(A) "
            "scales cannot thread through run_tinynpu.py without editing "
            "golden. The golden-pipeline W4 token stream is B3 stage-5 "
            "integration work. This run isolates the token-level effect of "
            "the W4 weight VALUES (installed by the same certified "
            "installer as the HellaSwag matrix). Greedy divergence is "
            "EXPECTED at some step for any weight perturbation; the "
            "decision signal is coherence of the diverged text plus the "
            "HellaSwag paired deltas, not divergence step alone.",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "versions": {"mlx": mx.__version__, "numpy": np.__version__},
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"token_sanity_{'_'.join(tiers)}_n{args.tokens}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"WROTE {p}")


if __name__ == "__main__":
    main()
