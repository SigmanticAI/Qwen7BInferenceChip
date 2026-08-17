#!/usr/bin/env python3
# chat_kv.py — interactive chat whose KV cache runs through the APEX KVQ
# codec: type a prompt, tokens stream back, and every layer's conversation
# memory is round-tripped through the compression path on every step.
#
# Backends (--backend):
#   twin  (default) the bit-exact-certified software twin of the golden codec
#                   (eval/kv_eval/kvq_numpy — same twin the S4/S5 accuracy
#                   evals used; certified against the arbiter by
#                   test_twin_bitexact.py: 2,254,080 checks / 0 fails).
#   fpga            the same interface backed by the KVQ engine on real
#                   hardware (AWS F2 / ECP5 board) — NOT WIRED YET; raises
#                   with a pointer to scripts/fpga/f2/BRINGUP.md stage 3.
#
# STATUS: VALIDATED 2026-07-18 on a quiet machine —
#   0.5B-Instruct kvq8: 39 tok @ 34.2 tok/s, coherent
#   7B-Instruct   kvq8: 41 tok @ 13.6 tok/s, coherent ("hello how are you"
#   → normal assistant reply; footer self-documents the KV path)
# Serialize against eval/EDA jobs before running (one heavy job per machine).
#
# Usage (quiet machine only):
#   python3 demo/chat_kv.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
#       --tier kvq8 --backend twin
#   (0.5B variant for quick smoke: mlx-community/Qwen2.5-0.5B-Instruct-4bit)

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval" / "kv_eval"))


def build_backend(name: str):
    if name == "twin":
        import kvq_numpy as tw
        return tw  # exposes qdq_values / qdq_keys — the certified twin
    if name == "fpga":
        raise NotImplementedError(
            "FPGA backend lands at F2 bring-up stage 3 "
            "(scripts/fpga/f2/BRINGUP.md): same qdq_values/qdq_keys interface, "
            "served by the KVQ engine over the OCL/PCIS bridge instead of "
            "kvq_numpy. Until then run --backend twin.")
    raise ValueError(f"unknown backend {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--tier", default="kvq8",
                    choices=["identity", "kvq8", "kvq4"])
    ap.add_argument("--backend", default="twin", choices=["twin", "fpga"])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    backend = build_backend(args.backend)

    from mlx_lm import load, stream_generate
    from kvq_cache_mlx import KVQMLXCache

    class BackendKVCache(KVQMLXCache):
        """KVQMLXCache with the codec calls routed through the backend
        object (twin today, FPGA later) — the ONLY difference from the
        eval hook, so the demo path stays the certified path."""
        def _qdq(self, x, is_key):
            import numpy as np
            import mlx.core as mx
            xb = np.array(x.astype(mx.float16)).view(np.uint16)
            B, H, T, D = xb.shape
            out = np.empty((B, H, T, D), dtype=np.float32)
            for b in range(B):
                for h in range(H):
                    if not is_key or self.tier == "kvq8":
                        out[b, h] = backend.qdq_values(
                            xb[b, h], 8 if self.tier == "kvq8" else 4)
                    else:
                        out[b, h] = backend.qdq_keys(xb[b, h], 4, self.G, ())
            return mx.array(out.astype(np.float16)).astype(x.dtype)

    print(f"[chat_kv] loading {args.model} …", file=sys.stderr)
    model, tokenizer = load(args.model)
    n_layers = len(model.layers)
    print(f"[chat_kv] {n_layers} layers · tier={args.tier} · "
          f"backend={args.backend} · G={args.group}", file=sys.stderr)

    history = []
    print("── APEX KV-chat — every token's KV cache round-trips through the "
          f"{'certified software twin' if args.backend == 'twin' else 'FPGA KVQ engine'}"
          f" ({args.tier}). Ctrl-D to exit. ──")
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        if not user:
            continue
        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True)
        caches = [BackendKVCache(args.tier, args.group, None)
                  for _ in range(n_layers)]
        t0, n_tok, reply = time.time(), 0, []
        print("apex> ", end="", flush=True)
        for resp in stream_generate(model, tokenizer, prompt,
                                    max_tokens=args.max_tokens,
                                    prompt_cache=caches):
            print(resp.text, end="", flush=True)
            reply.append(resp.text)
            n_tok += 1
        dt = time.time() - t0
        print(f"\n[{n_tok} tok · {dt:.1f}s · {n_tok / max(dt, 1e-9):.2f} tok/s "
              f"· KV path: {args.backend}/{args.tier}]", file=sys.stderr)
        history.append({"role": "assistant", "content": "".join(reply)})


if __name__ == "__main__":
    sys.exit(main())
