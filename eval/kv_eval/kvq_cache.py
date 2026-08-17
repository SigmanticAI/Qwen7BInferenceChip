# kvq_cache.py — transformers Cache subclass that routes every layer's
# post-RoPE K/V through the (bit-exact-certified) KVQ codec twin during a
# forward pass, plus the calibration recorder for KVQ4+ static outlier lanes.
#
# Methodology notes (disclose with results):
# - HF attention uses the tensors RETURNED by Cache.update(), so a full-
#   sequence forward with this cache scores the model as if attention read
#   dequantized KV — the same effect the tile's fp32 dequant bus has.
# - Keys are quantized post-RoPE (what a KV cache stores), per-channel over
#   the token axis in G-token groups, partial final group scales over g (§3.1).
# - Dequant is fp32 (D-010); the eval casts back to the model dtype (fp16),
#   one extra RNE our hardware does not have — conservative for us.
# - fp16 baseline runs tier="identity" through this same class so every tier
#   shares an identical code path; deltas isolate the codec alone.

import sys
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvq_numpy as tw  # noqa: E402


class KVQCache(DynamicCache):
    def __init__(self, tier: str, G: int = 128, outliers=None):
        super().__init__()
        self.tier = tier
        self.G = G
        self.outliers = outliers or {}  # {layer_idx: [head][k] channel indices}

    def _qdq(self, x: torch.Tensor, layer_idx: int, is_key: bool) -> torch.Tensor:
        B, H, T, D = x.shape
        xb = x.detach().to("cpu", torch.float16).numpy().view(np.uint16)
        out = np.empty((B, H, T, D), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                if not is_key or self.tier == "kvq8":
                    out[b, h] = tw.qdq_values(xb[b, h], 8 if self.tier == "kvq8" else 4)
                else:
                    idx = self.outliers.get(layer_idx, [[]] * H)[h] if self.tier == "kvq4p" else ()
                    out[b, h] = tw.qdq_keys(xb[b, h], 4, self.G, idx)
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

    def update(self, key_states, value_states, layer_idx, *a, **kw):
        k, v = super().update(key_states, value_states, layer_idx, *a, **kw)
        if self.tier == "identity":
            return k, v
        return self._qdq(k, layer_idx, True), self._qdq(v, layer_idx, False)


class KeyAmaxRecorder(DynamicCache):
    """Calibration: accumulate per-(layer, head, channel) |K| maxima."""

    def __init__(self, store: dict):
        super().__init__()
        self.store = store  # {layer_idx: np.ndarray [H, D]}

    def update(self, key_states, value_states, layer_idx, *a, **kw):
        k, v = super().update(key_states, value_states, layer_idx, *a, **kw)
        am = k.detach().abs().amax(dim=(0, 2)).to("cpu", torch.float32).numpy()  # [H, D]
        prev = self.store.get(layer_idx)
        self.store[layer_idx] = am if prev is None else np.maximum(prev, am)
        return k, v


def topk_outliers(store: dict, k: int = 2) -> dict:
    """{layer: [H][k] channel indices} — static top-k amax channels per head."""
    return {int(li): [np.argsort(-am[h])[:k].tolist() for h in range(am.shape[0])]
            for li, am in store.items()}
