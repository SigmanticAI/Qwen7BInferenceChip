# kvq_cache_mlx.py — MLX KVCache subclass that routes every layer's post-RoPE
# K/V through the bit-exact-certified KVQ codec twin (kvq_numpy) at read time.
# S5 analog of kvq_cache.py (the S4 HF hook), same semantics:
# - storage stays RAW fp16 (super() stores); the codec is applied to the full
#   buffer on every fetch, so group scales are computed over the tokens present
#   (cache-at-rest, G-token key groups aligned to position 0, partial final
#   group scales over g — §3.1).
# - dequant is fp32, cast back to the model dtype (fp16) — one extra RNE the
#   hardware's fp32 read bus does not have; conservative against us.
# - tier="identity" returns the raw tensors through this same class so every
#   tier (baseline included) shares an identical code path.
#
# mlx_lm's scaled_dot_product_attention dispatches quantized attention on
# hasattr(cache, "bits") — this class must never grow a `bits` attribute.

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvq_numpy as tw  # noqa: E402


class KVQMLXCache(KVCache):
    def __init__(self, tier: str, G: int = 128, outliers=None):
        super().__init__()
        self.tier = tier
        self.G = G
        self.outliers = outliers  # [H][k] channel indices for THIS layer, or None

    def _qdq(self, x: mx.array, is_key: bool) -> mx.array:
        xb = np.array(x.astype(mx.float16)).view(np.uint16)
        B, H, T, D = xb.shape
        out = np.empty((B, H, T, D), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                if not is_key or self.tier == "kvq8":
                    out[b, h] = tw.qdq_values(xb[b, h], 8 if self.tier == "kvq8" else 4)
                else:
                    idx = self.outliers[h] if self.tier == "kvq4p" else ()
                    out[b, h] = tw.qdq_keys(xb[b, h], 4, self.G, idx)
        return mx.array(out.astype(np.float16)).astype(x.dtype)

    def update_and_fetch(self, keys, values):
        k, v = super().update_and_fetch(keys, values)
        if self.tier == "identity":
            return k, v
        return self._qdq(k, True), self._qdq(v, False)


def make_kvq_caches(model, tier: str, G: int = 128, outliers=None):
    """One KVQMLXCache per layer ({layer_idx: [H][k]} outliers, kvq4p only)."""
    return [KVQMLXCache(tier, G, (outliers or {}).get(li))
            for li in range(len(model.layers))]
