"""l4_cases.py — the L4 (decoder-layer composition) case set, shared by
gen_l4_vectors.py and measure_bus_deltas.py.

Tile-scale configs only (the CAMPAIGN §3-§4 qualifier applies to every claim
made from these): D_model = 128 (the rmsnorm/tile envelope), head_dim in
{64, 128} (the CFG_D build points), T <= T_ROW_MAX = 128, tier CQ-8 (the
S8/B1 verified tier; grouped tiers inherit B1b/S12 tracks).

Case axes covered (IB_LAYER.md §5): the two verif/layer anchor geometries,
GQA (H_kv < H), Qwen rope_theta_base = 1e6, Qwen q/k/v biases, and the S8
self-inclusive q_pos composition.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_GOLDEN = Path(__file__).resolve().parents[3] / "golden"
if str(_GOLDEN) not in sys.path:
    sys.path.insert(0, str(_GOLDEN))

import numpy as np  # noqa: E402

from apex_golden import attention as at  # noqa: E402
from apex_golden import transformer as tf  # noqa: E402

TIER = at.TIER_CQ8

# name, H, head_dim, d_ffn, T, seed, H_kv, theta, bias, q_pos
# (q_pos None => legacy m = T; "self" => S8 self-inclusive duplicated row)
CASES = [
    ("l4_h2_hd64",     2, 64, 256,  8, 0x14B00001, 0, 1e4,  False, None),
    ("l4_h1_hd128",    1, 128, 344, 16, 0x14B00002, 0, 1e4,  False, None),
    ("l4_gqa_h2kv1",   2, 64, 256, 12, 0x14B00003, 1, 1e4,  False, None),
    ("l4_qwen_theta",  2, 64, 256,  8, 0x14B00004, 0, 1e6,  False, None),
    ("l4_bias",        2, 64, 256,  8, 0x14B00005, 0, 1e6,  True,  None),
    ("l4_selfinc",     2, 64, 256,  8, 0x14B00006, 0, 1e4,  False, "self"),
]


def make_weights(rng, H, hd, dff, H_kv=0, theta=1e4, bias=False):
    """Same construction family as verif/layer/gen_layer_vectors.py, extended
    with GQA/theta/bias axes. Non-pow2 s_w* on purpose: the NP-s grading must
    be exercised (a pow-2 s_w is already fp16-grade and would hide it)."""
    D = H * hd
    kvd = (H_kv or H) * hd

    def w8(shape):
        return rng.integers(-127, 128, shape).astype(np.int64)

    sp = 1.0 / (127.0 * math.sqrt(D))
    sd = 1.0 / (127.0 * math.sqrt(dff))
    gam = lambda: np.array(  # noqa: E731
        [int(round((0.7 + 0.6 * rng.random()) * 8192)) for _ in range(D)],
        dtype=np.int64)
    kw = {}
    if bias:
        # fp16-grid biases (the real-model biases arrive on the fp16 bus)
        f16 = lambda v: tf._f16(v)  # noqa: E731
        kw = dict(bq=f16(rng.normal(0, .05, D)),
                  bk=f16(rng.normal(0, .05, kvd)),
                  bv=f16(rng.normal(0, .05, kvd)))
    return tf.LayerWeights(
        Wq=w8((D, D)), Wk=w8((D, kvd)), Wv=w8((D, kvd)), Wo=w8((D, D)),
        Wg=w8((D, dff)), Wu=w8((D, dff)), Wd=w8((dff, D)),
        s_wq=sp * 1.01, s_wk=sp * 0.97, s_wv=sp * 1.0,
        s_wo=sp * 1.03, s_wg=sp * 0.99, s_wu=sp * 1.02, s_wd=sd * 1.01,
        gamma1=gam(), gamma2=gam(), H=H, head_dim=hd, H_kv=H_kv,
        rope_theta_base=theta, **kw)


def build_case(name):
    """(X, w, q_pos) for a named case. Deterministic per seed."""
    for (n, H, hd, dff, T, seed, H_kv, theta, bias, q_pos) in CASES:
        if n != name:
            continue
        rng = np.random.default_rng(seed)
        w = make_weights(rng, H, hd, dff, H_kv=H_kv, theta=theta, bias=bias)
        D = H * hd
        if q_pos == "self":
            # S8 self-inclusive composition: duplicate the decode token as
            # the last context row; q sits at its true position t = T-1.
            Xc = rng.normal(0, 1, (T, D))
            X = np.vstack([Xc, Xc[-1:]])
            return X, w, T - 1
        X = rng.normal(0, 1, (T + 1, D))
        return X, w, None
    raise KeyError(name)
