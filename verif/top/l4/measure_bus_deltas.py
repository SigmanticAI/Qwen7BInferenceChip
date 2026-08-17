#!/usr/bin/env python3
"""measure_bus_deltas.py — C-LBUS (D-030) pinned-delta measurement.

B3-finding-3 style: measure what each bus flag costs, per L4 case, against
(a) the legacy decoder_layer_fx output and (b) the float64 yardstick
decoder_layer_ref ON THE SAME INPUTS the pipeline saw (fp16 inputs for the
bus pipeline — comparing a bus run against the un-narrowed-input ref would
double-charge NP-x to the mode).

Output: the table pasted into docs/results/ib_layer_bus/RESULTS.md; the
BUS_ON e2e numbers are pinned as regression registers in
golden/tests/test_layer_bus.py section D.

Repro:  python3 verif/top/l4/measure_bus_deltas.py
"""
from __future__ import annotations

import numpy as np

from l4_cases import CASES, TIER, build_case, tf


def maxrel(a, b):
    """max |a-b| / max|b| (the projection-error metric family B3 used)."""
    den = float(np.max(np.abs(b)))
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / (den or 1.0)


def run(X, w, q_pos, bus):
    return tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=bus)


def main():
    modes = {
        "x":     tf.BusMode(x_f16=True),
        "rope":  tf.BusMode(rope_in_f16=True),
        "grade": tf.BusMode(scale_f16grade=True),
        "ON":    tf.BUS_ON,
    }
    hdr = (f"{'case':14s} {'d(x)':>9s} {'d(rope)':>9s} {'d(grade)':>9s} "
           f"{'d(ON)':>9s} {'err_leg':>9s} {'err_ON':>9s}")
    print("C-LBUS PINNED-DELTA MEASUREMENT (e2e r2, max-rel)")
    print(hdr)
    pins = {}
    for (name, *_rest) in [(c[0],) for c in CASES]:
        X, w, q_pos = build_case(name)
        leg = run(X, w, q_pos, None)
        outs = {k: run(X, w, q_pos, m) for k, m in modes.items()}
        d = {k: maxrel(o.r2, leg.r2) for k, o in outs.items()}
        # yardstick errors, each vs the ref on its OWN inputs
        ref_leg = tf.decoder_layer_ref(X, w, q_pos=q_pos)
        ref_bus = tf.decoder_layer_ref(tf._f16(X), w, q_pos=q_pos)
        err_leg = maxrel(leg.r2, ref_leg.y_ref)
        err_on = maxrel(outs["ON"].r2, ref_bus.y_ref)
        pins[name] = (d["ON"], err_leg, err_on)
        print(f"{name:14s} {d['x']:9.3e} {d['rope']:9.3e} {d['grade']:9.3e} "
              f"{d['ON']:9.3e} {err_leg:9.3e} {err_on:9.3e}")

    # per-checkpoint attribution on the first anchor, BUS_ON
    name = "l4_h2_hd64"
    X, w, q_pos = build_case(name)
    leg, on = run(X, w, q_pos, None), run(X, w, q_pos, tf.BUS_ON)
    print(f"\nper-checkpoint max-rel delta, {name}, BUS_ON vs legacy:")
    for f in ("h", "q_real", "K_real", "V_real", "q_rope", "K_rope", "attn",
              "attn_proj", "r1", "h2", "gate_real", "up_real", "swiglu",
              "ffn_out", "r2"):
        print(f"  {f:10s} {maxrel(getattr(on, f), getattr(leg, f)):9.3e}")

    print("\nPIN REGISTERS (paste into test_layer_bus.py section D):")
    for k, (don, el, eo) in pins.items():
        print(f'    "{k}": ({don:.6e}, {el:.6e}, {eo:.6e}),')


if __name__ == "__main__":
    main()
