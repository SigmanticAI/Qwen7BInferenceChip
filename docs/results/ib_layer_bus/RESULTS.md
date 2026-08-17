# C-LBUS (D-030) — Bus-Grade Composition Mode: Pinned-Delta Measurement

**Date:** 2026-07-22 · **Lane:** IB-LAYER S1 (`docs/design/IB_LAYER.md` §2b)
· **Ruling:** LEVEL_C_INTEGRATION.md §9.1 **R4** (design-approved; decision
number **D-030**) · **Style:** B3-finding-3 pinned-delta register
(`docs/design/B3_WEIGHT_PATH.md` §2 finding 3 precedent).

> **OWNER SIGN-OFF: PENDING.** Everything below is measured and gated in
> `golden/tests/test_layer_bus.py`, but per R4 **no RTL stage may gate
> against `BUS_ON` until the owner sign-off is recorded in
> `docs/design/IB_LAYER.md`.** The flags default OFF and the OFF path is
> byte-identical (section A gate), so landing this mode changes nothing
> until it is deliberately consumed.

## 1. What landed

`golden/apex_golden/transformer.py` gains `BusMode` / `BUS_OFF` / `BUS_ON`
and a keyword-only `bus=` parameter on `decoder_layer_fx` (default `None` ==
legacy, bit-identical — no existing golden function's default behaviour
changed; `decoder_layer_ref` untouched). Three flags:

| flag | narrowing point | hardware anchor |
|---|---|---|
| `x_f16` | layer-input rows enter on the fp16 bus (steady-state no-op: the real input IS the previous layer's fp16 `r2`) | residual row RAM / C-6 bus |
| `rope_in_f16` | q/K rows (bias already added, HF `rope(Wx+b)` order) narrowed to the S-2 bus BEFORE `rope_fx`; V untouched (its S-2 narrowing already exists at the attention-core input) | projection → S-2 dequant → `rope_row` → KVQ store |
| `scale_f16grade` | every host-computed dequant composite `f16_grade()`d (`weight_codec.f16_grade`, the C2 primitive): Q/K/V composites `s_h·s_w*`, per-head attn output `o8·s_out`, `_proj_epilogue` `s_out` (Wo, Wd), FFN gate/up composites | `apex_scale_quant` C2 (`scale_error` on non-graded sidebands) |

**There is deliberately NO fourth flag** for the residual operands / SwiGLU
`up` operand. With `scale_f16grade` set, every value feeding a single-RNE
narrowing is EXACT in float64 (o8 ≤ 8 significand bits × graded ≤ 11;
`acc` < 2³¹ × graded ≤ 11; fp16 × those ≤ 53), so the narrowings already in
`decoder_layer_fx` are directly RTL-realizable with one RNE each. That is
the **section-C lemma — 2,008 exact-Fraction checks** over three cases
(per-head `attn == o8·graded` reconstruction, both residual sums, every
`silu·up` product), not prose. v1 guard: `scale_f16grade` refuses
`T > CHUNK_T_MAX` (chunked-merge bus composition explicitly out of scope).

## 2. Measured deltas (the pinned registers)

`python3 verif/top/l4/measure_bus_deltas.py`, verbatim:

```
C-LBUS PINNED-DELTA MEASUREMENT (e2e r2, max-rel)
case                d(x)   d(rope)  d(grade)     d(ON)   err_leg    err_ON
l4_h2_hd64     2.634e-03 0.000e+00 3.762e-03 3.386e-03 4.144e-03 3.673e-03
l4_h1_hd128    3.058e-03 0.000e+00 3.147e-03 3.058e-03 3.524e-03 3.498e-03
l4_gqa_h2kv1   3.089e-03 0.000e+00 2.180e-03 2.907e-03 4.156e-03 4.080e-03
l4_qwen_theta  4.017e-03 0.000e+00 2.295e-03 3.826e-03 5.928e-03 3.664e-03
l4_bias        2.123e-03 0.000e+00 1.783e-03 2.404e-03 3.056e-03 2.619e-03
l4_selfinc     2.552e-03 0.000e+00 2.418e-03 3.224e-03 7.195e-03 6.376e-03
```

Columns: per-flag and BUS_ON e2e `r2` delta vs legacy (max-rel), then the
float64-yardstick error of legacy vs `decoder_layer_ref(X)` and of BUS_ON vs
`decoder_layer_ref(_f16(X))` — each pipeline against the ref on the inputs
it actually saw, so NP-x is not double-charged.

**Findings:**

1. **BUS_ON costs ~2–4e-3 e2e at tile scale — the same order as the
   pipeline's own distance from the float64 yardstick — and `err_ON ≤
   err_leg` on ALL six cases** (asserted with a 1.10 guard in section D).
   The mode re-quantizes; it does not degrade. (Tile-scale claim only —
   the CAMPAIGN §3-§4 qualifier applies; model-scale neutrality is a
   separate, later question on the real-weights pipeline.)
2. **`d(rope) = 0 e2e on all six cases — but the flag is NOT a no-op.**
   Checkpoint-level, rope-only: q̂/K̂ deltas 4–9e-4 with **33/254 fp16 words
   differing (q̂/K̂, anchor case)**, then `attn` comes back **bit-identical**
   — the ±1-ULP bus perturbations are absorbed by the KVQ store + feeder
   INT8 re-quantization in every case measured. Pinned as a measured fact
   (word counts + bit-identical attn), NOT claimed as a theorem. This is
   exactly why the L4 TB must check the q̂/K̂ **bus** checkpoints — where
   `rope_row` actually lives — and not only e2e.
3. Per-checkpoint BUS_ON attribution (anchor `l4_h2_hd64`): largest single
   jump is `ffn_out` at 2.0e-02 — the down-proj `calib_requant` pair moves
   when its `acc` shifts, an amplitude-jitter effect that `r2` absorbs back
   to 3.4e-03. Full table in the script output.

**Pin enforcement:** `golden/tests/test_layer_bus.py` section D pins
`(d_ON, err_leg, err_ON)` per case at rtol 1e-6 plus the rope-absorption
word counts — any drift forces a conscious re-measure + re-pin.

## 3. Gate evidence (S1, run 2026-07-22)

`make -C golden test` → rc=0, tail verbatim:

```
A: OFF bit-identity — legacy/None/BUS_OFF byte-identical on 17 fields + per-head (o8, s_out), 4 cases
B: graded s_out idempotent (and grading exercised), BUS_ON deterministic, r2 fp16-grid, chunked guard fires
C: realizability lemmas — 2008 exact-Fraction checks: attn o8*graded, both residual sums, silu*up all EXACT in f64 (single-RNE realizable); no extra narrowing flag needed
D: pinned registers — 6 cases x (d_ON, err_leg, err_ON) within rtol 1e-06; err_ON <= err_leg on all; rope-only absorption pinned (q̂/K̂ words 33/254 differ, attn bit-identical)
APEX LAYER-BUS GOLDEN (C-LBUS/D-030): OFF BIT-IDENTICAL, LEMMAS EXACT, DELTAS PINNED — ALL PASS
GOLDEN SUITE: contract + compute + attention + transformer + plumbing7b + effbits + masksem ALL PASS
GOLDEN SUITE (B3): + weightcodec ALL PASS
GOLDEN SUITE (IB-LAYER): + layerbus ALL PASS
```

The main banner line is byte-identical (the `gen_status.py` literal); the
new gate reports on its own line — the B3 pattern. No Verilator suite was
run for this change: no RTL was touched, and the legacy compute path is
byte-identical by the section-A gate (the L3/L2/smoke generators consume
`attention.py`, which is untouched); the full-matrix byte-identity re-check
happens at the next RTL-touching stage per the lane gates.

## 4. Repro

```
make -C golden test                          # incl. the layerbus gate
python3 verif/top/l4/measure_bus_deltas.py   # the tables above
python3 verif/top/l4/gen_l4_vectors.py       # S1 skeleton: 6 cases, 41,802
                                             # checkpoint words, determinism
                                             # self-check; op stream = S4 stub
```
