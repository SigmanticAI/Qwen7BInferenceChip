# Decoder-layer composition TB — RESULT

**Date:** 2026-07-13 (S2 evidence-hygiene re-run; first reported in commit `95275c5`)
**DUT:** `rtl/rope/rope.sv` + `rtl/asu/asu_silu.sv` + `rtl/misc/residual_add.sv` driven with the
bit-exact stage tensors of a real `apex_golden.transformer.decoder_layer_fx` decode step, in layer order
**Command:** `make -C verif/layer` (runs the golden-gate first, then Verilator AND Icarus)
**Golden arbiter:** `golden/apex_golden/transformer.py`

## Verdict: PASS — golden gate + both simulators

Verbatim output (`build/s2_logs/verif_layer.log`):

```
APEX ATTENTION GOLDEN: ALL 120 CASES WITHIN DERIVED BUDGET (hard + statistical), ALL SELF-CONSISTENCY CHECKS PASS
APEX DECODER-LAYER GOLDEN: ALL RoPE/SiLU LEMMAS + 9 LAYER CASES WITHIN DERIVED BUDGET, DETERMINISM OK
GOLDEN SUITE: contract + compute + attention + transformer ALL PASS
LAYER COMPOSITION: ALL TESTS PASSED (t=2776000)        <- Verilator
LAYER COMPOSITION: ALL TESTS PASSED (t=2776)           <- Icarus
```

Layer-case error table (worst e2e/|y| 1.9e-02 across H2/hd64 and H1/hd128 configs, d_ffn ≤ 344,
T ≤ 64 — tile scale, NOT model scale; see docs/launch/CAMPAIGN.md §3-§4 for the mandatory "tile scale" qualifier).
