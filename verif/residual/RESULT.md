# residual_add fp16 smoke — RESULT

**Date:** 2026-07-13 (S2 evidence-hygiene re-run; counts first reported in commit `95275c5`)
**DUT:** `rtl/misc/residual_add.sv` @ working tree
**Command:** `make -C verif/residual` · **Tools:** Verilator 5.044 + Icarus (both gated)
**Golden arbiter:** numpy fp16 add (C-6 residual contract)

## Verdict: PASS — both simulators

Verbatim output (`build/s2_logs/verif_residual.log`):

```
residual oracle: 20012 fp16 add vectors (12 directed + 20000 random) -> build
RESIDUAL_ADD SMOKE: ALL TESTS PASSED (t=20014000)      <- Verilator
  20012/20012 fp16 residual-add vectors bit-exact vs numpy fp16
RESIDUAL_ADD SMOKE: ALL TESTS PASSED (t=20014)         <- Icarus
  20012/20012 fp16 residual-add vectors bit-exact vs numpy fp16
RESIDUAL_ADD SMOKE: passed on Verilator AND Icarus
```
