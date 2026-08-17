# ASU SiLU exhaustive smoke — RESULT

**Date:** 2026-07-13 (S2 evidence-hygiene re-run; counts first reported in commit `95275c5`)
**DUT:** `rtl/asu/asu_silu.sv` + `rtl/asu/silu_lut_tables.svh` @ working tree
**Command:** `make -C verif/asu/silu` · **Tools:** Verilator 5.044 + Icarus (both gated)
**Golden arbiter:** `golden/apex_golden/transformer.py` `silu_fx`

## Verdict: PASS — exhaustive full-domain, both simulators

Verbatim output (`build/s2_logs/verif_asu_silu.log`):

```
silu oracle: 65536 entries -> build/silu_expected.hex (range [-1140,32757], lim=8192)
ASU SILU SMOKE: ALL TESTS PASSED (t=65541000)          <- Verilator
  65536/65536 input patterns bit-exact vs apex_golden silu_fx
ASU SILU SMOKE: ALL TESTS PASSED (t=65541)             <- Icarus
  65536/65536 input patterns bit-exact vs apex_golden silu_fx
ASU SILU: exhaustive-domain check passed on Verilator AND Icarus
```

All 2^16 input bit patterns checked — this is the complete input domain, not a sample.
