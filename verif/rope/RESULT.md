# RoPE smoke — RESULT

**Date:** 2026-07-13 (S2 evidence-hygiene re-run)
**DUT:** `rtl/rope/rope.sv` + `rtl/rope/rope_lut_tables.svh` @ working tree
**Command:** `make -C verif/rope/smoke` · **Tools:** Verilator 5.044 + Icarus (both gated)
**Golden arbiter:** `golden/apex_golden/transformer.py` `rope_fx`

## Verdict: PASS — both simulators

Verbatim output (`build/s2_logs/verif_rope_smoke.log`):

```
ROPE SMOKE: ALL TESTS PASSED (t=12290)
  12288/12288 channel-pair vectors bit-exact vs apex_golden rope_fx
  (head_dim in {64,128}, position m in 0..127, HF half-split)
ROPE SMOKE: swept-domain check passed on Verilator AND Icarus
```
