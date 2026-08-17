# DDR BRING-UP — REAL CARD MEMORY TRAINS, LOADS, AND HOLDS THE MODEL

**Date:** 2026-08-05 · **Image:** `afi-0d15cb7f7422bd69b` / `agfi-0a345ddb51285e847`
(apex-b64-ddr1-20260805; first DDR-enabled APEX image — TIMING MET WNS +0.392,
real CDC constraints, PRV-GREEN, ingested first try) · **Instance:**
`i-...` f2.6xlarge (terminated + verified) · **Tools:** `make_weight_image.py`,
`f2_ddr_load.py` (first hardware use of both).

## Results (all measured on the card)

```
PREFLIGHT  FUEL_STAT=0x9  (ddr_ready=1, ar_owner=LOAD)  FUEL_ERR=0x0
DDRLOAD    29,842,432 bytes, 468 BAR4 bursts, 0.28 s  (101.22 MB/s)
DDRVERIFY  (full) tensors=20 words_read=466,288 fails=0 -> PASS
FUEL_ERR   after: 0x0
```

- **AWS's real (encrypted) DDR4 controller TRAINS** under our CL — the
  behavioral model's biggest untested assumption, retired on silicon.
- **Real Qwen2.5-0.5B tensors (2 layers, 20 tensors, 28.5 MiB) are resident
  in card DRAM** in the walker's tensor-table layout, every word verified by
  full readback.
- The BAR4 burst loader works first try at 101 MB/s.

## Scope fence

This session proves TRAIN + LOAD + HOLD. It does NOT prove compute-from-DDR
on hardware — that claim belongs to the convergence image (E-5 walked fuel
path with its discriminators), deliberately not improvised here on pre-E-5
RTL. Sim proof of compute-from-DDR: fuel_proj_05b (host-driven) and
walk_fuel_proj (walker-driven), both bit-exact with one-byte poison RED.

## Cost

~15 min f2.6xlarge ≈ $0.45. Terminated + verified.
