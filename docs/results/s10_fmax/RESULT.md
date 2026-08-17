# S10b — routed ship-config Fmax: 34.11 MHz (ECP5-85F, seed 2)

**Date:** 2026-07-18 · **RTL:** post-A2/D-026 + S10a BRAM inference + S10b
pipelined divider, commit eac38ee · **Evidence:**
[`b1_ship_f40s2_route.log`](b1_ship_f40s2_route.log) (verbatim nextpnr log)
· [`BASELINE.md`](BASELINE.md) (the honest pre-S10 re-baseline)

## The number

**Post-route Fmax = 34.11 MHz** for the shipping-capacity KVQ engine
(`kvq_ship_top`: D=64, CQ-4+, G=128, k=2, SETS=8, DEPTH=128) on LFE5U-85F
speed-6, open flow (sv2v → yosys `synth_ecp5 -abc9` → nextpnr-ecp5, `--freq
40 --seed 2`). The 40 MHz constraint is not met and we say so — nextpnr's
final line is `ERROR: Max frequency … 34.11 MHz (FAIL at 40.00 MHz)`; the
routed netlist and bitstream are valid at any clock ≤ 34 MHz.

| milestone | Fmax | what changed |
|---|---|---|
| pre-S10 baseline (proxy D=16 config) | 9.19 MHz | un-pipelined fp16 divider |
| pre-S10 ship config | **unplaceable** (218% FF) | buffers in FFs, no BRAM inference |
| S10a | fits: 23% LUT, 38 DP16KD (18%) | clk-only memory ports → BRAM maps |
| **S10b (this result)** | **34.11 MHz** | divider fully pipelined (II=1) + readback skew cut |

**3.7× over the 9.19 MHz honest re-baseline** (BASELINE.md; the older
9.04 MHz figure was the same proxy measured pre-dedup and is retired), and
the shipping config now both fits and routes. Utilization at route: 19,368 logic LUTs (23%), 4,484
carry LUTs (5%), 38 DP16KD (18%), 1 MULT18X18D, 161 IO (44%).

## The routing-seed finding (worth the two lost days)

Seed 1 placement of this design is pathological: the router enters a
congestion rip-up loop around ~2.5k arcs and degrades monotonically — killed
after **35 h** (`--freq 100`) and again after **11 h** (`--freq 40`), with
per-1,000-iteration route time growing 272 s → 30,093 s. **Seed 2 places and
routes the identical netlist in ~4.5 minutes.** Lesson recorded for the flow:
`s10_baseline.sh` now takes FREQ/SEED, and a stuck route is a seed problem
before it is a frequency problem.

## Claim discipline

- This number never leads externally — the campaign leads on verification
  depth and honest accounting; 34 MHz is disclosed, not celebrated
  (claim-discipline rules: `docs/launch/CAMPAIGN.md` §4).
- The committed public artifacts are regenerated from THIS route:
  `docs/results/kvq_engine_ecp5-85f.bit` (ecppack of the seed-2 cfg) +
  `docs/results/kvq_ecp5_report.txt` — the linked evidence now matches the
  stated number (launch gate G1 requirement).
- Not comparable to the AWS F2 result (VU47P closes 250 MHz on the same RTL
  — different part, different toolchain; see
  `docs/results/f2_firstlight/RESULT.md`).

## Reproduce

```sh
# needs yosys/sv2v + an oss-cad-suite (nextpnr-ecp5, ecppack) on PATH
FREQ=40 SEED=2 bash scripts/fpga/s10_baseline.sh     # writes build/fpga/s10/
ecppack --input build/fpga/s10/b1_ship.cfg --bit kvq_engine_ecp5-85f.bit
```
