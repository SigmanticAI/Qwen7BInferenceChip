# F2 stage-2 hardware — ✅ SILICON GREEN (2026-07-22, Level-C I-A)

**All 18 committed S8 Qwen2.5-7B attention trace jobs replay BIT-EXACT on
real F2 silicon.** Verbatim session log: [`replay_silicon_PASS.log`](replay_silicon_PASS.log)

```
F2HOST RESULT: files=18 checks=27996 fails=0 -> PASS
```

Same files, same counts as the Verilator sim (`docs/results/f2_stage2_sim/`,
27,996 checks) — one artifact, two executors, both green.

## Configuration (the LC-1 two-clock CL)

- **AFI:** `afi-036d83cafa00d26ea` / AGFI `agfi-0ae06ea568e5667ba`
  (`apex-lc1-a2-20260722`, ingested 22:34→22:57 UTC) · **Instance:**
  f2.6xlarge us-west-2b, terminated after the session · **CL:** cl_apex at
  `comp/level-c-integration` 05efb2a — D=128, G=16, KVQ_DEPTH=256, whole CL
  body on `clk_tile` behind the `apex_ocl_cdc` 4-phase bridge
  (DECISION-LC-1) · **Tile clock:** AWS_CLK_GEN `clk_extra_a1`, recipe A2 =
  **15.625 MHz** (bring-up clock, NOT an Fmax claim).
- **DCP (attempt 5, build `2026_07_22-183309`, 46 m 18 s):** post-route
  timing **MET** — worst overall slack **+0.711 ns** (`clk_main_a0` shell
  path, identical to first light); tile path group `clk_out1_clk_mmcm_a` at
  **period=64.000 ns**, worst in-group slack +3.702 ns. Report:
  [`cl_apex.2026_07_22-183309.post_route_timing.rpt`](cl_apex.2026_07_22-183309.post_route_timing.rpt),
  build-log excerpt: [`dcp_a2_build_excerpt.log`](dcp_a2_build_excerpt.log)
  (the three "Clock recipe … not applied" CRITICAL WARNINGs are the B/C/HBM
  MMCMs — those groups are disabled and have no cells; the A-group applied
  clean). G-I5 ✅.

## Build/bring-up gotchas found (each cost an attempt or nearly did)

1. **The clkgen instance MUST be named `AWS_CLK_GEN`** (attempt 4:
   `u_clkgen` → `aws_clock_properties.tcl`, which hardcodes
   `WRAPPER/CL/AWS_CLK_GEN/...`, silently skipped the A-recipe → tile timed
   at the XCI-default 8 ns/125 MHz → u_squant −31.121 ns VIOLATED).
2. **AFI load resets the clkgen MMCMs to the DEFAULT recipe** (A1: a1 =
   125 MHz) — the bitstream's static MMCM properties and the manifest's
   `clock_recipe_a=A2` are NOT honored at load. The host must program the
   recipe before any BAR0 traffic: `sudo fpga-load-clkgen-recipe -S 0 -a 2`
   (the tool sequences SYS_RST assert → DRP program → MMCM lock wait →
   SYS_RST release). Verified after: a1/a2/a3 = 15.62/125.00/62.50 MHz
   ([`clkgen_final.txt`](clkgen_final.txt)).
3. **The CLI help-text recipe table is WRONG for A2** (prints A0's
   62.5/187.5/250 row twice). Ground truth is the C source
   (`sdk/userspace/fpga_libs/fpga_clkgen/fpga_clkgen_utils.c`):
   `clkgen_a_recipes[2] = {mult 12.500, div 1, div0 80, div1 10, div2 20}`
   → 1250/80 = **15.625 MHz** — byte-identical to the HDK build-side A2
   branch. DECISION-LC-1 stands; trust the source, not `--help`.
4. **Executor parity: the sim resets the CL before EVERY regops file**
   (`sim_main.cpp:198`) — each file's choreography assumes fresh tile/KVQ
   state. On hardware the first run went 16/18 with **zero data
   mismatches**; both `_c1` chunk files stalled their KVQ OCC poll reading
   256 (= 2·128 records left by their `_c0` predecessor — chunk-c0 jobs
   deliberately leave their store populated). Fix landed in
   `f2_host_run.py`: per-file toggle of the bridge's TILE_RST CSR (0x000C,
   gates `rst_n = rst_tile_n & ~tile_rst_q` for tile+mailbox while the
   bridge stays live). Discovery log:
   [`replay1_tile_rst_discovery.log`](replay1_tile_rst_discovery.log).
5. `f2_host_run.py` was validation-deferred and its imagined `fpga_pci`
   python API never existed in the kit — rewritten against the real
   `fpga_pci_wrapper.FpgaPCI` bindings (built by `sdk_setup.sh`), with
   sudo-safe path resolution (`~` is `/root` under sudo). Residual: the
   MMCM-lock preflight matches on the substring "lock" — which "Clock
   Group" satisfies — so it passes vacuously; the real assurance this
   session was the explicit recipe-program + frequency read-back before any
   BAR0 op. Follow-up chip: make the preflight assert a1 ≈ 15.62 directly.

## What ran (scope, honest)

The 18 regops files are the committed S8 real-model trace jobs
(`docs/results/s8_7b_token/artifact_trace`, Qwen2.5-7B, KVQ8/CQ-8 KV tier,
T up to 161 crossing the 128-chunk boundary) compiled by
`trace_to_regops.py` through the VERIFIED L3 choreography: full attention
core per job — K/V rows through the KVQ codec into the record store, scores,
fixed-point online softmax, P·V̂, output serialization, TIP/CSR/sticky-error
checks — driven over BAR0 through the mailbox, on the tile clocked at
15.625 MHz. Debug-tap expects (TAPF16/TAPSC/TAPPR) are skipped and counted,
as in sim (disclosed carve-out; the binding result checks translate 1:1).
Bridge smoke before the run: BAR0 0x0000 = 0x41394558 ("A9EX") through the
CDC — the two-clock bridge answering from silicon.

**Claim unlocked (G-I6 ✅, wording scope-locked):** *real Qwen2.5-7B
attention jobs — the same 18 bit-exact-verified trace jobs the sim
certifies — compute bit-exactly on AWS F2 silicon (VU47P), through the
two-clock CL at a 15.625 MHz bring-up tile clock.* This is
attention-on-silicon for host-sequenced single-tile jobs. It is NOT "7B
runs on the FPGA" (no full-layer/full-model execution on hardware), and
15.625 MHz is a correctness clock, not a performance number.

**Session spend:** devbox ~4.5 h ≈ $2.75 (includes ~3 h idle — a watcher
bug, disclosed) + f2.6xlarge ~1.3 h ≈ $2.60 + S3 ≈ **~$5.50 total.**

---

# History (2026-07-21): D=128 does NOT close 250 MHz (timing fail)

**Date:** 2026-07-21 · **Config:** cl_apex at APEX_CL_D=128, G=16,
KVQ_DEPTH=256, FEED_ROWS_MAX/STAGE_R_MAX=31 (the verified L3 reference build
the 18 stage-2 jobs need) · **Flow:** Vivado 2025.2, F2 Small Shell, DCP
build on the devbox · **Evidence:**
[`post_route_timing_d128_VIOLATED.rpt`](post_route_timing_d128_VIOLATED.rpt)

## Result: post-route timing VIOLATED

```
Slack (VIOLATED) : -34.766 ns   (required time - arrival time, 250 MHz)
Source:      WRAPPER/CL/u_tile/u_squant/out_slot_reg_rep[2]__0_replica/C
Destination: WRAPPER/CL/u_tile/u_squant/pack_reg[38]/D
Data Path Delay: 38.986 ns   (logic 18.495 / route 20.492)
```

Critical path ≈ **38.99 ns ≈ ~26 MHz** — a single intra-`u_squant`
(`apex_scale_quant`) combinational path from an output-slot register through
the fp16 scale-composition/pack logic to a pack register. The checkpoint is
named `post_route.VIOLATED.dcp`.

## Why (honest)

`apex_scale_quant` sizes its per-job element buffers to `max(CFG_D,
T_ROW_MAX)` (apex_top F-1). At D=128 that packing/mux path roughly doubles
vs first light's D=64, and it was **not** pipelined for a 250 MHz target —
first light (D=64) closed 250 MHz with +0.711 ns slack, so no one had hit
this path. It is a build-timing problem, not a functional one: the design is
bit-exact in simulation (`docs/results/f2_stage2_sim/`, 18/18 jobs,
checks=27996 fails=0). The F2 shell drives the CL at a **fixed 250 MHz**
(`clk_main_a0`), so meeting-or-slowing is the only choice — DECISION-F2-1.

## No F2 spend

A timing-violated bitstream computes wrong results at 250 MHz, so no AFI was
created and no f2.6xlarge was launched. Devbox stopped (DCP preserved).
Total spend this session: devbox time only (~$1).

## The fork (DECISION-F2-1, now forced)

1. **AWS_CLK_GEN + CDC** — run the whole CL on a slower AWS_CLK_GEN clock
   (≤~26 MHz, or higher with light `u_squant` pipelining) with a CDC boundary
   only at the shell OCL AXI-Lite (low-rate register traffic). Stage-2 is
   throughput-irrelevant, so a slow clock proves correctness on silicon
   fine. Cost: real RTL (clkgen IP + AXI-Lite CDC synchronizers) + re-verify
   the f2sim under two clocks + rebuild AFI. ~1–2 days.
2. **Pipeline `u_squant`** to close 250 MHz — touches verified tile RTL,
   ~10 stages on that path, high risk to the verification. Not recommended.
3. **Bank the green sim; fold CDC clocking into Level-C** — Level-C hardware
   bring-up (DRAM weight streaming, the walker) needs multi-clock/CDC work
   anyway; do it once there rather than a one-off now. Defers the
   attention-on-silicon claim but avoids a rushed CDC job.

## DECISION (owner, 2026-07-21): **Option 3 (B) — bank the sim, fold CDC into Level-C**

Deeper scoping made the call: the slowest stock AWS_CLK_GEN clock is 62.5 MHz
(recipe B1), which still does not clear the 38.99 ns (~26 MHz) `u_squant`
path — so route A is NOT "just add a slow clock." It requires EITHER
pipelining verified tile RTL (`apex_scale_quant`) with a full KVQ/L3 bit-exact
re-verify behind it, OR a dynamic-clock flow (`fpga-load-clkgen-dynamic`
into the AWS_CLK_GEN MMCMs, ~25 MHz) with real AWS-flow uncertainty — **3–5
days either way**, and every hour duplicates the AWS_CLK_GEN + OCL-AXI-CDC
clocking that Level-C integration must build regardless (walker + DRAM weight
streaming live in the same multi-clock CL).

Nothing external waits on it: the public launch shipped with the FPGA claim
deliberately scope-locked to register first-light (`docs/results/f2_firstlight/`),
and the bit-exact sim (`docs/results/f2_stage2_sim/`, 18/18, 27,996 checks,
0 fails) is banked evidence that the design computes real Qwen-7B attention
correctly. **The attention-on-silicon claim lands the first time the Level-C
CL boots — with clocking built once.**

**Inheritance for the Level-C integration session:** you own the AWS_CLK_GEN
extra-clock + OCL AXI clock-domain-crossing work. The stage-2 CL
(`scripts/fpga/f2/cl_apex/`, D=128/G=16/DEPTH=256, bit-exact in sim) and the
host runner (`scripts/fpga/f2/f2_host_run.py`) are ready to ride on top —
the same 18 regops replay the instant your CDC'd CL closes timing at the
slow clock. Reuse: the recipe table (`tools/aws-fpga/.../Clock_Recipes_User_Guide.md`),
the AWS_CLK_GEN integration pattern in the kit's `cl_mem_perf` example, and
this report's critical-path source (`u_squant/out_slot_reg → pack_reg`).
