# CLOCK_LADDER — can the b64_05B CL close timing at a 62.5 MHz tile clock?

**Date:** 2026-08-04, **revised 2026-08-05** against the post-E-5/E-6/DDR
tree (which added the walked fuel path, the on-tile requant epilogue, real
DDR, and — critically — the discovery that the CDC constraints had never
been read by any build, §1.2a) · **Method:** static analysis only (no
builds, no AWS, no hardware; every claim below carries a file:line receipt
from this tree, the committed result artifacts, or the pinned kit checkout
symlinked at `tools/aws-fpga`) · **Branch:** `comp/prompt-b-c` ·
**Landed by the revision:** the §7.2 host-side image-keying
(`scripts/fpga/f2/clock_key.py` + keyed gates in `remote_hw_exec.py` /
`apex_repl.py`, selftested) and the §8 ladder as a runnable script
(`scripts/fpga/f2/build_a0_62p5.sh`).

---

## 0. Verdict (30 seconds)

1. **Timing: very likely YES.** The b64_05B tile is the D=64 lineage, and a
   D=64 elaboration of this tile already closed **250 MHz** (4 ns) at first
   light with +0.711 ns to spare (§5.1). 62.5 MHz is a 16 ns period — 4×
   that proven budget. The blocks added since first light (E-lane residual /
   swiglu / rope / wide-norm, walker, GQA=2) have only ever been *constrained*
   at 64 ns, but their deepest cones are DSP-multiply + fp16-pack class
   (§6), well inside 16 ns on VU47P. Risk is nonzero only because **this
   exact geometry has never been built at 16 ns** — the check is one ~46 min
   devbox build whose timing report is a hard, loud gate.
2. **Premise corrections (three, load-bearing):**
   - The recipe that puts 62.5 MHz on `clk_extra_a1` is **A0, not "B1"**
     (§2.1). "B1" is a *group-B* recipe (62.5 MHz on `clk_extra_b1`, a clock
     this CL does not use); the "62.5 (B1)" naming in the older docs and the
     f2sim Makefile comment is a fossil from the pre-LC-1 scoping.
   - **There is no 31.25 MHz rung.** Stock group-A recipes are exactly
     A0=62.5 / A1=125 / A2=15.625 on a1 (§2.2). The ladder is binary:
     A2 (proven) or A0 (candidate). Dynamic clkgen bottoms out at 87 MHz.
   - The **+0.711 ns "wide-build slack"** cited in ESCALATION.md is the
     *shell* `clk_main_a0` path at 250 MHz — a constant of every green
     build, including first light — and says **nothing** about tile margin
     (§5.4).
3. **What a 62.5 image requires:** *no RTL change, no constraint-file change,
   no build-script edit.* The tile constraint derives automatically from
   `--clock_recipe_a` (§1.3), and `build_dcp.sh` forwards `"$@"` so
   `--clock_recipe_a A0` can be appended at invocation (the kit parses with
   **optparse** default-store semantics — last occurrence wins, and the kit
   echoes the effective value in its option banner, §7.1). The host-side
   image-keying the first edition demanded is now **LANDED** (§7.2):
   `clock_key.py` binds expected-a1-MHz to the AGFI, `remote_hw_exec` /
   `apex_repl` enforce it, and the selftests prove an A2 image cannot be
   flown at 62.5 nor an A0 image silently at 15.625.
4. **Honest speedup — regime-dependent, and the regime just changed.**
   In the *host-dispatched* regime the first edition measured, the 4× clock
   is worth **≤~3%** end-to-end (dispatch dominates: 0.215–0.26 s/job wall
   vs 1–5 ms tile-bound, §9.1). But **E-5/E-6 removed the per-op host
   dispatch inside a walk window**: the whole fuel-fed projection ran under
   ONE host poll on silicon, and the E-6 doc's labelled prediction is
   **~4.4 M tile cycles for a fully-walked layer ⇒ ~280 ms/layer at 15.625
   vs ~70 ms at 62.5 MHz (~1.7 s/token)** — in walked mode the tile clock
   ≈ the wall clock, and the 4× is real (§9.3). Caveat that keeps the claim
   honest: a fully-walked layer is NOT yet achievable on today's RTL
   (8-of-13 step families walk, E6_ON_SILICON.md "Honest scope"; I-B gap
   list) — so book A0 as *enabling infrastructure for the walked regime*,
   measured there, never as a demo-wide 4×.

---

## 1. How the tile clock reaches apex_top on the F2 CL

### 1.1 RTL path (synthesis branch)

- The shell delivers fixed clocks per `cl_ports.vh`; `clk_main_a0` is fixed
  250 MHz for all shell interfaces (`scripts/fpga/f2/BRINGUP.md:33`).
- `cl_apex.sv` instantiates the kit clock generator — instance name
  **`AWS_CLK_GEN`**, which is load-bearing: `aws_clock_properties.tcl`
  hardcodes `WRAPPER/CL/AWS_CLK_GEN/...`; any other name silently skips the
  recipe and the tile is timed at the XCI default 8 ns/125 MHz
  (`scripts/fpga/f2/cl_apex/design/cl_apex.sv:135-143`; proven the hard way
  in attempt 4, `docs/results/f2_stage2_hw/RESULT.md:34-37`).
- **`o_clk_extra_a1` → `clk_tile`** (`cl_apex.sv:167`). The *entire* CL body
  — bridge FSM (`cl_apex.sv:189,422`), mailbox (`cl_apex.sv:601`), and the
  tile itself, `u_tile` / `apex_top` at `.clk(clk_tile)`
  (`cl_apex.sv:661`, comment "DECISION-LC-1") — runs on `clk_tile`.
  Only the OCL AXI-Lite front (via `apex_ocl_cdc`, `cl_apex.sv:206-227`),
  sh_ddr, and the PCIS decode stay on `clk_main_a0`.
- Simulation branch: in-RTL divider under `APEX_SIM_TILE_DIV`, ratio
  `+tile_div=N` → 250/(2N) MHz (`cl_apex.sv:81-101`).

### 1.2 Where the frequency is constrained

The repo's own constraint file carries **only CDC exceptions** and says so:
"The recipe clocks themselves (clk_extra_a1 @ 15.625 MHz under
`--clock_recipe_a A2`) are created by the kit's generated clock constraints —
do NOT re-create them here"
(`scripts/fpga/f2/cl_apex/constraints/cl_synth_user.xdc:1-11`).
`synth_cl_apex.tcl` reads that xdc LATE and contains **no create_clock**
(`scripts/fpga/f2/cl_apex/build/scripts/synth_cl_apex.tcl:39-42`).

Kit-side mechanics (pinned kit v2.3.3, `tools/aws-fpga`):

- `aws_gen_clk_constraints.tcl` writes `create_clock` **only for
  `clk_main_a0` (4 ns) and `clk_hbm_ref`** into
  `generated_cl_clocks_aws.xdc`
  (`hdk/common/shell_stable/build/scripts/aws_gen_clk_constraints.tcl:176-196`);
  the extra-clock period variables it sets are vestigial for group A — and
  its A2 row is *wrong anyway* (`clk_extra_a1_period 128`, i.e. 7.8125 MHz,
  at `aws_gen_clk_constraints.tcl:48`), which is harmless only because it is
  never emitted for a1.
- The **actual** tile constraint is the Vivado *derived clock* of the
  A-group MMCM, whose physical configuration `aws_clock_properties.tcl`
  sets from `$clock_recipe_a`: A2 branch = mult 12.5, `CLKOUT0_DIVIDE_F 80`
  → a1 = 1250/80 = **15.625 MHz / 64 ns**
  (`hdk/common/shell_stable/build/scripts/aws_clock_properties.tcl:80-89`).
  The routed report confirms: the tile path group is
  `clk_out1_clk_mmcm_a {rise@0.000ns fall@32.000ns period=64.000ns}`
  (`docs/results/f2_stage2_hw/cl_apex.2026_07_22-183309.post_route_timing.rpt:1000`).

**Consequence:** the build constrains a1 **automatically from the recipe
flag**. Switching A2→A0 retimes the tile group from 64 ns to 16 ns with no
TCL edit (A0 branch: mult 15, div0 24 → 1500/24 = 62.5 MHz,
`aws_clock_properties.tcl:90-111`).

### 1.2a REVISION FINDING — the CDC constraints were never read before 2026-08-05

Discovered the evening this doc first shipped (commits `9b131a2` →
`fedac38` → `89c280d`): **every build up to and including
`cl_apex.2026_08_05-012859` parsed the kit's stock placeholder xdc, not
ours** — the CL-dir assembly step never copied `cl_synth_user.xdc`, so
routed DCPs carried ZERO ASYNC_REG cells and the tile↔shell crossings were
classified "No Common Phase / Timed (unsafe)"
(`scripts/fpga/f2/cl_apex/constraints/cl_synth_user.xdc:9-19` records the
whole incident). Consequences for this ladder:

- Every §5 row dated before 2026-08-05 (first light, I-A, wide probe) was
  built **without** our CDC exceptions. Their MET verdicts stand — the
  bogus synchronous timing of the crossings was *tighter* than the real
  requirement — but their netlists lack metastability hardening.
- `synth_cl_apex.tcl` now reads the xdc **from APEX_REPO_DIR** and refuses
  to build if it is missing (`synth_cl_apex.tcl:51-53`), and a **CDC gate**
  errors the build unless every synchronizer group actually matched cells
  (`synth_cl_apex.tcl:147-190`, "APEX CDC gate OK"). The A0 build inherits
  this corrected flow; `build_a0_62p5.sh check-timing` gate G2 asserts both
  lines in the log.
- The first build that really read the file, `cl_apex.2026_08_05-032359`
  (DDR=1, b64 geometry): **WNS +0.392, TNS 0.000, MET**, crossings now
  "Max Delay Datapath Only", and the worst path in the whole design moved
  **inside AWS's DDR4 core** on its own intra-domain clock
  (`cl_synth_user.xdc:21-36`). That last fact matters for reading the A0
  report — see the §5 red-herring note.

### 1.3 What the existing builds were constrained at

Every recorded two-clock build of this CL passed `--clock_recipe_a A2`
(64 ns tile):

- `scripts/fpga/f2/build_dcp.sh:22-23` and
  `scripts/fpga/f2/devbox_setup.sh:64-66` both pin
  `--aws_clk_gen --clock_recipe_a A2` (the pair is hard-gated by the kit:
  recipe flags abort without `--aws_clk_gen`, `build_dcp.sh:17-18`).
- I-A A2 build `2026_07_22-183309`: timing MET, tile group at 64 ns
  (`docs/results/f2_stage2_hw/RESULT.md:22-30`).
- Wide probe builds (D=128/DM=3584): recipe A2, tile group 64 ns
  (`docs/results/ic_dcp_probe/RESULT.md:4,45`, `RESULT_ROUTED.md:5,15`).
- The **b64_05B image lineage** ("b64v4" in session parlance): the tree
  holds **no local timing report** for any b64-family DCP — the reports
  live on the devbox EBS. The repo evidence is: the first b64_05b AFI was a
  mislabeled D=128 hybrid (commit `25ddb66`, caught on-card by the INFO_D
  identity read), and the corrected D=64/DMODEL=64/GQA=2/DM_MAX=896/
  QSTAGE=14 image **flew on 2026-08-03** — 4,380 programs on silicon, all
  bit-exact, INFO_D==64 audited 4,380/4,380, with the per-invocation clock
  gate ON, i.e. at **15.625 MHz = recipe A2**
  (`build/p05b_hw_check2.log`, image line + geometry-audit line + verdict).
  Since both build entry points pin A2 and the numeric clock gate passed at
  15.625, the existing b64 image is an **A2/64 ns-constrained** image; its
  timing report (devbox EBS) can say nothing about 16 ns closure.
- **REVISION — three newer A2 images** (all b64 geometry, all
  DDR=1, all with the real CDC constraints, all "clkgen A2"):
  DDR bring-up `agfi-0a345ddb51285e847` (WNS +0.392,
  `docs/results/prompt_on_chip/DDR_BRINGUP_RESULT.md`); E-5 convergence
  `agfi-0183a4b88c8d21163` (Slack MET +0.277,
  `SELF_RUNNING_CARD_RESULT.md`); E-6 convergence2
  `agfi-0bc20880b50f5faba` (Slack MET +0.297, `E6_ON_SILICON.md`). The E-6
  image is the geometry the A0 recipe pins (§7.1) — it carries the walked
  epilogue + fuel + DDR logic that postdates every earlier timing datum.

---

## 2. Premise corrections

### 2.1 The 62.5-on-a1 recipe is A0, not "B1"

Ground truth, per the project's own discipline of trusting source over help
text (`docs/results/f2_stage2_hw/RESULT.md:45-50`; memory: the CLI `--help`
table prints A0's row for A2):

- SDK C table `sdk/userspace/fpga_libs/fpga_clkgen/fpga_clkgen_utils.c:75-79`
  (input 100 MHz, VCO = 100·mult):
  | index | mult | div0/div1/div2 | a1 | a2 | a3 |
  |---|---|---|---|---|---|
  | 0 (**A0**) | 15 | 24/8/6 | **62.5** | 187.5 | 250 |
  | 1 (A1, load default) | 15 | 12/4/3 | 125 | 375 | 500 |
  | 2 (A2) | 12.5 | 80/10/20 | 15.625 | 125 | 62.5 |
- HDK build table agrees: A0 → clk_extra_a1 = 62.5
  (`tools/aws-fpga/hdk/docs/Clock_Recipes_User_Guide.md:28-30`), and the
  build-side MMCM branch matches (`aws_clock_properties.tcl:90-111`).
- "B1" is a **group-B** recipe: mult 11.875, div0 9.5, div1 19 → b0 = 125,
  **b1 = 62.5** (`fpga_clkgen_utils.c:85`; `aws_clock_properties.tcl:117-121`).
  Group B is *disabled* in this CL (`CLK_GRP_B_EN(0)`, `cl_apex.sv:137`) and
  b-clocks are unconnected (`cl_apex.sv:170-171`). The old lines
  "slowest stock AWS_CLK_GEN recipe is 62.5 MHz (B1)"
  (`docs/design/LEVEL_C_INTEGRATION.md:109`,
  `docs/results/f2_stage2_hw/RESULT.md:153-154`) predate the
  DECISION-LC-1 recon that surfaced A2; they name the right *frequency* and
  the wrong *recipe letter for this CL's clock*. A 62.5 MHz tile is reached
  on **a1 with recipe A0** — same wire, same constraints surface, same
  MMCM the A2 flow already exercises. *(Revision: the
  `verif/f2sim/Makefile:29` copy of the fossil is corrected; the one in
  `walk_fuel_layer.py:229` — "2 is the recipe-B1 sim ratio" — is owned by
  the walker track and still stands, comment-only.)*

### 2.2 There is no intermediate rung (31.25 is not a thing)

`aws_build_dcp_from_cl.py` accepts exactly `A0|A1|A2`
(`hdk/common/shell_stable/build/scripts/aws_build_dcp_from_cl.py:278-280`);
the runtime table has exactly 3 A-rows (`fpga_clkgen_utils.c:73`). Between
15.625 and 62.5 there is nothing stock. The alternatives to get ~31 MHz are
all off-ladder: dynamic clkgen is DEAD (min 87 MHz — sdk FREQ table, cited
at `cl_apex.sv:115-116` and `docs/design/LEVEL_C_INTEGRATION.md:118`), and a
custom MMCM/XCI or kit-TCL edit would create a novel constraint surface in
exactly the file family implicated in the HDPRVerify-41 saga — not worth it
for a 2× step. **The decision is binary: stay at A2 or build A0.** If A0
fails timing, the fallback is A2 (status quo), not a half-step.

---

## 3. The two-clock CDC: frequency-agnostic by construction

### 3.1 Mechanisms (both ratio-free)

- **OCL control path** — `apex_ocl_cdc.sv`: 4-phase **toggle req/ack**
  handshake, single-outstanding (matching the strictly single-outstanding
  downstream FSM), payload latched stable before req toggles and held until
  ack returns; 2FF synchronizers both directions
  (`scripts/fpga/f2/cl_apex/design/apex_ocl_cdc.sv:1-33`). A 4-phase
  handshake is correct at *any* clock ratio in either direction — there is
  no counter, divider, or sampling assumption anywhere in the protocol. The
  only frequency-sensitive element is the shell-side dead-clock guard
  (2^16 shell cycles ≈ 262 µs, `apex_ocl_cdc.sv:24-31,37`) — at 62.5 MHz
  the tile answers *sooner*, moving away from the guard.
- **Fuel data path** — `apex_afifo.sv`: standard Cummings dual-clock FIFO,
  gray-coded pointers, 2FF sync, FWFT read
  (`scripts/fpga/f2/cl_apex/design/apex_afifo.sv:1-31`) — the textbook
  ratio-free crossing; plus a 4-phase `fuel_req` return channel
  (constraints block at `cl_synth_user.xdc:135-161`).
- **Constraints carry no ratio either** *(re-verified on the post-fedac38
  file, the one builds actually read now, §1.2a)*: ASYNC_REG on every
  meta/sync pair; `set_max_delay -datapath_only 4.0` on single-bit toggles
  into their 2FF (`cl_synth_user.xdc:61-66,142-150`); `-datapath_only 16.0`
  on the quasi-static shell→tile payloads (`:70-81`) and on the afifo RAM
  fuel datapath (`:130-133`, constrained via CLK-pin startpoints since
  LUTRAM cells are not valid startpoints); `-datapath_only 4.0` on
  `eng_rec_q` tile→shell (`:161` — destination is the 4 ns shell domain,
  recipe-independent). At A0 the 16.0 payload bounds become exactly one
  destination-tile period, and every protocol stability guarantee they lean
  on (≥2 synchronized pointer/handshake updates) is denominated in tile
  cycles, i.e. ≥32 ns at 62.5 MHz — still ≥2× the bound. The file also
  resolves the tile clock **from the netlist**, never by the
  recipe-dependent generated-clock name (`:44-53` — `clk_out1_clk_mmcm_a`
  is an IP-generated name and a literal would silently stop matching).
  **No xdc edit is needed for A0.**
- The tile-domain reset is a 2FF synchronizer off live `clk_tile` edges
  (`cl_apex.sv:188-192`, false-path at `cl_synth_user.xdc:36-39`) — ratio
  free.

### 3.2 The "bit-exact at ALL ratios" proof inventory — what actually ran at div 2

The discipline is stated at `verif/f2sim/Makefile:27-30` ("Bit-exactness
must hold at ALL ratios") — div 2 *is* the 62.5 MHz ratio (250/(2·2)).
Committed div-2 (and odd-ratio) evidence:

| proof | ratios | artifact |
|---|---|---|
| LC-1 CDC bridge, 18-job D=128 replay (27,996 checks, 0 fails × 3) | **8, 7, 2** | `docs/results/levelc_integration/cdc_sim_3ratio_2026-07-21.log` (cycle counters scale 432.5M/378.4M/119.0M shell cycles — the tile genuinely ran slower); summarized `docs/design/LEVEL_C_INTEGRATION.md:129` |
| IB-FUEL s3 host-mode replay counts unchanged | **{8, 7, 2}** | `docs/design/IB_FUEL.md:109` |
| IB-FUEL s4 fuel-mode (weights FROM DDR), 18/18, 27,996 checks | **{8, 7, 2}** | `docs/design/IB_FUEL.md:110,471` |
| E-1/E-2 elane norm-feed (nfeed bit-exact vs golden, discriminators red) | **5 and 2** | claim `docs/design/E2E_TOY_LANE.md:108`, `docs/design/MASTER_TABLE.md:34`; div-2 record preserved: `build/f2_elane/green_div2/report.json` (`"tile_div": 2`, all grades equal, 274 caps) |
| E-3a walked norm-feed (walker-armed) | **5 and 2** claimed | claim `docs/design/E2E_TOY_LANE.md:143-145`; **caveat:** the committed run record `build/f2_elane_walk_simcheck/report.json` shows `"tile_div": 5` only — the div-2 walked run is claimed in the doc but its record was not preserved under a distinct out-dir. Cheap to re-run (`scripts/fpga/f2/elane_walk_norm.py --tile-div 2`) and worth folding into the A0 ladder (§8, rung 0). |

Conclusion: the crossing is structurally frequency-agnostic **and** the
62.5-equivalent ratio has green, committed bit-exactness evidence for the
bridge, the fuel path, and the host-armed E-lane; only the *walked* E-lane
div-2 record is doc-claimed rather than artifact-preserved.

*Revision check (2026-08-05):* still true after E-5/E-6 — every committed
walked-run record in `build/e6_regress/` and `build/f2_elane_walk_simcheck/`
carries `"tile_div": 5` only. The E-6 replication note ("chain B = 559,697
tile cycles from both drivers **at two different clock ratios**",
`E6_ON_SILICON.md`) strengthens the ratio-independence argument but its
non-5 record is likewise not preserved under a distinct out-dir. Rung 0 of
the ladder (§8) exists precisely to convert these claims into artifacts
before any spend.

---

## 4. clkgen runtime mechanics for a 62.5 MHz card

All verified-on-silicon behaviors carry over from the A2 bring-ups
(`docs/results/f2_stage2_hw/RESULT.md:32-68`,
`docs/design/MILESTONE_D_RUNBOOK.md:248-283`):

1. **AFI load RESETS the clkgen MMCMs to the default recipe A1
   (a1 = 125 MHz)** — neither the bitstream's static MMCM properties nor
   the manifest's `clock_recipe_a` are honored at load
   (`RESULT.md:38-44`). For an A0 image, 125 MHz is 2× the closed clock
   (vs 8× for A2) — still garbage-with-green-rc territory. The recipe step
   after every load remains mandatory.
2. **Program:** `sudo fpga-load-clkgen-recipe -S 0 -a 0` (integer index;
   index 0 = A0 — table `fpga_clkgen_utils.c:75-79`, bounds check at
   `:366`; the CLI is a thin wrapper,
   `sdk/userspace/fpga_mgmt_tools/src/fpga-load-clkgen-recipe:19`). The
   tool sequences SYS_RST assert → DRP → lock wait → release itself
   (`RESULT.md:41-43`).
3. **Verify numerically, by column, never by substring.** Expected
   `fpga-describe-clkgen` group-A row under A0: **a1=62.50, a2=187.50,
   a3=250.00**. Trap: under **A2** the table *also* prints 62.50 — in the
   **a3 column** (`docs/results/f2_stage2_hw/clkgen_final.txt:4`). Any
   check that greps for "62.5" anywhere would pass vacuously on an
   unprogrammed/mis-programmed card. `remote_hw_exec.parse_clkgen_a1`
   already takes the column from the header row precisely to defeat this
   class (`scripts/fpga/f2/remote_hw_exec.py:251-286`, selftest with a
   swapped-column fixture at `:1136-1141`); only its expected-value
   constant changes (§7.2).
4. **Bring-up order:** extra clocks are HELD STATIC LOW until MMCM lock;
   poll MMCM_LOCK_REG / release SYS_RST before any BAR0 traffic or the
   bridge's dead-clock guard poisons by design
   (`tools/aws-fpga/hdk/docs/AWS_CLK_GEN_spec.md` "MMCM Clock Behavior";
   `cl_apex.sv:126-129`). Identical at A0.
5. The `f2_host_run.py` MMCM preflight remains **vacuous** (substring
   "lock" matches the tool's own "Clock Group" header,
   `scripts/fpga/f2/f2_host_run.py:48-73`,
   `docs/results/f2_stage2_hw/RESULT.md:64-68`) — the numeric gate in
   `remote_hw_exec` is the real check, at any frequency.
6. **HDPRVerify-41 ingest gate:** exonerated for the recipe TCL by direct
   A/B (patched kit fails identically; the unpatched A2 I-A build ingested
   fine — `docs/results/f2_afi_ingest/ESCALATION.md:42` row 1). The trigger
   is DM=3584-class scale (`ESCALATION.md:46` row 5). The b64_05B geometry
   is DM_MAX=896, and **DM=896-class ingestion is empirically proven
   twice**: the mislabeled `afi-09e68d25f4eefb3d7` (commit `25ddb66`,
   loaded on-card) and the corrected b64_05b image that flew 2026-08-03
   (`build/p05b_hw_check2.log`). Recipe choice (A0 vs A2) is not a variable
   in that failure mode.

---

## 5. Existing timing evidence (all four committed reports)

| build | config | tile constraint | result | receipt |
|---|---|---|---|---|
| first light `2026_07_17-202346` | **D=64**, G=128, DEPTH=128, OUTLIER_K=0, single-clock (tile on `clk_main_a0`) | **4 ns (250 MHz)** | **MET, worst +0.711 ns** — full tile (MXE, KVQ bank, ASU softmax/RMSNorm, TIP, SEQ/CSR, seams, squant) | `docs/results/f2_firstlight/RESULT.md:10-16`, `cl_apex.2026_07_17-202346.post_route_timing.rpt:16` |
| stage-2 D=128 `2026-07-21` | D=128, G=16, DEPTH=256, single-clock | 4 ns | **VIOLATED −34.766 ns**: `u_squant/out_slot_reg → pack_reg`, data path **38.986 ns** (logic 18.495 / route 20.492) ⇒ ~26 MHz cap for *that D=128 cone* | `docs/results/f2_stage2_hw/RESULT.md:105-117`, `post_route_timing_d128_VIOLATED.rpt` |
| I-A A2 `2026_07_22-183309` | D=128, two-clock LC-1 | **64 ns** (A2) | MET; tile group worst listed slack +3.702 ns is a 4 ns **MaxDelay CDC exception** path, i.e. every intra-tile 64 ns path had ≥ that slack | `f2_stage2_hw/RESULT.md:22-27`; report `:996-1029` (Requirement 4.000ns MaxDelay), `:1000` (period=64.000ns) |
| wide probe `2026_07_27-195354` | D=128, GQA=4, **DM=3584** | 64 ns (A2) | routed MET, worst +0.711 ns (shell path); phys_opt stage +0.177 ns | `docs/results/ic_dcp_probe/RESULT.md:45`, `RESULT_ROUTED.md:15-16`, `ESCALATION.md:156,208` |
| **DDR bring-up `2026_08_05-032359`** | b64 geometry, **DDR=1**, first build with the real CDC xdc (§1.2a) | 64 ns (A2) | **MET, WNS +0.392 / TNS 0.000**; worst path INSIDE AWS's DDR4 core (intra-domain) | `cl_synth_user.xdc:21-36`, `docs/results/prompt_on_chip/DDR_BRINGUP_RESULT.md` |
| **E-5 convergence** (`agfi-0183a4b88c8d21163`) | b64 + walked fuel projections + DDR=1 | 64 ns (A2) | MET, worst slack **+0.277** | `SELF_RUNNING_CARD_RESULT.md` header |
| **E-6 convergence2** (`agfi-0bc20880b50f5faba`) | b64 + walked o8 epilogue (S2_PSJ/RT_FPRQ) + D-aware k_job + all of E-5 | 64 ns (A2) | MET, worst slack **+0.297** — the NEWEST netlist's A2 sign-off | `E6_ON_SILICON.md` header |

### 5.1 What this says about D=64 at 16 ns

The **entire D=64 tile closed 4 ns** — including the exact cone
(`apex_scale_quant` fp16 scale-composition/pack) that at D=128 blew out to
38.99 ns, because its buffer/mux depth scales with `max(CFG_D, T_ROW_MAX)`
(`f2_stage2_hw/RESULT.md:119-124`). The b64_05B image is CFG_D=64: the one
empirically-known killer path is in its small elaboration, with a 4×
looser budget than it has already met.

### 5.2 What it does NOT say

First light predates: the CDC bridge + mailbox, the B1 walker, the S12 mask
lane, GQA=2 (two KVQ engines), QSTAGE_H_MAX=14 staging, wide RMSNorm at
DM_MAX=896, and the whole E-lane (residual, layer-deq, swiglu, rope glue).
Those blocks have existed only under 64 ns constraints (and the wide-norm
family only in DM=3584 builds at 64 ns). None has a 16 ns data point. §6
argues structurally that none should fail 16 ns; the A0 build is the proof.

### 5.3 The b64_05B ("b64v4") image specifically

No timing summary for any b64-family DCP exists in this tree (checked
`build/` and `docs/results/` — the reports are on the devbox EBS). What the
tree does prove: the corrected image (D=64, DMODEL=64, GQA=2, DM_MAX=896,
QSTAGE=14 — `scripts/fpga/f2/tile_geom.py:131-132`) was built, ingested,
and ran 4,380 programs bit-exact on silicon at 15.625 MHz on 2026-08-03
(`build/p05b_hw_check2.log`). Both build entry points pin recipe A2
(§1.3), so that image is 64 ns-constrained.

### 5.4 The "+0.711 ns" red herring, retired

+0.711 ns appears as the worst overall slack in *both* first light
(`f2_firstlight/RESULT.md:14-15`) and the wide routed build
(`ESCALATION.md:156,208`) because it is the same **shell `clk_main_a0`
250 MHz path** — "identical to first light" is the I-A report's own wording
(`f2_stage2_hw/RESULT.md:23-24`). It is the shell's fixed margin, present
in every green build, and carries zero information about tile-clock
headroom. Do not read it as "the wide build barely made 64 ns."

### 5.5 The NEW red herring of the same class (revision)

Since the CDC fix, the overall WNS of every DDR=1 build (+0.392 / +0.277 /
+0.297) sits on **AWS's DDR4 core's own intra-domain path** on
`mmcm_clkout0` (`cl_synth_user.xdc:35-36`) — a clock the recipe does not
touch. Expect the A0 build's headline WNS to look almost unchanged
(~+0.3 ns) **whether or not the tile made 16 ns**. The verdict lives in the
`clk_out1_clk_mmcm_a` path group (`period=16.000ns`, in-group slack ≥ 0),
which is exactly what `build_a0_62p5.sh check-timing` gate G4 reads. Never
quote the overall WNS as the tile margin, in either direction.

---

## 6. Critical-path analysis at 16 ns (structural, block by block)

Measured anchors: D=64 full tile ≤ 4 ns (§5.1); D=128 squant cone 38.99 ns
under 4 ns pressure (§5). Everything else is never-timed-at-16 ns; ranked
by cone depth from the RTL:

| block | deepest per-cycle cone | assessment at 16 ns |
|---|---|---|
| `apex_scale_quant` (u_squant) @ D=64 | fp16 scale-composition/pack over `max(CFG_D=64, T_ROW_MAX)` slots | **proven** ≤4 ns in the first-light netlist class; 16 ns is 4× that. Lowest risk of the "known" paths |
| wide RMSNorm `asu_rmsnorm` @ DM_MAX=896 | per-emit chain `p_xg = 24b×24b` then `q_full = 40b×40b` (`rtl/asu/asu_rmsnorm.sv:342-343`) + RNE + sat16 — two chained DSP48E2 multiplies unpipelined; μ path `prod = sum2 × MU` (`:281,306`), 24b×17b at DM=896 | **the widest new arithmetic cone.** Two cascaded DSP multiplies + correction ≈ 8–10 ns worst-case on US+ — fits 16 ns, would NOT fit 4 ns. This is the path to read first in the A0 report (the standing instruction "the 45-bit μ-multiply timing must be READ from the build report, not assumed", `docs/design/LEVEL_C_INTEGRATION.md:210`, applies at 896 exactly as at 3584) |
| `rsqrt_unit` | none — 31-cycle serialized digit recurrence, add/compare/shift per cycle, no multiplier (`rtl/asu/rsqrt.sv:22-47`) | **negligible**; among the friendliest blocks in the design |
| residual (`apex_residual` / `residual_add_fx` / `f16_pack_real`) | one 57-bit-grid exact add + single-RNE narrowing per beat: grid-align, ≤57-bit add, 57-bit normalize (priority encode) + RNE (`rtl/top/glue/apex_residual.sv:34-38`; `rtl/misc/residual_add_fx.sv:40-50`; `F16_SIG_W = 57`, `rtl/misc/f16_arith_pkg.sv:42,88`) | carry chain ~57b (≈3–4 ns) + normalize/pack (≈4–5 ns) ≈ **7–9 ns** — fits 16 ns with margin. The "57-bit adder chain" is one adder + one pack per beat, not a chain of adders |
| swiglu (`asu_swiglu`) | silu LUT core + one f16 narrowing per gate beat; fp16 multiply + pack per up/product beat (`rtl/asu/asu_swiglu.sv:115-190`) | LUT + 1 DSP + pack ≈ 6–8 ns — fits |
| rope (`rope_pair_fx`) | fixed-point pair rotation (2 multiplies + add/round) | DSP class, fits |
| CDC / afifo / walker / mailbox / CSR | handshakes, gray pointers, FSMs — logic levels 0–4 in the committed reports | non-issues (the A2 report's tile-group entries are all 0-logic-level CDC exception paths, §5 row 3) |
| GQA=2 / QSTAGE=14 / DEPTH sizing | wider muxing and more RAM, not deeper cones per se | route-pressure risk, not logic-depth risk; watch congestion in the A0 report, not WNS alone |

*Revision — what E-5/E-6/DDR added since the table was drawn:* the fuel
burst reader + afifo (gray-pointer/FSM class, already rowed above), sh_ddr
+ its clock converter (AWS IP on its own clocks, not on `clk_tile`), and
the walked o8 epilogue — S2_PSJ serializer framing, the RT_FPRQ requant
route, the fp_rq_q window flag, and the S2_PDW drain fence. Those are mux /
FSM / squant-class logic: the requant arithmetic they route through is the
same `apex_scale_quant` cone the D=64 first light closed at 4 ns. No new
cone class was introduced; the deepest never-timed-at-16 ns cone is still
the wide-norm chained DSP pair (`asu_rmsnorm.sv:342-343`, re-verified at
those exact lines on this tree).

**Bottom line:** no identified cone in the b64_05B elaboration structurally
exceeds ~10 ns on this part; 16 ns should close with real margin. The two
things a 16 ns report must be read for: (a) the wide-norm chained-DSP emit
path, (b) route-dominated paths (the D=128 violation was 53% route — at
D=64 with GQA=2 the placer has slack, but the CL is bigger than first
light's). If A0 misses by a nose, the *only* sanctioned lever is
registering the norm emit chain — but that touches verified RTL and buys a
demo nothing (§9); take A2 instead.

---

## 7. Exactly what a 62.5 MHz image requires

### 7.1 Build side (no file edits required) — runnable: `build_a0_62p5.sh`

```sh
# on the devbox, CL assembled per BRINGUP.md step 3 — preferred form
# (exports the geometry, builds at A0, runs timing gates G1-G5):
bash scripts/fpga/f2/build_a0_62p5.sh build
# exactly equivalent to:
APEX_CL_D=64 APEX_CL_DMODEL=64 APEX_CL_GQA=2 APEX_CL_DM=896 \
APEX_CL_QSTAGE=14 APEX_CL_DDR=1 \
  bash scripts/fpga/f2/build_dcp.sh --clock_recipe_a A0
```

- `build_dcp.sh` appends `"$@"` after its pinned `--clock_recipe_a A2`
  (`build_dcp.sh:29-31`); `aws_build_dcp_from_cl.py` parses with
  **optparse** (`OptionParser`, `aws_build_dcp_from_cl.py:26,162` — the
  first edition said argparse; same default-`store` last-wins semantics,
  wrong library), so the **last** occurrence wins and A0 takes effect
  without editing the script. The kit echoes every effective option in its
  "Running CL builds" banner (`clock_recipe_a : A0`) — gate G1 asserts it;
  the manifest records it too (`aws_build_dcp_from_cl.py:144`) but that is
  informational only at load, §4.1.
- Geometry knobs are env-plumbed with historical defaults
  (`synth_cl_apex.tcl:65-76`) — **all SIX must be exported** (the revision
  adds `APEX_CL_DDR=1`: the E-5/E-6 lineage is DDR-enabled, and DDR=1
  additionally pulls `USE_64GB_DDR_DIMM` + `cl_ddr4.xci` +
  `cl_axi_clock_converter.xci`, `synth_cl_apex.tcl:90-135`) or the build
  silently reverts to the D=128/DDR=0 defaults — the exact mislabel trap
  already paid for once.
- Constraints: unchanged (§3.1), **and** the flow now hard-gates that they
  were read and matched (§1.2a; gate G2). Instance name `AWS_CLK_GEN`:
  already correct (`cl_apex.sv:140`). Expect the usual benign "Clock recipe
  … not applied" CRITICAL WARNINGs for the disabled B/C/HBM groups; **any
  A-group warning is fatal** (gate G3;
  `f2_stage2_hw/RESULT.md:28-30,34-37`).
- The **timing report is the gate**: tile path group `clk_out1_clk_mmcm_a`
  must show `period=16.000ns` and MET (gates G4/G5 — and per §5.5, read the
  tile GROUP, never the overall WNS). A0-with-16ns-MET is the entire
  hardware-side deliverable; everything else below is host tooling.

### 7.2 Host side — LANDED 2026-08-05 (image-keyed clock expectation)

The CL exposes no recipe-identity register and the manifest recipe is not
honored at load (§4.1), so the host binds expected-MHz to the **AGFI id**
it loaded. Single source of truth: **`scripts/fpga/f2/clock_key.py`** —
`RECIPE_A_A1_MHZ` (`:50`, from the SDK C table), `IMAGE_RECIPE`
(`:83`, per-AGFI constrained recipe + receipt, all 9 known images = A2;
same allowlist discipline as `tile_geom.IMAGES_WITH_BASE_ROW_FIX`), and
`expected_clock(agfi, run_recipe=None)` (`:125`) with the **asymmetric
rule**: unknown AGFI → refuse; deliberate underclock → legal, flagged;
**overclock → refuse, no override exists** (`:173`).

Status of the five original points:

| point | status |
|---|---|
| `remote_hw_exec.py` constants | KEYED: `TILE_CLK_MHZ/TOL` now derive from `clock_key` (`remote_hw_exec.py:121-123`); `resolve_clock_expectation` (`:360`) + `check_tile_clock(agfi=, run_recipe=)` (`:403`); `run_job_remote(agfi=, run_recipe=)` (`:492`); env `$APEX_F2_AGFI` / `$APEX_F2_RUN_RECIPE` reach `remote_config`/`attach` (`:833`) so prompt05b/batch_exec flights key WITHOUT code changes; CLI `--agfi/--run-recipe` (`:1560,1565`). Unkeyed callers keep the A2 default and the gate SAYS unkeyed — an A0 card refuses under them (safe direction). Selftest cases [15a-f] prove: A2 image at 62.5 REFUSED; A0 image at 15.625 REFUSED unless the arm is explicit; overclock/unknown/bare-run_recipe refused PRE-CONTACT |
| `apex_repl.py` bring-up | KEYED: `hw_bringup_cmds` derives the `-a N` recipe from the image and REFUSES unregistered AGFIs (`apex_repl.py:1141`); bring-up verifies against the keyed MHz and prints the key (`:1188`); footer + session banner name the keyed clock and disclose the underclock arm (`SESSION_CLOCK`, `:92`); CLI `--run-recipe` (`:1729`); per-job gate keyed via env export (`:1801`). Selftest: A2→`-a 2`, fake-A0→`-a 0`, unregistered/overclock refused, footer wording checked |
| `f2_host_run.py` preflight | still vacuous BY DESIGN and still skipped by the remote path (the numeric gate replaces it); docstring de-staled to say so and to name both recipes (`f2_host_run.py:49-60`) |
| `MILESTONE_D_RUNBOOK.md` §3.3 gate | UNCHANGED ON PURPOSE: that runbook documents a specific historical A2 flight of `agfi-0ae06ea…` — its 15.625 pin is correct for that image forever. The A0 on-card gate lives in `build_a0_62p5.sh card` |
| `verif/f2sim/Makefile:29` | comment fixed ("recipe A0"; the `walk_fuel_layer.py:229` copy is owned by the walker track) |

**The rule the selftests enforce:** a fast image can never be flown fast by
accident (registration is an explicit reviewed commit), a slow image can
never be flown fast at all, and the ONLY way to run below sign-off is the
disclosed `run_recipe` arm the §7.3 A/B needs.

### 7.3 A property worth exploiting: an A0 image can legally run at 15.625

The recipe is programmed at runtime into the same MMCM; running an
A0-constrained (16 ns) netlist with recipe A2 (64 ns) is running 4× slower
than sign-off — setup-safe by construction, hold unaffected (hold closure
is period-independent). This gives the ladder a clean A/B: **fly the A0
image first at `-a 2` and reproduce the entire proven suite bit-exact,
then flip only the recipe to `-a 0` and rerun.** Any divergence is then
attributable to frequency (a real >15.625 MHz timing escape), not to the
new netlist. *(Revision: this arm is now a first-class, disclosed mechanism
— `run_recipe`/`$APEX_F2_RUN_RECIPE`/`--run-recipe`, legal only downward,
flagged "UNDERCLOCK ARM" in every gate message and footer; §7.2.)*

---

## 8. Validation ladder for the A0 image — runnable: `build_a0_62p5.sh`

The ladder is now a script with one explicit subcommand per rung
(`scripts/fpga/f2/build_a0_62p5.sh`; `plan` prints, nothing auto-runs; its
`check-timing` gates G1–G5 are fixture-tested for every failure mode). The
prose rungs, with what changed in the revision:

0. **(local, free, before any spend)** Re-run the walked-norm demonstrator
   at the 62.5 ratio and preserve the record the docs currently only claim:
   `python3 scripts/fpga/f2/elane_walk_norm.py --tile-div 2 --out build/f2_elane_walk_div2`
   (§3.2 caveat, still open post-E-6). Also rerun the b64_05b twin gate at
   div 2: `verif/f2sim` `make run D=64 ... TILE_DIV=2` on the 7-job 0.5B
   set (green precedent at div 5:
   `docs/results/p2_05b_gate/f2sim_run_d64_div5.log`,
   `files=7 checks=1052 fails=0`).
1. **Timing-report gate (devbox):** `build_a0_62p5.sh build` = §7.1 build +
   gates G1 (kit echoed A0), G2 (repo xdc read + "APEX CDC gate OK" —
   §1.2a), G3 (no A-group recipe warning), G4
   (`clk_out1_clk_mmcm_a period=16.000ns`, and neither 64.000 nor the 8 ns
   XCI default), G5 (MET, no *VIOLATED* checkpoint, tarball). Then READ the
   tile-group paths for the wide-norm cone (§6) — the script prints them.
   FAIL = stop; ship nothing; status quo A2 stands.
2. **PRV gate (devbox, ~90 s):** `build_a0_62p5.sh prv` —
   `pr_verify -full_check -in_memory -additional <kit>/from_aws/cl_bb_routed.small_shell.dcp`
   (procedure: `docs/results/f2_afi_ingest/ESCALATION.md:158-165`).
   Expected clean at DM=896 scale (§4.6) — now proven **eight consecutive
   first-try ingestions** deep (the E-6 doc's "9th" was corrected to the
   account-verified 8 by MASTER_TABLE v21).
3. **Ingest:** `build_a0_62p5.sh ingest <tarball>` (create_afi.sh).
4. **REGISTER THE IMAGE (new, mandatory):** add the AGFI to
   `clock_key.IMAGE_RECIPE` as `(0, "<receipt>")` — one line, one commit —
   **before** any card session. Every host tool refuses an unregistered
   AGFI by design (§7.2); that refusal is the gate working.
5. **Card bring-up:** load → `sudo fpga-load-clkgen-recipe -S 0 -a 0` →
   `remote_hw_exec.py --check-clock --agfi <AGFI>` — **numeric a1-column
   check = 62.50** (§4.3; the a3=62.50-under-A2 trap makes column-aware
   parsing mandatory, and the keyed gate makes a bare-grep pass
   impossible) → MMCM-lock order per §4.4.
6. **Identity reads (the cheap go/no-go, per the §4-of-runbook idiom):**
   BAR0 `0x0000 = "A9EX"`, `VERSION`, KVQ `INFO_DIM = 0x40` (the read that
   caught the mislabeled image, commit `25ddb66`), `INFO_TIER = 0x1`
   (GQA=2 CQ-8-only truth, `scripts/fpga/f2/tile_geom.py:450-452`),
   `INFO_GROUP`, and — DDR image — `FUEL_STAT` ddr_ready=1 after the
   weight-image load (`DDR_BRINGUP_RESULT.md`).
7. **Canonical replay, A/B on the same image (§7.3):** ARM A first at
   `--run-recipe 2` / `$APEX_F2_RUN_RECIPE=2` (15.625, the disclosed
   underclock arm), then ARM B keyed plain (62.5): the 7-job real-0.5B
   trace replay (1,052 checks) via `remote_hw_exec`; the graded
   `prompt05b.py --layers 0,1 --executor hw` flight with full geometry
   audit, INFO_D per program, and token-identity verdict (the run that
   passed at 15.625 on 2026-08-03, `build/p05b_hw_check2.log`: 4,380
   programs, checks 34/34 per layer, TOKEN == PURE HOST: PASS); **and the
   E-5 + E-6 gates re-run bit-exact** — `walk_fuel_qkv` 144/144 with
   walk-off RED + DDR-poison RED (`SELF_RUNNING_CARD_RESULT.md`), and
   `walk_oproj`/`walk_qkv_oproj` r1 896/896 with both discriminators
   (`E6_ON_SILICON.md`). PASS criterion at 62.5: **byte-identical captures
   and identical verdicts** vs ARM A. (The D=128 18-job set does not apply
   — INFO_D refuses it by design.)
8. **Disclose** per §9: in walked mode the 4× is real and predicted
   (~280 → ~70 ms/layer); in host-dispatched mode it is ≤~3%. Never let
   the 4× clock number stand in for a 4× demo without the walked-path
   caveat and its 8-of-13-families fence.

---

## 9. Amdahl: what 4× tile clock is actually worth today

*(Revision framing: §9.1–9.2 measure the HOST-DISPATCHED regime and remain
correct for it. The walked regime E-5/E-6 opened on silicon is the §9.3
revision block — there the tile clock IS the wall clock.)*

### 9.1 What a job costs on the wire vs in the tile

**Tile-clock-bound cost** (sim cycle counters; shell cycles ÷ 2·tile_div =
tile cycles):

- 0.5B attention-class jobs (b64 twin, div 5,
  `docs/results/p2_05b_gate/f2sim_run_d64_div5.log`): 2,003–5,981 ops;
  per-job **24k–72k tile cycles** (e.g. 5,318-op job = 642,820 shell cyc →
  64,282 tile cyc → **12.1 tile cycles per BAR0 op**).
- Cross-check at div 8 (D=128 18-job set,
  `cdc_sim_3ratio_2026-07-21.log:1-2`): 32,845-op job = 7,206,928 shell cyc
  → 450k tile cyc → **13.7 cycles/op**. The tile-side choreography costs a
  consistent **~12–14 tile cycles per regop** (CDC crossing + FSM/mailbox +
  compute overlap).
- At **15.625 MHz** that is 0.83 µs/op → **1.5–4.6 ms per 0.5B attention
  job**, ~1.1 ms per 1,323-op projection K-split job. At **62.5 MHz**:
  0.21 µs/op → 0.4–1.2 ms. **Recoverable by the 4× clock: ≈0.8–3.5 ms per
  job.**

**Measured wall cost per job on silicon (the dispatch reality):**

| regime | measured | receipt |
|---|---|---|
| single ssh invocation per job | ~3.6 s/job | `scripts/fpga/f2/proj_sweep_batched.py:8-13`; `build/hw_s2_sweep/attrib_proof.json` (8 jobs: 29.04 s separate vs 6.52 s batched) |
| current b64_05B flight (2026-08-03) | **0.215 s/job executor** (2,190 jobs / 470.28 s), **~0.26 s/job wall** (575.05 s incl. emit/audit/grade; whole-token: 4,380 jobs / 1,160 s = 0.265) | `build/p05b_hw_check2.log` |
| best-batched regime (7B image, 29,696 K-split jobs) | 0.03–0.07 s/job in-batch, 0.025 s/job run-phase aggregate | `build/hw_s2_sweep/full_sweep5.log`, `sweep/sweep_result.json` (`s_per_job = 0.025`) |

### 9.2 The fraction that is tile-clock-bound, and the ceiling

| regime | tile-bound share | 4×-clock end-to-end saving |
|---|---|---|
| current flight (~0.26 s/job wall, mix dominated by ~1.3k-op K-splits) | 1.1–4.6 ms / 215–265 ms ≈ **0.5–1.8%** | ≈0.8–3.5 ms/job ⇒ **≤ ~1.6%** (typ. ~0.4–0.6%) |
| best-batched (0.025–0.07 s/job) | ~1.1 ms / 25–70 ms ≈ **1.6–4.4%** | **≤ ~3%** |
| hypothetical infinitely fast tile clock | — | caps at the same shares: **1.005–1.05×** |

Per-op view, same conclusion: measured wall ≈ 100–170 µs per regop at
0.215 s/job; the tile-domain leg is 0.83 µs of that (0.5–0.8%), shrinking
to 0.21 µs at 62.5 MHz.

### 9.3 What actually moves the number

The measured levers are dispatch-side and dwarf the clock: batching alone
was an **18×** (3.6 → 0.2 s/job, `docs/results/p2_multilayer/RESULTS.md:73-75`);
the walker's steady state is **3 MMIO per step** versus per-op BAR0
choreography (B1/D-028 lineage); fuel/DDR weight streaming removes the WB
triples entirely (IB-FUEL, proven in sim at div ∈ {8,7,2}). **Only after**
those make the tile the constraint does the 62.5 MHz image pay: in a
walker/fuel regime a job's wall collapses toward its tile-cycle cost, where
4× clock ≈ 4× throughput. That is the honest sequencing argument: the A0
image is *enabling infrastructure for the post-dispatch-fix world*, cheap
to validate now (one build + one card session), but it is **not** a speedup
of the demo as currently dispatched, and must not be sold as one.

**Revision — the post-dispatch-fix world partially arrived (E-5/E-6 on
silicon, 2026-08-05/06).** The walked fuel projection ran on the FPGA with
the host writing NOTHING inside the walk window
(`SELF_RUNNING_CARD_RESULT.md`), and the E-6 chain put a whole
QKV+OPROJ+epilogue+residual sequence under ONE descriptor
(`E6_ON_SILICON.md`). The E-6 doc's labelled PREDICTION (not measurement):
~2.19k tile cycles/job, **~4.4 M tile cycles for a fully-walked layer ⇒
~280 ms/layer at 15.625 MHz vs ~70 ms at 62.5 MHz ⇒ ~1.7 s/token at
62.5** — in that regime the 4× clock is ~4× wall. Two fences keep this
honest: (a) a fully-walked layer is NOT yet achievable — 8 of 13 step
families walk, the rest refuse with measured reasons (`E6_ON_SILICON.md`
"Honest scope"; I-B gap list); (b) the per-op host-dispatched numbers
above still govern everything outside a walk window. The A0 image is the
clock the walked path will be measured at; it is still not a 4× on the
demo as a whole.

---

## 10. Risk register & recommendation

| risk | likelihood | mitigation |
|---|---|---|
| A0 build misses 16 ns (wide-norm DSP chain or route congestion) | low (§6) — but the E-5/E-6 additions and DDR=1 have never been placed under 16 ns pressure together | loud, cheap failure: `build_a0_62p5.sh` rung 1, ~$1 + ~70 min; fallback = status quo A2 (see "what fails if timing does not close" below). Do NOT pipeline verified RTL for a demo |
| host tooling refuses / vacuously passes at the new frequency | ~~certain if §7.2 skipped~~ **retired**: §7.2 LANDED — image-keyed via `clock_key.py`, selftests prove both cross-wirings refuse (remote_hw_exec [15a-f], apex_repl bringup checks); the a1-column parser already defeated the a3=62.50 trap | residual: registering a new AGFI with the WRONG recipe index — mitigated by the receipt-string convention and the one-line reviewed commit |
| an A0 AGFI flown from an OLD checkout (no clock_key) | possible on stale clones | the old constants refuse a 62.5 card with rc=78 (the safe direction, first edition §7.2 row 1) — annoying, never silent-wrong |
| forgetting recipe after AFI load (now 125 = 2× over instead of 8×) | recurrent failure mode by design | unchanged discipline: program + numeric verify before any BAR0 op, per-invocation gate stays ON and is now keyed |
| reading the overall WNS as the tile verdict (§5.5) | new with DDR=1 (worst path lives in the DDR4 core at every recipe) | check-timing G4 reads the `clk_out1_clk_mmcm_a` group explicitly |
| conflating clock speedup with demo speedup | reputational | §9 numbers in the result doc; claim template: "correctness clock ×4; walked-path prediction ~280→~70 ms/layer (8-of-13 families fenced); host-dispatched path ≤~3%" |
| ingest regression | very low at DM=896 (§4.6; eight consecutive first-try ingestions, account-verified) | PRV rung 2 catches it locally in 90 s |

**HONEST RISK — what fails if timing does NOT close at 62.5.** The blocks
list, in order of prior: (1) the wide-norm chained DSP pair
(`asu_rmsnorm.sv:342-343`, two cascaded multiplies + RNE in one cycle,
never timed below 64 ns — §6 estimates 8–10 ns but that is an estimate,
not a datum); (2) route congestion on the b64+DDR=1 CL (the D=128 squant
violation was 53% route; DDR adds a large hard-IP footprint the placer
must work around); (3) any 0-logic-level CDC exception path whose 4.0 ns
datapath bound gets route-squeezed (would show as MaxDelay fails in the
tile group). **Fallback ladder:** (a) stay at A2 — the walked path already
earns its dispatch win there, just 4× slower per tile cycle (~280 ms/layer
predicted, still a working demo); (b) the ONLY sanctioned RTL lever is
registering the norm emit chain (one pipeline stage between `p_xg` and
`q_full`) — it touches verified RTL, needs the full L3/E-lane regression
suite re-run bit-exact, and is NOT worth it for a demo unless the walked
path is already the bottleneck at A2 and the miss is in that exact cone;
(c) there is no half-step — §2.2 stands, dynamic clkgen bottoms at 87 MHz
and a custom MMCM is a new constraint surface in the HDPRVerify-41 file
family. A0-fails ⇒ ship A2, file the failing path, decide (b) only with
the report in hand.

**Addendum 2026-08-05 (post-E-6):** the walker/fuel regime §9.3 argues for
now has a measured (sim) anchor — a fuel-fed k=896 projection job costs
~2.19k tile cycles (~3.3 weight B/tile-cycle sustained,
`docs/results/prompt_on_chip/WALKED_EPILOGUE_E6.md`, cross-verified to the
cycle by the replication driver at a second clock ratio). In that regime the
job wall IS tile-cycle-bound and the A0 image's 4× pays ~4×, exactly per
§9.3's sequencing argument. Still sim-derived; the A0 build itself remains
unbuilt and this doc's ladder unchanged.

**Recommendation.** Target **62.5 MHz via `--clock_recipe_a A0`** directly —
there is no stock intermediate rung, and the D=64 evidence base does not
justify inventing a custom ~31 MHz flow. The sequence, with the revision's
status: §7.2 host-side image-keying **(DONE — clock_key.py + keyed gates +
selftests)** → rung-0 local div-2 records → one A0 devbox build
(`build_a0_62p5.sh build`) read specifically for the wide-norm emit path →
PRV → ingest → **register the AGFI in clock_key.IMAGE_RECIPE** → the §8
A/B card session (A0 image at `--run-recipe 2` first, then keyed 62.5),
everything bit-exact both arms including the E-5/E-6 gates. Book the win
honestly: "tile correctness clock ×4; the walked path (where E-5/E-6
removed per-op dispatch) is predicted ~280→~70 ms/layer at it, fenced at
8-of-13 walkable families; the host-dispatched path barely moves."

## §9 — A0 attempt log (2026-08-07, measured)

| build | tree | WNS | verdict | the cone |
|---|---|---|---|---|
| a0 v1 | @8ebdf52 (pre-pipeline) | **-23.283** | VIOLATED | u_squant pack: RAM read -> 25x11 mult -> COMBINATIONAL 42/37 divide -> RNE -> pack, one cycle. 249 levels, 193 CARRY8. All top-10 into pack_reg. |
| a0 v2 | @c47e00d (squant 3-stage pipeline + 9-step restoring divider) | **-8.680** | VIOLATED | the squant cone is DEAD (14.6 ns recovered). New worst: **u_walk** — pc/FSM/err_code/lw_nsrc endpoints own the entire top-10. TNS -42,221 says a second TIER of 16-25 ns cones exists beyond the walker. |

| a0 v3 | @6480ccb (+ walker desc-check at true width) | **-6.985** | VIOLATED | TNS collapsed -42,221 -> **-14,820** (65%); the walker tier is dead too. The remaining worst is ~23 ns with a BROAD tail — A0 is a CAMPAIGN (multiple cone tiers), shelved behind the walked-attention fix. |

Tier 3, named from the a0 v3 report (same WNS −6.985, worst path now):
**u_feeder** — seam_feeder_quant's pack cone, 138 levels / 97 CARRY8:
the feeder's own C-1 quantize divide, the SAME disease as apex_scale_quant
and the same cure (3-stage drain pipeline + explicit small divider). The
tile has TWO C-1 instances; only one was pipelined. Campaign standing:
tier 1 squant (dead), tier 2 walker check (dead), tier 3 feeder (named,
un-fixed), tail TNS −14.8k beyond it.

The original next-move note (now executed): the walker's S2_CHECK/desc-legality blob evaluates
`walk_desc2_check` combinationally — including `(nh / nk) * nk != nh`, ANOTHER
combinational integer divide — feeding the FSM in one cycle. It runs ONCE per
descriptor load/kick, so multi-cycling it is throughput-free: same playbook as
the squant cone (split + explicit small divider), easier legality. After the
walker, re-measure; the -42k TNS tail decides whether A0 is a third patch or a
campaign. A2 remains the shipping recipe throughout; there is no 31.25 rung
(§2.2 — the ladder is binary).

## §9 addendum — 2026-08-12 A0 attempts on the merged tree (devbox3)

Two builds, both VIOLATED, ladder moved:
- merged tree as-is: WNS −8.681, TNS −45,688 — top 12 paths ALL
  `wc_snp_data_q → g_comp[*]/sc_mem_reg_bram DIN` (the D-033 snoop stage
  into the bank BRAM; dont_touch rightly forbids retiming into the bank).
- + second bundle stage (q2, @cfb1e43): WNS −8.411, TNS −26,280 — TNS
  nearly halved, but the WORST path is the SAME HOP from q2: the ~8.4ns
  is physical distance to the bank BRAM, not logic depth. Pipelining at
  the tile level cannot fix a single placement-length hop.

**Verdict:** A0 closure needs (a) a register stage INSIDE apex_wcomp_bank
per engine (placed with the bank, inside the keep_hierarchy region) and
likely (b) a pblock pinning the bank near the seam. Parked as a dedicated
campaign — at today's 5.8 s/token the tile is ~36 ms of wall, so A0 pays
only after the walked fraction dominates (E-7/FFN integration first).
q2 kept: pure margin at A2, halves the A0 TNS for the future campaign.

## §9 addendum 3 (2026-08-14): A0 tier 3 + the A1 pivot

A0 build 3 (in-bank ingress stage @14f16d1): WNS −11.51 TNS −42.7k —
WORSE. The critical path is now IN-BANK: snp_data_q_reg (the new stage)
→ sc_mem BRAM DIN, both inside the keep_hierarchy region. At 16ns the
D-033-protected write cone itself doesn't close, and dont_touch forbids
the tools from restructuring it. A0 = a redesign fight against the
walked-attention protections (pipeline INSIDE seq_walker_comp without
reopening D-033) — parked as its own campaign.

PIVOT: A1 = 31.25MHz (32ns period; today's ~20ns arrivals close with
margin) = 2× the walk for zero RTL risk. Build launched on the same
tree. Ladder value: walk 2.5s → ~1.25s/token.

## §9 addendum 4 (2026-08-14): A1 misfire — no 2x preset exists

A1 build: WNS −19.85 TNS −170k — recipe A1's a1 clock is ~125MHz (8ns),
NOT 31.25. The group-A preset ladder offers no step between A2 (15.62)
and A0 (62.5). Clock options forward: (a) the A0 redesign campaign
(pipeline the sc_mem write cone INSIDE the D-033 protections); (b)
investigate custom clkgen frequencies (does the F2 SDK/mgmt interface
accept non-preset MMCM programming? check fpga-load-clkgen-recipe
alternatives + the AWS_CLK_GEN spec on the next devbox session).
Devbox6 terminated. The 2026-08-13 speculative-clock spend: 3 builds,
~$11 — bought the definitive map of the clock wall.
