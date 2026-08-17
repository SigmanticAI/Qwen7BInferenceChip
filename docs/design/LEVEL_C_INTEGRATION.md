# LEVEL-C INTEGRATION — the serial combine step (contract & runbook)

**Branch:** `comp/level-c-integration` (based on `decoder-layer-and-fpga-fit`
@ f707a85, which carries the F2 CL, f2sim, and the stage-2 inheritance).
**Owns:** merging the three Level-C lanes, the apex_top wiring their stage-5
specs defer here, the ONE CDC/clocking build (willed to this session by
`docs/results/f2_stage2_hw/RESULT.md` §Inheritance), and — tranche I-B —
the full-layer + DRAM work. **Written 2026-07-21** by the wide-rmsnorm
session as side-work while B1 stage 3+ and the W4 adoption eval run.

Prime rules unchanged: golden is the arbiter; no PASS without pasted output;
`apex_pkg.sv` FROZEN (APEX_VERSION 0x0001_0000) — none of the three lanes
touched it (verified); serialize EDA vs evals (see §6 machine plan).

---

## 0. Two tranches — do not blur them

- **I-A — attention autonomous on silicon.** Merge the lanes, execute the
  two deferred stage-5 wirings, build the CDC'd CL, replay the 18 S8 trace
  jobs on an f2.6xlarge. Lands the parked attention-on-silicon claim
  (wording stays scope-locked per `docs/results/f2_stage2_hw/RESULT.md`).
- **I-B — the full decoder layer + DRAM weights → whole-7B-on-FPGA.**
  Integrate RoPE/SwiGLU/o-proj/residual into apex_top (today they are
  standalone composition-verified blocks — B1_WALKER.md scope note),
  extend the walker descriptor to walk the full layer (B1 was designed so
  this is a descriptor extension, not a rewrite), elaborate wide RMSNorm,
  and move to the DMA/DDR shell for weight streaming. This is the flagship
  ("full 7B decodes token-by-token on the FPGA") and gets its own staged
  plan appended here when I-A is green.

## 1. Inputs — lane states at time of writing

| lane | branch @ tip | state | stage-5 spec lives in |
|---|---|---|---|
| wide-RMSNorm | `comp/wide-rmsnorm` @ 807c782 | ✅ COMPLETE (stages 1–5 + synth probe; D≤128 anchor byte-identical; `verif/asu/wide` green; x_buf infers BRAM as-is) | none needed — merges inert at RMS_D_MAX=128 |
| B3 weight path | `comp/b3-weight-path` @ 4104897 | ✅ RTL COMPLETE (stages 0–3, 5/5 mutants; feeder deliberately NOT in `apex_sources.f`) | B3_WEIGHT_PATH.md stage 5 (route bit + CSR select + instantiation) |
| B1 walker | `comp/b1-walker` @ 959594a+ | 🟡 IN FLIGHT (stage 2 unit suite green — 5,940 emissions bit-exact, 3/3 mutants; stages 3–5 pending; contract gates: CQ-8 tier scope, tap mirror REQUIRED) | B1_WALKER.md stage 5 (WALK CSR window 0x5C+, mode mux, apex_top passive tap mirror) |

**Verified merge facts (2026-07-21):** all three lanes share merge-base
e578872 with this branch; pairwise file overlap between lanes is **zero**
(checked with `comm -12` on `git diff --name-only` sets). Expected conflicts:
none structural. Adjacent-line merges to watch when B1 lands its later
stages: `scripts/gen_status.py` SUITES list (wide-rmsnorm added `asu/wide`;
B1 will add its suite row), `docs/design/LEVEL_C_PARALLEL.md` §Status
(lanes append lines), and `docs/OPTIMIZATION.md` B5 (wide-rmsnorm struck one
clause). All trivial; resolve by keeping both sides.

**W4 adoption eval — n=1000 VERDICT IN (2026-07-21, merged evidence at
`docs/results/b3_w4_adoption/RESULTS.md`; n=10,042 confirmation pending on
the lane):** realization (A) tile-scale — the config the landed RTL geometry
implies — **COLLAPSES the model** (−32.2 pts, z=−17.2). G=32 through a
realization-(B)-style chain costs −0.003 ± 0.008 (z=−0.38) vs the INT8 feed
— the adoption path; G=16 adds nothing. Consequences for this contract:
1. The merged (A) feeder stays INERT — merging was and remains safe; but
   (A) is dead as a product config. **Enabling W4 (I-B) now requires a
   bounded W4-B extension:** G=32 grouping with the host-computed
   stripe-global requant scale consumed as a sideband (the RESULTS.md
   disclosure — a design delta from D-021; no (B) RTL exists today). That
   is a new contracted stage on the B3 lane, spec'd against
   B3_WEIGHT_PATH.md + the RESULTS disclosure, NOT an integration edit.
2. **Perf flag:** G=32 scale overhead ⇒ ~4.5 b/w effective, not 4.125 —
   re-run `perf/apex_perf_model.py`'s W4 assumption and recheck the
   ≥10 tok/s floor before quoting any W4-dependent number (§4b's "W4
   halves the dominant traffic term" needs the honest update).
Gate unchanged: 10k verdict + the W4-B stage land before any CSR enable.

## 2. Merge order (each step gated before the next)

1. **`comp/wide-rmsnorm`** — leaf module + own suite + additive docs. Gate:
   `make -C verif/asu/smoke smoke && make -C verif/asu/sb all && make -C
   verif/asu/wide all` + `make -C golden test`.
2. **`comp/b3-weight-path`** — inert module + golden codec. Gate: `make -C
   golden test` (banner byte-identity: gen_status.py:273 literal-matches
   `golden/Makefile`'s banner — B3 already preserved it; re-verify) +
   `make -C verif/mxe/w4 all`.
3. **`comp/b1-walker`** — LAST, once its lane completes; it's the only lane
   whose stage 5 performs apex_top surgery. Gate: its own suite + `verif/top/l3`
   in BOTH modes with the unchanged-check-count requirement (tap mirror).
4. **Full-matrix** (`scripts/run_full_matrix.sh`, daemonized) on the merged
   tree → STATUS.md regenerates (this is also where the `asu/wide` and B1
   suite rows first appear in STATUS). Byte-identical expectations: every
   pre-existing suite's counts unchanged.
5. **f2sim regression on the merged tree:** `make -C verif/f2sim clean build
   && make -C verif/f2sim run` → must reproduce `files=18 checks=27996
   fails=0` before any hardware step.

## 3. apex_top wiring (I-A) — execute the lane stage-5 specs, add nothing

- **B1 stage 5:** WALK descriptor CSR window at 0x5C+ in the apex_top glue
  (ERR_STICKY 0x58 pattern, apex_top.sv:864-918 incl. the rdata override
  mux), host/walker mode mux driving ds_* + glue/route/KVQ fanouts, and the
  **passive fs_*/ss_* tap mirror** (owner-gated requirement — walker mode
  must keep the L3 check count unchanged; tb hardwires those readys at
  tb_apex_l3.sv:342-343).
- **B3 stage 5:** instantiate `mxe_wfeed_w4` behind a route/CSR select
  (apex_pkg stays frozen — route-level + CSR selection per OPTIMIZATION B3),
  add to `apex_sources.f`, W4 disabled by default until the adoption verdict.
- **RMSNorm: KEEP `RMS_D_MAX=128` for the I-A CL.** RMSNorm is not in the
  attention job path; the wide elaboration is I-B's (hidden-D norm). Cost
  when I-B flips it, already measured (`verif/asu/wide/RESULT.md` synth
  probe): +1 RAMB36E2 + 5 DSP48E2 + ~1.2k LUT on US+; the 45-bit μ-multiply
  path timing must be read from the I-B build's report, not assumed.

## 4. The CDC/clocking build (I-A critical path) — inherited from F2 stage-2

Facts (all from `docs/results/f2_stage2_hw/RESULT.md`, verified):
`clk_main_a0` is FIXED at 250 MHz; the D=128 CL's `u_squant` path is
38.99 ns (~26 MHz); slowest stock AWS_CLK_GEN recipe is 62.5 MHz (B1) —
which does NOT clear it. Reuse pointers: kit Clock_Recipes_User_Guide, the
`cl_mem_perf` AWS_CLK_GEN integration example, the preserved D=128 DCP on
the devbox EBS volume, `scripts/fpga/f2/f2_host_run.py` + the 18 regops.

**DECISION-LC-1 — RESOLVED 2026-07-21 (kit facts adversarially verified,
12-agent recon):** `clk_tile` = AWS_CLK_GEN `o_clk_extra_a1` under **stock
recipe A2 = 15.625 MHz** (64 ns period vs the 38.99 ns `u_squant` path —
60% margin, zero verified-RTL surgery). Both originally-scoped routes were
wrong: dynamic clkgen bottoms out at **87 MHz** (sdk fpga_libs FREQ table —
"~25 MHz dynamic" was never real), and pipelining is unnecessary. Build:
`--aws_clk_gen --clock_recipe_a A2` (hard-gated pair), `aws_clk_gen.sv`
instantiated in the CL per cl_mem_perf, groups B/C/HBM off. RTL LANDED
(9c7ce41): `apex_ocl_cdc.sv` 4-phase bridge at the OCL boundary, entire CL
body wholesale on clk_tile, sim divider (+tile_div; 8 = the exact A2
ratio). Constraints: kit tcl auto-creates recipe clocks; cl_apex_cdc.xdc
adds ASYNC_REG (sync pairs) + max_delay (quasi-static payload buses).
⚠️ Bring-up order (AWS_CLK_GEN_spec.md:165): extra clocks HELD LOW until
MMCM lock — host polls MMCM_LOCK_REG / releases SYS_RST via the clkgen
AXIL (SDA) BEFORE any BAR0 traffic, else the bridge's dead-clock guard
poisons (by design). Gate G-I4 sim half **DONE 2026-07-21** (verify box, pinned Verilator 5.044; durable log `docs/results/levelc_integration/cdc_sim_3ratio_2026-07-21.log`): `F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS` at tile_div = 8 (the exact A2 ratio), 7 (odd), and 2 — cycle counts scale with the ratio (432M/378M/119M shell cycles), proving the tile genuinely ran slow. Remaining G-I4/I5: the same two-clock config through Vivado (DCP w/ --aws_clk_gen --clock_recipe_a A2) post-merge-3/3.

## 5. DRAM weight path (I-B) — RESEARCH RESOLVED 2026-07-21 (verified recon)

The original scoping ("move to the DMA shell") was wrong: **the XDMA shell
does not exist on F2** — the HDK build hard-rejects `xdma_shell` mode; the
Small Shell (0x10212415, 88% usable resources) is the only released shell.
The fuel line is Small-Shell-native and needs NO shell switch:

- **DDR:** one DDR4 DIMM per FPGA, 72-bit (64+ECC), **64 GB**. The
  controller (`sh_ddr.sv`) is instantiated IN the CL (it already is,
  tied off `DDR_PRESENT=0`) — I-B flips to `DDR_PRESENT=1` +
  `` `define USE_64GB_DDR_DIMM`` + the `cl_ddr4.xci` IP, and drives the
  512-bit `cl_sh_ddr_axi_*` AXI4 port set.
- **Host→DDR bulk load (no DMA driver needed):** AppPF **BAR4 is a
  128 GiB prefetchable BAR mapped straight onto the PCIS 512-bit AXI4
  bus** — the host `fpga_pci`-attaches BAR4 and memcpys; the CL decodes
  PCIS addresses to DDR (the kit's `cl_dram_hbm_dma` maps DDR at PCIS
  base `0x0`, 64 GB range — copy that decode). Load-once weights don't
  need SDE/XDMA throughput; even ~100 MB/s loads a 7 GB INT8 image in
  ~70 s, once per boot.
- **Tile-side reader (the real I-B RTL):** a sequential burst reader in
  the shell-clock domain + an async-FIFO crossing into clk_tile feeding
  the `xw` lane8 stream (and later the W4-B packed format). **Layout
  decision: the DDR image IS the wire format** — the host loader
  pre-swizzles weights offline into exactly the per-job beat stream the
  tile consumes, so the reader does zero reformatting; the walker's
  descriptor stream tells it (base, beats) per job.
- **Sim:** extend `sh_ddr.stub.sv` with a behavioral sparse memory model
  (assoc array) so f2sim covers loader→DDR→reader→tile end-to-end.
- Bandwidth sanity: tile @15.625 MHz consumes xw at ≤8 B/cycle ≈ 125 MB/s
  — DDR4 has ~150× headroom; the fuel line is capacity-bound, not
  bandwidth-bound, at demo clocks.
- Small-Shell errata to respect: HBM ECC scrubbing options break AFI load
  (we don't use HBM; noted for completeness).

## 6. Machine & spend plan (standing rule — **APPROVED by owner 2026-07-21**)

Turnkey launcher: `scripts/aws/apex_verify_box.sh` (launch/push/run/stop/
terminate; git-bundle transfer so lane branches never touch the public
remote; bootstrap symlinks `/opt/homebrew/bin/verilator` to the box's
verilator because the Makefiles and mutation_check*.py hardcode that path;
box verdicts are GREEN/RED gates — byte-identical anchor claims stay on the
pinned local 5.044). Launch params validated by EC2 dry-run 2026-07-21
("Request would have succeeded"): AMI ubuntu-24.04 via SSM, sg apex-f2-ssh,
key apex-f2, c6a.xlarge, 40 GB gp3, tags Name=apex-verify-box/apex=verify.

- **Mac = MLX evals only** (the certified instrument is Apple-silicon-bound).
- **AWS c6a-class (≤~$0.70/hr) for every Verilator/yosys job** when the Mac
  is busy; **devbox for Vivado DCPs**; budget cap ~$40 for I-A; instances
  tagged `apex-*`, stopped when idle; NEVER touch the verifagent/Catapult/g5
  instances on the account. Long jobs `nohup setsid`-daemonized always.
- AFI ingestion latency (20 min–hours) is the irreducible serial tail —
  schedule S14 (Sky130/formal) as fill work during those waits, on its own
  box, not on the Mac.

## 7. Acceptance gates (I-A)

| gate | requirement |
|---|---|
| G-I1 | per-merge: merged tree passes the merging lane's full suite + `make -C golden test`, pasted |
| G-I2 | post-merge full matrix green; STATUS.md regenerated; every pre-existing suite's counts byte-identical |
| G-I3 | `verif/top/l3` green in BOTH host and walker modes, walker mode with UNCHANGED check count |
| G-I4 | f2sim 18/18 jobs, 27,996 checks, 0 fails on the merged tree AND under the two-clock configuration |
| G-I5 | ✅ 2026-07-22 — A2 two-clock DCP (attempt 5, 46 m): timing MET, worst +0.711 ns, tile group `clk_out1_clk_mmcm_a` period=64.000 ns; report + build-log excerpt committed in `docs/results/f2_stage2_hw/`. (Attempt-4 lesson: clkgen instance MUST be named `AWS_CLK_GEN`.) |
| G-I6 | ✅ 2026-07-22 — `F2HOST RESULT: files=18 checks=27996 fails=0 -> PASS` on f2.6xlarge (AGFI agfi-0ae06ea568e5667ba, tile at 15.625 MHz via `fpga-load-clkgen-recipe -a 2`); verbatim log + bring-up gotchas (load resets MMCMs to default recipe; per-file TILE_RST executor parity; CLI help recipe table wrong for A2) in `docs/results/f2_stage2_hw/RESULT.md`. **I-A COMPLETE — attention-on-silicon claim landed, scope-locked.** |

## 9. TRANCHE I-B — full layer + DRAM weights (staged plan, appended 2026-07-22 on I-A green)

**Goal (flagship; wording stays banned until evidence):** a full 7B decoder
layer executes on the FPGA with weights streamed from DDR. "Whole-7B-on-FPGA"
is not claimable from any single I-B stage — the claim ladder gets a new rung
only when a full-layer run on silicon has a committed log.

**Three lanes, same discipline as I-A** (separate branches off
`comp/level-c-integration` @ 68359f5, zero expected file overlap, serial
combine back here with per-merge pasted gates; `apex_pkg.sv` stays FROZEN —
route-level + CSR additions only):

| lane | branch | scope | stage-0 deliverable | gates |
|---|---|---|---|---|
| **IB-LAYER** | `comp/ib-layer` | RoPE/SwiGLU/o-proj/residual into apex_top (today TB-side composition per B1_WALKER.md scope note) + wide-RMSNorm elaboration (RMS_D_MAX→3584; synth-probe cost known from `verif/asu/wide/RESULT.md`: +1 RAMB36E2 +5 DSP48E2 +~1.2k LUT; the 45-bit μ-multiply timing must be READ from the I-B build report, not assumed) | lane contract doc (stage list, CSR/route map, golden composition points vs `transformer.py`) | composition suite host-mode bit-exact vs golden; every existing suite byte-identical |
| **IB-WALK** | `comp/ib-walk` | D-028 walker descriptor extension to walk the full layer (B1 designed so this is a descriptor extension, not a rewrite); fold B1b prefetch-overlap in or defer it EXPLICITLY in the contract | lane contract doc + descriptor format delta vs D-028 | `verif/seq_walker` extended w/ mutants; L3 walker-mode UNCHANGED counts; layer-walk golden replay |
| **IB-FUEL** | `comp/ib-fuel` | §5 verbatim: `sh_ddr` DDR_PRESENT=1 + `USE_64GB_DDR_DIMM` + cl_ddr4.xci; BAR4→PCIS host bulk load (copy `cl_dram_hbm_dma` decode, DDR at PCIS 0x0); shell-clock burst reader + async FIFO into clk_tile feeding the `xw` lane8 stream; **DDR image IS the wire format** (host pre-swizzles); `sh_ddr.stub.sv` sparse behavioral model | lane contract doc + DDR image layout spec (per-job base/beats ↔ walker descriptor) | extended f2sim: loader→DDR→reader→tile replays the 18 jobs with weights FROM DDR, counts unchanged |

**Hardware step (after combine):** re-use the I-A bring-up truths verbatim
(RESULT.md gotchas: `AWS_CLK_GEN` naming, `fpga-load-clkgen-recipe -a 2`
after EVERY AFI load + frequency read-back, TILE_RST per-file parity).
Machine/spend plan unchanged (§6): Verilator on the Mac or c6a verify box,
devbox for DCPs, boxes stopped when idle.

### 9.1 Stage-0 reconcile — integration rulings (2026-07-22, all three lane contracts in)

Lane contracts landed: `IB_LAYER.md` (comp/ib-layer 1fce50d) ·
`IB_WALK.md` (comp/ib-walk 747f2f1) · `IB_FUEL.md` (comp/ib-fuel 2d2ad9e).
Rulings on the nine cross-lane questions; lanes proceed to stage 1 under
these. Conflicting text in a lane doc yields to this section.

- **R1 — fuel request record (WALK↔FUEL):** `{base_64B[29:0],
  beats_64B[25:0], tag}` — FUEL's 26-bit beats wins (WALK's proposed 20-bit
  cannot address the 67.9 MB down-proj tensor ≈ 1.06M beats). One record
  per weight-consuming phase in the fmt=1 descriptor (prefetch-friendly).
  WALK updates its field widths; FUEL freezes `fuel_req` at these widths.
- **R2 — I-B claim framing:** RATIFIED as "golden-driven replay on
  silicon" — host-loaded per-step calibration disclosed at
  **~39 MMIO/layer/step (H+11, derived; figure history H+5 → H+6
  [stage-4 fence: RQ patches through DPTR/DDATA] → H+11 [stage-5: the §3b
  JOBC composites are per-step calibration exactly like the RQ pairs — 4
  JC words + a DPTR re-point join the patch]; full history in IB_WALK.md
  §2.5)**; on-tile two-pass amax stays OUT of I-B (named future exit).
  The walker fmt=1 descriptor extension is **D-029**.
- **D-030 SIGNED (owner, 2026-07-23; collected via the integration
  session):** the golden bus-composition mode (C-LBUS — additive, three
  flags default-OFF, OFF-mode byte-identical, 2,008 exact-Fraction
  lemmas, pinned deltas err_ON ≤ err_leg) is ratified. IB-LAYER records
  the sign-off in IB_LAYER.md §0.1 citing this entry; S5 gates against
  BUS_ON from here on.
- **Pre-combine input (owner directive 2026-07-23): B3 fast-forwarded to
  205d6f5** — the CLOSED adoption matrix (every cell n=10,042: final
  recipe W4 G=32/(B) + DIRECT host prep, −1.0 pt vs shipped weights,
  beats the INT8 feed; direct@G16 +0.41 borderline at +11% traffic ⇒
  **feeder group size FROZEN at G=32**, G=16 validated fallback; B3 lane
  closed). Merged clean; gates re-run on the merged tree: golden ALL PASS
  incl. weightcodec + w4 suite 5/5 mutants.
- **Pending combine input — W4B feeder lane (D-031, `comp/w4b-feeder`):**
  at stage 1b lint-clean (owner report 2026-07-23). Becomes a combine
  input ONLY once its TB and sweep land; the combine does not wait for it
  unless the owner says so — if it lands in time, it merges under the
  same per-lane gate discipline (its (B)-sweep requirement per
  B3_WEIGHT_PATH.md §8), else it follows post-combine.
- **W-G2 ✅ GREEN (2026-07-24, combine-8):** the tile gate vs the L4
  harness — (i) segment-2 L4 composition harness on the merged tree
  (`verif/top/l4` compose: 6 cases, 66 EFS checkpoints vs `r.s_h`,
  self-checks 6/6, 3/3 signature-required mutants), (ii) the FIRST
  walkable fmt=1 tile case (`run_walkfmt2`: image loaded through the real
  window incl. deep SRAM words 21/22, WALKED clean, FMT_SUP=0011,
  walk_mmio=10, 1,157 checks / 0), (iii) every pre-existing suite
  byte-identical. Disclosed caveats riding to segment 1: h8 bit-exactness
  (deferred by carve-out — the o-proj GEMM will consume it), hd_ready's
  final arbitration (sticky pre-staged-snoop form landed after the gate
  proved the flip's arm-while-presented tie deadlocks; single-head
  arming until the q-projection path), and `RDATA == r.r1` (the
  segment-1 back-half drive plan is written in IB_LAYER §3c). Also
  landed: l3 run targets gained the missing `build` dependency (a
  stale-binary silently ran the old TB — the banner-grep gate caught it).
- **OWNER DECISION 2026-07-24 — CLOSE THE RTL GAPS BEFORE SILICON.** No
  F2 spend on the W-G3 shape. The next tranche (I-C) closes the gaps that
  bound the claim, then ONE silicon run lands a much larger result. Lanes:
  **IC-QPATH** = gap A (a narrow-AND-rope sink so a rotated q reaches the
  act stage) + F6(ii) (per-head `s_q` staging during a walk) — together
  they unlock the multi-head walk; **IC-BIAS** = gap B (the projection
  bias adder golden applies pre-narrowing). Gap D stays the stage-6 wide
  envelope; F1 is fixed. Silicon (a GQA-4/wide-D CL has never been BUILT,
  only linted) waits for those — and note the DDR path's first real test
  is silicon regardless, since `sh_ddr_beh.sv` is our own model.
- **W-G3 ✅ DECLARED (2026-07-24, combine-14; full package with 18 pasted
  log blocks in `docs/results/wg3/RESULT.md`).** Verified independently on
  the integration tree, not accepted from lane reports. Three results:
  (a) **tile-level walked half** — a real Qwen2.5-7B head (run 19/layer
  19/head 3, hd=128, T=20, CQ-8) walked from an fmt=1 descriptor at build
  point `KVQ_GQA_NENG=4 KVQ_DEPTH=256 RMS_D_MAX=3584 LAYER_DM_MAX=3584`:
  host 5,541 checks / walked 5,544 / 0 errors, walker mode losing no
  checks; (b) **K-rope across all four GQA engines** host-mode, 10,262
  checks / 0 errors, with the F5 legacy-pinned arm RED exactly as
  predicted (promoting F5 from ledger claim to measured tile result);
  (c) **fuel-fed projection GEMM** — real Qwen `Wq` weights arriving
  loader→DDR→reader→afifo→xw→tile with ALL 128 mailbox WB triples
  removed, matching an independently recomputed INT32 reference, with a
  negative-control arm that loads CLEANLY (SHA updated) and can only be
  caught by the result. Provenance: six links from the committed S8 run
  through a double committed-record gate (22 arrays, then 21 fields) to
  the byte-identical DDR image.
  **QUOTABLE:** single-head partial-layer walk at real 7B per-head
  geometry + GQA-4 K-rope store coverage + a fuel-fed projection GEMM,
  in simulation. **NOT CLAIMABLE:** a walked full layer · a walked
  multi-head walk (F6(ii)) · any walked projection (F1's fix is in, but
  gaps A/B/D remain) · a whole projection from fuel (**128 of 3584 input
  channels** — one legal descriptor) · **anything on silicon** —
  `sh_ddr_beh.sv` is our OWN model of an encrypted controller, so the DDR
  path has never met real hardware.
- **⚠️ TILE CAPABILITY GAPS + one DEFECT — found 2026-07-24, they bound
  what I-B can claim (full records: IB_LAYER.md §3c-1/2/3, verified with
  measurements by the segment-1 lane; F1 confirmed independently at
  source by the integration session):**
  - **F1 — DEFECT (D-029 erratum), fix in flight on `comp/ib-walk-f1`:**
    `seq_walker_pkg.sv:302` sizes the walker's N-split by the DESCRIPTOR
    FIELD width (`WALK2_N_JOB = (1<<DIM_W)-1 = 4095`), but the array
    enforces `n_dim <= MXE_N = 8` (`mxe_ctrl.sv:164`) ⇒ **every walked
    projection descriptor at 7B is REFUSED today** (QKV, o-proj, FFN).
    Hidden because `verif/seq_walker`'s scoreboard validates emissions
    against the same constant (self-consistent oracle) and W-G2 walked
    only attention descriptors. Fix = split the overloaded constant +
    make the TB assert descriptor legality against the tile's OWN
    imported limits + a reverting mutant.
  - **A — no narrow-AND-rope sink for `q`:** `rope_row`'s output reaches
    only the KV store path; `apex_scale_quant` MODE_QUANT emits
    unnarrowed, un-rotated `q8/s_q` (modes mutually exclusive). Measured:
    at q_pos=19 all 3584 q elements differ from `q_real` (max |Δ| 55.4).
  - **B — no projection-bias adder exists in any RTL**, while Qwen2.5
    q/k/v projections carry biases (o/gate/up/down do not).
  - **C — Q7 does not narrow:** golden (BUS_ON) narrows the projection
    row to fp16 before quantizing; the tile does not. Measured 70 q_pos=0
    rows: f16-narrowed reference matches `r.s_q` 70/70; exact-product
    (tile semantics) 49/70 — 21 rows off by ±1 ULP. A gate against
    re-derived tile semantics was REFUSED as a lying gate.
  - **D — `CFG_D` is overloaded** across per-head units (`rope_row`, KVQ
    banks, phase RAM) and D_model-wide units (feeder, stage bufs) ⇒ one
    build implies `head_dim == D_model`. This is the stage-6 wide
    envelope surfacing early, not a new defect.
  - **CONSEQUENCE FOR THE I-B CLAIM:** a walked FULL layer is not
    achievable on today's RTL. W-G3's honest shape = **partial-layer
    walk** (NORM1 | ROPE arm | STOREKV | SCORE | PV at real 7B geometry
    with GQA-4 banking, from a committed real trace step) **+ a
    separately fuel-fed projection GEMM** (host-driven, legal n=8, real
    weights from the DDR image). Owner settles the wording when W-G3 is
    green; A/B/C + F1's projection path are the follow-on RTL lane.
  - **F6 — `apex_top` cannot complete a MULTI-HEAD fmt=1 attention walk
    (2026-07-24, W-G3 executor lane; proven by A/B, not asserted).** Two
    independent structural causes: (i) `apex_top.sv:583` instantiates a
    SINGLE `seq_walker_comp u_wcomp` (no generate loop) indexed by record
    address only ⇒ at `kv_map=1` all four KV groups collide in
    `[0,T)/[T,2T)` — **this cache-instance count was NOT in the ledger
    before**; (ii) `hd_ready` is one sticky `hd_sq_seen_q` and the walk
    forces every host job port's ready low, so no later head's q row can
    be staged (the RTL says so itself at :618-621). A/B evidence, same
    walker/vectors/scoreboard/seed, topology the only variable:
    `control-Neng-caches 16139 checks / 0 errors` vs
    `apex_top-as-built 16139 / 980 errors` — all 980 are composite words
    (560/560 CS, 420/560 QS), and the surviving QS are EXACTLY heads
    21-27 = KV group 3, the last-staged group. RULING: (3) re-scope the
    tile-level walked half to `n_heads=1` at real 7B PER-HEAD geometry +
    host-mode GQA-4 K-rope coverage (buildable today, no RTL change) —
    NOW; (1) per-KV-head composite caches in apex_top (pre-validated by
    `tb_walker2_sb`, byte-identical at N_ENG=1) — in parallel, necessary
    but not sufficient; (2) per-head `s_q` — DEFERRED to the segment-1 q
    path; the interim-patch-CSR variant is REFUSED for now because it
    would move the disclosed ~39 MMIO/step figure.
  - **F5 — FALSE-RED TRAP (found building the K-rope gate; binds every
    lane writing a rope/K expectation):** `K_real` is NOT on the fp16
    grid, so the tile's path (rotating the f16 S-2 bus value) matches the
    LEGACY `K_f16`/`K_rope` trace field in only **4/80 measured rows**
    (max Δ 1.95e-3 ≈ 1 fp16 ULP) while matching **`BusMode(rope_in_f16=
    True)` in 80/80**. A gate pinned to the legacy field would fail a
    CORRECT tile in 76/80 rows. **Rule: pin all rope/K expectations to
    the D-030 arbiter, never to the legacy trace field.** (Directly
    affects the segment-1 lane's O3′-1 K-path gate — apply before
    building it.)
  - **F4 — the fuel line cannot feed the norm gammas** (its only sink is
    `xw`; gammas enter via `xg`), so the fenced walk consumes NO DDR
    tensor. This is the RTL-level reason the fuel exercise is a separate
    half, not a convenience choice.
  - **F3 — the step-enable fence is UNPOLICED:** `walk_desc2_check` has
    no dependency clauses, so IB_WALK §2.2's documented
    "en_qkv without en_storekv ⇒ ERR_DESC" does not exist; fence
    correctness is host responsibility wherever it is used.
- **Combine agenda (named reconciliations, owned by the integration
  session at merge time):** (a) WALK's fp16-RNE grade-narrowing of JC
  composite values vs D-030's canonical definition — LAYER states the
  canonical form, WALK conforms; (b) WALK's derived step ORDER
  (arm-before-stream, JC-before-push) vs LAYER's L4 host choreography —
  L4 harness is the arbiter on the merged tree; (c) phase-table residency
  — §3b memories say RESIDENT per config (no per-step phase rows); LAYER
  must strike or reconcile their stage-7 "per-step phase rows" inventory
  line. Plus the contracted remainder: walker2 apex_top instance +
  FMT_SUP→0b0011 flip together, against LAYER glue @ 7fd8822 + FUEL's
  physical reader; W-G2 tile gate vs the L4 harness; FUEL physical-reader
  wiring replaces the stub.
- **Cross-lane build note (WALK stage-4 lesson):** `seq_walker_pkg.sv`'s
  fmt=1 contract declarations trip UNUSEDPARAM under fatal `-Wall` in any
  build that compiles the pkg WITHOUT walker2 (consumers absent) — the
  scoped package waiver in the B1 suite is the sanctioned fix; lanes
  hitting it should copy that waiver, not add `-Wno-fatal`.
- **R3 — KVQ record addressing at H_kv=4 (AMENDED 2026-07-23 per WALK
  stage-1 escalation):** per-KV-head mapping onto banked engines stands,
  but the original shift formula (h >> log2(H/H_kv)) was WRONG — 7B has
  H/H_kv = 28/4 = 7, not a power of two. Adopted mapping = the golden GQA
  slicing `h // (H//H_kv)` (transformer.py), correct at every geometry.
  Sizing confirmed: 2T ≤ 256 per engine, exactly full at T=128 (DEPTH=256
  is the verified L3/f2 build config; the apex_top RTL default is 128).
  Consequence recorded as an IB-LAYER obligation: today's apex_kvq_bank is
  THREE per-TIER engines (D-024); per-KV-head banking needs **H_kv CQ-8
  engine instances** (flat-in-one-engine cannot fit: 4·2T = 1024 > 256).
- **R4 — golden bus-composition mode (LAYER Q1):** additive, flags
  default-OFF, banner byte-identical gate — design approved; the four
  narrowing points get a B3-finding-3-style pinned-delta measurement doc;
  **OWNER SIGN-OFF required at the S1 gate before any RTL consumes it**
  (golden is the arbiter; this is the one I-B decision reserved to the
  owner). Gets ARCHITECTURE decision entry D-030.
- **R5 — control ingress (LAYER Q2):** CONFIRMED glue-CSR-window control,
  zero new apex_top ports. Mux point = the B1 stage-5 idiom: single glue-
  level host/walker mode mux; the walker ROM later drives the SAME glue
  registers the host writes (one ingress point, L3-pattern tap mirror).
- **R6 — wide feeder probe (LAYER Q3):** YES — pull the yosys
  `seam_feeder_quant` D=3584 probe forward (cheap, Mac-safe); chunked-amax
  feeder stays speced as fallback. §9's IB-LAYER row is amended to include
  the feeder elaboration cost as unpriced-until-probed.
- **R7 — FUEL audit counting (FUEL Q1):** lane default ratified — executor
  postlude `FUEL_ERR==0` read per job OUTSIDE the counted stream; printed
  count stays exactly 27,996; the log discloses the postlude.
- **R8 — DDR AR ownership (FUEL Q2):** static LOAD/RUN AR mux per the lane
  draft (CSR-switched while idle; PCIS readback verify retained).
- **R9 — repo seam:** `babaf00` (WALK-window comment fix 0x5C–0x6C) is
  cherry-picked onto this branch (it predated the branch point on
  decoder-layer-and-fpga-fit).
- **Recorded nuance (no action):** LAYER stage 0 verified `rope.sv:193` /
  `residual_add.sv:136` are SV-`real` behavioral datapaths (sim-only) —
  the S2/S3 synthesizable integer re-implementations must be bit-exact vs
  BOTH the behavioral blocks and golden. §1's "decoder-layer RTL
  bit-exact" claims stay true as stated (those blocks were never claimed
  synthesizable/shipped in apex_top); any future wording must not imply
  they were.
- **Machine rule for stage 1+ (all lanes):** at most ONE Verilator build
  at a time across lanes (`pgrep -f verilator` before launching; wait if
  busy); Mac evals still outrank EDA per §6.

## 8. Kickoff prompt for the integration session (paste when B1 lands)

> Read docs/design/LEVEL_C_INTEGRATION.md first — it is your contract; then
> docs/results/f2_stage2_hw/RESULT.md (your CDC inheritance), the three lane
> contracts' stage-5 sections, and docs/results/f2_stage2_hw/RESULT.md. Work on
> `comp/level-c-integration` in the `../apex-integration` worktree. Execute
> §2 merge order with per-step pasted gates; then §3 wiring; then §4
> DECISION-LC-1 with the preserved DCP as your timing baseline. Machine plan
> §6; spend needs owner approval per line. Never report PASS without pasted
> output; apex_pkg stays frozen.
