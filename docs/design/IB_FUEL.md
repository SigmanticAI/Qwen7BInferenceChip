# IB-FUEL — DDR weight fuel line (Level-C tranche I-B, lane contract)

**Status:** 🟢 **s1–s4 DONE 2026-07-22/23; s4b DONE 2026-08-04; s5 HARDWARE
DONE 2026-08-05** (s1 DDR model §1.2; s2 PCIS→DDR decode + loader + image
tool §1.3; s3 reader + afifo + fuel_req + xw mux + FUEL CSRs §1.4; s4 18-job
fuel replay ×3 ratios + 3/3 RED mutants §1.5; s4b 0.5B weights-resident
§1.6). **s5 honest scope note (2026-08-05):** the s5 row's gate as
originally written (the D=128 18-job `F2HOST` replay with weights from the
DIMM) was **not** what ran — the DDR=1 images are the b64 0.5B geometry,
whose INFO_D refuses the D=128 job set by design. What s5 actually proved on
silicon: real DDR4 trains/loads/holds 28.5 MiB of 0.5B weights with full
readback (`agfi-0a345ddb51285e847`,
`docs/results/prompt_on_chip/DDR_BRINGUP_RESULT.md`); then the WALKER-driven
fuel path computed 144/144 QKV blocks from card DRAM bit-exact with walk-off
and one-byte-poison discriminators (`agfi-0183a4b88c8d21163`,
`SELF_RUNNING_CARD_RESULT.md`), and the fuel-fed OPROJ + on-tile requant
epilogue + residual chain r1 896/896 bit-exact (`agfi-0bc20880b50f5faba`,
`E6_ON_SILICON.md`). A host-mode from-DIMM silicon replay of the original
18-job set remains unrun and unclaimed.
**Branch:** `comp/ib-fuel` off `comp/level-c-integration` @ 335dea0.
**Owns (I-B):** `scripts/fpga/f2/cl_apex/` (CL top, mailbox, new fuel RTL,
constraints), `verif/f2sim/`, the DDR image tooling, and this doc. Any other
lane touching those files coordinates through the integration lead first.
**Implements:** `docs/design/LEVEL_C_INTEGRATION.md` §5 **verbatim** (the
resolved DRAM research) per its §9 IB-FUEL row. This doc adds the staged
plan, the DDR image layout spec, the CDC plan, and the fences — it does not
re-litigate §5's decisions.
**§9.1 reconcile (962acca, 2026-07-22) applied:** R1 froze `fuel_req` at
`{base_64B[29:0], beats_64B[25:0], tag}` — **beats count 64 B DDR words**,
superseding this doc's original 8 B lane-beat unit (§2.3 rewritten; the
divisibility invariant it induces is verified there). R7 ratified the
executor-postlude audit (printed count stays 27,996). R8 ratified the
static LOAD/RUN AR mux. Where any older phrasing here disagrees with §9.1,
§9.1 wins.
**Contract discipline:** `apex_pkg.sv` FROZEN (`APEX_VERSION 0x0001_0000`);
no tile-RTL edits (`rtl/` untouched by this lane); golden is the arbiter;
no PASS without pasted output; serialize EDA vs evals per §6 machine plan.

---

## 0. What the fuel line is

Today every weight beat reaches the tile by BAR0 register writes: the host
assembles a 64-bit `lane8` beat in two mailbox registers and pushes it
(`apex_f2_mailbox.sv` 0x040/0x044/0x048 → the tile's `xw` stream,
`cl_apex.sv:586`). That is the stage-2 replay design — correct, and orders
of magnitude too slow to feed a full layer (3 BAR0 writes per 8 B beat ≈
low single-digit MB/s vs the 125 MB/s the tile can drink).

The fuel line moves the weights into the F2 card's 64 GB DDR4 DIMM once,
then streams them to the tile at line rate:

```
host fpga_pci (AppPF BAR4, 128 GiB prefetchable)          [load once]
  └─► PCIS 512-bit AXI4 ─► CL decode (DDR @ PCIS 0x0) ─► sh_ddr ─► DIMM
                                                            │
walker / host (base, beats) ── fuel_req (tile→shell CDC) ──►│  [per job]
                                                            ▼
              sequential burst reader (clk_main_a0, shell domain)
                                                            │
                                  async FIFO (shell → clk_tile CDC)
                                                            ▼
                        xw mux ─► tile `xw` lane8 stream (unchanged port)
```

Two §5 decisions this lane is built around:

1. **The DDR image IS the wire format.** The host pre-swizzles offline into
   exactly the per-job beat stream the tile consumes. The reader does ZERO
   reformatting — it slices 512-bit DDR words into eight 64-bit beats and
   pushes them in order. Format changes (e.g. B3's W4-B packed stream) are
   image-generator changes, never reader changes.
2. **No shell switch.** The XDMA shell does not exist on F2; the Small
   Shell is the only released shell, and the DDR controller (`sh_ddr`)
   lives IN the CL. `cl_apex.sv:54` already instantiates it tied off via
   the kit's `unused_ddr_template.inc` (`DDR_PRESENT(0)`, AXI zeroed,
   stat-ack forced 1). I-B flips it real.

Kit facts re-verified against the pinned checkout (aws-fpga v2.3.3,
`tools/aws-fpga`, gitignored — fetched on the devbox by `hdk_setup.sh`):

- BAR4 = 64-bit prefetchable, **128 GiB**, maps the memory space onto PCIS
  (`hdk/docs/AWS_Shell_Interface_Specification.md:128,216`); host side is
  plain `fpga_pci` attach + memcpy — no DMA driver.
- PCIS into the CL: 512-bit AXI4, 64-bit addr, 16-bit ids
  (`cl_ports.vh:174-219`); today tied off by
  `unused_dma_pcis_template.inc` (`cl_apex.sv:59`).
- `sh_ddr` port set: one 512-bit AXI4 (`cl_sh_ddr_axi_*`, ADDR 64, ID 16,
  LEN 8), DDR4 x72 pins, stat bus, `sh_cl_ddr_is_ready`
  (`sh_ddr.stub.sv:19-119`). The synth model is encrypted; the stub is the
  open port surface. The stat bus MUST be wired through when
  `DDR_PRESENT=1` — DDR training runs over it (`cl_ports.vh:145-150`).
- 64 GB DIMM recipe (`hdk/docs/Supported_DDR_Modes.md`): `DDR_PRESENT(1)`
  + `` `define USE_64GB_DDR_DIMM`` in the CL top + read
  `${HDK_IP_SRC_DIR}/cl_ddr4/cl_ddr4.xci` in the synth tcl (the
  `cl_dram_hbm_dma` synth script line 46 is the copy source). The cl_ip
  IP tree is a devbox-side submodule — not present in the local checkout.
- PCIS→DDR decode precedent: `cl_dram_hbm_dma` maps **DDR at PCIS base
  0x0, 64 GB range** (`design/cl_dma_pcis_slv.sv:255`). We copy the decode
  idea, not the 2x2 SmartConnect — we have one target.
- Erratum inherited from §5: HBM ECC scrubbing options break AFI load. We
  never touch HBM. Noted, fenced.

Sizing sanity (recomputed, not quoted): tile @ 15.625 MHz consumes `xw` at
≤8 B/tile-cycle = **125 MB/s**. A single-outstanding 4 KiB-burst reader at
250 MHz sustains ≥ ~8 GB/s (64 data cycles + ~60 latency per burst) — ≥60×
headroom; raw DDR4 is ~150×. A Qwen2.5-7B layer is ~233 MB INT8 ≈ 3.64M
64 B words; 28 layers ≈ 6.1 GiB ≪ 64 GiB. The fuel line is capacity- and
correctness-bound, not bandwidth-bound, at demo clocks.

## 1. Staged plan — every stage gated, no stage claims the next

Baseline (stage 0, pasted below §1.1): the UNMODIFIED 18-job f2sim replay
on this branch's base. Every later stage re-runs it; the counts are the
regression invariant.

| stage | deliverable | gate (pasted output required) |
|---|---|---|
| **s1** ✅ DONE 2026-07-22 (§1.2) | Behavioral DDR model: NEW `verif/f2sim/sh_ddr_beh.sv` (module `sh_ddr`, port-identical to the kit stub; `DDR_PRESENT=0` → inert exactly like the stub, `=1` → sparse assoc-array memory behind a proper AXI4 slave: INCR, ≤4 KiB, ID-reflect, plusarg-seeded ready backpressure + latency). f2sim Makefile `SH_DDR :=` points at it (kit checkout stays untouched). | (a) directed AXI write/readback smoke green; (b) 18-job host-mode replay **byte-identical counts** `files=18 checks=27996 fails=0` with the model instantiated inert |
| **s2** ✅ DONE 2026-07-22 (§1.3) | PCIS→DDR write path in the CL: drop `unused_ddr_template.inc` + `unused_dma_pcis_template.inc` from `cl_apex.sv`, instantiate `sh_ddr` (`DDR_PRESENT` build-selectable) with the stat bus wired and `is_ready` exposed; PCIS decode DDR@0x0 (64 GiB window; outside → DECERR, never wedge; AW/W/B always PCIS-owned; AR/R ownership per §4 mux). Sim loader: `sim_main.cpp` PCIS master model, `+ddr_image=<bin>` `+ddr_manifest=<json>` loads before regops. Image tool: `scripts/fpga/f2/make_ddr_image.py` (§2 format) + byte-level cross-check vs the WB stream extracted from the regops. | (a) image loads, PCIS readback byte-identical, manifest sha256 verified in-sim; (b) 18-job host-mode replay counts unchanged (decode present, fuel unused) |
| **s3** ✅ DONE 2026-07-22 (§1.4) | Tile-side fuel path: shell-domain sequential burst reader (single-ID, INCR, 4 KiB-bounded, one outstanding), `apex_afifo` (gray-pointer dual-clock FIFO, §4) shell→tile, `fuel_req` 4-phase tile→shell request crossing, `xw` mux + FUEL CSR window (§5 draft map). **Host-stream path stays the reset default and byte-identical.** | (a) 18-job host-mode replay counts unchanged at `+tile_div` ∈ {8, 7, 2} (the LC-1 three-ratio discipline); (b) directed single-job fuel smoke green |
| **s4** ✅ DONE 2026-07-22 (§1.5) | End-to-end fuel gate: `scripts/fpga/f2/trace_to_fuel.py` (imports `trace_to_regops`, replaces each job's WB triples with one (base,beats) fuel program, all other ops byte-identical; default `trace_to_regops.py` output stays untouched) → f2sim replays **all 18 jobs with weights FROM DDR**. Plus ≥3 mutants, each must turn the run RED: reader lane-slice order swapped (the historic `lane8_beat_t` cast bug, resurrected on purpose), beats off-by-one, afifo pointer-width/wrap bug. | `F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS` in fuel mode at `+tile_div` ∈ {8, 7, 2}, all 27,996 original checks present, `FUEL_ERR==0` read per job in the executor POSTLUDE outside the counted stream (§9.1 R7 — printed count stays exactly 27,996, log discloses the postlude); host-mode default still 27,996; mutants 3/3 RED |
| **s4b** ✅ DONE 2026-08-04 (§1.6) | **0.5B/D=64 + WEIGHTS RESIDENT.** First `DDR_PRESENT=1` twin at the demo geometry (D=64 GQA=2 DM=896 QSTAGE=14 DMODEL=64); every s1–s4 gate re-run there. NEW tensor-granular image tool `scripts/fpga/f2/make_weight_image.py` (real golden-prepared Qwen2.5-0.5B tensors laid out in the walker fmt=1 `TENS_*` order, per-tensor manifest + sha256), the card-side BAR4 loader `scripts/fpga/f2/f2_ddr_load.py`, and `scripts/fpga/f2/fuel_proj_05b.py` — projection GEMMs whose weight beats come out of the resident tensor. §4.1 of the results doc also closes the s5 build gap (`USE_64GB_DDR_DIMM` + `cl_ddr4.xci` in `synth_cl_apex.tcl`, guarded, UNVERIFIED). | `docs/results/ib_fuel_05b/RESULTS.md`: 0.5B host-mode DDR=1 vs DDR=0 byte-identical (1052 checks); 0.5B fuel replay 7 files / 1052 checks / 0 fails / 7 FUELAUDIT ok; 20-tensor 28.5 MiB image `DDRLOAD RESULT: regions=20 words=466288 rb_fails=0 sha_fails=0 -> PASS`; `FUEL PROJECTION GATE: PASS` (8/8 bit-exact fuel-fed, mutated-DDR host-fed GREEN, mutated-DDR fuel-fed RED by the predicted +381) |
| **s5** ✅ **DONE 2026-08-05 — with a DEVIATION, see the Status note above** (the DDR=1 hardware landed at the b64 0.5B geometry via the DDR bring-up + walked-fuel flights, NOT via this row's 18-job D=128 from-DIMM replay, which remains unrun) | **OWNER-GATED hardware** (no spend without explicit owner approval per §6): DCP with `DDR_PRESENT=1` + `USE_64GB_DDR_DIMM` + `cl_ddr4.xci` + the LC-1 A2 clkgen pair; BAR4 load + readback + fuel-mode silicon replay via extended `f2_host_run.py`. Inherits the I-A bring-up truths **verbatim** (`docs/results/f2_stage2_hw/RESULT.md` gotchas): clkgen instance named `AWS_CLK_GEN` or the recipe silently does not apply; `sudo fpga-load-clkgen-recipe -S 0 -a 2` after EVERY AFI load + frequency read-back before BAR0; per-file TILE_RST executor parity; MMCM-lock before BAR0 or the OCL bridge poisons by design. New DDR-specific: poll `FUEL_STAT.ddr_ready` (DDR calibration) before loading; loader writes whole 64 B words only (§2 padding — no partial-strobe RMW against uninitialized ECC); load+verify DDR works with the tile held in reset (decode is all shell-domain — bring-up order freedom). | `F2HOST RESULT: files=18 checks=27996 fails=0 -> PASS` with weights from the DIMM, session log committed under `docs/results/` |

Anti-fabrication: a stage is DONE when its gate output is pasted into this
doc (or a committed `docs/results/` log it links). Nothing in this table is
claimable from the stage before it.

### 1.1 Stage-0 baseline (run on this branch base, 2026-07-22)

Local pinned Verilator 5.044, kit checkout + committed regops inputs from
the main working copy (read-only), `+tile_div=8` (the A2 ratio), unmodified
`verif/f2sim` at 335dea0 — last lines of `run_d128_div8.log`:

```
[.../job_s151_L04_h07_c1.regops.jsonl] 32845 ops, 575 checks, 0 fails (cyc 432469308)
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS
```

Same counts as `docs/results/f2_stage2_sim/` and the I-A silicon run; the
432M shell-cycle figure matches the committed three-ratio CDC log
(`docs/results/levelc_integration/cdc_sim_3ratio_2026-07-21.log`, div=8
row). This is the invariant every fuel stage re-runs.

### 1.2 s1 gate evidence (2026-07-22, local pinned Verilator 5.044)

Gate (a) — `make -C verif/f2sim behsmoke`, the directed AXI smoke on
`sh_ddr_beh.sv` alone, run twice (default knobs; then LFSR backpressure
`+ddr_stall=48879 +ddr_lat=3`) — both invocations:

```
DDRBEH RESULT: checks=801 fails=0 -> PASS
DDRBEH RESULT: checks=801 fails=0 -> PASS
```

(801 = is_ready + 5 write bursts × {bid,bresp} + 79 read beats × {512-bit
data, rid, rlast} across single/8-beat/64-beat bursts, poison-read of an
unwritten region, partial-strobe RMW-vs-poison, and an overwrite.)

Gate (b) — full f2sim rebuild with `SH_DDR := sh_ddr_beh.sv` substituted
for the kit stub, 18-job replay at `+tile_div=8`:

```
[.../job_s151_L04_h07_c1.regops.jsonl] 32845 ops, 575 checks, 0 fails (cyc 432469308)
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS
```

Stronger than the counts gate: `diff` of the full 18-file run log against
the stage-0 baseline log (same tree minus the substitution) is
**byte-identical** — every per-file ops/checks/cycle line matches, so the
inert branch is exactly the stub in 2-state sim, as designed.

### 1.3 s2 gate evidence (2026-07-22, local pinned Verilator 5.044)

Offline image build + cross-check (`make_ddr_image.py` from the 18
committed regops; independent digest agreement manifest = hashlib =
system `shasum -a 256` on a sampled region):

```
DDRIMG: jobs=18 words=47328 bytes=3028992 -> build/ddr_image
DDRIMG CHECK: jobs=18 fails=0 -> PASS
```

Gate (a) — loader→DDR→readback→hash in-sim (DDR=1 build; PCIS master in
`sim_main.cpp` writes the image through `apex_pcis_slv` into `sh_ddr_beh`,
reads every region back over PCIS, byte-compares vs the file, and checks
SHA-256 of the READBACK stream — C++ implementation, self-tested — against
the Python-hashlib manifest digests):

```
DDRLOAD RESULT: regions=18 words=47328 rb_fails=0 sha_fails=0 -> PASS
```

Gate (b) — 18-job host-mode replay with the decode present (DDR=1),
`+tile_div=8`:

```
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS
```

Full run log **byte-identical** to the stage-0 baseline (`diff` clean,
final cyc 432469308). DDR=0 parity build (today's default DCP config,
decode idling at the old template tie-off values) — same result, and its
full run log ALSO diffs byte-identical to the baseline:

```
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS
```

Behsmoke regression on the s2 tree: `checks=801 fails=0 -> PASS` both
knob sets. Disclosed not-yet-exercised: the out-of-range DECERR path in
`apex_pcis_slv` is structurally present but no gate drives it yet — the
s3 directed fuel smoke adds an out-of-range probe. (Closed in §1.4.)

### 1.4 s3 gate evidence (2026-07-22, local pinned Verilator 5.044)

New RTL: `apex_afifo.sv` (gray-pointer dual-clock FIFO, W=64 D=512),
`apex_fuel_reader.sv` (shell-domain burst reader, single-ID INCR ≤4 KiB
one-outstanding), `apex_fuel_ctl.sv` (tile-domain CSRs + 2-deep skid +
4-phase tile half + drain + stickies), the R8 AR mux + `rd_block` +
`xw` mux in `cl_apex.sv`, and the §4 constraint additions applied to
`cl_synth_user.xdc`. Tools: `trace_to_fuel.py` (WB triples → one DDR
record per job; counted checks per job untouched), executor `+fuel_audit`
postlude (R7) and `+ddr_decerr_probe`.

DECERR probe (loader → probe, DDR=1; closes the §1.3 disclosure — OOR
write DECERR, OOR read DECERR + poison payload + rlast, RUN-mode PCIS
read blocked DECERR while writes pass, FUEL_ERR clean after all):

```
DDRLOAD RESULT: regions=18 words=47328 rb_fails=0 sha_fails=0 -> PASS
DDRPROBE RESULT: fails=0 -> PASS
```

The probe EARNED its keep: first run caught `apex_fuel_ctl`'s
`fuel_src`/`ar_owner` output ports undriven (readback reads the internal
regs, so BAR0 looked right while the exported level never left the
module; `-Wno-fatal` swallowed the UNDRIVEN warning). Fixed + rerun
green.

Directed single-job fuel smoke — job_s019 replayed with weights FROM DDR
(prologue: ddr_ready poll, base/beats, one CTRL write = mode+tag+GO), R7
executor postlude outside the counted stream:

```
[.../job_s019_L19_h03.fuel.regops.jsonl] 11862 ops, 507 checks, 0 fails (cyc 3828092)
FUELAUDIT [...] err=00000000 stat=00000009 -> ok
F2SIM RESULT: files=1 checks=507 fails=0 -> PASS
```

507 = exactly the host-mode count for that job (baseline line: 27,601
ops, 507 checks — the removed ops are the uncounted WB triples).

Gate (a) three-ratio host-mode replay, DDR=1 build, fuel untouched
(reset defaults):

```
div8: F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS  (cyc 432469308,
      full log BYTE-IDENTICAL to the stage-0 baseline)
div7: F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS  (cyc 378411149)
div2: F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS  (cyc 118946062)
```

div7/div2 anchor lines (c0/c1/RESULT) match the committed three-ratio CDC
log field-for-field (`docs/results/levelc_integration/
cdc_sim_3ratio_2026-07-21.log`: 372105059/378411149 and
116962358/118946062) — the fuel RTL is cycle-invisible in host mode at
every ratio. DDR=0 parity build (today's DCP config, fuel RTL inert):

```
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS   (full log
BYTE-IDENTICAL to the stage-0 baseline)
```

Behsmoke regression on the final s3 tree: `checks=801 fails=0 -> PASS`
both knob sets.

### 1.5 s4 gate evidence (2026-07-22, local pinned Verilator 5.044)

**The gate (§9 wording — counts UNCHANGED at 27,996, R7 postlude
disclosed):** full 18-job fuel-mode replay, every weight beat FROM DDR
through loader→decode→sh_ddr→reader→afifo→xw mux, at all three tile
ratios. Each run: 18 per-file lines, printed total exactly 27,996, and 18
`FUELAUDIT ... -> ok` lines OUTSIDE the counted stream. Invoked with a
plain `> log 2>&1` redirect (no pipe — exit code unmasked):

```
div8: F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS   (final cyc 263401884)
div7: F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS   (final cyc 230492021)
div2: F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS   (final cyc  70722638)
```

Extra (disclosed, beyond the gate): the same 18-job fuel replay under
DDR-side LFSR backpressure `+ddr_stall=48879 +ddr_lat=40` —
`files=18 checks=27996 fails=0 -> PASS` (final cyc 263462908, ~61k cycles
longer than plain div8: the reader/afifo flow control genuinely stalled
on backpressure, and the tile still got every beat in order).

**Mutants — the committed pipefail-safe gate `verif/f2sim/mutants.sh`
(`make -C verif/f2sim mutants`).** Each mutation is applied ALONE to the
clean tree, built into a throwaway `obj_mut` (the real gate binary
untouched), and the sim's OWN exit code is captured via `${PIPESTATUS[0]}`
under `set -o pipefail` — `| tail` cannot mask a surviving mutant. A
clean-binary CONTROL must exit 0 first, so a harness that failed
everything could not launder a survivor. An `EXIT` trap reverts all three
RTL files unconditionally. Verbatim:

```
GATE[control]: exit=0 want=PASS got=PASS  OK   (checks=507 fails=0 -> PASS)
GATE[M1]:      exit=1 want=FAIL got=FAIL  OK   (checks=507 fails=469 -> FAIL)
GATE[M2]:      exit=1 want=FAIL got=FAIL  OK   (checks=507 fails=128 -> FAIL)
GATE[M3]:      exit=1 want=FAIL got=FAIL  OK   (checks=9   fails=1 [ABORTED] -> FAIL)
MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS
```

| mutant | mutation | failure signature |
|---|---|---|
| M1 | reader lane-slice swap (`fw_data = unpk_q[511:448]` — the historic lane8 cast-bug class) | wrong data, 469 result-check fails |
| M2 | record beats off-by-one (`fr_beats_64B + 1`) | result stream displaced one RO word (each `got` = previous `want`) AND the R7 postlude flags fifo residue (`stat=…8 -> FAIL`) — two independent layers fire |
| M3 | afifo read-pointer wrap bit lost | empty-compare breaks after the first 512-beat wrap, xw starves, choreography poll stalls (job ABORTS) |

3/3 RED via the sim's exit code (not just printed content), three
distinct signatures (value / ordering+residue / starvation) — the fuel
check surface covers data, ordering, and flow-control faults; and the
control proves the gate is not a fail-everything harness.

**Audit note (2026-07-23, exit-code-masking sweep, disclosed):** the
Makefile gates (`run`/`behsmoke`/`build`) are protected by
`.SHELLFLAGS := -o pipefail -c`, and the fuel ladder above used redirects
(no pipe), so neither could mask a failure. The exposure was the *original*
mutant runs: ad-hoc `./f2sim … | tail` under zsh (no pipefail), where the
verdict was read from printed `-> FAIL` content while `$?` reflected
`tail`. That is exactly the flagged failure class. Fix: the mutant gate is
now the committed `mutants.sh` with `set -o pipefail` + `${PIPESTATUS[0]}`
+ a clean-binary control + an EXIT-trap revert, re-run above with the exit
code independently confirming every verdict. (The rewrite itself had a
path bug that stranded M1 on disk mid-run; caught by `git status`,
reverted, and the EXIT trap added so it cannot recur.)

Executor note (disclosed): the very first ladder launch passed all
plusargs as ONE argv entry (zsh does not word-split unquoted `$VAR`); the
executor refused (`need BOTH +ddr_image= and +ddr_regions=`) — fail-loud,
never run a half-configured gate — and was relaunched with explicit args.

### 1.6 s4b gate evidence (2026-08-04) — see `docs/results/ib_fuel_05b/RESULTS.md`

That doc carries the pasted output; only the headline lines are repeated here.

```
DDRBEH RESULT: checks=801 fails=0 -> PASS                         (both knob sets)
0.5B host-mode  DDR=1  F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS
0.5B host-mode  DDR=0  F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS   (logs diff clean)
0.5B FUEL       DDR=1  F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS   (7/7 FUELAUDIT ok)
18-job FUEL   div5     F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS (18/18 ok)
MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS
CAPGATE: PASS (caps=505 values_matched=505/505)
WEIGHTIMG: layers=[0,1] tensors=20 bytes=29851648 (28.5 MiB)
DDRLOAD RESULT: regions=20 words=466288 rb_fails=0 sha_fails=0 -> PASS
FUEL PROJECTION GATE: PASS
```

Three findings that change what a later stage may assume, all detailed in §4
of the results doc:

1. **A projection program's xw stream is only half weights.** At D=64/K=896 a
   `gemm_job` projection emits 1792 WB triples, 896 of them the ACTIVATION
   (`inject_jobs` drives the same external xw port). `trace_to_fuel.py`'s
   whole-stream replacement is therefore correct ONLY for the attention jobs;
   projections fuel the TAIL (mode switch while the reader is idle).
2. **`gemm_job.stage_plan`'s `k*`-in-lane-0 permutation is activation-dependent**,
   so its weight stream cannot be a slice of a resident tensor. The image's own
   CFG_DM=64 C-1 feeder framing (`quant_rows_i8` per 64-wide frame → per-row
   amax 127 by construction) removes the permutation and is what makes
   weights-resident work today.
3. **`seq_walker_fmt.K_JOB = 2048` is tile-illegal at D=64** (2048/64 = 32
   stage rows > `apex_stage_buf` `R_MAX` 31, `cl_apex.sv:260`). For 0.5B only
   `Wd` (4864×896 → chunks 2048/2048/768) is affected. NOT fixed here —
   `jobs()` is IB-WALK's frozen decomposition and changing `K_JOB` re-orders
   `Wd`'s image bytes. `make_weight_image.py` records the chunking and refuses
   to be quiet about it.

## 2. DDR image layout spec — "the image IS the wire format"

### 2.1 One sentence

The per-job image region is the job's `xw` beat sequence — the exact 64-bit
values the stage-2 host writes as WB `{XW1,XW0}` pairs, in script order —
stored as little-endian u64s, padded to 64 B; the reader replays them
verbatim.

### 2.2 Beat format (normative)

- 1 beat = 8 bytes = one `lane8_beat_t.data[63:0]` payload
  (`rtl/apex_pkg.sv:57-60`: 8 packed INT8 lanes; **lane j = data[8j+7:8j]
  = image byte 8k+j** of beat k). Frozen package, so frozen format.
- `lane8_beat_t.last` is NOT stored. The reader drives `last=0` on every
  beat — exact parity with the trace path (`trace_to_regops.py:156`, "WB
  pushes last=0, TB parity"; `last` is informational per apex_pkg). No
  per-record last-assert exists in I-B; the record's spare bits became the
  §9.1 R1 `tag` (opaque, never data-interpreted).
- 1 DDR word = 64 B = one 512-bit AXI beat = 8 lane8 beats; beat k of a
  word occupies `wdata[64k+63:64k]` (AXI little-endian byte lanes) = file
  bytes 8k..8k+7. An x86 host memcpy of the image file through BAR4 lands
  byte-identical — no host-side swizzle beyond writing the u64 sequence.
- W4-B hook: when B3's W4-B lands, its packed stream flows through this
  same byte pipe; `make_ddr_image.py` grows a codec, the reader does not
  change. No W4 assumption anywhere in fuel RTL.

### 2.3 Region + record rules (normative — units per §9.1 R1)

- **Record** = `(base_64B, beats_64B, tag)`:
  - `base_64B[29:0]`: DDR byte address **>> 6** (region 64 B aligned by
    construction; 30 bits cover 64 GiB).
  - `beats_64B[25:0]`: region length in **64 B DDR words** (§9.1 R1 —
    supersedes this doc's original 8 B lane-beat count). `1 ≤ beats_64B ≤
    2^26-1` (max region 4 GiB; the 67.9 MB down-proj tensor ≈ 1.06M words
    — the case that killed WALK's 20-bit proposal — fits 63×; a full 7B
    layer ≈ 3.64M words fits 18×). `beats_64B=0` refused → sticky
    `FUEL_ERR.desc`.
  - `tag[7:0]`: opaque record tag, reflected in `FUEL_STAT` for audit;
    semantics owned by the D-029 fmt=1 descriptor (IB-WALK). The reader
    never interprets it.
- **Whole-word streaming — the divisibility invariant.** The reader
  streams exactly `beats_64B × 8` lane8 beats; there is NO tail-lane
  discard. A region's lane-beat count must therefore be ≡ 0 (mod 8) —
  equivalently its byte length ≡ 0 (mod 64) with no padding bytes ever
  entering the stream. This is an invariant of the real streams, not a
  new restriction: every row-granular INT8 tensor row in I-B scope is a
  64 B multiple (D=64/128-byte attention rows, hidden 3584, FFN 18944),
  and all 18 committed jobs verify — per-job lane-beat counts all ≡ 0
  (mod 8), checked 2026-07-22 against the committed regops.
  `make_ddr_image.py` hard-errors on violation (such an error is a format
  bug — escalate to the integration lead, do not pad into the stream).
- Full-word-only writes by the loader (the image is inherently whole
  words) — also keeps ECC out of read-modify-write against uninitialized
  memory on hardware.
- 4 KiB base alignment RECOMMENDED (the reader handles 64 B-granular
  bases — first burst trims to the 4 KiB boundary; AXI bursts never cross
  4 KiB).
- **Image file** = concatenation of regions in manifest order.
  **Manifest** (`ddr_image.json`): `{format: 1, jobs: [{job, base_64B,
  beats_64B, tag, sha256}]}` — sha256 per region is the load-verify
  contract.
- Measured envelope for the s4 gate (counted from the committed regops:
  `grep -c '"a":12352'`, one XW0 write per lane beat): 18 jobs, 378,624
  lane beats = **47,328 words ≈ 2.9 MiB**, per-job 432..4,112 words —
  trivially inside the behavioral model and the DIMM.

### 2.4 Descriptor handshake with IB-WALK

Interface owned here; descriptor internals owned there. **FROZEN per §9.1
R1** (freeze authorized there; IB-WALK is updating its field widths to
match):

- ONE request channel, `fuel_req` (tile domain), valid/ready, strictly
  in-order, **one record outstanding end-to-end** (matches D-006 job
  serialization). Payload = 64 bits, bit positions frozen now:

  | bits | field |
  |---|---|
  | [29:0] | `base_64B` |
  | [55:30] | `beats_64B` |
  | [63:56] | `tag` (opaque; D-029 semantics; tag width 8 is this lane's freeze) |

  **Producer verified + CONFIRMED 2026-07-22:** IB-WALK's
  `seq_walker_pkg.sv` `walk2_freq_t` (comp/ib-walk 7766e49) packs
  `{tag, beats64, base64}` MSB-first — every field at these exact
  positions, 64 B-word units, with a do-not-reorder note. Their tag value
  = `WALK2_TENS_*` tensor index (10 tensors, [3:0] informative today;
  [7:0] is this lane's width) — consistent with fuel semantics: the
  reader NEVER interprets tag, it is carried and echoed in
  `FUEL_STAT.last_tag` only. "One record per weight-consuming phase"
  realizes as one record per TENSOR consumed (QKV issues 3, template
  order) — plain N-records-in-order, already legal here.
  **s3 addendum from their reservation:** their comment reserves
  issue-ahead capability ("FIFO >= 2" — prefetch stays a deferred walker
  FSM change). The s3 `fuel_req` ingress is therefore a **2-deep skid**:
  it may ACCEPT a second record while the first streams; processing stays
  strictly serial and in-order (one burst in flight to DDR). No bit or
  ordering change.

- Two producers, muxed by walk mode exactly like the existing host/walker
  `ds_*` mux pattern: (a) host CSRs `FUEL_BASE/FUEL_BEATS/GO` (s3/s4, and
  forever — host mode remains the debug path); (b) the IB-WALK D-029
  fmt=1 descriptor's records — **one record per weight-consuming phase**
  (§9.1 R1, prefetch-friendly for B1b overlap later).
- Records are presented in consumption order; N records per job is legal;
  beat order across a job's records must equal the wire stream.
- The s4 gate uses the host-CSR producer with one record per job, so
  IB-FUEL completes without any IB-WALK dependency.

## 3. Error/audit model (the MB_STATUS lesson, kept)

An unread audit register is not an audit (`docs/results/f2_stage2_hw/RESULT.md`
lesson 4). Fuel adds sticky W1C `FUEL_ERR` bits, each checked at end of
job in fuel mode (placement per Q1): `axi_err` (any RRESP≠OKAY — record
aborted, reader idles), `desc` (beats=0 / range overflow), `mode_switch`
(src mux or AR-owner changed while reader busy, FIFO nonempty, or request
pending), `host_push` (mailbox XW_PUSH while fuel mode selected). Underrun
and overrun cannot exist by construction (valid/ready both sides; the
reader deasserts `rready` mid-burst when the unpack stage backs up —
legal AXI).

## 4. CDC plan — one new data crossing, one new control crossing

Domains after LC-1: shell `clk_main_a0` (fixed 250 MHz) / tile `clk_tile`
(A2 = 15.625 MHz). Everything fuel adds on the DDR side — PCIS decode,
`sh_ddr`, burst reader, unpack — lives WHOLLY in the shell domain (`sh_ddr`
runs on `clk_main_a0` exactly as in `cl_dram_hbm_dma`). The tile side —
FUEL CSRs, `xw` mux — lives WHOLLY behind the existing `apex_ocl_cdc`
bridge on `clk_tile`. New crossings:

1. **Data, shell→tile: `apex_afifo`** — new hand-rolled dual-clock FIFO
   (gray-coded pointers, 2FF synchronizers named `*_meta`/`*_sync` to
   match the `apex_ocl_cdc` constraint idiom), W=64 (beat payload), depth
   512 (one 4 KiB burst, one BRAM36). Hand-rolled, not XPM: f2sim stays
   pure open-source Verilator, and the constraint surface stays explicit —
   same reasoning as `apex_ocl_cdc.sv`, which is the pattern to copy
   (payload-stable-before-flag, 2FF flag sync, build owns metastability).
   Verified at `+tile_div` ∈ {8, 7, 2} like the bridge was.
2. **Control, tile→shell: `fuel_req`** — the `apex_ocl_cdc` 4-phase toggle
   req/ack recipe verbatim: payload `{base_w, beats}` latched stable
   BEFORE req toggles, single outstanding, ack after the reader accepts.
3. **Status, shell→tile:** single-bit levels only (`ddr_ready`,
   `rdr_busy`), each through its own ASYNC_REG 2FF pair. No multi-bit
   async counts anywhere — occupancy debug uses the FIFO's own
   domain-native flags.

**Constraint additions** (listed now, applied in s3 — NOT edited this
stage). ⚠ File-name truth: the landed constraints file is
`scripts/fpga/f2/cl_apex/constraints/cl_synth_user.xdc` — the LEVEL_C doc's
"cl_apex_cdc.xdc" is the same object; the synth tcl reads ONLY
`cl_synth_user.xdc` (`synth_cl_apex.tcl:41`), so the additions go there,
not into a new file that would never be read:

```
# fuel afifo: gray-pointer crossings (both directions)
set_property ASYNC_REG TRUE  [.. *u_fuel_fifo/wptr_meta_reg* / wptr_sync_reg*]
set_property ASYNC_REG TRUE  [.. *u_fuel_fifo/rptr_meta_reg* / rptr_sync_reg*]
set_max_delay -datapath_only 4.0  -from [.. wptr_gray_reg*] -to [.. wptr_meta_reg*]
set_max_delay -datapath_only 4.0  -from [.. rptr_gray_reg*] -to [.. rptr_meta_reg*]
# fuel afifo: RAM write-domain cells -> read-domain output register
set_max_delay -datapath_only 16.0 -from [.. *u_fuel_fifo/mem_reg*]
# fuel_req 4-phase (tile->shell), apex_ocl_cdc idiom
set_property ASYNC_REG TRUE  [.. *u_fuel_req/req_meta_reg* / req_sync_reg* / ack_*]
set_max_delay -datapath_only 4.0  toggle flags; 16.0 payload {base_w,beats} regs
# status levels (shell->tile): ASYNC_REG pairs + set_false_path -to first FF
```

Exact cell patterns finalize in s3 against the netlist names (the existing
header's rule applies: values reviewed at the first build against the
timing report; datapath_only bounds deliberately tighter than the protocol
needs).

## 5. FUEL CSR window (AS IMPLEMENTED, s3 — `apex_fuel_ctl.sv` +
`cl_apex.sv` bridge decode)

Lives in the BRIDGE window 0x0 (tile domain), next free space after 0x18 —
`csr_regs.sv` and the walker's CSR-space WALK window (0x5C+) untouched.

| addr | reg | bits |
|---|---|---|
| 0x0020 | FUEL_CTRL | [0] src (0=host mailbox xw — RESET DEFAULT, byte-identical path; 1=DDR fuel) · [1] ar_owner (0=PCIS LOAD, 1=reader RUN — §9.1 R8 static mux) · [8] GO (W1: pushes {tag from THIS write, FUEL_BEATS, FUEL_BASE} into the 2-deep skid) · [9] ABORT (W1: flush skid + abort in-flight record + auto-drain fifo) · [23:16] tag (RW) |
| 0x0024 | FUEL_BASE | [29:0] base_64B |
| 0x0028 | FUEL_BEATS | [25:0] beats_64B |
| 0x002C | FUEL_STAT | RO: [0] fifo_empty · [1] req_pend (skid nonempty or record in flight) · [2] rdr_busy · [3] ddr_ready · [15:8] last_tag (tag of the last COMPLETED record) |
| 0x0030 | FUEL_ERR | [0] axi_err — RO mirror of the SHELL-domain sticky (any RRESP≠OKAY; clears only on global reset, NOT W1C — disclosed asymmetry) · W1C sticky: [1] desc (GO dropped: skid full or beats_64B=0) · [2] mode_switch (src/ar_owner change REJECTED while busy) · [3] host_push (mailbox XW_PUSH while fuel src) |

Mode bits are quasi-static route-level style: a write that changes them
while busy (skid nonempty | record engaged | reader busy | fifo nonempty)
is REJECTED with old values kept + sticky `mode_switch`. RUN mode blocks
PCIS reads with DECERR at `apex_pcis_slv` (`rd_block`) — BAR4 readback
during RUN can never wedge the PF; writes stay PCIS-owned always. The
AR-mux alternative (ID-based concurrent sharing) was considered and
rejected: load-once flow does not justify R-interleave handling — §9.1 R8.

Reset domains (load-bearing, §4): the 4-phase/afifo/synchronizer state
resets ONLY on the global tile reset; the CSR/skid/sticky state also
resets on the bridge TILE_RST (per-file executor parity). TILE_RST while
fuel is busy loses skid contents by design — quiesce first (FUEL_STAT
idle), the runbook rule.

## 6. NOT in scope (fences — violations are contract bugs)

- **Walker descriptor internals** — IB-WALK. This lane consumes records on
  `fuel_req` (§2.4) and nothing else. No edits under `rtl/seq/`.
- **Datapath blocks** (RoPE/SwiGLU/o-proj/residual, wide-RMSNorm
  elaboration) — IB-LAYER. No edits under `rtl/top/`, `rtl/asu/`.
- **W4-B packed weight format** — B3 lane. Fuel carries bytes (§2.2 hook).
- **Any tile RTL edit** — `rtl/` is read-only to this lane; `apex_pkg.sv`
  frozen.
- **NO AWS spend, NO DCP builds, NO AFI, NO hardware in stages 0–4.** s5
  is owner-gated per §6 of the integration contract (devbox for DCPs,
  boxes stopped when idle, never touch the verifagent/Catapult instances).
- **No HBM, ever** (ECC-scrub AFI-load erratum fenced out with it).
- **Shared files untouched:** `scripts/gen_status.py`,
  `docs/design/LEVEL_C_*.md`, `docs/OPTIMIZATION.md`,
  `golden/`, `rtl/` — this lane's writes stay in
  `scripts/fpga/f2/`, `verif/f2sim/`, and this doc.
- **Honest-model disclosure:** `sh_ddr_beh.sv` is our fabrication of an
  encrypted controller. Mitigations: strict-AXI4 assertions in the model,
  a deliberately boring reader (single ID, INCR, ≤4 KiB, aligned,
  one-outstanding), and s5 silicon as the only arbiter of the real thing.
  No sim result is ever quoted as a hardware claim.

## 7. Stage-0 questions — RESOLVED by §9.1 (962acca)

- **Q1 → R7:** executor-postlude `FUEL_ERR==0` read per job, OUTSIDE the
  counted stream; printed count stays exactly 27,996; log discloses.
- **Q2 → R8:** static LOAD/RUN AR mux per the draft (CSR-switched while
  idle; PCIS readback verify retained).
- **Q3 → R1:** `fuel_req` frozen `{base_64B[29:0], beats_64B[25:0], tag}`
  (64 B-word units — §2.3 rewritten accordingly); one record per
  weight-consuming phase in the D-029 fmt=1 descriptor; bit positions
  frozen in §2.4.
- **New flag to IB-WALK (minor):** R1 left the `tag` width open; this lane
  froze **tag[7:0]** at payload bits [63:56] (§2.4) and mirrors it in
  `FUEL_STAT.last_tag`. If D-029 needs a different width, raise it before
  this lane's s3 RTL lands — after that it is an interface change.
- **Machine rule (all lanes, §9.1):** at most ONE Verilator build at a
  time across lanes — `pgrep -f verilator` before launching, wait if
  busy; Mac model evals outrank EDA.
