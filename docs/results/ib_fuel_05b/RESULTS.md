# IB-FUEL at 0.5B / D=64 — weights resident in card memory (SIM)

**Date:** 2026-08-04 · **Branch:** `comp/prompt-b-c` · **Machine:** local,
pinned Verilator 5.044 (2026-01-01), macOS.
**Tree:** the audit of §1 ran at `7d5cf58`; §2/§3 were then re-verified in full
at `d129c92` (the era-1 lane landed `2259f56` / `2ce9840` / `d129c92`
mid-session, and this lane imports `gemm_job.py`, so every selftest and the
whole projection gate were re-run on the settled tree — same verdicts, same
discriminator delta).
**Scope fence:** simulation only. No AWS instance was launched, no DCP or AFI
was built, `executor=hw` was never run. Every number below is a Verilator
number and is labelled as one.

---

## 0. What this closes, in one paragraph

IB-FUEL was SIM-COMPLETE at s4 (2026-07-22) at **D=128 with the 18 attention
jobs**, and `DDR=1` had never been exercised at any other geometry. This lane
(a) re-ran every existing IB-FUEL gate on today's tree, (b) built the first
`DDR_PRESENT=1` twin at the demo geometry **D=64 / GQA=2 / DM=896 / QSTAGE=14 /
DMODEL=64**, (c) added a TENSOR-granular weight-image tool that packs the real
golden-prepared Qwen2.5-0.5B tensors into the layout the walker's fmt=1 tensor
table addresses, and (d) proved **projection GEMMs whose weight bytes come out
of DDR** — bit-exact against golden, with a one-byte DDR discriminator that
turns the fuel run red and leaves the host-fed run green.

---

## 1. AUDIT BY EXECUTION — what passes TODAY

Every line below is pasted from the run, not retyped from the design doc.

### 1.1 The behavioural DDR model (s1 gate)

`make -C verif/f2sim behsmoke`, both knob sets (default; then
`+ddr_stall=48879 +ddr_lat=3`):

```
DDRBEH RESULT: checks=801 fails=0 -> PASS
DDRBEH RESULT: checks=801 fails=0 -> PASS
```

### 1.2 The DDR=1 build at the 0.5B geometry — NEW, and it just works

```
cd verif/f2sim && make build D=64 DDR=1 OBJ=obj_b64_05b_ddr1 \
  VFLAGS_EXTRA="+define+APEX_CL_DM=896 +define+APEX_CL_GQA=2 \
                +define+APEX_CL_QSTAGE=14 +define+APEX_CL_DMODEL=64"
```

Green, 3.8 s walltime. Verified from the build's own record
(`obj_b64_05b_ddr1/Vcl_apex__verFiles.dat`), not from the invocation:

```
+define+APEX_CL_D=64 +define+APEX_CL_DDR=1 +define+APEX_CL_DM=896
+define+APEX_CL_DMODEL=64 +define+APEX_CL_GQA=2 +define+APEX_CL_QSTAGE=14
```

No Makefile change was needed: `OBJ=` + `VFLAGS_EXTRA=` is the convention
`tile_geom.Image.defines()` already emits, and `DDR` was already a knob.

### 1.3 Host-mode parity at 0.5B — DDR=1 is cycle-invisible

The 7 committed 0.5B jobs (`build/p2_05b_regops_b64_05b`, D=64 attention),
`+tile_div=5`, on the DDR=0 twin and the new DDR=1 twin:

```
DDR=0 (obj_b64_05b)      F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS
DDR=1 (obj_b64_05b_ddr1) F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS
```

`diff` of the two full run logs: **byte-identical** (per-file ops/checks/cycle
lines all match). The s1/s2/s3 "cycle-invisible in host mode" property holds at
the new geometry.

### 1.4 The D=128 regression, on today's tree

18-job host-mode replay at `+tile_div=5` (a ratio the original ladder never
used — it ran 8/7/2):

```
DDR=1 (obj_d128_ddr1)  F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS  (60.4 s)
DDR=0 (obj_d128_ddr0)  F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS  (57.5 s)
```

`diff` of the two logs: **byte-identical**.

### 1.5 The s4 fuel gate, reproduced

`make_ddr_image.py` → `trace_to_fuel.py` → 18-job fuel replay at `+tile_div=5`
with `+ddr_decerr_probe +fuel_audit`:

```
DDRIMG: jobs=18 words=47328 bytes=3028992 -> build/ddr_image
DDRIMG CHECK: jobs=18 fails=0 -> PASS
DDRLOAD RESULT: regions=18 words=47328 rb_fails=0 sha_fails=0 -> PASS
DDRPROBE RESULT: fails=0 -> PASS
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS
```

18 `FUELAUDIT … -> ok` lines outside the counted stream (R7). 6 min 9 s.

### 1.6 Mutants + capgate

```
GATE[control]: exit=0 want=PASS got=PASS  OK
GATE[M1]:      exit=1 want=FAIL got=FAIL  OK
GATE[M2]:      exit=1 want=FAIL got=FAIL  OK
GATE[M3]:      exit=1 want=FAIL got=FAIL  OK
MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS          (49.8 s)

CAPGATE: PASS (job=job_s019_L19_h03 caps=505 values_matched=505/505
               tile_div=5 executor=sim:obj_d128_ddr0)
```

### 1.7 The fuel path at 0.5B/D=64 — the first time ever

The whole existing pipeline pointed at the 0.5B regops instead of the 7B ones
(`--regops build/p2_05b_regops_b64_05b`), run on `obj_b64_05b_ddr1`:

```
DDRIMG: jobs=7 words=744 bytes=47616
DDRIMG CHECK: jobs=7 fails=0 -> PASS
DDRLOAD RESULT: regions=7 words=744 rb_fails=0 sha_fails=0 -> PASS
DDRPROBE RESULT: fails=0 -> PASS
F2SIM RESULT: files=7 checks=1052 fails=0 -> PASS      (3.75 s, 7/7 FUELAUDIT ok)
```

**Every existing IB-FUEL gate is green at D=64/GQA=2/DM=896 with no RTL change
and no tooling change.** Nothing was config-stale in the fuel path itself.

---

## 2. THE WEIGHT IMAGE — `scripts/fpga/f2/make_weight_image.py` (new)

`make_ddr_image.py` is JOB-granular: it scrapes the xw beats a compiled regops
file would have pushed. That is useless for a card that HOLDS its weights —
the regions are keyed by job, they exist only for already-compiled jobs, and
(§4.2 below) a job's stream depends on its activation.

The new tool is TENSOR-granular and targets the layout the walker's fmt=1
descriptor already addresses:

* per layer, the 10 `TENS_*` slots of `seq_walker_fmt.tensor_shapes`;
* a weight tensor's payload is the `seq_walker_fmt.jobs()` **n-major /
  k-minor** concatenation of `gen_l3_vectors.wgt_beats_ws` blocks — the order
  that function's own docstring calls "the pre-swizzled DDR image" order;
* each tensor is its own 4 KiB-aligned region (IB_FUEL §2.3 RECOMMENDED, and
  required by the in-sim loader), so a fuel record can still address any 64 B
  sub-block inside it;
* manifest = per tensor per layer `{base_64B, beats_64B, bytes, shape, tag,
  sha256, gated, max_k_chunk, stage_rows_at_d, stageable_at_d}`.

It invents no quantization: every weight byte is a byte of
`L<nn>_{Wq,Wk,Wv,Wo,Wg,Wu,Wd}.npy` as golden prepared it. The vectorised block
encoder is cross-checked against `gen_l3_vectors.wgt_beats_ws` on sampled
blocks of every tensor, so the frozen emitter stays the definition.

Offline selftest: **11/11 PASS** (encoder ≡ `wgt_beats_ws`; the byte identity
`file[(8p+c)*8+r] == W[8p+r][c]`; payload = K·N and equals the `jobs()`
concatenation; `N % 8 != 0` refused; full build/check/plan round-trip; a single
flipped byte turns `check()` RED).

The real image, layers 0 and 1 of Qwen2.5-0.5B:

```
WEIGHTIMG: model=mlx-community/Qwen2.5-0.5B-Instruct-4bit layers=[0, 1]
           tensors=20 bytes=29851648 (28.5 MiB) -> build/ddr_weights_05b
image sha256 c5ce6e5d4396c2240b4b190db0717750247c32d2cf66b37df4505a43aaa6b347
WEIGHTIMG CHECK: tensors=20 fails=0 -> PASS
```

466,288 DDR words. Built in 0.54 s.

Loaded into the behavioural DDR over the PCIS decode and SHA-verified by the
executor's own C++ readback path:

```
DDRLOAD RESULT: regions=20 words=466288 rb_fails=0 sha_fails=0 -> PASS   (7.98 s)
```

### 2.1 The card-side loader — `scripts/fpga/f2/f2_ddr_load.py` (new)

**Investigated, stated plainly.** The write path is not new hardware and not a
new kit API:

| piece | where it already exists |
|---|---|
| DDR at PCIS byte 0x0, 64 GiB, out-of-range → DECERR | `apex_pcis_slv.sv:108` (this CL, landed s2) |
| BAR4 = 64-bit prefetchable, 128 GiB, mapped onto PCIS | kit `AWS_Shell_Interface_Specification.md` |
| `fpga_pci_attach(slot, pf, APP_PF_BAR4, BURST_CAPABLE)` | kit `sdk/userspace/include/fpga_pci.h:74-88` |
| `fpga_pci_write_burst` / `fpga_pci_peek64` | kit `fpga_pci.h:141`, bindings `fpga_pci_wrapper.pyx:37,61` |

**What does not exist is any USE of it in this repo.** `f2_host_run.py:112`
attaches `pci_attach(slot, 0, 0, 0)` — AppPF BAR0, the OCL register window —
and drives peek/poke only. Nothing here had ever attached BAR4. So the card
loader is a NEW HOST SCRIPT over EXISTING kit calls, not a new path through the
shell. That script is `f2_ddr_load.py`; its `--load`/`--verify` arms have never
touched silicon and say so in their own header.

Offline: **7/7 selftest PASS** (plan arithmetic at 64 B / 4 KiB / 64 KiB burst
sizes; region-crossing, duplicate-burst, past-the-image and non-word-aligned
plans all REFUSED). Plan for the real image:

```
IMAGE  build/ddr_weights_05b  29851648 B (28.5 MiB), 20 tensors, layers [0, 1]
PLAN   468 bursts of <= 65536 B over BAR4 (AppPF BAR4 -> PCIS 0x0 -> sh_ddr)
DDRPLAN: bursts=468 bytes=29842432
```

---

## 3. THE SIM PROOF — `scripts/fpga/f2/fuel_proj_05b.py` (new)

Eight projection GEMMs, two real layers, weights from DDR. `run` verdict:

```
FUEL-FED PROJECTION GEMMs — Qwen2.5-0.5B WEIGHTS RESIDENT IN CARD MEMORY
  image            : build/ddr_weights_05b  29851648 B (28.5 MiB), layers [0, 1],
                     20 tensors (14 weight + 6 aux)
  image sha256     : c5ce6e5d4396c2240b4b190db0717750247c32d2cf66b37df4505a43aaa6b347
  twin             : verif/f2sim/obj_b64_05b_ddr1/f2sim
                     (DDR_PRESENT=1, D=64 GQA=2 DM=896 QSTAGE=14 DMODEL=64)
  prompt           : ids [785, 6722, 315, 9625, 374] (last prefill step, T=5)
  activation       : golden quant_rows_i8 on 896/64 = 14 C-1 frames
                     (the CFG_DM=64 feeder framing) — WEIGHTS UNTOUCHED

  job                        k   n     DDR base  words  tag   @tensor+
  L00_Wq_n0_k0             896   8 0x         0    112    0          0
  L00_Wq_n1_k0             896   8 0x      1c00    112    0       7168
  L00_Wg_n0_k0             896   8 0x    1c0000    112    4          0
  L00_Wg_n1_k0             896   8 0x    1c1c00    112    4       7168
  L01_Wq_n0_k0             896   8 0x    e3c000    112    0          0
  L01_Wq_n1_k0             896   8 0x    e3dc00    112    0       7168
  L01_Wg_n0_k0             896   8 0x    ffc000    112    4          0
  L01_Wg_n1_k0             896   8 0x    ffdc00    112    4       7168
  weight bytes per job: 7168 — every one verified byte-identical to the
  resident tensor BEFORE the run

  RUNS (golden = compute.gemm_i8_ksplit on the same INT8 operands, after)
    control  HOST-fed (BAR0 WB)        rc=0 caps=72 bit-exact 8/8  fuel_audit ok=0 fail=0   0.7s
    CLAIM    FUEL-fed (from DDR)       rc=0 caps=72 bit-exact 8/8  fuel_audit ok=8 fail=0  16.5s
    control  HOST-fed + mutated DDR    rc=0 caps=72 bit-exact 8/8  fuel_audit ok=8 fail=0  20.3s
    RED ARM  FUEL-fed + mutated DDR    rc=0 caps=72 bit-exact 7/8  fuel_audit ok=8 fail=0  16.2s

  DISCRIMINATOR — one byte, named
    DDR byte 0x1c0 (= L00_Wq_n0_k0 + tensor byte 448): 1 -> -2
    it is weight W[k=56][lane 0] and the activation there is -127, so lane 0
    MUST move by 381
    measured lane-0 move on the mutant run: 381
------------------------------------------------------------------------------
  control HOST-fed bit-exact          : PASS
  FUEL-fed bit-exact + FUEL_ERR clean : PASS
  mutated DDR + host-fed stays GREEN  : PASS
  mutated DDR + FUEL-fed goes RED     : PASS   (by exactly the predicted delta)
  FUEL PROJECTION GATE                : PASS
```

Why each arm is there:

* the **host-fed control** proves the operands and the golden are right before
  DDR is implicated at all;
* the **fuel arm** is the claim: zero projection weight pushes in the program,
  every weight beat arriving PCIS → `sh_ddr` → `apex_fuel_reader` →
  `apex_afifo` → xw mux → MXE, and `FUEL_ERR == 0` + fifo-empty + no pending
  record in the R7 postlude on all 8 files;
* the **mutant + host-fed** arm is the half people forget: the same corrupted
  DDR is loaded and the run stays GREEN, so the flip is only load-bearing when
  the tile actually reads DDR;
* the **mutant + fuel-fed** arm goes RED on exactly the one job whose block was
  touched, and the accumulator moves by the byte-level prediction (+381), not
  merely "differs".

The accumulators are `cap` egress — no expectation for any RO lane exists
anywhere in the emitted programs (`_audit_no_result_expectations` is asserted
at emission), and golden is computed after the run by `compute.gemm_i8_ksplit`,
cross-checked against the plain integer product.

Offline selftest of the driver: **10/10 PASS**, including "a byte-mismatched
image is REFUSED" and "the resident bytes decode back to `W[:, 40:48]`" through
`gemm_job._beats_to_block`.

---

## 4. FINDINGS — what was structurally missing (none of it hidden)

### 4.1 The DDR=1 IMAGE build needed two things the sim does not — FIXED, UNVERIFIED

`synth_cl_apex.tcl` plumbed `APEX_CL_DDR` as a `-verilog_define` (so
`DDR_PRESENT=1` reaches `sh_ddr`), but the kit requires two more pieces for a
real DDR core (`hdk/docs/Supported_DDR_Modes.md` "Required RTL / Build Script
Modifications"):

1. a DDR-core macro where `sh_ddr.sv` elaborates — `` `define USE_64GB_DDR_DIMM ``;
2. the matching IP enlisted in the synthesis script — `cl_ddr4.xci`.

`synth_cl_header.tcl:55-60` already reads the encrypted `sh_ddr.sv` itself, so
those were the only two gaps. Both are now applied **only when
`APEX_CL_DDR != 0`**, so every `DDR=0` invocation — every image built to date —
produces a byte-identical `synth_design` argument list and no extra `read_ip`.
Proven with a standalone Tcl harness:

```
DDR=0 -> synth_design -mode out_of_context -top cl_apex -verilog_define XSDB_SLV_DIS
read_ip: IPDIR/cl_ddr4/cl_ddr4.xci
DDR=1 -> synth_design -mode out_of_context -top cl_apex -verilog_define USE_64GB_DDR_DIMM -verilog_define XSDB_SLV_DIS
```

⏸ **Never run through Vivado.** No `DDR_PRESENT=1` DCP has ever been built.
This block is a bring-up hypothesis, not a result.

### 4.2 A projection program's xw stream is only HALF weights

`trace_to_fuel.py` replaces EVERY WB triple with one fuel record — correct for
the 18 attention jobs, wrong for a projection. Measured on this tree at
D=64/K=896: a `gemm_job.build_gemm_job_full` program emits **1792 WB triples,
of which 896 are the ACTIVATION** (each stage row is staged by
`gen_l3_vectors.inject_jobs`, which drives the same external xw port with k=2
WS jobs) and only the last 896 — contiguous, from `s.wbeats` — are weights.

So this lane fuels the TAIL: staging stays on the mailbox, `FUEL_CTRL` switches
to `src=fuel` while the reader is idle and the FIFO empty, the record streams
the weight block. That is also the honest shape of the era-2 transition —
weights resident, activations still streamed — and the `host_push` sticky
covers a stray mailbox push after the switch.

### 4.3 `stage_plan`'s permutation is what blocks resident weights

`gemm_job.stage_plan` makes squant's MODE_QUANT the identity by putting the
activation's GLOBAL amax lane `k*` in lane 0 of every stage row. That
permutation is a function of the ACTIVATION, so the weight stream it induces
(`plan.Wst`) is activation-dependent — a resident, activation-independent
tensor cannot feed it without a per-job re-swizzle.

The way out is already in the design and costs nothing new: this image's own
C-1 feeder is elaborated at `CFG_DM=64`, so a `D_model` row is framed as
`D_model/64` independently-scaled 64-wide frames (layer05b's B-FEED-WIDTH).
Golden's own `quant_rows_i8` on those frames yields per-row amax 127 **by
construction** — MODE_QUANT is the identity with NO permutation, and the weight
stream is the natural tensor order. Measured on the real prompt row: 14/14
frames amax 127.

**The per-64-frame C-1 framing is what makes weights-resident possible on this
tile today.** Anything that keeps the full-row-amax staging keeps the
re-swizzle.

### 4.4 `K_JOB = 2048` is tile-illegal at D=64 — reported, not fixed

`seq_walker_fmt.jobs()` chunks k at `K_JOB = 2048` (= `apex_pkg` `K_MAX`). A
k-chunk needs `k/D` stage rows, and `apex_stage_buf`'s own elaboration guard
caps `R_MAX` at 31 (`cl_apex.sv:260 CL_ROWS_MAX = 31`). At D=64 that caps k at
**1984**, so a 2048 chunk cannot be staged. For 0.5B:

| tensor | (K, N) | k-chunks | stage rows at D=64 | stageable |
|---|---|---|---|---|
| Wq/Wk/Wv/Wo/Wg/Wu | K=896 | [896] | 14 | yes |
| **Wd** | (4864, 896) | [2048, 2048, 768] | 32, 32, 12 | **no** |

A fmt=1 walk of a 0.5B `Wd` would emit two tile-illegal descriptors. Not fixed
here: `jobs()` is IB-WALK's frozen decomposition and changing `K_JOB` also
re-orders `Wd`'s bytes in the image. `make_weight_image.py` now records
`max_k_chunk / stage_rows_at_d / stageable_at_d` per tensor and prints a loud
WARNING, and the manifest records the `K_JOB` the bytes were laid out at, so
the coupling can never bite silently.

### 4.5 g1 / g2 / phase byte order is UNVERIFIED

The aux tensors are carried little-endian 16-bit. No gate in this tree consumes
them from DDR — the walker's gamma/phase fetch path has never been driven end
to end — so their byte order is the tool's choice, not a verified contract.
Marked `"gated": false` in every manifest record. Do not quote them as proven.

---

## 5. WHAT REMAINS FOR THE CARD

### 5.1 The image build

```
export APEX_CL_D=64  APEX_CL_DMODEL=64  APEX_CL_QSTAGE=14 \
       APEX_CL_GQA=2 APEX_CL_DM=896     APEX_CL_DDR=1
bash scripts/fpga/f2/build_dcp.sh          # --aws_clk_gen --clock_recipe_a A2
```

New vs every image built so far: `APEX_CL_DDR=1`, which now also pulls in
`USE_64GB_DDR_DIMM` + `cl_ddr4.xci` (§4.1). Everything else is the b64_05b
recipe that already flew (`agfi-0ecab46b8a8376b21`), except that AGFI was built
before `APEX_CL_D` was env-plumbed — see `25ddb66`.

### 5.2 Bring-up order (each step exists because something bit)

1. AFI load, then `sudo fpga-load-clkgen-recipe -S <slot> -a 2` **and a
   frequency read-back**. An AFI load resets the MMCMs to the default recipe;
   the clkgen instance must be named `AWS_CLK_GEN` or the recipe silently does
   not apply; BAR0 before MMCM lock poisons the OCL bridge by design.
2. `INFO_D` identity read — the D=128-hybrid incident (`25ddb66`) was caught by
   exactly this and by nothing at build time.
3. Poll `FUEL_STAT[3] ddr_ready` (DDR calibration). Do not sleep, poll.
4. `FUEL_CTRL.ar_owner = 0` (LOAD). In RUN mode `apex_pcis_slv` answers PCIS
   READS with DECERR by design — a failed readback there is the mux working,
   not a broken DIMM.
5. `f2_ddr_load.py --load --verify` (whole 64 B words only; the tile may stay
   in reset — the decode is entirely shell-domain).
6. `fuel_proj_05b.py` jobs through `f2_host_run.py`, then the fuel replay of
   the 0.5B set. `FUEL_ERR` read per file.

### 5.3 Risks, ranked

1. **First DDR image ever through ingestion.** DDR4 adds an IP, a pin set and a
   training sequence to a CL that has only ever built without them. Timing
   closure and AFI ingestion are both unproven at `DDR_PRESENT=1`. §4.1's tcl
   block is itself unverified.
2. **`sh_ddr_beh.sv` is our fabrication of an encrypted controller.** Every
   number in §1–§3 is against that model. Mitigations are the ones IB_FUEL §6
   already names (strict-AXI4 assertions, a deliberately boring reader, silicon
   as the only arbiter) — no sim result here is a hardware claim.
3. **Resource headroom.** The DDR4 core is large and the b64_05b CL already
   needed an `m6a.4xlarge` for the GQA-4 route. Expect a longer route and check
   utilisation before assuming the b64 recipe still fits.
4. **Load time on the card is unmeasured.** 28.5 MiB over BAR4 in 468 bursts;
   the sim number (8.0 s) is a Verilator number and predicts nothing.
5. **`Wd` (§4.4)** blocks a full fmt=1 walk of a 0.5B layer at D=64
   independently of anything DDR.

### 5.4 Measured sim numbers, collected

| thing | measured |
|---|---|
| DDR=1 b64 twin build | 3.8 s |
| 0.5B host-mode, 7 jobs, DDR=1 vs DDR=0 | 1052 checks, 0 fails, logs byte-identical |
| 0.5B fuel replay, 7 jobs | 1052 checks, 0 fails, 7/7 FUELAUDIT ok, 3.75 s |
| 18-job D=128 fuel replay (s4) | 27996 checks, 0 fails, 18/18 ok, 6 m 09 s |
| mutants | control PASS + 3/3 RED, 49.8 s |
| weight image, 2 layers, 20 tensors | 29,851,648 B = 466,288 words, built 0.54 s |
| image load into behavioural DDR | 20 regions, 0 rb_fails, 0 sha_fails, 7.98 s |
| fuel-fed projection jobs | 8 jobs, 8/8 bit-exact, 16.5 s |
| discriminator | 1 byte → 1/8 jobs RED, lane-0 delta +381 = prediction |

---

## 6. Reproduce

```
cd verif/f2sim && make behsmoke
make build D=64 DDR=1 OBJ=obj_b64_05b_ddr1 \
  VFLAGS_EXTRA="+define+APEX_CL_DM=896 +define+APEX_CL_GQA=2 \
                +define+APEX_CL_QSTAGE=14 +define+APEX_CL_DMODEL=64"
cd ../.. 
python3 scripts/fpga/f2/make_weight_image.py selftest
python3 scripts/fpga/f2/make_weight_image.py build --layers 0,1 \
        --out build/ddr_weights_05b
python3 scripts/fpga/f2/make_weight_image.py check --out build/ddr_weights_05b
python3 scripts/fpga/f2/f2_ddr_load.py --selftest
python3 scripts/fpga/f2/fuel_proj_05b.py selftest
python3 scripts/fpga/f2/fuel_proj_05b.py run --layers 0,1 --tensors Wq,Wg --blocks 2
```

Gate-hygiene note: the 18-job invocations must use inline `$(ls …)` command
substitution, not an unquoted shell variable — zsh does not word-split
parameter expansions, and an unquoted `$REG` silently hands the executor ONE
argv entry (it then runs 1 file and reports `files=1 … [ABORTED]`). That is the
same class as the `+ddr_image` argv incident recorded in IB_FUEL §1.5; it bit
again here and was caught by the file count, not by an exit code.
