<p align="center">
  <img src="docs/assets/apex_banner.svg" width="520"
       alt="APEX — tinyNPU: the attention engine that remembers">
</p>

**A fully open, verification-first LLM-inference tile.** One transformer
decoder layer in real RTL — attention, KV-cache compression, softmax,
RMSNorm, RoPE, SwiGLU, residual — every block bit-exact against an
executable golden model, running real Qwen models through the verified
pipeline, and brought up on real FPGA hardware. Pre-silicon, and every
number in this repo is labeled as what it is: **measured or projected**.

> **TL;DR for the impatient:** GPUs are built to *compute*. This tile is
> built to *remember*. It does the attention math in hardware **and**
> compresses the conversation's memory (the KV cache) in the same datapath,
> so reading speed is designed to stay flat as the context grows. Verified
> numbers live one click away: [`STATUS.md`](STATUS.md) ·
> [results index](docs/results/) · [master table](docs/design/MASTER_TABLE.md).

---

## Table of contents

1. [Why this exists](#1-why-this-exists)
2. [The architecture](#2-the-architecture)
   - [One decode step through the tile](#one-decode-step-through-the-tile)
   - [Block-by-block tour](#block-by-block-tour)
   - [The layer walker: from job server to sequencer](#the-layer-walker-from-job-server-to-sequencer)
   - [Feeding the beast: weight streaming](#feeding-the-beast-weight-streaming)
3. [The KV-compression engine](#3-the-kv-compression-engine)
4. [How we know it's right](#4-how-we-know-its-right)
5. [Real models through the pipeline](#5-real-models-through-the-pipeline)
6. [On real hardware](#6-on-real-hardware)
7. [Performance: what's measured, what's projected](#7-performance-whats-measured-whats-projected)
8. [Repository layout](#8-repository-layout)
9. [Reproduce everything](#9-reproduce-everything)
10. [Method, prior art, and provenance](#10-method-prior-art-and-provenance)
11. [License](#11-license)
12. [Roadmap](#12-roadmap)

---

## 1. Why this exists

Large-model inference has two costs that dominate everything else at the
edge: **moving weights** and **remembering context**. The first is a
bandwidth problem. The second — the KV cache — is the quietly growing one:
every token you've read must be stored and re-read on every new token, so a
long conversation turns an inference engine into a memory-traffic engine.

Most accelerators treat the KV cache as a software problem: quantize it on
the CPU/GPU, store it, hope. APEX's bet is architectural: **put the KV
codec inside the datapath**, so keys and values are compressed the moment
they are produced and decompressed the moment they are consumed — with an
importance unit that watches which parts of the context actually matter and
spends bits where they count.

The second bet is methodological: **nothing ships without a bit-exact
golden model.** Every RTL block in this tree is verified against an
executable NumPy reference — not "close", not "within tolerance":
bit-identical, with mutation-tested testbenches and machine-generated
evidence under an anti-fabrication rule. See
[§4](#4-how-we-know-its-right).

The scope is deliberately honest: this is **one tile** — no DRAM
controller, no PCIe, no NoC (out of scope by charter). The architecture is
demonstrated end-to-end on real **Qwen2.5-0.5B** on FPGA hardware. The
architecture is sized for 7B-class models (head_dim = 128 exists in RTL,
and Qwen2.5-7B tokens have run through the software-verified golden
pipeline — not through silicon), and the
paper-architecture that wraps the tile into a full chip is specified with
per-number provenance in [`docs/spec/APEX7B_SPEC.md`](docs/spec/APEX7B_SPEC.md).

---

## Built with Sigmantic AI

APEX was developed with the help of **[Sigmantic AI](https://www.sigmanticai.com/)**'s
verification tooling. Sigmantic AI builds autonomous verification agents that
take hardware from natural-language spec to verified RTL — real UVM
testbenches, real simulators, honest sign-off. Underneath is the **Polaris
Engine**: a closed-loop planner that treats verification as scientific
discovery, choosing at every step the one simulation that will reveal the
most about your design.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/polaris-dark.svg">
  <img alt="The Polaris Engine — verification as scientific discovery: a closed loop that runs, each cycle, the one simulation that reveals the most." src="assets/polaris-light.svg" width="100%">
</picture>

## 2. The architecture

![APEX-7B block diagram](docs/spec/apex7b_block_diagram.svg)

### One decode step through the tile

A single new token enters as an activation vector `x`; the tile produces
the layer output `y`. Everything between is hardware:

```
              ┌──────────────────────────── apex_top ─────────────────────────────┐
              │                                                                   │
  x ──► seam ─┼─► RMSNorm ──► MXE: W_Q·x  W_K·x  W_V·x ──► RoPE ──► KVQ compress  │
              │    (ASU)      (systolic GEMM)              (Q,K)        │         │
              │                                                         ▼         │
              │                                     KV cache — INT4 + outlier     │
              │                                     lane, on-tile SRAM            │
              │                                                         │         │
              │              ┌──────────────────────────────────────────┤         │
              │              ▼                                          │         │
              │   MXE: Q·K̂ᵀ ──► online softmax ──► MXE: P·V̂             │         │
              │                    (ASU)              │                 │         │
              │        ▲                              │                 │         │
              │        └────── TIP importance ◄───────┤                 │         │
              │               (adaptive precision)    ▼                 │         │
              │                          MXE: W_O·attn ──► + residual ──┼──► y    │
              │                                                         │         │
              │   FFN: RMSNorm ──► MXE: W_gate/W_up ──► SwiGLU ──► MXE: W_down    │
              │                                                    ──► + residual │
              └───────────────────────────────────────────────────────────────────┘
```

Six ideas make this a tile rather than a pile of blocks:

- **One GEMM engine, many roles.** The MXE (INT8 systolic array) is
  time-multiplexed across all seven matrix jobs of a decoder layer — the
  QKV projections, the two attention products, the output projection, and
  the FFN matrices. The sequencer (§ below) choreographs it.
- **The KV cache lives compressed.** K and V never touch SRAM in fp16.
  They are quantized in-flight by the KVQ engine ([§3](#3-the-kv-compression-engine))
  and dequantized in-flight on the read side of the attention products.
- **Online softmax.** Scores stream through the ASU's numerically-stable
  online softmax — no materialized score matrix, no context-length-sized
  buffer.
- **Importance is a first-class signal.** The TIP unit tracks which cached
  tokens keep mattering and drives the codec's precision tiers per region
  (KVQ8 / KVQ4 / KVQ4+outliers) — the "adaptive" in adaptive precision.
- **Scales travel with data.** Quantization scales for activations and
  composites move through the same seams as the tensors they describe, so
  a job is bit-exact reproducible from its descriptor alone.
- **Everything is a job.** Each hardware operation is a self-describing
  descriptor (matrix shapes, routing, quantization contract). That is what
  makes host-driven and walker-driven execution *the same machine* — and
  what makes every flight replayable in simulation, bit for bit.

### Block-by-block tour

| block | dir | what it does |
|---|---|---|
| **MXE** | [`rtl/mxe`](rtl/mxe) | INT8 systolic GEMM engine — all seven matrix jobs of the layer, time-multiplexed |
| **KVQ** | [`rtl/kvq`](rtl/kvq) | the KV-cache codec: per-channel INT4 keys / per-token INT4 values, fp16 outlier lane, precision tiers — compression *in the datapath* ([§3](#3-the-kv-compression-engine)) |
| **ASU** | [`rtl/asu`](rtl/asu) | the nonlinear unit: online softmax, RMSNorm (narrow + wide), SiLU/SwiGLU — exhaustively verified against golden |
| **RoPE** | [`rtl/rope`](rtl/rope) | rotary position embedding applied to Q and K in-flight |
| **TIP** | [`rtl/tip`](rtl/tip) | token-importance tracking → per-region precision tier decisions |
| **SEQ** | [`rtl/seq`](rtl/seq) | CSR file, job decoder, and the **layer walker** — the on-tile sequencer (§ below) |
| **SEAM** | [`rtl/seam`](rtl/seam) | ingress/egress: activation framing, quantization at the boundary, scale transport |
| **XBR** | [`rtl/xbr`](rtl/xbr) | the routing fabric that lets one engine's egress feed the next engine's ingress without host round-trips |
| **TOP** | [`rtl/top`](rtl/top) | `apex_top` — composition + glue (GQA engine banking, W4 ingest, composite scale caches) |

The deep-dive is [`ARCHITECTURE.md`](ARCHITECTURE.md) — including the
design-decision register and the module provenance history.

### The layer walker: from job server to sequencer

The tile has two execution modes, and the difference is the project's
central bring-up story:

```
  HOST MODE  (proven first)              WALKED MODE  (the destination)
  ──────────────────────────            ───────────────────────────────
  host builds each job                   host kicks ONCE per layer
   │  descriptor, one at a time           │
   ▼                                      ▼
  ┌────┐  job   ┌──────┐                ┌────────────┐ fetches its own
  │host│ ─────► │ tile │                │ seq walker │ descriptors, walks
  └────┘ ◄───── └──────┘                │  (on-tile) │ QKV→attn→OPROJ→
      result, each job                  └─────┬──────┘ residual→FFN→DOWN
      (~ms of host latency              ┌─────▼──────┐
       per job, ~100× tax)              │   engines  │ chained on-tile via
                                        └────────────┘ the XBR fabric
```

In host mode every job pays a host round-trip — correct, measurable, and
roughly two orders of magnitude of throughput left on the table. The
**layer walker** ([`rtl/seq`](rtl/seq), design notes in
[`docs/design/B1_WALKER.md`](docs/design/B1_WALKER.md)) replaces the host
inside a layer: it fetches descriptors, sequences the engines, routes
intermediate tensors through the fabric, and grades composite scales at
every seam — so one kick executes a whole layer.

Walker status is tracked honestly in
[`docs/design/MASTER_TABLE.md`](docs/design/MASTER_TABLE.md) and
[`docs/design/PROMPT_ON_CHIP.md`](docs/design/PROMPT_ON_CHIP.md): the
walked layer is **bit-exact in simulation** (attention, norm, residual,
FFN and DOWN composition), host-mode is **proven on FPGA silicon**, and
walked attention on the FPGA is in active bring-up — a genuinely rare
synthesis-vs-simulation defect was root-caused there (the story is worth
reading: [`docs/results/prompt_on_chip/`](docs/results/prompt_on_chip/)).

### Feeding the beast: weight streaming

A real model's weights don't fit on a tile — they stream. The IB-FUEL lane
([`docs/design/IB_FUEL.md`](docs/design/IB_FUEL.md)) gives the walker a
fuel line from DDR: per-job weight records prefetched into an on-tile FIFO,
so the GEMM engine contracts over streamed weights without stalling, and
per-chunk records let the FFN's big matrices interleave with compute. The
W4 lane ([`docs/design/W4_DATAPATH.md`](docs/design/W4_DATAPATH.md))
halves that traffic again: weights travel as 4-bit groups and are unpacked
to INT8 at the feeder, with group scales riding the same transport as the
data they describe.

## 3. The KV-compression engine

The codec is the architectural heart. Per-channel INT4 keys, per-token
INT4 values, an fp16 outlier lane for the channels that refuse to
quantize, and tiered precision (KVQ8 / KVQ4 / KVQ4+) driven by measured
token importance — the published on-device KV-quantization family (cf.
KIVI, KVQuant), implemented as verified hardware.

What makes this implementation worth reading:

- **Every overhead is counted.** The compression ratio is asserted in the
  golden model *including* tags, outlier lanes, padding, and the scale
  bank — the number that survives is the number the SRAM actually sees.
  Whole-KV ratios and the method-family ceiling analysis:
  [`STATUS.md`](STATUS.md).
- **It's not a sidecar.** Compression happens between RoPE and the cache
  write; decompression happens inside the attention read path. There is no
  fp16 copy anywhere.
- **Accuracy is measured through the verified codec** — the same golden
  model the RTL is bit-exact against scores real models on the full
  HellaSwag validation set, 0.5B/1.5B/7B, with paired per-document
  statistics: [`docs/results/s4_head2head/`](docs/results/s4_head2head/RESULTS.md) ·
  [`docs/results/s5_eval7b/`](docs/results/s5_eval7b/RESULTS.md).
- **It's on real FPGA hardware** at the full shipping configuration
  (Lattice ECP5-85F, open toolchain — bitstream committed in-repo):
  [`docs/results/s10_fmax/`](docs/results/s10_fmax/RESULT.md).

## 4. How we know it's right

The methodology is the product as much as the RTL is:

```
   spec ──► golden model (NumPy, executable) ──► frozen vectors
                 │                                   │
                 │ bit-exact arbiter                 ▼
                 └────────────► RTL ◄──── testbenches + SVA + coverage
                                 │            │
                                 │            └── mutation gates: break the
                                 │                RTL on purpose — the TB
                                 ▼                must catch every mutant
                        composition levels
              L1 block → L2 pipeline → L3 full layer vs golden
                                 │
                                 ▼
                 same jobs replayed on FPGA silicon, bit for bit
```

- **Bit-exact, not approximately right.** Millions of checked values, zero
  tolerated mismatches, across every block and every composition level —
  current counts are machine-generated into [`STATUS.md`](STATUS.md) on
  every full-suite run (the only source of published counts, by rule).
- **Mutation-tested testbenches.** A green suite only counts if it turns
  red when the RTL is deliberately broken — mutants that survive are
  treated as verification bugs.
- **An anti-fabrication rule.** Published numbers are machine-extracted
  from suite logs; the limitations register
  ([`TRACEABILITY.md`](TRACEABILITY.md)) records every known gap, and
  claims link to the test and log line that back them.
- **Sim-to-silicon differential discipline.** Every FPGA flight is a
  register-op program that also runs in the Verilator twin; hardware
  captures are compared bit-for-bit against simulation. This is how a
  synthesis-level hardware defect was cornered and root-caused entirely
  from differential evidence — the toolchain, not the RTL, was lying
  ([`docs/results/prompt_on_chip/`](docs/results/prompt_on_chip/)).

## 5. Real models through the pipeline

Not a toy trace: `run_tinynpu.py --prompt` streams greedy **Qwen2.5-7B**
(through the software-verified golden pipeline — the FPGA-measured model
is **Qwen2.5-0.5B**) —
tokens through the golden fixed-point pipeline — the same executable
arbiter the RTL is verified against — producing coherent text, with
sampled hardware-shaped jobs traced and replayed bit-exact, and the
trace's real-model K/V rows replayed through the KVQ RTL in simulation
with zero mismatches. Full artifact run, scope, and counts:
[`docs/results/s8_7b_token/`](docs/results/s8_7b_token/RESULT.md).

Accuracy through the verified codec (full 10,042-document HellaSwag,
measured, reproducible from a clone): near-lossless at 0.5B/1.5B; at 7B,
KVQ8 shows no detectable effect at full-set power and KVQ4 a small, real,
quantified cost — which is precisely why the outlier tier exists. Numbers,
statistics, and honest caveats:
[`docs/results/s5_eval7b/`](docs/results/s5_eval7b/RESULTS.md).

## 6. On real hardware

Two independent hardware existence proofs, different parts, different
toolchains:

- **Lattice ECP5-85F (open flow):** the KV-compression engine placed and
  routed at the full shipping configuration; bitstream and P&R report
  committed. Routed Fmax disclosed with its full context — including the
  retirement of an earlier reduced-parameter figure:
  [`docs/results/s10_fmax/`](docs/results/s10_fmax/RESULT.md).
- **AWS F2 (Vivado, VU47P):** the complete tile builds clean and flies on
  real cloud FPGA silicon. First light — the image loading and answering
  its verified CSRs — is documented in
  [`docs/results/f2_firstlight/`](docs/results/f2_firstlight/RESULT.md);
  since then the campaign has progressed through host-mode attention
  (bit-exact on silicon), walked norm/residual chains, DDR weight
  streaming, and the walked-attention bring-up with its forensic defect
  hunt — the running log is
  [`docs/design/PROMPT_ON_CHIP.md`](docs/design/PROMPT_ON_CHIP.md) and
  [`docs/results/prompt_on_chip/`](docs/results/prompt_on_chip/).

### Run it yourself

The verified reference image is **`agfi-030a812cd224b409d`** (A2 recipe,
15.625 MHz tile; registered with its build receipt in
[`scripts/fpga/f2/clock_key.py`](scripts/fpga/f2/clock_key.py)). These steps
were executed verbatim against that image before this section was committed —
battery `193 checks, 0 fails`, then live prompts answered with every walked
value graded bit-exact:

```sh
# prove the silicon: boots an f2.6xlarge, loads the image, flies the
# 193-check battery + walked chains, prints the verdict, terminates itself
bash scripts/fpga/f2/run_walked_demo.sh agfi-030a812cd224b409d

# talk to it: an interactive prompt CLI on the walked pipeline
# ("The capital of France is" -> " Paris.", graded end to end)
bash scripts/fpga/f2/run_chat_demo.sh
```

Requirements: AWS CLI credentials with F2 access (`f2.6xlarge`,
us-west-2) and the team's model-weight artifacts (the DDR images derive
from `mlx-community/Qwen2.5-0.5B-Instruct-4bit`; the staging bucket is
configured in the scripts — external users should regenerate the weight
image with `scripts/fpga/f2/make_weight_image.py` against their own
bucket). Cost: about $2 and 30 minutes per full run.

## 7. Performance: what's measured, what's projected

This repo's rule: **a number is either measured on hardware/simulation, or
it is a projection from the calibrated analytic model — and it says
which.** No blended claims.

- **Measured** cycle counts, compression ratios, and check totals live in
  [`STATUS.md`](STATUS.md) and the per-campaign results directories.
- **Projected** (not measured) end-state performance (7B at reading speed on ~3 W with
  32k–64k context held flat by KV compression; 5–10× less energy per token
  than a desktop GPU at the same single-stream job — *never* a speed win
  over GPUs) comes from the analytic model with asserted calibration
  anchors and machine-printed assumptions:
  [`docs/results/perf_model/PERF_MODEL.md`](docs/results/perf_model/PERF_MODEL.md).
  The projection's three load-bearing unbuilt dependencies (native-W4
  weight path, hardware layer walker, wide LPDDR) are exactly the lanes
  under active construction in [§2](#2-the-architecture) — tracked with
  per-number provenance in
  [`docs/spec/FLOOR_TRACEABILITY.md`](docs/spec/FLOOR_TRACEABILITY.md).

## 8. Repository layout

```
rtl/           the chip — apex_pkg (contract) · mxe · kvq · asu · rope ·
               tip · seq (walker+csr) · seam · xbr · misc · top
golden/        bit-exact NumPy reference models (the verification arbiter)
verif/         testbenches, scoreboards, SVA, mutation gates, the
               Verilator F2 twin (verif/f2sim)
eval/          KV-codec model-accuracy harness (bit-exact-gated twin codec)
perf/          analytic performance model (ALL PROJECTED)
scripts/fpga/  the F2 flight stack: builds, AFI registry, DDR loader,
               flight drivers, token-loop harness
docs/spec/     APEX-7B paper architecture (spec · block diagram · floors)
docs/design/   living design notes per lane (walker, fuel, W4, clocks…)
docs/results/  committed evidence, one directory per campaign
reference/     the two verified GEMM designs APEX builds on
ARCHITECTURE.md · STATUS.md · TRACEABILITY.md
```

## 9. Reproduce everything

Requires Verilator 5.x and Python 3.11 + NumPy (Icarus optional as a
cross-check).

```sh
make -C golden test            # golden models vs frozen vectors,
                               # incl. the pinned compression-accounting gate
make -C verif/top/smoke smoke  # full attention tile, end-to-end
make -C verif/top l3           # Layer-3: real tile vs golden, all cases
make -C verif/seq_walker       # the walker suite + mutation gates
python3 perf/apex_perf_model.py --check   # perf-model calibration anchors
```

The model-accuracy matrices re-run from a clone (see the eval READMEs
beside each results file): `bash eval/kv_eval/run_matrix.sh` (0.5B/1.5B)
and `bash eval/kv_eval/run_matrix_7b.sh` (7B).

## 10. Method, prior art, and provenance

The codec method — per-channel INT4 K / per-token INT4 V with an fp16
outlier lane — is the published on-device KV-quantization approach (cf.
KIVI, KVQuant); hardware-native KV compression has prior art (Titanus
GLSVLSI'25, Kelle MICRO'25). **We do not claim method novelty.** APEX's
contribution is the *integrated, verified hardware*: to our knowledge the
only open, bit-exact-verified RTL implementation of in-datapath KV
compression with a real FPGA bitstream and padding-inclusive accounting.

All RTL in this tree is APEX's own. The two blocks that began as vendored
bring-up copies were replaced by documented clean-room rewrites; each
module's provenance header and the decision register in
[`ARCHITECTURE.md`](ARCHITECTURE.md) (§9–§10) keep that history auditable.

### 11. License

Apache License 2.0 — see [`LICENSE`](LICENSE). Use it, study it, build on
it; keep the notices.

## 12. Roadmap

In dependency order, tracked live in
[`docs/design/MASTER_TABLE.md`](docs/design/MASTER_TABLE.md):

1. **Walked attention on FPGA silicon** — the root-caused synthesis defect
   fix through validation, then the full walked layer on the card.
2. **Faster tile clock** — the 62.5 MHz timing-closure campaign.
3. **Sustained weight streaming + W4 end-to-end** — the bandwidth lanes
   that hold the token rate at long context.
4. **Measured tokens/second** — the token-loop harness replaces
   projections with silicon measurements, rung by rung.
5. **Sky130 signoff** — from FPGA-proven to silicon-signoff-ready.

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img alt="APEX — a transformer-inference chip on AWS F2. The chip walks the layers; the host only grades them." src="assets/hero-light.svg" width="100%">
</picture>

