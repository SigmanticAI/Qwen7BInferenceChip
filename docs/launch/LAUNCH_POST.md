# The verification-first LLM tile

*Show your work: an LLM inference tile where every claim traces to a log
line.*

---

Hardware claims are hard to evaluate from the outside. Is a compression
ratio quoted with or without its padding and metadata? Does "runs a 7B
model" mean silicon decoded a token, or a simulation did? Is a performance
number a measurement, a calibrated projection, or a target? Those questions
apply to every project in this space — very much including this one.

We built an LLM-inference tile so those questions always have checkable
answers: **verification first, claims second, and every number labeled with
what it actually is.** Today
we're publishing the whole thing — RTL, the golden model that arbitrates it,
the testbenches, the eval harness, an FPGA bitstream, and a performance model
that prints its own assumptions.

## What it is

**tinyNPU** (working name APEX) is a verified LLM-inference accelerator
tile: RMSNorm → Q/K/V projections → **per-channel INT4 KV-cache
compression** → attention (scores → fixed-point online softmax → P·V̂) →
output projection, extended to a full decoder layer (RoPE + SiLU/SwiGLU +
residual). It is one tile, not a chip — no DRAM controller, no PCIe, out of
scope by charter.

The precise sentence we stand behind: **verified decoder-layer and
KV-compression RTL of an architecture sized for 7B-class models
(head_dim=128 verified), demonstrated on a real 7B model, with a calibrated
projected performance model.**

Note what that sentence does *not* say. It does not say "we built a 7B
chip" — a tile this size decoding 7B today would be host-sequencing hundreds
of thousands of jobs per token, and the performance model quantifies exactly
how mandatory the missing pieces are (more below). Precision about scope is
the whole point of this post.

## Verification first

Every shipped RTL block is bit-exact against an executable golden arbiter —
a NumPy fixed-point model that is the single source of truth. Never the
other way around: when RTL and golden disagree, the RTL is wrong until
proven otherwise.

- **8,748,634 KV-codec arithmetic checks, 0 fails** (exhaustive golden
  sweeps of the synthesizable fp-arith, plus seeded-error mutation gates)
- **135,241 attention-tile checks** — full attention replays through the
  real tile against golden at head dims 64 and 128 (KVQ8, KVQ4+, and
  importance-driven mixed-tier cases; the remaining tier × head-dim corners
  are covered at the pairwise integration layer — the honest-limitations
  register in [`TRACEABILITY.md`](../../TRACEABILITY.md) has the exact map;
  total parsed from the committed suite log)
- **Exhaustive SiLU: 65,536/65,536** input patterns bit-exact
- **Mutation-tested testbenches** — we prove the checkers are alive by
  seeding bugs and requiring the suite to catch them
- **A machine-generated evidence page** ([`STATUS.md`](../../STATUS.md)),
  built by a script that parses the actual suite logs and re-runs the golden
  gate live, under an anti-fabrication rule: no count is published that
  wasn't parsed from a log

The point of this discipline isn't ceremony. The regression suite found real
bugs — an allocator race, a generator index conflation, a scale-set
exhaustion corner — before any of them could become a demo-day surprise.

## Honest compression accounting

KV-cache compression figures in this method family are usually quoted in
bits/value — and converting a bits/value figure into a realized compression
ratio requires counting every storage overhead, ours included. We publish
the number that matters — **what the RTL actually writes to storage, with
every overhead counted** (record tag, outlier lanes, padding, scale bank):

| accounting | value |
|---|---|
| Whole-KV, as stored by the shipped RTL records (D=64 / D=128) | **3.16× / 3.51×** |
| Zero-overhead ceiling of this method family (amortized codec math) | 3.66–3.88× |

That ceiling row applies to ours and to every "~4 bits/value" claim in this
method family: with INT4 payloads plus scale metadata, a compression ratio
above ~3.9× is not achievable by any layout, and stored ratios sit below the
ceiling once real record overheads are counted. Both accountings are pinned
and asserted on every run of `make -C golden test`
([`golden/tests/test_effective_bits.py`](../../golden/tests/test_effective_bits.py)),
and the stored records are bit-exact in RTL.

## Accuracy, measured through the verified codec

Model-accuracy numbers here are not simulated estimates — every eval routes
each layer's post-RoPE K/V through a vectorized twin of the golden codec,
re-certified bit-exact against the arbiter before every run (2,254,080
checks / 0 fails). The same golden model the RTL is verified against is the
one scoring HellaSwag.

Every number below is from the **full 10,042-document HellaSwag validation
set** — we ran the n=1000 pilot first, then re-ran everything at full size
before publishing, and the full-set data *changed some of our own wording*
(details in the results docs; that's the discipline working).

**Qwen2-0.5B and Qwen2-1.5B** (FP16 weights): *near-lossless — worst-case
≈1.9 points, typical ≈1 point* (deltas −0.008…−0.019, unpaired ±0.007). At
full-set power some of those costs are real, so we retired our own pilot
phrase "statistically indistinguishable." What stays true: tier-vs-tier
orderings are still noise (KVQ8 measures below the 4-bit tiers on 1.5B —
physically implausible), so we read no tier as beating another, ours or
anyone's. ([full results + disclosures](../results/s4_head2head/RESULTS.md))

**Qwen2.5-7B** (4-bit MLX weights — the 18 GB eval machine can't hold 7B
FP16; the baseline runs the same 4-bit weights through an identity hook on
the identical code path, so the deltas isolate the codec — full disclosure
in the results doc). Paired per-document statistics, the sharp instrument:

- **KVQ8: no detectable effect at full-set power** — −0.0005 ± 0.0009,
  z = −0.57; **9,965 of 10,042 documents score identically to baseline.**
- **KVQ4: a small, definitively real ≈1.2-point degradation** — −0.0117 ±
  0.0020, z = −5.9. (Our n=1000 pilot estimated −0.017; the full set says
  1.2 points. It is exactly why the KVQ4+ outlier tier exists — its D=128
  loadable mask is scheduled work, and until it lands we don't recommend
  plain KVQ4 for quality-critical 7B use.)
- To our knowledge this is the **first public 7B-model accuracy measurement
  in this ChannelQuant-family method line** (prior public figures in this
  line cover 0.5B/1.5B models; broader KV-quantization methods such as KIVI
  and KVQuant have published 7B results).
  ([full results + disclosures](../results/s5_eval7b/RESULTS.md))

## Placed and routed on a real FPGA

The KV-compression engine is **placed and routed on a real FPGA** — a
Lattice ECP5-85F, with the bitstream committed in-repo
([`docs/results/kvq_engine_ecp5-85f.bit`](../results/kvq_engine_ecp5-85f.bit))
alongside the P&R report. Not an emulator, not a synthesis estimate: a
routed design you can rebuild from the committed flow, at the **full
shipping configuration** (the committed bitstream and P&R report are both
generated from that exact route). One disclosure, because it matters: no
ECP5 board bring-up has happened yet — routed ≠ running on a bench.

The clock, stated plainly: the shipping configuration routes at **34.11
MHz** on this part with the open flow — up 3.7× from the 9.19 MHz
un-pipelined baseline that motivated the work (the fp16 divider is now
II=1-pipelined, and the storage buffers now infer block RAM instead of
overflowing the chip's flip-flops). 34 MHz on a small FPGA is not a
performance claim and we don't make one with it; it's the number that makes
the artifact honest. The committed bitstream and P&R report are regenerated
from exactly this route, so the linked evidence matches the stated number.
([full write-up incl. the routing-seed war story](../results/s10_fmax/RESULT.md))

And on a second FPGA — a different part, a different toolchain, stated
separately on purpose: the **full tile builds clean for AWS F2** (Vivado,
Virtex UltraScale+ VU47P). Zero-error synthesis and place-and-route in under
16 minutes, post-route timing met at the shell's fixed 250 MHz clock, and
the image **loads on a real F2 instance and answers its verified CSRs from
real hardware** — the same registers the test suites exercise in
simulation, reporting the same build truth over PCIe. Register first light only: no
attention job has run on F2 hardware yet, and we say so. Total cloud cost of
that entire bring-up arc: about $12.
([session log + timing report](../results/f2_firstlight/RESULT.md))

## A real 7B model through the pipeline

`run_tinynpu.py --prompt` streams greedy Qwen2.5-7B tokens through the
golden fixed-point pipeline — RMSNorm to logits, 28 layers per step, with
KV through the verified codec — and traces sampled hardware-shaped jobs for
replay.

The committed artifact run
([RESULT.md](../results/s8_7b_token/RESULT.md)): **150 greedy tokens**,
session length 161 — crossing the tile's 128-token chunk boundary on a real
workload — with `TRACE VERIFY: 18/18 jobs replay bit-exact`, including both
chunk-crossing heads traced as one job per chunk. The trace's 2,940
real-model fp16 K/V rows then replayed through the actual KVQ cores RTL
under Verilator: `checks=382200 fails=0`. The chain — real 7B model → golden
arbiter → traced hardware-shaped jobs → RTL bit-exact — is closed end to
end, with the tier scope disclosed (KVQ8 record path; grouped-key tiers
carry their own synthetic coverage).

## Projected performance — and what "projected" means

No APEX silicon exists. Every performance number in this section is the
output of an analytic model
([`docs/results/perf_model/PERF_MODEL.md`](../results/perf_model/PERF_MODEL.md))
that is calibrated to measured simulation and place-and-route anchors, prints
its own assumptions and citations, and asserts its calibration on every run
of `python3 perf/apex_perf_model.py --check`. The model is public; check its
math.

The projection, plainly labeled:

> **7B at reading speed on ~3 watts — with 32k–64k context.** Projected
> 12–14 tok/s short-context decode, TTFT(1k) ≈ 3.4 s (@0.8 GHz reference
> config), ~0.16–0.28 J/token —
> vs a desktop discrete GPU's ~1.4–3.7 J/token at the same single-stream
> job: **5–10× less energy per token** (single-stream, single-user,
> wall/board power, stated quantization on both sides, GPUs at stock
> settings — the model shows its comparator table and citations).

What the KV compression buys, projected: the ≥10 tok/s region stretches from
~19k context (FP16 KV) to **~67k context** with KVQ4 as stored (~74k at the
codec ceiling). At 128k, decode is 8.2 vs 4.3 tok/s — about 2× faster. The
headline isn't peak speed; it's **reading speed held flat as context
grows**, because the KV stream stops dominating the memory traffic.

And what the projection does **not** support — stated before anyone asks:

- **Desktop GPUs are much faster.** A 4090-class card does ~8–14× our
  projected batch-1 tok/s. The projected win is energy per token, never
  speed.
- **We are not the first to sub-0.5 J/token at the edge** — existing edge
  accelerators and Apple silicon already sit at ~0.3–0.5 J/token
  (single-stream, cited in the model's comparator table). The cell we're
  aiming at is energy per token *held flat to long context* via in-datapath
  KV compression, which none of the cited devices document.
- **Batched serving is a different job.** Amortized J/token on a batched GPU
  server is far lower; comparing our single-stream number against it would
  be a strawman in our favor, so we don't.
- **The hedge is honest:** a power-capped GPU or an optimized serving stack
  compresses the gap toward ~2–6×. We quote 5–10× against stock desktop
  settings, model shown.
- **Load-bearing pieces of the projection are unbuilt**, and the model names
  them: the native-W4 weight path (INT8 weights project below the 10 tok/s
  floor), the hardware layer-walker (today's host-sequenced contract is
  ~486k jobs/token ≈ 0.27 tok/s no matter how fast the memory is), and a
  ×64 LPDDR5X memory system (a single ×32 channel fails decode outright).
  The paper-architecture spec
  ([`docs/spec/APEX7B_SPEC.md`](../spec/APEX7B_SPEC.md)) carries these as
  floor-critical, each provenance-tagged.

## What we don't claim

A short register, so nobody has to reverse-engineer our restraint:

- Not silicon-proven, not tapeout-ready. One tile, no memory system.
- Not "runs a 7B model" as a chip claim — 150 real 7B tokens have been
  demonstrated through the golden pipeline, the architecture is sized for
  7B-class models, and the sampled jobs + real-model KV rows replay
  bit-exact in RTL simulation (18/18 jobs; 382,200 checks / 0 fails). The
  distance between "demonstrated on a real 7B model" and "a 7B chip" is
  exactly what the perf model measures.
- Not the first hardware-native KV compression — prior art exists (Titanus,
  GLSVLSI'25; Kelle, MICRO'25). The claim we do make: to our knowledge, the
  only *open, bit-exact-verified* RTL implementation of in-datapath KV
  compression with a real FPGA bitstream and padding-inclusive accounting.
- Not faster than GPUs. Not lower-energy than every edge device at short
  context. Not lossless at KVQ4 on 7B (see the paired stats above).

## Reproduce it

Everything above re-runs from a clone: `make -C golden test` (golden arbiter
+ pinned compression accounting), the Verilator suites behind every count in
[`STATUS.md`](../../STATUS.md), `python3 perf/apex_perf_model.py --check`
(perf-model calibration), and the eval matrices via the committed scripts
beside each results file. If a number in this post isn't reproducible from
the repo, file an issue — that's a bug in the post.

---

*How this post was gated before publication: [`CAMPAIGN.md`](CAMPAIGN.md) —
the claim-discipline rules and launch checklist are public on purpose.*
