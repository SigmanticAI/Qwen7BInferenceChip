# APEX NPU — Architecture & Contract Specification

**Version:** 0.2 · **Date:** 2026-07-09 (v0.1: 2026-07-07) · **Status:** v0.2 shipped — all decisions PROVEN (§11)
**Verification environment:** Verilator 5.044 (`--binary --timing --assert`) primary — pinned EXACTLY (it is the execution engine; 5.020 miscompiles the tile) · Icarus **≥12.0** secondary — a FLOOR, not an exact pin (relaxed from 13.0, 2026-07-27; the secondary simulator's job is *independence*, so more versions agreeing is stronger evidence, and the suites compare against the golden model rather than stored expectations, so a disagreeing version surfaces as a FAIL, not a silent pass. Measured basis: Icarus 12.0 on Ubuntu 24.04, off the author's machine, bit-exact on three exhaustive suites — SiLU 65,536/65,536, RoPE 12,288/12,288, residual 20,012/20,012 = 97,836 vectors, zero source edits, tool-path overrides only. Limits of that evidence, stated: 12.0 on three suites; 13.0 has never been run off-Mac; no other Icarus version tested anywhere) · Python 3.11 + NumPy golden models · gtkwave

---

## 0. What APEX is

**APEX** (Attention Processing EXecution tile) is a fully verified, locally
simulable LLM-inference **attention tile**: an integrated composition of a
systolic GEMM engine, a hardware KV-cache codec, and fixed-point softmax /
normalization — the full attention datapath, verified end to end. It executes one
transformer decode step for one attention layer entirely on-tile:

```
x → [RMSNorm] → [MXE: Q/K/V proj] → [KVQ: per-channel INT4 codec compress K,V]
  → [MXE-OS: Q·K̂ᵀ scores] → [TIP: importance/precision decision]
  → [ASU: online softmax] → [MXE-OS: P·V̂] → [MXE: output proj] → y
```

The thesis: **integration is the hard part** (prior efforts get leaf blocks passing but never a working top-level). APEX's deliverable is the
end-to-end verified tile plus the verification system that proves it.

Scope excludes DRAM controllers, PCIe, and multi-tile NoC — everything must be
verifiable on this machine. Area/timing are secondary to architectural
correctness and verification depth (we target clean synthesis-style RTL, but
sign-off is simulation + assertion + coverage based).

## 1. Block plan

| Block | Origin | What it is |
|---|---|---|
| `apex_pkg` | new | Single config/types package mirrored into TB + golden models (one shared config source) |
| **MXE** — Matrix eXecution Engine | rewrite; lineage run1/run2 | True weight-stationary **systolic** N×N (default 8×8) INT8×INT8 array with PE-to-PE pipelining (NOT the broadcast MAC wall of run1/run2 — critic item 1.15), INT32 accumulators, output-stationary mode for Q·Kᵀ and P·V, double-buffered weight SRAM with wide load port (fixes run2's 7× load-bound profile), descriptor-driven with conservative legality reject (run1 lineage) |
| **KVQ** — KV Quantization subsystem | new | Per-channel INT4 KV codec (CQ-8/CQ-4/CQ-4+), published KV-quant method (cf. KIVI, KVQuant); from-contract RTL: (a) AXI-compliant streaming read path with skid + `tready`, (b) partial-group flush CSR trigger, (c) byte-aligned banked SRAM record layout (§4), (d) bit-exact vs the golden model |
| **ASU** — Attention Softmax Unit | new | Fixed-point **online softmax** (running max + rescaled running sum, FlashAttention-style) over INT32 score stream; exp via 64-entry LUT + linear interpolation; emits Q1.15 probabilities. Plus **RMSNorm** lane built on our `rsqrt` unit (re-verified under Verilator first) |
| **TIP** — Token Importance & Precision | new | importance ratio test (`max·N > THRESHOLD·sum`, shift-add only) re-parameterized for SCORE_WIDTH=32 with per-layer `THRESHOLD_REG`; drives per-block KVQ tier select (CQ-4 vs CQ-4+/CQ-8) + saturating per-block importance accumulators (TIP-lite) |
| **XBR** — stream fabric | new; run1 `stream_skid` lineage | 2-deep skid buffers on **every** external and inter-block stream; single valid/ready discipline (§5) |
| **SEQ** — sequencer-lite | new | Descriptor queue walker FSM: fetch descriptor → dispatch block ops → barrier → next. (Not a RISC; LSU-class programmability is out of scope v0.1) |
| **CSR** | new; KVQ ISA style | One AXI-Lite-style 32-bit register window for the whole tile (§7) |

## 2. Numerics contract (frozen — the arbiter for every golden model and RTL block)

Resolves critic items 1.1–1.3. **Where prose and vectors disagree, vendored
golden vectors win** (KVQ contract discipline).

- **C-1 INT4 quantization** = per-channel INT4 codec contract verbatim: symmetric signed,
  `s = max(amax/qmax, EPS)`, `EPS = 2⁻¹⁴`, clamp **[-8, 7]** (−8 legal),
  **round-half-to-even**, FP16 scales, fp32 dequant bus. Alternative `[-7,7]`-clamp kernels are NOT contract-conformant; if used as
  oracles they get a documented translation shim, never silent mixing.
- **C-2 Rounding, one rule per stage, RNE everywhere it matters:**
  - Quant/dequant paths (KVQ): round-half-to-even (contract §1).
  - MXE requant epilogue INT32→INT8: **round-half-to-even** on
    `(acc × scale) >> shift`, then saturate. This *departs from* run1
    (floor-shift, open question UQ-001) and run2 (round-half-up): uniformity
    beats lineage. Golden model and RTL implement the same primitive;
    equivalence proven by exhaustive sweep over the requant operand domain.
  - ASU fixed-point: truncation only inside LUT interpolation (documented),
    final probability RNE to Q1.15.
- **C-3 Accumulators are INT32.** An earlier INT24-accumulator derivation
  (`11 + log2(K)`) fails at K=8960 (24.1 bits — an unclosed edge case). APEX has no silicon area pressure; INT32 removes the risk
  class entirely. SVA: no accumulator overflow assertion at INT32 with
  K ≤ 2¹⁶ and INT8×INT8 operands (21 + 16 = 37 > 32 — so K is **capped at
  K ≤ 2048** per descriptor, giving 21 + 11 = 32 exactly; the legality
  checker rejects K > 2048, and the golden model checks the true bound).
- **C-4 Dequant-on-feeder**: the MXE array sees exactly
  one dtype (INT8). W4 weights, when adopted (v1.1), unpack+dequant at the
  weight feeder, never inside PEs.
- **C-5 Scores:** Q·Kᵀ accumulates INT32; ASU consumes INT32 scores directly
  (no pre-narrowing); TIP ratio test runs on |INT32| magnitudes.

## 3. MXE microarchitecture decisions

- True systolic: weights pre-loaded stationary into PE regs (double-buffered:
  bank A computes while bank B loads — weight load port is `8×8=64` bytes/beat
  wide internally, fed by a 64-bit external stream); activations skew in
  west→east; partials flow north→south into INT32 column accumulators.
- Two dataflow modes (CSR/descriptor bit): **WS** (weight
  stationary — projections/FFN) and **OS** (output stationary — Q·Kᵀ, P·V̂).
- Descriptor (128-bit, run1 lineage): opcode, M/K/N tile dims, scale/shift for
  epilogue, src/dst buffer IDs, flags (mode, requant-enable, accumulate).
  Conservative legality reject: bad opcode, K > 2048, dims=0, buffer overrun
  → NO state change, `desc_error` pulse + sticky CSR bit (run1's UQ-003
  semantics, this time with the busy/done contract of §5 so the reject test
  cannot be polluted by a previous job).

## 4. KVQ record layout (resolves critic 1.4; D-026 A2 KV-REC-DEDUP)

Upstream unified record at D=64 is 1281 bits (not byte-aligned) and duplicates
the per-channel group scales into EVERY token's record. APEX v0.2 initially
inherited that composition (byte-aligned 1344b/2624b rows — ~21 b/v keys, an
EXPANSION vs fp16; the honesty gate below is what exposed it). **D-026
(2026-07-14) supersedes the key-record COMPOSITION** — the 64b-pad rule stays
normative for records and bank rows alike. Three structures:

```
KEY record (CQ-4/CQ-4+), one row per token:
[ tag:8b = {ssid[6:0],1} ][ outlier lanes: k×16b ][ int4 codes: D×4b ][ pad64 ]
  lane j = raw fp16 of the j-th mask-set channel (ascending channel order);
  keep-channel code = quant nibble, outlier-channel code = sentinel 4'd1
  D=64  k=2 → 8+32+256 = 296b → 320b row    D=64  k=5 → 344b → 384b row
  D=128 k=2 → 8+32+512 = 552b → 576b row    D=128 k=0 → 520b → 576b row

VALUE record (unchanged): [ tag:8b=00 ][ fp16 scale ][ D×BPV codes ][ pad64 ]
SRAM row width = pad64(max(key_raw, val_raw)) on keyed engines.

SCALE BANK (persistent, per engine): SCALE_SETS(=4) rows × D×16b — one row per
COMMITTED group (keep-channel scales; outlier lanes forced 0x0000). ssid =
commit sequence mod SCALE_SETS, stamped in the tag; reuse of a live set is a
host-contract fault (sticky SB_OVWR, IRQ_STATUS[1]). Bank rows survive D-020
soft reset exactly like the record SRAM. Golden arbiter of all three images:
cq_codec.pack_key_records / pack_value_records / sram_row_bits.
```
*(v0.1 of this section had 1312/2608 — not 64b multiples, contradicting the
rule one line above; caught by the KVQ implementation agent. Corrected. The
v0.2 scale-duplicating composition is RETIRED 2026-07-14 by D-026.)*

Effective bits/value after padding is recomputed and asserted in the golden
model — `golden/tests/test_effective_bits.py`, part of `make -C golden test`
(upstream's 4.13–4.38 figure ignores padding AND is bits/value, not a ratio;
ours pins all three accountings: stored-record, codec-ceiling, and the CSR
advertisement. Stored whole-KV incl. bank amortization: **3.16× at D=64 /
3.51× at D=128, shipped k≤2** (k=5: 2.64×); the codec ceiling is 3.66–3.88×;
a ratio >4× is impossible for this method family).
Production config: **D=64 and D=128 both supported; G=128; tiers CQ-4 default,
CQ-4+ optional (k=2), CQ-8 available.** The upstream gap "G=128 keys never ran
through the top-level FSM, CQ-8 never through the top" (critic 1.5) becomes an
APEX test obligation, not an inherited assumption.

## 5. Stream & job semantics (resolves critic 1.10/1.11 — SVA written first)

- One handshake: `valid`/`ready`, data stable while `valid && !ready`; every
  boundary crossing goes through a skid.
- **`done` implies all result beats have been ACCEPTED downstream** (post-skid,
  not post-generation). run1's failure mode (stale beat in the skid when the
  next job's reject was tested) is impossible by construction: SEQ will not
  assert `desc_ready` for job N+1 until job N's `done`.
- `busy` = (state ≠ IDLE) **or** (output beats pending anywhere in the block,
  skids included).
- These are SVA properties (`apex_stream_sva.svh`), compiled in every build —
  a Makefile *gate*, not an aspiration (two of six studied repos shipped SVA
  that never compiled).

## 6. ASU fixed-point plan

- Online softmax state per row: running max `m` (INT32), running sum `l`
  (Q16.16, 48b), output rescale factor applied on the fly (FA-3 pattern; no
  materialized score tensor — the MXE↔ASU handoff is a streamed tile queue).
- `exp(x)` for x ∈ [−16, 0] (post max-subtraction, scaled): **256-entry** LUT
  on the top 8 bits + linear interpolation on the remainder; max abs error
  budgeted ≤ 2⁻¹⁰, verified exhaustively over the input domain vs NumPy
  (D-014: a 64-entry figure is fp16-SIMD-specific — a fixed-point
  chord at h=0.25 has inherent error h²/8 ≈ 7.8e-3, measured 6.9e-3; 256
  entries measure 4.63e-4. ROM cost 1 KB — cheap).
- RMSNorm: `x · rsqrt(mean(x²) + ε)` with our `rsqrt` unit
  (30-cycle, multiplier-free) — gated on re-verification under Verilator with
  a fresh watchdog TB (their TB style hangs under Verilator; critic item 5).
- Wide-D RMSNorm (C-RMSW, landed 2026-07-20 on `comp/wide-rmsnorm`): the same
  `asu_rmsnorm` elaborates to `RMS_D_MAX` up to 8192 (7B hidden 3584 = 28·128)
  by replacing the pow-2 mean shift with one scalar multiply,
  `mean2 = (sum2·μ_D) >> (s+16)`, μ_D/s generated from the golden `_wide_mu`
  and provenance-gated (`verif/asu/wide` `params-check`). mean2 ≤ 2¹⁴
  independent of D, so the rsqrt operand range and the per-element emission
  datapath are unchanged; the D≤128 elaboration is bit-identical
  (byte-identical anchor-suite logs, zero test edits). Acceptance: bit-exact
  vs `rmsnorm_fx_wide` `(y, r, nrm)` — proven in `verif/asu/wide`
  (RESULT.md; k = 2..28 table sweep, sum2 = 2²⁷ corner, 4/4 mutation gate).
- Acceptance for ASU quality: softmax output vs float64 NumPy within
  ≤ 2 ULP of Q1.15 per element AND end-to-end attention output error bounded
  on real distributions (calibration data from the vendored test vectors).

## 7. CSR map (32-bit, KVQ-ISA style; one window for the tile)

`0x00 CTRL` (soft_reset, enable) · `0x04 STATUS` (idle, desc_error sticky,
per-block busy bits) · `0x08–0x1C INFO_*` (N, D, G, tier, version) ·
`0x20 TIER_CTRL` (KVQ tier + TIP override) · `0x24 THRESHOLD_REG` (TIP,
per-layer) · `0x28 FLUSH` (KVQ partial-group flush trigger — the missing
upstream feature) · `0x2C IMPORTANCE_BASE` (TIP accumulator readout window) ·
`0x30+ PERF_*` (cycle counters: load/compute/drain per block — so the run2
load-bound pathology is *measured*, not discovered).

## 8. Verification plan (the actual product)

**Layer 0 — foundation re-verification (before any new RTL):**
- V0.1 KVQ under **Verilator** with a C++ scoreboard built from
  `an independent fixed-point reference.hpp`, replaying all 9 frozen golden vectors (compile ≠
  verification — critic item 1). Vendor vectors + SHA256SUMS.
- V0.2 Stalling-consumer test demonstrating the upstream `tready` bug, then
  the fix (read-side skid + burst FSM stall), then the same test passing.
- V0.3 `the ratio-test unit` re-verified under Verilator at SCORE_WIDTH=32.
- V0.4 `rsqrt` fresh Verilator TB, all vectors, watchdogged.
- V0.5 run1 `stream_skid` standalone randomized TB + corrected SVA.
Only blocks passing V0 earn vendored status; failures get rewritten.

**Layer 1 — per-block:** bit-exact golden model per block (NumPy primary,
`an independent fixed-point reference` C++ where it exists), directed + constrained-random with
per-test seeds, SVA compiled always, manual coverage buckets with reachability
notes (run2 pattern). Mid-operation reset test REQUIRED per block (the hole
run2 left).

**Layer 2 — integration:** pairwise (MXE↔ASU tile queue, KVQ↔MXE dequant
feed, TIP↔KVQ tier control), each with a stalling/backpressure sweep.

**Layer 3 — end-to-end:** one full decode attention layer vs a float64 NumPy
golden transformer layer (weights + activations from a fixed-seed generator +
one real-calibration set), across all three KVQ tiers × {D=64, D=128} × K/V
lengths hitting full-group, partial-group, and eviction paths. Error budget
propagated analytically from C-1/C-2 and asserted, not eyeballed.

**Process:** every decision lands in this file with an ID; every RTL module
header cites the decisions it implements; `make sva` and `make coverage` are
build gates; a `STATUS.md` is updated only from fresh full-suite runs
(anti-fabrication: report generation is mechanically tied to the last run's
parsed results — never hand-written).

## 9. Resolved-decision register

| ID | Decision | Basis |
|---|---|---|
| D-001 | INT4 contract = per-channel INT4 codec [-8,7] RNE EPS=2⁻¹⁴ | golden vectors are arbiter |
| D-002 | RNE rounding in quant + requant epilogue | uniformity; critic 1.2 |
| D-003 | INT32 accumulators, K ≤ 2048/descriptor | closes an INT24 accumulator gap |
| D-004 | True systolic rewrite of MXE (not broadcast wall) | critic 1.15; timing at N≥8 |
| D-005 | Double-buffered weights + wide load port | run2 load-bound pathology |
| D-006 | `done` ⇒ post-skid acceptance; SEQ serializes jobs | run1 failure root cause |
| D-007 | KVQ read path gets skid + honors tready | upstream AXI bug |
| D-008 | Partial-group flush via CSR `FLUSH` | upstream open item |
| D-009 | Byte-aligned 64b-padded KVQ record | critic 1.4 |
| D-010 | fp32 KVQ read bus; narrowing (if any) outside parity boundary | contract §1 |
| D-011 | TIP at SCORE_WIDTH=32, THRESHOLD_REG per layer | critic 1.9 |
| D-012 | Verilator-first TB discipline; SVA compiled as build gate | the observed anti-pattern |
| D-013 | Vendored frozen vectors + SHA256SUMS; regeneration out of scope | provenance; critic item 10 |
| D-014 | ASU exp LUT is 256-entry (not the reference design's 64) | measured 6.9e-3 @64 vs 4.63e-4 @256; budget 2⁻¹⁰ |
| D-015 | KVQ vendors the V0-**patched** top (`verif/v0/kve/patched/`), never upstream verbatim | V0.2: upstream drops 7171/13824 beats under backpressure, 5450 corrupted; patch verified clean + full parity preserved |
| D-016 | KVQ vendored copy must also fix 3 latent V0 findings: unwritten-SRAM-read FSM hang (no timeout), ST_IDLE read-vs-write beat drop, residual-buffer cnt==G index alias | V0.1 code-review findings; the alias directly blocks the D-008 partial-flush retrofit |
| D-017 | TIP THRESHOLD datapath is a rewrite, not a lift: upstream THRESHOLD param is DEAD (RHS hardcodes ×10 as sum<<3+sum<<1; UNUSEDPARAM confirmed) and CMP_W assumes T<16 | V0.3 F-1; D-011's per-layer THRESHOLD_REG needs programmable shift-add + CMP_W = SUM_W + clog2(T_MAX+1) + tile-length framing guard (V0.3 F-2) |
| D-018 | rsqrt latency budget is 31 cycles (not the header's 30), 32/op back-to-back; ASU wrapper serializes issue (module has no busy/ready and silently drops mid-op requests) | V0.4: measured constant across 1.067M ops; doc off-by-one |
| D-019 | run1's failure root-caused: TB drain bug (final beat stranded by same-timestep y_ready drop) + the DUT busy/done wart D-006 already fixes; `stream_skid.sv` itself is clean (100k+ txns, 9 fresh SVA) and is VENDORED for XBR | V0.5 waveform-level evidence (cyc 193–228 trace) |
| D-020 | KVQ soft-reset semantics (closes verif/kvq/sb B-1..B-4): CTRL.soft_reset is never lost (reset has priority over the FSM case, incl. single-cycle states); it ABORTS-AND-DISCARDS all input-side/datapath work the same cycle (sync `clear` into cq_value_path/cq_key_path/cq_quant_unit_syn; SRAM+occupancy preserved); an in-flight m_axis read burst always COMPLETES (§5 has no reset carve-out — no valid retraction, tlast framing preserved) with STATUS.idle=0 until the final beat is accepted, and STATUS.idle also ANDs in the datapath busy flags so it can never lie; a read not yet at its first beat is dropped with 0 beats. Plus: CQ-4+ outlier readback preserves the raw fp16 sign bit (−0.0 → 32'h8000_0000, identity replay / D-010) via a sign-force on the outlier lane only | independent verification findings B-1..B-4 (verif/kvq/sb/RESULT.md); §5 no-retraction is arbiter |

## 9b. The attention numeric seam (D-021) — added 2026-07-08

Block-level work cannot expose this; integration forces it: KVQ's read bus is
**fp32** K̂/V̂ (D-010), but MXE multiplies **INT8×INT8** (C-4). And softmax is
NOT scale-invariant (scale = temperature), so score scales cannot be waved
away. Resolution, consistent with C-4 (array sees one dtype):

- **K̂ feeder requant**: per-token INT8 quantization of K̂ rows at the MXE
  feeder (C-1 machinery: amax → fp16 scale s_k[j], RNE, clamp), giving
  Q·K̂ᵀ INT32 accumulators with composite per-column scales s_q·s_k[j].
- **Score-dequant stage** (new small block, MXE→ASU path): per element,
  score_fx[j] = acc[j] · (s_q · s_k[j]) emitted in the ASU's Q·.SCORE_FRAC
  fixed-point format — because softmax needs true magnitudes.
- **P·V̂ path** (amended by the golden build — the v0 text was numerically
  unimplementable: per-token V̂ scales s_v[t] sit on the CONTRACTION axis of
  P·V̂ and cannot be applied in any INT32 epilogue): fold s_v[t] into the
  probability BEFORE P-requant (c[t] = p[t]·s_v[t], then per-row INT8 quant);
  a single composite epilogue scale then carries both descales. This is the
  S-4 resolution in `golden/apex_golden/attention.py`, which is normative.
- **Gate**: `golden/apex_golden/attention.py` implements this exact chain and
  the Layer-3 acceptance budget (attention output error vs float64 on real
  calibration tensors) is measured there FIRST; apex_top integration is
  blocked until the golden chain meets budget. If INT8 P or double-quantized
  K̂ blow the budget, the documented fallbacks are (a) INT16 P·V̂ lane or
  (b) mixed INT8×FP16 OS mode (the reference design's MXE choice) — decided by data.

| ID | Decision | Basis |
|---|---|---|
| D-021 | Attention seam = feeder-requant + score-dequant stage + INT8 P·V̂, gated on golden-measured error budget | fp32 read bus (D-010) × INT8 array (C-4) × softmax scale-sensitivity; surfaced by integration gap review 2026-07-08 |
| D-022 | **Attention-score tier policy: CQ-4 alone is quality-fragile for K̂** — measured e2e attention error on outlier-bearing tensors: CQ-4 25.7% of value scale vs CQ-4+ 4.8% vs CQ-8 2.8%. Error is score-side (INT4 K collapse), NOT the P·V̂ lane (≤1.2% everywhere) — so D-021's INT16-P fallback is REJECTED as pointless; the load-bearing mitigation is TIP-driven tier selection / CQ-4+ outlier masks on outlier-bearing layers. TIP is hereby promoted from nice-to-have to REQUIRED for quality. | golden Layer-3 measurement, 105 cases, 2026-07-08 (goldenE2E build); D-021 fallback triage |
| D-023 | Layer-3 gate needs an ABSOLUTE acceptance threshold (auditor: the a-priori hard bound is vacuous at 1.27e3×; composition+statistical gates bite but share derivation lineage). Adopt: e2e attention error ≤ 5% of value scale per tier-appropriate config (CQ-4+ / CQ-8; CQ-4 documented as out-of-quality-budget on outlier data per D-022) + the existing relative gates. | integration audit 2026-07-08 |
| D-024 | **In-tile runtime tiers = a KVQ TIER BANK** (`apex_kvq_bank`): three verified `kvq_engine` instances (CQ-8/CQ-4/CQ-4+) behind a live tier mux — engine TIER stays a synthesis parameter (VAL_BPV / KEYG generate / record layout / CR_* all elaborate from it), so runtime control is by BANKING verified engines, never by editing one. live_tier = TIER_CTRL.tier_sel (host) or auto_tier[rt_tip_blk] (TIP-auto: every accepted TIP decision beat writes the 128-entry per-block map — D-022 actuation). GRANULARITY (honest): quasi-static per KVQ transaction run — per token (values), per key GROUP (keys, tuser=0 + WRITE_ADDR base + D-008 FLUSH for partial groups), per record (reads), per block (auto mode); switch only at bank-quiescent points. CQ-4+ needs the build's OUTLIER_K/MASK_FILE ROM; without it engine 2 degenerates to CQ-4 and CSR INFO_TIER truthfully reports {mask?CQ4P:0, CQ4, CQ8} (never a hardwired 0x7). *(D-027 update 2026-07-23: the mask is now runtime-CSR-loadable — OUTLIER_K stays structural; INFO_TIER bit 2 became LIVE: TIERS[2] && mask_valid.)* Golden mirror: `attention_core(tier_map=...)` / `kvq_roundtrip_tiermap` (runs of equal tier compressed independently), gated in golden section E. | F-2/D-022 closure 2026-07-09; kvq_engine TIER structurally synthesis-time |
| D-026 | **A2 KV-REC-DEDUP (supersedes D-009's key-record COMPOSITION only; 64b-pad rule unchanged).** Key record = [tag {ssid[6:0],1}][OUTLIER_K fp16 lanes][D×4 codes][pad64]; group scales stored ONCE per committed group in a persistent `scale_bank_store` (SCALE_SETS=4 × D×16b, outlier lanes forced 0). ssid = commit-sequence mod SCALE_SETS stamped in the tag (readback primes the bank row one cycle off the tag); allocator is commit-time, wrap-with-fault (sticky SB_OVWR = IRQ_STATUS[1]; each D-008 flushed partial group consumes one set). Bank is deliberately NOT on dp_clear — rows persist with the records across D-020 soft reset; only hard reset clears (audit reset-attack bucket guards this). **Integration sizing rule (found the hard way — l3 adv_outlier caught set exhaustion):** any instantiation whose flow stores more than SCALE_SETS live groups before readback MUST size SETS ≥ ceil(T_ROW_MAX/KVQ_G) (next pow2; `apex_top` plumbs KVQ_SETS=8 for its envelope). SB_OVWR flags violations at runtime. Key rows 1344→320b (D=64 k≤2) / 2624→576b (D=128); stored whole-KV 1.23–1.28× → 3.16–3.51×, asserted three ways in `test_effective_bits.py`. (D-025 reserved: P6 per OPTIMIZATION.md.) | claim-detox audit 2026-07-13 (the ~21 b/v landmine); golden-first landing 2026-07-14, all KVQ+top suites re-passed with derived counts |

| D-027 | **S12 LOADABLE OUTLIER MASK — CQ-4+'s mask becomes a CSR runtime input; OUTLIER_K stays structural.** Engine AXI-Lite window 0x50–0x60: MASK0-3 stage 32 mask bits/word (staged readback; beyond-D bits RAZ/WI), MASK_CTRL bit0 commits — **effective iff popcount(staged) == OUTLIER_K** (lane budget elaborates the record width, so *which* channels is runtime, *how many* never is — D-024's principle), else sticky MASK_ERR (IRQ_STATUS[2], live mask unchanged). An effective commit while the store is logically occupied (occupancy > 0 or an open key group in any ST_K* phase) raises sticky MASK_SWAP (IRQ_STATUS[3]): D-026 records do NOT self-describe their mask (the 4'd1 sentinel is a placeholder, lane order = ascending-mask rank, bank outlier scales forced 0), so decoding pre-commit records post-commit is a host-contract violation the flag audits — hardware does not police beyond it (SB_OVWR philosophy; like SB_OVWR at allocator wrap, the flag also fires at legal re-encode boundaries since occupancy is monotone — the host W1Cs it). Live mask + ownership are hard-reset-only (records persist D-020 soft reset, so their decode key does; hard reset restores the build default: MASK_FILE if given, else zeros/invalid). **`mask_valid` is COMPUTED (popcount(live)==OUTLIER_K), never stored** — ROM builds valid at reset, the maskless b128 shape invalid until commit, malformed ROMs truthfully invalid. OUTLIER_K=0 builds keep the window reserved (DEADBEEF/WI; e0/e1 datapaths stay constant-folded — the bus is a select: build-ROM wire until first commit, so MASK_FILE builds are structurally byte-identical). Tile: engine 2 exports `mask_valid` through `apex_kvq_bank`; **CSR INFO_TIER bit 2 is now LIVE** = TIERS[2] && mask_valid_e2 (glue read-override, ERR_STICKY pattern; b128 reads 0x3 at reset → 0x7 after a valid commit — "INFO_TIER never lies" made dynamic). Numerals 0x5C/0x60 also appear in the tile WALK window (D-028) — physically separate buses, no conflict. Full contract + nine pinned refinements: `docs/design/S12_LOADABLE_MASK.md` §3/§5; executable contract `golden/tests/test_mask_semantics.py` (masksem, 13/13). | golden-first landing 2026-07-16 (79e8b2b); stages 1–4 landed 2026-07-23 on `comp/s12-mask`. Evidence: same-box A/B 79/79 count+gate lines byte-identical (zero test edits); `verif/kvq/mask` rom 800 + csr 1887 checks / 0 fails, 2 mutants caught; L3 28/28 incl. NEW `adv_outlier_d128_cq4p` 17,642 checks / 0 errors (**F-2 residual RETIRED**: D=128 CQ-4+ bit-exact at tile level through the CSR-loaded mask); walker split 24/28 + 4 refused, no checks lost; 27 pre-existing L3 case files byte-identical old-gen-vs-new; 3/3 tile mutants. Logs: `verif/kvq/mask/logs/` |

| D-028 | **B1 HARDWARE LAYER-WALKER + on-tile scale composition (L-T7).** `seq_layer_walker` emits the score+pv control stream autonomously from a compact layer descriptor; `seq_walker_comp` folds the fp16 seam composites on-tile. It drives `ds_*` exactly as the host does, so the verified `seq_walker` D-006 FSM is UNCHANGED and still serializes jobs for the walker — `apex_pkg` untouched (walker types live in a new `seq_walker_pkg`), `csr_regs` unmodified (WALK window 0x5C-0x6C is glue, ERR_STICKY pattern). **SCOPE IS FENCED, NOT ASPIRATIONAL:** (a) score+pv only — RoPE/SwiGLU/o-proj/residual are not in `apex_top`, and `store_kv`/`q-inject` are 87.5% of the L3 stream but are *injection scaffolding*, not decode control; (b) **CQ-8 only** — a non-CQ-8 descriptor is REFUSED (`WALK_ERR_TIER`), never walked with a relaxed equivalence, and the grouped-tier commit-time amax pass is a named follow-on (B1b); (c) the PV requant pair is a **host-LOADED descriptor field**, because `calib_requant(amax\|acc_o\|)` depends on the output of the very GEMM the descriptor configures — causally circular, so no on-tile unit can derive it in one pass. Autonomy is in SEQUENCING; that one calibration input is named, not hidden. **Scale source:** the composite unit SNOOPS the KVQ store stream and recomputes the record scale with `cq_fp_pkg::scale_from_amax` — the engine's own function, over the same beats — because at CQ-8 the stored record scale is bit-identical to the feeder's read-time `s_k[t]`/`s_v[t]`. That avoids porting `cqv_scale` out of the B2 lane's verified engine; Option A (an additive engine port) is the named fallback if a numeric gap ever appears. | golden-first landing 2026-07-21. Evidence: walker-mode L3 24/27 walked + 3/27 refused with NO checks lost; host-mode L3 **byte-identical** (27/27 cycles+checks+errors, 3,417,254 cycles both sides) vs a rebuilt pre-B1 tile; tile mutation gate 3/3; unit suites 5,940 emissions + 123,136 composite checks, 4/4 mutants. Three v1 contract claims were found FALSE and corrected before build (`docs/design/B1_WALKER.md` §8) |
| D-029 | **IB-WALK fmt=1 FULL-LAYER DESCRIPTOR (versioned walker wire format).** The walk descriptor gains a format nibble (`GEOM0[31:28]`) with read-side discovery at `WALK_STATUS[15:12]` FMT_SUP (landed tiles read 0 = fmt-0-only; the host never loads fmt=N unless bit N reads 1) and a v1.1 HARDENING: the landed v1 check now REFUSES an unknown fmt (WALK_ERR_DESC) instead of silently mis-walking it. fmt=1 is a 64-word image behind the EXISTING DPTR/DDATA window (addresses stay 0x5C-0x6C, zero new CSR): geometry/model scalars, step-enable mask, a 10-entry DDR tensor table at IB-FUEL's frozen `fuel_req` {base_64B[29:0], beats_64B[55:30], tag[63:56]} with the n-major/k-minor job decomposition (accumulate=(k0>0), C-KSPLIT on-tile), and a per-step patch region = STEP word + (H+2) requant slots + 4 JOBC composite slots -> **H+11 = 39 MMIO/layer/step derived** (figure history H+5 -> H+6 -> H+11, LEVEL_C_INTEGRATION §9.1; autonomy remains SEQUENCING — the I-B claim is golden-driven replay, R2). `seq_layer_walker2` wraps the UNCHANGED D-028 engine per head (the per-head v1 descriptor is synthesized from fmt=1 state — "a descriptor extension, not a rewrite" made structural) inside a micro-step program over the full IB_LAYER §3b step set (norm lj_* jobs; ROPE as a level-arm of the per-config-RESIDENT phase-K table; per-engine WRITE_ADDR store sequencing; deq/swiglu/residual via the single-ingress drive nets with JOBC composites REWRITTEN per chunk). GQA engine mapping is the GOLDEN `h // (H/H_kv)` via a division-free running remainder — the pow-2 shift form was wrong at Qwen-7B's 28/4=7 (§9.1 R3 amended on this lane's escalation). Fences: CQ-8 only (§A-1 carried), T<=128, one layer per kick, `n_kv_heads <= N_ENG` under per-KV-head banking, degenerate-FFN refused; registered-accept in every LAYER-port wait state (the §3b units' combinational job-ready drops at the accepting edge). | unit-level landing 2026-07-22 (`comp/ib-walk`, IB_WALK.md §4 stages 1-5). Evidence: full-§3b layer walk BIT-EXACT vs the golden-gated trace — 6,858 checks/0 errors at 7B geometry (28 heads, 4 engines, 3,816 KVQ writes), 309/480 at tile scale; L3 host AND walker modes byte-identical to a pristine-tree baseline after the glue delta (+walkfmt probe: FMT_SUP read through the real CSR bus, wrap proven by the refusal landing at the wrapped pointer); SV-vs-mirror check equivalence dynamic over 12,020 rows; 17 mutants killed across three gates. Four kill signatures pin the load-bearing hazards: m4_fmtskip — without the hardening a fmt=1 descriptor silently mis-walks as v1 (emitted `ROUTE 00ef`/`SJOB 8`); m5_gqamap — the pow-2 boundary bug puts head 7 on engine 0 at the non-pow-2 group (+CS corruption from the wrong cache); m10_jcorder — a JOBC order swap feeds swiglu gate*gate (killable only because the trace recipe differentiates s_wu); m12_waddr — a store master writing READ_ADDR silently LAUNCHES reads (caught as KVW-vs-KVWA). W-G2 tile gate + W-G3 flagship replay are COMBINE-owned (LEVEL_C §9.1 agenda: D-030 narrowing conformance, L4 step-order arbitration, §3b phase-table residency) |
| D-030 | **C-LBUS GOLDEN BUS-COMPOSITION MODE (owner-SIGNED 2026-07-23, LEVEL_C_INTEGRATION §9.1).** Additive golden mode with three flags (`x_f16`, `rope_in_f16`, `scale_f16grade`) defaulting OFF — OFF-mode proven byte-identical to legacy golden (banner literal unchanged; the arbiter's default semantics untouched). Pins the points where the tile narrows on a physical bus but legacy golden composed in unnarrowed float64; the feared fourth point (residual/SwiGLU operands) DISSOLVED — 2,008 exact-Fraction checks prove single-RNE realizability, no flag exists. Pinned deltas: BUS_ON e2e 2.4-3.8e-3, err_ON <= err_leg on all cases; d(rope)=0 e2e everywhere despite 33/254 differing q-hat/K-hat bus words (absorbed at KVQ/feeder requant — pinned as measured, which is WHY L4 keeps q-hat/K-hat bus checkpoints first-class). **Canonical JC grade = significand-RNE to 11 bits at RETAINED fp32 exponent of the FINAL composite — explicitly NOT an fp16 round-trip** (below 2^-24 the trip is off by orders of magnitude; the distinguishing example is in IB_LAYER §3b, and the divergence was LIVE in IB-WALK's first vectors — a float16-cast grader passed a mid-range-only suite and was caught by the published conformance checks). Choreography canon rides with it: arm-before-stream, JC-before-ITS-push, consumer-before-producer push in chained flows, one-pending-push w/ STATUS poll, 1-cycle CSR readback. BUS_ON is the L4 arbiter (S5 gates against it). | evidence: golden `layerbus` gate (OFF bit-identity + lemmas + pinned registers, in `make -C golden test`); docs/results/ib_layer_bus/RESULTS.md; IB-WALK conformance fix commit (canonical values regenerated, permanent generator selftests) |
| D-031 | **W4B GIVEN-SCALE FEEDER — the adopted G=32/(B) weight path in RTL** (`mxe_wfeed_w4b` + `w4b_fp_pkg`, `comp/w4b-feeder`): consumes the host-computed stripe-global requant scale as a SIDEBAND (the D-021 delta B3's adoption verdict disclosed — multi-job K-split stripes share one epilogue factor only the host can see), packed||scales DDR layout, ON-PATH placement per W4B_FEEDER.md §4 integration notes. Golden arbiter = `wfeed_w4_to_i8(realization="B")` + the thin given-scale oracle `wfeed_w4b_to_i8` (reduction identity gated). **Feeder group size FROZEN at G=32** by the closed adoption matrix (every cell n=10,042: final recipe = W4 G=32/(B) + DIRECT-from-source host prep, −1.0 pt vs shipped weights, BEATS the INT8 feed; direct@G16 +0.41 borderline at +11% traffic — G=16 validated fallback). Merged CSR-DISABLED at the I-B combine; enable gates on the full matrix rerun with the feeder instantiated. | lane complete 2026-07-24 (tip 1e04239): pkg 31,024 vectors; TB 403 job runs/0 errors across regimes; EXHAUSTIVE 4,063,104-point sweep; 5/5 signature mutants; suite green from clean checkout |

## 10. V0 outcome summary (2026-07-07)

| Block | Verdict | Vendored from |
|---|---|---|
| per-channel INT4 codec KVQ | PASS_WITH_FIXES — 7 configs × ~350k checks, 0 fails, first-ever Verilator run; G=128-keys obligation met | `verif/v0/kve/patched/` (D-015/D-016) |
| the ratio-test unit | PASS — 143 frozen + 10,673 tiles incl. W=32, 0 fails | APEX programmable-threshold design (D-017) |
| rsqrt_unit | PASS — 8/8 golden + 1.067M dense sweep, 0 mismatches | upstream verbatim, 31-cycle budget (D-018) |
| stream_skid | PASS — 100k+ randomized txns × 2 simulators, 9 SVA | run1 verbatim (D-019) |
| apex_golden (Python) | PASS — 9/9 frozen vectors bit-exact; compute properties all-pass | golden/ (this repo) |

## 11. v0.1 shipped envelope (2026-07-09 — from L3 verification + TRACEABILITY.md)

**apex_top v0.1 is a verified CQ-8 / D=64 / T≤64-per-job, host-sequenced
attention tile — no more, no less.** 56,406 L3 checks bit-exact vs the golden
chain inside that envelope; outside it, the feasibility register (F-1 T-cap,
F-2 CQ-4/4+ unreachable at tile level + misleading INFO_TIER, F-5a/b D=128
beat-count truncation + column aliasing) documents structural gaps, each
fenced by a kept-failing regression. Full-tier/full-D coverage exists at L2
through the real kvq_engine+feeder+MXE chain. Decision matrix: 25/28 PROVEN,
3 PARTIAL (D-004 structure test, D-005 overlap demo, D-022 TIP-driven tier
selection in-tile). v0.2 scope = the F-register + the 3 PARTIALs.
See TRACEABILITY.md (evidence matrix) and STATUS.md (machine-generated).

**Update 2026-07-21 (D-028, B1 walker):** the tile is no longer *only*
host-sequenced. `apex_top` now also runs the attention decode step
**autonomously** — the score+pv control stream is emitted on-tile from a
compact descriptor, at CQ-8, with the seam scale composites folded in hardware.
The envelope is explicitly fenced: score+pv only (the other layer blocks are
not integrated), CQ-8 only (other tiers are REFUSED, not degraded), and the PV
requant pair stays host-loaded (causally circular on-tile). **Host mode remains
the verified fallback and is byte-identical** — 27/27 L3 cases match the
pre-walker tile on cycles, checks and errors, which is the compat proof that
lets the walker land without re-qualifying the host path.

**Update 2026-07-09 (D-004/D-005 closure):** matrix now 27/28 PROVEN, 1 PARTIAL
(D-022 actuation, L-T3). D-004 closed by `verif/mxe/struct` — a whitebox
structural discriminator (per-PE (r+c)-offset arrival, per-hop psum/acc
staggering, single-cycle glitch hop locality) plus a mutation gate proving a
functionally-equivalent broadcast MAC wall passes the functional smoke but is
killed by the structural TB. D-005 closed by implementing load-under-compute
in `mxe_ctrl` (shadow-bank weight loader decoupled from the compute FSM; bank
swap only at wavefront-clear COMPUTE entries; external stream/job contract
unchanged, inside the D-006 envelope) and gating it in `verif/mxe/perf`
against a pinned copy of the sequential controller: overlap_cycles > 0 on
every multi-chunk job + strict cycle win (K=2048/M=64: 56901→53573, 1.062x;
K=64/M=16: 645→541, 1.192x; weight-BW-limited: up to 1.321x and full load
hiding), with a kept-failing overlap regression on the sequential baseline.

**Update 2026-07-09 (F-1/F-5 closure — the tile envelope is now
CQ-8 / D∈{64,128} / T≤128-per-job):** rtl/top only; apex_pkg untouched (no
contract field changed, APEX_VERSION stays 0x0001_0000).
F-5a: aj_nb/wj_nb/job_nb widened to the DERIVED width `NB_W = clog2(D/8)+1`
(apex_stage_buf header localparam; apex_top `STAGE_NB_W`), so nb = BPR = 16
(a full D=128 row) is expressible end-to-end (ports, glue, both TBs).
F-5b: apex_stage_buf PAT_D indexes the column block with the full
legality-checked `sel` (was `sel_q[2:0]` — blocks 8..15 aliased 0..7);
directed regression `verif/top/l3/tb_stagebuf_patd.sv` reads all 16 blocks
byte-exact and kills the reverted-index mutant.
F-1: new apex_top parameter **`T_ROW_MAX` = 128** — the per-job
attention-row envelope constant; sizes `seam_score_dequant` N_MAX and the
`apex_scale_quant` per-job buffers (`SQ_COLS_MAX = max(CFG_D, T_ROW_MAX)`;
the S-4 P-requant job is T columns); bounded by ASU SM_ROW_MAX=1024. A c8
row exceeding one stage-buffer row (T=128 at D=64) is staged across
act-stage bank1+bank0 by the host scripts.
Proof, all BIT-EXACT vs golden/apex_golden/attention.py from clean: smoke
runs the full chain at D=64 AND D=128; L3 runs 24 cases / 108k+ scripted
checks incl. full-length calib d64_T128 (T=128), adv/outlier1000 (T=128),
calib d64_T70, calib d128_T100 (D=128, T=100 partial-group), 2 random D=128
full chains, and the former F-5 wedge case `bug_d128_stagebuf_nb` — all
flipped from kept-failing to kept-PASSING regressions (*_T64sub substitutes
retired).

**Update 2026-07-09 (F-2 / D-022-actuation / F-3 closure — the tile is now
ALL-TIER: CQ-8 / CQ-4 / CQ-4+ / TIP-auto mixed, D∈{64,128}, T≤128):**
F-2 closed by D-024 (KVQ tier bank, see §9 row): grouped keys (rt_kv_user
tuser routing + WRITE_ADDR group sequencing), D-008 FLUSH reachable through
the tile CSR, OUTLIER_K/MASK_FILE plumbed as tile parameters (b64 builds
carry a real {5,50} mask ROM), CSR INFO_TIER reports the build TRUTH
({mask?CQ4P:0, CQ4, CQ8}; csr_regs TIERS parameter). D-022 actuation
closed: TIER_CTRL.tip_override=1 makes TIP DRIVE the tier — accepted td_*
decision beats write the per-block auto_tier map, and `tip_auto_mixed`
(L3) demonstrates outlier-bearing blocks restored at CQ-4+ / benign blocks
at CQ-4, checked bit-exact vs `attention_core(tier_map=...)` (golden
section E gates the tier-map model: uniform==single-tier bit-exact,
run-composition, and the load-bearing calib measurement 12.89%→4.03%).
F-3 closed (sticky half): every per-source sticky is latched in the tile
ERR_STICKY window (0x58, W1C, set-wins; bit 14 pulses TIP
frame_err_clear); every L3 script + smoke ends with set→clear→verify.
Proof from clean: smoke ×3 tiles (D=64 CQ-8, D=128 CQ-8, D=64 CQ-4+
grouped-keys+FLUSH+mask: 2176/4240/2176 checks, errors=0), L3 27 cases /
135,241 checks (adv_T1_cq4p T=1 tile-level FLUSH, adv_outlier1000_cq4p
T=128 8 full G=16 key groups, tip_auto_mixed 2-pass TIP-auto decode) +
coverage gate + 3/3 tile mutants; golden 120 cases + section E; L2 18,863
re-proven; verif/csr + verif/audit_csr re-run for the csr_regs TIERS
parameter. Fail-first evidence: verif/top/l3/logs/prefix_f2/
prefix_f3_keptfail_before_fix.log. Remaining tile fences: F-4 (no
tile-level eviction test), L-T7 (host sequencing), F-3 residual (TIP gets
no decision for tiles >8 — profiling uses 8-score tiles), F-2 residual
(one mask ROM per build; b128 ships maskless → INFO_TIER=0x3).

**v0.2 shipped envelope (closure, 2026-07-09 — sealed by the independent
from-clean reproduction, TRACEABILITY.md §0e):** apex_top v0.2 is a verified
**all-tier (CQ-8 / CQ-4 / CQ-4+ / TIP-auto mixed per-block), D∈{64,128},
T≤128-per-job, host-sequenced attention tile** with a structurally-
discriminated systolic MXE (D-004) and load-under-compute weight
double-buffering (D-005, up to 1.32x vs the pinned sequential baseline) —
135,241 L3 checks + 3 smoke tiles bit-exact vs the golden chain; decision
matrix 29/29 PROVEN (D-001..D-024 + C-1..C-5), zero PARTIAL. `apex_pkg.sv`
untouched across all of v0.2 (no contract change; APEX_VERSION stays
0x0001_0000). **Update (A2/D-026 closure, 2026-07-14):** the key-record
composition was superseded by D-026 (scale-bank dedup; §4) and the full KVQ +
top matrix re-passed with derived counts — decision matrix now 30/30
(D-001..D-024, D-026 + C-1..C-5; D-025 reserved). `apex_pkg.sv` still
untouched. v0.3 backlog, in envelope-value order: F-4 tile-level KVQ
eviction/capacity; L-T7 autonomous layer-walker (retire the host-sequenced
boundary); F-3 open half (TIP decisions for >8-score tiles); F-2 mask
residual (per-build ROM → loadable mask, b128 currently maskless); L-M6
MXE-phase PERF counters through the tile CSR window.
