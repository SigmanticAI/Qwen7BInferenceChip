# tinyNPU/APEX — Prioritized Optimization Roadmap (v0.3 → v1.0)

**Grounding reality (from ARCHITECTURE.md §11 + measured numbers):** Decode is doubly bound — host-transaction-bound first (~300 MMIO round-trips/step, tile idle ~90%) and memory-bandwidth-bound second (<40 GB/s straw, weight bytes > KV bytes at D=64). Prefill is compute-bound, but the binding compute is the **ASU divider (~540k cyc @ T=128), not the MXE (~65k cyc)** — the array is 8x faster than its softmax feed. Every ranking below follows from these three facts.

*Input note: lenses 1 (compute/dataflow) and 2 (memory/KV) arrived in full; lenses 3–6 arrived truncated — their named items (GQA, formal, multi-tile/DRAM/silicon, runtime outlier detection) are placed from the tier hints and the ARCHITECTURE decision register, flagged where the proposal text was not available.*

---

## Merges & dedupes

| Merged item | Source proposals | Rationale |
|---|---|---|
| **M1: Softmax emission bandwidth** | P1 (radix-4 divider) + P6 (FA-true deferred normalization) | Same bottleneck, two horizons. P1 is the safe microarchitectural fix (bit-identical, ships now); P6 solves it structurally but is a D-numbered contract amendment (golden-first, D-021 pattern). P1's own risk note says it ships first regardless — so they're one workstream, staged. |
| **M2: KV management stack** | TIP-EVICT (F-4) + walker autonomous-retire (P3 v2) | TIP-EVICT's autonomous mode explicitly pairs with L-T7; advisory mode does not. Split: advisory eviction is standalone, autonomous eviction is a walker feature. |
| **M3: MXE throughput track** | P4 (ingest-under-compute) + P5 (16x16 array) | Hard sequenced dependency: P5's beat-pair gathering doubles ingest beats, making P4 mandatory. P5 without P4 is 4x PEs behind an unfixed feed. |
| **M4: Bytes-per-token** | P2 (native W4) + KV-REC-DEDUP | *Not* actually overlapping — complementary halves of decode traffic. W4 attacks weight bytes (the larger term: 16 KB/layer INT8 projections vs 4–8 KB KV at D=64); DEDUP attacks KV bytes **and** on-tile SRAM capacity. Both kept, ranked separately. |
| TIP-eviction duplicates across lenses | KV lens TIP-EVICT is canonical | (Other-lens copies truncated; TIP-EVICT's H2O+sink formulation is the most complete and cites F-4 directly.) |

---

## Tier A — CHEAP WINS (do now, before/alongside FPGA bring-up)

| # | Item | Cost | Why now |
|---|---|---|---|
| A1 | **P1: radix-4 ASU divider** (single-lane variant, not multi-lane) | small | Pure microarchitecture in `asu_softmax.sv`; floor-division bit-identical, all 135,241 L3 checks stand unchanged. ~3x prefill immediately; unblocks everything prefill-shaped. The multi-lane variant's reorder risk is not worth taking pre-FPGA — radix-4 has no ordering hazard. |
| A2 | **KV-REC-DEDUP** — ✅ **LANDED 2026-07-14 as D-026** (record split: scales to persistent `scale_bank_store`, fp16 field D→OUTLIER_K lanes, ssid-stamped tags, SB_OVWR fault) | medium | Key rows 1344b → **320b** at shipped D=64/k≤2 (384b at k=5); stored whole-KV 1.23–1.28× → **3.16–3.51×**, asserted three ways in `golden/tests/test_effective_bits.py`. `apex_pkg` untouched, C-1 numerics identical, all KVQ + top suites re-passed with derived counts. Landed before FPGA SRAM sizing locked, as required. |
| A3 | **Golden-side eviction study** (TIP-EVICT policy, golden only) | small | D-021 pattern: extend `attention_core` with an eviction mask, measure H2O+sink accuracy at block granularity on the 120-case suite BEFORE any RTL. De-risks B2 for free and quantifies the F-3 stale-accumulator interaction. |
| A4 | **L-M6: MXE-phase PERF counters through the tile CSR** | small | Already last on the sanctioned v0.3 backlog; on FPGA it is the *only* way to attribute cycles (ingest vs compute vs flush vs host-idle). Every Tier B metric below assumes these exist. Ship before bring-up. |

---

## Tier B — v1.0 ARCHITECTURE BETS (worth a design cycle)

Ordered by leverage:

| # | Item | Cost | Dependency / gate |
|---|---|---|---|
| B1 ✅ **LANDED 2026-07-21 (D-028)** | **P3: hardware layer-walker (L-T7) + on-tile scale composition** | large | The decode multiplier: ~300 MMIO → ≤3 per step, recovering the ~90% tile idle. Keep host-sequenced mode as the verified fallback; walker-vs-script transaction-equivalence harness is non-negotiable (the L3 choreography is the spec). The fp16 composite unit must reproduce the golden's single-f32-narrowing (S-3) exactly. |
| B2 | **TIP-EVICT (F-4), advisory v1 → autonomous v2** | medium | #1 named v0.3 backlog item, 80% built (TIP accumulators exist; the stub at `kvq_engine.sv:294` is the gap). Advisory mode after A3's golden gate passes; autonomous retire lands with B1. Multiplicative with A2: ≈17x tokens-per-SRAM-byte combined. |
| B3 | **P2: native W4 weight path** (CSR/route-level variant, not descriptor flag) | medium | The biggest bytes/token lever left after KVQ — halves the *dominant* decode traffic term. Route-level + CSR selection keeps `apex_pkg` frozen (APEX_VERSION stays 0x0001_0000); needs the descriptor/CSR-agreement SVA and per-lane re-run of the C-2 requant equivalence sweep. |
| B4 | **P4: MXE ingest-under-compute + flush hiding** | medium | ~1.55x on big-K WS jobs; closes most of the measured 3.3x-off-ideal gap. Blast radius is producers (L3 scripts, stage-buf drain, golden beat gens, pinned perf baseline) — budget for that, and extend the D-004 staggering discriminator to the early-swap edge. Prerequisite for C1. |
| B5 | **GQA support** *(lens truncated — scoped from tier hint)* | medium | Grouped-query attention shrinks K/V head count, compounding with A2/B2 on both bandwidth and SRAM; standard in target models. Scope against the frozen contract before committing (likely job-sequencing + scale-sideband work, not datapath). **Golden landed (S6, 2026-07-14): `H_kv` in transformer.py + test_7b_plumbing.py; what remains here is the RTL/job-sequencing side. Same S6 push defined the other 7B host contracts (C-KSPLIT/C-RMSW/C-CHUNK) whose implied RTL deltas are: ~~asu_rmsnorm sum2-export/external-r~~ (**RETIRED 2026-07-20 — C-RMSW landed as the self-contained wide `asu_rmsnorm` on `comp/wide-rmsnorm`, verified in `verif/asu/wide`; sum2-export/external-r survives only as a resource-saving follow-on variant, WIDE_RMSNORM.md §5 note**), asu_softmax mmax/lsum export.** |
| B6 | **Formal verification of the §5 stream/job contract** *(lens truncated)* | medium | The valid/ready + done-implies-drained discipline (D-006/D-019) is exactly the property class where bounded model checking beats simulation. Directly deepens the verification moat; also the cheapest insurance on B1/B4's FSM restructuring. |

---

## Tier C — RESEARCH / PRODUCT-SCALE

| Item | Gate |
|---|---|
| **P5: 16x16 internal array** (stream-preserving, ARRAY_N in mxe_cfg_pkg) | Only after A1 **and** B4 — until then the array is not the prefill bottleneck and the extra columns starve. Descriptor-fusion utilization ≥70% or don't bother. |
| **P6: FA-true fusion (D-025)** — unnormalized P·V, per-row 1/l epilogue | Golden-first under the D-023 absolute gate; fixed-point headroom study (unnormalized eᵢ sum to T·32768). If golden fails, A1 already banked the win. |
| **Multi-tile / NoC** | Explicitly out of the §0 scope today; B1 (walker) is the stated prerequisite for the multi-tile story — do not start before it. |
| **DRAM controller / real memory system** | The <40 GB/s straw is currently modeled; real DDR/LPDDR changes the B3/A2 payoff constants but not their ordering. |
| **Runtime outlier detection** *(lens truncated)* | Extends CQ-4+ beyond the per-build mask ROM (F-2 residual: b128 ships maskless). The loadable-mask half is v0.3 backlog; *dynamic* detection is research. |

---

## Top 5 overall

| Rank | Item | One-line leverage | The metric that proves it |
|---|---|---|---|
| 1 | **B1 walker (L-T7)** ✅ **LANDED 2026-07-21 (D-028)** | The tile is idle ~90% of every decode step waiting on the host — no datapath change can beat removing that. | **L3 bit-exact in walker AND host mode: ACHIEVED** — 24/27 cases walked (CQ-8), 3/27 refused (grouped tier, B1b), no checks lost; host mode byte-identical. **MMIO/step: 3 in steady state** (`WALK_RQ` + `WALK_CTRL.go` + `WALK_STATUS` poll), 6 measured at cold start. Two honesty caveats: the *~300* baseline was never sourced (real host range is 356 at T=1 to 24,678 at T=128 — see `B1_WALKER.md` §8 row 13), and the **3** is read off the register interface, not measured across steps, because each L3 case is a single decode step. |
| 2 | **A2 KV-REC-DEDUP — ✅ LANDED (D-026, 2026-07-14)** | The flagship compression claim WAS false at the record level (~21 b/v > raw INT8's 8); D-026 made it true: keys 21.0 → 5.125 b/v incl. bank at shipped D=64/k=2, 4.2× key-SRAM in one move. | Golden effective-bits gate re-pinned: stored whole-KV 3.16× (D=64) / 3.51× (D=128) at k≤2; 6.125 b/v at the k=5 point (RTL-covered by the new sb k=5 config). |
| 3 | **A1 ASU divider** | Prefill's real ceiling is a 1-bit/cycle divider running 8x longer than all six GEMMs combined; smallest fix, largest single unblock. | Emission cycles/element 33 → ≤8; calib d64_T128 total cycles ≥3x down, 0 new fails. |
| 4 | **B2 TIP-EVICT (F-4)** | Eviction is infinite compression for dropped bytes (~5x effective context per H2O literature), and the importance state already exists in fabric. | Tokens-of-context held at fixed accuracy on the golden eviction suite: ≥4x at 20% cache. |
| 5 | **B3 native W4** | Weight bytes, not KV bytes, dominate decode traffic at D=64 — W4 halves the term KVQ never touched. | Modeled decode bytes/token down ≥35% (xw beats/job halved on the PERF counter), bit-exact vs extended golden. |

---

## Called shots

**Best strengthens the differentiator (in-fabric KV + verification moat): B2 TIP-EVICT.** It converts the story from "compresses KV" to "*manages* KV" — tier selection and eviction driven by the same verified in-fabric importance state, which no consumer NPU has — and it inherits the moat directly because the policy is gated golden-first (A3) in the D-021 pattern, i.e., the verification methodology *is* the feature.

**Highest-risk-if-ignored: A2 KV-REC-DEDUP — RESOLVED (D-026, 2026-07-14).** The ARCHITECTURE's own honesty rule ("effective bits/value after padding is recomputed and asserted") used to assert ~21 b/v for an "INT4" key — worse than shipping raw INT8, a credibility landmine under the product thesis. D-026 defused it before FPGA SRAM sizing froze: the gate now asserts 5.125 b/v keys / 3.16–3.51× stored whole-KV at the shipped configs, and every other KV win (eviction, GQA, tiers) now multiplies on an honest base.

---

## What would NOT move the needle (and why)

- **Scaling the array now (P5 as an early move).** Decode gains ~nothing at M=1, and prefill is divider-bound (A1) then ingest-bound (B4) — 4x the PEs behind those two chokepoints is 4x the area at ~1x the throughput. It only pays third, as sequenced in Tier C.
- **An INT16 P·V lane for quality.** Already measured and rejected: D-022 shows the P·V lane contributes ≤1.2% error everywhere; the error is score-side (INT4 K collapse), and the load-bearing mitigation (TIP-driven tiers) shipped in v0.2.
- **Deeper quantization below CQ-4 / more tiers.** Pre-D-026, record overhead (1024 of 1344 bits were the duplicated fp16 field) dominated key size — that's fixed; post-A2 the codes ARE the record (256 of 320 bits at D=64/k=2), but D-022 still says CQ-4 is already quality-fragile on the score side, so sub-4-bit chases quality we can't spare. The remaining lever is the eviction policy (B2), not code bits.
- **Flush hiding alone (P4's second half) without ingest overlap.** ~8% on its own; only worth doing bundled inside B4's FSM restructure.
- **Multi-lane dividers (P1's aggressive variant).** Radix-4 gets prefill under the MXE line already; the multi-lane in-order-emission hazard buys risk during bring-up for headroom nothing downstream needs until P5-era prefill rates.
- **Host-side MMIO micro-optimization (batching pokes, faster driver).** It shaves the constant, not the O(T) scale-composite round-trips per Q·K̂ᵀ job — only on-tile scale composition (inside B1) removes the term.

*Contract discipline reminder for all of the above: `apex_pkg.sv` has survived v0.1→v0.2 untouched (APEX_VERSION 0x0001_0000). Tier A and B1/B2/B4 all preserve that. The only items that plausibly force a version bump are P2-as-descriptor-flag (avoid — use the CSR/route variant) and P6/D-025 (a deliberate, golden-gated amendment).*