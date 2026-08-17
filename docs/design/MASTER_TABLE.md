# THE MASTER TABLE v22 (2026-08-06) — THE E-7/E-8 FLIGHT: a defect, precisely caught

v22 records one flight with two outcomes. The good: the 10th image ingested
first-try, E-6 REPLICATED on it, and the whole gate stack (clock key, DDR
sha, refuse-loudly) did its job under fire. The hard: **walked attention
does not survive silicon** — E-7/E-8 fail on the card while green in sim,
and the differential pins the defect to one seam. Every v21 §1 row still
stands; nothing hardware-proven was retracted.

## v22 §1 — NEW hardware-proven rows (append to v21 §1)

| claim (exact scope) | model | image | evidence |
|---|---|---|---|
| E-6 REPLICATED on a third image: walk_oproj + walk_qkv_oproj re-flown, 896+2192 caps, 86 checks 0 fails, same resident weights full-verified before AND after | 0.5B | `agfi-04cfe164ba90b8ab0` | WALKED_ATTENTION_DEFECT.md, `e7_hw_differential.json` (2026-08-06) |
| WALKED QKV inside a faulting walk is still BIT-EXACT: E-8's 217 hw captures == sim's first 217 of 437 | 0.5B | `agfi-04cfe164ba90b8ab0` | `build/e8_prefix_diff.log`, same doc |
| The DEFECT, on the record: every walked chain WITH score+pv latches WALK_ERR_SEQ (status 0x3703) at the QKV→SCORE boundary; every walked chain without it passes | 0.5B | `agfi-04cfe164ba90b8ab0` | WALKED_ATTENTION_DEFECT.md §fault-localization |
| The defect's boundary, EXACT (second session, same day): 7 host-mode attention jobs (both KV engines) ALL PASS on this image while walk_e6 fails identically → the fault is exclusively the WALKER driving the composite scale-cache. Not the bank's arithmetic, not the walker's chains, not the fetch machinery | 0.5B | `agfi-04cfe164ba90b8ab0` | `hostctl_result.json`, WALKED_ATTENTION_DEFECT.md [RESOLVED block] |
| **E-7ng: the WALKER-FETCHED GAMMA NORM on silicon** — one kick, no attention: fuel-fed OPROJ + on-tile epilogue + RES1 + NFEED + NORM2 with gamma FETCHED from card DRAM; r1 AND h2 896-bit-exact; gamma-poison (one resident g2 byte, PHYSICAL DDR reload) RED by the exact predicted codes with r1 held; DRAM restored + full-verified | 0.5B | `agfi-04cfe164ba90b8ab0` | E7NG_ON_SILICON.md (2026-08-06) |

## v22 §2 — status deltas against v21

* **E-7 (fetched-gamma one-kick chain) and E-8 (QKV composed in): sim-green,
  SILICON-RED.** They move from v21 §2 "sim-only" to a sharper truth:
  sim-only *because they fail on the card*. Not reproduced in sim: tile_div
  {1,2,5,16,32} and +ddr_lat/+ddr_stall sweeps all pass; the twin provably
  builds the same GQA bank (INFO_TIER checked read).
* **v21 §4 "QKV + attention in one kick — refused" is LIFTED in RTL** (the
  E-8 descriptor walks both; the act-bank collision was resolved by the
  co-residency work in @a772713). Its silicon debut hit the defect above.
* **Image ledger:** `agfi-04cfe164ba90b8ab0` (apex-e7e8-20260806, Slack MET
  +0.319, A2 from the DCP's own manifest) is registered in
  `clock_key.IMAGE_RECIPE` and healthy — E-6 runs on it. **10 images, 10
  first-try ingestions.**
* **Gap register add (blocks E-7/E-8/full-walk on silicon):** the
  walked-attention defect — now bounded to ONE seam: the walker driving the
  composite scale-cache (host mode healthy, walker chains healthy, fetch
  machinery healthy — all silicon-proven on this image the same day). The
  first hypothesis (select-crossing on the GQA bank) was REFUTED by the
  failing programs' own descriptors (H=1: the select never moves). The xdc
  exception audit is clean — no composite path swallowed. Open diagnosis
  paths: sim testbench latency modeling around the walker↔composite
  handshake; a debug CSR splitting err_frame/err_stale.
* Cards: `i-0029f4cb32ec22847` and `i-0ec2487d966f900f0` both
  terminated+verified (~$4 total for both sessions). Only the unrelated
  `apex-f2-fpga` box remains (6+ days, owner's call).

---

# THE MASTER TABLE v21 (2026-08-05) — CLOSE THE BOOKS: the reconciled current truth

v21 is the reconciliation pass: every row names its committed evidence;
every prediction is LABELLED; every fence carries its measured reason.
Sections v20 and below are the historical ledger — where a line there has
been overtaken it now carries an inline **[superseded]** marker. When this
file and an older section disagree, v21 wins.

## 1. PROVEN ON HARDWARE (image + committed evidence per row)

All results below ran at 15.625 MHz (recipe A2) — a **correctness clock**.
No row is a throughput claim. All result docs are in
`docs/results/prompt_on_chip/` unless pathed.

| claim (exact scope) | model | image (AGFI, all verified `available` 2026-08-05) | evidence |
|---|---|---|---|
| One attention op of a real 7B decode step, produce-mode, raw INT32 returned + host epilogue, token ' Paris' identical | 7B | `agfi-0ae06ea568e5667ba` | RESULT.md, `hw_result.json` (2026-07-29) |
| Projection GEMM family EXHAUSTIVE: 1024/1024 blocks of Wq/Wk/Wv/Wo (layer 0, full K=3584), raw INT32 bit-exact | 7B | `agfi-0ae06ea568e5667ba` | PROJ_SWEEP_RESULT.md, `proj_sweep_result.json` (2026-07-30) |
| Attention breadth: 28/28 heads of one real decode step, produce-mode bit-exact | 7B | `agfi-0ae06ea568e5667ba` | `hw_breadth_result.json` (fa4707c, 2026-07-30) |
| Batched transport valid on silicon: capture streams batched == separate; 4.5× @N=8 measured | 7B | `agfi-0ae06ea568e5667ba` | BATCHING_STUDY.md §e (2026-07-30) |
| RoPE 28/28 heads + SwiGLU 18,944/18,944 cols at TRUE 7B geometry, produce-mode bit-exact | 7B | `agfi-0cc7aa798fe3abce2` | NARROW_LANE_RESULT.md, `flight_result_hw.json` (2026-07-31) |
| ALL SIX op types of a real 7B decoder layer, proven INDIVIDUALLY at full width (residual 3,584/3,584 ×2 reassembled; RMSNorm-2 full row, 0 host arithmetic in the norm2m arm) | 7B | `agfi-09e947873048a6877` | SIX_OF_SIX_RESULT.md, `flight_result_hw_r4.json`, `flight_result_hw_norm2m.json` (2026-08-01) |
| C2: EVERY op type of layer 0 served by the tile during a real prompt, token identical; ×0.5 poison CHANGES the token | 7B | `agfi-09e947873048a6877` | C2_PROMPT_ALL_OPS_RESULT.md (2026-08-01). Limits in the claim: projections sampled 56/53,760 in that run (family exhaustive above); RMSNorm-2/SwiGLU/RoPE re-enter via the C-1 view, deltas printed; 27/28 layers on host |
| E-lane: the SEQUENCER armed and completed an in-tile residual→C-1→norm chain, host silent during the walk; walk_off discriminator RED | toy 128 | `agfi-006b1314fcbbb3505` | ELANE_WALKED_CHAIN_RESULT.md (2026-08-02). ARCHITECTURE claim — never a model claim |
| 0.5B 2-layer prompt flight on the b64 image: 4,380 programs bit-exact, INFO_D audited 4,380/4,380, token identical | 0.5B | `agfi-0ecab46b8a8376b21` | `docs/results/p2_multilayer/p05b_hw_check2.log` (2026-08-03) |
| The measured SPEED LADDER (§5): 578 → 194 → 133 s/layer, token identical at every rung | 0.5B | `agfi-0ecab46b8a8376b21` | `docs/results/p2_multilayer/p05b_{hw_check2,fat_hw2,collapse_hw}.log` + TRANSPORT_COLLAPSE.md (2026-08-03/04) |
| DDR bring-up: AWS's real DDR4 trains under our CL; 28.5 MiB real 0.5B weights loaded at 101 MB/s, FULL readback 466,288/466,288 exact | 0.5B | `agfi-0a345ddb51285e847` | DDR_BRINGUP_RESULT.md (2026-08-05). Train+load+hold only — compute-from-DDR is the next two rows |
| SELF-RUNNING CARD: the walker issued DDR fetches from its own tensor table, fuel streamed weights from card DRAM, MXE computed 144/144 QKV blocks bit-exact, host silent in the walk window; walk-off RED; one poisoned DDR byte moves lane0 by the exact predicted +1 | 0.5B | `agfi-0183a4b88c8d21163` | SELF_RUNNING_CARD_RESULT.md, `walkfuel_hw.cap.jsonl` (2026-08-05) |
| E-6 ON FPGA: ONE descriptor — walker-fetched weights, QKV + OPROJ + requant epilogue ON-TILE → deq → residual; r1 896/896 BIT-EXACT; walk-off RED; RQ-calibration perturbation moves 801/896 | 0.5B | `agfi-0bc20880b50f5faba` | E6_ON_SILICON.md, `walk_oproj.hw.cap.jsonl`, `walk_qkv_oproj.hw.cap.jsonl` (2026-08-05) |

## 2. PROVEN IN SIMULATION ONLY (true, but no card has run it)

* **The walked-step matrix** — 10/10 probes on the verilated b64 twin; the
  park/refusal classes per step. `docs/results/elane_walk_steps/STEP_MATRIX.md`.
* **E-3b/E-4/E-4b toy chains as flights** — walker-staged q (896/896 + 14/14
  scales at 0.5B geometry) and the one-kick {QSTAGE, SCORE, PV, NFEED, NSRC}
  chain. The RTL is IN the flown convergence images, but these specific
  chains have not themselves been flown; the flown walks are E-5/E-6's.
* **E-6 walk-window cycle counts** (2,448,960 / 5,596,970 sim cycles;
  ~2.19k tile cyc per fuel-fed job; ~3.3 weight B/tile-cycle) — measured in
  sim at two clock ratios by two independent drivers agreeing to the cycle
  (WALKED_EPILOGUE_E6.md + \_REPLICATION.md). The BIT-EXACTNESS flew; the
  cycle numbers did not.
* **IB-FUEL 18-job D=128 fuel replay** — 27,996 checks ×3 ratios from
  (behavioral) DDR, 3/3 mutants RED (`IB_FUEL.md` §1.5). The from-DIMM
  silicon replay of that exact set has NEVER run (the DDR=1 silicon images
  are b64/D=64; INFO_D refuses the D=128 set by design).
* **Transport-collapse equivalence** — collapse vs no-collapse field-by-field
  identical result JSONs (sim A/B, TRANSPORT_COLLAPSE.md §3); the hw arm
  measured only the collapse shape (133 s/layer, §5).
* The entire verification stack (l3/l4/unit/mutant suites, capgate, fuel
  mutants) — simulation by nature; listed once, not re-claimed per row.

## 3. PREDICTIONS — labelled, sim-derived, hardware-unvalidated

* **~70 ms/layer at a 62.5 MHz tile ⇒ ~1.7 s/token (0.5B, 24 layers)** for a
  FULLY-walked layer at the measured sim ingest (3.3 B/cyc); floor
  ~0.35 s/token at the 8 B/cyc peak; 4× those walls at the flown
  15.625 MHz. Requires the five un-walked steps (§4) to land first.
  WALKED_EPILOGUE_E6.md (the 2× tile-clock conversion error is already
  corrected there — receipts pass 2347ab6/a85ea8e).
* **~160–760× per-token IF the whole layer walks** — same source, same
  caveats; today's walked coverage leaves the token wall host-FFN-bound.
* **A0 (62.5 MHz) image closes timing** — structural analysis only, no build
  exists (CLOCK_LADDER.md; ≤~1.6% end-to-end gain at TODAY'S dispatch — the
  clock pays only in the walker/fuel regime).
* **0.89 h per 0.5B token at 24 layers offloaded** — extrapolation printed in
  the collapse log, NOT run.
* **A retired prediction, kept for the record:** the collapse transport model
  predicted ~40–45 s/layer; the card measured **133.4** — direction right
  (194→133), magnitude wrong ~3×; model retired below ~10 invocations/layer
  (TRANSPORT_COLLAPSE.md §4b).

## 4. FENCED — each with its measured reason (loud refusal, never a wedge)

* **NORM1 / NORM2 gamma fetch under fuel** — no xw→xg route exists; with fuel
  armed a gamma fetch would poison the weight stream. S2_CHECK refusal.
* **FFN gate/up + SWIGLU walked** — the walker's all-gate-then-all-up
  template order starves `asu_swiglu`'s per-64-frame phase alternation
  (apex_top.sv:1548 class). Refused.
* **DOWN walked** — mask-inseparable from FFN; k=4864 decomposition legal
  since ee393be but per-k-chunk act re-staging is not carried (S2_PAD note).
* **RES2 walked** — starved without DOWN.
* **QSTAGE under FPROJ** — fuel_src=1 disconnects the mailbox xw stream its
  k2 injection rides (cl_apex.sv:1074). Refused at S2_CHECK.
* **QKV + attention in one kick** — act-bank row collision. Refused.
* **attention-o8 → OPROJ-act seam** — a host-staged copy TODAY, measured by
  the V-flip discriminator (moves the walked o8, not r1). Named follow-on.
* **T ≤ 128 on 7B attention offload** — the mailbox exposes no per-chunk
  softmax state; chunked heads cannot be offloaded. In-script fence.
* **B-FEED-WIDTH** — at D_model=3584 the C-1 feeder chunks rows as 28×128
  with per-chunk scales (not golden's form); RMSNorm-2/SwiGLU/RoPE re-enter
  golden through that view in C2-class runs, deltas printed. Wide feeder
  unbuilt (IB_LAYER stage 6).
* **Fuel-mode drop MID-walk still wedges** — the S2_CHECK fence is
  pre-entry only; no non-retracting abort / timeout exists (STEP_MATRIX E-5
  note).
* **Mislabeled AFI** — `apex-b64-05b-20260803` (afi-09e68d25f4eefb3d7) is
  actually D=128 (25ddb66); never fly anything without the on-card INFO_D
  identity read.

## 5. THE MEASURED SPEED LADDER, and the honest per-token number

Same 2 layers (0.5B layers 0–1), same image (`agfi-0ecab46b8a8376b21`), same
prompt, token identical at every rung. Per-layer wall = emit + audit +
execute + grade. Logs committed in `docs/results/p2_multilayer/`.

| rung | s/layer (median) | date | log |
|---|---|---|---|
| thin per-op dispatch | **578.5** | 2026-08-03 | `p05b_hw_check2.log` |
| fat descriptors + push bursts + ssh mux (3.0×) | **193.7** | 2026-08-04 | `p05b_fat_hw2.log` |
| + invocation collapse 44→5 (4.35× total) | **133.4** | 2026-08-04 | `p05b_collapse_hw.log` |

**The honest per-token number today (measured):** one 0.5B token with layers
0–1 offloaded (8.33% of the step's MACs, 4 exact-egress families) costs
**270 s wall (~4.5 min)** vs **4 s** pure-host — the tile currently makes
the token ~67× SLOWER, and the claim is correctness + provenance, never
speed. The pre-ladder baseline at the same setting was 1,160 s (19.3 min,
2026-08-03). There is NO tokens/sec claim anywhere in this project; every
flight ran the 15.625 MHz correctness clock.

## 6. EXTERNAL — aws-fpga#799 (state checked 2026-08-05, `gh issue view`)

**OPEN.** One vendor comment (mjthimm/AWS, 2026-08-03): *working with AMD on
it, will keep us updated* — an acknowledgment, NOT a fix, no ETA. The v16
line "stays filed and unanswered" is superseded: it is now filed and
ACKNOWLEDGED. Consequence unchanged since 2026-08-01: #799 gates only the
WIDE image (one-pass full-row norm, unsliced residual — an optimization)
and the future wide-feeder track. **Nothing on the current critical path
waits on the vendor.**

## 7. CLAIM SHAPES — sentences the owner can say today, and sentences they cannot

**CAN say (each maps to a §1 row):**

* "On an FPGA, our layer sequencer fetched real Qwen2.5-0.5B weights from
  the card's own DRAM, computed the Q/K/V and output projections, applied
  the requant epilogue on-tile and fed the product into the residual — bit-
  exact against our golden model, with the host writing nothing during the
  walk."
* "Every one of the six compute op types of a real Qwen2.5-7B decoder layer
  has run through our blocks on an FPGA, at full model width, bit-exact, in
  produce mode."
* "For a real prompt, every op type of one 7B decoder layer was served by
  our blocks on an FPGA and the model emitted the identical next token —
  and scaling the tile's values changes the token, so the substitutions are
  load-bearing."
* "We built an end-to-end decoder-layer **design**, and 8 of its 13 steps
  execute under the sequencer against the real datapath, 3 of them
  fuel-fed under one descriptor on real hardware."
* "Our dispatch work took the measured per-layer wall from 578 s to 133 s,
  token-identical at every step."

**CANNOT say (with the boundary that stops each):**

* ~~"The model runs on our FPGA"~~ — 27/28 (7B) or 22/24 (0.5B) layers run
  on the host in every prompt-serving run. The supported shape is "part of
  the answer was computed on the FPGA."
* ~~"A full layer runs end to end on the FPGA"~~ — that is end-to-end
  **design**, not end-to-end **execution**: 5 of 13 steps are measured,
  fenced refusals (§4). "A walked subset of the layer" is the boundary.
* ~~"N tokens/sec"~~ or any throughput/speed-vs-GPU claim — every number was
  taken at a correctness clock; the only per-token walls measured make the
  tile slower (§5); the 1.7 s/token figure is a labelled prediction (§3).
* ~~"on our chip"~~ — there is no chip; say "**on an FPGA**". "Silicon" is
  house shorthand for the F2 card and must not appear in external claims
  without "FPGA" beside it.
* ~~mixing 0.5B and 7B~~ — the walked/DDR-resident claims are 0.5B ONLY;
  the 7B claims are host-staged per-op ONLY. "The self-running card ran the
  7B model's weights" is false in both halves.
* ~~toy-width chains as model claims~~ — the E-lane in-tile chains are
  toy-geometry ARCHITECTURE claims (128-wide flown; 64-wide b64 sim);
  never present them as Qwen results.
* ~~"AWS confirmed/fixed the shell defect"~~ — AWS acknowledged #799 and is
  working with AMD; nothing is confirmed or fixed (§6).
* ~~"the tile did the whole attention op including calibration"~~ — in the
  7B produce-mode runs the requant epilogue is host-applied golden code
  (standing fence); on-tile requant is proven only in the walked 0.5B
  OPROJ epilogue and walked-PV paths.

## 8. GAP REGISTER — everything open, measured blocker, honest size

| # | gap | measured blocker | honest size |
|---|---|---|---|
| G1 | Walk the five un-walked steps (NORM1/NORM2 gamma, FFN gate/up+SWIGLU, DOWN, RES2) | §4 fences: no xw→xg route; template order vs swiglu phase alternation; per-k-chunk act re-staging absent; RES2 starved | The era-2 long pole. SWIGLU order + DOWN re-staging are walker-FSM redesigns, not flips: **~1–2 weeks** at the house evidence bar; gamma route is RTL + a route fence audit (**~days**) |
| G2 | In-tile o8 → OPROJ-act re-staging (kill the host-staged seam) | V-flip discriminator measures the copy (E-6) | Named follow-on; **~days** (S2_PAE-class delta + discs) |
| G3 | Fully-walked layer → walked whole-token run (0.5B) | G1+G2; then per-token prediction (§3) becomes testable | After G1/G2: **~days** (driver + card session) |
| G4 | A0 62.5 MHz image | No 16 ns build exists; 5 hardcoded host 15.625/A2 points (CLOCK_LADDER §7.2) | **~1 devbox build + 1 card session + host image-keying**; worthless before G1-G3, ~4× after |
| G5 | Wide feeder (B-FEED-WIDTH, D=3584 one-scale rows) | seam_feeder_quant refuses D∉{64,128}; golden-form parity broken at 3584 (28 chunk scales) | **Weeks** (IB_LAYER stage 6) — and its image is then gated by #799 |
| G6 | #799 vendor fix (wide image ingestion) | Deterministic place_design defect at ~1.3M-cell scale; AWS+AMD investigating | **External, unbounded**; optimization-only since 2026-08-01 |
| G7 | Fuel mid-walk abort/timeout | Mode drop mid-walk wedges (fence is pre-entry only) | **~days** (non-retracting abort or watchdog + red tests) |
| G8 | IB-FUEL s5-as-written: 18-job D=128 from-DIMM silicon replay | Never run; needs a D=128 DDR=1 image (none exists) | **~1 build + card session** if ever wanted; the b64 flights already carry the compute-from-DDR claim |
| G9 | Walked KV record DATA path (STOREKV walks addressing only; record data via host squant pre-walk) | STEP_MATRIX STOREKV row | Multi-token decode prerequisite; unsized — needs a design pass first |
| G10 | E-3b/E-4b chains flown as flights | Sim-proven; RTL aboard the flown images; no card record of walk_qstage/walk_e4 themselves | **~1 card session** (drivers exist) |
| G11 | Walked-norm div-2 sim record not preserved (doc-claimed only) | CLOCK_LADDER §3.2 caveat | **Minutes** (local re-run, free) |
| G12 | tip/ lane | Parked since v19; never driven on hardware | Out of scope until a TIP claim is wanted |
| G13 | 7B walked/DDR anything | Everything walked/resident is 0.5B/D=64; 7B weights (~233 MB/layer INT8) never touched the fuel path | Gated behind G5/G6 + a wide image; do not schedule |

## Historical ledger below — v20 and earlier, kept verbatim with [superseded] markers

---

# THE MASTER TABLE v20 (2026-08-04) — THE ALL-IN SPRINT: SPEED + THE SELF-RUNNING CARD

## v20: owner went ALL-IN on era-1 (dispatch) + era-2 (walker + resident weights)
Baseline reality check first: one 0.5B token at the 8% offload setting cost
19.3 MINUTES (measured 2026-08-04, ' Paris' identical, discriminators green).
**[receipts correction, v21: the baseline log is dated 2026-08-03 —
`docs/results/p2_multilayer/p05b_hw_check2.log`, 1,160 s. The post-ladder
measured token is 270 s (~4.5 min), v21 §5.]**

**ERA-1 — MEASURED LADDER (same 2 layers, same image, same prompt):**
578 s/layer (baseline) -> 194 (fat descriptors + push bursts + ssh mux, 3.0x)
-> **133 s/layer (invocation collapse 44->5, 4.35x total)**. Token identical
at every rung. Remaining profile: executor per-program overhead + host
emit/grade — era-2 kills these as categories. Clock raise PARKED by Amdahl
(4x clock = <=1.6% today; CLOCK_LADDER.md has the ready A0 recipe).

**ERA-2 STATE:**
* W1 weights-resident: IB-FUEL green at 0.5B/D=64; real DDR-fed projection
  GEMMs BIT-EXACT with one-byte poison RED (2ec9b2b); weight packer + BAR4
  loader built. **First DDR=1 Vivado build FAILED in AWS's encrypted sh_ddr
  (Synth 8-5809 x3)** — the pre-named #1 risk; debug agent on it with devbox
  build access. If unfixable by configuration: era-2 weight residency
  re-routes (HBM / kit fix), and we need to know.
  **[superseded 2026-08-05: FIXED — the DDR=1 b64 build is TIMING MET +
  PRV-GREEN (89c280d) and THREE DDR=1 images flew: DDR bring-up
  agfi-0a345ddb51285e847, self-running card agfi-0183a4b88c8d21163, E-6
  agfi-0bc20880b50f5faba. See v21 §1.]**
* W2 walked layer: step matrix MEASURED (10/10 probes as predicted).
  E-3b CLOSED — the walker stages its own q rows, 0.5B geometry, all 14
  heads 896/896 bit-exact. E-4 first chain: **6 steps walked, 4 with the
  activation never leaving the tile** (one descriptor, two kicks).
  Fetch-gated steps (QKV/OPROJ/FFN) all park on the FUEL path == W1's
  deliverable — the two era-2 lanes converge on the DDR image.
* W2b single-kick fusion (walker-drivable l_nsrc): agent running.
  **[superseded: LANDED 2026-08-04 as E-4b (W2_MASK[13]); see
  E2E_TOY_LANE.md §4.]**
NEXT IMAGE after fusion lands: ONE b64 AFI with E-3b + fusion (+ DDR when
the synth fix lands) — then the walked chain flies on real silicon.
**[superseded: BUILT AND FLOWN 2026-08-05 — apex-convergence /
apex-convergence2; the walked fuel chains ran on real hardware. v21 §1.]**



## v19: E-lane FLOWN — every block you built has now run on an FPGA
`agfi-006b1314fcbbb3505` (image #3, E1+E2+E3a @ 47a5a3a): the WALKER armed and
completed an in-tile residual -> C-1 -> RMSNorm chain **on the FPGA**, 274
captures, bit-exact vs golden, 5/5 discriminators caught. The load-bearing one
is `walk_off`: same program, walker step-enable bit cleared -> the run FAILS,
so the sequencer really is driving. During the walk the host does NOTHING.
Evidence: `ELANE_WALKED_CHAIN_RESULT.md`.
THIS CLOSES the last unexercised block: seq/ (the walker) had never driven
real hardware before today. tip/ remains unexercised (parked lane).
SCOPE: 128-wide TOY config — an ARCHITECTURE claim, NOT a Qwen-7B claim, and
the chain is residual->norm, not the whole layer. E-3b (walker stages its own
q rows) and E-4 (synthetic full layer) remain open.



## v18: C2 landed — the prompt claim now covers ALL SIX op types
`layer_offload.py` (2e78dc3) rebinds golden's per-op globals so EVERY op type
of decoder layer 0 is served by the tile during a real "The capital of France
is" run. **On the FPGA** (`agfi-09e947873048a6877`, 136 programs, 43,470
captures): 6/6 op types served, token ' Paris' IDENTICAL to pure-host, and the
poison discriminator (tile values x0.5) CHANGES the token — the substitutions
are load-bearing. Evidence: `C2_PROMPT_ALL_OPS_RESULT.md`.
LIMITS IN THE CLAIM: projections SAMPLED (56/53,760 — though that op type is
separately proven 1024/1024 on the FPGA); RMSNorm-2/SwiGLU/RoPE re-enter as
C-1 reconstructions (B-FEED-WIDTH), deltas printed not hidden; 27 of 28
layers on the host. NOT "the model runs on the FPGA".

## v18 also: E-1/E-2 CLOSED IN RTL (7c0d4a5) — ops now chain ON-CHIP
The residual's row gained an internal egress port and asu_rmsnorm's x port
gained an internal source mux + LAYER unit-3 job (new err_code 9). An
activation goes residual -> C-1 -> norm WITHOUT leaving the tile, bit-exact
vs golden at tile_div 5 AND 2; RED on the pre-change twin three ways
(incl. a 2.1M-cycle stall — the norm waiting on a value with no path).
Demonstrator `scripts/fpga/f2/elane_norm_feed.py`. Also repaired two gates
that were ALREADY broken at HEAD (resfx TB missing jb_base since R3; its
mutation site stale) — now 4/4 mutants detected. NOT YET FLOWN: needs a new
image (E-lane image #3). E-3 (walker arming the new route) still open.

## v18 also: ZERO HOST ARITHMETIC in the full-row norm (951e488)
norm2m flown on the FPGA: the MXE accumulates the whole K-split on-tile
(sum2=120498), host moves ONE u32 verbatim (proven from the emitted
program). 94/94 stages green.



## v16 HEADLINE: all 6 op types now run at FULL 7B WIDTH on a NARROW build
The v9-v15 premise "the last 2 op types need a wide image, so we wait for
aws-fpga#799" is DEAD. Both were dissolved without the vendor:

* **Residual** — it is ELEMENTWISE, so the full 3584 row is 28 aligned
  128-element slices, each BIT-IDENTICAL to what one wide pass would produce.
  No RTL change at all. Proven 28/28 + 28/28, reassembled rows == golden
  (3c7757c, resid_full_check.py), then wired into the flight with a
  reassembly grade built from CAPTURED words (3ed6ca4).
* **RMSNorm-2** — golden's OWN `rmsnorm_fx_wide` is a chunked composition
  (chunk sums -> INT32 accumulate -> mu-scalar mean -> unchanged per-element
  datapath with ONE broadcast r). R4 (d444b6e) implements exactly that: the
  tile exports each chunk's sum2 and accepts an external SUM2+k, so mu,
  rsqrt and ALL 3584 per-element ops run ON THE TILE. Host arithmetic left:
  **27 INT32 adds** — and even those are removable, because the MXE computes
  the same sum2 as an x.x dot product (**120498**, bit-exact, two
  independent on-tile sources agree — 3ed6ca4, sum2_mxe.py).

**#799 is now an OPTIMIZATION, not a gate**: a wide image would do the norm
in one pass instead of 28 and the residual without slicing. Nice; not
load-bearing. It stays filed and unanswered. **[superseded 2026-08-03: AWS
responded — working with AMD, no fix, no ETA. Still OPEN; v21 §6.]**



## v13 CORRECTION (owner-prompted, verified by execution)
The v9-v12 framing "everything remaining waits on the vendor" was WRONG.
The flying image (05efb2a) predates ALL FOUR layer units — grep proof: 0
hits for rope_row/asu_swiglu/apex_residual/apex_layer_deq in its apex_top.
RoPE (head_dim=128) and SwiGLU (64-col chunks) are FULL-FIDELITY at narrow
width — proven on the narrow twin (rope PASS, swiglu PASS, probe PASS) —
and narrow images ingest 3-for-3. Only the two FULL-ROW ops (RMSNorm-2,
residual over 3584) truly need the wide image. New N-lane below.

## N-LANE (ours, TODAY): narrow image N1 -> silicon 4/6
| # | step | status |
|---|---|---|
| N1 | Narrow DCP @ 9b993bf (D=128 GQA=1 DM=128 DDR=0, A2, stock kit): LAYER window + R1+R2+R3 | ✅ **BUILT** 2026-07-31 — 54 min, routed, 96.9 MB tarball `2026_07_31-183025.Developer_CL.tar`. Devbox stopped, lock released |
| N2a | Local pr_verify vs the shell BB | ✅ **PRV-GREEN** — 0x HDPRVerify-41, 1x HDPRVerify-42 (locked-static, expected). **Independent confirmation for #799: the LAYER window's added area does NOT cross the threshold — the defect is bound to the WIDE config, exactly as filed** |
| N2b | AFI ingestion | ✅ **AVAILABLE first try** — **SUBMITTED** `afi-0408bd9c4fbf4a45c` / `agfi-0cc7aa798fe3abce2` (name apex-narrow-layerwindow-20260731, logs -> s3 afi-logs/narrow_20260731/). First APEX image EVER to contain rope_row/asu_swiglu/apex_layer_deq/apex_residual |
| N3 | Parity twin = the HEAD twin (obj_d128_ddr0 @ 9b993bf, capgate green) | ✅ already exists |
| N4 | Flight on silicon | ✅ **34/34 GREEN ON SILICON, 49.5 s** — RoPE 28/28 heads + SwiGLU 18,944/18,944 cols (FULL d_ffn) at REAL 7B fidelity; resid r1/r2 + norm2 as 128-wide UNIT demos; R3 refusal guard verified on the FPGA. ~25,900 captures, produce-mode, 0 baked expectations. Canonical 18-job regression on the new image: **18/18**. 2 driver bugs caught by refuse-loudly gates (missing remote attach; 80 MB single-batch upload), 0 integrity impact |
| R4 | Chunked wide RMSNorm-2 on a NARROW build (asu_rmsnorm ext SUM2+k; CSR 0x90/0x94; err_code 8) | ✅ **VERIFIED in sim** (d444b6e) | RED on pre-R4 twin: probe+norm2c poll-stall, norm2x graded FAIL 719/3584 (per-chunk LOCAL r — the wrong answer the arm prevents). GREEN: norm2c 3584/3584 codes + 28/28 scales + 28/28 chunk sums, armed total 120498; 5/5 discriminators; capgate 505/505; walker 24,789 checks 0 err; defaults byte-identical with feature off |
| R5 | Elementwise-residual workaround (no RTL) + MXE sum-of-squares | ✅ **VERIFIED in sim** (3c7757c, 3ed6ca4) | resid 28/28 + 28/28 bit-exact, reassembly == golden; MXE sum2 = 120498 == golden chunked sum, discriminators kill unsigned-multiply and wrong-operand variants. Also caught a REAL bug: `off=` was shadowed by a loop var, so all 28 slices recorded off=0 and would have graded against golden's FIRST 128 elements — silent wrong answer, caught at gen time, regression added |
| N6 | Narrow image #2 (R4) -> fly all 93 stages | ✅ **93/93 GREEN ON THE FPGA, 66.4 s** (`agfi-09e947873048a6877`); full-row residual reassembly 2/2 == golden; norm2c 4088 caps + norm2x 4060 caps; 41,237 captures, 0 baked expectations; canonical 18-job regression **18/18**; box terminated+verified, ~$2. Evidence: SIX_OF_SIX_RESULT.md. WAS: **INGESTING** — built (96.6 MB), **PRV-GREEN** (0x HDPRVerify-41), submitted `afi-093093be6d909aedc` / `agfi-09e947873048a6877`. Flight set **93/93 GREEN IN SIM** (a5dcfbc) incl. full-row residual reassembly 2/2 and norm2c/norm2x/probe. Second clean pr_verify on a logic-carrying narrow image = another independent #799 datapoint |
| N5 | Evidence + claim | ✅ **LANDED** — `docs/results/prompt_on_chip/NARROW_LANE_RESULT.md` + flight_result_{hw,sim}.json. Box terminated+verified, ~$2 |


## E-LANE (owner-approved 2026-07-31, scoped, NOT started) — a COMPLETE layer end-to-end, toy width
Full scope: `docs/design/E2E_TOY_LANE.md`. Needs NO vendor unblock. Claim is
an ARCHITECTURE claim (synthetic 128-dim 1-head layer, golden-arbitrated),
never a 7B claim. Audit 2026-07-31 found 4 of 5 historical blockers CLOSED
(F1, GAP A, GAP B-as-RTL, GAP C) and — the big one — **the walker ALREADY
sequences all 11 layer steps at true 7B geometry, bit-exact vs golden**
(emission-level; consumers stubbed). Remaining: E-1 residual internal
egress, E-2 rmsnorm job port + internal input (today its input is a
TOP-LEVEL PORT, host-streamed — found independently twice), E-3
walker-driven q staging, E-4 synthetic subject + golden arbiter.

## HOW FAR ARE WE (read this first)

```
  op types of a decoder layer, proven through OUR blocks:
    in SIMULATION   ████████████████████  6/6  attention · projections · RoPE · residual · RMSNorm · SwiGLU
    on the FPGA     ████████████████████  6/6  ALL SIX, at full 7B width, bit-exact
        attention (28/28 heads) · projections (1024/1024 blocks) · RoPE (28/28 heads)
        · SwiGLU (18,944/18,944 cols) · residual (3,584/3,584 x2) · RMSNorm-2 (3,584/3,584)
    achieved with NO wide image and NO vendor unblock (2026-08-01)
    NOTE: 6/6 = every op type proven INDIVIDUALLY. A layer running END TO END (ops feeding
    each other on-chip, no host staging) is the separate E-lane — see E2E_TOY_LANE.md.
  T10(B)  "attention + projections of a real 7B step on our silicon"  ✅ CLAIMABLE TODAY
  T10(A)  "every op of a real 7B layer on our silicon"                ⛔ needs ONE wide image
  distance to T10(A) = 1 external unblock (vendor — escalation FILED 2026-07-30,
  aws/aws-fpga#799) + ~1 day of our work that is ALREADY WRITTEN
```
**[superseded: the T10(A) ⛔ above was dissolved WITHOUT the wide image —
6/6 op types flew 2026-08-01 (SIX_OF_SIX) and C2 served every op type of
layer 0 during a real prompt the same day. The wide image remains only an
optimization (#799, v21 §6). The two lines are kept as the ladder's
historical shape.]**


**End state, re-scoped per the challenge into two honest claims:**
- **T10(A)** — *"every compute op of a real 7B decoder layer ran through our
  blocks on an FPGA"*: ONE layer, ONE decode step. The achievable "everything"
  claim. Gated on E1.
- **T10(B)** — *"the attention + projection op families of a real 7B decode
  ran through our blocks, produce-mode, bit-exact, token-identical"*:
  **already true**; breadth-extension (all 28 heads, Wq/Wk/Wv/Wo) is host
  software on the LIVE image, this week.
- T10-as-literally-worded (28 layers × 128 tokens, all ops) is arithmetically
  not a demo at host-driven job rates (~hours per decode step) — it returns
  only with the throughput track (DDR, walker steady-state), post-(A).

| # | Item | Status | Evidence / blocker |
|---|---|---|---|
| E1 | Place-only + pr_verify at GQA=1 / DM=3584 / DDR=0 | ✅ **VERIFIED — RED** (2026-07-30) | Same two slices at ONE engine. Fifth variable exonerated (recipe TCL, adjacency, directive, GQA count; version never varied). Threshold sits between DM=128 and DM=3584. Evidence: devbox e1_*.log, ANALYSIS.md @ e22c310 |
| A1-A3 | Route/ingest/bring-up of ANY wide image | ⏳ **ESCALATION FILED — waiting on vendor** | Filed 2026-07-30 (owner-authorized in-session) as public issue **aws/aws-fpga#799** — https://github.com/aws/aws-fpga/issues/799 — byte-level delta, 7-row exoneration matrix, stage bisection, repro, 3 asks. Publish-hygiene pass applied (account id / S3 bucket / internal paths stripped; artifacts offered via private channel). AWS support-case route confirmed CLOSED on this account (Basic plan; `SubscriptionRequiredException`) — paid upgrade would be needed for a formal case. Next: monitor the thread, hand over artifacts on request. **[update 2026-08-03: AWS replied on the thread — working with AMD, no fix/ETA; v21 §6]** |
| B1 | Fix SwiGLU cols truncation | ✅ **VERIFIED** (7faad3a) | Silent-wrong-answer defect: 7B tail chunk computed 4/2564 cols with NO error. Host fix + red/green regression + mutant gate; all selftests green. Walker2 l64/l128/l7b suites intentionally RED until R1. |
| R1 | ✅ **VERIFIED** RTL refusal fix (forced addition): jb_cols 12→7 narrowing must refuse not wrap (apex_top:1237 + apex_residual class) + walker2 per-unit chunk bounds (seq_walker_pkg:349) | ✅ fixed 681ca0e | oversized swiglu job now REFUSES (err code 2); walker2 per-unit bounds; l64/l128/l7b re-green with host expectations untouched. R1(c) residual: **no defect existed** — claim was wrong, nothing changed |
| B2-B5 | Layer-op emitter (gen_layer_ops.py) | ✅ **VERIFIED in sim** (247f49e) | RoPE 128/128 + resid-1/2 2040/2040 + wide RMSNorm-2 3584/3584 codes & 28/28 scales — all bit-exact vs the arbiter, produce-mode, 4/4 discriminators. RoPE needed only a HOST fix (q_sink + identity-GEMM recovery). |
| R2 | ✅ **VERIFIED** SwiGLU RTL defect (B-SWG-PHASE) FIXED — forced addition: gate/up skids latch the same beat (apex_top:1230/1231/1918 + stream_skid:81); up's `last` 2 beats early. No host workaround; red test in place | ✅ fixed 681ca0e — phase tracker steers each beat to the ACTIVE skid | swiglu bit-exact vs golden; probe flipped to regression-guard; **independently re-verified by the orchestrator on the wide twin: 6/6 stages PASS, 4,606 caps, 0 output expectations, 4/4 discriminators CAUGHT** |
| S1 | **Escalation package** — vendor-ready ESCALATION.md (7-row exoneration matrix, byte-level delta, repro, asks) | ✅ **VERIFIED + FILED** (7f29e23 → aws/aws-fpga#799) | Filed 2026-07-30; one honest gap flagged (2nd failed AFI id absent from all 8 local trees — omitted from the public issue, recoverable from account history if AWS asks) |
| S2 | **Batched execution** — batch_exec.py, marker-demux + per-file manifest | ✅ **VERIFIED ON SILICON** (0e0f8bf, 2026-07-30 evening) | MEASURED: attribution PASS on real silicon (8 produce-mode jobs, every capture identical batched vs separate); produce-mode class 3.63→0.82 s/job @ N=8 (**4.5×**), canonical class 5.31→1.67 (3.2×); marginal in-batch ~0.21 s/job → **~16× at scale**, NOT the projected 22-33× (caveat 3 fired, honestly recorded). BATCHING_STUDY.md §e |
| S3 | **Full 1024-block projection sweep, BATCHED** | ✅ **VERIFIED ON SILICON — 1024/1024 bit-exact** (2026-07-30) | ALL of Wq/Wk/Wv/Wo (8,192 output cols, full K=3584) of a real 7B decode step: 29,696 produce-mode jobs, 68/68 batches attributed, **36.0 min job stream = 0.073 s/job ≈ 50× vs separate** (unbatched truth ~30 h). C1 projections upgraded 0.78% → **100% exhaustive**. 3 transport incidents, all caught by refuse-loudly gates, 0 integrity impact. PROJ_SWEEP_RESULT.md + proj_sweep_result.json |
| R3 | ✅ **VERIFIED** Residual window base (B-RES-BASE lifted, ef04593) — LAYER_JOB[15:14] stride-1024 base in apex_residual/apex_top; footprint>DM refuses loudly (code 3); walker path pinned window-0 | ✅ fixed + red/green proven 2026-07-30 | RED: final emitter on pre-R3 twin (rebuilt from HEAD in a worktree) — probe rc=1 + resid 3113/3584 wrong. GREEN: 7/7 stages, resid_r1/r2 **3584/3584 FULL WIDTH**, 4/4 discriminators, resid_probe refusal guard. Regressions: capgate 505/505, walker2 l64/l128/l7b/refuse2 24,789 checks 0 errors. **T10(A) now footnote-free**; B-SER-FRAME stands per-job but no longer caps any claim |
| C1 | T10(B) breadth: whole attention stack of one real decode step | ✅ **VERIFIED ON SILICON** (fa4707c) **+ S3 exhaustive** | 28/28 heads + (now) **1024/1024 projection blocks** bit-exact, produce-mode. The attention+projection families of a real 7B decode step are proven on silicon with NO sampling caveat left. |
| T8' | T10(A): one layer, one decode step on silicon | ⛔ blocked-by-ESCALATION (+Lane B) | Lane B emitter still gets built and sim-proven against the HEAD twin — ready the day an image ingests. Date external. |

## Struck / cut / deferred by the challenge (with the receipts)

- ✂ **T1** — already built during the challenge itself (parity twin builds in
  **5 seconds**; estimate was wrong by 3 orders of magnitude). The real
  deliverable was the HYBRID recipe (05efb2a RTL × HEAD harness — a literal
  05efb2a twin can't build and has no capture egress) → shipped as
  `make paritytwin REF=<sha>`.
- ✂ **T2 (walker on silicon)** — CUT from the critical path: adds zero new op
  types; walked jobs hardwire requant_en=1 so they cannot run in the
  produce-mode the flagship claim rests on; consumes the serialized card.
  Stays as a nice-to-have behind T8'.
- ✂ **T3 (Vivado 2024.x)** — DEFERRED to last-resort: version was never the
  tested variable (all exonerations ran on 2025.2), the "everything-CL"
  already exists routed, and an older toolchain can ADD static deltas vs the
  BB. If ever run: one minor step (2025.1) before two majors.
- ✂ **T5** — free rider (k=2048 fix is in any tip-built image).
- ✂ **T7 (DDR)** — deferred AND re-blocked: not unlocked by A2 (the routed
  tarball is DDR=0; DDR needs its own build+ingestion), and it fixes the
  wrong term — measured: 0.78 ms of real MMIO inside a 3.8 s job wall;
  99.98% is per-invocation overhead. Throughput track, not demo track.
- ✂ **T9** — not a T10 precondition (a host that stages activations IS a
  data mover). Long track unchanged.
- ✂ **GQA-4** — cut from the T10-critical image entirely; it multiplies
  engines, not op types, and correlates 3/3 with the ingestion failure.

## Current critical path (post-filing)
**[superseded by v20/v21: the critical path moved OFF the vendor entirely —
era-1 dispatch + era-2 walker/resident-weights; see v21 §8 gap register.]**

**NOTHING INTERNAL REMAINS.** Every row is ✅ except the ones gated by the
vendor. The escalation IS FILED (aws/aws-fpga#799, 2026-07-30); the critical
path is now **the vendor's response** (or a tool/kit update). Our side of the
thread: monitor, answer triage questions, hand over preserved artifacts
(devbox EBS) on request. R3 is DONE (same day) — the full-width residual
rides the first wide image, footnote-free. The day a wide image ingests:
bring-up (~half day) → run the ALREADY-WRITTEN emitter on silicon (~1 day with
batching) → T10(A). Optional anytime: full 1024-block projection sweep (~$5),
first batched hardware run (validates the ~22-33x), walker on silicon.

## Reporting protocol (owner-refined 2026-07-30)

Unchanged: table delta on every answer; full rewrite on any verified row
change; additions only when forced; every rewrite names the critical-path row.
Status vocabulary: ✅ VERIFIED · 🔨 in-progress · ⏳ waiting-on-X ·
⛔ blocked-by-X · ✂ struck.

## Standing rules

One flying FPGA at a time · every silicon run validated against the
**hybrid** parity twin of the exact flying tree (`make paritytwin REF=<sha>`,
AGFI→commit map recorded) · clock frequency parse-verified before any job ·
canonical 18-job regops stay byte-identical · ingestion submissions always
carry --logs-storage-location.

## §perf addendum (2026-08-08): the ≥4 tok/s target (owner-set)

Budget at 0.5B: ~494 MB INT8 weights/token ⇒ 4 tok/s needs ~2 GB/s
sustained tile ingest. Ladder: full walk (~100× over host-dispatch) × A0
4× × ingest 3.3→≥32 B/cyc × W4 (2×, the FROZEN B3 recipe: G=32 direct,
−1.0 pt) ⇒ 4–8 tok/s window. Without W4 the target is marginal (~2), so
the target PROMOTES two lanes: (a) sustained-ingest engineering on the
fuel path, (b) the W4 weight datapath on the FPGA. Compute clears (~8
tok/s at A0). All numbers PREDICTION-labeled until the token loop
measures them.
