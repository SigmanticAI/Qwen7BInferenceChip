# CAMPAIGN.md — tinyNPU public launch: assets, claim discipline, gates

> **DRAFT — the launch has not happened.** This file is the plan for a single
> coordinated public drop and the claim-discipline rules every public word
> must follow. It is written to be publishable itself: transparency about
> *how* we make claims is part of the launch. Nothing ships until every gate
> in §5 passes.

## 1. The drop, in one paragraph

One coordinated release: a refreshed [`README.md`](../../README.md), the
launch post ([`LAUNCH_POST.md`](LAUNCH_POST.md)), and the already-committed
evidence they point at — the machine-generated verification roll-up
([`STATUS.md`](../../STATUS.md)), the honest compression-accounting gate, the
reproducible eval matrices (0.5B/1.5B and 7B), the ECP5 bitstream, the
paper-architecture spec ([`docs/spec/`](../spec/APEX7B_SPEC.md)), and the
calibrated performance model
([`docs/results/perf_model/`](../results/perf_model/PERF_MODEL.md)). Four
pillars, equal weight: **verification depth · a real FPGA artifact · honest
accounting · reproducibility**.

## 2. Messaging pillars

1. **Verification depth.** Every shipped RTL block is bit-exact vs an
   executable golden arbiter: 8,748,634 KVQ arithmetic checks / 0 fails,
   135,241 attention-tile checks, exhaustive SiLU 65,536/65,536,
   mutation-tested TBs, machine-generated evidence page under an
   anti-fabrication rule. Source: `STATUS.md` plus the committed per-suite
   run logs it parses (the 135,241 L3 total lives in the tee'd suite log
   under `verif/top/l3/`).
2. **A real FPGA artifact.** The KV-compression engine is placed and routed
   on a real FPGA today (ECP5-85F, bitstream in-repo). Source:
   `docs/results/kvq_engine_ecp5-85f.bit` + report — both regenerated
   2026-07-18 from the shipping-config seed-2 route (gate G1 ✅). Fmax is
   stated only from `docs/results/s10_fmax/RESULT.md`: 34.11 MHz, disclosed
   never led with.
3. **Honest accounting.** 3.16–3.51× whole-KV compression realized in actual
   RTL storage with every overhead counted (tag, outlier lanes, padding,
   scale bank), asserted in the golden model on every full-suite run
   (`make -C golden test`), bit-exact in
   RTL; the zero-overhead ceiling of this method family — ours and every
   "~4 bits/value" claim — is 3.66–3.88×. Source:
   `golden/tests/test_effective_bits.py`.
4. **Reproducibility.** Accuracy measured through a bit-exact-certified twin
   of the golden codec, re-runnable from a clone; a performance model that
   prints its assumptions and asserts its calibration on every `--check`
   run; every claim traceable to a committed artifact.

## 3. Approved claims (the only external wording)

| claim | exact wording | source artifact |
|---|---|---|
| Scope | "verified decoder-layer and KV-compression RTL of an architecture sized for 7B-class models (head_dim=128 verified), demonstrated on a real 7B model, with a calibrated projected performance model" | STATUS.md · docs/results/ |
| Verification | the pillar-1 sentence above, counts verbatim | STATUS.md |
| FPGA | "our KV-compression engine is placed and routed on a real FPGA today (ECP5-85F, bitstream in-repo)" | docs/results/kvq_engine_ecp5-85f.bit |
| FPGA (F2) | "the full tile builds clean for AWS F2 (Vivado, VU47P): zero-error synth + P&R, post-route timing met at the 250 MHz shell clock, and the image loads on a real F2 instance and answers its verified CSRs from silicon" — always with the register-first-light scope fence and never conflated with the ECP5 open-flow numbers | docs/results/f2_firstlight/RESULT.md |
| Compression | the pillar-3 sentence above, verbatim | golden/tests/test_effective_bits.py |
| Rigor | "to our knowledge the only open, bit-exact-verified RTL implementation of in-datapath KV compression with a real FPGA bitstream and padding-inclusive accounting" — always paired with the prior-art acknowledgment (Titanus GLSVLSI'25, Kelle MICRO'25) | — |
| Accuracy, 0.5B/1.5B | "near-lossless — worst-case ≈1.9 points, typical ≈1 point, measured and reproducible from a clone" (full 10,042-doc set; deltas −0.008…−0.019, unpaired ±0.007). NEVER "statistically indistinguishable" (retired by the full-set data); tier-vs-tier orderings stay noise (implausible ordering on 1.5B) — no tier "beats" another, ours or published | docs/results/s4_head2head/RESULTS.md (n=10,042 section) |
| Accuracy, 7B | KVQ8: "no detectable effect at full-set power" (−0.0005 ± 0.0009 paired, z=−0.57, 9,965/10,042 agree). KVQ4: "a small, definitively real ≈1.2-point degradation" (−0.0117 ± 0.0020, z=−5.9). "To our knowledge the first public 7B-model accuracy measurement in this ChannelQuant-family method line" (hedged, as written in the results doc) | docs/results/s5_eval7b/RESULTS.md (n=10,042 section) |
| Performance | "Projected 12–14 tok/s short-context decode, TTFT(1k) ≈3.4 s (@0.8 GHz reference config), ~0.16–0.28 J/token, from an analytic model calibrated to measured cycle/P&R anchors (calibration asserted by `--check` on every run), every output labeled PROJECTED, assumptions and citations machine-printed" | docs/results/perf_model/PERF_MODEL.md |
| GPU-facing headline | "7B at reading speed on ~3 watts — with 32k–64k context. Projected ~0.16–0.28 J/token vs a desktop discrete GPU's ~1.4–3.7 J/token at the same single-stream job: 5–10× less energy per token" — PROJECTED, with the §4 qualifiers attached | docs/results/perf_model/PERF_MODEL.md §5–6 |
| Long-context | "KVQ stretches the ≥10 tok/s region from ~19k to ~67k context as stored (~74k at the codec ceiling); at 128k, 8.2 vs 4.3 tok/s — ~2× faster. Keeps reading speed flat as context grows." PROJECTED | docs/results/perf_model/PERF_MODEL.md §3b |

`PERF_MODEL.md` is the **only** quotable source for performance numbers; the
spec (`docs/spec/APEX7B_SPEC.md`) quotes it, never freelances. Accuracy
wording comes **only** from the two RESULTS.md files above.

## 4. Banned wording (grep the drafts before every publish)

- Any performance number without a **PROJECTED** label. Nothing is
  "silicon-proven" or "tapeout-ready"; no silicon exists.
- Any KV-compression ratio above our stored 3.16–3.51× presented as ours.
  The 3.66–3.88× ceiling is quoted only as "zero-overhead ceiling of the
  method family, not a record layout". Never quote the CSR `INFO_CR` field
  externally (it reports the amortized no-pad codec figure, not storage).
- "7B chip" / "runs Qwen-7B" unqualified — use the approved scope sentence.
- "first/only hardware-native KV compression" — prior art precedes us
  (Titanus GLSVLSI'25, Kelle MICRO'25). Only the hedged rigor claim in §3.
- "faster than" any GPU — desktop discrete GPUs are ~8–14× faster at
  batch-1; the projected win is energy per token, never speed. No tok/s
  comparison *in our favor* vs GPUs; the only permitted speed statement is
  the model's own concession (PERF_MODEL.md §6: GPUs are faster, stated
  plainly).
- "first/only sub-0.5 J/token edge 7B" — existing edge accelerators and
  Apple silicon already sit at ~0.3–0.5 J/token (comparator table, cited).
- Unhedged "10×" — always "5–10× vs desktop discrete GPUs at stock
  settings, model shown" (a power-capped GPU or optimized serving stack
  compresses the gap to ~2–6×).
- Any J/token comparison without all four qualifiers: **single-stream,
  single-user, wall/board power, stated quantization on both sides.** Never
  compare against batched-serving numbers.
- "10 tok/s at 128k" — the projection says 8.2.
- Calling 7B KVQ4 "within noise" — the full-set paired test resolves
  −0.0117 as definitively real (z = −5.9), and we say so.
- "Statistically indistinguishable" for the 0.5B/1.5B tiers — retired by
  the full-set data (deltas up to 2.6σ); the approved phrase is the
  "near-lossless: worst ≈1.9 pts" form in §3.
- Any tier "beats" another tier or any published delta — tier orderings
  remain noise even at n=10,042 (implausible KVQ8-below-KVQ4 ordering on
  1.5B; wording rule in `docs/results/s4_head2head/RESULTS.md`).
- Comparisons framed against "default FP16 KV" in mainstream GPU serving
  stacks — they commonly ship 8-bit KV options; the perf model's table
  includes an INT8-KV column for exactly this reason.

## 5. Launch gates — ALL must pass before anything goes public

| gate | what | fills |
|---|---|---|
| **G1** | ✅ done 2026-07-18: 34.11 MHz recorded in `docs/results/s10_fmax/RESULT.md`; bitstream + P&R report regenerated from the shipping-config route | every `TODO(S10b-fmax)` (filled) |
| **G2** | ✅ done 2026-07-16: 150-token chunk-crossing run + RTL replay (382,200 checks/0 fails), `docs/results/s8_7b_token/RESULT.md` | every `TODO(S8-artifact)` (filled) |
| **G3** | ✅ done 2026-07-18: full 10,042-doc runs committed in `docs/results/s4_head2head/` and `docs/results/s5_eval7b/`; accuracy wording re-derived from the full-set stats (both §3 accuracy rows updated; "statistically indistinguishable" retired) | every `TODO(10k-eval)` (all filled) |
| **G4** | `STATUS.md` regenerated at the drop commit (`python scripts/gen_status.py`), all suites PASS | — |
| **G5** | `python3 perf/apex_perf_model.py --check` PASS and `make -C golden test` (incl. effbits) PASS at the drop commit | — |
| **G6** | TODO sweep: `grep -rn "TODO(" README.md docs/launch/LAUNCH_POST.md` returns nothing (this file is excluded — its gate table necessarily quotes the tag syntax; its own placeholders are the two ⚠️-marked §3 rows, cleared by G3) | — |
| **G7** | Banned-wording sweep of README + post + campaign against §4, by fresh eyes | — |
| **G8** | Repo-hygiene audit: only committed, intended-public files referenced; `.gitignore`d working material stays local; drop lands on `main` in one push | — |

Rule for G1–G3: the TODO slots are filled **only** by quoting the named
artifact — never estimated, never pre-written "expected" numbers.

## 6. Channels (single coordinated day)

- Repo: README + post land on `main` (the post also as a linkable page).
- Show HN thread — lead with verification + reproducibility, not perf.
- r/FPGA and r/LocalLLaMA — tailor: bitstream/verification for the former,
  KV-compression accuracy tables for the latter.
- X/Twitter + LinkedIn thread — the honest-accounting table and the
  "what we don't claim" register are the hook, not the headline numbers.
- All channels link the same evidence files; no channel gets a number the
  repo can't back.

## 7. After the drop (public roadmap only)

- KVQ4+ loadable outlier mask at D=128 (the 7B 4-bit-KV quality story runs
  through KVQ4+ — the paired 7B result in `docs/results/s5_eval7b/` is the
  motivation).
- Sky130 signoff + formal RTL≡netlist equivalence on the pipelined KVQ
  engine, with free tools.
- Optional: wall-metered single-stream desktop-GPU baseline with a committed
  re-run script, to replace cited comparator rows with our own measurement.
