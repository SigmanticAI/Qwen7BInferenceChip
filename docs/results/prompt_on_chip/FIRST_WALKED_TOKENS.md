# FIRST WALKED TOKENS — a real prompt answered with walked chains on silicon

**Date:** 2026-08-11 · **Image:** `agfi-0500f4afe435b5e71` (apex-full-
20260811, the merged D-033 stack) @A2 15.625 MHz · **Card:**
`i-0f5efed9674c4d6e3` (terminated+verified) · **Harness:**
`token_loop.py run --engine hw-walked --tokens 3 --ddr-attested`
(run ON the card, root executor, direct slot-0 attach) ·
**Record:** `build/token_loop/token_loop_hw-walked.json` (committed).

## What happened

Prompt `"The capital of France is"` (ids 785,6722,315,9625,374) →
generated ids **[12095, 13, 1084]** (" Paris. It") — greedy decode,
token-identical to the pure host-golden reference at every token.

Per generated token, all 24 layers' walked **{FPROJ, QKV, OPROJ, RES1}**
chains ran on the REAL tile with REAL DDR weight fetches:
**72/72 chains, 0 refused, every walked value bit-exact**
(144/144 QKV INT32 accumulators + 896/896 r1 f16 per chain) against
golden computed on the same operands. DDR image full-verified on load
(240 tensors, 5,595,456 words, 0 fails); descriptor drift fenced to the
re-point set; subject parity fenced against `wfl.golden_subject`.

## The numbers, labelled the way we always label

| quantity | value | label |
|---|---|---|
| end-to-end hybrid wall | 3 tokens / 43.38 s = **0.069 tok/s** | MEASURED — walked chains on silicon + all other steps python on the card CPU + per-chain executor attach |
| per-token wall | 19.15 / 12.09 / 12.15 s | MEASURED (first token carries one-time warmup) |
| walked MAC share | 44,040,192 / 516,196,352 = **8.53 %** | measured census (golden's own calls) per token |
| host-golden software pipeline alone | 0.58 tok/s | host software number, NOT a tile claim |
| prior measured baseline | 0.004 tok/s | the July-30 host-driven prompt demo |

The honest sentence: **the first tokens ever produced with walked chains
on APEX silicon, 17× faster end-to-end than the previous measured
baseline, with 8.53 % of the arithmetic on the walked path and every
walked value bit-exact.** The wall is dominated by per-chain transport
(24 executor invocations/token ≈ 0.5 s each) — not by the tile: the walk
window itself is ~36 ms at A2.

## Where the next wall-clock factors live (in order)

1. **Batch the 24 chains into ONE executor invocation per token**
   (f2_host_run takes many regops files; the builders already emit them
   independently) — removes ~11 s/token of attach overhead.
2. **Walk more of the layer** — attention is silicon-proven as of
   CACHE_SWEEP_FIX.md; FFN rungs 1–3 are sim-proven and IN this image.
   Each step walked moves python time onto the tile.
3. **A0 (62.5 MHz)** — 4× the tile clock; **W4** — 2× the fetch.

## Reproduce

```
# card: load agfi-0500f4afe435b5e71 + clkgen A2 + chmod sysfs resources
# ship code + weights, then:
python3 scripts/fpga/f2/f2_ddr_load.py --image build/ddr_weights_05b_24L \
    --load --verify --full-verify
sudo python3 scripts/fpga/f2/token_loop.py run --engine hw-walked \
    --tokens 3 --ddr-attested
```

## 2026-08-11 PM update — the transport ladder, measured

Same card class, same image (`agfi-0500f4afe435b5e71`), same 3-token run,
A/B on one card (`i-04034cee81240114a`, terminated+verified):

| transport | per-token wall (steady) | delta |
|---|---|---|
| per-layer invocations (the AM baseline) | 12.1 s (control re-run: 19.8 s first-token) | — |
| ONE batched invocation per step | 11.3 s | −0.8 s (transport was small as root) |
| + parallel program emission (12-wide pool) | **5.8 s** | **−5.5 s** (emission WAS the cost) |

Steady-state ≈ **5.8 s/token ≈ 0.17 tok/s** for the graded hybrid —
2.1× the AM number, 43× the July baseline. All gates held at every rung
(token identity, 72/72 walked chains bit-exact, cap-split exact).
Remaining per-token profile: golden python composition ~2.5–3 s (shrinks
only by WALKING more steps), pooled emission ~1 s, execution+grading
~1.5–2 s. Next levers in order: walk attention (E-7 mask) into the
chain, A0 (4×), W4 (2×).

## 2026-08-13 — walked FFN silicon debut + fast mode, one card

Card i-0da75de57dfaf5507 (terminated), settle image agfi-030a812cd224b409d:

- **WALKED FFN GREEN ON SILICON** (first flight of the rung-3 machinery
  through the token loop): all 24 layers, fs 16/16 + p_codes 1024/1024 +
  r2 896/896 BIT-EXACT per layer, ~0.2s exec/layer, gate PASS. The
  slice-image residency scheme (@0x2000_0000 alongside the main image)
  works; ar_owner quiesce needed between runs (FUEL_CTRL=0).
- **Fast mode measured**: MASK_B --fast 3, 6 tokens:
  [5.47, 5.00, 5.09, 5.49, 5.27, 5.32] s/token, gate PASS, honest
  labels (no identity claim). Confirms the anatomy: the remaining wall
  is hw exec ~2.5s (A0's target) + golden ~1.7 + emit ~1.2.
- Ladder: 12.1 → 5.8 → 5.4 (graded) → **5.1-5.5 fast** s/token. The next
  big rungs are A0 (in the oven: in-bank ingress stage build) and
  emission caching; e7 blocked on the softmax-emission cone (layer 3).

## 2026-08-14 early AM — the combined software stack, measured

Card i-061d96faa287d9b46 (terminated), settle image, one flight:
- **~4.0 s/token steady = 0.25 tok/s MEASURED** (fast 3 + emit-cache +
  golden-trim; tokens [5.31, 4.04, 3.94, 4.28, 3.99, 3.98], gate PASS).
- Golden trim: bitwise-identical outputs (ids+logits bits verified
  pre-commit); host golden 0.89-1.14 → 0.75 s/token.
- Clock recipes probed EMPIRICALLY on-card: A-group true values are
  15.62 / 62.5 / 125 — the CLI table lies (claims A2=62.5); NO
  intermediate preset exists. The 2x-for-free door is closed; the clock
  lane = the A0 redesign campaign (sc_mem cone pipeline, agent working)
  + possibly a clock-pin-swap build if B/C groups carry a real ~31MHz
  (probe their true values next card).

MEASURED LADDER: 12.1 → 5.8 → 5.4 → 5.1 → **4.0** s/token.
Anatomy now: walk ~2.5 (62%) + golden ~1.2 + emit ~0.35 + head ~0.16.

## 2026-08-14 PM — A0 62.5MHz MET AND MEASURED

Build 4 (3-stage sc_mem write pipeline) closed timing: WNS +0.015.
Flight (card i-0405f6fbfe876a76b): battery 193/193 BIT-EXACT AT 62.5MHz,
then fast 6-token: [4.70, 3.43, 3.30, 3.61, 3.32, 3.32] s/token —
**steady ~3.3 s/token = 0.30 tok/s MEASURED**, gate PASS.

MEASURED LADDER: 12.1 → 5.8 → 5.4 → 5.1 → 4.0 → **3.3** s/token.

The 4x clock delivered exactly the tile share (~0.65s saved): the
revealed next wall is EXECUTOR/AXI TRANSPORT (~1.5-1.8s of per-op OCL
chatter — thousands of regops each token) — a SOFTWARE lane (burst-mode
executor), not silicon. Remaining anatomy at 3.3: transport ~1.6 +
golden ~1.2 (e7 walked shrinks it) + emit 0.35 + head 0.16 + tile 0.2.

## 2026-08-14 late — burst executor measured + the attention verdict

- Burst executor at A0: steady **3.1 s/token = 0.32 tok/s MEASURED**
  (from 3.3; the 0.6s prediction over-attributed op chatter — the
  remaining transport wall is per-INVOCATION overhead: 24 executor
  launches+attaches per token; next cut = a persistent executor daemon
  or single-invocation batching across layers; --time-report now
  attributes it on-card).
- e7 attention: the emission fix (qs rebias flops, provably in the comb
  image) did NOT change silicon's transfer map — bit-identical old map
  (1/2 @T=1, [1/4,3/8] @T=2). The cone identification was wrong or the
  fold reformed. VERDICT: no more RTL guessing — NETLIST forensics on
  the comb DCP (s3 dcps/comb-2026_08_14-210124) tracing the actual qs
  cone, with the micro-flight map as ground truth.

MEASURED LADDER: 12.1 → 5.8 → 5.4 → 5.1 → 4.0 → 3.3 → **3.1** s/token.

## 2026-08-16 — THE BIG STACK: 1.78 s/token = 0.56 tok/s MEASURED

Sleep-cut (0.48s) + golden-trim-2 (5.6x golden, 14x head, bitwise-
certified) stacked on the A0 image: fast 6-token steady
[1.794, 1.769, 1.799, 1.78] ≈ **1.78 s/token = 0.56 tok/s**, GATE PASS.

MEASURED LADDER: 12.1 → 5.8 → 5.4 → 5.1 → 4.0 → 3.3 → 3.1 → **1.78**.

e7 fence (S-4 softmax-divider race, program-side): sim 24/24 but
silicon STILL RED, identical refuse — the drain reorder was not the
enabler. Disclosed next step: host-side poll backoff during the S-4
window. The complete mechanism (divider MSB-first quot; p/2^k; x1.5
double-sample; closed-form k(t,T)) is banked in E7_LIVE_T_DEFECT.md.

Remaining anatomy at 1.78: engine drain tail (~1.1 est) + golden ~0.2
+ head ~0.02 — FILETIMES attribution next card.

## 2026-08-18 night flight (image agfi-0993d61f190577fe2)

- fast 6-token: steady ~1.77 s/token = **0.56 tok/s CONFIRMED** on the
  fresh A0 current-tree image (GATE PASS).
- e7 + fence at A0: STILL RED (identical refuse) — the S-4 fence does
  not stop the race on silicon; next = host poll backoff (in-session).
- W4: still 0 cells despite the wired knob (the define does not reach
  synth_design — read the invocation echo in the night vivado log).
- Ops notes: --time-report is an f2_host_run flag (pass via
  APEX_F2HR_EXTRA), and battery regops must ship in the code tgz.
