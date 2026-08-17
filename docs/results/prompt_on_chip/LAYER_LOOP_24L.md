# THE LAYER LOOP — 24 LAYERS OF A REAL 0.5B DECODE STEP ON THE WALKED PATH

**Date:** 2026-08-05 · **Branch:** `comp/prompt-b-c` (from `6678a7c`, the
E-6-on-silicon tip) · **Driver:** `scripts/fpga/f2/walk_layers_05b.py`
(NEW — the loop ABOVE the frozen E-6 chain; `walk_fuel_layer.py`,
`walk_fuel_proj.py`, the emitters and `rtl/` untouched) · **Image tool:**
`make_weight_image.py` (unchanged — it already scaled) ·
**Twin:** `verif/f2sim/obj_e6r_b64_ddr1` (the E-6 DDR=1 twin, tile_div=2) ·
**SIM-MEASURED; no hardware flight in this session** (the `--executor hw`
arm exists, gated, unflown — see §5).

## The claim (exactly)

**The walked E-6 chain — sequencer-fetched DDR weights, QKV + OPROJ with
the on-tile requant epilogue, r1 through deq → residual — now runs for
EVERY LAYER of Qwen2.5-0.5B, back to back over one 24-layer resident
weight image, one fmt=1 descriptor per layer. Between layers the host does
data movement and control ONLY: it re-stages the (pre-computed) operand
rows, re-points the descriptor's four tensor-table bases into layer l's
regions, reloads the two per-layer calibration slots, re-arms fuel, and
kicks. All 24 layers' QKV blocks (144/layer) and r1 rows (896/layer) are
BIT-EXACT against golden.**

```
layers walked        : 24/24
every walked layer   : QKV 144/144 + r1 896/896 bit-exact -> PASS
LAYER LOOP GATE      : PASS
mask                 : {FPROJ, QKV, OPROJ, RES1} (wfl.MASK_B)
tile cycles / layer  : 559697  (min..max 559697..559697 — IDENTICAL every layer)
tile cycles / job    : 2186.3
```

## 1. The weight image at scale (all measured on this Mac today)

`make_weight_image.py build --layers 0..23` needed **zero code changes** —
the 2-layer default was a CLI choice, not a limit.

| item | measured |
|---|---|
| image size | **358,219,776 B = 341.6 MiB** (240 tensors = 10 × 24 layers; payload 358,109,184 B + 4 KiB alignment) |
| build wall | **3.66 s** (vectorised encoder, sampled `wgt_beats_ws` cross-checks included) |
| `check` (re-derive EVERY payload from the weight files) | **PASS 240/240, 3.38 s** |
| burst plan | 5,616 bursts ≤ 64 KiB (`f2_ddr_load.py --plan`, refusal-checked) |
| per-layer payload | 14,921,216 B = 14.23 MiB (weights 14,909,440 B + g1/g2 2 × 1,792 B + phase 8,192 B) |
| fits the card? | 341.6 MiB ≪ the F2 **64 GB DDR4 DIMM** (IB_FUEL §0) — ~0.5 % of capacity; max `base_64B` = 5,597,056 ≪ the walker's 2^30 tensor-table field |
| in-sim behavioral load | **13,990,016 shell cycles ≈ 109.3 s** per f2sim process (full PCIS write + full readback verify + sha, measured with a 1-op probe) |

**Card load time, PREDICTION (labelled):** at the loader's measured
101.22 MiB/s (DDR_BRINGUP_RESULT.md, the same `f2_ddr_load.py` on the
28.5 MiB image) the 341.6 MiB image loads in **~3.4 s** — one-time,
pre-flight. Not measured on silicon in this session.

### The per-layer base addressing the walker's tensor table needs

The layout is exactly periodic — **`base(L, T) = off(T) + L × 0xE3C000`**
(stride 14,925,824 B, uniform across all 24 layers, asserted by the
driver's selftest):

| T | off(T) | bytes | gated |
|---|---|---|---|
| Wq | 0x0000000 | 802,816 | yes |
| Wk | 0x00C4000 | 114,688 | yes |
| Wv | 0x00E0000 | 114,688 | yes |
| Wo | 0x00FC000 | 802,816 | yes |
| Wg | 0x01C0000 | 4,358,144 | yes |
| Wu | 0x05E8000 | 4,358,144 | yes |
| Wd | 0x0A10000 | 4,358,144 | yes |
| g1 | 0x0E38000 | 1,792 | no (carried, unconsumed by any gate) |
| g2 | 0x0E39000 | 1,792 | no |
| phase | 0x0E3A000 | 8,192 | no |

The driver never hardcodes this: bases come from the manifest per layer,
and selftest case 6 proves consecutive layers' 64-word descriptors differ
in EXACTLY {W_TENS0+WQ/WK/WV/WO, W_RQ0+14, W_JC0+JC_OPROJ} — the
"re-point + re-arm" claim checked structurally, not asserted in prose.

## 2. What the host does per layer (measured census, layer 0's artefact)

All operand VALUES are pre-computed once, before the loop (one golden
prefill of the committed prompt `[785, 6722, 315, 9625, 374]`, 4.9 s);
between layers the host moves data and writes control — it computes
nothing.

| segment | ops | what they are |
|---|---|---|
| pre-walk | 9,813 (5,484 w + 3,937 pw + 310 jf + 37 r + 34 poll) | stage 2 × 14 C-1 act frames into bank 1 (rows 0-13 'h' for QKV, rows 14-27 attn for OPROJ); preload the 896-word f16 layer-input row X_l; write the 64-word descriptor (re-pointed bases + RQ[14]/JC_OPROJ); fuel arm (1 poll + 1 ctrl write); WALK_GO |
| walk window | 1,586 (144 RO-advance w + 144 poll + 1,296 cap) | the 144-job raw-QKV RO drain — DATA plane, E-5's disclosed class; **0 control writes** (predicate-checked per layer, run refuses otherwise) |
| post-walk | 920 (896 cap + 4 poll + 3 w + 11 r + 2 rn) | ONE done-poll, hygiene, read the 896 r1 caps (produce mode — the walk's OUTPUT, never foreknown) |
| total | **12,319 regops/layer** | |

### The inter-layer seam, stated honestly

The walked set is {QKV, OPROJ, RES1}; the deepest walked activation is
**r1**. A layer's true output is r2 = r1 + FFN(norm2(r1)), and
NORM1/attention/NORM2/FFN/RES2 do NOT walk on today's RTL (loud S2_CHECK
refusals — WALKED_EPILOGUE_E6.md). So the carry is: layer l's walked r1
is graded bit-exact against golden r1_l FIRST, then the host stages
golden's own r2_l (pre-computed, moved not computed) as layer l+1's X row.
The loop drives and verifies 24 × {QKV, OPROJ, RES1} of a real decode
step; it does not make the unwalked five step families disappear.

## 3. The sweep (sim, one f2sim process per layer, tile_div=2)

| layer | QKV blocks bit-exact | r1 elements bit-exact | tile cycles | sim wall (s) |
|---|---|---|---|---|
| L00 | 144/144 | 896/896 | 559697 | 10.6 |
| L01 | 144/144 | 896/896 | 559697 | 9.4 |
| L02 | 144/144 | 896/896 | 559697 | 9.5 |
| L03 | 144/144 | 896/896 | 559697 | 9.9 |
| L04 | 144/144 | 896/896 | 559697 | 9.6 |
| L05 | 144/144 | 896/896 | 559697 | 9.6 |
| L06 | 144/144 | 896/896 | 559697 | 9.8 |
| L07 | 144/144 | 896/896 | 559697 | 9.4 |
| L08 | 144/144 | 896/896 | 559697 | 9.5 |
| L09 | 144/144 | 896/896 | 559697 | 9.5 |
| L10 | 144/144 | 896/896 | 559697 | 9.6 |
| L11 | 144/144 | 896/896 | 559697 | 9.7 |
| L12 | 144/144 | 896/896 | 559697 | 9.7 |
| L13 | 144/144 | 896/896 | 559697 | 9.5 |
| L14 | 144/144 | 896/896 | 559697 | 9.7 |
| L15 | 144/144 | 896/896 | 559697 | 9.5 |
| L16 | 144/144 | 896/896 | 559697 | 9.3 |
| L17 | 144/144 | 896/896 | 559697 | 9.6 |
| L18 | 144/144 | 896/896 | 559697 | 9.8 |
| L19 | 144/144 | 896/896 | 559697 | 9.5 |
| L20 | 144/144 | 896/896 | 559697 | 9.8 |
| L21 | 144/144 | 896/896 | 559697 | 9.5 |
| L22 | 144/144 | 896/896 | 559697 | 9.7 |
| L23 | 144/144 | 896/896 | 559697 | 9.6 |

Every layer is a separate f2sim process against the SAME 24-layer DDR
image; only the tensor-table bases and the two calibration slots differ per
layer. The cycle count is **identical to the cycle across all 24 layers**
(559697), which is the expected shape: the walked
window's work is geometry-determined, not data-determined. Golden prefill
for the whole sweep: 5.7 s; prompt ids [785, 6722, 315, 9625, 374].

## 4. Measured cycles, and the labelled PREDICTION

**MEASURED (sim, tile_div=2, 24 layers):**
- walk-window tile cycles/layer: **559697** (min..max 559697..559697)
- shell cycles/layer: 2238788
- tile cycles per fuel-fed job: **2186.3**
- sim wall/layer: 9.3..10.6 s (mean 9.6 s)

**PREDICTION (labelled, sim-derived, hardware-unvalidated).** Tile-cycle →
wall conversion is the corrected `cl_apex.sv:82` rule (tile period =
2·tile_div shell cycles; the E-6 receipts correction), and the walls count
the WALK WINDOW only — today's seam additionally spends ~12.3k host
regops/layer on staging/readback, MMIO-bound on hardware and NOT included:

- **walked-steps-only, per token** (what the mask above actually covers,
  x24 layers): 13432728 tile cycles =
  **860 ms @ the flown 15.625 MHz** ·
  **215 ms @ the analyzed 62.5 MHz**
- **IF-fully-walked, per token** (HYPOTHETICAL — three step families are
  still fenced, see `MASTER_TABLE.md` §4 and E2E_TOY_LANE §4):
  ~94.9M tile cycles = **~6.1 s @ flown** · **~1.5 s @ analyzed**

Neither figure is a hardware measurement. The bit-exactness above is
simulation; the flown E-5/E-6 results are the hardware anchors this
extrapolates from.

## 5. The hw arm (built, gated, NOT flown)

`--executor hw` routes through `remote_hw_exec.attach()` and REFUSES
loudly unless (a) `$APEX_F2_HOST` is set (no shim → SystemExit, nothing
runs) and (b) `--hw-ddr-attest` names an `f2_ddr_load.py --full-verify`
result whose `image_sha256` matches THIS image's manifest with 0 fails —
a sampled verify or a different image's sha is refused by name (selftest
cases 8-10 prove all three refusals). No instance was launched and no
hardware ran in this session.

## 6. Gates (this session)

`walk_layers_05b.py selftest` 14/14 · `make_weight_image.py selftest`
ALL PASS · `f2_ddr_load.py --selftest` ALL PASS · `walk_fuel_layer.py
selftest` ALL PASS · layer-0 subject parity vs the flown driver's own
builder asserted inside `run` (refuses on drift) · no RTL, emitter,
`verif/`, or fence-agent file touched.
