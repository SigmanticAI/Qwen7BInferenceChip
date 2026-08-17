# E-7ng ON THE FPGA — THE WALKER-FETCHED GAMMA NORM, WITHOUT ATTENTION

**Date:** 2026-08-06 · **Branch:** `comp/prompt-b-c` · **Image:**
`agfi-04cfe164ba90b8ab0` (apex-e7e8-20260806, @a772713; D=64 GQA=2 DM=896
QSTAGE=14 DDR=1, clkgen A2 verified 15.62 MHz by number) · **Instance:**
`i-0ec2487d966f900f0` f2.6xlarge us-west-2 (terminated + verified) ·
**Driver:** `scripts/fpga/f2/fly_e7ng.py`, `fly_e7ng_poison.py` ·
**Captures:** `build/e7ng/walk_e7ng.hw.cap.jsonl`,
`walk_e7ng_gpoison.hw.cap.jsonl` · **Records:** `e7ng_hw_report.json`,
`e7ng_poison_report.json` (build/e7ng/).

## The claim (exactly)

**On an FPGA, under ONE fmt=1 descriptor and one kick, APEX's sequencer
fetched real weights AND the norm's gamma tensor from the card's own DRAM,
computed the o-projection, applied the requant epilogue on-tile, fed the
product through deq into the residual, then normalized the resulting r1
with the FETCHED gamma and drained h2 through its own y-path — r1 AND h2
both BIT-EXACT against golden, host silent inside the walk window.**

```
walk_e7ng  mask 0xe9c0 = {OPROJ, RES1, NORM2, NFEED, NSRC, FPROJ, FGAM}
  sim (same-commit twin): rc=0 caps=138 grade=True     (fs/h2/r1 all exact)
  HW  (this image):       rc=0 caps=138 grade=True 1.6s

DISC gamma-poison (PHYSICAL DDR reload, one byte of resident g2, 19->83):
  h2 == the golden-PREDICTED poisoned codes : True
  r1 held bit-exact                         : True   -> RED, localized
  good image restored + FULL-verified after : 466,288/466,288, 0 fails
```

The poison arm is the load-bearing discriminator: one flipped byte of the
g2 tensor **in card DRAM** moves h2 by exactly the golden-predicted codes
while r1 does not move (gamma is downstream of the residual). The only
path from that byte to h2 is the walker's own FGAM fetch — the host never
touches gamma. The poison manifest re-hashes BOTH sha levels
(whole-image + per-tensor), so the loader's integrity gate stayed armed
rather than bypassed.

## Why this arm exists — read with WALKED_ATTENTION_DEFECT.md

E-7 proper (this chain + SCORE/PV) is sim-green and **silicon-red** — the
walked-attention defect, characterized the same day on the same card. This
run proves E-7's actual novelty — the fetched-gamma machinery (FGAM record,
`apex_gam_unpack`, the walker y-drain) — is **silicon-correct in
isolation**. Combined with the same-session control (7 committed host-mode
attention jobs, both KV engines, all pass on this image):

| arm | composite involved? | walked? | silicon |
|---|---|---|---|
| host attention jobs ×7 (both kv engines) | YES (host mode) | no | **PASS** |
| walk_e7ng (this doc) | no | YES | **PASS** |
| walk_e6c / e7 / e8 (score+pv walked) | YES (walk mode) | YES | FAIL |

The defect's boundary is now exact: **the walker driving the composite
scale-cache** — not the bank's arithmetic (host mode proves it), not the
walker's chains (this run proves them), not the fetch/epilogue/norm
machinery. One seam, on one image, sim-green.

## Scope fence

Toy geometry (D=64, H=1, T=1), the E-7 toy subject; an ARCHITECTURE claim,
not a model claim. The o-act is host-staged before the walk (walk_oproj's
silicon-proven shape) — with SCORE/PV masked out, `fp_r_attn = 0`
(`seq_layer_walker2.sv:686,707`): the o-act stages at row 0 and the y-drain
window is row 1, both re-derived and graded, not assumed. The level word
the walker leaves (`0x545`) was decoded back through `fmt.lctl` (rope
levels come from the DESCRIPTOR, not host staging), never baked as a raw
measured value.

## Cost

Shared card session with the host-control differential: f2.6xlarge ~50 min
≈ **$1.50**. Terminated + verified; account F2 sweep shows only the
unrelated `apex-f2-fpga` box.
