# C2 — A REAL PROMPT, EVERY OP TYPE OF ONE LAYER ON THE FPGA, TOKEN UNCHANGED

**Date:** 2026-08-01 · **Branch:** `comp/prompt-b-c` @ `2e78dc3` ·
**Image:** `agfi-09e947873048a6877` (narrow + R4) · **Instance:**
`i-0c0a86fffc367e1c7` f2.6xlarge us-west-2b (terminated + verified) ·
**Driver:** `scripts/fpga/f2/layer_offload.py` · **Record:**
`build/layer_offload/layer_offload_result.json`, log `build/c2_hw.log`.

## The claim (exactly)

**For decoder layer 0 of a real Qwen2.5-7B run on the prompt "The capital of
France is", EVERY op type that layer performs was served by APEX's blocks on
an FPGA, and the model emitted the IDENTICAL next token.**

```
  executor        : hw   (remote_hw_exec clock gate ON — a1 verified 15.62 MHz)
  tile jobs       : 136 programs, 43,470 capture records, 317.3s in the executor

  op type                    who         served/total        exact?
  projections q/k/v/o/g/u/d  TILE (hw)   56/53,760           BIT-EXACT
  RoPE (decode-token q)      TILE (hw)   3,584/3,584         reconstructed*
  attention (score + PV)     TILE (hw)   3,584/3,584         BIT-EXACT
  residual (r1, r2)          TILE (hw)   7,168/7,168         BIT-EXACT
  RMSNorm-2                  TILE (hw)   3,584/3,584         reconstructed*
  SwiGLU                     TILE (hw)   18,944/18,944       reconstructed*

  token OFFLOAD ON  : ids=[12095] text=' Paris'   (475s)
  token PURE HOST   : ids=[12095] text=' Paris'   (138s)
  token HOST+BUS_ON : ids=[12095] text=' Paris'   (139s)  -> bus mode alone changes nothing
  OP TYPES SERVED BY THE TILE: 6/6
  TOKEN IDENTITY: PASS      MILESTONE C2: PASS
```

**The substitutions are LOAD-BEARING, proven, not assumed.** The
discriminator scales the tile's returned values by 0.5 and the emitted token
CHANGES (ids=[279], max|dlogit| 12.97). A substitution that the pipeline
would have produced anyway cannot do that.

## Honest limits — read these with the claim, not after it

1. **Projections were SAMPLED**: 56 of 53,760 accumulators (`--proj-cols 8`).
   Full width is ~300k tile GEMM jobs. The other FIVE op types ran at FULL
   7B width. (Separately, the projection op type IS proven exhaustively on
   the FPGA — 1024/1024 blocks — in `PROJ_SWEEP_RESULT.md`.)
2. ***"reconstructed"** means the value re-enters golden through the tile's
   C-1 view rather than as raw bits, because RMSNorm-2/SwiGLU/RoPE egress
   only through the 128-wide C-1 feeder (B-FEED-WIDTH). The ledger prints
   the measured deltas (norm2 max_abs_delta_q78 = 6; swiglu max_abs_delta =
   6.7e-03) and the downstream code differences (46 and 411) rather than
   calling them bit-exact. RoPE's reconstruction IS exact where it is
   consumed — golden's next stage re-quantizes to identical codes+scale,
   checked 28/28 heads.
3. **27 of 28 layers ran on the host.** This is ONE layer.
4. Also host-side (printed in full by the tool every run): tokenizer,
   embeddings, lm_head+argmax, all data movement, the C-2 requant epilogues
   (tile returns raw INT32, host calibrates), the score-dequant composites
   (the online softmax itself runs on the tile's ASU), the whole-row C-1
   quantizations, the 27 chunk-sum adds in this variant, RMSNorm-1, and the
   RoPE of cached K rows.

**What this is NOT:** "the model runs on the FPGA". It is one layer's op
types, for a real prompt, with the output token unchanged.

## How the seam works (no golden edits)

`layer_offload.py` rebinds golden module globals — `gemm_i8_ksplit`,
`rope_fx`, `attention_core`, `_f16` at the r1/r2 and SwiGLU sites,
`rmsnorm_fx_wide` — to tile-job builders. `_f16` has several call sites, so
the wrapper locates them by SEARCHING transformer.py's own source at entry
and re-verifies each interception against independently tracked operands; a
wrong site map REFUSES (a selftest case). 68/68 substitution and consumption
checks passed. Golden itself is never edited.

## Cost / teardown

f2.6xlarge ~20 min ≈ **$0.60**. `terminate-instances` → `wait
instance-terminated` → describe reads **terminated**; account F2 sweep shows
only the unrelated `apex-f2-fpga` box, untouched.
