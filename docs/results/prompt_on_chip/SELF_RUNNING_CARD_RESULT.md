# THE SELF-RUNNING CARD — WALKER FETCHES WEIGHTS FROM CARD DRAM AND COMPUTES, ON THE FPGA

**Date:** 2026-08-05 · **Branch:** `comp/prompt-b-c` @ `da4d6c4` ·
**Image:** `afi-0b1c70e563e602575` / **`agfi-0183a4b88c8d21163`**
(apex-convergence-20260805: E-3b + E-4b walked chains + **E-5 walked fuel
projections** + DDR=1 + the first REAL CDC constraints; D=64 DMODEL=64
GQA=2 DM=896 QSTAGE=14, clkgen A2, Slack MET +0.277 ns, PRV-GREEN,
ingested first try) · **Instance:** f2.6xlarge us-west-2 (terminated) ·
**Captures:** `walkfuel_hw.cap.jsonl` (good), `walkfuel_poison.cap.jsonl`.

## The claim (exactly)

**On an FPGA, APEX's own layer sequencer issued DDR fetch requests from its
own descriptor tensor table, the fuel engine streamed real Qwen2.5-0.5B
weights out of the card's DRAM into the weight FIFO, and the MXE computed
all 144 Q/K/V projection blocks of layer 0 — every INT32 accumulator
BIT-EXACT against golden's `gemm_i8_ksplit`, with the host writing NOTHING
during the walk.**

```
[weights]  28.5 MiB real 0.5B tensors resident in card DRAM
           468 BAR4 bursts, 101 MB/s, FULL readback 466,288/466,288 words exact
[CLAIM]    walk_fuel_qkv: 6178 ops, 35 checks, 0 fails, 1296 captures
           144/144 jobs (Wq 112 + Wk 16 + Wv 16) ALL BIT-EXACT, bad=0
           host-write silence: 0 control writes / 144 RO advances in the walk window
[DISC A]   walk-off (W2_MASK[14] cleared): 4576 ops, 1 fail, ABORTED
           `poll stall @0x3200 want 0x100` — parks at S2_FETCH, 0 captures
[DISC B]   one poisoned byte in DDR (Wq[0,0] +1), manifest re-hashed so the
           loader's integrity gate still holds:
             lane0 851 -> 852, delta = +1 = x_codes[0] = the EXACT prediction
             all_bit_exact False, bad=1  -> RED by the predicted delta
```

Reproduced **three times** green across card resets (the main claim ran green
on every load of the good image).

## What this establishes that simulation could not

1. **AWS's real encrypted DDR4 controller trains and serves our fuel path.**
   The sim used `sh_ddr_beh.sv`, our own fabrication of that controller.
2. **The weights the MXE multiplied physically came from card DRAM** — DISC B
   is a byte-level causal proof, not an inference.
3. **The walker drove it** — DISC A removes the walker's step and the same
   program cannot proceed.

## Operational findings (recorded, not smoothed over)

- **AR-port ownership is a real protocol.** After a walked run `ar_owner=RUN`
  and the loader REFUSES to write (DECERR by design, §9.1 R8); after an
  aborted load the port can be left host-owned and a subsequent walk parks.
  **Recovery: reload the AFI + clkgen recipe + reload DDR** — proven twice.
- **The loader's sha256 gate is load-bearing**: it refused the first poisoned
  image (per-tensor sha) and then the whole-image sha. The poison used here
  re-hashes BOTH so the integrity gate stays armed rather than bypassed.
- `f2_ddr_load.py` needs its repo tree depth (`REPO = HERE.parents[2]`) and
  `--image` is the DIRECTORY (bin + json), not the bin.

## Scope fence

QKV projections of ONE layer, fenced to FPROJ (no o8 epilogue — that lives on
`pc_hasrq`, excluded here). The 128-wide toy chains (E-3b/E-4b) and the
0.5B multi-layer prompt driver are separate, already-proven claims. This does
NOT yet run a whole layer walked end to end.

## Cost

~45 min f2.6xlarge ≈ $1.25.
