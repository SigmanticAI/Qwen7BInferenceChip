# S10 — Honest FPGA re-baseline (before any pipelining) — 2026-07-14

Re-measured on the **post-D-026 RTL** with the open flow (sv2v → yosys
`synth_ecp5 -abc9` → nextpnr-ecp5, LFE5U-85F / CABGA381 / speed-6), because the
previously-quoted "9.04 / 9.77 MHz" was measured on a **proxy config (D=16,
DEPTH=2) of the pre-dedup design** — never a fair number for the shipping tile.
Reproduce: `scripts/fpga/s10_baseline.sh`.

## Results

| config | params | fits? | post-route Fmax | notes |
|---|---|---|---|---|
| **b0 proxy** | D=16, TIER=1, G=2, K=0, DEPTH=2 | ✅ yes | **9.19 MHz** | tiny; matches the old toy number on the new netlist |
| **b1 ship** | D=64, TIER=2, G=128, k=2, DEPTH=128 | ❌ **NO** | — (unplaceable) | 218% FF, 167% LUT, **0/208 BRAM** |

## The finding that matters more than the clock

**The shipping-capacity KVQ engine does not fit on the ECP5-85F today** — not
because of logic, but because its buffers collapse into flip-flops instead of
mapping to the FPGA's dedicated block-RAM. yosys prints
`Replacing memory \mem with list of registers` for every one, and nextpnr then
overflows the part (182,886 / 83,640 FF = 218%).

Dominant offender — **`residual_buffer`** (the key-group token buffer):
at DIM=64, DW=16, G=128 it is `128 × 1024 b = 131,072 flip-flops` by itself
(~72% of the FF overflow). Its own header says it is meant to infer "a true
$mem/BRAM." Others: `sram_controller` (record store, 128 × 320 b + a separate
`valid[128]` bit-array) and `scale_bank_store` (small, 8 × 1024 b).

**Root causes blocking BRAM inference (to fix in S10a, before any Fmax claim):**
1. `scale_bank_store.sv` and `sram_controller.sv` **async-reset the entire
   memory array** (`always @(posedge clk or negedge rst_n) … for i: mem[i]<=0`)
   — BRAMs cannot be async-cleared, so yosys refuses to map them. Fix: drop the
   array reset (rely on the `valid`/occupancy tracking already present; the
   D-020 audit reset-attack bucket proves contents-persist semantics anyway).
2. `sram_controller`'s separate `valid[DEPTH]` bit-array + async reset — fold
   validity into an occupancy pointer / init-to-zero pattern BRAM can express.
3. Confirm the wide (1024-b) rows split across parallel DP16KD blocks after (1).

## Revised S10 plan (data-driven)

- **S10a — BRAM inference (NEW, do FIRST):** make residual_buffer / sram_controller /
  scale_bank_store map to DP16KD so the ship config *places at all*. Pure
  memory-style rewrite, must stay bit-exact (re-run the full KVQ + top matrix,
  D-026 counts unchanged — these are storage, not datapath). Without this there
  is no ship-config Fmax to improve.
- **S10b — divider pipelining (the original S10):** the ~8-stage pipeline of the
  fp16 divide + the readback cut, targeting ≥100 MHz, per the architect plan
  (scratchpad s10_design.md). Only meaningful once b1 fits.

The proxy (b0) still routes at 9.19 MHz and is a valid apples-to-apples "before"
for S10b once we have a fitting ship config to compare against.
