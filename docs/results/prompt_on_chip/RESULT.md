# PROMPT ON CHIP — one attention op of a real 7B decode step computed on an FPGA

**Date:** 2026-07-29 · **Status: MILESTONE C+D PASS, in simulation AND on real
silicon.** · Branch `comp/prompt-b-c` · Contract: `docs/design/PROMPT_ON_CHIP.md`

## The claim (exact wording; nothing beyond this is supported)

> One attention operation of a real Qwen2.5-7B greedy decode step was **computed
> by our verified tile on an AWS F2 FPGA** — the tile's raw INT32 accumulators
> were returned to the host, requantized by the golden model's own functions,
> substituted into the model state, and the decode produced the **same token**
> as the pure-golden path.

NOT claimed: 7B runs on the chip · a layer runs on the chip · any throughput
number (15.625 MHz is a correctness clock; ~4.3 s/op is dominated by host
round-trips and the sequential MMIO contract, not by the tile).

## What makes this different from every previous silicon result

Every prior number (27,996 checks on silicon, 8.7M KVQ checks, …) was
**agreement-mode**: the driving program was compiled *from* golden's answer and
the read ops compared-and-discarded, so no value the tile produced ever reached
a host variable. This run is **produce-mode**: the job carries inputs + config
only (`requant_en=0`, so the raw `acc_o` lands on the RO lanes), the outputs are
`cap` ops with no baked expectation, and the host learns the numbers only by
reading them back. Golden is consulted *after* the run, as a grader.

## Hardware run (the deliverable)

Instance `i-0c6438df47f76e0ec` (f2.6xlarge, us-west-2), AGFI
`agfi-0ae06ea568e5667ba` (afi-036d83cafa00d26ea, shell 0x10212415, D=128,
GQA_NENG=1, CQ-8, maskless, tile on `clk_extra_a1` recipe A2).

```
[remote_hw_exec] executor 'hw' -> ubuntu@18.236.72.126:~/apex_d (clock gate ON)
[C] model=mlx-community/Qwen2.5-7B-4bit L=28 H=28 tier=CQ-8 G=128; prompt=5 tok + 1 new = 6 <= 128 OK
[C] offload target: layer 0 head 0 (every step)  mode=hw
[offload] poff_s004_L00_h00 T=5 CQ-8 hw: acc<-TILE via compute_job.grade_compute_job (hw) caps=232 out_hat == golden (4.611s)
    - poff_s004_L00_h00  T=5 D=128 CQ-8 G=128  4.611s
      acc source   : TILE via compute_job.grade_compute_job (hw)
      s_c source   : TILE (captured s_c, via grade_compute_job)
      substituted  : True   consumed by model: True
      tile vs golden: acc_o bit-exact | o8 bit-exact | out_hat bit-exact
      rq (scale,shift): tile=(65532, 23) golden=(65532, 23)
  tile vs golden  : out_hat bit-exact 5/5 ; acc_o bit-exact 5/5
  token OFFLOAD ON : ids=[12095] text=' Paris' (170s)
  token OFFLOAD OFF: ids=[12095] text=' Paris' (145s)
  TOKEN IDENTITY: PASS   (offload-on == pure-golden)
  MILESTONE C   : PASS
```

Full session log: `hw_prompt_session.log` · machine record: `hw_result.json` ·
sample capture records (283 lines of tile-produced values):
`hw_sample_captures.jsonl` · clock/image state: `clkgen_after_recipe.txt`,
`hw_clkgen_and_image.txt`.

## ⚠️ The bring-up trap this run walked into (and past)

`fpga-load-local-image` left `clk_extra_a1` at **125.00 MHz** — 8× the tile's
closed clock. Only after `sudo fpga-load-clkgen-recipe -S 0 -a 2` did it read
**15.62 MHz**. The in-repo lock preflight matches the substring "lock" and
passes vacuously, so `remote_hw_exec.py` refuses to execute any job until
`fpga-describe-clkgen` *parses* a1 = 15.625 ± 0.2 MHz. Without that gate this
session would have produced garbage while reporting success.

Also recorded: the BAR0 identity probe returned bridge ID `0x41394558` = "A9EX"
and KVQ `INFO_DIM = 128`; the probe's own baked expectation for the ID was
mistyped by the operator, which is why the probe line reads FAIL — the *tile*
was correct, the *expectation* was wrong. Kept in the log rather than
retconned.

## Simulation runs (same code path, Verilated tile)

1. `--prompt "The capital of France is"` L0/H0 → ` Paris`, 5 ops, 990 caps,
   bit-exact 5/5, TOKEN IDENTITY PASS.
2. Independent re-verification, different prompt and different site:
   `"The largest planet in our solar system is"` L13/H7 → ` the`, 8 ops,
   1,788 caps, bit-exact 8/8, TOKEN IDENTITY PASS
   (`sim_verify_L13H7_planet.log`). The token itself is the 4-bit/fixed-point
   pipeline's own greedy output on both paths — token *identity* is the gate,
   not token quality.

## Scope and fences

- **T ≤ 128 total tokens**, enforced in-script: chunked heads cannot be
  offloaded (the mailbox does not expose per-chunk softmax state).
- One head op per step; the other 27 heads and all 27 other layers run in the
  golden pipeline on the host.
- The host contributes the inputs and golden's own `calib_requant` /
  `requant_i32_to_i8` epilogue; the tile contributes `acc_o` and `s_c`.
- Provenance is printed per op (`acc source` / `s_c source`) so a TILE vs
  GOLDEN substitution can never be silently confused.

## Cost

10 instance-minutes ≈ **$0.33** (f2.6xlarge @ $1.98/hr, priced from the AWS
Pricing API in `d_preflight.sh`). Instance terminated at session end.

## Reproduce

```
verif/f2sim/obj_d128_ddr0/f2sim              # built by: make build DDR=0
make -C verif/f2sim capgate                  # Stage-A read-back gate, one job
~/.venvs/apex-eval/bin/python scripts/fpga/f2/prompt_offload.py \
    --prompt "The capital of France is" --max-tokens 1 \
    --offload-layer 0 --offload-head 0 --executor sim
# hardware: docs/design/MILESTONE_D_RUNBOOK.md (preflight -> launch -> AGFI ->
# recipe A2 -> VERIFY 15.62 -> APEX_F2_HOST=... --executor hw -> terminate)
```


## ADDENDUM 2026-07-29 late: PROJECTION GEMM on silicon — the second op family

Same day, second session (i-031df4980911bd120, 9 min, $0.33, terminated):
a **real Qwen2.5-7B layer-0 Wq projection — all 3584 INT8 channels — computed
by the MXE on the FPGA**: K split into 28 k=128 descriptors + tail, weights
pushed on the external xw port (3,712 beats), raw INT32 accumulators captured
with **zero result expectations in the program**, host-summed per the C-KSPLIT
contract, graded bit-exact vs golden's wide matmul: **equal 8/8 lanes,
29/29 jobs, worst diff 0.0** (`hw_projection_k128.log`). Both discriminators
red-test (a tampered lane and a tampered weight byte each fail the grade).

**Image-parity lesson, kept loud** (`hw_projection_k2048_FAILED_*.log`): the
first attempt chained 16 stage rows per k=2048 descriptor — bit-exact in the
HEAD-built twin, WRONG on silicon, because the flying image (05efb2a) predates
`38ec95c` (stage-buffer LOAD honors `sel` as base row; before it, always
row 0). Root-caused by git diff, not guessed. Rule reaffirmed: nothing goes to
silicon without validation against an image-parity twin; k=2048 chaining stays
sim-only until the next image.

Combined claim now supported: **the two op families that dominate a
transformer — attention and projection GEMM — have both executed on our
design on real silicon, produce-mode, bit-exact, in one day.**
