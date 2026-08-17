# S3 — THE FULL PROJECTION SWEEP ON SILICON (1024/1024 blocks bit-exact)

**Date:** 2026-07-30 (evening session) · **Branch:** `comp/prompt-b-c` ·
**Instance:** `i-013437f5cefdaca4b` (f2.6xlarge, us-west-2b, ~2.9 h ≈ $5.75
total session incl. the S2 validation) · **AGFI:**
`agfi-0ae06ea568e5667ba` (the C1/D image, shell 0x10212415) · **Driver:**
`scripts/fpga/f2/proj_sweep_batched.py` · **Machine records:**
`proj_sweep_result.json` (this directory), `hw_batch_attrib_proof.json`,
`hw_s2_sweep_session.log`; the full per-batch capture record
`build/hw_s2_sweep/sweep/batches.jsonl` (33.6 MB, kept out of git,
sha256 `cc0c8b759ad94cd7fa0018d50bc4c87aa164fe8b6e219c7cf42a54b87f79b5b1`).

## The claim (exactly)

**Every 8-column block of all four projection matrices — Wq (448 blocks),
Wk (64), Wv (64), Wo (448); 1024 blocks, 8,192 output columns, full
K=3584 contraction — of layer 0 of a real Qwen2.5-7B decode step (the
committed S8 prompt, final step) was computed by the tile on real F2
silicon as raw INT32 accumulators, and every one is bit-exact against
golden: 1024/1024 green, 0 red.** This upgrades C1's projection evidence
from 8 sampled blocks (0.78% of output columns) to **100%, exhaustive**.

What this is NOT: not a full layer (RoPE/residual/RMSNorm/SwiGLU remain
sim-only pending the wide image — aws/aws-fpga#799 **[SUPERSEDED
2026-07-31/08-01: all four subsequently flew WITHOUT the wide image — see
`NARROW_LANE_RESULT.md` and `SIX_OF_SIX_RESULT.md`]**), not multi-layer, not
a throughput claim (15.625 MHz is a correctness clock).

## Numbers

| quantity | value |
|---|---|
| tile jobs | **29,696** produce-mode K-split jobs (28 per block × 1024 + the k-chunk tails), `rows_per_desc=1` (the 05efb2a image-parity rule) |
| verdict | **1024/1024 blocks bit-exact** vs golden (raw INT32 accumulator equality; golden computed AFTER the runs from the same INT8 operands) |
| batches | 68 executor invocations across 4 resume passes, **68/68 ok** (every file complete + attribution proven) |
| job-stream wall | **36.0 min total = 0.073 s/job aggregate** |
| vs separate invocations | measured 3.63 s/job separate (n=8, matches C1's 3.73 median) → **~50× on the job stream**; unbatched this sweep is ~30 h |
| clock | clkgen recipe A2 verified **15.62 MHz by number** before the session AND per-invocation by `remote_hw_exec`'s numeric gate — 68 more gates, all green |

## Gates that make this evidence (not just output)

1. **Produce-mode, audited**: every program carries `cap` reads with ZERO
   baked output expectations — `audit_program` gated the build phase
   (0 violations over 29,696 programs) before anything ran.
2. **Attribution proven per file**: boundary markers + per-file manifests
   (batch_exec's two independent mechanisms); a file is only credited when
   its observed capture sequence equals its programmed manifest inside its
   marker window. 29,696/29,696 complete and attributed.
3. **Golden after the fact**: block golden = `gemm_i8_ksplit` AND an
   independent wide int64 matmul, asserted equal to each other, computed
   from the same operands after the hardware ran.
4. **Sim proof before silicon**: the driver went to hardware only after a
   sim run graded 2 blocks bit-exact AND matched the UNBATCHED
   `ksplit_matvec` accumulators on the same block (cross-check),
   `build/hw_s2_sweep/simproof*/`.
5. **Refuse-loudly transport**: a batch that executes nothing (0 boundary
   markers) aborts the sweep; nothing is ever credited from a dead
   transport.

## Session incidents (all caught by the gates, all fixed in-tree)

1. **Upload timeout at N=512** (first attempt): scp's per-file protocol
   exchange × 1025 files blew the 120 s control timeout — every batch
   "ran" for a uniform 121 s and executed NOTHING; batch_exec's
   zero-marker refusal caught it. Fix: tar-pipe upload (one stream) +
   `APEX_F2_CTL_TIMEOUT`; measured 64-job batch upload+run at 8.2 s.
2. **Mid-session ISP IP rotation**: the Mac's public IP changed and fell
   out of the security group at batch 40/58 — the cap fetch raised, the
   driver died, but per-batch persistence meant `--resume` lost nothing.
   Fix: new /32 SG rule (the SG's existing rule list shows the same
   maintenance by past sessions) + driver retry + resume.
3. **Transient scp resets (rc=255)**: intermittent residential-uplink
   flakiness; fixed operationally with an outer resume-until-done wrapper,
   smaller (256-job) batches, and ssh keepalive/ConnectionAttempts opts.
   Final pass ran 20/20 batches clean.

Nothing in any incident touched result integrity: a failed batch runs
nothing and credits nothing; only marker-bounded, manifest-matched
captures are graded.

## Teardown

`terminate-instances` → `wait instance-terminated` → describe reads
**terminated**; account-wide F-instance sweep afterwards: none left.
