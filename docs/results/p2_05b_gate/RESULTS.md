# P2 GATE — Qwen2.5-0.5B through the golden pipeline and the D=64 tile (2026-08-03)

VERDICT: **GO-WITH-CONDITIONS** for "most of a real 0.5B model's computation
served by the tile". Executed evidence below; conditions in §5.

## 1. Model (verified from the checkpoint, not assumed)

`mlx-community/Qwen2.5-0.5B-Instruct-4bit` (HF cache, 4-bit g64):
hidden 896, 24 layers, H=14, H_kv=2 (GQA group 7 — same ratio as 7B),
head_dim 896/14 = **64**, d_ffn 4864 = 76·64 = 38·128, rope_theta 1e6,
rms_norm_eps 1e-6, vocab 151936, **tie_word_embeddings = true** (7B: false;
`run_tinynpu.py:183` already handles it), q/k/v biases present (like 7B),
eos 151645 (scalar).

## 2. Golden pipeline: ZERO changes needed

`golden/apex_golden/{transformer,attention,compute,cq_codec}.py` are fully
geometry-parameterized (D/H/H_kv/head_dim/d_ffn flow from LayerWeights /
array shapes; `rmsnorm_fx_wide` takes any multiple of 128 — 896 = 7·128;
`gemm_i8_ksplit` K legal to 2^17−1 ≫ 4864). No file:line carries a 7B
assumption. `run_tinynpu.py` needed only CLI arguments:

```
python run_tinynpu.py --prepare --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --weights-dir build/s8_weights/Qwen2.5-0.5B-Instruct-4bit          # 6 s, 0.77 GB
python run_tinynpu.py --model ... --weights-dir ... \
    --prompt "The capital of France is" --max-new-tokens 4 \
    --trace-dir docs/results/p2_05b_gate/trace --trace-jobs 8 --ref-check 2
```

Measured (trace/run.json):
* generated: **" Paris. It is"** (ids 12095, 13, 1084, 374) — **4/4 greedy
  tokens identical to MLX** (`mlx_lm` greedy on the same checkpoint:
  "Paris. It is the largest city in"). ~1.2 s/token host golden.
* `--verify-trace`: **7/7 jobs replay bit-exact** (T=2..8, D=64, CQ-8).
* gamma Q2.13 folds engage (k up to 2, e.g. L23 g1 amax 12.38) — the fold
  machinery is load-bearing for 0.5B, compensation exact as designed.
* float64 yardstick: median per-layer e2e rel ≈ 7e-4 (step 0) / 0.039
  (step 1); worst layer 1.21 (L21, step 0, T=1) — the known noisy T=1
  corner; token agreement above is the end-to-end check.
* SiLU domain exceedances (|gate|>8 clamp): 799/933888 = 8.6e-4 (disclosed;
  clamp is part of the contract and mirrored by RTL, bit-exactness intact).
* 7B regression on the same tree: `--selftest` ALL PASS;
  `--verify-trace docs/results/s8_7b_token/artifact_trace` **18/18 bit-exact**.

## 3. Tile proof — real 0.5B attention jobs through the verilated CL, D=64

The 7 traced attention jobs (real 0.5B operands) compiled through the
verified L3 choreography and ran on the **D=64 cl_apex twin** (the tile's
default "b64 first-light" config — `cl_apex.sv:530 APEX_CL_D default 64`):

```
python scripts/fpga/f2/trace_to_regops.py --trace docs/results/p2_05b_gate/trace \
    --out build/p2_05b_regops
python docs/results/p2_05b_gate/retarget_info_tier.py     # see below
cd verif/f2sim && make build D=64 DDR=0
make run D=64 DDR=0 REGOPS="../../build/p2_05b_regops_b64cl/job_*.regops.jsonl"
```

**F2SIM RESULT: files=7 checks=1052 fails=0 → PASS** (f2sim_run_d64_div5.log)
— KVQ records, C-1 feeder codes+scales, Q·K̂ᵀ INT32 accs, score-dequant,
ASU softmax, P·V̂ accs, C-2 epilogue o8: all bit-exact vs golden at
head_dim=64. trace_to_regops' own cross-check also asserted golden o8 ==
stored trace per job at compile time.

Disclosed retarget (retarget_info_tier.py): ONE expectation per job — the
INFO_TIER build-identity read (BAR0 0x1014) — is 0x7 in the L3 d64
*reference* build (CQ-8/4/4+) but 0x3 on the CL b64 (`cl_apex.sv`
pins KVQ_OUTLIER_K=0, no CQ-4+ engine). The jobs are CQ-8; no datapath
expectation was touched. First run (unretargeted) failed exactly those 7
identity reads and nothing else — the checks demonstrably bite.

## 4. Tile-side inventory (file:line receipts)

| Item | Status | Receipt |
|---|---|---|
| head_dim=64 attention | **WORKS AS-IS on a D=64 build** (executed §3). A D=128 image REFUSES hd=64: descriptor check `head_dim != cfg_d` → WALK_ERR_DESC | rtl/seq/seq_walker_pkg.sv:509-510 (fmt=1), :148/:181 (v1); score composite √D baked per build (seq_walker_comp.sv:45 — D=64 constant 2^7 EXACT) |
| C-1 feeder at 64 | WORKS AS-IS (FEEDER_D_LEGAL = 64/128) | rtl/seam/seam_feeder_quant.sv:100-102 |
| D_model=896 model rows | WORKS AS-IS as 14×64 frames on b64 (CFG_DM=CFG_D; cl_apex never overrides CFG_DM). 7×128 framing NEEDS CONFIG: plumb APEX_CL_DM→CFG_DM split (CFG_D=64/CFG_DM=128 legal per g_chk_dm) | rtl/top/apex_top.sv:109-124, :1854; scripts/fpga/f2/gen_layer_ops.py:225-233 |
| Wide RMSNorm 896 | 896 = 7·128 LEGAL wide row. In-tile single-row: NEEDS CONFIG +define+APEX_CL_DM=896 (RMS_D_MAX). On the narrow build: ext_sum chunk composition, proven at 3584 (norm2c) | rtl/asu/asu_rmsnorm.sv:10-14, :62-73; cl_apex.sv:544-548 |
| Residual 896 | window stride 1024 ⇒ ONE window job covers 896 (7B needed 4). NEEDS CONFIG LAYER_DM_MAX≥896 (same APEX_CL_DM knob) | rtl/top/glue/apex_residual.sv:16-19,52; apex_top.sv:1529 |
| SwiGLU d_ffn=4864 | WORKS AS-IS: 4864 = 76·64 = 38·128, chunking divides exactly (walker chunk 64; host framing % D_TILE == 0) | rtl/seq/seq_walker_pkg.sv:377; gen_layer_ops.py:875 |
| GQA 14:2 kv_map | fmt=1 legality ✓ (14%2=0, 14·64=896); per-KV-head engines NEED CONFIG +define+APEX_CL_GQA=2 (N_ENG≥n_kv_heads; N_ENG legal 1..16) | seq_walker_pkg.sv:513-519; seq_layer_walker2.sv:271-272,747-748; cl_apex.sv:541 |
| Walker H=14 / RQ | WORKS AS-IS: H=14 ≤ WALK2_H_MAX=30; RQ slots needed H+2=16 ≤ 32 | seq_walker_pkg.sv:198, 210-211 |
| Multi-head q staging | NEEDS CONFIG/RTL-glue: QSTAGE_H_MAX param exists (≤30, ≤STAGE_R_MAX=31 ✓ for 14) but cl_apex does NOT plumb it (defaults 1) | apex_top.sv:138-148; cl_apex.sv (no QSTAGE reference) |
| T envelope | T≤128/job (WALK_T_MAX=T_ROW_MAX=128); KVQ_DEPTH 256 = 128 K-rec + 128 V-rec ✓; longer T = C-CHUNK host merge (already golden) | seq_walker_pkg.sv:44; apex_top.sv:149-156; cl_apex.sv:259 |
| Embeddings / LM head | host-side, same as 7B S8 (tied head handled at prepare) | run_tinynpu.py:183-189 |

## 5. Conditions attached to the GO

1. **One build serves everything**: b64 CL (CFG_D=64) with +define+APEX_CL_DM=896
   (wide norm / residual / one-window row) and +define+APEX_CL_GQA=2. All are
   existing parameters; only APEX_CL_DM→CFG_DM (if 128-framing wanted) and
   QSTAGE_H_MAX need new one-line cl_apex plumbing. ~0.5 day + regression.
2. **Host choreography at 0.5B geometry**: layer_offload/gen_layer_ops assume
   D_TILE=128 (`gemm_job.py:114`, `build_rope_stage` asserts hd==D_TILE) —
   a NEW 0.5B driver (new files; those files are owned elsewhere) must pass
   geometry instead. ~1-2 days to a C2-equivalent "all six op families of one
   0.5B layer" run in sim. Full-width 0.5B layer ≈ 29k GEMM jobs
   (vs ~300k for 7B) ≈ ~3 h in sim at the measured 0.38 s/job — feasible,
   vs "days" for 7B.
3. **Walker (fmt=1) full-layer mode** stays future scope for 0.5B exactly as
   for 7B (I-B capability gaps; walked FULL layer not achievable on today's
   RTL) — the demonstrated path is host-sequenced, per the C2 recipe.
4. Session T ≤ 128 per job envelope (sprint fence T≤128 unaffected).

## 6. Files

* `trace/` — the 7 replayable 0.5B jobs + manifest + run.json (weights cache
  regenerable in 6 s via --prepare; build/ artifacts gitignored by design).
* `retarget_info_tier.py` — the disclosed identity-check retarget.
* `f2sim_run_d64_div5.log` — the 0-fail twin run.
