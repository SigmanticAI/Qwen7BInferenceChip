#!/usr/bin/env python3
# wd_ksplit_05b.py — THE 0.5B DOWN-PROJECTION AT D=64: the K-SPLIT GEMM the
# D-blind chunk made tile-illegal, executed on the simulated b64 tile and
# graded bit-exact against golden's own gemm_i8_ksplit.
#
#   python3 scripts/fpga/f2/wd_ksplit_05b.py selftest   # offline, seconds
#   python3 scripts/fpga/f2/wd_ksplit_05b.py run        # the sim gate
#
# ══ WHY THIS FILE EXISTS (third erratum, W1 2026-08-05) ════════════════════
# seq_walker_fmt.K_JOB / seq_walker_pkg::WALK2_K_JOB chunked every walked
# projection's k at 2048 — the MXE's own ceiling (mxe_ctrl.sv:162) — for
# EVERY D. But a k-deep job's ACTIVATION arrives through the act stage
# buffer (apex_top.sv:2203, apex_stage_buf #(.D(CFG_DM), .R_MAX(31))), one
# CFG_DM-wide row per stage row, and that unit refuses rows > 31
# (apex_stage_buf.sv:167/:103-104). At D=64 a 2048-chunk is 32 rows: for
# Qwen2.5-0.5B only the down projection (Wd, k_total = d_ffn = 4864) has
# k > 2048, so a fmt=1 full-layer walk emitted two TILE-ILLEGAL descriptors
# at DOWN. The fix makes the chunk D-aware (fmt.k_job: 1984 at D=64 = 31
# whole stage rows; 2048 at D=128, byte-identical to the legacy streams).
#
# ══ THE CLAIM, EXACTLY ═════════════════════════════════════════════════════
#   * the WEIGHTS are the real golden-prepared Qwen2.5-0.5B L0 Wd tensor
#     (build/s8_weights, int8, exactly as golden stores them);
#   * the ACTIVATION codes are golden's own quantization of the real swiglu
#     row of the committed 0.5B prompt's last prefill step — the exact codes
#     golden's _proj_epilogue contracts with Wd;
#   * each n-block's K=4864 contraction runs as ceil-chunks of at most
#     fmt.k_job(64) = 1984 (31 stage rows) — separate host-pushed GEMM jobs
#     on the verilated b64_05b cl_apex twin, raw INT32 accumulators taken as
#     `cap` egress (zero baked expectations);
#   * the K-SPLIT RECOMBINATION is golden C-KSPLIT's host-side arm
#     (compute.py:43-44 "INT32 partial accumulators are summed — host-side,
#     or on-tile via the accumulate flag"): the host sums the per-chunk
#     partials with the INT32-prefix assert and grades the total bit-exact
#     against cp.gemm_i8_ksplit AND an independent int64 einsum;
#   * the PRE-FIX shape is demonstrated REFUSED: a 2048-deep chunk at D=64
#     cannot even be scripted (gemm_job.build_script's 32 > STAGE_R_MAX
#     refusal — the same bound apex_stage_buf enforces with job_error), and
#     seq_walker_fmt.assert_act_stageable rejects it by the stage buffer's
#     own rule.
#
# WHY HOST-SIDE RECOMBINATION AND NOT ONE ON-TILE ACCUMULATE CHAIN: the
# OS-accumulate chain requires every staged act row RESIDENT before the
# first descriptor fires — staging jobs are OP_GEMM_WS k=2 descriptors and
# ALWAYS clear the array accumulators (gemm_job.py build_chain_script note;
# mxe_ctrl.sv:286) — and one bank holds 31 rows. K=4864 is 76+ rows, so no
# single resident chain exists on this tile; the walked one-kick DOWN step
# (feeder-route re-staging between chunks) is the walked-full-layer lane's
# work, not this proof's. This file proves the DECOMPOSITION is one the
# tile ACCEPTS and that the recombined result is golden-exact.
#
# NOTE the staging framing: gemm_job.stage_plan re-lays the activation so
# squant MODE_QUANT is the identity (the F1-F3 asserts); the staged K may
# exceed the real K (duplicated amax lane, zero-padded W rows). The F3
# identity makes the TOTAL product exact; per-chunk partials are partials
# of the STAGED order. The walker-seam decomposition itself (fmt.jobs at
# cfg_d=64, [1984, 1984, 896] per n-block) is proven in selftest here and
# executed end-to-end by verif/seq_walker `make run_l05b`.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif/top/l3"),
           str(REPO / "verif/seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seq_walker_fmt as fmt                                   # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import compute as cp                          # noqa: E402

import gemm_job as gj                                          # noqa: E402
import tile_exec_bridge as bridge                              # noqa: E402
import tile_geom as tg                                         # noqa: E402

D_TILE_05B = 64
DEFAULT_WEIGHTS = REPO / "build/s8_weights/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_WORK = REPO / "build/wd_ksplit_05b"
CAPS_PER_JOB = gj.MXE_N + 1                    # 8 RO lanes + the meta word


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═════════════════════════ the offline proofs ══════════════════════════════

def selftest() -> int:
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("WD_KSPLIT_05B SELFTEST (offline, no executor)")
    # 1. the D-aware bound itself, against the consumers' own numbers
    chk("1 fmt.k_job(64) = 1984 = 31 whole stage rows",
        fmt.k_job(64) == 1984 == 31 * 64)
    chk("1b fmt.k_job(128) == the legacy K_JOB (D=128 byte-identity)",
        fmt.k_job(128) == fmt.K_JOB == 2048)
    chk("1c the host lane's own bound agrees (tile_geom.rows_per_desc_for)",
        tg.rows_per_desc_for(tg.IMG_05B, 64, executor="sim") * 64
        == fmt.k_job(64))

    # 2. the whole 0.5B family decomposes into descriptors the tile accepts
    n_jobs = 0
    for (kt, nt, nm) in [(896, 896, "Wq"), (896, 128, "Wk"), (896, 128, "Wv"),
                         (896, 896, "Wo"), (896, 4864, "Wg"),
                         (896, 4864, "Wu"), (4864, 896, "Wd")]:
        for (_, _, k, n, _a) in fmt.jobs(kt, nt, cfg_d=D_TILE_05B):
            fmt.assert_mxe_legal(1, k, n, f"0.5B {nm}")
            fmt.assert_act_stageable(k, D_TILE_05B, f"0.5B {nm}")
            n_jobs += 1
    chk("2 whole 0.5B family MXE-legal AND stageable at D=64", True,
        f"{n_jobs} jobs")
    js = fmt.jobs(4864, 896, cfg_d=D_TILE_05B)
    perk = [k for (_n0, _k0, k, _n, _a) in js[:3]]
    chk("2b Wd per-n-block k-split is [1984, 1984, 896], acc 0/1/1",
        perk == [1984, 1984, 896]
        and [a for (_, _, _, _, a) in js[:3]] == [0, 1, 1]
        and len(js) == 112 * 3)

    # 3. the PRE-FIX shape is refused — by the mirror's stage-buffer rule...
    try:
        fmt.assert_act_stageable(2048, 64, "pre-fix chunk")
        chk("3 assert_act_stageable rejects k=2048 @ D=64", False,
            "no refusal")
    except fmt.ActUnstageable as e:
        chk("3 assert_act_stageable rejects k=2048 @ D=64",
            "32 stage rows" in str(e))
    # ...and by the host job builder's own STAGE_R_MAX refusal (the same
    # bound apex_stage_buf enforces in RTL with job_error)
    with tg.at_d(D_TILE_05B):
        try:
            gj.build_script("prefix_refusal",
                            np.zeros((2048, 8), dtype=np.int64),
                            np.full(2048, 127, dtype=np.int64))
            chk("3b gemm_job.build_script refuses a 2048-deep chunk at D=64",
                False, "no refusal")
        except AssertionError as e:
            chk("3b gemm_job.build_script refuses a 2048-deep chunk at D=64",
                "exceed STAGE_R_MAX" in str(e), str(e)[:70])

    print("=" * 60)
    if fails:
        print(f"WD_KSPLIT_05B SELFTEST FAIL: {fails}")
        return 1
    print("WD_KSPLIT_05B SELFTEST: ALL PASS")
    return 0


# ═════════════════════════ the real activation ═════════════════════════════

def real_wd_operands(weights_dir: Path, layer: int):
    """The real L<layer> Wd tensor + golden's own INT8 codes of the real
    swiglu row it contracts (the committed 0.5B prompt, last prefill step).
    Golden composes; this file replays its quantizer on its own field."""
    import fuel_proj_05b as fp                 # golden_prefill + PROMPT_IDS
    import run_tinynpu as rt
    model = rt.GoldenModel(weights_dir)
    assert int(model.meta["head_dim"]) == D_TILE_05B
    fxs = fp.golden_prefill(model, fp.PROMPT_IDS, rt.TIER_MAP["kvq8"], 128)
    fx = fxs[layer]
    a8, _s = at.quant_rows_i8(np.asarray(fx.swiglu, dtype=np.float64)[None, :])
    x8 = np.asarray(a8[0], dtype=np.int64)
    W = np.load(weights_dir / f"L{layer:02d}_Wd.npy")
    assert W.shape == (4864, 896) and W.dtype == np.int8, W.shape
    assert x8.shape[0] == W.shape[0], (x8.shape, W.shape)
    return W, x8


# ═════════════════════════════ run + grade ═════════════════════════════════

def run_block(W, x8, n0: int, work: Path, binary: Path, layer: int) -> dict:
    """One n-block: K=4864 x 8 columns as ceil-chunks of <= fmt.k_job(64),
    each a separate host-pushed GEMM on the sim tile; host-side C-KSPLIT
    recombination; golden computed AFTER the runs from the same operands."""
    blk = np.asarray(W[:, n0 * 8:(n0 + 1) * 8], dtype=np.int64)
    with tg.at_d(D_TILE_05B):
        plan = gj.stage_plan(x8, blk)
        rpd = tg.rows_per_desc_for(tg.IMG_05B, D_TILE_05B, executor="sim")
        chunks = plan.chunks(rpd)
    # every chunk must sit inside BOTH consumer bounds — checked by the
    # INDEPENDENT rules, not by the constants that cut the chunks
    for (_r0, nr, K) in chunks:
        fmt.assert_act_stageable(K, D_TILE_05B, f"Wd n0={n0}")
        fmt.assert_mxe_legal(1, K, 8, f"Wd n0={n0}")
        assert nr <= gj.STAGE_R_MAX
    out = {"n0": n0, "chunks": [int(K) for (_r, _n, K) in chunks],
           "rows": int(plan.rows), "K_real": int(x8.shape[0]),
           "K_staged": int(plan.K_total), "jobs": []}
    partials = []
    for ci, (r0, nr, K) in enumerate(chunks):
        a = r0 * D_TILE_05B
        name = f"wd_L{layer:02d}_n{n0:03d}_k{ci}"
        with tg.at_d(D_TILE_05B):
            path, man = gj.build_gemm_job_full(plan.Wst[a:a + K],
                                               plan.xst[a:a + K],
                                               work / "jobs", name)
        tg.audit_geometry(path, D_TILE_05B)
        tg.retarget_info_tier(path, tg.IMG_05B)
        t0 = time.perf_counter()
        r = bridge.run_job(path, executor="sim", binary=str(binary),
                           tile_div=bridge.TILE_DIV_DEFAULT, timeout_s=3600)
        wall = time.perf_counter() - t0
        if not (r["ok"] and len(r["captures"]) == CAPS_PER_JOB):
            raise SystemExit(
                f"REFUSE: {name}: rc={r['rc']} ok={r['ok']} "
                f"caps={len(r['captures'])} (want {CAPS_PER_JOB}) — the tile "
                f"did not accept/complete this descriptor; log: {r['log'][-800:]}")
        acc = gj.decode_acc(r["captures"])
        partials.append(acc)
        out["jobs"].append({"job": name, "k": int(K), "rows": int(nr),
                            "regops": man["regops"], "checks": man["checks"],
                            "acc_i32": [int(v) for v in acc],
                            "wall_s": round(wall, 1)})
        eprint(f"    {name}: k={K} rows={nr} ACCEPTED, "
               f"{man['checks']} in-program checks, {wall:.1f}s")
    # the K-split recombination: golden C-KSPLIT's host-side arm
    total = gj.accumulate_partials(partials)
    gold_ks = cp.gemm_i8_ksplit(x8[None, :], blk)[0].astype(np.int64)
    gold_wide = np.einsum("k,kn->n", x8, blk)
    assert np.array_equal(gold_ks, gold_wide), \
        "golden disagrees with itself (gemm_i8_ksplit vs int64 einsum)"
    out["acc_i32"] = [int(v) for v in total]
    out["golden_i32"] = [int(v) for v in gold_ks]
    out["bit_exact"] = bool(np.array_equal(np.asarray(total, dtype=np.int64),
                                           gold_ks))
    out["n_lane_diff"] = int(np.sum(np.asarray(total, dtype=np.int64)
                                    != gold_ks))
    return out


def run(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary) if args.binary else tg.IMG_05B.binary
    if not binary.exists():
        eprint(f"REFUSE: the b64_05b sim twin is not built at {binary}.\n"
               f"  cd verif/f2sim && make build D=64 OBJ={tg.IMG_05B.obj} "
               f"VFLAGS_EXTRA=\"{tg.IMG_05B.defines()}\"")
        return 2
    if selftest():
        return 1
    eprint(f"[1] golden prefill (real 0.5B, layer {args.layer}) …")
    W, x8 = real_wd_operands(Path(args.weights_dir), args.layer)
    n0s = [int(v) for v in args.blocks.split(",")]
    eprint(f"[2] running Wd n-blocks {n0s} on {binary.parent.name} "
           f"(3 K-chunks each) …")
    results = []
    for n0 in n0s:
        assert 0 <= n0 < W.shape[1] // 8, f"n0={n0} outside 0..111"
        results.append(run_block(W, x8, n0, work, binary, args.layer))
    verdict = all(r["bit_exact"] for r in results)
    rep = {"tile": "verilated cl_apex b64_05b (D=64, CFG_DM=64, GQA=2)",
           "tensor": f"L{args.layer:02d}_Wd (4864 x 896), real 0.5B weights",
           "activation": "golden quant_rows_i8 of the real swiglu row "
                         "(committed prompt, last prefill step)",
           "k_job_bound": fmt.k_job(D_TILE_05B),
           "blocks": results, "all_bit_exact": verdict}
    (work / "result.json").write_text(json.dumps(rep, indent=1))
    for r in results:
        print(f"  n0={r['n0']:3d}: chunks={r['chunks']} "
              f"(staged rows {r['rows']}) -> "
              f"{'BIT-EXACT' if r['bit_exact'] else 'MISMATCH '} "
              f"({r['n_lane_diff']} lanes differ)")
    n_jobs = sum(len(r["jobs"]) for r in results)
    print(f"WD_KSPLIT_05B RESULT: blocks={len(results)} tile_jobs={n_jobs} "
          f"k_bound={fmt.k_job(D_TILE_05B)} -> "
          f"{'PASS' if verdict else 'FAIL'} (recombined accumulators vs "
          f"golden gemm_i8_ksplit)")
    return 0 if verdict else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    r.add_argument("--work", default=str(DEFAULT_WORK))
    r.add_argument("--binary", default=None)
    r.add_argument("--layer", type=int, default=0)
    r.add_argument("--blocks", default="0,56,111",
                   help="comma-separated Wd n-block indices (0..111)")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
