#!/usr/bin/env python3
# sum2_mxe.py — CAN THE MXE COMPUTE RMSNORM'S SUM-OF-SQUARES?  (measured)
#
# ══ THE QUESTION ═══════════════════════════════════════════════════════════
# golden's `rmsnorm_fx_wide` (golden/apex_golden/attention.py:124) needs
#     sum2 = SUM_i x[i]^2      over ALL D_model INT8 codes of the norm input
# accumulated in INT32 (its own bound: sum2 <= 2^27). asu_rmsnorm.sv can only
# do this over a D <= 128 pow-2 chunk, which is why the wide form exists and
# why "the host (or a scalar adder) accumulates chunk sums" is written into
# the golden docstring as a TODO for the RTL.
#
# The tile already has an INT8xINT8 -> INT32 systolic GEMM whose K-split to
# K=3584 is PROVEN ON SILICON (docs/results/prompt_on_chip/PROJ_SWEEP_RESULT
# .md; scripts/fpga/f2/gemm_job.py:ksplit_matvec). A dot product of the row
# WITH ITSELF is exactly sum(x^2). So: stage the row as the activation AND as
# the (single-column) weight block, and the MXE's raw INT32 accumulator lane 0
# IS sum2 — if nothing in the staging, the sign handling, or the K-split
# recombination breaks it.
#
# ══ THE FOUR THINGS THAT COULD HAVE BROKEN IT (all four checked here) ══════
# 1. OPERAND STAGING. gemm_job.stage_plan duplicates channel k* = argmax|x8|
#    into lane 0 of EVERY 128-lane stage row so squant MODE_QUANT is provably
#    the identity, and neutralises the duplicates ON THE WEIGHT SIDE (row 0
#    keeps the real weight row, every later duplicate and every tail pad gets
#    a ZERO weight row). With W = x8 itself that is still exact: the staged
#    product is SUM over USED lanes of x8[chan]*x8[chan] = SUM_k x8[k]^2, and
#    stage_plan's own F3 assertion (xst @ Wst == x8 @ W) proves it per run.
#    NOTE the asymmetry is real and load-bearing: the ACTIVATION side carries
#    x8[k*] in all 29 rows; only the WEIGHT side is masked. A "square it by
#    squaring the staged activation" shortcut would over-count k* 29 times.
# 2. INT8 RANGE / SIGN. stage_plan requires |x8| <= 127 on BOTH operands, and
#    F1 requires max|x8| == 127 exactly. A -128 code would be legal INT8 but
#    is REFUSED as a weight (asserted, not silently clipped). The RO lanes are
#    sign-extended by cap_decode.ro_lanes_to_i32, so a negative x8 contributes
#    a POSITIVE square only if the tile's multiply is genuinely signed — which
#    is why sum2 > 0 with 1748 negative codes is itself evidence.
# 3. K-SPLIT RECOMBINATION. 3584 real channels -> 29 staged rows -> chunks of
#    <= K_MAX=2048. gemm_job.accumulate_partials asserts every INT32 PREFIX
#    stays in range (C-KSPLIT). The chunk boundaries are NOT golden's chunk
#    boundaries (2048 vs 128) — integer addition is associative, and this file
#    proves that empirically by running BOTH chunkings and comparing.
# 4. THE IMAGE-PARITY RULE. `rows_per_desc=1` when executor='hw': the flying
#    image (05efb2a) predates the stage-buffer base-row change, so k=2048
#    chaining is HEAD-sim-only — it was bit-exact in sim and WRONG on silicon
#    (gemm_job.py:33-35, breadth_step.py:447-450, proj_sweep_batched.py:132).
#    mxe_sum2() picks rows_per_desc=1 for hw AUTOMATICALLY, and --check runs
#    the hw-legal chunking IN SIM too, so the number claimed for silicon is
#    produced by the chunking silicon will actually use.
#
# ══ WHAT IS GRADED, AND BY WHOM ════════════════════════════════════════════
# golden is the only arbiter. The tile program contains NO expectation of the
# accumulator (gemm_job.GemmXlate turns the RO lanes into `cap` egress and
# _audit_no_result_expectations proves it BY ADDRESS). After the run:
#   A. sum2_mxe == golden's OWN chunked accumulation (the 28x128 loop of
#      rmsnorm_fx_wide, replayed line-for-line here incl. its per-chunk
#      2^22 width proof and its 2^27 total bound);
#   B. feeding sum2_mxe through the REST of golden's wide-RMSNorm formula
#      reproduces attention.rmsnorm_fx_wide(x, gamma)'s (y, r, nrm) EXACTLY —
#      i.e. the MXE number is not merely equal, it DRIVES the arbiter's own
#      normalisation to the same fixed-point row.
# Two discriminators must go red, or the check refuses to call itself a proof.
#
# CLI:
#   python3 scripts/fpga/f2/sum2_mxe.py --selftest          # host-only
#   python3 scripts/fpga/f2/sum2_mxe.py --check \
#       --binary verif/f2sim/obj_d128_ddr0/f2sim            # SIM proof
#   python3 scripts/fpga/f2/sum2_mxe.py --check --executor hw   # silicon
#
# API:
#   sum2_weight_block(x8)            -> W[K,1] (the row as its own weight)
#   golden_sum2_chunked(x8, chunk)   -> int   (golden's own accumulation)
#   norm_from_sum2(sum2, x8, gamma)  -> (y, r, nrm) via golden's own tail
#   mxe_sum2(x8, ...)                -> dict (partials, sum2, grade, ...)

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gemm_job as gj                                            # noqa: E402
import tile_exec_bridge as bridge                                # noqa: E402
from apex_golden import attention as at                          # noqa: E402

DEFAULT_NPZ = REPO / "build" / "f2_layer" / "layer_step_full.npz"
DEFAULT_OUT = REPO / "build" / "f2_sum2"
GOLDEN_CHUNK = 128               # rmsnorm_fx_wide's own chunking


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ══ 1. the operands ════════════════════════════════════════════════════════
def sum2_weight_block(x8) -> np.ndarray:
    """The row as its own single-column weight block: W[k, 0] = x8[k].

    The MXE contracts acc[n] = SUM_k act[k] * W[k][n]; with W[:,0] = the same
    row, acc[0] = SUM_k x8[k]^2 = golden's sum2. Columns 1..7 are the MXE's
    own zero padding (gemm_job.build_script) and MUST read back 0 — a free
    check that the array is not smearing lanes.
    """
    x8 = np.asarray(x8, dtype=np.int64)
    assert x8.ndim == 1, f"expected a row, got shape {x8.shape}"
    bad = np.flatnonzero(np.abs(x8) > 127)
    assert bad.size == 0, (
        f"{bad.size} code(s) outside INT8-symmetric [-127,127], first "
        f"x8[{int(bad[0])}]={int(x8[bad[0]])} — the MXE weight port takes the "
        f"same symmetric range as the activation port; a -128 code cannot be "
        f"pushed as a weight and is REFUSED here rather than clipped")
    return x8[:, None].copy()


# ══ 2. golden, replayed ════════════════════════════════════════════════════
def golden_sum2_chunked(x8, chunk: int = GOLDEN_CHUNK) -> int:
    """sum2 exactly as attention.rmsnorm_fx_wide:150-158 accumulates it.

    Same cut (chunks of `chunk`), same per-chunk 22-bit width proof, same
    total bound. This is the number the MXE must reproduce.
    """
    x = [int(v) for v in np.asarray(x8, dtype=np.int64)]
    d = len(x)
    if d <= 128 and (d & (d - 1)) == 0:
        chunks = [x]
    else:
        assert 1 <= chunk <= 128 and (chunk & (chunk - 1)) == 0
        assert d % chunk == 0, f"wide D={d} must be a multiple of chunk={chunk}"
        chunks = [x[a:a + chunk] for a in range(0, d, chunk)]
    sum2 = 0
    for c in chunks:
        s2c = sum(int(xi) ** 2 for xi in c)
        assert s2c < 2 ** 22, "per-chunk 22-bit width proof (asu_rmsnorm.sv)"
        sum2 += s2c
    assert sum2 <= 2 ** 27, "C-RMSW host accumulator bound (D <= 8192)"
    return int(sum2)


def tail_sensitivity(x8, gamma, *, limit: int = 1 << 24) -> int:
    """The SMALLEST sum2 error that changes golden's normalisation output.

    The wide-RMSNorm tail is floor-heavy (mean2 = (sum2*mu)>>(s+16), then
    isqrt, then 2^13//den), so it ABSORBS small sum2 errors. Measuring that
    absorption is the reason the primary grade in this file is the exact
    INT32 equality against golden's chunked sum2 and NOT the y/r/nrm match:
    the y/r/nrm match is necessary, not sufficient. Returns the measured
    threshold delta (or `limit` if nothing inside the search moved it).
    """
    x8 = np.asarray(x8, dtype=np.int64)
    ref = at.rmsnorm_fx_wide([int(v) for v in x8],
                             [int(v) for v in np.asarray(gamma,
                                                         dtype=np.int64)])
    base = golden_sum2_chunked(x8)
    d = 1
    while d < limit and norm_from_sum2(base + d, x8, gamma) == ref:
        d += 1 if d < 64 else max(1, d // 8)
    if norm_from_sum2(base + d, x8, gamma) == ref:
        return int(limit)
    lo, hi = 1, d                                # exact threshold by bisection
    while lo < hi:
        mid = (lo + hi) // 2
        if norm_from_sum2(base + mid, x8, gamma) == ref:
            lo = mid + 1
        else:
            hi = mid
    return int(lo)


def norm_from_sum2(sum2: int, x8, gamma):
    """golden's rmsnorm_fx_wide TAIL, driven by an EXTERNAL sum2.

    Byte-for-byte attention.rmsnorm_fx_wide:161-175 with its accumulation
    loop replaced by the caller's `sum2`. Returns (y, r, nrm). If the tile's
    sum2 is right this must equal the arbiter's own output.
    """
    x = [int(v) for v in np.asarray(x8, dtype=np.int64)]
    g = [int(v) for v in np.asarray(gamma, dtype=np.int64)]
    d = len(x)
    assert len(g) == d
    mu, s = at._wide_mu(d)
    mean2 = (int(sum2) * mu) >> (s + 16)
    n2 = mean2 + 1                              # EPS_INT = 1
    den = math.isqrt(n2)
    r = (1 << 13) // den                        # UQ2.13
    nrm = min(den, 2 ** 14 - 1)
    y = []
    for i in range(d):
        p = x[i] * g[i]
        q = p * r
        t = q >> 18
        rem = q & (2 ** 18 - 1)
        t += 1 if (rem > 2 ** 17 or (rem == 2 ** 17 and (t & 1))) else 0
        y.append(max(-32768, min(32767, t)))
    return y, int(r), int(nrm)


# ══ 3. the reusable entry point ════════════════════════════════════════════
def mxe_sum2(x8, *, out_dir=None, name="sum2", executor: str = "sim",
             binary=None, rows_per_desc: int | None = None,
             timeout_s: int = 3600, verbose: bool = True,
             gamma=None) -> dict:
    """SUM_k x8[k]^2 on the tile's MXE, K-split, graded against golden.

    rows_per_desc=None applies the IMAGE-PARITY RULE: 1 on hw (the flying
    image predates the stage-buffer base-row change, so k=2048 chaining is
    sim-only), gemm_job.ROWS_PER_DESC (16) otherwise.

    Returns a dict with the per-chunk INT32 partials, the accumulated sum2,
    golden's own chunked sum2, the bit-exact grade, the padding-lane check,
    and — when `gamma` is given — the full rmsnorm_fx_wide equivalence.
    """
    x8 = np.asarray(x8, dtype=np.int64)
    W = sum2_weight_block(x8)
    if rows_per_desc is None:
        rows_per_desc = 1 if executor == "hw" else gj.ROWS_PER_DESC
    out_dir = Path(out_dir or DEFAULT_OUT)

    t0 = time.time()
    r = gj.ksplit_matvec(W, x8, out_dir=out_dir, name=name, executor=executor,
                         binary=binary, timeout_s=timeout_s,
                         rows_per_desc=rows_per_desc, verbose=verbose)
    wall = time.time() - t0

    acc = np.asarray(r["acc_i32"], dtype=np.int64)
    sum2_hw = int(acc[0])
    gold = golden_sum2_chunked(x8)
    res = {
        "D": int(x8.shape[0]), "executor": executor,
        "rows_per_desc": int(rows_per_desc),
        "K_staged": r["K_staged"], "rows": r["rows"], "kstar": r["kstar"],
        "n_jobs": len(r["jobs"]),
        "chunk_k": [j["k"] for j in r["jobs"]],
        "partials_lane0": [int(p[0]) for p in r["partials"]],
        "sum2_mxe": sum2_hw,
        "sum2_golden_chunked": gold,
        "sum2_equal": bool(sum2_hw == gold),
        "pad_lanes": [int(v) for v in acc[1:]],
        "pad_lanes_zero": bool(np.all(acc[1:] == 0)),
        "in_int32": bool(abs(sum2_hw) < 2 ** 31),
        "within_golden_2p27": bool(0 <= sum2_hw <= 2 ** 27),
        "n_negative_codes": int(np.sum(x8 < 0)),
        "amax_code": int(np.max(np.abs(x8))),
        "problems": list(r["problems"]),
        "wall_s": round(wall, 1),
        "inputs_sha256": r["inputs_sha256"],
    }
    if gamma is not None:
        y_hw, r_hw, nrm_hw = norm_from_sum2(sum2_hw, x8, gamma)
        y_g, r_g, nrm_g = at.rmsnorm_fx_wide([int(v) for v in x8],
                                             [int(v) for v in
                                              np.asarray(gamma,
                                                         dtype=np.int64)])
        res["norm_equal"] = bool(y_hw == y_g and r_hw == r_g
                                 and nrm_hw == nrm_g)
        res["r"] = r_hw
        res["nrm"] = nrm_hw
        res["r_golden"] = r_g
        res["nrm_golden"] = nrm_g
        res["n_y_diff"] = int(sum(1 for a, b in zip(y_hw, y_g) if a != b))
    res["ok"] = bool(res["sum2_equal"] and res["pad_lanes_zero"]
                     and res["in_int32"] and res["within_golden_2p27"]
                     and not res["problems"]
                     and res.get("norm_equal", True))
    return res


# ══ 4. the check script ════════════════════════════════════════════════════
def load_row(npz=None, key: str = "r1_8", gkey: str = "gamma2"):
    z = np.load(Path(npz or DEFAULT_NPZ), allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    return (np.asarray(z[key], dtype=np.int64),
            np.asarray(z[gkey], dtype=np.int64), meta)


def check(args) -> int:
    x8, gamma, meta = load_row(args.npz, args.key, args.gamma_key)
    out = Path(args.out or DEFAULT_OUT)
    print("=" * 78)
    print("CAN THE MXE COMPUTE RMSNORM'S SUM-OF-SQUARES?  (measured, not "
          "reasoned)")
    print("=" * 78)
    print(f"  row      {args.key} of {Path(args.npz or DEFAULT_NPZ).name}: "
          f"layer {meta['layer']} step {meta['step']} of a real Qwen2.5-7B "
          f"decode (tier {meta['tier']}, q_pos {meta['q_pos']})")
    print(f"  meaning  golden's OWN C-1 INT8 quantization of r1 = the "
          f"RMSNorm-2 INPUT row, D_model={x8.shape[0]}")
    print(f"  operand  min={int(x8.min())} max={int(x8.max())} "
          f"amax={int(np.abs(x8).max())} negatives={int(np.sum(x8 < 0))} "
          f"(the row is BOTH the activation and the weight column)")
    print(f"  executor {args.executor} ({args.binary or 'default twin'})")
    print("=" * 78)

    runs = []
    rpds = ([args.rows_per_desc] if args.rows_per_desc
            else ([1] if args.executor == "hw"
                  else [gj.ROWS_PER_DESC, 1]))
    for rpd in rpds:
        tag = ("k2048-chained" if rpd == gj.ROWS_PER_DESC
               else f"rows_per_desc={rpd}")
        note = (" [THE HW-LEGAL CHUNKING: image-parity rule]" if rpd == 1
                else " [sim/HEAD-only chunking]")
        print(f"\n[run  ] chunking {tag}{note}")
        r = mxe_sum2(x8, out_dir=out, name=f"sum2_{args.key}_r{rpd}",
                     executor=args.executor, binary=args.binary,
                     rows_per_desc=rpd, timeout_s=args.timeout_s,
                     verbose=args.verbose, gamma=gamma)
        runs.append(r)
        print(f"[plan ] K_real={r['D']} -> K_staged={r['K_staged']} in "
              f"{r['rows']} stage rows (k*={r['kstar']}), "
              f"{r['n_jobs']} tile job(s), k={r['chunk_k']}")
        print(f"[part ] INT32 lane-0 partials: {r['partials_lane0']}")
        print(f"[sum  ] MXE sum2            : {r['sum2_mxe']}")
        print(f"[gold ] golden chunked sum2 : {r['sum2_golden_chunked']} "
              f"(28 x 128 loop of rmsnorm_fx_wide)")
        print(f"[grade] sum2 bit-exact      : {r['sum2_equal']}   "
              f"pad lanes 1..7 all zero: {r['pad_lanes_zero']}   "
              f"in INT32: {r['in_int32']}   <= 2^27: "
              f"{r['within_golden_2p27']}")
        print(f"[grade] golden's WHOLE wide RMSNorm re-driven by the MXE "
              f"sum2 == arbiter output: {r['norm_equal']} "
              f"(r={r['r']} vs {r['r_golden']}, nrm={r['nrm']} vs "
              f"{r['nrm_golden']}, {r['n_y_diff']} of {r['D']} y values "
              f"differ)")
        print(f"[time ] {r['wall_s']} s")
        for p in r["problems"]:
            print(f"[grade] PROBLEM: {p}")

    chunk_parity = (len(runs) < 2 or
                    runs[0]["sum2_mxe"] == runs[1]["sum2_mxe"])
    if len(runs) > 1:
        print(f"\n[parity] the two K-split chunkings agree: {chunk_parity} "
              f"({runs[0]['sum2_mxe']} vs {runs[1]['sum2_mxe']}) — the "
              f"recombination is chunk-boundary independent")

    # ── DISCRIMINATION: a grader that cannot go red proves nothing ─────────
    gold = golden_sum2_chunked(x8)
    disc = {}
    # a: the equality actually used would REJECT an accumulator off by one
    disc["a_off_by_one"] = (int(runs[0]["sum2_mxe"]) + 1) != gold
    xb = x8.copy()
    i = int(np.argmin(np.abs(xb)))              # a small code: |dx^2| is small
    xb[i] = xb[i] + 1 if xb[i] < 127 else xb[i] - 1
    disc["b_one_code_moved"] = golden_sum2_chunked(xb) != gold
    # c: if the MXE multiply were UNSIGNED (negative codes read as 256+x) the
    #    answer would be a different number — so a sign bug cannot pass
    uns = int(np.sum((x8 & 0xFF) ** 2))
    disc["c_sign_matters"] = int(np.sum(x8 < 0)) > 0 and uns != gold
    # d: the SQUARE is not reproducible by the staged activation alone —
    #    proves the weight-side masking is what makes k* count once
    plan = gj.stage_plan(x8, sum2_weight_block(x8))
    naive = int(np.sum(plan.xst ** 2))
    disc["d_staging_asymmetry_matters"] = naive != gold
    print(f"\n[disc ] (a) the equality used rejects the measured sum2 + 1: "
          f"{disc['a_off_by_one']}")
    print(f"[disc ] (b) one INT8 code moved by 1 changes golden's sum2: "
          f"{disc['b_one_code_moved']}")
    print(f"[disc ] (c) an UNSIGNED multiply would give {uns} != {gold} over "
          f"the row's {int(np.sum(x8 < 0))} negative codes, so a sign bug "
          f"could not pass silently: {disc['c_sign_matters']}")
    print(f"[disc ] (d) squaring the STAGED activation instead ({naive}) != "
          f"golden ({gold}) — the weight-side k* masking is load-bearing: "
          f"{disc['d_staging_asymmetry_matters']}")
    thr = tail_sensitivity(x8, gamma)
    print(f"[disc ] (e) HONESTY NOTE: golden's norm TAIL absorbs sum2 errors "
          f"up to {thr - 1} (smallest delta that moves y/r/nrm = {thr}), so "
          f"the y/r/nrm match above is NECESSARY, not sufficient — the "
          f"load-bearing grade is the exact INT32 equality")
    discriminates = all(disc.values())

    ok = all(r["ok"] for r in runs) and chunk_parity and discriminates
    rec = {"key": args.key, "meta": meta, "executor": args.executor,
           "runs": runs, "chunk_parity": bool(chunk_parity),
           "discriminators": {k: bool(v) for k, v in disc.items()},
           "tail_sensitivity_delta": int(thr), "ok": bool(ok)}
    out.mkdir(parents=True, exist_ok=True)
    rp = out / f"sum2_result_{args.executor}.json"
    rp.write_text(json.dumps(rec, indent=1))

    print("-" * 78)
    print(f"VERDICT: the MXE {'CAN' if ok else 'CANNOT'} produce an INT32 "
          f"accumulator equal to the sum2 golden's chunked path accumulates")
    print(f"  sum2 = {runs[0]['sum2_mxe']} over {runs[0]['D']} INT8 codes, "
          f"contracted as {runs[0]['n_jobs']} K-split MXE job(s), "
          f"0 result expectations in any program")
    print(f"SUM2_MXE CHECK ({args.executor}): {'PASS' if ok else 'FAIL'}")
    print(f"record -> {rp}")
    return 0 if ok else 1


# ══ 5. host-only selftest ══════════════════════════════════════════════════
def _selftest() -> int:
    fails = 0

    def chk(n, cond, extra=""):
        nonlocal fails
        print(f"  [{n}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    # 1: the weight block IS the row
    x = np.array([3, -4, 127, 0, -1], dtype=np.int64)
    W = sum2_weight_block(x)
    chk("1 weight block", W.shape == (5, 1) and np.array_equal(W[:, 0], x),
        f"W[:,0] == x8, shape {W.shape}")

    # 2: -128 is REFUSED, never clipped
    try:
        sum2_weight_block(np.array([-128, 1], dtype=np.int64))
        chk("2 -128 refused", False, "a -128 code was accepted as a weight")
    except AssertionError as e:
        chk("2 -128 refused", "INT8-symmetric" in str(e), f"{str(e)[:52]}…")

    # 3: golden's chunked accumulation == a wide int64 square sum
    rng = np.random.default_rng(20260731)
    xs = rng.integers(-127, 128, 3584, dtype=np.int64)
    xs[7] = 127
    chk("3 chunked == wide", golden_sum2_chunked(xs) == int(np.sum(xs ** 2)),
        f"sum2={golden_sum2_chunked(xs)}")

    # 4: the staged product IS sum(x^2) — stage_plan's own F3, our operands
    p = gj.stage_plan(xs, sum2_weight_block(xs))
    chk("4 staged product", int(np.einsum("k,kn->n", p.xst, p.Wst)[0])
        == int(np.sum(xs ** 2)),
        f"{p.rows} rows, K_staged={p.K_total}, k*={p.kstar}")

    # 5: the staging asymmetry matters — squaring the STAGED activation
    #    over-counts k*, so the weight-side masking is load-bearing
    naive = int(np.sum(p.xst ** 2))
    chk("5 masking load-bearing", naive != int(np.sum(xs ** 2)),
        f"naive staged square {naive} vs true {int(np.sum(xs ** 2))} "
        f"(k* counted {p.rows}x)")

    # 6: chunk-boundary independence, host-side (2048 vs 128 vs 1664 tails)
    for rpd in (16, 8, 1):
        parts = [int(np.einsum("k,kn->n",
                               p.xst[c[0] * 128:c[0] * 128 + c[2]],
                               p.Wst[c[0] * 128:c[0] * 128 + c[2]])[0])
                 for c in p.chunks(rpd)]
        got = int(gj.accumulate_partials(
            [np.array([v] + [0] * 7, dtype=np.int64) for v in parts])[0])
        if got != int(np.sum(xs ** 2)):
            chk(f"6 rpd={rpd}", False, f"{got}")
            break
    else:
        chk("6 chunk independence", True,
            "rows_per_desc 16/8/1 all recombine to the same INT32")

    # 7: the REAL row — bounds golden itself asserts
    try:
        x8, gamma, meta = load_row()
        g = golden_sum2_chunked(x8)
        chk("7 real row", (int(np.abs(x8).max()) == 127 and g <= 2 ** 27
                           and g < 2 ** 31),
            f"D={x8.shape[0]} sum2={g} (2^{g.bit_length()} bits), "
            f"amax=127, layer {meta['layer']} step {meta['step']}")
        # 8: the arbiter tail, driven by golden's own sum2, IS the arbiter
        y1, r1, n1 = norm_from_sum2(g, x8, gamma)
        y2, r2, n2 = at.rmsnorm_fx_wide([int(v) for v in x8],
                                        [int(v) for v in gamma])
        chk("8 tail == arbiter", y1 == y2 and r1 == r2 and n1 == n2,
            f"r={r1} nrm={n1}, {len(y1)} y values identical")
        # 9: HONEST NEGATIVE — the tail ABSORBS small sum2 errors, so the
        #    y/r/nrm match is necessary but NOT sufficient; the primary grade
        #    stays the exact INT32 equality. Measure the absorption.
        thr = tail_sensitivity(x8, gamma)
        y3, r3, n3 = norm_from_sum2(g + 1, x8, gamma)
        chk("9 tail absorbs small errors (measured)",
            (y3, r3, n3) == (y2, r2, n2) and thr > 1
            and norm_from_sum2(g + thr, x8, gamma) != (y2, r2, n2),
            f"sum2+1 leaves r/nrm/y IDENTICAL; smallest sum2 delta that "
            f"moves golden's output = {thr} ({100.0 * thr / g:.2f}% of "
            f"{g}) -> the exact INT32 equality is the load-bearing grade")
    except FileNotFoundError as e:
        chk("7 real row", False, f"npz missing: {e}")

    print(f"SUM2_MXE SELFTEST: {'FAIL' if fails else 'PASS'} (fails={fails})")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--key", default="r1_8")
    ap.add_argument("--gamma-key", default="gamma2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--rows-per-desc", type=int, default=None)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    import remote_hw_exec                        # hw routing (no-op unless
    remote_hw_exec.attach(bridge, a)             # $APEX_F2_HOST is set)
    if a.selftest:
        return _selftest()
    if a.check:
        return check(a)
    ap.error("give --check or --selftest")


if __name__ == "__main__":
    sys.exit(main())
