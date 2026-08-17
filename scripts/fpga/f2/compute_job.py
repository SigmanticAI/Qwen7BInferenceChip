#!/usr/bin/env python3
# compute_job.py — MILESTONE B: compile an INPUT-ONLY attention job and grade
# it AFTERWARDS (audit component C1, docs/design/PROMPT_DEMO_AUDIT.md §2).
#
# ══ WHAT "INPUT-ONLY" MEANS HERE, EXACTLY ══════════════════════════════════
# The 18-job replay path (scripts/fpga/f2/trace_to_regops.py -> core_case)
# cannot be a compute demo: it runs golden `attention_core` FIRST and bakes the
# result into the program — not only into the checks (ERO carries `o8`, ESS
# carries `s_c`) but into the tile's own CONFIG: the PV descriptor's
# rq_scale/rq_shift come from `calib_requant(amax|acc_o|)`, i.e. from the
# OUTPUT (attention.py:396-398 -> gen_l3_vectors.py:422-423). That is the C1
# circularity. A job compiled that way cannot answer "what does the tile
# compute?", only "does the tile reproduce what we already computed?".
#
# The escape (audit §4 risk 1, and the ONLY one that needs no RTL) is the wg3
# precedent, verif/top/wg3/gen_wg3_fuel_case.py:135: run the PV GEMM with
# **requant_en = 0** and pop the RAW INT32 accumulators off the RO path.
# rtl/mxe/mxe_top.sv:195-199 is the whole mechanism:
#
#     r_beat.data[32*c +: 32] = requant_en_q ? {{24{rq_lanes[c][7]}}, rq_lanes[c]}
#                                            : raw_lanes[c];
#
# so with requant_en=0 the eight RO lanes carry `acc_o` itself. Nothing about
# the output is then needed to BUILD the job, and the epilogue (Q13) moves to
# the host, where cap_decode.epilogue calls GOLDEN's own calib_requant /
# requant_i32_to_i8 on the value the TILE produced.
#
# What this job's program contains, and what it does not:
#   INPUTS (all pure functions of q, K_f16, V_f16, tier, G, outlier_idx):
#     * the K/V fp16 rows and the q fp16 row (verbatim job tensors)
#     * the S-3 score composites  s_q·s_k[t]·2^SCORE_FRAC/√D   (CS pushes)
#     * the S-4 P-requant folds   s_v[t]·2^-15                 (QS pushes)
#     s_q/s_k/s_v are the Q5-Q7 INPUT-side stages (kvq_roundtrip +
#     quant_rows_i8) — computed on the host from the job tensors alone. They
#     are tile INPUTS (fp32 composite constants the RTL cannot derive itself),
#     so they must be host-supplied; what matters is that they are upstream of
#     Q8. NOTHING from Q8 onward (acc_s, score_fx, p_q115, c8, s_c, acc_o, rq)
#     enters the program.
#   OUTPUTS — every one of them a `cap` with NO expectation attached:
#     * RO lanes  -> acc_o, raw INT32, D values in D/8 beats  (requant_en=0)
#     * SS words  -> s_q (Q7 quant scale), s_c (S-4 c-scale)
#     * FS words  -> s_h (loader), s_k[t] (feeders), s_v[t] (feeders)
#   The host epilogue then needs s_c, and it TAKES IT FROM THE TILE.
#
# ══ THE ORDERING HAZARD, RESOLVED WITHOUT REORDERING ═══════════════════════
# Audit §2 C1(ii): CS/QS are pushed BEFORE the feeder jobs that publish
# s_k/s_v (gen_l3_vectors.py:379-388 vs :396-398, :430-433), so "capture the
# scales, then push the composites" would need a NEW choreography — and audit
# §4 risk 2 is that reordering forfeits the 27,996-check evidence base.
# It is not needed: s_k/s_v are deterministic functions of K_f16/V_f16 at a
# given tier/G/outlier set, so the host computes them BEFORE the run and the
# verified push order is preserved BYTE-FOR-BYTE. The captured feeder taps are
# then a genuine cross-check of the tile's Q5-Q7 path against a host
# prediction made from inputs only — no circularity, no reorder.
#
# ══ HOW THE PROGRAM IS BUILT (op-list rewrite, the wg3 mechanism) ══════════
# The verified L3 phases are IMPORTED AND CALLED VERBATIM (phase_a,
# loader_phase, store_kv_phase, inject_jobs, score_phase, pv_phase,
# final_phase). Two rewrites, both count-asserted the way
# gen_wg3_fuel_case.py:152-157 asserts its INFO_TIER swap:
#   1. SCRIPT level — `pv_phase` hardcodes rq_en=1 (gen_l3_vectors.py:422), so
#      its D/8 PV descriptor lines are replaced with the rq_en=0 encoding, and
#      EVERY DESC line in the script is then re-decoded to prove bit 65 is
#      clear nowhere-else-too (_assert_no_requant).
#   2. REGOP level — ComputeXlate overrides Xlate.r() so the capture-window
#      sites (ro_w0..ro_w7, ro_meta, fs, ss) emit `cap` WITHOUT `e` instead of
#      a compare. Every site already passes a `sem` (Stage A / D-P1), so no
#      existing file is touched and no op-script token needs a new spelling.
# The structural checks stay checks: CSR INFO_*/STATUS/TIER_CTRL, PERF (rn),
# ERR_STICKY, the TIP block field, the end-of-run FIFO-empty and MB_STATUS
# drop audit. None of those is derived from the attention output —
# _audit_no_output_expectations() proves it by address.
#
# ══ THE REPLACEMENT INVARIANT ══════════════════════════════════════════════
# Audit §2 C1(v): compile_job asserts `compiled o8 == trace o8`, the only
# compiler/golden cross-check, and an input-only compiler cannot keep it. It
# is replaced by an INPUTS DIGEST (sha256 over q/K/V/tier/G/outliers + the
# geometry + the cap-site census) written beside the regops, so a
# recompile-drift or a mis-sliced tensor is still caught.
#
# CLI:
#   python3 scripts/fpga/f2/compute_job.py --smoke     # build+run+grade, T=20
#   python3 scripts/fpga/f2/compute_job.py --npz <job.npz> [--out-dir D]
#   python3 scripts/fpga/f2/compute_job.py --selftest   # host-only, no sim
#
# API:  build_compute_job(q, K, V, tier, G, outlier_idx, out_dir, name) -> Path
#       build_compute_job_full(...) -> (Path, manifest, Prep)
#       grade_compute_job(captures, q, K, V, tier, G, outlier_idx) -> dict

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO / "verif" / "top" / "l3"))
sys.path.insert(0, str(HERE))

import gen_l3_vectors as g3                                    # noqa: E402
import trace_to_regops as t2r                                  # noqa: E402
import cap_decode as cd                                        # noqa: E402
from tile_exec_bridge import run_job                           # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden.fp import f16_bits_to_f64                     # noqa: E402

# Capture-window sites: these are the tile's OUTPUTS. In the verified replay
# path each is an `r` compare against a golden value; in compute mode each
# becomes a `cap` with no expectation.
CAP_SEMS = frozenset({"fs", "ss", "ro_meta"} | {f"ro_w{j}" for j in range(8)})

# Data addresses inside the capture windows (trace_to_regops.py:63-75). A
# remaining `r`/`rn` at any of these would be a surviving output expectation.
_B_MB = t2r.B_MB
OUTPUT_ADDRS = frozenset(
    [_B_MB + t2r.CAP["ro"][1] + 4 * j for j in range(8)]
    + [_B_MB + t2r.CAP["ro"][2],
       _B_MB + t2r.CAP["fs"][1], _B_MB + t2r.CAP["ss"][1]])

DEFAULT_NPZ = (REPO / "docs/results/s8_7b_token/artifact_trace"
               / "job_s019_L19_h03.npz")


# ── host prep: the INPUT-side stages only (Q5-Q7) ───────────────────────────
class Prep:
    """The input-derived quantities the verified op-script emitters need.

    Deliberately shaped like `AttnCore` so gen_l3_vectors' score_phase /
    pv_phase can be called VERBATIM — but every field downstream of Q8 is a
    ZERO placeholder, because the sites that would consume it are rewritten
    into captures. If a future emitter change started reading one of them the
    zeros would produce an obviously-wrong program rather than a quiet
    golden-derived one.
    """

    def __init__(self, q_f64, K_f16, V_f16, tier, G, outlier_idx):
        T, D = K_f16.shape
        self.T, self.D, self.tier, self.G = T, D, tier, G
        self.outlier_idx = tuple(int(c) for c in outlier_idx)
        self.q = np.asarray(q_f64, dtype=np.float64)
        self.K_f16, self.V_f16 = K_f16, V_f16
        # Q5: the KVQ roundtrip the tile will perform in its engines
        self.K_hat, self.V_hat, self.kb, self.vb = at.kvq_roundtrip(
            K_f16, V_f16, tier, G, self.outlier_idx)
        # Q6: feeder scales (what EFS taps will report)
        self.k8, self.s_k = at.quant_rows_i8(self.K_hat)
        self.v8, self.s_v = at.quant_rows_i8(self.V_hat)
        # Q7: the query quant (what the first ESS tap will report)
        q8m, s_qa = at.quant_rows_i8(self.q[None, :])
        self.q8, self.s_q = q8m[0], np.uint16(s_qa[0])
        # ── everything below Q8 is UNKNOWN to this compiler ────────────────
        self.s_c = np.uint16(0)          # tile-sourced (SS cap)
        self.o8 = np.zeros(D, dtype=np.int64)      # RO sites are caps
        self.acc_o = None                          # tile-sourced (RO caps)
        self.score_fx = np.zeros(T, dtype=np.int64)   # TAPSC (not observable)
        self.p_q115 = np.zeros(T, dtype=np.int64)     # TAPPR (not observable)
        self.rq = (0, 0)                 # requant DISABLED, not calibrated

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (np.asarray(self.K_f16, dtype=np.uint16),
                  np.asarray(self.V_f16, dtype=np.uint16)):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(np.ascontiguousarray(self.q).tobytes())
        h.update(f"{self.tier}|G={self.G}|oi={self.outlier_idx}|"
                 f"T={self.T}|D={self.D}".encode())
        return h.hexdigest()


# ── the two rewrites, both count-asserted ───────────────────────────────────
def _desc_line(**kw) -> str:
    return "DESC " + " ".join(f"{w:08x}" for w in g3.desc_words(**kw))


def _decode_desc(line: str) -> int:
    ws = [int(t, 16) for t in line.split()[1:]]        # w127_96 … w31_0
    v = 0
    for w in ws:
        v = (v << 32) | w
    return v


def _assert_no_requant(lines) -> int:
    """Every DESC in the program must carry requant_en=0 (apex_pkg bit 65)."""
    n = 0
    for ln in lines:
        if not ln.startswith("DESC "):
            continue
        n += 1
        v = _decode_desc(ln)
        if (v >> 65) & 1:
            raise AssertionError(
                f"requant_en=1 survives in the compute-mode program: {ln} "
                f"(rq_scale={(v >> 44) & 0xFFFF} rq_shift={(v >> 60) & 0x1F} "
                f"— those are golden-OUTPUT-derived; see audit §2 C1)")
    return n


def rq_off(lines, T: int, D: int) -> tuple[list[str], int]:
    """Rewrite pv_phase's PV descriptors from rq_en=1 to rq_en=0.

    pv_phase (gen_l3_vectors.py:421-423) hardcodes rq_en=1 and takes
    rq_scale/rq_shift from `r.rq`; with Prep.rq = (0, 0) the emitted line is
    exactly the rq_en=1/scale=0/shift=0 encoding, so the swap is a literal
    line substitution with an exact expected count (D/8, one per PV output
    block) — the gen_wg3_fuel_case.py:152-157 mechanism.
    """
    kw = dict(opcode=g3.OP_GEMM_OS, m=1, k=T, n=8, mode_os=1)
    old = _desc_line(rq_en=1, rq_scale=0, rq_shift=0, **kw)
    new = _desc_line(rq_en=0, rq_scale=0, rq_shift=0, **kw)
    want = D // 8
    hits = sum(1 for ln in lines if ln == old)
    if hits != want:
        raise AssertionError(
            f"expected {want} PV descriptor lines (one per output block), "
            f"found {hits} — pv_phase's descriptor encoding changed; the "
            f"rq_en=0 rewrite is no longer a safe substitution")
    out = [new if ln == old else ln for ln in lines]
    _assert_no_requant(out)
    return out, hits


def _audit_no_output_expectations(ops) -> None:
    """No surviving compare may target a capture-window DATA address."""
    bad = [o for o in ops if o["op"] in ("r", "rn", "poll")
           and o.get("a") in OUTPUT_ADDRS]
    if bad:
        raise AssertionError(
            f"{len(bad)} output expectation(s) survived the rewrite, first "
            f"{bad[0]} — Milestone B requires the tile's outputs to be "
            f"captured, never asserted")


class ComputeXlate(t2r.Xlate):
    """Xlate with the capture-window compares turned into pure `cap` egress.

    Stage A gave every `r()` call site a `sem` naming the SITE
    (trace_to_regops.py:102-115), which is exactly the hook this needs: the
    ERO lane words, the ERO meta word and the FS/SS scale words are the tile's
    outputs, everything else (CSR readbacks, PERF, ERR_STICKY, FIFO-empty,
    MB_STATUS) is structural and stays a counted check.
    """

    def __init__(self, D: int):
        super().__init__(D)
        self.n_capsites = 0

    def r(self, a, m, e, sem="r"):
        if sem in CAP_SEMS:
            self.cap(a, m, sem)          # NO `e`: pure egress, cannot fail
            self.n_capsites += 1
            return
        super().r(a, m, e, sem=sem)


# ── expected cap census (a short capture must be loud, not plausible) ───────
def cap_census(T: int, D: int) -> dict:
    """Where every capture in a compute-mode job comes from.

    ro : pv_phase emits one ERO per output block -> D/8 beats x (8 lanes + meta)
    fs : loader_phase 1 (s_h) + score_phase T (s_k) + pv_phase (D/8)*T (s_v,
         re-read per output block because each block re-streams the v8 chunks)
    ss : q injection 1 (s_q) + score_phase 1 (s_c)
    """
    bpr = D // 8
    return {"ro_lanes": bpr * 8, "ro_meta": bpr, "fs": 1 + T + bpr * T,
            "ss": 2, "total": bpr * 9 + 1 + T + bpr * T + 2}


# ── build ───────────────────────────────────────────────────────────────────
def build_script(name, q, K_f16, V_f16, tier=at.TIER_CQ8, G=128,
                 outlier_idx=(), mask_csr=False):
    """core_case's choreography, minus golden, plus rq_en=0.

    Mirrors gen_l3_vectors.core_case:576-602 phase for phase; the ONLY
    differences are (a) `Prep` instead of `attention_core` (so no Q8+ value
    exists at compile time), (b) the ESS after the q injection carries a
    placeholder because that site becomes a capture, (c) the PV descriptors
    are rewritten to rq_en=0.
    """
    K_f16 = np.asarray(K_f16, dtype=np.uint16)
    V_f16 = np.asarray(V_f16, dtype=np.uint16)
    T, D = K_f16.shape
    assert V_f16.shape == (T, D), f"{name}: K/V shape mismatch"
    assert T <= 128, f"{name}: T={T} beyond the F-1 T_ROW_MAX=128 envelope"
    p = Prep(q, K_f16, V_f16, tier, G, outlier_idx)

    s = g3.Script(D)
    s.emit(f"// COMPUTE MODE (requant_en=0, input-only): {name} "
           f"(T={T} D={D} {tier}) — the tile's RAW INT32 accumulators are "
           f"CAPTURED, the Q13 epilogue runs on the host")
    g3.phase_a(s, tiers_used=(g3.TIER_CODE[tier],))
    if mask_csr:
        g3.mask_load_phase(s, outlier_idx)
    g3.loader_phase(s)
    g3.store_kv_phase(s, K_f16, V_f16, tier)

    s.emit("// inject q row through squant MODE_QUANT (Q7 machinery); s_q is "
           "CAPTURED off the SS tap, not asserted")
    s.route(rdst=1, asrc=1)
    qpairs = [g3.decompose_f16(int(b))[:2]
              for b in g3.to16(np.asarray(p.q, dtype=np.float64))]
    g3.inject_jobs(s, qpairs, 1)
    s.ess(0, 1)                      # placeholder: this site becomes a `cap`
    s.aj(0, 1, 0, 1, s.BPR, 0)       # LOAD q8 -> act bank 1

    g3.score_phase(s, p, T)          # CS/QS composites: INPUT-derived
    g3.pv_phase(s, p, T)             # ERO sites -> captures; rq rewritten
    g3.final_phase(s, T)

    lines, n_rq = rq_off(s.lines, T, D)
    return lines, p, n_rq


def build_compute_job(q, K, V, tier=at.TIER_CQ8, G=128, outlier_idx=(),
                     out_dir=None, name="compute_job", mask_csr=False) -> Path:
    """Compile ONE input-only attention job -> the regops.jsonl PATH.

    This is the integration contract (one call, one path out). The manifest is
    written beside the regops as <name>.manifest.json; grade_compute_job()
    needs nothing from here — it re-derives the host prep from the same inputs.
    Use build_compute_job_full() when the manifest/prep objects are wanted
    in-process (the smoke does, to avoid re-deriving them).
    """
    return build_compute_job_full(q, K, V, tier, G, outlier_idx, out_dir,
                                  name, mask_csr)[0]


def build_compute_job_full(q, K, V, tier=at.TIER_CQ8, G=128, outlier_idx=(),
                           out_dir=None, name="compute_job", mask_csr=False):
    """As build_compute_job, returning (path, manifest_dict, Prep).

    q : fp16-grid float64 vector [D], or the uint16 fp16 BITS of one.
    K, V : uint16 fp16 bits [T, D] (as the S8 trace npz stores them), or
           float64 values, which are narrowed to the fp16 grid here.
    """
    q = np.asarray(q)
    if q.dtype == np.uint16:
        q = f16_bits_to_f64(q).reshape(-1)
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K)
    V = np.asarray(V)
    if K.dtype != np.uint16:
        K = g3.to16(np.asarray(K, dtype=np.float64))
    if V.dtype != np.uint16:
        V = g3.to16(np.asarray(V, dtype=np.float64))

    lines, prep, n_rq = build_script(name, q, K, V, tier, G, outlier_idx,
                                     mask_csr=mask_csr)
    T, D = prep.T, prep.D
    x = ComputeXlate(D)
    x.note(f"COMPUTE-MODE job {name}: T={T} D={D} {tier} requant_en=0 — "
           f"outputs are `cap` egress (no baked expectations); host epilogue "
           f"in scripts/fpga/f2/cap_decode.py")
    x.run(lines)
    _audit_no_output_expectations(x.ops)

    census = cap_census(T, D)
    if x.cap_seq != census["total"]:
        raise AssertionError(
            f"{name}: emitted {x.cap_seq} caps, census expects "
            f"{census['total']} ({census}) — the capture sites moved")

    out_dir = Path(out_dir or (REPO / "build" / "f2_compute"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.compute.regops.jsonl"
    with path.open("w") as f:
        for o in x.ops:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    man = {"name": name, "mode": "compute", "requant_en": 0,
           "T": T, "D": D, "tier": tier, "G": G,
           "outlier_idx": list(prep.outlier_idx),
           "regops": len(x.ops), "checks": x.n_checks,
           "caps": x.cap_seq, "cap_sites_rewritten": x.n_capsites,
           "cap_census": census, "pv_desc_rq_cleared": n_rq,
           "taps_skipped": x.n_tap_skipped,
           "inputs_sha256": prep.digest(),
           "host_prep": {"s_q": int(prep.s_q),
                         "s_k": [int(v) for v in prep.s_k],
                         "s_v": [int(v) for v in prep.s_v]}}
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return path, man, prep


# ── grade ───────────────────────────────────────────────────────────────────
def _fs(caps):
    return cd.fp16_bits(caps, sems=("fs",), addrs=(cd.FS_W,))


def _ss(caps):
    return cd.fp16_bits(caps, sems=("ss",), addrs=(cd.SS_W,))


def grade_compute_job(captures, q, K, V, tier=at.TIER_CQ8, G=128,
                      outlier_idx=(), prep=None) -> dict:
    """Decode the tile's captures, run the host epilogue, grade vs golden.

    Golden is computed HERE — after the run, from the same inputs the job was
    compiled from — so nothing golden ever reached the tile.

    Grades, in order of strength:
      acc_i32   the tile's RAW INT32 accumulators vs golden acc_o. This is the
                bit-exactness claim. cap_decode._selftest 5a is the reason it
                is graded separately: the requant quantum is 2^shift/scale
                accumulator counts, so an acc error smaller than that is
                INVISIBLE in out_hat.
      rq/o8     the HOST epilogue's calibration + INT8 codes vs golden's.
      out_hat   the float64 vector that re-enters the layer — the
                token-identity claim.
      scales    captured s_q/s_c and the s_k/s_v/s_h feeder taps vs the host
                prep (made from inputs BEFORE the run) and vs golden.
    """
    q = np.asarray(q)
    if q.dtype == np.uint16:
        q = f16_bits_to_f64(q).reshape(-1)
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K, dtype=np.uint16)
    V = np.asarray(V, dtype=np.uint16)
    T, D = K.shape
    bpr = D // 8
    if prep is None:
        prep = Prep(q, K, V, tier, G, outlier_idx)

    gold = at.attention_core(q, K, V, tier, G=G, outlier_idx=outlier_idx)

    res: dict = {"T": T, "D": D, "tier": tier, "G": G,
                 "n_caps": len(captures), "cap_census": cap_census(T, D),
                 "problems": []}

    # ── RO -> raw INT32 accumulators ───────────────────────────────────────
    acc = cd.ro_lanes_to_i32(captures, expect_n=D)
    res["acc_i32"] = [int(v) for v in acc]
    res["acc"] = cd.grade(acc, gold.acc_o)

    # ── SS -> s_q (Q7) then s_c (S-4) ──────────────────────────────────────
    ss = _ss(captures)
    if len(ss) != 2:
        res["problems"].append(f"expected 2 SS captures (s_q, s_c), got "
                               f"{len(ss)}: {[c['tag'] for c in ss]}")
    s_q_bits = ss[0]["bits"] if ss else None
    s_c_bits = ss[-1]["bits"] if len(ss) == 2 else None
    res["s_q"] = {"captured": s_q_bits, "host_prep": int(prep.s_q),
                  "golden": int(gold.s_q),
                  "equal": s_q_bits == int(gold.s_q)}
    res["s_c"] = {"captured": s_c_bits, "golden": int(gold.s_c),
                  "equal": s_c_bits == int(gold.s_c)}

    # ── FS -> s_h (loader), s_k[t] (score), s_v[t] x D/8 (pv) ─────────────
    fs = _fs(captures)
    n_fs_want = 1 + T + bpr * T
    if len(fs) != n_fs_want:
        res["problems"].append(f"expected {n_fs_want} FS captures "
                               f"(1 s_h + {T} s_k + {bpr}x{T} s_v), "
                               f"got {len(fs)}")
    bits = [c["bits"] for c in fs]
    s_k_cap = bits[1:1 + T]
    s_v_blocks = [bits[1 + T + b * T:1 + T + (b + 1) * T] for b in range(bpr)]
    res["s_h_loader"] = bits[0] if bits else None
    res["s_k"] = {"equal": s_k_cap == [int(v) for v in gold.s_k],
                  "host_prep_equal": s_k_cap == [int(v) for v in prep.s_k],
                  "n": len(s_k_cap)}
    sv_gold = [int(v) for v in gold.s_v]
    res["s_v"] = {"equal": all(b == sv_gold for b in s_v_blocks),
                  "host_prep_equal": all(
                      b == [int(v) for v in prep.s_v] for b in s_v_blocks),
                  "blocks": len(s_v_blocks),
                  "blocks_identical": all(b == s_v_blocks[0]
                                          for b in s_v_blocks)}

    # ── the HOST epilogue on the TILE's numbers ───────────────────────────
    if s_c_bits is None:
        res["problems"].append("no s_c capture — the host epilogue needs the "
                               "tile's c-scale; grading stops here")
        res["ok"] = False
        return res
    ep = cd.epilogue(acc, s_c_bits)
    res["epilogue"] = {"rq": tuple(int(v) for v in ep["rq"]),
                       "golden_rq": tuple(int(v) for v in gold.rq),
                       "s_out": ep["s_out"], "golden_s_out": gold.s_out,
                       "amax_acc": ep["amax_acc"]}
    res["rq_equal"] = tuple(ep["rq"]) == tuple(int(v) for v in gold.rq)
    res["o8"] = cd.grade(ep["o8"], gold.o8)
    res["out_hat"] = cd.grade(ep["out_hat"], gold.out_hat)
    res["out_hat_values"] = [float(v) for v in ep["out_hat"]]
    res["o8_values"] = [int(v) for v in ep["o8"]]

    res["ok"] = bool(res["acc"]["equal"] and res["o8"]["equal"]
                     and res["out_hat"]["equal"] and res["rq_equal"]
                     and res["s_q"]["equal"] and res["s_c"]["equal"]
                     and res["s_k"]["equal"] and res["s_v"]["equal"]
                     and not res["problems"])
    return res


# ── npz entry point + smoke ────────────────────────────────────────────────
def load_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    tier = {"CQ-8": at.TIER_CQ8, "CQ-4": at.TIER_CQ4,
            "CQ-4+": at.TIER_CQ4P}[meta["tier"]]
    return {"meta": meta, "tier": tier,
            "q": f16_bits_to_f64(z["q_f16"].astype(np.uint16)).reshape(-1),
            "K": z["K_f16"].astype(np.uint16),
            "V": z["V_f16"].astype(np.uint16),
            "G": int(meta.get("G", 128)),
            "trace": {k: z[k] for k in ("s_q", "s_k", "s_v", "s_c", "acc_o",
                                        "o8", "out_hat", "rq_scale",
                                        "rq_shift") if k in z.files}}


def smoke(npz_path=DEFAULT_NPZ, out_dir=None, executor="sim", binary=None,
          cap_out=None, timeout_s=1800) -> int:
    j = load_npz(npz_path)
    meta, tier = j["meta"], j["tier"]
    name = Path(npz_path).stem
    print("=" * 78)
    print(f"MILESTONE B SMOKE — input-only compute job from a REAL "
          f"Qwen2.5-7B attention op")
    print(f"  npz    {npz_path}")
    print(f"  op     step {meta['step']} layer {meta['layer']} head "
          f"{meta['head']} q_pos {meta['q_pos']} kv_head {meta['kv_head']}")
    print(f"  geom   T={meta['T']} D={meta['D']} {tier} G={j['G']} "
          f"score_frac={meta['score_frac']}")
    print("=" * 78)

    t0 = time.time()
    path, man, prep = build_compute_job_full(
        j["q"], j["K"], j["V"], tier, j["G"], (), out_dir, f"{name}_compute")
    print(f"[build] {path}")
    print(f"[build] {man['regops']} regops, {man['checks']} structural checks "
           f"(0 output expectations), {man['caps']} caps "
           f"({man['cap_sites_rewritten']} sites rewritten)")
    print(f"[build] PV descriptors with requant cleared: "
          f"{man['pv_desc_rq_cleared']} (= D/8); every DESC re-decoded, "
          f"bit65=0")
    print(f"[build] cap census {man['cap_census']}")
    print(f"[build] inputs sha256 {man['inputs_sha256'][:32]}… "
          f"({time.time() - t0:.1f}s)")

    # host prep vs the stored trace: a compiler sanity check on the INPUT side
    # only (NOT baked into the job; it is what the replaced o8 assert used to
    # cover, restricted to what an input-only compiler may legitimately know).
    tr = j["trace"]
    if "s_k" in tr:
        ok_k = np.array_equal(np.asarray(prep.s_k, dtype=np.uint16),
                              tr["s_k"].astype(np.uint16))
        ok_v = np.array_equal(np.asarray(prep.s_v, dtype=np.uint16),
                              tr["s_v"].astype(np.uint16))
        ok_q = int(prep.s_q) == int(tr["s_q"])
        print(f"[build] host prep vs stored S8 trace: s_q {ok_q} s_k {ok_k} "
              f"s_v {ok_v}  (input-side stages only)")

    t1 = time.time()
    print(f"[run  ] executor={executor}")
    r = run_job(path, executor=executor, binary=binary, cap_out=cap_out,
                timeout_s=timeout_s)
    print(r["log"].rstrip())
    print(f"[run  ] rc={r['rc']} ok={r['ok']} captures={len(r['captures'])} "
          f"reported={r['n_reported']} ({time.time() - t1:.1f}s)")
    for n in r["notes"]:
        print(f"[run  ] NOTE: {n}")
    if not r["captures"]:
        print("SMOKE: FAIL — no captures; nothing to grade")
        return 1

    g = grade_compute_job(r["captures"], j["q"], j["K"], j["V"], tier,
                          j["G"], (), prep=prep)
    gold = at.attention_core(j["q"], j["K"], j["V"], tier, G=j["G"])
    print("-" * 78)
    print(f"[grade] captures {g['n_caps']} (census "
          f"{g['cap_census']['total']})")
    print(f"[grade] RAW INT32 acc_o from the tile  : {g['acc']}")
    print(f"[grade]   first 8 lanes captured       : {g['acc_i32'][:8]}")
    print(f"[grade]   golden acc_o first 8         : "
          f"{[int(v) for v in gold.acc_o[:8]]}")
    print(f"[grade] s_q captured {g['s_q']['captured']} vs golden "
          f"{g['s_q']['golden']} -> {g['s_q']['equal']}")
    print(f"[grade] s_c captured {g['s_c']['captured']} vs golden "
          f"{g['s_c']['golden']} -> {g['s_c']['equal']}   (the epilogue's "
          f"only scale, TILE-SOURCED)")
    print(f"[grade] feeder taps: s_k {g['s_k']}  ")
    print(f"[grade]              s_v {g['s_v']}")
    if "epilogue" in g:
        e = g["epilogue"]
        print(f"[grade] HOST epilogue: calib_requant(amax={e['amax_acc']}) "
              f"-> rq={e['rq']}, golden rq={e['golden_rq']} -> "
              f"{g['rq_equal']}")
        print(f"[grade]   s_out {e['s_out']!r} vs golden "
              f"{e['golden_s_out']!r}")
        print(f"[grade] o8      : {g['o8']}")
        print(f"[grade] out_hat : {g['out_hat']}")
        print(f"[grade]   out_hat[:6] {[round(v, 8) for v in g['out_hat_values'][:6]]}")
        print(f"[grade]   golden [:6] "
              f"{[round(float(v), 8) for v in gold.out_hat[:6]]}")
    for pb in g["problems"]:
        print(f"[grade] PROBLEM: {pb}")

    # ── DISCRIMINATION: a grader that cannot go red proves nothing ─────────
    # (a) perturb ONE captured RO lane by one count — the raw-acc grade must
    #     catch it even though out_hat may not (the requant quantum caveat,
    #     cap_decode._selftest 5a).
    # (b) grade the SAME (correct) captures against golden recomputed from a
    #     CHANGED input (q -> -q) — must go red, or the comparison is a
    #     rubber stamp.
    disc = {}
    pert = [dict(c) for c in r["captures"]]
    tag = None
    for c in pert:
        if c["sem"].startswith("ro_w") and c["sem"] != "ro_meta":
            c["value"] = (c["value"] + 1) & 0xFFFFFFFF
            tag = c["tag"]
            break
    ga = grade_compute_job(pert, j["q"], j["K"], j["V"], tier, j["G"], (),
                           prep=prep)
    disc["a_one_count_perturbation"] = {
        "tag": tag, "acc_red": not ga["acc"]["equal"],
        "out_hat_red": not ga.get("out_hat", {}).get("equal", True),
        "ok_red": not ga["ok"]}
    gb = grade_compute_job(r["captures"], -j["q"], j["K"], j["V"], tier,
                           j["G"], ())
    disc["b_wrong_golden_q_negated"] = {"acc_red": not gb["acc"]["equal"],
                                        "ok_red": not gb["ok"]}
    print(f"[disc ] (a) one-count perturbation of {tag}: "
          f"acc RED={disc['a_one_count_perturbation']['acc_red']} "
          f"out_hat RED="
          f"{disc['a_one_count_perturbation']['out_hat_red']} "
          f"(out_hat may stay green — sub-quantum caveat)")
    print(f"[disc ] (b) correct captures vs golden(q -> -q): "
          f"acc RED={disc['b_wrong_golden_q_negated']['acc_red']} "
          f"verdict RED={disc['b_wrong_golden_q_negated']['ok_red']}")
    discriminates = (disc["a_one_count_perturbation"]["acc_red"]
                     and disc["a_one_count_perturbation"]["ok_red"]
                     and disc["b_wrong_golden_q_negated"]["acc_red"]
                     and disc["b_wrong_golden_q_negated"]["ok_red"])
    if not discriminates:
        print("[disc ] PROBLEM: the grader did NOT go red on a known-bad "
              "input — the green result above is not evidence")

    print("-" * 78)
    print(f"[check] executor-counted checks: {man['checks']} compiled "
          f"(compiler counts KVP polls) — the 487-style cap sites are NOT "
          f"checks by construction; the canonical replay arm "
          f"(build/f2_regops) is untouched, so no check is lost from it")
    verdict = "PASS" if (g["ok"] and r["ok"] and discriminates) else "FAIL"
    print(f"COMPUTE-MODE (rq_en=0) BIT-EXACT: acc_i32={g['acc']['equal']} "
          f"o8={g.get('o8', {}).get('equal')} "
          f"out_hat={g.get('out_hat', {}).get('equal')}")
    print(f"MILESTONE B SMOKE: {verdict}  "
          f"(tile raw INT32 -> host epilogue -> golden out_hat, "
          f"{len(r['captures'])} captured values, {man['checks']} structural "
          f"checks, 0 output expectations)")
    return 0 if verdict == "PASS" else 1


# ── selftest: build only, no executor ──────────────────────────────────────
def _selftest() -> int:
    fails = 0

    def chk(n, cond, extra=""):
        nonlocal fails
        print(f"  [{n}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    import tempfile
    rng = np.random.default_rng(20260729)
    D, T = 128, 5
    K = g3.to16(rng.normal(0, 1, (T, D)))
    V = g3.to16(rng.normal(0, 1, (T, D)))
    q = f16_bits_to_f64(g3.to16(rng.normal(0, 1, D))).reshape(D)
    tmp = Path(tempfile.mkdtemp(prefix="apex_cj_"))
    path, man, prep = build_compute_job_full(q, K, V, out_dir=tmp, name="st")
    ops = [json.loads(l) for l in path.read_text().splitlines()]
    chk("1 compiled", man["regops"] == len(ops) and man["caps"] > 0,
        f"{man['regops']} regops, {man['caps']} caps")
    chk("2 census", man["caps"] == man["cap_census"]["total"],
        f"{man['cap_census']}")
    chk("3 no cap carries an expectation",
        all("e" not in o for o in ops if o["op"] == "cap"))
    chk("4 no compare on a capture-window data address",
        not [o for o in ops if o["op"] in ("r", "rn", "poll")
             and o.get("a") in OUTPUT_ADDRS])
    descs = [o for o in ops if o["op"] == "w" and o["a"] == t2r.B_MB + 0x88]
    chk("5 requant field word is 0 in every descriptor push",
        all((o["d"] >> 1) & 1 == 0 for o in descs), f"{len(descs)} DS_W[2]")
    # the rewrite must be verifiable independently of build_compute_job
    s2 = build_script("st2", q, K, V)[0]
    chk("6 rewrite is verified by re-decoding", _assert_no_requant(s2) > 0,
        f"{_assert_no_requant(s2)} DESC lines, all bit65=0")
    # golden-value leak check: no golden acc_o value may appear as an
    # expectation anywhere in the program.
    # THRESHOLD, disclosed: only |acc| >= 256 is searched, and INFO_VER's
    # 0x00010000 is excluded. The surviving structural readbacks are small enum
    # and flag constants (INFO_N=8, INFO_TIER=3, STATUS bits, INFO_D=128,
    # INFO_G=16) plus INFO_VER; a golden INT8 code or a tiny accumulator
    # collides with those by PIGEONHOLE, not by leakage — an unrestricted
    # search reports two guaranteed false positives (0x1008 e=8, 0x1014 e=3).
    # The address-side proof is check 4, which admits no exceptions.
    gold = at.attention_core(q, K, V, at.TIER_CQ8, G=128)
    leak = {int(v) & 0xFFFFFFFF for v in gold.acc_o if abs(int(v)) >= 256}
    leak -= {0x00010000}
    # Only EXPECTATIONS are searched. The ~7k `w`/`pw` payloads carry the input
    # tensors (fp16-decomposition weight beats, gamma, the CS/QS composites);
    # over that many 32-bit words a collision with one of ~125 golden values is
    # a birthday hit, not a leak (observed: XW0 beat d=498 vs an acc of 498).
    hit = [o for o in ops if o["op"] in ("r", "cap") and o.get("e") in leak]
    chk("7 no golden output value survives as an expectation",
        not hit, f"{len(leak)} distinct golden |acc|>=256 values searched"
                 f"{'' if not hit else f'; first {hit[0]}'}")
    chk("8 s_c is NOT known at build time", int(prep.s_c) == 0
        and prep.rq == (0, 0) and not prep.o8.any())
    print(f"COMPUTE_JOB SELFTEST: {'FAIL' if fails else 'PASS'} "
          f"(fails={fails}; T={T} D={D}, {man['regops']} regops, "
          f"{man['caps']} caps, {man['checks']} structural checks)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=None, help="one S8 trace job .npz")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--executor", choices=("sim", "hw"), default="sim")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--cap-out", default=None)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--smoke", action="store_true",
                    help="build + run + decode + grade one job end to end")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    npz = Path(a.npz) if a.npz else DEFAULT_NPZ
    if a.build_only:
        j = load_npz(npz)
        path, man, _ = build_compute_job_full(
            j["q"], j["K"], j["V"], j["tier"], j["G"], (), a.out_dir,
            f"{npz.stem}_compute")
        print(json.dumps(man, indent=1))
        print(f"-> {path}")
        return 0
    if a.smoke or a.npz:
        return smoke(npz, a.out_dir, a.executor, a.binary, a.cap_out,
                     a.timeout_s)
    ap.error("give --smoke, --npz, --build-only or --selftest")


if __name__ == "__main__":
    sys.exit(main())
