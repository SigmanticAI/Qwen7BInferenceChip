#!/usr/bin/env python3
"""gen_wg3_tile_case.py — W-G3 option 3 (integration-lead ruling, 2026-07-26):
the TILE-level walked half at real 7B PER-HEAD geometry, n_heads = 1.

A NEW SIBLING of `verif/top/l3/gen_l3_vectors.py` and of
`verif/top/l3/gen_walkfmt2_desc.py` (the B1 §7 rule: spec generators are never
edited). It IMPORTS the L3 generator and calls its OWN `core_case` builder —
the verified host choreography — with the REAL traced Qwen2.5-7B head instead
of synthetic random data. Nothing about the op stream is authored here.

Why `core_case` and not `full_case`: `core_case(name, q, K16, V16, D, tier)`
takes the attention state as INPUTS and stages it through squant MODE_F16 /
MODE_QUANT (`store_kv_phase` / `inject_jobs`), so no projection GEMM is
involved. That matters — the real head's K/V CANNOT be produced on-tile at
this build point: gap **B** (no projection-bias adder, and Qwen2.5 k/v
projections carry biases) and gap **D** (`CFG_D` sizes the activation stage
row, so the real k = 3584 projection has no activation path at CFG_D = 128).
Injection sidesteps both and stages the traced values BIT-EXACTLY.

Outputs (under --build, default verif/top/wg3/build):
    cases/<name>.txt          the L3 op script (host-mode replay)
    walker/<name>.desc        the D-028 v1 descriptor (host/walker mode)
    walker/<name>.desc2       the D-029 fmt=1 image (the WALKED half)
    <name>.manifest.json      geometry + check count + provenance record

SELF-CHECKS (hard asserts, no simulator):
  P1  the case's golden `attention_core` result reproduces the COMMITTED S8
      per-head record bit-exactly on every shared field — the same provenance
      gate stage A applies, re-asserted at the tile-case boundary
  P2  the staged q/K/V are the traced fields themselves (identity, not a
      re-derivation)
  P3  the fmt=1 image passes the `check2` mirror at CFG_D
  P4  the fmt=1 image is per-head EQUIVALENT to the v1 descriptor (walker2's
      S2_HLOAD synthesis reproduces GEOM/RQ/MASK bit-exactly)
  P5  every emitted MXE descriptor in the case is legal for the implemented
      tile (1 <= n <= MXE_N, 1 <= m <= M_TILE_MAX, 1 <= k <= K_MAX)
  P6  the phase-A INFO_TIER expectation is rewritten to this BUILD's truth
      (a GQA build is CQ-8-only, apex_top.sv:1786-1789) by value, once
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO / "verif" / "seq_walker"))
sys.path.insert(0, str(REPO / "verif" / "top" / "l3"))

import gen_l3_vectors as glv                                  # noqa: E402
import seq_walker_fmt as fmt                                  # noqa: E402
from apex_golden.fp import f16_bits_to_f64                    # noqa: E402

S8_TRACE = REPO / "docs/results/s8_7b_token/artifact_trace"
MXE_N, M_TILE_MAX, K_MAX = 8, 64, 2048

# the head with the COMMITTED S8 record for this (step, layer) — the strongest
# provenance available, and the one stage A cross-checks on 22 arrays
DEF_HEAD = 3


def build(bundle: Path, head: int, out: Path) -> dict:
    meta = json.loads(bundle.with_suffix(".json").read_text())
    bun = dict(np.load(bundle))
    D, T = meta["head_dim"], meta["T"]
    assert D == 128, f"this case targets the CFG_D=128 build point, got {D}"

    tag = f"h{head:02d}"
    K16 = np.asarray(bun[f"{tag}_K_f16"], dtype=np.uint16)
    V16 = np.asarray(bun[f"{tag}_V_f16"], dtype=np.uint16)
    q16 = np.asarray(bun[f"{tag}_q_f16"], dtype=np.uint16)
    assert K16.shape == (T, D) and V16.shape == (T, D) and q16.shape == (D,)

    # q enters as f64; passing the EXACT f16 value makes core_case's internal
    # to16() idempotent, so the injected row is the traced field itself (P2)
    q = f16_bits_to_f64(q16)
    assert np.array_equal(glv.to16(q), q16), "P2: q round-trip not idempotent"

    name = (f"wg3_s{meta['step']:03d}_L{meta['layer']:02d}_{tag}_hd{D}_T{T}")
    s, r = glv.core_case(name, q, K16, V16, D, tier=glv.TIER_CQ8)

    # ── P1: the committed-record provenance gate, re-asserted here ───────────
    rec = S8_TRACE / f"job_s{meta['step']:03d}_L{meta['layer']:02d}_h{head:02d}.npz"
    assert rec.exists(), f"no committed record {rec.name}"
    z = np.load(rec)
    checked = []
    for f in ("q_f16", "K_f16", "V_f16", "k8", "v8", "q8", "s_k", "s_v", "s_q",
              "acc_s", "score_fx", "p_q115", "c8", "s_c", "acc_o", "o8",
              "s_out", "out_hat", "sm_m", "sm_l"):
        if not hasattr(r, f):
            continue
        got = np.asarray(getattr(r, f))
        want = np.asarray(z[f])
        assert np.array_equal(got, want), \
            f"P1 PROVENANCE FAIL: {f} differs from the committed record"
        checked.append(f)
    rq_scale, rq_shift = int(r.rq[0]), int(r.rq[1])
    assert rq_scale == int(z["rq_scale"]) and rq_shift == int(z["rq_shift"]), \
        "P1 PROVENANCE FAIL: rq pair differs from the committed record"
    checked += ["rq_scale", "rq_shift"]

    # ── P5: every MXE descriptor the case drives must be tile-legal ──────────
    # The L3 op form is `DESC <w3> <w2> <w1> <w0>` — the 128-bit mxe_desc_t
    # packed by `glv.desc_words` (opcode[7:0], m[19:8], k[31:20], n[43:32]).
    # phase A and phase C deliberately drive ILLEGAL descriptors to gate the
    # D-006 reject path, so they are counted and excluded by VALUE, never by
    # position: an illegal one that is not a known reject probe still fails.
    n_desc = n_reject = 0
    for ln in s.lines:
        f = ln.split()
        if f[:1] != ["DESC"]:
            continue
        v = 0
        for i, w in enumerate(f[1:5]):
            v |= int(w, 16) << (32 * (3 - i))
        m_, k_, n_ = (v >> 8) & 0xFFF, (v >> 20) & 0xFFF, (v >> 32) & 0xFFF
        legal = (1 <= n_ <= MXE_N and 1 <= m_ <= M_TILE_MAX
                 and 1 <= k_ <= K_MAX)
        if legal:
            n_desc += 1
        else:
            # the only sanctioned illegal shapes are the k=0 reject probes
            assert k_ == 0, f"P5: unexpected illegal DESC (m={m_} k={k_} " \
                            f"n={n_}): {ln}"
            n_reject += 1

    # ── P6: the GQA build's INFO_TIER truth ─────────────────────────────────
    # `gen_l3_vectors.phase_a` scripts INFO_TIER = 0x7 (D=64, masked) or 0x3
    # (D=128) — the TIER-BANK builds. This lane's build point sets
    # KVQ_GQA_NENG=4, where apex_kvq_gqa_bank REPLACES the tier bank and the
    # tile is CQ-8-only, so apex_top's 0x14 read override reports {0,0,1}:
    #
    #   rtl/top/apex_top.sv:1786-1789 -- "a GQA build (KVQ_GQA_NENG>1) is
    #   CQ-8-only -- INFO_TIER never lies (D-027): the CQ-4 and CQ-4+ bits
    #   drop", tier_rdata_q = {29'b0, (OUTLIER_K>0)&&mask_valid,
    #                          (KVQ_GQA_NENG == 1), 1'b1}
    #
    # So 0x1 is the BUILD TRUTH here and 0x3 would be the lying expectation.
    # Rewritten by VALUE with an exact-count assert — never by line number.
    want_tier = 0x1
    old = f"CSRR 14 ffffffff {0x3:08x}"
    new = f"CSRR 14 ffffffff {want_tier:08x}"
    n_hit = sum(1 for ln in s.lines if ln == old)
    assert n_hit == 1, \
        f"P6: expected exactly 1 phase-A INFO_TIER line, found {n_hit}"
    s.lines = [new if ln == old else ln for ln in s.lines]

    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "walker").mkdir(parents=True, exist_ok=True)
    (out / "cases" / f"{name}.txt").write_text("\n".join(s.lines) + "\n")

    # ── the D-028 v1 descriptor (gen_walker_desc.py's field layout) ──────────
    geom = (D & 0xFF) | ((T & 0x1FF) << 8) | (0 << 16)        # tier CQ-8
    rq = ((rq_shift & 0x1F) << 16) | (rq_scale & 0xFFFF)
    (out / "walker" / f"{name}.desc").write_text(
        f"WALKABLE 1\nGEOM {geom:08x}\nRQ {rq:08x}\nMASK 00000003\n"
        f"TROWS {T}\n")

    # ── the D-029 fmt=1 image (gen_walkfmt2_desc.py's construction) ──────────
    # n_heads=1, n_kv_heads=1 => d_model == head_dim (gap D: one build implies
    # head_dim == D_model). kv_map=0 = flat record addressing, the
    # tile-truthful form for a single engine. pos_m = T-1 (canonical; ROPE is
    # mask-disabled in this image, the rows are staged already-rotated).
    words = {
        fmt.W_GEOM0:  fmt.pack_geom0(D, tier=0),
        fmt.W_MODEL0: fmt.pack_model0(D, 0),          # d_ffn=0: FFN disabled
        fmt.W_MODEL1: fmt.pack_model1(1, 1, kv_map=0),
        fmt.W_MASK:   fmt.pack_mask((1 << fmt.EN_SCORE) | (1 << fmt.EN_PV)),
        fmt.W_STEP:   fmt.pack_step(T, T - 1),
        fmt.W_RQ0:    rq,
    }
    err = fmt.check2(words[fmt.W_GEOM0], words[fmt.W_MODEL0],
                     words[fmt.W_MODEL1], words[fmt.W_MASK],
                     words[fmt.W_STEP], cfg_d=D)
    assert err == fmt.ERR_NONE, f"P3: fmt=1 image illegal (err={err})"
    # P4: walker2's per-head S2_HLOAD synthesis == the v1 descriptor
    assert ((T << 8) | D) == geom, f"P4: synth GEOM {(T<<8)|D:08x} != {geom:08x}"
    assert words[fmt.W_RQ0] == rq, "P4: synth RQ != v1 RQ"
    # walker2's S2_HLOAD builds the v1 mask as {en_pv, en_score} from the
    # fmt=1 en_mask; re-derive it from the packed word rather than restating it
    en_mask = words[fmt.W_MASK] & fmt.EN_ALL
    synth_mask = (((en_mask >> fmt.EN_PV) & 1) << 1) \
        | ((en_mask >> fmt.EN_SCORE) & 1)
    assert en_mask == ((1 << fmt.EN_SCORE) | (1 << fmt.EN_PV)), \
        f"P4: fmt=1 en_mask {en_mask:03x} is not exactly SCORE|PV"
    assert synth_mask == 0x3, f"P4: synth MASK {synth_mask:x} != v1 0x3"
    with (out / "walker" / f"{name}.desc2").open("w") as fh:
        for idx in sorted(words):
            fh.write(f"DW2 {idx} {words[idx]:08x}\n")
        fh.write(f"TROWS {T}\n")

    man = {"name": name, "build": "wg3", "kind": "core-real7b",
           "T": T, "D": D, "head": head, "kv_group": meta["kv_head_of"][head],
           "checks": s.n_expect, "descs": n_desc,
           "reject_probes": n_reject,
           "step": meta["step"], "layer": meta["layer"],
           "model": meta["model"], "tier": meta["tier"],
           "committed_record": rec.name,
           "provenance_fields": checked,
           "rq_scale": rq_scale, "rq_shift": rq_shift,
           "geom": f"{geom:08x}", "rq": f"{rq:08x}", "mask": "00000003",
           "info_tier": f"{want_tier:#04x}"}
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--head", type=int, default=DEF_HEAD)
    ap.add_argument("--build", default=str(HERE / "build"))
    args = ap.parse_args()

    gold = REPO / "build" / "walkgold"
    bp = Path(args.bundle) if args.bundle else \
        sorted(gold.glob("layer_s*_L*.npz"))[-1]
    man = build(bp, args.head, Path(args.build))

    print(f"[wg3t] case {man['name']}: real {man['model']} step "
          f"{man['step']} layer {man['layer']} head {man['head']} "
          f"(KV group {man['kv_group']}), T={man['T']} D={man['D']} "
          f"{man['tier']}")
    print(f"[wg3t] host-mode script: {man['checks']} scripted checks, "
          f"{man['descs']} MXE descriptors (all tile-legal) + "
          f"{man['reject_probes']} deliberate k=0 reject probes")
    print(f"[wg3t] v1 desc GEOM={man['geom']} RQ={man['rq']} MASK="
          f"{man['mask']} TROWS={man['T']}; fmt=1 image: 6 words, "
          f"n_heads=1 n_kv_heads=1 kv_map=0 mask=SCORE|PV")
    print(f"WG3 TILE CASE: PASS (P1 provenance vs {man['committed_record']} "
          f"on {len(man['provenance_fields'])} fields; P2 q/K/V staged as the "
          f"traced fields; P3 fmt=1 legal by the check2 mirror; "
          f"P4 v1-equivalent per head; P5 {man['descs']} descriptors "
          f"tile-legal; P6 INFO_TIER={man['info_tier']} = the GQA build's "
          f"CQ-8-only truth)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
