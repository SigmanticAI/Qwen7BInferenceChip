#!/usr/bin/env python3
"""gen_qpath_case.py — I-C IC-QPATH: the GAP A / GAP C closure case.

Drives a q row through the tile's NEW narrow-AND-rope sink and gates every
stage of it against the SIGNED arbiter `decoder_layer_fx(bus=BUS_ON)`
(D-030) — never against a re-derivation of tile semantics.

    projection GEMM   h8[T] @ Wq                 (real MXE, K = D_model)
      -> MODE_F16     ONE RNE narrowing           == transformer.py:521 (NP-r)
      -> rope_row     q phase bank at q_pos       == transformer.py:538
      -> q_sink       l_fsrc_ext = 3 (new route)
      -> feeder       C-1 per-row INT8 quant      == attention.py:381 (Q7)
      -> act stage

The two gates, both against arbiter FIELDS:
    TAPF16  the 128 post-RoPE seam beats  ==  f64_to_f16_bits(r.q_rope)
    EFS     the feeder row scale          ==  r.heads[0].s_q

WHY THIS CASE AND ONLY THIS CASE (IB_LAYER.md §3c-3, gap D): `CFG_D` sizes
BOTH the per-head rope row and the D_model-wide activation stage, so a build
can only drive a real q projection where head_dim == D_model. Of the six
shipped L4 cases exactly `l4_h1_hd128` satisfies that (H=1, hd=128, D=128).
The other five join when C-KSPLIT lands (stage 6); `l4_bias` additionally
needs gap B. This is stated, not worked around.

A NEW SIBLING of gen_l3_vectors.py — that generator is IMPORTED and its
verified phase builders are called; no op-stream grammar is authored here
(the B1 §7 rule).

ARMS (the drive is byte-identical across all three; only the EXPECTATION
moves, so each arm isolates one composition step — the W-G3 F5 idiom)
    --arm sink      the new path. MUST PASS.
    --arm norope    GAP A discrimination: expectations pinned to the
                    un-ROTATED row (what MODE_QUANT's Q7 shortcut reaches).
                    MUST FAIL.
    --arm nonarrow  GAP C discrimination: expectations pinned to rope applied
                    to the UN-NARROWED exact product — the one-RNE narrowing
                    removed and nothing else. MUST FAIL.
If either discrimination arm ever passes, the gate has lost its power on that
step and the corresponding gap is no longer being tested.

SELF-CHECKS (hard asserts, no simulator):
  Q1  the arbiter's own composition is reproduced by golden PUBLIC fns in
      golden's order (f16 -> rope_fx -> quant_rows_i8), bit-exactly, so the
      expectations below are arbiter FIELDS and not a second implementation
  Q2  the injected h8/Wq reproduce the arbiter's pre-narrow q_real exactly
  Q3  the rotation is NON-TRIVIAL (q_pos != 0 => the row really rotates)
  Q4  the narrowing is LOAD-BEARING: the arbiter's exact-product row is off
      the fp16 grid, so MODE_QUANT's Q7 shortcut cannot reach r.s_q
  Q5  every emitted MXE descriptor is tile-legal (1<=n<=8, 1<=m<=64,
      1<=k<=2048)
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
sys.path.insert(0, str(REPO / "verif" / "top" / "l3"))
sys.path.insert(0, str(REPO / "verif" / "top" / "l4"))

import gen_l3_vectors as glv                                   # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import transformer as tf                      # noqa: E402
from apex_golden.fp import f64_to_f16_bits                     # noqa: E402
from l4_cases import TIER, build_case                          # noqa: E402

MXE_N, M_TILE_MAX, K_MAX = 8, 64, 2048
LAYER_CTRL, LAYER_PTR, LAYER_DATA, LAYER_STATUS = 0x70, 0x74, 0x78, 0x80
PH_BANK_Q = 1                      # LAYER_PTR[29:28] = 1 selects ph_q
CASE = "l4_h1_hd128"               # the one gap-D-drivable shipped case


def layer_ctrl(*, rope_en=0, rope_bank=0, rope_pos=0, fsrc_ext=0):
    """apex_top.sv LAYER_CTRL: [0] rope_en, [1] rope_bank, [3:2] ser_dst,
    [5:4] fsrc_ext, [6] resid_arm, [14:8] rope_pos, [17:15] kv_map."""
    assert 0 <= rope_pos < 128 and 0 <= fsrc_ext < 4
    return ((rope_en & 1) | ((rope_bank & 1) << 1) | ((fsrc_ext & 3) << 4)
            | ((rope_pos & 0x7F) << 8))


def arbiter(name):
    """The signed arbiter run + the fields this gate pins to."""
    X, w, q_pos = build_case(name)
    r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
    T, D = r.T, r.D_model
    m_q = T if q_pos is None else int(q_pos)
    assert r.H == 1 and r.head_dim == D, \
        f"gap D: this drive needs head_dim == D_model, got {r.head_dim}/{D}"

    # ── Q2: the pre-narrow projection, re-derived with PUBLIC golden fns ────
    s_h = tf.f16_bits_to_f64(r.s_h)
    acc_q = tf.gemm_i8_ksplit(r.h8[T:], np.asarray(w.Wq, dtype=np.int64))[0]
    comp_q = tf.f16_grade(float(s_h[T]) * w.s_wq)            # NP-s
    q_exact = acc_q.astype(np.float64) * comp_q              # PRE-narrow
    assert np.array_equal(tf._f16(q_exact), r.q_real), \
        "Q2: re-derived projection does not reproduce the arbiter's q_real"

    # ── Q1: golden's order reproduces the arbiter's own fields ─────────────
    theta = tf.rope_theta(D, w.rope_theta_base)
    q_rope = tf.rope_fx(r.q_real, m_q, theta)
    assert np.array_equal(q_rope, r.q_rope), \
        "Q1: f16 -> rope_fx does not reproduce the arbiter's q_rope"
    _, s_chk = at.quant_rows_i8(r.q_rope[None, :])
    assert int(s_chk[0]) == int(r.heads[0].s_q), \
        "Q1: quant_rows_i8(q_rope) does not reproduce the arbiter's s_q"

    # ── Q3: the rotation is real ───────────────────────────────────────────
    n_rot = int((f64_to_f16_bits(r.q_rope) != f64_to_f16_bits(r.q_real)).sum())
    assert m_q != 0 and n_rot > 0, \
        f"Q3: rotation is trivial at q_pos={m_q} ({n_rot} elements move)"

    # ── Q4: the narrowing is load-bearing on THIS row ──────────────────────
    # `nonarrow` = the SAME rotation applied to the un-narrowed exact product.
    # If that reproduced q_rope the narrowing would be unobservable here and
    # the gap-C arm would be a decoration, so the delta is asserted non-zero.
    off_grid = int((q_exact != r.q_real).sum())
    q_nonarrow = tf.rope_fx(q_exact, m_q, theta)
    _, s_exact = at.quant_rows_i8(q_nonarrow[None, :])
    n_narrow = int((f64_to_f16_bits(q_nonarrow) !=
                    f64_to_f16_bits(r.q_rope)).sum())
    assert off_grid > 0, "Q4: the exact-product row is already on the f16 grid"
    assert n_narrow > 0, \
        "Q4: dropping the narrowing changes no beat — the gap-C arm is blind"

    return dict(X=X, w=w, r=r, T=T, D=D, m_q=m_q, comp_q=comp_q,
                q_rope16=f64_to_f16_bits(r.q_rope).reshape(D),
                q_real16=f64_to_f16_bits(r.q_real).reshape(D),
                q_nonarrow16=f64_to_f16_bits(q_nonarrow).reshape(D),
                s_q=int(r.heads[0].s_q), s_q_exact=int(s_exact[0]),
                n_rot=n_rot, off_grid=off_grid, n_narrow=n_narrow,
                theta=theta)


def build(a, arm, out: Path) -> dict:
    r, T, D, w = a["r"], a["T"], a["D"], a["w"]
    BPR = D // 8
    s = glv.Script(D)
    name = f"qpath_{CASE}_{arm}"
    s.emit(f"// I-C IC-QPATH gap-A case: {name} (D={D} T={T} CQ-8, "
           f"arbiter = decoder_layer_fx BUS_ON / D-030)")

    glv.phase_a(s, tiers_used=(0,))

    # ── front half: x8 + gamma1 -> RMSNorm-1 -> feeder C-1 -> act bank0 ────
    # The L4 segment-2 front half, verbatim in shape; `s_h` per row is the
    # arbiter's own field, so h8 lands in the act stage golden-exactly.
    s.emit("// front half: RMSNorm-1 -> feeder C-1 (h8 -> act bank0)")
    s.route(fsrc=0, fdst=0, asrc=0)
    x8, _ = at.quant_rows_i8(tf._f16(np.asarray(a["X"], dtype=np.float64)))
    g1 = [int(v) for v in np.asarray(w.gamma1)]
    s.aj(0, 0, 0, T + 1, BPR, 0)
    s.fjob(T + 1)
    for t in range(T + 1):
        s.xrow(x8[t])
        s.grow(g1)
        s.efs(int(r.s_h[t]), t == T)

    # ── the RoPE q phase table -> LAYER RAM bank 1 (ph_q) ──────────────────
    ph = np.asarray(tf.rope_phase_q(float(a["m_q"]), a["theta"]),
                    dtype=np.uint16)
    assert ph.shape == (D // 2,), ph.shape
    assert int(ph.max()) < (1 << 14), "phase code exceeds the 14-bit RAM word"
    s.emit(f"// q phase row -> LAYER RAM bank {PH_BANK_Q} (ph_q), "
           f"{D // 2} codes at q_pos={a['m_q']}")
    s.csrw(LAYER_PTR, (PH_BANK_Q << 28) | 0)
    for c in range(D // 2):
        s.csrw(LAYER_DATA, int(ph[c]))

    # ── the expectation tables ─────────────────────────────────────────────
    if arm == "sink":
        tap_vals = [int(v) for v in a["q_rope16"]]
        efs_val = a["s_q"]
        note = ("post-RoPE seam beats == f64_to_f16_bits(r.q_rope); "
                "feeder scale == r.heads[0].s_q")
    elif arm == "norope":
        # GAP A discrimination: the un-ROTATED row — what the pre-I-C Q7
        # shortcut reaches, since rope_row had no path to the act stage.
        tap_vals = [int(v) for v in a["q_real16"]]
        efs_val = a["s_q"]
        note = (f"GAP-A arm: the UN-ROTATED row ({a['n_rot']}/{D} beats move "
                f"under rotation) — MUST FAIL")
    else:
        # GAP C discrimination: rotation kept, the one-RNE narrowing removed.
        tap_vals = [int(v) for v in a["q_nonarrow16"]]
        efs_val = a["s_q_exact"]
        note = (f"GAP-C arm: rope on the UN-NARROWED exact product "
                f"({a['n_narrow']}/{D} beats differ) — MUST FAIL")
    s.emit(f"// expect {len(tap_vals)} beats on dbg_f16: {note}")
    s.tap("TAPF16", tap_vals, 16)

    # ── phase D': the q projection through the NARROW-AND-ROPE SINK ────────
    # asrc=0: the act stage now loads from the FEEDER, not from squant's Q7
    # beat — that is the whole point of gap A.
    s.emit("// phase D': q projection -> MODE_F16 (one RNE) -> rope_row "
           "(q bank) -> q_sink -> feeder C-1 -> act stage")
    s.route(rdst=1, fsrc=0, fdst=0, asrc=0)
    s.csrp(glv.CSR["STATUS"], 0x1, 0x1)
    s.csrp(LAYER_STATUS, 0xF, 0x0)          # rr | res | swg | ldq all idle
    s.csrw(LAYER_CTRL, layer_ctrl(rope_en=1, rope_bank=1,
                                  rope_pos=a["m_q"] & 0x7F, fsrc_ext=3))
    comp_bits = glv.f32_bits_exact(a["comp_q"], "NP-s q composite")
    s.qjob(0, D)                              # MODE_F16: the ONE RNE
    for _ in range(D):
        s.qs(comp_bits)
    s.fjob(1)                                 # feeder: one row -> C-1
    s.ljob(BPR, 8)
    for j in range(BPR):
        s.aj(1, 0, 0, 1, BPR, T)              # act EMIT: h8 row T
        s.desc(glv.desc_words(glv.OP_GEMM_WS, 1, D, 8))
        s.wbeats(glv.wgt_beats_ws(np.asarray(w.Wq, dtype=np.int64), j))
    s.efs(efs_val, 1)                         # the feeder scale IS s_q
    s.aj(0, 1, 0, 1, BPR, 0)                  # LOAD q8 -> act bank1 row 0

    # ── the q row must NOT have become a KV record ─────────────────────────
    # In q_sink mode apex_top forces kv_s_tvalid low, so the rotated q row
    # reaches neither the KVQ write port nor the store-time scale snoop. This
    # case stores NO K/V at all, so the engine's occupancy must still read 0:
    # a leaking sink would have committed a record here. (Without this check
    # the leak is invisible — mutant Q2 caught the omission, not the RTL.)
    s.emit("// the sink must not leak into the KVQ: occupancy still 0")
    s.kvp(glv.KV["STATUS"], 0x1, 0x1)
    s.kvp(glv.KV["OCC"], 0xFFFFFFFF, 0)

    s.emit("// disarm the sink and quiesce")
    s.csrp(glv.CSR["STATUS"], 0x1, 0x1)
    s.csrp(LAYER_STATUS, 0xF, 0x0)
    s.csrw(LAYER_CTRL, layer_ctrl())
    glv.final_phase(s, 8)

    # ── Q5: descriptor legality against the CONSUMER's rule ────────────────
    n_desc = n_reject = 0
    for ln in s.lines:
        f = ln.split()
        if f[:1] != ["DESC"]:
            continue
        v = 0
        for i, wd in enumerate(f[1:5]):
            v |= int(wd, 16) << (32 * (3 - i))
        m_, k_, n_ = (v >> 8) & 0xFFF, (v >> 20) & 0xFFF, (v >> 32) & 0xFFF
        if 1 <= n_ <= MXE_N and 1 <= m_ <= M_TILE_MAX and 1 <= k_ <= K_MAX:
            n_desc += 1
        else:
            assert k_ == 0, f"Q5: unexpected illegal DESC: {ln}"
            n_reject += 1

    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "cases" / f"{name}.txt").write_text("\n".join(s.lines) + "\n")
    man = dict(name=name, arm=arm, case=CASE, D=D, T=T, q_pos=a["m_q"],
               checks=s.n_expect, tap_beats=len(tap_vals),
               descriptors=n_desc, reject_probes=n_reject,
               rotated_elements=a["n_rot"], off_grid_elements=a["off_grid"],
               s_q=a["s_q"], s_q_exact_product=a["s_q_exact"],
               arbiter="decoder_layer_fx(bus=BUS_ON) — D-030")
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return man


def corpus(n_seeds=14):
    """The segment-1 lane's q_pos=0 corpus, RECONSTRUCTED (its seeds were not
    committed) and pinned here as a permanent golden-side register: the tile's
    NEW order reproduces the arbiter's s_q on every row, while the pre-I-C
    exact-product order does not. Same shape as the lane's measurement —
    14 seeds x 3 bias-free geometries x their heads = 70 head-rows — and the
    same ±1 ULP signature; the miss COUNT is seed-dependent and is therefore
    reported, not inherited (the lane measured 21/70 on its own seeds)."""
    from l4_cases import make_weights
    geoms = [(2, 64, 256, 8, 0, 1e4), (1, 128, 344, 16, 0, 1e4),
             (2, 64, 256, 12, 1, 1e4)]
    n = ok_new = ok_old = 0
    ulp = []
    for si in range(n_seeds):
        for (H, hd, dff, T, H_kv, theta) in geoms:
            rng = np.random.default_rng(0x1C000000 + si * 0x10 + H)
            w = make_weights(rng, H, hd, dff, H_kv=H_kv, theta=theta)
            D = H * hd
            X = rng.normal(0, 1, (T + 1, D))
            r = tf.decoder_layer_fx(X, w, TIER, q_pos=0, bus=tf.BUS_ON)
            s_h = tf.f16_bits_to_f64(r.s_h)
            acc = tf.gemm_i8_ksplit(r.h8[T:],
                                    np.asarray(w.Wq, dtype=np.int64))[0]
            q_exact = acc.astype(np.float64) * tf.f16_grade(
                float(s_h[T]) * w.s_wq)
            for h in range(H):
                sl = slice(h * hd, (h + 1) * hd)
                want = int(r.heads[h].s_q)
                _, sn = at.quant_rows_i8(r.q_rope[sl][None, :])
                _, se = at.quant_rows_i8(q_exact[sl][None, :])
                n += 1
                ok_new += int(sn[0]) == want
                if int(se[0]) == want:
                    ok_old += 1
                else:
                    ulp.append(abs(int(se[0]) - want))
    assert ok_new == n, f"corpus: narrow-first missed {n - ok_new}/{n} rows"
    assert ok_old < n, "corpus: the pre-I-C order never differs — no gap C"
    assert max(ulp) == 1, f"corpus: deltas are not ±1 ULP (max {max(ulp)})"
    return n, ok_new, ok_old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(HERE / "build"))
    ap.add_argument("--arm", choices=("sink", "norope", "nonarrow"), default="sink")
    args = ap.parse_args()
    out = Path(args.build)
    out.mkdir(parents=True, exist_ok=True)

    a = arbiter(CASE)
    man = build(a, args.arm, out)

    n, ok_new, ok_old = corpus()
    print(f"[qpath] case {man['name']}: D={man['D']} T={man['T']} "
          f"q_pos={man['q_pos']}, {man['checks']} scripted checks "
          f"({man['tap_beats']} tap beats + feeder scale)")
    print(f"[qpath] Q3 rotation moves {man['rotated_elements']}/{man['D']} "
          f"elements; Q4 the exact product is off the f16 grid in "
          f"{man['off_grid_elements']}/{man['D']} elements")
    print(f"[qpath] s_q: arbiter={man['s_q']} "
          f"exact-product(pre-I-C)={man['s_q_exact_product']}")
    print(f"[qpath] CORPUS (q_pos=0, {n} head-rows, 14 seeds x 3 geometries): "
          f"narrow-first == r.s_q {ok_new}/{n}; "
          f"exact-product == r.s_q {ok_old}/{n} "
          f"-> {n - ok_old}/{n} differ by 1 ULP")
    print(f"[qpath] arbiter = {man['arbiter']}")
    print(f"QPATH CASE: PASS (Q1 golden order reproduces the arbiter's "
          f"q_rope AND s_q; Q2 re-derived projection == r.q_real; "
          f"Q3 rotation non-trivial; Q4 narrowing load-bearing; "
          f"Q5 {man['descriptors']} descriptors tile-legal, "
          f"{man['reject_probes']} sanctioned reject probes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
