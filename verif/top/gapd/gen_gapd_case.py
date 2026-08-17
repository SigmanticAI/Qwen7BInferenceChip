#!/usr/bin/env python3
"""gen_gapd_case.py — I-C GAP-D (IB_LAYER.md §3c-3): the SPLIT-BUILD case set.

GAP D measured: `CFG_D` was ONE build parameter feeding two incompatible
width families in apex_top — the PER-HEAD family (rope_row, the KVQ record
geometry, the phase RAM) and the D_MODEL-WIDE family (the feeder C-1 row,
the act/weight stage contraction rows).  One build therefore required
head_dim == D_model, and only 1 of the 6 shipped L4 cases (l4_h1_hd128)
was drivable through the S-2 -> rope composition.

THE SPLIT under test: apex_top now carries `CFG_DM` (D_model-wide row
length, default CFG_D == pre-split builds byte-identically).  This
generator drives the FIVE previously-blocked H=2/head_dim=64/D_model=128
L4 cases through a REAL apex_top built at the split point

    CFG_D = 64 (head_dim)   CFG_DM = 128 (D_model)

replaying the O3'-1 composition per KV head — the exact drive §3c-3 had to
fence to l4_h1_hd128:

    h8 (real RMSNorm-1 + feeder C-1, staged at nb = CFG_DM/8 = 16)
      -> K projection GEMM (K = 128 contraction, REAL MXE)
      -> serializer -> apex_scale_quant MODE_F16 (the ONE RNE narrowing)
      -> rope_row per KV head (CFG_D = 64: pairs INSIDE the head)
      -> dbg_f16 tap == f64_to_f16_bits(r.K_rope[:, head])   [bit-exact]

plus the V rows (rope off, == r.V_real) and the q row per query head
(q bank, == r.q_rope).  l4_bias additionally runs through apex_proj_bias
(PROJ_BIAS_EN=1 build) with rope ON — the full Qwen S-2 ordering
rope(Wx+b): bias add -> ONE narrowing -> per-head rotation.

THE ARBITER is decoder_layer_fx(bus=BUS_ON) — D-030.  Golden is never
edited; every expectation is a LayerFx field or re-derived with PUBLIC
golden functions and asserted against a LayerFx field BEFORE any RTL runs
(the §3c-3 premise, re-verified per case at generation time).

Self-checks (hard asserts, no simulator):
  G1  front half re-derive: x8 -> rmsnorm_fx_wide == r.h, C-1 == r.h8/r.s_h
  G2  K path: _f16(gemm(h8,Wk)·graded comp) == r.K_real (bits), and
      rope_fx per KV head == r.K_rope BIT-EXACT (the O3'-1 premise)
  G3  V path: gemm·comp == r.V_real;  q path: _f16 == r.q_real and
      rope_fx per query head == r.q_rope
  G4  every phase code fits the 14-bit RAM word; the ph_k table row for
      position t is rope_phase_q(t, theta)
  G5  mDM1 discriminator: the C-1 scale of h[0]'s FIRST 64 elements
      differs from r.s_h[0] — a feeder forced back to CFG_D=64 rows must
      fail the very first EFS check (signature, not luck)
  G6  (bias case) exact_elem window: every biased element exact-in-window
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
from l4_cases import TIER, build_case                          # noqa: E402

from apex_golden import transformer as tf                      # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

CSR, KV = glv.CSR, glv.KV
LAYER = {"CTRL": 0x70, "PTR": 0x74, "DATA": 0x78, "STATUS": 0x80}
PH_BANK_K, PH_BANK_Q, BIAS_BANK = 0, 1, 3
L_BIAS_EN = 1 << 18                       # LAYER_CTRL[18] (IB_LAYER.md §3d)

# the five §3c-3 GAP-D-blocked cases (head_dim != D_model); l4_bias runs
# only in the PROJ_BIAS_EN=1 build (its own binary), the other four in one
SPLIT_CASES = ["l4_h2_hd64", "l4_gqa_h2kv1", "l4_qwen_theta", "l4_selfinc"]
BIAS_CASES = ["l4_bias"]


def layer_ctrl(*, rope_en=0, rope_bank=0, rope_pos=0, bias_en=0):
    assert 0 <= rope_pos < 128
    return ((rope_en & 1) | ((rope_bank & 1) << 1) | ((rope_pos & 0x7F) << 8)
            | (L_BIAS_EN if bias_en else 0))


def selfcheck_front(name, X, w, r):
    """G1 — the segment-2 front-half re-derive (public fns only)."""
    Xf = tf._f16(np.asarray(X, dtype=np.float64))
    x8, _ = tf.quant_rows_i8(Xf)
    g1 = [int(v) for v in np.asarray(w.gamma1)]
    for t in range(r.T + 1):
        y, _, _ = tf.rmsnorm_fx_wide([int(v) for v in x8[t]], g1)
        assert np.array_equal(np.asarray(y, dtype=np.int64), r.h[t]), \
            f"{name}: h[{t}] re-derive"
    h8, s_h = tf.quant_rows_i8(r.h.astype(np.float64) / 256.0)
    assert np.array_equal(h8, r.h8) and np.array_equal(s_h, r.s_h), \
        f"{name}: h8/s_h re-derive"
    return x8, g1, [int(v) for v in r.s_h]


def graded_comps(w, r):
    """BUS_ON per-row composites, asserted fp16-grade fp32 (C2)."""
    s_h64 = f16_bits_to_f64(np.asarray(r.s_h, dtype=np.uint16))
    T = r.T
    ck = [glv.f32_bits_exact(float(tf.f16_grade(float(s_h64[t]) * w.s_wk)),
                             f"comp_k[{t}]") for t in range(T)]
    cv = [glv.f32_bits_exact(float(tf.f16_grade(float(s_h64[t]) * w.s_wv)),
                             f"comp_v[{t}]") for t in range(T)]
    cq = glv.f32_bits_exact(float(tf.f16_grade(float(s_h64[T]) * w.s_wq)),
                            "comp_q")
    return ck, cv, cq


def kpath_selfcheck(name, w, r, m_q):
    """G2/G3 — re-derive the K/V/q S-2+rope composition from PUBLIC fns and
    assert it reproduces the LayerFx fields bit-exactly (the §3c-3 premise,
    now on a head_dim != D_model case)."""
    T, hd, H = r.T, r.head_dim, r.H
    H_kv = r.H_kv or H
    theta = tf.rope_theta(hd, w.rope_theta_base)
    acc_k = tf.gemm_i8_ksplit(r.h8[:T], np.asarray(w.Wk, dtype=np.int64))
    acc_v = tf.gemm_i8_ksplit(r.h8[:T], np.asarray(w.Wv, dtype=np.int64))
    acc_q = tf.gemm_i8_ksplit(r.h8[T:], np.asarray(w.Wq, dtype=np.int64))[0]
    ck, cv, cq = graded_comps(w, r)
    K_re = acc_k.astype(np.float64) * np.array(
        [f16_bits_to_f64(np.array([0], np.uint16))[0] * 0 +  # keep dtype f64
         float(np.uint32(c).view(np.float32)) for c in ck])[:, None]
    V_re = acc_v.astype(np.float64) * np.array(
        [float(np.uint32(c).view(np.float32)) for c in cv])[:, None]
    q_re = acc_q.astype(np.float64) * float(np.uint32(cq).view(np.float32))
    if w.bk is not None:
        K_re = K_re + np.asarray(w.bk, dtype=np.float64)[None, :]
        V_re = V_re + np.asarray(w.bv, dtype=np.float64)[None, :]
        q_re = q_re + np.asarray(w.bq, dtype=np.float64)
    K_re, q_re = tf._f16(K_re), tf._f16(q_re)      # the ONE S-2 narrowing
    assert np.array_equal(f64_to_f16_bits(K_re), f64_to_f16_bits(r.K_real)), \
        f"{name}: K_real re-derive"
    assert np.array_equal(f64_to_f16_bits(V_re), f64_to_f16_bits(r.V_real)), \
        f"{name}: V_real re-derive"
    assert np.array_equal(f64_to_f16_bits(q_re), f64_to_f16_bits(r.q_real)), \
        f"{name}: q_real re-derive"
    pos_k = np.arange(T)
    for g in range(H_kv):
        sl = slice(g * hd, (g + 1) * hd)
        got = tf.rope_fx(r.K_real[:, sl], pos_k, theta)
        assert np.array_equal(got, r.K_rope[:, sl]), \
            f"{name}: K_rope head {g} re-derive (per-head rope != arbiter)"
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        got = tf.rope_fx(r.q_real[sl], m_q, theta)
        assert np.array_equal(got, r.q_rope[sl]), \
            f"{name}: q_rope head {h} re-derive"
    return acc_k, acc_v, acc_q, ck, cv, cq, theta


def emit_case(name, out: Path, *, with_bias: bool) -> dict:
    X, w, q_pos = build_case(name)
    r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos, bus=tf.BUS_ON)
    D, T, hd, H = r.D_model, r.T, r.head_dim, r.H
    H_kv = r.H_kv or H
    m_q = T if q_pos is None else int(q_pos)
    half = hd // 2
    assert hd != D, f"{name} is not a GAP-D case (head_dim == D_model)"
    assert with_bias == (w.bq is not None)

    x8, g1, s_h = selfcheck_front(name, X, w, r)               # G1
    acc_k, acc_v, acc_q, ck, cv, cq, theta = \
        kpath_selfcheck(name, w, r, m_q)                       # G2/G3

    # G4 — phase tables (14-bit codes; ph_k row t == rope_phase_q(t, theta))
    ph_k = np.asarray(tf.rope_phase_q(np.arange(T, dtype=np.float64), theta),
                      dtype=np.int64)                          # [T, half]
    ph_q = np.asarray(tf.rope_phase_q(float(m_q), theta), dtype=np.int64)
    assert ph_k.shape == (T, half) and ph_q.shape == (half,)
    assert int(ph_k.max(initial=0)) < (1 << 14) \
        and int(ph_q.max(initial=0)) < (1 << 14), "G4: phase code width"

    # G5 — the mDM1 discriminator: a feeder forced back to CFG_D=64 frames
    # the h stream as 64-element chunks, so EFS[t] would see the scale of
    # chunk t (= h[t//2][(t%2)*64:...]) instead of s_h[t]. Find the FIRST
    # EFS index inside the still-drivable window ((T+1)*64 elements consumed
    # before the stream stalls => EFS[0 .. (T+1)//2-1] execute) where the
    # two differ — the signature the mDM1 checker requires VERBATIM.
    flat = r.h.astype(np.float64).reshape(-1) / 256.0
    win = (T + 1) // 2
    divs = []
    for t in range(win):
        _, s_c = tf.quant_rows_i8(flat[t * 64:(t + 1) * 64][None, :])
        if int(s_c[0]) != int(r.s_h[t]):
            divs.append((t, int(s_c[0]), int(r.s_h[t])))
    assert divs, f"{name}: G5 discriminator degenerate in EFS window [0,{win})"
    mdm1 = {"efs_idx": divs[0][0], "got": divs[0][1], "exp": divs[0][2]}

    # G6 — bias case: every element exact-in-window (proven shape, IC-BIAS)
    b16 = {}
    if with_bias:
        for tag, vec in (("bq", w.bq), ("bk", w.bk), ("bv", w.bv)):
            b16[tag] = f64_to_f16_bits(np.asarray(vec, np.float64)) \
                .astype(np.uint16)
            assert np.array_equal(f16_bits_to_f64(b16[tag]),
                                  np.asarray(vec)), f"{name}: {tag} grid"

    K_exp = f64_to_f16_bits(r.K_rope).astype(np.uint16)        # rotated
    V_exp = f64_to_f16_bits(r.V_real).astype(np.uint16)        # unrotated
    q_exp = f64_to_f16_bits(r.q_rope).astype(np.uint16)        # rotated

    DM_BPR, HD_BPR = D // 8, hd // 8
    s = glv.Script(D)                       # D_model framing (BPR = 16)
    s.emit(f"// GAP-D split-build case {name}: CFG_D={hd} CFG_DM={D} "
           f"H={H} H_kv={H_kv} T={T} theta={w.rope_theta_base:g} "
           f"m_q={m_q} bias={'on' if with_bias else 'off'}")
    s.emit("// arbiter: decoder_layer_fx(BUS_ON).  dbg_f16 taps ==")
    s.emit("//   f64_to_f16_bits(r.K_rope / r.V_real / r.q_rope) per head")

    # ── preamble: build truth + illegal-desc probe (phase-A idiom) ─────────
    s.csrr(CSR["INFO_VER"], 0xFFFFFFFF, 0x00010000)
    s.csrr(CSR["INFO_D"], 0xFFFFFFFF, hd)   # INFO_D = the PER-HEAD family
    s.csrr(CSR["INFO_G"], 0xFFFFFFFF, glv.G_CFG)
    s.csrr(CSR["INFO_TIER"], 0xFFFFFFFF, 0x3)      # OUTK=0 build
    s.csrw(CSR["CTRL"], 0x1)
    s.csrw(CSR["TIER_CTRL"], 0x0)                  # CQ-8
    s.kvw(KV["CTRL"], 0x2)
    s.kvp(KV["STATUS"], 0x1, 0x1)
    s.desc(glv.desc_words(glv.OP_GEMM_WS, 1, 0, 1))  # K=0: §3 reject probe
    s.csrp(CSR["STATUS"], 0x2, 0x2)
    s.csrw(CSR["STATUS"], 0x2)

    # ── every dbg_f16 expectation, in drive order: K rows, V rows, q row ───
    taps = []
    for t in range(T):
        for g in range(H_kv):
            taps += list(K_exp[t, g * hd:(g + 1) * hd])
    for t in range(T):
        for g in range(H_kv):
            taps += list(V_exp[t, g * hd:(g + 1) * hd])
    taps += list(q_exp)
    s.tap("TAPF16", taps, 16)

    # ── phase B (model-wide family): x8 -> RMSNorm-1 -> feeder C-1 (rows of
    #    CFG_DM = 128) -> act stage at nb = CFG_DM/8 = 16 on a CFG_D=64 build
    s.emit(f"// phase B: front half — act LOAD nb={DM_BPR} (CFG_DM beats) on")
    s.emit(f"//   a CFG_D={hd} build: the split-parameter capability itself")
    s.csrp(CSR["STATUS"], 0x1, 0x1)
    s.route()
    s.aj(0, 0, 0, T + 1, DM_BPR, 0)
    s.fjob(T + 1)
    for t in range(T + 1):
        s.xrow(x8[t])
        s.grow(g1)
        s.efs(s_h[t], 1 if t == T else 0)

    # ── phase K (both families composed): per token, per KV head ───────────
    s.emit("// phase K: S-2 MODE_F16 -> rope_row per KV head -> KVQ record")
    s.emit(f"//   ph_k bank 0 loaded once: {T} rows x {half} pair codes")
    s.csrp(CSR["STATUS"], 0x1, 0x1)
    s.route(rdst=1)
    s.csrw(LAYER["PTR"], (PH_BANK_K << 28) | 0)
    for t in range(T):
        for c in range(half):
            s.csrw(LAYER["DATA"], int(ph_k[t][c]))

    def proj_head(W, ncol0, comp, arow, waddr, bias_vec):
        """One head-row: S-2 job (cols=hd) + serializer + the K=CFG_DM GEMM
        from the staged act row (nb = DM_BPR beats)."""
        if bias_vec is not None:
            s.csrp(LAYER["STATUS"], 0x10, 0x0)     # bias unit quiescent
            s.csrw(LAYER["PTR"], (BIAS_BANK << 28) | 0)
            for b in bias_vec[ncol0:ncol0 + hd]:
                s.csrw(LAYER["DATA"], int(b) & 0xFFFF)
        s.kvp(KV["STATUS"], 0x1, 0x1)
        s.kvw(KV["WADDR"], waddr)
        s.qjob(0, hd)                              # MODE_F16, per-head cols
        for _ in range(hd):
            s.qs(comp)
        s.ljob(HD_BPR, 8)
        for j in range(HD_BPR):
            s.aj(1, 0, 0, 1, DM_BPR, arow)         # act row: CFG_DM beats
            s.desc(glv.desc_words(glv.OP_GEMM_WS, 1, D, 8))
            s.wbeats(glv.wgt_beats_ws(W[:, ncol0:ncol0 + hd], j))

    # LEVEL-CHANGE DISCIPLINE (measured the hard way): rope's busy is NOT in
    # the CSR STATUS idle bits — it is LAYER_STATUS[3] (§3b). Changing
    # l_rope_pos while rope_row is still EMITTING the previous row rotates
    # that row's remaining pairs with the NEXT position's phases (observed:
    # first-divergence at the first pair emitted after the CTRL write). So
    # every LAYER_CTRL write below is guarded by BOTH idle polls.
    def rope_idle():
        s.csrp(CSR["STATUS"], 0x1, 0x1)
        s.csrp(LAYER["STATUS"], 0x8, 0x0)

    rec = 0
    for t in range(T):
        rope_idle()
        s.csrw(LAYER["CTRL"], layer_ctrl(rope_en=1, rope_bank=PH_BANK_K,
                                         rope_pos=t, bias_en=with_bias))
        for g in range(H_kv):
            proj_head(np.asarray(w.Wk, dtype=np.int64), g * hd, ck[t], t,
                      rec, b16.get("bk"))
            rec += 1
        if t == 0:
            # mDM3 probe: after the FIRST roped row the LAYER window must be
            # error-clean (a rope_row framed at CFG_DM refuses this row with
            # FRAME code 3 — the specific mutant signature)
            s.csrp(CSR["STATUS"], 0x1, 0x1)
            s.csrr(LAYER["STATUS"], 0x1F00, 0x0000)

    # ── phase V: rope OFF, same per-head S-2 framing ───────────────────────
    s.emit("// phase V: rope off — per-head S-2 narrowing == r.V_real")
    rope_idle()
    s.csrw(LAYER["CTRL"], layer_ctrl(bias_en=with_bias))
    for t in range(T):
        for g in range(H_kv):
            proj_head(np.asarray(w.Wv, dtype=np.int64), g * hd, cv[t], t,
                      rec, b16.get("bv"))
            rec += 1

    # ── phase q: q bank at m_q, one job per QUERY head ─────────────────────
    s.emit(f"// phase q: ph_q bank 1 at m_q={m_q}; per-head rotation of the")
    s.emit("//   decode row (act row T)")
    rope_idle()
    s.csrw(LAYER["PTR"], (PH_BANK_Q << 28) | 0)
    for c in range(half):
        s.csrw(LAYER["DATA"], int(ph_q[c]))
    s.csrw(LAYER["CTRL"], layer_ctrl(rope_en=1, rope_bank=PH_BANK_Q,
                                     rope_pos=m_q & 0x7F, bias_en=with_bias))
    for h in range(H):
        proj_head(np.asarray(w.Wq, dtype=np.int64), h * hd, cq, T,
                  rec, b16.get("bq"))
        rec += 1

    # ── final: quiescence FIRST (the last rope row is still draining when
    #    the ops run ahead), THEN the tap/sticky accounting ─────────────────
    s.emit("// final: quiescence + sticky accounting")
    rope_idle()
    s.kvp(KV["STATUS"], 0x1, 0x1)
    s.csrp(LAYER["STATUS"], 0x1F00, 0x0000)
    s.emit("ENDTAPS")
    s.n_expect += 3
    s.estk(0xFFFF, 0x0001)                  # exactly the phase-A reject
    s.emit("DONE")

    fn = out / f"gapd_{name}.txt"
    fn.write_text("\n".join(s.lines) + "\n")
    st = {"case": name, "H": H, "H_kv": H_kv, "head_dim": hd, "D_model": D,
          "T": T, "taps": len(taps), "expects": s.n_expect,
          "records": rec, "mdm1": mdm1, "file": str(fn)}
    (out / f"gapd_{name}.json").write_text(json.dumps(st, indent=1) + "\n")
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--bias", action="store_true",
                    help="emit the PROJ_BIAS_EN=1 case (l4_bias)")
    a = ap.parse_args()
    out = Path(a.build) / "cases"
    out.mkdir(parents=True, exist_ok=True)
    names = BIAS_CASES if a.bias else SPLIT_CASES
    stats = [emit_case(n, out, with_bias=a.bias) for n in names]
    for st in stats:
        print("  " + json.dumps(st))
    print(f"GAPD CASE: PASS ({len(stats)} split-build case(s), every "
          "expectation asserted against decoder_layer_fx BUS_ON pre-RTL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
