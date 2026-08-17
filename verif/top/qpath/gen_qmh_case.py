#!/usr/bin/env python3
"""gen_qmh_case.py — I-C IC-QPATH, F6 part (ii): the MULTI-HEAD fmt=1 case.

The W-G3 walked half extended past n_heads=1, at real 7B PER-HEAD geometry
(head_dim=128, T=20, CQ-8), driven by the REAL traced Qwen2.5-7B layer
(step 19 / layer 19) — the same bundle, the same provenance chain.

WHAT IS NEW vs `verif/top/wg3/gen_wg3_tile_case.py` (which this file does NOT
edit — the B1 §7 sibling rule):

  * `n_heads = NH > 1` and `n_kv_heads = NH`, `kv_map = 1`, so the walk runs
    the per-head loop across DISTINCT KV groups and exercises BOTH halves of
    F6: part (i)'s per-KV-head composite caches AND part (ii)'s per-head s_q.
  * every head's q row is staged BEFORE WALK_GO through the GAP-A SINK
    (MODE_F16 -> rope_row q bank -> q_sink -> feeder C-1 -> act stage bank 1
    row h), so the tile holds NH q rows and the glue holds NH scales.
  * the descriptor heads are the real 7B heads {0, G, 2G, ...} (G = the
    bundle's GQA group size, 7), i.e. one head per real KV group, so head h
    of the walk maps to real KV group h — exactly the mapping the walker's
    division-free g(h) = floor(h*n_kv/n_heads) produces at n_kv == n_heads.

WHY THIS IS A REAL TEST OF F6(ii): the traced layer's 28 heads carry 27
DISTINCT s_q values. A tile that kept one s_q — the pre-I-C single sticky —
composes every head's CS words from the last-staged scale, so all but one
head's scores are wrong. The gate is the `noqstage` arm below, which builds
the SAME image against a QSTAGE_H_MAX=1 tile and must FAIL.

SELF-CHECKS (hard asserts, no simulator):
  M1  ERRATUM F5 ON THE q SIDE: the walkgold bundle is a LEGACY (bus-off)
      trace — its `q_real` is off the fp16 grid and its `q_rope`/`q_f16` are
      rope applied to that un-narrowed row. Every expectation here is
      composed from the D-030 order instead (`_f16` -> `rope_fx` ->
      `quant_rows_i8`, public fns only), and M1 asserts it really DIFFERS
      from the legacy field, so a drift back onto the trap cannot pass.
  M2  each staged PRE-rope row is fp16-idempotent, so what MODE_F16 hands
      rope_row is exactly the narrowed row the arbiter rotates
  M3  the chosen heads' s_q values are DISTINCT (else the case could pass on
      a single-s_q tile and would prove nothing)
  M4  the fmt=1 image passes the check2 mirror, and n_heads <= the build's
      QSTAGE_H_MAX / STAGE_R_MAX
  M5  every emitted MXE descriptor is tile-legal
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

import gen_l3_vectors as glv                                   # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden import transformer as tf                      # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

MXE_N, M_TILE_MAX, K_MAX = 8, 64, 2048
LAYER_CTRL, LAYER_PTR, LAYER_DATA, LAYER_STATUS = 0x70, 0x74, 0x78, 0x80
PH_BANK_Q = 1


def layer_ctrl(*, rope_en=0, rope_bank=0, rope_pos=0, fsrc_ext=0, kv_map=0):
    assert 0 <= rope_pos < 128 and 0 <= fsrc_ext < 4 and 0 <= kv_map < 8
    return ((rope_en & 1) | ((rope_bank & 1) << 1) | ((fsrc_ext & 3) << 4)
            | ((rope_pos & 0x7F) << 8) | ((kv_map & 0x7) << 15))


def build(gold: Path, bundle: Path, nh: int, out: Path, arm: str,
          ngrp: int, walked: bool) -> dict:
    meta = json.loads(bundle.with_suffix(".json").read_text())
    z = dict(np.load(bundle))
    D, T, H_kv, grp = meta["head_dim"], meta["T"], meta["H_kv"], meta["gqa_group"]
    assert D == 128, f"this case targets CFG_D=128, got {D}"
    one_group = (ngrp == 1)
    if one_group:
        assert 1 < nh <= grp, f"n_heads must be in (1, {grp}] inside one group"
    else:
        assert 1 < nh <= H_kv, f"n_heads must be in (1, {H_kv}] one-per-group"
    m_q = T - 1                                    # the traced q position

    # ONE-GROUP mode: the first nh real heads, which all live in KV group 0
    # (the bundle's GQA group size is `grp`), so every head reads the SAME
    # engine and the only per-head state is the q row + its s_q. SPREAD mode:
    # one real head per KV group, which additionally crosses engines.
    heads = list(range(nh)) if one_group else [g * grp for g in range(nh)]
    n_kv = 1 if one_group else nh
    kvmap = 0 if one_group else 1
    q_real = np.asarray(z["q_real"])
    q_rope = np.asarray(z["q_rope"])

    # ── ERRATUM F5 APPLIES TO THE q SIDE TOO (measured here, 2026-07-26) ───
    # The walkgold bundle is a LEGACY (bus-off) trace: its `q_real` is NOT on
    # the fp16 grid, and its `q_rope` / per-head `q_f16` are rope applied to
    # that UN-narrowed row. Pinning the tap to those fields would fail a
    # CORRECT tile, exactly as F5 predicts for K. So every expectation below
    # is composed HERE from the D-030 arbiter's order using PUBLIC golden fns
    # — `_f16` (NP-r, transformer.py:521) then `rope_fx` (:538) then
    # `quant_rows_i8` (Q7) — and M1 asserts the two really do differ, so a
    # future drift back onto the legacy field cannot pass unnoticed.
    theta = tf.rope_theta(D, meta["rope_theta"])
    assert not np.array_equal(tf._f16(q_real), q_real), \
        "F5: the bundle's q_real IS on the f16 grid — re-check which trace " \
        "mode produced it before trusting this case's premise"

    stage16, expect16, s_q, q8_codes, n_legacy_diff = [], [], [], [], 0
    for hi, h in enumerate(heads):
        sl = slice(h * D, (h + 1) * D)
        pre_f64 = tf._f16(q_real[sl])                 # the ONE NP-r narrowing
        rot = tf.rope_fx(pre_f64, m_q, theta)         # the C-ROPE rotation
        pre = f64_to_f16_bits(pre_f64).reshape(D)     # staged PRE-rope
        post = f64_to_f16_bits(rot).reshape(D)        # expected POST-rope
        q8_a, sq_a = at.quant_rows_i8(rot[None, :])   # the C-1 codes + scale
        # M1: the arbiter's composition is NOT the legacy trace field
        legacy = np.asarray(z[f"h{h:02d}_q_f16"], dtype=np.uint16)
        n_legacy_diff += int((post != legacy).sum())
        # M2: staging a row that is already on the grid is idempotent under
        # MODE_F16's narrowing, so the staged row is exactly what rope sees
        assert np.array_equal(f64_to_f16_bits(f16_bits_to_f64(pre)).reshape(D),
                              pre), f"M2: staged row not fp16-idempotent (h{h})"
        stage16.append(pre)
        expect16.append(post)
        s_q.append(int(sq_a[0]))
        q8_codes.append(np.asarray(q8_a[0], dtype=np.int64))
    assert n_legacy_diff > 0, \
        "M1: the D-030 expectation equals the legacy field on every beat — " \
        "this case would not detect an un-narrowed tile (F5 blind spot)"

    # per-head golden attention cores on exactly the state the tile holds:
    # the D-030 rotated q this case stages, and the bundle's K/V rows.
    cores = []
    for hi, h in enumerate(heads):
        sl = slice(h * D, (h + 1) * D)
        rot = tf.rope_fx(tf._f16(q_real[sl]), m_q, theta)
        hk = heads[0] if one_group else h
        cores.append(at.attention_core(
            np.asarray(rot, dtype=np.float64),
            np.asarray(z[f"h{hk:02d}_K_f16"], dtype=np.uint16),
            np.asarray(z[f"h{hk:02d}_V_f16"], dtype=np.uint16),
            at.TIER_CQ8, G=128, outlier_idx=[]))
        assert int(cores[hi].s_q) == s_q[hi], \
            f"core s_q disagrees with the staged scale for head {h}"

    # M3: the heads must be distinguishable by s_q, or the case is blind
    assert len(set(s_q)) == nh, \
        f"M3: chosen heads share an s_q ({s_q}) — a single-s_q tile would pass"

    s = glv.Script(D)
    name = (f"qmh_s{meta['step']:03d}_L{meta['layer']:02d}_nh{nh}"
            f"_g{n_kv}_hd{D}_T{T}_{'walk' if walked else 'host'}")
    s.emit(f"// I-C IC-QPATH F6(ii) MULTI-HEAD case: {name}")
    s.emit(f"// real Qwen2.5-7B heads {heads} (one per KV group), "
           f"n_heads={nh} n_kv_heads={nh} kv_map=1, q_pos={m_q}")
    s.emit(f"// per-head s_q: {s_q} — {len(set(s_q))} distinct")

    glv.phase_a(s, tiers_used=(0,))
    glv.loader_phase(s)

    # ── the q phase row -> LAYER RAM bank 1 (ph_q) ─────────────────────────
    theta = tf.rope_theta(D, meta["rope_theta"])
    ph = np.asarray(tf.rope_phase_q(float(m_q), theta), dtype=np.uint16)
    assert ph.shape == (D // 2,) and int(ph.max()) < (1 << 14)
    s.emit(f"// q phase row -> LAYER RAM bank {PH_BANK_Q}, {D // 2} codes "
           f"at q_pos={m_q}")
    s.csrw(LAYER_PTR, (PH_BANK_Q << 28) | 0)
    for c in range(D // 2):
        s.csrw(LAYER_DATA, int(ph[c]))

    # ── per-engine init (phase_a only reaches engine 0) ────────────────────
    def quiesce():
        s.csrp(glv.CSR["STATUS"], 0x1, 0x1)
        s.kvp(glv.KV["STATUS"], 0x1, 0x1)
        s.csrp(LAYER_STATUS, 0xF, 0x0)

    s.emit(f"// per-engine D-020 soft reset on each of the {n_kv} GQA engines")
    for g in range(n_kv):
        quiesce()
        s.csrw(LAYER_CTRL, layer_ctrl(kv_map=g))
        s.kvw(glv.KV["CTRL"], 0x2)
        s.kvp(glv.KV["STATUS"], 0x1, 0x1)

    # ── K/V records: group g's rows into ENGINE g ─────────────────────────
    # The walked score/PV phases read them back per head, and the composite
    # cache harvests each record's scale on the store snoop — so the walk
    # exercises F6 part (i) (per-KV-head caches) at the same time.
    for g in range(n_kv):
        h = heads[0] if one_group else heads[g]
        K16 = np.asarray(z[f"h{h:02d}_K_f16"], dtype=np.uint16)
        V16 = np.asarray(z[f"h{h:02d}_V_f16"], dtype=np.uint16)
        assert K16.shape == (T, D) and V16.shape == (T, D)
        quiesce()
        s.emit(f"// KV group {g} (real head {h}) -> GQA engine {g}")
        s.csrw(LAYER_CTRL, layer_ctrl(kv_map=g))
        glv.store_kv_phase(s, K16, V16, glv.TIER_CQ8)

    # the post-RoPE q expectation, declared AFTER the K/V beats so the shared
    # dbg_f16 table is in run order
    s.emit(f"// expect {nh * D} POST-RoPE q beats on dbg_f16 (arbiter: the "
           f"D-030 composition, NOT the bundle's legacy q_rope — F5)")
    s.tap("TAPF16", [int(v) for e in expect16 for v in e], 16)

    # ── stage the NH q rows through the GAP-A SINK ────────────────────────
    # ONE arming of the sink (its rising edge resets the glue's s_q file write
    # pointer), then NH rows back to back: entry h <- head h's scale, act
    # bank 1 row h <- head h's codes.
    s.emit(f"// stage {nh} q rows through the gap-A sink: MODE_F16 (one RNE)")
    s.emit(f"//   -> rope_row (q bank, pos {m_q}) -> q_sink -> feeder C-1")
    s.emit(f"//   -> act bank 1 rows 0..{nh - 1}; the glue files each s_q")
    quiesce()
    s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0)
    s.csrw(LAYER_CTRL, layer_ctrl(rope_en=1, rope_bank=1, rope_pos=m_q,
                                  fsrc_ext=3))
    # ONE feeder job + ONE act-stage LOAD per row. It cannot be a single
    # rows=NH job: `inject_jobs` drives the act stage's EMIT port to source
    # each K=2 GEMM, and the stage buffer runs one job at a time (D-006), so a
    # rows=NH LOAD held open across the injections would deadlock against
    # them. Per row, the LOAD's `sel` is the BASE ROW — head h lands in act
    # bank 1 row h, which is what the walker's per-head emission selects.
    for hi in range(nh):
        s.fjob(1)
        pairs = [glv.decompose_f16(int(b))[:2] for b in stage16[hi]]
        glv.inject_jobs(s, pairs, 0)               # MODE_F16, no WADDR
        s.efs(s_q[hi], 1)                          # the row scale IS s_q
        s.aj(0, 1, 0, 1, s.BPR, hi)                # LOAD -> bank 1 row hi

    # ── PER-HEAD READBACK: prove head h's codes are in act bank 1 ROW h ────
    # Staging alone does not show WHERE the codes landed. A selector GEMM
    # does: emit bank 1 row h against W[k][n] = (k==n), so acc[n] = q8[n] and
    # the result beat is head h's first eight C-1 codes, bit-exact vs golden
    # `quant_rows_i8(q_rope)`. Without this the act-stage LOAD base row would
    # be untested and a tile that put every head's row at 0 would look green.
    s.emit(f"// per-head readback: emit act bank 1 row h through a selector "
           f"GEMM, expect head h's own q8[0:8] (golden quant_rows_i8)")
    s.route(rdst=0, fsrc=0, fdst=0, asrc=0, qsrc=0)
    # EVERY 8-element block of every row, not just the first: a row that is
    # right in its head and garbage in its tail would otherwise read as green,
    # and the tail is exactly what a base-row or beat-count fault corrupts.
    for hi in range(nh):
        q8 = q8_codes[hi]
        for blk in range(D // 8):
            sel_w = np.zeros((D, 8), dtype=np.int64)
            for n in range(8):
                sel_w[8 * blk + n][n] = 1
            s.aj(1, 1, 0, 1, s.BPR, hi)            # EMIT bank 1 row hi
            s.desc(glv.desc_words(glv.OP_GEMM_WS, 1, D, 8))
            s.wbeats(glv.wgt_beats_ws(sel_w, 0))
            s.ero([int(v) for v in q8[8 * blk:8 * blk + 8]], 1)

    # Each engine holds EXACTLY its 2T K/V records and nothing more: in
    # q_sink mode apex_top forces kv_s_tvalid low, so the NH rotated q rows
    # reached neither the write port nor the store-time scale snoop. A
    # leaking sink would show up here as extra records.
    s.emit(f"// the sink must not leak into any KVQ engine: occupancy "
           f"still exactly 2T = {2 * T} per engine")
    for g in range(n_kv):
        quiesce()
        s.csrw(LAYER_CTRL, layer_ctrl(kv_map=g))
        s.kvp(glv.KV["OCC"], 0xFFFFFFFF, 2 * T)

    s.emit("// disarm the sink")
    quiesce()
    s.csrw(LAYER_CTRL, layer_ctrl())

    # ── M5: descriptor legality ───────────────────────────────────────────
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
            assert k_ == 0, f"M5: unexpected illegal DESC: {ln}"
            n_reject += 1

    # ── the WALKED region ─────────────────────────────────────────────────
    # The TB opens a walk on a "phase F" marker and closes it on "final:"
    # (both walker-mode-only, so the HOST arm reads straight through). No op
    # lines are needed in between: the WALKER generates the whole score+pv
    # control stream for all NH heads from the fmt=1 image, and per head it
    #   * presents hd_valid/hd_head and waits for the glue's per-head ready,
    #   * re-emits act bank 1 row h  (its q8, via the new q_row_sel),
    #   * requests the CS/QS composites from the cache of KV group g(h),
    #     which the per-head s_q replay has just refreshed.
    # A missing per-head s_q is NOT a silent wrong answer here: the composite
    # unit raises err_stale (`!s_q_val`), the walker latches WALK_ERR, and
    # walk_end fails the run. That is what the `mh_noq` arm demonstrates.
    # ── the WALKED region is emitted ONLY for the walk variant ────────────
    # In host mode nothing drives score/PV, so its expectation ops (ESS/ERO)
    # would hang the interpreter. The HOST variant ends after staging: it is
    # the per-head VALUE gate (post-RoPE beats, s_q, per-head code readback);
    # the WALK variant adds the walked score/PV expectations on top.
    if walked:
      # ALL heads' passive-tap tables are declared BEFORE the walk opens. The
      # TB loads a tap table when it READS the op, and the walker runs ahead of
      # the interpreter (the TB is blocked in head h's ESS/ERO waits while the
      # tile is already streaming head h+1) — declaring them per head inside
      # the walked region loses the race and reports "unexpected beat".
      # Concatenation order == head order == the order the walk produces them.
      s.emit(f"// per-head score/softmax tap tables for all {nh} heads, in "
             f"walk order (declared pre-walk: the walker outruns the TB)")
      for hi in range(nh):
          s.tap("TAPSC", [int(v) for v in cores[hi].score_fx], 32)
      for hi in range(nh):
          s.tap("TAPPR", [int(v) for v in cores[hi].p_q115], 16)

      s.emit(f"// phase F: WALK_GO — the walker drives score+pv for all {nh} "
             f"heads from the fmt=1 image")
      # THE F6(ii) VALUE GATE. score_fx[t] = acc_s[t] * f32(s_q * s_k[t] *
      # 2^frac / sqrt(D)) — the CS composite the walker fetches from the cache
      # of head h's KV group, which the per-head s_q replay refreshes at the
      # head interlock. A tile that reused ONE s_q produces the wrong composite
      # for every head but the last staged, so these taps are RED. They are the
      # reason the walked arm is a test and not a smoke run.
      # THE PER-HEAD KV-GROUP GATE (I-C IC-MHWALK). The expectation set below
      # is the HOST generator's, op for op and `last` flag for `last` flag:
      # gen_l3_vectors.score_phase(:396-398) then :404, and pv_phase(:430-434)
      # then :435. In walker mode the TB skips the CONTROL ops and keeps the
      # EXPECTATION ops, so a walked head must satisfy exactly what a hosted
      # head does — the wg3 idiom, applied per head.
      #
      # WHY EFS IS NOT OPTIONAL HERE (this lane's finding): the walked K/V
      # record reads run through the feeder, whose per-row fp16 scale bus is
      # the tap `fs_*`. Those scales ARE s_k[t] (score) and s_v[t] (pv) of the
      # head's OWN KV group, so they are the direct, value-level proof that
      # head h read group g(h)'s records, in order, and not another group's.
      # Omitting them left 340 undrained beats per head — the final phase's
      # `leftover fs` check caught it — and, worse, left the per-head KVQ read
      # path checked only INDIRECTLY, through the composites.
      for hi in range(nh):
          c = cores[hi]
          s.emit(f"// head {hi} (real 7B head {heads[hi]}, KV group {hi}): "
                 f"score/softmax/P-requant/PV expectations, s_q={s_q[hi]}")
          # score: one feeder job per chunk, scl_last on the chunk's last row
          for c0, ncc in glv.chunks(T):
              for t in range(c0, c0 + ncc):
                  s.efs(int(c.s_k[t]), t == c0 + ncc - 1)
          s.ess(int(c.s_c), 1)
          # pv: BPR output blocks, each re-reading the T V-records in chunks
          for j in range(D // 8):
              for c0, ncc in glv.chunks(T):
                  for t in range(c0, c0 + ncc):
                      s.efs(int(c.s_v[t]), t == c0 + ncc - 1)
              s.ero([int(v) for v in c.o8[8 * j:8 * j + 8]], 1)

    # `final_phase` branches on T only as a PROXY for "did this case raise the
    # score-dequant and TIP frame stickies". The WALK variant runs score/PV at
    # T=20 and does; the HOST variant runs no score/PV at all, so passing 8
    # selects the genuinely-true no-frame-error tail (the krope idiom —
    # asserting the T>8 stickies there would be the lying expectation).
    glv.final_phase(s, T if walked else 8)

    # a GQA build is CQ-8-only (apex_top.sv INFO_TIER override) — rewrite the
    # phase-A expectation by VALUE with an exact-count assert, never by line
    old, new = "CSRR 14 ffffffff 00000003", "CSRR 14 ffffffff 00000001"
    n_hit = sum(1 for ln in s.lines if ln == old)
    assert n_hit == 1, f"expected 1 phase-A INFO_TIER line, found {n_hit}"
    s.lines = [new if ln == old else ln for ln in s.lines]

    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "walker").mkdir(parents=True, exist_ok=True)
    (out / "cases" / f"{name}.txt").write_text("\n".join(s.lines) + "\n")

    # ── the D-029 fmt=1 image: n_heads = nh, per-KV-head record addressing ─
    rq0 = int(cores[0].rq[0]), int(cores[0].rq[1])
    words = {
        fmt.W_GEOM0:  fmt.pack_geom0(D, tier=0),
        # d_model = n_heads * head_dim (the check2 clause). The
        # projection phases are DISABLED in this image (en_mask is
        # SCORE|PV only), so d_model is carried, never walked — gap D
        # is untouched by this case.
        fmt.W_MODEL0: fmt.pack_model0(nh * D, 0),
        fmt.W_MODEL1: fmt.pack_model1(nh, n_kv, kv_map=kvmap),
        fmt.W_MASK:   fmt.pack_mask((1 << fmt.EN_SCORE) | (1 << fmt.EN_PV)),
        fmt.W_STEP:   fmt.pack_step(T, m_q),
    }
    for hi in range(nh):
        sc, sh = int(cores[hi].rq[0]), int(cores[hi].rq[1])
        words[fmt.W_RQ0 + hi] = ((sh & 0x1F) << 16) | (sc & 0xFFFF)
    # M4: the image must pass the tile's OWN legality mirror
    err = fmt.check2(words[fmt.W_GEOM0], words[fmt.W_MODEL0],
                     words[fmt.W_MODEL1], words[fmt.W_MASK],
                     words[fmt.W_STEP], cfg_d=D)
    assert err == fmt.ERR_NONE, f"M4: fmt=1 image illegal (err={err})"
    with (out / "walker" / f"{name}.desc2").open("w") as fh:
        for idx in sorted(words):
            fh.write(f"DW2 {idx} {words[idx]:08x}\n")
        fh.write(f"TROWS {T}\n")
    geom = (D & 0xFF) | ((T & 0x1FF) << 8)
    (out / "walker" / f"{name}.desc").write_text(
        f"WALKABLE 1\nGEOM {geom:08x}\n"
        f"RQ {((rq0[1] & 0x1F) << 16) | (rq0[0] & 0xFFFF):08x}\n"
        f"MASK 00000003\nTROWS {T}\n")

    man = dict(name=name, arm=arm, n_heads=nh, n_kv_heads=n_kv,
               kv_map=kvmap, real_heads=heads, D=D, T=T,
               q_pos=m_q, s_q=s_q, distinct_s_q=len(set(s_q)),
               checks=s.n_expect, tap_beats=nh * D,
               legacy_beats_differ=n_legacy_diff,
               descriptors=n_desc, reject_probes=n_reject,
               step=meta["step"], layer=meta["layer"],
               arbiter="the committed layer bundle (decoder_layer_fx BUS_ON)")
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(HERE / "build"))
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-groups", type=int, default=0,
                    help="0 = one head per KV group (default); 1 = every head in KV group 0")
    ap.add_argument("--mode", default="walk", choices=("host", "walk"),
                    help="host = staging + per-head readback only; "
                         "walk = adds the walked score/PV gate")
    ap.add_argument("--arm", default="qstage",
                    choices=("qstage", "noqstage"))
    args = ap.parse_args()
    gold = REPO / "build" / "walkgold"
    bp = sorted(gold.glob("layer_s*_L*.npz"))[-1]
    man = build(gold, bp, args.heads, Path(args.build), args.arm,
                args.kv_groups, args.mode == "walk")
    print(f"[qmh] case {man['name']}: {man['n_heads']} heads "
          f"(real 7B heads {man['real_heads']}, one per KV group), "
          f"D={man['D']} T={man['T']} q_pos={man['q_pos']}")
    print(f"[qmh] per-head s_q {man['s_q']} — {man['distinct_s_q']} distinct "
          f"(a single-s_q tile cannot satisfy them all)")
    print(f"[qmh] {man['checks']} scripted checks, {man['tap_beats']} "
          f"post-RoPE tap beats, {man['descriptors']} tile-legal descriptors")
    print(f"[qmh] F5: expectations are the D-030 composition, NOT the "
          f"bundle's legacy q_rope/q_f16 fields "
          f"({man['legacy_beats_differ']}/{man['tap_beats']} beats differ)")
    print(f"QMH CASE: PASS (M1 D-030 expectation differs from the legacy "
          f"field; M2 staged rows fp16-idempotent; M3 s_q distinct "
          f"{man['distinct_s_q']}/{man['n_heads']}; M4 fmt=1 image legal; "
          f"M5 {man['descriptors']} descriptors tile-legal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
