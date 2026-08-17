#!/usr/bin/env python3
"""gen_layer_trace.py — IB-WALK stage 1: the fmt=1 FULL-LAYER op-stream spec.

The L3 pattern lifted to the decoder layer (IB_WALK.md §4 stage 1, gate
W-G1): this generator IS the spec the stage-2 walker RTL will be equivalence-
gated against, exactly as gen_l3_vectors.py's choreography is the D-028 spec.
It never edits the golden or the L3 generator; every tensor-dependent field
it emits is asserted equal to the corresponding `decoder_layer_fx` field at
generation time (golden-gated field-by-field), so the spec cannot drift from
the arbiter.

Per case it writes build/layer/<name>.ops:
  CASE/DW      the 64-word fmt=1 descriptor image (seq_walker_fmt.py layout;
               WSCALE words resv-0 per contract — s_w* carried as annotation
               lines until IB-LAYER answers Q1)
  FETCH        walker->reader fuel_req per weight-consuming step (frozen
               64-bit layout: base[29:0]/beats[55:30]/tag[63:56], 64 B units)
  DESC         projection GEMM jobs in decomposition order (n-major,
               k-minor, accumulate=(k0>0), rq_en on the LAST k-split of
               epilogue-bearing projections). ENCODING NOTE: emitted as
               OP_GEMM_OS + mode_os=1 + accumulate — the C-KSPLIT on-tile
               realization available under the FROZEN descriptor ("host-side,
               or on-tile via the accumulate flag", compute.py). Q1 may swap
               this to a WS-accumulate variant; regenerate on IB-LAYER's
               ruling.
  E LCTL/LJOB/LUJOB/LJC/KVWA — stage 5: the CONCRETE §3b step emissions
               (IB_LAYER.md §3b FROZEN): LAYER level words, norm jobs,
               LAYER_JOB pushes, alternating JOBC composites (rewritten per
               chunk), per-engine store WRITE_ADDR writes. The former
               PENDING-Q1 markers are retired.
  HEAD/E       per-head attention score+pv streams in the EXACT D-028 unit
               grammar (gen_walker_vectors.py "E" lines), engine select per
               §9.1 R3 (golden mapping h // (H//H_kv) — see the R3 note the
               selftest prints), composites via the stage-0 oracle.
  RQSLOT       the host-loaded requant table (per-head PV + oproj + down),
               golden-gated through the epilogue replica.

Self-gates (hard asserts): epilogue replica outputs == fx.attn_proj /
fx.ffn_out bit-exact; per-head cs/qs == walker_composite_golden over the
head's scales; CQ-8 store-scale identity (cache-source premise); emission
counts == the B1_STAGE1_NOTES §6 formulas; decomposition == golden chunking;
descriptor image passes the check2 mirror; determinism; and — D-029 erratum,
I-B — EVERY emitted DESC (attention and projection alike) is legal by the
TILE's own acceptance rule, `seq_walker_fmt.assert_mxe_legal`.

That last gate exists because this file is the walker's ORACLE: the stage-2
scoreboard compares walker emissions against expectations built from the same
decomposition constants, so a uniformly-wrong walker+mirror pair shows green.
It did: the n-split was sized by the 12-bit descriptor FIELD (4095) instead
of the implemented array width (MXE_N = 8), and every 7B projection
descriptor this spec blessed would have been refused by mxe_ctrl.

The SECOND ERRATUM (2026-07-30 audit) is the same shape in the LAYER lane:
LAYER_JOB `cols` was chunked at the 12-bit FIELD for every unit, but apex_top
wires only 7 of those bits to `asu_swiglu #(.COLS_MAX(64))`, so the tail chunk
of d_ffn=18944 (2564) arrived as 4 and was ACCEPTED — a silent wrong answer,
no job_error. Hence the second sweep, `seq_walker_fmt.assert_layer_job_legal`,
over every `E LUJOB` this spec emits.

The THIRD ERRATUM (W1, 2026-08-05) is the same shape in the k lane: the k
chunk was the frozen K_JOB (2048) for every D, but a k-deep job's ACT FAMILY
is k/CFG_D stage rows and apex_stage_buf caps R_MAX at 31 — so at D=64 a
2048-chunk (32 rows) was tile-illegal and the 0.5B down projection
(d_ffn=4864) could not walk. Hence the D-aware chunk
(`seq_walker_fmt.k_job`) and the third sweep, `assert_act_stageable`, over
every DESC this spec emits at the case's own row width.

Test-only knobs: `--mutate-n-mxe N` / `--mutate-lu-swiglu N` /
`--mutate-k-rows N` [`--out DIR`] regenerate with the n-split (resp. swiglu
cols, stage-row cap) bound overridden, which the matching legality gate must
reject — the Makefile's `mutants4` / `mutants5` / `mutants6` targets use
them to prove these gates are not decorative.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO / "verif" / "top" / "l3"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import seq_walker_fmt as fmt  # noqa: E402
import walker_composite_golden as wcg  # noqa: E402
from apex_golden import attention as at  # noqa: E402
from apex_golden import compute as cp  # noqa: E402
from apex_golden import transformer as tf  # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402

# ── cases: the two verif/layer tile-scale shapes + the 7B geometry ───────────
#   (name, H, head_dim, H_kv, d_ffn, T, seed, qwen_biases[, nfeed])
# The first three are FROZEN — their mask is EN_ALL (NFEED-free), so their
# .ops/.sub.ops are byte-identical across the E-3 change and they are the
# legacy-identity evidence. `layer_nf_hd128` is the E-3 addition: the SAME
# toy-128 geometry the E-lane demonstrates on (head_dim == D_model == 128,
# E2E_TOY_LANE.md §2), with the norm-feed step armed.
CASES = [
    ("layer_h2_hd64", 2, 64, 1, 256, 8, 0x1B0A11, False),
    ("layer_h1_hd128", 1, 128, 1, 344, 16, 0x1B0A12, False),
    ("qwen7b_T8", 28, 128, 4, 18944, 8, 0x1B0A13, True),
    ("layer_nf_hd128", 1, 128, 1, 344, 16, 0x1B0A12, False, True),
    # THIRD ERRATUM (W1, 2026-08-05): the Qwen2.5-0.5B geometry at D=64 —
    # the shape whose down projection (Wd, k_total = d_ffn = 4864) forced
    # the D-AWARE k chunk: at K_JOB=2048 its act family is 32 stage rows and
    # apex_stage_buf caps R_MAX at 31, so a fmt=1 walk of this layer emitted
    # two TILE-ILLEGAL descriptors at DOWN. With k_job(64)=1984 the split is
    # [1984, 1984, 896] = [31, 31, 14] rows — every job stageable, proven by
    # the third sweep below. The four cases above are FROZEN and
    # byte-identical across this change (their k totals never exceed 1984 at
    # D=64, and k_job(128) == K_JOB).
    ("layer_05b_hd64", 14, 64, 2, 4864, 8, 0x1B0A31, True),
]


def f16grid(x: np.ndarray) -> np.ndarray:
    return f16_bits_to_f64(f64_to_f16_bits(np.asarray(x, dtype=np.float64)))


def make_weights(rng, H, head_dim, H_kv, d_ffn, biases: bool):
    """verif/layer/gen_layer_vectors.py recipe + GQA + optional Qwen biases.
    Weights stored int8 (golden casts to int64 on use) so the 7B FFN matrices
    stay ~68 MB each instead of 543 MB."""
    D = H * head_dim
    kv = H_kv * head_dim

    def w(shape):
        return rng.integers(-127, 128, shape).astype(np.int8)

    s_proj = 1.0 / (127.0 * math.sqrt(D))
    s_down = 1.0 / (127.0 * math.sqrt(d_ffn))
    # s_wu deliberately differs from s_wg: with the verif/layer recipe's
    # shared s_proj the GATE and UP JOBC composites would be equal and a
    # comp_a/comp_b order swap (mutant m10) would be invisible
    s_up = s_proj * 0.5
    g = lambda: np.array([int(round((0.7 + 0.6 * rng.random()) * 8192))  # noqa: E731
                          for _ in range(D)], dtype=np.int64)
    kw = {}
    if biases:
        kw = dict(bq=f16grid(rng.normal(0, 0.02, D)),
                  bk=f16grid(rng.normal(0, 0.02, kv)),
                  bv=f16grid(rng.normal(0, 0.02, kv)))
    return tf.LayerWeights(
        Wq=w((D, D)), Wk=w((D, kv)), Wv=w((D, kv)), Wo=w((D, D)),
        Wg=w((D, d_ffn)), Wu=w((D, d_ffn)), Wd=w((d_ffn, D)),
        s_wq=s_proj, s_wk=s_proj, s_wv=s_proj, s_wo=s_proj,
        s_wg=s_proj, s_wu=s_up, s_wd=s_down,
        gamma1=g(), gamma2=g(), H=H, head_dim=head_dim, H_kv=H_kv, **kw)


def proj_epilogue_replica(vec_real, W8, s_w):
    """_proj_epilogue's internals, replicated to EXPOSE the (scale, shift)
    the golden computes but does not return. Gated by asserting the replica's
    OUTPUT equals the golden's field bit-exactly — if the recomputed acc or
    rq differed, the outputs would differ."""
    a8, s_a = at.quant_rows_i8(np.asarray(vec_real, dtype=np.float64)[None, :])
    acc = cp.gemm_i8_ksplit(a8, np.asarray(W8, dtype=np.int64))[0].astype(np.int64)
    amax = int(np.max(np.abs(acc), initial=0)) or 1
    scale, shift = at.calib_requant(amax)
    o8 = cp.requant_i32_to_i8(acc, scale, shift).astype(np.int64)
    s_a_val = float(f16_bits_to_f64(np.array([s_a[0]]))[0])
    s_out = s_a_val * float(s_w) * float(1 << shift) / float(scale)
    return o8.astype(np.float64) * s_out, (int(scale), int(shift)), s_out


# ── per-head attention emitters (the D-028 ROM, B1_STAGE1_NOTES §1/§2) ───────

def score_ops(D, T, s_q_bits, s_k_bits, s_v_bits):
    BPR = D // 8
    nbt = (T + 7) // 8
    split = nbt > BPR
    L = ["E ROUTE 00ef", f"E SJOB {T:x}"]
    L += [f"E CS {wcg.score_composite(int(s_q_bits), int(s_k_bits[t]), D):08x}"
          for t in range(T)]
    L += [f"E QJOB 1 {T:x}"]
    L += [f"E QS {wcg.p_requant_composite(int(s_v_bits[t])):08x}"
          for t in range(T)]
    for ci in range((T + 7) // 8):
        c0 = 8 * ci
        nc = min(8, T - c0)
        L += [f"E WJ 0 0 0 {nc:x} {BPR:x} 0", f"E FJOB {nc:x}"]
        L += [f"E KVW {t:x}" for t in range(c0, c0 + nc)]
        L += [f"E WJ 1 0 1 {nc:x} {BPR:x} 0", f"E AJ 1 1 0 1 {BPR:x} 0",
              f"E DESC 02 1 {D:x} {nc:x} 0 0 0 1"]
    L += [f"E AJ 0 1 0 1 {min(nbt, BPR):x} 0"]
    if split:
        L += [f"E AJ 0 0 0 1 {nbt - BPR:x} 0"]
    return L


def pv_ops(D, T, rq_scale, rq_shift):
    BPR = D // 8
    nbt = (T + 7) // 8
    split = nbt > BPR
    L = ["E ROUTE 00cf"]
    for j in range(BPR):
        L += [f"E AJ 1 1 0 1 {min(nbt, BPR):x} 0",
              f"E DESC 02 1 {T:x} 8 1 {rq_scale:x} {rq_shift:x} 1"]
        if split:
            L += [f"E AJ 1 0 0 1 {nbt - BPR:x} 0"]
        for ci in range((T + 7) // 8):
            c0 = 8 * ci
            nc = min(8, T - c0)
            L += [f"E WJ 0 0 0 {nc:x} {BPR:x} 0", f"E FJOB {nc:x}"]
            L += [f"E KVW {T + t:x}" for t in range(c0, c0 + nc)]
            L += [f"E WJ 1 0 2 {nc:x} 1 {j:x}"]
    return L


def drive_formulas(D, T):
    """B1_STAGE1_NOTES §6, walker-emitted drive ops per head."""
    BPR = D // 8
    NCH = (T + 7) // 8
    split = NCH > BPR
    score = 1 + 1 + T + 1 + T + 5 * NCH + T + (2 if split else 1)
    pv = 1 + BPR * ((2 if split else 1) + 1 + 3 * NCH + T)
    return score, pv


def check_desc_legality(lines, case: str) -> int:
    """D-029 erratum blind-spot gate (generator side): replay every DESC
    emission this spec claims through the TILE's acceptance rule.

    Field positions are shared by the 8-field (v1 attention) and 9-field
    (stage-2 projection) forms: [2]=opcode [3]=m [4]=k [5]=n, all hex."""
    n = 0
    for ln in lines:
        f = ln.split()
        if len(f) < 6 or f[0] != "E" or f[1] != "DESC":
            continue
        fmt.assert_mxe_legal(int(f[3], 16), int(f[4], 16), int(f[5], 16),
                             f"{case}: '{ln}'")
        n += 1
    return n


def check_lujob_legality(lines, case: str) -> int:
    """SECOND-ERRATUM blind-spot gate (generator side): replay every LAYER_JOB
    push this spec claims through the CONSUMING UNIT's acceptance rule.

    The same argument as check_desc_legality one lane over — this file is the
    walker's ORACLE, so a walker+mirror pair that both chunk LAYER_JOB cols at
    the 12-bit field shows green while asu_swiglu (a 7-bit port) silently
    computes 4 columns of the 2564 it was handed. Line form: `E LUJOB <unit>
    <cols>`, both hex."""
    n = 0
    for ln in lines:
        f = ln.split()
        if len(f) != 4 or f[0] != "E" or f[1] != "LUJOB":
            continue
        fmt.assert_layer_job_legal(int(f[2], 16), int(f[3], 16),
                                   f"{case}: '{ln}'")
        n += 1
    return n


def check_desc_stageable(lines, case: str, cfg_d: int) -> int:
    """THIRD-ERRATUM blind-spot gate (generator side): replay every DESC
    emission's ACT FAMILY through the STAGE BUFFER's acceptance rule.

    The same argument as check_desc_legality one consumer further on: a
    descriptor's k can be MXE-legal (k <= 2048) while its activation family
    is unstageable (k/CFG_D rows > apex_stage_buf R_MAX=31 at D=64), and a
    walker+mirror pair that both chunk k at the frozen K_JOB shows green on
    a stream no build can feed. Reads seq_walker_fmt.SB_* only (never the
    decomposition bounds); the `mutants6` target proves it fires."""
    n = 0
    for ln in lines:
        f = ln.split()
        if len(f) < 6 or f[0] != "E" or f[1] != "DESC":
            continue
        fmt.assert_act_stageable(int(f[4], 16), cfg_d, f"{case}: '{ln}'")
        n += 1
    return n


def desc9(line: str) -> str:
    """v1 8-field 'E DESC op m k n rqen rs rh mos' -> stage-2 9-field form
    with accumulate inserted after n (attention jobs never accumulate)."""
    f = line.split()
    assert f[0] == "E" and f[1] == "DESC" and len(f) == 10, line
    return " ".join(f[:6] + ["0"] + f[6:])


# ── projection DESC emission (decomposition order; §2.6) ─────────────────────

def proj_desc_lines(k_total, n_total, m_dim, rq_pair=None, prefix="",
                    n_job=None, cfg_d=None):
    """rq_pair=(scale, shift) puts the epilogue on the LAST k-split of every
    n-split (the accumulate-chain end); None = no epilogue (dequant path).
    prefix='E ' renders TB-expectation lines for the .sub.ops stream.
    `cfg_d` is the build's stage-row width — the k chunk is D-dependent
    (third erratum; seq_walker_fmt.k_job)."""
    L = []
    ks = fmt.jobs(k_total, n_total, n_job, cfg_d=cfg_d)
    last_k0 = max(k0 for (_, k0, _, _, _) in ks)
    for (n0, k0, k, n, acc) in ks:
        ep = rq_pair is not None and k0 == last_k0
        rs, rh = rq_pair if ep else (0, 0)
        L.append(f"{prefix}DESC 02 {m_dim:x} {k:x} {n:x} {acc} {1 if ep else 0} "
                 f"{rs:x} {rh:x} 1")
    return L


# ── case builder ─────────────────────────────────────────────────────────────

def build_case(name, H, head_dim, H_kv, d_ffn, T, seed, biases, inject=None,
               n_job=None, nfeed=False):
    """`inject=(w, X, fx)` replaces the synthetic recipe with an already-built
    layer step — the W-G3 hook (`verif/walkgold/`) that feeds a REAL traced
    Qwen2.5-7B layer through this same compiler. `n_job` overrides the N-split
    width (see `seq_walker_fmt.jobs`). All three default to the original
    behaviour, so the three synthetic cases stay byte-identical.

    `nfeed=True` (E-3, E2E_TOY_LANE.md §4) sets the previously-reserved mask
    bit EN_NFEED, which arms the walker's in-tile norm-feed step at the
    r1 -> NORM2 seam: LCTL with feeder source code 4, the C-1 feeder job for
    the row family, and the LAYER_JOB unit-3 NORM/EGRESS push — the three
    verbs the HOST demonstrator issues (scripts/fpga/f2/elane_norm_feed.py
    stage [3]), in that order, emitted by the sequencer instead."""
    D = H * head_dim
    if inject is None:
        rng = np.random.default_rng(seed)
        w = make_weights(rng, H, head_dim, H_kv, d_ffn, biases)
        # S8 self-inclusive composition: rows [x_0..x_{T-1}] + the decode row
        # duplicating x_{T-1}, q at RoPE position T-1 (transformer.py q_pos)
        Xc = rng.normal(0, 1, (T, D))
        X = np.vstack([Xc, Xc[-1]])
        fx = tf.decoder_layer_fx(X, w, "CQ-8", G=128, q_pos=T - 1)
    else:
        w, X, fx = inject

    # golden-gated epilogue requant pairs (the host-loaded RQ table)
    o_rep, rq_o, s_out_o = proj_epilogue_replica(fx.attn, w.Wo, w.s_wo)
    assert np.array_equal(o_rep, fx.attn_proj), f"{name}: oproj replica"
    d_rep, rq_d, s_out_d = proj_epilogue_replica(fx.swiglu, w.Wd, w.s_wd)
    assert np.array_equal(d_rep, fx.ffn_out), f"{name}: down replica"

    # stage 5: the JC table (§3b JOBC composites, host-loaded per step like
    # the RQ pairs). s_h2 is the golden's own quantizer replayed on the
    # golden's own h2 field — identical by construction (fx does not expose
    # it). Values are grade-narrowed by THIS lane's definition (fp16 RNE,
    # exact f32 widen) — reconcile vs D-030's canonical narrowing at combine.
    _h2_8, s_h2m = at.quant_rows_i8(fx.h2.astype(np.float64)[None, :] / 256.0)
    s_h2 = float(f16_bits_to_f64(np.array([s_h2m[0]]))[0])
    jc = [0] * fmt.JC_SLOTS
    jc[fmt.JC_OPROJ] = fmt.grade_f32(s_out_o)
    jc[fmt.JC_DOWN] = fmt.grade_f32(s_out_d)
    jc[fmt.JC_GATE] = fmt.grade_f32(s_h2 * w.s_wg)
    jc[fmt.JC_UP] = fmt.grade_f32(s_h2 * w.s_wu)

    # CQ-8 store-scale identity per head (the scale-cache premise, §A-2):
    # the scale of the STORED row equals the feeder's read-time scale
    group = H // (w.H_kv or H)
    for h, hd_res in enumerate(fx.heads):
        sk_store = at.quant_rows_i8(f16_bits_to_f64(hd_res.K_f16))[1]
        sv_store = at.quant_rows_i8(f16_bits_to_f64(hd_res.V_f16))[1]
        assert np.array_equal(sk_store, hd_res.s_k), f"{name} h{h}: s_k identity"
        assert np.array_equal(sv_store, hd_res.s_v), f"{name} h{h}: s_v identity"

    # R3: per-engine record count fits DEPTH=256
    assert 2 * T <= 256, f"{name}: 2T={2 * T} exceeds engine DEPTH"

    # descriptor image
    shapes = fmt.tensor_shapes(D, d_ffn, H_kv, head_dim)
    bases = fmt.image_bases(shapes)
    nbytes = fmt.tensor_bytes(shapes)
    words = [0] * fmt.DESC_WORDS
    words[fmt.W_GEOM0] = fmt.pack_geom0(head_dim)
    words[fmt.W_MODEL0] = fmt.pack_model0(D, d_ffn)
    words[fmt.W_MODEL1] = fmt.pack_model1(H, H_kv, 1)
    # FFN interleave (2026-08-08): DOWN decoupled from EN_FFN — the
    # composed full layer sets BOTH bits explicitly (EN_ALL itself stays
    # frozen: it feeds legacy byte-identity checks).
    # EN_DOWN joins only when d_ffn satisfies the E-7b DOWN alignment
    # fences (64-chunk AND CFG_D-row aligned); unaligned stress shapes
    # (l128 DF=344) walk FFN-only — DOWN stays refused by its own fence.
    dn = (1 << fmt.EN_DOWN) if (d_ffn % 64 == 0 and d_ffn % head_dim == 0) \
         else 0
    words[fmt.W_MASK] = fmt.pack_mask(
        dn | (fmt.EN_ALL_NFEED if nfeed else fmt.EN_ALL))
    for t in range(fmt.TENS_N):
        words[fmt.W_TENS0 + t] = fmt.pack_tens(bases[t])
    words[fmt.W_STEP] = fmt.pack_step(T, T - 1)
    for h, hd_res in enumerate(fx.heads):
        words[fmt.W_RQ0 + h] = fmt.pack_rq(*hd_res.rq)
    words[fmt.W_RQ0 + H] = fmt.pack_rq(*rq_o)
    words[fmt.W_RQ0 + H + 1] = fmt.pack_rq(*rq_d)
    for i in range(fmt.JC_SLOTS):
        words[fmt.W_JC0 + i] = jc[i]
    assert fmt.check2(words[0], words[1], words[2], words[3],
                      words[fmt.W_STEP], cfg_d=head_dim) == fmt.ERR_NONE
    kv_dim = H_kv * head_dim

    def fetch(t, chunk=None, cols=64):
        # chunk=c: the FFN interleave's per-chunk record — a 64-col slice
        # of Wg/Wu at base + c*d_model beats, d_model beats long (the
        # contiguous column-major job blocks; seq_layer_walker2
        # ffn_chunk_off, 2026-08-08)
        if chunk is None:
            b = bases[t]
            n = fmt.beats64_of_bytes(nbytes[t])
        else:
            n = fmt.beats64_of_bytes(D * cols)
            b = bases[t] + chunk * fmt.beats64_of_bytes(D * 64)
        fmt.pack_freq(b, n, t)                       # width-check via asserts
        return f"E FETCH {t:x} {b:x} {n:x}"

    L = [f"CASE {name} fmt=1 D={head_dim} T={T} H={H} HKV={H_kv} "
         f"DM={D} DF={d_ffn} pos={T - 1} tier=CQ8"]
    L += [f"DW {i:02d} {words[i]:08x}" for i in range(fmt.DESC_WORDS)]
    for nm, sv in [("wq", w.s_wq), ("wk", w.s_wk), ("wv", w.s_wv),
                   ("wo", w.s_wo), ("wg", w.s_wg), ("wu", w.s_wu),
                   ("wd", w.s_wd)]:
        L += [f"WSCALE {nm} {np.float32(sv).view(np.uint32):08x}"]

    # ── template order (IB_WALK.md §2.3, concrete per IB_LAYER.md §3b) ──────
    # ROPE is a LEVEL arm (the phase-K table is per-config RESIDENT in the
    # LAYER RAM — l_rope_bank=0 reads row l_rope_pos; no per-step row fetch).
    # deq/swiglu/residual jobs are ARMED (LCTL level, JC composites, LUJOB
    # pushes) BEFORE the GEMMs that produce their streams — the L3
    # arm-before-stream discipline. Choreography reconciled at combine
    # against LAYER's canonical L4 host order (IB_LAYER.md §3b): rules 1/3
    # were CONFIRMED as derived; rule 2 (consumer-before-producer) is
    # APPLIED in steps_tail — residual before deq in both chained flows.
    lv_rope = fmt.lctl(rope_en=1, rope_pos=T - 1)
    lv_o = fmt.lctl(rope_en=1, rope_pos=T - 1, ser_dst=1, resid_arm=1)
    lv_g = fmt.lctl(rope_en=1, rope_pos=T - 1, ser_dst=2, resid_arm=1)
    lv_d = fmt.lctl(rope_en=1, rope_pos=T - 1, ser_dst=1, resid_arm=1,
                    fsrc_ext=2)                  # down-proj eats swiglu-p
    # E-3: the norm-feed level. The walker's NFEED step touches ONE field —
    # fsrc_ext -> code 4 (the residual row's internal egress) — so the word
    # is the LAST level word (lv_o, armed by PC_LVLO) with only that field
    # moved. Composing it that way here is what makes the spec sensitive to
    # a walker that re-armed the other levels behind the change.
    lv_nf = fmt.lctl(rope_en=1, rope_pos=T - 1, ser_dst=1, resid_arm=1,
                     fsrc_ext=4)
    assert lv_nf == lv_o | 0x80, "NFEED level must move fsrc_ext ALONE"

    def steps_tail():
        """The post-attention step stream (shared by .ops and .sub.ops).

        CONSUMER-BEFORE-PRODUCER (combine, IB_LAYER.md §3b choreography
        rule 2 / IB_WALK.md §4 stage-5 flag (b)(i)): in the chained
        deq->residual flows the RESIDUAL job is pushed BEFORE the deq
        JOBC+JOB (LUJOB 2 before LJC+LUJOB 0) — the walker ROM's one-pc
        swap, mirrored here. JC still lands IMMEDIATELY before ITS push
        (rule 3: the push resets the JOBC index)."""
        S = [f"E LCTL {lv_o:04x}", f"E LUJOB 2 {D:x}",
             f"E LJC {jc[fmt.JC_OPROJ]:08x}", f"E LUJOB 0 {D:x}",
             fetch(fmt.TENS_WO)] + proj_desc_lines(D, D, 1, rq_pair=rq_o,
                                                   prefix="E ", n_job=n_job,
                                                   cfg_d=head_dim)
        # E-3 (optional step): r1 -> C-1 -> RMSNorm, entirely in-tile. Emitted
        # BEFORE the gamma-2 fetch — the row reaches the norm first and gamma
        # releases its emission, exactly the host demonstrator's order. The
        # feeder job frames the row family (d_model / FEED_DM rows == H at
        # head_dim == FEED_DM, the walker's division-free row count) and the
        # unit-3 push is UNCHUNKED (it names a window, not a stream).
        if nfeed:
            S += [f"E LCTL {lv_nf:04x}", f"E FJOB {H:x}",
                  f"E LUJOB {fmt.LU_NORM} {D:x}"]
        S += [fetch(fmt.TENS_G2), f"E LJOB {D:x}"]
        # swiglu cols split at the 12-bit field; JOBC rewritten per chunk
        # (the §3b index-reset rule)
        S += [f"E LCTL {lv_g:04x}"]
        # LAYER_JOB cols: the CONSUMING UNIT's bound (fmt.lu_chunks(.., unit)),
        # NOT the MXE array width the projection jobs use and NOT the 12-bit
        # field either. SECOND ERRATUM (2026-07-30 audit): this call used to
        # take the field bound for every unit, which split d_ffn=18944 into
        # [4095,4095,4095,4095,2564] — and apex_top wires only 7 of those 12
        # bits to asu_swiglu (COLS_MAX=64), so the tail chunk arrived as 4 and
        # was accepted. 296x64 is the legal split.
        # FFN INTERLEAVE (2026-08-08, fence-2 closure): per 64-col chunk —
        # the swiglu job + its JC pair, the Wg chunk record + gate jobs,
        # the Wu chunk record + up jobs; the walker loops PC_USWI..PC_WUJ
        # d_ffn/64 times (the emission stream confirmed verb-for-verb
        # against the RTL, run2_l64 trace_all 2026-08-08). The old
        # all-swiglu-then-all-gate-then-all-up order deadlocked the real
        # datapath (STEP_MATRIX FENCE 2) and never ran outside this TB.
        # the chunk width comes from the SAME single-source bound the
        # m14 spec mutant overrides (fmt.LU_CHUNK[LU_SWIGLU] — the swiglu
        # unit's COLS_MAX), so a generator sized by the wrong bound still
        # emits illegal LUJOBs the legality gate must refuse.
        C = fmt.LU_CHUNK[fmt.LU_SWIGLU]
        nch = (d_ffn + C - 1) // C
        for c in range(nch):
            cw = min(C, d_ffn - C * c)
            S += [f"E LJC {jc[fmt.JC_GATE]:08x}",
                  f"E LJC {jc[fmt.JC_UP]:08x}",
                  f"E LUJOB 1 {cw:x}"]
            S += [fetch(fmt.TENS_WG, chunk=c, cols=cw)]                + proj_desc_lines(D, cw, 1, prefix="E ", n_job=n_job,
                                 cfg_d=head_dim)
            S += [fetch(fmt.TENS_WU, chunk=c, cols=cw)]                + proj_desc_lines(D, cw, 1, prefix="E ", n_job=n_job,
                                 cfg_d=head_dim)
        # DOWN (PC_LVLD/UDOWN/WDF/WDJ) composes only on aligned shapes
        # (the E-7b fences); PC_URES2 rides its OWN mask bit and runs
        # regardless — pc order LVLD(26) URES2(27) UDOWN(28).
        aligned = d_ffn % 64 == 0 and d_ffn % head_dim == 0
        if aligned:
            S += [f"E LCTL {lv_d:04x}"]
        S += [f"E LUJOB 2 {D:x}"]
        if aligned:
            S += [f"E LJC {jc[fmt.JC_DOWN]:08x}", f"E LUJOB 0 {D:x}"]
            S += [fetch(fmt.TENS_WD)] + proj_desc_lines(d_ffn, D, 1,
                                                        rq_pair=rq_d,
                                                        prefix="E ",
                                                        n_job=n_job,
                                                        cfg_d=head_dim)
        return S

    def steps_head():
        """The pre-attention step stream (shared by .ops and .sub.ops)."""
        S = [fetch(fmt.TENS_G1), f"E LJOB {D:x}"]
        S += [fetch(fmt.TENS_WQ)] + proj_desc_lines(D, D, 1, prefix="E ",
                                                    n_job=n_job,
                                                    cfg_d=head_dim)
        S += [fetch(fmt.TENS_WK)] + proj_desc_lines(D, H_kv * head_dim, T,
                                                    prefix="E ", n_job=n_job,
                                                    cfg_d=head_dim)
        S += [fetch(fmt.TENS_WV)] + proj_desc_lines(D, H_kv * head_dim, T,
                                                    prefix="E ", n_job=n_job,
                                                    cfg_d=head_dim)
        S += [f"E LCTL {lv_rope:04x}"]
        # engine id rides IN the expectation (checked by the scoreboard)
        for g in range(H_kv):
            S += [f"E KVWA {g:x} {T - 1:x}", f"E KVWA {g:x} {2 * T - 1:x}"]
        return S

    L += ["STEP NORM1+QKV+ROPE+STOREKV"] + steps_head()
    n_att = 0
    for h, hd_res in enumerate(fx.heads):
        g = h // group
        L += [f"HEAD {h:x} ENG {g:x} RQSLOT {h:x}"]
        sc = score_ops(head_dim, T, hd_res.s_q, hd_res.s_k, hd_res.s_v)
        pv = pv_ops(head_dim, T, *hd_res.rq)
        f_sc, f_pv = drive_formulas(head_dim, T)
        assert len(sc) == f_sc and len(pv) == f_pv, \
            f"{name} h{h}: emitter vs §6 formulas ({len(sc)}/{f_sc}, " \
            f"{len(pv)}/{f_pv})"
        L += sc + pv
        n_att += len(sc) + len(pv)
    L += ["STEP OPROJ+RES1+NORM2+FFN+RES2"] + steps_tail()
    L += ["END"]

    for h in range(H):
        L.insert(1 + fmt.DESC_WORDS + 7 + h,
                 f"RQSLOT {h:x} {fx.heads[h].rq[0]:x} {fx.heads[h].rq[1]:x}")
    L.insert(1 + fmt.DESC_WORDS + 7 + H,
             f"RQSLOT {H:x} {rq_o[0]:x} {rq_o[1]:x}")
    L.insert(1 + fmt.DESC_WORDS + 7 + H + 1,
             f"RQSLOT {H + 1:x} {rq_d[0]:x} {rq_d[1]:x}")

    n_proj = 2 * len(fmt.jobs(D, D, n_job, cfg_d=head_dim)) \
        + 2 * len(fmt.jobs(D, kv_dim, n_job, cfg_d=head_dim)) \
        + 2 * len(fmt.jobs(D, d_ffn, n_job, cfg_d=head_dim)) \
        + len(fmt.jobs(d_ffn, D, n_job, cfg_d=head_dim))
    n_layer = sum(1 for x in L if x.split(" ", 2)[:2][-1] in
                  ("LCTL", "LUJOB", "LJC", "LJOB", "KVWA") and
                  x.startswith("E "))
    stats = {"lines": len(L), "att_emissions": n_att,
             "fetches": 9, "proj_descs": n_proj, "layer_ops": n_layer,
             "rq_slots": H + 2, "jc_slots": fmt.JC_SLOTS}

    # ── .sub.ops: the FULL walkable fmt=1 stream (stage 5 — every step is
    # now concrete per §3b, so mask = EN_ALL; the name is kept for target
    # continuity). GROUP/REC feed the per-engine caches, HEADSQ/HEAD the
    # hd_* interlock, ENGSEL the store-phase engine expectation; every
    # expectation is an "E" line in walker2 emission order (attention DESCs
    # desc9-converted; the shared steps_head/steps_tail emit 9-field forms).
    S = [f"CASE {name} D={head_dim} T={T} H={H} HKV={H_kv} DM={D} DF={d_ffn}"]
    S += [f"DW {i:02d} {words[i]:08x}" for i in range(fmt.DESC_WORDS)]
    for g in range(H_kv):
        hd0 = fx.heads[g * group]
        S += [f"GROUP {g}"]
        for r in range(2 * T):
            row = hd0.K_f16[r] if r < T else hd0.V_f16[r - T]
            S += [f"REC {r:04x} " + " ".join(f"{int(v):04x}" for v in row)]
    for h in range(H):
        S += [f"HEADSQ {h:x} {int(fx.heads[h].s_q):04x}"]
    for h in range(H):
        S += [f"HEAD {h:x} {h // group:x}"]
    S += steps_head()
    for h, hd_res in enumerate(fx.heads):
        att = score_ops(head_dim, T, hd_res.s_q, hd_res.s_k, hd_res.s_v) \
            + pv_ops(head_dim, T, *hd_res.rq)
        S += [desc9(x) if x.startswith("E DESC") else x for x in att]
    S += steps_tail()
    S += ["END"]

    stats["sub_emissions"] = sum(1 for x in S if x.startswith("E "))
    # D-029 erratum: the spec may not bless a descriptor the tile refuses.
    # Swept over the WALKABLE stream (.sub.ops), which carries every DESC in
    # walker-emission order — attention (m=1, n<=8) and projection alike.
    stats["mxe_legal"] = check_desc_legality(S, name)
    n_desc_ops = sum(1 for x in L if x.startswith("E DESC"))
    assert stats["mxe_legal"] == n_desc_ops and stats["mxe_legal"] >= n_proj, \
        (f"{name}: DESC legality sweep covered {stats['mxe_legal']} of "
         f"{n_desc_ops} emissions ({n_proj} projections)")
    # SECOND ERRATUM: the spec may not bless a LAYER_JOB the consuming unit
    # would mis-read. Same sweep, same stream, the LAYER lane.
    stats["lu_legal"] = check_lujob_legality(S, name)
    n_lu_ops = sum(1 for x in L if x.startswith("E LUJOB"))
    assert stats["lu_legal"] == n_lu_ops and stats["lu_legal"] >= 4, \
        (f"{name}: LAYER_JOB legality sweep covered {stats['lu_legal']} of "
         f"{n_lu_ops} pushes")
    # THIRD ERRATUM: the spec may not bless a DESC whose act family no legal
    # stage buffer can hold. Same sweep, same stream, the k lane at THIS
    # case's row width.
    stats["sb_legal"] = check_desc_stageable(S, name, head_dim)
    assert stats["sb_legal"] == stats["mxe_legal"], \
        (f"{name}: stageability sweep covered {stats['sb_legal']} of "
         f"{stats['mxe_legal']} DESCs")
    return "\n".join(L) + "\n", "\n".join(S) + "\n", stats


def qstage_ops(D, qc_word, H):
    """E-3b: the walker-era q-staging emission template — per head, the
    HOST's own staging choreography (gen_layer_ops.build_rope_stage +
    gen_l3_vectors.inject_jobs order, verbatim): feeder job, MODE_F16
    squant job, the ONE per-step composite repeated D times (the per-tensor
    weight-scale model, seq_walker_pkg W2_QC note), the serializer job,
    D/8 x {loader-row act EMIT, k2 WS injection descriptor}, then the
    act-bank-1 LOAD at row h (the row W_S_AJEM's q_row_sel re-emits)."""
    BPR = D // 8
    L = []
    for h in range(H):
        L += ["E FJOB 1", f"E QJOB 0 {D:x}"]
        L += [f"E QS {qc_word:08x}"] * D
        L += [f"E SERJ {BPR:x} 8"]
        for _ in range(BPR):
            L += ["E AJ 1 0 0 1 1 0", "E DESC 01 1 2 8 0 0 0 0 0"]
        L += [f"E AJ 0 1 0 1 {BPR:x} {h:x}"]
    return L


def build_qstage_case(name, head_dim, T, seed, nfeed=False):
    """E-3b: the walker-staged-q case — mask {ROPE, STOREKV, QSTAGE, SCORE,
    PV} (+ NFEED + NSRC on the 128-wide variant, whose row family matches
    the spec-side FEED_DM constant), H=1 (the TB's Q_ROWS=1 walker instance;
    deeper staging is a tile parameter, refused above it — see refuse2).
    The stream is the E-4 walk shape: rope arm, store addressing,
    WALKER-STAGED q, per-head attention (and the in-tile norm feed).
    E-4b: a QSTAGE+NFEED image MUST name EN_NSRC (the S2_CHECK fence — no
    host-held l_nsrc serves codes->act and codes->norm in one kick), and
    the walker then OWNS the level: the NFEED level word carries nsrc=1 at
    bit 19 alongside the code-4 flip — the ONE-KICK chain this whole lane
    was after. Golden-gated exactly as build_case: fx = decoder_layer_fx
    supplies every tensor-dependent expectation (record rows, s_q/s_k/s_v,
    RQ)."""
    H, H_kv, d_ffn = 1, 1, 256
    D = H * head_dim
    rng = np.random.default_rng(seed)
    w = make_weights(rng, H, head_dim, H_kv, d_ffn, False)
    Xc = rng.normal(0, 1, (T, D))
    X = np.vstack([Xc, Xc[-1]])
    fx = tf.decoder_layer_fx(X, w, "CQ-8", G=128, q_pos=T - 1)

    qc = fmt.grade_f32(2.0 ** -6)
    # E-4b: NFEED in a QSTAGE walk requires NSRC (the fence), and NSRC is
    # what makes the walker own l_nsrc for the walk.
    mask = ((1 << fmt.EN_ROPE) | (1 << fmt.EN_STOREKV)
            | (1 << fmt.EN_QSTAGE) | (1 << fmt.EN_SCORE)
            | (1 << fmt.EN_PV)
            | (((1 << fmt.EN_NFEED) | (1 << fmt.EN_NSRC)) if nfeed else 0))
    words = [0] * fmt.DESC_WORDS
    words[fmt.W_GEOM0] = fmt.pack_geom0(head_dim)
    words[fmt.W_MODEL0] = fmt.pack_model0(D, d_ffn)
    words[fmt.W_MODEL1] = fmt.pack_model1(H, H_kv, 1)
    words[fmt.W_MASK] = fmt.pack_mask(mask)
    words[fmt.W_STEP] = fmt.pack_step(T, T - 1)
    for h, hd_res in enumerate(fx.heads):
        words[fmt.W_RQ0 + h] = fmt.pack_rq(*hd_res.rq)
    words[fmt.W_QC] = qc
    assert fmt.check2(words[0], words[1], words[2], words[3],
                      words[fmt.W_STEP], cfg_d=head_dim) == fmt.ERR_NONE, \
        f"{name}: the QSTAGE image must be walker-legal"

    # the level-word ladder this walk produces (every word composed by the
    # PACKAGE-mirror function; change events are what the TB scoreboards)
    lv_rope = fmt.lctl(rope_en=1, rope_pos=T - 1)
    lv_qs = fmt.lctl(rope_en=1, rope_bank=1, rope_pos=T - 1, fsrc_ext=3)
    lv_qsx = fmt.lctl(rope_en=1, rope_bank=1, rope_pos=T - 1, fsrc_ext=0)
    # E-4b: the walked NFEED level flips fsrc_ext -> 4 AND nsrc -> 1 on the
    # SAME edge (seq_layer_walker2 S2_NFL), so it is ONE change event whose
    # word carries bit 19 — the owned-l_nsrc arm the tile mux copies.
    lv_nf = fmt.lctl(rope_en=1, rope_bank=1, rope_pos=T - 1, fsrc_ext=4,
                     nsrc=1)
    assert lv_qs != lv_rope and lv_qsx != lv_qs and lv_nf != lv_qsx, \
        f"{name}: every QSTAGE level word must be a change event"
    assert lv_nf == (lv_qsx | 0x80 | (1 << 19)), \
        f"{name}: the NFEED word must move fsrc_ext and nsrc ALONE"

    S = [f"CASE {name} D={head_dim} T={T} H={H} HKV={H_kv} DM={D} "
         f"DF={d_ffn}"]
    S += [f"DW {i:02d} {words[i]:08x}" for i in range(fmt.DESC_WORDS)]
    for g in range(H_kv):
        hd0 = fx.heads[g]
        S += [f"GROUP {g}"]
        for r in range(2 * T):
            row = hd0.K_f16[r] if r < T else hd0.V_f16[r - T]
            S += [f"REC {r:04x} " + " ".join(f"{int(v):04x}" for v in row)]
    for h in range(H):
        S += [f"HEADSQ {h:x} {int(fx.heads[h].s_q):04x}"]
    for h in range(H):
        S += [f"HEAD {h:x} 0"]
    S += [f"E LCTL {lv_rope:04x}"]
    for g in range(H_kv):
        S += [f"E KVWA {g:x} {T - 1:x}", f"E KVWA {g:x} {2 * T - 1:x}"]
    S += [f"E LCTL {lv_qs:04x}"]
    S += qstage_ops(head_dim, qc, H)
    S += [f"E LCTL {lv_qsx:04x}"]
    for h, hd_res in enumerate(fx.heads):
        att = score_ops(head_dim, T, hd_res.s_q, hd_res.s_k, hd_res.s_v) \
            + pv_ops(head_dim, T, *hd_res.rq)
        S += [desc9(x) if x.startswith("E DESC") else x for x in att]
    if nfeed:
        S += [f"E LCTL {lv_nf:04x}", f"E FJOB {H:x}",
              f"E LUJOB {fmt.LU_NORM} {D:x}"]
    S += ["END"]
    stats = {"sub_emissions": sum(1 for x in S if x.startswith("E "))}
    # the blind-spot sweeps, same as build_case (the LAYER sweep only
    # where the spec-side FEED_DM constant matches the row family)
    stats["mxe_legal"] = check_desc_legality(S, name)
    stats["sb_legal"] = check_desc_stageable(S, name, head_dim)
    if nfeed:
        stats["lu_legal"] = check_lujob_legality(S, name)
    return "\n".join(S) + "\n", stats


def refuse2_vectors() -> str:
    """Directed walker2 refusals (all must raise WALK_ERR_DESC, CFG_D=64):
    (a) en_ffn with d_ffn=0 — the degenerate-FFN fence (a k=0 GEMM
    descriptor is illegal; refuse, never emit); (b) n_kv_heads > N_ENG(4)
    under kv_map=01 — the R3 build-envelope rule; (c) D-029 erratum —
    t_rows(65) > M_TILE_MAX(64) with QKV enabled: the K/V projection jobs
    carry m = t_rows, so walking this descriptor would hand mxe_ctrl an
    m_dim it refuses. WALK_T_MAX is 128, twice the implemented M, so this
    fence is reachable by a pkg-LEGAL descriptor — which is why it must be
    a fence and not a comment. (The stage-2 PENDING-Q1 refusal case retired
    at stage 5: every step is walkable now.)"""
    sub_mask = fmt.EN_ALL
    L = ["# refusal: en_ffn with d_ffn=0 (degenerate FFN) -> DESC err"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(128, 0)
    w[fmt.W_MODEL1] = fmt.pack_model1(2, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask(sub_mask)
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    L += ["# refusal: n_kv_heads=8 > N_ENG=4 under kv_map=01 -> DESC err"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(512, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(8, 8, 1)
    w[fmt.W_MASK] = fmt.pack_mask(sub_mask)
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (b) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    L += ["# refusal (D-029 erratum): t_rows=65 > M_TILE_MAX=64 with QKV en"
          " -> DESC err (K/V projection m_dim would be tile-illegal)"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(128, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(2, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask(sub_mask)
    w[fmt.W_STEP] = fmt.pack_step(fmt.MXE_M_MAX + 1, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (c) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    # (d) E-3: the norm-feed envelope. The C-1 feeder takes at most
    # FEED_ROWS_MAX rows per job (16 in the TB's default instantiation) and
    # the NFEED step frames d_model as H rows, so H=17 asks the feeder for a
    # job it would answer with job_error — AFTER the walker had taken the
    # push handshake and committed to waiting on nf_busy, i.e. a silent
    # wedge. The walk is refused at S2_CHECK instead. Reachable by a
    # pkg-LEGAL descriptor (H_MAX is 30), which is why it must be a fence.
    L += ["# refusal (E-3): en_nfeed with H=17 > the feeder's ROWS_MAX(16)"
          " -> DESC err (the C-1 feeder job would be refused mid-step)"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(17 * 64, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(17, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask(1 << fmt.EN_NFEED)
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (d) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    # (e) E-3b: the QSTAGE staging-depth fence. Act bank 1 holds Q_ROWS
    # staged q rows (the tile's QSTAGE_H_MAX; the TB instantiates the
    # default 1) — a deeper head count would stage rows the attention walk
    # cannot select. pkg-LEGAL (H_MAX=30), so the FENCE is what fires.
    L += ["# refusal (E-3b): en_qstage with H=2 > Q_ROWS(1) -> DESC err"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(2 * 64, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(2, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask(1 << fmt.EN_QSTAGE)
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (e) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    # (f) E-3b: QSTAGE and QKV both enabled would produce q twice (the
    # walked projection AND the staging step) — refused, never walked.
    L += ["# refusal (E-3b): en_qstage with en_qkv -> DESC err (double q)"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(64, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(1, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask((1 << fmt.EN_QSTAGE) | (1 << fmt.EN_QKV))
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (f) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    # (g) E-4b: NSRC without NFEED — the walker would own the norm x-mux
    # for a walk with nothing to feed it, pinning l_nsrc 0 over whatever
    # the host had armed: a silent semantic change under a mis-built
    # image. Refused. pkg-LEGAL (bit 13 is EN_NSRC now, not resv), so the
    # FENCE is what fires.
    L += ["# refusal (E-4b): en_nsrc without en_nfeed -> DESC err"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(64, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(1, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask((1 << fmt.EN_NSRC) | (1 << fmt.EN_ROPE))
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (g) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    # (h) E-4b: QSTAGE+NFEED WITHOUT NSRC — the measured E-4 wedge shape:
    # l_nsrc is host-held for this image, and no single held value serves
    # both codes->act (staging) and codes->norm (feed); the walk would
    # wait forever on nf_busy at the staging drain. Refused, never walked.
    # pkg-LEGAL and QSTAGE/NFEED-fence-clean (H=1, FEED_DM==CFG_D), so
    # THIS fence is what fires.
    L += ["# refusal (E-4b): en_qstage with en_nfeed but no en_nsrc"
          " -> DESC err (the measured one-kick wedge shape)"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(64, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(1, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask((1 << fmt.EN_QSTAGE) | (1 << fmt.EN_NFEED))
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (h) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]
    # (i) THIRD ERRATUM: the FPROJ act-family STAGE-CAPACITY fence. The
    # fuel-fed projection's activation family is d_model/CFG_D PAT_ROW rows
    # in the act stage buffer, and the walker is handed the instantiated
    # rows-per-bank bound (STAGE_ROWS; the TB instantiates the default 16,
    # the flying builds 31). H=17 at head_dim=64 makes fp_rows=17 > 16 —
    # an act family this instance cannot hold, refused at S2_CHECK before
    # any state changes (the same k/CFG_D <= R_MAX arithmetic that made the
    # 0.5B Wd chunk D-aware, fenced at the per-BUILD bound). pkg-LEGAL
    # (H_MAX=30, mask bits all defined), so a FENCE is what fires; the
    # FPROJ fence family shares one refusal code with its fuel-armed clause
    # (wf_ready is randomized in the TB), and every clause answers DESC.
    L += ["# refusal (third erratum): en_fproj with H=17 -> fp_rows 17 >"
          " STAGE_ROWS(16) -> DESC err (act family exceeds the stage bank)"]
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(64)
    w[fmt.W_MODEL0] = fmt.pack_model0(17 * 64, 256)
    w[fmt.W_MODEL1] = fmt.pack_model1(17, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask((1 << fmt.EN_FPROJ) | (1 << fmt.EN_QKV))
    w[fmt.W_STEP] = fmt.pack_step(8, 7)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP], cfg_d=64) \
        == fmt.ERR_NONE, "case (i) must be pkg-legal so the FENCE is what fires"
    L += [f"DW {i:02d} {w[i]:08x}" for i in range(fmt.DESC_WORDS)]
    L += ["EXPECT_REFUSE 2", "END"]

    # (j)..(o) E-7/E-7b cross-bit fences (all pkg-LEGAL so S2_CHECK is what
    # fires; the FPROJ family shares one DESC code with its randomized
    # wf_ready clause, per the case-(i) note).
    def _case(tag, why, mask, d_model=64, d_ffn=256, H=1, Hkv=1):
        nonlocal_l = [f"# refusal ({tag}): {why}"]
        ww = [0] * fmt.DESC_WORDS
        ww[fmt.W_GEOM0] = fmt.pack_geom0(64)
        ww[fmt.W_MODEL0] = fmt.pack_model0(d_model, d_ffn)
        ww[fmt.W_MODEL1] = fmt.pack_model1(H, Hkv, 1)
        ww[fmt.W_MASK] = fmt.pack_mask(mask)
        ww[fmt.W_STEP] = fmt.pack_step(8, 7)
        assert fmt.check2(ww[0], ww[1], ww[2], ww[3], ww[fmt.W_STEP],
                          cfg_d=64) == fmt.ERR_NONE, \
            f"case ({tag}) must be pkg-legal so the FENCE is what fires"
        nonlocal_l += [f"DW {i:02d} {ww[i]:08x}"
                       for i in range(fmt.DESC_WORDS)]
        nonlocal_l += ["EXPECT_REFUSE 2", "END"]
        return nonlocal_l

    L += _case("E-7 j", "en_fgam without en_fproj -> the gamma fetch would "
               "park at S2_FETCH (fuel unarmed)",
               (1 << fmt.EN_FGAM) | (1 << fmt.EN_NORM2))
    L += _case("E-7 k", "en_fgam under fproj without a NORM step -> a gamma "
               "window with no consumer",
               (1 << fmt.EN_FGAM) | (1 << fmt.EN_FPROJ) | (1 << fmt.EN_QKV))
    L += _case("E-7 l", "NORM2 under FPROJ without FGAM -> the E-6 fuel-"
               "poisoning refusal, kept verbatim",
               (1 << fmt.EN_NORM2) | (1 << fmt.EN_FPROJ) | (1 << fmt.EN_QKV))
    L += _case("E-7b m", "en_down without en_fproj -> the Wd fetch park "
               "wedge",
               (1 << fmt.EN_DOWN) | (1 << fmt.EN_RES2))
    L += _case("E-7b n", "en_down without en_res2 under fproj -> the "
               "starvation pair rule (the OPROJ<->RES1 argument)",
               (1 << fmt.EN_DOWN) | (1 << fmt.EN_FPROJ))
    L += _case("E-7b o", "walked DOWN at the 0.5B d_ffn=4864 > k_job(64)="
               "1984 -> one-k-split fence (76-row act family vs the 31-row "
               "stage bank; no in-tile re-staging source)",
               (1 << fmt.EN_DOWN) | (1 << fmt.EN_RES2) | (1 << fmt.EN_FPROJ),
               d_ffn=4864)
    L += _case("E-7 p", "QKV+attention now COMPOSE but the stacked act "
               "families must fit the stage bank: H=9 -> 9 q rows + 9 QKV "
               "rows = 18 > STAGE_ROWS(16, the TB instantiation)",
               (1 << fmt.EN_QKV) | (1 << fmt.EN_SCORE) | (1 << fmt.EN_PV)
               | (1 << fmt.EN_FPROJ),
               d_model=9 * 64, H=9)
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = HERE / "build" / "layer"
    # test-only knobs (Makefile `mutants4`): override the MXE n-split bound
    # and redirect the output, so the legality gate can be PROVEN to fire.
    while argv:
        a = argv.pop(0)
        if a == "--mutate-n-mxe":
            fmt.N_MXE = int(argv.pop(0))
            print(f"SPEC MUTANT: seq_walker_fmt.N_MXE overridden -> "
                  f"{fmt.N_MXE} (the pre-erratum field-width bound)")
        elif a == "--mutate-k-rows":
            fmt.K_ROWS = int(argv.pop(0))
            print(f"SPEC MUTANT: seq_walker_fmt.K_ROWS overridden -> "
                  f"{fmt.K_ROWS} (k_job(64) = {fmt.k_job(64)} — the "
                  f"pre-third-erratum D-blind chunk)")
        elif a == "--mutate-lu-swiglu":
            fmt.LU_CHUNK[fmt.LU_SWIGLU] = int(argv.pop(0))
            print(f"SPEC MUTANT: the swiglu LAYER_JOB chunk bound overridden "
                  f"-> {fmt.LU_CHUNK[fmt.LU_SWIGLU]} (the pre-erratum "
                  f"field-width bound)")
        elif a == "--out":
            out = Path(argv.pop(0))
        else:
            print(f"unknown argument {a!r}")
            return 2
    out.mkdir(parents=True, exist_ok=True)
    tot = {"att_emissions": 0, "fetches": 0, "proj_descs": 0,
           "layer_ops": 0, "sub_emissions": 0, "mxe_legal": 0, "lu_legal": 0,
           "sb_legal": 0}
    for case in CASES:
        # the optional 9th tuple field is the E-3 nfeed flag, passed by
        # KEYWORD so it cannot slide into build_case's `inject` hook
        args = case[:8]
        kw = {"nfeed": True} if len(case) > 8 and case[8] else {}
        text, sub, st = build_case(*args, **kw)
        # determinism gate: a fresh RNG from the same seed must reproduce
        text2, sub2, _ = build_case(*args, **kw)
        assert text == text2 and sub == sub2, f"{case[0]}: NON-DETERMINISTIC"
        (out / f"{case[0]}.ops").write_text(text)
        (out / f"{case[0]}.sub.ops").write_text(sub)
        for k in tot:
            tot[k] += st[k]
        print(f"wrote {out.name}/{case[0]}.ops: {st['lines']} lines, "
              f"{st['att_emissions']} attention emissions, "
              f"{st['fetches']} fetches, {st['proj_descs']} proj DESCs, "
              f"{st['layer_ops']} layer ops, {st['rq_slots']} rq + "
              f"{st['jc_slots']} jc slots "
              f"(+ .sub.ops: {st['sub_emissions']} expected emissions, "
              f"{st['mxe_legal']} DESCs MXE-legal)")
    # E-3b: the walker-staged-q cases (E-4 walk shape; 128 adds NFEED)
    for nm, hd, sd, nf in [("qstage_h1_hd64", 64, 0x1B0A21, False),
                           ("qstage_h1_hd128", 128, 0x1B0A22, True)]:
        sub, st = build_qstage_case(nm, hd, 8, sd, nfeed=nf)
        sub2, _ = build_qstage_case(nm, hd, 8, sd, nfeed=nf)
        assert sub == sub2, f"{nm}: NON-DETERMINISTIC"
        (out / f"{nm}.sub.ops").write_text(sub)
        print(f"wrote {out.name}/{nm}.sub.ops: {st['sub_emissions']} "
              f"expected emissions, {st['mxe_legal']} DESCs MXE-legal"
              + (f", {st['lu_legal']} LAYER_JOBs legal" if nf else ""))
    rv = refuse2_vectors()
    n_ref = rv.count("EXPECT_REFUSE")
    (out / "refuse2.ops").write_text(rv)
    print(f"wrote {out.name}/refuse2.ops: {n_ref} directed walker2 refusals")

    print()
    print("NOTE (R3, escalated in the stage-1 report): H/H_kv = 7 at the 7B")
    print("  geometry is NOT a power of two, so §9.1 R3's formula")
    print("  'h >> log2(H/H_kv)' is inapplicable there; this spec uses the")
    print("  golden mapping h // (H//H_kv) (transformer.py GQA slicing).")
    print("  Engine count: per-KV-head banking needs H_kv CQ-8 engine")
    print("  instances (2T = 256 = DEPTH exactly at T=128, zero headroom);")
    print("  today's apex_kvq_bank has THREE per-TIER engines (D-024) — the")
    print("  per-head instantiation is IB-LAYER build-out.")
    print()
    print(f"LAYER TRACE SELFTEST: PASS ({len(CASES)} cases; epilogue "
          f"replicas bit-exact; store-scale identity held on every head; "
          f"emitters == §6 formulas; {tot['att_emissions']} attention "
          f"emissions, {tot['fetches']} fetches, {tot['proj_descs']} proj "
          f"DESCs total; {tot['mxe_legal']} DESCs MXE-LEGAL by the tile's "
          f"own rule, n<={fmt.MXE_N_MAX} m<={fmt.MXE_M_MAX} "
          f"k<={fmt.MXE_K_MAX}; {tot['sb_legal']} act families STAGEABLE by "
          f"the stage buffer's own rule, ceil(k/D)<={fmt.SB_R_MAX} rows; "
          f"{tot['lu_legal']} LAYER_JOBs LEGAL by the "
          f"consuming unit's own rule, swiglu<={fmt.SWG_COLS_MAX} "
          f"deq<={fmt.DEQ_COLS_MAX} resid<={fmt.RES_COLS_MAX})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
