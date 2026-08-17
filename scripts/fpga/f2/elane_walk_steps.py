#!/usr/bin/env python3
# elane_walk_steps.py — E-lane step-matrix probes (E2E_TOY_LANE.md item 1 of
# the E-3b/E-4 lane): WHICH of the fmt=1 layer walker's steps can run WALKED
# against the REAL datapath today, ENUMERATED BY EXECUTION on the b64 sim
# twin (IMG_05B: CFG_D=CFG_DM=64, GQA_NENG=2, QSTAGE_H_MAX=14, DM_MAX=896),
# at the E-lane toy fence: head_dim == D_model == CFG_D = 64, single head.
#
# ══ WHAT A PROBE IS ════════════════════════════════════════════════════════
# One fmt=1 descriptor per layer step (or step class), mask enabling ONLY
# that step, loaded and kicked on the REAL tile (the verilated cl_apex twin,
# not the stubbed emission TB). Each probe states its expected outcome as
# CHECKED reads/polls, so the run is the measurement:
#
#   GREEN     the walk retires clean and the step's effect is observed on
#             the real datapath (graded vs golden where there is data).
#   EXPECT-RED the walk must NOT complete — the probe's checked reads state
#             the wedge/refusal precisely (rc != 0 IS the receipt, and the
#             receipt reads before it localize the wedge).
#
# ══ THE PRE-MEASURED RTL FACTS THE EXPECTATIONS ENCODE (file:line) ═════════
#   F1  Every K_FETCH state (PC_G1F/WQF/WKF/WVF/WOF/WGF/WUF/WDF) presents
#       wf_valid to apex_fuel_ctl's walker producer, which asserts ready
#       ONLY in fuel mode (apex_fuel_ctl.sv:262-class 'records accepted only
#       in fuel mode (src=1)'). With FUEL_CTRL.src=0 (reset default; the
#       only mode any DCP image flies today) the walker parks in S2_FETCH
#       forever. There is NO abort path out of it: seq_layer_walker2.sv:712
#       takes abort only in {S2_CHECK, S2_STEP, S2_DONE}.
#   F2  In walk mode the route levels are the WALKER's (apex_top.sv:553-559)
#       and seq_layer_walker.sv:302-308 pins them: fsrc=fdst=asrc=wsrc=qsrc=
#       kvu=1, rdst = pv_route ? 0 : 2. Consequences measured by the probes:
#       rt_wgt_src=1 means a walked GEMM can NEVER consume the external xw
#       stream (fuel OR host), and rt_res_dst is never 1, so no walked MXE
#       result can ever reach the serializer -> deq/swiglu/residual units.
#   F3  The fmt=1 sequencer pushes NO aj/wj stage-buffer jobs for its
#       projection steps (S2_PJOB emits ds descriptors only,
#       seq_layer_walker2.sv:814-827), so a walked projection's operands
#       are never staged/emitted: the MXE starves in ingest.
#   F4  The LAYER-unit pushes (S2_JCA/JCB/S2_LU) land on the REAL units
#       (apex_top.sv:1164-1174 walker ingress), which then wait for their
#       input streams (serializer / deq) that F2+F3 make unreachable.
#   F5  PC_NFEED (E-3a) is the one step whose data path IS closed in-tile;
#       PC_NORM1/PC_NORM2 (K_NORM) complete on dn_rms via the lj tie
#       (apex_top.sv:864-868) — the norm is stream-triggered, so its x can
#       come from the walked NFEED and gamma from the (not-held-off) xg
#       port, but its OWN pc is fetch-gated (PC_G2F precedes PC_NORM2).
#
# NEW FILE ON PURPOSE (the elane rule): gen_layer_ops / gemm_job / walk_job
# are concurrently owned or frozen — this emitter IMPORTS the frozen
# vocabulary via tile_geom.at_d(64) and adds only the probe programs.
# Zero baked output expectations (audit_program enforced) on produced data;
# refusal/wedge probes use checked reads of STATUS bits (the probe idiom).
#
# CLI:
#   python3 scripts/fpga/f2/elane_walk_steps.py --selftest
#   python3 scripts/fpga/f2/elane_walk_steps.py --build [--out DIR]
#   python3 scripts/fpga/f2/elane_walk_steps.py --smoke [--binary PATH]
#       [--tile-div N] [--out DIR] [--only NAME[,NAME]]
#
# The matrix (docs/results form) is printed at the end of --smoke and
# written to <out>/step_matrix.json.

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "golden"), str(REPO / "verif" / "top" / "l3"),
           str(REPO / "verif" / "seq_walker"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_layer_ops as glo                                    # noqa: E402
import gen_l3_vectors as g3                                    # noqa: E402
import cap_decode as cd                                        # noqa: E402
import seq_walker_fmt as fmt                                   # noqa: E402
import tile_geom as tg                                         # noqa: E402
from apex_golden import attention as at                        # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits    # noqa: E402

from gen_layer_ops import (                                    # noqa: E402
    LayerScript, BANK_RESID, LU_RESID, LU_DEQ,
    LST_LDQ, LST_SWG, LST_RES, LST_ROPE, LST_ERR, L_CTRL,
    _translate, _emit, _prologue, _drain_and_check, inject_frame,
    identity_readback, grade_resid, eprint,
)
from elane_norm_feed import (                                  # noqa: E402
    LCTL_NSRC, LCTL_FSRC_EXT2, LST_NFEED, nfjob, stalled_at_poll,
)

# ── WALK CSR window (rtl/seq/seq_walker_pkg.sv:74-83) ──────────────────────
W_CTRL, W_DPTR, W_DDATA, W_STATUS = 0x5C, 0x60, 0x64, 0x68
W_ST_BUSY = 0x0001
W_ST_ESTK = 0x0100
W_ST_ECODE = 0x0E00
W_ST_FMTSUP = 0xF000
FMT_SUP_LAYER = 0x3

D_PROBE = 64                     # the E-lane toy fence on the b64 twin
LST_BUSY5 = LST_LDQ | LST_SWG | LST_RES | 0x8 | 0x10 | LST_NFEED
DEFAULT_OUT = REPO / "build" / "f2_elane_steps"


# ═══════════════ golden toy references at width d ══════════════════════════

def _f32_val(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def toy_step_d(d: int, seed: int = 20260803) -> dict:
    """elane_norm_feed.toy_step at row width d — same golden functions,
    applied the same way (transformer.py's r1 -> NORM2 entry), no baked
    values. Kept a sibling because the frozen toy_step captures D_TILE=128
    at import time and this lane's fence is the b64 twin's CFG_D=64."""
    rng = np.random.default_rng(seed)
    x_bits = f64_to_f16_bits(rng.normal(0.0, 1.5, d))
    x_vals = f16_bits_to_f64(np.asarray(x_bits, dtype=np.uint16))
    o8 = rng.integers(-127, 128, d, dtype=np.int64)
    comp_bits = fmt.grade_f32(2.0 ** -6)
    b = o8.astype(np.float64) * _f32_val(comp_bits)
    r1_bits = f64_to_f16_bits(x_vals + b)
    r1_vals = f16_bits_to_f64(np.asarray(r1_bits, dtype=np.uint16))
    r1_codes, r1_scales = at.quant_rows_i8(r1_vals[None, :])
    g2 = rng.integers(-12288, 12289, d, dtype=np.int64)
    h2, _r, _n = at.rmsnorm_fx([int(c) for c in r1_codes[0]],
                               [int(v) for v in g2])
    h2_vals = np.asarray(h2, dtype=np.float64) / 256.0
    h2_codes, h2_scales = at.quant_rows_i8(h2_vals[None, :])
    return dict(
        seed=int(seed), d=int(d),
        x_bits=np.asarray(x_bits, dtype=np.uint16),
        o8=o8, comp=int(comp_bits),
        r1_bits=np.asarray(r1_bits, dtype=np.uint16),
        r1_codes=np.asarray(r1_codes[0], dtype=np.int64),
        r1_scale=int(r1_scales[0]),
        gamma2=g2,
        h2_codes=np.asarray(h2_codes[0], dtype=np.int64),
        h2_scale=int(h2_scales[0]),
    )


# ═══════════════ fmt=1 probe descriptors (H=1 toy) ═════════════════════════

def probe_desc(d: int, mask: int, *, t_rows: int = 1, pos_m: int = 0,
               d_ffn: int = 0, jc: bool = False) -> list[int]:
    """A 64-word fmt=1 image at the toy fence: H=1, H_kv=1, head_dim ==
    d_model == d (the build's CFG_D), kv_map=01 (per-KV-head engines — the
    b64_05b twin is a GQA_NENG=2 build). Legality mirrored at emission.

    jc=True loads well-formed (positive-normal fp16-grade) composites into
    the four W_JC slots — MEASURED necessity: the walker rewrites the REAL
    JOBC file from these words before every unit push (S2_JCA/JCB), and
    apex_layer_deq / asu_swiglu REFUSE a zero composite at job accept
    (LAYER err_code 2) — see the p_jc0 probe, which keeps that refusal as
    its own measured receipt."""
    w = [0] * fmt.DESC_WORDS
    w[fmt.W_GEOM0] = fmt.pack_geom0(d)
    w[fmt.W_MODEL0] = fmt.pack_model0(d, d_ffn)
    w[fmt.W_MODEL1] = fmt.pack_model1(1, 1, 1)
    w[fmt.W_MASK] = fmt.pack_mask(mask)
    w[fmt.W_STEP] = fmt.pack_step(t_rows, pos_m)
    if jc:
        for slot in range(4):
            w[fmt.W_JC0 + slot] = fmt.grade_f32(2.0 ** -6)
    assert fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                      cfg_d=d) == fmt.ERR_NONE, f"illegal probe image {mask:#x}"
    return w


def load_descriptor(s: LayerScript, words: list[int]):
    s.csrw(W_DPTR, 0)
    for v in words:
        s.csrw(W_DDATA, v)


def _finish(s, x_note, out_dir, name, meta):
    x = _translate(s, note=x_note)
    man = _emit(x, out_dir, name, meta)
    tg.retarget_info_tier(man["path"], tg.IMG_05B)
    return man


# ═══════════════ the probes ════════════════════════════════════════════════

def build_nfeed(step, *, out_dir, name, nfeed=True) -> dict:
    """E-3a's walked NFEED chain at the b64 toy width (GREEN expected):
    host produces r1 via CSR path, the WALKER arms code 4 + feeder job +
    unit-3 job, r1 -> C-1 -> RMSNorm in-tile, host drains and grades vs
    golden. nfeed=False is the walk-off discriminator (EXPECT-RED: the
    route-arm poll can never satisfy)."""
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// STEP-MATRIX probe {name}: PC_NFEED on the b64 twin (d={d}).")
    _prologue(s)
    s.lptr(BANK_RESID, 0)
    s.ldata(step["x_bits"])
    s.lctl(ser_dst=1, resid_arm=1)
    s.route(rdst=1)
    s.pmode(True)
    s.lujob(LU_RESID, d)
    s.ljc(step["comp"])
    s.lujob(LU_DEQ, d)
    inject_frame(s, step["o8"])
    s.lpoll(LST_LDQ | LST_RES, 0x0)
    s.route(rdst=0, asrc=0)
    s.csrw(L_CTRL, LCTL_NSRC)
    s.csrr(W_STATUS, W_ST_FMTSUP, FMT_SUP_LAYER << 12)
    load_descriptor(s, probe_desc(
        d, (1 << fmt.EN_NFEED) if nfeed else (1 << fmt.EN_RES1)))
    s.csrw(W_CTRL, 0x3)
    s.csrp(L_CTRL, LCTL_FSRC_EXT2, LCTL_FSRC_EXT2)   # walker armed code 4
    s.lpoll(LST_NFEED, 0x0)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    s.efs(0, 0)
    s.csrw(L_CTRL, 0)
    s.aj(0, glo.ACT_BANK, 0, 1, s.BPR, 0)
    s.fjob(1)
    s.grow_n(step["gamma2"])
    s.efs(0, 0)
    s.route(rdst=0, wsrc=0, asrc=0)
    for base in range(0, d, glo.MXE_N):
        identity_readback(s, 0, base)
    s.lrptr(0)
    s.erd(d)
    s.pmode(False)
    _drain_and_check(s)
    return _finish(s, f"STEP {name}: walked NFEED @b64", out_dir, name, dict(
        stage="nfeed" if nfeed else "nfeed_off", d=d,
        expect="GREEN bit-exact" if nfeed else
               "EXPECT-RED: route-arm poll stalls (walker armed nothing)",
        seed=step["seed"]))


def build_rope(*, out_dir, name, pos_m=3) -> dict:
    """PC_ROPE (K_LVL) against the real level registers (GREEN expected):
    the walk retires clean and LAYER_CTRL reads back rope_en=1, bank=0
    (K table), pos=pos_m, ser_dst=0 — armed by the WALKER (in walk mode the
    l_*_q registers track the walker's nets, apex_top.sv:1206-1219)."""
    s = LayerScript(D_PROBE)
    s.emit(f"// STEP-MATRIX probe {name}: PC_ROPE arms the REAL level regs.")
    _prologue(s)
    s.lctl()                                     # host leaves levels at 0
    s.csrr(L_CTRL, 0x7FFF, 0)
    load_descriptor(s, probe_desc(D_PROBE, 1 << fmt.EN_ROPE, pos_m=pos_m))
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.emit("// the WALKER's arm, read on the real register while walk_en "
           "still holds the walker's nets on it")
    s.csrr(L_CTRL, 0x7FFF, fmt.lctl(rope_en=1, rope_bank=0, rope_pos=pos_m))
    s.csrw(W_CTRL, 0x0)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: walked ROPE arm", out_dir, name, dict(
        stage="rope_arm", d=D_PROBE, pos_m=pos_m,
        expect="GREEN: walk retires; LAYER_CTRL = walker's rope arm"))


def build_storekv(*, out_dir, name) -> dict:
    """PC_STORE (K_STORE) against the real KVQ bank (GREEN expected for the
    ADDRESSING): the walker polls engine STATUS idle and programs
    WRITE_ADDR K@T-1=0 then V@2T-1=1 over the real AXI-Lite, B-responses
    and all. The record DATA still arrives via the host squant path — that
    is measured by storing a record AFTER the walk at the walker-programmed
    address and reading it back would be the full loop; this probe's claim
    is the walked addressing transaction completing against the real bank."""
    s = LayerScript(D_PROBE)
    s.emit(f"// STEP-MATRIX probe {name}: PC_STORE — walked KVQ WRITE_ADDR "
           f"programming on the real bank (T=1, H_kv=1 -> engine 0, K then "
           f"V records).")
    _prologue(s)
    load_descriptor(s, probe_desc(D_PROBE, 1 << fmt.EN_STOREKV))
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    s.emit("// host mode again: the engine accepted two WRITE_ADDR "
           "transactions (bvalid seen twice by the walker or it would still "
           "be busy); engine still idle and error-free")
    s.kvp(g3.KV["STATUS"], 0x1, 0x1)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: walked STOREKV addr", out_dir, name,
                   dict(stage="storekv", d=D_PROBE,
                        expect="GREEN: walk retires clean; engine idle after "
                               "two real WRITE_ADDR transactions"))


def build_res1(*, out_dir, name) -> dict:
    """PC_URES1 (K_UJOB -> residual): CONTROL-GREEN / DATA-STARVED expected.
    The walker's unit-2 push lands on the real apex_residual (walker ingress
    apex_top.sv:1164-1174), the job is accepted, the walk RETIRES — and the
    unit then waits forever for the deq stream that no walked step can
    produce (facts F2-F4). The probe records both: walk clean, residual
    busy STUCK. No drain (the unit cannot go idle); per-file TILE_RST is
    the documented recovery."""
    s = LayerScript(D_PROBE)
    s.emit(f"// STEP-MATRIX probe {name}: PC_URES1 push accepted; unit "
           f"data-starved (measured).")
    _prologue(s)
    load_descriptor(s, probe_desc(D_PROBE, 1 << fmt.EN_RES1))
    s.csrw(W_CTRL, 0x3)
    s.csrp(W_STATUS, W_ST_BUSY, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY | W_ST_ESTK | W_ST_ECODE, 0x0)
    s.csrw(W_CTRL, 0x0)
    s.emit("// the receipt: residual BUSY (job accepted, awaiting the deq "
           "stream that walked steps cannot route — F2: rt_res_dst is "
           "never 1 in a walk), no LAYER error")
    s.lstat(LST_RES | LST_ERR, LST_RES)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: walked RES1 push", out_dir, name, dict(
        stage="res1", d=D_PROBE,
        expect="CONTROL-GREEN: walk retires, push accepted; residual busy "
               "stuck=1 (data-starved), no error"))


def build_oproj(*, out_dir, name) -> dict:
    """PC_LVLO + PC_UOPRJ + PC_WOF (EXPECT-RED at the fetch): the walker
    arms ser_dst=1 on the real level regs, writes the real JOBC file,
    pushes the real deq job (accepted; ldq busy) — then parks FOREVER in
    S2_FETCH because wf_ready needs fuel mode (fact F1). The receipts
    localize it: ser_dst armed, ldq busy, WALK busy with NO error sticky
    (a wedge, not a refusal). rc != 0 (final busy check) is the measurement."""
    s = LayerScript(D_PROBE)
    s.emit(f"// STEP-MATRIX probe {name}: OPROJ chain wedges at the Wo "
           f"FETCH (fuel mode off on every flyable image).")
    _prologue(s)
    load_descriptor(s, probe_desc(D_PROBE, 1 << fmt.EN_OPROJ, jc=True))
    s.csrw(W_CTRL, 0x3)
    s.emit("// receipts BEFORE the wedge point: the walker armed ser_dst=1 "
           "(PC_LVLO) and the deq job was accepted (PC_UOPRJ; ldq busy)")
    s.csrp(L_CTRL, 0xC, 0x4)                     # ser_dst == 1
    s.lpoll(LST_LDQ, LST_LDQ)
    s.emit("// the wedge: WALK busy, NO error sticky — S2_FETCH parked on "
           "wf_ready (apex_fuel_ctl accepts records only in fuel mode). "
           "This checked read EXPECTS busy to have fallen; it CANNOT — the "
           "resulting FAIL rc is the measured receipt.")
    s.csrr(W_STATUS, W_ST_ESTK, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY, 0x0)             # <- measured: FAILS (busy=1)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: OPROJ fetch wedge", out_dir, name, dict(
        stage="oproj", d=D_PROBE,
        expect="EXPECT-RED rc!=0: ser_dst armed + ldq busy receipts GREEN, "
               "then WALK busy stuck at S2_FETCH (no fuel), no err sticky"))


def build_qkv(*, out_dir, name) -> dict:
    """PC_WQF (EXPECT-RED): the FIRST enabled pc of a QKV-only walk is the
    Wq fetch — S2_FETCH parks immediately (fact F1). Even with fuel mode
    on, facts F2+F3 starve the GEMM (weights unreachable at rt_wgt_src=1;
    no act EMIT jobs) — that variant needs the DDR twin and is recorded as
    analysis in the matrix; THIS probe measures the fetch park."""
    s = LayerScript(D_PROBE)
    s.emit(f"// STEP-MATRIX probe {name}: QKV walk parks at the Wq fetch.")
    _prologue(s)
    load_descriptor(s, probe_desc(D_PROBE, 1 << fmt.EN_QKV))
    s.csrw(W_CTRL, 0x3)
    s.csrr(W_STATUS, W_ST_ESTK, 0x0)             # wedge, not refusal
    s.csrr(W_STATUS, W_ST_BUSY, 0x0)             # <- measured: FAILS (busy=1)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: QKV fetch wedge", out_dir, name, dict(
        stage="qkv", d=D_PROBE,
        expect="EXPECT-RED rc!=0: WALK busy stuck at S2_FETCH (Wq), no err"))


def build_ffn(*, out_dir, name) -> dict:
    """PC_LVLG + PC_USWI + PC_WGF (EXPECT-RED at the gate fetch): the
    swiglu JOBC pair and unit job land on the real asu_swiglu (accepted;
    swg busy awaiting its gate frame), then the walker parks at the Wg
    fetch. d_ffn=64 == the unit's COLS_MAX so the job is one chunk."""
    s = LayerScript(D_PROBE)
    s.emit(f"// STEP-MATRIX probe {name}: FFN chain wedges at the Wg FETCH; "
           f"swiglu push accepted first (receipt).")
    _prologue(s)
    load_descriptor(s, probe_desc(D_PROBE, 1 << fmt.EN_FFN, d_ffn=64,
                                  jc=True))
    s.csrw(W_CTRL, 0x3)
    s.emit("// receipts: ser_dst=2 (PC_LVLG) then the swiglu job accepted")
    s.csrp(L_CTRL, 0xC, 0x8)                     # ser_dst == 2
    s.lpoll(LST_SWG, LST_SWG)
    s.csrr(W_STATUS, W_ST_ESTK, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY, 0x0)             # <- measured: FAILS (busy=1)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: FFN fetch wedge", out_dir, name, dict(
        stage="ffn", d=D_PROBE,
        expect="EXPECT-RED rc!=0: ser_dst=2 + swg busy receipts GREEN, then "
               "WALK busy stuck at S2_FETCH (Wg)"))


def build_jc0(*, out_dir, name) -> dict:
    """MEASURED FINDING kept as its own probe: the walker's S2_JCA write
    lands on the REAL JOBC file, and a descriptor whose JC slot is 0 (not a
    positive-normal fp16-grade composite) has its deq job REFUSED by the
    real apex_layer_deq at accept — LAYER sticky + err_code 2 (GRADE/JOB
    class), push consumed, walker moves on and parks at the Wo fetch.
    First measured 2026-08-04 on the b64 twin (this probe's first run)."""
    s = LayerScript(D_PROBE)
    s.emit(f"// STEP-MATRIX probe {name}: zero JOBC composite from the "
           f"descriptor -> real-unit refusal (code 2), walker unwedged "
           f"past the push, parked at the fetch.")
    _prologue(s)
    load_descriptor(s, probe_desc(D_PROBE, 1 << fmt.EN_OPROJ))  # JC slots 0
    s.csrw(W_CTRL, 0x3)
    s.emit("// receipts: the REAL deq unit refused the walker's job (sticky "
           "+ code 2), did NOT stay busy, and the walker proceeded to the "
           "S2_FETCH park (busy, no walker sticky)")
    s.lpoll(LST_ERR, LST_ERR)
    s.lstat(0x1F00 | LST_LDQ, (2 << 9) | LST_ERR)
    s.csrr(W_STATUS, W_ST_ESTK, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY, 0x0)             # <- measured: FAILS (busy=1)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: zero-JOBC refusal", out_dir, name, dict(
        stage="jc0", d=D_PROBE,
        expect="EXPECT-RED rc!=0: LAYER sticky+code2 receipts GREEN (real "
               "unit refused the zero composite), then WALK busy stuck at "
               "S2_FETCH"))


def build_norm2(step, *, out_dir, name) -> dict:
    """PC_NFEED + PC_G2F (EXPECT-RED at the gamma fetch): the walked NFEED
    completes IN-TILE (nf_busy falls — receipt), then the walker parks at
    the G2 fetch (fact F1). Fact F5's second half, measured: even with fuel
    on, the fetched gamma lands on the xw/MXE-weight stream — there is NO
    xw -> xg route in apex_top — so a walked NORM step needs BOTH fuel mode
    and a gamma routing arm (RTL delta), or gamma stays host-fed."""
    d = step["d"]
    s = LayerScript(d)
    s.emit(f"// STEP-MATRIX probe {name}: NFEED completes, then the walk "
           f"parks at PC_G2F (gamma fetch).")
    _prologue(s)
    s.lptr(BANK_RESID, 0)
    s.ldata(step["x_bits"])
    s.lctl(ser_dst=1, resid_arm=1)
    s.route(rdst=1)
    s.pmode(True)
    s.lujob(LU_RESID, d)
    s.ljc(step["comp"])
    s.lujob(LU_DEQ, d)
    inject_frame(s, step["o8"])
    s.lpoll(LST_LDQ | LST_RES, 0x0)
    s.route(rdst=0, asrc=0)
    s.csrw(L_CTRL, LCTL_NSRC)
    load_descriptor(s, probe_desc(
        d, (1 << fmt.EN_NFEED) | (1 << fmt.EN_NORM2)))
    s.csrw(W_CTRL, 0x3)
    s.emit("// receipts: the walker armed code 4 and the whole in-tile feed "
           "path went quiet (NFEED walked to completion)")
    s.csrp(L_CTRL, LCTL_FSRC_EXT2, LCTL_FSRC_EXT2)
    s.lpoll(LST_NFEED, 0x0)
    s.efs(0, 0)                    # pop the walked feed's r1 C-1 row scale
    s.pmode(False)
    s.csrr(W_STATUS, W_ST_ESTK, 0x0)
    s.csrr(W_STATUS, W_ST_BUSY, 0x0)             # <- measured: FAILS (busy=1)
    s.emit("DONE")
    return _finish(s, f"STEP {name}: NORM2 gamma-fetch wedge", out_dir, name,
                   dict(stage="norm2", d=d,
                        expect="EXPECT-RED rc!=0: NFEED receipts GREEN "
                               "(code-4 armed, nf_busy fell), then WALK busy "
                               "stuck at PC_G2F"))


# ═══════════════ grading ═══════════════════════════════════════════════════

def grade_nfeed_d(caps, step: dict) -> dict:
    """elane_norm_feed.grade_nfeed at width d (same decoders, same golden
    references, d-parameterized)."""
    d = step["d"]
    out = {}
    out["r1"] = grade_resid(caps, step["r1_bits"])
    acc = cd.ro_lanes_to_i32(caps, strict=False)
    got = acc[:d].astype(np.int64)
    codes_ok = got.size == d and np.array_equal(got, step["h2_codes"])
    out["h2_codes"] = {"equal": bool(codes_ok), "got": int(got.size)}
    fs = [t for t in cd.fp16_bits(caps) if t["sem"] == "fs"]
    fs_bits = [int(t["bits"]) for t in fs]
    scales_ok = (len(fs_bits) == 2 and fs_bits[0] == step["r1_scale"]
                 and fs_bits[1] == step["h2_scale"])
    out["scales"] = {"equal": bool(scales_ok), "got": fs_bits,
                     "want": [step["r1_scale"], step["h2_scale"]]}
    out["equal"] = bool(out["r1"]["equal"] and codes_ok and scales_ok)
    return out


# ═══════════════ build/run plan ════════════════════════════════════════════

def build_all(out_dir: Path, step: dict) -> list[dict]:
    """Every probe, with its expected verdict encoded."""
    with tg.at_d(D_PROBE):
        mans = [
            build_nfeed(step, out_dir=out_dir, name="p_nfeed"),
            build_nfeed(step, out_dir=out_dir, name="p_nfeed_off",
                        nfeed=False),
            build_rope(out_dir=out_dir, name="p_rope"),
            build_storekv(out_dir=out_dir, name="p_storekv"),
            build_res1(out_dir=out_dir, name="p_res1"),
            build_oproj(out_dir=out_dir, name="p_oproj"),
            build_qkv(out_dir=out_dir, name="p_qkv"),
            build_ffn(out_dir=out_dir, name="p_ffn"),
            build_jc0(out_dir=out_dir, name="p_jc0"),
            build_norm2(step, out_dir=out_dir, name="p_norm2"),
        ]
    return mans


# expected executor verdict per stage: True = rc 0, False = rc != 0
EXPECT_OK = {
    "nfeed": True, "nfeed_off": False, "rope_arm": True, "storekv": True,
    "res1": True, "oproj": False, "qkv": False, "ffn": False, "jc0": False,
    "norm2": False,
}


def smoke(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = toy_step_d(D_PROBE, args.seed)
    plan = build_all(out, step)
    if args.only:
        keep = set(args.only.split(","))
        plan = [m for m in plan if m["name"] in keep]

    rows = []
    for man in plan:
        t0 = time.time()
        r = glo._run([man["path"]], binary=args.binary,
                     tile_div=args.tile_div, timeout_s=args.timeout,
                     cap_out=str(out / f"{man['name']}.cap.jsonl"))
        wall = time.time() - t0
        want_ok = EXPECT_OK[man["stage"]]
        verdict = bool(r["ok"]) == want_ok
        row = dict(name=man["name"], stage=man["stage"], rc=r["rc"],
                   ok=bool(r["ok"]), want_ok=want_ok, as_expected=verdict,
                   caps=len(r["captures"]), wall=round(wall, 1),
                   expect=man["expect"])
        if man["stage"] == "nfeed" and r["ok"]:
            g = grade_nfeed_d(r["captures"], step)
            row["grade"] = {k: v for k, v in g.items() if k != "r1"} | {
                "r1_equal": g["r1"]["equal"]}
            row["as_expected"] = verdict and g["equal"]
        elif man["stage"] == "nfeed_off":
            # THE DISCRIMINATOR: rc!=0 alone is not a bite — on a wrong/
            # broken binary every probe fails and 'as expected' would be
            # vacuous (the 2026-08-04 wrong-twin failure mode; see
            # elane_norm_feed's discriminator-verdict hygiene note). The
            # red must be AT the discriminated seam: the executor aborted
            # ON the LAYER_CTRL route-arm poll (the walker armed nothing).
            # Measured on the clean b64 twin: 'poll stall ... [1070] ...
            # want 00000080/00000080'.
            stalled = stalled_at_poll(r.get("log"), 0x1000 + L_CTRL)
            row["stalled_at_route_arm"] = stalled
            row["as_expected"] = bool(verdict and stalled)
        rows.append(row)
        eprint(f"  [{man['name']:<12}] rc={r['rc']} ok={r['ok']} "
               f"want_ok={want_ok} as_expected={row['as_expected']} "
               f"caps={len(r['captures'])} {wall:.1f}s")

    matrix = dict(binary=args.binary or "(bridge default)",
                  image="b64_05b (CFG_D=CFG_DM=64, GQA=2, QSTAGE=14)",
                  fence="toy: head_dim == d_model == 64, H=1",
                  tile_div=args.tile_div, seed=step["seed"], probes=rows)
    (out / "step_matrix.json").write_text(json.dumps(matrix, indent=1))

    print(f"\nSTEP-MATRIX probes @ b64 twin (tile_div={args.tile_div})")
    for r in rows:
        print(f"  {r['name']:<14} rc={r['rc']} ok={r['ok']} "
              f"want_ok={r['want_ok']} AS_EXPECTED={r['as_expected']}")
    ok = all(r["as_expected"] for r in rows)
    print(f"ELANE STEP MATRIX: {'PASS' if ok else 'FAIL'} "
          f"({sum(r['as_expected'] for r in rows)}/{len(rows)} probes "
          f"behaved exactly as stated)")
    return 0 if ok else 1


def _selftest() -> int:
    ok = True

    def chk(name, cond, note=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {note}" if note and not cond else ""))

    print("[1] toy references at d=64 are golden-derived and deterministic")
    st = toy_step_d(D_PROBE)
    chk("r1 round-trips the fp16 grid", np.array_equal(
        st["r1_bits"], f64_to_f16_bits(f16_bits_to_f64(st["r1_bits"]))))
    chk("determinism", toy_step_d(D_PROBE)["h2_scale"] == st["h2_scale"])
    chk("norm consumes codes", np.array_equal(
        np.asarray(at.quant_rows_i8(np.asarray(
            at.rmsnorm_fx([int(c) for c in st["r1_codes"]],
                          [int(v) for v in st["gamma2"]])[0],
            dtype=np.float64)[None, :] / 256.0)[0][0], dtype=np.int64),
        st["h2_codes"]))

    print("[2] every probe image is walker-legal at cfg_d=64")
    for m in (fmt.EN_NFEED, fmt.EN_ROPE, fmt.EN_STOREKV, fmt.EN_RES1,
              fmt.EN_OPROJ, fmt.EN_QKV):
        w = probe_desc(D_PROBE, 1 << m)
        chk(f"mask bit {m} image legal",
            fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                       cfg_d=D_PROBE) == fmt.ERR_NONE)
    w = probe_desc(D_PROBE, 1 << fmt.EN_FFN, d_ffn=64)
    chk("FFN image legal (d_ffn=64)",
        fmt.check2(w[0], w[1], w[2], w[3], w[fmt.W_STEP],
                   cfg_d=D_PROBE) == fmt.ERR_NONE)

    print("[3] the programs build, audit clean, and carry the b64 identity")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mans = build_all(Path(td), st)
        for m in mans:
            a = m["audit"]
            chk(f"{m['name']}: audit clean (caps={a['caps']})",
                a["caps_with_e"] == 0 and a["violations"] == 0)
        chk("all 10 probes built", len(mans) == 10)
        # the INFO_TIER retarget must have landed (b64_05b reads 0x1)
        p = Path(mans[0]["path"]).read_text()
        chk("INFO_TIER expectation retargeted to the GQA build",
            f'"a":{0x1000 + 0x14}' in p)

    print("[4] the grader discriminates")
    caps = []
    i = 0

    def cap(sem, val):
        nonlocal i
        caps.append({"tag": f"{sem}_{i}", "i": i, "sem": sem,
                     "value": int(val) & 0xFFFFFFFF})
        i += 1
    cap("fs", st["r1_scale"])
    for base in range(0, D_PROBE, 8):
        for j in range(8):
            cap(f"ro_w{j}", int(st["h2_codes"][base + j]))
    cap("fs", st["h2_scale"])
    for b in st["r1_bits"]:
        cap("lrd", int(b))
    chk("all-correct capture set grades equal",
        grade_nfeed_d(caps, st)["equal"])
    bad = [dict(c) for c in caps]
    for c in bad:
        if c["sem"] == "ro_w2":
            c["value"] += 1
            break
    chk("a moved code goes red", not grade_nfeed_d(bad, st)["equal"])

    print(f"ELANE STEP-MATRIX SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tile-div", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--only", default=None,
                    help="comma-separated probe names to run")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.build:
        step = toy_step_d(D_PROBE, args.seed)
        for m in build_all(Path(args.out), step):
            print(f"  built {m['name']}: {m['path']} (audit={m['audit']})")
        return 0
    if args.smoke:
        return smoke(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
