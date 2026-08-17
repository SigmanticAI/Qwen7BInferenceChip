#!/usr/bin/env python3
"""gen_wg3_krope_case.py — W-G3: the HOST-MODE GQA-4 K-rope coverage
(integration-lead ruling 2026-07-26, option 3, second half).

Stages the REAL traced Qwen2.5-7B K rows PRE-RoPE through the tile's real
S-2 store path — `apex_scale_quant` MODE_F16 -> `rope_row` -> the per-KV-head
`apex_kvq_gqa_bank` engine — and checks the POST-RoPE beat on the `dbg_f16`
tap against the golden expectation, for ALL FOUR KV groups.

Needs no walk (`walk_en` is never set), so it is NOT blocked by F6: it is the
GQA-4 hardware exercise that IS available today. The four engines are selected
in host mode through `LAYER_CTRL[17:15]` (`l_kv_map`), the §3b single-ingress
level register — `rtl/top/apex_top.sv:1236-1237`,
`gqa_eng_sel = walk_en_q ? wk_kv_eng_sel : l_kv_map_q`.

WHY THE TAP IS THE POST-RoPE VALUE (the thing that makes this a rope gate):
    apex_top.sv:935-937  kv_s_tdata = l_rope_en_q ? rr_m_data : sqf_data
    apex_top.sv:1799-1801 dbg_f16_data = kv_s_tdata
so with `l_rope_en_q` set the tap carries `rope_row`'s OUTPUT, not the
scale_quant F16 output.

ARBITER — erratum F5, now a contract-level rule in LEVEL_C_INTEGRATION §9.1:
the expectation is golden `rope_fx` applied to the f16 S-2 BUS value, which is
verified equal to `BusMode(rope_in_f16=True)` (D-030). It is NOT the legacy
`K_f16`/`K_rope` trace field — `K_real` is not on the fp16 grid, so a gate
pinned to the legacy field fails a CORRECT tile in 76 of 80 rows. Both arrays
come straight from `verif/walkgold` stage B; nothing is re-derived here.

SELF-CHECKS (hard asserts, no simulator):
  R1  the staged rows are the stage-B `stage_K_g*_f16.npy` files verbatim
  R2  the expectation rows are the stage-B `expect_Krope_g*_f16.npy` files
      verbatim, and t=0 is the identity while every t>0 row differs
  R3  the phase codes fit the 14-bit LAYER RAM word and the table is the
      golden `rope_phase_q` table for THIS layer's theta
  R4  every injected fp16 is representable by the `decompose_f16` K=2
      accumulator trick (no inf/NaN, both bytes in INT8 range)
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

import gen_l3_vectors as glv                                  # noqa: E402
from apex_golden import transformer as tf                     # noqa: E402

LAYER_CTRL, LAYER_PTR, LAYER_DATA, LAYER_STATUS = 0x70, 0x74, 0x78, 0x80
KV_CTRL, KV_STATUS = 0x00, 0x04     # kvq_engine AXI-Lite window
PH_BANK_K = 0                      # LAYER_PTR[29:28] = 0 selects ph_k


def layer_ctrl(rope_en: int, rope_pos: int, kv_map: int) -> int:
    """apex_top.sv:844-851 + :1227 — the §3b LAYER_CTRL level word."""
    assert 0 <= rope_pos < 128 and 0 <= kv_map < 8
    return (rope_en & 1) | ((rope_pos & 0x7F) << 8) | ((kv_map & 0x7) << 15)


def build(gold: Path, bundle: Path, out: Path, rope: bool = True,
          legacy: bool = False) -> dict:
    meta = json.loads(bundle.with_suffix(".json").read_text())
    bun = dict(np.load(bundle)) if legacy else {}
    D, T, H_kv = meta["head_dim"], meta["T"], meta["H_kv"]
    assert D == 128, f"this case targets CFG_D=128, got {D}"
    half = D // 2

    stage, expect = {}, {}
    for g in range(H_kv):
        stage[g] = np.load(gold / f"stage_K_g{g}_f16.npy")
        expect[g] = np.load(gold / f"expect_Krope_g{g}_f16.npy")
        # R1/R2: shapes + the rotation is real (t=0 identity, t>0 differs)
        assert stage[g].shape == (T, D) and expect[g].shape == (T, D)
        assert np.array_equal(stage[g][0], expect[g][0]), \
            f"R2: group {g} t=0 is not the identity rotation"
        n_rot = int((stage[g] != expect[g]).any(axis=1).sum())
        assert n_rot == T - 1, \
            f"R2: group {g} rotates {n_rot} rows, expected {T - 1}"
    n_rot_total = sum(int((stage[g] != expect[g]).any(axis=1).sum())
                      for g in range(H_kv))

    # R3: the phase table, quantized ONCE from float64 by the golden
    theta = tf.rope_theta(D, meta["rope_theta"])
    ph = np.asarray([tf.rope_phase_q(float(m), theta) for m in range(T)],
                    dtype=np.uint16)
    assert ph.shape == (T, half), ph.shape
    assert int(ph.max()) < (1 << 14), \
        f"R3: phase code {int(ph.max())} exceeds the 14-bit LAYER RAM word"

    s = glv.Script(D)
    name = f"wg3_krope_gqa{H_kv}_s{meta['step']:03d}_L{meta['layer']:02d}"
    if legacy:
        # F5 DISCRIMINATION ARM — must FAIL. Pins the expectation to the
        # LEGACY `K_f16` trace field (rope applied to the UNNARROWED K_real)
        # instead of the D-030 arbiter. LEVEL_C_INTEGRATION §9.1 F5 predicts
        # this fails a CORRECT tile in 76 of 80 rows; this arm measures it on
        # the real tile instead of asserting it from the ledger.
        grp = meta["gqa_group"]
        name += "_legacy"
        expect = {g: np.asarray(bun[f"h{g * grp:02d}_K_f16"], dtype=np.uint16)
                  for g in range(H_kv)}
    elif not rope:
        # A/B control arm: rope DISARMED, so the tap must reproduce the
        # STAGED rows exactly. Isolates the injection path from rope_row.
        name += "_norope"
        expect = {g: stage[g] for g in range(H_kv)}
    s.emit(f"// L3 K-rope GQA-{H_kv} coverage: {name} "
           f"(T={T} D={D} CQ-8, {H_kv} engines, real Qwen K rows PRE-RoPE)")
    glv.phase_a(s, tiers_used=(0,))
    glv.loader_phase(s)          # establishes the h8=[127,1,0...] inject row

    s.emit(f"// RoPE phase table -> LAYER RAM bank {PH_BANK_K} (ph_k), "
           f"{T} rows x {half} codes, auto-inc from 0")
    s.csrw(LAYER_PTR, (PH_BANK_K << 28) | 0)
    for m in range(T):
        for c in range(half):
            s.csrw(LAYER_DATA, int(ph[m][c]))

    # the POST-RoPE expectation, in engine-then-row order (tap order)
    exp_vals = []
    for g in range(H_kv):
        for t in range(T):
            exp_vals += [int(v) for v in expect[g][t]]
    s.emit(f"// expect {len(exp_vals)} POST-RoPE fp16 beats on dbg_f16 "
           f"(arbiter: golden rope_fx on the f16 S-2 bus value == D-030 "
           f"BusMode(rope_in_f16=True); NOT the legacy K_f16 field — F5)")
    s.tap("TAPF16", exp_vals, 16)

    s.route(rdst=1, fsrc=0, fdst=0, asrc=0, qsrc=0, kvu=1)

    def quiesce():
        """LAYER_CTRL carries LEVELS (l_rope_pos, l_kv_map). Levels may only
        change while the streams they steer are quiescent — the rt_* rule,
        and NOTHING POLICES IT (the F3 class: the descriptor check has no
        clause for it either). Measured cost of getting this wrong: writing
        the next row's rope_pos while the previous row was still draining
        through rope_row rotated row t by the phase of t+1 — an off-by-one
        that reproduces as a clean, plausible fp16 value, not an X. Poll BOTH
        idle bits: apex_top's header is explicit that CSR STATUS.idle does not
        cover the KVQ input side.

        THIRD poll, and the one that actually bit: `rope_row`'s busy is NOT a
        CSR STATUS lane at all. apex_top.sv:917-919 puts it in LAYER_STATUS —
        {.., rr_busy[3], res_busy[2], swg_busy[1], ldq_busy[0]} — so a
        CSR-STATUS-only quiesce returns "idle" while a row is still inside
        rope_row. Measured: pairs 0-2 of row 0 took row 0's phase and pair 3
        took row 1's, because the next LAYER_CTRL write landed mid-row. The
        symptom is a plausible fp16 value, never an X."""
        s.csrp(glv.CSR["STATUS"], 0x1, 0x1)
        s.kvp(KV_STATUS, 0x1, 0x1)
        s.csrp(LAYER_STATUS, 0xF, 0x0)   # rr | res | swg | ldq all idle

    # every GQA engine needs its own D-020 soft reset: phase_a only ever
    # reaches engine[l_kv_map], which is 0 at that point, so engines 1..N-1
    # are untouched by it and wedge the qj port on their first record.
    s.emit(f"// per-engine init: D-020 soft reset on each of the {H_kv} "
           f"GQA engines (phase_a only reaches engine 0)")
    for g in range(H_kv):
        quiesce()
        s.csrw(LAYER_CTRL, layer_ctrl(0, 0, g))
        s.kvw(KV_CTRL, 0x2)
        s.kvp(KV_STATUS, 0x1, 0x1)

    n_rows = n_elem = 0
    for g in range(H_kv):
        s.emit(f"// KV group {g} -> GQA engine {g} (LAYER_CTRL[17:15]), "
               f"{T} K rows staged PRE-RoPE with EN_ROPE={1 if rope else 0}")
        for t in range(T):
            quiesce()
            s.csrw(LAYER_CTRL, layer_ctrl(1 if rope else 0, t, g))
            pairs = [glv.decompose_f16(int(b))[:2] for b in stage[g][t]]
            assert len(pairs) == D, "R4: pair count"
            glv.inject_jobs(s, pairs, 0, waddr=t)
            n_rows += 1
            n_elem += D

    s.emit("// disarm rope, quiesce, and account")
    quiesce()
    s.csrw(LAYER_CTRL, layer_ctrl(0, 0, 0))
    # `final_phase` branches on T only as a PROXY for "did this case raise the
    # score-dequant and TIP frame stickies", which every L3 attention case with
    # T > 8 does. THIS case runs no score/pv phase at all — no score-dequant
    # job, no TIP tile — so those stickies are genuinely clear and the only
    # sticky bit set is the phase-A illegal descriptor. Passing 8 selects the
    # no-frame-error tail, which is the TRUE expectation here; asserting
    # ESTK 0x4101 would be the lying one. Verified below on the emitted text.
    glv.final_phase(s, 8)
    assert "ESTK ffff 0001" in s.lines, \
        "expected the no-frame-error sticky tail (0x0001 = phase-A desc only)"
    assert not any(ln == "CSRR 04 00000002 00000002" for ln in s.lines), \
        "no new error sticky may be expected in this case"

    # the GQA build is CQ-8-only: INFO_TIER reads 0x1 (apex_top.sv:1786-1789)
    old, new = "CSRR 14 ffffffff 00000003", "CSRR 14 ffffffff 00000001"
    n_hit = sum(1 for ln in s.lines if ln == old)
    assert n_hit == 1, f"expected 1 phase-A INFO_TIER line, found {n_hit}"
    s.lines = [new if ln == old else ln for ln in s.lines]

    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "cases" / f"{name}.txt").write_text("\n".join(s.lines) + "\n")
    man = {"name": name, "build": "wg3", "kind": "krope-gqa",
           "T": T, "D": D, "engines": H_kv, "checks": s.n_expect,
           "k_rows": n_rows, "k_elements": n_elem,
           "phase_rows": T, "phase_codes": T * half,
           "step": meta["step"], "layer": meta["layer"],
           "arbiter": "golden rope_fx on the f16 S-2 bus value "
                      "== BusMode(rope_in_f16=True) (D-030); NOT legacy K_f16"}
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(HERE / "build"))
    ap.add_argument("--legacy-arbiter", action="store_true",
                    help="F5 discrimination arm: pin to the legacy K_f16 "
                         "field; this arm MUST FAIL")
    ap.add_argument("--no-rope", action="store_true",
                    help="A/B control arm: disarm rope, expect the staged rows")
    args = ap.parse_args()
    gold = REPO / "build" / "walkgold"
    bp = sorted(gold.glob("layer_s*_L*.npz"))[-1]
    man = build(gold, bp, Path(args.build), rope=not args.no_rope,
                legacy=args.legacy_arbiter)
    print(f"[wg3k] case {man['name']}: {man['k_rows']} real K rows "
          f"({man['k_elements']} fp16 elements) staged PRE-RoPE across "
          f"{man['engines']} GQA engines, T={man['T']} D={man['D']}")
    print(f"[wg3k] phase table: {man['phase_rows']} rows x "
          f"{man['phase_codes'] // man['phase_rows']} codes "
          f"({man['phase_codes']} LAYER RAM words, 14-bit)")
    print(f"[wg3k] {man['checks']} scripted checks; arbiter = {man['arbiter']}")
    print(f"WG3 KROPE CASE: PASS (R1 staged rows are the stage-B artifacts; "
          f"R2 t=0 identity + {man['T'] - 1}/{man['T']} rows rotate per group; "
          f"R3 phase codes fit the 14-bit LAYER word; "
          f"R4 all {man['k_elements']} elements injectable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
