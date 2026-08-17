#!/usr/bin/env python3
"""gen_rope_row_vectors.py — golden vectors for rope_row (IB-LAYER S2).

Arbiter: golden/apex_golden/transformer.py C-ROPE — expected output beat
y[k] is the float64-exact rotation of the fp16 inputs, narrowed once:
  pair j = k mod half, c = cos_fx(u_q[j])/2^14, s = sin_fx(u_q[j])/2^14
  y[j]      = f16(x[j]*c - x[j+half]*s)        (lo pass, k < half)
  y[j+half] = f16(x[j+half]*c + x[j]*s)        (hi pass, k >= half)
The per-pair composition is CROSS-CHECKED at gen time against the full
rope_fx(m, theta) function on the real-tensor rows (bit-identical, or the
generator refuses to emit).

Row classes (counts printed):
  sweep   : every phase code 0..2^14-1 appears at least once (asserted)
  tensor  : golden L4 case K rows + q rows (fp16 bus values, phases from
            rope_phase_q with the case's theta — incl. the 1e6 base case)
  tie_n   : engineered NORMAL-path rounding ties     (R4 kill guarantee)
  tie_s   : engineered SUBNORMAL-path rounding ties  (R6 kill guarantee)
  sat     : saturating rotations, |result| >= 65536 -> fp16 inf (R7)
  zsign   : +-0 inputs x phases with c==0 / s==0 exactly (zero-sign rules)
  subn    : subnormal/zero-dense random rows
  rand    : random finite rows
  bad     : early-last and missing-last frames (frame_error paths)

File format (build/vectors_rope_d<D>.txt):
  ROW <kind>            good row: HALF phase lines, D input lines, D expected
  BADE <pos>            early-last row: pos+1 input lines (last at pos)
  BADM <extra>          missing-last: D+extra input lines, last on final beat
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "golden"))
sys.path.insert(0, str(_REPO / "verif" / "top" / "l4"))

import numpy as np  # noqa: E402

from apex_golden import transformer as tf  # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f16_bits  # noqa: E402

CS = 1 << 14


def f16v(bits):
    return f16_bits_to_f64(np.array([bits], dtype=np.uint16)).item()


def v24(bits):
    e = (bits >> 10) & 0x1F
    m = bits & 0x3FF
    assert e != 0x1F
    mag = m if e == 0 else ((1024 + m) << (e - 1))
    return -mag if (bits >> 15) else mag


def pair_expect(u_q, xi, xih):
    """Golden per-pair rotation of fp16 bit patterns -> (y_i, y_ih) bits."""
    c = float(tf.cos_fx(np.array([u_q]))[0]) / CS
    s = float(tf.sin_fx(np.array([u_q]))[0]) / CS
    lo = f16v(xi) * c - f16v(xih) * s          # exact in float64 (rope.sv hdr)
    hi = f16v(xih) * c + f16v(xi) * s
    return (int(f64_to_f16_bits(np.array([lo]))[0]),
            int(f64_to_f16_bits(np.array([hi]))[0]))


def row_expect(phases, x):
    d = len(x)
    half = d // 2
    y = [0] * d
    for j in range(half):
        y[j], y[j + half] = pair_expect(phases[j], x[j], x[j + half])
    return y


def cs_ints(u_q):
    return (int(tf.cos_fx(np.array([u_q]))[0]),
            int(tf.sin_fx(np.array([u_q]))[0]))


def tie_info(u_q, xi, xih):
    """(lo_tie_normal, lo_tie_subnormal) for the exact product P_lo."""
    c, s = cs_ints(u_q)
    P = v24(xi) * c - v24(xih) * s
    if P == 0:
        return (False, False)
    mag = abs(P)
    p = mag.bit_length() - 1
    E = p - 38
    if E >= -14:
        sh = p - 10
        if sh <= 0:
            return (False, False)
        return ((mag & ((1 << sh) - 1)) == (1 << (sh - 1)), False)
    sh = 14
    return (False, (mag & ((1 << sh) - 1)) == (1 << (sh - 1)))


def rand_f16(rng, n, sub_frac=0.0):
    out = []
    for _ in range(n):
        b = int(rng.integers(0, 1 << 16))
        if (b >> 10) & 0x1F == 0x1F:
            b &= ~0x7C00
        if sub_frac and rng.random() < sub_frac:
            b &= ~0x7C00                       # force subnormal/zero
        out.append(b)
    return out


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build")
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    outdir.mkdir(parents=True, exist_ok=True)
    half = D // 2
    rng = np.random.default_rng(0x0DE0 + D)
    rows = []                                  # (kind, phases, x) or bads

    # ── sweep: all 2^14 phase codes ─────────────────────────────────────────
    codes = list(range(CS))
    n_sweep = (CS + half - 1) // half
    seen = set()
    for r in range(n_sweep):
        ph = [codes[(r * half + j) % CS] for j in range(half)]
        seen.update(ph)
        rows.append(("sweep", ph, rand_f16(rng, D)))
    assert len(seen) == CS, "phase sweep incomplete"

    # ── tensor rows: golden L4 cases (self-check vs full rope_fx) ───────────
    from l4_cases import TIER, build_case  # noqa: E402
    n_tensor = 0
    for name in (("l4_h2_hd64", "l4_qwen_theta") if D == 64
                 else ("l4_h1_hd128",)):
        X, w, q_pos = build_case(name)
        r = tf.decoder_layer_fx(X, w, TIER, q_pos=q_pos)
        theta = tf.rope_theta(w.head_dim, w.rope_theta_base)
        m_q = r.T if q_pos is None else q_pos
        for g_i in range(r.H_kv):
            sl = slice(g_i * w.head_dim, (g_i + 1) * w.head_dim)
            for t in range(r.T):
                xb = [int(v) for v in f64_to_f16_bits(r.K_real[t, sl])]
                ph = [int(v) for v in tf.rope_phase_q(t, theta)]
                exp_full = f64_to_f16_bits(
                    tf.rope_fx(f16_bits_to_f64(
                        np.array(xb, dtype=np.uint16)), t, theta))
                got = row_expect(ph, xb)
                assert got == [int(v) for v in exp_full], \
                    "per-pair composition disagrees with rope_fx"
                rows.append(("tensor", ph, xb))
                n_tensor += 1
        for h_i in range(r.H):
            sl = slice(h_i * w.head_dim, (h_i + 1) * w.head_dim)
            xb = [int(v) for v in f64_to_f16_bits(r.q_real[sl])]
            ph = [int(v) for v in tf.rope_phase_q(m_q, theta)]
            rows.append(("tensor", ph, xb))
            n_tensor += 1

    # ── engineered rounding ties (normal + subnormal paths) ─────────────────
    ties_n, ties_s = [], []
    while len(ties_n) < 48 or len(ties_s) < 48:
        u = int(rng.integers(0, CS))
        xi = int(rng.integers(0, 1 << 16))
        if (xi >> 10) & 0x1F == 0x1F:
            xi &= ~0x7C00
        if len(ties_s) < 48 and rng.random() < 0.6:
            xi &= 0x83FF                       # subnormal candidates
        tn, ts = tie_info(u, xi, 0)
        if tn and len(ties_n) < 48:
            ties_n.append((u, xi))
        if ts and len(ties_s) < 48:
            ties_s.append((u, xi))
    for kind, lst in (("tie_n", ties_n), ("tie_s", ties_s)):
        for i in range(0, len(lst), half):
            chunk = lst[i:i + half]
            ph = [u for (u, _) in chunk] + [0] * (half - len(chunk))
            x = [xi for (_, xi) in chunk] + [0] * (half - len(chunk))
            rows.append((kind, ph, x + [0] * half))

    # ── saturating rotations -> inf (R7) ────────────────────────────────────
    sat_ph = []
    for u in range(CS):
        c, s = cs_ints(u)
        if abs(c) + abs(s) > int(1.30 * CS):   # |c|+|s| ~ sqrt2 region
            sat_ph.append(u)
        if len(sat_ph) >= half:
            break
    assert len(sat_ph) >= 8
    sat_x = []
    for j in range(min(half, len(sat_ph))):
        c, s = cs_ints(sat_ph[j])
        sat_x.append(0x7BFF | (0x8000 if c < 0 else 0))       # x_i
    sat_row_ph = sat_ph + [0] * (half - len(sat_ph))
    xi_row = sat_x + [0] * (half - len(sat_x))
    xih_row = []
    for j in range(half):
        if j < len(sat_ph):
            c, s = cs_ints(sat_ph[j])
            xih_row.append(0x7BFF | (0x8000 if s > 0 else 0))  # -> lo adds up
        else:
            xih_row.append(0)
    rows.append(("sat", sat_row_ph, xi_row + xih_row))
    exp_sat = row_expect(sat_row_ph, xi_row + xih_row)
    assert any((e & 0x7FFF) == 0x7C00 for e in exp_sat), \
        "saturation battery produced no inf"

    # ── zero-sign battery: +-0 x with c==0 / s==0 phases ────────────────────
    u_c0 = next(u for u in range(CS) if cs_ints(u)[0] == 0)
    u_s0 = next(u for u in range(CS) if cs_ints(u)[1] == 0)
    zx = [0x0000, 0x8000]
    zrows = []
    for u in (u_c0, u_s0, 0x155):
        for a in zx:
            for b in zx + [0x3C00, 0xBC00]:
                zrows.append((u, a, b))
                zrows.append((u, b, a))
    for i in range(0, len(zrows), half):
        chunk = zrows[i:i + half]
        ph = [u for (u, _, _) in chunk] + [0] * (half - len(chunk))
        xi = [a for (_, a, _) in chunk] + [0] * (half - len(chunk))
        xih = [b for (_, _, b) in chunk] + [0] * (half - len(chunk))
        rows.append(("zsign", ph, xi + xih))

    # ── subnormal-dense + random rows ───────────────────────────────────────
    for _ in range(24):
        rows.append(("subn", [int(rng.integers(0, CS)) for _ in range(half)],
                     rand_f16(rng, D, sub_frac=0.7)))
    for _ in range(64):
        rows.append(("rand", [int(rng.integers(0, CS)) for _ in range(half)],
                     rand_f16(rng, D)))

    # ── emit ────────────────────────────────────────────────────────────────
    counts = {}
    fn = outdir / f"vectors_rope_d{D}.txt"
    with fn.open("w") as fh:
        bad_slots = {7: ("BADE", D // 3), 23: ("BADM", 5), 41: ("BADE", 0)}
        for ri, (kind, ph, x) in enumerate(rows):
            if ri in bad_slots:
                bk, arg = bad_slots[ri]
                fh.write(f"{bk} {arg}\n")
                if bk == "BADE":
                    for i in range(arg + 1):
                        fh.write(f"{int(rng.integers(0, 0x7C00)):04x}\n")
                else:
                    for i in range(D + arg):
                        fh.write(f"{int(rng.integers(0, 0x7C00)):04x}\n")
                counts["bad"] = counts.get("bad", 0) + 1
            exp = row_expect(ph, x)
            fh.write(f"ROW {kind}\n")
            for u in ph:
                fh.write(f"{u:04x}\n")
            for b in x:
                fh.write(f"{b:04x}\n")
            for e in exp:
                fh.write(f"{e:04x}\n")
            counts[kind] = counts.get(kind, 0) + 1
    total = sum(counts.values())
    print(f"ROPEROW VECTORS D={D}: "
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" total_rows={total} (tensor composition == rope_fx: VERIFIED"
          + f" on {n_tensor} rows)")


if __name__ == "__main__":
    main()
