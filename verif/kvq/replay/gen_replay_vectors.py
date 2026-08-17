#!/usr/bin/env python3
"""gen_replay_vectors.py — S8 trace → KVQ cores-TB replay vectors.

Converts run_tinynpu.py job traces (docs/results/s8_7b_token/*/job_*.npz —
attention/KVQ jobs sampled from a REAL Qwen2.5-7B token stream) into the
verif/kvq/cores vector format, so tb_cores.sv replays the exact fp16 rows the
model produced through the real cq_value_path / cq_key_path RTL and checks
scale / payload / code / x̂ BIT-EXACT against what the golden pipeline stored
during generation.

Fidelity chain: expectations here are derived from the blob fields STORED in
the trace (scales, payloads, sidecars), not from a fresh golden run — and a
built-in self-check recomputes the golden compression from the raw rows and
asserts it matches the stored fields, so trace, golden, and TB expectations
are pinned to each other. (run_tinynpu.py --verify-trace independently pins
the trace to golden attention_core.)

Tier mapping (one output dir per (D, tier) — the TB is built per config):
  CQ-8 : TIER=0. Keys AND values are per-token INT8 value records — every
         traced row becomes a val.txt line (key grouping does not exist in
         CQ-8; the traced G only matters for CQ-4 key groups).
  CQ-4 : TIER=1. Values → val.txt (INT4); keys → key.txt per-channel groups
         at the traced G (full + partial/flush groups per §3.1).
  CQ-4+: TIER=2 — refused for now (runner refuses kvq4p until S12; the
         mask-ROM build flow would go here).

Usage (generation is pure Python — safe on a busy machine):
  python gen_replay_vectors.py --trace ../../../docs/results/s8_7b_token/smoke_trace --name smoke
Then, ON A QUIET MACHINE ONLY (serialize EDA vs model jobs):
  make -C verif/kvq/replay run NAME=smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "golden"))

from apex_golden import cq_codec as cq                           # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f32_bits      # noqa: E402

TIER_CODE = {"CQ-8": 0, "CQ-4": 1, "CQ-4+": 2}


def widen_f16(u16: int) -> int:
    return int(f64_to_f32_bits(f16_bits_to_f64(np.array([u16], np.uint16)))[0])


class ReplayGen:
    def __init__(self, D: int, tier: str, G: int, outdir: Path):
        self.D, self.tier, self.G = D, tier, G
        self.tcode = TIER_CODE[tier]
        self.bits = 8 if tier == "CQ-8" else 4
        self.out = outdir
        self.out.mkdir(parents=True, exist_ok=True)
        self.val_lines: list[str] = []
        self.key_lines: list[str] = []
        self.ng = 0
        self.prov: list[dict] = []
        self.outlier: list[int] = []

    # ── value records (per-token; K and V in CQ-8, V in CQ-4) ────────────────
    def add_value_rows(self, raw: np.ndarray, scales: np.ndarray,
                       payload: np.ndarray, job: str, tensor: str) -> None:
        T, D = raw.shape
        assert D == self.D
        pay_bytes = D if self.bits == 8 else D // 2
        for t in range(T):
            s = int(scales[t])
            pay = payload[t * pay_bytes:(t + 1) * pay_bytes]
            codes = (cq.unpack_int8(pay) if self.bits == 8
                     else cq.unpack_int4(pay, D))
            # self-check: stored blob fields == fresh golden compression
            s_g = int(cq.scale_from_amax(cq.amax_bits(raw[t]), self.bits))
            codes_g = cq.quant_codes(raw[t], np.uint16(s_g), self.bits)
            assert s_g == s and np.array_equal(codes_g, codes), \
                f"self-check: stored {tensor} blob != golden ({job} row {t})"
            hat = cq.dequant_f32(codes, np.uint16(s))
            self.prov.append({"kind": "val", "idx": len(self.val_lines),
                              "job": job, "tensor": tensor, "row": t})
            self.val_lines.append(" ".join([
                " ".join(f"{int(w):04x}" for w in raw[t]),
                f"{s:04x}",
                " ".join(f"{int(b):02x}" for b in pay),
                " ".join(f"{int(c) & 0xFF:02x}" for c in codes),
                " ".join(f"{int(h):08x}" for h in hat)]))

    # ── key groups (CQ-4/CQ-4+ grouped per-channel records) ──────────────────
    def add_key_blob(self, raw: np.ndarray, z: np.lib.npyio.NpzFile,
                     job: str) -> None:
        T, D = raw.shape
        G = int(z["kb_G"])
        assert G == self.G, f"job G={G} != dir G={self.G}"
        keep = z["kb_keep"]
        outlier = list(int(v) for v in z["kb_outlier"])
        if self.outlier and outlier != self.outlier:
            raise SystemExit(f"ABORT: mixed outlier sets in one dir ({job})")
        self.outlier = outlier
        is_out = np.zeros(D, bool)
        is_out[outlier] = True
        nk = len(keep)
        scales, payload = z["kb_scales"], z["kb_payload"]
        # self-check against a fresh golden compression of the raw rows
        kb_g = cq.compress_keys(raw, 4, G, np.asarray(outlier, np.int64))
        assert (np.array_equal(kb_g.scales, scales)
                and np.array_equal(kb_g.payload, payload)
                and np.array_equal(kb_g.sidecar,
                                   z["kb_sidecar"].reshape(T, -1))), \
            f"self-check: stored key blob != golden ({job})"
        sc_base = byte_base = 0
        for a in range(0, T, G):
            e = min(a + G, T)
            g = e - a
            gn = g * nk
            nbytes = (gn + 1) // 2
            codes = cq.unpack_int4(payload[byte_base:byte_base + nbytes],
                                   gn).reshape(g, nk)
            byte_base += nbytes
            field = np.zeros(D, np.uint16)
            code_full = np.zeros((g, D), np.int64)
            for ci, cc in enumerate(keep):
                field[cc] = scales[sc_base + ci]
                code_full[:, cc] = codes[:, ci]
            sc_base += nk
            self.prov.append({"kind": "key_group", "idx": self.ng,
                              "job": job, "base": a, "g": g})
            self.key_lines.append(str(g))
            self.key_lines.append("1" if g == G else "0")
            for t in range(g):
                self.key_lines.append(
                    " ".join(f"{int(w):04x}" for w in raw[a + t]))
            self.key_lines.append(
                " ".join(f"{int(field[c]):04x}" for c in range(D)))
            for t in range(g):
                self.key_lines.append(" ".join(
                    f"{(1 if is_out[c] else int(code_full[t, c]) & 0xFF):02x}"
                    for c in range(D)))
            for t in range(g):
                self.key_lines.append(" ".join(
                    f"{(int(raw[a + t, c]) if is_out[c] else int(field[c])):04x}"
                    for c in range(D)))
            for t in range(g):
                row = [widen_f16(int(raw[a + t, c])) if is_out[c] else
                       int(cq.dequant_f32(np.array([code_full[t, c]]),
                                          np.uint16(field[c]))[0])
                       for c in range(D)]
                self.key_lines.append(" ".join(f"{v:08x}" for v in row))
            self.ng += 1

    def write(self, meta: dict) -> None:
        k_out = len(self.outlier)
        (self.out / "cfg.txt").write_text(
            f"{self.D} {self.tcode} {self.G} {k_out} "
            f"{len(self.val_lines)} {self.ng}\n")
        (self.out / "val.txt").write_text("\n".join(self.val_lines) + "\n")
        if self.key_lines:
            (self.out / "key.txt").write_text("\n".join(self.key_lines) + "\n")
        if k_out:
            with open(self.out / "mask.u8.hex", "w") as f:
                mask = np.zeros(self.D, int)
                mask[self.outlier] = 1
                for c in range(self.D):
                    f.write(f"{mask[c]:02x}\n")
        (self.out / "provenance.json").write_text(json.dumps(
            {"source": meta, "entries": self.prov}, indent=1))
        print(f"[replay-gen] {self.out}: D={self.D} TIER={self.tcode} "
              f"G={self.G} K_OUT={k_out} NV={len(self.val_lines)} "
              f"NG={self.ng}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True,
                    help="run_tinynpu.py --trace-dir output directory")
    ap.add_argument("--name", required=True,
                    help="output name under build/ (e.g. smoke)")
    args = ap.parse_args()
    tdir = Path(args.trace)
    man = json.loads((tdir / "manifest.json").read_text())
    if not man["jobs"]:
        sys.exit("ABORT: trace has no jobs")

    gens: dict[tuple, ReplayGen] = {}
    for j in man["jobs"]:
        z = np.load(tdir / j["file"])
        meta = json.loads(str(z["meta_json"]))
        D, tier, G = meta["D"], meta["tier"], meta["G"]
        if tier == "CQ-4+":
            sys.exit("ABORT: CQ-4+ replay needs the S12 mask-ROM flow")
        key = (D, tier, G)
        if key not in gens:
            gens[key] = ReplayGen(D, tier, G,
                                  HERE / "build" / args.name)
        if len(gens) > 1:
            sys.exit(f"ABORT: mixed (D,tier,G) configs in one trace: "
                     f"{list(gens)} — split traces or extend to multi-dir")
        gen = gens[key]
        K_raw = z["K_f16"].astype(np.uint16)
        V_raw = z["V_f16"].astype(np.uint16)
        if int(z["kb_kind"]) == 0:                    # CQ-8: keys as values
            gen.add_value_rows(K_raw, z["kb_scales"], z["kb_payload"],
                               j["file"], "K")
        else:                                         # CQ-4 grouped keys
            gen.add_key_blob(K_raw, z, j["file"])
        gen.add_value_rows(V_raw, z["vb_scales"], z["vb_payload"],
                           j["file"], "V")

    (D, tier, G), gen = next(iter(gens.items()))
    gen.write({"trace": str(tdir), "manifest_jobs": len(man["jobs"]),
               "run": man.get("run", {}).get("model", "?"),
               "git": man.get("run", {}).get("git", "?")})
    print(f"[replay-gen] SELF-CHECK OK: every stored blob field matches a "
          f"fresh golden compression of the raw rows")
    print(f"[replay-gen] next (QUIET MACHINE ONLY): "
          f"make -C verif/kvq/replay run NAME={args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
