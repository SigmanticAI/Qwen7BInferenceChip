#!/usr/bin/env python
"""gen_gqa_vectors.py — IB-LAYER S4b per-KV-head GQA bank suite vectors
(verif/kvq/gqa).

Golden arbiter: golden/apex_golden/cq_codec.py — CQ-8 value-record semantics
via the SAME primitives every KVQ suite imports (compress_values /
decompress_values; per-token fp16 amax scale, INT8 codes, fp32 dequant) —
never re-derived. The bank under test (rtl/top/glue/apex_kvq_gqa_bank.sv)
routes N_ENG verified kvq_engine instances; the numeric truth per engine is
the engine's own (L3/f2-verified at CQ-8), so these vectors' job is to make
ROUTING errors numerically visible:

  * every (engine, record) row is DISTINCT BY CONSTRUCTION — element 0
    carries an engine tag fp16((e+1)) and element 1 a record tag
    fp16((r+1)/8), so a store or readback landing in the wrong engine can
    never alias another engine's golden row (asserted below);
  * per-engine record counts DIFFER (NREC = 4,5,6,7) so OCCUPANCY reads
    distinguish engines;
  * a REWRITE row for (e2, rec 1) proves same-address stores do not leak
    across engines.

Geometry: D=64, BITS=8 (CQ-8), N_ENG=4, DEPTH=256 (the R3 sizing point —
2T <= 256; the TB stores at low addresses and reads DEPTH-1 as the D-016a
unwritten-address probe, so keep every NREC < DEPTH-1).

Emits (build/vec/): stim.f16.hex (sum(NREC)*D x %04x, engine-major record-
major) · exp_hat.f32.hex (same index space, %08x) · stim2.f16.hex /
exp2_hat.f32.hex (the e2-rec1 rewrite row) · gen_meta.txt.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))
from apex_golden import cq_codec as cq                    # noqa: E402

D, BITS = 64, 8
N_ENG = 4
DEPTH = 256
NREC = [4, 5, 6, 7]                    # per-engine record counts (distinct)
REWRITE_E, REWRITE_R = 2, 1
SEED = 0x54B0
OUT = Path(__file__).resolve().parent / "build" / "vec"
OUT.mkdir(parents=True, exist_ok=True)

assert max(NREC) < DEPTH - 1, "keep DEPTH-1 unwritten for the D-016a probe"
rng = np.random.default_rng(SEED)


def gen_rows(n, e_tag, r0):
    """n fp16 rows, tagged: [0]=engine tag, [1]=record tag (aliasing-proof)."""
    rows = np.asarray(rng.uniform(-4.0, 4.0, (n, D)),
                      dtype=np.float16).view(np.uint16)
    for r in range(n):
        rows[r, 0] = np.float16(e_tag + 1.0).view(np.uint16)
        rows[r, 1] = np.float16((r0 + r + 1) / 8.0).view(np.uint16)
    return rows


stim = [gen_rows(NREC[e], float(e), 0) for e in range(N_ENG)]
blobs = [cq.compress_values(stim[e], BITS) for e in range(N_ENG)]
hats = [cq.decompress_values(blobs[e]) for e in range(N_ENG)]

# the rewrite row: same shape machinery, distinct content (engine tag offset)
stim2 = gen_rows(1, 10.0 + REWRITE_E, 16)
hat2 = cq.decompress_values(cq.compress_values(stim2, BITS))

# self-checks (generator fails, never the TB, on a bad construction)
for e in range(N_ENG):
    assert np.all(np.ptp(stim[e].view(np.float16).astype(np.float64),
                         axis=1) > 0), "degenerate row"
for r in range(min(NREC)):
    rows = [stim[e][r].tobytes() for e in range(N_ENG)]
    assert len(set(rows)) == N_ENG, f"cross-engine alias at record {r}"
assert stim2[0].tobytes() != stim[REWRITE_E][REWRITE_R].tobytes()
for e in range(N_ENG):                 # scales positive normal fp16
    for s in blobs[e].scales:
        assert 0 < (int(s) & 0x7C00) < 0x7C00, hex(int(s))

with open(OUT / "stim.f16.hex", "w") as f:
    for e in range(N_ENG):
        for row in stim[e]:
            for v in row:
                f.write("%04x\n" % int(v))

with open(OUT / "exp_hat.f32.hex", "w") as f:
    for e in range(N_ENG):
        for row in hats[e]:
            for v in row:
                f.write("%08x\n" % int(v))

with open(OUT / "stim2.f16.hex", "w") as f:
    for v in stim2[0]:
        f.write("%04x\n" % int(v))

with open(OUT / "exp2_hat.f32.hex", "w") as f:
    for v in hat2[0]:
        f.write("%08x\n" % int(v))

with open(OUT / "gen_meta.txt", "w") as f:
    f.write("D=%d BITS=%d N_ENG=%d DEPTH=%d SEED=0x%X\n"
            % (D, BITS, N_ENG, DEPTH, SEED))
    f.write("NREC=%s total_rec=%d rewrite=(e%d,r%d)\n"
            % (NREC, sum(NREC), REWRITE_E, REWRITE_R))

print("GQA VECTORS: %d records over %d engines (+1 rewrite row), "
      "golden = cq_codec compress/decompress_values (CQ-8), "
      "aliasing self-checks PASS" % (sum(NREC), N_ENG))
