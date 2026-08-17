#!/usr/bin/env python
"""gen_mask_vectors.py — S12/D-027 loadable-mask suite vectors (verif/kvq/mask).

Golden arbiter: golden/apex_golden/cq_codec.py — the SAME packers the D-026
suites import (records/banks straight from pack_key_records, never re-derived)
and the same composition the D-027 golden gates pin executable
(golden/tests/test_mask_semantics.py §B: swap at a record-empty boundary,
scale-bank ssid sequence CONTINUING across the swap).

One geometry, two builds of tb_kvq_mask.sv consume these vectors:
  * rom build (IS_CSR=0): MASK_FILE = mask_m1.u8.hex — stores/reads run1 only.
  * csr build (IS_CSR=1): no ROM — stages+commits M1 over CSR, then runs the
    IDENTICAL run1 (records/banks/readback byte-identical to the rom build by
    both matching these vectors), then swaps to M2 (MASK_SWAP flagged) and
    runs run2 with the ssid sequence continuing, a soft reset injected
    mid-run2 (persistence: the expectations only hold if the live mask, bank
    rows and allocator all survive D-020).

Geometry (fits SCALE_SETS=4 with NO allocator wrap — SB_OVWR must stay 0):
  D=64  TIER=2(bits=4)  G=8  K=2  DEPTH=64
  M1 = {5, 50}   (the b64 ship mask — golden/vectors/.../outlier_mask.u8.hex)
  M2 = {5, 60}   (one channel moved — the masksem §B/§D shape)
  run1 = 12 tokens: group0 full (8, ssid 0) + group1 PARTIAL (4, flush,
         ssid 1) — partial scales over g=4 per §3.1
  run2 = 16 tokens under M2: groupA (8, ssid 2) + groupB (8, ssid 3)
  token t is stored at SRAM addr t (bases 0, 8, 12, 20)

Directed content: token 3 carries -0.0 in outlier ch 5 AND keep ch 7 (the
B-4 signed-zero identity under a CSR-committed mask).

Emits (build/vec/): mask_m1.u8.hex (%02x x D) · stim.f16.hex (28*D x %04x)
· exp_rec.hex (28 x SRAMW-bit rows) · exp_bank.f16.hex (4 x D*16-bit rows,
set index = row) · exp_hat.f32.hex (28*D x %08x) · gen_meta.txt
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))
from apex_golden import cq_codec as cq                    # noqa: E402

D, G, K, BITS = 64, 8, 2, 4
T1, T2 = 12, 16
M1 = [5, 50]
M2 = [5, 60]
SEED = 0xD027
OUT = Path(__file__).resolve().parent / "build" / "vec"
OUT.mkdir(parents=True, exist_ok=True)

SRAMW = cq.sram_row_bits(D, BITS, K, True)
rng = np.random.default_rng(SEED)


def gen_kf16(t, d):
    return np.asarray(rng.uniform(-4.0, 4.0, (t, d)),
                      dtype=np.float16).view(np.uint16)


K1 = gen_kf16(T1, D)
K2 = gen_kf16(T2, D)
# directed: -0.0 in an outlier channel AND a keep channel of the same token
K1[3, 5] = 0x8000
K1[3, 7] = 0x8000

b1 = cq.compress_keys(K1, BITS, G, M1)
b2 = cq.compress_keys(K2, BITS, G, M2)
n1 = len(b1.groups)
assert n1 == 2 and [e - a for a, e in b1.groups] == [8, 4], b1.groups
r1, bank1, ss1 = cq.pack_key_records(b1, SRAMW, ssid0=0)
r2, bank2, ss2 = cq.pack_key_records(b2, SRAMW, ssid0=n1)
n2 = len(b2.groups)
assert n2 == 2 and ss1 == [0, 1] and ss2 == [2, 3], (ss1, ss2)
assert n1 + n2 <= cq.SCALE_SETS, "allocator would wrap -> SB_OVWR pollution"
hat1 = cq.decompress_keys(b1)
hat2 = cq.decompress_keys(b2)

with open(OUT / "mask_m1.u8.hex", "w") as f:
    for c in range(D):
        f.write("%02x\n" % (1 if c in M1 else 0))

with open(OUT / "stim.f16.hex", "w") as f:
    for row in (*K1, *K2):
        for v in row:
            f.write("%04x\n" % int(v))

with open(OUT / "exp_rec.hex", "w") as f:
    for rec in (*r1, *r2):
        f.write("%0*x\n" % (SRAMW // 4, int.from_bytes(rec.tobytes(), "little")))

with open(OUT / "exp_bank.f16.hex", "w") as f:
    for bank in (bank1, bank2):
        for row in bank:
            v = 0
            for c in range(D):
                v |= int(row[c]) << (16 * c)
            f.write("%0*x\n" % (D * 16 // 4, v))

with open(OUT / "exp_hat.f32.hex", "w") as f:
    for hat in (hat1, hat2):
        for row in hat:
            for v in row:
                f.write("%08x\n" % int(v))

with open(OUT / "gen_meta.txt", "w") as f:
    f.write("D=%d G=%d K=%d BITS=%d T1=%d T2=%d SRAMW=%d seed=0x%X\n"
            % (D, G, K, BITS, T1, T2, SRAMW, SEED))
    f.write("M1=%s M2=%s ssids=%s+%s groups1=%s groups2=%s\n"
            % (M1, M2, ss1, ss2, b1.groups, b2.groups))

print("gen_mask_vectors: T1=%d T2=%d groups=%d+%d ssids=%s+%s SRAMW=%d -> %s"
      % (T1, T2, n1, n2, ss1, ss2, SRAMW, OUT))
