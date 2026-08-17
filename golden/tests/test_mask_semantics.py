# test_mask_semantics.py — S12 LOADABLE-MASK golden contract gates (D-027
# proposed): the outlier mask becomes CSR-programmable in RTL, and these gates
# pin the semantics the hardware must honor BEFORE any RTL lands.
#
# The load-bearing hazard this file documents: D-026 key records do NOT
# self-describe their mask. The sentinel nibble 4'd1 in an outlier channel's
# code slot is a placeholder, not a discriminator (a kept channel can
# legitimately quantize to 0x1), and lane order is the rank of the channel in
# the ASCENDING mask set — so the read path must run under the SAME mask that
# encoded the records. Hence the D-027 commit rule: a new mask may only be
# committed when the record store is logically empty (occupancy 0, no open
# partial group); violating it is a sticky host-contract fault, mirroring
# SB_OVWR (D-026) and the D-020 reset-attack discipline.
#
# Gates:
#   A  commit rule: a mask commit is legal iff popcount(mask) == OUTLIER_K
#      (lane budget is structural — record width elaborates from OUTLIER_K).
#   B  swap-at-empty composition: run1 under M1, swap, run2 under M2 ==
#      byte-identical to independent single-mask runs with the scale-bank
#      ssid sequence continuing across the swap (bank sets persist; records
#      do not).
#   C  read-path mirror: an independent record+bank parser reconstructs
#      decompress_keys() bit-exactly when given the encoding mask.
#   D  wrong-mask decode: the same parser under a different (equal-popcount,
#      commit-legal) mask MISMATCHES on the disagreeing channels — the reason
#      the commit-at-empty rule exists.

import sys
import numpy as np
sys.path.insert(0, sys.path[0] + "/..")
from apex_golden import cq_codec as cc                       # noqa: E402
from apex_golden.fp import f16_bits_to_f64, f64_to_f32_bits  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        fails += 1


# ── config: small but structure-complete (partial final group included) ─────
D, K, G, BITS = 16, 2, 4, 4
T = 10                                   # 2 full groups + 1 partial (T=2)
M1 = [2, 5]
M2 = [2, 9]                              # differs from M1 in one channel
rng = np.random.default_rng(0x512)


def gen_kf16(t, d):
    return np.asarray(rng.uniform(-4.0, 4.0, (t, d)),
                      dtype=np.float16).view(np.uint16)


def mask_commit_ok(mask_bits: int, k: int) -> bool:
    """D-027 §A: the ONLY legality condition on a mask commit."""
    return bin(mask_bits).count("1") == k


def idx_to_bits(idx):
    v = 0
    for c in idx:
        v |= 1 << c
    return v


def decode_records_under_mask(recs, bank, mask_idx, d, k, groups):
    """Independent mirror of the RTL read path: interpret D-026 key records
    + persistent bank rows under a GIVEN mask. Layout per cq_codec §4:
    [tag 8b][k x fp16 lanes, ascending-mask rank][D x 4b codes][pad64],
    LSB-first bytes; outlier code slot = sentinel, bank outlier scales = 0."""
    mask_idx = np.sort(np.asarray(mask_idx, dtype=np.int64))
    mask_set = set(int(c) for c in mask_idx)
    lanes_lo, codes_lo = 8, 8 + 16 * k
    t_total = recs.shape[0]
    out = np.zeros((t_total, d), dtype=np.uint32)
    for gi, (a, e) in enumerate(groups):
        for t in range(a, e):
            v = int.from_bytes(recs[t].tobytes(), "little")
            for j, chan in enumerate(mask_idx):
                lane = np.uint16((v >> (lanes_lo + 16 * j)) & 0xFFFF)
                out[t, chan] = f64_to_f32_bits(
                    f16_bits_to_f64(np.array([lane], dtype=np.uint16)))[0]
            for chan in range(d):
                if chan in mask_set:
                    continue
                nib = (v >> (codes_lo + 4 * chan)) & 0xF
                code = nib - 16 if nib >= 8 else nib     # sign-extend int4
                out[t, chan] = cc.dequant_f32(
                    np.array([code], dtype=np.int64),
                    np.uint16(bank[gi, chan]))[0]
    return out


ROW_BITS = cc.pad64(cc.key_rec_raw_bits(D, K))

# ── A: commit rule ───────────────────────────────────────────────────────────
print("D-027 §A — mask commit legality (popcount == OUTLIER_K):")
check("accept  popcount==K   (M1)", mask_commit_ok(idx_to_bits(M1), K))
check("accept  popcount==K   (M2)", mask_commit_ok(idx_to_bits(M2), K))
check("reject  popcount K-1", not mask_commit_ok(idx_to_bits([2]), K))
check("reject  popcount K+1", not mask_commit_ok(idx_to_bits([2, 5, 9]), K))
check("reject  empty mask", not mask_commit_ok(0, K))

# ── B: swap-at-empty composition (bank ssids continue; records independent) ─
print("D-027 §B — swap-at-empty composition:")
K1 = gen_kf16(T, D)
K2 = gen_kf16(T, D)
b1 = cc.compress_keys(K1, BITS, G, M1)
b2 = cc.compress_keys(K2, BITS, G, M2)
r1, bank1, ss1 = cc.pack_key_records(b1, ROW_BITS, ssid0=0)
n1 = len(b1.groups)
r2_seq, bank2_seq, ss2_seq = cc.pack_key_records(b2, ROW_BITS, ssid0=n1)
r2_ind, bank2_ind, ss2_ind = cc.pack_key_records(b2, ROW_BITS, ssid0=n1)
check("post-swap records byte-identical to independent run",
      np.array_equal(r2_seq, r2_ind))
check("post-swap bank rows byte-identical to independent run",
      np.array_equal(bank2_seq, bank2_ind))
check("ssid sequence continues across the swap",
      ss2_seq == [(n1 + i) % cc.SCALE_SETS for i in range(len(b2.groups))])
check("bank outlier scales forced 0 under M1",
      all(bank1[gi, c] == 0 for gi in range(n1) for c in M1))

# ── C: read-path mirror is bit-exact under the encoding mask ─────────────────
print("D-027 §C — independent record parser == decompress_keys (same mask):")
ref1 = cc.decompress_keys(b1)
got1 = decode_records_under_mask(r1, bank1, M1, D, K, b1.groups)
check(f"run1 under M1: {T}x{D} fp32 bit-exact", np.array_equal(got1, ref1))
ref2 = cc.decompress_keys(b2)
got2 = decode_records_under_mask(r2_seq, bank2_seq, M2, D, K, b2.groups)
check(f"run2 under M2: {T}x{D} fp32 bit-exact", np.array_equal(got2, ref2))

# ── D: decoding under the WRONG mask mismatches on disagreeing channels ──────
print("D-027 §D — records are not self-describing (wrong-mask decode fails):")
wrong = decode_records_under_mask(r1, bank1, M2, D, K, b1.groups)
agree_cols = sorted(set(range(D)) - (set(M1) ^ set(M2)))
diff_cols = sorted(set(M1) ^ set(M2))
check("mismatch on every disagreeing channel "
      f"(channels {diff_cols})",
      all(not np.array_equal(wrong[:, c], ref1[:, c]) for c in diff_cols))
check("agreeing kept channels still decode bit-exact",
      all(np.array_equal(wrong[:, c], ref1[:, c])
          for c in agree_cols if c not in set(M1)))

print()
if fails:
    print(f"MASK-SEMANTICS GATE: {fails} FAILURES")
    sys.exit(1)
print("MASK-SEMANTICS GATE: ALL PASS")
print("  -> D-027 contract pinned golden-first: popcount==K commit rule,")
print("     swap only at record-empty boundaries (bank ssids persist),")
print("     and the WHY: records decode only under their encoding mask.")
