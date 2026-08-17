#!/usr/bin/env bash
# walk_smoke.sh — the walker-mode gate, in the order that makes the green
# result mean something. Everything here runs on the Verilated silicon twin
# (verif/f2sim/obj_d128_ddr0/f2sim); nothing touches build/f2_regops.
#
#   1. host-only selftest       — the compiler's own invariants (16 checks)
#   2. WALK-window probe        — FMT_SUP reads back; the A-1 refusal gate
#                                 (tier CQ-4 / T=0 / fmt=2) refuses with the
#                                 documented code and the sticky W1C-clears
#   3. one real 7B head, walked — T=20 npz, two-pass rq (non-circular),
#                                 WITH the negative controls: the same program
#                                 with the kick neutered MUST go red
#   4. all 18 trace jobs        — the walk is not a one-off (T=13..128)
#
# Usage:  scripts/fpga/f2/walk_smoke.sh [OUTDIR]
set -o pipefail
set -e
cd "$(dirname "$0")/../../.."          # repo root
OUT="${1:-build/f2_walk}"
PY="${PYTHON:-python3}"
W=scripts/fpga/f2/walk_job.py
NPZ=docs/results/s8_7b_token/artifact_trace/job_s019_L19_h03.npz

echo "=== 1/4 host-only selftest ============================================"
"$PY" "$W" --selftest

echo "=== 2/4 WALK-window + A-1 refusal-gate probe (on the twin) ==========="
"$PY" "$W" --probe-only --out-dir "$OUT"

echo "=== 3/4 one real 7B head walked, with negative controls ==============="
# --rekick also MEASURES the per-step WALK cost (B1_WALKER.md §3 could only
# derive it) by walking pv a second time off the already-loaded descriptor.
"$PY" "$W" --npz "$NPZ" --rq tile --rekick --out-dir "$OUT" \
     --cap-out "$OUT/job_s019_L19_h03.walk.caps.jsonl"

echo "=== 4/4 every trace job, walker mode =================================="
fail=0
for f in docs/results/s8_7b_token/artifact_trace/job_*.npz; do
  n=$(basename "$f" .npz)
  log="$OUT/sweep/$n.walk.log"
  mkdir -p "$OUT/sweep"
  # own subdir: the sweep must not clobber step 3's manifest/captures
  if "$PY" "$W" --npz "$f" --rq trace --no-controls --out-dir "$OUT/sweep" \
        > "$log" 2>&1; then
    printf '  %-24s %s | %s\n' "$n" \
      "$(grep -o 'WALKER-MODE SMOKE: [A-Z]*' "$log")" \
      "$(grep 'reduction on the walked' "$log" | sed 's/.*: *//')"
  else
    printf '  %-24s FAIL (see %s)\n' "$n" "$log"; fail=$((fail + 1))
  fi
done
echo "----------------------------------------------------------------------"
if [ "$fail" -ne 0 ]; then
  echo "WALKER-MODE GATE: FAIL ($fail job(s) red)"; exit 1
fi
echo "WALKER-MODE GATE: PASS (selftest + refusal gate + controls + 18 jobs)"
