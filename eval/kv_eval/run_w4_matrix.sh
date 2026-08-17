#!/bin/bash
# run_w4_matrix.sh — B3 stage-6 ADOPTION GATE matrix, serialized (one 18 GB
# machine). Qwen2.5-7B x weight tiers {w-int8, w4-tile-a, w4-g32-b, w4-g16-b}
# with identity KV, paired against the committed S5 base samples.
#
# QUIET GATE: same policy as S5's run_matrix_7b.sh — refuses to load the
# model until QUIET_MIN consecutive EDA-quiet minutes, then runs the n=100
# harness-equivalence preflight vs the committed S5 base samples, then the
# matrix. Intended launch (daemonized per the 07-17 lesson; the mkdir MUST
# precede the redirect — the target dir does not exist on a fresh checkout
# and the forked child dies on the redirect before this script ever runs):
#   mkdir -p docs/results/b3_w4_adoption && \
#   ( nohup caffeinate -ims bash eval/kv_eval/run_w4_matrix.sh \
#       > docs/results/b3_w4_adoption/nohup.out 2>&1 & )
# (subshell double-fork orphans the job — macOS has no setsid; same pattern
#  as scripts/run_full_matrix.sh)
set -uo pipefail
cd "$(dirname "$0")/../.."
source ~/.venvs/apex-eval/bin/activate
OUT=docs/results/b3_w4_adoption
mkdir -p "$OUT"
LOG="$OUT/matrix_run.log"
MODEL=mlx-community/Qwen2.5-7B-4bit
LIMIT="${LIMIT:-1000}"
QUIET_MIN="${QUIET_MIN:-10}"
TIERS="${TIERS:-w-int8 w4-tile-a w4-g32-b w4-g16-b}"
BASE10K=docs/results/s5_eval7b/Qwen2.5-7B-4bit_base_n10042.samples.json

echo "=== B3-W4 QUIET GATE start $(date) (need ${QUIET_MIN} quiet minutes) ===" | tee -a "$LOG"
quiet=0
while [ "$quiet" -lt "$QUIET_MIN" ]; do
  n=$(ps aux | grep -cE "[v]erilator|[o]bj_dir/V|[i]verilog|[v]vp |[y]osys|[n]extpnr")
  if [ "$n" -eq 0 ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "quiet-gate: eda_procs=$n consecutive_quiet=${quiet}/${QUIET_MIN} $(date)" | tee -a "$LOG"
  [ "$quiet" -lt "$QUIET_MIN" ] && sleep 60
done

echo "=== machine quiet $(date); n=100 harness-equivalence preflight ===" | tee -a "$LOG"
python eval/kv_eval/run_hellaswag_w4.py --wtier base --limit 100 \
    --preflight-against "$BASE10K" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== ABORT: preflight failed $(date) ===" | tee -a "$LOG"
  exit 1
fi

echo "=== B3-W4 MATRIX START $(date) (LIMIT=$LIMIT TIERS=$TIERS) ===" | tee -a "$LOG"
for wtier in $TIERS; do
  name="Qwen2.5-7B-4bit_${wtier}_n${LIMIT}"
  if [ -f "$OUT/${name}.json" ]; then
    echo "=== skip $wtier (exists) ===" | tee -a "$LOG"
    continue
  fi
  echo "=== tier $wtier start $(date) ===" | tee -a "$LOG"
  python eval/kv_eval/run_hellaswag_w4.py --wtier "$wtier" --limit "$LIMIT" \
      2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "=== ABORT: tier $wtier failed $(date) ===" | tee -a "$LOG"
    exit 1
  fi
  echo "=== tier $wtier done $(date) ===" | tee -a "$LOG"
done

echo "=== PAIRED STATS vs committed S5 base $(date) ===" | tee -a "$LOG"
python eval/kv_eval/w4_paired_stats.py --base "$BASE10K" \
    "$OUT"/Qwen2.5-7B-4bit_*_n${LIMIT}.samples.json 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== ABORT: paired-stats failed $(date) — matrix is NOT done ===" | tee -a "$LOG"
  exit 1
fi
echo "=== B3-W4 MATRIX DONE $(date) ===" | tee -a "$LOG"
exit 0
