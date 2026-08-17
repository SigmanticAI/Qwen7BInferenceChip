#!/bin/bash
# run_matrix.sh — S4 full head-to-head matrix, serialized (one 18 GB machine).
# 2 models x {fp16, kvq8, kvq4, kvq4p}; LIMIT=n docs per run (default 1000 —
# the n used by published ChannelQuant-family results; LIMIT=10042 = the full
# HellaSwag validation set required before public use). Skip check keys on
# the SAME LIMIT, so full runs never skip because n=1000 results exist.
# Each run re-certifies the twin gate first (enforced inside run_hellaswag.py).
set -uo pipefail
cd "$(dirname "$0")/../.."
source ~/.venvs/apex-eval/bin/activate
LIMIT="${LIMIT:-1000}"
OUT=docs/results/s4_head2head
mkdir -p "$OUT"
LOG="$OUT/matrix_run.log"
# The log is APPEND-accumulated: remember where THIS run starts so the final
# summary/exit status can never quote stale RESULT lines from a prior run.
RUN_START_LINE=$(wc -l < "$LOG" 2>/dev/null || echo 0)
echo "=== S4 MATRIX START (LIMIT=$LIMIT) $(date) ===" | tee -a "$LOG"
ran_any=0
for model in Qwen/Qwen2-0.5B Qwen/Qwen2-1.5B; do
  for tier in fp16 kvq8 kvq4 kvq4p; do
    name="$(basename $model)_$tier"
    if [ -f "$OUT/$(basename $model)_${tier}_n${LIMIT}.json" ]; then
      echo "--- skip $name (result exists)" | tee -a "$LOG"
      continue
    fi
    echo "--- $name START $(date)" | tee -a "$LOG"
    ran_any=1
    python eval/kv_eval/run_hellaswag.py --model "$model" --tier "$tier" \
      --limit "$LIMIT" --outdir "$OUT" 2>&1 | grep -vE "%\|" | tee -a "$LOG" | tail -2
    rc=${PIPESTATUS[0]}
    echo "--- $name EXIT $rc $(date)" | tee -a "$LOG"
    if [ "$rc" -ne 0 ]; then
      echo "TIER $name FAILED rc=$rc — ABORTING MATRIX (a failed rerun must not exit 0 on stale RESULT lines)" | tee -a "$LOG"
      exit "$rc"
    fi
  done
done
echo "=== S4 MATRIX DONE (LIMIT=$LIMIT) $(date) ===" | tee -a "$LOG"
# Summary restricted to THIS run's slice of the accumulated log: a plain
# grep over $LOG would print RESULT lines from prior runs and set the
# script's exit status from them.
new_results=$(tail -n +"$((RUN_START_LINE + 1))" "$LOG" | grep -c "^RESULT" || true)
tail -n +"$((RUN_START_LINE + 1))" "$LOG" | grep "^RESULT" || true
if [ "$ran_any" -eq 1 ] && [ "$new_results" -eq 0 ]; then
  echo "=== MATRIX ERROR: tiers ran but THIS run produced no RESULT lines ===" | tee -a "$LOG"
  exit 1
fi
exit 0
