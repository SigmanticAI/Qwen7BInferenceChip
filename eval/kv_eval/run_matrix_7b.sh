#!/bin/bash
# run_matrix_7b.sh — S5 EVAL-7B matrix, serialized (one 18 GB machine).
# Qwen2.5-7B (4-bit MLX weights) x {base, kvq8, kvq4} at n=1000.
# Each run re-certifies the twin gate first (enforced inside run_hellaswag_mlx.py).
#
# QUIET GATE: another session runs Verilator regressions on this machine;
# model evals and EDA jobs swap-kill each other on one machine. This script
# refuses to load the model until it has seen QUIET_MIN consecutive minutes
# with zero EDA-ish processes, then runs the identity pre-flight, then the
# matrix. Intended launch: caffeinate -ims bash eval/kv_eval/run_matrix_7b.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source ~/.venvs/apex-eval/bin/activate
OUT=docs/results/s5_eval7b
mkdir -p "$OUT"
LOG="$OUT/matrix_run.log"
# audit#2: the log is APPEND-accumulated and git-committed, so a summary grep
# over the whole file reports PRIOR runs' RESULT lines as if they were this
# invocation's. Record the pre-run length and scope the summary to our own tail.
RUN_START_LINE=$(wc -l < "$LOG" 2>/dev/null || echo 0)
ran_any=0
MODEL=mlx-community/Qwen2.5-7B-4bit
LIMIT="${LIMIT:-1000}"
QUIET_MIN="${QUIET_MIN:-10}"

echo "=== S5 QUIET GATE start $(date) (need ${QUIET_MIN} consecutive quiet minutes) ===" | tee -a "$LOG"
quiet=0
while [ "$quiet" -lt "$QUIET_MIN" ]; do
  n=$(ps aux | grep -cE "[v]erilator|[o]bj_dir/V|[i]verilog|[v]vp |[y]osys|[n]extpnr")
  if [ "$n" -eq 0 ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "quiet-gate: eda_procs=$n consecutive_quiet=${quiet}/${QUIET_MIN} $(date)" | tee -a "$LOG"
  [ "$quiet" -lt "$QUIET_MIN" ] && sleep 60
done
echo "=== machine quiet $(date); running identity pre-flight ===" | tee -a "$LOG"

python eval/kv_eval/test_mlx_identity.py --model "$MODEL" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== ABORT: identity pre-flight failed $(date) ===" | tee -a "$LOG"
  exit 1
fi

echo "=== S5 MATRIX START $(date) ===" | tee -a "$LOG"
skipped=0
for tier in base kvq8 kvq4; do
  name="Qwen2.5-7B-4bit_$tier"
  if [ -f "$OUT/${name}_n${LIMIT}.json" ]; then
    echo "--- skip $name (result exists)" | tee -a "$LOG"
    skipped=$((skipped+1))
    continue
  fi
  echo "--- $name START $(date)" | tee -a "$LOG"
  python eval/kv_eval/run_hellaswag_mlx.py --model "$MODEL" --tier "$tier" \
    --limit "$LIMIT" --outdir "$OUT" 2>&1 | grep -vE "%\|" | tee -a "$LOG" | tail -2
  rc=${PIPESTATUS[0]}
  ran_any=$((ran_any+1))
  echo "--- $name EXIT $rc $(date)" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "TIER $name FAILED rc=$rc — ABORTING MATRIX (audit#2: a failed rerun must not exit 0 on stale RESULT lines)" | tee -a "$LOG"
    exit "$rc"
  fi
done
echo "=== S5 MATRIX DONE $(date) ===" | tee -a "$LOG"
echo "TIERS EXECUTED THIS INVOCATION: $ran_any   SKIPPED (result already present): $skipped"
if [ "$ran_any" -eq 0 ]; then
  echo "NO TIER EXECUTED — this invocation computed NOTHING. The RESULT lines in" >&2
  echo "$LOG are from PRIOR runs; delete $OUT/*_n${LIMIT}.json to force a real run." >&2
  exit 3
fi
# summary scoped to THIS invocation only (never prior runs' lines)
tail -n +$((RUN_START_LINE + 1)) "$LOG" | grep "^RESULT"
