#!/bin/bash
# s8_artifact_run.sh — S8 GOLDEN-7B-TOKEN artifact run, quiet-gated.
#
# Streams a 150-token Qwen2.5-7B greedy generation through the golden
# fixed-point pipeline (run_tinynpu.py), long enough to CROSS the T=128
# chunk boundary (C-CHUNK per-chunk tile jobs past step 128), with sampled
# RTL-replayable job traces and float64-yardstick probes at steps
# {0,1,2,50,100,150}; then verifies the whole trace bit-exact.
#
# QUIET GATE (one heavy job per machine): refuses to start until QUIET_MIN consecutive
# minutes with zero EDA-ish processes — model runs and EDA jobs are
# serialized on this one 18 GB machine.
# Intended launch: caffeinate -ims bash scripts/s8_artifact_run.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/apex-eval/bin/python"
OUT=docs/results/s8_7b_token
LOG="$OUT/artifact_run.log"
QUIET_MIN="${QUIET_MIN:-10}"
MAX_NEW="${MAX_NEW:-150}"
PROMPT=$'Here are five interesting facts about the Moon:\n1.'
mkdir -p "$OUT"

if [ ! -f build/s8_weights/Qwen2.5-7B-4bit/meta.json ]; then
  echo "ABORT: no weight cache — run: run_tinynpu.py --prepare" | tee -a "$LOG"
  exit 1
fi

echo "=== S8 ARTIFACT QUIET GATE start $(date) (need ${QUIET_MIN} consecutive quiet minutes) ===" | tee -a "$LOG"
quiet=0
while [ "$quiet" -lt "$QUIET_MIN" ]; do
  n=$(ps aux | grep -cE "[v]erilator|[o]bj_dir/V|[i]verilog|[v]vp |[y]osys|[n]extpnr")
  if [ "$n" -eq 0 ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "quiet-gate: eda_procs=$n consecutive_quiet=${quiet}/${QUIET_MIN} $(date)" | tee -a "$LOG"
  [ "$quiet" -lt "$QUIET_MIN" ] && sleep 60
done

echo "=== S8 ARTIFACT RUN START $(date) (max_new=${MAX_NEW}) ===" | tee -a "$LOG"
"$PY" run_tinynpu.py --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" \
  --tier kvq8 --trace-dir "$OUT/artifact_trace" --trace-jobs 16 \
  --ref-check 3 --ref-every 50 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== RUN EXIT $rc $(date) ===" | tee -a "$LOG"
[ "$rc" -ne 0 ] && exit "$rc"

echo "=== TRACE VERIFY ===" | tee -a "$LOG"
"$PY" run_tinynpu.py --verify-trace "$OUT/artifact_trace" 2>&1 | tee -a "$LOG"
vrc=${PIPESTATUS[0]}
echo "=== S8 ARTIFACT DONE verify_rc=$vrc $(date) ===" | tee -a "$LOG"
exit "$vrc"
