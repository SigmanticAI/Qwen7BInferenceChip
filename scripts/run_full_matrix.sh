#!/bin/bash
# run_full_matrix.sh — the G4 gate: fresh re-run of every evidence suite,
# then STATUS.md regeneration (gen_status.py refuses stale logs by design).
# One 18 GB machine: strictly serial; run caffeinated + daemonized:
#   ( nohup caffeinate -ims bash scripts/run_full_matrix.sh \
#       > build/full_matrix_run.log 2>&1 & )
set -uo pipefail
cd "$(dirname "$0")/.."
# suite Makefiles default PYTHON ?= python3 (public clones); this machine's
# numpy-equipped interpreter lives in the apex-eval venv:
[ -x "$HOME/.venvs/apex-eval/bin/python" ] && \
  export PYTHON="$HOME/.venvs/apex-eval/bin/python"
fail=0
run() { echo "== $* START $(date)"; if ! "$@"; then echo "== $* FAILED"; fail=1; fi; echo "== $* END $(date)"; }

run make -C golden test
run python3 perf/apex_perf_model.py --check

SUITES="verif/asu/sb verif/asu/silu verif/csr verif/audit_csr verif/audit_seq \
verif/kvq/audit verif/kvq/fparith verif/kvq/sb verif/kvq/smoke verif/layer \
verif/mxe/perf verif/mxe/sb verif/mxe/struct verif/residual verif/rope/smoke \
verif/seam verif/seq verif/tip verif/top/l2 verif/top/l3 verif/top/smoke"
for d in $SUITES; do
  if grep -qE '^all:' "$d/Makefile" 2>/dev/null; then run make -C "$d" all
  else run make -C "$d"; fi
done

# S8 real-model replay (durable logs refreshed from the committed trace)
run make -C verif/kvq/replay gen NAME=artifact \
    TRACE="$PWD/docs/results/s8_7b_token/artifact_trace"
run make -C verif/kvq/replay run NAME=artifact

if [ "$fail" -eq 0 ]; then
  run python3 scripts/gen_status.py
else
  echo "== SKIPPING gen_status.py — $fail suite group(s) FAILED"
fi
echo "== FULL MATRIX DONE fail=$fail $(date)"
exit "$fail"
