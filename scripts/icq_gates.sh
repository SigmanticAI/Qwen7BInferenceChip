#!/bin/bash
# icq_gates.sh — IC-QPATH lane gate runner (baseline + post-edit).
#
#   ./scripts/icq_gates.sh <tag>
#
# Runs the byte-identity gate set with an EXPLICIT rc per target into
# build/icq/<tag>/ and prints a summary table. Every target is gated on the
# machine rule (one Verilator build at a time across lanes).
#
# pipefail is load-bearing: the tee lesson.
set -u -o pipefail

TAG="${1:?usage: icq_gates.sh <tag>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/build/icq/$TAG"
mkdir -p "$OUT"
SUM="$OUT/summary.txt"
: > "$SUM"

wait_verilator() {
  until ! pgrep -f '[v]erilator_bin'; do sleep 30; done
}

run() {
  local name="$1"; shift
  wait_verilator
  echo "=== $name ==="
  ( "$@" ) > "$OUT/$name.log" 2>&1
  local rc=$?
  printf '%-28s rc=%d\n' "$name" "$rc" | tee -a "$SUM"
  return 0
}

cd "$REPO"

run golden            make -C golden test
run l3_all            make -C verif/top/l3 all
run l4_levels         make -C verif/top/l4 levels
run l4_compose        make -C verif/top/l4 compose
run l4_mutate         make -C verif/top/l4 mutate
run kvq_gqa           make -C verif/kvq/gqa all
run kvq_mask          make -C verif/kvq/mask all
run seq_walker        make -C verif/seq_walker all
run wcomp             make -C verif/top/wcomp all

echo
echo "── summary ($TAG) ──"
cat "$SUM"
