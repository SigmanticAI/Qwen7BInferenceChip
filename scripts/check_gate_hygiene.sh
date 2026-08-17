#!/bin/bash
# check_gate_hygiene.sh — make the gate-integrity invariants MECHANICAL.
#
# Three defect classes have each shipped in this repo, survived review, and
# been found only by running on a second machine. All three share a shape: the
# gate still prints its banner and exits 0 while checking nothing.
#
#   A. `cmd | tee log` with no pipefail  -> the recipe's status is tee's, so a
#      failing checker passes. (The original "lying gate".)
#   B. `.SHELLFLAGS := -o pipefail -c` as the ONLY protection -> requires GNU
#      make >= 3.82. Stock macOS ships 3.81, which SILENTLY IGNORES it, so the
#      gate is inert on the machine where results are certified and fine on
#      Linux CI — which is why it hid for so long.
#   C. `set -o pipefail` with no `SHELL := /bin/bash` -> make falls back to
#      /bin/sh. On macOS that is bash-in-POSIX-mode and pipefail works; on
#      Ubuntu it is dash, where pipefail does not exist. Invisible locally,
#      silently disabled on Linux. This one DEGRADES rather than errors.
#
# Enforcing these by memory is how they got here. Run this instead:
#     scripts/check_gate_hygiene.sh          # exits 1 on any violation
#
# Note: including verif/tools.mk supplies SHELL, so files that include it
# satisfy (C) structurally.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
note() { printf '  %-58s %s\n' "$1" "$2"; fail=1; }

echo "== (C) recipes using pipefail without a bash SHELL =="
while IFS= read -r f; do
  grep -q 'pipefail' "$f" || continue
  grep -qE '^SHELL[[:space:]]*:?=' "$f" && continue
  grep -qE '^include .*verif/tools\.mk' "$f" && continue   # inherits SHELL
  note "$f" "pipefail but no SHELL := /bin/bash (dash on Linux -> inert)"
done < <(git ls-files '*Makefile' '*.mk')
[ "$fail" = 0 ] && echo "  none"

echo "== (B) .SHELLFLAGS as the ONLY protection on a piped gate recipe =="
b=0
while IFS= read -r f; do
  grep -qE '^\.SHELLFLAGS' "$f" || continue
  # Test LOGICAL recipe lines: a backslash-continued recipe is ONE shell
  # invocation, so `set -o pipefail; \` on its first physical line protects the
  # pipe several lines below. Checking physical lines reports those as
  # violations — a checker that cries wolf is itself a bad gate, so join first.
  if awk '
      /^\t/ {
        line = line $0
        if (sub(/\\$/, "", line)) next          # keep accumulating
        # Principled exclusions, so the checker does not cry wolf:
        #  * `echo ... | tee`  — echo cannot meaningfully fail; pipefail is moot.
        #  * `... || { ...; exit 1; }` / `|| exit` — failure is explicitly handled,
        #    and for `grep -q X || exit 1` the content check governs the status.
        if (line ~ /\|/ && line !~ /set -o pipefail/ &&
            line !~ /\|\| *(\{|exit)/ && line !~ /@?echo .*\| *tee/ &&
            line ~ /coverage_report|mutation|mutate|_check\.py|check\.py|gen_[a-z_]*\.py/) found=1
        line = ""; next
      }
      { line = "" }
      END { exit !found }' "$f"; then
    note "$f" "piped gate relies on .SHELLFLAGS (inert under macOS make 3.81)"; b=1
  fi
done < <(git ls-files '*Makefile' '*.mk')
[ "$b" = 0 ] && echo "  none"

echo "== (D) hardcoded absolute tool paths (machine-specific) =="
d=0
while IFS= read -r f; do
  [ "$f" = "verif/tools.mk" ] && continue
  [ "$f" = "scripts/check_gate_hygiene.sh" ] && continue
  if grep -qE '^[^#]*(/opt/homebrew/bin|/usr/local/Cellar)/' "$f"; then
    note "$f" "hardcoded tool path — use verif/tools.mk / \$VAR"; d=1
  fi
done < <(git ls-files '*Makefile' '*.mk' '*.py' '*.sh')
[ "$d" = 0 ] && echo "  none"

echo
if [ "$fail" = 0 ]; then echo "GATE HYGIENE: clean"; else echo "GATE HYGIENE: violations above"; fi
exit "$fail"
