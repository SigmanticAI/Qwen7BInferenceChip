# =============================================================================
# verif/tools.mk — simulator/tool resolution + the version pin.
#
# WHY THIS EXISTS (clean-machine audit, 2026-07-27):
# 52 tracked build files hardcoded macOS-Homebrew absolute tool paths
# (/opt/homebrew/bin/{verilator,vvp,sv2v,iverilog,yosys}). That made every
# suite unrunnable on any machine but the author's Mac — our own Linux verify
# boxes only worked because their bootstrap symlinks a FAKE /opt/homebrew path.
# That crutch got written into the runbook (LEVEL_C_INTEGRATION §6) instead of
# the cause being fixed, so it survived until the suites were first run on a
# clean machine.
#
# The subtlety, and why this file does more than swap in PATH lookups:
# the hardcoded path was ALSO — accidentally — the project's only version pin.
# Nothing in the tree checks the simulator version; the path simply could not
# resolve to anything but the author's Homebrew Verilator 5.044. A bare
# `VERILATOR ?= verilator` would therefore delete the pin and silently accept
# whatever is on PATH, and Verilator 5.020 is known to miscompile the tile
# build (NBA-in-loop in sram_controller). A suite that runs green on the wrong
# simulator is a lying gate — so restoring portability REQUIRES making the pin
# explicit and loud, not dropping it.
#
# Usage in a suite Makefile (REPO must already be defined):
#     include $(REPO)/verif/tools.mk
# Suites that also drive Icarus add, before the include:
#     NEED_IVERILOG := 1
#
# Escape hatches:
#   non-PATH install     make VERILATOR=/opt/verilator-5.044/bin/verilator
#   deliberate other ver make VERILATOR_REQ=5.045
# =============================================================================

# SHELL is set HERE, not left to each suite, because forgetting it is invisible
# on macOS and silently disabling on Linux: /bin/sh is bash-in-POSIX-mode on a
# Mac (so `set -o pipefail` works and the gates bite) but DASH on Ubuntu, where
# pipefail does not exist — the gate degrades instead of erroring, so it looks
# green while checking nothing. Inheriting it through this include makes the
# invariant structural instead of forty files each remembering it. A suite may
# still override SHELL after the include if it ever needs to.
SHELL         := /bin/bash

VERILATOR     ?= verilator
IVERILOG      ?= iverilog
VVP           ?= vvp
SV2V          ?= sv2v
YOSYS         ?= yosys

# Pinned in ARCHITECTURE.md. The two pins are deliberately DIFFERENT in kind:
#   Verilator is the primary execution engine — an unexpected version silently
#   changes what gets built (5.020 miscompiles the tile), so it is EXACT.
#   Icarus is the secondary cross-check, and its job is INDEPENDENCE: more
#   versions agreeing is stronger corroboration, not weaker. It is a FLOOR.
#   (Owner decision, recorded in ARCHITECTURE.md, on measured evidence: Icarus
#   12.0 — the apt default on Ubuntu 24.04 — passes asu/silu 65536/65536,
#   rope/smoke 12288/12288 and residual 20012/20012 bit-exact off-Mac. An exact
#   13.0 pin would force every external reproducer to build Icarus from source
#   for no measured benefit, and the risk is bounded because the suites compare
#   against the golden model: a disagreeing version FAILS, it cannot pass
#   silently.)
VERILATOR_REQ ?= 5.044
IVERILOG_MIN  ?= 12.0

# The assertions run at PARSE time rather than as a `check-tools` prerequisite:
# that way they fire on every invocation of every target without wiring a
# prerequisite into each suite, and they cannot be bypassed by invoking a
# sub-target directly. Goals that need no simulator are exempt so `make clean`
# still works on a machine with no tools installed.
#
# Dry runs (`make -n`) must also work toolless — they inspect the build without
# executing it. MAKEFLAGS holds the short-flag cluster as its first word, but
# ONLY when flags were passed; with a bare `make VAR=value` the first word is
# the assignment itself, and a value like /nonexistent would false-positive a
# naive `findstring n`. So the first word is consulted only when it is a flag
# cluster (contains no '=').
_MK_FIRST := $(firstword $(MAKEFLAGS))
ifeq ($(findstring =,$(_MK_FIRST)),)
_MK_DRYRUN := $(findstring n,$(_MK_FIRST))
else
_MK_DRYRUN :=
endif

ifeq ($(strip $(_MK_DRYRUN)$(filter clean help print-%,$(MAKECMDGOALS))),)

_VL_VER := $(shell $(VERILATOR) --version 2>/dev/null | awk '{print $$2}')

ifeq ($(_VL_VER),)
$(error Verilator not found as '$(VERILATOR)'. Install Verilator $(VERILATOR_REQ) on PATH, or point at it: make VERILATOR=/path/to/verilator)
endif

ifneq ($(_VL_VER),$(VERILATOR_REQ))
$(error Verilator version mismatch: '$(VERILATOR)' reports $(_VL_VER), this tree pins $(VERILATOR_REQ) (ARCHITECTURE.md). 5.020 is known to miscompile the tile build, so this is refused rather than warned. Use: make VERILATOR=/path/to/$(VERILATOR_REQ)/verilator -- or, deliberately: make VERILATOR_REQ=$(_VL_VER))
endif

ifdef NEED_IVERILOG
_IV_VER := $(shell $(IVERILOG) -V 2>/dev/null | head -1 | awk '{print $$4}')

ifeq ($(_IV_VER),)
$(error Icarus Verilog not found as '$(IVERILOG)'. Install Icarus >= $(IVERILOG_MIN), or point at it: make IVERILOG=/path/to/iverilog)
endif

# Floor comparison, dotted-numeric, no dependency on GNU sort -V (BSD/macOS
# sort lacks it on older releases).
_IV_OK := $(shell awk -v a='$(_IV_VER)' -v b='$(IVERILOG_MIN)' 'BEGIN{ \
    n=split(a,x,"."); split(b,y,"."); \
    for(i=1;i<=3;i++){ xi=x[i]+0; yi=y[i]+0; \
      if(xi>yi){print "ok"; exit} if(xi<yi){exit} } print "ok" }')

ifneq ($(_IV_OK),ok)
$(error Icarus too old: '$(IVERILOG)' reports $(_IV_VER), this tree requires >= $(IVERILOG_MIN) (ARCHITECTURE.md). Install a newer Icarus, or point at one: make IVERILOG=/path/to/iverilog)
endif
endif

endif
