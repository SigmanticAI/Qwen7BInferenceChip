# verif/f2sim — F2 stage-2 regops executor: the verilated cl_apex driven at
# its OCL AXI-Lite pins by the SAME regops files the F2 host runner uses.
# Needs the aws-fpga kit checkout for cl_ports.vh (gitignored; see
# scripts/fpga/f2/BRINGUP.md): make AWS_FPGA=/path/to/aws-fpga
SHELL       := /bin/bash
export CXXFLAGS := -std=gnu++17
.SHELLFLAGS := -o pipefail -c
REPO      := ../..
AWS_FPGA  ?= $(REPO)/tools/aws-fpga
VERILATOR ?= verilator
D         ?= 128

SHELL_INC := $(AWS_FPGA)/hdk/common/shell_stable/design/interfaces
# IB-FUEL s1: the kit's EMPTY sh_ddr.stub.sv is replaced by the behavioral
# sparse model (same module name + port list; DDR_PRESENT=0 -> inert, byte-
# identical to the stub in 2-state sim). Kit checkout stays untouched.
# See docs/design/IB_FUEL.md §1 s1.
SH_DDR    := sh_ddr_beh.sv
CLDIR     := $(REPO)/scripts/fpga/f2/cl_apex/design
SRCS      := $(SH_DDR) \
             $(shell sed -n 's|^[^\#].*|$(REPO)/&|p' $(CLDIR)/apex_sources.f) \
             $(CLDIR)/apex_ocl_cdc.sv $(CLDIR)/apex_pcis_slv.sv \
             $(CLDIR)/apex_afifo.sv $(CLDIR)/apex_fuel_reader.sv \
             $(CLDIR)/apex_fuel_ctl.sv \
             $(CLDIR)/apex_f2_mailbox.sv $(CLDIR)/cl_apex.sv

# DECISION-LC-1 two-clock sim: tile clock = clk_main/(2*TILE_DIV), divider
# in-RTL under APEX_SIM_TILE_DIV, ratio set at RUN time via +tile_div.
# TILE_DIV=5 -> 25 MHz (the dynamic-clkgen plan), 2 -> 62.5 MHz (recipe A0
# on this CL's a1 — "B1" here was a pre-LC-1 fossil, see CLOCK_LADDER.md §2.1),
# 7 -> odd ratio (CDC phase shake). Bit-exactness must hold at ALL ratios.
TILE_DIV  ?= 5

# IB-FUEL s2: DDR=1 builds the CL with DDR_PRESENT=1 (the fuel config the
# sim certifies); DDR=0 builds today's default DCP config (decode idle).
# Host-mode replay counts must be byte-identical under BOTH.
DDR       ?= 1

OBJ := obj_d$(D)_ddr$(DDR)

.PHONY: build run clean behsmoke mutants capgate
# No --timing: cl_apex + apex_top are synthesizable RTL; this C++ harness
# supplies the clock. --timing pulls in the coroutine runtime (teardown
# mutex fault) and is unnecessary for a purely clocked DUT. SVA off too
# (--assert belongs to the RTL's own TBs, not this pin-level driver).
# -Werror-USERERROR: `-Wno-fatal` (needed for the kit's own WIDTH warnings)
# also demotes every elaboration-time $error to a warning — and the `| tail -3`
# below then hides it, so an ILLEGAL BUILD KNOB produced a green build and a
# silently mis-elaborated twin. The elaboration guards (apex_top g_chk_dm,
# cl_apex g_chk_qstage/g_chk_dmodel, asu_rmsnorm g_cfg_err_*) are the only
# thing standing between a typo'd +define and a wrong design, so USERERROR is
# promoted back to a hard error. Verified: +define+APEX_CL_QSTAGE=99 now fails
# the build instead of producing obj_*/f2sim.
build:
	set -o pipefail; $(VERILATOR) --cc --exe --build -j 0 -Wno-fatal -Werror-USERERROR -CFLAGS -std=gnu++17 $(if $(TRACE),--trace --trace-depth 6) \
	  --timescale 1ns/1ps +define+APEX_CL_D=$(D) +define+APEX_SIM_TILE_DIV $(VFLAGS_EXTRA) \
	  +define+APEX_CL_DDR=$(DDR) \
	  -I$(SHELL_INC) -I$(CLDIR) -I$(REPO)/rtl/asu -I$(REPO)/rtl/rope -I$(REPO)/verif/common \
	  --top-module cl_apex --Mdir $(OBJ) \
	  $(SRCS) sim_main.cpp -o f2sim 2>&1 | tail -3

# REGOPS: space-separated regops.jsonl paths. Default = EVERY compiled trace
# job, so `make run` is the real gate; override for a single-job bisect, e.g.
#   make run REGOPS=../../build/f2_regops/job_s019_L19_h03.regops.jsonl
# Regenerate the inputs with: python3 scripts/fpga/f2/trace_to_regops.py
REGOPS ?= $(sort $(wildcard $(REPO)/build/f2_regops/job_*.regops.jsonl))
run: | build
	@test -n "$(REGOPS)" || { echo "no regops: run scripts/fpga/f2/trace_to_regops.py"; exit 1; }
	set -o pipefail; ./$(OBJ)/f2sim +tile_div=$(TILE_DIV) $(REGOPS) 2>&1 | tee run_d$(D)_div$(TILE_DIV).log | tail -20

# IB-FUEL s1 gate (a): directed AXI smoke on the behavioral DDR model alone
# (standalone — no CL, no kit includes). Runs twice: default knobs, then
# +ddr_stall (LFSR ready-backpressure + rvalid gaps) with short latency —
# identical checks must pass both ways.
behsmoke:
	set -o pipefail; $(VERILATOR) --cc --exe --build -j 0 -Wno-fatal -CFLAGS -std=gnu++17 \
	  --timescale 1ns/1ps --top-module sh_ddr --Mdir obj_beh \
	  sh_ddr_beh.sv tb_ddr.cpp -o ddrbeh 2>&1 | tail -2
	set -o pipefail; ./obj_beh/ddrbeh | tee beh_default.log | tail -2
	set -o pipefail; ./obj_beh/ddrbeh +ddr_stall=48879 +ddr_lat=3 | tee beh_stall.log | tail -2

# IB-FUEL s4 RED-mutant gate — pipefail-safe (mutants.sh sets -o pipefail so
# `| tail` cannot mask a surviving mutant). Requires the DDR=1 gate binary
# (obj_d128_ddr1) + the fuel regops/image to exist. See IB_FUEL.md §1.5.
mutants: | build
	AWS_FPGA=$(AWS_FPGA) ./mutants.sh

# STAGE-A READ-BACK GATE — one job, ~2 s, no Verilator rebuild.
# Recompiles ONE trace job with trace_to_regops.py --shadow-cap into a SCRATCH
# dir, replays it through scripts/fpga/f2/tile_exec_bridge.py (so the bridge's
# argv/plusarg/summary contract is tested against the REAL binary, not its
# fake-executor selftest), and asserts every value the tile handed back equals
# the baked expectation for the SAME site — re-derived from the regops file
# here, not read off the executor's own `F2SIM CAPGATE:` self-report.
#
# NOT `| build`, unlike run/mutants: build is .PHONY, so ordering on it
# re-verilates (minutes) every invocation and a 2-second gate nobody runs is
# not a gate. The binary must already exist; if it does not, the script says
# so and fails. Uses the bridge's own resolution order, which prefers the
# DDR=0 silicon twin (obj_d128_ddr0) over this Makefile's DDR=1 default.
#
# Scratch defaults to a fresh temp dir and is REFUSED if it lands inside
# build/f2_regops — shadow-cap output there would silently redefine the
# canonical 18-job gate set (audit N10). Kept on failure, removed on success.
#   make capgate                          # verdict line only
#   make capgate CAPGATE_VERBOSE=1        # + per-step evidence
#   make capgate CAPGATE_JOB=job_s027_L23_h02 CAPGATE_SCRATCH=/tmp/cg
PYTHON          ?= python3
CAPGATE_JOB     ?= job_s019_L19_h03
CAPGATE_SCRATCH ?=
CAPGATE_BIN     ?=
CAPGATE_CLEAN   ?= --clean
capgate:
	@$(PYTHON) $(REPO)/scripts/fpga/f2/smoke_bridge_sim.py \
	  --job $(CAPGATE_JOB) --tile-div $(TILE_DIV) $(CAPGATE_CLEAN) \
	  $(if $(CAPGATE_VERBOSE),,--quiet) \
	  $(if $(CAPGATE_SCRATCH),--scratch $(CAPGATE_SCRATCH)) \
	  $(if $(CAPGATE_BIN),--binary $(CAPGATE_BIN))

clean:
	rm -rf obj_d* obj_beh obj_mut run_d*.log run_s*.log beh_*.log ddrload_*.log
