# =============================================================================
# verif/kvq/replay/Makefile — S8 trace → RTL replay of real-model KVQ jobs.
#
# Replays the fp16 K/V rows that a REAL Qwen2.5-7B token stream produced
# (run_tinynpu.py job traces) through the actual cq_value_path / cq_key_path
# RTL, reusing verif/kvq/cores/tb_cores.sv unchanged, and checks scale /
# payload / code / x̂ bit-exact against the values the golden pipeline stored
# at generation time.
#
#   make gen NAME=smoke TRACE=$(REPO)/docs/results/s8_7b_token/smoke_trace
#       pure Python — safe on a busy machine
#   make run NAME=smoke
#       ⚠ builds + runs Verilator — QUIET MACHINE ONLY (EDA and
#       model jobs serialize on this host; check
#       `ps aux | grep -cE "verilator|nextpnr|yosys"` first)
# =============================================================================
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
export CXXFLAGS := -std=gnu++17
REPO      := ../../..
include $(REPO)/verif/tools.mk
CORESDIR  := ../cores
CORES     := $(REPO)/rtl/kvq/cores
PYTHON ?= python3

NAME  ?= smoke
TRACE ?= $(REPO)/docs/results/s8_7b_token/smoke_trace

# same DUT set as verif/kvq/cores (package first)
CORE_SRCS := $(CORES)/cq_fp_pkg.sv \
             $(CORES)/cq_scale_unit.sv $(CORES)/cq_quant_unit.sv \
             $(CORES)/cq_dequant_unit.sv $(CORES)/scale_bank.sv \
             $(CORES)/scale_bank_store.sv \
             $(CORES)/cq_rne_div_pipe.sv $(CORES)/cq_quant_pipe.sv \
             $(CORES)/cq_scale_pipe.sv \
             $(CORES)/residual_buffer.sv $(CORES)/sram_controller.sv \
             $(CORES)/cq_value_path.sv $(CORES)/cq_key_path.sv

.PHONY: gen run clean

gen:
	@mkdir -p build logs
	set -o pipefail; $(PYTHON) gen_replay_vectors.py --trace $(TRACE) --name $(NAME) \
	  | tee logs/gen_$(NAME).log
	@grep -q "SELF-CHECK OK" logs/gen_$(NAME).log

# cfg.txt: "D TIER G K_OUT NV NG" — build tb_cores with the traced config
run:
	@test -f build/$(NAME)/cfg.txt || \
	  { echo "no vectors — run: make gen NAME=$(NAME) TRACE=..."; exit 1; }
	@mkdir -p logs
	read D TIER G K NV NG < build/$(NAME)/cfg.txt; \
	$(VERILATOR) --binary --timing --assert -j 0 --timescale 1ns/1ps -Wall \
	  -Wno-DECLFILENAME -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL \
	  --top-module tb_cores \
	  -GD=$$D -GTIER=$$TIER -GG=$$G -GK_OUT=$$K \
	  -GVECDIR='"build/$(NAME)"' -GCFGNAME='"$(NAME)"' \
	  -Mdir build/obj_$(NAME) -o sim $(CORESDIR)/tb_cores.sv $(CORE_SRCS) \
	  2>&1 | tee logs/build_$(NAME).log
	set -o pipefail; ./build/obj_$(NAME)/sim 2>&1 | tee logs/run_$(NAME).log
	@grep -q "CORES RESULT \[$(NAME)\]: PASS" logs/run_$(NAME).log \
	  || { echo "REPLAY GATE FAIL [$(NAME)]"; exit 1; }
	@! grep -q "%Error\|Assertion" logs/run_$(NAME).log \
	  || { echo "REPLAY GATE FAIL [$(NAME)]: sim error"; exit 1; }
	@echo "S8 RTL REPLAY [$(NAME)]: real-model KVQ rows bit-exact in RTL"

clean:
	rm -rf build logs
