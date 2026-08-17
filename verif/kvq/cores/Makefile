# =============================================================================
# verif/kvq/cores/Makefile — clean-room UNIT suite for the reimplemented KVQ
# per-channel INT4 datapath cores (rtl/kvq/cores). Streams FRESH golden vectors
# (gen_cores_vectors.py, arbiter golden/apex_golden/cq_codec.py) into
# cq_value_path / cq_key_path and checks bit-exact across D∈{64,128}, both INT
# tiers, full + partial(flush) key groups, and the -0.0 / +0.0 outlier lane.
#
#   make cores  -> lint (cores RTL, -Wall, zero waivers) + gen + all configs
#   make lint   -> verilator --lint-only -Wall on the cores hierarchy
#   make gen    -> regenerate vectors from the golden model
# =============================================================================
SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
# Verilator's runtime needs C++14+. Some hosts ship a `c++` that defaults to an
# older standard; force a modern one so `--binary` links cleanly from clean.
export CXXFLAGS := -std=gnu++17
REPO      := ../../..
include $(REPO)/verif/tools.mk
CORES     := $(REPO)/rtl/kvq/cores
PYTHON ?= python3

# cores RTL, package first
CORE_SRCS := $(CORES)/cq_fp_pkg.sv \
             $(CORES)/cq_scale_unit.sv $(CORES)/cq_quant_unit.sv \
             $(CORES)/cq_dequant_unit.sv $(CORES)/scale_bank.sv \
             $(CORES)/scale_bank_store.sv \
             $(CORES)/cq_rne_div_pipe.sv $(CORES)/cq_quant_pipe.sv \
             $(CORES)/cq_scale_pipe.sv \
             $(CORES)/residual_buffer.sv $(CORES)/sram_controller.sv \
             $(CORES)/cq_value_path.sv $(CORES)/cq_key_path.sv

# DUT lint: -Wall, NO waivers (new RTL). TB build relaxes TB-only styles.
VLINT   := --lint-only -Wall --timing
VBUILD  := --binary --timing --assert -j 0 --timescale 1ns/1ps -Wall \
           -Wno-DECLFILENAME -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL

# name D TIER G K
CFGS := d64_cq4:64:1:8:0 d128_cq4:128:1:8:0 d64_cq8:64:0:2:0 \
        d128_cq8:128:0:2:0 d64_cq4p:64:2:4:2 d128_cq4p:128:2:8:2

.PHONY: cores lint gen clean $(addprefix run_,d64_cq4 d128_cq4 d64_cq8 d128_cq8 d64_cq4p d128_cq4p)

cores: lint gen run_all summary

lint:
	$(VERILATOR) $(VLINT) --top-module kvq_engine \
	  $(REPO)/rtl/kvq/kvq_engine.sv $(CORE_SRCS)
	@echo "CORES LINT CLEAN (-Wall, zero waivers on rtl/kvq/cores)"

gen:
	@mkdir -p build
	set -o pipefail; $(PYTHON) gen_cores_vectors.py build | tee build/gen.log
	@grep -q "CORES GEN OK" build/gen.log

# build+run one config: $1=name $2=D $3=TIER $4=G $5=K
define RUNCFG
run_$(1):
	@mkdir -p build logs
	$(VERILATOR) $(VBUILD) --top-module tb_cores \
	  -GD=$(2) -GTIER=$(3) -GG=$(4) -GK_OUT=$(5) \
	  -GVECDIR='"build/$(1)"' -GCFGNAME='"$(1)"' \
	  -Mdir build/obj_$(1) -o sim tb_cores.sv $(CORE_SRCS) \
	  2>&1 | tee logs/build_$(1).log
	set -o pipefail; ./build/obj_$(1)/sim 2>&1 | tee logs/run_$(1).log
	@grep -q "CORES RESULT \[$(1)\]: PASS" logs/run_$(1).log \
	  || { echo "GATE FAIL [$(1)]"; exit 1; }
	@! grep -q "%Error\|Assertion" logs/run_$(1).log \
	  || { echo "GATE FAIL [$(1)]: sim error"; exit 1; }
endef
$(eval $(call RUNCFG,d64_cq4,64,1,8,0))
$(eval $(call RUNCFG,d128_cq4,128,1,8,0))
$(eval $(call RUNCFG,d64_cq8,64,0,2,0))
$(eval $(call RUNCFG,d128_cq8,128,0,2,0))
$(eval $(call RUNCFG,d64_cq4p,64,2,4,2))
$(eval $(call RUNCFG,d128_cq4p,128,2,8,2))

run_all: run_d64_cq4 run_d128_cq4 run_d64_cq8 run_d128_cq8 run_d64_cq4p run_d128_cq4p

summary:
	@echo ""
	@echo "==================== KVQ CORES SUMMARY ===================="
	@grep -h "CONFIG\|CORES RESULT" logs/run_*.log | sed 's/^/  /'
	@echo "=========================================================="

clean:
	rm -rf build logs
