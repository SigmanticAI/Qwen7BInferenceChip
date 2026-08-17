# =============================================================================
# sim/Makefile -- Verilator build/run for the gemm_systolic_accel structural
# harness (tb/gemm_tb_top.sv). NON-UVM, plain SystemVerilog.
#
# Targets:
#   make compile   - elaborate + build a Verilator --binary executable
#   make run       - execute the built binary
#   make all       - compile then run (default)
#   make clean     - remove build directory (throwaway, never committed)
#
# Variables:
#   N        - lane count, forwarded as +define+TB_N=<N> (default 8)
#   BUILD_DIR - Verilator obj dir (default build/obj_dir_N<N>), throwaway
#
# Example:
#   make -C sim all           # N=8 (default)
#   make -C sim all N=16      # N=16 build
# =============================================================================

N         ?= 8
TOP        = gemm_tb_top
BUILD_DIR ?= build/obj_dir_N$(N)

RTL_DIR = ../rtl
TB_DIR  = ../tb

RTL_SRCS = \
    $(RTL_DIR)/stream_skid.sv \
    $(RTL_DIR)/gemm_pe_array.sv \
    $(RTL_DIR)/gemm_wbuf.sv \
    $(RTL_DIR)/gemm_abuf.sv \
    $(RTL_DIR)/gemm_requant.sv \
    $(RTL_DIR)/gemm_ctrl.sv \
    $(RTL_DIR)/gemm_systolic_accel.sv

TB_SRCS = \
    $(TB_DIR)/gemm_tb_pkg.sv \
    $(TB_DIR)/gemm_tb_top.sv

VERILATOR      = verilator
VERILATOR_OPTS = --binary -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
                 --timescale 1ns/1ps \
                 --top-module $(TOP) \
                 -Mdir $(BUILD_DIR) \
                 +define+TB_N=$(N) \
                 -y $(TB_DIR)

.PHONY: all compile run clean

all: compile run

compile:
	mkdir -p $(BUILD_DIR)
	$(VERILATOR) $(VERILATOR_OPTS) $(RTL_SRCS) $(TB_SRCS)

run:
	$(BUILD_DIR)/V$(TOP)

clean:
	rm -rf build
