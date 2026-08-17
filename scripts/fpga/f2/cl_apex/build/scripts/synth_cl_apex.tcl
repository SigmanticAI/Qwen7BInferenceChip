# synth_cl_apex.tcl — Vivado synthesis entry for the APEX F2 CL.
# ⏸ VALIDATION-DEFERRED: authored against the aws-fpga v2.3.3
# cl_axil_reg_access pattern; first run on the EC2 dev box validates it.
# Prereqs: CL dir assembled per BRINGUP.md step 3 (CL_TEMPLATE copy with
# this design/), env APEX_REPO_DIR = apex repo checkout on the dev box.

# Common kit header (sets HDK dirs, part, shell DCP, src_post_enc_dir …)
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl

###############################################################################
print "Reading APEX RTL (from APEX_REPO_DIR, unencrypted own-IP)"
###############################################################################
set apex_dir $::env(APEX_REPO_DIR)
# The kit's shell headers set `default_nettype none and Vivado directives
# bleed across read_verilog calls (single compilation unit). APEX RTL is
# written to standard `default_nettype wire semantics (typed ANSI ports
# without an explicit net kind) — same as Verilator/yosys/iverilog read it.
# Restore the default before our sources; Synth 8-6735 otherwise rejects
# every typed port (first casualty: apex_kvq_bank.sv 'tier_sel').
set prelude_f [file join [pwd] apex_nettype_prelude.v]
set fp [open $prelude_f w]
puts $fp "`default_nettype wire"
close $fp
read_verilog -sv $prelude_f
set fp [open "${apex_dir}/scripts/fpga/f2/cl_apex/design/apex_sources.f" r]
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line eq "" || [string index $line 0] eq "#"} { continue }
    read_verilog -sv ${apex_dir}/${line}
}
close $fp

###############################################################################
print "Reading CL top + defines (design/ of this CL)"
###############################################################################
read_verilog -sv [glob ${src_post_enc_dir}/*.{s,}v]

###############################################################################
print "Reading user constraints"
###############################################################################
# 2026-08-05: this used to read ${constraints_dir}/cl_synth_user.xdc — the copy
# inside the ASSEMBLED CL dir. The assembly step never copied ours, so every
# build up to cl_apex.2026_08_05-012859 silently parsed the kit's stock
# placeholder and ran with NO CDC constraints at all: no ASYNC_REG anywhere,
# and the tile<->shell payload crossings timed as ordinary synchronous paths
# (report_clock_interaction: "No Common Phase / Timed (unsafe)"). That is what
# failed the first DDR build at -0.850 ns, not the DDR IP.
#
# Read it from APEX_REPO_DIR instead — the same source of truth the RTL comes
# from — so a CL dir assembled without it cannot silently disarm the CDC rules.
set apex_xdc ${apex_dir}/scripts/fpga/f2/cl_apex/constraints/cl_synth_user.xdc
if {![file exists $apex_xdc]} {
  error "APEX: cl_synth_user.xdc not found at $apex_xdc — refusing to build with unconstrained CDC"
}
print "APEX: user constraints from APEX_REPO_DIR: $apex_xdc"
read_xdc $apex_xdc
set_property PROCESSING_ORDER LATE [get_files cl_synth_user.xdc]

# Synthesis — mirrors cl_axil_reg_access's invocation VERBATIM (the kit flow
# defines ${CL} and ${DEVICE_TYPE}; the first build attempt died on invented
# variable names here — lesson recorded in BRINGUP.md)
update_compile_order -fileset sources_1
puts "AWS FPGA: Starting synthesis at [clock format [clock seconds] -format %T]"
# I-C: GQA/wide-D are env-selectable; unset reproduces the shipped narrow CL.
if {![info exists ::env(APEX_CL_GQA)] || [string trim $::env(APEX_CL_GQA)] eq ""} { set ::env(APEX_CL_GQA) 1 }
if {![info exists ::env(APEX_CL_DM)] || [string trim $::env(APEX_CL_DM)] eq ""} { set ::env(APEX_CL_DM) 128 }
if {![info exists ::env(APEX_CL_DDR)] || [string trim $::env(APEX_CL_DDR)] eq ""} { set ::env(APEX_CL_DDR) 0 }
# 2026-08-03: APEX_CL_D was HARDCODED to 128 below (it predates any non-7B
# build), so the first "b64_05b" build silently produced a D=128 hybrid that
# no sim had ever validated — caught on the card by the per-program INFO_D
# identity read (got 0x80 want 0x40), not by any build-time gate. All THREE
# geometry knobs are now env-plumbed with the old values as defaults, so
# every historical invocation is byte-identical.
if {![info exists ::env(APEX_CL_D)] || [string trim $::env(APEX_CL_D)] eq ""} { set ::env(APEX_CL_D) 128 }
if {![info exists ::env(APEX_CL_DMODEL)] || [string trim $::env(APEX_CL_DMODEL)] eq ""} { set ::env(APEX_CL_DMODEL) $::env(APEX_CL_D) }
if {![info exists ::env(APEX_CL_QSTAGE)] || [string trim $::env(APEX_CL_QSTAGE)] eq ""} { set ::env(APEX_CL_QSTAGE) 1 }
# W4-DEBUT: env APEX_CL_W4EN (the EN spelling matches the tile suite's
# -GW4EN=1 and the combined build's export) -> cl_apex.sv's `APEX_CL_W4
# guard -> apex_top W4_LANE -> generate g_w4. The 2026-08-14 combined build
# exported APEX_CL_W4EN=1 but this script never consumed it, so the synth
# log showed ZERO w4 cells (clock_key.py agfi-04236db8bb2899ac0 note).
# Default 0 keeps every historical invocation byte-identical (lane absent,
# CSR 0x9C-0xAC read 0xDEADBEEF). Sim twin: verif/f2sim
#   make build VFLAGS_EXTRA=+define+APEX_CL_W4=1   (cl_apex has no top
# param, so the twin sets the SAME define this script sets below).
if {![info exists ::env(APEX_CL_W4EN)] || [string trim $::env(APEX_CL_W4EN)] eq ""} { set ::env(APEX_CL_W4EN) 0 }

# APEX_CL_D=128: the stage-2 trace jobs are Qwen2.5-7B head_dim=128. cl_apex
# defaults the build-selectable head dim to 64 (`ifndef APEX_CL_D); WITHOUT
# this define the AFI builds D=64 and cannot run the D=128 regops (the exact
# config gap that made first-light's AGFI unusable for these jobs).
## Clocking + CDC IPs required by aws_clk_gen (DECISION-LC-1, group A only;
## paths verbatim from the kit's cl_mem_perf/CL_TEMPLATE synth scripts):
read_ip [ list \
  ${HDK_SHELL_DESIGN_DIR}/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/clk_mmcm_a/clk_mmcm_a.xci \
  ${HDK_SHELL_DESIGN_DIR}/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_clk_axil_xbar/cl_clk_axil_xbar.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_register_slice_light/cl_axi_register_slice_light.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter_light/cl_axi_clock_converter_light.xci \
]

## IB-FUEL s5 — what APEX_CL_DDR=1 additionally needs.
##
## `-verilog_define APEX_CL_DDR=1` below only turns sh_ddr's DDR_PRESENT
## parameter on. That ALONE is not a buildable configuration:
##   (a) a DDR-core macro visible where sh_ddr.sv elaborates
##       (`define USE_64GB_DDR_DIMM -> the 64 GB DIMM core), per
##       hdk/docs/Supported_DDR_Modes.md;
##   (b) the matching DDR XCI, `cl_ddr4.xci` (same doc's table; the kit's
##       own cl_dram_hbm_dma reads exactly this one); AND
##   (c) cl_axi_clock_converter.xci.
##
## (c) is the one the docs never mention and the first DDR build died on
## (2026-08-04, Synth 8-5809 "Error generated from encrypted envelope" x3 at
## RTL elaboration). sh_ddr's ENCRYPTED body instantiates the full-width
## (512b AXI4) clock converter to cross from clk_main_a0 to the DDR4 core's
## own UI clock. Because that instantiation is inside a protected envelope,
## Vivado cannot name the unresolved module and degrades the message to the
## contentless 8-5809 — the error looks like it comes from AWS's IP, but it
## is only ever a missing file in THIS script.
##
## Bisected on the dev box with a 60-line wrapper that instantiates nothing
## but sh_ddr (~/ddr_probe): DDR_PRESENT=0 elaborates; DDR_PRESENT=1 fails;
## adding cl_axi_clock_converter alone makes it pass. Necessary AND
## sufficient — axi_register_slice, axi_register_slice_light,
## cl_axi3_256b_reg_slice and cl_axi_interconnect_64G_ddr each individually
## still FAIL, and the kit's whole cl_dram_hbm_dma IP list PASSes.
## NB cl_axi_clock_converter_light (already read above for the AXI-Lite
## paths) is a DIFFERENT IP and does not satisfy this.
##
## All three are applied ONLY when APEX_CL_DDR != 0, so every DDR=0
## invocation — i.e. every image built to date — is byte-identical.
set apex_ddr_defines {}
if {$::env(APEX_CL_DDR) != 0} {
  print "APEX: APEX_CL_DDR=$::env(APEX_CL_DDR) -> USE_64GB_DDR_DIMM + cl_ddr4.xci + cl_axi_clock_converter.xci"
  read_ip [ list \
    ${HDK_IP_SRC_DIR}/cl_ddr4/cl_ddr4.xci \
    ${HDK_IP_SRC_DIR}/cl_axi_clock_converter/cl_axi_clock_converter.xci \
  ]
  lappend apex_ddr_defines -verilog_define USE_64GB_DDR_DIMM
}

synth_design -mode out_of_context \
             -top ${CL} \
             {*}$apex_ddr_defines \
             -verilog_define XSDB_SLV_DIS \
             -verilog_define APEX_CL_D=$::env(APEX_CL_D) \
             -verilog_define APEX_CL_DMODEL=$::env(APEX_CL_DMODEL) \
             -verilog_define APEX_CL_QSTAGE=$::env(APEX_CL_QSTAGE) \
             -verilog_define APEX_CL_GQA=$::env(APEX_CL_GQA) \
             -verilog_define APEX_CL_DM=$::env(APEX_CL_DM) \
             -verilog_define APEX_CL_DDR=$::env(APEX_CL_DDR) \
             -verilog_define APEX_CL_W4=$::env(APEX_CL_W4EN) \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers
puts "AWS FPGA: Synthesis complete at [clock format [clock seconds] -format %T]"

###############################################################################
# CDC GATE — the constraints must have MATCHED the netlist, not just been read.
#
# Reading the file is not evidence that it did anything: every
# `get_cells -hier -filter {NAME =~ ...}` in it silently matches nothing if the
# hierarchy is ever renamed, and the build would sail on with the crossings
# unconstrained again — the same silent failure as the missing-file one, just
# one level in. Assert on the two groups whose datapath_only bounds carry the
# tile<->shell payload crossings, plus overall ASYNC_REG coverage.
###############################################################################
# A bare "count cells with ASYNC_REG" gate would NOT bite: 899 cells already
# carry it from Xilinx IP/XPM before our file is read at all (measured). The
# gate must therefore check OUR synchronizers specifically, and check that
# EVERY bit of each group got the property — a pattern that matches a renamed
# subset is the same silent failure one level in.
set apex_sync_pats {
  *u_ocl_cdc/ack_meta_reg*        *u_ocl_cdc/ack_sync_reg*
  *u_ocl_cdc/req_meta_reg*        *u_ocl_cdc/req_sync_reg*
  *u_fuel_rdr/req_meta_reg*       *u_fuel_rdr/req_sync_reg*
  *u_fuel_rdr/abt_meta_reg*       *u_fuel_rdr/abt_sync_reg*
  *u_fuel_ctl/ack_meta_reg*       *u_fuel_ctl/ack_sync_reg*
  *u_fuel_ctl/rdr_busy_meta_reg*  *u_fuel_ctl/rdr_busy_sync_reg*
  *u_fuel_ctl/ddr_rdy_meta_reg*   *u_fuel_ctl/ddr_rdy_sync_reg*
  *u_fuel_ctl/axi_err_meta_reg*   *u_fuel_ctl/axi_err_sync_reg*
  *u_fuel_fifo/rptr_gray_meta_reg* *u_fuel_fifo/rptr_gray_sync_reg*
  *u_fuel_fifo/wptr_gray_meta_reg* *u_fuel_fifo/wptr_gray_sync_reg*
  *arown_meta_reg*                *arown_sync_reg*
}
set apex_bad {}
foreach p $apex_sync_pats {
  set all [get_cells -hier -quiet -filter "NAME =~ $p"]
  if {[llength $all] == 0} { lappend apex_bad "$p -> NO CELLS"; continue }
  set ar [get_cells -hier -quiet -filter "NAME =~ $p && ASYNC_REG"]
  if {[llength $ar] != [llength $all]} {
    lappend apex_bad "$p -> only [llength $ar]/[llength $all] carry ASYNC_REG"
  }
}
# the two payload groups whose datapath_only bounds carry the failing crossings
foreach p {*u_ocl_cdc/resp_rdata_reg* *u_fuel_ctl/eng_rec_q_reg*} {
  if {[llength [get_cells -hier -quiet -filter "NAME =~ $p"]] == 0} {
    lappend apex_bad "$p -> NO CELLS (payload bound would be dead)"
  }
}
if {[llength $apex_bad] != 0} {
  error "APEX CDC gate FAILED — cl_synth_user.xdc did not match the netlist:\n  [join $apex_bad "\n  "]"
}
puts "APEX CDC gate OK: all [llength $apex_sync_pats] synchronizer groups carry ASYNC_REG; both payload groups matched"

# Common kit footer (checkpoints, reports)
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
