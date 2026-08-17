# cl_synth_user.xdc — cl_apex user constraints (DECISION-LC-1 two-clock CL).
#
# The recipe clocks themselves (clk_extra_a1 @ 15.625 MHz under
# --clock_recipe_a A2) are created by the kit's generated clock constraints
# (aws_gen_clk_constraints.tcl) — do NOT re-create them here. This file
# carries ONLY the CDC exceptions for apex_ocl_cdc (4-phase toggle req/ack
# bridge between clk_main_a0 and clk_tile = o_clk_extra_a1).
#
# ⚠ HISTORY (2026-08-05): this file was NEVER READ by any build up to and
# including cl_apex.2026_08_05-012859. synth_cl_apex.tcl read
# ${constraints_dir}/cl_synth_user.xdc — the copy inside the assembled CL dir —
# and the CL-dir assembly step never copied this file, so every build to date
# parsed the kit's stock placeholder ("# Add synthesis constraint here",
# md5 cfc49654c75edddc6d562366ea9a8503) instead. Consequences, all confirmed on
# the routed DCP: ZERO cells carried ASYNC_REG; report_clock_interaction
# classified clk_out1_clk_mmcm_a <-> clk_main_a0 as "No Common Phase /
# Timed (unsafe)"; and the DDR=1 build failed timing at -0.850 ns on exactly
# the two payload crossings bounded below. The tcl now reads this file from
# APEX_REPO_DIR and gates on it actually matching cells.
#
# OUTCOME — cl_apex.2026_08_05-032359, the first build that actually read this
# file. post_route (Design: top), all numbers from the report, not inferred:
#   overall            WNS +0.392  TNS 0.000  "All user specified timing
#                      constraints are met"; checkpoint is post_route.dcp, NOT
#                      post_route.VIOLATED.dcp; pr_verify HDPRVerify-42 present,
#                      HDPRVerify-41 absent.
#   clk_out1 -> main   -0.850 / 108-of-108 failing / "No Common Phase, Timed
#                      (unsafe)"   ->   +1.795 / 0-of-105 / "Max Delay
#                      Datapath Only"
#   main -> clk_out1   +1.562 / "Timed (unsafe)"  ->  +3.101 / "Max Delay
#                      Datapath Only"
#   main -> main       -0.778 / 181 failing  ->  +0.419 / 0 failing  (the
#                      knock-on congestion cleared once the bogus crossings
#                      stopped competing for the router)
# The worst path in the whole design is now inside AWS's DDR4 core on
# mmcm_clkout0 (intra-domain), not in our CDC at all.
#
# The datapath_only bounds below are one destination-clock period, except the
# shell->tile payloads where the destination period is 64 ns and 16.0 is
# already ~8x tighter than the protocol needs.

# ── 2FF synchronizer pairs (metastability protection) ────────────────────────
# ── the two domains, resolved from the design ───────────────────────────────
# Never hard-code the generated-clock name: clk_out1_clk_mmcm_a is emitted by
# the clk_mmcm_a IP and is recipe-dependent, so a literal would silently stop
# matching. u_fuel_ctl is tile-side, u_fuel_rdr is shell-side (apex_fuel_ctl.sv:46).
# Self-guarding: if either resolves empty, every set_max_delay using it fails
# loudly with "CRITICAL WARNING [Constraints 18-540] ... requires -from to be
# non-empty" rather than quietly widening to all clocks.
set apex_clk_tile  [get_clocks -quiet -of_objects [get_pins -quiet -filter {IS_CLOCK} \
                      -of_objects [get_cells -hier -filter {NAME =~ *u_fuel_ctl/ack_meta_reg*}]]]
set apex_clk_shell [get_clocks -quiet -of_objects [get_pins -quiet -filter {IS_CLOCK} \
                      -of_objects [get_cells -hier -filter {NAME =~ *u_fuel_rdr/req_meta_reg*}]]]

set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_ocl_cdc/ack_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_ocl_cdc/ack_sync_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_ocl_cdc/req_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_ocl_cdc/req_sync_reg*}]

# ── toggle flags: single-bit domain crossings into their 2FF pairs ──────────
set_max_delay -datapath_only 4.0 \
  -from [get_cells -hier -filter {NAME =~ *u_ocl_cdc/req_tgl_reg*}] \
  -to   [get_cells -hier -filter {NAME =~ *u_ocl_cdc/req_meta_reg*}]
set_max_delay -datapath_only 4.0 \
  -from [get_cells -hier -filter {NAME =~ *u_ocl_cdc/ack_tgl_reg*}] \
  -to   [get_cells -hier -filter {NAME =~ *u_ocl_cdc/ack_meta_reg*}]

# ── quasi-static payload buses (launched before req/ack toggles, held
#    stable until the handshake completes — bounded for skew only) ──────────
set_max_delay -datapath_only 16.0 -from [get_cells -hier -filter {NAME =~ *u_ocl_cdc/addr_q_reg*}]
set_max_delay -datapath_only 16.0 -from [get_cells -hier -filter {NAME =~ *u_ocl_cdc/wdat_q_reg*}]
# wstrb_q: DELIBERATELY ABSENT. The register is optimized away entirely —
# the downstream CSR/mailbox bus ignores t_wstrb (all writes are full 32-bit),
# so synthesis removes wstrb_q and the pattern matches ZERO cells. Constraining
# it raised "CRITICAL WARNING [Vivado 12-4739] No valid object(s) found", which
# is worse than useless: it trains readers to scroll past critical warnings.
# There is no register, therefore no crossing. Re-add if the FSM ever honours
# byte enables.
set_max_delay -datapath_only 16.0 -from [get_cells -hier -filter {NAME =~ *u_ocl_cdc/is_wr_q_reg*}]
set_max_delay -datapath_only 16.0 -from [get_cells -hier -filter {NAME =~ *u_ocl_cdc/resp_rdata_reg*}]
set_max_delay -datapath_only 16.0 -from [get_cells -hier -filter {NAME =~ *u_ocl_cdc/resp_code_reg*}]

# ── tile-domain reset synchronizer (rst_main_n sampled on clk_tile) ─────────
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *t_rstn_q_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *rst_tile_n_reg*}]
set_false_path -to [get_cells -hier -filter {NAME =~ *t_rstn_q_reg*}]

# ═══ IB-FUEL s3 crossings (docs/design/IB_FUEL.md §4) — values follow the
#     same review-at-first-build rule as the header states ═══════════════════

# ── apex_afifo u_fuel_fifo: gray-pointer 2FF pairs (both directions) ────────
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_fifo/rptr_gray_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_fifo/rptr_gray_sync_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_fifo/wptr_gray_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_fifo/wptr_gray_sync_reg*}]
set_max_delay -datapath_only 4.0 \
  -from [get_cells -hier -filter {NAME =~ *u_fuel_fifo/wptr_gray_q_reg*}] \
  -to   [get_cells -hier -filter {NAME =~ *u_fuel_fifo/wptr_gray_meta_reg*}]
set_max_delay -datapath_only 4.0 \
  -from [get_cells -hier -filter {NAME =~ *u_fuel_fifo/rptr_gray_q_rd_reg*}] \
  -to   [get_cells -hier -filter {NAME =~ *u_fuel_fifo/rptr_gray_meta_reg*}]
# RAM payload: shell writes mem[], tile reads it combinationally
# (apex_afifo.sv:92, assign rdata = mem[rptr_bin_q[...]]). This IS a real
# cross-domain path and it is the FUEL DATAPATH: measured endpoints are
# u_tile/u_mxe/u_skid_wgt/m_data_reg[*], i.e. DDR -> reader -> afifo -> MXE
# weights. Before this line it was completely unconstrained and was the
# critical path of the whole clk_main_a0 -> clk_out1_clk_mmcm_a direction
# (slack 2.432 ns against a synchronous 4.000 ns requirement it has no business
# being held to) — the same "passes until it doesn't" trap that produced the
# -0.850 ns violation in the other direction.
#
# Safe by construction: gray-coded pointers, 2FF synchronized in BOTH
# directions (apex_afifo.sv:62, :88), and a slot is stable for >=2 synchronized
# gray-pointer updates before the read domain can address it (:12-18).
#
# The earlier `-from [get_cells ... mem_reg*]` never applied — LUTRAM cells are
# not valid timing startpoints and Vivado rejected ~90 of them with
# "WARNING [Constraints 18-402] ... is not a valid startpoint". The working
# idiom is AWS's own, from the shell constraints applied to every CL
# (hdk/common/shell_stable/build/constraints/cl_ddr_timing_aws.xdc), which
# reaches the RAM through its CLK *pins* for exactly this structure:
#     set_false_path -from [get_pins -of_objects \
#                        [get_cells -hierarchical -filter {NAME =~ *ram_reg*}] \
#                        -filter {REF_PIN_NAME == CLK}] \
#                    -to   [get_cells -hierarchical -filter {NAME =~ *rd_do_reg[*]}]
# Same idiom here, but bounded rather than false-pathed: -datapath_only still
# caps the crossing, where a false path would let it grow without limit.
# 16.0 ns against a 64 ns destination period and a >=128 ns protocol guarantee,
# and the paths measure ~1.5 ns today.
set_max_delay -datapath_only 16.0 \
  -from [get_pins -quiet -of_objects [get_cells -hier -filter {NAME =~ *u_fuel_fifo/mem_reg*}] \
           -filter {REF_PIN_NAME == CLK}] \
  -to   [get_clocks $apex_clk_tile]

# ── fuel_req 4-phase (tile ctl -> shell reader) + ack return ────────────────
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_rdr/req_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_rdr/req_sync_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_rdr/abt_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_rdr/abt_sync_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/ack_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/ack_sync_reg*}]
set_max_delay -datapath_only 4.0 \
  -from [get_cells -hier -filter {NAME =~ *u_fuel_ctl/req_tgl_out_reg*}] \
  -to   [get_cells -hier -filter {NAME =~ *u_fuel_rdr/req_meta_reg*}]
set_max_delay -datapath_only 4.0 \
  -from [get_cells -hier -filter {NAME =~ *u_fuel_ctl/abort_tgl_out_reg*}] \
  -to   [get_cells -hier -filter {NAME =~ *u_fuel_rdr/abt_meta_reg*}]
set_max_delay -datapath_only 4.0 \
  -from [get_cells -hier -filter {NAME =~ *u_fuel_rdr/ack_tgl_out_reg*}] \
  -to   [get_cells -hier -filter {NAME =~ *u_fuel_ctl/ack_meta_reg*}]
# record payload: tile -> shell. NOTE the destination here is the FAST domain
# (clk_main_a0, 4 ns), unlike every other payload bound in this file. And
# apex_fuel_ctl.sv:174-176 latches eng_rec_q in the SAME tile edge that toggles
# req_tgl_out -- not before it. So the guarantee the reader gives us is only
# "2 shell cycles" (req_meta -> req_sync -> req_edge, apex_fuel_reader.sv:121
# and :140), i.e. 8 ns, NOT the 128 ns that the shell->tile payloads enjoy.
# Bounded at one destination period, consistent with the toggle bounds above
# and comfortably met (measured 0.36-0.77 ns at post-route).
# Was 16.0 until 2026-08-05: never enforced, because this file was not being
# read at all, and 2x looser than the protocol proves had it been.
set_max_delay -datapath_only 4.0 -from [get_cells -hier -filter {NAME =~ *u_fuel_ctl/eng_rec_q_reg*}]

# ── single-bit status levels (shell -> tile, and ar_owner tile -> shell) ────
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/rdr_busy_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/rdr_busy_sync_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/ddr_rdy_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/ddr_rdy_sync_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/axi_err_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_fuel_ctl/axi_err_sync_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *arown_meta_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *arown_sync_reg*}]
# These four were written -to-only and Vivado REJECTED all four with
# "CRITICAL WARNING [Constraints 18-540] set_max_delay -datapath_only requires
# -from to be non-empty" — i.e. the single-bit status crossings have been
# completely unconstrained. -datapath_only is only legal with a -from, so give
# it one: the launching domain, resolved from the design rather than hard-coding
# the recipe-dependent generated-clock name (clk_out1_clk_mmcm_a is an IP-
# generated name and would silently stop matching if the recipe changed).
# (apex_clk_tile / apex_clk_shell are resolved once at the top of this file.)

# shell -> tile levels (destination period 64 ns; 4.0 leaves the 2FF pair the
# maximum possible settling window, and is met with >3 ns to spare)
set_max_delay -datapath_only 4.0 -from [get_clocks $apex_clk_shell] \
  -to [get_cells -hier -filter {NAME =~ *u_fuel_ctl/rdr_busy_meta_reg*}]
set_max_delay -datapath_only 4.0 -from [get_clocks $apex_clk_shell] \
  -to [get_cells -hier -filter {NAME =~ *u_fuel_ctl/ddr_rdy_meta_reg*}]
set_max_delay -datapath_only 4.0 -from [get_clocks $apex_clk_shell] \
  -to [get_cells -hier -filter {NAME =~ *u_fuel_ctl/axi_err_meta_reg*}]
# tile -> shell (ar_owner); destination period is 4 ns, so 4.0 is exactly one
set_max_delay -datapath_only 4.0 -from [get_clocks $apex_clk_tile] \
  -to [get_cells -hier -filter {NAME =~ *arown_meta_reg*}]
