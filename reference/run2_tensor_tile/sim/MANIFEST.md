VERIFICATION INTEGRATION MANIFEST
tensor_core_tile -- non-UVM Verilator verification package
================================================================

STATUS: GENERATED / ASSEMBLED ONLY. This manifest documents and cross-checks
already-generated artifacts on disk. NO COMPILE was run and NO SIMULATION was
run in this session by this node. See rtl/MANIFEST.md for the separately-owned
RTL integration manifest (module hierarchy, port cross-check, RTL status);
this file covers the FULL verification package (TB + SVA + coverage + RTL
integration for build purposes).

No source file (.v/.sv/.svh) was created or modified by this node. Only this
manifest (sim/MANIFEST.md) and the filelist (sim/filelist.f) were written.

----------------------------------------------------------------
1. ARTIFACTS COLLECTED
----------------------------------------------------------------

RTL (7 files, compiled in all configurations):
  rtl/tensor_core_tile.v   323 lines   top module: tensor_core_tile
  rtl/ctrl_fsm.v           211 lines   module: ctrl_fsm
  rtl/weight_buffer.v       59 lines   module: weight_buffer
  rtl/act_fifo.v            96 lines   module: act_fifo
  rtl/pe_array.v            51 lines   module: pe_array
  rtl/pe.v                  47 lines   module: pe
  rtl/requant_unit.v        88 lines   module: requant_unit

TB top (compiled in all configurations, on command line):
  tb/tb_tensor_core_tile.sv   330 lines   top module: tb_tensor_core_tile
    - instantiates DUT as `dut`: tensor_core_tile #(N=8, ACC_W=32,
      DW_OUT=8, SCALE_W=16, SHIFT_W=5, AFIFO_DEPTH=16)
    - drives clk (10-time-unit period, `timescale 1ns/1ps)
    - fixed `include order (per in-file contract, comment at line 95):
        1. tct_scoreboard.svh   (line 278)
        2. tct_stimulus.svh     (line 285)
        3. tct_sva_bind.svh     (line 292, guarded `ifdef TCT_SVA_BIND)
        4. tct_coverage.svh     (line 303, guarded `ifdef TCT_COVERAGE)

TB includes (NOT on command line; textual `include, resolved via -Itb):
  tb/tct_scoreboard.svh   318 lines   owner: Scoreboard/Checker Agent
  tb/tct_stimulus.svh     393 lines   owner: Stimulus/Sequence Agent

Optional SVA include (NOT on command line; via -Iassertions,
guarded by `ifdef TCT_SVA_BIND / +define+TCT_SVA_BIND):
  assertions/tct_sva_bind.svh   234 lines   owner: Coverage and SVA Agent
    - 17 `assert property` statements (grep count, this session)
    - 0 covergroups/coverpoints (file explicitly avoids SV functional
      coverage constructs -- Verilator 5.044 does not support them)
    - relies on hierarchical refs into `dut` (dut.compute_active,
      dut.u_ctrl_fsm.state/.acc_clr/.acc_en/.load_full) -- requires those
      exact identifiers to exist in the RTL as elaborated
    - requires Verilator `--assert` flag to actually evaluate at runtime
      (without it, assertions are parsed/elaborated only, not checked)

Optional coverage include (NOT on command line; via -Icoverage,
guarded by `ifdef TCT_COVERAGE / +define+TCT_COVERAGE):
  coverage/tct_coverage.svh   278 lines   owner: Coverage and SVA Agent
    - manual bucket-counter functional coverage (always-blocks + counters),
      NOT SV covergroups (0 `covergroup` declarations found; file text
      only mentions the word in comments explaining why they are avoided)
    - relies on hierarchical refs into `dut` (dut.u_ctrl_fsm.state,
      dut.u_act_fifo.full/.empty/.flush, dut.compute_active,
      dut.REQUANT_LANE[j].u_requant_unit.sat/.q_out) and on `localparam
      int N` already declared in tb_tensor_core_tile.sv before the
      include point

----------------------------------------------------------------
2. BUILD CONFIGURATIONS (exact command lines, as supplied)
----------------------------------------------------------------

BASE:
  verilator --binary --timing -Wno-fatal -Irtl -Itb \
    tb/tb_tensor_core_tile.sv \
    rtl/tensor_core_tile.v rtl/ctrl_fsm.v rtl/weight_buffer.v \
    rtl/act_fifo.v rtl/pe_array.v rtl/pe.v rtl/requant_unit.v \
    --top-module tb_tensor_core_tile

SVA (adds to BASE):
  -Iassertions +define+TCT_SVA_BIND --assert

COVERAGE (adds to BASE):
  -Icoverage +define+TCT_COVERAGE

SVA + COVERAGE may be combined in one invocation (both -I paths, both
defines, and --assert together).

NONE of these three configurations were executed in this session. No
obj_dir/ or other Verilator build output exists in this package and none
was created by this node (per file-hygiene rule: build outputs are
throwaway and are not to be committed).

----------------------------------------------------------------
3. SOURCE-AGENT MAP (artifact ownership, preserved from upstream agents)
----------------------------------------------------------------
  rtl/pe.v                     -> datapath (RTL agent)
  rtl/pe_array.v                -> datapath (RTL agent)
  rtl/requant_unit.v            -> datapath (RTL agent)
  rtl/weight_buffer.v           -> register_storage (RTL agent)
  rtl/act_fifo.v                -> register_storage (RTL agent)
  rtl/ctrl_fsm.v                -> fsm_control (RTL agent)
  rtl/tensor_core_tile.v        -> interface_protocol (RTL agent)
  tb/tb_tensor_core_tile.sv     -> TB harness/top-level integration agent
  tb/tct_scoreboard.svh         -> Scoreboard/Checker Agent
  tb/tct_stimulus.svh           -> Stimulus/Sequence Agent
  assertions/tct_sva_bind.svh   -> Coverage and SVA Agent
  coverage/tct_coverage.svh     -> Coverage and SVA Agent

(RTL-internal hierarchy, parameters, and port-by-port cross-check are
owned and documented in rtl/MANIFEST.md; not duplicated here in full.)

----------------------------------------------------------------
4. DEPENDENCY GRAPH
----------------------------------------------------------------

RTL instantiation edges:
  tensor_core_tile -> ctrl_fsm
  tensor_core_tile -> weight_buffer
  tensor_core_tile -> act_fifo
  tensor_core_tile -> pe_array
  tensor_core_tile -> requant_unit   (N=8 instances, generate loop,
                                       label REQUANT_LANE[j])
  pe_array         -> pe             (N*N=64 instances, nested generate)

TB structural edges:
  tb_tensor_core_tile -> tensor_core_tile   (instance `dut`)
  tb_tensor_core_tile --`include--> tct_scoreboard.svh
  tb_tensor_core_tile --`include--> tct_stimulus.svh
  tb_tensor_core_tile --`include--> tct_sva_bind.svh    [guarded: TCT_SVA_BIND]
  tb_tensor_core_tile --`include--> tct_coverage.svh    [guarded: TCT_COVERAGE]

Include-file -> RTL hierarchical-reference dependencies (textual coupling,
not compile-order dependencies, but load-bearing at elaboration time):
  tct_sva_bind.svh  references: dut.compute_active, dut.u_ctrl_fsm.state,
                                 dut.u_ctrl_fsm.acc_clr, dut.u_ctrl_fsm.acc_en,
                                 dut.u_ctrl_fsm.load_full
  tct_coverage.svh  references: dut.u_ctrl_fsm.state, dut.u_act_fifo.full,
                                 dut.u_act_fifo.empty, dut.u_act_fifo.flush,
                                 dut.compute_active,
                                 dut.REQUANT_LANE[j].u_requant_unit.sat,
                                 dut.REQUANT_LANE[j].u_requant_unit.q_out

These hierarchical paths were checked by name against rtl/ctrl_fsm.v,
rtl/act_fifo.v, rtl/tensor_core_tile.v, and rtl/requant_unit.v instance
labels found via grep in this session; a full elaboration-time check of
every referenced signal was NOT performed here (would require an actual
Verilator compile, which was not run -- see section 7).

Leaf modules (no sub-instantiations): pe, requant_unit, weight_buffer,
act_fifo, ctrl_fsm.
Intermediate: pe_array (depends only on pe).
Top RTL: tensor_core_tile.
Top TB: tb_tensor_core_tile (depends on tensor_core_tile + 2 mandatory
includes + 2 optional guarded includes).

----------------------------------------------------------------
5. FILELIST
----------------------------------------------------------------
See sim/filelist.f (RTL + TB top only, in the exact order of the BASE
command line above). The four .svh files are NOT listed there because
they are textual `include targets resolved via -I search paths, not
independent command-line source files, per the task's explicit
instruction.

----------------------------------------------------------------
6. COMPILE ORDER
----------------------------------------------------------------
As transcribed from the supplied BASE Verilator command line (verbatim,
not re-derived from the dependency graph -- see note in sim/filelist.f):

  1. tb/tb_tensor_core_tile.sv
  2. rtl/tensor_core_tile.v
  3. rtl/ctrl_fsm.v
  4. rtl/weight_buffer.v
  5. rtl/act_fifo.v
  6. rtl/pe_array.v
  7. rtl/pe.v
  8. rtl/requant_unit.v

Textually included (not separate compile-order entries, resolved at
`include time by the preprocessor via -I search path):
  - tb/tct_scoreboard.svh        (via -Itb)
  - tb/tct_stimulus.svh          (via -Itb)
  - assertions/tct_sva_bind.svh  (via -Iassertions, if +define+TCT_SVA_BIND)
  - coverage/tct_coverage.svh    (via -Icoverage, if +define+TCT_COVERAGE)

Required -I search paths by configuration:
  BASE:            -Irtl -Itb
  BASE+SVA:        -Irtl -Itb -Iassertions
  BASE+COVERAGE:   -Irtl -Itb -Icoverage
  BASE+SVA+COVERAGE: -Irtl -Itb -Iassertions -Icoverage

----------------------------------------------------------------
7. PACKAGE IMPORTS
----------------------------------------------------------------
NONE. This is a non-UVM package by design (confirmed via grep for
`package` / `import .*::` across all 12 collected artifacts in this
session: zero matches). No SystemVerilog package compilation units exist
in this package, so no package pre-compile step is required in any of
the three build configurations.

----------------------------------------------------------------
8. UNRESOLVED QUESTIONS / GAPS
----------------------------------------------------------------
  a. NOT COMPILED / NOT RUN this session: none of BASE, SVA, or COVERAGE
     configurations (nor any combination) were invoked with verilator in
     this session by this node. Verilator 5.044 is present on this
     machine (`verilator --version` confirmed), but running the build
     was out of scope for this integration-manifest task ("Do NOT create
     build outputs"). Compile/elaborate/simulate status for all three
     configurations is therefore: NOT COMPILED / NOT RUN.
  b. The `--assert` flag is required for the SVA configuration's 17
     `assert property` statements to be evaluated at runtime; without it
     they are elaborated but not checked. This is documented in-file by
     the SVA agent and repeated here for build-script authors.
  c. tct_coverage.svh and tct_sva_bind.svh both depend on internal
     hierarchical signal names inside `dut` (u_ctrl_fsm, u_act_fifo,
     REQUANT_LANE[j].u_requant_unit) matching exactly what
     rtl/tensor_core_tile.v instantiates. This node cross-checked
     instance/module names by static grep only; it did NOT run an actual
     Verilator elaboration to confirm every hierarchical signal resolves
     (e.g. exact bit-level existence of dut.u_ctrl_fsm.load_full as an
     internal wire vs. port). Recommend the first real compile attempt
     treat any "Unable to find" hierarchical reference errors from
     Verilator as expected verification work, not an integration defect.
  d. No coverage or assertion PASS/FAIL results exist anywhere in this
     package -- none have ever been run. Any future report claiming
     "assertions passed" or "coverage converged" for this design must be
     backed by a real pasted verilator run in that session.
  e. No missing file dependencies were found: all 12 artifacts listed in
     the task (7 RTL + 1 TB top + 2 TB includes + 1 SVA include + 1
     coverage include) exist on disk at the stated paths and were read
     in full or in part during this integration pass.
