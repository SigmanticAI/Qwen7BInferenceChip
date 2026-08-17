RTL INTEGRATION MANIFEST
tensor_core_tile -- INT8 systolic-array GEMM accelerator tile
================================================================

STATUS: GENERATED, INTEGRATED, LINT-CLEAN (Verilator --lint-only).
NOT COMPILED with a cycle-accurate simulator, NOT SIMULATED, NOT
FUNCTIONALLY VERIFIED in this session. No testbench was provided or run.
iverilog is NOT INSTALLED in this environment (see tool evidence below) --
iverilog-based elaboration was NOT RUN / could not be run for that reason.

----------------------------------------------------------------
1. MODULE HIERARCHY (top down)
----------------------------------------------------------------
tensor_core_tile                    (rtl/tensor_core_tile.v)
  |-- ctrl_fsm                      (rtl/ctrl_fsm.v)
  |-- weight_buffer                 (rtl/weight_buffer.v)
  |-- act_fifo                      (rtl/act_fifo.v)
  |-- pe_array                      (rtl/pe_array.v)
  |     `-- pe  x (N*N instances)   (rtl/pe.v)
  |-- requant_unit  x N (generate loop, one lane per output column)
                                     (rtl/requant_unit.v)

----------------------------------------------------------------
2. SOURCE-AGENT MAP
----------------------------------------------------------------
  rtl/pe.v               -> datapath
  rtl/pe_array.v         -> datapath
  rtl/requant_unit.v     -> datapath
  rtl/weight_buffer.v    -> register_storage
  rtl/act_fifo.v         -> register_storage
  rtl/ctrl_fsm.v         -> fsm_control
  rtl/tensor_core_tile.v -> interface_protocol

----------------------------------------------------------------
3. PARAMETER DEFAULTS (top-level, tensor_core_tile)
----------------------------------------------------------------
  N            = 8    (array dimension, NxN PEs, NxN weight entries)
  ACC_W        = 32   (per-PE signed accumulator width)
  DW_OUT       = 8    (requantized output element width, INT8)
  SCALE_W      = 16   (unsigned requant scale width)
  SHIFT_W      = 5    (requant arithmetic right-shift amount width)
  AFIFO_DEPTH  = 16   (activation FIFO depth, exposed top-level parameter)
  WBUF_DEPTH   = 64   (see NOTE below -- not an independent top-level
                        parameter in the generated RTL; documented here
                        for traceability against the architecture contract)

  NOTE (documentation-level, not a functional defect):
  weight_buffer.v does not expose an independent DEPTH parameter. Its
  storage size is the localparam-derived N*N entries (AW = clog2(N*N)),
  i.e. for N=8 the effective depth is exactly 64, matching WBUF_DEPTH=64
  in the architecture contract. Because the weight buffer is inherently
  one full NxN weight matrix (not a general-depth queue like act_fifo),
  deriving its size from N rather than accepting an independent WBUF_DEPTH
  parameter is architecturally consistent and was NOT changed. Flagging
  for awareness only -- if the intent was an independently configurable
  WBUF_DEPTH decoupled from N*N, route to register_storage (weight_buffer
  owner) and interface_protocol (top-level port owner) for a parameter
  rename/exposure; this is a naming/exposure question, not a wiring bug,
  so it was not altered here.

----------------------------------------------------------------
4. TOP-LEVEL PORT LIST (tensor_core_tile)
----------------------------------------------------------------
  clk                          input
  rst_n                        input   (synchronous active-low)
  start                        input   (single-cycle pulse)
  cfg_mode        [1:0]        input   (reserved, tied off/observed only)
  busy                         output
  done                         output
  overflow                     output  (sticky, cleared on reset/start)
  w_load_valid                 input
  w_load_ready                 output
  w_load_data     [7:0]        input
  w_load_last                  input
  a_in_valid                   input
  a_in_ready                   output
  a_in_data       [N*8-1:0]    input
  a_in_last                    input
  y_out_valid                  output
  y_out_ready                  input
  y_out_data      [N*DW_OUT-1:0] output
  y_out_last                   output
  req_scale       [SCALE_W-1:0] input
  req_shift       [SHIFT_W-1:0] input

----------------------------------------------------------------
5. DEPENDENCY GRAPH (edges = "instantiates")
----------------------------------------------------------------
  tensor_core_tile -> ctrl_fsm
  tensor_core_tile -> weight_buffer
  tensor_core_tile -> act_fifo
  tensor_core_tile -> pe_array
  tensor_core_tile -> requant_unit  (N instances, generate loop)
  pe_array         -> pe            (N*N instances, nested generate loop)

  Leaf modules (no sub-instantiations): pe, requant_unit, weight_buffer,
  act_fifo, ctrl_fsm.
  Intermediate: pe_array (depends only on pe).
  Top: tensor_core_tile (depends on all of the above).

----------------------------------------------------------------
6. PORT / PARAMETER CROSS-CHECK RESULTS
----------------------------------------------------------------
All five submodule instantiations in tensor_core_tile.v were checked
port-by-port and parameter-by-parameter against the actual module
definitions:

  - u_ctrl_fsm     (ctrl_fsm.v)     : 23/23 ports match by name, direction,
                                       and width (ADDR_W/ROW_W formulas
                                       identical between the two files).
                                       Parameter N passed correctly.
  - u_weight_buffer(weight_buffer.v): 7/7 ports match. Only overridable
                                       parameter (N) passed correctly;
                                       AW/RW are localparams (correctly
                                       not passed as overrides).
  - u_act_fifo     (act_fifo.v)     : 8/8 ports match. W and DEPTH passed
                                       explicitly and correctly (W =
                                       AFIFO_W = N*8+1, matching the
                                       {a_in_last, a_in_data} packing).
  - u_pe_array     (pe_array.v)     : 7/7 ports match. N and ACC_W passed
                                       correctly.
  - u_pe (inside pe_array.v)        : 7/7 ports match. ACC_W passed
                                       correctly; 8-bit a_in/w_in slices
                                       verified correctly indexed per
                                       row i / column j.
  - u_requant_unit (requant_unit.v) : 5/5 ports match. ACC_W, SCALE_W,
                                       SHIFT_W, DW_OUT all passed
                                       correctly; per-lane acc_in slice
                                       (acc_flat[(drain_row*N+j)*ACC_W +:
                                       ACC_W]) correctly indexes the
                                       flattened accumulator array.

RESULT: NO port mismatches, NO width mismatches, NO name collisions
found. No top-level wiring fixes were required.

----------------------------------------------------------------
7. FILELIST / COMPILE ORDER (leaf-first)
----------------------------------------------------------------
See rtl/filelist.f (paths relative to rtl/):

  1. pe.v               (leaf)
  2. pe_array.v          (depends on pe)
  3. requant_unit.v      (leaf)
  4. weight_buffer.v     (leaf)
  5. act_fifo.v          (leaf)
  6. ctrl_fsm.v          (leaf)
  7. tensor_core_tile.v  (top, depends on all above)

This same order is valid for both Verilator (`-f filelist.f`) and
iverilog (`-g2012 -f filelist.f`), since Verilog does not strictly
require leaf-first ordering for single-file-per-module compilation
but leaf-first is used here as the canonical/most portable order for
simulators that do care (older two-pass tools, some lint flows).

----------------------------------------------------------------
8. UNRESOLVED INTEGRATION ISSUES
----------------------------------------------------------------
  - None blocking. One documentation/naming observation (WBUF_DEPTH
    parameter exposure, see section 3 NOTE) is flagged for awareness;
    it does not affect functional correctness and was not changed.
  - No semantic/logic issues were found requiring routing to a source
    agent (datapath, fsm_control, register_storage) or to the Global
    Spec Agent.
