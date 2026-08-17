// mutant/mxe_array.sv — SCRATCH MUTANT (broadcast MAC wall) for the D-004
// structural discriminator.
// NOT product RTL; lives under verif/mxe/struct/ only and is never compiled
// into any shipping build.
//
// This is a *realistic* broadcast MAC wall (the run1/run2 microarchitecture
// that critic item 1.15 flagged and D-004 forbids), made functionally
// EQUIVALENT to the true systolic mxe_array at the module boundary:
//   - same ports, same weight-load column-shift protocol, same clr/row tag
//     semantics, same non-destructive drain, latency (3 cycles) well inside
//     mxe_ctrl's FLUSH window;
//   - activations are broadcast to ALL columns in the same cycle (one input
//     register stage, NO west-edge skew, NO act hops);
//   - each column reduces with a combinational adder tree (NO psum hops);
//   - ALL column accumulators are written in the SAME cycle (single tag
//     delay, NO MXE_N+c stagger).
//
// Expected use (verif/mxe/struct/Makefile mutation gate):
//   tb_mxe_smoke  built with this file must print ALL TESTS PASSED
//     (functional equivalence — the functional suite cannot discriminate);
//   tb_mxe_struct built with this file must FAIL
//     (the structural discriminator catches the missing PE-to-PE pipelining).
//
// The whitebox probe aliases act_h / psum_v / acc keep the same names and
// shapes as the real mxe_array so tb_mxe_struct compiles unchanged; they are
// the natural "same role" nets of a broadcast wall (act_h = the broadcast
// wire, psum_v = the combinational adder-tree taps).

module mxe_array
  import apex_pkg::*;
  import mxe_cfg_pkg::*;
(
  input  logic clk,
  input  logic rst_n,            // synchronous, active-low

  // compute pipeline advance
  input  logic en,

  // activation beat injection (lane k = data[8k +: 8])
  input  logic [8*MXE_N-1:0] act_lanes,
  input  logic               beat_valid,
  input  logic               beat_clear,
  input  logic [ROW_W-1:0]   beat_row,

  // stationary-weight double-buffer control (same protocol as the real array)
  input  logic               live_sel,
  input  logic               wload_en,
  input  logic [8*MXE_N-1:0] wload_lanes,

  // accumulator drain read (combinational; stable while en==0)
  input  logic [ROW_W-1:0]   rd_row,
  output acc_t               rd_lanes [MXE_N]
);

  // ── weight banks with the east→west column shift protocol ────────────────
  act_t w_bank0 [MXE_N][MXE_N];
  act_t w_bank1 [MXE_N][MXE_N];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int r = 0; r < int'(MXE_N); r++)
        for (int c = 0; c < int'(MXE_N); c++) begin
          w_bank0[r][c] <= '0;
          w_bank1[r][c] <= '0;
        end
    end else if (wload_en) begin
      for (int r = 0; r < int'(MXE_N); r++) begin
        if (!live_sel) begin       // live bank 0 computes → bank 1 loads
          for (int c = 0; c < int'(MXE_N) - 1; c++)
            w_bank1[r][c] <= w_bank1[r][c+1];
          w_bank1[r][MXE_N-1] <= act_t'(wload_lanes[8*r +: 8]);
        end else begin             // live bank 1 computes → bank 0 loads
          for (int c = 0; c < int'(MXE_N) - 1; c++)
            w_bank0[r][c] <= w_bank0[r][c+1];
          w_bank0[r][MXE_N-1] <= act_t'(wload_lanes[8*r +: 8]);
        end
      end
    end
  end

  // ── broadcast input register (ALL rows same cycle — the "wall") ──────────
  act_t act_q [MXE_N];

  typedef struct packed {
    logic             valid;
    logic             clr;
    logic [ROW_W-1:0] row;
  } tag_t;
  tag_t t0, t1;

  acc_t col_q [MXE_N];

  // whitebox probe aliases (same names/shapes as the real mxe_array):
  //   act_h[r][c]  = the broadcast activation wire (identical for every c)
  //   psum_v[r+1][c] = combinational adder-tree prefix tap (no hops)
  act_t act_h  [MXE_N][MXE_N+1];
  acc_t psum_v [MXE_N+1][MXE_N];

  always_comb begin
    for (int r = 0; r < int'(MXE_N); r++)
      for (int c = 0; c <= int'(MXE_N); c++)
        act_h[r][c] = act_q[r];                       // broadcast: no hops
    for (int c = 0; c < int'(MXE_N); c++)
      psum_v[0][c] = '0;
    for (int r = 0; r < int'(MXE_N); r++)
      for (int c = 0; c < int'(MXE_N); c++)
        psum_v[r+1][c] = psum_v[r][c]
          + acc_t'(16'(act_q[r]) *
                   16'(live_sel ? w_bank1[r][c] : w_bank0[r][c]));
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int r = 0; r < int'(MXE_N); r++) act_q[r] <= '0;
      for (int c = 0; c < int'(MXE_N); c++) col_q[c] <= '0;
      t0 <= '0;
      t1 <= '0;
    end else if (en) begin
      for (int r = 0; r < int'(MXE_N); r++)
        act_q[r] <= act_t'(act_lanes[8*r +: 8]);
      t0 <= '{valid: beat_valid, clr: beat_clear, row: beat_row};
      for (int c = 0; c < int'(MXE_N); c++)
        col_q[c] <= psum_v[MXE_N][c];                 // one result register
      t1 <= t0;
    end
  end

  // ── accumulators: ALL columns written in the SAME cycle ──────────────────
  acc_t acc [MXE_N][M_TILE_MAX];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int c = 0; c < int'(MXE_N); c++)
        for (int m = 0; m < int'(M_TILE_MAX); m++)
          acc[c][m] <= '0;
    end else if (en && t1.valid) begin
      for (int c = 0; c < int'(MXE_N); c++)
        acc[c][t1.row] <= (t1.clr ? acc_t'(0) : acc[c][t1.row]) + col_q[c];
    end
  end

  for (genvar c = 0; c < int'(MXE_N); c++) begin : g_rd
    assign rd_lanes[c] = acc[c][rd_row];
  end

  // broadcast-wall result latency (act_q + col_q stages). Elaboration
  // sanity: must stay well inside the flush window that the real array's
  // tag pipe (TAGP_DEPTH) sizes — otherwise the mutant would stop being
  // functionally equivalent under mxe_ctrl and the equivalence half of the
  // mutation gate would be void.
  localparam int unsigned BCAST_LAT = 2;
  initial begin
    if (BCAST_LAT + 1 >= TAGP_DEPTH)
      $fatal(1, "bcast mutant latency not hidden by the ctrl FLUSH window");
  end

  // consume the probe-only broadcast wires for lint (-Wall, no waivers)
  logic unused_probe;
  always_comb begin
    unused_probe = 1'b0;
    for (int r = 0; r < int'(MXE_N); r++)
      for (int c = 0; c <= int'(MXE_N); c++)
        unused_probe = unused_probe ^ (^act_h[r][c]);
  end

endmodule
