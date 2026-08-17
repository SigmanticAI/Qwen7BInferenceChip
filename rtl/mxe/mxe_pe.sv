// mxe_pe.sv — one weight-stationary processing element of the MXE systolic
// array.
//
// Implements: D-004 (TRUE systolic: PE-to-PE pipelining, 1 cycle/hop, no
//             broadcast), D-005 (double-buffered stationary weight: the live
//             bank computes while the shadow bank loads via a column shift
//             chain), C-3 (INT32 partial sums; INT8xINT8 products).
//
// Dataflow (per ARCHITECTURE.md §3):
//   - holds ONE INT8 weight per bank (bank 0 / bank 1). `live_sel` picks the
//     computing bank; the OTHER bank (~live_sel) is the load target of the
//     row's weight shift chain (wload_in -> wload_out, east -> west).
//   - INT8 activation enters from the WEST (act_in) and is registered one
//     cycle to the EAST (act_out).
//   - INT32 partial sum enters from the NORTH (psum_in); the PE emits
//     psum_in + act_in*w_live one cycle later to the SOUTH (psum_out).
//   - `en` advances the compute pipeline (act/psum regs); weight loading
//     (`wload_en`) is independent so bank ~live_sel can load while the
//     compute pipeline is frozen or running (D-005).
//
// Numerics: |act*w| <= 128*128 = 2^14, held exactly in a signed 16-bit
// product, then sign-extended into the INT32 partial-sum add. With the
// legality cap K <= 2048 (D-003) the column sum can never overflow INT32.

module mxe_pe
  import apex_pkg::*;
(
  input  logic clk,
  input  logic rst_n,      // synchronous, active-low

  // compute pipeline
  input  logic en,         // advance act/psum registers
  input  act_t act_in,     // west
  output act_t act_out,    // east (registered, 1 cycle/hop)
  input  acc_t psum_in,    // north
  output acc_t psum_out,   // south (registered, 1 cycle/hop)

  // stationary weight, double-buffered (D-005)
  input  logic live_sel,   // bank that COMPUTES; ~live_sel is loading
  input  logic wload_en,   // shift strobe for the loading bank
  input  act_t wload_in,   // from east neighbour's wload_out (or array edge)
  output act_t wload_out   // this PE's loading-bank value, to west neighbour
);

  act_t w_bank [2];
  act_t w_live;
  logic signed [15:0] prod;

  assign w_live    = w_bank[live_sel];
  assign wload_out = w_bank[!live_sel];

  // exact signed 8x8 product in 16 bits (|p| <= 2^14 < 2^15)
  assign prod = 16'(act_in) * 16'(w_live);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w_bank[0] <= '0;
      w_bank[1] <= '0;
    end else if (wload_en) begin
      w_bank[!live_sel] <= wload_in;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      act_out  <= '0;
      psum_out <= '0;
    end else if (en) begin
      act_out  <= act_in;
      psum_out <= psum_in + acc_t'(prod);
    end
  end

endmodule
