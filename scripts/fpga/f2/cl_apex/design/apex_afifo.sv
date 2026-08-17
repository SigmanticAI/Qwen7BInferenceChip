// apex_afifo.sv — IB-FUEL s3: dual-clock (asynchronous) FIFO, gray-coded
// pointers, 2FF synchronizers. The fuel line's ONE data crossing:
// shell-domain burst reader -> clk_tile xw stream (IB_FUEL.md §4).
//
// Hand-rolled, not XPM, deliberately: f2sim stays pure open-source
// (a comment line must not START with the tool's name — metacomment
// parsing), and the constraint surface stays explicit — the same reasoning
// as apex_ocl_cdc.sv, whose *_meta/*_sync naming this module copies so
// cl_synth_user.xdc can carry uniform ASYNC_REG + max_delay exceptions.
//
// Design (standard Cummings-style):
//   * (LOG2D+1)-bit binary pointers, gray-coded for the crossing; full =
//     wgray equals rgray-synced with the two MSBs inverted; empty = rgray
//     equals wgray-synced.
//   * Storage is a plain reg array with an UNREGISTERED read (FWFT:
//     rvalid/rdata present the head combinationally). The read payload
//     path mem[] -> read-domain consumers is CDC-safe by the pointer
//     protocol (a slot is stable from >=2 synced gray updates before the
//     read side can select it) and carries a datapath_only max_delay in
//     the xdc. At W=64 x DEPTH=512 this infers LUTRAM (~0.5k LUTs) — a
//     registered-BRAM variant is an s5 optimization IF LUT pressure
//     appears in the build report, not before.
//
// RESET RULE (load-bearing, IB_FUEL.md §4): both sides reset ONLY by their
// domain's GLOBAL reset (rst_main_n-derived). Never gate either side with
// the bridge's TILE_RST — resetting one side's pointers while the other
// side keeps counting is pointer-parity corruption by construction. The
// fuel contract keeps the FIFO empty across per-file TILE_RST toggles
// (records are fully consumed per job; the executor postlude checks
// fifo_empty).

`timescale 1ns/1ps

module apex_afifo #(
  parameter int unsigned W     = 64,
  parameter int unsigned LOG2D = 9              // depth = 2**LOG2D
) (
  // ── write side ───────────────────────────────────────────────────────────
  input  logic         wclk,
  input  logic         wrst_n,
  input  logic         wvalid,
  output logic         wready,
  input  logic [W-1:0] wdata,
  // ── read side ────────────────────────────────────────────────────────────
  input  logic         rclk,
  input  logic         rrst_n,
  output logic         rvalid,
  input  logic         rready,
  output logic [W-1:0] rdata
);

  localparam int unsigned D = 1 << LOG2D;

  function automatic logic [LOG2D:0] bin2gray(input logic [LOG2D:0] b);
    bin2gray = b ^ (b >> 1);
  endfunction

  logic [W-1:0] mem [0:D-1];

  // write domain
  logic [LOG2D:0] wptr_bin_q, wptr_gray_q;
  logic [LOG2D:0] rptr_gray_meta, rptr_gray_sync;   // 2FF: read gray -> wclk
  wire  [LOG2D:0] wptr_bin_n  = wptr_bin_q + {{LOG2D{1'b0}}, 1'b1};
  wire            full        = (wptr_gray_q ==
                                 {~rptr_gray_sync[LOG2D:LOG2D-1],
                                   rptr_gray_sync[LOG2D-2:0]});
  assign wready = !full;

  always_ff @(posedge wclk) begin
    if (!wrst_n) begin
      wptr_bin_q     <= '0;
      wptr_gray_q    <= '0;
      rptr_gray_meta <= '0;
      rptr_gray_sync <= '0;
    end else begin
      rptr_gray_meta <= rptr_gray_q_rd;
      rptr_gray_sync <= rptr_gray_meta;
      if (wvalid && wready) begin
        mem[wptr_bin_q[LOG2D-1:0]] <= wdata;
        wptr_bin_q  <= wptr_bin_n;
        wptr_gray_q <= bin2gray(wptr_bin_n);
      end
    end
  end

  // read domain
  logic [LOG2D:0] rptr_bin_q, rptr_gray_q_rd;
  logic [LOG2D:0] wptr_gray_meta, wptr_gray_sync;   // 2FF: write gray -> rclk
  wire  [LOG2D:0] rptr_bin_n = rptr_bin_q + {{LOG2D{1'b0}}, 1'b1};
  wire            empty      = (rptr_gray_q_rd == wptr_gray_sync);
  assign rvalid = !empty;
  assign rdata  = mem[rptr_bin_q[LOG2D-1:0]];

  always_ff @(posedge rclk) begin
    if (!rrst_n) begin
      rptr_bin_q     <= '0;
      rptr_gray_q_rd <= '0;
      wptr_gray_meta <= '0;
      wptr_gray_sync <= '0;
    end else begin
      wptr_gray_meta <= wptr_gray_q;
      wptr_gray_sync <= wptr_gray_meta;
      if (rvalid && rready) begin
        rptr_bin_q     <= rptr_bin_n;
        rptr_gray_q_rd <= bin2gray(rptr_bin_n);
      end
    end
  end

endmodule
