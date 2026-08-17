// apex_ocl_cdc.sv — OCL AXI-Lite clock-domain-crossing bridge (DECISION-LC-1).
//
// Sits between the F2 shell's OCL AXI-Lite (clk_main_a0, fixed 250 MHz) and
// the cl_apex body (FSM + mailbox + tile) which moves wholesale onto the
// slow tile clock. The downstream FSM is STRICTLY single-outstanding
// (cl_apex.sv: writes-then-reads, one at a time, hang-guarded, always
// completes), so a 4-phase toggle req/ack bridge is sufficient and fully
// verifiable in 2-state simulation; metastability protection is the build's
// job (ASYNC_REG on the *_meta/*_sync pairs + false_path/max_delay on the
// payload buses — see cl_apex_cdc.xdc notes).
//
// Protocol per transaction (shell side):
//   1. Accept ONE transaction: AW+W together (write wins arbitration,
//      mirroring the downstream FSM) or AR. Latch {is_write, addr, wdata,
//      wstrb} — the payload is stable BEFORE req toggles and until ack
//      returns (safe multi-bit crossing: payload settled while req in
//      flight).
//   2. Toggle req. Tile side syncs req (2FF), runs the transaction against
//      the downstream FSM using its exact accept protocol (awvalid+wvalid
//      asserted together; single-beat), captures {resp, rdata}, toggles ack.
//   3. Shell side syncs ack (2FF), returns B or R to the shell.
//
// Hang behavior: the downstream FSM always completes (its own guards return
// SLVERR), so the shell-side guard here (2^GUARD_W shell cycles) fires ONLY
// if the tile clock is dead (stopped MMCM / tile reset held). After a guard
// fire the bridge latches `poisoned` and answers every transaction SLVERR
// until rst_shell_n — req/ack toggle parity can no longer be trusted, and a
// dead tile domain is not a recoverable-in-band condition. The host remedy
// is a global reset. This keeps the shell bus live (never wedges the PF).
//
// Reset: both domains reset asynchronously-asserted/synchronously-released
// by their own synchronized copies of rst_main_n (generated in cl_apex).

`timescale 1ns/1ps

module apex_ocl_cdc #(
  parameter int unsigned GUARD_W = 16   // shell-cycle guard (~262 us @250MHz)
) (
  // ── shell side (clk_shell = clk_main_a0) ─────────────────────────────────
  input  logic        clk_shell,
  input  logic        rst_shell_n,
  input  logic        s_awvalid,
  input  logic [31:0] s_awaddr,
  output logic        s_awready,
  input  logic        s_wvalid,
  input  logic [31:0] s_wdata,
  input  logic [3:0]  s_wstrb,
  output logic        s_wready,
  output logic        s_bvalid,
  output logic [1:0]  s_bresp,
  input  logic        s_bready,
  input  logic        s_arvalid,
  input  logic [31:0] s_araddr,
  output logic        s_arready,
  output logic        s_rvalid,
  output logic [31:0] s_rdata,
  output logic [1:0]  s_rresp,
  input  logic        s_rready,

  // ── tile side (clk_tile) — mirrors the shell OCL toward the cl_apex FSM ──
  input  logic        clk_tile,
  input  logic        rst_tile_n,
  output logic        t_awvalid,
  output logic [31:0] t_awaddr,
  input  logic        t_awready,
  output logic        t_wvalid,
  output logic [31:0] t_wdata,
  output logic [3:0]  t_wstrb,
  input  logic        t_wready,
  input  logic        t_bvalid,
  input  logic [1:0]  t_bresp,
  output logic        t_bready,
  output logic        t_arvalid,
  output logic [31:0] t_araddr,
  input  logic        t_arready,
  input  logic        t_rvalid,
  input  logic [31:0] t_rdata,
  input  logic [1:0]  t_rresp,
  output logic        t_rready
);

  // ── shell-domain state ────────────────────────────────────────────────────
  typedef enum logic [1:0] { S_IDLE, S_WAIT, S_BRESP, S_RRESP } s_st_e;
  s_st_e              s_st;
  logic               req_tgl;        // toggles once per transaction
  logic               is_wr_q;
  logic [31:0]        addr_q, wdat_q;
  logic [3:0]         wstrb_q;
  logic               poisoned;
  logic [GUARD_W-1:0] guard_q;

  // ack toggle, synchronized into shell domain (build: ASYNC_REG pair)
  logic ack_tgl;                      // tile-domain source
  logic ack_meta, ack_sync, ack_seen;

  // response payload (tile-domain regs, stable from ack toggle until next
  // req — sampled here only after ack_sync edge; build: false_path/max_delay)
  logic [31:0] resp_rdata;
  logic [1:0]  resp_code;

  wire ack_edge = (ack_sync != ack_seen);

  logic [31:0] s_rdata_q;
  logic [1:0]  s_rresp_q;

  // poisoned still ACCEPTS transactions (AXI: response only after accept)
  // and answers them SLVERR locally, never touching the dead tile domain
  assign s_awready = (s_st == S_IDLE) && s_awvalid && s_wvalid;
  assign s_wready  = s_awready;
  assign s_arready = (s_st == S_IDLE) &&
                     !(s_awvalid && s_wvalid) && s_arvalid;
  assign s_bvalid  = (s_st == S_BRESP);
  assign s_bresp   = s_rresp_q;       // shared response register
  assign s_rvalid  = (s_st == S_RRESP);
  assign s_rdata   = s_rdata_q;
  assign s_rresp   = s_rresp_q;

  always_ff @(posedge clk_shell) begin
    if (!rst_shell_n) begin
      s_st          <= S_IDLE;
      req_tgl       <= 1'b0;
      is_wr_q       <= 1'b0;
      addr_q        <= '0;
      wdat_q        <= '0;
      wstrb_q       <= '0;
      poisoned      <= 1'b0;
      guard_q       <= '0;
      ack_meta      <= 1'b0;
      ack_sync      <= 1'b0;
      ack_seen      <= 1'b0;
      s_rdata_q     <= '0;
      s_rresp_q     <= 2'b00;
    end else begin
      ack_meta <= ack_tgl;
      ack_sync <= ack_meta;

      unique case (s_st)
        S_IDLE: begin
          guard_q <= '0;
          if (s_awvalid && s_wvalid) begin            // accepted this cycle
            if (poisoned) begin
              s_rresp_q <= 2'b10;                     // local SLVERR
              s_st      <= S_BRESP;
            end else begin
              is_wr_q <= 1'b1;
              addr_q  <= s_awaddr;
              wdat_q  <= s_wdata;
              wstrb_q <= s_wstrb;
              req_tgl <= ~req_tgl;
              s_st    <= S_WAIT;
            end
          end else if (s_arvalid) begin               // accepted this cycle
            if (poisoned) begin
              s_rresp_q <= 2'b10;
              s_rdata_q <= 32'hDEAD_CDC0;
              s_st      <= S_RRESP;
            end else begin
              is_wr_q <= 1'b0;
              addr_q  <= s_araddr;
              req_tgl <= ~req_tgl;
              s_st    <= S_WAIT;
            end
          end
        end

        S_WAIT: begin
          guard_q <= guard_q + {{(GUARD_W-1){1'b0}}, 1'b1};
          if (ack_edge) begin
            ack_seen  <= ack_sync;
            s_rdata_q <= resp_rdata;    // stable since before ack toggled
            s_rresp_q <= resp_code;
            s_st      <= is_wr_q ? S_BRESP : S_RRESP;
          end else if (&guard_q) begin  // tile domain dead
            poisoned  <= 1'b1;
            s_rresp_q <= 2'b10;
            s_rdata_q <= 32'hDEAD_CDC0;
            s_st      <= is_wr_q ? S_BRESP : S_RRESP;
          end
        end

        S_BRESP: if (s_bready) s_st <= S_IDLE;
        S_RRESP: if (s_rready) s_st <= S_IDLE;

        default: s_st <= S_IDLE;
      endcase
    end
  end

  // ── tile-domain state ─────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    T_IDLE, T_WR_ADDR, T_WR_B, T_RD_AR, T_RD_R, T_ACK
  } t_st_e;
  t_st_e t_st;

  logic req_meta, req_sync, req_seen;
  logic aw_done_q, w_done_q;

  wire req_edge = (req_sync != req_seen);

  assign t_awaddr  = addr_q;          // stable while req in flight (CDC-safe)
  assign t_araddr  = addr_q;
  assign t_wdata   = wdat_q;
  assign t_wstrb   = wstrb_q;
  assign t_awvalid = (t_st == T_WR_ADDR) && !aw_done_q;
  assign t_wvalid  = (t_st == T_WR_ADDR) && !w_done_q;
  assign t_bready  = (t_st == T_WR_B);
  assign t_arvalid = (t_st == T_RD_AR);
  assign t_rready  = (t_st == T_RD_R);

  always_ff @(posedge clk_tile) begin
    if (!rst_tile_n) begin
      t_st       <= T_IDLE;
      req_meta   <= 1'b0;
      req_sync   <= 1'b0;
      req_seen   <= 1'b0;
      ack_tgl    <= 1'b0;
      aw_done_q  <= 1'b0;
      w_done_q   <= 1'b0;
      resp_rdata <= '0;
      resp_code  <= 2'b00;
    end else begin
      req_meta <= req_tgl;
      req_sync <= req_meta;

      unique case (t_st)
        T_IDLE: if (req_edge) begin
          req_seen  <= req_sync;
          aw_done_q <= 1'b0;
          w_done_q  <= 1'b0;
          t_st      <= is_wr_q ? T_WR_ADDR : T_RD_AR;
        end

        // downstream FSM accepts AW+W together (awready = awvalid && wvalid),
        // but tolerate split acceptance anyway
        T_WR_ADDR: begin
          if (t_awvalid && t_awready) aw_done_q <= 1'b1;
          if (t_wvalid  && t_wready)  w_done_q  <= 1'b1;
          if ((aw_done_q || (t_awvalid && t_awready)) &&
              (w_done_q  || (t_wvalid  && t_wready)))
            t_st <= T_WR_B;
        end
        T_WR_B: if (t_bvalid) begin
          resp_code <= t_bresp;
          t_st      <= T_ACK;
        end

        T_RD_AR: if (t_arready) t_st <= T_RD_R;
        T_RD_R:  if (t_rvalid) begin
          resp_rdata <= t_rdata;
          resp_code  <= t_rresp;
          t_st       <= T_ACK;
        end

        // separate state: resp_* registers settle one full tile cycle before
        // ack toggles — the shell side then syncs ack through 2FF, so the
        // payload is long-stable before it is sampled (the max_delay bound)
        T_ACK: begin
          ack_tgl <= ~ack_tgl;
          t_st    <= T_IDLE;
        end

        default: t_st <= T_IDLE;
      endcase
    end
  end

endmodule
