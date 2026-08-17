// apex_gam_unpack.sv — E-7 (E2E_TOY_LANE lane, 2026-08-05): the xw -> xg
// gamma unpacker, the route whose ABSENCE was the measured NORM1/NORM2
// walk fence ("no xw->xg route exists", STEP_MATRIX.md p_norm2 class,
// re-verified by execution before this module was written).
//
// One 64-bit xw beat carries FOUR little-endian int16 Q2.13 gamma values —
// exactly make_weight_image.py gamma_payload's layout (`int16 gamma row,
// little-endian`), so element 4j+i of the fetched G1/G2 tensor is beat j's
// bits [16*i +: 16]. The unpacker shifts them out one per accepted g beat.
//
// WINDOW CONTRACT (the walker's lw_gsrc level, registered in apex_top):
//   * s_ready is high only while the window is OPEN and the hold register
//     is empty — a beat can never be swallowed outside a window, and the
//     MXE weight mux (the xw stream's only other consumer) is held off by
//     the same level, so ownership is exclusive by construction.
//   * The fetch delivers EXACTLY d_model gammas and asu_rmsnorm consumes
//     EXACTLY d_model per row (g "consumed DURING EMISSION, exactly D
//     beats per row" — asu_rmsnorm.sv:104-106), so when the norm's done
//     fires (the walker's S2_NORM exit, which closes the window) cnt is 0
//     and nothing is in flight. busy is exported into the tile's
//     blk_busy[2] anyway (belt over that argument): a mis-sized fetch
//     would hold tile_idle low — loud, never a silent mis-route.
//   * No `last` handling: the fuel beat stream carries last=0 by the WB
//     parity rule (cl_apex fuel_xw_beat.last = 1'b0); framing is the
//     norm's own x-stream contract, not this stream's.
`ifndef APEX_GAM_UNPACK_SV
`define APEX_GAM_UNPACK_SV

module apex_gam_unpack (
  input  logic               clk,
  input  logic               rst_n,      // synchronous, active low

  input  logic               win,        // the registered gamma window level

  // xw beat side (64-bit lane8 beat payload)
  input  logic               s_valid,
  output logic               s_ready,
  input  logic [63:0]        s_data,

  // gamma element side (int16 Q2.13, LE order within each beat)
  output logic               m_valid,
  input  logic               m_ready,
  output logic signed [15:0] m_gamma,

  output logic               busy        // element(s) still buffered
);

  logic [63:0] hold_q;
  logic [2:0]  cnt_q;                    // 0..4 elements remaining

  assign s_ready = win && (cnt_q == 3'd0);
  assign m_valid = (cnt_q != 3'd0);
  assign m_gamma = hold_q[15:0];
  assign busy    = (cnt_q != 3'd0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      hold_q <= '0;
      cnt_q  <= '0;
    end else if (s_valid && s_ready) begin
      hold_q <= s_data;
      cnt_q  <= 3'd4;
    end else if (m_valid && m_ready) begin
      hold_q <= {16'b0, hold_q[63:16]};
      cnt_q  <= cnt_q - 3'd1;
    end
  end

endmodule

`endif // APEX_GAM_UNPACK_SV
