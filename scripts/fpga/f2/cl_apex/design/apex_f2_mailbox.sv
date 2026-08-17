// apex_f2_mailbox.sv — F2 stage-2 mailbox: the host drives the APEX tile's
// streams, job pulses, and route levels one register access at a time, and
// pops output beats back — turning the F2 host program into the L3
// testbench (BRINGUP.md §2.3½). Throughput is irrelevant by design: the
// stage-2 goal is bit-exact replay of the committed S8 trace on silicon.
//
// ⚠ STATUS: stage-2 RTL, Verilator-linted, NOT yet simulated or built —
// same discipline as the stage-1 CL (which then built cleanly): every claim
// waits for logs. The tile side is the verified apex_top contract; this
// file is CL-side plumbing only.
//
// Register map (byte addr inside window 0x3___, all 32-bit):
//   -- input streams (write pushes; read returns {24'd free_slots, 8'h0}) --
//   0x000 XA_PUSH    [8]=last, [7:0]=x (INT8 activation)
//   0x010 XG_PUSH    [15:0]=gamma (Q2.13)
//   0x020 QS_PUSH    [31:0] scale_quant sideband word
//   0x030 CS_PUSH    [31:0] score-dequant composite word
//   0x040 XW_ASM0    lane8 data[31:0]      (assembly regs, write-only)
//   0x044 XW_ASM1    lane8 data[63:32]
//   0x048 XW_PUSH    [0]=last → pushes {last, ASM1, ASM0}
//   0x080 DS_ASM0..3 mxe_desc_t[31:0]..[127:96]   (0x080/84/88/8C)
//   0x090 DS_PUSH    any write pushes the assembled descriptor
//   -- job pulses (write fires; read returns {31'd0, ready}) --------------
//   0x100 FJ_FIRE    [11:0]=rows
//   0x104 QJ_FIRE    [12]=mode, [11:0]=cols
//   0x108 DJ_FIRE    [11:0]=cols
//   0x10C LJ_FIRE    [11:0]={lanes[3:0], beats[7:0]}
//   0x110 AJ_FIRE    {sel[4:0], nb[NBW-1:0], rows[4:0], pat[1:0], bank, op}
//   0x114 WJ_FIRE    same packing as AJ_FIRE
//   -- route levels (RW registers, change only while paths idle) ----------
//   0x140 ROUTE0     {tip_blk[6:0], kv_user, squant_src, res_dst[1:0],
//                     wgt_src, act_src, feeder_dst, feeder_src}  [14:0]
//   0x144 ROUTE1     {imp_hi[15:0], imp_lo[15:0]}
//   0x148 IMP_CLEAR  write 1 = one-cycle rt_imp_clear pulse
//   -- output captures (STATUS read; W0..: head peek; ADV write pops) -----
//   0x200 RO_STATUS  {23'd count, valid, 8'h0}   0x204..0x220 RO_W0..W7
//   0x224 RO_META    [0]=last                    0x228 RO_ADV
//   0x240 FS_STATUS  same shape                  0x244 FS_W0 {last,data16}
//   0x248 FS_ADV
//   0x250 SS_STATUS                              0x254 SS_W0   0x258 SS_ADV
//   0x260 TD_STATUS                              0x264 TD_W0
//                                                {blk[6:0], tier[1:0], fp16}
//   0x268 TD_ADV
//   0x2F0 MB_STATUS  sticky {ovfl_out, ovfl_in} (W1C by writing 0x2F0)
//
// Contract: the host NEVER pushes into a full FIFO (check free first) and
// never fires a job whose ready is 0 — violations set sticky MB_STATUS bits
// and the access is dropped (auditable, same philosophy as SB_OVWR).

`timescale 1ns/1ps

module apex_f2_mailbox
  import apex_pkg::*;
#(
  parameter int unsigned NBW = 4          // stage-buf nb field width (D=64)
)(
  input  wire         clk,
  input  wire         rst_n,

  // register bus from the cl_apex OCL FSM (window 0x3): 1-cycle strobes,
  // rdata valid combinationally in the strobe cycle
  input  wire  [11:0] mb_addr,
  input  wire  [31:0] mb_wdata,
  input  wire         mb_write,
  input  wire         mb_read,
  output logic [31:0] mb_rdata,

  // ── tile stream masters ─────────────────────────────────────────────────
  output logic        xa_valid,
  input  wire         xa_ready,
  output logic signed [7:0] xa_x,
  output logic        xa_last,
  output logic        xg_valid,
  input  wire         xg_ready,
  output logic signed [15:0] xg_gamma,
  output logic        qs_valid,
  input  wire         qs_ready,
  output logic [31:0] qs_data,
  output logic        cs_valid,
  input  wire         cs_ready,
  output logic [31:0] cs_data,
  output logic        xw_valid,
  input  wire         xw_ready,
  output lane8_beat_t xw_beat,
  output logic        ds_valid,
  input  wire         ds_ready,
  output mxe_desc_t   ds_desc,

  // ── job pulses ──────────────────────────────────────────────────────────
  output logic        fj_valid,
  input  wire         fj_ready,
  output logic [11:0] fj_rows,
  output logic        qj_valid,
  input  wire         qj_ready,
  output logic        qj_mode,
  output logic [11:0] qj_cols,
  output logic        dj_valid,
  input  wire         dj_ready,
  output logic [11:0] dj_cols,
  output logic        lj_valid,
  input  wire         lj_ready,
  output logic [7:0]  lj_beats,
  output logic [3:0]  lj_lanes,
  output logic        aj_valid,
  input  wire         aj_ready,
  output logic        aj_op,
  output logic        aj_bank,
  output logic [1:0]  aj_pat,
  output logic [4:0]  aj_rows,
  output logic [NBW-1:0] aj_nb,
  output logic [4:0]  aj_sel,
  output logic        wj_valid,
  input  wire         wj_ready,
  output logic        wj_op,
  output logic        wj_bank,
  output logic [1:0]  wj_pat,
  output logic [4:0]  wj_rows,
  output logic [NBW-1:0] wj_nb,
  output logic [4:0]  wj_sel,

  // ── route levels ────────────────────────────────────────────────────────
  output logic        rt_feeder_src,
  output logic        rt_feeder_dst,
  output logic        rt_act_src,
  output logic        rt_wgt_src,
  output logic [1:0]  rt_res_dst,
  output logic        rt_squant_src,
  output logic        rt_kv_user,
  output logic [6:0]  rt_tip_blk,
  output logic [15:0] rt_imp_hi,
  output logic [15:0] rt_imp_lo,
  output logic        rt_imp_clear,

  // ── tile output slaves (always-accept into capture FIFOs) ───────────────
  input  wire         ro_valid,
  output logic        ro_ready,
  input  lane32_beat_t ro_beat,
  input  wire         fs_valid,
  output logic        fs_ready,
  input  wire  [15:0] fs_data,
  input  wire         fs_last,
  input  wire         ss_valid,
  output logic        ss_ready,
  input  wire  [15:0] ss_data,
  input  wire         ss_last,
  input  wire         td_valid,
  output logic        td_ready,
  input  wire         td_fp16,
  input  kvq_tier_e   td_tier,
  input  wire  [6:0]  td_blk
);

  localparam int unsigned DEPTH = 16;
  localparam int unsigned PW    = $clog2(DEPTH);

  // ── tiny sync FIFO (per stream) as a macro-ish generate pattern ────────
  // in-FIFOs: host pushes, tile drains via valid/ready.
  // out-FIFOs: tile pushes (always-accept while not full), host peeks+pops.

  // Generic storage
  typedef struct packed { logic ovfl; } mb_err_t;
  logic ovfl_in_q, ovfl_out_q;

`define MB_INFIFO(NAME, W)                                                    \
  logic [W-1:0] NAME``_mem [0:DEPTH-1];                                       \
  logic [PW:0]  NAME``_wp, NAME``_rp;                                         \
  wire  [PW:0]  NAME``_cnt  = NAME``_wp - NAME``_rp;                          \
  wire          NAME``_full = (NAME``_cnt == DEPTH[PW:0]);                    \
  wire          NAME``_emp  = (NAME``_cnt == '0);                             \
  wire [W-1:0]  NAME``_head = NAME``_mem[NAME``_rp[PW-1:0]];

  `MB_INFIFO(xa, 9)                       // {last, x[7:0]}
  `MB_INFIFO(xg, 16)
  `MB_INFIFO(qs, 32)
  `MB_INFIFO(cs, 32)
  `MB_INFIFO(xw, 65)                      // {last, data[63:0]}
  `MB_INFIFO(ds, 128)
  `MB_INFIFO(ro, 257)                     // {last, data[255:0]} (capture)
  `MB_INFIFO(fs, 17)                      // {last, data[15:0]}  (capture)
  `MB_INFIFO(ss, 17)                      // (capture)
  `MB_INFIFO(td, 10)                      // {blk[6:0], tier[1:0], fp16}

  // assembly registers
  logic [63:0]  xw_asm_q;
  logic [127:0] ds_asm_q;

  // job fire pulse registers (1-cycle valids; fields latched at fire)
  // (declared as module outputs above; driven in the clocked block)

  // ── input-stream drains (FIFO head -> tile) ─────────────────────────────
  assign xa_valid = !xa_emp;
  assign {xa_last, xa_x} = xa_head;
  assign xg_valid = !xg_emp;
  assign xg_gamma = xg_head;
  assign qs_valid = !qs_emp;
  assign qs_data  = qs_head;
  assign cs_valid = !cs_emp;
  assign cs_data  = cs_head;
  assign xw_valid = !xw_emp;
  // lane8_beat_t packs {data[63:0], last} — data in [64:1], last in [0].
  // The FIFO word is stored {last, data}, so it must be re-ordered, not cast.
  assign xw_beat  = lane8_beat_t'({xw_head[63:0], xw_head[64]});
  assign ds_valid = !ds_emp;
  assign ds_desc  = mxe_desc_t'(ds_head);

  // ── output-stream captures (tile -> FIFO, accept while not full) ───────
  assign ro_ready = !ro_full;
  assign fs_ready = !fs_full;
  assign ss_ready = !ss_full;
  assign td_ready = !td_full;

  // ── register read mux (combinational, valid in the strobe cycle) ───────
  function automatic logic [31:0] stat(input logic [PW:0] cnt,
                                       input logic nonempty);
    stat = {8'h0, 15'(DEPTH - cnt), nonempty, 8'h0};
  endfunction

  always_comb begin
    unique case (mb_addr)
      12'h000: mb_rdata = {8'h0, 15'(DEPTH - xa_cnt), 1'b0, 8'h0};
      12'h010: mb_rdata = {8'h0, 15'(DEPTH - xg_cnt), 1'b0, 8'h0};
      12'h020: mb_rdata = {8'h0, 15'(DEPTH - qs_cnt), 1'b0, 8'h0};
      12'h030: mb_rdata = {8'h0, 15'(DEPTH - cs_cnt), 1'b0, 8'h0};
      12'h048: mb_rdata = {8'h0, 15'(DEPTH - xw_cnt), 1'b0, 8'h0};
      12'h090: mb_rdata = {8'h0, 15'(DEPTH - ds_cnt), 1'b0, 8'h0};
      12'h100: mb_rdata = {31'd0, fj_ready};
      12'h104: mb_rdata = {31'd0, qj_ready};
      12'h108: mb_rdata = {31'd0, dj_ready};
      12'h10C: mb_rdata = {31'd0, lj_ready};
      12'h110: mb_rdata = {31'd0, aj_ready};
      12'h114: mb_rdata = {31'd0, wj_ready};
      12'h140: mb_rdata = {17'd0, rt_tip_blk, rt_kv_user, rt_squant_src,
                           rt_res_dst, rt_wgt_src, rt_act_src,
                           rt_feeder_dst, rt_feeder_src};
      12'h144: mb_rdata = {rt_imp_hi, rt_imp_lo};
      12'h200: mb_rdata = stat(ro_cnt, !ro_emp);
      12'h204: mb_rdata = ro_head[31:0];
      12'h208: mb_rdata = ro_head[63:32];
      12'h20C: mb_rdata = ro_head[95:64];
      12'h210: mb_rdata = ro_head[127:96];
      12'h214: mb_rdata = ro_head[159:128];
      12'h218: mb_rdata = ro_head[191:160];
      12'h21C: mb_rdata = ro_head[223:192];
      12'h220: mb_rdata = ro_head[255:224];
      12'h224: mb_rdata = {31'd0, ro_head[256]};
      12'h240: mb_rdata = stat(fs_cnt, !fs_emp);
      12'h244: mb_rdata = {15'd0, fs_head[16], fs_head[15:0]};
      12'h250: mb_rdata = stat(ss_cnt, !ss_emp);
      12'h254: mb_rdata = {15'd0, ss_head[16], ss_head[15:0]};
      12'h260: mb_rdata = stat(td_cnt, !td_emp);
      12'h264: mb_rdata = {22'd0, td_head};
      12'h2F0: mb_rdata = {30'd0, ovfl_out_q, ovfl_in_q};
      default: mb_rdata = {20'hDEAD3, mb_addr};
    endcase
  end

  // ── clocked side: pushes, pops, fires, routes ──────────────────────────
`define MB_PUSH(NAME, EXPR)                                                   \
  begin                                                                       \
    if (!NAME``_full) begin                                                   \
      NAME``_mem[NAME``_wp[PW-1:0]] <= (EXPR);                                \
      NAME``_wp <= NAME``_wp + 1'b1;                                          \
    end else ovfl_in_q <= 1'b1;                                               \
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      xa_wp <= '0; xa_rp <= '0; xg_wp <= '0; xg_rp <= '0;
      qs_wp <= '0; qs_rp <= '0; cs_wp <= '0; cs_rp <= '0;
      xw_wp <= '0; xw_rp <= '0; ds_wp <= '0; ds_rp <= '0;
      ro_wp <= '0; ro_rp <= '0; fs_wp <= '0; fs_rp <= '0;
      ss_wp <= '0; ss_rp <= '0; td_wp <= '0; td_rp <= '0;
      xw_asm_q <= '0; ds_asm_q <= '0;
      ovfl_in_q <= 1'b0; ovfl_out_q <= 1'b0;
      fj_valid <= 1'b0; qj_valid <= 1'b0; dj_valid <= 1'b0;
      lj_valid <= 1'b0; aj_valid <= 1'b0; wj_valid <= 1'b0;
      fj_rows <= '0; qj_mode <= 1'b0; qj_cols <= '0; dj_cols <= '0;
      lj_beats <= '0; lj_lanes <= '0;
      aj_op <= 1'b0; aj_bank <= 1'b0; aj_pat <= '0; aj_rows <= '0;
      aj_nb <= '0; aj_sel <= '0;
      wj_op <= 1'b0; wj_bank <= 1'b0; wj_pat <= '0; wj_rows <= '0;
      wj_nb <= '0; wj_sel <= '0;
      rt_feeder_src <= 1'b0; rt_feeder_dst <= 1'b0; rt_act_src <= 1'b0;
      rt_wgt_src <= 1'b0; rt_res_dst <= '0; rt_squant_src <= 1'b0;
      rt_kv_user <= 1'b0; rt_tip_blk <= '0;
      // match the golden driver's power-on levels (tb_apex_l3.sv:819-820:
      // "legacy default: suggestions never reach CQ8"). Scripts that care
      // write ROUTE1 explicitly; those that do not must see the TB's default.
      rt_imp_hi <= 16'hFFFF; rt_imp_lo <= 16'h0000; rt_imp_clear <= 1'b0;
    end else begin
      rt_imp_clear <= 1'b0;                        // 1-cycle pulse default

      // input-stream drains
      if (xa_valid && xa_ready) xa_rp <= xa_rp + 1'b1;
      if (xg_valid && xg_ready) xg_rp <= xg_rp + 1'b1;
      if (qs_valid && qs_ready) qs_rp <= qs_rp + 1'b1;
      if (cs_valid && cs_ready) cs_rp <= cs_rp + 1'b1;
      if (xw_valid && xw_ready) xw_rp <= xw_rp + 1'b1;
      if (ds_valid && ds_ready) ds_rp <= ds_rp + 1'b1;

      // output-stream captures
      if (ro_valid && ro_ready) `MB_PUSH(ro, {ro_beat.last, ro_beat.data})
      if (fs_valid && fs_ready) `MB_PUSH(fs, {fs_last, fs_data})
      if (ss_valid && ss_ready) `MB_PUSH(ss, {ss_last, ss_data})
      if (td_valid && td_ready) `MB_PUSH(td, {td_blk, 2'(td_tier), td_fp16})

      // job valid handshakes (held until ready per stream contract)
      if (fj_valid && fj_ready) fj_valid <= 1'b0;
      if (qj_valid && qj_ready) qj_valid <= 1'b0;
      if (dj_valid && dj_ready) dj_valid <= 1'b0;
      if (lj_valid && lj_ready) lj_valid <= 1'b0;
      if (aj_valid && aj_ready) aj_valid <= 1'b0;
      if (wj_valid && wj_ready) wj_valid <= 1'b0;

      // host writes
      if (mb_write) begin
        unique case (mb_addr)
          12'h000: `MB_PUSH(xa, mb_wdata[8:0])
          12'h010: `MB_PUSH(xg, mb_wdata[15:0])
          12'h020: `MB_PUSH(qs, mb_wdata)
          12'h030: `MB_PUSH(cs, mb_wdata)
          12'h040: xw_asm_q[31:0]  <= mb_wdata;
          12'h044: xw_asm_q[63:32] <= mb_wdata;
          12'h048: `MB_PUSH(xw, {mb_wdata[0], xw_asm_q})
          12'h080: ds_asm_q[31:0]    <= mb_wdata;
          12'h084: ds_asm_q[63:32]   <= mb_wdata;
          12'h088: ds_asm_q[95:64]   <= mb_wdata;
          12'h08C: ds_asm_q[127:96]  <= mb_wdata;
          12'h090: `MB_PUSH(ds, ds_asm_q)
          // A fire is accepted when no pulse is pending OR the pending one is
          // being consumed this very cycle (valid && ready) — the later
          // assignment wins over the clear above, so the pulse is never lost.
          // Only a write onto a pending-and-unconsumed pulse is a contract
          // violation: dropped and flagged in MB_STATUS.
          12'h100: if (!fj_valid || fj_ready) begin fj_valid <= 1'b1;
                     fj_rows <= mb_wdata[11:0]; end
                   else ovfl_in_q <= 1'b1;
          12'h104: if (!qj_valid || qj_ready) begin qj_valid <= 1'b1;
                     qj_mode <= mb_wdata[12]; qj_cols <= mb_wdata[11:0]; end
                   else ovfl_in_q <= 1'b1;
          12'h108: if (!dj_valid || dj_ready) begin dj_valid <= 1'b1;
                     dj_cols <= mb_wdata[11:0]; end
                   else ovfl_in_q <= 1'b1;
          12'h10C: if (!lj_valid || lj_ready) begin lj_valid <= 1'b1;
                     lj_lanes <= mb_wdata[11:8]; lj_beats <= mb_wdata[7:0]; end
                   else ovfl_in_q <= 1'b1;
          12'h110: if (!aj_valid || aj_ready) begin aj_valid <= 1'b1;
                     {aj_sel, aj_nb, aj_rows, aj_pat, aj_bank, aj_op}
                       <= mb_wdata[13+NBW:0]; end
                   else ovfl_in_q <= 1'b1;
          12'h114: if (!wj_valid || wj_ready) begin wj_valid <= 1'b1;
                     {wj_sel, wj_nb, wj_rows, wj_pat, wj_bank, wj_op}
                       <= mb_wdata[13+NBW:0]; end
                   else ovfl_in_q <= 1'b1;
          12'h140: {rt_tip_blk, rt_kv_user, rt_squant_src, rt_res_dst,
                    rt_wgt_src, rt_act_src, rt_feeder_dst, rt_feeder_src}
                     <= mb_wdata[14:0];
          12'h144: {rt_imp_hi, rt_imp_lo} <= mb_wdata;
          12'h148: rt_imp_clear <= mb_wdata[0];
          12'h228: if (!ro_emp) ro_rp <= ro_rp + 1'b1; else ovfl_out_q <= 1'b1;
          12'h248: if (!fs_emp) fs_rp <= fs_rp + 1'b1; else ovfl_out_q <= 1'b1;
          12'h258: if (!ss_emp) ss_rp <= ss_rp + 1'b1; else ovfl_out_q <= 1'b1;
          12'h268: if (!td_emp) td_rp <= td_rp + 1'b1; else ovfl_out_q <= 1'b1;
          12'h2F0: begin ovfl_in_q <= 1'b0; ovfl_out_q <= 1'b0; end
          default: ;                      // unknown mailbox write: ignored
        endcase
      end
    end
  end

`undef MB_INFIFO
`undef MB_PUSH

endmodule
