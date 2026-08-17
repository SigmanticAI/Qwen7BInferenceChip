// apex_kvq_bank.sv — in-tile KVQ TIER BANK: three verified, UNMODIFIED
// kvq_engine instances (one per per-channel INT4 codec tier) behind a live tier-select
// mux. This is the F-2/D-022 closure mechanism (D-024): kvq_engine's TIER is
// a synthesis parameter (VAL_BPV, the KEYG grouped-key generate, the record
// layout and the CR_* info registers are all elaborated from it), so runtime
// tier control is realized by BANKING verified engines, not by editing one.
//
//   e0: TIER=0 (CQ-8)  — per-token INT8 K+V, value path only
//   e1: TIER=1 (CQ-4)  — per-channel grouped INT4 keys, INT4 values
//   e2: TIER=2 (CQ-4+) — CQ-4 plus the static OUTLIER_K/MASK_FILE fp16 lane.
//       With OUTLIER_K=0 (no mask supplied) e2 still elaborates but
//       degenerates to CQ-4 semantics — the tile CSR INFO_TIER truthfully
//       drops the CQ-4+ bit in that build (see apex_top).
//
// ROUTING CONTRACT (granularity documented honestly — D-024):
//   tier_sel is a quasi-static LEVEL. Every interface of the bank — the
//   AXI-Lite window, the s_axis write stream (+tuser key/value select), the
//   m_axis read bus, flush_req — routes to engine[tier_sel] COMBINATIONALLY.
//   The host/tile may change tier_sel only while the bank is quiescent
//   (selected engine STATUS.idle polled, no m_axis beat pending): tier
//   granularity is therefore PER STORE RUN (a WRITE_ADDR-based token/group
//   sequence) and PER READ — per-token for values, per-group for keys,
//   per-block in the apex_top TIP-auto mode. Mid-stream tier switches are
//   host contract violations (same class as rt_* route levels).
//   Each engine keeps its own SRAM/address space, CTRL.enable and D-020
//   soft reset: the host enables/resets each tier it uses (3 CTRL writes).
//
// Non-selected engines see valid=0 on every input and ready=0 on m_axis;
// their registered outputs are ignored by the muxes. irq is the OR of the
// three engines (each is maskable via its own IRQ_MASK); evict_needed /
// evict_addr follow the selected engine.
//
// Every port of every engine is driven/consumed — full -Wall, no waivers on
// this file (vendored-file waivers stay scoped to the engines' internals).

`default_nettype none

module apex_kvq_bank
  import apex_pkg::*;
#(
  parameter int unsigned CFG_D     = 64,   // VECTOR_DIM
  parameter int unsigned KVQ_G     = 128,  // KEY_GROUP (e1/e2)
  parameter int unsigned KVQ_DEPTH = 128,  // SRAM records per engine
  parameter int unsigned KVQ_SETS  = 4,    // D-026 persistent scale-bank sets:
                                           // every key group whose records are
                                           // still read back must keep its set
                                           // live (commit allocator wraps with
                                           // SB_OVWR fault past this many)
  parameter int unsigned OUTLIER_K = 0,    // CQ-4+ fp16 lanes (e2 only)
  parameter              MASK_FILE = ""    // outlier-mask ROM hex (e2 only)
)(
  input  wire                          clk,
  input  wire                          rst_n,

  // live tier select (quasi-static; see routing contract above)
  input  wire kvq_tier_e               tier_sel,

  // AXI-Lite window (routed to engine[tier_sel])
  input  wire  [7:0]                   axil_awaddr,
  input  wire                          axil_awvalid,
  output logic                         axil_awready,
  input  wire  [31:0]                  axil_wdata,
  input  wire                          axil_wvalid,
  output logic                         axil_wready,
  output logic [1:0]                   axil_bresp,
  output logic                         axil_bvalid,
  input  wire                          axil_bready,
  input  wire  [7:0]                   axil_araddr,
  input  wire                          axil_arvalid,
  output logic                         axil_arready,
  output logic [31:0]                  axil_rdata,
  output logic [1:0]                   axil_rresp,
  output logic                         axil_rvalid,
  input  wire                          axil_rready,

  // fp16 KV write stream in (tuser: 0=key, 1=value — apex_top rt_kv_user)
  input  wire  [15:0]                  s_axis_kv_tdata,
  input  wire                          s_axis_kv_tvalid,
  output logic                         s_axis_kv_tready,
  input  wire                          s_axis_kv_tlast,
  input  wire                          s_axis_kv_tuser,

  // fp32 decompressed read bus out
  output logic [31:0]                  m_axis_kv_tdata,
  output logic                         m_axis_kv_tvalid,
  input  wire                          m_axis_kv_tready,
  output logic                         m_axis_kv_tlast,

  // D-008 partial-group flush (CSR FLUSH -> engine[tier_sel])
  input  wire                          flush_req,

  // observability
  output logic                         irq,          // OR of the three engines
  output logic                         evict_needed, // engine[tier_sel]
  output logic [$clog2(KVQ_DEPTH)-1:0] evict_addr,
  output logic                         m_pending,    // any m_axis beat pending
  // D-027 (S12): e2's live-mask validity — the tile INFO_TIER CQ-4+ term.
  // Not tier_sel-muxed: CQ-4+ availability is a property of engine 2 alone.
  output logic                         mask_valid_cq4p
);

  // per-engine wire bundles (index = kvq_tier_e code)
  logic        e_awready [3];
  logic        e_wready  [3];
  logic [1:0]  e_bresp   [3];
  logic        e_bvalid  [3];
  logic        e_arready [3];
  logic [31:0] e_rdata   [3];
  logic [1:0]  e_rresp   [3];
  logic        e_rvalid  [3];
  logic        e_s_tready[3];
  logic [31:0] e_m_tdata [3];
  logic        e_m_tvalid[3];
  logic        e_m_tlast [3];
  logic        e_irq     [3];
  logic        e_evict   [3];
  logic [$clog2(KVQ_DEPTH)-1:0] e_evict_addr [3];
  logic        e_mask_valid [3];

  // input gating per engine (valid/ready/flush fan-out)
  logic        sel      [3];
  logic        e_awvalid[3];
  logic        e_wvalid [3];
  logic        e_bready [3];
  logic        e_arvalid[3];
  logic        e_rready [3];
  logic        e_s_tvalid[3];
  logic        e_m_tready[3];
  logic        e_flush  [3];

  // defensive index: kvq_tier_e has no code 3 (the CSR clamps 2'b11 to CQ4
  // and TIP never emits it) — map it to CQ4 anyway so the mux can never
  // index out of range. Fan-out and mux are SEPARATE fine-grained assigns
  // (a single always_comb would look like a false combinational cycle
  // through the engines' combinational AXI-ready terms).
  logic [1:0] idx;
  assign idx = (2'(tier_sel) == 2'd3) ? 2'(KVQ_CQ4) : 2'(tier_sel);

  generate
    for (genvar gs = 0; gs < 3; gs++) begin : g_sel
      assign sel[gs]        = (idx == 2'(gs));
      assign e_awvalid[gs]  = axil_awvalid && sel[gs];
      assign e_wvalid[gs]   = axil_wvalid && sel[gs];
      assign e_bready[gs]   = axil_bready && sel[gs];
      assign e_arvalid[gs]  = axil_arvalid && sel[gs];
      assign e_rready[gs]   = axil_rready && sel[gs];
      assign e_s_tvalid[gs] = s_axis_kv_tvalid && sel[gs];
      assign e_m_tready[gs] = m_axis_kv_tready && sel[gs];
      assign e_flush[gs]    = flush_req && sel[gs];
    end
  endgenerate

  // output muxes (tier_sel is quasi-static; engines register everything)
  assign axil_awready     = e_awready [idx];
  assign axil_wready      = e_wready  [idx];
  assign axil_bresp       = e_bresp   [idx];
  assign axil_bvalid      = e_bvalid  [idx];
  assign axil_arready     = e_arready [idx];
  assign axil_rdata       = e_rdata   [idx];
  assign axil_rresp       = e_rresp   [idx];
  assign axil_rvalid      = e_rvalid  [idx];
  assign s_axis_kv_tready = e_s_tready[idx];
  assign m_axis_kv_tdata  = e_m_tdata [idx];
  assign m_axis_kv_tvalid = e_m_tvalid[idx];
  assign m_axis_kv_tlast  = e_m_tlast [idx];
  assign evict_needed     = e_evict   [idx];
  assign evict_addr       = e_evict_addr[idx];

  assign irq       = e_irq[0] | e_irq[1] | e_irq[2];
  assign m_pending = e_m_tvalid[0] | e_m_tvalid[1] | e_m_tvalid[2];
  assign mask_valid_cq4p = e_mask_valid[2];
  // e0/e1 export mask_valid too (constant 1 at their OUTLIER_K=0) — consumed
  // here so every engine port stays driven/consumed (full -Wall, no waivers)
  wire _unused_mask_ok = &{1'b0, e_mask_valid[0], e_mask_valid[1]};

  generate
    for (genvar gi = 0; gi < 3; gi++) begin : g_eng
      kvq_engine #(
        .VECTOR_DIM (CFG_D),
        .TIER       (gi),
        .KEY_GROUP  (KVQ_G),
        .OUTLIER_K  (gi == 2 ? OUTLIER_K : 32'd0),
        .SCALE_SETS (KVQ_SETS),
        .SCALE_WIDTH(16),
        .SRAM_DEPTH (KVQ_DEPTH),
        .COORD_WIDTH(16),
        .OUT_WIDTH  (32),
        .MASK_FILE  (gi == 2 ? MASK_FILE : "")
      ) u_eng (
        .clk              (clk),
        .rst_n            (rst_n),
        .axil_awaddr      (axil_awaddr),
        .axil_awvalid     (e_awvalid[gi]),
        .axil_awready     (e_awready[gi]),
        .axil_wdata       (axil_wdata),
        .axil_wvalid      (e_wvalid[gi]),
        .axil_wready      (e_wready[gi]),
        .axil_bresp       (e_bresp[gi]),
        .axil_bvalid      (e_bvalid[gi]),
        .axil_bready      (e_bready[gi]),
        .axil_araddr      (axil_araddr),
        .axil_arvalid     (e_arvalid[gi]),
        .axil_arready     (e_arready[gi]),
        .axil_rdata       (e_rdata[gi]),
        .axil_rresp       (e_rresp[gi]),
        .axil_rvalid      (e_rvalid[gi]),
        .axil_rready      (e_rready[gi]),
        .s_axis_kv_tdata  (s_axis_kv_tdata),
        .s_axis_kv_tvalid (e_s_tvalid[gi]),
        .s_axis_kv_tready (e_s_tready[gi]),
        .s_axis_kv_tlast  (s_axis_kv_tlast),
        .s_axis_kv_tuser  (s_axis_kv_tuser),
        .m_axis_kv_tdata  (e_m_tdata[gi]),
        .m_axis_kv_tvalid (e_m_tvalid[gi]),
        .m_axis_kv_tready (e_m_tready[gi]),
        .m_axis_kv_tlast  (e_m_tlast[gi]),
        .flush_req        (e_flush[gi]),
        .irq              (e_irq[gi]),
        .evict_needed     (e_evict[gi]),
        .evict_addr       (e_evict_addr[gi]),
        .mask_valid       (e_mask_valid[gi])
      );
    end
  endgenerate

endmodule

`default_nettype wire
