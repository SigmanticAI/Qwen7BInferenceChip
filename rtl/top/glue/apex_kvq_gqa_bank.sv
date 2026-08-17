// apex_kvq_gqa_bank.sv — in-tile per-KV-HEAD CQ-8 BANK: N_ENG verified,
// UNMODIFIED kvq_engine instances (one per KV head) behind a live engine-
// select mux. This is the LEVEL_C_INTEGRATION.md §9.1 R3-AMENDED closure
// mechanism (IB-LAYER S4b, approved plan IB_LAYER.md §0.1): GQA at
// H_kv=4/T=128 cannot fit one engine's flat record space (4·2T = 1024 >
// 256), so — exactly like the D-024 tier bank (apex_kvq_bank, which stays
// untouched/verified as the default build's subsystem) — capacity is
// realized by BANKING verified engines, never by editing one.
//
//   e[g], g in 0..N_ENG-1 : TIER=0 (CQ-8) — per-token INT8 K+V, value path
//                           only; one engine per KV head, each with its own
//                           SRAM/address space (2T <= KVQ_DEPTH records:
//                           K at [0,T), V at [T,2T) — the L3 store map).
//
// MAPPING CONTRACT (R3 as amended — golden GQA slicing, transformer.py):
//   engine index = KV-head index. The query-head -> KV-head mapping
//   h // (H/H_kv) — H/H_kv = 7 for Qwen2.5-7B, NOT a power of two, no
//   shift tricks — lives in the SEQUENCER (seq_layer_walker2 g_idx/sk_g in
//   walk mode; host software via the LAYER_CTRL[17:15] l_kv_map carve in
//   host mode). This bank NEVER computes the mapping: eng_sel is consumed
//   as a ready engine index.
//
// ROUTING CONTRACT (D-024 granularity, documented honestly):
//   eng_sel is a quasi-static LEVEL. Every interface of the bank — the
//   AXI-Lite window, the s_axis write stream (+tuser key/value select), the
//   m_axis read bus, flush_req — routes to engine[eng_sel] COMBINATIONALLY.
//   The select may change only while the bank is quiescent (selected engine
//   STATUS.idle polled, no m_axis beat pending): per STORE RUN and per READ
//   — the walker's poll-idle-before-WRITE_ADDR / poll-before-READ_ADDR
//   invariants satisfy this by construction. Mid-stream select switches are
//   host contract violations (same class as rt_* route levels / tier_sel).
//   Each engine keeps its own CTRL.enable and D-020 soft reset: the host/
//   walker enables/resets each engine it uses (N_ENG CTRL writes).
//
// Non-selected engines see valid=0 on every input and ready=0 on m_axis;
// their registered outputs are ignored by the muxes. irq is the OR of the
// N_ENG engines (each maskable via its own IRQ_MASK); evict_needed /
// evict_addr follow the selected engine. mask_valid (constant 1 at
// OUTLIER_K=0) is consumed internally — this bank has no CQ-4+ lane, so
// the tile's INFO_TIER CQ-4+ term is 0 by construction in a GQA build.
//
// Every port of every engine is driven/consumed — full -Wall, no waivers on
// this file (vendored-file waivers stay scoped to the engines' internals).

`default_nettype none

module apex_kvq_gqa_bank #(
  parameter int unsigned N_ENG     = 4,    // per-KV-head CQ-8 engines (H_kv)
  parameter int unsigned CFG_D     = 64,   // VECTOR_DIM
  parameter int unsigned KVQ_G     = 128,  // KEY_GROUP (engine parameter
                                           // parity with the verified e0
                                           // shape; CQ-8 keys are per-token)
  parameter int unsigned KVQ_DEPTH = 256,  // SRAM records per engine; the
                                           // R3 sizing rule is 2T <= DEPTH
                                           // (exactly full at T=128/256)
  parameter int unsigned KVQ_SETS  = 4,    // D-026 scale-bank sets (engine
                                           // parameter parity; CQ-8 value
                                           // scales ride the records)
  localparam int unsigned ENG_W    = (N_ENG > 1) ? $clog2(N_ENG) : 1
)(
  input  wire                          clk,
  input  wire                          rst_n,

  // live engine select (quasi-static; see routing contract above)
  input  wire  [ENG_W-1:0]             eng_sel,

  // AXI-Lite window (routed to engine[eng_sel])
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

  // fp16 KV write stream in (tuser: 0=key, 1=value — apex_top rt_kv_user;
  // CQ-8 stores K and V through the value path, tuser=1 — the L3 flow)
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

  // D-008 partial-group flush (CSR FLUSH -> engine[eng_sel]; a CQ-8 engine
  // with no open group treats it as the documented no-op)
  input  wire                          flush_req,

  // observability
  output logic                         irq,          // OR of the N_ENG engines
  output logic                         evict_needed, // engine[eng_sel]
  output logic [$clog2(KVQ_DEPTH)-1:0] evict_addr,
  output logic                         m_pending     // any m_axis beat pending
);

  // per-engine wire bundles (index = KV-head engine index)
  logic        e_awready [N_ENG];
  logic        e_wready  [N_ENG];
  logic [1:0]  e_bresp   [N_ENG];
  logic        e_bvalid  [N_ENG];
  logic        e_arready [N_ENG];
  logic [31:0] e_rdata   [N_ENG];
  logic [1:0]  e_rresp   [N_ENG];
  logic        e_rvalid  [N_ENG];
  logic        e_s_tready[N_ENG];
  logic [31:0] e_m_tdata [N_ENG];
  logic        e_m_tvalid[N_ENG];
  logic        e_m_tlast [N_ENG];
  logic        e_irq     [N_ENG];
  logic        e_evict   [N_ENG];
  logic [$clog2(KVQ_DEPTH)-1:0] e_evict_addr [N_ENG];
  logic        e_mask_valid [N_ENG];

  // input gating per engine (valid/ready/flush fan-out)
  logic        sel      [N_ENG];
  logic        e_awvalid[N_ENG];
  logic        e_wvalid [N_ENG];
  logic        e_bready [N_ENG];
  logic        e_arvalid[N_ENG];
  logic        e_rready [N_ENG];
  logic        e_s_tvalid[N_ENG];
  logic        e_m_tready[N_ENG];
  logic        e_flush  [N_ENG];

  // defensive index: at non-power-of-two N_ENG the select space has codes
  // >= N_ENG (the walker's build-envelope fence refuses such geometries and
  // the l_kv_map host contract is 0..N_ENG-1) — clamp them to engine 0
  // anyway so the mux can never index out of range; at power-of-two N_ENG
  // (the H_kv=4 build) this constant-folds away. Fan-out and mux are
  // SEPARATE fine-grained assigns (a single always_comb would look like a
  // false combinational cycle through the engines' combinational AXI-ready
  // terms — the D-024 tier-bank finding).
  logic [ENG_W-1:0] idx;
  assign idx = (32'(eng_sel) >= N_ENG) ? '0 : eng_sel;

  generate
    for (genvar gs = 0; gs < int'(N_ENG); gs++) begin : g_sel
      assign sel[gs]        = (idx == ENG_W'(gs));
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

  // output muxes (eng_sel is quasi-static; engines register everything)
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

  // OR-reductions over the engine array (loop form: N_ENG is a parameter)
  always_comb begin
    irq       = 1'b0;
    m_pending = 1'b0;
    for (int unsigned i = 0; i < N_ENG; i++) begin
      irq       |= e_irq[i];
      m_pending |= e_m_tvalid[i];
    end
  end

  // every engine exports mask_valid (constant 1 at OUTLIER_K=0) — consumed
  // here so every engine port stays driven/consumed (full -Wall, no waivers)
  logic unused_mask_ok;
  always_comb begin
    unused_mask_ok = 1'b0;
    for (int unsigned i = 0; i < N_ENG; i++)
      unused_mask_ok &= e_mask_valid[i];
  end
  wire _unused_ok = &{1'b0, unused_mask_ok};

  generate
    for (genvar gi = 0; gi < int'(N_ENG); gi++) begin : g_eng
      kvq_engine #(
        .VECTOR_DIM (CFG_D),
        .TIER       (0),                 // CQ-8 per-KV-head store (R3/L4 v1)
        .KEY_GROUP  (KVQ_G),
        .OUTLIER_K  (32'd0),
        .SCALE_SETS (KVQ_SETS),
        .SCALE_WIDTH(16),
        .SRAM_DEPTH (KVQ_DEPTH),
        .COORD_WIDTH(16),
        .OUT_WIDTH  (32),
        .MASK_FILE  ("")
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
