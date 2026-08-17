// =============================================================================
// File        : gemm_systolic_accel.sv
// Module      : gemm_systolic_accel
// -----------------------------------------------------------------------------
// PURPOSE
//   Top-level structural integration of the INT8 GEMM systolic accelerator.
//   This file performs PURE WIRING: it instantiates the pre-existing
//   submodules (stream_skid x3, gemm_ctrl, gemm_wbuf, gemm_abuf,
//   gemm_pe_array, gemm_requant) and connects them. No new datapath or FSM
//   computation is introduced here -- only glue logic and bus packing /
//   unpacking of the {last,data} skid payloads.
//
// ASM-006 (result packing convention, inherited/preserved, NOT reinterpreted):
//   The result stream (y_valid/y_ready/y_data/y_last) is produced row-major,
//   m-outer / n-inner across output tiles, and within each output tile one
//   packed N-wide beat is emitted per accumulator row i = 0..N-1 (N beats per
//   tile). Within each beat, y_data packs N signed INT8 result lanes with
//   lane/column 0 in the least-significant byte (y_data[DATA_W-1:0]) and
//   lane N-1 in the most-significant byte -- this matches the a_vec/w_vec/
//   acc_row lane-packing convention used throughout gemm_pe_array and
//   gemm_requant (element i at bits [i*DATA_W +: DATA_W]). y_last is
//   asserted on the final beat (row i=N-1) of the final output tile
//   (m_idx==tiles_m-1 && n_idx==tiles_n-1) of the job, exactly as computed by
//   gemm_ctrl (is_last_tile && row_i==N-1); this top level does not alter
//   that computation, it only relays it through the result skid.
//
// SKID PLACEMENT (REQ-PROTO-002/003, REQ-GEMM-009, REQ-TIME-001)
//   A stream_skid instance is placed on the weight_in, activation_in, and
//   result_out external streaming channels so that (a) those external-facing
//   ready/valid outputs are pure registered outputs (never combinationally
//   dependent on the paired valid, satisfying REQ-PROTO-003 at the boundary)
//   and (b) the internal core (gemm_ctrl) always sees/drives an
//   already-registered, de-skidded handshake on those channels, per the
//   ASM-006 interface assumption documented in gemm_ctrl.sv. The descriptor
//   channel is the one exception: it is wired DIRECTLY to gemm_ctrl with no
//   skid, per REQ-GEMM-001/REQ-PROTO-004 (see dedicated note below) --
//   gemm_ctrl's desc_ready is itself a registered function of FSM state, so
//   REQ-PROTO-003 is still satisfied without an extra skid stage. Specifically:
//     - weight_in     : skid WIDTH=PW+1, payload packed as {w_last, w_data}
//                        (w_last in bit PW, w_data in bits [PW-1:0]).
//     - activation_in : skid WIDTH=PW+1, payload packed as {a_last, a_data}
//                        (a_last in bit PW, a_data in bits [PW-1:0]).
//     - result_out    : skid WIDTH=PW+1, payload packed as {y_last, y_data}
//                        on the s_* (ctrl-facing) side, unpacked back into
//                        discrete y_last/y_data on the m_* (external) side.
//   Packing convention used consistently on all three packed channels:
//   bit [PW] = *_last flag, bits [PW-1:0] = *_data payload. This convention
//   is purely a top-level wiring artifact of feeding one WIDTH-parameterized
//   generic skid per channel; it is not visible to, and does not alter, any
//   FSM/datapath semantics inside gemm_ctrl or the compute submodules.
//
//   result_out "done" re-timing (REQ-GEMM-012 / REQ-TIME-003):
//   gemm_ctrl's internal "done" output pulses when the final result beat is
//   accepted INTO the result skid (pre-skid, i.e. on ctrl_y_valid &&
//   ctrl_y_ready && ctrl_y_last), not when the host actually accepts that
//   beat at the external y_* port. Under host backpressure the result skid
//   can hold a beat, so the pre-skid pulse would precede true host
//   acceptance of the final beat. To guarantee "done" reflects true host
//   acceptance, the top-level "done" output is intentionally NOT connected
//   to gemm_ctrl's done pin (left unconnected on u_ctrl); instead it is
//   re-timed to the EXTERNAL (post-result-skid) handshake:
//     assign done = y_valid && y_ready && y_last;
//   This is a one-cycle pulse (y_last is asserted only on the final beat of
//   a job), firing exactly once per job, exactly when the host accepts the
//   final beat.
//
// DESCRIPTOR CHANNEL WIRING (REQ-GEMM-001 / REQ-PROTO-004)
//   Unlike the other three external channels, the descriptor channel is
//   wired DIRECTLY (no stream_skid) to gemm_ctrl: desc_valid/desc_data pass
//   straight through to gemm_ctrl, and the external desc_ready output is
//   gemm_ctrl's desc_ready directly. gemm_ctrl registers desc_ready as a
//   function of its own FSM state (asserted only while IDLE), so this
//   remains a purely registered ready with no combinational ready<-valid
//   loop, while correctly guaranteeing desc_ready is low whenever the core
//   is busy (a stream_skid here would incorrectly let desc_ready go high
//   based on skid occupancy alone, independent of core busy/idle state).
//
// RESET / CLOCKING (REQ-GEMM-010)
//   Single clk, single synchronous active-low rst_n, fanned out unchanged to
//   every child that has an rst_n port (all three stream_skid instances,
//   gemm_ctrl, gemm_pe_array, gemm_requant). gemm_wbuf/gemm_abuf have no
//   rst_n port (behavioral SRAM model, contents undefined until written per
//   their own headers) and are therefore NOT connected to rst_n, as required.
//
// PARAMETERIZATION (REQ-GEMM-014)
//   All widths are derived from the parameters below via localparams
//   (PW, WAW, AAW, ROWSEL_W); no lane count, address width, or packed width
//   is hardcoded, so N may be changed (e.g. to 16) without editing this
//   source.
//
// REQUIREMENTS OWNED BY THIS FILE
//   REQ-GEMM-008 (writeback path wiring), REQ-GEMM-009 (handshake at all
//   external boundaries via skids), REQ-GEMM-010 (reset fanout), REQ-GEMM-014
//   (parameterization), REQ-PROTO-001/002/003/004, ASM-006 (documented above,
//   computation preserved from gemm_ctrl unaltered).
// =============================================================================

module gemm_systolic_accel #(
    parameter int N            = 8,
    parameter int DATA_W       = 8,
    parameter int ACC_W        = 32,
    parameter int SCALE_W      = 32,
    parameter int SHIFT_W      = 6,
    parameter int WBUF_DEPTH   = 64,
    parameter int ABUF_DEPTH   = 64,
    parameter int MAX_TILES_M  = 8,
    parameter int MAX_TILES_K  = 8,
    parameter int MAX_TILES_N  = 8,
    parameter int DESC_W       = 128,
    localparam int PW          = N*DATA_W,
    localparam int WAW         = $clog2(WBUF_DEPTH),
    localparam int AAW         = $clog2(ABUF_DEPTH),
    localparam int ROWSEL_W    = $clog2(N)
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // descriptor channel (external)
    input  logic                  desc_valid,
    output logic                  desc_ready,
    input  logic [DESC_W-1:0]     desc_data,

    // weight streaming channel (external)
    input  logic                  w_valid,
    output logic                  w_ready,
    input  logic [PW-1:0]         w_data,
    input  logic                  w_last,

    // activation streaming channel (external)
    input  logic                  a_valid,
    output logic                  a_ready,
    input  logic [PW-1:0]         a_data,
    input  logic                  a_last,

    // result streaming channel (external)
    output logic                  y_valid,
    input  logic                  y_ready,
    output logic [PW-1:0]         y_data,
    output logic                  y_last,

    // status
    output logic                  busy,
    output logic                  done,
    output logic                  err_overflow
);

    // =========================================================================
    // Internal nets : descriptor channel
    //   REQ-GEMM-001/REQ-PROTO-004: wired directly to gemm_ctrl, no skid
    //   (see header note above). No separate internal nets needed; external
    //   desc_valid/desc_data/desc_ready connect straight to u_ctrl below.
    // =========================================================================

    // =========================================================================
    // Internal nets : weight channel
    //   s_* side packs external {w_last, w_data} (WIDTH = PW+1).
    //   m_* side (de-skidded) is unpacked into ds_w_data / ds_w_last for ctrl
    //   and gemm_wbuf.
    // =========================================================================
    logic [PW:0]           w_skid_s_data;      // {w_last, w_data}
    logic                  ds_w_valid;
    logic                  ds_w_ready;
    logic [PW:0]           ds_w_packed;        // {ds_w_last, ds_w_data}
    logic [PW-1:0]         ds_w_data;
    logic                  ds_w_last;

    // =========================================================================
    // Internal nets : activation channel (same pattern as weight channel)
    // =========================================================================
    logic [PW:0]           a_skid_s_data;      // {a_last, a_data}
    logic                  ds_a_valid;
    logic                  ds_a_ready;
    logic [PW:0]           ds_a_packed;        // {ds_a_last, ds_a_data}
    logic [PW-1:0]         ds_a_data;
    logic                  ds_a_last;

    // =========================================================================
    // Internal nets : result channel
    //   ctrl drives its own y_valid/y_data/y_last into the skid's s_* side;
    //   ctrl's y_ready input is driven by the skid's s_ready. External
    //   y_valid/y_data/y_last/y_ready are the skid's m_* side.
    // =========================================================================
    logic                  ctrl_y_valid;
    logic                  ctrl_y_ready;
    logic [PW-1:0]         ctrl_y_data;
    logic                  ctrl_y_last;
    logic [PW:0]           y_skid_s_data;      // {ctrl_y_last, ctrl_y_data}
    logic [PW:0]           y_skid_m_data;      // {y_last, y_data} (external side)

    // =========================================================================
    // Internal nets : weight buffer control/data
    // =========================================================================
    logic                  wbuf_we;
    logic [WAW-1:0]        wbuf_waddr;
    logic                  wbuf_re;
    logic [WAW-1:0]        wbuf_raddr;
    logic [PW-1:0]         w_rdata;            // wbuf.rdata -> pe_array.w_vec

    // =========================================================================
    // Internal nets : activation buffer control/data
    // =========================================================================
    logic                  abuf_we;
    logic [AAW-1:0]        abuf_waddr;
    logic                  abuf_re;
    logic [AAW-1:0]        abuf_raddr;
    logic [PW-1:0]         a_rdata;            // abuf.rdata -> pe_array.a_vec

    // =========================================================================
    // Internal nets : PE array control/data
    // =========================================================================
    logic                  pe_clear_acc;
    logic                  pe_mac_en;
    logic [ROWSEL_W-1:0]   pe_row_sel;
    logic [N*ACC_W-1:0]    pe_acc_row;         // pe_array.acc_row -> requant.acc_in

    // =========================================================================
    // Internal nets : requant control/data
    // =========================================================================
    logic                  rq_in_valid;
    logic [SCALE_W-1:0]    rq_req_scale;
    logic [SHIFT_W-1:0]    rq_req_shift;
    logic                  rq_out_valid;
    logic [PW-1:0]         rq_y_vec;
    logic                  rq_ovf;

    // -------------------------------------------------------------------------
    // Packing / unpacking assigns (glue only, no computation).
    // -------------------------------------------------------------------------
    assign w_skid_s_data = {w_last, w_data};
    assign ds_w_last      = ds_w_packed[PW];
    assign ds_w_data      = ds_w_packed[PW-1:0];

    assign a_skid_s_data = {a_last, a_data};
    assign ds_a_last      = ds_a_packed[PW];
    assign ds_a_data      = ds_a_packed[PW-1:0];

    assign y_skid_s_data = {ctrl_y_last, ctrl_y_data};
    assign y_last        = y_skid_m_data[PW];
    assign y_data         = y_skid_m_data[PW-1:0];

    // REQ-GEMM-012 / REQ-TIME-003: "done" is intentionally re-timed to the
    // EXTERNAL (post-result-skid) handshake so it reflects true host
    // acceptance of the final result beat, not the pre-skid internal accept
    // event inside gemm_ctrl (see header note above). One-cycle pulse:
    // y_last is high only on the final beat of a job.
    assign done = y_valid && y_ready && y_last;

    // =========================================================================
    // Skid buffer : weight_in  (payload packed {w_last, w_data})
    // =========================================================================
    stream_skid #(
        .WIDTH (PW+1)
    ) u_skid_weight (
        .clk     (clk),
        .rst_n   (rst_n),
        .s_valid (w_valid),
        .s_ready (w_ready),
        .s_data  (w_skid_s_data),
        .m_valid (ds_w_valid),
        .m_ready (ds_w_ready),
        .m_data  (ds_w_packed)
    );

    // =========================================================================
    // Skid buffer : activation_in  (payload packed {a_last, a_data})
    // =========================================================================
    stream_skid #(
        .WIDTH (PW+1)
    ) u_skid_act (
        .clk     (clk),
        .rst_n   (rst_n),
        .s_valid (a_valid),
        .s_ready (a_ready),
        .s_data  (a_skid_s_data),
        .m_valid (ds_a_valid),
        .m_ready (ds_a_ready),
        .m_data  (ds_a_packed)
    );

    // =========================================================================
    // Skid buffer : result_out  (payload packed {y_last, y_data})
    //   s_* side = ctrl-facing; m_* side = external top-level result port.
    // =========================================================================
    stream_skid #(
        .WIDTH (PW+1)
    ) u_skid_result (
        .clk     (clk),
        .rst_n   (rst_n),
        .s_valid (ctrl_y_valid),
        .s_ready (ctrl_y_ready),
        .s_data  (y_skid_s_data),
        .m_valid (y_valid),
        .m_ready (y_ready),
        .m_data  (y_skid_m_data)
    );

    // =========================================================================
    // Control FSM
    // =========================================================================
    gemm_ctrl #(
        .N           (N),
        .DATA_W      (DATA_W),
        .ACC_W       (ACC_W),
        .SCALE_W     (SCALE_W),
        .SHIFT_W     (SHIFT_W),
        .WBUF_DEPTH  (WBUF_DEPTH),
        .ABUF_DEPTH  (ABUF_DEPTH),
        .MAX_TILES_M (MAX_TILES_M),
        .MAX_TILES_K (MAX_TILES_K),
        .MAX_TILES_N (MAX_TILES_N),
        .DESC_W      (DESC_W)
    ) u_ctrl (
        .clk           (clk),
        .rst_n         (rst_n),

        .desc_valid    (desc_valid),
        .desc_ready    (desc_ready),
        .desc_data     (desc_data),

        .w_valid       (ds_w_valid),
        .w_ready       (ds_w_ready),
        .w_last        (ds_w_last),

        .a_valid       (ds_a_valid),
        .a_ready       (ds_a_ready),
        .a_last        (ds_a_last),

        .y_valid       (ctrl_y_valid),
        .y_ready       (ctrl_y_ready),
        .y_data        (ctrl_y_data),
        .y_last        (ctrl_y_last),

        .busy          (busy),
        // intentionally unconnected: see REQ-GEMM-012/REQ-TIME-003 note above --
        // top-level "done" is re-timed to the external result handshake, not
        // ctrl's pre-skid pulse. An unconnected output pin is legal Verilog.
        /* verilator lint_off PINCONNECTEMPTY */
        .done          (),
        /* verilator lint_on PINCONNECTEMPTY */
        .err_overflow  (err_overflow),

        .wbuf_we       (wbuf_we),
        .wbuf_waddr    (wbuf_waddr),
        .wbuf_re       (wbuf_re),
        .wbuf_raddr    (wbuf_raddr),

        .abuf_we       (abuf_we),
        .abuf_waddr    (abuf_waddr),
        .abuf_re       (abuf_re),
        .abuf_raddr    (abuf_raddr),

        .pe_clear_acc  (pe_clear_acc),
        .pe_mac_en     (pe_mac_en),
        .pe_row_sel    (pe_row_sel),

        .rq_in_valid   (rq_in_valid),
        .rq_req_scale  (rq_req_scale),
        .rq_req_shift  (rq_req_shift),
        .rq_out_valid  (rq_out_valid),
        .rq_y_vec      (rq_y_vec),
        .rq_ovf        (rq_ovf)
    );

    // =========================================================================
    // Weight buffer storage
    // =========================================================================
    gemm_wbuf #(
        .N      (N),
        .DATA_W (DATA_W),
        .DEPTH  (WBUF_DEPTH)
    ) u_wbuf (
        .clk    (clk),
        .we     (wbuf_we),
        .waddr  (wbuf_waddr),
        .wdata  (ds_w_data),
        .re     (wbuf_re),
        .raddr  (wbuf_raddr),
        .rdata  (w_rdata)
    );

    // =========================================================================
    // Activation buffer storage
    // =========================================================================
    gemm_abuf #(
        .N      (N),
        .DATA_W (DATA_W),
        .DEPTH  (ABUF_DEPTH)
    ) u_abuf (
        .clk    (clk),
        .we     (abuf_we),
        .waddr  (abuf_waddr),
        .wdata  (ds_a_data),
        .re     (abuf_re),
        .raddr  (abuf_raddr),
        .rdata  (a_rdata)
    );

    // =========================================================================
    // PE systolic accumulator array
    // =========================================================================
    gemm_pe_array #(
        .N      (N),
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_pe_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear_acc  (pe_clear_acc),
        .mac_en     (pe_mac_en),
        .a_vec      (a_rdata),
        .w_vec      (w_rdata),
        .row_sel    (pe_row_sel),
        .acc_row    (pe_acc_row)
    );

    // =========================================================================
    // Requantization pipeline
    // =========================================================================
    gemm_requant #(
        .N       (N),
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .SCALE_W (SCALE_W),
        .SHIFT_W (SHIFT_W)
    ) u_requant (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (rq_in_valid),
        .acc_in     (pe_acc_row),
        .req_scale  (rq_req_scale),
        .req_shift  (rq_req_shift),
        .out_valid  (rq_out_valid),
        .y_vec      (rq_y_vec),
        .ovf        (rq_ovf)
    );

endmodule
