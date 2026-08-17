// kvq_engine.sv — APEX KVQ streaming subsystem top (clean-room).
//
// This is APEX's own streaming controller for the per-channel INT4 KV-cache
// codec. It is written from the numerics/stream contract, not ported from any
// third party: the datapath math lives entirely in the clean-room cores under
// rtl/kvq/cores/ (cq_value_path / cq_key_path / sram_controller / …), which are
// verified bit-exact against golden/apex_golden/cq_codec.py. This module is
// only the *sequencer*: it assembles fp16 tokens off AXI-Stream, drives the
// cores, lays down §4 byte-aligned records into the record SRAM, and replays
// them back on the fp32 read stream — all under the AXI-Lite CSR contract.
//
// WHAT THE ENGINE MUST DO (re-derived from ARCHITECTURE.md §2/§4/§5/§7 and the
// APEX decision register D-006/D-008/D-009/D-016/D-020/D-026/D-027; where prose
// and golden vectors disagree, the vectors win):
//
//  * Values / CQ-8 keys (per-token): collect D fp16 beats, take one per-token
//    scale over all D dims, quant to INT4/INT8, pack §5, store a value record.
//  * CQ-4 / CQ-4+ keys (per-channel, grouped): buffer up to KEY_GROUP tokens,
//    take a per-CHANNEL amax across the g buffered tokens (a partial final
//    group scales over g, not G), turn each into an fp16 group scale, quantize
//    every keep channel to INT4, and emit one unified per-channel key record
//    per token at grp_base+idx. Outlier channels (CQ-4+) bypass quant: their
//    raw fp16 is stored with code +1 for identity replay.
//  * Read/decompress: writing READ_ADDR launches a one-record read; the record
//    tag byte selects the value vs key dequant lane; D fp32 channels stream out
//    one per beat, tlast on the last.
//
// APEX CONTRACT BEHAVIORS (each re-derived here, none inherited):
//
//  D-009/§4  RECORD LAYOUT — a byte stream, LSB-first, padded to a 64-bit
//            boundary. Same image the record-SRAM stores one-per-row:
//              key   : [tag=8'h01][ fp16 field : D×16 ][ int4 codes : D×4 ][pad]
//              value : [tag=8'h00][ fp16 scale : 16   ][ payload   : D×BPV][pad]
//            The tag's bit 0 (1=key) drives the read-side dequant select.
//
//  D-016(a)  BOUNDED READ WAIT + ERROR — the record SRAM only acks a read of a
//            written address, so an unwritten-address read would wait forever.
//            ST_RWAIT gives it RD_WAIT_MAX cycles (it answers in 1); on timeout
//            the read is DROPPED with zero output beats and RD_ERR is latched
//            sticky in IRQ_STATUS[0] (W1C), mirrored in STATUS[1], and raised on
//            `irq` when IRQ_MASK[0] is set.
//
//  D-016(b)  NO ACCEPTED-BEAT LOSS — if a decompress request and an accepted
//            s_axis beat land together, the beat wins and the read is deferred
//            (read_pending) to the next idle; tready drops for the whole read
//            walk so nothing can be accepted while unserviceable.
//
//  D-008     PARTIAL-GROUP FLUSH — flush_req (1-cycle; wired to CSR FLUSH) freezes
//            an open partial key group at its current fill and runs it through the
//            same scale/quant/emit datapath, unwedging the engine. A flush latched
//            mid-token (flush_pending) fires once that token completes; a flush
//            with no open group is a no-op. Flushing quantizes whole tokens only.
//
//  D-020     SOFT-RESET = ABORT-AND-DISCARD — CTRL.soft_reset takes priority over
//            the state case (so a reset landing in a single-cycle phase is never
//            lost); dp_clear rides it into the datapath cores the same cycle so an
//            aborted token/group can never finish and store later. Input-side
//            state is dropped immediately; an in-flight read burst (ST_OUTPUT/
//            ST_OFLUSH) instead ALWAYS completes (§5 forbids retracting a valid
//            beat), and STATUS.idle stays 0 until its final beat is accepted.
//            SRAM contents/occupancy are preserved. The CQ-4+ outlier lane forces
//            the fp32 sign bit from the stored raw fp16 so an exact -0.0 replays
//            as -0.0 (identity contract, D-010).
//
//  D-027     LOADABLE OUTLIER MASK (S12) — the CQ-4+ outlier mask is a CSR
//            runtime input (executable contract: golden test_mask_semantics.py).
//            MASK0-3 (0x50-0x5C) stage 32 mask bits/word (staged readback);
//            MASK_CTRL (0x60) bit0 commits: effective iff popcount(staged) ==
//            OUTLIER_K, else sticky MASK_ERR (IRQ_STATUS[2]) and the live mask
//            is unchanged. An EFFECTIVE commit while the store is logically
//            occupied (occupancy > 0, or an open key group in any ST_K* phase)
//            raises sticky MASK_SWAP (IRQ_STATUS[3]): D-026 records do not
//            self-describe their mask, so decoding pre-commit records post-
//            commit is a host-contract violation the flag audits — the
//            hardware does not police beyond it (SB_OVWR philosophy). The
//            live mask + ownership persist through D-020 soft reset exactly
//            like scale_bank_store (records persist, so their decode key
//            must); hard reset restores the build default (MASK_FILE if
//            given, else zeros/invalid). mask_valid is COMPUTED as
//            popcount(live) == OUTLIER_K, never stored. OUTLIER_K=0 builds
//            keep 0x50-0x60 reserved (DEADBEEF / write-ignored).
//
// The FSM phase encoding (state[3:0], ST_IDLE..ST_OFLUSH below) and the record
// image are an APEX verification interface: verif/kvq/sb whiteboxes `dut.state`
// to reset the engine in every reachable phase, and the smoke/flush/collide TBs
// peek the record SRAM (`u_sram.mem`). Tier codes match apex_pkg::kvq_tier_e
// (0=CQ-8, 1=CQ-4, 2=CQ-4+).

`default_nettype none

module kvq_engine #(
    parameter integer VECTOR_DIM    = 16,   // D: real configs 64/128; small proxy default
    parameter integer TIER          = 1,    // 0 = CQ-8, 1 = CQ-4, 2 = CQ-4+ (kvq_tier_e)
    parameter integer KEY_GROUP     = 2,    // G: shipped 128; proxy default
    parameter integer OUTLIER_K     = 0,    // top-k FP16 key channels (CQ-4+)
    parameter integer SCALE_SETS    = 4,    // D-026: persistent scale-bank sets
    parameter integer SCALE_WIDTH   = 16,   // fp16 per-axis scale width
    parameter integer SRAM_DEPTH    = 2,    // records; real capacity per-instantiation
    parameter integer COORD_WIDTH   = 16,   // fp16 input element width
    parameter integer OUT_WIDTH     = 32,   // fp32 decompressed output element (D-010)
    parameter         MASK_FILE     = ""    // hard-reset outlier-mask default hex
                                            // (OUTLIER_K>0 only; CSR-loadable, D-027)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- AXI-Lite Control ----
    input  wire [7:0]              axil_awaddr,
    input  wire                    axil_awvalid,
    output wire                    axil_awready,
    input  wire [31:0]             axil_wdata,
    input  wire                    axil_wvalid,
    output wire                    axil_wready,
    output wire [1:0]              axil_bresp,
    output reg                     axil_bvalid,
    input  wire                    axil_bready,
    input  wire [7:0]              axil_araddr,
    input  wire                    axil_arvalid,
    output wire                    axil_arready,
    output reg  [31:0]             axil_rdata,
    output wire [1:0]              axil_rresp,
    output reg                     axil_rvalid,
    input  wire                    axil_rready,

    // ---- AXI-Stream Write (incoming KV vectors, fp16) ----
    input  wire [COORD_WIDTH-1:0]  s_axis_kv_tdata,
    input  wire                    s_axis_kv_tvalid,
    output reg                     s_axis_kv_tready,
    input  wire                    s_axis_kv_tlast,
    input  wire                    s_axis_kv_tuser,  // 0=K, 1=V

    // ---- AXI-Stream Read (decompressed output, fp32) ----
    output reg  [OUT_WIDTH-1:0]    m_axis_kv_tdata,
    output reg                     m_axis_kv_tvalid,
    input  wire                    m_axis_kv_tready,
    output reg                     m_axis_kv_tlast,

    // ---- APEX D-008: partial-group flush request (CSR FLUSH bit) ----
    input  wire                    flush_req,

    // ---- APEX D-016(a): error interrupt (IRQ_STATUS & IRQ_MASK) ----
    output wire                    irq,

    // ---- Eviction signal to Memory Hierarchy Controller ----
    output wire                    evict_needed,
    output wire [$clog2(SRAM_DEPTH)-1:0] evict_addr,

    // ---- D-027 (S12 stage 4): live-mask validity, the tile INFO_TIER truth
    //      term. COMPUTED (popcount(live bus) == OUTLIER_K, see the mask
    //      section); constant 1 at OUTLIER_K=0.
    output wire                    mask_valid
);

    // =======================================================================
    // Derived geometry
    // =======================================================================
    localparam integer ADDR_WIDTH  = $clog2(SRAM_DEPTH);
    localparam integer VAL_BPV     = (TIER == 0) ? 8 : 4;   // value payload bits/elem
    localparam integer PAY_BITS    = VECTOR_DIM * VAL_BPV;  // packed value payload/token

    // grouped-key path is active only for the per-channel INT4 tiers (CQ-4/CQ-4+)
    localparam bit     KEYG        = (TIER != 0);
    localparam integer GW          = (KEY_GROUP > 1) ? $clog2(KEY_GROUP) : 1;
    localparam [GW-1:0] GRP_LAST   = GW'(KEY_GROUP - 1);

    // ── §4/D-026 byte-aligned record fields (LSB-first, pad to 64b) ─────────
    // KEY record (A2): [tag {ssid[6:0],1'b1}][OUTLIER_K fp16 lanes][D×4 codes]
    // [pad]. Group scales live in the persistent scale_bank_store, one row per
    // committed group, addressed by the ssid stamped in the tag byte.
    localparam integer TAG_BITS       = 8;
    localparam integer KEY_CODES_BITS = VECTOR_DIM * 4;
    localparam integer KEY_LANE_BITS  = OUTLIER_K * COORD_WIDTH;     // outlier lanes
    localparam integer VAL_REC_RAW    = TAG_BITS + SCALE_WIDTH + PAY_BITS;
    localparam integer KEY_REC_RAW    = TAG_BITS + KEY_LANE_BITS + KEY_CODES_BITS;
    localparam integer REC_RAW        = (KEYG && (KEY_REC_RAW > VAL_REC_RAW))
                                      ? KEY_REC_RAW : VAL_REC_RAW;
    localparam integer SRAM_WIDTH     = 64 * ((REC_RAW + 63) / 64);  // pad to 64b
    localparam integer VAL_SCALE_LO   = TAG_BITS;                    // value fp16 scale
    localparam integer VAL_PAY_LO     = TAG_BITS + SCALE_WIDTH;      // value payload
    localparam integer KEY_LANE_LO    = TAG_BITS;                    // outlier lanes
    localparam integer KEY_CODES_LO   = TAG_BITS + KEY_LANE_BITS;    // key per-ch codes
    localparam integer SSET_W         = (SCALE_SETS > 1) ? $clog2(SCALE_SETS) : 1;
    localparam [TAG_BITS-1:0] TAG_VALUE = 8'h00;

    // beat counters: "last element of a token" is D-1, width-exact
    localparam integer CNT_W = $clog2(VECTOR_DIM) + 1;
    localparam [CNT_W-1:0] LAST_ELEM = CNT_W'(VECTOR_DIM - 1);

    // D-016(a): the record SRAM acks a valid read in one cycle; cap the wait
    localparam [3:0] RD_WAIT_MAX = 4'd8;

    localparam [31:0] ISA_VERSION  = 32'h00_02_00_01; // APEX KVQ engine ISA rev

    // a grouped key group writes grp_base+idx: the whole group must fit in SRAM
    // sim-only elaboration guard (out of the synthesis netlist; the check is
    // a build-time config assert, not hardware)
`ifndef SYNTHESIS
    generate
        if (KEYG && (SRAM_DEPTH < KEY_GROUP)) begin : g_cfg_err
            initial $fatal(1, "kvq_engine: SRAM_DEPTH (%0d) < KEY_GROUP (%0d) — grouped key writes would truncate", SRAM_DEPTH, KEY_GROUP);
        end
        if (OUTLIER_K > 0 && VECTOR_DIM > 128) begin : g_mask_cfg_err
            initial $fatal(1, "kvq_engine: VECTOR_DIM (%0d) > 128 with OUTLIER_K > 0 — the D-027 MASK0-3 window addresses 128 channels", VECTOR_DIM);
        end
    endgenerate
`endif

    // =======================================================================
    // AXI-Lite CSR map (KVQ-ISA window; the tile CSR maps onto this via glue)
    // =======================================================================
    localparam [7:0] REG_CTRL             = 8'h00;
    localparam [7:0] REG_STATUS           = 8'h04;
    localparam [7:0] REG_INFO_DIM         = 8'h08;
    localparam [7:0] REG_INFO_TIER        = 8'h0C;
    localparam [7:0] REG_INFO_GROUP       = 8'h10;
    localparam [7:0] REG_INFO_SRAM_DEPTH  = 8'h14;
    localparam [7:0] REG_INFO_CR_K        = 8'h18;
    localparam [7:0] REG_INFO_CR_V        = 8'h1C;
    localparam [7:0] REG_INFO_VERSION     = 8'h20;
    localparam [7:0] REG_OCCUPANCY        = 8'h24;
    localparam [7:0] REG_WRITE_ADDR       = 8'h28;
    localparam [7:0] REG_READ_ADDR        = 8'h2C;
    localparam [7:0] REG_KV_SELECT        = 8'h30;
    localparam [7:0] REG_IRQ_MASK         = 8'h34;
    localparam [7:0] REG_IRQ_STATUS       = 8'h38;
    localparam [7:0] REG_INFO_OUTLIER_K   = 8'h3C;
    localparam [7:0] REG_INFO_SCALE_DEPTH = 8'h40;
    localparam [7:0] REG_INFO_RESID_DEPTH = 8'h44;
    localparam [7:0] REG_INFO_SCALE_SETS  = 8'h48;  // D-026 persistent bank sets
    // D-027 loadable outlier mask (S12). Same numerals as the tile-CSR WALK
    // window (0x5C-0x6C, D-028) — PHYSICALLY SEPARATE BUS, no conflict: these
    // decode in the engine's own AXI-Lite window behind apex_kvq_bank.
    localparam [7:0] REG_MASK0            = 8'h50;  // staged mask, ch 0-31
    localparam [7:0] REG_MASK1            = 8'h54;  //   "     "   ch 32-63
    localparam [7:0] REG_MASK2            = 8'h58;  //   "     "   ch 64-95
    localparam [7:0] REG_MASK3            = 8'h5C;  //   "     "   ch 96-127
    localparam [7:0] REG_MASK_CTRL        = 8'h60;  // W bit0: commit · R: {owned,valid}

    reg        ctrl_enable;
    reg        ctrl_reset;              // 1-cycle soft-reset pulse (D-020)
    reg [ADDR_WIDTH-1:0] write_addr;
    reg [ADDR_WIDTH-1:0] read_addr;
    reg        kv_select;
    reg [3:0]  irq_mask;
    reg [3:0]  irq_status;              // [0] RD_ERR (D-016a) · [1] SB_OVWR
                                        // (D-026 live-set reuse) · [2] MASK_ERR
                                        // · [3] MASK_SWAP (D-027); sticky, W1C
    reg        read_req;                // pulse: a decompress/read was requested

    assign irq = |(irq_status & irq_mask);

    // D-020(B-2): the soft-reset pulse doubles as the datapath abort strobe so
    // an aborted token/group cannot silently complete a cycle later.
    wire dp_clear = ctrl_reset;

    // =======================================================================
    // Token assembly — one token in flight, shift-filled LSB-first
    // =======================================================================
    reg  [VECTOR_DIM*COORD_WIDTH-1:0] tok_vec;
    reg  [$clog2(VECTOR_DIM):0]       in_count;

    // A newly accepted fp16 beat shifts into the assembly register from the top:
    // element k of the token lands at tok_vec[k*COORD_WIDTH +: COORD_WIDTH].
    // Expressed inline at each accept site as
    //   tok_vec <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]}

    // =======================================================================
    // Outlier mask (D-027): live bus = the CSR-committed mask once owned, else
    // the build default (MASK_FILE ROM if given, zeros otherwise). k=0 → tied
    // off and the whole mask CSR window stays reserved.
    // =======================================================================
    localparam bit HAS_MASK_FILE = (MASK_FILE != "");

    // Staged/live/ownership state. HARD-reset-only: soft reset preserves the
    // live mask exactly like scale_bank_store — records persist across D-020,
    // so their decode key must persist too (D-027 §4). Written only from the
    // AXI-Lite CSR block below; every write is param-gated off at OUTLIER_K=0
    // so these flops prune to constants there.
    reg  [VECTOR_DIM-1:0] mask_stage;      // CSR-staged (MASK0-3 readback view)
    reg  [VECTOR_DIM-1:0] mask_live;       // committed mask (datapath source)
    reg                   mask_csr_owned;  // 1 = mask_live owns the bus

    wire [VECTOR_DIM-1:0] outlier_mask_bus;
    wire [VECTOR_DIM-1:0] mask_build_rom;
    generate
        if (OUTLIER_K > 0 && HAS_MASK_FILE) begin : g_mask_rom
            reg [7:0] mask_mem [0:VECTOR_DIM-1];
            initial $readmemh(MASK_FILE, mask_mem);
            genvar mc;
            for (mc = 0; mc < VECTOR_DIM; mc = mc + 1) begin : g_mbit
                assign mask_build_rom[mc] = mask_mem[mc][0];
            end
        end else begin : g_mask_zero
            // OUTLIER_K=0, or the maskless OUTLIER_K>0 build shape (D-027:
            // b128 ships no ROM — the mask arrives by CSR commit and the
            // engine reads invalid until it does)
            assign mask_build_rom = '0;
        end
    endgenerate
    // Until the first successful commit the bus IS the unchanged build-ROM
    // wire (MASK_FILE builds byte-identical out of hard reset — D-027 §6);
    // the OUTLIER_K term keeps k=0 datapaths constant-folded exactly as
    // before (the commit path can only ever write an all-zero mask there).
    assign outlier_mask_bus = ((OUTLIER_K > 0) && mask_csr_owned)
                            ? mask_live : mask_build_rom;

    // =======================================================================
    // Value datapath core (per-token compress + one-channel fp32 readback)
    // =======================================================================
    reg                        cqv_in_valid;
    wire                       cqv_out_valid;
    wire                       cqv_busy;
    wire [VECTOR_DIM*8-1:0]    cqv_codes_unused;
    wire [SCALE_WIDTH-1:0]     cqv_scale;
    wire [VECTOR_DIM*8-1:0]    cqv_pay;
    // S10 dec_op_r: the read-side dequant OPERANDS are registered with a
    // one-beat index skew — the record unpack + 64:1 beat mux runs one cycle
    // AHEAD of the dequant multiply, splitting the rd_data->m_axis cone in
    // two. dec_op_r = {selected code byte, value scale, key scale/raw lane,
    // beat index}; beat 0 is preloaded in the ST_RWAIT exit cycle, beat i+1
    // is captured in the same ST_OUTPUT cycle that advances out_count to i+1
    // (sram_rd_data is stable for the whole burst — no new read can launch
    // while a burst is in flight — so per-beat capture is race-free).
    wire [$clog2(VECTOR_DIM)-1:0] dec_idx;
    wire [31:0]                dec_hat;
    reg [$clog2(VECTOR_DIM)-1:0] dec_idx_r;   // beat index of dec_op_r
    reg [7:0]                  val_code_r;    // selected value code byte
    reg [SCALE_WIDTH-1:0]      val_scale_r;   // value record scale field
    reg [7:0]                  key_code_r;    // selected key code byte
    reg [COORD_WIDTH-1:0]      key_scale_r;   // selected key scale / raw lane

    reg [$clog2(VECTOR_DIM):0] out_count;   // read-side channel beat counter
    assign dec_idx = dec_idx_r;

    cq_value_path #(.D(VECTOR_DIM), .DW(COORD_WIDTH)) u_vpath (
        .clk(clk), .rst_n(rst_n), .clear(dp_clear), .bits(4'(VAL_BPV)),
        .in_valid(cqv_in_valid), .in_vec(tok_vec), .busy(cqv_busy),
        .out_valid(cqv_out_valid), .out_scale(cqv_scale),
        .out_codes(cqv_codes_unused), .out_pay(cqv_pay),
        // S10: the beat's code/scale arrive pre-selected and REGISTERED
        // (dec_op_r); the path-internal dec_idx mux collapses over the
        // replicated byte, leaving only the dequant cone after the register
        .dec_codes({VECTOR_DIM{val_code_r}}), .dec_scale(val_scale_r),
        .dec_idx(dec_idx), .dec_hat(dec_hat)
    );

    // §4 value record image (the SRAM_WIDTH cast zero-fills the pad)
    wire [SRAM_WIDTH-1:0] val_wr_data;
    assign val_wr_data = SRAM_WIDTH'({cqv_pay[PAY_BITS-1:0], cqv_scale, TAG_VALUE});

    // =======================================================================
    // Record SRAM (behavioral; one §4 record per row). Named u_sram: the smoke,
    // flush, collide and scoreboard TBs peek u_sram.mem to score stored records.
    // =======================================================================
    reg                    sram_wr_en;
    reg [ADDR_WIDTH-1:0]   sram_wr_addr;
    reg [SRAM_WIDTH-1:0]   sram_wr_data;
    reg                    sram_rd_en;
    reg [ADDR_WIDTH-1:0]   sram_rd_addr;
    wire [SRAM_WIDTH-1:0]  sram_rd_data;
    wire                   sram_rd_valid;
    wire [ADDR_WIDTH:0]    sram_occupancy;
    wire                   sram_full;

    sram_controller #(
        .SRAM_DEPTH (SRAM_DEPTH),
        .DATA_WIDTH (SRAM_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_sram (
        .clk(clk), .rst_n(rst_n),
        .wr_en(sram_wr_en), .wr_addr(sram_wr_addr), .wr_data(sram_wr_data),
        .rd_en(sram_rd_en), .rd_addr(sram_rd_addr), .rd_data(sram_rd_data),
        .rd_valid(sram_rd_valid), .occupancy(sram_occupancy), .full(sram_full)
    );

    assign evict_needed = sram_full;
    assign evict_addr   = '0;

    // Unpack a stored VALUE payload into per-element signed 8b codes (§5): INT8 →
    // one byte per elem; INT4 → nibble per elem, sign-extended. The dec_op_r
    // skew selects one byte per beat from here (S10).
    wire [VECTOR_DIM*8-1:0] unpacked_codes;
    genvar gu;
    generate
        for (gu = 0; gu < VECTOR_DIM; gu = gu + 1) begin : g_vunpack
            if (VAL_BPV == 8) begin : g_i8
                assign unpacked_codes[gu*8 +: 8] = sram_rd_data[VAL_PAY_LO + gu*8 +: 8];
            end else if (gu % 2 == 0) begin : g_i4lo
                assign unpacked_codes[gu*8 +: 8] =
                    {{4{sram_rd_data[VAL_PAY_LO + (gu/2)*8 + 3]}},
                        sram_rd_data[VAL_PAY_LO + (gu/2)*8 +: 4]};
            end else begin : g_i4hi
                assign unpacked_codes[gu*8 +: 8] =
                    {{4{sram_rd_data[VAL_PAY_LO + (gu/2)*8 + 7]}},
                        sram_rd_data[VAL_PAY_LO + (gu/2)*8 + 4 +: 4]};
            end
        end
    endgenerate

    // =======================================================================
    // Grouped KEY datapath (CQ-4/CQ-4+). The whole block is inside the generate
    // so a TIER-0 build has no out-of-range key-record selects.
    // =======================================================================
    reg                        kp_in_valid;
    reg                        kp_group_start;
    reg                        kp_group_last;
    reg                        kp_flush;        // D-008: 1-cycle flush pulse
    wire                       kp_busy;
    wire                       kp_group_valid;
    wire                       kp_tok_valid;
    wire [GW-1:0]              kp_tok_idx;
    wire [31:0]                kp_dec_hat;

    wire [SRAM_WIDTH-1:0]             key_wr_data;
    wire [VECTOR_DIM*8-1:0]           key_codes_ext;
    wire [COORD_WIDTH-1:0]            key_cur_scale;   // per-beat dequant scale
    wire [COORD_WIDTH-1:0]            key_cur_scale_c; // its NEXT-beat comb select
    assign key_cur_scale = key_scale_r;                // registered view (S10)

    // S10 dec_op_r selection indices: '0 while preloading beat 0 in ST_RWAIT;
    // out_count+1 (and the outlier-lane walk one step ahead) while a beat is
    // being advanced in ST_OUTPUT. Captured into the dec_op_r registers in the
    // same cycle out_count/olane_cnt increment.
    wire dec_adv = (state == ST_OUTPUT) && (!m_axis_kv_tvalid || m_axis_kv_tready);
    wire [$clog2(VECTOR_DIM)-1:0] dec_sel_idx =
        dec_adv ? (out_count[$clog2(VECTOR_DIM)-1:0] + 1'b1) : '0;
    wire dec_ol_hit = out_is_key && outlier_mask_bus[out_count[$clog2(VECTOR_DIM)-1:0]];
    wire [$clog2(VECTOR_DIM):0] dec_sel_olane =
        dec_adv ? (olane_cnt + {{$clog2(VECTOR_DIM){1'b0}}, dec_ol_hit}) : '0;

    // D-026 persistent scale bank plumbing (FSM-side strobes/allocator)
    reg                    bank_wr_en;
    reg  [SSET_W-1:0]      alloc_ptr;       // next scale set to commit
    reg  [SSET_W-1:0]      bank_wr_set;     // latched at strobe time: a 1-token
                                            // group commits and advances in the
                                            // SAME cycle — the write must not
                                            // see the advanced pointer
    reg  [SCALE_SETS-1:0]  set_live;        // committed-at-least-once flags
    reg                    kemit_banked;    // this group's bank row written
    reg                    bank_primed;     // ST_RWAIT one-cycle bank hold
    reg  [$clog2(VECTOR_DIM):0] olane_cnt;  // outlier-lane walk (read side)
    reg                    sb_ovwr_pulse;   // D-026 SB_OVWR strobe → IRQ[1]

    generate
        if (KEYG) begin : g_key
            wire [VECTOR_DIM*COORD_WIDTH-1:0] kp_scales_bus;
            wire [$clog2(KEY_GROUP+1)-1:0]    kp_gout_unused;
            wire [(VECTOR_DIM/2)*8-1:0]       kp_pay_unused;
            wire [VECTOR_DIM*8-1:0]           kp_tok_codes;  // compacted keep codes
            wire [VECTOR_DIM*COORD_WIDTH-1:0] kp_emit_vec;   // emitting token raw fp16

            cq_key_path #(.D(VECTOR_DIM), .DW(COORD_WIDTH), .G(KEY_GROUP)) u_kpath (
                .clk(clk), .rst_n(rst_n), .clear(dp_clear),
                .outlier_mask(outlier_mask_bus),
                .in_valid(kp_in_valid), .in_vec(tok_vec),
                .group_start(kp_group_start), .group_last(kp_group_last),
                .flush(kp_flush), .busy(kp_busy),
                .group_valid(kp_group_valid), .scales_bus(kp_scales_bus),
                .g_out(kp_gout_unused),
                .tok_valid(kp_tok_valid), .tok_idx(kp_tok_idx),
                .tok_pay(kp_pay_unused),
                .tok_codes(kp_tok_codes), .emit_vec(kp_emit_vec),
                // S10: the beat's code and scale/lane arrive pre-selected and
                // REGISTERED (dec_op_r, one-beat skew); the path-internal
                // muxes collapse over the replicated operands
                .dec_codes({VECTOR_DIM{key_code_r}}),
                .dec_scales({VECTOR_DIM{key_cur_scale}}),
                .dec_idx(dec_idx), .dec_hat(kp_dec_hat)
            );

            // D-026 persistent scale bank: one D×16 row per committed group,
            // outlier lanes forced 0 so the image is deterministic (golden
            // packer pins every bit). NOT cleared by dp_clear — records
            // survive a soft reset, so their scales must too.
            wire [VECTOR_DIM*COORD_WIDTH-1:0] bank_wr_row;
            wire [VECTOR_DIM*COORD_WIDTH-1:0] bank_rd_row;
            genvar bc;
            for (bc = 0; bc < VECTOR_DIM; bc = bc + 1) begin : g_bankrow
                assign bank_wr_row[bc*COORD_WIDTH +: COORD_WIDTH] =
                    outlier_mask_bus[bc] ? {COORD_WIDTH{1'b0}}
                                         : kp_scales_bus[bc*COORD_WIDTH +: COORD_WIDTH];
            end
            // S10: the row read launches COMBINATIONALLY in the ST_RWAIT ack
            // cycle (one cycle earlier than the old registered strobe), so
            // bank_rd_row has settled by the bank_primed exit cycle and the
            // beat-0 dec_op_r preload captures the final row — the FSM's
            // prime/exit cycle count is unchanged.
            wire bank_rd_go = (state == ST_RWAIT) && sram_rd_valid
                            && sram_rd_data[0] && !bank_primed;
            scale_bank_store #(.SETS(SCALE_SETS), .D(VECTOR_DIM), .SW(COORD_WIDTH))
            u_bank_store (
                .clk(clk), .rst_n(rst_n),
                .wr_en(bank_wr_en), .wr_set(bank_wr_set), .wr_row(bank_wr_row),
                .rd_en(bank_rd_go), .rd_set(sram_rd_data[1 +: SSET_W]),
                .rd_row(bank_rd_row)
            );

            // Store-side scatter (D-026): codes for all D channels (keep →
            // compacted quant code, outlier → sentinel 4'd1) + OUTLIER_K raw
            // fp16 lanes in ascending channel order. ki walks the compacted
            // keep-code stream; oj walks the lane slots.
            reg [KEY_CODES_BITS-1:0]   key_codes4;
            reg [$clog2(VECTOR_DIM):0] ki;
            integer cc;
            if (OUTLIER_K > 0) begin : g_lanes
                reg [KEY_LANE_BITS-1:0]    key_lanes16;
                reg [$clog2(VECTOR_DIM):0] oj;
                always @* begin
                    key_lanes16 = '0;
                    key_codes4  = '0;
                    ki          = '0;
                    oj          = '0;
                    for (cc = 0; cc < VECTOR_DIM; cc = cc + 1) begin
                        if (outlier_mask_bus[cc]) begin
                            key_lanes16[oj*COORD_WIDTH +: COORD_WIDTH]
                                = kp_emit_vec[cc*COORD_WIDTH +: COORD_WIDTH];
                            key_codes4[cc*4 +: 4] = 4'd1;
                            oj = oj + 1'b1;
                        end else begin
                            key_codes4[cc*4 +: 4] = kp_tok_codes[ki*8 +: 4];
                            ki = ki + 1'b1;
                        end
                    end
                end
                // §4/D-026 key record image: {pad, codes, lanes, tag} LSB-first;
                // tag = {ssid[6:0], 1'b1} with ssid = the committing scale set
                assign key_wr_data = SRAM_WIDTH'({key_codes4, key_lanes16,
                                                  {(7-SSET_W){1'b0}}, alloc_ptr, 1'b1});
                // Read side: NEXT beat's outlier lane / bank scale (S10 skew —
                // dec_sel_olane walks one step ahead of olane_cnt, in step
                // with the sequential ST_OUTPUT beat order); registered into
                // key_scale_r before feeding the dequant.
                localparam integer OLW = (OUTLIER_K > 1) ? $clog2(OUTLIER_K) : 1;
                wire [COORD_WIDTH-1:0] key_sel_lane =
                    sram_rd_data[KEY_LANE_LO + dec_sel_olane[OLW-1:0]*COORD_WIDTH +: COORD_WIDTH];
                assign key_cur_scale_c = outlier_mask_bus[dec_sel_idx]
                                       ? key_sel_lane
                                       : bank_rd_row[dec_sel_idx*COORD_WIDTH +: COORD_WIDTH];
            end else begin : g_nolanes
                always @* begin
                    key_codes4 = '0;
                    ki         = '0;
                    for (cc = 0; cc < VECTOR_DIM; cc = cc + 1) begin
                        key_codes4[cc*4 +: 4] = kp_tok_codes[ki*8 +: 4];
                        ki = ki + 1'b1;
                    end
                end
                assign key_wr_data = SRAM_WIDTH'({key_codes4,
                                                  {(7-SSET_W){1'b0}}, alloc_ptr, 1'b1});
                assign key_cur_scale_c = bank_rd_row[dec_sel_idx*COORD_WIDTH +: COORD_WIDTH];
                // K=0: no outlier lanes — the emitting token's raw fp16 bus
                // and the lane walk counters are structurally unused
                wire _unused_nolanes_ok = &{1'b0, kp_emit_vec, olane_cnt,
                                            dec_sel_olane};
            end

            // Unpack a stored KEY record: sign-extend the per-channel INT4 codes
            genvar gk;
            for (gk = 0; gk < VECTOR_DIM; gk = gk + 1) begin : g_kunpack
                assign key_codes_ext[gk*8 +: 8] =
                    {{4{sram_rd_data[KEY_CODES_LO + gk*4 + 3]}},
                        sram_rd_data[KEY_CODES_LO + gk*4 +: 4]};
            end
        end else begin : g_no_key
            assign kp_busy        = 1'b0;
            assign kp_group_valid = 1'b0;
            assign kp_tok_valid   = 1'b0;
            assign kp_tok_idx     = '0;
            assign kp_dec_hat     = '0;
            assign key_wr_data    = '0;
            assign key_codes_ext  = '0;
            assign key_cur_scale_c = '0;
            // sink the (TIER-0 unused) key-path drive registers, key-record
            // unpack, and D-026 bank plumbing for -Wall
            wire _unused_nokey_ok = &{1'b0, kp_in_valid, kp_group_start,
                                      kp_group_last, kp_flush, key_codes_ext,
                                      bank_wr_en, alloc_ptr,
                                      bank_wr_set, set_live,
                                      kemit_banked, bank_primed, olane_cnt,
                                      dec_sel_olane, sb_ovwr_pulse};
        end
    endgenerate

    // D-020(B-4): the CQ-4+ outlier identity lane preserves the raw fp16 sign.
    // Outliers are stored raw (fp16 field, code +1) and read back as +1·raw
    // through cq_dequant, whose zero-flush emits +0.0 — which would lose the
    // sign of an exact -0.0. For any nonzero value the dequant sign already
    // equals the raw sign, so forcing bit 31 from the stored field is the
    // identity everywhere except -0.0, where it restores 32'h8000_0000. Keep
    // channels and the value path are untouched.
    // (D-026: for outlier channels key_cur_scale IS the stored raw fp16 lane,
    // so the sign source is unchanged; keep channels take the bank scale and
    // are untouched by the force, exactly as before.)
    wire [31:0]            kp_dec_hat_sx = outlier_mask_bus[dec_idx]
                                         ? {key_cur_scale[COORD_WIDTH-1], kp_dec_hat[30:0]}
                                         : kp_dec_hat;

    // =======================================================================
    // Streaming FSM. state[3:0] phase codes are an APEX whitebox interface
    // (verif/kvq/sb parks the engine in each phase to test mid-op reset).
    // =======================================================================
    localparam [3:0] ST_IDLE     = 4'd0,
                     ST_COLLECT  = 4'd1,   // gather a value / CQ-8-key token
                     ST_COMPRESS = 4'd2,   // wait cq_value_path out_valid
                     ST_STORE    = 4'd3,   // write the value record
                     ST_RLOAD    = 4'd4,   // launch a record read (single cycle)
                     ST_RWAIT    = 4'd5,   // capture read data (bounded, D-016a)
                     ST_OUTPUT   = 4'd6,   // stream fp32 beats (honors tready, D-007)
                     ST_KCOLLECT = 4'd7,   // gather one key token of a group
                     ST_KFEED    = 4'd8,   // hand the token to cq_key_path (1 cyc)
                     ST_KACCEPT  = 4'd9,   // await next key token OR flush (D-008)
                     ST_KEMIT    = 4'd10,  // drain cq_key_path emissions → SRAM
                     ST_OFLUSH   = 4'd11;  // hold the final beat until accepted (D-007)
    reg [3:0] state;
    reg       idle;
    reg       out_is_key;                 // read-side dequant lane select

    reg [GW-1:0]         grp_tok_cnt;     // tokens fed to cq_key_path this group
    reg [ADDR_WIDTH-1:0] grp_base;        // SRAM base address of the group

    reg       read_pending;               // D-016(b): read deferred behind a beat
    reg       flush_pending;              // D-008: flush latched mid-token
    reg [3:0] rwait_cnt;                  // D-016(a): bounded ST_RWAIT counter
    reg       rd_err_pulse;               // D-016(a): 1-cycle RD_ERR strobe → IRQ

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            in_count         <= '0;
            out_count        <= '0;
            tok_vec          <= '0;
            s_axis_kv_tready <= 1'b1;
            m_axis_kv_tvalid <= 1'b0;
            m_axis_kv_tdata  <= '0;
            m_axis_kv_tlast  <= 1'b0;
            sram_wr_en       <= 1'b0;
            sram_wr_addr     <= '0;
            sram_wr_data     <= '0;
            sram_rd_en       <= 1'b0;
            sram_rd_addr     <= '0;
            cqv_in_valid     <= 1'b0;
            kp_in_valid      <= 1'b0;
            kp_group_start   <= 1'b0;
            kp_group_last    <= 1'b0;
            kp_flush         <= 1'b0;
            grp_tok_cnt      <= '0;
            grp_base         <= '0;
            out_is_key       <= 1'b0;
            read_pending     <= 1'b0;
            flush_pending    <= 1'b0;
            rwait_cnt        <= '0;
            rd_err_pulse     <= 1'b0;
            bank_wr_en       <= 1'b0;
            alloc_ptr        <= '0;
            bank_wr_set      <= '0;
            set_live         <= '0;
            dec_idx_r        <= '0;
            val_code_r       <= '0;
            val_scale_r      <= '0;
            key_code_r       <= '0;
            key_scale_r      <= '0;
            kemit_banked     <= 1'b0;
            bank_primed      <= 1'b0;
            olane_cnt        <= '0;
            sb_ovwr_pulse    <= 1'b0;
            idle             <= 1'b1;
        end else begin
            // single-cycle strobes default low each cycle
            sram_wr_en     <= 1'b0;
            sram_rd_en     <= 1'b0;
            cqv_in_valid   <= 1'b0;
            kp_in_valid    <= 1'b0;
            kp_group_start <= 1'b0;
            kp_group_last  <= 1'b0;
            kp_flush       <= 1'b0;
            rd_err_pulse   <= 1'b0;
            bank_wr_en     <= 1'b0;
            sb_ovwr_pulse  <= 1'b0;

            // D-008: latch a flush that arrives while a key group is open but
            // cannot be actioned this cycle (mid-token). Consumed in ST_KACCEPT.
            if (flush_req && (state == ST_KCOLLECT || state == ST_KFEED
                              || state == ST_KACCEPT))
                flush_pending <= 1'b1;

            // ── D-020 soft reset: priority over the state case ───────────────
            // Runs BEFORE (and can pre-empt) the case so a reset landing in a
            // single-cycle phase is never overwritten by a later NBA to state.
            // Input-side work is discarded now; dp_clear aborts the cores this
            // same cycle. A read burst in ST_OUTPUT/ST_OFLUSH is the exception:
            // §5 forbids retracting a live beat, so those phases keep running
            // (below) and idle stays 0 until the final beat is accepted.
            if (ctrl_reset) begin
                in_count      <= '0;
                grp_tok_cnt   <= '0;
                read_pending  <= 1'b0;
                flush_pending <= 1'b0;
                rwait_cnt     <= '0;
                // D-026: in-flight bank state only — the bank rows, set_live
                // and alloc_ptr persist with the stored records (D-020); an
                // uncommitted group vanishes with no bank side effect
                kemit_banked  <= 1'b0;
                bank_primed   <= 1'b0;
                olane_cnt     <= '0;
                if (!(state == ST_OUTPUT || state == ST_OFLUSH)) begin
                    state            <= ST_IDLE;
                    idle             <= 1'b1;
                    s_axis_kv_tready <= ctrl_enable;
                    m_axis_kv_tvalid <= 1'b0;
                    m_axis_kv_tlast  <= 1'b0;
                end
            end

            if (!ctrl_reset || state == ST_OUTPUT || state == ST_OFLUSH)
            case (state)
                // ---- idle: wait for a write beat or a decompress request ----
                ST_IDLE: begin
                    idle             <= 1'b1;
                    s_axis_kv_tready <= ctrl_enable;
                    m_axis_kv_tvalid <= 1'b0;
                    m_axis_kv_tlast  <= 1'b0;
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        // D-016(b): an accepted beat always wins; a coincident
                        // read is deferred, never dropped.
                        if (read_req) read_pending <= 1'b1;
                        idle     <= 1'b0;
                        tok_vec  <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count <= 'd1;
                        if (KEYG && !s_axis_kv_tuser) begin
                            grp_base <= write_addr;   // new group base (cnt==0)
                            state    <= ST_KCOLLECT;
                        end else begin
                            state    <= ST_COLLECT;
                        end
                    end else if (read_req || read_pending) begin
                        read_pending     <= 1'b0;
                        // D-016(b): no beat can be accepted during the read walk
                        s_axis_kv_tready <= 1'b0;
                        idle             <= 1'b0;
                        state            <= ST_RLOAD;
                    end
                end

                // ---- value / CQ-8-key: per-token compress ----
                ST_COLLECT: begin
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        tok_vec  <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count <= in_count + 1'b1;
                        if (s_axis_kv_tlast || in_count == LAST_ELEM) begin
                            state            <= ST_COMPRESS;
                            s_axis_kv_tready <= 1'b0;
                            cqv_in_valid     <= 1'b1;   // present token to the core
                        end
                    end
                end

                ST_COMPRESS: begin
                    if (cqv_out_valid) state <= ST_STORE;
                end

                ST_STORE: begin
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= write_addr;
                    sram_wr_data <= val_wr_data;
                    in_count     <= '0;
                    s_axis_kv_tready <= ctrl_enable;
                    state        <= ST_IDLE;
                end

                // ---- grouped key: collect G tokens, then emit their records ----
                ST_KCOLLECT: begin
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        tok_vec  <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count <= in_count + 1'b1;
                        if (s_axis_kv_tlast || in_count == LAST_ELEM) begin
                            state            <= ST_KFEED;
                            s_axis_kv_tready <= 1'b0;
                        end
                    end
                end

                ST_KFEED: begin
                    // present the assembled token to cq_key_path (single cycle)
                    kp_in_valid    <= 1'b1;
                    kp_group_start <= (grp_tok_cnt == '0);
                    kp_group_last  <= (grp_tok_cnt == GRP_LAST);
                    if (grp_tok_cnt == GRP_LAST) begin
                        grp_tok_cnt   <= '0;
                        flush_pending <= 1'b0;   // full group: nothing partial left
                        kemit_banked  <= 1'b0;   // D-026: fresh bank commit
                        state         <= ST_KEMIT;
                    end else begin
                        grp_tok_cnt <= grp_tok_cnt + 1'b1;
                        in_count    <= '0;
                        state       <= ST_KACCEPT;
                    end
                end

                ST_KACCEPT: begin
                    // await the next token of the group — or a flush (D-008)
                    s_axis_kv_tready <= ctrl_enable;
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        tok_vec  <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count <= 'd1;
                        state    <= ST_KCOLLECT;
                    end else if (flush_req || flush_pending) begin
                        // D-008: freeze the partial group (grp_tok_cnt tokens) and
                        // push it through the same scale/quant/emit datapath. An
                        // incoming beat in the same cycle wins (handled above);
                        // the latched flush then fires after that token.
                        flush_pending    <= 1'b0;
                        kp_flush         <= 1'b1;
                        grp_tok_cnt      <= '0;
                        kemit_banked     <= 1'b0;   // D-026: fresh bank commit
                        s_axis_kv_tready <= 1'b0;
                        state            <= ST_KEMIT;
                    end
                end

                ST_KEMIT: begin
                    // cq_key_path scales the group then pulses tok_valid per token
                    if (kp_tok_valid) begin
                        sram_wr_en   <= 1'b1;
                        sram_wr_addr <= grp_base + ADDR_WIDTH'(kp_tok_idx);
                        sram_wr_data <= key_wr_data;
                        // D-026: commit this group's scale row once, into the
                        // set its records' tags name (alloc_ptr, pre-advance).
                        // Reusing a still-live set is the documented host-
                        // contract violation → sticky SB_OVWR (IRQ_STATUS[1]).
                        if (!kemit_banked) begin
                            bank_wr_en          <= 1'b1;
                            bank_wr_set         <= alloc_ptr;  // pre-advance
                            sb_ovwr_pulse       <= set_live[alloc_ptr];
                            set_live[alloc_ptr] <= 1'b1;
                            kemit_banked        <= 1'b1;
                        end
                    end
                    if (kp_group_valid) begin
                        alloc_ptr        <= alloc_ptr + 1'b1;  // commit-time, wraps
                        s_axis_kv_tready <= ctrl_enable;
                        state            <= ST_IDLE;
                    end
                end

                // ---- read / decompress ----
                ST_RLOAD: begin
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= read_addr;
                    rwait_cnt    <= '0;
                    state        <= ST_RWAIT;
                end

                ST_RWAIT: begin
                    // §4/D-026 tag: bit 0 selects the dequant lane; for a key
                    // record, bits [7:1] name the scale set — prime the bank
                    // row for ONE cycle before streaming (the row read itself
                    // launches combinationally in the ack cycle, S10, so the
                    // row has settled by this exit cycle). The dequant
                    // operands are the REGISTERED dec_op_r views (S10), fed
                    // from sram_rd_data + the primed bank row — both stable
                    // through the burst; no new read can launch mid-burst.
                    if (sram_rd_valid && KEYG && sram_rd_data[0] && !bank_primed) begin
                        bank_primed <= 1'b1;
                        out_is_key  <= 1'b1;
                    end else if (sram_rd_valid || bank_primed) begin
                        out_count   <= '0;
                        olane_cnt   <= '0;
                        if (!bank_primed) out_is_key <= 1'b0;  // value record
                        bank_primed <= 1'b0;
                        // S10 dec_op_r: preload beat 0 (dec_sel_* are '0 here)
                        dec_idx_r   <= '0;
                        val_code_r  <= unpacked_codes[7:0];
                        val_scale_r <= sram_rd_data[VAL_SCALE_LO +: SCALE_WIDTH];
                        key_code_r  <= key_codes_ext[7:0];
                        key_scale_r <= key_cur_scale_c;
                        state       <= ST_OUTPUT;
                    end else if (rwait_cnt == RD_WAIT_MAX) begin
                        // D-016(a): unwritten address — the SRAM never acks. Drop
                        // the read (zero beats) and flag RD_ERR.
                        rd_err_pulse <= 1'b1;
                        state        <= ST_IDLE;
                    end else begin
                        rwait_cnt <= rwait_cnt + 1'b1;
                    end
                end

                ST_OUTPUT: begin
                    // D-007: only advance the registered output stage when the
                    // beat currently on the bus (if any) has been accepted; data
                    // stays stable while valid && !ready.
                    if (!m_axis_kv_tvalid || m_axis_kv_tready) begin
                        m_axis_kv_tdata  <= out_is_key ? kp_dec_hat_sx : dec_hat;
                        m_axis_kv_tvalid <= 1'b1;
                        m_axis_kv_tlast  <= (out_count == LAST_ELEM);
                        // D-026: walk the record's outlier lanes in beat order
                        if (out_is_key && outlier_mask_bus[out_count[$clog2(VECTOR_DIM)-1:0]])
                            olane_cnt <= olane_cnt + 1'b1;
                        if (out_count == LAST_ELEM) begin
                            out_count <= '0;
                            state     <= ST_OFLUSH;
                        end else begin
                            out_count <= out_count + 1'b1;
                        end
                        // S10 dec_op_r: capture the NEXT beat's operands in
                        // the same cycle the counters increment (one-beat
                        // skew; the post-LAST_ELEM capture is garbage and is
                        // never consumed — ST_OFLUSH drives no new beat)
                        dec_idx_r  <= dec_sel_idx;
                        val_code_r <= unpacked_codes[dec_sel_idx*8 +: 8];
                        key_code_r <= key_codes_ext[dec_sel_idx*8 +: 8];
                        key_scale_r <= key_cur_scale_c;
                    end
                end

                ST_OFLUSH: begin
                    // D-007: the tlast beat sits in the output register; wait for
                    // its acceptance, then drop tvalid and return to idle.
                    if (m_axis_kv_tready) begin
                        m_axis_kv_tvalid <= 1'b0;
                        m_axis_kv_tlast  <= 1'b0;
                        state            <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // =======================================================================
    // AXI-Lite write channel — single-flight bvalid handshake
    // =======================================================================
    // NOTE (claim-detox 2026-07-13; A2/D-026 landed 2026-07-14): these INFO_CR
    // figures are the CODEC-level denominators — key scales amortized over
    // KEY_GROUP, no tag, no 64b pad, no bank-row term. Post-A2 the stored
    // records approach this (whole-KV stored 3.16-3.51x at shipped k<=2) but
    // the CSR formula still omits tag/pad/lanes. The honest three-way
    // accounting (stored / codec / this CSR) is pinned in
    // golden/tests/test_effective_bits.py; never quote CR_* externally.
    localparam integer VAL_EFF_DEN = VECTOR_DIM * VAL_BPV + SCALE_WIDTH;
    localparam integer KEY_EFF_DEN = (TIER == 0)
                                   ? (VECTOR_DIM * VAL_BPV + SCALE_WIDTH)
                                   : (VECTOR_DIM * 4 +
                                      (SCALE_WIDTH * VECTOR_DIM) / KEY_GROUP);
    localparam [31:0] CR_V_FIXED = 32'((VECTOR_DIM * COORD_WIDTH * 256) / VAL_EFF_DEN);
    localparam [31:0] CR_K_FIXED = 32'((VECTOR_DIM * COORD_WIDTH * 256) / KEY_EFF_DEN);

    wire wr_accept = axil_awvalid && axil_wvalid && (!axil_bvalid || axil_bready);
    assign axil_awready = axil_wvalid && (!axil_bvalid || axil_bready);
    assign axil_wready  = axil_awvalid && (!axil_bvalid || axil_bready);
    assign axil_bresp   = 2'b00;

    // ── D-027 mask commit decode (state in the mask section above; faults on
    //    the SB_OVWR discipline: sticky, W1C, same-cycle set beats clear) ────
    function automatic [7:0] f_mask_popcount(input [VECTOR_DIM-1:0] v);
        integer pi;
        begin
            f_mask_popcount = '0;
            for (pi = 0; pi < VECTOR_DIM; pi = pi + 1)
                f_mask_popcount = f_mask_popcount + {7'b0, v[pi]};
        end
    endfunction

    // 128-bit CSR view of the staged mask: constant part-selects for the four
    // MASK words; the VECTOR_DIM'() truncation on the write side makes
    // beyond-D mask bits RAZ/WI in every config.
    wire [127:0] mask_stage_x = 128'(mask_stage);

    // mask_valid is COMPUTED from the LIVE bus, never stored: a MASK_FILE
    // build reads valid out of hard reset, the maskless OUTLIER_K>0 build
    // reads invalid until the first commit, and a malformed ROM (popcount !=
    // OUTLIER_K) truthfully reads invalid — the D-024 "INFO_TIER never lies"
    // rule extended to bad build inputs. Exported (stage 4) as the port the
    // tile INFO_TIER bit-2 term consumes.
    assign mask_valid = (f_mask_popcount(outlier_mask_bus) == 8'(OUTLIER_K));

    // Commit decode (structural no-ops at OUTLIER_K=0 — window stays
    // reserved). "Logically occupied" (D-027 §3) = records at rest OR an open
    // key group anywhere between first accepted beat and last emitted record:
    // the whole span where a swap could interleave with a mask consumer
    // (grouped amax skip / scale park / record scatter / bank-row zeroing on
    // the store side; the outlier-lane walk, bank-vs-lane scale select and
    // sign force on a key read burst — which needs a written record, so the
    // occupancy term already covers it).
    wire mask_commit_fire  = wr_accept && (axil_awaddr == REG_MASK_CTRL)
                           && axil_wdata[0] && (OUTLIER_K > 0);
    wire mask_commit_legal = (f_mask_popcount(mask_stage) == 8'(OUTLIER_K));
    wire mask_key_open     = KEYG && (state == ST_KCOLLECT || state == ST_KFEED
                                   || state == ST_KACCEPT || state == ST_KEMIT);
    wire mask_swap_haz     = (sram_occupancy != '0) || mask_key_open;
    wire mask_err_set      = mask_commit_fire && !mask_commit_legal;
    wire mask_swap_set     = mask_commit_fire && mask_commit_legal && mask_swap_haz;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axil_bvalid  <= 1'b0;
            ctrl_enable  <= 1'b0;
            ctrl_reset   <= 1'b0;
            write_addr   <= '0;
            read_addr    <= '0;
            kv_select    <= 1'b0;
            irq_mask     <= '0;
            irq_status   <= '0;
            read_req     <= 1'b0;
            // D-027 §4: HARD reset only — restores the build default (the
            // mux in the mask section falls back to mask_build_rom)
            mask_stage     <= '0;
            mask_live      <= '0;
            mask_csr_owned <= 1'b0;
        end else begin
            ctrl_reset <= 1'b0;    // 1-cycle pulses
            read_req   <= 1'b0;
            // D-016(a)/D-026/D-027: RD_ERR/SB_OVWR/MASK_ERR/MASK_SWAP are
            // sticky; a set (FSM pulse or same-cycle commit fault) beats a
            // same-cycle W1C
            irq_status <= irq_status | {mask_swap_set, mask_err_set,
                                        sb_ovwr_pulse, rd_err_pulse};
            if (axil_bvalid && axil_bready) axil_bvalid <= 1'b0;
            if (wr_accept) begin
                axil_bvalid <= 1'b1;
                case (axil_awaddr)
                    REG_CTRL: begin
                        ctrl_reset  <= axil_wdata[0];
                        ctrl_enable <= axil_wdata[1];
                    end
                    REG_WRITE_ADDR: write_addr <= axil_wdata[ADDR_WIDTH-1:0];
                    REG_READ_ADDR: begin
                        read_addr <= axil_wdata[ADDR_WIDTH-1:0];
                        read_req  <= 1'b1;     // writing READ_ADDR launches a read
                    end
                    REG_KV_SELECT:  kv_select  <= axil_wdata[0];
                    REG_IRQ_MASK:   irq_mask   <= axil_wdata[3:0];
                    REG_IRQ_STATUS: irq_status <= (irq_status & ~axil_wdata[3:0])
                                                  | {mask_swap_set, mask_err_set,
                                                     sb_ovwr_pulse, rd_err_pulse};  // W1C
                    // D-027 staged mask words + commit. Write-ignored at
                    // OUTLIER_K=0 (window reserved). Beyond-D staged bits
                    // drop at the VECTOR_DIM'() truncation (WI); a commit
                    // takes effect iff popcount(staged) == OUTLIER_K, else
                    // sticky MASK_ERR; an effective commit while occupied
                    // also raises MASK_SWAP (both via the OR terms above).
                    REG_MASK0: if (OUTLIER_K > 0)
                        mask_stage <= VECTOR_DIM'({mask_stage_x[127:32], axil_wdata});
                    REG_MASK1: if (OUTLIER_K > 0)
                        mask_stage <= VECTOR_DIM'({mask_stage_x[127:64], axil_wdata,
                                                   mask_stage_x[31:0]});
                    REG_MASK2: if (OUTLIER_K > 0)
                        mask_stage <= VECTOR_DIM'({mask_stage_x[127:96], axil_wdata,
                                                   mask_stage_x[63:0]});
                    REG_MASK3: if (OUTLIER_K > 0)
                        mask_stage <= VECTOR_DIM'({axil_wdata, mask_stage_x[95:0]});
                    REG_MASK_CTRL: if (mask_commit_fire && mask_commit_legal) begin
                        mask_live      <= mask_stage;
                        mask_csr_owned <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    // =======================================================================
    // AXI-Lite read channel — single-flight rvalid handshake
    // =======================================================================
    assign axil_arready = !axil_rvalid || axil_rready;
    assign axil_rresp   = 2'b00;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axil_rvalid <= 1'b0;
            axil_rdata  <= '0;
        end else begin
            if (axil_rvalid && axil_rready) axil_rvalid <= 1'b0;
            if (axil_arvalid && (!axil_rvalid || axil_rready)) begin
                axil_rvalid <= 1'b1;
                case (axil_araddr)
                    REG_CTRL:            axil_rdata <= {30'b0, ctrl_enable, 1'b0};
                    // STATUS: [0] idle · [1] RD_ERR (IRQ_STATUS[0] mirror) ·
                    // [2] rsvd · [3] sram_full. D-020(B-2): idle also requires the
                    // datapath cores quiesced (§5 busy covers ALL in-flight work).
                    REG_STATUS:          axil_rdata <= {28'b0, sram_full, 1'b0, irq_status[0],
                                                        idle && !cqv_busy && !kp_busy};
                    REG_INFO_DIM:        axil_rdata <= 32'(VECTOR_DIM);
                    REG_INFO_TIER:       axil_rdata <= 32'(TIER);
                    REG_INFO_GROUP:      axil_rdata <= 32'(KEY_GROUP);
                    REG_INFO_SRAM_DEPTH: axil_rdata <= 32'(SRAM_DEPTH);
                    REG_INFO_CR_K:       axil_rdata <= CR_K_FIXED;
                    REG_INFO_CR_V:       axil_rdata <= CR_V_FIXED;
                    REG_INFO_VERSION:    axil_rdata <= ISA_VERSION;
                    REG_OCCUPANCY:       axil_rdata <= {{(31-ADDR_WIDTH){1'b0}}, sram_occupancy};
                    REG_WRITE_ADDR:      axil_rdata <= {{(32-ADDR_WIDTH){1'b0}}, write_addr};
                    REG_READ_ADDR:       axil_rdata <= {{(32-ADDR_WIDTH){1'b0}}, read_addr};
                    REG_KV_SELECT:       axil_rdata <= {31'b0, kv_select};
                    REG_IRQ_MASK:        axil_rdata <= {28'b0, irq_mask};
                    REG_IRQ_STATUS:      axil_rdata <= {28'b0, irq_status};
                    REG_INFO_OUTLIER_K:  axil_rdata <= 32'(OUTLIER_K);
                    REG_INFO_SCALE_DEPTH:axil_rdata <= 32'(VECTOR_DIM);
                    REG_INFO_RESID_DEPTH:axil_rdata <= 32'(KEY_GROUP);
                    REG_INFO_SCALE_SETS: axil_rdata <= 32'(SCALE_SETS);
                    // D-027: staged-mask readback (the STAGED value, not the
                    // live mask — §1) + commit status. Reserved at k=0.
                    REG_MASK0:     axil_rdata <= (OUTLIER_K > 0)
                                              ? mask_stage_x[31:0]   : 32'hDEAD_BEEF;
                    REG_MASK1:     axil_rdata <= (OUTLIER_K > 0)
                                              ? mask_stage_x[63:32]  : 32'hDEAD_BEEF;
                    REG_MASK2:     axil_rdata <= (OUTLIER_K > 0)
                                              ? mask_stage_x[95:64]  : 32'hDEAD_BEEF;
                    REG_MASK3:     axil_rdata <= (OUTLIER_K > 0)
                                              ? mask_stage_x[127:96] : 32'hDEAD_BEEF;
                    // R: bit0 mask_valid (computed: popcount(live)==OUTLIER_K)
                    //    bit1 mask_csr_owned (live source = CSR, not the ROM)
                    REG_MASK_CTRL: axil_rdata <= (OUTLIER_K > 0)
                                              ? {30'b0, mask_csr_owned, mask_valid}
                                              : 32'hDEAD_BEEF;
                    default:             axil_rdata <= 32'hDEAD_BEEF;
                endcase
            end
        end
    end

    // Sink structurally-unused input bits so the sequencer stays -Wall clean
    // with no waivers (pad/tag bits of sram_rd_data, cqv_pay's high half; the
    // D-027 mask words consume axil_wdata param-dependently — beyond-D bits
    // and the whole k=0 window are write-ignored — so the full bus is sunk.
    // outlier_mask_bus left the list: mask_valid reads it in every build).
    wire _unused_ok = &{1'b0, axil_wdata, axil_araddr[1:0],
                        axil_awaddr[1:0], cqv_pay, sram_rd_data,
                        key_code_r, dec_sel_olane};

endmodule

`default_nettype wire
