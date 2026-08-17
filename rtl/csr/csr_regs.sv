// csr_regs.sv — CSR: one 32-bit register window for the whole APEX tile.
//
// Implements: ARCHITECTURE.md §7 (register map, KVQ-ISA style), §1 CSR row,
//             D-008 (FLUSH trigger for the KVQ partial-group flush),
//             D-011 (THRESHOLD_REG per layer), and the run2 lesson via the
//             PERF_* per-block busy-cycle counters ("MEASURE load vs compute
//             vs drain, don't discover it").
//
// ── Bus (documented simple bus; an AXI-Lite adapter is FUTURE WORK) ────────
//   Request : {addr[7:0], wdata[31:0], write, read} sampled on posedge clk.
//             write/read are single-cycle strobes; fully pipelined (a new
//             request may be issued every cycle). If write && read are both
//             asserted the WRITE is performed and rdata returns the
//             PRE-write value (read-before-write).
//   Response: exactly one cycle later: ready pulses for one cycle; rdata
//             holds the read data (32'h0 for a write-only request).
//   Alignment: addr[1:0] != 0 (unaligned) is treated as RESERVED: the write
//             is ignored, the read returns 32'hDEAD_BEEF (KVQ ISA
//             convention). Reserved / unmapped word addresses behave the
//             same way. Writes to reserved or read-only registers have NO
//             side effects (but are still acknowledged with ready).
//
// ── Register map (word-aligned byte addresses, §7 exactly) ─────────────────
//   0x00 CTRL        RW  [0] enable (level out). [1] soft_reset: write-1 =>
//                        1-cycle soft_reset pulse out (self-clearing, reads
//                        0). Consumers (SEQ per its abort contract, KVQ per
//                        D-020) latch the pulse; CSR contents are NOT
//                        affected by soft_reset — only hard rst_n clears
//                        this file, so status stays observable.
//   0x04 STATUS      RO/W1C [0] idle = ~|block_busy_i (RO, computed
//                        combinationally — it can never lie). [1]
//                        desc_error sticky: set by a desc_error_i pulse,
//                        cleared by writing 1 to bit 1 (W1C); a set and a
//                        W1C in the same cycle => stays SET (set wins —
//                        never lose an error). [8+i] busy bit of block i
//                        (RO, live block_busy_i). Other bits read 0; writes
//                        to bits other than [1] are ignored.
//   0x08 INFO_N      RO  MXE_N               (from FROZEN apex_pkg)
//   0x0C INFO_D      RO  CFG_D parameter     (configured KVQ D)
//   0x10 INFO_G      RO  CFG_G parameter     (configured KVQ group size)
//   0x14 INFO_TIER   RO  supported-tier bitmap, bit index = kvq_tier_e code:
//                        [0] CQ8, [1] CQ4, [2] CQ4P. Reports the TIERS
//                        parameter — the integration (apex_top) passes the
//                        TRUTH of its build: CQ4P only when a real outlier
//                        mask is configured (F-2 closure; default 3'b111
//                        keeps standalone-suite behavior).
//   0x18 INFO_VERSION RO APEX_VERSION        (from FROZEN apex_pkg)
//   0x1C reserved    ->  0xDEADBEEF
//   0x20 TIER_CTRL   RW  [1:0] KVQ tier select (kvq_tier_e; the reserved
//                        encoding 2'b11 is CLAMPED to CQ4 = the §4 default,
//                        and reads back clamped). [8] tip_override (TIP
//                        decision override enable). Reset: {CQ4, override=0}.
//   0x24 THRESHOLD_REG RW [4:0] TIP per-layer threshold (D-011/D-017;
//                        downstream tip_decide clamps 0 -> 1). Reset: 1.
//   0x28 FLUSH       WO  any write => 1-cycle flush pulse out (KVQ partial-
//                        group flush trigger, D-008 — wired to KVQ
//                        flush_req at tile integration). Reads 0.
//   0x2C IMPORTANCE_BASE RW/window: write sets imp_rd_addr[IMP_AW-1:0]
//                        (TIP accumulator read address); read returns
//                        {14'b0, imp_rd_tier_i[1:0], imp_rd_data_i[15:0]}
//                        — TIP's combinational readout window (tip_top
//                        imp_rd_* port group).
//   0x30 PERF_CTRL   RW  [0] perf_en (level). [1] write-1 = clear ALL
//                        counters (self-clearing, reads 0). Clear and count
//                        in the same cycle => clear wins. A single write
//                        may set both bits: counters restart from 0.
//   0x34 PERF_CYCLES RO  timebase: +1 every cycle perf_en is set.
//   0x38+4i PERF_BUSY_i RO (i = 0..N_BLOCKS-1): +1 every cycle perf_en &&
//                        block_busy_i[i] — per-block busy-cycle counter, so
//                        utilization = PERF_BUSY_i / PERF_CYCLES exposes a
//                        load-bound block directly (run2 lesson). Finer
//                        load/compute/drain phase events plug in as extra
//                        lanes at integration.
//   All counters are PERF_W bits, SATURATE at all-ones (never wrap — a
//   pegged counter is honest, a wrapped one lies), read zero-extended.
//   above the last PERF_BUSY_i: reserved -> 0xDEADBEEF.
//
// Hard reset (rst_n, synchronous active-low): everything to the documented
// reset values; counters to 0.

module csr_regs
  import apex_pkg::*;
#(
  parameter int unsigned N_BLOCKS = 8,    // busy lanes / PERF_BUSY_* counters (1..8)
  parameter int unsigned CFG_D    = 64,   // INFO_D
  parameter int unsigned CFG_G    = 128,  // INFO_G
  parameter int unsigned IMP_AW   = 7,    // IMPORTANCE_BASE addr width (tip BLOCKS=128)
  parameter int unsigned PERF_W   = 32,   // counter width (≤ 32; small builds for TB saturation)
  parameter logic [2:0]  TIERS    = 3'b111 // INFO_TIER truth bitmap (see header)
) (
  input  logic                clk,
  input  logic                rst_n,             // synchronous, active-low

  // simple bus (see header; AXI-Lite adapter is future work)
  input  logic [7:0]          addr,
  input  logic [31:0]         wdata,
  input  logic                write,
  input  logic                read,
  output logic [31:0]         rdata,
  output logic                ready,

  // CTRL
  output logic                enable,
  output logic                soft_reset,        // 1-cycle pulse

  // STATUS
  input  logic [N_BLOCKS-1:0] block_busy_i,      // live per-block busy
  input  logic                desc_error_i,      // pulse: sets the sticky bit
  output logic                desc_error_sticky,

  // TIER_CTRL / THRESHOLD_REG (TIP/KVQ quasi-static controls)
  output kvq_tier_e           tier_sel,
  output logic                tip_override,
  output logic [4:0]          threshold,

  // FLUSH (D-008)
  output logic                flush,             // 1-cycle pulse

  // IMPORTANCE_BASE window (TIP tip_top imp_rd_* port group)
  output logic [IMP_AW-1:0]   imp_rd_addr,
  input  logic [15:0]         imp_rd_data_i,
  input  logic [1:0]          imp_rd_tier_i
);

  if (N_BLOCKS < 1 || N_BLOCKS > 8) begin : gen_nblocks_err
    $error("csr_regs: N_BLOCKS must be 1..8 (STATUS busy field is [15:8])");
  end
  if (PERF_W < 2 || PERF_W > 32) begin : gen_perfw_err
    $error("csr_regs: PERF_W must be 2..32");
  end
  if (IMP_AW < 1 || IMP_AW > 7) begin : gen_impaw_err
    $error("csr_regs: IMP_AW must be 1..7 (IMPORTANCE_BASE field is [6:0])");
  end

  // ── word-index decode ──────────────────────────────────────────────────────
  localparam logic [5:0] A_CTRL      = 6'h00;  // 0x00
  localparam logic [5:0] A_STATUS    = 6'h01;  // 0x04
  localparam logic [5:0] A_INFO_N    = 6'h02;  // 0x08
  localparam logic [5:0] A_INFO_D    = 6'h03;  // 0x0C
  localparam logic [5:0] A_INFO_G    = 6'h04;  // 0x10
  localparam logic [5:0] A_INFO_TIER = 6'h05;  // 0x14
  localparam logic [5:0] A_INFO_VER  = 6'h06;  // 0x18
  localparam logic [5:0] A_TIER_CTRL = 6'h08;  // 0x20
  localparam logic [5:0] A_THRESH    = 6'h09;  // 0x24
  localparam logic [5:0] A_FLUSH     = 6'h0A;  // 0x28
  localparam logic [5:0] A_IMP       = 6'h0B;  // 0x2C
  localparam logic [5:0] A_PERF_CTRL = 6'h0C;  // 0x30
  localparam logic [5:0] A_PERF_CYC  = 6'h0D;  // 0x34
  localparam logic [5:0] A_PERF_EVT0 = 6'h0E;  // 0x38 + 4*i

  logic [5:0] widx;
  logic       aligned;
  assign widx    = addr[7:2];
  assign aligned = (addr[1:0] == 2'b00);

  logic wr_en, w1c_fire, wr_flush, wr_srst, wr_pclr;
  assign wr_en    = write && aligned;
  assign w1c_fire = wr_en && (widx == A_STATUS)    && wdata[1];
  assign wr_srst  = wr_en && (widx == A_CTRL)      && wdata[1];
  assign wr_flush = wr_en && (widx == A_FLUSH);
  assign wr_pclr  = wr_en && (widx == A_PERF_CTRL) && wdata[1];

  // ── state ──────────────────────────────────────────────────────────────────
  logic              perf_en;
  logic [PERF_W-1:0] perf_cyc;
  logic [PERF_W-1:0] perf_cnt [N_BLOCKS];

  localparam logic [PERF_W-1:0] PERF_MAX = {PERF_W{1'b1}};

  // ── read mux (combinational, PRE-write values; registered into rdata) ─────
  logic        idle;
  logic [7:0]  busy8;
  assign idle  = ~|block_busy_i;
  assign busy8 = 8'(block_busy_i);

  logic [31:0] rmux;
  always_comb begin
    rmux = 32'hDEAD_BEEF;                      // reserved / unaligned default
    if (aligned) begin
      case (widx)
        A_CTRL:      rmux = {30'b0, 1'b0, enable};     // soft_reset reads 0
        A_STATUS:    rmux = {16'b0, busy8, 6'b0, desc_error_sticky, idle};
        A_INFO_N:    rmux = 32'(MXE_N);
        A_INFO_D:    rmux = 32'(CFG_D);
        A_INFO_G:    rmux = 32'(CFG_G);
        A_INFO_TIER: rmux = {29'b0, TIERS};            // build-truth bitmap
        A_INFO_VER:  rmux = APEX_VERSION;
        A_TIER_CTRL: rmux = {23'b0, tip_override, 6'b0, 2'(tier_sel)};
        A_THRESH:    rmux = {27'b0, threshold};
        A_FLUSH:     rmux = 32'h0;                     // WO: reads 0
        A_IMP:       rmux = {14'b0, imp_rd_tier_i, imp_rd_data_i};
        A_PERF_CTRL: rmux = {30'b0, 1'b0, perf_en};    // clear bit reads 0
        A_PERF_CYC:  rmux = 32'(perf_cyc);
        default: begin
          for (int unsigned i = 0; i < N_BLOCKS; i++)
            if (widx == A_PERF_EVT0 + 6'(i)) rmux = 32'(perf_cnt[i]);
        end
      endcase
    end
  end

  // ── registers ──────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ready             <= 1'b0;
      rdata             <= '0;
      enable            <= 1'b0;
      soft_reset        <= 1'b0;
      desc_error_sticky <= 1'b0;
      tier_sel          <= KVQ_CQ4;            // §4 production default
      tip_override      <= 1'b0;
      threshold         <= 5'd1;
      flush             <= 1'b0;
      imp_rd_addr       <= '0;
      perf_en           <= 1'b0;
      perf_cyc          <= '0;
      for (int unsigned i = 0; i < N_BLOCKS; i++) perf_cnt[i] <= '0;
    end else begin
      // response pipeline (1-cycle, fully pipelined)
      ready <= read || write;
      rdata <= read ? rmux : 32'h0;

      // self-clearing pulse outputs
      soft_reset <= wr_srst;
      flush      <= wr_flush;

      // RW register writes (RO / WO-pulse / reserved fall to the default arm)
      if (wr_en) begin
        case (widx)
          A_CTRL:      enable <= wdata[0];
          A_TIER_CTRL: begin
            tier_sel     <= (wdata[1:0] == 2'b11) ? KVQ_CQ4
                                                  : kvq_tier_e'(wdata[1:0]);
            tip_override <= wdata[8];
          end
          A_THRESH:    threshold   <= wdata[4:0];
          A_IMP:       imp_rd_addr <= wdata[IMP_AW-1:0];
          A_PERF_CTRL: perf_en     <= wdata[0];
          default: ;   // no side effects anywhere else
        endcase
      end

      // desc_error sticky: set WINS over a same-cycle W1C
      if (desc_error_i)  desc_error_sticky <= 1'b1;
      else if (w1c_fire) desc_error_sticky <= 1'b0;

      // PERF counters: clear wins over count; saturate, never wrap
      if (wr_pclr) begin
        perf_cyc <= '0;
        for (int unsigned i = 0; i < N_BLOCKS; i++) perf_cnt[i] <= '0;
      end else if (perf_en) begin
        if (perf_cyc != PERF_MAX) perf_cyc <= perf_cyc + 1'b1;
        for (int unsigned i = 0; i < N_BLOCKS; i++)
          if (block_busy_i[i] && perf_cnt[i] != PERF_MAX)
            perf_cnt[i] <= perf_cnt[i] + 1'b1;
      end
    end
  end

  // wdata bits outside implemented fields are intentionally ignored
  logic unused_ok;
  assign unused_ok = &{1'b0, wdata[31:9], wdata[7:IMP_AW]};

endmodule
