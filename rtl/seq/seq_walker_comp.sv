// seq_walker_comp.sv — B1: on-tile fp16 scale composition + store-time scale
//                      cache. Produces the qs_*/cs_* words the host folds today.
//
// Implements: docs/design/B1_WALKER.md §2 (the S-3/S-4 bit-exact numeric
//             contract), §A-2 (scale source = snoop + cq_fp_pkg recompute),
//             §A-1 (CQ-8 only), §5 stream semantics, D-006 (job-style
//             error pulse + sticky, NO state change on a rejected frame).
//             D-021 lineage: the single-f32 narrowing (S-3) is load-bearing.
// Arbiter:    verif/top/l3/walker_composite_golden.py
//             (score_composite / p_requant_composite — fp16 bit patterns in,
//             uint32 bit patterns out, exactly this module's port semantics).
//             Spec emitter: verif/top/l3/gen_l3_vectors.py:375-384, :133-140.
//
// WHY THIS BLOCK EXISTS: the host computes cs/qs by reading the fp16 scale
// taps and folding on the CPU — the O(T) round-trip term OPTIMIZATION.md:86
// names. Folding on-tile removes it.
//
// ── the scale-source problem and its solution (§A-2) ─────────────────────────
// score_phase emits ALL T cs words and ALL T qs words BEFORE the K-record
// reads that surface s_k[t] on fs_* (gen_l3_vectors.py:379-384 vs :393). The
// host gets away with it by precomputing from the golden; a tap-fed unit
// cannot. But the KVQ already holds the answer: at CQ-8 the stored per-record
// fp16 scale is BIT-IDENTICAL to the feeder's read-time s_k[t]/s_v[t] (the
// row's max |code| is always exactly 127, so RNE_f16(amax/127) == s_record;
// verified over the suite tensors and 13k random tokens). cqv_scale is an
// internal wire (kvq_engine.sv:255/:279/:290), so rather than port it out of
// the B2 lane's verified engine, this block SNOOPS the same fp16 beats the
// engine consumes (the nets behind apex_top's passive dbg_f16_* mirror) and
// recomputes the scale with cq_fp_pkg::scale_from_amax — the SAME package
// function the engine itself uses. Function identity by shared construction,
// not re-derivation.
//
// ── bit-exactness proofs (verified in python vs the golden; §2, notes §2) ────
// P1  INPUT DOMAIN. Every scale producer floors at EPS = 2^-14 = 16'h0400 and
//     can otherwise emit only a positive NORMAL fp16 (seam_feeder_quant.sv:175,
//     :192; cq_fp_pkg.sv:56-57; apex_scale_quant.sv:298,:313). The unpack below
//     assumes positive-normal ({1'b1, m} hidden bit, plain exponent rebias); a
//     denormal or negative input would silently corrupt it while still passing
//     the software oracle's own asserts, so verif/ binds an SVA on every
//     consumed scale. This is a CHECKED precondition, not an assumption.
// P2  qs (S-4 fold) is a PURE EXPONENT REBIAS — zero arithmetic. Exhaustively
//     verified over all 30,720 positive-normal fp16: f32 = ((e+97)<<23)|(m<<13).
//     Derivation: (e-15) - 15 + 127 = e + 97. Never overflows (e in [1,30] ->
//     e32 in [98,127]); fp16-grade (low 13 bits zero) holds STRUCTURALLY.
// P3  cs @ D=64: the constant 2^SCORE_FRAC/sqrt(D) = 2^10/8 = 128 = 2^7 EXACTLY.
//     The 11x11 significand product is <= 22 bits, exact in f32, so the whole
//     composite is exact — no rounding hardware at all. E32 in [106,165]:
//     always normal, no clamp logic reachable.
// P4  cs @ D=128: the constant is 64*sqrt(2) = 90.509668..., NOT a power of
//     two. The f32-rounded constant is a DIFFERENT number and mismatches the
//     golden on 86,592 of 419,629 reachable product significands, so the
//     hardware constant is the full f64 significand CM (53 bits).
// P5  cs @ D=128 double rounding is INNOCUOUS: over ALL 419,629 reachable
//     product significands, RN24(RN53(P*CM)) == RN24(P*CM) — there are zero
//     exact ties at the f32 boundary. So one RNE24 of the exact 75-bit product
//     reproduces the golden's two-step f64->f32 narrowing. Exponents cannot
//     affect it (E32 in [105,166]: no f32 subnormal or overflow is reachable),
//     which is what makes the significand-space proof exhaustive.
// P6  Rounding mode MUST be RNE: the golden is numpy .astype(np.float32), the
//     IEEE-754 default. ~49.6% of reachable D=128 products round up, so a
//     truncating implementation diverges immediately.
`ifndef SEQ_WALKER_COMP_SV
`define SEQ_WALKER_COMP_SV

module seq_walker_comp
  import apex_pkg::*;
  import seq_walker_pkg::*;
#(
  parameter int unsigned CFG_D = 64          // build geometry; 64 or 128
) (
  input  logic                       clk,
  input  logic                       rst_n,      // synchronous, active low

  // ── store-time scale harvest (§A-2): the accepted fp16 beats going into
  //    KVQ, plus the record address the host programmed in WRITE_ADDR ────────
  input  logic                       snp_valid,  // kv_s_tvalid && kv_s_tready
  input  logic [15:0]                snp_data,   // one fp16 element per beat
  input  logic                       snp_last,   // tlast: element D-1
  input  logic [WALK_SC_AW-1:0]      snp_addr,   // KVQ WRITE_ADDR (record idx)
  input  logic                       snp_flush,  // drop the in-flight record
                                                 // (D-020 soft-reset abort)

  // ── s_q, latched from the ss_* tap at the end of q-inject ────────────────
  input  logic                       sq_valid,
  input  logic [15:0]                sq_data,

  // ── composite request / response ─────────────────────────────────────────
  input  logic                       req_valid,
  output logic                       req_ready,
  input  logic                       req_is_qs,  // 0 = cs (S-3), 1 = qs (S-4)
  input  logic [WALK_SC_AW-1:0]      req_idx,    // scale-cache entry
  output logic                       res_valid,
  input  logic                       res_ready,
  output logic [31:0]                res_data,

  // ── §3-style error pulses (consumed, sticky in glue, NO state change) ────
  output logic                       err_frame,  // tlast at the wrong beat
  output logic                       err_stale,  // requested an unwritten entry
  // DBG-v5 (2026-08-08): which of the TWO stale terms fired at the last
  // error — [0]=sc_val miss (record cache), [1]=s_q_val miss (the q-scale
  // tap). Purely observational; latched on the same edge as err_stale.
  output logic                       dbg_stale_sc,
  output logic                       dbg_stale_sq
);

  localparam int unsigned BEAT_W = $clog2(WALK_D_MAX + 1);

  if (!(CFG_D == 64 || CFG_D == 128)) begin : g_chk_d
    $error("seq_walker_comp: CFG_D must be 64 or 128 (D-021)");
  end

  // ── fp16 unpack (P1: positive-normal by contract, SVA-checked in verif/) ──
  // {exp[4:0], mant[9:0]}; the hidden bit is implicit. Sign is never consumed
  // — every scale is positive by contract, which the verif/ SVA enforces.

  // ── P2: qs (S-4) = pure exponent rebias, no arithmetic ───────────────────
  // f32 biased exp = (e - 15) - OUT_FRAC + 127. Written in terms of
  // WALK_OUT_FRAC so a change to the ASU Q1.15 contract cannot silently
  // desynchronise this from the golden.
  function automatic logic [31:0] comp_qs(input logic [14:0] sv);
    logic [7:0] e32;
    begin
      e32     = {3'b0, sv[14:10]} + 8'(112 - WALK_OUT_FRAC);
      comp_qs = {1'b0, e32, sv[9:0], 13'b0};       // fp16-grade by structure
    end
  endfunction

  // ── 2026-08-14 QS EXPONENT SLEDGEHAMMER (E7 softmax-emission micro-flight,
  //    /private/tmp/apex-microflight/build/hw_verdict.json + hwcaps/) ────────
  // The 12-program micro-flight decoded silicon's walked S-4 transfer
  // NUMERICALLY: every solved PV weight on the settle image is EXACTLY
  //   c8'[t] = quant_rows_i8( p[t] * s_v * 2^-15 * w(t,T) ),
  // w uniquely pinned per row over 168/168 head-rows (hw s_c bits included):
  //   T=1: [1/2]   T=2: [1/4, 3/8]   T=3: [1/4, 1/4, 3/8]
  //   T=5: [1/16, 1/8, 1/8, 1/4, 1/4]
  // i.e. the qs word leaves this block with its EXPONENT offset by a
  // row-indexed quantity (pure 2^-k rows, plus exact significand x1.5 on the
  // 3/8 rows — an exponent-half artifact), while its mantissa placement is
  // otherwise intact. The cs words of the SAME flights are bit-exact on
  // silicon (fs 644/644), which exonerates everything cs and qs SHARE —
  // sc_mem, sk_q, res_q, comp_q, the tile transport muxes — and pins the
  // fault inside the qs-EXCLUSIVE cone: comp_qs's rebias adder + word
  // formation + the is_qs_q leg of the ST_RND result mux. No stored scale
  // cell can produce the measured exponents by mis-addressing (stored e=8 ->
  // e32=105, unwritten e=0 -> 97; measured 101..104), so the corruption is
  // ARITHMETIC: a D-033-class synthesis fold that let walker row-counter
  // bits (the only row-varying state; req_idx = t_rows + cw is an adjacent
  // 9-bit add in the requester) reach the 8-bit rebias adder through a
  // replicated exponent operand mux. Sim (Verilator) is green on identical
  // source — one RTL expression, two cones, one wrong (the D-033 lesson).
  //
  // THE FIX A KEEP CAN'T BE: the whole qs word is formed in its own
  // registered stage, from its own dont_touch OPERAND flop (qs_sv_q, loaded
  // on the same edge as sk_q) into its own dont_touch RESULT flop
  // (qs_word_q, loaded in ST_MUL — the value only needs sk_q, which settled
  // at request accept). The rebias adder now lives ALONE between two forced
  // flops: nothing to share with cs_word's exponent adder, nothing adjacent
  // to the requester's t_rows + cw add, and replication can only copy the
  // flops' D nets, never re-derive their function. ST_RND then selects
  // between two REGISTERS. Values and cycle timing are bit-identical in sim
  // (qs_sv_q == sk_q[14:0] by construction; res_q/res_valid unchanged);
  // the silicon A/B on the next image is the proof either way.
  (* dont_touch = "true" *) logic [14:0] qs_sv_q;    // qs operand flop
  (* dont_touch = "true" *) logic [31:0] qs_word_q;  // qs result flop

  // ── scale cache: 2T x 16b, K records at [0,T), V at [T,2T) ───────────────
  // Deliberately NOT reset (house rule: stored bytes are don't-care after
  // reset because every consumer re-loads first) — the VALID bitmap is.
  logic [15:0]              sc_mem [WALK_SC_ENT];
  // 2026-08-08 THE KILL-SHOT (TERM_READ_VERDICT.md): five silicon netlists
  // read sc_val[0]==0 after a proven committed write, rewrite-resistant,
  // sim-green — consistent with Vivado re-inferring this decoded-write /
  // indexed-read vector as distributed RAM with a broken address mapping
  // (writes land in other cells; INIT=0 cells read "stale"; only the
  // WALKED path ever reads it, so no host test could see it). These
  // attributes make RAM inference impossible: the vector MUST be flops.
  (* ram_style = "registers", dont_touch = "true" *)
  logic [WALK_SC_ENT-1:0]   sc_val;
  logic [15:0]              s_q_q;
  logic                     s_q_val;

  // ── snoop: amax reduce over a record's D fp16 beats, then the KVQ's own
  //    scale function. 15-bit masked magnitude max — fp16 magnitude ordering
  //    is the integer ordering of {exp,mant}, so a plain unsigned compare is
  //    the same reduce cq_value_path performs on the same beats. ────────────
  logic [15:0]        amax_r;
  logic [BEAT_W-1:0]  beat_c;
  logic [15:0]        snp_mag;
  logic [15:0]        amax_nxt;
  logic               beat_is_last;

  assign snp_mag      = {1'b0, snp_data[14:0]};
  assign amax_nxt     = (snp_mag > amax_r) ? snp_mag : amax_r;
  assign beat_is_last = (int'({{(32-BEAT_W){1'b0}}, beat_c}) == int'(CFG_D) - 1);

  // ── THE 2026-08-08 SILICON REWRITE (semantically identical, structurally
  //    defensive) ─────────────────────────────────────────────────────────
  // Four instrument flights (WALK_DBG v2/v3/v4, agfi-...e1d/...260) proved
  // every EXTERNAL observable of this block healthy on silicon while the
  // walked-attention stale persisted: snp beats counted (2), both frames
  // ending at beat D-1 (no err_frame), commits at addresses 0 and 1, on
  // ENGINE 0, map 0 — and the read of sc_val[0] on the same engine still
  // returned 0. Sim (Verilator) is green on the identical netlist source.
  // The remaining suspect class is SYNTHESIS of this block's structure:
  // an indexed single-bit NBA into a wide resettable vector SHARING a
  // branch with a RAM-inferred array write. The rewrite separates them:
  // the commit strobe is computed ONCE, the RAM write gets its own block,
  // and the valid vector becomes an explicit per-bit decoded register —
  // three forms Vivado cannot merge or mangle. Bit-for-bit the same
  // behavior in simulation (all suites re-run green); the silicon A/B on
  // the next image is the proof either way.
  wire sc_commit = snp_valid && !snp_flush && snp_last && beat_is_last;

  // ── 2026-08-14 A0 WRITE-CONE PIPELINE (CLOCK_LADDER.md §9 addendum 4
  //    lineage; the sledgehammer discipline, applied INSIDE the unit) ──────
  // A0 build 3 (in-bank ingress stage @14f16d1): WNS −11.51 with the WHOLE
  // top-path INSIDE the keep_hierarchy region — snp_data_q_reg (the bank's
  // ingress stage) -> this unit's sc_mem BRAM DIN, one 16 ns cycle carrying
  // the amax compare/mux PLUS the ENTIRE combinational scale_from_amax
  // (priority-encode binade search + 41-bit barrel + a 12-step restoring
  // divide of 54-bit compare-subtracts + RNE + fp16 pack) PLUS the BRAM
  // setup. ~27.5 ns of logic; dont_touch (rightly, D-033) forbids the tools
  // from restructuring it. So the RTL restructures it — the ONLY legal
  // mover of logic across a D-033 fence is this file.
  //
  // THE FORM (no new arithmetic exists here — the cq_scale_pipe.sv S10
  // precedent): scale_from_amax is ALREADY published by cq_fp_pkg as
  // registered-composable stage functions (scale_front / div_front /
  // div_step / div_back / scale_back), and the pipelined composition is
  // certified bit-exact against the combinational twin by the fparith
  // exhaustive proof + this unit's own comp sweeps. Three register cuts:
  //
  //   W1  scale_front + div_front of the COMMIT-CYCLE amax_nxt
  //       (cone: 15-bit amax compare/mux + f16 decode + binade search +
  //        barrel shift + the divider's saturation pre-compare ≈ 9-11 ns)
  //   W2  restoring-divide steps 11..5
  //       (cone: 7 x 54-bit compare-subtract ≈ 11 ns)
  //   W3  = THE REGISTERED WRITE PORT: steps 4..0 + div_back (the one RNE
  //       tie decision) + scale_back (renormalize + pack) into
  //       scwp_{we,addr,data}
  //       (cone: 5 x compare-subtract + increment + pack ≈ 11 ns)
  //   W4  edge: the BRAM write and the sc_val decode fire FROM the wp
  //       flops — flop -> DIN/ADDR/WE with zero logic, so the BRAM setup
  //       is paid on a register-to-pin hop and nothing else.
  //
  // D-033 DISCIPLINE, kept item by item: the commit strobe is computed
  // ONCE (sc_commit above) and then travels as a FLOP CHAIN — replication
  // can copy a flop (same D net) but can never re-derive its function; the
  // RAM write keeps its own block; the valid vector keeps its per-bit
  // decoded-register form (now decoded from the wp flops, the SAME cycle
  // as the RAM write, so sc_val can never lead sc_mem); every stage
  // register is dont_touch so no cross-stage merge/retime can reconstitute
  // the one-cycle cone this pipeline exists to kill.
  //
  // ALIGNMENT (the tile q/q2 + bank ingress argument, fourth application):
  // commits now land THREE cycles after the staged last beat instead of
  // ZERO. The read side is decode-coupled by construction — sc_val[i] sets
  // on the SAME edge sc_mem[i] is written, so a request in the shrinking
  // window before a commit lands is REFUSED LOUDLY (err_stale, D-006 no
  // state change), never answered from stale bits. And no legal requester
  // can hit that window: every composite request is issued by
  // seq_layer_walker's W_S_CSREQ/W_S_QSREQ states, which sit behind the
  // walk kick + W_CHECK(tile_idle) + W_S_FENCE(tile_idle) + W_S_SJOB, and
  // the kick itself sits behind the STOREKV step's idle-polled WRITE_ADDR
  // choreography and the entire QSTAGE step (PC_STORE=9 < PC_QSTAGE=10 <
  // PC_ATTN=11, seq_layer_walker2.sv) — the closest structural
  // commit->read distance is the V-record's own D-beat stream plus AXIL
  // polls, ≥ ~70 cycles at D=64, against a +3 shift. Host-mode requests
  // ride CSR round-trips and are slower still. snp_flush is deliberately
  // NOT wired into the pipe: a record that reached sc_commit COMPLETED its
  // frame — the same record the KVQ engine stored — and D-020 aborts
  // in-flight FRAMES, not landed stores (matching the old code, where the
  // write had already happened by the time any flush could arrive).
  // rst_n DOES clear the stage valids: a pre-reset commit must not land
  // after reset cleared sc_val (the old code had no such window; the pipe
  // must not invent one).
  cq_fp_pkg::scale_front_t sc_front;
  cq_fp_pkg::div_state_t   sc_div0;
  assign sc_front = cq_fp_pkg::scale_front(amax_nxt, 8);
  assign sc_div0  = cq_fp_pkg::div_front(sc_front.n, sc_front.d);

  // W1 registers (valid chain reset-cleared; data free-running, the bank
  // ingress idiom — association is carried by the valid, not the data)
  (* dont_touch = "true" *) logic                    scw1_v;
  (* dont_touch = "true" *) logic [WALK_SC_AW-1:0]   scw1_addr;
  (* dont_touch = "true" *) cq_fp_pkg::div_state_t   scw1_div;
  (* dont_touch = "true" *) logic signed [31:0]      scw1_e;
  (* dont_touch = "true" *) logic                    scw1_eps;
  always_ff @(posedge clk) begin
    if (!rst_n) scw1_v <= 1'b0;
    else        scw1_v <= sc_commit;
  end
  always_ff @(posedge clk) begin
    scw1_addr <= snp_addr;
    scw1_div  <= sc_div0;
    scw1_e    <= sc_front.e;
    scw1_eps  <= sc_front.eps;
  end

  // W1 -> W2 cone: restoring steps 11..5 (each step body IS
  // cq_fp_pkg::div_step at a constant index — no pipe-private arithmetic)
  cq_fp_pkg::div_state_t sc_div_hi;
  always_comb begin
    sc_div_hi = scw1_div;
    for (int i = 11; i >= 5; i--)
      sc_div_hi = cq_fp_pkg::div_step(sc_div_hi, i);
  end

  (* dont_touch = "true" *) logic                    scw2_v;
  (* dont_touch = "true" *) logic [WALK_SC_AW-1:0]   scw2_addr;
  (* dont_touch = "true" *) cq_fp_pkg::div_state_t   scw2_div;
  (* dont_touch = "true" *) logic signed [31:0]      scw2_e;
  (* dont_touch = "true" *) logic                    scw2_eps;
  always_ff @(posedge clk) begin
    if (!rst_n) scw2_v <= 1'b0;
    else        scw2_v <= scw1_v;
  end
  always_ff @(posedge clk) begin
    scw2_addr <= scw1_addr;
    scw2_div  <= sc_div_hi;
    scw2_e    <= scw1_e;
    scw2_eps  <= scw1_eps;
  end

  // W2 -> W3 cone: steps 4..0 + the RNE decision + fp16 pack
  cq_fp_pkg::div_state_t sc_div_lo;
  always_comb begin
    sc_div_lo = scw2_div;
    for (int i = 4; i >= 0; i--)
      sc_div_lo = cq_fp_pkg::div_step(sc_div_lo, i);
  end
  wire [12:0] sc_m     = cq_fp_pkg::div_back(sc_div_lo);
  wire [15:0] sc_scale = cq_fp_pkg::scale_back(sc_m, scw2_e, scw2_eps);

  // W3 = the registered write port: addr/data/we flopped immediately
  // before the BRAM, so the W4 write edge sees registers and nothing else
  (* dont_touch = "true" *) logic                    scwp_we;
  (* dont_touch = "true" *) logic [WALK_SC_AW-1:0]   scwp_addr;
  (* dont_touch = "true" *) logic [15:0]             scwp_data;
  always_ff @(posedge clk) begin
    if (!rst_n) scwp_we <= 1'b0;
    else        scwp_we <= scw2_v;
  end
  always_ff @(posedge clk) begin
    scwp_addr <= scw2_addr;
    scwp_data <= sc_scale;
  end

  always_ff @(posedge clk) begin
    if (scwp_we) sc_mem[scwp_addr] <= scwp_data;    // RAM: its own block
  end

  always_ff @(posedge clk) begin
    for (int unsigned i = 0; i < WALK_SC_ENT; i++) begin
      if (!rst_n)
        sc_val[i] <= 1'b0;
      else if (scwp_we && (scwp_addr == WALK_SC_AW'(i)))
        sc_val[i] <= 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      amax_r   <= '0;
      beat_c   <= '0;
      s_q_q    <= '0;
      s_q_val  <= 1'b0;
      err_frame <= 1'b0;
    end else begin
      err_frame <= 1'b0;                            // 1-cycle pulse

      if (sq_valid) begin
        s_q_q   <= sq_data;
        s_q_val <= 1'b1;
      end

      if (snp_flush) begin
        // D-020: the engine drops the in-flight token with no store, so the
        // walker must drop it too — never commit a partial record's amax.
        amax_r <= '0;
        beat_c <= '0;
      end else if (snp_valid) begin
        if (snp_last) begin
          // Frame contract: tlast must land on element D-1. A short/long
          // frame would silently mis-associate the record, so it raises the
          // error pulse and commits NOTHING (§3 discipline).
          if (!beat_is_last) err_frame <= 1'b1;
          amax_r <= '0;
          beat_c <= '0;
        end else begin
          amax_r <= amax_nxt;
          beat_c <= beat_c + BEAT_W'(1);
        end
      end
    end
  end

  // ── composite pipeline ───────────────────────────────────────────────────
  // Sequential rather than throughput-optimised on purpose: the phase needs T
  // composites against a GEMM that costs orders more cycles, so a 3-cycle
  // walk per word is free and keeps the D=128 multiply off the critical path.
  typedef enum logic [1:0] {
    ST_IDLE = 2'd0, ST_MUL = 2'd1, ST_RND = 2'd2, ST_RES = 2'd3
  } state_e;
  state_e state;

  logic [21:0] p_q;              // s_q * s_k significand product (exact)
  logic [7:0]  ebase_q;          // e1 + e2, widened
  logic [31:0] res_q;
  logic        is_qs_q;
  logic [15:0] sk_q;

  assign req_ready = (state == ST_IDLE);
  assign res_valid = (state == ST_RES);
  assign res_data  = res_q;

  // D=128 constant: the f64 significand of 64*sqrt(2) (P4 — an f32-rounded
  // constant is a different number and fails on 86,592 reachable products).
  localparam logic [52:0] CM_D128 = 53'h16A09E667F3BCC;

  logic [74:0] m_full, m_norm;
  logic [1:0]  m_sh;
  logic [23:0] sig24;
  logic        g_bit, s_bit, rnd_up;
  logic [24:0] sig_rnd;

  assign m_full = 75'(p_q) * 75'({22'b0, CM_D128});

  always_comb begin
    // normalize so the MSB sits at bit 74 (m_full is in [2^72, 2^75))
    if (m_full[74])      begin m_norm = m_full;        m_sh = 2'd0; end
    else if (m_full[73]) begin m_norm = m_full << 1;   m_sh = 2'd1; end
    else                 begin m_norm = m_full << 2;   m_sh = 2'd2; end
  end
  assign sig24   = m_norm[74:51];
  assign g_bit   = m_norm[50];
  assign s_bit   = |m_norm[49:0];
  assign rnd_up  = g_bit & (s_bit | sig24[0]);        // P6: RNE
  assign sig_rnd = {1'b0, sig24} + 25'(rnd_up);

  // ── the cs word, per build geometry (declared before use) ────────────────
  // P3 (D=64): exact — normalize the 22-bit product and add exponents.
  // P5 (D=128): one RNE24 of the exact 75-bit product == the golden's
  //             two-step f64->f32 narrowing on the whole reachable set.
  // Reads the combinational sig_rnd/m_sh above, which are settled by ST_RND
  // because p_q was registered in ST_MUL.
  function automatic logic [31:0] cs_word(input logic [21:0] p,
                                          input logic [7:0]  ebase);
    logic [7:0]  e32;
    logic [22:0] frac;
    begin
      if (CFG_D == 64) begin
        // P = 1.f x 2^(20+nrm) exactly; x 2^(SCORE_FRAC-3) is an exponent add.
        // biased = (20+nrm) + (e1-25) + (e2-25) + (SCORE_FRAC-3) + 127
        //        = ebase + nrm + 94 + SCORE_FRAC
        frac = p[21] ? {p[20:0], 2'b00} : {p[19:0], 3'b000};
        e32  = ebase + 8'(94 + WALK_SCORE_FRAC) + {7'b0, p[21]};
      end else begin
        frac = sig_rnd[24] ? sig_rnd[23:1] : sig_rnd[22:0];
        e32  = ebase + (8'd75 - {6'b0, m_sh}) + 8'd30 + {7'b0, sig_rnd[24]};
      end
      cs_word = {1'b0, e32, frac};
    end
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= ST_IDLE;
      p_q       <= '0;
      ebase_q   <= '0;
      res_q     <= '0;
      is_qs_q   <= 1'b0;
      sk_q      <= '0;
      qs_sv_q   <= '0;
      qs_word_q <= '0;
      err_stale <= 1'b0;
      dbg_stale_sc <= 1'b0;
      dbg_stale_sq <= 1'b0;
    end else begin
      err_stale    <= 1'b0;                           // 1-cycle pulse
      dbg_stale_sc <= 1'b0;
      dbg_stale_sq <= 1'b0;

      unique case (state)
        ST_IDLE: if (req_valid) begin
          is_qs_q <= req_is_qs;
          sk_q    <= sc_mem[req_idx];
          // 2026-08-14 qs sledgehammer: dedicated operand flop, same edge,
          // same RAM read — bit-identical to sk_q[14:0] by construction.
          qs_sv_q <= sc_mem[req_idx][14:0];
          // A request for an entry the snoop never wrote (or before s_q
          // landed) would compose from stale bits — refuse loudly instead.
          if (!sc_val[req_idx] || (!req_is_qs && !s_q_val)) begin
            err_stale    <= 1'b1;
            dbg_stale_sc <= !sc_val[req_idx];
            dbg_stale_sq <= (!req_is_qs && !s_q_val);
            state        <= ST_IDLE;
          end else begin
            state <= ST_MUL;
          end
        end

        ST_MUL: begin
          // qs needs no product; cs forms the exact 11x11 significand product
          p_q     <= {1'b1, s_q_q[9:0]} * {1'b1, sk_q[9:0]};
          ebase_q <= {3'b0, s_q_q[14:10]} + {3'b0, sk_q[14:10]};
          // 2026-08-14 qs sledgehammer: the whole qs word (rebias adder +
          // mantissa placement) forms HERE, alone between two dont_touch
          // flops — the fold target of the micro-flight verdict is gone.
          qs_word_q <= comp_qs(qs_sv_q);
          state   <= ST_RND;
        end

        ST_RND: begin
          state <= ST_RES;
          if (is_qs_q) begin
            res_q <= qs_word_q;              // a REGISTER, not a shared cone
          end else begin
            res_q <= cs_word(p_q, ebase_q);
          end
        end

        ST_RES: if (res_ready) state <= ST_IDLE;

        default: state <= ST_IDLE;
      endcase
    end
  end

  // The snoop's sign bit is deliberately discarded: the amax reduce is over
  // MAGNITUDES (cq_value_path masks the same bit on the same beats), so
  // snp_data[15] carries no information this block may use.
  // Sign bits are never consumed: the amax reduce is over MAGNITUDES (the
  // same bit cq_value_path masks on the same beats), and every cached scale is
  // positive by contract (P1), which the verif/ SVA enforces rather than
  // assumes.
  logic unused_ok;
  assign unused_ok = &{1'b0, snp_data[15], s_q_q[15], sk_q[15]};

endmodule

`endif // SEQ_WALKER_COMP_SV
