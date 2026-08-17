# KVQ fp-arith — synthesizable integer rewrite of cq_fp_pkg (proof suite)

**What changed:** `rtl/kvq/cores/cq_fp_pkg.sv` previously computed
`scale_from_amax` / `quant_one` / `dequant_one` in `real` (IEEE-754 double)
arithmetic — bit-exact in simulation but unsynthesizable (yosys: syntax error
on `TOK_REAL`). It is now pure INTEGER hardware: exact significand/exponent
decode, exact integer divide with round-half-to-even from the true remainder,
and exact integer multiply + normalize for dequant. Same function signatures,
still single-cycle combinational — `cq_scale_unit` / `cq_quant_unit` /
`cq_dequant_unit` and the `cq_value_path` / `cq_key_path` FSMs are unchanged
(the units' wrappers only had comment updates). The f16→f32 outlier widen is
`dequant_one(code=+1, raw)`, exact by construction (≤20-bit product in a
24-bit fp32 significand ⇒ zero rounding ⇒ no double-rounding possible).

**Critical path (documented per plan):** quant_one's exact unsigned divide
(≤40-bit dividend / ≤40-bit divisor after the binade shift) is the deepest
cone. Latency is non-critical in KVQ (serialized per-channel walk); if timing
closure ever demands it, the divide is the seam to pull into a bounded
bit-serial handshake unit. Yosys `stat` on the flattened engine shows the
`$div` cells synthesize structurally.

**Gate (`make all`, exit 0):**

1. `make prove` — `int_model.py` (op-for-op Python mirror of the SV) vs the
   golden arbiter `golden/apex_golden/cq_codec.py`, EXHAUSTIVE:
   scale over all 63 488 finite fp16 amax (incl. defensive negatives) × qmax
   {7,127}; quant over all finite fp16 x × every distinct golden-producible
   scale per qmax (24 833 / 23 385, incl. boundary/EPS/subnormal/negative
   extras); dequant over codes × the same scale sets; widen exhaustive.
   → ALL PASS.
2. `make gen` + `make run` — `tb_fparith.sv` proves the SHIPPING SV (package
   functions AND the unit instances) against GOLDEN-derived vectors (nothing
   trusts int_model): scale + widen fully exhaustive with explicit expected
   words at both fn and unit level; dequant exhaustive over ALL finite fp16
   scale patterns (incl. ±0/subnormal/negative) × codes −256..255 via
   order-sensitive 64-bit checksums; quant exhaustive over all finite x ×
   the dense scale sets via per-scale checksums (3.06e9 golden codes pinned),
   with strided + directed-boundary unit-instance re-checks.
   → `FPARITH TB: ALL PASS (checks=8748634 fails=0)`.
3. `make mut` — 2 seeded-error checks, both MUST be caught and were:
   * `mut_tie` (RNE tie → half-up): CAUGHT by the quant checksums. (Scale is
     tie-free by construction — odd divisors 7/127 admit no exact-half
     remainder — so the tie mutation is only observable in quant; that it IS
     caught also demonstrates the checksums detect single-word deviations.)
   * `mut_eps` (EPS=2⁻¹⁴ floor removed): CAUGHT immediately by the explicit
     scale sweep (`amax=0x0001 exp=0400 got=5092`).
4. `make synth` — sv2v (`--define=SYNTHESIS`) + yosys
   `hierarchy -top kvq_engine; proc; opt; stat` completes; the flattened
   netlist greps CLEAN for `real` / `$bitstoreal` / `$realtobits` / `$itor` /
   `$rtoi` / `shortreal`. The only sim-only construct in the synthesis
   filelist (kvq_engine's SRAM_DEPTH elab guard) is behind
   `ifndef SYNTHESIS`.

**Out-of-contract notes (honest exclusions):** fp16 inf/nan (exp==31) inputs
are excluded from all sweeps — the codec never produces them and the golden's
own float→int cast on them is undefined; the RTL decodes them defensively as
65504. ±0 scales are excluded from the QUANT sweep only (golden x/0 is
undefined); they are fully swept in dequant.

**Verilator 5.044 footgun (documented in the TB):** `continue` inside a
for-loop whose body also contains a `#delay` is miscompiled (statement updates
silently lost — reproduced standalone). The TB uses if-guards, never
`continue`, in timed loops.

**System gates after the swap (numerics must not move by one bit):**
`make -C golden test`, `verif/kvq/smoke` (exact recorded check counts),
`verif/kvq/sb`, `verif/top/smoke` (both D), `verif/top/l2`, `verif/top/l3`
— all green from clean; see the respective logs.
