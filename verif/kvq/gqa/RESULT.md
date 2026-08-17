# verif/kvq/gqa — IB-LAYER S4b per-KV-head GQA bank suite (RESULT)

**DUT:** `rtl/top/glue/apex_kvq_gqa_bank.sv` — N_ENG per-KV-head CQ-8
`kvq_engine` instances behind the live `eng_sel` mux (the D-024 banking
pattern; LEVEL_C_INTEGRATION.md §9.1 R3-AMENDED / IB_LAYER.md §0.1 approved
S4b plan). The `h // (H/H_kv)` query→KV-head mapping is SEQUENCER-side by
contract — the bank consumes a ready engine index and is verified as a
ROUTER; per-engine numerics are the L3/f2-verified engine's own.

**Status: ALL GATES GREEN (2026-07-26).** `make all` rc=0:

```
LINT CLEAN (bank -Wall; waivers scoped to vendored engine internals only)
LINT CLEAN (apex_top @ KVQ_GQA_NENG=4/KVQ_DEPTH=256 — parameter-ON elaboration)
GQA VECTORS: 22 records over 4 engines (+1 rewrite row), golden = cq_codec
  compress/decompress_values (CQ-8), aliasing self-checks PASS
GQA BANK gqa4: checks=1859 fails=0
GQA BANK gqa4: ALL PASS
mS1: CAUGHT (hang; signature ['phase=info e1'])
mS2: CAUGHT (value; signature ['hat e'])
mS3: CAUGHT (sva; signature ['[SVA §5] s_axis'])
GQA SUITE: ALL PASS (gqa4 1859 checks, 3/3 signature mutants caught)
```

Check-count pin 1859 was DERIVED from the TB structure first (Makefile
header) and confirmed identical on the first run. Golden arbiter:
`golden/apex_golden/cq_codec.py` value-record primitives, imported never
re-derived; the §5 stream SVA pack (`verif/kvq/smoke/kvq_axis_sva.sv`,
`bind kvq_engine`) rides into all four engine instances and the run log is
grep-gated for `[SVA`. Mutants are patched COPIES of the bank (l3/mutate.py
signature discipline — a mutant failing WITHOUT its signature fails the
gate; mS3's measured catch is the per-engine SVA at the mis-routed engine,
with the B1c per-engine OCCUPANCY checks as the value-class backstop).

## ×H_kv synth probe (§1b flow — EXECUTED 2026-07-26)

House sv2v→yosys flow (sv2v 0.0.13, yosys 0.66 — the §1b toolchain),
generated wrapper `gqa_probe_n{1,4}` binding
`apex_kvq_gqa_bank #(N_ENG, CFG_D=128, KVQ_G=16, KVQ_DEPTH=256, KVQ_SETS=8)`
(the I-B GQA geometry: 7B head_dim, the R3 2T<=256 depth, tile SETS
derivation at G=16), identical commands per target
(`sv2v --define=SYNTHESIS … > flat.v` then `yosys -p "read_verilog flat.v;
hierarchy -top gqa_probe_nN; synth_xilinx -family xcup -abc9; stat"`; ECP5
via `synth_ecp5 -abc9`). CHECK: "Found and reported 0 problems" all four.
Verbatim mapping lines (the record SRAM infers block RAM as-is, per engine):

```
mapping memory ...sram_controller.mem via $__XILINX_BLOCKRAM_SDP_   (xcup, n1 and n4)
mapping memory gqa_probe_n4.u_bank.g_eng[0..3].u_eng.u_sram.mem via $__PDPW16KD_   (ecp5 n4: 4 lines, one per engine)
mapping memory gqa_probe_n1.u_bank.g_eng[0].u_eng.u_sram.mem via $__PDPW16KD_      (ecp5 n1)
```

| US+ (`synth_xilinx -family xcup -abc9`) | N_ENG=1 | N_ENG=4 | Δ = measured ×H_kv cost |
|---|---|---|---|
| RAMB18E2 | 31 | **124** | +93 (exactly 4×31 — the select fabric adds no BRAM) |
| DSP48E2 | 1 | **4** | +3 |
| LUT2-6 total | 10,211 | **41,147** | +30,936 (4.03× — ~303 LUT select fabric) |
| FF total (FDCE/FDPE; engines are async-reset) | 9,150 | **36,600** | +27,450 (exactly 4×) |
| CARRY4 | 461 | **1,844** | +1,383 (exactly 4×) |

| ECP5 (`synth_ecp5 -abc9`) | N_ENG=1 | N_ENG=4 | Δ |
|---|---|---|---|
| DP16KD / MULT18X18D | 30 / 1 | **120 / 4** | +90 / +3 (4×) |
| LUT4 / TRELLIS_FF | 7,825 / 9,222 | **31,141 / 36,846** | +23,316 / +27,624 (≈4×) |

**Verdict:** the per-KV-head bank costs, MEASURED, almost exactly N_ENG ×
the single CQ-8 engine at the same geometry — the eng_sel fan-out/mux
fabric is ~0.3k LUT and zero BRAM/DSP. At the I-B target (VU47P
Small-Shell CL) 124 RAMB18E2 + 41k LUT is small against the part,
consistent with the §0.1 R3 assessment ("4 CQ-8 engines at DEPTH=256 are
small against that part") — now measured, not argued.

**Scope caveats (mirroring §1b verbatim):** yosys mapping evidence only —
**no P&R, no timing** (timing comes only from the I-B Vivado build report,
integration-owned); the post-synth netlists were not re-simulated. Absolute
LUT counts are abc9-flow-specific.

## Repro

```
make -C verif/kvq/gqa all        # lint + ON-elab lint + vectors + run + 3 mutants + gate
verif/kvq/gqa/synth_probe.sh     # the ×H_kv probe (4 syntheses + cell tables)
```
