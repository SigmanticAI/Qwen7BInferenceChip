# F2 stage-2 sim executor — status

> ## ✅ RESOLVED 2026-07-20 (commit 811aa11) — all 18 jobs pass
> `F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS`. Evidence:
> `docs/results/f2_stage2_sim/`. Root cause was `apex_f2_mailbox.sv:200`
> mis-ordering `lane8_beat_t` fields on the **external weight stream** (cast
> instead of re-order → every weight beat shifted right one bit), plus a
> `cl_apex` tile config that did not match the verified L3 reference build
> (`KVQ_DEPTH` 128 vs 256 stalled every T>64 job on the KVQ `OCC` poll).
>
> **Everything below this line is the historical debug trail and its
> conclusions are SUPERSEDED.** In particular the "prime remaining
> hypothesis" (decoupled per-stream FIFOs breaking an implicit ordering) was
> wrong — the mailbox never dropped or re-ordered anything, now proven by the
> `MB_STATUS` drop audit reading 0 across 1.86M register ops. Kept for the
> record of what was ruled out and how.

## (historical, 2026-07-18)

**Milestone this session: the CONTROL PATH runs end to end on the simulated
FPGA.** job_s019 executes all 27,600 register ops to completion (225k
cycles): phase-A CSR sanity, tier-bank init, KV store, **descriptor
dispatch**, score phase, P·V phase — every poll succeeds, the tile sequences
exactly as the L3 choreography drives it.

**Root-cause bug FIXED — the mailbox was held in reset.** A signal rename
(`rstn` → `rst_main_n_sync`) updated the tile and FSM but missed the mailbox
instantiation (`.rst_n(rstn & ...)`); with `default_nettype wire`, `rstn`
became an undriven implicit wire = 0, so the mailbox's clocked write logic
never ran while its combinational read mux still returned values — reads
worked, writes silently dropped. One-word fix (line 303). Verified: mailbox
ROUTE0 read/write round-trips, stream pushes drain into the tile,
descriptors now dispatch (the SEQ→MXE→desc_error chain the earlier probe
proved was reachable).

**Open (narrowed): output DATA mismatch.** 506 checks, 36 pass (all the
control/CSR checks), 470 fail — every output-capture value (EFS feeder
scales, ESS softmax scales, ERO result beats). The tile produces plausible
fp16 values that don't match the golden `attention_core` result the regops
carry. Control is right; the fed DATA or the capture ordering is off.
Next: diff my input-stream translation (XR/GR/WB/QS/CS + the KVQ store
path) against what the L3 TB's `drive_*` tasks put on the same ports — most
likely a byte/lane ordering or a stream the compiler doesn't feed. The
executor + VCD trace (`make TRACE=1`) make this directly inspectable.

**Honesty:** no hardware claim until sim passes all 18 jobs AND the F2
session logs match. Stage-2 RTL builds clean on VU47P (250 MHz) — a build
result, not yet an execution result.

## Phase-F debug — suspects CLEARED this pass (don't re-tread)

Symptom sharpened: golden feeder scales are a TIGHT cluster ~0x35xx (≈0.35,
what per-row INT8 scales look like on normalized data); the tile's are LARGE
and SCATTERED (0x3b–0x43, ≈0.9–3.9). So the tile computes on wrong-magnitude
K̂ — an input-reconstruction/framing divergence, not capture and not control
(all 36 control/CSR checks pass; every poll succeeds; cross-check
`r.o8 == trace.o8` confirms the golden itself is correct).

Ruled out by direct test/inspection:
- **KVQ_G**: built the tile at both 16 and 128 → 470 vs 471 fails (the +1 is
  only the INFO_GROUP CSR check). CQ-8 keys are ungrouped (`gen_l3_vectors`
  §515 `key_grouped = tier != TIER_CQ8`), so G doesn't touch this job's data.
  Kept at 16 (the L3 reference build, passes INFO_GROUP).
- **WB translation**: `wbeats` emits one 64-bit lane8 beat per WB
  (`{b:016x}`); the Xlate WB splits XW0/XW1 and pushes one beat — faithful.
- **AJ/WJ NBW packing**: Xlate.NBW = clog2(D/8)+1 = 5 at D=128, matches the
  tile's STAGE_NB_W; sel/nb land at the right bits.

Prime remaining hypothesis: the mailbox's DECOUPLED per-stream FIFOs break an
implicit ordering the direct L3 TB gets for free. inject_jobs emits, per
weight-row j: AJ (act loader beat) → DESC (K=2 GEMM acc=127·w0+w1) → 8×WB
(weights). The native TB's `drive_*` tasks handshake in program order, so the
weights for descriptor N are loaded before N dispatches. Through the mailbox,
ds/xw/xa are independent FIFOs the tile drains at its own pace — if a
descriptor dispatches before its WB weights land, the GEMM reconstructs K on
stale weights → wrong K̂ → the scattered feeder scales. NEXT: waveform the
weight-stage load vs ds dispatch for the first inject GEMM (TRACE=1), and if
confirmed, add per-inject ordering (poll wj/aj-consumed before DESC push, or
gate DESC on weight-stage-ready) in the mailbox host sequence.
