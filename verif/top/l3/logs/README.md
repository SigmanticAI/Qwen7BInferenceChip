# verif/top/l3/logs — durable fail-first evidence for the F-1/F-5 closure
(2026-07-09; `make clean` wipes build/, so the reproduce-then-fix logs live here)

- `prefix_f5a_keptfail_before_fix.log` — the F-5a wedge on the PRE-FIX tile
  (git ff8b467 apex_top/apex_stage_buf + its own TB/case, rebuilt in a detached
  worktree): `drive_g stall @400911` on case bug_d128_stagebuf_nb (D=128, 4-bit
  nb fields truncate BPR=16 to 0).
- `prefix_f1_keptfail_before_fix.log` — the F-1 cap demonstrated on the FIXED
  RTL with T_ROW_MAX patched back to 64 (patched copy, rtl/ untouched):
  case calib_d64_T128 dies `drive_cs stall @928783` (T=128 dequant job rejected).
- `run_unit_stagebuf_mutant.log` — tb_stagebuf_patd vs a copy of the FIXED
  apex_stage_buf with the PAT_D index reverted to sel_q[2:0]: fails exactly at
  `[PAT_D sel=8 beat=0]` (block 8 aliased block 0) — the directed unit test
  discriminates the F-5b aliasing.

The kept-PASSING versions run in every `make all` (case bug_d128_stagebuf_nb,
the `unit` Makefile stage, and the full-length named cases).

## F-2 / D-022-actuation / F-3 closure pass (Task B1, 2026-07-09)

- `prefix_f2_keptfail_before_fix.log` (+ `probe_f2_tierctrl.txt`) — the F-2 /
  D-022 gap on the PRE-FIX tile (working tree before the D-024 tier bank, l3
  obj_b64 build of the F-1/F-5-closed RTL): TIER_CTRL=CQ4 written, then the
  KVQ engine's own INFO_TIER polled expecting 1 — reads 0 forever (tier select
  unwired, single TIER=0 engine): `KVP 0c timeout (got 00000000)`.
- `prefix_f3_keptfail_before_fix.log` (+ `probe_f3_sticky.txt`) — F-3 on the
  same pre-fix tile: an illegal descriptor sets err_sticky[0]; the ERR_STICKY
  W1C write (0x58) is a reserved no-op, so `[ESTK] got 00000001 exp 00000000`
  and the 0x58 readback returns DEADBEEF.

Kept-PASSING flips in every `make all`: per-engine INFO_TIER routing checks in
EVERY case's phase A, cases adv_T1_cq4p / adv_outlier1000_cq4p / tip_auto_mixed,
and the set->W1C->verify ERR_STICKY tail in every case's final phase.
