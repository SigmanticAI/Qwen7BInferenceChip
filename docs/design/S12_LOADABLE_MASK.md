# S12 — LOADABLE-MASK design (proposed decision D-027)

**Status:** design + golden gates landed; RTL/TB not started (serialized
behind the launch-critical machine queue).
**Motivation:** KVQ4-alone is quality-fragile (D-022: 25.7% vs 4.8% e2e error
on outlier tensors; the paired 7B eval resolves KVQ4 as a likely-real −0.017,
`docs/results/s5_eval7b/RESULTS.md`). The mitigation tier KVQ4+ needs an
outlier mask, but the mask is a synthesis-time ROM (`MASK_FILE` →
`$readmemh`), so the D=128 build ships maskless and KVQ4+ does not exist at
D=128 at the tile level (F-2 residual). S12 makes the mask a CSR-programmable
runtime input.

## 1. What stays structural (unchanged)

- **`OUTLIER_K` remains a synthesis parameter.** The key-record width
  elaborates from it (`KEY_LANE_BITS = OUTLIER_K × 16`, D-026 layout), so the
  lane *budget* is baked; S12 makes *which channels* runtime-selectable, never
  *how many*. This mirrors D-024's principle (tier is structural; runtime
  control is by selection among verified structures).
- **D-026 record/bank layout unchanged.** No new record fields; no
  mask-generation tags.

## 2. The load-bearing hazard (why the contract is shaped this way)

D-026 key records do **not** self-describe their mask:

- the sentinel nibble `4'd1` in an outlier channel's code slot is a
  placeholder, not a discriminator — a kept channel can legitimately quantize
  to `0x1`;
- lane order is the rank of the channel in the ascending mask set;
- bank rows force outlier-channel scales to `0x0000`.

So the read path must run under the **same mask that encoded the records**.
This is proven executable in `golden/tests/test_mask_semantics.py` §C/§D: an
independent record parser is bit-exact under the encoding mask and provably
mismatches on every disagreeing channel under a different commit-legal mask.

## 3. D-027 contract (proposed)

1. **Mask registers** in the engine AXI-Lite window (8-bit address space,
   free from `0x4C`):
   - `0x50 MASK0` … `0x5C MASK3`: 32 mask bits each, channel c = bit c of
     word c/32. D=64 uses MASK0/1; D=128 uses all four. Read-back returns the
     *staged* value.
   - `0x60 MASK_CTRL`: write 1 to COMMIT the staged mask; read returns
     `{mask_valid, …}`.
2. **Commit legality** (golden §A): a commit takes effect iff
   `popcount(staged) == OUTLIER_K`. An illegal commit leaves the live mask
   unchanged and raises a sticky `MASK_ERR` (next free `IRQ_STATUS` bit,
   W1C, same discipline as `SB_OVWR`).
3. **Commit-at-empty** (golden §B/§D): a commit while the record store is
   logically occupied (occupancy > 0 or an open partial key group) raises
   sticky `MASK_SWAP` and — same philosophy as `SB_OVWR` — the hardware does
   not attempt to police readback correctness beyond the flag; decoding
   old-mask records under a new mask is a host-contract violation the flag
   makes auditable. Scale-bank ssid sequencing continues across a legal swap
   (bank rows persist; records do not — golden §B pins the composition).
4. **Reset semantics:** the live mask is **not** on `dp_clear` — it survives
   D-020 soft reset exactly like `scale_bank_store` (records persist, so
   their decode key must persist; the audit reset-attack bucket extends to
   the mask). Hard reset reloads the build default: `MASK_FILE` contents if
   given, else zeros with `mask_valid = 0`.
5. **`INFO_TIER` truth update:** CQ-4+ availability becomes
   `(OUTLIER_K > 0) && mask_valid` — the b128 build (OUTLIER_K=2, no ROM)
   reports `0x3` out of reset and `0x7` after a valid commit, keeping the
   D-024 "INFO_TIER never lies" rule.
6. **Backward compatibility:** builds with `MASK_FILE` behave identically
   out of hard reset (default-loaded, `mask_valid = 1`) — the existing
   matrix must pass **byte-identical** with zero test edits before any new
   TB is added (the S10a/S10b regression discipline).

## 4. Landing plan (staged, machine-aware)

| stage | what | machine need |
|---|---|---|
| ✅ 0 | golden gates (`make -C golden test` target `masksem`) + this doc | none |
| ✅ 1 | engine RTL: staged/live mask regs + commit FSM + faults + INFO_TIER (02f8c6d) | edit only |
| ✅ 2 | existing full KVQ matrix re-pass — same-box A/B, 79/79 count+gate lines byte-identical, zero test edits (febe4ec; logs in `verif/kvq/mask/logs/stage2_ab/`) | Verilator (c6a box) |
| ✅ 3 | new TB `verif/kvq/mask`: rom 799 + csr 1884 checks / 0 fails, 2 mutants caught (72c251c; re-pinned 800/1887 at stage 4 for the port checks) | Verilator (c6a box) |
| ✅ 4 | tile plumb + b128 L3: `adv_outlier_d128_cq4p` PASS (17,642 checks / 0 errors, D=128 CQ-4+ through the CSR-loaded mask, tile INFO_TIER 0x3→0x7 live) — **F-2 residual RETIRED** (d4161f7 + ebbc699; walker split 24/28+4 refused, no checks lost) | Verilator (c6a box) |
| 5 | D-027 entry in ARCHITECTURE.md §9 + STATUS rows + task-board close — see §6 checklist | none |

Stages 2–4 ran 2026-07-22/23 on the lane's own verify box
(`APEX_BOX_TAG=apex-s12-verify`, Verilator v5.044 built from source = the
pinned toolchain; the Mac stayed on the eval queue throughout).

## 6. Stage-5 close-out checklist (for the next session / integration)

1. **ARCHITECTURE.md §9**: land D-027 as decided (contract = §3 + the §5
   refinements). ARCHITECTURE.md is shared — coordinate via HANDOFF.
2. **STATUS/gen_status rows**: register `verif/kvq/mask` (rom+csr+mutants)
   and the L3 case delta (27→28 cases; walker split 24+4). The masksem
   golden target already rides `make -C golden test`.
3. **Local-pinned confirmation pass** (box == local at v5.044, so this is
   belt-and-braces per the box-script doctrine): re-run `verif/kvq` matrix +
   `verif/kvq/mask` + `verif/top/l3` on the Mac when the eval queue drains.
4. **sv2v legs**: `verif/kvq/fparith synth` (needs sv2v; SYN_SRCS includes
   kvq_engine.sv) + an engine-level sv2v+yosys probe of the D-027 code
   (HAS_MASK_FILE generate, mask CSR block) before any ECP5/F2 rebuild.
5. **Upstream the box-script fixes** to `comp/level-c-integration`:
   `APEX_BOX_TAG` per-lane boxes + the detach-before-fetch push fix.
6. **Task board**: S12 stages 1–4 DONE on `comp/s12-mask` (worktree
   `../apex-s12`); B2 eviction coordination unchanged (right-of-way was
   honored — `kvq_engine.sv` evict stub untouched).
7. Spawned chip (separate session): Mac-absolute REPO paths in 11 remaining
   verif/golden scripts outside this lane's dirs.

## 5. Stage-1 implementation notes (D-027 refinements, pinned before RTL ran)

Contract refinements made while implementing §3 — none alter the golden
gates; each is the resolution of a point the contract left open, chosen for
truth-preservation and verifiability:

1. **`mask_valid` is COMPUTED, never stored:** `popcount(live mask) ==
   OUTLIER_K`, evaluated on the live bus. A `MASK_FILE` build reads valid out
   of hard reset; the maskless `OUTLIER_K>0` build (b128 ship shape) reads
   invalid until the first commit; **a malformed ROM (popcount ≠ K) truthfully
   reads invalid** — "INFO_TIER never lies" now extends to bad build inputs.
2. **Live-mask source is a select, not a reload:** `outlier_mask_bus =
   (OUTLIER_K>0 && mask_csr_owned) ? mask_live : mask_build_rom`. Until the
   first successful commit the bus IS the unchanged build-ROM wire, so §3.6
   backward compatibility is structural; there is no non-constant async-reset
   value to break synthesis, and `k=0` datapaths keep constant-folding (the
   commit path can only ever write an all-zero mask there).
3. **`OUTLIER_K=0` builds keep 0x50–0x60 reserved** (reads `0xDEADBEEF`,
   writes ignored): masks are meaningless without lanes, and live regs would
   be dead flops in two of the three bank engines of every ship config.
4. **Fault split:** `MASK_ERR` = a REJECTED commit (illegal popcount; live
   mask unchanged). `MASK_SWAP` = an EFFECTIVE commit while occupied. An
   illegal commit while occupied raises `MASK_ERR` only — nothing swapped.
5. **"Open partial key group" precisely:** the engine FSM in any grouped-key
   phase (`ST_KCOLLECT/ST_KFEED/ST_KACCEPT/ST_KEMIT`) — from first accepted
   beat to last emitted record, covering the `ST_KEMIT` scatter window where
   `grp_tok_cnt` is already 0 but records are still being written under the
   old mask. Occupancy > 0 covers every read-burst window (a burst needs a
   written record). Like `SB_OVWR` at allocator wrap, `MASK_SWAP` fires at
   every legal §B re-encode boundary too (occupancy is monotone until hard
   reset); the host W1Cs it after confirming it re-encodes before reading.
6. **`MASK_CTRL` readback:** bit0 = `mask_valid`, bit1 = `mask_csr_owned`
   (live source is the CSR commit, not the ROM), rest 0.
7. **RAZ/WI beyond D:** staged-mask bits ≥ VECTOR_DIM read 0 and ignore
   writes (out-of-range channels cannot be staged); elaboration guard rejects
   `OUTLIER_K>0` with `VECTOR_DIM>128` (the window addresses 128 channels).
8. **Address-numeral collision disclosure:** MASK3/MASK_CTRL share numerals
   0x5C/0x60 with the tile-CSR WALK window (D-028) — physically separate
   buses (engine AXI-Lite behind `apex_kvq_bank` vs tile `csr_regs`); no
   conflict (B1's §8 audit row 11 confirmed the same).
9. **Stage-4 port plan (disclosed deviation):** the tile `INFO_TIER` bit-2
   truth needs a live `mask_valid` engine output. Existing engine-level TBs
   name every port under `-Wall`, so the port lands at stage 4 ONLY, with
   mechanical `.mask_valid()` sink hookups in the existing TBs — the sole
   test edit of the lane, disclosed in that commit, with the full matrix
   re-passed byte-identical after it. Stages 1–3 add no ports.
