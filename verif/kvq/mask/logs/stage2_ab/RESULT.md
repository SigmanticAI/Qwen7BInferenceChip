# S12 stage 2 — existing-KVQ-matrix compat proof (A/B, same box)

**Verdict: PASS — all 79 count/gate lines byte-identical.**

- A (base) = `s12-base-repro` @ 640fd5f = babaf00 + the two repro fixes
  (a50b1ad path portability, 981fb31 sed portability) — ZERO RTL delta.
- B (branch) = `comp/s12-mask` @ b0f15d5 (stage-1 D-027 engine RTL on top).
- Same box (apex-s12-verify, c6a.xlarge), same toolchain: Verilator 5.044
  built from the v5.044 tag (= the pinned local version).
- Legs: verif/kvq smoke (7 parity configs + stall + 4 regressions, lint
  -Wall) · sb (9 runs, coverage, 5 mutants CAUGHT) · audit (reset storm,
  -0.0 directed, coverage, b1/b2/b4 fix-reverts CAUGHT) · fparith
  prove+gen+run+mut (8,748,634 checks / 0 fails, 2 mutants caught).
- Extraction: `base_counts.txt` / `branch_counts.txt` (grep of CONFIG/GATE/
  MUTATION/COVERAGE lines, whitespace-normalized); `diff <(sort A) <(sort B)`
  = empty (raw diff differs only by the base side's two-file concat order).
- fparith sv2v synth leg NOT run on the box (no sv2v there); it is cores-only
  but its SYN_SRCS includes kvq_engine.sv → re-run locally at stage 5
  alongside the local-pinned confirmation pass.
- Zero test edits in the stage-2 legs. The two repro fixes are test-INFRA
  portability fixes present identically on BOTH sides of the A/B
  (Mac-absolute REPO paths; BSD `sed -i ''` whose Linux misparse produced
  five bogus mutant escapes on the UNTOUCHED base tree — first fresh-clone
  run of that leg ever).

Raw logs: base_640fd5f_smoke_sb_audit.log (SSH-cut during fparith; those
legs' gates all landed before the cut) + base_640fd5f_fparith.log (nohup
completion) + branch_b0f15d5_full.log (single chain).
