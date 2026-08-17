# PROMPT-ON-CHIP — the contract for "a user types a prompt and our chip computes part of the answer"

**Owner objective, confirmed 2026-07-28.** This supersedes tranche I-C's
ordering as the top-level goal. Everything below is prioritised against one
question: *does it move a real prompt closer to being partly computed by our
hardware?*

---

## 0. The finding that reorders everything

**The chip has never computed a value the host did not already know.**

- The regop vocabulary is `note / w / pw / jf / poll / r / rn`. The `r` op
  reads a BAR0 address, compares it against a **baked-in** expected value,
  and **discards the read** (`f2_host_run.py`, `verif/f2sim/sim_main.cpp`).
  There is no operation that returns data, so no number the tile produced has
  ever reached a host variable.
- The program driving the tile is compiled **from golden's answer**:
  `gen_layer_trace.py`'s `build_case(..., inject=(w, X, fx))` derives the
  requant pairs, JOBC composites and CQ-8 store-scale assertions from the
  golden `LayerFx`.

So the pipeline has always been: **run golden → compile a program from
golden's answer → run tile → confirm agreement.** Every number this project
has published — 27,996 checks on silicon, 22,674 multi-head checks, 8.7M KVQ
checks — is a statement that the tile **AGREED**, never that the tile
**PRODUCED**.

That is not a criticism of the evidence: agreement against an executable
golden arbiter is exactly what verification should prove, and it is why the
RTL is trustworthy. But it is the honest distance between what exists and
"a prompt ran on our chip", and no amount of additional verification closes
it. **A value has to come back.**

## 1. Milestones

| # | milestone | what it proves | gate |
|---|---|---|---|
| **A** | **Read-back** — a regop that RETURNS a value; both executors in lockstep | the tile can produce a number the host reads, rather than only confirm one | in **f2sim**: capture at every existing `r` site and assert `cap(x) == baked e(x)` across all 27,996; 18-job regression + mutants unchanged |
| **B** | **Compute mode** — drive a job whose expectations are NOT derived from golden for that input; golden is consulted only to CHECK afterwards | the tile computes, then we grade it — not the reverse | a job where the host supplies only inputs + config; outputs come back via A and are compared to golden **after** the run |
| **C** | **Prompt seam** — `run_tinynpu.py` offloads one real operation of a real Qwen2.5-7B decode step to the tile and consumes the returned value | a user's prompt is partly computed by our hardware | greedy decode produces the SAME token as the pure-golden path, with the offloaded op's value provably from the tile |
| **D** | **Silicon** (owner-gated spend) | the above, on the FPGA | replay of C on hardware |

## 2. The fast path — and what it does NOT require

**Milestone D runs on `agfi-0ae06ea568e5667ba`** (`afi-036d83cafa00d26ea`,
state=available), which is already ingested and has already executed 18 jobs /
27,996 checks on silicon. It is the narrow single-engine CL.

Therefore the prompt-on-chip demo **does not require**:
- the HDPRVerify-41 AFI ingestion gate to be solved (that blocks only the
  GQA-4/wide-D full-layer track),
- the DDR fuel line,
- the wide feeder (`seam_feeder_quant` / `apex_stage_buf` still refuse
  D=3584 on their own legality checks),
- a full decoder layer on hardware.

Milestones A–C are **pure simulation and host software**. No spend.

## 3. Scope discipline (what the demo may and may not claim)

The honest claim at Milestone C/D is: *"one operation of a real 7B decode
step was computed by our verified tile and its result used to produce the
token."* It is **not** "7B runs on our chip", not "the layer runs on our
chip", and 15.625 MHz remains a correctness clock, never a throughput number.
The existing claim-ladder discipline applies unchanged.

## 4. Status — SPRINT STATE (updated 2026-07-29 by S14/prompt session; deadline: FPGA by 2026-07-30)

> **CLOSED 2026-07-29: Milestones A-D ALL PASSED — in simulation AND on
> silicon** (`docs/results/prompt_on_chip/RESULT.md`,
> `agfi-0ae06ea568e5667ba`, token ' Paris' identical, produce-mode). The
> table below is the historical sprint state, kept verbatim. Everything
> after (C1 breadth, S3 sweep, N-lane, 6/6, C2, E-lane, DDR/walker fuel)
> is ledgered in `MASTER_TABLE.md`.

**Owner directive 2026-07-29: maximum parallelism, FPGA-or-bust by tomorrow.**
Full 7-agent readiness audit: `docs/design/PROMPT_DEMO_AUDIT.md` (verdict: the
3-component plan was necessary but not sufficient; 6 components total).

| component | owner | state |
|---|---|---|
| A `cap` op (3 executors) | integration lane (`comp/cap-op`) | CODE LANDED dfab768; **gate never run**; N9 parity defects to fix in-sprint |
| 4 capture EGRESS + per-site tags | prompt session (`comp/prompt-b-c`) | IN FLIGHT (sprint phase 1) |
| 6 env provisioning + A-gate run | prompt session | IN FLIGHT (phase 1) |
| 5 bridge + decoder/grader + A/B harness | prompt session | IN FLIGHT (phase 1-2) |
| 1 input-only compiler (rq_en=0 raw-INT32 attention job + host epilogue via golden requant fns) | prompt session | phase 2 |
| 2 seam: `tf.attention_core` REBIND + hard T≤128 fence (avoids editing the frozen arbiter; chunk asymmetry moot under the fence) | prompt session | phase 2 |
| D hardware session (agfi-0ae06ea…, clkgen A2 + REAL freq verify — the lock preflight is vacuous) | prompt session, owner spend PRE-APPROVED | tomorrow, after C green |

Honest claim at the end (unchanged discipline): "one attention operation of a
real 7B decode step was computed by our tile (raw INT32 accumulators returned
and used) and the token matched the pure-golden path."

## 5. Corrections to the record made while writing this

- **HDPRVerify-41 root cause is NOT confirmed.** The control
  `apex-lc1-a2-20260722` was built WITH `--clock_recipe_a A2` on an unpatched
  kit, ingested successfully, and ran on silicon — so writing the static
  shell MMCM cannot by itself be sufficient to fail ingestion. An earlier
  "confirmed" in this project's notes was premature; the A/B is still open.
- **The multi-head 22,674-check result IS tile-level.** It runs
  `Vtb_apex_l3`, verilated from `rtl/top/apex_top.sv` + the full tile RTL
  (`verif/top/qpath/Makefile`), gated on `L3 PASS` **and**
  `WALKFMT2: fmt=1 WALKED clean`. It is not the walker-unit scoreboard. It is
  simulation, and its expectations are golden-derived per §0 — but it is the
  real tile.
