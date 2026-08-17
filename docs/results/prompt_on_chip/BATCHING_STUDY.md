# BATCHING STUDY — N regops jobs in ONE executor invocation

**Date:** 2026-07-30 · **Branch:** `comp/prompt-b-c` · **New code:**
`scripts/fpga/f2/batch_exec.py` · **Machine record:**
`batching_sim_bench.json` (this directory) · **Executor used: SIM ONLY —
no hardware was launched or touched for this study.** All silicon numbers
below are the previously-committed C1 measurements
(`hw_breadth_result.json`, `hw_breadth_step.log`) plus projections clearly
marked as such.

## Why this is the biggest available speed win

Measured on silicon (C1, 2026-07-30): one job costs a **median 3.73 s wall
(n=232 projection jobs, min 3.37 / max 4.97; heads 3.9–4.5 s)** —
`hw_breadth_result.json` `projections[].job_wall_s` — of which only
**~0.78 ms is real MMIO** (recorded in `docs/design/MASTER_TABLE.md`, the
T7 row; the `:44-46` line citation this doc originally carried predates the
table's later rewrites). That is
**~99.98% per-invocation overhead**, and none of it is the tile. It needs
NO new image: both executors already take many regops files in one
invocation (`f2_host_run.py:78` `nargs="+"`; `sim_main.cpp` argv loop) and
both isolate state per file (`f2_host_run.py:146` toggles TILE_RST 0x000C
before EVERY file; `sim_main.cpp:603` calls `reset()` before every file —
the per-file isolation contract was proven on silicon 2026-07-22 by the
`_c1` chunk-stall fix).

## (a) What is per-INVOCATION vs per-FILE (from the code, cited)

**`tile_exec_bridge.run_job` (sim path)** — per invocation: binary
resolution + temp-cap-file setup (`tile_exec_bridge.py:219-232`); ONE
`subprocess.run` = process spawn + Verilator model construction + cap-file
open (+ optional DDR image load) (`sim_main.cpp:553-599`); then summary
scan + cap parse + audit (`tile_exec_bridge.py:258-283`). Per file:
`reset()` + the op loop (`sim_main.cpp:600-625`).

**`remote_hw_exec.run_job_remote` (hw path)** — per invocation, in order:

1. clock-gate probe — 1 ssh round trip (`sudo -n fpga-describe-clkgen`)
   (`remote_hw_exec.py:441-464`)
2. `mkdir -p` on the instance — 1 ssh (`:471-479`)
3. scp upload of the regops files + the runner itself (`:481-498`)
4. the job ssh (`:500-513`), inside which the remote side pays: python
   interpreter start, SDK Cython-binding import + `pci_attach`
   (`f2_host_run.py:109-123`), and the (vacuous) clkgen-wait preflight
   (`f2_host_run.py:120-121`)
5. scp fetch of the capture file (`:534-541`)
6. local summary/paranoia parsing (`:550-580`)

Per FILE inside `f2_host_run.py` (`:137-267`): TILE_RST toggle with two
10 ms sleeps (`:146-149`), `json.loads` of every line (measured locally:
764k lines/s → **3.4 ms** for a 2,579-op job), the op loop (the only real
MMIO, ~0.78 ms/job), one status print.

So ~3.7 s of every 3.73 s job is items 1–6 — three ssh/scp handshakes plus
remote python+SDK start — wrapped around **~25 ms** of per-file work.
Notably, `run_job_remote` already accepts a LIST of files: batching needed
**zero executor changes**, only a correct demultiplexer.

## (b) Implementation — `scripts/fpga/f2/batch_exec.py` (NEW file)

`run_jobs_batched(paths, …)` plans a batch, hands ALL files to
`tile_exec_bridge.run_job` (or the `remote_hw_exec.attach()` shim — the hw
path inherits batching for free) in ONE invocation, and demultiplexes the
capture egress back to per-file results.

**Attribution** (the part that needs care — tags are unique per job, NOT
across jobs, and `i` is global): two independent mechanisms, both required
to agree:

1. **Boundary markers** — a one-line regops file interleaved before every
   job and after the last: `cap` of BAR0 `0x0000` (the RO bridge ID
   register, reads `0x41394558` "A9EX", side-effect-free) with tag
   `__bm_<nonce>_<k>__`. Markers are SEPARATE files, so job files stay
   byte-identical to their canonical artifacts; each marker costs one extra
   per-file reset (measured in sim: below noise, ≤~1 ms). Captures between
   marker k and k+1 belong to file k regardless of what happened inside it.
2. **Per-file manifests** — each job file's programmed `cap` sequence
   (tag/addr/mask, in order) is extracted up front; the demuxed segment
   must equal it (complete) or be a strict prefix (file aborted →
   flagged PARTIAL, never silently attributed). Anything else = hard
   ATTRIBUTION FAILURE.

A mid-batch stall does NOT kill the batch: both executors continue with the
next file (`f2_host_run.py:137` / `sim_main.cpp:600` loops), and the demux
reports exactly which file stopped after how many caps.

Found while implementing (worth knowing): `sim_main.cpp`'s flat-JSON
extractor matches the literal pattern `"key":` — a space after the colon
(python's `json.dumps` default!) makes it silently SKIP every op (file
"runs", 0 executed). `batch_exec._jline` emits compact separators, matching
`trace_to_regops.py`.

### Attribution proof (pasted; fake-executor selftest, 7/7)

```
  [1] ok  A -> ['0x11111111', '0x11111111'], B -> ['0x22222222', '0x22222222', '0x22222222'] (same tags, different files, correctly split)
  [2] ok  duplicate cross-file tags handled (attribution is by marker window, never by tag)
  [3] ok  abort detected: B partial 1/3 (note: PARTIAL: 1/3 programmed caps executed — …), C still bit-correct, batch ok=False
  [4] ok  tampered stream flagged: ATTRIBUTION FAILURE: observed caps do not match the programm…
  [5] ok  missing marker raised: capture 'not_the_programmed_tag_0' (i=1) arrived outside any…
  [6] ok  batched == separate for 3/3 files (tag/addr/mask/value; `i` renumbering excepted)
  [7] ok  job files byte-identical after batching (markers are separate files)
BATCH_EXEC SELFTEST: PASS (fails=0)
```

### Attribution proof on the REAL verilated executor (pasted)

Two synthetic jobs whose captured VALUES differ (scratch reg 0x0008 written
0x11111111 vs 0x22222222), plus two real committed produce-mode jobs whose
capture streams differ — batched vs one-at-a-time, capture-for-capture:

```
  [scratch] batched: ok=True A=['0x11111111', '0x11111111'] B=['0x22222222', '0x22222222']
  [real poff_s000_L00_h00.compute.regops.jsonl] batched n=164/164 complete=True  == separate: True
  [real poff_s000_L13_h07.compute.regops.jsonl] batched n=164/164 complete=True  == separate: True
  [real] the two files' capture streams differ from each other: True (required for a meaningful proof)
BATCH_EXEC PROOF (real sim): PASS
```

## (c) SIM measurement — N=1,4,16,64, one-at-a-time vs batched (pasted)

Jobs = the 13 committed produce-mode `poff_*.compute.regops.jsonl`
programs (2.6 MB total), cycled under unique names. Binary =
`verif/f2sim/obj_d128_ddr0/f2sim` (the silicon twin). Record:
`batching_sim_bench.json`.

```
  N=  1  separate=    0.07s ( 0.069s/job)   batched=   0.06s ( 0.065s/job)   speedup= 1.06x   ok=True/True  captures_equal=True
  N=  4  separate=    0.32s ( 0.079s/job)   batched=   0.35s ( 0.087s/job)   speedup= 0.91x   ok=True/True  captures_equal=True
  N= 16  separate=    2.33s ( 0.146s/job)   batched=   2.26s ( 0.141s/job)   speedup= 1.03x   ok=True/True  captures_equal=True
  N= 64  separate=   10.16s ( 0.159s/job)   batched=  10.01s ( 0.156s/job)   speedup= 1.01x   ok=True/True  captures_equal=True
  per-invocation constant (sim, est from sep-vs-batch deltas): median 2.4 ms (spread -10.5–4.7 ms)
BATCH_EXEC BENCH: PASS
```

Direct micro-measurement of the sim constant (20 invocations of a 2-op
file through the bridge): **median 6.1 ms** (min 4.7, max 42.7). Marginal
cost of one marker file inside a batch: below measurement noise.

**Honest reading of the sim curve: the speedup is ~1.0×, and that is the
CORRECT result, not a failure.** The sim's per-invocation constant is
~2–6 ms against ~65–160 ms jobs (~4%), so there is almost nothing to
recover — `captures_equal=True` at every N is the real product of this
measurement (batching is a pure transport change; per-file results are
bit-identical). The sim measures the *correctness* of batching and the
*shape* T_sep − T_bat ≈ (N−1)·C_inv; the *magnitude* of the win is a
property of the silicon transport, whose constant is ~600× larger.

## (d) Projected silicon speedup — PROJECTION, clearly labeled

Inputs: C_hw ≈ **3.7 s** per invocation (median 3.73 s job wall minus
~25 ms per-file work; measured over 262 invocations in C1);
per-file in-batch cost ≈ TILE_RST 20 ms + parse ~3–10 ms + MMIO ~0.8 ms +
marker ~21 ms ≈ **~50 ms/job** (parse and marker terms grounded above;
TILE_RST sleeps from `f2_host_run.py:147-149`); one-time upload
~32 B/regop (24 MB for C1's 747,796 regops; ~190 MB for a full-layer run).

| run | measured / prior projection (one-at-a-time) | batched (PROJECTED) | factor on the job stream |
|---|---|---|---|
| C1, 260 jobs | **984 s = 16.4 min measured** job stream (Σ `job_wall_s` + head walls) inside a ~35 min / $1.17 card session | 3.7 s + 260×~0.05 s + upload/fetch ~10–20 s ≈ **~30–45 s** | **~22–33×** |
| full-layer `--full`, 2,076 jobs | 2.19 h (projected at 3.8 s/job, `hw_breadth_step.log:139`) | 3.7 s + 2076×~0.05 s + ~190 MB upload ≈ **~3–5 min** | **~26–44×** |

Even taking the per-file cost at 4× the estimate (200 ms/job), C1 batched
is ~60 s → **>16×**; the "10×+" claim survives a large error margin. Note
the Amdahl split for a whole card session: batching collapses the *job
stream* (16.4 of C1's ~35 card-minutes); the golden-side prefill/grading
(337 s) and session setup are untouched — pre-building programs before the
card comes up is what converts the job-stream win into card-minutes.

**Honest caveats:**

1. The sim per-invocation constant (6 ms) says nothing about the silicon
   constant (3.7 s) — different transports. The projection stands on the
   silicon-measured 3.73 s median and 0.78 ms MMIO figures, both from
   committed artifacts, neither re-measured here.
2. **No batched run has ever executed on hardware.** One long ssh
   invocation carrying 260+ files, sudo session lifetime, single-scp of
   hundreds of files, and remote parse at that scale are all untested;
   `timeout_s` must be scaled with N.
3. The projection assumes everything outside TILE_RST+parse+MMIO is
   strictly per-invocation. Any hidden per-file term on the remote path
   (e.g., capture-file flush behavior at 11k+ caps) degrades it.
4. Practical batch size on hw: ~32–64 jobs/invocation amortizes C_hw to
   <0.12 s/job while keeping a single stall's blast radius and the ssh
   timeout bounded; 260-in-one is not required to get the win.

## (e) MEASURED ON SILICON — 2026-07-30 evening session (caveat 2 retired)

Instance `i-013437f5cefdaca4b` (f2.6xlarge, us-west-2b), AGFI
`agfi-0ae06ea568e5667ba`, clkgen A2 verified **15.62 MHz by number** through
both gates (box-side awk gate + `remote_hw_exec --check-clock`) before any
job. Session log: `build/hw_s2_sweep/session.log`; machine record:
`build/hw_s2_sweep/attrib_proof.json`.

**1. Attribution holds on real silicon.** 8 committed produce-mode K-split
jobs (`bs_L00_s010_Wv_n0000_k0..k15`, 72 real tile captures), run separate
then batched: **every file's capture stream identical**
(tag/addr/mask/value), `HW ATTRIBUTION PROOF: PASS`.

**2. The speedup is real but class-dependent — the 22–33× projection was
optimistic; the study's own caveat 3 fired.**

| job class | separate | batched | speedup | why |
|---|---|---|---|---|
| produce-mode K-split (C1 class, ~2.6k ops) | 3.63 s/job (n=8; matches C1's 3.73 median) | **0.82 s/job @ N=8** | **4.5×** | marginal in-batch cost measured **~0.21 s/job** — ~4× the ~50 ms estimate, exactly the caveat-3 "hidden per-file term" case |
| canonical 18-job replay (~27.6k checked ops each) | 5.31 s/job (n=3) | **1.67 s/job @ N=18** | 3.2× | this class is MMIO/parse-bound: its floor IS the 27.6k-op stream, no invocation overhead to recover |

Revised at-scale projection from the MEASURED marginal (~4.8 s setup +
N×0.21 s): N=512 → ~0.22 s/job → the C1 job stream ≈ 62 s vs 984 s ≈
**~16×** (not 22–33×). Batch-8/18 measurements above are direct; the
first at-scale run is the full projection sweep (section f).

**3. Correction to the (d) full-layer row:** 2,076 jobs assumed the SIM
chunking (k=2048 chaining). On silicon the image-parity rule
(`rows_per_desc=1`, 05efb2a) makes one 8-column block ~28 K-split jobs —
`--full` is **~28.7k jobs ≈ 29 h separate**, viable only batched
(~2 h at the measured marginal). This is why `proj_sweep_batched.py`
exists.

## Files

- `scripts/fpga/f2/batch_exec.py` — implementation + `--selftest` (fake
  executor, no build needed) + `--prove` (real sim) + `--bench`; main now
  attaches `remote_hw_exec` (breadth_step:1001 idiom) so `--executor hw`
  batches over one ssh
- `scripts/fpga/f2/proj_sweep_batched.py` — the full-sweep driver built on
  run_jobs_batched + the verified gemm_job/breadth_step primitives
- `docs/results/prompt_on_chip/batching_sim_bench.json` — machine record
  of the N=1,4,16,64 sim measurement
- `build/hw_s2_sweep/` — 2026-07-30 silicon session log + attribution
  record (curated copies land in this directory with the sweep result)
- this file — the study
