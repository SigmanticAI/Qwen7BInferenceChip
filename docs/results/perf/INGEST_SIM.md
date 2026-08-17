# IB-FUEL ingest — SIM measurement + bottleneck prototypes (perf/ingest-lane)

**Every number in this document is SIM-measured** on the behavioral twin
(`verif/f2sim` verilated `cl_apex` + `sh_ddr_beh.sv`, Verilator 5.044,
base `comp/prompt-b-c` @ 05ea8c5). **No number here is a silicon claim** —
the DDR model is our own fabrication of an encrypted controller
(IB_FUEL.md §6 honest-model disclosure), deliberately stricter than real
hardware; silicon remains the only arbiter of the real thing.

**Why:** the 4 tok/s budget needs sustained DDR→tile weight ingest of
~2 GB/s. At the 250 MHz shell clock that is **8 B/shell-cycle** — exactly
the architectural ceiling of today's fuel line (the 64-bit-wide
`apex_afifo` write side accepts at most one 8 B lane beat per shell
cycle). Anything the line loses to bubbles, turnarounds, or framing is
therefore unrecoverable margin against the budget. This lane built the
measurement first, then prototyped fixes for what the measurement showed.

Fenced out of scope, untouched: `rtl/seq/seq_walker_comp.sv`,
`rtl/top/glue/apex_wcomp_bank.sv` (task fence), all walker descriptor
internals (IB-WALK's), and any AWS/hardware work.

---

## 0. Measurement rig (s1, s1b — commits 205cced, 4e4ab04)

Harness-side monitor compiled into `verif/f2sim/sim_main.cpp` under
`-DAPEX_INGEST_MON` — **zero RTL change for measurement**; plain builds
carry no monitor and refuse `+ingest_mon` loudly. Per shell cycle it
entry-samples (the settled low-phase values the imminent posedge
consumes): the reader's AXI AR/R handshakes, the afifo write side
(`fifo_wvalid/wready`), the afifo read side (counted only on `clk_tile`
posedges), `fuel_rdr_busy`, and the 4-phase `req/ack` toggles. Occupancy
is the committed push−pop delta (write-domain view); a 10-bucket
histogram accumulates per cycle. Gap timers: rlast→next-AR (inter-burst),
ack→next-AR (record framing), ack→req-toggle (the ctl+CDC half of the
framing gap), plus walker `wf` handshake counts. Snapshots print at every
CYCMARK note and at file end; `verif/f2sim/ingest_report.py` reduces two
snapshots to the window metrics quoted below.

Monitor build (fresh `--Mdir`, IMG_05B geometry, DDR=1):

```
make -C verif/f2sim build D=64 DDR=1 OBJ=obj_ing_s1b VFLAGS_EXTRA=\
  '+define+APEX_CL_DM=896 +define+APEX_CL_GQA=2 +define+APEX_CL_QSTAGE=14
   +define+APEX_CL_DMODEL=64 --public-flat-rw -CFLAGS -DAPEX_INGEST_MON'
```

Measurement program: **the E-6 walk gate**
(`scripts/fpga/f2/walk_fuel_layer.py run --mon`), tile_div=2, layer-0
0.5B weights resident in (behavioral) DDR. The measured window is the
E-6 driver's own CYCMARK `E6R-PREGO → E6R-WALKDONE` delta — the
host-free walk window. Chain B ({QKV+OPROJ+RES1} under one descriptor,
4 walker-issued fuel records: Wq 12 544 + Wk 1 792 + Wv 1 792 + Wo
12 544 = 28 672 words = 1.75 MiB) is the full-walked-layer measurement;
chain A ({OPROJ+RES1}, one 12 544-word record) is the single-record
cross-check. The gate's own verdict (r1 bit-exact, host-free window,
QKV 144/144, walk-off/poison/rq-delta discriminators all RED) is the
correctness harness for every A/B below — a perf change that broke
bit-exactness would turn the gate red before any number was quoted.

**Monitor validity.** The s1b monitor run reproduces the committed E-6
replication number exactly: chain B walk window **2 238 788 shell cycles
= 559 697 tile cycles at div2** (WALKED_EPILOGUE_E6_REPLICATION.md's own
figure), and both baseline runs land every CYCMARK on the same absolute
cycle. The monitor build is cycle-invisible, as designed (`--public-flat-rw`
+ observation-only C++; no RTL touched).

## 1. The path, and the theoretical bounds

```
sh_ddr_beh (250 MHz shell)  — AR accept → first R beat: 25 cyc (default),
   │  512-bit R beats           one read burst outstanding (model-strict)
   ▼
apex_fuel_reader  — INCR, AxSIZE=6, ≤4 KiB bursts, ONE outstanding;
   │                unpack: 1×512b word → 8×64b lane pushes
   ▼  64-bit lane beats
apex_afifo W=64 D=512  — the shell→tile CDC; ≤1 push/shell-cycle
   ▼  (FWFT read side, clk_tile)
apex_fuel_ctl → xw mux → tile (≤1 lane beat/tile-cycle)
```

| bound | B/shell-cycle | GB/s @ 250 MHz |
|---|---|---|
| AXI R channel (512-bit) | 64.0 | 16.0 |
| **afifo write side (W=64)** | **8.0** | **2.000** |
| tile demand at div2 (8 B/tile-cycle, tile=62.5 MHz) | ≤2.0 | ≤0.500 |
| tile demand at div8 (15.625 MHz) | ≤0.5 | ≤0.125 |

Baseline steady-state prediction from the RTL (before measuring): the
reader accepts an R beat only when its unpack stage is EMPTY
(`m_rready = F_COLLECT && unpk_empty`), so each 64 B word costs 1 accept
+ 8 push cycles = **9 cycles → 7.11 B/cyc**, an 11% self-inflicted loss
against the afifo bound before any DDR latency is counted. Per 64-word
burst: 1 (AR) + 25 (latency) + 576 (9×64) = 602 cycles worst-case
un-overlapped.

## 2. Baseline (tree = monitor commits only; binary `obj_ing_s1b`)

E-6 gate verdict: **PASS** (all six checks; both baseline runs).

Chain B — the full walked layer, PREGO→WALKDONE window, div2:

| metric | SIM-measured |
|---|---|
| window | 2 238 788 shell cycles (559 697 tile cycles) |
| DDR-side achieved | 0.820 B/cyc (0.205 GB/s-equiv) |
| afifo push side | 0.820 B/cyc — equals pop side; FIFO returns to 0 |
| **supply capability while FIFO not full** | **7.106 B/cyc = 88.8% of the 8 B/cyc bound (1.777 GB/s-equiv)** |
| reader busy | 2 219 564 cyc (99.1% of window) = 229 376 push + 1 961 336 fifo-full stall + **28 852 bubble** |
| bursts | 448, **avg 64.00 words** (all full 4 KiB), inter-burst gap 1.0 cyc |
| records | 4; inter-record reader idle avg 4 769.7 cyc (see §3.3) |
| occupancy | mean 509.1/512; 68.6% of window cycles pegged FULL, 30.6% in 449–511 |

Chain A agrees: supply 7.108 B/cyc, 196 bursts × 64.00, bubbles/word ≈
1.004. The walked layer at div2 is **demand-bound** — the tile's own
consumption (0.82 B/cyc average over the window) paces the walk, the
FIFO sits pegged full, and E-6 wall-clock is insensitive to supply-side
changes. The number that carries the 2 GB/s budget is the **supply
capability**: 7.11 B/cyc = **1.78 GB/s-equivalent — below the ~2 GB/s
target before any fix**.

## 3. Bottleneck decomposition (SIM-measured)

1. **Unpack serialization (the 1-in-9 bubble) — the top supply-side
   bottleneck.** Chain B bubbles = 28 852 ≈ 1.006 per word (28 672
   words): exactly the predicted one dead accept-cycle per 64 B word.
   Costs 11.1% of the afifo bound. → fixed in §4.1.
2. **AR turnaround + DDR latency.** Inter-burst gap measured 1.0 cycle
   (rlast→AR), and the 25-cycle first-beat latency is almost fully
   hidden at div2 — the unpack tail (8 lanes draining against a full
   FIFO at ≤1 pop/4 shell cycles) covers it. It is NOT hidden when the
   FIFO is shallow (record starts from empty, or any faster consumer):
   exposed latency shows up as bubbles. → prototyped in §4.2.
3. **Fuel-record framing.** Inter-record reader idle avg 4 769.7 cyc
   (3 gaps) — and the split counter (s1b) shows where it lives:
   ack→req-toggle avg **4 766.7 cyc** (the walker issuing each tensor's
   record just-in-time — its 2-deep skid accept-ahead exists, the walker
   FSM simply doesn't run ahead: IB-WALK's documented deferred prefetch,
   fenced out of this lane), req-toggle→AR avg **3.0 cyc** (the fuel
   line's own reader half: 2FF req sync + AR issue). The fuel-lane
   framing machinery costs single-digit shell cycles per record; the
   whole gap is hidden behind tail consumption in every demand-bound
   regime measured here. → measured, no RTL change justified (§4.3).
4. **Burst length — measured optimal, no fix needed.** 448/448 bursts at
   the full 64-word (4 KiB) maximum: the tensor bases are 4 KiB-aligned
   by the image builder and the reader's boundary trim never fires
   mid-tensor. "Longer bursts" is not on the table (4 KiB is the AXI
   protocol ceiling per burst).

## 4. Fixes — one commit each, before/after SIM-measured

### 4.1 F1 — reader unpack skid (remove the 1-in-9 bubble) — commit 7c9a91d

One wire in `apex_fuel_reader.sv`: `m_rready` also asserts while the
current word's LAST lane is leaving (`unpk_cnt==1 && push`); the
F_COLLECT load is written after the shift and wins the nonblocking race,
so the next 512-bit word lands exactly as the old word's last lane
departs. Steady state: 8 cycles/word — the afifo write-side bound.
Backpressure is unchanged (a full FIFO freezes the lane count above 1,
so `rready` stays low). DRAIN/ABORT/F_ACK semantics untouched.

Before/after, same program, same knobs, SIM-measured (chain B window;
chain A in parentheses):

| metric | baseline s1b | F1 |
|---|---|---|
| supply capability | 7.106 (7.108) B/cyc | **7.993 (7.996) B/cyc — 99.9% of bound** |
| GB/s-equiv @ 250 MHz | 1.777 | **1.998** |
| bubbles | 28 852 ≈ 1.006/word | **188 (47)** — 4 (1) record-start latency exposures only |
| walk window | 2 238 788 shell cyc | **2 238 788 — IDENTICAL to the cycle** |
| E-6 gate | PASS | PASS (all six checks, discriminators RED) |

The unchanged window is the demand-bound proof: at div2 the tile paces
the walk, so a supply-side fix moves supply headroom, not E-6 wall-clock.

### 4.2 F2 — pipelined AR issue (prefetch depth 2) — commit fa47a39

Reader: the issue side (AR pointers + a 2-deep burst-length queue) splits
from the collect side, so the next burst's AR is presented while the
current burst streams — at most two un-retired ARs, single ID, INCR,
in-order, ≤4 KiB each; the rlast cross-check runs against the
issue-order queue head; ABORT drains every outstanding burst. Model:
`+ddr_pipe=2` lets `sh_ddr_beh` accept ONE read AR ahead (latency counts
down while queued); **the default stays 1** — knob-for-knob identical
for every existing gate, and the honest-model stance is unchanged (the
real controller pipelines more than either setting; silicon is the only
arbiter).

The E-6 gate on the F2 tree: **PASS** (all six checks; chain B window
again 2 238 788 — identical to the cycle). At div2 defaults F2 changes
nothing measurable, exactly as §3.2 predicts (latency hides under
full-FIFO stalls; the strict model takes one AR at a time). The A/B that
isolates it is the latency-stress probe matrix — chain A program, six
runs, every run green (FUELAUDIT ok) and **all six capture-egress files
byte-identical** (bit-stability across trees and timing knobs):

| tree | knobs | supply B/cyc | exposed-latency bubbles |
|---|---|---|---|
| F1 (before) | default (lat=25) | 7.996 | 47 |
| F1 (before) | +ddr_lat=100 | 7.547 (94.3% of bound) | 6 017 |
| F1 (before) | +ddr_lat=100 +ddr_pipe=2 | 7.547 — model would accept early, reader never issues early | 6 017 |
| F2 (after) | default (lat=25) | 7.996 | 47 |
| F2 (after) | +ddr_lat=100 | 7.547 — reader issues early, strict model refuses | 6 017 |
| **F2 (after)** | **+ddr_lat=100 +ddr_pipe=2** | **7.992 (99.9% of bound)** | **103** |

Both controls are exact no-ops; the pair together overlaps the whole
100-cycle latency under the previous burst's 512 data cycles. The 103
residual bubbles are the record-START exposure (first burst of a record
has nothing to overlap with) — irreducible by AR prefetch, addressable
only by walker-side record prefetch (fenced, IB-WALK).

### 4.3 F3 — record framing: measured, and why no RTL change ships

The candidate list named "fuel-record framing overhead"; the split
counters (s1b) put a measured number on it, per record boundary at div2:

| segment | SIM-measured | owner |
|---|---|---|
| ack → next req toggle | 4 766.7 cyc avg | ~all walker just-in-time issue (the 2-deep skid accepts ahead — `wfhs` proves records arrive — but the walker FSM issues each tensor's record only when its phase needs it: IB-WALK's documented deferred prefetch, outside this lane's fence) |
| req toggle → AR at DDR | 3.0 cyc avg | the fuel line itself (2FF req sync + AR issue) |
| first-burst latency exposure | ~26–47 cyc/record (the F1 tree's only remaining bubbles) | DDR model latency, addressed by F2 under `+ddr_pipe=2` |

The fuel-lane framing machinery costs **single-digit shell cycles per
record** — against records of 1 792–12 544 words that is < 0.01% and it
is fully hidden behind tail consumption in every demand-bound regime
measured. No RTL change is justified by this number; the ~4.8k-cycle
walker-side gap is real but belongs to IB-WALK's deferred walker-FSM
prefetch (their reservation, their lane). What this lane CAN say,
SIM-measured: when that prefetch lands, the fuel line's skid + 4-phase
machinery adds essentially nothing on top.

## 5. Where the line lands vs the budget

Supply capability of the fuel line, SIM-measured on the E-6 walked layer
(chain B unless noted), against the ~2 GB/s (= 8 B/cyc @ 250 MHz) budget:

| tree | conditions | supply B/cyc | GB/s-equiv | % of afifo bound |
|---|---|---|---|---|
| baseline | default | 7.106 | 1.777 | 88.8% |
| + F1 | default | 7.993 | 1.998 | 99.9% |
| + F1 + F2 | default | 7.993 | 1.998 | 99.9% |
| + F1 + F2 | lat=100 stress, strict model | 7.547 (chain A) | 1.887 | 94.3% |
| + F1 + F2 | lat=100 stress, `+ddr_pipe=2` | 7.992 (chain A) | 1.998 | 99.9% |

After F1+F2 the fuel line delivers **within 0.1% of its architectural
ceiling** in every regime measured — the line itself is no longer where
ingest margin is lost. What remains between here and *clearing* 2 GB/s
with margin is structural, and out of this lane's scope:

- **afifo width** (W=64 → 128 doubles the ceiling to 16 B/cyc =
  4 GB/s @ 250 MHz): a CDC + constraint-surface change plus a read-side
  splitter; PROJECTION ONLY, not measured here, and it must clear the
  LUT/BRAM budget question the afifo header already flags.
- **walker-side record prefetch** (IB-WALK's reserved FSM change):
  removes the per-record start exposure and the 4.8k-cycle inter-tensor
  reader idle — the fuel line's skid + 4-phase already add ~nothing on
  top (§4.3), so the walker change lands on prepared ground.
- **tile demand**: at every supported sim ratio the tile drinks ≤2
  B/shell-cycle (div2) — the 2 GB/s budget presumes the roadmap
  consumer, not today's single xw port at demo clocks. The fuel line is
  now ready to feed that consumer up to its afifo bound.

## 6. Battery (final tree = F1 + F2 + monitor commits)

Logs under `docs/results/perf/ingest_sim/` and `build/battery/` (run via
the sequential pipefail-safe script; every verdict read from the sim's
own exit code + counted output, never through a pipe).

| gate | result |
|---|---|
| behsmoke (modified `sh_ddr_beh`, both knob sets) | **PASS** 2× `checks=801 fails=0` — the `+ddr_pipe` addition is knob-for-knob inert at default |
| 18-job HOST-mode replay, D=128 div8 | **PASS** `files=18 checks=27996 fails=0`, final cyc 433 174 172 |
| host-mode cycle parity vs pre-change | **IDENTICAL** — a reference twin built at 99cd1ee (RTL = the 05ea8c5 base, zero fuel-path changes) replays the same 18 regops to the SAME final cycle and the same 18 per-file cumulative cycles (`host_div8_ref_99cd1ee.log` vs `host_div8_final.log`). The fuel RTL stays cycle-invisible inert in host mode. **Stale-constant note:** IB_FUEL.md's 432 469 308 (2026-07-22) does not hold on `comp/prompt-b-c` — the +704 864 drift pre-dates this lane (present in the pre-change reference), i.e. it came with the branch's own post-July RTL evolution, not with F1/F2. |
| 18-job FUEL-mode replay div8 / div7 / div2 (three-ratio CDC discipline) | **PASS ×3** — every ratio `files=18 checks=27996 fails=0` with 18/18 `FUELAUDIT ok` (final cyc 264 106 748 / 231 108 777 / 70 921 850; div8 is **exactly +704 864** over the July constant — the same mode-independent branch drift as host mode, i.e. F1/F2 add zero net cycles to the demand-bound replay) |
| mutants (control + M1/M2/M3) | **PASS** — control exit 0; M1 (lane-slice swap, 469 fails) / M2 (beats off-by-one) / M3 (afifo wrap) all RED **on the F1+F2 tree**: the check surface still bites after the perf changes |
| capgate | **PASS** `caps=505 values_matched=505/505` (DDR=0 silicon-twin binary) |
| E-6 walk gate (final tree) | **PASS** — §4.2 (all six checks, discriminators RED) |
| walk_fuel_layer selftest | **PASS** (ALL PASS) |

Process disclosure: the first battery pass's *summary aggregation* aborted
mid-list — the script file was edited while bash was still reading it (a
self-inflicted truncate-in-place; the per-gate LOGS above were already
complete and are the cited evidence), and the launch line's `| tail`
masked the aborted script's exit status — the exact pipeline-masking
class this repo's audits keep flagging, reproduced first-hand. The
remaining legs (mutants, ddr0+capgate, selftest) re-ran from an immutable
copy with no pipes near a verdict (`battery_tail_summary.txt`), and the
committed `verif/f2sim/ingest_battery.sh` is the corrected single script.
