# perf/ — the APEX analytic performance model (S7)

**Everything this model outputs is a PROJECTION.** No APEX silicon exists.
The verified artifacts today are the bit-exact simulation suites (STATUS.md)
and one ECP5-85F bitstream of the KVQ engine at 9.04 MHz. This directory
answers "what would the architecture do at ASIC scale?" with arithmetic on
stated assumptions — it is a roofline calculator, not a benchmark.

## Run it

```sh
python3 perf/apex_perf_model.py          # regenerates docs/results/perf_model/
python3 perf/apex_perf_model.py --check  # calibration anchors only (CI gate)
# or: make -C perf
```

Pure Python 3.9 stdlib — no pip installs; reproducible from a clone.

## What keeps it honest

1. **Calibration anchors are measured and asserted.** `--check` fails the run
   if the model stops reproducing the measured cycle data it claims to be
   calibrated to: ~300 MMIO/host-sequenced step, 33 cyc/element ASU divider
   (540,672 cyc @ T=128) vs ~65k MXE cycles, 30.6% measured MXE utilization
   (docs/OPTIMIZATION.md), 9.04 MHz ECP5 Fmax
   (docs/results/kvq_ecp5_report.txt), and the weight-port decode roofline
   (0.010 / 0.113 / 0.453 tok/s at 9.04 / 100 / 400 MHz).
2. **KV traffic uses the stored-record accounting** pinned by
   `golden/tests/test_effective_bits.py` (4.5625 b/v = 3.51× at D=128 as the
   RTL stores it), never the codec ceiling (4.125 b/v = 3.88×, plotted as a
   separate labeled curve).
3. **Every assumption is printed** into the generated report before any
   result, including the list of *unbuilt* features the projections assume
   (GQA, B1 walker, B3 native W4, A1/B4 utilization targets).
4. **The report is machine-generated** (`docs/results/perf_model/PERF_MODEL.md`
   plus CSVs and SVG charts). Never hand-edit it; edit the assumptions here
   and regenerate.

## Headline outputs (all PROJECTED; see the report for the full context)

* Decode at 7B is memory-bound: 64×64 INT8 + LPDDR5X ×64 (68.3 GB/s peak,
  65–75% sustained) + W4 weights → ~12–14 tok/s short-context.
* KVQ stretches the ≥10 tok/s region from ~19k to ~67k context as stored
  (~74k at the codec ceiling); at 128k ctx decode is ~2× faster with KVQ on.
* TTFT(1k) ≈ 3.4 s at 64×64 / 0.8 GHz / 65% utilization; prefill sizes the
  array, decode sizes the memory.
* ~0.16–0.28 J/token at ~3 W (DRAM pJ/bit is the dominant uncertainty).
* Without the B1 hardware walker, host MMIO alone caps decode at well under
  1 tok/s — no memory system can fix that.
