# golden/Makefile — the golden-model regression gate (ARCHITECTURE.md §8).
# `make -C golden test` runs the full suite: frozen-vector contract parity,
# compute-contract properties, and the Layer-3 attention chain + D-021
# error-budget gate. Every target exits non-zero on any failure — this is a
# build gate, not a report generator.

PYTHON ?= python3

.PHONY: test contract compute attention transformer effbits plumbing7b masksem weightcodec layerbus w4bfeeder w4lane lut-tables

test: contract compute attention transformer plumbing7b effbits masksem weightcodec layerbus w4bfeeder w4lane
	@echo "GOLDEN SUITE: contract + compute + attention + transformer + plumbing7b + effbits + masksem ALL PASS"
	@echo "GOLDEN SUITE (B3): + weightcodec ALL PASS"
	@echo "GOLDEN SUITE (IB-LAYER): + layerbus ALL PASS"
	@echo "GOLDEN SUITE (W4B): + w4bfeeder ALL PASS"
	@echo "GOLDEN SUITE (W4LANE): + w4lane ALL PASS"

# IB-LAYER S1: C-LBUS (D-030) hardware bus-grade composition mode — OFF-mode
# bit-identity, RTL-realizability lemmas, pinned deltas. The main banner
# above is exact-matched as a literal by scripts/gen_status.py, so it stays
# byte-identical and this gate reports on its own line (the B3 pattern).
layerbus:
	$(PYTHON) tests/test_layer_bus.py

# B3: native-W4 weight path — the feeder's bit-exact arbiter (C-4) plus the
# stage-0b realization measurement. NOTE: the banner above is exact-matched
# as a literal by scripts/gen_status.py:273, so it stays byte-identical and
# the new gate reports on its own line.
weightcodec:
	$(PYTHON) tests/test_weight_codec.py

# W4-B (D-031): given-scale feeder oracle — reduction identity to the landed
# realization-"B" arbiter + stripe-shared-scale semantics + corners.
w4bfeeder:
	$(PYTHON) tests/test_w4b_feeder.py

# W4 INGEST LANE (D-031 combine): DIRECT host prep + stripe segmentation +
# the tile wire framing + job-level GEMM composition (W4_DATAPATH.md).
# Same banner rule as above: new gate, own line, existing banners untouched.
w4lane:
	$(PYTHON) tests/test_w4_lane.py

# S12: loadable-outlier-mask contract gates (D-027 proposed) — commit rule,
# swap-at-empty composition, and the records-need-their-encoding-mask proof.
masksem:
	$(PYTHON) tests/test_mask_semantics.py

# S6: GQA + 7B plumbing contracts (C-KSPLIT / C-RMSW / C-CHUNK / QKV biases)
plumbing7b:
	$(PYTHON) tests/test_7b_plumbing.py

# Honest KV-compression accounting (ARCHITECTURE.md §4 promise): pins the
# stored-record, codec-ceiling, and CSR-advertised bits/value so record-layout
# or codec changes must update the numbers consciously.
effbits:
	$(PYTHON) tests/test_effective_bits.py

contract:
	$(PYTHON) tests/test_contract.py

compute:
	$(PYTHON) tests/test_compute.py

attention:
	$(PYTHON) tests/test_attention.py

transformer: lut-tables
	$(PYTHON) tests/test_transformer.py

# RoPE/SiLU LUT .svh must be byte-identical to the golden tables (provenance
# gate, D-013 spirit): regenerate into a temp and diff the checked-in copy.
lut-tables:
	$(PYTHON) gen_rope_lut_tables.py --out /tmp/rope_lut_tables.svh
	diff -q /tmp/rope_lut_tables.svh ../rtl/rope/rope_lut_tables.svh
	$(PYTHON) gen_silu_lut_tables.py --out /tmp/silu_lut_tables.svh
	diff -q /tmp/silu_lut_tables.svh ../rtl/asu/silu_lut_tables.svh
