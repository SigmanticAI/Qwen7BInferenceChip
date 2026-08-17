# perf/Makefile — regenerate the (PROJECTED) performance model report.
PY ?= python3

.PHONY: all check
all:
	$(PY) apex_perf_model.py

check:
	$(PY) apex_perf_model.py --check
