# =============================================================================
# verif/mxe/Makefile — everything MXE: the implementer's smoke test, the
# independent verification suite, the D-004 structural discriminator, and
# the D-005 load-under-compute perf evidence.
#
#   make all    - run all four suites end to end (each gates on its own result)
#   make smoke  - implementer's 4-test smoke (verif/mxe/smoke)
#   make sb     - independent suite (verif/mxe/sb): lint, vectors, build,
#                 6 run configurations, coverage aggregation gate
#   make struct - D-004 systolic structural discriminator (whitebox timing law
#                 + glitch hop locality + broadcast-wall mutation check)
#   make perf   - D-005 perf evidence (overlap>0 + cycle win vs the pinned
#                 sequential baseline ctrl; kept-failing overlap regression)
# =============================================================================

.PHONY: all smoke sb struct perf clean

all: smoke sb struct perf

smoke:
	$(MAKE) -C smoke smoke

sb:
	$(MAKE) -C sb all

struct:
	$(MAKE) -C struct all

perf:
	$(MAKE) -C perf all

clean:
	$(MAKE) -C smoke clean
	$(MAKE) -C sb clean
	$(MAKE) -C struct clean
	$(MAKE) -C perf clean
