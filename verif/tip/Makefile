# =============================================================================
# verif/tip/Makefile — everything TIP: the implementer's smoke suite plus the
# independent verification suite (scoreboard TB + SVA binds + protocol
# adversaries + phase-targeted mid-op reset + coverage gate + mutation gate).
#
#   make all    - run both suites end to end (each gates on its own result)
#   make smoke  - implementer's 7-run smoke (verif/tip/smoke)
#   make sb     - independent suite (verif/tip/sb): lint, vectors, 2 builds,
#                 7 run configurations, coverage aggregation gate, 4 mutants
# =============================================================================

.PHONY: all smoke sb clean

all: smoke sb

smoke:
	$(MAKE) -C smoke smoke

sb:
	$(MAKE) -C sb all

clean:
	$(MAKE) -C smoke clean
	$(MAKE) -C sb clean
