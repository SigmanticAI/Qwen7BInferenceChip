# =============================================================================
# verif/kvq/Makefile — everything KVQ (ARCHITECTURE.md §1 KVQ row, D-015/D-016/
# D-008/D-009):
#
#   make smoke  - implementer suite: lint + python-golden gen + 8 parity/stall
#                 configs + 4 reproduce-then-fix regressions (verif/kvq/smoke)
#   make sb     - INDEPENDENT verification suite (verif/kvq/sb): vector-driven
#                 scoreboard vs golden/apex_golden, bound SVA pack, adversarial
#                 protocol + mid-op reset runs, coverage gate, mutation checks.
#                 NOTE: currently exits non-zero — real DUT findings, see
#                 verif/kvq/RESULT.md (B-1..B-4).
#   make fparith- synthesizable integer fp-arith proof (verif/kvq/fparith):
#                 the cq_fp_pkg real->integer rewrite gate — exhaustive golden
#                 sweeps, 2 seeded-error checks, sv2v+yosys synthesizability.
#   make all    - all of the above
# =============================================================================

.PHONY: all smoke sb fparith clean

all: smoke sb fparith

smoke:
	$(MAKE) -C smoke smoke

sb:
	$(MAKE) -C sb all

fparith:
	$(MAKE) -C fparith all

clean:
	$(MAKE) -C smoke clean
	$(MAKE) -C sb clean
	$(MAKE) -C fparith clean
