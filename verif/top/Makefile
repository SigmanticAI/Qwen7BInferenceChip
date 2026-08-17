# =============================================================================
# verif/top/Makefile — the tile verification stack:
#   smoke : three directed full-chain micro-cases (D=64 CQ-8, D=128 CQ-8,
#           D=64 CQ-4+ grouped keys + D-008 FLUSH + outlier mask — D-024)
#   l2    : Layer-2 pairwise chains (MXE->scored->ASU, KVQ->feeder->MXE all
#           3 tiers + FLUSH, scored->fork->{ASU,TIP}) — coverage-gated
#   l3    : Layer-3 named + random attention replays through the REAL
#           apex_top (CQ-8 / CQ-4+ / TIP-auto mixed tiers; feasibility
#           register F-1..F-5 in l3/, F-2 closed by D-024) —
#           coverage-gated + 3-mutant mutation gate
#
#   make all    - l2 + l3 (each is its own hard gate; smoke has its own)
#   make smoke / make l2 / make l3 / make clean
# =============================================================================

.PHONY: all smoke l2 l3 clean

all: l2 l3

smoke:
	$(MAKE) -C smoke smoke

l2:
	$(MAKE) -C l2 all

l3:
	$(MAKE) -C l3 all

clean:
	$(MAKE) -C smoke clean
	$(MAKE) -C l2 clean
	$(MAKE) -C l3 clean
