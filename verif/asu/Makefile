# =============================================================================
# verif/asu/Makefile — everything ASU: the implementer's smoke plus the
# independent verification suite (scoreboard TBs + SVA + protocol adversaries
# + mid-op resets + coverage gate + mutation gate).
#
#   make all    - run all suites end to end (each gates on its own result)
#   make smoke  - implementer's 3-TB smoke (verif/asu/smoke)
#   make sb     - independent suite (verif/asu/sb)
#   make silu   - asu_silu exhaustive-domain check (Verilator + Icarus)
#   make wide   - WIDE-D RMSNorm suite (C-RMSW, verif/asu/wide)
# =============================================================================

.PHONY: all smoke sb silu wide clean

all: smoke sb silu wide

smoke:
	$(MAKE) -C smoke smoke

sb:
	$(MAKE) -C sb all

# asu_silu: exhaustive-domain bit-exact check (Verilator primary + Icarus
# secondary) vs the golden arbiter apex_golden.transformer.silu_fx (C-SILU).
silu:
	$(MAKE) -C silu silu

# WIDE-D RMSNorm (C-RMSW, docs/design/WIDE_RMSNORM.md): bit-exact vs the
# golden arbiter rmsnorm_fx_wide at RMS_D_MAX = 3584 and 8192.
wide:
	$(MAKE) -C wide all

clean:
	$(MAKE) -C smoke clean
	$(MAKE) -C sb clean
	$(MAKE) -C silu clean
	$(MAKE) -C wide clean
