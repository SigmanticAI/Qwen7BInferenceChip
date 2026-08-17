#!/bin/bash
# devbox_setup.sh — one-shot CL assembly + DCP build, runs ON the F2 dev box
# (FPGA Developer AMI Ubuntu, launched from the apex repo's f2 flow).
# Intended use from the local machine:
#   scp -i ~/.ssh/apex-f2.pem -r rtl scripts/fpga/f2 ubuntu@<ip>:~/apex_src/
#   ssh -i ~/.ssh/apex-f2.pem ubuntu@<ip> 'bash apex_src/f2/devbox_setup.sh'
# (scp lays rtl/ and f2/ side-by-side under ~/apex_src — this script expects
#  that exact layout and rebuilds the repo-relative paths cl_apex needs.)
set -euo pipefail

SRC="$HOME/apex_src"
[ -d "$SRC/rtl" ] && [ -d "$SRC/f2" ] || {
  echo "ABORT: expected $SRC/rtl and $SRC/f2 (scp per header)"; exit 1; }

# 0. Vivado env — the Ubuntu FPGA Developer AMI installs under
# /opt/Xilinx/<ver>/Vivado but non-login (nohup/ssh-command) shells don't
# source it; hdk_setup.sh hard-fails without vivado on PATH.
if ! command -v vivado >/dev/null 2>&1; then
  for s in /opt/Xilinx/*/Vivado/settings64.sh /tools/Xilinx/Vivado/*/settings64.sh; do
    if [ -f "$s" ]; then set +u; # shellcheck disable=SC1090
      source "$s"; set -u; break; fi
  done
fi
command -v vivado >/dev/null 2>&1 || { echo "ABORT: vivado not found"; exit 1; }

# 1. aws-fpga kit, pinned release
if [ ! -d "$HOME/aws-fpga" ]; then
  git clone --depth 1 --branch v2.3.3 https://github.com/aws/aws-fpga.git "$HOME/aws-fpga" \
    || git clone --depth 1 --branch f2 https://github.com/aws/aws-fpga.git "$HOME/aws-fpga"
fi
cd "$HOME/aws-fpga"
set +u; source hdk_setup.sh; set -u          # sets AWS_FPGA_REPO_DIR, Vivado env

# 2. assemble the APEX repo layout the synth tcl expects (APEX_REPO_DIR)
export APEX_REPO_DIR="$HOME/apex_repo"
mkdir -p "$APEX_REPO_DIR/scripts/fpga"
rm -rf "$APEX_REPO_DIR/rtl" "$APEX_REPO_DIR/scripts/fpga/f2"
cp -R "$SRC/rtl" "$APEX_REPO_DIR/rtl"
cp -R "$SRC/f2"  "$APEX_REPO_DIR/scripts/fpga/f2"

# 3. assemble cl_apex from CL_TEMPLATE + our design/build files
CLX="$AWS_FPGA_REPO_DIR/hdk/cl/examples"
export CL_DIR="$CLX/cl_apex"
rm -rf "$CL_DIR"
cp -R "$CLX/CL_TEMPLATE" "$CL_DIR"
# our design replaces the template's
rm -f "$CL_DIR"/design/*.sv "$CL_DIR"/design/*.vh 2>/dev/null || true
cp "$APEX_REPO_DIR"/scripts/fpga/f2/cl_apex/design/* "$CL_DIR/design/"
cp "$APEX_REPO_DIR"/scripts/fpga/f2/cl_apex/build/scripts/synth_cl_apex.tcl \
   "$CL_DIR/build/scripts/"
# DECISION-LC-1: our CDC constraints REPLACE the template's benign stub
# (ASYNC_REG + max_delay for apex_ocl_cdc; recipe clocks come from the kit)
if compgen -G "$APEX_REPO_DIR/scripts/fpga/f2/cl_apex/constraints/*.xdc" > /dev/null; then
  cp "$APEX_REPO_DIR"/scripts/fpga/f2/cl_apex/constraints/*.xdc \
     "$CL_DIR/build/constraints/"
fi

echo "== CL assembled at $CL_DIR; Vivado: $(command -v vivado || echo MISSING)"
vivado -version | head -1 || true

# 4. DCP build (30-90 min class) — log survives the session
# DECISION-LC-1: two-clock CL — --aws_clk_gen is a HARD GATE for the recipe
# flag; A2 puts clk_extra_a1 (the tile clock) at 15.625 MHz.
cd "$CL_DIR/build/scripts"
nohup python3 aws_build_dcp_from_cl.py -c cl_apex \
  --aws_clk_gen --clock_recipe_a A2 \
  > "$HOME/apex_dcp_build.log" 2>&1 &
echo "== DCP build launched (pid $!) — poll: tail -f ~/apex_dcp_build.log"
echo "== tarball lands in $CL_DIR/build/checkpoints/to_aws/"
