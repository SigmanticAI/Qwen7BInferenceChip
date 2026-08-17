#!/bin/bash
# f2_smoke_session.sh — runs ON the f2.6xlarge: SDK setup, AFI load, BAR0 smoke.
set -uo pipefail
AGFI=agfi-0f7c93ffa798ecc3f
log(){ echo "[$(date +%H:%M:%S)] $*"; }
log "== F2 SMOKE SESSION start"
if [ ! -d "$HOME/aws-fpga" ]; then
  git clone --depth 1 --branch f2 https://github.com/aws/aws-fpga.git "$HOME/aws-fpga" 2>&1 | tail -1
fi
cd "$HOME/aws-fpga"
set +u; source sdk_setup.sh >/tmp/sdk_setup.log 2>&1; rc=$?; set -u
log "sdk_setup rc=$rc ($(command -v fpga-load-local-image || echo 'NO mgmt tools'))"
sudo fpga-describe-local-image-slots -H 2>&1 | head -4
log "loading $AGFI into slot 0"
sudo fpga-load-local-image -S 0 -I "$AGFI" 2>&1 | tail -3
sudo fpga-describe-local-image -S 0 -H 2>&1 | head -6
# C probe: bridge ID / scratch RW / KVQ INFO / tile CSR STATUS via BAR0
cat > /tmp/probe.c <<'EOC'
#include <stdio.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>
static int rd(pci_bar_handle_t h, uint64_t a, uint32_t *v){int r=fpga_pci_peek(h,a,v);printf("  peek 0x%04lx = 0x%08x%s\n",a,*v,r?" (ERR)":"");return r;}
int main(void){
  uint32_t v; int fails=0;
  if (fpga_mgmt_init()) { printf("mgmt_init FAIL\n"); return 2; }
  pci_bar_handle_t h = PCI_BAR_HANDLE_INIT;
  if (fpga_pci_attach(0, FPGA_APP_PF, APP_PF_BAR0, 0, &h)) { printf("attach FAIL\n"); return 2; }
  printf("BAR0 attached\n");
  rd(h,0x0000,&v); if(v!=0x41394558){printf("  ^ FAIL want A9EX 0x41394558\n");fails++;}
  fpga_pci_poke(h,0x0008,0xA5A55A5A); rd(h,0x0008,&v); if(v!=0xA5A55A5A){printf("  ^ FAIL scratch\n");fails++;}
  fpga_pci_poke(h,0x0008,0x5A5AA5A5); rd(h,0x0008,&v); if(v!=0x5A5AA5A5){printf("  ^ FAIL scratch2\n");fails++;}
  printf(" KVQ INFO window:\n");
  rd(h,0x2008,&v); rd(h,0x200C,&v); rd(h,0x2010,&v); rd(h,0x2020,&v);
  printf(" tile CSR window:\n");
  rd(h,0x1000,&v); rd(h,0x1004,&v);
  printf(" bridge status:\n");
  rd(h,0x0010,&v); rd(h,0x0018,&v);
  printf(fails?"PROBE: %d FAILS\n":"PROBE: ALL PASS\n",fails);
  return fails?1:0;
}
EOC
gcc -I sdk/userspace/include /tmp/probe.c -o /tmp/probe -lfpga_mgmt 2>&1 | head -3
log "running probe"
sudo /tmp/probe
rc=$?
log "== F2 SMOKE SESSION done probe_rc=$rc"
exit $rc
