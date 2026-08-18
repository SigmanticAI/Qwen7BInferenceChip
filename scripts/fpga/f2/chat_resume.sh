#!/bin/bash
# chat_resume.sh — resume a STOPPED chat card in ~5 minutes.
#
#   bash scripts/fpga/f2/chat_resume.sh <instance-id>
#
# Starts the stopped instance (disk already has code + weights), reloads
# the FPGA image + clock + DDR weights (FPGA state is volatile across
# stop/start — that is the ~5 min), and drops you back into the chat.
# On exit the card is STOPPED again (KEEP behaviour is sticky here);
# terminate for good with:
#   aws ec2 terminate-instances --region us-west-2 --instance-ids <id>
set -euo pipefail

IID="${1:?usage: chat_resume.sh <instance-id> (the id run_chat_demo.sh printed)}"
AGFI="${2:-agfi-030a812cd224b409d}"
REGION=us-west-2
KEY=~/.ssh/apex-f2.pem
SSH="ssh -n -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
SSHT="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
RCARD='/Users/nabilabdelazizferhattaleb/Desktop/apex-promptdemo'

STATE=$(aws ec2 describe-instances --region $REGION --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
if [ "$STATE" = "stopping" ]; then
  echo "== card is still shutting down — waiting for it to finish (~2 min)"
  aws ec2 wait instance-stopped --region $REGION --instance-ids "$IID"
fi
echo "== starting $IID"
aws ec2 start-instances --region $REGION --instance-ids "$IID" --output text >/dev/null
cleanup () {
  echo "== stopping $IID (resume again with this script)"
  aws ec2 stop-instances --region $REGION --instance-ids "$IID" --output text >/dev/null
}
trap cleanup EXIT
aws ec2 wait instance-running --region $REGION --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region $REGION --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "   up @ $IP"
for i in $(seq 1 30); do $SSH "ubuntu@${IP}" true 2>/dev/null && break; sleep 15; done

echo "== reloading FPGA image (the slot needs ~1 min after a cold start — retrying until ready)"
$SSH "ubuntu@${IP}" "source ~/aws-fpga/sdk_setup.sh >/dev/null 2>&1; \
  ok=0; for i in \$(seq 1 12); do \
    if sudo fpga-load-local-image -S 0 -I $AGFI >/dev/null 2>&1; then ok=1; break; fi; \
    echo \"   slot not ready yet (try \$i/12) — waiting 20s\"; sleep 20; \
  done; \
  [ \$ok = 1 ] || { echo 'FPGA slot never became ready — card left RUNNING for diagnosis'; exit 9; }; \
  echo '   image loaded'; \
  sudo fpga-load-clkgen-recipe -S 0 -a 2 >/dev/null && \
  sudo fpga-describe-clkgen -S 0 | grep -q 15.62 && echo '   clock verified (15.62)' && \
  sudo chmod 666 /sys/bus/pci/devices/0000:34:00.0/resource* /sys/bus/pci/devices/0000:34:00.1/resource* 2>/dev/null; \
  echo '   loading + verifying DDR weights (~1 min, quiet)'; \
  python3 '$RCARD/scripts/fpga/f2/f2_ddr_load.py' \
    --image '$RCARD/build/ddr_weights_05b_24L' --load --verify --full-verify 2>&1 | tail -1" || {
  rc=$?
  if [ "$rc" = "9" ]; then
    trap - EXIT
    echo "== NOT stopping the card (rc=9) — ssh in and check, or re-run this script"
    exit 9
  fi
  exit "$rc"
}

echo "== ensuring chat deps (self-heals cards stopped before deps landed)"
$SSH "ubuntu@${IP}" "sudo apt-get install -y -q python3-numpy python3-venv python3-pip >/dev/null 2>&1; \
  sudo python3 -m venv --system-site-packages /opt/apexchat 2>/dev/null || true; \
  sudo /opt/apexchat/bin/pip install -q --upgrade pip >/dev/null 2>&1; \
  sudo /opt/apexchat/bin/pip install -q transformers tokenizers 2>&1 | tail -1; \
  sudo /opt/apexchat/bin/python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(\"mlx-community/Qwen2.5-0.5B-Instruct-4bit\", allow_patterns=[\"tokenizer*\",\"*.json\"])' 2>&1 | tail -1; \
  sudo /opt/apexchat/bin/python3 -c 'import numpy, transformers' && echo '   deps ok'"

echo "== back in the chat (exit to stop the card again)"
$SSHT -t "ubuntu@${IP}" "sudo bash -c \"cd '$RCARD' && /opt/apexchat/bin/python3 scripts/fpga/f2/token_chat.py --engine hw-walked --ddr-attested\"" || true
