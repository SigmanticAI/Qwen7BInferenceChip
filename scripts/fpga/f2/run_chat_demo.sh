#!/bin/bash
# run_chat_demo.sh — an INTERACTIVE prompt CLI on the real chip.
#
#   cd ~/Desktop/apex-promptdemo
#   bash scripts/fpga/f2/run_chat_demo.sh
#
# Boots the card, loads the walked full-stack image, stages weights
# (S3, datacenter link), full-verifies DDR, then drops YOU into
# `token_chat.py` over ssh: type a short prompt, the chip walks every
# layer, the answer comes back with the walked chains graded bit-exact.
# Type `exit` (or ctrl-C) to leave — the card terminates automatically.
# Envelope: prompt + answer <= 8 tokens (today's walker session fence).
# Cost ~$1.65/h while you chat; setup ~20-25 min.
set -euo pipefail

AGFI="${1:-agfi-0500f4afe435b5e71}"
KEEP="${KEEP:-0}"     # KEEP=1 -> STOP the card on exit instead of
                      # terminating: disk (code+weights) survives, resume
                      # with chat_resume.sh in ~5 min for ~$3/mo idle cost.
REGION=us-west-2
AMI=ami-07fd0100b41f6c579
SG=sg-0766e253ceeaa3b74
KEY=~/.ssh/apex-f2.pem
SSH="ssh -n -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
SSHT="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
RCARD='/Users/nabilabdelazizferhattaleb/Desktop/apex-promptdemo'
S3ASSETS=s3://apex-f2-dcp-099597653601/flight/tokloop_assets.tgz

echo "== [1/6] launching the card"
IID=$(aws ec2 run-instances --region $REGION --image-id $AMI \
  --instance-type f2.6xlarge --key-name apex-f2 --security-group-ids $SG \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=apex-chat-demo}]' \
  --query 'Instances[0].InstanceId' --output text)
cleanup () {
  if [ "$KEEP" = "1" ]; then
    echo "== KEEP=1: STOPPING $IID (disk survives; ~\$3/mo idle)"
    aws ec2 stop-instances --region $REGION --instance-ids "$IID" --output text >/dev/null
    echo "   resume later: bash scripts/fpga/f2/chat_resume.sh $IID"
  else
    echo "== terminating $IID"
    aws ec2 terminate-instances --region $REGION --instance-ids "$IID" --output text >/dev/null
    aws ec2 wait instance-terminated --region $REGION --instance-ids "$IID" \
      && echo "   terminated (verified)"
  fi
}
trap cleanup EXIT
aws ec2 wait instance-running --region $REGION --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region $REGION --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "   $IID @ $IP"
for i in $(seq 1 30); do $SSH "ubuntu@${IP}" true 2>/dev/null && break; sleep 20; done

echo "== [2/6] FPGA image + clock + python deps (numpy, transformers)"
$SSH "ubuntu@${IP}" "test -d ~/aws-fpga || git clone -q https://github.com/aws/aws-fpga.git ~/aws-fpga; \
  cd ~/aws-fpga && source sdk_setup.sh >/tmp/sdk.log 2>&1; \
  sudo fpga-load-local-image -S 0 -I $AGFI >/dev/null && \
  sudo fpga-load-clkgen-recipe -S 0 -a 2 >/dev/null && \
  sudo fpga-describe-clkgen -S 0 | grep -q 15.62 && echo '   image + clock ok' && \
  sudo chmod 666 /sys/bus/pci/devices/0000:34:00.0/resource* /sys/bus/pci/devices/0000:34:00.1/resource* 2>/dev/null; \
  sudo mkdir -p '$RCARD' && sudo chown -R ubuntu:ubuntu /Users; \
  sudo apt-get install -y -q python3-numpy python3-venv python3-pip >/dev/null 2>&1; \
  sudo python3 -m venv --system-site-packages /opt/apexchat 2>/dev/null || true; \
  sudo /opt/apexchat/bin/pip install -q --upgrade pip >/dev/null 2>&1; \
  sudo /opt/apexchat/bin/pip install -q transformers tokenizers 2>&1 | tail -1; \
  sudo /opt/apexchat/bin/python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(\"mlx-community/Qwen2.5-0.5B-Instruct-4bit\", allow_patterns=[\"tokenizer*\",\"*.json\"])' 2>&1 | tail -1; \
  sudo /opt/apexchat/bin/python3 -c 'import numpy, transformers' && echo '   deps + tokenizer ok'"

echo "== [3/6] shipping the code"
CODE=$(mktemp -t apexcode.XXXX).tgz
COPYFILE_DISABLE=1 tar -czf "$CODE" scripts/fpga/f2 golden run_tinynpu.py \
  $(find verif -name "*.py" -not -path "*build*" -not -path "*__pycache__*")
scp -q -i $KEY -o StrictHostKeyChecking=no "$CODE" "ubuntu@${IP}:/tmp/code.tgz"
rm -f "$CODE"
$SSH "ubuntu@${IP}" "mkdir -p '$RCARD/build' && tar -xzf /tmp/code.tgz -C '$RCARD' 2>/dev/null; echo '   code staged'"

echo "== [4/6] weights from S3"
WURL=$(aws s3 presign "$S3ASSETS" --region $REGION --expires-in 3600)
$SSH "ubuntu@${IP}" "curl -s -f -o /tmp/assets.tgz '$WURL' && \
  tar -xzf /tmp/assets.tgz -C '$RCARD/build' 2>/dev/null; \
  ls '$RCARD/build/ddr_weights_05b_24L/ddr_image.bin' >/dev/null && echo '   staged'"

echo "== [5/6] loading + FULL-VERIFYING all 24 layers into DDR"
$SSH "ubuntu@${IP}" "source ~/aws-fpga/sdk_setup.sh >/dev/null 2>&1; \
  python3 '$RCARD/scripts/fpga/f2/f2_ddr_load.py' \
    --image '$RCARD/build/ddr_weights_05b_24L' --load --verify --full-verify 2>&1 | tail -2"

echo "== [6/6] YOUR PROMPT — type away (exit to quit; card cleans up after)"
$SSHT -t "ubuntu@${IP}" "sudo bash -c \"cd '$RCARD' && /opt/apexchat/bin/python3 scripts/fpga/f2/token_chat.py --engine hw-walked --ddr-attested\"" || true
