#!/bin/bash
# run_token_demo.sh — watch the chip answer a prompt, one command.
#
#   cd ~/Desktop/apex-promptdemo
#   bash scripts/fpga/f2/run_token_demo.sh
#
# Boots an f2.6xlarge, loads the full-stack walked image
# (agfi-0500f4afe435b5e71), ships the code + weights (weights come from
# S3 at datacenter speed), full-verifies the DDR image, then runs the
# token loop with the hw-walked engine and STREAMS the run live:
# you watch every layer's walked chain grade BIT-EXACT on the silicon,
# then the generated tokens and the measured timing. Terminates the
# card automatically at the end (trap — even on ctrl-C).
# Cost ~ $1.50-2. Total ~ 25-35 min (most of it card boot + DDR verify).
set -euo pipefail

AGFI="${1:-agfi-0500f4afe435b5e71}"
TOKENS="${2:-3}"
REGION=us-west-2
AMI=ami-07fd0100b41f6c579
SG=sg-0766e253ceeaa3b74
KEY=~/.ssh/apex-f2.pem
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
RCARD='/Users/nabilabdelazizferhattaleb/Desktop/apex-promptdemo'
S3ASSETS=s3://apex-f2-dcp-099597653601/flight/tokloop_assets.tgz

echo "== [1/7] launching the card"
IID=$(aws ec2 run-instances --region $REGION --image-id $AMI \
  --instance-type f2.6xlarge --key-name apex-f2 --security-group-ids $SG \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=apex-token-demo}]' \
  --query 'Instances[0].InstanceId' --output text)
cleanup () {
  echo "== terminating $IID"
  aws ec2 terminate-instances --region $REGION --instance-ids "$IID" --output text >/dev/null
  aws ec2 wait instance-terminated --region $REGION --instance-ids "$IID" \
    && echo "   terminated (verified)"
}
trap cleanup EXIT
aws ec2 wait instance-running --region $REGION --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region $REGION --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "   $IID @ $IP"
for i in $(seq 1 30); do $SSH "ubuntu@${IP}" true 2>/dev/null && break; sleep 20; done

echo "== [2/7] FPGA image + clock + access"
$SSH "ubuntu@${IP}" "test -d ~/aws-fpga || git clone -q https://github.com/aws/aws-fpga.git ~/aws-fpga; \
  cd ~/aws-fpga && source sdk_setup.sh >/tmp/sdk.log 2>&1; \
  sudo fpga-load-local-image -S 0 -I $AGFI >/dev/null && \
  sudo fpga-load-clkgen-recipe -S 0 -a 2 >/dev/null && \
  sudo fpga-describe-clkgen -S 0 | grep -q 15.62 && echo '   image loaded, tile clock verified' && \
  sudo chmod 666 /sys/bus/pci/devices/0000:34:00.0/resource* /sys/bus/pci/devices/0000:34:00.1/resource* 2>/dev/null; \
  sudo mkdir -p '$RCARD' && sudo chown -R ubuntu:ubuntu /Users && \
  sudo apt-get install -y -q python3-numpy >/dev/null 2>&1; python3 -c 'import numpy' && echo '   numpy ok'"

echo "== [3/7] shipping the code"
CODE=$(mktemp -t apexcode.XXXX).tgz
COPYFILE_DISABLE=1 tar -czf "$CODE" scripts/fpga/f2 golden run_tinynpu.py \
  $(find verif -name "*.py" -not -path "*build*" -not -path "*__pycache__*")
scp -q -i $KEY -o StrictHostKeyChecking=no "$CODE" "ubuntu@${IP}:/tmp/code.tgz"
rm -f "$CODE"
$SSH "ubuntu@${IP}" "mkdir -p '$RCARD/build' && tar -xzf /tmp/code.tgz -C '$RCARD' 2>/dev/null; echo '   code staged'"

echo "== [4/7] weights from S3 (datacenter link)"
WURL=$(aws s3 presign "$S3ASSETS" --region $REGION --expires-in 3600)
$SSH "ubuntu@${IP}" "curl -s -f -o /tmp/assets.tgz '$WURL' && \
  tar -xzf /tmp/assets.tgz -C '$RCARD/build' 2>/dev/null; \
  ls '$RCARD/build/ddr_weights_05b_24L/ddr_image.bin' >/dev/null && echo '   weights + image staged'"

echo "== [5/7] loading + FULL-VERIFYING all 24 layers into DDR"
$SSH "ubuntu@${IP}" "source ~/aws-fpga/sdk_setup.sh >/dev/null 2>&1; \
  python3 '$RCARD/scripts/fpga/f2/f2_ddr_load.py' \
    --image '$RCARD/build/ddr_weights_05b_24L' --load --verify --full-verify 2>&1 | tail -2"

echo "== [6/7] THE RUN — watch the silicon walk every layer:"
echo "   (each line = one layer's walked chain, graded bit-exact live)"
$SSH -t "ubuntu@${IP}" "sudo bash -c \"cd '$RCARD' && python3 scripts/fpga/f2/token_loop.py run --engine hw-walked --tokens $TOKENS --ddr-attested\"" || true

echo "== [7/7] pulling the signed record"
scp -q -i $KEY -o StrictHostKeyChecking=no \
  "ubuntu@${IP}:$RCARD/build/token_loop/token_loop_hw-walked.json" \
  build/token_loop/token_loop_hw-walked.card.json 2>/dev/null \
  && echo "   record -> build/token_loop/token_loop_hw-walked.card.json"
echo "   (card terminates now)"
