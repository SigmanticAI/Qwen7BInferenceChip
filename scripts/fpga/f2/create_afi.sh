#!/bin/bash
# create_afi.sh — DCP tarball → AFI/AGFI (S15 prep).
# ⏸ VALIDATION-DEFERRED: never executed. Needs AWS credentials + an S3
# bucket IN THE SAME REGION as the EC2 API endpoint. Owner approves spend
# first (ingestion itself is free; S3 pennies). See BRINGUP.md §3 step 5.
set -euo pipefail

: "${AWS_FPGA_REPO_DIR:?source aws-fpga/hdk_setup.sh first}"
TARBALL="${1:?usage: create_afi.sh <path/to/*.Developer_CL.tar> [name]}"
NAME="${2:-apex-tile-$(date +%Y%m%d)}"
: "${S3_BUCKET:?export S3_BUCKET=<same-region bucket for DCPs + AFI logs>}"
: "${AWS_REGION:?export AWS_REGION=<F2 region, e.g. us-east-1>}"

# Preferred path: the kit's own wrapper (uploads, calls create-fpga-image,
# polls every 300 s, prints AFI + AGFI ids):
python3 "$AWS_FPGA_REPO_DIR/hdk/scripts/create_afi.py" \
  --tarball "$TARBALL" --name "$NAME" \
  --bucket "$S3_BUCKET" --region "$AWS_REGION" \
  2>&1 | tee "afi_${NAME}.log"

# Fallback (manual) if the wrapper's interface drifted from v2.3.3 docs:
#   aws s3 cp "$TARBALL" "s3://$S3_BUCKET/dcp/"
#   aws ec2 create-fpga-image --region "$AWS_REGION" --name "$NAME" \
#     --input-storage-location Bucket="$S3_BUCKET",Key="dcp/$(basename "$TARBALL")" \
#     --logs-storage-location  Bucket="$S3_BUCKET",Key="afi-logs/"
#   aws ec2 describe-fpga-images --fpga-image-ids <afi-...>   # poll → available
# Ingestion "can take hours depending on design size" (kit docs). Record the
# AGFI id in this log — fpga-load-local-image takes the AGFI, not the AFI.
