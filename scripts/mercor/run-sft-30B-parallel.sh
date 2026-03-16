#!/bin/bash
#
# run-sft-30B-parallel.sh
#
# Launches 4 independent 30B domain SFT jobs in parallel, one per node.
#
#   Node 1 (master, local): investment_banking
#   Node 2 (85.234.91.242): management_consulting
#   Node 3 (85.234.91.90):  law
#   Node 4 (92.38.143.214): mix
#
# Usage:
#   HF_TOKEN=hf_xxx BASE_FOLDER=/data bash /data/run-sft-30B-parallel.sh

set -euo pipefail

BASE_FOLDER="${BASE_FOLDER:-/data}"
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
HF_ORG="${HF_ORG:-eigen-ai-labs}"

WORKER_IPS=(85.234.91.242 85.234.91.90 92.38.143.214)
WORKER_DOMAINS=(management_consulting law mix)

# Launch workers via SSH in background
for i in "${!WORKER_IPS[@]}"; do
  IP="${WORKER_IPS[$i]}"
  DOMAIN="${WORKER_DOMAINS[$i]}"
  echo "=== Starting ${DOMAIN} on ${IP} ==="
  ssh -o StrictHostKeyChecking=no user@"${IP}" \
    "HF_TOKEN='${HF_TOKEN}' BASE_FOLDER='${BASE_FOLDER}' SLIME_IMAGE='${SLIME_IMAGE}' HF_ORG='${HF_ORG}' DOMAIN='${DOMAIN}' bash /data/run-sft-30B-domain-node.sh" &
done

# Run investment_banking locally on master node
echo "=== Starting investment_banking locally ==="
HF_TOKEN="${HF_TOKEN}" BASE_FOLDER="${BASE_FOLDER}" SLIME_IMAGE="${SLIME_IMAGE}" \
  HF_ORG="${HF_ORG}" DOMAIN=investment_banking \
  bash /data/run-sft-30B-domain-node.sh &

wait

echo ""
echo "=== All 4 domain nodes complete ==="
