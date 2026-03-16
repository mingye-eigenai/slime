#!/bin/bash
#
# Launch lr=5e-5 training on local node (this node)
#
# Usage:
#   bash /data/launch-sft-combined-lr5e5.sh
#

set -euo pipefail

SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
BASE_FOLDER="${BASE_FOLDER:-/data}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

echo "=== Launching Qwen3-30B SFT v2 sequential (lr=5e-5, 7 epochs, bs=16) on local node ==="

# Kill any existing slime containers
sudo docker kill slime-sft-lr5e5 2>/dev/null || true
sudo docker rm slime-sft-lr5e5 2>/dev/null || true
sleep 2

sudo docker run -d --gpus all --ipc=host --shm-size=16g \
    --userns=host \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    --name slime-sft-lr5e5 \
    -v /data:/data \
    -e BASE_FOLDER="${BASE_FOLDER}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
-e HF_TOKEN="<HF_TOKEN>" \
    "${SLIME_IMAGE}" \
    bash /data/run-sft-v2-sequential-lr5e5.sh

echo "Local container started (name: slime-sft-lr5e5)"
echo "Logs: sudo docker logs -f slime-sft-lr5e5"
