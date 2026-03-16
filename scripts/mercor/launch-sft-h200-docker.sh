#!/bin/bash
#
# launch-sft-h200-docker.sh — Run on the master node (92.38.143.19) to launch
# the Qwen3-235B SFT training inside the slime Docker container.
#
# Usage:
#   BASE_FOLDER=/data bash /data/launch-sft-h200-docker.sh

set -euo pipefail

SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
BASE_FOLDER="${BASE_FOLDER:-/data}"
MASTER_ADDR="${MASTER_ADDR:-92.38.143.19}"

echo "=== Pulling ${SLIME_IMAGE} ==="
sudo docker pull "${SLIME_IMAGE}"

echo "=== Launching training container ==="
exec sudo docker run --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    --device /dev/infiniband \
    -v /data:/data \
    -v /home/user/.ssh:/root/.ssh:ro \
    -e BASE_FOLDER="${BASE_FOLDER}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e SLIME_IMAGE="${SLIME_IMAGE}" \
    "${SLIME_IMAGE}" \
    bash /data/run-qwen3-235B-A22B-sft-h200-4nodes.sh
