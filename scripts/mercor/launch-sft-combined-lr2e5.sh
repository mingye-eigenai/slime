#!/bin/bash
#
# Launch lr=2e-5 training on worker node (92.38.143.26) via SSH
#
# Usage:
#   bash /data/launch-sft-combined-lr2e5.sh
#

set -euo pipefail

SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
BASE_FOLDER="${BASE_FOLDER:-/data}"
WORKER_SSH="ubuntu@92.38.143.26"

echo "=== Launching Qwen3-30B SFT v2 sequential (lr=2e-5, 7 epochs, bs=16) on worker node ==="

# Check SSH connectivity
if ! ssh -o ConnectTimeout=5 ${WORKER_SSH} "echo ok" &>/dev/null; then
    echo "ERROR: Cannot SSH to worker node ${WORKER_SSH}"
    exit 1
fi
echo "  SSH to ${WORKER_SSH}: OK"

# Kill any existing slime containers on worker
ssh ${WORKER_SSH} "sudo docker kill slime-sft-lr2e5 2>/dev/null || true; sudo docker rm slime-sft-lr2e5 2>/dev/null || true"
sleep 2

# Launch training on worker node
ssh ${WORKER_SSH} "sudo docker run -d --name slime-sft-lr2e5 \
    --gpus all --ipc=host --shm-size=16g \
    --userns=host \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    -v /data:/data \
    -e BASE_FOLDER=${BASE_FOLDER} \
    -e MASTER_ADDR=127.0.0.1 \
-e HF_TOKEN=<HF_TOKEN> \
    ${SLIME_IMAGE} \
    bash /data/run-sft-v2-sequential-lr2e5.sh"

echo "Worker container started on ${WORKER_SSH} (name: slime-sft-lr2e5)"
echo "Logs: ssh ${WORKER_SSH} 'sudo docker logs -f slime-sft-lr2e5'"
