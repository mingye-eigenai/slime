#!/bin/bash
#
# launch-2node-format-reward.sh — 2-node non-colocate launcher
#
# Node 0 (local):  8 GPU training  (TP4 × CP2)
# Node 1 (remote): 8 GPU inference (4x TP2 SGLang engines)
#
# Usage:
#   sudo bash scripts/launch-2node-format-reward.sh
#

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "=== 2-Node APEX Docker Training Launcher (Format Reward 128K) ==="
echo "SLIME_ROOT: ${SLIME_ROOT}"

# ── Node Configuration ────────────────────────────────────────────────────────
# Node 0 (head/training): local machine, IB IP
NODE0_IB_IP="192.168.240.18"
# Node 1 (worker/inference): remote machine
NODE1_SSH="ubuntu@92.38.143.26"
NODE1_IB_IP="192.168.242.95"

# ── Docker / Training Configuration ──────────────────────────────────────────
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
export APEX_POOL_SIZE="${APEX_POOL_SIZE:-1}"
export APEX_BASE_PORT="${APEX_BASE_PORT:-9000}"
export APEX_DOCKER_CMD="${APEX_DOCKER_CMD:-docker}"
export APEX_SESSION_CONCURRENCY="${APEX_SESSION_CONCURRENCY:-128}"
export APEX_MCP_HOST="${APEX_MCP_HOST:-100.105.251.30}"
export APEX_MCP_PORT_MAP="${APEX_MCP_PORT_MAP:-/data/world_port_map_4container.json}"
export APEX_JUDGE_MODEL="${APEX_JUDGE_MODEL:-anthropic/claude-haiku-4-5-20251001}"
export APEX_JUDGE_API_BASE="${APEX_JUDGE_API_BASE:-}"
export APEX_JUDGE_API_KEY="${APEX_JUDGE_API_KEY:-<ANTHROPIC_API_KEY>}"
export APEX_GRADING_DIR="${APEX_GRADING_DIR:-/data/mingye_b200-1/archipelago/grading}"
export WANDB_KEY="${WANDB_KEY:-<WANDB_API_KEY>}"
export APEX_TRAIN_DATA="${APEX_TRAIN_DATA:-/data/apex_docker_tasks_law_filtered_v2.jsonl}"
APEX_ROLLOUT_BATCH_SIZE="${APEX_ROLLOUT_BATCH_SIZE:-32}"
APEX_N_SAMPLES="${APEX_N_SAMPLES:-8}"

# ── Verify prerequisites ─────────────────────────────────────────────────────
echo ""
echo "Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found on local node"
    exit 1
fi

# Check SSH to worker node
if ! ssh -o ConnectTimeout=5 ${NODE1_SSH} "echo ok" &>/dev/null; then
    echo "ERROR: Cannot SSH to worker node ${NODE1_SSH}"
    exit 1
fi
echo "  SSH to ${NODE1_SSH}: OK"

# Check IB connectivity
if ! ping -c 1 -W 2 ${NODE1_IB_IP} &>/dev/null; then
    echo "ERROR: Cannot reach worker IB IP ${NODE1_IB_IP}"
    exit 1
fi
echo "  IB to ${NODE1_IB_IP}: OK"

# ── Common Docker args ────────────────────────────────────────────────────────
DOCKER_COMMON=(
    --gpus all --ipc=host --shm-size=16g
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576
    --network host
    -v /var/run/docker.sock:/var/run/docker.sock
    -v /usr/bin/docker:/usr/bin/docker
    -v /data:/data
)

ENV_COMMON=(
    -e APEX_POOL_SIZE="${APEX_POOL_SIZE}"
    -e APEX_BASE_PORT="${APEX_BASE_PORT}"
    -e APEX_DOCKER_CMD="${APEX_DOCKER_CMD}"
    -e APEX_JUDGE_MODEL="${APEX_JUDGE_MODEL}"
    -e APEX_JUDGE_API_BASE="${APEX_JUDGE_API_BASE}"
    -e APEX_JUDGE_API_KEY="${APEX_JUDGE_API_KEY}"
    -e APEX_GRADING_DIR="${APEX_GRADING_DIR}"
    -e APEX_TRAIN_DATA="${APEX_TRAIN_DATA}"
    -e APEX_SESSION_CONCURRENCY="${APEX_SESSION_CONCURRENCY}"
    -e APEX_MCP_HOST="${APEX_MCP_HOST}"
    -e APEX_MCP_PORT_MAP="${APEX_MCP_PORT_MAP}"
    -e APEX_ROLLOUT_BATCH_SIZE="${APEX_ROLLOUT_BATCH_SIZE}"
    -e APEX_N_SAMPLES="${APEX_N_SAMPLES}"
    -e WANDB_KEY="${WANDB_KEY}"
)

# ── Step 1: Start worker node (Node 1) ───────────────────────────────────────
echo ""
echo "=== Starting worker node (Node 1: ${NODE1_SSH}) ==="
echo "  Role: Ray worker + SGLang inference (8 GPU)"

# Kill any existing slime containers on worker
ssh ${NODE1_SSH} "sudo docker kill slime-worker 2>/dev/null || true; sudo docker rm slime-worker 2>/dev/null || true"
sleep 2

# Start worker container in background via SSH (no --rm, named for easy log access)
ssh ${NODE1_SSH} "sudo docker run -d --name slime-worker \
    --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    -v /data:/data \
    -v /data/mingye_b200-1/slime:/root/slime \
    -e HEAD_ADDR=${NODE0_IB_IP} \
    -e WORKER_IB_IP=${NODE1_IB_IP} \
    ${ENV_COMMON[*]} \
    ${SLIME_IMAGE} \
    bash /root/slime/scripts/run-2node-inner-worker.sh"

echo "  Worker container started on ${NODE1_SSH} (name: slime-worker)"
echo "  Logs: ssh ${NODE1_SSH} 'sudo docker logs -f slime-worker'"

# ── Step 2: Start head node (Node 0, local) ──────────────────────────────────
echo ""
echo "=== Starting head node (Node 0: local) ==="
echo "  Role: Ray head + training (8 GPU, TP4×CP2)"
echo "  Context: 128K tokens"
echo ""

# Kill any existing slime containers on head
docker kill slime-head 2>/dev/null || true; docker rm slime-head 2>/dev/null || true
sleep 2

# Ensure save directory is world-writable (NFS root squash maps container root→nobody)
SAVE_DIR="/data/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113_rl_1_plus_1_conserv"
mkdir -p "${SAVE_DIR}"
chmod -R a+rwX "${SAVE_DIR}" 2>/dev/null || true

docker run -d --name slime-head \
    "${DOCKER_COMMON[@]}" \
    -v /data/mingye_b200-1/slime:/root/slime \
    -e MASTER_ADDR="${NODE0_IB_IP}" \
    "${ENV_COMMON[@]}" \
    "${SLIME_IMAGE}" \
    bash /root/slime/scripts/run-2node-inner-head-no-bonus.sh

echo "Head container started (name: slime-head)"
echo "Logs: sudo docker logs -f slime-head"

# ── Auto-record experiment ──────────────────────────────────────────────────
INNER_SCRIPT="/data/mingye_b200-1/slime/scripts/run-2node-inner-head-no-bonus.sh"
RECORD_PY="/data/experiments/record.py"
if [ -f "${RECORD_PY}" ] && [ -f "${INNER_SCRIPT}" ]; then
    EXP_ID=$(python3 "${RECORD_PY}" auto-launch \
        --script "${INNER_SCRIPT}" \
        --launcher "${BASH_SOURCE[0]}" \
        --head-container slime-head \
        --worker-ssh "${NODE1_SSH}" \
        2>/dev/null) || true
    if [ -n "${EXP_ID}" ]; then
        echo "Experiment recorded: ${EXP_ID}"
        echo "  Dashboard: file:///data/experiments/index.html"
    fi
fi
