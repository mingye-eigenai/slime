#!/bin/bash
#
# launch-3node-new.sh — 3-node non-colocate launcher
#
# Node 0 (local/head):  8 GPU training  (TP4 × CP2)
# Node 1 (159):         8 GPU inference (8x TP1 SGLang)
# Node 2 (246):         8 GPU inference (8x TP1 SGLang)
#

set -euo pipefail

echo "=== 3-Node APEX RL Training Launcher ==="
echo "  No KL loss, efficiency bonus coef=0.2, from scratch"

# ── Node Configuration ────────────────────────────────────────────────────────
NODE0_IB_IP="192.168.242.129"   # this node (head)
NODE1_SSH="mingye@192.168.240.159"
NODE1_IB_IP="192.168.240.159"
NODE2_SSH="mingye@192.168.240.246"
NODE2_IB_IP="192.168.240.246"

# ── Docker Configuration ─────────────────────────────────────────────────────
SLIME_IMAGE="slimerl/slime:latest"
export APEX_POOL_SIZE="${APEX_POOL_SIZE:-1}"
export APEX_BASE_PORT="${APEX_BASE_PORT:-9000}"
export APEX_DOCKER_CMD="${APEX_DOCKER_CMD:-docker}"
export APEX_SESSION_CONCURRENCY="${APEX_SESSION_CONCURRENCY:-128}"
export APEX_MCP_HOST="${APEX_MCP_HOST:-100.105.251.30}"
export APEX_MCP_PORT_MAP="${APEX_MCP_PORT_MAP:-/data/world_port_map_8replica_latest.json}"
export APEX_JUDGE_MODEL="${APEX_JUDGE_MODEL:-anthropic/claude-haiku-4-5-20251001}"
export APEX_JUDGE_API_BASE="${APEX_JUDGE_API_BASE:-}"
export APEX_JUDGE_API_KEY="${APEX_JUDGE_API_KEY:-<ANTHROPIC_API_KEY>}"
export APEX_GRADING_DIR="${APEX_GRADING_DIR:-/data/mingye_b200-1/archipelago/grading}"
export WANDB_KEY="${WANDB_KEY:-<WANDB_API_KEY>}"
export APEX_TRAIN_DATA="${APEX_TRAIN_DATA:-/data/apex_docker_tasks_law_filtered_v2.jsonl}"
APEX_ROLLOUT_BATCH_SIZE="${APEX_ROLLOUT_BATCH_SIZE:-16}"
APEX_N_SAMPLES="${APEX_N_SAMPLES:-16}"

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

# ── Step 0: Kill old containers on all nodes ─────────────────────────────────
echo ""
echo "=== Cleaning up all nodes ==="

# Local (this node)
echo "  Cleaning local node..."
sudo docker kill slime-head slime-worker 2>/dev/null || true
sudo docker rm slime-head slime-worker 2>/dev/null || true

# Node 1 (159)
echo "  Cleaning ${NODE1_SSH}..."
ssh ${NODE1_SSH} "sudo docker kill slime-head slime-worker 2>/dev/null; sudo docker rm slime-head slime-worker 2>/dev/null" 2>/dev/null || true

# Node 2 (246)
echo "  Cleaning ${NODE2_SSH}..."
ssh ${NODE2_SSH} "sudo docker kill slime-head slime-worker 2>/dev/null; sudo docker rm slime-head slime-worker 2>/dev/null" 2>/dev/null || true

sleep 5

# ── Step 1: Start worker node 1 (159) ────────────────────────────────────────
echo ""
echo "=== Starting worker node 1 (${NODE1_SSH}) ==="

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

echo "  Worker 1 started on ${NODE1_SSH}"

# ── Step 2: Start worker node 2 (246) ────────────────────────────────────────
echo ""
echo "=== Starting worker node 2 (${NODE2_SSH}) ==="

ssh ${NODE2_SSH} "sudo docker run -d --name slime-worker \
    --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    -v /data:/data \
    -v /data/mingye_b200-1/slime:/root/slime \
    -e HEAD_ADDR=${NODE0_IB_IP} \
    -e WORKER_IB_IP=${NODE2_IB_IP} \
    ${ENV_COMMON[*]} \
    ${SLIME_IMAGE} \
    bash /root/slime/scripts/run-2node-inner-worker.sh"

echo "  Worker 2 started on ${NODE2_SSH}"

# ── Step 3: Start head node (local) ──────────────────────────────────────────
echo ""
echo "=== Starting head node (local: ${NODE0_IB_IP}) ==="

SAVE_DIR="/data/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113_rl_3node_no_kl_effbonus0.2"
mkdir -p "${SAVE_DIR}"
chmod -R a+rwX "${SAVE_DIR}" 2>/dev/null || true

sudo docker run -d --name slime-head \
    --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /data:/data \
    -v /data/mingye_b200-1/slime:/root/slime \
    -e MASTER_ADDR="${NODE0_IB_IP}" \
    "${ENV_COMMON[@]}" \
    "${SLIME_IMAGE}" \
    bash /data/run-3node-inner-head-new.sh

echo ""
echo "=== All containers started ==="
echo "  Head logs:     sudo docker logs -f slime-head"
echo "  Worker 1 logs: ssh ${NODE1_SSH} 'sudo docker logs -f slime-worker'"
echo "  Worker 2 logs: ssh ${NODE2_SSH} 'sudo docker logs -f slime-worker'"
