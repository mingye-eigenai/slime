#!/bin/bash
#
# run-2node-inner-head.sh — Inner script for Node 0 (head): Ray head + training job
# Runs INSIDE the Docker container on the head node.
#

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
rm -rf /tmp/ray

set -ex

# ── Install missing deps ─────────────────────────────────────────────────────
pip install fastmcp requests 2>/dev/null || true

# ── Patch SGLang 0.5.9 bug ───────────────────────────────────────────────────
SGLANG_TM="/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py"
if [ -f "$SGLANG_TM" ]; then
    sed -i 's/await self\.send_to_scheduler\.send_pyobj(obj)/self.send_to_scheduler.send_pyobj(obj)/g' "$SGLANG_TM"
    echo "[PATCH] Fixed SGLang tokenizer_manager.py"
fi

export PYTHONBUFFERED=16
export WANDB_KEY=${WANDB_KEY:-"<WANDB_API_KEY>"}

# ── APEX Docker Configuration ────────────────────────────────────────────────
export APEX_POOL_SIZE=${APEX_POOL_SIZE:-1}
export APEX_BASE_PORT=${APEX_BASE_PORT:-9000}
export APEX_DOCKER_CMD=${APEX_DOCKER_CMD:-"docker"}
export APEX_JUDGE_MODEL=${APEX_JUDGE_MODEL:-"anthropic/claude-haiku-4-5-20251001"}
export APEX_JUDGE_API_BASE=${APEX_JUDGE_API_BASE:-""}
export APEX_JUDGE_API_KEY=${APEX_JUDGE_API_KEY:-"<ANTHROPIC_API_KEY>"}
export APEX_GRADING_DIR=${APEX_GRADING_DIR:-"/data/mingye_b200-1/archipelago/grading"}

# ── NVLink detection ──────────────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -co "NV[0-9]" || echo 0)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_DIR="${MODEL_DIR:-/data}"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

APEX_TRAIN_DATA="${APEX_TRAIN_DATA:-/data/apex_docker_tasks_law_filtered_v2.jsonl}"

# ── 2-Node Non-Colocate Config ───────────────────────────────────────────────
# Node 0 (head): 8 GPU for training  — TP4 × CP2 × EP1 = 8
# Node 1 (worker): 8 GPU for inference — 4x TP2 SGLang engines

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B-Thinking-2507
   --ref-load ${MODEL_DIR}/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113-ref
   --load ${MODEL_DIR}/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113_rl_1_plus_1_conserv/
   --save ${MODEL_DIR}/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113_rl_1_plus_1_conserv/
   --save-interval 20
   --no-load-optim
)

ROLLOUT_ARGS=(
   --prompt-data ${APEX_TRAIN_DATA}
   --input-key messages
   --rollout-shuffle

   --custom-generate-function-path slime.rollout.apex_docker_rollout.generate_with_apex_docker
   --custom-rm-path slime.rollout.rm_hub.archipelago.compute_archipelago_reward

   --mcp-max-steps ${APEX_MCP_MAX_STEPS:-20}
   --mcp-save-rollouts
   --mcp-max-tool-result-chars ${APEX_MAX_TOOL_RESULT_CHARS:-20000}
   --mcp-max-context-tokens ${APEX_MAX_CONTEXT_TOKENS:-96000}

   --num-rollout 1000
   --rollout-batch-size ${APEX_ROLLOUT_BATCH_SIZE:-32}
   --n-samples-per-prompt ${APEX_N_SAMPLES:-8}
   --rollout-max-response-len 131072
   --rollout-temperature 1.0

   --over-sampling-batch-size ${APEX_ROLLOUT_BATCH_SIZE:-32}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size ${APEX_GLOBAL_BATCH_SIZE:-256}
   --balance-data
)

PERF_ARGS=(
   # Training parallelism: TP4 × CP2 = 8 GPUs
   --tensor-model-parallel-size 4
   --context-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 4

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 65536
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --use-kl-loss
   --kl-loss-coef 0.001
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project mercor_rl
   --wandb-group qwen3-30b-law-only-1plus1-trial
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   # 8 rollout GPUs on node 1, 8x TP1 engines
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.85
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-server-concurrency 64
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ── Launch Ray Head ───────────────────────────────────────────────────────────
# MASTER_ADDR should be the IB IP for fast NCCL communication
export MASTER_ADDR=${MASTER_ADDR:-"192.168.240.159"}

ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/
ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus 8 \
    --disable-usage-stats \
    --port 6381 \
    --dashboard-port 8267 \
    --dashboard-agent-listen-port 52400

echo "Ray head started at ${MASTER_ADDR}:6381"
echo "Waiting for worker node to join..."

# Wait for worker node (16 GPUs total = head 8 + worker 8)
for i in $(seq 1 60); do
    GPU_COUNT=$(ray status 2>/dev/null | grep -oP '\d+\.\d+ GPU' | head -1 | grep -oP '^\d+' || echo 0)
    echo "  Attempt $i: ${GPU_COUNT} GPUs available"
    if [ "$GPU_COUNT" -ge 16 ]; then
        echo "Worker node joined! Total GPUs: ${GPU_COUNT}"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Worker node did not join after 5 minutes"
        ray status
        exit 1
    fi
    sleep 5
done

# Build runtime environment
EXTRA_PYTHONPATH="/data/mingye_b200-1:/root/slime/container_overlay"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXTRA_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"0\",
    \"NCCL_SOCKET_IFNAME\": \"ib0,ib1\",
    \"APEX_POOL_SIZE\": \"${APEX_POOL_SIZE}\",
    \"APEX_BASE_PORT\": \"${APEX_BASE_PORT}\",
    \"APEX_DOCKER_CMD\": \"${APEX_DOCKER_CMD}\",
    \"APEX_JUDGE_MODEL\": \"${APEX_JUDGE_MODEL}\",
    \"APEX_JUDGE_API_BASE\": \"${APEX_JUDGE_API_BASE}\",
    \"APEX_JUDGE_API_KEY\": \"${APEX_JUDGE_API_KEY}\",
    \"APEX_GRADING_DIR\": \"${APEX_GRADING_DIR}\",
    \"APEX_SESSION_CONCURRENCY\": \"${APEX_SESSION_CONCURRENCY:-128}\",
    \"APEX_MCP_HOST\": \"${APEX_MCP_HOST:-100.105.251.30}\",
    \"APEX_MCP_PORT_MAP\": \"${APEX_MCP_PORT_MAP:-/data/world_port_map_4container.json}\",
    \"APEX_ROLLOUT_DATA_DIR\": \"/data/apex_rollout_data/1_plus_1_conserv\",
    \"APEX_INCOMPLETE_PENALTY\": \"${APEX_INCOMPLETE_PENALTY:-0.1}\"
  }
}"

RAY_ADDRESS="http://127.0.0.1:8267"

# Submit job — NON-colocate: 8 train GPUs + 8 rollout GPUs
SUBMIT_OUTPUT=$(ray job submit --address="${RAY_ADDRESS}" --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/slime/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} 2>&1)

echo "${SUBMIT_OUTPUT}"
JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'raysubmit_\w+' | head -1)
echo "Submitted job: ${JOB_ID}"

# Poll until completion
while true; do
    STATUS=$(ray job status "${JOB_ID}" --address="${RAY_ADDRESS}" 2>&1)
    if echo "${STATUS}" | grep -qiE "SUCCEEDED|FAILED|STOPPED"; then
        echo "Job ${JOB_ID} finished:"
        echo "${STATUS}"
        ray job logs "${JOB_ID}" --address="${RAY_ADDRESS}" 2>/dev/null | tail -50
        break
    fi
    sleep 30
done
