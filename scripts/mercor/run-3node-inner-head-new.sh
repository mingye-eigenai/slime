#!/bin/bash
#
# run-3node-inner-head-new.sh — 3-Node Non-Colocate head script
#   Node 0 (head/this): 8 GPU training  — TP4 × CP2
#   Node 1 (worker): 8 GPU inference — 8x TP1 SGLang
#   Node 2 (worker): 8 GPU inference — 8x TP1 SGLang
#
# Changes from previous run:
#   - KL loss coef: 0 (removed constraint)
#   - Efficiency bonus coef: 0.2 (was 0.1)
#   - Training from scratch (load from SFT ref checkpoint)

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
SCRIPT_DIR="/root/slime/scripts"
MODEL_DIR="${MODEL_DIR:-/data}"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

APEX_TRAIN_DATA="${APEX_TRAIN_DATA:-/data/apex_docker_tasks_law_filtered_v2.jsonl}"

# ── 3-Node Non-Colocate Config ───────────────────────────────────────────────
SAVE_DIR="${MODEL_DIR}/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113_rl_3node_no_kl_effbonus0.2"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B-Thinking-2507
   --ref-load ${MODEL_DIR}/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113-ref
   --load ${MODEL_DIR}/Qwen3-30B-A3B-sft-opus-mix-lr5e5-iter113-ref
   --save ${SAVE_DIR}/
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
   --rollout-batch-size ${APEX_ROLLOUT_BATCH_SIZE:-16}
   --n-samples-per-prompt ${APEX_N_SAMPLES:-16}
   --rollout-max-response-len 131072
   --rollout-temperature 1.0

   --over-sampling-batch-size ${APEX_ROLLOUT_BATCH_SIZE:-16}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size ${APEX_GLOBAL_BATCH_SIZE:-256}
   --balance-data
   --efficiency-bonus-coef 0.2
)

PERF_ARGS=(
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
   --kl-coef 0
   --use-kl-loss
   --kl-loss-coef 0.0
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

CUSTOM_ARGS=(
   --custom-config-path /root/slime/scripts/mis_mask.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
   --get-mismatch-metrics
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
   --wandb-group qwen3-30B-3node-no-kl-effbonus0.2
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus 16
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
export MASTER_ADDR=${MASTER_ADDR:-"192.168.242.129"}

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
echo "Waiting for 2 worker nodes to join..."

# Wait for 2 worker nodes (24 GPUs total)
for i in $(seq 1 120); do
    GPU_COUNT=$(ray status 2>/dev/null | grep -oP '\d+\.\d+ GPU' | head -1 | grep -oP '^\d+' || echo 0)
    echo "  Attempt $i: ${GPU_COUNT} GPUs available"
    if [ "$GPU_COUNT" -ge 24 ]; then
        echo "Both worker nodes joined! Total GPUs: ${GPU_COUNT}"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Worker nodes did not join after 10 minutes"
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
    \"APEX_MCP_PORT_MAP\": \"${APEX_MCP_PORT_MAP:-/data/world_port_map_8replica_latest.json}\",
    \"APEX_INCOMPLETE_PENALTY\": \"${APEX_INCOMPLETE_PENALTY:-0.5}\",
    \"APEX_ROLLOUT_DATA_DIR\": \"/data/apex_rollout_data/rl_3node_no_kl_effbonus0.2\"
  }
}"

RAY_ADDRESS="http://127.0.0.1:8267"

# Submit job
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
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} 2>&1)

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
