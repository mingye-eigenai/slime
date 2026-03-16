#!/bin/bash
#
# run-qwen3-30B-A3B-sft-pipeline.sh
#
# Parameterized single-node 30B SFT training script.
# Called inside Docker by run-sft-30B-domain-node.sh.
#
# Required env vars:
#   PROMPT_DATA      — path to training JSONL
#   CKPT_SAVE_DIR    — megatron checkpoint save/load dir
#   SAVE_HF_PATH     — path to write HF-format model at end
#   NUM_EPOCH        — number of epochs
#   SAVE_INTERVAL    — steps between checkpoints (≈ steps per epoch)
#   BASE_FOLDER      — base directory (default /data)

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

ulimit -n 65536

if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set"
  exit 1
fi

PROMPT_DATA="${PROMPT_DATA:?PROMPT_DATA must be set}"
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:?CKPT_SAVE_DIR must be set}"
SAVE_HF_PATH="${SAVE_HF_PATH:?SAVE_HF_PATH must be set}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
NUM_EPOCH="${NUM_EPOCH:-}"
if [ -z "${NUM_ROLLOUT}" ] && [ -z "${NUM_EPOCH}" ]; then
  echo "Either NUM_ROLLOUT or NUM_EPOCH must be set"
  exit 1
fi
SAVE_INTERVAL="${SAVE_INTERVAL:?SAVE_INTERVAL must be set}"
NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Single-node: Ray head on loopback
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export PYTHONBUFFERED=16
export MODEL_ARGS_ROTARY_BASE=10000000

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/mingye_b200-1/slime/scripts/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507
   --ref-load ${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507_torch_dist
   --load ${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507_torch_dist
   --save ${CKPT_SAVE_DIR}/
   --save-interval ${SAVE_INTERVAL}
)
# NOTE: do NOT use --save-hf. AutoBridge keeps padded vocab (152064).
# After training, convert manually with:
#   python tools/convert_torch_dist_to_hf.py \
#     --input-dir ${CKPT_SAVE_DIR}/iter_XXXXXXX \
#     --output-dir ${SAVE_HF_PATH} \
#     --origin-hf-dir ${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507 \
#     --vocab-size 151936
if [ "${NO_SAVE_OPTIM}" = "1" ]; then
  CKPT_ARGS+=(--no-save-optim)
  CKPT_ARGS+=(--finetune)
fi

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${PROMPT_DATA}
   --input-key messages
   --rollout-shuffle
   $([ -n "${NUM_ROLLOUT}" ] && echo "--num-rollout ${NUM_ROLLOUT}" || echo "--num-epoch ${NUM_EPOCH}")
   --rollout-batch-size ${BATCH_SIZE}
   --global-batch-size ${BATCH_SIZE}

   --loss-type sft_loss
   --loss-mask-type qwen3
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

LR="${LR:-3e-5}"
MIN_LR="${MIN_LR:-1e-6}"

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR}
   --lr-decay-style cosine
   --min-lr ${MIN_LR}
   --lr-warmup-fraction 0.15
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project qwen3-30B-sft
   --wandb-group ${WANDB_RUN_NAME:-sft}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export no_proxy="127.0.0.1,${MASTER_ADDR}"
RAY_memory_usage_threshold=0.98 ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir /data/mingye_b200-1/slime \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
