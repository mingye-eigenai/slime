#!/bin/bash
#
# SFT training: Qwen3-30B-A3B with combined augmented data
# lr=5e-5, 5 epochs, batch_size=16
# Run on local node inside slime docker container
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

set -ex

# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
  export MASTER_ADDR="127.0.0.1"
fi

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export MODEL_ARGS_ROTARY_BASE=10000000
source /root/slime/scripts/models/qwen3-30B-A3B.sh

SAVE_NAME="Qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507
   --ref-load ${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507_torch_dist
   --load ${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507_torch_dist/
   --save ${BASE_FOLDER}/${SAVE_NAME}/
   --save-interval 92
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /data/apex_sft_opus_aug_plus_code_opt_zhanlin_v2_0314.jsonl
   --input-key messages
   --rollout-shuffle
   --num-epoch 7
   --rollout-batch-size 16
   --global-batch-size 16

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

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-5
   --lr-decay-style cosine
   --min-lr 5e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project mercor_sft_data_augmentation
   --wandb-group qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5
   --wandb-key "${WANDB_KEY:-<WANDB_API_KEY>}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# launch the master node of ray in container
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_API_KEY\": \"${WANDB_KEY:-<WANDB_API_KEY>}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir /root/slime \
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
