#!/bin/bash

# for rerun the task — clean up master and all worker containers
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

# stop any leftover worker containers on remote nodes
WORKER_IPS_CLEANUP=(85.234.91.242 85.234.91.90 92.38.143.214)
for _W in "${WORKER_IPS_CLEANUP[@]}"; do
  ssh -o StrictHostKeyChecking=no user@"${_W}" \
    "docker rm -f slime-worker 2>/dev/null; true" &
done
wait

set -ex

ulimit -n 65536

# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

# Default to the master node IP; can be overridden by env var
MASTER_ADDR="${MASTER_ADDR:-92.38.143.19}"
WORKER_IPS=(85.234.91.242 85.234.91.90 92.38.143.214)
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
export MODEL_ARGS_ROTARY_BASE=5000000

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# NCCL / InfiniBand settings for multi-node communication
# Adjust NCCL_SOCKET_IFNAME and NCCL_IB_HCA to match your network interfaces
# (run `ibstat` or `ip link` on the nodes to find the right names)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3          # RoCE v2; use 0 for RoCE v1 or native IB
export NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7
export NCCL_NET_GDR_LEVEL=2         # enable GPU Direct RDMA over IB
export NCCL_IB_TC=136               # traffic class for IB (DSCP EF)
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
# ib0 exists on all 4 nodes with IPs in the same 192.168.240.0/20 subnet
# using it for both NCCL socket fallback and Gloo to avoid per-node interface name differences
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN              # set to INFO for verbose NCCL logging
export GLOO_SOCKET_IFNAME=ib0

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/mingye_b200-1/slime/scripts/models/qwen3-235B-A22B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507
   --ref-load ${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507_torch_dist
   --load ${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507_least_turns_v2/
   --save ${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507_least_turns_v2/
   --save-interval 1000
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /data/apex_sft_least_turns_p95.jsonl
   --input-key messages
   # --apply-chat-template
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 128
   --global-batch-size 128

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
   --context-parallel-size 2
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
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
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-235B-sft
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export no_proxy="127.0.0.1,${MASTER_ADDR}"
RAY_memory_usage_threshold=0.98 ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
for WORKER_IP in "${WORKER_IPS[@]}"; do
  echo "Starting Ray worker container on ${WORKER_IP}"
  ssh -o StrictHostKeyChecking=no user@"${WORKER_IP}" \
    "docker run -d --name slime-worker \
      --gpus all --network host --ipc=host --shm-size=16g \
      --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
      --device /dev/infiniband \
      -v /data:/data \
      -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA=${NCCL_IB_HCA} \
      -e NCCL_NET_GDR_LEVEL=2 -e NCCL_IB_TC=136 -e NCCL_IB_TIMEOUT=22 -e NCCL_IB_RETRY_CNT=13 \
      -e NCCL_SOCKET_IFNAME=ib0 -e NCCL_DEBUG=${NCCL_DEBUG} \
      -e GLOO_SOCKET_IFNAME=ib0 \
      -e NCCL_NVLS_ENABLE=${HAS_NVLINK} \
      ${SLIME_IMAGE} \
      ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
        --node-ip-address ${WORKER_IP} --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port=8265 --block" &
done
wait


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"NCCL_IB_DISABLE\": \"0\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"NCCL_IB_HCA\": \"${NCCL_IB_HCA}\",
    \"NCCL_NET_GDR_LEVEL\": \"2\",
    \"NCCL_IB_TC\": \"136\",
    \"NCCL_IB_TIMEOUT\": \"22\",
    \"NCCL_IB_RETRY_CNT\": \"13\",
    \"NCCL_SOCKET_IFNAME\": \"ib0\",
    \"NCCL_DEBUG\": \"${NCCL_DEBUG}\",
    \"GLOO_SOCKET_IFNAME\": \"ib0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir /data/mingye_b200-1/slime \
   -- python3 train_async.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
