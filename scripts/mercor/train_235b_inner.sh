#!/bin/bash
set -ex
ulimit -n 65536

BASE_FOLDER="/data"
MASTER_ADDR="92.38.143.19"
WORKER_IPS=(92.38.143.26 85.234.91.98 85.234.91.76)

export no_proxy="127.0.0.1,${MASTER_ADDR}"
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_TC=136
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=ib0

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then export NCCL_NVLS_ENABLE=1; else export NCCL_NVLS_ENABLE=0; fi

export MODEL_ARGS_ROTARY_BASE=5000000

source /root/slime/scripts/models/qwen3-235B-A22B.sh

# Start ray head
RAY_memory_usage_threshold=0.98 ray start --head \
  --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Wait for workers
echo "Waiting for Ray workers to join..."
for i in $(seq 1 120); do
  NODE_COUNT=$(ray status 2>/dev/null | grep -c "node_" || echo 0)
  if [ "$NODE_COUNT" -ge 4 ]; then
    echo "All 4 nodes connected after $((i * 5)) seconds"
    break
  fi
  sleep 5
done

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\",
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
    \"NCCL_DEBUG\": \"WARN\",
    \"GLOO_SOCKET_IFNAME\": \"ib0\",
    \"WANDB_API_KEY\": \"<WANDB_API_KEY>\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir /root/slime \
   -- python3 train_async.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507 \
   --ref-load ${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507_torch_dist \
   --load ${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507_torch_dist/ \
   --save ${BASE_FOLDER}/Qwen3-235B-A22B-sft-v2-0314-unmasked-lr5e6/ \
   --save-interval 1000 \
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
   --prompt-data /data/apex_sft_opus_aug_plus_code_opt_zhanlin_v2_0314_unmasked.jsonl \
   --input-key messages \
   --rollout-shuffle \
   --num-epoch 5 \
   --rollout-batch-size 16 \
   --global-batch-size 16 \
   --loss-type sft_loss \
   --loss-mask-type qwen3 \
   --calculate-per-token-loss \
   --disable-compute-advantages-and-returns \
   --debug-train-only \
   --optimizer adam \
   --lr 5e-6 \
   --lr-decay-style cosine \
   --min-lr 5e-7 \
   --lr-warmup-fraction 0.1 \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --clip-grad 1 \
   --optimizer-cpu-offload \
   --overlap-cpu-optimizer-d2h-h2d \
   --use-precision-aware-optimizer \
   --use-wandb \
   --wandb-project mercor_sft_data_augmentation \
   --wandb-group qwen3-235B-A22B-sft-v2-0314-unmasked-lr5e6 \
   --wandb-key <WANDB_API_KEY> \
   --tensor-model-parallel-size 4 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 2 \
   --expert-model-parallel-size 32 \
   --expert-tensor-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 9216 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --no-save-optim \
   --attention-backend flash
