# SFT Training Runbook

## Prerequisites

- All nodes share `/data` via NFS
- Docker image `slimerl/slime:latest` on all nodes (`sudo docker images`)
- SSH access from master to all worker nodes (`ssh user@<worker_ip>`)
- InfiniBand connectivity (ib0 interface, 192.168.240.0/20 subnet)

### Current Cluster

| Role | IP |
|------|----|
| Master | 92.38.143.19 |
| Worker 1 | 92.38.143.26 |
| Worker 2 | 85.234.91.98 |
| Worker 3 | 85.234.91.76 |

---

## Training (30B and 235B)

Both models run inside `slimerl/slime:latest` docker containers to ensure all dependencies (sglang, megatron, etc.) are available.

### Architecture

1. Master container: starts ray head, submits training job
2. Worker containers (multi-node only): join ray cluster
3. All containers mount `/data` from NFS

### Docker Common Args

```bash
DOCKER_COMMON=(
  --gpus all --network host --ipc=host --shm-size=16g
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576
  --device /dev/infiniband
  -v /data:/data
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7
  -e NCCL_NET_GDR_LEVEL=2 -e NCCL_IB_TC=136
  -e NCCL_IB_TIMEOUT=22 -e NCCL_IB_RETRY_CNT=13
  -e NCCL_SOCKET_IFNAME=ib0 -e NCCL_DEBUG=WARN
  -e GLOO_SOCKET_IFNAME=ib0 -e NCCL_NVLS_ENABLE=1
)
```

### 30B (1 node, 8x H200)

```bash
# Cleanup
sudo docker rm -f slime-master 2>/dev/null

# Write inner script
cat > /data/train_30b_inner.sh << 'EOF'
#!/bin/bash
set -ex
export MODEL_ARGS_ROTARY_BASE=10000000
source /root/slime/scripts/models/qwen3-30B-A3B.sh
export no_proxy="127.0.0.1,127.0.0.1"
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265
sleep 10

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM/","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"1","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}}' \
  --working-dir /root/slime \
  -- python3 train_async.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/Qwen3-30B-A3B-Thinking-2507 \
  --ref-load /data/Qwen3-30B-A3B-Thinking-2507_torch_dist \
  --load /data/Qwen3-30B-A3B-Thinking-2507_torch_dist/ \
  --save /data/<SAVE_DIR>/ \
  --save-interval 92 \
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
  --prompt-data /data/<DATA>.jsonl \
  --input-key messages --rollout-shuffle \
  --num-epoch <EPOCHS> \
  --rollout-batch-size 16 --global-batch-size 16 \
  --loss-type sft_loss --loss-mask-type qwen3 \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns --debug-train-only \
  --optimizer adam --lr <LR> --lr-decay-style cosine --min-lr <MIN_LR> \
  --lr-warmup-fraction 0.1 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 \
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
  --use-wandb --wandb-project mercor_sft_data_augmentation \
  --wandb-group <WANDB_GROUP> \
  --wandb-key <WANDB_KEY> \
  --tensor-model-parallel-size 4 --sequence-parallel \
  --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 20480 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash
EOF

# Launch
sudo docker run --name slime-master \
  ${DOCKER_COMMON[@]} \
  -v /data/train_30b_inner.sh:/train_inner.sh:ro \
  slimerl/slime:latest bash /train_inner.sh
```

**30B parallelism:** tp=4, ep=8, cp=1, max_tokens_per_gpu=20480

### 235B (4 nodes, 32x H200)

```bash
# Cleanup
sudo docker rm -f slime-master 2>/dev/null
for ip in 92.38.143.26 85.234.91.98 85.234.91.76; do
  ssh user@$ip "sudo docker rm -f slime-worker 2>/dev/null" &
done; wait

# Start workers
for ip in 92.38.143.26 85.234.91.98 85.234.91.76; do
  ssh user@$ip "sudo docker run -d --name slime-worker \
    ${DOCKER_COMMON[@]} slimerl/slime:latest \
    ray start --address=92.38.143.19:6379 --num-gpus 8 \
      --node-ip-address $ip --disable-usage-stats --block" &
done

# Write inner script (same pattern as 30B but with 4-node args)
# Key differences:
#   --actor-num-nodes 4
#   MODEL_ARGS_ROTARY_BASE=5000000
#   --context-parallel-size 2
#   --expert-model-parallel-size 32
#   --max-tokens-per-gpu 9216
#   Wait for 4 nodes before submitting

# Launch master
sudo docker run --name slime-master \
  ${DOCKER_COMMON[@]} \
  -v /data/train_235b_inner.sh:/train_inner.sh:ro \
  slimerl/slime:latest bash /train_inner.sh
```

**235B parallelism:** tp=4, ep=32, cp=2, max_tokens_per_gpu=9216

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--prompt-data` | SFT data (.jsonl with `messages` key) |
| `--save` | Output checkpoint dir |
| `--lr` | Learning rate (typical: 5e-5, 2e-5, 1e-5, 5e-6) |
| `--num-epoch` | Epochs |
| `--global-batch-size` | Batch size |
| `--loss-mask-type qwen3` | Mask system/user turns |
| `MODEL_ARGS_ROTARY_BASE` | 10000000 for 30B, 5000000 for 235B |

### Model Args Source

- 30B: `/root/slime/scripts/models/qwen3-30B-A3B.sh` (inside container)
- 235B: `/root/slime/scripts/models/qwen3-235B-A22B.sh` (inside container)
- On NFS: `/data/mingye_b200-1/slime/scripts/models/`

---

## Convert Checkpoint to HF

```bash
SAVE_DIR="/data/<checkpoint_dir>"
LATEST=$(cat ${SAVE_DIR}/latest_checkpointed_iteration.txt)
ITER_DIR="${SAVE_DIR}/iter_$(printf '%07d' ${LATEST})"

python3 /data/mingye_b200-1/slime/tools/convert_torch_dist_to_hf.py \
  --input-dir "${ITER_DIR}" \
  --output-dir /data/<model>-hf \
  --origin-hf-dir /data/<base_model> \
  --vocab-size 151936
```

Base models:
- 30B: `/data/Qwen3-30B-A3B-Thinking-2507`
- 235B: `/data/Qwen3-235B-A22B-Thinking-2507`

---

## Eval

**Must use python 3.12** (`/data/eval_venv/bin/python`) for tool schema loading.

### 1. Start vLLM

```bash
/data/eval_venv/bin/vllm serve /data/<model>-hf \
  --trust-remote-code \
  --max-model-len 131072 \
  --reasoning-parser deepseek_r1 \
  --tensor-parallel-size <4 for 30B, 8 for 235B> \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching
```

### 2. Add Model Config

Add entry to `MODEL_CONFIGS` dict in `/data/eval_qwen3_local_with_snapshots_latest.py`:
```python
"<model_name>": {
    "api_type": "openai",
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model": "/data/<model>-hf",
    "api_key": "dummy",
    "extra_params": {},
    "context_window": 131072,
},
```

### 3. Verify Tool Schema

Check that tools load (should see 73 tools, not 0):
```bash
/data/eval_venv/bin/python /data/eval_qwen3_local_with_snapshots_latest.py \
  --model <model_name> --eval-dir /tmp/test_eval --dry-run
```

### 4. Run Eval

```bash
/data/eval_venv/bin/python /data/eval_qwen3_local_with_snapshots_latest.py \
  --model <model_name> \
  --eval-dir /data/apex_eval_<model_name> \
  --no-eval --resume
```

**Parallel eval** (30B, 2 vLLM instances on 4 GPUs each):
```bash
# Split tasks into shards, use --subset + --api-port
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve ... --port 8001 &
python eval... --subset shard0.txt --api-port 8000 &
python eval... --subset shard1.txt --api-port 8001 &
```

---

## Grading

```bash
/data/eval_venv/bin/python /data/grade_eval_results_latest.py \
  --model <model_name> \
  --eval-dir /data/apex_eval_<model_name> \
  --judge-model anthropic/claude-sonnet-4-5 \
  --judge-api-key "<key>" \
  --resume
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'sglang'` | Run inside docker, not bare metal |
| Ray dashboard connection refused | Wait longer, or install `pip install 'ray[default]'` on host |
| Tool schema empty in eval | Use python 3.12 (`eval_venv`), check `ARCHIPELAGO_ROOT` path |
| 400 context window errors loop | Already fixed in eval script (auto-skips) |
| Docker permission denied | Use `sudo docker` |

## Reference Scripts

- 30B example: `/data/run-qwen3-30B-A3B-sft.sh`
- 235B example: `/data/run-qwen3-235B-A22B-sft-v2-0314-unmasked.sh`
- Pipeline (train→convert→eval→grade): `/data/run-235b-sft-eval-pipeline.sh`
