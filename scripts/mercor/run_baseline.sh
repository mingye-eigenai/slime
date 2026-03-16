#!/bin/bash
set -e
export PATH="$HOME/.local/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH"

MODEL_NAME="qwen3_235b_base"
MODEL_PATH="/data/Qwen3-235B-A22B-Thinking-2507"
EVAL_DIR="/data/apex_eval_235b_base"
EVAL_SCRIPT="/data/eval_qwen3_local_with_snapshots_latest.py"
DATA_ROOT="/data/apex-agents"
TOOL_ROOT="/data/mingye_b200-1/archipelago/mcp_servers"
GRADE_SCRIPT="/data/grade_eval_results_latest.py"
VLLM="/data/eval_venv/bin/vllm"
PYTHON="/data/eval_venv/bin/python"
PORT=8000

echo "$(date): === Baseline Auto-Runner ==="

# ── Step 1: Wait for current eval to finish ──
echo "$(date): Waiting for current eval batches to finish..."
while true; do
    EVAL_PROCS=$(ps aux | grep "eval_qwen3_local_with_snapshots_latest.py.*qwen3_235b_opus_sonnet_best_v2_lr5e6" | grep -v grep | wc -l)
    if [ "$EVAL_PROCS" -eq 0 ]; then
        echo "$(date): All eval batches finished."
        break
    fi
    echo "$(date): $EVAL_PROCS eval processes still running. Waiting 60s..."
    sleep 60
done

# ── Step 2: Wait for current grading to finish ──
echo "$(date): Waiting for grading to finish..."
while true; do
    GRADE_PROCS=$(ps aux | grep "grade_eval_results_latest.py" | grep -v grep | wc -l)
    if [ "$GRADE_PROCS" -eq 0 ]; then
        echo "$(date): Grading finished."
        break
    fi
    echo "$(date): Grading still running. Waiting 30s..."
    sleep 30
done

# ── Step 3: Kill vLLM ──
echo "$(date): Killing vLLM server..."
pkill -f "vllm serve" || true
sleep 10
# Make sure it's dead
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5

# ── Step 4: Start vLLM with base model ──
echo "$(date): Starting vLLM with base model: $MODEL_PATH"
nohup $VLLM serve "$MODEL_PATH" \
    --port $PORT \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --reasoning-parser deepseek_r1 \
    --enable-prefix-caching \
    > /data/vllm_serve_base.log 2>&1 &
VLLM_PID=$!
echo "$(date): vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "$(date): Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "$(date): vLLM is ready! (waited ${i}0s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "$(date): ERROR: vLLM failed to start after 20 min. Check /data/vllm_serve_base.log"
        exit 1
    fi
    sleep 10
done

# ── Step 5: Split tasks into 4 batches ──
echo "$(date): Splitting 480 tasks into 4 batches..."
python3 -c "
import json
with open('$DATA_ROOT/tasks_and_rubrics.json') as f:
    tasks = json.load(f)
ids = sorted([t['task_id'] for t in tasks])
n = len(ids)
bs = (n + 3) // 4
for i in range(4):
    batch = ids[i*bs:(i+1)*bs]
    with open(f'/tmp/baseline_batch_{i}.txt', 'w') as f:
        f.write('\n'.join(batch) + '\n')
    print(f'Batch {i}: {len(batch)} tasks')
"

# ── Step 6: Run 4 eval batches ──
echo "$(date): Starting 4 eval batches..."
mkdir -p "$EVAL_DIR"
for i in 0 1 2 3; do
    nohup $PYTHON "$EVAL_SCRIPT" \
        --model "$MODEL_NAME" \
        --api-port $PORT \
        --eval-dir "$EVAL_DIR" \
        --data-root "$DATA_ROOT" \
        --tool-root "$TOOL_ROOT" \
        --subset "/tmp/baseline_batch_${i}.txt" \
        --no-eval \
        > "/data/baseline_eval_batch_${i}.log" 2>&1 &
    echo "$(date): Batch $i started (PID: $!)"
done

# ── Step 7: Wait for all eval batches to finish ──
echo "$(date): Waiting for baseline eval to complete..."
while true; do
    EVAL_PROCS=$(ps aux | grep "eval_qwen3_local_with_snapshots_latest.py.*$MODEL_NAME" | grep -v grep | wc -l)
    DONE=$(wc -l < "$EVAL_DIR/results_${MODEL_NAME}.jsonl" 2>/dev/null || echo 0)
    echo "$(date): $EVAL_PROCS processes running, $DONE/480 tasks done"
    if [ "$EVAL_PROCS" -eq 0 ]; then
        echo "$(date): All baseline eval batches finished. $DONE tasks completed."
        break
    fi
    sleep 120
done

# ── Step 8: Run grading ──
echo "$(date): Starting grading for baseline..."
python3 "$GRADE_SCRIPT" \
    --model "$MODEL_NAME" \
    --eval-dir "$EVAL_DIR" \
    --resume \
    > /data/grading_base.log 2>&1

echo "$(date): === Baseline complete! ==="
echo "$(date): Results: $EVAL_DIR"
echo "$(date): Grading: /data/grading_base.log"

# Print summary
tail -20 /data/grading_base.log
