#!/bin/bash
set -e

# ── Config ───────────────────────────────────────────────────────────────────
SAVE_DIR="/data/Qwen3-235B-A22B-sft-v2-0314-unmasked-lr5e6"
HF_DIR="/data/Qwen3-235B-A22B-sft-v2-0314-unmasked-lr5e6-hf"
EVAL_DIR="/data/apex_eval_235b_v2_0314_unmasked_lr5e6"
MODEL_NAME="qwen3_235b_v2_0314_unmasked_lr5e6"
JUDGE_MODEL="anthropic/claude-sonnet-4-5"
JUDGE_KEY="<ANTHROPIC_API_KEY>"
EVAL_VENV="/data/eval_venv"
PYTHON="${EVAL_VENV}/bin/python"

echo "============================================================"
echo "Step 1: SFT Training (4 nodes, 235B)"
echo "============================================================"
bash /data/run-qwen3-235B-A22B-sft-v2-0314-unmasked.sh
echo "Training complete."

echo "============================================================"
echo "Step 2: Convert torch_dist checkpoint to HF"
echo "============================================================"
LATEST_ITER=$(cat ${SAVE_DIR}/latest_checkpointed_iteration.txt)
ITER_DIR="${SAVE_DIR}/iter_$(printf '%07d' ${LATEST_ITER})"
echo "Converting ${ITER_DIR} -> ${HF_DIR}"

${PYTHON} /data/mingye_b200-1/slime/tools/convert_torch_dist_to_hf.py \
  --input-dir "${ITER_DIR}" \
  --output-dir "${HF_DIR}" \
  --origin-hf-dir /data/Qwen3-235B-A22B-Thinking-2507 \
  --vocab-size 151936

echo "HF conversion complete: ${HF_DIR}"

echo "============================================================"
echo "Step 3: Start vLLM server (8 GPU, local)"
echo "============================================================"
pkill -f "vllm" || true
sleep 5

# Add model config to eval script if not exists
if ! grep -q "${MODEL_NAME}" /data/eval_qwen3_local_with_snapshots_latest.py; then
  ${PYTHON} -c "
with open('/data/eval_qwen3_local_with_snapshots_latest.py') as f:
    content = f.read()
config = '''    \"${MODEL_NAME}\": {
        \"api_type\": \"openai\",
        \"api_url\": \"http://localhost:8000/v1/chat/completions\",
        \"model\": \"${HF_DIR}\",
        \"api_key\": \"dummy\",
        \"extra_params\": {},
        \"context_window\": 131072,
    },
    \"sonnet\":'''
content = content.replace('    \"sonnet\":', config)
with open('/data/eval_qwen3_local_with_snapshots_latest.py', 'w') as f:
    f.write(content)
print('Model config added to eval script')
"
fi

nohup ${EVAL_VENV}/bin/vllm serve "${HF_DIR}" \
  --trust-remote-code \
  --max-model-len 131072 \
  --reasoning-parser deepseek_r1 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  > /data/vllm_serve_235b_unmasked_lr5e6.log 2>&1 &
VLLM_PID=$!
echo "vLLM starting (PID: ${VLLM_PID})"

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM ready after $((i * 10)) seconds"
    break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "ERROR: vLLM process died. Check /data/vllm_serve_235b_unmasked_lr5e6.log"
    exit 1
  fi
  sleep 10
done

echo "============================================================"
echo "Step 4: Verify tool schema injection"
echo "============================================================"
${PYTHON} -c "
import sys, os, json
sys.path.insert(0, '/data')
# Simulate what eval does to load tools
exec(open('/data/eval_qwen3_local_with_snapshots_latest.py').read().split('def main')[0])
tools = load_tool_schemas()
if len(tools) == 0:
    print('ERROR: No tools loaded! Tool schema injection will fail.')
    sys.exit(1)
tools_json = '\n'.join(json.dumps(t) for t in tools)
print(f'OK: {len(tools)} tools loaded, total schema length: {len(tools_json)} chars')
# Verify tools are non-empty in the prompt
system_content = 'test'
system_content += '\n\n# Tools\n\n<tools>\n' + tools_json + '\n</tools>'
if '<tools>\n</tools>' in system_content:
    print('ERROR: Tools section is empty!')
    sys.exit(1)
print('Tool schema injection verified.')
"

echo "============================================================"
echo "Step 5: Run eval (480 tasks)"
echo "============================================================"
${PYTHON} /data/eval_qwen3_local_with_snapshots_latest.py \
  --model "${MODEL_NAME}" \
  --eval-dir "${EVAL_DIR}" \
  --no-eval \
  --resume

echo "Eval complete."

echo "============================================================"
echo "Step 6: Grade with Claude Sonnet 4.5"
echo "============================================================"
${PYTHON} /data/grade_eval_results_latest.py \
  --model "${MODEL_NAME}" \
  --eval-dir "${EVAL_DIR}" \
  --judge-model "${JUDGE_MODEL}" \
  --judge-api-key "${JUDGE_KEY}" \
  --resume

echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"

kill ${VLLM_PID} 2>/dev/null || true
