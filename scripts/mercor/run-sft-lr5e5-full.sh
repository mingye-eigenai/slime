#!/bin/bash
#
# Full pipeline: train lr=5e5 -> convert last ckpt to HF -> upload to HF
# Runs on local node inside slime docker container
#
set -ex

# ===== PHASE 1: Training =====
bash /data/run-qwen3-30B-sft-combined-lr5e5.sh

# ===== PHASE 2: Convert last checkpoint to HF =====
SAVE_DIR="/data/Qwen3-30B-A3B-sft-v2-0314-lr5e5"
ORIGIN_HF="/data/Qwen3-30B-A3B-Thinking-2507"

# Find the last checkpoint iteration
LAST_ITER=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")
echo "Last checkpoint iteration: ${LAST_ITER}"

# Determine input dir - could be iter_XXXXXXX or release format
if [ -d "${SAVE_DIR}/iter_$(printf '%07d' ${LAST_ITER})" ]; then
    INPUT_DIR="${SAVE_DIR}/iter_$(printf '%07d' ${LAST_ITER})"
elif [ -d "${SAVE_DIR}/release" ]; then
    INPUT_DIR="${SAVE_DIR}/release"
else
    echo "ERROR: Cannot find checkpoint directory for iteration ${LAST_ITER}"
    ls "${SAVE_DIR}/"
    exit 1
fi

OUTPUT_DIR="/data/Qwen3-30B-A3B-sft-v2-0314-lr5e5-hf"
echo "Converting ${INPUT_DIR} -> ${OUTPUT_DIR}"

cd /root/slime
PYTHONPATH=/root/Megatron-LM python3 tools/convert_torch_dist_to_hf.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF}" \
    --force

echo "Conversion done!"

# ===== PHASE 3: Upload to HuggingFace =====
export HF_TOKEN="<HF_TOKEN>"
REPO_NAME="eigen-ai-labs/Qwen3-30B-A3B-sft-v2-0314-lr5e5"

echo "Uploading to ${REPO_NAME} ..."
huggingface-cli upload "${REPO_NAME}" "${OUTPUT_DIR}" . --repo-type model
echo "Upload complete: ${REPO_NAME}"
