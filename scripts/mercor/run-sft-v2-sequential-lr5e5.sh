#!/bin/bash
#
# Sequential training on local node (lr=5e-5):
#   1. Train with masked data -> convert -> upload
#   2. Train with unmasked data -> convert -> upload
#
set -ex

ORIGIN_HF="/data/Qwen3-30B-A3B-Thinking-2507"
export HF_TOKEN="<HF_TOKEN>"

# ===================================================================
# RUN 1: MASKED (original step_loss_mask on error turns)
# ===================================================================
echo "=========================================="
echo "RUN 1: MASKED DATA (lr=5e-5)"
echo "=========================================="

# Swap in masked data
cp /data/apex_sft_opus_aug_plus_code_opt_zhanlin_v2_0314_masked.jsonl \
   /data/apex_sft_opus_aug_plus_code_opt_zhanlin_v2_0314.jsonl

bash /data/run-qwen3-30B-sft-combined-lr5e5.sh

# Convert
SAVE_DIR="/data/Qwen3-30B-A3B-sft-v2-0314-lr5e5"
LAST_ITER=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")
if [ -d "${SAVE_DIR}/iter_$(printf '%07d' ${LAST_ITER})" ]; then
    INPUT_DIR="${SAVE_DIR}/iter_$(printf '%07d' ${LAST_ITER})"
elif [ -d "${SAVE_DIR}/release" ]; then
    INPUT_DIR="${SAVE_DIR}/release"
else
    echo "ERROR: Cannot find checkpoint for iter ${LAST_ITER}"; exit 1
fi

OUTPUT_DIR="/data/Qwen3-30B-A3B-sft-v2-0314-lr5e5-hf"
cd /root/slime
PYTHONPATH=/root/Megatron-LM python3 tools/convert_torch_dist_to_hf.py \
    --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF}" --force

huggingface-cli upload "eigen-ai-labs/Qwen3-30B-A3B-sft-v2-0314-lr5e5" "${OUTPUT_DIR}" . --repo-type model
echo "RUN 1 DONE: masked lr5e5"

# Clean up torch_dist checkpoints to save space
rm -rf "${SAVE_DIR}"

# ===================================================================
# RUN 2: UNMASKED (reasoning on error turns included in loss)
# ===================================================================
echo "=========================================="
echo "RUN 2: UNMASKED DATA (lr=5e-5)"
echo "=========================================="

# Swap in unmasked data
cp /data/apex_sft_opus_aug_plus_code_opt_zhanlin_v2_0314_unmasked.jsonl \
   /data/apex_sft_opus_aug_plus_code_opt_zhanlin_v2_0314.jsonl

# Override save name and wandb group for this run
export SFT_V2_SAVE_NAME="Qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5"
export SFT_V2_WANDB_GROUP="qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5"

# Patch training script inline for run 2
sed -i 's|SAVE_NAME="Qwen3-30B-A3B-sft-v2-0314-lr5e5"|SAVE_NAME="Qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5"|' /data/run-qwen3-30B-sft-combined-lr5e5.sh
sed -i 's|wandb-group qwen3-30B-A3B-sft-v2-0314-lr5e5|wandb-group qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5|' /data/run-qwen3-30B-sft-combined-lr5e5.sh

bash /data/run-qwen3-30B-sft-combined-lr5e5.sh

# Convert
SAVE_DIR="/data/Qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5"
LAST_ITER=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")
if [ -d "${SAVE_DIR}/iter_$(printf '%07d' ${LAST_ITER})" ]; then
    INPUT_DIR="${SAVE_DIR}/iter_$(printf '%07d' ${LAST_ITER})"
elif [ -d "${SAVE_DIR}/release" ]; then
    INPUT_DIR="${SAVE_DIR}/release"
else
    echo "ERROR: Cannot find checkpoint for iter ${LAST_ITER}"; exit 1
fi

OUTPUT_DIR="/data/Qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5-hf"
cd /root/slime
PYTHONPATH=/root/Megatron-LM python3 tools/convert_torch_dist_to_hf.py \
    --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF}" --force

huggingface-cli upload "eigen-ai-labs/Qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5" "${OUTPUT_DIR}" . --repo-type model
echo "RUN 2 DONE: unmasked lr5e5"

# Clean up
rm -rf "${SAVE_DIR}"

# Restore training script
sed -i 's|SAVE_NAME="Qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5"|SAVE_NAME="Qwen3-30B-A3B-sft-v2-0314-lr5e5"|' /data/run-qwen3-30B-sft-combined-lr5e5.sh
sed -i 's|wandb-group qwen3-30B-A3B-sft-v2-0314-unmasked-lr5e5|wandb-group qwen3-30B-A3B-sft-v2-0314-lr5e5|' /data/run-qwen3-30B-sft-combined-lr5e5.sh

echo "ALL DONE on local node"
