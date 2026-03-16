#!/bin/bash
# run-law-gemini.sh
# Upload opus-law HF model, then run gemini-law SFT and upload.
# Runs on 85.234.91.90 (law node).

set -euo pipefail

BASE_FOLDER="${BASE_FOLDER:-/data}"
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
HF_ORG="${HF_ORG:-eigen-ai-labs}"

OPUS_HF="${BASE_FOLDER}/Qwen3-30B-A3B-sft-opus-law-hf"
GEMINI_CKPT="${BASE_FOLDER}/Qwen3-30B-A3B-sft-gemini-law"
GEMINI_HF="${BASE_FOLDER}/Qwen3-30B-A3B-sft-gemini-law-hf"
GEMINI_DATA="${BASE_FOLDER}/apex_sft_gemini_law.jsonl"

# --- Step 1: upload opus HF model ---
echo ""
echo "=== Uploading opus-law to HuggingFace ==="
sudo chmod -R a+w "${OPUS_HF}" || true
python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
repo_id = "${HF_ORG}/Qwen3-30B-A3B-Thinking-2507-sft-opus-law"
local_dir = "${OPUS_HF}"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Uploading {local_dir} -> {repo_id} ...")
api.upload_large_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
print(f"Upload complete: https://huggingface.co/{repo_id}")
PYEOF

# --- Step 2: run gemini-law SFT ---
echo ""
echo "============================================================"
echo "  Training: gemini-law"
echo "  data:      ${GEMINI_DATA}"
echo "  ckpt:      ${GEMINI_CKPT}"
echo "  hf output: ${GEMINI_HF}"
echo "  num_epoch: 5  save_interval: 2  batch_size: 128"
echo "============================================================"

sudo docker pull "${SLIME_IMAGE}"

sudo docker run --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    -v /data:/data \
    -v /home/user/.ssh:/root/.ssh:ro \
    -e BASE_FOLDER="${BASE_FOLDER}" \
    -e PROMPT_DATA="${GEMINI_DATA}" \
    -e CKPT_SAVE_DIR="${GEMINI_CKPT}" \
    -e SAVE_HF_PATH="${GEMINI_HF}" \
    -e NUM_EPOCH=5 \
    -e SAVE_INTERVAL=2 \
    -e BATCH_SIZE=128 \
    -e NO_SAVE_OPTIM=1 \
    "${SLIME_IMAGE}" \
    bash /data/run-qwen3-30B-A3B-sft-pipeline.sh

echo ""
echo "=== gemini-law training done. Uploading to HuggingFace ==="
sudo chmod -R a+w "${GEMINI_HF}" || true
python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
repo_id = "${HF_ORG}/Qwen3-30B-A3B-Thinking-2507-sft-gemini-law"
local_dir = "${GEMINI_HF}"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Uploading {local_dir} -> {repo_id} ...")
api.upload_large_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
print(f"Upload complete: https://huggingface.co/{repo_id}")
PYEOF

echo ""
echo "=== law (gemini) all done ==="
