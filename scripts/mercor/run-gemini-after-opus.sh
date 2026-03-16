#!/bin/bash
# run-gemini-after-opus.sh
# Upload opus HF model, then run gemini SFT and upload.
# Called when opus training is already complete.
#
# Required env vars:
#   HF_TOKEN, DOMAIN, BASE_FOLDER (default /data), SLIME_IMAGE, HF_ORG

set -euo pipefail

BASE_FOLDER="${BASE_FOLDER:-/data}"
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
HF_ORG="${HF_ORG:-eigen-ai-labs}"
DOMAIN="${DOMAIN:?DOMAIN must be set}"
NUM_EPOCH=5

D="${DOMAIN//_/-}"

case "${DOMAIN}" in
  investment_banking)
    GEMINI_DATA="${BASE_FOLDER}/apex_sft_gemini_investment_banking.jsonl"
    GEMINI_SAMPLES=278; GEMINI_BATCH=128
    ;;
  management_consulting)
    GEMINI_DATA="${BASE_FOLDER}/apex_sft_gemini_management_consulting.jsonl"
    GEMINI_SAMPLES=193; GEMINI_BATCH=128
    ;;
  law)
    GEMINI_DATA="${BASE_FOLDER}/apex_sft_gemini_law.jsonl"
    GEMINI_SAMPLES=162; GEMINI_BATCH=128
    ;;
  mix)
    GEMINI_DATA="${BASE_FOLDER}/apex_sft_gemini_mix.jsonl"
    GEMINI_SAMPLES=633; GEMINI_BATCH=128
    ;;
  *)
    echo "Unknown DOMAIN: ${DOMAIN}"
    exit 1
    ;;
esac

# ceil(samples/batch) * epochs — use all data, no drop_last
GEMINI_ROLLOUT_PER_EPOCH=$(( (GEMINI_SAMPLES + GEMINI_BATCH - 1) / GEMINI_BATCH ))
GEMINI_NUM_ROLLOUT=$(( GEMINI_ROLLOUT_PER_EPOCH * NUM_EPOCH ))
GEMINI_SAVE_INTERVAL=${GEMINI_ROLLOUT_PER_EPOCH}

OPUS_HF="${BASE_FOLDER}/Qwen3-30B-A3B-sft-opus-${D}-hf"
GEMINI_CKPT="${BASE_FOLDER}/Qwen3-30B-A3B-sft-gemini-${D}"
GEMINI_HF="${BASE_FOLDER}/Qwen3-30B-A3B-sft-gemini-${D}-hf"

echo ""
echo "=== Uploading opus-${D} to HuggingFace ==="
sudo chmod -R a+w "${OPUS_HF}" || true
python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
repo_id = "${HF_ORG}/Qwen3-30B-A3B-Thinking-2507-sft-opus-${D}"
local_dir = "${OPUS_HF}"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Uploading {local_dir} -> {repo_id} ...")
api.upload_large_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
print(f"Upload complete: https://huggingface.co/{repo_id}")
PYEOF

echo ""
echo "============================================================"
echo "  Training: gemini-${D}"
echo "  data:      ${GEMINI_DATA}"
echo "  ckpt:      ${GEMINI_CKPT}"
echo "  hf output: ${GEMINI_HF}"
echo "  num_rollout: ${GEMINI_NUM_ROLLOUT} (${GEMINI_ROLLOUT_PER_EPOCH}/epoch × ${NUM_EPOCH} epochs)  save_interval: ${GEMINI_SAVE_INTERVAL}  batch_size: ${GEMINI_BATCH}"
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
    -e NUM_ROLLOUT="${GEMINI_NUM_ROLLOUT}" \
    -e SAVE_INTERVAL="${GEMINI_SAVE_INTERVAL}" \
    -e BATCH_SIZE="${GEMINI_BATCH}" \
    -e NO_SAVE_OPTIM=1 \
    "${SLIME_IMAGE}" \
    bash /data/run-qwen3-30B-A3B-sft-pipeline.sh

echo ""
echo "=== gemini-${D} training done. Uploading to HuggingFace ==="
sudo chmod -R a+w "${GEMINI_HF}" || true
python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
repo_id = "${HF_ORG}/Qwen3-30B-A3B-Thinking-2507-sft-gemini-${D}"
local_dir = "${GEMINI_HF}"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Uploading {local_dir} -> {repo_id} ...")
api.upload_large_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
print(f"Upload complete: https://huggingface.co/{repo_id}")
PYEOF

echo ""
echo "=== [${DOMAIN}] gemini pipeline complete ==="
