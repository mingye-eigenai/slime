#!/bin/bash
#
# run-sft-opus-gemini-pipeline.sh
#
# Runs three sequential SFT training jobs:
#   1. Continue least_turns training (5 more epochs, loads from existing checkpoint)
#   2. Opus-only training from base model
#   3. Gemini-only training from base model
#
# Each job converts the final checkpoint to HF format and uploads to HuggingFace.
#
# Usage:
#   HF_TOKEN=hf_xxx BASE_FOLDER=/data bash /data/run-sft-opus-gemini-pipeline.sh

set -euo pipefail

BASE_FOLDER="${BASE_FOLDER:-/data}"
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
MASTER_ADDR="${MASTER_ADDR:-92.38.143.19}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
HF_ORG="${HF_ORG:-eigen-ai-labs}"

# ---------------------------------------------------------------------------
# Helper: run one training job then upload the HF checkpoint
#
# Args:
#   $1  RUN_NAME        — human-readable name for logging
#   $2  PROMPT_DATA     — training JSONL path
#   $3  CKPT_SAVE_DIR   — megatron checkpoint dir (load + save)
#   $4  SAVE_HF_PATH    — path to write HF model (overwritten at each save)
#   $5  HF_REPO_SUFFIX  — repo name under HF_ORG
#   $6  NUM_EPOCH       — total epochs to train up to (e.g. 8 if 3 done + 5 more)
#   $7  SAVE_INTERVAL   — steps between checkpoints (≈ steps per epoch)
#   $8  NO_SAVE_OPTIM   — 1 = model only (default), 0 = include optimizer states
# ---------------------------------------------------------------------------
run_and_upload() {
  local RUN_NAME="$1"
  local PROMPT_DATA="$2"
  local CKPT_SAVE_DIR="$3"
  local SAVE_HF_PATH="$4"
  local HF_REPO="${HF_ORG}/$5"
  local NUM_EPOCH="$6"
  local SAVE_INTERVAL="$7"
  local NO_SAVE_OPTIM="${8:-1}"

  echo ""
  echo "============================================================"
  echo "  Training: ${RUN_NAME}"
  echo "  data:          ${PROMPT_DATA}"
  echo "  ckpt:          ${CKPT_SAVE_DIR}"
  echo "  hf output:     ${SAVE_HF_PATH}"
  echo "  hf repo:       ${HF_REPO}"
  echo "  num_epoch:     ${NUM_EPOCH}  save_interval: ${SAVE_INTERVAL}  no_save_optim: ${NO_SAVE_OPTIM}"
  echo "============================================================"

  sudo docker pull "${SLIME_IMAGE}"

  sudo docker run --gpus all --ipc=host --shm-size=16g \
      --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
      --network host \
      --device /dev/infiniband \
      -v /data:/data \
      -v /home/user/.ssh:/root/.ssh:ro \
      -e BASE_FOLDER="${BASE_FOLDER}" \
      -e MASTER_ADDR="${MASTER_ADDR}" \
      -e SLIME_IMAGE="${SLIME_IMAGE}" \
      -e PROMPT_DATA="${PROMPT_DATA}" \
      -e CKPT_SAVE_DIR="${CKPT_SAVE_DIR}" \
      -e SAVE_HF_PATH="${SAVE_HF_PATH}" \
      -e NUM_EPOCH="${NUM_EPOCH}" \
      -e SAVE_INTERVAL="${SAVE_INTERVAL}" \
      -e NO_SAVE_OPTIM="${NO_SAVE_OPTIM}" \
      "${SLIME_IMAGE}" \
      bash /data/run-qwen3-235B-A22B-sft-h200-4nodes-pipeline.sh

  echo ""
  echo "=== [${RUN_NAME}] Training done. Uploading to ${HF_REPO} ==="

  python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
repo_id = "${HF_REPO}"
local_dir = "${SAVE_HF_PATH}"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Uploading {local_dir} -> {repo_id} ...")
api.upload_large_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
print(f"Upload complete: https://huggingface.co/{repo_id}")
PYEOF
}

# ---------------------------------------------------------------------------
# Job 1: Continue least_turns training (+5 epochs, total 8)
#   - loads from existing checkpoint (which has optimizer states)
#   - 957 samples / 128 batch = 7 steps/epoch
# ---------------------------------------------------------------------------
run_and_upload \
  "least_turns_continue" \
  "/data/apex_sft_least_turns_p95.jsonl" \
  "${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507_least_turns_v2" \
  "${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507-least-turns-hf" \
  "Qwen3-235B-A22B-Thinking-2507-least-turns-v2" \
  8 \
  7 \
  1

# ---------------------------------------------------------------------------
# Job 2: Opus-only training from base model
#   - 1960 samples / 128 batch = 16 steps/epoch
# ---------------------------------------------------------------------------
run_and_upload \
  "opus" \
  "/data/apex_sft_opus_80k.jsonl" \
  "${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507-sft-opus" \
  "${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507-sft-opus-hf" \
  "Qwen3-235B-A22B-Thinking-2507-sft-opus" \
  8 \
  16 \
  1

# ---------------------------------------------------------------------------
# Job 3: Gemini-only training from base model
#   - 1865 samples / 128 batch = 15 steps/epoch
# ---------------------------------------------------------------------------
run_and_upload \
  "gemini" \
  "/data/apex_sft_gemini.jsonl" \
  "${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507-sft-gemini" \
  "${BASE_FOLDER}/Qwen3-235B-A22B-Thinking-2507-sft-gemini-hf" \
  "Qwen3-235B-A22B-Thinking-2507-sft-gemini" \
  8 \
  15 \
  1

echo ""
echo "=== All 3 jobs complete ==="
