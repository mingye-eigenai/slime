#!/bin/bash
#
# run-sft-30B-domain-node.sh
#
# Per-node script: runs opus SFT for a given domain across 3 LRs,
# converts final checkpoint to HF and uploads after each.
#
# Usage:
#   HF_TOKEN=hf_xxx DOMAIN=investment_banking bash /data/run-sft-30B-domain-node.sh
#
# DOMAIN must be one of: investment_banking | management_consulting | law | mix

set -euo pipefail

BASE_FOLDER="${BASE_FOLDER:-/data}"
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
HF_ORG="${HF_ORG:-eigen-ai-labs}"
DOMAIN="${DOMAIN:?DOMAIN must be set}"
NUM_EPOCH="${NUM_EPOCH:-8}"
WANDB_API_KEY="${WANDB_API_KEY:-<WANDB_API_KEY>}"

# LRs to sweep (overridable via env)
LR_LIST=(${LR_LIST_OVERRIDE:-5e-5 3e-5 1e-5})

case "${DOMAIN}" in
  investment_banking)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_investment_banking_filled_fixed.jsonl"
    OPUS_SAMPLES=247; BATCH_SIZE=16
    ;;
  management_consulting)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_management_consulting_filled_fixed.jsonl"
    OPUS_SAMPLES=260; BATCH_SIZE=16
    ;;
  law)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_law_filled_fixed.jsonl"
    OPUS_SAMPLES=100; BATCH_SIZE=8
    ;;
  mix)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_mix_filled_fixed.jsonl"
    OPUS_SAMPLES=607; BATCH_SIZE=32
    ;;
  gemini_mix_rewritten)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_gemini_mix_rewritten.jsonl"
    OPUS_SAMPLES=627; BATCH_SIZE=32
    ;;
  opus_mix_combined)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_mix_combined_trimmed.jsonl"
    OPUS_SAMPLES=715; BATCH_SIZE=32
    ;;
  opus_all)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_all_fixed.jsonl"
    OPUS_SAMPLES=1939; BATCH_SIZE=32
    ;;
  opus_sonnet_all)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_sonnet_all_fixed_trimmed.jsonl"
    OPUS_SAMPLES=3494; BATCH_SIZE=32
    ;;
  opus_all_v2)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_all_v2_trimmed.jsonl"
    OPUS_SAMPLES=1452; BATCH_SIZE=32
    ;;
  opus_all_v2_60)
    OPUS_DATA="${BASE_FOLDER}/apex_sft_opus_all_v2_60_trimmed.jsonl"
    OPUS_SAMPLES=1622; BATCH_SIZE=32
    ;;
  *)
    echo "Unknown DOMAIN: ${DOMAIN}. Must be: investment_banking | management_consulting | law | mix"
    exit 1
    ;;
esac

# Allow env var override
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-${BATCH_SIZE}}"

# Compute ceil-based rollout counts
ROLLOUT_PER_EPOCH=$(( (OPUS_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))
NUM_ROLLOUT=$(( ROLLOUT_PER_EPOCH * NUM_EPOCH ))
# Save every epoch
SAVE_INTERVAL=${ROLLOUT_PER_EPOCH}

D="${DOMAIN//_/-}"

# ---------------------------------------------------------------------------
# Helper: run one training job then upload HF checkpoint
# ---------------------------------------------------------------------------
run_and_upload() {
  local RUN_NAME="$1"
  local PROMPT_DATA="$2"
  local CKPT_SAVE_DIR="$3"
  local SAVE_HF_PATH="$4"
  local HF_REPO="${HF_ORG}/$5"
  local SAVE_INTERVAL="$6"
  local BATCH_SIZE="$7"
  local NUM_ROLLOUT="$8"
  local LR="$9"

  echo ""
  echo "============================================================"
  echo "  Training: ${RUN_NAME}"
  echo "  data:          ${PROMPT_DATA}"
  echo "  ckpt:          ${CKPT_SAVE_DIR}"
  echo "  hf output:     ${SAVE_HF_PATH}"
  echo "  hf repo:       ${HF_REPO}"
  echo "  lr:            ${LR}"
  echo "  num_rollout:   ${NUM_ROLLOUT} (${ROLLOUT_PER_EPOCH}/epoch × ${NUM_EPOCH} epochs)  save_interval: ${SAVE_INTERVAL}  batch_size: ${BATCH_SIZE}"
  echo "============================================================"

  sudo docker run --gpus all --ipc=host --shm-size=16g \
      --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
      --network host \
      -v /data:/data \
      -v /home/user/.ssh:/root/.ssh:ro \
      -e BASE_FOLDER="${BASE_FOLDER}" \
      -e PROMPT_DATA="${PROMPT_DATA}" \
      -e CKPT_SAVE_DIR="${CKPT_SAVE_DIR}" \
      -e SAVE_HF_PATH="${SAVE_HF_PATH}" \
      -e NUM_ROLLOUT="${NUM_ROLLOUT}" \
      -e SAVE_INTERVAL="${SAVE_INTERVAL}" \
      -e BATCH_SIZE="${BATCH_SIZE}" \
      -e NO_SAVE_OPTIM=1 \
      -e LR="${LR}" \
      -e WANDB_API_KEY="${WANDB_API_KEY}" \
      -e WANDB_RUN_NAME="${RUN_NAME}" \
      "${SLIME_IMAGE}" \
      bash /data/run-qwen3-30B-A3B-sft-pipeline.sh

  echo ""
  echo "=== [${RUN_NAME}] Training done. Converting final checkpoint to HF ==="

  # Find the latest checkpoint iteration
  local LATEST_ITER
  LATEST_ITER=$(cat "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [ -z "${LATEST_ITER}" ]; then
    echo "!!! No checkpoint found in ${CKPT_SAVE_DIR}, skipping conversion"
    return 1
  fi
  local ITER_DIR="${CKPT_SAVE_DIR}/iter_$(printf '%07d' ${LATEST_ITER})"
  echo "  Converting ${ITER_DIR} -> ${SAVE_HF_PATH} (vocab_size=151936)"

  sudo docker run --gpus all --ipc=host --shm-size=16g \
      --network host \
      -v /data:/data \
      "${SLIME_IMAGE}" \
      python3 /data/mingye_b200-1/slime/tools/convert_torch_dist_to_hf.py \
        --input-dir "${ITER_DIR}" \
        --output-dir "${SAVE_HF_PATH}" \
        --origin-hf-dir "${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507" \
        --vocab-size 151936

  sudo chmod -R a+w "${SAVE_HF_PATH}" || true

  echo "=== [${RUN_NAME}] Uploading to ${HF_REPO} ==="

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

# --- sweep LRs ---
FAILED_LRS=()
for LR in "${LR_LIST[@]}"; do
  # tag: 3e-5 -> 3e5, 1e-5 -> 1e5, 5e-6 -> 5e6
  LR_TAG="${LR//-/}"   # remove minus sign: 3e-5 -> 3e5

  CKPT="${BASE_FOLDER}/Qwen3-30B-A3B-sft-${D}-lr${LR_TAG}"
  HF_OUT="${BASE_FOLDER}/Qwen3-30B-A3B-sft-${D}-lr${LR_TAG}-hf"
  REPO_NAME="Qwen3-30B-A3B-Thinking-2507-sft-${D}-lr${LR_TAG}"
  RUN_NAME="${D}-lr${LR_TAG}"

  set +e
  run_and_upload \
    "${RUN_NAME}" \
    "${OPUS_DATA}" \
    "${CKPT}" \
    "${HF_OUT}" \
    "${REPO_NAME}" \
    "${SAVE_INTERVAL}" \
    "${BATCH_SIZE}" \
    "${NUM_ROLLOUT}" \
    "${LR}"
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then
    echo "!!! [${RUN_NAME}] FAILED with exit code ${RC} — continuing to next LR"
    FAILED_LRS+=("${LR}")
  fi
done

# --- upload intermediate epoch checkpoints if requested ---
# UPLOAD_EPOCHS: space-separated list of epoch numbers to also convert+upload (e.g. "3")
if [ -n "${UPLOAD_EPOCHS:-}" ]; then
  for LR in "${LR_LIST[@]}"; do
    LR_TAG="${LR//-/}"
    CKPT="${BASE_FOLDER}/Qwen3-30B-A3B-sft-${D}-lr${LR_TAG}"
    for EPOCH_NUM in ${UPLOAD_EPOCHS}; do
      ITER_NUM=$(printf "%07d" $(( ROLLOUT_PER_EPOCH * EPOCH_NUM - 1 )))
      ITER_DIR="${CKPT}/iter_${ITER_NUM}"
      if [ ! -d "${ITER_DIR}" ]; then
        echo "!!! Epoch ${EPOCH_NUM} checkpoint not found: ${ITER_DIR} — skipping"
        continue
      fi
      EPOCH_HF="${CKPT}-e${EPOCH_NUM}-hf"
      EPOCH_REPO="${HF_ORG}/Qwen3-30B-A3B-Thinking-2507-sft-${D}-lr${LR_TAG}-e${EPOCH_NUM}"
      echo ""
      echo "=== Converting epoch ${EPOCH_NUM} (${ITER_DIR}) to HF ==="
      sudo docker run --gpus all --ipc=host --shm-size=16g \
          --network host \
          -v /data:/data \
          "${SLIME_IMAGE}" \
          python3 /data/mingye_b200-1/slime/tools/convert_torch_dist_to_hf.py \
            --input-dir "${ITER_DIR}" \
            --output-dir "${EPOCH_HF}" \
            --origin-hf-dir "${BASE_FOLDER}/Qwen3-30B-A3B-Thinking-2507" \
            --vocab-size 151936
      sudo chmod -R a+w "${EPOCH_HF}" || true
      echo "=== Uploading epoch ${EPOCH_NUM} to ${EPOCH_REPO} ==="
      python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
repo_id = "${EPOCH_REPO}"
local_dir = "${EPOCH_HF}"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Uploading {local_dir} -> {repo_id} ...")
api.upload_large_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
print(f"Upload complete: https://huggingface.co/{repo_id}")
PYEOF
    done
  done
fi

if [ ${#FAILED_LRS[@]} -gt 0 ]; then
  echo ""
  echo "!!! WARNING: the following LRs failed: ${FAILED_LRS[*]}"
  exit 1
fi

echo ""
echo "=== [${DOMAIN}] All done ==="
