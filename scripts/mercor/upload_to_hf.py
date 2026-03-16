#!/usr/bin/env python3
"""Upload converted HF model to HuggingFace."""
import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", "<HF_TOKEN>")
REPO_ID = "eigen-ai-labs/Qwen3-235B-A22B-Thinking-2507-least-turns-v1"
LOCAL_DIR = "/data/Qwen3-235B-A22B-Thinking-2507_least_turns_v1_hf"

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo ready: {REPO_ID}")
except Exception as e:
    print(f"Repo creation: {e}")

print(f"Uploading from {LOCAL_DIR} to {REPO_ID} ...")
api.upload_large_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)
print("Upload complete!")
