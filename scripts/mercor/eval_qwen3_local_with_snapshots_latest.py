#!/usr/bin/env python3

"""
Batch evaluation of Qwen3-235B on the APEX-Agents benchmark.

Downloads world files and task files from HuggingFace as needed,
sets up per-task filesystem environments, runs the tool-using agent,
and reports aggregate accuracy.

Usage:
    conda run -n imagegen_latest python eval_qwen3_batch.py [options]

Options:
    --world WORLD_ID    Only run tasks for a specific world
    --task TASK_ID      Only run a single task
    --resume            Skip tasks that already have results
    --retry-errors      With --resume, retry tasks that had errors
    --max-tasks N       Maximum number of tasks to evaluate
    --dry-run           List tasks without running them
"""

import argparse
import asyncio
import copy
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import random
import signal

import requests

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "qwen3_sft": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "Qwen3-235B-A22B-Thinking-2507-least-turns-v1",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,   # 128k
    },
    "qwen3_base": {
        "api_type": "openai",
        "api_url": "http://localhost:8001/v1/chat/completions",
        "model": "Qwen3-235B-A22B-Thinking-2507",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,   # 128k
    },
    "qwen3_30b_base": {
        "api_type": "openai",
        "api_url": "http://localhost:8010/v1/chat/completions",
        "model": "Qwen3-30B-A3B-Thinking-2507",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_gemini_ib": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-gemini-investment-banking",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_ib": {
        "api_type": "openai",
        "api_url": "http://localhost:8001/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-investment-banking",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_gemini_law": {
        "api_type": "openai",
        "api_url": "http://localhost:8002/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-gemini-law",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_law": {
        "api_type": "openai",
        "api_url": "http://localhost:8003/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-law",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_gemini_mc": {
        "api_type": "openai",
        "api_url": "http://localhost:8004/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-gemini-management-consulting",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_mc": {
        "api_type": "openai",
        "api_url": "http://localhost:8005/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-management-consulting",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_mix": {
        "api_type": "openai",
        "api_url": "http://localhost:8014/v1/chat/completions",
        "model": "Qwen3-30B-A3B-Thinking-2507-sft-opus-mix",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_gemini_mix": {
        "api_type": "openai",
        "api_url": "http://localhost:8009/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-gemini-mix",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_mix_lr5e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-mix-lr5e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_mix_lr5e5_e3": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-mix-lr5e5-e3",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_all_lr5e5_e3": {
        "api_type": "openai",
        "api_url": "http://localhost:8010/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-all-lr5e5-e3",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_all_v2_lr5e5_e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-all-v2-lr5e5-e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_sonnet_all_lr5e5_e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-sonnet-all-lr5e5-e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_all_lr5e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8012/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-all-lr5e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_mix_lr1e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8007/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-mix-lr1e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_law_lr5e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8001/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-law-lr5e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_gemini_mix_rewritten": {
        "api_type": "openai",
        "api_url": "http://localhost:8010/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-gemini-mix-rewritten-lr5e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_gemini_mix_rewritten_e3": {
        "api_type": "openai",
        "api_url": "http://localhost:8011/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-gemini-mix-rewritten-lr5e5-e3",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_ib_lr1e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8002/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-investment-banking-lr1e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_mc_lr1e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8003/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-management-consulting-lr1e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_mc_lr5e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8004/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-management-consulting-lr5e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_opus_ib_lr5e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8005/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-investment-banking-lr5e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_base_v2": {
        "api_type": "openai",
        "api_url": "http://localhost:8006/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_combined_e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-mix-combined-lr5e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_combined_e3": {
        "api_type": "openai",
        "api_url": "http://localhost:8003/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-Thinking-2507-sft-opus-mix-combined-lr5e5-e3",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_sft": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-235B-A22B-Thinking-2507-SFT",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_base": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-235B-A22B-Thinking-2507",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_least_turns": {
        "api_type": "openai",
        "api_url": "http://localhost:8050/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-235B-A22B-Thinking-2507-least-turns-v1",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_opus_mix_lr1e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8050/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-235B-A22B-sft-opus-mix-lr1e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_opus_mix_base_lr1e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8050/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-235B-A22B-Thinking-2507-sft-opus-mix-base-lr1e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_opus_sonnet_best_v2_lr1e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8050/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-235B-A22B-Thinking-2507-sft-opus-sonnet-best-v2-lr1e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_opus_sonnet_best_v2_lr5e6": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "/data/Qwen3-235B-A22B-sft-opus-sonnet-best-v2-lr5e6-hf",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_235b_base": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "/data/Qwen3-235B-A22B-Thinking-2507",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_zhanlin_v1_e3": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-sft-zhanlin-v1-e3",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_zhanlin_v1_e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/home/mingye/Qwen3-30B-A3B-sft-zhanlin-v1-e5",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_zhanlin_v2_e3": {
        "api_type": "openai",
        "api_url": "http://localhost:8008/v1/chat/completions",
        "model": "/data/Qwen3-30B-A3B-sft-zhanlin_v2-e3-hf",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3_30b_zhanlin_v2_lr1e5_e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "/data/Qwen3-30B-A3B-sft-zhanlin_v2-lr1e5-e5-hf",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "qwen3": {
        "api_type": "openai",
        "api_url": "https://api-web.eigenai.com/api/v1/chat/completions",
        "model": "qwen3-235b-a22b-thinking-2507-fp8",
        "api_key": "<OPENAI_API_KEY>",
        "extra_params": {"reasoning_effort": "high"},
        "context_window": 131072,   # 128k
    },
    "gpt5": {
        "api_type": "openai_responses",
        "api_url": "https://api.openai.com/v1/responses",
        "model": "gpt-5.2-codex",
        "api_key": "<OPENAI_API_KEY>",
        "extra_params": {},
        "no_temperature": True,
        "context_window": 131072,   # 128k
    },
    "opus": {
        "api_type": "openai",
        "api_url": "https://api.gmi-serving.com/v1/chat/completions",
        "model": "anthropic/claude-opus-4.6",
        "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjEyZjE2YThiLTY4N2ItNDNlMC1iZmI0LTkyNzZjNTRjYTBjNSIsInNjb3BlIjoiaWVfbW9kZWwiLCJjbGllbnRJZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMCJ9._91vgFeqkqinvP4APID_CswtX2gvlf8LbDF7kJa4CYM",
        "extra_params": {},
        "context_window": 131072,   # 128k (capped for Qwen3 SFT compatibility)
    },
    "qwen3_30b_v2_0314_unmasked_lr2e5": {
        "api_type": "openai",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "model": "/data/Qwen3-30B-A3B-sft-v2-0314-unmasked-lr2e5-hf",
        "api_key": "dummy",
        "extra_params": {},
        "context_window": 131072,
    },
    "sonnet": {
        "api_type": "openai",
        "api_url": "https://api.gmi-serving.com/v1/chat/completions",
        "model": "anthropic/claude-sonnet-4.6",
        "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjEyZjE2YThiLTY4N2ItNDNlMC1iZmI0LTkyNzZjNTRjYTBjNSIsInNjb3BlIjoiaWVfbW9kZWwiLCJjbGllbnRJZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMCJ9._91vgFeqkqinvP4APID_CswtX2gvlf8LbDF7kJa4CYM",
        "extra_params": {},
        "context_window": 131072,   # 128k (capped for Qwen3 SFT compatibility)
    },
    "gemini3": {
        "api_type": "google_genai",
        "model": "gemini-3-flash-preview",
        "api_key": "AIzaSyBIBu5vpapXTg4gFsMawhCu3GzYjDgLrgw",
        "extra_params": {},
        "context_window": 1048576,  # 1M
    },
}

# Active model config — set in main() from --model flag
_cfg = MODEL_CONFIGS["qwen3"]
_TEMPERATURE = 0.0  # Sampling temperature — set in main() from --temperature flag

ARCHIPELAGO_ROOT = "/data/mingye_b200-1/archipelago/mcp_servers"
EVAL_DIR = "/data/home/mingye/apex_eval"
RESULTS_FILE = os.path.join(EVAL_DIR, "results.jsonl")
LOGS_DIR = os.path.join(EVAL_DIR, "logs")

HF_SNAPSHOT = "/data/apex-agents"

MAX_STEPS = 100
MAX_TOOL_OUTPUT_CHARS = 50000
TOOL_CALL_TIMEOUT = 60       # per-tool timeout (seconds), matches loop agent
LLM_RESPONSE_TIMEOUT = 600   # per-LLM-call timeout (seconds)
TASK_TIMEOUT = 10800          # overall task timeout (3 hours), matches loop agent

# Retry config (matches loop agent: 10 retries, base_backoff=5, jitter=5)
MAX_RETRIES = 10
RETRY_BASE_BACKOFF = 5
RETRY_JITTER = 5
RETRIABLE_STATUS_CODES = {400, 408, 429, 500, 502, 503, 529}

SYSTEM_PROMPT = (
    "You are an agent that completes tasks independently.\n"
    "Use the tools provided to you to complete the task to the best of your ability."
)

# ── MCP Server Registry ─────────────────────────────────────────────────────

MCP_SERVERS = {
    "filesystem": {
        "path": "filesystem/mcp_servers/filesystem_server",
        "extra_paths": ["filesystem/packages/mcp_schema"],
        "tools": [
            "tools.list_files:list_files",
            "tools.read_text_file:read_text_file",
            "tools.read_image_file:read_image_file",
            "tools.get_directory_tree:get_directory_tree",
            "tools.search_files:search_files",
            "tools.get_file_metadata:get_file_metadata",
        ],
    },
    "spreadsheets": {
        "path": "spreadsheets/mcp_servers/sheets_server",
        "extra_paths": ["spreadsheets/packages/mcp_schema"],
        "tools": [
            "tools.create_spreadsheet:create_spreadsheet",
            "tools.delete_spreadsheet:delete_spreadsheet",
            "tools.read_tab:read_tab",
            "tools.read_csv:read_csv",
            "tools.list_tabs_in_spreadsheet:list_tabs_in_spreadsheet",
            "tools.add_tab:add_tab",
            "tools.delete_tab:delete_tab",
            "tools.edit_spreadsheet:edit_spreadsheet",
            "tools.add_content_text:add_content_text",
            "tools.delete_content_cell:delete_content_cell",
            "tools.create_chart:create_chart",
            "tools.filter_tab:filter_tab",
        ],
    },
    "code": {
        "path": "code/mcp_servers/code_execution_server",
        "extra_paths": ["code/packages/mcp_schema"],
        "tools": [
            "tools.code_exec:code_exec",
        ],
    },
    "calendar": {
        "path": "calendar/mcp_servers/calendar_server",
        "extra_paths": ["calendar/packages/mcp_schema"],
        "tools": [
            "tools.list_events:list_events",
            "tools.read_event:read_event",
            "tools.create_event:create_event",
            "tools.update_event:update_event",
            "tools.delete_event:delete_event",
        ],
    },
    "chat": {
        "path": "chat/mcp_servers/chat_server",
        "extra_paths": ["chat/packages/mcp_schema"],
        "tools": [
            "tools.list_channels:list_channels",
            "tools.get_channel_history:get_channel_history",
            "tools.get_thread_replies:get_thread_replies",
            "tools.get_user_profile:get_user_profile",
            "tools.get_users:get_users",
            "tools.post_message:post_message",
            "tools.reply_to_thread:reply_to_thread",
            "tools.add_reaction:add_reaction",
            "tools.delete_post:delete_post",
        ],
    },
    "mail": {
        "path": "mail/mcp_servers/mail_server",
        "extra_paths": ["mail/packages/mcp_schema"],
        "tools": [
            "tools.list_mails:list_mails",
            "tools.read_mail:read_mail",
            "tools.search_mail:search_mail",
            "tools.send_mail:send_mail",
            "tools.reply_mail:reply_mail",
            "tools.reply_all_mail:reply_all_mail",
            "tools.forward_mail:forward_mail",
        ],
    },
    "pdfs": {
        "path": "pdfs/mcp_servers/pdf_server",
        "extra_paths": ["pdfs/packages/mcp_schema"],
        "tools": [
            "tools.create_pdf:create_pdf",
            "tools.read_pdf_pages:read_pdf_pages",
            "tools.read_image:read_image",
            "tools.read_page_as_image:read_page_as_image",
            "tools.search_pdf:search_pdf",
        ],
    },
    "documents": {
        "path": "documents/mcp_servers/docs_server",
        "extra_paths": ["documents/packages/mcp_schema"],
        "tools": [
            "tools.create_document:create_document",
            "tools.delete_document:delete_document",
            "tools.get_document_overview:get_document_overview",
            "tools.read_document_content:read_document_content",
            "tools.read_image:read_image",
            "tools.add_content_text:add_content_text",
            "tools.edit_content_text:edit_content_text",
            "tools.delete_content_text:delete_content_text",
            "tools.add_image:add_image",
            "tools.modify_image:modify_image",
            "tools.apply_formatting:apply_formatting",
            "tools.header_footer:header_footer",
            "tools.page_margins:page_margins",
            "tools.page_orientation:page_orientation",
            "tools.comments:comments",
        ],
    },
    "presentations": {
        "path": "presentations/mcp_servers/slides_server",
        "extra_paths": ["presentations/packages/mcp_schema"],
        "tools": [
            "tools.create_slides:create_deck",
            "tools.delete_slides:delete_deck",
            "tools.add_slide:add_slide",
            "tools.edit_slides:edit_slides",
            "tools.add_image:add_image",
            "tools.modify_image:modify_image",
            "tools.insert_chart:insert_chart",
            "tools.insert_table:insert_table",
            "tools.add_shape:add_shape",
            "tools.read_slides:read_slides",
            "tools.read_completedeck:read_completedeck",
            "tools.read_individualslide:read_individualslide",
            "tools.read_image:read_image",
        ],
    },
}


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_dataset():
    """Load tasks and world descriptions from the HuggingFace cache."""
    tasks_path = os.path.join(HF_SNAPSHOT, "tasks_and_rubrics.json")
    worlds_path = os.path.join(HF_SNAPSHOT, "world_descriptions.json")

    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(worlds_path) as f:
        worlds_list = json.load(f)

    worlds = {w["world_id"]: w for w in worlds_list}
    return tasks, worlds


def download_world(world_id):
    """Download and extract a world zip if not already done."""
    world_dir = os.path.join(EVAL_DIR, world_id)
    if os.path.isdir(world_dir) and os.listdir(world_dir):
        return world_dir

    zip_name = f"{world_id}.zip"
    zip_path = os.path.join(HF_SNAPSHOT, "world_files_zipped", zip_name)

    # Download from HuggingFace if not in cache
    if not os.path.exists(zip_path):
        print(f"  Downloading {zip_name} from HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="mercor/apex-agents",
            filename=f"world_files_zipped/{zip_name}",
            repo_type="dataset",
        )

    # Extract
    print(f"  Extracting {zip_name}...")
    os.makedirs(world_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(world_dir)

    return world_dir



def download_task_files(task_id):
    """Download task-specific files from HuggingFace if they exist."""
    # Check if task_files exist in HF cache
    task_files_dir = os.path.join(HF_SNAPSHOT, "task_files", task_id)
    if os.path.isdir(task_files_dir):
        return task_files_dir

    # Try to download
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="mercor/apex-agents",
            repo_type="dataset",
            allow_patterns=[f"task_files/{task_id}/*"],
        )
        if os.path.isdir(task_files_dir):
            return task_files_dir
    except Exception as e:
        print(f"  Warning: Could not download task files for {task_id}: {e}")

    return None


def setup_task_filesystem(world_dir, task_id, task_input_files):
    """
    Create an isolated per-task working directory by deep-copying the world's
    filesystem/ and .apps_data/ directories, then overlaying any task-specific
    files on top.  This prevents cross-task contamination when multiple tasks
    share the same world.

    Returns (fs_root, apps_data_root) paths.  Caller should call
    cleanup_task_filesystem(task_id) after the task completes.
    """
    task_work_root = os.path.join(EVAL_DIR, "task_workdirs", task_id)
    if os.path.exists(task_work_root):
        shutil.rmtree(task_work_root)
    os.makedirs(task_work_root, exist_ok=True)

    # ── Deep-copy world filesystem ──
    world_fs = os.path.join(world_dir, "filesystem")
    task_fs_dir = os.path.join(task_work_root, "filesystem")
    if os.path.isdir(world_fs):
        shutil.copytree(world_fs, task_fs_dir)
    else:
        os.makedirs(task_fs_dir, exist_ok=True)

    # ── Deep-copy world .apps_data ──
    world_apps = os.path.join(world_dir, ".apps_data")
    task_apps_dir = os.path.join(task_work_root, ".apps_data")
    if os.path.isdir(world_apps):
        shutil.copytree(world_apps, task_apps_dir)
    else:
        os.makedirs(task_apps_dir, exist_ok=True)

    # ── Overlay task-specific files ──
    if task_input_files:
        task_files_dir = download_task_files(task_id)
        if task_files_dir:
            task_fs_source = os.path.join(task_files_dir, "filesystem")
            if os.path.isdir(task_fs_source):
                for item in os.listdir(task_fs_source):
                    src = os.path.join(task_fs_source, item)
                    dst = os.path.join(task_fs_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

            task_apps_source = os.path.join(task_files_dir, ".apps_data")
            if os.path.isdir(task_apps_source):
                for item in os.listdir(task_apps_source):
                    src = os.path.join(task_apps_source, item)
                    dst = os.path.join(task_apps_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

    return task_fs_dir, task_apps_dir


def cleanup_task_filesystem(task_id):
    """Remove the per-task working directory to free disk space."""
    task_work_root = os.path.join(EVAL_DIR, "task_workdirs", task_id)
    if os.path.isdir(task_work_root):
        shutil.rmtree(task_work_root, ignore_errors=True)


# ── Schema Flattening ────────────────────────────────────────────────────────

def _flatten_schema(schema):
    """Flatten JSON schema by resolving $ref and removing $defs."""
    if not isinstance(schema, dict):
        return schema

    schema = schema.copy()
    defs = schema.pop("$defs", {})

    def resolve_refs(obj, visiting=None):
        if visiting is None:
            visiting = set()
        if not isinstance(obj, dict):
            return obj

        if "$ref" in obj:
            ref_path = obj["$ref"]
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                if def_name in visiting:
                    result = {"type": "object"}
                    for key, value in obj.items():
                        if key != "$ref":
                            result[key] = resolve_refs(value, visiting)
                    return result
                if def_name in defs:
                    visiting.add(def_name)
                    try:
                        definition = defs[def_name]
                        if isinstance(definition, bool):
                            resolved = {} if definition else {"not": {}}
                        else:
                            resolved = resolve_refs(definition.copy(), visiting)
                    finally:
                        visiting.discard(def_name)
                    for key, value in obj.items():
                        if key != "$ref":
                            resolved[key] = resolve_refs(value, visiting)
                    return resolved
            result = {"type": "object"}
            for key, value in obj.items():
                if key != "$ref":
                    result[key] = resolve_refs(value, visiting)
            return result

        result = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                result[key] = resolve_refs(value, visiting)
            elif isinstance(value, list):
                result[key] = [
                    resolve_refs(item, visiting) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    return resolve_refs(schema)


def _flatten_schema_for_gemini(schema):
    """Comprehensive JSON Schema flattener for Gemini Function Calling.

    Gemini only accepts: type, properties, required, items, enum, description,
    nullable, format.  This function inlines $ref, collapses anyOf/oneOf/allOf,
    and strips every unsupported keyword.
    """
    if not isinstance(schema, dict):
        return schema

    from copy import deepcopy

    _UNSUPPORTED = frozenset({
        "$defs", "$ref", "default", "title", "additionalProperties", "const",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minItems", "maxItems", "minLength", "maxLength", "pattern",
        "uniqueItems", "examples", "prefixItems", "discriminator",
    })

    def _type_name(t):
        if "type" in t:
            return str(t["type"])
        ref = t.get("$ref")
        if isinstance(ref, str) and "/" in ref:
            return ref.split("/")[-1]
        return "unknown"

    def _clean(obj, defs, seen, exclude=None):
        skip = _UNSUPPORTED | (exclude or set())
        result = {}
        for k, v in obj.items():
            if k in skip:
                continue
            if k == "properties" and isinstance(v, dict):
                result[k] = {p: _resolve(s, defs, seen) for p, s in v.items()}
            else:
                result[k] = _resolve(v, defs, seen)
        return result

    def _resolve(obj, defs, seen):
        if isinstance(obj, list):
            return [_resolve(i, defs, seen) for i in obj]
        if not isinstance(obj, dict):
            return obj

        if "$defs" in obj:
            defs = {**(defs or {}), **obj["$defs"]}

        # $ref — inline
        ref = obj.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/") and defs:
            ref_key = ref.split("/")[-1]
            if ref_key in defs:
                if ref_key in seen:
                    return {"type": "object", "description": f"(recursive: {ref_key})"}
                resolved = _resolve(deepcopy(defs[ref_key]), defs, seen | {ref_key})
                for k, v in obj.items():
                    if k not in _UNSUPPORTED and k != "$ref":
                        resolved[k] = _resolve(v, defs, seen)
                return resolved

        # anyOf / oneOf — pick first non-null variant
        for union_key in ("anyOf", "oneOf"):
            variants = obj.get(union_key)
            if not isinstance(variants, list) or len(variants) == 0:
                continue
            non_null = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
            if len(non_null) == 0:
                result = _clean(obj, defs, seen, exclude={union_key})
                result.setdefault("type", "string")
                return result
            if len(non_null) == 1:
                result = _clean(obj, defs, seen, exclude={union_key})
                result.update(_resolve(non_null[0], defs, seen))
                return result
            # Multiple non-null variants — collapse to first, add description
            field_desc = obj.get("description")
            result = _clean(obj, defs, seen, exclude={union_key})
            result.update(_resolve(non_null[0], defs, seen))
            names = [_type_name(v) for v in non_null]
            note = f"(one of: {', '.join(names)})"
            result["description"] = f"{field_desc} {note}" if field_desc else note
            return result

        # allOf — merge
        all_of = obj.get("allOf")
        if isinstance(all_of, list) and len(all_of) > 0:
            merged = _clean(obj, defs, seen, exclude={"allOf"})
            for sub in all_of:
                rs = _resolve(sub, defs, seen)
                if isinstance(rs, dict):
                    if "properties" in rs and "properties" in merged:
                        merged["properties"].update(rs.pop("properties"))
                    merged.update(rs)
            return merged

        # Default — recurse, strip unsupported
        prefix_items = obj.get("prefixItems")
        result = _clean(obj, defs, seen)
        if result.get("type") == "array" and "items" not in result:
            if isinstance(prefix_items, list) and len(prefix_items) > 0:
                result["items"] = _resolve(prefix_items[0], defs, seen)
            else:
                result["items"] = {"type": "string"}
        return result

    return _resolve(schema, None, set())


# ── Dynamic Tool Loading ─────────────────────────────────────────────────────

def load_tool_schemas():
    """Load tool schemas (OpenAI format) once. Returns list of tool dicts."""
    from fastmcp import FastMCP

    os.environ.setdefault("APP_FS_ROOT", "/tmp")
    os.environ["USE_INDIVIDUAL_TOOLS"] = "true"

    mcp = FastMCP("schema-extractor")
    loaded_servers = []

    for server_name, config in MCP_SERVERS.items():
        server_path = os.path.join(ARCHIPELAGO_ROOT, config["path"])
        extra_paths = [os.path.join(ARCHIPELAGO_ROOT, p) for p in config.get("extra_paths", [])]

        paths_to_add = [server_path] + extra_paths
        original_path = sys.path[:]
        original_modules = set(sys.modules.keys())
        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)

        server_tools = []
        for tool_spec in config["tools"]:
            module_path, func_name = tool_spec.split(":")
            try:
                mod = __import__(module_path, fromlist=[func_name])
                fn = getattr(mod, func_name)
                # Register with prefixed name to avoid collisions
                prefixed_name = f"{server_name}_{func_name}"
                mcp.tool(fn, name=prefixed_name)
                server_tools.append(prefixed_name)
            except Exception as e:
                print(f"  WARN: Schema load failed for {server_name}.{func_name}: {e}")

        # Restore sys.path and purge server-local modules
        sys.path[:] = original_path
        generic_prefixes = ("models", "utils", "tools", "middleware")
        new_modules = set(sys.modules.keys()) - original_modules
        for mod_name in new_modules:
            mod_obj = sys.modules.get(mod_name)
            if mod_obj is None:
                continue
            if mod_name.split(".")[0] in generic_prefixes:
                del sys.modules[mod_name]
                continue
            mod_file = getattr(mod_obj, "__file__", "") or ""
            mod_paths = getattr(mod_obj, "__path__", []) or []
            all_locations = [mod_file] + list(mod_paths)
            if any(d in loc for d in paths_to_add for loc in all_locations if loc):
                del sys.modules[mod_name]

        if server_tools:
            loaded_servers.append(f"{server_name}({len(server_tools)})")

    print(f"  Loaded servers: {', '.join(loaded_servers)}")

    async def _extract():
        tools_list = await mcp.list_tools()
        result = []
        for tool in tools_list:
            params = _flatten_schema(tool.parameters)
            # Gemini needs additional flattening (oneOf, anyOf, title, etc.)
            if _cfg["api_type"] == "google_genai":
                params = _flatten_schema_for_gemini(params)
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                },
            })
        return result

    tools_openai = asyncio.run(_extract())
    print(f"  Total tools: {len(tools_openai)}")
    return tools_openai


def load_tool_functions(fs_root, apps_data_root):
    """
    Load tool function implementations with correct filesystem paths.
    Must be called per-task to ensure config modules pick up the right paths.

    Returns (tool_functions, tool_server_paths) where tool_server_paths maps
    each tool name to the sys.path entries needed for its lazy imports.
    """
    # Set environment variables BEFORE importing modules
    os.environ["APP_FS_ROOT"] = fs_root
    os.environ["APP_APPS_DATA_ROOT"] = apps_data_root
    os.environ["USE_INDIVIDUAL_TOOLS"] = "true"

    tool_functions = {}
    tool_server_paths = {}

    for server_name, config in MCP_SERVERS.items():
        server_path = os.path.join(ARCHIPELAGO_ROOT, config["path"])
        extra_paths = [os.path.join(ARCHIPELAGO_ROOT, p) for p in config.get("extra_paths", [])]

        paths_to_add = [server_path] + extra_paths
        original_path = sys.path[:]
        original_modules = set(sys.modules.keys())
        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)

        for tool_spec in config["tools"]:
            module_path, func_name = tool_spec.split(":")
            try:
                mod = __import__(module_path, fromlist=[func_name])
                fn = getattr(mod, func_name)
                prefixed_name = f"{server_name}_{func_name}"
                tool_functions[prefixed_name] = fn
                tool_server_paths[prefixed_name] = paths_to_add
            except Exception as e:
                pass  # Silently skip — schema loading already warned

        # Restore sys.path and purge modules for next server
        sys.path[:] = original_path
        generic_prefixes = ("models", "utils", "tools", "middleware")
        new_modules = set(sys.modules.keys()) - original_modules
        for mod_name in new_modules:
            mod_obj = sys.modules.get(mod_name)
            if mod_obj is None:
                continue
            if mod_name.split(".")[0] in generic_prefixes:
                del sys.modules[mod_name]
                continue
            mod_file = getattr(mod_obj, "__file__", "") or ""
            mod_paths = getattr(mod_obj, "__path__", []) or []
            all_locations = [mod_file] + list(mod_paths)
            if any(d in loc for d in paths_to_add for loc in all_locations if loc):
                del sys.modules[mod_name]

    return tool_functions, tool_server_paths


# ── Tool Execution ───────────────────────────────────────────────────────────

class _ToolTimeout(Exception):
    pass


def _tool_timeout_handler(signum, frame):
    raise _ToolTimeout()


def _coerce_pydantic_args(fn, arguments):
    """If fn expects a Pydantic BaseModel parameter, construct it from the dict."""
    from pydantic import BaseModel
    sig = inspect.signature(fn)
    for param_name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            continue
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            # LLM sent {"request": {...}} — construct the model
            raw = arguments.get(param_name, arguments)
            if isinstance(raw, dict):
                return {param_name: ann(**raw)}
    return arguments


def _setup_server_context(name, tool_server_paths):
    """Temporarily add the tool's server paths to sys.path and purge stale generic modules."""
    paths = tool_server_paths.get(name, [])
    if not paths:
        return []
    generic_prefixes = ("models", "utils", "tools", "middleware")
    for mod_name in list(sys.modules.keys()):
        if mod_name.split(".")[0] in generic_prefixes:
            del sys.modules[mod_name]
    added = []
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    return added


def _teardown_server_context(added_paths):
    """Remove temporarily added paths from sys.path."""
    for p in added_paths:
        if p in sys.path:
            sys.path.remove(p)


def execute_tool(name, arguments, tool_functions, fs_root, tool_server_paths=None):
    """Execute a tool call with a per-tool timeout (matches loop agent)."""
    if tool_server_paths is None:
        tool_server_paths = {}

    if name == "code_code_exec":
        return _local_code_exec(arguments, fs_root)

    fn = tool_functions.get(name)
    if fn is None:
        return f"Unknown tool: {name}"

    # Coerce dict args into Pydantic models if needed
    try:
        arguments = _coerce_pydantic_args(fn, arguments)
    except Exception as e:
        return f"Tool argument error for '{name}': {type(e).__name__}: {e}"

    # Restore server context for lazy imports
    added_paths = _setup_server_context(name, tool_server_paths)

    # Set per-tool timeout via SIGALRM (matches loop agent's tool_call_timeout)
    old_handler = signal.signal(signal.SIGALRM, _tool_timeout_handler)
    signal.alarm(TOOL_CALL_TIMEOUT)
    try:
        if asyncio.iscoroutinefunction(fn):
            result = asyncio.run(fn(**arguments))
        else:
            result = fn(**arguments)
    except _ToolTimeout:
        return f"Tool call timed out after {TOOL_CALL_TIMEOUT}s"
    except TypeError as e:
        return f"Tool argument error for '{name}': {e}\nArgs: {json.dumps(arguments, default=str)[:500]}"
    except Exception as e:
        return f"Tool error for '{name}': {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        _teardown_server_context(added_paths)

    result = str(result) if result is not None else "(no output)"
    if len(result) > MAX_TOOL_OUTPUT_CHARS:
        result = result[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [truncated, {len(result)} total chars]"
    return result


def _local_code_exec(arguments, fs_root):
    """Local code_exec via subprocess."""
    if "request" in arguments:
        req = arguments["request"]
        cmd = req.get("code") if isinstance(req, dict) else None
    else:
        cmd = arguments.get("code")

    if not cmd:
        return "Error: Required parameter 'code' (command to execute)"

    # Detect inline python and write to temp file
    inline_py = re.match(
        r"""^(python3?)\s+-c\s+(['"])(.*)\2\s*$""", cmd.strip(), re.DOTALL
    )
    if inline_py:
        py_bin, _, py_code = inline_py.groups()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir="/tmp", delete=False, prefix="agent_"
        ) as f:
            f.write(py_code)
            tmp_path = f.name
        cmd = f"{py_bin} {tmp_path}"
    else:
        first_line = cmd.strip().split("\n")[0].strip()
        python_indicators = ("import ", "from ", "def ", "class ", "print(", "# ", "if __name__")
        if first_line.startswith(python_indicators):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir="/tmp", delete=False, prefix="agent_"
            ) as f:
                f.write(cmd)
                tmp_path = f.name
            cmd = f"python3 {tmp_path}"

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=300, cwd=fs_root,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 300 seconds"
    except Exception as e:
        return f"Error: {e}"


# ── Streaming API Call ────────────────────────────────────────────────────────

def _estimate_token_count(messages, tools=None):
    """Rough token count estimate for messages + tool defs.

    Uses ~3.2 chars per token (conservative for multilingual/code content).
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block.get("text", "")))
        elif isinstance(content, str):
            total_chars += len(content)
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            total_chars += len(fn.get("name", ""))
            total_chars += len(fn.get("arguments", ""))
        total_chars += len(msg.get("reasoning", "") or "")
    if tools:
        total_chars += len(json.dumps(tools))
    return int(total_chars / 3.2)


def _truncate_single_message(msg, max_chars=80000):
    """Truncate an individual message's content if it exceeds max_chars."""
    content = msg.get("content", "")
    if isinstance(content, str) and len(content) > max_chars:
        msg = dict(msg)
        msg["content"] = content[:max_chars] + f"\n... [truncated from {len(content)} chars to {max_chars}]"
    return msg


def _truncate_messages_for_api(messages, tools=None):
    """Truncate message history to fit within context_window.

    1. Truncate any oversized individual messages (tool responses, etc.)
    2. Preserves the system prompt and initial user message. Removes the oldest
       conversation turns (assistant + its tool responses) first. Inserts a short
       notice so the model knows earlier context was omitted.
    """
    context_window = _cfg.get("context_window")
    if not context_window:
        return messages

    max_output_tokens = 8192
    max_input_tokens = context_window - max_output_tokens

    # Step 1: Truncate oversized individual messages (except system prompt)
    # This prevents a single huge tool response from blowing up context
    max_msg_chars = 80000  # ~25k tokens — generous but bounded
    messages = [messages[0]] + [_truncate_single_message(m, max_msg_chars) for m in messages[1:]]

    if _estimate_token_count(messages, tools) <= max_input_tokens:
        return messages

    if len(messages) <= 2:
        return messages

    prefix = messages[:2]   # system + initial user prompt
    middle = messages[2:]

    # Group middle messages into turns.  Each turn starts with an assistant
    # or user message and includes all following tool-role messages.
    turns = []
    current_turn = []
    for msg in middle:
        role = msg.get("role", "")
        if role in ("assistant", "user") and current_turn:
            turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)

    # Drop oldest turns until the remaining messages fit
    removed = 0
    while len(turns) > 1:
        flat = [m for t in turns for m in t]
        if _estimate_token_count(prefix + flat, tools) <= max_input_tokens:
            break
        turns.pop(0)
        removed += 1

    if removed > 0:
        flat = [m for t in turns for m in t]
        notice = {
            "role": "user",
            "content": (f"[Note: {removed} earlier conversation turns were "
                        f"omitted to fit the {context_window} token context window]"),
        }
        result = prefix + [notice] + flat
        print(f"  [Context window] Removed {removed} oldest turns to fit "
              f"{context_window} token limit "
              f"(~{_estimate_token_count(result, tools)} tokens remaining)")
        return result

    return messages


def _is_context_window_error(status_code, error_body):
    """Detect context window errors that should NOT be retried."""
    patterns = [
        "token count exceeds", "context_length_exceeded", "context length exceeded",
        "maximum context length", "maximum number of tokens", "prompt is too long",
        "input too long", "exceeds the model's maximum context",
    ]
    body_lower = (error_body or "").lower()
    return any(p in body_lower for p in patterns)


def call_api(messages, tools, temperature=0.0):
    """Dispatch to OpenAI-compatible or Anthropic API based on active config.

    Retries up to MAX_RETRIES times on transient errors with exponential
    backoff + jitter, matching the loop agent's retry behavior.

    Automatically truncates messages to fit within the configured context_window.
    """
    messages = _truncate_messages_for_api(messages, tools)
    for attempt in range(MAX_RETRIES + 1):
        try:
            if _cfg["api_type"] == "anthropic":
                return _call_api_anthropic(messages, tools, temperature)
            elif _cfg["api_type"] == "openai_responses":
                return _call_api_openai_responses(messages, tools, temperature)
            elif _cfg["api_type"] == "google_genai":
                return _call_api_google_genai(messages, tools, temperature)
            return _call_api_openai(messages, tools, temperature)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            body = ""
            try:
                body = e.response.text[:2000] if e.response is not None else ""
            except Exception:
                pass
            if status == 400 and ("context length" in body or "input_tokens" in body or "maximum" in body.lower()):
                raise  # never retry context window errors
            if status not in RETRIABLE_STATUS_CODES:
                raise  # non-retriable HTTP error
            if attempt >= MAX_RETRIES:
                raise
            backoff = RETRY_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, RETRY_JITTER)
            print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] HTTP {status}, waiting {backoff:.1f}s...")
            time.sleep(backoff)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt >= MAX_RETRIES:
                raise
            backoff = RETRY_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, RETRY_JITTER)
            print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] {type(e).__name__}, waiting {backoff:.1f}s...")
            time.sleep(backoff)
        except Exception as e:
            # Catch genai SDK errors and other transient errors
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "500" in err_str or "503" in err_str or "overloaded" in err_str:
                if attempt >= MAX_RETRIES:
                    raise
                backoff = RETRY_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, RETRY_JITTER)
                print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] {type(e).__name__}: {str(e)[:200]}, waiting {backoff:.1f}s...")
                time.sleep(backoff)
            else:
                raise


def _flatten_tool_content(messages):
    """Convert structured tool content blocks to plain strings for OpenAI API."""
    out = []
    for msg in messages:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
            flat = copy.copy(msg)
            texts = [b["text"] for b in msg["content"] if b.get("type") == "text"]
            flat["content"] = "\n".join(texts) if texts else ""
            out.append(flat)
        else:
            out.append(msg)
    return out


def _repair_and_parse_tool_call_json(raw_text):
    """Try to parse tool call JSON with progressive repairs for common model errors.

    Returns parsed dict on success, None on failure.
    Common issues: unbalanced braces, invalid escapes (\\( \\'), extra braces.
    """
    import re as _re_repair

    text = raw_text.strip()
    if not text:
        return None

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix unbalanced braces
    open_count = text.count('{')
    close_count = text.count('}')
    fixed = text
    if open_count > close_count:
        fixed = text + '}' * (open_count - close_count)
    elif close_count > open_count:
        # Remove extra trailing braces
        excess = close_count - open_count
        tmp = list(fixed)
        for i in range(len(tmp) - 1, -1, -1):
            if tmp[i] == '}' and excess > 0:
                tmp.pop(i)
                excess -= 1
                if excess == 0:
                    break
        fixed = ''.join(tmp)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: fix invalid JSON escape sequences
    # JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX
    # Models often produce \' \( \) \| etc. in shell commands
    def _fix_escapes(s):
        result = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                next_ch = s[i + 1]
                if next_ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
                    result.append(s[i:i+2])
                    i += 2
                elif next_ch == 'u' and i + 5 < len(s):
                    result.append(s[i:i+6])
                    i += 6
                else:
                    # Invalid escape: double the backslash
                    result.append('\\\\')
                    result.append(next_ch)
                    i += 2
            else:
                result.append(s[i])
                i += 1
        return ''.join(result)

    fixed_esc = _fix_escapes(fixed)
    try:
        return json.loads(fixed_esc)
    except json.JSONDecodeError:
        pass

    # Attempt 4: extract just the name via regex and use empty arguments
    name_m = _re_repair.search(r'"name"\s*:\s*"([^"]+)"', text)
    if name_m:
        return {"name": name_m.group(1), "arguments": {}}

    return None


def _call_api_openai(messages, tools, temperature=0.0):
    headers = {
        "Authorization": f"Bearer {_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _cfg["model"],
        "messages": _flatten_tool_content(messages),
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # For local SFT models, tool definitions are injected into the system
    # prompt directly; don't pass tools to the API to avoid needing vLLM's
    # tool call parser (which doesn't work well with this SFT model).
    if not _cfg.get("api_url", "").startswith("http://localhost"):
        payload["tools"] = tools
    payload.update(_cfg["extra_params"])

    resp = requests.post(_cfg["api_url"], headers=headers, json=payload, stream=True, timeout=600)
    if resp.status_code != 200:
        try:
            error_body = resp.text[:2000]
        except Exception:
            error_body = "(could not read error body)"
        print(f"  [API Error] Status {resp.status_code}: {error_body}")
        resp.raise_for_status()

    content_parts, reasoning_parts = [], []
    tool_calls_map = {}
    finish_reason = None
    usage = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Usage comes in the final chunk with stream_options.include_usage
        if chunk.get("usage"):
            usage = chunk["usage"]

        choices = chunk.get("choices") or [{}]
        choice = choices[0] if choices else {}
        delta = choice.get("delta", {})
        fr = choice.get("finish_reason")
        if fr:
            finish_reason = fr

        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])

        for tc in (delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            if idx not in tool_calls_map:
                tool_calls_map[idx] = {
                    "id": tc.get("id", f"call_{idx}"),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            fn = tc.get("function", {})
            if fn.get("name"):
                tool_calls_map[idx]["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]
            if tc.get("id"):
                tool_calls_map[idx]["id"] = tc["id"]

    # Fallback: parse tool calls from content/reasoning if structured tool_calls is empty.
    # Some SFT models emit <tool_call> JSON in text instead of structured tool_calls.
    parsed_tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else None
    raw_content = "".join(content_parts) or None
    raw_reasoning = "".join(reasoning_parts) or None

    # Also treat structured tool_calls with empty function names as missing
    has_valid_tool_calls = parsed_tool_calls and any(
        tc.get("function", {}).get("name") for tc in parsed_tool_calls
    )
    if not has_valid_tool_calls:
        parsed_tool_calls = None
        text_to_parse = (raw_content or "") + (raw_reasoning or "")
        if "<tool_call>" in text_to_parse:
            import re as _re
            # Match tool calls with or without closing tags
            _tc_pattern = _re.compile(
                r'<tool_call>\s*(.*?)(?:\s*</tool_call>|\Z)',
                _re.DOTALL,
            )
            fallback_tcs = []
            for i, m in enumerate(_tc_pattern.finditer(text_to_parse)):
                tc_json = _repair_and_parse_tool_call_json(m.group(1))
                if tc_json and "name" in tc_json:
                    fallback_tcs.append({
                        "id": f"call_fallback_{i}",
                        "type": "function",
                        "function": {
                            "name": tc_json["name"],
                            "arguments": json.dumps(tc_json.get("arguments", {})),
                        },
                    })
            if fallback_tcs:
                parsed_tool_calls = fallback_tcs
                raw_content = None  # tool calls were in content, clear it
                finish_reason = "tool_calls"

    return {
        "content": raw_content,
        "reasoning": raw_reasoning,
        "tool_calls": parsed_tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def _call_api_anthropic(messages, tools, temperature=0.0, max_tokens=32768):
    """Call Anthropic Messages API, converting from OpenAI message format."""
    # ── Convert messages ──
    system_text = ""
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system_text = msg["content"]
            continue

        if role == "tool":
            # Flatten structured content blocks to string for Anthropic tool_result
            raw = msg["content"]
            if isinstance(raw, list):
                content_str = "\n".join(b["text"] for b in raw if b.get("type") == "text")
            else:
                content_str = raw
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": content_str,
            }
            # Merge consecutive tool_result blocks into one user message
            if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                last_content = anthropic_messages[-1]["content"]
                if isinstance(last_content, list):
                    last_content.append(tool_result)
                    continue
            anthropic_messages.append({"role": "user", "content": [tool_result]})
            continue

        if role == "assistant":
            content_blocks = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                try:
                    input_data = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    input_data = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": input_data,
                })
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue

        # user
        anthropic_messages.append({"role": msg.get("role", "user"), "content": msg["content"]})

    # ── Convert tool schemas ──
    anthropic_tools = []
    for tool in tools:
        fn = tool["function"]
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    if anthropic_tools:
        anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}

    # ── Add cache_control to last user message ──
    if anthropic_messages:
        last_user = None
        for m in reversed(anthropic_messages):
            if m["role"] == "user":
                last_user = m
                break
        if last_user:
            content = last_user["content"]
            if isinstance(content, list) and content:
                content[-1]["cache_control"] = {"type": "ephemeral"}
            elif isinstance(content, str):
                last_user["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]

    # ── Build request ──
    extra = _cfg.get("extra_params", {})
    thinking_config = extra.get("thinking")
    headers = {
        "x-api-key": _cfg["api_key"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    if extra.get("anthropic_beta"):
        headers["anthropic-beta"] = extra["anthropic_beta"]
    payload = {
        "model": _cfg["model"],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if extra.get("speed"):
        payload["speed"] = extra["speed"]
    if thinking_config:
        payload["thinking"] = thinking_config
        payload["temperature"] = 1  # required when thinking is enabled
    else:
        payload["temperature"] = temperature
    if system_text:
        payload["system"] = [
            {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
        ]
    if anthropic_messages:
        payload["messages"] = anthropic_messages
    if anthropic_tools:
        payload["tools"] = anthropic_tools

    resp = requests.post(_cfg["api_url"], headers=headers, json=payload, stream=True, timeout=600)
    if resp.status_code != 200:
        try:
            error_body = resp.text[:2000]
        except Exception:
            error_body = "(could not read error body)"
        print(f"  [API Error] Status {resp.status_code}: {error_body}")
        resp.raise_for_status()

    # ── Parse Anthropic streaming SSE ──
    content_parts = []
    thinking_parts = []
    tool_calls_list = []
    current_block_type = None
    current_tool_id = ""
    current_tool_name = ""
    current_tool_input = ""
    stop_reason = None
    usage = {"input_tokens": 0, "output_tokens": 0}

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type", "")

        if msg_type == "message_start":
            msg_usage = data.get("message", {}).get("usage", {})
            usage["input_tokens"] = msg_usage.get("input_tokens", 0)
            usage["output_tokens"] = msg_usage.get("output_tokens", 0)
            usage["cache_read_input_tokens"] = msg_usage.get("cache_read_input_tokens", 0)
            usage["cache_creation_input_tokens"] = msg_usage.get("cache_creation_input_tokens", 0)

        elif msg_type == "content_block_start":
            block = data.get("content_block", {})
            current_block_type = block.get("type")
            if current_block_type == "tool_use":
                current_tool_id = block.get("id", "")
                current_tool_name = block.get("name", "")
                current_tool_input = ""

        elif msg_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                content_parts.append(delta.get("text", ""))
            elif delta_type == "input_json_delta":
                current_tool_input += delta.get("partial_json", "")
            elif delta_type == "thinking_delta":
                thinking_parts.append(delta.get("thinking", ""))

        elif msg_type == "content_block_stop":
            if current_block_type == "tool_use":
                tool_calls_list.append({
                    "id": current_tool_id,
                    "type": "function",
                    "function": {
                        "name": current_tool_name,
                        "arguments": current_tool_input or "{}",
                    },
                })
            current_block_type = None

        elif msg_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason", stop_reason)
            delta_usage = data.get("usage", {})
            if delta_usage.get("output_tokens"):
                usage["output_tokens"] = delta_usage["output_tokens"]

        elif msg_type == "message_stop":
            break

    # Map Anthropic stop_reason → OpenAI finish_reason
    if stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif stop_reason == "end_turn":
        finish_reason = "stop"
    else:
        finish_reason = stop_reason

    # Log cache usage
    if usage:
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        if cache_read or cache_create:
            print(f"  [Cache] read={cache_read}, created={cache_create}")

    return {
        "content": "".join(content_parts) or None,
        "reasoning": "".join(thinking_parts) or None,
        "tool_calls": tool_calls_list if tool_calls_list else None,
        "finish_reason": finish_reason,
        "usage": usage,
    }


_genai_client = None


def _call_api_google_genai(messages, tools, temperature=0.0):
    """Call Google Gemini via native genai SDK (handles thought signatures automatically)."""
    from google import genai
    from google.genai import types

    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=_cfg["api_key"])

    # Convert OpenAI tool definitions to genai FunctionDeclarations
    func_decls = []
    for tool in tools:
        fn = tool["function"]
        func_decls.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {"type": "object", "properties": {}}),
        ))
    genai_tools = [types.Tool(function_declarations=func_decls)]

    # Convert OpenAI messages to genai Contents
    contents = []
    system_instruction = None
    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system_instruction = msg["content"]
            continue

        if role == "user":
            content = msg["content"]
            if isinstance(content, list):
                texts = [b["text"] for b in content if b.get("type") == "text"]
                content = "\n".join(texts)
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            continue

        if role == "assistant":
            parts = []
            # Reconstruct thinking parts (Gemini thought signatures require these)
            for tp in msg.get("_thinking_parts", []):
                parts.append(types.Part(text=tp, thought=True))
            if msg.get("content"):
                parts.append(types.Part(text=msg["content"]))
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                try:
                    args = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    args = {}
                part_kwargs = {}
                sig = tc.get("_thought_signature")
                if sig:
                    # Decode base64 back to bytes if needed
                    if tc.get("_thought_signature_is_bytes"):
                        import base64
                        sig = base64.b64decode(sig)
                    part_kwargs["thought_signature"] = sig
                parts.append(types.Part(
                    function_call=types.FunctionCall(name=fn["name"], args=args),
                    **part_kwargs,
                ))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue

        if role == "tool":
            raw = msg["content"]
            if isinstance(raw, list):
                result_str = "\n".join(b["text"] for b in raw if b.get("type") == "text")
            else:
                result_str = str(raw)
            contents.append(types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(
                    name=msg.get("name", ""),
                    response={"result": result_str},
                ))
            ]))
            continue

    # Make the API call
    config = types.GenerateContentConfig(
        tools=genai_tools,
        temperature=temperature,
        max_output_tokens=32768,
    )
    if system_instruction:
        config.system_instruction = system_instruction

    resp = _genai_client.models.generate_content(
        model=_cfg["model"],
        contents=contents,
        config=config,
    )

    # Parse response into standard format, preserving thought signatures
    content_parts = []
    thinking_parts = []
    tool_calls_list = []
    tc_idx = 0

    if resp.candidates and resp.candidates[0].content:
        for part in resp.candidates[0].content.parts:
            if part.text:
                if getattr(part, "thought", False):
                    thinking_parts.append(part.text)
                else:
                    content_parts.append(part.text)
            if part.function_call:
                tc_entry = {
                    "id": f"call_{tc_idx}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args) if part.function_call.args else {}),
                    },
                }
                # Preserve thought_signature — Gemini requires it on subsequent turns
                sig = getattr(part, "thought_signature", None)
                if sig:
                    # Store as base64 string if bytes, for JSON serialization
                    if isinstance(sig, bytes):
                        import base64
                        tc_entry["_thought_signature"] = base64.b64encode(sig).decode("ascii")
                        tc_entry["_thought_signature_is_bytes"] = True
                    else:
                        tc_entry["_thought_signature"] = sig
                tool_calls_list.append(tc_entry)
                tc_idx += 1

    # Extract usage
    usage = None
    if resp.usage_metadata:
        um = resp.usage_metadata
        pt = um.prompt_token_count or 0
        ct = um.candidates_token_count or 0
        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

    return {
        "content": "".join(content_parts) or None,
        "reasoning": "".join(thinking_parts) or None,
        "_thinking_parts": thinking_parts if thinking_parts else None,
        "tool_calls": tool_calls_list if tool_calls_list else None,
        "finish_reason": "tool_calls" if tool_calls_list else "stop",
        "usage": usage,
    }


def _call_api_openai_responses(messages, tools, temperature=0.0, max_tokens=32768):
    """Call OpenAI Responses API, converting from internal Chat Completions message format."""
    # ── Convert messages to Responses API input items ──
    instructions = ""
    input_items = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            instructions = msg["content"]
            continue

        if role == "user":
            content = msg["content"]
            if isinstance(content, list):
                texts = [b["text"] for b in content if b.get("type") == "text"]
                content = "\n".join(texts)
            input_items.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            if msg.get("content"):
                input_items.append({"role": "assistant", "content": msg["content"]})
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                input_items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": fn["name"],
                    "arguments": fn["arguments"],
                })
            continue

        if role == "tool":
            raw = msg["content"]
            if isinstance(raw, list):
                content_str = "\n".join(b["text"] for b in raw if b.get("type") == "text")
            else:
                content_str = str(raw)
            input_items.append({
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": content_str,
            })
            continue

    # ── Convert tool schemas ──
    responses_tools = []
    for tool in tools:
        fn = tool["function"]
        responses_tools.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })

    # ── Build request ──
    headers = {
        "Authorization": f"Bearer {_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _cfg["model"],
        "input": input_items,
        "max_output_tokens": max_tokens,
        "stream": True,
    }
    if not _cfg.get("no_temperature"):
        payload["temperature"] = temperature
    if instructions:
        payload["instructions"] = instructions
    if responses_tools:
        payload["tools"] = responses_tools

    extra = _cfg.get("extra_params", {})
    for k, v in extra.items():
        if k not in payload:
            payload[k] = v

    resp = requests.post(
        _cfg["api_url"], headers=headers, json=payload,
        stream=True, timeout=600
    )
    if resp.status_code != 200:
        try:
            error_body = resp.text[:2000]
        except Exception:
            error_body = "(could not read error body)"
        print(f"  [API Error] Status {resp.status_code}: {error_body}")
        resp.raise_for_status()

    # ── Parse streaming SSE ──
    content_parts = []
    reasoning_parts = []
    tool_calls = []
    current_fc = {}
    usage = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = data.get("type", "")

        if event_type == "response.output_text.delta":
            content_parts.append(data.get("delta", ""))

        elif event_type == "response.function_call_arguments.delta":
            idx = data.get("output_index", 0)
            if idx in current_fc:
                current_fc[idx]["arguments"] += data.get("delta", "")

        elif event_type == "response.output_item.added":
            item = data.get("item", {})
            idx = data.get("output_index", 0)
            if item.get("type") == "function_call":
                current_fc[idx] = {
                    "call_id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "arguments": "",
                }

        elif event_type == "response.output_item.done":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "") or "{}",
                    },
                })
            elif item.get("type") == "reasoning":
                for block in item.get("content", []):
                    if block.get("type") == "thinking":
                        reasoning_parts.append(block.get("thinking", ""))

        elif event_type == "response.completed":
            resp_data = data.get("response", {})
            u = resp_data.get("usage", {})
            if u:
                usage = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }
            break

    if tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    return {
        "content": "".join(content_parts) or None,
        "reasoning": "".join(reasoning_parts) or None,
        "tool_calls": tool_calls if tool_calls else None,
        "finish_reason": finish_reason,
        "usage": usage,
    }


# ── Agent Loop (per task) ────────────────────────────────────────────────────

def run_single_task(task, tools_openai, tool_functions, fs_root, tool_server_paths=None):
    """
    Run the agent on a single task.  Matches the loop agent behaviour:
    - 100 max steps, 3-hour overall timeout
    - Retries on transient API errors (handled inside call_api)
    - Per-tool 60 s timeout (handled inside execute_tool)
    - Empty/invalid LLM response → append "continue" (matches loop agent)
    """
    task_prompt = task["prompt"]
    system_content = SYSTEM_PROMPT

    # For local SFT models: inject tool definitions into system prompt
    # (mimics the Qwen3 chat template's <tools> injection) so we don't
    # need to pass tools via the API and can avoid vLLM's tool parser.
    if _cfg.get("api_url", "").startswith("http://localhost"):
        tools_json = "\n".join(json.dumps(t) for t in tools_openai)
        system_content += (
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call>"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task_prompt},
    ]

    total_tool_calls = 0
    tool_call_log = []
    usage_log = []  # per-step usage
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 5
    task_start = time.time()

    for step in range(MAX_STEPS):
        # Overall task timeout (matches loop agent's 3-hour timeout)
        if time.time() - task_start > TASK_TIMEOUT:
            return {
                "final_answer": None,
                "steps": step,
                "tool_calls": total_tool_calls,
                "tool_call_log": tool_call_log,
                "usage_log": usage_log,
                "total_usage": total_usage,
                "messages": messages,
                "error": f"Task timed out after {TASK_TIMEOUT}s",
            }

        t0 = time.time()
        try:
            response = call_api(messages, tools_openai, temperature=_TEMPERATURE)
        except Exception as e:
            return {
                "final_answer": None,
                "steps": step + 1,
                "tool_calls": total_tool_calls,
                "tool_call_log": tool_call_log,
                "usage_log": usage_log,
                "total_usage": total_usage,
                "messages": messages,
                "error": f"API Error: {e}",
            }
        elapsed = time.time() - t0

        # Track token usage from this step
        step_usage = response.get("usage")
        if step_usage:
            # Normalize: OpenAI uses prompt_tokens/completion_tokens, Anthropic uses input_tokens/output_tokens
            pt = step_usage.get("prompt_tokens") or step_usage.get("input_tokens", 0)
            ct = step_usage.get("completion_tokens") or step_usage.get("output_tokens", 0)
            normalized = {"step": step + 1, "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
            usage_log.append(normalized)
            total_usage["prompt_tokens"] += pt
            total_usage["completion_tokens"] += ct
            total_usage["total_tokens"] += pt + ct

        # Handle empty/invalid response: append "continue" (matches loop agent)
        if not response["tool_calls"] and not response["content"]:
            consecutive_empty += 1
            print(f"  [Step {step + 1}] Empty response, sending 'continue' ({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY})")
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"  Early termination: {MAX_CONSECUTIVE_EMPTY} consecutive empty responses")
                return {
                    "final_answer": None,
                    "steps": step + 1,
                    "tool_calls": total_tool_calls,
                    "tool_call_log": tool_call_log,
                    "usage_log": usage_log,
                    "total_usage": total_usage,
                    "messages": messages,
                    "error": f"Early termination: {MAX_CONSECUTIVE_EMPTY} consecutive empty responses",
                }
            messages.append({"role": "user", "content": "continue"})
            continue
        else:
            consecutive_empty = 0

        # For local SFT models: use raw <tool_call> XML in content and
        # <tool_response> in user messages (matching the Qwen3 chat template).
        use_xml_tool_format = _cfg.get("api_url", "").startswith("http://localhost")

        if response["tool_calls"]:

            if use_xml_tool_format:
                # Build assistant content with <tool_call> XML tags
                tc_xml_parts = []
                # Preserve reasoning in <think> tags to match training format
                if response["reasoning"]:
                    tc_xml_parts.append(f"<think>\n{response['reasoning']}\n</think>\n")
                if response["content"]:
                    tc_xml_parts.append(response["content"])
                for tc in response["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args_str = tc["function"]["arguments"]
                    try:
                        fn_args = json.loads(fn_args_str)
                        args_json = json.dumps(fn_args)
                    except json.JSONDecodeError:
                        args_json = fn_args_str
                    tc_xml_parts.append(
                        f'<tool_call>\n{{"name": "{fn_name}", "arguments": {args_json}}}\n</tool_call>'
                    )
                assistant_msg = {"role": "assistant", "content": "\n".join(tc_xml_parts)}
                messages.append(assistant_msg)

                # Execute tools and collect results as <tool_response> in a user message
                tool_response_parts = []
                for tc in response["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args_str = tc["function"]["arguments"]
                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    result = execute_tool(fn_name, fn_args, tool_functions, fs_root, tool_server_paths)
                    total_tool_calls += 1

                    tool_call_log.append({
                        "step": step + 1, "tool": fn_name,
                        "args": fn_args, "result_length": len(result),
                        "result_preview": result[:500],
                    })
                    tool_response_parts.append(f"<tool_response>\n{result}\n</tool_response>")

                messages.append({
                    "role": "user",
                    "content": "\n".join(tool_response_parts),
                })
            else:
                assistant_msg = {"role": "assistant", "tool_calls": response["tool_calls"]}
                if response["content"]:
                    assistant_msg["content"] = response["content"]
                if response["reasoning"]:
                    assistant_msg["reasoning"] = response["reasoning"]
                if response.get("_thinking_parts"):
                    assistant_msg["_thinking_parts"] = response["_thinking_parts"]
                messages.append(assistant_msg)

                for tc in response["tool_calls"]:
                    tc_id = tc["id"]
                    fn_name = tc["function"]["name"]
                    fn_args_str = tc["function"]["arguments"]

                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    result = execute_tool(fn_name, fn_args, tool_functions, fs_root, tool_server_paths)
                    total_tool_calls += 1

                    tool_call_log.append({
                        "step": step + 1, "tool": fn_name,
                        "args": fn_args, "result_length": len(result),
                        "result_preview": result[:500],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": fn_name,
                        "content": [{"type": "text", "text": result}],
                    })
        else:
            # No tool calls, content present → normally task complete.
            # But first check: if the content contains <tool_call> tags that
            # the streaming parser / fallback regex failed to extract, parse
            # and execute them here instead of treating them as final answer.
            final = response["content"]

            if final and "<tool_call>" in final:
                import re as _re2
                _tc_re2 = _re2.compile(
                    r'<tool_call>\s*(.*?)(?:\s*</tool_call>|\Z)',
                    _re2.DOTALL,
                )
                recovered_tcs = []
                for m in _tc_re2.finditer(final):
                    tc_json = _repair_and_parse_tool_call_json(m.group(1))
                    if tc_json and "name" in tc_json:
                        recovered_tcs.append(tc_json)

                if recovered_tcs:
                    print(f"  [Step {step + 1}] Recovered {len(recovered_tcs)} tool call(s) from content")
                    # Process as tool calls instead of final answer
                    if use_xml_tool_format:
                        # Build assistant message preserving the raw content
                        assistant_msg = {"role": "assistant", "content": final}
                        if response["reasoning"]:
                            assistant_msg["reasoning"] = response["reasoning"]
                        messages.append(assistant_msg)

                        tool_response_parts = []
                        for tc_json in recovered_tcs:
                            fn_name = tc_json["name"]
                            fn_args = tc_json.get("arguments", {})

                            result = execute_tool(fn_name, fn_args, tool_functions, fs_root, tool_server_paths)
                            total_tool_calls += 1

                            tool_call_log.append({
                                "step": step + 1, "tool": fn_name,
                                "args": fn_args, "result_length": len(result),
                                "result_preview": result[:500],
                            })
                            tool_response_parts.append(f"<tool_response>\n{result}\n</tool_response>")

                        messages.append({
                            "role": "user",
                            "content": "\n".join(tool_response_parts),
                        })
                        continue  # next step in the agent loop
                    else:
                        # Non-XML path: use structured tool messages
                        assistant_msg = {"role": "assistant", "content": final}
                        if response["reasoning"]:
                            assistant_msg["reasoning"] = response["reasoning"]
                        messages.append(assistant_msg)

                        for i, tc_json in enumerate(recovered_tcs):
                            fn_name = tc_json["name"]
                            fn_args = tc_json.get("arguments", {})
                            tc_id = f"call_recovered_{step}_{i}"

                            result = execute_tool(fn_name, fn_args, tool_functions, fs_root, tool_server_paths)
                            total_tool_calls += 1

                            tool_call_log.append({
                                "step": step + 1, "tool": fn_name,
                                "args": fn_args, "result_length": len(result),
                                "result_preview": result[:500],
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": fn_name,
                                "content": [{"type": "text", "text": result}],
                            })
                        continue  # next step in the agent loop

            final_msg = {"role": "assistant", "content": final}
            if response["reasoning"]:
                final_msg["reasoning"] = response["reasoning"]
            messages.append(final_msg)
            return {
                "final_answer": final,
                "steps": step + 1,
                "tool_calls": total_tool_calls,
                "tool_call_log": tool_call_log,
                "usage_log": usage_log,
                "total_usage": total_usage,
                "messages": messages,
                "error": None,
            }

    return {
        "final_answer": None,
        "steps": MAX_STEPS,
        "tool_calls": total_tool_calls,
        "tool_call_log": tool_call_log,
        "usage_log": usage_log,
        "total_usage": total_usage,
        "messages": messages,
        "error": f"Reached max steps ({MAX_STEPS}) without final answer",
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_task(task, agent_result):
    """
    Evaluate the agent's answer against the gold response and rubric.
    Uses LLM judge to check each rubric criterion semantically.
    Returns a dict with evaluation details.
    """
    gold = task["gold_response"]
    answer = agent_result.get("final_answer") or ""
    rubric = task.get("rubric", [])

    # Skip file-based gold answers (snap_ references)
    if gold.startswith("snap_"):
        return {
            "evaluable": False,
            "reason": "file-based gold answer",
            "pass": None,
        }

    if not answer.strip():
        return {
            "evaluable": True,
            "rubric_results": [{"criteria": c["criteria"], "passed": False, "reasoning": "Empty answer"} for c in rubric],
            "rubric_pass_count": 0,
            "rubric_total": len(rubric),
            "rubric_score": 0.0,
            "pass": 0.0,
        }

    # Use LLM judge to check each rubric criterion
    rubric_results = []
    for criterion in rubric:
        criteria_text = criterion["criteria"]
        passed, reasoning = _llm_judge_criterion(criteria_text, answer, gold)
        rubric_results.append({
            "criteria": criteria_text,
            "passed": passed,
            "reasoning": reasoning,
        })

    rubric_pass_count = sum(1 for r in rubric_results if r["passed"])
    rubric_total = len(rubric_results)

    if rubric_total > 0:
        rubric_score = rubric_pass_count / rubric_total
    else:
        rubric_score = 0.0

    return {
        "evaluable": True,
        "rubric_results": rubric_results,
        "rubric_pass_count": rubric_pass_count,
        "rubric_total": rubric_total,
        "rubric_score": rubric_score,
        "pass": rubric_score,
    }


LLM_JUDGE_PROMPT = """You are an evaluation judge. Determine whether the AI response satisfies this criterion.

Criterion: {criterion}

AI Response: {answer}

Gold Answer: {gold}

Rules:
- If criterion says "States Yes", the response must agree/say yes, not disagree.
- For numbers, allow +-2% tolerance for large numbers, exact match for percentages.
- Response need not use exact wording, but must convey the same meaning/conclusion.

Reply with exactly one line: PASS: reason OR FAIL: reason"""

# Gemini 3 Flash via GMI Cloud (OpenAI-compatible)
GMI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjEyZjE2YThiLTY4N2ItNDNlMC1iZmI0LTkyNzZjNTRjYTBjNSIsInNjb3BlIjoiaWVfbW9kZWwiLCJjbGllbnRJZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMCJ9._91vgFeqkqinvP4APID_CswtX2gvlf8LbDF7kJa4CYM"
GMI_BASE_URL = "https://api.gmi-serving.com/v1"
GMI_MODEL = "google/gemini-3-flash-preview"
_gmi_client = None


def _get_gmi_client():
    global _gmi_client
    if _gmi_client is None:
        from openai import OpenAI
        _gmi_client = OpenAI(api_key=GMI_API_KEY, base_url=GMI_BASE_URL)
    return _gmi_client


def _llm_judge_criterion(criteria_text, answer, gold, max_retries=2):
    """Use Gemini 3 Flash to judge if a rubric criterion is satisfied.

    Returns (passed: bool, reasoning: str)
    """
    prompt = LLM_JUDGE_PROMPT.format(
        criterion=criteria_text,
        answer=answer[:8000],
        gold=gold[:4000],
    )

    client = _get_gmi_client()

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GMI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            content = resp.choices[0].message.content or ""
            content = content.strip()

            # Parse PASS/FAIL - search all lines since model may add reasoning first
            for line in content.split("\n"):
                line = line.strip()
                if line.upper().startswith("PASS"):
                    return True, line
                elif line.upper().startswith("FAIL"):
                    return False, line

            # Fallback: search anywhere in content
            content_upper = content.upper()
            if "PASS" in content_upper and "FAIL" not in content_upper:
                return True, content.split("\n")[0].strip()
            elif "FAIL" in content_upper:
                return False, content.split("\n")[0].strip()
            else:
                return False, f"Ambiguous judge response: {content[:100]}"

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return False, f"Judge error: {e}"

    return False, "Judge failed after retries"


# ── Results Management ───────────────────────────────────────────────────────

def load_existing_results():
    """Load existing results from JSONL file."""
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results[r["task_id"]] = r
    return results


def save_result(result):
    """Append a single result to the JSONL file."""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj):
    """Fallback serializer for non-standard types (bytes, etc.)."""
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    return str(obj)


def save_task_log(task_id, agent_result):
    """Save detailed task log to a separate file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{task_id}.json")
    with open(log_path, "w") as f:
        json.dump(agent_result, f, indent=2, ensure_ascii=False, default=_json_default)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch eval on APEX-Agents")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen3",
                        help="Model to use (default: qwen3)")
    parser.add_argument("--world", help="Only run tasks for a specific world ID")
    parser.add_argument("--task", help="Only run a specific task ID")
    parser.add_argument("--resume", action="store_true", help="Skip tasks with existing results")
    parser.add_argument("--retry-errors", action="store_true",
                        help="With --resume, retry tasks that had errors (skip only successful results)")
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks to run (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="List tasks without running")
    parser.add_argument("--tool-root", help="Path to MCP server tool code (default: archipelago/mcp_servers)")
    parser.add_argument("--data-root", help="Path to dataset root with tasks_and_rubrics.json and world_files_zipped/ (default: HF cache)")
    parser.add_argument("--eval-dir", help="Path to eval working directory for results, logs, and task workdirs (default: /data/apex_eval)")
    parser.add_argument("--subset", help="Path to a file with task IDs (one per line) to run only those tasks")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--api-port", type=int, help="Override the API port in model config (for parallel vLLM servers)")
    parser.add_argument("--no-eval", action="store_true", help="Skip inline LLM judge evaluation (grade separately later)")
    args = parser.parse_args()

    # Set active model config
    global _cfg, _TEMPERATURE, RESULTS_FILE, LOGS_DIR, ARCHIPELAGO_ROOT, HF_SNAPSHOT, EVAL_DIR
    _cfg = MODEL_CONFIGS[args.model]
    _TEMPERATURE = args.temperature
    if args.api_port:
        # Override the port in the API URL for parallel server setups
        import re as _re_port
        _cfg = dict(_cfg)  # make a mutable copy
        _cfg["api_url"] = _re_port.sub(r':\d+/', f':{args.api_port}/', _cfg["api_url"])
    if args.tool_root:
        ARCHIPELAGO_ROOT = args.tool_root
    if args.data_root:
        HF_SNAPSHOT = args.data_root
    if args.eval_dir:
        EVAL_DIR = args.eval_dir
    RESULTS_FILE = os.path.join(EVAL_DIR, f"results_{args.model}.jsonl")
    LOGS_DIR = os.path.join(EVAL_DIR, f"logs_{args.model}")

    print("=" * 80)
    print(f"APEX-Agents Batch Evaluation — {_cfg['model']} ({args.model})")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    tasks, worlds = load_dataset()
    print(f"  {len(tasks)} tasks across {len(worlds)} worlds")

    # Filter tasks
    if args.subset:
        with open(args.subset) as f:
            raw = f.read().strip()
        # Support both JSON array of objects and plain text (one ID per line)
        if raw.startswith("["):
            subset_data = json.loads(raw)
            subset_ids = {item["task_id"] if isinstance(item, dict) else item for item in subset_data}
        else:
            subset_ids = {line.strip() for line in raw.splitlines() if line.strip()}
        tasks = [t for t in tasks if t["task_id"] in subset_ids]
        print(f"  Subset filter: {len(subset_ids)} IDs, matched {len(tasks)} tasks")
    elif args.task:
        tasks = [t for t in tasks if t["task_id"] == args.task]
    elif args.world:
        tasks = [t for t in tasks if t["world_id"] == args.world]

    # Include all tasks — file-answer tasks will be graded via Archipelago grading
    file_tasks = [t for t in tasks if t["gold_response"].startswith("snap_")]
    print(f"  Total tasks: {len(tasks)} ({len(file_tasks)} file-answer)")

    tasks_to_run = tasks

    # Resume support
    existing_results = {}
    if args.resume:
        existing_results = load_existing_results()
        if args.retry_errors:
            # Only skip tasks with successful (non-error) results
            successful = {tid: r for tid, r in existing_results.items() if not r.get("error")}
            before = len(tasks_to_run)
            tasks_to_run = [t for t in tasks_to_run if t["task_id"] not in successful]
            print(f"  Total existing: {len(existing_results)} ({len(existing_results) - len(successful)} errors)")
            print(f"  Skipped (successful): {before - len(tasks_to_run)}")
        else:
            before = len(tasks_to_run)
            tasks_to_run = [t for t in tasks_to_run if t["task_id"] not in existing_results]
            print(f"  Skipped (already done): {before - len(tasks_to_run)}")
        print(f"  Remaining: {len(tasks_to_run)}")

    if args.max_tasks > 0:
        tasks_to_run = tasks_to_run[:args.max_tasks]

    if args.dry_run:
        print(f"\nModel config:")
        print(f"  api_type: {_cfg['api_type']}")
        print(f"  api_url:  {_cfg['api_url']}")
        print(f"  model:    {_cfg['model']}")
        print(f"\nDry run — {len(tasks_to_run)} tasks would be evaluated:")
        for t in tasks_to_run:
            has_files = "+" if t.get("task_input_files") else " "
            print(f"  {t['task_id']} [{t['domain']}] {has_files} {t['prompt'][:80]}...")
        return

    # Load tool schemas once
    print("\nLoading tool schemas...")
    tools_openai = load_tool_schemas()

    # Group tasks by world for efficient processing
    from collections import defaultdict
    tasks_by_world = defaultdict(list)
    for t in tasks_to_run:
        tasks_by_world[t["world_id"]].append(t)

    print(f"\nRunning {len(tasks_to_run)} tasks across {len(tasks_by_world)} worlds...")
    print(f"Results: {RESULTS_FILE}")
    print(f"Logs:    {LOGS_DIR}/")

    # Track results
    total_run = 0
    task_scores = []  # per-task rubric scores for macro-averaging
    total_error = 0
    total_skipped = 0

    for world_id, world_tasks in tasks_by_world.items():
        world_name = worlds.get(world_id, {}).get("world_name", world_id)
        print(f"\n{'=' * 70}")
        print(f"World: {world_name} ({world_id})")
        print(f"Tasks: {len(world_tasks)}")
        print(f"{'=' * 70}")

        # Download and extract world files
        try:
            world_dir = download_world(world_id)
        except Exception as e:
            print(f"  ERROR: Could not set up world: {e}")
            for t in world_tasks:
                result = {
                    "task_id": t["task_id"],
                    "world_id": world_id,
                    "domain": t["domain"],
                    "error": f"World setup failed: {e}",
                    "pass": False,
                }
                save_result(result)
                total_error += 1
            continue

        for task_idx, task in enumerate(world_tasks):
            task_id = task["task_id"]
            total_run += 1

            print(f"\n  ── Task {task_idx + 1}/{len(world_tasks)} "
                  f"(overall {total_run}/{len(tasks_to_run)}) ──")
            print(f"  ID:     {task_id}")
            print(f"  Domain: {task['domain']}")
            print(f"  Prompt: {task['prompt'][:100]}...")

            # Setup per-task filesystem
            try:
                fs_root, apps_data_root = setup_task_filesystem(
                    world_dir, task_id, task.get("task_input_files")
                )
            except Exception as e:
                print(f"  ERROR: Filesystem setup failed: {e}")
                result = {
                    "task_id": task_id, "world_id": world_id,
                    "domain": task["domain"], "error": str(e), "pass": False,
                }
                save_result(result)
                total_error += 1
                continue

            # Load tool functions with correct paths
            try:
                tool_functions, tool_server_paths = load_tool_functions(fs_root, apps_data_root)
            except Exception as e:
                print(f"  ERROR: Tool loading failed: {e}")
                result = {
                    "task_id": task_id, "world_id": world_id,
                    "domain": task["domain"], "error": str(e), "pass": False,
                }
                save_result(result)
                total_error += 1
                continue

            # Save initial snapshot (before agent runs)
            task_work_root = os.path.join(EVAL_DIR, "task_workdirs", task_id)
            snapshots_dir = os.path.join(EVAL_DIR, "snapshots")
            os.makedirs(snapshots_dir, exist_ok=True)
            initial_snap_path = os.path.join(snapshots_dir, f"{task_id}_initial.zip")
            try:
                with zipfile.ZipFile(initial_snap_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(task_work_root):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            arcname = os.path.relpath(fpath, task_work_root)
                            try:
                                zf.write(fpath, arcname)
                            except (PermissionError, OSError):
                                pass
            except Exception as e:
                print(f"  Warning: Could not save initial snapshot: {e}")

            # Run the agent
            print(f"  Running agent (max {MAX_STEPS} steps)...")
            t0 = time.time()
            agent_result = run_single_task(task, tools_openai, tool_functions, fs_root, tool_server_paths)
            elapsed = time.time() - t0

            print(f"  Completed in {elapsed:.1f}s: "
                  f"{agent_result['steps']} steps, "
                  f"{agent_result['tool_calls']} tool calls")

            if agent_result.get("error"):
                print(f"  Error: {agent_result['error']}")

            if agent_result.get("final_answer"):
                ans = agent_result["final_answer"]
                print(f"  Answer: {ans[:200]}{'...' if len(ans) > 200 else ''}")

            # Evaluate
            if args.no_eval:
                eval_result = {"evaluable": False, "reason": "skipped (--no-eval)"}
                total_skipped += 1
                print(f"  Eval: SKIPPED (--no-eval)")
            else:
                eval_result = evaluate_task(task, agent_result)

                if eval_result.get("evaluable"):
                    score = eval_result["rubric_score"]
                    cp = eval_result["rubric_pass_count"]
                    ct = eval_result["rubric_total"]
                    task_scores.append(score)
                    print(f"  Eval: {cp}/{ct} criteria passed (score={score:.0%})")
                    if score < 1.0:
                        print(f"  Gold: {task['gold_response'][:200]}...")
                else:
                    total_skipped += 1
                    print(f"  Eval: SKIPPED ({eval_result.get('reason', 'unknown')})")

            # Save results
            result_record = {
                "task_id": task_id,
                "task_name": task.get("task_name", ""),
                "world_id": world_id,
                "domain": task["domain"],
                "prompt": task["prompt"][:500],
                "gold_response": task["gold_response"][:500],
                "final_answer": (agent_result.get("final_answer") or "")[:1000],
                "steps": agent_result["steps"],
                "tool_calls": agent_result["tool_calls"],
                "tools_used": sorted(set(t["tool"] for t in agent_result["tool_call_log"])),
                "elapsed_seconds": round(elapsed, 1),
                "error": agent_result.get("error"),
                "eval": eval_result,
                "pass": eval_result.get("rubric_score", 0),
            }
            save_result(result_record)
            save_task_log(task_id, agent_result)

            # Save final snapshot (ZIP of task workdir) before cleanup
            task_work_root = os.path.join(EVAL_DIR, "task_workdirs", task_id)
            snapshots_dir = os.path.join(EVAL_DIR, "snapshots")
            os.makedirs(snapshots_dir, exist_ok=True)
            final_snap_path = os.path.join(snapshots_dir, f"{task_id}_final.zip")
            try:
                with zipfile.ZipFile(final_snap_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(task_work_root):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            arcname = os.path.relpath(fpath, task_work_root)
                            try:
                                zf.write(fpath, arcname)
                            except (PermissionError, OSError):
                                pass
                print(f"  Snapshot saved: {final_snap_path}")
            except Exception as e:
                print(f"  Warning: Could not save snapshot: {e}")

            # Clean up per-task working directory to free disk space
            cleanup_task_filesystem(task_id)

            # Progress summary
            avg_score = sum(task_scores) / len(task_scores) if task_scores else 0
            print(f"\n  Progress: avg score {avg_score:.1%} over {len(task_scores)} tasks, "
                  f"{total_error} errors, {total_skipped} skipped")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"Total tasks run:     {total_run}")
    print(f"Evaluated:           {len(task_scores)}")
    print(f"Errors:              {total_error}")
    print(f"Skipped (files):     {total_skipped}")
    if task_scores:
        print(f"Avg score:           {sum(task_scores)/len(task_scores):.1%}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Detailed logs in: {LOGS_DIR}/")


if __name__ == "__main__":
    main()
