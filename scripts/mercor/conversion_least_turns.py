#!/usr/bin/env python3
"""
Convert APEX agentic trajectory logs to Slime SFT format for Qwen3-235B-A22B.

Input:  Directory of traj_*_logs.json files (APEX agent trajectory logs)
Output: JSONL with 'messages' + 'tools' columns in Qwen3 structured format

The output uses Qwen3's native structured fields so that apply_chat_template
handles all rendering (thinking tags, tool call tags, tool response wrapping):
  - Tool definitions   -> top-level 'tools' list (passed to apply_chat_template)
  - Thinking/reasoning  -> assistant message 'reasoning_content' field
  - Tool calls          -> assistant message 'tool_calls' list field
  - Tool results        -> role: "tool" messages with plain 'content'

Usage:
    python conversion.py \\
        --input-dir /dev-shared/mingye/apex_sft_data \\
        --output /path/to/output.jsonl

Slime training command (do NOT use --apply-chat-template for SFT):
    python3 train_async.py \\
        --rollout-function-path slime.rollout.sft_rollout.generate_rollout \\
        --prompt-data /path/to/output.jsonl \\
        --input-key messages \\
        --tool-key tools \\
        --loss-type sft_loss \\
        --loss-mask-type qwen3 \\
        --calculate-per-token-loss \\
        --disable-compute-advantages-and-returns \\
        --debug-train-only \\
        ...
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def build_tool_definitions(tool_names: list[str]) -> list[dict]:
    """Build structured tool definitions for the 'tools' parameter.

    Since the trajectory logs only store tool names (no schemas), we create
    minimal definitions. apply_chat_template will render them into the
    system prompt's <tools> block.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Use the {name} tool.",
            },
        }
        for name in tool_names
    ]


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def parse_configure(entries: list[dict]) -> tuple[str, str, list[str]]:
    """Extract system prompt, user task, and tool names from configure entries.

    Returns (system_prompt, user_task, tool_names).
    """
    system_prompt = ""
    user_task = ""
    tool_names: list[str] = []

    for entry in entries:
        mt = entry.get("log_extra", {}).get("message_type", "")
        if mt != "configure":
            continue

        msg = entry.get("log_message", "")

        # Entry with tool names list
        payload = entry.get("log_extra", {}).get("payload", None)
        if isinstance(payload, list) and payload and isinstance(payload[0], str):
            tool_names = payload
            continue

        # Entry with "System: ... User: ..." prompt
        if msg.startswith("System:") or msg.startswith("System:\n"):
            parts = msg.split("\nUser: ", 1)
            if len(parts) == 2:
                system_prompt = parts[0].replace("System:", "", 1).strip()
                user_task = parts[1].strip()
            else:
                system_prompt = msg.replace("System:", "", 1).strip()

    return system_prompt, user_task, tool_names


def group_by_steps(entries: list[dict]) -> list[list[dict]]:
    """Group log entries by steps. Returns list of steps, each a list of entries.

    Entries before the first step marker are discarded (configure/meta entries).
    """
    steps: list[list[dict]] = []
    current: list[dict] = []

    for entry in entries:
        mt = entry.get("log_extra", {}).get("message_type", "")
        if mt == "step":
            if current:
                steps.append(current)
            current = []
        elif mt in ("reasoning", "response", "tool_call", "tool_result", "final_answer"):
            current.append(entry)

    if current:
        steps.append(current)

    return steps


def parse_tool_call_args(payload: Any) -> dict:
    """Parse tool call arguments from the log payload into a dict."""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"raw": payload}
    elif isinstance(payload, dict):
        return payload
    return {}


def extract_tool_result_text(entry: dict) -> str:
    """Extract the text content from a tool_result entry."""
    payload = entry.get("log_extra", {}).get("payload", None)
    if isinstance(payload, list):
        parts = []
        for item in payload:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts) if parts else ""
    elif isinstance(payload, str):
        return payload
    return ""


def step_to_messages(step_entries: list[dict]) -> list[dict]:
    """Convert a single step's entries into a list of chat messages.

    Within a step, the pattern is:
      [reasoning?] [response?] (tool_call tool_result)* [final_answer?]

    We batch all tool calls from a step into one assistant message,
    and all their results into individual role:"tool" messages.
    """
    messages: list[dict] = []

    reasoning_text = ""
    response_text = ""
    tool_calls: list[dict] = []  # structured tool_call objects
    tool_results: list[str] = []  # plain text results
    final_answer_text = ""

    for entry in step_entries:
        mt = entry.get("log_extra", {}).get("message_type", "")

        if mt == "reasoning":
            reasoning_text += entry.get("log_message", "")

        elif mt == "response":
            response_text += entry.get("log_message", "")

        elif mt == "tool_call":
            name = entry.get("log_extra", {}).get("name", "")
            payload = entry.get("log_extra", {}).get("payload", "")
            args = parse_tool_call_args(payload)
            tool_calls.append({
                "function": {
                    "name": name,
                    "arguments": args,
                },
            })

        elif mt == "tool_result":
            result_text = extract_tool_result_text(entry)
            tool_results.append(result_text)

        elif mt == "final_answer":
            final_answer_text += entry.get("log_message", "")

    # Build assistant message
    has_reasoning = bool(reasoning_text.strip())
    visible_text = final_answer_text.strip() or response_text.strip()
    has_content = bool(visible_text) or bool(tool_calls)

    if has_reasoning or has_content:
        msg: dict[str, Any] = {"role": "assistant"}

        if has_reasoning:
            msg["reasoning_content"] = reasoning_text.strip()

        msg["content"] = visible_text

        if tool_calls:
            msg["tool_calls"] = tool_calls

        messages.append(msg)

    # Tool results -> individual role:"tool" messages
    for result_text in tool_results:
        messages.append({"role": "tool", "content": result_text})

    return messages


def convert_trajectory(entries: list[dict]) -> tuple[list[dict], list[dict]] | None:
    """Convert a full trajectory log into Qwen3 messages + tools.

    Returns (messages, tools) or None if the trajectory cannot be converted.
    """
    system_prompt, user_task, tool_names = parse_configure(entries)

    if not user_task:
        return None

    messages: list[dict] = []

    # System message (plain text, no tool defs - those go in 'tools' param)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # User message
    messages.append({"role": "user", "content": user_task})

    # Group remaining entries by step and convert
    steps = group_by_steps(entries)
    for step_entries in steps:
        # Validate: tool call count must match tool result count
        tc_count = sum(
            1 for e in step_entries
            if e.get("log_extra", {}).get("message_type") == "tool_call"
        )
        tr_count = sum(
            1 for e in step_entries
            if e.get("log_extra", {}).get("message_type") == "tool_result"
        )
        if tc_count != tr_count:
            logger.warning(
                "Skipping trajectory: step has %d tool_calls but %d tool_results",
                tc_count, tr_count,
            )
            return None

        step_msgs = step_to_messages(step_entries)
        messages.extend(step_msgs)

    # Validate: must have at least one assistant turn
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    if assistant_count == 0:
        return None

    # Build structured tool definitions
    tools = build_tool_definitions(tool_names) if tool_names else []

    return messages, tools


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert APEX trajectory logs to Slime SFT format for Qwen3."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing traj_*_logs.json files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (.jsonl)",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=0,
        help="Max trajectories to process (0 = all)",
    )
    args = parser.parse_args()

    # Discover trajectory files (nested: task_*/model_name/rollout_*.json)
    pattern = os.path.join(args.input_dir, "task_*", "*", "rollout_*.json")
    traj_files = sorted(glob.glob(pattern))
    if not traj_files:
        logger.error("No trajectory files found matching %s", pattern)
        sys.exit(1)

    if args.max_trajs > 0:
        traj_files = traj_files[: args.max_trajs]

    logger.info("Found %d trajectory files", len(traj_files))

    # Convert
    rows: list[dict] = []
    skipped = 0
    for traj_file in traj_files:
        with open(traj_file) as f:
            entries = json.load(f)

        result = convert_trajectory(entries)
        if result is None:
            skipped += 1
            logger.warning("Skipped %s (could not convert)", traj_file)
            continue

        messages, tools = result
        rows.append({"messages": messages, "tools": tools})

    logger.info(
        "Converted %d trajectories (%d skipped)", len(rows), skipped
    )

    if not rows:
        logger.error("No trajectories converted, exiting")
        sys.exit(1)

    # Write output
    output_path = args.output
    if not output_path.endswith(".jsonl"):
        output_path += ".jsonl"
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d samples to %s", len(rows), output_path)

    # Print a sample for verification
    if rows:
        sample_msgs = rows[0]["messages"]
        sample_tools = rows[0]["tools"]
        print(f"\n--- Sample (first trajectory, {len(sample_msgs)} messages, {len(sample_tools)} tools) ---")
        for i, msg in enumerate(sample_msgs):
            role = msg["role"]
            content = msg.get("content", "")
            extras = []
            if msg.get("reasoning_content"):
                extras.append(f"reasoning={len(msg['reasoning_content'])} chars")
            if msg.get("tool_calls"):
                names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                extras.append(f"tool_calls=[{', '.join(names)}]")
            extra_str = f"  ({', '.join(extras)})" if extras else ""
            content_preview = (content[:120].replace("\n", "\\n") + "...") if content else "(empty)"
            print(f"  [{i:2d}] {role:10s} | {content_preview}{extra_str}")


if __name__ == "__main__":
    main()
