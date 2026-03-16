#!/usr/bin/env python3
"""
Convert APEX agentic trajectory logs to Slime SFT format for Qwen3-235B-A22B.

Input:  Directory of traj_*_logs.json files (APEX agent trajectory logs)
Output: JSONL with 'messages' column in Qwen3 structured format

Tool definitions are embedded directly in the system prompt string (Qwen3 native
format with <tools>...</tools> XML block), so no separate 'tools' key is needed.

  - Tool definitions   -> appended to system message 'content' as Qwen3 tools block
  - Thinking/reasoning  -> assistant message 'reasoning_content' field
  - Tool calls          -> assistant message 'tool_calls' list field (with 'id')
  - Tool results        -> role: "tool" messages with 'content' and 'tool_call_id'

Usage:
    python conversion.py \\
        --input-dir /dev-shared/mingye/apex_sft_data \\
        --output /path/to/output.jsonl

Slime training command (do NOT use --apply-chat-template for SFT):
    python3 train_async.py \\
        --rollout-function-path slime.rollout.sft_rollout.generate_rollout \\
        --prompt-data /path/to/output.jsonl \\
        --input-key messages \\
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
# Tool schemas (loaded from MCP tool_schemas.json)
# ---------------------------------------------------------------------------

# Maps tool name -> full {type: "function", function: {name, description, parameters}}
_TOOL_DEFS: dict[str, dict] = {}


def load_tool_schemas(schemas_path: str) -> None:
    """Load real MCP tool schemas from JSON file (call once at startup)."""
    with open(schemas_path) as f:
        tools = json.load(f)
    for tool in tools:
        name = tool["function"]["name"]
        _TOOL_DEFS[name] = tool
    logger.info("Loaded %d tool schemas from %s", len(_TOOL_DEFS), schemas_path)


def build_tool_definitions(tool_names: list[str]) -> list[dict]:
    """Build tool definitions for the 'tools' parameter.

    Looks up each tool name in the loaded MCP schemas.
    Falls back to a minimal name-only definition if not found.
    """
    defs = []
    for name in tool_names:
        if name in _TOOL_DEFS:
            defs.append(_TOOL_DEFS[name])
        else:
            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Use the {name} tool.",
                },
            })
    return defs


def format_tools_for_system_prompt(tools: list[dict]) -> str:
    """Format tool definitions as a Qwen3-style string to embed in the system prompt.

    Produces the <tools>...</tools> block followed by <tool_call> usage instructions,
    matching the format expected by Qwen3's chat template.
    """
    if not tools:
        return ""

    parts = [
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>"
    ]
    for tool in tools:
        parts.append(json.dumps(tool, ensure_ascii=False))
    parts.append(
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )
    return "\n".join(parts)


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
    tool_calls: list[dict] = []  # structured tool_call objects with id
    tool_results: list[tuple[str, str]] = []  # (ref, plain text result)
    final_answer_text = ""

    for entry in step_entries:
        mt = entry.get("log_extra", {}).get("message_type", "")

        if mt == "reasoning":
            reasoning_text += entry.get("log_message", "")

        elif mt == "response":
            response_text += entry.get("log_message", "")

        elif mt == "tool_call":
            name = entry.get("log_extra", {}).get("name", "")
            ref = entry.get("log_extra", {}).get("ref", "")
            payload = entry.get("log_extra", {}).get("payload", "")
            args = parse_tool_call_args(payload)
            tc: dict[str, Any] = {
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
            if ref:
                tc["id"] = ref
            tool_calls.append(tc)

        elif mt == "tool_result":
            ref = entry.get("log_extra", {}).get("ref", "")
            result_text = extract_tool_result_text(entry)
            tool_results.append((ref, result_text))

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

    # Tool results -> individual role:"tool" messages with tool_call_id
    for ref, result_text in tool_results:
        tool_msg: dict[str, Any] = {"role": "tool", "content": result_text}
        if ref:
            tool_msg["tool_call_id"] = ref
        messages.append(tool_msg)

    return messages


def convert_trajectory(entries: list[dict]) -> list[dict] | None:
    """Convert a full trajectory log into Qwen3 messages with tools in system prompt.

    Returns messages list or None if the trajectory cannot be converted.
    Tool definitions are embedded directly in the system prompt content.
    """
    system_prompt, user_task, tool_names = parse_configure(entries)

    if not user_task:
        return None

    messages: list[dict] = []

    # Build tool definitions and embed in system prompt
    tools = build_tool_definitions(tool_names) if tool_names else []
    tools_str = format_tools_for_system_prompt(tools)

    system_content = system_prompt
    if tools_str:
        system_content = (system_content + "\n\n" + tools_str) if system_content else tools_str

    if system_content:
        messages.append({"role": "system", "content": system_content})

    # User message
    messages.append({"role": "user", "content": user_task})

    # Group remaining entries by step and convert
    steps = group_by_steps(entries)
    for step_entries in steps:
        step_msgs = step_to_messages(step_entries)
        messages.extend(step_msgs)

    # Validate: must have at least one assistant turn
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    if assistant_count == 0:
        return None

    return messages


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
    parser.add_argument(
        "--tool-schemas",
        default=None,
        help="Path to tool_schemas.json (MCP tool definitions). "
             "Tool defs are embedded in the system prompt, not as a separate field.",
    )
    args = parser.parse_args()

    # Load tool schemas
    if args.tool_schemas:
        load_tool_schemas(args.tool_schemas)

    # Discover trajectory files
    pattern = os.path.join(args.input_dir, "traj_*_logs.json")
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
        try:
            with open(traj_file) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            skipped += 1
            logger.warning("Skipped %s (bad JSON: %s)", traj_file, e)
            continue

        result = convert_trajectory(entries)
        if result is None:
            skipped += 1
            logger.warning("Skipped %s (could not convert)", traj_file)
            continue

        messages = result
        rows.append({"messages": messages})

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
        print(f"\n--- Sample (first trajectory, {len(sample_msgs)} messages) ---")
        for i, msg in enumerate(sample_msgs):
            role = msg["role"]
            content = msg.get("content", "")
            extras = []
            if msg.get("reasoning_content"):
                extras.append(f"reasoning={len(msg['reasoning_content'])} chars")
            if msg.get("tool_calls"):
                tc_info = []
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    tc_id = tc.get("id", "no-id")
                    tc_info.append(f"{name}({tc_id[:12]}...)")
                extras.append(f"tool_calls=[{', '.join(tc_info)}]")
            if msg.get("tool_call_id"):
                extras.append(f"tool_call_id={msg['tool_call_id'][:16]}...")
            extra_str = f"  ({', '.join(extras)})" if extras else ""
            content_preview = (content[:120].replace("\n", "\\n") + "...") if content else "(empty)"
            print(f"  [{i:2d}] {role:10s} | {content_preview}{extra_str}")


if __name__ == "__main__":
    main()
