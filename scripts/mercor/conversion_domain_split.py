#!/usr/bin/env python3
"""
Convert apex_sft_data_opus trajectories to per-domain JSONL files.

Domain is determined by matching the task user-prompt against
/data/mercor_training/data/tasks task.json, then looking up the
world name prefix in /data/mercor_training/data/world/worlds.

For each task:
  - Reads all rollout_*.json files
  - Selects the best rollout: minimum (error_steps / total_steps),
    break ties by fewest total steps
  - Writes separate JSONL files per domain

Usage:
    python conversion_domain_split.py \
        --input-dir /data/apex_sft_data_opus \
        --tasks-dir /data/mercor_training/data/tasks \
        --worlds-dir /data/mercor_training/data/world/worlds \
        --tool-schemas /data/mingye_b200-1/mcp_tool_schemas.json \
        --output-dir /data
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

_TOOL_DEFS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

def load_tool_schemas(schemas_path: str) -> None:
    with open(schemas_path) as f:
        tools = json.load(f)
    for tool in tools:
        name = tool["function"]["name"]
        _TOOL_DEFS[name] = tool
    logger.info("Loaded %d tool schemas from %s", len(_TOOL_DEFS), schemas_path)


def build_tool_definitions(tool_names: list[str]) -> list[dict]:
    defs = []
    for name in tool_names:
        if name in _TOOL_DEFS:
            defs.append(_TOOL_DEFS[name])
        else:
            defs.append({"type": "function", "function": {"name": name, "description": f"Use the {name} tool."}})
    return defs


def format_tools_for_system_prompt(tools: list[dict]) -> str:
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
# Error detection
# ---------------------------------------------------------------------------

def is_tool_result_error(text: str) -> bool:
    if text.startswith("Internal error:"):
        return True
    if '"success"' in text:
        try:
            j = json.loads(text)
            if j.get("success") is False:
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def parse_configure(entries: list[dict]) -> tuple[str, str, list[str]]:
    system_prompt = ""
    user_task = ""
    tool_names: list[str] = []
    for entry in entries:
        mt = entry.get("log_extra", {}).get("message_type", "")
        if mt != "configure":
            continue
        msg = entry.get("log_message", "")
        payload = entry.get("log_extra", {}).get("payload", None)
        if isinstance(payload, list) and payload and isinstance(payload[0], str):
            tool_names = payload
            continue
        if msg.startswith("System:") or msg.startswith("System:\n"):
            parts = msg.split("\nUser: ", 1)
            if len(parts) == 2:
                system_prompt = parts[0].replace("System:", "", 1).strip()
                user_task = parts[1].strip()
            else:
                system_prompt = msg.replace("System:", "", 1).strip()
    return system_prompt, user_task, tool_names


def extract_tool_result_text(entry: dict) -> str:
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


def group_by_steps(entries: list[dict]) -> list[list[dict]]:
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
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"raw": payload}
    elif isinstance(payload, dict):
        return payload
    return {}


def step_to_messages(step_entries: list[dict]) -> list[dict]:
    messages: list[dict] = []
    reasoning_text = ""
    response_text = ""
    tool_calls: list[dict] = []
    tool_results: list[tuple[str, str]] = []
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
            tc: dict[str, Any] = {"function": {"name": name, "arguments": args}}
            if ref:
                tc["id"] = ref
            tool_calls.append(tc)
        elif mt == "tool_result":
            ref = entry.get("log_extra", {}).get("ref", "")
            result_text = extract_tool_result_text(entry)
            tool_results.append((ref, result_text))
        elif mt == "final_answer":
            final_answer_text += entry.get("log_message", "")

    step_has_error = any(is_tool_result_error(text) for _, text in tool_results)

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
        if step_has_error:
            msg["step_loss_mask"] = 0
        messages.append(msg)

    for ref, result_text in tool_results:
        tool_msg: dict[str, Any] = {"role": "tool", "content": result_text}
        if ref:
            tool_msg["id"] = ref
        if step_has_error:
            tool_msg["step_loss_mask"] = 0
        messages.append(tool_msg)

    return messages


def compute_rollout_stats(entries: list[dict]) -> tuple[int, int]:
    """Return (total_steps, error_steps)."""
    steps = group_by_steps(entries)
    total_steps = len(steps)
    error_steps = 0
    for step_entries in steps:
        tool_results_in_step = [
            e for e in step_entries
            if e.get("log_extra", {}).get("message_type") == "tool_result"
        ]
        if any(is_tool_result_error(extract_tool_result_text(e)) for e in tool_results_in_step):
            error_steps += 1
    return total_steps, error_steps


def convert_trajectory(entries: list[dict]) -> list[dict] | None:
    system_prompt, user_task, tool_names = parse_configure(entries)
    if not user_task:
        return None

    messages: list[dict] = []
    tools = build_tool_definitions(tool_names) if tool_names else []
    tools_str = format_tools_for_system_prompt(tools)
    system_content = system_prompt
    if tools_str:
        system_content = (system_content + "\n\n" + tools_str) if system_content else tools_str
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_task})

    steps = group_by_steps(entries)
    for step_entries in steps:
        tc_count = sum(1 for e in step_entries if e.get("log_extra", {}).get("message_type") == "tool_call")
        tr_count = sum(1 for e in step_entries if e.get("log_extra", {}).get("message_type") == "tool_result")
        if tc_count != tr_count:
            logger.warning("Skipping trajectory: step has %d tool_calls but %d tool_results", tc_count, tr_count)
            return None
        messages.extend(step_to_messages(step_entries))

    if sum(1 for m in messages if m["role"] == "assistant") == 0:
        return None

    return messages


# ---------------------------------------------------------------------------
# Domain lookup
# ---------------------------------------------------------------------------

def build_prompt_to_domain(tasks_dir: str, worlds_dir: str) -> dict[str, str]:
    """Build a mapping from task user-prompt -> domain name."""
    # world_id -> domain prefix
    world_to_domain: dict[str, str] = {}
    for world_dir in os.listdir(worlds_dir):
        if "--(" not in world_dir:
            continue
        world_id = world_dir.split("--(")[1].rstrip(")")
        # domain is everything before '-world-'
        domain = world_dir.split("-world-")[0] if "-world-" in world_dir else world_dir.split("--(")[0]
        world_to_domain[world_id] = domain

    prompt_to_domain: dict[str, str] = {}
    for task_dir in os.listdir(tasks_dir):
        task_json = os.path.join(tasks_dir, task_dir, "task.json")
        if not os.path.exists(task_json):
            continue
        try:
            with open(task_json) as f:
                t = json.load(f)
            world_id = t.get("world_id", "")
            domain = world_to_domain.get(world_id)
            if not domain:
                continue
            msgs = t.get("task_prompt_messages", [])
            user_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "").strip()
            if user_prompt:
                prompt_to_domain[user_prompt] = domain
        except Exception:
            pass

    logger.info("Built domain lookup: %d prompts across domains: %s",
                len(prompt_to_domain),
                {d: sum(1 for v in prompt_to_domain.values() if v == d)
                 for d in set(prompt_to_domain.values())})
    return prompt_to_domain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks-dir", default="/data/mercor_training/data/tasks")
    parser.add_argument("--worlds-dir", default="/data/mercor_training/data/world/worlds")
    parser.add_argument("--tool-schemas", default=None)
    parser.add_argument("--prefix", default="apex_sft_opus", help="Output file prefix")
    args = parser.parse_args()

    if args.tool_schemas:
        load_tool_schemas(args.tool_schemas)

    prompt_to_domain = build_prompt_to_domain(args.tasks_dir, args.worlds_dir)

    task_dirs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith("task_")
    ])
    logger.info("Found %d task directories", len(task_dirs))

    domain_rows: dict[str, list[dict]] = {}
    stats = {"matched": 0, "unmatched": 0, "no_rollout": 0, "convert_fail": 0}

    for task_dir in task_dirs:
        task_path = os.path.join(args.input_dir, task_dir)
        rollout_files = sorted(glob.glob(os.path.join(task_path, "rollout_*.json")))
        if not rollout_files:
            stats["no_rollout"] += 1
            continue

        # Get user_task from first readable rollout (all rollouts share same task)
        user_task = None
        for rf in rollout_files:
            try:
                with open(rf) as f:
                    entries = json.load(f)
                _, user_task, _ = parse_configure(entries)
                if user_task:
                    break
            except Exception:
                pass

        if not user_task:
            stats["no_rollout"] += 1
            continue

        domain = prompt_to_domain.get(user_task)
        if domain is None:
            stats["unmatched"] += 1
            logger.debug("Unmatched task: %s", user_task[:80])
            continue

        stats["matched"] += 1

        # Select best rollout: min error_ratio, tie-break by fewest steps
        best_file = None
        best_error_ratio = float("inf")
        best_steps = float("inf")

        for rf in rollout_files:
            try:
                with open(rf) as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            total_steps, error_steps = compute_rollout_stats(entries)
            if total_steps == 0:
                continue

            error_ratio = error_steps / total_steps
            if error_ratio < best_error_ratio or (error_ratio == best_error_ratio and total_steps < best_steps):
                best_file = rf
                best_error_ratio = error_ratio
                best_steps = total_steps

        if best_file is None:
            stats["convert_fail"] += 1
            continue

        try:
            with open(best_file) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            stats["convert_fail"] += 1
            logger.warning("Skipped %s: %s", best_file, e)
            continue

        result = convert_trajectory(entries)
        if result is None:
            stats["convert_fail"] += 1
            logger.warning("Skipped %s (could not convert)", best_file)
            continue

        domain_rows.setdefault(domain, []).append({"messages": result})

    logger.info("Stats: %s", stats)
    for d, rows in domain_rows.items():
        logger.info("  %s: %d samples", d, len(rows))

    # Write per-domain JSONL files
    for domain, rows in domain_rows.items():
        out_path = os.path.join(args.output_dir, f"{args.prefix}_{domain.replace('-', '_')}.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d samples -> %s", len(rows), out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
