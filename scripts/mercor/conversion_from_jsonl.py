#!/usr/bin/env python3
"""
Convert APEX trajectory JSONL (with embedded logs) to Slime SFT format.

Input:  JSONL where each line has 'logs' (list of log entries) and metadata
Output: JSONL with 'messages' column in Qwen3 structured format

Reuses all conversion logic from conversion_combined.py but reads from
a JSONL file instead of a directory tree.

Usage:
    python conversion_from_jsonl.py \
        --input /data/opus_trajectories.jsonl \
        --tool-schemas /data/tool_schemas.json \
        --output /data/opus_trajectories_converted.jsonl
"""

import argparse
import json
import logging
import sys

# Import conversion logic from existing script
sys.path.insert(0, "/data")
from conversion_combined import (
    convert_trajectory,
    load_tool_schemas,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert APEX trajectory JSONL to Slime SFT format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL with 'logs' field per record",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (.jsonl)",
    )
    parser.add_argument(
        "--tool-schemas",
        default=None,
        help="Path to tool_schemas.json (MCP tool definitions).",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=0,
        help="Max trajectories to process (0 = all)",
    )
    args = parser.parse_args()

    if args.tool_schemas:
        load_tool_schemas(args.tool_schemas)

    # Read input JSONL
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.max_trajs > 0:
        records = records[: args.max_trajs]

    logger.info("Loaded %d records from %s", len(records), args.input)

    # Convert
    rows = []
    skipped = 0
    masked_steps_total = 0
    for rec in records:
        task_id = rec.get("task_id", "unknown")
        logs = rec.get("logs", [])

        if not logs:
            skipped += 1
            logger.warning("Skipped %s (no logs)", task_id)
            continue

        result = convert_trajectory(logs)
        if result is None:
            skipped += 1
            logger.warning("Skipped %s (could not convert)", task_id)
            continue

        messages = result
        masked = sum(1 for m in messages if m.get("step_loss_mask") == 0)
        masked_steps_total += masked
        rows.append({"messages": messages})

    logger.info("Converted %d trajectories (%d skipped)", len(rows), skipped)
    logger.info("Total messages with step_loss_mask=0: %d", masked_steps_total)

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

    # Print sample
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
            if msg.get("id"):
                extras.append(f"id={msg['id'][:16]}...")
            if msg.get("step_loss_mask") == 0:
                extras.append("MASKED")
            extra_str = f"  ({', '.join(extras)})" if extras else ""
            content_preview = (content[:120].replace("\n", "\\n") + "...") if content else "(empty)"
            print(f"  [{i:2d}] {role:10s} | {content_preview}{extra_str}")


if __name__ == "__main__":
    main()
