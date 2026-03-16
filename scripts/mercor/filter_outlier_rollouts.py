#!/usr/bin/env python3
"""
Filter outlier rollouts from SFT data.

For each task (identified by user message):
1. Select anchor: lowest error ratio, then fewest steps
2. Keep other rollouts that are similar quality to anchor
3. Discard outliers that are significantly worse

Thresholds use min(absolute, relative) - stricter when numbers are small.

Usage:
    python filter_outlier_rollouts.py \
        --input /data/apex_sft_opus_all_combined.jsonl \
        --output /data/apex_sft_opus_all_filtered.jsonl
"""

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_rollout(row):
    msgs = row["messages"]
    n_asst = sum(1 for m in msgs if m["role"] == "assistant")
    n_errors = sum(
        1 for m in msgs
        if m["role"] == "assistant" and m.get("step_loss_mask") == 0
    )
    n_tc = sum(
        len(m.get("tool_calls") or [])
        for m in msgs if m["role"] == "assistant"
    )
    err_ratio = n_errors / n_asst if n_asst > 0 else 0
    tools_used = set()
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tools_used.add(tc["function"]["name"])
    return {
        "n_asst": n_asst,
        "n_errors": n_errors,
        "n_tc": n_tc,
        "err_ratio": err_ratio,
        "tools_used": tools_used,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d samples", len(rows))

    # Group by task
    tasks = defaultdict(list)
    for idx, row in enumerate(rows):
        user_msg = row["messages"][1]["content"]
        metrics = analyze_rollout(row)
        tasks[user_msg].append({"idx": idx, "row": row, **metrics})

    logger.info("Unique tasks: %d", len(tasks))

    # Filter
    kept_indices = set()
    removed = []
    tasks_trimmed = 0

    for user_msg, rollout_list in tasks.items():
        # Anchor: lowest error ratio, then fewest steps
        sorted_r = sorted(rollout_list, key=lambda x: (x["err_ratio"], x["n_asst"]))
        anchor = sorted_r[0]
        kept_indices.add(anchor["idx"])

        if len(rollout_list) == 1:
            continue

        a = anchor
        task_removed = 0
        for r in sorted_r[1:]:
            # Steps: min(anchor*2, anchor+5)
            step_thresh = min(a["n_asst"] * 2, a["n_asst"] + 5)
            step_ok = r["n_asst"] <= step_thresh

            # Errors: anchor + 5
            err_ok = r["n_errors"] <= a["n_errors"] + 5

            # Error ratio: anchor + 15%
            err_ratio_ok = r["err_ratio"] <= a["err_ratio"] + 0.15

            # Tool coverage: >= 80% overlap with anchor
            if a["tools_used"] and r["tools_used"]:
                all_tools = a["tools_used"] | r["tools_used"]
                common_tools = a["tools_used"] & r["tools_used"]
                coverage_ok = len(common_tools) / len(all_tools) >= 0.80
            else:
                coverage_ok = True

            if step_ok and err_ok and err_ratio_ok and coverage_ok:
                kept_indices.add(r["idx"])
            else:
                removed.append(r)
                task_removed += 1

        if task_removed > 0:
            tasks_trimmed += 1

    kept = [r for tlist in tasks.values() for r in tlist if r["idx"] in kept_indices]

    logger.info("Kept: %d, Removed: %d (from %d tasks)", len(kept), len(removed), tasks_trimmed)

    # Quality comparison
    for label, items in [("KEPT", kept), ("REMOVED", removed)]:
        if not items:
            continue
        steps = [r["n_asst"] for r in items]
        errors = [r["n_errors"] for r in items]
        ratios = [r["err_ratio"] for r in items]
        logger.info(
            "  %s (%d): steps p50=%d/p90=%d  errors p50=%d/p90=%d  err%% p50=%.1f%%/p90=%.1f%%",
            label, len(items),
            statistics.median(steps), sorted(steps)[int(len(steps) * 0.9)],
            statistics.median(errors), sorted(errors)[int(len(errors) * 0.9)],
            statistics.median(ratios) * 100, sorted(ratios)[int(len(ratios) * 0.9)] * 100,
        )

    if args.dry_run:
        logger.info("Dry run - not writing output")
        return

    output_path = args.output
    if not output_path.endswith(".jsonl"):
        output_path += ".jsonl"

    with open(output_path, "w") as f:
        for idx, row in enumerate(rows):
            if idx in kept_indices:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d samples to %s", len(kept), output_path)


if __name__ == "__main__":
    main()
