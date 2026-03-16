#!/usr/bin/env python3
"""
Grade eval results using the Archipelago grading system.

Reads trajectories from eval log files, constructs grading inputs, and calls
the Archipelago grading CLI for proper criterion-level evaluation.

Since we don't have Docker containers (no final snapshots), we use the world
filesystem as both initial and final snapshots. This means file-change-based
criteria will see no changes, and grading will be based on the agent's final
answer and trajectory text. This is still much more rigorous than the simple
LLM judge used during eval, because:
  - Each criterion is evaluated individually (not a single 0-10 score)
  - Proper structured grading prompt with strict evaluation standards
  - apex_v1_grade_score: pass_count / total_count

Usage:
  python grade_eval_results.py \
      --model qwen3_30b_opus_mix_lr5e5_e3 \
      --eval-dir /data/home/mingye/apex_eval_v3 \
      [--subset eval_subset_law.txt] \
      [--judge-model openai/google/gemini-3-flash-preview] \
      [--max-tasks 5]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Constants ────────────────────────────────────────────────────────────────

GRADING_DIR = "/data/home/mingye/archipelago/grading"
GRADING_VENV = "/data/home/mingye/grading_venv"
TASKS_AND_RUBRICS = "/data/home/mingye/apex-agents/tasks_and_rubrics.json"

DEFAULT_JUDGE_MODEL = "openai/google/gemini-3-flash-preview"
DEFAULT_JUDGE_API_BASE = "https://api.gmi-serving.com/v1"
DEFAULT_JUDGE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjEyZjE2YThiLTY4N2ItNDNlMC1iZmI0LTkyNzZjNTRjYTBjNSIsInNjb3BlIjoiaWVfbW9kZWwiLCJjbGllbnRJZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMCJ9._91vgFeqkqinvP4APID_CswtX2gvlf8LbDF7kJa4CYM"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    return str(obj)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
    with open(path, "a") as f:
        f.write(line)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_tasks_and_rubrics():
    """Load task metadata and rubrics."""
    with open(TASKS_AND_RUBRICS) as f:
        tasks = json.load(f)

    task_map = {}
    for t in tasks:
        task_map[t["task_id"]] = t
    return task_map


def load_trajectory(log_dir, task_id):
    """Load a saved trajectory from per-task log file."""
    log_path = os.path.join(log_dir, f"{task_id}.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        return json.load(f)


# ── Snapshot Creation ────────────────────────────────────────────────────────

def create_snapshot_zip(world_dir, output_path):
    """Create a ZIP snapshot from a world filesystem directory.

    The grading system expects files under 'filesystem/' prefix in the ZIP.
    """
    fs_dir = os.path.join(world_dir, "filesystem")
    if not os.path.isdir(fs_dir):
        # If no filesystem dir, create empty ZIP
        with zipfile.ZipFile(output_path, "w") as zf:
            pass
        return

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(fs_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, world_dir)
                try:
                    zf.write(fpath, arcname)
                except (PermissionError, OSError):
                    pass  # Skip unreadable files


def get_or_create_world_snapshot(world_id, eval_dir, cache_dir):
    """Get or create a cached ZIP snapshot for a world."""
    cache_path = os.path.join(cache_dir, f"{world_id}.zip")
    if os.path.exists(cache_path):
        return cache_path

    world_dir = os.path.join(eval_dir, world_id)
    if not os.path.isdir(world_dir):
        return None

    print(f"  Creating snapshot for {world_id}...")
    create_snapshot_zip(world_dir, cache_path)
    return cache_path


# ── Grading Config Builder ──────────────────────────────────────────────────

def build_grading_configs(task, rubric_criteria, trajectory_data, elapsed,
                          tmpdir, judge_model, judge_extra_args=None):
    """Create all JSON config files needed by the Archipelago grading CLI.

    Args:
        task: Task dict from tasks_and_rubrics.json
        rubric_criteria: List of {verifier_id, criteria} dicts
        trajectory_data: Agent result dict with 'messages', 'final_answer', etc.
        elapsed: Time elapsed in seconds (approximate)
        tmpdir: Temp directory for config files
        judge_model: LLM judge model identifier
        judge_extra_args: Optional extra args for litellm

    Returns dict of file paths keyed by config name.
    """
    paths = {}
    task_id = task["task_id"]
    world_id = task.get("world_id")

    # grading_settings.json
    grading_settings = {
        "llm_judge_model": judge_model,
        "llm_judge_extra_args": judge_extra_args,
    }
    paths["grading_settings"] = os.path.join(tmpdir, "grading_settings.json")
    with open(paths["grading_settings"], "w") as f:
        json.dump(grading_settings, f, indent=2)

    # verifiers.json — one per rubric criterion
    verifiers = []
    for i, criterion in enumerate(rubric_criteria):
        verifiers.append({
            "verifier_id": criterion["verifier_id"],
            "verifier_version": 1,
            "world_id": world_id,
            "task_id": task_id,
            "eval_config_id": "ec_output_llm",
            "verifier_values": {
                "criteria": criterion["criteria"],
                "expected_file_type": "All output (modified files and final message in console)",
                "artifacts_to_reference": [],
            },
            "verifier_index": i,
            "verifier_dependencies": None,
        })
    paths["verifiers"] = os.path.join(tmpdir, "verifiers.json")
    with open(paths["verifiers"], "w") as f:
        json.dump(verifiers, f, indent=2)

    # eval_configs.json
    eval_configs = [{
        "eval_config_id": "ec_output_llm",
        "eval_config_name": "Output LLM Verifier",
        "eval_defn_id": "output_llm",
        "eval_config_values": {},
    }]
    paths["eval_configs"] = os.path.join(tmpdir, "eval_configs.json")
    with open(paths["eval_configs"], "w") as f:
        json.dump(eval_configs, f, indent=2)

    # scoring_config.json
    scoring_config = {
        "scoring_config_id": "sc_default",
        "scoring_config_name": "Default Scoring (apex_v1)",
        "scoring_defn_id": "apex_v1_grade_score",
        "scoring_config_values": {},
    }
    paths["scoring_config"] = os.path.join(tmpdir, "scoring_config.json")
    with open(paths["scoring_config"], "w") as f:
        json.dump(scoring_config, f, indent=2)

    # trajectory.json — convert agent messages to LiteLLM format
    clean_messages = []
    for msg in trajectory_data.get("messages", []):
        clean_msg = {}
        for k, v in msg.items():
            # Skip internal fields that aren't part of the LiteLLM format
            if k.startswith("_"):
                continue
            clean_msg[k] = v
        clean_messages.append(clean_msg)

    trajectory = {
        "messages": clean_messages,
        "output": {"final_answer": trajectory_data.get("final_answer")},
        "status": "completed" if not trajectory_data.get("error") else "error",
        "time_elapsed": elapsed,
    }
    paths["trajectory"] = os.path.join(tmpdir, "trajectory.json")
    with open(paths["trajectory"], "w") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False, default=_json_default)

    return paths


# ── Grading Execution ────────────────────────────────────────────────────────

def run_grading(task_id, config_paths, initial_snap, final_snap,
                grading_dir=GRADING_DIR, grading_venv=GRADING_VENV):
    """Run the Archipelago grading CLI for a single task.

    Returns parsed grading output dict, or None on error.
    """
    output_path = os.path.join(
        os.path.dirname(config_paths["trajectory"]), "results.json"
    )

    # Use uv to run with the correct Python/venv
    cmd = [
        "uv", "run",
        "--project", grading_dir,
        "--python", os.path.join(grading_venv, "bin", "python"),
        "python", "-m", "runner.main",
        "--grading-run-id", f"grade_{task_id}",
        "--trajectory-id", f"traj_{task_id}",
        "--initial-snapshot", initial_snap,
        "--final-snapshot", final_snap,
        "--trajectory", config_paths["trajectory"],
        "--grading-settings", config_paths["grading_settings"],
        "--verifiers", config_paths["verifiers"],
        "--eval-configs", config_paths["eval_configs"],
        "--scoring-config", config_paths["scoring_config"],
        "--output", output_path,
    ]

    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = grading_venv

    try:
        result = subprocess.run(
            cmd, cwd=grading_dir,
            capture_output=True, text=True, timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: Grading timed out for {task_id}")
        return None

    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else "(no stderr)"
        print(f"    ERROR: Grading CLI exit {result.returncode}: {stderr}")
        return None

    if not os.path.exists(output_path):
        print(f"    ERROR: No output file produced for {task_id}")
        return None

    with open(output_path) as f:
        return json.load(f)


# ── Score Processing ─────────────────────────────────────────────────────────

def process_grading_output(grading_output, task, trajectory_data):
    """Convert grading output to a score record."""
    task_id = task["task_id"]
    domain = task.get("domain", "Unknown")

    grading_run_status = grading_output.get("grading_run_status", "error")
    verifier_results = grading_output.get("verifier_results", [])
    scoring_results = grading_output.get("scoring_results", {})

    final_score = scoring_results.get("final_score", 0.0)
    scoring_values = scoring_results.get("scoring_method_result_values", {})
    strict_pass = final_score >= 0.99

    # Convert verifier_results to simplified rubric_results
    rubric_results = []
    for vr in verifier_results:
        vrv = vr.get("verifier_result_values", {})
        rubric_results.append({
            "verifier_id": vr.get("verifier_id"),
            "score": vr.get("score", 0.0),
            "grade": vrv.get("judge_grade", "unknown"),
            "rationale": vrv.get("grade_rationale", ""),
            "status": vr.get("status", "ok"),
        })

    return {
        "task_id": task_id,
        "world_id": task.get("world_id"),
        "domain": domain,
        "rubric_results": rubric_results,
        "verifier_results": verifier_results,
        "rubric_score": final_score,
        "strict_pass": strict_pass,
        "grading_status": grading_run_status,
        "scoring_values": scoring_values,
        "final_answer": (trajectory_data.get("final_answer") or "")[:1000],
        "steps": trajectory_data.get("steps", 0),
        "tool_calls": trajectory_data.get("tool_calls", 0),
        "error": trajectory_data.get("error"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grade eval results using the Archipelago grading system"
    )
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., qwen3_30b_opus_mix_lr5e5_e3)")
    parser.add_argument("--eval-dir", default="/data/home/mingye/apex_eval_v3",
                        help="Eval directory containing logs and world dirs")
    parser.add_argument("--output-dir",
                        help="Output directory (default: {eval-dir}/grading_{model})")
    parser.add_argument("--subset",
                        help="File with task IDs (one per line) to grade")
    parser.add_argument("--task", help="Grade a single task by ID")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Max tasks to grade (0=all)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"LLM judge model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--judge-api-base", default=DEFAULT_JUDGE_API_BASE,
                        help="API base URL for LLM judge")
    parser.add_argument("--judge-api-key", default=DEFAULT_JUDGE_API_KEY,
                        help="API key for LLM judge")
    parser.add_argument("--grading-dir", default=GRADING_DIR,
                        help=f"Path to Archipelago grading runner")
    parser.add_argument("--grading-venv", default=GRADING_VENV,
                        help="Path to grading virtual environment")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks that already have grading results")
    parser.add_argument("--dry-run", action="store_true",
                        help="List tasks without running grading")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    log_dir = os.path.join(eval_dir, f"logs_{args.model}")
    output_dir = args.output_dir or os.path.join(eval_dir, f"grading_{args.model}")
    scores_dir = os.path.join(output_dir, "scores")
    results_path = os.path.join(output_dir, "results.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    snap_cache_dir = os.path.join(output_dir, "snapshot_cache")

    # Build judge extra args for litellm
    judge_extra_args = {}
    if args.judge_api_base:
        judge_extra_args["api_base"] = args.judge_api_base
    if args.judge_api_key:
        judge_extra_args["api_key"] = args.judge_api_key
    if not judge_extra_args:
        judge_extra_args = None

    # ── Load data ─────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"Archipelago Grading — {args.model}")
    print(f"  Eval dir:     {eval_dir}")
    print(f"  Log dir:      {log_dir}")
    print(f"  Judge model:  {args.judge_model}")
    print(f"  Output dir:   {output_dir}")
    print("=" * 80)

    if not os.path.isdir(log_dir):
        print(f"\nERROR: Log directory not found: {log_dir}")
        sys.exit(1)

    print("\nLoading tasks and rubrics...")
    task_map = load_tasks_and_rubrics()
    print(f"  Loaded {len(task_map)} tasks from {TASKS_AND_RUBRICS}")

    # Find completed tasks (have log files)
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
    completed_tasks = []
    for lf in sorted(log_files):
        task_id = lf.replace(".json", "")
        if task_id in task_map:
            completed_tasks.append(task_id)
    print(f"  Found {len(completed_tasks)} completed task logs in {log_dir}")

    # Apply filters
    if args.subset:
        with open(args.subset) as f:
            subset_ids = {line.strip() for line in f if line.strip()}
        completed_tasks = [t for t in completed_tasks if t in subset_ids]
        print(f"  After subset filter: {len(completed_tasks)} tasks")
    elif args.task:
        completed_tasks = [t for t in completed_tasks if t == args.task]

    if args.max_tasks > 0:
        completed_tasks = completed_tasks[:args.max_tasks]

    # Check rubric availability
    tasks_with_rubric = [
        t for t in completed_tasks
        if task_map[t].get("rubric") and len(task_map[t]["rubric"]) > 0
    ]
    print(f"  Tasks with rubric criteria: {len(tasks_with_rubric)}/{len(completed_tasks)}")

    tasks_to_grade = tasks_with_rubric

    if args.resume:
        existing = 0
        filtered = []
        for tid in tasks_to_grade:
            score_path = os.path.join(scores_dir, f"{tid}.json")
            if os.path.exists(score_path):
                existing += 1
            else:
                filtered.append(tid)
        print(f"  Resume: {existing} already graded, {len(filtered)} remaining")
        tasks_to_grade = filtered

    print(f"\n  Tasks to grade: {len(tasks_to_grade)}")

    if args.dry_run:
        for tid in tasks_to_grade:
            t = task_map[tid]
            n_criteria = len(t.get("rubric", []))
            print(f"  {tid} [{t['domain']}] {n_criteria} criteria")
        return

    # ── Create output dirs ────────────────────────────────────────────────
    for d in [scores_dir, snap_cache_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Grade tasks ───────────────────────────────────────────────────────
    stats = {
        "total": 0,
        "graded": 0,
        "strict_pass": 0,
        "errors": 0,
        "per_domain": defaultdict(lambda: {
            "graded": 0, "strict_pass": 0, "scores": [], "tasks": 0,
        }),
    }

    for idx, task_id in enumerate(tasks_to_grade):
        task = task_map[task_id]
        domain = task.get("domain", "Unknown")
        rubric = task.get("rubric", [])
        world_id = task.get("world_id")

        stats["total"] += 1
        stats["per_domain"][domain]["tasks"] += 1

        print(f"\n  [{idx+1}/{len(tasks_to_grade)}] {task_id} [{domain}] "
              f"({len(rubric)} criteria)")

        # Load trajectory
        traj = load_trajectory(log_dir, task_id)
        if not traj:
            print(f"    SKIP: No trajectory found")
            stats["errors"] += 1
            continue

        # Get/create world snapshot (initial state)
        snap_path = get_or_create_world_snapshot(
            world_id, eval_dir, snap_cache_dir
        )
        if not snap_path:
            print(f"    SKIP: No world directory for {world_id}")
            stats["errors"] += 1
            continue

        # Check for per-task snapshots from eval_qwen3_local_with_snapshots.py
        initial_snap = os.path.join(eval_dir, "snapshots", f"{task_id}_initial.zip")
        final_snap = os.path.join(eval_dir, "snapshots", f"{task_id}_final.zip")
        if os.path.exists(initial_snap) and os.path.exists(final_snap):
            print(f"    Using per-task snapshots (initial + final)")
            use_initial = initial_snap
            use_final = final_snap
        else:
            # Fallback: use world snapshot as both (no file diff)
            use_initial = snap_path
            use_final = snap_path

        # Build grading configs
        with tempfile.TemporaryDirectory() as tmpdir:
            elapsed = traj.get("total_usage", {}).get("total_tokens", 0) * 0.01  # rough estimate
            config_paths = build_grading_configs(
                task, rubric, traj, elapsed,
                tmpdir, args.judge_model, judge_extra_args,
            )

            # Run grading
            t0 = time.time()
            grading_output = run_grading(
                task_id, config_paths,
                initial_snap=use_initial,
                final_snap=use_final,
                grading_dir=args.grading_dir,
                grading_venv=args.grading_venv,
            )
            grading_time = time.time() - t0

        if not grading_output:
            stats["errors"] += 1
            # Save error record
            error_record = {
                "task_id": task_id,
                "world_id": world_id,
                "domain": domain,
                "rubric_score": 0.0,
                "strict_pass": False,
                "grading_status": "error",
                "steps": traj.get("steps", 0),
                "tool_calls": traj.get("tool_calls", 0),
            }
            save_json(os.path.join(scores_dir, f"{task_id}.json"), error_record)
            append_jsonl(results_path, error_record)
            continue

        # Process results
        score_record = process_grading_output(grading_output, task, traj)
        score_record["grading_time_seconds"] = round(grading_time, 1)

        # Save
        save_json(os.path.join(scores_dir, f"{task_id}.json"), score_record)
        append_jsonl(results_path, score_record)

        # Update stats
        stats["graded"] += 1
        rubric_score = score_record["rubric_score"]
        strict_pass = score_record["strict_pass"]
        ds = stats["per_domain"][domain]
        ds["graded"] += 1
        ds["scores"].append(rubric_score)

        if strict_pass:
            stats["strict_pass"] += 1
            ds["strict_pass"] += 1

        # Print results
        sv = score_record.get("scoring_values", {})
        passed_count = sv.get("passed_count", "?")
        total_count = sv.get("total_count", "?")
        status_str = "PASS" if strict_pass else f"{rubric_score:.0%}"
        print(f"    Score: {passed_count}/{total_count} criteria = {status_str} "
              f"({grading_time:.1f}s)")

        # Print per-criterion details
        for rr in score_record.get("rubric_results", []):
            grade = rr.get("grade", "?")
            criteria_text = ""
            for c in rubric:
                if c["verifier_id"] == rr.get("verifier_id"):
                    criteria_text = c["criteria"][:60]
                    break
            mark = "+" if rr.get("score", 0) >= 0.99 else "-"
            print(f"      [{mark}] {grade}: {criteria_text}...")

        # Running summary
        if stats["graded"] > 0:
            yield_rate = stats["strict_pass"] / stats["graded"] * 100
            print(f"    Running: {stats['strict_pass']}/{stats['graded']} "
                  f"strict pass ({yield_rate:.1f}%)")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("GRADING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Model:           {args.model}")
    print(f"Judge:           {args.judge_model}")
    print(f"Tasks graded:    {stats['graded']}")
    print(f"Strict pass:     {stats['strict_pass']}")
    print(f"Errors:          {stats['errors']}")

    if stats['graded'] > 0:
        yield_rate = stats['strict_pass'] / stats['graded'] * 100
        print(f"Pass rate:       {yield_rate:.1f}%")

    if stats["per_domain"]:
        print(f"\nPer-domain breakdown:")
        for dom in sorted(stats["per_domain"]):
            ds = stats["per_domain"][dom]
            if ds["graded"] > 0:
                avg = sum(ds["scores"]) / len(ds["scores"]) * 100
                pass_rate = ds["strict_pass"] / ds["graded"] * 100
            else:
                avg = 0
                pass_rate = 0
            print(f"  {dom:30s}  {ds['strict_pass']:2d}/{ds['graded']:2d} pass "
                  f"({pass_rate:5.1f}%)  avg={avg:5.1f}%")

    # Save summary
    summary = {
        "model": args.model,
        "judge_model": args.judge_model,
        "total_tasks": stats["total"],
        "graded": stats["graded"],
        "strict_pass": stats["strict_pass"],
        "errors": stats["errors"],
        "pass_rate": round(stats["strict_pass"] / stats["graded"] * 100, 2)
            if stats["graded"] > 0 else 0,
        "per_domain": {
            dom: {
                "graded": ds["graded"],
                "strict_pass": ds["strict_pass"],
                "avg_score": round(sum(ds["scores"]) / len(ds["scores"]) * 100, 2)
                    if ds["scores"] else 0,
                "pass_rate": round(ds["strict_pass"] / ds["graded"] * 100, 2)
                    if ds["graded"] > 0 else 0,
            }
            for dom, ds in stats["per_domain"].items()
        },
    }
    save_json(summary_path, summary)

    print(f"\nResults: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Scores:  {scores_dir}/")


if __name__ == "__main__":
    main()
