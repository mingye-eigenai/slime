#!/usr/bin/env python3
"""
Score SFT trajectory data across 3 versions using LLM-as-judge (Deepseek).

Rubrics:
  R1. Error rate            - automatic (no LLM needed)
  R2a. Error acknowledgment - per error-recovery pair (1-3)
  R2b. Root cause diagnosis - per error-recovery pair (1-3)
  R2c. Targeted fix         - per error-recovery pair (1-3)
  R3a. Arg structure        - per tool call (1-3)
  R3b. Arg value compliance - per tool call (1-3)
  R4.  Reasoning-action alignment - per assistant turn w/ text+toolcall (1-3)
  R5.  Info reuse efficiency - per entry (1-5)
  Overall. Holistic score   - per entry (1-10)

Usage:
    python3 score_sft.py [--concurrency N] [--source original|deepseek|235b|all]
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = "https://api-web.eigenai.com/api/v1/chat/completions"
API_KEY = "<OPENAI_API_KEY>"
JUDGE_MODEL = "deepseek"

CONCURRENCY = 15

DATA_SOURCES = {
    "original": "/data/apex_sft_least_turns_p95.jsonl",
    "deepseek":  "/data/rewrite_output/apex_sft_least_turns_p95_rewrite_deepseek.jsonl",
    "235b":      "/data/rewrite_output/apex_sft_least_turns_p95_rewrite_235b.jsonl",
}
OUTPUT_DIR = "/data/score_output"

# Error detection patterns for R1 (tool-returned failures)
ERROR_PATTERNS = [
    r"validation error", r"missing required", r"type=missing",
    r"No such file", r"FileNotFoundError", r"PermissionError",
    r"Traceback \(most recent call last\)",
    r'"error":\s*true', r'"success":\s*false', r'"status":\s*"error"',
    r"Internal error:", r"Tool call timed out", r"Fatal error:",
    r"cannot open", r"command not found", r"SyntaxError",
]

# ── Data helpers ──────────────────────────────────────────────────────────────

def is_error_tool_result(content: str) -> bool:
    return any(re.search(p, content, re.IGNORECASE) for p in ERROR_PATTERNS)


def get_tool_content(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return str(content)


def get_assistant_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content) if content else ""


def get_tool_calls(msg: dict) -> list:
    tcs = msg.get("tool_calls", [])
    return tcs if isinstance(tcs, list) else []


def extract_tool_schema(system_content: str, tool_name: str) -> str:
    """Extract the schema for a specific tool from the system prompt."""
    pattern = rf'\{{[^{{}}]*"name":\s*"{re.escape(tool_name)}"[^{{}}]*\}}'
    # Try to find in <tools> block
    tools_match = re.search(r'<tools>(.*?)</tools>', system_content, re.DOTALL)
    if tools_match:
        tools_text = tools_match.group(1)
        # Find the JSON object containing this tool name
        for line in tools_text.split('\n'):
            if f'"name": "{tool_name}"' in line or f'"name":"{tool_name}"' in line:
                # Return a wider window
                idx = tools_text.find(line)
                return tools_text[max(0, idx-20):idx+500]
    return f"(schema for {tool_name} not found)"


def compute_r1(messages: list) -> dict:
    """Compute R1 error rate automatically."""
    total_tool_calls = 0
    error_turns = []
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            total_tool_calls += 1
            content = get_tool_content(m)
            if is_error_tool_result(content):
                error_turns.append(i)
    rate = len(error_turns) / total_tool_calls if total_tool_calls > 0 else 0.0
    return {
        "error_count": len(error_turns),
        "total_tool_calls": total_tool_calls,
        "error_rate": round(rate, 4),
        "error_turn_indices": error_turns,
    }


def find_recovery_pairs(messages: list, error_turn_indices: list) -> list:
    """For each error turn, find the next assistant turn (recovery turn)."""
    pairs = []
    for err_idx in error_turn_indices:
        for j in range(err_idx + 1, len(messages)):
            if messages[j]["role"] == "assistant":
                text = get_assistant_text(messages[j])
                if text.strip():  # Only if there's text to evaluate
                    pairs.append({"error_turn": err_idx, "recovery_turn": j})
                break
    return pairs


def find_r3_candidates(messages: list) -> list:
    """Find all assistant turns with tool calls for R3 scoring."""
    candidates = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            tcs = get_tool_calls(m)
            for tc in tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                if name:
                    candidates.append({"turn": i, "tool": name, "args": args, "call_id": tc.get("id", "")})
    return candidates


def find_r4_candidates(messages: list) -> list:
    """Find assistant turns that have both text AND tool calls."""
    candidates = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            text = get_assistant_text(m)
            tcs = get_tool_calls(m)
            if text.strip() and tcs:
                tool_names = [tc.get("function", {}).get("name", "") for tc in tcs if isinstance(tc, dict)]
                candidates.append({"turn": i, "text": text, "tool_names": tool_names})
    return candidates


# ── LLM API ───────────────────────────────────────────────────────────────────

async def call_api(session, messages, semaphore, max_retries=3):
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(
                    API_URL, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt * 5)
                        continue
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt * 3)
                continue
            raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_json_response(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ── Scoring prompts ───────────────────────────────────────────────────────────

R2_SYSTEM = """You are an expert evaluator of AI agent trajectories.
Score the quality of error recovery using three dimensions.
Return ONLY a JSON object, no other text."""

def build_r2_prompt(messages, pairs):
    """Build prompt to score all error-recovery pairs in one call."""
    items = []
    for p in pairs[:5]:  # Cap at 5 pairs per entry to control token count
        err_content = get_tool_content(messages[p["error_turn"]])[:400]
        rec_text = get_assistant_text(messages[p["recovery_turn"]])[:500]
        items.append({
            "error_output": err_content,
            "recovery_text": rec_text,
        })

    prompt = f"""Evaluate these {len(items)} error-recovery event(s) from an AI agent trajectory.

For each event, score:
- r2a (Error Acknowledgment): Does the assistant explicitly acknowledge the previous step failed?
  1=ignores/pretends success, 2=vaguely acknowledges, 3=explicitly states what failed
- r2b (Root Cause Diagnosis): Does the assistant correctly identify WHY it failed?
  1=no diagnosis/wrong, 2=partial/vague, 3=clear correct root cause named
- r2c (Targeted Fix): Is the corrective action directly addressing the diagnosed cause?
  1=random retry/unrelated action, 2=somewhat related fix, 3=directly targets root cause

Events:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return JSON:
{{"scores": [{{"r2a": int, "r2b": int, "r2c": int, "reasoning": "brief"}}]}}

The "scores" array must have exactly {len(items)} elements."""
    return prompt


R3_SYSTEM = """You are an expert evaluator of AI agent tool usage compliance.
Score whether tool calls correctly follow the defined schema.
Return ONLY a JSON object, no other text."""

def build_r3_prompt(system_content, candidates):
    """Build prompt to score schema compliance for tool calls."""
    # Extract relevant schemas
    items = []
    for c in candidates[:8]:  # Cap at 8 tool calls
        schema = extract_tool_schema(system_content, c["tool"])
        items.append({
            "tool": c["tool"],
            "schema_excerpt": schema[:400],
            "actual_args": c["args"][:400],
        })

    prompt = f"""Evaluate these tool calls for schema compliance.

For each call, score:
- r3a (Argument Structure): Is the argument nesting/wrapping correct per the schema?
  1=wrong structure (missing required wrapper, wrong nesting), 2=minor structural issue, 3=correct structure
- r3b (Argument Value Compliance): Are the argument values valid per the schema constraints?
  1=clearly violates a schema constraint, 2=questionable/edge case, 3=fully compliant values

Tool calls to evaluate:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return JSON:
{{"scores": [{{"tool": str, "r3a": int, "r3b": int, "reasoning": "brief"}}]}}

The "scores" array must have exactly {len(items)} elements."""
    return prompt


R4_SYSTEM = """You are an expert evaluator of AI agent reasoning quality.
Score whether the assistant's stated reasoning matches its actual actions.
Return ONLY a JSON object, no other text."""

def build_r4_prompt(candidates):
    """Build prompt to score reasoning-action alignment."""
    items = []
    for c in candidates[:6]:  # Cap at 6
        items.append({
            "reasoning_text": c["text"][:400],
            "tools_called": c["tool_names"],
        })

    prompt = f"""Evaluate reasoning-action alignment for these assistant turns.

Score r4 (Reasoning-Action Consistency):
Does the assistant's stated reasoning actually match what tool(s) it called?
1=mismatch (says "check X" but calls tool for Y, or reasoning doesn't justify the action)
2=partial match (general direction correct but specifics inconsistent)
3=full match (reasoning clearly explains and justifies the exact tool calls made)

Turns to evaluate:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return JSON:
{{"scores": [{{"r4": int, "reasoning": "brief"}}]}}

The "scores" array must have exactly {len(items)} elements."""
    return prompt


R5_SYSTEM = """You are an expert evaluator of AI agent efficiency.
Score how well the agent reuses previously gathered information.
Return ONLY a JSON object, no other text."""

def build_r5_prompt(messages):
    """Build prompt to score information reuse efficiency."""
    # Build a compact trajectory summary
    lines = []
    tool_results_seen = {}  # tool_name -> first result summary

    for i, m in enumerate(messages):
        if m["role"] == "system":
            continue
        elif m["role"] == "user":
            lines.append(f"[{i}] USER: {get_assistant_text(m)[:200]}")
        elif m["role"] == "assistant":
            text = get_assistant_text(m)
            tcs = get_tool_calls(m)
            if tcs:
                names = [tc.get("function", {}).get("name", "?") for tc in tcs if isinstance(tc, dict)]
                prefix = f" (text: {text[:100]})" if text else ""
                lines.append(f"[{i}] ASSISTANT calls: {names}{prefix}")
            elif text:
                lines.append(f"[{i}] ASSISTANT: {text[:200]}")
        elif m["role"] == "tool":
            content = get_tool_content(m)[:200]
            lines.append(f"[{i}] TOOL result: {content}")

    trajectory = "\n".join(lines[:60])  # Cap length

    prompt = f"""Evaluate information reuse efficiency for this agent trajectory.

Score r5 (Information Reuse / No Redundant Work):
Does the agent remember and use information from earlier tool results, avoiding redundant calls?
1=very poor (repeatedly queries same info, ignores previous results)
2=poor (some redundancy, occasionally forgets earlier results)
3=moderate (mostly reuses info but has some unnecessary repetition)
4=good (clearly builds on previous results, minimal redundancy)
5=excellent (perfectly efficient, every tool call is necessary and informed by prior context)

Trajectory:
{trajectory}

Return JSON:
{{"r5": int, "reasoning": "1-2 sentences explaining the score"}}"""
    return prompt


OVERALL_SYSTEM = """You are an expert evaluator of AI agent trajectory quality.
Give a holistic overall score for the complete trajectory.
Return ONLY a JSON object, no other text."""

def build_overall_prompt(messages, r1, r2_scores, r3_scores, r4_scores, r5):
    """Build prompt for holistic overall scoring."""
    # Compute averages from sub-scores
    r2_avg = None
    if r2_scores:
        r2_vals = [s for s in r2_scores if s]
        if r2_vals:
            r2_avg = round(sum((s.get("r2a",0)+s.get("r2b",0)+s.get("r2c",0))/3 for s in r2_vals) / len(r2_vals), 2)

    r3_avg = None
    if r3_scores:
        r3_vals = [s for s in r3_scores if s]
        if r3_vals:
            r3_avg = round(sum((s.get("r3a",0)+s.get("r3b",0))/2 for s in r3_vals) / len(r3_vals), 2)

    r4_avg = None
    if r4_scores:
        r4_vals = [s for s in r4_scores if s]
        if r4_vals:
            r4_avg = round(sum(s.get("r4",0) for s in r4_vals) / len(r4_vals), 2)

    # Get final answer
    final_msg = next((m for m in reversed(messages) if m["role"] == "assistant" and get_assistant_text(m).strip()), None)
    final_answer = get_assistant_text(final_msg)[:600] if final_msg else "(none)"

    user_task = get_assistant_text(messages[1]) if len(messages) > 1 else ""

    prompt = f"""Give an overall quality score (1-10) for this AI agent trajectory.

Task: {user_task[:300]}

Sub-scores computed:
- R1 Error Rate: {r1['error_rate']:.1%} ({r1['error_count']} errors / {r1['total_tool_calls']} tool calls)
- R2 Error Recovery Quality: {r2_avg if r2_avg is not None else 'N/A (no errors)'} / 3.0
- R3 Schema Compliance: {r3_avg if r3_avg is not None else 'N/A'} / 3.0
- R4 Reasoning-Action Alignment: {r4_avg if r4_avg is not None else 'N/A'} / 3.0
- R5 Info Reuse Efficiency: {r5 if r5 else 'N/A'} / 5.0

Final answer given:
{final_answer}

Score the overall trajectory quality holistically (1-10):
- 9-10: Excellent — minimal errors, perfect recovery reasoning, efficient, clear thinking
- 7-8: Good — few errors or good recovery, mostly aligned reasoning
- 5-6: Moderate — some errors with partial recovery, some redundancy
- 3-4: Poor — frequent errors, weak recovery reasoning, misaligned actions
- 1-2: Very poor — fails the task or shows no coherent reasoning

Return JSON:
{{"overall": int, "reasoning": "2-3 sentences"}}"""
    return prompt


# ── Main scoring logic ─────────────────────────────────────────────────────────

async def score_entry(session, source_name, entry_idx, entry, semaphore):
    messages = entry["messages"]
    system_content = messages[0].get("content", "") if messages and messages[0]["role"] == "system" else ""

    # R1: Automatic
    r1 = compute_r1(messages)

    # Identify candidates
    recovery_pairs = find_recovery_pairs(messages, r1["error_turn_indices"])
    r3_candidates = find_r3_candidates(messages)
    r4_candidates = find_r4_candidates(messages)

    r2_scores = []
    r3_scores = []
    r4_scores = []
    r5 = None
    overall = None
    overall_reasoning = ""

    try:
        # R2: Score error recovery (if any errors)
        if recovery_pairs:
            prompt = build_r2_prompt(messages, recovery_pairs)
            resp = await call_api(session, [
                {"role": "system", "content": R2_SYSTEM},
                {"role": "user", "content": prompt},
            ], semaphore)
            parsed = parse_json_response(resp)
            raw_scores = parsed.get("scores", [])
            for pair, score in zip(recovery_pairs[:5], raw_scores):
                r2_scores.append({
                    "error_turn": pair["error_turn"],
                    "recovery_turn": pair["recovery_turn"],
                    "r2a": score.get("r2a"),
                    "r2b": score.get("r2b"),
                    "r2c": score.get("r2c"),
                    "reasoning": score.get("reasoning", ""),
                })

        # R3: Schema compliance (sample up to 8 tool calls)
        if r3_candidates:
            prompt = build_r3_prompt(system_content, r3_candidates[:8])
            resp = await call_api(session, [
                {"role": "system", "content": R3_SYSTEM},
                {"role": "user", "content": prompt},
            ], semaphore)
            parsed = parse_json_response(resp)
            raw_scores = parsed.get("scores", [])
            for cand, score in zip(r3_candidates[:8], raw_scores):
                r3_scores.append({
                    "turn": cand["turn"],
                    "tool": cand["tool"],
                    "r3a": score.get("r3a"),
                    "r3b": score.get("r3b"),
                    "reasoning": score.get("reasoning", ""),
                })

        # R4: Reasoning-action alignment
        if r4_candidates:
            prompt = build_r4_prompt(r4_candidates[:6])
            resp = await call_api(session, [
                {"role": "system", "content": R4_SYSTEM},
                {"role": "user", "content": prompt},
            ], semaphore)
            parsed = parse_json_response(resp)
            raw_scores = parsed.get("scores", [])
            for cand, score in zip(r4_candidates[:6], raw_scores):
                r4_scores.append({
                    "turn": cand["turn"],
                    "r4": score.get("r4"),
                    "reasoning": score.get("reasoning", ""),
                })

        # R5: Info reuse
        prompt = build_r5_prompt(messages)
        resp = await call_api(session, [
            {"role": "system", "content": R5_SYSTEM},
            {"role": "user", "content": prompt},
        ], semaphore)
        parsed = parse_json_response(resp)
        r5 = parsed.get("r5")

        # Overall
        prompt = build_overall_prompt(messages, r1, r2_scores, r3_scores, r4_scores, r5)
        resp = await call_api(session, [
            {"role": "system", "content": OVERALL_SYSTEM},
            {"role": "user", "content": prompt},
        ], semaphore)
        parsed = parse_json_response(resp)
        overall = parsed.get("overall")
        overall_reasoning = parsed.get("reasoning", "")

    except Exception as e:
        print(f"  [{source_name}] Entry {entry_idx}: scoring error — {e}")

    return {
        "idx": entry_idx,
        "source": source_name,
        "r1": r1,
        "r2_scores": r2_scores,
        "r3_scores": r3_scores,
        "r4_scores": r4_scores,
        "r5": r5,
        "overall": overall,
        "overall_reasoning": overall_reasoning,
    }


async def run_source(source_name, input_path, out_dir, concurrency, max_entries=None):
    progress_path = out_dir / f"progress_{source_name}.jsonl"
    output_path = out_dir / f"scores_{source_name}.jsonl"

    # Load entries
    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    if max_entries:
        entries = entries[:max_entries]
    print(f"[{source_name}] Loaded {len(entries)} entries from {input_path}")

    # Load already-done
    done = {}
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                done[obj["idx"]] = obj
        print(f"[{source_name}] Resuming: {len(done)}/{len(entries)} already scored")

    todo = [i for i in range(len(entries)) if i not in done]
    print(f"[{source_name}] Scoring {len(todo)} entries with concurrency={concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    start = time.time()
    completed = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            score_entry(session, source_name, i, entries[i], semaphore)
            for i in todo
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            done[result["idx"]] = result

            with open(progress_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if completed % 25 == 0 or completed == len(todo):
                elapsed = time.time() - start
                rate = completed / elapsed
                remaining = (len(todo) - completed) / rate if rate > 0 else 0
                overall_vals = [v["overall"] for v in done.values() if v.get("overall")]
                avg_overall = sum(overall_vals) / len(overall_vals) if overall_vals else 0
                print(f"[{source_name}] {len(done)}/{len(entries)} done | "
                      f"{rate:.1f}/s | ETA {remaining/60:.1f}min | avg overall={avg_overall:.2f}")

    # Write final output in order
    with open(output_path, "w") as f:
        for i in range(len(entries)):
            if i in done:
                f.write(json.dumps(done[i], ensure_ascii=False) + "\n")

    print(f"[{source_name}] Done → {output_path}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["original", "deepseek", "235b", "all"], default="all")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max-entries", type=int, default=None, help="Limit entries for testing")
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    sources = list(DATA_SOURCES.keys()) if args.source == "all" else [args.source]

    # Run all sources concurrently
    await asyncio.gather(*[
        run_source(name, DATA_SOURCES[name], out_dir, args.concurrency, args.max_entries)
        for name in sources
    ])

    # Print comparison summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name in sources:
        score_path = out_dir / f"scores_{name}.jsonl"
        if score_path.exists():
            scores = []
            r1s, r2s, r3s, r4s, r5s = [], [], [], [], []
            with open(score_path) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("overall"):
                        scores.append(obj["overall"])
                    r1s.append(obj["r1"]["error_rate"])
                    if obj.get("r2_scores"):
                        for s in obj["r2_scores"]:
                            if s.get("r2a") and s.get("r2b") and s.get("r2c"):
                                r2s.append((s["r2a"]+s["r2b"]+s["r2c"])/3)
                    if obj.get("r3_scores"):
                        for s in obj["r3_scores"]:
                            if s.get("r3a") and s.get("r3b"):
                                r3s.append((s["r3a"]+s["r3b"])/2)
                    if obj.get("r4_scores"):
                        for s in obj["r4_scores"]:
                            if s.get("r4"):
                                r4s.append(s["r4"])
                    if obj.get("r5"):
                        r5s.append(obj["r5"])

            def avg(lst): return f"{sum(lst)/len(lst):.2f}" if lst else "N/A"
            print(f"\n[{name}]")
            print(f"  Overall:      {avg(scores)}/10  (n={len(scores)})")
            print(f"  R1 Error Rate: {avg(r1s)} (lower=better)")
            print(f"  R2 Recovery:  {avg(r2s)}/3.0")
            print(f"  R3 Schema:    {avg(r3s)}/3.0")
            print(f"  R4 Alignment: {avg(r4s)}/3.0")
            print(f"  R5 Reuse:     {avg(r5s)}/5.0")


if __name__ == "__main__":
    asyncio.run(main())
