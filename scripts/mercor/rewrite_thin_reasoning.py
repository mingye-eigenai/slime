#!/usr/bin/env python3
"""
Rewrite thin/generic reasoning_content for code_exec steps in SFT trajectories.
Uses deepseek model via eigenai API.

Usage:
    python3 rewrite_thin_reasoning.py [--input FILE] [--output FILE] [--concurrency N]
"""

import argparse
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = "https://api-web.eigenai.com/api/v1/chat/completions"
API_KEY = "<OPENAI_API_KEY>"
MODEL = "deepseek"
CONCURRENCY = 15

INPUT_FILE = "/data/apex_sft_opus_mix_filled_fixed_with_domain_cleaned_v2.jsonl"
OUTPUT_FILE = "/data/apex_sft_opus_mix_filled_fixed_with_domain_cleaned_v3.jsonl"
PROGRESS_FILE = "/data/rewrite_thin_reasoning_progress.jsonl"

THIN_CHAR_THRESHOLD = 200  # reasoning shorter than this is always rewritten

GENERIC_PATTERNS = [
    'now i have all the data',
    'let me write a python script',
    'now let me run the code',
    'let me write a comprehensive',
    'i need to use code execution',
    'now let me compute',
    'let me calculate this',
    'now let me write',
    "i'll create a file",
    'let me run this',
    'now i have all the information i need',
    'let me write the code',
    'i need to use the proper format',
]

SYSTEM_PROMPT = """You are rewriting the reasoning_content for an AI assistant's code execution step in a financial analysis trajectory. The reasoning represents the assistant's INTERNAL THINKING before executing Python code — what led to this code, why this approach, and what to expect.

## Your Task

Given:
- The user's original request
- Recent conversation context (previous tool calls and results)
- The Python code about to be executed
- The code execution result (for grounding)

Rewrite the reasoning_content to be high-quality for SFT training.

## What GOOD code_exec reasoning looks like

Good reasoning has 3 parts:

### Part 1: ANALYTICAL SETUP (what and why)
State the financial question this code answers, and why code execution is the right approach.
- "The user needs the exit equity value under the revised debt structure. This requires an iterative solver because the revolver draw depends on the cash sweep, which depends on the revolver draw — a circular reference that can't be resolved in a single pass."

### Part 2: DATA + FORMULA GROUNDING (where from and how)
Cite specific numbers from previous tool results and explain the formulas.
- "From the 'Debt Schedule' tab: entry TLB = $1,034M (cell C15), interest rate = 11% (C16). The mandatory amortization is 2% of average annual balance: amort = 0.02 × (beg_bal + end_bal) / 2, which makes end_bal itself circular — solved iteratively."

### Part 3: EXPECTED OUTPUT (what we'll get)
Briefly state what the code should produce.
- "The script will output: (1) year-by-year TLB balances, (2) interest expense per year, (3) exit equity value, (4) IRR and MoM. I'll compare the base case against the spreadsheet to validate before reading the revised scenario."

## What BAD code_exec reasoning looks like

BAD: Pure narration with no analytical content
- "Let me write a Python script to calculate the answer."
- "Now let me run the code to compute the new scenario."
- "I need to use code execution for this calculation."
- "Now I have all the data I need. Let me write a comprehensive Python script."

BAD: Restating what will happen without explaining why
- "I'll create a file called calc.py, write the LBO model code, and execute it to get the IRR."

BAD: History recap instead of forward-looking analysis
- "I've already read the debt schedule, the assumptions tab, and the projections. Now let me compute..."

BAD: Vague references when you have specific numbers
- "Using the relevant financial data from the spreadsheet, I'll compute the enterprise value."

## Style rules

1. Lead with the financial insight or analytical goal, not "I need to..." or "Let me..."
2. Cite specific numbers with their source: "$613.6M EBITDA from 'Projections' tab row 45"
3. Name the formula and explain the logic: "Exit EV = EBITDA × multiple, where the multiple..."
4. For iterative/circular models: explain WHY iteration is needed
5. For error recovery — there are two distinct cases:

   **Case A: Tool API format error (previous call had correct code but wrong wrapper)**
   Keep it short. State the format issue and that the code logic is unchanged.
   GOOD: "The previous call failed with a pydantic validation error — I passed {\\"code\\": \\"...\\"} at the top level instead of nesting it inside {\\"request\\": {\\"action\\": \\"exec\\", \\"code\\": \\"...\\"}}. The Python calculation logic is identical; only the argument wrapper changes."
   BAD: Re-explaining the entire financial model again when only the JSON format was wrong.

   **Case B: Code logic error (previous code ran but produced wrong/unexpected results)**
   This needs real analytical diagnosis. State: (1) what the output was, (2) why it's wrong, (3) what the root cause is, (4) what specifically changes in the fix.
   GOOD: "The script output EBITDA of -$200M for FY2026, which is impossible for a company with $3.1B revenue and 30% margins. Root cause: operating_expense was defined as +0.308 (the margin) instead of -(1 - 0.308) = -0.692, so expenses were added to revenue instead of subtracted. Fix: line 45 changes from `op_exp_pct = 0.308` to `op_exp_pct = -(1 - 0.308)`."
   BAD: "The code had an error. Let me fix it and try again."

   **Case C: Code ran but results don't match the spreadsheet baseline**
   Explain the discrepancy and what assumption or data point differs.
   GOOD: "My model produces IRR of 27.19% vs the spreadsheet's 27.22%. The 3bp gap comes from the SOFR curve: I used flat 3.49% for all years, but the model uses a forward curve (3.49% in 2025, 3.34% in 2027). Updating to the stepped curve should close the gap."
   BAD: "There's a small rounding difference. Let me re-run."

6. Keep it proportional: simple calculations get 2-3 sentences, complex LBO/DCF models get a full paragraph
7. Write from first person ("I"), as a single autonomous agent

## Output

Output ONLY the rewritten reasoning_content. No preamble, no "Here is the rewritten reasoning:" — just the reasoning text itself."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_thin_reasoning(reasoning: str) -> bool:
    """Check if reasoning is thin (short or generic boilerplate)."""
    r = reasoning.strip()
    if len(r) < THIN_CHAR_THRESHOLD:
        return True
    rl = r.lower()
    return any(p in rl for p in GENERIC_PATTERNS)


def extract_code_from_tool_call(tc: dict) -> str:
    """Extract Python code from a code_exec tool call arguments."""
    args = tc.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args[:2000]
    # Handle both {"code": "..."} and {"request": {"action": "exec", "code": "..."}}
    if isinstance(args, dict):
        if "code" in args:
            return args["code"][:3000]
        req = args.get("request", {})
        if isinstance(req, dict) and "code" in req:
            return req["code"][:3000]
    return str(args)[:2000]


def build_recent_context(msgs: list, target_idx: int, window: int = 6) -> str:
    """Build recent context (last few tool steps before target)."""
    start = max(0, target_idx - window)
    lines = []
    for i in range(start, target_idx):
        m = msgs[i]
        role = m.get("role", "")
        if role == "user":
            content = str(m.get("content", ""))[:500]
            lines.append(f"[USER]: {content}")
        elif role == "tool":
            content = str(m.get("content", ""))[:500]
            lines.append(f"[TOOL RESULT]: {content}")
        elif role == "assistant":
            tc = m.get("tool_calls", [])
            if tc:
                names = [t.get("function", {}).get("name", "?") for t in tc]
                lines.append(f"[ASSISTANT called: {', '.join(names)}]")
                r = str(m.get("reasoning_content", "") or "")
                if r.strip():
                    lines.append(f"[ASSISTANT reasoning]: {r[:300]}")
            else:
                content = str(m.get("content", "") or "")[:300]
                if content.strip():
                    lines.append(f"[ASSISTANT]: {content}")
    return "\n".join(lines)


def build_rewrite_prompt(msgs: list, target_idx: int, code: str, result: str,
                         current_reasoning: str, user_request: str) -> str:
    """Build the user prompt for rewriting one reasoning block."""
    recent = build_recent_context(msgs, target_idx)
    return f"""## User's Original Request:
{user_request}

## Recent Context (last few tool steps):
{recent}

## Current Code Being Executed:
```python
{code[:3000]}
```

## Code Execution Result (for grounding — use specific numbers from here):
{result[:2000]}

## Current Reasoning (to be rewritten):
{current_reasoning}

Rewrite the reasoning to be high-quality for SFT training. Follow the system prompt guidelines."""


# ── Identify targets ─────────────────────────────────────────────────────────

def find_rewrite_targets(samples: list) -> list:
    """Find all (sample_idx, msg_idx) pairs that need rewriting."""
    targets = []
    for idx, data in enumerate(samples):
        msgs = data["messages"]
        # Get user request (first user message)
        user_request = ""
        for m in msgs:
            if m.get("role") == "user":
                user_request = str(m.get("content", ""))[:1000]
                break

        i = 0
        while i < len(msgs):
            m = msgs[i]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc_list = m["tool_calls"]
                has_code_exec = any("code_exec" in tc.get("function", {}).get("name", "")
                                    for tc in tc_list)
                if has_code_exec:
                    reasoning = str(m.get("reasoning_content", "") or "").strip()
                    if is_thin_reasoning(reasoning):
                        # Find the code and result
                        code_tc = next((tc for tc in tc_list
                                        if "code_exec" in tc.get("function", {}).get("name", "")),
                                       None)
                        code = extract_code_from_tool_call(code_tc) if code_tc else ""

                        # Find corresponding tool result
                        result = ""
                        for tc_idx, tc in enumerate(tc_list):
                            if "code_exec" in tc.get("function", {}).get("name", ""):
                                result_idx = i + 1 + tc_idx
                                if result_idx < len(msgs) and msgs[result_idx].get("role") == "tool":
                                    result = str(msgs[result_idx].get("content", ""))[:2000]
                                break

                        targets.append({
                            "sample_idx": idx,
                            "msg_idx": i,
                            "reasoning": reasoning,
                            "code": code,
                            "result": result,
                            "user_request": user_request,
                        })
                i += 1 + len(tc_list)
            else:
                i += 1
    return targets


# ── API call ──────────────────────────────────────────────────────────────────

def call_api(messages, max_retries=5):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=300)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def rewrite_one(target, samples):
    """Rewrite one thin reasoning block."""
    msgs = samples[target["sample_idx"]]["messages"]
    prompt = build_rewrite_prompt(
        msgs, target["msg_idx"], target["code"], target["result"],
        target["reasoning"], target["user_request"]
    )
    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        new_reasoning = call_api(api_messages)
        return target["sample_idx"], target["msg_idx"], new_reasoning
    except Exception as e:
        print(f"  Error on sample {target['sample_idx']} msg {target['msg_idx']}: {e}")
        return target["sample_idx"], target["msg_idx"], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--progress", default=PROGRESS_FILE)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    # Load data
    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples from {args.input}")

    # Find targets
    targets = find_rewrite_targets(samples)
    print(f"Found {len(targets)} thin/generic code_exec reasoning blocks to rewrite")

    # Load progress
    done = set()
    rewrites = {}
    progress_path = Path(args.progress)
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                key = (obj["sample_idx"], obj["msg_idx"])
                done.add(key)
                rewrites[key] = obj["new_reasoning"]
        print(f"Resuming: {len(done)} already done")

    todo = [t for t in targets if (t["sample_idx"], t["msg_idx"]) not in done]
    print(f"Remaining: {len(todo)}")

    if todo:
        start = time.time()
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(rewrite_one, t, samples): t for t in todo}
            for fut in as_completed(futures):
                sample_idx, msg_idx, new_reasoning = fut.result()
                completed += 1
                key = (sample_idx, msg_idx)

                if new_reasoning:
                    rewrites[key] = new_reasoning
                    with open(progress_path, "a") as f:
                        f.write(json.dumps({
                            "sample_idx": sample_idx, "msg_idx": msg_idx,
                            "new_reasoning": new_reasoning
                        }, ensure_ascii=False) + "\n")
                else:
                    failed += 1

                if completed % 20 == 0 or completed == len(todo):
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(todo) - completed) / rate if rate > 0 else 0
                    print(f"  {completed + len(done)}/{len(targets)} done "
                          f"({failed} failed) | {rate:.1f}/s | ETA {remaining/60:.1f}min")

    # Apply rewrites
    print(f"\nApplying {len(rewrites)} rewrites...")
    for (sample_idx, msg_idx), new_reasoning in rewrites.items():
        if new_reasoning:
            samples[sample_idx]["messages"][msg_idx]["reasoning_content"] = new_reasoning

    # Write output
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Done! Output: {args.output}")
    print(f"Total rewrites applied: {len(rewrites)}")


if __name__ == "__main__":
    main()
