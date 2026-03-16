#!/usr/bin/env python3
"""
rewrite_gemini_reasoning.py

For gemini SFT data: rewrite verbose/filler reasoning_content.
- Keeps good reasoning unchanged
- Simplifies reasoning that has filler mixed with real analysis
- Deletes reasoning that is pure narration with no analytical value

Only processes flagged entries (those matching filler heuristics).
"""

import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
API_URL = "https://api-web.eigenai.com/api/v1/chat/completions"
API_KEY = "<OPENAI_API_KEY>"


def call_llm(prompt: str, system_prompt: str = None, max_retries: int = 5) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "gpt-oss-120b",
        "messages": messages,
        "temperature": 0.5,
        "reasoning_effort": "low",
        "max_tokens": 1500,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {type(e).__name__} (status={status_code})")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Filler detection heuristics
# ---------------------------------------------------------------------------

FILLER_PATTERNS = [
    r"^here's my",
    r"^okay, here's my",
    r"^okay, here's the summary",
    r"^here's my attempt",
    r"^here's my thought",
    r"^okay, let me get a handle",
    r"^okay, so i'm thinking",
    r"^alright, let's break this down",
    r"^okay, let's break this down",
    r"tailored for an expert audience",
    r"^right, let's",
    r"^alright, let's",
]


def is_filler(reasoning: str) -> bool:
    lower = reasoning.lower()
    return any(re.search(p, lower) for p in FILLER_PATTERNS)


# ---------------------------------------------------------------------------
# Rewrite prompt
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """\
You are lightly polishing the reasoning_content field of an AI agent's tool-calling trace. The boilerplate opener and bold section header have already been stripped. What remains is the actual analytical content, but it may still start with a filler phrase like "Right," or "Alright," or have minor padding.

Your job is LIGHT EDITING ONLY:
- Remove or shorten any remaining filler sentence starters ("Right, so...", "Alright,", "Let me just...")
- If the first sentence is still purely narration of the tool call (e.g. "I need to open this file to read it"), remove it only if something more substantive follows
- Otherwise keep the content intact — do not delete real reasoning
- Do NOT start the result with "I've already/just..." or "Here's my..."
- Write in first person, plain prose

IMPORTANT: Always output the cleaned text. NEVER output EMPTY or delete everything — if the content has any analytical value at all, keep it.

Output ONLY the cleaned text. No preamble, no explanation."""


def mechanical_strip(reasoning: str) -> str:
    """
    Mechanically remove boilerplate opener and bold section header.
    Returns the stripped content (may still have minor filler like 'Right, so...').
    """
    lines = reasoning.strip().split('\n')
    result = []
    i = 0

    # Skip leading filler lines: opener paragraphs and bold headers
    while i < len(lines):
        line = lines[i].strip()
        lower = line.lower()

        # Skip blank lines at start
        if not line:
            i += 1
            continue

        # Skip opener sentences matching filler patterns
        if any(re.search(p, lower) for p in FILLER_PATTERNS):
            i += 1
            continue

        # Skip standalone bold headers like **Section Title**
        if re.match(r'^\*\*[^*]+\*\*$', line):
            i += 1
            continue

        # First non-filler, non-header line — start keeping from here
        break

    result = lines[i:]

    # Remove leading blank lines from result
    while result and not result[0].strip():
        result.pop(0)

    return '\n'.join(result).strip()


# Placeholder patterns that indicate hallucinated/template content
PLACEHOLDER_PATTERNS = [
    r'\[assume the user',
    r'\[insert ',
    r'\[your ',
    r'\[e\.g\.',
]


def is_placeholder(text: str) -> bool:
    """True if text contains template placeholders (hallucinated content)."""
    return any(re.search(p, text, re.IGNORECASE) for p in PLACEHOLDER_PATTERNS)


def rewrite_reasoning(reasoning: str, tool_calls: list, context_snippet: str = "") -> str:
    """
    1. Mechanically strip opener and bold header
    2. If placeholder/hallucinated content: delete (return "")
    3. If stripped content is substantive: light LLM polish
    4. If stripped content is too short to be useful: delete
    """
    stripped = mechanical_strip(reasoning)

    # Delete hallucinated placeholder content
    if is_placeholder(stripped):
        return ""

    # If nothing meaningful remains after stripping
    if len(stripped) < 40:
        return ""

    # Light LLM polish of the stripped content
    tool_summary = ", ".join(tc['function']['name'] for tc in tool_calls)
    parts = []
    if context_snippet:
        parts.append(f"## Recent context (last tool result):\n{context_snippet[:400]}\n")
    parts.append(f"## Tool call being made: {tool_summary}\n")
    parts.append(f"## Reasoning to lightly polish:\n{stripped}")
    prompt = "\n".join(parts)

    result = call_llm(prompt, REWRITE_SYSTEM_PROMPT).strip()
    # Guard: if LLM still returns EMPTY for some reason, fall back to stripped version
    if not result or result.upper() == "EMPTY":
        return stripped
    return result


# ---------------------------------------------------------------------------
# Get last tool result for context
# ---------------------------------------------------------------------------

def get_last_tool_result(messages: list, msg_idx: int) -> str:
    for i in range(msg_idx - 1, -1, -1):
        if messages[i].get('role') == 'tool':
            content = messages[i].get('content', '')
            if isinstance(content, list):
                content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# Process one trajectory
# ---------------------------------------------------------------------------

def process_one(line_idx: int, raw_line: str) -> tuple:
    """
    For each assistant message with non-empty reasoning_content:
    - If flagged as filler: rewrite
    - Otherwise: keep as-is
    Returns (line_idx, output_line, num_rewritten, num_deleted, error_msg)
    """
    try:
        trajectory = json.loads(raw_line)
        messages = trajectory.get("messages", [])

        num_rewritten = 0
        num_deleted = 0

        for i, msg in enumerate(messages):
            if msg.get('role') != 'assistant':
                continue
            rc = msg.get('reasoning_content', '')
            if not rc or not rc.strip():
                continue
            if not is_filler(rc):
                continue

            tool_calls = msg.get('tool_calls', [])
            context = get_last_tool_result(messages, i)

            try:
                new_rc = rewrite_reasoning(rc, tool_calls, context)
                if new_rc == "":
                    del messages[i]['reasoning_content']
                    num_deleted += 1
                elif new_rc != rc:
                    messages[i]['reasoning_content'] = new_rc
                    num_rewritten += 1
            except Exception:
                pass  # keep original on failure

        trajectory["messages"] = messages
        return (line_idx, json.dumps(trajectory, ensure_ascii=False), num_rewritten, num_deleted, None)

    except Exception as e:
        return (line_idx, raw_line.strip(), 0, 0, str(e))


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all(input_file: str, output_file: str, max_workers: int = 20):
    with open(input_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    total = len(lines)
    print(f"Loaded {total} trajectories from {input_file}")
    print(f"Processing with {max_workers} workers...\n")

    results = [None] * total
    completed = total_rewritten = total_deleted = failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, i, line): i for i, line in enumerate(lines)}
        for future in as_completed(futures):
            line_idx, output_line, num_rewritten, num_deleted, error = future.result()
            results[line_idx] = output_line
            completed += 1
            total_rewritten += num_rewritten
            total_deleted += num_deleted
            if error:
                failed += 1

            elapsed = time.time() - start_time
            rate = completed / elapsed * 60 if elapsed > 0 else 0
            status = f"✗ {error}" if error else f"✓ rewritten={num_rewritten} deleted={num_deleted}"
            print(f"  [{completed}/{total}] Line {line_idx}: {status}  ({rate:.1f}/min)")

    with open(output_file, 'w') as f:
        for line in results:
            if line is not None:
                f.write(line + '\n')

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! Output: {output_file}")
    print(f"Trajectories: {total} | Rewritten: {total_rewritten} | Deleted: {total_deleted} | Failed: {failed}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test single example
# ---------------------------------------------------------------------------

def test_single(input_file: str, example_idx: int = 0):
    with open(input_file) as f:
        for i, line in enumerate(f):
            if i == example_idx:
                trajectory = json.loads(line)
                break

    messages = trajectory["messages"]
    flagged = [(i, msg) for i, msg in enumerate(messages)
               if msg.get('role') == 'assistant'
               and msg.get('reasoning_content', '').strip()
               and is_filler(msg['reasoning_content'])]

    print(f"Trajectory {example_idx}: {len(messages)} messages, {len(flagged)} flagged for rewrite\n")

    for msg_idx, msg in flagged[:5]:
        rc = msg['reasoning_content']
        tool_name = msg['tool_calls'][0]['function']['name'] if msg.get('tool_calls') else '(no tools)'
        context = get_last_tool_result(messages, msg_idx)
        print(f"--- Msg {msg_idx} ({tool_name}) ---")
        print(f"BEFORE [{len(rc)} chars]: {rc[:200]}...")
        try:
            new_rc = rewrite_reasoning(rc, msg.get('tool_calls', []), context)
            if new_rc == "":
                print(f"AFTER: [DELETED]")
            else:
                print(f"AFTER [{len(new_rc)} chars]: {new_rc[:200]}...")
        except Exception as e:
            print(f"ERROR: {e}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rewrite verbose gemini reasoning")
    parser.add_argument("--mode", choices=["test", "batch"], default="test")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--example", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "test":
        test_single(args.input, example_idx=args.example)
    else:
        output = args.output or args.input.replace(".jsonl", "_rewritten.jsonl")
        process_all(args.input, output, max_workers=args.workers)
