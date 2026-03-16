#!/usr/bin/env python3
"""
Rewrite assistant text messages in SFT trajectories to unify style.
Runs two versions in parallel: 235B thinking and Deepseek V3.1.

Usage:
    python3 rewrite_sft.py [--input FILE] [--concurrency N]
"""

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from pathlib import Path

import aiohttp

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = "https://api-web.eigenai.com/api/v1/chat/completions"
API_KEY = "<OPENAI_API_KEY>"

MODELS = {
    "235b": "qwen3-235b-thinking",
    "deepseek": "deepseek",
}

CONCURRENCY = {
    "235b": 10,
    "deepseek": 15,
}

INPUT_FILE = "/data/apex_sft_least_turns_p95.jsonl"
OUTPUT_DIR = "/data/rewrite_output"

SYSTEM_PROMPT = """You are a professional editor. Your task is to rewrite assistant messages from an AI agent trajectory.

The agent completes real-world tasks using tools (filesystem, calendar, code execution, Excel, etc.).

Rewriting rules:
- Keep the exact same meaning, decisions, and reasoning
- Keep all specific values, file names, numbers, and technical details unchanged
- Use a consistent, clear, professional first-person style ("I'll...", "Let me...", "Now I...")
- Be concise — remove filler phrases like "Great!", "Certainly!", "Of course!"
- Maintain a logical flow that shows step-by-step thinking
- Do NOT change any tool call arguments or results
- Do NOT add new reasoning or change the conclusion

Return ONLY a JSON object in this exact format (no markdown, no extra text):
{"rewrites": ["rewritten message 1", "rewritten message 2", ...]}

The array must have exactly the same number of elements as the input."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_assistant_text_messages(messages):
    """
    Return list of (index, text_content) for assistant messages that have text.
    Skips assistant messages that are pure tool_calls with no text.
    """
    result = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            result.append((i, content.strip()))
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text:
                result.append((i, text))
    return result


def build_context_summary(messages, target_indices):
    """Build a compact context showing the trajectory up to the first target message."""
    first_target = min(target_indices)
    lines = []
    for i, msg in enumerate(messages[:first_target]):
        role = msg["role"]
        if role == "system":
            continue
        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            lines.append(f"USER: {str(content)[:500]}")
        elif role == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            lines.append(f"TOOL RESULT: {str(content)[:300]}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            content = msg.get("content", "")
            if tool_calls:
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                lines.append(f"ASSISTANT (called tools: {', '.join(names)})")
            elif content:
                if isinstance(content, list):
                    content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                lines.append(f"ASSISTANT: {str(content)[:200]}")
    return "\n".join(lines)


def build_rewrite_prompt(entry, text_messages):
    """Build the user prompt for the rewrite request."""
    context = build_context_summary(entry["messages"], [i for i, _ in text_messages])

    texts_json = json.dumps([text for _, text in text_messages], ensure_ascii=False, indent=2)

    prompt = f"""Here is the task context (truncated):
{context}

Rewrite the following {len(text_messages)} assistant message(s) to have a consistent, professional style.
Keep all reasoning, decisions, and technical details intact.

Messages to rewrite:
{texts_json}"""
    return prompt


async def call_api(session, model_id, messages, semaphore, max_retries=3):
    """Call the eigenai chat completion API with retry."""
    payload = {
        "model": model_id,
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
            async with semaphore:
                async with session.post(
                    API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt * 5
                        print(f"  Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                    data = await resp.json()
                    return data
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt * 3)
                continue
            raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_rewrites(response_text, expected_count):
    """Parse JSON rewrites from model response, with fallback."""
    # Strip thinking tags if present
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    # Try to find JSON object
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            rewrites = obj.get("rewrites", [])
            if isinstance(rewrites, list) and len(rewrites) == expected_count:
                return rewrites
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing as plain JSON array
    arr_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if arr_match:
        try:
            arr = json.loads(arr_match.group())
            if isinstance(arr, list) and len(arr) == expected_count:
                return arr
        except json.JSONDecodeError:
            pass

    return None  # Failed to parse


async def rewrite_entry(session, model_name, model_id, entry_idx, entry, semaphore):
    """Rewrite all assistant text messages in one entry."""
    text_messages = get_assistant_text_messages(entry["messages"])

    if not text_messages:
        return entry_idx, entry  # Nothing to rewrite

    user_prompt = build_rewrite_prompt(entry, text_messages)
    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = await call_api(session, model_id, api_messages, semaphore)
        response_text = response["choices"][0]["message"]["content"] or ""
        rewrites = parse_rewrites(response_text, len(text_messages))

        if rewrites is None:
            print(f"  [{model_name}] Entry {entry_idx}: failed to parse rewrites, keeping original")
            return entry_idx, entry

        # Apply rewrites
        new_entry = json.loads(json.dumps(entry))  # deep copy
        for (msg_idx, _), new_text in zip(text_messages, rewrites):
            msg = new_entry["messages"][msg_idx]
            if isinstance(msg.get("content"), list):
                # Replace text blocks, keep non-text blocks
                new_content = []
                replaced = False
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text" and not replaced:
                        new_content.append({"type": "text", "text": new_text})
                        replaced = True
                    else:
                        new_content.append(block)
                msg["content"] = new_content
            else:
                msg["content"] = new_text

        return entry_idx, new_entry

    except Exception as e:
        print(f"  [{model_name}] Entry {entry_idx}: error — {e}")
        return entry_idx, None  # Signal failure


async def run_model(model_name, model_id, entries, concurrency, output_path, progress_path):
    """Run rewriting for one model across all entries."""
    # Load already-done indices
    done = set()
    results = {}
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                idx = obj["idx"]
                done.add(idx)
                results[idx] = obj["entry"]
        print(f"[{model_name}] Resuming: {len(done)}/{len(entries)} already done")

    todo = [i for i in range(len(entries)) if i not in done]
    print(f"[{model_name}] Processing {len(todo)} entries with concurrency={concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    start = time.time()
    completed = 0
    failed = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            rewrite_entry(session, model_name, model_id, i, entries[i], semaphore)
            for i in todo
        ]

        for coro in asyncio.as_completed(tasks):
            entry_idx, result = await coro
            completed += 1

            if result is not None:
                results[entry_idx] = result
                # Append to progress file
                with open(progress_path, "a") as f:
                    f.write(json.dumps({"idx": entry_idx, "entry": result}, ensure_ascii=False) + "\n")
            else:
                failed += 1
                # Keep original on failure
                results[entry_idx] = entries[entry_idx]
                with open(progress_path, "a") as f:
                    f.write(json.dumps({"idx": entry_idx, "entry": entries[entry_idx]}, ensure_ascii=False) + "\n")

            if completed % 50 == 0 or completed == len(todo):
                elapsed = time.time() - start
                rate = completed / elapsed
                remaining = (len(todo) - completed) / rate if rate > 0 else 0
                print(f"[{model_name}] {completed + len(done)}/{len(entries)} done "
                      f"({failed} failed) | {rate:.1f}/s | ETA {remaining/60:.1f}min")

    # Write final output in original order
    with open(output_path, "w") as f:
        for i in range(len(entries)):
            entry = results.get(i, entries[i])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"[{model_name}] Done! {len(entries)} entries in {elapsed/60:.1f}min → {output_path}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--concurrency-235b", type=int, default=CONCURRENCY["235b"])
    parser.add_argument("--concurrency-deepseek", type=int, default=CONCURRENCY["deepseek"])
    parser.add_argument("--model", choices=["235b", "deepseek", "both"], default="both")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Load entries
    entries = []
    with open(args.input) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    print(f"Loaded {len(entries)} entries from {args.input}")

    tasks = []
    if args.model in ("235b", "both"):
        tasks.append(run_model(
            "235b", MODELS["235b"], entries,
            args.concurrency_235b,
            out_dir / "apex_sft_least_turns_p95_rewrite_235b.jsonl",
            out_dir / "progress_235b.jsonl",
        ))
    if args.model in ("deepseek", "both"):
        tasks.append(run_model(
            "deepseek", MODELS["deepseek"], entries,
            args.concurrency_deepseek,
            out_dir / "apex_sft_least_turns_p95_rewrite_deepseek.jsonl",
            out_dir / "progress_deepseek.jsonl",
        ))

    # Run both models concurrently
    await asyncio.gather(*tasks)
    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
