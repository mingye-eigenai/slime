#!/usr/bin/env python3
"""Step 3: Rewrite thin/generic reasoning on code_exec steps."""
import json, re, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://api-web.eigenai.com/api/v1/chat/completions"
API_KEY = "<OPENAI_API_KEY>"
MODEL = "deepseek"

INPUT = "/data/apex_sft_all_models_best_step2.jsonl"
OUTPUT = "/data/apex_sft_all_models_best_step3.jsonl"

THIN_CHAR_THRESHOLD = 200
GENERIC_PATTERNS = [
    'now i have all the data', 'let me write a python script',
    'now let me run the code', 'let me write a comprehensive',
    'i need to use code execution', 'now let me compute',
    'let me calculate this', 'now let me write', "i'll create a file",
    'let me run this', 'now i have all the information i need',
    'let me write the code', 'i need to use the proper format',
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

### Part 2: DATA + FORMULA GROUNDING (where from and how)
Cite specific numbers from previous tool results and explain the formulas.

### Part 3: EXPECTED OUTPUT (what we'll get)
Briefly state what the code should produce.

## Style rules

1. Lead with the financial insight or analytical goal, not "I need to..." or "Let me..."
2. Cite specific numbers with their source
3. Name the formula and explain the logic
4. For iterative/circular models: explain WHY iteration is needed
5. For error recovery: Case A (format error) keep short. Case B (logic error) diagnose. Case C (mismatch) explain discrepancy.
6. Keep it proportional: simple calculations get 2-3 sentences, complex models get a full paragraph
7. Write from first person ("I"), as a single autonomous agent

## Output

Output ONLY the rewritten reasoning_content. No preamble."""

def is_thin_reasoning(reasoning):
    r = reasoning.strip()
    if len(r) < THIN_CHAR_THRESHOLD:
        return True
    return any(p in r.lower() for p in GENERIC_PATTERNS)

def extract_code(tc):
    args = tc.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try: args = json.loads(args)
        except: return args[:2000]
    if isinstance(args, dict):
        if "code" in args: return args["code"][:3000]
        req = args.get("request", {})
        if isinstance(req, dict) and "code" in req: return req["code"][:3000]
    return str(args)[:2000]

def build_context(msgs, idx, window=6):
    start = max(0, idx - window)
    lines = []
    for i in range(start, idx):
        m = msgs[i]
        role = m.get("role", "")
        if role == "user":
            lines.append(f"[USER]: {str(m.get('content',''))[:500]}")
        elif role == "tool":
            lines.append(f"[TOOL RESULT]: {str(m.get('content',''))[:500]}")
        elif role == "assistant":
            tc = m.get("tool_calls", [])
            if tc:
                names = [t.get("function",{}).get("name","?") for t in tc]
                lines.append(f"[ASSISTANT called: {', '.join(names)}]")
                r = str(m.get("reasoning_content","") or "")
                if r.strip(): lines.append(f"[ASSISTANT reasoning]: {r[:300]}")
            else:
                c = str(m.get("content","") or "")[:300]
                if c.strip(): lines.append(f"[ASSISTANT]: {c}")
    return "\n".join(lines)

def find_targets(samples):
    targets = []
    for idx, data in enumerate(samples):
        msgs = data["messages"]
        user_req = ""
        for m in msgs:
            if m.get("role") == "user":
                user_req = str(m.get("content",""))[:1000]
                break
        i = 0
        while i < len(msgs):
            m = msgs[i]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc_list = m["tool_calls"]
                has_ce = any("code_exec" in tc.get("function",{}).get("name","") for tc in tc_list)
                if has_ce:
                    reasoning = str(m.get("reasoning_content","") or "").strip()
                    if is_thin_reasoning(reasoning):
                        code_tc = next((tc for tc in tc_list if "code_exec" in tc.get("function",{}).get("name","")), None)
                        code = extract_code(code_tc) if code_tc else ""
                        result = ""
                        for ti, tc in enumerate(tc_list):
                            if "code_exec" in tc.get("function",{}).get("name",""):
                                ri = i + 1 + ti
                                if ri < len(msgs) and msgs[ri].get("role") == "tool":
                                    result = str(msgs[ri].get("content",""))[:2000]
                                break
                        targets.append({
                            "sample_idx": idx, "msg_idx": i, "reasoning": reasoning,
                            "code": code, "result": result, "user_request": user_req,
                        })
                i += 1 + len(tc_list)
            else:
                i += 1
    return targets

def call_api(messages, max_retries=5):
    payload = {"model": MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 4096}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=300)
            if resp.status_code == 429:
                time.sleep(2 ** attempt * 5)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise
    raise RuntimeError("Failed")

def rewrite_one(target, samples):
    msgs = samples[target["sample_idx"]]["messages"]
    ctx = build_context(msgs, target["msg_idx"])
    prompt = f"""## User's Original Request:
{target['user_request']}

## Recent Context:
{ctx}

## Current Code Being Executed:
```python
{target['code'][:3000]}
```

## Code Execution Result (for grounding):
{target['result'][:2000]}

## Current Reasoning (to be rewritten):
{target['reasoning']}

Rewrite the reasoning to be high-quality for SFT training."""
    
    api_msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        new_r = call_api(api_msgs)
        return target["sample_idx"], target["msg_idx"], new_r
    except Exception as e:
        print(f"  Error on {target['sample_idx']}:{target['msg_idx']}: {e}")
        return target["sample_idx"], target["msg_idx"], None

def main():
    with open(INPUT) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples")

    targets = find_targets(samples)
    print(f"Found {len(targets)} thin reasoning blocks to rewrite")

    if not targets:
        with open(OUTPUT, 'w') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print("Nothing to rewrite, copied as-is")
        return

    start = time.time()
    completed = 0
    failed = 0
    rewrites = {}

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(rewrite_one, t, samples): t for t in targets}
        for fut in as_completed(futures):
            si, mi, new_r = fut.result()
            completed += 1
            if new_r:
                rewrites[(si, mi)] = new_r
            else:
                failed += 1
            if completed % 50 == 0 or completed == len(targets):
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  {completed}/{len(targets)} ({failed} failed) {rate:.1f}/s")

    # Apply
    for (si, mi), new_r in rewrites.items():
        samples[si]["messages"][mi]["reasoning_content"] = new_r

    with open(OUTPUT, 'w') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    print(f"\nDone! Rewrote {len(rewrites)}, failed {failed}")
    print(f"Output: {OUTPUT}")

if __name__ == '__main__':
    main()
