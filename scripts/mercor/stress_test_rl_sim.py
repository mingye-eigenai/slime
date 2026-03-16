#!/usr/bin/env python3
"""
Full RL training simulation with persistent MCP connections.

Simulates the actual RL training flow:
  1. Batch create all sessions (via runner)
  2. Establish persistent MCP connections to backends (one per session)
  3. Run K steps — each step: all sessions call a tool concurrently
  4. Disconnect MCP clients
  5. Batch destroy all sessions

Usage:
    python3 scripts/stress_test_rl_sim.py \
        --runner-ports 9040,9010,9020,9030 \
        --batch-size 32 --rollout 16 --steps 10
"""

import asyncio
import json
import argparse
import statistics
import time
import uuid

import httpx


# Backend offset from runner port
BACKEND_OFFSET = {
    "calendar": 1, "chat": 2, "code_execution": 3, "excel": 4,
    "filesystem": 5, "mail": 6, "pdfs": 7, "powerpoint": 8, "word": 9,
}

TOOL_SERVER = {
    "list_files": "filesystem", "read_text_file": "filesystem",
    "read_image_file": "filesystem", "search_files": "filesystem",
    "get_file_metadata": "filesystem", "get_directory_tree": "filesystem",
    "create_event": "calendar", "list_events": "calendar",
    "read_event": "calendar", "update_event": "calendar",
    "delete_event": "calendar",
    "list_channels": "chat", "get_channel_history": "chat",
    "post_message": "chat", "reply_to_thread": "chat",
    "code_exec": "code_execution",
    "list_mails": "mail", "read_mail": "mail", "send_mail": "mail",
    "reply_mail": "mail", "reply_all_mail": "mail",
    "forward_mail": "mail", "search_mail": "mail",
}


def fmt(lats):
    if not lats:
        return "n/a"
    return (f"avg={statistics.mean(lats):.2f}s  "
            f"p50={statistics.median(lats):.2f}s  "
            f"max={max(lats):.2f}s")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--runner-ports", required=True,
                        help="Comma-separated runner ports for N containers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of tool call steps per trajectory")
    parser.add_argument("--tool", default="list_files")
    parser.add_argument("--tool-args", default='{"path": "/"}')
    args = parser.parse_args()

    runner_ports = [int(x) for x in args.runner_ports.split(",")]
    N = len(runner_ports)
    tool_args = json.loads(args.tool_args)
    bs = args.batch_size
    rollout = args.rollout
    steps = args.steps

    server = TOOL_SERVER.get(args.tool)
    if not server:
        print(f"ERROR: Unknown tool '{args.tool}'")
        return
    offset = BACKEND_OFFSET[server]

    spc = (bs // N) * rollout
    total = N * spc

    print(f"{'='*70}")
    print(f"  RL Training Simulation")
    print(f"  batch_size={bs}  rollout={rollout}  N={N}  steps={steps}")
    print(f"  {spc} sessions/container × {N} containers = {total} total")
    print(f"  Tool: {args.tool} → {server} (runner_port + {offset})")
    print(f"  Runner ports: {runner_ports}")
    print(f"{'='*70}")

    # Build session list: (runner_port, backend_port, sid)
    sessions = []
    for rp in runner_ports:
        bp = rp + offset
        for _ in range(spc):
            sid = f"rl_{uuid.uuid4().hex[:8]}"
            sessions.append((rp, bp, sid))

    # ── Phase 1: Create all sessions ─────────────────────────────────────
    print(f"\n  Phase 1: Creating {total} sessions...")
    t0 = time.time()
    create_ok = 0
    async with httpx.AsyncClient(timeout=120) as hc:
        tasks = []
        for rp, _, sid in sessions:
            tasks.append(hc.post(f"http://{args.host}:{rp}/sessions/create",
                                 json={"session_id": sid}))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, httpx.Response) and r.status_code == 200:
                create_ok += 1
    create_wall = time.time() - t0
    print(f"  Create: {create_ok}/{total}  wall={create_wall:.1f}s")

    if create_ok == 0:
        print("  FATAL: No sessions created")
        return

    # ── Phase 2: Establish persistent MCP connections ────────────────────
    from fastmcp import Client as FastMCPClient
    from fastmcp.client.transports import StreamableHttpTransport

    print(f"\n  Phase 2: Connecting {total} MCP clients (persistent)...")
    t0 = time.time()
    clients = []  # parallel list with sessions
    connect_ok = 0
    for _, bp, sid in sessions:
        try:
            transport = StreamableHttpTransport(
                f"http://{args.host}:{bp}/mcp/",
                headers={"X-Session-Id": sid},
            )
            c = FastMCPClient(transport=transport, timeout=60)
            await c.__aenter__()
            clients.append(c)
            connect_ok += 1
        except Exception as e:
            clients.append(None)
    connect_wall = time.time() - t0
    print(f"  Connected: {connect_ok}/{total}  wall={connect_wall:.1f}s")

    active_indices = [i for i, c in enumerate(clients) if c is not None]
    if not active_indices:
        print("  FATAL: No MCP connections established")
        return

    # ── Phase 3: Run steps ───────────────────────────────────────────────
    print(f"\n  Phase 3: Running {steps} steps ({len(active_indices)} concurrent tool calls each)...")
    step_walls = []
    step_lats = []

    for step in range(1, steps + 1):
        t0 = time.time()

        async def call_one(idx):
            t_start = time.time()
            try:
                await clients[idx].call_tool(args.tool, tool_args)
                return True, time.time() - t_start
            except Exception:
                return False, time.time() - t_start

        results = await asyncio.gather(*[call_one(i) for i in active_indices])
        wall = time.time() - t0

        ok = sum(1 for r, _ in results if r)
        lats = [lat for r, lat in results if r]
        avg_lat = statistics.mean(lats) if lats else 0

        step_walls.append(wall)
        step_lats.append(avg_lat)

        # Print every step for first 5, then every 5th
        if step <= 5 or step % 5 == 0 or step == steps:
            print(f"    Step {step:>3}: {ok}/{len(active_indices)}  "
                  f"wall={wall:.2f}s  {fmt(lats)}")

    # ── Phase 4: Disconnect + Destroy ────────────────────────────────────
    print(f"\n  Phase 4: Disconnecting and destroying...")
    for c in clients:
        if c is not None:
            try:
                await c.__aexit__(None, None, None)
            except Exception:
                pass

    t0 = time.time()
    async with httpx.AsyncClient(timeout=30) as hc:
        dtasks = [hc.post(f"http://{args.host}:{rp}/sessions/destroy",
                          json={"session_id": sid})
                  for rp, _, sid in sessions]
        await asyncio.gather(*dtasks, return_exceptions=True)
    destroy_wall = time.time() - t0
    print(f"  Destroy wall: {destroy_wall:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  SUMMARY")
    print(f"{'#'*70}")
    print(f"  Config: batch_size={bs} rollout={rollout} N={N} steps={steps}")
    print(f"  Total sessions: {total}")
    print(f"  Session create:    {create_wall:.1f}s")
    print(f"  MCP connect:       {connect_wall:.1f}s")
    print(f"  Step wall times:   {fmt(step_walls)}")
    print(f"  Step avg latency:  {fmt(step_lats)}")
    if step_walls:
        print(f"  Total step time:   {sum(step_walls):.1f}s for {steps} steps")
        print(f"  Projected 100 steps: {sum(step_walls)/steps*100:.0f}s "
              f"({sum(step_walls)/steps*100/60:.1f} min)")
    print(f"  Session destroy:   {destroy_wall:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
