"""Shared console pretty-printer for the examples."""

from __future__ import annotations

import json

from langgraph_patterns.events import Event


def print_event(event: Event) -> None:
    if event.type == "run_start":
        print("=== run started ===")
    elif event.type == "node_start":
        print(f"[node >] {event.node}")
    elif event.type == "node_end":
        print(f"[node .] {event.node}")
    elif event.type == "agent_message":
        print(f"  ({event.agent}) {event.data.get('content', '')}")
    elif event.type == "tool_call":
        print(f"  ({event.agent}) -> tool {event.data.get('tool')}({event.data.get('args')})")
    elif event.type == "tool_result":
        print(f"  ({event.agent}) <- {str(event.data.get('result'))[:120]}")
    elif event.type == "interrupt":
        print(f"[interrupt] {json.dumps(event.data.get('payload'), default=str)[:200]}")
    elif event.type == "run_end":
        print("=== run finished ===")
        if event.data.get("interrupted"):
            print("(paused on interrupt — resume with a Command)")


def print_final_state(final_state: dict) -> None:
    print("\nFinal state:")
    print(json.dumps(final_state, indent=2, default=str)[:2000])
