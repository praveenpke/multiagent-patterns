"""Run the checkpointing pattern: stop mid-pipeline, then resume from SQLite.

Keyless. Demonstrates that a *new* graph instance over the same SQLite file
continues exactly where the previous run stopped.
"""

import tempfile
from pathlib import Path

from _console import print_event, print_final_state

from langgraph_patterns.events import stream_events
from langgraph_patterns.memory.checkpointing import (
    build_checkpointing,
    make_input,
    sqlite_checkpointer,
)

db = str(Path(tempfile.mkdtemp()) / "checkpoints.sqlite")
thread = {"configurable": {"thread_id": "demo"}}

print("--- run, stopping after stage 2 (simulated crash) ---")
graph1 = build_checkpointing(checkpointer=sqlite_checkpointer(db))
graph1.invoke(make_input("quarterly usage data"), thread, interrupt_after=["transform"])
print("state so far:", graph1.get_state(thread).values["completed"])

print("\n--- new graph instance over the same SQLite file resumes ---")
graph2 = build_checkpointing(checkpointer=sqlite_checkpointer(db))
final_state = {}
for event in stream_events(graph2, None, thread_id="demo"):
    print_event(event)
    if event.type == "run_end":
        final_state = event.data["final_state"]
print_final_state(final_state)
