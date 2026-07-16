"""Run the HITL handoff pattern: pause on interrupt, then resume approved.

Keyless: uses scripted fake models. The example auto-approves after showing
the interrupt payload — in the playground the approval comes from the UI.
"""

from _console import print_event, print_final_state
from langgraph.types import Command

from langgraph_patterns.events import stream_events
from langgraph_patterns.patterns.hitl import build_hitl, make_input

graph = build_hitl()
thread_id = "hitl-demo"

print("--- run until human gate ---")
for event in stream_events(
    graph, make_input("Archive all inactive user accounts."), thread_id=thread_id
):
    print_event(event)

print("\n--- human approves; resume ---")
final_state = {}
for event in stream_events(
    graph,
    Command(resume={"approved": True, "feedback": "verified the account list"}),
    thread_id=thread_id,
):
    print_event(event)
    if event.type == "run_end":
        final_state = event.data["final_state"]
print_final_state(final_state)
