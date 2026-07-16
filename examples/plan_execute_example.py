"""Run the plan-and-execute pattern end to end (keyless)."""

from _console import print_event, print_final_state

from langgraph_patterns.events import stream_events
from langgraph_patterns.patterns.plan_execute import build_plan_execute, make_input

graph = build_plan_execute()
final_state = {}
for event in stream_events(graph, make_input("Compare supervisor and hierarchical orchestration.")):
    print_event(event)
    if event.type == "run_end":
        final_state = event.data["final_state"]
print_final_state(final_state)
