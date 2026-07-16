"""Run the supervisor pattern end to end (keyless: uses scripted fake models)."""

from _console import print_event, print_final_state

from langgraph_patterns.events import stream_events
from langgraph_patterns.patterns.supervisor import build_supervisor, make_input

graph = build_supervisor()
final_state = {}
for event in stream_events(graph, make_input("Explain the trade-offs of multi-agent systems.")):
    print_event(event)
    if event.type == "run_end":
        final_state = event.data["final_state"]
print_final_state(final_state)
