"""Run the hierarchical delegation pattern end to end (keyless)."""

from _console import print_event, print_final_state

from langgraph_patterns.events import stream_events
from langgraph_patterns.patterns.hierarchical import build_hierarchical, make_input

graph = build_hierarchical()
final_state = {}
for event in stream_events(graph, make_input("Produce a researched write-up on agent memory.")):
    print_event(event)
    if event.type == "run_end":
        final_state = event.data["final_state"]
print_final_state(final_state)
