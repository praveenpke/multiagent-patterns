"""Run the parallel fan-out pattern end to end (keyless)."""

from _console import print_event, print_final_state

from langgraph_patterns.events import stream_events
from langgraph_patterns.patterns.fanout import build_fanout, make_input

graph = build_fanout()
final_state = {}
for event in stream_events(graph, make_input("Should we migrate our agents to a graph runtime?")):
    print_event(event)
    if event.type == "run_end":
        final_state = event.data["final_state"]
print_final_state(final_state)
