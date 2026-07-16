"""Run the ReAct pattern end to end (keyless: uses scripted fake models)."""

from _console import print_event, print_final_state

from langgraph_patterns.events import stream_events
from langgraph_patterns.patterns.react import build_react, make_input

graph = build_react()
final_state = {}
for event in stream_events(graph, make_input("What is 12 * (7 + 5)?")):
    print_event(event)
    if event.type == "run_end":
        final_state = event.data["final_state"]
print_final_state(final_state)
