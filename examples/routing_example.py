"""Run the conditional routing pattern on three differently-shaped inputs (keyless)."""

from _console import print_event

from langgraph_patterns.events import stream_events
from langgraph_patterns.patterns.routing import build_routing, make_input

graph = build_routing()
for text in ["What is 25 * 16?", "What is LangGraph used for?", "hey there"]:
    print(f"\n### input: {text!r}")
    for event in stream_events(graph, make_input(text)):
        print_event(event)
