"""Run the episodic memory pattern twice to show the store growing (keyless)."""

from _console import print_event, print_final_state

from langgraph_patterns.events import stream_events
from langgraph_patterns.memory.episodic import EpisodicMemory, SEED_EPISODES, build_episodic, make_input

memory = EpisodicMemory(list(SEED_EPISODES))
graph = build_episodic(memory)

for text in [
    "How do I make my LangGraph agent resumable?",
    "Remind me how resumable agents work?",
]:
    print(f"\n### input: {text!r}")
    final_state = {}
    for event in stream_events(graph, make_input(text)):
        print_event(event)
        if event.type == "run_end":
            final_state = event.data["final_state"]
    print_final_state(final_state)
