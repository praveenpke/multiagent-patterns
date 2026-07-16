import pytest

from langgraph_patterns.events import run_pattern
from langgraph_patterns.patterns.routing import build_routing, make_input


@pytest.mark.parametrize(
    ("text", "expected_route", "check"),
    [
        ("What is 25 * 16?", "math", "400"),
        ("What is LangGraph used for?", "lookup", "Lookup result"),
        ("hey there", "chat", "Hello"),
    ],
)
def test_routing_dispatches_by_classification(text, expected_route, check):
    graph = build_routing()
    result = run_pattern(graph, make_input(text))

    classifier = [e for e in result.events_of("agent_message") if e.agent == "classifier"]
    assert classifier[0].data["content"] == f"Classified as: {expected_route}"
    assert expected_route in {e.node for e in result.events_of("node_start")}

    final = result.final_state["messages"][-1]["content"]
    assert check in final
    assert result.final_state["route"] == expected_route
