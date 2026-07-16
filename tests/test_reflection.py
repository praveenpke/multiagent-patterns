from langgraph_patterns.events import run_pattern
from langgraph_patterns.patterns.reflection import build_reflection, make_input


def test_reflection_critique_then_approval():
    graph = build_reflection()
    result = run_pattern(graph, make_input("Why do guardrails matter?"))

    assert result.final_state["approved"] is True
    assert result.final_state["iterations"] == 2
    assert "Improved draft" in result.final_state["draft"]

    critic_msgs = [
        e.data["content"] for e in result.events_of("agent_message") if e.agent == "critic"
    ]
    assert any("Too vague" in m for m in critic_msgs)
    assert critic_msgs[-1] == "APPROVED"


def test_reflection_max_iterations_guardrail():
    graph = build_reflection(max_iterations=1)
    result = run_pattern(graph, make_input("task"))
    # Budget stops the loop after the first (unapproved) round.
    assert result.final_state["approved"] is False
    assert result.final_state["iterations"] == 1
    assert result.final_state["draft"].startswith("Rough draft")
