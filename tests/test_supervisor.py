from langgraph_patterns.events import run_pattern
from langgraph_patterns.patterns.supervisor import (
    DEFAULT_AGENTS,
    build_supervisor,
    make_input,
)


def test_supervisor_runs_end_to_end_with_fake_models():
    graph = build_supervisor()
    result = run_pattern(graph, make_input("Explain multi-agent trade-offs."))

    assert result.events[0].type == "run_start"
    assert result.events[-1].type == "run_end"
    assert not result.interrupted

    # Node lifecycle events cover the supervisor and both specialists.
    started = {e.node for e in result.events_of("node_start")}
    assert {"supervisor", "researcher", "writer"} <= started
    ended = {e.node for e in result.events_of("node_end")}
    assert started == ended

    # Each specialist spoke exactly once (fake router visits each once).
    speakers = [e.agent for e in result.events_of("agent_message")]
    assert speakers.count("researcher") == 1
    assert speakers.count("writer") == 1

    # Final typed state: named AIMessages from both agents, budget respected.
    names = [m.get("type") for m in result.final_state["messages"]]
    assert names[0] == "human" and "ai" in names
    assert result.final_state["iterations"] <= 6
    assert result.final_state["next_agent"] == "FINISH"


def test_supervisor_max_iterations_guardrail():
    graph = build_supervisor(max_iterations=1)
    result = run_pattern(graph, make_input("anything"))
    # One routing turn + forced FINISH turn.
    assert result.final_state["iterations"] == 2
    speakers = [e.agent for e in result.events_of("agent_message")]
    # Only the first agent got to act before the budget cut in.
    assert speakers.count(DEFAULT_AGENTS[1].name) == 0
