from langgraph_patterns.events import run_pattern
from langgraph_patterns.patterns.hierarchical import build_hierarchical, make_input


def test_hierarchical_two_level_delegation():
    graph = build_hierarchical()
    result = run_pattern(graph, make_input("Write up agent memory."))

    started = {e.node for e in result.events_of("node_start")}
    assert {
        "director",
        "research_manager",
        "writing_manager",
        "searcher",
        "summarizer",
        "drafter",
        "editor",
    } <= started

    # All four workers reported exactly once, in team order.
    workers = [
        e.agent
        for e in result.events_of("agent_message")
        if e.agent in {"searcher", "summarizer", "drafter", "editor"}
    ]
    assert workers == ["searcher", "summarizer", "drafter", "editor"]

    names = [m.get("content", "") for m in result.final_state["messages"]]
    assert any("[editor]" in c for c in names)
    assert result.final_state["route"] == "FINISH"


def test_hierarchical_iteration_budget():
    graph = build_hierarchical(max_iterations=2)
    result = run_pattern(graph, make_input("anything"))
    # Budget cuts the run short well before all four workers act.
    workers = [
        e.agent
        for e in result.events_of("agent_message")
        if e.agent in {"searcher", "summarizer", "drafter", "editor"}
    ]
    assert len(workers) <= 2
    assert result.events[-1].type == "run_end"
