from langgraph_patterns.events import run_pattern
from langgraph_patterns.patterns.fanout import build_fanout, make_input


def test_fanout_runs_all_branches_and_merges():
    graph = build_fanout()
    result = run_pattern(graph, make_input("Adopt a graph runtime?"))

    # All three analyst branches ran (same node, three parallel tasks).
    analyst_starts = [e for e in result.events_of("node_start") if e.node == "analyst"]
    assert len(analyst_starts) == 3

    analyses = result.final_state["analyses"]
    assert len(analyses) == 3
    for perspective in ("optimist", "skeptic", "pragmatist"):
        assert any(f"[{perspective}]" in a for a in analyses)

    assert "3 perspectives" in result.final_state["summary"]
    # Merge happens after every branch reported.
    order = [e.node for e in result.events_of("node_start")]
    assert order.index("merge") > max(i for i, n in enumerate(order) if n == "analyst")


def test_fanout_custom_perspectives():
    graph = build_fanout(perspectives=["red-team", "blue-team"])
    result = run_pattern(graph, make_input("Ship it?"))
    assert len(result.final_state["analyses"]) == 2
