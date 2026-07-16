from langgraph.types import Command

from langgraph_patterns.events import run_pattern
from langgraph_patterns.patterns.hitl import build_hitl, make_input


def test_hitl_interrupts_then_resumes_approved():
    graph = build_hitl()
    thread_id = "hitl-approve"

    paused = run_pattern(graph, make_input("Archive stale accounts."), thread_id=thread_id)
    assert paused.interrupted is True
    interrupts = paused.events_of("interrupt")
    assert len(interrupts) == 1
    assert "Proposal" in interrupts[0].data["payload"]["proposal"]
    assert interrupts[0].data["payload"]["question"] == "Approve this action?"
    # Paused before any execution (the result channel was never written).
    assert paused.final_state.get("result", "") == ""

    resumed = run_pattern(
        graph,
        Command(resume={"approved": True, "feedback": "looks safe"}),
        thread_id=thread_id,
    )
    assert resumed.interrupted is False
    assert resumed.final_state["approved"] is True
    assert resumed.final_state["result"].startswith("Executed approved action")
    assert {"human_gate", "execute"} <= {e.node for e in resumed.events_of("node_start")}


def test_hitl_rejection_aborts():
    graph = build_hitl()
    thread_id = "hitl-reject"
    run_pattern(graph, make_input("Delete production database."), thread_id=thread_id)
    resumed = run_pattern(
        graph,
        Command(resume={"approved": False, "feedback": "too risky"}),
        thread_id=thread_id,
    )
    assert resumed.final_state["approved"] is False
    assert "Aborted by human" in resumed.final_state["result"]
    assert "too risky" in resumed.final_state["result"]
    assert "abort" in {e.node for e in resumed.events_of("node_start")}
