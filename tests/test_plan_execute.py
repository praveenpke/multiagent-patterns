from langchain_core.messages import AIMessage

from langgraph_patterns.events import run_pattern
from langgraph_patterns.models import ScriptedChatModel
from langgraph_patterns.patterns.plan_execute import build_plan_execute, make_input


def test_plan_execute_runs_all_steps_then_synthesizes():
    graph = build_plan_execute()
    result = run_pattern(graph, make_input("Compare supervisor and hierarchical."))

    assert len(result.final_state["plan"]) == 3
    assert len(result.final_state["results"]) == 3
    assert result.final_state["current_step"] == 3
    assert "3 executed plan steps" in result.final_state["final_answer"]

    executor_events = [
        e for e in result.events_of("agent_message") if e.agent == "executor"
    ]
    assert len(executor_events) == 3
    order = [e.node for e in result.events_of("node_start")]
    assert order[0] == "planner" and order[-1] == "synthesizer"


def test_plan_execute_truncates_plan_to_step_budget():
    planner = ScriptedChatModel(
        responder=lambda msgs: AIMessage(
            content="\n".join(f"{i}. step {i}" for i in range(1, 9))
        )
    )
    graph = build_plan_execute(max_steps=2, planner_model=planner)
    result = run_pattern(graph, make_input("big task"))
    assert len(result.final_state["plan"]) == 2
    assert len(result.final_state["results"]) == 2
    truncation = [
        e
        for e in result.events_of("agent_message")
        if e.agent == "planner" and "truncating" in e.data["content"]
    ]
    assert truncation
