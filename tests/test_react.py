import pytest
from langchain_core.messages import AIMessage

from langgraph_patterns.events import run_pattern
from langgraph_patterns.guardrails import ToolNotAllowedError
from langgraph_patterns.models import ScriptedChatModel
from langgraph_patterns.patterns.react import build_react, make_input
from langgraph_patterns.tools import calculator, web_search


def test_react_calculates_with_tool_and_event_stream():
    graph = build_react()
    result = run_pattern(graph, make_input("What is 12 * (7 + 5)?"))

    tool_calls = result.events_of("tool_call")
    tool_results = result.events_of("tool_result")
    assert len(tool_calls) == 1 and tool_calls[0].data["tool"] == "calculator"
    assert tool_results[0].data["result"] == "144"

    # Final answer references the observation; loop stayed within budget.
    final_messages = result.final_state["messages"]
    assert "144" in final_messages[-1]["content"]
    assert result.final_state["steps"] <= 6

    nodes = {e.node for e in result.events_of("node_start")}
    assert {"agent", "tools"} <= nodes


def test_react_build_time_allowlist_rejects_unlisted_tool():
    with pytest.raises(ToolNotAllowedError):
        build_react(tools=[calculator, web_search], tool_allowlist=["web_search"])


def test_react_runtime_allowlist_blocks_hallucinated_tool():
    # A model that asks for a tool that exists but is outside its allowlist.
    def responder(messages):
        if any(m.type == "tool" for m in messages):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1"}],
        )

    graph = build_react(
        tools=[calculator],
        tool_allowlist=["calculator"],
        model=ScriptedChatModel(responder=responder),
    )
    result = run_pattern(graph, make_input("hi"))
    blocked = result.events_of("tool_result")[0]
    assert "not in the allowlist" in blocked.data["result"]


def test_react_step_budget_guardrail():
    # A model that always asks for another tool call would loop forever
    # without the budget.
    def responder(messages):
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}
            ],
        )

    graph = build_react(max_steps=3, model=ScriptedChatModel(responder=responder))
    result = run_pattern(graph, make_input("loop forever"))
    assert result.final_state["steps"] == 3
    assert result.events[-1].type == "run_end"
