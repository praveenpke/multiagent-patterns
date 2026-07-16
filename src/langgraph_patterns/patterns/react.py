"""ReAct pattern — a Reason-Act-Observe loop for tool-using agents.

The agent node reasons over the conversation and either emits tool calls or a
final answer. A tool-executor node runs each requested tool through
:func:`langgraph_patterns.guardrails.run_tool_safely`, which enforces the
per-agent tool allowlist and converts failures into structured error strings
the model can react to.

Guardrails: ``max_steps`` bounds the reason/act loop; tools are validated
against the allowlist at build time and again per call at run time.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.guardrails import enforce_tool_allowlist, run_tool_safely
from langgraph_patterns.models import get_chat_model, last_human_text
from langgraph_patterns.patterns.supervisor import message_text
from langgraph_patterns.registry import PatternInfo
from langgraph_patterns.tools import DEFAULT_TOOLS

AGENT_NAME = "react-agent"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they help you "
    "answer, then give a concise final answer."
)

_MATH_RE = re.compile(r"[-+]?[\d(][\d\s.+\-*/()%]*[\d)]")


def _fake_responder(messages: list[BaseMessage]) -> AIMessage:
    """Deterministic ReAct script: one tool call, then a final answer."""
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    if tool_results:
        last = tool_results[-1]
        return AIMessage(
            content=(
                f"Using the {last.name} result — {str(last.content)[:200]} — "
                f"the answer to {last_human_text(messages)!r} is above."
            )
        )
    question = last_human_text(messages)
    match = _MATH_RE.search(question)
    if match and any(op in match.group(0) for op in "+-*/%"):
        return AIMessage(
            content="I will compute this with the calculator.",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": match.group(0).strip()},
                    "id": "call_1",
                }
            ],
        )
    return AIMessage(
        content="I will look this up.",
        tool_calls=[{"name": "web_search", "args": {"query": question}, "id": "call_1"}],
    )


class ReActState(BaseModel):
    """Shared typed state for the ReAct pattern."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    steps: int = 0


MERMAID = """graph TD
    __start__([start]) --> agent[agent: reason]
    agent -->|tool calls| tools[tools: act]
    tools -->|observation| agent
    agent -->|final answer / step budget| __end__([end])"""


def build_react(
    tools: Sequence[BaseTool] | None = None,
    *,
    max_steps: int = 6,
    model: BaseChatModel | None = None,
    tool_allowlist: Sequence[str] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    checkpointer: Any = None,
):
    """Build a compiled ReAct graph.

    Args:
        tools: LangChain tools; defaults to the offline demo tools
            (calculator, web_search, current_time).
        max_steps: hard budget on agent reasoning steps (guardrail).
        model: chat model; defaults to Claude when configured, else a
            scripted model that performs one tool call then answers.
        tool_allowlist: tool names this agent may call. Validated at build
            time and re-checked on every call. ``None`` allows the given tools.
        system_prompt: agent system prompt.
        checkpointer: optional LangGraph checkpointer.
    """
    tools = list(tools if tools is not None else DEFAULT_TOOLS)
    tools = enforce_tool_allowlist(AGENT_NAME, tools, tool_allowlist)
    allowlist = list(tool_allowlist) if tool_allowlist is not None else [
        t.name for t in tools
    ]
    base_model = model or get_chat_model(_fake_responder)
    bound = base_model.bind_tools(tools) if tools else base_model
    system = SystemMessage(content=system_prompt)

    def agent_node(state: ReActState) -> dict[str, Any]:
        reply = bound.invoke([system, *state.messages])
        text = message_text(reply)
        if text:
            emit_agent_message(AGENT_NAME, text)
        return {"messages": [reply], "steps": state.steps + 1}

    def tools_node(state: ReActState) -> dict[str, Any]:
        last = state.messages[-1]
        results: list[ToolMessage] = []
        for tool_call in getattr(last, "tool_calls", None) or []:
            output = run_tool_safely(AGENT_NAME, tool_call, tools, allowlist)
            results.append(
                ToolMessage(
                    content=output,
                    name=tool_call.get("name", ""),
                    tool_call_id=tool_call.get("id", ""),
                )
            )
        return {"messages": results}

    def should_continue(state: ReActState) -> str:
        last = state.messages[-1]
        if state.steps >= max_steps:
            emit_agent_message(
                AGENT_NAME, f"Step budget ({max_steps}) reached — stopping."
            )
            return END
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=text)]}


INFO = PatternInfo(
    name="react",
    title="ReAct",
    category="orchestration",
    description=(
        "Reason-Act-Observe loop: the agent alternates between reasoning and "
        "allowlisted tool calls until it can answer, within a step budget."
    ),
    mermaid=MERMAID,
    default_text="What is 12 * (7 + 5)?",
    build=build_react,
    make_input=make_input,
)
