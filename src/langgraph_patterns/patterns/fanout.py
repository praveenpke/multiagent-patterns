"""Parallel fan-out pattern — run analysts concurrently, merge their results.

A dispatch node fans the task out to one analyst per perspective using
LangGraph's ``Send`` API; all analysts run in the same superstep and their
outputs accumulate through an ``operator.add`` reducer; a merge node folds
them into a single summary.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import get_chat_model
from langgraph_patterns.patterns.supervisor import message_text
from langgraph_patterns.registry import PatternInfo

DEFAULT_PERSPECTIVES = ["optimist", "skeptic", "pragmatist"]

ANALYST_SYSTEM = (
    "You are the {perspective} analyst. Analyze the task strictly from that "
    "perspective in 2-3 sentences, prefixed with your perspective name."
)
MERGER_SYSTEM = (
    "You are a synthesis agent. Merge the parallel analyses into one balanced "
    "summary that references each perspective."
)


def _fake_analyst(messages: list[BaseMessage]) -> AIMessage:
    prompt = str(messages[-1].content)
    perspective = "analyst"
    task = ""
    for line in prompt.splitlines():
        if line.startswith("Perspective: "):
            perspective = line.removeprefix("Perspective: ").strip()
        if line.startswith("Task: "):
            task = line.removeprefix("Task: ").strip()
    return AIMessage(
        content=f"[{perspective}] From the {perspective} angle, {task[:60]!r} looks "
        f"{'promising' if perspective == 'optimist' else 'risky' if perspective == 'skeptic' else 'workable with constraints'}."
    )


def _fake_merger(messages: list[BaseMessage]) -> AIMessage:
    prompt = str(messages[-1].content)
    n = prompt.count("[")
    return AIMessage(content=f"Merged summary across {n} perspectives: balanced view combining optimism, skepticism, and pragmatism.")


class FanoutState(BaseModel):
    """Shared typed state for the parallel fan-out pattern.

    ``analyses`` uses an ``operator.add`` reducer so concurrent analyst
    branches can append without clobbering each other.
    """

    task: str = ""
    perspective: str = ""  # set per-branch via Send payloads
    analyses: Annotated[list[str], operator.add] = Field(default_factory=list)
    summary: str = ""


MERMAID = """graph TD
    __start__([start]) --> dispatch[dispatch]
    dispatch -->|Send| a1[analyst: optimist]
    dispatch -->|Send| a2[analyst: skeptic]
    dispatch -->|Send| a3[analyst: pragmatist]
    a1 --> merge[merge results]
    a2 --> merge
    a3 --> merge
    merge --> __end__([end])"""


def build_fanout(
    perspectives: Sequence[str] | None = None,
    *,
    analyst_model: BaseChatModel | None = None,
    merger_model: BaseChatModel | None = None,
    checkpointer: Any = None,
):
    """Build a compiled parallel fan-out / merge graph.

    Args:
        perspectives: one concurrent analyst branch per entry
            (default: optimist, skeptic, pragmatist).
        analyst_model: model shared by analyst branches.
        merger_model: model that merges the parallel analyses.
        checkpointer: optional LangGraph checkpointer.
    """
    perspectives = list(perspectives or DEFAULT_PERSPECTIVES)
    analyst = analyst_model or get_chat_model(_fake_analyst)
    merger = merger_model or get_chat_model(_fake_merger)

    def dispatch_node(state: FanoutState) -> dict[str, Any]:
        emit_agent_message(
            "dispatcher", f"Fanning out to {len(perspectives)} analysts: {', '.join(perspectives)}"
        )
        return {}

    def fan_out(state: FanoutState) -> list[Send]:
        # Send payloads go straight to the node, so pass a typed state instance.
        return [
            Send("analyst", FanoutState(task=state.task, perspective=p))
            for p in perspectives
        ]

    def analyst_node(state: FanoutState) -> dict[str, Any]:
        reply = analyst.invoke(
            [
                SystemMessage(content=ANALYST_SYSTEM.format(perspective=state.perspective)),
                HumanMessage(content=f"Perspective: {state.perspective}\nTask: {state.task}"),
            ]
        )
        text = message_text(reply)
        emit_agent_message(state.perspective, text)
        return {"analyses": [text]}

    def merge_node(state: FanoutState) -> dict[str, Any]:
        reply = merger.invoke(
            [
                SystemMessage(content=MERGER_SYSTEM),
                HumanMessage(
                    content=f"Task: {state.task}\nAnalyses:\n" + "\n".join(state.analyses)
                ),
            ]
        )
        summary = message_text(reply)
        emit_agent_message("merger", summary)
        return {"summary": summary}

    graph = StateGraph(FanoutState)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("merge", merge_node)
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", fan_out, ["analyst"])
    graph.add_edge("analyst", "merge")
    graph.add_edge("merge", END)
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    return {"task": text}


INFO = PatternInfo(
    name="fanout",
    title="Parallel Fan-out",
    category="control-flow",
    description=(
        "Dispatch fans the task out to concurrent analyst branches via the Send "
        "API; a reducer accumulates their outputs and a merge node summarizes."
    ),
    mermaid=MERMAID,
    default_text="Should we migrate our agents to a graph runtime?",
    build=build_fanout,
    make_input=make_input,
)
