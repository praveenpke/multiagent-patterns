"""Reflection pattern — critique-and-rewrite loop with a max-iteration budget.

A generator drafts an answer; a critic either approves it (``APPROVED``) or
returns actionable critique that feeds the next draft. The loop ends on
approval or when the iteration budget is exhausted, whichever comes first.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import get_chat_model
from langgraph_patterns.patterns.supervisor import message_text
from langgraph_patterns.registry import PatternInfo

APPROVED = "APPROVED"

GENERATOR_SYSTEM = (
    "You are a writer. Produce the best possible answer for the task. If a "
    "previous critique is provided, revise your draft to address every point."
)
CRITIC_SYSTEM = (
    "You are a strict reviewer. If the draft fully answers the task, reply with "
    f"exactly {APPROVED}. Otherwise reply with a short, actionable critique."
)


def _fake_generator(messages: list[BaseMessage]) -> AIMessage:
    prompt = str(messages[-1].content)
    if "Previous critique" in prompt:
        return AIMessage(
            content=(
                "Improved draft (v2): a structured answer with a clear thesis, "
                "supporting evidence, and a concrete example — critique addressed."
            )
        )
    return AIMessage(content="Rough draft (v1): a first-pass answer to the task.")


def _fake_critic(messages: list[BaseMessage]) -> AIMessage:
    prompt = str(messages[-1].content)
    if "(v1)" in prompt:
        return AIMessage(
            content="Too vague: add a thesis, evidence, and one concrete example."
        )
    return AIMessage(content=APPROVED)


class ReflectionState(BaseModel):
    """Shared typed state for the reflection pattern."""

    task: str = ""
    draft: str = ""
    critique: str = ""
    iterations: int = 0
    approved: bool = False


MERMAID = """graph TD
    __start__([start]) --> generate[generate draft]
    generate --> critique[critique]
    critique -->|needs work| generate
    critique -->|approved / budget| __end__([end])"""


def build_reflection(
    *,
    max_iterations: int = 3,
    generator_model: BaseChatModel | None = None,
    critic_model: BaseChatModel | None = None,
    checkpointer: Any = None,
):
    """Build a compiled reflection (critique-and-rewrite) graph.

    Args:
        max_iterations: hard budget on draft/critique rounds (guardrail).
        generator_model: drafting model (default: Claude or scripted fake).
        critic_model: reviewing model (default: Claude or scripted fake).
        checkpointer: optional LangGraph checkpointer.
    """
    generator = generator_model or get_chat_model(_fake_generator)
    critic = critic_model or get_chat_model(_fake_critic)

    def generate_node(state: ReflectionState) -> dict[str, Any]:
        prompt = f"Task: {state.task}"
        if state.critique:
            prompt += (
                f"\n\nPrevious draft:\n{state.draft}"
                f"\n\nPrevious critique (address every point):\n{state.critique}"
            )
        reply = generator.invoke(
            [SystemMessage(content=GENERATOR_SYSTEM), HumanMessage(content=prompt)]
        )
        draft = message_text(reply)
        emit_agent_message("generator", draft)
        return {"draft": draft, "iterations": state.iterations + 1}

    def critique_node(state: ReflectionState) -> dict[str, Any]:
        reply = critic.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(content=f"Task: {state.task}\n\nDraft:\n{state.draft}"),
            ]
        )
        verdict = message_text(reply).strip()
        approved = verdict.upper().startswith(APPROVED)
        emit_agent_message("critic", APPROVED if approved else verdict)
        return {"approved": approved, "critique": "" if approved else verdict}

    def should_continue(state: ReflectionState) -> str:
        if state.approved:
            return END
        if state.iterations >= max_iterations:
            emit_agent_message(
                "critic",
                f"Iteration budget ({max_iterations}) reached — returning last draft.",
            )
            return END
        return "generate"

    graph = StateGraph(ReflectionState)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique", should_continue, {"generate": "generate", END: END}
    )
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    return {"task": text}


INFO = PatternInfo(
    name="reflection",
    title="Reflection",
    category="orchestration",
    description=(
        "Critique-and-rewrite loop: a generator drafts, a critic approves or "
        "returns actionable feedback, bounded by a max-iterations budget."
    ),
    mermaid=MERMAID,
    default_text="Write a short explanation of why agent guardrails matter.",
    build=build_reflection,
    make_input=make_input,
)
