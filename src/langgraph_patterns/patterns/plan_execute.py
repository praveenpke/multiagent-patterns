"""Plan-and-execute pattern — a planner writes a step list, an executor runs it.

The planner produces a numbered plan; the executor completes one step per
superstep (so every step is visible in the event stream and checkpointable);
a synthesizer folds the step results into the final answer.

Guardrails: ``max_steps`` truncates over-long plans before execution begins.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import get_chat_model
from langgraph_patterns.patterns.supervisor import message_text
from langgraph_patterns.registry import PatternInfo

PLANNER_SYSTEM = (
    "You are a planner. Break the task into a short numbered list of concrete "
    "steps (3-5). Reply with the numbered steps only, one per line."
)
EXECUTOR_SYSTEM = (
    "You are an executor. Complete the given step using the task context and "
    "prior step results. Reply with the step result only."
)
SYNTHESIZER_SYSTEM = (
    "You are a synthesizer. Combine the step results into one final answer "
    "for the original task."
)


def _fake_planner(messages: list[BaseMessage]) -> AIMessage:
    task = str(messages[-1].content).removeprefix("Task: ")
    return AIMessage(
        content=(
            f"1. Clarify what {task[:60]!r} requires\n"
            "2. Gather the key facts\n"
            "3. Compose the answer"
        )
    )


def _fake_executor(messages: list[BaseMessage]) -> AIMessage:
    prompt = str(messages[-1].content)
    step = re.search(r"Current step: (.+)", prompt)
    return AIMessage(content=f"Completed: {step.group(1) if step else 'step'} — ok.")


def _fake_synthesizer(messages: list[BaseMessage]) -> AIMessage:
    prompt = str(messages[-1].content)
    n = prompt.count("Completed:")
    return AIMessage(
        content=f"Final answer synthesized from {n} executed plan steps."
    )


class PlanExecuteState(BaseModel):
    """Shared typed state for the plan-and-execute pattern."""

    task: str = ""
    plan: list[str] = Field(default_factory=list)
    current_step: int = 0
    results: list[str] = Field(default_factory=list)
    final_answer: str = ""


MERMAID = """graph TD
    __start__([start]) --> planner[planner]
    planner --> executor[execute step n]
    executor -->|more steps| executor
    executor -->|plan done| synthesizer[synthesizer]
    synthesizer --> __end__([end])"""


def _parse_plan(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^(?:\d+[\.\)]|[-*])\s*(.+)$", line)
        if match:
            steps.append(match.group(1).strip())
    return steps or ([text.strip()] if text.strip() else [])


def build_plan_execute(
    *,
    max_steps: int = 5,
    planner_model: BaseChatModel | None = None,
    executor_model: BaseChatModel | None = None,
    synthesizer_model: BaseChatModel | None = None,
    checkpointer: Any = None,
):
    """Build a compiled plan-and-execute graph.

    Args:
        max_steps: hard cap on executed plan steps (guardrail — longer plans
            are truncated with a visible event).
        planner_model / executor_model / synthesizer_model: chat models
            (default: Claude when configured, else deterministic fakes).
        checkpointer: optional LangGraph checkpointer.
    """
    planner = planner_model or get_chat_model(_fake_planner)
    executor = executor_model or get_chat_model(_fake_executor)
    synthesizer = synthesizer_model or get_chat_model(_fake_synthesizer)

    def planner_node(state: PlanExecuteState) -> dict[str, Any]:
        reply = planner.invoke(
            [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=f"Task: {state.task}")]
        )
        plan = _parse_plan(message_text(reply))
        if len(plan) > max_steps:
            emit_agent_message(
                "planner",
                f"Plan has {len(plan)} steps — truncating to budget of {max_steps}.",
            )
            plan = plan[:max_steps]
        emit_agent_message(
            "planner", "Plan:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        )
        return {"plan": plan, "current_step": 0, "results": []}

    def executor_node(state: PlanExecuteState) -> dict[str, Any]:
        step = state.plan[state.current_step]
        context = "\n".join(state.results)
        reply = executor.invoke(
            [
                SystemMessage(content=EXECUTOR_SYSTEM),
                HumanMessage(
                    content=(
                        f"Task: {state.task}\nPrior results:\n{context or '(none)'}\n"
                        f"Current step: {step}"
                    )
                ),
            ]
        )
        result = message_text(reply)
        emit_agent_message("executor", f"Step {state.current_step + 1}/{len(state.plan)}: {result}")
        return {"results": state.results + [result], "current_step": state.current_step + 1}

    def synthesizer_node(state: PlanExecuteState) -> dict[str, Any]:
        reply = synthesizer.invoke(
            [
                SystemMessage(content=SYNTHESIZER_SYSTEM),
                HumanMessage(
                    content=f"Task: {state.task}\nStep results:\n" + "\n".join(state.results)
                ),
            ]
        )
        answer = message_text(reply)
        emit_agent_message("synthesizer", answer)
        return {"final_answer": answer}

    def should_continue(state: PlanExecuteState) -> str:
        return "executor" if state.current_step < len(state.plan) else "synthesizer"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner", should_continue, {"executor": "executor", "synthesizer": "synthesizer"}
    )
    graph.add_conditional_edges(
        "executor", should_continue, {"executor": "executor", "synthesizer": "synthesizer"}
    )
    graph.add_edge("synthesizer", END)
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    return {"task": text}


INFO = PatternInfo(
    name="plan_execute",
    title="Plan and Execute",
    category="orchestration",
    description=(
        "A planner writes a numbered step list, an executor completes one step "
        "per superstep, and a synthesizer folds the results into the answer."
    ),
    mermaid=MERMAID,
    default_text="Compare supervisor and hierarchical orchestration.",
    build=build_plan_execute,
    make_input=make_input,
)
