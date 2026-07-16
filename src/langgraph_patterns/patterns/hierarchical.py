"""Hierarchical delegation pattern — a two-level agent tree.

A director routes work to mid-level managers; each manager delegates to its
own workers and reports back up. Authority is scoped: workers only ever talk
to their manager, managers only to the director.

Guardrails: ``max_iterations`` bounds the total number of routing decisions
across both levels; unparseable decisions bubble up (worker → manager →
director → finish) instead of looping.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import get_chat_model, last_human_text
from langgraph_patterns.patterns.supervisor import FINISH, message_text
from langgraph_patterns.registry import PatternInfo

DONE = "DONE"


class WorkerSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str
    responder: Callable[[list[BaseMessage]], AIMessage] | None = None
    model: BaseChatModel | None = None


class ManagerSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    workers: list[WorkerSpec]


class HierarchicalState(BaseModel):
    """Shared typed state for the hierarchical delegation pattern."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    route: str = ""
    iterations: int = 0


def _named_responder(name: str, text: str) -> Callable[[list[BaseMessage]], AIMessage]:
    def responder(messages: list[BaseMessage]) -> AIMessage:
        task = last_human_text(messages)
        return AIMessage(content=f"[{name}] {text} (task: {task[:80]!r})")

    return responder


DEFAULT_MANAGERS = [
    ManagerSpec(
        name="research_manager",
        description="Owns fact-finding: delegates to searcher and summarizer.",
        workers=[
            WorkerSpec(
                name="searcher",
                description="Finds raw source material.",
                system_prompt="You find raw source material for the task. Reply with findings only.",
                responder=_named_responder("searcher", "found 3 relevant sources"),
            ),
            WorkerSpec(
                name="summarizer",
                description="Condenses findings into key points.",
                system_prompt="You condense the findings in this conversation into key points.",
                responder=_named_responder("summarizer", "condensed findings into 3 key points"),
            ),
        ],
    ),
    ManagerSpec(
        name="writing_manager",
        description="Owns the deliverable: delegates to drafter and editor.",
        workers=[
            WorkerSpec(
                name="drafter",
                description="Writes the first draft from the research.",
                system_prompt="You write a first draft using the research in this conversation.",
                responder=_named_responder("drafter", "wrote the first draft from the key points"),
            ),
            WorkerSpec(
                name="editor",
                description="Polishes the draft into the final answer.",
                system_prompt="You polish the latest draft into a final answer.",
                responder=_named_responder("editor", "polished the draft into the final answer"),
            ),
        ],
    ),
]


def build_mermaid(managers: Sequence[ManagerSpec]) -> str:
    lines = ["graph TD", "    __start__([start]) --> director{director}"]
    for manager in managers:
        lines.append(f"    director -->|delegate| {manager.name}{{{manager.name}}}")
        lines.append(f"    {manager.name} -->|report| director")
        for worker in manager.workers:
            lines.append(f"    {manager.name} -->|assign| {worker.name}[{worker.name}]")
            lines.append(f"    {worker.name} --> {manager.name}")
    lines.append(f"    director -->|{FINISH}| __end__([end])")
    return "\n".join(lines)


MERMAID = build_mermaid(DEFAULT_MANAGERS)


def _spoken(messages: Sequence[AnyMessage]) -> set[str]:
    return {m.name for m in messages if isinstance(m, AIMessage) and m.name}


def build_hierarchical(
    managers: Sequence[ManagerSpec] | None = None,
    *,
    max_iterations: int = 12,
    director_model: BaseChatModel | None = None,
    manager_models: dict[str, BaseChatModel] | None = None,
    checkpointer: Any = None,
):
    """Build a compiled two-level hierarchical delegation graph.

    Args:
        managers: mid-level managers, each with scoped workers. Defaults to a
            research team + writing team demo tree.
        max_iterations: hard budget on routing decisions across both levels.
        director_model: routing model for the top level.
        manager_models: optional per-manager routing models.
        checkpointer: optional LangGraph checkpointer.
    """
    managers = list(managers or DEFAULT_MANAGERS)
    manager_names = [m.name for m in managers]
    manager_models = manager_models or {}

    def _fake_director(messages: list[BaseMessage]) -> AIMessage:
        seen = _spoken([m for m in messages if isinstance(m, BaseMessage)])
        for manager in managers:
            if not all(w.name in seen for w in manager.workers):
                return AIMessage(content=manager.name)
        return AIMessage(content=FINISH)

    def _fake_manager(manager: ManagerSpec) -> Callable[[list[BaseMessage]], AIMessage]:
        def responder(messages: list[BaseMessage]) -> AIMessage:
            seen = _spoken([m for m in messages if isinstance(m, BaseMessage)])
            for worker in manager.workers:
                if worker.name not in seen:
                    return AIMessage(content=worker.name)
            return AIMessage(content=DONE)

        return responder

    director = director_model or get_chat_model(_fake_director)
    director_system = SystemMessage(
        content=(
            "You are the director of an agent organization. Managers:\n"
            + "\n".join(f"- {m.name}: {m.description}" for m in managers)
            + f"\n\nReply with exactly one word: the manager to delegate to next, "
            f"or {FINISH} when the user's request is fully handled."
        )
    )

    def director_node(state: HierarchicalState) -> dict[str, Any]:
        if state.iterations >= max_iterations:
            emit_agent_message(
                "director", f"Iteration budget ({max_iterations}) reached — finishing."
            )
            return {"route": FINISH, "iterations": state.iterations + 1}
        reply = director.invoke([director_system, *state.messages])
        text = message_text(reply).strip()
        choice = next(
            (n for n in manager_names if n.lower() in text.lower()), FINISH
        )
        emit_agent_message(
            "director",
            f"Delegating to {choice}" if choice != FINISH else "Work complete — finishing.",
        )
        return {"route": choice, "iterations": state.iterations + 1}

    graph = StateGraph(HierarchicalState)
    graph.add_node("director", director_node)
    graph.add_edge(START, "director")

    for manager in managers:
        worker_names = [w.name for w in manager.workers]
        manager_model = manager_models.get(manager.name) or get_chat_model(
            _fake_manager(manager)
        )
        manager_system = SystemMessage(
            content=(
                f"You are {manager.name}: {manager.description}\nWorkers:\n"
                + "\n".join(f"- {w.name}: {w.description}" for w in manager.workers)
                + f"\n\nReply with exactly one word: the worker to assign next, or "
                f"{DONE} when your part of the task is complete."
            )
        )

        def manager_node(
            state: HierarchicalState,
            *,
            _model: BaseChatModel = manager_model,
            _system: SystemMessage = manager_system,
            _name: str = manager.name,
            _workers: list[str] = worker_names,
        ) -> dict[str, Any]:
            if state.iterations >= max_iterations:
                emit_agent_message(
                    _name, f"Iteration budget ({max_iterations}) reached — reporting up."
                )
                return {"route": DONE, "iterations": state.iterations + 1}
            reply = _model.invoke([_system, *state.messages])
            text = message_text(reply).strip()
            choice = next((n for n in _workers if n.lower() in text.lower()), DONE)
            emit_agent_message(
                _name,
                f"Assigning {choice}" if choice != DONE else "Team done — reporting to director.",
            )
            return {"route": choice, "iterations": state.iterations + 1}

        def manager_route(
            state: HierarchicalState, *, _workers: list[str] = worker_names
        ) -> str:
            return state.route if state.route in _workers else "director"

        graph.add_node(manager.name, manager_node)
        graph.add_conditional_edges(
            manager.name,
            manager_route,
            {**{n: n for n in worker_names}, "director": "director"},
        )

        for worker in manager.workers:
            worker_model = worker.model or get_chat_model(
                worker.responder
                or _named_responder(worker.name, "completed the assignment")
            )

            def worker_node(
                state: HierarchicalState,
                *,
                _model: BaseChatModel = worker_model,
                _spec: WorkerSpec = worker,
            ) -> dict[str, Any]:
                reply = _model.invoke(
                    [SystemMessage(content=_spec.system_prompt), *state.messages]
                )
                text = message_text(reply)
                emit_agent_message(_spec.name, text)
                return {"messages": [AIMessage(content=text, name=_spec.name)]}

            graph.add_node(worker.name, worker_node)
            graph.add_edge(worker.name, manager.name)

    def director_route(state: HierarchicalState) -> str:
        return state.route if state.route in manager_names else END

    graph.add_conditional_edges(
        "director", director_route, {**{n: n for n in manager_names}, END: END}
    )
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=text)]}


INFO = PatternInfo(
    name="hierarchical",
    title="Hierarchical Delegation",
    category="orchestration",
    description=(
        "Two-level agent tree: a director delegates to managers, managers assign "
        "scoped workers and report back up, all within one iteration budget."
    ),
    mermaid=MERMAID,
    default_text="Produce a short researched write-up on agent memory.",
    build=build_hierarchical,
    make_input=make_input,
)
