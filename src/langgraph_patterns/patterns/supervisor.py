"""Supervisor pattern — a central router delegates to specialist subagents.

A supervisor node inspects the conversation and either routes to one of the
specialist agents or finishes. Each specialist appends a named ``AIMessage``
to the shared, typed message state and control returns to the supervisor.

Guardrails: ``max_iterations`` bounds the routing loop; the supervisor is
forced to ``FINISH`` once the budget is exhausted, and unparseable routing
decisions also finish rather than loop.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import Responder, get_chat_model, last_human_text
from langgraph_patterns.registry import PatternInfo

FINISH = "FINISH"


def message_text(message: BaseMessage) -> str:
    """Flatten message content (which may be a list of blocks) to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


class SupervisorState(BaseModel):
    """Shared typed state for the supervisor pattern."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    next_agent: str = ""
    iterations: int = 0


class AgentSpec(BaseModel):
    """A specialist agent managed by the supervisor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str
    #: deterministic fake-mode responder (messages -> AIMessage)
    responder: Callable[[list[BaseMessage]], AIMessage] | None = None
    #: optional explicit chat model (overrides get_chat_model)
    model: BaseChatModel | None = None


# ---------------------------------------------------------------------------
# Default demo agents (deterministic in fake mode)
# ---------------------------------------------------------------------------


def _researcher_responder(messages: list[BaseMessage]) -> AIMessage:
    task = last_human_text(messages)
    return AIMessage(
        content=(
            f"Research notes on {task!r}: the topic breaks down into three parts — "
            "background, current state of the art, and open questions. "
            "Key fact: multi-agent systems trade coordination overhead for specialization."
        )
    )


def _writer_responder(messages: list[BaseMessage]) -> AIMessage:
    task = last_human_text(messages)
    notes = ""
    for message in messages:
        if isinstance(message, AIMessage) and message.name == "researcher":
            notes = message_text(message)
    return AIMessage(
        content=(
            f"Final answer for {task!r}: drawing on the research"
            f"{' notes' if notes else ''}, specialization plus a routing supervisor "
            "gives you focused agents with a single point of control."
        )
    )


DEFAULT_AGENTS = [
    AgentSpec(
        name="researcher",
        description="Gathers facts and background information for the task.",
        system_prompt=(
            "You are a research specialist. Gather the key facts needed to answer "
            "the user's request. Reply with concise research notes only."
        ),
        responder=_researcher_responder,
    ),
    AgentSpec(
        name="writer",
        description="Writes the final polished answer using prior agent output.",
        system_prompt=(
            "You are a writing specialist. Using the conversation so far (including "
            "any research notes), write the final answer for the user."
        ),
        responder=_writer_responder,
    ),
]


def _default_router_responder(agents: Sequence[AgentSpec]) -> Responder:
    """Fake router: visit each agent once, in order, then FINISH."""

    def responder(messages: list[BaseMessage]) -> AIMessage:
        seen = {m.name for m in messages if isinstance(m, AIMessage) and m.name}
        for spec in agents:
            if spec.name not in seen:
                return AIMessage(content=spec.name)
        return AIMessage(content=FINISH)

    return responder


def _parse_route(text: str, agent_names: Sequence[str]) -> str:
    cleaned = text.strip()
    for name in agent_names:
        if name.lower() in cleaned.lower():
            return name
    return FINISH


def build_mermaid(agent_names: Sequence[str]) -> str:
    lines = ["graph TD", "    __start__([start]) --> supervisor{supervisor}"]
    for name in agent_names:
        lines.append(f"    supervisor -->|route| {name}[{name}]")
        lines.append(f"    {name} --> supervisor")
    lines.append(f"    supervisor -->|{FINISH}| __end__([end])")
    return "\n".join(lines)


MERMAID = build_mermaid([a.name for a in DEFAULT_AGENTS])


def build_supervisor(
    agents: Sequence[AgentSpec] | None = None,
    *,
    max_iterations: int = 6,
    router_model: BaseChatModel | None = None,
    checkpointer: Any = None,
):
    """Build a compiled supervisor graph.

    Args:
        agents: specialist agents; defaults to a researcher + writer demo pair.
        max_iterations: hard budget on supervisor routing turns (guardrail).
        router_model: chat model used for routing decisions; defaults to
            Claude when ``ANTHROPIC_API_KEY`` is set, else a scripted router
            that visits each agent once.
        checkpointer: optional LangGraph checkpointer.

    Returns:
        A compiled LangGraph graph with :class:`SupervisorState` state.
    """
    agents = list(agents or DEFAULT_AGENTS)
    agent_names = [a.name for a in agents]
    router = router_model or get_chat_model(_default_router_responder(agents))

    router_system = SystemMessage(
        content=(
            "You are a supervisor routing work between specialist agents.\n"
            "Agents:\n"
            + "\n".join(f"- {a.name}: {a.description}" for a in agents)
            + f"\n\nGiven the conversation, reply with exactly one word: the name of "
            f"the next agent to act, or {FINISH} when the user's request is fully "
            "answered."
        )
    )

    def supervisor_node(state: SupervisorState) -> dict[str, Any]:
        if state.iterations >= max_iterations:
            emit_agent_message(
                "supervisor",
                f"Iteration budget ({max_iterations}) reached — finishing.",
            )
            return {"next_agent": FINISH, "iterations": state.iterations + 1}
        decision = router.invoke([router_system, *state.messages])
        choice = _parse_route(message_text(decision), agent_names)
        emit_agent_message(
            "supervisor",
            f"Routing to {choice}" if choice != FINISH else "All done — finishing.",
        )
        return {"next_agent": choice, "iterations": state.iterations + 1}

    def route(state: SupervisorState) -> str:
        return state.next_agent if state.next_agent in agent_names else END

    def make_agent_node(spec: AgentSpec):
        model = spec.model or get_chat_model(
            spec.responder or (lambda msgs: AIMessage(content=f"[{spec.name}] done"))
        )

        def agent_node(state: SupervisorState) -> dict[str, Any]:
            reply = model.invoke(
                [SystemMessage(content=spec.system_prompt), *state.messages]
            )
            text = message_text(reply)
            emit_agent_message(spec.name, text)
            return {"messages": [AIMessage(content=text, name=spec.name)]}

        return agent_node

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    for spec in agents:
        graph.add_node(spec.name, make_agent_node(spec))
        graph.add_edge(spec.name, "supervisor")
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route, {**{n: n for n in agent_names}, END: END}
    )
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=text)]}


INFO = PatternInfo(
    name="supervisor",
    title="Supervisor",
    category="orchestration",
    description=(
        "A central supervisor routes the conversation between specialist agents "
        "(researcher, writer) until the task is complete, with a hard iteration budget."
    ),
    mermaid=MERMAID,
    default_text="Explain the trade-offs of multi-agent systems.",
    build=build_supervisor,
    make_input=make_input,
)
