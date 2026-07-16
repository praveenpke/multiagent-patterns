"""Structured event stream shared by every pattern.

Patterns emit ``agent_message`` / ``tool_call`` / ``tool_result`` events from
inside their nodes (via :func:`emit_event`), and :func:`stream_events` merges
those with node lifecycle events synthesized from LangGraph's debug stream.

The playground backend forwards these events verbatim over SSE, and the tests
assert on them, so the schema is intentionally small and stable:

``run_start``      → run began (data: input)
``node_start``     → a graph node started (node)
``node_end``       → a graph node finished (node)
``agent_message``  → an agent produced text (agent, data.content)
``tool_call``      → an agent requested a tool (agent, data.tool, data.args)
``tool_result``    → a tool returned (agent, data.tool, data.result)
``interrupt``      → graph paused for human input (data.payload, data.thread_id)
``run_end``        → run finished (data.final_state, data.interrupted)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Iterator, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

EventType = Literal[
    "run_start",
    "node_start",
    "node_end",
    "agent_message",
    "tool_call",
    "tool_result",
    "interrupt",
    "run_end",
    "error",
]

#: Internal LangGraph node names that should not surface in the timeline.
_HIDDEN_NODES = {"__start__", "__end__"}


class Event(BaseModel):
    """A single structured event emitted while a pattern graph runs."""

    type: EventType
    node: str | None = None
    agent: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)


class RunResult(BaseModel):
    """Collected result of :func:`run_pattern`."""

    events: list[Event]
    final_state: dict[str, Any]
    interrupted: bool = False
    thread_id: str | None = None

    def events_of(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type == event_type]


def emit_event(
    type: EventType,
    *,
    node: str | None = None,
    agent: str | None = None,
    **data: Any,
) -> None:
    """Emit a custom event from inside a graph node.

    Safe to call outside a LangGraph runtime (no-op) so nodes stay usable in
    plain ``graph.invoke`` calls too.
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    writer({"type": type, "node": node, "agent": agent, "data": data})


def emit_agent_message(agent: str, content: str, **extra: Any) -> None:
    emit_event("agent_message", agent=agent, content=content, **extra)


def emit_tool_call(agent: str, tool: str, args: dict[str, Any]) -> None:
    emit_event("tool_call", agent=agent, tool=tool, args=args)


def emit_tool_result(agent: str, tool: str, result: Any) -> None:
    emit_event("tool_result", agent=agent, tool=tool, result=serialize(result))


def serialize(value: Any) -> Any:
    """Convert graph state values (messages, models, ...) to JSON-safe data."""
    if isinstance(value, BaseMessage):
        out: dict[str, Any] = {"type": value.type, "content": value.content}
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
            ]
        return out
    if isinstance(value, BaseModel):
        return serialize(value.model_dump())
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _ensure_config(
    config: dict[str, Any] | None, thread_id: str | None
) -> tuple[dict[str, Any], str | None]:
    config = dict(config or {})
    configurable = dict(config.get("configurable") or {})
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    config["configurable"] = configurable
    return config, configurable.get("thread_id")


def stream_events(
    graph: Any,
    input: Any,
    *,
    config: dict[str, Any] | None = None,
    thread_id: str | None = None,
) -> Iterator[Event]:
    """Run a compiled pattern graph and yield structured :class:`Event`s.

    Works for fresh runs and for resuming interrupted runs (pass a
    ``langgraph.types.Command(resume=...)`` as ``input`` with the same
    ``thread_id``). The final ``run_end`` event carries the serialized final
    state and whether the run is paused on an interrupt.
    """
    config, tid = _ensure_config(config, thread_id)
    yield Event(type="run_start", data={"input": serialize(input), "thread_id": tid})

    last_values: Any = None
    interrupted = False
    interrupt_payload: Any = None

    try:
        for mode, chunk in graph.stream(
            input, config, stream_mode=["debug", "custom", "updates", "values"]
        ):
            if mode == "debug":
                kind = chunk.get("type")
                payload = chunk.get("payload", {})
                name = payload.get("name")
                if name in _HIDDEN_NODES:
                    continue
                if kind == "task":
                    yield Event(type="node_start", node=name)
                elif kind == "task_result":
                    yield Event(type="node_end", node=name)
            elif mode == "custom":
                if isinstance(chunk, dict) and "type" in chunk:
                    yield Event(
                        type=chunk["type"],
                        node=chunk.get("node"),
                        agent=chunk.get("agent"),
                        data=chunk.get("data") or {},
                    )
            elif mode == "updates":
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    interrupted = True
                    interrupts = chunk["__interrupt__"]
                    values = [getattr(i, "value", i) for i in interrupts]
                    interrupt_payload = values[0] if len(values) == 1 else values
                    yield Event(
                        type="interrupt",
                        data={
                            "payload": serialize(interrupt_payload),
                            "thread_id": tid,
                        },
                    )
            elif mode == "values":
                last_values = chunk
    except Exception as exc:  # surface failures as events, then re-raise
        yield Event(type="error", data={"message": f"{type(exc).__name__}: {exc}"})
        raise

    yield Event(
        type="run_end",
        data={
            "final_state": serialize(last_values) if last_values is not None else {},
            "interrupted": interrupted,
            "thread_id": tid,
        },
    )


def run_pattern(
    graph: Any,
    input: Any,
    *,
    config: dict[str, Any] | None = None,
    thread_id: str | None = None,
) -> RunResult:
    """Convenience wrapper: run a graph, collect events, return the result."""
    if thread_id is None and getattr(graph, "checkpointer", None) is not None:
        thread_id = uuid.uuid4().hex
    events = list(stream_events(graph, input, config=config, thread_id=thread_id))
    run_end = events[-1]
    return RunResult(
        events=events,
        final_state=run_end.data.get("final_state", {}),
        interrupted=bool(run_end.data.get("interrupted")),
        thread_id=thread_id,
    )
