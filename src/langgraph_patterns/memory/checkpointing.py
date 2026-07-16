"""Checkpointing pattern — durable, resumable runs via a SQLite checkpointer.

:func:`sqlite_checkpointer` returns a ``SqliteSaver`` backed by a file, and
the demo graph is a three-stage pipeline whose state (an append-only stage
log) is checkpointed after every superstep. A run stopped mid-pipeline —
process crash included — resumes from the last completed stage by invoking
the graph with ``None`` on the same ``thread_id``, even from a brand-new
graph instance.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

import operator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.registry import PatternInfo

STAGES = ["ingest", "transform", "report"]


def sqlite_checkpointer(path: str) -> SqliteSaver:
    """Create a SQLite-backed checkpointer at *path* (``':memory:'`` works too)."""
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)


class PipelineState(BaseModel):
    """Shared typed state for the checkpointing pattern."""

    task: str = ""
    completed: Annotated[list[str], operator.add] = Field(default_factory=list)
    report: str = ""


MERMAID = """graph TD
    __start__([start]) --> ingest[stage 1: ingest]
    ingest --> transform[stage 2: transform]
    transform --> report[stage 3: report]
    report --> __end__([end])
    checkpoints[(SQLite checkpoints)] -.saved after every stage.-> ingest
    checkpoints -.resume on same thread_id.-> transform"""


def build_checkpointing(*, checkpointer: Any = None, db_path: str | None = None):
    """Build a compiled checkpointed pipeline graph.

    Args:
        checkpointer: explicit LangGraph checkpointer. Defaults to a SQLite
            checkpointer at *db_path*, or an in-memory saver when neither is
            given.
        db_path: path for the default SQLite checkpointer file.

    Resume semantics: run with a ``thread_id``; if the run stops partway
    (crash, static interrupt, ...), invoke the graph again with input
    ``None`` and the same ``thread_id`` to continue from the last checkpoint.
    """
    if checkpointer is None:
        if db_path is not None:
            checkpointer = sqlite_checkpointer(db_path)
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()

    def make_stage(name: str, index: int):
        def stage(state: PipelineState) -> dict[str, Any]:
            emit_agent_message(
                "pipeline",
                f"Stage {index + 1}/{len(STAGES)} ({name}) complete for task {state.task[:60]!r}. "
                "Checkpoint saved.",
            )
            update: dict[str, Any] = {"completed": [name]}
            if name == "report":
                update["report"] = (
                    f"Report for {state.task!r}: stages {state.completed + [name]} "
                    "all completed (resumable at every stage)."
                )
            return update

        return stage

    graph = StateGraph(PipelineState)
    previous = START
    for i, name in enumerate(STAGES):
        graph.add_node(name, make_stage(name, i))
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    return {"task": text}


INFO = PatternInfo(
    name="checkpointing",
    title="Checkpointing",
    category="memory",
    description=(
        "A pipeline checkpointed to SQLite after every stage: stop it anywhere "
        "and resume on the same thread_id — even from a new process."
    ),
    mermaid=MERMAID,
    default_text="Process the quarterly usage data.",
    needs_checkpointer=True,
    build=build_checkpointing,
    make_input=make_input,
)
