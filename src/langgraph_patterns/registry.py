"""Pattern registry.

Every pattern module defines an :class:`PatternInfo` named ``INFO``; the
registry collects them so callers (CLI, tests, the playground backend) can
enumerate patterns, render diagrams, and build graphs uniformly.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict


class PatternInfo(BaseModel):
    """Metadata + factory for one pattern."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    title: str
    category: Literal["orchestration", "control-flow", "memory"]
    description: str
    mermaid: str
    default_text: str
    supports_hitl: bool = False
    needs_checkpointer: bool = False
    #: build(**kwargs) -> compiled LangGraph graph
    build: Callable[..., Any]
    #: make_input(text) -> graph input dict for a free-text task
    make_input: Callable[[str], dict[str, Any]]


_PATTERN_MODULES = [
    "langgraph_patterns.patterns.supervisor",
    "langgraph_patterns.patterns.react",
    "langgraph_patterns.patterns.reflection",
    "langgraph_patterns.patterns.hierarchical",
    "langgraph_patterns.patterns.plan_execute",
    "langgraph_patterns.patterns.hitl",
    "langgraph_patterns.patterns.routing",
    "langgraph_patterns.patterns.fanout",
    "langgraph_patterns.memory.episodic",
    "langgraph_patterns.memory.checkpointing",
]


def get_registry() -> dict[str, PatternInfo]:
    """Return ``{pattern_name: PatternInfo}`` for every available pattern."""
    import importlib

    registry: dict[str, PatternInfo] = {}
    for module_name in _PATTERN_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue  # pattern not shipped yet (phased build)
        info = getattr(module, "INFO", None)
        if info is not None:
            registry[info.name] = info
    return registry
