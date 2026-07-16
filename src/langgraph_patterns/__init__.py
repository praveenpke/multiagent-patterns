"""langgraph_patterns — reusable LangGraph multi-agent orchestration patterns.

Every pattern is a factory function that returns a compiled LangGraph graph
with a typed Pydantic state schema, a Mermaid diagram, guardrails, and a
structured event stream consumable via :func:`stream_events`.

Patterns run against Anthropic Claude when ``ANTHROPIC_API_KEY`` is set and
against deterministic scripted fake models otherwise, so everything works
keyless (tests, examples, and the playground included).
"""

from langgraph_patterns.events import Event, RunResult, run_pattern, stream_events
from langgraph_patterns.guardrails import ToolNotAllowedError, enforce_tool_allowlist
from langgraph_patterns.models import ScriptedChatModel, get_chat_model

__all__ = [
    "Event",
    "RunResult",
    "run_pattern",
    "stream_events",
    "ToolNotAllowedError",
    "enforce_tool_allowlist",
    "ScriptedChatModel",
    "get_chat_model",
]

__version__ = "0.1.0"
