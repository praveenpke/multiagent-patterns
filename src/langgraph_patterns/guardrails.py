"""Guardrails shared across patterns.

Two kinds of guardrails ship with the library:

* **Step budgets** — every looping pattern takes a ``max_iterations`` /
  ``max_steps`` argument and enforces it inside its conditional edges, so a
  runaway model can never spin a graph forever.
* **Tool allowlists** — agents that use tools declare which tools they may
  call. :func:`enforce_tool_allowlist` validates at build time, and
  :func:`run_tool_safely` re-checks at call time (a model hallucinating a tool
  name yields a structured error result instead of arbitrary execution).
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.tools import BaseTool

from langgraph_patterns.events import emit_tool_call, emit_tool_result


class ToolNotAllowedError(Exception):
    """Raised when an agent is configured or asks for a tool outside its allowlist."""


def enforce_tool_allowlist(
    agent_name: str,
    tools: Sequence[BaseTool],
    allowlist: Sequence[str] | None,
) -> list[BaseTool]:
    """Validate that every tool given to *agent_name* is in its allowlist.

    ``allowlist=None`` means "no restriction". Returns the tools unchanged on
    success; raises :class:`ToolNotAllowedError` otherwise.
    """
    if allowlist is None:
        return list(tools)
    allowed = set(allowlist)
    for tool in tools:
        if tool.name not in allowed:
            raise ToolNotAllowedError(
                f"Agent {agent_name!r} was given tool {tool.name!r} which is not in "
                f"its allowlist {sorted(allowed)}"
            )
    return list(tools)


def run_tool_safely(
    agent_name: str,
    tool_call: dict[str, Any],
    tools: Sequence[BaseTool],
    allowlist: Sequence[str] | None = None,
) -> str:
    """Execute one model-requested tool call with allowlist + existence checks.

    Emits ``tool_call`` and ``tool_result`` events. Errors (unknown tool, tool
    outside allowlist, tool raised) come back as error strings so the graph
    can feed them to the model instead of crashing the run.
    """
    name = tool_call.get("name", "")
    args = tool_call.get("args") or {}
    emit_tool_call(agent_name, name, args)

    by_name = {t.name: t for t in tools}
    if allowlist is not None and name not in set(allowlist):
        result = f"ERROR: tool {name!r} is not in the allowlist for agent {agent_name!r}"
    elif name not in by_name:
        result = f"ERROR: unknown tool {name!r}"
    else:
        try:
            result = str(by_name[name].invoke(args))
        except Exception as exc:
            result = f"ERROR: tool {name!r} failed: {exc}"

    emit_tool_result(agent_name, name, result)
    return result
