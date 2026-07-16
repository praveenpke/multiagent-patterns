"""Demo tools used by the tool-calling patterns.

All tools are local and deterministic (no network, no side effects) so the
library stays runnable keyless and offline. Swap in your own LangChain tools
when using the patterns for real work.
"""

from __future__ import annotations

import ast
import datetime
import operator

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. "2 * (3 + 4)"."""
    try:
        value = _safe_eval(ast.parse(expression, mode="eval"))
    except Exception as exc:
        return f"ERROR: could not evaluate {expression!r}: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


@tool
def web_search(query: str) -> str:
    """Search the web for a query (offline demo stub with canned snippets)."""
    return (
        f"[demo search results for {query!r}] "
        "1. LangGraph docs — build stateful multi-agent applications as graphs. "
        "2. Multi-agent patterns — supervisor, reflection, plan-and-execute, ReAct. "
        "3. Checkpointing enables durable, resumable agent runs."
    )


@tool
def current_time() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


DEFAULT_TOOLS = [calculator, web_search, current_time]
