"""Conditional routing pattern — dynamic edge selection from agent output.

A classifier labels the request and a conditional edge routes it to the
matching specialist handler (math, lookup, or chat). Unrecognized labels fall
back to the chat handler instead of failing.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import get_chat_model, last_human_text
from langgraph_patterns.patterns.supervisor import message_text
from langgraph_patterns.registry import PatternInfo
from langgraph_patterns.tools import calculator

ROUTES = ["math", "lookup", "chat"]

CLASSIFIER_SYSTEM = (
    "Classify the user's request. Reply with exactly one word:\n"
    "- math: arithmetic or numeric computation\n"
    "- lookup: a factual question that needs information retrieval\n"
    "- chat: greetings, small talk, or anything else"
)

_MATH_RE = re.compile(r"\d[\d\s.]*[+\-*/%][\d\s.()+\-*/%]*\d")


def _fake_classifier(messages: list[BaseMessage]) -> AIMessage:
    text = last_human_text(messages)
    if _MATH_RE.search(text):
        return AIMessage(content="math")
    if "?" in text or any(w in text.lower() for w in ("what", "who", "when", "where", "how")):
        return AIMessage(content="lookup")
    return AIMessage(content="chat")


def _fake_lookup(messages: list[BaseMessage]) -> AIMessage:
    q = last_human_text(messages)
    return AIMessage(content=f"Lookup result for {q!r}: (demo) see the LangGraph docs overview.")


def _fake_chat(messages: list[BaseMessage]) -> AIMessage:
    return AIMessage(content="Hello! Ask me a math question or a factual one.")


class RoutingState(BaseModel):
    """Shared typed state for the conditional routing pattern."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    route: str = ""


MERMAID = """graph TD
    __start__([start]) --> classifier{classifier}
    classifier -->|math| math[math handler]
    classifier -->|lookup| lookup[lookup handler]
    classifier -->|chat / fallback| chat[chat handler]
    math --> __end__([end])
    lookup --> __end__([end])
    chat --> __end__([end])"""


def build_routing(
    *,
    classifier_model: BaseChatModel | None = None,
    handler_models: dict[str, BaseChatModel] | None = None,
    checkpointer: Any = None,
):
    """Build a compiled conditional-routing graph (classify → dispatch).

    Args:
        classifier_model: model labeling the request as math/lookup/chat.
        handler_models: optional per-route handler model overrides.
        checkpointer: optional LangGraph checkpointer.
    """
    handler_models = handler_models or {}
    classifier = classifier_model or get_chat_model(_fake_classifier)

    def classifier_node(state: RoutingState) -> dict[str, Any]:
        reply = classifier.invoke(
            [SystemMessage(content=CLASSIFIER_SYSTEM), *state.messages]
        )
        text = message_text(reply).strip().lower()
        route = next((r for r in ROUTES if r in text), "chat")
        emit_agent_message("classifier", f"Classified as: {route}")
        return {"route": route}

    def math_node(state: RoutingState) -> dict[str, Any]:
        text = last_human_text(state.messages)
        match = re.search(r"[-+]?[\d(][\d\s.+\-*/()%]*[\d)]", text)
        expression = match.group(0).strip() if match else text
        result = calculator.invoke({"expression": expression})
        answer = f"{expression} = {result}"
        emit_agent_message("math", answer)
        return {"messages": [AIMessage(content=answer, name="math")]}

    def make_llm_handler(name: str, system_prompt: str, fake):
        model = handler_models.get(name) or get_chat_model(fake)

        def handler(state: RoutingState) -> dict[str, Any]:
            reply = model.invoke([SystemMessage(content=system_prompt), *state.messages])
            text = message_text(reply)
            emit_agent_message(name, text)
            return {"messages": [AIMessage(content=text, name=name)]}

        return handler

    def route(state: RoutingState) -> str:
        return state.route if state.route in ROUTES else "chat"

    graph = StateGraph(RoutingState)
    graph.add_node("classifier", classifier_node)
    graph.add_node("math", math_node)
    graph.add_node(
        "lookup",
        make_llm_handler(
            "lookup", "Answer the factual question concisely.", _fake_lookup
        ),
    )
    graph.add_node(
        "chat",
        make_llm_handler("chat", "You are a friendly conversational agent.", _fake_chat),
    )
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier", route, {r: r for r in ROUTES}
    )
    for r in ROUTES:
        graph.add_edge(r, END)
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=text)]}


INFO = PatternInfo(
    name="routing",
    title="Conditional Routing",
    category="control-flow",
    description=(
        "A classifier labels each request and a conditional edge dispatches it "
        "to the matching handler (math, lookup, chat) with a safe fallback."
    ),
    mermaid=MERMAID,
    default_text="What is 25 * 16?",
    build=build_routing,
    make_input=make_input,
)
