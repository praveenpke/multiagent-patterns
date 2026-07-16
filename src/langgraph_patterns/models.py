"""Model layer.

:func:`get_chat_model` returns a real Anthropic Claude chat model when
``ANTHROPIC_API_KEY`` is present, and a deterministic :class:`ScriptedChatModel`
otherwise. Every pattern supplies its own *responder* — a pure function from
the message history to the next ``AIMessage`` — so each pattern runs fully
offline with sensible, reproducible output.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

#: A responder maps the full message history to the next assistant message.
Responder = Callable[[list[BaseMessage]], AIMessage]

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


class ScriptedChatModel(BaseChatModel):
    """Deterministic fake chat model driven by a responder function.

    The responder receives the full message list (system prompt included) and
    returns the next :class:`AIMessage`. Tool binding is recorded on the copy
    returned by :meth:`bind_tools` so responders can emit tool calls that the
    graph's tool executor will run.
    """

    responder: Any  # Responder; kept as Any for pydantic-model compatibility
    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responder(list(messages))
        if not isinstance(message, AIMessage):
            message = AIMessage(content=str(message))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self.model_copy(update={"bound_tools": list(tools)})


def anthropic_enabled() -> bool:
    """True when a real Anthropic model should be used."""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and not os.environ.get(
        "LGP_FORCE_FAKE"
    )


def get_chat_model(
    responder: Responder,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Return a chat model for a pattern.

    Args:
        responder: deterministic fallback used when no API key is configured.
        model: optional Anthropic model id override (default: ``LGP_MODEL``
            env var, then ``claude-opus-4-8``).
        max_tokens: max output tokens for the real model.
    """
    if anthropic_enabled():
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or os.environ.get("LGP_MODEL", DEFAULT_ANTHROPIC_MODEL),
            max_tokens=max_tokens,
        )
    return ScriptedChatModel(responder=responder)


def last_human_text(messages: Sequence[BaseMessage]) -> str:
    """Utility for responders: content of the most recent human message."""
    for message in reversed(messages):
        if message.type == "human":
            return str(message.content)
    return ""
