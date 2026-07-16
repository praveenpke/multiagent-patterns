"""Episodic memory pattern — similarity recall over past interactions.

:class:`EpisodicMemory` is a tiny local store: episodes are embedded with
plain TF-IDF (no external services, no model downloads) and recalled by
cosine similarity. The graph recalls relevant past episodes, answers with
them as context, then memorizes the new interaction — so the store grows as
the pattern is used.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import get_chat_model
from langgraph_patterns.patterns.supervisor import message_text
from langgraph_patterns.registry import PatternInfo

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class Episode(BaseModel):
    query: str
    response: str


class EpisodicMemory:
    """In-process episodic store with TF-IDF cosine-similarity recall."""

    def __init__(self, episodes: list[Episode] | None = None) -> None:
        self.episodes: list[Episode] = list(episodes or [])

    def add(self, query: str, response: str) -> None:
        self.episodes.append(Episode(query=query, response=response))

    def _idf(self) -> dict[str, float]:
        n = len(self.episodes)
        df: Counter[str] = Counter()
        for episode in self.episodes:
            df.update(set(_tokenize(f"{episode.query} {episode.response}")))
        return {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}

    @staticmethod
    def _vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def recall(self, query: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Episode, float]]:
        """Return up to *k* most similar past episodes with their scores."""
        if not self.episodes:
            return []
        idf = self._idf()
        query_vec = self._vector(_tokenize(query), idf)
        scored = [
            (ep, self._cosine(query_vec, self._vector(_tokenize(f"{ep.query} {ep.response}"), idf)))
            for ep in self.episodes
        ]
        scored = [(ep, s) for ep, s in scored if s >= min_score]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


SEED_EPISODES = [
    Episode(
        query="How do I add persistence to a LangGraph agent?",
        response="Compile the graph with a checkpointer (e.g. SqliteSaver) and pass a thread_id.",
    ),
    Episode(
        query="What is a supervisor pattern?",
        response="A router agent that delegates each turn to a specialist and decides when to finish.",
    ),
    Episode(
        query="How do I pause a graph for human approval?",
        response="Call interrupt() inside a node and resume with Command(resume=...) on the same thread.",
    ),
]

ANSWER_SYSTEM = (
    "Answer the user's question. If past episodes are provided, use them as "
    "context and mention when a past interaction informed your answer."
)


def _fake_answerer(messages: list[BaseMessage]) -> AIMessage:
    prompt = str(messages[-1].content)
    recalled = prompt.count("- past:")
    q = prompt.splitlines()[0].removeprefix("Question: ")
    if recalled:
        return AIMessage(
            content=f"Answer to {q!r}, informed by {recalled} recalled past episode(s)."
        )
    return AIMessage(content=f"Answer to {q!r} (no similar past episodes found).")


class EpisodicState(BaseModel):
    """Shared typed state for the episodic memory pattern."""

    query: str = ""
    recalled: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""


MERMAID = """graph TD
    __start__([start]) --> recall[recall similar episodes]
    recall --> answer[answer with recalled context]
    answer --> memorize[memorize this interaction]
    memorize --> __end__([end])
    store[(TF-IDF episode store)] -.recall.-> recall
    memorize -.write.-> store"""


def build_episodic(
    memory: EpisodicMemory | None = None,
    *,
    k: int = 3,
    answer_model: BaseChatModel | None = None,
    checkpointer: Any = None,
):
    """Build a compiled episodic-memory graph (recall → answer → memorize).

    Args:
        memory: episodic store shared across runs; defaults to a fresh store
            seeded with a few demo episodes.
        k: max episodes to recall per query.
        answer_model: answering model.
        checkpointer: optional LangGraph checkpointer.
    """
    memory = memory if memory is not None else EpisodicMemory(list(SEED_EPISODES))
    answerer = answer_model or get_chat_model(_fake_answerer)

    def recall_node(state: EpisodicState) -> dict[str, Any]:
        hits = memory.recall(state.query, k=k)
        recalled = [
            {"query": ep.query, "response": ep.response, "score": round(score, 3)}
            for ep, score in hits
        ]
        emit_agent_message(
            "memory",
            f"Recalled {len(recalled)} similar past episode(s)"
            + (f" (top score {recalled[0]['score']})" if recalled else "."),
        )
        return {"recalled": recalled}

    def answer_node(state: EpisodicState) -> dict[str, Any]:
        context = "\n".join(
            f"- past: Q: {r['query']} A: {r['response']} (similarity {r['score']})"
            for r in state.recalled
        )
        reply = answerer.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(
                    content=f"Question: {state.query}\nRecalled episodes:\n{context or '(none)'}"
                ),
            ]
        )
        answer = message_text(reply)
        emit_agent_message("answerer", answer)
        return {"answer": answer}

    def memorize_node(state: EpisodicState) -> dict[str, Any]:
        memory.add(state.query, state.answer)
        emit_agent_message(
            "memory", f"Memorized this interaction (store now holds {len(memory.episodes)} episodes)."
        )
        return {}

    graph = StateGraph(EpisodicState)
    graph.add_node("recall", recall_node)
    graph.add_node("answer", answer_node)
    graph.add_node("memorize", memorize_node)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "answer")
    graph.add_edge("answer", "memorize")
    graph.add_edge("memorize", END)
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    return {"query": text}


INFO = PatternInfo(
    name="episodic_memory",
    title="Episodic Memory",
    category="memory",
    description=(
        "Recalls similar past interactions with local TF-IDF cosine similarity, "
        "answers with them as context, then memorizes the new interaction."
    ),
    mermaid=MERMAID,
    default_text="How do I make my LangGraph agent resumable?",
    build=build_episodic,
    make_input=make_input,
)
