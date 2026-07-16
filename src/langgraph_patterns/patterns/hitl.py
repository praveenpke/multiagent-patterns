"""HITL handoff pattern — pause for human approval, then resume.

An agent drafts a proposal for a consequential action; the graph then pauses
on a LangGraph ``interrupt`` and waits for a human decision. Resuming with
``Command(resume={"approved": True|False, "feedback": "..."})`` routes to the
execute or abort branch.

Requires a checkpointer (an ``InMemorySaver`` is used by default) because
interrupt/resume relies on checkpoints. The playground uses this pattern for
its approve/reject flow.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from langgraph_patterns.events import emit_agent_message
from langgraph_patterns.models import get_chat_model
from langgraph_patterns.patterns.supervisor import message_text
from langgraph_patterns.registry import PatternInfo

PROPOSER_SYSTEM = (
    "You are an operations agent. Draft a short, concrete action proposal for "
    "the task (what you will do, to what, with what effect). Reply with the "
    "proposal only — it will be reviewed by a human before execution."
)


def _fake_proposer(messages: list[BaseMessage]) -> AIMessage:
    task = str(messages[-1].content).removeprefix("Task: ")
    return AIMessage(
        content=(
            f"Proposal: to accomplish {task[:80]!r} I will draft the change, "
            "apply it to the target system, and post a summary. No rollback risk."
        )
    )


class HITLState(BaseModel):
    """Shared typed state for the human-in-the-loop handoff pattern."""

    task: str = ""
    proposal: str = ""
    approved: bool | None = None
    feedback: str = ""
    result: str = ""


MERMAID = """graph TD
    __start__([start]) --> propose[agent: propose action]
    propose --> gate{human gate<br/>interrupt}
    gate -->|approved| execute[execute action]
    gate -->|rejected| abort[abort]
    execute --> __end__([end])
    abort --> __end__([end])"""


def build_hitl(
    *,
    proposer_model: BaseChatModel | None = None,
    checkpointer: Any = None,
):
    """Build a compiled human-in-the-loop approval graph.

    The run pauses at the ``human_gate`` node with an interrupt payload
    ``{"proposal", "question"}``. Resume with
    ``Command(resume={"approved": bool, "feedback": str})`` on the same
    ``thread_id``.

    Args:
        proposer_model: model drafting the action proposal.
        checkpointer: LangGraph checkpointer; defaults to ``InMemorySaver``
            (interrupts require one).
    """
    proposer = proposer_model or get_chat_model(_fake_proposer)
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    def propose_node(state: HITLState) -> dict[str, Any]:
        reply = proposer.invoke(
            [SystemMessage(content=PROPOSER_SYSTEM), HumanMessage(content=f"Task: {state.task}")]
        )
        proposal = message_text(reply)
        emit_agent_message("proposer", proposal)
        return {"proposal": proposal}

    def human_gate_node(state: HITLState) -> dict[str, Any]:
        decision = interrupt(
            {
                "proposal": state.proposal,
                "question": "Approve this action?",
            }
        )
        if isinstance(decision, dict):
            approved = bool(decision.get("approved"))
            feedback = str(decision.get("feedback", ""))
        else:
            approved = str(decision).strip().lower() in {"y", "yes", "true", "approve", "approved"}
            feedback = ""
        emit_agent_message(
            "human", ("Approved" if approved else "Rejected") + (f": {feedback}" if feedback else "")
        )
        return {"approved": approved, "feedback": feedback}

    def execute_node(state: HITLState) -> dict[str, Any]:
        result = f"Executed approved action: {state.proposal}"
        emit_agent_message("executor", result)
        return {"result": result}

    def abort_node(state: HITLState) -> dict[str, Any]:
        result = "Aborted by human reviewer" + (
            f" — feedback: {state.feedback}" if state.feedback else "."
        )
        emit_agent_message("executor", result)
        return {"result": result}

    def route(state: HITLState) -> str:
        return "execute" if state.approved else "abort"

    graph = StateGraph(HITLState)
    graph.add_node("propose", propose_node)
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("abort", abort_node)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "human_gate")
    graph.add_conditional_edges("human_gate", route, {"execute": "execute", "abort": "abort"})
    graph.add_edge("execute", END)
    graph.add_edge("abort", END)
    return graph.compile(checkpointer=checkpointer)


def make_input(text: str) -> dict[str, Any]:
    return {"task": text}


INFO = PatternInfo(
    name="hitl",
    title="HITL Handoff",
    category="control-flow",
    description=(
        "The agent proposes an action, the graph pauses on a LangGraph interrupt "
        "for human approval, and resumes into the execute or abort branch."
    ),
    mermaid=MERMAID,
    default_text="Archive all inactive user accounts older than one year.",
    supports_hitl=True,
    needs_checkpointer=True,
    build=build_hitl,
    make_input=make_input,
)
