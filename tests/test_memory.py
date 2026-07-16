from langgraph_patterns.events import run_pattern
from langgraph_patterns.memory.checkpointing import (
    build_checkpointing,
    make_input as checkpoint_input,
    sqlite_checkpointer,
)
from langgraph_patterns.memory.episodic import (
    EpisodicMemory,
    build_episodic,
    make_input as episodic_input,
)


def test_episodic_memory_recall_and_memorize():
    memory = None  # default: seeded store
    graph = build_episodic(memory)
    result = run_pattern(graph, episodic_input("How do I make my agent resumable?"))

    memory_events = [e for e in result.events_of("agent_message") if e.agent == "memory"]
    assert "Recalled" in memory_events[0].data["content"]
    assert "Memorized" in memory_events[-1].data["content"]

    recalled = result.final_state["recalled"]
    assert recalled, "seeded checkpointer episode should be recalled"
    assert any("checkpointer" in r["response"] for r in recalled)
    assert all(r["score"] > 0 for r in recalled)
    assert "informed by" in result.final_state["answer"]


def test_episodic_memory_grows_across_runs():
    memory = EpisodicMemory()
    graph = build_episodic(memory)

    first = run_pattern(graph, episodic_input("What color is the sky on Mars?"))
    assert first.final_state["recalled"] == []  # empty store: nothing to recall
    assert len(memory.episodes) == 1

    second = run_pattern(graph, episodic_input("Tell me about the sky color on Mars"))
    assert len(second.final_state["recalled"]) == 1  # recalls the first run
    assert "Mars" in second.final_state["recalled"][0]["query"]


def test_checkpoint_resume_across_graph_instances(tmp_path):
    db = str(tmp_path / "checkpoints.sqlite")
    thread_id = "resume-test"
    config = {"configurable": {"thread_id": thread_id}}

    # Run the pipeline but stop after stage 2 (simulates a crash/stop mid-run).
    graph1 = build_checkpointing(checkpointer=sqlite_checkpointer(db))
    graph1.invoke(
        checkpoint_input("quarterly data"), config, interrupt_after=["transform"]
    )
    partial = graph1.get_state(config)
    assert partial.values["completed"] == ["ingest", "transform"]
    assert partial.next == ("report",)

    # A brand-new graph instance over the same SQLite file resumes and finishes.
    graph2 = build_checkpointing(checkpointer=sqlite_checkpointer(db))
    result = run_pattern(graph2, None, thread_id=thread_id)
    assert result.final_state["completed"] == ["ingest", "transform", "report"]
    assert "resumable at every stage" in result.final_state["report"]
    # Only the remaining stage ran on resume.
    assert {e.node for e in result.events_of("node_start")} == {"report"}
