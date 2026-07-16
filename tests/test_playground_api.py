import json

import pytest
from fastapi.testclient import TestClient

from playground.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_list_patterns(client):
    response = client.get("/api/patterns")
    assert response.status_code == 200
    patterns = {p["name"]: p for p in response.json()}
    expected = {
        "supervisor",
        "react",
        "reflection",
        "hierarchical",
        "plan_execute",
        "hitl",
        "routing",
        "fanout",
        "episodic_memory",
        "checkpointing",
    }
    assert expected <= set(patterns)
    for p in patterns.values():
        assert p["mermaid"].startswith("graph")
        assert p["description"] and p["default_text"]
    assert patterns["hitl"]["supports_hitl"] is True


def test_get_pattern_and_404(client):
    assert client.get("/api/patterns/react").json()["title"] == "ReAct"
    assert client.get("/api/patterns/nope").status_code == 404


def test_run_pattern_streams_events(client):
    response = client.post("/api/patterns/react/run", json={"text": "What is 3 * 9?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert "tool_call" in types and "tool_result" in types
    run_end = events[-1]
    assert run_end["data"]["interrupted"] is False
    assert "27" in run_end["data"]["final_state"]["messages"][-1]["content"]


def test_hitl_run_pauses_then_resume_completes(client):
    run = client.post("/api/patterns/hitl/run", json={"text": "Rotate the API keys."})
    events = parse_sse(run.text)
    interrupt = next(e for e in events if e["type"] == "interrupt")
    thread_id = interrupt["data"]["thread_id"]
    assert thread_id
    assert "proposal" in interrupt["data"]["payload"]
    assert events[-1]["data"]["interrupted"] is True

    resume = client.post(
        "/api/patterns/hitl/resume",
        json={"thread_id": thread_id, "approved": True, "feedback": "ok"},
    )
    resumed = parse_sse(resume.text)
    assert resumed[-1]["type"] == "run_end"
    assert resumed[-1]["data"]["interrupted"] is False
    assert resumed[-1]["data"]["final_state"]["result"].startswith("Executed approved action")


def test_resume_rejected_on_non_hitl_pattern(client):
    response = client.post(
        "/api/patterns/react/resume",
        json={"thread_id": "x", "approved": True},
    )
    assert response.status_code == 400
