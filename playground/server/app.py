"""Playground backend.

Endpoints:

* ``GET  /api/patterns``                — pattern catalog (name, description,
  Mermaid diagram, default input, HITL support).
* ``POST /api/patterns/{name}/run``     — run a pattern; streams the pattern's
  structured event stream over SSE. HITL patterns pause on an ``interrupt``
  event that carries the ``thread_id`` needed to resume.
* ``POST /api/patterns/{name}/resume``  — resume an interrupted HITL run with
  an approve/reject decision; streams the rest of the events over SSE.

Graphs are built once per pattern at startup (with in-memory checkpointers
where required), so episodic memory accumulates across playground runs and
interrupted HITL threads stay resumable.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph_patterns.events import Event, stream_events
from langgraph_patterns.registry import PatternInfo, get_registry

app = FastAPI(title="LangGraph Patterns Playground", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REGISTRY: dict[str, PatternInfo] = get_registry()
GRAPHS: dict[str, Any] = {}


def get_graph(name: str) -> Any:
    if name not in GRAPHS:
        info = REGISTRY[name]
        GRAPHS[name] = info.build()
    return GRAPHS[name]


class RunRequest(BaseModel):
    text: str = ""


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""


def pattern_payload(info: PatternInfo) -> dict[str, Any]:
    return {
        "name": info.name,
        "title": info.title,
        "category": info.category,
        "description": info.description,
        "mermaid": info.mermaid,
        "default_text": info.default_text,
        "supports_hitl": info.supports_hitl,
    }


@app.get("/api/patterns")
def list_patterns() -> list[dict[str, Any]]:
    return [pattern_payload(info) for info in REGISTRY.values()]


@app.get("/api/patterns/{name}")
def get_pattern(name: str) -> dict[str, Any]:
    info = REGISTRY.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown pattern {name!r}")
    return pattern_payload(info)


def sse(events: Iterator[Event]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event.model_dump(), default=str)}\n\n"


def sse_response(events: Iterator[Event]) -> StreamingResponse:
    return StreamingResponse(
        sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/patterns/{name}/run")
def run_pattern_endpoint(name: str, request: RunRequest) -> StreamingResponse:
    info = REGISTRY.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown pattern {name!r}")
    graph = get_graph(name)
    text = request.text.strip() or info.default_text
    thread_id = uuid.uuid4().hex if getattr(graph, "checkpointer", None) else None
    events = stream_events(graph, info.make_input(text), thread_id=thread_id)
    return sse_response(events)


@app.post("/api/patterns/{name}/resume")
def resume_pattern_endpoint(name: str, request: ResumeRequest) -> StreamingResponse:
    from langgraph.types import Command

    info = REGISTRY.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown pattern {name!r}")
    if not info.supports_hitl:
        raise HTTPException(status_code=400, detail=f"pattern {name!r} does not support resume")
    graph = get_graph(name)
    command = Command(resume={"approved": request.approved, "feedback": request.feedback})
    events = stream_events(graph, command, thread_id=request.thread_id)
    return sse_response(events)


class SPAStaticFiles(StaticFiles):
    """Static files with SPA fallback: unknown paths serve index.html so
    client-side routes like /p/react work on direct navigation/refresh."""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        from starlette.exceptions import HTTPException as StarletteHTTPException

        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


# Serve the built frontend when available (playground/web/dist).
_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if _DIST.is_dir():
    app.mount("/", SPAStaticFiles(directory=str(_DIST), html=True), name="web")
