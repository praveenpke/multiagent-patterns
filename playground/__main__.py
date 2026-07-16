"""Run the playground: ``uv run python -m playground`` (from the repo root).

Serves the API and, when built (``npm run build`` in playground/web), the
React frontend at http://127.0.0.1:8000.
"""

import os

import uvicorn


def main() -> None:
    host = os.environ.get("PLAYGROUND_HOST", "127.0.0.1")
    port = int(os.environ.get("PLAYGROUND_PORT", "8000"))
    print(f"LangGraph patterns playground -> http://{host}:{port}")
    uvicorn.run("playground.server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
