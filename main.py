"""
Start the Research Agent web app.

    python main.py

Then open:
    http://127.0.0.1:8000        web page
    http://127.0.0.1:8000/docs   interactive API docs

What happens here:
  FastAPI = the web framework (routes, validation, docs)   -> research_agent/api/app.py
  uvicorn = the web SERVER that runs the FastAPI app (an "ASGI server")

Host/port/reload come from settings (.env: API_HOST, API_PORT, API_RELOAD).
Stop the server with Ctrl+C.
"""

import uvicorn

from research_agent.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"\n  Research Agent -> http://{settings.api_host}:{settings.api_port}\n")
    uvicorn.run(
        # Given as an import string (not the object) so reload can re-import it.
        "research_agent.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        # reload: restart automatically when you edit code under src/
        reload=settings.api_reload,
        reload_dirs=["src"] if settings.api_reload else None,
    )


if __name__ == "__main__":
    main()
