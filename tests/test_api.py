"""
Web layer tests with FastAPI's TestClient (no real server, no network).

The pipeline and agent are replaced with fakes (monkeypatch), so these tests
check only HTTP concerns: routing, validation, status codes and SSE streaming.
"""

import json

import pytest
from fastapi.testclient import TestClient

from research_agent.api import app as app_module
from research_agent.models import (
    Paper,
    QueryPlan,
    RecentWindow,
    ResearchSummary,
    SearchResult,
)
from research_agent.sources.openalex import OpenAlexRateLimitError

FAKE_RESULT = SearchResult(
    user_question="q",
    query_plan=QueryPlan(search_query="gnn drug discovery"),
    recent_window=RecentWindow(from_year=2024, source="default", explanation="default"),
    bucket_counts={"relevant": 1, "recent": 0, "highly_cited": 0},
    unique_candidates=1,
    papers=[Paper(title="A Paper", retrieval_reasons=["relevant"])],
    summary=ResearchSummary(answer="An answer."),
)


async def fake_search(question, *, on_step=lambda _: None, settings=None, client=None):
    on_step("1) planning")
    on_step("2) searching")
    return FAKE_RESULT


async def fake_agent(question, *, on_step=lambda _: None, settings=None):
    on_step("--- Agent step 1 ---")
    return f"Agent answer to: {question}"


async def failing_search(question, **kwargs):
    raise OpenAlexRateLimitError("budget used up")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "run_paper_search", fake_search)
    monkeypatch.setattr(app_module, "run_tool_agent", fake_agent)
    with TestClient(app_module.app) as test_client:  # `with` runs startup/shutdown (lifespan)
        yield test_client


def parse_sse(text: str) -> list[tuple[str, object]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_index_serves_web_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Research Agent" in response.text


def test_search_returns_result(client):
    response = client.post("/api/search", json={"question": "graph neural networks"})
    assert response.status_code == 200
    assert response.json()["summary"]["answer"] == "An answer."


def test_search_validates_input(client):
    assert client.post("/api/search", json={"question": "x"}).status_code == 422  # too short


def test_agent_returns_answer(client):
    response = client.post("/api/agent", json={"question": "find attention papers"})
    assert response.json()["answer"] == "Agent answer to: find attention papers"


def test_rate_limit_maps_to_429(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_paper_search", failing_search)
    response = client.post("/api/search", json={"question": "graph neural networks"})
    assert response.status_code == 429
    assert "budget" in response.json()["detail"]


def test_search_stream_sends_steps_then_result(client):
    response = client.get("/api/search/stream", params={"question": "graph neural networks"})
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names == ["step", "step", "result"]
    assert events[-1][1]["summary"]["answer"] == "An answer."


def test_stream_reports_failure_event(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_paper_search", failing_search)
    response = client.get("/api/search/stream", params={"question": "graph neural networks"})
    ((name, data),) = parse_sse(response.text)
    assert name == "failure"
    assert data["status"] == 429
