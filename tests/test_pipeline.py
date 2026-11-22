"""
End-to-end pipeline tests with fakes for every outside service.

Written BEFORE the LangGraph port and kept unchanged through it: the same
inputs must give the same reading list, summary and progress steps, whichever
way the steps are orchestrated.

Fakes: the planning and synthesis LLM calls, the Gemini embedder and the
OpenAlex client. The real retrieval code (buckets, dedupe, rerank, citation
expansion, selection) runs unchanged.
"""

import asyncio
import re

import pytest

from research_agent import pipeline
from research_agent.config import Settings
from research_agent.models import Paper, QueryPlan, ResearchSummary

PLAN = QueryPlan(
    search_query="diffusion models image generation",
    core_query="diffusion models",
    reason="test plan",
)


def paper(pid: str, title: str, *, year=2022, cites=100, refs=()) -> Paper:
    return Paper(
        openalex_id=f"https://openalex.org/{pid}",
        title=title,
        year=year,
        cited_by_count=cites,
        referenced_works=list(refs),
    )


# Two relevant papers both cite W_LAND -> citation expansion fetches it.
# OFF matches the keywords but not the meaning -> the similarity filter drops it.
R1 = paper("W1", "Diffusion models for image synthesis", refs=["W_LAND"])
R2 = paper("W2", "Guided diffusion for image generation", refs=["W_LAND"])
OFF = paper("W3", "Diffusion of innovations in organisations", cites=50_000)
H1 = paper("W4", "Denoising diffusion probabilistic models", year=2020, cites=9_000)
LAND = paper("W_LAND", "Score-based generative modelling with diffusion", year=2019)


class FakeOpenAlex:
    network_requests = 0
    cache = None

    def __init__(self) -> None:
        self.id_requests: list[list[str]] = []

    async def search_works(self, query, *, sort, **kwargs):
        if sort.startswith("relevance"):
            return [R1, R2, OFF]
        return [H1]  # recent + highly cited buckets

    async def get_works_by_ids(self, ids):
        self.id_requests.append(list(ids))
        return [LAND] if "W_LAND" in ids else []


class FakeEmbedder:
    """Query = [1, 0]; on-topic papers point the same way, 'innovations' doesn't."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def embed_query(self, text):
        return [1.0, 0.0]

    async def embed_documents(self, texts):
        return [[0.05, 0.95] if "innovations" in t else [0.95, 0.05] for t in texts]


def settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "x",
        "target_final_papers": 3,
        "select_relevant": 1,
        "select_recent": 1,
        "select_highly_cited": 1,
        "select_cited_by_top": 1,
        "expansion_min_citers": 2,
        "europepmc_enabled": False,
    } | overrides
    return Settings(_env_file=None, **values)


@pytest.fixture
def fakes(monkeypatch):
    calls: dict = {}

    async def fake_plan(question, *, current_year=None, cache=None):
        calls["plan"] = question
        return PLAN

    async def fake_summary(question, papers, *, recent_window, abstract_chars, cache=None):
        calls["summary_papers"] = [p.title for p in papers]
        return ResearchSummary(answer="fake answer")

    embedder = {"value": FakeEmbedder()}
    monkeypatch.setattr(pipeline, "generate_query_plan", fake_plan)
    monkeypatch.setattr(pipeline, "summarize_papers", fake_summary)
    monkeypatch.setattr(pipeline, "create_embedder", lambda s: embedder["value"])
    monkeypatch.setattr(pipeline, "create_llm_cache", lambda s: None)
    return {"calls": calls, "embedder": embedder}


def step_numbers(steps: list[str]) -> list[str]:
    """'1)', '2)', ... in the order the progress messages arrived."""
    return [m.group(1) for s in steps if (m := re.match(r"\s*(\d)\)", s))]


def titles(papers) -> set[str]:
    return {p.title for p in papers}


def test_full_search_runs_every_step_in_order(fakes):
    steps: list[str] = []
    client = FakeOpenAlex()

    result = asyncio.run(
        pipeline.run_paper_search(
            "diffusion models for images", on_step=steps.append, settings=settings(), client=client
        )
    )

    assert step_numbers(steps) == ["1", "2", "3", "4", "5", "6", "7"]
    assert result.user_question == "diffusion models for images"
    assert result.query_plan == PLAN
    assert result.summary.answer == "fake answer"
    # Expansion fetched the co-cited landmark; selection used its quota for it.
    assert client.id_requests == [["W_LAND"]]
    assert result.bucket_counts["cited_by_top"] == 1
    assert result.unique_candidates == 5  # R1, R2, OFF, H1 + LAND
    assert len(result.papers) == 3
    assert LAND.title in titles(result.papers)
    assert OFF.title not in titles(result.papers)
    # Synthesis saw exactly the selected papers.
    assert fakes["calls"]["summary_papers"] == [p.title for p in result.papers]


def test_retrieval_only_keeps_candidates_and_shortlist(fakes):
    reading = asyncio.run(
        pipeline.retrieve_reading_list("diffusion", settings=settings(), client=FakeOpenAlex())
    )

    assert titles(reading.candidates) == titles([R1, R2, OFF, H1, LAND])
    assert all(p.similarity is not None for p in reading.candidates)
    assert reading.shortlist is not None
    assert OFF.title not in titles(reading.shortlist)
    assert reading.bucket_counts["relevant"] == 3
    assert reading.europepmc_status is not None
    assert "summary_papers" not in fakes["calls"]  # retrieval only: no synthesis


def test_without_embedder_rerank_is_skipped(fakes):
    fakes["embedder"]["value"] = None
    steps: list[str] = []

    reading = asyncio.run(
        pipeline.retrieve_reading_list(
            "diffusion", on_step=steps.append, settings=settings(), client=FakeOpenAlex()
        )
    )

    assert any("Semantic scoring skipped" in s for s in steps)
    assert reading.shortlist is None
    assert all(p.similarity is None for p in reading.candidates)
    assert len(reading.papers) == 3


def test_expansion_can_be_turned_off(fakes):
    client = FakeOpenAlex()
    steps: list[str] = []

    reading = asyncio.run(
        pipeline.retrieve_reading_list(
            "diffusion",
            on_step=steps.append,
            settings=settings(citation_expansion_enabled=False),
            client=client,
        )
    )

    assert step_numbers(steps) == ["1", "2", "3", "4", "6"]
    assert client.id_requests == []
    assert "cited_by_top" not in reading.bucket_counts
    assert LAND.title not in titles(reading.candidates)
