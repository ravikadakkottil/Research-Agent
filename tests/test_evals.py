"""
Tests for the eval machinery itself.

A buggy scorer silently produces wrong numbers, and every decision made from
them would be wrong. So the scoring code gets unit tests like any other code.
"""

import asyncio
from pathlib import Path

import pytest

from research_agent.cache import ResponseCache
from research_agent.evals import judge as judge_module
from research_agent.evals.golden import GoldenPaper, GoldenTopic
from research_agent.evals.judge import RelevanceJudge, Verdict
from research_agent.evals.metrics import RunMetrics, evaluate_run
from research_agent.evals.runner import aggregate
from research_agent.models import (
    Paper,
    PaperUsed,
    QueryPlan,
    ReadingList,
    RecentWindow,
    ResearchSummary,
)
from research_agent.retrieval.selection import paper_identity

RAG = GoldenPaper(
    title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    kind="foundational",
    openalex_ids=["W3027879771"],
    dois=["10.48550/arxiv.2005.11401"],
    titles=["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
)
SURVEY = GoldenPaper(
    title="RAG survey",
    kind="recent",
    openalex_ids=["W4389984066"],
    titles=["Retrieval-Augmented Generation for Large Language Models: A Survey"],
)
TOPIC = GoldenTopic(id="rag", question="rag question", must_include=[RAG, SURVEY])


# --- golden matching -------------------------------------------------------------
@pytest.mark.parametrize(
    "paper",
    [
        Paper(openalex_id="https://openalex.org/W3027879771"),  # by OpenAlex ID
        Paper(doi="https://doi.org/10.48550/ARXIV.2005.11401"),  # by DOI (any case)
        Paper(title="retrieval-augmented generation for knowledge intensive NLP tasks!"),  # title
    ],
)
def test_golden_matches_any_identifier(paper):
    assert RAG.matches(paper)


def test_golden_does_not_match_other_paper():
    assert not RAG.matches(Paper(openalex_id="W1", doi="10.1/x", title="Something else"))


# --- metrics ---------------------------------------------------------------------
def make_reading(papers, candidates=None) -> ReadingList:
    return ReadingList(
        query_plan=QueryPlan(search_query="q"),
        recent_window=RecentWindow(from_year=2024, source="default", explanation=""),
        bucket_counts={},
        candidates=candidates if candidates is not None else papers,
        papers=papers,
    )


def test_recall_precision_and_selection_drop():
    found_rag = Paper(openalex_id="W3027879771", title="RAG", retrieval_reasons=["highly_cited"])
    off_topic = Paper(openalex_id="W9", title="Parkinson's disease", retrieval_reasons=["recent"])
    survey_in_candidates_only = Paper(openalex_id="W4389984066", title="Survey")
    reading = make_reading(
        [found_rag, off_topic], candidates=[found_rag, off_topic, survey_in_candidates_only]
    )
    verdicts = {
        paper_identity(found_rag): Verdict(relevant=True),
        paper_identity(off_topic): Verdict(relevant=False),
    }

    m = evaluate_run(TOPIC, 1, reading, verdicts, current_year=2026)

    assert m.recall == 0.5  # RAG in final list, survey not
    assert m.candidate_recall == 1.0  # both were retrieved
    assert m.dropped_in_selection == ["RAG survey"]  # -> selection lost it
    assert m.recall_foundational == 1.0
    assert m.recall_recent == 0.0
    assert m.precision == 0.5
    assert m.recent_precision == 0.0  # the "recent" pick was off-topic


def test_rerank_and_selection_drops_are_told_apart():
    rag = Paper(openalex_id="W3027879771", title="RAG")
    survey = Paper(openalex_id="W4389984066", title="Survey")
    other = Paper(openalex_id="W9", title="Other")
    reading = make_reading([other], candidates=[rag, survey, other])
    reading.shortlist = [survey, other]  # rerank dropped RAG; selection then dropped the survey

    m = evaluate_run(TOPIC, 1, reading, {}, current_year=2026)

    assert m.candidate_recall == 1.0
    assert m.shortlist_recall == 0.5
    assert m.recall == 0.0
    assert m.dropped_in_rerank == [RAG.title]
    assert m.dropped_in_selection == ["RAG survey"]


def test_future_dated_papers_are_counted():
    reading = make_reading([Paper(openalex_id="W1", year=2027), Paper(openalex_id="W2", year=2025)])
    assert evaluate_run(TOPIC, 1, reading, {}, current_year=2026).future_dated == 1


def test_unjudged_papers_are_excluded_from_precision():
    a, b = Paper(openalex_id="W1"), Paper(openalex_id="W2")
    m = evaluate_run(
        TOPIC,
        1,
        make_reading([a, b]),
        {paper_identity(a): Verdict(relevant=True)},
        current_year=2026,
    )
    assert m.precision == 1.0  # b had no verdict -> not counted either way


def test_grounding_flags_invented_citations():
    papers = [Paper(openalex_id="W1", title="Real Paper About RAG")]
    summary = ResearchSummary(
        answer="a",
        papers_used=[PaperUsed(title="Real paper about RAG"), PaperUsed(title="Invented Paper")],
    )
    m = evaluate_run(TOPIC, 1, make_reading(papers), {}, current_year=2026, summary=summary)
    assert m.grounding == 0.5


# --- aggregation -------------------------------------------------------------------
def test_aggregate_macro_average_and_spread():
    runs = [
        RunMetrics(topic_id="a", run=1, recall=1.0),
        RunMetrics(topic_id="a", run=2, recall=0.5),
        RunMetrics(topic_id="b", run=1, recall=0.0),
        RunMetrics(topic_id="b", run=2, recall=0.5),
        RunMetrics(topic_id="b", run=3, error="boom"),  # failed runs are ignored
    ]
    overall, per_topic = aggregate(runs)
    assert per_topic["a"]["recall"] == 0.75
    assert per_topic["b"]["recall"] == 0.25
    assert overall["recall"]["mean"] == 0.5
    # run 1 overall = 0.5, run 2 overall = 0.5 -> no spread
    assert overall["recall"]["spread"] == 0.0


# --- judge -------------------------------------------------------------------------
def test_judge_caches_per_paper_and_ignores_bad_indices(tmp_path: Path, monkeypatch):
    calls = []

    async def fake_ask_json(system, user, schema, *, cache=None, model=None):
        calls.append(user)
        return schema.model_validate(
            {
                "judgements": [
                    {"index": 1, "relevant": True, "reason": "on topic"},
                    {"index": 99, "relevant": True},  # invented index -> ignored
                ]
            }
        )

    monkeypatch.setattr(judge_module, "ask_json", fake_ask_json)
    cache = ResponseCache(tmp_path / "judge.sqlite", ttl_seconds=60)
    judge = RelevanceJudge("judge-model", cache)
    paper = Paper(openalex_id="W1", title="A")

    first = asyncio.run(judge.judge("q", [paper]))
    second = asyncio.run(judge.judge("q", [paper]))  # served from cache

    assert first == second == {paper_identity(paper): Verdict(relevant=True, reason="on topic")}
    assert len(calls) == 1
