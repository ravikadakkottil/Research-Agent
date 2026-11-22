"""Dedupe + balanced selection (pure logic, no network)."""

from research_agent.models import Paper
from research_agent.retrieval.selection import (
    deduplicate_papers,
    normalize_title,
    paper_identity,
    select_balanced_papers,
)

QUOTAS = {"relevant": 4, "recent": 3, "highly_cited": 3}


def test_normalize_title_ignores_case_and_punctuation():
    assert normalize_title("Attention Is All You Need!") == normalize_title(
        "attention is all you need"
    )


def test_identity_prefers_doi_then_openalex_then_title():
    assert paper_identity(Paper(doi="https://doi.org/10.1/ABC", openalex_id="W1")) == (
        "doi:https://doi.org/10.1/abc"
    )
    assert paper_identity(Paper(openalex_id="W1", title="T")) == "openalex:w1"
    assert paper_identity(Paper(title="Some Title")) == "title:sometitle"


def test_deduplicate_merges_reasons_without_mutating_input():
    paper = Paper(title="Same", openalex_id="W1")
    buckets = {"relevant": [paper], "recent": [paper.model_copy()], "highly_cited": []}

    merged = deduplicate_papers(buckets)

    assert len(merged) == 1
    assert merged[0].retrieval_reasons == ["relevant", "recent"]
    assert paper.retrieval_reasons == []  # original untouched


def test_select_respects_target_count():
    candidates = [
        Paper(title=f"P{i}", openalex_id=f"W{i}", retrieval_reasons=["relevant"]) for i in range(20)
    ]
    assert len(select_balanced_papers(candidates, target_count=5, quotas=QUOTAS)) == 5


def test_select_balances_buckets():
    relevant = [
        Paper(openalex_id=f"R{i}", cited_by_count=100, retrieval_reasons=["relevant"])
        for i in range(10)
    ]
    recent = [
        Paper(openalex_id=f"N{i}", year=2026, retrieval_reasons=["recent"]) for i in range(10)
    ]
    cited = [
        Paper(openalex_id=f"C{i}", cited_by_count=9999, retrieval_reasons=["highly_cited"])
        for i in range(10)
    ]

    chosen = select_balanced_papers(relevant + recent + cited, target_count=10, quotas=QUOTAS)
    ids = [p.openalex_id for p in chosen]

    # Each perspective is represented despite "highly_cited" having far more citations.
    assert sum(i.startswith("R") for i in ids) >= 4
    assert sum(i.startswith("N") for i in ids) >= 3
    assert sum(i.startswith("C") for i in ids) >= 3


def test_recent_quota_prefers_on_topic_over_newest():
    newest_generic = Paper(
        openalex_id="G",
        year=2026,
        cited_by_count=5000,
        similarity=0.67,
        retrieval_reasons=["recent"],
    )
    older_on_topic = Paper(
        openalex_id="T",
        year=2025,
        cited_by_count=300,
        similarity=0.78,
        retrieval_reasons=["recent"],
    )
    chosen = select_balanced_papers(
        [newest_generic, older_on_topic], target_count=1, quotas={"recent": 1}
    )
    assert chosen[0].openalex_id == "T"


def test_recent_quota_without_rerank_falls_back_to_citations():
    # similarity None (rerank skipped) -> ties at 0, so citations decide
    popular = Paper(openalex_id="P", year=2024, cited_by_count=900, retrieval_reasons=["recent"])
    newer = Paper(openalex_id="N", year=2026, cited_by_count=10, retrieval_reasons=["recent"])
    chosen = select_balanced_papers([newer, popular], target_count=1, quotas={"recent": 1})
    assert chosen[0].openalex_id == "P"


def test_select_prefers_multi_bucket_papers_for_leftover_slots():
    multi = Paper(openalex_id="M", retrieval_reasons=["relevant", "recent"])
    singles = [Paper(openalex_id=f"S{i}", retrieval_reasons=["relevant"]) for i in range(5)]
    # quotas of 0 -> everything comes from the leftover phase
    chosen = select_balanced_papers(singles + [multi], target_count=1, quotas={})
    assert chosen[0].openalex_id == "M"


def test_deduplicate_merges_same_title_with_different_ids():
    # OpenAlex often has a preprint AND a conference record of one paper.
    preprint = Paper(
        openalex_id="W1",
        doi="10.48550/arxiv.2005.11401",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    )
    proceedings = Paper(
        openalex_id="W2",
        doi="10.5555/neurips.2020",
        title="Retrieval-augmented generation for knowledge-intensive NLP tasks.",
    )
    merged = deduplicate_papers({"relevant": [preprint], "highly_cited": [proceedings]})
    assert len(merged) == 1
    assert merged[0].openalex_id == "W1"
    assert merged[0].retrieval_reasons == ["relevant", "highly_cited"]


def test_deduplicate_keeps_short_generic_titles_apart():
    a = Paper(openalex_id="W1", title="Introduction")
    b = Paper(openalex_id="W2", title="Introduction")
    assert len(deduplicate_papers({"relevant": [a, b]})) == 2
