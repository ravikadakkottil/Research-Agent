"""
Deduplicate papers across buckets, then pick a balanced reading list.

Deterministic Python — no LLM ranking here (transparent and testable).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from research_agent.models import Paper, RetrievalReason


# ---------------------------------------------------------------------------
# Identity + dedupe
# ---------------------------------------------------------------------------
def normalize_title(title: str | None) -> str:
    """Fallback identity key when DOI / OpenAlex ID are missing."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def paper_identity(paper: Paper) -> str:
    """Prefer stable IDs: DOI > OpenAlex ID > normalized title."""
    if paper.doi and paper.doi.strip():
        return f"doi:{paper.doi.strip().lower()}"
    if paper.openalex_id and paper.openalex_id.strip():
        return f"openalex:{paper.openalex_id.strip().lower()}"
    if title_key := normalize_title(paper.title):
        return f"title:{title_key}"
    return f"fallback:{id(paper)}"  # should be rare


# Titles shorter than this ("Introduction", "Editorial") are too generic to merge on.
MIN_TITLE_KEY_FOR_DEDUPE = 20


def deduplicate_papers(buckets: dict[RetrievalReason, list[Paper]]) -> list[Paper]:
    """
    Merge all buckets. A paper found by several buckets is kept once,
    with every reason recorded (e.g. ["relevant", "highly_cited"]).

    Also merges records with the SAME TITLE but different IDs: OpenAlex often
    holds a paper twice (arXiv preprint + conference version, different DOIs).
    The first record seen is kept.
    """
    merged: dict[str, Paper] = {}
    key_by_title: dict[str, str] = {}
    for reason, papers in buckets.items():
        for paper in papers:
            key = paper_identity(paper)
            title_key = normalize_title(paper.title)
            if len(title_key) >= MIN_TITLE_KEY_FOR_DEDUPE:
                key = key_by_title.setdefault(title_key, key)
            if key not in merged:
                # model_copy: don't mutate the bucket's object; fresh reasons list.
                merged[key] = paper.model_copy(update={"retrieval_reasons": [reason]})
            elif reason not in merged[key].retrieval_reasons:
                merged[key].retrieval_reasons.append(reason)
    return list(merged.values())


# ---------------------------------------------------------------------------
# Sort keys for each perspective
# ---------------------------------------------------------------------------
# `similarity` is None when the semantic rerank was skipped -> counts as 0, so
# the keys fall back to the keyword-only ordering.
def _n_reasons(p: Paper) -> int:
    return len(p.retrieval_reasons)


def _sim(p: Paper) -> float:
    return p.similarity or 0.0


def _key_relevant(p: Paper) -> tuple:
    return (_sim(p), _n_reasons(p), p.cited_by_count, p.year or 0)


def _key_recent(p: Paper) -> tuple:
    return (p.year or 0, p.cited_by_count, _n_reasons(p))


def _key_highly_cited(p: Paper) -> tuple:
    return (p.cited_by_count, _n_reasons(p), p.year or 0)


def _key_multi(p: Paper) -> tuple:
    return (_n_reasons(p), _sim(p), p.cited_by_count, p.year or 0)


# ---------------------------------------------------------------------------
# Balanced selection
# ---------------------------------------------------------------------------
def select_balanced_papers(
    candidates: list[Paper],
    *,
    target_count: int,
    quotas: dict[RetrievalReason, int],
) -> list[Paper]:
    """
    Soft quotas per perspective, then fill the rest with multi-bucket papers.

      1) up to quotas["relevant"] relevant papers
      2) up to quotas["recent"] recent papers
      3) up to quotas["highly_cited"] highly cited papers
      4) remaining slots -> papers found by several buckets / strongest leftovers

    "Soft" = if a bucket has fewer papers, we don't force the count.
    """
    selected: list[Paper] = []
    chosen: set[str] = set()

    def add_from(papers: Iterable[Paper], limit: int) -> None:
        added = 0
        for paper in papers:
            if added >= limit or len(selected) >= target_count:
                return
            key = paper_identity(paper)
            if key not in chosen:
                selected.append(paper)
                chosen.add(key)
                added += 1

    sort_keys: dict[RetrievalReason, Callable[[Paper], tuple]] = {
        "relevant": _key_relevant,
        "recent": _key_recent,
        "highly_cited": _key_highly_cited,
    }
    for reason, key_fn in sort_keys.items():
        in_bucket = [p for p in candidates if reason in p.retrieval_reasons]
        add_from(sorted(in_bucket, key=key_fn, reverse=True), quotas.get(reason, 0))

    leftovers = [p for p in candidates if paper_identity(p) not in chosen]
    add_from(sorted(leftovers, key=_key_multi, reverse=True), target_count)
    return selected
