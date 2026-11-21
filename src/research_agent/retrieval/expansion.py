"""
Citation expansion ("snowballing"): find landmarks keyword search can't.

Keyword search asks "which papers contain these words?". Some landmarks don't:
"The spread of true and false news online" never says "misinformation", and
ReAct's abstract doesn't say "tool use". But nearly every good paper on the
topic CITES them. So:

    top candidates ("seeds", most on-topic first)
      -> read their reference lists (came with the search results: no extra request)
      -> count how many seeds cite each referenced paper (co-citation)
      -> cited by >= min_citers seeds:
           already a candidate -> tag it "cited_by_top" (it competes for that quota,
                                  e.g. a landmark that lost on raw citation count)
           not retrieved yet   -> fetch it (one batched request), tag "cited_by_top"

Generic papers cited everywhere (Adam, ImageNet, LSTM) can pass the count too.
The rerank removes most of them (their meaning is far from the question), and
papers found ONLY by expansion must clear a stricter similarity bar
(`is_weak_expansion`), because whole-field backbones land just above the
normal threshold.
"""

from __future__ import annotations

from collections import Counter

from research_agent.models import Paper
from research_agent.retrieval.selection import normalize_title, paper_identity
from research_agent.sources.openalex import OpenAlexClient, short_id

REASON = "cited_by_top"


def is_weak_expansion(paper: Paper, min_similarity: float) -> bool:
    """Found only via citations, and not similar enough to the question to keep."""
    return (
        paper.retrieval_reasons == [REASON]
        and paper.similarity is not None
        and paper.similarity < min_similarity
    )


def pick_seeds(candidates: list[Paper], count: int) -> list[Paper]:
    """
    The papers whose references we trust: most similar to the question first
    (if reranked), then those found by several buckets, then most cited.
    """
    ranked = sorted(
        candidates,
        key=lambda p: (p.similarity or 0.0, len(p.retrieval_reasons), p.cited_by_count),
        reverse=True,
    )
    return ranked[:count]


def count_cocitations(seeds: list[Paper], *, min_citers: int) -> list[tuple[str, int]]:
    """
    [(openalex_id, number_of_seeds_citing_it), ...] for IDs cited by >= min_citers
    seeds, most co-cited first. Pure function (no I/O), so it's easy to test.
    """
    counts: Counter[str] = Counter()
    for seed in seeds:
        counts.update(set(seed.referenced_works))  # set: a paper counts once per seed
    eligible = [(work_id, n) for work_id, n in counts.items() if n >= min_citers]
    # Most co-cited first; ID as tie-breaker so the result is deterministic.
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return eligible


def _tag(paper: Paper, cocited_by: int) -> Paper:
    reasons = list(paper.retrieval_reasons)
    if REASON not in reasons:
        reasons.append(REASON)
    return paper.model_copy(update={"retrieval_reasons": reasons, "cocited_by": cocited_by})


async def expand_by_citations(
    client: OpenAlexClient,
    candidates: list[Paper],
    *,
    seed_count: int,
    min_citers: int,
    max_papers: int,
) -> tuple[list[Paper], list[Paper]]:
    """
    Returns (candidates, new_papers):
      candidates  the input list, with co-cited ones tagged "cited_by_top" (copies)
      new_papers  papers not retrieved before, fetched and tagged (at most max_papers)
    """
    cocited = dict(count_cocitations(pick_seeds(candidates, seed_count), min_citers=min_citers))
    if not cocited:
        return candidates, []

    # 1) Tag candidates we already have.
    have_ids: set[str] = set()
    tagged = []
    for paper in candidates:
        pid = short_id(paper.openalex_id) if paper.openalex_id else ""
        have_ids.add(pid)
        tagged.append(_tag(paper, cocited[pid]) if pid in cocited else paper)

    # 2) Fetch the most co-cited papers we don't have yet.
    wanted = [work_id for work_id in cocited if work_id not in have_ids][:max_papers]
    if not wanted:
        return tagged, []
    fetched = await client.get_works_by_ids(wanted)

    # Skip duplicates of papers we already have under another record (other DOI/ID).
    have_keys = {paper_identity(p) for p in candidates}
    have_titles = {normalize_title(p.title) for p in candidates if p.title}
    new_papers = []
    for paper in fetched:
        title_key = normalize_title(paper.title)
        if paper_identity(paper) in have_keys or (title_key and title_key in have_titles):
            continue
        paper = paper.model_copy(update={"retrieval_reasons": []})
        new_papers.append(_tag(paper, cocited.get(short_id(paper.openalex_id or ""), 0)))
    new_papers.sort(key=lambda p: p.cocited_by or 0, reverse=True)
    return tagged, new_papers
