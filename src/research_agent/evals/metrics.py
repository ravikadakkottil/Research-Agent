"""
Score one pipeline run against the golden topic.

Metrics (all 0..1, higher is better, None = not applicable):

  recall               must-have papers in the FINAL reading list
  candidate_recall     must-have papers anywhere in the retrieved candidates
  shortlist_recall     must-have papers that survived the semantic rerank
                       candidate > shortlist  -> the rerank threshold dropped them
                       shortlist > recall     -> selection dropped them
  recall_foundational  recall over foundational must-haves only
  recall_recent        recall over recent must-haves only
  precision            judged-relevant papers / judged papers in the final list
  recent_precision     same, but only for papers chosen via the "recent" bucket
  grounding            summary's papers_used that really are in the list (full mode)

Plus diagnostics: recent-bucket median citations, future-dated papers, latency,
what the Europe PMC search did, and which must-haves only it (or only citation
expansion) found.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from statistics import median

from pydantic import BaseModel

from research_agent.evals.golden import GoldenPaper, GoldenTopic
from research_agent.evals.judge import Verdict
from research_agent.models import Paper, ReadingList, ResearchSummary
from research_agent.retrieval.selection import normalize_title, paper_identity

GROUNDING_TITLE_SIMILARITY = 0.85


class PaperRecord(BaseModel):
    """One selected paper, as stored in the results file for later inspection."""

    title: str | None
    year: int | None
    cited_by_count: int
    reasons: list[str]
    similarity: float | None = None
    cocited_by: int | None = None
    europepmc_only: bool = False
    relevant: bool | None  # None = judge gave no verdict
    judge_reason: str = ""
    is_golden: bool


class RunMetrics(BaseModel):
    topic_id: str
    run: int
    error: str | None = None

    recall: float | None = None
    candidate_recall: float | None = None
    shortlist_recall: float | None = None  # None = rerank skipped
    recall_foundational: float | None = None
    recall_recent: float | None = None
    precision: float | None = None
    recent_precision: float | None = None
    expansion_precision: float | None = None  # papers chosen via "cited_by_top"
    grounding: float | None = None

    recent_median_citations: float | None = None
    future_dated: int = 0
    latency_s: float = 0.0
    openalex_requests: int = 0

    query: str = ""
    recent_window: str = ""
    missed: list[str] = []  # must-haves not in the final list
    dropped_in_rerank: list[str] = []  # must-haves retrieved but below the similarity threshold
    dropped_in_selection: list[str] = []  # must-haves in the selection pool but not chosen
    found_by_expansion: list[str] = []  # must-haves retrieved ONLY by citation expansion
    found_by_europepmc: list[str] = []  # must-haves retrieved ONLY by the Europe PMC search
    europepmc: str | None = None  # "ok (25 papers)" / "skipped (...)" / "failed (...)" / "off"
    papers: list[PaperRecord] = []


def _found(golden: list[GoldenPaper], papers: list[Paper]) -> list[bool]:
    return [any(g.matches(p) for p in papers) for g in golden]


def _ratio(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _precision(papers: list[Paper], verdicts: dict[str, Verdict]) -> float | None:
    judged = [verdicts[paper_identity(p)].relevant for p in papers if paper_identity(p) in verdicts]
    return _ratio(judged)


def _grounding(summary: ResearchSummary, papers: list[Paper]) -> float | None:
    """Share of cited titles that match a paper actually in the reading list."""
    selected = [normalize_title(p.title) for p in papers]

    def is_grounded(title: str) -> bool:
        key = normalize_title(title)
        return any(
            SequenceMatcher(None, key, s).ratio() >= GROUNDING_TITLE_SIMILARITY for s in selected
        )

    return _ratio([is_grounded(u.title) for u in summary.papers_used])


def evaluate_run(
    topic: GoldenTopic,
    run: int,
    reading: ReadingList,
    verdicts: dict[str, Verdict],
    *,
    current_year: int,
    summary: ResearchSummary | None = None,
    latency_s: float = 0.0,
    openalex_requests: int = 0,
) -> RunMetrics:
    golden = topic.must_include
    in_final = _found(golden, reading.papers)
    in_candidates = _found(golden, reading.candidates)
    in_pool = _found(golden, reading.selection_pool)

    def recall_of(kind: str) -> float | None:
        return _ratio([f for g, f in zip(golden, in_final, strict=True) if g.kind == kind])

    recent_papers = [p for p in reading.papers if "recent" in p.retrieval_reasons]
    expansion_papers = [p for p in reading.papers if "cited_by_top" in p.retrieval_reasons]
    expansion_only = [p for p in reading.candidates if p.retrieval_reasons == ["cited_by_top"]]
    europepmc_only = [p for p in reading.candidates if p.europepmc_only]
    recent_citations = [p.cited_by_count for p in recent_papers]

    records = []
    for p in reading.papers:
        verdict = verdicts.get(paper_identity(p))
        records.append(
            PaperRecord(
                title=p.title,
                year=p.year,
                cited_by_count=p.cited_by_count,
                reasons=list(p.retrieval_reasons),
                similarity=p.similarity,
                cocited_by=p.cocited_by,
                europepmc_only=p.europepmc_only,
                relevant=verdict.relevant if verdict else None,
                judge_reason=verdict.reason if verdict else "",
                is_golden=any(g.matches(p) for g in golden),
            )
        )

    window = reading.recent_window
    return RunMetrics(
        topic_id=topic.id,
        run=run,
        recall=_ratio(in_final),
        candidate_recall=_ratio(in_candidates),
        shortlist_recall=_ratio(in_pool) if reading.shortlist is not None else None,
        recall_foundational=recall_of("foundational"),
        recall_recent=recall_of("recent"),
        precision=_precision(reading.papers, verdicts),
        recent_precision=_precision(recent_papers, verdicts),
        expansion_precision=_precision(expansion_papers, verdicts),
        grounding=_grounding(summary, reading.papers) if summary else None,
        recent_median_citations=median(recent_citations) if recent_citations else None,
        future_dated=sum(1 for p in reading.papers if (p.year or 0) > current_year),
        latency_s=round(latency_s, 2),
        openalex_requests=openalex_requests,
        query=reading.query_plan.describe(),
        recent_window=f"{window.from_year}+ ({window.source})",
        missed=[g.title for g, f in zip(golden, in_final, strict=True) if not f],
        dropped_in_rerank=[
            g.title
            for g, cand, pool in zip(golden, in_candidates, in_pool, strict=True)
            if cand and not pool
        ],
        dropped_in_selection=[
            g.title
            for g, final, pool in zip(golden, in_final, in_pool, strict=True)
            if pool and not final
        ],
        found_by_expansion=[g.title for g in golden if any(g.matches(p) for p in expansion_only)],
        found_by_europepmc=[g.title for g in golden if any(g.matches(p) for p in europepmc_only)],
        europepmc=reading.europepmc_status,
        papers=records,
    )
