"""
The fixed research pipeline (Python decides the steps):

    question
      -> LLM #1 query plan                         ┐
      -> recent window (user > field pace > default) │ retrieve_reading_list()
      -> OpenAlex multi-bucket retrieval (parallel)  │  = the RETRIEVAL stage
         + Europe PMC if biomedical (fail-soft)      │
      -> dedupe                                      │
      -> semantic scoring (Gemini embeddings)        │
      -> citation expansion (co-cited landmarks)     │
      -> similarity threshold (rerank)               │
      -> balanced selection                        ┘
      -> LLM #2 synthesis                          -> run_paper_search() = both stages
      -> SearchResult

Two entry points:
  retrieve_reading_list()  retrieval only — used by evals to score paper selection
                           cheaply, and later by LangGraph as separate nodes
  run_paper_search()       retrieval + synthesis — what the CLI and web app use

No print() or input() here. Progress is reported through `on_step`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import date

from research_agent.cache import ResponseCache
from research_agent.config import Settings, get_settings
from research_agent.embeddings import Embedder, create_embedder
from research_agent.llm import create_llm_cache
from research_agent.models import Paper, ReadingList, SearchResult
from research_agent.planning import generate_query_plan
from research_agent.retrieval.buckets import retrieve_buckets
from research_agent.retrieval.expansion import expand_by_citations, is_weak_expansion
from research_agent.retrieval.recency import resolve_recent_window
from research_agent.retrieval.rerank import filter_by_similarity, score_papers
from research_agent.retrieval.second_source import fetch_europepmc
from research_agent.retrieval.selection import deduplicate_papers, select_balanced_papers
from research_agent.sources.europepmc import EuropePMCClient, create_europepmc_client
from research_agent.sources.openalex import OpenAlexClient, create_openalex_client
from research_agent.synthesis import summarize_papers
from research_agent.tracing import (
    current_trace_url,
    observe,
    paper_brief,
    papers_brief,
    trace,
)

StepCallback = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
async def retrieve_reading_list(
    user_question: str,
    *,
    on_step: StepCallback = _noop,
    settings: Settings | None = None,
    client: OpenAlexClient | None = None,
) -> ReadingList:
    """Retrieval stage only: plan, recent window, buckets, dedupe, rerank, selection."""
    settings = settings or get_settings()
    with trace("retrieval", input={"question": user_question}) as root:
        async with AsyncExitStack() as stack:
            res = await _open_resources(stack, settings, client)
            reading = await _retrieve(user_question, settings, res, on_step)
        root.update(output={"papers": papers_brief(reading.papers)})
        return reading


async def run_paper_search(
    user_question: str,
    *,
    on_step: StepCallback = _noop,
    settings: Settings | None = None,
    client: OpenAlexClient | None = None,
) -> SearchResult:
    """The full pipeline: retrieval stage + LLM synthesis."""
    settings = settings or get_settings()
    started = time.perf_counter()

    # One trace per question; every step below nests inside it (see tracing.py).
    with trace("search", input={"question": user_question}) as root:
        async with AsyncExitStack() as stack:
            res = await _open_resources(stack, settings, client)
            llm_cache = res.llm_cache
            reading = await _retrieve(user_question, settings, res, on_step)

            on_step("\n7) Asking LLM to synthesize the research...")
            summary = await summarize_papers(
                user_question,
                reading.papers,
                recent_window=reading.recent_window,
                abstract_chars=settings.abstract_chars_for_llm,
                cache=llm_cache,
            )
            llm_stats = f"hits={llm_cache.hits} misses={llm_cache.misses}" if llm_cache else "off"
            on_step(f"\nDone in {time.perf_counter() - started:.1f}s  (LLM cache {llm_stats})")

        root.update(output={"answer": summary.answer, "papers": papers_brief(reading.papers)})
        if url := await current_trace_url():
            on_step(f"Trace: {url}")

    return SearchResult(
        user_question=user_question,
        query_plan=reading.query_plan,
        recent_window=reading.recent_window,
        bucket_counts=reading.bucket_counts,
        unique_candidates=len(reading.candidates),
        papers=reading.papers,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
@dataclass
class _Resources:
    client: OpenAlexClient
    llm_cache: ResponseCache | None
    embedder: Embedder | None
    europepmc: EuropePMCClient | None


async def _open_resources(
    stack: AsyncExitStack, settings: Settings, client: OpenAlexClient | None
) -> _Resources:
    """
    Create what this run needs and register cleanup on the stack.

    AsyncExitStack: if WE create the client, we also close it at the end
    (even on errors). A client passed in by the caller is left open for them.
    """
    if client is None:
        client = await stack.enter_async_context(create_openalex_client(settings))
    llm_cache = create_llm_cache(settings)
    if llm_cache:
        stack.callback(llm_cache.close)
    embedder = create_embedder(settings)
    if embedder:
        await stack.enter_async_context(embedder)
    europepmc = create_europepmc_client(settings)
    if europepmc:
        await stack.enter_async_context(europepmc)
    return _Resources(client, llm_cache, embedder, europepmc)


async def _retrieve(
    user_question: str,
    settings: Settings,
    res: _Resources,
    on_step: StepCallback,
) -> ReadingList:
    client, llm_cache, embedder = res.client, res.llm_cache, res.embedder
    current_year = date.today().year

    on_step("1) Asking LLM for a search plan...")
    with observe("plan", input={"question": user_question}) as obs:
        plan = await generate_query_plan(user_question, current_year=current_year, cache=llm_cache)
        window = resolve_recent_window(
            plan,
            user_question,
            current_year=current_year,
            default_years=settings.default_recent_years,
            min_years=settings.min_recent_years,
            max_years=settings.max_recent_years,
        )
        obs.update(output={"plan": plan.model_dump(), "recent_window": window.model_dump()})
    on_step(f"   relevance query = {plan.search_query}")
    on_step(f"   core query      = {plan.strict_query}")
    if plan.alternatives:
        on_step(f"   alternatives    = {' / '.join(plan.alternatives)}")
    on_step(f"   reason = {plan.reason}")
    on_step(f"   recent = {window.from_year}-present  [{window.source}] {window.explanation}")

    on_step("\n2) Searching OpenAlex (relevant / recent / highly cited, in parallel)...")
    if plan.biomedical and res.europepmc:
        on_step("   + Europe PMC (biomedical topic)")
    t0 = time.perf_counter()
    before = _counters(client)  # the client may be shared (web app): report THIS run only

    async def europepmc_step() -> tuple[list[Paper], str]:
        with observe("europe pmc", input={"biomedical": plan.biomedical}) as obs:
            papers, status = await fetch_europepmc(res.europepmc, client, plan, settings)
            obs.update(output={"status": status, "papers": papers_brief(papers, 10)})
            return papers, status

    # "retriever" type: Langfuse shows it as a retrieval step; the OpenAlex
    # requests made by both parallel tasks nest inside it.
    with observe("retrieve", as_type="retriever") as obs:
        buckets, (epmc_papers, epmc_status) = await asyncio.gather(
            retrieve_buckets(client, plan, window, settings),
            europepmc_step(),
        )
        if epmc_papers:
            # Ranked by citations in Europe PMC -> they join the highly_cited perspective.
            buckets["highly_cited"] = sorted(
                [*buckets["highly_cited"], *epmc_papers],
                key=lambda p: p.cited_by_count,
                reverse=True,
            )
        bucket_counts = {name: len(papers) for name, papers in buckets.items()}
        http = _http_stats(client, before)
        obs.update(
            output={name: papers_brief(papers, 10) for name, papers in buckets.items()},
            metadata={"counts": bucket_counts, "europepmc": epmc_status, "http": http},
        )
    for name, count in bucket_counts.items():
        on_step(f"   {name:<13} = {count}")
    on_step(f"   europe pmc    = {epmc_status}")
    on_step(f"   took {time.perf_counter() - t0:.1f}s  ({http})")

    on_step("\n3) Deduplicating...")
    with observe("dedupe", input={"papers": sum(bucket_counts.values())}) as obs:
        candidates = deduplicate_papers(buckets)
        obs.update(output={"unique": len(candidates)})
    on_step(f"   unique candidates = {len(candidates)}")

    async def score(papers: list[Paper]) -> list[Paper]:
        if embedder is None or not papers:
            return papers
        return await score_papers(
            user_question, papers, embedder, doc_chars=settings.embedding_doc_chars
        )

    # Similarity first: expansion picks its seeds by it (most on-topic papers).
    if embedder is None:
        on_step("\n4) Semantic scoring skipped (no GEMINI_API_KEY, or RERANK_ENABLED=false)")
    else:
        on_step("\n4) Semantic scoring (Gemini embeddings)...")
        t0 = time.perf_counter()
        with observe("semantic scoring", input={"papers": len(candidates)}) as obs:
            candidates = await score(candidates)
            by_sim = sorted(candidates, key=lambda p: p.similarity or 0.0, reverse=True)
            obs.update(
                output={"top": papers_brief(by_sim[:5]), "bottom": papers_brief(by_sim[-3:])}
            )
        on_step(f"   scored {len(candidates)} papers  (took {time.perf_counter() - t0:.1f}s)")

    if settings.citation_expansion_enabled:
        on_step("\n5) Citation expansion (what do the top papers cite?)...")
        t0 = time.perf_counter()
        with observe(
            "citation expansion",
            input={
                "seeds": settings.expansion_seed_count,
                "min_citers": settings.expansion_min_citers,
            },
        ) as obs:
            candidates, new_papers = await expand_by_citations(
                client,
                candidates,
                seed_count=settings.expansion_seed_count,
                min_citers=settings.expansion_min_citers,
                max_papers=settings.expansion_max_papers,
            )
            new_papers = await score(new_papers)
            candidates = candidates + new_papers
            bucket_counts["cited_by_top"] = sum(
                1 for p in candidates if "cited_by_top" in p.retrieval_reasons
            )
            obs.update(
                output={
                    "co_cited": bucket_counts["cited_by_top"],
                    "newly_fetched": [
                        f"cited by {p.cocited_by} | {paper_brief(p)}" for p in new_papers
                    ],
                }
            )
        on_step(
            f"   {bucket_counts['cited_by_top']} co-cited papers, {len(new_papers)} newly fetched"
            f"  (took {time.perf_counter() - t0:.1f}s)"
        )
        for p in new_papers[:5]:
            on_step(f"   + cited by {p.cocited_by} top papers | {p.title or '(untitled)'}")

    shortlist = None
    if embedder is not None:
        with observe(
            "similarity filter",
            input={
                "candidates": len(candidates),
                "min_similarity": settings.rerank_min_similarity,
                "citation_only_min": settings.expansion_min_similarity,
            },
        ) as obs:
            shortlist = filter_by_similarity(
                candidates,
                min_similarity=settings.rerank_min_similarity,
                min_keep=settings.target_final_papers,
            )
            # Citation-only papers need a stricter bar (see config: expansion_min_similarity),
            # unless removing them would leave fewer papers than the reading list needs.
            weak = {
                id(p) for p in shortlist if is_weak_expansion(p, settings.expansion_min_similarity)
            }
            if weak and len(shortlist) - len(weak) >= settings.target_final_papers:
                removed = [p for p in shortlist if id(p) in weak]
                shortlist = [p for p in shortlist if id(p) not in weak]
            else:
                weak, removed = set(), []
            kept = {id(p) for p in shortlist}
            obs.update(
                output={
                    "kept": len(shortlist),
                    "dropped": papers_brief([p for p in candidates if id(p) not in kept]),
                    "citation_only_removed": papers_brief(removed),
                }
            )
        on_step(
            f"\n   rerank: kept {len(shortlist)}/{len(candidates)} with similarity >= "
            f"{settings.rerank_min_similarity}"
            + (
                f" ({len(weak)} citation-only papers below "
                f"{settings.expansion_min_similarity} removed)"
                if weak
                else ""
            )
        )
        for p in sorted(candidates, key=lambda p: p.similarity or 0.0)[:3]:
            on_step(f"   lowest: {p.similarity:.2f} | {p.title or '(untitled)'}")

    on_step("\n6) Selecting a balanced reading list...")
    quotas = {
        "relevant": settings.select_relevant,
        "recent": settings.select_recent,
        "highly_cited": settings.select_highly_cited,
        "cited_by_top": settings.select_cited_by_top,
    }
    pool = shortlist if shortlist is not None else candidates
    with observe("select", input={"pool": len(pool), "quotas": quotas}) as obs:
        selected = select_balanced_papers(
            pool, target_count=settings.target_final_papers, quotas=quotas
        )
        obs.update(
            output=[f"{paper_brief(p)} | {', '.join(p.retrieval_reasons)}" for p in selected]
        )
    for i, p in enumerate(selected, start=1):
        reasons = ", ".join(p.retrieval_reasons)
        sim = f"{p.similarity:.2f} | " if p.similarity is not None else ""
        on_step(f"   [{i:>2}] {p.year or '????'} | {sim}{p.title or '(untitled)'} | {reasons}")

    return ReadingList(
        query_plan=plan,
        recent_window=window,
        bucket_counts=bucket_counts,
        candidates=candidates,
        shortlist=shortlist,
        papers=selected,
        europepmc_status=epmc_status,
    )


def _counters(client: OpenAlexClient) -> tuple[int, int, int]:
    cache = client.cache
    return (client.network_requests, cache.hits if cache else 0, cache.misses if cache else 0)


def _http_stats(client: OpenAlexClient, before: tuple[int, int, int]) -> str:
    network, hits, misses = (
        now - then for now, then in zip(_counters(client), before, strict=True)
    )
    if client.cache is None:
        return f"{network} network requests, cache off"
    return f"{network} network requests, cache hits={hits} misses={misses}"
