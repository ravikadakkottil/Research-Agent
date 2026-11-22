"""
The research pipeline as a LangGraph graph.

    START
      -> plan        LLM #1 query plan + recent window (user > field pace > default)
      -> retrieve    OpenAlex buckets in parallel (+ Europe PMC if biomedical, fail-soft)
      -> dedupe
      -> score       semantic similarity (Gemini embeddings); skipped without an embedder
      -> expand      citation expansion            } conditional edges: each runs only
      -> filter      similarity threshold          } when its setting / embedder is on
      -> select      balanced reading list
      -> synthesize  LLM #2 grounded summary   (search graph only)
      -> END

Vocabulary (LangGraph):
  state     SearchState: a dict every node reads from. A node returns only the keys it
            changed, and LangGraph merges them in (later values overwrite earlier ones).
  context   SearchContext: per-run things that are NOT data, such as settings, the HTTP
            clients and the progress callback. Nodes get it as `runtime.context`.
  node      one async function per step (the `_node_*` functions below)
  edge      a fixed "next step"; a conditional edge calls a router function that
            returns the next node's name

Two compiled graphs share the same nodes:
  retrieve_reading_list()  plan ... select           evals score paper selection with it
  run_paper_search()       plan ... select + synthesize   what the CLI and web app use

No print() or input() here. Progress is reported through `on_step`. Each node keeps
its own Langfuse span, so traces look the same as before the port (see tracing.py).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from research_agent.cache import ResponseCache
from research_agent.config import Settings, get_settings
from research_agent.embeddings import Embedder, create_embedder
from research_agent.llm import create_llm_cache
from research_agent.models import (
    Paper,
    QueryPlan,
    ReadingList,
    RecentWindow,
    ResearchSummary,
    SearchResult,
)
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
        state = await _run(retrieval_graph(), user_question, settings, client, on_step)
        reading = _reading_list(state)
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

    # One trace per question; every node's span nests inside it (see tracing.py).
    with trace("search", input={"question": user_question}) as root:
        state = await _run(search_graph(), user_question, settings, client, on_step)
        reading = _reading_list(state)
        summary = state["summary"]
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
# State and context
# ---------------------------------------------------------------------------
class SearchState(TypedDict, total=False):
    """What flows between nodes. total=False: keys appear as nodes fill them in."""

    question: str
    plan: QueryPlan
    window: RecentWindow
    buckets: dict[str, list[Paper]]
    bucket_counts: dict[str, int]
    europepmc_status: str
    candidates: list[Paper]
    shortlist: list[Paper] | None  # None = rerank skipped
    papers: list[Paper]
    summary: ResearchSummary


@dataclass
class _Resources:
    client: OpenAlexClient
    llm_cache: ResponseCache | None
    embedder: Embedder | None
    europepmc: EuropePMCClient | None


@dataclass
class SearchContext:
    """Per-run dependencies, given to every node as `runtime.context`."""

    settings: Settings
    res: _Resources
    on_step: StepCallback
    started: float  # perf_counter() at the start, for the "Done in" message


Ctx = Runtime[SearchContext]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build(*, with_synthesis: bool) -> CompiledStateGraph:
    g = StateGraph(SearchState, context_schema=SearchContext)
    g.add_node("plan", _node_plan)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("dedupe", _node_dedupe)
    g.add_node("score", _node_score)
    g.add_node("expand", _node_expand)
    g.add_node("filter", _node_filter)
    g.add_node("select", _node_select)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "dedupe")
    g.add_edge("dedupe", "score")
    # The list of possible targets lets LangGraph draw the branches.
    g.add_conditional_edges("score", _route_after_score, ["expand", "filter", "select"])
    g.add_conditional_edges("expand", _route_after_expand, ["filter", "select"])
    g.add_edge("filter", "select")

    if with_synthesis:
        g.add_node("synthesize", _node_synthesize)
        g.add_edge("select", "synthesize")
        g.add_edge("synthesize", END)
    else:
        g.add_edge("select", END)
    return g.compile(name="search" if with_synthesis else "retrieval")


# Compiled once, reused by every run (a compiled graph holds no per-run data).
@lru_cache(maxsize=1)
def retrieval_graph() -> CompiledStateGraph:
    return _build(with_synthesis=False)


@lru_cache(maxsize=1)
def search_graph() -> CompiledStateGraph:
    return _build(with_synthesis=True)


def _route_after_score(state: SearchState, runtime: Ctx) -> Literal["expand", "filter", "select"]:
    if runtime.context.settings.citation_expansion_enabled:
        return "expand"
    return _route_after_expand(state, runtime)


def _route_after_expand(state: SearchState, runtime: Ctx) -> Literal["filter", "select"]:
    return "filter" if runtime.context.res.embedder is not None else "select"


async def _run(
    graph: CompiledStateGraph,
    user_question: str,
    settings: Settings,
    client: OpenAlexClient | None,
    on_step: StepCallback,
) -> SearchState:
    started = time.perf_counter()
    async with AsyncExitStack() as stack:
        res = await _open_resources(stack, settings, client)
        context = SearchContext(settings=settings, res=res, on_step=on_step, started=started)
        return await graph.ainvoke({"question": user_question}, context=context)


def _reading_list(state: SearchState) -> ReadingList:
    return ReadingList(
        query_plan=state["plan"],
        recent_window=state["window"],
        bucket_counts=state["bucket_counts"],
        candidates=state["candidates"],
        shortlist=state.get("shortlist"),
        papers=state["papers"],
        europepmc_status=state["europepmc_status"],
    )


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


# ---------------------------------------------------------------------------
# Nodes: each reads the state, does one step, returns the keys it changed
# ---------------------------------------------------------------------------
async def _node_plan(state: SearchState, runtime: Ctx) -> SearchState:
    settings, on_step = runtime.context.settings, runtime.context.on_step
    question = state["question"]
    current_year = date.today().year

    on_step("1) Asking LLM for a search plan...")
    with observe("plan", input={"question": question}) as obs:
        plan = await generate_query_plan(
            question, current_year=current_year, cache=runtime.context.res.llm_cache
        )
        window = resolve_recent_window(
            plan,
            question,
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
    return {"plan": plan, "window": window}


async def _node_retrieve(state: SearchState, runtime: Ctx) -> SearchState:
    settings, res, on_step = runtime.context.settings, runtime.context.res, runtime.context.on_step
    plan, client = state["plan"], res.client

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
            retrieve_buckets(client, plan, state["window"], settings),
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
    return {"buckets": buckets, "bucket_counts": bucket_counts, "europepmc_status": epmc_status}


async def _node_dedupe(state: SearchState, runtime: Ctx) -> SearchState:
    runtime.context.on_step("\n3) Deduplicating...")
    with observe("dedupe", input={"papers": sum(state["bucket_counts"].values())}) as obs:
        candidates = deduplicate_papers(state["buckets"])
        obs.update(output={"unique": len(candidates)})
    runtime.context.on_step(f"   unique candidates = {len(candidates)}")
    return {"candidates": candidates}


async def _score(papers: list[Paper], question: str, runtime: Ctx) -> list[Paper]:
    """Similarity to the question (unchanged papers when there's no embedder)."""
    embedder = runtime.context.res.embedder
    if embedder is None or not papers:
        return papers
    return await score_papers(
        question, papers, embedder, doc_chars=runtime.context.settings.embedding_doc_chars
    )


async def _node_score(state: SearchState, runtime: Ctx) -> SearchState:
    # Similarity first: expansion picks its seeds by it (most on-topic papers).
    on_step = runtime.context.on_step
    if runtime.context.res.embedder is None:
        on_step("\n4) Semantic scoring skipped (no GEMINI_API_KEY, or RERANK_ENABLED=false)")
        return {}  # nothing changed
    on_step("\n4) Semantic scoring (Gemini embeddings)...")
    t0 = time.perf_counter()
    candidates = state["candidates"]
    with observe("semantic scoring", input={"papers": len(candidates)}) as obs:
        candidates = await _score(candidates, state["question"], runtime)
        by_sim = sorted(candidates, key=lambda p: p.similarity or 0.0, reverse=True)
        obs.update(output={"top": papers_brief(by_sim[:5]), "bottom": papers_brief(by_sim[-3:])})
    on_step(f"   scored {len(candidates)} papers  (took {time.perf_counter() - t0:.1f}s)")
    return {"candidates": candidates}


async def _node_expand(state: SearchState, runtime: Ctx) -> SearchState:
    settings, on_step = runtime.context.settings, runtime.context.on_step
    on_step("\n5) Citation expansion (what do the top papers cite?)...")
    t0 = time.perf_counter()
    with observe(
        "citation expansion",
        input={"seeds": settings.expansion_seed_count, "min_citers": settings.expansion_min_citers},
    ) as obs:
        candidates, new_papers = await expand_by_citations(
            runtime.context.res.client,
            state["candidates"],
            seed_count=settings.expansion_seed_count,
            min_citers=settings.expansion_min_citers,
            max_papers=settings.expansion_max_papers,
        )
        new_papers = await _score(new_papers, state["question"], runtime)
        candidates = candidates + new_papers
        co_cited = sum(1 for p in candidates if "cited_by_top" in p.retrieval_reasons)
        obs.update(
            output={
                "co_cited": co_cited,
                "newly_fetched": [
                    f"cited by {p.cocited_by} | {paper_brief(p)}" for p in new_papers
                ],
            }
        )
    on_step(
        f"   {co_cited} co-cited papers, {len(new_papers)} newly fetched"
        f"  (took {time.perf_counter() - t0:.1f}s)"
    )
    for p in new_papers[:5]:
        on_step(f"   + cited by {p.cocited_by} top papers | {p.title or '(untitled)'}")
    # A new dict, not an in-place edit: a node should only change state by returning it.
    return {
        "candidates": candidates,
        "bucket_counts": {**state["bucket_counts"], "cited_by_top": co_cited},
    }


async def _node_filter(state: SearchState, runtime: Ctx) -> SearchState:
    settings, on_step = runtime.context.settings, runtime.context.on_step
    candidates = state["candidates"]
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
        weak = {id(p) for p in shortlist if is_weak_expansion(p, settings.expansion_min_similarity)}
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
            f" ({len(weak)} citation-only papers below {settings.expansion_min_similarity} removed)"
            if weak
            else ""
        )
    )
    for p in sorted(candidates, key=lambda p: p.similarity or 0.0)[:3]:
        on_step(f"   lowest: {p.similarity:.2f} | {p.title or '(untitled)'}")
    return {"shortlist": shortlist}


async def _node_select(state: SearchState, runtime: Ctx) -> SearchState:
    settings, on_step = runtime.context.settings, runtime.context.on_step
    on_step("\n6) Selecting a balanced reading list...")
    quotas = {
        "relevant": settings.select_relevant,
        "recent": settings.select_recent,
        "highly_cited": settings.select_highly_cited,
        "cited_by_top": settings.select_cited_by_top,
    }
    shortlist = state.get("shortlist")
    pool = shortlist if shortlist is not None else state["candidates"]
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
    return {"papers": selected}


async def _node_synthesize(state: SearchState, runtime: Ctx) -> SearchState:
    ctx = runtime.context
    llm_cache = ctx.res.llm_cache
    ctx.on_step("\n7) Asking LLM to synthesize the research...")
    summary = await summarize_papers(
        state["question"],
        state["papers"],
        recent_window=state["window"],
        abstract_chars=ctx.settings.abstract_chars_for_llm,
        cache=llm_cache,
    )
    llm_stats = f"hits={llm_cache.hits} misses={llm_cache.misses}" if llm_cache else "off"
    ctx.on_step(f"\nDone in {time.perf_counter() - ctx.started:.1f}s  (LLM cache {llm_stats})")
    return {"summary": summary}


# ---------------------------------------------------------------------------
# HTTP stats for the progress messages
# ---------------------------------------------------------------------------
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
