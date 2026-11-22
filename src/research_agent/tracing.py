"""
Tracing with Langfuse: see inside every run.

A TRACE is one question's timeline. Inside it, OBSERVATIONS nest like folders:

    trace "search"
      ├─ span "plan"
      │    └─ generation "llm: QueryPlan"      prompt, raw reply, tokens, latency
      ├─ retriever "retrieve"
      │    ├─ span "openalex /works"           query params, result count, cache hit
      │    └─ ...
      ├─ span "semantic scoring"
      │    └─ embedding "gemini embed"         texts, cache hits
      ├─ ...
      └─ span "synthesize"
           └─ generation "llm: ResearchSummary"

Nesting is automatic: Langfuse is built on OpenTelemetry, which keeps the
"current observation" in a context variable. Whatever starts inside a `with`
block becomes its child, even across `await` and asyncio.gather (each task
copies the context it was created in).

Design rules:
  - Optional: no LANGFUSE_* keys, or TRACING_ENABLED=false -> every helper
    here is a no-op, and the langfuse package isn't even imported.
  - Never breaks a run: spans are sent by a background thread in batches;
    a Langfuse outage only loses traces.
  - No secrets in traces: API keys are added to requests AFTER the traced
    parameters are captured (see sources/openalex.py).
  - Small payloads: steps log compact summaries (titles, counts, scores),
    not whole paper objects.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from research_agent.config import get_settings

if TYPE_CHECKING:  # only for type hints; never imported at runtime when tracing is off
    from langfuse import Langfuse

    from research_agent.models import Paper


class _NoopObservation:
    """Stands in for a Langfuse observation when tracing is off: accepts and ignores calls."""

    def update(self, **_: Any) -> None:
        pass

    def score_trace(self, **_: Any) -> None:
        pass


NOOP = _NoopObservation()


@lru_cache
def _client(public_key: str, secret_key: str, base_url: str) -> Langfuse:
    from langfuse import Langfuse  # imported lazily: pulls in OpenTelemetry

    return Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)


def get_langfuse() -> Langfuse | None:
    """
    The shared Langfuse client, or None when tracing is off.

    Keys are passed explicitly: pydantic-settings reads .env into Settings but
    does NOT put the values into os.environ, where the SDK would look for them.
    """
    s = get_settings()
    if not (s.tracing_enabled and s.langfuse_public_key and s.langfuse_secret_key):
        return None
    return _client(
        s.langfuse_public_key.get_secret_value(),
        s.langfuse_secret_key.get_secret_value(),
        s.langfuse_base_url.rstrip("/"),
    )


@contextmanager
def observe(name: str, *, as_type: str = "span", **fields: Any) -> Iterator[Any]:
    """
    One observation (a step) inside the current trace:

        with observe("dedupe", input={"papers": 60}) as obs:
            ...
            obs.update(output={"unique": 41})

    as_type: span | generation (an LLM call) | embedding | retriever | tool | agent | chain.
    `fields`: input, output, metadata, model, model_parameters, usage_details, ...
    An exception is recorded on the observation (level ERROR) and re-raised.
    """
    lf = get_langfuse()
    if lf is None:
        yield NOOP
        return
    with lf.start_as_current_observation(name=name, as_type=as_type, **fields) as obs:
        try:
            yield obs
        except BaseException as exc:
            obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])
            raise


@contextmanager
def trace(
    name: str,
    *,
    as_type: str = "chain",
    input: Any = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """
    Start a new trace (one question), or, if a trace is already running, a child span.

    The second case is how evals work: the eval runner opens the trace (so it can
    attach scores to it), and the pipeline's own trace() becomes a step inside it.

    tags / session_id / metadata are copied onto every observation created inside
    (propagate_attributes), so traces can be filtered by them in the Langfuse UI.
    Sessions group traces: all 39 runs of one eval share a session.
    """
    lf = get_langfuse()
    if lf is None:
        yield NOOP
        return
    if _inside_trace():
        with observe(name, as_type=as_type, input=input, metadata=metadata) as obs:
            yield obs
        return

    from langfuse import propagate_attributes

    attributes = {
        # Propagated metadata must be strings (Langfuse limit: 200 chars each).
        "metadata": {k: str(v)[:200] for k, v in (metadata or {}).items()} or None,
        "tags": tags,
        "session_id": session_id,
    }
    with observe(name, as_type=as_type, input=input) as obs:
        with propagate_attributes(
            trace_name=name, **{k: v for k, v in attributes.items() if v is not None}
        ):
            yield obs


def _inside_trace() -> bool:
    """
    Is a span already active? Asked of OpenTelemetry directly: Langfuse's
    get_current_trace_id() logs a 'No active span' warning when the answer is no.
    """
    from opentelemetry import trace as otel_trace

    return otel_trace.get_current_span().get_span_context().is_valid


async def current_trace_url() -> str | None:
    """
    Link to the running trace in the Langfuse UI (None when tracing is off).

    The first call asks the Langfuse API for the project id (a blocking HTTP
    request), so it runs in a worker thread instead of freezing the event loop.
    """
    lf = get_langfuse()
    if lf is None or not (trace_id := lf.get_current_trace_id()):
        return None
    try:
        return await asyncio.to_thread(lf.get_trace_url, trace_id=trace_id)
    except Exception:  # noqa: BLE001 - a missing link must never fail a run
        return None


def score_trace(name: str, value: float | None, comment: str | None = None) -> None:
    """Attach a score (e.g. eval recall) to the running trace. None values are skipped."""
    lf = get_langfuse()
    if lf is not None and value is not None:
        lf.score_current_trace(name=name, value=float(value), comment=comment)


def flush_tracing() -> None:
    """
    Send buffered spans now. Call before a short-lived process exits (CLI, evals);
    otherwise the last batch may be lost.
    """
    lf = get_langfuse()
    if lf is not None:
        lf.flush()


# ---------------------------------------------------------------------------
# Compact views of papers for trace payloads
# ---------------------------------------------------------------------------
def paper_brief(p: Paper) -> str:
    """One line per paper: '2017 | 0.74 | 120,345 cites | Attention Is All You Need'."""
    sim = f"{p.similarity:.2f} | " if p.similarity is not None else ""
    return (
        f"{p.year or '????'} | {sim}{p.cited_by_count:,} cites | {(p.title or '(untitled)')[:120]}"
    )


def papers_brief(papers: Iterable[Paper], limit: int = 25) -> list[str]:
    papers = list(papers)
    lines = [paper_brief(p) for p in papers[:limit]]
    if len(papers) > limit:
        lines.append(f"... and {len(papers) - limit} more")
    return lines
