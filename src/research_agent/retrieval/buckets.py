"""
Multi-bucket retrieval: look at the topic from three perspectives.

Relevance alone can miss foundational or very new work, so we run:
  relevant      — best keyword match (title + abstract + full text)
  recent        — most-cited papers published inside the RecentWindow
  highly_cited  — most cited overall (foundational / influential)

Which query each bucket uses (Step 4, fix 3 — see QueryPlan):
  relevant      plan.search_query         (full text, forgiving)
  recent        plan.strict_query         (title + abstract)
  highly_cited  plan.strict_query AND each alternative phrasing, merged and
                sorted by citations — so landmarks that use different
                vocabulary ("false news" vs "misinformation") are found too

Why "recent" and "highly_cited" search only TITLE + ABSTRACT (Step 4, fix 1):
  both sort by citations. With full-text matching, huge generic papers that
  mention the topic once ("Cancer statistics, 2024", "GPT-4 Technical Report")
  win on citations. Requiring the topic in the title/abstract keeps papers that
  are ABOUT it. The relevance bucket keeps full text: its ranking already
  rewards real topical matches (baseline precision 0.92).

Why "recent" sorts by citations, not newest-first (Step 4, fix 2):
  newest-first returned days-old, uncited (often off-topic) papers. Within the
  window, citations show which recent work the field has actually noticed.
  `to_publication_date:<today>` drops records OpenAlex dates in the future.

The three searches are independent, so they run IN PARALLEL with
asyncio.gather: total time ~= the slowest search, not the sum of all three.
"""

from __future__ import annotations

import asyncio
from datetime import date

from research_agent.config import Settings
from research_agent.models import Paper, QueryPlan, RecentWindow, RetrievalReason
from research_agent.sources.openalex import OpenAlexClient


def recent_window_filter(window: RecentWindow, today: date) -> str:
    """Publication-date range for the recent bucket: window start .. today (no future dates)."""
    return f"from_publication_date:{window.from_year}-01-01,to_publication_date:{today.isoformat()}"


async def retrieve_buckets(
    client: OpenAlexClient,
    plan: QueryPlan,
    window: RecentWindow,
    settings: Settings,
    *,
    today: date | None = None,
) -> dict[RetrievalReason, list[Paper]]:
    today = today or date.today()

    def highly_cited(query: str):
        # Sorted by citations, so older landmark papers surface.
        return client.search_works(
            query,
            per_page=settings.highly_cited_count,
            sort="cited_by_count:desc",
            scope="title_abstract",
        )

    relevant, recent, *cited_lists = await asyncio.gather(
        client.search_works(
            plan.search_query,
            per_page=settings.relevant_count,
            sort="relevance_score:desc",
        ),
        # Recency is decided by an OpenAlex date FILTER, not by the LLM.
        client.search_works(
            plan.strict_query,
            per_page=settings.recent_count,
            sort="cited_by_count:desc",
            filters=recent_window_filter(window, today),
            scope="title_abstract",
        ),
        *(highly_cited(q) for q in [plan.strict_query, *plan.alternatives]),
    )
    merged_cited = sorted(
        (p for papers in cited_lists for p in papers),
        key=lambda p: p.cited_by_count,
        reverse=True,
    )
    return {"relevant": relevant, "recent": recent, "highly_cited": merged_cited}
