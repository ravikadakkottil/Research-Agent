"""
Europe PMC as a second source, merged into OpenAlex records.

    biomedical question?  no -> skipped
      -> Europe PMC: relevance query, full text, most cited first (top N)
      -> look up their DOIs in OpenAlex (ONE batched request)
           found      -> use the OpenAlex record: same IDs, citation counts and
                         reference lists as every other candidate (citation
                         expansion needs those); if OpenAlex has no abstract,
                         take Europe PMC's (the reason these papers were missed)
           not found  -> keep the Europe PMC record as is
      -> papers join the highly_cited bucket (they were ranked by citations)

FAIL-SOFT: Europe PMC is slow and variable. Any error or the time limit means
"no extra papers", never a failed question. The status string says what
happened, so the eval can tell "didn't help" from "didn't answer".
"""

from __future__ import annotations

import asyncio

from research_agent.config import Settings
from research_agent.models import Paper, QueryPlan
from research_agent.sources.europepmc import EuropePMCClient
from research_agent.sources.openalex import OpenAlexClient


def merge_with_openalex(epmc_papers: list[Paper], openalex_papers: list[Paper]) -> list[Paper]:
    """Replace each Europe PMC record with its OpenAlex twin (matched by DOI) when there is one."""
    by_doi = {p.doi.lower(): p for p in openalex_papers if p.doi}
    merged = []
    for epmc in epmc_papers:
        twin = by_doi.get(epmc.doi.lower()) if epmc.doi else None
        if twin is None:
            merged.append(epmc)
            continue
        update: dict = {"europepmc_only": True}
        if not twin.abstract and epmc.abstract:
            update["abstract"] = epmc.abstract
        merged.append(twin.model_copy(update=update))
    return merged


async def fetch_europepmc(
    epmc: EuropePMCClient | None,
    client: OpenAlexClient,
    plan: QueryPlan,
    settings: Settings,
) -> tuple[list[Paper], str]:
    """Returns (papers, status). Never raises."""
    if epmc is None:
        return [], "off"
    if not plan.biomedical:
        return [], "skipped (not a biomedical topic)"

    async def run() -> list[Paper]:
        found = await epmc.search(plan.search_query, page_size=settings.europepmc_count)
        dois = [p.doi for p in found if p.doi]
        twins = await client.get_works_by_dois(dois) if dois else []
        return merge_with_openalex(found, twins)

    try:
        papers = await asyncio.wait_for(run(), timeout=settings.europepmc_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - a bonus source must never fail the question
        reason = "timed out" if isinstance(exc, TimeoutError) else type(exc).__name__
        return [], f"failed ({reason})"
    return papers, f"ok ({len(papers)} papers)"
