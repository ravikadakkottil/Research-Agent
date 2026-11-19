"""The buckets send the right query, scope, sort and date range to OpenAlex."""

import asyncio
from datetime import date

from research_agent.config import Settings
from research_agent.models import Paper, QueryPlan, RecentWindow
from research_agent.retrieval.buckets import retrieve_buckets

WINDOW = RecentWindow(from_year=2023, source="default", explanation="")
SETTINGS = Settings(_env_file=None, groq_api_key="x")


class RecordingClient:
    """Stands in for OpenAlexClient: records each search_works call, returns canned papers."""

    def __init__(self, results: dict[str, list[Paper]] | None = None) -> None:
        self.calls: list[dict] = []
        self.results = results or {}

    async def search_works(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.results.get(query, [])


def run(client, plan):
    return asyncio.run(retrieve_buckets(client, plan, WINDOW, SETTINGS, today=date(2026, 11, 19)))


def test_each_bucket_uses_its_query_scope_and_sort():
    client = RecordingClient()
    plan = QueryPlan(
        search_query="perovskite solar cells efficiency", core_query="perovskite solar cells"
    )

    run(client, plan)

    relevant, recent, highly_cited = client.calls  # no alternatives -> 3 searches
    assert relevant["query"] == "perovskite solar cells efficiency"  # longer query, full text
    assert relevant["sort"] == "relevance_score:desc"
    assert relevant.get("scope", "all") == "all"

    assert recent["query"] == "perovskite solar cells"  # short core query, strict scope
    assert recent["sort"] == "cited_by_count:desc"  # influential recent work, not newest
    assert recent["scope"] == "title_abstract"
    assert recent["filters"] == "from_publication_date:2023-01-01,to_publication_date:2026-11-19"

    assert highly_cited["query"] == "perovskite solar cells"
    assert highly_cited["sort"] == "cited_by_count:desc"
    assert highly_cited["scope"] == "title_abstract"


def test_core_query_falls_back_to_search_query():
    client = RecordingClient()
    run(client, QueryPlan(search_query="graph neural networks"))
    assert client.calls[1]["query"] == client.calls[2]["query"] == "graph neural networks"


def test_alternatives_add_highly_cited_searches_merged_by_citations():
    classic = Paper(
        openalex_id="W1", title="The spread of true and false news online", cited_by_count=9000
    )
    other = Paper(openalex_id="W2", title="Misinformation on social media", cited_by_count=500)
    client = RecordingClient({"misinformation": [other], "false news": [classic]})
    plan = QueryPlan(
        search_query="misinformation social media",
        core_query="misinformation",
        alternative_queries=["false news"],
    )

    buckets = run(client, plan)

    assert [c["query"] for c in client.calls[2:]] == ["misinformation", "false news"]
    assert all(c["scope"] == "title_abstract" for c in client.calls[2:])
    assert buckets["highly_cited"] == [classic, other]  # merged, most cited first
