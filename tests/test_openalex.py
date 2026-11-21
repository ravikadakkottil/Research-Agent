"""
OpenAlex parsing + client behaviour.

The client tests use httpx.MockTransport: a fake server function that receives
each request and returns a canned response. No network, fully deterministic.
Async code is driven with asyncio.run() inside normal test functions.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from research_agent.cache import ResponseCache
from research_agent.config import Settings
from research_agent.sources.openalex import (
    OpenAlexClient,
    OpenAlexRateLimitError,
    normalize_paper_id,
    parse_work,
    reconstruct_abstract,
    work_path,
)

SAMPLE_WORK = {
    "id": "https://openalex.org/W123",
    "doi": "https://doi.org/10.1/abc",
    "display_name": "A Paper",
    "publication_year": 2024,
    "cited_by_count": 42,
    "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
    "primary_location": {"source": {"display_name": "Nature"}},
    "abstract_inverted_index": {"learning": [1], "machine": [0]},
}
OK = {"results": [SAMPLE_WORK]}


# --- pure helpers -------------------------------------------------------------
def test_reconstruct_abstract_orders_words():
    assert reconstruct_abstract({"learning": [1], "machine": [0], "fun": [2]}) == (
        "machine learning fun"
    )
    assert reconstruct_abstract(None) == ""


def test_parse_work_maps_fields():
    paper = parse_work(SAMPLE_WORK)
    assert paper.title == "A Paper"
    assert paper.authors == ["Ada Lovelace"]
    assert paper.venue == "Nature"
    assert paper.abstract == "machine learning"


def test_parse_work_handles_missing_location():
    paper = parse_work({**SAMPLE_WORK, "primary_location": None}, include_abstract=False)
    assert paper.venue is None
    assert paper.abstract == ""


def test_normalize_paper_id_variants():
    assert normalize_paper_id("https://openalex.org/w123") == "W123"
    assert normalize_paper_id("10.1/abc") == "https://doi.org/10.1/abc"
    assert normalize_paper_id("https://doi.org/10.1/abc") == "https://doi.org/10.1/abc"


def test_work_path_uses_doi_prefix_form():
    assert work_path("W123") == "/works/W123"
    assert work_path("https://doi.org/10.1/abc") == "/works/doi:10.1/abc"


# --- client with a fake server -------------------------------------------------
class FakeServer:
    """Plays back a list of responses in order and records every request."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)


def make_settings(**overrides) -> Settings:
    defaults = {"groq_api_key": "x", "http_retry_wait_seconds": 0, "cache_enabled": False}
    return Settings(_env_file=None, **{**defaults, **overrides})


def make_client(
    server: Callable[[httpx.Request], httpx.Response],
    *,
    cache: ResponseCache | None = None,
    **settings_overrides,
) -> OpenAlexClient:
    return OpenAlexClient(
        make_settings(**settings_overrides), cache=cache, transport=httpx.MockTransport(server)
    )


async def _search(client: OpenAlexClient, query: str = "q", **kwargs):
    async with client:
        return await client.search_works(query, per_page=5, **kwargs)


def test_search_sends_api_key_and_filter():
    server = FakeServer(httpx.Response(200, json=OK))
    client = make_client(server, openalex_api_key="secret")

    papers = asyncio.run(_search(client, filters="from_publication_date:2024-01-01"))

    assert papers[0].title == "A Paper"
    params = server.requests[0].url.params
    assert params["api_key"] == "secret"
    assert params["filter"] == "from_publication_date:2024-01-01"


def test_server_error_is_retried_then_succeeds():
    server = FakeServer(httpx.Response(503), httpx.Response(500), httpx.Response(200, json=OK))
    client = make_client(server)

    papers = asyncio.run(_search(client))

    assert len(papers) == 1
    assert len(server.requests) == 3  # 2 failures + 1 success


def test_retries_give_up_after_max_attempts():
    server = FakeServer(*[httpx.Response(500) for _ in range(10)])
    client = make_client(server, http_max_retries=2)

    with pytest.raises(Exception, match="server error 500"):
        asyncio.run(_search(client))
    assert len(server.requests) == 3  # 1 try + 2 retries


def test_short_429_is_retried():
    server = FakeServer(
        httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json=OK)
    )
    papers = asyncio.run(_search(make_client(server)))
    assert len(papers) == 1


def test_budget_exhausted_429_fails_fast_without_retry():
    server = FakeServer(httpx.Response(429, headers={"Retry-After": "78000"}))
    with pytest.raises(OpenAlexRateLimitError, match="OPENALEX_API_KEY"):
        asyncio.run(_search(make_client(server)))
    assert len(server.requests) == 1


def test_client_error_is_not_retried():
    server = FakeServer(httpx.Response(400))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_search(make_client(server)))
    assert len(server.requests) == 1  # a bad request won't fix itself


def test_cache_answers_repeat_requests_without_network(tmp_path: Path):
    server = FakeServer(httpx.Response(200, json=OK))  # only ONE response available

    async def run_twice():
        cache = ResponseCache(tmp_path / "c.sqlite", ttl_seconds=60)
        async with make_client(server, cache=cache) as client:
            first = await client.search_works("q", per_page=5)
            second = await client.search_works("q", per_page=5)  # served from cache
            return first, second, client.network_requests, cache.hits

    first, second, network, hits = asyncio.run(run_twice())
    assert first == second
    assert (network, hits) == (1, 1)


def test_parallel_requests_overlap():
    """Three searches with gather should take ~1 delay, not 3 (proves they run concurrently)."""
    delay = 0.2

    async def slow_server(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay)
        return httpx.Response(200, json=OK)

    async def three_searches():
        async with make_client(slow_server) as client:
            loop = asyncio.get_running_loop()
            start = loop.time()
            await asyncio.gather(*(client.search_works(f"q{i}", per_page=5) for i in range(3)))
            return loop.time() - start

    elapsed = asyncio.run(three_searches())
    assert elapsed < delay * 2.5  # sequential would be >= 0.6s


# --- search scope (Step 4) -------------------------------------------------------
def test_title_abstract_scope_uses_filter_not_fulltext_search():
    server = FakeServer(httpx.Response(200, json=OK))
    client = make_client(server)

    asyncio.run(
        _search(
            client,
            "tool use, LLM agents: a|b",
            filters="from_publication_date:2024-01-01",
            scope="title_abstract",
        )
    )

    params = server.requests[0].url.params
    assert "search" not in params  # no full-text matching
    # commas / pipes / colons would break OpenAlex filter syntax -> removed
    assert params["filter"] == (
        "title_and_abstract.search:tool use LLM agents a b,from_publication_date:2024-01-01"
    )


def test_default_scope_is_full_search():
    server = FakeServer(httpx.Response(200, json=OK))
    asyncio.run(_search(make_client(server), "rag"))
    params = server.requests[0].url.params
    assert params["search"] == "rag"
    assert "filter" not in params


# --- citation expansion support ---------------------------------------------------
def test_parse_work_keeps_short_reference_ids():
    work = {
        **SAMPLE_WORK,
        "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
    }
    assert parse_work(work).referenced_works == ["W1", "W2"]


def test_search_asks_for_reference_lists():
    server = FakeServer(httpx.Response(200, json=OK))
    asyncio.run(_search(make_client(server), "rag"))
    assert "referenced_works" in server.requests[0].url.params["select"]


def test_get_works_by_ids_uses_one_or_filter_per_chunk(monkeypatch):
    monkeypatch.setattr("research_agent.sources.openalex.MAX_IDS_PER_REQUEST", 2)
    server = FakeServer(*(httpx.Response(200, json=OK) for _ in range(2)))
    client = make_client(server)

    async def go():
        async with client:
            return await client.get_works_by_ids(["W1", "https://openalex.org/W2", "W3", "W1"])

    papers = asyncio.run(go())

    filters = sorted(r.url.params["filter"] for r in server.requests)
    assert filters == ["openalex_id:W1|W2", "openalex_id:W3"]  # deduped, chunked
    assert len(papers) == 2  # one canned work per response


def test_get_works_by_dois_normalizes_dedupes_and_skips_unsafe():
    server = FakeServer(httpx.Response(200, json=OK))
    client = make_client(server)

    async def go():
        async with client:
            return await client.get_works_by_dois(
                ["10.1/ABC", "https://doi.org/10.1/abc", "10.2/x,y", "10.3/z"]
            )

    asyncio.run(go())

    params = server.requests[0].url.params
    # same DOI twice (case differs) -> once; "," would break the filter -> skipped
    assert params["filter"] == "doi:https://doi.org/10.1/abc|https://doi.org/10.3/z"
    assert "referenced_works" in params["select"]  # needed by citation expansion
