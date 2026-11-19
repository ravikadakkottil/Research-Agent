"""
The ONLY module that talks to OpenAlex.

Step 3 upgrades, all hidden inside this client (callers just `await` methods):
  - ASYNC (httpx.AsyncClient): many requests can be in flight at once.
  - CONCURRENCY LIMIT (asyncio.Semaphore): at most N at a time, to stay polite.
  - RETRIES (tenacity): temporary failures are retried with exponential backoff.
  - CACHE (ResponseCache): identical requests are answered from disk.

This is plain HTTP to a database API — no LLM involved here.
"""

from __future__ import annotations

import asyncio
import re
import ssl
from typing import Any, Literal

import certifi
import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from research_agent.cache import ResponseCache
from research_agent.config import Settings
from research_agent.models import Paper

# Fields we ask OpenAlex for. Smaller responses = faster + cheaper.
BASE_FIELDS = "id,doi,display_name,publication_year,cited_by_count,authorships,primary_location"
FIELDS_WITH_ABSTRACT = BASE_FIELDS + ",abstract_inverted_index"

# A 429 asking us to wait at most this long is a short burst limit -> retry.
# Longer (or no Retry-After) means the daily budget is gone -> fail fast.
MAX_SHORT_RETRY_AFTER_SECONDS = 10

# Where keywords must appear:
#   "all"             title + abstract + FULL TEXT (OpenAlex `search=`): broad, but a famous
#                     paper that mentions the topic once in its body also matches
#   "title_abstract"  title + abstract only: the paper is actually ABOUT the topic
SearchScope = Literal["all", "title_abstract"]

# Characters with special meaning inside an OpenAlex `filter=` value
# ("," separates filters, "|" means OR, ":" separates key and value).
_FILTER_SPECIAL = re.compile(r"[,|:]+")


def title_abstract_filter(query: str) -> str:
    """OpenAlex filter that searches only titles and abstracts (special chars removed)."""
    cleaned = " ".join(_FILTER_SPECIAL.sub(" ", query).split())
    return f"title_and_abstract.search:{cleaned}"


class OpenAlexRateLimitError(RuntimeError):
    """OpenAlex answered 429 and waiting a few seconds won't help (daily budget used up)."""


class RetryableHTTPError(RuntimeError):
    """A temporary failure worth retrying (5xx, or a short 429)."""


# ---------------------------------------------------------------------------
# Pure helpers (no network) — easy to unit test
# ---------------------------------------------------------------------------
def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    OpenAlex stores abstracts as an "inverted index": word -> positions.

        {"machine": [0], "learning": [1]}  ->  "machine learning"
    """
    if not inverted_index:
        return ""
    positioned = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    positioned.sort()
    return " ".join(word for _, word in positioned)


def parse_work(work: dict[str, Any], *, include_abstract: bool = True) -> Paper:
    """Convert one OpenAlex work payload into our simpler Paper model."""
    authors = [
        (a.get("author") or {}).get("display_name", "Unknown")
        for a in (work.get("authorships") or [])[:5]
        if isinstance(a, dict)
    ]
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    abstract = reconstruct_abstract(work.get("abstract_inverted_index")) if include_abstract else ""
    return Paper(
        openalex_id=work.get("id"),
        title=work.get("display_name"),
        year=work.get("publication_year"),
        cited_by_count=work.get("cited_by_count") or 0,
        authors=authors,
        venue=venue,
        doi=work.get("doi"),
        abstract=abstract,
    )


def normalize_paper_id(paper_id: str) -> str:
    """
    Accept the ID forms an LLM might pass and return one OpenAlex understands:

        W2741809807 / https://openalex.org/W2741809807  ->  W2741809807
        10.1234/abc / https://doi.org/10.1234/abc       ->  https://doi.org/10.1234/abc
    """
    value = (paper_id or "").strip()
    if not value:
        raise ValueError("paper_id is empty")

    if "openalex.org/" in value.lower():
        value = value.rstrip("/").split("/")[-1]
    if value.lower().startswith("https://doi.org/"):
        value = value[len("https://doi.org/") :]
    if value.lower().startswith("10."):
        return f"https://doi.org/{value}"
    if re.fullmatch(r"[Ww]\d+", value):
        return value.upper()
    return value


def work_path(paper_id: str) -> str:
    """
    URL path for one work. DOIs use OpenAlex's `doi:` form so the URL stays clean:
        W123                     -> /works/W123
        https://doi.org/10.1/abc -> /works/doi:10.1/abc
    """
    normalized = normalize_paper_id(paper_id)
    if normalized.startswith("https://doi.org/"):
        normalized = "doi:" + normalized[len("https://doi.org/") :]
    return f"/works/{normalized}"


def short_id(openalex_url: str) -> str:
    """https://openalex.org/W123 -> W123"""
    return openalex_url.rstrip("/").rsplit("/", 1)[-1]


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class OpenAlexClient:
    """
    Async OpenAlex client. Use it as a context manager so connections get closed:

        async with create_openalex_client(settings) as client:
            papers = await client.search_works("graph neural networks", per_page=10)

    `transport` lets tests plug in a fake server (httpx.MockTransport): no network.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        cache: ResponseCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.openalex_base_url,
            timeout=settings.http_timeout_seconds,
            # known CA bundle (fixes SSL on some corporate setups)
            verify=ssl.create_default_context(cafile=certifi.where()),
            transport=transport,
        )
        self._limit = asyncio.Semaphore(settings.http_max_concurrency)
        self.cache = cache
        self.network_requests = 0  # real HTTP calls made (cache hits don't count)

    async def __aenter__(self) -> OpenAlexClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()
        if self.cache:
            self.cache.close()

    # --- low level: cache -> retries -> one request -----------------------------
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = dict(params or {})

        key = ResponseCache.make_key(path, query) if self.cache else None
        if key and (cached := self.cache.get(key)) is not None:
            return cached

        data = await self._get_with_retries(path, {**self._auth_params(), **query})

        if key:
            self.cache.set(key, data)
        return data

    def _auth_params(self) -> dict[str, str]:
        auth: dict[str, str] = {}
        if self._settings.openalex_api_key:
            auth["api_key"] = self._settings.openalex_api_key.get_secret_value()
        if self._settings.openalex_mailto:
            auth["mailto"] = self._settings.openalex_mailto
        return auth

    async def _get_with_retries(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Retry only TEMPORARY problems:
          - httpx.TransportError: timeout, connection reset, DNS hiccup
          - RetryableHTTPError:   5xx server error, short 429
        Waits grow exponentially (base, 2x, 4x, ...) plus random jitter so many
        clients don't all retry at the same instant.
        """
        base = self._settings.http_retry_wait_seconds
        retrying = AsyncRetrying(
            retry=retry_if_exception_type((httpx.TransportError, RetryableHTTPError)),
            stop=stop_after_attempt(self._settings.http_max_retries + 1),
            # `initial=` (tenacity 9.1). 9.2 renames it `multiplier=`, but google-genai
            # requires tenacity < 9.2, so we stay on the 9.1 name.
            wait=wait_exponential_jitter(initial=base, max=base * 16, jitter=base),
            reraise=True,  # after the last attempt, raise the real error
        )
        async for attempt in retrying:
            with attempt:
                return await self._get_once(path, params)
        raise AssertionError("unreachable")  # tenacity always returns or raises

    async def _get_once(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._limit:  # wait here if too many requests are in flight
            self.network_requests += 1
            response = await self._http.get(path, params=params)

        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            if retry_after is not None and retry_after <= MAX_SHORT_RETRY_AFTER_SECONDS:
                raise RetryableHTTPError(f"429, retry after {retry_after}s")
            raise OpenAlexRateLimitError(self._rate_limit_hint())
        if response.status_code >= 500:
            raise RetryableHTTPError(f"OpenAlex server error {response.status_code}")
        response.raise_for_status()  # other 4xx = our bug (bad filter etc.): don't retry
        return response.json()

    def _rate_limit_hint(self) -> str:
        if self._settings.openalex_api_key:
            return "OpenAlex rate limit hit (429). Your daily budget is used up; it resets daily."
        return (
            "OpenAlex rate limit hit (429). No OPENALEX_API_KEY set, so you share a tiny "
            "budget with your whole network. Get a free key at "
            "https://openalex.org/settings/api and add OPENALEX_API_KEY=... to .env"
        )

    # --- searches --------------------------------------------------------------
    async def search_works(
        self,
        query: str,
        *,
        per_page: int,
        sort: str = "relevance_score:desc",
        filters: str | None = None,
        scope: SearchScope = "all",
        include_abstract: bool = True,
    ) -> list[Paper]:
        """Keyword search over works, with optional sort, OpenAlex filter string and scope."""
        params: dict[str, Any] = {
            "per_page": per_page,
            "sort": sort,
            "select": FIELDS_WITH_ABSTRACT if include_abstract else BASE_FIELDS,
        }
        filter_parts = [filters] if filters else []
        if scope == "title_abstract":
            filter_parts.insert(0, title_abstract_filter(query))
        else:
            params["search"] = query
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        payload = await self._get("/works", params)
        return [
            parse_work(w, include_abstract=include_abstract) for w in payload.get("results", [])
        ]

    # --- single paper + citation graph ------------------------------------------
    async def get_work(self, paper_id: str) -> dict[str, Any]:
        """Raw payload for one work, incl. abstract and reference IDs (lookups are free)."""
        return await self._get(
            work_path(paper_id), {"select": FIELDS_WITH_ABSTRACT + ",referenced_works"}
        )

    async def get_citing_works(self, paper_id: str, limit: int) -> list[Paper]:
        """Papers that CITE this one (incoming), most cited first."""
        payload = await self._get(
            "/works",
            {
                "filter": f"cites:{normalize_paper_id(paper_id)}",
                "per_page": limit,
                "sort": "cited_by_count:desc",
                "select": BASE_FIELDS,
            },
        )
        return [parse_work(w, include_abstract=False) for w in payload.get("results", [])]

    async def get_referenced_works(self, paper_id: str, limit: int) -> list[Paper]:
        """Papers this one CITES (its bibliography / outgoing references)."""
        parent = await self._get(work_path(paper_id), {"select": "id,referenced_works"})
        ref_ids = [short_id(r) for r in (parent.get("referenced_works") or [])[:limit]]
        if not ref_ids:
            return []
        payload = await self._get(
            "/works",
            {
                "filter": f"openalex_id:{'|'.join(ref_ids)}",  # OR filter: W1|W2|W3
                "per_page": limit,
                "select": BASE_FIELDS,
            },
        )
        return [parse_work(w, include_abstract=False) for w in payload.get("results", [])]


def create_openalex_client(settings: Settings) -> OpenAlexClient:
    """Build a client with the on-disk cache if enabled in settings."""
    cache = (
        ResponseCache(
            settings.cache_dir / "openalex.sqlite",
            ttl_seconds=settings.cache_ttl_hours * 3600,
        )
        if settings.cache_enabled
        else None
    )
    return OpenAlexClient(settings, cache=cache)
