"""
Europe PMC: a second paper source for biomedical questions.

Why a second source (Step 5c):
  OpenAlex has NO abstract for some publishers' papers (e.g. Nature journals).
  Its title+abstract search then sees only the title, and a landmark whose
  title doesn't name the topic is never found:
      "Search-and-replace genome editing without double-strand breaks..."  (prime editing)
      "Tumor mutational load predicts survival after immunotherapy..."     (TMB)
  Europe PMC indexes PubMed/PMC abstracts AND full text, so a citation-sorted
  search finds both.

Why biomedical questions only:
  Europe PMC covers life sciences. For "tool use in language models" it returns
  gene-expression statistics papers: noise to embed and filter.

Why semantic scholar isn't used: without an API key its shared rate limit
rejected 9 of 12 probe requests (2026-11-22).

Europe PMC is slow and variable (1-30 s per query), so the pipeline treats it
as a BONUS: a hard time limit, and any failure just means "OpenAlex only".
"""

from __future__ import annotations

import re
import ssl
from typing import Any

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

# Footnote markers ("disease<sup>1</sup>") would glue numbers onto words.
_SUPERSCRIPT = re.compile(r"<sup>.*?</sup>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
# Characters with meaning in Europe PMC's query syntax (fields, groups, phrases).
_QUERY_SPECIAL = re.compile(r'[:()"\[\]{}]+')
_BOOLEAN_WORDS = {"AND", "OR", "NOT"}


class EuropePMCServerError(RuntimeError):
    """5xx or 429: worth one retry."""


# ---------------------------------------------------------------------------
# Pure helpers (no network)
# ---------------------------------------------------------------------------
def clean_query(query: str) -> str:
    """Plain keywords only: strip syntax characters, lowercase AND/OR/NOT (operators)."""
    words = _QUERY_SPECIAL.sub(" ", query).split()
    return " ".join(w.lower() if w in _BOOLEAN_WORDS else w for w in words)


def strip_markup(text: str | None) -> str:
    """Europe PMC abstracts and titles contain HTML (<i>, <sup>, ...)."""
    text = _SUPERSCRIPT.sub("", text or "")
    return " ".join(_TAG.sub("", text).split())


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_result(result: dict[str, Any]) -> Paper:
    """One Europe PMC search result ("core" result type) -> Paper."""
    doi = (result.get("doi") or "").strip().lower()
    journal = ((result.get("journalInfo") or {}).get("journal") or {}).get("title")
    authors = [
        a.strip().rstrip(".") for a in (result.get("authorString") or "").split(",") if a.strip()
    ][:5]
    return Paper(
        title=strip_markup(result.get("title")).rstrip(".") or None,
        year=_int_or_none(result.get("pubYear")),
        cited_by_count=_int_or_none(result.get("citedByCount")) or 0,
        authors=authors,
        venue=journal or result.get("journalTitle"),
        doi=f"https://doi.org/{doi}" if doi else None,  # OpenAlex's DOI form
        abstract=strip_markup(result.get("abstractText")),
        europepmc_only=True,
    )


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class EuropePMCClient:
    """
    Async Europe PMC search client (no API key needed).

    `transport` lets tests plug in a fake server (httpx.MockTransport).
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
            base_url=settings.europepmc_base_url,
            timeout=settings.europepmc_timeout_seconds,
            verify=ssl.create_default_context(cafile=certifi.where()),
            headers={"User-Agent": "research-agent"},
            transport=transport,
        )
        self.cache = cache
        self.network_requests = 0

    async def __aenter__(self) -> EuropePMCClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()
        if self.cache:
            self.cache.close()

    async def search(self, query: str, *, page_size: int, sort: str = "CITED desc") -> list[Paper]:
        """
        Search titles, abstracts AND full text; most cited first by default.
        resultType=core includes abstracts (and was no slower than "lite" in probes).
        """
        params = {
            "query": clean_query(query),
            "format": "json",
            "resultType": "core",
            "pageSize": page_size,
            "sort": sort,
        }
        key = ResponseCache.make_key("/search", params) if self.cache else None
        if key and (cached := self.cache.get(key)) is not None:
            payload = cached
        else:
            payload = await self._get_with_retry(params)
            if key:
                self.cache.set(key, payload)
        results = (payload.get("resultList") or {}).get("result") or []
        return [parse_result(r) for r in results if r.get("title")]

    async def _get_with_retry(self, params: dict[str, Any]) -> dict[str, Any]:
        # One retry for server errors only. Timeouts are NOT retried: a slow
        # Europe PMC would double the wait, and the caller has a hard limit anyway.
        base = self._settings.http_retry_wait_seconds
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(EuropePMCServerError),
            stop=stop_after_attempt(2),
            wait=wait_exponential_jitter(initial=base, max=base * 4, jitter=base),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                self.network_requests += 1
                response = await self._http.get("/search", params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    raise EuropePMCServerError(f"Europe PMC returned {response.status_code}")
                response.raise_for_status()
                return response.json()
        raise AssertionError("unreachable")


def create_europepmc_client(settings: Settings) -> EuropePMCClient | None:
    """None when disabled in settings. Uses its own on-disk cache file."""
    if not settings.europepmc_enabled:
        return None
    cache = (
        ResponseCache(
            settings.cache_dir / "europepmc.sqlite",
            ttl_seconds=settings.cache_ttl_hours * 3600,
        )
        if settings.cache_enabled
        else None
    )
    return EuropePMCClient(settings, cache=cache)
