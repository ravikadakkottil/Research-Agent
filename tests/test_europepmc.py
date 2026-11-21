"""Europe PMC: parsing, client behaviour, merging into OpenAlex records, fail-soft step."""

import asyncio
from pathlib import Path

import httpx

from research_agent.cache import ResponseCache
from research_agent.config import Settings
from research_agent.models import Paper, QueryPlan
from research_agent.retrieval.second_source import fetch_europepmc, merge_with_openalex
from research_agent.retrieval.selection import deduplicate_papers
from research_agent.sources.europepmc import (
    EuropePMCClient,
    clean_query,
    parse_result,
    strip_markup,
)

PRIME = {
    "title": "Search-and-replace genome editing without double-strand breaks or donor DNA.",
    "pubYear": "2019",
    "citedByCount": 3822,
    "doi": "10.1038/S41586-019-1711-4",
    "authorString": "Anzalone AV, Randolph PB, Davis JR.",
    "journalInfo": {"journal": {"title": "Nature"}},
    "abstractText": "Most variants that contribute to disease<sup>1</sup> are <i>hard</i> to fix.",
}
NO_DOI = {"title": "A paper without DOI", "pubYear": "2020", "citedByCount": "7"}


def settings(**overrides) -> Settings:
    defaults = {
        "groq_api_key": "x",
        "http_retry_wait_seconds": 0,
        "cache_enabled": False,
        "europepmc_enabled": True,
    }
    return Settings(_env_file=None, **{**defaults, **overrides})


def payload(*results) -> dict:
    return {"resultList": {"result": list(results)}}


# --- pure helpers -------------------------------------------------------------------
def test_clean_query_removes_syntax_and_neutralises_operators():
    assert clean_query('CRISPR: "Cas9" (editing) AND NOT x') == "CRISPR Cas9 editing and not x"


def test_strip_markup_drops_footnotes_and_tags():
    assert (
        strip_markup(PRIME["abstractText"])
        == "Most variants that contribute to disease are hard to fix."
    )


def test_parse_result_maps_fields_to_openalex_style():
    paper = parse_result(PRIME)
    assert (
        paper.title == "Search-and-replace genome editing without double-strand breaks or donor DNA"
    )
    assert paper.doi == "https://doi.org/10.1038/s41586-019-1711-4"  # OpenAlex's DOI form
    assert (paper.year, paper.cited_by_count, paper.venue) == (2019, 3822, "Nature")
    assert paper.authors == ["Anzalone AV", "Randolph PB", "Davis JR"]
    assert paper.europepmc_only is True
    assert parse_result(NO_DOI).doi is None


# --- client -------------------------------------------------------------------------
class FakeServer:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)


def search(server, *, cache=None, times=1):
    client = EuropePMCClient(settings(), cache=cache, transport=httpx.MockTransport(server))

    async def go():
        async with client:
            return [
                await client.search("CRISPR-Cas9 genome editing", page_size=25)
                for _ in range(times)
            ]

    return asyncio.run(go())


def test_search_sends_citation_sorted_core_query():
    server = FakeServer(httpx.Response(200, json=payload(PRIME, {"title": ""})))
    [papers] = search(server)
    params = server.requests[0].url.params
    assert params["sort"] == "CITED desc" and params["resultType"] == "core"
    assert params["pageSize"] == "25"
    assert len(papers) == 1  # untitled result dropped


def test_search_retries_one_server_error():
    server = FakeServer(httpx.Response(503), httpx.Response(200, json=payload(PRIME)))
    [papers] = search(server)
    assert len(server.requests) == 2 and len(papers) == 1


def test_search_uses_cache(tmp_path: Path):
    server = FakeServer(httpx.Response(200, json=payload(PRIME)))
    cache = ResponseCache(tmp_path / "epmc.sqlite", ttl_seconds=60)
    first, second = search(server, cache=cache, times=2)
    assert first == second and len(server.requests) == 1


# --- merging into OpenAlex ----------------------------------------------------------
OPENALEX_TWIN = Paper(
    openalex_id="https://openalex.org/W2981137429",
    doi="https://doi.org/10.1038/s41586-019-1711-4",
    title="Search-and-replace genome editing without double-strand breaks or donor DNA",
    cited_by_count=5455,
    abstract="",  # OpenAlex has no abstract for this paper: the reason it was missed
    referenced_works=["W1", "W2"],
)


def test_merge_prefers_openalex_record_and_fills_missing_abstract():
    epmc = [parse_result(PRIME), parse_result(NO_DOI)]
    merged = merge_with_openalex(epmc, [OPENALEX_TWIN])

    prime, no_doi = merged
    assert prime.openalex_id == OPENALEX_TWIN.openalex_id
    assert prime.cited_by_count == 5455  # OpenAlex counts, consistent with other candidates
    assert prime.referenced_works == ["W1", "W2"]  # usable as an expansion seed
    assert prime.abstract.startswith("Most variants")  # taken from Europe PMC
    assert prime.europepmc_only is True
    assert no_doi.title == "A paper without DOI"  # kept as a Europe PMC record
    assert OPENALEX_TWIN.abstract == ""  # input not mutated


def test_dedupe_clears_europepmc_only_and_shares_abstract():
    from_openalex = OPENALEX_TWIN
    from_epmc = merge_with_openalex([parse_result(PRIME)], [OPENALEX_TWIN])[0]
    [paper] = deduplicate_papers({"relevant": [from_openalex], "highly_cited": [from_epmc]})
    assert paper.retrieval_reasons == ["relevant", "highly_cited"]
    assert paper.europepmc_only is False  # an OpenAlex search found it too
    assert paper.abstract.startswith("Most variants")


# --- the fail-soft step -------------------------------------------------------------
class FakeEPMC:
    def __init__(self, papers=None, error: Exception | None = None, delay: float = 0) -> None:
        self.papers, self.error, self.delay = papers or [], error, delay

    async def search(self, query, *, page_size):
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.papers


class FakeOpenAlex:
    def __init__(self) -> None:
        self.dois: list[str] = []

    async def get_works_by_dois(self, dois):
        self.dois = dois
        return [OPENALEX_TWIN]


BIO = QueryPlan(search_query="CRISPR-Cas9 genome editing", biomedical=True)


def run_step(epmc, plan=BIO, **overrides):
    return asyncio.run(fetch_europepmc(epmc, FakeOpenAlex(), plan, settings(**overrides)))


def test_step_merges_and_reports_ok():
    papers, status = run_step(FakeEPMC([parse_result(PRIME)]))
    assert status == "ok (1 papers)"
    assert papers[0].openalex_id == OPENALEX_TWIN.openalex_id


def test_step_skips_non_biomedical_and_disabled():
    plan = QueryPlan(search_query="tool use language models", biomedical=False)
    assert run_step(FakeEPMC([parse_result(PRIME)]), plan) == (
        [],
        "skipped (not a biomedical topic)",
    )
    assert run_step(None) == ([], "off")


def test_step_never_raises():
    assert run_step(FakeEPMC(error=httpx.ConnectError("down"))) == ([], "failed (ConnectError)")
    slow = FakeEPMC([parse_result(PRIME)], delay=1)
    assert run_step(slow, europepmc_timeout_seconds=0.01) == ([], "failed (timed out)")
