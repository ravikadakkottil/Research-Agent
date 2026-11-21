"""Agent grounding: tools remember what they returned; cited IDs are checked against it."""

import asyncio

from research_agent.agent.grounding import check_citations, cited_ids
from research_agent.agent.tools import AgentTools
from research_agent.models import Paper

GAT = Paper(openalex_id="https://openalex.org/W2766453196", title="Graph Attention Networks")
HAN = Paper(
    openalex_id="https://openalex.org/W2911286998",
    title="Heterogeneous Graph Attention Network",
    abstract="x" * 2000,
)


def test_cited_ids_in_order_without_duplicates():
    answer = "GAT [W2766453196] and HAN (https://openalex.org/W2911286998); again [W2766453196]."
    assert cited_ids(answer) == ["W2766453196", "W2911286998"]
    assert cited_ids("no ids, not even W12 or WORD") == []


def test_check_citations_splits_verified_and_unverified():
    seen = {"W2766453196": GAT}
    sources, unverified = check_citations("GAT [W2766453196], made up [W1234567]", seen)
    assert sources == [GAT]
    assert unverified == ["W1234567"]


class FakeOpenAlex:
    async def search_works(self, query, *, per_page):
        return [GAT, HAN]

    async def get_work(self, paper_id):
        return {
            "id": "https://openalex.org/W42",
            "display_name": "Looked up",
            "referenced_works": [],
        }

    async def get_citing_works(self, paper_id, limit):
        return []

    async def get_referenced_works(self, paper_id, limit):
        return []


def test_tools_remember_results_and_show_short_ids_and_abstracts():
    tools = AgentTools(FakeOpenAlex(), default_limit=5)

    async def go():
        search = await tools.search_papers("graph attention")
        await tools.get_paper("W42")
        return search

    search = asyncio.run(go())

    assert set(tools.seen) == {"W2766453196", "W2911286998", "W42"}
    first, second = search["papers"]
    assert first["openalex_id"] == "W2766453196"  # the exact form the agent must cite
    assert len(second["abstract"]) == 300  # short abstract: what the paper is ABOUT
    assert "similarity" not in first  # pipeline-only fields stay out of the LLM context
