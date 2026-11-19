"""
Agent tools: async Python methods + the JSON schemas the LLM sees.

The model never sees this Python code — only TOOL_SCHEMAS.
Good descriptions = better tool selection and argument filling.

AgentTools holds the OpenAlex client, so every tool shares one connection
pool, one cache and one concurrency limit.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.models import Paper
from research_agent.sources.openalex import OpenAlexClient, normalize_paper_id, parse_work

MAX_TOOL_LIMIT = 10
ABSTRACT_CHARS_FOR_TOOLS = 1000


def _compact(paper: Paper) -> dict[str, Any]:
    """Tool results go into the LLM context, so keep them small."""
    data = paper.model_dump(exclude={"retrieval_reasons", "abstract"})
    if paper.abstract:
        data["abstract"] = paper.abstract[:ABSTRACT_CHARS_FOR_TOOLS]
    return data


class AgentTools:
    def __init__(self, client: OpenAlexClient, default_limit: int) -> None:
        self._client = client
        self._default_limit = default_limit

    def _clamp_limit(self, limit: int | None) -> int:
        value = limit if limit is not None else self._default_limit
        return max(1, min(int(value), MAX_TOOL_LIMIT))

    # --- tool implementations --------------------------------------------------
    async def search_papers(self, query: str, limit: int | None = None) -> dict[str, Any]:
        papers = await self._client.search_works(
            query, per_page=self._clamp_limit(limit), include_abstract=False
        )
        return {"query": query, "count": len(papers), "papers": [_compact(p) for p in papers]}

    async def get_paper(self, paper_id: str) -> dict[str, Any]:
        work = await self._client.get_work(paper_id)
        refs = work.get("referenced_works") or []
        return {
            **_compact(parse_work(work)),
            "reference_count": len(refs),
            "sample_reference_ids": refs[:5],  # lets the model follow up
        }

    async def get_citations(self, paper_id: str, limit: int | None = None) -> dict[str, Any]:
        papers = await self._client.get_citing_works(paper_id, self._clamp_limit(limit))
        return {
            "paper_id": normalize_paper_id(paper_id),
            "relation": "papers_that_cite_this_work",
            "count": len(papers),
            "papers": [_compact(p) for p in papers],
        }

    async def get_references(self, paper_id: str, limit: int | None = None) -> dict[str, Any]:
        papers = await self._client.get_referenced_works(paper_id, self._clamp_limit(limit))
        return {
            "paper_id": normalize_paper_id(paper_id),
            "relation": "papers_referenced_by_this_work",
            "count": len(papers),
            "papers": [_compact(p) for p in papers],
        }

    # --- dispatch ----------------------------------------------------------------
    async def execute(self, name: str, raw_args: str | None) -> str:
        """
        Parse the model's JSON arguments, run the matching tool, return JSON text.

        Errors are RETURNED (not raised) so the model can read them and retry.
        """
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid JSON arguments: {raw_args}"})

        # Only names declared in TOOL_SCHEMAS can be called — never arbitrary methods.
        if name not in TOOL_NAMES:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = await getattr(self, name)(**args)
        except TypeError as exc:
            return json.dumps({"error": f"Bad arguments for {name}: {exc}", "args": args})
        except Exception as exc:  # noqa: BLE001 - surface API errors to the model
            return json.dumps({"error": str(exc), "tool": name, "args": args})

        return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schemas (what the model sees)
# ---------------------------------------------------------------------------
def _schema(name: str, description: str, props: dict, required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_PAPER_ID = {"type": "string", "description": "OpenAlex ID (W...) or DOI."}


def _limit(what: str) -> dict[str, Any]:
    return {"type": "integer", "description": f"Max {what} (1-10). Default 5."}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "search_papers",
        "Search academic papers on OpenAlex by keyword query. "
        "Use this first when the user asks about a research topic.",
        {
            "query": {"type": "string", "description": "Short academic search query (keywords)."},
            "limit": _limit("papers to return"),
        },
        ["query"],
    ),
    _schema(
        "get_paper",
        "Get details for one paper, including abstract when available. "
        "Pass an OpenAlex work ID (e.g. W2741809807) or a DOI.",
        {"paper_id": _PAPER_ID},
        ["paper_id"],
    ),
    _schema(
        "get_citations",
        "List papers that CITE a given paper (incoming citations). "
        "Useful to see later work built on this paper.",
        {"paper_id": _PAPER_ID, "limit": _limit("citing papers")},
        ["paper_id"],
    ),
    _schema(
        "get_references",
        "List papers REFERENCED by a given paper (its bibliography). "
        "Useful to find foundational prior work.",
        {"paper_id": _PAPER_ID, "limit": _limit("referenced papers")},
        ["paper_id"],
    ),
]

TOOL_NAMES = frozenset(s["function"]["name"] for s in TOOL_SCHEMAS)
