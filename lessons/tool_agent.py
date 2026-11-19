"""
Stage 2 — Tool Calling (agent lesson)

Learning goals:
  - Tools            = Python functions the model is allowed to request
  - Tool schemas     = JSON descriptions the model reads (name, args, meaning)
  - Tool selection   = model chooses WHICH tool to call
  - Tool arguments   = model fills WHOSE args (query, paper_id, ...)
  - Tool results     = YOUR code runs the tool and returns data
  - Agent loop       = LLM ↔ tools ↔ LLM until a final answer

Flow:

    User
      ↓
    LLM
      ↓
    "I need to search papers"   ← model decides this (tool selection)
      ↓
    Tool Call                   ← model emits name + arguments
      ↓
    search_papers() / get_paper() / get_citations() / get_references()
      ↓
    Results                     ← Python executes OpenAlex calls
      ↓
    LLM                         ← sees tool results, may call again or answer

Difference from paper_search.py (Stage 1 style):
  There, YOUR Python hardcoded: query → OpenAlex → summarize.
  Here, the LLM chooses tools and arguments dynamically.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import certifi
import requests
from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Put it in your .env file.")

OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "learner@example.com")
# Free personal key → your own daily budget (see paper_search.py for details).
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
client = Groq(api_key=api_key)

# Keep the same model family you already use for learning.
MODEL = "openai/gpt-oss-20b"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# Safety rails for the agent loop (prevents infinite tool calling).
MAX_AGENT_STEPS = 8
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_GRAPH_LIMIT = 5


# ---------------------------------------------------------------------------
# OpenAlex helpers (shared by tools)
# ---------------------------------------------------------------------------
def openalex_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """One HTTP GET to OpenAlex with SSL bundle that works on corporate Windows."""
    merged = {"mailto": OPENALEX_MAILTO}
    if OPENALEX_API_KEY:
        merged["api_key"] = OPENALEX_API_KEY
    if params:
        merged.update(params)
    response = requests.get(url, params=merged, timeout=30, verify=certifi.where())
    if response.status_code == 429:
        # Raised as a normal error so the agent loop reports it back to the LLM.
        raise RuntimeError(
            "OpenAlex rate limit (429): daily budget used up. "
            "Set OPENALEX_API_KEY in .env (free at https://openalex.org/settings/api)."
        )
    response.raise_for_status()
    return response.json()


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            positioned.append((pos, word))
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


def normalize_paper_id(paper_id: str) -> str:
    """
    Accept several ID forms the model might pass:

      W2741809807
      https://openalex.org/W2741809807
      https://doi.org/10.1234/abc
      10.1234/abc
    """
    value = (paper_id or "").strip()
    if not value:
        raise ValueError("paper_id is empty")

    # OpenAlex URL → W...
    if "openalex.org/" in value.lower():
        value = value.rstrip("/").split("/")[-1]

    # DOI URL → bare DOI
    if value.lower().startswith("https://doi.org/"):
        value = value[len("https://doi.org/") :]

    # Bare DOI → OpenAlex DOI filter form
    if value.lower().startswith("10."):
        return f"https://doi.org/{value}"

    # Bare OpenAlex work id
    if re.fullmatch(r"[Ww]\d+", value):
        return value.upper() if value[0] == "w" else value

    return value


def compact_work(work: dict[str, Any], *, include_abstract: bool = False) -> dict[str, Any]:
    """Shrink OpenAlex payloads so tool results fit in the LLM context."""
    authors = [
        a.get("author", {}).get("display_name", "Unknown")
        for a in (work.get("authorships") or [])[:5]
        if isinstance(a, dict)
    ]
    venue = ((work.get("primary_location") or {}).get("source") or {}).get(
        "display_name"
    )
    paper = {
        "openalex_id": work.get("id"),
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count") or 0,
        "authors": authors,
        "venue": venue,
        "doi": work.get("doi"),
    }
    if include_abstract:
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        paper["abstract"] = abstract[:1000] if abstract else ""
    return paper


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# These are normal Python functions. The LLM cannot run them directly.
# Your agent loop calls them when the model requests a tool.
# ---------------------------------------------------------------------------
def search_papers(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[str, Any]:
    """Search OpenAlex for works matching a keyword query."""
    limit = max(1, min(int(limit), 10))
    payload = openalex_get(
        OPENALEX_WORKS_URL,
        {
            "search": query,
            "per_page": limit,
            "sort": "relevance_score:desc",
            "select": (
                "id,doi,display_name,publication_year,cited_by_count,"
                "authorships,primary_location"
            ),
        },
    )
    papers = [compact_work(w) for w in payload.get("results", [])]
    return {
        "query": query,
        "count": len(papers),
        "papers": papers,
    }


def get_paper(paper_id: str) -> dict[str, Any]:
    """Fetch one paper by OpenAlex ID or DOI."""
    normalized = normalize_paper_id(paper_id)
    work = openalex_get(
        f"{OPENALEX_WORKS_URL}/{normalized}",
        {
            "select": (
                "id,doi,display_name,publication_year,cited_by_count,"
                "authorships,primary_location,abstract_inverted_index,"
                "referenced_works"
            ),
        },
    )
    paper = compact_work(work, include_abstract=True)
    refs = work.get("referenced_works") or []
    paper["reference_count"] = len(refs)
    # Keep a few reference IDs so the model can follow up with get_paper / get_references.
    paper["sample_reference_ids"] = refs[:5]
    return paper


def get_citations(paper_id: str, limit: int = DEFAULT_GRAPH_LIMIT) -> dict[str, Any]:
    """
    Papers that CITE this paper (incoming citations).

    OpenAlex filter: cites:<id>  →  works that cite the given work.
    """
    normalized = normalize_paper_id(paper_id)
    limit = max(1, min(int(limit), 10))
    payload = openalex_get(
        OPENALEX_WORKS_URL,
        {
            "filter": f"cites:{normalized}",
            "per_page": limit,
            "sort": "cited_by_count:desc",
            "select": (
                "id,doi,display_name,publication_year,cited_by_count,"
                "authorships,primary_location"
            ),
        },
    )
    papers = [compact_work(w) for w in payload.get("results", [])]
    return {
        "paper_id": normalized,
        "relation": "papers_that_cite_this_work",
        "count": len(papers),
        "papers": papers,
    }


def get_references(paper_id: str, limit: int = DEFAULT_GRAPH_LIMIT) -> dict[str, Any]:
    """
    Papers this work cites (its bibliography / outgoing references).

    OpenAlex stores these on the work as `referenced_works` (list of IDs).
    We fetch the parent paper, then hydrate up to `limit` referenced works.
    """
    normalized = normalize_paper_id(paper_id)
    limit = max(1, min(int(limit), 10))

    parent = openalex_get(
        f"{OPENALEX_WORKS_URL}/{normalized}",
        {"select": "id,referenced_works"},
    )
    ref_ids = (parent.get("referenced_works") or [])[:limit]
    if not ref_ids:
        return {
            "paper_id": normalized,
            "relation": "papers_referenced_by_this_work",
            "count": 0,
            "papers": [],
        }

    # OpenAlex OR filter: openalex_id:W1|W2|W3
    id_filter = "|".join(ref_ids)
    payload = openalex_get(
        OPENALEX_WORKS_URL,
        {
            "filter": f"openalex_id:{id_filter}",
            "per_page": limit,
            "select": (
                "id,doi,display_name,publication_year,cited_by_count,"
                "authorships,primary_location"
            ),
        },
    )
    papers = [compact_work(w) for w in payload.get("results", [])]
    return {
        "paper_id": normalized,
        "relation": "papers_referenced_by_this_work",
        "count": len(papers),
        "papers": papers,
    }


# Map tool NAME (string from the model) → Python function.
# This is "tool dispatch": the bridge between LLM request and real code.
TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "search_papers": search_papers,
    "get_paper": get_paper,
    "get_citations": get_citations,
    "get_references": get_references,
}


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# The model never sees your Python source. It only sees these JSON schemas.
# Good descriptions = better tool selection and better argument filling.
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search academic papers on OpenAlex by keyword query. "
                "Use this first when the user asks about a research topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short academic search query (keywords).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max papers to return (1-10). Default 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper",
            "description": (
                "Get details for one paper, including abstract when available. "
                "Pass an OpenAlex work ID (e.g. W2741809807) or a DOI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "OpenAlex ID (W...) or DOI.",
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_citations",
            "description": (
                "List papers that CITE a given paper (incoming citations). "
                "Useful to see later work built on this paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "OpenAlex ID (W...) or DOI.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max citing papers (1-10). Default 5.",
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_references",
            "description": (
                "List papers REFERENCED by a given paper (its bibliography). "
                "Useful to find foundational prior work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "OpenAlex ID (W...) or DOI.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max referenced papers (1-10). Default 5.",
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
]


SYSTEM_PROMPT = """
You are a research assistant with tools for OpenAlex academic papers.

Available tools:
- search_papers(query, limit?): find papers by keywords
- get_paper(paper_id): details + abstract for one paper
- get_citations(paper_id, limit?): later papers that cite it
- get_references(paper_id, limit?): earlier papers it cites

How to work:
1. Decide whether you need a tool.
2. Prefer search_papers first for topic questions.
3. Use returned openalex_id values for follow-up tools.
4. You may call multiple tools across turns.
5. When you have enough evidence, stop calling tools and answer clearly.
6. Ground claims in tool results. Do not invent papers or DOIs.
7. Mention paper titles and years when useful.
""".strip()


# ---------------------------------------------------------------------------
# Execute one tool call from the model
# ---------------------------------------------------------------------------
def execute_tool_call(tool_call: Any) -> str:
    """
    tool_call comes from the LLM response.

    It typically looks like:
      tool_call.function.name       = "search_papers"
      tool_call.function.arguments  = '{"query": "transformers survey", "limit": 5}'

    Your job:
      1) parse arguments JSON   ← Tool arguments
      2) pick the Python fn     ← Tool selection (already done by model; you dispatch)
      3) run it                 ← Tool execution
      4) return string result   ← Tool results (must be text for the next LLM turn)
    """
    name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON arguments: {raw_args}"})

    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = TOOL_FUNCTIONS[name](**args)
    except TypeError as exc:
        # Wrong / missing arguments — send error back so the model can retry.
        return json.dumps({"error": f"Bad arguments for {name}: {exc}", "args": args})
    except Exception as exc:  # noqa: BLE001 - educational agent should surface API errors
        return json.dumps({"error": str(exc), "tool": name, "args": args})

    return json.dumps(result, ensure_ascii=False)


def assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """
    Convert the SDK message object into a plain dict for the next request.

    When the model requests tools, we MUST echo back the assistant message
    including tool_calls, then append role=tool messages that match each id.
    """
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return payload


# ---------------------------------------------------------------------------
# AGENT LOOP
# ---------------------------------------------------------------------------
def run_tool_agent(user_question: str) -> str:
    """
    The agent loop:

      while True:
          LLM sees messages + tool schemas
          if LLM returns tool_calls:
              run tools, append results, continue
          else:
              return final text answer

    That loop is what people mean by a simple "agent".
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for step in range(1, MAX_AGENT_STEPS + 1):
        print(f"\n--- Agent step {step}/{MAX_AGENT_STEPS} ---")
        print("LLM is thinking (may select a tool)...")

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,          # ← Tool schemas given to the model
            tool_choice="auto",   # ← model may call a tool OR answer directly
            temperature=0.2,
        )

        message = completion.choices[0].message
        finish_reason = completion.choices[0].finish_reason

        # Case 1: model wants one or more tools.
        if message.tool_calls:
            print(f"Tool selection: {len(message.tool_calls)} call(s) requested")
            # Keep the assistant's tool request in history.
            messages.append(assistant_message_to_dict(message))

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments
                print(f"  → calling {name}")
                print(f"    arguments = {args}")

                tool_result = execute_tool_call(tool_call)

                # Show a short preview so you can see "tool results" live.
                preview = tool_result if len(tool_result) < 400 else tool_result[:400] + "..."
                print(f"    result preview = {preview}")

                # Role "tool" feeds the result back to the model.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": tool_result,
                    }
                )

            # Loop again so the LLM can read results and decide next action.
            continue

        # Case 2: no tool call → final answer.
        final_text = message.content or ""
        print(f"No tool call (finish_reason={finish_reason}). Final answer ready.")
        messages.append({"role": "assistant", "content": final_text})
        return final_text

    return (
        "Stopped: reached MAX_AGENT_STEPS without a final answer. "
        "Try a narrower question, or raise MAX_AGENT_STEPS."
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 2 — Tool Calling Agent")
    print("=" * 60)
    print(
        "\nConcepts you will see in the terminal:\n"
        "  Tools / schemas / selection / arguments / results / agent loop\n"
    )

    question = input("Enter your research question: ").strip()
    if not question:
        raise SystemExit("No question entered. Exiting.")

    answer = run_tool_agent(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
