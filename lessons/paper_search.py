"""
Paper-search GenAI pipeline (lesson 2):

    User
      ↓
    LLM #1 --------------->  generate a short academic search query (JSON)
      ↓
    OpenAlex --------------->  multiple retrieval buckets
      ↓
    Python ---------------->  deduplicate + balanced selection
      ↓
    LLM #2 --------------->  research synthesis (JSON)
      ↓
    Final answer for the user

Why multiple retrieval buckets?
  Relevance alone can miss foundational or very new work.
  A good research assistant balances:
    - relevance
    - recency
    - foundational / highly cited importance
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

import certifi
import requests
from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------------------------
# Setup: keys, clients, model
# ---------------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Put it in your .env file."
    )

# Optional but recommended by OpenAlex ("polite pool"):
# send an email so they know who is calling the free API.
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "learner@example.com")

# OpenAlex now meters usage with a daily budget.
# Without a key, all requests from your network's IP share one small free
# budget (easily used up on an office network → 429 errors).
# A free personal key (https://openalex.org/settings/api) gives you your own.
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")

client = Groq(api_key=api_key)

# Same idea as lesson 1. Change this string if you want another Groq model.
MODEL = "openai/gpt-oss-20b"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


# ---------------------------------------------------------------------------
# Configurable retrieval settings (easy to tweak while learning)
# ---------------------------------------------------------------------------
# How many papers to pull from each OpenAlex perspective.
RELEVANT_COUNT = 10
RECENT_COUNT = 10
HIGHLY_CITED_COUNT = 10

# "Last N years" window for the recent bucket.
# Example: if today is 2026 and RECENT_YEARS=3 → from 2024 onward.
RECENT_YEARS = 3

# Final set size sent to the summarizer LLM.
TARGET_FINAL_PAPERS = 12

# Soft targets for a balanced final mix.
# Remaining slots prefer papers that appear in multiple buckets.
SELECT_RELEVANT = 4
SELECT_RECENT = 3
SELECT_HIGHLY_CITED = 3


# ---------------------------------------------------------------------------
# Small helper: call the LLM and parse JSON
# ---------------------------------------------------------------------------
def ask_llm_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """
    One LLM call that MUST return JSON.

    Python string  ->  Groq API  ->  JSON string  ->  Python dict
    """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = completion.choices[0].message.content
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Step A: User question  ->  LLM  ->  search query
# ---------------------------------------------------------------------------
QUERY_SYSTEM_PROMPT = """
You turn a user's research question into a short OpenAlex search query.

Return ONLY JSON with this shape:
{
  "search_query": string,   // 3 to 8 academic keywords, no punctuation fluff
  "reason": string          // one short sentence explaining the query
}

Rules:
- Prefer technical keywords over conversational words.
- Do not include years unless the user asked for a time range.
- Do not wrap the query in quotes.
""".strip()


def generate_search_query(user_question: str) -> dict[str, Any]:
    """LLM #1: rewrite the human question into search keywords."""
    return ask_llm_json(QUERY_SYSTEM_PROMPT, user_question)


# ---------------------------------------------------------------------------
# Step B helpers: parse OpenAlex works into clean paper dicts
# ---------------------------------------------------------------------------
def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    OpenAlex often stores abstracts as an "inverted index":

        {"machine": [0, 5], "learning": [1], ...}

    That means: word -> list of positions.
    We rebuild the readable sentence for the summarizer LLM.
    """
    if not inverted_index:
        return ""

    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            positioned.append((pos, word))

    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


def parse_openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAlex work object into our simpler paper schema."""
    authors = [
        a.get("author", {}).get("display_name", "Unknown")
        for a in work.get("authorships", [])[:5]
        if isinstance(a, dict)
    ]
    venue = (
        (work.get("primary_location") or {})
        .get("source", {})
        or {}
    ).get("display_name")

    return {
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count") or 0,
        "authors": authors,
        "venue": venue,
        "doi": work.get("doi"),
        "openalex_id": work.get("id"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        # Filled later when we know which bucket(s) retrieved this paper.
        "retrieval_reasons": [],
    }


def search_openalex(
    search_query: str,
    *,
    per_page: int,
    sort: str,
    extra_filters: str | None = None,
) -> list[dict[str, Any]]:
    """
    Call OpenAlex /works with configurable sort / filter.

    This is NOT an LLM. It is a normal HTTP GET to a database API.
    """
    params: dict[str, Any] = {
        "search": search_query,
        "per_page": per_page,
        "sort": sort,
        # Select only fields we need (smaller, faster response).
        "select": (
            "id,doi,display_name,publication_year,cited_by_count,"
            "authorships,abstract_inverted_index,primary_location"
        ),
        "mailto": OPENALEX_MAILTO,
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    if extra_filters:
        params["filter"] = extra_filters

    # verify=certifi.where() uses a known CA bundle. On some Windows /
    # corporate setups, the default SSL path fails; certifi fixes that.
    response = requests.get(
        OPENALEX_WORKS_URL,
        params=params,
        timeout=30,
        verify=certifi.where(),
    )
    # 429 = "Too Many Requests": the daily budget is used up.
    if response.status_code == 429:
        hint = (
            "Your personal daily budget is used up; it resets daily."
            if OPENALEX_API_KEY
            else "No OPENALEX_API_KEY set, so you share a tiny budget with your whole "
            "network. Get a free key at https://openalex.org/settings/api and add "
            "OPENALEX_API_KEY=... to .env"
        )
        raise SystemExit(f"\nOpenAlex rate limit hit (429). {hint}")
    response.raise_for_status()
    payload = response.json()

    return [parse_openalex_work(work) for work in payload.get("results", [])]


def recent_from_year(recent_years: int = RECENT_YEARS) -> int:
    """
    Compute the first year that counts as "recent".

    Example: year=2026, recent_years=3 → 2024
    (covers 2024, 2025, 2026).
    """
    current_year = datetime.now().year
    return current_year - recent_years + 1


# ---------------------------------------------------------------------------
# Step B: multi-bucket retrieval
# ---------------------------------------------------------------------------
def retrieve_relevant_papers(search_query: str) -> list[dict[str, Any]]:
    """Bucket A: strongest keyword / topical match."""
    return search_openalex(
        search_query,
        per_page=RELEVANT_COUNT,
        sort="relevance_score:desc",
    )


def retrieve_recent_papers(search_query: str) -> list[dict[str, Any]]:
    """
    Bucket B: papers from the last RECENT_YEARS.

    Recency is decided by OpenAlex's publication-date filter — not by the LLM.
    """
    start_year = recent_from_year()
    # from_publication_date is an OpenAlex date filter (YYYY-MM-DD).
    return search_openalex(
        search_query,
        per_page=RECENT_COUNT,
        sort="publication_date:desc",
        extra_filters=f"from_publication_date:{start_year}-01-01",
    )


def retrieve_highly_cited_papers(search_query: str) -> list[dict[str, Any]]:
    """
    Bucket C: influential / foundational work.

    We still pass `search=` so results stay on-topic, but sort by citations
    so older landmark papers can surface even if they are not the newest.
    """
    return search_openalex(
        search_query,
        per_page=HIGHLY_CITED_COUNT,
        sort="cited_by_count:desc",
    )


def retrieve_paper_buckets(search_query: str) -> dict[str, list[dict[str, Any]]]:
    """Run all three retrieval strategies and return them labeled."""
    return {
        "relevant": retrieve_relevant_papers(search_query),
        "recent": retrieve_recent_papers(search_query),
        "highly_cited": retrieve_highly_cited_papers(search_query),
    }


# ---------------------------------------------------------------------------
# Step C: deduplicate across buckets
# ---------------------------------------------------------------------------
def normalize_title(title: str | None) -> str:
    """Fallback identity key when DOI / OpenAlex ID are missing."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def paper_identity(paper: dict[str, Any]) -> str:
    """
    Prefer stable IDs for deduplication:

      1) DOI
      2) OpenAlex ID
      3) normalized title
    """
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"

    openalex_id = (paper.get("openalex_id") or "").strip().lower()
    if openalex_id:
        return f"openalex:{openalex_id}"

    title_key = normalize_title(paper.get("title"))
    if title_key:
        return f"title:{title_key}"

    # Last resort: object identity proxy (should be rare).
    return f"fallback:{id(paper)}"


def deduplicate_papers(
    buckets: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Merge papers from all buckets.

    If the same paper appears in multiple buckets, keep one copy and
    accumulate every retrieval reason (relevant / recent / highly_cited).
    """
    merged: dict[str, dict[str, Any]] = {}

    for reason, papers in buckets.items():
        for paper in papers:
            key = paper_identity(paper)
            if key not in merged:
                # Copy so we do not mutate the bucket lists in place oddly.
                merged[key] = {**paper, "retrieval_reasons": [reason]}
            else:
                reasons = merged[key]["retrieval_reasons"]
                if reason not in reasons:
                    reasons.append(reason)

    return list(merged.values())


# ---------------------------------------------------------------------------
# Step D: deterministic balanced selection (no LLM ranking yet)
# ---------------------------------------------------------------------------
def _sort_relevant(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Prefer papers that also hit other buckets, then higher citations, then newer.
    return sorted(
        papers,
        key=lambda p: (
            len(p.get("retrieval_reasons", [])),
            p.get("cited_by_count") or 0,
            p.get("year") or 0,
        ),
        reverse=True,
    )


def _sort_recent(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        papers,
        key=lambda p: (
            p.get("year") or 0,
            p.get("cited_by_count") or 0,
            len(p.get("retrieval_reasons", [])),
        ),
        reverse=True,
    )


def _sort_highly_cited(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        papers,
        key=lambda p: (
            p.get("cited_by_count") or 0,
            len(p.get("retrieval_reasons", [])),
            p.get("year") or 0,
        ),
        reverse=True,
    )


def _sort_multi_criteria(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill leftover slots with papers that satisfy multiple perspectives."""
    return sorted(
        papers,
        key=lambda p: (
            len(p.get("retrieval_reasons", [])),
            p.get("cited_by_count") or 0,
            p.get("year") or 0,
        ),
        reverse=True,
    )


def select_balanced_papers(
    candidates: list[dict[str, Any]],
    target_count: int = TARGET_FINAL_PAPERS,
) -> list[dict[str, Any]]:
    """
    Build a transparent final set without another LLM.

    Strategy (soft targets — never force exact counts if papers are missing):
      1) take up to SELECT_RELEVANT papers tagged "relevant"
      2) take up to SELECT_RECENT papers tagged "recent" (not already chosen)
      3) take up to SELECT_HIGHLY_CITED papers tagged "highly_cited"
      4) fill remaining slots with multi-bucket / strongest leftover papers

    Why this exists:
      Retrieval finds ~30 raw hits. The summarizer works better with a
      smaller, balanced reading list than with every candidate.
    """
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    def try_add(paper: dict[str, Any]) -> bool:
        if len(selected) >= target_count:
            return False
        key = paper_identity(paper)
        if key in selected_keys:
            return False
        selected.append(paper)
        selected_keys.add(key)
        return True

    relevant = _sort_relevant(
        [p for p in candidates if "relevant" in p.get("retrieval_reasons", [])]
    )
    recent = _sort_recent(
        [p for p in candidates if "recent" in p.get("retrieval_reasons", [])]
    )
    highly_cited = _sort_highly_cited(
        [p for p in candidates if "highly_cited" in p.get("retrieval_reasons", [])]
    )

    # Phase 1–3: soft quotas per perspective.
    added_relevant = 0
    for paper in relevant:
        if added_relevant >= SELECT_RELEVANT:
            break
        if try_add(paper):
            added_relevant += 1

    added_recent = 0
    for paper in recent:
        if added_recent >= SELECT_RECENT:
            break
        if try_add(paper):
            added_recent += 1

    added_cited = 0
    for paper in highly_cited:
        if added_cited >= SELECT_HIGHLY_CITED:
            break
        if try_add(paper):
            added_cited += 1

    # Phase 4: remaining slots → prefer papers with multiple retrieval reasons.
    leftovers = _sort_multi_criteria(
        [p for p in candidates if paper_identity(p) not in selected_keys]
    )
    for paper in leftovers:
        if len(selected) >= target_count:
            break
        try_add(paper)

    return selected


# ---------------------------------------------------------------------------
# Step E: papers  ->  LLM  ->  research synthesis
# ---------------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = """
You are a research assistant synthesizing a BALANCED reading set.

The papers you receive may include:
- foundational / highly cited older work
- highly relevant topical matches
- recent developments
- papers that are both recent and influential

Each paper may include "retrieval_reasons" such as:
["relevant"], ["recent"], ["highly_cited"], or combinations.

Important rules:
- Newer papers are NOT automatically better.
- Foundational papers may be older but still define the field.
- Ground your answer ONLY in the provided papers.
- Do not invent papers, findings, citations, or years.
- If evidence is thin, say so in gaps_or_cautions.

Explain:
1) foundational ideas
2) important recent developments
3) how the field has evolved
4) current gaps or uncertainties

Return ONLY JSON with this exact shape:
{
  "answer": "Short overall answer to the user's question.",
  "foundational_findings": ["..."],
  "recent_developments": ["..."],
  "evolution_of_the_field": "...",
  "papers_used": [
    {
      "title": "...",
      "year": 2025,
      "why_relevant": "..."
    }
  ],
  "gaps_or_cautions": "..."
}
""".strip()


def summarize_papers(user_question: str, papers: list[dict[str, Any]]) -> dict[str, Any]:
    """LLM #2: read the selected papers and synthesize a balanced answer."""
    # Keep abstracts short so the prompt stays inside model limits.
    compact_papers = []
    for paper in papers:
        compact_papers.append(
            {
                "title": paper.get("title"),
                "year": paper.get("year"),
                "authors": paper.get("authors"),
                "cited_by_count": paper.get("cited_by_count"),
                "venue": paper.get("venue"),
                "retrieval_reasons": paper.get("retrieval_reasons", []),
                "abstract": (paper.get("abstract") or "")[:800],
            }
        )

    user_prompt = json.dumps(
        {
            "user_question": user_question,
            "papers": compact_papers,
        },
        indent=2,
    )
    return ask_llm_json(SUMMARY_SYSTEM_PROMPT, user_prompt)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def format_reasons(paper: dict[str, Any]) -> str:
    reasons = paper.get("retrieval_reasons") or []
    return ", ".join(reasons) if reasons else "unlabeled"


def print_selected_papers(papers: list[dict[str, Any]]) -> None:
    for i, paper in enumerate(papers, start=1):
        year = paper.get("year") or "????"
        title = paper.get("title") or "(untitled)"
        print(f"   [{i}] {year} | {title} | {format_reasons(paper)}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_paper_search(user_question: str) -> dict[str, Any]:
    """
    Orchestrator: glue the steps together in Python.

    Architecture:
      LLM #1 → OpenAlex (multi-bucket) → Python (dedupe + select) → LLM #2
    """
    print("\n1) Asking LLM to generate a search query...")
    query_info = generate_search_query(user_question)
    search_query = query_info["search_query"]
    print(f"   query  = {search_query}")
    print(f"   reason = {query_info.get('reason')}")

    print("\n2) Searching OpenAlex...")
    print(f"   recent window  = {recent_from_year()}-present ({RECENT_YEARS} years)")
    buckets = retrieve_paper_buckets(search_query)
    print(f"   relevant papers   = {len(buckets['relevant'])}")
    print(f"   recent papers     = {len(buckets['recent'])}")
    print(f"   highly cited      = {len(buckets['highly_cited'])}")

    print("\n3) Deduplicating papers...")
    candidates = deduplicate_papers(buckets)
    print(f"   unique candidates = {len(candidates)}")

    print("\n4) Selecting balanced paper set...")
    selected = select_balanced_papers(candidates)
    print(f"   selected papers   = {len(selected)}\n")
    print_selected_papers(selected)

    print("\n5) Asking LLM to synthesize the research...")
    summary = summarize_papers(user_question, selected)

    return {
        "user_question": user_question,
        "generated_query": query_info,
        "bucket_counts": {
            "relevant": len(buckets["relevant"]),
            "recent": len(buckets["recent"]),
            "highly_cited": len(buckets["highly_cited"]),
        },
        "unique_candidates": len(candidates),
        "papers": selected,
        "summary": summary,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("📚 Agentic Research Paper Search")
    print("=" * 60)

    # Ask the user for a question directly in the terminal
    USER_QUESTION = input("\nEnter your research question: ").strip()
    if not USER_QUESTION:
        raise SystemExit("No question entered. Exiting.")

    result = run_paper_search(USER_QUESTION)

    print("\n" + "=" * 60)
    print("FINAL STRUCTURED RESULT")
    print("=" * 60)
    print(
        json.dumps(
            {
                "user_question": result["user_question"],
                "generated_query": result["generated_query"],
                "bucket_counts": result["bucket_counts"],
                "unique_candidates": result["unique_candidates"],
                "paper_count": len(result["papers"]),
                "summary": result["summary"],
            },
            indent=2,
        )
    )
