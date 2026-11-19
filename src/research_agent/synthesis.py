"""
LLM #2: read the selected papers and write a balanced, grounded synthesis.
"""

from __future__ import annotations

import json

from research_agent.cache import ResponseCache
from research_agent.llm import ask_json
from research_agent.models import Paper, RecentWindow, ResearchSummary
from research_agent.prompts import SUMMARY_PROMPT


async def summarize_papers(
    user_question: str,
    papers: list[Paper],
    *,
    recent_window: RecentWindow,
    abstract_chars: int,
    cache: ResponseCache | None = None,
) -> ResearchSummary:
    # Send only what the model needs; truncate abstracts to keep the prompt small.
    compact = [
        {
            **p.model_dump(include={"title", "year", "authors", "cited_by_count", "venue"}),
            "retrieval_reasons": p.retrieval_reasons,
            "abstract": p.abstract[:abstract_chars],
        }
        for p in papers
    ]
    user_prompt = json.dumps(
        {
            "user_question": user_question,
            # Tells the model what "recent" meant for this search.
            "recent_means": f"published {recent_window.from_year} or later",
            "papers": compact,
        },
        indent=2,
        ensure_ascii=False,
    )
    return await ask_json(SUMMARY_PROMPT, user_prompt, ResearchSummary, cache=cache)
