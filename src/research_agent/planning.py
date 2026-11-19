"""
LLM #1: turn the user's question into a QueryPlan.

The plan has two jobs:
  1) search keywords for OpenAlex
  2) inputs for the recent-window decision (user time range + field pace)

The LLM only *proposes*; retrieval/recency.py decides (and checks) in Python.
"""

from __future__ import annotations

from datetime import date

from research_agent.cache import ResponseCache
from research_agent.llm import ask_json
from research_agent.models import QueryPlan
from research_agent.prompts import QUERY_PLAN_PROMPT


async def generate_query_plan(
    user_question: str,
    *,
    current_year: int | None = None,
    cache: ResponseCache | None = None,
) -> QueryPlan:
    year = current_year or date.today().year
    system_prompt = QUERY_PLAN_PROMPT.format(current_year=year)
    return await ask_json(system_prompt, user_question, QueryPlan, cache=cache)
