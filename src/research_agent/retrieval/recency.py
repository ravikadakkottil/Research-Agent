"""
Decide what "recent" means for THIS question.

Priority (first one that applies wins):

  1) USER time range   — "since 2020", "last 5 years"            (option 1)
  2) FIELD PACE        — LLM's judgement: LLMs fast, medicine slow (option 2)
  3) DEFAULT           — settings.default_recent_years

Key pattern: the LLM PROPOSES, Python DECIDES.
  - A user time range is only trusted if the LLM's quoted phrase really
    appears in the question (stops the LLM inventing one).
  - "last N years" is computed in Python, not trusted to LLM arithmetic.
  - The field-pace suggestion is clamped to sane bounds.

Pure function, no network → easy to unit test.
"""

from __future__ import annotations

import re

from research_agent.models import QueryPlan, RecentWindow

EARLIEST_SENSIBLE_YEAR = 1900
_LAST_N_YEARS = re.compile(r"\b(?:last|past)\s+(\d{1,2})\s+years?\b")


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace, so small formatting differences don't matter."""
    return " ".join(text.casefold().split())


def _user_from_year(plan: QueryPlan, question: str, current_year: int) -> int | None:
    """Return the user's start year if it is grounded in the question, else None."""
    phrase = plan.user_time_phrase
    if not phrase or _normalize(phrase) not in _normalize(question):
        return None  # no phrase, or the LLM "quoted" words the user never wrote

    # "last 5 years" -> compute it ourselves (5 years incl. this one: 2022..2026)
    match = _LAST_N_YEARS.search(_normalize(phrase))
    if match:
        return current_year - int(match.group(1)) + 1

    year = plan.user_from_year
    if year is not None and EARLIEST_SENSIBLE_YEAR <= year <= current_year:
        return year
    return None


def resolve_recent_window(
    plan: QueryPlan,
    question: str,
    *,
    current_year: int,
    default_years: int,
    min_years: int,
    max_years: int,
) -> RecentWindow:
    # 1) The user explicitly asked for a time range.
    user_year = _user_from_year(plan, question, current_year)
    if user_year is not None:
        return RecentWindow(
            from_year=user_year,
            source="user",
            explanation=f'you asked for "{plan.user_time_phrase}"',
        )

    # 2) The LLM's view of how fast this field moves (clamped).
    if plan.suggested_recent_years is not None:
        years = max(min_years, min(plan.suggested_recent_years, max_years))
        reason = plan.time_reason or "LLM field-pace estimate"
        return RecentWindow(
            from_year=current_year - years + 1,
            source="field_pace",
            explanation=f"{years}-year window: {reason}",
        )

    # 3) Fallback.
    return RecentWindow(
        from_year=current_year - default_years + 1,
        source="default",
        explanation=f"default {default_years}-year window",
    )
