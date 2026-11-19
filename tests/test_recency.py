"""Recent-window decision: user > field pace > default (pure logic)."""

from research_agent.models import QueryPlan
from research_agent.retrieval.recency import resolve_recent_window

YEAR = 2026
BOUNDS = {"current_year": YEAR, "default_years": 3, "min_years": 1, "max_years": 10}


def plan(**kwargs) -> QueryPlan:
    return QueryPlan(search_query="q", **kwargs)


def test_user_since_year_wins_over_field_pace():
    window = resolve_recent_window(
        plan(user_time_phrase="since 2020", user_from_year=2020, suggested_recent_years=2),
        "CRISPR delivery advances since 2020",
        **BOUNDS,
    )
    assert (window.from_year, window.source) == (2020, "user")


def test_last_n_years_is_computed_in_python_not_trusted_from_llm():
    # LLM arithmetic is off (says 2020); Python computes 2026-5+1 = 2022.
    window = resolve_recent_window(
        plan(user_time_phrase="last 5 years", user_from_year=2020),
        "vaccine adjuvants in the Last 5 Years",
        **BOUNDS,
    )
    assert (window.from_year, window.source) == (2022, "user")


def test_invented_user_phrase_is_ignored():
    # The question has no time range; the LLM made one up -> fall back to field pace.
    window = resolve_recent_window(
        plan(user_time_phrase="since 2015", user_from_year=2015, suggested_recent_years=2),
        "recent advances in LLM agents",
        **BOUNDS,
    )
    assert (window.from_year, window.source) == (2025, "field_pace")


def test_future_user_year_is_rejected():
    window = resolve_recent_window(
        plan(user_time_phrase="since 2030", user_from_year=2030),
        "papers since 2030",
        **BOUNDS,
    )
    assert window.source == "default"


def test_field_pace_is_clamped():
    too_big = resolve_recent_window(plan(suggested_recent_years=50), "q", **BOUNDS)
    too_small = resolve_recent_window(plan(suggested_recent_years=0), "q", **BOUNDS)
    assert too_big.from_year == YEAR - 10 + 1
    assert too_small.from_year == YEAR  # 1 year = this year only


def test_default_when_llm_gives_nothing():
    window = resolve_recent_window(plan(), "q", **BOUNDS)
    assert (window.from_year, window.source) == (2024, "default")
