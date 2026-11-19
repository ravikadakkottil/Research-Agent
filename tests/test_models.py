"""LLM output validation: Pydantic catches wrong shapes early."""

import pytest
from pydantic import ValidationError

from research_agent.models import QueryPlan, ResearchSummary


def test_query_plan_parses_llm_json():
    raw = (
        '{"search_query": "crispr delivery", "reason": "r", "user_time_phrase": "since 2020",'
        ' "user_from_year": 2020, "suggested_recent_years": 3, "time_reason": "moderate"}'
    )
    plan = QueryPlan.model_validate_json(raw)
    assert plan.user_from_year == 2020


def test_query_plan_tolerates_missing_optional_fields():
    plan = QueryPlan.model_validate_json('{"search_query": "x"}')
    assert plan.suggested_recent_years is None


def test_query_plan_rejects_empty_query():
    with pytest.raises(ValidationError):
        QueryPlan.model_validate_json('{"search_query": ""}')


def test_alternative_queries_are_cleaned_and_capped():
    plan = QueryPlan(
        search_query="misinformation social media",
        core_query="  misinformation ",
        alternative_queries=[
            "fake news",
            " ",
            "Fake  News",
            "false news",
            "rumors",
            "misinformation",
        ],
    )
    assert plan.strict_query == "misinformation"
    # blanks + duplicates dropped, capped at MAX_ALTERNATIVE_QUERIES (2)
    assert plan.alternative_queries == ["fake news", "false news"]


def test_alternatives_drop_redundant_and_ambiguous_queries():
    plan = QueryPlan(
        search_query="x",
        core_query="CRISPR Cas9",
        alternative_queries=["CRISPR Cas9 system", "ICI"],
    )
    # superset of the core words -> can only return a subset of its results;
    # a lone short abbreviation -> ambiguous across fields
    assert plan.alternatives == []
    plan = QueryPlan(
        search_query="x", core_query="CRISPR Cas9", alternative_queries=["genome editing"]
    )
    assert plan.alternatives == ["genome editing"]


def test_alternatives_skip_repeats_of_core_query():
    plan = QueryPlan(search_query="x", core_query="CRISPR", alternative_queries=["crispr"])
    assert plan.alternatives == []


def test_summary_requires_answer():
    with pytest.raises(ValidationError):
        ResearchSummary.model_validate_json('{"foundational_findings": []}')
