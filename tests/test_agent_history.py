"""History budget: older tool results are shortened so a request fits the LLM's limit."""

import copy
import json

import httpx
from groq import APIStatusError, RateLimitError

from research_agent.agent.history import estimate_tokens, fit_history, shorten_tool_result
from research_agent.llm import describe_llm_limit


def search_result(n: int, tag: str) -> str:
    papers = [
        {
            "openalex_id": f"W{tag}{i:05d}",
            "title": f"Paper {tag}{i}",
            "year": 2020,
            "cited_by_count": 10 * i,
            "authors": ["A. Author", "B. Author"],
            "venue": "Some Venue",
            "abstract": "word " * 60,
        }
        for i in range(n)
    ]
    return json.dumps({"query": tag, "count": n, "papers": papers})


def history(batches: int) -> list[dict]:
    """system + user, then `batches` rounds of (assistant tool call, tool result)."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    for b in range(batches):
        call = {"id": f"c{b}", "type": "function", "function": {"name": "search_papers"}}
        msgs.append({"role": "assistant", "content": None, "tool_calls": [call]})
        tool_msg = {"role": "tool", "tool_call_id": f"c{b}", "name": "search_papers"}
        msgs.append({**tool_msg, "content": search_result(5, str(b + 1))})
    return msgs


def test_shorten_keeps_what_is_needed_to_cite():
    short = json.loads(shorten_tool_result(search_result(3, "7")))
    assert short["query"] == "7" and short["count"] == 3
    assert short["papers"][1] == {
        "openalex_id": "W700001",
        "title": "Paper 71",
        "year": 2020,
        "cited_by_count": 10,
    }
    assert "note" in short

    paper = json.dumps({"openalex_id": "W42000", "title": "T", "abstract": "x" * 900})
    assert json.loads(shorten_tool_result(paper))["openalex_id"] == "W42000"
    assert "abstract" not in shorten_tool_result(paper)

    error = json.dumps({"error": "Unknown tool: x"})
    assert shorten_tool_result(error) == error  # errors are already small
    assert shorten_tool_result("not json" * 100) == ("not json" * 100)[:300]


def test_under_budget_nothing_changes():
    msgs = history(2)
    assert fit_history(msgs, max_tokens=100_000) is msgs


def test_over_budget_shortens_oldest_first_and_keeps_the_latest():
    msgs = history(3)
    original = copy.deepcopy(msgs)
    # Budget: exactly enough once the OLDEST result (index 3) is shortened.
    oldest_shortened = list(msgs)
    oldest_shortened[3] = {**msgs[3], "content": shorten_tool_result(msgs[3]["content"])}
    budget = estimate_tokens(oldest_shortened)
    assert estimate_tokens(msgs) > budget

    fitted = fit_history(msgs, budget)

    assert msgs == original  # the state's history is never changed
    assert fitted[3]["content"] != msgs[3]["content"]  # oldest result: shortened
    assert fitted[5] is msgs[5]  # middle one: not needed, untouched
    assert fitted[7] is msgs[7]  # latest result: kept in full
    assert estimate_tokens(fitted) <= budget


def test_latest_batch_shrinks_only_when_allowed():
    msgs = history(1)
    tiny = 10
    assert fit_history(msgs, tiny)[3] is msgs[3]  # keep_latest=True (the llm node)
    assert fit_history(msgs, tiny, keep_latest=False)[3] != msgs[3]  # the finish node


def _groq_error(cls, status: int, message: str):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))
    return cls(message, response=response, body=None)


def test_llm_limit_messages_are_friendly_and_hide_the_org_id():
    too_big = _groq_error(
        APIStatusError, 413, "Request too large for model in organization `org_123` ..."
    )
    per_day = _groq_error(
        RateLimitError,
        429,
        "Rate limit reached in organization `org_123`. Please try again in 4m37.776s.",
    )
    no_hint = _groq_error(RateLimitError, 429, "Rate limit reached")
    other = _groq_error(APIStatusError, 500, "server error")

    assert "too large" in describe_llm_limit(too_big)
    assert describe_llm_limit(per_day) == (
        "The LLM provider's rate limit was reached. Try again in 4m37.776s."
    )
    assert describe_llm_limit(no_hint).endswith("Try again in a few minutes.")
    assert describe_llm_limit(other) is None
    assert describe_llm_limit(ValueError("x")) is None
    for exc in (too_big, per_day):
        assert "org_" not in describe_llm_limit(exc)
