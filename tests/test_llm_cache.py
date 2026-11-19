"""LLM response cache: a hit must skip the API call entirely."""

import asyncio
from pathlib import Path

import httpx
from groq import BadRequestError

from research_agent import llm
from research_agent.cache import ResponseCache
from research_agent.models import QueryPlan


class _Msg:
    content = '{"search_query": "graph neural networks"}'


class _Choice:
    message = _Msg()


class _Completion:
    choices = [_Choice()]


class FakeGroq:
    """Stands in for AsyncGroq; counts calls instead of hitting the network."""

    def __init__(self) -> None:
        self.calls = 0
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls += 1
        return _Completion()


class FlakyGroq(FakeGroq):
    """First call fails like Groq's 400 json_validate_failed; the second succeeds."""

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            raise BadRequestError(
                "Error code: 400 - {'error': {'code': 'json_validate_failed'}}",
                response=httpx.Response(400, request=request),
                body={"error": {"code": "json_validate_failed"}},
            )
        return _Completion()


def test_invalid_json_from_model_is_retried(monkeypatch):
    fake = FlakyGroq()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    plan = asyncio.run(llm.ask_json("system", "question", QueryPlan))
    assert plan.search_query == "graph neural networks"
    assert fake.calls == 2


def test_second_identical_call_is_served_from_cache(tmp_path: Path, monkeypatch):
    fake = FakeGroq()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    cache = ResponseCache(tmp_path / "llm.sqlite", ttl_seconds=60)

    async def ask_twice():
        a = await llm.ask_json("sys", "question", QueryPlan, cache=cache)
        b = await llm.ask_json("sys", "question", QueryPlan, cache=cache)
        return a, b

    first, second = asyncio.run(ask_twice())

    assert first == second
    assert fake.calls == 1  # second answer came from disk
    assert cache.hits == 1


def test_different_prompt_is_a_cache_miss(tmp_path: Path, monkeypatch):
    fake = FakeGroq()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    cache = ResponseCache(tmp_path / "llm.sqlite", ttl_seconds=60)

    async def ask_two_questions():
        await llm.ask_json("sys", "question A", QueryPlan, cache=cache)
        await llm.ask_json("sys", "question B", QueryPlan, cache=cache)

    asyncio.run(ask_two_questions())
    assert fake.calls == 2
