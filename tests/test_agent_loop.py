"""
Agent loop tests: a scripted fake LLM + fake OpenAlex, no network.

Written against the hand-written loop BEFORE porting it to LangGraph
(characterization tests), so the graph version must behave the same:
the messages the LLM sees, the order of steps, parallel tool calls,
the step limit and the grounding check.
"""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from research_agent.agent import loop
from research_agent.config import Settings
from tests.test_agent_grounding import FakeOpenAlex


class LoopOpenAlex(FakeOpenAlex):
    async def get_work(self, paper_id):
        # A full-length ID: grounding only counts W + 4 or more digits as a citation.
        return {"id": "https://openalex.org/W4200000042", "display_name": "Looked up"}


def tool_call(call_id: str, name: str, **args) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def reply(content: str | None = None, calls: list | None = None) -> SimpleNamespace:
    """A Groq-style completion with one choice."""
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeLLM:
    """Returns the scripted replies in order; records what it was sent."""

    def __init__(self, replies: list) -> None:
        self._replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))  # the loop keeps mutating `messages`
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeLLM with the given replies; returns it."""

    @asynccontextmanager
    async def fake_openalex(_settings):
        yield LoopOpenAlex()

    monkeypatch.setattr(loop, "create_openalex_client", fake_openalex)

    def install(replies: list) -> FakeLLM:
        llm = FakeLLM(replies)
        monkeypatch.setattr(loop, "get_client", lambda: llm)
        return llm

    return install


def settings(**overrides) -> Settings:
    return Settings(max_agent_steps=4, agent_default_limit=5, **overrides)


def run(question: str, **kw):
    steps: list[str] = []
    result = asyncio.run(
        loop.run_tool_agent(question, on_step=steps.append, settings=kw.pop("s", settings()))
    )
    return result, steps


def test_tool_call_then_grounded_answer(fake_llm):
    llm = fake_llm(
        [
            reply(calls=[tool_call("c1", "search_papers", query="graph attention")]),
            reply("GAT [W2766453196] is key; also [W9999999]."),
        ]
    )
    result, steps = run("graph attention papers")

    # The answer and its grounding: W9999999 was never returned by a tool.
    assert result.answer == "GAT [W2766453196] is key; also [W9999999]."
    assert [p.title for p in result.sources] == ["Graph Attention Networks"]
    assert result.unverified_ids == ["W9999999"]

    # Step messages, in order (the web UI and CLI show these).
    assert steps[0] == "\n--- Agent step 1/4 ---"
    assert steps[1] == "Tool selection: 1 call(s)"
    assert steps[2] == '  -> search_papers({"query": "graph attention"})'
    assert steps[3].startswith("     search_papers result: ")
    assert steps[4:] == [
        "\n--- Agent step 2/4 ---",
        "No tool call -> final answer.",
        "Grounding check: 1 cited paper(s) verified, 1 unverified W9999999",
    ]

    # What the LLM saw: system + user first, then its own tool request + the tool result.
    assert len(llm.requests) == 2
    first, second = (r["messages"] for r in llm.requests)
    assert [m["role"] for m in first] == ["system", "user"]
    assert first[1]["content"] == "graph attention papers"
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["id"] == "c1"
    assert second[3]["tool_call_id"] == "c1"
    assert second[3]["name"] == "search_papers"
    assert json.loads(second[3]["content"])["count"] == 2
    for request in llm.requests:
        assert request["tool_choice"] == "auto"
        assert {t["function"]["name"] for t in request["tools"]} >= {"search_papers"}


def test_parallel_tool_calls_keep_their_order(fake_llm):
    llm = fake_llm(
        [
            reply(
                calls=[
                    tool_call("a", "get_paper", paper_id="W4200000042"),
                    tool_call("b", "search_papers", query="x"),
                ]
            ),
            reply("Done [W4200000042]."),
        ]
    )
    result, steps = run("q")

    tool_msgs = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]  # same order as requested
    assert steps[1] == "Tool selection: 2 call(s)"
    assert [p.title for p in result.sources] == ["Looked up"]
    assert result.unverified_ids == []


def test_direct_answer_without_tools(fake_llm):
    llm = fake_llm([reply("No papers needed.")])
    result, steps = run("hi")

    assert len(llm.requests) == 1
    assert result.answer == "No papers needed."
    assert result.sources == [] and result.unverified_ids == []
    assert steps == [
        "\n--- Agent step 1/4 ---",
        "No tool call -> final answer.",
        "Grounding check: 0 cited paper(s) verified, 0 unverified",
    ]


def test_step_limit_stops_the_loop(fake_llm):
    # The model never stops asking for tools.
    llm = fake_llm([reply(calls=[tool_call("c", "search_papers", query="x")])])
    result, steps = run("q", s=Settings(max_agent_steps=2, agent_default_limit=5))

    assert len(llm.requests) == 2
    assert result.answer.startswith("Stopped: reached MAX_AGENT_STEPS")
    assert result.sources == []
    assert "\n--- Agent step 2/2 ---" in steps
    assert not any(s.startswith("Grounding check") for s in steps)


def test_many_steps_fit_inside_langgraphs_recursion_limit(fake_llm):
    # 15 steps = 30+ node runs: LangGraph's default limit (25) would raise
    # GraphRecursionError. Our own step limit must be the one that stops the run.
    llm = fake_llm([reply(calls=[tool_call("c", "search_papers", query="x")])])
    result, _ = run("q", s=Settings(max_agent_steps=15, agent_default_limit=5))

    assert len(llm.requests) == 15
    assert result.answer.startswith("Stopped: reached MAX_AGENT_STEPS")


def test_graph_shape():
    graph = loop.agent_graph().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert edges == {
        ("__start__", "llm"),
        ("llm", "tools"),
        ("llm", "ground"),
        ("tools", "llm"),
        ("tools", "out_of_steps"),
        ("ground", "__end__"),
        ("out_of_steps", "__end__"),
    }


def test_tool_errors_go_back_to_the_model(fake_llm):
    llm = fake_llm([reply(calls=[tool_call("c", "no_such_tool")]), reply("ok")])
    run("q")

    tool_msg = llm.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "Unknown tool" in json.loads(tool_msg["content"])["error"]
