"""
Tracing tests.

Langfuse is built on OpenTelemetry, so a real Langfuse client can be pointed
at an IN-MEMORY span exporter: spans are captured in a list instead of being
sent anywhere. That lets the tests check nesting and payloads offline.
"""

import asyncio
import json

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from research_agent import pipeline, tracing
from research_agent.agent import loop as agent_loop
from research_agent.tracing import NOOP, observe, trace
from tests.test_agent_loop import fake_llm, reply, tool_call  # noqa: F401  (fake_llm: fixture)
from tests.test_agent_loop import settings as agent_settings
from tests.test_openalex import make_client
from tests.test_pipeline import FakeOpenAlex, fakes  # noqa: F401  (fakes is a fixture)
from tests.test_pipeline import settings as pipeline_settings

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module")
def langfuse_client():
    from langfuse import Langfuse

    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",  # unreachable on purpose: nothing leaves the machine
        span_exporter=EXPORTER,
        flush_at=1,
    )
    yield client
    client.shutdown()


@pytest.fixture
def spans(langfuse_client, monkeypatch):
    """Turn tracing on for one test; returns a function giving the finished spans."""
    monkeypatch.setattr(tracing, "get_langfuse", lambda: langfuse_client)
    EXPORTER.clear()

    def finished():
        langfuse_client.flush()
        return {s.name: s for s in EXPORTER.get_finished_spans()}

    return finished


def test_tracing_off_is_a_noop():
    # conftest sets TRACING_ENABLED=false
    assert tracing.get_langfuse() is None
    with trace("search", input={"q": 1}) as root, observe("step") as step:
        assert root is NOOP and step is NOOP
        step.update(output="ignored")


def test_steps_nest_inside_the_trace_even_across_gather(spans):
    async def child(name: str) -> None:
        with observe(name):
            await asyncio.sleep(0)

    async def run() -> None:
        with trace("search", input={"question": "q"}):
            with observe("retrieve"):
                await asyncio.gather(child("bucket a"), child("bucket b"))

    asyncio.run(run())
    got = spans()

    root, retrieve = got["search"], got["retrieve"]
    assert retrieve.parent.span_id == root.context.span_id
    for name in ("bucket a", "bucket b"):
        assert got[name].parent.span_id == retrieve.context.span_id  # gather kept the parent
        assert got[name].context.trace_id == root.context.trace_id


def test_nested_trace_becomes_a_child_span(spans):
    """Evals open a trace, and the pipeline's own trace() must not start a second one."""
    with trace("eval: rag"):
        with trace("retrieval"):
            pass
    got = spans()
    assert got["retrieval"].parent.span_id == got["eval: rag"].context.span_id


def test_errors_are_recorded_and_reraised(spans):
    with pytest.raises(ValueError), observe("bad step"):
        raise ValueError("boom")
    attrs = spans()["bad step"].attributes
    assert attrs.get("langfuse.observation.level") == "ERROR"
    assert "boom" in attrs.get("langfuse.observation.status_message", "")


@pytest.mark.usefixtures("fakes")
def test_graph_node_spans_nest_inside_the_search_trace(spans):
    """LangGraph runs each node as its own task; their spans must still join the trace."""
    asyncio.run(
        pipeline.run_paper_search("diffusion", settings=pipeline_settings(), client=FakeOpenAlex())
    )
    got = spans()

    root = got["search"]
    nodes = ("plan", "retrieve", "dedupe", "semantic scoring", "citation expansion")
    for name in (*nodes, "similarity filter", "select"):
        assert got[name].parent.span_id == root.context.span_id, name


def test_agent_spans_nest_inside_the_agent_trace(spans, fake_llm):  # noqa: F811  (fixture)
    fake_llm(
        [
            reply(calls=[tool_call("c1", "search_papers", query="graph attention")]),
            reply("GAT [W2766453196]."),
        ]
    )
    asyncio.run(agent_loop.run_tool_agent("q", settings=agent_settings()))
    got = spans()

    root = got["agent"]
    for name in ("agent step 1", "tool: search_papers", "agent step 2", "grounding check"):
        assert got[name].parent.span_id == root.context.span_id, name
        assert got[name].context.trace_id == root.context.trace_id, name
    assert got["agent step 1"].attributes["langfuse.observation.type"] == "generation"
    assert got["tool: search_papers"].attributes["langfuse.observation.type"] == "tool"


def test_openalex_span_never_contains_the_api_key(spans):
    def server(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "secret-123"  # the key IS sent to OpenAlex
        return httpx.Response(200, json={"meta": {"count": 1}, "results": [{"display_name": "T"}]})

    async def run() -> None:
        async with make_client(server, openalex_api_key="secret-123") as client:
            await client.search_works("graph networks", per_page=5)

    asyncio.run(run())
    span = spans()["openalex /works"]
    dumped = json.dumps(dict(span.attributes))
    assert "secret-123" not in dumped
    assert "graph networks" in dumped  # ...but the query is visible
    assert json.loads(span.attributes["langfuse.observation.output"])["returned"] == 1
