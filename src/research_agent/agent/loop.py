"""
The tool-calling agent, as a LangGraph graph:

    START -> llm --(tool calls?)--> tools --(steps left?)--> llm      (the loop)
              |                       +--(no steps left)--> out_of_steps -> END
              +--(final answer)--> ground -> END

    llm           one LLM turn: sees messages + tool schemas, may request tools
    tools         runs the requested tools in parallel, appends one result per call
    ground        checks the answer's [W...] IDs against what the tools returned
    out_of_steps  MAX_AGENT_STEPS reached without a final answer

State holds what changes during a run (the message history, the step count, the
result). Context holds what doesn't (settings, LLM client, tools, progress callback).
`research-agent --graph agent` prints the graph as Mermaid.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from research_agent.agent.grounding import check_citations
from research_agent.agent.tools import TOOL_SCHEMAS, AgentTools
from research_agent.config import Settings, get_settings
from research_agent.llm import get_client, usage_details
from research_agent.models import AgentResult
from research_agent.prompts import AGENT_PROMPT
from research_agent.sources.openalex import create_openalex_client
from research_agent.tracing import current_trace_url, observe, score_trace, trace

PREVIEW_CHARS = 400
STOPPED_ANSWER = (
    "Stopped: reached MAX_AGENT_STEPS without a final answer. "
    "Try a narrower question, or raise MAX_AGENT_STEPS in .env."
)


def _noop(_: str) -> None:
    pass


# --- state and context ----------------------------------------------------------------


class AgentState(TypedDict, total=False):
    # A REDUCER: nodes return only their NEW messages and LangGraph appends them
    # (operator.add on lists). Without it, each node would have to return the whole
    # history, and a node that forgot one message would silently drop it.
    messages: Annotated[list[dict[str, Any]], operator.add]
    step: int  # LLM turns taken so far
    result: AgentResult


@dataclass
class AgentContext:
    settings: Settings
    llm: Any  # the Groq client
    tools: AgentTools  # also remembers every paper it returned (tools.seen)
    on_step: Callable[[str], None]


Ctx = Runtime[AgentContext]


# --- nodes ----------------------------------------------------------------------------


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """
    Echo the assistant's tool request back into history.

    The API requires: assistant message WITH tool_calls, followed by one
    role="tool" message per call id.
    """
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return payload


async def _node_llm(state: AgentState, runtime: Ctx) -> dict:
    ctx = runtime.context
    s = ctx.settings
    step = state.get("step", 0) + 1
    ctx.on_step(f"\n--- Agent step {step}/{s.max_agent_steps} ---")
    with observe(
        f"agent step {step}",
        as_type="generation",
        model=s.llm_model,
        model_parameters={"temperature": s.llm_temperature},
        input=state["messages"],
    ) as gen:
        completion = await ctx.llm.chat.completions.create(
            model=s.llm_model,
            messages=state["messages"],
            tools=TOOL_SCHEMAS,  # what the model may call
            tool_choice="auto",  # model may call a tool OR answer directly
            temperature=s.llm_temperature,
        )
        message = _assistant_message_to_dict(completion.choices[0].message)
        gen.update(output=message, usage_details=usage_details(completion))
    return {"messages": [message], "step": step}


async def _traced_tool(tools: AgentTools, call: dict[str, Any]) -> str:
    name, args = call["function"]["name"], call["function"]["arguments"]
    with observe(f"tool: {name}", as_type="tool", input=args) as obs:
        result = await tools.execute(name, args)
        obs.update(output=result)
        return result


async def _node_tools(state: AgentState, runtime: Ctx) -> dict:
    ctx = runtime.context
    calls = state["messages"][-1]["tool_calls"]
    ctx.on_step(f"Tool selection: {len(calls)} call(s)")
    for call in calls:
        ctx.on_step(f"  -> {call['function']['name']}({call['function']['arguments']})")

    # If the model asks for several tools at once, run them in parallel.
    results = await asyncio.gather(*(_traced_tool(ctx.tools, call) for call in calls))

    replies = []
    for call, result in zip(calls, results, strict=True):
        name = call["function"]["name"]
        preview = result if len(result) < PREVIEW_CHARS else result[:PREVIEW_CHARS] + "..."
        ctx.on_step(f"     {name} result: {preview}")
        replies.append(
            {"role": "tool", "tool_call_id": call["id"], "name": name, "content": result}
        )
    return {"messages": replies}


async def _node_ground(state: AgentState, runtime: Ctx) -> dict:
    """Check the answer's [W...] citations against what the tools actually returned."""
    ctx = runtime.context
    ctx.on_step("No tool call -> final answer.")
    answer = state["messages"][-1]["content"] or ""
    with observe("grounding check", input={"papers_seen": len(ctx.tools.seen)}) as obs:
        sources, unverified = check_citations(answer, ctx.tools.seen)
        obs.update(
            output={"verified": [p.title for p in sources], "unverified": unverified},
            level="WARNING" if unverified else "DEFAULT",
        )
    ctx.on_step(
        f"Grounding check: {len(sources)} cited paper(s) verified, "
        f"{len(unverified)} unverified{' ' + ', '.join(unverified) if unverified else ''}"
    )
    return {"result": AgentResult(answer=answer, sources=sources, unverified_ids=unverified)}


async def _node_out_of_steps(state: AgentState, runtime: Ctx) -> dict:
    return {"result": AgentResult(answer=STOPPED_ANSWER)}


# --- routing --------------------------------------------------------------------------


def _route_after_llm(state: AgentState) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else "ground"


def _route_after_tools(state: AgentState, runtime: Ctx) -> str:
    return "llm" if state["step"] < runtime.context.settings.max_agent_steps else "out_of_steps"


# --- graph ----------------------------------------------------------------------------


@lru_cache
def agent_graph() -> CompiledStateGraph:
    """Built once; per-run data goes in through the input state and the context."""
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("llm", _node_llm)
    g.add_node("tools", _node_tools)
    g.add_node("ground", _node_ground)
    g.add_node("out_of_steps", _node_out_of_steps)

    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", _route_after_llm, ["tools", "ground"])
    g.add_conditional_edges("tools", _route_after_tools, ["llm", "out_of_steps"])
    g.add_edge("ground", END)
    g.add_edge("out_of_steps", END)
    return g.compile(name="agent")


def _recursion_limit(max_agent_steps: int) -> int:
    # LangGraph stops any run after `recursion_limit` node executions (default 25)
    # as a guard against endless loops. Each agent step is 2 nodes (llm + tools),
    # plus the final node, so the default would cut off MAX_AGENT_STEPS above ~11.
    # Our own step limit is the real guard; this just has to stay out of its way.
    return 2 * max_agent_steps + 5


# --- public entry point ---------------------------------------------------------------


async def run_tool_agent(
    user_question: str,
    *,
    on_step: Callable[[str], None] = _noop,
    settings: Settings | None = None,
) -> AgentResult:
    settings = settings or get_settings()
    # One "agent" trace per question: each LLM turn is a generation, each tool call
    # a "tool" observation, so the trace reads like the agent's reasoning log.
    with trace("agent", as_type="agent", input={"question": user_question}) as root:
        result = await _run(user_question, settings, on_step)
        root.update(
            output={
                "answer": result.answer,
                "sources": [p.title for p in result.sources],
                "unverified_ids": result.unverified_ids,
            }
        )
        # Scores make bad runs searchable in Langfuse (filter: unverified_citations > 0).
        score_trace("verified_citations", len(result.sources))
        score_trace("unverified_citations", len(result.unverified_ids))
        if url := await current_trace_url():
            on_step(f"Trace: {url}")
        return result


async def _run(
    user_question: str, settings: Settings, on_step: Callable[[str], None]
) -> AgentResult:
    async with create_openalex_client(settings) as client:
        context = AgentContext(
            settings=settings,
            llm=get_client(),
            tools=AgentTools(client, default_limit=settings.agent_default_limit),
            on_step=on_step,
        )
        state = await agent_graph().ainvoke(
            {
                "messages": [
                    {"role": "system", "content": AGENT_PROMPT},
                    {"role": "user", "content": user_question},
                ]
            },
            context=context,
            config={"recursion_limit": _recursion_limit(settings.max_agent_steps)},
        )
    return state["result"]
