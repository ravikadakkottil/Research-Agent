"""
The agent loop:

    while steps remain:
        LLM sees messages + tool schemas
        if it returns tool_calls -> run them (in parallel), append results, loop again
        else                     -> that's the final answer
    then: grounding check (agent/grounding.py) -> cited [W...] IDs become checked sources
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from research_agent.agent.grounding import check_citations
from research_agent.agent.tools import TOOL_SCHEMAS, AgentTools
from research_agent.config import Settings, get_settings
from research_agent.llm import get_client, usage_details
from research_agent.models import AgentResult
from research_agent.prompts import AGENT_PROMPT
from research_agent.sources.openalex import create_openalex_client
from research_agent.tracing import current_trace_url, observe, score_trace, trace

PREVIEW_CHARS = 400


def _noop(_: str) -> None:
    pass


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
        result = await _agent_loop(user_question, settings, on_step)
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


async def _agent_loop(
    user_question: str, settings: Settings, on_step: Callable[[str], None]
) -> AgentResult:
    llm = get_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": user_question},
    ]

    async with create_openalex_client(settings) as client:
        tools = AgentTools(client, default_limit=settings.agent_default_limit)

        for step in range(1, settings.max_agent_steps + 1):
            on_step(f"\n--- Agent step {step}/{settings.max_agent_steps} ---")
            with observe(
                f"agent step {step}",
                as_type="generation",
                model=settings.llm_model,
                model_parameters={"temperature": settings.llm_temperature},
                input=list(messages),  # a copy: `messages` keeps growing after this step
            ) as gen:
                completion = await llm.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,  # what the model may call
                    tool_choice="auto",  # model may call a tool OR answer directly
                    temperature=settings.llm_temperature,
                )
                message = completion.choices[0].message
                gen.update(
                    output=_assistant_message_to_dict(message),
                    usage_details=usage_details(completion),
                )

            if not message.tool_calls:
                on_step("No tool call -> final answer.")
                return _grounded(message.content or "", tools, on_step)

            calls = message.tool_calls
            on_step(f"Tool selection: {len(calls)} call(s)")
            for tc in calls:
                on_step(f"  -> {tc.function.name}({tc.function.arguments})")
            messages.append(_assistant_message_to_dict(message))

            # If the model asks for several tools at once, run them in parallel.
            results = await asyncio.gather(*(_traced_tool(tools, tc) for tc in calls))

            for tc, result in zip(calls, results, strict=True):
                preview = result if len(result) < PREVIEW_CHARS else result[:PREVIEW_CHARS] + "..."
                on_step(f"     {tc.function.name} result: {preview}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": result,
                    }
                )

    return AgentResult(
        answer=(
            "Stopped: reached MAX_AGENT_STEPS without a final answer. "
            "Try a narrower question, or raise MAX_AGENT_STEPS in .env."
        )
    )


async def _traced_tool(tools: AgentTools, tc: Any) -> str:
    with observe(f"tool: {tc.function.name}", as_type="tool", input=tc.function.arguments) as obs:
        result = await tools.execute(tc.function.name, tc.function.arguments)
        obs.update(output=result)
        return result


def _grounded(answer: str, tools: AgentTools, on_step: Callable[[str], None]) -> AgentResult:
    """Check the answer's [W...] citations against what the tools actually returned."""
    with observe("grounding check", input={"papers_seen": len(tools.seen)}) as obs:
        sources, unverified = check_citations(answer, tools.seen)
        obs.update(
            output={"verified": [p.title for p in sources], "unverified": unverified},
            level="WARNING" if unverified else "DEFAULT",
        )
    on_step(
        f"Grounding check: {len(sources)} cited paper(s) verified, "
        f"{len(unverified)} unverified{' ' + ', '.join(unverified) if unverified else ''}"
    )
    return AgentResult(answer=answer, sources=sources, unverified_ids=unverified)
