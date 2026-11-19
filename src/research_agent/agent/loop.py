"""
The agent loop:

    while steps remain:
        LLM sees messages + tool schemas
        if it returns tool_calls -> run them (in parallel), append results, loop again
        else                     -> that's the final answer
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from research_agent.agent.tools import TOOL_SCHEMAS, AgentTools
from research_agent.config import Settings, get_settings
from research_agent.llm import get_client
from research_agent.prompts import AGENT_PROMPT
from research_agent.sources.openalex import create_openalex_client

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
) -> str:
    settings = settings or get_settings()
    llm = get_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": user_question},
    ]

    async with create_openalex_client(settings) as client:
        tools = AgentTools(client, default_limit=settings.agent_default_limit)

        for step in range(1, settings.max_agent_steps + 1):
            on_step(f"\n--- Agent step {step}/{settings.max_agent_steps} ---")
            completion = await llm.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=TOOL_SCHEMAS,  # what the model may call
                tool_choice="auto",  # model may call a tool OR answer directly
                temperature=settings.llm_temperature,
            )
            message = completion.choices[0].message

            if not message.tool_calls:
                on_step("No tool call -> final answer.")
                return message.content or ""

            calls = message.tool_calls
            on_step(f"Tool selection: {len(calls)} call(s)")
            for tc in calls:
                on_step(f"  -> {tc.function.name}({tc.function.arguments})")
            messages.append(_assistant_message_to_dict(message))

            # If the model asks for several tools at once, run them in parallel.
            results = await asyncio.gather(
                *(tools.execute(tc.function.name, tc.function.arguments) for tc in calls)
            )

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

    return (
        "Stopped: reached MAX_AGENT_STEPS without a final answer. "
        "Try a narrower question, or raise MAX_AGENT_STEPS in .env."
    )
