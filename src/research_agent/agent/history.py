"""
Keep the agent's message history inside the LLM's request size limit.

The LLM has no memory: every step re-sends the whole history, and every tool result
stays in it. On Groq's free tier ONE request may not exceed the tokens-per-minute
limit (8000 for gpt-oss-20b), so after ~7 searches a request is refused (HTTP 413).

fit_history() shortens OLDER tool results, oldest first, until the history fits:
each paper keeps its ID, title, year and citation count (enough to cite it), but
loses authors, venue and abstract. The newest results stay complete, because that's
what the model is reasoning about right now. The full results stay in the graph
state and the trace; only the copy sent to the LLM is shortened.

Token counts are ESTIMATED from characters: measured on live traces, our JSON
messages run at ~3.5-4.1 characters per token, plus ~300 tokens for the tool schemas.
Dividing by 4 and adding 400 errs slightly on the high side, which is the safe side.
"""

from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 4
FIXED_OVERHEAD_TOKENS = 400  # tool schemas + chat template
_KEEP_PAPER_FIELDS = ("openalex_id", "title", "year", "cited_by_count")
_KEEP_TOP_FIELDS = ("query", "paper_id", "relation", "count", "error")


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages)) // CHARS_PER_TOKEN + FIXED_OVERHEAD_TOKENS


def shorten_tool_result(content: str) -> str:
    """The same tool result, reduced to what's needed to cite its papers."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content[:300]
    if not isinstance(data, dict):
        return content[:300]

    def paper(p: dict[str, Any]) -> dict[str, Any]:
        return {k: p[k] for k in _KEEP_PAPER_FIELDS if k in p}

    if "papers" in data:  # search_papers, get_citations, get_references
        short = {k: data[k] for k in _KEEP_TOP_FIELDS if k in data}
        short["papers"] = [paper(p) for p in data["papers"]]
    elif "openalex_id" in data:  # get_paper
        short = paper(data)
    else:  # an error, or something unexpected: already small
        return content
    short["note"] = "shortened to save space; call get_paper(id) for details"
    return json.dumps(short)


def _latest_tool_batch_start(messages: list[dict[str, Any]]) -> int:
    """Index of the first tool reply after the last assistant message."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return i + 1
    return len(messages)


def fit_history(
    messages: list[dict[str, Any]], max_tokens: int, *, keep_latest: bool = True
) -> list[dict[str, Any]]:
    """
    A copy of `messages` that fits `max_tokens`, if shortening tool results can do it.

    keep_latest=True leaves the newest batch of tool results untouched. The input list
    and its dicts are never changed (they belong to the graph state).
    """
    if estimate_tokens(messages) <= max_tokens:
        return messages
    out = list(messages)
    stop = _latest_tool_batch_start(out) if keep_latest else len(out)
    for i in range(stop):
        if out[i]["role"] != "tool":
            continue
        out[i] = {**out[i], "content": shorten_tool_result(out[i]["content"])}
        if estimate_tokens(out) <= max_tokens:
            break
    return out
