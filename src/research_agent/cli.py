"""
Terminal interface — the ONLY module that uses input() / print().

Usage:
    research-agent                       # pipeline mode, asks for a question
    research-agent "CRISPR since 2020"   # pipeline mode, question as argument
    research-agent --agent "..."         # tool-calling agent mode
    research-agent --no-cache "..."      # always hit OpenAlex (skip the cache)
    research-agent --clear-cache         # delete cached OpenAlex responses
    python -m research_agent ...         # same as research-agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pydantic import ValidationError

from research_agent.agent.loop import run_tool_agent
from research_agent.cache import ResponseCache
from research_agent.config import Settings, get_settings
from research_agent.embeddings import EmbeddingError
from research_agent.llm import LLMOutputError
from research_agent.pipeline import run_paper_search
from research_agent.sources.openalex import OpenAlexRateLimitError
from research_agent.tracing import flush_tracing

RULE = "=" * 60


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Find relevant, influential and recent papers and summarize them.",
    )
    parser.add_argument("question", nargs="*", help="research question (prompted if omitted)")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="use the tool-calling agent (LLM chooses tools) instead of the fixed pipeline",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="skip the OpenAlex + LLM caches (fresh results)"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="delete cached OpenAlex + LLM responses, exit"
    )
    return parser.parse_args(argv)


def _clear_cache(settings: Settings) -> None:
    for name in ("openalex.sqlite", "europepmc.sqlite", "llm.sqlite", "embeddings.sqlite"):
        path = settings.cache_dir / name
        cache = ResponseCache(path, ttl_seconds=0)
        cache.clear()
        cache.close()
        print(f"Cleared cache: {path}")


async def _run(question: str, *, agent: bool, settings: Settings) -> None:
    if agent:
        result = await run_tool_agent(question, on_step=print, settings=settings)
        print(f"\n{RULE}\nFINAL ANSWER\n{RULE}\n{result.answer}")
        print(f"\n{RULE}\nSOURCES (checked against tool results)\n{RULE}")
        for p in result.sources:
            print(f"- {p.title} ({p.year}) · cited by {p.cited_by_count} · {p.openalex_id}")
        if not result.sources:
            print("- none: the answer cites no papers from the tools (general background only)")
        if result.unverified_ids:
            print(f"! Unverified IDs (no tool returned them): {', '.join(result.unverified_ids)}")
        return

    print()
    result = await run_paper_search(question, on_step=print, settings=settings)
    output = result.model_dump(mode="json", exclude={"papers"})
    output["paper_count"] = len(result.papers)
    print(f"\n{RULE}\nFINAL STRUCTURED RESULT\n{RULE}")
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Windows terminals can choke on emoji / non-ASCII titles; replace instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        settings = get_settings()
        if args.no_cache:
            # model_copy: same settings, one value changed (the original stays untouched)
            settings = settings.model_copy(update={"cache_enabled": False})
        if args.clear_cache:
            _clear_cache(settings)
            return 0

        mode = "Tool-calling agent" if args.agent else "Research paper search"
        print(f"{RULE}\n{mode}\n{RULE}")
        question = (
            " ".join(args.question).strip() or input("\nEnter your research question: ").strip()
        )
        if not question:
            print("No question entered. Exiting.")
            return 1

        # asyncio.run: start the event loop, run our async code, close the loop.
        try:
            asyncio.run(_run(question, agent=args.agent, settings=settings))
        finally:
            flush_tracing()  # send buffered spans before the process exits
    except OpenAlexRateLimitError as exc:
        print(f"\n{exc}")
        return 1
    except LLMOutputError as exc:
        print(f"\nThe LLM returned an unexpected format. Try again.\n{exc}")
        return 1
    except EmbeddingError as exc:
        print(f"\n{exc}\n(Daily Gemini quota? Set RERANK_ENABLED=false in .env to run without it.)")
        return 1
    except ValidationError as exc:  # e.g. GROQ_API_KEY missing from .env
        print(f"\nConfiguration error (check your .env):\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
