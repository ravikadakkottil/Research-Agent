"""
Run the retrieval eval against the golden dataset.

    python evals/run_eval.py                         # 13 topics x 3 runs, retrieval mode
    python evals/run_eval.py --label baseline
    python evals/run_eval.py --mode full --runs 1    # + summary & grounding (slow on free tier)
    python evals/run_eval.py --topics rag,crispr --runs 1   # quick check

Results: printed table + evals/results/<timestamp>_<label>.json
Compare two results: python evals/compare.py OLD.json NEW.json

Modes:
  retrieval  plan -> search -> select, then judge. Cheap: scores paper selection.
  full       + LLM summary and the grounding check. ~5k extra tokens per run;
             Groq free tier allows ~8k tokens/min per model, so use --runs 1.

Caches: the pipeline runs UNCACHED by default (honest run-to-run variance).
The judge's verdicts ARE cached per (question, paper) so they stay consistent.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Free-tier rate limits: let the Groq SDK wait-and-retry more patiently during evals.
os.environ.setdefault("LLM_MAX_RETRIES", "8")

from research_agent.cache import ResponseCache  # noqa: E402 - must follow the env tweak
from research_agent.config import get_settings  # noqa: E402
from research_agent.evals.golden import load_golden  # noqa: E402
from research_agent.evals.judge import RelevanceJudge  # noqa: E402
from research_agent.evals.runner import (  # noqa: E402
    build_report,
    format_report,
    run_eval,
    save_report,
)
from research_agent.sources.openalex import (  # noqa: E402
    OpenAlexRateLimitError,
    create_openalex_client,
)

EVALS_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", type=int, default=3, help="runs per topic (default 3)")
    p.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    p.add_argument("--topics", help="comma-separated topic ids (default: all)")
    p.add_argument("--label", default="eval", help="name for the results file")
    p.add_argument("--cache", action="store_true", help="use pipeline caches (less variance)")
    p.add_argument("--fresh-judge", action="store_true", help="ignore cached judge verdicts")
    p.add_argument("--golden", type=Path, default=EVALS_DIR / "golden.lock.json")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings().model_copy(update={"cache_enabled": args.cache})

    topics = load_golden(args.golden)
    if args.topics:
        wanted = set(args.topics.split(","))
        topics = [t for t in topics if t.id in wanted]
        if unknown := wanted - {t.id for t in topics}:
            print(f"Unknown topic ids: {', '.join(sorted(unknown))}")
            return 1

    judge_cache = (
        None
        if args.fresh_judge
        else ResponseCache(settings.cache_dir / "judge.sqlite", ttl_seconds=365 * 24 * 3600)
    )
    judge = RelevanceJudge(settings.eval_judge_model, judge_cache)

    print(
        f"Eval '{args.label}': {len(topics)} topics x {args.runs} runs, mode={args.mode}, "
        f"pipeline cache={'on' if args.cache else 'off'}, judge={judge.model}\n"
    )
    try:
        async with create_openalex_client(settings) as client:
            results = await run_eval(
                topics,
                runs=args.runs,
                mode=args.mode,
                settings=settings,
                client=client,
                judge=judge,
                on_progress=print,
            )
    except OpenAlexRateLimitError as exc:
        print(f"\nStopped: {exc}")
        return 1
    finally:
        if judge_cache:
            judge_cache.close()

    report = build_report(
        results,
        label=args.label,
        mode=args.mode,
        runs=args.runs,
        settings=settings,
        judge=judge,
        use_cache=args.cache,
    )
    path = save_report(report, EVALS_DIR / "results")
    print("\n" + format_report(report))
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(asyncio.run(main()))
