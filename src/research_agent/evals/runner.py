"""
Run the eval: every golden topic x N runs -> metrics -> aggregated report.

Why N runs? The LLM writes a different search query each time, so scores
wobble even with identical code. Running each topic several times shows that
natural variance ("spread"). An improvement smaller than the spread is noise.

Aggregation:
  per-topic   mean over that topic's runs
  overall     mean over topics (macro average: every topic counts equally)
  spread      std-dev of the overall score across run #1, run #2, ... (needs runs >= 2)
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

from pydantic import BaseModel

from research_agent import __version__
from research_agent.config import Settings
from research_agent.evals.golden import GoldenTopic
from research_agent.evals.judge import JUDGE_VERSION, RelevanceJudge
from research_agent.evals.metrics import RunMetrics, evaluate_run
from research_agent.pipeline import retrieve_reading_list
from research_agent.sources.openalex import OpenAlexClient, OpenAlexRateLimitError
from research_agent.synthesis import summarize_papers

Mode = Literal["retrieval", "full"]

METRICS = [
    "recall",
    "candidate_recall",
    "shortlist_recall",
    "recall_foundational",
    "recall_recent",
    "precision",
    "recent_precision",
    "expansion_precision",
    "grounding",
]


class EvalReport(BaseModel):
    meta: dict[str, Any]
    overall: dict[str, dict[str, float | None]]  # metric -> {"mean": .., "spread": ..}
    per_topic: dict[str, dict[str, float | None]]  # topic -> metric -> mean
    runs: list[RunMetrics]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
async def run_one(
    topic: GoldenTopic,
    run: int,
    *,
    mode: Mode,
    settings: Settings,
    client: OpenAlexClient,
    judge: RelevanceJudge,
) -> RunMetrics:
    started = time.perf_counter()
    requests_before = client.network_requests

    reading = await retrieve_reading_list(topic.question, settings=settings, client=client)
    summary = None
    if mode == "full":
        summary = await summarize_papers(
            topic.question,
            reading.papers,
            recent_window=reading.recent_window,
            abstract_chars=settings.abstract_chars_for_llm,
        )
    latency = time.perf_counter() - started  # judge time is NOT pipeline latency

    verdicts = await judge.judge(topic.question, reading.papers)
    return evaluate_run(
        topic,
        run,
        reading,
        verdicts,
        current_year=date.today().year,
        summary=summary,
        latency_s=latency,
        openalex_requests=client.network_requests - requests_before,
    )


async def run_eval(
    topics: list[GoldenTopic],
    *,
    runs: int,
    mode: Mode,
    settings: Settings,
    client: OpenAlexClient,
    judge: RelevanceJudge,
    on_progress: Callable[[str], None] = lambda _: None,
) -> list[RunMetrics]:
    """Topics run one after another (gentle on free-tier rate limits)."""
    results: list[RunMetrics] = []
    total = len(topics) * runs
    for run in range(1, runs + 1):
        for topic in topics:
            label = f"[{len(results) + 1}/{total}] run {run} · {topic.id}"
            try:
                metrics = await run_one(
                    topic, run, mode=mode, settings=settings, client=client, judge=judge
                )
                on_progress(
                    f"{label:<40} recall={_fmt(metrics.recall)} "
                    f"precision={_fmt(metrics.precision)} ({metrics.latency_s:.1f}s)"
                )
            except OpenAlexRateLimitError:
                raise  # budget gone: no point continuing
            except Exception as exc:  # noqa: BLE001 - one failed run shouldn't kill the eval
                metrics = RunMetrics(
                    topic_id=topic.id, run=run, error=f"{type(exc).__name__}: {exc}"
                )
                on_progress(f"{label:<40} FAILED: {metrics.error[:120]}")
            results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def aggregate(results: list[RunMetrics]) -> tuple[dict, dict]:
    ok = [r for r in results if r.error is None]
    topic_ids = list(dict.fromkeys(r.topic_id for r in results))
    run_ids = sorted({r.run for r in ok})

    per_topic: dict[str, dict[str, float | None]] = {}
    for tid in topic_ids:
        mine = [r for r in ok if r.topic_id == tid]
        per_topic[tid] = {m: _mean([getattr(r, m) for r in mine]) for m in METRICS}

    overall: dict[str, dict[str, float | None]] = {}
    for m in METRICS:
        macro = _mean([per_topic[t][m] for t in topic_ids])
        # overall score of each run index (run 1, run 2, ...), then their spread
        by_run = [_mean([getattr(r, m) for r in ok if r.run == run]) for run in run_ids]
        by_run = [v for v in by_run if v is not None]
        overall[m] = {"mean": macro, "spread": pstdev(by_run) if len(by_run) >= 2 else None}
    return overall, per_topic


def build_report(
    results: list[RunMetrics],
    *,
    label: str,
    mode: Mode,
    runs: int,
    settings: Settings,
    judge: RelevanceJudge,
    use_cache: bool,
) -> EvalReport:
    overall, per_topic = aggregate(results)
    ok = [r for r in results if r.error is None]
    meta = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "git": _git_state(),
        "mode": mode,
        "runs": runs,
        "topics": len(per_topic),
        "failed_runs": len(results) - len(ok),
        "pipeline_cache": use_cache,
        "reranked": any(r.shortlist_recall is not None for r in ok),
        "mean_latency_s": round(mean(r.latency_s for r in ok), 2) if ok else None,
        "openalex_requests": sum(r.openalex_requests for r in results),
        "judge": {"model": judge.model, "version": JUDGE_VERSION, "llm_calls": judge.llm_calls},
        "settings": settings.model_dump(
            include={
                "llm_model",
                "relevant_count",
                "recent_count",
                "highly_cited_count",
                "default_recent_years",
                "target_final_papers",
                "select_relevant",
                "select_recent",
                "select_highly_cited",
                "rerank_enabled",
                "embedding_model",
                "embedding_dimensions",
                "rerank_min_similarity",
                "citation_expansion_enabled",
                "expansion_seed_count",
                "expansion_min_citers",
                "expansion_max_papers",
                "expansion_min_similarity",
                "select_cited_by_top",
            }
        ),
    }
    return EvalReport(meta=meta, overall=overall, per_topic=per_topic, runs=results)


def save_report(report: EvalReport, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = results_dir / f"{stamp}_{report.meta['label']}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def format_report(report: EvalReport) -> str:
    cols = ["recall", "candidate_recall", "shortlist_recall", "precision", "recent_precision"]
    heads = ["recall", "cand.rec", "short.rec", "precision", "recent.prec"]
    lines = [f"{'topic':<26}" + "".join(f"{h:>12}" for h in heads)]
    lines.append("-" * len(lines[0]))
    for tid, metrics in report.per_topic.items():
        lines.append(f"{tid:<26}" + "".join(f"{_fmt(metrics[c]):>12}" for c in cols))
    lines.append("-" * len(lines[0]))
    lines.append(
        f"{'OVERALL (mean)':<26}" + "".join(f"{_fmt(report.overall[c]['mean']):>12}" for c in cols)
    )
    lines.append(
        f"{'  ± spread across runs':<26}"
        + "".join(f"{_fmt(report.overall[c]['spread']):>12}" for c in cols)
    )
    extra = report.overall
    lines.append("")
    lines.append(
        f"recall foundational={_fmt(extra['recall_foundational']['mean'])}  "
        f"recall recent={_fmt(extra['recall_recent']['mean'])}  "
        f"expansion precision={_fmt(extra['expansion_precision']['mean'])}  "
        f"grounding={_fmt(extra['grounding']['mean'])}  "
        f"failed runs={report.meta['failed_runs']}  "
        f"mean latency={report.meta['mean_latency_s']}s  "
        f"OpenAlex requests={report.meta['openalex_requests']}"
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _git_state() -> dict[str, Any]:
    """Record which code produced these scores (so results are comparable later)."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    return {
        "commit": git("rev-parse", "--short", "HEAD") or "no-commits-yet",
        "uncommitted_changes": bool(git("status", "--porcelain")),
    }
