"""
Check every must-include paper in golden.yaml against OpenAlex, then write
golden.lock.json with what OpenAlex actually stores.

    python evals/verify_golden.py

Why verify: an answer key with a wrong DOI or a typo'd title would make the eval
report "missed" papers the system could never have matched. Verify the key
before trusting any score built on it.

Why a lock file: OpenAlex often stores a DIFFERENT DOI than the one we wrote
(e.g. arXiv DOI -> conference DOI) and sometimes a different title (preprint vs
journal version). The lock records every identifier for each paper, so the eval
can match any of them.

    golden.yaml       human-edited  (what we mean)
    golden.lock.json  generated     (how OpenAlex identifies it)
"""

from __future__ import annotations

import asyncio
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
import yaml

from research_agent.config import get_settings
from research_agent.retrieval.selection import normalize_title
from research_agent.sources.openalex import create_openalex_client

EVALS_DIR = Path(__file__).parent
GOLDEN = EVALS_DIR / "golden.yaml"
LOCK = EVALS_DIR / "golden.lock.json"
TITLE_MATCH_THRESHOLD = 0.9


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def bare_doi(value: str | None) -> str:
    return (value or "").lower().removeprefix("https://doi.org/")


async def verify() -> int:
    topics = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["topics"]
    problems = 0
    locked_topics: list[dict[str, Any]] = []

    async with create_openalex_client(get_settings()) as client:
        for topic in topics:
            print(f"\n[{topic['id']}] {topic['question']}")
            locked_papers = []
            for paper in topic["must_include"]:
                label = f"{paper['year']} {paper['title'][:68]}"
                try:
                    work = await client.get_work(f"https://doi.org/{paper['doi']}")
                except httpx.HTTPStatusError as exc:
                    problems += 1
                    print(f"  MISSING  {label}  (DOI {paper['doi']} -> {exc.response.status_code})")
                    continue

                found_title = work.get("display_name") or ""
                similarity = title_similarity(paper["title"], found_title)
                ok = similarity >= TITLE_MATCH_THRESHOLD
                problems += not ok
                openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
                print(
                    f"  {'OK    ' if ok else 'TITLE?'}  {label}  | {openalex_id}"
                    f" | {work.get('publication_year')} | cited {work.get('cited_by_count'):,}"
                )
                if not ok:
                    print(f"          OpenAlex title: {found_title}  (similarity {similarity:.2f})")

                locked_papers.append(
                    {
                        "title": paper["title"],
                        "kind": paper["kind"],
                        "openalex_ids": [openalex_id],
                        # every DOI and title this paper is known by
                        "dois": sorted({bare_doi(paper["doi"]), bare_doi(work.get("doi"))} - {""}),
                        "titles": sorted({paper["title"], found_title} - {""}),
                        "year": work.get("publication_year"),
                        "cited_by_count": work.get("cited_by_count"),
                    }
                )
            locked_topics.append(
                {
                    "id": topic["id"],
                    "field": topic.get("field", ""),
                    "question": topic["question"],
                    "must_include": locked_papers,
                }
            )

    if problems:
        print(f"\n{problems} problem(s) found — fix golden.yaml; lock file NOT written.")
        return 1

    LOCK.write_text(
        json.dumps({"topics": locked_topics}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",  # explicit: Windows would otherwise use its legacy codepage
    )
    print(f"\nAll entries verified. Wrote {LOCK.name} ({len(locked_topics)} topics).")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(asyncio.run(verify()))
