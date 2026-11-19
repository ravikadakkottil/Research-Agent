"""
LLM-as-judge: is each paper in the reading list relevant to the question?

Why an LLM judge? Labelling relevance by hand for hundreds of (question, paper)
pairs doesn't scale. A strong LLM is a decent, cheap grader — but NOT perfect,
so each verdict keeps a short `reason` you can spot-check in the results file.

Design choices:
  - Stronger model than the pipeline (settings.eval_judge_model).
  - Batched: one call judges all unjudged papers of a run.
  - Cached PER (question, paper): a paper judged once is never re-judged, so
    verdicts stay consistent across runs and comparisons (and cost nothing).
  - JUDGE_VERSION is part of the cache key: change the prompt -> bump it ->
    old verdicts are ignored.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from research_agent.cache import ResponseCache
from research_agent.llm import ask_json
from research_agent.models import Paper
from research_agent.retrieval.selection import paper_identity

JUDGE_VERSION = 1
ABSTRACT_CHARS = 500

JUDGE_PROMPT = """
You are an expert research librarian reviewing a reading list.

For EACH paper, decide whether it is relevant to the research question:
would a researcher studying this question want it in a 12-paper reading list?

- relevant = the paper's main subject addresses the question: methods, findings,
  reviews, or foundational work the topic directly builds on.
- not relevant = a different topic that merely shares some words, or too tangential.

Judge only from the title, year and abstract given. If the abstract is empty,
judge from the title. Be strict but fair.

Return ONLY JSON:
{"judgements": [{"index": 1, "relevant": true, "reason": "max 15 words"}]}
with exactly one entry per paper, using the given indices.
""".strip()


class Verdict(BaseModel):
    relevant: bool
    reason: str = ""


class _JudgedItem(Verdict):
    index: int


class _JudgeOutput(BaseModel):
    judgements: list[_JudgedItem]


class RelevanceJudge:
    def __init__(self, model: str, cache: ResponseCache | None) -> None:
        self.model = model
        self._cache = cache
        self.llm_calls = 0

    def _key(self, question: str, paper: Paper) -> str:
        return ResponseCache.make_key(
            "judge",
            {
                "v": JUDGE_VERSION,
                "model": self.model,
                "question": question,
                "paper": paper_identity(paper),
            },
        )

    async def judge(self, question: str, papers: list[Paper]) -> dict[str, Verdict]:
        """Return verdicts keyed by paper_identity(). Papers the LLM skipped are absent."""
        verdicts: dict[str, Verdict] = {}
        todo: list[Paper] = []
        for paper in papers:
            cached = self._cache.get(self._key(question, paper)) if self._cache else None
            if cached is not None:
                verdicts[paper_identity(paper)] = Verdict.model_validate(cached)
            else:
                todo.append(paper)

        if todo:
            listing = [
                {
                    "index": i,
                    "title": p.title,
                    "year": p.year,
                    "abstract": p.abstract[:ABSTRACT_CHARS],
                }
                for i, p in enumerate(todo, start=1)
            ]
            prompt = json.dumps(
                {"research_question": question, "papers": listing}, ensure_ascii=False
            )
            self.llm_calls += 1
            output = await ask_json(JUDGE_PROMPT, prompt, _JudgeOutput, model=self.model)

            for item in output.judgements:
                if not 1 <= item.index <= len(todo):
                    continue  # ignore invented indices
                paper = todo[item.index - 1]
                verdict = Verdict(relevant=item.relevant, reason=item.reason)
                verdicts[paper_identity(paper)] = verdict
                if self._cache:
                    self._cache.set(self._key(question, paper), verdict.model_dump())

        return verdicts
