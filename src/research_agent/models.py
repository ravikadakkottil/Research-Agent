"""
Data shapes used across the app, as Pydantic models.

Why models instead of plain dicts?
  - paper.title instead of paper.get("title"): typos fail loudly, editors autocomplete.
  - LLM output is VALIDATED: if the model forgets a key or returns the wrong
    type, you get a clear error instead of a KeyError three functions later.
  - model_dump() / model_dump_json() convert back to dict / JSON for prompts and output.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# The three retrieval perspectives. Literal = only these exact strings are allowed.
RetrievalReason = Literal["relevant", "recent", "highly_cited"]

# Extra phrasings searched per question (each costs one OpenAlex search).
MAX_ALTERNATIVE_QUERIES = 2


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
class Paper(BaseModel):
    """One scholarly work, simplified from OpenAlex's large payload."""

    openalex_id: str | None = None
    title: str | None = None
    year: int | None = None
    cited_by_count: int = 0
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    doi: str | None = None
    abstract: str = ""
    # Which bucket(s) found this paper. Filled during deduplication.
    retrieval_reasons: list[RetrievalReason] = Field(default_factory=list)
    # Meaning-match with the question (cosine, ~0..1). Filled by the semantic rerank.
    similarity: float | None = None


# ---------------------------------------------------------------------------
# LLM #1 output: the search plan
# ---------------------------------------------------------------------------
class QueryPlan(BaseModel):
    """What the planning LLM returns for a user question."""

    # Three query shapes, because OpenAlex search requires EVERY word to match:
    #   search_query        3-6 keywords, full-text RELEVANCE search (forgiving: long
    #                       papers contain most words somewhere)
    #   core_query          2-4 words naming the topic, for the STRICT title+abstract
    #                       searches; one padding word ("stability", "deep learning")
    #                       would knock out a landmark paper that doesn't use it
    #   alternative_queries other names the field uses ("fake news" for
    #                       "misinformation"), so landmarks written in different
    #                       vocabulary are still found
    search_query: str = Field(min_length=1, description="3-6 academic keywords")
    core_query: str = ""  # empty -> falls back to search_query
    alternative_queries: list[str] = Field(default_factory=list)
    reason: str = ""

    # Option 1 — the USER stated a time range ("since 2020", "last 5 years").
    # user_time_phrase must be copied word-for-word from the question; Python
    # checks that, so the LLM cannot invent a time range the user never asked for.
    user_time_phrase: str | None = None
    user_from_year: int | None = None

    # Option 2 — the LLM's judgement of how fast this field moves.
    suggested_recent_years: int | None = None
    time_reason: str = ""

    @field_validator("core_query", "search_query")
    @classmethod
    def _strip(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("alternative_queries")
    @classmethod
    def _clean_alternatives(cls, values: list[str]) -> list[str]:
        """Strip, drop blanks/duplicates, cap the count (LLMs over-deliver; don't fail)."""
        seen: set[str] = set()
        cleaned = []
        for value in values:
            value = " ".join(value.split())
            if value and value.lower() not in seen:
                seen.add(value.lower())
                cleaned.append(value)
        return cleaned[:MAX_ALTERNATIVE_QUERIES]

    @property
    def strict_query(self) -> str:
        """Query for the title+abstract buckets."""
        return self.core_query or self.search_query

    @property
    def alternatives(self) -> list[str]:
        """
        Alternative phrasings worth an extra search. Dropped:
          - the core query again ("retrieval-augmented generation" vs "... augmented ...")
          - a SUPERSET of the core words ("CRISPR Cas9 system" vs "CRISPR Cas9"):
            every word must match, so it can only return a subset of the core's results
          - a lone short abbreviation ("ICI"): too ambiguous across fields
        """
        core_words = _words(self.strict_query)
        kept = []
        for query in self.alternative_queries:
            words = _words(query)
            if not words or words >= core_words:
                continue
            if len(words) == 1 and len(next(iter(words))) <= 3:
                continue
            kept.append(query)
        return kept

    def describe(self) -> str:
        """One-line summary for logs and eval reports."""
        parts = [f"relevance: {self.search_query}", f"core: {self.strict_query}"]
        if self.alternatives:
            parts.append("alt: " + " / ".join(self.alternatives))
        return " | ".join(parts)


def _words(text: str) -> set[str]:
    """Lowercase word set; hyphens and punctuation split words."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class RecentWindow(BaseModel):
    """The final decision on what counts as 'recent' for this question."""

    from_year: int
    source: Literal["user", "field_pace", "default"]
    explanation: str


# ---------------------------------------------------------------------------
# LLM #2 output: the research synthesis
# ---------------------------------------------------------------------------
class PaperUsed(BaseModel):
    title: str
    year: int | None = None
    why_relevant: str = ""


class ResearchSummary(BaseModel):
    answer: str
    foundational_findings: list[str] = Field(default_factory=list)
    recent_developments: list[str] = Field(default_factory=list)
    evolution_of_the_field: str = ""
    papers_used: list[PaperUsed] = Field(default_factory=list)
    gaps_or_cautions: str = ""


# ---------------------------------------------------------------------------
# Retrieval stage result (steps 1-4, before synthesis)
# ---------------------------------------------------------------------------
class ReadingList(BaseModel):
    query_plan: QueryPlan
    recent_window: RecentWindow
    bucket_counts: dict[str, int]
    candidates: list[Paper]  # every unique paper retrieved (with similarity if reranked)
    # Candidates that passed the semantic rerank; None = rerank skipped.
    shortlist: list[Paper] | None = None
    papers: list[Paper]  # the balanced final selection

    @property
    def selection_pool(self) -> list[Paper]:
        """What selection chose from: the shortlist if reranked, else all candidates."""
        return self.shortlist if self.shortlist is not None else self.candidates


# ---------------------------------------------------------------------------
# Whole pipeline result
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    user_question: str
    query_plan: QueryPlan
    recent_window: RecentWindow
    bucket_counts: dict[str, int]
    unique_candidates: int
    papers: list[Paper]
    summary: ResearchSummary
