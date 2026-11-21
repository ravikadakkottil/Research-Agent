"""
Grounding check for the agent's free-text answer.

The pipeline's summary is validated JSON that can only cite retrieved papers.
The agent writes free Markdown, so "don't invent papers" was only a request in
the prompt, and it failed: one answer named Veličković's *Graph Attention
Networks* but gave the ID and abstract of *Heterogeneous Graph Attention
Network*.

So the contract is now:
  1. the prompt asks the agent to cite every paper as [W123] (an OpenAlex ID
     copied from a tool result)
  2. the tools record every paper they returned (AgentTools.seen)
  3. after the answer, THIS module checks every cited ID:
       returned by a tool -> a source, listed with its REAL title/year/citations
                             (a wrong label in the text becomes visible next to it)
       never returned     -> "unverified" (made up, or from memory)

What it cannot catch: papers named WITHOUT an ID. The prompt forbids that, and
the UI says when an answer cites nothing.
"""

from __future__ import annotations

import re

from research_agent.models import Paper

# W followed by digits, bare or inside an OpenAlex URL ("openalex.org/W123").
OPENALEX_ID = re.compile(r"\bW\d{4,}\b")


def cited_ids(answer: str) -> list[str]:
    """OpenAlex IDs mentioned in the answer, in order of first appearance."""
    return list(dict.fromkeys(OPENALEX_ID.findall(answer or "")))


def check_citations(answer: str, seen: dict[str, Paper]) -> tuple[list[Paper], list[str]]:
    """Split the answer's IDs into (sources the tools returned, unverified IDs)."""
    sources, unverified = [], []
    for paper_id in cited_ids(answer):
        if paper_id in seen:
            sources.append(seen[paper_id])
        else:
            unverified.append(paper_id)
    return sources, unverified
