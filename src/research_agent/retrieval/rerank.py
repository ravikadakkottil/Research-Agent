"""
Semantic rerank: keep candidates whose MEANING matches the question.

Keyword search can't tell "diffusion models" (ML) from "diffusion of
innovations" (sociology): both contain the word. With short queries
(good for recall) the citation-sorted buckets then fill up with famous papers
from other fields. The rerank fixes that after retrieval:

    question      --embed (RETRIEVAL_QUERY)-->    q
    each paper    --embed (RETRIEVAL_DOCUMENT)--> d   (title + abstract)
    similarity    = cosine(q, d)
    keep papers with similarity >= threshold (always keep at least `min_keep`)

This is the classic two-stage pattern of production search:
  stage 1  cheap, broad retrieval (keywords)  -> high recall
  stage 2  precise scoring of a small set     -> high precision
"""

from __future__ import annotations

from research_agent.embeddings import Embedder, cosine_similarity
from research_agent.models import Paper


def paper_text(paper: Paper, max_chars: int) -> str:
    """What gets embedded for a paper. Papers without an abstract use the title alone."""
    title = paper.title or ""
    abstract = (paper.abstract or "")[:max_chars]
    return f"{title}\n\n{abstract}".strip()


async def score_papers(
    question: str, papers: list[Paper], embedder: Embedder, *, doc_chars: int
) -> list[Paper]:
    """Copies of `papers` with `similarity` filled in (same order)."""
    if not papers:
        return []
    query_vector = await embedder.embed_query(question)
    doc_vectors = await embedder.embed_documents([paper_text(p, doc_chars) for p in papers])
    return [
        p.model_copy(update={"similarity": round(cosine_similarity(query_vector, v), 4)})
        for p, v in zip(papers, doc_vectors, strict=True)
    ]


def filter_by_similarity(
    papers: list[Paper], *, min_similarity: float, min_keep: int
) -> list[Paper]:
    """
    Papers at/above the threshold, most similar first.

    `min_keep` is a safety net: for an unusual question where everything scores
    low, keep the best `min_keep` anyway rather than return an empty list.
    """
    ranked = sorted(papers, key=lambda p: p.similarity or 0.0, reverse=True)
    kept = [p for p in ranked if (p.similarity or 0.0) >= min_similarity]
    return kept if len(kept) >= min_keep else ranked[:min_keep]
