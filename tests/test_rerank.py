"""
Semantic rerank + the Gemini embedder, with fake vectors (no real API calls).

Fake 2-D "embeddings" make the geometry easy to see: the question points
along x; on-topic papers point roughly along x; off-topic ones along y.
"""

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent import embeddings as embeddings_module
from research_agent.cache import ResponseCache
from research_agent.config import Settings
from research_agent.embeddings import (
    EmbeddingError,
    GeminiEmbedder,
    cosine_similarity,
    server_retry_delay,
)
from research_agent.models import Paper
from research_agent.retrieval.rerank import filter_by_similarity, paper_text, score_papers
from research_agent.retrieval.selection import select_balanced_papers

SETTINGS = Settings(_env_file=None, groq_api_key="x")


# --- cosine ----------------------------------------------------------------------
def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)  # length doesn't matter
    assert cosine_similarity([1, 0], [0, 3]) == pytest.approx(0.0)  # unrelated
    assert cosine_similarity([1, 1], [1, 0]) == pytest.approx(1 / math.sqrt(2))
    assert cosine_similarity([0, 0], [1, 0]) == 0.0  # no division by zero


# --- reranker ----------------------------------------------------------------------
class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    async def embed_query(self, text):
        return [1.0, 0.0]

    async def embed_documents(self, texts):
        return [self.vectors[t.split("\n")[0]] for t in texts]  # keyed by title


def test_score_and_filter_drop_off_topic_paper():
    ddpm = Paper(openalex_id="W1", title="Denoising Diffusion Probabilistic Models")
    weak_ties = Paper(openalex_id="W2", title="The Strength of Weak Ties", cited_by_count=50000)
    embedder = FakeEmbedder({ddpm.title: [0.9, 0.1], weak_ties.title: [0.1, 0.9]})

    scored = asyncio.run(
        score_papers("diffusion models", [ddpm, weak_ties], embedder, doc_chars=100)
    )
    kept = filter_by_similarity(scored, min_similarity=0.6, min_keep=1)

    assert [p.title for p in kept] == [ddpm.title]
    assert scored[1].similarity < 0.2
    assert ddpm.similarity is None  # inputs are not mutated


def test_filter_keeps_minimum_when_everything_scores_low():
    papers = [Paper(openalex_id=f"W{i}", similarity=s) for i, s in enumerate([0.3, 0.5, 0.4])]
    kept = filter_by_similarity(papers, min_similarity=0.9, min_keep=2)
    assert [p.similarity for p in kept] == [0.5, 0.4]  # best two, most similar first


def test_paper_text_uses_title_and_truncated_abstract():
    assert paper_text(Paper(title="T", abstract="abcdef"), max_chars=3) == "T\n\nabc"
    assert paper_text(Paper(title="T"), max_chars=3) == "T"  # no abstract -> title only


def test_selection_prefers_similar_papers_in_relevant_bucket():
    close = Paper(openalex_id="W1", similarity=0.9, retrieval_reasons=["relevant"])
    far = Paper(
        openalex_id="W2", similarity=0.6, cited_by_count=999, retrieval_reasons=["relevant"]
    )
    picked = select_balanced_papers([far, close], target_count=1, quotas={"relevant": 1})
    assert picked == [close]


# --- Gemini embedder (fake SDK client) ---------------------------------------------
class RateLimited(Exception):
    code = 429  # what the google-genai APIError carries


class FakeGenAI:
    """Mimics client.aio.models.embed_content. Can fail the first N calls with a 429."""

    def __init__(self, fail_first: int = 0, drop_one: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_first = fail_first
        self.drop_one = drop_one
        self.aio = SimpleNamespace(models=self)

    async def embed_content(self, *, model, contents, config):
        self.calls.append({"contents": list(contents), "task": config.task_type})
        if len(self.calls) <= self.fail_first:
            raise RateLimited()
        # vector = [len(text), 1] -> easy to check which text produced which vector
        vectors = [SimpleNamespace(values=[float(len(t)), 1.0]) for t in contents]
        return SimpleNamespace(embeddings=vectors[:-1] if self.drop_one else vectors)


def make_embedder(fake, tmp_path: Path | None = None, **overrides) -> GeminiEmbedder:
    settings = SETTINGS.model_copy(update=overrides)
    cache = ResponseCache(tmp_path / "emb.sqlite", ttl_seconds=60) if tmp_path else None
    return GeminiEmbedder(settings, cache=cache, client=fake)


def test_embedder_batches_and_caches(tmp_path: Path):
    fake = FakeGenAI()
    embedder = make_embedder(fake, tmp_path, embedding_batch_size=2)

    first = asyncio.run(embedder.embed_documents(["a", "bb", "ccc"]))
    second = asyncio.run(embedder.embed_documents(["bb", "dddd"]))  # "bb" is cached

    assert first == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert second == [[2.0, 1.0], [4.0, 1.0]]
    assert [c["contents"] for c in fake.calls] == [["a", "bb"], ["ccc"], ["dddd"]]
    assert {c["task"] for c in fake.calls} == {"RETRIEVAL_DOCUMENT"}


def test_query_and_documents_use_different_task_types():
    fake = FakeGenAI()
    asyncio.run(make_embedder(fake).embed_query("q"))
    assert fake.calls[0]["task"] == "RETRIEVAL_QUERY"


def test_embedder_retries_rate_limits(monkeypatch):
    # no real waiting in tests
    monkeypatch.setattr(embeddings_module, "wait_exponential_jitter", lambda **_: lambda _: 0)
    fake = FakeGenAI(fail_first=2)
    assert asyncio.run(make_embedder(fake).embed_documents(["a"])) == [[1.0, 1.0]]
    assert len(fake.calls) == 3


def test_server_retry_delay_is_read_from_google_error():
    class QuotaError(Exception):
        code = 429
        details = {
            "error": {
                "code": 429,
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.Help"},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
                ],
            }
        }

    assert server_retry_delay(QuotaError()) == 34.0
    assert server_retry_delay(RateLimited()) is None  # no RetryInfo -> use backoff
    assert server_retry_delay(None) is None


def test_embedder_rejects_wrong_vector_count():
    with pytest.raises(EmbeddingError, match="asked for 2"):
        asyncio.run(make_embedder(FakeGenAI(drop_one=True)).embed_documents(["a", "b"]))
