"""
Text embeddings: turn text into a vector of numbers that captures its MEANING.

Two texts about the same idea get vectors pointing in a similar direction,
even with no words in common ("diffusion models" ~ "denoising score matching"),
and texts that share words but not meaning point elsewhere ("diffusion of
innovations"). Cosine similarity measures that direction: 1 = same meaning.

Task types (Gemini): the question is embedded as RETRIEVAL_QUERY and papers as
RETRIEVAL_DOCUMENT. The model is trained so that a query lands close to the
documents that ANSWER it, not just to texts that look like it.

`Embedder` is a Protocol (an interface): the reranker only needs
"something with embed_query / embed_documents". Tests pass a fake one, and a
different provider (OpenAI, a local model) could be swapped in later.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from research_agent.cache import ResponseCache
from research_agent.config import Settings

Vector = list[float]

QUERY_TASK = "RETRIEVAL_QUERY"
DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"

# Gemini errors worth retrying: rate limit (429) and temporary server problems.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class Embedder(Protocol):
    async def embed_query(self, text: str) -> Vector: ...

    async def embed_documents(self, texts: list[str]) -> list[Vector]: ...


class EmbeddingError(RuntimeError):
    """The embedding API failed after retries (or returned the wrong number of vectors)."""


def cosine_similarity(a: Vector, b: Vector) -> float:
    """
    cos(angle between a and b) = (a . b) / (|a| * |b|).

    Gemini only normalizes full-size (3072-dim) vectors, so with 768 dims we
    divide by the lengths ourselves.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


def _is_retryable(exc: BaseException) -> bool:
    return getattr(exc, "code", None) in RETRYABLE_STATUS


# Longest server-requested wait we'll honour (free tier: per-MINUTE limits).
MAX_SERVER_WAIT_SECONDS = 70.0


def server_retry_delay(exc: BaseException | None) -> float | None:
    """
    Seconds the server asked us to wait, from a Google error's RetryInfo
    ({"@type": ".../google.rpc.RetryInfo", "retryDelay": "34s"}), else None.

    Same idea as HTTP's Retry-After: when the server says how long to wait,
    obey it instead of guessing with exponential backoff.
    """
    details = getattr(exc, "details", None)
    error = details.get("error", {}) if isinstance(details, dict) else {}
    for item in error.get("details", []) if isinstance(error, dict) else []:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


class GeminiEmbedder:
    """
    gemini-embedding-001 via the google-genai SDK, with a per-text cache.

    Only texts NOT in the cache are sent, in batches. A paper seen in an earlier
    run (or by another bucket) is never embedded twice.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        cache: ResponseCache | None = None,
        client: Any = None,  # injectable for tests; default = real Gemini client
    ) -> None:
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        self.max_attempts = settings.http_max_retries + 2  # free tier: 429s are common
        self.cache = cache
        self.api_calls = 0
        if client is None:
            from google import genai  # imported lazily: only needed for real calls

            if settings.gemini_api_key is None:
                raise EmbeddingError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self._client = client

    async def __aenter__(self) -> GeminiEmbedder:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        aio = getattr(self._client, "aio", None)
        if aio is not None and hasattr(aio, "aclose"):
            await aio.aclose()
        if self.cache:
            self.cache.close()

    # --- public -------------------------------------------------------------------
    async def embed_query(self, text: str) -> Vector:
        return (await self._embed([text], QUERY_TASK))[0]

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return await self._embed(texts, DOCUMENT_TASK)

    # --- internals ---------------------------------------------------------------
    def _key(self, text: str, task: str) -> str:
        return ResponseCache.make_key(
            "embedding",
            {"model": self.model, "dims": self.dimensions, "task": task, "text": text},
        )

    async def _embed(self, texts: list[str], task: str) -> list[Vector]:
        vectors: list[Vector | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            hit = self.cache.get(self._key(text, task)) if self.cache else None
            if hit is not None:
                vectors[i] = hit
            else:
                missing.append(i)

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            fresh = await self._call_api([texts[i] for i in batch], task)
            for i, vector in zip(batch, fresh, strict=True):
                vectors[i] = vector
                if self.cache:
                    self.cache.set(self._key(texts[i], task), vector)

        return [v for v in vectors if v is not None]

    async def _call_api(self, texts: list[str], task: str) -> list[Vector]:
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dimensions)
        backoff = wait_exponential_jitter(initial=2, max=30)

        def wait(retry_state) -> float:
            # Free tier: 100 texts/minute (each text in a batch counts). A 429 says
            # exactly how long until the minute resets, so wait that long.
            delay = server_retry_delay(retry_state.outcome.exception())
            if delay is not None:
                return min(delay + 1.0, MAX_SERVER_WAIT_SECONDS)
            return backoff(retry_state)

        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait,
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    self.api_calls += 1
                    response = await self._client.aio.models.embed_content(
                        model=self.model, contents=texts, config=config
                    )
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding failed: {exc}") from exc

        vectors = [list(e.values or []) for e in (response.embeddings or [])]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors


def create_embedder(settings: Settings) -> GeminiEmbedder | None:
    """The configured embedder, or None when rerank is off or no key is set."""
    key = settings.gemini_api_key
    if not settings.rerank_enabled or key is None or not key.get_secret_value().strip():
        return None
    cache = None
    if settings.embedding_cache_enabled:
        cache = ResponseCache(
            settings.cache_dir / "embeddings.sqlite",
            ttl_seconds=settings.embedding_cache_days * 24 * 3600,
        )
    return GeminiEmbedder(settings, cache=cache)
